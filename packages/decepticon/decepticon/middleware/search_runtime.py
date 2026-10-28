from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Annotated, NotRequired
from uuid import uuid4

from anyio import to_thread
from deepagents.backends.protocol import BackendProtocol
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    OmitFromInput,
    ToolCallRequest,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import JsonValue, ValidationError

from decepticon.artifacts import ActorContext, ArtifactError, ArtifactRegistry, MaterializationError
from decepticon.middleware.artifact_tools import (
    build_artifact_tools,
    persist_artifacts,
    registry_from_state,
)
from decepticon.middleware.search_evidence import (
    SpecialistHandoff,
    import_specialist_edits,
    scoped_backend,
)
from decepticon.middleware.search_policy import (
    SearchContractError,
    SearchSession,
    current_events,
    merge_search_events,
)
from decepticon.middleware.search_tools import build_search_tools
from decepticon.workflows import WorkflowCatalog, WorkflowSnapshot
from decepticon_core.contracts.slots import SLOTS_PER_ROLE
from decepticon_core.registry.roles import RoleRegistry

_MUTATIONS = frozenset(
    {
        "create_artifact",
        "edit_artifact",
        "commit_artifact",
        "publish_artifacts",
        "review_search",
        "finalize_search",
        "commit_opplan",
        "update_objective",
        "record_plan_fact",
        "revoke_plan_fact",
        "load_opplan",
        "bind_workflow_task",
    }
)
_DOCUMENT_WRITERS = frozenset(
    {
        "verifier",
        "finding_verifier",
        "validator",
        "bugclaw_validator",
        "report",
        "finding_reporter",
        "report_drafter",
    }
)


def merge_recovery_events(
    left: dict[str, bool] | None, right: dict[str, bool] | None
) -> dict[str, bool]:
    """Merge per-task recovery receipts without merging registry snapshots."""
    return {**(left or {}), **(right or {})}


_POLICY = """<artifact-search-v2>
The runtime owns document UUIDs, filenames, numbers and committed revisions.
Use create_artifact/edit_artifact for working drafts, commit_artifact at a plan,
experiment, verification or publication milestone; inspect_artifacts to reuse work.
HYP_01_title.md hypotheses, AP_01_title.md candidate paths and PATH_01_title.md
executed paths have UUID links. Preserve supported/refuted/unresolved hypotheses.
Informational observations, negatives, blocked and untested coverage belong to
info_entry records in one information log; only vulnerabilities get finding files.
The orchestrator is the single registry writer. Specialists edit their assigned
committed finding in place; runtime imports verifier/reporter edits with their role.
Never overwrite raw evidence: capture a new file for each independent experiment.
After every specialist return inspect recon, prior experiments, HYP/AP/PATH and
current reviewed research facts. Reuse capabilities and negative results, propose
discriminating alternatives, commit OPPLAN changes and call review_search with the
current revision and next ready objective. Declaring a task no-finding is not the
whole search goal. Continue until independently verified High/Critical and required
coverage/mission completion, or a supported budget/scope/infrastructure/search stop.
Do not repeat identical failed payloads as new hypotheses or inflate severity.
Publish one assessment_report at report/security-assessment-report.md with summary,
scope/limits, vulnerability links, observed paths, Info/coverage and remediation.
Reconcile OPPLAN and review_search before finalize_search. File existence alone
never closes the run. All target activity remains bounded by the approved RoE.
</artifact-search-v2>"""


class SearchRuntimeState(AgentState):
    fixed_submission_objective: NotRequired[Annotated[str, OmitFromInput]]
    artifact_registry: NotRequired[Annotated[str, OmitFromInput]]
    search_state: NotRequired[Annotated[str, OmitFromInput]]
    search_receipt: NotRequired[Annotated[str, OmitFromInput]]
    search_last_review: NotRequired[Annotated[str, OmitFromInput]]
    search_task_events: Annotated[NotRequired[dict[str, str]], OmitFromInput, merge_search_events]
    search_recovery_events: Annotated[
        NotRequired[dict[str, bool]], OmitFromInput, merge_recovery_events
    ]


def _document_error(messages: list[BaseMessage], call_id: str, detail: str) -> list[BaseMessage]:
    own = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id == call_id
    ]
    retained = [message for message in messages if message not in own]
    return [
        *retained,
        ToolMessage(
            content="\n".join([*(message.text for message in own), detail]),
            status="error",
            tool_call_id=call_id,
        ),
    ]


class SearchRuntimeMiddleware(AgentMiddleware):
    state_schema = SearchRuntimeState

    def __init__(self, backend: BackendProtocol) -> None:
        super().__init__()
        self.backend = backend
        self._materialize_lock = Lock()
        self.tools = [*build_artifact_tools(backend), *build_search_tools(backend)]

    def _materialize(self, registry: ArtifactRegistry, workspace: str | None) -> ArtifactRegistry:
        with self._materialize_lock:
            return persist_artifacts(registry, self.backend, workspace)

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, object]:
        configurable = get_config().get("configurable", {})
        policy_version = configurable.get("search_policy_version")
        if policy_version not in {None, 2}:
            return {}
        if policy_version is None:
            pin = state.get("workflow_snapshot")
            if pin:
                workflow = WorkflowSnapshot.model_validate_json(pin)
            elif state.get("objectives"):
                return {}
            else:
                workflow = WorkflowCatalog.default().resolve(
                    configurable.get("workflow_id", "red-team"),
                    configurable.get("workflow_version"),
                )
            if workflow.document.metadata.search_goal != "verified-high":
                return {}
        previous_state = state.get("search_state")
        previous = SearchSession.model_validate_json(previous_state) if previous_state else None
        run_id = configurable.get("search_run_id") or (
            previous.run_id if previous and previous.disposition is None else str(uuid4())
        )
        if previous and previous.run_id == run_id:
            return {}
        session = SearchSession(
            thread_id=str(configurable.get("thread_id", "")),
            run_id=str(run_id),
            started_at=datetime.now(UTC).isoformat(),
            max_dispatches=configurable.get("search_max_dispatches"),
        )
        return {"search_state": session.model_dump_json(), "search_receipt": ""}

    def _enabled(self, state: AgentState) -> bool:
        return bool(state.get("search_state")) and get_config().get("configurable", {}).get(
            "search_policy_version"
        ) in {None, 2}

    def before_model(self, state, runtime):
        # Registration is checkpointed before filesystem publication. A process
        # crash/retry therefore reuses the same UUIDs and immutable revisions.
        if not state.get("fixed_submission_objective"):
            return None
        registry = registry_from_state(state.get("artifact_registry"))
        if registry.materialized:
            return None
        published = self._materialize(registry, state.get("workspace_path"))
        return {"artifact_registry": published.model_dump_json()}

    async def abefore_model(self, state, runtime):
        return await to_thread.run_sync(self.before_model, state, runtime)

    def _request(self, request: ModelRequest) -> ModelRequest:
        if not self._enabled(request.state):
            names = {tool.name for tool in self.tools}
            return request.override(
                tools=[
                    tool
                    for tool in request.tools
                    if (tool.name if isinstance(tool, BaseTool) else tool.get("name")) not in names
                ]
            )
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append(
            {
                "type": "text",
                "text": _POLICY + "\nCurrent search: " + request.state.get("search_state", ""),
            }
        )
        return request.override(system_message=SystemMessage(content_blocks=blocks))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelCallResult]
    ) -> ModelCallResult:
        return handler(self._request(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelCallResult]]
    ) -> ModelCallResult:
        return await handler(self._request(request))

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, object] | None:
        if not self._enabled(state):
            return None
        latest = next(
            (message for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            None,
        )
        if latest is None or latest.tool_calls:
            return None
        session = SearchSession.model_validate_json(state.get("search_state", ""))
        if session.disposition is not None and state.get("search_receipt"):
            return None
        if session.final_attempts >= 3:
            raise SearchContractError(
                "Repeated premature closure: OPPLAN search remains unfinished; preserve checkpoint for resumption"
            )
        session = session.model_copy(update={"final_attempts": session.final_attempts + 1})
        return {
            "search_state": session.model_dump_json(),
            "jump_to": "model",
            "messages": [
                HumanMessage(
                    content="The search has no validated terminal receipt. Inspect prior recon, hypothesis/path UUIDs and new results, replan through OPPLAN, then review_search. Continue useful discriminating experiments; finalize_search only when its evidence and stopping conditions hold.",
                    additional_kwargs={"decepticon_internal": True},
                )
            ],
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, object] | None:
        return self.after_model(state, runtime)

    def _guard(self, request: ToolCallRequest) -> ToolMessage | None:
        calls = next(
            (
                message.tool_calls
                for message in reversed(request.state["messages"])
                if isinstance(message, AIMessage)
            ),
            [],
        )
        mutations = [call for call in calls if call["name"] in _MUTATIONS]
        if mutations and (len(mutations) > 1 or any(call["name"] == "task" for call in calls)):
            return ToolMessage(
                content="Serialize document/search mutations separately from task dispatch.",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        if request.tool_call["name"] == "task":
            session = SearchSession.model_validate_json(request.state["search_state"])
            recovery = request.state.get("search_recovery_events", {})
            if any(
                key.startswith(session.run_id + ":") and not resolved
                for key, resolved in recovery.items()
            ):
                return ToolMessage(
                    content="Document restoration pending; retry publish_artifacts before task dispatch.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            registry_raw = request.state.get("artifact_registry")
            if registry_raw:
                registry = registry_from_state(registry_raw)
                if registry.artifacts and not registry.materialized:
                    return ToolMessage(
                        content="Document publication pending; retry publish_artifacts before task dispatch.",
                        tool_call_id=request.tool_call["id"],
                        status="error",
                    )
            events = current_events(session, request.state.get("search_task_events", {}))
            if set(events) - set(session.reviewed_task_ids):
                return ToolMessage(
                    content="Review returned results and revise OPPLAN through review_search before the next task.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            count = sum(call["name"] == "task" for call in calls)
            if count > 1 and request.state.get("fixed_investigation_id") and any(
                call["name"] == "task"
                and call["args"].get("task_id") == request.state["fixed_investigation_id"]
                for call in calls
            ):
                return ToolMessage(
                    content="Serialize the fixed investigator return because it registers shared documents.",
                    tool_call_id=request.tool_call["id"], status="error",
                )
            if count > 1 and any(
                call["name"] == "task" and call["args"].get("subagent_type") in _DOCUMENT_WRITERS
                for call in calls
            ):
                return ToolMessage(
                    content="Serialize verifier/reporter tasks because they commit shared document revisions.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            if session.disposition or (
                session.max_dispatches is not None and len(events) + count > session.max_dispatches
            ):
                return ToolMessage(
                    content="Search is terminal or its runtime dispatch budget is exhausted.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        return None

    def _returned(
        self, request: ToolCallRequest, result: ToolMessage | Command[str]
    ) -> ToolMessage | Command[str]:
        if request.tool_call["name"] == "publish_artifacts":
            if isinstance(result, Command) and result.update:
                update = dict(result.update)
                raw = update.get("artifact_registry")
                messages = update.get("messages", [])
                succeeded = bool(messages) and all(
                    isinstance(message, ToolMessage) and message.status == "success"
                    for message in messages
                )
                if succeeded and raw and registry_from_state(raw).materialized:
                    cleared = {
                        key: True
                        for key, resolved in request.state.get("search_recovery_events", {}).items()
                        if not resolved
                    }
                    if cleared:
                        return Command(update={**update, "search_recovery_events": cleared})
            return result
        if request.tool_call["name"] != "task":
            return result
        session = SearchSession.model_validate_json(request.state["search_state"])
        call_id = request.tool_call["id"]
        if not call_id:
            raise SearchContractError("Task return requires a tool call identity")
        args = request.tool_call["args"]
        update = dict(result.update) if isinstance(result, Command) and result.update else {}
        update.pop("artifact_registry", None)
        if isinstance(result, ToolMessage):
            update["messages"] = [result]
        failed = any(
            isinstance(message, ToolMessage) and message.status == "error"
            for message in update.get("messages", [])
        )
        specialist_role = str(args.get("subagent_type", ""))
        fixed_investigator = (
            os.environ.get("DECEPTICON_MEGATRON_RUNTIME") == "1"
            and specialist_role == "exploit"
            and bool(request.state.get("fixed_investigation_id"))
            and args.get("task_id") == request.state.get("fixed_investigation_id")
        )
        managed_reporter = (
            os.environ.get("DECEPTICON_MEGATRON_RUNTIME") == "1"
            and bool(request.state.get("fixed_submission_objective"))
            and specialist_role == "finding_reporter"
            and args.get("task_id") in (request.state.get("fixed_handoff_objective_ids") or [])
            and any(row.get("id") == args.get("task_id") and row.get("workflow_stage") == "report"
                    for row in request.state.get("objectives", []))
        )
        role = "" if failed else specialist_role
        registry = registry_from_state(request.state.get("artifact_registry"))
        try:
            if failed:
                # Restore canonical files even if a failed non-writer edited
                # them. Only document writers update the registry state;
                # non-writer tasks may fail in parallel.
                restored = (
                    self._materialize(registry, request.state.get("workspace_path"))
                    if registry.artifacts
                    else registry
                )
                if specialist_role in _DOCUMENT_WRITERS:
                    update["artifact_registry"] = restored.model_dump_json()
                update.pop("search_state", None)
                update.pop("search_receipt", None)
                update.pop("search_last_review", None)
                return Command(
                    update={
                        **update,
                        "search_task_events": {
                            session.run_id + ":" + call_id: str(args.get("task_id", ""))
                        },
                    }
                )
            if fixed_investigator:
                from decepticon.middleware.investigation_submission import register_submission
                imported = register_submission(registry,
                    scoped_backend(self.backend, request.state.get("workspace_path")),
                    args["task_id"])
            else:
                imported = registry
            imported = import_specialist_edits(
                imported,
                scoped_backend(self.backend, request.state.get("workspace_path")),
                SpecialistHandoff(
                    context=ActorContext(
                        actor=session.run_id + ":" + call_id,
                        role=role,
                        objective_id=args.get("task_id"),
                    ),
                    assignment=json.dumps(args),
                    scope_digest=get_config().get("configurable", {}).get("research_scope_digest"),
                ),
            )
            if managed_reporter:
                from decepticon.middleware.assessment import assemble_assessment
                imported = assemble_assessment(imported,
                    request.state.get("fixed_handoff_bindings") or {},
                    scoped_backend(self.backend, request.state.get("workspace_path")))
            if imported != registry:
                update["artifact_registry"] = imported.model_dump_json()
                if fixed_investigator or managed_reporter:
                    if fixed_investigator:
                        update["fixed_submission_objective"] = args["task_id"]
                else:
                    imported = self._materialize(imported, request.state.get("workspace_path"))
                    update["artifact_registry"] = imported.model_dump_json()
        except MaterializationError as error:
            # Keep a successfully imported specialist revision for retry.
            # Failed-return restoration has no imported state and falls back
            # to the original checkpoint registry instead.
            if "artifact_registry" not in update:
                if specialist_role in _DOCUMENT_WRITERS:
                    pending = registry.model_copy(update={"materialized": False})
                    update["artifact_registry"] = pending.model_dump_json()
            update["search_recovery_events"] = {session.run_id + ":" + call_id: False}
            update["messages"] = _document_error(
                update.get("messages", []),
                call_id,
                f"Document publication pending; call publish_artifacts: {error}",
            )
        except (ArtifactError, ValidationError) as error:
            detail = f"Specialist result returned; document commit rejected: {error}"
            try:
                restored = (
                    self._materialize(registry, request.state.get("workspace_path"))
                    if registry.artifacts
                    else registry
                )
                if specialist_role in _DOCUMENT_WRITERS:
                    update["artifact_registry"] = restored.model_dump_json()
            except MaterializationError as restore_error:
                if specialist_role in _DOCUMENT_WRITERS:
                    pending = registry.model_copy(update={"materialized": False})
                    update["artifact_registry"] = pending.model_dump_json()
                update["search_recovery_events"] = {session.run_id + ":" + call_id: False}
                detail += f"; restoration pending; call publish_artifacts: {restore_error}"
            update["messages"] = _document_error(
                update.get("messages", []),
                call_id,
                detail,
            )
        update["search_task_events"] = {
            session.run_id + ":" + call_id: str(args.get("task_id", ""))
        }
        update.pop("search_state", None)
        update.pop("search_receipt", None)
        update.pop("search_last_review", None)
        return Command(update=update)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[str]],
    ) -> ToolMessage | Command[str]:
        if not self._enabled(request.state):
            return handler(request)
        error = self._guard(request)
        return error if error else self._returned(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[str]]],
    ) -> ToolMessage | Command[str]:
        if not self._enabled(request.state):
            return await handler(request)
        error = self._guard(request)
        if error:
            return error
        result = await handler(request)
        return await to_thread.run_sync(self._returned, request, result)


def get_search_middleware(
    role: str | None = None, backend: BackendProtocol | None = None, **_deps: JsonValue
) -> list[SearchRuntimeMiddleware]:
    specification = RoleRegistry.get(role) if role else None
    orchestrator = role == "decepticon" or (
        specification is not None
        and specification.llm_role_fallback == "decepticon"
        and specification.slots == SLOTS_PER_ROLE["decepticon"]
    )
    return [SearchRuntimeMiddleware(backend)] if orchestrator and backend is not None else []
