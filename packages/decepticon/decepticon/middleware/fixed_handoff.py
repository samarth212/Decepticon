"""Append finding handoffs without asking the coordinator to rebuild its plan."""
from copy import deepcopy
import json
from uuid import NAMESPACE_URL, uuid5
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import SystemMessage, ToolMessage

from decepticon.artifacts import ArtifactKind, ArtifactRegistry
from decepticon.middleware.fixed_investigation import FixedInvestigationState, investigation_id
from decepticon.middleware.workflow_opplan import validate_workflow_plan
from decepticon.tools.opplan import (
    _evidence_error, _inspect_objective_rows, _persist_opplan_to_backend,
    _read_text_from_backend, _scoped_opplan_backend,
)
from decepticon.workflows import WorkflowSnapshot
from decepticon_core.types.engagement import Objective

HANDOFF_NOTE = 'megatron:fixed-handoff:v1:'


class FixedHandoffState(FixedInvestigationState):
    fixed_handoff_bindings: Annotated[NotRequired[dict[str, str]], OmitFromInput]
    fixed_handoff_objective_ids: Annotated[NotRequired[list[str]], OmitFromInput]


class FixedHandoffMiddleware(AgentMiddleware):
    state_schema = FixedHandoffState

    def __init__(self, backend):
        self.backend = backend

    def before_model(self, state, runtime):
        raw = state.get('artifact_registry')
        pin = state.get('workflow_snapshot')
        if not raw or not pin:
            return None
        snapshot = WorkflowSnapshot.model_validate_json(pin)
        if (snapshot.id, snapshot.version) != ('bug-bounty', '2'):
            return None
        existing = state.get('objectives') or []
        source_id = state.get('fixed_investigation_id') or investigation_id(state.get('engagement_name', ''))
        sources = [row for row in existing if row['id'] == source_id
                   and row.get('status') == 'completed' and row.get('workflow_stage') == 'validate']
        if not sources:
            return None
        registry = ArtifactRegistry.model_validate_json(raw)
        if not registry.materialized:
            return None  # Native publication/recovery must finish first.
        rows = deepcopy(existing)
        ids = {row['id'] for row in rows}
        bindings = dict(state.get('fixed_handoff_bindings') or {})
        handoff_ids = list(state.get('fixed_handoff_objective_ids') or [])
        prior = set(state.get('fixed_prior_artifact_ids') or [])
        for source in sources:
            for artifact in registry.artifacts.values():
                candidate_id = str(artifact.document_uuid)
                if (artifact.draft.kind != ArtifactKind.FINDING or not artifact.revisions
                        or artifact.draft != artifact.revisions[-1].draft
                        or candidate_id in prior):
                    continue
                # This job has one fixed investigator. Bind each newly committed
                # candidate by registry UUID, not by filenames or editable notes.
                if candidate_id in bindings and bindings[candidate_id] != source['id']:
                    raise RuntimeError('Finding is already bound to another investigation')
                bindings[candidate_id] = source['id']
                key = f"{state['engagement_name']}:{source['id']}:{artifact.document_uuid}"
                verify_id = str(uuid5(NAMESPACE_URL, 'megatron:verify:' + key))
                report_id = str(uuid5(NAMESPACE_URL, 'megatron:report:' + key))
                path = '/workspace/' + artifact.current_path
                identity = (f'Finding UUID {artifact.document_uuid}; canonical file {path}; '
                            f'committed revision {artifact.revision}. ')
                specs = [
                    (verify_id, 'verify', 'finding_verifier', source['id'],
                     'Independently verify ' + artifact.label,
                     identity + 'Read the approved scope and RoE. Independently reproduce the claim '
                     'with fresh positive and negative controls within all existing limits. Edit only '
                     'the assigned canonical finding; return a verified, false_positive or unverified '
                     'verdict with exact evidence paths and caveats. Set evidence_pointer/evidence_manifest '
                     'to your actual fresh proof files so native import can record their hashes. '
                     'Do not infer success from the hunter report. Investigator inputs: '
                     + json.dumps(source.get('evidence_refs', [])),
                     ['Independent reproduction and controls are saved, or an explicit blocker is recorded.',
                      'The canonical finding has an independent verdict and rationale.',
                      'Record /workspace/plan/artifact-registry.json as workflow evidence after native import.']),
                    (report_id, 'report', 'finding_reporter', verify_id,
                     'Prepare the reviewed draft for ' + artifact.label,
                     identity + 'Read the independent verdict and evidence. Preserve that verdict; '
                     'complete and check the canonical finding only. Refuted or blocked claims remain '
                     'coverage/limitations, not verified vulnerabilities. Return to the coordinator for '
                     'the aggregate /workspace/report/security-assessment-report.md. No external submission.',
                     ['The canonical draft preserves the independent verdict and cited evidence.',
                      'The coordinator saves and reviews /workspace/report/security-assessment-report.md before completion.']),
                ]
                for identifier, stage, owner, dependency, title, description, criteria in specs:
                    note = HANDOFF_NOTE + key + ':' + stage
                    if identifier in ids:
                        row = next(row for row in rows if row['id'] == identifier)
                        if (row.get('owner') != owner
                                or row.get('workflow_stage') != stage or row.get('blocked_by') != [dependency]):
                            raise RuntimeError('Fixed handoff objective identity changed; review required')
                        if identifier not in handoff_ids:
                            handoff_ids.append(identifier)
                        continue
                    row = Objective(id=identifier, phase='reporting', workflow_stage=stage,
                        title=title, description=description, acceptance_criteria=criteria,
                        priority=1, owner=owner, status='pending', blocked_by=[dependency],
                        notes=note).model_dump(mode='json')
                    rows.append(row)
                    ids.add(identifier)
                    handoff_ids.append(identifier)
        identity = {'fixed_handoff_bindings': bindings, 'fixed_handoff_objective_ids': handoff_ids}
        if rows == existing:
            if not bindings and not handoff_ids:
                return None
            return identity if any(state.get(key) != value for key, value in identity.items()) else None
        workspace = state.get('workspace_path')
        inspection = _inspect_objective_rows(rows, state.get('plan_facts', []))
        if inspection.issues:
            raise RuntimeError('Fixed handoff produced an invalid objective graph')
        error = validate_workflow_plan(snapshot, [Objective.model_validate(row) for row in rows],
            lambda refs: _evidence_error(self.backend, workspace, refs))
        if error:
            raise RuntimeError('Fixed handoff rejected: ' + error)
        revision = state.get('plan_revision', 0)
        backend = _scoped_opplan_backend(self.backend, workspace)
        if backend is None:
            raise RuntimeError('Fixed handoff requires the engagement workspace backend')
        disk, error = _read_text_from_backend(backend, '/workspace/plan/opplan.json')
        if error or disk is None:
            raise RuntimeError('Fixed handoff cannot read the saved plan')
        saved = json.loads(disk)
        facts = state.get('plan_facts') or []
        # Recover a save completed just before its graph checkpoint was lost.
        saved_rows = {row['id']: row for row in saved.get('objectives', [])}
        recovered = (saved.get('revision') == revision + 1
            and saved.get('engagement_name') == state['engagement_name']
            and saved.get('threat_profile') == state.get('threat_profile', '')
            and saved.get('facts', []) == facts
            and len(saved.get('objectives', [])) == len(rows)
            and saved_rows == {row['id']: row for row in rows})
        if not recovered:
            error = _persist_opplan_to_backend(self.backend, workspace, rows,
                state['engagement_name'], state.get('threat_profile', ''), revision=revision + 1,
                facts=facts, strict=True, expected_revision=revision)
            if error:
                raise RuntimeError('Fixed handoff could not save plan: ' + error)
        return {'objectives': rows, 'plan_revision': revision + 1, **identity}

    async def abefore_model(self, state, runtime):
        from anyio import to_thread
        return await to_thread.run_sync(self.before_model, state, runtime)

    @staticmethod
    def _request(request):
        controlled_ids = set(request.state.get('fixed_handoff_objective_ids') or [])
        handoffs = [row for row in request.state.get('objectives', []) if row['id'] in controlled_ids]
        if not handoffs:
            return request
        text = ('Megatron has already added the verifier/reporter objectives in application code. '
                'Preserve the completed scope/investigation and their evidence references. '
                'Use get_objective/update_objective for acceptance/status, and review_search before '
                'task dispatch. Revisions are bound by the runtime; omit plan_revision on review_search '
                'and finalize_search. Successful managed worker returns can be accepted directly with '
                'status completed; required artifact paths are bound and checked by the runtime. '
                'Use a typed outcome (finding, no-finding or objective-met), not a verdict as outcome. '
                'Use the existing objective IDs; include the '
                'assigned canonical UUID/path in each task description. Never self-verify. '
                'Complete verification before reporting; produce honest coverage for rejected/blocked '
                'claims. The coordinator owns the aggregate assessment. Plan replacement is unavailable '
                'for this fixed handoff. Existing scope, budget, proof and finalization gates still apply.')
        content = list(request.system_message.content_blocks) if request.system_message else []
        content.append({'type': 'text', 'text': text})
        tools = [tool for tool in request.tools
                 if (tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', None)) != 'commit_opplan']
        return request.override(system_message=SystemMessage(content=content), tools=tools)

    def wrap_model_call(self, request, handler):
        return handler(self._request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._request(request))

    @staticmethod
    def _plan_replacement_error(request):
        if request.tool_call['name'] == 'commit_opplan' and request.state.get('fixed_handoff_objective_ids'):
            return ToolMessage(content='This fixed handoff plan is application-managed. Use '
                'get_objective/update_objective, review_search and task with the existing IDs; '
                'commit_opplan cannot replace its objectives.',
                tool_call_id=request.tool_call['id'], status='error')
        return None

    def wrap_tool_call(self, request, handler):
        error = self._plan_replacement_error(request)
        return error if error is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        error = self._plan_replacement_error(request)
        return error if error is not None else await handler(request)
