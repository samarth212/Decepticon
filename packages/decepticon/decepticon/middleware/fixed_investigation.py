"""Dispatch the first bounded investigation without a coordinator model call."""
import json
from typing import Annotated, NotRequired
from uuid import NAMESPACE_URL, uuid5

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import AIMessage

from decepticon.middleware.opplan_dispatch import validate_task_plan
from decepticon.middleware.workflow_opplan import validate_workflow_plan
from decepticon.tools.opplan import (
    _evidence_error, _persist_opplan_to_backend, _read_text_from_backend,
    _scoped_opplan_backend, _write_text_to_backend,
)
from decepticon.workflows import WorkflowSnapshot
from decepticon_core.types.engagement import Objective
from decepticon.middleware.investigation_submission import submission_path

SEED_NOTE = 'megatron:fixed-first-investigation:v1'


def investigation_id(engagement):
    return str(uuid5(NAMESPACE_URL, 'megatron:fixed-investigation:' + engagement))


class FixedInvestigationState(AgentState):
    fixed_investigation_dispatched: Annotated[NotRequired[bool], OmitFromInput]
    fixed_investigation_id: Annotated[NotRequired[str], OmitFromInput]
    fixed_prior_artifact_ids: Annotated[NotRequired[list[str]], OmitFromInput]


class FixedInvestigationMiddleware(AgentMiddleware):
    state_schema = FixedInvestigationState

    def __init__(self, backend):
        self.backend = backend

    @hook_config(can_jump_to=['tools'])
    def before_model(self, state, runtime):
        if state.get('fixed_investigation_dispatched'):
            return None
        pin = state.get('workflow_snapshot')
        if not pin:
            return None
        snapshot = WorkflowSnapshot.model_validate_json(pin)
        if (snapshot.id, snapshot.version) != ('bug-bounty', '2'):
            return None
        rows = state.get('objectives') or []
        seeded = [row for row in rows if row['id'] == state.get('fixed_investigation_id')
                  or row['id'] == investigation_id(state.get('engagement_name', ''))]
        # Never replace an existing operator/model plan or replay an old run.
        if rows and not seeded:
            return None
        workspace = state.get('workspace_path')
        name = state.get('engagement_name')
        if not workspace or not name:
            raise RuntimeError('Fixed investigation requires an engagement workspace')
        backend = _scoped_opplan_backend(self.backend, workspace)
        if backend is None:
            raise RuntimeError('Fixed investigation requires a workspace backend')
        scope, error = _read_text_from_backend(backend, '/workspace/scope.md')
        if error or not scope or not scope.strip():
            raise RuntimeError('Fixed investigation requires the operator scope.md')
        revision = state.get('plan_revision', 0)
        if not rows:
            if revision or any(isinstance(m, AIMessage) for m in state.get('messages', [])):
                raise RuntimeError('Refusing to seed a previously started investigation')
            # This records the supplied scope, not a claimed investigation result.
            scope_path = '/workspace/plan/program-scope.md'
            current, _ = _read_text_from_backend(backend, scope_path)
            if current is not None and current != scope:
                raise RuntimeError('Existing program scope differs; operator review required')
            error = _write_text_to_backend(backend, scope_path, scope)
            if error:
                raise RuntimeError('Could not save the supplied program scope')
            scope_id = str(uuid5(NAMESPACE_URL, 'megatron:fixed-scope:' + name))
            investigation = investigation_id(name)
            objectives = [
                Objective(id=scope_id, phase='recon', workflow_stage='scope',
                    title='Record operator-provided scope', description='Copy the approved scope without expanding it.',
                    acceptance_criteria=['Operator scope is recorded verbatim.'], priority=1,
                    status='completed', owner='decepticon', evidence_refs=[scope_path],
                    outcome='objective-met'),
                Objective(id=investigation, phase='initial-access', workflow_stage='validate',
                    title='Investigate the approved application scope',
                    description='Perform one bounded investigation under the supplied scope and RoE.',
                    acceptance_criteria=[
                        'Preserve executed tests, positive and negative controls, and exact evidence paths.',
                        'Return candidates or an explicit negative/blocker without self-verifying findings.',
                        'Save findings/info_observations-and-negative-results.md.',
                    ], priority=1, status='in-progress', owner='exploit',
                    blocked_by=[scope_id], notes=SEED_NOTE),
            ]
            error = validate_workflow_plan(snapshot, objectives,
                lambda refs: _evidence_error(self.backend, workspace, refs))
            if error:
                raise RuntimeError('Fixed investigation plan rejected: ' + error)
            rows = [objective.model_dump(mode='json') for objective in objectives]
            error = _persist_opplan_to_backend(self.backend, workspace, rows, name, '',
                revision=1, facts=[], strict=True, expected_revision=0)
            if error:
                raise RuntimeError('Could not seed fixed investigation: ' + error)
            revision = 1
            seeded = [rows[1]]
        if len(seeded) != 1:
            raise RuntimeError('Ambiguous fixed investigation objective')
        objective = seeded[0]
        if objective.get('status') != 'in-progress':
            return None
        proposed = dict(state, objectives=rows, plan_revision=revision)
        error = validate_task_plan(proposed, task_id=objective['id'],
            plan_revision=revision, subagent_type='exploit', backend=self.backend)
        if error:
            raise RuntimeError('Fixed investigation dispatch rejected: ' + error)
        instruction = (state.get('scan_scope') or {}).get('instruction', '')
        description = (
            'Execute the fixed investigation now. Read /workspace/scope.md and /workspace/plan/roe.json. '
            'Follow every scope, gateway, method, data, duration and budget restriction. '
            'This assignment does not authorize additional actions or targets. '
            'Collect the minimum surface information needed within those restrictions, then test a candidate '
            'with positive and negative controls. Return evidence paths and candidate metadata, or a supported '
            'negative/blocker. Do not self-approve findings. Save '
            '/workspace/findings/info_observations-and-negative-results.md. '
            'Independent finding_verifier and finding_reporter work follows in separate assignments.\n\n'
            'Operator-provided scope:\n' + scope + '\n\nLaunch instruction:\n' + instruction
        )
        description += (
            '\n\nRequired structured return: before returning, write JSON to '
            + submission_path(objective['id'])
            + '. The parent runtime registers it automatically; do not create registry entries or canonical finding files. '
            'Exact schema: {"schema_version":1,"outcome":"candidates","summary":"Tests, negatives, caveats",'
            '"candidates":[{"key":"candidate-1","title":"Claim","slug":"candidate-claim",'
            '"severity":"high","body":"Full claim, reproduction steps, controls, impact and caveats",'
            '"evidence_paths":["findings/evidence/actual-proof.txt"]}]}. '
            'Use actual saved nonempty evidence files, engagement-relative paths under an evidence directory. '
            'No extra fields or verification verdicts. For no finding or blocked work use outcome '
            '"no-finding" or "blocked", candidates: [], and explain coverage/blockers in summary. '
            'This submission is required even when no candidate was found. Never invent evidence.'
        )
        return {
            'objectives': rows, 'plan_revision': revision,
            'fixed_investigation_dispatched': True,
            'fixed_investigation_id': objective['id'],
            'fixed_prior_artifact_ids': list(json.loads(state.get('artifact_registry') or '{}').get('artifacts', {})),
            'messages': [AIMessage(content='', id='megatron-first-' + objective['id'], tool_calls=[{
                'name': 'task', 'id': 'megatron-task-' + objective['id'], 'type': 'tool_call',
                'args': {'description': description, 'subagent_type': 'exploit',
                         'task_id': objective['id'], 'plan_revision': revision},
            }])],
            'jump_to': 'tools',
        }

    @hook_config(can_jump_to=['tools'])
    async def abefore_model(self, state, runtime):
        from anyio import to_thread
        return await to_thread.run_sync(self.before_model, state, runtime)
