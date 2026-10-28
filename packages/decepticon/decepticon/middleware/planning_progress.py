"""Stop coordinator planning loops before they consume the whole model budget."""
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Annotated, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config

from decepticon.tools.opplan import OPPLAN_TOOL_NAMES

PLANNING_CALL_LIMIT = 8
PLANNING_TOOLS = OPPLAN_TOOL_NAMES | {'write_todos', 'read_todos', 'review_search'}
EVIDENCE_DIRS = ('evidence', 'recon', 'findings', 'report')


class PlanningProgressState(AgentState):
    planning_calls_without_progress: Annotated[NotRequired[int], OmitFromInput]
    planning_seen_results: Annotated[NotRequired[list[str]], OmitFromInput]
    planning_evidence: Annotated[NotRequired[dict[str, str]], OmitFromInput]


def evidence_snapshot(workspace):
    """Only saved nonempty evidence counts; plan files and logs never count."""
    root = Path(workspace).resolve()
    snapshot = {}
    for directory in EVIDENCE_DIRS:
        base = root / directory
        if base.is_symlink() or not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as stream:
                while block := stream.read(65536):
                    size += len(block)
                    digest.update(block)
            if size:
                snapshot[str(path.relative_to(root))] = digest.hexdigest()
    return snapshot


class PlanningProgressMiddleware(AgentMiddleware):
    state_schema = PlanningProgressState

    def before_model(self, state, runtime):
        workspace = state.get('workspace_path')
        if not workspace:
            raise RuntimeError('Planning progress cutoff requires an engagement workspace')
        seen = list(state.get('planning_seen_results') or [])
        count = state.get('planning_calls_without_progress', 0)
        previous_evidence = state.get('planning_evidence')
        evidence = evidence_snapshot(workspace)
        new_evidence = previous_evidence is not None and any(
            previous_evidence.get(path) != digest for path, digest in evidence.items())
        messages = state.get('messages') or []
        last_ai = next((index for index in range(len(messages) - 1, -1, -1)
                       if isinstance(messages[index], AIMessage)), None)
        if last_ai is not None:
            names = {call['id']: call['name'] for call in messages[last_ai].tool_calls}
            for message in messages[last_ai + 1:]:
                if not isinstance(message, ToolMessage):
                    continue
                key = message.id or message.tool_call_id
                if key in seen:
                    continue
                seen.append(key)
                name = names.get(message.tool_call_id, message.name)
                if name == 'task' and message.status == 'success' and not str(message.content).lower().startswith('error'):
                    count = 0
                elif name in PLANNING_TOOLS:
                    count += 1
        if new_evidence:
            count = 0
        if count >= PLANNING_CALL_LIMIT:
            marker = Path(workspace) / 'plan/progress-cutoff.json'
            record = {'code': 'planning_stalled', 'engagement_name': state.get('engagement_name'),
                'thread_id': (get_config().get('configurable') or {}).get('thread_id'),
                'planning_calls': count, 'limit': PLANNING_CALL_LIMIT}
            # No tool output, credentials, target data or model prose in this marker.
            with tempfile.NamedTemporaryFile(mode='w', dir=marker.parent,
                    prefix='.progress-cutoff-', delete=False) as stream:
                stream.write(json.dumps(record) + '\n')
            Path(stream.name).replace(marker)
            raise RuntimeError(f'PLANNING_STALLED: stopped after {count} planning calls '
                'without a successful specialist dispatch or new saved evidence')
        return {'planning_calls_without_progress': count,
            'planning_seen_results': seen[-64:], 'planning_evidence': evidence}

    async def abefore_model(self, state, runtime):
        from anyio import to_thread
        return await to_thread.run_sync(self.before_model, state, runtime)
