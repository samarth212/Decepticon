"""Server-owned bookkeeping for the fixed job; no model or target calls."""
from pathlib import PurePosixPath

from langchain_core.messages import AIMessage, ToolMessage


def successful_objective_task_return(state, objective_id):
    calls = {call['id'] for message in state.get('messages', []) if isinstance(message, AIMessage)
             for call in message.tool_calls if call['name'] == 'task'
             and call['args'].get('task_id') == objective_id}
    for message in reversed(state.get('messages', [])):
        if isinstance(message, ToolMessage) and message.tool_call_id in calls:
            return message.status == 'success' and not str(message.content).lower().startswith('error')
    return False


def completion_evidence(state, target, refs):
    """Bind mandatory artifacts from the pinned workflow, not model recollection."""
    from decepticon.workflows import WorkflowSnapshot

    result = list(refs if refs is not None else target.get('evidence_refs', []))
    pin = state.get('workflow_snapshot')
    if pin:
        snapshot = WorkflowSnapshot.model_validate_json(pin)
        stage = next((row for row in snapshot.document.metadata.stages
                      if row.id == target.get('workflow_stage')), None)
        if stage:
            result.extend(stage.required_artifacts)
    normalized = []
    for ref in result:
        path = PurePosixPath(ref)
        if (not path.is_absolute() and '..' not in path.parts and str(path) == ref):
            ref = '/workspace/' + ref
        if ref not in normalized:
            normalized.append(ref)
    # The existing native evidence validator still checks every path and file.
    return normalized
