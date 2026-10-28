"""Offline native registration tests: start with files, never a seeded registry."""
import json
import asyncio
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml

from deepagents.backends import FilesystemBackend
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgentMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from decepticon.artifacts import ArtifactError, ArtifactRegistry, MaterializationError
from decepticon.middleware.fixed_investigation import FixedInvestigationMiddleware
from decepticon.middleware.fixed_handoff import FixedHandoffMiddleware
from decepticon.middleware.investigation_submission import register_submission, submission_path
from decepticon.middleware.search_evidence import scoped_backend
from decepticon.middleware.search_policy import SearchSession
from decepticon.middleware.search_runtime import SearchRuntimeMiddleware
from decepticon.middleware.search_runtime import SearchRuntimeState
from decepticon.middleware.workflow import WorkflowState
from decepticon.middleware.opplan import OPPLANMiddleware
from decepticon.middleware.opplan_dispatch import planned_task_tool
from decepticon.tools.opplan import _persist_opplan_to_backend
from decepticon.workflows import WorkflowCatalog


class ScriptedCoordinator(BaseChatModel):
    """Deterministic test driver, never a provider or a claim of model quality."""
    root: Path
    investigator: str
    step: int = 0

    @property
    def _llm_type(self):
        return 'offline-submission-chain'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        latest = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        if latest and latest.status == 'error':
            raise AssertionError(latest.content)
        plan = json.loads((self.root / 'workspace/plan/opplan.json').read_text())
        objectives = {o['workflow_stage']: o for o in plan['objectives']}
        registry_path = self.root / 'workspace/plan/artifact-registry.json'
        registry = ArtifactRegistry.model_validate_json(registry_path.read_text()) if registry_path.exists() else ArtifactRegistry()
        finding = next((a for a in registry.artifacts.values() if a.draft.kind == 'finding'), None)
        report = next((a for a in registry.artifacts.values() if a.draft.kind == 'assessment_report'), None)
        step = self.step
        self.step += 1
        if step == 0:
            name, args = 'task', {'task_id': self.investigator, 'subagent_type': 'exploit', 'description': 'Write synthetic structured submission.'}
        elif step == 1:
            name, args = 'update_objective', {'objective_id': self.investigator, 'status': 'completed', 'outcome': 'finding', 'evidence_refs': ['/workspace/plan/artifact-registry.json', '/workspace/findings/info_observations-and-negative-results.md']}
        elif step in (2, 5):
            stage = 'verify' if step == 2 else 'report'
            name, args = 'review_search', {'request': {'next_objective_id': objectives[stage]['id'], 'rationale': 'Review synthetic evidence and dispatch the assigned independent stage.'}}
        elif step in (3, 6):
            stage = 'verify' if step == 3 else 'report'
            row = objectives[stage]
            name, args = 'task', {'task_id': row['id'], 'subagent_type': row['owner'], 'description': row['description']}
        elif step in (4, 9):
            stage = 'verify' if step == 4 else 'report'
            name, args = 'update_objective', {'objective_id': objectives[stage]['id'], 'status': 'completed', 'outcome': 'objective-met', 'evidence_refs': []}
        elif step == 7:
            name, args = 'create_artifact', {'operation_key': 'assessment', 'draft': {'kind': 'assessment_report', 'title': 'Offline fixture assessment', 'slug': 'fixture-assessment', 'body': 'Synthetic test only; no real vulnerability claim.', 'links': [{'relation': 'reports', 'document_uuid': str(finding.document_uuid), 'revision': finding.revision, 'content_hash': finding.content_hash}]}}
        elif step == 8:
            name, args = 'commit_artifact', {'request': {'document_uuid': str(report.document_uuid), 'expected_revision': 0, 'operation_key': 'assessment-commit', 'reason': 'Save fixture assessment'}}
        elif step == 10:
            name, args = 'review_search', {'request': {'rationale': 'All synthetic objectives reconciled.'}}
        elif step == 11:
            name, args = 'finalize_search', {'request': {'disposition': 'goal_met', 'report_document_uuid': str(report.document_uuid), 'reason': 'Synthetic fixture proof only.'}}
        elif step == 12:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='Offline fixture finished.'))])
        else:
            raise AssertionError('Finalization did not complete')
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': f'fixture-{step}', 'type': 'tool_call'}]))])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'workspace').mkdir()
        (self.root / 'workspace/scope.md').write_text('Synthetic offline fixture only.')
        self.backend = FilesystemBackend(root_dir=self.root, virtual_mode=True)
        self.scoped = scoped_backend(self.backend, '/workspace')
        self.state = {'workspace_path': '/workspace', 'engagement_name': 'submission-test',
            'workflow_snapshot': WorkflowCatalog.default().resolve('bug-bounty', '2').model_dump_json(),
            'messages': [HumanMessage(content='Offline fixture')]}
        self.state.update(FixedInvestigationMiddleware(self.backend).before_model(self.state, None))
        self.identifier = self.state['fixed_investigation_id']
        self.state['search_state'] = SearchSession(thread_id='fixture', run_id='run',
            started_at='2026-10-28T00:00:00Z').model_dump_json()
        self.submission = {'schema_version': 1, 'outcome': 'candidates', 'summary': 'Synthetic controls only.',
            'candidates': [{'key': 'first', 'title': 'Synthetic claim', 'slug': 'synthetic-claim',
                'severity': 'high', 'body': 'Fixture claim, not a real vulnerability.',
                'evidence_paths': ['findings/evidence/control.txt']}]}
        self.scoped.write('/workspace/findings/evidence/control.txt', 'synthetic proof bytes')
        self.save()

    def save(self):
        path = self.root / submission_path(self.identifier).lstrip('/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.submission))

    def register(self, registry=None):
        return register_submission(registry or ArtifactRegistry(), self.scoped, self.identifier)

    def returned(self, failed=False, task_id=None):
        request = SimpleNamespace(tool_call={'name': 'task', 'id': 'investigator',
            'args': {'subagent_type': 'exploit', 'task_id': task_id or self.identifier}}, state=self.state)
        runtime = SearchRuntimeMiddleware(self.backend)
        with patch.dict('os.environ', {'DECEPTICON_MEGATRON_RUNTIME': '1'}), patch(
                'decepticon.middleware.search_runtime.get_config', return_value={'configurable': {}}):
            result = runtime._returned(request, ToolMessage(content='Investigation returned',
                tool_call_id='investigator', status='error' if failed else 'success'))
        return runtime, result.update

    def test_return_registers_before_handoff_without_coordinator_tools(self):
        runtime, update = self.returned()
        registry = ArtifactRegistry.model_validate_json(update['artifact_registry'])
        self.assertFalse(registry.materialized)
        finding = next(a for a in registry.artifacts.values() if a.draft.kind == 'finding')
        self.assertEqual(finding.draft.metadata['verification_status'], 'unverified')
        self.assertNotIn('verified_evidence_hashes', finding.draft.metadata)
        self.assertEqual(finding.revisions[0].author.role, 'exploit')
        self.state.update(update)
        self.state.update(runtime.before_model(self.state, None))
        source = self.state['objectives'][1]
        source.update(status='completed', outcome='finding', evidence_refs=[
            '/workspace/plan/artifact-registry.json', '/workspace/findings/info_observations-and-negative-results.md'])
        self.state['plan_revision'] = 2
        self.assertIsNone(_persist_opplan_to_backend(self.backend, '/workspace', self.state['objectives'],
            'submission-test', '', revision=2, facts=[], strict=True, expected_revision=1))
        handoff = FixedHandoffMiddleware(self.backend).before_model(self.state, None)
        self.assertEqual([o['workflow_stage'] for o in handoff['objectives'][-2:]], ['verify', 'report'])
        self.assertIn(str(finding.document_uuid), handoff['objectives'][-2]['description'])

    def test_retry_preserves_identity_and_revisions(self):
        first = self.register()
        self.assertEqual(self.register(first), first)

    def test_changed_retry_rejected(self):
        first = self.register()
        self.submission['candidates'][0]['body'] = 'Changed claim'
        self.save()
        with self.assertRaises(ArtifactError):
            self.register(first)

    def test_no_finding_and_blocked_are_coverage_only(self):
        for outcome in ('no-finding', 'blocked'):
            self.submission.update(outcome=outcome, candidates=[])
            self.save()
            result = self.register()
            self.assertEqual([a.draft.kind for a in result.artifacts.values()], ['info_entry'])

    def test_rejects_self_verdict_duplicate_keys_and_missing_evidence(self):
        original = json.loads(json.dumps(self.submission))
        for change in ('verdict', 'duplicate', 'missing', 'traversal', 'empty'):
            self.submission = json.loads(json.dumps(original))
            candidate = self.submission['candidates'][0]
            if change == 'verdict':
                candidate['verification_status'] = 'verified'
            elif change == 'duplicate':
                self.submission['candidates'].append(candidate.copy())
            elif change == 'empty':
                candidate['evidence_paths'] = []
            else:
                candidate['evidence_paths'] = ['../../secret' if change == 'traversal' else 'findings/evidence/missing.txt']
            self.save()
            with self.subTest(change=change), self.assertRaises((ArtifactError, ValidationError)):
                self.register()

    def test_failed_or_unrelated_task_does_not_register(self):
        for kwargs in ({'failed': True}, {'task_id': 'unrelated'}):
            _, update = self.returned(**kwargs)
            self.assertNotIn('artifact_registry', update)

    def test_invalid_submission_is_a_recoverable_tool_error(self):
        self.submission['candidates'][0]['verification_status'] = 'verified'
        self.save()
        _, update = self.returned()
        self.assertNotIn('artifact_registry', update)
        self.assertEqual(update['messages'][0].status, 'error')

    def test_missing_submission_and_parallel_dispatch_are_rejected(self):
        (self.root / submission_path(self.identifier).lstrip('/')).unlink()
        runtime, update = self.returned()
        self.assertEqual(update['messages'][0].status, 'error')
        self.assertIn('Missing investigation submission', update['messages'][0].content)
        calls = [{'name': 'task', 'id': name, 'args': {'task_id': self.identifier, 'subagent_type': 'exploit'}} for name in ('one', 'two')]
        self.state['messages'] = [AIMessage(content='', tool_calls=calls)]
        blocked = runtime._guard(SimpleNamespace(tool_call=calls[0], state=self.state))
        self.assertEqual(blocked.status, 'error')
        self.assertIn('Serialize the fixed investigator', blocked.content)

    def test_opt_out_does_not_import_submission(self):
        request = SimpleNamespace(tool_call={'name': 'task', 'id': 'investigator',
            'args': {'subagent_type': 'exploit', 'task_id': self.identifier}}, state=self.state)
        with patch.dict('os.environ', {'DECEPTICON_MEGATRON_RUNTIME': '0'}), patch(
                'decepticon.middleware.search_runtime.get_config', return_value={'configurable': {}}):
            result = SearchRuntimeMiddleware(self.backend)._returned(request,
                ToolMessage(content='returned', tool_call_id='investigator'))
        self.assertNotIn('artifact_registry', result.update)

    def test_publication_failure_retains_checkpoint_for_retry(self):
        runtime, update = self.returned()
        self.state.update(update)
        saved = self.state['artifact_registry']
        with patch.object(runtime, '_materialize', side_effect=MaterializationError('fixture', 'unavailable')):
            with self.assertRaises(MaterializationError):
                runtime.before_model(self.state, None)
        self.assertEqual(self.state['artifact_registry'], saved)
        published = runtime.before_model(self.state, None)
        self.state.update(published)
        self.assertIsNone(runtime.before_model(self.state, None))
        self.assertEqual(set(ArtifactRegistry.model_validate_json(saved).artifacts),
            set(ArtifactRegistry.model_validate_json(published['artifact_registry']).artifacts))

    def pipeline(self, asynchronous):
        (self.root / submission_path(self.identifier).lstrip('/')).unlink()
        initial = deepcopy(self.state)
        actors = []
        case = self

        class FixtureState(WorkflowState, SearchRuntimeState):
            pass

        class Setup(AgentMiddleware):
            state_schema = FixtureState

            def before_agent(self, state, runtime):
                return {k: v for k, v in initial.items() if k not in ('messages', 'jump_to')}

        def investigator(state):
            actors.append('exploit')
            case.assertEqual(len(state['messages']), 1)
            case.save()
            return {'messages': [AIMessage(content='Synthetic submission saved.') ]}

        def edit_finding(role):
            def specialist(state):
                actors.append(role)
                case.assertEqual(len(state['messages']), 1)
                registry = ArtifactRegistry.model_validate_json((case.root / 'workspace/plan/artifact-registry.json').read_text())
                finding = next(a for a in registry.artifacts.values() if a.draft.kind == 'finding')
                path = case.root / 'workspace' / finding.current_path
                parts = path.read_text().split('---', 2)
                metadata = yaml.safe_load(parts[1])
                if role == 'finding_verifier':
                    proof = case.root / 'workspace/findings/evidence/fresh-controls.txt'
                    proof.write_text('Synthetic positive: 200 fixture data; matched negative: 403. No network.')
                    metadata.update(result_kind='vulnerability', verification_status='verified',
                        verification_rationale='Synthetic positive and negative controls saved independently.',
                        evidence_pointer='findings/evidence/fresh-controls.txt',
                        evidence_manifest=[{'path': 'findings/evidence/fresh-controls.txt'}],
                        affected_target='fixture.invalid', affected_component='fixture', cvss_score=7.5,
                        cvss_vector='CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N', cvss_version='3.1')
                else:
                    case.assertEqual(metadata['verification_status'], 'verified')
                    case.assertTrue(metadata['verified_evidence_hashes'])
                    metadata.update(report_status='complete', verification_status='false_positive')
                body = '## Summary\nSynthetic fixture.\n## Verification\nSynthetic controls only.\n## Impact\nNo real claim.\n## Remediation\nFixture only.'
                path.write_text('---\n' + '\n'.join(f'{json.dumps(k)}: {json.dumps(v)}' for k, v in metadata.items()) + '\n---\n' + body)
                return {'messages': [AIMessage(content='Synthetic specialist result saved.')]}
            return specialist

        subagents = SubAgentMiddleware(backend=self.backend, subagents=[
            CompiledSubAgent(name='exploit', description='Fixture hunter', runnable=RunnableLambda(investigator)),
            CompiledSubAgent(name='finding_verifier', description='Fixture verifier', runnable=RunnableLambda(edit_finding('finding_verifier'))),
            CompiledSubAgent(name='finding_reporter', description='Fixture reporter', runnable=RunnableLambda(edit_finding('finding_reporter')))])
        subagents.tools = [planned_task_tool(subagents.tools[0], backend=self.backend)]
        graph = create_agent(ScriptedCoordinator(root=self.root, investigator=self.identifier),
            middleware=[Setup(), subagents, OPPLANMiddleware(backend=self.backend), SearchRuntimeMiddleware(self.backend),
                FixedInvestigationMiddleware(self.backend), FixedHandoffMiddleware(self.backend)], checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'fixture', 'search_run_id': 'run', 'research_scope_digest': 'a' * 64}, 'recursion_limit': 200}
        with patch.dict('os.environ', {'DECEPTICON_MEGATRON_RUNTIME': '1'}):
            payload = {'messages': [HumanMessage(content='Run offline fixture')]}
            result = asyncio.run(graph.ainvoke(payload, config)) if asynchronous else graph.invoke(payload, config)
        self.assertEqual(actors, ['exploit', 'finding_verifier', 'finding_reporter'])
        registry = ArtifactRegistry.model_validate_json(result['artifact_registry'])
        finding = next(a for a in registry.artifacts.values() if a.draft.kind == 'finding')
        self.assertEqual([r.author.role for r in finding.revisions], ['exploit', 'finding_verifier', 'reporter'])
        self.assertEqual(finding.draft.metadata['verification_status'], 'verified')
        self.assertEqual(json.loads(result['search_receipt'])['disposition'], 'goal_met')

    def test_sync_raw_submission_to_final_receipt(self):
        self.pipeline(False)

    def test_async_raw_submission_to_final_receipt(self):
        self.pipeline(True)
