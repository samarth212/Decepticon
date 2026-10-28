import tempfile
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend

from decepticon.artifacts import (
    ActorContext, ArtifactDraft, ArtifactError, ArtifactRegistry, CommitRequest,
    DraftUpdate, commit, create, update_draft,
)
from decepticon.middleware.assessment import assemble_assessment
from decepticon.middleware.search_evidence import evidence_hashes


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.backend = FilesystemBackend(root_dir=Path(temp.name), virtual_mode=True)
        self.backend.write('/workspace/findings/evidence/proof.txt', 'synthetic controls')
        self.registry = ArtifactRegistry()
        self.ids = []

    def finding(self, verdict='verified', reviewed=True):
        key = str(len(self.ids))
        draft = ArtifactDraft(kind='finding', title='Synthetic candidate', slug='synthetic-candidate',
            body='Fixture evidence and caveats. No live claim.', severity='high',
            metadata={'verification_status': verdict, 'report_status': 'draft',
                'evidence_pointer': 'findings/evidence/proof.txt'})
        draft = draft.model_copy(update={'metadata': {**draft.metadata,
            'verified_evidence_hashes': evidence_hashes(draft, self.backend)}})
        self.registry = create(self.registry, draft, key)
        identifier = self.registry.operations[key].document_uuid
        self.registry = commit(self.registry, CommitRequest(document_uuid=identifier,
            expected_revision=0, operation_key=key+'-verify', reason='Synthetic verification'),
            ActorContext(actor='fixture-verifier', role='finding_verifier'))
        if reviewed:
            self.registry = update_draft(self.registry, DraftUpdate(document_uuid=identifier,
                expected_draft_version=1, draft=draft.model_copy(update={'metadata': {
                    **draft.metadata, 'report_status': 'complete'}})), ActorContext(actor='fixture', role='reporter'))
            self.registry = commit(self.registry, CommitRequest(document_uuid=identifier,
                expected_revision=1, operation_key=key+'-report', reason='Synthetic report'),
                ActorContext(actor='fixture-reporter', role='reporter'))
        self.ids.append(str(identifier))

    def assemble(self, registry=None):
        return assemble_assessment(registry or self.registry, self.ids, self.backend)

    def test_pinned_report_preserves_findings_and_retries(self):
        self.finding()
        result = self.assemble()
        report = next(a for a in result.artifacts.values() if a.draft.kind == 'assessment_report')
        finding = self.registry.artifacts[self.ids[0]]
        self.assertEqual(result.artifacts[self.ids[0]], finding)
        self.assertEqual(report.draft.links[0].revision, finding.revision)
        self.assertEqual(report.draft.links[0].content_hash, finding.content_hash)
        self.assertEqual(report.draft.metadata['human_approval'], 'pending')
        self.assertEqual(self.assemble(result), result)

    def test_waits_for_every_reporter(self):
        self.finding()
        self.finding(reviewed=False)
        self.assertEqual(self.assemble(), self.registry)

    def test_changed_proof_is_rejected(self):
        self.finding()
        self.backend.edit('/workspace/findings/evidence/proof.txt', 'synthetic controls', 'changed')
        with self.assertRaises(ArtifactError):
            self.assemble()

    def test_refuted_claim_is_never_promoted(self):
        self.finding(verdict='false_positive')
        report = next(a for a in self.assemble().artifacts.values() if a.draft.kind == 'assessment_report')
        self.assertIn('Unconfirmed or refuted claim', report.draft.body)
        self.assertNotIn('## Verified finding:', report.draft.body)

    def test_existing_operator_assessment_is_preserved(self):
        self.finding()
        self.registry = create(self.registry, ArtifactDraft(kind='assessment_report',
            title='Operator draft', slug='operator-draft', body='Preserve me'), 'operator')
        with self.assertRaises(ArtifactError):
            self.assemble()
