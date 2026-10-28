"""Package reviewed findings using native registry transitions, without inference."""
from hashlib import sha256

from decepticon.artifacts import (
    ActorContext, ArtifactDraft, ArtifactError, ArtifactKind, ArtifactLink,
    CommitRequest, DraftUpdate, commit, create, update_draft,
)
from decepticon.middleware.search_evidence import evidence_hashes

GENERATOR = 'fixed-assessment-v1'


def assemble_assessment(registry, candidate_ids, backend):
    """Wait for every bound finding; never promote a verdict or close the search."""
    if not candidate_ids:
        return registry
    findings = []
    for identifier in sorted(candidate_ids):
        artifact = registry.artifacts.get(identifier)
        if artifact is None or artifact.draft.kind != ArtifactKind.FINDING:
            raise ArtifactError('Assessment references an unknown finding')
        if not artifact.revisions or artifact.draft != artifact.revisions[-1].draft:
            return registry
        latest = artifact.revisions[-1]
        if latest.author.role not in {'reporter', 'finding_reporter', 'report_drafter'}:
            return registry
        metadata = latest.draft.metadata
        if metadata.get('report_status') != 'complete':
            return registry
        verdicts = [r for r in artifact.revisions if r.author.role in {'finding_verifier', 'verifier', 'validator'}]
        if not verdicts:
            raise ArtifactError('Assessment requires an independent finding verdict')
        verified = verdicts[-1].draft.metadata
        verdict = metadata.get('verification_status')
        if verdict not in {'verified', 'false_positive', 'unverified'} or verdict != verified.get('verification_status'):
            raise ArtifactError('Assessment finding differs from its independent verdict')
        if verdict == 'verified':
            hashes = verified.get('verified_evidence_hashes')
            if not hashes or hashes != evidence_hashes(latest.draft, backend):
                raise ArtifactError('Assessment proof differs from independently saved evidence')
        findings.append(artifact)
    sections = [
        '# Security assessment — draft for human review',
        '## Summary',
        'This assessment packages committed investigator, verifier and reporter work. '
        'It does not perform new tests, grant submission approval, or declare the workflow complete.',
        '## Scope and limitations',
        'Testing is limited to the operator-approved [scope](../scope.md) and '
        '[rules of engagement](../plan/roe.json). Conclusions apply only to the recorded '
        'tests and controls. Preserve each finding’s caveats, redaction limits and '
        'remediation recommendations during human review.',
    ]
    links = []
    for artifact in findings:
        latest = artifact.revisions[-1]
        metadata = latest.draft.metadata
        verdict = metadata['verification_status']
        category = 'Verified finding' if verdict == 'verified' else 'Unconfirmed or refuted claim'
        sections.extend([
            f'## {category}: {artifact.label} — {latest.draft.title}',
            f'Verdict: {verdict}. Reported severity: {latest.draft.severity}. '
            f'CVSS: {metadata.get("cvss_score", "not recorded")} '
            f'({metadata.get("cvss_vector", "vector not recorded")}). '
            f'[Canonical finding, revision {latest.revision}](../{latest.relative_path}).',
            latest.draft.body,
        ])
        links.append(ArtifactLink(relation='reports', document_uuid=artifact.document_uuid,
            revision=latest.revision, content_hash=latest.content_hash))
    sections.extend(['## Coverage and negative results',
        '[Saved observations and negative results](../findings/info_observations-and-negative-results.md).',
        '## Approval', 'Human review is required before any external submission.'])
    draft = ArtifactDraft(kind='assessment_report', title='Security assessment',
        slug='security-assessment-report', body='\n\n'.join(sections), links=tuple(links),
        metadata={'generated_by': GENERATOR, 'report_status': 'draft', 'human_approval': 'pending'})
    reports = [a for a in registry.artifacts.values() if a.draft.kind == ArtifactKind.ASSESSMENT_REPORT]
    context = ActorContext(actor=GENERATOR, role='decepticon')
    if reports:
        report = reports[0]
        if report.draft.metadata.get('generated_by') != GENERATOR:
            raise ArtifactError('An existing assessment is not runtime-owned; preserve it for review')
        if report.draft == draft and report.revisions and report.revisions[-1].draft == draft:
            return registry
        registry = update_draft(registry, DraftUpdate(document_uuid=report.document_uuid,
            expected_draft_version=report.draft_version, draft=draft), context)
    else:
        registry = create(registry, draft, GENERATOR + ':create')
        report = registry.artifacts[str(registry.operations[GENERATOR + ':create'].document_uuid)]
    fingerprint = sha256(draft.model_dump_json().encode()).hexdigest()
    return commit(registry, CommitRequest(document_uuid=report.document_uuid,
        expected_revision=report.revision,
        operation_key=f'{GENERATOR}:{report.revision}:{fingerprint}',
        reason='Assemble committed reviewed findings for human approval'), context)
