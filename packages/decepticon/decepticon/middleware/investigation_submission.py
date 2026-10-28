"""Register a fixed investigator's typed output through the parent registry writer."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decepticon.artifacts import (
    ActorContext, ArtifactDraft, ArtifactError, ArtifactRegistry, CommitRequest, commit, create,
)
from decepticon.middleware.search_evidence import evidence_hashes


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    title: str = Field(min_length=1)
    slug: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    severity: Literal["critical", "high", "medium", "low"]
    body: str = Field(min_length=1)
    evidence_paths: list[str] = Field(min_length=1)


class InvestigationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    outcome: Literal["candidates", "no-finding", "blocked"]
    summary: str = Field(min_length=1)
    candidates: list[Candidate] = Field(max_length=50)

    @model_validator(mode="after")
    def consistent(self):
        if bool(self.candidates) != (self.outcome == "candidates"):
            raise ValueError("Only outcome candidates may contain candidates, and it must contain at least one")
        if len({item.key for item in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate keys must be unique")
        return self


def submission_path(objective_id: str) -> str:
    # Called only with the runtime-owned fixed objective UUID.
    from uuid import UUID
    return f"/workspace/findings/submissions/{UUID(objective_id)}.json"


def register_submission(registry: ArtifactRegistry, backend, objective_id: str) -> ArtifactRegistry:
    path = submission_path(objective_id)
    files = backend.download_files([path])
    if len(files) != 1 or files[0].error or not files[0].content:
        raise ArtifactError(f"Missing investigation submission: {path}. Save the structured result; do not repeat target requests.")
    raw = files[0].content
    if len(raw) > 1_000_000:
        raise ArtifactError("Investigation submission exceeds 1 MB")
    submission = InvestigationSubmission.model_validate_json(raw)
    drafts = []
    for candidate in submission.candidates:
        draft = ArtifactDraft(kind="finding", title=candidate.title, slug=candidate.slug,
            severity=candidate.severity, body=candidate.body, metadata={
                "verification_status": "unverified", "report_status": "draft",
                "source_objective_id": objective_id,
                "evidence_manifest": [{"path": path} for path in candidate.evidence_paths],
            })
        # Existence/hashes establish saved inputs, never an independent verdict.
        hashes = evidence_hashes(draft, backend)
        draft = draft.model_copy(update={"metadata": {**draft.metadata, "investigator_evidence_hashes": hashes}})
        drafts.append((candidate.key, draft))
    drafts.append(("coverage", ArtifactDraft(kind="info_entry", title="Investigation coverage",
        slug="investigation-coverage", body=submission.summary,
        metadata={"source_objective_id": objective_id, "outcome": submission.outcome})))
    context = ActorContext(actor="fixed-investigation:" + objective_id, role="exploit", objective_id=objective_id)
    for key, draft in drafts:
        operation = f"investigation:{objective_id}:{'info' if key == 'coverage' and draft.kind == 'info_entry' else 'candidate'}:{key}"
        registry = create(registry, draft, operation)
        identifier = registry.operations[operation].document_uuid
        registry = commit(registry, CommitRequest(document_uuid=identifier, expected_revision=0,
            operation_key=operation + ":commit", reason="Register investigator submission; independent verification pending"), context)
    return registry
