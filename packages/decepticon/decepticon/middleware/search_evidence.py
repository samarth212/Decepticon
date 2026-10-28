from __future__ import annotations

import hashlib

import yaml
from deepagents.backends.protocol import BackendProtocol
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from decepticon.artifacts import (
    ActorContext,
    ArtifactDraft,
    ArtifactError,
    ArtifactKind,
    ArtifactRegistry,
    CommitRequest,
    DraftUpdate,
    commit,
    update_draft,
)
from decepticon.middleware.filesystem import (
    EngagementFilesystemBackend,
    _rebind_sandbox_per_run,
)
from decepticon.tools.reporting.evidence_contract import (
    _FrontmatterLoader,
    evidence_path,
)
from decepticon.tools.reporting.finding_quality import check_finding_content

_METADATA = TypeAdapter(dict[str, JsonValue])
_SERVER_FIELDS = frozenset(
    {
        "schema_version",
        "document_type",
        "document_uuid",
        "id",
        "revision",
        "content_hash",
        "title",
        "slug",
        "links",
        "severity",
    }
)
_VERIFIERS = frozenset({"verifier", "finding_verifier", "validator", "bugclaw_validator"})
_REPORTERS = frozenset({"report", "finding_reporter", "report_drafter"})


class SpecialistHandoff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    context: ActorContext
    assignment: str
    scope_digest: str | None = None


class ProofContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    scope_digest: str | None = None
    run_id: str | None = None


def evidence_hashes(draft: ArtifactDraft, backend: BackendProtocol) -> dict[str, str]:
    paths: list[str] = []
    pointer = draft.metadata.get("evidence_pointer")
    if isinstance(pointer, str):
        paths.append(pointer)
    manifest = draft.metadata.get("evidence_manifest", [])
    if isinstance(manifest, list):
        for item in manifest:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.append(str(item["path"]))
    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        try:
            relative = evidence_path(path)
        except ValueError:
            raise ArtifactError("Use engagement-relative evidence_pointer/evidence_manifest paths under an evidence directory") from None
        response = backend.download_files(["/workspace/" + relative])
        if len(response) != 1 or response[0].error or not response[0].content:
            raise ArtifactError(f"Missing exact evidence bytes: {relative}")
        hashes[relative] = hashlib.sha256(response[0].content).hexdigest()
    return hashes


def import_specialist_edits(
    registry: ArtifactRegistry, backend: BackendProtocol, handoff: SpecialistHandoff
) -> ArtifactRegistry:
    context = handoff.context
    if context.role not in _VERIFIERS | _REPORTERS:
        return registry
    for artifact in tuple(registry.artifacts.values()):
        if artifact.draft.kind != ArtifactKind.FINDING or not artifact.revisions:
            continue
        if (
            str(artifact.document_uuid) not in handoff.assignment
            and artifact.current_path not in handoff.assignment
        ):
            continue
        result = backend.read("/workspace/" + artifact.current_path, limit=10_000)
        if result.error or result.file_data is None:
            continue
        raw = result.file_data["content"]
        if raw == artifact.revisions[-1].markdown:
            continue
        parts = raw.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            continue
        try:
            parsed = yaml.load(parts[1], Loader=_FrontmatterLoader)
        except yaml.YAMLError as error:
            raise ArtifactError("Specialist frontmatter is invalid YAML") from error
        metadata = _METADATA.validate_python(parsed)
        if metadata.get("document_uuid") != str(artifact.document_uuid):
            continue
        draft = ArtifactDraft.model_validate(
            {
                **artifact.draft.model_dump(mode="json"),
                "title": metadata.get("title", artifact.draft.title),
                "severity": metadata.get("severity", artifact.draft.severity),
                "body": parts[2].strip(),
                "metadata": {
                    key: value for key, value in metadata.items() if key not in _SERVER_FIELDS
                },
            }
        )
        role = "reporter" if context.role in _REPORTERS else "finding_verifier"
        if role == "finding_verifier":
            hashes = evidence_hashes(draft, backend)
            if draft.metadata.get("verification_status") == "verified" and not hashes:
                raise ArtifactError("Verified verdict requires evidence_pointer or evidence_manifest referencing saved nonempty proof files. Repair metadata using the existing independent proof; do not repeat target requests just to repair the document.")
            draft = draft.model_copy(
                update={
                    "metadata": {
                        **draft.metadata,
                        "verified_evidence_hashes": hashes,
                        "verified_scope_digest": handoff.scope_digest,
                    }
                }
            )
        actor = context.model_copy(update={"role": role})
        registry = update_draft(
            registry,
            DraftUpdate(
                document_uuid=artifact.document_uuid,
                expected_draft_version=artifact.draft_version,
                draft=draft,
            ),
            actor,
        )
        registry = commit(
            registry,
            CommitRequest(
                document_uuid=artifact.document_uuid,
                expected_revision=artifact.revision,
                operation_key=f"{context.actor}:{artifact.document_uuid}:{hashlib.sha256(raw.encode()).hexdigest()}",
                reason="Capture independently returned specialist document",
            ),
            actor,
        )
    return registry


def qualifying_findings(
    registry: ArtifactRegistry, backend: BackendProtocol, context: ProofContext
) -> tuple[str, ...]:
    matches: list[str] = []
    for artifact in registry.artifacts.values():
        if artifact.draft.kind != ArtifactKind.FINDING or not artifact.revisions:
            continue
        latest = artifact.revisions[-1]
        draft = latest.draft
        verdicts = [version for version in artifact.revisions if version.author.role in _VERIFIERS]
        if not verdicts or verdicts[-1].draft.metadata.get("verification_status") != "verified":
            continue
        if context.run_id is not None and not verdicts[-1].author.actor.startswith(
            context.run_id + ":"
        ):
            continue
        verified = verdicts[-1].draft
        if verified.metadata.get("verified_scope_digest") != context.scope_digest:
            continue
        if not verified.metadata.get("verified_evidence_hashes") or verified.metadata[
            "verified_evidence_hashes"
        ] != evidence_hashes(draft, backend):
            continue
        if (
            draft.metadata.get("verification_status") != "verified"
            or draft.metadata.get("result_kind") != "vulnerability"
        ):
            continue
        if draft.severity not in {"high", "critical"} or verified.severity not in {
            "high",
            "critical",
        }:
            continue
        if draft.severity == "critical" and verified.severity != "critical":
            continue
        evidence_fields = (
            "evidence_pointer",
            "evidence_manifest",
            "affected_target",
            "affected_component",
        )
        if any(draft.metadata.get(key) != verified.metadata.get(key) for key in evidence_fields):
            continue
        score = draft.metadata.get("cvss_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 7:
            continue
        if check_finding_content(
            artifact.current_path,
            latest.markdown,
            evidence_exists=lambda path: _exists(backend, path),
        ):
            continue
        matches.append(str(artifact.document_uuid))
    return tuple(matches)


def _exists(backend: BackendProtocol, path: str) -> bool:
    result = backend.read("/workspace/" + path, limit=1)
    return result.error is None and result.file_data is not None


def scoped_backend(backend: BackendProtocol, workspace: str | None) -> BackendProtocol:
    return EngagementFilesystemBackend(_rebind_sandbox_per_run(backend), workspace)
