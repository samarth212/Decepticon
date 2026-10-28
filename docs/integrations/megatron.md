# Megatron local runtime

Megatron consumes this local fork, not an upstream release image or a second
checkout. Its build records this repository's committed revision; changes must
be committed and the image rebuilt before a new job adopts them.

`DECEPTICON_MEGATRON_RUNTIME=1` opts the coordinator into the native
`decepticon.integrations.megatron` middleware stack. It retains trusted workflow
selection, immutable pins, OPPLAN validation, authorization, independent-verdict
imports and finalization checks. The hooks execute only their bound bounty work;
they do not authorize targets or relax the supplied rules of engagement.

The fork owns fixed investigation/handoff identity, the planning-progress cutoff,
server-bound revision bookkeeping and saved-proof validation. Shared cross-process
tool slots are enabled by `MEGATRON_LIMITS_DIR`; without that setting the existing
per-instance concurrency behavior remains available. Configured USD budgets fail
closed when accounting cannot provide a trustworthy spend value.

The integration contract is `megatron-local-v1`. Megatron validates the source
repository and built image revision before launching a job. Its dashboard, Temporal
worker, job isolation, provider configuration and launch adapter remain downstream.
The runtime modules must not be copied or patched into images by Megatron.

This migration preserves the existing runtime behavior; it does not fix the
remaining unregistered-investigator-output gap. Candidate registration still
needs to become a native structured, idempotent handoff. Offline tests that begin
with a registered candidate do not establish that the full agent flow works.
