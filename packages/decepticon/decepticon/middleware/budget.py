"""BudgetEnforcementMiddleware — per-engagement and per-agent USD spend caps.

Decepticon engagements can spin for hours with multiple specialist sub-agents
running in parallel, each driving an expensive frontier model. The
LiteLLM-only timeout knob (``DECEPTICON_LLM__TIMEOUT``) is necessary but not
sufficient — a long-running engagement on ``claude-opus-4-8 → fallback →
fallback`` can quietly accrue hundreds of dollars in spend before a human
operator notices.

This middleware exposes a single financial guardrail::

    DECEPTICON_BUDGET__ENGAGEMENT_USD    hard cap for the whole engagement
    DECEPTICON_BUDGET__PER_AGENT_USD     hard cap per specialist agent
    DECEPTICON_BUDGET__SOFT_WARN_AT_PCT  emit a stream warning at this fraction
    DECEPTICON_BUDGET__POLL_SECONDS      how often to re-query LiteLLM

Spend numbers come from the LiteLLM proxy's ``/spend/tags`` API. The
middleware tags every outbound completion with its scope keys
(``engagement:<slug>`` and ``agent:<slug>:<agent>`` via request
``metadata.tags``), the proxy records them in its spend logs, and
``/spend/tags`` returns the cumulative USD total per tag. We query it
before every budget-controlled model call. Accounting errors block the call.

Enforcement
-----------
- ``soft_warn`` threshold (default 70%) → emit a ``budget_warning`` custom
  stream event so the operator UI shows a yellow banner, but the agent
  proceeds.
- ``hard_pause`` threshold (100%) → raise ``BudgetExceeded`` from
  ``wrap_model_call``, which bubbles up to the agent runtime and terminates
  the run with an explanatory message. The HITL gate (§2.2) will eventually
  catch this and offer the operator a "raise budget" prompt; until that
  lands, the engagement just stops.

Disabled by default
-------------------
A non-positive ``ENGAGEMENT_USD`` cap means the middleware is a no-op. This
preserves the current behavior for users who haven't opted in.

When a cap is enabled, a credential chain containing an unpriced metered
Cheaper Inference or Kimi route is rejected before a model call. Their
placeholder $0 LiteLLM prices cannot enforce a USD cap, including when such
a route is reached as a fallback.

Architectural notes
-------------------
- LiteLLM proxy spend is the source of truth, NOT a local counter. A
  local counter would drift on crash/restart; LiteLLM persists every call.
- Spend attribution rides the documented proxy API (``metadata.tags`` in,
  ``/spend/tags`` out) rather than SQL against LiteLLM's internal Prisma
  tables, so proxy upgrades can't break enforcement on a schema change.
- The middleware does NOT block tool calls — only LLM calls. The agent can
  finish whatever tool round-trip is in flight and emit a final synthesis
  message before the next inference fails.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from typing import Any, ClassVar

from langchain.agents.middleware import AgentMiddleware
from typing_extensions import override

from decepticon_core.types.llm import AuthMethod

log = logging.getLogger(__name__)

# AuthMethods whose config/litellm.yaml routes carry a placeholder
# input_cost_per_token=0 / output_cost_per_token=0 (no published rate card
# from the gateway) rather than a real per-token price. LiteLLM's
# /spend/tags aggregates SUM(spend) computed from those model_info fields
# times usage, so a $0 route reports $0 spend there no matter how much the
# provider actually charges. Budget enforcement built on that API is
# unable to enforce caps for these methods: tracked spend never moves.
# A configured cap therefore blocks a chain containing either method. See
# the "ponytail: per-token cost unverified" comments in config/litellm.yaml.
_SPEND_UNTRACKED_METHODS: frozenset[AuthMethod] = frozenset(
    {AuthMethod.CHEAPER_INFERENCE_API, AuthMethod.KIMI_API}
)


class BudgetAccountingUnavailable(RuntimeError):
    """A configured budget cannot establish a trustworthy current spend."""


class BudgetExceeded(RuntimeError):
    """Raised when an engagement or per-agent budget has been hit."""

    def __init__(self, scope: str, spent_usd: float, cap_usd: float) -> None:
        self.scope = scope
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"Budget exceeded ({scope}): spent ${spent_usd:.4f} of ${cap_usd:.2f} cap. "
            "Raise the cap via DECEPTICON_BUDGET__* or end the engagement."
        )


class BudgetSpendUntracked(RuntimeError):
    """Raised when an active route has no usable price for a configured cap."""

    def __init__(self, methods: frozenset[AuthMethod]) -> None:
        self.methods = methods
        names = ", ".join(sorted(method.value for method in methods))
        super().__init__(
            f"Cannot enforce DECEPTICON_BUDGET__* caps while {names} is in the "
            "active credential chain: its LiteLLM route has a placeholder $0 "
            "price, so /spend/tags cannot measure provider charges. Disable "
            "that credential for this run or remove the budget cap."
        )


class BudgetCredentialCheckUnavailable(RuntimeError):
    """Raised when a configured budget cannot verify the active auth chain."""

    def __init__(self) -> None:
        super().__init__(
            "Cannot verify the active credential chain for DECEPTICON_BUDGET__* "
            "caps; model call blocked because an unpriced fallback could bypass "
            "the spending limit."
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("invalid float for %s=%r; falling back to %s", name, raw, default)
        return default


class _SpendCache:
    """Tiny TTL cache over (scope_key) → (spent_usd, fetched_at).

    Bounded at ``_MAX_ENTRIES`` so a runaway agent-id-explosion can't grow
    memory unbounded. Entries are evicted in insertion order (FIFO) on
    overflow — cache misses cost one extra LiteLLM proxy round-trip, not
    correctness, so FIFO is fine.
    """

    _MAX_ENTRIES: ClassVar[int] = 1024

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, float]] = {}

    def get(self, key: str) -> float | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, fetched_at = entry
        if (time.monotonic() - fetched_at) > self._ttl:
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: str, value: float) -> None:
        self._entries[key] = (value, time.monotonic())
        if len(self._entries) > self._MAX_ENTRIES:
            first_key = next(iter(self._entries))
            self._entries.pop(first_key, None)


def _emit_spend_untracked_event(methods: frozenset[AuthMethod]) -> None:
    """Emit a ``budget_spend_untracked`` custom stream event for the dashboard."""
    try:
        from langgraph.config import get_stream_writer  # noqa: PLC0415

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    if writer is None:
        return
    try:
        writer(
            {
                "type": "budget_spend_untracked",
                "methods": sorted(m.value for m in methods),
            }
        )
    except Exception as e:  # noqa: BLE001
        log.warning("failed to emit budget_spend_untracked stream event: %s", e)


def _emit_warning_event(scope: str, spent: float, cap: float, frac: float) -> None:
    """Emit a ``budget_warning`` custom stream event for the dashboard."""
    try:
        from langgraph.config import get_stream_writer  # noqa: PLC0415

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    if writer is None:
        return
    try:
        writer(
            {
                "type": "budget_warning",
                "scope": scope,
                "spent_usd": round(spent, 4),
                "cap_usd": cap,
                "used_pct": round(frac * 100, 1),
            }
        )
    except Exception as e:  # noqa: BLE001
        log.warning("failed to emit budget_warning stream event: %s", e)


class BudgetEnforcementMiddleware(AgentMiddleware):
    """Hard-pause when engagement or per-agent USD spend caps are hit.

    Construct once per engagement (or globally — the middleware reads scope
    from request metadata each call). Spend numbers are pulled from the
    injected ``spend_provider`` callable, which is expected to return the
    cumulative USD spend for a given scope key. The default provider
    queries the LiteLLM proxy ``/spend/tags`` API using the same proxy
    URL/key the LLM factory resolves; injecting a stub provider makes
    this trivially unit-testable.
    """

    def __init__(
        self,
        *,
        engagement_cap_usd: float | None = None,
        per_agent_cap_usd: float | None = None,
        soft_warn_at_pct: float | None = None,
        poll_seconds: float | None = None,
        spend_provider: Any = None,
    ) -> None:
        super().__init__()
        self._engagement_cap = (
            engagement_cap_usd
            if engagement_cap_usd is not None
            else _env_float("DECEPTICON_BUDGET__ENGAGEMENT_USD", 0.0)
        )
        self._per_agent_cap = (
            per_agent_cap_usd
            if per_agent_cap_usd is not None
            else _env_float("DECEPTICON_BUDGET__PER_AGENT_USD", 0.0)
        )
        self._soft_warn = (
            soft_warn_at_pct
            if soft_warn_at_pct is not None
            else _env_float("DECEPTICON_BUDGET__SOFT_WARN_AT_PCT", 0.7)
        )
        poll = (
            poll_seconds
            if poll_seconds is not None
            else _env_float("DECEPTICON_BUDGET__POLL_SECONDS", 5.0)
        )
        self._cache = _SpendCache(ttl_seconds=poll)
        self._spend_provider = spend_provider or _default_litellm_spend_provider
        self._warned_scopes: set[str] = set()
        self._disclosed_untracked_spend = False
        # awrap_model_call runs _enforce in a thread-pool executor, so two
        # concurrent agents sharing an engagement can enter _check_one on
        # different threads. This lock keeps the cache-miss→fetch→set window
        # and the soft-warn check-then-add on _warned_scopes atomic, so the
        # "at most one HTTP fetch per poll interval" and "soft-warn fires once
        # per scope" invariants hold under concurrency. It only serializes
        # budget-enforce threads — the event loop itself is never blocked
        # (enforcement runs off-loop via asyncio.to_thread).
        self._lock = threading.Lock()

    def _enabled(self) -> bool:
        return self._engagement_cap > 0 or self._per_agent_cap > 0

    def _reject_if_spend_untracked(self) -> None:
        """Fail closed when a configured budget could miss a billed fallback.

        The whole active credential chain matters: a priced primary may
        fall back to an unpriced route during a single engagement. A warning
        cannot make the configured hard cap enforceable in that case.
        """
        if not self._enabled():
            return
        try:
            # Keep factory import lazy: factory and middleware initialization
            # otherwise form a module import cycle.
            from decepticon.llm.factory import auth_inventory  # noqa: PLC0415

            active = set(auth_inventory().resolved_chain)
        except Exception as exc:  # noqa: BLE001
            log.error("could not resolve auth inventory for budget preflight; blocking model call")
            raise BudgetCredentialCheckUnavailable() from exc
        untracked = frozenset(active & _SPEND_UNTRACKED_METHODS)
        if not untracked:
            return
        with self._lock:
            if not self._disclosed_untracked_spend:
                self._disclosed_untracked_spend = True
                log.error("%s", BudgetSpendUntracked(untracked))
                _emit_spend_untracked_event(untracked)
        raise BudgetSpendUntracked(untracked)

    def _scope_keys(self, request: Any) -> tuple[str, str]:
        """Pull engagement_id and agent_id from request state; fall back to env."""
        state = getattr(request, "state", None) or {}
        get = state.get if hasattr(state, "get") else (lambda _k, _d=None: None)
        engagement = (
            get("engagement_name")
            or get("engagement_id")
            or os.environ.get("DECEPTICON_ENGAGEMENT_ID", "")
            or "default-engagement"
        )
        agent_name = ""
        runtime = getattr(request, "runtime", None)
        if runtime is not None:
            agent_name = getattr(runtime, "agent_name", "") or ""
        agent_name = agent_name or "default-agent"
        return engagement, agent_name

    def _check_one(self, scope_kind: str, scope_key: str, cap_usd: float) -> None:
        if cap_usd <= 0:
            return
        # Hold the lock across the whole check so a cache miss serializes
        # concurrent agents onto a single fetch (the loser sees the warm
        # cache), and so the soft-warn check-then-add can't double-fire.
        with self._lock:
            cached = None
            if cached is None:
                try:
                    cached = float(self._spend_provider(scope_key))
                except Exception as exc:
                    raise BudgetAccountingUnavailable(
                        f"Spend lookup failed for {scope_key}; model call blocked"
                    ) from exc
                if not math.isfinite(cached) or cached < 0:
                    raise BudgetAccountingUnavailable(
                        f"Invalid spend for {scope_key}; model call blocked"
                    )
                self._cache.set(scope_key, cached)
            frac = cached / cap_usd if cap_usd else 0.0
            from decepticon.middleware.run_controls import emit_custom_event

            payload = {
                "scope": scope_kind,
                "spent_usd": round(cached, 4),
                "cap_usd": cap_usd,
                "used_pct": round(frac * 100, 1),
            }
            emit_custom_event("budget_usage", payload)
            if frac >= 1.0:
                emit_custom_event("budget_exceeded", payload)
                raise BudgetExceeded(scope_kind, cached, cap_usd)
            warn_key = f"{scope_kind}:{scope_key}"
            already_warned = warn_key in self._warned_scopes
            if frac >= self._soft_warn and not already_warned:
                self._warned_scopes.add(warn_key)
                log.warning(
                    "budget soft-warn: scope=%s spent=$%.4f of $%.2f (%.0f%%)",
                    scope_kind,
                    cached,
                    cap_usd,
                    frac * 100,
                )
                _emit_warning_event(scope_kind, cached, cap_usd, frac)

    def _enforce(self, request: Any) -> None:
        if not self._enabled():
            return
        self._reject_if_spend_untracked()
        engagement, agent = self._scope_keys(request)
        self._check_one("engagement", f"engagement:{engagement}", self._engagement_cap)
        self._check_one("agent", f"agent:{engagement}:{agent}", self._per_agent_cap)

    def _tag_request(self, request: Any) -> Any:
        """Attach scope tags to the outbound completion via ``metadata.tags``.

        The LiteLLM proxy records these tags on every spend-log row, which
        is what makes the ``/spend/tags`` lookup in the default provider
        work. Tags are merged non-destructively: existing ``model_settings``
        keys (e.g. an ``extra_body.thinking`` block) and pre-existing tags
        are preserved.
        """
        if not self._enabled():
            return request
        engagement, agent = self._scope_keys(request)
        scope_tags = [f"engagement:{engagement}", f"agent:{engagement}:{agent}"]
        settings = dict(getattr(request, "model_settings", None) or {})
        extra_body = dict(settings.get("extra_body") or {})
        metadata = dict(extra_body.get("metadata") or {})
        tags = list(metadata.get("tags") or [])
        tags.extend(t for t in scope_tags if t not in tags)
        metadata["tags"] = tags
        extra_body["metadata"] = metadata
        settings["extra_body"] = extra_body
        return request.override(model_settings=settings)

    @override
    def wrap_model_call(self, request, handler):
        self._enforce(request)
        return handler(self._tag_request(request))

    @override
    async def awrap_model_call(self, request, handler):
        # The default spend provider does a blocking HTTP round-trip to the
        # LiteLLM proxy (at most once per poll interval, per scope); run it
        # off the event loop so concurrent agents aren't stalled behind it.
        await asyncio.to_thread(self._enforce, request)
        return await handler(self._tag_request(request))


def _default_litellm_spend_provider(scope_key: str) -> float:
    """Default spend provider: query the LiteLLM proxy ``/spend/tags`` API.

    ``scope_key`` looks like ``engagement:<slug>`` or ``agent:<slug>:<agent>``
    and matches the tags ``_tag_request`` attaches to every completion this
    middleware wraps. ``/spend/tags`` aggregates ``SUM(spend)`` per distinct
    tag proxy-side, so the response is one row per tag — bounded by tag
    cardinality, not log volume.

    Proxy URL and key come from the same ``DecepticonConfig`` source the LLM
    factory uses (``DECEPTICON_LLM__PROXY_URL`` / ``__PROXY_API_KEY``), so
    any environment that can run an agent can also enforce its budget.

    Fail closed on configuration, transport, HTTP, and response errors.
    A valid response with no matching tag means no recorded spend.
    """
    try:
        from decepticon_core.utils.config import load_config  # noqa: PLC0415

        config = load_config()
        base_url = config.llm.proxy_url.rstrip("/")
        api_key = config.llm.proxy_api_key
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot resolve LiteLLM proxy config for budget query: %s", exc)
        raise BudgetAccountingUnavailable("Cannot resolve accounting configuration") from exc
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(
            f"{base_url}/spend/tags",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            raise ValueError("spend response must be a list")
        result = 0.0
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("individual_request_tag"), str):
                raise ValueError("invalid spend row")
            raw = row.get("total_spend")
            if raw is None or isinstance(raw, bool):
                raise ValueError("missing or invalid spend")
            spent = float(raw)
            if not math.isfinite(spent) or spent < 0:
                raise ValueError("invalid spend")
            if row["individual_request_tag"] == scope_key:
                result += spent
        return result
    except Exception as exc:
        raise BudgetAccountingUnavailable("LiteLLM accounting unavailable; model call blocked") from exc
