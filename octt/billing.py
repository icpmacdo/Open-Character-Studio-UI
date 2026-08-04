"""Official Tinker spend, read from the console billing API.

Everything else in this package *estimates* cost: ``tinker_client.estimate_tinker_cost``
multiplies a planned token count by the pinned rate card in :mod:`octt.models`.
That is the right instrument for a pre-spend gate, but it cannot tell you what a
run actually cost — token counts drift, prefill caches hit, retries re-sample,
and checkpoint storage accrues after the run exits.

This module reads the *invoiced* numbers instead, from the two endpoints that
back https://tinker.thinkingmachines.ai/usage and /billing/balance:

  ``GET /api/v1/billing/invoices/breakdowns?starting_on=&ending_before=&window_size=``
      Metronome invoice breakdown. One entry per window (``DAY`` or ``HOUR``),
      each carrying ``line_items[]`` keyed by charge type and
      ``pricing_group_values.base_model``. ``quantity`` is in millions of tokens
      (GB-months for storage), ``unit_price`` and ``total`` are in USD *cents*.

  ``GET /api/v1/billing/credits``
      Credit grants (research grant, promo) with amount and validity window.

Two things about these endpoints are load-bearing and worth stating plainly:

  - **They are not part of the Tinker SDK.** The SDK's ``/api/v1/*`` surface is
    train/sample/weights/telemetry only; its sole billing awareness is pausing
    on a 402. These live on the *console* host, which is a different origin from
    ``TINKER_BASE_URL``.
  - **``TINKER_API_KEY`` does not authenticate them.** Both ``Authorization:
    Bearer`` and ``X-Api-Key`` get a 302 to the WorkOS login. They accept only a
    browser session cookie, which is why ``TINKER_SESSION_COOKIE`` exists and
    why it expires. When it does, we raise rather than report stale numbers —
    see ``docs/COST_CONTROLS.md`` for the browser-snippet fallback.

``window_size=HOUR`` is what makes per-run attribution possible: run manifests
record ``created_at``/``updated_at``, so a run's hours can be intersected with
the hours Tinker billed. That join is *temporal, not causal* — if two runs
touched the same base model in the same hour, billing cannot separate them, and
:func:`attribute_run` says so rather than silently splitting the money.

Pure stdlib and side-effect-free on import, so this stays usable from the
dry-run tier and from machines without the training stack.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from octt import models

# --------------------------------------------------------------------------
# Endpoint / schema constants
# --------------------------------------------------------------------------

CONSOLE_BASE_URL = "https://tinker.thinkingmachines.ai"
BREAKDOWNS_PATH = "/api/v1/billing/invoices/breakdowns"
CREDITS_PATH = "/api/v1/billing/credits"

SESSION_COOKIE_ENV = "TINKER_SESSION_COOKIE"
CONSOLE_BASE_URL_ENV = "TINKER_CONSOLE_URL"

# Line-item names exactly as Metronome emits them.
CHARGE_TRAIN = "Tinker Training Token"
CHARGE_SAMPLE = "Tinker Sampler Sample"
CHARGE_PREFILL = "Tinker Sampler Prefill"
CHARGE_PREFILL_HIT = "Tinker Sampler Prefill Cache Hit"
CHARGE_PREFILL_MISS = "Tinker Sampler Prefill Cache Miss"
CHARGE_STORAGE = "Tinker Checkpoint Storage"

#: Charges whose ``quantity`` is millions of tokens.
TOKEN_CHARGES = frozenset(
    {CHARGE_TRAIN, CHARGE_SAMPLE, CHARGE_PREFILL, CHARGE_PREFILL_HIT, CHARGE_PREFILL_MISS}
)

#: Credit offsets arrive as negative line items named "<grant name> applied".
CREDIT_NAME_SUFFIX = " applied"

#: Fraction of the pinned prefill rate that a cached prefill token bills at.
#: Tinker splits prefill into Cache Hit / Cache Miss; ``models.price_prefill``
#: predates that split and equals the *miss* rate. Measured at exactly 0.2 on
#: all eight models billed in July 2026 (e.g. 27B: 37.2c hit vs 186c miss).
PREFILL_CACHE_HIT_MULTIPLIER = 0.2

#: Which pinned price in :mod:`octt.models` a billed charge should match, and
#: the multiplier to apply to it. Getting the multiplier wrong turns a correct
#: rate card into eight phantom drift reports.
_CHARGE_TO_PRICE_FIELD: dict[str, tuple[str, float]] = {
    CHARGE_TRAIN: ("price_train", 1.0),
    CHARGE_SAMPLE: ("price_sample", 1.0),
    CHARGE_PREFILL: ("price_prefill", 1.0),
    CHARGE_PREFILL_HIT: ("price_prefill", PREFILL_CACHE_HIT_MULTIPLIER),
    CHARGE_PREFILL_MISS: ("price_prefill", 1.0),
}

WINDOW_DAY = "DAY"
WINDOW_HOUR = "HOUR"


class BillingAuthError(RuntimeError):
    """No usable session cookie, or the console rejected the one we sent."""


class BillingFetchError(RuntimeError):
    """The console answered, but not with a billing payload we understand."""


# --------------------------------------------------------------------------
# Row model
# --------------------------------------------------------------------------


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class UsageRow:
    """One billed line item inside one breakdown window."""

    window_start: datetime | None
    window_end: datetime | None
    charge: str
    base_model: str | None
    region: str | None
    quantity: float
    unit_price_cents: float
    total_cents: float

    @property
    def total_usd(self) -> float:
        return self.total_cents / 100.0

    @property
    def unit_price_usd(self) -> float:
        """Billed rate in USD per 1M tokens (per GB-month for storage)."""
        return self.unit_price_cents / 100.0

    @property
    def is_credit(self) -> bool:
        """True for grant/promo offsets, which carry a negative ``total``."""
        return self.charge.endswith(CREDIT_NAME_SUFFIX)

    @property
    def token_millions(self) -> float:
        return self.quantity if self.charge in TOKEN_CHARGES else 0.0

    @property
    def day(self) -> str:
        return self.window_start.strftime("%Y-%m-%d") if self.window_start else "unknown"


@dataclass(frozen=True)
class CreditGrant:
    """A credit grant as shown on the Billing → Credit grants tab."""

    product: str
    amount_usd: float
    starting_at: datetime | None
    ending_before: datetime | None
    priority: int | None = None

    def is_active(self, at: datetime) -> bool:
        if self.starting_at and at < self.starting_at:
            return False
        return not (self.ending_before and at >= self.ending_before)


def parse_breakdowns(payload: Mapping[str, Any]) -> list[UsageRow]:
    """Flatten a ``/billing/invoices/breakdowns`` payload into :class:`UsageRow`.

    Window bounds come from the *invoice* entry (``breakdown_start_timestamp``),
    not the line item — a line item's ``ending_before`` is the enclosing
    invoice's end, not its own window, and using it collapses every row onto the
    same timestamp.
    """
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise BillingFetchError(
            "breakdowns payload has no 'data' list; got keys "
            f"{sorted(payload)!r} — the session cookie may have expired"
        )

    rows: list[UsageRow] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        start = _parse_ts(entry.get("breakdown_start_timestamp")) or _parse_ts(
            entry.get("start_timestamp")
        )
        end = _parse_ts(entry.get("breakdown_end_timestamp")) or _parse_ts(
            entry.get("end_timestamp")
        )
        for item in entry.get("line_items") or ():
            if not isinstance(item, Mapping):
                continue
            quantity = float(item.get("quantity") or 0.0)
            total = float(item.get("total") or 0.0)
            if quantity == 0.0 and total == 0.0:
                continue
            groups = item.get("pricing_group_values") or {}
            rows.append(
                UsageRow(
                    window_start=start,
                    window_end=end,
                    charge=str(item.get("name") or "unknown"),
                    base_model=groups.get("base_model"),
                    region=groups.get("region"),
                    quantity=quantity,
                    unit_price_cents=float(item.get("unit_price") or 0.0),
                    total_cents=total,
                )
            )
    return rows


def parse_credits(payload: Mapping[str, Any]) -> list[CreditGrant]:
    """Flatten a ``/billing/credits`` payload into :class:`CreditGrant`."""
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise BillingFetchError("credits payload has no 'data' list")

    grants: list[CreditGrant] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        product = (entry.get("product") or {}).get("name") or entry.get("name") or "credit"
        schedule = (entry.get("access_schedule") or {}).get("schedule_items") or []
        for item in schedule:
            if not isinstance(item, Mapping):
                continue
            grants.append(
                CreditGrant(
                    product=str(product),
                    amount_usd=float(item.get("amount") or 0.0) / 100.0,
                    starting_at=_parse_ts(item.get("starting_at")),
                    ending_before=_parse_ts(item.get("ending_before")),
                    priority=entry.get("priority"),
                )
            )
    return grants


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def session_cookie_header(raw: str | None = None) -> str:
    """Return a ``Cookie:`` header value, or raise :class:`BillingAuthError`.

    Accepts either a full header (``wos-session=abc; other=def``, what DevTools
    shows) or a bare session value, which we wrap in the AuthKit default cookie
    name. Prefer pasting the full header: the cookie name is set by the console,
    not by us, and could change.
    """
    value = (raw if raw is not None else os.environ.get(SESSION_COOKIE_ENV, "")).strip()
    if not value:
        raise BillingAuthError(
            f"{SESSION_COOKIE_ENV} is not set. The Tinker billing endpoints do not "
            "accept TINKER_API_KEY — they need a browser session cookie. Copy the "
            "'cookie' request header from any tinker.thinkingmachines.ai request in "
            "DevTools into .env, or use the browser snippet (octt spend --snippet)."
        )
    if "=" not in value:
        value = f"wos-session={value}"
    return value


def _console_base_url() -> str:
    return os.environ.get(CONSOLE_BASE_URL_ENV, CONSOLE_BASE_URL).rstrip("/")


def _get_json(path: str, params: Mapping[str, str] | None, cookie: str, timeout: float) -> Any:
    url = _console_base_url() + path
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Cookie": cookie,
            "Accept": "application/json",
            "User-Agent": "octt-billing/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # A 302 to the WorkOS login means the cookie is dead. urllib follows
            # redirects by default, so we'd otherwise "succeed" into an HTML
            # login page and parse it as an empty billing period.
            final_url = response.geturl()
            if "workos.com" in final_url or "/callback" in final_url:
                raise BillingAuthError(
                    f"{SESSION_COOKIE_ENV} was rejected (redirected to login). "
                    "The session has expired — re-copy the cookie from DevTools."
                )
            body = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        if exc.code in (401, 403):
            raise BillingAuthError(
                f"{SESSION_COOKIE_ENV} was rejected with HTTP {exc.code}. "
                "The session has expired — re-copy the cookie from DevTools."
            ) from exc
        raise BillingFetchError(f"GET {path} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise BillingFetchError(f"GET {path} failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise BillingFetchError(
            f"GET {path} did not return JSON — the session cookie is probably expired"
        ) from exc


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class BillingClient:
    """Thin, read-only client for the two console billing endpoints."""

    def __init__(self, cookie: str | None = None, timeout: float = 30.0) -> None:
        self._cookie = session_cookie_header(cookie)
        self._timeout = timeout

    def fetch_breakdowns(
        self, start: datetime, end: datetime, window_size: str = WINDOW_DAY
    ) -> list[UsageRow]:
        """Rows for ``[start, end)``.

        Hourly windows are requested one UTC day at a time: the console serves
        ``window_size=HOUR`` happily for a day, and chunking keeps a long
        per-run query from depending on an undocumented server-side cap.
        """
        if window_size == WINDOW_HOUR:
            rows: list[UsageRow] = []
            for chunk_start, chunk_end in _day_chunks(start, end):
                rows.extend(self._fetch_window(chunk_start, chunk_end, window_size))
            return rows
        return self._fetch_window(start, end, window_size)

    def _fetch_window(self, start: datetime, end: datetime, window_size: str) -> list[UsageRow]:
        payload = _get_json(
            BREAKDOWNS_PATH,
            {
                "starting_on": _iso_z(start),
                "ending_before": _iso_z(end),
                "window_size": window_size,
            },
            self._cookie,
            self._timeout,
        )
        return parse_breakdowns(payload)

    def fetch_credits(self) -> list[CreditGrant]:
        return parse_credits(_get_json(CREDITS_PATH, None, self._cookie, self._timeout))


def _day_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    end_utc = end.astimezone(UTC)
    while cursor < end_utc:
        nxt = cursor + timedelta(days=1)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


# --------------------------------------------------------------------------
# Snapshots (browser fallback)
# --------------------------------------------------------------------------

#: Paste into the DevTools console on tinker.thinkingmachines.ai. Downloads the
#: same two payloads the cookie path fetches, for machines with no live session.
BROWSER_SNIPPET = """\
// Run in the DevTools console on https://tinker.thinkingmachines.ai/usage
// Downloads a snapshot that `octt spend --snapshot <file>` can read offline.
(async (startISO, endISO, windowSize) => {
  const q = new URLSearchParams({
    starting_on: startISO, ending_before: endISO, window_size: windowSize,
  });
  const get = async (p) => (await fetch(p, {credentials: 'include'})).json();
  const snapshot = {
    fetched_at: new Date().toISOString(),
    window_size: windowSize,
    breakdowns: await get('/api/v1/billing/invoices/breakdowns?' + q),
    credits: await get('/api/v1/billing/credits'),
  };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(snapshot)], {type: 'application/json'}));
  a.download = 'tinker-spend-snapshot.json';
  document.body.appendChild(a); a.click(); a.remove();
})('%(start)s', '%(end)s', '%(window)s');
"""


def browser_snippet(start: datetime, end: datetime, window_size: str = WINDOW_DAY) -> str:
    return BROWSER_SNIPPET % {
        "start": _iso_z(start),
        "end": _iso_z(end),
        "window": window_size,
    }


def load_snapshot(path: Path) -> tuple[list[UsageRow], list[CreditGrant]]:
    """Read a snapshot written by :data:`BROWSER_SNIPPET`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    breakdowns = payload.get("breakdowns")
    if breakdowns is None:
        raise BillingFetchError(f"{path} has no 'breakdowns' key — not an octt spend snapshot")
    credits_payload = payload.get("credits") or {"data": []}
    return parse_breakdowns(breakdowns), parse_credits(credits_payload)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpendSummary:
    """Totals over a set of rows, with credits held separate from gross spend."""

    rows: tuple[UsageRow, ...]

    @property
    def charged_rows(self) -> tuple[UsageRow, ...]:
        return tuple(r for r in self.rows if not r.is_credit)

    @property
    def gross_usd(self) -> float:
        """What the usage was worth at list price, before grants."""
        return sum(r.total_usd for r in self.charged_rows)

    @property
    def credits_usd(self) -> float:
        """Grant/promo offsets applied (negative)."""
        return sum(r.total_usd for r in self.rows if r.is_credit)

    @property
    def net_usd(self) -> float:
        """Out-of-pocket after grants — what an invoice would actually charge."""
        return self.gross_usd + self.credits_usd

    @property
    def token_millions(self) -> float:
        return sum(r.token_millions for r in self.charged_rows)

    @property
    def storage_gb_months(self) -> float:
        return sum(r.quantity for r in self.charged_rows if r.charge == CHARGE_STORAGE)

    def by(self, *keys: str) -> dict[tuple[Any, ...], SpendSummary]:
        """Group charged rows by attribute names, e.g. ``by("base_model", "charge")``."""
        buckets: dict[tuple[Any, ...], list[UsageRow]] = {}
        for row in self.charged_rows:
            buckets.setdefault(tuple(getattr(row, k) for k in keys), []).append(row)
        return {k: SpendSummary(tuple(v)) for k, v in buckets.items()}

    def ranked(self, *keys: str) -> list[tuple[tuple[Any, ...], SpendSummary]]:
        """:meth:`by`, sorted most-expensive first."""
        return sorted(self.by(*keys).items(), key=lambda kv: -kv[1].gross_usd)


def summarize(rows: Iterable[UsageRow]) -> SpendSummary:
    return SpendSummary(tuple(rows))


@dataclass(frozen=True)
class GrantBalance:
    """How much grant credit is left, and whether we could actually tell."""

    granted_usd: float
    consumed_usd: float
    #: False when ``consumed_usd`` only covers the reporting window rather than
    #: the whole grant period — then ``remaining_usd`` is an *upper bound*.
    complete: bool

    @property
    def remaining_usd(self) -> float:
        return self.granted_usd - self.consumed_usd


def grant_period_start(grants: Sequence[CreditGrant], now: datetime) -> datetime | None:
    """Earliest start among grants active at *now*."""
    starts = [g.starting_at for g in grants if g.is_active(now) and g.starting_at]
    return min(starts) if starts else None


def grant_balance(
    grants: Sequence[CreditGrant], consumption_rows: Iterable[UsageRow], *, complete: bool
) -> GrantBalance:
    """Grant credit remaining, given rows covering the consumption period.

    The subtlety that makes this worth its own function: credit is consumed over
    the *grant's* lifetime, not over whatever month you happen to be reporting
    on. Subtracting only the reporting window's credits overstates the balance —
    for a July-only window it read $4,646 against a true $4,389, because ~$233
    had been spent before July. Callers that cannot cover the full period must
    pass ``complete=False`` so the number is labelled as an upper bound.
    """
    granted = sum(g.amount_usd for g in grants)
    consumed = -sum(r.total_usd for r in consumption_rows if r.is_credit)
    return GrantBalance(granted_usd=granted, consumed_usd=consumed, complete=complete)


# --------------------------------------------------------------------------
# Rate-card drift
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceDrift:
    base_model: str
    charge: str
    billed_usd_per_mtok: float
    pinned_usd_per_mtok: float

    @property
    def delta_pct(self) -> float:
        if self.pinned_usd_per_mtok == 0:
            return float("inf")
        return (
            100.0
            * (self.billed_usd_per_mtok - self.pinned_usd_per_mtok)
            / self.pinned_usd_per_mtok
        )

    @property
    def underestimates(self) -> bool:
        """True when Tinker charges *more* than the rate card predicts.

        This is the only direction that can hurt: ``preflight`` would clear a
        budget the run then blows through. The opposite direction (pinned above
        billed) makes the gate conservative, which ``octt/models.py``
        deliberately does for models whose post-increase prices were unpublished.
        """
        return self.billed_usd_per_mtok > self.pinned_usd_per_mtok


def price_drift(rows: Iterable[UsageRow], tolerance_pct: float = 1.0) -> list[PriceDrift]:
    """Compare billed unit prices against the pinned rate card in :mod:`octt.models`.

    ``octt preflight`` gates spend on those pinned numbers, so if Tinker changes
    a rate the gate silently starts lying. This is the check that catches it.

    Two things this deliberately does *not* flag:

      - Charges with no pinned counterpart (checkpoint storage, credit offsets)
        and $0 legacy prefill lines — they have nothing to compare against.
      - Cached prefill billed at :data:`PREFILL_CACHE_HIT_MULTIPLIER` of the
        pinned prefill rate, which is the real price, not drift.
    """
    seen: set[tuple[str, str]] = set()
    drifts: list[PriceDrift] = []
    for row in rows:
        mapping = _CHARGE_TO_PRICE_FIELD.get(row.charge)
        if mapping is None or not row.base_model or row.unit_price_cents <= 0:
            continue
        field, multiplier = mapping
        spec = models.CANDIDATES.get(row.base_model)
        if spec is None:
            continue
        pinned_base = getattr(spec, field, None)
        if pinned_base is None:
            continue
        key = (row.base_model, row.charge)
        if key in seen:
            continue
        seen.add(key)
        pinned = float(pinned_base) * multiplier
        billed = row.unit_price_usd
        if pinned == 0 or abs(billed - pinned) / pinned * 100.0 > tolerance_pct:
            drifts.append(PriceDrift(row.base_model, row.charge, billed, pinned))
    return drifts


# --------------------------------------------------------------------------
# Per-run attribution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunWindow:
    """The time span and model set of a recorded run, read from its manifest."""

    run_dir: Path
    run_id: str
    persona: str | None
    model: str | None
    teacher: str | None
    execution_mode: str | None
    start: datetime
    end: datetime

    @property
    def base_models(self) -> frozenset[str]:
        """Models whose usage this run could plausibly have caused."""
        return frozenset(m for m in (self.model, self.teacher) if m)

    @property
    def is_real(self) -> bool:
        return self.execution_mode == "real"


def load_run_window(run_dir: Path) -> RunWindow:
    """Read ``<run_dir>/manifest.json`` into a :class:`RunWindow`.

    The window is ``created_at`` → the latest of ``updated_at`` and every stage's
    ``recorded_at``, snapped outward to hour boundaries so it lines up with what
    ``window_size=HOUR`` returns.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    created = manifest.get("created_at")
    if created is None:
        raise BillingFetchError(f"{manifest_path} has no created_at; cannot bound the run")

    stamps = [float(created), float(manifest.get("updated_at") or created)]
    for stage in (manifest.get("stages") or {}).values():
        if isinstance(stage, Mapping) and stage.get("recorded_at"):
            stamps.append(float(stage["recorded_at"]))

    start = datetime.fromtimestamp(min(stamps), tz=UTC).replace(
        minute=0, second=0, microsecond=0
    )
    end_raw = datetime.fromtimestamp(max(stamps), tz=UTC)
    end = end_raw.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    return RunWindow(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id") or run_dir.name),
        persona=manifest.get("persona"),
        model=manifest.get("model"),
        teacher=manifest.get("teacher"),
        execution_mode=manifest.get("execution_mode"),
        start=start,
        end=end,
    )


def discover_run_windows(runs_root: Path) -> list[RunWindow]:
    """Every run under ``runs_root`` that has a readable manifest."""
    found: list[RunWindow] = []
    for manifest_path in sorted(Path(runs_root).glob("*/manifest.json")):
        try:
            found.append(load_run_window(manifest_path.parent))
        except (BillingFetchError, ValueError, json.JSONDecodeError):
            continue
    return found


@dataclass(frozen=True)
class RunAttribution:
    """Billed usage that overlaps a run's window, plus what makes it ambiguous."""

    window: RunWindow
    summary: SpendSummary
    #: Other runs sharing both an hour and a base model with this one. Their
    #: spend is inside ``summary`` too — billing has no run_id to separate them.
    contended_runs: tuple[str, ...]
    #: Rows in the window from models this run never touched, hence excluded.
    excluded_models: tuple[str, ...]

    @property
    def is_exclusive(self) -> bool:
        return not self.contended_runs


def attribute_run(
    window: RunWindow,
    rows: Iterable[UsageRow],
    others: Sequence[RunWindow] = (),
) -> RunAttribution:
    """Intersect billed rows with a run's hours and models.

    This is a temporal join, not a causal one. Tinker bills per (hour, base
    model) with no run identifier, so when another run overlaps on both axes the
    two are genuinely inseparable — we report the contention rather than
    inventing a split.
    """
    rows = list(rows)
    mine = window.base_models
    in_window = [
        r
        for r in rows
        if not r.is_credit
        and r.window_start is not None
        and r.window_start < window.end
        and (r.window_end or r.window_start) > window.start
    ]

    attributed = [r for r in in_window if r.base_model in mine]
    excluded = sorted(
        {r.base_model for r in in_window if r.base_model and r.base_model not in mine}
    )

    contended = sorted(
        other.run_id
        for other in others
        if other.run_id != window.run_id
        and other.start < window.end
        and other.end > window.start
        and other.base_models & mine
    )

    return RunAttribution(
        window=window,
        summary=summarize(attributed),
        contended_runs=tuple(contended),
        excluded_models=tuple(excluded),
    )
