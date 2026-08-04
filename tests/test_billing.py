"""Tests for official-billing ingestion (`octt/billing.py`).

The payloads below are trimmed captures of real
``/api/v1/billing/invoices/breakdowns`` responses, so the field quirks they
encode are the API's, not a guess:

  - A line item's ``ending_before`` is the *enclosing invoice's* end
    (2026-08-01), not its own window. Only ``breakdown_start_timestamp`` on the
    invoice entry identifies the window; trusting the line item collapses every
    hour onto one timestamp.
  - Credit offsets ("… applied") carry ``quantity: null`` and
    ``unit_price: null`` with a negative ``total``.
  - ``Tinker Sampler Prefill`` can bill at ``unit_price: 0`` — superseded by the
    Cache Hit/Miss split — so a zero rate is not a parse failure.
  - ``total`` and ``unit_price`` are USD *cents*; ``quantity`` is millions of
    tokens, except for storage where it is GB-months.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from octt import billing

# --------------------------------------------------------------------------
# Fixtures (trimmed real payloads)
# --------------------------------------------------------------------------

INVOICE_START = "2026-07-01T00:00:00Z"
INVOICE_END = "2026-08-01T00:00:00Z"


def _window(start: str, end: str, line_items: list[dict]) -> dict:
    return {
        "breakdown_start_timestamp": start,
        "breakdown_end_timestamp": end,
        "start_timestamp": INVOICE_START,
        "end_timestamp": INVOICE_END,
        "status": "DRAFT",
        "total": 0,
        "line_items": line_items,
    }


def _item(name: str, quantity, unit_price, total, groups=None) -> dict:
    return {
        "name": name,
        "quantity": quantity,
        "unit_price": unit_price,
        "total": total,
        "pricing_group_values": groups,
        "product_type": "UsageProductListItem",
        "credit_type": {"name": "USD (cents)"},
        # Deliberately the invoice end, exactly as the API sends it.
        "ending_before": INVOICE_END,
    }


BREAKDOWNS = {
    "data": [
        _window(
            "2026-07-25T03:00:00Z",
            "2026-07-25T04:00:00Z",
            [
                _item(
                    "Tinker Checkpoint Storage",
                    0.8035115329647073,
                    10,
                    8.035115329647073,
                    {"region": "GCP:us-south1"},
                ),
                _item("Tinker Research Grant applied", None, None, -8.035115329647073, None),
            ],
        ),
        _window(
            "2026-07-25T20:00:00Z",
            "2026-07-25T21:00:00Z",
            [
                # Free legacy prefill line — zero rate, real quantity.
                _item(
                    "Tinker Sampler Prefill",
                    0.027464,
                    0,
                    0,
                    {"base_model": "Qwen/Qwen3.5-4B"},
                ),
                # 2.0 Mtok train on 4B at the pinned 0.737 USD/Mtok = 73.7c.
                _item(
                    "Tinker Training Token",
                    2.0,
                    73.7,
                    147.4,
                    {"base_model": "Qwen/Qwen3.5-4B"},
                ),
                # 1.0 Mtok sample on 9B at the pinned 1.995 USD/Mtok = 199.5c.
                _item(
                    "Tinker Sampler Sample",
                    1.0,
                    199.5,
                    199.5,
                    {"base_model": "Qwen/Qwen3.5-9B"},
                ),
                _item("Tinker Research Grant applied", None, None, -346.9, None),
                # Zero-quantity, zero-total noise the API emits; must be dropped.
                _item("Tinker Sampler Sample", 0, 0, 0, {"base_model": "Qwen/Qwen3.6-27B"}),
            ],
        ),
    ]
}

CREDITS = {
    "data": [
        {
            "product": {"name": "Tinker Research Grant"},
            "type": "CREDIT",
            "priority": 50,
            "access_schedule": {
                "schedule_items": [
                    {
                        "amount": 500000,
                        "starting_at": "2026-05-21T03:00:00Z",
                        "ending_before": "2027-05-21T03:00:00Z",
                    }
                ],
                "credit_type": {"name": "USD (cents)"},
            },
        },
        {
            "product": {"name": "Tinker Promotion"},
            "type": "CREDIT",
            "priority": 50,
            "access_schedule": {
                "schedule_items": [
                    {
                        "amount": 15000,
                        "starting_at": "2025-11-18T03:00:00Z",
                        "ending_before": "2026-11-18T03:00:00Z",
                    }
                ],
                "credit_type": {"name": "USD (cents)"},
            },
        },
    ]
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_window_timestamps_come_from_the_invoice_not_the_line_item():
    rows = billing.parse_breakdowns(BREAKDOWNS)
    starts = {row.window_start for row in rows}
    assert starts == {
        datetime(2026, 7, 25, 3, tzinfo=UTC),
        datetime(2026, 7, 25, 20, tzinfo=UTC),
    }
    # If we had trusted line_items[].ending_before every row would land here.
    assert all(row.window_end != datetime(2026, 8, 1, tzinfo=UTC) for row in rows)


def test_zero_rows_dropped_and_null_credit_fields_survive():
    rows = billing.parse_breakdowns(BREAKDOWNS)
    # 2 storage-hour rows + 3 real charges + 1 credit in hour 20 == 6.
    assert len(rows) == 6
    assert not any(r.quantity == 0 and r.total_cents == 0 for r in rows)
    credits = [r for r in rows if r.is_credit]
    assert len(credits) == 2
    assert all(r.quantity == 0.0 for r in credits)
    assert sum(r.total_cents for r in credits) == pytest.approx(-354.935115, rel=1e-6)


def test_cents_to_usd_and_token_quantities():
    rows = billing.parse_breakdowns(BREAKDOWNS)
    train = next(r for r in rows if r.charge == billing.CHARGE_TRAIN)
    assert train.total_usd == pytest.approx(1.474)
    assert train.unit_price_usd == pytest.approx(0.737)
    assert train.token_millions == pytest.approx(2.0)

    storage = next(r for r in rows if r.charge == billing.CHARGE_STORAGE)
    # Storage quantity is GB-months, so it must not count as tokens.
    assert storage.token_millions == 0.0
    assert storage.region == "GCP:us-south1"
    assert storage.base_model is None


def test_parse_rejects_a_login_page_instead_of_reporting_zero_spend():
    with pytest.raises(billing.BillingFetchError):
        billing.parse_breakdowns({"detail": "Not Found"})


def test_parse_credits():
    grants = billing.parse_credits(CREDITS)
    assert {g.product for g in grants} == {"Tinker Research Grant", "Tinker Promotion"}
    research = next(g for g in grants if g.product == "Tinker Research Grant")
    assert research.amount_usd == pytest.approx(5000.0)
    assert research.is_active(datetime(2026, 7, 26, tzinfo=UTC))
    assert not research.is_active(datetime(2027, 6, 1, tzinfo=UTC))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_summary_keeps_gross_credits_and_net_distinct():
    summary = billing.summarize(billing.parse_breakdowns(BREAKDOWNS))
    assert summary.gross_usd == pytest.approx(3.549351, rel=1e-6)
    assert summary.credits_usd == pytest.approx(-3.549351, rel=1e-6)
    # Fully grant-covered: worth $3.55 of usage, $0.00 out of pocket.
    assert summary.net_usd == pytest.approx(0.0, abs=1e-9)
    assert summary.token_millions == pytest.approx(3.027464)
    assert summary.storage_gb_months == pytest.approx(0.8035115, rel=1e-6)


def test_ranked_grouping_is_most_expensive_first_and_excludes_credits():
    summary = billing.summarize(billing.parse_breakdowns(BREAKDOWNS))
    ranked = summary.ranked("base_model", "charge")
    assert ranked[0][0] == ("Qwen/Qwen3.5-9B", billing.CHARGE_SAMPLE)
    assert ranked[0][1].gross_usd == pytest.approx(1.995)
    assert all(not key[1].endswith(" applied") for key, _ in ranked)


# --------------------------------------------------------------------------
# Grant balance
# --------------------------------------------------------------------------


def _credit_row(day: str, usd: float) -> billing.UsageRow:
    moment = datetime.fromisoformat(day).replace(tzinfo=UTC)
    return billing.UsageRow(
        moment, moment + timedelta(days=1), "Tinker Research Grant applied",
        None, None, 0.0, 0.0, -usd * 100,
    )


def test_grant_remaining_must_span_the_grant_not_the_report_window():
    grants = billing.parse_credits(CREDITS)
    # $233 burned before July, $503.91 during it — the real July 2026 shape.
    before_july = [_credit_row("2026-06-15", 233.00)]
    during_july = [_credit_row("2026-07-26", 503.91)]

    windowed = billing.grant_balance(grants, during_july, complete=False)
    full = billing.grant_balance(grants, before_july + during_july, complete=True)

    # This is the bug: reporting on July alone overstated the balance by the
    # $233 spent earlier, reading $4,646 against a true $4,389.
    assert windowed.remaining_usd == pytest.approx(4646.09, abs=0.01)
    assert full.remaining_usd == pytest.approx(4413.09, abs=0.01)
    assert windowed.remaining_usd > full.remaining_usd
    # An incomplete balance is an upper bound and must be flagged as one.
    assert not windowed.complete
    assert full.complete


def test_grant_period_start_is_the_earliest_active_grant():
    grants = billing.parse_credits(CREDITS)
    now = datetime(2026, 7, 26, tzinfo=UTC)
    # Promotion starts 2025-11-18, research grant 2026-05-21.
    assert billing.grant_period_start(grants, now) == datetime(2025, 11, 18, 3, tzinfo=UTC)
    # Once the promotion lapses, the window starts at the research grant.
    later = datetime(2026, 12, 1, tzinfo=UTC)
    active = [g for g in grants if g.is_active(later)]
    assert billing.grant_period_start(active, later) == datetime(2026, 5, 21, 3, tzinfo=UTC)


def test_grant_balance_ignores_charge_rows():
    grants = billing.parse_credits(CREDITS)
    rows = billing.parse_breakdowns(BREAKDOWNS)
    balance = billing.grant_balance(grants, rows, complete=True)
    # Only the negative "applied" lines count as consumption, not gross charges.
    assert balance.consumed_usd == pytest.approx(3.549351, rel=1e-6)
    assert balance.granted_usd == pytest.approx(5150.0)


def test_cli_snapshot_labels_the_balance_as_an_upper_bound(tmp_path, capsys):
    from octt import cli

    cli.main(["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07"])
    out = capsys.readouterr().out
    assert "upper bound" in out


def test_cli_credit_floor_blocks_even_on_an_upper_bound(tmp_path, capsys):
    from octt import cli

    # If even the optimistic figure is under the floor, the real one is too.
    args = ["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07"]
    assert cli.main([*args, "--min-credit-usd", "6000"]) == 2
    assert "no higher" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Rate-card drift
# --------------------------------------------------------------------------


def test_no_drift_when_billed_rates_match_the_pinned_rate_card():
    # 73.7c/Mtok train on 4B and 199.5c sample on 9B are exactly what
    # octt/models.py pins, which is the state we want to stay in.
    assert billing.price_drift(billing.parse_breakdowns(BREAKDOWNS)) == []


def test_drift_detected_when_tinker_changes_a_rate():
    bumped = json.loads(json.dumps(BREAKDOWNS))
    for item in bumped["data"][1]["line_items"]:
        if item["name"] == billing.CHARGE_TRAIN:
            item["unit_price"] = 90.0  # $0.90/Mtok vs the pinned $0.737
    drifts = billing.price_drift(billing.parse_breakdowns(bumped))
    assert len(drifts) == 1
    assert drifts[0].base_model == "Qwen/Qwen3.5-4B"
    assert drifts[0].billed_usd_per_mtok == pytest.approx(0.90)
    assert drifts[0].pinned_usd_per_mtok == pytest.approx(0.737)
    assert drifts[0].delta_pct > 20


#: Every distinct (base_model, charge, unit_price-in-cents) Tinker actually billed
#: in July 2026, transcribed from a live breakdowns response. Kept as a literal
#: because it is the ground truth ``octt preflight``'s rate card is judged against.
REAL_JULY_2026_RATES = [
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", billing.CHARGE_SAMPLE, 49.5),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", billing.CHARGE_PREFILL_HIT, 3.9),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", billing.CHARGE_PREFILL_MISS, 19.5),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", billing.CHARGE_SAMPLE, 622.5),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", billing.CHARGE_PREFILL_HIT, 49.8),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", billing.CHARGE_PREFILL_MISS, 249),
    ("Qwen/Qwen3.5-397B-A17B", billing.CHARGE_SAMPLE, 750),
    ("Qwen/Qwen3.5-397B-A17B", billing.CHARGE_PREFILL_HIT, 60),
    ("Qwen/Qwen3.5-397B-A17B", billing.CHARGE_PREFILL_MISS, 300),
    ("Qwen/Qwen3.5-4B", billing.CHARGE_SAMPLE, 100.5),
    ("Qwen/Qwen3.5-4B", billing.CHARGE_TRAIN, 73.7),
    ("Qwen/Qwen3.5-4B", billing.CHARGE_PREFILL_HIT, 6.6),
    ("Qwen/Qwen3.5-4B", billing.CHARGE_PREFILL_MISS, 33),
    ("Qwen/Qwen3.5-9B", billing.CHARGE_SAMPLE, 199.5),
    ("Qwen/Qwen3.5-9B", billing.CHARGE_TRAIN, 146.3),
    ("Qwen/Qwen3.5-9B", billing.CHARGE_PREFILL_HIT, 13.2),
    ("Qwen/Qwen3.5-9B", billing.CHARGE_PREFILL_MISS, 66),
    ("Qwen/Qwen3.6-27B", billing.CHARGE_SAMPLE, 559.5),
    ("Qwen/Qwen3.6-27B", billing.CHARGE_TRAIN, 410.3),
    ("Qwen/Qwen3.6-27B", billing.CHARGE_PREFILL_HIT, 37.2),
    ("Qwen/Qwen3.6-27B", billing.CHARGE_PREFILL_MISS, 186),
    ("Qwen/Qwen3.6-35B-A3B", billing.CHARGE_SAMPLE, 133.5),
    ("Qwen/Qwen3.6-35B-A3B", billing.CHARGE_TRAIN, 117.7),
    ("Qwen/Qwen3.6-35B-A3B", billing.CHARGE_PREFILL_HIT, 10.8),
    ("Qwen/Qwen3.6-35B-A3B", billing.CHARGE_PREFILL_MISS, 54),
    ("thinkingmachines/Inkling", billing.CHARGE_SAMPLE, 468),
    ("thinkingmachines/Inkling", billing.CHARGE_PREFILL_HIT, 37.4),
    ("thinkingmachines/Inkling", billing.CHARGE_PREFILL_MISS, 187),
]


def _rate_rows():
    return [
        billing.UsageRow(None, None, charge, model, None, 1.0, cents, cents)
        for model, charge, cents in REAL_JULY_2026_RATES
    ]


def test_cached_prefill_bills_at_one_fifth_of_the_miss_rate():
    # models.price_prefill predates the Cache Hit/Miss split and equals the miss
    # rate. If the hit multiplier is wrong, every model reports phantom drift.
    by_model: dict[str, dict[str, float]] = {}
    for model, charge, cents in REAL_JULY_2026_RATES:
        by_model.setdefault(model, {})[charge] = cents
    ratios = {
        model: rates[billing.CHARGE_PREFILL_HIT] / rates[billing.CHARGE_PREFILL_MISS]
        for model, rates in by_model.items()
    }
    assert len(ratios) == 8
    for model, ratio in ratios.items():
        assert ratio == pytest.approx(billing.PREFILL_CACHE_HIT_MULTIPLIER, rel=1e-6), model


def test_pinned_rate_card_matches_every_real_billed_rate_except_promo_models():
    drifts = billing.price_drift(_rate_rows())
    # 22 of the 28 July rates match models.py exactly. Two exceptions, both
    # deliberate (2026-07-30 refresh): Inkling is pinned at its now-published
    # undiscounted list rate but July billed the limited-time 50% promo, and
    # Nano is pinned at the full rate its ended promo moved it to while July
    # still billed the old half rate.
    assert {d.base_model for d in drifts} == {
        "thinkingmachines/Inkling",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    }
    assert len(drifts) == 6


def test_promo_drift_is_conservative_so_the_budget_gate_still_holds():
    drifts = billing.price_drift(_rate_rows())
    # Billed below pinned => preflight over-estimates => the gate is safe.
    assert all(not d.underestimates for d in drifts)
    # Both exceptions billed exactly half the pinned list rate in July.
    assert all(d.delta_pct == pytest.approx(-50.0, abs=0.3) for d in drifts)


def test_a_rate_rise_above_the_pinned_card_is_flagged_as_dangerous():
    rows = [billing.UsageRow(None, None, billing.CHARGE_TRAIN, "Qwen/Qwen3.5-4B", None, 1.0, 90.0, 90.0)]
    drift = billing.price_drift(rows)[0]
    assert drift.underestimates


def test_zero_rate_and_storage_lines_are_not_drift():
    # A $0 prefill line and a GB-month storage line have no pinned counterpart;
    # flagging them would make --check-prices permanently red.
    rows = billing.parse_breakdowns(BREAKDOWNS)
    flagged = {(d.base_model, d.charge) for d in billing.price_drift(rows)}
    assert ("Qwen/Qwen3.5-4B", billing.CHARGE_PREFILL) not in flagged
    assert not any(charge == billing.CHARGE_STORAGE for _, charge in flagged)


# --------------------------------------------------------------------------
# Per-run attribution
# --------------------------------------------------------------------------


def _write_manifest(tmp_path, name, *, model, teacher, start_hour, end_hour, mode="real"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    base = datetime(2026, 7, 25, tzinfo=UTC)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "persona": "pirate",
                "model": model,
                "teacher": teacher,
                "execution_mode": mode,
                "created_at": (base + timedelta(hours=start_hour)).timestamp(),
                "updated_at": (base + timedelta(hours=end_hour)).timestamp(),
                "stages": {"dpo": {"recorded_at": (base + timedelta(hours=end_hour)).timestamp()}},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_run_window_snaps_outward_to_hour_boundaries(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r",
                "model": "Qwen/Qwen3.5-4B",
                "created_at": datetime(2026, 7, 25, 20, 31, tzinfo=UTC).timestamp(),
                "updated_at": datetime(2026, 7, 25, 21, 14, tzinfo=UTC).timestamp(),
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    window = billing.load_run_window(run_dir)
    # Hourly billing buckets are whole hours, so a 20:31–21:14 run must claim
    # 20:00–22:00 or its own spend falls outside the query.
    assert window.start == datetime(2026, 7, 25, 20, tzinfo=UTC)
    assert window.end == datetime(2026, 7, 25, 22, tzinfo=UTC)


def test_attribution_filters_by_both_hour_and_model(tmp_path):
    run_dir = _write_manifest(
        tmp_path,
        "pirate-4b",
        model="Qwen/Qwen3.5-4B",
        teacher="Qwen/Qwen3.5-4B",
        start_hour=20,
        end_hour=20,
    )
    rows = billing.parse_breakdowns(BREAKDOWNS)
    attribution = billing.attribute_run(billing.load_run_window(run_dir), rows)

    # Only the 4B rows in hour 20: the $1.474 train line. The 9B sample is a
    # different model, the 03:00 storage row a different hour.
    assert attribution.summary.gross_usd == pytest.approx(1.474)
    assert attribution.excluded_models == ("Qwen/Qwen3.5-9B",)
    assert attribution.is_exclusive


def test_overlapping_runs_are_reported_not_silently_split(tmp_path):
    mine = _write_manifest(
        tmp_path, "run-a", model="Qwen/Qwen3.5-4B", teacher=None, start_hour=20, end_hour=20
    )
    _write_manifest(
        tmp_path, "run-b", model="Qwen/Qwen3.5-4B", teacher=None, start_hour=20, end_hour=20
    )
    _write_manifest(
        tmp_path, "run-c", model="Qwen/Qwen3.6-27B", teacher=None, start_hour=20, end_hour=20
    )

    known = billing.discover_run_windows(tmp_path)
    assert len(known) == 3
    rows = billing.parse_breakdowns(BREAKDOWNS)
    attribution = billing.attribute_run(billing.load_run_window(mine), rows, known)

    # run-b shares both the hour and the model, so the money is inseparable.
    # run-c shares the hour but not the model, so it is not contention.
    assert attribution.contended_runs == ("run-b",)
    assert not attribution.is_exclusive
    # The figure is still reported — as an upper bound, not a silent 50/50 split.
    assert attribution.summary.gross_usd == pytest.approx(1.474)


def test_dry_run_manifests_are_flagged_as_spending_nothing(tmp_path):
    run_dir = _write_manifest(
        tmp_path,
        "dry",
        model="Qwen/Qwen3.5-4B",
        teacher=None,
        start_hour=20,
        end_hour=20,
        mode="dry-run",
    )
    assert not billing.load_run_window(run_dir).is_real


# --------------------------------------------------------------------------
# Auth + snapshots
# --------------------------------------------------------------------------


def test_missing_cookie_raises_and_names_the_api_key_trap(monkeypatch):
    monkeypatch.delenv(billing.SESSION_COOKIE_ENV, raising=False)
    with pytest.raises(billing.BillingAuthError) as excinfo:
        billing.session_cookie_header()
    # The failure mode we most want to pre-empt: reaching for TINKER_API_KEY.
    assert "TINKER_API_KEY" in str(excinfo.value)


def test_bare_cookie_value_is_wrapped_full_header_passes_through(monkeypatch):
    monkeypatch.setenv(billing.SESSION_COOKIE_ENV, "abc123")
    assert billing.session_cookie_header() == "wos-session=abc123"
    monkeypatch.setenv(billing.SESSION_COOKIE_ENV, "wos-session=xyz; other=1")
    assert billing.session_cookie_header() == "wos-session=xyz; other=1"


def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"breakdowns": BREAKDOWNS, "credits": CREDITS}), encoding="utf-8")
    rows, grants = billing.load_snapshot(path)
    assert billing.summarize(rows).gross_usd == pytest.approx(3.549351, rel=1e-6)
    assert len(grants) == 2


def test_snapshot_rejects_unrelated_json(tmp_path):
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(billing.BillingFetchError):
        billing.load_snapshot(path)


def test_browser_snippet_embeds_the_requested_window():
    snippet = billing.browser_snippet(
        datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC), billing.WINDOW_DAY
    )
    assert "2026-07-01T00:00:00.000Z" in snippet
    assert "2026-08-01T00:00:00.000Z" in snippet
    assert billing.BREAKDOWNS_PATH in snippet


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def _snapshot(tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"breakdowns": BREAKDOWNS, "credits": CREDITS}), encoding="utf-8")
    return path


def test_cli_spend_reports_billed_totals(tmp_path, capsys):
    from octt import cli

    assert cli.main(["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07"]) == 0
    out = capsys.readouterr().out
    assert "gross billed" in out
    assert "3.55" in out
    assert "Tinker Research Grant" in out


def test_cli_spend_json_is_machine_readable(tmp_path, capsys):
    from octt import cli

    assert (
        cli.main(
            ["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["gross_usd"] == pytest.approx(3.5494, abs=1e-3)
    assert payload["net_usd"] == pytest.approx(0.0, abs=1e-3)
    assert payload["grant_remaining_usd"] == pytest.approx(5150.0 - 3.5494, abs=1e-2)
    assert payload["price_drift"] == []

    # Quantities must survive a zero-quantity row sorting first in the bucket.
    prefill = next(
        r for r in payload["by_model_charge"] if r["charge"] == billing.CHARGE_PREFILL
    )
    assert prefill["quantity"] == pytest.approx(0.027464)
    train = next(r for r in payload["by_model_charge"] if r["charge"] == billing.CHARGE_TRAIN)
    assert train["quantity"] == pytest.approx(2.0)


def test_cli_spend_snapshot_reports_the_span_it_actually_covers(tmp_path, capsys):
    from octt import cli

    # Asking for a month but handing over a two-hour snapshot must not print
    # the month — a stale file would otherwise masquerade as current data.
    cli.main(["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-01"])
    out = capsys.readouterr().out
    assert "spanned by the snapshot" in out
    assert "2026-07-25 03:00Z" in out
    assert "2026-01" not in out


def test_cli_spend_budget_gate_blocks(tmp_path, capsys):
    from octt import cli

    args = ["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07"]
    assert cli.main([*args, "--max-gross-usd", "1.00"]) == 2
    assert "BLOCKED" in capsys.readouterr().out
    assert cli.main([*args, "--max-gross-usd", "10.00"]) == 0


def test_cli_spend_credit_floor_gate_blocks(tmp_path, capsys):
    from octt import cli

    args = ["spend", "--snapshot", str(_snapshot(tmp_path)), "--month", "2026-07"]
    assert cli.main([*args, "--min-credit-usd", "10000"]) == 2
    assert "BLOCKED" in capsys.readouterr().out
    assert cli.main([*args, "--min-credit-usd", "100"]) == 0


def test_cli_spend_snippet_needs_no_cookie(monkeypatch, capsys):
    from octt import cli

    monkeypatch.delenv(billing.SESSION_COOKIE_ENV, raising=False)
    assert cli.main(["spend", "--snippet", "--month", "2026-07"]) == 0
    assert billing.CREDITS_PATH in capsys.readouterr().out


def test_cli_spend_without_cookie_fails_loudly(monkeypatch, capsys):
    from octt import cli

    monkeypatch.delenv(billing.SESSION_COOKIE_ENV, raising=False)
    # Must not silently report $0.00 spend when it cannot authenticate.
    assert cli.main(["spend", "--month", "2026-07"]) == 2
    out = capsys.readouterr().out
    assert "auth error" in out
    assert "0.00" not in out
