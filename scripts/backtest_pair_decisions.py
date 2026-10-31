"""Pair retention/exit/entry rules; directions only, with no shares or accounting."""

from __future__ import annotations

import math

if __package__:
    from .backtest_signal_policy import guard_existing_pairs
else:
    from backtest_signal_policy import guard_existing_pairs


def decide_pairs(active_pairs, predictions, instruction, *, h_in, tickers):
    """Generate final target pair directions from one signal-time instruction.

    +1 means long sector/short SPY; -1 means short sector/long SPY. h_in is an
    explicit decimal-return parameter; h_out=h_in/2. No threshold is selected.
    Only the final targets are returned, so a reversal need not be executed as
    two separately charged orders. Missing inputs are guarded before ranking.
    """
    if not math.isfinite(h_in) or h_in <= 0:
        raise ValueError("h_in must be a positive finite decimal return")
    if len(set(tickers)) != 9 or len(tickers) != 9:
        raise ValueError("Expected nine distinct sector tickers")
    if len(active_pairs) > 4 or any(t not in tickers or isinstance(d, bool)
                                   or d not in (-1, 1) for t, d in active_pairs.items()):
        raise ValueError("Old pairs must contain at most four known sectors with directions +/-1")
    actions = {"EVALUATE_SIGNAL", "CLOSE_ALL_TO_CASH", "FINAL_LIQUIDATION", "NO_ORDER_OUTSIDE_PHASE"}
    action = instruction["action"]
    if action not in actions:
        raise ValueError(f"Unknown signal instruction: {action}")
    force_flat = bool(instruction["force_flat"])
    if force_flat != (action in {"CLOSE_ALL_TO_CASH", "FINAL_LIQUIDATION"}):
        raise ValueError("Signal instruction action and force_flat disagree")
    old = dict(active_pairs)
    def numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")
    values = {ticker: numeric(predictions.get(ticker)) for ticker in tickers}
    targets = guard_existing_pairs(old, instruction)
    ranking = []
    retained = set()
    reason = action.lower()
    if action == "EVALUATE_SIGNAL":
        complete = (bool(instruction["signal_complete"]) and set(predictions) == set(tickers)
                    and all(math.isfinite(values[t]) for t in tickers))
        if not complete:
            action, targets, reason = "CLOSE_ALL_TO_CASH", {}, "incomplete_signal_guard"
        else:
            ranking = sorted(tickers, key=lambda t: (-abs(values[t]), tickers.index(t)))
            targets = {}
            for ticker in tickers:
                if ticker in old and old[ticker] * values[ticker] > h_in / 2:
                    targets[ticker] = old[ticker]
                    retained.add(ticker)
            for ticker in ranking[:4]:
                if len(targets) == 4:
                    break
                if ticker not in targets and abs(values[ticker]) > h_in:
                    targets[ticker] = 1 if values[ticker] > 0 else -1
            reason = "retain_then_top_four_entries"
    rows = []
    for ticker in tickers:
        previous, target = old.get(ticker, 0), targets.get(ticker, 0)
        value = values[ticker]
        rank = ranking.index(ticker) + 1 if ranking else None
        if action == "NO_ORDER_OUTSIDE_PHASE":
            decision, detail = "NO_ORDER", "outside_phase_order_schedule"
        elif action in {"CLOSE_ALL_TO_CASH", "FINAL_LIQUIDATION"}:
            decision = "EXIT" if previous else "STAY_FLAT"
            detail = reason
        elif ticker in retained:
            decision, detail = "RETAIN", "same_direction_strictly_above_exit_threshold"
        elif target:
            decision = "REVERSE" if previous else "ENTER"
            detail = "top_four_strictly_above_entry_threshold"
        else:
            decision = "EXIT" if previous else "STAY_FLAT"
            if rank > 4:
                detail = "outside_top_four_for_new_entry"
            elif abs(value) <= h_in:
                detail = "at_or_below_entry_threshold"
            else:
                detail = "capacity_filled_by_retained_or_higher_ranked_pairs"
        rows.append(dict(ticker=ticker, y_hat=value, abs_rank=rank,
                         previous_direction=previous, target_direction=target,
                         retained=ticker in retained, decision=decision, reason=detail))
    return dict(targets), rows, dict(
        effective_action=action, reason=reason, previous_pairs=len(old), target_pairs=len(targets),
        entered_pairs=sum(row["decision"] == "ENTER" for row in rows),
        exited_pairs=sum(row["decision"] == "EXIT" for row in rows),
        reversed_pairs=sum(row["decision"] == "REVERSE" for row in rows),
        retained_pairs=sum(row["decision"] == "RETAIN" for row in rows),
        order_scheduled=action != "NO_ORDER_OUTSIDE_PHASE")


def replay_pair_targets(signals, daily, *, h_in, tickers):
    """Replay logical target directions, resetting to flat for each model/scheme/phase.

    This is a decision preview assuming target instructions are carried out. The
    later execution engine should call decide_pairs with the actual current pair
    state. This function never computes weights, prices, fees, PnL or NAV.
    """
    import pandas as pd

    lookup = {key: dict(zip(group.ticker, group.y_hat)) for key, group in
              signals.groupby(["date", "model", "scheme"], sort=False)}
    daily_rows, sector_rows = [], []
    for (model, scheme, split), group in daily.groupby(["model", "scheme", "split"], sort=True):
        active = {}
        for instruction in group.sort_values("date").to_dict("records"):
            values = lookup.get((instruction["date"], model, scheme), {})
            target, details, summary = decide_pairs(active, values, instruction, h_in=h_in, tickers=tickers)
            metadata = {key: instruction[key] for key in ("date", "model", "scheme", "split", "signal_at", "action_at")}
            old_text = "|".join(f"{t}:{active[t]:+d}" for t in tickers if t in active)
            new_text = "|".join(f"{t}:{target[t]:+d}" for t in tickers if t in target)
            daily_rows.append(metadata | dict(input_action=instruction["action"], h_in=h_in, h_out=h_in / 2,
                                              previous_pair_directions=old_text, target_pair_directions=new_text) | summary)
            sector_rows.extend(metadata | row for row in details)
            active = target
    return pd.DataFrame(daily_rows), pd.DataFrame(sector_rows)


def validate_pair_decisions(tickers):
    """Hand-specified boundary/state cases, independent of realized returns."""
    import pandas as pd

    h = 0.001
    t = list(tickers)
    normal = dict(action="EVALUATE_SIGNAL", force_flat=False, signal_complete=True)
    records = []

    def case(name, old, values, expected, instruction=None):
        original = dict(old)
        predictions = dict.fromkeys(t, 0.0) | values
        result, rows, summary = decide_pairs(old, predictions, instruction or normal, h_in=h, tickers=tickers)
        passed = result == expected and old == original and len(result) <= 4
        records.append(dict(check=name, passed=passed, detail="Expected final pair directions and unchanged old state"))
        if not passed:
            raise AssertionError(f"{name}: expected {expected}, got {result}")
        return result, rows, summary

    case("weak_signals_remain_flat", {}, {t[0]: 0.0009}, {})
    case("zero_forecasts_do_not_create_tied_portfolio", {}, {}, {})
    case("strict_positive_entry_boundary", {}, {t[0]: h}, {})
    case("strict_negative_entry_boundary", {}, {t[0]: -h}, {})
    case("just_above_entry_boundary", {}, {t[0]: math.nextafter(h, math.inf)}, {t[0]: 1})
    case("long_retained_inside_entry_band", {t[0]: 1}, {t[0]: 0.0007}, {t[0]: 1})
    case("short_retained_inside_entry_band", {t[0]: -1}, {t[0]: -0.0007}, {t[0]: -1})
    case("strict_long_exit_boundary", {t[0]: 1}, {t[0]: h / 2}, {})
    case("strict_short_exit_boundary", {t[0]: -1}, {t[0]: -h / 2}, {})
    case("below_exit_threshold_closes", {t[0]: 1}, {t[0]: 0.0004}, {})
    case("mixed_directions_top_four", {}, dict(zip(t, [0.009, -0.008, 0.007, -0.006, 0.005, 0.004, 0.003, 0.002, 0.0011])),
         {t[0]: 1, t[1]: -1, t[2]: 1, t[3]: -1})
    strong = dict(zip(t, [0.009, -0.008, 0.007, -0.006, 0.005, 0.004, 0.003, 0.0008, -0.0007]))
    case("held_ninth_rank_not_forced_out", {t[8]: -1}, strong, {t[8]: -1, t[0]: 1, t[1]: -1, t[2]: 1})
    case("retained_pairs_consume_capacity", {t[7]: 1, t[8]: -1}, strong,
         {t[7]: 1, t[8]: -1, t[0]: 1, t[1]: -1})
    four_old = {ticker: 1 for ticker in t[5:9]}
    case("four_retained_not_replaced_by_stronger_new_signals", four_old,
         {ticker: (0.0007 if ticker in four_old else 0.01) for ticker in t}, four_old)
    case("eligible_reversal", {t[0]: 1}, {t[0]: -0.002}, {t[0]: -1})
    case("reverse_entry_boundary_requires_full_entry_rule", {t[0]: 1}, {t[0]: -h}, {})
    case("opposite_signal_below_entry_threshold_exits_only", {t[0]: 1}, {t[0]: -0.0007}, {})
    case("reverse_outside_top_four_exits_only", {t[8]: 1},
         dict(zip(t, [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, -0.0015])),
         {t[0]: 1, t[1]: 1, t[2]: 1, t[3]: 1})
    ties = {ticker: (0.002 if i % 2 == 0 else -0.002) for i, ticker in enumerate(t)}
    expected_ties = {ticker: (1 if ties[ticker] > 0 else -1) for ticker in t[:4]}
    case("ticker_order_breaks_absolute_value_ties", {}, ties, expected_ties)
    case("input_row_order_does_not_change_ties", {}, dict(reversed(list(ties.items()))), expected_ties)
    for action in ("CLOSE_ALL_TO_CASH", "FINAL_LIQUIDATION"):
        case(action.lower(), {t[0]: 1, t[1]: -1}, ties, {},
             dict(action=action, force_flat=True, signal_complete=False))
    case("outside_phase_creates_no_entry", {}, ties, {},
         dict(action="NO_ORDER_OUTSIDE_PHASE", force_flat=False, signal_complete=True))
    for name, values in (("missing_prediction_guard", {ticker: 0.002 for ticker in t[:-1]}),
                         ("nonfinite_prediction_guard", dict.fromkeys(t, float("nan")))):
        result, _, summary = decide_pairs({t[0]: 1}, values, normal, h_in=h, tickers=tickers)
        passed = result == {} and summary["effective_action"] == "CLOSE_ALL_TO_CASH"
        records.append(dict(check=name, passed=passed, detail="Incomplete signal closes all targets before ranking"))
        if not passed:
            raise AssertionError(name)
    _, _, reversal = case("reversal_is_one_final_target_change", {t[0]: 1}, {t[0]: -0.002}, {t[0]: -1})
    if reversal["reversed_pairs"] != 1 or reversal["entered_pairs"] or reversal["exited_pairs"]:
        raise AssertionError("Reversal event must not be duplicated as two orders")
    poisoned = normal | dict(target_relative_return=-999, label_valid=False, common_scoreable=False)
    case("future_fields_on_instruction_ignored", {}, ties, expected_ties, poisoned)

    for name, old, threshold in (("invalid_old_direction", {t[0]: 2}, h),
                                  ("too_many_old_pairs", dict.fromkeys(t[:5], 1), h),
                                  ("invalid_threshold", {}, 0.0)):
        try:
            decide_pairs(old, ties, normal, h_in=threshold, tickers=tickers)
        except ValueError:
            records.append(dict(check=name, passed=True, detail="Invalid state/configuration rejected"))
        else:
            raise AssertionError(name)
    # Replay verifies that a weaker first test signal cannot inherit a validation
    # holding, even when the preceding validation sample ends before liquidation.
    signal_rows, instructions = [], []
    for split, day, value in (("validation", "2022-12-28", 0.002), ("test", "2023-01-03", 0.0007)):
        date = pd.Timestamp(day)
        instructions.append(dict(date=date, model="fixture", scheme="S0", split=split,
                                 signal_at=date, action_at=date + pd.Timedelta(days=1), **normal))
        signal_rows.extend(dict(date=date, model="fixture", scheme="S0", ticker=ticker,
                                y_hat=value if ticker == t[0] else 0.0) for ticker in t)
    replay, _ = replay_pair_targets(pd.DataFrame(signal_rows), pd.DataFrame(instructions), h_in=h, tickers=tickers)
    test_first = replay.loc[replay["split"].eq("test")].iloc[0]
    passed = test_first.previous_pairs == 0 and test_first.target_pairs == 0
    records.append(dict(check="phase_replay_starts_flat", passed=bool(passed),
                        detail="Test phase does not inherit validation holdings"))
    if not passed:
        raise AssertionError("phase_replay_starts_flat")
    return pd.DataFrame(records)
