"""Integration checks for fresh-run wiring; no upstream model training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fresh_backtest_pipeline as p


def temporary_workspace():
    # Explicit project-local parent avoids Windows AppContainer TEMP redirection.
    parent = Path(__file__).resolve().parents[1] / "results" / "pipeline_tests"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="fixture_", dir=parent)


def fixture(root):
    phases = {"validation": ("2020-01-03", "2020-01-08"), "test": ("2023-01-04", "2023-01-09")}
    signals, daily, prices, rates = [], [], [], []
    for phase, (first, last) in phases.items():
        opens = pd.bdate_range(first, last)
        for i, day in enumerate(opens):
            signal_day = day - pd.offsets.BDay(1)
            for m in p.MODELS:
                for s in p.SCHEMES:
                    daily.append(dict(date=signal_day, model=m, scheme=s, split=phase,
                        signal_at=(signal_day+pd.Timedelta(hours=18)).tz_localize(p.NY),
                        action_at=(day+pd.Timedelta(hours=9, minutes=30)).tz_localize(p.NY),
                        action="FINAL_LIQUIDATION" if i == len(opens)-1 else "EVALUATE_SIGNAL",
                        force_flat=i == len(opens)-1, signal_complete=True, reason="fixture"))
                    for j, ticker in enumerate(p.TICKERS):
                        # Both signs, retained pairs and a reversal; all six schemes.
                        value = (1 if j % 2 else -1) * .004 if j < 4 else 0.
                        if i == 2:
                            value = -value
                        signals.append(dict(date=signal_day, ticker=ticker, model=m, scheme=s, split=phase, y_hat=value))
            for j, ticker in enumerate([*p.TICKERS, "SPY"]):
                prices.append(dict(date=day, ticker=ticker, open=100+j+i*(j%3-1),
                                   dividends=.2 if i == 1 else 0., observed=True))
            if i:
                days = (day-opens[i-1]).days
                rates.append(dict(entry_date=opens[i-1], exit_date=day, holding_calendar_days=days,
                    interval_valid=True, rate_known_at_entry=True, annual_yield_pct=3.65,
                    observation_date=opens[i-1]-pd.Timedelta(days=2),
                    assumed_release_date=opens[i-1]-pd.Timedelta(days=1), risk_free_return_act365=.0001*days))
    ctx = dict(root=root, out=root/"results/reproduction/fixture", signals=pd.DataFrame(signals),
               daily=pd.DataFrame(daily), prices=pd.DataFrame(prices), rates=pd.DataFrame(rates))
    p.save(root/"data/interim/etf_prices_aligned.csv", ctx["prices"])
    p.save(root/"data/interim/risk_free_open_intervals.csv", ctx["rates"])
    p.save(ctx["out"]/"inputs/trading_signal_inputs.csv", ctx["signals"])
    p.save(ctx["out"]/"inputs/daily_signal_status.csv", ctx["daily"])
    config = dict(run_id="fixture", version=p.VERSION, runtime=p.runtime(),
        input_hashes={"data/interim/etf_prices_aligned.csv":p.sha(root/"data/interim/etf_prices_aligned.csv")}, code_hashes={})
    ctx["config"] = config
    p.write_json(ctx["out"]/"run_config.json", config)
    return ctx, phases


class FreshPipelineTests(unittest.TestCase):
    def test_complete_pair_and_overlay_without_historical_results(self):
        with temporary_workspace() as temp:
            ctx, phases = fixture(Path(temp))
            with patch.dict(p.PHASE_OPENS, phases, clear=True):
                scores = p.select_threshold(ctx)
                self.assertEqual(len(scores), 5)
                main = p.run_pair_phases(ctx)
                self.assertEqual(len(main), 12)
                p.run_benchmarks(ctx)
                sensitivity = p.run_sensitivities(ctx)
                self.assertEqual(len(sensitivity), 96)
                p.finish_pair_run(ctx)
                # Missing historical results/backtest is deliberate.
                self.assertFalse((ctx["root"]/"results/backtest").exists())
                connected = p.connect_pair_run(ctx["root"])
                first = p.start_overlay_run(connected)
                p.run_overlay_phase(connected, "validation")
                p.run_overlay_phase(connected, "test")
                summary, ranked, yearly = p.finish_overlay_run(connected)
                self.assertEqual(len(ranked), 12)
                self.assertEqual(len(summary), 8)
                self.assertTrue(len(yearly) > 0)
                # Relative-risk fields must identify the reinvestment benchmark,
                # which differs from historical fixed-share SPY columns.
                self.assertNotIn("annualized_tracking_error", ranked.columns)
                self.assertNotIn("information_ratio", ranked.columns)
                self.assertTrue(ranked.relative_performance_benchmark.eq("spy_reinvest").all())
                winner = ranked.iloc[0]
                book = connected["overlay_selected_books"][("test", winner.series_id)]
                ri = connected["reinvest_books"]["test"]
                active = (book.sort_values("open_date").holding_interval_return.iloc[1:].to_numpy()
                          - ri.sort_values("open_date").holding_interval_return.iloc[1:].to_numpy())
                tracking_error = np.std(active, ddof=1) * np.sqrt(252)
                self.assertAlmostEqual(winner.annualized_tracking_error_vs_spy_reinvest, tracking_error)
                if tracking_error > 1e-12:
                    self.assertAlmostEqual(winner.information_ratio_vs_spy_reinvest,
                                           np.mean(active) * 252 / tracking_error)
                self.assertTrue(p.read_json(first/"complete.json")["strategy_replayed"])
                second = p.start_overlay_run(connected)
                self.assertNotEqual(first, second)
                self.assertFalse((second/"complete.json").exists())
                # A changed upstream quote must reject stale 05 even with saved tables.
                path = ctx["root"]/"data/interim/etf_prices_aligned.csv"
                path.write_bytes(path.read_bytes()+b"\n")
                with self.assertRaisesRegex(ValueError, "Input/code/result changed"):
                    p.connect_pair_run(ctx["root"])

    def test_independent_audit_rejects_corrupt_nav(self):
        with temporary_workspace() as temp:
            ctx, phases = fixture(Path(temp))
            with patch.dict(p.PHASE_OPENS, phases, clear=True):
                inputs = p.phase_inputs(ctx, "validation")
                frames = p.pair_replay(inputs, 15, p.main_cost_parameters())
                p.audit_pair_book(*frames, inputs, p.main_cost_parameters())
                damaged = frames[0].copy()
                damaged.loc[1, "nav_post"] += .01
                with self.assertRaisesRegex(AssertionError, "independent_cash_and_nav"):
                    p.audit_pair_book(damaged, frames[1], frames[2], inputs, p.main_cost_parameters())

    def test_missing_fresh_run_never_falls_back(self):
        with temporary_workspace() as temp:
            root = Path(temp)
            (root/"results/backtest/spy_overlay_final_20261101").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "Run the NEW notebook 05"):
                p.connect_pair_run(root)


if __name__ == "__main__":
    unittest.main()
