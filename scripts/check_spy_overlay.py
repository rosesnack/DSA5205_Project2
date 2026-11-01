"""Independent numerical/accounting checks for the exploratory SPY overlay."""
import unittest

import numpy as np
import pandas as pd

from backtest_pair_decisions import decide_pairs, validate_pair_decisions
from backtest_spy_overlay import (
    ASSETS, TICKERS, holding_episodes, overlay_step, overlay_weights, replay_overlay,
)


class OverlayChecks(unittest.TestCase):
    def step(self, cash=1., shares=None, previous=None, current=None, dividends=None,
             pairs=None, liquidate=False, **kwargs):
        prices = np.full(10, 100.)
        return overlay_step(cash, np.zeros(10) if shares is None else shares,
            prices if previous is None else previous, prices if current is None else current,
            np.zeros(10) if dividends is None else dividends,
            overlay_weights(pairs or {}, liquidate=liquidate)[0],
            **(dict(rf=0., calendar_days=0) | kwargs))

    def test_existing_signal_rules(self):
        self.assertTrue(validate_pair_decisions(TICKERS).passed.all())

    def test_no_signal_full_spy(self):
        weights, short, scale = overlay_weights({})
        np.testing.assert_array_equal(weights, [0.]*9+[1.])
        self.assertEqual((short, scale), (0., 1.))

    def test_four_shorts_scale(self):
        weights, short, scale = overlay_weights(dict.fromkeys(TICKERS[:4], -1))
        np.testing.assert_allclose(weights[:4], [-.0625]*4)
        self.assertEqual((short, scale, weights[-1]), (.5, .5, 1.25))
        self.assertEqual(np.abs(weights).sum(), 1.5)

    def test_mixed_scale_includes_long(self):
        pairs = dict(zip(TICKERS[:4], [-1, -1, -1, 1]))
        weights, short, scale = overlay_weights(pairs)
        np.testing.assert_allclose(weights[:4], [-1/12, -1/12, -1/12, 1/12])
        self.assertAlmostEqual(weights[-1], 7/6)
        self.assertAlmostEqual(np.abs(weights).sum(), 1.5)
        self.assertAlmostEqual(scale, 2/3)

    def test_exact_short_boundary_no_scale(self):
        weights, _, scale = overlay_weights(dict.fromkeys(TICKERS[:2], -1))
        self.assertEqual(scale, 1.)
        self.assertEqual(np.abs(weights).sum(), 1.5)

    def test_all_long_weights(self):
        weights, _, _ = overlay_weights(dict.fromkeys(TICKERS[:4], 1))
        self.assertEqual(weights[-1], .5)
        self.assertEqual(np.abs(weights).sum(), 1.)

    def test_initial_fee_funded_inside_cap(self):
        cash, q, assets, row = self.step(pairs=dict.fromkeys(TICKERS[:4], -1))
        expected = 1/(1+.0003*1.5)
        self.assertAlmostEqual(row['nav_post'], expected, places=14)
        self.assertAlmostEqual(cash, 0., places=14)
        self.assertAlmostEqual(row['actual_gross_post'], 1.5, places=14)
        self.assertAlmostEqual(row['execution_fee'], 1-expected, places=14)
        self.assertEqual(row['financing_interest'], 0.)

    def test_spy_legs_netted_before_fees(self):
        # Opposite overlays cancel their SPY changes: only the SPY core remains.
        _, _, assets, row = self.step(pairs={TICKERS[0]: 1, TICKERS[1]: -1})
        spy = [a for a in assets if a['ticker'] == 'SPY']
        self.assertEqual(len(spy), 1)
        expected = 1/(1+.0003*1.25)
        self.assertAlmostEqual(spy[0]['market_value_post'], expected, places=14)
        self.assertAlmostEqual(row['traded_notional'], expected*1.25, places=14)

    def test_weekend_financing_and_borrow_separate(self):
        q = np.zeros(10); q[0] = -.0025; q[-1] = .0125
        _, _, _, row = self.step(cash=0., shares=q, pairs={TICKERS[0]: -1},
            calendar_days=3, rf=.04*3/365, execution_bps=0)
        self.assertAlmostEqual(row['financing_base'], .25)
        self.assertAlmostEqual(row['financing_interest'], .25*.05*3/365)
        self.assertAlmostEqual(row['borrow_fee'], .25*.01*3/365)
        self.assertEqual(row['cash_interest'], 0.)
        self.assertAlmostEqual(row['nav_post'], 1-.25*.06*3/365)

    def test_negative_cash_increases_funding(self):
        q = np.zeros(10); q[0] = -.0025; q[-1] = .0145
        _, _, _, row = self.step(cash=-.2, shares=q, calendar_days=1, rf=.04/365)
        self.assertAlmostEqual(row['financing_base'], .45)
        self.assertEqual(row['cash_interest_base'], 0.)

    def test_positive_cash_above_collateral_only(self):
        q = np.zeros(10); q[0] = -.0025; q[-1] = .0125
        _, _, _, row = self.step(cash=.4, shares=q, calendar_days=1, rf=.04/365)
        self.assertAlmostEqual(row['cash_interest_base'], .15)
        self.assertEqual(row['financing_base'], 0.)
        self.assertAlmostEqual(row['cash_interest'], .15*.04/365)

    def test_old_short_dividend_and_final_exit(self):
        q = np.zeros(10); q[0] = -.0025; q[-1] = .0125
        div = np.zeros(10); div[0] = 2.; div[-1] = 1.
        cash, new, _, row = self.step(cash=0., shares=q, dividends=div, liquidate=True,
            calendar_days=1, rf=0., funding_spread=0., borrow_rate=0.)
        self.assertAlmostEqual(row['dividend_cash'], -.0025*2+.0125)
        self.assertAlmostEqual(row['execution_fee'], 1.5*.0003)
        np.testing.assert_array_equal(new, np.zeros(10))
        self.assertAlmostEqual(cash, 1+.0075-.00045)

    def test_new_ex_date_buyer_no_dividend(self):
        _, _, _, row = self.step(dividends=np.full(10, 10.))
        self.assertEqual(row['dividend_cash'], 0.)

    def test_missing_execution_quote_stops(self):
        prices = np.full(10, 100.); prices[-1] = np.nan
        with self.assertRaises(ValueError):
            self.step(current=prices)

    def test_hold_episode_rebalance_and_reversal(self):
        dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07'])
        frame = pd.DataFrame(dict(ticker=TICKERS[0], open_date=dates, target_shares=[1., .8, -.5, 0.]))
        ep = holding_episodes(frame, 'fixture')
        self.assertEqual([r['holding_intervals'] for r in ep], [2, 1])
        self.assertEqual([r['calendar_days'] for r in ep], [4, 1])
        self.assertEqual([r['end_reason'] for r in ep], ['reversal', 'phase_end'])

    def test_missing_signal_restores_spy_and_final_liquidates(self):
        dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])
        rows, predictions, prices = [], [], []
        for i, day in enumerate(dates):
            signal_date = day-pd.Timedelta(days=1)
            final = i == 2
            rows.append(dict(date=signal_date, model='fixture', scheme='S0', split='validation',
                signal_at=signal_date.tz_localize('America/New_York')+pd.Timedelta(hours=18),
                action_at=day.tz_localize('America/New_York')+pd.Timedelta(hours=9, minutes=30),
                action='FINAL_LIQUIDATION' if final else 'EVALUATE_SIGNAL',
                force_flat=final, signal_complete=i == 0, signal_reason='OK' if i == 0 else 'fixture_missing'))
            predictions.extend(dict(date=signal_date, model='fixture', scheme='S0', ticker=t,
                y_hat=.003 if t == TICKERS[0] else 0.) for t in TICKERS if i != 1 or t != TICKERS[-1])
            prices.extend(dict(date=day, ticker=t, open=100., dividends=0.) for t in ASSETS)
        rates = pd.DataFrame(dict(entry_date=dates[:-1], exit_date=dates[1:],
            assumed_release_date=['2019-12-30', '2020-01-01'], observation_date=['2019-12-27', '2019-12-30'],
            annual_yield_pct=[0., 0.], holding_calendar_days=[1, 3], interval_valid=True,
            rate_known_at_entry=True, risk_free_return_act365=0.))
        book, assets, _, _ = replay_overlay(pd.DataFrame(predictions), pd.DataFrame(rows), pd.DataFrame(prices), rates, 15)
        self.assertEqual(book.effective_action.tolist(), ['EVALUATE_SIGNAL', 'RESTORE_SPY', 'FINAL_LIQUIDATION'])
        self.assertEqual(book.target_pairs.tolist(), [1, 0, 0])
        self.assertEqual(len(book), 3)
        spy = assets.loc[assets.ticker.eq('SPY')]
        self.assertAlmostEqual(spy.actual_weight_post.iloc[1], 1.)
        self.assertEqual(assets.loc[assets.open_date.eq(dates[-1]), 'target_shares'].abs().sum(), 0.)
        self.assertAlmostEqual(np.prod(1+book.holding_interval_return.iloc[1:]), book.nav_post.iloc[-1])

    def test_threshold_change_only_affects_expected_entries(self):
        instruction = dict(action='EVALUATE_SIGNAL', force_flat=False, signal_complete=True)
        predictions = dict.fromkeys(TICKERS, 0.) | {TICKERS[0]: .0012}
        high = decide_pairs({}, predictions, instruction, h_in=.0015, tickers=TICKERS)[0]
        low = decide_pairs({}, predictions, instruction, h_in=.0010, tickers=TICKERS)[0]
        self.assertEqual(high, {})
        self.assertEqual(low, {TICKERS[0]: 1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
