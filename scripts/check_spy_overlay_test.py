"""Hand-calculated checks for the new dividend-reinvestment comparator."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest_spy_overlay_test import build_summary, replay_spy_reinvest, sha256


class ReinvestmentChecks(unittest.TestCase):
    def fixture(self, dividends=(10.,2.,3.), prices=(100.,100.,100.)):
        dates=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'])
        quotes=pd.DataFrame(dict(date=dates,ticker='SPY',open=prices,dividends=dividends))
        rates=pd.DataFrame(dict(entry_date=dates[:-1],exit_date=dates[1:],
            assumed_release_date=['2019-12-30','2020-01-01'],holding_calendar_days=[1,3],
            interval_valid=True,rate_known_at_entry=True,risk_free_return_act365=0.))
        return quotes,rates,pd.DatetimeIndex(dates)

    def test_manual_dividend_reinvestment(self):
        book,assets=replay_spy_reinvest(*self.fixture())
        q0=1/(100*1.0003)
        q1=q0*(1+2/(100*1.0003))
        self.assertAlmostEqual(assets.target_shares.iloc[1],q1,places=14)
        self.assertAlmostEqual(book.nav_post.iloc[-1],q1*(103-.03),places=14)
        self.assertEqual(book.dividend_cash.iloc[0],0.)
        self.assertAlmostEqual(book.dividend_cash.iloc[-1],q1*3,places=14)
        self.assertTrue(book.cash_post.iloc[:-1].abs().lt(1e-12).all())

    def test_no_dividend_fixed_share_equivalence(self):
        book,assets=replay_spy_reinvest(*self.fixture(dividends=(0.,0.,0.),prices=(100.,102.,105.)))
        q=1/(100*1.0003)
        np.testing.assert_allclose(assets.target_shares.iloc[:-1],q,rtol=1e-12,atol=1e-14)
        self.assertAlmostEqual(book.nav_post.iloc[-1],q*105*(1-.0003),places=14)

    def test_final_liquidation_and_full_return_product(self):
        book,assets=replay_spy_reinvest(*self.fixture())
        self.assertEqual(assets.target_shares.iloc[-1],0.)
        self.assertTrue(np.isnan(book.holding_interval_return.iloc[0]))
        self.assertAlmostEqual(np.prod(1+book.holding_interval_return.iloc[1:]),book.nav_post.iloc[-1],places=14)
        self.assertGreater(book.execution_fee.iloc[0],0.)
        self.assertGreater(book.execution_fee.iloc[-1],0.)

    def test_unavailable_rate_rejected(self):
        prices,rates,dates=self.fixture()
        rates['assumed_release_date']='2021-01-01'
        with self.assertRaises(ValueError): replay_spy_reinvest(prices,rates,dates)

    def test_invalid_quote_rejected(self):
        prices,rates,dates=self.fixture()
        prices.loc[1,'open']=np.nan
        with self.assertRaises(ValueError): replay_spy_reinvest(prices,rates,dates)

    def test_independent_phase_start(self):
        a,_=replay_spy_reinvest(*self.fixture(),phase='validation')
        b,_=replay_spy_reinvest(*self.fixture(),phase='test')
        self.assertEqual(b.cash_prev.iloc[0],1.)
        self.assertEqual(b['split'].unique().tolist(),['test'])
        np.testing.assert_array_equal(a.nav_post,b.nav_post)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReinvestmentChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parents[1]/'results/backtest/spy_overlay_test_20261101'
    out.mkdir(parents=True,exist_ok=True)
    (out/'reinvestment_unit_checks.json').write_text(json.dumps(dict(
        tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        source_sha256=sha256(Path(__file__))),indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
