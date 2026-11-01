"""Explicit regeneration entry point for the already fixed 10/15 bp experiments.

This is optional. Normal GitHub upload needs only the read-only delivery checker.
Fresh replay writes outside the repository, leaving every saved result intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from verify_spy_overlay_delivery import verify_delivery


def prepare_output(root,target):
    root=Path(root).resolve();target=Path(target).resolve()
    if target.is_relative_to(root) or not target.is_relative_to(root.parent) or target==root.parent:
        raise ValueError('Use a new directory outside the repository but within the Project2 workspace, e.g. ../tmp/spy_replay_test_15bp')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Replay output directory must be empty; saved results are never overwritten')
    return target


def run(root,phase,entry,target):
    root=Path(root).resolve();target=prepare_output(root,target)
    verify_delivery(root)
    from backtest_phase_evaluation import load_phase_inputs
    from backtest_spy_overlay import replay_overlay,validate_run,read_frame,save_frame,sha256,write_json
    from backtest_spy_overlay_test import replay_spy_reinvest,build_summary
    import pandas as pd
    (signals,daily,prices,rates,opens),input_checks=load_phase_inputs(root,phase)
    reasons=read_frame(root/'results/backtest/daily_signal_status.csv',parse_dates=['date'],
        usecols=['date','model','scheme','reason'])
    daily=daily.merge(reasons.rename(columns={'reason':'signal_reason'}),
        on=['date','model','scheme'],validate='one_to_one')
    target.mkdir(parents=True,exist_ok=True)
    write_json(target/'replay_config.json',dict(phase=phase,entry_bps=entry,exit_bps=entry/2,
        source_delivery_sha256=sha256(root/'results/backtest/github_sync_20261101/delivery_manifest.json'),
        trading_rules='Unchanged original overlay_weights, overlay_step, replay_overlay and decide_pairs',
        model_refitting=False,old_results_overwritten=False))
    book,assets,trades,decisions=replay_overlay(signals,daily,prices,rates,entry)
    checks=validate_run(book,assets,rates,opens,entry)
    ri,_=replay_spy_reinvest(prices,rates,opens,phase)
    summary,curves,episodes=build_summary(root,[(entry,book,assets)],ri,phase)
    for name,frame in [('daily.csv',book),('assets.csv',assets),('trades.csv',trades),
        ('decisions.csv',decisions),('checks.csv',checks),('input_checks.csv',pd.DataFrame(input_checks)),
        ('summary.csv',summary),('nav_curves.csv',curves),('holding_episodes.csv',episodes)]:
        save_frame(target/name,frame)
    verify_delivery(root,quiet=True)
    print(f'PASS: {len(checks)} runtime checks; fresh output {target}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--phase',choices=['validation','test'],required=True)
    p.add_argument('--entry-bps',type=int,choices=[10,15],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.project_root,a.phase,a.entry_bps,a.output_dir)
