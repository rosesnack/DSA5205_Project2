"""Read-only GitHub delivery verification; Python standard library only.

Historical experiments retain their original manifests. Only the five explicitly
registered documentation edits may differ from their archived pre-sync bytes.
All data, forecasts, strategy code, ledgers, tables and figures remain exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SYNC='results/backtest/github_sync_20261101'
TEST='results/backtest/spy_overlay_test_20261101'
FINAL='results/backtest/spy_overlay_final_20261101'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_delivery(root,*,quiet=False):
    root=Path(root).resolve();sync=root/SYNC
    manifest=read_json(sync/'delivery_manifest.json')
    revisions=read_json(sync/'document_revisions.json')
    for rel,item in revisions.items():
        if sha256(root/item['snapshot'])!=item['before_sha256']:
            raise AssertionError(f'Historical document snapshot changed: {rel}')
        if sha256(root/rel)!=item['after_sha256']:
            raise AssertionError(f'Registered current document changed: {rel}')
    checked=set();accepted=set()
    def check(rel,expected):
        current=sha256(root/rel)
        if current!=expected:
            revision=revisions.get(rel)
            if not (revision and expected==revision['before_sha256'] and current==revision['after_sha256']):
                raise AssertionError(f'Missing or altered source/result: {rel}')
            accepted.add(rel)
        checked.add(rel)
    frozen=read_json(root/TEST/'frozen_protocol.json')
    for rel,expected in frozen['protected_hashes'].items(): check(rel,expected)
    stages=read_json(root/TEST/'all_steps_manifest.json')
    for rel,expected in stages['outputs'].items(): check(f'{TEST}/{rel}',expected)
    final=read_json(root/FINAL/'final_manifest.json')
    for rel,expected in final['outputs'].items(): check(rel,expected)
    check('scripts/finalize_spy_overlay.py',final['code_sha256'])
    for audit_path in [root/TEST/'final_audit.json',root/'results/backtest/spy_overlay_validation_20261101/final_audit.json']:
        for name,expected in read_json(audit_path)['source_hashes'].items():
            check('scripts/'+name,expected)
    for rel,expected in manifest['payload_hashes'].items():
        if sha256(root/rel)!=expected: raise AssertionError(f'GitHub payload changed: {rel}')
        checked.add(rel)
    selection=read_json(root/FINAL/'selection_record.json')
    assert selection['series_id']=='overlay_15_gbdt_S1' and selection['ex_post_selection']
    assert manifest['selected_series_id']==selection['series_id']
    notebook=read_json(root/'notebooks/06_spy_overlay_final_results.ipynb')
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==6 and all(c['execution_count'] is not None for c in cells)
    assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
    result=dict(passed=True,unique_files_checked=len(checked),registered_document_updates=sorted(accepted),
        upload_files=len(manifest['payload_hashes'])+1,selected_series_id=selection['series_id'],
        numerical_results_and_existing_strategy_code_unchanged=True,no_retraining_or_replay=True)
    if not quiet: print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    verify_delivery(p.parse_args().project_root)
