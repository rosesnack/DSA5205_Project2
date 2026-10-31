"""Step 15: read-only provenance verification, inventory and portable handoff.

Uses only the Python standard library. Never imports a fitting or replay engine.
Archives live outside the repository to avoid recursive packaging and hash cycles.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time
from urllib.parse import unquote
import zipfile


HANDOFF = Path("results/backtest/handoff")
EXCLUDED = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
FINAL_NOTE = "最终回测交接说明_yae.md"
STAGES = {9: "training_oof", 10: "threshold_selection", 11: "phase_evaluation",
          12: "investment_benchmarks", 13: "consistency_audit", 14: "reporting"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked(records, name, passed, detail):
    records.append(dict(check=name, passed=bool(passed), detail=detail))
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def resolve_payload(root, relative):
    # The only parent-relative payload is the authoritative sibling protocol.
    if relative == "../5205proj2.md":
        return root.parent / "5205proj2.md"
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Invalid payload path: {relative}")
    return root.joinpath(*p.parts)


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        p = resolve_payload(root, relative)
        if not p.is_file() or sha256(p) != expected:
            raise AssertionError(f"Missing or changed file: {relative}")


def review_file(relative):
    p = PurePosixPath(relative)
    if relative == "../5205proj2.md" or len(p.parts) == 1 or p.parts[0] == "scripts":
        return True
    if relative == "notebooks/05_portfolio_backtest.ipynb":
        return True
    if p.parts[:3] == ("results", "backtest", "reporting"):
        return "sensitivity" not in p.parts
    if p.parts[:3] == ("results", "backtest", "consistency_audit"):
        return True
    if p.parts[:2] in (("results", "elastic_net"), ("results", "gbdt")) and len(p.parts) == 3:
        return p.suffix in (".json", ".md", ".png") or p.name in {
            "forecast_metrics.csv", "selected_hyperparameters.csv", "loss_difference_uncertainty.csv"}
    if p.parts[:2] == ("results", "backtest"):
        if p.suffix == ".json" and len(p.parts) <= 4:
            return True
        return p.name in {"prediction_comparison_metrics.csv", "validation_candidate_scores.csv",
            "validation_strategy_summary.csv", "selection_checks.csv", "selection_logic_checks.csv",
            "phase_summary.csv", "evaluation_checks.csv", "benchmark_summary.csv",
            "strategy_benchmark_comparison.csv", "benchmark_checks.csv", "logic_checks.csv",
            "signal_quantiles.csv", "threshold_exceedance.csv", "threshold_candidates.csv",
            "elastic_net_selected_diagnostic.csv", "gbdt_selected_diagnostic.csv", "oof_checks.csv"}
    return False


def file_role(relative):
    if relative == "../5205proj2.md": return "authoritative_research_protocol"
    p = PurePosixPath(relative)
    if p.parts[0] == "data": return "original_data_snapshot"
    if p.parts[0] == "notebooks": return "notebook_or_upstream_data_audit"
    if p.parts[0] == "scripts": return "source_code"
    if p.parts[:2] in (("results", "elastic_net"), ("results", "gbdt")):
        return "original_model_results_and_diagnostics"
    if "reporting" in p.parts: return "formal_reporting_and_cost_sensitivity"
    if "threshold_selection" in p.parts: return "validation_threshold_selection"
    if "phase_evaluation" in p.parts: return "frozen_protocol_and_formal_phase_books"
    if "investment_benchmarks" in p.parts: return "investment_benchmark_books"
    if "consistency_audit" in p.parts: return "independent_accounting_audit"
    if "training_oof" in p.parts: return "separate_training_diagnostic"
    if p.parts[0] == "results": return "historical_preparation_or_engineering_result"
    return "documentation_or_original_environment_specification"


def payload_inventory(root):
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & EXCLUDED or HANDOFF in rel.parents:
            continue
        if p.suffix in {".pyc", ".tmp"} or p.name == ".DS_Store":
            continue
        if p.is_symlink():
            raise ValueError(f"Symlinks are not packaged: {rel}")
        files.append((rel.as_posix(), p))
    files.append(("../5205proj2.md", root.parent / "5205proj2.md"))
    return [dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p), role=file_role(rel),
                 include_review=review_file(rel), include_full=True) for rel, p in sorted(files)]


def check_links(root, docs, pending_manifest=None):
    count = 0
    for rel in docs:
        doc = root / rel
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            target = unquote(target.strip("<> ").split("#")[0])
            if not target or re.match(r"^[a-zA-Z]+:", target):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists() and resolved != pending_manifest:
                raise AssertionError(f"Broken link in {rel}: {target}")
            count += 1
    return count


def build_handoff(root):
    out = root / HANDOFF
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        return verify_handoff(root)
    out.mkdir(exist_ok=True, parents=True)
    baseline = read_json(out / "sources_before_step15.json")
    checks = []
    immutable = {k: v for k, v in baseline["source_hashes"].items()
                 if k not in baseline["editable_documents"]}
    verify_hashes(root, immutable)
    checked(checks, "existing_immutable_files_preserved", True, f"{len(immutable)} baseline hashes unchanged")

    prior = read_json(root / "results/backtest/reporting/run_manifest.json")
    verify_hashes(root, prior["source_hashes"])
    verify_hashes(root, prior["code_hashes"])
    verify_hashes(root, {x["path"]: x["sha256"] for x in prior["outputs"]})
    checked(checks, "step14_registered_sources_code_outputs", prior["status"] == "passed",
            f"{len(prior['source_hashes'])} sources, {len(prior['code_hashes'])} code files, {len(prior['outputs'])} outputs")
    stages = []
    for step, folder in STAGES.items():
        path = root / f"results/backtest/{folder}/run_manifest.json"
        m = read_json(path)
        passed = bool(m["checks"]) and all(c["passed"] for c in m["checks"])
        checked(checks, f"step_{step}_checks_passed", passed, str(len(m["checks"])))
        stages.append(dict(step=step, path=path.relative_to(root).as_posix(), sha256=sha256(path),
                           check_count=len(m["checks"]), all_passed=passed, completed_utc=m["completed_utc"]))
    threshold = read_json(root / "results/backtest/threshold_selection/selected_threshold.json")
    protocol = read_json(root / "results/backtest/phase_evaluation/protocol_config.json")
    checked(checks, "frozen_public_threshold", protocol["status"] == "frozen" and
            protocol["threshold"]["h_in"] == .0015 and protocol["threshold"]["h_out"] == .00075 and
            threshold["selected_public_entry_bps"] == 15 and threshold["selected_public_exit_bps"] == 7.5,
            "15 / 7.5 bps, original authority preserved")
    main = read_csv(root / "results/backtest/reporting/main_metrics.csv")
    sensitivity = read_csv(root / "results/backtest/reporting/sensitivity_metrics.csv")
    checked(checks, "main_path_coverage", len(main) == 16 and
            {x["split"] for x in main} == {"validation", "test"}, "12 strategy paths + 4 benchmark paths")
    checked(checks, "cost_scenario_coverage", len(sensitivity) == 96 and
            len({x["scenario_id"] for x in sensitivity}) == 8, "All six strategies, both phases, eight frozen settings")

    notebook = read_json(root / "notebooks/05_portfolio_backtest.ipynb")
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    code_hash = hashlib.sha256(json.dumps(code_cells, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    checked(checks, "notebook_existing_code_and_outputs_unchanged", code_hash == baseline["notebook_code_cells_sha256"],
            "17 previously executed code cells retained byte-equivalent as parsed JSON; documentation updated only")
    checked(checks, "notebook_native_kernel_caveat_preserved",
            notebook["metadata"]["validation"]["jupyter_kernel_execution_verified"] is False,
            "Sequential Python execution is recorded; no native Jupyter execution claim")
    distributions = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "ipython",
                     "nbformat", "nbconvert", "ipykernel", "notebook", "tzdata"]
    versions = {}
    for name in distributions:
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = None
    write_json(out / "environment_snapshot.json", dict(recorded_utc=now(), python_version=platform.python_version(),
        platform=platform.platform(), installed_distribution_versions=versions,
        provenance="Installed distribution metadata only; no installation, fitting or environment change",
        original_requirements_sha256=sha256(root / "requirements.txt"),
        requirements_note="Original modeling requirements remain unchanged; not a lockfile for this Python 3.11.5 backtest run",
        compatibility_note="Recorded versions describe the observed environment, not verified cross-version or cross-machine reproduction"))
    write_json(out / "upstream_validation.json", dict(stages=stages,
        step14_protected_files=len(prior["source_hashes"]), step14_code_files=len(prior["code_hashes"]),
        step14_outputs=len(prior["outputs"]), immutable_baseline_files=len(immutable),
        note="Counts are per-stage acceptance checks, not independent replications; historical pending flags are not rewritten"))
    rows = payload_inventory(root)
    required = ["../5205proj2.md", FINAL_NOTE, "scripts/backtest_handoff.py", "notebooks/05_portfolio_backtest.ipynb",
                "results/backtest/threshold_selection/selected_threshold.json",
                "results/backtest/phase_evaluation/protocol_config.json",
                "results/backtest/reporting/main_metrics.csv", "results/backtest/reporting/report_中文.md"]
    lookup = {x["path"]: x for x in rows}
    checked(checks, "review_contains_reading_and_verification_entrypoints",
            all(lookup[p]["include_review"] for p in required), str(len(required)))
    checked(checks, "full_contains_original_notebooks_and_all_books", all(p in lookup for p in [
        "notebooks/01_download_clean_data.ipynb", "notebooks/02_features_panel_data.ipynb",
        "notebooks/03_elastic_net.ipynb", "notebooks/04_gbdt.ipynb",
        "results/backtest/threshold_selection/checkpoints/entry_15_daily.csv",
        "results/backtest/phase_evaluation/test_daily.csv",
        "results/backtest/investment_benchmarks/benchmark_daily.csv"])
        and sum("reporting/sensitivity/checkpoints/" in p for p in lookup) == 84,
        "Original 01-04 notebooks, selected validation/test/benchmark books and all 14 six-file sensitivity checkpoints")
    write_csv(out / "file_inventory.csv", rows)
    count = check_links(root, ["README.md", FINAL_NOTE, "results/backtest/reporting/report_中文.md"],
                        pending_manifest=manifest_path.resolve())
    checked(checks, "documentation_local_links_resolve", count > 0,
            f"{count} relative links; final manifest self-link is checked again after publication")
    write_csv(out / "handoff_checks.csv", checks)
    outputs = [dict(path=p.relative_to(root).as_posix(), sha256=sha256(p))
               for p in sorted(out.iterdir()) if p.is_file() and p != manifest_path]
    manifest = dict(schema_version=1, scope="Step 15: final inventory, documentation and handoff", status="passed",
        completed_utc=now(), checks=checks, check_count=len(checks), inventory_files=len(rows),
        review_payload_files=sum(x["include_review"] for x in rows),
        full_payload_bytes=sum(x["bytes"] for x in rows),
        review_payload_bytes=sum(x["bytes"] for x in rows if x["include_review"]),
        inventory_path=(HANDOFF / "file_inventory.csv").as_posix(), outputs=outputs,
        package_layout="Sibling 5205proj2.md and DSA5205_Project2-main directory; relative paths preserved",
        archive_attestation="Detached archive_manifest.json beside archives; excluded from archive payload to avoid hash cycles",
        source_code=(Path(__file__).resolve().relative_to(root)).as_posix(),
        source_code_sha256=sha256(Path(__file__).resolve()),
        experiments_run_in_step15=False, portfolio_replay=False, model_retraining=False, threshold_reselection=False,
        step16_executed=False, prior_notebook_code_and_outputs_unchanged=True,
        evidence_boundary=protocol["evidence_boundary"],
        current_status="Steps 1-15 complete; step 16 optional and not executed",
        historical_manifests="Earlier pending flags are preserved snapshots; this file owns step-15 completion status")
    write_json(manifest_path, manifest)
    print(f"Step 15: {len(checks)} handoff checks passed; {len(rows)} payload files indexed", flush=True)
    return verify_handoff(root)


def verify_handoff(root, profile=None):
    out = root / HANDOFF
    marker = out / "archive_profile.json"
    if profile is None:
        profile = read_json(marker)["profile"] if marker.exists() else "full"
    m = read_json(out / "run_manifest.json")
    if m["status"] != "passed" or not all(c["passed"] for c in m["checks"]):
        raise AssertionError("Handoff is incomplete")
    verify_hashes(root, {x["path"]: x["sha256"] for x in m["outputs"]})
    rows = read_csv(root / m["inventory_path"])
    selected = [x for x in rows if profile == "full" or x["include_review"] == "True"]
    verify_hashes(root, {x["path"]: x["sha256"] for x in selected})
    verify_hashes(root, {m["source_code"]: m["source_code_sha256"]})
    if profile == "full":
        actual = {x["path"] for x in payload_inventory(root)}
        if actual != {x["path"] for x in rows}:
            raise AssertionError("Full payload file set changed; compare file_inventory.csv")
        check_links(root, ["README.md", FINAL_NOTE, "results/backtest/reporting/report_中文.md"])
    else:
        check_links(root, [FINAL_NOTE, "results/backtest/reporting/report_中文.md"])
    print(f"PASS: {profile} handoff, {len(selected)} payload hashes and {len(m['outputs'])} handoff-output hashes; no experiments", flush=True)
    return m


def archive_members(root, profile):
    out = root / HANDOFF
    m = read_json(out / "run_manifest.json")
    rows = read_csv(root / m["inventory_path"])
    hashes = {x["path"]: x["sha256"] for x in rows if profile == "full" or x["include_review"] == "True"}
    hashes.update({x["path"]: x["sha256"] for x in m["outputs"]})
    hashes[(HANDOFF / "run_manifest.json").as_posix()] = sha256(out / "run_manifest.json")
    prefix = "DSA5205_Project2_yae_20261031"
    entries = {}
    for rel, digest in hashes.items():
        name = f"{prefix}/5205proj2.md" if rel == "../5205proj2.md" else f"{prefix}/{root.name}/{rel}"
        entries[name] = dict(path=resolve_payload(root, rel), sha256=digest)
    marker = f"{prefix}/{root.name}/{HANDOFF.as_posix()}/archive_profile.json"
    data = (json.dumps(dict(profile=profile, scope="review is for reading only; full contains all data and ledgers",
                            created_by="scripts/backtest_handoff.py"), indent=2) + "\n").encode("utf-8")
    entries[marker] = dict(data=data, sha256=hashlib.sha256(data).hexdigest())
    return entries


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise AssertionError(f"Archive file set differs: {path.name}")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise AssertionError(f"Unsafe archive name: {name}")
            with archive.open(name) as stream:
                if digest_stream(stream) != expected[name]["sha256"]:
                    raise AssertionError(f"Archive payload changed: {name}")


def build_archives(root, target):
    verify_handoff(root, "full")
    if target == root or root in target.parents:
        raise ValueError("Archive directory must be outside the repository")
    target.mkdir(exist_ok=True, parents=True)
    manifest_path = target / "archive_manifest.json"
    if manifest_path.exists():
        return verify_archives(root, target)
    packages = []
    for profile in ["review", "full"]:
        path = target / f"DSA5205_Project2_yae_{profile}_20261031.zip"
        entries = archive_members(root, profile)
        if path.exists():
            verify_zip(path, entries)
        else:
            print(f"Packaging {profile}: {len(entries)} files", flush=True)
            temp = path.with_suffix(".zip.tmp")
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                for index, (name, item) in enumerate(sorted(entries.items()), 1):
                    if "data" in item:
                        archive.writestr(name, item["data"])
                    else:
                        archive.write(item["path"], name)
                    if index % 200 == 0: print(f"  {profile}: {index}/{len(entries)}", flush=True)
            verify_zip(temp, entries)
            temp.replace(path)
        packages.append(dict(profile=profile, path=path.name, bytes=path.stat().st_size,
                             sha256=sha256(path), member_count=len(entries), every_member_sha256_verified=True))
        print(f"PASS: {path.name}, {path.stat().st_size / 1024**2:.2f} MiB, all archived file hashes verified", flush=True)
    # Recheck the workspace after compression, before publishing the detached attestation.
    verify_handoff(root)
    write_json(manifest_path, dict(schema_version=1, status="passed", completed_utc=now(), packages=packages,
        handoff_manifest_sha256=sha256(root / HANDOFF / "run_manifest.json"),
        archive_contents="Original relative layout, external authoritative protocol included as sibling; caches and archives excluded",
        review_limitation="Reading package omits source data and full ledgers; do not run Notebook 05 or full pipeline from it",
        full_scope="Complete repository snapshot plus authoritative protocol; excludes environments, caches and VCS internals",
        experiments_run=False, step16_executed=False))
    return packages


def verify_archives(root, target):
    manifest = read_json(target / "archive_manifest.json")
    if manifest["status"] != "passed" or manifest["handoff_manifest_sha256"] != sha256(root / HANDOFF / "run_manifest.json"):
        raise AssertionError("Archive provenance differs")
    for item in manifest["packages"]:
        path = target / item["path"]
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise AssertionError(f"Archive hash differs: {path.name}")
        verify_zip(path, archive_members(root, item["profile"]))
    print("PASS: both archive SHA-256 values and every archived member verified", flush=True)
    return manifest["packages"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true", help="Read-only; fail if anything is absent or changed")
    parser.add_argument("--profile", choices=["full", "review"], help="Default: use archive marker, otherwise full")
    parser.add_argument("--archives-dir", type=Path, help="Build archives, or verify them when --verify-only is supplied")
    args = parser.parse_args()
    root = args.project_root.resolve()
    started = time.perf_counter()
    if args.verify_only:
        verify_handoff(root, args.profile)
        if args.archives_dir: verify_archives(root, args.archives_dir.resolve())
    else:
        build_handoff(root)
        if args.archives_dir: build_archives(root, args.archives_dir.resolve())
    print(f"Finished in {time.perf_counter()-started:.1f}s; no fitting, replay, reselection or optional extension", flush=True)
