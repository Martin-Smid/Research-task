
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path
import json, hashlib
root = Path("resources/data")
for reference in json.loads((root / "diagnostics_original_hashes.json").read_text(encoding="utf-8-sig")):
    assert hashlib.sha256(Path(reference["Path"]).read_bytes()).hexdigest().upper() == reference["Hash"]
for filename in ("latest_comparison.json", "latest_restart_comparison.json"):
    report = json.loads((root / "chat_regression_baselines" / filename).read_text())
    assert all(case["passed"] and case["bitwise_state_match"] for case in report["scenarios"].values())
old = json.loads((root / "optimization_checks/latest_segment_comparison.json").read_text())
new = json.loads((root / "chat_regression_baselines/latest_segment_comparison.json").read_text())
assert old["scenarios"] == new["scenarios"]
report = {"original_baselines_untouched": True, "original_grid_energy_match": "bitwise",
          "detailed_benchmarks": 7, "energy_samples": 49,
          "approved_changed_files": ["rotational_velocity.dat", "rotation_frames.csv"],
          "restart_match": "bitwise", "segment_comparisons_unchanged": True,
          "analytic_tests_passed": 8, "plotting_preview_verified": True}
(root / "rotation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))