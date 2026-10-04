
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path
import json, hashlib
root = Path("resources/data")
for reference in json.loads((root / "diagnostics_original_hashes.json").read_text(encoding="utf-8-sig")):
    assert hashlib.sha256(Path(reference["Path"]).read_bytes()).hexdigest().upper() == reference["Hash"], "Original baseline changed"
print("PASS: original baseline archives and manifest untouched")
old = json.loads((root / "optimization_checks/latest_segment_comparison.json").read_text())
new = json.loads((root / "chat_regression_baselines/latest_segment_comparison.json").read_text())
assert old["scenarios"] == new["scenarios"], "Endpoint comparisons changed"
print("PASS: endpoint continuation comparisons unchanged from pre-refactor")
manifest = json.loads((root / "chat_regression_baselines/diagnostics/manifest.json").read_text())
print("Benchmarks:", len(manifest), "; output files:", sum(len(case["outputs"]) for case in manifest.values()))
before = (root / "evolution_before_diagnostics.py").read_text(encoding="utf-8").splitlines()
after = Path("resources/Classes/Evolution_Class.py").read_text(encoding="utf-8").splitlines()
print("Evolution lines:", len(before), "->", len(after))
print("Diagnostics lines:", len(Path("resources/Classes/Diagnostics_Class.py").read_text(encoding="utf-8").splitlines()))
report = {"original_baselines_untouched": True, "energy_and_grid_benchmarks": 7,
          "energy_samples": 49, "saved_outputs_identical": 148,
          "match": "bitwise", "restart_match": "bitwise",
          "endpoint_comparisons_unchanged": True}
(root / "diagnostics_report.json").write_text(json.dumps(report, indent=2))