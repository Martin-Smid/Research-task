import sys, json, importlib.util, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import chats_playground as p
root = Path(__file__).parent / "baselines" / "optimization_checks"
new = json.loads((p.DEFAULT_BASELINE_DIR / "latest_segment_comparison.json").read_text())
module = importlib.import_module("resources.Classes.Simulation_Class")
for name, attribute in (("old_propagator", "Propagator_Class"), ("old_evolution", "Evolution_Class")):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "archive" / f"{name}.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    setattr(module, attribute, getattr(old, attribute))
p.check_endpoint_segments(baseline_dir=root, scenario_names=p.SCENARIOS)
old_report = json.loads((root / "latest_segment_comparison.json").read_text())
assert old_report["scenarios"] == new["scenarios"], "Segment comparison changed from pre-patch code"
print("PASS: segment results/differences identical to pre-patch code")