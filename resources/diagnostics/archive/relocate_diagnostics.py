
# Historical snapshot/migration: retained for review, not for execution.
if __name__ == "__main__":
    raise SystemExit("Archived file; use the documented active diagnostics instead.")

from pathlib import Path
import ast, hashlib, json

root = Path.cwd()
diag = root / "resources/diagnostics"
path = root / "chats_playground.py"
source = path.read_text(encoding="utf-8")
source = source.replace('PROJECT_ROOT / "resources" / "data" / "chat_regression_baselines"',
                        'PROJECT_ROOT / "resources" / "diagnostics" / "baselines" / "chat_regression_baselines"')
source = source.replace('PROJECT_ROOT / "resources" / "data" / "chat_runtime"',
                        'PROJECT_ROOT / "resources" / "diagnostics" / "runtime"')
source = source.replace('Baseline arrays are written below resources/data/. That directory is ignored by\nGit, while this harness remains versioned with the source code.',
                        'Frozen baseline arrays live below resources/diagnostics/baselines/.\nGenerated replay runs and compiler caches are ignored by Git.')
path.write_text(source, encoding="utf-8")
for name in ("check_optimization.py", "verify_old_segments.py"):
    path = diag / name
    source = path.read_text(encoding="utf-8")
    source = source.replace('Path(__file__).parent / "optimization_checks"',
                            'Path(__file__).parent / "baselines" / "optimization_checks"')
    source = source.replace('root / f"{name}.py"', 'Path(__file__).parent / "archive" / f"{name}.py"')
    path.write_text(source, encoding="utf-8")
path = diag / "rotation_playground.py"
source = path.read_text(encoding="utf-8").replace('from types import SimpleNamespace\n',
    'from types import SimpleNamespace\nfrom pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2]))\n', 1)
path.write_text(source, encoding="utf-8")
for path in (diag / "archive").glob("*.py"):
    source = path.read_text(encoding="utf-8-sig")
    body = ast.parse(source).body
    header_end = 0
    for statement in body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            header_end = statement.end_lineno
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str) and header_end == 0:
            header_end = statement.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    lines[header_end:header_end] = ['\n# Historical snapshot/migration: retained for review, not for execution.\n',
                                 'if __name__ == "__main__":\n',
                                 '    raise SystemExit("Archived file; use the documented active diagnostics instead.")\n\n']
    path.write_text(''.join(lines), encoding="utf-8")
path = diag / "reports/diagnostics_original_hashes.json"
references = json.loads(path.read_text(encoding="utf-8-sig"))
for reference in references:
    filename = Path(reference["Path"]).name
    target = diag / "baselines/chat_regression_baselines" / filename
    assert hashlib.sha256(target.read_bytes()).hexdigest().upper() == reference["Hash"]
    reference["Path"] = target.relative_to(root).as_posix()
path.write_text(json.dumps(references, indent=2), encoding="utf-8")
path = root / ".gitignore"
source = path.read_text(encoding="utf-8")
source += '''\n# Frozen reference arrays and diagnostics source stay versioned.\nresources/diagnostics/runtime/\nresources/diagnostics/logs/\nresources/diagnostics/baselines/checkpoint_diagnostics/\nresources/diagnostics/baselines/**/restart_runs/\nresources/diagnostics/baselines/**/segment_runs/\nresources/diagnostics/baselines/**/latest_*.json\nresources/diagnostics/reports/*.png\n'''
path.write_text(source, encoding="utf-8")
path = root / "docs/rotation_curves.md"
source = path.read_text(encoding="utf-8")
source = source.replace('python rotation_playground.py', 'python resources/diagnostics/rotation_playground.py')
source = source.replace('Without `--km-s`, the\noriginal simulation velocity units are retained.',
                        'The new plotting entry point converts physical velocities to km/s by default;\nuse `--native-units` to keep simulation units.')
source += '''\n## Convenient plotting entry point\n\n```text\npython plot_rot_curves.py\npython plot_rot_curves.py -g\npython plot_rot_curves.py -N\npython plot_rot_curves.py simulation_20261004_...\npython plot_rot_curves.py resources/data/simulation_... resources/data/simulation_...\n```\n\nNo directory means the newest run with a rotation output file. Both gas and\nN-body components are selected by default; `-g` selects gas and `-N` selects\nN-body, while both flags select both types. Component types come from\n`run_config.json`, with a name-based fallback for legacy runs. Each run uses\nits own latest saved time, or the nearest time requested through `--time`.\nThe figure is saved in the first run directory and displayed; `--no-show` only\nsaves it. The examples entry point delegates to this same implementation.\n'''
path.write_text(source, encoding="utf-8")
path = root / "plot_rot_curves.py"
source = path.read_text(encoding="utf-8").replace('plotted, labels, display_units = False, [], None', 'plotted, display_units = False, None')
source = source.replace('        labels.append(f"{selected_time:g}")\n', '')
source = source.replace('directories = [p if p.is_dir() else DATA_DIRECTORY / p for p in directories]',
                        'directories = [DATA_DIRECTORY / p if not p.is_dir() and p.parent == Path(".") else p for p in directories]')
path.write_text(source, encoding="utf-8")
print("Paths updated; original frozen reference hashes verified unchanged")