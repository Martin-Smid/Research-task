"""Local before/after checks; original regression baselines stay untouched."""
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import chats_playground as p

root = Path(__file__).parent / "baselines" / "optimization_checks"
root.mkdir(exist_ok=True)
mode = sys.argv[1]
for order in (2, 4, 6):
    for name in ("wave", "wave_baryons", "wave_baryons_sink", "baryons_only"):
        sim, components = p.build_scenario("wave" if name == "wave" else "wave_baryons")
        if name == "wave_baryons_sink":
            components["sink"] = p._add_sink(sim)
        if name == "baryons_only":
            sim.wave_vectors.clear()
            components["wave"] = []
        sim.order_of_evolution = order
        sim.initialize_simulation()
        calls = {"poisson": 0, "sink": 0}
        for owner, method, key in ((sim.propagator, "solve_poisson", "poisson"),
                                   (sim.evolution, "_compute_sink_potential_analytic_kspace", "sink")):
            original = getattr(owner, method)
            def counted(*args, original=original, key=key, **kwargs):
                calls[key] += 1
                return original(*args, **kwargs)
            setattr(owner, method, counted)
        sim.evolution.evolve(sim.wave_functions, save_every=sim.num_steps,
                             diagnostics_every=sim.num_steps)
        state = {}
        p._capture_component_state(state, "final", components)
        p._capture_evolution_state(state, sim)
        path = root / f"{name}_{order}.npz"
        if mode == "before":
            assert not path.exists(), f"Refusing to overwrite {path}"
            p.np.savez(path, **state)
        else:
            with p.np.load(path) as reference:
                assert set(reference.files) == set(state)
                assert all(p._array_digest(reference[k]) == p._array_digest(v)
                           for k, v in state.items()), f"Changed results: {name}/{order}"
        print(f"CHECK {mode} {name}/{order}: exact; calls={calls}")

# Warm GPU timing of repeated Poisson solves, independently of output writing.
density = p.cp.ones((16, 16, 16), dtype=p.cp.float64)
for _ in range(5):
    sim.propagator.solve_poisson(density)
p.cp.cuda.Stream.null.synchronize()
start = perf_counter()
for _ in range(100):
    sim.propagator.solve_poisson(density)
p.cp.cuda.Stream.null.synchronize()
print(f"POISSON 100 calls: {perf_counter() - start:.6f}s")
