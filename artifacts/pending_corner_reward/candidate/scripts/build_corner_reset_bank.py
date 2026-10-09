"""Extract recorded approach states for the G3-to-G5 practice experiment."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-update", type=int, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    args = parser.parse_args()
    summary = json.loads(args.traces.with_name("summary.json").read_text())
    if (summary["course"] != "arena_38m_stacked" or summary["scenario"] != "course"
            or summary["update"] != args.source_update or summary["seed"] != args.source_seed):
        raise ValueError("Trace metadata does not match the declared source or corner course")
    with np.load(args.traces, allow_pickle=False) as trace:
        mask = ((trace["gate"] == 2) & trace["active"]
                & (trace["offsets"][:, :, 0] < -1.5)
                & (trace["offsets"][:, :, 0] > -4)
                & (trace["pos"][:, :, 2] > 0.4))
        rows, worlds = np.nonzero(mask)
        rows, worlds = rows[::5], worlds[::5]
        if len(rows) == 0:
            raise ValueError("No live pre-G3 approach states found")
        values = {name: trace[name][rows, worlds] for name in
                  ("pos", "vel", "quat", "angular", "rpm", "action")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values,
                       source_update=args.source_update, source_seed=args.source_seed,
                       gate_center=np.array([29., 24., 1.]),
                       gate_normal=np.array([0.33035042, -0.94385836, 0.]))
    print(f"Saved {len(rows)} states from {len(set(worlds.tolist()))} flights to {args.output}")


if __name__ == "__main__":
    main()
