"""Fixed-point tiny exact TSP calibration; not a simulation of the steady model."""

import itertools
import math
import random
import numpy as np


def calibrate(p):
    rng = random.Random(p["seed"])
    rows = []
    point_sets = []
    hub = (1, 0)

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * p["radius"] / p["speed"]

    for n in p["stop_counts"]:
        durations = []
        for sample in range(p["samples"]):
            points = []
            for _ in range(n):
                r = math.sqrt(rng.random())
                angle = rng.random() * 2 * math.pi
                points.append((r * math.cos(angle), r * math.sin(angle)))
            optimum = min(
                distance(hub, seq[0])
                + sum(distance(a, b) for a, b in zip(seq, seq[1:]))
                + distance(seq[-1], hub)
                for seq in itertools.permutations(points)
            )
            durations.append(optimum)
            point_sets.append(
                {
                    "stops": n,
                    "sample": sample,
                    "hub": hub,
                    "points": points,
                    "exact_travel_time": optimum,
                }
            )
        rows.append(
            {
                "stops": n,
                "mean_travel_time": float(np.mean(durations)),
                "std_travel_time": float(np.std(durations)),
                "sqrt_stops": math.sqrt(n),
            }
        )
    x = np.array([row["sqrt_stops"] for row in rows])
    y = np.array([row["mean_travel_time"] for row in rows])
    b = float(x @ y / (x @ x))
    return {
        "calibrated_b_no_stem": b,
        "rows": [row | {"prediction": b * row["sqrt_stops"]} for row in rows],
        "fixed_points": point_sets,
        "assumptions": "disc, boundary hub, exact TSP up to eight stops; zero-stem fitted BHH; no operational policy claim",
    }
