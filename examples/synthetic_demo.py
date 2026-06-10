"""Synthetic demo: the naive-vs-clustered standard error story, no API calls.

Simulates a GPQA-Diamond-shaped eval (198 items x 5 epochs, binary scores,
correlated within item) and prints the side-by-side report. The temperature-0
extreme repeats each item's score identically across epochs, so the naive SE
is sqrt(5) too small for zero added information.

Run:  python examples/synthetic_demo.py
"""

import numpy as np

from evalconfidence import ItemResult, standard_error

N_ITEMS, N_EPOCHS = 198, 5


def simulate(rng, *, temperature_zero=False):
    # Heterogeneous item difficulty (mean ~0.8), the realistic GPQA shape.
    p_item = rng.beta(8, 2, N_ITEMS)
    results = []
    for i in range(N_ITEMS):
        if temperature_zero:
            draws = [float(rng.random() < p_item[i])] * N_EPOCHS
        else:
            draws = (rng.random(N_EPOCHS) < p_item[i]).astype(float)
        results.extend(
            ItemResult(item_id=f"q{i}", model_id="demo/model-a", score=float(s), epoch=k)
            for k, s in enumerate(draws)
        )
    return results


def main():
    rng = np.random.default_rng(42)

    print(f"=== Realistic case: {N_ITEMS} items x {N_EPOCHS} epochs at temperature > 0 ===")
    print(standard_error(simulate(rng)))

    print(f"\n=== Temperature-0 extreme: identical epochs, design effect ~= {N_EPOCHS} ===")
    print(standard_error(simulate(rng, temperature_zero=True)))


if __name__ == "__main__":
    main()
