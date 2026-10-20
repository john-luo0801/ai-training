"""Write data/sample_houses_1m.csv with one million training rows."""

from pathlib import Path

import numpy as np

ROWS = 1_000_000
CHUNK = 100_000
OUTPUT = Path(__file__).resolve().parent / "sample_houses_1m.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    neighborhoods = np.array(["north", "south", "east", "west"])
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,price,size_sqft,bedrooms,age_years,neighborhood\n")
        for start in range(0, ROWS, CHUNK):
            count = min(CHUNK, ROWS - start)
            index = np.arange(start + 1, start + 1 + count)
            size = rng.integers(600, 3501, size=count)
            bedrooms = rng.integers(1, 6, size=count)
            age = rng.integers(0, 51, size=count)
            hood = neighborhoods[rng.integers(0, len(neighborhoods), size=count)]
            price = 50 * size + 10000 * bedrooms - 500 * age
            handle.writelines(
                f"H{i:07d},{p},{s},{b},{a},{h}\n"
                for i, p, s, b, a, h in zip(index, price, size, bedrooms, age, hood)
            )
            print(f"wrote {start + count:,} / {ROWS:,}")
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
