"""Write larger sample CSVs for training, prediction, and target-only runs."""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent
NEIGHBORHOODS = np.array(["north", "south", "east", "west"])


def write_houses(path: Path, n: int, *, blank_price: bool, seed: int, id_prefix: str) -> None:
    rng = np.random.default_rng(seed)
    size = rng.integers(600, 3501, size=n)
    bedrooms = rng.integers(1, 6, size=n)
    age = rng.integers(0, 51, size=n)
    hood = NEIGHBORHOODS[rng.integers(0, len(NEIGHBORHOODS), size=n)]
    price = 50 * size + 10000 * bedrooms - 500 * age
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,price,size_sqft,bedrooms,age_years,neighborhood\n")
        for index in range(n):
            value = "" if blank_price else str(int(price[index]))
            handle.write(
                f"{id_prefix}{index + 1:05d},{value},{size[index]},{bedrooms[index]},{age[index]},{hood[index]}\n"
            )
    print(f"Saved {path.name} ({n:,} rows, {path.stat().st_size / 1024:.1f} KB)")


def write_target_only(path: Path, n: int, *, blank_price: bool, seed: int, id_prefix: str) -> None:
    rng = np.random.default_rng(seed)
    size = rng.integers(600, 3501, size=n)
    bedrooms = rng.integers(1, 6, size=n)
    age = rng.integers(0, 51, size=n)
    price = 50 * size + 10000 * bedrooms - 500 * age
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("id,price\n")
        for index in range(n):
            value = "" if blank_price else str(int(price[index]))
            handle.write(f"{id_prefix}{index + 1:05d},{value}\n")
    print(f"Saved {path.name} ({n:,} rows, {path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    write_houses(DATA / "sample_houses_to_predict_10k.csv", 10_000, blank_price=True, seed=7, id_prefix="P")
    write_target_only(DATA / "sample_target_only_10k.csv", 10_000, blank_price=False, seed=11, id_prefix="T")
    write_target_only(
        DATA / "sample_target_only_to_predict_10k.csv", 10_000, blank_price=True, seed=13, id_prefix="Q"
    )


if __name__ == "__main__":
    main()
