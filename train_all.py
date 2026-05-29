"""Quick model training on a 10% sample for the demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.config import Config

SAMPLE_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1


def main():
    root = Config.get_project_root()
    ratings = pd.read_parquet(root / "data/processed/ratings.parquet")

    # Take a sample first, then split — much faster
    ratings = ratings.sample(frac=SAMPLE_FRAC, random_state=42)
    n = len(ratings)
    print(f"Working with {n:,} ratings ({SAMPLE_FRAC*100:.0f}% sample)")

    # Split
    test = ratings.sample(frac=0.1, random_state=42)
    rest = ratings.drop(test.index)
    val = rest.sample(frac=0.1, random_state=42)
    train = rest.drop(val.index)
    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    # ---- Popularity ----
    from src.models.popularity import PopularityModel
    print("\n=== Popularity ===")
    m = PopularityModel()
    m.train(train, val)
    metrics = m.evaluate(test)
    print(f"  NDGC@10={metrics['ndcg@10']:.4f}, HR@10={metrics['hit_rate@10']:.4f}, RMSE={metrics['rmse']:.4f}")
    m.save()

    # ---- SVD ----
    from src.models.svd_model import SVDModel
    print("\n=== SVD ===")
    m = SVDModel()
    m.train(train, val)
    metrics = m.evaluate(test)
    print(f"  NDGC@10={metrics['ndcg@10']:.4f}, HR@10={metrics['hit_rate@10']:.4f}, RMSE={metrics['rmse']:.4f}")
    m.save()

    # ---- LightGCN (small version) ----
    from src.models.lightgcn import LightGCNModel
    print("\n=== LightGCN ===")
    m = LightGCNModel()
    m.epochs = 20  # shorter for demo
    m.train(train, val)
    metrics = m.evaluate(test)
    print(f"  NDGC@10={metrics['ndcg@10']:.4f}, HR@10={metrics['hit_rate@10']:.4f}, RMSE={metrics['rmse']:.4f}")
    m.save()

    print("\nDone! All models trained and saved.")


if __name__ == "__main__":
    main()
