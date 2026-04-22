"""
Train models on synthetic historical data:
  1. Next-best-game classifier (Random Forest)
  2. Churn probability regressor (Gradient Boosted)
  3. High-roller trajectory classifier (Gradient Boosted)

Models are saved to /app/models/ and loaded by the inference loop.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

MODEL_DIR = "/app/models"
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(42)

NUM_SAMPLES = 10000

FEATURE_COLS = [
    "avg_bet", "win_rate",
    "pct_slots", "pct_baccarat", "pct_roulette", "pct_blackjack", "pct_poker",
    "fnb_spend", "hotel_spend",
    "category_diversity", "spend_per_minute",
    "high_roller_similarity",
]

# Order matches the dirichlet draws below.
GAME_TYPES = ["slots", "baccarat", "roulette", "blackjack", "poker"]
OFFER_TYPES = ["free_play", "fnb_voucher", "hotel_upgrade", "cashback"]


def generate_synthetic_data(n: int) -> pd.DataFrame:
    """Generate synthetic player feature snapshots with labels."""
    # 5-way dirichlet over game mix so that pct_* columns sum to 1. The prior
    # is biased toward baccarat + slots to match the Macau-style floor.
    game_mix = np.random.dirichlet(np.array([2.5, 3.5, 0.6, 1.2, 0.6]), n)
    data = {
        "avg_bet": np.random.lognormal(3.5, 1.2, n).clip(1, 5000),
        "win_rate": np.random.beta(4, 6, n),
        "pct_slots":     game_mix[:, 0],
        "pct_baccarat":  game_mix[:, 1],
        "pct_roulette":  game_mix[:, 2],
        "pct_blackjack": game_mix[:, 3],
        "pct_poker":     game_mix[:, 4],
        "fnb_spend": np.random.exponential(30, n).clip(0, 500),
        "hotel_spend": np.random.exponential(50, n).clip(0, 1000),
        "category_diversity": np.random.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25]),
        "spend_per_minute": np.random.lognormal(1.5, 1.0, n).clip(0.1, 500),
        "high_roller_similarity": np.random.beta(2, 5, n),
    }
    df = pd.DataFrame(data)

    # --- Labels ---

    # Next-best-game: recommend the player's 2nd-most-played game (cross-sell).
    game_prefs = df[["pct_slots", "pct_baccarat", "pct_roulette",
                     "pct_blackjack", "pct_poker"]].values
    next_game_idx = np.argsort(game_prefs, axis=1)[:, -2]
    df["next_best_game"] = [GAME_TYPES[i] for i in next_game_idx]

    # Churn probability: right-skewed so most players are low-churn (~0.1-0.3)
    # with a meaningful tail (~15-20% above 0.3, ~5% above 0.5)
    raw_churn = (
        0.35 * (1 - df["spend_per_minute"] / df["spend_per_minute"].quantile(0.85))
        + 0.25 * (1 - df["category_diversity"] / 3.0)
        + 0.20 * (1 - df["win_rate"])
        + 0.20 * np.random.uniform(0, 0.4, n)
    ).clip(0, 1)
    # Mild power transform: skew toward low but keep a real tail
    churn_score = np.power(raw_churn, 1.3).clip(0, 1)
    df["churn_probability"] = churn_score

    # Offer sensitivity: based on spending patterns
    offer_idx = np.zeros(n, dtype=int)
    offer_idx[df["avg_bet"] > df["avg_bet"].median()] = 0   # free_play for high bettors
    offer_idx[df["fnb_spend"] > df["fnb_spend"].quantile(0.7)] = 1  # fnb_voucher
    offer_idx[df["hotel_spend"] > df["hotel_spend"].quantile(0.7)] = 2  # hotel_upgrade
    offer_idx[df["category_diversity"] == 1] = 3  # cashback for single-category
    df["offer_sensitivity"] = [OFFER_TYPES[i] for i in offer_idx]

    # High-roller trajectory: True if similarity > 0.5 AND high spend velocity
    df["high_roller_trajectory"] = (
        (df["high_roller_similarity"] > 0.45)
        & (df["spend_per_minute"] > df["spend_per_minute"].quantile(0.6))
        & (df["category_diversity"] >= 2)
    ).astype(int)

    return df


def train_and_save():
    print("Generating synthetic training data...")
    df = generate_synthetic_data(NUM_SAMPLES)
    X = df[FEATURE_COLS]

    # 1. Next-best-game classifier
    print("Training next-best-game classifier...")
    y_game = df["next_best_game"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_game, test_size=0.2, random_state=42)
    clf_game = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_game.fit(X_train, y_train)
    acc = clf_game.score(X_test, y_test)
    print(f"  Next-best-game accuracy: {acc:.3f}")
    joblib.dump(clf_game, os.path.join(MODEL_DIR, "next_game_model.pkl"))

    # 2. Churn probability regressor
    print("Training churn probability regressor...")
    y_churn = df["churn_probability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)
    reg_churn = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg_churn.fit(X_train, y_train)
    r2 = reg_churn.score(X_test, y_test)
    print(f"  Churn regressor R2: {r2:.3f}")
    joblib.dump(reg_churn, os.path.join(MODEL_DIR, "churn_model.pkl"))

    # 3. High-roller trajectory classifier
    print("Training high-roller trajectory classifier...")
    y_hr = df["high_roller_trajectory"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_hr, test_size=0.2, random_state=42)
    clf_hr = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_hr.fit(X_train, y_train)
    acc = clf_hr.score(X_test, y_test)
    print(f"  High-roller trajectory accuracy: {acc:.3f}")
    joblib.dump(clf_hr, os.path.join(MODEL_DIR, "high_roller_model.pkl"))

    # 4. Offer sensitivity classifier
    print("Training offer sensitivity classifier...")
    y_offer = df["offer_sensitivity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_offer, test_size=0.2, random_state=42)
    clf_offer = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_offer.fit(X_train, y_train)
    acc = clf_offer.score(X_test, y_test)
    print(f"  Offer sensitivity accuracy: {acc:.3f}")
    joblib.dump(clf_offer, os.path.join(MODEL_DIR, "offer_model.pkl"))

    print(f"All models saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_and_save()
