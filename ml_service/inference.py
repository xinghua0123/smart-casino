"""
ML Inference Loop:
  1. Query RisingWave for latest player features + high-roller similarity
  2. Run all 4 models (next-game, churn, offer, high-roller trajectory)
  3. Upsert predictions into RisingWave recommendations_tbl

Runs every 10 seconds (same pattern as the RisingWave traffic prediction demo).
"""

import os
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras

RW_HOST = os.environ.get("RISINGWAVE_HOST", "risingwave")
RW_PORT = os.environ.get("RISINGWAVE_PORT", "4566")
MODEL_DIR = "/app/models"
INFERENCE_INTERVAL_S = 10

FEATURE_COLS = [
    "avg_bet", "win_rate",
    "pct_slots", "pct_baccarat", "pct_blackjack",
    "fnb_spend", "hotel_spend",
    "category_diversity", "spend_per_minute",
    "high_roller_similarity",
]


def load_models():
    return {
        "next_game": joblib.load(os.path.join(MODEL_DIR, "next_game_model.pkl")),
        "churn": joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl")),
        "offer": joblib.load(os.path.join(MODEL_DIR, "offer_model.pkl")),
        "high_roller": joblib.load(os.path.join(MODEL_DIR, "high_roller_model.pkl")),
    }


def get_rw_connection():
    conn = psycopg2.connect(
        host=RW_HOST, port=RW_PORT, user="root", dbname="dev"
    )
    conn.autocommit = True
    return conn


def fetch_features(conn) -> pd.DataFrame:
    """Fetch the latest player features with high-roller similarity."""
    query = """
    SELECT
        p.player_id,
        p.avg_bet,
        p.win_rate,
        p.pct_slots,
        p.pct_baccarat,
        p.pct_blackjack,
        p.fnb_spend,
        p.hotel_spend,
        p.category_diversity,
        p.spend_per_minute,
        COALESCE(h.high_roller_similarity, 0.0) AS high_roller_similarity
    FROM mv_player_features p
    LEFT JOIN mv_player_high_roller_similarity h
        ON p.player_id = h.player_id AND p.window_start = h.window_start
    """
    return pd.read_sql(query, conn)


def run_inference(models, df: pd.DataFrame) -> list[tuple]:
    if df.empty:
        return []

    X = df[FEATURE_COLS].fillna(0)

    next_games = models["next_game"].predict(X)
    churn_probs = models["churn"].predict(X).clip(0, 1)
    offers = models["offer"].predict(X)
    hr_trajectories = models["high_roller"].predict(X)
    hr_scores = df["high_roller_similarity"].values

    now = datetime.now(timezone.utc)
    results = []
    for i in range(len(df)):
        results.append((
            df.iloc[i]["player_id"],
            str(next_games[i]),
            round(float(churn_probs[i]), 4),
            str(offers[i]),
            round(float(hr_scores[i]), 4),
            bool(hr_trajectories[i]),
            now,
        ))
    return results


def upsert_predictions(conn, predictions: list[tuple]):
    """Write predictions to RisingWave recommendations_tbl."""
    if not predictions:
        return
    cur = conn.cursor()
    # Delete existing rows and insert fresh (RisingWave doesn't support ON CONFLICT UPDATE on all versions)
    cur.execute("DELETE FROM recommendations_tbl")
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO recommendations_tbl
           (player_id, next_best_game, churn_probability, offer_sensitivity,
            high_roller_score, high_roller_trajectory, ts)
           VALUES %s""",
        predictions,
        page_size=200,
    )
    cur.close()


def main():
    print("Loading models...")
    models = load_models()

    print("Waiting for RisingWave features to be available...")
    conn = None
    for attempt in range(120):
        try:
            conn = get_rw_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM mv_player_features")
            count = cur.fetchone()[0]
            cur.close()
            if count > 0:
                print(f"  Found {count} player feature rows. Starting inference loop.")
                break
            conn.close()
            conn = None
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            if attempt % 10 == 0:
                print(f"  Attempt {attempt}: {e}")
        time.sleep(5)

    if conn is None:
        print("No features available yet, starting loop anyway...")
        conn = get_rw_connection()

    print(f"Inference loop running every {INFERENCE_INTERVAL_S}s")
    cycle = 0
    while True:
        try:
            df = fetch_features(conn)
            predictions = run_inference(models, df)
            upsert_predictions(conn, predictions)

            cycle += 1
            if cycle % 6 == 0:  # Log every ~60s
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Inference cycle {cycle}: {len(predictions)} predictions"
                )

        except psycopg2.OperationalError:
            print("RisingWave connection lost, reconnecting...")
            try:
                conn = get_rw_connection()
            except Exception:
                pass
        except Exception as e:
            print(f"Inference error: {e}")

        time.sleep(INFERENCE_INTERVAL_S)


if __name__ == "__main__":
    main()
