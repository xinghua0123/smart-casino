# Smart Casino Floor — Real-Time Gaming Recommendation with ML

A demo showing RisingWave as a real-time ML feature store for personalized gaming recommendations, churn prediction, and high-roller lookalike detection.

## Architecture

```
Kafka Topics                    RisingWave                         ML Service (Python)
─────────────                   ──────────                         ───────────────────
gaming_events ──┐
fnb_events    ──┼──► Source Tables ──► Feature MVs ──────────────► scikit-learn Models
hotel_events  ──┘         │              │                              │
                          │         (session stats,                (predictions)
                          │          game preferences,                  │
                          │          HR similarity)                     ▼
                          │                                    Kafka: recommendations
                          │         Recommendation MV ◄────────────────┘
                          │              │
                          ▼              ▼
                     Business Rules   Streamlit Dashboard (localhost:8501)
```

## Components

| Component | Description |
|-----------|-------------|
| **data_producer** | Generates realistic event streams for 200 simulated players across 4 archetypes (casual, regular, high_roller, emerging) |
| **risingwave_sql** | 4 SQL files: Kafka sources, feature MVs (5-min windows), high-roller similarity scoring, recommendation delivery with business rules |
| **ml_service** | Trains 4 models on synthetic data (next-game, churn, offer sensitivity, HR trajectory), then runs inference loop querying RisingWave every 10s |
| **dashboard** | Streamlit app with KPI metrics, High Roller Radar scatter plot, recommendation table, game/tier distributions |

## ML Models

1. **Next-Best-Game** (Random Forest) — recommends cross-sell game based on current play pattern
2. **Churn Probability** (Gradient Boosted Regressor) — predicts likelihood of player leaving
3. **Offer Sensitivity** (Random Forest) — which reward type converts best (free play, F&B voucher, hotel upgrade, cashback)
4. **High-Roller Trajectory** (Gradient Boosted Classifier) — is this player on track to become a high roller?

## Key RisingWave Features Demonstrated

- `TUMBLE()` windows for real-time feature engineering
- Materialized views as a streaming feature store
- Cross-source JOIN for multi-category player profiles
- High-roller reference profile (continuously updated centroid)
- Cosine-like similarity scoring in SQL
- ML prediction feedback loop (predictions ingested back as a source)
- Business rule layer on top of ML predictions

## Quick Start

```bash
cd smart-casino-floor

# Start everything
docker compose up --build

# Wait ~30s for services to initialize, then open:
#   Dashboard:  http://localhost:8501
#   RisingWave: psql -h localhost -p 4566 -U root -d dev
```

## Explore the Data

```sql
-- Player features (real-time)
SELECT * FROM mv_player_features LIMIT 10;

-- High roller radar
SELECT player_id, high_roller_similarity, avg_bet, spend_per_minute
FROM mv_high_roller_radar ORDER BY high_roller_similarity DESC LIMIT 10;

-- ML recommendations with business rules
SELECT player_id, next_best_game, action_type, churn_probability, high_roller_trajectory
FROM mv_actionable_recommendations
WHERE action_type = 'VIP_UPGRADE_CANDIDATE';

-- High roller reference profile
SELECT * FROM mv_high_roller_reference_profile;
```

## Cleanup

```bash
docker compose down -v
```
