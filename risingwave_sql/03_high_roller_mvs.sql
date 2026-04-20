-- ============================================================
-- 03: High Roller Lookalike Detection
-- ============================================================

-- Per-player similarity to high-roller profile
-- Uses a weighted score based on behavioral signals that correlate with high rollers.
-- Reference thresholds derived from the high_roller archetype in the data producer:
--   avg_bet ~$200-2000, pct_blackjack ~0.45, pct_poker ~0.30,
--   fnb_spend ~$80-300, hotel_spend >0, category_diversity ~3, spend_per_min high
CREATE MATERIALIZED VIEW mv_player_high_roller_similarity AS
SELECT
    player_id,
    window_start,
    tier,
    archetype,
    avg_bet,
    cumulative_gaming_spend,
    cumulative_fnb_spend,
    spend_per_minute,
    category_diversity,
    -- Weighted similarity score (0-1) based on high-roller behavioral signals
    (
        0.25 * LEAST(avg_bet / 500.0, 1.0)
      + 0.15 * LEAST(pct_blackjack / 0.45, 1.0)
      + 0.15 * LEAST(pct_poker / 0.30, 1.0)
      + 0.10 * LEAST(fnb_spend / 150.0, 1.0)
      + 0.10 * LEAST(hotel_spend / 200.0, 1.0)
      + 0.10 * LEAST(category_diversity::DOUBLE PRECISION / 3.0, 1.0)
      + 0.15 * LEAST(spend_per_minute / 50.0, 1.0)
    ) AS high_roller_similarity
FROM mv_player_features;
