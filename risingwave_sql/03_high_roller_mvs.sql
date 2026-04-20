-- ============================================================
-- 03: High Roller Lookalike Detection
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_player_high_roller_similarity CASCADE;

-- Per-player similarity to high-roller profile
-- Uses a weighted score based on behavioral signals that correlate with high rollers.
-- Reference thresholds derived from the high_roller archetype in the data producer:
--   avg_bet ~$200-2000, pct_blackjack ~0.45, pct_poker ~0.30,
--   fnb_spend ~$80-300, hotel_spend >0, category_diversity ~3, spend_per_min high,
--   cumulative_theo_win high (key casino-value signal)
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
    theo_win_window,
    cumulative_theo_win,
    effective_house_edge,
    -- Weighted similarity score (0-1) based on high-roller behavioral signals.
    -- Re-balanced to include cumulative_theo_win — the casino's best proxy for
    -- "how much this player is actually worth to us".
    (
        0.20 * LEAST(avg_bet / 500.0, 1.0)
      + 0.12 * LEAST(pct_blackjack / 0.45, 1.0)
      + 0.12 * LEAST(pct_poker / 0.30, 1.0)
      + 0.08 * LEAST(fnb_spend / 150.0, 1.0)
      + 0.08 * LEAST(hotel_spend / 200.0, 1.0)
      + 0.08 * LEAST(category_diversity::DOUBLE PRECISION / 3.0, 1.0)
      + 0.12 * LEAST(spend_per_minute / 50.0, 1.0)
      + 0.20 * LEAST(cumulative_theo_win / 1500.0, 1.0)
    ) AS high_roller_similarity
FROM mv_player_features;
