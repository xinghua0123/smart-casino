-- ============================================================
-- 04: Recommendation Delivery — join ML predictions with business rules
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_dashboard_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_theo_by_tier CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_high_roller_radar CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_actionable_recommendations CASCADE;
DROP TABLE IF EXISTS recommendations_tbl;

-- Table for ML predictions (inference loop upserts via SQL INSERT)
CREATE TABLE recommendations_tbl (
    player_id               VARCHAR PRIMARY KEY,
    next_best_game          VARCHAR,
    churn_probability       DOUBLE PRECISION,
    offer_sensitivity       VARCHAR,
    high_roller_score       DOUBLE PRECISION,
    high_roller_trajectory  BOOLEAN,
    ts                      TIMESTAMPTZ
);


-- Actionable recommendations: combine ML predictions with rule-based logic.
-- Offer value is now scaled off TheoWin (industry-standard reinvestment %)
-- instead of bare avg_bet, so comps match the player's true casino value.
CREATE MATERIALIZED VIEW mv_actionable_recommendations AS
SELECT
    r.player_id,
    r.next_best_game,
    r.churn_probability,
    r.offer_sensitivity,
    r.high_roller_score,
    r.high_roller_trajectory,
    f.tier,
    f.avg_bet,
    f.cumulative_gaming_spend,
    f.theo_win_window,
    f.cumulative_theo_win,
    f.effective_house_edge,
    -- Business rule: escalate offer for high-churn-risk silver+ players
    CASE
        WHEN r.churn_probability > 0.45 AND f.tier IN ('silver', 'gold', 'platinum', 'diamond')
            THEN 'URGENT_RETENTION'
        WHEN r.high_roller_trajectory AND f.tier NOT IN ('platinum', 'diamond')
            THEN 'VIP_UPGRADE_CANDIDATE'
        WHEN r.churn_probability > 0.38
            THEN 'RETENTION_OFFER'
        ELSE 'STANDARD_RECOMMENDATION'
    END AS action_type,
    -- Offer value = reinvestment % of cumulative theo, tiered by action type.
    --   URGENT_RETENTION:      40% of theo  (aggressive save)
    --   VIP_UPGRADE_CANDIDATE: 35% of theo  (grow them)
    --   RETENTION_OFFER:       25% of theo  (standard reinvestment)
    --   STANDARD_RECOMMENDATION: 15% of theo (baseline loyalty)
    -- Floor on avg_bet ensures new players with little theo history still get a sensible offer.
    CASE
        WHEN r.churn_probability > 0.45 AND f.tier IN ('silver', 'gold', 'platinum', 'diamond')
            THEN ROUND(GREATEST(f.cumulative_theo_win * 0.40, f.avg_bet * 2.0))
        WHEN r.high_roller_trajectory
            THEN ROUND(GREATEST(f.cumulative_theo_win * 0.35, f.avg_bet * 1.5))
        WHEN r.churn_probability > 0.38
            THEN ROUND(GREATEST(f.cumulative_theo_win * 0.25, f.avg_bet * 1.0))
        ELSE ROUND(GREATEST(f.cumulative_theo_win * 0.15, f.avg_bet * 0.5))
    END AS offer_value,
    r.ts AS recommendation_ts
FROM recommendations_tbl r
LEFT JOIN mv_player_features f
    ON r.player_id = f.player_id;


-- ============================================================
-- Monitoring / Dashboard views
-- ============================================================

-- High Roller Radar: top emerging candidates
-- Keep only the latest row per player so candidate rankings are unique by player_id.
CREATE MATERIALIZED VIEW mv_high_roller_radar AS
SELECT
    s.player_id,
    s.tier,
    s.archetype,
    s.high_roller_similarity,
    s.avg_bet,
    s.cumulative_gaming_spend,
    s.cumulative_fnb_spend,
    s.spend_per_minute,
    s.category_diversity,
    s.theo_win_window,
    s.cumulative_theo_win,
    s.effective_house_edge,
    s.window_start
FROM mv_player_high_roller_similarity AS s
JOIN (
    SELECT
        player_id,
        MAX(window_start) AS latest_window_start
    FROM mv_player_high_roller_similarity
    WHERE archetype != 'high_roller'
      AND high_roller_similarity > 0.4
    GROUP BY player_id
) AS latest
    ON s.player_id = latest.player_id
   AND s.window_start = latest.latest_window_start
WHERE s.archetype != 'high_roller'
  AND s.high_roller_similarity > 0.4;


-- Theo-win breakdown by tier (for the dashboard bar chart)
CREATE MATERIALIZED VIEW mv_theo_by_tier AS
SELECT
    f.tier,
    COUNT(DISTINCT f.player_id)           AS players,
    SUM(t.cumulative_theo_win)            AS total_theo_win,
    AVG(t.cumulative_theo_win)            AS avg_theo_per_player,
    AVG(t.effective_house_edge)           AS avg_effective_house_edge
FROM mv_player_features f
JOIN mv_player_theo_cumulative t ON f.player_id = t.player_id
GROUP BY f.tier;


-- Overall stats for dashboard
CREATE MATERIALIZED VIEW mv_dashboard_stats AS
SELECT
    COUNT(DISTINCT player_id)   AS active_players,
    AVG(avg_bet)                AS avg_bet_all,
    AVG(spend_per_minute)       AS avg_spend_per_min,
    SUM(total_bet)              AS total_wagered,
    SUM(theo_win_window)        AS theo_win_window,
    AVG(effective_house_edge)   AS avg_house_edge
FROM mv_player_features;
