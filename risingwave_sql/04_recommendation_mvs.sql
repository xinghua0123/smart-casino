-- ============================================================
-- 04: Recommendation Delivery — join ML predictions with business rules
-- ============================================================

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


-- Actionable recommendations: combine ML predictions with rule-based logic
CREATE MATERIALIZED VIEW mv_actionable_recommendations AS
SELECT
    r.player_id,
    r.next_best_game,
    r.churn_probability,
    r.offer_sensitivity,
    r.high_roller_score,
    r.high_roller_trajectory,
    s.tier,
    s.avg_bet,
    s.cumulative_gaming_spend,
    -- Business rule: escalate offer for high-churn-risk gold+ players
    CASE
        WHEN r.churn_probability > 0.45 AND s.tier IN ('silver', 'gold', 'platinum', 'diamond')
            THEN 'URGENT_RETENTION'
        WHEN r.high_roller_trajectory AND s.tier NOT IN ('platinum', 'diamond')
            THEN 'VIP_UPGRADE_CANDIDATE'
        WHEN r.churn_probability > 0.38
            THEN 'RETENTION_OFFER'
        ELSE 'STANDARD_RECOMMENDATION'
    END AS action_type,
    -- Offer value scales with player value and churn risk
    CASE
        WHEN r.churn_probability > 0.45 AND s.tier IN ('silver', 'gold', 'platinum', 'diamond')
            THEN ROUND(s.avg_bet * 2.0)
        WHEN r.high_roller_trajectory
            THEN ROUND(s.avg_bet * 1.5)
        WHEN r.churn_probability > 0.38
            THEN ROUND(s.avg_bet * 1.0)
        ELSE ROUND(s.avg_bet * 0.5)
    END AS offer_value,
    r.ts AS recommendation_ts
FROM recommendations_tbl r
LEFT JOIN mv_player_session_features s
    ON r.player_id = s.player_id;


-- ============================================================
-- Monitoring / Dashboard views
-- ============================================================

-- High Roller Radar: top emerging candidates
CREATE MATERIALIZED VIEW mv_high_roller_radar AS
SELECT
    player_id,
    tier,
    archetype,
    high_roller_similarity,
    avg_bet,
    cumulative_gaming_spend,
    cumulative_fnb_spend,
    spend_per_minute,
    category_diversity,
    window_start
FROM mv_player_high_roller_similarity
WHERE archetype != 'high_roller'
  AND high_roller_similarity > 0.4;


-- Overall stats for dashboard
CREATE MATERIALIZED VIEW mv_dashboard_stats AS
SELECT
    COUNT(DISTINCT player_id) AS active_players,
    AVG(avg_bet) AS avg_bet_all,
    AVG(spend_per_minute) AS avg_spend_per_min,
    SUM(total_bet) AS total_wagered
FROM mv_player_features;
