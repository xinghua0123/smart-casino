-- ============================================================
-- 02: Feature Materialized Views — real-time ML feature store
-- ============================================================

-- 5-minute tumbling window: per-player session-level features
CREATE MATERIALIZED VIEW mv_player_session_features AS
SELECT
    player_id,
    window_start,
    window_end,
    COUNT(*)                                    AS games_played,
    AVG(bet_amount)                             AS avg_bet,
    SUM(bet_amount)                             AS total_bet,
    SUM(payout)                                 AS total_payout,
    AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)   AS win_rate,
    -- Game type distribution
    AVG(CASE WHEN game_type = 'slots'     THEN 1.0 ELSE 0.0 END) AS pct_slots,
    AVG(CASE WHEN game_type = 'roulette'  THEN 1.0 ELSE 0.0 END) AS pct_roulette,
    AVG(CASE WHEN game_type = 'blackjack' THEN 1.0 ELSE 0.0 END) AS pct_blackjack,
    AVG(CASE WHEN game_type = 'poker'     THEN 1.0 ELSE 0.0 END) AS pct_poker,
    MAX(total_gaming_spend)                     AS cumulative_gaming_spend,
    MAX(tier)                                   AS tier,
    MAX(archetype)                              AS archetype
FROM TUMBLE(gaming_events_src, ts, INTERVAL '5 MINUTES')
GROUP BY player_id, window_start, window_end;


-- Rolling 30-min F&B spend per player
CREATE MATERIALIZED VIEW mv_player_fnb_features AS
SELECT
    player_id,
    window_start,
    window_end,
    COUNT(*)            AS fnb_orders,
    SUM(spend_amount)   AS fnb_spend,
    MAX(total_fnb_spend) AS cumulative_fnb_spend
FROM TUMBLE(fnb_events_src, ts, INTERVAL '5 MINUTES')
GROUP BY player_id, window_start, window_end;


-- Rolling hotel activity per player
CREATE MATERIALIZED VIEW mv_player_hotel_features AS
SELECT
    player_id,
    window_start,
    window_end,
    COUNT(*)                AS hotel_events,
    SUM(charge_amount)      AS hotel_spend,
    COUNT(DISTINCT action)  AS hotel_action_types
FROM TUMBLE(hotel_events_src, ts, INTERVAL '5 MINUTES')
GROUP BY player_id, window_start, window_end;


-- Combined cross-category player profile (latest 5-min window)
-- This is the main feature table the ML service queries
CREATE MATERIALIZED VIEW mv_player_features AS
SELECT
    g.player_id,
    g.window_start,
    g.window_end,
    -- Gaming features
    g.games_played,
    g.avg_bet,
    g.total_bet,
    g.total_payout,
    g.win_rate,
    g.pct_slots,
    g.pct_roulette,
    g.pct_blackjack,
    g.pct_poker,
    g.cumulative_gaming_spend,
    g.tier,
    g.archetype,
    -- F&B features (0 if no activity)
    COALESCE(f.fnb_orders, 0)           AS fnb_orders,
    COALESCE(f.fnb_spend, 0.0)          AS fnb_spend,
    COALESCE(f.cumulative_fnb_spend, 0.0) AS cumulative_fnb_spend,
    -- Hotel features (0 if no activity)
    COALESCE(h.hotel_events, 0)         AS hotel_events,
    COALESCE(h.hotel_spend, 0.0)        AS hotel_spend,
    COALESCE(h.hotel_action_types, 0)   AS hotel_action_types,
    -- Cross-category diversity (how many categories active)
    (CASE WHEN g.games_played > 0 THEN 1 ELSE 0 END
     + CASE WHEN COALESCE(f.fnb_orders, 0) > 0 THEN 1 ELSE 0 END
     + CASE WHEN COALESCE(h.hotel_events, 0) > 0 THEN 1 ELSE 0 END
    ) AS category_diversity,
    -- Spend velocity (total spend / 5 min)
    (g.total_bet + COALESCE(f.fnb_spend, 0.0) + COALESCE(h.hotel_spend, 0.0)) / 5.0
        AS spend_per_minute
FROM mv_player_session_features g
LEFT JOIN mv_player_fnb_features f
    ON g.player_id = f.player_id AND g.window_start = f.window_start
LEFT JOIN mv_player_hotel_features h
    ON g.player_id = h.player_id AND g.window_start = h.window_start;
