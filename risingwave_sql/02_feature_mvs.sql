-- ============================================================
-- 02: Feature Materialized Views — real-time ML feature store
-- ============================================================
--
-- House advantage (a.k.a. house edge) by game type — industry standards:
--   Slots:      7.50%   (class III video slots, blended)
--   Roulette:   5.26%   (double-zero American wheel)
--   Blackjack:  0.75%   (basic-strategy, 6-deck, H17)
--   Poker:      2.50%   (rake-equivalent vs. the house)
--
-- Theoretical Win (Theo / Theo Win) is the casino's expected profit from a
-- player's wagering, independent of short-term luck:
--     theo_win = Σ ( bet_amount × house_edge_of_that_game )
-- It is the single most important metric in casino marketing & loyalty —
-- reinvestment (offers, comps) is typically set as 25–40% of cumulative theo.
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_player_features CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_player_session_features CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_player_fnb_features CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_player_hotel_features CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_player_theo_cumulative CASCADE;

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
    -- Theoretical Win: per-bet house-edge contribution, summed over the window
    SUM(
        CASE game_type
            WHEN 'slots'     THEN bet_amount * 0.0750
            WHEN 'roulette'  THEN bet_amount * 0.0526
            WHEN 'blackjack' THEN bet_amount * 0.0075
            WHEN 'poker'     THEN bet_amount * 0.0250
            ELSE 0.0
        END
    ) AS theo_win_window,
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


-- Cumulative theoretical win per player (running sum across all windows).
-- This is the lifetime-value proxy used by the loyalty / comp system.
CREATE MATERIALIZED VIEW mv_player_theo_cumulative AS
SELECT
    player_id,
    SUM(theo_win_window)                    AS cumulative_theo_win,
    SUM(total_bet)                          AS cumulative_wagered,
    -- Effective (blended) house edge given the player's actual game mix
    CASE WHEN SUM(total_bet) > 0
         THEN SUM(theo_win_window) / SUM(total_bet)
         ELSE 0.0 END                       AS effective_house_edge
FROM mv_player_session_features
GROUP BY player_id;


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
    -- Theo win (this window + running total)
    g.theo_win_window,
    COALESCE(t.cumulative_theo_win, 0.0)    AS cumulative_theo_win,
    COALESCE(t.effective_house_edge, 0.0)   AS effective_house_edge,
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
    ON g.player_id = h.player_id AND g.window_start = h.window_start
LEFT JOIN mv_player_theo_cumulative t
    ON g.player_id = t.player_id;
