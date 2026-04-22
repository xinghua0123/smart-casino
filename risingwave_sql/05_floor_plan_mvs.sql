-- ============================================================
-- 05: Floor Plan — per-table activity + raise/lower-limit recommendations
-- ============================================================
--
-- This is the "map view" of the casino floor. Each gaming event is tagged with a
-- physical table_id and an (x, y) position. We roll up per-table metrics in a
-- 5-minute TUMBLE window and then apply business rules to suggest whether each
-- table should RAISE_LIMIT (packed + betting near the ceiling), LOWER_LIMIT
-- (cold + few players, limit too high for entry-level traffic), HOT, COLD, or HOLD.
-- The dashboard renders this as a 2D scatter colored by action_type.
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_table_recommendations CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_table_latest CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_table_activity CASCADE;
DROP TABLE IF EXISTS tables_dim;


-- Static dimension table: one row per physical table on the floor.
-- Kept in sync with TABLE_LAYOUT in data_producer/producer.py.
CREATE TABLE tables_dim (
    table_id    VARCHAR PRIMARY KEY,
    game_type   VARCHAR,
    table_x     DOUBLE PRECISION,
    table_y     DOUBLE PRECISION,
    limit_min   DOUBLE PRECISION,
    limit_max   DOUBLE PRECISION
);

-- Macau-style floor. Baccarat is the hero: a large standard pit in the
-- middle-right plus a VIP room on the top-right. Slots dominate the left
-- (casual volume). Blackjack is a secondary pit. Roulette & poker are
-- intentionally minor (2 tables each) — they do not drive theo on an
-- Asian floor.
--
-- Penny slots — 4x2 (low limits, near entrance)
INSERT INTO tables_dim VALUES
    ('slots_01', 'slots',  0.60, 7.80,  1.0,  10.0),
    ('slots_02', 'slots',  1.80, 7.80,  1.0,  10.0),
    ('slots_03', 'slots',  3.00, 7.80,  1.0,  10.0),
    ('slots_04', 'slots',  4.20, 7.80,  1.0,  10.0),
    ('slots_05', 'slots',  0.60, 8.80,  1.0,  10.0),
    ('slots_06', 'slots',  1.80, 8.80,  1.0,  10.0),
    ('slots_07', 'slots',  3.00, 8.80,  1.0,  10.0),
    ('slots_08', 'slots',  4.20, 8.80,  1.0,  10.0),
-- Standard slots — 4x2
    ('slots_09', 'slots',  0.60, 3.50,  5.0,  25.0),
    ('slots_10', 'slots',  1.80, 3.50,  5.0,  25.0),
    ('slots_11', 'slots',  3.00, 3.50,  5.0,  25.0),
    ('slots_12', 'slots',  4.20, 3.50,  5.0,  25.0),
    ('slots_13', 'slots',  0.60, 4.50,  5.0,  25.0),
    ('slots_14', 'slots',  1.80, 4.50,  5.0,  25.0),
    ('slots_15', 'slots',  3.00, 4.50,  5.0,  25.0),
    ('slots_16', 'slots',  4.20, 4.50,  5.0,  25.0),
-- Blackjack standard pit — 2x2 (trimmed from 2x3 to free middle-floor for baccarat)
    ('bj_01',    'blackjack', 5.80, 3.00,  25.0, 500.0),
    ('bj_02',    'blackjack', 7.00, 3.00,  25.0, 500.0),
    ('bj_03',    'blackjack', 5.80, 4.50,  25.0, 500.0),
    ('bj_04',    'blackjack', 7.00, 4.50,  25.0, 500.0),
-- High-limit blackjack — 2 tables at top
    ('bj_05',    'blackjack', 5.80, 7.50, 500.0, 2000.0),
    ('bj_06',    'blackjack', 7.00, 7.50, 500.0, 2000.0),
-- ——— BACCARAT (the hero) — standard 4x2 pit, center-right ———
    ('bac_01',   'baccarat',  8.40, 3.50, 100.0,  500.0),
    ('bac_02',   'baccarat',  9.60, 3.50, 100.0,  500.0),
    ('bac_03',   'baccarat', 10.80, 3.50, 100.0,  500.0),
    ('bac_04',   'baccarat', 12.00, 3.50, 100.0,  500.0),
    ('bac_05',   'baccarat',  8.40, 4.50, 100.0,  500.0),
    ('bac_06',   'baccarat',  9.60, 4.50, 100.0,  500.0),
    ('bac_07',   'baccarat', 10.80, 4.50, 100.0,  500.0),
    ('bac_08',   'baccarat', 12.00, 4.50, 100.0,  500.0),
-- ——— BACCARAT VIP — 2 high-limit tables (velvet rope) ———
    ('bac_vip_01','baccarat', 10.80, 7.50,  500.0, 10000.0),
    ('bac_vip_02','baccarat', 12.00, 7.50,  500.0, 10000.0),
-- Roulette — 2 wheels only (small share on a Macau floor) — front row
    ('rou_01',   'roulette',  6.20, 1.00, 10.0, 200.0),
    ('rou_02',   'roulette',  7.40, 1.00, 10.0, 200.0),
-- Poker — 2 tables only (small share) — front-right
    ('pok_01',   'poker',     9.00, 1.00, 50.0, 1000.0),
    ('pok_02',   'poker',    10.20, 1.00, 50.0, 1000.0);


-- 5-min tumbling per-table activity
CREATE MATERIALIZED VIEW mv_table_activity AS
SELECT
    table_id,
    window_start,
    window_end,
    MAX(game_type)                 AS game_type,
    MAX(table_x)                   AS table_x,
    MAX(table_y)                   AS table_y,
    MAX(limit_min)                 AS limit_min,
    MAX(limit_max)                 AS limit_max,
    COUNT(DISTINCT player_id)      AS active_players,
    COUNT(*)                       AS bets,
    AVG(bet_amount)                AS avg_bet,
    SUM(bet_amount)                AS total_bet,
    MAX(bet_amount)                AS max_bet,
    MIN(bet_amount)                AS min_bet,
    -- Theo Win on this table's window (same formula as player-side)
    SUM(
        CASE game_type
            WHEN 'slots'     THEN bet_amount * 0.0750
            WHEN 'baccarat'  THEN bet_amount * 0.0115
            WHEN 'roulette'  THEN bet_amount * 0.0526
            WHEN 'blackjack' THEN bet_amount * 0.0075
            WHEN 'poker'     THEN bet_amount * 0.0250
            ELSE 0.0
        END
    )                              AS theo_win_window
FROM TUMBLE(gaming_events_src, ts, INTERVAL '5 MINUTES')
WHERE table_id IS NOT NULL
GROUP BY table_id, window_start, window_end;


-- Latest window per table (for "what's happening right now")
CREATE MATERIALIZED VIEW mv_table_latest AS
SELECT t.*
FROM mv_table_activity t
JOIN (
    SELECT table_id, MAX(window_start) AS latest_window_start
    FROM mv_table_activity
    GROUP BY table_id
) AS lt
    ON t.table_id = lt.table_id
   AND t.window_start = lt.latest_window_start;


-- Floor-plan recommendations: LEFT JOIN static dim with live activity so cold
-- tables (no events in the latest window) still appear on the map.
CREATE MATERIALIZED VIEW mv_table_recommendations AS
SELECT
    d.table_id,
    d.game_type,
    d.table_x,
    d.table_y,
    d.limit_min,
    d.limit_max,
    COALESCE(a.active_players, 0)        AS active_players,
    COALESCE(a.bets, 0)                  AS bets,
    COALESCE(a.avg_bet, 0.0)             AS avg_bet,
    COALESCE(a.max_bet, 0.0)             AS max_bet,
    COALESCE(a.total_bet, 0.0)           AS total_bet,
    COALESCE(a.theo_win_window, 0.0)     AS theo_win_window,
    a.window_start,
    -- ---- Business rule layer -------------------------------------------------
    -- RAISE_LIMIT: packed (≥3 players) AND average bet near the ceiling (≥70% of max)
    --   → market is willing to pay; pushing up captures more theo.
    -- LOWER_LIMIT: cold (≤1 player, few bets) AND the min limit is above entry-level
    --   → limit is scaring off casual traffic; drop it to attract foot traffic.
    -- HOT: healthy occupancy + strong theo_win contribution this window.
    -- COLD: nearly no activity this window.
    -- HOLD: everything else — leave it alone.
    CASE
        WHEN COALESCE(a.active_players, 0) >= 3
             AND COALESCE(a.avg_bet, 0.0) >= d.limit_max * 0.70
            THEN 'RAISE_LIMIT'
        WHEN COALESCE(a.active_players, 0) <= 1
             AND COALESCE(a.bets, 0) < 8
             AND d.limit_min > 10.0
            THEN 'LOWER_LIMIT'
        WHEN COALESCE(a.active_players, 0) >= 2
             AND COALESCE(a.theo_win_window, 0.0) >= 300.0
            THEN 'HOT'
        WHEN COALESCE(a.bets, 0) < 3
            THEN 'COLD'
        ELSE 'HOLD'
    END                                  AS action_type,
    -- Suggested new limit range (illustrative — the pit boss decides)
    CASE
        WHEN COALESCE(a.active_players, 0) >= 3
             AND COALESCE(a.avg_bet, 0.0) >= d.limit_max * 0.70
            THEN ROUND((d.limit_min * 1.5)::numeric, 0)
        WHEN COALESCE(a.active_players, 0) <= 1
             AND COALESCE(a.bets, 0) < 8
             AND d.limit_min > 10.0
            THEN GREATEST(ROUND((d.limit_min * 0.5)::numeric, 0), 1.0)
        ELSE ROUND(d.limit_min::numeric, 0)
    END                                  AS suggested_limit_min,
    CASE
        WHEN COALESCE(a.active_players, 0) >= 3
             AND COALESCE(a.avg_bet, 0.0) >= d.limit_max * 0.70
            THEN ROUND((d.limit_max * 1.5)::numeric, 0)
        WHEN COALESCE(a.active_players, 0) <= 1
             AND COALESCE(a.bets, 0) < 8
             AND d.limit_min > 10.0
            THEN ROUND(d.limit_max::numeric, 0)
        ELSE ROUND(d.limit_max::numeric, 0)
    END                                  AS suggested_limit_max
FROM tables_dim d
LEFT JOIN mv_table_latest a
    ON d.table_id = a.table_id;
