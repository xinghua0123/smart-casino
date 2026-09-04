-- ============================================================
-- 05: Floor Plan — live occupancy + starting-minimum recommendations
-- ============================================================
--
-- This is the "map view" of the casino floor. Each gaming event is tagged with a
-- physical table_id and an (x, y) position. We roll up per-table metrics in a
-- 1-minute TUMBLE window and then compare occupancy within each pit. When one
-- table is crowded while a peer is underused, the rules raise the crowded
-- table's starting minimum and lower the underused table's starting minimum to
-- spread demand. These recommendations apply only to baccarat and blackjack;
-- slot machines are load-monitoring only. Maximum limits are never changed.
-- The dashboard renders this as a 2D scatter colored by action_type.
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mv_table_recommendations CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_pit_live_load CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_table_live_load CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_table_latest CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_floor_latest_window CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_table_activity CASCADE;
DROP TABLE IF EXISTS tables_dim;


-- Static dimension table: one row per physical table on the floor.
-- Kept in sync with TABLE_LAYOUT in data_producer/producer.py.
CREATE TABLE tables_dim (
    table_id    VARCHAR PRIMARY KEY,
    game_type   VARCHAR,
    table_x     DOUBLE PRECISION,
    table_y     DOUBLE PRECISION,
    pit_group   VARCHAR,
    seat_capacity INT,
    limit_min   DOUBLE PRECISION,
    limit_max   DOUBLE PRECISION,
    minimum_floor DOUBLE PRECISION
);

-- Macau-style 3-game floor: slots, baccarat (hero), blackjack.
--   Slots (left):     8 standard               =  8 machines (casual volume)
--   Baccarat:         8 left + 8 main + 8 VIP  = 24 tables (the hero)
--   Blackjack (lower): 4 tables                =  4 tables (secondary pit)
-- Total 36 tables.
--
-- Baccarat left pit — 4x2 (entry-level tables, top-left)
INSERT INTO tables_dim VALUES
    ('bac_left_01', 'baccarat', 0.60, 7.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_02', 'baccarat', 1.80, 7.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_03', 'baccarat', 3.00, 7.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_04', 'baccarat', 4.20, 7.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_05', 'baccarat', 0.60, 8.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_06', 'baccarat', 1.80, 8.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_07', 'baccarat', 3.00, 8.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
    ('bac_left_08', 'baccarat', 4.20, 8.50, 'baccarat_left', 7, 50.0, 500.0, 25.0),
-- Standard slots — 4x2 (mid-left)
    ('slots_09', 'slots',  0.60, 3.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_10', 'slots',  1.80, 3.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_11', 'slots',  3.00, 3.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_12', 'slots',  4.20, 3.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_13', 'slots',  0.60, 4.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_14', 'slots',  1.80, 4.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_15', 'slots',  3.00, 4.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
    ('slots_16', 'slots',  4.20, 4.50, 'slots_standard', 1,   5.0,    25.0,   1.0),
-- ——— BACCARAT standard pit — 4x2, center of floor ———
    ('bac_01',   'baccarat',  5.80, 3.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_02',   'baccarat',  7.00, 3.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_03',   'baccarat',  8.20, 3.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_04',   'baccarat',  9.40, 3.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_05',   'baccarat',  5.80, 4.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_06',   'baccarat',  7.00, 4.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_07',   'baccarat',  8.20, 4.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
    ('bac_08',   'baccarat',  9.40, 4.50, 'baccarat_main', 7, 100.0,   500.0,  50.0),
-- ——— BACCARAT VIP — 4x2 high-limit room, top-center ———
    ('bac_vip_01','baccarat', 5.80, 7.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_02','baccarat', 7.00, 7.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_03','baccarat', 8.20, 7.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_04','baccarat', 9.40, 7.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_05','baccarat', 5.80, 8.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_06','baccarat', 7.00, 8.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_07','baccarat', 8.20, 8.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
    ('bac_vip_08','baccarat', 9.40, 8.50, 'baccarat_vip', 7, 500.0, 10000.0, 100.0),
-- Blackjack — single 2x2 pit below the main baccarat pit
    ('bj_01',    'blackjack', 7.00, 0.60, 'blackjack_main', 7, 25.0, 500.0, 10.0),
    ('bj_02',    'blackjack', 8.20, 0.60, 'blackjack_main', 7, 25.0, 500.0, 10.0),
    ('bj_03',    'blackjack', 7.00, 1.60, 'blackjack_main', 7, 25.0, 500.0, 10.0),
    ('bj_04',    'blackjack', 8.20, 1.60, 'blackjack_main', 7, 25.0, 500.0, 10.0);


-- 1-minute tumbling activity gives the dashboard a responsive demand signal.
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
            WHEN 'blackjack' THEN bet_amount * 0.0075
            ELSE 0.0
        END
    )                              AS theo_win_window
FROM TUMBLE(gaming_events_src, ts, INTERVAL '1 MINUTE')
WHERE table_id IS NOT NULL
GROUP BY table_id, window_start, window_end;


-- Use one floor-wide clock. A table with no activity in this exact window must
-- be zero rather than retaining its own last non-empty window indefinitely.
CREATE MATERIALIZED VIEW mv_floor_latest_window AS
SELECT MAX(window_start) AS window_start
FROM mv_table_activity;


-- Active tables in the latest floor-wide window.
CREATE MATERIALIZED VIEW mv_table_latest AS
SELECT t.*
FROM mv_table_activity t
JOIN mv_floor_latest_window w
    ON t.window_start = w.window_start;


-- All physical tables, including zero-activity tables, enriched with occupancy.
CREATE MATERIALIZED VIEW mv_table_live_load AS
SELECT
    d.table_id,
    d.game_type,
    d.table_x,
    d.table_y,
    d.pit_group,
    d.seat_capacity,
    d.limit_min,
    d.limit_max,
    d.minimum_floor,
    COALESCE(a.active_players, 0)        AS active_players,
    COALESCE(a.bets, 0)                  AS bets,
    COALESCE(a.avg_bet, 0.0)             AS avg_bet,
    COALESCE(a.max_bet, 0.0)             AS max_bet,
    COALESCE(a.total_bet, 0.0)           AS total_bet,
    COALESCE(a.theo_win_window, 0.0)     AS theo_win_window,
    a.window_start,
    CASE WHEN d.seat_capacity > 0
         THEN LEAST(
             COALESCE(a.active_players, 0)::DOUBLE PRECISION / d.seat_capacity,
             1.0
         )
         ELSE 0.0 END                    AS occupancy_rate
FROM tables_dim d
LEFT JOIN mv_table_latest a
    ON d.table_id = a.table_id;


-- Weighted occupancy and the hottest/coldest table for each comparable pit.
CREATE MATERIALIZED VIEW mv_pit_live_load AS
SELECT
    pit_group,
    SUM(active_players)::DOUBLE PRECISION / SUM(seat_capacity) AS pit_occupancy_rate,
    MIN(occupancy_rate) AS pit_min_occupancy_rate,
    MAX(occupancy_rate) AS pit_max_occupancy_rate
FROM mv_table_live_load
GROUP BY pit_group;


-- Starting-minimum recommendations for baccarat and blackjack only. The
-- 85%/35% pair detects a genuine within-pit imbalance, so changes redirect
-- demand instead of moving every table's minimum in the same direction.
CREATE MATERIALIZED VIEW mv_table_recommendations AS
SELECT
    l.table_id,
    l.game_type,
    l.table_x,
    l.table_y,
    l.pit_group,
    l.seat_capacity,
    l.limit_min,
    l.limit_max,
    l.active_players,
    l.bets,
    l.avg_bet,
    l.max_bet,
    l.total_bet,
    l.theo_win_window,
    l.window_start,
    l.occupancy_rate,
    p.pit_occupancy_rate,
    p.pit_min_occupancy_rate,
    p.pit_max_occupancy_rate,
    CASE
        WHEN l.game_type = 'slots' THEN 'MONITOR_ONLY'
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate >= 0.85
             AND p.pit_min_occupancy_rate <= 0.35
             AND l.limit_min < l.limit_max
            THEN 'RAISE_MINIMUM'
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate <= 0.35
             AND p.pit_max_occupancy_rate >= 0.85
             AND l.limit_min > l.minimum_floor
            THEN 'LOWER_MINIMUM'
        WHEN l.occupancy_rate >= 0.85 THEN 'BUSY'
        WHEN l.occupancy_rate <= 0.20 THEN 'IDLE'
        ELSE 'BALANCED'
    END                                  AS action_type,
    -- Move by one familiar denomination step; never exceed the table maximum or
    -- go below the configured floor for its pit.
    CASE
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate >= 0.85
             AND p.pit_min_occupancy_rate <= 0.35
             AND l.limit_min < l.limit_max
            THEN LEAST(l.limit_max, CASE
                WHEN l.limit_min < 5.0   THEN 2.0
                WHEN l.limit_min < 25.0  THEN 10.0
                WHEN l.limit_min < 50.0  THEN 50.0
                WHEN l.limit_min < 100.0 THEN 100.0
                WHEN l.limit_min < 500.0 THEN 200.0
                WHEN l.limit_min < 1000.0 THEN 1000.0
                ELSE ROUND((l.limit_min * 1.5)::NUMERIC, 0)
            END)
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate <= 0.35
             AND p.pit_max_occupancy_rate >= 0.85
             AND l.limit_min > l.minimum_floor
            THEN GREATEST(l.minimum_floor, CASE
                WHEN l.limit_min <= 5.0   THEN 2.0
                WHEN l.limit_min <= 25.0  THEN 10.0
                WHEN l.limit_min <= 50.0  THEN 25.0
                WHEN l.limit_min <= 100.0 THEN 50.0
                WHEN l.limit_min <= 500.0 THEN 300.0
                ELSE ROUND((l.limit_min * 0.75)::NUMERIC, 0)
            END)
        ELSE ROUND(l.limit_min::NUMERIC, 0)
    END                                  AS suggested_limit_min,
    ROUND(l.limit_max::NUMERIC, 0)        AS suggested_limit_max,
    CASE
        WHEN l.game_type = 'slots'
            THEN 'Slot machine load is monitored only; no starting-minimum recommendation applies.'
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate >= 0.85
             AND p.pit_min_occupancy_rate <= 0.35
            THEN 'Crowded table with spare capacity in the same pit; raise the starting minimum to redirect demand.'
        WHEN l.game_type IN ('baccarat', 'blackjack')
             AND l.occupancy_rate <= 0.35
             AND p.pit_max_occupancy_rate >= 0.85
            THEN 'Underused table beside a crowded peer; lower the starting minimum to attract overflow.'
        WHEN l.occupancy_rate >= 0.85
            THEN 'Demand is high across this pit; monitor capacity before changing one table in isolation.'
        WHEN l.occupancy_rate <= 0.20
            THEN 'This table is underused, but there is no crowded peer to redirect yet.'
        ELSE 'Occupancy is within the balanced operating range.'
    END                                  AS recommendation_reason
FROM mv_table_live_load l
JOIN mv_pit_live_load p
    ON l.pit_group = p.pit_group;
