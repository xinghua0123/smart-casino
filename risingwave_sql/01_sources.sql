-- ============================================================
-- 01: Kafka Sources — ingest gaming, F&B, and hotel event streams
-- ============================================================

CREATE SOURCE gaming_events_src (
    event_id        VARCHAR,
    event_type      VARCHAR,
    player_id       VARCHAR,
    tier            VARCHAR,
    archetype       VARCHAR,
    game_type       VARCHAR,
    -- Physical-floor fields (for Floor Plan view)
    table_id        VARCHAR,
    table_x         DOUBLE PRECISION,
    table_y         DOUBLE PRECISION,
    limit_min       DOUBLE PRECISION,
    limit_max       DOUBLE PRECISION,
    bet_amount      DOUBLE PRECISION,
    payout          DOUBLE PRECISION,
    won             BOOLEAN,
    session_count   INT,
    total_gaming_spend DOUBLE PRECISION,
    ts              TIMESTAMPTZ
) WITH (
    connector = 'kafka',
    topic = 'gaming_events',
    properties.bootstrap.server = 'kafka:9092',
    scan.startup.mode = 'latest'
) FORMAT PLAIN ENCODE JSON;

CREATE SOURCE fnb_events_src (
    event_id        VARCHAR,
    event_type      VARCHAR,
    player_id       VARCHAR,
    tier            VARCHAR,
    item            VARCHAR,
    spend_amount    DOUBLE PRECISION,
    total_fnb_spend DOUBLE PRECISION,
    ts              TIMESTAMPTZ
) WITH (
    connector = 'kafka',
    topic = 'fnb_events',
    properties.bootstrap.server = 'kafka:9092',
    scan.startup.mode = 'latest'
) FORMAT PLAIN ENCODE JSON;

CREATE SOURCE hotel_events_src (
    event_id        VARCHAR,
    event_type      VARCHAR,
    player_id       VARCHAR,
    tier            VARCHAR,
    action          VARCHAR,
    charge_amount   DOUBLE PRECISION,
    ts              TIMESTAMPTZ
) WITH (
    connector = 'kafka',
    topic = 'hotel_events',
    properties.bootstrap.server = 'kafka:9092',
    scan.startup.mode = 'latest'
) FORMAT PLAIN ENCODE JSON;

-- ML predictions table is created in 04_recommendation_mvs.sql
-- (inference service writes directly via SQL INSERT)
