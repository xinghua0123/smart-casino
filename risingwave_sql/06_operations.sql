-- Additive streaming state: one ordered snapshot and one current row per table.
CREATE SOURCE IF NOT EXISTS operational_events_src (
    event_id VARCHAR, kind VARCHAR, entity_id VARCHAR, run_id VARCHAR,
    seq BIGINT, wall_time DOUBLE PRECISION, sim_minute DOUBLE PRECISION,
    state_json VARCHAR, table_id VARCHAR, pit VARCHAR, status VARCHAR,
    occupied INT, capacity INT, minimum DOUBLE PRECISION
) WITH (
    connector = 'kafka', topic = 'operational_events',
    properties.bootstrap.server = 'kafka:9092', scan.startup.mode = 'earliest'
) FORMAT PLAIN ENCODE JSON;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ops_latest_snapshot AS
SELECT event_id, run_id, seq, wall_time, sim_minute, state_json FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY wall_time DESC, seq DESC) AS rn
    FROM operational_events_src WHERE kind = 'snapshot'
) WHERE rn = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ops_table_state AS
SELECT table_id, pit, status, occupied, capacity, minimum, run_id, seq, wall_time, sim_minute FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY table_id ORDER BY wall_time DESC, seq DESC) AS rn
    FROM operational_events_src WHERE kind = 'table'
) WHERE rn = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ops_pit_state AS
SELECT pit, SUM(occupied) AS seated,
       SUM(CASE WHEN status = 'open' THEN capacity ELSE 0 END) AS open_capacity,
       SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_tables,
       MIN(wall_time) AS oldest_table_event
FROM mv_ops_table_state GROUP BY pit;
