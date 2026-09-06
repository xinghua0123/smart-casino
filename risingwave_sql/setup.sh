#!/bin/sh
set -eu
RW_HOST="${RISINGWAVE_HOST:-localhost}"
RW_PORT="${RISINGWAVE_PORT:-4566}"
sql() { psql -v ON_ERROR_STOP=1 -h "$RW_HOST" -p "$RW_PORT" -U root -d dev "$@"; }
ready=0
for i in $(seq 1 60); do
    if sql -c 'SELECT 1' >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
done
[ "$ready" = 1 ] || exit 1
sql -c 'CREATE TABLE IF NOT EXISTS demo_schema_migrations (version VARCHAR PRIMARY KEY)'
# Preserve existing player features, predictions and chat history on restart.
if ! sql -c 'SELECT 1 FROM mv_player_latest_features LIMIT 1' >/dev/null 2>&1; then
    for file in 01_sources 02_feature_mvs 03_high_roller_mvs 04_recommendation_mvs; do
        sql -f "/sql/$file.sql"
    done
fi
sql -f /sql/06_operations.sql
installed=$(sql -Atc "SELECT COUNT(*) FROM demo_schema_migrations WHERE version='ops-preview-1'")
if [ "$installed" = 0 ]; then
    # One-time rebuild of floor-only derived views; player and action history are retained.
    sql -f /sql/05_floor_plan_mvs.sql
    sql -c "INSERT INTO demo_schema_migrations VALUES ('ops-preview-1')"
fi
echo 'Schema ready; existing data retained.'
