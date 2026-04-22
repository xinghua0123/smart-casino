#!/bin/bash
# Run all SQL files against RisingWave in order

RW_HOST="${RISINGWAVE_HOST:-localhost}"
RW_PORT="${RISINGWAVE_PORT:-4566}"

echo "Waiting for RisingWave to be ready..."
for i in $(seq 1 60); do
    if psql -h "$RW_HOST" -p "$RW_PORT" -U root -d dev -c "SELECT 1" >/dev/null 2>&1; then
        echo "RisingWave is ready."
        break
    fi
    echo "  Attempt $i/60..."
    sleep 3
done

FAILED=0
for f in /sql/01_sources.sql /sql/02_feature_mvs.sql /sql/03_high_roller_mvs.sql /sql/04_recommendation_mvs.sql /sql/05_floor_plan_mvs.sql; do
    echo "Running $f ..."
    if ! psql -h "$RW_HOST" -p "$RW_PORT" -U root -d dev -f "$f"; then
        echo "WARNING: $f had errors (continuing)"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "Some SQL statements failed. Check logs above."
    exit 1
fi
echo "All SQL applied successfully."
