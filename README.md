# Smart Casino Floor 2.0

**A live floor operations workspace: observe demand, compare options, approve an action, and follow its outcome.**

Smart Casino Floor is a local reference demo built with **Kafka, RisingWave, Python, and Streamlit**. A durable simulator runs a 36-position casino floor. Managers can set an operating goal, preview the next 15 or 30 minutes, and dispatch a reviewed table action. Execution confirmations and subsequent observations return through the streaming pipeline.

Version 2.0 adds an operations action center, constrained scenario planning, an optional natural-language goal parser, event-triggered replanning, and an English guided tour. Player analytics, synthetic ML recommendations, and read-only SQL chat remain available on a separate page.

[Architecture](#architecture) · [Quick start](#quick-start) · [Try the workflow](#try-the-workflow) · [AI capabilities](#ai-capabilities) · [Validation and limits](#validation-and-limits)

## What you can do

| Capability | Behavior in 2.0 |
| --- | --- |
| **Live floor** | Inspect actual simulated seats, queues, table minimums, and dealer assignments across main baccarat, entry baccarat, VIP, blackjack, and slots. Switch occupancy, queue-pressure, and modeled-Theo layers. |
| **Future floor** | Compare keeping the current setup, opening a reserve table, or changing a table minimum. Preview the selected scenario at +15 / +30 minutes. |
| **Goal and constraints** | Set the area, horizon, wait target, additional staff allowance, VIP protection, excluded tables, and priority. Review interpreted fields before evaluating. |
| **Action center** | Create a task, assign an owner, approve it, and explicitly dispatch it. The simulator must confirm execution before the task enters observation. |
| **Replanning** | Resource loss, significant queue changes, and expiry invalidate affected plans. Replacement plans retain constraints and require fresh approval. |
| **Evidence and learning** | Inspect physical events, decision history, +5 / +15 minute observations, and saved forecast errors. Data gaps and intervening actions limit comparisons. |
| **Guided tour** | A first-visit English walkthrough highlights the real controls. Replay it with **Start guided tour**; Back, Next, Skip, and Escape are supported. |
| **Player analytics** | Review high-roller candidates, Theo by tier, stored recommendations, and natural-language SQL questions using retained player history. |

The starting floor has **24 baccarat tables, 4 blackjack tables, and 8 slot machines**. Six baccarat tables are reserves, leaving 30 open positions and 162 available seats. One player can occupy at most one seat; closed tables contribute no open capacity.

## Architecture

![Smart Casino Floor 2.0 architecture: streamed observations, manager-reviewed commands, optional AI, and player analytics](architecture.png)

[Open scalable SVG](architecture.svg) · [Diagram source](architecture.py)

### Operational observations and commands

1. **The floor simulator** owns physical state: players, seating, queues, tables, staff, and command receipts. It saves `/state/floor.json` and emits floor snapshots, per-table state, and gaming / F&B / hotel events.
2. **Kafka** carries four active input streams: `operational_events`, `gaming_events`, `fnb_events`, and `hotel_events`. Compose also creates two reserved recommendation topics; the current ML writeback uses SQL, not those topics.
3. **RisingWave** continuously maintains the latest floor snapshot and current table / pit state. The operations service polls `mv_ops_latest_snapshot` every second. It does not bypass RisingWave by reading the simulator's state file.
4. **The operations service** evaluates candidates, checks resource and policy constraints, tracks approvals, and records plans, tasks, commands, and observations in a **SQLite WAL ledger**. It serves the Streamlit operations workspace through HTTP.
5. **The manager** approves and then dispatches a specific action. The simulator polls `/commands`, validates the request, prepares the action, and applies it. Receipts return inside streamed floor snapshots through **Kafka → RisingWave → operations service**. Dispatch alone is not proof of execution.

The task lifecycle is:

```text
PENDING → ACCEPTED → EXECUTING → OBSERVING → CLOSED
```

Rejection, cancellation, expiry, and execution failure are recorded separately. The backend rechecks freshness, table version, staff availability, conflicts, limits, and cooldowns. Command IDs and persisted receipts prevent duplicate application on retries. The dashboard refreshes the operations workspace every three seconds; background processing continues independently of page navigation and chat.

### Forecasting and AI boundaries

The scenario engine uses observable arrivals, current seats and queues, budget constraints, and a known dining-group signal. Each candidate starts from the same saved snapshot and uses the same random seed and external-demand assumptions. It compares **one table action at a time**, with low / base / high demand scenarios.

An optional LLM translates a goal into typed constraint changes. **Python business logic performs forecasting, ranking, validation, replanning, and command execution.** Replanning explanations are generated from the checked conditions and scenario results; this release does not depend on an LLM to make numerical forecasts or autonomously run the floor.

### Retained player analytics

Gaming, F&B, and hotel sources also feed five-minute player feature views, cumulative Theo, and high-roller similarity views. The ML service reads `mv_player_latest_features` every ten seconds and writes predictions into `recommendations_tbl` through SQL. RisingWave joins those predictions with business rules for analytics and suggested offers.

The **Player analytics & chat** page queries RisingWave directly. Its separate LLM agent generates read-only SQL and summarizes results; chat memory is persisted in `chat_messages`. Historical player records survive scenario resets and must not be interpreted as the number of people currently seated.

## Quick start

Requirements: Docker with the Compose plugin, available local ports listed below, and network access to pull base images and install build dependencies. No AI key is required for the displayed goal template, manual planning, or action workflow.

```bash
git clone https://github.com/xinghua0123/smart-casino.git
cd smart-casino

# Build images and start the full demo.
docker compose up --build -d

# Inspect initialization and service status.
docker compose ps --all
docker compose logs --tail=60 rw-setup operations-service
```

Wait for schema setup to complete and the operations snapshot to become fresh. Open **[localhost:8501](http://localhost:8501)**. Initial startup time depends on image builds and source backfill; a running HTTP service does not by itself mean telemetry is ready.

```bash
curl http://localhost:8090/health
# Ready for planning: {"status": "ok", "stream_fresh": true}
```

| Endpoint | Purpose |
| --- | --- |
| `http://localhost:8501` | Streamlit operations workspace and player analytics |
| `http://localhost:8090` | Operations API; published on host loopback only |
| `localhost:4566` | RisingWave PostgreSQL-compatible SQL; user `root`, database `dev` |
| `http://localhost:5691` | RisingWave console / health endpoint |
| `kafka:9092` | Kafka address advertised to containers; a host port is also published |

This Compose setup is for local demos. Dashboard and infrastructure ports use Docker's default host bindings; the application has demo identities rather than enterprise authentication.

### Development with existing local images

The development overlay mounts the current source into previously built dashboard and producer images. It is useful on a workstation that already has `smart-casino-floor-dashboard:latest` and `smart-casino-floor-data-producer:latest` with the required dependencies:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
```

This overlay is not a substitute for building dependencies on a clean machine. The recorded 2.0 validation used these cached runtimes; a clean online build was not completed in that environment because the external base-image metadata request stalled.

### Stop and resume without deleting data

```bash
# Stop this project's containers, preserving state.
docker compose stop

# Resume existing containers with their current image / mount configuration.
docker compose start
```

For a code or configuration update, use the appropriate `up` command above. `docker compose down` removes containers and the network while retaining named volumes. Adding `-v` deletes the simulator state, operations ledger, and RisingWave data; it is not needed for routine shutdown.

## Try the workflow

1. Open **Start guided tour**, or follow the steps below. The tour only navigates and explains; it never approves or dispatches an action for you.
2. Click **Reset floor scenario**, wait for **LIVE**, then click **Dining group arrives** to introduce a main-pit demand signal.
3. Under **Give the floor a goal**, use the English example and click **Interpret goal**:

   > For the next 30 minutes, keep the main floor wait within 5 minutes, with no additional staff. Keep VIP minimums unchanged.

4. Check **Confirm constraints**. For a follow-up, enter **Exclude B08** and interpret again; existing constraints remain in place.
5. Click **Evaluate feasible scenarios**, compare the results, and select **Scenario to preview**. Use **Map time** to inspect +15 / +30 minutes. An unmet wait target is shown explicitly and does not cause constraints to be relaxed automatically.
6. Select a feasible action and click **Create action for review**. In **Action center**, choose an owner, click **Approve & assign**, and then **Dispatch to floor**. An existing automatic task can also be reviewed there.
7. Watch execution confirmation and +5 / +15 minute observations. Use **Evidence & learning** to inspect events and forecast error reviews.

The simulation runs at **10× speed**: one demo minute takes about six real seconds. Opening preparation takes about 12 seconds; the two observation intervals take about 30 and 90 seconds after application. Plans have a ten-demo-minute approval window, so generate a fresh plan if a walkthrough takes longer.

Other scenarios exercise resource loss and telemetry interruption. **Reassign relief dealers** can invalidate an opening plan. **Interrupt telemetry** stops telemetry; after 15 seconds the last known values remain visible while new planning, approval, and dispatch are paused. **Resume telemetry** restores the stream.

[Detailed acceptance scenarios and implementation limits](docs/V2_ACCEPTANCE.md)

## AI capabilities

| Component | What it does | Without a provider key |
| --- | --- | --- |
| Operations goal parser | Optional OpenAI, Claude, or OpenRouter call; returns schema-checked constraint changes. Supports follow-up exclusions and protects prior hard constraints. | Explicitly labeled limited templates and manual constraints remain available. Unsupported wording produces an English hint. |
| Scenario engine and replanner | Deterministic candidate simulation, resource checks, ranking, invalidation, and replacement plans. | Fully available; no LLM required. |
| Player analytics SQL chat | Separate agent with Claude, OpenAI, OpenRouter, and Azure OpenAI options; queries RisingWave and summarizes results. | Analytics charts remain available; model-backed answers require configuration. |
| Player ML inference | Four scikit-learn models trained on synthetic data: next game, churn, offer sensitivity, and high-roller trajectory. | Fully available; models are trained when the ML image is built. |

For goal parsing, use **AI connection** in the operations sidebar, or supply the following variables through the shell or a local, git-ignored `.env` file:

| Variable | Meaning |
| --- | --- |
| `OPS_LLM_API_KEY` | Operations parser provider key |
| `OPS_LLM_PROVIDER` | `OpenAI`, `Claude`, or `OpenRouter` |
| `OPS_LLM_MODEL` | A model identifier available to your account |
| `OPS_LLM_BASE_URL` | Optional provider-compatible gateway URL; set explicitly for environment-based OpenRouter configuration |

Choose the correct model for the selected provider; the default model field is OpenAI-specific. A configured external provider receives the entered goal and current constraints. Keys are not written to the operations ledger. The analytics chat has its own provider settings; `LLM_API_KEY` is passed into the dashboard container by Compose.

The SQL chat store defaults to RisingWave. `dashboard/chat_store.py` also supports `CHAT_STORE_*` connection settings for another PostgreSQL-compatible backend when explicitly wired into the dashboard environment.

## State ownership and persistence

| Owner | Persistent store | Contents |
| --- | --- | --- |
| Simulator | `simulator-state` → `/state/floor.json` | Physical floor state, player positions, resource assignments, and applied-command receipts |
| Operations service | `operations-state` → `/state/operations.sqlite` | Plans, tasks, timelines, commands / controls, snapshot history, and observations |
| RisingWave | `risingwave-state` → `/root/.risingwave` | Streaming catalog and materialized data, ML predictions, and SQL chat memory |
| Browser / Streamlit session | Session state; local browser preference for the tour | Current UI selections and transient configuration; tour dismissal is remembered in the browser |

**Reset floor scenario** starts a new simulation run while preserving historical tasks, plans, and player events. Setup records schema migrations so normal restarts do not rebuild existing player history. Kafka has no named persistent volume in this demo: stopping existing containers retains their filesystem, but replacing the Kafka container is not a durable broker recovery strategy.

## Data and metric reference

| Object | Role |
| --- | --- |
| `mv_ops_latest_snapshot` | Latest ordered full-floor snapshot consumed by the operations service |
| `mv_ops_table_state` | Latest status, actual occupied seats, capacity, and effective minimum per table |
| `mv_ops_pit_state` | Current seats, open capacity, and open-table count per pit |
| `mv_player_session_features` | Five-minute player activity and window Theo |
| `mv_player_theo_cumulative` | Cumulative wagered amount, Theo, and effective house edge |
| `mv_player_latest_features` | One latest feature row per player, used by ML inference |
| `recommendations_tbl` | Predictions written back by the ML service |
| `mv_actionable_recommendations` | Predictions joined with player features and demo offer rules |
| `mv_high_roller_radar`, `mv_theo_by_tier`, `mv_dashboard_stats` | Retained player analytics |
| `mv_table_live_load`, `mv_table_recommendations` | Retained SQL floor signals joined with streamed seat / minimum state; operations tasks are managed separately |

The operational **wait** metric is the average elapsed queue age of guests currently waiting, rather than an end-to-end service-level guarantee. Open capacity excludes closed and preparing tables. Scenario "served" and queue-departure counts are floor-wide; the wait, queue, and Theo comparison metrics refer to the selected pit.

**Theoretical Win (Theo)** is modeled expected gaming win, not realized profit:

```text
Theo from observed wagers = sum(bet amount × configured house edge)
Modeled hourly Theo = sum(occupied seats × minimum × 1.4 × house edge × 1.5 × 60)
```

The demo uses house edges of **1.15% for baccarat, 0.75% for blackjack, and 7.50% for slots**. These are model assumptions, not a representation of every game's rules or a specific casino's measured edge. Additional-dealer cost is modeled separately at HK$120 per hour. Demand ranges use 0.8× / 1.2× assumptions and are not calibrated confidence intervals. Changes before and after an action do not establish causal revenue lift.

Example read-only SQL, after startup:

```sql
-- Actual current table state, not a visitor-count estimate.
SELECT table_id, pit, status, occupied, capacity, minimum, seq
FROM mv_ops_table_state
ORDER BY pit, table_id;

-- Inspect the snapshot timestamp as well as its sequence.
SELECT run_id, seq, sim_minute, wall_time
FROM mv_ops_latest_snapshot;

-- Retained player value, separate from current floor occupancy.
SELECT player_id, tier, ROUND(cumulative_theo_win::numeric, 2) AS theo
FROM mv_player_latest_features
ORDER BY cumulative_theo_win DESC NULLS LAST
LIMIT 10;
```

## Repository map

| Path | Responsibility |
| --- | --- |
| `casino/` | Shared floor layout, metrics, physical simulation, and action validation |
| `data_producer/floor_simulator.py` | Active 2.0 producer: durable state, command polling, and Kafka events |
| `operations_service/` | HTTP API, SQLite ledger, task lifecycle, forecasting, and goal parser |
| `risingwave_sql/` | Six SQL modules and migration-aware setup; `06_operations.sql` adds operational state |
| `ml_service/` | Synthetic training and periodic player prediction writeback |
| `dashboard/app.py`, `dashboard/tour.html` | Operations workspace and guided onboarding |
| `dashboard/analytics.py`, `dashboard/agent.py`, `dashboard/chat_store.py` | Player analytics, SQL chat, and chat persistence |
| `tests/` | Domain / ledger tests, page smoke checks, and live integration scenarios |
| `docs/` | Implementation plan, acceptance walkthrough, and recorded live checks |
| `architecture.py` | Reproducible source for `architecture.png` and `architecture.svg` |

The active Compose entrypoint is `floor_simulator.py`; the earlier producer script remains in the repository as a reference.

## Validation and limits

The 2.0 validation record includes **25 unit tests**, **15 live integration checks**, operations / analytics page smoke checks, and browser checks for the English goal flow and responsive tour. Live checks cover execution receipts, duplicate dispatch, restart recovery, staff loss, stale telemetry, automatic opportunities, and forecast review.

```bash
# Offline domain and ledger tests; no services required.
python3 -B -m unittest discover -s tests -v

# The following require the local demo to be running.
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T dashboard python < tests/dashboard_smoke.py
python3 -B tests/live_acceptance.py
```

The live acceptance script changes local demo scenarios and restarts application services. Run it when those effects are intended. [Recorded results](docs/LIVE_TEST_RESULTS.json) include the run timestamp; they are not a claim that every external integration has been tested.

Remaining boundaries:

- This is a simulated floor, with no live casino-management or WDTS integration.
- Forecasts and player ML use demo assumptions / synthetic data, without venue-specific calibration.
- One-action candidate search is not a global multi-table optimizer; forecast review does not automatically retrain or deploy policies.
- Real provider calls were not exercised in the recorded release validation. Tests cover mocked LLM output, invalid constraints, and failure fallback.
- Demo operator names are not enterprise identities. Production authentication, authorization, audit hardening, availability, and broker durability are future work.

### Regenerate the architecture diagram

```bash
# Use a Python environment with Matplotlib installed.
python3 -m pip install matplotlib
python3 architecture.py
```

The generator writes both outputs next to its source and does not start any demo services.

## Versions

- [`1.0`](https://github.com/xinghua0123/smart-casino/tree/1.0): the prior player-analytics and floor-recommendation baseline.
- [`2.0`](https://github.com/xinghua0123/smart-casino/tree/2.0): operations workflow, scenario planning, goal parsing, replanning, persistent execution observations, and English onboarding.

[Implementation scope](docs/V2_PLAN.md) · [Acceptance and runbook](docs/V2_ACCEPTANCE.md)
