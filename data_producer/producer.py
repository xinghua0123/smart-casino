"""
Casino event producer — generates realistic streams of gaming, F&B, and hotel events
for ~200 simulated players with varying behavioral profiles.
"""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

KAFKA_BOOTSTRAP = "kafka:9092"
TOPICS = {
    "gaming": "gaming_events",
    "fnb": "fnb_events",
    "hotel": "hotel_events",
}

# --- Player archetypes -----------------------------------------------------------

ARCHETYPES = {
    "casual": {
        "weight": 0.50,
        "games": {"slots": 0.70, "roulette": 0.20, "blackjack": 0.08, "poker": 0.02},
        "avg_bet": (5, 25),
        "session_gap_s": (60, 300),
        "fnb_prob": 0.15,
        "hotel_prob": 0.05,
        "avg_fnb_spend": (10, 40),
    },
    "regular": {
        "weight": 0.30,
        "games": {"slots": 0.30, "roulette": 0.25, "blackjack": 0.30, "poker": 0.15},
        "avg_bet": (25, 100),
        "session_gap_s": (30, 180),
        "fnb_prob": 0.25,
        "hotel_prob": 0.15,
        "avg_fnb_spend": (30, 80),
    },
    "high_roller": {
        "weight": 0.12,
        "games": {"slots": 0.05, "roulette": 0.20, "blackjack": 0.45, "poker": 0.30},
        "avg_bet": (200, 2000),
        "session_gap_s": (15, 90),
        "fnb_prob": 0.40,
        "hotel_prob": 0.50,
        "avg_fnb_spend": (80, 300),
    },
    "emerging": {
        "weight": 0.08,
        "games": {"slots": 0.25, "roulette": 0.25, "blackjack": 0.30, "poker": 0.20},
        "avg_bet": (50, 300),
        "session_gap_s": (20, 120),
        "fnb_prob": 0.30,
        "hotel_prob": 0.25,
        "avg_fnb_spend": (40, 120),
    },
}

GAME_TYPES = ["slots", "roulette", "blackjack", "poker"]

# --- Physical casino floor layout ------------------------------------------------
# Each table has a fixed (x, y) position on the floor, a game_type, and a live
# betting-limit range. The floor-plan view on the dashboard uses (x, y) verbatim
# for rendering, and the per-table MVs use limit_min/limit_max to decide whether
# the limit should be raised (packed + betting near ceiling) or lowered
# (cold + few players). Keep this in sync with `tables_dim` seed in 05_floor_plan_mvs.sql.
#
#   Grid is roughly 10 wide x 8 tall. Slots cluster at the left (high foot traffic
#   area near entrance), table games in the middle, and poker lounge on the right.
TABLE_LAYOUT: list[tuple[str, str, float, float, float, float]] = (
    # Penny slots row (low limits, near entrance) — 8 machines
    [(f"slots_{i:02d}", "slots",
      0.4 + ((i - 1) % 4) * 0.55,
      6.3 + ((i - 1) // 4) * 0.7,
      1.0, 10.0) for i in range(1, 9)]
    # Standard slots row (mid limits) — 8 machines
    + [(f"slots_{i:02d}", "slots",
        0.4 + ((i - 9) % 4) * 0.55,
        2.8 + ((i - 9) // 4) * 0.7,
        5.0, 25.0) for i in range(9, 17)]
    # Standard blackjack pit — 6 tables
    + [(f"bj_{i:02d}", "blackjack",
        4.1 + ((i - 1) % 2) * 0.7,
        2.5 + ((i - 1) // 2) * 0.9,
        25.0, 500.0) for i in range(1, 7)]
    # High-limit blackjack tables — 2 tables
    + [("bj_07", "blackjack", 5.7, 6.0, 500.0, 2000.0),
       ("bj_08", "blackjack", 5.7, 6.9, 500.0, 2000.0)]
    # Roulette (entrance-side, mid limits) — 4 wheels
    + [(f"rou_{i:02d}", "roulette",
        4.1 + (i - 1) * 0.55,
        0.8,
        10.0, 200.0) for i in range(1, 5)]
    # Poker lounge (back, high stakes) — 4 tables
    + [(f"pok_{i:02d}", "poker",
        7.3 + ((i - 1) % 2) * 0.9,
        3.3 + ((i - 1) // 2) * 1.0,
        50.0, 1000.0) for i in range(1, 5)]
)

# Group tables by game_type for fast lookup
_TABLES_BY_GAME: dict[str, list[tuple[str, str, float, float, float, float]]] = {}
for _t in TABLE_LAYOUT:
    _TABLES_BY_GAME.setdefault(_t[1], []).append(_t)


def pick_table(game_type: str, bet: float) -> tuple[str, str, float, float, float, float]:
    """Pick a table for this bet. Prefer tables whose [limit_min, limit_max] contains
    the bet; fall back to the nearest tier if none match."""
    options = _TABLES_BY_GAME.get(game_type, [])
    if not options:
        return ("unknown", game_type, 0.0, 0.0, 0.0, 0.0)
    fits = [t for t in options if t[4] <= bet <= t[5]]
    if fits:
        return random.choice(fits)
    # No tier matches — bet is below all mins or above all maxes; pick the closest tier.
    max_allowed = max(t[5] for t in options)
    if bet > max_allowed:
        top_tables = [t for t in options if t[5] == max_allowed]
        return random.choice(top_tables)
    min_allowed = min(t[4] for t in options)
    bottom_tables = [t for t in options if t[4] == min_allowed]
    return random.choice(bottom_tables)


FNB_ITEMS = [
    "cocktail", "beer", "wine", "steak_dinner", "burger",
    "sushi_platter", "dessert", "coffee", "champagne", "lobster",
]
HOTEL_ACTIONS = ["checkin", "room_service", "spa", "minibar", "checkout"]
TIERS = ["bronze", "silver", "gold", "platinum", "diamond"]


def pick_weighted(options: dict) -> str:
    items = list(options.keys())
    weights = list(options.values())
    return random.choices(items, weights=weights, k=1)[0]


def assign_tier(archetype: str) -> str:
    if archetype == "high_roller":
        return random.choice(["platinum", "diamond"])
    if archetype == "emerging":
        return random.choice(["silver", "gold"])
    if archetype == "regular":
        return random.choice(["silver", "gold", "bronze"])
    return random.choice(["bronze", "silver"])


class Player:
    def __init__(self, player_id: int, archetype: str, profile: dict):
        self.player_id = f"P{player_id:04d}"
        self.archetype = archetype
        self.profile = profile
        self.tier = assign_tier(archetype)
        self.total_gaming_spend = 0.0
        self.total_fnb_spend = 0.0
        self.session_count = 0
        # Emerging players gradually increase bets
        self.bet_escalation = 1.0

    def next_event_delay(self) -> float:
        lo, hi = self.profile["session_gap_s"]
        return random.uniform(lo, hi)


def create_players(n: int = 200) -> list[Player]:
    players = []
    archetypes = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[a]["weight"] for a in archetypes]
    for i in range(n):
        arch = random.choices(archetypes, weights=weights, k=1)[0]
        players.append(Player(i, arch, ARCHETYPES[arch]))
    return players


def generate_gaming_event(player: Player) -> dict:
    game = pick_weighted(player.profile["games"])
    lo, hi = player.profile["avg_bet"]
    bet = round(random.uniform(lo, hi) * player.bet_escalation, 2)
    # Pick a physical table on the floor — bet range must match the table limits.
    table_id, _, table_x, table_y, limit_min, limit_max = pick_table(game, bet)
    # Win probability varies by game
    win_probs = {"slots": 0.35, "roulette": 0.45, "blackjack": 0.48, "poker": 0.42}
    won = random.random() < win_probs.get(game, 0.40)
    payout = round(bet * random.uniform(1.5, 5.0), 2) if won else 0.0
    player.total_gaming_spend += bet
    player.session_count += 1
    # Emerging players escalate over time
    if player.archetype == "emerging":
        player.bet_escalation = min(player.bet_escalation + 0.005, 3.0)
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "gaming",
        "player_id": player.player_id,
        "tier": player.tier,
        "archetype": player.archetype,
        "game_type": game,
        "table_id": table_id,
        "table_x": table_x,
        "table_y": table_y,
        "limit_min": limit_min,
        "limit_max": limit_max,
        "bet_amount": bet,
        "payout": payout,
        "won": won,
        "session_count": player.session_count,
        "total_gaming_spend": round(player.total_gaming_spend, 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def generate_fnb_event(player: Player) -> dict | None:
    if random.random() > player.profile["fnb_prob"]:
        return None
    lo, hi = player.profile["avg_fnb_spend"]
    spend = round(random.uniform(lo, hi), 2)
    player.total_fnb_spend += spend
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "fnb",
        "player_id": player.player_id,
        "tier": player.tier,
        "item": random.choice(FNB_ITEMS),
        "spend_amount": spend,
        "total_fnb_spend": round(player.total_fnb_spend, 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def generate_hotel_event(player: Player) -> dict | None:
    if random.random() > player.profile["hotel_prob"]:
        return None
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "hotel",
        "player_id": player.player_id,
        "tier": player.tier,
        "action": random.choice(HOTEL_ACTIONS),
        "charge_amount": round(random.uniform(20, 500), 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main():
    print("Waiting for Kafka to be ready...")
    producer = None
    for attempt in range(30):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            break
        except Exception:
            time.sleep(2)
    if producer is None:
        raise RuntimeError("Could not connect to Kafka")

    players = create_players(200)
    print(f"Created {len(players)} players. Starting event generation...")

    # Track next-event time per player
    next_event = {p.player_id: time.time() + random.uniform(0, 10) for p in players}

    event_count = 0
    while True:
        now = time.time()
        for player in players:
            if now < next_event[player.player_id]:
                continue

            # Gaming event (always)
            gaming_evt = generate_gaming_event(player)
            producer.send(TOPICS["gaming"], value=gaming_evt)
            event_count += 1

            # F&B event (probabilistic)
            fnb_evt = generate_fnb_event(player)
            if fnb_evt:
                producer.send(TOPICS["fnb"], value=fnb_evt)
                event_count += 1

            # Hotel event (probabilistic)
            hotel_evt = generate_hotel_event(player)
            if hotel_evt:
                producer.send(TOPICS["hotel"], value=hotel_evt)
                event_count += 1

            next_event[player.player_id] = now + player.next_event_delay()

            if event_count % 500 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent {event_count} events")

        time.sleep(0.1)


if __name__ == "__main__":
    main()
