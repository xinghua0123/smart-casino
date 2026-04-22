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

# Game-mix priors per archetype. Macau-style 3-game floor: baccarat is the
# hero (~88% of Macau GGR), slots carry casual volume, blackjack is the
# secondary pit. Roulette and poker have been removed — they do not move
# the needle on theo win in the Asian market.
ARCHETYPES = {
    "casual": {
        "weight": 0.50,
        # Casual: mostly slots, some entry-level baccarat / blackjack.
        "games": {"slots": 0.68, "baccarat": 0.20, "blackjack": 0.12},
        "avg_bet": (5, 25),
        "session_gap_s": (60, 300),
        "fnb_prob": 0.15,
        "hotel_prob": 0.05,
        "avg_fnb_spend": (10, 40),
    },
    "regular": {
        "weight": 0.30,
        # Regulars lean into baccarat (the Asian staple) + blackjack; slots are a filler.
        "games": {"baccarat": 0.52, "blackjack": 0.28, "slots": 0.20},
        "avg_bet": (25, 100),
        "session_gap_s": (30, 180),
        "fnb_prob": 0.25,
        "hotel_prob": 0.15,
        "avg_fnb_spend": (30, 80),
    },
    "high_roller": {
        "weight": 0.12,
        # High rollers are a baccarat VIP segment first, blackjack second.
        "games": {"baccarat": 0.65, "blackjack": 0.30, "slots": 0.05},
        "avg_bet": (200, 2000),
        "session_gap_s": (15, 90),
        "fnb_prob": 0.40,
        "hotel_prob": 0.50,
        "avg_fnb_spend": (80, 300),
    },
    "emerging": {
        "weight": 0.08,
        # Upwardly mobile — testing baccarat VIP while still playing the pit.
        "games": {"baccarat": 0.48, "blackjack": 0.32, "slots": 0.20},
        "avg_bet": (50, 300),
        "session_gap_s": (20, 120),
        "fnb_prob": 0.30,
        "hotel_prob": 0.25,
        "avg_fnb_spend": (40, 120),
    },
}

GAME_TYPES = ["slots", "baccarat", "blackjack"]

# --- Physical casino floor layout ------------------------------------------------
# Each table has a fixed (x, y) position on the floor, a game_type, and a live
# betting-limit range. The floor-plan view on the dashboard uses (x, y) verbatim
# for rendering, and the per-table MVs use limit_min/limit_max to decide whether
# the limit should be raised (packed + betting near ceiling) or lowered
# (cold + few players). Keep this in sync with `tables_dim` seed in 05_floor_plan_mvs.sql.
#
#   Macau-style 3-game layout.
#     Slots (left):    8 penny + 8 standard  = 16 tables — casual volume
#     Baccarat (mid):  8 standard + 8 VIP    = 16 tables — the hero
#     Blackjack (rgt): 4 tables              =  4 tables — secondary pit
#   Total 36 tables.
TABLE_LAYOUT: list[tuple[str, str, float, float, float, float]] = (
    # Penny slots — 4x2 grid (low limits, near entrance, top-left)
    [(f"slots_{i:02d}", "slots",
      0.6 + ((i - 1) % 4) * 1.2,
      7.8 + ((i - 1) // 4) * 1.0,
      1.0, 10.0) for i in range(1, 9)]
    # Standard slots — 4x2 grid (mid-left)
    + [(f"slots_{i:02d}", "slots",
        0.6 + ((i - 9) % 4) * 1.2,
        3.5 + ((i - 9) // 4) * 1.0,
        5.0, 25.0) for i in range(9, 17)]
    # ——— BACCARAT standard pit — 4x2 grid, center of floor ———
    + [(f"bac_{i:02d}", "baccarat",
        5.8 + ((i - 1) % 4) * 1.2,
        3.5 + ((i - 1) // 4) * 1.0,
        100.0, 500.0) for i in range(1, 9)]
    # ——— BACCARAT VIP room — 4x2 grid (top-center/right), behind velvet rope ———
    + [(f"bac_vip_{i:02d}", "baccarat",
        5.8 + ((i - 1) % 4) * 1.2,
        7.5 + ((i - 1) // 4) * 1.0,
        500.0, 10000.0) for i in range(1, 9)]
    # Blackjack — single 2x2 pit on the right
    + [(f"bj_{i:02d}", "blackjack",
        11.2 + ((i - 1) % 2) * 1.2,
        3.5 + ((i - 1) // 2) * 1.0,
        25.0, 500.0) for i in range(1, 5)]
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
    # Win probability varies by game — baccarat is near 50/50 (tiny house edge).
    win_probs = {"slots": 0.35, "blackjack": 0.48, "baccarat": 0.49}
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
