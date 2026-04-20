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
