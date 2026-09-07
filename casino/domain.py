"""Physical floor and observable state. No future scenario scripts live here."""
import copy
import random
import uuid

PITS = {"main": "Main baccarat", "entry": "Entry baccarat", "vip": "VIP baccarat", "blackjack": "Blackjack", "slots": "Slots"}
EDGES = {"baccarat": .0115, "blackjack": .0075, "slots": .075}
BASE_RATES = {"main": 2.0, "entry": 1.2, "vip": .7, "blackjack": .6, "slots": .35}
DEFAULT_CONSTRAINTS = {"pit": "main", "horizon": 30, "max_wait": 5.0, "extra_staff": 0, "protect_vip": True, "excluded": [], "priority": "wait"}


def layout():
    rows = []
    groups = [("entry", "bac_left_", 8, .6, 7.5, 50, 500, 25),
              ("main", "bac_", 8, 5.8, 3.5, 100, 500, 50),
              ("vip", "bac_vip_", 8, 5.8, 7.5, 500, 10000, 200),
              ("blackjack", "bj_", 4, 7.0, .6, 25, 500, 10),
              ("slots", "slots_", 8, .6, 3.5, 5, 25, 1)]
    for pit, prefix, count, x, y, low, high, floor in groups:
        for i in range(count):
            tid = f"{prefix}{i + (9 if pit == 'slots' else 1):02d}"
            game = pit if pit in ("slots", "blackjack") else "baccarat"
            reserve = pit in ("main", "entry", "vip") and i >= 6
            rows.append(dict(id=tid, pit=pit, game=game, x=x+(i % (2 if pit == 'blackjack' else 4))*1.2,
                             y=y+(i // (2 if pit == 'blackjack' else 4)), capacity=1 if game == "slots" else 7,
                             minimum=low, maximum=high, minimum_floor=floor, status="closed" if reserve else "open",
                             version=1, occupied=0, dealer=None, theo=0.0, cooldown_until=0.0))
    return rows


def initial_state(seed=42):
    state = dict(run_id=str(uuid.uuid4()), seq=0, minute=0.0, speed=10, seed=seed, wall_time=0.0,
                 tables=layout(), players=[], dealers=[], receipts={}, control_receipts=[], events=[], signal=None,
                 arrival_rates=dict(BASE_RATES), observed_arrivals=[], observed_departures=[], session_minutes=18.0,
                 total_arrivals=0, served=0, abandoned=0, total_theo=0.0, cost_totals={p:0.0 for p in PITS}, outage=False, scenario="Normal operations")
    for table in state["tables"]:
        if table["game"] != "slots" and table["status"] == "open":
            did = f"D{len(state['dealers'])+1:02d}"
            table["dealer"] = did
            state["dealers"].append(dict(id=did, name=f"Dealer {did}", skills=[table["game"]], status="assigned", table=table["id"], version=1, extra=False))
    for did, skills, extra in [("F01", ["baccarat", "blackjack"], False), ("F02", ["baccarat"], False), ("X01", ["baccarat"], True)]:
        state["dealers"].append(dict(id=did, name={"F01":"Alex Chan", "F02":"Mei Wong", "X01":"Reserve shift"}[did], skills=skills, status="available", table=None, version=1, extra=extra))
    rng = random.Random(seed)
    for t in state["tables"]:
        if t["status"] == "open":
            for _ in range(5 if t["capacity"] > 1 else 1):
                add_player(state, t["pit"], rng, seated_at=t)
    return state


def record(state, kind, entity, detail):
    state["events"].append(dict(id=str(uuid.uuid4()), minute=round(state["minute"], 2), kind=kind, entity=entity, detail=detail))
    state["events"] = state["events"][-80:]


def add_player(state, pit, rng, seated_at=None):
    state["total_arrivals"] += 1
    base = {"entry": 50, "main":100, "vip":500, "blackjack":25, "slots":5}[pit]
    player = dict(id=f"P{state['total_arrivals']:06d}", pit=pit, budget=base*rng.choice([1,1,2,3,5]),
                  table=seated_at["id"] if seated_at else None, arrived=state["minute"],
                  seated=state["minute"] if seated_at else None, dwell=rng.uniform(12,24),
                  gaming_spend=0.0, fnb_spend=0.0, bets=0)
    state["players"].append(player)
    if seated_at:
        seated_at["occupied"] += 1
        state["served"] += 1
    record(state, "SEATED" if seated_at else "QUEUE_JOINED", player["id"], pit)
    return player


def metrics(state, pit=None):
    tables = [t for t in state["tables"] if pit is None or t["pit"] == pit]
    players = [p for p in state["players"] if pit is None or p["pit"] == pit]
    queue = [p for p in players if p["table"] is None]
    capacity = sum(t["capacity"] for t in tables if t["status"] == "open")
    seated = sum(t["occupied"] for t in tables)
    wait = sum(max(0,state["minute"]-p["arrived"]) for p in queue)/max(1,len(queue))
    theo = sum(t["occupied"]*t["minimum"]*1.4*EDGES[t["game"]]*1.5*60 for t in tables)
    # Queue-size / seat-turnover estimate, separate from elapsed queue age.
    estimate=0.0
    for group in {p["pit"] for p in queue}:
        n=sum(p["pit"]==group for p in queue)
        seats=sum(t["capacity"] for t in tables if t["pit"]==group and t["status"]=="open")
        if not seats:
            estimate=None
            break
        estimate+=n*n*state.get("session_minutes",18)/seats
    estimate=round(estimate/max(1,len(queue)),2) if estimate is not None else None
    return dict(queue=len(queue), wait=round(wait,2), estimated_wait=estimate, seated=seated, capacity=capacity,
                theo_total=round(sum(t["theo"] for t in tables),2),
                labor_cost_total=round(sum(v for p,v in state.get("cost_totals",{}).items() if pit is None or p==pit),2),
                occupancy=round(seated/max(1,capacity),3), theo_hour=round(theo,2), open_tables=sum(t["status"]=="open" for t in tables))


def public_state(state):
    result = copy.deepcopy(state)
    result["metrics"] = metrics(state)
    result["pits"] = {pit:metrics(state,pit) for pit in PITS}
    return result
