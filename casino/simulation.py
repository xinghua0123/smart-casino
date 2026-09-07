"""Stateful physical simulation. Commands are acknowledged only at their effective time."""
import random
from casino.domain import add_player, record, BASE_RATES, initial_state, metrics


def control(state, command):
    cid = command["id"]
    if cid in state["control_receipts"]:
        return
    kind = command["kind"]
    if kind == "reset":
        done = state["control_receipts"][-100:]
        state.clear()
        state.update(initial_state())
        state["control_receipts"] = done
    elif kind == "surge":
        rng = random.Random(11)
        for _ in range(28):
            add_player(state,"main",rng)
        state["signal"] = dict(kind="GROUP_DINING_ENDED", pit="main", minute=state["minute"],
                               expires=state["minute"]+30, extra_arrivals_per_minute=1.0,
                               explanation="Group dining ended · assumed +1 arrival/min for 30 demo minutes")
        state["scenario"] = "Dining group / main-pit surge"
        record(state,"BUSINESS_SIGNAL","restaurant",state["signal"]["explanation"])
    elif kind == "staff_shortage":
        for d in state["dealers"]:
            if d["status"] in ("available", "reserved"):
                d.update(status="unavailable",version=d["version"]+1)
        record(state,"STAFF_UNAVAILABLE","floor","Available and reserved relief staff reassigned")
    elif kind == "restore_staff":
        for d in state["dealers"]:
            if d["status"] == "unavailable":
                d.update(status="available",version=d["version"]+1)
        record(state,"STAFF_AVAILABLE","floor","Relief staff returned")
    elif kind == "outage":
        state["outage"] = True
    elif kind == "resume":
        state["outage"] = False
        record(state,"STREAM_RECOVERED","floor","Telemetry resumed; gap excluded from observation review")
    elif kind == "close_table":
        t = next(t for t in state["tables"] if t["id"] == command["table"])
        t.update(status="paused",version=t["version"]+1)
        for p in state["players"]:
            if p["table"] == t["id"]:
                p.update(table=None,seated=None,arrived=state["minute"])
        t["occupied"] = 0
        record(state,"TABLE_PAUSED",t["id"],"Maintenance")
    else:
        raise ValueError("Unknown scenario control")
    state["control_receipts"].append(cid)
    state["control_receipts"] = state["control_receipts"][-200:]


def validate_action(state, action, constraints, expected_version=None):
    t = next((t for t in state["tables"] if t["id"] == action.get("table")),None)
    if not t:
        return "Unknown table"
    if expected_version is not None and t["version"] != expected_version:
        return "Table configuration changed"
    if t["pit"] != constraints["pit"] or t["id"] in constraints["excluded"]:
        return "Table is outside approved scope or excluded"
    if t["cooldown_until"] > state["minute"]:
        return "Table is in its 10-minute action cooldown"
    if action["kind"] == "OPEN_TABLE":
        if t["status"] != "closed":
            return "Table is no longer closed"
        d = next((d for d in state["dealers"] if d["id"] == action.get("dealer")),None)
        if not d or d["status"] != "available" or t["game"] not in d["skills"]:
            return "Required dealer unavailable or not qualified"
        if d["extra"] and constraints["extra_staff"] < 1:
            return "Additional staff is forbidden"
    elif action["kind"] == "SET_MINIMUM":
        if t["status"] != "open" or t["game"] == "slots":
            return "Only running table games may change minimum"
        if t["pit"] == "vip" and constraints["protect_vip"]:
            return "VIP minimum is protected"
        if not t["minimum_floor"] <= action["minimum"] <= t["maximum"] or action["minimum"] == t["minimum"]:
            return "Minimum is unchanged or outside permitted limits"
    else:
        return "Unsupported action"
    return None


def receive_command(state, command):
    cid = command["id"]
    if cid in state["receipts"]:
        return
    error = None
    if command["run_id"] != state["run_id"]:
        error = "Scenario has changed"
    elif state["minute"] > command["expires"]:
        error = "Command expired"
    else:
        error = validate_action(state,command["action"],command["constraints"],command["table_version"])
    receipt = dict(id=cid,status="FAILED" if error else "PREPARING",reason=error,
                   minute=state["minute"],command=command)
    state["receipts"][cid] = receipt
    if error:
        return
    t = next(t for t in state["tables"] if t["id"] == command["action"]["table"])
    if command["action"]["kind"] == "OPEN_TABLE":
        d = next(d for d in state["dealers"] if d["id"] == command["action"]["dealer"])
        d.update(status="reserved",table=t["id"],version=d["version"]+1)
        t.update(status="opening",version=t["version"]+1)
    receipt["effective_at"] = state["minute"] + (0 if command.get("immediate") else 2 if command["action"]["kind"] == "OPEN_TABLE" else .5)
    record(state,"COMMAND_RECEIVED",t["id"],cid)
    if command.get("immediate"):
        advance(state,0)


def seat_waiting_guests(state):
    for t in state["tables"]:
        t["occupied"] = sum(p["table"] == t["id"] for p in state["players"])
    for p in sorted(state["players"],key=lambda p:(p["arrived"],p["id"])):
        if p["table"]: continue
        matches = [t for t in state["tables"] if t["pit"] == p["pit"] and t["status"] == "open"
                   and t["occupied"] < t["capacity"] and t["minimum"] <= p["budget"]]
        if matches:
            t = min(matches,key=lambda t:(t["occupied"]/t["capacity"],t["id"]))
            p.update(table=t["id"],seated=state["minute"])
            t["occupied"] += 1
            state["served"] += 1
            record(state,"SEATED",p["id"],t["id"])


def advance(state, dt):
    rng = random.Random(state["seed"]+state["seq"])
    state["minute"] += dt
    costs=state.setdefault("cost_totals",{})
    for dealer in state["dealers"]:
        if dealer["extra"] and dealer["status"]=="assigned":
            pit=next(t["pit"] for t in state["tables"] if t["id"]==dealer["table"])
            costs[pit]=costs.get(pit,0)+120/60*dt
    for r in state["receipts"].values():
        if r["status"] != "PREPARING" or state["minute"] < r["effective_at"]:
            continue
        a = r["command"]["action"]
        t = next(t for t in state["tables"] if t["id"] == a["table"])
        before=metrics(state,t["pit"])
        previous_minimum=t["minimum"]
        error = None
        if a["kind"] == "OPEN_TABLE":
            d = next(d for d in state["dealers"] if d["id"] == a["dealer"])
            if d["status"] != "reserved" or d["table"] != t["id"] or t["status"] != "opening":
                error = "Opening interrupted: dealer or table unavailable"
                if t["status"] == "opening": t["status"] = "closed"
                if d["status"] == "reserved": d.update(status="available",table=None)
            else:
                d.update(status="assigned",version=d["version"]+1)
                t.update(status="open",dealer=d["id"])
        elif t["status"] != "open":
            error = "Minimum change interrupted: table not running"
        else:
            t["minimum"] = a["minimum"]
            # New minimum applies to subsequent wagers; guests below their budget rejoin the queue.
            for p in state["players"]:
                if p["table"] == t["id"] and p["budget"] < t["minimum"]:
                    p.update(table=None,seated=None,arrived=state["minute"])
                    record(state,"TRANSFER_QUEUED",p["id"],t["id"])
        t["version"] += 1
        if not error: t["cooldown_until"] = state["minute"]+10
        r.update(status="FAILED" if error else "APPLIED",reason=error,minute=state["minute"])
        record(state,"COMMAND_FAILED" if error else "COMMAND_APPLIED",t["id"],error or r["id"])
        if not error:
            seat_waiting_guests(state)
            r["impact"]=dict(before=before,after=metrics(state,t["pit"]),
                             previous_minimum=previous_minimum,minimum=t["minimum"],pit=t["pit"])

    left = []
    for p in state["players"]:
        if p["table"] and state["minute"]-p["seated"] >= p["dwell"]:
            record(state,"LEFT_SEAT",p["id"],p["table"])
            state["observed_departures"].append(state["minute"])
            continue
        if not p["table"] and state["minute"]-p["arrived"] > 15:
            state["abandoned"] += 1
            record(state,"QUEUE_LEFT",p["id"],p["pit"])
            continue
        left.append(p)
    state["players"] = left
    for pit, rate in BASE_RATES.items():
        signal = state["signal"]
        if signal and signal["pit"] == pit and signal["expires"] > state["minute"]:
            rate += signal["extra_arrivals_per_minute"]
        n = int(rate*dt) + int(rng.random() < (rate*dt)%1)
        for _ in range(n):
            add_player(state,pit,rng)
            state["observed_arrivals"].append(dict(minute=state["minute"],pit=pit))
    seat_waiting_guests(state)
    state["observed_arrivals"] = [v for v in state["observed_arrivals"] if v["minute"] > state["minute"]-10]
    state["observed_departures"] = [v for v in state["observed_departures"] if v > state["minute"]-10]
    state["seq"] += 1
