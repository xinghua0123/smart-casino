"""Observable-state forecasting and constrained candidate evaluation (no LLM arithmetic)."""
import copy
import random
import uuid
from casino.domain import BASE_RATES, DEFAULT_CONSTRAINTS, EDGES, PITS
from casino.simulation import validate_action

VERSION = "scenario-engine-2.0-preview.1"


def constraints_checked(values):
    c = copy.deepcopy(DEFAULT_CONSTRAINTS)
    if not isinstance(values,dict) or set(values)-set(c):
        raise ValueError("Unsupported constraint fields")
    c.update(values)
    if c["pit"] not in PITS or c["priority"] not in ("wait","cost","theo"):
        raise ValueError("Unknown pit or priority")
    if type(c["horizon"]) is not int or c["horizon"] not in (15,30):
        raise ValueError("Horizon must be 15 or 30 minutes")
    if type(c["extra_staff"]) is not int or not 0 <= c["extra_staff"] <= 2:
        raise ValueError("Additional staff must be between 0 and 2")
    if type(c["max_wait"]) not in (int,float) or not 0 < c["max_wait"] <= 30:
        raise ValueError("Wait target must be between 0 and 30 minutes")
    if type(c["protect_vip"]) is not bool or not isinstance(c["excluded"],list) or any(not isinstance(v,str) for v in c["excluded"]):
        raise ValueError("Invalid VIP protection or table exclusions")
    c["max_wait"] = float(c["max_wait"])
    c["excluded"] = sorted(set(c["excluded"]))
    return c


def forecast(snapshot, action=None, factor=1.0, horizon=30):
    rng = random.Random(417)
    tables = copy.deepcopy(snapshot["tables"])
    players = []
    for p in snapshot["players"]:
        players.append(dict(id=p["id"],pit=p["pit"],budget=p["budget"],table=p["table"],
                            arrived=-(snapshot["minute"]-p["arrived"]),
                            leave_at=rng.expovariate(1/snapshot.get("session_minutes",18)) if p["table"] else None))
    rates = {}
    for pit in PITS:
        observed = sum(v["pit"] == pit for v in snapshot["observed_arrivals"])/min(10,max(1,snapshot["minute"]))
        rates[pit] = (.7*observed+.3*BASE_RATES[pit]) if snapshot["minute"] >= 3 else BASE_RATES[pit]
    signal = snapshot.get("signal")
    # External arrivals and dwell assumptions are drawn once per call with the same seed for every candidate.
    arrivals = {}
    for minute in range(1,horizon+1):
        rows = []
        for pit, rate in rates.items():
            if signal and signal["pit"] == pit and snapshot["minute"]+minute < signal["expires"]:
                rate = max(rate,BASE_RATES[pit]+signal["extra_arrivals_per_minute"])
            rate *= factor
            n = int(rate)+int(rng.random()<rate%1)
            base = {"entry":50,"main":100,"vip":500,"blackjack":25,"slots":5}[pit]
            for i in range(n):
                rows.append(dict(id=f"F{minute}-{pit}-{i}",pit=pit,budget=base*rng.choice([1,1,2,3,5]),table=None,
                                 arrived=minute,leave_at=None,dwell=rng.uniform(12,24)))
        arrivals[minute] = rows
    points = []
    served = 0
    abandoned = 0
    for minute in range(horizon+1):
        if minute:
            if action and minute == (2 if action["kind"] == "OPEN_TABLE" else 1):
                t = next(t for t in tables if t["id"] == action["table"])
                if action["kind"] == "OPEN_TABLE": t["status"] = "open"
                elif action["kind"] == "CONSOLIDATE_TABLE":
                    t["status"]="closed"
                    for p in players:
                        if p["table"]==t["id"]: p.update(table=None,leave_at=None,arrived=minute)
                else:
                    t["minimum"] = action["minimum"]
                    for p in players:
                        if p["table"] == t["id"] and p["budget"] < t["minimum"]:
                            p.update(table=None,leave_at=None,arrived=minute)
            abandoned += sum(not p["table"] and minute-p["arrived"]>15 for p in players)
            players = [p for p in players if (p["table"] and p["leave_at"] > minute) or (not p["table"] and minute-p["arrived"]<=15)]
            players.extend(copy.deepcopy(arrivals[minute]))
        for t in tables: t["occupied"] = sum(p["table"]==t["id"] for p in players)
        for p in players:
            if p["table"]: continue
            matches=[t for t in tables if t["pit"]==p["pit"] and t["status"]=="open" and t["occupied"]<t["capacity"] and t["minimum"]<=p["budget"]]
            if matches and minute:
                t=min(matches,key=lambda t:(t["occupied"]/t["capacity"],t["id"]))
                p.update(table=t["id"],leave_at=minute+p.get("dwell",18))
                t["occupied"]+=1
                served+=1
        pits={}
        for pit in PITS:
            queue=[p for p in players if p["pit"]==pit and not p["table"]]
            ts=[t for t in tables if t["pit"]==pit]
            cap=sum(t["capacity"] for t in ts if t["status"]=="open")
            pits[pit]=dict(queue=len(queue),wait=round(sum(minute-p["arrived"] for p in queue)/max(1,len(queue)),2),
                           seated=sum(t["occupied"] for t in ts),capacity=cap,
                           theo_hour=round(sum(t["occupied"]*t["minimum"]*1.4*EDGES[t["game"]]*1.5*60 for t in ts),2))
        points.append(dict(minute=minute,pits=pits,tables=[{k:t[k] for k in ("id","occupied","status","minimum")} for t in tables],
                           served=served,abandoned=abandoned))
    return points


def create_plan(snapshot, values, goal="Manual constraints"):
    c=constraints_checked(values)
    known={t["id"] for t in snapshot["tables"]}
    if set(c["excluded"])-known: raise ValueError("Excluded table does not exist")
    candidates=[dict(label="Keep current setup",action=None,reason=None)]
    for t in snapshot["tables"]:
        if t["pit"]!=c["pit"] or t["id"] in c["excluded"]: continue
        if t["status"]=="closed":
            dealer=next((d for d in snapshot["dealers"] if d["status"]=="available" and t["game"] in d["skills"] and (not d["extra"] or c["extra_staff"]>0)),None)
            a=dict(kind="OPEN_TABLE",table=t["id"],dealer=dealer["id"] if dealer else None)
            candidates.append(dict(label=f"Open {t['id']}",action=a,reason=validate_action(snapshot,a,c,t["version"])))
        elif t["status"]=="open" and t["game"]!="slots":
            if snapshot.get("signal") and snapshot["signal"]["kind"]=="TOUR_GROUP_DEPARTED" and snapshot["signal"]["expires"]>snapshot["minute"]:
                a=dict(kind="CONSOLIDATE_TABLE",table=t["id"])
                candidates.append(dict(label=f"Consolidate {t['id']} and release its dealer",action=a,reason=validate_action(snapshot,a,c,t["version"])))
            for low in sorted(set([max(t["minimum_floor"],t["minimum"]/2),min(t["maximum"],t["minimum"]*2)])):
                if low == t["minimum"]: continue
                a=dict(kind="SET_MINIMUM",table=t["id"],minimum=low)
                candidates.append(dict(label=f"{t['id']} · HK${low:g} minimum",action=a,reason=validate_action(snapshot,a,c,t["version"])))
    results=[]
    baseline=forecast(snapshot,horizon=30)
    for item in candidates:
        item.update(id=str(uuid.uuid4()),feasible=item["reason"] is None)
        if not item["feasible"]:
            results.append(item)
            continue
        action=item["action"]
        points=forecast(snapshot,action,horizon=30) if action else baseline
        low=forecast(snapshot,action,.8,30)[c["horizon"]]["pits"][c["pit"]]
        high=forecast(snapshot,action,1.2,30)[c["horizon"]]["pits"][c["pit"]]
        end=points[c["horizon"]]["pits"][c["pit"]]
        added=bool(action and action["kind"]=="OPEN_TABLE" and next(d for d in snapshot["dealers"] if d["id"]==action["dealer"])["extra"])
        cost=round(120*c["horizon"]/60,2) if added else 0
        item.update(points=points,metrics=end,wait_range=sorted([low["wait"],high["wait"]]),queue_range=sorted([low["queue"],high["queue"]]),
                    additional_cost=cost,target_met=end["wait"]<=c["max_wait"],
                    served=points[c["horizon"]]["served"],abandoned=points[c["horizon"]]["abandoned"],
                    table_version=next(t["version"] for t in snapshot["tables"] if t["id"]==action["table"]) if action else None)
        results.append(item)
    def rank(x):
        if not x["feasible"]: return (1,1,1,1)
        v=x["metrics"]
        target=not x["target_met"]
        if c["priority"]=="cost": return (0,target,x["additional_cost"],v["wait"])
        if c["priority"]=="theo": return (0,target,-v["theo_hour"],v["wait"])
        return (0,target,v["wait"],v["queue"])
    results.sort(key=rank)
    best=results[0]
    return dict(id=str(uuid.uuid4()),run_id=snapshot["run_id"],seq=snapshot["seq"],minute=snapshot["minute"],
                wall_time=snapshot["wall_time"],expires=snapshot["minute"]+10,constraints=c,goal=goal,
                engine=VERSION,seed=417,signal=copy.deepcopy(snapshot.get("signal")),candidates=results,
                recommended=best["id"],status="VALID",invalid_reason=None,replacement_id=None,
                explanation=("A feasible scenario meets the forecast wait target." if best["target_met"] else
                             "No evaluated scenario meets the wait target. Consider relaxing the table exclusions or allowing an additional dealer; constraints were not changed."),
                assumptions=["Observable arrivals + known dining signal; no hidden future events", "18-minute mean residual session; demo model", "Low/high demand = 0.8× / 1.2×; not statistical confidence", "Theo is modeled expected gaming win, not actual profit; cost assumes HK$120/hour per extra dealer", "One table action per scenario; this is a bounded candidate search, not a claim of global optimality"])
