"""Durable producer: one authoritative floor, command receipts, and Kafka telemetry."""
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from urllib.request import urlopen
from kafka import KafkaProducer
from casino.domain import initial_state, public_state, EDGES
from casino.simulation import advance, control, receive_command

STATE_FILE=os.getenv("SIM_STATE","/state/floor.json")
OPS=os.getenv("OPS_URL","http://operations-service:8090")


def save(state):
    os.makedirs(os.path.dirname(STATE_FILE),exist_ok=True)
    temp=STATE_FILE+".tmp"
    with open(temp,"w") as f:
        json.dump(state,f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,STATE_FILE)


def telemetry(state,producer):
    state["wall_time"]=time.time()
    public=public_state(state)
    common=dict(run_id=state["run_id"],seq=state["seq"],wall_time=state["wall_time"],sim_minute=state["minute"])
    producer.send("operational_events",key=b"snapshot",value=dict(common,event_id=str(uuid.uuid4()),kind="snapshot",entity_id="floor",state_json=json.dumps(public)))
    for t in state["tables"]:
        producer.send("operational_events",key=t["id"].encode(),value=dict(common,event_id=str(uuid.uuid4()),kind="table",entity_id=t["id"],table_id=t["id"],pit=t["pit"],
                      status=t["status"],occupied=t["occupied"],capacity=t["capacity"],minimum=t["minimum"],state_json=json.dumps(t)))


def gaming(state,producer):
    rng=random.Random(state["seq"]+30)
    tables={t["id"]:t for t in state["tables"]}
    ts=datetime.now(timezone.utc).isoformat()
    for p in state["players"]:
        if not p["table"] or rng.random()>.25: continue
        t=tables[p["table"]]
        bet=round(min(t["maximum"],p["budget"],t["minimum"]*rng.uniform(1,1.5)),2)
        won=rng.random()<.49
        p["gaming_spend"]+=bet; p["bets"]+=1
        state["total_theo"]+=bet*EDGES[t["game"]]
        t["theo"]+=bet*EDGES[t["game"]]
        arch="high_roller" if p["pit"]=="vip" else "emerging" if p["budget"]>=200 else "regular" if p["pit"]=="main" else "casual"
        tier="diamond" if arch=="high_roller" else "gold" if arch=="emerging" else "silver" if arch=="regular" else "bronze"
        # Scenario-prefixed identity prevents reset runs from merging cumulative player history.
        pid=state["run_id"][:8]+"-"+p["id"]
        base=dict(event_id=str(uuid.uuid4()),player_id=pid,tier=tier,ts=ts)
        producer.send("gaming_events",value=dict(base,event_type="gaming",archetype=arch,game_type=t["game"],table_id=t["id"],table_x=t["x"],table_y=t["y"],
                      limit_min=t["minimum"],limit_max=t["maximum"],bet_amount=bet,payout=round(bet*1.98,2) if won else 0,won=won,session_count=p["bets"],total_gaming_spend=round(p["gaming_spend"],2)))
        if rng.random()<.08:
            spend=round(rng.uniform(20,100),2); p["fnb_spend"]+=spend
            producer.send("fnb_events",value=dict(base,event_id=str(uuid.uuid4()),event_type="fnb",item="dinner",spend_amount=spend,total_fnb_spend=p["fnb_spend"]))
        if rng.random()<.02:
            producer.send("hotel_events",value=dict(base,event_id=str(uuid.uuid4()),event_type="hotel",action="checkin",charge_amount=200))


def main():
    try:
        with open(STATE_FILE) as f: state=json.load(f)
    except FileNotFoundError: state=initial_state()
    producer=None
    while producer is None:
        try: producer=KafkaProducer(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS","kafka:9092"),value_serializer=lambda v:json.dumps(v).encode(),acks="all",retries=5)
        except Exception: time.sleep(2)
    while True:
        start=time.monotonic()
        try:
            try:
                with urlopen(OPS+"/commands",timeout=2) as r: batch=json.load(r)
                for c in batch["controls"]: control(state,c)
                if not state["outage"]:
                    for c in batch["commands"]: receive_command(state,c)
            except Exception as exc:
                print("Command channel:",type(exc).__name__,flush=True)
            if not state["outage"]:
                advance(state,state["speed"]/60)
                gaming(state,producer)
                # Persist before publishing: recovery can replay telemetry, never an applied action.
                state["wall_time"]=time.time()
                save(state)
                telemetry(state,producer)
                producer.flush(timeout=5)
            else:
                # The operational clock continues through telemetry outages; no hidden observations are fabricated.
                state["minute"]+=state["speed"]/60
                state["seq"]+=1
                save(state)
        except Exception as exc:
            print("Simulation tick:",repr(exc),flush=True)
        time.sleep(max(.05,1-(time.monotonic()-start)))


if __name__=="__main__": main()
