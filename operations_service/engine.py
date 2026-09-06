"""Single-writer transactional action ledger; all public mutations run under the service lock."""
import copy
import json
import sqlite3
import time
import uuid
from casino.domain import metrics, PITS, DEFAULT_CONSTRAINTS
from casino.simulation import validate_action
from operations_service.planning import create_plan

FINAL={"CLOSED","REJECTED","EXPIRED","FAILED","CANCELLED"}
ACTIVE={"PENDING","ACCEPTED","EXECUTING","OBSERVING"}
STALE_SECONDS=15


class Engine:
    def __init__(self,path):
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT,body TEXT,PRIMARY KEY(kind,id))")
        self.db.commit()
        self.snapshot=self.get("snapshot","latest")
        self.db_error=None

    def get(self,kind,key):
        row=self.db.execute("SELECT body FROM records WHERE kind=? AND id=?",(kind,key)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self,kind,limit=None,statuses=None):
        query="SELECT body FROM records WHERE kind=?"
        args=[kind]
        if statuses:
            query+=" AND json_extract(body,'$.status') IN ("+','.join('?' for _ in statuses)+")"
            args.extend(statuses)
        query+=" ORDER BY rowid DESC"
        if limit is not None:
            query+=" LIMIT ?";args.append(limit)
        return [json.loads(r[0]) for r in self.db.execute(query,args)]

    def put(self,kind,value,key=None):
        self.db.execute("INSERT INTO records VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",(kind,key or value["id"],json.dumps(value)))

    def fresh(self):
        return bool(self.snapshot and not self.db_error and time.time()-self.snapshot["wall_time"]<STALE_SECONDS)

    def require_fresh(self):
        if not self.fresh(): raise ValueError("Live data is stale or unavailable. Wait for telemetry recovery.")

    def note(self,title,detail,plan_id=None):
        self.put("notice",dict(id=str(uuid.uuid4()),title=title,detail=detail,wall_time=time.time(),plan_id=plan_id,run_id=self.snapshot["run_id"] if self.snapshot else None))

    def plan(self,constraints,goal="Manual constraints"):
        self.require_fresh()
        plan=create_plan(self.snapshot,constraints,goal)
        plan["snapshot_metrics"]={p:metrics(self.snapshot,p) for p in PITS}
        self.put("plan",plan)
        self.db.commit()
        return plan

    def task_from_plan(self,plan_id,candidate_id,owner="Mei Wong"):
        self.require_fresh()
        plan=self.get("plan",plan_id)
        if not plan: raise ValueError("Plan not found")
        if plan["status"]!="VALID" or plan["run_id"]!=self.snapshot["run_id"] or self.snapshot["minute"]>plan["expires"]:
            raise ValueError("Plan has expired or changed; generate a fresh scenario")
        candidate=next((c for c in plan["candidates"] if c["id"]==candidate_id),None)
        if not candidate or not candidate["feasible"] or not candidate["action"]: raise ValueError("Select a feasible action scenario")
        for task in self.all("task"):
            if task["plan_id"]==plan_id and task["candidate_id"]==candidate_id: return task
            if task["status"] in ACTIVE and task["action"]["table"]==candidate["action"]["table"]: raise ValueError("This table already has an active task")
        error=validate_action(self.snapshot,candidate["action"],plan["constraints"],candidate["table_version"])
        if error: raise ValueError(error)
        task=dict(id=str(uuid.uuid4()),plan_id=plan_id,candidate_id=candidate_id,run_id=plan["run_id"],
                  title=candidate["label"],action=candidate["action"],constraints=plan["constraints"],
                  table_version=candidate["table_version"],created=self.snapshot["minute"],expires=plan["expires"],
                  owner=owner,status="PENDING",reason=None,timeline=[],observations={},
                  predicted=candidate["metrics"],before=metrics(self.snapshot,plan["constraints"]["pit"]),
                  command_id=None,effective=None,closed=None)
        self.transition(task,"PENDING","Created from a saved scenario; awaiting manager approval")
        self.db.commit()
        return task

    def transition(self,task,status,reason):
        task.update(status=status,reason=reason)
        task["timeline"].append(dict(status=status,reason=reason,minute=self.snapshot["minute"] if self.snapshot else 0,wall_time=time.time()))
        if status in FINAL: task["closed"]=self.snapshot["minute"] if self.snapshot else 0
        self.put("task",task)

    def mutate_task(self,tid,operation,owner=None,reason=None):
        task=self.get("task",tid)
        if not task: raise ValueError("Task not found")
        if operation in ("accept","execute"):
            self.require_fresh()
            target="ACCEPTED" if operation=="accept" else "EXECUTING"
            if task["status"]==target or operation=="execute" and task["command_id"]: return task
            if task["status"]!=("PENDING" if operation=="accept" else "ACCEPTED"): raise ValueError("Invalid task transition")
            if task["run_id"]!=self.snapshot["run_id"] or self.snapshot["minute"]>task["expires"]: raise ValueError("Task expired")
            error=validate_action(self.snapshot,task["action"],task["constraints"],task["table_version"])
            if error: raise ValueError(error)
            for other in self.all("task"):
                if other["id"]==tid or other["status"] not in ("ACCEPTED","EXECUTING"): continue
                if other["action"]["table"]==task["action"]["table"] or task["action"].get("dealer") and other["action"].get("dealer")==task["action"].get("dealer"):
                    raise ValueError("Table or dealer reserved by another accepted task")
            if owner:
                if owner not in ("Mei Wong","Alex Chan","Pit Manager"): raise ValueError("Unknown operator")
                task["owner"]=owner
            if operation=="execute": task["command_id"]=task["id"]
            self.transition(task,target,"Manager approved" if operation=="accept" else "Command queued; waiting for physical preparation and telemetry acknowledgement")
        elif operation in ("reject","cancel"):
            if task["status"] not in ("PENDING","ACCEPTED"): raise ValueError("Only unexecuted tasks can be rejected or cancelled")
            if not reason or not reason.strip(): raise ValueError("A reason is required")
            self.transition(task,"REJECTED" if operation=="reject" else "CANCELLED",reason[:500])
        elif operation=="close":
            if task["status"]!="OBSERVING" or "5" not in task["observations"]: raise ValueError("Complete the five-minute observation before closing")
            self.transition(task,"CLOSED","Reviewed by manager; observations are not causal attribution")
        else: raise ValueError("Unknown task operation")
        self.db.commit()
        return task

    def commands(self):
        commands=[]
        if self.fresh():
            for t in self.all("task"):
                if t["status"]=="EXECUTING":
                    commands.append(dict(id=t["command_id"],run_id=t["run_id"],expires=t["expires"],action=t["action"],constraints=t["constraints"],table_version=t["table_version"]))
        return dict(commands=commands,controls=[c for c in reversed(self.all("control")) if not c["done"]])

    def scenario(self,kind,table=None,request_id=None):
        if kind not in ("surge","reset","staff_shortage","restore_staff","outage","resume","close_table"): raise ValueError("Unknown scenario")
        if kind=="close_table" and (not self.snapshot or table not in [t["id"] for t in self.snapshot["tables"]]): raise ValueError("Unknown table")
        cid=request_id or str(uuid.uuid4())
        existing=self.get("control",cid)
        if existing: return existing
        c=dict(id=cid,kind=kind,table=table,done=False,wall_time=time.time())
        self.put("control",c)
        self.db.commit()
        return c

    def ingest(self,state):
        old=self.snapshot
        if old and state["run_id"]==old["run_id"] and state["seq"]<=old["seq"]: return
        if old and state["wall_time"]<old["wall_time"]: return
        self.snapshot=state
        self.put("snapshot",state,"latest")
        sample=dict(id=f"{state['run_id']}:{state['seq']}",run_id=state["run_id"],minute=state["minute"],wall_time=state["wall_time"],pits={p:metrics(state,p) for p in PITS})
        self.put("sample",sample)
        for c in self.all("control"):
            if c["id"] in state["control_receipts"] and not c["done"]:
                c["done"]=True; self.put("control",c)
        for task in self.all("task"):
            if task["status"] in FINAL: continue
            if task["run_id"]!=state["run_id"]:
                self.transition(task,"EXPIRED","Demo scenario reset; historical actions retained")
                continue
            receipt=state["receipts"].get(task["command_id"])
            if task["status"]=="EXECUTING" and receipt:
                if receipt["status"]=="APPLIED":
                    task["effective"]=receipt["minute"]
                    task["effective_metrics"]=metrics(state,task["constraints"]["pit"])
                    self.transition(task,"OBSERVING","Physical execution confirmed by the Kafka → RisingWave stream")
                elif receipt["status"]=="FAILED": self.transition(task,"FAILED",receipt["reason"])
            if task["status"] in ("PENDING","ACCEPTED"):
                reason=None
                if state["minute"]>task["expires"]: reason="Approval window expired"
                else: reason=validate_action(state,task["action"],task["constraints"],task["table_version"])
                if reason:
                    self.transition(task,"EXPIRED",reason)
                    self.note("Action needs review",task["title"]+": "+reason,task["plan_id"])
            if task["status"]=="EXECUTING" and not receipt and state["minute"]>task["expires"]+2:
                self.transition(task,"FAILED","No execution receipt before command expiry")
            if task["status"]=="OBSERVING":
                for period in (5,15):
                    if str(period) not in task["observations"] and state["minute"]>=task["effective"]+period:
                        samples=sorted([s for s in self.all("sample") if s["run_id"]==state["run_id"] and task["effective"]<=s["minute"]<=state["minute"]],key=lambda s:s["minute"])
                        complete=len(samples)>2 and all(b["minute"]-a["minute"]<1 and b["wall_time"]-a["wall_time"]<8 for a,b in zip(samples,samples[1:]))
                        observed=metrics(state,task["constraints"]["pit"])
                        base=task.get("effective_metrics",task["before"])
                        task["observations"][str(period)]=dict(metrics=observed,complete=complete,minute=state["minute"],
                            observed_theo=round(observed["theo_total"]-base.get("theo_total",0),2),
                            observed_extra_cost=round(observed["labor_cost_total"]-base.get("labor_cost_total",0),2))
                        self.put("task",task)
                if "15" in task["observations"]: self.transition(task,"CLOSED","15-minute observation complete; observed change is not causal lift")
        # Re-evaluate saved candidates against resource changes; create a replacement for the newest invalidated interactive plan.
        for index,plan in enumerate(self.all("plan",statuses=["VALID"])):
            if plan["status"]!="VALID": continue
            reason=None
            if plan["run_id"]!=state["run_id"]: reason="Demo scenario changed"
            elif state["minute"]>plan["expires"]: reason="Forecast approval window expired"
            else:
                related=[t for t in self.all("task") if t["plan_id"]==plan["id"] and t["status"] in ("EXECUTING","OBSERVING","CLOSED")]
                if not related:
                    for c in plan["candidates"]:
                        if c["feasible"] and c["action"]:
                            reason=validate_action(state,c["action"],plan["constraints"],c["table_version"])
                            if reason: break
                    before=plan["snapshot_metrics"][plan["constraints"]["pit"]]["queue"]
                    now=metrics(state,plan["constraints"]["pit"])["queue"]
                    if not reason and abs(now-before)>=12: reason=f"Queue changed materially: {before} → {now}"
            if reason:
                plan.update(status="INVALID",invalid_reason=reason)
                if index==0 and plan["run_id"]==state["run_id"] and "expired" not in reason.lower():
                    replacement=create_plan(state,plan["constraints"],plan["goal"])
                    replacement["snapshot_metrics"]={p:metrics(state,p) for p in PITS}
                    self.put("plan",replacement)
                    plan["replacement_id"]=replacement["id"]
                    self.note("Plan updated after a live event",reason+". Replacement keeps your constraints; approval is required.",replacement["id"])
                self.put("plan",plan)
        if self.fresh(): self.generate_opportunity()
        # Keep bounded observation samples and notification history; action and plan audit records are retained.
        self.db.execute("DELETE FROM records WHERE kind='sample' AND rowid NOT IN (SELECT rowid FROM records WHERE kind='sample' ORDER BY rowid DESC LIMIT 3000)")
        self.db.execute("DELETE FROM records WHERE kind='notice' AND rowid NOT IN (SELECT rowid FROM records WHERE kind='notice' ORDER BY rowid DESC LIMIT 100)")
        self.db.commit()

    def generate_opportunity(self):
        pit="main"
        if metrics(self.snapshot,pit)["queue"]<4: return
        tasks=[t for t in self.all("task") if t["run_id"]==self.snapshot["run_id"] and t["constraints"]["pit"]==pit]
        if any(t["status"] in ACTIVE or self.snapshot["minute"]-(t["closed"] or t["created"])<10 for t in tasks): return
        plan=create_plan(self.snapshot,DEFAULT_CONSTRAINTS,"Queue pressure: automatic operational opportunity")
        plan["snapshot_metrics"]={p:metrics(self.snapshot,p) for p in PITS}
        best=next((c for c in plan["candidates"] if c["feasible"] and c["action"]),None)
        baseline=next(c for c in plan["candidates"] if not c["action"])
        if not best or best["metrics"]["queue"]>=baseline["metrics"]["queue"] and best["metrics"]["wait"]>=baseline["metrics"]["wait"]: return
        self.put("plan",plan)
        self.task_from_plan(plan["id"],best["id"])
        self.note("Queue pressure: action ready for review",best["label"],plan["id"])

    def overview(self):
        plans=self.all("plan",limit=30)
        errors=[]
        samples=self.all("sample")
        for plan in plans[:100]:
            for h in (15,30):
                target=plan["minute"]+h
                eligible=[s for s in samples if s["run_id"]==plan["run_id"] and target<=s["minute"]<=target+.7]
                if not eligible: continue
                actual=min(eligible,key=lambda s:s["minute"])
                baseline=next(c for c in plan["candidates"] if not c["action"])
                affected=any(t["run_id"]==plan["run_id"] and t["effective"] is not None and plan["minute"]<=t["effective"]<=target for t in self.all("task"))
                intervening=sorted([s for s in samples if s["run_id"]==plan["run_id"] and plan["minute"]<=s["minute"]<=target],key=lambda s:s["minute"])
                gap=len(intervening)<2 or any(b["wall_time"]-a["wall_time"]>8 for a,b in zip(intervening,intervening[1:]))
                predicted=baseline["points"][h]["pits"][plan["constraints"]["pit"]]["queue"]
                observed=actual["pits"][plan["constraints"]["pit"]]["queue"]
                errors.append(dict(plan_id=plan["id"][:8],horizon=h,predicted_queue=predicted,observed_queue=observed,error=observed-predicted,
                                   evaluation="Excluded: intervening action" if affected else "Excluded: telemetry gap" if gap else "Comparable baseline"))
        return dict(snapshot=self.snapshot,fresh=self.fresh(),age=round(time.time()-self.snapshot["wall_time"],1) if self.snapshot else None,
                    db_error=self.db_error,tasks=self.all("task"),notices=[n for n in self.all("notice") if self.snapshot and n.get("run_id")==self.snapshot["run_id"]][:15],reviews=errors[:30],
                    plans=[{k:p[k] for k in ("id","goal","status","minute","invalid_reason","replacement_id")} for p in plans[:20]])
