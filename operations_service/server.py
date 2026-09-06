import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import psycopg2
from operations_service.engine import Engine
from operations_service.copilot import parse_goal

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
LOCK=threading.RLock()
ENGINE=None


def poll():
    conn=None
    while True:
        try:
            if conn is None or conn.closed:
                conn=psycopg2.connect(host=os.getenv("RISINGWAVE_HOST","risingwave"),port=os.getenv("RISINGWAVE_PORT","4566"),user="root",dbname="dev",connect_timeout=4)
                conn.autocommit=True
            with conn.cursor() as cursor:
                cursor.execute("SELECT state_json FROM mv_ops_latest_snapshot")
                row=cursor.fetchone()
            with LOCK:
                ENGINE.db_error=None
                if row: ENGINE.ingest(json.loads(row[0]))
        except Exception as exc:
            logging.warning("Snapshot ingestion unavailable: %s",type(exc).__name__)
            with LOCK: ENGINE.db_error="RisingWave snapshot unavailable; actions paused"
            if conn:
                conn.close(); conn=None
        time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def respond(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            path=urlparse(self.path).path
            with LOCK:
                if path=="/health": result=dict(status="ok",stream_fresh=ENGINE.fresh())
                elif path=="/overview": result=ENGINE.overview()
                elif path=="/commands": result=ENGINE.commands()
                elif path.startswith("/plans/"):
                    result=ENGINE.get("plan",path.split("/")[-1])
                    if not result: raise ValueError("Plan not found")
                else: return self.respond({"error":"Not found"},404)
            self.respond(result)
        except ValueError as exc: self.respond({"error":str(exc)},400)
        except Exception:
            logging.exception("GET failed")
            self.respond({"error":"Service request failed"},500)

    def do_POST(self):
        try:
            size=int(self.headers.get("Content-Length","0"))
            if not 0<size<32000: raise ValueError("Invalid request size")
            body=json.loads(self.rfile.read(size))
            path=urlparse(self.path).path
            if path=="/parse-goal":
                result=parse_goal(body["text"],body["constraints"],body.get("llm"))
            else:
                with LOCK:
                    if path=="/plans": result=ENGINE.plan(body["constraints"],body.get("goal","Manual constraints"))
                    elif path=="/tasks": result=ENGINE.task_from_plan(body["plan_id"],body["candidate_id"],body.get("owner","Mei Wong"))
                    elif path.startswith("/tasks/"): result=ENGINE.mutate_task(path.split("/")[-1],body["operation"],body.get("owner"),body.get("reason"))
                    elif path=="/scenario": result=ENGINE.scenario(body["kind"],body.get("table"),body.get("request_id"))
                    else: return self.respond({"error":"Not found"},404)
            self.respond(result)
        except (ValueError,KeyError,TypeError) as exc: self.respond({"error":str(exc)},400)
        except Exception:
            with LOCK: ENGINE.db.rollback()
            logging.exception("POST failed")
            self.respond({"error":"Service request failed"},500)


def main():
    global ENGINE
    path=os.getenv("OPS_DB","/state/operations.sqlite")
    os.makedirs(os.path.dirname(path),exist_ok=True)
    ENGINE=Engine(path)
    threading.Thread(target=poll,daemon=True).start()
    logging.info("Operations API ready on :8090")
    ThreadingHTTPServer(("0.0.0.0",8090),Handler).serve_forever()


if __name__=="__main__": main()
