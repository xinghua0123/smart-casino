"""Runs against the isolated local demo. Resets its floor scenario, never deletes its ledger."""
import json
import time
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

URL='http://localhost:8090'
RESULTS=[]

def api(path,body=None):
    request=urllib.request.Request(URL+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)

def wait_for(label,predicate,timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            result=predicate()
            if result:
                print('PASS',label,flush=True);RESULTS.append(label);return result
        except (urllib.error.URLError,TimeoutError,ConnectionError):pass
        time.sleep(1)
    raise AssertionError('Timed out: '+label)

def reset():
    old=api('/overview')['snapshot']['run_id'];api('/scenario',{'kind':'reset'})
    return wait_for('fresh reset through Kafka/RisingWave',lambda:(d:=api('/overview'))['snapshot']['run_id']!=old and d['fresh'] and d)

def open_task():
    c={'pit':'main','horizon':30,'max_wait':5,'extra_staff':0,'protect_vip':True,'excluded':['bac_07'],'priority':'wait'}
    p=api('/plans',{'constraints':c,'goal':'Acceptance: open B08 without additional staff'})
    candidate=next(c for c in p['candidates'] if c['feasible'] and c['action'] and c['action']['kind']=='OPEN_TABLE')
    t=api('/tasks',{'plan_id':p['id'],'candidate_id':candidate['id']})
    same=api('/tasks',{'plan_id':p['id'],'candidate_id':candidate['id']});assert same['id']==t['id']
    return t,p

def task(tid):return next(t for t in api('/overview')['tasks'] if t['id']==tid)

reset()
t,p=open_task()
api('/tasks/'+t['id'],{'operation':'accept','owner':'Mei Wong'})
api('/tasks/'+t['id'],{'operation':'execute'})
api('/tasks/'+t['id'],{'operation':'execute'})
wait_for('physical execution confirmation',lambda:task(t['id'])['status']=='OBSERVING',40)
d=api('/overview');assert next(x for x in d['snapshot']['tables'] if x['id']=='bac_08')['status']=='open'
assert len([e for e in task(t['id'])['timeline'] if e['status']=='OBSERVING'])==1
print('PASS duplicate dispatch has one effect',flush=True);RESULTS.append('duplicate dispatch has one effect')
wait_for('five-minute observation saved',lambda:'5' in task(t['id'])['observations'],50)
# Verify both the state producer and the ledger recover without duplicating the action.
subprocess.run(['docker','compose','-f','docker-compose.yml','-f','docker-compose.dev.yml','restart','data-producer','operations-service'],check=True,stdout=subprocess.DEVNULL)
wait_for('services recover live telemetry',lambda:api('/overview')['fresh'],40)
assert task(t['id'])['status'] in ('OBSERVING','CLOSED')
assert len([e for e in task(t['id'])['timeline'] if e['status']=='OBSERVING'])==1
print('PASS restart preserves action and receipt',flush=True);RESULTS.append('restart preserves action and receipt')
reset();t,p=open_task();api('/tasks/'+t['id'],{'operation':'accept'})
api('/scenario',{'kind':'staff_shortage'})
wait_for('staff loss expires accepted action',lambda:task(t['id'])['status']=='EXPIRED',30)
updated=api('/plans/'+p['id']);assert updated['status']=='INVALID' and updated['replacement_id']
replacement=api('/plans/'+updated['replacement_id']);assert replacement['constraints']==p['constraints']
assert not any(c['feasible'] and c['action'] and c['action']['kind']=='OPEN_TABLE' for c in replacement['candidates'])
print('PASS automatic replanning retains constraints',flush=True);RESULTS.append('automatic replanning retains constraints')
api('/scenario',{'kind':'outage'})
wait_for('outage marks data stale',lambda:not api('/overview')['fresh'],30)
try:api('/plans',{'constraints':p['constraints']})
except urllib.error.HTTPError as exc:assert exc.code==400
else:raise AssertionError('stale plan unexpectedly accepted')
print('PASS stale data blocks planning',flush=True);RESULTS.append('stale data blocks planning')
api('/scenario',{'kind':'resume'})
wait_for('telemetry recovery',lambda:api('/overview')['fresh'],30)
reset();api('/scenario',{'kind':'surge'})
wait_for('dining signal and automatic opportunity',lambda:(d:=api('/overview'))['snapshot']['signal'] and any(t['status']=='PENDING' and t['run_id']==d['snapshot']['run_id'] for t in d['tasks']),30)
p=api('/plans',{'constraints':{'excluded':['bac_08']},'goal':'Acceptance review snapshot'})
assert not any(c['action'] and c['action']['table']=='bac_08' for c in p['candidates'])
wait_for('saved forecast reaches error review',lambda:any(r['plan_id']==p['id'][:8] for r in api('/overview')['reviews']),120)
Path('docs/LIVE_TEST_RESULTS.json').write_text(json.dumps({'passed':RESULTS,'total':len(RESULTS),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
print('All live acceptance checks passed.',flush=True)
