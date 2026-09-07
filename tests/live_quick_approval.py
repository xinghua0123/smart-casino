"""Real-stream test for the simplified demo. Resets the local simulation only."""
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

BASE='http://localhost:8090'
results=[]

def api(path,body=None):
    request=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as r: return json.load(r)

def wait(label,check,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            value=check()
            if value:
                results.append(label);print('PASS',label,flush=True);return value
        except (urllib.error.URLError,ConnectionError,TimeoutError): pass
        time.sleep(.5)
    raise AssertionError('Timed out: '+label)

def ready(kind):
    data=api('/overview')
    return next((t for t in data['tasks'] if t['run_id']==data['snapshot']['run_id'] and t['status']=='PENDING' and t.get('suggestion_kind')==kind),None)

def applied(tid):
    data=api('/overview')
    task=next(t for t in data['tasks'] if t['id']==tid)
    return (data,task) if task['status']=='OBSERVING' else None

wait('live telemetry ready',lambda:api('/health')['stream_fresh'])
old=api('/overview')['snapshot']['run_id']
api('/scenario',{'kind':'reset'})
wait('fresh demo reset',lambda:api('/overview')['snapshot']['run_id']!=old)
api('/scenario',{'kind':'surge'})
opening=wait('automatic opening suggestion',lambda:ready('OPEN_TABLE'))
wait('automatic busy-table minimum suggestion',lambda:ready('RAISE_MINIMUM'))
started=time.monotonic()
api('/tasks/'+opening['id'],{'operation':'approve'})
api('/tasks/'+opening['id'],{'operation':'approve'})
data,task=wait('one approval opens the table through the stream',lambda:applied(opening['id']),15)
latency=round(time.monotonic()-started,2)
impact=task['impact']
assert impact['after']['queue']<impact['before']['queue'],impact
assert impact['after']['estimated_wait']<impact['before']['estimated_wait'],impact
assert impact['after']['capacity']==impact['before']['capacity']+7
assert next(t for t in data['snapshot']['tables'] if t['id']==task['action']['table'])['status']=='open'
assert sum(e['status']=='OBSERVING' for e in task['timeline'])==1
results.append('confirmed queue and estimated wait decrease; no duplicate application')
minimum=wait('minimum suggestion remains independently actionable',lambda:ready('RAISE_MINIMUM'))
api('/tasks/'+minimum['id'],{'operation':'approve'})
data,changed=wait('one approval changes the actual table minimum',lambda:applied(minimum['id']),15)
assert next(t for t in data['snapshot']['tables'] if t['id']==changed['action']['table'])['minimum']==changed['action']['minimum']
report={'passed':results,'opening_confirmation_seconds':latency,'opening_impact':impact,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
Path('docs/QUICK_APPROVAL_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'confirmation_seconds':latency,'queue_before':impact['before']['queue'],'queue_after':impact['after']['queue'],'estimated_wait_before':impact['before']['estimated_wait'],'estimated_wait_after':impact['after']['estimated_wait']}),flush=True)
