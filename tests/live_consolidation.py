"""Exercise departure and one-click consolidation through the local live stream."""
import json
import time
import urllib.request


def api(path,body=None):
    req=urllib.request.Request('http://localhost:8090'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)


def wait(check):
    for _ in range(60):
        d=api('/overview')
        if check(d):return d
        time.sleep(.5)
    raise AssertionError('Timed out')


old=api('/overview')['snapshot']['run_id']
api('/scenario',{'kind':'reset'})
wait(lambda d:d['snapshot']['run_id']!=old and d['fresh'])
api('/scenario',{'kind':'tour_departure'})
def pending(d):
    return next((t for t in d['tasks'] if t['run_id']==d['snapshot']['run_id'] and t['status']=='PENDING' and t.get('suggestion_kind')=='CONSOLIDATE_TABLE'),None)
d=wait(pending)
t=pending(d)
api('/tasks/'+t['id'],{'operation':'approve'})
d=wait(lambda d:any(v['id']==t['id'] and v['status']=='OBSERVING' for v in d['tasks']))
t=next(v for v in d['tasks'] if v['id']==t['id'])
i=t['impact']
assert i['after']['queue']==i['before']['queue']==0
assert i['after']['seated']==i['before']['seated']
assert i['after']['open_tables']==i['before']['open_tables']-1
assert i['after']['available_dealers']==i['before']['available_dealers']+1
assert i['after']['occupancy']>i['before']['occupancy']
print(json.dumps({'title':t['title'],'impact':i},indent=2))
