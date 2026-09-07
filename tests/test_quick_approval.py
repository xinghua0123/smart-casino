"""One-click approval must change the physical floor, never only the UI ledger."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from casino.domain import initial_state, metrics
from casino.simulation import advance, control, receive_command
from operations_service.engine import Engine


class QuickApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=str(Path(self.temp.name)/'ops.db')
        self.e=Engine(self.path)
        self.s=initial_state()
        control(self.s,{'id':'surge','kind':'surge'})
        advance(self.s,0)
        self.feed()

    def tearDown(self):
        self.e.db.close()
        self.temp.cleanup()

    def feed(self):
        self.s['seq']+=1
        self.s['wall_time']=time.time()
        self.e.ingest(copy.deepcopy(self.s))

    def suggestion(self,kind):
        return next(t for t in self.e.all('task') if t.get('suggestion_kind')==kind and t['status']=='PENDING')

    def test_surge_generates_opening_and_busy_table_suggestions_without_duplicates(self):
        self.suggestion('OPEN_TABLE');self.suggestion('RAISE_MINIMUM')
        ids={t['id'] for t in self.e.all('task')}
        for _ in range(3): self.feed()
        self.assertEqual(ids,{t['id'] for t in self.e.all('task')})

    def test_approve_opens_immediately_and_reduces_queue_and_estimated_wait(self):
        t=self.suggestion('OPEN_TABLE')
        before=metrics(self.s,'main')
        approved=self.e.mutate_task(t['id'],'approve')
        self.assertEqual(approved['status'],'EXECUTING')
        self.e.mutate_task(t['id'],'approve')
        commands=self.e.commands()['commands']
        self.assertEqual(len(commands),1)
        receive_command(self.s,commands[0])
        receipt=self.s['receipts'][commands[0]['id']]
        self.assertEqual(receipt['status'],'APPLIED')
        self.assertLess(metrics(self.s,'main')['queue'],before['queue'])
        self.assertLess(metrics(self.s,'main')['estimated_wait'],before['estimated_wait'])
        self.assertEqual(metrics(self.s,'main')['capacity'],before['capacity']+7)
        after=copy.deepcopy(self.s)
        receive_command(self.s,commands[0])
        self.assertEqual(self.s,after)
        self.feed()
        self.assertEqual(self.e.get('task',t['id'])['status'],'OBSERVING')
        self.assertEqual(self.e.get('task',t['id'])['impact'],receipt['impact'])
        self.e.db.close();self.e=Engine(self.path)
        self.e.mutate_task(t['id'],'approve')
        self.assertEqual(self.e.commands()['commands'],[])

    def test_approve_changes_minimum_without_an_execute_step(self):
        t=self.suggestion('RAISE_MINIMUM')
        self.e.mutate_task(t['id'],'approve')
        receive_command(self.s,self.e.commands()['commands'][0])
        table=next(v for v in self.s['tables'] if v['id']==t['action']['table'])
        self.assertEqual(table['minimum'],t['action']['minimum'])
        self.assertEqual(self.s['receipts'][t['id']]['status'],'APPLIED')
        self.assertTrue(all(p['budget']>=table['minimum'] for p in self.s['players'] if p['table']==table['id']))

    def test_resource_loss_between_approval_and_application_is_not_success(self):
        t=self.suggestion('OPEN_TABLE')
        self.e.mutate_task(t['id'],'approve')
        command=self.e.commands()['commands'][0]
        control(self.s,{'id':'shortage','kind':'staff_shortage'})
        receive_command(self.s,command);self.feed()
        self.assertEqual(self.e.get('task',t['id'])['status'],'FAILED')
        self.assertEqual(next(v for v in self.s['tables'] if v['id']==t['action']['table'])['status'],'closed')

    def test_stale_approval_does_not_queue_a_command(self):
        t=self.suggestion('OPEN_TABLE')
        self.e.snapshot['wall_time']=time.time()-60
        with self.assertRaises(ValueError): self.e.mutate_task(t['id'],'approve')
        self.assertIsNone(self.e.get('task',t['id'])['command_id'])

    def test_staff_shortage_does_not_invent_an_opening(self):
        control(self.s,{'id':'shortage','kind':'staff_shortage'})
        self.s['minute']+=1
        self.feed()
        self.assertFalse(any(t['status']=='PENDING' and t.get('suggestion_kind')=='OPEN_TABLE' for t in self.e.all('task')))
