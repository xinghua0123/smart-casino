"""Consolidation restores spare staff without stranding or losing guests."""
import copy
import tempfile
import time
import unittest
from casino.domain import initial_state, metrics, DEFAULT_CONSTRAINTS
from casino.simulation import control, receive_command, validate_action
from operations_service.engine import Engine


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.e=Engine(self.tmp.name+'/ops.db')
        self.s=initial_state()
        control(self.s,{'id':'departure','kind':'tour_departure'})
        self.feed()

    def tearDown(self):
        self.e.db.close()
        self.tmp.cleanup()

    def feed(self):
        self.s['seq']+=1
        self.s['wall_time']=time.time()
        self.e.ingest(copy.deepcopy(self.s))

    def task(self):
        return next(t for t in self.e.all('task') if t.get('suggestion_kind')=='CONSOLIDATE_TABLE' and t['status']=='PENDING')

    def test_departure_approve_moves_guests_and_releases_dealer_once(self):
        self.assertEqual(metrics(self.s,'main')['seated'],6)
        task=self.task()
        table=next(t for t in self.s['tables'] if t['id']==task['action']['table'])
        dealer=table['dealer']
        guests={p['id']:(p['seated'],p['dwell']) for p in self.s['players']}
        before=metrics(self.s,'main')
        self.e.mutate_task(task['id'],'approve')
        cmd=self.e.commands()['commands'][0]
        receive_command(self.s,cmd)
        self.feed()
        after=metrics(self.s,'main')
        self.assertEqual(after['queue'],0)
        self.assertEqual(after['capacity'],before['capacity']-7)
        self.assertEqual(after['seated'],before['seated'])
        self.assertGreater(after['occupancy'],before['occupancy'])
        self.assertEqual(guests,{p['id']:(p['seated'],p['dwell']) for p in self.s['players']})
        self.assertEqual(next(d for d in self.s['dealers'] if d['id']==dealer)['status'],'available')
        self.assertEqual(self.e.get('task',task['id'])['status'],'OBSERVING')
        state=copy.deepcopy(self.s)
        receive_command(self.s,cmd)
        self.assertEqual(state,self.s)

    def test_incompatible_destination_seats_block_approval(self):
        task=self.task()
        for p in self.s['players']:
            if p['table']==task['action']['table']:p['budget']=100
        for t in self.s['tables']:
            if t['pit']=='main' and t['id']!=task['action']['table']:t['minimum']=500
        self.assertIn('compatible',validate_action(self.s,task['action'],DEFAULT_CONSTRAINTS))

    def test_new_queue_between_approval_and_application_fails_without_moves(self):
        task=self.task()
        self.e.mutate_task(task['id'],'approve')
        cmd=self.e.commands()['commands'][0]
        control(self.s,{'id':'surge','kind':'surge'})
        before=copy.deepcopy(self.s['tables'])
        receive_command(self.s,cmd)
        self.assertEqual(self.s['receipts'][cmd['id']]['status'],'FAILED')
        self.assertEqual(before,self.s['tables'])
