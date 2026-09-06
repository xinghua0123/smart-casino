import copy
import tempfile
import time
import unittest
from pathlib import Path
from casino.domain import initial_state, DEFAULT_CONSTRAINTS, metrics
from casino.simulation import advance, control, receive_command, validate_action
from operations_service.planning import create_plan, forecast, constraints_checked
from operations_service.engine import Engine
from operations_service.copilot import parse_goal


class FloorTests(unittest.TestCase):
    def setUp(self):
        self.s=initial_state()
        self.s['wall_time']=time.time()
    def command(self):
        return dict(id='cmd',run_id=self.s['run_id'],expires=10,action=dict(kind='OPEN_TABLE',table='bac_08',dealer='F01'),constraints=dict(DEFAULT_CONSTRAINTS),table_version=1)
    def test_physical_invariants_over_turnover_and_surge(self):
        control(self.s,{'id':'surge','kind':'surge'})
        for _ in range(160):
            advance(self.s,.5)
            ids=[p['id'] for p in self.s['players']]
            self.assertEqual(len(ids),len(set(ids)))
            for t in self.s['tables']:
                self.assertLessEqual(t['occupied'],t['capacity'])
                self.assertEqual(t['occupied'],sum(p['table']==t['id'] for p in self.s['players']))
    def test_command_has_separate_preparation_and_effective_ack(self):
        c=self.command();receive_command(self.s,c);receive_command(self.s,c)
        self.assertEqual(len(self.s['receipts']),1)
        self.assertEqual(self.s['receipts']['cmd']['status'],'PREPARING')
        advance(self.s,1);self.assertEqual(self.s['receipts']['cmd']['status'],'PREPARING')
        advance(self.s,1.1);self.assertEqual(self.s['receipts']['cmd']['status'],'APPLIED')
        version=next(t['version'] for t in self.s['tables'] if t['id']=='bac_08')
        recovered=copy.deepcopy(self.s);receive_command(recovered,c)
        self.assertEqual(next(t['version'] for t in recovered['tables'] if t['id']=='bac_08'),version)
    def test_resource_loss_during_preparation_fails_without_opening(self):
        receive_command(self.s,self.command());control(self.s,{'id':'staff','kind':'staff_shortage'});advance(self.s,3)
        self.assertEqual(self.s['receipts']['cmd']['status'],'FAILED')
        self.assertEqual(next(t['status'] for t in self.s['tables'] if t['id']=='bac_08'),'closed')
    def test_extra_staff_is_not_free_relief(self):
        c=self.command();c['action']['dealer']='X01'
        self.assertIn('forbidden',validate_action(self.s,c['action'],c['constraints']))
    def test_vip_minimum_protected(self):
        a=dict(kind='SET_MINIMUM',table='bac_vip_01',minimum=1000)
        c={**DEFAULT_CONSTRAINTS,'pit':'vip'}
        self.assertIn('protected',validate_action(self.s,a,c))
    def test_expired_or_old_run_commands_rejected(self):
        c=self.command();c['run_id']='old';receive_command(self.s,c)
        self.assertEqual(self.s['receipts']['cmd']['status'],'FAILED')
    def test_control_idempotence(self):
        c={'id':'dining','kind':'surge'};control(self.s,c);n=len(self.s['players']);control(self.s,c)
        self.assertEqual(n,len(self.s['players']))
    def test_future_hidden_dwell_does_not_change_forecast(self):
        other=copy.deepcopy(self.s)
        for p in other['players']:p['dwell']=10000
        self.assertEqual(forecast(self.s),forecast(other))
    def test_comparable_scenarios_are_reproducible(self):
        self.assertEqual(forecast(self.s),forecast(self.s))
        for point in forecast(self.s):
            for t in point['tables']:
                cap=next(v['capacity'] for v in self.s['tables'] if v['id']==t['id'])
                self.assertLessEqual(t['occupied'],cap)
    def test_exclusions_and_infeasible_staff(self):
        control(self.s,{'id':'staff','kind':'staff_shortage'})
        plan=create_plan(self.s,{**DEFAULT_CONSTRAINTS,'excluded':['bac_08']})
        self.assertFalse(any(c['action'] and c['action']['table']=='bac_08' for c in plan['candidates']))
        self.assertTrue(all(not c['feasible'] for c in plan['candidates'] if c['action'] and c['action']['kind']=='OPEN_TABLE'))
    def test_goal_followup_preserves_constraints(self):
        a=parse_goal('主厅未来半小时，等待控制在五分钟内，不能增加人手，VIP最低投注额保持不变。',DEFAULT_CONSTRAINTS)
        b=parse_goal('不考虑 B08',a['constraints'])
        self.assertEqual(b['constraints']['excluded'],['bac_08'])
        self.assertTrue(b['constraints']['protect_vip'])
        self.assertEqual(b['constraints']['extra_staff'],0)
        self.assertEqual(b['constraints']['max_wait'],5)
    def test_unknown_fields_and_unsupported_templates(self):
        with self.assertRaises(ValueError): constraints_checked({'execute':'DROP TABLE'})
        with self.assertRaises(ValueError):parse_goal('buy ten aircraft',DEFAULT_CONSTRAINTS)
        with self.assertRaises(ValueError):create_plan(self.s,{**DEFAULT_CONSTRAINTS,'excluded':['bac_99']})


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=str(Path(self.temp.name)/'ops.db')
        self.e=Engine(self.path);self.s=initial_state();self.feed()
    def tearDown(self):
        self.e.db.close();self.temp.cleanup()
    def feed(self):
        self.s['wall_time']=time.time();self.s['seq']+=1;self.e.ingest(copy.deepcopy(self.s))
    def task(self):
        p=self.e.plan(DEFAULT_CONSTRAINTS)
        c=next(c for c in p['candidates'] if c['feasible'] and c['action'] and c['action']['kind']=='OPEN_TABLE')
        return self.e.task_from_plan(p['id'],c['id'])
    def test_creation_and_dispatch_idempotence(self):
        t=self.task()
        same=self.e.task_from_plan(t['plan_id'],t['candidate_id'])
        self.assertEqual(t['id'],same['id'])
        self.e.mutate_task(t['id'],'accept');self.e.mutate_task(t['id'],'execute');self.e.mutate_task(t['id'],'execute')
        self.assertEqual(len(self.e.commands()['commands']),1)
    def test_stale_telemetry_blocks_approval(self):
        t=self.task();self.e.snapshot['wall_time']=time.time()-60
        with self.assertRaises(ValueError):self.e.mutate_task(t['id'],'accept')
        self.assertEqual(self.e.commands()['commands'],[])
    def test_accepted_resource_conflict_and_plan_replacement(self):
        t=self.task();self.e.mutate_task(t['id'],'accept')
        control(self.s,{'id':'short','kind':'staff_shortage'});self.feed()
        self.assertEqual(self.e.get('task',t['id'])['status'],'EXPIRED')
        p=self.e.get('plan',t['plan_id'])
        self.assertEqual(p['status'],'INVALID');self.assertIsNotNone(p['replacement_id'])
    def test_restart_does_not_lose_dispatch_or_duplicate_ack(self):
        t=self.task();self.e.mutate_task(t['id'],'accept');self.e.mutate_task(t['id'],'execute')
        cmd=self.e.commands()['commands'][0]
        self.e.db.close();self.e=Engine(self.path)
        self.assertEqual(self.e.commands()['commands'][0]['id'],cmd['id'])
        receive_command(self.s,cmd);advance(self.s,2.1);self.feed();self.feed()
        actual=self.e.get('task',t['id'])
        self.assertEqual(actual['status'],'OBSERVING')
        self.assertEqual(sum(x['status']=='OBSERVING' for x in actual['timeline']),1)
    def test_observations_and_reset_preserve_history(self):
        t=self.task();self.e.mutate_task(t['id'],'accept');self.e.mutate_task(t['id'],'execute')
        receive_command(self.s,self.e.commands()['commands'][0]);advance(self.s,2.1);self.feed()
        for _ in range(12):advance(self.s,.5);self.feed()
        self.assertIn('5',self.e.get('task',t['id'])['observations'])
        control(self.s,{'id':'reset','kind':'reset'});self.feed()
        self.assertEqual(self.e.get('task',t['id'])['status'],'EXPIRED')
        self.assertGreater(len(self.e.get('task',t['id'])['timeline']),3)
    def test_illegal_transitions_rejected(self):
        t=self.task()
        with self.assertRaises(ValueError):self.e.mutate_task(t['id'],'execute')
        with self.assertRaises(ValueError):self.e.mutate_task(t['id'],'reject',reason='')
        self.e.mutate_task(t['id'],'reject',reason='Not needed')
        with self.assertRaises(ValueError):self.e.mutate_task(t['id'],'accept')
    def test_fifteen_minute_observation_closes_once_and_survives_restart(self):
        t=self.task();self.e.mutate_task(t['id'],'accept');self.e.mutate_task(t['id'],'execute')
        receive_command(self.s,self.e.commands()['commands'][0]);advance(self.s,2.1);self.feed()
        for _ in range(31):advance(self.s,.5);self.feed()
        actual=self.e.get('task',t['id'])
        self.assertEqual(actual['status'],'CLOSED')
        self.assertEqual(set(actual['observations']),{'5','15'})
        self.assertTrue(actual['observations']['15']['complete'])
        self.assertGreaterEqual(actual['observations']['15']['observed_theo'],0)
        self.e.db.close();self.e=Engine(self.path);self.feed()
        recovered=self.e.get('task',t['id'])
        self.assertEqual(recovered['observations'],actual['observations'])
        self.assertEqual(sum(x['status']=='CLOSED' for x in recovered['timeline']),1)
    def test_out_of_order_state_is_ignored(self):
        old=copy.deepcopy(self.s);self.s['minute']=5;self.feed();self.e.ingest(old)
        self.assertEqual(self.e.snapshot['minute'],5)


class ModelBoundaryTests(unittest.TestCase):
    def test_llm_structured_patch_and_failure_fallback(self):
        import sys
        import types
        from unittest.mock import patch
        def fake_response(**kwargs):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"priority":"cost"}'))])
        fake=types.SimpleNamespace(OpenAI=lambda **kwargs:types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_response))))
        with patch.dict(sys.modules,{'openai':fake}):
            result=parse_goal('Please prioritize operating expense',DEFAULT_CONSTRAINTS,{'api_key':'test-only-stub'})
        self.assertEqual(result['constraints']['priority'],'cost')
        self.assertTrue(result['mode'].startswith('LLM'))
        def broken(**kwargs): raise RuntimeError('offline')
        with patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=broken)}):
            result=parse_goal('不考虑 B08',DEFAULT_CONSTRAINTS,{'api_key':'test-only-stub'})
        self.assertIn('bac_08',result['constraints']['excluded'])
        self.assertIn('unavailable',result['mode'])
    def test_llm_cannot_inject_command_fields(self):
        import sys
        import types
        from unittest.mock import patch
        fake=types.SimpleNamespace(OpenAI=lambda **kwargs:types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs:types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"execute":"DROP TABLE x"}'))])))))
        with patch.dict(sys.modules,{'openai':fake}):
            result=parse_goal('不考虑 B08',DEFAULT_CONSTRAINTS,{'api_key':'test-only-stub'})
        self.assertNotIn('execute',result['constraints'])
        self.assertTrue(result['mode'].startswith('Template'))


class AdditionalBoundaryTests(unittest.TestCase):
    def test_llm_cannot_silently_relax_previous_protection(self):
        import sys
        import types
        from unittest.mock import patch
        fake=types.SimpleNamespace(OpenAI=lambda **kwargs:types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kwargs:types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"protect_vip":false,"extra_staff":2,"excluded":[]}'))])))))
        prior={**DEFAULT_CONSTRAINTS,'excluded':['bac_07']}
        with patch.dict(sys.modules,{'openai':fake}):
            result=parse_goal('不考虑 B08',prior,{'api_key':'test-only-stub'})
        self.assertTrue(result['constraints']['protect_vip'])
        self.assertEqual(result['constraints']['extra_staff'],0)
        self.assertEqual(result['constraints']['excluded'],['bac_07','bac_08'])


class MinimumAndCostTests(unittest.TestCase):
    def test_minimum_change_reseats_guests_over_budget(self):
        state=initial_state()
        command=dict(id='minimum',run_id=state['run_id'],expires=10,
                     action=dict(kind='SET_MINIMUM',table='bac_01',minimum=200),
                     constraints=dict(DEFAULT_CONSTRAINTS),table_version=1)
        receive_command(state,command)
        self.assertEqual(next(t['minimum'] for t in state['tables'] if t['id']=='bac_01'),100)
        advance(state,.6)
        self.assertEqual(state['receipts']['minimum']['status'],'APPLIED')
        self.assertEqual(next(t['minimum'] for t in state['tables'] if t['id']=='bac_01'),200)
        self.assertTrue(all(p['budget']>=200 for p in state['players'] if p['table']=='bac_01'))
        self.assertEqual(sum(t['occupied'] for t in state['tables']),sum(p['table'] is not None for p in state['players']))
    def test_extra_staff_cost_starts_after_execution(self):
        state=initial_state()
        command=dict(id='extra',run_id=state['run_id'],expires=10,
                     action=dict(kind='OPEN_TABLE',table='bac_08',dealer='X01'),
                     constraints={**DEFAULT_CONSTRAINTS,'extra_staff':1},table_version=1)
        receive_command(state,command);advance(state,2.1)
        self.assertEqual(metrics(state,'main')['labor_cost_total'],0)
        advance(state,1)
        self.assertEqual(metrics(state,'main')['labor_cost_total'],2)

if __name__=='__main__':unittest.main()
