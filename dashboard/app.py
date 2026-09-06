"""Smart Casino Floor · operations workspace. The backend owns every durable action."""
import html
import json
import os
import uuid
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

OPS=os.getenv('OPS_URL','http://localhost:8090')
PITS={'main':'Main baccarat','entry':'Entry baccarat','vip':'VIP baccarat','blackjack':'Blackjack','slots':'Slots'}
DEFAULT={'pit':'main','horizon':30,'max_wait':5.0,'extra_staff':0,'protect_vip':True,'excluded':[],'priority':'wait'}
st.set_page_config(page_title='Floor Operations · Smart Casino',page_icon='♠',layout='wide')
st.markdown('''<style>
.stApp{background:#0c1220;color:#e6edf6} [data-testid="stSidebar"]{background:#111c2d}
.block-container{padding-top:4rem;max-width:1550px} h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#152135;border:1px solid #26374f;padding:16px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#9daec5} [data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:#29394e!important;border-radius:12px}
.stButton>button{border-radius:8px} .eyebrow{color:#7f9bb8;font-size:12px;letter-spacing:2px}
.hero{display:flex;justify-content:space-between;align-items:center;margin:0 0 18px}.hero h1{font-size:34px;margin:4px 0}
.badge{background:#173d36;color:#87e1bd;padding:7px 12px;border-radius:20px;font-size:12px}
.small{font-size:13px;color:#9daec5}.stTabs [data-baseweb="tab-list"]{gap:20px}
</style>''',unsafe_allow_html=True)


def api(path,body=None):
    req=Request(OPS+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=40) as r: return json.load(r)
    except HTTPError as exc:
        try: message=json.load(exc).get('error',str(exc))
        except Exception: message=str(exc)
        raise ValueError(message)
    except (URLError,TimeoutError) as exc: raise ValueError('Operations service unavailable. The page will retry automatically.') from exc


def action(path,body):
    try:
        result=api(path,body)
        st.session_state.flash='Saved. Live state will update after the next streamed event.'
        return result
    except ValueError as exc:
        st.error(str(exc))
        return None


if 'constraints' not in st.session_state: st.session_state.constraints=dict(DEFAULT)
for key,value in st.session_state.constraints.items():
    if 'constraint_'+key not in st.session_state: st.session_state['constraint_'+key]=value

with st.sidebar:
    st.markdown('### ♠ Smart Casino')
    st.caption('OPERATIONS WORKSPACE · 2.0')
    st.page_link('app.py',label='Floor operations',icon='🗺️')
    st.page_link('pages/1_Player_Analytics.py',label='Player analytics & chat',icon='📊')
    if st.button('Start guided tour',key='start_tour',use_container_width=True):
        st.session_state.tour_request=str(uuid.uuid4())
    st.divider()
    st.markdown('**Demo scenarios**')
    st.caption('10× clock · 1 demo minute = 6 real seconds. Actions affect this local simulation.')
    for label,kind in [('Dining group arrives','surge'),('Reassign relief dealers','staff_shortage'),('Restore relief dealers','restore_staff'),('Interrupt telemetry','outage'),('Resume telemetry','resume'),('Reset floor scenario','reset')]:
        if st.button(label,key='scenario_'+kind,use_container_width=True):
            if action('/scenario',{'kind':kind,'request_id':str(uuid.uuid4())}): st.success('Scenario queued')
    st.divider()
    with st.expander('AI connection',expanded=False):
        provider=st.selectbox('Provider',['OpenAI','Claude','OpenRouter'],key='ops_provider')
        key=st.text_input('API key',type='password',key='ops_key')
        model=st.text_input('Model',value='gpt-4o-mini',key='ops_model')
        base=st.text_input('Base URL (optional)',value='',key='ops_base')
        st.caption('Optional. Without a key, supported goal templates and manual constraints remain available. Keys are not written to the action ledger.')
    st.caption('Kafka → RisingWave → Operations API\n\nVersion 2.0 · 1.0 baseline preserved')


def map_figure(snapshot,point,layer,selected):
    fig=go.Figure()
    projected={t['id']:t for t in point['tables']} if point else {}
    zones=[('entry',2.4,9.7),('vip',7.6,9.7),('slots',2.4,5.6),('main',7.6,5.6),('blackjack',7.6,2.7)]
    pits=point['pits'] if point else snapshot['pits']
    for pit,x,y in zones:
        q=pits[pit]['queue']
        fig.add_annotation(x=x,y=y,text=f"<b>{PITS[pit].upper()}</b> · {q} waiting",showarrow=False,font=dict(size=11,color='#abc0d7'))
    for t in snapshot['tables']:
        v=projected.get(t['id'],t)
        occ=v['occupied']/t['capacity']
        closed=v['status']!='open'
        color='#273449' if closed else '#ec875b' if occ>=.85 else '#4ab9a6' if occ>.4 else '#6c91b7'
        if layer=='Queue pressure' and not closed: color='#ec875b' if pits[t['pit']]['queue']>5 else '#4ab9a6'
        if layer=='Theo' and not closed:
            edge={'baccarat':.0115,'blackjack':.0075,'slots':.075}[t['game']]
            color='#d6af63' if v['occupied']*v['minimum']*edge>15 else '#607faf'
        short=t['id'].replace('bac_left_','E').replace('bac_vip_','V').replace('bac_','B').replace('slots_','S').replace('bj_','J')
        text=f"<b>{short}</b><br>HK${v['minimum']:g}<br>{v['occupied']}/{t['capacity']}" if not closed else f"<b>{short}</b><br>{v['status'].upper()}"
        table_theo=v['occupied']*v['minimum']*1.4*{'baccarat':.0115,'blackjack':.0075,'slots':.075}[t['game']]*1.5*60
        fig.add_trace(go.Scatter(x=[t['x']],y=[t['y']],mode='markers+text',text=[text],textposition='middle center',textfont=dict(size=9,color='#ffffff'),
                                customdata=[[t['id']]],marker=dict(size=51,symbol='circle' if t['game']=='blackjack' else 'square',color=color,line=dict(color='#f5d48c' if selected==t['id'] else '#3a4b63',width=3 if selected==t['id'] else 1)),
                                hovertemplate=f"<b>{t['id']}</b><br>{v['status']} · {v['occupied']}/{t['capacity']} seats<br>HK${v['minimum']:g} minimum<br>Modeled Theo HK${table_theo:,.0f}/hr<extra></extra>",showlegend=False))
    fig.update_layout(height=570,margin=dict(t=5,b=10,l=0,r=0),paper_bgcolor='#101b2c',plot_bgcolor='#101b2c',
                      xaxis=dict(visible=False,range=[-.3,10.4]),yaxis=dict(visible=False,range=[-.1,10.4]),clickmode='event+select',uirevision='floor-map')
    return fig


def render_planner(snapshot,fresh):
    left,right=st.columns([1.1,1])
    with left:
        st.markdown('#### Give the floor a goal')
        st.caption('Describe the outcome and constraints. Review the interpreted fields before evaluating scenarios.')
        with st.form('goal_form'):
            goal=st.text_area('Operational goal',value='For the next 30 minutes, keep the main floor wait within 5 minutes, with no additional staff. Keep VIP minimums unchanged.',height=100)
            parsed=st.form_submit_button('Interpret goal',use_container_width=True)
        if parsed:
            llm={'provider':st.session_state.ops_provider,'api_key':st.session_state.ops_key,'model':st.session_state.ops_model,'base_url':st.session_state.ops_base}
            if llm['provider']=='OpenRouter' and not llm['base_url']: llm['base_url']='https://openrouter.ai/api/v1'
            with st.spinner('Interpreting constraints…'):
                result=action('/parse-goal',{'text':goal,'constraints':st.session_state.constraints,'llm':llm})
            if result:
                st.session_state.constraints=result['constraints']
                for k,v in result['constraints'].items(): st.session_state['constraint_'+k]=v
                st.session_state.parse_result=result
                st.session_state.goal=goal
                if result['changes']: st.session_state.pop('plan_id',None)
        result=st.session_state.get('parse_result')
        if result:
            st.caption(result['mode'])
            if result['warning']: st.warning(result['warning'])
            if result['changes']:
                labels={'excluded':'Excluded tables','pit':'Area','horizon':'Horizon (min)','max_wait':'Wait target (min)','extra_staff':'Additional dealers','protect_vip':'Protect VIP minimums','priority':'Priority'}
                for field,value in result['changes'].items():
                    display=', '.join(value) if isinstance(value,list) else PITS.get(value,value) if isinstance(value,str) else str(value)
                    st.write(f"{labels[field]}: {display}")
        else: st.caption('Template mode available without an API key. Follow-up example: Exclude B08.')
    with right:
        st.markdown('#### Confirm constraints')
        with st.form('constraints_form'):
            a,b=st.columns(2)
            pit=a.selectbox('Area',list(PITS),format_func=PITS.get,key='constraint_pit')
            horizon=b.selectbox('Forecast horizon',[15,30],format_func=lambda v:f'{v} minutes',key='constraint_horizon')
            wait=a.number_input('Maximum wait (min)',min_value=.5,max_value=30.0,step=.5,key='constraint_max_wait')
            staff=b.number_input('Additional dealers',min_value=0,max_value=2,step=1,key='constraint_extra_staff')
            priority=st.selectbox('First priority',['wait','cost','theo'],format_func=lambda v:{'wait':'Reduce waiting','cost':'Limit additional cost','theo':'Increase modeled Theo'}[v],key='constraint_priority')
            protect=st.checkbox('Keep VIP minimums unchanged',key='constraint_protect_vip')
            excluded=st.multiselect('Exclude tables',[t['id'] for t in snapshot['tables']],key='constraint_excluded')
            submit=st.form_submit_button('Evaluate feasible scenarios',type='primary',use_container_width=True,disabled=not fresh)
        if submit:
            c=dict(pit=pit,horizon=horizon,max_wait=float(wait),extra_staff=int(staff),priority=priority,protect_vip=protect,excluded=excluded)
            st.session_state.constraints=c
            with st.spinner('Evaluating the same demand scenarios for every candidate…'):
                plan=action('/plans',{'constraints':c,'goal':st.session_state.get('goal','Manual constraints')})
            if plan:
                st.session_state.plan_id=plan['id']
                st.session_state.candidate_id=plan['recommended']
    plan_id=st.session_state.get('plan_id')
    if not plan_id:
        st.info('Evaluate a plan to preview future occupancy and compare actions. The baseline is evaluated with the same external demand assumptions.')
        return None,None
    try: plan=api('/plans/'+plan_id)
    except ValueError as exc: st.warning(str(exc)); return None,None
    if plan['status']!='VALID':
        st.warning('Plan needs review: '+str(plan['invalid_reason']))
        if plan.get('replacement_id') and st.button('Review automatically updated plan'):
            st.session_state.plan_id=plan['replacement_id']; st.session_state.pop('candidate_id',None); st.rerun()
    else:
        st.caption(f"Snapshot #{plan['seq']} · demo minute {plan['minute']:.1f} · approval window ends {plan['expires']:.1f} · {plan['engine']}")
    st.info(plan['explanation'])
    feasible=[c for c in plan['candidates'] if c['feasible']]
    current=st.session_state.get('candidate_id',plan['recommended'])
    ids=[c['id'] for c in feasible]
    if current not in ids: current=ids[0]
    chosen=st.selectbox('Scenario to preview',ids,index=ids.index(current),format_func=lambda v:next(('★ ' if c['id']==plan['recommended'] else '')+c['label'] for c in feasible if c['id']==v),key='scenario_select_'+plan['id'])
    st.session_state.candidate_id=chosen
    candidate=next(c for c in feasible if c['id']==chosen)
    rows=[]
    for c in feasible:
        rows.append({'Scenario':c['label'],'Wait / min':c['metrics']['wait'],'Queue':c['metrics']['queue'],'Theo / hr (HK$)':c['metrics']['theo_hour'],'Extra cost (HK$)':c['additional_cost'],'Wait target':'Met' if c['target_met'] else 'Not met','New seatings · floor':c['served'],'Queue departures · floor':c['abandoned']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    a,b=st.columns([3,1])
    a.caption(f"Demand sensitivity for selected scenario: wait {candidate['wait_range'][0]:.1f}–{candidate['wait_range'][1]:.1f} min · queue {candidate['queue_range'][0]}–{candidate['queue_range'][1]}. Scenario range, not statistical confidence.")
    if b.button('Create action for review',type='primary',disabled=not fresh or plan['status']!='VALID' or not candidate['action'],use_container_width=True):
        task=action('/tasks',{'plan_id':plan['id'],'candidate_id':candidate['id']})
        if task:
            st.session_state.focus_table=task['action']['table']; st.success('Action created. Open Action center to approve and dispatch.')
    with st.expander('Assumptions and unavailable actions'):
        for a in plan['assumptions']: st.write('• '+a)
        if plan.get('signal'): st.write('Observed business signal:',plan['signal'])
        for c in plan['candidates']:
            if not c['feasible']: st.write(c['label']+' — '+str(c['reason']))
    return plan,candidate


def render_actions(data):
    tasks=data['tasks']
    active=[t for t in tasks if t['status'] in ('PENDING','ACCEPTED','EXECUTING','OBSERVING')]
    st.markdown('#### Every recommendation has an owner')
    st.caption('Approve → dispatch → physical confirmation → 5 / 15 minute observation. Additional dealers are checked again before dispatch.')
    chosen_table=st.session_state.get('focus_table')
    only=st.checkbox('Only selected table',value=False,key='filter_table')
    show=st.checkbox('Include completed and expired actions',value=False)
    visible=tasks if show else active
    if only and chosen_table: visible=[t for t in visible if t['action']['table']==chosen_table]
    if not visible: st.info('No actions in this view. Trigger the dining-group scenario or create an action from a plan.')
    for t in visible[:15]:
        with st.container(border=True):
            a,b=st.columns([3,1])
            a.markdown(f"**{t['title']}** · `{t['status']}`")
            a.caption(f"Owner: {t['owner']} · Area: {PITS[t['constraints']['pit']]} · Created {t['created']:.1f} · Approval due {t['expires']:.1f} demo min")
            b.caption('Task '+t['id'][:8])
            if b.button('Locate / inspect',key='locate_'+t['id']):
                st.session_state.pending_table=t['action']['table']; st.session_state.plan_id=t['plan_id']; st.rerun()
            st.write(t['reason'])
            before=t['before']; predicted=t['predicted']
            st.caption(f"At creation: {before['queue']} waiting / {before['wait']:.1f} min · Scenario forecast: {predicted['queue']} waiting / {predicted['wait']:.1f} min")
            if t['status']=='PENDING':
                owner=st.selectbox('Assign owner',['Mei Wong','Alex Chan','Pit Manager'],key='owner_'+t['id'])
                if st.button('Approve & assign',key='accept_'+t['id'],type='primary',disabled=not data['fresh']):
                    if action('/tasks/'+t['id'],{'operation':'accept','owner':owner}): st.rerun()
            elif t['status']=='ACCEPTED':
                if st.button('Dispatch to floor',key='execute_'+t['id'],type='primary',disabled=not data['fresh']):
                    if action('/tasks/'+t['id'],{'operation':'execute'}): st.rerun()
            if t['status'] in ('PENDING','ACCEPTED'):
                with st.expander('Reject / cancel'):
                    reason=st.text_input('Reason',key='reason_'+t['id'])
                    if st.button('Reject action' if t['status']=='PENDING' else 'Cancel action',key='reject_'+t['id']):
                        if action('/tasks/'+t['id'],{'operation':'reject' if t['status']=='PENDING' else 'cancel','reason':reason}): st.rerun()
            for period,observation in t['observations'].items():
                m=observation['metrics']
                st.write(f"**Observed +{period} min:** {m['queue']} waiting · {m['wait']:.1f} min wait · {m['seated']}/{m['capacity']} seated · modeled Theo HK${m['theo_hour']:,.0f}/hr")
                if 'observed_theo' in observation:
                    st.caption(f"Observed wagers generated HK${observation['observed_theo']:,.0f} Theo during this interval · extra labor cost HK${observation['observed_extra_cost']:,.0f}. These are simulation observations, not incremental profit.")
                if not observation['complete']: st.warning('Observation contains a telemetry gap; exclude from effect conclusions.')
            if t['status']=='OBSERVING' and '5' in t['observations']:
                if st.button('Reviewed · close action',key='close_'+t['id']):
                    if action('/tasks/'+t['id'],{'operation':'close'}): st.rerun()
            with st.expander('Decision & execution timeline'):
                st.dataframe(pd.DataFrame(t['timeline'])[['minute','status','reason']],hide_index=True,use_container_width=True)
                st.caption('Observed changes do not establish causal incremental revenue.')


def select_on_map():
    event=st.session_state.get('live_floor', {})
    points=event.get('selection', {}).get('points', [])
    if points:
        st.session_state.pending_table=points[0]['customdata'][0]


@st.fragment(run_every='3s')
def workspace():
    if st.session_state.get('pending_table'):
        tid=st.session_state.pop('pending_table')
        st.session_state.focus_table=tid
        st.session_state.table_inspector=tid
    try: data=api('/overview')
    except ValueError as exc: st.warning(str(exc)); return
    s=data['snapshot']
    if not s: st.info('Waiting for the first floor snapshot through Kafka and RisingWave…'); return
    state_label='LIVE · '+str(data['age'])+'s old' if data['fresh'] else 'STALE · actions paused'
    st.markdown(f'<div class="hero"><div><div class="eyebrow">SMART CASINO / FLOOR OPERATIONS</div><h1>Decide ahead. Act with confidence.</h1><div class="small">{html.escape(s["scenario"])} · demo minute {s["minute"]:.1f} · 10× simulation</div></div><span class="badge">{state_label}</span></div>',unsafe_allow_html=True)
    if not data['fresh']: st.error('Telemetry is stale. Values below are the last observed state; new plans and dispatch are paused. Resume telemetry in the sidebar.')
    if st.session_state.get('flash'): st.caption(st.session_state.pop('flash'))
    m=s['metrics']; cols=st.columns(5)
    for col,label,value in zip(cols,['Occupied seats','Guests waiting','Average queue age','Open positions','Active actions'],[f"{m['seated']} / {m['capacity']}",m['queue'],f"{m['wait']:.1f} min",f"{m['open_tables']} / 36",sum(t['status'] in ('PENDING','ACCEPTED','EXECUTING','OBSERVING') for t in data['tasks'])]): col.metric(label,value)
    if data['notices']:
        n=data['notices'][0]
        st.info(n['title']+' · '+n['detail'])
    tabs=st.tabs(['Floor & scenarios','Action center','Evidence & learning'])
    with tabs[0]:
        plan,candidate=None,None
        if st.session_state.get('plan_id'):
            try:
                plan=api('/plans/'+st.session_state.plan_id)
                candidate=next((v for v in plan['candidates'] if v['id']==st.session_state.get('candidate_id',plan['recommended'])),None)
            except ValueError: pass
        a,b,c=st.columns([1,1,2])
        horizon=a.radio('Map time',[0,15,30],format_func=lambda x:'Now' if x==0 else f'+{x} min',horizontal=True)
        layer=b.selectbox('Map layer',['Occupancy','Queue pressure','Theo'])
        ids=[t['id'] for t in s['tables']]
        focus=st.session_state.get('focus_table','bac_08')
        if focus not in ids: focus=ids[0]
        focused=c.selectbox('Inspect table',ids,index=ids.index(focus),key='table_inspector')
        st.session_state.focus_table=focused
        point=candidate['points'][horizon] if candidate and horizon else None
        if horizon and not candidate: st.warning('Generate a scenario first. Showing current observations.')
        if point and plan['status']!='VALID': st.warning('Historical forecast: '+str(plan['invalid_reason'])+' · Generate a fresh plan below before creating an action.')
        if point: st.caption(f"Saved forecast from demo minute {plan['minute']:.1f} · {candidate['label']} · +{horizon} min. The live state continues independently.")
        selection=st.plotly_chart(map_figure(s,point,layer,focused),use_container_width=True,on_select=select_on_map,selection_mode='points',key='live_floor')
        t=next(t for t in s['tables'] if t['id']==st.session_state.focus_table)
        st.write(f"**{t['id']}** · {t['status']} · {t['occupied']}/{t['capacity']} seats · HK${t['minimum']:g} minimum · dealer {t['dealer'] or 'unassigned'}")
        related=[a for a in data['tasks'] if a['action']['table']==t['id']]
        if related: st.caption('Related actions: '+' · '.join(a['status']+' '+a['id'][:8] for a in related[:4]))
        legend={'Occupancy':'Orange: at least 85% occupied · green: 40–85% · blue: below 40%', 'Queue pressure':'Orange: more than 5 guests waiting in the area · green: 5 or fewer', 'Theo':'Gold: modeled Theo above HK$1,890/hr · blue: below threshold'}[layer]
        st.caption(legend+' · gray: unavailable · gold outline: selected. Queue age is the elapsed wait of guests currently in line.')
        st.divider()
        render_planner(s,data['fresh'])
    with tabs[1]: render_actions(data)
    with tabs[2]:
        st.markdown('#### Streaming evidence')
        st.caption(f"Kafka operational_events → RisingWave mv_ops_latest_snapshot · run {s['run_id'][:8]} · sequence {s['seq']} · event age {data['age']}s")
        if s.get('signal'): st.info(s['signal']['explanation']+' · This relationship is a demo assumption, not a calibrated causal claim.')
        st.markdown('**Relief staff & resources**')
        st.dataframe(pd.DataFrame([d for d in s['dealers'] if d['id'].startswith(('F','X'))]),hide_index=True,use_container_width=True)
        st.markdown('**Recent physical events**')
        st.dataframe(pd.DataFrame(list(reversed(s['events']))[:25]),hide_index=True,use_container_width=True)
        st.markdown('**Forecast error review**')
        if data['reviews']: st.dataframe(pd.DataFrame(data['reviews']),hide_index=True,use_container_width=True)
        else: st.info('Saved forecasts become reviewable after 15 demo minutes (about 90 seconds).')
        st.caption('Baseline comparisons with an intervening action or telemetry gap are excluded. Forecast errors support later model calibration, not automatic policy changes.')
        with st.expander('Plan audit trail'):
            st.dataframe(pd.DataFrame(data['plans']),hide_index=True,use_container_width=True)
            if data['plans']:
                pid=st.selectbox('Restore saved plan',[p['id'] for p in data['plans']])
                if st.button('Open saved plan'):
                    st.session_state.plan_id=pid; st.session_state.pop('candidate_id',None); st.rerun()

workspace()

# The tour lives outside the refreshing workspace so live updates do not reset it.
tour_html=Path(__file__).with_name('tour.html').read_text()
components.html(tour_html.replace('__TOUR_REQUEST__',json.dumps(st.session_state.get('tour_request',''))),height=0)
