"""Independent 12-check simulator audit; writes only this audit's JSON report."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from paper2_model.tests.test_return_simulation import simulation
from paper2_model.tests.test_channel import manual_snapshot
from paper2_model.return_simulation import DataUnit
from paper2_model.channel import ChannelConfig, evaluate_schedule
from paper1.paper2_dataset import Paper2Dataset, _read_events

results = []
def check(name, fn):
    try:
        detail = fn()
        results.append(dict(name=name, passed=True, detail=detail))
    except Exception as exc:
        results.append(dict(name=name, passed=False, detail=f'{type(exc).__name__}: {exc}'))

def relay_chain():
    s, m = simulation(3, 4, [DataUnit(0, 0, 1., 0, .25)])
    r = s.run({1:[(0,1,0)],2:[(1,2,0)],3:[(2,3,0)]})
    assert r['delays_s'][0] == 3.75 and r['base_received_bits'][0] == 1
    assert r['positive_tx_slot_count'][0] == 3
    return 'Three hops: completion=4 s, delay=3.75 s; one original bit delivered.'

def reject_halfduplex():
    c = ChannelConfig()
    for links in [[(0,1,0),(1,4,1)], [(0,2,0),(1,2,1)]]:
        try: evaluate_schedule(manual_snapshot(c), links)
        except ValueError: continue
        raise AssertionError('illegal relay schedule accepted')

def interference():
    c=ChannelConfig(transmit_power_w=1,total_bandwidth_hz=2,num_subchannels=2,noise_psd_w_per_hz=1)
    snap=manual_snapshot(c)
    r=evaluate_schedule(snap,[(0,4,0),(1,2,0)])['links']
    np.testing.assert_allclose([x['sinr'] for x in r],[1,2])
    np.testing.assert_allclose([x['rate_bps'] for x in r],[1,np.log2(3)])
    return 'Exact SINRs: 2/(1+1)=1; 3/(1+0.5)=2.'

def relay_exit():
    s,m=simulation(2,4,[DataUnit(0,0,2.,0)],eligible=[[1,1],[1,1],[1,0],[1,0]],returns=[4,2])
    r=s.run({1:[(0,1,0)]})
    np.testing.assert_allclose(r['final_queues_bits'][:,0],[1,1])
    assert r['residual_data_bits']==2 and np.isnan(r['delays_s'][0])
    assert next(x for x in r['exit_stranded'] if x['uav_id']==1)['stranded_bits']==1

def overflow():
    s,m=simulation(2,3,[DataUnit(0,0,1.,0),DataUnit(1,1,1.,0)],buffer_bits=1)
    r=s.run({1:[(0,1,0)]})
    assert r['buffer_constraint_violated'] and not r['performance_metrics_valid']
    assert r['final_queues_bits'][1].sum()==2 and r['conservation_passed']
    return 'Overflow is retained and flagged INVALID, not silently dropped or prevented.'

def merged_fragments():
    s,m=simulation(3,5,[DataUnit(0,0,3.,0)])
    r=s.run({1:[(0,1,0)],2:[(0,2,0),(1,3,1)],3:[(0,3,0)],4:[(2,3,0)]})
    assert r['base_received_bits'][0]==3 and r['delivery_time_s'][0]==5

def zero_service():
    s,m=simulation(1,2,[DataUnit(0,0,1.,0)])
    r=s.run({})
    assert r['complete_data_unit_count']==0 and r['all_data_mean_delay_s'] is None
    assert np.isnan(r['first_tx_time_s'][0]) and r['residual_data_bits']==1

def last_arrival():
    s,m=simulation(1,2,[DataUnit(0,0,1.,1,1.9)])
    r=s.run({1:[(0,1,0)]})
    assert r['residual_data_bits']==1 and not r['all_data_complete']

def fractional_id_rejected():
    s,m=simulation(1,2,[DataUnit(0,0,1.,0)])
    try: s.run({1:[(0,1,0)]},lambda t,*args: [(0,1,.5,1)] if t==1 else [])
    except ValueError: return
    raise AssertionError('allocation data_id=0.5 silently converted to 0')

def bad_timestamp_rejected():
    try:
        s,m=simulation(1,2,[DataUnit(0,0,1.,0,10.)])
        r=s.run({1:[(0,1,0)]})
    except ValueError: return
    raise AssertionError(f'inconsistent timestamp accepted; delay={r["delays_s"][0]} s')

def bad_size_rejected():
    try:
        s,m=simulation(1,2,[DataUnit(0,0,-1.,0)])
        s.run({})
    except ValueError: return
    raise AssertionError('negative-size DataUnit not rejected at simulator boundary')

def real_input():
    folder=ROOT/'paper2_baselines'/'direct_return'/'input'/'input_ref_5to10mbit_v10'
    d=Paper2Dataset(folder); ev=_read_events(folder/'collection_events.csv')
    dt=d.scenario['slot_seconds']; eligible=d.node_states['scheduling_eligible']
    assert len(ev)==8000 and len({x['data_id'] for x in ev})==8000
    for e in ev:
        t=e['collection_complete_time_s']; u=e['source_uav_id']; path=d.paths[u]
        assert path[0,3]<=t<path[-1,3]
        assert int(np.floor(t/dt))==e['enqueue_slot_zero_based']
        assert e['available_from_slot_zero_based']==e['enqueue_slot_zero_based']+1
        assert 625000<=e['data_bytes']<=1250000 and e['data_bits']==8*e['data_bytes']
    for u,path in enumerate(d.paths):
        slots=np.arange(len(eligible))
        expected=(slots*dt>=path[0,3]-1e-9)&((slots+1)*dt<=path[-1,3]+1e-9)
        np.testing.assert_array_equal(eligible[:,u],expected)
        durations=np.diff(path[:,3]); lengths=np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1)
        assert np.all(lengths<=10*durations+1e-7)
    return {'events':len(ev),'bits':sum(e['data_bits'] for e in ev),'slots':len(eligible)}

for fn in [relay_chain,reject_halfduplex,interference,relay_exit,overflow,merged_fragments,
           zero_service,last_arrival,real_input,fractional_id_rejected,bad_timestamp_rejected,bad_size_rejected]:
    check(fn.__name__,fn)
report={'passed':sum(x['passed'] for x in results),'total':len(results),'checks':results}
Path(__file__).with_name('audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
