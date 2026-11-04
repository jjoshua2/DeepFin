"""UNRUN pure mechanism controls; no concrete operational classification/admission."""
from types import FunctionType, SimpleNamespace


def run_source_controls(original,candidate):
    receipts=[]
    def fixture():
        stops=['/synthetic/eval%d/STOP'%i for i in range(6)]
        pause='/synthetic/ordinary/PAUSE'
        return {'preserved':[{'path':p,'present':True,'sha256':'a'*64} for p in stops+[pause]]+[{'path':'/synthetic/history','present':False,'sha256':None},{'path':'/synthetic/runtime','present':False,'sha256':None}], 'ordinary_pause':pause,'evaluation_stops':stops,'abort_paths':['/synthetic/private/'+p for p in ('STOP','PAUSE','CANCEL','YIELD')]}
    def setup():
        events=[]
        state={'present':set(),'bad_hash':set(),'now':10.,'retained_error':None}
        class FakeReads:
            def __init__(self,cap):
                events.append(('Reads',cap))
            def checked(self,ref,*,metadata=False):
                events.append(('checked',dict(ref),metadata))
                if ref['path'] in state['bad_hash']:
                    raise ValueError('synthetic integrity failure')
                return b'synthetic pinned evidence'
        def lexists(path):
            events.append(('lexists',str(path)))
            return str(path) in state['present']
        ns=dict(original)
        ns.update(Reads=FakeReads,os=SimpleNamespace(path=SimpleNamespace(lexists=lexists)),time=SimpleNamespace(monotonic=lambda:state['now']))
        fn=original['controls_unchanged']
        ns['controls_unchanged']=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        controls=fixture()
        class FakeRetained:
            output_root=original['Path']('/synthetic/output')
            limits={'deadline_unix':1000}
            stop_paths=tuple(original['Path'](p) for p in controls['abort_paths'])
            def check(self):
                events.append(('retained.check',))
                if state['retained_error']:
                    raise state['retained_error']
            def charge(self,n):
                events.append(('retained.charge',n))
        Guard=candidate['make_guard_type'](ns)
        return controls,Guard,FakeRetained(),state,events
    def expect_error(operation):
        try:
            operation()
        except (ValueError,RuntimeError,KeyError,TypeError) as error:
            return type(error).__name__
        raise AssertionError('expected fail-closed refusal')
    def mark(name,details=None):
        receipts.append({'id':name,'pass':True,'details':details})
    def partition(c,excluded=()):
        return {'runtime_paths':[r['path'] for r in c['preserved'] if r['path'] not in excluded], 'preservation_only':[{'path':p,'reason':'synthetic explicitly reviewed unrelated historical control','source_ref':{'path':'/synthetic/proof/source','sha256':'b'*64},'review_ref':{'path':'/synthetic/proof/review','sha256':'c'*64}} for p in excluded]}

    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c)
    assert g.partition_observation['runtime_count'] == 9 and g.partition_observation['preservation_only_count'] == 0
    e.clear();g.check();g.check()
    assert sum(x == ('Reads',2 << 20) for x in e) == 2 and sum(x == ('retained.check',) for x in e) == 2
    assert g.charge.__func__ is original['ParentDeadlineGuard'].charge
    mark('unclassified_defaults_runtime_actual_charge_inherited')
    for mode in ('omit','unknown','overlap','duplicate'):
        c,G,r,s,e=setup();p=partition(c)
        if mode == 'omit':p['runtime_paths'].pop()
        if mode == 'unknown':p['runtime_paths'].append('/unknown')
        if mode == 'overlap':p=partition(c,('/synthetic/history',));p['runtime_paths'].append('/synthetic/history')
        if mode == 'duplicate':p['runtime_paths'].append(p['runtime_paths'][0])
        expect_error(lambda:G(r,{'active_deadline_monotonic':100},c,partition=p))
    mark('partition_complete_disjoint_known_unique_refusals')
    for key in ('reason','source_ref','review_ref'):
        c,G,r,s,e=setup();p=partition(c,('/synthetic/history',));del p['preservation_only'][0][key]
        expect_error(lambda:G(r,{'active_deadline_monotonic':100},c,partition=p))
    mark('each_exemption_requires_reason_source_and_review_refs')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);s['present'].add('/synthetic/runtime');expect_error(g.check)
    mark('runtime_absent_marker_creation_refuses')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);s['bad_hash'].add(c['ordinary_pause']);expect_error(g.check)
    mark('runtime_present_integrity_failure_refuses')
    for suffix in ('STOP','PAUSE','CANCEL','YIELD'):
        c,G,r,s,e=setup();path='/synthetic/current/'+suffix;g=G(r,{'active_deadline_monotonic':100},c,active_paths=[path]);s['present'].add(path);expect_error(g.check)
    mark('declared_active_stop_pause_cancel_yield_each_refuses')
    for paths in (['relative'],['/same','/same'],[None]):
        c,G,r,s,e=setup();expect_error(lambda:G(r,{'active_deadline_monotonic':100},c,active_paths=paths))
    mark('active_paths_relative_duplicate_nonstr_refusals')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c,partition=partition(c,('/synthetic/history',)));s['present'].add('/synthetic/history');g.check();expect_error(g.verify_preservation_at_closure)
    assert g.partition_observation['legacy_equivalence'] is False
    mark('declared_preservation_only_mutation_detected_at_closure_not_runtime')
    c,G,r,s,e=setup();claim={'active_deadline_monotonic':100};g=G(r,claim,c);c['preserved'].clear();claim['active_deadline_monotonic']=1000;e.clear();g.check();assert any(x[0] == 'checked' for x in e);s['now']=100;expect_error(g.check)
    mark('new_contract_freezes_schema_paths_deadline_no_alias_extension')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);s['now']=100;e.clear();expect_error(g.check);assert not e
    mark('deadline_refuses_before_dynamic_io')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);s['retained_error']=RuntimeError('synthetic custody loss');expect_error(g.check)
    mark('retained_custody_data_guard_error_propagates')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);r.stop_paths=();expect_error(g.check)
    mark('retained_active_path_binding_change_refuses')
    c,G,r,s,e=setup();g=G(r,{'active_deadline_monotonic':100},c);e.clear();g.charge(3);flow=[x for x in e if x[0].startswith('retained.')];assert flow == [('retained.check',),('retained.charge',3),('retained.check',)]
    g.verify_preservation_at_closure()
    mark('actual_inherited_charge_check_charge_check_order')
    return {'status':'PASS_GENERIC_PARTITION_MECHANISM_ONLY','count':len(receipts),'controls':receipts,'concrete_operational_classification':False,'actual_filesystem_or_custody_executed':False,'legacy_equivalence':False,'gpu_or_live_admission':False}
