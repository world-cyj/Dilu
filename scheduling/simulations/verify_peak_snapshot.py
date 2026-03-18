"""verify_peak_snapshot.py - 在峰值时刻验证互补调度效果"""
import ast, sys, importlib
sys.path.insert(0, '.')

WL = 'workload/instances-npu-200.txt'
with open(WL) as f:
    events = [ast.literal_eval(l.strip()) for l in f]


def analyze_snapshot(mod):
    """分析当前集群状态的互补性指标"""
    if not mod.active_npus:
        return 0, 0.0, 0.0
    coloc = 0
    comp_score = 0.0
    for npu in mod.active_npus:
        inst_list = list(npu.instances.values())
        if len(inst_list) < 2:
            continue
        has_v = any(i.vector_req > i.cube_req * 1.5 for i in inst_list)
        has_c = any(i.cube_req > i.vector_req * 0.8 for i in inst_list)
        if has_v and has_c:
            coloc += 1
        v_util = npu.current_vector_req / npu.total_vector
        c_util = npu.current_cube_req   / npu.total_cube
        if max(v_util, c_util) > 0:
            comp_score += min(v_util, c_util) / max(v_util, c_util)
    comp_score /= len(mod.active_npus)
    total_inst = sum(len(npu.instances) for npu in mod.active_npus)
    density = total_inst / len(mod.active_npus)
    return coloc, density, comp_score


results = {}
for label, mod_name in [
    ('K8s',      'baseline.scheduler_k8s_npu'),
    ('INFless-R','baseline.scheduler_infless_r_npu'),
    ('Dilu-NPU', 'baseline.scheduler_dilu_npu'),
]:
    for k in list(sys.modules):
        if mod_name.split('.')[-1] in k:
            del sys.modules[k]
    mod = importlib.import_module(mod_name)
    from baseline.scheduler_dilu_npu import PortManager
    mod.port_manager = PortManager()
    mod.new_npus     = [mod.NPU(i, n['ip'], n['index'])
                        for i, n in enumerate(mod.nodes_info)]
    mod.active_npus  = []

    peak_cards = 0
    best_coloc = 0
    best_density = 0.0
    best_comp = 0.0

    # Only process start events to build peak state
    for ev in events:
        if ev['Action'] != 'start':
            continue
        mod.schedule_instance(ev['Instance'])
        n = len(mod.active_npus)
        if n > peak_cards:
            peak_cards = n
            coloc, density, comp = analyze_snapshot(mod)
            best_coloc   = coloc
            best_density = density
            best_comp    = comp

    total_inst = sum(len(npu.instances) for npu in mod.active_npus)
    final_coloc, final_density, final_comp = analyze_snapshot(mod)
    vf, cf, mf = mod.calc_fragmentation()

    results[label] = dict(
        peak_cards=peak_cards,
        total_inst=total_inst,
        final_cards=len(mod.active_npus),
        coloc=final_coloc,
        density=final_density,
        comp=final_comp,
        vf=vf, cf=cf, mf=mf)

print('\n=== 峰值状态互补调度效果 (只计start事件, 200实例) ===')
print(f'{"Scheduler":<12} {"PeakCards":>10} {"FinalCards":>10} '
      f'{"TotalInst":>10} {"Density":>8} {"ColocCards":>10} {"CompScore":>10}')
print('-'*78)
for lbl, r in results.items():
    print(f'{lbl:<12} {r["peak_cards"]:10d} {r["final_cards"]:10d} '
          f'{r["total_inst"]:10d} {r["density"]:8.2f} '
          f'{r["coloc"]:10d} {r["comp"]:10.4f}')

print()
print(f'{"Scheduler":<12} {"VFrag":>8} {"CFrag":>8} {"MFrag":>8} {"AvgFrag":>8}')
print('-'*50)
for lbl, r in results.items():
    avg = (r['vf']+r['cf']+r['mf'])/3
    print(f'{lbl:<12} {r["vf"]:8.3f} {r["cf"]:8.3f} {r["mf"]:8.3f} {avg:8.3f}')

print()
base = results.get('K8s', {})
dilu = results.get('Dilu-NPU', {})
if base and dilu:
    print('Dilu-NPU vs K8s (peak state):')
    dc = dilu['coloc'] - base['coloc']
    dd = dilu['density'] - base['density']
    dcs= dilu['comp'] - base['comp']
    dvf= base['vf'] - dilu['vf']
    dcf= base['cf'] - dilu['cf']
    print(f'  共置互补卡数改善: +{dc}')
    print(f'  平均每卡密度改善: {dd:+.3f}')
    print(f'  互补利用率改善:   {dcs:+.4f}')
    print(f'  Vector碎片改善:   {dvf:+.3f}')
    print(f'  Cube碎片改善:     {dcf:+.3f}')
