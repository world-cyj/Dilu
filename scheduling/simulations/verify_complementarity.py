"""verify_complementarity.py - 验证互补调度的真实效果"""
import ast, sys, importlib
sys.path.insert(0, '.')

WL = 'workload/instances-npu-200.txt'
with open(WL) as f:
    events = [ast.literal_eval(l.strip()) for l in f]

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

    peak_cards   = 0
    peak_density = 0.0   # max instances per active card
    colocation_pairs = 0 # vector-heavy + cube-heavy on same card

    for ev in events:
        if ev['Action'] == 'start':
            mod.schedule_instance(ev['Instance'])
        else:
            mod.delete_instance(ev['Instance'])

        n_active = len(mod.active_npus)
        if n_active > peak_cards:
            peak_cards = n_active

    # Final snapshot: measure per-card density and complementarity
    total_instances = sum(len(npu.instances) for npu in mod.active_npus)
    n_active = len(mod.active_npus)
    avg_density = total_instances / max(n_active, 1)

    # Count cards that have both vector-heavy and cube-heavy tasks
    for npu in mod.active_npus:
        inst_list = list(npu.instances.values())
        has_v_heavy = any(i.vector_req > i.cube_req * 2 for i in inst_list)
        has_c_heavy = any(i.cube_req > i.vector_req     for i in inst_list)
        if has_v_heavy and has_c_heavy:
            colocation_pairs += 1

    vf, cf, mf = mod.calc_fragmentation()
    # Complementarity score: how well V and C are both utilized per card
    comp_score = 0.0
    for npu in mod.active_npus:
        v_util = npu.current_vector_req / npu.total_vector
        c_util = npu.current_cube_req   / npu.total_cube
        # ideal: both utilizations high -> score high
        comp_score += min(v_util, c_util) / max(max(v_util, c_util), 0.001)
    comp_score /= max(n_active, 1)

    results[label] = dict(
        peak_cards=peak_cards, final_cards=n_active,
        total_inst=total_instances, avg_density=avg_density,
        colocation=colocation_pairs, comp_score=comp_score,
        vf=vf, cf=cf, mf=mf)

print('\n=== 互补调度效果验证 (200实例) ===')
print(f'{"Scheduler":<12} {"Peak":>6} {"Final":>6} {"Inst":>6} '
      f'{"Density":>8} {"ColocPairs":>10} {"CompScore":>10}')
print('-'*68)
for lbl, r in results.items():
    print(f'{lbl:<12} {r["peak_cards"]:6d} {r["final_cards"]:6d} '
          f'{r["total_inst"]:6d} {r["avg_density"]:8.2f} '
          f'{r["colocation"]:10d} {r["comp_score"]:10.4f}')

print()
base = results.get('K8s', {})
dilu = results.get('Dilu-NPU', {})
if base and dilu:
    print('Dilu-NPU vs K8s:')
    print(f'  共置互补对数: {base["colocation"]} -> {dilu["colocation"]} '
          f'(+{dilu["colocation"]-base["colocation"]})')
    print(f'  平均每卡密度: {base["avg_density"]:.2f} -> {dilu["avg_density"]:.2f}')
    print(f'  互补利用率分: {base["comp_score"]:.4f} -> {dilu["comp_score"]:.4f}')
