#!/usr/bin/env python3
"""
run_all_en_figures.py - Generate all English-version paper figures
Usage:
  cd /mnt/caoyujia/Dilu/paper_figures/scripts
  python3 run_all_en_figures.py
Output: ../output/en_fig{1-7}_*.pdf
"""
import subprocess, sys, os

scripts = [
    ('en_fig1_scheduler_comparison.py',  'Fig1: Scheduler Comparison'),
    ('en_fig2_stable_load_latency.py',   'Fig2: Stable Load Latency & SVR'),
    ('en_fig3_burst_timeline.py',        'Fig3: Burst Traffic Timeline'),
    ('en_fig4_colocation_comparison.py', 'Fig4: Colocation Comparison'),
    ('en_fig5_overflow_scaling.py',      'Fig5: Overflow Auto-scaling'),
    ('en_fig6_ablation.py',              'Fig6: Ablation Study'),
    ('en_fig7_hgss_profile.py',          'Fig7: HGSS-NPU Profiling Results'),
]

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, '..', 'output'), exist_ok=True)

ok, fail = [], []
for script, desc in scripts:
    print(f'\n[{script}] {desc}')
    r = subprocess.run([sys.executable, os.path.join(here, script)],
                       capture_output=True, text=True, cwd=here)
    if r.returncode == 0:
        print(f'  OK  {r.stdout.strip()}')
        ok.append(script)
    else:
        print(f'  FAIL\n{r.stderr[-600:]}')
        fail.append(script)

print(f'\n=== Done {len(ok)}/{len(ok)+len(fail)} figures ===')
for s in ok:   print(f'  [OK]   {s}')
for s in fail: print(f'  [FAIL] {s}')
