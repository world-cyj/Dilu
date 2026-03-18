# Dilu-NPU: Experiments Guide

Step-by-step verification on Ascend 910B3 (4-card container).

---

## Environment

```
Container : 4x Ascend 910B3 NPU, 64 GB each
Base path : /mnt/caoyujia/Dilu/
Python    : 3.8+  (numpy, matplotlib)
ACL       : CANN pyACL (optional; simulate mode works without it)
```

```bash
pip install numpy matplotlib
```

---

## Step 1 — Generate NPU Workload

```bash
cd /mnt/caoyujia/Dilu/scheduling/simulations/workload
python service_generator_npu.py --sizes 100 200 400 800 1600 3200
head -3 instances-npu-100.txt
```

Expected fields: `vector_req`, `vector_lim`, `cube_req`, `cube_lim`.
ResNet: high `vector_req` (>=20). YOLO: high `cube_req` (>=8).

---

## Step 2 — Baseline Comparison

```bash
cd /mnt/caoyujia/Dilu/scheduling/simulations
mkdir -p logs
python comp_baselines_npu.py --workload workload/instances-npu-3200.txt
# Quick test:
python comp_baselines_npu.py --workload workload/instances-npu-100.txt
```

Outputs: `logs/npu-baselines-timeline.png`, `logs/npu-baselines-summary.txt`

Expect: Dilu-NPU peak card count < K8s (Exclusive) peak.

---

## Step 3 — Individual Schedulers

```bash
cd /mnt/caoyujia/Dilu/scheduling/simulations
python baseline/scheduler_dilu_npu.py workload/instances-npu-3200.txt
python baseline/scheduler_k8s_npu.py  workload/instances-npu-100.txt
python baseline/scheduler_infless_l_npu.py workload/instances-npu-3200.txt
python baseline/scheduler_infless_r_npu.py workload/instances-npu-3200.txt
```

---

## Step 4 — HGSS Profiling (Simulate)

```bash
cd /mnt/caoyujia/Dilu/profiling/inference/methods
python HGSS_NPU.py --model resnet152  --qos 0.04 --simulate
python HGSS_NPU.py --model gpt2_large --qos 0.06 --simulate
python HGSS_NPU.py --model bert_base  --qos 0.05 --simulate
```

Results: `profiling/inference/observations/profiling_results_npu/<model>_hgss_npu.csv`

Expect: `vector_req` and `cube_req` columns. ResNet: vector_req~16-24. GPT2: higher cube_req.

---

## Step 5 — Complementarity Verification

```bash
cd /mnt/caoyujia/Dilu/scheduling/simulations
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from baseline import scheduler_dilu_npu as s
from baseline.scheduler_dilu_npu import PortManager
s.port_manager = PortManager()
ja = {'service_name':'resnet-a','type':'inference','gpu_num':1,
      'vector_req':28,'vector_lim':34,'cube_req':4,'cube_lim':6,'memory':[5]}
jb = {'service_name':'yolo-b','type':'inference','gpu_num':1,
      'vector_req':12,'vector_lim':18,'cube_req':14,'cube_lim':18,'memory':[4]}
r1 = s.schedule_instance(ja)
r2 = s.schedule_instance(jb)
print('ResNet card:', r1['selected_npus'][0]['index'])
print('YOLO   card:', r2['selected_npus'][0]['index'])
assert r1['selected_npus'][0]['index'] == r2['selected_npus'][0]['index']
print('PASS: complementary jobs co-located on same card')
EOF
```

---

## Step 6 — Day/Night Quota Controller

```bash
cd /mnt/caoyujia/Dilu/scheduling
# Night mode: cards 2,3 idle -> consolidate onto card 0
python npu_quota_controller.py --simulate --night
# Day mode: restore all cards to full quota
python npu_quota_controller.py --simulate
```

Expected NIGHT log lines:
```
acl.rt.set_device_res_limit(device=0, type=VECTOR, value=40)
acl.rt.set_device_res_limit(device=2, type=VECTOR, value=1)
acl.rt.set_device_res_limit(device=3, type=VECTOR, value=1)
```

---

## File Map

```
scheduling/simulations/workload/
    service_generator_npu.py          workload generator (Step 1)
    instances-npu-{100..3200}.txt     generated workloads

scheduling/simulations/
    comp_baselines_npu.py             comparison runner  (Step 2)
    baseline/
        scheduler_dilu_npu.py         Dilu 2-D NPU scheduler
        scheduler_k8s_npu.py          K8s exclusive baseline
        scheduler_infless_l_npu.py    INFless-L baseline
        scheduler_infless_r_npu.py    INFless-R baseline

profiling/inference/methods/
    HGSS_NPU.py                       HGSS profiler      (Step 4)

scheduling/
    npu_quota_controller.py           day/night controller (Step 6)
```

---

## Design Mapping: GPU -> NPU

| Aspect | Dilu GPU | Dilu-NPU |
|---|---|---|
| Resource dims | 1-D SM% | 2-D Vector(0-40)+Cube(0-20) |
| Core limiting | CUDA MPS thread% | `acl.rt.set_device_res_limit` |
| Score formula | a*(1-SM)+b*(1-Mem) | a*(1-V)+b*(1-C)+g*(1-Mem) |
| Profiling | HGSS.sh (bash) | HGSS_NPU.py (Python) |
| Elasticity | RCKM ms-level | NPUQuotaController min-level |
