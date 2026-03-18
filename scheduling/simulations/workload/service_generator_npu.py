"""
NPU-adapted Service Generator for Dilu on Ascend 910B3.

Replaces the single-dimension sm_req/sm_lim with a 2-D resource model:
  - vector_req / vector_lim  (max 40 Vector Cores per card)
  - cube_req  / cube_lim     (max 20 Cube  Cores per card)
  - memory                   (max 64 GB per card)

Model-type profiles are grounded in the Ascend 910B3 characteristics:
  CV (ResNet152, VGG19)        : Vector-heavy  (matrix-free)
  YOLO / detection             : Cube-heavy    (conv-intensive)
  NLP (BERT, RoBERTa)          : balanced Cube/Vector
  LLM (GPT2-large, LLaMA2-7B) : Cube-heavy + large memory

Physical constants (single NPU card):
  TOTAL_VECTOR = 40
  TOTAL_CUBE   = 20
  TOTAL_MEMORY = 64  GB
"""

import random
import string
import numpy as np
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Physical NPU limits ───────────────────────────────────────────────────────
TOTAL_VECTOR = 40    # Vector Cores per card
TOTAL_CUBE   = 20    # Cube  Cores per card
TOTAL_MEMORY = 64    # GB per card
NUM_CARDS    = 4     # cards in the container environment

# ── Model-type resource profiles ─────────────────────────────────────────────
# Each profile: (vector_req_frac, cube_req_frac, memory_gb, label)
# fractions are relative to TOTAL_VECTOR / TOTAL_CUBE
MODEL_PROFILES = {
    # CV: heavy vector (element-wise activations), light cube
    'resnet':   dict(v_lo=0.25, v_hi=0.45, c_lo=0.05, c_hi=0.15, mem_lo=3,  mem_hi=6),
    'vgg':      dict(v_lo=0.28, v_hi=0.48, c_lo=0.04, c_hi=0.12, mem_lo=3,  mem_hi=6),
    # Object-detection: heavier cube (large conv kernels)
    'yolo':     dict(v_lo=0.10, v_hi=0.25, c_lo=0.20, c_hi=0.40, mem_lo=3,  mem_hi=6),
    # NLP balanced
    'bert':     dict(v_lo=0.15, v_hi=0.30, c_lo=0.15, c_hi=0.30, mem_lo=3,  mem_hi=6),
    'roberta':  dict(v_lo=0.15, v_hi=0.30, c_lo=0.15, c_hi=0.30, mem_lo=3,  mem_hi=6),
    # LLM: cube-heavy + large memory
    'gpt2':     dict(v_lo=0.18, v_hi=0.32, c_lo=0.22, c_hi=0.40, mem_lo=10, mem_hi=16),
    'llama2':   dict(v_lo=0.20, v_hi=0.35, c_lo=0.25, c_hi=0.45, mem_lo=16, mem_hi=24),
}

# Mapping: task-type → candidate model profiles
TYPE_PROFILES = {
    'inference':     ['resnet', 'vgg', 'yolo', 'bert', 'roberta'],
    'llm-inference': ['gpt2', 'llama2'],
    'training':      ['bert', 'roberta', 'resnet', 'vgg'],   # same archs, higher resource
}

# Training tasks use more cores (multi-epoch, larger batch)
TRAINING_SCALE = 1.3   # scale factor on req fractions for training


def generate_service_name(prefix):
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}-{suffix}"


def _clamp_int(val, lo, hi):
    return max(lo, min(hi, int(round(val))))


def _sample_req_lim(lo_frac, hi_frac, total, scale=1.0, lim_headroom=0.08):
    """Sample integer request and limit cores within [lo_frac*total, hi_frac*total]."""
    req_frac = random.uniform(lo_frac, hi_frac) * scale
    req = _clamp_int(req_frac * total, 1, total)
    # limit = req + small headroom, capped at total
    lim = _clamp_int(req + lim_headroom * total, req, total)
    return req, lim


def _make_instance(task_type, profile_key=None):
    """Return a single instance dict with NPU 2-D resource fields."""
    if profile_key is None:
        profile_key = random.choice(TYPE_PROFILES[task_type])

    p = MODEL_PROFILES[profile_key]
    scale = TRAINING_SCALE if task_type == 'training' else 1.0

    vector_req, vector_lim = _sample_req_lim(p['v_lo'], p['v_hi'], TOTAL_VECTOR, scale)
    cube_req,   cube_lim   = _sample_req_lim(p['c_lo'], p['c_hi'], TOTAL_CUBE,   scale)
    memory = random.randint(p['mem_lo'], p['mem_hi'])

    if task_type == 'training':
        gpu_num = random.choice([2, 4])
        memory_list = [memory] * gpu_num
    else:
        gpu_num = 1
        memory_list = [memory]

    return {
        'service_name':  generate_service_name(task_type.replace('-', '_')),
        'type':          task_type,
        'model':         profile_key,
        'gpu_num':       gpu_num,
        'vector_req':    vector_req,
        'vector_lim':    vector_lim,
        'cube_req':      cube_req,
        'cube_lim':      cube_lim,
        # Legacy sm_* fields kept for baseline-scheduler compatibility
        'sm_requests':   round(vector_req / TOTAL_VECTOR, 4),
        'sm_limits':     round(vector_lim / TOTAL_VECTOR, 4),
        'memory':        memory_list,
    }


def generate_instances(total_instances, train_ratio=2):
    """
    Generate a mixed workload.

    Parameters
    ----------
    total_instances : int
        Total number of service instances to generate.
    train_ratio : int
        Rough fraction denominator for training tasks
        (train_count = total * train_ratio / 10).
    """
    llm_count   = int(total_instances * 0.20)
    train_count = int(total_instances * train_ratio / 10)
    inf_count   = total_instances - train_count - llm_count

    instances = []

    # ── Training instances ────────────────────────────────────────────────────
    for _ in range(train_count):
        instances.append(_make_instance('training'))

    # ── Standard inference instances ─────────────────────────────────────────
    # Ensure a realistic mix of CV and NLP models
    cv_keys  = ['resnet', 'vgg', 'yolo']
    nlp_keys = ['bert', 'roberta']
    for i in range(inf_count):
        # 55 % CV, 45 % NLP  (matches Dilu paper workload composition)
        profile = random.choice(cv_keys if i % 20 < 11 else nlp_keys)
        instances.append(_make_instance('inference', profile))

    # ── LLM inference instances ───────────────────────────────────────────────
    for _ in range(llm_count):
        instances.append(_make_instance('llm-inference'))

    return instances


def shuffle_and_delete(instances, total_instances):
    """Attach start/delete events with timestamps (identical logic to original)."""
    random.shuffle(instances)
    start_time = datetime.now()
    events = []

    for i, instance in enumerate(instances):
        events.append({
            'time':     start_time + timedelta(seconds=i),
            'action':   'start',
            'instance': instance,
        })

    # Delete ~60 % of inference / llm-inference
    deletable = [e for e in events if e['instance']['type'] in ['inference', 'llm-inference']]
    num_deletes = int(total_instances * 0.60)
    for event in random.sample(deletable, min(num_deletes, len(deletable))):
        events.append({
            'time':     event['time'] + timedelta(minutes=random.randint(1, 3)),
            'action':   'delete',
            'instance': event['instance'],
        })

    # Delete ~10 % of training
    deletable_train = [e for e in events if e['instance']['type'] == 'training']
    num_train_del = int(total_instances * 0.10)
    for event in random.sample(deletable_train, min(num_train_del, len(deletable_train))):
        events.append({
            'time':     event['time'] + timedelta(minutes=random.randint(10, 15)),
            'action':   'delete',
            'instance': event['instance'],
        })

    events.sort(key=lambda x: x['time'])
    return events


def generate_and_save(total_instances, train_ratio=2, output_path=None):
    """Generate instances, create events and write to file (or print)."""
    if output_path is None:
        output_path = f"instances-npu-{total_instances}.txt"

    instances = generate_instances(total_instances, train_ratio)
    events    = shuffle_and_delete(instances, total_instances)

    with open(output_path, 'w') as f:
        for event in events:
            line = {
                'Time':     event['time'].strftime('%Y-%m-%d %H:%M:%S'),
                'Action':   event['action'],
                'Instance': event['instance'],
            }
            f.write(str(line) + '\n')

    print(f"[NPU Generator] Written {len(events)} events → {output_path}")
    return events


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser(description='NPU workload generator for Dilu')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[100, 200, 400, 800, 1600, 3200],
                        help='Total instance counts to generate')
    parser.add_argument('--train_ratio', type=int, default=2,
                        help='Training ratio denominator (train = total * ratio / 10)')
    parser.add_argument('--outdir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory')
    args = parser.parse_args()

    for n in args.sizes:
        path = os.path.join(args.outdir, f'instances-npu-{n}.txt')
        generate_and_save(n, train_ratio=args.train_ratio, output_path=path)

    print('\n[NPU Generator] Summary of resource model:')
    print(f'  Vector Cores : 0 – {TOTAL_VECTOR} per card')
    print(f'  Cube   Cores : 0 – {TOTAL_CUBE}  per card')
    print(f'  Memory       : up to {TOTAL_MEMORY} GB per card')
    print(f'  Cards in env : {NUM_CARDS}')
    print('\n[NPU Generator] Model profiles:')
    for k, v in MODEL_PROFILES.items():
        print(f'  {k:10s}: Vector [{v["v_lo"]*TOTAL_VECTOR:.0f}–{v["v_hi"]*TOTAL_VECTOR:.0f}]'
              f'  Cube [{v["c_lo"]*TOTAL_CUBE:.0f}–{v["c_hi"]*TOTAL_CUBE:.0f}]'
              f'  Mem [{v["mem_lo"]}–{v["mem_hi"]} GB]')
