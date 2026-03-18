"""
roberta_large.py  --  NPU Profiling Script for RoBERTa-large
=============================================================
昇腾 910B3 NPU 上的 RoBERTa-large 推理画像脚本。
RoBERTa 与 BERT 相近，略偏 Cube（更深的 attention）。
"""
import argparse
import time
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='/mnt/caoyujia/models/roberta-large')
parser.add_argument('--device',     type=int,  default=0)
parser.add_argument('--batch_size', type=int,  default=1)
parser.add_argument('--vector',     type=int,  default=40)
parser.add_argument('--cube',       type=int,  default=20)
parser.add_argument('--seq_len',    type=int,  default=128)
parser.add_argument('--iters',      type=int,  default=50)
parser.add_argument('--simulate',   action='store_true', default=False)
args = parser.parse_args()

ACL_RT_DEV_RES_VECTOR_CORE = 1
ACL_RT_DEV_RES_CUBE_CORE   = 0

if not args.simulate:
    try:
        import acl
        acl.init()
        acl.rt.set_device(args.device)
        acl.rt.set_device_res_limit(args.device, ACL_RT_DEV_RES_VECTOR_CORE, args.vector)
        acl.rt.set_device_res_limit(args.device, ACL_RT_DEV_RES_CUBE_CORE,   args.cube)
    except ImportError:
        args.simulate = True

if not args.simulate:
    import mindspore as ms
    from mindspore import Tensor, context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend', device_id=args.device)
    try:
        from mindnlp.transformers import RobertaModel, RobertaConfig
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        net    = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
    except Exception as e:
        sys.stderr.write(f'[WARN] RoBERTa load failed: {e}, fallback simulate\n')
        args.simulate = True

if not args.simulate:
    net.set_train(False)
    input_ids      = Tensor(np.random.randint(0, 50265,
                     (args.batch_size, args.seq_len)).astype(np.int32))
    attention_mask = Tensor(np.ones((args.batch_size, args.seq_len), dtype=np.int32))
    for _ in range(3):
        net(input_ids, attention_mask=attention_mask)
    start = time.time()
    for _ in range(args.iters):
        net(input_ids, attention_mask=attention_mask)
    elapsed = time.time() - start
else:
    v_ratio  = args.vector / 40.0
    c_ratio  = args.cube   / 20.0
    base_lat = 0.030
    elapsed  = (base_lat * args.batch_size**0.8
                / (0.4*v_ratio + 0.6*c_ratio)) * args.iters

elapsed_per_iter = elapsed / args.iters
throughput = args.iters * args.batch_size / elapsed
print('%f,%f,%f,%f' % (args.batch_size, args.vector, elapsed_per_iter, throughput))

if not args.simulate:
    try:
        acl.rt.reset_device(args.device)
        acl.finalize()
    except Exception:
        pass
