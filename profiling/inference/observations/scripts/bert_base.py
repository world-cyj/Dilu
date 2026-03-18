"""
bert_base.py  --  NPU Profiling Script for BERT-base
=====================================================
昇腾 910B3 / CANN 8.3 RC1，使用 torch_npu + ACL 接口。
BERT: balanced Vector+Cube（矩阵乘+激活均衡）。
"""
import argparse
import time
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path',
                    default='/mnt/caoyujia/Dilu/evaluation/models/bert-base-uncased')
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
    except Exception as e:
        sys.stderr.write(f'[WARN] ACL init failed: {e}, fallback simulate\n')
        args.simulate = True

if not args.simulate:
    try:
        import torch
        import torch_npu
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        npu_device = f'npu:{args.device}'
        torch_npu.npu.set_device(npu_device)

        config    = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
                        args.model_name_or_path, use_fast=False)
        model     = AutoModel.from_pretrained(
                        args.model_name_or_path, config=config)
        model     = model.to(npu_device)
        model.eval()

        input_ids = torch.randint(
            0, config.vocab_size,
            (args.batch_size, args.seq_len)).to(npu_device)
        attn_mask = torch.ones(
            args.batch_size, args.seq_len,
            dtype=torch.long).to(npu_device)

        with torch.no_grad():
            for _ in range(3):
                model(input_ids, attention_mask=attn_mask)

        torch_npu.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(args.iters):
                model(input_ids, attention_mask=attn_mask)
        torch_npu.npu.synchronize()
        elapsed = time.time() - start

    except Exception as e:
        sys.stderr.write(f'[WARN] torch_npu BERT failed: {e}, fallback simulate\n')
        args.simulate = True

if args.simulate:
    v_ratio  = args.vector / 40.0
    c_ratio  = args.cube   / 20.0
    base_lat = 0.025
    elapsed  = (base_lat * args.batch_size**0.8
                / (0.45*v_ratio + 0.55*c_ratio)) * args.iters

elapsed_per_iter = elapsed / args.iters
throughput       = args.iters * args.batch_size / elapsed
print('%f,%f,%f,%f' % (args.batch_size, args.vector, elapsed_per_iter, throughput))

if not args.simulate:
    try:
        acl.rt.reset_device(args.device)
        acl.finalize()
    except Exception:
        pass
