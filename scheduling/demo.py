# 保证无论从何目录运行都加载本项目的 npu 包（否则可能加载到系统其它 npu 导致 ACL available: False）
import os
import sys
_sched_dir = os.path.dirname(os.path.abspath(__file__))
if _sched_dir not in sys.path:
    sys.path.insert(0, _sched_dir)

from npu.acl_rt_wrapper import (
    is_acl_available,
    init_device,
    set_device_res_limit,
    reset_device_res_limit,
    get_device_res_limit,
    init_context,
    apply_device_res_limit,
    ACL_RT_DEV_RES_CUBE_CORE,
    ACL_RT_DEV_RES_VECTOR_CORE,
)
print("ACL available:", is_acl_available())
# ACL 约束：必须先 init_device，再 set_device_res_limit，最后 init_context；且 set_device_res_limit 必须在 create_context/算子之前
if is_acl_available():
    ok, err = init_device(0)
    print("init_device(0):", ok, err)
    if ok:
        ok, err = set_device_res_limit(0, ACL_RT_DEV_RES_CUBE_CORE, 10)
        print("set_device_res_limit(0, CUBE, 10):", ok, err)
    if ok:
        ok, err = set_device_res_limit(0, ACL_RT_DEV_RES_VECTOR_CORE, 20)
        print("set_device_res_limit(0, VECTOR, 20):", ok, err)
    if ok:
        ok, err = init_context(0)
        print("init_context(0):", ok, err)
    if ok:
        a = get_device_res_limit(0, ACL_RT_DEV_RES_CUBE_CORE)
        print("get_device_res_limit(CUBE):", a)

import torch
if hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.device("npu:%d" % 0)
elif torch.cuda.is_available():
    device = torch.device("cuda:%d" % 0)
else:
    device = torch.device("cpu")
print(device)