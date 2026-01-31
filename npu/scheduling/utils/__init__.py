# NPU 资源层：ACL 封装与资源池（华为 910B3）
from .acl_rt_wrapper import (
    set_device_res_limit,
    get_device_res_limit,
    reset_device_res_limit,
    init_device,
    init_context,
    apply_device_res_limit,
    is_acl_available,
    ACL_RT_DEV_RES_CUBE_CORE,
    ACL_RT_DEV_RES_VECTOR_CORE,
)

__all__ = [
    "set_device_res_limit",
    "get_device_res_limit",
    "reset_device_res_limit",
    "init_device",
    "init_context",
    "apply_device_res_limit",
    "is_acl_available",
    "ACL_RT_DEV_RES_CUBE_CORE",
    "ACL_RT_DEV_RES_VECTOR_CORE",
]
