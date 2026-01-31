# -*- coding: utf-8 -*-
"""
ACL 运行时封装：华为 NPU 910B3 核心与显存管控。
- set_device_res_limit(device_id, res_type, value)  # 在 set_device 之后、算子执行之前调用
- get_device_res_limit(device_id, res_type)
- reset_device_res_limit(device_id)
- init_device(device_id) / init_context(device_id)
无 ACL 环境时提供 stub，便于在无 NPU 环境做接口联调。
"""
from __future__ import absolute_import

# 资源类型：与 ACL 定义一致
ACL_RT_DEV_RES_CUBE_CORE = 0   # AI Core / Cube Core
ACL_RT_DEV_RES_VECTOR_CORE = 1 # Vector Core

_acl_available = False
_acl_rt = None

try:
    import acl
    _acl_rt = acl.rt
    _acl_available = True
except ImportError:
    pass


def _check_device_id(device_id):
    if not (0 <= device_id <= 7):
        raise ValueError("device_id must be 0-7, got %s" % device_id)


def set_device_res_limit(device_id, res_type, value):
    """
    设置单卡某类核心数量上限。
    **ACL 约束**：必须在 set_device(device_id) 之后、create_context/算子执行之前调用；
    同一 device 多次设置以最后一次为准。
    推荐顺序：init_device(device_id) -> set_device_res_limit(cube) -> set_device_res_limit(vector) -> init_context(device_id)。
    :param device_id: 设备 ID (0~3 单机 4 卡)
    :param res_type: ACL_RT_DEV_RES_CUBE_CORE 或 ACL_RT_DEV_RES_VECTOR_CORE
    :param value: 核心数量 (Cube 0~20, Vector 0~40 for 910B3)
    :return: (True, None) 成功; (False, error_msg) 失败
    """
    _check_device_id(device_id)
    if not _acl_available:
        return True, None  # stub: 无 NPU 时直接通过
    try:
        # 约束：必须先 set_device，再 set_device_res_limit，且不能在 create_context 之后
        ret = _acl_rt.set_device(device_id)
        if ret != 0:
            return False, "acl.rt.set_device returned %s (must call before set_device_res_limit)" % ret
        ret = _acl_rt.set_device_res_limit(device_id, res_type, int(value))
        if ret == 0:
            return True, None
        return False, "acl.rt.set_device_res_limit returned %s (call after set_device, before create_context/operators)" % ret
    except Exception as e:
        return False, str(e)


def get_device_res_limit(device_id, res_type):
    """
    获取当前某卡某类核心限制值。
    :return: (value, None) 成功; (None, error_msg) 失败
    """
    _check_device_id(device_id)
    if not _acl_available:
        return 0, None  # stub
    try:
        value, ret = _acl_rt.get_device_res_limit(device_id, res_type)
        if ret == 0:
            return value, None
        return None, "acl.rt.get_device_res_limit returned %s" % ret
    except Exception as e:
        return None, str(e)


def reset_device_res_limit(device_id):
    """
    重置该卡核心限制为默认（满配）。
    :return: (True, None) 成功; (False, error_msg) 失败
    """
    _check_device_id(device_id)
    if not _acl_available:
        return True, None
    try:
        ret = _acl_rt.reset_device_res_limit(device_id)
        if ret == 0:
            return True, None
        return False, "acl.rt.reset_device_res_limit returned %s" % ret
    except Exception as e:
        return False, str(e)


def init_device(device_id):
    """初始化设备。在 set_device_res_limit 之前调用。"""
    _check_device_id(device_id)
    if not _acl_available:
        return True, None
    try:
        ret = _acl_rt.set_device(device_id)
        if ret == 0:
            return True, None
        return False, "acl.rt.set_device returned %s" % ret
    except Exception as e:
        return False, str(e)


def init_context(device_id):
    """创建并绑定 context。必须在 set_device 和 set_device_res_limit 之后调用。"""
    _check_device_id(device_id)
    if not _acl_available:
        return True, None
    try:
        ret = _acl_rt.create_context(device_id)
        if ret == 0:
            return True, None
        return False, "acl.rt.create_context returned %s" % ret
    except Exception as e:
        return False, str(e)


def is_acl_available():
    return _acl_available


def apply_device_res_limit(device_id, cube_limit, vector_limit):
    """
    按 ACL 约束顺序执行：set_device -> set_device_res_limit(CUBE) -> set_device_res_limit(VECTOR)。
    之后再由调用方在需要时调用 init_context(device_id)；在调用 init_context 或执行算子之前完成本函数。
    :return: (True, None) 成功; (False, error_msg) 失败
    """
    ok, err = init_device(device_id)
    if not ok:
        return False, err
    ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_CUBE_CORE, int(cube_limit))
    if not ok:
        return False, err
    ok, err = set_device_res_limit(device_id, ACL_RT_DEV_RES_VECTOR_CORE, int(vector_limit))
    if not ok:
        return False, err
    return True, None
