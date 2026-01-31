# -*- coding: utf-8 -*-
"""
NPU Horizontal Scaling Utils Adapter
适配器模式：复用 scheduling/utils_npu.py 的实例管理功能
为 horizontal_scaling 组件提供兼容的接口
"""
import os
import sys

# 添加 scheduling 目录到 path
scheduling_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scheduling")
if scheduling_dir not in sys.path:
    sys.path.insert(0, scheduling_dir)

# 复用 scheduling 中的组件
try:
    from utils_npu import start_instance as _start_instance_npu
    from utils_npu import stop_instance as _stop_instance_npu
    _npu_utils_available = True
except ImportError as e:
    print(f"[HorizontalScalingUtilsNPU] Warning: utils_npu not available: {e}")
    _npu_utils_available = False


def start_instance(selected_npus, instance_id, image_name, service_name, args, allocated_port, ip_address="127.0.0.1"):
    """
    启动NPU实例（兼容horizontal_scaling接口）
    
    Args:
        selected_npus: 选中的NPU列表 [{"id", "ip", "index"}, ...]
        instance_id: 实例ID
        image_name: 镜像名称（NPU场景下忽略）
        service_name: 服务名称
        args: 参数字典，包含 cube_requests, cube_limits, vector_requests, vector_limits, memory, MODEL_PATH 等
        allocated_port: 分配的端口
        ip_address: IP地址
    
    Returns:
        None（后台启动进程）
    """
    if not _npu_utils_available:
        raise RuntimeError("NPU utils not available")
    
    # 复用 scheduling/utils_npu.py 的实现
    # 注意：utils_npu.start_instance 的签名略有不同
    return _start_instance_npu(
        selected_npus=selected_npus,
        instance_id=instance_id,
        image_name=image_name,
        service_name=service_name,
        args=args,
        allocated_port=allocated_port,
        ip_address=ip_address
    )


def stop_instance(service_name, instance_id, ip_address="127.0.0.1"):
    """
    停止NPU实例（兼容horizontal_scaling接口）
    
    Args:
        service_name: 服务名称
        instance_id: 实例ID
        ip_address: IP地址
    
    Returns:
        None
    """
    if not _npu_utils_available:
        raise RuntimeError("NPU utils not available")
    
    # 复用 scheduling/utils_npu.py 的实现
    return _stop_instance_npu(
        service_name=service_name,
        instance_id=instance_id,
        ip_address=ip_address
    )


# 兼容原有接口的别名
start_npu_instance = start_instance
stop_npu_instance = stop_instance


if __name__ == "__main__":
    print("Testing HorizontalScalingUtilsNPU Adapter...")
    
    if not _npu_utils_available:
        print("NPU utils not available, skipping tests")
        sys.exit(1)
    
    print("Utils adapter is ready to use")
    print("Functions available:")
    print("  - start_instance(selected_npus, instance_id, image_name, service_name, args, allocated_port, ip_address)")
    print("  - stop_instance(service_name, instance_id, ip_address)")
