#!/usr/bin/env python3
"""
批量复制NPU相关文件到npu目录
"""
import os
import shutil
from pathlib import Path

# 定义源文件和目标目录的映射
FILE_MAPPINGS = {
    # Scheduling核心文件
    "scheduling/npu_resource.py": "scheduling/core/npu_resource.py",
    "scheduling/scheduler_npu.py": "scheduling/core/scheduler_npu.py",
    "scheduling/utils_npu.py": "scheduling/core/utils_npu.py",
    "scheduling/resource_recommender.py": "scheduling/core/resource_recommender.py",
    "scheduling/scaler_npu.py": "scheduling/core/scaler_npu.py",
    
    # ACL封装
    "scheduling/npu/acl_rt_wrapper.py": "scheduling/utils/acl_rt_wrapper.py",
    "scheduling/npu/__init__.py": "scheduling/utils/__init__.py",
    
    # Demo脚本
    "scheduling/scripts_demo/run_demo_npu.py": "scheduling/scripts/demo/run_demo_npu.py",
    "scheduling/scripts_demo/llm_inference_npu.py": "scheduling/scripts/demo/llm_inference_npu.py",
    "scheduling/scripts_demo/npu_worker_entry.py": "scheduling/scripts/demo/npu_worker_entry.py",
    "scheduling/scripts_demo/verify_core_functions.py": "scheduling/scripts/demo/verify_core_functions.py",
    "scheduling/scripts_demo/test_resource_recommendation.py": "scheduling/scripts/demo/test_resource_recommendation.py",
    
    # Profiling文件
    "profiling/npu/profile_npu.py": "profiling/inference/profile_npu.py",
    "profiling/npu/run_one_observation_npu.py": "profiling/inference/run_one_observation_npu.py",
    
    # Adaptive scaling文件
    "adaptive_2D_scaling/horizontal_scaling/scaler_npu.py": "adaptive_scaling/horizontal/scaler_npu.py",
    "adaptive_2D_scaling/horizontal_scaling/scheduler_npu_adapter.py": "adaptive_scaling/horizontal/scheduler_npu_adapter.py",
    "adaptive_2D_scaling/horizontal_scaling/utils_npu_adapter.py": "adaptive_scaling/horizontal/utils_npu_adapter.py",
    
    # Workload生成器
    "scheduling/simulations/workload/service_generator_npu.py": "scheduling/scripts/demo/service_generator_npu.py",
}

def copy_files():
    """复制所有NPU文件"""
    base_dir = Path("/vllm-workspace/Dilu")
    npu_dir = base_dir / "npu"
    
    print("开始复制NPU文件...")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for src_rel, dst_rel in FILE_MAPPINGS.items():
        src_path = base_dir / src_rel
        dst_path = npu_dir / dst_rel
        
        # 确保目标目录存在
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not src_path.exists():
            print(f"⚠ 源文件不存在: {src_rel}")
            fail_count += 1
            continue
        
        try:
            # 读取源文件内容
            with open(src_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 写入目标文件
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ {src_rel} -> {dst_rel}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 复制失败 {src_rel}: {e}")
            fail_count += 1
    
    print("=" * 60)
    print(f"复制完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    
    # 创建__init__.py文件
    create_init_files(npu_dir)
    
    return success_count, fail_count

def create_init_files(npu_dir):
    """创建必要的__init__.py文件"""
    init_locations = [
        "scheduling",
        "scheduling/core",
        "scheduling/scripts",
        "scheduling/scripts/demo",
        "scheduling/utils",
        "profiling",
        "profiling/inference",
        "adaptive_scaling",
        "adaptive_scaling/horizontal",
    ]
    
    print("\n创建__init__.py文件...")
    for loc in init_locations:
        init_file = npu_dir / loc / "__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""NPU {loc.replace("/", ".")} module"""\n')
            print(f"✓ 创建 {loc}/__init__.py")

if __name__ == "__main__":
    copy_files()
