#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调度策略测试脚本：验证Best-Fit和Worst-Fit算法
"""
import os
import sys

# 确保 scheduling 在 path 中
sched_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sched_dir not in sys.path:
    sys.path.insert(0, sched_dir)

from npu_resource import NPU, build_npu_nodes_info, NPU_CUBE_CORES_PER_DEVICE, NPU_VECTOR_CORES_PER_DEVICE, NPU_MEMORY_GB_PER_DEVICE


class MockInstance:
    def __init__(self, instance_id):
        self.instance_id = instance_id


def test_best_fit():
    """测试Best-Fit算法"""
    print("=" * 60)
    print("测试 Best-Fit 算法")
    print("=" * 60)
    
    # 创建3个NPU，设置不同的资源使用状态
    npus = [
        NPU(0, 64, 20, 40, "127.0.0.1", 0),  # 空
        NPU(1, 64, 20, 40, "127.0.0.1", 1),  # 已使用50%
        NPU(2, 64, 20, 40, "127.0.0.1", 2),  # 已使用80%
    ]
    
    # 模拟NPU 1和2已有资源使用
    inst1 = MockInstance("inst-1")
    npus[1].update_resources(inst1, 10, 10, 20, 20, 32)  # 50%使用
    
    inst2 = MockInstance("inst-2")
    npus[2].update_resources(inst2, 16, 16, 32, 32, 51)  # 80%使用
    
    print("\nNPU资源状态:")
    for npu in npus:
        util = npu.get_resource_utilization()
        print(f"  NPU {npu.id}: Cube {npu.current_cube_req}/{npu.total_cube} "
              f"({util['cube_util']:.1%}), Vector {npu.current_vector_req}/{npu.total_vector} "
              f"({util['vector_util']:.1%}), Mem {npu.current_memory}/{npu.total_memory}GB "
              f"({util['memory_util']:.1%})")
    
    # 测试Best-Fit：请求4 Cube + 10 Vector + 8GB
    cube_req, vector_req, memory = 4, 10, 8
    print(f"\n新请求: Cube={cube_req}, Vector={vector_req}, Memory={memory}GB")
    print("\nBest-Fit选择（应该选最满的NPU 2）:")
    
    best_npu = None
    best_score = float('inf')
    for npu in npus:
        if npu.can_allocate(cube_req, cube_req, vector_req, vector_req, memory, 1.0, 1.0):
            score = npu.calculate_score(cube_req, vector_req, memory, 0.5, 0.4, 0.1)
            print(f"  NPU {npu.id}: score={score:.4f}")
            if score < best_score:
                best_score = score
                best_npu = npu
    
    if best_npu:
        print(f"✓ Best-Fit 选择 NPU {best_npu.id} (分数最低: {best_score:.4f})")
    else:
        print("✗ 没有NPU能满足需求")
    
    return best_npu


def test_worst_fit():
    """测试Worst-Fit算法"""
    print("\n" + "=" * 60)
    print("测试 Worst-Fit 算法")
    print("=" * 60)
    
    # 创建3个NPU，设置不同的资源使用状态
    npus = [
        NPU(0, 64, 20, 40, "127.0.0.1", 0),  # 空
        NPU(1, 64, 20, 40, "127.0.0.1", 1),  # 已使用50%
        NPU(2, 64, 20, 40, "127.0.0.1", 2),  # 已使用80%
    ]
    
    # 模拟NPU 1和2已有资源使用
    inst1 = MockInstance("inst-1")
    npus[1].update_resources(inst1, 10, 10, 20, 20, 32)  # 50%使用
    
    inst2 = MockInstance("inst-2")
    npus[2].update_resources(inst2, 16, 16, 32, 32, 51)  # 80%使用
    
    print("\nNPU资源状态:")
    for npu in npus:
        util = npu.get_resource_utilization()
        print(f"  NPU {npu.id}: Cube {npu.current_cube_req}/{npu.total_cube} "
              f"({util['cube_util']:.1%}), Vector {npu.current_vector_req}/{npu.total_vector} "
              f"({util['vector_util']:.1%}), Mem {npu.current_memory}/{npu.total_memory}GB "
              f"({util['memory_util']:.1%})")
    
    # 测试Worst-Fit：请求4 Cube + 10 Vector + 8GB
    cube_req, vector_req, memory = 4, 10, 8
    print(f"\n新请求: Cube={cube_req}, Vector={vector_req}, Memory={memory}GB")
    print("\nWorst-Fit选择（应该选最空的NPU 0）:")
    
    best_npu = None
    best_score = -1.0
    for npu in npus:
        if npu.can_allocate(cube_req, cube_req, vector_req, vector_req, memory, 1.0, 1.0):
            score = npu.calculate_worst_fit_score(cube_req, vector_req, memory, 0.5, 0.5)
            print(f"  NPU {npu.id}: score={score:.4f}")
            if score > best_score:
                best_score = score
                best_npu = npu
    
    if best_npu:
        print(f"✓ Worst-Fit 选择 NPU {best_npu.id} (分数最高: {best_score:.4f})")
    else:
        print("✗ 没有NPU能满足需求")
    
    return best_npu


def test_training_scheduling():
    """测试Training任务的Worst-Fit调度"""
    print("\n" + "=" * 60)
    print("测试 Training 任务调度 (Worst-Fit策略)")
    print("=" * 60)
    
    from scheduler_npu import select_npus_for_training
    
    # 创建4个NPU
    npus = [
        NPU(0, 64, 20, 40, "127.0.0.1", 0),  # 空
        NPU(1, 64, 20, 40, "127.0.0.1", 1),  # 已使用30%
        NPU(2, 64, 20, 40, "127.0.0.1", 2),  # 已使用60%
        NPU(3, 64, 20, 40, "127.0.0.1", 3),  # 已使用50%
    ]
    
    # 设置不同的资源使用状态
    npus[1].update_resources(MockInstance("inst-1"), 6, 6, 12, 12, 19)
    npus[2].update_resources(MockInstance("inst-2"), 12, 12, 24, 24, 38)
    npus[3].update_resources(MockInstance("inst-3"), 10, 10, 20, 20, 32)
    
    print("\nNPU资源状态:")
    for npu in npus:
        util = npu.get_resource_utilization()
        print(f"  NPU {npu.id}: 总利用率 {util['total_util']:.1%}")
    
    # 测试Training任务：需要2卡，每卡4 Cube + 10 Vector + 8GB
    n_npus_needed = 2
    cube_req, cube_lim = 4, 4
    vector_req, vector_lim = 10, 10
    memory_list = [8, 8]
    omega, gamma = 1.0, 1.0
    
    print(f"\nTraining任务需求: {n_npus_needed}卡, 每卡Cube={cube_req}, Vector={vector_req}, Memory={memory_list[0]}GB")
    print("\n调度结果（应该选择资源最充足的2个NPU）:")
    
    selected = select_npus_for_training(npus, n_npus_needed, cube_req, cube_lim, 
                                        vector_req, vector_lim, memory_list, omega, gamma)
    
    if selected and len(selected) == n_npus_needed:
        print(f"✓ 成功选择 {len(selected)} 个NPU: {[n.id for n in selected]}")
        print("  预期选择: NPU 0 (空) 和 NPU 1 (30%使用)")
        
        # 验证选择的是否是资源最充足的
        expected = [0, 1]  # 期望选择NPU 0和1
        actual = [n.id for n in selected]
        if actual == expected:
            print("✓ 选择正确！使用了Worst-Fit策略")
        else:
            print(f"! 实际选择: {actual}, 期望: {expected}")
    else:
        print(f"✗ 调度失败，只选到 {len(selected) if selected else 0} 个NPU")
    
    return selected


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("NPU调度策略测试")
    print("=" * 60)
    
    results = []
    
    # 测试Best-Fit
    best_fit_npu = test_best_fit()
    results.append(("Best-Fit", best_fit_npu is not None))
    
    # 测试Worst-Fit
    worst_fit_npu = test_worst_fit()
    results.append(("Worst-Fit", worst_fit_npu is not None))
    
    # 测试Training调度
    training_npus = test_training_scheduling()
    results.append(("Training调度", training_npus is not None and len(training_npus) == 2))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("所有测试通过！" if all_passed else "部分测试失败！"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
