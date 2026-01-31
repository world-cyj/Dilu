# -*- coding: utf-8 -*-
"""
NPU 资源模型：单机 4 张 910B3，每卡 20 Cube + 40 Vector + 64GB 显存。
用于调度器中的 NPU 池与实例配额（cube/vector request & limit, memory）。
"""
# 910B3 单卡规格
NPU_CUBE_CORES_PER_DEVICE = 20
NPU_VECTOR_CORES_PER_DEVICE = 40
NPU_MEMORY_GB_PER_DEVICE = 64

# 单机 4 卡：仅用 index 0~3，ip 固定为 localhost
NPU_NUM_DEVICES = 4
NPU_DEFAULT_IP = "127.0.0.1"


def build_npu_nodes_info():
    """单机 4 卡：[(ip, index), ...]"""
    return [
        {"ip": NPU_DEFAULT_IP, "index": i}
        for i in range(NPU_NUM_DEVICES)
    ]


class NPU:
    """
    单张 NPU 卡的资源视图。
    配额维度：Cube request/limit、Vector request/limit、Memory(GB)。
    """
    def __init__(self, id, total_memory, total_cube, total_vector, ip_address, index):
        self.id = id
        self.total_memory = total_memory
        self.total_cube = total_cube
        self.total_vector = total_vector
        self.current_cube_req = 0
        self.current_cube_lim = 0
        self.current_vector_req = 0
        self.current_vector_lim = 0
        self.current_memory = 0
        self.instances = {}
        self.ip_address = ip_address
        self.index = index

    def update_resources(self, instance, cube_req, cube_lim, vector_req, vector_lim, memory):
        self.current_cube_req += cube_req
        self.current_cube_lim += cube_lim
        self.current_vector_req += vector_req
        self.current_vector_lim += vector_lim
        self.current_memory += memory
        self.instances[instance.instance_id] = instance

    def can_allocate(self, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma):
        return (
            self.current_cube_req + cube_req <= omega * self.total_cube
            and self.current_cube_lim + cube_lim <= gamma * self.total_cube
            and self.current_vector_req + vector_req <= omega * self.total_vector
            and self.current_vector_lim + vector_lim <= gamma * self.total_vector
            and self.current_memory + memory <= self.total_memory
        )

    def calculate_score(self, cube_req, vector_req, memory, alpha, beta, delta=0.0):
        """Best-Fit 碎片率：越小越适合放置（cube/vector 各占一部分）。"""
        cube_frac = (self.current_cube_req + cube_req) / max(1, self.total_cube)
        vector_frac = (self.current_vector_req + vector_req) / max(1, self.total_vector)
        mem_frac = (self.current_memory + memory) / max(1, self.total_memory)
        # 综合碎片：1 - 利用率，越小越"满"，Best-Fit 选分数最小的
        core_fragmentation = 1.0 - (cube_frac + vector_frac) / 2.0
        memory_fragmentation = 1.0 - mem_frac
        return (alpha * core_fragmentation + beta * memory_fragmentation
                + delta * (1.0 - (cube_frac + vector_frac) / 2.0))

    def calculate_worst_fit_score(self, cube_req, vector_req, memory, alpha=0.5, beta=0.5):
        """
        Worst-Fit 算法：选择资源最充足的NPU（分数越大越适合）

        与Best-Fit相反，Worst-Fit选择利用率最低的NPU，适用于：
        1. Training任务：需要预留足够资源空间，避免资源碎片化
        2. 大模型推理：需要确保有足够资源应对峰值

        算法思路：
        - 计算分配后的资源利用率（Cube/Vector/Memory）
        - 利用率越低，分数越高（越适合Worst-Fit）
        - 选择分数最高的NPU

        Args:
            cube_req: 请求的Cube核心数
            vector_req: 请求的Vector核心数
            memory: 请求的内存(GB)
            alpha: Cube/Vector资源权重
            beta: Memory资源权重

        Returns:
            Worst-Fit分数，越大表示该NPU越适合放置（资源越充足）
        """
        # 计算分配后的剩余资源比例（剩余越多，分数越高）
        remaining_cube = (self.total_cube - self.current_cube_req - cube_req) / max(1, self.total_cube)
        remaining_vector = (self.total_vector - self.current_vector_req - vector_req) / max(1, self.total_vector)
        remaining_memory = (self.total_memory - self.current_memory - memory) / max(1, self.total_memory)

        # 确保剩余资源不为负
        remaining_cube = max(0.0, remaining_cube)
        remaining_vector = max(0.0, remaining_vector)
        remaining_memory = max(0.0, remaining_memory)

        # 综合剩余资源比例：越大表示资源越充足
        core_remaining = (remaining_cube + remaining_vector) / 2.0

        # Worst-Fit分数：越大越适合放置
        return alpha * core_remaining + beta * remaining_memory

    def get_resource_utilization(self):
        """获取当前资源利用率"""
        return {
            "cube_util": self.current_cube_req / max(1, self.total_cube),
            "vector_util": self.current_vector_req / max(1, self.total_vector),
            "memory_util": self.current_memory / max(1, self.total_memory),
            "total_util": (self.current_cube_req / max(1, self.total_cube) +
                          self.current_vector_req / max(1, self.total_vector) +
                          self.current_memory / max(1, self.total_memory)) / 3.0
        }
