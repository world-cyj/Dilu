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
        # 综合碎片：1 - 利用率，越小越“满”，Best-Fit 选分数最小的
        core_fragmentation = 1.0 - (cube_frac + vector_frac) / 2.0
        memory_fragmentation = 1.0 - mem_frac
        return (alpha * core_fragmentation + beta * memory_fragmentation 
                + delta * (1.0 - (cube_frac + vector_frac) / 2.0))
