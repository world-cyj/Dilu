## 任务计划

### 任务1: 完善模型支持 - 添加BERT/ResNet等NPU推理脚本

基于现有的GPU版本(`scripts_tasks/run_BERT_INF_batch.py`, `run_Resnet152_INF_batch.py`, `run_GPT2_INF_batch.py`)，创建NPU版本：

**需要创建的文件：**
1. `scheduling/scripts_demo/bert_inference_npu.py` - BERT文本分类NPU服务
2. `scheduling/scripts_demo/resnet_inference_npu.py` - ResNet图像分类NPU服务  
3. `scheduling/scripts_demo/gpt2_inference_npu.py` - GPT2文本生成NPU服务
4. `scheduling/scripts_demo/cv_models_npu.py` - CV模型定义(ResNet/VGG/AlexNet)

**统一架构设计：**
- 所有服务使用与`llm_inference_npu.py`相同的模式
- 启动时应用ACL资源限制
- 支持动态批处理
- 提供/health和/predict端点
- 兼容torch_npu和ACL fallback

### 任务2: 运行端到端测试 - 使用Qwen3-4B验证完整流程

**需要创建的文件：**
1. `scheduling/scripts_demo/test_qwen3_4b_e2e.py` - Qwen3-4B端到端测试脚本

**测试流程：**
1. 启动scheduler_npu.py调度器
2. 使用/schedule_with_sla提交Qwen3-4B推理服务
3. 等待服务健康检查通过
4. 发送推理请求并验证响应
5. 测试垂直缩放(/reschedule_with_limits)
6. 测试水平扩容(启动多个实例)
7. 清理并删除实例

### 任务3: 补充Worst-Fit算法 - 完善调度策略

**需要修改的文件：**
1. `scheduling/npu_resource.py` - 添加Worst-Fit算法
2. `scheduling/scheduler_npu.py` - 添加调度策略选择和Worst-Fit支持

**算法实现：**
```python
# Worst-Fit: 选择资源最充足的NPU（分数最大）
def calculate_worst_fit_score(self, cube_req, vector_req, memory, alpha, beta):
    # 与Best-Fit相反，选择利用率最低的NPU
    # 适用于training任务，避免资源碎片化
```

**调度策略映射：**
- inference: Best-Fit (默认)
- llm-inference: Best-Fit + 显存最小优先
- training: Worst-Fit (多卡训练需要更多资源空间)

请确认这个计划后，我将开始编写完整的代码。