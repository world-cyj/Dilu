import subprocess
import shlex
import os
import sys
import tempfile

def start_instance(selected_npus, instance_id, image_name, service_name, args, allocated_port, ip_address):
    """启动实例：在本机创建子进程，设置 ACL 资源限制后执行任务"""
    npu_indexes = [str(npu['index']) for npu in selected_npus]
    npu_index = npu_indexes[0]  # 暂时使用第一个 NPU 设备
    
    # 提取资源参数
    cube_requests = args.get('cube_requests', 0.25)
    cube_limits = args.get('cube_limits', 0.75)
    vector_requests = args.get('vector_requests', 0.25)
    vector_limits = args.get('vector_limits', 0.75)
    memory = args.get('memory', 10)
    
    # 构建启动脚本
    script_content = f'''
import os
import sys
import acl
import acldvpp

# 初始化 ACL
acl.init()

# 设置设备
device_id = {npu_index}
acldvpp.set_device(device_id)

# 设置设备资源限制
# 注意：实际值需要根据硬件和需求调整
cube_core_count = int({cube_limits} * 100)  # 假设总 Cube Core 为 100
vector_core_count = int({vector_limits} * 100)  # 假设总 Vector Core 为 100

# 设置资源限制
ret = acl.rt.set_device_res_limit(device_id, acl.RT_ACL_RES_TYPE_CUBE, cube_core_count)
if ret != 0:
    print(f"Failed to set cube core limit: {ret}")
    sys.exit(1)

ret = acl.rt.set_device_res_limit(device_id, acl.RT_ACL_RES_TYPE_VECTOR, vector_core_count)
if ret != 0:
    print(f"Failed to set vector core limit: {ret}")
    sys.exit(1)

# 执行实际任务
os.environ['DEVICE_ID'] = str(device_id)
os.environ['ACL_DEVICE_INDEX'] = str(device_id)
os.environ['cube_requests'] = str({cube_requests})
os.environ['cube_limits'] = str({cube_limits})
os.environ['vector_requests'] = str({vector_requests})
os.environ['vector_limits'] = str({vector_limits})
os.environ['memory'] = str({memory})
os.environ['is_llm'] = str({args.get('is_llm', 0)})
os.environ['priority'] = str({args.get('priority', 'low')})

# 执行命令
import subprocess
command = "{}"
print(f"Executing command: {{command}}")
subprocess.run(command, shell=True)

# 释放资源
acl.rt.reset_device(device_id)
acldvpp.destroy_resource()
acl.finalize()
'''
    
    # 处理命令参数
    task_command = args.get('COMMAND', '')
    if "deepspeed" not in service_name:
        task_command += f" --port {allocated_port}"
    
    script_content = script_content.format(task_command)
    
    # 创建临时脚本文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # 执行脚本
        print(f"Starting instance {instance_id} on NPU {npu_index}")
        print(f"Executing script: {script_path}")
        
        # 启动子进程
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 记录进程信息
        pid_file = f'/tmp/dilu_instance_{instance_id}.pid'
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"Instance {instance_id} started with PID: {process.pid}")
        print(f"PID file saved to: {pid_file}")
        
    except Exception as e:
        print(f"Error starting instance: {e}")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)

def stop_instance(service_name, instance_id, ip_address):
    """停止实例：根据 PID 文件终止进程"""
    pid_file = f'/tmp/dilu_instance_{instance_id}.pid'
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            print(f"Stopping instance {instance_id} with PID: {pid}")
            subprocess.run(['kill', '-9', str(pid)], check=True)
            os.remove(pid_file)
            print(f"Instance {instance_id} stopped successfully")
        except Exception as e:
            print(f"Error stopping instance: {e}")
    else:
        print(f"PID file not found for instance {instance_id}")
