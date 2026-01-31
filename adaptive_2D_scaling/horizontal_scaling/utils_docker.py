import subprocess
import os
import shlex
import tempfile

def start_instance(selected_npus, instance_id, image_name, service_name, args, allocated_port, ip_address):
    npu_indices = ','.join([str(npu['index']) for npu in selected_npus])
    
    cube_requests = str(args.get('cube_requests', 0.25))
    cube_limits = str(args.get('cube_limits', 0.75))
    vector_requests = str(args.get('vector_requests', 0.25))
    vector_limits = str(args.get('vector_limits', 0.75))
    is_llm = str(args.get('is_llm', 0))
    priority = args.get('priority', 'low')
    
    if "deepspeed" not in service_name:
        command = args.get('COMMAND', '') + f" --port {allocated_port} "
    else:
        command = args.get('COMMAND', '') 
    
    log_file = f'/vllm-workspace/Dilu/workloads/job_logs/{service_name}-{instance_id}.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(f'''
import os
import subprocess
import acl
import acldvpp

npu_index = {npu_indices.split(',')[0]}
acldvpp.set_device(npu_index)
acl.rt.set_device(npu_index)

cube_core_count = int({cube_limits} * 100)
vector_core_count = int({vector_limits} * 100)
acl.rt.set_device_res_limit(npu_index, acl.RT_ACL_RES_TYPE_CUBE, cube_core_count)
acl.rt.set_device_res_limit(npu_index, acl.RT_ACL_RES_TYPE_VECTOR, vector_core_count)

os.environ['ACL_DEVICE_INDEX'] = str(npu_index)
os.environ['requests_rate'] = '{cube_requests}'
os.environ['limits_rate'] = '{cube_limits}'
os.environ['is_llm'] = '{is_llm}'
os.environ['priority'] = '{priority}'

command = {shlex.quote(command)}
log_file = {shlex.quote(log_file)}

print(f"Executing command: {{command}}")
print(f"Logging to: {{log_file}}")

with open(log_file, 'w') as log:
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=log, 
        stderr=log
    )
    
pid_file = f'/vllm-workspace/Dilu/workloads/pids/{service_name}-{instance_id}.pid'
os.makedirs(os.path.dirname(pid_file), exist_ok=True)
with open(pid_file, 'w') as f:
    f.write(str(process.pid))
        ''')
        temp_script_path = f.name
    
    try:
        python_command = f"python3 {temp_script_path}"
        print(f"Executing Python command: {{python_command}}")
        result = subprocess.run(python_command, shell=True, capture_output=True, text=True)
        
        print(f"Python command output: {{result.stdout}}")
        print(f"Python command error: {{result.stderr}}")
        print(f"Python command return code: {{result.returncode}}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Python command failed with return code {{result.returncode}}")
    finally:
        os.unlink(temp_script_path)

def stop_instance(service_name, instance_id, ip_address):
    pid_file = f'/vllm-workspace/Dilu/workloads/pids/{service_name}-{instance_id}.pid'
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            try:
                pid = int(f.read().strip())
                subprocess.run(['kill', '-9', str(pid)], capture_output=True)
                print(f"Stopped instance {instance_id} with PID {pid}")
            except ValueError:
                print(f"Invalid PID in file {{pid_file}}")
        os.unlink(pid_file)
    else:
        print(f"PID file not found: {{pid_file}}")
    
    log_file = f'/vllm-workspace/Dilu/workloads/job_logs/{service_name}-{instance_id}.log'
    if os.path.exists(log_file):
        os.unlink(log_file)
        print(f"Removed log file: {{log_file}}")
