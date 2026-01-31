from flask import Flask, request, jsonify
import threading
import uuid
import utils_docker


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            if not self.available_ports:
                return None  #       
            return self.available_ports.pop()

    def release_port(self, port):
        with self.lock:
            if 15000 <= port <= 20000:
                self.available_ports.add(port)

class NPU:
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
        return (self.current_cube_req + cube_req <= omega * self.total_cube and
                self.current_cube_lim + cube_lim <= gamma * self.total_cube and
                self.current_vector_req + vector_req <= omega * self.total_vector and
                self.current_vector_lim + vector_lim <= gamma * self.total_vector and
                self.current_memory + memory <= self.total_memory)

    def calculate_score(self, cube_req, vector_req, memory, alpha, beta):
        cube_fragmentation = 1 - (self.current_cube_req + cube_req) / self.total_cube
        vector_fragmentation = 1 - (self.current_vector_req + vector_req) / self.total_vector
        memory_fragmentation = 1 - (self.current_memory + memory) / self.total_memory
        return alpha * (cube_fragmentation + vector_fragmentation) / 2 + beta * memory_fragmentation



class Instance:
    def __init__(self, id, memory, cube_requests, cube_limits, vector_requests, vector_limits, image, type, service_name, allocated_port):
        self.instance_id = id
        self.memory = memory
        self.cube_req = cube_requests
        self.cube_lim = cube_limits
        self.vector_req = vector_requests
        self.vector_lim = vector_limits
        self.image = image
        self.type = type
        self.service_name = service_name
        self.deployed_npus = []
        self.port = allocated_port

    def assign_to_npu(self, deployed_npu):
        self.deployed_npus.append(deployed_npu)

    def assign_to_gpu(self, deployed_gpu):
        self.deployed_gpus.append(deployed_gpu)

app = Flask(__name__)
nodes_info = [
    {"ip": "localhost", "index": 0},
    {"ip": "localhost", "index": 1},
    {"ip": "localhost", "index": 2},
    {"ip": "localhost", "index": 3},
]


alpha = 0.6
beta = 0.4
omega = 1
gamma = 1.5
new_npus = [NPU(i, 40, 1, 1, node['ip'], node['index']) for i, node in enumerate(nodes_info)]
active_npus = []
lock = threading.Lock()

def find_colocated_NPUs(service_name):
    candidate_npus = set()
    for active_npu in active_npus:
        early_instnace_existance = False
        # to find early instance
        for _, instance in active_npu.instances.items():
            if instance.service_name == service_name:
                early_instnace_existance = True
                break
        if early_instnace_existance:
            # judge that whether having colocated training task
            for _, instance in active_npu.instances.items():
                if instance.type == 'training':
                    for npu in instance.deployed_npus:
                        candidate_npus.add(npu)
                    candidate_npus.remove(active_npu) # at the same time, remove current active_npu
        # otherwise, current active_npu does not have the early instance
    return list(candidate_npus)



def select_optimal_NPU(candidate_npus, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma, alpha, beta):
    best_score = float('inf')
    best_npu = None
    for npu in candidate_npus:
        if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma):
            score = npu.calculate_score(cube_req, vector_req, memory, alpha, beta)
            if score < best_score:
                best_score = score
                best_npu = npu
    return best_npu



def update_deploy_info_npu(instance, cube_req, cube_lim, vector_req, vector_lim, memory, best_npu, selected_npus):
    best_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, memory)
    instance.assign_to_npu(best_npu)
    selected_npus.append({'id': best_npu.id, 'ip': best_npu.ip_address, 'index': best_npu.index})



def print_cluster_resources():
    print("Current Cluster Resource Usage:")
    print("==============active_npus==================")
    for npu in active_npus:  # Assuming new_npus should also be monitored
        print(f"NPU {npu.id} on Node {npu.index} of [{npu.ip_address}]:")
        print(f"  Total Memory: {npu.total_memory} GB")
        print(f"  Used Memory: {npu.current_memory} GB")
        print(f"  Cube Core Total: {npu.total_cube}")
        print(f"  Vector Core Total: {npu.total_vector}")
        print(f"  Current Cube Requests: {npu.current_cube_req}")
        print(f"  Current Cube Limits: {npu.current_cube_lim}")
        print(f"  Current Vector Requests: {npu.current_vector_req}")
        print(f"  Current Vector Limits: {npu.current_vector_lim}")
        print(f"  Instances: {len(npu.instances)} running")
        for instance_id, details in npu.instances.items():
            print(f"    Instance {instance_id}: Cube Req {details.cube_req}, Cube Lim {details.cube_lim}, Vector Req {details.vector_req}, Vector Lim {details.vector_lim}, Mem {details.memory} GB")
    print("")
    
    print("==============new_npus==================")
    for npu in new_npus: 
        print(f"NPU {npu.id} on Node {npu.index} of [{npu.ip_address}]:")
        print(f"  Total Memory: {npu.total_memory} GB")
        print(f"  Used Memory: {npu.current_memory} GB")
        print(f"  Cube Core Total: {npu.total_cube}")
        print(f"  Vector Core Total: {npu.total_vector}")
        print(f"  Current Cube Requests: {npu.current_cube_req}")
        print(f"  Current Cube Limits: {npu.current_cube_lim}")
        print(f"  Current Vector Requests: {npu.current_vector_req}")
        print(f"  Current Vector Limits: {npu.current_vector_lim}")
        print(f"  Instances: {len(npu.instances)} running")
        for instance_id, details in npu.instances.items():
            print(f"    Instance {instance_id}: Cube Req {details.cube_req}, Cube Lim {details.cube_lim}, Vector Req {details.vector_req}, Vector Lim {details.vector_lim}, Mem {details.memory} GB")
    print("")


    

def start_instance(selected_npus, instance_id, args, allocated_port):
    ip_address = selected_npus[0]['ip']
    utils_docker.start_instance(selected_npus, instance_id, args['image'], args['service_name'], args, allocated_port, ip_address)


def stop_instance(service_name, instance_id, ip_address):
    utils_docker.stop_instance(service_name, instance_id, ip_address)



@app.route('/schedule', methods=['POST'])
def schedule_instances():
    data = request.get_json()
    instance_id = str(uuid.uuid4())  # Generate a unique instance ID
    allocated_port = port_manager.allocate_port()
    if not allocated_port:
        return jsonify({'error': 'No available ports'}), 503
    
    n_npus_needed = data['num']
    cube_req = data['cube_requests']
    cube_lim = data['cube_limits']
    vector_req = data['vector_requests']
    vector_lim = data['vector_limits']
    memory = data['memory']
    image = data['image']
    type = data['type']
    service_name = data['service_name']
    selected_npus = []

    instance = Instance(instance_id, memory, cube_req, cube_lim, vector_req, vector_lim, image, type, service_name, allocated_port)

    with lock:
        if type == 'inference': # non-llm model execute best-fit allocation algorithm
            best_npu = None
            # find the NPUs which coloates the eary instances of current instance with the dp/pp training instances
            # first, LB_npus
            LB_npus = find_colocated_NPUs(service_name)
            
            best_npu = select_optimal_NPU(LB_npus, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma, alpha, beta)
            if best_npu:
                update_deploy_info_npu(instance, cube_req, cube_lim, vector_req, vector_lim, memory, best_npu, selected_npus)
            else:
                # second, active NPUs
                left_active_npus = set(active_npus) - set(LB_npus)
                best_npu = select_optimal_NPU(left_active_npus, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma, alpha, beta)
                if best_npu:
                    update_deploy_info_npu(instance, cube_req, cube_lim, vector_req, vector_lim, memory, best_npu, selected_npus)
                else:
                    # third, new_npus/the whole new npus
                    best_npu = select_optimal_NPU(new_npus, cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma, alpha, beta)
                    if best_npu:
                        update_deploy_info_npu(instance, cube_req, cube_lim, vector_req, vector_lim, memory, best_npu, selected_npus)
                        # update avtivate_npu list and new_npu
                        new_npus.remove(best_npu)
                        active_npus.append(best_npu)
                    else: # no enough NPUs
                        return jsonify({'error': 'Not enough resources'}), 400
          
        elif type == 'llm-inference':  # llm model worst-fit algorithm
            best_npu = None
            best_fit_score = float('inf') 
            for npu in active_npus:
                if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, memory, omega, gamma):
                    score = npu.current_memory 
                    if score < best_fit_score:
                        best_fit_score = score
                        best_npu = npu

            if best_npu:
                best_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, memory)
                instance.assign_to_npu(best_npu)
                selected_npus.append({'id': best_npu.id, 'ip': best_npu.ip_address, 'index': best_npu.index})
            else:
                node_memory_groups = {}
                for npu in active_npus:
                    if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, 0, omega, gamma):
                        node_memory_groups.setdefault(npu.ip_address, []).append(npu)
                found = False
                
                for node, npus_on_node in node_memory_groups.items():
                    npus_on_node.sort(key=lambda x: x.current_memory, reverse=True)           
                    total_available_memory = sum(npu.total_memory - npu.current_memory for npu in npus_on_node)
                    if total_available_memory >= memory:
                        remaining_memory = memory
                        for npu in npus_on_node:
                            if remaining_memory <= 0:
                                break
                            alloc_memory = min(npu.total_memory - npu.current_memory, remaining_memory)
                            npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, alloc_memory) 
                            instance.assign_to_npu(npu)
                            selected_npus.append({'id': npu.id, 'ip': npu.ip_address, 'index': npu.index})
                            remaining_memory -= alloc_memory
                        found = True
                        break
                    
                if not found:
                    # If no active NPUs can handle it, start a new NPU instance
                    if new_npus:
                        new_npu = new_npus.pop(0)
                        new_npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, memory)
                        instance.assign_to_npu(new_npu)
                        active_npus.append(new_npu)
                        selected_npus.append({'id': new_npu.id, 'ip': new_npu.ip_address, 'index': new_npu.index})
                    else:
                        return jsonify({'error': 'Unable to allocate resources, and no new NPUs available'}), 400
                    
        elif type == 'training':
            node_npus = {}
            # Group NPUs by node and check if each can accommodate its part of the task
            for npu in active_npus:
                if all(npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, mem, omega, gamma) for mem in memory):
                    node_npus.setdefault(npu.ip_address, []).append(npu)
            # Check if there's a node with enough NPUs
            allocated_npus = None
            for node_ip, npus_on_node in node_npus.items():
                if len(npus_on_node) >= n_npus_needed:
                    # Ensure these NPUs can be allocated according to memory requirements
                    if all(len([npu for npu in npus_on_node if npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, mem, omega, gamma)]) >= 1 for mem in memory):
                        allocated_npus = npus_on_node[:n_npus_needed]
                        break
            # If not enough NPUs on any active node, check new NPUs
            if not allocated_npus:
                for node_ip in nodes_info:  # Iterate over all possible node IPs directly from nodes_info
                    new_npus_on_node = [npu for npu in new_npus if npu.ip_address == node_ip['ip'] and all(npu.can_allocate(cube_req, cube_lim, vector_req, vector_lim, mem, omega, gamma) for mem in memory)]
                    if len(new_npus_on_node) + len(node_npus.get(node_ip['ip'], [])) >= n_npus_needed:
                        # Take needed NPUs from active and new NPUs
                        allocated_npus = node_npus.get(node_ip['ip'], []) + new_npus_on_node[:n_npus_needed - len(node_npus.get(node_ip['ip'], []))]
                        # Update active and new NPU lists
                        for npu in allocated_npus:
                            if npu in new_npus:
                                new_npus.remove(npu)
                                active_npus.append(npu)
                        break  # Found a node with enough resources, break the loop

            # Update resource information and collect the selected NPUs
            if allocated_npus and len(allocated_npus) == n_npus_needed:
                for npu, mem in zip(allocated_npus, memory):
                    npu.update_resources(instance, cube_req, cube_lim, vector_req, vector_lim, mem)
                    instance.assign_to_npu(npu)
                    selected_npus.append({'id': npu.id, 'ip': npu.ip_address, 'index': npu.index})
            else:
                return jsonify({'error': 'Not enough resources'}), 400

    thrd = threading.Thread(target=start_instance, args=(selected_npus, instance_id, data, allocated_port))
    thrd.start()
    print_cluster_resources()
    return jsonify({'selected_npus': selected_npus, 'instance_id': instance_id, 'port': allocated_port}), 200


@app.route('/delete_instance', methods=['POST'])
def delete_instance():
    data = request.get_json()
    instance_id = data['instance_id']
    
    allocated_port = None 
    service_name = None
    ip_address = None
    found = False
    with lock:
        for npu in active_npus:
            if instance_id in npu.instances:
                instance = npu.instances.pop(instance_id)
                npu.current_cube_req -= instance.cube_req
                npu.current_cube_lim -= instance.cube_lim
                npu.current_vector_req -= instance.vector_req
                npu.current_vector_lim -= instance.vector_lim
                npu.current_memory -= instance.memory
                found = True
                ip_address = instance.deployed_npus[0].ip_address
                
                allocated_port = instance.port
                service_name = instance.service_name
                # Check if the NPU is now free and if so, move it to new_npus
                if not npu.instances and npu.current_memory == 0 and npu.current_cube_req == 0 and npu.current_cube_lim == 0 and npu.current_vector_req == 0 and npu.current_vector_lim == 0:
                    active_npus.remove(npu)
                    new_npus.append(npu)
                break

    if not found:
        return jsonify({'error': 'Instance not found'}), 404

    port_manager.release_port(allocated_port)  #                  
    thrd = threading.Thread(target=stop_instance, args=(service_name, instance_id, ip_address,))
    thrd.start()
    print_cluster_resources()
    return jsonify({'status': 'deleted', 'instance_id': instance_id}), 200


if __name__ == '__main__':
    port_manager = PortManager()
    app.run(debug=False, host='0.0.0.0', port=5000)
