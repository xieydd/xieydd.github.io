# All You Need to Know About Topology Awareness in Kubernetes


Recently, I've been working on some NUMA-aware scheduling tasks on an internally developed platform, involving the discovery of Kubernetes node resource topology and scheduling. However, due to my limited knowledge, I often find myself struggling to grasp the full picture. This article is an attempt to summarize and organize my understanding.

## Why Topology Awareness is Needed

According to the [official Kubernetes documentation](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/), more and more systems are utilizing CPUs and hardware accelerators like GPUs and DPUs to support low-latency tasks and high-throughput parallel computing tasks.

However, the explanation seems a bit unclear. The fundamental reason lies in the issues brought by the Von Neumann architecture. As the saying goes, there is no silver bullet. The Von Neumann architecture separates memory and processors, with both instructions and data stored in memory, laying the foundation for the universality of modern computing. However, it also poses a hidden risk: as memory capacity increases exponentially, data transfer between the CPU and memory becomes a bottleneck. Currently, most devices in servers are connected via high-speed PCIe buses, and the bus layout may vary depending on the server's purpose. As shown in the figure below (found online, not drawn by me), in the left diagram, GPUs reside in different PCIe domains, making direct P2P copying between GPU memories impossible. To copy memory from GPU 0 to GPU 2, it must first be copied via PCIe to the memory connected to CPU 0, then transferred to CPU 1 via QPI link, and finally transferred to GPU 2 via PCIe again. This process adds significant overhead in terms of latency and bandwidth, whereas the right diagram can achieve ultra-high-speed communication through GPU P2P connections. In summary, topology affects communication between devices, impacting business stability and efficiency, necessitating some technical means to make businesses topology-aware.

{{< figure src="gpu-cpu-pcie-topo.jpg" title="PCIe Topo (figure 1)" >}}

## Types of Topology

Currently, the types of topology that need to be aware of include:
- GPU Topology Awareness
- NUMA Topology Awareness

### GPU Topology Manager

There are several implementation solutions in the industry:
- [Volcano GPU Topology Awareness](https://github.com/volcano-sh/volcano/issues/1472)
- [Baidu Intelligent Cloud GPU Topology-Aware Scheduling](https://mp.weixin.qq.com/s/uje27_MHBh8fMzWATusVwQ)

Volcano is not fully implemented yet, and Baidu's solution is closed-source, so we can only get a glimpse through shared information.

#### Why

Why is GPU topology awareness needed? Let's start with a diagram from NVIDIA, which describes the topology of the mainstream V100 GPU graphics card in servers.

{{< figure src="gpu-topo.png" title="GPU Topo (figure 2)">}}

Each V100 GPU has 6 NVLink channels, and 8 GPUs cannot achieve full connectivity, with a maximum of 2 NVLink connections between 2 GPUs. For instance, there are 2 NVLink connections between GPU0 and GPU3, and between GPU0 and GPU4, while there is only one NVLink connection between GPU0 and GPU1, and no NVLink connection between GPU0 and GPU6. Therefore, communication between GPU0 and GPU6 still requires PCIe. The unidirectional communication bandwidth of NVLink is 25 GB/s, and the bidirectional bandwidth is 50 GB/s, while the communication bandwidth of PCIe is 16 GB/s. Thus, if GPUs are allocated incorrectly during GPU training, such as a training task Pod requesting two cards, GPU0 and GPU6, cross-GPU communication may become a bottleneck for the training task.

Topology information can be viewed by executing the following command on a node:

```
# nvidia-smi topo -m
	   GPU0	GPU1 GPU2 GPU3 GPU4	GPU5 GPU6 GPU7
GPU0	 X 	PIX	 PHB  PHB  SYS	SYS	 SYS  SYS
GPU1	PIX	 X 	 PHB  PHB  SYS	SYS	 SYS  SYS
GPU2	PHB	PHB	 X 	  PIX  SYS	SYS	 SYS  SYS
GPU3	PHB	PHB	 PIX  X    SYS	SYS	 SYS  SYS
GPU4	SYS	SYS	 SYS  SYS  X 	PIX	 PHB  PHB
GPU5	SYS	SYS	 SYS  SYS  PIX	 X 	 PHB  PHB
GPU6	SYS	SYS	 SYS  SYS  PHB	PHB	 X 	  PIX
GPU7	SYS	SYS	 SYS  SYS  PHB	PHB	 PIX  X 
Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```

#### How

I won't parse it all here, as there isn't a complete implementation to look at. I'll outline some general ideas based on my understanding.

The first step is awareness, which involves using a daemon component to gather information on NVIDIA GPUs, network topology, NVLink, and PCIe. The second step is the scheduler, which defines strategies. Strategy 1: Preferably schedule GPUs with the most NVLinks under the same NUMA node to a Pod; Strategy 2: Preferably allocate GPUs and network cards under the same PCI switch to the same Pod. The general process is as follows:
1. GPU-device-plugin or other daemon processes construct the GPU topology information CRD for the node;
2. The Pod defines a topology strategy, such as Strategy 1 or Strategy 2;
3. The newly defined scheduler filters nodes that do not meet the strategy during the filter and priority phases and scores nodes that meet the strategy highly;
4. The discovery and update of GPU devices on the node are handled by the device-plugin and kubelet, as referenced in [this article](https://www.infoq.cn/article/tdfgiikxh9bcgknywl6s).

Currently, GPU topology information can be queried through the official [nvml (NVIDIA Management Library)](https://github.com/NVIDIA/go-nvml) interface.

### NUMA Topology Awareness

#### Why

When discussing NUMA topology awareness, it's essential to first explain what NUMA is and why it needs to be aware of it.

{{< figure src="numa-arch.png" title="NUMA Topo (figure 3)" >}}

{{< figure src="cpu-cache-latency.png" title="CPU Cache Latency (figure 4)">}}

The two figures above provide the answer. Modern CPUs often use the NUMA architecture, which stands for "Non-Uniform Memory Access." Why use non-uniformity? Isn't uniformity better? The answer is no, because if UMA (Uniform Memory Access) is used, as the number of physical cores on the northbridge increases and CPU frequency rises, the bus bandwidth cannot keep up, and conflicts over accessing the same memory will become more severe. Returning to the NUMA architecture, each NUMA node has its own physical CPU cores, and the cores within each NUMA node also share the L3 Cache. Additionally, memory is distributed across each NUMA node. Some CPUs with hyper-threading enabled will present two logical cores for each physical CPU core in the operating system.

From a business perspective, if programs run on the same NUMA node, they can better share some L3 Cache, which has a very fast access speed. If the L3 Cache is not hit, data can be read from memory, significantly reducing access speed.

In today's container-dominated world, the issue of incorrect CPU allocation is particularly severe. Because nodes are now oversold, with many containers running simultaneously, what happens if the same process is allocated to different NUMA nodes:
- CPU contention leads to frequent context switching time;
- Frequent process switching causes CPU cache failures;
- Cross-NUMA memory access results in more severe performance bottlenecks.

In summary, in modern CPU architectures, if NUMA topology relationships are not considered, incorrect CPU allocation can lead to performance issues, affecting business SLAs.

#### How

The previous section explained why NUMA-aware scheduling is needed. So how can NUMA topology be sensed, and what existing solutions are available? Here, I briefly list some projects in the Kubernetes ecosystem. If you have any additions, feel free to comment:
- [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/) Official
- [Crane NUMA Topology Awareness](https://gocrane.io/docs/tutorials/node-resource-tpolology-scheduler-plugins/)
- [Koordinator Fine-grained CPU Orchestration](https://koordinator.sh/docs/user-manuals/fine-grained-cpu-orchestration)

##### Kubernetes Topology Manager

The Topology Manager is a kubelet component designed to coordinate a set of components responsible for these optimizations. The Topology Manager addresses a historical issue where the CPU Manager and Device Manager worked independently and were unaware of each other. 
Let's first look at the implementation of the Kubernetes Topology Manager. I don't want to reinvent the wheel here, so you can refer to a [great article](https://developer.aliyun.com/article/784148) summarized by a colleague from Alibaba. Here's a summary:
1. Find the topology hints for different resources, i.e., topology information. The CPU selection criteria prioritize the smallest number of NUMA nodes involved, with the smallest number of sockets involved as a secondary priority. The device manager prioritizes the smallest number of NUMA nodes involved while meeting resource requests.
2. Merge hints from different topology types, i.e., union, to select the optimal strategy.

If a selection is made, that's good. If not, what happens? Kubernetes provides kubelet configuration strategies:
- best-effort: The Kubernetes node will accept the Pod, but the effect may not meet expectations.
- restricted: The node will refuse to accept the Pod, and if the Pod is rejected, its status will become Terminated.
- single-NUMA-node: The node will refuse to accept the Pod, and if the Pod is rejected, its status will become Terminated. This is more restrictive than restricted, as the selected NUMA node count must be 1.

Therefore, we see that the Kubernetes Topology Manager is still centered around NUMA, performing complete fair shortest path selection for different resources (NIC, GPU, CPU). Moreover, this process occurs after the Pod is scheduled to a specific node, which brings several issues:
1. There is a high probability that the Pod will be Terminated, making it unusable in production.
2. Configuring the topology-manager-policy on nodes is inconvenient, as kubelet needs to be restarted every time parameters are configured. In some special versions, this may restart all node Pods, as detailed in [this article](https://www.likakuli.com/posts/kubernetes-kubelet-restart/).

So, we think of several optimization solutions:
1. A daemon process similar to kubelet can discover topology relationships and expose them externally;
2. Topology awareness can be placed in the kube-scheduler to sense and guide scheduling when assigning nodes to Pods;
3. Provide declarative and flexible topology manager policies.

The following topology-aware solutions are based on the above ideas.

##### Crane NUMA Topology Awareness

Let's first look at the architecture diagram of Crane NUMA-aware scheduling.
{{< figure src="crane-numa-aware-arch.png" title="Crane NUMA Topology Aware (figure 5)" >}}
The general process is as follows:
1. Crane-Agent collects resource topology from nodes, including NUMA, Socket, and device information, and aggregates it into the NodeResourceTopology custom resource object.
2. Crane-Scheduler references the NodeResourceTopology object of the node to obtain detailed resource topology structure during scheduling, and allocates topology resources to the Pod while scheduling it to the node, writing the results into the Pod's annotations.
3. Crane-Agent watches the Pod being scheduled on the node, retrieves the topology allocation results from the Pod's annotations, and performs fine-grained CPUSet allocation according to the user-defined CPU binding strategy.

This essentially addresses the shortcomings of the Kubernetes Topology Manager mentioned above. However, how do we configure strategies? Here are two strategy configuration options:
1. Business side, specify strategies by labeling Pods:
    1. none: This strategy does not perform special CPUSet allocation, and the Pod will use the node's CPU shared pool.
    2. exclusive: This strategy corresponds to kubelet's static strategy, where the Pod will exclusively occupy CPU cores, and no other Pod can use them.
    3. NUMA: This strategy specifies a NUMA Node, and the Pod will use the CPU shared pool on that NUMA Node.
    4. immovable: This strategy fixes the Pod on certain CPU cores, but these cores belong to the shared pool, and other Pods can still use them.
2. Node side
    1. The default [cpu manager policy](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) is static, allowing Pods with certain resource characteristics on the node to have enhanced CPU affinity and exclusivity; the topology manager policy is SingleNUMANodePodLevel.
    2. If the node does not have the `topology.crane.io/topology-awareness` label, the topology manager policy is none.

There is a particularly notable feature here. By default, kubelet's static CPU manager strategy only applies to Pods with qos as guaranteed and resource requests as integers, and the allocated CPUs cannot be occupied by other processes. However, with the cooperation of crane-agent and Crane NUMA-aware scheduling, Pods and bound-core Pods can share resources, allowing for the benefits of fewer context switches and higher cache affinity with bound cores, while also allowing other workloads to share resources, improving resource utilization. Moreover, the requirements for Pods are relaxed, as any container with a CPU limit greater than or equal to 1 and equal to the CPU request can be set to bind cores. Let's experiment:

```shell
$ cat nginx
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  annotations:
    topology.crane.io/topology-awareness: 'true'
    topology.crane.io/cpu-policy: 'immovable'
spec:
  containers:
  - image: nginx
    imagePullPolicy: Always
    name: nginx
    resources:
      requests:
        cpu: 2
        memory: 1Gi
      limits:
        cpu: 2
        memory: 1Gi
$ k exec -it nginx /bin/bash
$ taskset -cp 1 # 查看绑核
pid 1's current affinity list: 0,1
# 查看 burstable pod 的 cpuset 信息
$ cat /sys/fs/cgroup/cpuset/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod2260198d_db73_41f0_8ae3_387e09d3b9ec.slice/cri-containerd-6a5dfa37f9ce9102e1f781160d1fecb11b17dc835e5d72b9d7f573b515af86b3.scope/cpuset.cpus
0-9

# change to exclusive 
annotations:
    topology.crane.io/topology-awareness: 'true'
    topology.crane.io/cpu-policy: 'exclusive'
$ cat /sys/fs/cgroup/cpuset/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod2260198d_db73_41f0_8ae3_387e09d3b9ec.slice/cri-containerd-6a5dfa37f9ce9102e1f781160d1fecb11b17dc835e5d72b9d7f573b515af86b3.scope/cpuset.cpus
2-9
```

As expected, for those unfamiliar with cpuset, you can refer to [Linux Cgroup Introduction: cpuset](https://zhuanlan.zhihu.com/p/121588317).

##### Koordinator Fine-grained CPU Orchestration

Koordinator and Crane have similar architectures in NUMA awareness, with koordlet replacing crane-agent and koord-scheduler replacing crane-scheduler. Even the CRD describing node topology is named the same, NRT. Here are a few different points:
1. Koordinator supports more CPU manager strategies, in addition to static, it also supports the full-pcpus-only strategy for requesting complete physical cores, and the distribute-cpus-across-NUMA strategy for even distribution when multiple NUMAs are required.
2. Koordinator supports more scheduling strategies based on NUMA topology, such as bin-packing, which prioritizes scheduling to a single node or the most idle node.
3. Additionally, Koordinator has finer granularity in PodQos and CPU manager compared to Crane, which is why it's called Fine-grained CPU Orchestration. I'll write a separate article to explain it in detail later.

## References

Standing on the shoulders of giants, thanks again.
- https://kubernetes.io/
- https://github.com/volcano-sh/volcano/
- https://mp.weixin.qq.com/s/uje27_MHBh8fMzWATusVwQ
- https://www.infoq.cn/article/tdfgiikxh9bcgknywl6s
- https://github.com/NVIDIA/go-nvml
- https://gocrane.io/docs/
- https://koordinator.sh/docs/user-manuals
- https://www.likakuli.com/posts/kubernetes-kubelet-restart/
- https://zhuanlan.zhihu.com/p/121588317
