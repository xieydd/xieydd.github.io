# 关于远东


截止到 2024 年，有着超过 6 年的 AI Infra 的经验，具体如下：
1. 2018-2021.2（含实习）在 AI 算法公司 [Unisound](https://www.unisound.com/) 负责 Atlas 超算平台的研发和运维，支持 NLP 以及 CV 模型训练。主要工作如下：
- 构建大规模智能调度系统
- 高性能分布式文件存储 Lustre 的优化
- 构建多级缓存云原生系统加速 AI 训练

2. 在 Unisound 从事 8 Bit 训练及推理优化工作，落地模型在 NPU 以及 NVIDIA Edge Device 的优化工作。

3. 2021.2-2023.5 在 Tencent Cloud 构建公有云大规模 AI 平台， 具体工作如下：
- 通过 EKS （Elastic Kubernetes Service） 构建高性能、可伸缩的弹性离线训练平台。
- 结合公有云对象存储以及加速器 GooseFS, 构建云上高性能缓存调度系统

4. 构建 FinOps 基础设施帮助公共云中的客户更轻松地管理优化云成本，提升云资源利用率，主要工作如下：
- 通过调度时以及重调度优化，通过高低优任务识别，智能弹性扩缩容。于此同时截止腾讯如意内核以及可观测性，确保服务质量的同时进行成本优化
- 在内部云中大规模推出降低成本计划，通过资源的合理分配，提升资源利用率

5. 2023.5-至今在初创公司 [Tensorchord](https://vectorchord.ai) 负责在 GCP 上构建 Serverless Inference 平台 [ModelZ](https://modelz.ai/)， 提供极致的冷启动优化模型服务推理服务。
- 通过构建缓存模型服务、镜像预热等手段优化模型服务的冷启动时间
- 引入 JuiceFS 构建高性能缓存调度系统，提升模型服务的性能

6. 负责整个向量数据库 VectorChord 的云服务研发和客户支持 [VectorChord Cloud](https://cloud.vectorchord.ai)。
- 在 AWS 上构建基于 Postgres 的向量数据库，实现控制面、数据面分离，BYOC(Bring Your Own Cloud)、BYOD(Bring Your Own Data) 等功能
- 引入云原生架构，实现 Postgres 存算分离、高可用、Backup、PITR(Point-In-Time Recovery)、In-Place Upgrade 等功能

技能栈：Kubernetes, GCP, AWS, Kubeflow, FinOps, RAG, Vector Database, Storage Accelerate, Tensorflow , Pytorch, Cloud Native, MLOps, AI Infra, etc.

🌱 目前专注在 MLOps 以及 FinOps 领域,贡献了一些开源项目：
1. [fluid](https://github.com/fluid-cloudnative/fluid) Fluid, elastic data abstraction and acceleration for BigData/AI applications in cloud. (Project under CNCF)
2. [crane](https://github.com/gocrane/crane) Crane is a FinOps Platform for Cloud Resource Analytics and Economics in Kubernetes clusters. The goal is not only to help users to manage cloud cost easier but also ensure the quality of applications.
3. [crane-scheduler](https://github.com/gocrane/crane-scheduler) Crane scheduler is a Kubernetes scheduler which can schedule pod based on actual node load.
4. [creator](https://github.com/gocrane/creator) Creator is the brain of crane project, contains crane core algorithm module and evaluation module.
5. [openmodelz](https://github.com/tensorchord/openmodelz) One-click machine learning deployment (LLM, text-to-image and so on) at scale on any cluster (GCP, AWS, Lambda labs, your home lab, or even a single machine).
6. [clusternet](https://github.com/clusternet/clusternet) [CNCF Sandbox Project] Managing your Kubernetes clusters (including public, private, edge, etc.) as easily as visiting the Internet
7. [vectorchord](https://github.com/tensorchord/VectorChord) Scalable, fast, and disk-friendly vector search in Postgres, the successor of pgvecto.rs.


📫  您如果想联系我，可以直接发送邮件，邮箱地址 xieydd@gmail.com,或者您可以加我的微信 `echo -n 'eGlleWRkX2hhaGEK' | base64 -d`.


