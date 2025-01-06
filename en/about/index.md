# About xieydd


📫 If you wish to contact me, you can send an email to xieydd@gmail.com, or add my WeChat `echo -n 'eGlleWRkX2hhaGEK' | base64 -d`.

💻

As of 2024, I have over 6 years of experience in AI Infrastructure:

### 2018-2021.2 (including internship) [Unisound](https://www.unisound.com/)

1. At the AI algorithm company [Unisound](https://www.unisound.com/), I was responsible for the development and operation of the Atlas supercomputing platform, supporting NLP and CV model training. Key responsibilities included:
  - Developing a large-scale intelligent scheduling system to optimize multi-tenant resource allocation
  - Enhancing the performance of the high-performance distributed file system Lustre
  - Building a multi-layer cache cloud-native architecture to accelerate AI model training

2. Worked on 8 Bit training and inference optimization at Unisound, optimizing models for NPU and NVIDIA Edge Devices.

### 2021.2-2023.5 Tencent Cloud

1. Developed a large-scale AI platform for public cloud:
  - Built a high-performance, scalable elastic offline training platform using EKS (Elastic Kubernetes Service).
  - Integrated public cloud object storage and the GooseFS accelerator to create a high-performance cache scheduling system on the cloud

2. Established FinOps infrastructure to help public cloud customers manage and optimize cloud costs more effectively, enhancing cloud resource utilization:
  - Optimized scheduling and rescheduling, identified high and low priority tasks, and implemented intelligent elastic scaling.
  - Combined Tencent's Ruyi kernel scheduler optimization and observability to optimize costs while maintaining service quality
  - Launched a large-scale cost reduction initiative in the internal cloud, improving resource utilization through efficient resource allocation

### 2023.5-present [Tensorchord](https://tensorchord.ai)

1. Leading the development of the Serverless Inference platform [ModelZ](https://modelz.ai/) on GCP, providing optimized cold start model service inference:
  - Reduced model service cold start time through cache model services and image preheating
  - Implemented JuiceFS to build a high-performance cache scheduling system, enhancing model service performance

2. Leading the Cloud Team, developing the vector database VectorChord's cloud service and customer support [VectorChord Cloud](https://vectorchord.ai):
  - Built a vector database based on Postgres on AWS, achieving control and data plane separation, BYOC (Bring Your Own Cloud), BYOD (Bring Your Own Data) capabilities
  - Implemented cloud-native architecture to achieve Postgres storage and compute separation, high availability, Backup, PITR (Point-In-Time Recovery), In-Place Upgrade features

Skill set: Kubernetes, GCP, AWS, Kubeflow, FinOps, RAG, Vector Database, Storage Acceleration, Tensorflow, Pytorch, Cloud Native, MLOps, AI Infrastructure, etc.

🌱 Currently focusing on MLOps and FinOps, contributing to several open source projects:
1. [fluid](https://github.com/fluid-cloudnative/fluid) Fluid, elastic data abstraction and acceleration for BigData/AI applications in the cloud. (Project under CNCF)
2. [crane](https://github.com/gocrane/crane) Crane is a FinOps Platform for Cloud Resource Analytics and Economics in Kubernetes clusters. The goal is to help users manage cloud costs more easily while ensuring application quality.
3. [crane-scheduler](https://github.com/gocrane/crane-scheduler) Crane scheduler is a Kubernetes scheduler that can schedule pods based on actual node load.
4. [creator](https://github.com/gocrane/creator) Creator is the brain of the crane project, containing the core algorithm module and evaluation module.
5. [openmodelz](https://github.com/tensorchord/openmodelz) One-click machine learning deployment (LLM, text-to-image, etc.) at scale on any cluster (GCP, AWS, Lambda labs, your home lab, or even a single machine).
6. [clusternet](https://github.com/clusternet/clusternet) [CNCF Sandbox Project] Managing your Kubernetes clusters (including public, private, edge, etc.) as easily as browsing the Internet
7. [vectorchord](https://github.com/tensorchord/VectorChord) Scalable, fast, and disk-friendly vector search in Postgres, the successor of pgvecto.rs.
