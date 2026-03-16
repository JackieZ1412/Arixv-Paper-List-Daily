## March Week 1

### 3.2
* [cs.IR] [**FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation**](https://arxiv.org/abs/2602.23671)
  * [Sequential Recommendation & Linear Attention] 主流推荐系统的二次复杂度注意力机制难以处理长用户序列，而线性注意力虽为理想替代方案，但现有研究存在三大缺陷：时间信号与语义信号耦合干扰且忽略行为周期性、位置信息表征不足、仅聚焦短序列与浅层架构。本文提出FuXi-Linear面向长序列推荐的线性复杂度模型，核心设计两大专属通道：**时间保留通道**，独立基于时间数据计算周期性注意力权重，彻底避免时序与语义信号的串扰；**线性位置通道**，通过可学习核在线性复杂度内实现位置信息的有效融合。同时验证了该模型在千级序列尺度下具备鲁棒的幂律缩放特性，这一特性在现有线性推荐研究中极少被探索。在数千token的长序列数据集上的实验表明，FuXi-Linear在推荐质量上超越SOTA模型，预填充阶段推理速度较竞品基线提升最高10×，解码阶段提升最高21×，且已开源代码。

* [cs.IR] [**Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA**](https://arxiv.org/abs/2602.23372)
  * [GraphRAG & Multi-Hop QA] 现有GraphRAG系统依赖昂贵的LLM构建图谱且推理过程需GPU支撑，成本高且门槛高，难以规模化落地。本文提出SPRIG面向多跳问答的纯CPU、线性时间、无token开销的GraphRAG流水线，核心用轻量级命名实体识别（NER）驱动的共现图谱替代LLM图谱构建，采用个性化页面排名（PPR）完成图谱检索，在Recall@10基本无损失的前提下，大幅降低计算成本。实验结果明确了纯CPU友好型图谱检索对多跳召回的有效适用场景，也界定了强词汇混合方法（RRF）即可满足需求的场景，为摆脱token成本与GPU依赖、实现GraphRAG的平民化落地提供了切实可行的技术路径。

* [cs.DB] [**GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search**](https://arxiv.org/abs/2602.23999)
  * [GPU-native ANNS & IVF Optimization] GPU端近似最近邻搜索中，图基索引召回率和吞吐量高但构建与存储成本高昂，聚类基索引构建和扩展高效却需大量探针才能保证高召回，进而挤占内存带宽与计算资源。本文提出IVF-RaBitQ纯GPU原生的ANNS解决方案，将聚类基的IVF与RaBitQ量化融合为高效的GPU索引构建/检索流水线，兼顾快速建索引、高吞吐检索、高召回率与低存储需求。索引构建阶段，设计可扩展的GPU原生RaBitQ量化方法，实现规模化的快速高精度低位编码；检索阶段，开发RaBitQ编码的GPU原生距离计算方案与融合检索核，在高召回下实现高吞吐。该方案已集成至NVIDIA cuVS库，在cuVS Bench多数据集上的实验表明，IVF-RaBitQ在召回率、吞吐量、索引构建时间和存储占用上均表现出优异的性能边界；在召回率≈0.95时，查询吞吐量较SOTA图基方法CAGRA提升2.2×，索引构建速度平均快7.7×；相较聚类基方法IVF-PQ，平均吞吐量提升超2.7×，且无需访问原始向量做重排序。

### 3.3
* [cs.OS] [**Token Management in Multi-Tenant AI Inference Platforms**](https://arxiv.org/abs/2603.00356)
  * [Multi-Tenant LLM Inference & Resource Management] 多租户AI推理平台难以平衡资源利用率与服务等级保障，传统方案存在明显缺陷：专属端点在模型空闲时浪费资源，速率限制忽略推理请求的异构成本，无法应对动态需求。本文提出“令牌池（token pools）”控制平面抽象，将推理容量表示为推理原生单位（令牌吞吐量、KV缓存、并发数）的显式权限。与速率限制不同，令牌池从同一容量模型授权请求准入与自动扩缩容，确保承诺与配置一致；支持多维度突发模式控制，通过动态权限限制实现资源消耗细粒度管控，同时允许低优先级流量的空闲资源回填。该设计无需修改底层推理运行时或集群调度器，即可支持优先级感知分配、差异化服务等级与债务公平机制。在Kubernetes集群与vLLM后端的实验表明，过载时令牌池通过选择性限流临时流量，为保障型工作负载维持有界P99延迟（无准入控制的基线会导致所有工作负载延迟无界退化）；容量稀缺时，弹性工作负载可通过债务公平机制收敛至公平共享状态。

* [cs.AR] [**HAVEN: High-Bandwidth Flash Augmented Vector Engine for Large-Scale Approximate Nearest-Neighbor Search Acceleration**](https://arxiv.org/abs/2603.01175)
  * [ANNS Hardware Acceleration & HBF Integration] RAG依赖大规模ANNS为LLM检索语义相关上下文，IVF-PQ虽平衡内存效率与检索精度，但高召回需重排序（需读取全精度向量）；十亿级向量数据库因GPU HBM容量有限，需存储于CPU DRAM或SSD，跨设备数据迁移导致延迟与吞吐量大幅下降。本文提出HAVEN高带宽闪存增强的GPU架构，集成新兴堆叠3D NAND技术的高带宽闪存（HBF）——具备TB级容量与数百GB/s读取带宽，通过片上HBF与近存储搜索单元补充HBM，使全精度向量数据库可完全驻留于设备端，消除重排序时的PCIe与DDR瓶颈。通过详细建模重构的3D NAND子阵列、功耗约束下的HBF带宽及端到端IVF-PQ流水线，实验表明HAVEN在十亿级数据集上，重排序吞吐量提升最高20×，延迟降低最高40×，较GPU-DRAM与GPU-SSD系统优势显著，使高召回检索能达到此前仅无重排序场景可实现的吞吐量，为内存中心型AI加速器提供新方向。

* [cs.DB] [**Disk-Resident Graph ANN Search: An Experimental Evaluation**](https://arxiv.org/abs/2603.01779)
  * [Disk-based ANNS & Performance Analysis] 数据量增长与内存容量限制推动磁盘驻留图基ANNS成为内存驻留方案的实用替代，但此类系统在存储、布局、执行范式上差异显著，其核心性能权衡缺乏系统性认知。本文对磁盘驻留图基ANNS方法开展全面实验研究：首先将系统拆解为存储策略、磁盘布局、缓存管理、查询执行、更新机制五大技术组件，构建现有设计的统一分类体系；其次对各组件的代表性策略进行细粒度评估，分析吞吐量、召回率与资源利用率的权衡；再通过端到端实验与参数敏感性分析，评估不同配置下的整体系统性能；最后揭示四大非直观发现：向量维度从根本上影响组件有效性，需维度感知设计；现有布局策略I/O利用率极低（≤15%）；页面大小对可行性与效率至关重要，优化布局下小页面更优；更新策略的原地/异地设计存在工作负载依赖的显著权衡。基于这些发现，提炼系统设计与配置的实用准则，并指出未来研究方向。

* [cs.DB] [**VectorMaton: Efficient Vector Search with Pattern Constraints via an Enhanced Suffix Automaton**](https://arxiv.org/abs/2603.01525)
  * [Constrained ANNS & Pattern Filtering] 向量数据库中ANNS是核心操作，但实际应用中向量常伴随序列等辅助信息，需支持辅助数据约束的细粒度检索；现有方法仅支持属性或范围过滤，无法处理序列属性的模式谓词。随着向量数据库采用SQL风格查询接口，支持序列属性（文本、生物序列）的模式谓词（如LIKE、CONTAINS）与向量相似度搜索结合成为刚需。本文定义新问题：给定带序列关联的向量集合，检索序列包含查询模式的最近邻向量。为此提出VectorMaton自动机基索引，将模式过滤与高效向量搜索集成，索引大小与数据集相当。核心设计为增强后缀自动机，实现模式约束的快速过滤与向量检索的协同优化。在真实数据集上的实验表明，VectorMaton在相同精度下查询吞吐量较所有基线提升最高10×，索引大小减少最高18×，高效支持带模式约束的向量搜索。

* [cs.DB] [**SEAnet: A Deep Learning Architecture for Data Series Similarity Search**](https://arxiv.org/abs/2603.01448)
  * [Data Series Similarity Search & Deep Embedding] 海量数据序列分析的核心操作是相似度搜索，SAX基索引虽为当前SOTA，但在高频、弱相关、高噪声等特定数据集上性能不佳。本文提出深度嵌入近似（DEA）数据序列摘要技术家族，基于深度学习构建；设计专为DEA优化的SEAnet架构，将平方和保留特性融入深度网络设计，并通过SEAtrans编码器增强；提出SEAsam与SEAsamE采样策略，使SEAnet能高效训练于海量数据集。该架构通过深度学习学习数据序列的高质量嵌入表示，提升相似度搜索性能。在7个多样化合成与真实数据集上的全面实验验证，SEAnet学习的DEA能提供高质量数据序列摘要，相似度搜索结果优于现有方案，已发表于IEEE Transactions on Knowledge and Data Engineering（2023年12月）。

* [cs.CL] [**KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging**](https://arxiv.org/abs/2603.00907)
  * [LLM KV Cache & Asymmetric Merging] LLM的KV缓存对计算与内存需求日益增长，KV合并虽为有效解决方案，但现有方法依赖KV不对称的经验观察与梯度基海森近似，缺乏理论基础，压缩与推理开销非最优。本文建立理论框架，通过投影权重的谱能量分布表征KV不对称性：查询/键（Q/K）权重的集中谱诱导特征同质性，值（V）权重的分散谱保留特征异质性。基于此提出KVSlimmer高效算法，通过数学精确公式捕捉完整海森信息，仅利用前向传播变量推导闭式解，实现无梯度、内存与时间高效的KV合并。在多种模型与基准上的实验表明，KVSlimmer持续优于SOTA方法；以Llama3.1-8B-Instruct为例，LongBench平均得分提升0.92，内存成本降低29%，延迟减少28%。

* [cs.RO] [**KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning**](https://arxiv.org/abs/2602.23592)
  * [Embodied Planning & KV Cache Management] 内存增强型LLM在复杂长视野具身规划中表现优异，但现有方法常以原始文本存储内存，导致提示过长、预填充延迟高；虽可存储复用KV缓存，但频繁更新严重削弱效率优势（该文已被DAC 2026接收）。本文提出KEEP KV缓存中心的内存管理系统，三大核心创新：静态-动态内存构建算法，通过混合粒度内存组减少KV缓存重计算；多跳内存重计算算法，动态识别不同内存组间的重要交叉注意力，迭代重构内存交互；层平衡内存加载，消除不同层间KV缓存加载与交叉注意力计算的不平衡。在ALFRED数据集上的实验表明，KEEP较文本基内存方法实现2.68×加速，精度损失可忽略；较KV重计算方法CacheBlend（EuroSys'25），成功率提升4.13%，首令牌时间（TTFT）减少1.90×，已开源代码。

### 3.4
* [cs.DB] [**Virtual-Memory Assisted Buffer Management In Tiered Memory**](https://arxiv.org/abs/2603.03271)
  * [Tiered Memory & Buffer Management] 分层内存架构在数据库领域应用日益广泛，该架构以主机处理器片上DRAM为本地内存（主层），字节可寻址、缓存一致的额外内存资源（如远程NUMA内存、小芯片附加内存、RDMA/CXL互联内存）为远程内存（RMem，次层），RMem速度介于本地DRAM与磁盘之间。传统DRAM-磁盘双层虚拟内存辅助缓冲管理技术难以直接扩展至n层（DRAM-RMem-磁盘）场景，且页面迁移易成为性能瓶颈。本文提出vmcacheⁿ n层虚拟内存辅助缓冲池，利用虚拟内存子系统与操作系统调用实现跨内存层的页面迁移；为解决迁移瓶颈，引入move_pages2系统调用，赋予vmcacheⁿ对页面迁移过程的细粒度控制。实验表明，在TPC-C工作负载下，vmcacheⁿ的查询吞吐量较vmcache提升最高4×，有效适配分层内存架构的缓冲管理需求。

* [cs.DB] [**V3DB: Audit-on-Demand Zero-Knowledge Proofs for Verifiable Vector Search over Committed Snapshots**](https://arxiv.org/abs/2603.03065)
  * [Verifiable ANNS & Zero-Knowledge Proof] 稠密检索服务是语义搜索、推荐、检索增强生成的核心，但客户端仅能获取top-k结果，缺乏结果生成的可审计证据，服务提供商的不可信性可能导致结果失真。本文提出V3DB可验证、带版本的向量搜索服务，支持对不可信服务提供商执行的近似最近邻检索进行按需审计正确性校验。V3DB为每个语料快照生成承诺，将IVF-PQ搜索流水线标准化为固定流程的五步查询语义；给定公共快照承诺与查询嵌入，服务返回top-k结果，被质疑时生成简洁零知识证明，验证结果确为基于承诺快照执行公开语义的产物，且不泄露嵌入语料与私有索引内容。为提升证明实用性，V3DB通过多集合相等/包含检查与轻量级边界条件结合，避免电路内昂贵的排序与随机访问。基于Plonky2的原型实现表明，其证明速度较纯电路基线提升最高22×，峰值内存消耗降低40%，验证时间仅需毫秒级，已开源代码。

* [cs.DC] [**Token Management in Multi-Tenant AI Inference Platforms**](https://arxiv.org/abs/2603.00356)
  * [Multi-Tenant LLM Inference & Resource Scheduling] 多租户AI推理平台需在动态需求下平衡资源利用率与服务等级保障，传统方案存在明显缺陷：专属端点导致空闲模型浪费资源，速率限制忽略推理请求的异构成本，无法实现精准管控。本文提出“令牌池（token pools）”控制平面抽象，将推理容量表示为推理原生单位（令牌吞吐量、KV缓存、并发数）的显式权限。与速率限制不同，令牌池从同一容量模型授权请求准入与自动扩缩容，确保承诺与配置的一致性；支持多维度突发模式捕捉，通过动态权限限制实现资源消耗细粒度管控，同时允许低优先级流量回填空闲资源。该设计无需修改底层推理运行时或集群调度器，即可支持优先级感知分配、差异化服务等级与债务公平机制。在Kubernetes集群与vLLM后端的实验表明，过载时令牌池通过选择性限流临时流量，为保障型工作负载维持有界P99延迟（无准入控制的基线会导致所有工作负载延迟无界退化）；容量稀缺时，弹性工作负载可通过债务公平机制收敛至公平共享状态。

* [cs.IR] [**APAO: Adaptive Prefix-Aware Optimization for Generative Recommendation**](https://arxiv.org/abs/2603.02730)
  * [Generative Recommendation & Training-Inference Alignment] 生成式推荐作为序列推荐的新兴范式，将任务建模为自回归生成过程，基于用户交互历史预测下一个物品的离散令牌；现有模型通常采用令牌级似然目标（如交叉熵损失）训练，推理时通过多步束搜索生成排序物品候选，但存在严重的训练-推理不一致问题：训练假设总能获得真实历史，忽略束搜索在推理时会剪枝低概率分支，导致正确物品可能因初始令牌（前缀）得分低而被过早丢弃。本文提出自适应前缀感知优化（APAO）框架，引入前缀级优化损失，使训练目标与推理场景更好对齐；设计自适应最差前缀优化策略，训练时动态聚焦最脆弱的前缀，增强模型在束搜索约束下保留正确候选的能力。作者提供理论分析验证框架的有效性与效率，在多个数据集上的实验表明，APAO持续缓解训练-推理不一致问题，提升各类生成式推荐骨干模型的性能，已开源代码。