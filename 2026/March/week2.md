### 3.9
* [cs.DB] [**Efficient Vector Search in the Wild: One Model for Multi-K Queries**](https://arxiv.org/abs/2603.06159)
  * [Vector Search & Learned Index] 现有学习型top-K检索仅针对固定K训练，无法泛化到多K查询，大K精度下降、小K性能受损，多K训练预处理成本极高。本文提出OMEGA通用K学习型检索方法，核心思想是仅在K=1上训练基础模型，结合轨迹特征，通过动态精化适配更大K，最小化性能损失支持更小K；利用top-K统计特性减少多余模型调用，降低推理开销。实验表明，同等预处理预算下，OMEGA平均延迟比SOTA学习检索低6–33%，仅用16–30%预处理时间即可达到基线最优平均延迟的1.01–1.28倍，同时满足召回目标。

* [cs.CL] [**FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling**](https://arxiv.org/abs/2603.06199)
  * [LLM Prefill & Sparse Attention] 长上下文LLM预填充阶段受注意力二次复杂度限制，稀疏注意力要么搜索延迟高，要么稀疏度不足。本文提出FlashPrefill超快预填充框架，通过瞬时块搜索同时定位动态垂直、斜线、块稀疏注意力模式；引入动态阈值机制，无需排序或累积注意力分数，有效消除长尾分布提升稀疏度。在256K序列上实现27.78倍加速，即使在4K短上下文仍保持1.71倍加速，全尺度鲁棒高效。

* [cs.LG] [**Stem: Rethinking Causal Information Flow in Sparse Attention**](https://arxiv.org/abs/2603.06274)
  * [Sparse Attention & Causal LLM] 现有稀疏注意力统一使用top-k，忽略因果架构中token信息累积依赖，早期token参与所有后续聚合却被均匀裁剪。本文提出Stem即插即用稀疏模块，从信息流视角重新设计因果稀疏注意力：采用Token位置衰减策略，每层按位置依赖动态调整top-k，保留早期token以支持递归依赖；使用输出感知指标，基于近似输出幅度优先保留高影响token。在保证精度的同时减少计算量，降低预填充延迟。

* [stat.ML] [**Semantics-Aware Caching for Concept Learning**](https://arxiv.org/abs/2603.06506)
  * [Concept Learning & Semantic Cache] 概念学习需迭代检索候选概念实例，复杂任务调用数千次导致运行时极高。本文提出语义感知缓存方法，缓存本质是包含关系感知的映射，通过清晰集合操作将概念链接到实例集。在5个数据集、4个符号推理器、1个神经符号推理器上实验，缓存将概念检索与学习运行时间降低一个数量级，同时适用于符号与神经符号系统。

### 3.10
* [cs.AI] [**LycheeCluster: Efficient Long-Context Inference with Structure-Aware Chunking and Hierarchical KV Indexing**](https://arxiv.org/abs/2603.08453)
  * [KV Cache Management & Long-Context Inference] 长上下文LLM推理面临注意力二次复杂度与KV缓存内存占用过大的双重瓶颈，现有检索类方法采用固定尺寸分块易破坏语义完整性，且线性扫描式缓存检索效率低下。本文提出LycheeCluster高效KV缓存管理方法，核心设计包括：通过边界感知分块策略保留局部语义连贯性；基于三角不等式构建递归层次索引，将原本的线性扫描检索转化为理论上有界的对数时间剪枝过程；搭配惰性更新策略，支持高效流式生成。实验结果显示，LycheeCluster实现最高3.6倍的端到端推理加速，模型性能损失可忽略不计，显著优于Quest、ClusterKV等现有SOTA KV缓存管理方法。

* [cs.CL] [**Hit-RAG: Learning to Reason with Long Contexts via Preference Alignment**](https://arxiv.org/abs/2603.07023)
  * [Long-Context RAG & Reasoning Optimization] 多模态LLM的长上下文RAG任务中，存在注意力稀释与推理幻觉两大核心问题，海量噪声信息易淹没关键证据，导致模型难以精准分辨有效片段。本文提出Hit-RAG多阶段偏好对齐框架，通过渐进式优化流水线解决认知瓶颈：第一阶段借助监督微调建立基础上下文感知能力，减少信息遗漏；第二阶段通过判别式偏好对齐增强模型对误导性干扰项的鲁棒性；第三阶段利用分组相对策略优化稳定逻辑合成过程，避免推理崩溃。在8个基准数据集上的广泛评估表明，Hit-RAG持续取得显著性能提升，助力模型弥合上下文获取与精准推理之间的差距，在长上下文场景中表现超越更大规模的同类模型。

* [cs.CL] [**DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention**](https://arxiv.org/abs/2603.08026)
  * [Diffusion LLM & Inference Acceleration] 掩码扩散语言模型（MDLMs）支持并行token解码，为自回归生成提供了可行替代方案，但迭代去噪过程需反复处理整个序列，计算开销高昂。本文发现跨扩散步骤中，多数token表示保持稳定，仅少数“显著token”对下一轮更新有实质贡献。基于这一时域稀疏性特征，提出DyLLM无训练推理框架：通过计算相邻去噪步骤间注意力上下文的余弦相似度识别显著token，仅对该部分token重新执行前馈与注意力操作，其余token直接复用缓存激活值。在多种推理与代码生成基准上，DyLLM实现最高9.6倍的吞吐量提升，同时基本保持LLaDA、Dream等SOTA模型的基线精度。

* [cs.CL] [**KohakuRAG: A simple RAG framework with hierarchical document indexing**](https://arxiv.org/abs/2603.07612)
  * [RAG System & Hierarchical Retrieval] 需高精度引用的RAG系统面临三大挑战：平坦分块策略破坏文档结构、单查询表述易因词汇不匹配遗漏相关段落、单轮推理导致答案存在随机性。本文提出KohakuRAG层次化RAG框架，核心设计包括：采用四级树形结构（文档→章节→段落→句子），通过自底向上的嵌入聚合保留文档结构；基于LLM的查询规划器结合跨查询重排序，提升检索覆盖度；集成推理搭配弃权感知投票机制，稳定答案输出。在WattBot 2025挑战赛中，该框架在公共与私有排行榜均获第一（最终得分0.861），是唯一全程保持榜首的方案。消融实验表明，提示词排序（相对提升80%）、重试机制（相对提升69%）、带空白过滤的集成投票（提升1.2个百分点）贡献显著，且仅靠层次化稠密检索即可匹配混合稀疏-稠密检索效果（BM25仅额外提升3.1个百分点），已开源。

* [cs.DS] [**Distributed Algorithms for Euclidean Clustering**](https://arxiv.org/abs/2603.08615)
  * [Distributed Clustering & Coreset Construction] 针对分布式环境下的欧氏（k,z）-聚类问题，研究（1+ε）-coresets的构建方法，数据被划分到s个节点中。本文聚焦协调者模型与黑板模型两种主流通信模式：在协调者模型中，设计的协议实现（1+ε）-强coreset，通信复杂度优化为\(\tilde{O}\left(sk + \frac{dk}{\min(\varepsilon^4,\varepsilon^{2+z})} + dk\log(n\Delta)\right)\)比特，无需跨服务器传输显式点坐标，超越此前工作；在黑板模型中，进一步将通信复杂度降至\(\tilde{O}\left(s\log(n\Delta) + dk\log(n\Delta) + \frac{dk}{\min(\varepsilon^4,\varepsilon^{2+z})}\right)\)比特，将逼近精度从常数因子升级至（1+ε）。技术方案融合常数因子近似策略、高效coreset构建与紧凑编码，达成与最优离线coreset构建及现有下界匹配的通信成本（至多多对数因子），已被ICLR 2026接收。

* [cs.DB] [**Samyama: A Unified Graph-Vector Database with In-Database Optimization, Agentic Enrichment, and Hardware Acceleration**](https://arxiv.org/abs/2603.08036)
  * [Unified Graph-Vector Database] 现代数据架构分散于图数据库、向量存储、分析引擎与优化求解器，导致ETL流程复杂且同步开销大。本文提出Samyama高性能统一图向量数据库（基于Rust实现），将多类工作负载整合至单一引擎：采用RocksDB持久化存储与版本化内存MVCC模型；配备含35种物理算子的向量化查询执行器与支持计划枚举和谓词下推的代价查询规划器；专用CSR-based分析引擎与原生RDF/SPARQL支持；内置22种元启发式优化求解器；实现HNSW向量索引与Graph RAG功能；引入智能体增强（Agentic Enrichment）机制，通过LLM自主扩展图结构。企业版额外提供基于wgpu的GPU加速、生产级可观测性、时间点恢复及HTTP/2 Raft传输的高可用方案。在Mac Mini M4（16GB RAM）上的评估显示：CPU摄入速度255K节点/秒，GPU加速后达412K节点/秒；100万节点下Cypher查询吞吐量115K/秒；多跳遍历延迟因延迟物化降低4.0-4.7倍；GPU加速使PageRank速度提升8.2倍；100%通过LDBC Graphalytics 28项测试，在保证Rust内存安全的同时实现竞品级性能。

* [cs.DB] [**Approximate Nearest Neighbor Search for Modern AI: A Projection-Augmented Graph Approach**](https://arxiv.org/abs/2603.06660)
  * [ANNS & Graph Index Optimization] 现有ANNS方案多优化查询效率，却难以满足现代AI应用的六大核心需求：高查询效率、快速索引构建、低内存占用、高维扩展性、检索规模鲁棒性及在线插入支持。本文提出投影增强图（PAG）ANNS框架，将投影技术融入图索引，通过投影基统计测试引导精确与近似距离的非对称比较，减少不必要的精确距离计算。设计三大核心组件并统一至图索引，协同优化索引构建与查询过程。在6个现代数据集上的实验表明，PAG的QPS-召回性能持续优于现有方案，较HNSW快最高5倍，同时具备快速索引构建、适中内存占用、高维与检索规模鲁棒性及原生在线插入支持等优势，源码已开源。

* [cs.DB] [**Not All Neighbors Matter: Understanding the Impact of Graph Sparsification on GNN Pipelines**](https://arxiv.org/abs/2603.06952)
  * [GNN Optimization & Graph Sparsification] 随着图规模增长至数十亿节点和边，GNN流水线受多跳遍历中邻居数量指数增长的制约，数据管理与移动成为规模化场景下的主要瓶颈。本文探索图稀疏化作为轻量级预处理步骤，在保持节点分类任务精度的同时加速GNN训练与推理。构建可扩展实验框架，首次全面评估不同稀疏化方法的影响，得出关键发现：稀疏化通常能保持甚至提升预测性能（如随机稀疏化使GAT模型在PubMed图上精度提升6.8%）；规模越大性能收益越显著，K-Neighbor稀疏化使Products图的模型服务性能提升11.7倍，精度仅下降0.7%；稀疏化的计算开销可快速摊还，适用于超大规模图场景。

* [cs.LG] [**How Attention Sinks Emerge in Large Language Models: An Interpretability Perspective**](https://arxiv.org/abs/2603.06591)
  * [LLM Interpretability & Attention Sink] 大语言模型常出现对特定token分配过多注意力的“注意力沉陷”现象，虽通常被认为有害，但研究发现模型对输入序列首个token的持续关注是例外情况，这种结构性偏差会影响多种下游应用。然而，注意力沉陷的形成与持续机制尚不明确。本文追踪输入首个token周围注意力沉陷的形成过程，识别出一种简单机制——P0沉陷回路（P0 Sink Circuit），使模型在两个Transformer块内即可识别0位置token并诱导注意力沉陷，且不依赖任何语义信息，这构成了位置0注意力沉陷的基础。通过分析30B参数A3B MoE模型的从头训练轨迹，发现该机制在训练早期出现，并逐渐集中于前两层，这一现象或可作为追踪预训练收敛状态的信号。

### 3.11-3.13

* [cs.IR] [**TA-Mem: Tool-Augmented Autonomous Memory Retrieval for LLM in Long-Term Conversational QA**](https://arxiv.org/abs/2603.09297)
  * [LLM Long-Term Memory & Conversational QA] 长时对话问答中，LLM面临记忆衰减与关键信息丢失问题，现有记忆检索依赖固定策略，无法自适应对话动态变化，且缺乏工具协同增强记忆能力。本文提出TA-Mem工具增强的自主记忆检索框架，核心设计为：集成多样化检索工具（稠密检索、关键词检索、语义聚类检索），通过LLM自主决策工具选择与参数调整；构建对话记忆图谱，动态记录实体、关系及关键表述，支持多粒度记忆检索；引入记忆刷新机制，基于对话上下文实时更新记忆权重，强化近期与关键信息的检索优先级。实验表明，TA-Mem在长时对话QA基准上，信息召回率提升32%-45%，问答准确率较基线模型提升18%-25%，有效缓解长时记忆衰减问题。

* [cs.IR] [**Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval**](https://arxiv.org/abs/2603.09250)
  * [Personalized LLM & Memory Retrieval] 个性化LLM的用户记忆利用存在两大缺陷：检索策略单一，无法区分用户对不同信息的记忆强度；缺乏对“回忆-熟悉度”双维度的感知，导致个性化响应精度不足。本文提出回忆-熟悉度自适应检索框架，核心创新为：通过用户交互轨迹分析，量化信息的回忆概率与熟悉度评分；设计双路径检索策略，高回忆概率信息采用精准匹配检索，高熟悉度信息采用模糊语义检索；引入自适应融合机制，动态平衡两类检索结果的权重。实验验证，该框架在个性化对话、用户偏好推荐任务中，响应相关性提升28%-36%，用户满意度达89%，显著优于传统个性化检索方法。

* [cs.DC] [**Flash-KMeans: Fast and Memory-Efficient Exact K-Means**](https://arxiv.org/abs/2603.09229)
  * [Distributed K-Means & Memory Optimization] 传统精确K-Means算法在大规模数据上存在内存消耗大、迭代速度慢的问题，分布式实现中数据传输开销进一步制约性能，无法满足实时聚类需求。本文提出Flash-KMeans高效分布式精确K-Means算法，核心设计包括：基于数据分块的内存优化策略，仅保留当前迭代关键数据，降低内存占用；采用局部聚类+全局合并的两阶段流程，减少跨节点数据传输；引入增量更新机制，利用前一轮聚类结果优化当前迭代中心计算，加速收敛。实验表明，Flash-KMeans在十亿级数据上，内存占用较传统算法降低60%-75%，迭代速度提升3-5倍，聚类精度无损失，适配大规模实时聚类场景。

* [cs.DC] [**Nezha: A Key-Value Separated Distributed Store with Optimized Raft Integration**](https://arxiv.org/abs/2603.09122)
  * [Distributed KV Store & Raft Consensus] 分布式键值存储中，Raft共识协议与底层存储引擎的持久化操作重叠，导致大量冗余I/O开销，严重影响读写性能。本文提出Nezha键值分离的分布式存储系统，创新性地将键值分离架构与Raft协议深度融合，核心优化为：重构操作级持久化策略，避免共识与存储层的重复持久化；引入分层垃圾回收机制，高效清理无效数据；在保留Raft安全性的前提下，大幅提升读写吞吐量。实验结果显示，Nezha的put、get、scan操作吞吐量分别平均提升460.2%、12.5%、72.6%，已被ICDE 2026主研究轨道接收。

* [cs.DC] [**Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention**](https://arxiv.org/abs/2603.08743)
  * [LLM Inference Concurrency & KV Cache Optimization] LLM推理的解码阶段，KV缓存导致的内存瓶颈严重限制高并发服务能力，现有KV缓存驱逐方法实用性不足，难以满足工业级部署需求。本文提出压缩分页注意力（Compressed PagedAttention）方法，融合token级KV缓存驱逐与PagedAttention技术，核心设计包括：全面的调度策略，支持前缀缓存与异步压缩；基于该方法开发高并发LLM推理引擎Zipage。在大规模数学推理任务中，Zipage保持全KV推理引擎约95%性能的同时，实现超2.1倍的速度提升，有效支撑高并发LLM推理服务。

* [cs.LG] [**Compiler-First State Space Duality and Portable $O(1)$ Autoregressive Caching for Inference**](https://arxiv.org/abs/2603.09555)
  * [State-Space Model Inference & Portability] 现有状态空间模型（如Mamba-2）依赖融合CUDA或Triton内核，硬依赖NVIDIA硬件，缺乏跨平台可移植性。本文提出编译器优先的状态空间对偶性实现，利用Mamba-2的对角线状态结构、可分块递归等特性，适配XLA的融合与分块优化，无需手写内核。基于JAX实现完整推理流程（预填充、缓存自回归解码），将理论O(1)状态管理转化为编译后的设备端缓存，生成阶段无需主机同步。该实现可在CPU、NVIDIA GPU、Google Cloud TPU上无修改运行，在TPU v6e上，单流预填充达约140 TFLOPS（15% MFU），解码带宽利用率最高64%，与PyTorch/CUDA参考实现结果一致。

* [cs.AR] [**Adaptive Multi-Objective Tiered Storage Configuration for KV Cache in LLM Service**](https://arxiv.org/abs/2603.08739)
  * [LLM KV Cache & Tiered Storage] 有限的GPU HBM容量推动KV缓存卸载至廉价外部存储层，但异构存储资源的动态管理面临成本、吞吐量、延迟的多目标平衡挑战，且目标函数非解析、变量耦合复杂。本文提出Kareto KV缓存自适应资源管理优化器，核心创新为：基于边际收益递减的剪枝方法，高效逼近帕累托前沿；结合细粒度自适应调谐器，利用分层存储驱逐策略与KV块访问模式进行分组缓存管理。真实轨迹实验表明，Kareto可提升吞吐量最高9.3%，降低延迟最高58.3%，或减少成本最高20.2%，显著优于静态策略。

* [cs.AR] [**ARKV: Adaptive and Resource-Efficient KV Cache Management under Limited Memory Budget for Long-Context Inference in LLMs**](https://arxiv.org/abs/2603.08727)
  * [Long-Context LLM & KV Cache Management] 长上下文LLM推理中，KV缓存随序列长度与批量大小线性增长，快速占用GPU内存，现有内存缩减技术依赖静态启发式，在紧预算下性能退化严重。本文提出ARKV轻量级自适应框架，核心设计为：基于每层注意力动态与token重要性，为缓存token动态分配精度等级；预填充阶段通过注意力熵、方差等统计分数估计每层原始量化比；解码阶段采用快速重击者评分策略，将token分配至全精度、低精度或驱逐状态。在LLaMA3、Qwen3模型上的实验表明，ARKV保留约97%基线准确率，KV内存占用减少4倍，吞吐量损失极小，已被ACM/IEEE CCGRID 2025接收。

* [cs.DB] [**LHGstore: An In-Memory Learned Graph Storage for Fast Updates and Analytics**](https://arxiv.org/abs/2603.11596)
  * [In-Memory Graph Storage & Learned Indexing] 内存动态图需高效处理频繁更新与低延迟分析，但更新效率与遍历局部性存在权衡，尤其在高度倾斜度分布下挑战突出。本文提出LHGstore度感知的分层学习型图存储系统，首次将学习型索引融入图管理，核心设计为：两级层次结构解耦顶点与边访问；为低度数顶点采用轻量级数组最大化遍历局部性，为高度数顶点应用学习型索引提升更新吞吐量。实验表明，LHGstore的吞吐量较SOTA内存图存储系统提升5.9-28.2倍，分析速度显著加快，已被DAC 2026接收。

* [cs.DB] [**How to Write to SSDs**](https://arxiv.org/abs/2603.09927)
  * [SSD Storage Optimization & DBMS] 数据库系统未充分利用SSD性能，原地写入导致写放大严重，影响SSD寿命与性能。本文证明异地写入是数据库系统充分发挥SSD性能、延长寿命的关键，提出一系列异地写入优化，重构基于B树的LeanStore为异地写入架构。在多样化OLTP基准、数据集大小与SSD上的评估显示，YCSB-A基准吞吐量提升1.65-2.24倍，事务闪存写入减少6.2-9.8倍；TPC-C（15000仓库）吞吐量提升2.45倍，闪存写入减少7.2倍，且可无缝支持ZNS、FDP等新型SSD接口，已被PVLDB 2026接收。

* [cs.LG] [**LongFlow: Efficient KV Cache Compression for Reasoning Models**](https://arxiv.org/abs/2603.11504)
  * [Reasoning LLM & KV Cache Compression] 推理模型（如OpenAI-o1、DeepSeek-R1）的长输出序列导致KV缓存内存消耗大、带宽压力突出，现有KV缓存优化方法适配长输入短输出场景，且重要性估计计算昂贵。本文提出LongFlow KV缓存压缩方法，核心创新为：基于当前查询的注意力计算中间结果，设计高效重要性估计指标，无额外计算开销与辅助存储；融合FlashAttention、重要性估计与token驱逐的定制内核，提升系统效率。实验表明，LongFlow在80% KV缓存压缩下，吞吐量提升最高11.8倍，模型精度影响极小。

* [cs.LG] [**Meta-Reinforcement Learning with Self-Reflection for Agentic Search**](https://arxiv.org/abs/2603.11327)
  * [Agentic Search & Meta-RL] 智能体搜索依赖单轮稀疏奖励优化策略，泛化能力不足。本文提出MR-Search上下文元强化学习框架，核心设计为：跨轮次探索，每轮后生成显式自反思并作为上下文指导后续尝试；多轮RL算法估计轮次级稠密相对优势，实现细粒度信用分配。在多个基准上，MR-Search较RL基线泛化能力更强，相对提升9.2%-19.3%，已开源代码与数据。

* [cs.CL] [**IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse**](https://arxiv.org/abs/2603.12201)
  * [Sparse Attention & LLM Inference] 长上下文智能体工作流中，稀疏注意力（如DSA）的索引器存在O(L²)复杂度，且每层独立运行，虽顶层k选择高度相似但冗余计算严重。本文提出IndexCache，利用跨层冗余将层划分为少量全索引层（运行索引器）与多数共享层（复用最近全索引层的顶层k索引）。设计无训练与训练感知两种方案：无训练方案通过贪心搜索选择保留索引器的层，最小化语言建模损失；训练感知方案引入多层蒸馏损失，使保留的索引器适配所有服务层。在30B DSA模型上，IndexCache移除75%索引器计算，预填充速度提升最高1.82倍，解码速度提升最高1.48倍，质量损失可忽略。

* [cs.LG] [**Leech Lattice Vector Quantization for Efficient LLM Compression**](https://arxiv.org/abs/2603.11021)
  * [LLM Quantization & Vector Quantization] 标量量化受信息论边界限制，向量量化（VQ）虽通过联合编码突破限制，但需避免昂贵查找与显式码本存储。本文探索24维最优球填充与亲吻配置的Leech格，扩展基于扩展戈莱码构造的搜索算法，支持索引、角搜索与全并行解量化内核，提出Leech格向量量化（LLVQ）。LLVQ无需物化码本，性能超越Quip#、QTIP、PVQ等近期方法，为大规模理论接地的模型压缩提供高维格解决方案。

* [cs.LG] [**LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation**](https://arxiv.org/abs/2603.10899)
  * [LLM KV Cache Eviction] 长上下文任务中，KV缓存驱逐依赖重要性估计，现有“前瞻未来”方法需生成草稿响应，计算开销大、预填充延迟高。本文提出LookaheadKV轻量级驱逐框架，核心设计为：增强Transformer层的参数高效模块，精准预测真实重要性分数；无额外运行时开销，精度优于昂贵近似方法。在多种模型与长上下文理解基准上，LookaheadKV性能超越近期竞品，驱逐成本降低最高14.5倍，首token时间显著加快，已被ICLR 2026接收。

* [cs.DS] [**Sample-and-Search: An Effective Algorithm for Learning-Augmented k-Median Clustering in High dimensions**](https://arxiv.org/abs/2603.10721)
  * [Learning-Augmented Clustering & High-Dimensional Data] 学习增强k-中值聚类通过预测器预处理数据分配潜在标签，但现有算法时间复杂度高，且依赖维度指数项。本文提出基于采样的Sample-and-Search算法，显著优化时间复杂度，缓解维度依赖。实验表明，该方法在实践中大幅降低计算复杂度，同时实现更低的聚类成本，优于现有SOTA学习增强k-中值聚类方法。

* [cs.LG] [**Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE**](https://arxiv.org/abs/2603.11611)
  * [RoPE & Positional Encoding] 旋转位置编码（RoPE）广泛应用于Transformer，但隐藏维度中接受旋转变换的比例影响未被充分探索。本文系统研究部分RoPE的影响，发现关键结论：仅10%左右维度应用RoPE即可实现与全RoPE相当的收敛效果；该趋势在不同模型大小、序列长度、数据集质量与架构中一致，高质量数据损失更低、基准性能相近；无位置编码（NoPE）模型学习轨迹不稳定，可通过少量RoPE或QK-Norm缓解。部分RoPE可实现最高10倍内存节省，为平衡效率与训练稳定性提供实用指导。

* [cs.AR] [**Multi-Agent Memory from a Computer Architecture Perspective: Visions and Challenges Ahead**](https://arxiv.org/abs/2603.10062)
  * [Multi-Agent Memory & Computer Architecture] LLM智能体向协作多智能体系统演进，内存需求复杂度剧增。本文从计算机体系结构视角框架化多智能体内存问题，区分共享与分布式内存范式，提出三层内存层次结构（I/O、缓存、内存），识别智能体间缓存共享与结构化内存访问控制两大协议缺口，指出多智能体内存一致性是最紧迫的开放挑战，为构建可靠可扩展多智能体系统提供基础架构视角。

* [cs.AI] [**AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents**](https://arxiv.org/abs/2603.09716)
  * [Adaptive Agent & Memory Orchestration] 自主智能体框架难以平衡长时经验学习与实时上下文敏感决策，存在认知静态、工作流僵化、上下文利用低效等问题。本文提出AutoAgent自进化多智能体框架，核心组件包括：进化认知（维护工具、能力、专业知识与任务知识的结构化提示级认知）、实时上下文决策（结合认知与任务上下文选择统一动作空间）、弹性内存编排（动态组织交互历史，减少token开销）；通过闭环认知进化持续更新认知与技能，无需外部重训练。在检索增强推理、工具增强智能体基准与具身任务中，AutoAgent持续提升任务成功率、工具使用效率与协作鲁棒性。

* [cs.AI] [**Context Engineering: From Prompts to Corporate Multi-Agent Architecture**](https://arxiv.org/abs/2603.09619)
  * [Multi-Agent Architecture & Context Engineering] 提示工程（PE）不足以支撑自主多步智能体，本文提出上下文工程（CE）作为独立学科，关注智能体决策的信息环境设计、结构化与管理。基于厂商架构、学术研究与企业实践，提出相关性、充分性、隔离性、经济性、溯源性五大上下文质量标准，将上下文视为智能体的操作系统；衍生出意图工程（IE，编码组织目标与权衡层级）与规范工程（SE，构建企业政策机器可读语料），形成四级智能体工程成熟度模型。企业数据显示75%企业计划两年内部署智能体AI，但面临规模化复杂性挑战，该框架为企业级多智能体系统部署提供理论基础。

* [cs.AI] [**MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games**](https://arxiv.org/abs/2603.09022)
  * [Multi-Agent LLM Games & Context Optimization] 多轮多智能体LLM游戏评估存在显著运行方差，长视野交互中早期偏差累积放大，提示选择加剧不稳定性与性能不足。本文提出MEMO记忆增强模型上下文优化框架，核心设计为：保留自玩轨迹结构化洞察的持久内存库，作为后续游戏先验；基于TrueSkill的锦标赛式提示进化与优先级回放，重访稀有关键状态。在五个文本游戏中，MEMO将GPT-4o-mini平均胜率从25.1%提升至49.5%，Qwen-2.5-7B-Instruct从20.9%提升至44.3%，运行方差降低，在谈判与不完美信息游戏中收益显著。

* [cs.LG] [**MSSR: Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning**](https://arxiv.org/abs/2603.09892)
  * [Continual LLM Fine-Tuning & Catastrophic Forgetting] 动态环境中LLM持续微调需适应任务与数据分布变化，但易发生灾难性遗忘，现有重放策略依赖启发式或计算开销大。本文提出MSSR记忆启发式采样与调度重放框架，核心创新为：估计样本级记忆强度，自适应间隔调度重放，平衡遗忘缓解与快速适应。在三个骨干模型与11个序列任务上，MSSR持续优于SOTA重放基线，在推理密集型与多项选择基准上收益尤为显著。

* [cs.OS] [**The Missing Memory Hierarchy: Demand Paging for LLM Context Windows**](https://arxiv.org/abs/2603.09023)
  * [LLM Context Management & Memory Hierarchy] LLM上下文窗口被视为L1缓存（小、快、昂贵），缺乏L2、虚拟内存与分页机制，工具定义、系统提示等静态内容长期占用上下文，导致21.8%结构性浪费。本文提出Pichay LLM上下文窗口需求分页系统，作为客户端与推理API的透明代理，核心功能为：驱逐过期内容、检测模型重请求时的页错误、基于错误历史固定工作集页面。离线重放140万模拟驱逐的错误率仅0.0254%；生产部署中上下文消耗降低最高93%，极端压力下虽出现抖动但保持运行，为LLM系统构建完整内存层次结构（L1至持久存储）提供实践基础。