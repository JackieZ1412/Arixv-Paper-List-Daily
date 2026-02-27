### 2.23
* [cs.DB] [**Efficient Filtered-ANN via Learning-based Query Planning**](https://arxiv.org/pdf/2602.17914)
  * [ANNS & Query Optimization] 过滤式近似最近邻搜索中，预过滤需为不同谓词构建专属索引，计算成本高；后过滤易因选择性低导致召回损失且存在冗余计算，单一执行策略无法适配所有查询场景。针对该问题，本文提出基于学习的查询规划框架，核心设计为两步式决策流程：先通过轻量级选择性估计器，结合预计算统计信息分别对分类、数值范围及混合谓词做精准选择性预测；再训练双层MLP分类器作为核心规划器，依据数据集统计特征和估计选择性，为每个查询动态选择预过滤或后过滤策略。该框架兼容各类ANN索引与过滤类型，无需GPU且推理开销极低。实验结果表明，在ArXiv、SIFT等数据集上，框架召回率保持≥90%的同时实现最高4×加速，索引构建成本较ACORN-1降低20.24×，显著优于传统固定策略与现有过滤式ANN方法。

* [cs.DB] [**From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents**](https://arxiv.org/abs/2602.17913)
  * [Agent Memory & Provenance] 长视野智能体的历史压缩存在**写前查询壁垒**，压缩决策早于未来查询需求，易丢失关键约束导致答案无法验证；而保留原始日志虽能保证溯源性，但会带来高昂的token消耗与推理延迟。本文提出TierMem溯源感知的分层内存框架，核心创新为基于证据充分性的检索机制：设计快速摘要索引与不可变原始日志存储两层内存结构，默认从摘要索引查询，通过运行时充分性路由器，仅在摘要证据不足时升级至原始日志；同时将验证后的结果作为新摘要单元写回，并关联其原始溯源。实验在LoCoMo数据集上验证，该框架准确率达0.851（接近原始日志的0.873），输入token消耗减少54.1%，推理延迟降低60.7%，实现验证性与效率的平衡。

* [cs.DB] [**Multi-Attribute Group Fairness in k-NN Queries on Vector Databases**](https://arxiv.org/abs/2602.17858)
  * [ANNS & Fairness] 向量数据库k近邻查询的多属性组公平性研究存在空白，现有方法仅关注检索效率或单属性过滤，无法同时满足多个受保护属性的比例表示约束，且多属性公平性约束下的查询问题计算复杂度高。本文提出多属性组公平的k-NN查询框架，核心设计为：适配局部敏感哈希（LSH）加速候选生成，为受保护属性的笛卡尔积构建轻量级索引，快速检索满足联合计数约束的候选集；设计后处理阶段构建跨所有属性的公平k近邻结果，针对2个属性提出多项式时间的流基精确算法，针对3个及以上属性设计整数线性规划（ILP）精确解法。该框架提供理论性能保证，明确了效率-公平性的权衡关系。实验表明，现有向量检索方法无法直接适配公平性需求，而该框架具备良好的通用性与可扩展性，在保证检索质量的同时满足多属性公平约束。

* [cs.DC] [**Collaborative Processing for Multi-Tenant Inference on Memory-Constrained Edge TPUs**](https://arxiv.org/abs/2602.17808)
  * [Edge Inference & Multi-Tenant Optimization] 内存受限的边缘TPU多租户推理场景中，存在租户间资源竞争、内存利用率低、推理延迟高等核心问题，现有单机推理方法未考虑多租户的协同处理策略，无法适配边缘设备的资源约束。针对该问题，本文聚焦边缘TPU的多租户推理协同处理优化，核心围绕租户间的资源调度、内存共享、计算任务拆分展开设计，通过构建协同处理架构，实现边缘TPU内存资源的精细化分配与计算任务的并行调度，缓解多租户资源竞争，提升整体推理吞吐量与资源利用率。该方法针对性解决边缘设备内存受限的痛点，适配物联网、边缘智能等实际多租户推理场景。

* [cs.CL] [**Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention**](https://arxiv.org/abs/2602.18145)
  * [LLM Hallucination Detection] 大语言模型的上下文幻觉检测面临两大挑战：幻觉内容与上下文存在浅层语义关联，难以通过简单语义匹配识别；现有方法忽略token的上下文频率特征，无法区分模型生成的真实关联内容与虚假幻觉内容。本文提出基于频率感知注意力的上下文幻觉检测方法，核心设计为：挖掘token在上下文窗口中的频率分布特征，构建频率感知的注意力权重计算机制，强化低频率关键上下文token的注意力聚焦，弱化模型生成的无频率支撑的幻觉token；设计幻觉检测分类器，结合频率感知注意力特征与语义特征，实现对上下文幻觉的精准识别。该方法无需额外微调，可无缝集成至现有LLM，有效提升幻觉检测的准确率，降低漏检与误检率。

* [cs.CL] [**Sink-Aware Pruning for Diffusion Language Models**](https://arxiv.org/abs/2602.17664)
  * [Diffusion LLM & Pruning Optimization] 扩散语言模型的剪枝方法直接沿用自回归LLM的注意力锚点保留策略，但扩散模型的注意力锚点位置在生成轨迹中方差极高，锚点具有瞬时性，并非结构必需，盲目保留会造成计算冗余，影响剪枝效率。针对该问题，本文提出锚点感知剪枝（Sink-Aware Pruning）方法，核心创新为：通过量化生成轨迹中注意力锚点的位置方差，自动识别扩散模型中的不稳定锚点；对不稳定锚点进行针对性剪枝，保留真正的结构关键节点，打破自回归模型的剪枝思维定式。该方法无需重新训练，在匹配计算量的前提下，实现了更优的生成质量-效率权衡，显著优于现有扩散模型剪枝基线，有效降低扩散语言模型的推理成本。

### 2.24
* [cs.AR] [**pHNSW: PCA-Based Filtering to Accelerate HNSW Approximate Nearest Neighbor Search**](https://arxiv.org/abs/2602.19242)
  * [ANNS & Hardware-Software Co-Optimization] 经典HNSW高维近邻搜索存在计算开销大、数据访问模式不规则且体量大的问题，严重制约检索效率，单纯算法或硬件层面的优化难以兼顾精度与性能。本文提出pHNSW算法-硬件协同优化方案，核心通过PCA滤波实现双维度加速：算法侧，利用PCA对数据集做降维滤波，减少邻居节点访问量与距离计算的计算负载，同时保证检索精度；硬件侧，设计定制化pHNSW处理器并开发专属指令，针对性优化检索吞吐量与能效。基于65nm工艺节点完成处理器RTL设计，并结合DDR4、HBM1.0标准做评估，实验结果显示，相较标准HNSW实现，pHNSW在CPU上查询吞吐量（QPS）提升14.47×-21.37×，GPU上提升5.37×-8.46×，能耗最高降低57.4%，实现效率与能效的双重优化。

* [cs.AR] [**HillInfer: Efficient Long-Context LLM Inference on the Edge with Hierarchical KV Eviction using SmartSSD**](https://arxiv.org/abs/2602.18750)
  * [Edge LLM Inference & KV Cache Management] 边缘设备部署LLM的长上下文推理受限于内存与计算资源，KV缓存随上下文长度线性增长成为核心瓶颈，现有KV驱逐方法为富内存平台设计，在边缘设备上会产生高昂的数据传输开销，无法直接适配。本文提出HillInfer基于SmartSSD的边缘长上下文LLM推理框架，核心为感知重要性的分层KV缓存管理：跨CPU与SmartSSD协同管理KV缓存池，在存储侧完成KV数据重要性评估，从根源减少不必要的数据迁移；设计基于预取的自适应流水线，让GPU、CPU、SmartSSD间的计算与KV数据传输过程重叠，最小化端到端推理延迟且不损失精度。在搭载商用GPU的PC端实现该框架，多模型多基准测试表明，相较基线方法推理速度提升最高8.56×，且模型推理精度完全保留。

* [cs.IR] [**Adaptive Multi-Agent Reasoning for Text-to-Video Retrieval**](https://arxiv.org/abs/2602.19040)
  * [Text-to-Video Retrieval & Multi-Agent Reasoning] 零样本文本到视频检索方法虽通过大规模预训练提升了跨模态对齐能力，但缺乏查询相关的时序推理能力，在包含时序、逻辑、因果关系的复杂查询上表现不佳，难以适配短视频平台的规模化检索需求。本文提出自适应多智能体推理检索框架，核心根据查询需求动态编排专用智能体并完成多轮推理：设计检索智能体实现大规模视频语料的高效检索，推理智能体完成零样本上下文时序推理，查询重构智能体优化模糊查询并恢复多轮推理中的性能衰减；由编排智能体基于中间反馈与推理结果协调各智能体执行，同时引入融合检索性能内存与历史推理轨迹的通信机制，提升协同决策效率。在跨越8年的三个TRECVid基准数据集上的实验表明，该框架性能较CLIP4Clip提升一倍，且大幅超越当前SOTA方法，适配复杂查询下的规模化文本到视频检索场景。

### 2.25
* [cs.DC] [**Untied Ulysses: Memory-Efficient Context Parallelism via Headwise Chunking**](https://arxiv.org/abs/2602.21196)
  * [LLM Distributed Training & Context Parallelism] Transformer长序列处理的主流上下文并行方法（如Ring Attention、DeepSpeed Ulysses）虽实现维度扩展，但未优化内存效率，限制了支持的序列长度；而流水线分布式、激活卸载等进阶技术虽能延长上下文，却会牺牲训练吞吐量。本文提出UPipe内存高效的上下文并行技术，核心创新为**注意力头级细粒度分块**，针对自注意力层做精细化切分，大幅降低激活内存占用。该方法在32B规模Transformer上，将注意力层中间张量内存占用降低87.5%，同时训练速度与现有上下文并行技术持平；在单台8×H100节点训练Llama3-8B时，可支持500万token的上下文长度，较现有方法提升超25%，突破了激活内存的瓶颈限制。

* [cs.AR] [**FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill**](https://arxiv.org/abs/2602.20515)
  * [LLM Inference Acceleration & FPGA Optimization] 长上下文LLM推理的预填充阶段因全上下文自注意力计算成为性能瓶颈，稀疏注意力虽减少计算量，但稀疏模式的动态性导致计算变为内存受限型，且GPU推理存在高能耗问题。本文提出FAST-Prefill首个面向动态稀疏注意力的长上下文预填充FPGA加速器，核心设计三大定制化模块：融合流水线单元（基于内存感知执行顺序生成稀疏索引，减少大张量与不规则内存访问）、活跃度驱动的双层KV缓存（降低访存流量）、混合矩阵处理单元（结合DSP与LUT位平面分解，提升矩阵乘法吞吐量）。在Alveo U280上实现并基于Llama/Qwen模型测试（4K-128K上下文），相较Nvidia A5000 GPU，首token延迟（TTFT）平均加速2.5×，能效提升4.5×，适配长上下文预填充的低延迟、高能效需求。

* [cs.IR] [**Multi-Vector Index Compression in Any Modality**](https://arxiv.org/abs/2602.21202)
  * [Multi-Modal Retrieval & Multi-Vector Compression] 晚交互范式下的多向量检索是跨模态检索的主流方案，但其计算与存储成本随文档长度线性增长，在富图像、视频、音频的语料中开销极高。本文研究**查询无关的多向量索引压缩方法**，在固定向量预算下实现跨模态压缩，提出四种核心方案：序列调整、内存token、分层池化，以及创新的**注意力引导聚类（AGC）**——通过注意力机制识别文档语义显著区域作为聚类质心，并为token聚合分配权重。在文本（BEIR）、视觉文档（ViDoRe）、视频（MSR-VTT等）检索任务中验证，AGC持续优于序列调整、内存token等参数化压缩方法，比非参数化分层池化拥有更高的索引尺寸灵活性，且性能与未压缩的全索引持平甚至更优，适配全模态的多向量检索压缩需求。

* [cs.IR] [**Position-Aware Sequential Attention for Accurate Next Item Recommendations**](https://arxiv.org/abs/2602.21052)
  * [Sequential Recommendation & Attention Optimization] 序列自注意力推荐模型的加法位置嵌入存在固有缺陷：位置信息与物品嵌入语义纠缠、在深度架构中传播性弱、难以捕捉丰富的序列模式，仅让注意力机制对序列顺序实现表面敏感。本文提出**位置感知的序列注意力机制**，设计可学习的位置核仅在位置空间中运作，与语义相似度解耦并直接调制注意力权重；在每个注意力块中应用该核，实现自适应多尺度序列建模，让模型更精准捕捉时序依赖。在标准的下一个物品预测基准数据集上的实验表明，该位置核注意力机制持续优于各类强基线模型，显著提升了序列推荐的准确性。

### 2.26
* [cs.DB] [**RAC: Relation-Aware Cache Replacement for Large Language Models**](https://arxiv.org/abs/2602.21547)
  * [LLM Cache Optimization] LLM服务缓存替换的现有方法（启发式/学习型）均依赖近期、频率等有限窗口统计特征，而真实LLM工作负载存在**长重用距离、稀疏局部重复**的特点，导致这类特征鲁棒性不足，缓存命中率受限。本文提出RAC关系感知的缓存替换策略，核心挖掘请求间的语义关系指导驱逐决策，融合两类关系感知信号：一是**主题流行度**，在主题层面聚合访问证据，捕捉长视野的重用规律；二是**结构重要性**，利用主题内的局部依赖结构，区分缓存条目未来的重用价值。多负载的广泛评估表明，RAC在各类场景下均保持高有效性，缓存命中率较SOTA基线持续提升20%~30%，适配LLM服务的实际缓存需求。

* [cs.DB] [**I/O Optimizations for Graph-Based Disk-Resident Approximate Nearest Neighbor Search: A Design Space Exploration**](https://arxiv.org/abs/2602.21514)
  * [Disk-based ANNS & I/O Optimization] 基于SSD的图结构近似最近邻搜索已成为I/O受限任务，I/O耗时占查询延迟的70%~90%，现有优化缺乏系统化的设计空间探索。本文提出**I/O优先的磁盘型ANN框架**，从内存布局、磁盘布局、搜索算法三个维度梳理并组织优化技术；构建页级复杂度模型，阐释页局部性与路径长度如何共同决定页读取量，并完成实证验证。基于四个公共数据集的统一实现，量化了单因素效果与跨维度协同作用，发现内存驻留导航、动态宽度是增益最显著的独立策略，页混洗与页搜索单独效果弱但互补性强。据此设计的OctopusANN组合策略大幅降低I/O开销，在Recall@10=90%的匹配条件下，吞吐量较SOTA的Starling提升4.1%~37.9%，较DiskANN提升87.5%~149.5%。最后提炼出不同并发级别、精度约束下存储中心/混合设计的选型准则，强调系统化组合而非孤立调优是提升磁盘型ANN性能的关键。

* [cs.DB] [**PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing**](https://arxiv.org/abs/2602.21247)
  * [ANNS Index Construction & Scalability] 主流图基ANN索引（HNSW、Vamana）虽查询性能优异，但构建过程依赖随机访问密集的束搜索，存在**搜索瓶颈**，导致索引构建耗时极长，可扩展性受限。本文提出PiPNN超可扩展的图基近邻索引构建算法，核心创新为**HashPrune在线剪枝算法**，可动态维护稀疏的边集合，让算法能将数据集划分为重叠子问题，通过稠密矩阵乘法核高效完成批量距离计算，并将边子集流式输入HashPrune。HashPrune保证索引构建过程中的内存有界，使PiPNN无需额外中间内存即可构建更高质量的索引。实验表明，PiPNN构建SOTA索引的速度较Vamana（DiskANN）提升达11.6×，较HNSW提升达12.9×；可扩展性显著优于近期的快速图构建算法，较MIRAGE快至少19.1×、较FastKCNA快17.3×，且生成的索引查询吞吐量更高；首次实现单台多核机器在20分钟内为十亿级数据集构建高质量ANN索引。

* [cs.IR] [**AQR-HNSW: Accelerating Approximate Nearest Neighbor Search via Density-aware Quantization and Multi-stage Re-ranking**](https://arxiv.org/abs/2602.21600)
  * [HNSW Optimization & ANNS Acceleration] 向量数据库扩展至十亿级嵌入时，HNSW面临内存消耗剧增、距离计算开销主导查询延迟、异构数据分布下性能不佳的核心瓶颈。本文提出AQR-HNSW自适应量化与重排序的HNSW优化框架，通过三大策略的协同融合提升可扩展性：一是**密度感知的自适应量化**，在保留距离关系的前提下实现4倍压缩；二是**多阶段重排序**，减少35%的无效计算；三是**量化优化的SIMD实现**，跨架构实现每周期16~64次操作。标准基准测试表明，AQR-HNSW的查询吞吐量（QPS）较SOTA的HNSW实现提升2.5~3.3×，同时召回率保持98%以上；索引图内存占用减少75%，索引构建速度提升5×，已被DAC 2026接收。

* [cs.CL] [**Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases**](https://arxiv.org/abs/2602.21222)
  * [LoRA Adapter Composition & Vector Retrieval] LoRA等参数高效微调方法实现了LLM的任务专属适配，但为未见过的任务高效组合多个专用适配器仍面临挑战，缺乏零样本泛化能力。本文提出基于向量数据库相似度检索的**任务感知LoRA适配器动态组合框架**，核心设计为：将22个涵盖常识推理、问答、自然语言推理、情感分析的数据集训练样本嵌入，构建任务感知的向量数据库；推理时检索最相似的训练样本，通过核采样计算任务相似度分布，利用检索加权融合策略动态合并相关LoRA适配器。评估了线性、拼接、TIES、幅度剪枝四种合并方法，结果表明该数据集中心的检索方法性能持平甚至超越单独微调的任务专属适配器，其中线性合并在PIQA上达70.95%、RTE上达77.62%，大幅优于单任务基线（分别为46%、52%）。该框架无需额外训练检索器，采用冻结嵌入，实现了高效、可解释的适配器组合，为无需全模型重训的可扩展多任务学习提供了新方向。


### 2.27
* [cs.DB] [**Optimizing SSD-Resident Graph Indexing for High-Throughput Vector Search**](https://arxiv.org/abs/2602.22805)
  * [SSD-based ANNS & Graph Index Optimization] 基于SSD的图结构向量检索因图遍历的访问局部性差，存在严重的CPU利用率不足和读放大问题，存储阻塞制约了检索吞吐量与延迟性能。本文提出VeloANN高吞吐SSD图索引优化框架，核心从数据布局和运行时调度双维度缓解存储阻塞：采用分层压缩与基于亲和性的数据放置策略，将相关向量聚合在同一页，减少碎片与过度预取；设计记录级缓冲池，按向量邻居分组管理记录并持久化保存热记录，在内存受限下消除过量页交换；引入协程基异步运行时实现轻量级任务调度，降低磁盘I/O中断的CPU调度开销，结合异步预取与束感知搜索策略优先利用缓存数据。实验表明，VeloANN吞吐量较SOTA磁盘型ANN系统提升最高5.8×，延迟降低最高3.25×，仅用内存型系统10%的内存占用即可实现其0.92×的吞吐量。

* [cs.DB] [**AlayaLaser: Efficient Index Layout and Search Strategy for Large-scale High-dimensional Vector Similarity Search**](https://arxiv.org/abs/2602.23342)
  * [On-disk ANNS & High-dimensional Vector Retrieval] 磁盘型图基ANNS被普遍认为受I/O成本限制，但研究发现向量维度提升至数百/数千维时，其性能实际受计算约束而非I/O约束，现有方法仅聚焦I/O优化而忽略计算开销，存在巨大性能提升空间（该文已被SIGMOD 2026接收）。本文提出AlayaLaser大规模高维向量检索的磁盘型图索引系统，先通过适配的屋顶线模型分析现有系统性能瓶颈，再设计新型磁盘数据布局，利用现代CPU的SIMD指令缓解计算约束；同时提出度基节点缓存、聚类基入口点选择、早期调度策略等一系列优化技术。在多类大规模高维向量数据集上的实验验证，AlayaLaser不仅显著超越现有磁盘型图索引系统，性能还持平甚至优于内存型索引系统。

* [cs.DB] [**Workload-Aware Incremental Reclustering in Cloud Data Warehouses**](https://arxiv.org/abs/2602.23289)
  * [Cloud Data Warehouse & Data Reclustering] 云数据仓库依赖微分区和元数据实现高效数据剪枝，而动态云环境中持续的数据写入和工作负载演化，让现有自动聚类方法缺乏灵活性，全量重聚类成本过高。本文提出WAIR工作负载感知的增量重聚类算法，核心将重聚类策略与聚类键选择解耦，引入**边界微分区**概念（位于查询范围边界的微分区），仅对剪枝效率最关键的边界微分区进行重聚类。该算法实现了接近全排序表布局的查询性能，且重聚类成本远低于现有方法并存在理论上界；同时基于该算法实现了重聚类服务原型，在TPC-H、DSB基准和真实工作负载上的评估表明，WAIR在提升查询性能的同时降低了整体成本（该文已被SIGMOD 2026接收）。

* [cs.IR] [**Adaptive Prefiltering for High-Dimensional Similarity Search: A Frequency-Aware Approach**](https://arxiv.org/abs/2602.22214)
  * [High-Dimensional ANNS & Adaptive Prefiltering] 高维相似度检索的统一搜索策略无法利用真实查询分布的异构性，导致计算资源分配低效，冗余的距离计算增加检索开销。本文提出频率感知的自适应预过滤框架，核心基于查询频率模式和聚类一致性指标动态分配计算预算：按齐普夫分布将查询空间划分为不同频率层级，结合历史访问模式和局部密度特征为各层级分配差异化搜索策略；设计轻量级频率跟踪机制，引入基于一致性的回退策略为未见过的查询提供优雅的性能退化。在ImageNet-1k的CLIP嵌入实验表明，该框架在保持召回率不变的情况下，较静态nprobe选择减少20.4%的距离计算，且在GPU加速的FAISS索引上保持亚毫秒级延迟。

* [cs.CL] [**Fine-Tuning Without Forgetting In-Context Learning: A Theoretical Analysis of Linear Attention Models**](https://arxiv.org/abs/2602.23197)
  * [LLM Fine-tuning & In-Context Learning] LLM微调虽能提升下游任务零样本性能，但会退化模型的上下文学习能力，导致微调模型在未见过的任务上表现不佳，而这一现象的理论机理尚未明确。本文以线性注意力模型为研究对象，从理论层面分析微调目标如何修改注意力参数，并界定了导致少样本性能退化的条件：全量更新注意力参数会损害上下文学习能力，而仅限制更新值矩阵可在提升零样本性能的同时保留上下文学习能力；引入辅助少样本损失虽能提升目标任务的上下文学习性能，但会牺牲未见过任务的该能力。作者通过实验充分验证了上述理论结论，为LLM微调保留上下文学习能力提供了理论指导和实践准则。

* [cs.CL] [**InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models**](https://arxiv.org/abs/2602.23200)
  * [LLM KV Cache & Hardware-Aware Quantization] LLM解码阶段的KV缓存随序列长度线性增长，成为内存与延迟瓶颈，现有量化方法未充分结合硬件特性，调优成本高且解量化效率低。本文提出InnerQ硬件感知的无调优KV缓存量化方案，核心创新为**内维度分组量化**，将缓存矩阵沿内维度分组，使解量化与向量-矩阵乘法对齐，实现GPU计算单元间的缩放因子复用，减少内存访问并加速解量化。为保证激进压缩下的保真度，InnerQ融合三项设计：基于局部统计为每组选择对称/非对称量化的混合量化策略；为最新token和注意力锚点token设置高精度窗口，缓解离群值泄露；对键缓存做通道级归一化，在预填充阶段一次性计算并融入查询，无运行时开销。在Llama模型上的实验表明，InnerQ的少样本GSM8K性能与非量化KV缓存相当且优于现有量化方法，解码速度较现有方法提升最高22%，较半精度向量-矩阵乘法提升最高88%。