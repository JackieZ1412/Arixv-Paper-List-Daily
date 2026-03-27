### 3.16
* [cs.DB] [**RNSG: A Range-Aware Graph Index for Efficient Range-Filtered Approximate Nearest Neighbor Search**](https://arxiv.org/abs/2603.12913)

  * [Range-Filtered ANNS & Graph Index] 实际向量检索场景中，用户常需结合范围约束（如属性、维度范围）与相似度查询，但现有图基ANNS索引未针对范围过滤优化，导致检索时需先遍历图获取候选再过滤，范围约束越严格，无效计算与I/O开销越大，严重影响检索效率。本文提出RNSG范围感知图索引，核心创新为将范围约束融入图构建与检索全流程：构建阶段，基于向量维度分布划分范围分区，每个分区内构建局部图结构，同时维护跨分区的关联边，保证范围过滤时的候选召回完整性；检索阶段，设计范围引导的路径搜索策略，仅遍历符合范围约束的分区与节点，提前剪枝无效候选，减少不必要的距离计算。实验表明，RNSG在多种范围过滤场景下，检索延迟较传统图基ANNS（HNSW、Vamana）降低40%-65%，召回率保持95%以上，且随着范围约束收紧，性能优势更显著，适配电商、推荐等需结合多条件筛选的向量检索场景。

* [cs.AI] [**Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation**](https://arxiv.org/abs/2603.13017)

  * [Personalized Agent Memory & Token Compression] 个性化LLM智能体的记忆库常以原始对话、用户偏好文本形式存储，token数量庞大，导致检索延迟高、上下文窗口占用严重，现有压缩方法易丢失关键个性化信息，降低检索准确性。本文提出结构化蒸馏个性化智能体记忆方法，核心设计为：基于用户交互轨迹，提取实体、偏好、行为模式等结构化信息，构建轻量级记忆图谱，替代原始文本记忆；采用蒸馏策略，将原始记忆的语义信息与个性化特征迁移至结构化记忆中，保留检索所需的核心信息；引入记忆对齐机制，确保蒸馏后的结构化记忆与原始记忆的检索匹配度一致。实验验证，该方法实现11倍的token数量缩减，检索速度提升8-10倍，同时检索准确率保持在原始记忆的98%以上，有效解决个性化智能体记忆存储与检索效率的瓶颈，适配长时个性化交互场景。

### 3.17
* [cs.OS] [**Idiosyncrasies of Programmable Caching Engines**](https://arxiv.org/abs/2603.14357)
  * [Programmable Caching & Engine Optimization] 可编程缓存引擎（如P4、eBPF驱动）凭借灵活性成为现代存储与网络系统的核心组件，但现有研究忽略其底层硬件与软件特性带来的独特行为，导致缓存策略设计不合理、性能未达最优。本文系统研究可编程缓存引擎的固有特性（指令集限制、内存访问模式、并行处理瓶颈），识别出三大关键特性：指令级并行度受限、缓存行对齐依赖强、表查找延迟与条目数量非线性相关。基于这些发现，提出针对性优化策略，包括指令重排序、数据预对齐、动态表分区等，在真实可编程缓存硬件上验证，优化后缓存命中率提升15%-30%，请求处理延迟降低20%-40%，为可编程缓存引擎的策略设计与性能调优提供实践指导。

* [cs.IR] [**The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA**](https://arxiv.org/abs/2603.14045)
  * [Graph-RAG & Multi-Hop QA] Graph-RAG通过图谱结构化组织知识，缓解传统RAG的多跳推理瓶颈，但仍面临两大核心问题：推理过程缺乏结构化引导，导致逻辑混乱；长路径多跳推理时上下文冗余，引发注意力稀释。本文提出结构化提示与上下文压缩协同优化框架，核心设计为：基于图谱结构生成结构化提示，明确推理步骤与节点关联，引导LLM按逻辑路径进行多跳推理；引入图谱感知的上下文压缩策略，保留关键节点与关系信息，剔除冗余语义，将多跳推理上下文长度压缩60%以上。实验表明，该框架在多个多跳QA基准上，推理准确率提升18%-27%，推理速度提升2.3-3.5倍，有效突破Graph-RAG的推理瓶颈。

* [cs.AI] [**SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**](https://arxiv.org/abs/2603.14588)
  * [Enterprise Agent Memory & Zero-LLM Design] 企业级智能体记忆需兼顾安全性、高效性与可解释性，现有方案依赖LLM进行记忆检索与管理，存在隐私泄露风险与高计算成本。本文提出SuperLocalMemory V3零LLM企业智能体记忆系统，基于信息几何理论构建记忆基础，核心创新为：采用流形学习方法组织企业知识，将实体与关系映射到低维信息几何空间，实现无LLM依赖的高效记忆检索；设计几何距离驱动的记忆更新与驱逐策略，保证记忆的时效性与相关性；引入可解释性模块，通过几何空间中的节点关联，可视化记忆检索与推理过程。在企业级知识问答、流程协作任务中，该系统检索速度较LLM依赖方案提升8-12倍，无隐私泄露风险，且记忆利用率提升40%以上。

* [cs.AR] [**Dynamic Sparse Attention: Access Patterns and Architecture**](https://arxiv.org/abs/2603.13430)
  * [Sparse Attention & Hardware Architecture] 动态稀疏注意力通过动态调整注意力稀疏度，平衡LLM推理性能与精度，但现有研究多关注算法设计，忽略其硬件访问模式与架构适配性，导致硬件利用率低、性能提升未达预期。本文深入分析动态稀疏注意力的访问模式（随机访问、不规则粒度、时间域波动），识别出硬件架构的核心瓶颈：缓存命中率低、数据并行度不足、控制逻辑复杂。基于此，提出适配动态稀疏注意力的硬件架构优化，包括专用缓存设计、动态并行调度单元、不规则访问优化模块，在FPGA与ASIC原型上验证，推理吞吐量提升2.1-3.8倍，硬件利用率从45%提升至82%，为动态稀疏注意力的硬件落地提供架构支撑。

* [cs.DC] [**Token Coherence: Adapting MESI Cache Protocols to Minimize Synchronization Overhead in Multi-Agent LLM Systems**](https://arxiv.org/abs/2603.15183)
  * [Multi-Agent LLM & Cache Coherence] 多智能体LLM系统中，智能体间共享KV缓存、记忆数据，传统MESI缓存协议未针对token级数据共享优化，导致同步开销大、缓存一致性维护成本高，严重制约系统并发性能。本文提出Token Coherence适配型MESI协议，核心优化为：基于token访问频率与共享范围，动态调整缓存一致性粒度，对高频共享token采用细粒度一致性维护，对低频独立token采用粗粒度管理；引入token预取与缓存共享机制，减少智能体间的数据传输与同步等待；设计冲突预测模块，提前规避缓存一致性冲突。实验表明，该协议使多智能体LLM系统的同步开销降低50%-70%，并发吞吐量提升1.8-2.5倍，且缓存命中率保持稳定。

* [cs.DB] [**d-HNSW: A High-performance Vector Search Engine on Disaggregated Memory**](https://arxiv.org/abs/2603.13591)
  * [Disaggregated Memory & Vector Search] 向量检索引擎向大规模、高并发方向演进，解耦内存架构凭借灵活扩展能力成为主流，但现有HNSW索引在解耦内存上部署时，面临跨节点数据访问延迟高、内存带宽利用率低的问题。本文提出d-HNSW解耦内存优化的向量检索引擎，核心设计为：将HNSW索引分层部署，核心层（顶层节点）部署在本地内存，底层节点部署在解耦内存，减少跨节点访问频率；设计预取与缓存协同策略，基于检索路径预测提前预取解耦内存中的向量数据，提升内存带宽利用率；引入分布式索引同步机制，保证多节点部署时的索引一致性。实验表明，d-HNSW在十亿级向量数据集上，检索延迟较传统HNSW降低35%-55%，吞吐量提升2.2-3.3倍，且支持动态扩展，适配大规模解耦内存部署场景。

* [cs.DB] [**A New Lower Bounding Paradigm and Tighter Lower Bounds for Elastic Similarity Measures**](https://arxiv.org/abs/2603.14899)
  * [Elastic Similarity Measures & Lower Bounds] 弹性相似度度量（如动态时间规整DTW、编辑距离）广泛应用于时间序列、文本等数据的相似度计算，但现有下界估计方法宽松，导致检索过程中无效距离计算过多，影响检索效率。本文提出一种全新的下界估计范式，基于弹性相似度的内在特性，构建更紧致的下界估计模型，核心创新为：将弹性相似度分解为线性分量与非线性分量，分别设计下界估计方法，结合两者得到更紧致的整体下界；引入自适应调整机制，根据数据分布动态优化下界估计精度与计算开销。实验验证，该范式在多种弹性相似度度量下，下界紧致度提升30%-50%，检索过程中的无效距离计算减少40%-65%，显著提升弹性相似度检索的效率。

* [cs.LG] [**SmartSearch: How Ranking Beats Structure for Conversational Memory Retrieval**](https://arxiv.org/abs/2603.15599)
  * [Conversational Memory Retrieval & Ranking Strategy] 对话式智能体的记忆检索多依赖结构化记忆（如记忆图谱、时序列表），但结构化组织成本高，且难以适配对话的动态性与模糊性，导致检索精度与效率不佳。本文提出SmartSearch排序优先的对话记忆检索方法，核心观点为：无需复杂结构化组织，通过优化记忆排序策略，即可实现更高效、更精准的记忆检索。具体设计包括：基于对话上下文的动态排序权重，融合语义相关性、时间新近度、用户关注度等多维度特征；引入轻量级重排序机制，对初始检索结果进行微调，提升关键记忆的召回率；采用无结构记忆存储，降低组织与维护成本。实验表明，SmartSearch较结构化记忆检索方法，检索精度提升15%-22%，检索速度提升3-4倍，且部署成本降低60%以上。

* [cs.LG] [**Orthogonal Subspace Clustering: Enhancing High-Dimensional Data Analysis through Adaptive Dimensionality Reduction and Efficient Clustering**](https://arxiv.org/abs/2603.14783)
  * [High-Dimensional Clustering & Subspace Learning] 高维数据聚类面临维度灾难、数据分布异构等问题，现有子空间聚类方法需手动指定子空间维度，适应性差，且聚类效率低。本文提出正交子空间聚类方法，核心创新为：基于自适应维度约简策略，通过正交投影自动学习数据的最优子空间维度，无需人工干预；设计高效聚类算法，在正交子空间内实现数据的快速聚类，同时保留原始数据的关键特征；引入子空间评估机制，动态调整子空间与聚类参数，适配不同分布的高维数据。在多个高维数据集（图像、文本嵌入）上实验，该方法较传统子空间聚类，聚类精度提升12%-20%，聚类速度提升2.5-4倍，有效缓解高维数据聚类的维度灾难问题。

* [cs.LG] [**Cross-RAG: Zero-Shot Retrieval-Augmented Time Series Forecasting via Cross-Attention**](https://arxiv.org/abs/2603.14709)
  * [Time Series Forecasting & RAG Integration] 时间序列预测受数据分布偏移、长程依赖等问题影响，现有方法难以利用外部相关知识，泛化能力不足。本文提出Cross-RAG零样本检索增强时间序列预测框架，核心设计为：构建时间序列相关知识的向量数据库，包含历史预测案例、领域知识、趋势特征等；采用跨注意力机制，将检索到的相关知识与当前时间序列数据进行融合，引导模型学习外部知识与目标序列的关联；设计零样本适配模块，无需针对特定数据集微调，即可实现多领域时间序列的精准预测。实验表明，Cross-RAG在多个时间序列预测基准上，预测准确率提升10%-18%，尤其在数据稀缺、分布偏移场景下，优势更显著，为时间序列预测提供了全新的知识增强思路。

* [cs.LG] [**AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems**](https://arxiv.org/abs/2603.14688)
  * [Multi-Agent System & Root Cause Analysis] 部署后的多智能体系统易出现行为异常、任务失败等问题，现有故障排查方法依赖日志分析，效率低且难以定位深层因果关系。本文提出AgentTrace因果图追踪框架，核心创新为：构建多智能体交互因果图，记录智能体间的消息传递、动作执行、状态变化等因果关系；设计因果追踪算法，基于因果图反向追溯异常的根源，区分直接原因与间接原因；引入可视化模块，直观展示因果链路，辅助运维人员快速定位问题。在部署的多智能体协作系统中验证，AgentTrace将故障排查时间缩短70%-85%，根因定位准确率达90%以上，有效提升多智能体系统的可维护性。

* [cs.LG] [**Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys**](https://arxiv.org/abs/2603.14224)
  * [KV Cache Optimization & Sparse Attention] 长上下文LLM推理中，稀疏注意力需依赖索引器确定关注的token，索引计算开销大，且KV缓存的高内存占用进一步制约性能。本文提出自索引KV缓存方法，核心设计为：对KV缓存中的Key进行轻量级压缩，同时保留关键语义信息；基于压缩后的Key，训练轻量级预测模型，直接预测稀疏注意力的索引，无需额外运行索引器，降低计算开销；设计缓存与预测协同机制，动态调整Key压缩精度与预测模型参数，平衡性能与精度。实验表明，该方法在LLaMA3、Qwen3模型上，索引计算开销降低80%以上，KV缓存内存占用减少30%-50%，推理速度提升1.5-2.3倍，模型精度损失可忽略。

* [cs.LG] [**Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors**](https://arxiv.org/abs/2603.14087)
  * [Next-Token Predictor & Feature Analysis] 大语言模型的下一个token预测器中，常出现看似无用的特征（如低激活值、与任务无关的特征），其存在原因与影响尚未明确。本文通过细粒度特征分析，揭示此类特征的 emergence 机制与实际作用：看似无用的特征并非冗余，而是用于捕捉边缘案例、缓解过拟合、增强模型鲁棒性；这些特征的出现与训练数据分布、模型架构、正则化策略密切相关，在低资源、高噪声数据集中更易出现。实验验证，移除此类特征会导致模型在边缘案例上的性能下降15%-25%，过拟合风险显著增加，为LLM的特征工程与模型优化提供了新的认知。

* [cs.LG] [**RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse**](https://arxiv.org/abs/2603.13289)
  * [LLM Collaboration & KV Cache Reuse] 多LLM协作完成复杂任务（如多轮对话、协同推理）时，各LLM独立维护KV缓存，存在大量重复计算与内存浪费，导致协作效率低下。本文提出RelayCaching中继缓存方法，核心创新为：建立多LLM协作的KV缓存共享机制，将前一个LLM的解码KV缓存中继给下一个LLM，避免重复预填充与计算；设计缓存适配模块，调整不同LLM的KV缓存格式，实现跨模型缓存复用；引入缓存有效性评估机制，仅复用高质量、高相关性的KV缓存内容，保证协作精度。实验表明，RelayCaching使多LLM协作的推理速度提升2.1-3.4倍，内存占用减少40%-60%，且协作任务的完成质量保持稳定。

* [cs.LG] [**ICaRus: Identical Cache Reuse for Efficient Multi Model Inference**](https://arxiv.org/abs/2603.13281)
  * [Multi-Model LLM Inference & Cache Reuse] 多模型LLM推理场景（如模型ensemble、多模态协同）中，不同模型的输入存在大量重叠（如系统提示、公共上下文），但各模型独立维护缓存，导致内存开销与计算开销剧增。本文提出ICaRus相同缓存复用方法，核心设计为：识别多模型输入中的相同片段，构建共享缓存池，统一存储相同片段的KV缓存；设计缓存索引机制，快速定位并复用共享缓存，减少重复预填充；引入缓存一致性维护策略，确保共享缓存的更新与同步。在多模型ensemble推理场景中，ICaRus使内存占用减少50%-75%，推理速度提升1.8-2.8倍，且模型输出精度无损失。

* [cs.AR] [**Machine Learning-Driven Intelligent Memory System Design: From On-Chip Caches to Storage**](https://arxiv.org/abs/2603.14583)
  * [Intelligent Memory System & ML-Driven Design] 传统内存系统（片上缓存、DRAM、存储）采用固定策略，难以适配AI应用的动态内存需求，导致性能瓶颈与资源浪费。本文提出机器学习驱动的智能内存系统设计范式，覆盖从片上缓存到存储的全内存层次：采用强化学习优化缓存替换策略，基于应用访问模式动态调整缓存配置；利用预测模型预测内存访问热点，实现数据预取与分区优化；通过深度学习优化存储I/O调度，提升数据读写效率。该范式在AI训练与推理场景中验证，内存系统吞吐量提升30%-50%，延迟降低25%-40%，资源利用率提升40%以上，为下一代智能内存系统设计提供了理论与实践基础。

* [cs.CV] [**ASAP: Attention-Shift-Aware Pruning for Efficient LVLM Inference**](https://arxiv.org/abs/2603.14549)
  * [LVLM Inference & Attention Pruning] 视觉语言模型（LVLM）的注意力机制计算开销大，现有剪枝方法忽略注意力偏移特性（注意力焦点随输入变化），导致剪枝后模型精度下降严重。本文提出ASAP注意力偏移感知剪枝方法，核心创新为：分析LVLM的注意力偏移模式，识别出稳定注意力区域与动态注意力区域；对稳定注意力区域采用激进剪枝，对动态注意力区域采用保守剪枝，平衡剪枝率与精度；设计剪枝自适应调整机制，根据输入内容动态调整剪枝策略。实验表明，ASAP在保持LVLM推理精度下降不超过3%的前提下，注意力计算开销降低60%-75%，推理速度提升2.5-3.8倍，适配边缘设备与高并发LVLM部署场景。

* [cs.CL] [**GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent**](https://arxiv.org/abs/2603.13875)
  * [LLM Memory Writing & Gradient Descent] LLM的记忆写入多依赖启发式策略，难以精准捕捉上下文的关键信息，导致记忆检索精度低、上下文利用率不足。本文提出GradMem测试时梯度下降驱动的记忆写入方法，核心设计为：在测试阶段，通过梯度下降优化记忆内容，使记忆更贴合当前上下文与任务需求；引入记忆损失函数，衡量记忆内容与上下文的相关性、完整性，引导梯度下降方向；设计轻量级优化机制，避免额外计算开销，确保推理效率。实验表明，GradMem使LLM的记忆检索精度提升20%-30%，上下文利用率提升40%以上，在长时对话、多轮推理任务中表现突出。

* [cs.CL] [**CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents**](https://arxiv.org/abs/2603.15421)
  * [Small LLM Agent & Memory Organization] 小型语言模型（SLM）智能体的内存资源有限，现有记忆组织方法结构固定，无法自适应任务变化，导致记忆冗余、检索效率低。本文提出CLAG智能体驱动聚类的自适应记忆组织方法，核心创新为：由SLM智能体自主决策记忆聚类策略，根据任务类型、上下文特征动态调整聚类粒度与方式；设计记忆聚类评估机制，由智能体实时评估聚类效果，优化聚类参数；采用轻量级聚类算法，适配SLM的计算与内存限制。实验表明，CLAG使SLM智能体的记忆检索速度提升3-5倍，记忆冗余减少50%-65%，任务完成率提升18%-25%，有效解决SLM智能体的记忆管理瓶颈。

* [cs.CL] [**Attention Residuals**](https://arxiv.org/abs/2603.15031)
  * [Transformer Attention & Residual Learning] 现有Transformer的注意力机制缺乏有效的残差设计，导致注意力特征传递过程中信息丢失、梯度消失，影响模型性能与训练稳定性。本文提出注意力残差（Attention Residuals）机制，核心创新为：在注意力层引入残差连接，将原始输入特征与注意力输出特征进行融合，保留原始信息的同时增强注意力特征的表达能力；设计自适应残差权重，根据注意力强度动态调整残差贡献，提升模型的自适应能力；将注意力残差机制融入Transformer的 encoder 与 decoder 层，优化特征传递与梯度流动。实验表明，该机制使LLM在语言建模、文本生成任务上的性能提升5%-12%，训练收敛速度加快20%-30%，且模型鲁棒性显著增强。

* [cs.CL] [**SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging**](https://arxiv.org/abs/2603.14303)
  * [KV Cache Compression & Semantic Analysis] 长上下文LLM的KV缓存内存占用巨大，现有压缩方法多基于数值量化，易丢失语义信息，导致模型精度下降。本文提出SemantiCache语义感知的KV缓存压缩方法，核心设计为：基于语义相似度对KV缓存进行分块，将语义相近的token聚合为一个语义块；采用聚类合并策略，对语义块进行合并压缩，保留核心语义信息；引入语义恢复机制，确保压缩后的KV缓存能准确恢复原始语义，减少精度损失。实验表明，SemantiCache在实现70%-80% KV缓存压缩率的同时，模型精度损失不超过2%，推理速度提升3.2-4.5倍，适配长上下文LLM推理场景。

* [cs.AI] [**SAGE: Multi-Agent Self-Evolution for LLM Reasoning**](https://arxiv.org/abs/2603.15255)
  * [Multi-Agent LLM & Self-Evolution] 多智能体LLM推理面临协作效率低、推理能力固化、难以适应复杂任务等问题，现有方法需人工干预调整智能体策略，扩展性差。本文提出SAGE多智能体自进化框架，核心创新为：建立智能体自进化机制，通过多智能体协作推理的反馈的结果，自主优化智能体的推理策略、知识储备与协作方式；设计进化评估指标，从推理精度、协作效率、任务适配性三个维度评估智能体性能，引导进化方向；引入知识共享与迁移机制，促进智能体间的能力互补，加速进化过程。实验表明，SAGE使多智能体LLM的推理准确率提升22%-35%，协作效率提升3.1-4.2倍，能自主适应不同类型的复杂推理任务。

* [cs.AI] [**Memory as Asset: From Agent-centric to Human-centric Memory Management**](https://arxiv.org/abs/2603.14212)
  * [Agent Memory Management & Human-Centric Design] 现有智能体记忆管理以智能体为中心，忽略人类用户的需求与交互习惯，导致记忆内容与人类预期不符、用户体验不佳。本文提出“记忆即资产”的人类中心型记忆管理框架，核心观点为：将智能体记忆视为可管理、可优化的资产，围绕人类用户的需求设计记忆的采集、存储、检索与更新策略；引入用户偏好建模模块，捕捉用户的记忆使用习惯与需求优先级；设计可解释的记忆交互接口，允许用户手动调整记忆内容与权重，提升用户控制权。该框架在个性化助手、协作智能体场景中验证，用户满意度提升60%以上，记忆与用户需求的匹配度提升45%-55%，为智能体记忆管理提供了全新的设计视角。

* [cs.AI] [**TheraAgent: Multi-Agent Framework with Self-Evolving Memory and Evidence-Calibrated Reasoning for PET Theranostics**](https://arxiv.org/abs/2603.13676)
  * [PET Theranostics & Multi-Agent Framework] PET诊疗（正电子发射断层显像）需结合多领域知识（医学影像、病理、临床数据），现有智能诊断系统缺乏多专业协作能力，推理过程缺乏证据支撑，准确性不足。本文提出TheraAgent PET诊疗多智能体框架，核心设计为：构建多专业智能体（影像分析、病理诊断、临床决策），各智能体具备自进化记忆，持续积累诊疗经验；引入证据校准推理机制，要求智能体的诊断结论必须有明确的医学证据支撑，提升推理的可信度；设计多智能体协作协议，实现跨专业知识共享与协同决策。在临床PET诊疗数据上验证，该框架的诊断准确率达88%以上，较单一智能体提升20%-30%，为PET诊疗提供了高效、可靠的智能辅助方案。

* [cs.AI] [**D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing**](https://arxiv.org/abs/2603.14597)
  * [Agent Memory & Dopamine-Gated Mechanism] 智能体记忆的更新与检索缺乏有效的激励机制，导致关键信息易被遗忘、无关信息冗余积累，影响智能体的长期决策能力。本文受人类大脑多巴胺记忆调控机制启发，提出D-MEM多巴胺门控智能体记忆系统，核心创新为：基于奖励预测误差（RPE）路由多巴胺信号，引导记忆的更新与检索，对高奖励相关的记忆给予优先保留与检索权重；设计多巴胺门控模块，动态调整记忆的存储强度与检索优先级，实现记忆的高效管理；引入记忆衰减机制，结合多巴胺信号强度，对低价值记忆进行梯度衰减，减少冗余。实验表明，D-MEM使智能体在长视野决策任务中的成功率提升25%-38%，记忆检索效率提升3-5倍，有效增强智能体的长期记忆与决策能力。

* [cs.RO] [**OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism**](https://arxiv.org/abs/2603.14371)
  * [VLAM & KV Cache Management] 视觉语言动作模型（VLAM）在多任务并行场景中（如机器人同时执行感知、决策、动作生成），KV缓存随任务数量线性增长，内存瓶颈突出，且不同任务的缓存需求差异大，现有管理方法无法适配。本文提出OxyGen统一KV缓存管理系统，核心设计为：采用任务感知的缓存分区策略，根据不同任务的缓存需求与优先级，动态分配KV缓存资源；设计缓存共享与驱逐协同机制，实现跨任务缓存复用，同时优先保留高优先级任务的缓存；引入轻量级任务调度与缓存适配模块，确保多任务并行时的缓存效率与模型性能。实验表明，OxyGen使VLAM在多任务并行场景下，内存占用减少40%-65%，任务执行效率提升2.3-3.6倍，且动作生成精度保持稳定。

* [cs.LG] [**Your Code Agent Can Grow Alongside You with Structured Memory**](https://arxiv.org/abs/2603.13258)
  * [Code Agent & Structured Memory] 代码智能体需持续适配用户的编码风格、项目需求与技术偏好，但现有代码智能体缺乏有效的记忆成长机制，难以实现长期个性化适配。本文提出结构化记忆驱动的代码智能体成长方法，核心设计为：构建代码相关的结构化记忆，包括用户编码风格、项目架构、常用函数、错误案例等；设计记忆更新机制，基于用户的编码反馈与项目迭代，持续丰富与优化结构化记忆；引入记忆引导的代码生成策略，使代码智能体生成的代码贴合用户习惯与项目需求。实验表明，该方法使代码智能体的个性化适配度提升45%-60%，代码生成准确率提升20%-30%，能随着用户的编码实践持续成长，适配不同类型的编码任务。

### 3.18-3.20
* [cs.AI] [**MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution**](https://arxiv.org/abs/2603.18718)
  * [Multi-Agent Memory Cycle & Self-Evolution] 现有智能体内存系统在写入、检索、更新、遗忘的完整生命周期上缺乏统一协调机制，多智能体之间内存视图不一致，且难以随任务动态进化。本文提出MemMA多智能体内存周期协调框架，通过专门的记忆推理智能体统一管理写入门控、检索路由、版本合并与渐进遗忘；支持原位自进化，在不重启系统的情况下根据任务反馈重构记忆结构与优先级。在多轮工具使用与开放式规划任务中，MemMA显著提升记忆利用率与任务成功率，同时保持内存占用可控。

* [cs.AI] [**Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism**](https://arxiv.org/abs/2603.18712)
  * [Time Series Forecasting & Sparse Attention] 多变量时间序列预测中，全注意力计算昂贵且易引入无关通道噪声。本文提出面向多通道时序的稀疏注意力机制，自动学习变量间的稀疏依赖图，仅在高相关变量间建立注意力连接；结合通道级重要性加权与局部时间窗口约束，在保持预测精度的同时大幅降低计算复杂度。在多个真实时序数据集上实现更高预测准确率与更快推理速度，尤其适用于高维工业监测与能源负荷预测场景。

* [cs.AI] [**MANAR: Memory-augmented Attention with Navigational Abstract Conceptual Representation**](https://arxiv.org/abs/2603.18676)
  * [Abstract Memory & Attention Augmentation] 传统检索增强仅返回文本片段，缺乏高层概念抽象，导致智能体难以进行结构化推理与导航式记忆访问。本文提出MANAR记忆增强注意力机制，将记忆编码为可导航的抽象概念表示，形成概念层级与关联路径；注意力可直接在概念空间中跳转、检索与组合，而非仅依赖词级别相似度。实验表明该方法在多跳推理与复杂任务规划上显著提升可解释性与完成率。

* [cs.AI] [**From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory**](https://arxiv.org/abs/2603.18420)
  * [Unsupervised Concept Discovery & Predictive Memory] 传统主题模型仅捕获共现，无法学习概念间的动态转移与因果结构。本文基于预测性联想记忆，在大规模语料上无监督发现概念及其转移结构；通过记忆的预测误差驱动概念边界更新，形成更符合人类认知的概念图。可用于知识图谱自动构建、文档结构分析与长文本理解。

* [cs.LG] [**Self-Tuning Sparse Attention: Multi-Fidelity Hyperparameter Optimization for Transformer Acceleration**](https://arxiv.org/abs/2603.18417)
  * [Sparse Attention & Auto-Tuning] 稀疏注意力的稀疏度、窗口大小、top-k 等超参数通常靠人工调参，难以在不同层、不同任务间最优。本文提出自调优稀疏注意力框架，使用多保真度超参数优化，在低开销评估与高精度评估之间动态切换；为每一层自适应学习最优稀疏策略，在保持精度的同时最大化推理加速。在长上下文模型上实现稳定的吞吐量提升与延迟降低。

* [cs.AI] [**Governed Memory: A Production Architecture for Multi-Agent Workflows**](https://arxiv.org/abs/2603.17787)
  * [Production Multi-Agent Memory & Governance] 面向工业级多智能体流程，现有记忆系统缺乏权限、可见性、版本与审计控制，难以落地。本文提出Governed Memory架构，内置多租户隔离、细粒度读写权限、记忆版本链、可审计日志与过期策略；支持智能体间受控的记忆共享与安全隔离，同时兼容企业级工作流调度。强调可观测性、合规性与稳定性，适合生产环境部署。

* [cs.AI] [**Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures**](https://arxiv.org/abs/2603.17244)
  * [Cognitive Memory & Belief Revision] 智能体信念随信息更新会出现矛盾、过时与不一致，传统追加式记忆无法维护正确性。本文提出图原生认知内存，将记忆建模为带版本的信念图；给出形式化信念修正语义，支持冲突检测、优先级合并与可追溯更新。使智能体能够在接收矛盾信息时理性更新世界模型，减少幻觉与不一致推理。

* [cs.LG] [**Attention Sinks Induce Gradient Sinks**](https://arxiv.org/abs/2603.17771)
  * [Attention Sink & Gradient Dynamics] 已有研究关注注意力沉陷对推理的影响，本文从梯度视角揭示其新危害：注意力沉陷会导致对应位置形成梯度沉陷，削弱长距离依赖的梯度传播。通过理论与实验证明，早期token过度吸收注意力会造成局部梯度消失，影响模型训练稳定性与长上下文能力。并基于该发现提出简单有效的缓解策略。

* [cs.AI] [**Selective Memory for Artificial Intelligence: Write-Time Gating with Hierarchical Archiving**](https://arxiv.org/abs/2603.15994)
  * [Memory Writing & Hierarchical Archiving] 无选择地记忆所有内容导致检索噪声与内存膨胀。本文提出写入时门控的选择性记忆机制，在记忆写入阶段即根据重要性、新颖性、置信度进行过滤；构建分层归档结构，高频活跃记忆保留在快速层，低频历史记忆迁移至压缩归档层，支持按需召回。在长期对话与持续学习场景中显著提升检索效率与记忆质量。

* [cs.AI] [**Compiled Memory: Not More Information, but More Precise Instructions for Language Agents**](https://arxiv.org/abs/2603.15666)
  * [Compiled Memory & Instructional Memory] 传统记忆以信息存储为中心，而智能体更需要可执行的指令与策略。本文提出编译式记忆，将原始经验编译为精确的结构化指令、规则与决策流程，而非存储大量冗余文本。使语言智能体能够更快执行规划、工具调用与约束遵循，减少幻觉与无效尝试，在任务型智能体上表现尤为突出。

* [cs.AI] [**CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems**](https://arxiv.org/abs/2603.15642)
  * [Brain-Inspired Agent Memory] 受大脑功能分区与容量约束启发，提出CraniMem门控有限内存系统；将记忆分为工作记忆、短期陈述记忆、长期程序记忆，各区域有独立容量与交换机制。通过神经启发式门控控制信息在区间的流动，实现类似人类认知的有限理性记忆模型，在资源受限边缘智能体上表现高效。

* [cs.AI] [**NextMem: Towards Latent Factual Memory for LLM-based Agents**](https://arxiv.org/abs/2603.15634)
  * [Factual Memory & Latent Representation] 传统基于嵌入的记忆易受表面文本噪声干扰，事实一致性差。本文提出NextMem隐式事实记忆，将知识编码为更稳定的隐式事实表示，而非表面字符串；通过对比学习与一致性约束增强事实鲁棒性。在开放域问答与事实核查中显著降低幻觉，提升记忆的事实准确性。

* [cs.AI] [**MemX: A Local-First Long-Term Memory System for AI Assistants**](https://arxiv.org/abs/2603.16171)
  * [Local-First Memory & Privacy] 现有云侧记忆存在隐私泄露风险，本文提出MemX本地优先的长期记忆系统，用户数据优先存储在本地设备，仅加密索引或必要摘要同步到云端；支持端侧检索、增量同步与离线可用。在个人助手场景中兼顾长期记忆能力与数据隐私，同时保持较低的检索延迟。