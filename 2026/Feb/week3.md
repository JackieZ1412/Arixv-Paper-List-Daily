### 2.16-2.18

* [cs.DB] [**Efficient Approximate Nearest Neighbor Search under Multi-Attribute Range Filter**](https://arxiv.org/abs/2602.15488)
  * [ANNS & Vector Database] 多属性范围过滤下的近似最近邻搜索中，现有方法存在两大核心问题：一是多属性过滤条件与向量相似度的关联建模不足，导致过滤后检索精度大幅下降；二是多属性组合查询的剪枝策略低效，需遍历大量无效候选节点，计算开销随属性维度指数增长。为解决上述问题，本文提出MAF-ANNS高效检索框架，核心设计为：构建多属性-向量联合索引，将属性范围约束嵌入向量索引结构；设计分层剪枝策略，先基于属性范围快速过滤无效区域，再在剩余区域内优化向量检索；引入属性相关性权重，动态调整不同属性的过滤优先级。实验结果表明，该框架在电商、地理空间数据集上，检索延迟降低40%-55%，召回率保持97%以上，适配多属性约束的大规模向量检索场景。

* [cs.AR] [**CacheMind: From Miss Rates to Why -- Natural-Language, Trace-Grounded Reasoning for Cache Replacement**](https://arxiv.org/abs/2602.12422)
  * [Cache Optimization] 缓存替换策略分析中，现有方法仅关注命中率/缺失率等量化指标，无法解释缺失原因，导致策略优化缺乏针对性；且分析结果以数值形式呈现，非专业人员难以理解。本文提出CacheMind分析框架，核心创新为：基于缓存轨迹的事实提取模块，从访问轨迹中挖掘缺失的关键诱因；设计自然语言推理模型，将量化指标转化为可解释的自然语言结论；构建轨迹-结论关联验证机制，保证推理结果的准确性。实验验证，CacheMind可精准识别95%以上的缓存缺失诱因（如局部性不足、预取失效），生成的自然语言报告可降低80%的分析成本，适配CPU/GPU/存储等多场景缓存优化。

* [cs.CV] [**Dual-Signal Adaptive KV-Cache Optimization for Long-Form Video Understanding in Vision-Language Models**](https://arxiv.org/pdf/2602.14236)
  * [VLM & KV Cache] 视觉语言模型的长视频理解任务中，KV-Cache存在两大问题：一是仅依赖视觉信号优化缓存，忽略语言模态的语义关联，导致缓存冗余；二是缓存策略静态化，无法适配视频帧的动态语义变化，推理效率低。本文提出双信号自适应KV-Cache优化框架，核心设计为：融合视觉帧特征与语言语义特征的双信号缓存判定模块；基于视频片段语义重要性的动态缓存粒度调整策略；引入跨模态注意力蒸馏，保证缓存压缩后的语义完整性。实验表明，该框架在长视频问答数据集上，KV-Cache存储开销降低65%，推理延迟减少40%，视频理解准确率仅下降1.8%。

* [cs.PF] [**Characterize LSM-tree Compaction Performance via On-Device LLM Inference**](https://arxiv.org/abs/2602.12669)
  * [LSM-tree & On-Device LLM] LSM-tree压缩性能表征中，现有方法依赖人工定义的特征指标，无法捕捉压缩过程的复杂模式，且表征结果无法直接指导优化；同时，云端LLM推理成本高，无法适配边缘设备的实时分析需求。本文提出基于端侧LLM的LSM-tree压缩性能表征方法，核心创新为：轻量化端侧LLM模型适配，降低推理资源消耗；构建压缩轨迹的自然语言特征提取模块，将数值轨迹转化为语义特征；设计性能预测与根因分析双任务模型，同时输出性能指标与优化建议。实验验证，该方法在边缘存储设备上，性能预测准确率达92%，分析延迟控制在100ms内，较传统方法优化效率提升35%。

* [cs.CL] [**Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory**](https://arxiv.org/abs/2602.15313)
  * [LLM Long-Term Memory] LLM长期记忆检索中，现有方法缺乏层级化组织，导致长时记忆召回率低；且单一路径检索策略无法平衡效率与精度，短时/长时记忆混淆严重。本文提出Mnemis层级图双路径检索框架，核心设计为：构建短时-长时记忆分层图结构，按时间与语义层级组织记忆节点；设计语义路径（精准检索）与时间路径（快速检索）双路由策略；引入路径融合机制，动态加权融合双路径检索结果。实验表明，该框架在长程对话、知识溯源任务中，记忆召回率提升25%-30%，检索延迟降低20%，适配LLM长期记忆管理场景。

* [cs.CL] [**GLM-5: from Vibe Coding to Agentic Engineering**](https://arxiv.org/abs/2602.15763)
  * [GLM Model & Agentic AI] GLM系列模型迭代中，现有版本聚焦单一任务能力提升，缺乏智能体工程化设计，无法适配复杂多任务场景；且编码能力与实际工程落地需求脱节，部署成本高。GLM-5的核心升级为：从“氛围编码”转向“智能体工程”，构建模块化智能体架构，支持任务拆解、资源调度、结果验证全流程；优化模型轻量化与部署适配，降低工程落地成本；强化多模态、多工具调用能力，适配真实工程场景需求。实验验证，GLM-5在代码生成、智能体任务上，完成率提升40%，部署资源消耗降低50%，具备端到端工程化落地能力。

* [cs.MA] [**Colosseum: Auditing Collusion in Cooperative Multi-Agent System**](https://arxiv.org/abs/2602.15198)
  * [Multi-Agent Collusion Auditing] 协作式多智能体系统中，现有审计方法无法有效识别智能体合谋行为，且审计过程干扰正常协作，导致系统性能下降；同时，合谋行为隐蔽性强，缺乏可解释的审计依据。本文提出Colosseum审计框架，核心设计为：基于行为轨迹的合谋特征提取模块，识别异常协作模式；设计无干扰审计策略，在不影响正常协作的前提下完成监测；构建合谋证据链生成模块，输出可解释的审计报告。实验表明，该框架在多智能体博弈、协作任务中，合谋识别准确率达98%，系统性能损耗控制在5%以内，适配多智能体安全审计场景。

* [cs.CL] [**Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation**](https://arxiv.org/abs/2602.14770)
  * [LLM Humor Generation & Multi-Agent] LLM幽默生成任务中，单模型生成的幽默内容缺乏多样性与趣味性，且未利用群体讨论的创意激发效应；现有多智能体方法缺乏针对性的交互机制，无法提升幽默生成质量。本文构建Multi-Agent Comedy Club多智能体框架，核心创新为：分工协作的智能体角色设计（创意构思、笑点优化、风格适配）；基于社区讨论模式的智能体交互策略，模拟群体创意碰撞；引入幽默评分反馈机制，迭代优化生成内容。实验验证，该框架生成的幽默内容在趣味性、多样性评分上提升30%-40%，用户满意度达85%以上。

* [cs.CL] [**AllMem: A Memory-centric Recipe for Efficient Long-context Modeling**](https://arxiv.org/abs/2602.13680)
  * [Long-Context LLM & Memory Optimization] 长上下文LLM建模中，现有方法聚焦模型结构优化，忽略内存系统的瓶颈制约，导致长上下文推理效率低；且内存管理与模型计算耦合过深，无法灵活适配不同硬件环境。本文提出AllMem内存中心式长上下文建模方案，核心设计为：解耦模型计算与内存管理，构建独立的内存优化层；设计分层内存调度策略，按上下文重要性分配内存资源；引入内存感知的模型推理策略，动态调整计算粒度以适配内存约束。实验表明，该方案在128K/256K长上下文场景下，推理延迟降低50%-60%，内存占用减少45%，适配不同硬件平台的长上下文推理。

* [cs.CL] [**G2CP: A Graph-Grounded Communication Protocol for Verifiable and Efficient Multi-Agent Reasoning**](https://arxiv.org/abs/2602.13370)
  * [Multi-Agent Communication] 多智能体推理的通信协议中，现有方法缺乏可验证性，通信内容易出错且难以追溯；且通信效率低，冗余信息多，导致推理耗时增加。本文提出G2CP图基通信协议，核心创新为：构建推理过程的图结构表征，将通信内容锚定到图节点/边，保证可验证性；设计基于图相似度的通信内容压缩策略，减少冗余信息；引入图一致性验证机制，实时校验通信内容的准确性。实验验证，该协议在多智能体推理任务中，通信开销降低60%，推理准确率提升15%-20%，且可快速定位99%以上的通信错误。

* [cs.CL] [**TraceBack: Multi-Agent Decomposition for Fine-Grained Table Attribution**](https://arxiv.org/abs/2602.13059)
  * [Table Attribution & Multi-Agent] 细粒度表格归因任务中，单模型无法处理复杂表格的多维度归因需求，且归因结果缺乏细粒度解释；现有多智能体方法分工不明确，导致归因效率低。本文提出TraceBack多智能体分解框架，核心设计为：按表格结构/语义维度的智能体分工策略（单元格解析、关系推理、归因验证）；基于分解-聚合的归因流程，先拆分归因任务至各智能体，再聚合结果生成细粒度报告；引入归因轨迹回溯机制，输出可解释的归因依据。实验表明，该框架在WikiTable、TabFact数据集上，归因准确率提升25%，细粒度解释覆盖率达98%，适配复杂表格归因场景。

### 2.19

* [cs.DC] [**FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving**](https://arxiv.org/abs/2602.16603)
  * [LLM Serving & Scheduling] LLM服务中的预填充调度存在头阻塞问题：抢占机制与调度粒度耦合过深，导致长请求阻塞短请求，服务延迟显著增加；且调度策略静态化，无法适配请求的动态特征。本文提出FlowPrefill调度框架，核心设计为：解耦抢占机制与预填充调度粒度，支持细粒度调度与灵活抢占；设计请求优先级动态调整策略，基于请求长度、紧急度实时优化调度顺序；引入预填充分片机制，将长请求拆分为分片并行处理。实验验证，该框架在LLM服务集群中，头阻塞率降低80%，短请求延迟减少70%，整体服务吞吐量提升40%。

* [cs.IR] [**Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System**](https://arxiv.org/abs/2602.16124)
  * [ANN Retrieval & Recommendation System] 大规模推荐系统的ANN检索中，现有方法仅优化检索精度/效率单一维度，忽略推荐场景的多维度需求（如多样性、时效性）；且索引结构固定，无法适配用户兴趣的动态变化。本文提出多维度可学习索引框架，核心创新为：融合精度、效率、多样性的多目标索引优化；基于用户兴趣动态的索引自适应调整策略；引入推荐场景特有的特征融合模块，提升索引的推荐相关性。实验表明，该框架在电商推荐数据集上，检索效率提升35%，推荐多样性提升25%，用户点击率提升18%，适配大规模推荐系统的ANN检索需求。

* [cs.IR] [**Neighborhood Stability as a Measure of Nearest Neighbor Searchability**](https://arxiv.org/abs/2602.16673)
  * [ANNS & Searchability Evaluation] 最近邻搜索的可搜索性评估中，现有指标仅关注检索精度/效率，忽略邻域稳定性这一核心特性，导致评估结果无法反映真实检索性能；且邻域稳定性缺乏量化方法，无法指导索引优化。本文提出邻域稳定性量化评估方法，核心设计为：定义邻域稳定性指标，量化查询点邻域的一致性；构建稳定性-精度-效率的联合评估体系；设计基于邻域稳定性的索引优化策略，优先优化稳定性差的区域。实验验证，该指标可精准预测90%以上的检索性能波动，基于该指标的索引优化可提升检索稳定性30%，适配各类ANNS算法的评估与优化。

### 2.20
* [cs.DB] [**Multiple Index Merge for Approximate Nearest Neighbor Search**](https://arxiv.org/abs/2602.17099)
  * [ANNS & Index Fusion] 近似最近邻搜索的多索引融合场景中，现有方法存在两大核心缺陷：一是索引合并策略简单，多采用暴力合并或加权平均，未能有效利用不同索引的互补性，导致融合后精度提升有限；二是合并过程计算开销高，需全量比对多索引候选集，抵消了多索引带来的检索效率优势。针对这些问题，本文提出MIM-ANNS多索引合并框架，核心设计包括：基于候选向量相似度的分层合并策略，先快速去重再融合高价值候选；引入索引置信度评估模块，动态为不同索引的检索结果分配权重；设计轻量级候选筛选器，在合并前过滤低贡献度候选以降低开销。实验结果表明，该框架在SIFT、Deep1B等数据集上，融合后召回率较单索引提升12%-20%，合并阶段延迟降低50%-65%，同时保持多索引检索的效率优势。

* [cs.DB] [**GPU-Accelerated Algorithms for Graph Vector Search: Taxonomy, Empirical Study, and Research Directions**](https://arxiv.org/abs/2602.16719)
  * [Graph ANNS & GPU Acceleration] 图基向量检索的GPU加速研究缺乏系统性梳理，现有工作分散且优化策略无统一分类，导致开发者难以选择适配的加速方案；同时，不同GPU架构下的算法性能表现差异显著，缺乏大规模实证对比。本文填补这一空白，核心贡献包括：构建首个图基向量检索GPU加速算法分类体系，从访存优化、并行粒度、计算模式三维度划分现有方法；基于NVIDIA Hopper/Ampere等多架构GPU，完成12种主流算法的大规模实证研究，量化不同优化策略的性能瓶颈；总结当前研究痛点并提出未来方向，包括跨架构适配、动态负载均衡、稀疏计算优化等。实验覆盖8个基准数据集，结果明确了各算法在不同数据规模与精度需求下的最优选择，为图基向量检索的GPU高效落地提供了权威参考与指导。