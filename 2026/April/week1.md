### 3.30
* [cs.CL] MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization
  * [LLM Agent Memory & Lifelong Personalization] 现有大模型智能体的用户记忆评测多集中在短对话、单领域场景，缺乏对终身跨域、长时程用户记忆能力的系统性评估。本文提出 MemoryCD 基准，聚焦终身跨域个性化场景，构建包含多领域用户偏好、长期交互历史、渐进式信息更新的评测数据集；设计记忆准确性、偏好一致性、抗干扰性、跨域泛化四大维度指标，全面评估 LLM 智能体的长上下文用户记忆能力。同时基于该基准评测了主流智能体架构，发现现有记忆机制在长期信息遗忘、跨域偏好冲突、噪声干扰上存在显著短板，为下一代长时用户记忆系统提供评测标准与改进方向。

### 3.31
* [cs.IR] [**GAAMA: Graph Augmented Associative Memory for Agents**](https://arxiv.org/abs/2603.27910)
  * [Agent Associative Memory & Graph Augmentation] 多会话交互的AI智能体长时记忆易丢失结构关联，现有图基记忆方法存在中心节点主导检索、分层推理能力弱的问题。本文提出GAAMA图增强关联记忆系统，通过三步流水线构建概念介导的分层知识图谱：保留原始对话的逐字片段、提取原子事实与主题级概念节点、合成高阶反思信息；设计四种节点类型（片段、事实、反思、概念）与五种结构边，以概念节点提供跨域遍历路径。检索时融合余弦相似度K近邻搜索与边类型感知的个性化PageRank，在LoCoMo-10基准上取得78.9%的平均奖励，显著优于调优RAG与多款主流记忆基线模型。

* [cs.IR] [**The Price of Meaning: Why Every Semantic Memory System Forgets**](https://arxiv.org/abs/2603.27116)
  * [Semantic Memory & Forgetting Mechanism] 主流AI语义记忆系统通过语义组织实现泛化与概念检索，但该特性背后存在固有遗忘与干扰代价。本文针对语义连续核阈值记忆系统做形式化证明，得出四大核心结论：语义有用的表示具有有限有效秩、有限局部维度会导致检索邻域存在竞争信息、内存增长时记忆保留率趋于0、阈值调优无法消除联想诱导的错误回忆。在向量检索、图记忆、注意力上下文等五大架构上的验证表明，纯语义系统均存在遗忘与错误回忆问题，推理增强系统仅能将平滑退化转化为灾难性失效，而无干扰的系统需牺牲语义泛化能力。

* [cs.LG] [**HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention**](https://arxiv.org/abs/2603.28458)
  * [Sparse Attention & Hierarchical Indexing] 细粒度稀疏注意力的索引器需遍历全部历史token，存在O(L²)层瓶颈，随上下文长度增长性能急剧下降。本文提出HISA分层索引稀疏注意力，将扁平token扫描转化为两阶段分层检索：先通过块级粗过滤对池化块表征打分并剪枝无关区域，再在候选块内执行原始token级索引；该方法可直接替换现有索引器，无需额外训练，且保留精确的top-k稀疏模式。核级基准测试中，32K上下文实现2倍加速、128K上下文实现4倍加速，在DeepSeek-V3.2上的实测与原DSA精度高度匹配，token选择集的平均交并比超99%。

* [cs.LG] [**IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression**](https://arxiv.org/abs/2603.28430)
  * [KV Cache Compression & Hardware-Aligned Rotation] 正交特征解相关的量化方法存在O(d²)存储与计算开销，现有3D旋转变换的硬件对齐性差、局部混合能力有限。本文提出IsoQuant基于SO(4)等斜旋转的分块旋转框架，将4D块表示为四元数并采用闭式变换，设计全旋转与快速旋转两种变体及2D轻量版本；在d=128维度下，全旋转将前向旋转计算量从RotorQuant的2408次FMA降至1024次，快速旋转进一步降至512次。在18种CUDA融合配置下，平均核级加速4.5-4.7倍，峰值超6倍，且重建均方误差与基线相当。

* [cs.LG] [**KVSculpt: KV Cache Compression as Distillation**](https://arxiv.org/abs/2603.27819)
  * [KV Cache Compression & Distillation] 现有KV缓存序列长度压缩方法局限于选择或合并原始缓存条目，难以最大化压缩效率。本文提出KVSculpt，将缓存压缩转化为蒸馏问题，在连续嵌入空间优化无约束的小尺寸KV对以保留各层注意力行为：通过L-BFGS优化Key，最小二乘闭式求解Value，二者交替迭代；同时引入自适应预算分配，通过轻量试点压缩为各层和KV头重新分配压缩资源。在Qwen2.5-1.5B-Instruct的2048token上下文实验中，KL散度较Select+Fit方法降低3.5-4.1倍，自适应分配额外实现1.3倍降幅，且无推理开销；实验还发现各层压缩难度差异可达100倍，单层内KV头差异最高467倍。

* [cs.LG] [**KV Cache Quantization for Self-Forcing Video Generation: A 33-Method Empirical Study**](https://arxiv.org/abs/2603.27469)
  * [Video Generation & KV Cache Quantization] 自驱动视频生成的KV缓存随生成长度线性增长，内存瓶颈显著，现有量化方法缺乏针对该场景的系统性评测。本文基于Wan2.1自驱动框架，对33种量化与缓存策略变体开展实证研究，涵盖610次提示级观测与63次基准汇总，从峰值显存、运行时、压缩比、成像质量等多维度评估。得出三大核心结论：FlowCache启发的软剪枝INT4适配是最优实用方案，实现5.42-5.49倍压缩且显存从19.28GB降至11.7GB；PRQ_INT4等高精度方法因运行时或内存成本过高不适合部署；单纯压缩不足够，部分方法因重建/保留BF16缓冲区仍超基线显存。

* [cs.LG] [**TurboAngle: Near-Lossless KV Cache Compression via Uniform Angle Quantization**](https://arxiv.org/abs/2603.27467)
  * [KV Cache Compression & Angle Quantization] 为实现KV缓存的近无损压缩，本文提出TurboAngle，在快速沃尔什-哈达玛域对角度进行量化，通过随机对角旋转使连续元素对在单位圆上近似均匀分布；引入每层早增强机制，为各层独立配置K/V码本大小，为模型关键层分配更高精度。在7个1B-7B参数模型上，每层早增强在4个模型上实现无损压缩，6个模型达到近无损，每元素量化为3.28-3.67个角度比特；非对称范数量化（Key8比特、Value4比特对数空间）在Mistral-7B上实现每元素6.56总比特，困惑度仅上升0.0014且无需校准数据；层组敏感性分析发现模型存在Key/Value主导层与负迁移层（提高精度反而降低性能）。

* [cs.LG] [**ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference**](https://arxiv.org/abs/2603.27138)
  * [KV Cache Offloading & CPU-GPU Collaboration] 长上下文LLM推理的KV缓存卸载至DRAM时，易因GPU-CPU数据传输频繁、CPU计算负载过高导致GPU利用率低下。本文提出ScoutAttention高效KV缓存卸载框架，设计GPU-CPU协同的分块稀疏注意力大幅降低CPU负载；创新层前瞻CPU预计算算法，让CPU提前一层启动注意力计算，并结合异步周期性召回机制维持极小CPU计算负载。实验表明，该方法精度与基线的偏差不超过2.4%，推理速度较现有卸载方法提升2.1倍，已被DAC 2026接收。

* [cs.CL] [**Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers**](https://arxiv.org/abs/2603.26380)
  * [Hybrid Transformer & Dynamic Attention Routing] 标准全注意力随序列长度呈二次复杂度，滑动窗口注意力以牺牲感受野为代价提升效率，现有混合注意力采用静态交替模式，计算分配灵活性差。本文提出Switch Attention动态细粒度混合注意力，为每层每个token动态选择全注意力分支（全局信息聚合）或滑动窗口分支（高效局部模式匹配）；设计自适应正则化目标提升模型效率，并通过持续预训练将全注意力架构迁移至混合架构。在23个基准数据集的4K常规与32K长上下文实验中，该方法在效率与性能上实现良好平衡。

* [cs.CL] [**ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory**](https://arxiv.org/abs/2603.26182)
  * [Clinical Decision Making & Multi-Agent Dual-Memory] LLM在临床诊断中难以实现复杂非线性推理，现有方法缺乏假设驱动的迭代推理过程。本文提出ClinicalAgents多智能体临床决策框架，采用蒙特卡洛树搜索实现动态编排，让协调器迭代生成假设、验证证据并在信息缺失时回溯；核心设计双记忆架构：可变工作记忆维护演化的患者状态，静态经验记忆通过主动反馈环检索临床指南与历史病例。实验表明，该框架的诊断准确率与可解释性显著优于单智能体与主流多智能体基线。

* [cs.AI] [**Codebase-Memory: Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via MCP**](https://arxiv.org/abs/2603.27277)
  * [Code Exploration & Knowledge Graph] LLM代码智能体探索代码库时依赖文件读取与字符搜索，token消耗大且缺乏结构理解。本文提出Codebase-Memory开源系统，基于模型上下文协议（MCP）构建Tree-Sitter驱动的持久化知识图谱，通过多阶段流水线支持66种编程语言，包含并行工作池、调用图遍历、影响分析与社区发现。在31个真实仓库上的评估表明，该系统的答案质量达83%（文件探索智能体为92%），但token消耗减少90%、工具调用次数减少52%；针对中心节点检测、调用者排序等图原生查询，在19种语言上匹配或超越文件探索智能体。

### 4.1
* [cs.OS] [**StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving**](https://arxiv.org/abs/2603.28795)
  * [LLM Serving & Step-Level Cache Reuse] 现有LLM缓存多聚焦前缀KV复用，对推理步骤级别的重复计算利用不足。本文提出StepCache步骤级缓存复用框架，在解码阶段缓存完整推理子步骤，通过轻量级校验判断可复用性，对不匹配部分仅做选择性补丁修正，而非完全重算。在多轮对话、工具调用、链式推理场景下显著降低重复计算，提升服务吞吐量，同时保证输出一致性，适合高并发LLM推理服务部署。

* [cs.CL] [**MemFactory: Unified Inference & Training Framework for Agent Memory**](https://arxiv.org/abs/2603.29493)
  * [Agent Memory & Unified Train-Infer Framework] 当前智能体内存模块训练与推理割裂，记忆写入、检索、更新机制难以端到端优化。本文提出MemFactory统一框架，将记忆表征、记忆门控、检索打分、遗忘策略全部纳入可训练流程，支持从训练到推理的无缝迁移；内置记忆蒸馏、记忆对齐、稀疏检索等通用组件，可快速适配不同智能体架构，显著提升长期记忆稳定性与检索准确率。

* [cs.CL] [**MemRerank: Preference Memory for Personalized Product Reranking**](https://arxiv.org/abs/2603.29247)
  * [Personalized Reranking & User Preference Memory] 传统商品重排序依赖短期会话特征，缺乏长期用户偏好记忆，个性化能力有限。本文提出MemRerank偏好记忆重排模型，构建用户长期行为记忆库，动态捕捉细粒度偏好迁移与隐式兴趣；在重排阶段将记忆特征与当前会话融合，显著提升点击率与个性化匹配度，在电商推荐类任务上取得稳定收益。

* [cs.CV] [**Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism**](https://arxiv.org/abs/2603.29252)
  * [Long Video Understanding & Visual Memory] 多模态大模型在长视频理解中受限于视觉上下文长度，易丢失关键时序信息。本文引入视觉记忆机制，对长视频帧序列进行结构化记忆压缩，保留关键帧、物体轨迹与事件时序关系；通过记忆检索式注意力替代全帧注意力，在不显著增加计算量的前提下将有效视频理解长度大幅扩展，提升长视频问答、叙事与细节追踪能力。

非常抱歉，是格式解析出了问题，我完全按你**最初原始格式**重新生成，纯文本链接，不做任何跳转处理：

### 4.2

* [cs.DB] [**Making Array-Based Translation Practical for Modern, High-Performance Buffer Management**](https://arxiv.org/abs/2604.00423)
  * [Buffer Management & Array-Based Translation] 现代数据库缓冲池需适配扫描密集型分析、向量检索等多样化负载，但现有逻辑页到物理帧的转换机制难以兼顾低开销、大页兼容与细粒度管理。本文提出Calico缓冲池框架，基于数组映射翻译机制，通过三级优化突破传统限制：多级翻译与路径缓存适配稀疏分层页标识符，内存打孔回收冷数据翻译内存，组预取挖掘并行性；该框架解耦逻辑翻译与OS页表，支持DBMS完全控制驱逐与I/O策略。在PostgreSQL及pgvector上的验证表明，向量检索场景内存内加速3.9倍、超内存场景加速6.5倍，扫描密集查询加速3倍，性能匹配或超越现有SOTA。

* [cs.DB] [**Fiber-Navigable Search: A Geometric Approach to Filtered ANN**](https://arxiv.org/abs/2604.00102)
  * [Filtered ANNS & Geometric Navigation] 元数据过滤后的近邻图（纤维子图）连通性与几何特性易发生突变，导致传统检索算法性能下降。本文提出几何导向的纤维导航检索框架，设计两阶段搜索策略：当局部几何特征有利时，结合全图探索与过滤邻居下降；通过局部信号将检索失败分为拓扑切割、几何折叠、真实盆地三类，并引入轻量级锚点结构定位纤维存在簇，实现针对性重启。实验表明，该方法在过滤检索任务上性能超越FAISS HNSW，且三类失败模式随过滤选择性呈现可预测变化。

* [cs.LG] [**Temporal Memory for Resource-Constrained Agents: Continual Learning via Stochastic Compress-Add-Smooth**](https://arxiv.org/abs/2604.00067)
  * [Continual Learning & Temporal Memory] 资源受限智能体需在固定内存预算下持续学习，避免遗忘旧经验。本文提出基于随机过程的时间记忆框架，将记忆建模为[0,1]区间上的桥扩散过程，终端边缘分布编码当前状态，中间分布编码历史经验；通过“压缩-添加-平滑”三步递归整合新经验，无需反向传播与数据存储。在d维高斯混合模型上验证，该框架时间复杂度为O(LKd²)（L为分段数、K为混合分量数），记忆保留半衰期与L线性相关，且能生成时间相干的历史“叙事回放”，为持续学习提供可解析的数学模型。

* [cs.AI] [**OmniMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory**](https://arxiv.org/abs/2604.01007)
  * [Multimodal Agent Memory & Autoresearch] 终身智能体的多模态记忆设计空间庞大，手动探索效率低下。本文提出OmniMem框架，通过自主研究流水线自动发现最优记忆架构：内循环无需人工干预，执行约50次实验，诊断失效模式、提出架构修改并修复数据管道漏洞。从初始基线（LoCoMo F1=0.117）出发，最终在LoCoMo上F1提升411%（至0.598），在Mem-Gallery上提升214%（至0.797）；关键发现显示，漏洞修复（+175%）、架构变更（+44%）与提示工程（特定类别+188%）的贡献远超超参数调优，验证了自主研究在复杂AI系统设计中的优势。

* [cs.CL] [**Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation**](https://arxiv.org/abs/2604.00131)
  * [Agent Memory Control & Decay-Driven Activation] 现有记忆增强LLM智能体依赖“始终开启”的检索与扁平存储，导致历史增长时干扰加剧、延迟升高。本文提出Oblivion记忆控制框架，模拟人类选择性遗忘机制，通过衰减驱动的可访问性降低实现遗忘，而非显式删除；框架分离读写路径：读路径基于智能体不确定性与内存缓冲充足度决定是否访问记忆，避免冗余检索；写路径强化对响应有贡献的记忆。在长视野交互基准上的实验表明，Oblivion能动态适配记忆访问与强化策略，在上下文变化中平衡学习与遗忘，提升智能体推理有效性。

* [cs.CL] [**LinearARD: Linear-Memory Attention Distillation for RoPE Restoration**](https://arxiv.org/abs/2604.00004)
  * [RoPE Scaling & Attention Distillation] 大模型上下文窗口扩展常通过RoPE缩放与轻量持续预训练实现，但易导致短文本任务性能退化。本文提出LinearARD线性内存注意力蒸馏方法，通过冻结原生RoPE教师模型与RoPE缩放学生模型的注意力结构一致性，恢复学生模型性能；该方法对齐Q/Q、K/K、V/V自相关矩阵的行分布，而非隐藏状态，并引入线性内存核，利用逐token对数求和指数统计量与反向传播中的对数重新计算，规避二次内存瓶颈。在LLaMA2-7B（4K扩展至32K）上验证，仅用425万训练token（LongReD需2.56亿）即可恢复98.3%的短文本性能，且在长上下文基准上超越现有SOTA。

### 4.3

* [cs.OS] [**HACache: Leveraging Read Performance with Cache in a Heterogeneous Array**](https://arxiv.org/abs/2604.01655)
  * [Heterogeneous Storage & Cache Scheduling] 针对异构存储阵列（HDD + SSD 等多层介质）的读性能瓶颈，提出 HACache 异构缓存架构，通过动态感知不同存储层的带宽与延迟特征，智能调度热点数据在各级缓存间迁移；统一缓存管理接口，最大化利用各层硬件特性，显著提升混合负载下的吞吐与平均读延迟，尤其适合数据库、大数据分析等密集 I/O 场景。

* [cs.IR] [**STABLE: Efficient Hybrid Nearest Neighbor Search via Magnitude-Uniformity and Cardinality-Robustness**](https://arxiv.org/abs/2604.01617)
  * [Hybrid ANN Search & Vector Distribution Robustness] 现有向量检索在向量模长差异大、聚类分布不均时召回率急剧下降。本文提出 STABLE 混合检索框架，基于模长均匀化（Magnitude-Uniformity）与基数鲁棒性（Cardinality-Robustness）设计量化与图索引结合策略，缓解向量分布偏移带来的检索退化，在高方差真实向量数据集上实现更高效率与稳定性。

* [cs.DB] [**BBC: Improving Large-k Approximate Nearest Neighbor Search with a Bucket-based Result Collector**](https://arxiv.org/abs/2604.01960)
  * [Large-k ANN & Result Aggregation] 当需要返回大规模近邻结果（large-k）时，传统 ANN 索引存在候选聚合低效、重复计算、内存占用高等问题。提出 BBC 桶式结果收集器，通过分桶管理候选集、增量合并与去重剪枝，大幅提升大规模 k 值下的检索效率，降低延迟与内存开销，适配推荐、批量检索等场景。

* [cs.CL] [**DeltaMem: Towards Agentic Memory Management via Reinforcement Learning**](https://arxiv.org/abs/2604.01560)
  * [Agent Memory & RL-Based Control] 传统智能体内存管理依赖人工规则（相似度、时间衰减等），难以自适应复杂任务流。本文提出 DeltaMem，使用强化学习动态决策记忆的写入、检索、更新与遗忘，根据任务反馈学习最优内存策略；通过增量更新（delta update）减少开销，在多轮交互、长时序任务中显著提升记忆利用率与任务完成率。

* [cs.AI] [**Novel Memory Forgetting Techniques for Autonomous AI Agents: Balancing Relevance and Efficiency**](https://arxiv.org/abs/2604.02280)
  * [Agent Memory Forgetting & Relevance-Efficiency Balance] 自主智能体在持续交互中内存会无限膨胀，导致检索变慢、噪声增加。系统性提出一系列新型遗忘机制：分层遗忘、重要性衰减遗忘、冲突记忆修剪等，在保留高相关记忆的同时控制内存规模；实验验证可有效缓解记忆膨胀，同时维持任务性能不下降。

* [cs.AI] [**Hierarchical Memory Orchestration for Personalized Persistent Agents**](https://arxiv.org/abs/2604.01670)
  * [Persistent Agent & Hierarchical Memory] 个性化持久智能体需要长期记忆用户偏好，但单层记忆结构难以兼顾细粒度与检索速度。提出分层记忆编排架构，分为瞬时工作记忆、短期会话记忆、长期用户档案记忆三层，设计跨层信息同步与优先级调度机制，实现稳定个性化与高效检索的统一。

* [cs.AI] [**ContextBudget: Budget-Aware Context Management for Long-Horizon Search Agents**](https://arxiv.org/abs/2604.01664)
  * [Long-Horizon Agent & Context Budget Control] 长视野搜索智能体在多步推理中上下文长度极易超限，导致截断丢失关键信息。提出 ContextBudget 上下文预算管理机制，根据 token 预算动态选择最有价值的历史信息、中间结果与检索片段，在有限上下文窗口内最大化信息效用，提升复杂推理与多跳搜索任务效果。

* [cs.AI] [**Exploring Robust Multi-Agent Workflows for Environmental Data Management**](https://arxiv.org/abs/2604.01647)
  * [Multi-Agent & Environmental Data] 面向环境数据管理（气象、水文、遥感等多源异构时序数据），设计鲁棒的多智能体协作工作流；智能体分工负责数据清洗、特征提取、异常检测、趋势预测，通过可靠内存共享与冲突消解机制提升整体系统稳定性与预测精度。

* [cs.AI] [**ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context**](https://arxiv.org/abs/2604.01599)
  * [Agent-Native Memory & LLM-Curated Context] 提出 ByteRover 原生智能体内存系统，由 LLM 直接对历史交互、知识片段做精细化分层整理，构建层级化上下文结构；摒弃简单的相似度检索，采用结构化导航式记忆访问，大幅提升长对话、复杂任务规划中的理解连贯性与决策可靠性。