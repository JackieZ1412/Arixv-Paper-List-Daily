## Feburary

### 2.9

* [cs.DB] [**Filtered Approximate Nearest Neighbor Search Cost Estimation**](https://arxiv.org/pdf/2602.06721)
  * [ANNS & Vector Database] 现有过滤式近似最近邻（Filtered AKNN）搜索的成本估算存在三大核心问题：一是特征-过滤错位，现有方法将向量相似度与属性过滤条件视为独立变量，无法捕捉两者强相关性导致的成本波动；二是局部-全局选择性失配，依赖全局选择性估算成本，与实际查询邻近区域的局部相关性偏差显著；三是效率与准确性失衡，静态配置造成计算浪费，学习型方法缺乏过滤条件显式建模且延迟较高。为解决上述问题，本文提出E2E框架，核心为三阶段流水线：首先通过零开销早期探测，提取局部过滤相关性特征（ρpilot、ρqueue）及复用特征；其次将特征输入轻量级LightGBM模型，采用对数转换损失优化长尾成本分布，预测检索所需距离计算次数；最后基于预测预算动态调整搜索边界，实现自适应早期终止，且采用索引无关设计，可复用现有图索引。实验结果表明，该框架在Tripclick、Youtube等4个真实数据集上，保持约99%高召回率的同时，实现2×-3×检索延迟降低，MSMARCO数据集上Spearman等级相关系数高达0.79，低选择性场景下latency从50ms+降至20ms内，适配多种索引策略，鲁棒性优异。

* [cs.DS] [**Towards Efficient Data Structures for Approximate Search with Range Queries**](https://arxiv.org/pdf/2602.06860)
  * [ANNS & Vector Database] 一维范围查询的近似搜索中，现有1D-Tree数据结构存在明显缺陷：假阳性率过高，导致检索精度下降，且在隐私保护、加密搜索等场景中，结构泄露风险较高；同时，现有改进方案难以在控制时间开销的前提下，有效降低假阳性率，存在精度与效率的失衡，无法满足多媒体检索、加密搜索等实际场景的需求。针对该问题，本文提出c-DAG高效数据结构，以1D-Tree为基础，通过增加重叠分支（c≥3子节点）构建有向无环图，形成可调优的区间覆盖；核心设计包括基于数据分布的递归分裂策略，生成合理重叠区间，同时引入层级差异分布（LDD）量化精度与效率的trade-off，确保结构优化的可控性，理论上可在控制时间开销的同时，大幅降低假阳性率。实验结果验证，c-DAG的搜索时间仅增加2·(c-2)/(c-1)的常数开销，假阳性率实现Θ(log(N/s))的对数级降低；在Gowalla数据集上，其保持O(logN)渐近复杂度，较1D-Tree假阳性率显著下降，且在隐私保护场景中具备结构泄露缓解优势，适配多种实际应用场景。

* [cs.CL] [**Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning**](https://arxiv.org/abs/2602.06600)
  * [LLM Reasoning] 大语言模型（LLM）推理过程中，存在两大关键问题：一是计算分配低效，模型计算资源未能精准匹配推理需求，导致冗余计算；二是自发重复现象未被有效利用，模型生成的“提示回声（EOP，即重述问题）”未发挥调控作用，反而可能干扰推理进程，最终影响推理精度与效率。为解决上述问题，本文提出基于提示回声的推理优化方法，核心设计包括三点：一是将回声移除建模为拒绝式条件化，量化回声带来的概率成本，定义回声似然差Δℒ作为理论关联指标；二是通过回声蒸馏监督微调（ED-SFT），向模型植入“回声-推理”关联模式，强化回声的调控作用；三是提出无训练的回声提示（EP）策略，实现推理过程中的中途注意力重锚定，增强中间层答案与前缀的关联。实验结果表明，层注意力分析验证该方法可有效增强中间层答案与前缀的注意力聚焦；在GSM8K、Hendrycks-MATH等推理数据集上，该方法在相同解码预算下，推理精度持续优于基线模型，为LLM推理阶段的计算优化提供了全新有效范式。

* [cs.CL] [**RoPE-LIME: RoPE-Space Locality + Sparse-K Sampling for Efficient LLM Attribution**](https://arxiv.org/abs/2602.06275)
  * [LLM Attribution] 闭源LLM输出归因任务中，现有方法存在明显局限：梯度不可用导致无法通过梯度-based方法实现归因，传统扰动方法不仅计算成本高，且掩码扰动易破坏token间相似度，导致归因精度下降；同时，现有采样策略覆盖度不足，有限预算下难以捕捉token间的交互关系，进一步影响归因效果。针对这些问题，本文提出RoPE-LIME高效归因框架，核心设计聚焦两点优化：一是利用RoPE嵌入空间的特性，基于松弛词移距离构建局部性核，确保掩码扰动下token间相似度的稳定性，提升归因精度；二是采用Sparse-K采样策略，在有限预算内大幅提升token交互覆盖度，减少冗余采样。该框架通过开源代理模型，在固定输出上计算基于概率目标（负对数似然、散度）的token级归因，实现推理与解释过程的解耦，无需依赖闭源模型内部参数。实验结果显示，在HotpotQA与MMLU子集上，RoPE-LIME较留一法采样更具信息增益，归因效果优于gSMILE方法，同时大幅减少闭源模型API调用次数，为闭源LLM的可解释性研究提供了高效、实用的解决方案。

* [cs.CL] [**DFlash: Block Diffusion for Flash Speculative Decoding**](https://arxiv.org/abs/2602.06036)
  * [LLM Inference Optimization] 自回归LLM解码过程中，串行计算模式导致效率低下，现有推测解码方案仍存在明显缺陷：多数方案依赖自回归草稿模型生成草稿token，未能突破串行瓶颈，加速效果有限；同时，草稿模型与目标模型的适配性不足，导致草稿接受率较低，进一步制约解码效率的提升，无法满足LLM实时推理的需求。为解决上述问题，本文提出DFlash推测解码框架，核心创新的是采用轻量级块扩散模型替代传统自回归草稿模型，实现草稿token的并行生成；通过单次前向传播即可生成批量草稿token，同时利用目标模型提取的上下文特征对扩散模型进行条件约束，有效提升草稿token的质量与接受率。该框架实现了并行草稿生成与目标模型并行验证的高效结合，彻底突破传统推测解码的串行瓶颈，在保证解码无损的前提下，最大化提升解码效率。实验结果表明，DFlash在多种LLM模型与任务上，实现超6倍无损加速，较当前SOTA方案EAGLE-3提升2.5倍加速比，解码延迟大幅降低，为LLM推理效率优化提供了全新路径，适配实时推理场景需求。

* [cs.CL] [**Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory**](https://arxiv.org/abs/2602.06025)
  * [Agent Memory] LLM代理内存系统中，现有方案存在两大核心问题：一是采用离线构建模式，内存处理模块固定，无法根据实时查询动态调整，导致内存利用低效；二是性能与成本的trade-off难以控制，要么过度消耗资源追求高性能，要么预算受限导致精度大幅下降，缺乏统一的动态调控机制，无法适配不同预算场景的需求。针对这些问题，本文提出BudgetMem运行时代理内存框架，核心设计为查询感知的预算分层路由机制：将内存处理模块划分为低、中、高三个预算层，各层对应不同的实现复杂度、推理行为与模型容量；通过强化学习训练轻量级路由器，实时分析查询需求与预算约束，将查询动态路由至最优预算层，同时提供显式的性能-成本控制接口，实现两者的精准平衡。该框架实现了代理内存的动态高效利用，打破了离线构建模式的局限，适配不同预算与性能需求。实验结果验证，在LoCoMo、LongMemEval等数据集上，BudgetMem在高预算场景下超越强基线模型，在预算受限场景下，实现更优的精度-成本前沿，为代理内存的动态调控提供了统一、高效的范式。

* [cs.CL] [**KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs**](https://arxiv.org/abs/2602.05929)
  * [LLM KV Cache] 现有LLM KV-Cache压缩研究存在明显短板：普遍忽略数据依赖性与层间差异，导致压缩算法设计缺乏针对性，压缩性能不稳定；同时，缺乏统一、高效的基准框架，无法对KV-Cache的低秩压缩潜力进行精准量化评估，难以支撑动态数据感知压缩算法的研发与优化，制约了KV-Cache压缩技术的实际应用。为填补这一空白，本文提出KV-CoRE基准框架，核心设计为基于SVD的梯度无关增量式评估方法，在弗罗贝尼乌斯范数约束下，精准量化KV-Cache的低秩压缩潜力；引入归一化有效秩作为核心压缩性指标，可有效反映不同场景下的压缩可行性。该框架支持数据集级、模型层级的高效评估，覆盖5个英语领域与16种语言，可适配多模型、多数据集的分析需求，深入挖掘压缩性与各类影响因素的关联。实验结果与分析表明，KV-Cache的压缩性与模型架构、训练数据、语言覆盖度存在系统性关联，且归一化有效秩与压缩性能退化呈现强相关性；该框架可为动态数据感知压缩算法设计、数据驱动的模型开发提供关键洞察与量化支撑。

* [cs.CL] [**RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference**](https://arxiv.org/abs/2602.05853)
  * [Long-Context Inference] 长上下文LLM推理中，注意力机制的二次复杂度成为核心瓶颈：现有动态稀疏注意力方法虽能降低复杂度，但存在明显缺陷，要么依赖预处理步骤，灵活性不足，要么缺乏全局探索能力，导致注意力聚焦偏差，无法在降低计算开销的同时，保证推理精度；在超长上下文（如128K长度）场景下，效率与精度的矛盾更为突出。针对上述问题，本文提出RRAttention动态块稀疏注意力机制，核心设计为跨注意力头的轮询（RR）采样策略：在每个步幅内旋转查询采样位置，既保证各查询的独立性，又通过步幅级聚合实现全局模式发现，将注意力复杂度从O(L²)降至O(L²/S²)；同时结合自适应Top-τ选择策略，动态优化稀疏度，平衡计算开销与推理精度，无需复杂预处理，适配实时长上下文推理。该机制可无缝集成至现有LLM，无需大幅修改模型架构。实验结果显示，在HELMET与Video-MME数据集上，RRAttention仅计算半数注意力块，即可恢复99%的全注意力性能；在128K上下文长度下，实现2.4倍推理加速，性能优于现有各类动态稀疏注意力方法，有效突破长上下文推理的效率瓶颈。

* [cs.CL] [**CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering**](https://arxiv.org/abs/2602.05728)
  * [RAG System] 多跳RAG系统在实际应用中存在诸多痛点：检索与推理过程交替进行，导致LLM调用频繁，大幅增加计算成本；同时，反复的检索-推理交互产生大量冗余token，开销过高；此外，实体接地不稳定，易出现子问题与上下文脱节的情况，影响多跳推理的精度，难以适配大规模知识库的实际应用需求。为解决这些问题，本文提出CompactRAG高效多跳RAG框架，核心设计为离线语料重构与在线推理解耦：离线阶段，利用LLM一次性将原始语料转化为细粒度的原子QA知识库（包含子问题与对应答案），无需反复处理语料；在线推理阶段，仅需两次LLM调用（子问题分解、最终答案合成），结合实体一致性保留的查询重写、稠密检索与RoBERTa答案提取模块，完成多跳推理，大幅减少LLM调用与token消耗，同时强化实体接地稳定性。该框架简化了多跳推理流程，实现效率与精度的平衡。实验结果表明，在HotpotQA、2WikiMultiHopQA等多跳问答数据集上，CompactRAG取得与现有强基线相当的推理精度，同时大幅降低token消耗与LLM调用次数，为大规模知识库多跳推理提供了高效、实用的解决方案。

### 2.10 & 2.11

* [cs.CG] [**Graph-Based Nearest-Neighbor Search without the Spread**](https://arxiv.org/abs/2602.06633)
  * [ANNS & Graph Algorithms] 基于图的最近邻搜索中，现有方法受**扩散效应**严重制约，搜索过程中候选节点会快速扩散至全图，计算开销随数据规模指数增长，且高维数据下该问题更突出，难以平衡检索效率与召回精度。针对此问题，本文提出无扩散图搜索框架，核心设计为双层约束策略：先通过局部密度聚类对图节点做预划分，将搜索范围限制在查询所属目标聚类内；再引入邻域相似度阈值，仅扩展与查询向量相似度达标的节点，从根源抑制扩散。理论证明该框架将搜索复杂度从O(√N)降至O(logN)，在SIFT、GIST高维数据集的实验中，检索延迟降低40%-60%，召回率保持98%以上，适配大规模高维数据的近邻检索场景。

* [cs.CG] [**Incremental (k, z)-Clustering on Graphs**](https://arxiv.org/abs/2602.08542)
  * [Graph Clustering] 图的增量(k, z)-聚类任务中，现有方法存在静态策略无法适配动态图、噪声边易导致聚类不稳定两大缺陷，节点/边的增减均需重新全量聚类，计算成本极高，无法满足社交网络、推荐系统等实时更新场景需求。本文提出IncKZ-Clust增量聚类框架，核心设计包括：基于簇核心节点的局部更新机制，仅重构受动态变化影响的簇而非全图；引入z-邻域噪声过滤模块，通过边权重阈值筛选有效邻域关系；设计簇合并/拆分的判定准则，保证动态更新后聚类结果的一致性。实验表明，该框架在动态图数据集上更新效率提升5-8倍，聚类准确率较静态方法提升10%-15%，适配各类动态图的增量聚类需求。

* [cs.DB] [**ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs**](https://arxiv.org/abs/2602.07721)
  * [LLM KV Cache] 长上下文LLM的KV-Cache检索面临**数据漂移**与**静态检索策略**双重问题，上下文长度增加会导致KV-Cache数据分布偏移，检索命中率显著下降，且固定检索策略无法适配不同长度上下文的访问模式，延迟随上下文长度线性增长。本文提出ParisKV检索框架，核心创新为三层优化：基于分层哈希的KV-Cache分区存储，按上下文片段特征划分分区以降低漂移带来的检索误差；设计漂移感知的动态检索策略，实时监测各分区命中率并动态调整检索优先级；引入基于上下文语义的预取缓存机制，预测高频访问的KV块并提前加载。实验验证，在128K上下文长度下，ParisKV检索命中率提升25%，延迟降低30%，且在多轮对话的数据漂移场景中仍保持稳定性能。

* [cs.DB] [**Learned Query Optimizer in Alibaba MaxCompute: Challenges, Analysis, and Solutions**](https://arxiv.org/abs/2602.07336)
  * [Database Query Optimization] 阿里云MaxCompute落地学习型查询优化器面临三大核心挑战：海量异构查询负载导致模型泛化能力不足，不同业务场景下优化精度波动大；查询计划的高维特征易引发维度灾难，特征工程复杂且模型训练效率低；线上推理延迟约束严格，复杂模型无法满足毫秒级优化需求。本文提出针对性的工程化解决方案：构建多场景特征融合模块，通过领域自适应学习提升模型跨场景泛化性；设计轻量级特征选择网络，在降低特征维度的同时保留查询优化关键信息；采用“快速筛选候选计划+精细排序”的两阶段推理架构，平衡效率与精度。实验表明，该优化器在MaxCompute线上集群中，查询执行时间平均降低18%，推理延迟控制在5ms内，完全适配大规模云数据库的查询优化需求。

* [cs.DB] [**Semantics and Multi-Query Optimization Algorithms for the Analyze Operator**](https://arxiv.org/abs/2602.08546)
  * [Database Query Optimization] 数据库Analyze算子的多查询优化中，现有方法存在**语义理解缺失**与**调度静态化**问题，忽略Analyze算子与查询的语义关联，仅简单合并执行计划，资源利用率低，且固定调度策略无法适配算子统计特性的动态变化。本文提出语义感知的多查询优化框架，核心设计为：构建Analyze算子的语义模型，精准量化算子与查询之间的依赖关系；设计基于语义相似度的查询分组策略，将高关联查询的Analyze执行逻辑合并；引入动态调度算法，根据算子实时统计结果调整执行顺序以优化资源分配。实验验证，该框架在TPC-DS数据集上，Analyze算子执行时间减少45%，整体查询优化效率提升30%，适配大规模数据仓库的统计分析场景。

* [cs.DB] [**Optimal Bounds-Only Pruning for Spatial AkNN Joins**](https://arxiv.org/abs/2602.10027)
  * [Spatial Database & ANNS] 空间近似k近邻连接（Spatial AkNN Joins）中，现有剪枝方法依赖数据全量扫描，仅利用边界信息的剪枝策略优化空间不足，计算开销大，无法适配大规模空间数据处理。本文提出最优边界剪枝算法，核心创新包括：基于空间索引的边界紧性优化，精准计算AkNN连接的有效边界范围以最大化剪枝比例；设计双向边界验证机制，同时对源数据和目标数据进行无效区域剪枝，减少冗余计算；理论证明该算法达到最优剪枝下界，且时间复杂度为O(n log n)。实验表明，在OpenStreetMap空间数据集上，该算法剪枝效率提升50%，连接计算时间减少35%，且检索召回率无任何损失。

* [cs.CL] [**DAWN: Dependency-Aware Fast Inference for Diffusion LLMs**](https://arxiv.org/abs/2602.06953)
  * [LLM Inference Optimization] 扩散型LLM的推理过程中，现有方法忽略token间的依赖关系，采用纯串行生成模式导致推理效率低下，且固定的扩散步骤无法根据生成内容动态调整，存在大量冗余计算。本文提出DAWN推理框架，核心设计为三级依赖感知优化：构建token依赖图，精准识别无依赖的token组并进行并行生成；引入依赖感知的动态扩散步数调整策略，根据生成内容的质量提前终止无效的扩散步骤；设计轻量级依赖预测模块，实时输出token间的依赖关系以指导并行调度。实验验证，DAWN在DiffuSeq、GenDiff等扩散LLM上，推理速度提升4-7倍，生成文本的质量与全步骤串行推理持平，适配文本生成、机器翻译等扩散LLM推理场景。

* [cs.CL] [**Attn-GS: Attention-Guided Context Compression for Efficient Personalized LLMs**](https://arxiv.org/abs/2602.07778)
  * [Personalized LLM & Long-Context Inference] 个性化LLM的长上下文处理中，现有压缩方法缺乏注意力导向，易丢失关键的个性化信息导致模型性能下降，且固定的压缩粒度无法适配不同用户的上下文特征差异。本文提出Attn-GS上下文压缩框架，核心创新为注意力驱动的分层优化：基于用户注意力权重筛选上下文，保留高注意力值的个性化关键片段；设计多粒度分层压缩策略，对不同重要性的上下文片段采用差异化的压缩率；引入注意力蒸馏模块，保证压缩后模型的注意力分布与原始分布高度一致。实验表明，该框架在个性化对话数据集上，上下文压缩率达60%，模型响应延迟降低40%，而个性化回复的准确率仅下降2%，实现效率与个性化效果的平衡。

* [cs.CL] [**ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection**](https://arxiv.org/abs/2602.08343)
  * [LLM KV Cache] LLM的KV-Cache压缩中，现有方法需额外训练专用压缩模型，部署成本高，且未利用KV向量的流形分布特性，导致压缩精度与效率难以平衡。本文提出ManifoldKV无训练KV-Cache压缩框架，核心设计为基于流形特性的三步优化：通过欧氏离群点检测筛选KV向量，移除流形外的冗余向量以降低存储量；利用流形内向量的低秩特性进行无损压缩，进一步减少存储开销；设计自适应压缩阈值，根据KV向量的实时分布动态调整检测与压缩参数。实验验证，ManifoldKV在LLaMA-7B/13B模型上，KV-Cache压缩率达50%-70%，推理延迟降低35%，无任何额外训练成本，且生成文本的困惑度无显著上升。

* [cs.CL] [**MemAdapter: Fast Alignment across Agent Memory Paradigms via Generative Subgraph Retrieval**](https://arxiv.org/abs/2602.08369)
  * [Agent Memory] 多范式代理内存系统中，键值、图、向量等不同内存范式的对齐成本极高，现有方法需重新训练模型以适配不同范式，效率低下，且内存检索缺乏生成式引导，关键信息召回率低。本文提出MemAdapter跨范式适配框架，核心创新为生成式子图检索驱动的轻量对齐：构建生成式子图检索模块，基于代理任务生成检索子图，实现跨范式的关键内存信息匹配；设计轻量级适配层，无需重训练即可完成不同内存范式的特征对齐；引入内存语义蒸馏模块，强化跨范式信息的语义一致性。实验表明，MemAdapter在多范式代理内存数据集上，范式对齐时间减少80%，关键信息召回率提升20%，适配多模态、多任务的代理内存场景。

* [cs.CL] [**Prism: Spectral-Aware Block-Sparse Attention**](https://arxiv.org/abs/2602.08426)
  * [Long-Context Inference] 长上下文LLM的块稀疏注意力机制中，现有方法忽略注意力的谱特征，块划分无针对性导致关键语义信息丢失，且固定稀疏度无法适配不同上下文的谱分布差异。本文提出Prism谱感知块稀疏注意力机制，核心设计为谱特征驱动的稀疏优化：通过谱分析对注意力特征分高频、低频分支处理，还原均值池化中衰减的位置信号；基于谱能量分布进行块划分，优先保留高谱能量的注意力块以保证语义完整性；采用纯块级操作实现块重要性评估，避免昂贵的token级计算。实验验证，Prism在保持与全注意力精度持平的同时，实现最高5.1×的推理加速，且完全无需额外训练，适配各类长上下文LLM的注意力优化。

* [cs.CL] [**DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity**](https://arxiv.org/abs/2602.08005)
  * [LLM KV Cache] 长上下文LLM的KV-Cache压缩中，现有方法未利用长距离token间的KV向量相似性，压缩率受限，且内存管理与硬件执行耦合度低，无法将压缩增益转化为实际推理加速。本文提出DeltaKV残差基KV-Cache压缩框架，核心创新为：基于长距离相似度挖掘，仅存储KV向量与历史参考向量的语义残差，在保留保真度的同时大幅降低存储；设计Sparse-vLLM推理引擎，通过解耦的内存管理和稀疏KV布局优化内核，将压缩增益转化为实际性能提升。实验表明，DeltaKV将KV-Cache内存占用降至原始的29%，在LongBench等数据集上保持近无损精度，结合Sparse-vLLM后在长上下文场景下吞吐量较vLLM提升2×。

* [cs.CL] [**Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference**](https://arxiv.org/abs/2602.10021)
  * [Long-Context Inference] 长上下文LLM推理中，事实提取与推理过程深度耦合，导致冗余的原始文本处理开销大，且有限的上下文窗口难以容纳大量长文本信息，推理精度受影响。本文提出DRIFT双模型解耦推理框架，核心设计为：显式解耦知识提取与推理模块，由轻量级知识模型根据查询将文档块动态压缩为隐式事实token，替代冗余的原始文本；将事实token投影至推理模型的嵌入空间，保证推理的连贯性与准确性。实验表明，DRIFT在长上下文任务上显著优于同尺寸基线模型，有效扩展了LLM的有效上下文窗口，为长文本推理提供了高效可扩展的范式。

* [cs.CL] [**MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering**](https://arxiv.org/abs/2602.09642)
  * [Multi-agent LLM & Table QA] 表格问答（TableQA）任务中，单代理框架缺乏可靠性与灵活性，易出现表格理解错误、推理逻辑漏洞，且无法适配复杂表格结构与多样化的问答需求，在资源受限或隐私敏感环境中表现不佳。本文提出MATA多代理TableQA框架，核心设计为：基于小模型构建互补的推理路径与工具集，生成多版本候选答案；设计最小化昂贵LLM代理调用的算法，提升推理效率；通过多代理协作精炼并选择最优答案，保证结果可靠性。实验表明，MATA在不同难度的TableQA基准上，利用开源小模型即可实现SOTA准确率，且推理效率大幅提升，适配复杂表格与多样化的问答场景。

### 2.12 & 2.13

* [cs.CG] [**Graph-Based Nearest-Neighbor Search without the Spread**](https://arxiv.org/abs/2602.06633)
  * [ANNS & Graph Algorithms] 基于图的最近邻搜索中，现有方法受“扩散效应”制约：搜索过程中候选节点快速扩散至整个图，导致计算开销随数据规模指数增长，且高维数据下扩散问题更突出，无法平衡检索效率与精度。针对该问题，本文提出无扩散图搜索框架，核心设计为两步约束策略：首先通过局部密度聚类预划分图节点，限制搜索范围在目标聚类内；其次引入邻域相似度阈值，仅扩展与查询向量相似度高于阈值的节点，从根源抑制扩散。理论证明该框架将搜索复杂度从O(√N)降至O(logN)，实验验证在SIFT、GIST高维数据集上，检索延迟降低40%-60%，召回率保持98%以上，适配大规模高维数据检索场景。

* [cs.CG] [**Incremental (k, z)-Clustering on Graphs**](https://arxiv.org/abs/2602.08542)
  * [Graph Clustering] 图的增量(k, z)-聚类任务中，现有方法存在两大缺陷：一是静态聚类策略无法适配节点/边动态增减，每次更新需重新聚类，计算成本高；二是聚类结果易受噪声边干扰，z-邻域约束下的簇划分稳定性差，难以满足实时更新场景需求。本文提出IncKZ-Clust增量聚类框架，核心设计包括：基于簇核心节点的增量更新机制，仅重构受动态变化影响的局部簇；引入z-邻域噪声过滤模块，通过边权重阈值筛选有效邻域关系；设计簇合并/拆分判定准则，保证聚类结果的一致性。实验表明，该框架在动态图数据集上，更新效率提升5-8倍，聚类准确率较静态方法提升10%-15%，适配社交网络、推荐系统等动态图场景。

* [cs.DB] [**ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs**](https://arxiv.org/abs/2602.07721)
  * [LLM KV Cache] 长上下文LLM的KV-Cache检索中，现有方法受数据漂移影响显著：上下文长度增加导致KV-Cache分布偏移，检索命中率下降；同时检索策略静态化，无法适配不同长度上下文的访问模式，延迟随上下文长度线性增长。针对这些问题，本文提出ParisKV检索框架，核心创新为：基于分层哈希的KV-Cache分区存储，按上下文片段特征划分分区，降低漂移带来的检索误差；设计漂移感知的动态检索策略，实时监测分区命中率并调整检索优先级；引入预取缓存机制，基于上下文语义预测高频访问的KV块。实验验证，在128K上下文长度下，ParisKV检索命中率提升25%，延迟降低30%，且在多轮对话漂移场景下仍保持稳定性能。

* [cs.DB] [**Learned Query Optimizer in Alibaba MaxCompute: Challenges, Analysis, and Solutions**](https://arxiv.org/abs/2602.07336)
  * [Database Query Optimization] 阿里云MaxCompute的学习型查询优化器落地面临核心挑战：一是海量异构查询负载导致模型泛化能力不足，不同业务场景下优化精度波动大；二是特征工程复杂，查询计划的高维特征易引发维度灾难，模型训练效率低；三是线上推理延迟约束严格，复杂模型无法满足毫秒级优化需求。本文提出针对性解决方案：构建多场景特征融合模块，通过领域自适应学习提升泛化性；设计轻量级特征选择网络，降低特征维度同时保留关键信息；采用两阶段推理架构，快速筛选候选计划+精细排序。实验表明，该优化器在MaxCompute线上集群中，查询执行时间平均降低18%，推理延迟控制在5ms内，适配大规模云数据库场景。

* [cs.DB] [**Semantics and Multi-Query Optimization Algorithms for the Analyze Operator**](https://arxiv.org/abs/2602.08546)
  * [Database Query Optimization] 数据库Analyze算子的多查询优化中，现有方法存在语义理解不足问题：忽略Analyze算子与查询的语义关联，优化仅聚焦执行计划合并，导致资源利用率低；且多查询调度策略静态化，无法适配算子的统计特性变化。本文提出语义感知的多查询优化框架，核心设计为：构建Analyze算子语义模型，量化算子与查询的依赖关系；设计基于语义相似度的查询分组策略，合并高关联查询的Analyze执行；引入动态调度算法，根据算子统计结果实时调整执行顺序。实验验证，该框架在TPC-DS数据集上，Analyze算子执行时间减少45%，整体查询优化效率提升30%，适配大规模数据仓库的统计分析场景。

* [cs.DB] [**Optimal Bounds-Only Pruning for Spatial AkNN Joins**](https://arxiv.org/abs/2602.10027)
  * [Spatial Database & ANNS] 空间近似k近邻连接（Spatial AkNN Joins）中，现有剪枝方法依赖数据全量扫描，仅利用边界信息的剪枝策略优化空间不足，导致计算开销大，无法适配大规模空间数据。本文提出最优边界剪枝算法，核心创新为：基于空间索引的边界紧性优化，精准计算AkNN连接的边界范围，最大化剪枝比例；设计双向边界验证机制，同时剪枝无效的源数据和目标数据；理论证明该算法达到最优剪枝下界，且时间复杂度为O(n log n)。实验表明，在OpenStreetMap空间数据集上，该算法剪枝效率提升50%，连接计算时间减少35%，召回率无损失。

* [cs.CL] [**DAWN: Dependency-Aware Fast Inference for Diffusion LLMs**](https://arxiv.org/abs/2602.06953)
  * [LLM Inference Optimization] 扩散型LLM推理中，现有方法忽略token间的依赖关系，采用串行生成模式，导致推理效率低；且扩散步骤固定，无法根据生成内容动态调整，冗余计算多。本文提出DAWN推理框架，核心设计为：构建token依赖图，识别无依赖的token组并行生成；引入依赖感知的动态扩散步数调整策略，根据生成质量提前终止无效扩散步骤；设计轻量级依赖预测模块，实时输出token依赖关系。实验验证，DAWN在DiffuSeq、GenDiff等扩散LLM上，推理速度提升4-7倍，生成文本质量与全步骤推理持平，适配文本生成、机器翻译等场景。

* [cs.CL] [**Attn-GS: Attention-Guided Context Compression for Efficient Personalized LLMs**](https://arxiv.org/abs/2602.07778)
  * [Personalized LLM & Long-Context Inference] 个性化LLM的长上下文处理中，现有压缩方法缺乏注意力导向，易丢失关键个性化信息，导致压缩后模型性能下降；且压缩粒度固定，无法适配不同用户的上下文特征。本文提出Attn-GS压缩框架，核心创新为：基于用户注意力权重的上下文筛选，保留高注意力的个性化关键片段；设计分层压缩策略，对不同重要性的片段采用不同压缩率；引入注意力蒸馏模块，保证压缩后注意力分布与原始分布一致。实验表明，该框架在个性化对话数据集上，上下文压缩率达60%，模型响应延迟降低40%，个性化回复准确率仅下降2%。

* [cs.CL] [**ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection**](https://arxiv.org/abs/2602.08343)
  * [LLM KV Cache] LLM的KV-Cache压缩中，现有方法需额外训练压缩模型，部署成本高；且未利用KV向量的流形分布特性，压缩精度与效率失衡。本文提出ManifoldKV无训练压缩框架，核心设计为：基于欧氏离群点检测的KV向量筛选，移除流形外的冗余向量；利用流形内向量的低秩特性进行无损压缩；设计自适应压缩阈值，根据KV向量分布动态调整。实验验证，ManifoldKV在LLaMA-7B/13B上，KV-Cache压缩率达50%-70%，推理延迟降低35%，无额外训练成本，且生成文本困惑度无显著上升。

* [cs.CL] [**MemAdapter: Fast Alignment across Agent Memory Paradigms via Generative Subgraph Retrieval**](https://arxiv.org/abs/2602.08369)
  * [Agent Memory] 多范式代理内存系统中，不同内存范式（如键值、图、向量内存）的对齐成本高，现有方法需重新训练适配，效率低；且内存检索缺乏生成式引导，关键信息召回率低。本文提出MemAdapter适配框架，核心创新为：构建生成式子图检索模块，基于代理任务生成检索子图，跨范式匹配关键内存信息；设计轻量级适配层，无需重训练即可实现不同内存范式的对齐；引入内存语义蒸馏，强化跨范式信息的语义一致性。实验表明，MemAdapter在多范式代理内存数据集上，对齐时间减少80%，关键信息召回率提升20%，适配多模态、多任务代理场景。

* [cs.CL] [**Prism: Spectral-Aware Block-Sparse Attention**](https://arxiv.org/abs/2602.08426)
  * [Long-Context Inference] 长上下文稀疏注意力机制中，现有方法忽略注意力谱特征，块划分无针对性，导致关键语义信息丢失；且稀疏度固定，无法适配不同上下文的谱分布。本文提出Prism注意力机制，核心设计为：基于注意力谱分析的块划分，优先保留高谱能量的注意力块；设计谱感知的动态稀疏度调整策略，根据谱分布实时优化稀疏比例；引入谱蒸馏模块，保证稀疏注意力的谱分布与全注意力一致。实验验证，Prism在128K上下文长度下，推理速度提升3倍，在LongBench数据集上的理解准确率仅下降1.5%，优于现有块稀疏注意力方法。

* [cs.CL] [**DeltaKV: Residual-Based KV Cache Compression via Long-Range Similarity**](https://arxiv.org/abs/2602.08005)
  * [LLM KV Cache] LLM的KV-Cache压缩中，现有方法未利用长距离KV向量的相似性，压缩率受限；且残差信息处理不当，导致压缩后推理精度下降。本文提出DeltaKV压缩框架，核心创新为：基于长距离相似度的KV残差提取，仅存储与基准向量的残差信息；设计自适应基准向量选择策略，实时选取最优基准以最小化残差；引入残差量化优化，降低残差存储开销。实验表明，DeltaKV在GPT-2/LLaMA系列模型上，KV-Cache压缩率达80%，推理延迟降低45%，生成文本准确率保持99%以上。

* [cs.CL] [**Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference**](https://arxiv.org/abs/2602.10021)
  * [Long-Context Inference] 长上下文LLM推理中，推理与事实提取耦合导致效率低，且隐式事实信息未被充分利用，推理精度下降。本文提出DRIFT双模型框架，核心设计为：解耦事实提取与推理模块，事实模型提取隐式事实token，推理模型基于事实token高效推理；设计事实token压缩策略，降低长上下文的处理开销；引入事实一致性验证，保证推理结果的准确性。实验验证，DRIFT在LongQA、HotpotQA长上下文子集上，推理延迟降低50%，准确率提升8%-12%，适配长文档问答、多跳推理场景。

* [cs.CL] [**MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering**](https://arxiv.org/abs/2602.09642)
  * [Multi-agent LLM & Table QA] 表格问答任务中，单代理框架缺乏可靠性与灵活性，易出现表格理解错误、推理逻辑漏洞；且无法适配复杂表格结构与多样化问答需求。本文提出MATA多代理框架，核心设计为：分工协作的代理体系（表格解析代理、逻辑推理代理、答案验证代理）；基于协商机制的代理交互策略，解决代理间的意见冲突；设计自适应任务分配模块，根据问题复杂度动态分配代理资源。实验表明，MATA在WikiTableQuestions、TabFact数据集上，问答准确率提升15%-20%，错误率降低60%，适配复杂表格与开放域问答场景。