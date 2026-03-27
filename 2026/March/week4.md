### 3.23
* [cs.DB] [**Low-Latency Stateful Stream Processing through Timely and Accurate Prefetching**](https://arxiv.org/abs/2603.19890)
  * [Stateful Stream Processing & Prefetching] 有状态流处理引擎因数据通路与状态I/O强耦合，状态访问处于关键路径导致CPU阻塞、延迟上升。本文提出**Keyed Prefetching**，在上游算子提前提取未来访问键，在元组到达前主动将对应状态预取到内存，让I/O与计算重叠以隐藏大状态访问延迟；搭配**Timestamp-Aware Caching**缓存淘汰策略，统一管理已访问与预取条目，高效利用内存。两项技术结合可降低长运行实时查询延迟，且不牺牲吞吐量，已收录于**ICDE 2026**。

* [cs.DB/LG/IR] [**A Super Fast K-means for Indexing Vector Embeddings**](https://arxiv.org/abs/2603.20009)
  * [Vector Clustering & Index Acceleration] 面向高维向量嵌入聚类提出**SuperKMeans**，通过可靠剪枝无用维度减少数据访问与计算开销，在CPU上比FAISS、Scikit-Learn快**最高7倍**，GPU上比cuVS快**最高4倍**，且保持检索任务所需的聚类中心质量。创新提出**Early Termination by Recall**机制，在中心质量不再提升时提前终止迭代，进一步缩短运行时间且不损失检索效果，已开源实现。

* [cs.OS] [**2DIO: A Cache-Accurate Storage Microbenchmark**](https://arxiv.org/abs/2603.19971)
  * [Storage Benchmark & Cache Behavior] 现有存储基准工具只能生成行为规整的缓存命中率曲线，无法复现真实系统的性能悬崖与平台区。本文提出**2DIO**缓存精准存储微基准，用精简参数三元组编码工作负载，同时捕获短期 recency 与长期 frequency；可生成可调的复杂缓存行为，参数可移植、可扫描，能精准复现真实负载特征，将发表于**EuroSys 2026**。

* [cs.LG] [**The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference**](https://arxiv.org/abs/2603.19664)
  * [KV Cache & Residual Stream] 本文从理论与实验证明：**Transformer 每一层的 K/V 完全由残差流(residual stream)唯一确定**，从残差向量重计算 K/V 可做到**比特级完全一致**，无任何近似误差。基于该结论提出**KV-Direct**方案：只存残差向量（单token约5KB）而非完整KV（约136KB），按需重计算，峰值内存可从103MB降至42MB；在同等缓存预算下保持**100% token匹配**，远超H2O、StreamingLLM、SnapKV、TOVA等基线（均降至5–28%），中等批量下重计算甚至比读缓存快**最高5倍**。

### 3.24-3.25
* [cs.OS] [**GateANN: I/O-Efficient Filtered Vector Search on SSDs**](https://arxiv.org/abs/2603.21466)
  * [Filtered ANNS & SSD I/O Optimization] 现有SSD上的过滤式向量检索要么后过滤造成大量无效I/O，要么需要重建过滤感知索引开销过高。本文提出GateANN，将图遍历与向量读取解耦，遍历仅需邻居列表与近似距离，无需全精度向量；通过**图隧道（graph tunneling）**在内存中先完成过滤谓词检查，非匹配节点完全在内存中路由，不触发SSD读。实验将SSD读取减少最高10倍，吞吐量提升最高7.6倍。

* [cs.DB] [**GEM: A Native Graph-based Index for Multi-Vector Retrieval**](https://arxiv.org/abs/2603.20336)
  * [Multi-Vector Retrieval & Graph Index] 多向量检索无法直接复用单向量索引，会丢失细粒度语义且效率低。本文提出原生图索引GEM，直接在向量集上构建邻近图，保留多向量语义；先做集合级聚类降冗余，再在簇内建局部图并全局连通；针对非度量特性解耦建图度量与最终相关性，注入语义捷径。查询时多入口束搜索+簇提示剪枝+量化距离估计，速度最高提升16倍，精度持平或更优，已收录 **SIGMOD 2026**。

* [cs.IR] [**Rethinking Retrieval-Augmentation as Synthesis: A Query-Aware Context Merging Approach**](https://arxiv.org/abs/2603.20286)
  * [RAG & Context Merging] 标准RAG的retrieve-then-select会截断长尾关键证据，同时在高分块上浪费token。本文提出MergeRAG，将检索增强重定义为信息密度最大化的合成问题，通过双路径机制重构上下文：对称合并恢复弱信号与桥接证据，非对称合并用熵导向锚点消除冗余；搭配层次并行合并减少信息损失。F1最高提升13.7，EM提升11.5。

* [cs.DC] [**PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving**](https://arxiv.org/abs/2603.23049)
  * [RAG Serving & KV Cache Reuse] RAG预填充阶段计算量大，KV缓存复用受限于命中率低、CPU-GPU传输开销大、SSD I/O慢。本文提出PCR，三项核心优化：前缀树缓存+前瞻LRU淘汰提升命中率；层间重叠流水线化KV加载与GPU计算隐藏通信延迟；基于队列的预取将KV从SSD提前载入DRAM。平均TTFT最高加速2.47倍。

* [cs.DC] [**ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression**](https://arxiv.org/abs/2603.17435)
  * [LLM Inference & Lossless Compression] 传统无损压缩破坏SIMT并行度，导致推理变慢。本文提出ZipServ，张量核感知的三位图编码（TCA-TBE）实现并行常数时间解码；融合解压缩-GEMM核（ZipGEMM）直接在张量核寄存器中解压缩，消除中间缓冲区。模型体积减少30%，核级加速最高2.21倍，端到端平均加速1.22倍（vs vLLM），已收录 **ASPLOS 2026**。

* [cs.DB] [**FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search**](https://arxiv.org/abs/2603.21710)
  * [ANNS & Graph Index Merging] 分布式向量库合并、读写分离场景需要高效合并多个图索引，现有方法重构开销大。本文提出FGIM框架：邻近图转k-NNG提取候选邻居→k-NNG精炼补高质量邻居→转回邻近图提升导航性。接入HNSW等主流图索引，比HNSW增量构建最高快3.5倍，对无增量支持的方法平均快7.9倍，检索性能持平或更优，已收录 **SIGMOD 2026**。

* [cs.DS] [**Fast Nearest Neighbor Search for $\ell_p$ Metrics**](https://arxiv.org/abs/2603.21148)
  * [NNS Theoretical Bound & $\ell_p$ Metric] 针对$\ell_p$（p>2）度量空间，给出随机化数据结构，实现$p^{O(1)+\log\log p}$近似比，查询时间$\text{poly}(d\log n)$，空间$\text{poly}(dn)$，在快速查询场景下优于现有SOTA结果，为高维近似检索提供理论下界与算法支撑。

* [cs.CL] [**EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction**](https://arxiv.org/abs/2603.22910)
  * [KV Cache Compression & Similarity Reconstruction] 现有低秩压缩是不可逆变换，无法在内存充足时切回全精度。本文提出EchoKV，利用层间/层内注意力头相似性，用轻量网络从部分子集重建残差KV组件，支持标准/压缩推理按需切换；两阶段微调训练成本极低（7B模型约1 A100小时）。在LongBench、RULER上超过同类压缩方法，短上下文保持高吞吐。

* [cs.CL] [**An experimental study of KV cache reuse strategies in chunk-level caching systems**](https://arxiv.org/abs/2603.20218)
  * [Chunk-Level Caching & KV Reuse] 块级缓存（CLC）预计算检索块KV加速推理，但忽略块间交叉注意力导致质量下降。本文通过大量实验证明现有CLC方法存在本质局限，并发现不同技术之间互补，进而提出组合式设计，在不显著增加开销的前提下获得更高准确率。

* [cs.CV] [**WorldCache: Content-Aware Caching for Accelerated Video World Models**](https://arxiv.org/abs/2603.22286)
  * [Video World Model & Feature Caching] 视频世界模型（DiT）推理成本高，现有特征缓存复用会造成重影、模糊。本文提出WorldCache感知感知动态缓存，引入运动自适应阈值、显著加权漂移估计、混合与扭曲最优近似、阶段感知阈值调度，无需训练即可实现运动一致的特征复用。在Cosmos-Predict2.5-2B上加速2.3倍，保留99.4%质量。

* [cs.LG] [**KV Cache Optimization Strategies for Scalable and Efficient LLM Inference**](https://arxiv.org/abs/2603.20397)
  * [KV Cache Survey & Deployment Guidance] 系统性综述KV缓存优化五大方向：缓存驱逐、压缩、混合内存、新注意力机制、组合策略；分析每种方法的机制、权衡与内存/吞吐/精度表现；映射到7类实际部署场景（长上下文、高吞吐服务、边缘、多轮对话等），给出可落地选型指南，指出自适应多阶段流水线是未来方向。


### 3.26
* [cs.DB] [**PDET-LSH: Scalable In-Memory Indexing for High-Dimensional Approximate Nearest Neighbor Search with Quality Guarantees**](https://arxiv.org/abs/2603.24920)
  * [High-Dimensional ANNS & LSH] 针对高维向量检索场景，现有LSH方法在可扩展性与检索质量间难以平衡。本文提出PDET-LSH，基于概率密度估计树构建内存可扩展索引，在保持理论近似保证的同时，大幅提升高维数据下的检索效率；通过自适应分区与密度感知哈希，减少哈希碰撞带来的精度损失，在亿级高维向量上实现比传统LSH更高召回与更低内存开销。

* [cs.DB] [**TaCo: Data-adaptive and Query-aware Subspace Collision for High-dimensional Approximate Nearest Neighbor Search**](https://arxiv.org/abs/2603.24919)
  * [Subspace ANNS & Query Awareness] 高维数据中有效信息常集中在局部子空间，全局哈希效率低下。本文提出TaCo数据自适应、查询感知的子空间碰撞机制，根据数据分布动态选择判别性子空间进行哈希，并针对不同查询调整哈希策略；利用子空间碰撞实现高效候选过滤，在多个高维基准上显著优于固定LSH与全空间检索方法。

* [cs.CL] [**Adaptive Chunking: Optimizing Chunking-Method Selection for RAG**](https://arxiv.org/abs/2603.25333)
  * [RAG Chunking & Adaptive Selection] 固定分块策略无法适配不同文档结构与查询类型，导致RAG检索效果不稳定。本文提出自适应分块框架，根据文档结构（文本、表格、图表）与查询意图动态选择最优分块方法（固定长度、语义、标题层次、递归分块等）；通过轻量级评估器在线打分选择策略，在通用RAG基准上持续提升检索准确率与上下文利用率。

* [cs.CL] [**MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens**](https://arxiv.org/abs/2603.23516)
  * [Extreme Long Context & Sparse Attention] 现有长上下文模型难以高效扩展到100M token级别，内存与计算开销呈指数增长。本文提出MSA内存稀疏注意力，将外部记忆库与注意力机制深度结合，采用记忆感知稀疏路由与分层检索式注意力；在保持端到端可微的同时，支持单序列扩展至100M token，且计算与内存开销接近线性增长。

* [cs.CL] [**Evaluating Chunking Strategies For Retrieval-Augmented Generation in Oil and Gas Enterprise Documents**](https://arxiv.org/abs/2603.24556)
  * [Domain-Specific RAG & Chunking Evaluation] 针对油气行业企业文档（含公式、表格、专业术语、层次化结构），系统评估主流分块策略在RAG中的实际效果。实验表明固定长度分块在专业领域文档中表现最差，**语义分块+层次标题分块**组合最适合工业文档；同时给出领域适配的分块参数建议与评估指标体系，为垂直领域RAG工程落地提供参考。