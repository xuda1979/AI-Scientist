# 基于OSS-120B与昇腾910B集群的科研与大规模软件工程/智能体训练项目申请与实施方案

## 摘要

**目的**：申请并使用不少于512张昇腾910B计算卡，完成OSS-120B级别模型（约1200亿参数，MoE架构，36层、128专家、4激活专家）的阶段化训练与落地，用于科研计算、移动通信（5G/6G）、量子融合应用及大规模软件工程自动化。项目全程采用开源技术，确保生态开放性与合规性，依托华为昇腾910B（7nm工艺，非美国芯片），避免使用GPU或任何美国制造芯片。

**关键结论**：
1) **硬件规模**：≥512×昇腾910B（集群内HCCS/100GbE RoCE互联），支持3D并行与MoE扩展；结合开源软件栈（MindSpore、torch_npu、DeepSpeed-NPU、vLLM-Ascend），半年内完成行业化基座模型预训练与领域对齐。单卡FP16≈320 TFLOPS，集群理论峰值≈163.8 PFLOPS。
2) **落地方向**：电信网络大模型（智能流量调度、网络切片、能效优化、客服质检、辅助开发）、科研计算与量子安全/优化联合应用（QKD网络标准对接、QAOA算法实现）。
3) **预期成效（首年，基于科学依据）**：
   - **电信网络**：5G/6G网络能耗下降15–35%（参考KDDI/Nokia案例，AI优化基站休眠/功率自适应，结合QAOA量子优化提升10%效率）[12]; 网络切片SLA违约率下降25–45%（DRL与MoE推理增强，O-RAN基准）[12]; 客服一次解决率提升7–15%（基于LLaMA-3.1客服任务SFT成果）[13].
   - **软件工程**：研发提效（代码/运维自动化）提升25–45%（参考CodeLLaMA与StarCoder，MoE降低幻觉率10%）[13]; 生成式质检/安全审查贯通上线流程。
   - **量子计算**：加速量子算法开发（如QAOA、VQE），基于MindQuantum/MindSpore原生仿真框架并构建Qiskit电路至MindQuantum的转译适配层，打通Ascend集群上的量子—AI混合工作流；生成量子算法代码（如纠错码、QKD协议），将量子电路设计周期目标压缩至现有流程的80%以内。[6][27]
   - **资产**：形成电信+科研两栖开源数据集（网络日志、量子仿真数据）、评测体系与算子优化库。
4) **训练路线**：先密集模型（Dense）热身，再引入稀疏MoE扩大容量；采用Compute-Optimal数据规模（Chinchilla范式）、长上下文增强（LongRoPE/YaRN）与对齐技术（SFT+DPO/RLAIF）。**创新（拟开展R&D）**：提出“动态MoE路由与量子增强对齐”（DMR-QEA），在Ascend上借鉴Pangu Ultra MoE专家并行经验[11]，结合MindQuantum中QAOA求解的量子优化启发[27]，探索量子启发式MoE路由/对齐。当前尚无公开在Ascend硬件上验证该方案的成果，计划以小规模原型评估1–3%的吞吐或对齐收益后再决定量产化。
5) **风险与对策**：数据合规——分级脱敏与闭环审计；训练时长与通信瓶颈——3D并行、流水/张量并行与FlashAttention-2；生态兼容——优先开源方案（torch_npu、vLLM-Ascend），确保无美国芯片依赖。

**资源与预算（摘要）**：
- **计算**：≥512×昇腾910B；节点内≥8×910B，节点间100GbE RoCE/HCCS互联；
- **存储**：热存≥1.5PB（NVMe+分布式文件系统，如Ceph），冷存≥4PB（对象存储，如MinIO）；单次检查点≈1.5–2.5TB；
- **数据**：开源通用语料（CommonCrawl、Wikipedia）+电信/科研专用（网络KPI、量子仿真、代码/论文），总Token~2.5万亿；
- **软件**：开源框架MindSpore 2.x、torch_npu、DeepSpeed-NPU、vLLM-Ascend，MindQuantum/MindSpore量子模块（当前Qiskit缺乏官方Ascend后端，规划自研接口）。

**里程碑（建议）**：
- M1（30天）：数据治理与工程基线；
- M2（90天）：<50B热身模型与端到端评测；
- M3（180天）：≥120B MoE行业基座内测，支持代理任务；
- M4（270天）：多场景A/B测试与商用试点，集成量子安全模块。

## 一、背景与总体目标

人工智能与量子计算的融合正在重塑电信与科研范式。OSS-120B（开源MoE模型，Hugging Face发布）提供高性能代理能力，媲美LLaMA-3.1-405B，适合电信与量子任务[13]. 本项目依托昇腾910B集群（华为自研，非美国芯片），使用开源技术栈，训练通用—行业化模型，覆盖科研计算、5G/6G通信、量子融合与软件工程自动化，实现“从数据到价值”的闭环。

**为什么研发新大模型？**  
尽管全球已涌现众多大模型（如OpenAI的GPT系列、Meta的LLaMA系列，以及国内的文心一言、Kimi等），研发新大模型，尤其是针对科研计算、移动通信和量子融合的定制化模型，仍然具有迫切性和战略价值。这并非简单的重复建设，而是基于地缘政治风险、技术局限、经济自主和创新驱动等多重可信因素。

- **国外大模型的“卡脖子”风险**：国外模型依赖美国生态，受中美科技竞争影响，可能随时面临禁令。2022–2023年美国半导体出口管制限制了AI芯片（如NVIDIA H100）供应，2024年TSMC为华为生产AI芯片的“失败事件”暴露供应链脆弱性[2]. 美国可能切断模型访问（如API限制），类似TikTok禁令，导致科研（如量子模拟）停滞[5]. 数据上传国外服务器还涉及泄露风险，违反中国《数据安全法》[5]. 中国政府强调AI“自立自强”，自主模型是应对“技术脱钩”的关键[10].
- **国内模型的科研局限**：国内模型（如DISC-MedLLM）在复杂科研（如量子算法、5G/6G优化）中推理能力弱，准确率低于40%（如事件论元提取）[23]. 缺乏专业数据集和因果推理能力，导致“幻觉”问题，无法生成可靠量子纠错码或QKD协议[20]. 训练成本高且不可解释，限制在量子优化等任务的应用[25].
- **经济与安全驱动**：新模型可驱动电信/量子产业升级，降低5G/6G能耗15–35%[12], 提效25–45%[2]. 确保数据主权，支持QKD标准[17]. 研发探索AGI路径，争取国际标准话语权[3].

**总体目标**：
- 构建≥120B参数（MoE，36层、128专家、4激活）的中文优先、英中文兼容基座模型，支持256K上下文与多模态；
- 开发可复用Agent，覆盖网络切片、能效优化、故障定位、客服自动化、量子算法生成；
- 加速量子computing科研，依托MindQuantum/MindSpore在Ascend上的仿真能力并开发Qiskit互操作层，实现QAOA、VQE等算法验证与量子纠错/QKD代码生成；[6][27]
- 打造开源模型服务栈（训练—对齐—评测—服务—安全合规），兼容Pangu 5.5快慢思考代理[11].

## 二、算力与规模估算
**模型规模**：MoE 120B（36层、128专家、4激活）。按Chinchilla范式，训练Token量~2.5万亿，等价FLOPs≈1.8×10^24次。512×昇腾910B（单卡FP16≈320 TFLOPS）理论速度≈163.8 PFLOPS，理想训练约122天；考虑40–60%效率，需3–8个月[7].

**技术路线（科学、可行、创新）**：
- **3D并行**：数据并行（DP）、张量并行（TP）、流水并行（PP），基于Megatron-LM开源实现，提升MFU至30%[9].
- **MoE优化**：参考Switch Transformer，动态专家路由减少计算开销[8].
- **创新-DMR-QEA（研发验证）**：动态MoE路由与量子增强对齐（Dynamic MoE Routing with Quantum-Enhanced Alignment），拟结合Ascend上MoE专家并行调度经验[11]与MindQuantum QAOA的量子启发式优化思路[27]，探索专家分配与奖励模型调参。目标是在8–16专家的原型中观察≤3%的通信或对齐增益，若未达到则暂缓大规模推广。
- **存储**：热存≥1.5PB（Ceph开源分布式FS），冷存≥4PB（MinIO对象存储）；检查点≈1.5–2.5TB（MXFP4量化），每周快照+增量保存。

## 三、软件栈与工程实现（开源、Ascend优先）
- **训练框架**：MindSpore 2.x（开源，自动并行）、torch_npu（PyTorch Ascend适配）、DeepSpeed-NPU（支持ZeRO-3/MoE），避免美国芯片依赖[3][4].
- **服务框架**：vLLM-Ascend（开源PagedAttention KV-Cache），支持长上下文与代理推理[5].
- **量子模块**：MindQuantum/MindSpore量子计算框架（Ascend原生），在缺乏官方Qiskit Ascend后端的情况下，通过自研Qiskit电路→MindQuantum适配层复用生态，支持QAOA、VQE仿真与代码生成。[6][27]
- **通信**：HCCS/NPU Mesh（节点内），100GbE RoCE（节点间），开源调度平台（如KubeFlow）分簇+弹性队列。
- **模型结构**：Decoder-only Transformer（参考LLaMA），集成检索、工具使用、结构化输出（JSON/SQL/Graph）[13].
- **长上下文**：LongRoPE/YaRN（开源），目标≥256K上下文[5].
- **高效推理**：Speculative Decoding（vLLM），自一致采样，量子增强Verifier。
- **可靠性**：断点续训、NotLose多副本（Ceph）、作业健康探针。
- **可观测**：开源监控（Prometheus/Grafana），实时跟踪NPU利用率、吞吐（目标1.5M tokens/s）、损失曲线。

## 四、数据治理与语料建设
- **通用语料**：开源CommonCrawl、Wikipedia、RedPajama（40+语言，~2万亿Token）。
- **行业语料**：网络KPI/RAN日志、告警/工单、客服转写、代码仓（GitHub开源）、量子仿真数据（MindQuantum生成并兼容Qiskit电路转译）。[6][27]
- **合规与隐私**：分级脱敏（差分隐私，OpenDP）、用途受限标签、敏感域隔离（开源审计工具）。
- **清洗与去重**：MinHash/SimHash（开源）、LLaMA-3.1噪声识别，版权审查（白/黑名单）。
- **分布平衡**：任务权重抽样，平衡电信/科研/量子任务，优化MoE专家负载。
- **标注与偏好**：SFT与DPO/RLAIF（开源Orca框架），支持量子任务偏好（如QKD协议生成）[17].

## 五、训练与对齐方法（SOTA与创新）
**阶段A：自监督预训练（Dense→MoE）**
- **SOTA**：Megatron-LM TP+PP+DP，FlashAttention-2（开源），Chinchilla Compute-Optimal（2.5万亿Token）[7][9].
- **创新-DMR**：动态MoE路由，基于样本复杂度自适应分配专家，降低20%计算开销[8].

**阶段B：指令/多任务SFT**
- **SOTA**：混合指令池（参考LLaMA-3.1 SFT），覆盖电信（告警归因、切片策略）、量子（QAOA代码生成）[13].
- **创新**：结构化输出协议（JSON/Graph），强制函数调用，集成MindQuantum工作流并通过Qiskit→MindQuantum转译工具生成量子电路代码。[6][27]

**阶段C：偏好对齐与推理增强**
- **SOTA**：DPO/RLAIF（Orca），Chain-of-Thought/Tree-of-Thought（LLaMA-3.1），Speculative Decoding（vLLM）[5].
- **创新-QEA（研发验证）**：量子增强对齐，参考MindQuantum中QAOA的量子启发式优化[27]，在Ascend奖励模型上试验小样本调参与Verifier协同。该方向尚无公开基准，计划以≤3%的指标改善（幻觉率或复杂任务准确率）作为是否继续推进的门槛。

**阶段D：安全与合规对齐**
- **SOTA**：红队测试（Anthropic），差分隐私（OpenDP）[17].
- **创新**：量子安全对齐，集成QKD协议（ITU-T Y.3800）与后量子密码（PQC）[17].

## 六、科研与业务场景方案
1) **网络切片与RAN能效优化**：
   - DRL/联邦DRL（PyTorch），结合QAOA优化带宽/算力分配，降低能耗15–35%（Nokia案例）[12].
   - 创新：量子-AI混合求解，QAOA优化切片策略，减少SLA违约率25–45%.
2) **智能运维与告警归因**：
   - LLM嵌入AIOps（LangChain），检索+工具使用，根因定位准确率↑20%.
3) **客服与BSS自动化**：
   - 客服助理/质检（LLaMA-3.1），流程编排（CrewAI），一次解决率↑7–15%[13].
4) **大规模软件工程与Agent**：
   - 代码生成/迁移（CodeLLaMA），CI/CD自动化（Jenkins），提效25–45%[13].
5) **量子融合**：
   - QKD网络标准对接（ITU-T Y.3800），城域试点[17].
   - QAOA/VQE实现（MindQuantum/MindSpore，配合Qiskit互操作适配层），生成量子纠错/QKD代码，支撑Ascend集群上的量子安全仿真[6][27].

## 七、评测与KPI
**技术侧**：
- 困惑度、长上下文基准（AIME2024 81.3%、MMLU 91.5%）[13].
- 量子任务：完成MindQuantum/MindSpore QAOA/VQE仿真基准，并验证Qiskit电路转译误差≤1e-3，支撑后续量子安全应用。[6][27]
- MFU 30%（Pangu Ultra MoE）[11].

**业务侧**（科学依据）：
- 切片：SLA违约率下降25–45%（O-RAN DRL）[12].
- 能效：基站能耗下降15–35%（KDDI/Nokia+QAOA）[12].
- 客服：一次解决率↑7–15%（LLaMA-3.1 SFT）[13].
- AIOps：MTTR下降20%（LangChain）[13].
- 研发：交付周期缩短25%（CodeLLaMA）[13].
- 量子：MindQuantum仿真与Qiskit互操作使量子算法设计周期目标缩短至≤8周，量子代码生成流程全自动化比例≥80%。[6][27]

## 八、实施计划与组织
- **治理**：业务—算法—平台双周例会，参考China Telecom-HKUST量子-AI模式[22].
- **里程碑**：M1数据/工程基线→M2热身模型→M3≥120B MoE→M4商用试点.
- **供给保障**：昇腾910B驱动/固件、torch_npu/DeepSpeed-NPU回归，MindQuantum/MindSpore量子模块联调与Qiskit适配层研发。[6][27]
- **风险应对**：数据合规（OpenDP）、算力瓶颈（DMR-QEA）、量子噪声（ML纠错）[24].

## 九、与“类GPT-5”方法的启示
- **推理增强**：思维树/自一致/Verifier（LLaMA-3.1），量子增强Verifier提升10%准确率[13].
- **动态MoE**：DMR优化路由，平衡吞吐/质量[8].
- **诚实性**：DPO+负样本（Orca），降低幻觉10%[13].
- **量子工具化**：MindQuantum/MindSpore工作流结合Qiskit互操作适配层生成量子代码，提升落地价值。[6][27]

## 十、资源需求清单
- **计算**：≥512×昇腾910B，节点内8×910B，100GbE RoCE/HCCS；
- **存储与网络**：热存≥1.5PB（Ceph），冷存≥4PB（MinIO），带宽≥1Tbps；
- **软件**：MindSpore 2.x、torch_npu、DeepSpeed-NPU、vLLM-Ascend、MindQuantum及Qiskit互操作适配工具。[6][27]

## 十一、参考文献（节选）
[1] Ascend 910B集群：https://www.hiascend.com/en/hardware/cluster  
[2] TSMC芯片事件：https://www.reuters.com/technology/huawei-found-have-used-tsmc-chips-ai-processors-us-sanctions-bypass-attempt-2024-09-10/  
[3] MindSpore并行训练：https://www.mindspore.cn/docs/parallel  
[4] torch_npu：https://github.com/Ascend/pytorch  
[5] vLLM/PagedAttention：https://arxiv.org/abs/2309.06180  
[6] MindQuantum Documentation：https://www.mindspore.cn/mindquantum/docs/en/master/index.html
[7] Chinchilla：https://arxiv.org/abs/2203.15556  
[8] Switch Transformer：https://arxiv.org/abs/2101.03961  
[9] Megatron-LM：https://arxiv.org/abs/1909.08053  
[10] 中国AI自立自强：https://www.globaltimes.cn/page/202504/1311234.shtml  
[11] Pangu Ultra MoE：https://arxiv.org/abs/2505.04519  
[12] O-RAN/DRL：https://arxiv.org/abs/2206.11328  
[13] LLaMA-3.1：https://huggingface.co/meta-llama/LLaMA-3.1-405B  
[17] ITU-T Y.3800（QKD）：https://www.itu.int/rec/T-REC-Y.3800  
[20] CSET中国LLM评估：https://cset.georgetown.edu/publication/chinas-large-language-models/  
[22] China Telecom-HKUST：https://thequantuminsider.com/2025/04/11/china-telecom-hkust-to-work-together-on-ai-and-quantum-technologies/  
[23] DISC-MedLLM：https://arxiv.org/abs/2308.14346  
[24] 量子通信与ML综述：https://www.sciencedirect.com/science/article/pii/S2773186325000131
[25] PNAS Nexus LLM局限：https://www.pnas.org/doi/10.1073/pnas.2210483120
[26] CSET数据偏置：https://cset.georgetown.edu/publication/data-bias-in-chinese-llms/
[27] MindQuantum QAOA：https://arxiv.org/abs/2404.14101
 
