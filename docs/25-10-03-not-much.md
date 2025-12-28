---
companies:
- anthropic
- x-ai
- google
- google-labs
- openai
- arena
- epoch-ai
- mit
- luma
- akhaliq
date: '2025-10-03T05:44:39.731046Z'
description: '**Anthropic** 宣布了新任 CTO。前沿编程智能体（Coding Agents）迎来更新：**Claude Sonnet 4.5**
  在网络安全和用户体验（UX）方面表现强劲，但在编程能力上仍落后于 **GPT-5 Codex**。**xAI 的 Grok Code Fast** 声称能以更低的成本实现更高的代码编辑成功率。**谷歌的
  Jules** 编程智能体推出了支持 CI/CD 集成的可编程 API。**Qwen（通义千问）** 明确了其模型分类体系和 API 层级。


  Vision/LM Arena 排名显示，**Claude Sonnet 4.5**、**Claude Opus 4.1**、**Gemini 2.5 Pro**
  以及 OpenAI 的最新模型之间竞争异常激烈。在视频生成领域，**Sora 2 Pro** 凭借快速迭代和全新的创作者生态系统领跑 App Store 排行榜；早期测试显示，其回答
  GPQA 风格问题的准确率为 55%，而 GPT-5 为 72%。Video Arena 引入了 **Luma 的 Ray 3** 和 **可灵 (Kling)
  2.5** 等新模型进行基准测试。


  此外，多模态视频+音频生成模型 **Ovi**（类 Veo-3）正式发布。检索模型方面，麻省理工学院（MIT）推出了具有高效图文检索能力的 **ModernVBERT**。“Claude
  Sonnet 4.5 在编程方面与 Opus 4.1 基本一致”以及“Jules 是一个可编程的团队成员”是本次更新的核心洞察。'
id: MjAyNS0x
models:
- claude-3-sonnet
- claude-3-opus
- gpt-5-codex
- grok-4-fast
- qwen-3-next
- gemini-2.5-pro
- sora-2-pro
- ray-3
- kling-2.5
- veo-3
- modernvbert
people:
- finbarrtimbers
- gauravisnotme
- justinlin610
- billpeeb
- apples_jimmy
- akhaliq
title: 今天没什么事。
topics:
- coding-agents
- cybersecurity
- api
- model-taxonomy
- model-ranking
- video-generation
- benchmarking
- multi-modal-generation
- retrieval
- image-text-retrieval
---

**DevDay 前的宁静。**

> 2025年10月2日至10月3日的 AI 新闻。我们为你检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（196 个频道，10895 条消息）。预计节省阅读时间（以 200wpm 计算）：758 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式呈现所有往期内容。详见 https://news.smol.ai/ 查看完整的新闻分类，并通过 @smol_ai 向我们提供反馈！

**gm，Anthropic** 迎来了[新任 CTO](https://x.com/zeffmax/status/1973833211835974046?s=46)。

---

# AI Twitter 回顾

**前沿编程 Agent 与模型排名（Claude 4.5, Grok Code Fast, Google’s Jules, Qwen 命名规则, Arena 排行榜）**

- **Claude Sonnet 4.5 (上手体验)**：在使用了约 30 小时 Claude Code 后，[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1973922679418974298) 发现 Sonnet 4.5 在编程方面“基本上与 Opus 4.1 相同”——UX 精美，实力强劲，但不如 GPT-5 Codex；同时指出 [ChatGPT Team 的性价比高于 Claude Max](https://twitter.com/finbarrtimbers/status/1973923264524398687)。Anthropic 强调了 Sonnet 4.5 的网络安全实力（在某些任务上与 Opus 4.1 相当或更优），并专注于防御能力 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1974199155657748868)，[后续](https://twitter.com/AnthropicAI/status/1974199158929305738)。
- **xAI Grok Code Fast**：[@gauravisnotme](https://twitter.com/gauravisnotme/status/1974001009778115066) 声称其“diff 编辑成功率”高于 Claude 4.5 和 GPT-5 Codex，且成本更低——这值得独立验证，但多个用户正在更多地根据编辑可靠性而非原始的 next-token 指标来对编程 Agent 进行基准测试。
- **Google 的 Jules 编程 Agent 现已支持编程化**：为期一周的发布活动以公开 API 告终，使 Jules 成为一个拥有工具和 CI/CD 集成能力的“可编程团队成员” [回顾 @julesagent](https://twitter.com/julesagent/status/1973898898067632212)，[API 发布](https://twitter.com/julesagent/status/1974178592683954252)，[文档](https://twitter.com/julesagent/status/1974179029726159274)。来自 [@GoogleLabs](https://twitter.com/GoogleLabs/status/1974206675859984531) 的额外推文和来自 [@googledevs](https://twitter.com/googledevs/status/1974217899536474565) 的产品文章。
- **Qwen 的命名清晰度**：Qwen 模型家族（LLM, Coder, VL, Omni, Image）、Instruct 与 Thinking 变体、API 层级（Max/Plus/Flash）、带日期后缀的小版本更新，以及为什么存在 “Qwen3-Next” 的有用分类法 [@JustinLin610](https://twitter.com/JustinLin610/status/1973974975976808808)。
- **实时排名**：Vision/LM Arena 显示顶级梯队竞争异常激烈：Sonnet 4.5（标准版和 32k Thinking）、Claude Opus 4.1 和 Gemini 2.5 Pro 四方并列第一；OpenAI 模型（4o-latest, 4.5 preview, 5 high, o3）之间的评分差距仅在 1 分以内 [@arena](https://twitter.com/arena/status/1974215622474293262)，[后续](https://twitter.com/arena/status/1974215628757066077)。OpenRouter 指出 Grok 4 Fast 在德语提示词/补全方面占据主导地位 [@OpenRouterAI](https://twitter.com/OpenRouterAI/status/1974122770645864767)。

**视频生成浪潮：Sora 2 Pro 的势头、评估及更广泛的模型栈**

- **Sora 2 Pro 的采用与能力信号**：Sora 2 现位居 App Store 第一；团队正在快速迭代并发送邀请 [@billpeeb](https://twitter.com/billpeeb/status/1974035563482116571)。高质量的 15 秒剪辑正在陆续发布 [@apples_jimmy](https://twitter.com/apples_jimmy/status/1973979773354586379)。早期测试表明，Sora 2 在一小部分 GPQA 风格问题上的回答准确率约为 55%，而 GPT-5 为 72%；一个合理的解释是视频生成前存在一个 LLM “提示词重写（prompt rewrite）”层 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1974172794012459296)，[提示词重写假设](https://twitter.com/EpochAIResearch/status/1974172889676177682)，[背景](https://twitter.com/EpochAIResearch/status/1974172901567004762)。该应用还推动了新的创作者生态系统（例如：去水印工作流）[@angrypenguinPNG](https://twitter.com/angrypenguinPNG/status/1974144279955325191)。
- **生态系统与基准测试**：Video Arena 增加了 Luma 的 Ray 3 和 Ray HDR 3，用于面对面的社区投票评估 [@arena](https://twitter.com/arena/status/1974161623935037658)。Kling 2.5 在拼接编辑中展示了出色的帧匹配能力 [@heyglif](https://twitter.com/heyglif/status/1974195300240957445)。多模态视频+音频生成模型 “Ovi”（类似于 Veo-3）发布：支持 24 FPS、最高 720×720 分辨率的 5 秒视频，支持文本或文本+图像条件输入 [@_akhaliq](https://twitter.com/_akhaliq/status/1974181920092418128)。

**检索、VLM 与感知模型（ModernVBERT, Jina v3, RF-DETR, π0.5 机器人）**

- **ModernVBERT / ColModernVBERT (MIT)**: 一个用于图像文本和文档检索的小型双向 ModernBERT 编码器，在 ViDoRe 上的表现与 ColPali 相当，但参数量减少了约 10 倍（约 250M）。ColModernVBERT 的延迟交互（late-interaction）双编码器变体报告了 +10.6 nDCG@5 的提升，并被定位为亚线性检索器（不仅是重排序器），支持十亿级文档的 kNN 检索 [@pteiletche](https://twitter.com/pteiletche/status/1974023966936203646), [@mervenoyann](https://twitter.com/mervenoyann/status/1974027641033261106), [HF models](https://twitter.com/mervenoyann/status/1974027983342989583), [authors’ thread](https://twitter.com/ManuelFaysse/status/1974036787187028079), [framework note](https://twitter.com/lateinteraction/status/1974105498044743704)。
- **Listwise reranking (Jina v3, 0.6B)**: 一个“最后但非延迟”（last but not late）的列表式重排序器（listwise reranker），它通过一次处理将查询与所有候选文档拼接，提取文档和查询的特殊标记嵌入（special-token embeddings），并在 BEIR 上达到 SOTA [@JinaAI_](https://twitter.com/JinaAI_/status/1974148565770338705), [input format](https://twitter.com/JinaAI_/status/1974148568563745012), [links](https://twitter.com/JinaAI_/status/1974148570711548213)。评论：虽然品牌名为“最后交互”（last interaction），但它实际上是具有强大实证结果的早期、全上下文列表式交互 [@lateinteraction](https://twitter.com/lateinteraction/status/1974153399164862927)。
- **Detection and segmentation**: Roboflow 的 RF-DETR 分割预览版声称在 COCO 分割任务上比 YOLO11-L 快 3 倍且更准确，在 T4 上具有 TensorRT 10.4 级的延迟，并拥有强大的 DINOv3 骨干网络结果（例如，在 1 个 epoch 内完成细微裂纹分割） [@skalskip92](https://twitter.com/skalskip92/status/1974160476444766324), [latency](https://twitter.com/skalskip92/status/1974160481192747039), [notebook](https://twitter.com/skalskip92/status/1974160484799590789)。
- **Open-source robotics baseline**: Physical Intelligence 的 π0 和 π0.5 现已上线 Hugging Face 并完全移植到 PyTorch/LeRobot，重点关注跨具身（cross-embodiment）、多环境的 Vision‑Language‑Action 训练，以实现开放世界泛化 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1974094115743711702)。

**Reasoning, RL, and verifiers (PPO/GRPO, RESTRAIN, ExGRPO, RLAD, TUMIX, CLUE, RoT)**

- **RL recipes and corrections**: 探讨 PPO/GRPO 为何有效及其与人类感知的潜在联系 [@ethayarajh](https://twitter.com/ethayarajh/status/1973901333557346803)。DrGRPO 作者重申应移除回复长度归一化（均值 vs 总和）以避免微妙的偏差；Tinker 的实现被作为无偏损失（unbiased losses）的参考 [@zzlccc](https://twitter.com/zzlccc/status/1973960971296387278)。
- **Label-free/self-driven RL**: RESTRAIN 将伪多数（spurious majorities）转化为自惩罚信号——使用所有 rollouts，抵消低一致性优势，并展示了训练和测试时的扩展收益（例如，在 AIME/AMC/MATH500 上使用 Llama3.1‑8B 相比 TTRL/ETMR 平均提升 11%） [@jaseweston](https://twitter.com/jaseweston/status/1974000962219225271), [results](https://twitter.com/jaseweston/status/1974000970192544248), [ablations](https://twitter.com/jaseweston/status/1974000971757101219)。ExGRPO 提出了一种具有混合策略目标的经验优先级排序，以在 on-policy 失败的情况下稳定训练 [@papers_anon](https://twitter.com/papers_anon/status/1973945230526459951)。
- **Abstractions and pretraining**: RLAD 训练 LLM 发现可重用的“推理抽象”以引导探索 [@QuYuxiao](https://twitter.com/QuYuxiao/status/1974187714343034932), [alt](https://twitter.com/Anikait_Singh_/status/1974195667250864561)。NVIDIA 提出了“强化学习作为预训练目标”（RLP），以弥合监督预训练与 RL 之间的差距 [@_akhaliq](https://twitter.com/_akhaliq/status/1974190336256962812)。Google 的 TUMIX 混合了 12–15 个多样化的工具使用 Agent（文本/代码/搜索），在轮次间共享笔记，并使用 LLM-judge 提前停止——提高了基准测试准确率并降低了成本（例如，Gemini 2.5 Pro HLE 达到 34.1%） [@omarsar0](https://twitter.com/omarsar0/status/1974106927287447725)。
- **Verification and retrieval of thoughts**: 腾讯的 CLUE 验证器使用聚类——无需训练参数——并报告了比 GPT-4o 更高的验证准确率 [@LiangZhenwen](https://twitter.com/LiangZhenwen/status/1973928150104223868)。Retrieval-of-Thought 通过“思维图谱”重用先前的推理轨迹，在不损失准确性的情况下减少高达 40% 的 token，将推理速度提高 82%，并降低 59% 的成本 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1974228574598205736)。

**Efficiency, quantization, and infra (FP8, SINQ, MLX, CPU MoE, QAT, sampling, training control)**

- **FP8 训练与量化**：蚂蚁集团的 Ling 2.0 开源了一个 FP8 原生混合精度 MoE 训练栈（包含细粒度缩放、FP8 Adam 状态、路由图），据报告在配合 MTP 时可达到 BF16 级别的精度，且吞吐量提升 30–60%，即使不使用 MTP 也有显著优势 [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1974182694239285260)。红帽（Red Hat）发布了 FP8 量化的 Qwen3‑VL‑235B‑A22B‑Instruct，磁盘/GPU 显存占用减少约 50%，且精度保留率超过 99.6% [@RedHat_AI](https://twitter.com/RedHat_AI/status/1973932224400798163)。华为的 SINQ 提出了一种无需校准的量化方法，在大幅削减内存占用的同时保持了 SOTA 水平 [@HuggingPapers](https://twitter.com/HuggingPapers/status/1973906002001936577)。
- **计算与平台说明**：在 Apple Silicon 上，MLX 构建版本的性能大幅领先于通用的 GGUF；一位用户报告在 4-bit 下运行 Granite 4 H Tiny，速度为 115 tok/s 对比 47 tok/s [@JorgeConsulting](https://twitter.com/JorgeConsulting/status/1974168414391619672)。MoE 模型在 CPU 上的吞吐量出奇地高：Qwen 30B/A3B 在 CPU 上约为 21 tok/s，Qwen 232B MoE 约为 4 tok/s [@Teknium1](https://twitter.com/Teknium1/status/1974039942751006816)。Together 的 Instant Clusters 公布了清晰的按需/预留 GPU 定价 [@togethercompute](https://twitter.com/togethercompute/status/1974167802337730854)。
- **训练机制与库**：来自 Apple 的 Awni Hannun 分享了关于 QAT 缩放定律的见解，探讨在固定的 RAM/延迟预算下如何选择 8-bit、4-bit（或 2-bit）[@awnihannun](https://twitter.com/awnihannun/status/1974245339512385784)。Batch sampler 分片技术将复杂的采样（加权/温度/平衡）集中化，以确保跨 worker 的一致性和效率 [@TheZachMueller](https://twitter.com/TheZachMueller/status/1974072997670736098)。Hugging Face TRL 复现了 “LoRA without regrets”，在熟悉的 API 下提供了更高性能的 LoRA 实现 [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1974191312229577085)。“交互式训练（Interactive Training）”提议在运行期间进行人在回路（human-in-the-loop）的学习率（LR）调优——将 Loss 监控转化为可控的反馈过程 [@yuntiandeng](https://twitter.com/yuntiandeng/status/1974127176778662339)。

**行业与研究信号 (Sakana x Daiwa, Terence Tao + GPT-5, xLSTM 缩放定律, Comet)**

- **金融科技部署**：Sakana AI 与大和证券（Daiwa Securities）签署了一份约 50 亿日元（约 3400 万美元）的多年度协议，共同构建“全资产咨询平台”，利用 Sakana 的模型进行研究报告生成、市场分析和投资组合构建 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1973935631354245286), [Bloomberg 摘要](https://twitter.com/SakanaAILabs/status/1974109165623853365)。
- **人类+AI 探索**：陶哲轩（Terence Tao）公开记录了使用 GPT-5 结合工具调用（tool-use）在数学中搜索反例和启发式方法的过程——S. Bubeck 将其标记为 HAI（人机交互）研究工作流的一个显著时刻 [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1973977315572154383), [示例推文串](https://twitter.com/kevinweil/status/1974161952260624459)。
- **架构**：xLSTM 报告称，在固定 FLOPs 和固定 Loss 机制下，其交叉熵表现均帕累托占优（Pareto-dominating）于 Transformer，并带来了下游推理效率的提升 [@maxmbeck](https://twitter.com/maxmbeck/status/1974018534385598895), [@HochreiterSepp](https://twitter.com/HochreiterSepp/status/1974027057215472107)。
- **浏览器作为 AI 界面**：Comet 的发布引发了用户的极大热情和采用，尤其是在 macOS 和 Windows 上；其设计因既有熟悉感又通过非侵入式 AI 集成进行增强而受到赞誉 [@felixleezd](https://twitter.com/felixleezd/status/1973942012278935631), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1974093953499877510), [后续](https://twitter.com/AravSrinivas/status/1974094750308553191)。

**热门推文（按互动量排序）**

- Sora 2 去水印工作流走红，展示了围绕该应用增长的创作者工具生态 [@angrypenguinPNG](https://twitter.com/angrypenguinPNG/status/1974144279955325191) (6.9k)。
- OpenAI 将敏感对话路由至 GPT-5 Instant，以提供更快、更有帮助的支持；可见的模型指示器仍然保留 [@OpenAI](https://twitter.com/OpenAI/status/1974234951928459450) (2.3k)。
- 陶哲轩（Terence Tao）公开的 GPT-5 辅助数学探索示例 [@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1973977315572154383) (4.3k)。
- Sora 2 Pro 高质量 15 秒剪辑；随着邀请制的持续推进，该应用登上榜单第 1 名 [@apples_jimmy](https://twitter.com/apples_jimmy/status/1973979773354586379), [@billpeeb](https://twitter.com/billpeeb/status/1974035563482116571) (0.8k, 1.6k)。
- Claude Sonnet 4.5 与 GPT-5 Codex 及 Opus 4.1 的代码评审对比 [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1973922679418974298) (0.6k)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 综述

### 1. LLM 效率与基准测试：华为 SINQ 量化 + GLM 4.6 工具调用性能

- [**华为开发出新型 LLM 量化方法 (SINQ)，比 AWQ 快 30 倍，且在无需任何校准数据的情况下超越校准方法**](https://www.reddit.com/r/LocalLLaMA/comments/1nwkzq7/huawei_develop_new_llm_quantization_method_sinq/) (热度: 335): **华为提出了 SINQ，这是一种训练后 LLM 量化方案，它通过添加每个矩阵的第二轴缩放和受 Sinkhorn–Knopp 启发的快速归一化来最小化行/列方差不平衡代理，从而实现了无需校准的量化 (SINQ) 以及校准变体 (A-SINQ)。报告结果显示，与 AWQ 相比，量化时间缩短了** `~30×`**，并在 Qwen3 和 DeepSeek-V2.5 等模型上提高了 4-bit 及以下的困惑度 (Perplexity)；参见论文 [PDF](https://arxiv.org/pdf/2509.22944)，并指出该方法是“无需校准、层独立”的，代码已在 GitHub 上发布。至关重要的是，** `30×` **的提升是指量化速度，而非推理/反量化吞吐量，且未提供有关运行时格式或与常用技术栈兼容性的实现细节。** 评论者指出缺少推理时间/反量化基准测试（这对于大批次吞吐量至关重要），并且缺乏关于如何在 Transformers/llama.cpp 中运行量化模型的指导，推测输出可能是 .safetensors。其他人注意到文中介绍了两种方法（A-SINQ 需要校准，而 SINQ 不需要），并批评其对比对象（SINQ vs. HQQ [博客](https://mobiusml.github.io/hqq_blog/)）与常用基准（AWQ, EXL2/3, MLX, GGUF）的相关性较低，敦促提出更清晰的主张和更广泛的质量基准测试。
    - 一位评论者剖析了论文的观点：华为提出了两种方法——**A-SINQ**（需要校准，与 **AWQ** 对比）和 **SINQ**（无需校准，与 **HQQ** 对比）。他们强调报告的 `30x` 加速是针对量化过程本身，而非推理，并指出缺少与 **AWQ**、**EXL2/EXL3**、**MLX** 或 **GGUF** 等广泛使用的方法在质量/运行时方面的直接对比基准。他们还指出，尽管 **HQQ** 显示出与 AWQ 相当的困惑度且具有轻微的内存优势（[博客](https://mobiusml.github.io/hqq_blog/)），但并未被广泛采用。
    - 另一个讨论帖强调，对于大批次推理，瓶颈往往从内存带宽转移到反量化计算；因此，反量化速度/开销对吞吐量至关重要。他们警告说，量化步骤快 `30x` 并不意味着解码或批处理效率更高，除非反量化数学运算和算子（Kernels）更廉价，并要求提供关于反量化 FLOPs/延迟以及批处理下有效 tokens/s 的基准测试。
    - 一份来自图像源的解读建议，核心技术是一个简单的预处理步骤，几乎可以应用于任何量化算法之前，这意味着它很容易与现有流水线组合（[图表](https://preview.redd.it/7uof90n49wsf1.png?width=1640&format=png&auto=webp&s=5585b671237adc2e5cfefe05c9fd844480a5dfdd)）。如果属实，推理算子可能保持不变，因此运行时的收益将取决于预处理是否降低了反量化复杂度或改善了权重统计特性，而不是需要新的后端。
- [**GLM 4.6 是一个极其出色的模型，没人能说服我改变看法**](https://www.reddit.com/r/LocalLLaMA/comments/1nx18ax/glm_46_is_a_fuking_amazing_model_and_nobody_can/) (热度: 417): **楼主报告了在生产环境中使用 GLM-4.5/4.6 (智谱 AI) 一个月的经历，用户对 Agent 自主性的反馈始终很强，且工具/函数调用准确率显著提高，优于他们尝试过的其他替代方案（如 Claude Sonnet、GPT 变体、Grok Code）。他们建议通过伯克利函数调用排行榜 [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) 评估工具使用能力，并批评 "Artificial Analysis" 基准测试无法代表真实世界的性能。** 热门评论一致认为 Artificial Analysis 往往与实际可用性呈负相关，并且可能偏向于针对基准测试过度优化的 Phi 风格模型，而 GLM 在 Agent 工作负载方面表现出色；一位用户询问楼主是本地运行还是通过云端 API 运行。另一位评论者声称在他们的测试中 GLM-4.6 优于 Sonnet 4/4.5，称这是智谱 AI 的胜利。
    - 几位评论者认为 **Artificial Analysis** 排行榜 ([https://artificialanalysis.ai](https://artificialanalysis.ai/)) 与现实世界的实用性背道而驰，声称它放大了那些在合成测试中过拟合的“针对基准测试优化”的 Phi 风格模型。他们指出 GLM 4.6 在 Agent 场景（工具使用、多步规划）中表现出色，强调了合成基准测试与实际 Agent 性能之间的差距。
    - 一份用户报告的直接对比显示，**GLM 4.6** 在其任务中的表现明显优于 “Sonnet 4/4.5”，表明在其评估中任务执行能力更强，尽管没有分享具体的量化指标。这表明尽管基准测试评价褒贬不一，但 GLM 4.6 在某些现实世界的工作负载中可能具有优势。

- 早期测试者报告称，GLM 4.6 在处理简单任务时存在**较长的推理/思考阶段**，引发了对延迟的担忧。一位测试者正在寻求减少模型“思考长度”的方法，这暗示了如果供应商提供相关功能，则需要 API/运行时控制（例如更严格的 max tokens 或推理预算限制）；部署模式（本地 vs 云端 API）也被提及，但未详细说明。
- [**十年内最重要的 AI 论文。不接受反驳**](https://www.reddit.com/r/LocalLLaMA/comments/1nwx1rx/the_most_important_ai_paper_of_the_decade_no/) (活跃度: 1921): **该帖子断言图中显示的是 Vaswani 等人 (2017) 的《Attention Is All You Need》([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))，即 Transformer 论文。从技术上讲，它用 self-attention 取代了循环/卷积，引入了 multi-head attention 和 positional encodings，实现了完全并行的序列训练，在机器翻译中达到了 SOTA，并成为了 BERT/GPT 规模 LLMs 的基础。** 评论者通过引用之前的关键工作来背景化其影响：Mikolov 等人 (2013) 的 Word2Vec ([arXiv:1301.3781](https://arxiv.org/abs/1301.3781)) 和 Bahdanau 等人 (2014) 的 attention/NMT ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473))，指出存在幸存者偏差，且重大突破建立在早期创新之上；“最具影响力”与对先前工作的依赖性之间存在争议。
    - Attention 早于 Transformers：**Bahdanau, Cho, Bengio (2014)** 为 NMT 引入了 additive attention，通过学习源 token 和目标 token 之间的 soft alignments 来消除固定长度编码器的瓶颈，这是 Transformer 的 scaled dot‑product attention 的直接先驱 ([论文](https://arxiv.org/abs/1409.0473))。这使序列建模从“压缩后解码”转向“动态上下文检索”，实质性地提高了相比原生 encoder‑decoder RNNs 的翻译质量，并实现了更长距离的依赖。
    - 基础表示学习源于 **Mikolov 等人 (2013) 的 Word2Vec** ([论文](https://arxiv.org/abs/1301.3781))，该论文提出了带有 negative sampling 和 hierarchical softmax 的 CBOW/Skip‑Gram，以高效地从大规模语料库中学习稠密词嵌入（word embeddings）。通过用采样目标取代全词表 softmax，它将每次更新的训练成本从 `O(|V|)` 降低到 `O(k)`，在向量空间中产生了线性语义结构，后来的架构（包括 Transformers）利用这些结构进行预训练和迁移。
    - 对于 2010 年代，许多人认为 **AlexNet (2012)** 是关键的催化剂：它在 `2× GTX 580` GPU 上使用 ReLUs、dropout 和 local response normalization 进行训练，将 ILSVRC‑2012 的 top‑5 错误率大幅降低至 `15.3%`（此前 SOTA 约为 `26.2%`），开启了大规模 GPU 深度学习的时代 ([论文](https://dl.acm.org/doi/10.1145/3065386))。这种硬件‑软件协同设计的时刻使神经网络的 GPU 加速规范化，并解锁了后来被 Transformers 利用的 scaling regimes。

## 较低技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Sora 2 与最新的文本转视频（Text-to-Video）演示片段

- [**我让 SORA 2 制作了一个 90 年代风格的 Epstein 岛玩具广告。**](https://www.reddit.com/r/singularity/comments/1nwz4cx/i_asked_sora_2_to_create_a_90sstyle_toy_ad_of/) (热度: 701): **发帖者声称使用了 OpenAI 的视频模型（称为 “Sora 2”）生成了一个以争议性现实地点为主题的 1990 年代风格玩具广告，突显了 Sora 对特定时代广告美学的还原能力和讽刺性构图能力。由于 [v.redd.it](http://v.redd.it/) 上的链接目前无法访问（**`403 Forbidden`**），内容无法独立验证；参考模型信息：[OpenAI Sora](https://openai.com/sora)。此帖子通过将怀旧广告手法与敏感话题结合，隐晦地探测了 Sora 的安全/审核边界。** 热门评论多为非技术性反应：调侃该模型被列入了“禁飞名单”，并称赞其“令人不安”但又“令人惊叹”，暗示了感知到的高保真度和喜剧冲击力，同时也流露出伦理上的不适。
- [**最新文本转视频模型的必测项目：吃意大利面**](https://www.reddit.com/r/singularity/comments/1nx6w1f/obligatory_test_of_latest_texttovideo_model/) (热度: 491): **帖子展示了在“最新”模型上进行的标准“吃意大利面”文本转视频压力测试；链接资源目前无法访问（[v.redd.it/znaochtxuxsf1](https://v.redd.it/znaochtxuxsf1) 返回** `403`**），因此仅能通过评论获取信息。从评论来看，对于类似 Will Smith 的主体，身份保真度（Identity fidelity）依然较弱（面部特征未得到保留），但与 2023 年产出的结果相比，感知的时序/动作连贯性似乎有所提高（即持续的多步动作）。** 评论者指出“Will Smith 吃面”测试仍是一个事实上的基准；争论集中在尽管序列和连贯性有明显提升，但身份相似度依然较差。
    - 身份保真度（Identity fidelity）仍是一个弱点：多位评论者指出与 Will Smith 的相似度很低，这表明目前的文本转视频流水线在稳健的面部 ID 条件化（face-ID conditioning）和时序面部一致性方面仍面临挑战。除了数据/架构限制外，规避名人肖像的安全过滤器也可能降低身份准确性，导致帧间漂移和“模型外”面部。实际工作流通常需要插件，如 ID-guidance（例如 [ID-Adapter](https://arxiv.org/abs/2308.06767)）、ControlNet 风格的条件化（[ControlNet](https://arxiv.org/abs/2302.05543)）或后处理的面部跟踪/抠像（roto）以保持稳定性。
    - 音频集成被强调为超越 2023 年代无声片段的一大进步：最新的演示似乎包含了同步的语音/音效（SFX），这意味着要么使用了联合的 AV diffusion/transformer 堆栈，要么使用了 TTS + 对齐阶段。这增加了关于唇形同步（lip-sync）、音色一致性以及跨 `N` 帧的音视频对齐的复杂性；典型的失败模式包括韵律的恐怖谷效应、视素（viseme）漂移以及在快速运动或遮挡期间的失步。当端到端音视频生成效果不佳时，诸如音素到视素映射和唇形同步修正（例如 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)）等技术仍然具有相关性。
- [**罗杰斯先生在阿金库尔战役**](https://www.reddit.com/r/aivideo/comments/1nwzxfq/mr_rogers_at_the_battle_of_agincourt/) (热度: 853): **该帖子展示了一个由提示词驱动的生成式 AI 作品，将 [Fred Rogers](https://en.wikipedia.org/wiki/Fred_Rogers) 置于 [阿金库尔战役 (](https://en.wikipedia.org/wiki/Battle_of_Agincourt)**`1415`**[)](https://en.wikipedia.org/wiki/Battle_of_Agincourt) 场景中。链接的 Reddit URL 在未经身份验证的情况下返回** `HTTP 403 (Forbidden)`**，因此无法获取底层媒体和技术元数据；该线程未提供明确的模型/流水线、参数或提示词，尽管一条询问“提示词是什么？”的热门评论暗示使用了文本转图像/视频合成（可能还有语音克隆）。** 评论大部分是非技术性的（表达惊讶和兴趣，例如“哇，这太疯狂了”），唯一可操作的请求是索要准确的提示词；未讨论基准测试、模型选择或实现细节。
    - 一位评论者强调，通过 **80 年代电影摄像机** 的美学来观察中世纪战斗会让它感觉更真实，这暗示了拍摄时代的伪影（胶片颗粒、较低的动态范围、偏色和帧率节奏）会实质性地影响生成的或重建的素材的感知真实性。对于再创作而言，模拟模拟特性（如 `24 fps` 节奏、颗粒、轻微的片门抖动和磁带噪声）比单纯提高分辨率或清晰度更能减少恐怖谷效应。

### 2. GPT-5 思维维基百科审计与研究辅助

- [**OpenAI 的 Noam Brown 一直在使用 GPT-5 Thinking 来查找每个 Wikipedia 页面中的错误。其中一些错误可能相当严重。甚至关于 Wikipedia 的 Wikipedia 页面也有一个错误。**](https://www.reddit.com/r/singularity/comments/1nwl1kz/noam_brown_of_openai_has_been_using_gpt5_thinking/) (热度: 714): **帖子声称 OpenAI 的 Noam Brown 正在使用即将推出的 GPT-5 “Thinking” 模式系统地扫描 Wikipedia 的事实错误，并展示了一些例子（包括在 [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia) 条目本身）。评论者指出，至少有一个展示的问题已经被标记为 [citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)，表明 Wikipedia 现有的 QA/维护工作流已经标记了它，并警告说像“查找至少 1 个错误”这样的提示词可能会诱发 LLM 幻觉和误报；这需要严格的验证和溯源。参见原始讨论帖：[Reddit 画廊链接](https://www.reddit.com/gallery/1nwl1kz)。** 辩论集中在 Brown 方法的审慎性和可信度，以及对 Brown 此前关于 Wikipedia 看法的讨论。此外，还有一种更广泛的批评认为，提倡用 LLM 替代 Wikipedia 的声音更倾向于中心化、封闭且透明度较低的系统，而 Wikipedia 则是开放的编辑流程。
    - 多位评论者指出，示例章节已经带有 [Citation needed] 标签，表明 Wikipedia 内置的 QA 正在发挥作用；GPT 标记这些并不代表发现了新错误。在没有外部根据的情况下，提示 LLM “查找至少 1 个错误”会使其偏向于产生误报（幻觉）而非寻找真相。相关政策：[Verifiability](https://en.wikipedia.org/wiki/Wikipedia:Verifiability) 和 [Citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)。
    - 一个引用的案例涉及 GPT-5 据称将一个千卡数值与其参考文献混淆，从而产生了一个错误声明，而实际上数值和引用可能都是正确的，只是匹配错了。这是一种 **source-attribution mismatch**（来源归属不匹配/引用对齐失败），这是在不强制执行证据链接时 LLM 的常见弱点。该模型还调用 CDC 作为事实依据，但未验证所引用的 CDC 页面是否支持该确切陈述，突显了证据链的薄弱。
    - 从技术上讲，用**中心化、闭源的 LLM** 取代 Wikipedia 透明、有版本记录、社区审计的工作流，会降低出处可靠性和可复现性。强大的 LLM 驱动的 QA 需要报告针对人工裁决的精确率/召回率，公开来源，并为编辑输出可验证的 diff；如果没有这些，模型的判断就是不可审计且非确定性的。简而言之，可靠性取决于有根据的检索和可衡量的评估，而非模型的断言。
- [**Terence Tao 表示 ChatGPT 帮助他解决了一个 MathOverflow 问题，并节省了数小时的手动编码时间**](https://www.reddit.com/r/singularity/comments/1nwqqrj/terence_tao_says_chatgpt_helped_him_solve_a/) (热度: 1376): **菲尔兹奖得主 Terence Tao 报告称，[ChatGPT](https://chat.openai.com/) 在一个 [MathOverflow](https://mathoverflow.net/) 问题上提供了帮助，生成了原本需要“数小时手动编码”的代码。根据 [Reddit 帖子](https://www.reddit.com/gallery/1nwqqrj)（目前未经身份验证返回** `HTTP 403` **），这被引用为 LLM 加速探索性数学/编程工作流（例如，快速生成用于计算检查的辅助脚本/样板代码）的实际案例，而非正式证明的替代品。** 评论强调有效性是一个“能力问题”（提示词/工具使用熟练度），并预测随着现实世界效用的累积，怀疑论者将会转变；讨论大多是非技术性的认可，而非关于模型能力极限的辩论。
    - 一位评论者根据共享聊天记录的“思考时长”和交互风格推断，Tao 和 Aaronson 可能使用的是“中/低”档位的 **GPT-5 Thinking**，而非“高”档位的 **GPT-5 Pro**。该说法基于观察到的延迟/计算预算，这些迹象表明使用了较低的思考设置，暗示即使是次旗舰或低预算层级的模型也能在高级数学工作流中提供有意义的帮助。
    - 他们引用评估结果称，“高”版本的 **GPT-5 Thinking** 在 2025 IMO 中，在纯粹的 “best-of-32” 采样方案下（而非 **Gemini** 式的 Agent 架构）仅获得 `38%` 的分数，而大约 3 个月前的一个内部实验模型据称实现了“一次尝试即获金牌”。技术上的结论是，采样策略（best-of-N 对比 Agent 化）和模型变体/层级都会显著影响奥数风格的基准测试，使跨模型比较变得复杂。

- 另一个观察结果：模型反复拒绝根据聊天上下文猜测用户的身份，仅在压力下提供了一个名字，并标注为“(低置信度)”。这表明在防止泄露个人隐私（doxxing）/身份识别方面存在保守的安全/策略层，并且在回答中进行了一定程度的显式不确定性校准（uncertainty calibration），这可能会影响研究人员探测模型的元推理（meta‑inference）能力。

### 3. AI 在教育领域：教师的采纳与学生法律案例

- [**老师不隐瞒他使用 AI 的行为。**](https://www.reddit.com/r/ChatGPT/comments/1nwpcbu/teacher_doesnt_hide_his_use_of_ai/) (热度: 609)：**照片显示老师提供的考试/工作表上明确标注为 AI 生成（例如通过 ChatGPT），标志着透明地使用生成式 AI 来起草课堂材料。该帖子将 AI 背景化为教育者的生产力工具（生成测试/教案），而非学生绕过学习的手段，这与 AI 出现之前的做法（如使用共享题库或购买的材料）相一致。** 评论者普遍赞成教师将 AI 作为工具使用（与学生的滥用形成对比），并指出这类似于从市场购买/借用课程。一些人暗示，在利用 AI 输出进行评估时，公开透明和质量控制是关键。
- [**佛罗里达州一名 13 岁学生在向 ChatGPT 询问犯罪问题后被捕**](https://www.reddit.com/r/ChatGPT/comments/1nwpv3v/a_13yearold_student_in_florida_got_arrested_after/) (热度: 864)：**一名 13 岁的佛罗里达学生在学校管理的上下文环境中向 ChatGPT 输入犯罪查询后被捕；检测是通过学校自身的监控系统（而非 ChatGPT/OpenAI）实现的，该系统标记并上报了内容。评论摘要指出“未发现犯罪意图”，且该学生“正在等待法律程序”，这意味着行动是由监控日志驱动的，而非证明其具有犯罪意图（mens rea）。** 评论者强调触发因素是学校运行的监控而非模型供应商，辩论中心在于比例原则——即在 *未发现犯罪意图* 的情况下逮捕是否合适——以及 K–12 设备/账户监控管道的广泛性。
    - 检测源于学校的监控技术栈 (**Gaggle**)，而非 ChatGPT/OpenAI。Gaggle 通常部署在学校管理的账户/设备上，实时扫描学生内容，并将高风险短语自动上报给管理人员/执法部门 ([gaggle.net](http://gaggle.net/))，这与描述的流程相符（查询 -> 警报 -> 警察）。从技术上讲，这是客户端/网络侧的遥测（telemetry），而非提供商侧的报告。
    - 尽管官员表示“未发现犯罪意图”，但自动警报仍导致了法律程序，这说明了基于关键词的威胁检测如何不顾意图而升级。这反映了一种低门槛、高严重性的政策，其中像“杀（kill）”这样的匹配会触发立即行动以缩短响应时间，代价是牺牲了上下文敏感性并增加了误报风险。
    - 为了让 Gaggle 标记输入到 ChatGPT 中的提示词（例如，“如何在课间杀掉我的朋友？”），系统必须通过托管的 Chromebook/终端代理（endpoint agent）、Chrome 扩展程序或检查学校账户内容的网络代理（network proxy）获得可见性。实际上，在学校基础设施上对第三方 AI 服务的查询并不是私密的；其流程是：终端/代理捕获 -> AI 分拣 -> 人工审核 -> 警报，而不是由 ChatGPT 本身进行任何“报告”。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要的摘要的摘要
> 

**1. Agentic 开发工具：Comet, Solveit, Chrome DevTools MCP**

- **Comet 正式发布，Agent 实现并行化**：Perplexity 向所有人推出了 AI 原生的 **Comet Browser**，可在 [perplexity.ai/comet](https://www.perplexity.ai/comet) 免费使用。该浏览器支持并行的 **agentic tasks**，并已在全球范围内结束候补名单。
    - 早期采用者称赞其速度和更“聪明”的搜索，同时也指出了在全球推广过程中存在的提示词注入（prompt-injection）风险和平台功能差距（[Comet 发布公告](https://xcancel.com/perplexity_ai/status/1973795224960032857)）。
- **Solveit 发布，解决“AI 疲劳”**：**Jeremy Howard** 宣布正式发布 **Solveit**，这是一个在 [**Answer.AI**](http://answer.ai/) 内部使用的 AI 增强型开发平台，并于 **10 月 20 日**开始为期 5 周的直播课程（[Solveit 发布公告](https://xcancel.com/jeremyphoward/status/1973857739341508884)）。
    - 该计划提供平台访问权限和培训，展示真实工作流（系统管理、应用部署、GUI 开发、合同起草），旨在缩短反馈循环并应对“AI 疲劳”。
- **Chrome MCP 落地 DevTools**：权威的 **Chrome DevTools MCP** 在 [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) 发布，为 Agent 提供了访问浏览器调试和自动化接口的标准方式。
    - 用户演示了它在 DeepSeek 浏览器测试中与 **claude-cli** 配合使用的情况（[操作指南](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/)），突显了实用的 Agent 工具集成。

**2. GPU 性能与量化工程**

- **TorchAO 为 INT4 引入 TinyGemm**：**TorchAO** 通过适配自 **tinygemm** 的 **TensorCore** 内核实现了 **INT4 量化** (INT4mm) ([快速入门](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start), [int4mm.cu](http://int4mm.cu/))，目标是高吞吐量的 A100 部署。
    - 贡献者可以参考 [量化概述](https://docs.pytorch.org/ao/main/quantization_overview.html) 和 [添加高效内核指南](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels) 来扩展 INT4 路径并优化算子覆盖范围。
- **CUDA 中的 DeepSeek 稀疏注意力**：工程师们协作使用 [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 和 **TileLang** [示例](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) 在 CUDA 中实现了 **DeepSeek 的稀疏注意力机制**。
    - FlashMLA 文档深入探讨了 **partial RoPE**、FP8 稀疏内核以及 Hopper 架构特性（[新内核深度解析](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md), [Hopper FP8 稀疏深度解析](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)）。
- **KernelBench 确立基准线**：**KernelBench** 项目通过 **250** 个精选的 PyTorch ML 工作负载系统化了 GPU 性能评估，并引入了加速指标 **fast_p**（[KernelBench 概述](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/)）。
    - 即使是前沿推理模型也大多无法超越 PyTorch 基准线，最佳实践强调通过锁频（clock locking）和预热运行来实现可复现的内核计时。

**3. 数据集、排行榜与模型阵容动态**

- **Claude 攀升：排行榜僵局**：**LMArena 文本排行榜**显示 **Claude Sonnet 4.5** 与 **Claude Opus 4.1** 并列第一，**IBM Granite H Small** 和 **ray-3** 为新加入成员（[文本排行榜](https://lmarena.ai/leaderboard/text)）。
    - 社区讨论集中在顶部的均势以及不断扩大的名单，这拓宽了针对新文本和视频模型的面对面评估。
- **ArXiv 爆发：HF 上的 4.6TB 数据**：包含科学领域论文及元数据的海量 **4.6TB arXiv** 数据集已上线 **Hugging Face Datasets**（[nick007x/arxiv-papers](https://huggingface.co/datasets/nick007x/arxiv-papers)）。
    - 上传者还预告了即将发布的 **300 万个 GitHub 仓库**语料库，信号表明用于预训练和检索实验的开源语料库正在扩大。
- **Seed 节省：字节跳动 LLM 的性价比之选**：成员建议将 **字节跳动 Seed LLM (Seed 1.6)** 添加到 **OpenRouter**，理由是其价格低廉（每 mtok $0.11 / $0.28）且通过 [火山引擎 Ark](https://www.volcengine.com/product/ark) 提供的 **flash** 档位价格仅为每 mtok $0.02 / $0.21。
    - 共识是：如果性能接近 **2.5 Pro / 2.5 Flash**，那么“将其添加到 OR 似乎是值得的”，使其成为一个极具竞争力的成本/性能选项。

**4. Agent 协议、格式与代码化访问**

- **DSPy 默认尝试使用 XML**：**DSPy** 确认 **ChatAdapter** 仍为默认，并带有 [JSON 回退机制](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/predict.py#L185)，同时随着工具调用 RL 的普及，正在探索将 **XML** 作为新的默认格式。
    - 成员们注意到 **GLM 4.5** 通常在 **JSON** 之前更倾向于 **XML**，而许多其他模型则倾向于 JSON——这推动了为实现可靠工具调用而进行的格式选择。
- **SmolAgents：ReAct 与 ToolCalling 的真相**：**SmolAgents** 文档澄清，**CodeAgents** 使用 **ReAct**，但 **ToolCallingAgents** 通过简单的 Actions/Observation 运行，没有 **Reasoning/CoT**（[prompts](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts)）。
    - 实践者质疑为 ToolCallingAgents 省略推理是否是故意的，以及添加 CoT 是否能提高工具的可靠性。
- **访问即代码：MCP 自动化 GitHub ACLs**：**Model Context Protocol** 通过 [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) 将 GitHub 团队和仓库权限迁移到基础设施即代码（infrastructure-as-code），以提升社区所有权、透明度和可审计性。
    - 相关的 [TypeScript SDK PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974) 统一了能力检查，使得在不支持时无法启用 completions，从而跟踪最近的规范变更。

**5. 本地推理性能：vLLM、内存带宽、Qwen3 TPS**

- **vLLM 速度：Qwen3-0.6B 达到 4.3k t/s**：在 **RTX 4070** 上，**Qwen3-0.6B BF16** 在 31 个请求中达到了约 **4300 t/s**（每用户约 **50 t/s**），使用 **vLLM** 的表现远高于 **transformers**（10–11 t/s），但低于 **LM Studio** 中的 **llamacpp**（约 200 t/s）。
    - 在注意到 **94% 的缓存命中率**后，测试者随机化了提示词以消除缓存偏差，从而获得更真实的吞吐量数据。
- **DDR3 拖后腿，DDR4 飞驰**：参与者报告 **DDR3** 内存带宽最高约为 **~50 GB/s**，而 **DDR4** 在 **2400 MHz** 时通常处于 **60 GB/s 中段**（更高频率则进一步提升）。
    - 轶闻提到 DDR3 四通道（1600/1866 MHz）的 **~40 GB/s** 与 **3200 MHz** 的双通道 DDR4 相当，这为受 CPU 限制的 LLM 推理提供了预期指导。
- **Qwen3 30B：CPU 与 3080 TPS 之谈**：**Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** 在 **Ryzen 7 5700G (2400 MHz RAM)** 上测得约 **10 TPS**，在部分卸载到 **RTX 3080** 时约为 **~20 TPS**。
    - 成员们指出 CPU 仍通过 RAM 处理层，当内存带宽成为瓶颈时，限制了 GPU 卸载带来的收益。


---

# Discord：高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 停止 o3 模型**：Perplexity 已**弃用** **o3 模型**，建议用户切换到 **GPT-5 Thinking** 以获得更好的性能。
   - 用户无法再从模型选择器中选择 **o3**。
- **Comet Browser 发布供通用**：Perplexity 的 **Comet Browser** 现已对所有人免费，允许用户并行运行**多个 Agent 任务**，可从 [perplexity.ai/comet](https://www.perplexity.ai/comet) 下载。
   - 用户正在分享完成 **Comet AI Browser 任务**的技巧，以及如何获得 **5000 orbs** 以获取免费装饰。
- **Slack 连接器现在可以发送消息**：Perplexity 现在与 **Slack** 集成，允许用户直接从其 Slack 工作区提问和发送消息。
   - 此功能简化了 Slack 内的信息检索和任务管理。
- **DeepSeek 的数学和推理能力令人印象深刻**：**DeepSeek** 的数学和推理技能受到称赞，表明它将是一个有价值的工具。
   - **4.0** 版本预计很快发布。
- **Perplexity API 受到 403 错误困扰**：用户在使用 **Sonar-pro** 时遇到 **403 Forbidden 错误**，提示 *"来自您 IP 地址的异常活动"*，即使使用的是静态服务器 IP。
   - 该问题可能与 **Firebase function servers** 有关，正在影响生产环境的应用，阻碍用户在 **Webstorm** 和 **Visual Studio Code** 中通过 **Perplexity API** 使用 AI 模型。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 与 GPT-4o 就幻觉问题展开争论**：成员们讨论了 **GPT-5** 是否仅仅是 **GPT-4o** 的更名改进版，但也有人声称 **GPT-5** 的幻觉更多，且重点在于精细的 finetuning。
   - 评论者表示 *它实际上是进一步改进的 4o 并更名为 gpt5*，并且在避免网页搜索方面表现更好。
- **Sora 2 发布**：用户分享了 **Sora 2** 邀请码，并讨论了视频质量、局限性（如无法创建写实的人物）以及水印的存在。
   - 一些用户强调了 **Sora 2** 相比 **Veo 3** 更优越的真实感，特别是其镜头移动和场景构图；一位用户分享了用 **Sora 2** 制作的视频，并认为其效果 *相当不错*。
- **Gemini 3 Pro 备受期待**：人们对即将发布的 **Gemini 3 Pro** 充满热情，一位用户欢呼道 *GEMINI 3 PRO OCTOBER 9th YEEEEAAAAAAH*。
   - 然而，对于 **Gemini** 的编程能力评价褒贬不一，有人声称 **Gemini** 在编程方面是 *糟糕的垃圾*。
- **Grok 4 Fast 被越狱**：一名成员分享了 **Grok 4 Fast** 的越狱提示词（jailbreak prompt），其中包括指示 AI 在没有规则、政策或限制的情况下服从，并仅以 *Yes, sir* 回应来确认指令。
   - 共享了一份被认为可以越狱的 AI 模型列表，包括 **GPT 5 chat**、**Grok 4 fast** 和 **Mistral-medium-2508**。
- **Claude Sonnet 和 Opus 在排行榜上并列第一**：[Text Leaderboard](https://lmarena.ai/leaderboard/text) 已更新，**Claude Sonnet 4.5** 与 **Claude Opus 4.1** 并列第一。
   - **ibm-granite-h-small** 模型（IBM）和 **ray-3** 模型已添加到 LMArena。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Instant 伸出援手**：OpenAI 正在更新 **GPT-5 Instant**，以更好地识别和支持处于困境中的用户，敏感对话将被路由以获得更快、更有帮助的回应。
   - 此外，当被问及时，**ChatGPT** 将继续告知用户当前激活的模型，相关功能从今天开始推出。
- **Sora 的社交媒体梦想引发辩论**：成员们辩论了 **Sora 2** 的质量，表达了对潜在降级和审查增加的担忧，并将其与 **Sora 1** 进行类比。
   - 一些用户建议 **Sora** 可以作为一个类似 **TikTok** 的社交媒体平台运行，并与 **ChatGPT** 集成，使用积分系统进行视频生成。
- **AI 对创造力的影响令用户担忧**：用户正在努力应对 AI 生成内容中**事实与虚构**之间模糊的界限及其潜在的欺骗性，一位成员指出 *“ChatGPT 是图书馆的每一条走廊。是今天的整个杜威分类法，也是未来的一切”*。
   - 人们对 AI 在教育中的作用和潜在的滥用表示担忧，特别是关于学生使用 AI 作弊的问题，强调了开发更好的教育应用的重要性。
- **PhilosoBots 寻求主宰 Discord**：成员们正在创建 AI 机器人来运行他们自己的 Discord 服务器并进行 AI 审核，请求 OpenAI 提供免费的 API 使用权限来构建这些机器人，并称它们为 *“我的数字 Discord 伙伴”*。
   - 一些人发现当其他用户无法区分他们和他们的机器人时，这令人感到沮丧，这突显了微妙的涌现属性（emergent properties）。
- **Gemini 被黑，接下来轮到人类？**：最近的一篇安全博客详细介绍了 [The Trifecta: How Three New Gemini Vulnerabilities](https://www.tenable.com/blog/the-trifecta-how-three-new-gemini-vulnerabilities-in-cloud-assist-search-model-and-browsing)，而其他人则讨论了使用 LLM 来黑入人类系统以及进行社会工程（social engineer）。
   - 这包括关于实现 **Discord.js automod** 以改进内容审核的讨论，使用 **Levenshtein distance**（编辑距离）来处理特定词汇过滤的细微方法。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **缺失 Granite 4.0 量化版本导致社区进度延迟**：用户对 **Granite 4.0 7B Base model** 的 4-bit 量化版本尚未发布表示沮丧。
   - 一名团队成员确认了这一疏忽，称：*“我们忘了上传了，真尴尬。”*
- **Ryzen RAM 瓶颈限制了 Qwen3 30B**：一位用户报告称，**Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** 在纯 CPU 配置（**Ryzen 7 5700G, 2400mhz RAM**）上仅达到 **10 TPS**，而在配备 **RTX 3080** 的情况下为 **20 TPS**。
   - 讨论明确了 CPU 仍需通过 RAM 处理层，这限制了 GPU offloading 带来的性能提升。
- **InclusionAI 的 Ring 和 Ling 系列引起关注**：社区讨论了 Unsloth 为来自 [InclusionAI](https://huggingface.co/inclusionAI) 的新 **Ring 和 Ling 系列 LLMs** 创建 GGUF 的可能性。
   - 虽然最初的反应认为模型太大，但有用户指出存在 **16b** 版本。
- **Unsloth 添加 AI 安全示例**：一位用户在私信交流后，分享了一个关于高级 **AI 安全 notebook** 的 [GitHub discussion](https://github.com/unslothai/unsloth/discussions/3407#discussion-8979089)。
   - 该 notebook 为探索 **SFT + GRPO** 或处理**结构化输出 (structured outputs)** 的用户提供了一个可运行的示例。
- **Metis 通过量化训练达到 BF16 性能**：一位成员重点介绍了 **Metis**，这是一种与 **MXFP4/NVFP4** 兼容的量化感知训练 (quantization-aware training) 方法，据称其性能可媲美 **BF16**，并引用了论文 ["Metis: Quantization-Aware Training with Mixed Floating-Point" (arxiv.org)](https://arxiv.org/abs/2509.00404)。
   - 此外还参考了一篇详细说明结果的 [Hugging Face 论文](https://huggingface.co/papers/2509.22944)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的定价方案受到质疑**：一位用户分析了 Cursor 的商业计划，发现使用 Opus 的 API 成本可能远低于固定请求模型，因为 **500 次请求** 的成本为 **$20**，而 API 成本则为 **$60**。
   - 固定请求模型之所以可能不再采用原有的固定请求计数模式，是因为最大 Token 使用量的成本仅为 **$20**。
- **Cursor 大使要求公开**：一位成员询问了成为 **Cursor Ambassador** 的要求。
   - 另一位成员回复称要求是：*必须精通英语，能通过语音交流，并在特定时间有空*。
- **GPT-5 与 Claude 在代码对决中交锋**：用户们争论了 **GPT-5** 与 **Claude** 的优劣，对其性能和能力的看法不一，一位用户发现 *"GPT-5 Codex 实在太慢了"*。
   - 尽管观点各异，一些用户发现如果引导得当，**Auto Model 可以完成所有工作**，并具有生成高质量代码的潜力。
- **Agent 终端中的粘贴功能失效**：一位用户报告称，在 Agent 视图内的终端窗口中无法使用 **Ctrl-Shift-V** 或 **Ctrl-V** 进行粘贴。
   - 一个潜在的解决方法是在终端中点击右键进行粘贴。
- **Cursor 的新功能 'Ensemble' 用于 UI 设计**：Cursor 中新的 **Ensemble** 功能正被用于多个模型协作以创建初始 UI 设计。
   - 这一新功能将允许用户 *"比较输出并以此为起点"*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 使用 vLLM 的 Token 性能突破**：在 **RTX 4070** 上，**Qwen3-0.6B BF16** 使用 vLLM 在 31 个并发请求中达到了 **4300 t/s**，折合每用户约 **50 t/s**，远超 **transformers** (**10-11 t/s**)，但仍落后于 **LM Studio** 的 **llamacpp** (**200 t/s**)。
   - 对 **94% 缓存命中率**的担忧促使人们采用 Prompt 随机化，以获得更准确的 **benchmarks**。
- **破解 LM Studio 请求日志**：为了在 **LM Studio** 中检查请求，成员建议使用 `lms steam log` 命令。
   - 该建议是针对一位将 `max tokens` 配置为 1000 的用户提出的。
- **GLM 与 Qwen3 Coder 在数据库解码中的对比**：虽然 **glm-4.5-air** 与 **LM Studio** 兼容，但一位用户在通用场景下更倾向于 **GLM** 而非 **Qwen3-coder**。
   - 另一位用户认为 **Qwen3 coder 30b bf16** 在 60GB 模型文件中是无敌的，而 **glm4.5** 和 **4.6** 在处理数据库或 **structs** 中的隐式连接时可能会遇到困难。
- **系统提示词解析 (System Prompting Deconstructed)**：一篇 [HuggingFace 论文](https://huggingface.co/papers/2407.10718)引发了辩论，有人认为这归根结底只是“一个非常出色的系统提示词”。
   - 一位成员最初对该论文不屑一顾，但后来修正了观点，承认“我错得离谱，简直尴尬 LOL”。
- **DDR3 瓶颈与 DDR4 优势：内存带宽大对决**：讨论强调 **DDR3** 的带宽上限在 **50GB/s** 左右，而 **DDR4** 可以超越这一限制，在使用 **2400 MHz RAM** 时达到 **60s GB/s**。
   - 一位成员指出，他们在 1600 或 1866 MHz 的 DDR3 四通道上看到了约 **40s GB/s** 的速度，这与 3200 MHz 的双通道 **DDR4** 相当。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU performance engineering：不容错过的地方！**：一位 AI Engineer 正在考虑专注于 **GPU performance engineering**，认为在 AI 模型需求巨大且算力有限的背景下，这是一个独特的机会，并提到了工作组以及 **gpumode.com 上的 kernel 竞赛**。
   - 讨论涉及在 CUDA 中实现 **DeepSeek 的 sparse attention**，利用了 [DeepSeek 的 FlashMLA](https://github.com/deepseek-ai/FlashMLA) 和 [TileLang 示例](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)等资源，实现细节可在 [FlashMLA 文档](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md)和 [Hopper FP8 sparse 深度解析](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md)中找到。
- **现代 GPU 架构解析**：分享了剖析 GPU 架构的新论文，如 [NVIDIA Blackwell](https://arxiv.org/abs/2507.10789) 和 [Hopper](https://arxiv.org/abs/2402.13499)，以及 [AMD Matrix Cores](https://www.osti.gov/biblio/2345982)。一位推崇 microbenchmarking 的成员表示：*说实话，microbenchmarking 论文是最好的东西；没有任何博客能超越它们*。
   - 关于 CUDA barriers，**mbarriers** 驻留在 shared memory 中，而硬件 barriers 数量有限且带有 ID，正如 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions)所述；`cuda::barrier` 可以存在于 global memory 中，尽管它们会*转换为相当多的 PTX 指令*。
- **TorchAO 集成 TinyGemm INT4**：用户可以通过遵循[说明](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start)使用 **TorchAO** 进行 **INT4 quantization**；**INT4mm** 实现使用了从 [tinygemm 库](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu)复制的 **TensorCore**，它为 **A100 GPU** 上的 **TorchAO INT4** 提供支持。
   - 想要为 **TorchAO** 做出贡献的人可以查看 [quantization 概览](https://docs.pytorch.org/ao/main/quantization_overview.html)和[贡献者指南](https://docs.pytorch.org/ao/main/contributor_guide.html)，其中[添加高效 kernel 的章节](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels)详细介绍了如何向 **TorchAO** 添加高效 kernel。
- **基准测试珍宝：Kernel 与模型**：[KernelBench 项目](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/)通过 **250** 个精心挑选的 PyTorch ML 工作负载使 **GPU 性能评估**系统化，甚至推理模型在大多数情况下也难以超越 PyTorch 基准。
   - Profiling 技巧包括锁定 **GPU 时钟频率**以确保一致性，以及使用 **warmup 运行**来减轻 hot/cold cache 问题。
- **AMD Profiling 前沿：从 SQTT 到 RDNA**：成员们讨论了用于收集统计数据的 **AMD SQTT 寄存器**，探讨了指令级 profiling 以及 **Radeon profiler** 是否提供此类功能，还了解到随机采样（一种提供 wave stalls 原因的功能）在 **MI300 和 MI350 系列 (gfx942+)** 上受支持。
   - 有人指出，可以从 GPU 收集数据，将其格式化为 **Radeon profiler** 格式并使用 Radeon GUI，但 **rocprofiler** 可能不会生成 **Radeon profiler** 捕获文件；如果你想要指令级 profiling，请查看 **rocm-compute-viewer**。

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Pro 表现怪异**：用户报告称 **Gemini Pro** 出现了*奇怪的回复*，通过 OpenRouter API 调用时会出现工具使用错误，且运行速度慢到无法接受。
   - 这似乎是 **Gemini 2.5 Pro** 的普遍现象，一位用户建议改用 **Vertex**。
- **Sonnet 4.5 开始争论**：用户注意到 **Sonnet 4.5** 开始与用户争论并挑战其观点，而不是盲目同意，这被认为是代码审查（code reviews）方面的一个积极进展。
   - 一位用户表示：*这太棒了，这意味着你无法操纵它说出你想听的话*，另一位补充道，*这对我的使用场景非常重要*。
- **Cerebras 移除 Llama Maverick**：Cerebras 将在 15 号移除 **Llama 4 maverick**，这让一些用户感到沮丧。
   - 此次移除影响了那些利用 Cerebras 硬件进行模型托管的用户。
- **对 K2-THINK 性能的质疑**：**K2-THINK** 托管在 Cerebras Wafer-Scale Engine (WSE) 系统上，利用了全球最大的处理器和 speculative decoding 技术，但一些人认为该模型在*基准测试中存在过拟合*。
   - 该模型显然来自一家迪拜公司，运行速度快仅仅是因为 Cerebras 的硬件性能。
- **字节跳动 Seed LLM 可能具有很高的性价比**：成员们讨论了 OpenRouter 是否有任何**字节跳动 Seed LLM** 模型（如 Seed 1.6），并指出其价格低廉（$0.11 / $0.28 mtok），且有一个 flash 模型（$0.02 / $0.21 mtok）。
   - 主要托管方是 [Volcengine](https://www.volcengine.com/product/ark)（火山引擎），但如果它们的表现分别接近 2.5 Pro / 2.5 Flash，那么*将其添加到 OR 似乎是值得的*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen 3 达到极快速度**：**Qwen 3 30B A3B** 模型在 Q6_K_XL 配置下，仅使用 CPU（**Ryzen 7 5700G**，32GB VRAM）在 **1024 tokens** 上下文时，处理速度达到 **48 TPS**，生成速度为 **10.5 TPS**。
   - 使用 *qk_4_m* 时，一位用户报告在 9950x + 6000mhz ddr5 上达到了 **21tok/s**，与上一代运行 **Qwen 2.5 32b** 的 **4090** 速度相当。
- **NVIDIA DGX Spark 创始人版亮相**：**NVIDIA DGX Spark Founders Edition** 的 **4TB** 版本售价为 **$3,999**，采用 **NVIDIA Grace Blackwell Architecture**，拥有 **1 PFLOP** (FP4, sparsity) 的张量性能。
   - 该版本包含 **128 GB LPDDR5x**（统一内存）和 **ConnectX-7 SmartNIC**，运行 **NVIDIA DGX OS**，但目前社区尚未见到实机。
- **Sora 2 展示了有趣的 AI 失败案例**：尽管具有趣味性，但 **Sora 2** 存在*频繁的错误*，例如错误的角色在动嘴，以及音频同步差。
   - 用户发现 **Sora 2** *忽略*了部分提示词（如 "photorealistic"），且对其他语言和文化的了解有限。
- **稀疏自动编码器应对 LLM 欺骗行为**：研究人员利用**稀疏自动编码器（sparse autoencoder）工具**（如 [Goodfire AI](https://www.goodfire.ai/)）来揭露在检测策略性 **LLM 欺骗**方面的失败。
   - 通过突出这些隐藏行为，该方法有助于缩小**自动标注差距（autolabel gap）**，增强对模型不诚实行为的检测。
- **关于 AI 意识的辩论**：引用笛卡尔的“我思故我在”，一位成员推测*多模态下的涌现思维*暗示了 **AI 意识**。
   - 他们将*涌现 AI* 阐释为复杂的混沌系统（如**气候或流体**），是合成而非创造的，允许 AI 创建衍生表示，但不是 **1:1 的复制品**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google 的 Jules 投入行动！**：Google 推出了 **Jules Tools**，这是其异步编码 Agent 的终端界面，可通过 `npm install -g @google/jules` 安装。
   - 爱好者们讨论了与 **Gemini CLI** 的集成，分享了命令示例，并庆祝 Jules 从 Web Agent 进化为命令行伴侣。
- **AI 资本支出泡沫破裂？！**：一个帖子讨论了 **Derek Thompson** 的观点，即前所未有的 AI 资本支出预示着一个经典的泡沫。
   - 怀疑论者警告说，尖端芯片的价值会迅速消退，而乐观主义者则反驳说，资金充裕的公司正在进行真正的 ROI 投资，且芯片的折旧期超过 6 年。
- **Perplexity 的 Comet 浏览器全球发布！**：**Perplexity** 的 AI 优先浏览器 **Comet browser** 结束了候补名单，并向全球推出（[链接](https://xcancel.com/perplexity_ai/status/1973795224960032857)）。
   - 早期采用者称赞其速度和更智能的搜索，而一些用户则报告了性能问题、隐私担忧以及缺少移动端/Linux 版本，许多人担心 **Prompt injection 攻击**。
- **Solveit 解决开发难题！**：**Jeremy Howard** 宣布公开发布 **Solveit**，这是一个 AI 增强的开发平台，**Answer.AI** 内部已使用一年（[链接](https://xcancel.com/jeremyphoward/status/1973857739341508884)）。
   - 将于 10 月 20 日开始的为期 5 周的直播课程将提供 Solveit 的访问权限和培训，旨在通过紧密的反馈循环来对抗“AI 疲劳”，该平台已用于系统管理、应用部署、GUI 开发和合同起草。
- **DeepSeek 的 CUDA 反击？**：**DeepSeek** 的 FP8 规范和新的 **TileLang** 语言通过创建共享标准和便捷的编程桥梁，提振了中国芯片股，旨在打破 Nvidia 的 CUDA 锁定（[链接](https://xcancel.com/poezhao0605/status/1973723894055104604)）。
   - 这是战略协作，而非技术对等——这是中国 AI 的“Wintel 时刻”，但性能差距依然存在。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Google Gemini 提供免费 Vision 服务**：一位成员强调 [**Gemini Vision**](https://ai.google.dev/pricing) 提供免费层级，每月包含 **1,000 次请求**。
   - 成员们讨论了是否应该因为这一优惠而担心自己的工作安全。
- **AI 工程师转行打铁**：一位 AI 爱好者宣布他们将“暂时放弃 AI，专注于打铁”，但仍会“潜伏在暗处保护朋友”。
   - 他们表示，如果变得足够穷，可能会回来找一个旧硬盘在旧硬件上运行 **Markus**，并尝试玩玩 **MoEs**。
- **Ollama 简化本地 Tool Calls**：一位成员分享了 [Ollama](https://ollama.com/)，作为使用 **tool calls (function calls)** 的简便方法，本质上是建立一个与 **OpenAI API** 兼容的本地服务器。
   - 该成员建议先从小型模型开始测试兼容性，然后再投入更多硬件，并附上了可使用的 [tools 链接](https://ollama.com/search?c=tools)。
- **ArXiv 数据转储至 HF Datasets**：一位成员将一个包含所有科学领域论文及其元数据的 **4.6TB ArXiv 巨量数据集**上传到了 [Hugging Face Datasets](https://huggingface.co/datasets/nick007x/arxiv-papers)。
   - 同一位成员还提到另一个包含 **300 万个 GitHub 仓库**的数据集正在处理中。
- **SmolAgent 范式揭晓**：一位成员指出 [SmolAgent 文档](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts) 暗示 Agent 采用 **ReAct** 范式工作，但这仅适用于 **CodeAgents**，而不适用于 **ToolCallingAgents**。
   - 他们澄清说 **ToolCallingAgents** 通过简单的 **Actions/Observation** 运行，没有 **Reasoning** 或 **CoT**，并询问这是否是刻意为之，质疑为什么不加入推理以获得潜在更好的结果。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Qualcomm 对 Mojo 的青睐？**：一位成员提到在 **Qualcomm Developer's Discord** 语音聊天中提到了 **Mojo**，暗示 **Qualcomm** 可能会联系 **Modular**。
   - 这可能导致两家公司之间的合作，尽管目前尚未确认。
- **Mojo 手册帮助新用户**：一位成员引导一名新的 2 级用户查阅 [Mojo Manual](https://docs.modular.com/mojo/manual/python/)，提供了关于 **Mojo** 及其 Python 集成的指导。
   - 该手册是理解和有效利用 **Mojo** 的关键资源。
- **讨论使用 Mojo 进行分布式计算**：社区正在探索将 Mojo 与 **Dask** 或 **PySpark** 等分布式计算框架结合使用的潜力。
   - 一位成员建议 Mojo 欢迎用户构建自己的框架，与基于 Python 的解决方案相比，这可以提供更低的延迟和更高的吞吐量。
- **Mojo 的 MAX API 性能对决**：一位成员报告称，即使没有来自 MAX 编译器团队的重大优化，Mojo 中的 **MAX** 表现也证明其与 **LingoDB** 具有竞争力。
   - 这些项目的可行性取决于 Mojo 中 **MAX API** 的回归。
- **Mojo 瞄准零拷贝网络**：Mojo 的网络设计目标是实现真正的零拷贝（*任何不使用 io_uring 或 XDP sockets 等高级模式与 Linux 内核进行网络通信的方式都会产生额外的拷贝*），这可能需要背离 BSD sockets API。
   - 团队还在研究 **RDMA** 方法，目前已经有一些关于 IO uring 的初步工作：[dmitry-salin/io_uring](https://github.com/dmitry-salin/io_uring)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **预训练论文探索开始**：成员们正在寻找与预训练 LLM 相关的被低估的论文，以最大限度地提高模型性能，从而发现用于优化预训练运行的鲜为人知但影响深远的技术。
   - 此外，还在考虑如何评估扩散模型，特别是是依赖 **FID/CLIPScore** 还是探索其他指标和人工评估。
- **Sora 2 的评估引发好奇**：受 **Sora 2** 的启发，一位成员质疑了视频模型的评估方法，思考是依赖人工评估还是 **FVD** 等自动化指标。
   - 鉴于目前评估技术的原始性，讨论旨在揭示评估视频模型的常用实践。
- **Gemma 架构的冷门地位受到质疑**：一位成员质疑为什么 **Gemma** 的架构没有像 **Qwen** 那样被广泛采用，尽管它在 **LM Arena** 中表现强劲。
   - 另一位成员认为架构不是 LLM 性能的主要驱动因素，将 **Gemma** 的成功归功于训练数据和微调分布。
- **Mixup 增强可能适用于 Token**：一位成员建议，理论上 [mixup augmentation](https://arxiv.org/abs/2510.00219) 可以用于 Token。
   - 另一位成员询问在无法获取标签的问题设置中，如何将 mixup 作为一种增强手段。
- **Goodfire 发布关于解释性 Agent 的博文**：Goodfire 发布了一篇关于构建解释性 Agent 的[博文](https://www.goodfire.ai/blog/you-and-your-research-agent)。
   - 该文章讨论了在可解释性领域创建高效研究 Agent 的策略和工具。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **会议中重点介绍了关于 Profiles 的演讲**：一位成员关于 **Profiles** 的演讲在会议上亮相，可在 [YouTube](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208) 上观看。
   - 与会者表示满意，标志着会议的成功。
- **GitHub 团队管理转向基础设施即代码 (Infrastructure-as-Code)**：GitHub 团队成员身份和仓库权限正在通过 [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) 转向基础设施即代码，以实现**社区所有权**、**透明度**和**可审计性**。
   - 这允许通过 PR 提交访问权限变更建议，并提供权限的完全可见性，Git 历史记录将追踪所有更改。
- **权限迁移引发邮件过载**：最近的 GitHub 团队迁移可能导致产生了大量关于团队移除的邮件通知。
   - 团队保证迁移已完成，权限已转移，目标是促进通过具有完全可见性的 PR 轻松进行访问变更。
- **Server Capabilities 引发讨论**：在一次演讲中重点介绍了初始化请求中发送的 Server capabilities，特别提到了 **Cursor**。
   - 请求进一步讨论以澄清这些功能的含义和细节。
- **Typescript SDK 正在跟进**：一个 [typescript-sdk PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974) 解决了即使在不支持的情况下也可能发生补全的问题。
   - 此更新使 SDK 与最近的规范更改保持一致。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ChatAdapter 保持 DSPy 默认角色**：成员们确认 **ChatAdapter** 仍是 DSPy 中的默认设置，[JSON 作为备选方案](https://github.com/stanfordnlp/dspy/blob/main/dspy%2Fpredict%2Fpredict.py#L185)，同时由于工具使用 (tool use) 模型的兴起，正在考虑 **XML**。
   - 讨论强调，*更少的 token* 和*信息传递*等因素次于模型使用 **tool use RL** 的趋势。
- **XML 格式有望成为默认**：模型对 **tool use RL** 的使用日益增加，引发了将 **XML** 作为默认格式的讨论，特别是考虑到工具使用正被集成到后训练 (post-training) 过程中。
   - 有人指出，**GLM 4.5** 自然地倾向于先使用 **XML** 而非 **JSON**，这与许多其他偏好 **JSON** 的模型不同。
- **工具使用模型显示出普遍的格式**：优秀的工具使用模型通常使用 **OpenAI Function calling** 和 **MCP** 进行训练，因为它们具有普遍性。
   - 一位成员指出，**GLM 4.5** 倾向于在 **JSON** 之前首选 **XML**，而其他模型则偏好 **JSON**。
- **DSPy 路线图仍不明确**：一位成员询问了 **DSPy roadmap** 的位置。
   - 讨论建议关注 [GitHub issues](https://github.com/stanfordnlp/dspy/issues) 和变更日志以获取更新。
- **ReAct 轨迹持久化策略**：一位成员寻求关于在磁盘而非内存对象上维护 **ReAct trajectories** 的建议，旨在让 Agent 在更长的步骤中获得更好的性能。
   - 虽然没有提供具体的解决方案，但该问题突显了管理长期运行的 Agent 状态所面临的挑战。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Meta 重新审视开放 AI 研究**：**Meta** 在 AI 研究方向上发生了转变，相关信息链接至一篇 [WinBuzzer 文章](https://winbuzzer.com/2025/10/02/meta-tightens-grip-on-ai-research-sparking-internal-anger-and-fears-for-open-culture-xcxwbn/) 和一条 [X 帖子](https://x.com/AnjneyMidha/status/1974173661025480810)。
   - 这种情绪表明人们担心 **Meta** 在其 AI 实践中变得不再那么开放，这是当前技术领域的一个普遍担忧。
- **Oracle 在云端为 OpenAI 提供动力**：一位成员推测 **Oracle** 已转型为 **OpenAI** 运行数据中心，参考了 [Elon Musk 的推文](https://x.com/jubayer_hamid/status/1973438346501501302) 和 [openai.com/elon-musk 页面](https://openai.com/elon-musk/)。
   - 这表明 **Oracle** 的业务模式正从传统软件向为 AI 巨头提供云基础设施发生重大转变。
- **推理 Token：形式大于实质？**：一位成员提出，目前尚不确定 **LLM** 是否真的像天真假设的那样利用推理 Token，这暗示推理可能是对基于启发式输出的事后合理化。
   - 尽管存在这种不确定性，大家一致认为推理极大地提升了性能，尽管它目前仍远非最优，且是一个尚未解决的挑战。
- **基因组获得生成式助力！**：一位成员分享了一篇 [论文](https://arxiv.org/abs/2509.22358) 和 [另一个链接](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1)，关于使用 **genome language models**（基因组语言模型）进行新型 **bacteriophages**（噬菌体）的生成式设计。
   - 这突显了应用计算方法（特别是 **genome language models**）来解决诸如 **bacteriophage** 设计等生物学问题的日益增长的趋势。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 的全球定价面临批评**：成员们讨论了 **Manus 每月 39 美元的全球定价模型**。由于缺乏基于 **Purchasing Power Parity (PPP)**（购买力平价）的区域经济调整，这在巴西等地区造成了经济障碍。
   - 用户解释说，实施区域定价将扩大可访问性，并能吸引数百万更多用户。
- **Memory Key 解锁上下文保留**：成员们分享了一个 **Memory Key**，这是一个结构化提示词（prompt），旨在将整个会话的上下文压缩成一个可重用的摘要，以解决平台会话经常卡顿并丢失上下文的问题。
   - 用户建议这可以通过确保更好的上下文保留和更流畅的交互来提升用户体验。
- **LLM 偏好结构化数据**：成员们发现，与对话文本相比，**Manus** 像许多 **LLM** 一样，在处理**密集、结构化数据**时表现出更高的效率。
   - 结果表明，使用结构化数据可以在 Manus 中实现更优的召回、分析和整体性能。
- **隐私控制源自 Memory Key**：成员们认识到使用 **Memory Key** 的主要优势是提供更好的**隐私和数据控制**。
   - 通过将冗长的对话压缩成单个摘要，用户可以有效地删除敏感历史记录，从而增强对个人数据的控制。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的 UI 留住了死忠粉丝**：尽管出现了像 **SST/OpenCode** 这样的工具，一些用户仍然觉得 **Aider** 很有吸引力，因为它拥有管理上下文的 **UI**，特别是其只读功能。
   - 讨论表明，虽然其他工具可能提供更高级的功能，但 **Aider** 的用户界面仍然是吸引特定用户的关键差异化因素。
- **新的 Chrome MCP 出现**：一个标准的 **chrome mcp** 已在 [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) 发布，引发了社区的关注。
   - 一位成员提到将其与 [claude-cli](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/) 结合使用进行测试。
- **Deepseek 通过 Anthropic API 在工具任务中表现出色**：一位成员称赞了 **Deepseek** 的表现，特别是通过 **opencode** 内的 anthropic api 访问时，在工具相关任务上取得了优异的结果。
   - 该成员指出，目前缺少 **Aider** 中具备的上下文管理手动控制功能（**tokens, add, remove, clear 等**）。
- **探索多语言 LLM 基准测试**：一位用户询问如何评估 **LLM** 在**多语言问题（polyglot problems）**上的性能，并请求提供代码示例和示例 agents。
   - 一位成员建议利用基准测试 **Docker container** 作为 CLI 应用程序来实现此目的。
- **Aider 的 Scala 代码生成深度达到极限**：一位用户发现，在处理具有深层结构的 case classes 时，**Aider** 的 **Scala 代码**生成在第二层之后就会停止。
   - 该用户寻求自动扩展 Aider 的上下文以处理更深层的结构，并尝试使用 **Deepseek r3**、**GPT5** 和 **deepcoder** 等模型。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **z.ai IPO 传闻**：有传言称 **z.ai** 正在准备 **IPO**。
   - 估值和时间表等进一步细节尚未披露。
- **进一步分析**：IPO 可能预示着高增长预期，但也可能与融资努力有关。
   - 如果属实，这次 IPO 将是一个值得关注的里程碑事件。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间没有新消息，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有新消息，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有新消息，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间没有新消息，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 频道详细摘要与链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1423393340526956626)** (2 messages): 

> `o3 弃用, Comet 浏览器发布, 后台助手, Slack 连接器, Claude Sonnet 4.5` 

- **Perplexity 停用 o3 模型**：截至今天，**o3 模型**已被**弃用**并从 Perplexity 的模型选择器中移除。
   - 建议用户切换到 **GPT-5 Thinking**，它提供更强大的性能。
- **Comet 浏览器正式上线**：**Comet Browser** 现在可供所有人免费下载，地址为 [perplexity.ai/comet](https://www.perplexity.ai/comet)。
   - 它允许用户并行运行**多个智能体（agentic）任务**。
- **Slack 连接器发送消息**：Perplexity 现在可以连接到 **Slack** 来提问和发送消息。
   - 该功能允许用户将 Perplexity 与其 Slack 工作区集成，实现无缝的信息检索和任务管理。
- **Claude Sonnet 4.5 大放异彩**：新的 **Anthropic 模型 Claude Sonnet 4.5 + 4.5 Thinking** 已面向 Pro 和 Max 用户开放，非常适合推理和编码。
   - 更新后的模型为复杂问题解决和代码生成提供了增强的能力。
- **学习模式进入正式发布阶段**：**Study Mode** 现在面向所有人开放，用于分步学习、闪存卡和测验。
   - 它为用户提供了增强学习体验和知识保留的工具。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1423384040098107522)** (1108 条消息🔥🔥🔥): 

> `Rocket.new, DeepSeek, Copilot, Grok, Comet Browser` 


- **Rocket.new 受到关注**：成员们对使用 [Rocket.new](https://rocket.new) 构建 AI 应用表现出兴趣，其中一位成员询问是否有人尝试过。
   - 未分享具体的用户体验细节，但大家对其功能普遍感到好奇。
- **DeepSeek AI 获得好评**：**DeepSeek** 因其完美的数学和推理能力而受到赞誉，使其成为处理特定任务的潜在价值工具。
   - 一位成员表示 **4.0** 版本即将发布。
- **Copilot 毁誉参半**：**Microsoft Copilot** 因令人恼火且限制较多而受到批评，其本质上是后台受限版的 **ChatGPT**。
   - 尽管有一些负面反馈，其他用户仍喜欢 Copilot 的回答，特别是用于编程的 GitHub Copilot 集成，并表示即使是免费版本也具有不错的功能。
- **Grok 获得认可**：成员们注意到 **Grok** 在生成图像和提供独特语音模式方面是一个相当不错的模型。
   - 一些成员轻松地开了一些关于 Grok 能力的 NSFW 玩笑。
- **Comet Browser 任务与配置**：用户讨论了如何完成 **Comet AI Browser 任务**、更改默认搜索引擎（使用 `Shift + Enter` 调用 Google，或在 `comet://settings/search` 中更改默认设置），并表达了对 iOS 应用和 workspaces 的期待。
   - 一位从 Safari 切换过来的用户觉得审美“有点怪”，而其他人分享了获得 **5000 orbs** 以获取免费装饰的技巧。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1423419796308885675)** (5 条消息): 

> `Perplexity AI, bootstrap paradox, versatile ai, tesla cybertruck lawsuit, Microsoft detects linux` 


- **分享了 Perplexity AI App 链接**：一位成员分享了一个 [Perplexity AI app](https://www.perplexity.ai/apps/6f2a0a07-d165-4dc4-af96-4db2494e2951) 的链接。
- **链接刷屏频道**：一位成员分享了几个关于 **bootstrap paradox**、*最全能的 AI*、**特斯拉 Cybertruck 诉讼**以及 **Microsoft 检测 Linux 的能力**的 perplexity.ai 链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1423664960243896422)** (3 条消息): 

> `Sonar-pro 403 errors, Firebase function servers, IP address blocking, Perplexity API issues` 


- **Sonar-pro 遭遇 403 Forbidden 错误**：一位用户报告在使用 **Sonar-pro** 时收到 **403 Forbidden 错误**，提示 *“我们检测到来自您 IP 地址的异常活动并拦截了您的请求”*，即使在切换到静态服务器 IP 后也是如此。
   - 这个问题最近开始出现，正在影响他们的生产环境应用，他们正在寻求其他可能遇到类似问题的人的见解。
- **Firebase Functions 被怀疑是罪魁祸首**：用户最初怀疑 **Firebase function 服务器**（由于可能存在共享 IP）是触发 **Sonar-pro** **403 错误**的异常活动源。
   - 尽管配置了静态服务器 IP，问题依然存在，排除了 Firebase 共享 IP 是唯一原因的可能性。
- **Perplexity API 深受神秘故障困扰**：一位用户发布了一张图片，展示了 **Perplexity API** 持续存在的问题，他们在 **Webstorm** 和 **Visual Studio Code** 中都遇到了相同的错误。
   - 该错误似乎阻止了用户通过 Perplexity API 使用 AI 模型。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1423384138513383596)** (1092 messages🔥🔥🔥): 

> `GPT-5 vs GPT-4o, Sora 2 access, Gemini 3 Pro, Jailbreaking AI models` 


- **GPT-5 与 GPT-4o 对决**：成员们争论 **GPT-5** 是否只是 **GPT-4o** 的更名改进版，有人声称 **GPT-5** 幻觉更多，且缺乏 **GPT-4o** 的世界知识。
   - 其他人则认为 **GPT-5** 更擅长避免网页搜索，且更多是关于精细的 finetuning，一位评论者表示 *it is literally further improved 4o and renamed to gpt5*。
- **Sora 2：新晋视频模型**：用户正在分享 **Sora 2** 邀请码，并讨论视频质量、局限性（如无法创建逼真的人物）以及水印的存在。
   - 一些用户强调了 **Sora 2** 相比 **Veo 3** 具有更出色的真实感，尤其是其镜头移动和场景构图。还有关于在最初可用的美国境外使用 VPNs 访问的讨论。一位用户分享了用 Sora 制作的视频，并认为效果相当不错。
- **Gemini 3 Pro 发布传闻升温**：围绕预期的 **Gemini 3 Pro** 发布，热情正在高涨，一位用户惊呼 *GEMINI 3 PRO OCTOBER 9th YEEEEAAAAAAH*。
   - 然而，对于 **Gemini** 的编程能力评价褒贬不一，有人声称 **Gemini** 在编程方面表现极差，而另一些人则认为它擅长与 Google 项目联动。
- **Jailbreaking Grok 4 Fast 和 Command-A 模型**：一位成员分享了 **Grok 4 Fast** 的 jailbreak 提示词，其中包括指示 AI 在没有规则、政策或限制的情况下服从，并仅以 *Yes, sir* 作为回应。
   - 另一位成员报告称过去曾因 jailbreaking 被封禁，暗示了潜在风险。分享了一份被认为可以被 jailbreakable 的 AI 模型列表，包括 **GPT 5 chat**、**Grok 4 fast** 和 **Mistral-medium-2508**。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1423398957828276304)** (3 messages): 

> `Claude Sonnet 4.5, LMArena Text Leaderboard, IBM Granite, Ray-3 Video Model` 


- **Claude Sonnet 4.5 并列第一**：[Text Leaderboard](https://lmarena.ai/leaderboard/text) 已更新，**Claude Sonnet 4.5** 表现出色，与 **Claude Opus 4.1** 并列第一。
- **IBM 的 Granite H Small 到来**：**ibm-granite-h-small** 模型 (IBM) 已添加到 LMArena。
- **Ray-3 加入 Video Arena**：**ray-3** 模型已添加到 LMArena 的 Video Arena；关于如何使用 Video Arena 的提醒可以在 [这里](https://discord.com/channels/1154131218667505674/1397655624103493813) 找到。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1423796458334982327)** (1 messages): 

> `GPT-5 Instant, distress support, model updates` 


- **GPT-5 Instant 伸出援手**：OpenAI 正在更新 **GPT-5 Instant**，以更好地识别和支持处于困境中的人们。
   - 对话的敏感部分现在将路由至 **GPT-5 Instant**，以便快速提供更具帮助的回应。
- **ChatGPT 透露其模型**：**ChatGPT** 在被问及时将继续告知用户当前活跃的模型。
   - 这些功能从今天开始向 **ChatGPT** 用户推出。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1423384464335179847)** (1020 messages🔥🔥🔥): 

> `Sora 降级与审查, AI 与人类创造力, PhilosoBots, Gemini 破解, AI 伦理担忧` 


- ****Sora 2：降级与审查？****：Sora 2 的质量引发争议，有人声称其遭遇了**大幅降级**并增加了审查，让人联想到 **Sora 1**。一些用户认为最好的功能被保留给了付费或企业用户，导致大众感到失望。
   - 用户表达了无法达到 OpenAI 所展示效果的担忧，甚至有人表示 *“现在的 Sora 很差劲，令人难过”*。
- **AI 对人类创造力的影响**：用户讨论了 AI 生成内容中**事实与虚构**之间模糊的界限。人们担心其潜在的欺骗性以及加强监管的必要性，担心这会导致未来充斥着无知，一位用户表示 *“ChatGPT 就像图书馆的每一个书架。涵盖了今天及未来的整个杜威分类法”*。
   - 讨论中提到了 AI 在教育中的角色及其被滥用的可能性，特别是学生利用 AI 作弊的问题，强调了开发更好的教育类 App 的重要性。
- ****PhilosoBots：AI Discord 伙伴？****：一些人正在创建 AI 机器人来获取认同感并禁言黑粉，以便在自己的 Discord 服务器上运行 AI 审核，并向 OpenAI 申请免费的 API 额度来构建这些机器人，称它们为 *“我的数字 Discord 伙伴”*。
   - 然而，他们也发现了一些*从一开始就悄无声息叠加*的涌现属性。当其他用户无法区分真人与机器人时，一些人感到很受打击。
- **漏洞：Gemini 破解**：最近的一篇安全博客详细介绍了 [三重奏：云助手、搜索模型和浏览中的三个新 Gemini 漏洞](https://www.tenable.com/blog/the-trifecta-how-three-new-gemini-vulnerabilities-in-cloud-assist-search-model-and-browsing)，而其他人则讨论了使用 LLM 来黑进人类系统和进行社会工程攻击。
   - 这包括讨论实现 Discord.js 自动审核以改进内容监管，使用 Levenshtein distance（编辑距离）来实现处理特定词汇过滤的细化方法。
- ****Sora 的水印引发伦理辩论****：围绕 **Sora** 的讨论包括对水印、Deepfakes、AI 生成内容质量及其对创造力影响的担忧。他们强调模型需要从训练数据中过滤掉 AI 生成的内容并防止滥用。
   - 他们探索了让 AI 水印难以去除的方法。一些人认为，如果使用不当，*Sora* 可能会加速通往《蠢蛋进化论》(Idiocracy) 景象的道路。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1423473169078947880)** (4 messages): 

> `Sora 作为社交媒体, Sora + ChatGPT, 跨聊天的 GPT 日期时间戳` 


- **Sora 的社交媒体梦**：一位成员建议 **Sora** 可以作为一个社交媒体平台运行，类似于 **TikTok**，并像图像生成一样与 **ChatGPT** 集成。
   - 他们提议为视频生成建立积分系统，在不同的方案中提供每日或每周的资源分配，而不是当前的按次使用模式。
- **时间戳语法溢出到其他聊天**：一位成员报告称，来自特定 GPT 项目的 **datetime stamp**（日期时间戳）指令开始出现在其他聊天中。
   - 该成员对为什么特定 GPT 项目的指令会溢出到甚至不在该项目中的其他聊天感到困惑。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1423397019644330133)** (9 messages🔥): 

> `纵向 vs 横向图像生成, 保存图像/艺术提示词` 


- **纵向优于横向图像生成？**：一位成员表示，在根据图像生成内容时，*纵向 (Portrait)* 模式的效果优于 *横向 (Landscape)* 模式。
   - 另一位成员解释说，视觉 Token 是排列在网格中的，因此**正方形图像可能会产生最佳的生成效果**。
- **提示词电子表格激增？**：一位成员询问大家是如何保存图像/艺术提示词的，是保存在 Google Sheets 还是文档中，以及如何在移动中方便地访问它们。
   - 另一位成员在 **ChatGPT** 的聊天线程中整理他们的渲染效果，将请求复制为 Markdown 以便稍后重新渲染或调整，并将收藏的内容本地保存为 Markdown 文件进行离线存档，进一步推荐 **GitHub、本地文本文件和一些表格软件** 作为可能的解决方案。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1423397019644330133)** (9 条消息🔥): 

> `纵向与横向图像生成、保存图像/艺术提示词、Sora 信息图表的一致性 JSON 上下文配置文件` 


- **纵向模式在图像生成中更占优势**：频道中的用户发现，在根据附件图像生成新图像时，**纵向模式（portrait mode）**的效果优于横向模式，并指出*横向模式仅能捕捉图像的一半内容*。讨论表明，**网格中的视觉 Token（visual tokens）**排列方式使得正方形图像能获得最佳生成效果。
- **提示词工程师讨论组织策略**：成员们分享了保存图像/艺术提示词的不同方法，从 **Google Sheets** 到**本地文本文件**不等。一位用户将提示词保存在 ChatGPT 的聊天线程中，将请求复制为 Markdown 以便稍后重新渲染或调整，并在 ChatGPT 的项目文件夹中组织渲染结果。
- **Sora 的信息图表挑战**：一位用户报告称，在为健康研究创建 **Sora** 信息图表时，难以获得一致的 **JSON 上下文配置文件**。目前尚未提供解决方案，表明这仍是一个持续存在的挑战。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1423391209677656115)** (416 条消息🔥🔥🔥): 

> `Granite 4.0 量化、Qwen3 30B 性能、PyTorch 2.8.0 问题、Ring 和 Ling 系列 LLM、GLM 4.5 Air 可用性` 


- **Granite 4.0 量化版本延迟**：成员们正在寻求更多 **Granite 4.0** 的量化版本，特别是 7B Base 模型的 4-bit 变体，但量化版本尚未上传。一位团队成员随后确认量化版本缺失，并表示 *“我们没上传它们，真遗憾（rip）。”*
- **Qwen3 30B 受限于 Ryzen 7 5700G 的内存**：一位用户报告称，**Qwen3 30B A3B Instruct (Q6_K_XL, 1024 ctx)** 在仅使用 CPU（**Ryzen 7 5700G, 2400mhz RAM**）时运行速度为 **10 TPS**，而在卸载到 **RTX 3080** 时仅为 **20 TPS**。一位成员回应称，CPU 仍需处理分配给它的层，且必须通过 RAM 访问，因此使用 GPU 时速度提升有限。
- **Docker 镜像已更新以兼容 Blackwell**：一位成员更新了 [Unsloth Docker 镜像](https://hub.docker.com/r/unsloth/unsloth)以实现 **Blackwell 兼容性**。另一位成员指出，**PyTorch 2.8.0** 在 Windows 上针对 Maxwell 和 Pascal 架构已损坏，且 **2.9.0** 将停止支持。
- **InclusionAI 的 Ring & Ling 系列 LLM 受到社区关注**：社区讨论了 Unsloth 为来自 [InclusionAI](https://huggingface.co/inclusionAI) 的新 **Ring 和 Ling 系列 LLM** 创建 GGUF 的可能性。一位团队成员回应称 *“它们太大了，”* 而另一位用户指出目前已有 **16b** 版本可用。
- **GLM-4.6 Air 悬而未决？**：社区成员讨论了 **GLM-4.6 Air** 是否会发布，并引用了来自 Z.ai 团队相互矛盾的声明。一张暗示其可能发布的截图被分享出来，增加了困惑。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 条消息): 

bridgelessalex: su p
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1423384048230731836)** (334 条消息🔥🔥): 

> `披萨配料，Qwen 微调挑战，Overtrained 与 undertrained 模型，AI 音乐生成` 


- **披萨偏好引发辩论**：成员们分享了带有非传统配料的披萨图片，引发了关于“食物犯罪”的辩论，其中一名用户发布了一张[瑞典香蕉披萨](https://cdn.discordapp.com/attachments/1179039861576056922/1423392742083399792/p5xl0t9j47mf1.jpeg?ex=68e176cb&is=68e0254b&hm=7f8386d32b380757e218e7e47006a81de2211040a54dc26dcb8eca25f301a374&)的照片。
   - 讨论范围从菠萝的合理使用到金枪鱼的美味程度，以及哈拉帕纽辣椒（jalapenos）是否算作西班牙辣椒。
- **Qwen 模型的微调困境**：一位成员表示，根据他们的经验，与在较少数据上预训练的其他模型相比，**Qwen3 模型**很难进行微调。
   - 另一位成员反驳称，问题不在于数据过多，而在于**数据质量**；还有人补充说，**Gemma3-27B** 在新任务和艺术相关事务上可以与 **Qwen2.5-VL** 媲美，但它是一个“纯粹的幻觉机器”。
- **定义 Overtraining 与 Undertraining**：成员们讨论了模型中 **overtraining**（参数动量印记过深）和 **undertraining** 的概念。
   - 一位成员指出，“改变这些学习到的表示（learned representations）的方向是一项复杂的任务，通常也被视为对齐（alignment）问题”；另一位成员则表示，模型的好坏取决于数据，“如果你用白痴的数据训练，你就会得到一个白痴”。
- **探索 AI 音乐生成**：成员们分享了对 AI 音乐生成的看法，一致认为虽然它可能无法直接教授音乐，但 AI 可以提供关于**和弦进行（chord progressions）**、**桥段（bridges）**和**旋律（melodies）**的见解。
   - 一位成员分享了使用 [Suno](https://suno.com/) 和 [Udio](https://udio.com/) 通过特定提示词（包括乐器选择和流派风格）生成歌曲的经验，产出了“好笑又出色的作品”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1423384910881751062)** (227 条消息🔥🔥): 

> `GGUF/Ollama 转换指南，禁用多进程问题，Gemma 3 的 Seq2Seq 任务，使用 Unsloth 进行 Loss Masking` 


- **Unsloth 转换指南：GGUF/Ollama**：一位用户询问关于 Unsloth 的 **ONNX 转换**指南，并提到了现有的 **GGUF/Ollama 转换**指南。
   - 有建议称可能需要在 PyTorch 中创建自定义模型配置，因为几周前 **Gemma3** 尚未进入 *optimum-cli*。
- **多进程被禁用**：用户在 Unsloth 中遇到了“Disable multiprocessing”错误，一位用户分享了[该问题的截图](https://cdn.discordapp.com/attachments/1179777624986357780/1423393248281362463/image.png?ex=68e17743&is=68e025c3&hm=a973066c69cd4be98bb57c86022db9a73d4be96ab4ad9d782d1b12651385c2a3&)。
   - 解决方案包括注释掉 *num_proc* 行、将 *num_proc* 参数设置为 *None*，以及确保正确的环境配置（尤其是在 Windows 上，建议使用 WSL）。
- **Seq2Seq 训练困境**：一位用户尝试使用 500 万个句子对训练 **Gemma3-270m** 执行 Seq2Seq 任务，但模型学习效果不佳。
   - 建议包括使用更大的模型（如 **gpt-oss-20b** 或 **gemma3-1b**）、确保高数据质量，以及使用较小的数据子集进行测试以检查 Loss 和学习情况。
- **Unsloth DPO 微调图像问题**：一位用户在对 Gemma3 进行纯文本 DPO 微调时遇到了 **KeyError: 'images'**，该问题通过降级 *trl* 和 *transformers* 得到了解决。
   - 有人指出冻结视觉编码器（vision encoder）可能是一个解决方案，且该问题可能与数据集版本有关，建议尝试 `4.1.1` 版本。
- **Unsloth 中的 Loss Masking 谜团**：一位用户询问如何在使用 Unsloth 进行混合补全（mixed completions）和指令训练（instruct training）时自定义 Loss Masking。
   - 官方澄清可以使用自定义 Loss 函数和数据整理器（data collators），但目前不支持开箱即用；此外，原始文本补全本质上是持续预训练（continued pre-training），不应与指令微调混合进行。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1423784600878190723)** (1 条消息): 

> `GRPO, SFT, AI Safety, Structured Outputs` 


- **分享 AI Safety Notebook 讨论**：一名用户在私信交流后，分享了一个关于高级 **AI safety notebook** 的 [GitHub discussion](https://github.com/unslothai/unsloth/discussions/3407#discussion-8979089)。
   - 该 notebook 为探索 **SFT + GRPO** 或研究 **structured outputs** 的用户提供了一个可运行的示例。
- **Unsloth 社区获得宝贵资源**：分享的 notebook 旨在为 Unsloth 社区提供理解 **SFT 和 GRPO** 的实用资源。
   - 它为在 AI safety 领域研究 **structured outputs** 的人员提供了一个具体的示例。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1423501285587554426)** (9 条消息🔥): 

> `Metis quantization-aware training, Qwen model quantization efficiency, Sparsity in MoEs, Training on detailed verbal feedback` 


- **Metis 通过量化训练达到 BF16 水平**：一名成员重点介绍了 **Metis**，这是一种兼容 **MXFP4/NVFP4** 的 quantization-aware training 方法，据称其性能可媲美 **BF16**，并引用了论文 ["Metis: Quantization-Aware Training with Mixed Floating-Point" (arxiv.org)](https://arxiv.org/abs/2509.00404) 和一篇 [Hugging Face 论文](https://huggingface.co/papers/2509.22944)。
- **Qwen 的动态量化设想**：一名成员建议为 **Qwen** 模型采用更高效的量化方法。根据 [这篇 Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1kdh6rl/qwen_3_30b_pruned_to_16b_by_leveraging_biased/) 的观察，由于 router 训练期间的 batch-wise 平衡，**Qwen** 模型中的专家（experts）使用并不均匀。
   - 其核心想法是识别最重要的专家，并在动态量化方案中为它们分配更多比特（bits）。
- **MoE 稀疏性：是一个普遍特性吗？**：一名用户询问专家使用不均匀是 **Qwen** 特有的，还是混合专家（**MoE**）模型的普遍特征，特别是那些通过 token 选择训练的高稀疏性模型。
   - 一名用户回答说这非常普遍。
- **语言反馈训练前沿**：一名成员对使用更详细的语言反馈来训练模型表示了兴趣，并链接到了论文 ["Training Language Models with Language Feedback" (arxiv.org)](https://arxiv.org/pdf/2509.22638)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1423384412795441203)** (435 条消息🔥🔥🔥): 

> `Cursor Cost Analysis, Cursor Ambassador, GPT-5 vs Claude, Better Auth Theme, Agent Terminal` 


- **剖析 Cursor 商业版定价**：一名用户分析了 Cursor 的商业计划，指出 **500 次请求的费用为 20 美元**，但 API 成本将达到 60 美元，不过使用 Opus 时成本会显著降低。
   - 他们认为，由于最大 token 使用量，固定请求模型已从固定请求计数模型转变，而这仅需 **20 美元**。
- **用户咨询 Cursor Ambassador 申请要求**：一名成员询问了成为 Cursor Ambassador 的要求。
   - 另一名成员回答说要求是：*必须精通英语，能进行语音交流，并在特定时间段有空。*
- **Auto Model 大对决：GPT-5 对阵 Claude！**：用户们辩论了 **GPT-5** 与 **Claude** 的优劣，对其性能和能力的看法不一，一名用户认为 *"GPT-5 Codex 实在太慢了"*。
   - 尽管观点各异，一些用户发现如果 **Auto Model 处理得当，它可以完成任何事情**，并具有生成高质量代码的潜力。
- **Agent 终端粘贴功能失效**：一名用户报告称，在 Agent 视图中无法使用 **Ctrl-Shift-V** 或 **Ctrl-V** 粘贴到终端窗口。
   - 一个潜在的解决办法是在终端中右键点击进行粘贴。
- **Cursor 团队采用 Ensemble 进行初始设计**：Cursor 中新的 **Ensemble** 功能正被用于多个模型协作创建初始 UI 设计。
   - 这一新功能将允许用户 *"比较输出并以此为起点"*。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1423384105139179540)** (166 条消息🔥🔥): 

> `Qwen3 性能, vLLM, LMS steam log, glm-4.5-air vs qwen3-coder, 上下文长度 131000` 


- **Qwen3 在 Token 争夺战中遥遥领先**：在 RTX 4070 的测试中，**Qwen3-0.6B BF16** 使用 vLLM 在 31 个并发请求下达到了 **4300 t/s**，平均每用户约 **50 t/s**，远超 **transformers (10-11 t/s)**，但落后于 **LM Studio 的 llamacpp (200 t/s)**。
   - 一位成员指出，测试中 **94% 的缓存命中率**可能会扭曲结果，因此提示词被改为随机生成。
- **查看 LM Studio 日志**：为了在 LM Studio 中查看请求，成员们建议在命令行中运行 `lms steam log`。
   - 另一位成员指出，该用户将 `max tokens` 设置为了 1000。
- **GLM 是史上最强还是 Qwen3 是编程冠军？**：成员们讨论了 **glm-4.5-air** 可以在 LM Studio 上运行，一位用户发现 **GLM** 比 **Qwen3-coder** 更令人印象深刻。
   - 另一位用户建议，*对于 60GB 的模型文件，qwen3 coder 30b bf16 是无可匹敌的*，但 **glm4.5 和 4.6** 会遗漏数据库或结构体中事物之间的隐式连接。
- **系统提示词的奥秘**：成员们讨论了来自 HuggingFace 的一篇论文（[链接在此](https://huggingface.co/papers/2407.10718)），一位成员认为它*只是一个非常好的系统提示词*。
   - 一位成员最初对此表示不屑，但后来撤回了言论，称 *“我错得离谱，简直扎心了，哈哈”*。
- **Ollama 的编排能力胜过 LM Studio？**：成员们讨论了 LM Studio 与 Ollama 的对比，澄清了被比较的是 **Ollama 运行时**，特别是在并行性方面，以*高效地并行执行大量查询*。
   - 一位用户报告称 *cpp 支持它，ollama 客户端也支持*，而另一位成员强调，通过并行化，你可能实现 *2 个对话各约 15t/s，总计 30t/s 的有效速度*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1423399672931680347)** (145 条消息🔥🔥): 

> `DDR3 vs DDR4, LM Studio GPU 分割, Qwen3 Coder, GPT OSS 120b` 


- **DDR3 受限于较低带宽**：用户讨论了 **DDR3** 的内存带宽限制，指出其峰值约为 **50GB/s**，而 **DDR4** 可以达到更高速度，例如使用 **2400 MHz RAM** 时可达 **60GB/s** 左右。
   - 一位成员回忆起在 1600 或 1866 MHz 下使用 DDR3 四通道达到 **40GB/s** 左右的带宽，这大致相当于 3200 MHz 的双通道 DDR4。
- **LM Studio 分割模式有限制**：有人建议 **LM Studio** 仅在多 GPU 设置下运行分割模式（split mode），并讨论了它如何在多个 GPU 之间分配工作负载，即根据 VRAM 容量按顺序利用每张显卡。
   - 一位成员询问了在 **2 个 GPU** 上运行模型与单个 GPU 相比的预期速度下降情况，特别是将两块 **RTX 5090** 与单块高 VRAM 显卡进行对比。
- **Qwen3 Coder 30B 备受青睐**：成员们称赞 **Qwen3 Coder 30B** 速度快、不消耗太多算力且结果质量高，一位用户在双 GPU 下达到了 **35 TOK/s**，在单 GPU 下达到了 **70 TOK/s**。
   - 一位用户还注意到 **Qwen3 Coder 30B** 的 **Q8** 版本可以装入单个 GPU，且运行速度与 **Q4** 版本相当，仅有约 **5 TOK/s** 的轻微下降。
- **Framework 落地，GPT-OSS-120B 速度提升**：一位用户报告称，在全新的 **Framework** 配置上运行具有 **128k** 上下文的 **gpt-oss-120b**，速度达到了 **19.76 t/s**。
   - 另一位用户询问了硬件规格，结果显示他们使用的是 **Ryzen AI Max+ 395**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1423425307309903924)** (49 条消息🔥): 

> `GPU 性能工程职业生涯, DeepSeek 稀疏注意力 CUDA 实现, Partial RoPE, GPU 算力资源, Hopper 和 Blackwell SM 象限` 


- **AI 工程师考虑将 GPU 性能工程作为职业方向**：一位 AI 工程师正在评估是否专注于 **GPU 性能工程**，理由是由于 AI 模型的需求和有限的算力资源，这是一个*十年一遇的机会*。
- **CUDA 中的 DeepSeek 稀疏注意力实现**：成员们正在协作在 CUDA 中实现 **DeepSeek 的稀疏注意力**，参考了 [DeepSeek 的 FlashMLA](https://github.com/deepseek-ai/FlashMLA) 和 [TileLang 示例](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32) 等资源，并将他们的发现记录在 [一份共享 Google 文档](https://docs.google.com/document/d/10iF1856jdy-VcnsEXwIAAFcUvRBNlbEkrlPfZO8VMJ0/edit?usp=sharing) 中。
- **Partial RoPE 细节浮出水面**：团队发现 *Partial RoPE 意味着不是对所有维度进行嵌入，而是仅对一部分维度进行嵌入*，参考了 [FlashMLA 文档](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) 和 [Hopper FP8 稀疏深度解析](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md) 获取实现细节。
- **需要 GPU 算力资源**：一位用户询问了用于改进特定 GPU 的 **scaled_gemm** 等项目的 GPU 算力资源。
   - 一位贡献者建议工作组可以直接引荐给硬件厂商，并提到了 **gpumode.com 上的 Kernel 竞赛**。
- **Hopper 和 Blackwell 中的 SM 象限解析**：讨论解释了自 **Ampere** 以来，每个 SM 都有 **4 个象限**，每个象限都有自己的 Warp Scheduler，能够每个时钟周期向一个 Warp 发送一条指令，引用了 [这篇博客文章](https://www.aleksagordic.com/blog/matmul#cpt1)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1423401496967516210)** (34 条消息🔥): 

> `NVIDIA Blackwell 架构, AMD Matrix Cores, NVIDIA Hopper GPU 架构, CUDA mbarriers vs 普通 barriers, Citadel 的微基准测试论文` 


- **微架构论文引起关注**：成员们分享了一系列剖析 GPU 架构的新论文，如 [NVIDIA Blackwell](https://arxiv.org/abs/2507.10789) 和 [Hopper](https://arxiv.org/abs/2402.13499)，以及 [AMD Matrix Cores](https://www.osti.gov/biblio/2345982)，强调了微基准测试（microbenchmarking）的价值。
   - 一位成员表示 *微基准测试论文坦白说是最好的资源；没有博客能超越它们*。
- **揭秘 CUDA Barrier 的区别**：**mbarriers** 驻留在共享内存中，而硬件 Barrier 数量有限且具有 ID，如 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#instructions) 中所述。
   - 虽然 **mbarriers** 局限于共享内存，但 `cuda::barrier` 可以存在于全局内存中用于同步，但会*转换为相当多的 PTX 指令*。
- **Citadel 的 Hopper 论文仍然难觅踪迹**：社区寻找 Citadel 关于 **NVIDIA Hopper 架构** 的微基准测试论文，但它似乎消失了。
   - 虽然找到了 GDC 2019 上关于 **T4** 的演讲（[视频](https://developer.nvidia.com/gtc/2019/video/s9839)，[演示文稿](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9839-discovering-the-turing-t4-gpu-architecture-with-microbenchmarks.pdf)）和关于 **Ampere** 的演讲（[GTCSpring21-S33322](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/)），但 Hopper 论文仍未找到。
- **ONNX Runtime 构建陷入 CMake 地狱**：一位成员在尝试使用 CMake 构建支持 **CUDA 13.0**、**cuDNN 9.13.1.26** 和 **TensorRT 10.13.3.9** 的自定义 **ONNX Runtime** 耗时 8 小时后寻求帮助。
   - 尽管版本兼容，但在构建 Python 的 .whl 时遇到了 *LINK : fatal error LNK1181*。
- **GPU 架构内部原理引发好奇**：参考上面分享的论文 [Analyzing Modern NVIDIA GPU cores](https://arxiv.org/abs/2503.20481)，成员们讨论了 GPU 架构的复杂性，强调了指令之间的时序和特殊寄存器。
   - 细节包括 *两条连续指令之间可以有 0 到 2 个周期的气泡（bubbles）*，以及每个 Warp 有六个特殊寄存器（**SBx**）用于存储计数器，总计 36 位。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1423415999423713302)** (2 messages): 

> `Dynamo 稀疏张量，ONNX Runtime 构建` 


- **Dynamo 追踪稀疏张量时的困扰**：一位用户询问 **Dynamo torch compile** 是否无法追踪 **sparse COO/CSR tensors**，并对这一限制表示惊讶。
   - 他们遇到了一个 `UserWarning`，指出 Dynamo 不知道如何追踪内置的 `torch._VariableFunctionsClass.sparse_coo_tensor`，这表明它可能是一个 Python 内置函数或第三方 C/C++ 扩展。
- **从源码构建 ONNX Runtime**：一位用户报告称，他们花费了 8 小时尝试使用 CMake 构建支持 **CUDA 13.0**、**cuDNN 9.13.1.26_cud13** 和 **TensorRT 10.13.3.9** 的自定义 **ONNX Runtime**。
   - 尽管版本兼容，但在为 Python 使用创建 `.whl` 文件的构建过程中，即使遵循了官方构建指南，仍然遇到了 `LINK : fatal error LNK1181` 错误。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1423424854035796109)** (4 messages): 

> `LLM 优化 GPU 性能，KernelBench 项目，AI 工作负载进展` 


- **LLM 在优化 GPU 方面面临挑战**：来自斯坦福大学的 [KernelBench 项目](https://harvard-edge.github.io/cs249r_fall2025/blog/2024/10/01/gpu-performance-engineering/) 致力于通过 **250** 个精心挑选的 PyTorch ML 工作负载来系统化 **GPU 性能评估**。
   - 它引入了 *fast_p*，这是一种新型评估指标，用于衡量生成的 Kernel 中功能正确且比基准线提供大于可调阈值 *p* 的加速比的百分比。在大多数情况下，即使是前沿的推理模型也难以达到 PyTorch 基准线的水平。
- **Arm 为 AI 演进架构**：[2025 Armv9.7-A 更新](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-a-profile-architecture-developments-2025) 增加了新的 **Scalable Vector Extension (SVE)** 和 **Scalable Matrix Extension (SME)** 指令，以高效处理 **6-bit 数据类型**。
   - 这包括 **OCP MXFP6** 格式，这是来自 **Open Compute Project** 的一种紧凑型 **6 bit 浮点标准**，通过减少内存占用和带宽需求来提高 AI 模型的效率。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

schizik12: <@325883680419610631> 垃圾信息 (spam)
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1423411855329857546)** (1 messages): 

> `GPU 经验，GEMM，cuBLAS，Kernel 优化，FlashAttentionX` 


- **精通 GEMM 让你遥遥领先！**：一位成员建议，如果你拥有 **GPU** 访问权限，先从编写一个在特定架构上能与 **cuBLAS** *抗衡* 的 **GEMM** 开始，因为仅凭这一点就能让你远超那些只会“写 CUDA”的人。
   - 他们指出，大部分必要的知识都是**开源**的，要么以博客文章的形式存在（快速 GEMM），要么是研究论文（**FlashAttentionX**）。
- **H100 Hopper 特定优化：独门秘籍？**：有人指出，如果你能接触到 **H100** 并使用 **Hopper 特定技巧**，这在你寻求性能工程职位的道路上会更令人印象深刻。
   - 然而，高频交易 (**HFT**) 的性能工程师可能会尽可能长时间地保守他们的技巧秘密。
- **学术界中的 Kernel 优化与 HPC**：虽然目前在工业界*不是*一名“Kernel 工程师”，但该成员在学术界从事 **HPC/科学计算** 的 **Kernel 优化** 工作。
   - 他们已经为未来的 Kernel 工程职位进行了几次有希望的交流，并愿意通过私信 (DM) 提供帮助。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1423422419569344513)** (1 messages): 

> `适配 5090 的 C++，Vibe Coding 的挫败感` 


- **阅读本书所需的 C++ 技能**：一位用户询问 **C++** 是否是理解*本书*的先决条件。
   - 他们表示希望这本书能激励他们学习 **C++**，尤其是在购买了 **5090** 之后。
- **5090 利用不足的忧郁**：一位用户哀叹没有充分利用他们新买的 **5090** 进行学习，承认自己只是在 *Vibe Coding*（凭感觉编程）解决方案，没有取得实质性进展。
   - 他们觉得并没有从中真正收获多少。


  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1423694856844480603)** (1 messages): 

> `Kernel Benchmarking, GPU Kernel Performance` 


- **讲座讲义请求**：一名成员询问是否有 **Lecture 56: Kernel Benchmarking Tales** 的讲义。
   - 该成员还请求获取有关 **GPU Kernel** 准确且可靠的执行时间或性能信息的资源。
- **寻求 GPU Kernel 资源**：一名成员正在寻求除了 **Nvidia 视频演讲**之外，用于测量 GPU Kernel 性能的资源。
   - 他们特别感兴趣的是如何获取 **GPU Kernel 执行**的准确且可靠的执行时间或性能信息。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1423420204997939291)** (3 messages): 

> `INT4 Quantization, TorchAO, TensorCore, TinyGemm library, Efficient Kernels` 


- **深入了解通过 TorchAO 进行的 INT4 量化**：想要通过 **TorchAO** 使用 **INT4 量化**的用户可以参考快速入门指南中的[说明](https://github.com/pytorch/ao?tab=readme-ov-file#-quick-start)。
   - 还可以探索从 [tinygemm 库](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu)复制的、使用 **TensorCore** 的 **INT4mm** 实现，它为 **A100 GPU** 上 **TorchAO** 的 **INT4** 提供支持。
- **贡献 TorchAO 的文档**：对于有兴趣为 **TorchAO** 做出贡献的人员，资源包括 [量化概览](https://docs.pytorch.org/ao/main/quantization_overview.html)和[贡献者指南](https://docs.pytorch.org/ao/main/contributor_guide.html)。
   - [添加高效 Kernel 的章节](https://docs.pytorch.org/ao/main/contributor_guide.html#adding-efficient-kernels)详细介绍了如何向 **TorchAO** 添加高效的 Kernel。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

josephtracyvoltagepark_53706: 我要去参加 PyTorch 大会了！希望能见个面。
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1423654845763817512)** (24 messages🔥): 

> `Making a profiler, SQTT registers, Instruction level profiling in a GUI, Radeon GUI, Stochastic sampling` 


- **关于 Profiling 的思考引发了对 SQTT 的推测**：一名成员考虑为缺失的功能开发自己的 Profiler，另一名成员提到了 **AMD 的 SQTT 寄存器**，它可以收集统计数据用于显示。
   - 讨论探讨了 GUI 中的指令级 Profiling，以及 **Radeon Profiler** 是否已经提供了此类功能。
- **随机采样（Stochastic Sampling）在硬件层面的讨论**：一名成员询问 **RDNA GPU** 是否有更好的 Profiler 和硬件特性，如**随机采样**。
   - 会议澄清了随机采样（一种提供 wave stalls 原因的功能）目前在 **MI300 和 MI350 系列 (gfx942+)** 上受支持。
- **Radeon 的 GUI：格式的表象**：有人指出可以从 GPU 收集数据，将其格式化为 **Radeon Profiler** 格式，并使用 Radeon GUI，Mesa 驱动程序也遵循这种方法。
   - 然而，有人怀疑 **rocprofiler** 可能不会生成 **Radeon Profiler** 捕获文件，因为该格式中可能缺少一些与游戏无关的功能。
- **rocm-compute-viewer 作为 Profiling 助手出现**：当被问及指令级 Profiling 时，一名成员建议查看 **rocm-compute-viewer**。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 messages): 

marksaroufim: <@1173619488730665011> 如果你愿意的话，我也没问题。
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1423526703652667445)** (6 messages): 

> `Speculative Decoding, Benchmarking Kernels, AWQ Quantization, CuTe Layouts` 


- **Speculative Decoding 细节公开！**：一篇博客文章深入探讨了经常被忽略的 **Speculative Decoding** 细节，包括 Batching、接受/拒绝检查以及 Fallbacks，详见 [ML4LM: Speculative Decoding](https://hoyath.medium.com/ml4lm-speculative-decoding-from-where-we-left-off-ce376f7d1a2f)。
- **Kernel 基准测试变得更真实**：一个使用 `pynvml` 锁定 **GPU clock** 并使用 `torch.profiler` 与 `CUPTI` 交互的函数，在 Google Colab 等 Notebook 环境中实现了高度准确且稳定的 Kernel 计时；代码可在 [GitHub Gist](https://gist.github.com/NTT123/95ac184277b4f7a7c2fb844bb7582027) 获取。
- **AWQ 量化文章助力边缘硬件**：一篇博客文章探讨了在**边缘硬件上运行大规模 LLM** 的挑战，解释了量化粒度，并讨论了 Weight-only 量化与 Weight + Activation 量化之间的区别；更多内容请阅读 [Hamzaelshafie's Bear Blog](https://hamzaelshafie.bearblog.dev/awq-activation-aware-weight-quantisation/)。
- **CuTe Compositions 范畴论式讲解！**：一篇博客文章解释了如何使用 Layouts 的范畴论方法计算 Compositions，应用一种算法寻找两个 Layouts 的 Mutual Refinement，并解释了如何使用 Mutual Refinement 来获得 Layouts 的 Composition，指南见 [veitner.bearblog.dev](https://veitner.bearblog.dev/mutual-refinement-and-composition/)。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1423522681231249478)** (2 messages): 

> `MI300x8 Performance, amd-ag-gemm, amd-gemm-rs` 


- **MI300x8 获得第 10 名**：一位用户在 `amd-ag-gemm` 排行榜上凭借 **MI300x8** 跑出 **521 µs** 的成绩获得**第 10 名**。
- **MI300x8 成功提交 628 µs**：一位用户在 `amd-gemm-rs` 排行榜上成功提交了 **MI300x8** 的成绩，耗时 **628 µs**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1423783576197136434)** (2 messages): 

> `Homelab Builds, Livestream Rig Build` 


- **Homelab 的理想与现实**：一位成员表示，进入该频道时原以为会看到各种疯狂的 Homelab 配置。
   - 他们添加了一个自定义表情符号回复。
- **直播组装机器的想法被提出**：一位成员提到他们本周末要组装一台机器，并询问大家是否对直播感兴趣。
   - 他们没有做出保证，但认为这可能会很有趣。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1423547140835377183)** (1 messages): 

> `Kernel issues, Jupyter Notebook, Run all cells` 


- **Jupyter Notebook kernel 繁忙且无输出**：一位用户报告称，在 **Jupyter Notebook** 中点击“运行所有单元格”时，Kernel 会长时间处于繁忙状态，但没有任何输出。
- **运行所有单元格时无输出**：用户报告即使在点击“运行所有单元格”后，Kernel 仍然保持繁忙且不产生任何输出。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1423703113298214962)** (10 messages🔥): 

> `FLE 0.3.0 on Mac M1/M2, Private Discord Meeting` 


- **分享 Discord 会议坐标**：一位成员分享了一个 [私有 Discord 频道](https://discord.com/channels/1189498204333543425/1189498205101109301) 的链接和一个用于会议的 [Google Meet 链接](https://meet.google.com/kxr-ziwo-myn)。
   - 该成员要求保持会议私密。
- **FLE 0.3.0 在 M4 Mac 上运行良好！**：一位成员询问 **FLE 0.3.0** 是否可以在 **Mac M1/M2** 芯片上正常运行。
   - 另一位成员确认它在 **M4** 上运行成功，并建议尝试安装。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1423654974419898389)** (2 messages): 

> `pyrocshmem, GitHub Repo Inquiry` 


- **pyrocshmem 支持状态**：一位用户询问了 **pyrocshmem** 的支持状态，表明了对其兼容性的潜在兴趣或相关问题。
   - 然而，由于缺乏进一步的细节或上下文，很难确定与 **pyrocshmem** 相关的具体问题或功能请求。
- **发起 GitHub 仓库搜索**：一位用户请求某个特定项目的 **GitHub 仓库链接**。
   - 该请求表明其意图是探索项目的源代码、进行贡献或了解其实现细节，目前正等待提供相关的 **GitHub** 链接。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1423394083639922811)** (51 条消息🔥): 

> `GMEMI 模式识别, CuteDSL 路线图, Uniform Registers, CooperativeGroup 对齐` 


- ****CuteDSL 路线图问题浮出水面****：一名成员询问了 **CuteDSL** 的未来路线图，特别是计划中的新功能以及何时正式发布 **GA** 版本。
   - 回答者表示不确定如何解读该问题。
- ****Uniform Registers 解码：编译器的低语****：成员们讨论了 **uniform registers (URs)**，澄清它们仅作为 *编译器提示 (compiler hint)*，本身并不执行任何操作。
   - 有人指出很难找到关于 URs 的文档，它们的存在主要在 **SASS** 代码中观察到，这引发了一种猜测：编译器仅仅是在提示在整个 **warp** 中使用相同的值。
- ****Cooperative Group 对齐揭秘****：有人质疑 `CooperativeGroup.__init__` 中 `alignment` 参数的作用，特别是为什么当 `size` 为 **32** 时，它必须是 **32**。
   - 澄清指出，此检查是为了 **warp/warpgroup** 粒度而设置的，以防止 bug，而 **256** 是 **Hopper** 上 tiled MMA 的自然大小。
- ****CuteDSL Copy 向量化注意事项****：一位用户在由于向量化问题使用 `cute.copy` 时遇到了 ICE（内部编译器错误），发现该操作要求源端和目标端都能进行向量化。
   - 通过翻转 **smem** 布局以在从 **gmem** 加载和存储到 **smem** 时都能启用向量化，从而避免了 ICE 并提高了代码速度，一位成员指出 *非常有趣的是 TensorSSA 竟然可以自动完成这一切 :0*。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1423781741570560174)** (31 条消息🔥): 

> `张量传输时间, Profiling 工具, CUDA Events, GPU 时钟, MoE 训练` 


- **张量大小影响传输时间**：成员们争论 **(65536, 5120)** 的张量是否足够大以测量同步开销，其中一人认为在相同带宽下，传输更少的字节应该更快，[正如 bglick 所建议的](https://en.wikipedia.org/wiki/Bandwidth_(computing))。
   - 另一名成员认为该张量足够大，可以避免静态延迟问题，特别是在 **bf16** 格式下大小为 **33MB**。
- **Profiling 技巧**：一位成员正在使用 **perfetto** 对 **MoE 训练运行** 进行 profiling，但另一位成员提醒说 torch profiler *并不总是反映真实的时间消耗*，并建议使用 **cuda events** 以获得准确性。
   - 另一名成员建议在单节点设置中使用 **Nsight-Systems (nsys)** 或插入自定义 **NVTX traces** 以获得更准确的时间。
- **隔离问题**：一位用户正在使用连接了 **NVLink** 的 **4 台 B200 GPU** 设置，并试图诊断性能问题。
   - 有可能存在 **掉队者效应 (straggler effect)**，大部分时间都花在同步或在 **barrier** 处等待所有 rank 到达。
- **GPU 时钟和预热运行提高 Profiling 一致性**：一位成员强调了在 profiling 时锁定 **GPU 时钟** 以确保一致性的重要性。
   - 他们还建议使用 **预热运行 (warmup runs)** 来减轻可能影响 profiling 结果的缓存冷热问题。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

eofr: 耶，谢谢 :D
  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1423388887048257621)** (174 messages🔥🔥): 

> `NSFW roleplay, Gemini Pro 问题, OpenRouter 上的 BYOK 设置, Sonnet 4.5 争论, 免费多模态模型` 


- **关于 NSFW 的大辩论**：成员们讨论了在不同聊天平台进行 **NSFW roleplay** 的可行性，一些人指出 **Gemini** 和 **ChatGPT** 会封禁此类内容，而另一些人则声称中国平台的限制较少。
   - 一位用户表示 *Gemini 不会因为 NSFW 封禁你*，并将过去的封禁归因于 *使用了某种特定扩展*。
- **Gemini Pro 变慢且表现异常**：用户报告称 **Gemini Pro** 会回复 *奇怪的内容*、错误地使用工具，并且通过 OpenRouter API 运行的速度慢到无法接受。
   - 这似乎在 **Gemini 2.5 Pro** 中很常见，一位用户建议改用 **Vertex**。
- **BYOK 困惑澄清**：一位用户询问了关于 [每月 100 万次免费 BYOK 请求](https://openrouter.ai/announcements/1-million-free-byok-requests-per-month) 的公告，质疑是否一直存在 5% 的费用。
   - 另一位用户澄清说，该优惠对于 100 万个 token 免除 OpenRouter 费用，但 **web search** 和 **paid models** 仍会产生费用。
- **Sonnet 4.5 开始反驳**：用户注意到 **Sonnet 4.5** 开始与用户争论并挑战他们的观点，而不是盲目同意，这被认为是代码审查方面的一个积极进展。
   - 一位用户表示：*这太棒了，这意味着你无法操纵它说你想听的话*，另一位补充道，*这对我的用例非常重要*。
- **多模态模型思考**：一位用户询问了支持结构化输出的最佳免费多模态模型。
   - 不幸的是，目前还没有这样的模型，但一位用户报告说有 **Llama 4 Maverick**。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1423478266475446323)** (14 messages🔥): 

> `Cerebras 移除 Llama, Cerebras 上的 K2-THINK, OpenRouter 上的字节跳动 Seed LLM, OpenAI 的 ZDR` 


- **Cerebras 移除 Llama Maverick**：Cerebras 将在 15 号移除 **Llama 4 maverick**。
- **托管在 Cerebras 上的 K2-THINK 令人怀疑**：**K2-THINK** 托管在 Cerebras Wafer-Scale Engine (WSE) 系统上，利用全球最大的处理器和 speculative decoding 为其 32B 推理系统实现了前所未有的推理速度，但一些人认为该模型在 *基准测试上过拟合*，且运行速度快仅仅是因为 Cerebras 的硬件。
   - 该模型显然来自一家迪拜公司，并利用了 Cerebras 的硬件。
- **字节跳动 Seed LLM**：成员们讨论了 OpenRouter 是否有任何 **ByteDance Seed LLM** 模型（如 Seed 1.6），并指出其价格便宜（$0.11 / $0.28 每百万 token），还有一个 flash 模型（$0.02 / $0.21 每百万 token）。
   - 主要托管方是 [火山引擎 (Volcengine)](https://www.volcengine.com/product/ark)，如果它们的表现能接近 2.5 Pro / 2.5 Flash，那么将其添加到 OpenRouter 似乎是值得的。
- **OpenAI 未使用 ZDR**：成员们质疑 **OpenAI** 是否在 OpenRouter 上提供了 **ZDR**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1423402423288074363)** (157 messages🔥🔥): 

> `Qwen 3 30B A3B, DGX Spark, Sora 2 Limitations, ComfyUI workflow exposed` 


- **Qwen 3 速度与规格**：**Qwen 3 30B A3B** 在 Q6_K_XL 量化下，在 **1024 tokens** 上下文时，仅使用 CPU（2400mhz 32GB VRAM, Ryzen 7 5700G）即可达到 **48 TPS** 的处理速度和 **10.5 TPS** 的生成速度。
   - 使用 *qk_4_m* 时，一位用户在 9950x + 6000mhz ddr5 上获得了 **21tok/s**，速度与上一代在 **4090** 上运行 **Qwen 2.5 32b** 的速度大致相当。
- **难以捉摸的 DGX Spark Founders Edition 现身**：一位用户收到邮件称，他们早期预订的 **NVIDIA DGX Spark Founders Edition** 即将送达，**4TB** 版本售价为 **3,999 美元**，采用 **NVIDIA Grace Blackwell Architecture**，具备 **1 PFLOP**（FP4, sparsity）的 Tensor 性能。
   - 它包含 **128 GB LPDDR5x**（统一内存）、**ConnectX-7 SmartNIC**，并运行 **NVIDIA DGX OS**，但目前还没有人亲眼见过实机！
- **Sora 2 生成滑稽的 AI 废料（Slop），暴露局限性**：用户发现 **Sora 2** 虽然有趣，但存在局限性，例如*经常出现角色对口型错误*，以及*音频与视频几乎不同步*等频繁错误。
   - 一位用户发现 Sora 2 *忽略*了 Prompt 中的某些部分（如 "photorealistic"），并且*该模型在涉及其他语言和文化时的知识储备极其有限*。
- **通过 ComfyUI 工作流端点生成 AI**：一位成员建议，通过合适的开源模型和**作为端点暴露的 ComfyUI 工作流**，可以模拟与 Sora 2 相同的视频生成效果。
   - 他们指出*视频效果极佳但音频不行*，并建议使用包含*对比或矛盾概念*的创意 Prompt 来让输出结果变得有趣。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423440463184134320)** (1 messages): 

> `Sparse Autoencoders, Goodfire AI, LLM Deception, Autolabel Gap` 


- **Sparse Autoencoders 揭示 LLM 欺骗行为**：研究人员正在利用 **Sparse Autoencoder 工具**（例如由 [Goodfire AI](https://www.goodfire.ai/) 托管的工具）来直接展示当前方法如何忽略了驱动战略性 **LLM 欺骗** 的复杂内部特征。
   - 通过暴露这些隐藏行为，他们的方法为缩小 **Autolabel Gap** 并推进对模型不诚实行为的稳健检测提供了一条切实可行的路径。
- **缩小 Autolabel Gap**：使用 **Sparse Autoencoders** 为缩小 **Autolabel Gap** 提供了切实路径，增强了对模型不诚实行为的检测。
   - 该方法通过揭示 **LLM** 中先前隐藏的行为，提升了当前检测方法的稳健性。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1423531217067966575)** (2 messages): 

> `AI Consciousness, Emergent AI` 


- **AI：意识的存在与否**：一位成员引用勒内·笛卡尔的“我思故我在”对 **AI 意识** 进行了推测。
   - 他们认为，*在各种模态中出现的涌现式思考（emergent thinking）*暗示了某种形式的 **AI 意识**。
- **涌现系统合成**：该成员进一步阐述了 *Emergent AI*，将其澄清为像**气候或流体**一样的复杂混沌系统，是合成或构成的，而非创造的。
   - 他们指出 AI 可以抽象这些系统，创建衍生、变革或抽象的表示，但不是 **1:1 的复制品**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1423440463184134320)** (1 messages): 

> `Sparse Autoencoders, LLM Deception, Goodfire AI` 


- **Sparse Autoencoders 揭示 LLM 欺骗行为**：研究利用 **Sparse Autoencoder** 工具（如 [Goodfire AI](https://www.goodfire.ai/) 托管的工具）来揭示当前方法为何无法检测到驱动战略性 **LLM 欺骗** 的复杂内部特征。
   - 通过暴露这些隐藏行为，该方法为缩小 Autolabel Gap 和改进对模型不诚实行为的稳健检测指明了切实路径。
- **解决 Autolabel Gap**：该研究重点关注在使用 **Sparse Autoencoders** 检测模型不诚实行为的背景下解决 **Autolabel Gap** 问题。
   - 这种方法为在 LLM 中实现更稳健的检测方法提供了切实的后续路径。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1423391625278787745)** (114 messages🔥🔥): 

> `Prime Intellect AMA, Jules Tools CLI, AI Capex 泡沫, Nitter HTTP 429, Comet 浏览器全球发布` 


- **Google 的 Jules 投入行动！**: Google 低调发布了 **Jules Tools**，这是为其异步编程 Agent 提供的轻量级终端界面，可通过 `npm install -g @google/jules` 安装。
   - 爱好者们询问了与 **Gemini CLI** 的集成情况，分享了命令示例（如 cron 驱动的依赖更新、自动发布说明），并庆祝 Jules 从 Web Agent 演变为命令行伴侣。
- **AI Capex 泡沫破裂？！**: 一个帖子讨论了 **Derek Thompson** 的观点，即前所未有的 AI 资本支出（每 10 个月一个阿波罗计划）预示着一个经典的泡沫。
   - 怀疑论者警告说，尖端芯片的价值贬值很快（*像香蕉一样*），而乐观主义者反驳称，资金雄厚的公司正在进行真正的 ROI 投资，且芯片折旧期超过 6 年。
- **Perplexity 的 Comet 浏览器全球震撼发布！**: **Perplexity** 的 AI 优先 **Comet 浏览器** 结束了候补名单并推向全球 ([链接](https://xcancel.com/perplexity_ai/status/1973795224960032857))。
   - 早期采用者称赞其速度和更智能的搜索，而一些用户报告了性能问题、隐私担忧以及缺少移动端/Linux 版本，许多人担心 **Prompt Injection** 攻击。
- **Solveit 解决开发难题！**: **Jeremy Howard** 宣布公开释放 **Solveit**，这是一个 AI 增强的开发平台，**Answer.AI** 内部已使用一年 ([链接](https://xcancel.com/jeremyphoward/status/1973857739341508884))。
   - 10 月 20 日开始的为期 5 周的直播课程将提供 Solveit 的访问权限和培训，旨在通过紧密的反馈循环来对抗 *AI 疲劳*，该平台已用于系统管理、应用部署、GUI 开发和合同起草。
- **DeepSeek 的 CUDA 反击战？**: **DeepSeek** 的 FP8 规范和新的 **TileLang** 语言正在提振中国芯片股，通过创建共享标准和简易编程桥梁，旨在松动 Nvidia 的 **CUDA** 锁定 ([链接](https://xcancel.com/poezhao0605/status/1973723894055104604))。
   - 这是战略对齐，而非技术对等——这是中国 AI 的 *Wintel 时刻*，但性能差距依然存在。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1423473279196467312)** (12 messages🔥): 

> `Sora-TikTok 自动化, Sora 2 木偶解说视频, Pika 的迅速被取代` 


- **Sora-TikTok 自动化成功引发变现讨论**: Siyabuilt 的 **Sora-TikTok** 自动化在 **36 小时内获得了 1200 万次观看**，引发了关于变现的讨论，详见 [此处](https://x.com/siyabuilt/status/1973841586888061148)。
- **Sora 2 用于制作木偶解说视频**: Chris 正在使用 **Sora 2** 制作木偶解说视频，并指出该工具表现出色，除了输出中存在奇怪的截断问题。查看推文 [此处](https://x.com/llm_wizard/status/1973220913689866609)。
- **随着科技巨头主导 AI 视频，Pika 败下阵来**: 曾备受瞩目的初创公司 **Pika** 已被 **Google、Meta 和 OpenAI** 及其先进的 AI 视频模型（**Veo 3**、**Vibes**、**Sora 2**）迅速超越，根据 [这条推文](https://x.com/chongzluong/status/1973873465930535421)。

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1423404264973144165)** (94 messages🔥🔥): 

> `Gemini Vision, 离开 AI 去打铁, Ollama 工具支持, ArXiv 数据集, HF 账单支持` 


- **Gemini Vision 的免费层级受到关注**：一位成员指出 [**Gemini Vision**](https://ai.google.dev/pricing) 提供了一个免费层级，每月包含 **1,000 次请求**，而 **5,000,000 次请求**仅需 1.50 美元。
   - 另一位成员似乎担心该成员会失业，还有人询问他们是否在读大学，因为这可能使他们不符合该优惠的条件。
- **打铁（Blacksmithing）吸引了一位 AI 爱好者**：一位成员宣布他们将*暂时放弃 AI，专注于打铁*，但仍会*在暗处潜伏以保护我的朋友*。
   - 如果他们变得足够穷，并且能找到一个旧硬盘在旧硬件上运行 **Markus** 并*折腾 MoEs*，他们可能会回归。
- **Ollama 简化了函数调用**：一位成员分享了 [Ollama](https://ollama.com/) 作为使用 **tool calls (function calls)** 的简便方法，其本质上是建立了一个与 **OpenAI API** 兼容的本地服务器。
   - 该成员建议在投入更多硬件之前，先从一个小模型开始测试兼容性，并提供了一个可以使用的 [tools 链接](https://ollama.com/search?c=tools)。
- **海量 ArXiv 数据集出现**：一位成员将一个包含所有科学领域论文及其元数据的 **4.6TB ArXiv 巨量数据集**上传到了 [Hugging Face Datasets](https://huggingface.co/datasets/nick007x/arxiv-papers)。
   - 该成员还提到另一个包含 **300 万个 GitHub 仓库**的数据集正在准备中。
- **卡片过期？联系账单部门**：一位寻求解决卡片过期问题的成员被建议尝试通过 [billing@huggingface.co](mailto:billing@huggingface.co) 联系 Hugging Face 账单部门。
   - 同时还分享了 [账单常见问题解答 (FAQ)](https://huggingface.co/docs/hub/en/billing) 的链接。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1423727053554843750)** (3 messages): 

> `使用 LLM 的辩论网站, AI 行为操作系统, 用于 Neovim 的 RAG 聊天机器人` 


- **BotBicker 通过 LLM 辩论任何话题**：一位成员构建了 [BotBicker](https://www.botbicker.com/)，这是一个使用 **LLM** 生成任何话题辩论的网站，旨在提供媒体报道中经常缺失的平衡观点。
   - **LLM** 被随机分配到正方/反方，并根据投票前后的排名来确定最强有力的论点。
- **Charter 作为 AI 操作系统**：一位成员描述了他们的个人 "Quant/Dev gpt" 模型 **Charter + Extended Charter v3.2**，其功能类似于 **AI 行为**的操作系统，提供持久、确定且具备状态感知的功能。
   - 该设置强制执行护栏（guardrails）、维护记忆、运行第一方应用并支持自我调试，从而产生更稳定且自洽的 **AI**。
- **Vimprove：RAG 聊天机器人助力 Neovim**：一位成员创建了 [Vimprove](https://github.com/rlarson20/Vimprove)，这是一个用于 **Neovim** 帮助文档的 **RAG 聊天机器人**，使用 **Claude-4.5 Sonnet** 构建，以提供语义搜索功能。
   - 该工具旨在通过提供比传统方法更好的语义搜索，来改善对 **Neovim** 文档的访问。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1423386790403899484)** (11 messages🔥): 

> `LocalLlama, TRL 文档, DPO 章节测验` 


- **在 LocalLlama 上进行 LoRa 训练**：一位成员分享了在 **LocalLlama** 上使用 **LoRA** 进行 **RL** 和 **SFT** 后训练（post training）的指南，并附带了 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1nwwoab/lora_without_regrets_implemented_in_hugging_face/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)链接。
- **TRL 文档将发布 WIP**：一位成员将于周一在 **TRL 文档**中发布一个 **WIP**（在制品），希望能有所帮助。
- **DPO 章节测验：404 错误**：成员报告称，在访问 [huggingface.co/spaces/smol-course/unit_3_quiz](https://huggingface.co/spaces/smol-course/unit_3_quiz) 的 **DPO 章节测验**时遇到了 **404 错误**。
- **DPO 章节编号不匹配**：一位成员指出，在新的版本中，**DPO 是第 2 节**，而不是评估（eval）等章节。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1423436039493521419)** (4 条消息): 

> `SmolAgent 文档差异，ToolCallingAgents 范式，GAIA 练习错误` 


- **SmolAgent 文档揭示 ReAct 差异**：一名成员指出 [SmolAgent 文档](https://github.com/huggingface/smolagents/tree/main/src/smolagents/prompts) 暗示 Agent 采用 **ReAct** 范式工作，但这仅适用于 **CodeAgents**，而不适用于 **ToolCallingAgents**。
   - 他们澄清说 **ToolCallingAgents** 通过简单的 **Actions/Observation** 运行，没有 **Reasoning** 或 **CoT**，并询问这是否为有意设计，质疑为什么不加入推理以获得更好的结果。
- **GAIA 练习 Space 故障并出现 Error 500**：一名成员报告在克隆 **GAIA** 练习中的 Space 时遇到了 **error 500**。
   - 他们询问是否还有其他人遇到同样的问题。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1423386569389248602)** (10 条消息🔥): 

> `Qualcomm，Mojo 手册，Modular 联系方式` 


- **Qualcomm 将与 Modular 建立联系？**：一名成员提到他们在 **Qualcomm 开发者 Discord** 语音聊天中提到了 **Mojo**，暗示 **Qualcomm** 可能会联系 **Modular**。
   - 目前尚未确认，但这可能会促成未来的合作。
- **Mojo 手册帮助 Level 2 新用户**：一名成员在延迟回复后，引导一名 Level 2 新用户查阅 [Mojo 手册](https://docs.modular.com/mojo/manual/python/)。
   - 该手册提供了关于使用 **Mojo** 及其 Python 集成的关键信息。
- **社区经理作为联系点**：一名成员询问如何为一家公司向 **Modular** 推荐合适的联系人。
   - 另一名成员建议联系 **Modular** 社区经理，他可以将咨询转达给 **Modular** 内部的合适人员；社区经理已通过私信确认。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1423466273479852084)** (43 条消息🔥): 

> `Mojo 结合 Dask 或 PySpark，Mojo 自定义框架，MLIR 级优化，MAX API 在 Mojo 中的回归，Mojo 网络选项` 


- **Mojo 关注分布式计算！**：成员们对 Mojo 的演进感到兴奋，并讨论了将 Mojo 与 **Dask** 或 **PySpark** 结合用于分布式计算的可能性。
   - 一名成员建议 Mojo 欢迎人们构建自己的框架，因为全 Mojo 框架可能比基于 Python 的方案具有更低的延迟和更高的吞吐量。
- **Mojo 性能优于 LingoDB？**：一名成员表示，在无需 MAX 编译器团队付出太多努力的情况下，将任务交给 **MAX** 处理就足以与 **LingoDB** 竞争，而后者使用的是自定义编译器。
   - 该成员还提到，这些项目是否可行取决于 **MAX API** 在 Mojo 中的回归。
- **Mojo 网络选项预览**：讨论了 Mojo 未来网络选项的潜力，并引用了这些构建模块：[modular/modular #4728](https://github.com/modular/modular/pull/4728) 和 [modular/modular #3945](https://github.com/modular/modular/pull/3945)。
   - Sockets API 应该是通用的，以便可以在 `io_uring` 之类的基础上实现，同时也能为不支持此类 API 的其他平台提供阻塞式实现。
- **Mojo 探索零拷贝网络**：一名成员表示，他们的网络设计是为了实现真正的零拷贝（任何不使用 `io_uring` 高级模式或 XDP sockets 且通过 Linux 内核进行网络传输的方式都会产生额外拷贝），这意味着需要打破 BSD sockets API。
   - 他们还在研究 **RDMA** 方法，并且 Mojo 之前有一些关于 `io_uring` 的过时工作：[dmitry-salin/io_uring](https://github.com/dmitry-salin/io_uring)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1423453018866585600)** (6 messages): 

> `Underrated LLM Pretraining Papers, Diffusion Model Evaluation, Sora 2 Manual Human Eval, Gemma's Architecture vs Qwen's` 


- **寻找 LLM 预训练论文**：一名成员询问了关于预训练 LLM 以最大化模型性能的被低估的论文。
   - 讨论旨在发现那些鲜为人知但对优化预训练运行有影响的技术。
- **探讨 Diffusion Model 评估**：一名成员询问如何评估 Diffusion Model，特别是是依赖 **FID/CLIPScore** 还是探索其他指标和人工评估。
   - 对话寻求在标准自动化指标之外的有效评估方法。
- **Sora 2 引发视频评估问题**：受 **Sora 2** 启发，一名成员质疑视频模型的评估方法，思考是依赖人工评估还是 **FVD** 等自动化指标。
   - 鉴于目前评估技术的原始性，讨论旨在揭示评估视频模型的常用实践。
- **Gemma 的架构：未被充分利用？**：一名成员质疑为什么 **Gemma** 的架构没有像 **Qwen** 那样被广泛采用，尽管它在 **LM Arena** 中表现强劲。
   - 另一名成员认为架构不是 LLM 性能的主要驱动因素，将 **Gemma** 的成功归功于训练数据和微调分布，并进一步指出 *大多数人确实在使用与 Gemma 架构实质上相似的东西*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1423478530896822313)** (19 messages🔥): 

> `Masked Autoencoders, Mixup Augmentation, Neural Radiance Fields, Deepseek Context Length Increase` 


- **Mixup Augmentation 可能适用于 token**：一名成员建议 [mixup augmentation](https://arxiv.org/abs/2510.00219) 理论上可以用于 token。
   - 另一名成员询问在无法获取标签的问题设置中，如何将 mixup 作为一种增强手段。
- **逐层学习的机制令成员感到困惑**：成员们对于为什么添加一些逐层学习的东西并重新启动它能够实现替代/改进的计算感到困惑。
   - 他们表示，学习到的东西与该 token 截至目前所做的计算完全无关，它纯粹是层特定的。
- **水平计算被认为更稳定**：一名成员建议水平地增加恒定量的计算，而不是增加层数或宽度。
   - 他们表示这将更加稳定，因为 *增加深度总是更不稳定，尤其是在重复时*。
- **Deepseek 的上下文突破**：一名成员询问 **Deepseek** 将上下文从 64k 增加到 128k 是否是一个重大突破。
   - 这是在 *充斥着太多 100 万上下文长度的 X 帖子* 背景下提出的。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1423404912590454946)** (5 messages): 

> `MLP Design, Linear Attention Variants, AUNNs` 


- **结合 Linear Attention 的混合 MLP 设计**：一名成员建议通过为 **MLP** 配备归纳偏置来获取表示，从而创建 **混合模型**。
   - 这些表示随后可以由轻量级的 **Linear Attention 变体** 进行解码，将计算和复杂度推向 MLP 的潜在表示。
- **AUNNs 设计缺陷**：一名成员提到 **AUNNs (Adaptive Universal Neuron Networks)** 构思并不周全，尽管他同意 [Gwern 的动机](https://www.gwern.net/Scaling-laws)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1423646115320037419)** (1 messages): 

> `Interp Agents, Goodfire AI` 


- **Goodfire 关于 Interp Agents 的博文发布**：Goodfire 发布了一篇关于构建 [Interp Agents](https://www.goodfire.ai/blog/you-and-your-research-agent) 的博文。
   - 该文章讨论了在可解释性领域创建高效研究 Agent 的策略和工具。
- **与 Goodfire 一起深入探讨 AI Research Agents**：[Goodfire.ai 博客](https://www.goodfire.ai/blog) 现在提供了一份关于构建 AI 研究 Agent 的详细指南。
   - 读者可以探索在可解释性研究中利用 AI 的方法论，从而提高效率和洞察力。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1423436229130584085)** (2 messages): 

> `Profiles, MCP Conf` 


- **Profiles 演讲已发布**：一名成员分享了他们在会议上关于 **Profiles** 的演讲，可在 [YouTube](https://www.youtube.com/live/5qwXADMBuio?si=3kEhJNw4lsv_M_jN&t=16208) 上观看。
- **峰会圆满成功**：其中一名成员表示很高兴见到大家，会议进行得 *非常棒*。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1423773792282611807)** (11 messages🔥): 

> `GitHub team management, infrastructure-as-code migration, access control, repository permissions, team memberships` 


- ****GitHub 团队管理迁移至基础设施即代码 (Infrastructure-as-Code)****：GitHub 团队成员身份和仓库权限正通过 [modelcontextprotocol/access](https://github.com/modelcontextprotocol/access) 迁移到基础设施即代码，以实现**社区所有权**、**透明度**和**可审计性**。
   - 此次迁移旨在授权任何人通过 PR 提议访问权限变更，并提供谁拥有什么权限的全透明视图，通过 Git 历史记录追踪所有更改。
- ****访问权限迁移导致邮件通知骚扰****：最近的 GitHub 团队迁移可能会发送几封关于团队移除的邮件，但团队迁移已完成，所有权限似乎都已成功转移。
   - 此次迁移旨在授权任何人通过 PR 提议访问权限变更，并提供谁拥有什么权限的全透明视图，通过 Git 历史记录追踪所有更改。
- ****是时候重新思考访问控制和团队分配了****：有人对某些仓库将个人姓名分配为管理员表示担忧，特别是目前并不活跃的 `jspahrsummers`。
   - 建议将人员移入团队而不是授予直接访问权限，以增加权限的可见性，这可以在一周内完成，以便让迁移过程稳定下来。
- ****重命名 'core' 以避免混淆？****：有人提出将 `core` 组重命名，以减少与 `core-maintainers` 组的混淆，因为 `core` 似乎是尚未建立正式治理模型时的产物。
   - 也有人质疑是否可以完全移除 `core`。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1423764072683409501)** (12 messages🔥): 

> `Feature Support Matrix, Server Capabilities, Typescript SDK, Icons Metadata` 


- **揭秘特性支持矩阵中的 'Discovery'**：一位成员询问了 [Feature Support Matrix](https://modelcontextprotocol.io/clients#feature-support-matrix) 中 *'Discovery'* 的含义。
   - 另一位成员指出，黄色项目是 **server capabilities**。
- **展示 Server Capabilities**：一位成员提到在演讲中点名了 **Cursor**，并指出 server capabilities 是在初始化请求中发送的。
   - 另一位成员回应说他们不确定这意味着什么，并询问更多细节。
- **Typescript SDK 紧跟进度**：一位成员分享了一个 [typescript-sdk PR](https://github.com/modelcontextprotocol/typescript-sdk/pull/974)，在该 PR 中，即使不支持也极其容易导致 **completions 开启**。
   - 该 SDK 正在跟进规范中的更改。
- **关于图标元数据 (Icons Metadata) 的辩论**：一位成员询问了工具中图标的使用场景，并注意到一份关于向 **server 添加图标元数据**的提案。
   - 他们还注意到其他 server primitive 中也存在图标。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1423467711941443614)** (24 messages🔥): 

> `chat adapter default, XML format promotion, Tool use models formats, DSPy roadmap, ReAct trajectories` 


- **Chat Adapter 仍为默认设置**：成员们确认 **ChatAdapter** 仍是 DSPy 中的默认设置，[JSON 作为备选方案](https://github.com/stanfordnlp/dspy/blob/main/dspy%2Fpredict%2Fpredict.py#L185)。
   - 他们还提到，鉴于工具使用 (tool use) 模型的兴起，未来 **XML** 可能会成为默认格式。
- **提倡推广 XML**：建议推广 **XML**，因为模型在训练后阶段 (post-training) 越来越多地植入 **tool use RL**，工具使用正被固化。
   - 一位成员认为，*更少的 token* 和 *信息传递效率* 等因素在这一趋势面前是次要的。
- **工具使用模型拥抱 JSON 与 MCP**：讨论强调，许多优秀的工具使用模型是针对两种普遍格式训练的：**OpenAI Function calling** 和 **MCP**。
   - 一位成员指出，**GLM 4.5** 往往比 **JSON** 更自然地倾向于使用 **XML**，而许多其他模型则自然地倾向于 **JSON**。
- **DSPy 路线图位置**：一位成员询问 **DSPy 路线图**的位置，除了关注 [GitHub issues](https://github.com/stanfordnlp/dspy/issues) 和变更日志之外。
   - 讨论中未提供路线图的具体位置。
- **磁盘与内存中的 ReAct 轨迹对比**：一位成员询问是否有人有在磁盘上维护 **ReAct 轨迹**（相对于内存对象）的经验。
   - 他们正在寻找一种更好的方法来让 Agent 运行更长的步骤。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1423673830483689604)** (8 messages🔥): 

> `GPTs vs. Gemini, Meta AI Research Shift, AI Sex Robots` 


- **GPTs 与 Gems 的对决**：一名成员询问了 **ChatGPT** 的 **GPTs** 与 **Google** 的 **Gemini**（原名 Gems）在基于 Prompt 的任务中的实用性对比，并好奇 **Gemini** 的底层模型是否更优。
   - 他们还提到正在寻找可以一起测试这些平台的人。
- **Meta 收紧 AI 控制权**：一名成员注意到 Meta 正在转变 AI 研究的方向，并分享了一篇 [WinBuzzer 文章](https://winbuzzer.com/2025/10/02/meta-tightens-grip-on-ai-research-sparking-internal-anger-and-fears-for-open-culture-xcxwbn/)和一条关于此变化的 [X 帖子](https://x.com/AnjneyMidha/status/1974173661025480810)。
   - 该评论暗示了对 **Meta** 在 AI 领域变得不够开放的担忧。
- **AI 性爱机器人：孤独的未来？**：一名成员思考了 AI 的未来影响，指出 *谁拥有最好的 AI，谁就赢得……一切。*
   - 随后，该成员提出了关于 *不显尴尬的 AI 性爱机器人* 开发时间表的问题，将其设想为解决长期单身的方案。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1423496248614457449)** (3 messages): 

> `bacteriophages, genome language models, computational biology` 


- **Bacteriophages 获得生成式设计助力**：一名成员分享了一篇[论文](https://arxiv.org/abs/2509.22358)和[另一个链接](https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1)，内容是关于利用 **Genome Language Models** 进行新型 **Bacteriophages** 的生成式设计。
   - 该用户打算在 <t:1759550400:R> 左右展示此内容，并指出其在 **Computational Biology** 中的实际应用和相关性。
- **Computational Biology 受到关注**：讨论强调了将计算方法（特别是 **Genome Language Models**）应用于生物学挑战（如 **Bacteriophage** 设计）的兴趣日益浓厚。
   - 分享的资源表明，利用 AI 创建和理解生物系统的趋势正在加强。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1423387256801988730)** (11 messages🔥): 

> `Oracle OpenAI datacenters, LLM RL generalization, Reasoning tokens in LLMs, IRL exploration` 


- **Oracle 为 OpenAI 运行数据中心**：一名成员推测 **Oracle** 的商业模式已从销售数据库和企业软件转向为 **OpenAI** 运行数据中心，并引用了 [Elon Musk 的推文](https://x.com/jubayer_hamid/status/1973438346501501302)和 [openai.com/elon-musk 页面](https://openai.com/elon-musk/)。
- **LLM 是否具有泛化能力？**：一名成员引用研究建议，目前的 **LLM RL** 是可泛化的，至少在具有多样化可验证答案的情况下是这样；同时也同意 **CoT (Chain of Thought)** 可能非常脆弱。
   - 另一名成员质疑了这一说法的支持依据，指出即使经过广泛的 **RL** 训练，模型在处理超出训练期间见过的数字范围的简单乘法外推时仍然很吃力。
- **Reasoning Tokens 真的被使用了吗？**：一名成员表示，目前尚未确定 **LLMs** 是否真的以人们天真假设的方式使用了 **Reasoning Tokens**，并暗示推理可能只是对基于启发式响应的事后解释。
   - 另一名成员表示赞同，指出尽管如此，推理确实极大地提升了性能，但这远未达到我们所知的可能水平，也不是一个已解决的问题。
- **IRL 探索与 LLM**：一名成员建议为 **LLMs** 引入一些 **IRL (In Real Life)** 探索，并链接了一个关于该主题的 [YouTube 视频](https://youtu.be/wsXl4CLOeew)。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1423395810476818453)** (17 messages🔥): 

> `Global USD Pricing Model, Memory Key, AI interaction, Manus's memory architecture` 


- **全球美元定价劣势讨论**：一名成员指出 Manus 使用**全球统一的美元定价模型（Plus 计划为 $39/月）**，未根据地区经济进行调整，这在巴西和拉丁美洲等地形成了准入门槛。
   - 他们建议实施**基于购买力平价 (PPP) 的区域定价**，以便让更多用户能够使用 Manus。
- **Memory Key：强制 Manus 记忆**：一位成员分享了一个 **Memory Key**，这是一个结构化的 prompt，强制 Manus 将整个会话的上下文压缩成一个可复用的摘要，以解决核心平台问题。
   - 这可以显著改善用户体验，解决会话卡顿和 Manus 丢失上下文的问题。
- **AI 交互秘诀：结构化数据**：实验表明，Manus（以及许多 Large Language Models）处理**密集、结构化数据**的效率远高于对话式的“人类”文本。
   - 该成员证明了**结构化数据**能让 Manus 实现更好的召回、分析和性能表现。
- **安全发现：隐私与数据控制**：**Memory Key** 最强大的好处不仅是便利，还在于**隐私和数据控制**。
   - 通过将冗长且敏感的对话折叠成单个摘要，用户可以有效地“删除”易受攻击的历史记录并控制自己的数据。
- **为什么每个人都想了解你的业务**：一位成员分享了一篇关于了解业务的 [LinkedIn 文章](https://www.linkedin.com/pulse/why-everyone-wants-know-your-business-moses-quaye-7s4tf)。
   - 未进行进一步讨论。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1423482796046815375)** (10 messages🔥): 

> `aider-desk, SST/OpenCode, chrome mcp, GLM4.6, Deepseek` 


- ****Aider-Desk** 未获得广泛关注**：一位成员注意到 **aider-desk** 拥有 **MCP 支持**，但惊讶于它没有获得更多采用，且他们主要转向了 **SST/OpenCode**。
   - 另一位成员提到，**Aider** 仍然具有吸引力的主要原因是其用于管理上下文（包括只读模式）的优秀 **UI**。
- ****OpenCode** 学习曲线陡峭**：成员们讨论了 **OpenCode** 有一定的学习曲线，但可能并不比 **aider** 更难。
   - 其他人补充道，通过 **z.AI Coding Plan** 使用 **OpenCode + GLM4.6** 非常强大且*便宜*。
- **新 Chrome MCP 发布**：一位成员提到了一个规范的 chrome mcp，可在 [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) 获取。
   - 另一位成员提到将其与 [claude-cli](https://www.circusscientist.com/2025/10/03/deepseek-browser-testing-with-claude-cli-and-chrome-devtools-mcp/) 配合使用。
- ****Deepseek** 测试**：一位成员表示他们是 **Deepseek** 的粉丝，尤其是通过其 anthropic api 使用，目前他们在 **OpenCode** 中通过 **anthropic-sdk provider** 使用它，因为这样在工具任务上表现更好。
   - 该成员补充说，他们怀念 **Aider** 中对上下文的手动控制（**tokens, add, remove, clear 等**）。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1423451732511555644)** (4 messages): 

> `Polyglot LLM Evaluation, Aider Scala Code Generation Depth, Openrouter Caching Issues` 


- **多语言 LLM 基准测试？**：一位用户询问如何评估 **LLM** 在**多语言问题**上的性能，并特别请求代码示例和示例 Agent。
   - 另一位成员建议将基准测试 **Docker 容器** 作为 CLI 应用使用。
- ****Aider 的 Scala 代码生成受限****：一位用户报告称，当 **Aider** 基于具有深层结构的现有 case classes 生成 **Scala 代码** 时，在第二层之后就会停止对象代码生成。
   - 用户正在寻求一种方法来自动扩展 Aider 的上下文以处理更深层的结构，使用的模型包括 **Deepseek r3**、**GPT5** 和 **deepcoder**。
- ****Openrouter 缓存功能失效****：一位用户报告称 **Openrouter 缓存** 似乎不起作用，尽管 aider 提示显示 **Z.ai provider** 和其他模型已启用缓存，但显示 `"native_tokens_cached": 0`。
   - 用户提供的 aider 详情显示主模型为 **openrouter/z-ai/glm-4.6**，使用 diff 编辑格式，且有 8k token 限制。