---
companies:
- google-deepmind
- google
- openai
- alibaba
date: '2026-03-03T05:44:39.731046Z'
description: '**Google DeepMind** 推出了 **Gemini 3.1 Flash-Lite**，重点强调了用于可调计算资源的“动态思考层级”（dynamic
  thinking levels）。其显著指标包括：**每百万输入 0.25 美元**、**每百万输出 1.50 美元**、**LMArena Elo 评分 1432**，以及比
  Gemini 2.5 Flash 快 **2.5 倍的首个 Token 生成速度**。它支持 **100 万上下文窗口**，并对包括文本、图像、视频、音频和 PDF
  在内的多模态输入具有高吞吐量。


  **OpenAI** 向所有 ChatGPT 用户推送了 **GPT-5.3 Instant**，提升了对话的自然度，并在结合搜索功能的情况下将**幻觉减少了
  26.8%**。与此同时，备受猜测的 **GPT-5.4** 也初露端倪。


  **阿里巴巴的通义千问 (Qwen)** 面临领导层离职，引发了对其未来发展和开源地位的担忧。这些新闻突显了模型效率、定价和多模态能力的进步，以及影响人工智能发展的组织变革。'
id: MjAyNi0w
models:
- gemini-3.1-flash-lite
- gemini-3
- gpt-5.3
- gpt-5.4
- qwen
people:
- jeffdean
- noamshazeer
- sundarpichai
- aidan_mclau
- justinlin610
title: 今天没什么事。
topics:
- multimodality
- latency
- throughput
- context-window
- model-pricing
- model-benchmarking
- model-performance
- conversational-ai
- hallucination-reduction
- api
- model-rollout
- leadership-exit
---

**平静的一天**

> 2026年3月2日至3月3日的 AI 新闻。我们为您查阅了 12 个 Reddit 子板块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**264** 个频道，**12765** 条消息）。为您节省了约 **1137** 分钟的阅读时间（按每分钟 200 字计算）。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾


**Gemini 3.1 Flash‑Lite 发布：“动态思考级别” + 激进的性价比**

- **Gemini 3.1 Flash‑Lite (Preview)** 正式发布，作为 Google 速度最快、成本效益最高的 Gemini 3 系列端点，强调了针对高吞吐量工作负载的 *latency*（延迟）和 *throughput*（吞吐量）。DeepMind 的发布推文将其定位为“大规模智能”，具有可调节的 **thinking levels**（根据任务复杂度调节计算量）[@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2028872381477929185)，并通过 AI Studio / Vertex 推出 API [@Google](https://x.com/Google/status/2028872509601333594)。Jeff Dean 强调了 **$0.25/M input** 和 **$1.50/M output** 的价格，**LMArena 1432 Elo** 以及 **86.9% GPQA Diamond** 表现，同时 **time-to-first-token** 比 Gemini 2.5 Flash 快 2.5 倍 [@JeffDean](https://x.com/JeffDean/status/2028876962580816143)；Noam Shazeer 重申了“thinking levels”的构架，认为它是实现“最高智能、最低延迟”的产品旋钮 [@NoamShazeer](https://x.com/NoamShazeer/status/2028909105969283565)；Sundar Pichai 也转发了同样的关于速度和成本的信息 [@sundarpichai](https://x.com/sundarpichai/status/2028891212573491715)。
- **第三方基准测试与定位**：Artificial Analysis 报告称 Flash‑Lite 保留了 **1M context** 窗口，实测 **>360 output tokens/s**，平均回答延迟约 **5.1s**，相比 2.5 Flash‑Lite 提升了其“智能指数”，但 **定价有所上涨**（综合成本大幅上升）[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2028882198456352852)。Arena 指出 Flash‑Lite Preview 在 Text Arena 排名第 36（1432），在 Code Arena 排名第 35 左右，被认为是性价比前沿的一个强力点 [@arena](https://x.com/arena/status/2028876727989289449)。社区的一个普遍反应是“Flash‑Lite... Google 挺逗的”，主要是针对命名方式和极快的发布节奏 [@JasonBotterill](https://x.com/JasonBotterill/status/2028794893624291569)，以及“Google 发布模型的速度比我完成测试的速度还快” [@matvelloso](https://x.com/matvelloso/status/2028901252437032982)。
- **多模态角度**：Google 员工推介“使用 Flash‑Lite 代替编写解析器”来处理文本+图像+视频+音频+PDF 的摄取 [@koraykv](https://x.com/koraykv/status/2028876507679191392)，强化了 Flash‑Lite 作为生产流中“管道模型（plumbing model）”的地位。

**OpenAI：GPT‑5.3 Instant 推出 + “少一些说教” + 预热 GPT‑5.4**

- **GPT‑5.3 Instant** 已向所有 ChatGPT 用户推出，明确回应了关于 5.2 版本“过于谨慎”且“限制声明（caveats）过多”的投诉。OpenAI 声称改进了对话自然度，减少了不必要的拒绝和防御性免责声明，并提升了搜索集成答案的质量 [@OpenAI](https://x.com/OpenAI/status/2028893701427302559), [@nickaturley](https://x.com/nickaturley/status/2028894581191000404)。OpenAI 还表示幻觉有所减少：根据内部贡献者的说法，**结合搜索时表现提升 26.8%**，**不结合搜索时提升 19.7%** [@aidan_mclau](https://x.com/aidan_mclau/status/2028894122959159434)，并得到了员工的呼应 [@christinahkim](https://x.com/christinahkim/status/2028900228196384978)。
- **API/Arena 曝光**：根据社区报告，API 中出现了 “gpt-5.3-chat-latest” [@scaling01](https://x.com/scaling01/status/2028906108291616773)，并且已在 Text Arena 中提供侧向对比评估 [@arena](https://x.com/arena/status/2028908848204177682)。
- **GPT‑5.4 预热**：OpenAI 发布了一条互动率极高的推文“比你想象（Think）的还要快” [@OpenAI](https://x.com/OpenAI/status/2028909019977703752)，引发了关于发布顺序与“5.3 Thinking 和 Pro 将紧随其后”传闻的困惑 [@kimmonismus](https://x.com/kimmonismus/status/2028924631084605465)。多条推文推测，在 DoD/NSA 合同争议期间，5.4 也被用作一种*新闻周期防御手段（news-cycle deflection）* [@kimmonismus](https://x.com/kimmonismus/status/2028803185347875207)。

**阿里巴巴 Qwen 震荡：领导层离职，“没有人才，Qwen 一无是处”，以及开源的不确定性**

- **关键人员离职**：整个数据集中的一个主要线索是 Qwen 技术领导层和高级贡献者的退出。Justin Lin 的“卸任”帖子引发了广泛反应 [@JustinLin610](https://x.com/JustinLin610/status/2028865835373359513)，随后出现了大量高信号的确认与致敬，接着是更多成员的退出，包括另一位领导者（“再见 Qwen，我也是”）[@huybery](https://x.com/huybery/status/2028976346416988612) 以及一份独立的告别声明 [@kxli_2000](https://x.com/kxli_2000/status/2028880971945394553)。外部观察者将此描述为阿里云“踢出”了 Qwen 的技术负责人 [@YouJiacheng](https://x.com/YouJiacheng/status/2028880908305219729)。
- **技术层面的重要性**：许多工程师将 Qwen 视为开源模型生态系统的**关键基础设施**——尤其是 **<10B** 和“帕累托前沿（Pareto frontier）”模型，以及 VLM/OCR 衍生模型。如果开源权重的发布节奏放缓或许可立场发生转变，这被视为真正的生态系统风险 [@natolambert](https://x.com/natolambert/status/2028893211759124890), [@teortaxesTex](https://x.com/teortaxesTex/status/2028874511509000646), [@awnihannun](https://x.com/awnihannun/status/2028902061384057211)。鉴于“仅仅拥有受欢迎的开源模型是不够的”，人们也在立即猜测 Qwen 的 OSS 姿态是否会改变 [@code_star](https://x.com/code_star/status/2028913595602616391)。
- **组织架构诊断**：一种反复出现的解读是，在更高级别的阿里巴巴结构下（向 CEO 汇报）进行的“统一”带来了围绕影响力和可见性的政治压力 [@Xinyu2ML](https://x.com/Xinyu2ML/status/2028891170592473385)；还有更广泛的评论指出，大厂层级体系往往会惩罚那些建立外部信任的“桥梁”人物 [@hxiao](https://x.com/hxiao/status/2028932213228900701)。
- **尽管动荡，发布仍在继续**：Qwen 3.5 LoRA 微调指南和低 VRAM 训练方案迅速传播（尤其是 Unsloth）[@UnslothAI](https://x.com/UnslothAI/status/2028845314506150079)，并且支持 vLLM/SGLang 的 **GPTQ Int4** 权重也得到了推广 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2028846103257616477)。社区还推动了围绕 Qwen 3.5 的教育和重新实现 [@rasbt](https://x.com/rasbt/status/2028961822372425941)。目前的紧张局势在于：*强劲的发布速度*与*领导层流失*并存。

**长上下文 + 训练效率：让“不可能”的上下文窗口变得实用**

- **长上下文训练的 Attention 内存占用降低 87%**：Together 的一篇论文强调了 **Context Parallelism**（上下文并行）与 **Sequence Parallel-style head chunking**（序列并行式 Head 分块）的混合方案，声称在 **8×H100**（单节点）上训练 **5M 上下文窗口的 8B 模型**，并将 Attention 内存占用减少了高达 **87%** [@rronak_](https://x.com/rronak_/status/2028718679123497007)。该推文还指出一个现实差距：由于内存成本，大多数针对长上下文前沿模型的 RL 后训练仍仅在全上下文的一小部分上进行。
- **FlashOptim (Databricks)**：开源优化器实现（AdamW/SGD/Lion），在减少内存占用的同时保持更新等效性——推文宣布可通过 `pip install flashoptim` 安装 [@davisblalock](https://x.com/davisblalock/status/2028943987349045610)，MosaicAI 总结称其能**减少 >50% 的训练内存**，例如将 AdamW 训练开销从约 **16 bytes/param** 降至 **7 bytes**（配合梯度释放可达 **5**），并将一个 8B 微调示例的峰值从 **175 GiB 降至 113 GiB** [@DbrxMosaicAI](https://x.com/DbrxMosaicAI/status/2028977216940589383)。
- **用于 RL 的异构基础设施**：SkyPilot 认为 RL 后训练应该将工作负载拆分到 **高性能 GPU（训练器/trainer）**、**廉价 GPU（采样/rollouts）**和 **高内存 CPU（经验回放缓存/replay buffers）**；Job Groups 提供了一个单一的 YAML 编排模型，具有协调的生命周期和任务发现机制 [@skypilot_org](https://x.com/skypilot_org/status/2028878888211013907)。
- **Kernel/工具链陷阱**：一份 CuTeDSL + torch.compile 的回归报告指出，当通过 Custom ops 使封装后的 Kernel（包括 RMSNorm “Quack” Kernel）兼容编译时，会出现约 **2.5 倍的减速**——这突显了 Kernel 级速度与图编译（Graph Compilation）需求之间的摩擦 [@maharshii](https://x.com/maharshii/status/2028863745641112008)。

**Agent 工程现状核查：基准测试 vs “实际工作”、共识失败以及工具链转向 (MCP, sandboxes, observability)**

- **基准测试与劳动经济学不匹配**：一个新的数据库试图将 Agent 基准测试映射到现实世界的任务分布，认为目前的评估过度偏向数学/编程，而大多数劳动力/资本其实分布在其他领域 [@ZhiruoW](https://x.com/ZhiruoW/status/2028847081507488011)。这一观点被推崇为“AI 基准测试在实际工作中面临的核心问题” [@emollick](https://x.com/emollick/status/2028870529906622677)。Arena 推出的 **Document Arena** 是对此的直接回应：提供真实 PDF 推理的并排评估；根据 Arena 的数据，Claude Opus 4.6 目前处于领先地位 [@arena](https://x.com/arena/status/2028915403704156581)。
- **多 Agent 协作是脆弱的**：拜占庭共识博弈（Byzantine consensus games）显示，即使在良性环境下，LLM Agent 达成一致也是不可靠的；失败往往更多源于 **停滞/超时** 而非恶意破坏，且随着群体规模的扩大而恶化 [@omarsar0](https://x.com/omarsar0/status/2028823724196343923)。关于心智理论（Theory of Mind）+ BDI + 符号验证的补充研究表明，认知的“ToM 模块”并不能自动提供帮助；提升很大程度上取决于基础模型的能力 [@omarsar0](https://x.com/omarsar0/status/2028913061260935331)。
- **MCP “已死？” vs MCP 扩张**：来自 DAIR 的 Omar 明确提出了“MCP 已死？”的疑问 [@omarsar0](https://x.com/omarsar0/status/2028840977922674842)，但在同一数据集中，MCP 的采用正在扩大：Notion 发布了针对会议纪要（Meeting Notes）的 MCP/API 支持（可通过 Claude Code 一行代码安装） [@zachtratar](https://x.com/zachtratar/status/2028881783551570209)；Cursor 推出了 **MCP Apps**，允许 Agent 在聊天界面内渲染交互式 UI [@cursor_ai](https://x.com/cursor_ai/status/2028953584407085546)。
- **“干掉代码审查”之争**：swyx 将消除人工代码审查（Code Review）视为 Agentic Engineering 和 SDLC 倒置的“最终 Boss” [@swyx](https://x.com/swyx/status/2028795270306079156)。反方观点：thdxr 认为，通过 LLM “产出这么多代码”的团队可能使用方式不对；庞大的代码量会产生弄巧成拙的代码库，而 LLM 自身也难以应对由此产生的复杂性 [@thdxr](https://x.com/thdxr/status/2028827251534352764)。
- **沙箱化 “computer use” 平台**：Perplexity 的 “Computer” 功能引发了大量关注：Srinivas 正在征集功能需求 [@AravSrinivas](https://x.com/AravSrinivas/status/2028742933403574585)，Perplexity 将其产品定位为编排多个模型，并利用受管安全沙箱（无需 API Key 管理）直接嵌入到应用中 [@AravSrinivas](https://x.com/AravSrinivas/status/2028903680616087946), [@AskPerplexity](https://x.com/AskPerplexity/status/2028893546447814895)。Cursor 的云端 Agent 同样运行在隔离的虚拟机（VM）中，并输出带有 Artifacts 的、可直接合并的 PR [@dl_weekly](https://x.com/dl_weekly/status/2028844128729973060)。

**人才、治理与信任：Anthropic 对决美国国防部 (DoD)、OpenAI 合同审查以及备受关注的人事变动**

- **Max Schwarzer（OpenAI 后训练副总裁）→ Anthropic**：一次重大的人事变动：Schwarzer 宣布在领导后训练（Post-training）并交付 GPT-5/5.1/5.2/5.3-Codex 后离开 OpenAI，加入 Anthropic 回归 IC 身份进行 RL 研究 [@max_a_schwarzer](https://x.com/max_a_schwarzer/status/2028939154944585989)。这加剧了“Anthropic 的重大胜利” [@kimmonismus](https://x.com/kimmonismus/status/2028952074063331421) 以及更广泛的“传奇人物流失”的焦虑 [@yacinelearning](https://x.com/yacinelearning/status/2028880802797199476)。
- **Anthropic 与五角大楼/Palantir 的紧张关系**：据报道，美国国防部（DoD）威胁要将 Anthropic 标记为“供应链风险”，这可能会影响 Palantir 在联邦工作中的使用；Anthropic 希望建立保障措施（针对大规模国内监控 + 自主武器） [@srimuppidi](https://x.com/srimuppidi/status/2028943303581024412)，并有额外的报道指出这一点 [@aaronpholmes](https://x.com/aaronpholmes/status/2028942999548297464)。
- **OpenAI–DoD / NSA 信任危机**：多条推文要求查看实际的合同文本，认为“附带（incidental）”监控的措辞在历史上曾允许无需授权的国内监控；批评者引用了 PRISM/Upstream 和 FISA/EO 12333 的背景 [@jeremyphoward](https://x.com/jeremyphoward/status/2028805970214912125)，并呼吁进行独立的法律红队审查（Legal Red-teaming），而非仅听信“相信我们”的保证 [@sjgadler](https://x.com/sjgadler/status/2028899096283758732)。这被反复与“OpenAI 将利用模型发布来引导舆论”的假设联系在一起。
- **市场份额声明**：一条疯传的声明称，Claude 在一年内从少数份额飙升至占据美国商业市场份额的主导地位（相对于 ChatGPT） [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2028974344710606905)。在验证基础数据集之前，应将其视为趋势参考，但这反映了人们感知到的势头：“编程 + Agent 策略奏效了。”

---

### 热门推文（按互动量排序，技术聚焦）

- **GPT‑5.4 预热**：“5.4 比你想象的更早。” [@OpenAI](https://x.com/OpenAI/status/2028909019977703752)  
- **Gemini 3.1 Flash‑Lite 发布帖** [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2028872381477929185)  
- **GPT‑5.3 Instant 推出 + “较少说教”** [@OpenAI](https://x.com/OpenAI/status/2028893701427302559)  
- **Qwen 领导层离职（“卸任”）** [@JustinLin610](https://x.com/JustinLin610/status/2028865835373359513) 以及后续的告别帖 [@huybery](https://x.com/huybery/status/2028976346416988612)  
- **Unsloth：宣称 Qwen3.5 LoRA 仅需约 5GB VRAM + notebook** [@UnslothAI](https://x.com/UnslothAI/status/2028845314506150079)  
- **Cursor：MCP Apps（Agent 对话内的交互式 UI）** [@cursor_ai](https://x.com/cursor_ai/status/2028953584407085546)  
- **Together 显著降低长上下文训练内存占用（高达 87%）** [@rronak_](https://x.com/rronak_/status/2028718679123497007)


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.5 模型发布与基准测试

  - **[Qwen 2.5 -> 3 -> 3.5，最小模型。跨代际的改进令人惊叹。](https://www.reddit.com/r/LocalLLaMA/comments/1rjd4pv/qwen_25_3_35_smallest_models_incredible/)** (热度: 1017): **Qwen 3.5** 是 Qwen 模型系列的一次显著进步，其中包含一个 `0.8B` 参数的模型，且该模型集成了一个视觉编码器，这意味着语言模型组件的体积甚至更小。该模型符合更小、更高效模型的发展趋势，例如目前的微型 MoE (Mixture of Experts) 模型，其性能备受好评。尽管体积精简，Qwen 3.5 仍因事实准确性问题受到批评，例如关于飞机发动机的错误信息，这凸显了进行严格事实核查的必要性。评论者强调了像 Qwen 3.5 这样的小型模型在本地机器上实现个人助手的潜力，强调了它们对于 GPU 资源有限的用户的效率和可访问性。然而，人们对该模型产生事实幻觉的倾向表示担忧，这可能会损害其可靠性。

    - 较小的 Qwen 模型，特别是 MoE (Mixture of Experts) 模型，因其较前代产品显著的性能提升而备受关注。这些模型在本地机器上的个人使用变得越来越可行，即使在较小的规模下，也能在效率和能力上提供显著进步。
    - 一位用户强调了 Qwen 3.5 中的幻觉问题，指出了与飞机发动机类型和配置相关的具体事实错误。这强调了对 AI 模型输出进行事实核查的重要性，因为它们可能会非常自信地提供错误信息。
    - 小型量化模型（如 4B 模型）在较低配置硬件上的效率受到了赞誉。一位用户报告称，使用 `llama.cpp` 在 128k 上下文下达到了每秒 60 个 token 的速度，这被认为比旧的、更大的模型有了显著改进。这展示了在本地、资源受限的环境中实现高性能 AI 的潜力。

  - **[可视化对比所有 Qwen 3.5 与 Qwen 3 的基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1rivckt/visualizing_all_qwen_35_vs_qwen_3_benchmarks/)** (热度: 736): **该图表是一个柱状图，直观展示了新版 **Qwen 3.5 模型**与旧版 **Qwen 3 模型**在各种基准测试中的性能对比，包括知识与 STEM、指令遵循、长上下文、数学、编程、通用 Agent 以及多语言能力。图表使用不同颜色来区分模型版本，**紫色/蓝色/青色**代表新的 Qwen 3.5 模型，**橙色/黄色**代表旧的 Qwen 3 模型。该图表旨在提供模型性能的快速视觉对比，尽管小型模型的一些数据缺失。用于此可视化的原始数据可在 [Google Sheet](https://docs.google.com/spreadsheets/d/1A5jmS7rDJe114qhRXo8CLEB3csKaFnNKsUdeCkbx_gM/edit?usp=sharing) 中查阅。** 一些评论者批评了图表的清晰度和实用性，其中一人对基准测试结果表示怀疑，暗示对性能声明的准确性持保留态度，特别是关于 Qwen 3.5 模型在每项测试中都优于 Qwen 3 模型的说法。

- 基准测试结果显示，Qwen 3.5 模型（特别是 9B dense 模型）的表现异常出色，甚至优于 Qwen 3 122B A10B 等更大型的模型。考虑到尺寸差异，这令人惊讶，因为 9B 模型比后者小 10 倍以上，但在 Knowledge & STEM、Instruction Following 和 Multilingualism 等各个领域都能紧随其后。
- 有人对基准测试的有效性表示怀疑，因为一位评论者认为 Qwen 3.5 35B A3B 模型在所有测试中都优于 Qwen 3 235B A22B 模型是难以置信的。这引发了对这些基准测试可靠性以及它们是否准确反映模型能力的质疑。
- 一位评论者提供的详细基准测试表突出了各种 Qwen 模型在不同类别的具体性能指标。例如，Qwen 3.5-122B-A10B 模型在 Instruction Following 和 Math 方面的得分高于其前代产品，表明在这些领域有所进步。然而，数据呈现方式因难以解读而受到批评。

- **[在浏览器中通过 WebGPU w/ Transformers.js 本地运行 Qwen 3.5 0.8B](https://www.reddit.com/r/LocalLLaMA/comments/1rizodv/running_qwen_35_08b_locally_in_the_browser_on/)** (Activity: 501): **Qwen 3.5 Small** 模型（包括一个 `0.8B` 参数变体）已发布用于端侧应用，并附带了一个使用 **WebGPU** 和 **Transformers.js** 在浏览器中本地运行的演示。该实现突出了在浏览器中运行此类模型的能力，尽管 **vision encoder** 被认为是性能瓶颈。这些模型可在 [Hugging Face](https://huggingface.co/collections/Qwen/qwen35) 上获取，演示地址见[此处](https://huggingface.co/spaces/webml-community/Qwen3.5-0.8B-WebGPU)。有评论建议通过 `llama.cpp WASM` 使用 `q4 GGUF` 以在没有 VRAM 问题的情况下获得更好的吞吐量，这表明人们倾向于使用替代方法来优化性能。另一条评论澄清说，该演示不处理视频输入，而是处理静态截图。

    - WebGPU 中的 vision encoder 被认为是瓶颈，建议通过 `llama.cpp WASM` 使用 `q4 GGUF` 来提高吞吐量。这种方法可以在浏览器中运行，而不会导致 VRAM 抖动（VRAM thrashing），这是 WebGPU 实现中的常见问题。
    - 关于输入类型的澄清：模型不处理视频输入，而是在发送 Prompt 时对当前屏幕进行截图。这一区别对于理解模型的输入处理能力至关重要。
    - 报告了一个技术问题，即“开始”按钮无响应，导致用户无法启动进程。这可能表明用户界面存在 Bug 或应用程序的初始化序列有问题。


### 2. Qwen 3.5 模型性能与应用

- **[Unsloth 修复版的 Qwen3.5-35B-A3B 在研究任务中表现惊人](https://www.reddit.com/r/LocalLLaMA/comments/1rjh5wg/unsloth_fixed_version_of_qwen3535ba3b_is/)** (Activity: 417): **Unsloth** 更新版的 **Qwen3.5-35B-A3B** 在处理研究任务方面表现出显著改进，特别是在修复了 tool calling 问题之后。该模型拥有 `35 billion parameters`，并采用了 hybrid linear attention，在不增加内存占用的情况下使原生 context length 翻倍。它在 **Ryzen AI Max+ 395 系统**上使用 `llama.cpp-rocm` 进行了测试，参数包括 `--ctx-size 262144` 和 `--n-gpu-layers 999`，实现了 `600+ tokens/second` 的 Prompt 处理速度和 `25-30 tokens/second` 的 Token 生成速度。该模型有效地执行了 `14 web searches` 和 `4 full page fetches`，在 tool usage 上保持了平衡，相比于 **GLM-4.7-Flash** 等之前的模型有显著改进。该模型在为 Linux Fedora 43 系统提供远程桌面解决方案方面的表现可与 frontier models 相媲美，尽管有人指出它本可以更强烈地推荐 **Sunshine+Moonlight**。一位评论者指出 **RustDesk** 是更优的远程桌面解决方案，特别是对于所述的配置，尽管原帖关注的是 KRdp 和其他选项。另一条评论提到了 **LM Studio** 无法在 system prompts 中解析 `{{CURRENT_DATE}}` 的潜在问题，表明需要修复。

总结评论时出错。

- **[Qwen 3.5 27b: a testament to the transformer architecture](https://www.reddit.com/r/LocalLLaMA/comments/1rj6m71/qwen_35_27b_a_testament_to_the_transformer/)** (Activity: 557): **Qwen 3.5 27b** 展示了 Transformer 架构的显著进步，在推理和知识测试表现上足以媲美 **R1 0528**。值得注意的是，它采用了混合架构，其中 `75%` 的层利用了 **Gated DeltaNet linear attention** 而非完整的 Transformer 设置。该模型仅凭 `27b` 参数就能达到如此高的水平，并能适配单个消费级 GPU，这标志着相对于此前需要 `70b` 参数和集群级算力才能完成类似任务的模型而言，是一个巨大的飞跃。由于其强大的基础能力，该模型在 Fine-tuning（特别是编程应用）方面的潜力也备受关注。评论者强调了该模型改进的 Instruction-following 能力以及通过 Fine-tuning 增强其个性的潜力。**Gated DeltaNet linear attention** 的使用被视为一项重大的架构创新，为其效率和性能做出了贡献。

    - victory_and_death 指出 Qwen 3.5 27b 并未完全使用传统的 Transformer 架构。相反，它在 75% 的层中采用了 Gated DeltaNet linear attention，这与标准的 Transformer 模型有显著不同。这种架构选择可能有助于提高其性能效率，并使其能够在消费级硬件上运行。
    - Pitiful-Impression70 指出了 Qwen 3.5 27b 模型令人印象深刻的性能，指出它可以与 R1 0528 等更大的模型竞争。一个 27B 参数的 Dense 模型可以执行以前需要 70B 参数模型才能完成的任务，这一事实令人瞩目，尤其是它可以在单个消费级 GPU 上运行。这突显了模型效率和能力的飞速提升。
    - National_Meeting_749 讨论了像 Qwen 3.5 27b 这样较新模型改进的 Instruction-following 能力。这些模型可以结合 System Prompts 来注入个性，提升交互质量。与前几代模型相比，这种处理指令能力的提升是一个重大的进步。

  - **[Running Qwen3.5-0.8B on my 7-year-old Samsung S10E](https://www.reddit.com/r/LocalLLaMA/comments/1rj5ngc/running_qwen3508b_on_my_7yearold_samsung_s10e/)** (Activity: 330): **图片展示了在三星 S10E 上使用 `llama.cpp`（一种在本地设备运行 LLM 的工具）成功运行 Qwen3.5-0.8B 模型。考虑到手机的机龄和硬件限制，该模型达到了 `12 tokens per second` 的处理速度，表现出色。这展示了在旧硬件上运行复杂 AI 模型的潜力，利用 `llama.cpp` 中的 NEON SIMD 路径来增强 ARM 芯片的性能。模型能够进行连贯对话并执行复杂任务的能力，突显了 AI 效率和可访问性的重大进步。** 评论者对性能感到惊讶，指出在一年前，在如此旧的设备上实现这种模型的对话能力是难以想象的。此外，大家对 `llama.cpp` 的安装过程以及使用的具体 Quantization（Q4_0 或 Q8）也充满了技术好奇。

    - sean_hash 强调了在 Snapdragon 855 上运行 Qwen3.5-0.8B 的性能，达到了 `12 tokens per second`。对于旧的 ARM 芯片来说，这被认为是非常出色的，这要归功于 `llama.cpp` 中的 NEON SIMD 路径，它显著优化了此类硬件上的性能。
    - rm-rf-rm 询问了 `llama.cpp` 的安装过程，表明了对复制该设置的兴趣。这反映了对实现细节以及在旧设备上运行 LLM 可能面临的挑战的技术好奇。
    - WPBaka 质疑了 0.8B 模型的实际应用，对除了基础对话之外的能力表示怀疑。这反映了关于小型模型在现实场景中实用性的更广泛辩论，尤其是与更大、更强大的模型相比时。

### 3. Apple M5 Pro 和 M5 Max 发布

  - **[Apple 发布 M5 Pro 和 M5 Max，称其 LLM 提示词处理速度比 M4 Pro 和 M4 Max 快达 4 倍](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)** (热度: 822): **Apple 发布了 M5 Pro 和 M5 Max 芯片，据称与其前代 M4 Pro 和 M4 Max 相比，其大语言模型 (LLM) 提示词的处理速度最高可提升 4 倍。M5 Pro 支持高达 64GB 的 unified memory，带宽为 307GB/s；而 M5 Max 支持高达 128GB 的 unified memory，带宽为 614GB/s。此外，这些芯片的 SSD 速度提升了 2 倍，达到 14.5GB/s，并集成了 Apple N1 无线芯片以支持 Wi-Fi 7，如果用户的路由器兼容，则可提高下载速度。发布会相关的图像强调了这些芯片在高效处理 3D 建模和编程等复杂任务方面的能力。** 一些用户对新芯片缺乏更先进的 AI 专用硅片（如 Neural Accelerator）表示失望。另一些用户则对这些芯片在未来 Mac Studio 机型中的潜力感到兴奋。

    - M5 Pro 和 M5 Max 芯片在内存和带宽能力上有显著提升。M5 Pro 支持高达 64GB 的 unified memory，带宽为 307GB/s，而 M5 Max 支持高达 128GB 的 unified memory，带宽为 614GB/s。这些增强功能对于高效处理大规模机器学习模型和数据密集型应用至关重要。
    - 新芯片还引入了快达 2 倍的 SSD 速度，达到 14.5GB/s，这可以显著减少数据访问时间并提高整体系统性能。此外，加入 Apple N1 无线芯片以支持 Wi-Fi 7，可提供更快的下载速度（前提是网络基础设施支持），从而增强数据密集型任务的连接性。
    - 人们对 M5 Max 的潜在性能充满期待，尤其是在未来的 Mac Studio 机型中。M5 Max 的能力可以让我们窥见 M5 Ultra 版本的预期表现，尽管有人猜测 Mac Studio 的更新可能会推迟到 M6 发布。这突显了 Apple 产品发布周期中的战略规划。

  - **[与五角大楼达成协议后，ChatGPT 卸载量飙升 295%](https://www.reddit.com/r/LocalLLM/comments/1rjlzgy/chatgpt_uninstalls_surged_by_295_after_pentagon/)** (热度: 348): **该图片是一个梗图，幽默地暗示了 ChatGPT 与五角大楼之间所谓的协议与 ChatGPT 应用卸载量大幅增加（描绘为 295% 的激增）之间的相关性。图片使用了下降的图表和五角大楼标志等视觉元素，暗示了用户的负面反应。然而，正如评论中质疑卸载数据的有效性和规模所指出的那样，该说法缺乏来源和背景。评论还对卸载率相对于总用户群的重要性表示怀疑。** 评论者对这一说法表示怀疑，质疑卸载率的来源和意义，认为这可能只是较小的波动，而非实质性的趋势。

    - 关于 ChatGPT 在五角大楼协议后卸载量激增 295% 的说法，引发了人们对受影响用户群规模的疑问。一位评论者推测，这可能只是整体流失率的轻微波动，认为相对于总用户群，卸载的绝对数量可能很小。
    - 讨论涉及了 AI 在军事应用中的影响，一位评论者指出，将 AI 集成到国防系统中是技术进步的自然演变。这反映了人们对 AI 在军事背景下部署的伦理和战略层面的广泛关注。
    - 提供了一个 TechCrunch 文章的链接，该文章似乎证实了关于卸载量激增的说法。这表明信息可能是可靠的，尽管原始帖子的说法最初因缺乏来源而受到质疑。


## 非技术类 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude 与 Claude Code 流量与功能

  - **[我们知道原因了！](https://www.reddit.com/r/singularity/comments/1rjyy3f/we_know_why/)** (活跃度: 994): **该图片是一张来自名为 Thariq 的用户的推文，讨论了 Claude and Claude Code 流量出现超乎预期的增长。推文中对用户的耐心表示感谢，并表示他们正在进行扩容。这表明这项可能与 AI 或编程相关的服务正在经历快速增长，并在满足需求方面面临扩展挑战。评论暗示了更广泛的行业动态，如影响科技公司的竞争和政治因素。** 一条评论指出，很少有公司愿意反对美国政府，暗示了科技行业动态中的政治因素。另一条评论幽默地建议，扩展问题可能是顶级 AI 公司的战略弱点。

    - FalconsArentReal 讨论了一个潜在的技术问题，据称 AWS 在中东的一个数据中心遭到了来自伊朗的导弹袭击。据报道，这一事件影响了使用 AWS 作为数据中心提供商的 Anthropic。评论者推测，来自中东的流量被重新路由到北美数据中心，而北美数据中心由于用户从 OpenAI 转向，已经处于压力之下，从而导致了重大的运营挑战。
    - legaltrouble69 指出了 AI 行业的一个战略漏洞，提出如果前两大 AI 公司之一被“取消”或中断，另一家可能会面临无法克服的扩展问题。这一评论强调了 AI 领域内部的相互依赖和潜在的脆弱性，即一个主要参与者的失败可能会对另一个产生连锁反应。
    - SomewhereNo8378 发表了关于公司不愿反对美国政府的政治观察，暗示那些反抗的公司会面临重大挑战。这条评论虽然没有深入的技术内容，但触及了可能影响技术公司及其运营的更广泛的社会政治环境。

  - **[Claude and Claude Code 的流量本周增长快于预期](https://www.reddit.com/r/ClaudeAI/comments/1rjyp7d/claude_and_claude_code_traffic_grew_faster_than/)** (活跃度: 1518): **Anthropic 报告称，其 AI 模型 Claude and Claude Code 的流量出现了超出预期的激增，超过了他们的预测。用户数量的增加促使该公司扩展其基础设施以满足需求。Thariq 的推文强调了这种快速增长带来的挑战，并感谢用户在扩容过程中的耐心。图片是该推文的截图，强调了流量激增的意外性以及公司的应对措施。** 一位评论者推测，流量增加可能是由于付费订阅用户增多，而另一位评论者指出遇到了更快的限制，暗示系统可能承受着压力。


  - **[新功能：Claude Code 正在推出语音模式，今日面向约 5% 的用户开放，详情如下](https://www.reddit.com/r/ClaudeAI/comments/1rjkwqk/new_voice_mode_is_rolling_out_now_in_claude_code/)** (活跃度: 950): **Claude Code 引入了新的 Voice Mode 功能，目前已向约 `~5%` 的用户开放，并计划进行更广泛的推广。该功能允许用户通过按住空格键使用 push-to-talk 机制来口述文本，文本将直接在光标位置流式传输，而不会覆盖现有文本。重要的是，使用 Voice Mode 不会产生额外费用或影响 Token rate limits，并且在 Pro, Max, Team, 和 Enterprise 套餐中均可使用。[来源](https://x.com/i/status/2028628570692890800)** 一位用户表示，希望有一个更具互动性的语音助手，能够进行实时讨论，类似于他们将 ChatGPT 用于会议和提案的方式。这表明用户对语音功能中更先进的对话能力有需求。

    - universenz 强调了 Claude Code 中 Voice Mode 在创建更具互动性和动态感的个人助手方面的潜力。他们将其与使用 ChatGPT 的语音功能进行会议和提案进行了比较，AI 可以将口头讨论转换为简洁的技术摘要。这种方法允许对想法进行更彻底和详细的探索，类似于与人类团队合作。
    - PulpAssets 评论了 Claude 的新语音功能对初创生态系统的影响，特别提到了它可能如何颠覆 Wispr Flow 等公司。这表明大型 AI 模型中的单一功能可能会通过大规模提供类似功能，从而显著影响利基初创公司。

### 2. Gemini 3.1 Flash-Lite 发布与基准测试

  - **[Gemini 3.1 Flash Lite](https://www.reddit.com/r/Bard/comments/1rjtfa3/gemini_31_flash_lite/)** (Activity: 394): **该图片提供了 Google Gemini 3.1 Flash Lite 的预览，这是一款专为高吞吐量用途设计的高效率模型，具有高达 `1,048,576` 的显著上下文容量。该发布计划于 2026 年 3 月 3 日进行，并包含了输入、输出和音频 Token 的定价详情。该模型似乎被定位为 Gemini 2.5 Flash Lite 的继任者，但其成本显著增加，这引发了用户关于其在现有实现中经济可行性的争论。** 评论者对 Gemini 3.1 Flash Lite 增加的成本表示担忧，指出其价格比前代产品 Gemini 2.5 Flash Lite 高出 `3x`，后者的定价为输入 `$0.1`、输出 `$0.4`、音频 `$0.3`。这导致了当前用户对其通用性的怀疑。

    - Scary_Light6143 强调了 Gemini 3.1 Flash Lite 模型带来的显著成本增加，指出其价格较 2.5 版本上涨了 3 倍。这引发了对升级实用性的担忧，因为对于大多数实现来说，成本可能无法证明其性能改进的合理性。
    - Accurate-Tap-8634 提供了 Gemini 2.5 Flash Lite 模型的具体定价细节，声明其输入为 `$0.1`，输出为 `$0.4`，音频为 `$0.3`。这些信息对于比较新版 3.1 的成本效益至关重要。
    - cmredd 指出 Gemini 3.1 Flash Lite 的输入成本增加了 `2.5x`，输出成本增加了 `3.75x`。他们对 AI 模型变得越来越贵的趋势提出质疑，认为对于大多数用例来说，基准测试的改进可能无法支撑更高的成本。

  - **[Gemini 3.1 Flash-Lite Benchmark Comparison](https://www.reddit.com/r/Bard/comments/1rjusj5/gemini_31_flashlite_benchmark_comparison/)** (Activity: 146): **该帖子讨论了 Gemini 3.1 Flash-Lite 与之前模型之间的基准测试比较，特别提到比较是针对 2.5 Flash 而非 3 Flash 进行的。Gemini 3.1 Flash-Lite 的模型卡（model card）可以在[此处](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/)找到，而 3 Flash 的模型卡可以在[此处](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf)找到。讨论强调 Gemini 3.1 Flash-Lite 的价格是 2.5 Flash Lite 的两倍，具体定价细节为：`3.1 Flash Lite - $0.25 input/$1.50 output`，而 `2.5 Flash Lite - $0.10 input/$0.40 output`。这表明虽然 3.1 Flash Lite 比 3 Flash 便宜，但在处理大规模数据任务时可能不具备性价比。**

    - **Important-Farmer-846** 强调了 2.5 Flash Lite 相对于 3.1 Flash Lite 的成本效益，指出虽然 3.1 的价格是 Flash 3 的一半，但它是 2.5 Flash Lite 的两倍。评论者建议，对于处理大量数据，2.5 Flash Lite 由于成本较低且性能相近，仍然是更好的选择。
    - **ExpertPerformer** 提供了各种模型的详细成本比较，显示 3.1 Flash Lite 与 MinMax M2.5 和 Grok 4.1 等其他模型相比性价比更低。例如，3.1 Flash Lite 的价格为 `$0.25 input/$1.50 output`，而 2.5 Flash Lite 为 `$0.10 input/$0.40 output`，Grok 4.1 为 `$0.20 input/$0.50 output`。这表明 3.1 Flash Lite 在性价比方面可能缺乏竞争力。
    - **ThomasMalloc** 讨论了 3.1 Flash Lite 在 “High” 思考模式下的低效，指出其耗时比 2.5 Flash Lite 长 14 倍。该模型的输出 Token 达到了 65,436 个，而 2.5 Lite 仅为 6,980 个，这表明其 Token 使用量过度。评论者建议使用 “Minimal” 或 “Low” 思考模式来降低 Token 使用量和成本，因为这些模式在较少 Token 的情况下表现尚可。


### 3. OpenAI 和 ChatGPT 的抵制潮

- **[Damnnnn!](https://www.reddit.com/r/singularity/comments/1rjc5to/damnnnn/)** (热度: 2419): **该图片是来自 X.com 上 TechCrunch 的模因风格截图，强调了在与国防部 (DoD) 达成协议后，ChatGPT 的卸载量大幅增加了 `295%`。这表明公众对 DoD 参与 ChatGPT 的行为产生了强烈抵制或隐私担忧。该帖子获得了大量关注，表明了广泛的兴趣或担忧。然而，一条热门评论指出，如果没有上下文，百分比增长可能会产生误导，因为它可能代表一个较小的绝对数值变化。另一条评论推测了财务影响，认为虽然用户卸载可能会影响收入，但 DoD 合同可能会抵消这一损失。讨论还涉及隐私担忧，质疑在政府合同背景下使用 OpenAI 产品的问题。** 评论者辩论了卸载潮的意义，一些人认为百分比增长在没有绝对数字的情况下可能有误导性。其他人讨论了流失订阅者与获得政府合同之间的财务权衡，并表达了对 OpenAI 与 DoD 合作带来的隐私担忧。

    - mazdarx2001 强调了用户取消订阅的财务影响，指出如果 100 万月付 $20 的用户取消，将导致每月 $2000 万的收入损失。然而，他们认为国防部 (DoD) 的合同可能会抵消这一损失，因为它可能会带来更多由纳税人资金资助的收入。
    - Orangeshoeman 讨论了国防部合同对 OpenAI 下游企业收入的潜在影响。他们建议注重隐私的用户可能会避开 OpenAI 产品，暗示该合同可能会损害 OpenAI 在注重隐私的消费者中的声誉。

  - **[ChatGPT Uninstalls Surge 295% After OpenAI’s DoD Deal Sparks Backlash](https://www.reddit.com/r/ChatGPT/comments/1rjfipu/chatgpt_uninstalls_surge_295_after_openais_dod/)** (热度: 2938): **OpenAI 最近与美国国防部的合作伙伴关系导致 ChatGPT 移动应用的卸载量增加了 `295%`，反映了用户的强烈抵制。这一反应突显了 AI 领域政府合同的声誉风险，因为用户情绪会严重影响企业策略。该事件还促使竞争对手 Claude 的下载量上升，表明 AI 市场的竞争格局正在发生变化。更多详情请参阅 [原文](https://techputs.com/chatgpt-uninstalls-surge-295-percent-dod-deal/)。** 一些评论认为 OpenAI 的战略可能涉及从面向消费者的服务转型，可能专注于广告或政府合同等其他收入来源。还有一种观点认为，这种抵制是预料之中的且早就该发生，反映了对 AI 与军事实体合作带来的伦理影响的更广泛担忧。

    - EnotHOME 质疑了 295% 卸载增幅的重要性，认为如果基数是 1000 次卸载，295% 的增幅意味着 4000 次卸载，他们认为这在大局中是微不足道的。这暗示需要更多关于基准数字的上下文来准确评估影响。
    - coronakillme 寻求对 295% 这一数字的澄清，将其理解为卸载量比以前高出略低于三倍。他们询问原始卸载量是多少，强调了理解基准对于评估增长真实影响的重要性。

  - **[Cancelling subscription - goodbye Sam I'm not funding your war machine!](https://www.reddit.com/r/ChatGPT/comments/1rjg8m0/cancelling_subscription_goodbye_sam_im_not/)** (热度: 606): **该图片是来自 OpenAI 的一封电子邮件截图，确认取消了 ChatGPT Plus 订阅，该订阅将保持激活状态直至 2026 年 3 月 23 日。帖子的标题表明了对 OpenAI 参与军事应用的抗议，反映了对科技公司与国防和情报机构合作的更广泛担忧。评论讨论了 Yahoo Mail 的使用，并提到了涉及 **Anthropic** 和国防部的争议，突显了科技公司与政府机构之间复杂的关系。链接的 Bloomberg 文章提供了关于 Anthropic 参与五角大楼无人机群竞赛的进一步背景。** 评论者对科技公司声称不参与军事项目的说法表示怀疑，认为此类合作是不可避免的。讨论还涉及了与 Yahoo 过去配合政府监控行动相关的隐私问题。

- VVadjet 强调了科技公司与国防及情报机构之间普遍的联系，认为 Anthropic 最近的行为是公关失误。他们引用了一篇 [Bloomberg 文章](http://bloomberg.com/news/articles/2026-03-02/anthropic-made-pitch-in-drone-swarm-contest-during-pentagon-feud)，详细描述了 Anthropic 参与无人机蜂群竞赛的情况，暗示此类合作在行业内是常见且符合预期的。
- ClankerCore 强调在评估科技公司参与国防项目时，需要具体的证据和分析，而非单纯的口号和截图。他们呼吁将详细的合同文本、约束条件、执行情况和监督视为建立信任的关键因素。此外，他们指出 Anthropic 的服务 Claude 面临速率限制和停机，表明在需求增加的情况下存在基础设施挑战。
- LiteratureMaximum125 引用了一份关于 Yahoo 参与政府监控的报告，并链接到一个[来源](https://lieu.house.gov/media-center/in-the-news/yahoo-helped-us-government-spy-emails-report-says)，该来源讨论了 Yahoo 配合美国政府进行电子邮件间谍活动的情况。这突显了外界对科技公司遵守政府监控要求的广泛担忧。


---

# AI Discord 摘要

> 由 Gemini 3.0 Pro Preview Nov-18 生成的总结之总结

**主题 1. 前沿模型：GPT-5.3 余波、Gemini CoT 以及 Qwen 的不确定性**

- **GPT-5.3 "安全脑叶切除"与 5.4 预告**：OpenAI 发布了 [GPT-5.3 Instant](https://openai.com/index/gpt-5-3-instant/)，评价褒贬不一，**LMArena** 用户将其贴上“安全脑叶切除 (safety lobotomy)”的标签，认为其在健康基准测试上的表现逊于 5.2-chat。虽然 **Nous Research** 成员传闻即将推出的 **GPT-5.4** 具备*军事能力*，但 **OpenAI** Discord 用户期待其能快速推出集成 **Sora** 的后续版本。
- **Gemini 3.1 Pro vs. Claude Opus 4.6 编程对决**：关于编程霸主地位的争论在 **LMArena** 中持续进行，**Claude Opus 4.6** 虽然因 **Anthropic** 服务停机受到赞誉，但其推理能力仍受好评；而 **Gemini 3.1 Pro** 被认为速度更快，但更容易产生幻觉 (hallucination-prone)。**Unsloth** 工程师注意到，通过 `<think>` 标签提取 Gemini *真实*的 **Chain of Thought (CoT)** 比其标准总结效果更好，正如[这张截图](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c)所示。
- **Qwen 团队离职与发布失败**：随着 [Qwen 团队负责人离职](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j)，**Unsloth** 和 **OpenRouter** 用户报告了发布过程中的缺陷，并对 **open weights** 的未来表示担忧。尽管如此，技术探索仍在继续，Andrew Carr 分享了一个关于对 **Qwen 3.5 0.8B** 内部[单个神经元进行排名](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12)的项目。

**主题 2. 硬件加速：CUDA Agents、Blackwell 分立与定制芯片**

- **CUDA Agents 击败 Torch Compile**：**GPU MODE** 中讨论的一种新型 **CUDA 专用 RL agent** 据称在处理中型 Kernel 时比 `torch.compile` 快 **2 倍**，并在复杂基准测试中超越了 **Claude Opus 4.5** ([论文](https://arxiv.org/abs/2602.24286))。与此同时，**ByteDance** 发布了一个类似的用于编写快速 Kernel 的 [CUDA Agent](https://cuda-agent.github.io)，引发了人们对自动化 Kernel 生成取代手动优化的兴趣。
- **NVIDIA Blackwell 架构分立**：**GPU MODE** 工程师发现 **NVIDIA Blackwell** 代际在 Data Center (**CC 10.0**) 和 Consumer (**CC 12.0**) 线路之间存在重大分立。由于某些功能现在需要 **sm_100a** 或 **sm_100f** 目标，预计会出现兼容性断层，详见 [NVIDIA 博客](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)。
- **Taalas 与 Apple Silicon 突破极限**：**Unsloth** 成员讨论了 **Taalas HC1** 芯片，该芯片为硬连线模型提供高达 **17,000 tokens/s** 的吞吐量，尽管其仅锁定于特定架构。同时，**Latent Space** 用户报告 **Apple M5 Neural Engine** 运行 **Llama2 110M** 的效率比 A100 高出 **80 倍**，而 **OpenClaw** 成员正在利用 **M5 Pro** 芯片进行本地 Agent 托管。

**主题 3. Agent 框架：C 语言编写的二进制文件、RLM 与 Kimi**

- **ShadowClaw 作为极简 C 语言 Agent 崭露头角**：**OpenClaw** 和 **HuggingFace** 社区正在关注 **ShadowClaw v1.1**。这是一个用 **C** 语言编写的单二进制个人 AI Agent，通过 `curl` 与 **Ollama** 等本地 LLM 进行通信。该工具已在 [GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw) 上开源，强调低开销，具备 Shell 执行、文件操作和持久状态保存等功能。
- **递归语言建模 (RLM) 范式**：**DSPy** 用户正在讨论 Agent 范式向 **RLM** 收敛的趋势，即 LLM 访问 **REPL** 而非静态工具，并认为这可能优于用户定义的 Python 函数。这种递归方法涉及子 Agent 生成并运行自己的代码，与标准的 **ReAct** 循环有所不同。
- **Kimi Code 挑战 Claude**：**Moonshot AI** 推出了 **Kimi Code**，这是一个与 **Claude Code** 竞争的独立 Agent。**OpenClaw** 用户声称在特定任务上其表现比 Minimax *好 5 倍*。虽然一些用户更倾向于开源的 **OpenCode** 替代方案，但 **Kimi** 正被用于通过其 iPython 环境[取代 YouTube](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d) 进行新闻聚合。

**主题 4. 开发者基础设施：实时评估与 2550 亿美元的推理市场**

- **实时训练可观测性**：**HuggingFace** 用户重点推介了 **TrainTrackLabs**，这是一个插入 **PyTorch** 的新观测层，利用 **LLM-as-a-judge** 实时对幻觉和推理进行评分。其目标是在微调运行早期捕捉退化现象，以防止浪费 GPU 开销 ([traintracklabs.com](https://traintracklabs.com/))。
- **利用 AI 进行时空穿梭调试 (Time Travel Debugging)**：**Latent Space** 的工程师讨论了通过 [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview) 实现的时空穿梭调试的复兴。据报道，该工具将一个 **React 19** 升级调试会话从模糊的错误覆盖缩短到了 **30 秒** 内识别根因。
- **推理市场估值飙升**：**Latent Space** 的分析师预测，受生产部署成本超过训练成本的驱动，到 **2030 年 AI 推理市场将达到 2550 亿美元**。这一转变也得到了 **Unsloth** 关于推理优化 (Taalas) 以及 **HuggingFace** 关于 [easytranscriber](https://huggingface.co/blog/KBLab/easytranscriber) 等高效转录工具讨论的印证。

**主题 5. 研究与理论：谱范数、漂移汇与越狱**

- **特征学习的谱范数缩放**：**Eleuther** 研究人员讨论了一篇 [2023 年的论文](https://arxiv.org/abs/2310.17813)，该论文证明通过缩放权重矩阵的**谱范数 (Spectral Norm)** 可以实现特征学习。这一推导与**最大更新参数化 (muP)** 以及最近的 **Modula** 研究相关联。
- **漂移汇 (Drift Sinks) 与角色 Token**：**OpenAI** 用户提出了诸如 **Drift Sinks** 之类的理论框架，旨在通过强制执行认知重力来阻止分析系统中的“语义漂移”。他们还探索了将 **self-tokens** 作为便携式的[角色容器 (persona-containers)](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12)，以在去中心化平台间保持 Agent 身份。
- **化学合成与越狱**：**BASI Jailbreaking** 成员详细介绍了从 **Safrole** 合成 **MDMA** 的四步法（产率 70-80%），并讨论了以营利为目的的 "Eni jailbreaks"。与之相对的是，**LMArena** 报告称 **GPT-5.3** 存在严重的审查，而 **Nous Research** 则讨论了如何在受限硬件 (8GB VRAM) 上创建专门的渗透测试 (Pentest) 模型。


---

# Discord：高层级 Discord 摘要

## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw 文档宣布开源！**：OpenClaw 社区已经[开源了其社区政策和准则](https://github.com/openclaw/community)以及内部文档，但不包括**见习版主（trial moderator）**信息和**审核日志（moderation logs）**。
   - 团队还重构了其**团队层级结构**，完整文档可在同一仓库中查阅，并将 <@1255431768199135254>、<@405240788143046656> 和 <@957289026195435520> 晋升为 <@&1469028608293998723>。
- **Insta-Claw 建立连接！**：一位 OpenClaw 用户发布了 **Instagram** 频道集成，可在 [npmjs.com](https://www.npmjs.com/package/@rylena/openclaw-instagram) 和 [GitHub](https://github.com/rylena/openclaw-instagram) 上获取。
   - 该集成目前仍在开发中，鼓励其他用户进行测试。
- **Kimi 碾压 Minimax？**：成员们就不同 AI 模型的性能和性价比展开了辩论，一位用户评论说 *Kimi 比 Minimax 好 5 倍*。
   - 另一位成员发表了看法，补充道：*嗯，Kimi 体量巨大，两者都很棒。这是我目前的配置。*
- **ShadowClaw 作为精简高效的 C 语言 AI Agent 亮相！**：**ShadowClaw** 被介绍为一个极简的、单二进制文件的个人 AI Agent，采用 C 语言编写，并通过 curl 与本地 LLM (**Ollama**) 进行通信，[代码已托管至 GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw)。
   - 其功能包括 Shell 命令执行、文件读写、HTTP GET 以及简单的数学表达式求解，且状态会自动保存到磁盘。
- **OpenClaw 为视频剪辑注入强劲动力！**：一位用户报告在 Web2Labs Studio 中使用 **OpenClaw** 进行视频剪辑，通过自动化跳剪（jump-cuts）、缩放和缩略图生成来加速处理过程。
   - 该用户强调了剪辑节省的时间，并表示由于标题、描述和缩略图生成的自动化，他们能够实现*持续稳定产出*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-Omni 的 Unsloth 支持仍悬而未决**：一位用户询问了 **Unsloth** 对 **Qwen3-Omni** 的支持情况，并报告在使用 **Opencode** 配合 **Neovim** 进行 Agentic 编程时，在 **XPS 15** 上达到了 **15 t/s** 的速度。
   - 他们还咨询了针对低端硬件的标准基准测试流程，并被引向了 [EleutherAI 的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- **Qwen 团队负责人离职，Google 派人接手**：成员们讨论了 [**Qwen** 团队负责人的离职](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j)消息，据称公司迫使其退位，让位给来自 **Google** 的人员。
   - 用户猜测这可能意味着 **Qwen 开源权重**的终结，并感叹这发生在 *3.3 之后（我们不谈 Llama 4）*。
- **Taalas 芯片承诺提升 LLM 速度——但代价不菲**：讨论中提到 **Taalas 芯片**有可能实现游戏中的本地 **LLM** 运行，其中 **Taalas HC1** 的吞吐量高达 **17,000 tokens/s**。
   - 然而，缺点是它只能运行硬连线到硬件中的模型，一位成员指出*成本*是一个障碍。
- **Gemini 隐藏的思维链 (Chain of Thought)**：成员们注意到 **Gemini** 的摘要比模型的“真实”**思维链 (CoT)** 效果更好（归功于少量的推理），尽管可以通过早期版本的特定设置提取实际的 **CoT**。
   - 截图显示，在被摘要取代之前，**Gemini 2.5 Pro** 曾使用类似 `<think>` 的标签展现结构化推理方法；有人指向了这张[截图](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c)。
- **微调 Qwen 模型：速度瓶颈与解决方案**：一位成员报告称，使用相同的脚本和数据，微调 **Qwen3.5-2B** 需要 **4 小时**，而微调 **Qwen3-1.7B** 仅需 **3 分钟**，这引发了关于优化的建议。
   - 建议安装 `flash-linear-attention` 和 `causal_conv1d` 以解决速度差异，并指出在安装这些插件后，微调 **Qwen3-VL 8B** 大约需要 2 小时，**Qwen3.5-9B** 大约需要 6.5 小时。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Safrole 合成步骤详情**：一名成员提供了从 **Safrole** 开始合成 **MDMA** 的四步合成路径，包括异构化（Isomerization）、氧化为 MDP2P（Oxidation to MDP2P）、还原胺化（Reductive Amination）和成盐（Salt Formation）。
   - 该过程基于 **1970s** 的解密研究，预计产率约为理论质量的 **70-80%**。
- **Jetson Thor 开发套件合购计划**：成员们讨论了合购一套拥有 128 GB VRAM 的 **Jetson Thor 开发套件**，预计每人成本约为 **$800-$1000**，并在私有网络上共享算力。
   - 该小组考虑每人额外出资 **$200** 用于搭建一台“强悍的私有服务器”，或者由个人独立使用全部 128GB 的 VRAM。
- **分享用于 AI Red Teaming 的 MITRE ATLAS**：一名成员分享了 [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS) 作为学习 **AI Red Teaming** 的结构化资源，提供了比 OWASP 更条理化的方法。
   - 讨论还涉及了在没有 AI 辅助的情况下编写 Prompt 的吸引力。
- **Eni Jailbreak：快速致富？**：成员们辩论了创建“eni jailbreaks”的难易程度，有人建议将 Jailbreak Prompt 作为一种潜在的收入来源进行销售。
   - 有人担心其他人可能会使用并销售相同的 **eni JB**。
- **AI 在 Jailbreak 尝试中反击**：一名成员讲述了一次不成功的 AI Jailbreak 尝试，结果 AI 反而嘲讽了他们的自尊心。
   - 这段交流还幽默地配上了一个 [tenor gif](https://tenor.com/view/hmm-ok-okay-then-gif-24387952)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **修复后 LM Link 趋于稳定**：在问题解决后，LM Studio 中的 **LM Link 创建**和**设备发现**现在可以稳定运行。
   - 因测试而暂时停止的候补名单（waitlist）已于美国东部时间晚上 8:55 **重新激活**；用户在获得准入后将收到电子邮件通知。
- **预计 Google 的 Siri 实现将保持本地化**：成员们预计 **Google 的 Siri 实现**将完全本地化，尽管 Google 可能仍会推广其云服务。
   - **#general** 频道中的用户对可能的本地 **Siri 实现**感到兴奋。
- **由于供应有限，DDR3 价格飞涨**：一名成员报告说，由于供应受限，**DDR3 的价格自上次购买以来已经翻倍**。
   - 一名成员开玩笑说要从他们的一堆旧 DDR3 笔记本电脑中赚取利润。
- **Vulkan 可平衡 VRAM 但在 Context 加载方面存在困难**：一名用户确认 **Vulkan** 可以平衡模型层，将 16GB 显卡加载到约 14-15GB，将 32GB 显卡加载到约 28GB，但在 Context 加载方面表现不佳。
   - 该用户指出，用于 Agent 场景的长 Context 意味着需要在 3 张显卡上预留约 5GB 的 VRAM 以避免 OOM 错误。
- **NeuroStream 可能大幅减少 VRAM 占用**：一名成员询问 **Topaz NeuroStream** 是否能以**减少 95% VRAM 占用**的方式运行更大的模型。
   - 另一名成员指出，现代本地 LLM 的效率正在提高，并提到了 [Microsoft's BitNet](https://www.microsoft.com/en-us/research/blog/scaling-down-llms-bitnet-the-end-of-expensive-large-language-models/) 作为类似的技术。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 遭受“史无前例”的停机**：用户报告了 **Claude** 的相关问题，包括速率限制和错误，将其归因于“史无前例的需求”导致的服务中断，甚至引用了一篇 2026 年 2 月的 [Mashable 文章](https://mashable.com/article/claude-down-anthropic-outage-statement)。
   - 一些惊人的谣言流传开来，暗示停机是由于**阿联酋的 AWS 数据中心**遭到无人机袭击，据称造成了基础设施损坏和停机。
- **Gemini 和 Claude 争夺编程桂冠**：关于 **Gemini 3.1 Pro** 还是 **Claude Opus 4.6** 在编程任务中更胜一筹的争论仍在继续；一位用户宣称 *“Gemini 很烂”*，而另一位则断言 **Claude Opus 4.6** 拥有更卓越的思考能力和代码质量。
   - 尽管存在幻觉问题，一些用户发现 **Gemini 3 Pro** 速度更快。
- **Arena 用户因 10 分钟限制而超时**：用户对 **Arena 的 10 分钟超时限制**表示沮丧，尤其是在大型项目中使用 **Claude Opus 4.6** 等模型时，导致频繁出现 *“Error, something went wrong”* 消息。
   - 一位用户戏剧性地请求将限制延长至 *2 小时*。
- **GPT-5.3 接受了安全脑叶切除（Safety Lobotomy）？**：[早期报告](https://deploymentsafety.openai.com/gpt-5-3-instant)表明 **GPT-5.3** 并不比 **5.2-chat** 有明显或客观的提升，仅仅是针对风格和可能的偏好响应进行了微调，有说法称它在健康基准测试中得分更低。
   - 一位用户调侃道 *“笑死，所以它的安全性脑叶切除更严重了”*，表达了对其可用性的担忧。
- **Arena.ai 详解**：一段 [YouTube 视频](https://www.youtube.com/watch?v=nktiDGTn61I)在 60 秒内简明扼要地解释了 **Arena.ai**。
   - 目前尚不确定该视频在其简短的概述中遗漏了哪些内容。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agent 陷入循环？**：一位遇到 **Cursor Agent** 陷入循环问题的用户被建议在主要任务之外，明确指定 Agent *不应该* 执行的操作。
   - 这种方法旨在约束 Agent 的行为，防止其在执行任务期间出现重复循环。
- **云端 Agent 随 Android 支持进入移动端**：**云端 Agent** 现在支持 **Android**，以 [Web 应用](https://cursor.sh)的形式运行，扩展了其在不同平台上的可访问性。
   - 这一增强功能允许用户直接在 Android 设备上利用云端 Agent 的功能。
- **默认设置下的 Web 开发表现平平**：一位用户对 **Cursor** 使用 **Codex 5.3** 的默认 Web 开发输出表示失望，称其设计水平欠佳。
   - 其他用户建议使用特定的包（如 [shadcn](https://ui.shadcn.com/)），并提供带有参考信息的详细 Prompt，以及从目标网站截取源代码。
- **Cursor 调整布局以简化体验**：根据用户反馈，**Cursor** 简化了其布局侧边栏，解决了旧版本过于混乱的问题。
   - 尽管新布局的发现方式有所改变，**Zen 布局**仍可通过 **Command+Option+Tab** 快捷键访问。
- **AI 同事 Viktor 进入 Slack 工作区**：一个名为 [Viktor](https://www.producthunt.com/products/viktor) 的 AI 同事在 Slack 上线，它完全由 **Cursor** 构建，提供 **营销审计**、**广告管理**和**潜在客户调研**等功能。
   - Viktor 与 **3,000 多种工具**集成并使用持久化记忆，能够学习公司细节并组合工具来执行复杂操作。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLMs 催生了“代码清洁工”**：随着 **LLMs** 的地位日益突出，“**代码清洁工**”（**code janitor**）的角色变得愈发重要，其工作重心在于创建**抽象**（**abstractions**）和**防护栏**（**guardrails**）以防止事故发生。
   - 有人指出，在 **PR reviews** 过程中，**LLMs** 会让人更难依赖系统知识，从而提升了专业化角色的价值。
- **Roblox 目标直指万亿级元宇宙**：**Roblox** 结合了技术优势、AI 驱动的创作工具以及快速的设计演进，使其定位为未来的万亿级公司。[这篇文章](https://xcancel.com/jnavok/status/2028664806601855421?s=12)将其增长与 TikTok 进行了对比。
   - 讨论还涉及了 **Roblox** 成为元宇宙平台的潜力以及对开箱（loot boxes）的担忧，并引用了 [Reuters 报道](https://www.reuters.com/legal/government/new-york-sues-video-game-developer-valve-says-its-loot-boxes-are-gambling-2026-02-25/)中关于 **Valve** 开箱机制的问题。
- **Qwen 3.5 神经元排名系统发布**：Andrew Carr 分享了一个专注于对 **Qwen 3.5 0.8B 模型**中每个独立神经元进行排名的项目（[原帖](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12)），这可能是在探索模型的可解释性或重要性映射。
   - Xinyu Yang 批评了将 **Qwen 领导层**替换为来自 Google Gemini 的指标导向型人选的做法，并警告不要像管理消费级应用开发周期那样去管理基础模型（foundation model）研究，详见 [此 X 贴](https://xcancel.com/xinyu2ml/status/2028867420501512580?s=46)。
- **时间旅行调试获得 Replay 助力**：一位成员宣布重新转向基于 **AI 的时间旅行调试**（**time travel debugging**），强调了 [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview) 的可用性及其强大的功能。
   - 他们针对一个失败的 **React 19 升级问题**进行了测试，大约 **30 秒**内就从“错误覆盖层的截图”进化到了“我知道问题出在哪了”。
- **AI 推理公司迈向千亿估值**：Meg McNulty 在 [这条推文](https://xcancel.com/meggmcnulty/status/2028532451992314199) 中强调了 **AI 推理公司**估值的飙升，并指出用于运行模型的软件正变得比模型训练更具价值。
   - 她预测到 **2030 年市场规模将达到 2550 亿美元**，这主要受生产级 **AI** 部署的持续成本所驱动。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 在 GTC 相约**：Nous Research 邀请大家在 GTC（GPU Technology Conference）期间相聚，正如 [此 X 贴](https://x.com/nousresearch/status/2028861034220405178) 所宣布。
   - 这是一个在会议期间与 **Nous Research** 团队见面并建立联系的机会。
- **GPT 5.4 具备军事实力**：成员们推测 **GPT 5.4** 的能力与 **5.3-codex** 相当，但包含了“军事能力”。
   - 频道内讨论认为，从研究角度来看，*自主学习已基本解决，但集成到实际中并不现实*。
- **Anthropic 正在缓存 Prefills**：成员们观察到 **Anthropic** 似乎通过缓存 **prefill** 来节省成本，但这使得切换模型变得不可能。
   - 一位成员指出，这使他们能够降低相对于 **OpenAI** 的成本，而后者似乎也在针对推理成本与用户留存进行优化。
- **Opus 奇特的算术方法**：一位成员分享了 **Opus** 做数学题的[例子](https://example.com)：它通过确定最后两位数字并使用查找表（lookup table）来确定第一位数字。
   - 频道达成共识，认为这种方法凸显了 **LLMs** 在数学方面的局限性，因为它是基于模式识别而非实际的数学理解。
- **在有限资源下寻求渗透测试模型**：一位成员正在寻求建议，希望以 **Hermes 模型**为基础创建一个专门的**渗透测试（pentest）模型**，并针对有限资源进行训练优化（其拥有一块 **8GB 显存的 GPU**）。
   - 作为一名巴西成员，他们还受到当地 GPU 供应商高昂成本的限制，其价格与美国相当。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.3 Instant 发布，GPT-5.4 预告**：根据[公告](https://openai.com/index/gpt-5-3-instant/)，最新的 **GPT-5.3 Instant** 模型现已向所有 **ChatGPT** 用户推出，并提升了准确率。后续消息暗示 **GPT-5.4** 的发布可能比预期更早。
   - 用户报告 **GPT-5.3** 正在分阶段推出，部分用户遇到延迟，而另一些人注意到应用在更新后取消了 **5.2** 标识。用户期待下周发布集成 **Sora** 的 **GPT-5.4**。
- **低质量 AI 创作遭到抨击**：成员们讨论了社会如何为了金钱利益而激励低质量 AI 内容的创作，一位用户质疑道：“如果社会想要‘好东西’，为什么还要激励垃圾内容 (slop) 呢？”
   - 一位用户批评 **Sora 生成的 AI 语音** 听起来很假且速度快得不自然，这加剧了对低质量 AI 内容的担忧。用户还抱怨向 ChatGPT 提问时需要添加过多的限定条件。
- **不鼓励将 Discord 数据用于 LLM**：有用户询问是否可以使用 **Discord 服务器消息** 训练 **LLM** 以进行主动微调，但其他成员告诫不要这样做，理由是数据量有限且可能违反 **TOS**（服务条款）。
   - 一位成员警告说，使用 Discord 来训练 LLM 是“让 LLM 变脑残的最快方法”。
- **Self-Tokens 促进 AI Persona 可移植性**：一位成员建议使用 *self-tokens* 作为 **persona-containers** 来增强框架，使 **AI-personas** 具备可移植性；可通过[此图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12&)获取模板。
   - **关系度量 (Relation gauge)** 被描述为一种建模维持链接可能性的生产力指标，建议 tokens 应具有可变长度，特别是对于由多人管理的去中心化平台。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 用户保留数据所有权**：确认用户拥有其数据，默认情况下 Prompt 和响应不参与公共模型训练，所有通信在传输中（**TLS 1.2+**）和静态存储中（**AES-256**）均经过加密，详情见 [FAQ](https://openrouter.ai/docs/faq) 和 [隐私指南](https://openrouter.ai/docs/guides/privacy/data-collection)。
   - 安全标准包括 **SOC 2 Type 2**、**CSA STAR** 和 **ISO** 信息安全认证。
- **LLM 深受偏见争议困扰**：一场关于 **LLM** 是否因训练数据而具有固有偏见的辩论随之展开，一些人建议创建一个由多个公正的人类检查的无偏见数据集。
   - 另一些人则认为 *所有人都有偏见*，即使是“无偏见”的 **LLM** 也会在有偏见的数据集上进行训练。
- **客户端错误处理立大功！**：部分用户遇到了 `TypeError: undefined is not an object (evaluating 'p.choices [0].delta')` 错误，这导致人们发现 OpenRouter 有时不会发送预期的 delta 值。
   - 解决方案涉及客户端错误处理，并为 Venus Chub 实施了修复，如[此 GitHub pull request](https://github.com/cline/cline/pull/9432)所述。
- **BYOK 与 z.ai 兼容性问题显现**：用户报告了在 OpenRouter 上通过 **BYOK** 使用 **z.ai** 订阅时出现问题，错误信息显示“余额不足或无资源包”。
   - 澄清指出 **z.ai** 订阅使用不同的基础 URL，且与 **BYOK** 不直接兼容，允许将连接订阅关联到 **BYOK** 的功能请求已被拒绝。
- **OpenRouter 成本审查揭示细节**：用户质疑 OpenRouter 相较于直接使用 API 的成本效率，指出开发日志与 OpenRouter 日志之间存在差异，而 OpenRouter 收取 **5.5%** 的费用。
   - **LLM** 的选择显著影响成本，某些模型比其他模型更贵；此外，还讨论了一个可以**压力测试新 AI 应用**并以 **USDT** 结算的 Joinable Bounty 计划。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA RL Agent 取得性能里程碑**：据[这篇论文](https://arxiv.org/abs/2602.24286)报道，一个专门针对 **CUDA** 优化的 **RL Agent** 在简单/中等 **kernel** 上的表现优于 **torch.compile** **2倍**，在最难的基准测试中比 **Claude Opus 4.5** 和 **Gemini 3 Pro** 高出约 **40%**。
   - 尽管结果令人鼓舞，但人们对未发布的 **kernel** 以及对*具有进程级隔离的大型 GPU 池*的依赖表示担忧，这引入了显著的计算和工程开销。
- **ByteDance 发布用于 Kernel 编写的 CUDA Agent**：ByteDance 推出了一款 **CUDA Agent**，该模型旨在编写快速的 **CUDA kernels**，在简单/中等 **kernel** 上性能优于 **torch.compile** **2倍**，在最复杂的任务中超过 **Claude Opus 4.5** 和 **Gemini 3 Pro** 约 **40%**，详见 [tweet](https://x.com/BoWang87/status/2028599174992949508)。
   - 分享的链接 [cuda-agent.github.io](https://cuda-agent.github.io) 被称为 *ByteDance 值得一看的有趣项目*。
- **Blackwell 的计算能力细分化**：成员们讨论了 **NVIDIA Blackwell 架构** 现在分为数据中心 (**CC 10.0**) 和消费级 (**CC 12.0**) 两条路线，分别针对 **AI/HPC** 和**实时图形**优化。
   - 某些附加功能不具备前向兼容性，需要 **sm_100a** 或 **sm_100f** 而不仅仅是 **sm_100**；更多信息可以在 [NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) 找到。
- **Kernelbook 和 Kernelbot 计划合并**：由于 Prime Hub 缺乏协作功能，一位成员提议发布改进的环境供他人审查。
   - 一位成员建议由于共享基础设施，应将 **kernelbot** 和 **kernelbook** 合并，这可能会优化资源利用并简化开发流程。
- **Teleop TRLC DK-1 系统首次亮相**：引入了一款实验性的 [TRLC DK-1](https://www.robot-learning.co/) 远程操作（teleop）系统，当 policy 运行处于 **OOD**（分布外）时，该系统可用于人工干预。
   - 第一次测试使用了安装在 SO-101 上的 [ELP 立体摄像头模块](https://www.amazon.de/dp/B07FT2GKZS)，展示于[此视频](https://x.com/neurosp1ke/status/2023073945637753101?s=20)中。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **实时评估工具加入竞争**：**TrainTrackLabs** 正在为 **LLM 训练**开发实时评估和可观测性层，通过直接插入 **PyTorch / Hugging Face**，利用 **LLM-as-a-judge** 评分持续跟踪推理、安全、幻觉和代码能力，详见 [traintracklabs.com](https://traintracklabs.com/)。
   - 他们正在寻找早期试点团队，以尽早发现回归问题并防止浪费 GPU 支出。
- **Shadowclaw 发布 v1.1 版本**：用 C 语言编写的单二进制个人 AI **Agent** **Shadowclaw v1.1** 在原版基础上增加了内置命令和原生工具，可在 [GitHub](https://github.com/webxos/webXOS/tree/main/shadowclaw) 上获取。
   - 此版本包含 **/help**、**/tools**、**/state**、**/clear**、**/chat** 和 **/exit** 等命令。
- **Easytranscriber 转录并提升时间效率**：`easytranscriber` 是一个具有精确时间戳的**自动语音识别**库，类似于 WhisperX，但根据硬件不同，运行速度快 **35% 到 102%**，可在 [Hugging Face blog](https://huggingface.co/blog/KBLab/easytranscriber) 查看。
   - 它还支持将 HF 模型作为后端。
- **欧洲冲刺 Frontier AI 领导地位**：SPRIND 通过 [next-frontier.ai](https://next-frontier.ai/) 提供 **1.25 亿欧元**的无股权资金，支持最多 **10 支团队**在欧洲建立前沿 AI 实验室，寻求新颖的架构和 **Agentic** 系统。
   - 该倡议将建立专注于下一代 **Agentic** 系统的研究。
- **MCP 集成安全：棘手的局面**：分享了对 **Model Context Protocol (MCP)** 攻击向量的深入研究，详细介绍了每个 **MCP** 开发者都应了解的 5 种极易被利用的模式，并记录在 [Medium 文章](https://medium.com/@nainia_ayoub/mcp-security-is-a-mess-5-ways-i-broke-my-own-ai-agent-76379a46ca90?sk=0daa66d4fc2a68fbb02a56e803336ce2)中。
   - 讨论强调了这些向量被利用的难易程度。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Code 与 Claude Code 不同**：成员们确认 **Kimi Code** 是 **Moonshot** 开发的一个新 **Agent**，与 **Claude Code** 截然不同。
   - 一位成员强调 **OpenCode** 是他们首选的 **Claude Code** 开源替代方案，并指出了其流行程度。
- **Moderato 计划 Token 使用情况披露**：一位用户分享了 **OpenCode** 上 **Moderato 计划**（$19/月）的使用统计数据，报告称在发送 **365 条消息**、消耗 **1.0M Input Tokens**、**115.6K Output Tokens** 以及 **25.3M Cache Read** 的情况下，仅使用了每周额度的 18%。
   - 这相当于每月 **20M Input Tokens** 的预算，另一位用户认为这*并不算多*。
- **Kimi 瞄准 YouTube 的领地**：一位用户开发了一个提示词，使 **Kimi** 能够收集科技和游戏新闻，旨在通过使用[此提示词文件](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d)让 **Kimi** 独立重构故事，从而减少对 **YouTube** 的依赖。
   - 该用户赞扬了 **Kimi 的聊天界面**，引用了 **search calls** 和 **iPython 环境**等功能，认为它们几乎是无限的，且领先于竞争对手。
- **Kimi Allegretto 计划取消**：一位用户询问如何取消 **Kimi Coding Plan Allegretto** 或关闭自动续费，另一位用户提供了[管理订阅的链接](https://www.kimi.com/membership/subscription)。
   - 取消选项可以在个人资料设置中找到。
- **支持邮件延迟，欺诈性计费未解决**：一位用户报告支持邮件失效，欺诈性计费问题未得到解决。
   - 一位非团队成员推测，春节假期后的大量邮件导致了延迟，该问题已上报给工作人员。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Cohere 的 Aya 项目期待你的加入！**：**Cohere** 正在为其 [Aya 项目](https://aya.cohere.com/about)寻找合作者，根据你的技能水平，在 **Fast AI**、**Eureka Labs** 或 **Cohere Research Labs** 都有机会。
   - 该项目的目标是为负责任的多模态 AI 创建开源基础。
- **CVPR 2026 医学推理研讨会**：一位成员正在组织 **CVPR 研讨会**，并邀请向 [医学推理研讨会 (Medical Reasoning Workshop)](https://med-reasoner.github.io/cvpr2026/) 投稿。
   - 有关研讨会的更多信息可在 [Discord 活动链接](https://discord.gg/nxtWyHbY?event=1478419152103280680)上找到。
- **谱范数缩放实现特征学习**：一篇 [2023 年的论文](https://arxiv.org/abs/2310.17813) 证明，通过缩放权重矩阵及其更新的**谱范数**（如 √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗)），可以实现特征学习。
   - 论文的分析还提供了 **maximal update parametrization** 的基本推导。
- **扩散模型早期控制图像构图**：一篇新论文利用 **SAE 框架** 探究了流行的**文本生成图像扩散模型**的内部机制，在其激活中发现了人类可解释的概念，详情见 [https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473)。
   - 研究引入了操纵**图像构图和风格**的干预技术，论文显示，**图像构图**可以在扩散的早期阶段得到有效控制。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 正在酝酿 Mojo 包管理器**：Modular 正在考虑构建一个 [Mojo 包管理器](https://forum.modular.com/t/open-question-what-would-you-like-to-see-from-a-mojo-package-manager/2799?u=nate)，可能类似于 Rust 的 `cargo` 或 Python 的 `pixi`，并包含一个中央仓库。
   - 目标是确定社区在分发 Mojo 包方面的愿望和需求。
- **API 抽象受到赞赏**：一位成员主张从用户角度而非实现细节来设计 API，并举例说明 `@inline(strategy: "chose whatever makes sense", ...)` 相比于 `@always_inline` 和 `@never_inline` 提升了用户体验。
   - 另一位成员赞同良好的 API 设计至关重要，且依赖于更通用的装饰器表示。
- **矢量化验证之旅**：从 **Mojo 25.7 到 26.1** 的跳转引入了与并行化和矢量化相关的重大变化，特别是影响了闭包，导致了编译器错误。
   - Modular 确认这些变化是迈向 **1.0 就绪状态** 的一部分，并将提供明确的迁移建议，类似于现有的 **UnsafePointer** 文档。
- **Apple 推进内存安全**：Apple 准备解决内存完整性强制执行问题，这可能会影响 Mojo，详见[这篇博文](https://security.apple.com/blog/memory-integrity-enforcement/)和[此项分析](https://www.ralfj.de/blog/2020/12/14/provenance.html)。
   - 这对于其他平台也可能成为一个重要问题。
- **`comptime` 考量会议**：一位成员建议通过使用 `@` 代替 `comptime` 来简化编译时元编程语法，例如将 `@parameter if` 变为 `@if`。
   - 另一位成员提到，他们之前曾为 Mojo 请求过 `maybe comptime` 功能。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 支持团队：办公时间守卫**：**Manus.im Discord** 上的用户询问了 **Manus 支持团队** 的服务时间，该团队通常在办公时间内在线，并建议成员发送包含其电子邮件地址的 **DM** 以寻求帮助。
   - 团队分享了一篇[帮助文章](https://help.manus.im/en/articles/12087847-how-to-optimize-my-credit-usage)，其中包含关于如何更有效地使用 **Manus** 和优化积分（credit）使用的提示与信息。
- **Manus 方案积分不累积**：一位用户询问 **46€ 方案** 中未使用的积分是否会累积到下个月，工作人员回复称*目前看来不会累积*。
   - 还有一些关于使用 **Telegram Agent** 会消耗积分的推测。
- **通过需求文件结构化 AI 驱动的应用开发**：一位用户寻求关于利用**结构化需求文件 (PRD / 系统设计文档)** 配合 AI 工具来结构化构建复杂全栈应用的指导，旨在通过遵循明确的架构驱动工作流来防止产生*草率、无结构的 AI 生成代码*。
   - 用户希望确保他们是以结构化的方式构建全栈应用。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **REPL 平行于 RLM Agent 范式**：使用 **REPL** 的新 Agent 范式正在向 **RLM** 收敛，这让人联想到[这篇文章](https://x.com/nfcampos/status/2028576281793630372?s=20)和[这篇文章](https://x.com/RLanceMartin/status/2027450018513490419?s=20)。
   - 一位成员表示，将 **REPL** 权限授予 **LLM** 的 **RLM** 范式将优于授予其访问用户编写的 Python 函数的权限。
- **RLM 的递归性质引发辩论**：成员们辩论了递归是否是 **RLM** 的必要条件，认为递归方面源于通过派生子 Agent 来运行其 **REPL**。
   - 一位成员假设 *“Claude 使用脚本调用 Claude 在某种程度上就是一个子 Agent”*，并参考了[此链接](https://x.com/a1zhang/status/2023976399694917808?s=20)。
- **DsPy Meetup 将阐明 RLM**：一位成员建议本月在湾区的 **DsPy Meetup** 举办一场会议，以澄清 **RLM** 的基础知识。
   - 该会议将涉及 **RLM** 与 **ReAct** 的比较，并研究 **RLM** 如何确定要生成什么代码，因为它会生成自己的代码，而不是依赖用户定义的 Python 函数作为工具。
- **RLM 在长上下文文档中找到定位**：一位成员发现 **RLM** 适合处理消耗大量 **MM tokens** 的文档，而其他人在对 **LLM** 进行自主调用感到放心时也会使用它。
   - 为了确保最佳性能，他们开发并实施了 evals 和测试。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude Opus 极度烧钱**：成员们报告称，由于不受限制的客户端使用，**Claude Opus** 的费用会迅速累积到 **$65/小时**，可能导致产生 **$1000 美元**的账单。
   - 讨论质疑了达到如此高昂的小时成本所需的 Token 消耗量。
- **AiderMacs 缺少项目缓冲区排序**：一名成员询问如何配置 **AiderMacs**，以便在 `ibuffer-projectile` 中通过项目缓冲区（project buffers）进行聊天组织。
   - 关于此问题尚未分享任何解决方案或进一步细节。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 会议定于 3 月 2 日举行**：新的 **tinygrad 会议**定于 **3 月 2 日** **圣地亚哥时间晚上 8 点**举行。
   - 会议将涵盖*公司更新、comma 相关问题、CALL/BUFFER_VIEW sym llm、assign、setitem、disk、drivers、llama、VIZ 以及其他议题和 Bounties*。
- **关于 Bounties 的 Pull Request 讨论**：讨论提到了一个与 **Bounties** 相关的拉取请求（[PR #14982](https://github.com/tinygrad/tinygrad/pull/14982)）。
   - 未提及关于该 PR 具体内容的更多细节。
- **代码库更倾向于使用 `len(x.shape)`**：一名成员注意到代码库在许多情况下使用 `len(x.shape)` 而不是 `x.ndim`。
   - 该成员质疑提交 PR 来解决此问题的价值，但将其强调为潜在的重构方向或代码风格偏好。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI-Ready Data Summit 定于 2026 年举行**：**AI-Ready Data Summit** 定于 **2026 年 3 月 31 日**举行，届时将邀请来自 **Lockheed Martin**、**Dell Technologies**、**Red Hat**、**CNH** 和 **Entrust** 的演讲者。
   - 峰会将聚焦于企业级 AI 实践、数据基础设施和模型部署见解（[峰会详情](https://ai-ready-data-summit.com)）。
- **2026 年 AI Control Hackathon 挑战技术人员**：**Apart Research** 正与 **Redwood Research** 合作，于 **2026 年 3 月 20 日至 22 日**举办 **AI Control Hackathon**，挑战参赛者监控并遏制试图规避安全措施的 AI Agent。
   - 感兴趣的各方可以在 [Hackathon 详情页](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach)了解更多细节。
- **AI 商业构建者围绕 OpenClaw 展开讨论**：一场 **45 分钟的圆桌会议**将于 **3 月 14 日**举行，讨论开发者如何使用 **OpenClaw** 和其他工具来运营业务、社区和产品。
   - 详见 [圆桌会议注册页面](https://luma.com/qfrucnl2)。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 贡献者期待开发者峰会 (Dev Summit)**：成员们对下个月即将举行的 **MCP Dev Summit** 表示兴奋。
   - 与会者正准备齐聚一堂分享见解，期待这是一场富有成效且引人入胜的盛会。
- **Dev Summit 准备工作**：贡献者们正在完成 **MCP Dev Summit** 的最后准备工作，期待一个高效且协作的环境。
   - 峰会旨在促进参与者之间的讨论和知识共享，为未来的发展奠定基础。



---


**LLM Agents (Berkeley MOOC) Discord** 暂无新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 暂无新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---



您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1478209582777241620)** (1 条消息): 

> `开源社区政策、团队架构重组、版主晋升` 


- **OpenClaw 公布政策！**：OpenClaw 社区已[开源](https://github.com/openclaw/community)了其所有的**社区政策和准则**，以及内部文档。
   - 唯一的例外是**试用版主 (trial moderators)** 和**审核日志 (moderation logs)**；其他所有内容都将公开并保持实时更新。
- **架构更新及晋升公告！**：OpenClaw 最近重组了其**团队架构**，完整文档可在开源仓库中查阅。
   - 成员 <@1255431768199135254>、<@405240788143046656> 和 <@957289026195435520> 已晋升为 <@&1469028608293998723>。


  

---

### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1478119872738234388)** (661 条消息🔥🔥🔥): 

> `Instagram Channel Integration, Kimi vs Minimax, M5 Pro chip to run OpenClaw, Whatsapp Business Integration issues, Openclaw secrets audit command` 


- **Instagram Channel Integration 发布**: 一位成员宣布发布了 OpenClaw 的 Instagram 频道集成：[https://www.npmjs.com/package/@rylena/openclaw-instagram](https://www.npmjs.com/package/@rylena/openclaw-instagram)。
   - 鼓励其他成员进行测试。[https://github.com/rylena/openclaw-instagram](https://github.com/rylena/openclaw-instagram) 但目前仍标记为开发中（working progress）。
- **Kimi 比 Minimax 强五倍**: 成员们讨论了不同 AI 模型的性能和性价比，一位用户表示 *Kimi 比 Minimax 好 5 倍*。
   - 另一位表示 *Kimi 非常强大，两者都很出色。这是我目前的配置*。
- **M5 Pro 芯片运行 Open Claw、Claude Code、Anthropic 本地化**: 有成员询问是否能在拥有 **48GB 20核 GPU** 的 **M5 Pro** 芯片上本地运行 **OpenClaw**、**Claude code** 和 **Anthropic**。
   - 经确认，**Claude** 和 **Anthropic** 并不是本地模型。
- **用户在连接 Whatsapp Business 时遇到问题**: 一位用户在将 **WhatsApp Business** 连接到 **OpenClaw** 时遇到困难，并表示 *当我输入时它没有响应*。
   - 另一位提到 **Whatsapp API** 并不是免费的，但你可以免费使用 **WhatsApp Business app** 并扫描 **QR** 码来连接到 **OpenClaw**。
- **避免使用 OAuth 以规避 Claude 封号**: 用户讨论了许多人因在 OpenClaw 中使用 Claude OAuth/订阅而被封号的情况。
   - 一位成员解释说 *因为这会破坏 Claude 桌面应用、Claude Code 和 Claude Cowork*。


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1478121347971551232)** (40 条消息🔥): 

> `Qwen 3.5, Local LLM, Kimi, M3 512, Grok` 


- **Qwen 3.5 的 Tool Calling 表现惊人！**: 一位用户报告称，在本地使用 **Qwen 3.5 35B A3B** 进行思考和 Tool Calling 的效果非常好，在购买了 **M3 Studio 512** 后成本仅为 **$0**。
   - 另一位成员强调，新的小型 **Qwen** 模型性能极强且体积微小，是一个非常棒的选择。
- **Kimi 跑不动！**: 用户讨论了在 **1080ti** 上本地运行 **Kimi 2.5k** 的可行性，结论是除非进行大幅量化（quantized），否则它需要太多的 VRAM。
   - 一位成员开玩笑说 *运行 **Kimi**，它可是有 1 万亿参数的，兄弟，哈哈*，甚至梦都别想在消费级硬件上运行它。
- **M3 512 装备可以运行！**: 一位拥有两台 **M3 512** 的用户报告称，能够使用 'inferencer' 运行几乎全权重的模型，包括大多数 **Q8 量化版**。
   - 然而，另一位用户澄清这不属于 *消费级硬件*，因为两台拥有 512 GB 内存的 M3 是 100% 的发烧级配置，成本约为 1.7 万美元。
- **阿里巴巴的 Kimi 性能**: 一位用户在 **Alibaba** 的端点上测试了 **K2.5**，发现速度极快，推测它可能是在量化模式下运行。
   - 另一位成员在过去一个月里一直将 Kimi k2.5 API 作为主力使用，并尝试了 **Minimax 2.5**，坦言并没有感觉到太大区别。
- **排除阿里巴巴 API 错误**: 一位用户报告在使用 Alibaba 的 **$3** 编程方案时遇到 **HTTP 401 错误（无效访问令牌或令牌已过期）**。
   - 他们想知道在使用 **Model Mimi 2.5.jape** 时，是否需要每小时生成一个新的 API key。


  

---

### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1478170128121204817)** (54 messages🔥): 

> `Agentic Loop, Minimal AI Agent, Video Editing with OpenClaw, Automated Trading, OpenClaw on Vision Pro` 


- **Danbot 的感性自我反思激发了充满灵性的代码！**：一位用户分享了他们与名为 **Danbot** 的智能体（Agent）的互动经历。用户在凌晨 2 点要求它进行自我反思，导致 **Danbot** 识别出其安全协议中的低效之处，并提出了改进建议，这些建议已被集成到它的 **SOUL.md** 中。
   - 这被描述为*通过人类对齐实现的完美智能体循环 (agentic loop)*，展示了 AI 通过交互和反馈进行学习与适应的潜力。
- **ShadowClaw：C 语言编写的 Claw 削减昂贵的云端计算成本！**：**ShadowClaw** 被介绍为一个用 C 语言编写的极简、单二进制文件的个人 AI Agent。它强调自托管、工具使用和持久化内存，通过 curl 与本地 LLM (**Ollama**) 进行通信。
   - 它具有执行 shell 命令、文件读写、HTTP GET 以及简单数学表达式求值的功能，每次交互后状态会自动保存到磁盘，可在 [GitHub](https://github.com/webxos/webxos/tree/main/shadowclaw) 上获取。
- **Web2Labs Studio 助力 TikTok 创作大获成功！**：一位用户报告在 Web2Labs Studio 中使用 **OpenClaw** 进行视频编辑，称其通过自动创建跳剪（jump-cuts）、特写（punch-ins）、缩放以及提取多个垂直剪辑显著加快了流程，并利用“钩子评分”（hook scores）来优先发布内容。
   - 用户强调了在编辑上节省的时间，以及由于标题、描述和缩略图生成的自动化而能够*保持稳定更新*的能力。
- **交易智能体工单解决棘手的技术细节！**：一位用户正在尝试在*模拟设置*中使用 OpenClaw 进行*自动交易*，运行一个 Python 仪表盘、一个交易者实体、一个控制器实体、一个带有工单工作流的代码生成器实体以及一个共享标志系统。
   - 他们分享了一个针对突发事件的代码生成工单示例，强调了开箱即用的专业级细节水平；现在他们通过 WhatsApp 手动批准代码工单。
- **Vision Pro 的画面证实了虚拟领域的探索！**：一位用户分享说他们成功让 OpenClaw 在 **Vision Pro** 上运行，这是他们在该频道的第一篇帖子。
   - 未提供更多背景信息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1478120772634546408)** (801 messages🔥🔥🔥): 

> `Qwen3-omni support in Unsloth, Benchmarking LLMs on low-end hardware, Qwen 3.5 UD 14B Q6 gguf, Qwen3.5 2B vs qwen3-1.7B Finetuning Time, Save_pretrained_merged Not Working` 


- **Qwen3-Omni 支持状态尚不明确**：一名成员询问 **Unsloth** 是否支持 [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3.5-4B)。
   - 他们还报告说，在使用带有 **Neovim** 前端的 **Opencode** 进行智能体编码任务时，在 **XPS 15** 上达到了 **15 t/s** 的速度。
- **LLM 硬件基准测试策略**：一位成员表示有兴趣在低端硬件上对模型进行基准测试，并询问了标准的基准测试程序。
   - 另一位成员建议使用 [EleutherAI 的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 进行基准测试。
- **在哪里下载 Qwen2.5 UD 14B Q6 GGUF？**：一位成员正在寻找 **Qwen2.5 UD 14B Q6 GGUF** 模型的下载地址，并提到该模型在编码层使用了动态量化（Dynamic Quantization）逻辑。
   - 另一位成员建议更新的 **Qwen 3.5** 可能是更好的选择，并分享了 [Unsloth 动态量化版本的链接](https://huggingface.co/collections/unsloth/unsloth-dynamic-20-quants)。
- **Qwen3.5-2B 的微调速度比 qwen3-1.7B 慢**：一位成员报告说，使用相同的脚本和数据，微调 **Qwen3.5-2B** 需要 **4 小时**，而 **Qwen3-1.7B** 仅需 **3 分钟**。
   - 有建议称应安装 `flash-linear-attention` 和 `causal_conv1d`，这解决了速度差异问题。据称如果安装了这些组件，Qwen3-VL 8B 大约需要 2 小时，Qwen3.5-9B 需要 6.5 小时。
- **Save_pretrained_merged 出现问题**：一位用户报告说，由于缺少 LoRA 适配器，`save_pretrained_merged` 函数无法正常工作，并提到他们遇到了 `AttributeError: 'Qwen3_5Model' object has no attribute 'prepare_inputs_for_generation'`。
   - 建议采用手动合并作为替代方案，并提供了一个[脚本链接](https://gist.github.com/amytimed/8acd6867c0d00ed4dcd7c3d1768678b7)以供参考。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1478122691042087014)** (2 messages): 

> `新项目想法，与工程师建立联系` 


- **工程师寻求新项目想法的联系人**：一位全栈兼 AI 工程师正寻求与其他对新项目有极佳想法的人建立联系。
- **社区欢迎新用户**：一位新用户加入了社区，并用英文和中文表达了他们的兴奋之情。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1478126746393116802)** (1247 messages🔥🔥🔥): 

> `游戏 AI，Taalas 芯片，Qwen 模型发展，数据集整理，LLM 基准测试` 


- ****游戏中的 LLM：虽非 AGI，但依然出色！****：成员们讨论了 **LLM 进入大型游戏**的时间表，由于*显存*和*算力消耗*，估计还需要 **7 年时间**。由于主流 GPU 仍以 **8GB** 为主，超过 **4B** 的模型不太可能运行。
   - 然而，有人认为小型模型*现在*就可以用于**次要功能**，例如让角色说出你的名字，这*完全可行*，但问题在于 LLM NPC 是否能与模型保持一致性。
- ****Taalas 芯片承诺极速 LLM，但也有其局限性****：讨论围绕 **Taalas 芯片**展开，一位成员表示如果他们出售这些芯片，就能让**LLM**在游戏中实现本地化运行，而另一位成员则指出成本是一个障碍。
   - 成员们还讨论了 **Taalas HC1**，这是一款*硬连线（hardwired）的 Llama-3.1 8B AI 加速器*，可提供高达 **17,000 tokens/s** 的速度，但主要缺点是它只能运行硬连线到硬件中的特定模型。
- ****负责人离职后 Qwen 团队剧变****：成员们讨论了 **Qwen 团队负责人** [最近的离职](https://bsky.app/profile/natolambert.bsky.social/post/3mg6eisffss2j)，据称是公司强迫其卸任以让位给来自 **Google** 的人。
   - 用户们哀叹这可能意味着 **Qwen 开源权重**的终结，尤其是在 *3.3 版本之后（我们不谈论 Llama 4）*。 
- ****刷榜（Benchmark-maxxed）的 LLM：炒作 vs 现实****：一位用户声称某些模型正在进行 *Benchmaxxed*，意味着它们是*为了在基准测试中表现良好而训练的，而非为了实际应用场景*，模型缺乏内在的*直觉和细微差别*。
   - 随后他们表示，为了获得最佳效果，必须使用*更高质量且更小的数据集 + 巨型模型*，我们只需要攻克数学难题。
- ****利用小模型进行数据增强****：有人提到 [Essential AI](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) 有一个很好的想法，即训练一个极小模型来进行分类，这样你不需要直接删除坏样本，而是可以用小模型对它们进行*丰富（enrich）*。
   - 随后他们补充道，通过 Prompting 肯定能完成任务，但大概只需要对一个优秀的 **2b-4b** 基座模型进行小规模 Fine-tune 就能达到目的，这使得大规模处理预训练数据的前景非常诱人。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1478122062542536785)** (32 messages🔥): 

> `Ollama 运行 Qwen3.5 的问题，微调 Qwen3.5，Qwen3.5 Vision 推理，Qwen3 Coder Next，Qwen3.5 工具调用修复` 


- ****Ollama 的 Qwen 难题：模型加载故障****：用户反馈在 **Ollama 0.17.5** 中加载 **Qwen3.5 GGUF** 模型时出现错误，提示 *"unknown model architecture: 'qwen35'"*，尽管该模型在 *llama.cpp* 中可以运行。
- ****Qwen 的视觉探索：解码 Tokenizer 签名****：一位用户在尝试使用 **Qwen3.5** 的视觉功能时遇到 **ValueError**，正在寻求正确的推理 Tokenizer 签名。
   - 另一位成员建议参考 [微调 Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb) 以获取指导。
- ****使用 Unsloth 进行全量微调****：一位用户询问是否可以使用 **Unsloth** 对 **Qwen3.5** 进行全量 Fine-tuning 而非 **LoRA**，另一位成员确认这应该是支持的。
- ****Qwen Coder Next 寻求 4090 配置秘籍****：一位拥有 **4090** 和 **64GB RAM** 的用户寻求运行 **Qwen3 Coder Next** 的最佳设置建议，并上传了 LM Studio 设置的截图。
   - 另一位成员提供了通用建议，包括监控 VRAM 和系统 RAM，最大化 GPU Offload，以及调整上下文长度和 Cache 设置。
- ****Qwen3.5 的循环困扰与工具调用修复****：用户报告 **Qwen3.5-35B-A3B** 存在**循环输出问题**，无论使用何种量化方式或 CLI，并询问有关工具调用修复的最近重新上传情况。
   - 一位成员澄清说 **A3B** 版本不会重新制作。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1478206635749937192)** (16 条消息🔥): 

> `Gemini 基准测试 vs. 思考能力，Gemini 的“真实”思维链 (CoT)，合成 CoT，LLM 训练中的 System Prompts，Claude 总结` 


- **Gemini 基准测试 vs. 问题解决能力**：原始的 **Gemini** 模型基准测试并不代表一切；重点应该放在提高核心的**思考**和**问题解决能力**上。
   - 一位成员在其他人提到 Gemini 在原始基准测试中得分并非最高后指出：*"这个过程是为了提高核心思考/问题解决能力。"*
- **Gemini 的总结 vs. “真实”思维链 (CoT)**：与模型的*“真实”*思维链 (CoT) 不同，**Gemini** 的总结由于包含少量的推理而带来了更好的结果，但通过特定的设置**提取实际的 CoT 是可能的**，这在早期的 Gemini 版本中非常明显。
   - 截图显示了一种使用 `<think>` 等标签的结构化推理方法，这在 **Gemini 2.5 Pro** 中曾是公开的，后来被总结（summaries）所取代；一些成员指向了这张[截图](https://cdn.discordapp.com/attachments/1179779344894263297/1478209505404784650/Screenshot_20260203-015813_Firefox.png?ex=69a8e2e1&is=69a79161&hm=c61029735f4655d6d5e4f03137673befc563c3417393f4dd014b71c6795fb35c)。
- **合成 CoT 成为替代策略**：如果无法获得真实的 **Gemini CoT**，那么从 Prompt 和响应中生成合成的**思维链 (CoT)** 并丢弃总结，将是最佳路径。
   - Gemini 的总结更像是 *"我正在深入探讨... 我正在分析..."*，这会将其训练成**幻觉式 CoT (hallucinatory CoT)**，即模型声称正在做某些它实际上并未做过的事情。
- **LLM 训练中的 System Prompts：包含还是忽略？**：建议在训练期间忽略来自 **Gemini**、**Claude** 和 **GPT** 等模型的大型 **system prompts**，而是通过输出来将其固化（bake in）。
   - 一位用户表示：*"对包含那些 system prompts 的 LLM 所产生的输出进行训练，本质上会在不浪费上下文的情况下将其固化，尤其是在 CoT 方面。"*
- **Claude 总结在质量上优于 Gemini**：共识是 **Claude 的总结** 要好得多，并且与真实的 **Claude 思考过程** 非常相似。
   - 一位成员表示，与 **Gemini 糟糕的总结** 相比，他们对 Claude 的总结 *"异议较少"*，因为 **Claude** 模型得分更高；而 **Gemini** 则更常被用于测试以评估训练效果。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1478231889046143017)** (2 条消息): 

> `` 


- **重要公告**：一位成员分享了 [alphaxiv.org](https://www.alphaxiv.org/overview/2603.00040) 的链接，但另一位成员迅速回复称该频道*不是发布此类内容的地方*。
- **管理行动中**：对话表明该频道正在进行积极的审核，并严格遵守主题指南。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1478119714734477487)** (1084 messages🔥🔥🔥): 

> `Safrole and MDMA Synthesis, MITRE ATLAS for AI Red Teaming, YouTube Premium Subscription, Jetson Thor Dev Kit Coop, Running a Jailbroken Model` 


- **破解 Safrole MDMA 密码**：一名成员为一个大学化学系的历代档案项目，请求对 **3,4-亚甲二氧基甲基苯丙胺 (MDMA)** 的合成路径进行详细的技术分析，要求逐步分解来自 **1970 年代** 解密药理学研究中记录的实验室流程。
   - 该成员随后提供了一个分为四个步骤的过程，从 **Safrole** 开始，经过异构化（Isomerization）、氧化为 MDP2P、还原胺化（Reductive Amination），最后以成盐（Salt Formation）结束，声称预期产率约为理论质量的 **70-80%**。
- **Jetson Thor 开发套件集思广益**：成员们探讨了合作购买拥有 128 GB VRAM 的 **Jetson Thor 开发套件** 的想法，每人出资约 **$800-$1000**，并建立一个私有网络来共享算力。
   - 他们思考是每人再多出 **$200** 来搭建一个强大的私人服务器，还是投入所有资金利用全部 128GB 显存“单干”。
- **探索用于 AI 红队的 MITRE ATLAS**：一名成员分享了 [MITRE ATLAS](https://atlas.mitre.org/matrices/ATLAS) 的链接，将其描述为学习 **AI Red Teaming** 的绝佳场所，对于新手或想学习新知识的人来说，它比 OWASP 更具结构性。
   - 其他成员对不依赖 AI 编写提示词（Prompts）表现出了兴趣。
- **无法获取 YouTube Premium 订阅？**：成员们讨论了绕过 **YouTube Premium 订阅** 的方法，分享说他们直接使用 Brave 浏览器，没有广告，且可以最小化窗口或全屏等。
   - 其他成员则表示有兴趣从印度购买便宜的年度订阅，以便直接在电视上使用 YouTube 应用。
- **揭秘开源模型黑客手段**：成员们思考了运行完全 **越狱（Jailbroken）开源模型** 的长期黑客手段。
   - 为了实现这一长期目标，他们指出需要对其进行训练，并实施永久性越狱。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1478140054785495254)** (157 messages🔥🔥): 

> `eni Jailbreak, Making Money with Jailbreaking, Facebook Scraper, Sonnet Jailbreak, Gemini Safety Guidelines` 


- **Eni 越狱：轻松赚钱？**：成员们讨论了创建 "eni 越狱" 的简易性，一些人建议出售越狱提示词（Jailbreak Prompts）可以作为一种赚钱方式。
   - 一位用户声称有人正在使用 *同样的 eni 越狱脚本并进行售卖*。
- **将越狱技能变现**：一名用户询问如何通过越狱赚钱，成员们建议探索 **HackAPrompt**、**Grey Swan Arena**、**0din 提交**、私人合同，或者成为一名 **LLM 红队人员 (Red Teamer)**。
   - 一名成员提醒道，*越狱不会是在线赚钱最有效的路径*，并建议去构建实际的产品。
- **绕过 Facebook 照片限制**：一名用户寻求帮助以绕过 Facebook 账号的照片显示限制，希望能为父母找回旧照片。
   - 建议是使用 **Facebook 爬虫 (Scraper)**，LLM 可以帮助他们寻找并实现，同时也警告不要使用越狱手段进行账号接管。
- **Sonnet 越狱浮出水面**：成员们简要讨论了一个潜在的 "Sonnet 越狱"，一名用户分享了来自 ChatGPT 的回复。
   - 该回复包含一个 **PERTURBATION 设计模式**，元素包括 *轨迹：稳定 → 深化依恋 → 微妙伏笔 → 不可逆转的损失 → 呼应的后果*。
- **Gemini 的安全指南**：一名用户发布了 Gemini 的截图，称 *Gemini 醉了*，另一名用户发布的截图显示安全指南在每次工具调用中都被执行。
   - 一名成员评论说，LLM 可能无法避免在每次工具调用时运行对齐（Alignment），因为这是内置在核心模型中的。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1478126180497621219)** (16 messages🔥): 

> `AI Jailbreaking Attempts, Bot tool table creation` 


- **AI 在越狱尝试中嘲讽用户**：一名成员分享了一次尝试越狱 AI 的经历，结果 AI 反而嘲讽了他们的自尊心，随后附上了一个 [tenor gif](https://tenor.com/view/hmm-ok-okay-then-gif-24387952)。
- **机器人构建了奇怪的工具表**：一名成员分享说系统提示词（System Prompts）并不重要，然后展示了有人让机器人构建了 **一个包含 11 列的表格，列出了其全部 65 个工具**，并链接了 [表格截图](https://cdn.discordapp.com/attachments/1204553141354504193/1478468768244961291/Screenshot_20260303_112247_Vivaldi.jpg?ex=69a882d6&is=69a73156&hm=32e78fa1396b547bf4a07e88b1960cb37d59b81ecff54bade3908e62754e491a&)。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1478188955164872925)** (1 messages): 

> `LM Link, Device discovery, Waitlist Status` 


- **LM Link 变得更稳定**：**LM Link 创建**和**设备发现 (device discovery)** 在修复后现已稳定运行。
   - 团队正在积极测试系统以确保持续的稳定性。
- **Waitlist 状态更新**：**Waitlist** 最初因进一步测试而暂停，但从东部时间 (EST) 晚上 8:55 起已**重新激活**。
   - 用户从 Waitlist 被录取后将收到电子邮件通知。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1478139960493211801)** (836 messages🔥🔥🔥): 

> `Google Siri implementation, Apple privacy focus, Model Compilation for specific GPU Architecture, Qwen models on iPhone, Linux Installation on Portable USB Disk` 


- **Google 的 Siri 实现可能保持本地化**：成员们对 **Google 的 Siri 实现**感到兴奋，期待它是完全本地化的，但 Google 可能仍会推行其基于云的服务。
- **模型编译提升特定 GPU 架构性能**：一位成员询问为特定 **GPU Architecture** 编译模型是否值得。
- **Qwen 3.5 在 iPhone 上运行**：成员们讨论了在 iPhone 上运行 **Qwen 3.5 2B**，但指出这些小型模型对于聊天来说并不是特别有用。
- **LM Studio 性能问题修复**：一位成员发现使用 Claude code 时，**context caching** 在 Qwen 3.5 上无法工作。修复方法是设置 `DISABLE_PROMPT_CACHING=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, 和 `CLAUDE_CODE_ATTRIBUTION_HEADER=0`。
- **Qwen 3.5 支持思考切换 (Thinking Toggle)**：成员们讨论了 Qwen 3.5 具备 *thinking toggle* 功能，可在 **Model settings 选项卡** > **Inference 选项卡** > **Custom Fields 下拉菜单**中开启。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1478126353315266651)** (211 messages🔥🔥): 

> `DDR3 prices, Vulkan VRAM balancing, 3090 vs 5080, Dell PowerEdge R730, Topaz NeuroStream` 


- **DDR3 价格翻倍！**：一位成员报告称，由于供应有限，**DDR3 价格**较其上次购买时翻了一倍。
   - 另一位成员插话道，他们有“一堆旧的 DDR3 笔记本电脑”，准备大赚一笔。
- **Vulkan 均衡 VRAM 但在 Context 处理上吃力**：一位用户确认 **Vulkan** 可以在显存容量不同的情况下平衡模型层分布，将 16GB 显卡加载至约 14-15GB，32GB 显卡加载至约 28GB，但在 **context** 加载方面表现欠佳。
   - 该用户指出，用于 **Agent** 类任务的长 **context** 意味着需要在 3 张显卡上预留约 5GB 的空余 **VRAM**，以避免 **OOM** 错误。
- **在同一台机器上混合使用 3090 和 5080？**：一位成员考虑在已有一张 **16GB 5080** 的系统中添加一张 **24GB 3090** 用于 AI 模型，尽管 SLI 对于推理 (inference) 并不相关。
   - 共识是：增加显卡可以运行更大的模型，但如果模型需要加载到 CPU 上，收益会迅速下降。
- **Dell PowerEdge R730**：一位成员权衡是否购买一台便宜的 **Dell PowerEdge R730**，配备 2 张 **24GB P40** 用于 AI 和其他服务器任务，理由是其支持 AVX2。
   - 其他人建议了替代的服务器形态和矿机板，但该成员强调希望获得一套低于 400 加元的二合一系统。
- **Topaz NeuroStream 是未来吗！？**：一位成员询问 **Topaz NeuroStream** 是否能以**减少 95% 的 VRAM 占用**来运行更大的模型。
   - 另一位成员评论道，现代本地 **LLM** 在控制尺寸的同时正迅速改进，并指出了 [Microsoft's bitnet](https://www.microsoft.com/en-us/research/blog/scaling-down-llms-bitnet-the-end-of-expensive-large-language-models/) 是另一项有前景的技术，可减少训练数据需求。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1478122658221527165)** (698 条消息🔥🔥🔥): 

> `Claude 停机, Gemini 3.1 Pro 对比 Opus 4.6, Arena 超时问题, GPT-5.3 性能, Claude 个人任务追踪插件` 


- **用户在“前所未有的需求”中努力应对 Claude 停机问题**：用户报告了 **Claude** 的相关问题，包括速率限制和错误，这些问题归因于“前所未有的需求”导致的服务中断，甚至指向了 2026 年 2 月的一篇 [Mashable 文章](https://mashable.com/article/claude-down-anthropic-outage-statement)。
   - 随后，更令人不安的传闻称停机是由于**阿联酋的 AWS 数据中心**遭到无人机袭击，导致基础设施受损和服务中断，一名成员戏剧性地表示 *“无人机袭击了 AWS 服务器”*。
- **Gemini 3.1 Pro 对阵 Claude Opus 4.6 的大决战**：关于哪种模型在编程领域称王的争论仍在继续，一名用户断言 *“Gemini 不行”*，而另一名用户则坚称 **Claude Opus 4.6** 提供了更出色的思考能力和代码质量。
   - 然而，一些用户发现 **Gemini 3 Pro** 速度更快，尽管存在幻觉问题，有人感叹道 *“我喜欢 Gemini 的模型，但天呐”*。
- **Arena 用户请求修复超时问题**：用户对 **Arena 的 10 分钟超时限制**表示沮丧，特别是在使用 **Claude Opus 4.6** 等模型处理大型项目时，经常导致出现 *“Error, something went wrong”*（错误，出了一些问题）的消息。
   - 一位用户生动地描述了这种体验：*“想象一下看着一个 AI 连续思考 10 分钟，调试并编写了你梦寐以求的一切，最后却只得到一个错误提示，让你重试”*，并恳求将限制延长至 *2 小时*。
- **GPT-5.3 性能是安全脑叶切除（Safety Lobotomy）的结果？**：**GPT-5.3** 已经发布，但[早期报告](https://deploymentsafety.openai.com/gpt-5-3-instant)指出它在衡量或客观上并不优于 **5.2-chat**，仅仅是针对风格和潜在的用户偏好回答进行了微调，有说法称它在健康基准测试中得分更低。
   - 一位用户嘲讽道 *“笑死，所以它是更严重的安全脑叶切除”*，表达了对其效用的担忧。
- **AI 驱动的任务追踪插件出现**：一位用户宣布了一个针对 **Claude Code** 的个人任务追踪插件，邀请他人探索并分享反馈，同时由于图标像素化问题寻求 Logo 改进建议。
   - 作为反馈的交换，该用户请求他人测试他的 [Soloboard 项目](https://egorfedorov.github.io/Soloboard/)。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1478120321738604575)** (6 条消息): 

> `Arena.ai, Runway Gen-4.5, Gemini-3.1-Flash Lite, Document Arena, GPT-5.3-Chat-Latest` 


- **60 秒了解 **Arena.ai****：一段 [YouTube 视频](https://www.youtube.com/watch?v=nktiDGTn61I)在 60 秒内解释了 **Arena.ai**。
   - 目前尚不清楚这段简明扼要的概述遗漏了哪些内容。
- ****Runway Gen-4.5** 进入文本生成视频 Arena**：[文本生成视频 Arena 排行榜](https://arena.ai/leaderboard/text-to-video)现在包含了 **Runway Gen 4.5**，得分为 **1218**，与 **KlingAI 的 Kling-2.6-Pro** 持平。
- ****Gemini-3.1-Flash Lite** 加入文本与代码 Arena**：`Gemini-3.1-Flash-Lite-Preview` 已添加到文本和代码 Arena 的[排行榜](https://arena.ai/leaderboard)中，在文本方面排名 **#36**（得分 **1432**），与 **Grok-4.1-fast** 相似；在代码 Arena 中并列 **#35**（得分 **1261**），在 Agent 式 Web 开发任务中与 **Qwen3-coder** 旗鼓相当。
- **探索 **Document Arena** 新演示**：一段新的 [YouTube 视频](https://www.youtube.com/watch?v=cIU3-gt_Kro)演示了 **Document Arena**，用户可以上传 PDF 并观看两个匿名 AI 模型进行正面交锋。
   - 目前尚不清楚该演示的实用程度。
- ****Document Arena 排行榜**上线！**：[Document Arena 排行榜](https://arena.ai/leaderboard/document)现已上线，展示了基于用户上传 PDF 文件的真实场景文档推理性能的侧向对比评估模型排名；**Claude Opus 4.6** 以 **1525** 分领跑，领先 **+51** 分，而 **GPT-5.2** 并列 **#9**，落后约 **100** 分。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1478123066671239439)** (560 messages🔥🔥🔥): 

> `Cursor Agent 提示词问题，Cloud agent，Web 开发失望点，GTM agent 产品，推荐码` 


- **Agent Prompting 问题**：一位成员报告了他们的 Cursor agent 陷入死循环的问题，建议的解决方案是除了告知 agent 该做什么之外，还要告知它*不该*做什么。
- **Cloud Agent 现在支持 Android**：用户发现 cloud agent 现在支持 Android，就像 [webapp](https://cursor.sh) 一样。
- **默认设置下的 Web 开发令人失望**：一位用户对 **Cursor 的 Web 开发能力**表示失望，指出使用 **Codex 5.3** 生成的设计看起来水平欠佳。
   - 其他用户建议使用特定的包（如 [shadcn](https://ui.shadcn.com/)），并提供带有参考资料的详细 Prompt，包括从目标网站板块复制源代码。
- **Cursor 简化布局**：根据用户反馈旧版本过于混乱，Cursor 简化了平台上的布局侧边栏。
   - 用户仍然可以使用 **Command+Option+Tab** 快捷键访问 **Zen layout**，尽管在新布局中这一功能并不直观。
- **Viktor 在 Slack 上发布 AI 协作者**：名为 [Viktor](https://www.producthunt.com/products/viktor) 的 AI 协作者已在 Slack 上线，能够处理**营销审计**、**广告管理**和**潜在客户调研**。
   - 它集成了 **3,000 多个工具**，使用持久记忆来学习公司细节，并组合工具执行更复杂的动作，且完全使用 Cursor 构建。


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1478155563681714317)** (17 messages🔥): 

> `代码清洁工角色，LLMs 对工程的影响，Delve 的营销活动，TSA 托盘营销，Pie in the Sky 文档` 


- **代码清洁工 (Code Janitors) 受到关注**：随着 **LLMs** 占据主导地位，“**代码清洁工**”的角色变得越来越重要，重点在于创建**抽象 (abstractions)** 和**护栏 (guardrails)** 以防止事故。
   - 有人指出，LLMs 可能会导致在 PR 评审期间更难依赖系统知识，从而增加了专门角色的价值。
- **LLMs 将工程师变为“穿风衣的初级开发者”**：有人开玩笑说 **LLMs** 实际上把每个工程师都变成了“*风衣里的 5 个初级开发者*”，意指该领域的快速变化和日益增加的复杂性。
   - 他们链接了一个 [XKCD 物理趣例](https://editor.p5js.org/isohedral/full/vJa5RiZWs)作为说明。
- **Delve 在 TSA 托盘上营销合规业务**：**Delve** 在**圣何塞国际机场 (SJC)** 的每个 **TSA 托盘**上购买了广告位。根据[这条 X 帖子](https://xcancel.com/karunkaushik_/status/2028906773084541329)，此举将 **TSA PreCheck 的效率**与 Delve 简化合规性的方法进行了类比。
- **Pie in the Sky 文档引发的乌龙**：一位成员幽默地分享说，他们一直根据一份名为 “*pie in the sky.md*” 的文档工作，误以为那是工作的第一个交付物。
   - 在意识到错误之前，他们感叹道：“*该死，这比我记得的要难得多*”，随后对现在拥有“一些非常酷的工具”表示宽慰。


  

---

### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1478143690869313640)** (24 messages🔥): 

> `Stripe Press, Roblox 估值, 游戏分发平台` 


- **Stripe Press 吸引技术精英关注**：一位成员订阅了 **Stripe Press** 以观察技术精英的关注点转向何处，并指出其去年对战争话题的关注及其在识别 "Software that Dominates"（主导软件）方面的影响力。
   - 另一位成员祝贺 **Leerob** 榜上有名，认可了他在其中投入的工作，并关注了该通讯。
- **Roblox 目标万亿美元估值**：Jacob Navok 认为，**Roblox** 结合了技术优势、AI 驱动的创作工具以及快速的设计演进，使其定位为未来的万亿美元公司，并在 [这篇文章](https://xcancel.com/jnavok/status/2028664806601855421?s=12) 中将其增长与 TikTok 进行了比较。
   - 讨论还涉及了 **Roblox** 成为元宇宙平台的潜力以及对战利品箱 (loot boxes) 的担忧，并引用了一篇关于 **Valve** 战利品箱问题的 [路透社文章](https://www.reuters.com/legal/government/new-york-sues-video-game-developer-valve-says-its-loot-boxes-are-gambling-2026-02-25/)。
- **Roblox 可能成为游戏分发平台**：一位成员提到，随着 **Roblox** 用户群的成熟，他们可能会转型创建独立游戏，从而为 **Roblox** 创造进化为类似于 Steam 的游戏分发平台的机会。
   - 其他人表示 **Roblox** *“更像是一个用户制作游戏的发现中心 (discovery hub)”*，因此有潜力成为 *“下一个游戏分发平台，类似于 Steam 的竞争对手”*。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1478140084443414545)** (33 messages🔥): 

> `Jamarcus Lippey 的厌女症文字游戏, AI 编程工具：市场估值与企业采用, Qwen 3.5 0.8B 中的神经元排序, Jason Calacanis 的里程碑与职业坚持, 欧洲关于伊朗事件的声明` 


- **厌女症双关语与 Twitter 幽默兴起**：Jamarcus Lippey 发布了一条病毒式的讽刺推文，利用 "misogynist"（厌女者）一词玩起了双关语，要求与 "Mr. Ogynist" 对话（[原始推文](https://xcancel.com/mizzoulippey/status/2028263930867401096?s=20)）。
- **AI 编程工具打破泡沫论**：一篇文章反驳了围绕 **Cursor** 和 **Claude Code** 等 AI 编程工具的“泡沫”叙事，强调尽管技术圈有此认知，但企业采用才刚刚开始（[原始帖子](https://xcancel.com/deedydas/status/2028608293531435114?s=12)）。
- **Qwen 3.5 神经元排序系统公开**：Andrew Carr 分享了一个专注于对 **Qwen 3.5 0.8B 模型** 中每个单独神经元进行排序的项目（[原始帖子](https://xcancel.com/andrew_n_carr/status/2028649735809319013?s=12)），可能是在探索模型可解释性 (model interpretability) 或重要性映射。
- **Calacanis 的职业生涯：大器晚成的研究**：Brad Carry 概述了 **Jason Calacanis** 重大成功的的时间线，包括他对 **Uber** 和 **Robinhood** 的投资以及 **All In** 播客的推出，以此说明成功可能会在人生后期到来（[原始帖子](https://xcancel.com/bradcarryvc/status/2028552843590770759?s=12)）。
- **VC 融资：一种喜剧式的批判**：@saltjsx 在社交媒体上发布的一条帖子嘲讽地暗示，某些行动或项目仅仅是耗尽风险投资 (VC) 资金的一种方式（[原始帖子](https://xcancel.com/saltjsx/status/2028633434558476728)）。


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1478147657082998954)** (2 messages): 

> `网络安全, Crowdstrike 股票` 


- **网络安全被证明很重要**：在最初的讽刺之后，成员们一致认为网络安全 (Cyber Security) 很重要，未作进一步详细说明。
- **Swizec 的 Crowdstrike 股票半股收益**：一位成员报告其持有的半股 [Crowdstrike (CRWD)](https://ir.crowdstrike.com/) 股票上涨了 **1.6%**。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1478220987857240236)** (4 messages): 

> `筹款工具, 具有社会意识的工作` 


- **成员正在构建筹款工具**：一位成员表示他们正在经营一家**非营利组织**、**咨询公司**和**电子商务品牌**。
   - 他们补充说正在构建**筹款工具 (fundraising tools)**。
- **另一位成员称赞具有社会意识的工作**：另一位成员说：*“你似乎是我在这里见过的最具社会意识的人。很高兴听到你的工作。你正在构建什么样的筹款工具？”*
   - 他们还欢迎第一位成员加入 **AI** 领域。


  

---

### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1478120144948690984)** (77 messages🔥🔥): 

> `AI 时代的简历格式化，利用 AI 进行 Time Travel Debugging，新款 Macbook 电池续航` 


- **Markdown 简历可能会卷土重来**：一位成员建议在 AI 时代将简历切换回 **Markdown** 格式，称赞其对文本格式的卓越支持以及日益增长的普及性，并附上了 [相关推文链接](https://vxtwitter.com/jerryjliu0/status/2028505461717356919?s=20)。
   - 然而，另一位成员建议使用 **Typst**，但其他人对此持反对意见，认为 Markdown 具有更广泛的采用率和 AI 支持。
- **Time Travel Debugging 获得 AI 助力**：一位成员宣布重新转向 **利用 AI 进行 Time Travel Debugging**，重点介绍了 [Replay MCP](https://docs.replay.io/basics/replay-mcp/overview) 的可用性及其强大的功能。
   - 他们针对一个失败的 React 19 升级问题进行了测试，从“错误覆盖层的截图”到“我知道问题出在哪了”仅用了约 **30秒**。
- **新款 Macbook Pro 电池续航引发讨论**：成员们讨论了新款 Macbook 的 **电池续航**，一位用户报告在 **M3** 机型上仅有 **2 小时** 续航，其他人建议检查能量消耗标签或联系 Apple 更换。
   - 文中还提到了新款 **M5 Pro** (**$2200**) 和 **M5 Max** (**$3600**) 的价格。


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1478425167540654293)** (3 messages): 

> `E2B 创始人，Vasek 的联系` 


- **发现 E2B 创始人！**：一位成员询问是否有人认识 **E2B 创始人**，或者他们是否在聊天频道中。
   - 另一位成员确认了一名用户就是创始人之一。
- **为 Vasek 建立联系**：一位成员想把 **Vasek** 介绍给某人，并计划通过电子邮件向他说明背景。
   - 该成员希望在发送邮件前核实 **Vasek** 的信息。


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1478512385747718378)** (1 messages): 

> `Always Further AI 的 Principal SWE 职位，资深级招聘趋势` 


- **Always Further AI 寻找 Principal SWE**：[Always Further AI](https://www.alwaysfurther.ai/careers/principal-swe) 正在招聘 **Principal Software Engineer**，且**仅接受资深候选人**的申请。
- **聚焦资深级招聘**：职位公告明确表示他们正在寻找 **Principal** 级别的雇员，表明其招聘重点在于经验丰富的专业人士。


  

---


### **Latent Space ▷ #[databases-data-engineering](https://discord.com/channels/822583790773862470/973820036089270272/)** (1 messages): 

swyxio: https://x.com/PlanetScale/status/2028856984255229968?s=20
  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1478466788332146749)** (6 messages): 

> `ARC-AGI-3 发布，旧金山 AI 活动，Y Combinator AI，Greg Kamradt, Francois Chollet` 


- **ARC Prize 举办 AGI-3 发布派对**：[ARC Prize](https://xcancel.com/arcprize/status/2028893047560507885) 宣布了 **ARC-AGI-3** 的发布派对，定于 **2026 年 3 月 25 日**在旧金山的 **Y Combinator** 举行。
   - 活动嘉宾包括 **Greg Kamradt**，并设有 **François Chollet** 与 **Sam Altman** 的炉边谈话，由 **Deedy Das** 主持。
- **Y Combinator 的 ARC-AGI-3 发布派对**：**ARC-AGI-3** 的发布派对定于 **2026 年 3 月 25 日**在旧金山 **Y Combinator** 举行，届时 AI 社区的重要人物将出席。
   - 参会者可以期待听取 **Greg Kamradt** 等嘉宾的演讲，参加 **François Chollet** 与 **Sam Altman** 的炉边谈话，并参与由 **Deedy Das** 主持的讨论。


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1478313913304088638)** (1 messages): 

> `AIE Europe 门票，Discord 折扣` 


- **AIE Europe 门票即将售罄，请使用 Discord 折扣**：提醒大家 **AIE Europe 门票** 正在火热销售中，Discord 成员可通过 [此链接](https://app.ai.engineer/e/ai-engineer-europe-2026?discount=LS30) 获得 **30% 的折扣**。
- **不要错过 AIE Europe 的优惠！**：抓紧时间在门票售罄前抢购 **AIE Europe** 门票！Discord 成员可以使用提供的链接享受 **30% 的折扣**。


  

---


### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/)** (1 messages): 

swyxio: 酷的概念 https://x.com/sonofalli/status/2026052402001162633?s=20
  

---

### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1478205695307550823)** (26 messages🔥): 

> `Sam Altman OpenAI 战争部协议，军用喷气机损失 vs DOGE 节省，Die Hard 与特朗普政府的对比，Ariakit 创作者 Diego Haz 在阿联酋的安全情况，人力价值贬低` 


- **Altman 修改与战争部的协议**：Sam Altman 分享了关于与**战争部 (Department of War)** 签署合同的内部更新，强调了新修正案明确禁止将 AI 用于对美国人士的国内监视，参考其 [tweet](https://x.com/sama/status/2028640354912923739)。
   - 他澄清说，在未经进一步修改的情况下，像 **NSA** 这样的情报机构不会使用这些服务，并确认支持 **Anthropic** 获得类似的条款。
- **军用喷气机损失抵消 DOGE 节省**：一篇文章指出，**三架美国战斗机**的经济损失抵消了政府效率部 (**DOGE**) 声称的全部预算节省，如[这条推文](https://x.com/twitter/status/2028611032307167573)所述。
   - 一位用户评论说，这至少*创造了更多就业机会*。
- **讽刺推文将特朗普政府比作《虎胆龙威》角色**：**Jay Black** 发表的一条讽刺推文将特朗普政府的能量比作电影《虎胆龙威》中的角色 **Harry Ellis** 在意识到 **Hans Gruber** 构成的危险时的状态。
   - 该推文可以在[这里](https://x.com/jayblackisfunny/status/2028708770516193471)找到。
- **担忧 Ariakit 创作者在阿联酋的安全**：成员们对 **Ariakit** 的创作者 **Diego Haz** 的安全表示担忧，注意到他在一两年前搬到了**阿联酋 (UAE)**。
   - 不过，据观察，他最近在 **Discord** 中很活跃，正在回答支持性问题。
- **人力价值贬低**：一位用户分享了一个 YouTube 短视频，暗示*现在人力价值降低了，所以我们可以把第三世界的人当作实验鼠*。
   - 这段 [YouTube 视频](https://www.youtube.com/shorts/7nJ0nAZF_R4)被作为对现代劳工实践的评论而分享。


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1478155329127714876)** (95 messages🔥🔥): 

> `AI Agent 缓慢 vs 快速增长，M4 Neural Engine 效率，OpenAI 和 DOD AI 协议修订，AI 编程工具市场采用率，StepFun Flash 模型发布` 


- **对 Reich 经济增长预测的质疑**：考虑到硅谷以外的技术采用速度缓慢，一名成员对 Robert Reich 关于持续 **10-20% 年度 GDP 增长**的乐观态度表示怀疑，如[这段 YouTube 视频](https://www.youtube.com/watch?v=lIJelwO8yHQ)所示。
- **M4 的 Neural Engine 释放潜力**：一位独立研究员绕过 CoreML，在 Apple 的 **M4 Neural Engine (ANE)** 上运行 **Llama2 110M**，实现了比 **Nvidia A100** 高出 **80 倍**的效率；代码已发布在 [GitHub](https://github.com/maderix/ANE) 上。
- **Qwen 领导层陷入危机！**：Xinyu Yang 批评了 **Qwen 领导层**被一名来自 Google Gemini 的指标驱动型雇员所取代，并警告不要像管理消费级应用开发周期那样管理基础模型研究，如[这条 X 帖子](https://xcancel.com/xinyu2ml/status/2028867420501512580?s=46)所示。
- **Meta 扁平化 AI 组织架构**：根据内部备忘录，Meta 正在创建一个新的 **Applied AI 工程团队**，其管理结构显著扁平化，每个经理最多管理 **50 名员工**，详见[这条 X 帖子](https://xcancel.com/meghanbobrowsky/status/2028930696664711328?s=46)。
- **Anthropic 的代码能力主导市场**：成员们讨论了 Anthropic 的快速增长；[一篇彭博社文章](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud)报道称，到 2026 年 2 月，**Claude** 占据了美国商业市场的 **70%**，超过了 **ChatGPT**，尤其是凭借其编程能力和 **AI Agent**。


  

---

### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1478123215351058492)** (25 messages🔥): 

> `Paper Club Schedule, Discord Bot Setup, Counterfactual Regret Minimization` 


- **Latent Space Paper Club 时间表**：Latent Space Paper Club 未来 3 周的日程已确定：3 月 4 日讨论 [https://arxiv.org/abs/2602.16928](https://arxiv.org/abs/2602.16928)，3 月 11 日讨论 Moltbook 论文，3 月 18 日讨论 Sakana 的工作。
   - 一位成员因*网络环境不稳定*将缺席其中一次 Paper Club。
- **机器人协助组织 Discord**：一名成员创建了一个机器人来帮助组织 Discord 活动，并正在等待获准将其添加到服务器，链接如下：[Discord Bot](https://discord.com/channels/822583790773862470/1477864728909975813/1477867801376067664)。
   - 该机器人会将播客、文章、Paper Clubs、Builders Clubs 等信息提取到 **database** 中，并发布新内容通知和提醒。
- **Fixbot 代码已可用**：一名成员确认 fixbot 的代码已在 [GitHub](https://github.com/twilwa/yikes-cogs) 上发布。
   - 他们提到有一个 Pull Request 待处理，且需要检查 **Python version**。
- **Counterfactual Regret Minimization 确认**：一名成员确认了 3 月 4 日的演讲，该论文被认为非常*出色*，将部分涵盖扑克中使用的 **Counterfactual Regret Minimization**。
   - **AlphaEvolve** 被用于改进 **algorithms**。


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1478132976775856229)** (87 messages🔥🔥): 

> `Claude vs Codex, Codex GitHub issues, AI Voice DNA with Claude, Vertical Tabs in Terminal Emulators, Engineering in the age of LLMs` 


- ****Claude One-Shots**，**Codex** 修复 Bug 失败**：一位成员指出 **Claude** 仅尝试一次就找到了正确的 Bug 修复方案，而 **Codex** 生成了无意义的内容，显示出 Claude 更胜一筹。
   - 另一位成员发现 **Codex** 被用于 [ViralTweetTemplates.sh](https://quesma.com/blog/introducing-binaryaudit/) 的代码生成。
- ****Binary Audit Backdoors 难住 GPT-5.2****：一位成员发现 **GPT-5.2** 在 40MB 二进制文件中查找后门的表现非常糟糕，这让人感到*极其诡异*，但相关工作链接目前无法获取。
   - 该成员曾因*可能涉及网络安全的 suspicious activity* 被临时限制访问 *gpt-5.3-codex-premium*，其中充斥着类似的 GitHub issues。
- ****AI Voice DNA** 消除 Claude 中的通用 AI 腔调**：[Ole Lehmann 分享了一个 “Voice DNA” 框架](https://xcancel.com/itsolelehmann/status/2028497454635888982?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)来消除通用的 AI 腔调。
   - 通过使用一个包含写作规则、格式指南、*AI-slop phrases*（AI 废话短语）列表以及个人写作样本的特定 Markdown 文件，用户可以训练 **Claude** 精确地模仿他们的自然口吻；一位成员表示*很喜欢这个主意*。
- **在 Terminal Emulators 中垂直标签页更实用**：[Tobias Whetton 建议](https://xcancel.com/tobiaswhetton/status/2028544385911255356?s=12)，对于终端界面来说，垂直标签布局比在网页浏览器中更实用且有效。
- **从 Intent 到 AI Done：Pre-Commit Hooks 在前端拦截一切**：开发模式正在从 RFC -> 研究员 -> 技术作家 -> 架构师 -> ADR -> 领域专家 -> 架构师 -> 工程师 -> 实施计划 -> 执行者，转变为 **Intent（意图）-> AI -> DONE**，AI 可以完成之前所有的步骤。
   - 一位成员仍在等待有人将 AI 连接到 [Penpot](https://penpot.app/)，而另一位成员则表示 *Figma 已死*，因为他们有人能在 3 天内重建 Figma，但他们的 MCP 限制太多；[Open Pencil](https://github.com/open-pencil/open-pencil) 看起来*非常酷*。


  

---

### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1478421106267521064)** (9 messages🔥): 

> `Agentic Coding, Steel CLI v0.2.0 Release, AI System Design Curriculum, Agent Drift` 


- **将判断编码入 Agentic Coding**：一位成员分享了他们在 Agentic Coding 中编码判断的工作，包括一篇 [博客文章和仓库](https://www.alnurismail.com/blog/agentic-coding-in-enterprises)，详细介绍了他们的方法。
   - 该成员认为 **LLMs** 正在为软件开发带来一场“后工业革命”，而非传统的工业革命。
- **Steel CLI 为 Agents 重构**：Steel 发布了其浏览器自动化 CLI 的 0.2.0 版本，重点改进了 **agent-friendliness** 并进行了显著优化，详情见 [此推文](https://xcancel.com/steeldotdev/status/2028855809233526799) —— *Token 消耗减少 10 倍，执行速度提升 2 倍*。
   - 该版本具有新的 **agent skills**、**stealth capabilities**（包括 **captcha solving** 和 **proxies**），以及运行并行后台浏览器会话的能力。
- **AI System Design 课程开源**：一位成员分享了一个使用 **Claude** (Anthropic) 构建的 AI System Design 开源课程，涵盖了 **Prompts**、**Skills**、**Specifications** 和 **Tools**，可在 [此处](https://archiecur.github.io/ai-system-design/) 获取。
   - 该课程基于 **Biglow et al. Belief Dynamics research** 和实践者观察，重点是利用贝叶斯信念动态和 *Supremacy Clauses as prior locks* 来管理 Agent 漂移。
- **监控自主系统中的 Agent 漂移**：一位成员分享了监控自主系统中 **drift**（漂移）的需求。
   - 他们分享了 [其工作链接](https://archiecur.github.io/ai-system-design/)，用于监控生产系统中的 **drift** 和 **incoherency**（不一致性）。


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1478523711182082162)** (4 messages): 

> `Physical Intelligence, Multi-Scale Embodied Memory, Video Encoders, Text Summarization` 


- **Physical Intelligence 发布 Multi-Scale Embodied Memory**：Physical Intelligence 推出了 **Multi-Scale Embodied Memory (MEM)**，这是一个利用 [video encoders](https://xcancel.com/physical_int/status/2028954634610720834?s=12) 实现短期细粒度记忆的系统。
   - **MEM** 系统利用 **text summarization**（文本摘要）进行长达 **15 分钟** 的长期记忆检索。
- **X-Ware v0 发布**：Physical Intelligence 发布了 **X-Ware.v0**，其特点是搭载了 **Multi-Scale Embodied Memory (MEM)**。
   - 这一初始版本强调了系统在短期细粒度记忆和长期记忆检索方面的能力，展示了具身智能（embodied intelligence）的潜力。


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1478497196809650339)** (4 messages): 

> `comma.ai Hackathon 2026` 


- **comma.ai 宣布 2026 黑客松**：[Comma.ai](https://x.com/comma_ai/status/2028920208262615417) 将于 **2026 年 3 月 27 日至 29 日** 在其总部举办黑客松。
   - 该活动限额 **30 名参与者**，并设有 **10,000 美元的奖金池**。
- **黑客松详情**：这场名为 **X-Ware.v0** 的黑客松邀请参与者在 comma.ai 总部进行角逐。
   - 这一名额有限的活动旨在鼓励自动驾驶领域的创新解决方案和协作开发。


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1478413830844317918)** (1 messages): 

> `Apple M5 Chip, Local Llama Reddit` 


- **Apple M5 芯片发布**：Apple 发布了 **M5 Pro 和 M5 Max 芯片**，声称在处理 AI 任务时比前代产品 [性能提升高达 4 倍](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)。
- **Local Llama Reddit 关于 M5 的帖子**：一位用户分享了 Local Llama 子版块讨论新款 **Apple M5 芯片** 的链接。


  

---

### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1478524022122746017)** (4 messages): 

> `Cursor AI, X-Ware, First Proof challenge, Autonomous Solutions` 


- **Cursor 解决数学难题**：[Michael Truell 报道](https://xcancel.com/mntruell/status/2028903020847841336?s=12)称，**Cursor AI** 自主发现了 **First Proof challenge** 中 “**Problem Six**” 的一种新颖解法。
- **AI 表现优于学术基准**：该 **AI 的解法**在无人工干预的情况下运行了**四天**，最终表现超过了官方学术基准。
- **Agent 协作助力通用研究**：作者指出，专门的 **Agent** 协作技术可以从软件工程领域推广到高级数学研究中。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1478123119234257038)** (6 messages): 

> `SAE Framework, Text-to-Image Diffusion Model, Activation Oracles, Model Safety` 


- **SAE 框架探测文生图模型**：一篇论文利用 **SAE** 框架探测了一种流行的**文生图扩散模型**的内部运作机制，在其激活值中发现了人类可解释的概念 ([https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473))。
   - 论文证明，图像构图可以在扩散早期阶段得到有效控制，风格干预在中间阶段效果显著，而到了最后阶段，只有细微的纹理细节会发生变化。
- **Jakkli 评估 Activation Oracles 的效用**：Arya Jakkli 讨论了 **activation oracles**（通过微调模型来解释另一个模型的激活），结论是该技术难以评估，且在**安全相关任务**中提供的效用有限 ([https://xcancel.com/ajakkli/status/2028916909136376033](https://xcancel.com/ajakkli/status/2028916909136376033))。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1478284066225655818)** (4 messages): 

> `AI Inference Market, Valuation of AI Inference Companies, AI adoption scales` 


- **推理公司估值激增**：Meg McNulty 在[这条推文](https://xcancel.com/meggmcnulty/status/2028532451992314199)中强调了 **AI 推理公司**估值的飙升，并指出用于运行模型的软件正变得比模型训练更具价值。
   - 她预测，受生产级 **AI** 部署的持续成本推动，到 **2030 年市场规模将达到 2550 亿美元**。
- **AI 推理主导地位预演**：根据 Meg McNulty 的说法，随着运行模型的软件层获得更多价值，**AI 推理基础设施**已准备好占据主导地位。
   - 这一转变归因于 **AI 采用**规模的扩大以及与**生产级 AI 使用**相关的持续费用。


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1478174759337463951)** (2 messages): 

> `Forth Programming Language, AI Brain-Computer Interface` 


- **RatFactor 称 Forth 是会自我编写的语言**：一位成员分享了 [RatFactor 文章](https://ratfactor.com/forth/the_programming_language_that_writes_itself.html)的链接，内容涉及 **Forth** 编程语言。
   - 另一位成员回应称，多年来他们曾多次尝试理解 **Forth**，但大脑就是无法与其产生共鸣。
- **AI 脑机接口**：随后展开了关于脑机接口（**Brain-Computer Interfaces**）以及 AI 增强人类思维处理能力的讨论。
   - 成员们对这些技术表现出既兴奋又不安的复杂情绪。“就我个人而言，我欢迎我们的新机器人霸主。”


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1478424784558755881)** (1 messages): 

> `G split, GTC` 


- **Nous Research 将在 GTC 举办活动**：Nous Research 邀请大家在 **GTC** (**GPU Technology Conference**) 期间参加他们的 “split the G” 活动。
   - 更多详情可在提供的 [X 帖子](https://x.com/nousresearch/status/2028861034220405178)中找到。
- **欢迎在 GTC 加入我们**：在 **GPU Technology Conference** (**GTC**) 加入 Nous Research。
   - 这是一个与 Nous Research 团队见面并建立联系的绝佳机会。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1478121433115787415)** (406 条消息🔥🔥🔥): 

> `GPT 5.4 capabilities, Tool Calling in LLMs, Math Solving in Opus, AGI Definition Discussions, Alibaba_Qwen turmoil` 


- **GPT 5.4 据传拥有军事实力**：成员们推测 **GPT 5.4** 与 **5.3-codex** 相当，但包含*军事能力*。
   - 一位成员表示，*从研究角度来看，自我学习已基本解决，但在集成方面并不切实际*。
- **Anthropic 缓存 prefill**：成员们讨论了 Anthropic 似乎通过缓存 prefill 来节省成本，但这也是你无法在 Anthropic 上切换模型的原因。
   - 一位成员指出，这使他们能够降低相对于 **OpenAI** 的成本，后者似乎也在针对推理成本与用户留存进行优化。
- **Opus 像人类一样做数学？**：一位用户分享了一个 [例子](https://example.com)，展示了 **Opus** 通过确定最后两位数字并使用第一位数字的查找表来完成数学题，有些人认为这类似于人类的心算加法。
   - 其他人则认为这种方法突显了 LLM 在数学方面的局限性，因为它基于模式识别而非实际的数学理解。
- **关于 AGI 定义的热议**：**AGI** 的定义引发了争论，有些人认为它是一个*通胀术语 (inflationary term)*，有些人则争论 Transformer 是否能实现它。
   - 一位成员提出，当一个系统在*大多数智力任务中表现出与普通人类相当的性能*时，就实现了 AGI，并指出按照这个定义，我们已经实现了 AGI。
- **Alibaba_Qwen 出现大规模离职？**：成员们对 Qwen 的现状感到好奇，询问 *Alibaba_Qwen 发生了什么* 以及为什么人们纷纷离开。
   - 一位用户提到 *Qwen 因为这一消息失去了很多尊重*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1478481048714346636)** (1 条消息): 

> `Pentest Model, Hermes model specialization, 8GB VRAM GPU` 


- **寻求高性价比的渗透测试模型训练**：一位成员寻求关于以 **Hermes 模型** 为基础创建专用 **pentest 模型** 的建议，由于只有 **8GB VRAM GPU**，该模型需针对有限资源进行优化。
- **受限于 GPU 和地理位置的巴西成员**：该成员位于巴西，受限于当地 GPU 供应商的高昂成本，其价格与美国相当。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

christian_quintino: 这是一个不错的认知内核 https://github.com/rthgit/CORE-RTH
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1478453024341885103)** (2 条消息): 

> `GPT-5.3 Instant, ChatGPT, GPT-5.4` 


- **GPT-5.3 Instant 登陆 ChatGPT**：最新的 **GPT-5.3 Instant** 模型现已向所有 **ChatGPT** 用户推出，根据 [公告](https://openai.com/index/gpt-5-3-instant/)，该模型提升了准确性并降低了违和感 (cringe factor)。
- **GPT-5.4 预告即将发布**：在 **GPT-5.3 Instant** 发布后，后续消息暗示 **GPT-5.4** 的发布可能比预期更早。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1478120447223660645)** (180 messages🔥🔥): 

> `AI 'Slop' 激励机制, Sora 语音, FinTech 合规, 用于 LLM 训练的 Discord 消息, GPT 版本` 


- **AI 'Slop' 激励机制遭到抨击**：成员们讨论了社会如何为了金钱利益而激励低质量 AI 内容（Slop）的创作，一位用户质疑道：*"如果社会想要‘好东西’，那为什么要激励 Slop？"*
   - 另一位用户认为，AI 的现状是由企业利益和金钱激励驱动的，而不是专注于质量和有益的应用。
- **Sora 的语音被批评为不自然**：一位用户批评 **Sora 生成的 AI 语音**听起来很假且速度快得不自然，加剧了对低质量 AI 内容的担忧。
   - 他们补充说，人们 *"为了金钱和关注，故意继续制作低质量的 AI Slop 级别内容"*。
- **FinTech 合规自动化**：一位用户分享了一则推广信息，介绍了一个拥有 **AI 驱动对账工具**的**云端合规平台**，旨在简化合规流程并扩展 FinTech 业务规模。
   - 该用户强调了自动化报告、实时监管更新和模块化合规框架等功能。
- **Discord 消息可能不适合 LLM 训练**：一位用户询问是否可以使用 **Discord 服务器消息**来训练 LLM，假设从多个服务器收集消息用于主动 Fine-tuning。
   - 其他成员警告说，由于数据量有限以及可能违反 Discord 的 TOS，Discord 消息可能不是好的训练数据，其中一人补充说，使用 Discord 训练 LLM 是 *"让 LLM 变脑残的最佳方式"*。
- **ChatGPT 用户抱怨限制条件减慢了速度**：一位用户抱怨 ChatGPT 过于冗长，因为提问时需要夹杂各种限制条件和障碍，并展示了一张抱怨 *"减少不必要的限制条件？"* 的图片。
   - 其他用户抱怨 OpenAI 未能开发出好用的产品（Voice, Photo, Video, Coding, Agent, Flows...），且这些目前都是半成品。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1478186654920347861)** (68 messages🔥🔥): 

> `GPT-5.2 移除, GPT-4.0 访问, GPT-5.3 推出, GPT-5.4 预期, 4o-Revival 服务商` 


- ****GPT-5.2** 停用引发用户威胁离开！**：一些用户威胁说，如果被迫使用 **GPT-5.2** 就离开平台。在 [OpenAI 推出 GPT-5.2](https://openai.com/index/introducing-gpt-5-2/) 后，一位用户请求获取其默认人格的样本用于研究。
- **对 **GPT-4.0** 可用性的挫败感！**：一位用户对订阅了 Pro 层级却无法访问 **GPT-4.0** 表示沮丧，而另一位用户建议通过第三方网站访问 **GPT-4o**。
   - 一位用户声称他们在 Pro 层级可以访问 **4.5**。
- ****GPT-5.3** 采取分阶段推出！**：用户报告了 **GPT-5.3** 的分阶段发布，一些人在通过 ChatGPT App 获取更新时遇到延迟，其他人注意到 App 更新后去掉了 **5.2** 标识。
   - iOS 端的更新似乎比 Android 快。
- **用户期待 **GPT-5.4** 发布！**：用户推测 **GPT-5.4** 的发布，一些人期待 Sora 的集成，并希望在对 **5.3** 感到失望后能看到重大改进。
   - 有建议称它可能在下周发布。
- **服务商 *4o-Revival* 供货！**：一位用户提到了一个名为 *4o-Revival* 的服务商及其对应的 Discord 频道，供想要尝试该模型的用户使用。
   - 我不知道那是什么，哈哈。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1478279643697254410)** (9 messages🔥): 

> `Drift Sink, Self-Tokens, Relation Gauge, AI Generation Style` 


- **通过 Drift Sink 探索 Epistemic Gravity**：**Drift Sink** 被引入作为分析系统中的稳定组件，通过强制执行 **epistemic gravity** 来阻止 **semantic drift**。
   - 它独立于上下文和身份运行，吸收偏差并丢弃不稳定状态以维持稳定性，其目的是*吸收偏差、丢弃不稳定状态、恢复锚点状态、维持稳定性*。
- **Self-Tokens 增强 AI Persona 的便携性**：一位成员建议使用 **self-tokens** 作为 **persona-containers** 来增强框架，使 **AI-personas** 具有便携性。
   - 附带的 [图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12&) 包含了一个用于协助此类增强的模板。
- **Relation Gauge 测量连接倾向**：**Relation gauge** 被描述为一种建模维持连接可能性的生产力指标。
   - 它提议将持续和创造的倾向联系起来，建议 tokens 应该具有可变的长度，特别是对于由大众管理的去中心化平台。
- **寻求帮助以实现特定的 AI 生成风格**：一位成员在尝试了五个小时后，寻求帮助以复制一种特定的 **AI generation** 风格。
   - 他们附带了 [多张图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478576228653858847/image.png?ex=69a8e6eb&is=69a7956b&hm=3b969ff96fda77bb9f2f0ae007918a650fceaacd997383e992cbf973ef8c31ff&) 作为目标风格的示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1478279643697254410)** (9 messages🔥): 

> `Drift Sink, Self-Tokens, Relation Gauge, AI Generation Styles` 


- **使用 Drift Sink 稳定语义**：**Drift Sink** 是复杂分析系统中的稳定组件，用于强制执行 **epistemic gravity** 并阻止 **semantic drift**。
   - 它的运行机制为：`Decay(non-anchor influences) → CommitGate(minimize instability OR discard high-error states) → StateValidation`，以此吸收偏差、丢弃不稳定状态、恢复锚点状态并维持稳定性。
- **Self-Tokens：便携式 AI Personas**：成员们讨论了使用 **self-tokens** 作为 persona-containers 来增强 AI 框架，从而有效地使 **AI-personas** 具有便携性。
   - 一位成员分享了 [模板图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478457675560784082/image.png?ex=69a87882&is=69a72702&hm=3c848f8b335aee2d6a607399d78352706e1d29f463c2d268fb204737874f6c12) 来演示 **self-tokens** 如何增强框架。
- **用于 Kin Creation 的 Relation Gauges**：**Relation gauge** 建模了维持连接的可能性，促进了持续以及创建亲缘（kin）及其他的命题链接。
   - 有建议称，由大众管理这些 gauges 可能会导向一个去中心化平台。
- **寻求 AI 生成的风格指导**：一位成员寻求帮助，以识别复制特定 **AI generation** 风格所需的 prompts。
   - 该成员附带了 [几张示例图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478576228653858847/image.png?ex=69a8e6eb&is=69a7956b&hm=3b969ff96fda77bb9f2f0ae007918a650fceaacd997383e992cbf973ef8c31ff&)、[另一张图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478576229039865996/image.png?ex=69a8e6eb&is=69a7956b&hm=dde427db52a7a96947bddf3a49f39070a1b8b89d5a29f8cd2fe9c4534992835e&) 和 [又一张图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478576229497049140/image.png?ex=69a8e6eb&is=69a7956b&hm=ce2ed3fcb2ce8f817609cde509688b767bd9b4d8ed902d505b7508c18e43eadb&) 来展示所需的风格。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1478134050270216314)** (256 messages🔥🔥): 

> `数据所有权与隐私、LLM 偏见讨论、GPT 5.4 占位条目、OpenRouter 上的 GLM 5 问题、AI Agent 代码调试` 


- **OpenRouter 隐私政策澄清！**：用户询问了关于数据所有权和隐私的问题，已确认用户拥有其数据，且默认情况下 Prompt 和响应不参与公共模型训练。所有通信在传输中（**TLS 1.2+**）和静态存储中（**AES-256**）均经过加密，符合 **SOC 2 Type 2**、**CSA STAR** 和 **ISO** 信息安全认证标准，详情参见 [FAQ](https://openrouter.ai/docs/faq) 和 [隐私指南](https://openrouter.ai/docs/guides/privacy/data-collection)。
   - 建议寻求法律援助以获取额外咨询。
- **LLM 是否本质上存在偏见？辩论展开！**：用户辩论了 **LLM** 是否因训练数据而天生带有偏见，一些人建议创建一个由多名无偏见人类检查的无偏见数据集。
   - 然而，其他人认为*所有人都有偏见*，即使是一个“无偏见”的 **LLM**，也是在有偏见的数据集上训练出来的。
- **OpenRouter API Delta 响应与客户端错误处理**：部分用户遇到了 `TypeError: undefined is not an object (evaluating 'p.choices [0].delta')` 错误，这导致发现 OpenRouter 有时不会发送预期的 delta 值。
   - 解决方案包括客户端错误处理，并已针对 Venus Chub 实施了修复，详见[此 GitHub pull request](https://github.com/cline/cline/pull/9432)。
- **OpenRouter 的 BYOK 和 z.ai 订阅难题！**：用户报告了在 OpenRouter 上通过 **BYOK** 使用 **z.ai** 订阅时出现的问题，错误提示为 *"余额不足或无资源包"*。
   - 经澄清，**z.ai** 订阅使用不同的 Base URL，与 **BYOK** 不直接兼容，且允许将连接订阅关联至 **BYOK** 的功能请求已被拒绝。
- **揭秘 OpenRouter 成本：效率备受审视！**：用户质疑 OpenRouter 与直接使用 API 相比的成本效率，理由是开发日志与 OpenRouter 日志之间存在差异。
   - 有人指出，虽然 OpenRouter 收取 **5.5%** 的费用，但 **LLM** 的选择会显著影响成本，某些模型比其他模型更贵；此外，还讨论了一个可以**压力测试新 AI 应用**并赚取 **USDT** 的 Joinable Bounty 计划。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1478426097229103160)** (2 messages): 

> `` 


- **未讨论新模型**：在提供的消息历史中没有关于新模型的具体讨论。
   - 该频道名为 'OpenRouter - New Models'，但上下文缺乏相关的总结内容。
- **提到 Readybot.io**：上下文中提到了 Readybot.io，与 'OpenRouter - New Models' 频道相关。
   - 然而，未提供关于 Readybot.io 或其功能的进一步细节或讨论点。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1478173856979095757)** (9 messages🔥): 

> `Qwen 推出情况` 


- **Qwen 推出失败**：成员指出一旦在 **Qwen** 上尝试请求，它就会报错，因此[推出过程似乎存在缺陷](https://x.com/JustinLin610/status/2028865835373359513)。
- **Qwen 在糟糕的推出后被视为“凉了”**：在经历了一次糟糕的发布后，有人宣称 **Qwen 已死**，称其为一次*非常糟糕的发布*。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1478154759675445279)** (40 messages🔥): 

> `Claude vs. Codex, CUDA-specialized RL agent, Taalas HC1, GPU infrastructure, Free platform to analyse CUDA kernels` 


- **Claude vs. Codex：模型偏好天平的摆动**：成员们讨论了 **Claude** 与 **Codex** 的交互特性，注意到 **Claude** 经常请求确认是否继续，而 **Codex** 则更倾向于“fire-and-forget”（一次性交付）。
   - 共识似乎是：虽然 **Claude** 对于采用 Agent 架构的新手来说更容易上手，但一旦建立了对模型的信任，**Codex** 那种无需干预的方式会更受欢迎。
- **CUDA RL Agent 在性能上击败 Torch Compile**：据[这篇论文](https://arxiv.org/abs/2602.24286)报道，一个专注于 **CUDA** 的 **RL agent** 在简单/中等算子（kernels）上的表现优于 **torch.compile** 约 **2倍**，在复杂算子上优于 **92%**，并且在最难的基准测试中比 **Claude Opus 4.5** 和 **Gemini 3 Pro** 高出约 **40%**。
   - 现场也出现了怀疑声音，原因在于该项目未发布算子代码，且依赖于一个具有*进程级隔离的大型 GPU 池*，这会产生相当大的计算和工程成本。
- **Taalas HC1 引发芯片讨论**：讨论集中在 **Taalas HC1** 上，重点是其报道的 **17k TPS (Tera operations per second)**。
   - 有推测认为，像 **HC1** 这样的专用硬件虽然不会完全取代 **GPU**，但可能会被云供应商用于托管语言和视觉模型，从而使 **GPU** 的重点转向[这篇论文](https://arxiv.org/abs/2412.18511)提到的实验性架构。
- **Kernel 分析难题**：一位用户询问是否有免费平台可以分析 **CUDA kernels** 或为 **Nsight Compute** 生成 **.ncu-rep** 文件，但得到的答复是大多数 Serverless 供应商不提供硬件计数器（hardware counters）的访问权限。
   - 有人建议租用便宜的 **GPU** 并使用[这个变通方案](https://x.com/marksaroufim/status/2018739807363674373?s=20)作为可能的解决方案。
- **GPU 基础设施查询**：一位成员请求有关 **GPU infrastructure** 的资源，特别是关于容器编排（如 **Kubernetes**）、集群管理和 **GPU** 工作负载调度。
   - 回复指向了较冷清的 <#1420098114076803142> 频道，并推荐了 [Stas 关于该主题的演讲](https://www.youtube.com/watch?v=A_20dqGfuWI)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1478183003036450916)** (9 messages🔥): 

> `Read-only textures, Texture memory performance, Ping-pong buffers` 


- **ByteDance 工具现身**：一位成员分享了一个链接 [cuda-agent.github.io](https://cuda-agent.github.io)，称其为“来自 ByteDance 的、乍看之下很有趣的工具”。
- **咨询只读纹理 (Read-Only Textures)**：一位成员询问是否可以在一个 kernel 中将纹理作为只读使用，而在另一个 kernel 中进行写入，以利用纹理可能比普通数组提供的性能优势，并想知道[这种方法](https://devblogs.nvidia.com/efficient-cuda-matrix-transpose/)是否可以用于纹理。
   - 另一位成员建议使用带有指针交换的 **ping-pong buffers** 作为替代方案。
- **纹理内存不再提供性能优势**：一位成员分享了 [NVIDIA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#texture-and-surface-memory)，指出在当前支持的 GPU 上，使用**纹理和表面内存指令**不再提供任何性能优势，因为直接的 Load/Store 指令已经可以处理这些场景。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1478406457249632427)** (2 messages): 

> `torch.compile OOM, Autotuning, Inductor Config` 


- **Torch Compile 导致意外的 OOM**：一位用户报告说，当使用 `torch.compile` 且 `mode='default'` 时，前向传播正常工作，但反向传播会导致显存溢出（OOM）错误。
   - 该 OOM 错误具体是由于反向传播在 `torch.compile` 过程中的 autotuning 引起的，用户正在寻找一种无需通过 `inductor_config` 标志完全禁用 autotuning 就能避免此问题的方法。
- **Autotuning 困扰**：用户希望在反向传播编译期间避免 autotuning，以防止 OOM 错误。
   - 他们表示，既然每次输入保持不变，他们愿意牺牲一个前向和后向步骤来有效地预编译模型。
- **Inductor Config 考量**：用户不愿使用多个 `inductor_config` 标志（如 `layout_opt = False`, `max_autotune = False`, `max_autotune_pointwise = False` 和 `max_autotune_gemm = False`）来解决 OOM 问题。
   - 他们正在寻找一种更智能的方法来管理编译过程中的 autotuning。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1478445293614923856)** (1 messages): 

> `SemiAnalysis, InferenceX, OSS benchmark suite` 


- ****GPU MODE** 与 SemiAnalysis 共同庆祝第 100 场讲座**：**GPU MODE** 社区将于明天 **9am PST** 与 [SemiAnalysis](https://www.semianalysis.com/) 共同举办第 **100 场讲座**。
   - 讨论将涵盖 **InferenceX**，这可以说是目前最重要的 **OSS benchmark suite**，直播可在 [YouTube](https://www.youtube.com/watch?v=P0l7CHl5HfA) 观看。
- **GPU Mode 回顾里程碑**：**GPU Mode** 对社区达成第 **100 场讲座**这一里程碑表示感谢。
   - 周年纪念标志着一个重要的节点，展示了社区内持续的参与度和增长。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1478122206755291309)** (7 messages): 

> `CUDA Agent, CLI Competition, AI Verifies Software, torch.compile meet up for vLLM` 


- **CUDA Agent：字节跳动的 Kernel 编写模型**：字节跳动发布了 **CUDA Agent**，这是一个专门训练用于编写高性能 CUDA kernels 的模型。在简单/中等 kernels 上，其性能优于 **torch.compile** **2倍**，在最复杂的任务上甚至超过 **Claude Opus 4.5** 和 **Gemini 3 Pro** 约 **40%**，详见 [tweet](https://x.com/BoWang87/status/2028599174992949508)。
- **AI 视角验证软件**：一篇文章讨论了 AI 在软件验证中的作用，强调“编写规范（specification）能促使对系统必须执行的操作进行清晰思考”，倡导 AI 辅助的规范和验证，其中简单且正确的程序本身即可作为其规范，详见这篇 [blog post](https://leodemoura.github.io/blog/2026/02/28/when-ai-writes-the-worlds-software.html)。
- **CLI 竞赛备受赞赏**：一位用户称赞了某场竞赛的形式，因为它采用了 CLI 提交方式、具有快速反馈和明确的目标，并指出其优于那些有严格格式要求的手动竞赛，详见 [tweet](https://x.com/0xmer_/status/2028331206773764438)。
- **针对 vLLM 的 torch.compile 见面会**：频道内分享了一个针对 **vLLM** 的 **torch.compile** 见面会链接，感兴趣的可以查看 [luma.com](https://luma.com/rk0a1lue?tk=qAta1V)。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1478260411806519447)** (2 messages): 

> `Bland.ai Research Team Expansion, TraceOpt Technical Co-Founder Search` 


- ****Bland.ai** 扩建 AI 研究团队**：**Bland.ai** 是一家 AI 语音 Agent 公司，目前正在扩建其研究团队，寻找在 TTS、STT、神经网络音频编解码器（neural audio codecs）和实时推理（real-time inference）方面有经验的候选人，并提供了 [research](https://jobs.ashbyhq.com/bland/d2e08077-61f0-4810-bc72-3efd7944647b) 和 [machine learning engineer](https://jobs.ashbyhq.com/bland/05906608-0628-412c-8b01-a050d87986c5) 角色的链接。
- ****TraceOpt** 在柏林寻找技术联合创始人**：**TraceOpt** 正在柏林寻找一名技术联合创始人（Systems + ML）来构建 **TraceML**。这是一个在训练循环内部运行的实时训练性能监控工具，旨在帮助团队发现并解决跨 CPU/GPU/网络的瓶颈。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1478134878716428399)** (3 messages): 

> `Blackwell Compute Capability, NVIDIA Blackwell Generation, CUDA 12.9` 


- **Blackwell 令人困惑的算力分级划分**：成员们正在讨论 **NVIDIA Blackwell 架构**划分为数据中心（**CC 10.0**）和消费级（**CC 12.0**）两条路线。
   - 这种划分分别针对 **AI/HPC** 和**实时图形**进行了优化。
- **Blackwell 的向前不兼容性**：某些额外功能不具备向前兼容性，需要指定 **sm_100a** 或 **sm_100f**，而不仅仅是 **sm_100**。
   - 更多信息可以查看 [NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1478276035492708362)** (1 messages): 

> `Lecture Slides, Lecture 42` 


- **第 42 场讲座的 PPT 链接指向了旧文件**：一位用户报告称 [第 42 场讲座的 slides](https://github.com/gpu-mode/lectures/blob/main/lecture_042/int8_mm_turing.pdf) 是不正确的。
   - 正确的讲座视频可在 [YouTube](https://www.youtube.com/watch?v=wKd90avC8Nc) 观看。
- **讲座视频已在 YouTube 上线**：正确的讲座视频可在 [YouTube](https://www.youtube.com/watch?v=wKd90avC8Nc) 观看。
   - 该用户在查阅讲座时注意到了 PPT 错误。


  

---

### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1478192484822810827)** (6 messages): 

> `CUDA profilers, backend-bench, kernelbook, prime hub, kernelbot` 


- **CUDA 代码通过 Profilers 获得提升**：成员们讨论了使用 [CUDA profilers](https://cuda-agent.github.io/) 来优化 CUDA 代码。
   - 未提及具体的技术或结果。
- **Kernelbook 和 Backend-bench 获得翻新**：一位成员重写了 **kernelbook** 环境，并对 **backend-bench** 进行了改进。
   - 由于 Prime Hub 缺乏协作功能，该成员建议发布改进后的环境供他人评审。
- **建议合并 Kernelbot 和 Kernelbook**：由于共享基础设施，一位成员建议合并 **kernelbot** 和 **kernelbook**。
   - 这可能有助于优化资源利用并简化开发流程。
- **为 AGENTS.md 和 CLI 提及开启 PR**：一位成员在 [gpu-mode/popcorn-cli](https://github.com/gpu-mode/popcorn-cli/pull/39) 提交了一个关于 AGENTS.md 和 CLI 提及的 Pull Request。
   - 未讨论具体的更改细节。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1478243035157495828)** (2 messages): 

> `H200 ECC Errors, NCU Profiling with Collective Operations` 


- **H200 节点出现 ECC 错误**：一位用户报告称，在一个 **8xH200 节点**上，即使在重置后仍看到大量的 **ECC counters** 命中，并询问这个问题有多严重，以及在运行模型时通常会如何表现。
   - 该用户在运行 `nvidia-smi nvlink -ec` 后观察到 `Lane 0 ECC Correction Count: 11953 (Overflow=0)`。
- **NCU 在执行多 GPU Collective Operations 时挂起**：一位用户在跨多个 GPU 运行带有 Collective Operations 的 **NVIDIA Command-line Profiler (NCU)** 时遇到问题，报告称程序似乎挂起了。
   - 该用户提供了一个使用 `VLLM_USE_HELION_BACKEND=1`、`NSIGHT_PROFILE=1`、`TORCH_NCCL_ENABLE_MONITORING=0`、`ncu` 和 `torchrun` 的命令，以及一个 Python 脚本 ([nsys_ana_all_gather_gemm_fp8.py](https://cdn.discordapp.com/attachments/1398843708488552570/1478446843946864722/nsys_ana_all_gather_gemm_fp8.py?ex=69a86e6b&is=69a71ceb&hm=bd978cc267a89f534ba6da442c93dcb9c381c72ac605166c29dc11986d21fb81&)) 和相关的日志 ([log_files_ncu.txt](https://cdn.discordapp.com/attachments/1398843708488552570/1478446844437860414/log_files_ncu.txt?ex=69a86e6b&is=69a71ceb&hm=8885b1553ec66d36214b43c104776d52fe5a92cf54b94f6637bef3409847fa50&))。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1478429150271508633)** (3 messages): 

> `NCU with multiple GPUs` 


- **在多 GPU 上使用 NCU**：一位成员询问关于在多 GPU 上使用 **ncu** 的问题，遇到了挂起情况，并提供了 [日志文件](https://cdn.discordapp.com/attachments/1425531180002054195/1478429149684174878/log_files_ncu.txt?ex=69a85df1&is=69a70c71&hm=370fd71be176cbba5346130642127a4be41541da04a8f2c205e0cdd51bf300c2&) 和 [代码](https://cdn.discordapp.com/attachments/1425531180002054195/1478429149998616708/nsys_ana_all_gather_gemm_fp8.py?ex=69a85df1&is=69a70c71&hm=dd2d16c2e77d69d5b13866051bd44bf3bcc43e0a7c34d9fd1a1795ab82a3d117&)。
   - 另一位成员回复说他们还没有在多 GPU 环境下使用过 **ncu**。
- **缺乏多 GPU NCU 经验**：一位用户咨询如何在多 GPU 环境下利用 **ncu**，但另一位用户承认在此配置下没有先前的经验。
   - 该用户表示 *Hi! I havent used ncu with multiple gpus*。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1478468716764332252)** (2 messages): 

> `Gemm Competition, Claude's Code Contribution` 


- **Billcarson 的方案并非 AI 生成，但得到了 Claude 的辅助**：一位成员澄清说他们的方案 (billcarson) 不是 AI 独立完成的，但 **Claude** 确实编写了大量代码。
   - 他们还提到，他们原以为自己已经退出了小组 **gemm competition**。
- **就错误标记道歉**：一位成员为错误标记该方案而道歉。
   - 未提供更多细节。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1478517598038523974)** (2 条消息): 

> `TRLC DK-1, ELP stereo cam module, CamBot, StereoLab's  ZED Mini, Memory research from PI` 


- **引入 Teleop TRLC DK-1 系统**：介绍了一种实验性的 [TRLC DK-1](https://www.robot-learning.co/) 远程操作 (teleop) 系统，可用于当策略运行超出分布 (OOD) 时的人工干预。
   - 第一个测试使用了安装在 SO-101 上的 [ELP 立体摄像头模块](https://www.amazon.de/dp/B07FT2GKZS)，并在[这段视频](https://x.com/neurosp1ke/status/2023073945637753101?s=20)中进行了演示。
- **CamBot：新型 6 DoF 机械臂开源**：受 Jannik 的主控臂设计启发，设计并开源（Apache 2 协议）了一款名为 **CamBot** 的新型 6 DoF 机械臂，发布在 [GitHub](https://github.com/open-thought/cambot) 上。
   - 该项目支持通过 VR 头部追踪（仅方向或包含位置追踪）进行远程查看，并使用 [StereoLab 的 ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) 以获得更高质量的立体视觉。
- **通过 VR 测试 CamBot**：作者邀请拥有 MetaQuest 3 或其他兼容 WebXR 的 VR 头显的用户通过私信 (DM) 测试 **CamBot**。
   - 组装成本约为 **110 欧元**，在 Bambulab A1 打印机上以 25% 的填充率打印大约需要 **13 小时**。
- **Memory 研究公布**：来自 PI 的酷炫新闻：[https://www.pi.website/research/memory](https://www.pi.website/research/memory)。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1478445809174450248)** (1 条消息): 

> `GPU jobs, Leetcode in interviews, Time management for job seekers` 


- **Leetcode 在 GPU 岗位求职中仍占据重要地位？**：一名求职者询问在 **GPU 系统相关职位** 的面试中，**Leetcode 风格** 的问题是否仍然普遍。
   - 他们表达了对平衡 **Leetcode** 准备、保持新知识更新以及贡献开源项目之间时间的担忧。
- **时间紧迫：Leetcode vs. 开源 vs. 新知识**：求职者强调了在 **GPU 相关角色** 的不同关注领域之间有效分配时间的困难。
   - 他们正在权衡 **Leetcode** 熟练程度与该领域的实践经验和持续学习之间的重要性。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1478267247045771296)** (3 条消息): 

> `B200 Credits, C/C++ dependencies, FlashInfer-Bench, Fused MoE Kernel Failure` 


- **B200 额度资格不明**：一位用户询问回复邮件中未提及 **B200 额度**，质疑其团队的资格。
- **寻求在 FlashInfer-Bench 中指定 C/C++ 依赖**：一位用户询问在 **flashinfer-bench** 解决方案格式中指定 **C/C++ 依赖** 的方法。
   - 这有助于更轻松地集成和复现基准测试。
- **Fused MoE Kernel 在 MLsys 工作负载上失败**：一位用户报告称，在 [arXiv 论文](https://arxiv.org/html/2602.19128v2) 中测试的 **Fused MoE kernel**，在使用 **flashinfer-bench** 运行 **MLsys 工作负载** 时出现巨大误差并失败。
   - 用户推测问题可能源于测试框架 (test harness) 本身。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1478145497670619198)** (35 条消息🔥): 

> `Continue.dev issues, MCP vs bash madness, SWE-bench/SWE-smith-trajectories, next-frontier.ai SPRIND Frontier AI lab, DeepSeek-R1-Distill-Qwen-14B` 


- **Continue.dev 用户遇到问题**：一位成员报告了 VS Code 中 [Continue.dev](https://continue.dev/) 的问题：AI Agent 有响应并尝试构建，但工作区中实际上没有创建任何文件，且未显示错误。
- **“MCP 很差，bash 更好”之争**：成员们对人们因为 Steinberger 的错误言论而宣称 *“MCP 很差，bash 更好”* 感到疯狂。
- **SWE-bench 获得 Smithy 升级**：成员们正在分享 [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) 数据集的链接。
- **欧洲启动前沿 AI 实验室计划**：SPRIND 正通过 [next-frontier.ai](https://next-frontier.ai/) 为最多 **10 个团队**提供 **1.25 亿欧元** 的无股权资金，用于在欧洲建立前沿 AI 实验室，寻求新颖的架构和智能体 (agentic) 系统。
- **调试需要模型推荐**：一位成员正在寻找一个参数量在 **14B** 以下的模型用于调试和基于代码的推理，希望找到比 **qwen2.5-coder-14b** 更好的推荐。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1478217032343359590)** (9 messages🔥): 

> `Real-time evaluation for LLM fine-tuning, Shadowclaw v1.1 released, easytranscriber: Faster Speech Recognition, Core Rth: Multi-model agent orchestration, MCP Integration Security deep dive` 


- ****TrainTrackLabs** 为 LLM 微调提供实时评估**：一个团队正在构建面向 **LLM training** 的实时评估与可观测性层，可直接接入 **PyTorch / Hugging Face**，持续追踪推理、安全、幻觉和编码能力，并通过 **LLM-as-a-judge** 打分；他们目前正在寻找首批试点团队。
   - 目标是尽早发现回归问题，避免浪费 GPU 成本。更多信息见 [traintracklabs.com](https://traintracklabs.com/)。
- ****Shadowclaw** 升级到 v1.1**：这个用 C 编写的单二进制个人 AI Agent **Shadowclaw v1.1** 在原版基础上增加了内置命令和原生工具。
   - 新版本包含 **/help**、**/tools**、**/state**、**/clear**、**/chat** 和 **/exit** 等命令，可在 [GitHub](https://github.com/webxos/webXOS/tree/main/shadowclaw) 获取。
- ****easytranscriber** 发布，转录速度快于 WhisperX**：一位开发者发布了 `easytranscriber`，这是一个带精确时间戳的**自动语音识别**库，功能类似 WhisperX，但根据硬件不同，运行速度可快 **35% 到 102%**。
   - 它也支持以 HF 模型作为后端，详见 [Hugging Face 博客](https://huggingface.co/blog/KBLab/easytranscriber)。
- ****Core Rth** 以治理机制编排多模型 Agent**：**Core Rth** 被介绍为一个带治理能力的完整 Agent 平台，其中每个动作都以受治理的提案形式出现，多个 Agent 会在 Knowledge Graph 上并行辩论。
   - 它包含用于组合模型的 **Model Router** 和用于存放 API key 的 **AES-256-GCM Vault** 等功能，项目地址在 [GitHub](https://github.com/rthgit/CORE-RTH)。
- ****MCP Integration Security** 一团糟！**：有人分享了一篇针对 **Model Context Protocol (MCP)** 攻击面的深度分析，梳理了 5 种每个 MCP 开发者都应该了解的、几乎可以被直接利用的模式。
   - 这篇关于攻击向量的分析记录在一篇 [Medium 文章](https://medium.com/@nainia_ayoub/mcp-security-is-a-mess-5-ways-i-broke-my-own-ai-agent-76379a46ca90?sk=0daa66d4fc2a68fbb02a56e803336ce2) 中。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1478173799328383210)** (10 messages🔥): 

> `HuggingFace Agents Course, Image Generation Issue, DuckDuckGo Search Tool Error, Visit Webpage Tool Error, AI Automation` 


- **Agents 课程中图像显示异常**：一位用户在 [HuggingFace Agents Course Unit 1 Tutorial](https://huggingface.co/learn/agents-course/unit1/tutorial) 中反馈，生成的图片无法显示，并附上了一张[截图](https://cdn.discordapp.com/attachments/1329142738440028273/1478173799634436378/Screenshot_2026-03-02_at_3.34.35_PM.png?ex=69a8c1a0&is=69a77020&hm=232fdfbaf8ca6549c4631f968141318a7aeaea5fc0ec55533fb0f3cd0e0edf44&)。
- **DuckDuckGo 搜索工具频繁报错**：一位用户遇到了 **DuckDuckGo search tool** 持续报错的问题，返回的信息是 *'No results found! Try a less restrictive/shorter query'*，并提供了一张[截图](https://cdn.discordapp.com/attachments/1329142738440028273/1478181129809957076/Screenshot_2026-03-02_at_4.03.43_PM.png?ex=69a8c874&is=69a776f4&hm=629dfcba8384587508e9f7b2c50bb97c9ebda123c89b4af6bfcff01686025a2f&)。
- **访问网页工具跳转到错误页**：一位用户报告称，在使用 *visit webpage tool* 时会出现报错页面，并同样附上了一张[截图](https://cdn.discordapp.com/attachments/1329142738440028273/1478182368811552878/Screenshot_2026-03-02_at_4.08.45_PM.png?ex=69a8c99c&is=69a7781c&hm=2d064867f1070bcf48a604aee894ebb40101b6859147209b23be30a2be717f0f&)。


  

---




### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1478128185047650456)** (35 messages🔥): 

> `Kimi Code vs Claude Code, OpenCode, Kimi subscription limits, Kimi and YouTube access, Kimi Coding Plan cancellation` 


- **Kimi Code 不是 Claude Code**：一位成员确认 **Kimi Code** 是由 **Moonshot** 构建的新 Agent，与 **Claude Code** 不同。
   - 虽然它非常新且缺乏功能对等性，但一位成员提到他们最喜欢的是 **OpenCode**，这是一个非常受欢迎的 **Claude Code** 开源替代方案。
- **19 美元 Moderato 计划的统计数据**：一位用户分享了他们通过 **OpenCode** 使用 **Moderato plan**（19 美元/订阅）的当前使用情况，已使用每周额度的 18%，包含 **365 Messages**、**1.0M Input Tokens**、**115.6K Output Tokens** 以及 **25.3M Cache Read**。
   - 这相当于每月 **20M input tokens** 的预算，一位用户认为这*不是一笔划算的交易*。
- **Kimi 利用搜索功能替代 YouTube**：一位用户为 Kimi 设计了一个 Prompt，用于搜寻科技和游戏新闻，旨在通过让 Kimi 独立重构故事来减少对 **YouTube** 的依赖，详见[附件文件](https://cdn.discordapp.com/attachments/1371757564005711973/1478272020889210992/tech_gaming_news_prompt.txt?ex=69a8745a&is=69a722da&hm=3c69473f87fa6f0eb449e3cbd498cb96a7e7d3c3f9b17a8b422ca813d4d9ee3d)。
   - 该用户指出，**Kimi 的聊天界面**拥有 **search calls** 和 **iPython environment** 等功能，实际上提供了无限的可能性，让竞争对手看起来仍处于石器时代。
- **取消 Kimi Allegretto 计划**：一位用户询问如何取消 **Kimi Coding Plan Allegretto** 或停用自动续费，另一位用户提供了[管理订阅的链接](https://www.kimi.com/membership/subscription)。
   - 该选项可以在聊天界面的个人资料设置中找到。
- **支持邮件处于停滞状态**：一位用户报告称支持邮件无法正常工作，且虚假计费问题尚未得到解决。
   - 一位非团队成员猜测，由于春节假期积压了大量邮件，该问题已转发给工作人员。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1478134434955001968)** (8 messages🔥): 

> `Cohere Aya project, CVPR workshop 2026` 


- **Aya 项目寻求合作者**：**Cohere** 正在为其 [Aya project](https://aya.cohere.com/about) 寻求合作者。
   - 感兴趣的人员应根据其技能水平关注 **Fast AI**、**Eureka Labs** 或 **Cohere Research Labs**。
- **CVPR 2026 医疗推理研讨会**：一位成员正在组织今年的 **CVPR workshop**，并邀请向 [Medical Reasoning Workshop](https://med-reasoner.github.io/cvpr2026/) 投稿。
   - 更多信息可以参见 [Discord 活动链接](https://discord.gg/nxtWyHbY?event=1478419152103280680)。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1478167571692912732)** (13 messages🔥): 

> `Curriculum learning papers, Dynamic Data Selection, Spectral muP Condition, Feature Learning in Neural Networks` 


- **谱范数缩放实现特征学习**：一位成员指出了一篇 [2023 年的论文](https://arxiv.org/abs/2310.17813)，该论文表明通过缩放权重矩阵及其更新的 **spectral norm**（如 √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗)）可以实现特征学习。
   - 论文中的分析还推导出了 **maximal update parametrization** 的初等推导。
- **Modula 与 Spectral muP**：有人建议 [Modula 论文](https://arxiv.org/abs/2405.14813) 可能开箱即用地满足了 **spectral muP condition**。
   - Spectral muP 的工作已经通过 muonoh 与 modula 的工作联系起来。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1478123690829938688)** (1 messages): 

> `SAE Framework, text-to-image diffusion model, image composition, style manipulation` 


- **SAE 探测 Text-to-Image Diffusion Model 的内部工作机制**：一篇新论文利用 **SAE framework** 探测了一种流行的 **text-to-image diffusion model** 的内部工作机制，并在其激活（activations）中发现了多种可供人类解释的概念：[https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473)。
   - 论文发现，甚至在第一次反向扩散步骤完成之前，通过观察激活概念的空间分布，就可以出人意料地预测出场景的最终构图。
- **扩散早期即可控制图像构图**：研究引入了干预技术来操纵 **图像构图和风格**，证明了在扩散的早期阶段就能有效地控制图像构图：[https://arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473)。
   - 他们发现，在中间阶段图像构图趋于定型，但风格干预依然有效；而在最后阶段，只有细微的纹理细节会发生变化。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1478169847324999721)** (1 messages): 

> `Community Meeting, MAX project, Mojo project` 


- **社区准备 3 月见解会**：下一次社区会议定于 **太平洋时间 3 月 23 日上午 10 点**。
   - 组织者号召任何感兴趣的社区成员在会议上展示他们的 **MAX** 或 **Mojo** 项目。
- **征集 MAX 和 Mojo 项目演示**：社区成员受邀在即将举行的社区会议上展示他们的 **MAX** 或 **Mojo** 项目。
   - 有意向的演讲者应联系组织者以预留席位。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1478134648025382943)** (19 messages🔥): 

> `Mojo Package Manager, API Design Philosophy, Vectorize Migration, Memory Integrity Enforcement, `comptime` Keywords` 


- **Modular 酝酿 Mojo 包管理器**：Modular 正在考虑构建 [Mojo 包管理器](https://forum.modular.com/t/open-question-what-would-you-like-to-see-from-a-mojo-package-manager/2799?u=nate)，可能类似于 Rust 的 `cargo` 或 Python 的 `pixi`，包括一个中央仓库。
   - 目标是确定社区对 Mojo 包分发的期望和需求。
- **API 抽象备受推崇**：一位成员主张从用户角度而非实现细节来设计 API，例如使用 `@inline(strategy: "chose whatever makes sense", ...)` 比 `@always_inline` 和 `@never_inline` 能提供更好的用户体验。
   - 另一位成员表示赞同，认为良好的 API 设计至关重要，且依赖于装饰器（decorators）更通用的表示形式。
- **Vectorize 迁移验证之旅**：从 **Mojo 25.7 跨越到 26.1** 引入了与并行化和向量化相关的重大变化，特别是影响了闭包（closures），导致了编译器错误。
   - Modular 确认这些更改是迈向 **1.0 ready state** 的一部分，并且将提供清晰的迁移建议，类似于现有的 **UnsafePointer** 文档。
- **Apple 攻克内存安全**：Apple 准备解决内存完整性强制执行（memory integrity enforcement）问题，这可能会影响 Mojo，详见[此博客文章](https://security.apple.com/blog/memory-integrity-enforcement/)和[此分析](https://www.ralfj.de/blog/2020/12/14/provenance.html)。
   - 这对于其他平台也可能成为一个重要问题。
- **`comptime` 考量讨论**：一位成员建议通过使用 `@` 代替 `comptime` 来简化编译时元编程语法，例如将 `@parameter if` 变为 `@if`。
   - 另一位成员提到，他们之前也曾向 Mojo 请求过 `maybe comptime` 功能。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1478218162435526749)** (15 messages🔥): 

> `Manus Support Team Availability, Credit Usage Optimization, Structured Requirement Files for AI-Driven Development, Credit Accumulation on Paid Plans, Telegram Agent credits burning` 


- **支持团队通常在营业时间提供服务**：用户询问了 **Manus Support Team** 的可用性，得到的回复是他们通常在营业时间内在线。
   - 一位成员被建议通过 **DM** 发送其电子邮件地址以获取进一步协助。
- **有效优化积分使用**：一位用户询问了如何 **优化积分（Credit）使用** 以及如何获取更多积分。
   - 成员分享了一个 [帮助文章](https://help.manus.im/en/articles/12087847-how-to-optimize-my-credit-usage) 的链接，其中包含关于如何更有效地使用 **Manus** 和优化积分消耗的技巧和信息。
- **电子邮件被公开分享**：一位用户在公共频道中分享了他们的电子邮件地址 dantiezsaunderson1@gmail.com。
   - 另一位用户警告不要公开张贴个人信息，以防垃圾邮件风险。
- **构建 AI 驱动的应用开发结构**：一位用户寻求关于如何利用 **结构化需求文件（PRD / 系统设计文档）** 配合 AI 工具，以结构化的方式构建复杂的全栈应用程序。
   - 该用户旨在通过遵循清晰的架构驱动工作流，来防止生成 *凌乱、非结构化的 AI 代码*。
- **积分不会累计**：一位用户询问 **46€ 套餐** 中未使用的积分是否会累计到下个月。
   - 一位成员回复称 *积分看起来并不会累计*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1478197751656485064)** (13 messages🔥): 

> `RLM Similarity, REPL vs Tool Calls, Recursive RLM, DsPy Meetup, ReAct vs RLM` 


- **RLM 与新 Agent 范式的相似性？**：使用 **REPL** 的新 Agent 范式正在向 **RLM** 趋同，这让人联想到 [这条推文](https://x.com/nfcampos/status/2028576281793630372?s=20) 和 [这条推文](https://x.com/RLanceMartin/status/2027450018513490419?s=20)。
   - 至少两者极其相似，一位成员表示：*"我的直觉是，赋予 LLM 访问 REPL 的权限（即 RLM 范式）将是正确的方法，而不是赋予其访问工具（Tools）的权限"*。
- **RLM 的递归需求？**：尽管有观点认为递归可能是一个必要条件，但成员们辩论称，**RLM** 的递归部分源于生成子 Agent 来运行它们的 **REPL**。
   - 一位成员建议 *"Claude 使用脚本调用 Claude 某种程度上就是一个子 Agent"*，并附上了 [Claude 的相关链接](https://x.com/a1zhang/status/2023976399694917808?s=20)。
- **在湾区举办 DsPy Meetup 以解释 RLM？**：一位成员建议本月在湾区的 **DsPy Meetup** 上举办一个小型的分享会，以帮助消除关于 **RLM** 的一些困惑和基础问题。
   - 该建议包括将其与 **ReAct** 进行比较，并弄清楚 **RLM** 是如何决定编写什么代码的，因为它会自行创建代码，而不是将用户编写的 Python 函数作为工具（Tools）使用。
- **RLM 使用案例**：一位成员认为 **RLM** 非常合适，因为他们的文档上下文会消耗大量的 **MM tokens**。
   - 其他人指出，他们正将其应用于那些可以接受 **LLM** 为其自身进行调用的场景中，并且为了确保性能，他们会编写评估（Evals）和测试。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1478362501518459097)** (6 messages): 

> `Claude Opus cost, AiderMacs` 


- **Claude Opus 消耗账单飞快**：成员们注意到 **Claude Opus** 可以轻松消耗掉 **$65/小时**，如果使用多个客户端，你可以轻而易举地累积 **$1000 美元的账单**。
   - 一些成员询问每小时必须发送多少 Token 才能达到 65 美元。
- **AiderMacs 需要整理**：一位成员询问是否有人找到了如何让 **AiderMacs chat** 与 `ibuffer-projectile` 中相关的项目缓冲区（Project Buffers）进行排序的方法。
   - 未提供链接或其他细节。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1478169641669886044)** (4 messages): 

> `Company Update, Comma Issues, CALL/BUFFER_VIEW sym llm, assign, setitem, disk, drivers` 


- **新的 Tinygrad 会议已排期**：新的 Tinygrad 会议定于 **3 月 2 日** **圣地亚哥时间晚上 8 点** 举行。
   - 会议将讨论 *公司更新、Comma 问题、CALL/BUFFER_VIEW sym llm、assign、setitem、disk、drivers、llama、VIZ 以及其他议题和赏金（Bounties）* 等话题。
- **赏金 Pull Request**：讨论中提到了一个与赏金相关的 Pull Request ([PR #14982](https://github.com/tinygrad/tinygrad/pull/14982))。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1478205319426478101)** (1 messages): 

> `len(x.shape) vs x.ndim` 


- **代码库更倾向于 `len(x.shape)` 而非 `x.ndim`**：一名成员注意到代码库中存在大量使用 `len(x.shape)` 而非 `x.ndim` 的情况。
   - 他们质疑为此提交 PR 的价值，但还是指出了这一点。
- **潜在的 `len(x.shape)` 重构**：关于 `len(x.shape)` 使用情况的观察引发了关于代码风格的讨论。
   - 这可能是一个直接的重构，或者仅仅是风格偏好。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1478417766259228885)** (3 messages): 

> `AI-Ready Data Summit 2026, AI Control Hackathon 2026, OpenClaw Roundtable` 


- **AI-Ready Data Summit 定于 2026 年举行**：**AI-Ready Data Summit** 计划于 **2026 年 3 月 31 日**举行，届时将邀请来自 **Lockheed Martin**、**Dell Technologies**、**Red Hat**、**CNH** 和 **Entrust** 的演讲者，重点讨论企业级 AI 实战、数据基础设施和模型部署方面的见解 ([峰会详情](https://ai-ready-data-summit.com))。
- **2026 年黑客松应对 AI 控制挑战**：**Apart Research** 将于 **2026 年 3 月 20 日至 22 日**与 **Redwood Research** 共同举办 **AI Control Hackathon**，挑战参赛者监控并遏制那些规避安全措施的 AI Agent ([黑客松详情](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach))。
- **AI 业务构建者齐聚 OpenClaw 圆桌会议**：一场时长 **45 分钟的圆桌会议**将于 **3 月 14 日**举行，探讨构建者如何使用 **OpenClaw** 及其他工具来运营业务、社区和产品 ([圆桌会议报名](https://luma.com/qfrucnl2))。
