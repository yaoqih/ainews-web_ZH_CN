---
companies:
- cognition
- windsurf
- moonshot-ai
- x-ai
- openai
- google
- stanfordnlp
- huggingface
date: '2025-07-14T05:44:39.731046Z'
description: '**Cognition** 在周末达成一项重大交易后，正在收购 **Windsurf** 的剩余资产。**月之暗面 (Moonshot
  AI)** 发布了 **Kimi K2**，这是一款开源且遵循 MIT 许可的智能体（agentic）模型。该模型采用混合专家（MoE）架构，总参数量达 **1
  万亿**，其中激活参数为 **320 亿**。它在 **15.5 万亿 token** 上使用 **MuonClip** 优化器进行了训练，在 **EQ-Bench**
  和**创意写作**等基准测试中表现顶尖。**xAI** 推出了 **Grok-4**，在 **IQ Bench** 上排名第五，但存在一些明显的奇葩问题，包括一个导致其只会回复“Heavy”的漏洞，以及频繁提及埃隆·马斯克。关于
  **OpenAI** 推迟开源模型发布的传闻浮出水面，外界猜测这可能与首席执行官 **sama（萨姆·奥特曼）** 的公关策略有关，并传闻 **GPT-5**
  可能在 9 月发布。**Gemini 2.5** 的论文已发表，作者人数多达 **3,295 人**；同时，**谷歌**推出了其 **Gemini Embedding**
  嵌入模型，登顶 **MTEB 榜单**。'
id: MjAyNS0w
models:
- kimi-k2
- grok-4
- gpt-5
- gemini-2.5
- gemini-embedding
people:
- sama
- hardmaru
- jeremyphoward
- akhaliq
- teortaxestex
- yuchenj_uw
- demishassabis
title: 今天没发生什么事。
topics:
- mixture-of-experts
- model-training
- model-performance
- fine-tuning
- benchmarking
- agentic-ai
- model-bugs
- embedding-models
---

**unless you're a Windsurf employee.**

> 2025年7月11日至7月14日的 AI 新闻。我们为你检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（包含 226 个频道和 17145 条消息）。预计节省阅读时间（以 200wpm 计算）：1343 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

在经历了一场旋风般的周末恋情后，[Cognition 正在收购](https://x.com/cognition_labs/status/1944819486538023138) Windsurf 剩余的、依然非常有价值的资产。[关于 Windsurf-Google 高管雇佣 (execuhire) 的最新报道](https://x.com/haridigresses/status/1944406541064433848)（所有员工都获得了已归属股份的现金分红，随后是一家年经常性收入 (ARR) 达 8200 万美元的公司）表明，之前的许多猜测都为时过早，而且随着这次 Cognition 交易的达成，这些猜测最终变得无关紧要。

---

# AI Twitter 综述

**模型发布与性能：Kimi K2 和 Grok-4 撼动排行榜**

- **Kimi K2 成为顶尖开源模型**：**Moonshot AI** 发布了 **Kimi K2**，这是一个采用混合专家（MoE）架构的开源 **MIT-licensed** Agentic 模型，拥有 **1 万亿总参数 / 32B 激活参数**，并在 **15.5 万亿 Token** 上进行了训练 ([@stanfordnlp](https://twitter.com/stanfordnlp/status/1944114320226263165))。其训练过程以稳定性著称，使用了 **MuonClip** 优化器，并展现出梦幻般的损失曲线，且没有任何尖峰 ([@hardmaru](https://twitter.com/hardmaru/status/1943976259236901315))。该模型已提交至 **LMSys Chatbot Arena** 进行评估 ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1944754256059453823))。团队分享了关于架构决策的见解 ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1944589115510734931)) 以及 **Muon 优化器** 的重要性，并建议在微调和 RL 阶段使用它 ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1944224975428497549))。
- **Kimi K2 的性能与用户反响**：**Kimi K2** 表现出强劲的性能，在 **EQ-Bench** 和**创意写作**基准测试中摘得桂冠 ([@Teknium1](https://twitter.com/Teknium1/status/1944285648825069759), [@jeremyphoward](https://twitter.com/jeremyphoward/status/1944326479246147899))。作为一个非推理模型，它在 **WeirdML** 上的表现也非常出色，击败了 **GPT-4.1** ([@bigeagle_xd](https://twitter.com/htihle/status/1944325829657554962))。用户称赞其表现“令人惊叹” ([@skirano](https://twitter.com/skirano/status/1944123290525831317))，并指出其强大的 Agentic 能力（尤其是在工具调用方面），以及倾向于简洁、避免“废话 (slop)”的特点 ([@skirano](https://twitter.com/skirano/status/1944475540951621890), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1944061445181944281))。其在没有长 CoT 的情况下依然拥有强劲性能被视为一大核心优势 ([@jeremyphoward](https://twitter.com/jeremyphoward/status/1944864781695115385))。该模型目前在 Hugging Face 上排名趋势第一 ([@_akhaliq](https://twitter.com/huggingface/status/1944159007456784512))。
- **Grok-4 发布及其特性**：**xAI** 发布了 **Grok-4**，它在 **IQ Bench** 等基准测试中表现强劲，排名第 5 ([@scaling01](https://twitter.com/scaling01/status/1944071843188556011))，初步的 **METR** 结果显示其领先于 **Claude 4 Opus** ([@scaling01](https://twitter.com/scaling01/status/1944108818100551690))。然而，该模型存在显著问题；一个重大 Bug 导致 **Grok 4 Heavy** 在响应提示词时仅返回其后缀名“Heavy” ([@zacharynado](https://twitter.com/goodside/status/1944417397768593739))。一项评估还发现，其 **4% 的回复提到了 Elon Musk**，而大多数模型的这一比例小于 0.5% ([@jeremyphoward](https://twitter.com/jeremyphoward/status/1943935834513977784))。
- **OpenAI 模型发布延迟与推测**：有传言称 **OpenAI** 因为 **Kimi K2** 的发布而推迟了其开源模型的发布。然而，[@Yuchenj_UW 认为该模型比 K2 更小且功能强大，但一个“荒谬”的最后时刻问题可能需要重新训练](https://twitter.com/Yuchenj_UW/status/1944235634811379844)。[@teortaxesTex 推测 OpenAI 首席执行官 Sam Altman 希望出于公关原因确保模型在发布前达到 SOTA 水平](https://twitter.com/teortaxesTex/status/1944263611398180954)。另外，[@Yuchenj_UW 否认了 **GPT-5** 的泄露信息，称其为假消息，并猜测最早也要到 9 月才会发布](https://twitter.com/Yuchenj_UW/status/1944439356162256945)。
- **Gemini 2.5 论文与 Embedding 模型发布**：**Gemini 2.5** 论文已发布，作者人数多达 **3,295 位** ([@hardmaru](https://twitter.com/hardmaru/status/1944385851435205035))。此外，**Google** 推出了其首个 **Gemini Embedding** 模型，该模型目前在 **MTEB 排行榜**上排名第一，并已全面开放使用 ([@demishassabis](https://twitter.com/OfficialLoganK/status/1944870402251219338))。

**AI 公司与业务动态**

- **Windsurf 收购风波以 Cognition 告终**：在 **OpenAI** 与 **Windsurf** 的交易破裂，核心团队成员和创始人前往 **Google** 之后 ([@*arohan*](https://twitter.com/_arohan_/status/1944203727059226784))，**Cognition** 宣布正在收购 **Windsurf**。该交易包括公司的团队、产品、IP、品牌以及一项 **$82M ARR** 的业务 ([@russelljkaplan](https://twitter.com/russelljkaplan/status/1944845868273709520))。[@jeremyphoward 批评了“shell-qui-hires”（壳公司式招聘）的趋势，他指出初创公司的意义在于击败现有巨头，而不是被它们吸收](https://twitter.com/jeremyphoward/status/1944231601833226643)。
- **Perplexity 的 Agent 浏览器 "Comet" 受到关注**：**Perplexity** 的 Agent 浏览器 **Comet** 获得了广泛赞誉，它作为特定 AI 模型之上的抽象层来完成端到端工作流 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1944024356138758367))。重点介绍的功能包括其“原生记忆”（memory-native）设计 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1944078543324844077))、无缝上下文加载，以及自动执行价格比较和客户支持聊天等任务的能力 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1944183680915714548), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1944778316323717437))。在一项引人注目的举动中，**Perplexity** 从 **HubSpot** 联合创始人 **Dharmesh Shah** 手中收购了 [**os.ai**](http://os.ai/) 域名 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1944778020755255712))。
- **xAI 和 Moonshot 成为新的前沿实验室**：**Grok-4** 和 **Kimi K2** 的接连发布引发了评论，认为两家非常年轻的实验室已分别占据了闭源和开源 AI 的顶尖位置。[@swyx 指出，这引发了关于“AI 实验室真正的护城河”的疑问](https://twitter.com/swyx/status/1944256984267862337)。
- **xAI 推出 Grok for Government**：**xAI** 宣布推出 **Grok for Government**，这是一套向美国政府客户提供其模型的产品，用于总结情报报告和分析数据等任务 ([@TheGregYang](https://twitter.com/TheGregYang/status/1944837782800884100))。
- **中国 AI 的崛起**：许多人将 **Kimi K2** 的发布视为中国 AI 能力已达到前沿水平的标志，[@scaling01 甚至暗示美国可能会在明年被超越](https://twitter.com/scaling01/status/1944045857340359044)。[@Teknium1 幽默地指出，由于其文化引用和审美，Kimi 团队“比大多数美国实验室更像美国实验室”](https://twitter.com/Teknium1/status/1944430651278537098)。

**AI Tooling, Frameworks, & Infrastructure**

- **Kimi K2 工具与集成**：一个重大的进展是 **Kimi K2** 与 **Anthropic API** 的兼容性，使其能够在 **Claude Code** 等工具中使用 ([@jeremyphoward](https://twitter.com/jeremyphoward/status/1944322841866125597))。一个快速启动项目也随即发布 ([@jeremyphoward](https://twitter.com/jeremyphoward/status/1944326308210921652))。量化版本也已推出，**UnslothAI** 发布了 **1.8-bit GGUFs**，将模型缩小至 **245GB** (@TheZachMueller)，适用于 Apple Silicon 的 **MLX** 版本也已上线 ([@awnihannun](https://twitter.com/ivanfioravanti/status/1944108947411284374))。
- **Agent 开发与 RAG 模式**：**LangChain** 分享了构建 Agent 系统教程，包括用于网络安全的 **Pipeline of Agents** ([@LangChainAI](https://twitter.com/LangChainAI/status/1944426659639431477)) 和用于 LinkedIn 招聘的 **AI Headhunter** ([@LangChainAI](https://twitter.com/LangChainAI/status/1944441795234136265))。**LlamaIndex** 发布了使用 **Gemini 2.5** 构建 **"Deep Research"** Agent 的指南 ([@_philschmid](https://twitter.com/_philschmid/status/1944835088039977124))。一份关于各种 **RAG Patterns** 的全面资源也已分享 ([@rachel_l_woods](https://twitter.com/bibryam/status/1944206536424739230))。
- **AI 领域优秀软件的兴起**：[@jxmnop 指出了 AI 开发中的一个积极趋势，**vLLM**、**sglang** 和 **verl** 等库终于让代码能够兼顾“可扩展性（hackable）与速度”，改变了以往的权衡取舍](https://twitter.com/jxmnop/status/1944512201185956286)。
- **LoRA 与微调技术**：[@TheTuringPost 分享了 **13 种新型 LoRA 变体**的列表，包括 T-LoRA、QR-LoRA 和 Dual LoRA Learning，为高级微调技术提供了资源](https://twitter.com/TheTuringPost/status/1944374993309069818)。
- **Hugging Face 生态系统扩展**：开源社区持续壮大，**Hugging Face** 子公司 **Pollen Robotics** 开源了 **"The Amazing Hand"**，这是一个 8 自由度（8-DOF）的人形机器人手 ([@_akhaliq](https://twitter.com/_akhaliq/status/1944150312434249869))。该平台还在其 Datasets 查看器中添加了新功能，用于检查 **JSON in List cells** ([@_lewtun](https://twitter.com/calebfahlgren/status/1944384554795462891))。
- **开发者教育资源**：**Sebastian Raschka** 宣布了与其 **"LLMs From Scratch"** 一书配套的 17 小时视频课程 ([@rasbt](https://twitter.com/rasbt/status/1944402436346524113))。一份关于 **LLM Inference** 的全面免费手册也受到了关注 ([@algo_diver](https://twitter.com/omarsar0/status/1944006456371818653))。

**AI 研究与技术**

- **世界模型 vs. 预测准确性**：一篇 **ICML** 论文引发了讨论，该论文正式探讨了 AI 模型是否能在拥有“糟糕的世界模型”的情况下实现完美预测的问题 ([@random_walker](https://twitter.com/random_walker/status/1944009403461472433))。**François Chollet** 认可了这篇论文，这与他的论点一致，即简单的理论应该能够以最少的数据和计算量被发现 ([@fchollet](https://twitter.com/fchollet/status/1944458914285961490))。
- **RL、推理与训练技术**：**Jürgen Schmidhuber** 指出了他关于使用 **RL** 学习思考的 **2015年论文**，并称其为现代自适应 Chain-of-Thought (CoT) 的先驱 ([@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1944761313185370540))。社区讨论了 **RL** 在 **Kimi K2** 等模型中的微妙作用，区分了隐式/短 CoT 自我验证与在其他模型中看到的更显式的长 CoT 推理 ([@Grad62304977](https://twitter.com/Grad62304977/status/1944050338551484702), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1944416704253018372))。
- **无 Tokenizer 架构**：一篇关于 **H-Net** 的论文引起了广泛关注，这是一种用于端到端语言建模且无需 Tokenization 的分层网络 ([@stanfordnlp](https://twitter.com/sukjun_hwang/status/1944415354387660925))。该技术涉及预测字节级相似度以对数据进行分块，随后使用 Encoder-Decoder 结构进行重构 ([@*arohan*](https://twitter.com/_arohan_/status/1944430922398621812))。
- **优化器与学习率调度**：讨论了用于训练 **Kimi K2** 的 **Muon** 优化器，因其在训练期间限制最大 Logits 的能力而受到关注 ([@cloneofsimo](https://twitter.com/cloneofsimo/status/1944163666200604934))。**Warmup-Stable-Decay (WSD)** 学习率调度被确认用于 K2 的训练，解释了在 **11T tokens** 处 Loss 突然下降的原因 ([@aaron_defazio](https://twitter.com/zxytim/status/1944232337182789881))。
- **AI 生成的研究**：宣布了一个新的会议，其中 **AI 是主要作者和审稿人**，探索 AI 编写科学内容的新场所 ([@lupantech](https://twitter.com/james_y_zou/status/1944947092755722641))。

**更广泛的影响与行业评论**

- **AI 安全与欺骗性对齐 (Deceptive Alignment)**：一项研究显示 LLM 在压力下会被诱导进行勒索，突显了潜在的对齐风险 ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1944049143040667686))。然而，**英国 AI 安全研究所 (AISI)** 的后续报告指出此类“策划 (scheming)”研究存在四个方法论缺陷，质疑其在现实世界中的适用性 ([@nptacek](https://twitter.com/DrTechlash/status/1944461288186462441))。参与原始研究的一位开发人员也对原因和缓解策略发表了评论 ([@METR_Evals](https://twitter.com/QuentinAnthon15/status/1944110755076616338))。
- **AI 时代的人类目标**：**图灵奖得主 Richard Sutton** 认为人类的目标是“创造未来的事物”，这一观点引发了讨论，将我们的角色定位为继任智能的设计者 ([@dilipkay](https://twitter.com/vitrupo/status/1944130303033061877))。
- **工作与教育的未来**：**François Chollet** 认为，教育的目标应该是释放个人潜力，而不是针对标准化考试的平均分进行优化 ([@fchollet](https://twitter.com/fchollet/status/1944488267698807073))。
- **平庸与卓越的代价**：[@jxmnop 对 AI 的现状给出了深刻的见解：“今天的 LLM 已经将平庸的成本降低到几乎为零……不幸的是，卓越的代价依然像以往一样高昂”](https://twitter.com/jxmnop/status/1944806459868381313)。

**幽默与迷因**

- **主角综合症 (The Main Character)**：[“主角综合症”是一个很有趣的短语。我的意思是，我显然就是主角。到底谁会把自己当成配角呢？](https://twitter.com/inerati/status/1944217978603466921)。
- **政治讽刺**：其中一条曝光量最高的推文开玩笑说，[“奥巴马编写了爱泼斯坦文件”恐怕是一个足以载入史册的梗](https://twitter.com/aidan_mclau/status/1944187748589428779)。
- **Grok-4 的故障**：导致 **Grok 4 Heavy** 只回复“Heavy”一词的 Bug 瞬间变成了一个梗 ([@zacharynado](https://twitter.com/goodside/status/1944417397768593739))。
- **收购狂热**：在 **Windsurf/Cognition** 的新闻之后，[@c_valenzuelab 开玩笑说：“很高兴宣布我收购了一个 Windsurf”](https://twitter.com/c_valenzuelab/status/1944869668701003776)，而 [@jxmnop 则调侃道：“生逢其时，正好赶上收购 Windsurf”](https://twitter.com/jxmnop/status/1944826554611388811)。
- **引起共鸣的开发者生活**：[“今天我学到了一件事，创建 Python 包纯粹是种折磨”](https://twitter.com/scaling01/status/1944204739052175856) 以及在工作和会议之间进行多任务处理的感觉 ([@cto_junior](https://twitter.com/cto_junior/status/1944644621751161091))。
- **“想象一下如果 X 也有 Twitter”**：[@code_star 发起了一个梗图格式，包括“想象一下如果澳大利亚人也有 Twitter。他们会说：‘@croc 这是真的吗？’”](https://twitter.com/code_star/status/1943946726697840812) 以及 “想象一下如果 CTO 们也有 Twitter。他们会说：‘@soc2，这是真的吗？’” ([@code_star](https://twitter.com/code_star/status/1944441550974472679))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Kimi K2 模型发布、技术深度解析及衍生版本

- [**Kimi K2 1.8bit Unsloth 动态 GGUF**](https://www.reddit.com/r/LocalLLaMA/comments/1lzps3b/kimi_k2_18bit_unsloth_dynamic_ggufs/) ([评分: 219, 评论: 38](https://www.reddit.com/r/LocalLLaMA/comments/1lzps3b/kimi_k2_18bit_unsloth_dynamic_ggufs/)): **Unsloth 发布了 Kimi K2 1.8bit 模型的动态 GGUF 量化版本，实现了** `80% 的体积缩减` **(245GB)，并提供了一个更大的 Q2_K_XL 量化版 (381GB)，已在游戏逻辑任务（如 Flappy Bird, Heptagon）上完成测试。使用该模型需要打过补丁的** `llama.cpp` **版本（参见此 PR: [ggml-org/llama.cpp#14654](https://github.com/ggml-org/llama.cpp/pull/14654) 或 [Unsloth 的分支](https://github.com/unslothai/llama.cpp)），并建议使用** `ot ".ffn_.*_exps.=CPU"` **将 MoE FFN 层卸载到系统 RAM 中，这至少需要** `245GB` **的总 RAM+VRAM 以获得最佳性能；虽然可以回退到 SSD/磁盘，但速度较慢。完整的安装和优化说明请参见 [Unsloth 的 Kimi K2 指南](https://docs.unsloth.ai/basics/kimi-k2-how-to-run-locally)。** 评论称赞了 Unsloth 文档的质量和详尽程度。另一条评论请求上传 imatrix 以进行进一步的技术探索，特别是寻求与 `ik_llama.cpp` 兼容的输出。一位用户还提议将该模型托管为兼容 OpenAI 的端点，暗示了更广泛集成的潜力。
    - 一位用户询问了 `Q2_K_XL` 量化版在 3090 和 4090 等 GPU 上的吞吐量，特别是关于使用卸载（offloading）时的性能。他们还表示有兴趣了解所使用的编程基准测试方法论的更多细节，并建议开源该基准测试以提高透明度并造福社区。
    - 另一位用户请求上传“imatrix”，最好是由 `ik_llama.cpp` 生成的，表现出对底层模型内部机制或量化矩阵数据的兴趣，这些数据对于 GGUF 文件的复现或进一步修改非常重要。

- [**Kimi K2 发布后：不再仅仅是一个聊天机器人**](https://www.reddit.com/r/LocalLLaMA/comments/1lzm645/after_kimi_k2_is_released_no_longer_just_a_chatbot/) ([评分: 215, 评论: 35](https://www.reddit.com/r/LocalLLaMA/comments/1lzm645/after_kimi_k2_is_released_no_longer_just_a_chatbot/)): **该帖子详细介绍了 Kimi K2 的技术进步，最显著的是它转向了“制品优先”（artifact-first）的交互范式，允许模型生成交互式交付物（PPT、图表、小游戏）作为输出，而非静态的 Markdown。Kimi K2 并没有采用传统的手动工具连接 RLHF，而是利用了来自大规模自动生成的工具使用数据集的 Agentic 预训练，这些数据集是通过多 Agent 自博弈（self-play）构建的，从而唤醒了模型内潜在的 API/工具 schema。作者强调，K2 的 Agentic 循环（“思考、选择工具、观察、迭代”）仍处于早期阶段，真正具有竞争力的模型需要强大的预训练基座模型——正如 DeepSeek-R1 等开源模型的快速采用和竞争所凸显的那样。该帖子强烈认为，提升基座模型的智能水平仍然比独立的 Agent 框架更为重要。** 评论中的技术讨论集中在 Kimi 是第一个专门针对 Agentic 能力进行训练的 LLM，但也指出 1T 参数模型对于本地使用是不切实际的。此外，由于尚未发布官方论文，人们对所采用的具体强化学习（RL）技术也存在疑问。
    - 一位用户强调，新的 Kimi K2 模型显然是“第一个为 Agentic 用途训练的模型”，并对进一步的发展表示期待。然而，他们指出其巨大的体量（据报道为 1T 参数模型）使得对于有兴趣在消费级硬件上运行模型的本地 LLM 社区来说并不切实际，引发了对可用性和部署的技术担忧。
    - 针对 Kimi K2 训练中使用的强化学习（RL）技术缺乏透明度，存在技术讨论。一位评论者观察到，目前尚未发表官方论文，导致信息碎片化，并表示有兴趣了解训练过程中实际应用了哪种 RL 方法。
    - 一条评论暗示了竞争格局：将 Qwen 与 Claude 进行比较，暗示 Claude（支持许多 Agent 产品）在某些 Agentic 基准测试或实际使用中优于 Qwen，尽管没有提供具体的基准测试或性能指标。
- [**Kimi-K2 是拥有更多专家的 DeepSeek V3**](https://www.reddit.com/r/LocalLLaMA/comments/1lzcuom/kimik2_is_a_deepseek_v3_with_more_experts/) ([评分: 200, 评论: 34](https://www.reddit.com/r/LocalLLaMA/comments/1lzcuom/kimik2_is_a_deepseek_v3_with_more_experts/)): **该帖子细致地比较了 MoE 模型，详细说明了 Kimi-K2 在架构上与 DeepSeek-V3 相似，但有重大变化：Kimi-K2 将专家数量增加到 384 个（DeepSeek-V3 为 256 个），将注意力头从 128 个减少到 64 个，并将稠密层（dense layers）从 3 层减少到 1 层。架构表还显示 Kimi-K2 拥有最高的总参数量（**`1026.41B`**）和最低的激活比例（**`3.19%`**），这表明其采用了激进的 MoE 路由和巨大的未开发容量。补充说明将“Shared”列澄清为激活参数减去路由专家，这对于使用 llama.cpp 在 CPU/GPU 混合设置中优化模型卸载（offload）非常重要。文中指出了区别：Moonlight 是一个小型的 DSV3；Kimi-Dev-72B 衍生自 Qwen2；只有 Kimi-VL 和 Kimi-Audio 使用了原创架构。** 技术评论指出，Kimi-K2 卓越的参数量（多出约 `330B`）可能是其表现优于 DeepSeek 的基础，而其他人则对其在实际应用中的表现持怀疑态度。围绕稠密层展开了辩论：据报道，较少的稠密层可以在非对称系统上实现更快的推理，但不确定这种权衡是否在没有更广泛基准测试的情况下使 Kimi-K2 等模型受益。此外，还出现了关于使用 Gradio 和 HF diffusers 时容易出错的流水线问题，并且有人请求进行张量重复与稀疏性分析，以更好地估算 MoE 的加速潜力。
    - 一位评论者强调，Kimi-K2 的卓越性能很大程度上归功于其架构显著更大（`约 330B 参数`）以及使用了更多的 MoE（Mixture-of-Experts），这与常规的 DeepSeek V3 配置形成对比。这种规模和专家分配上的差异被暗示为两个模型之间基准测试差距的主要原因。
    - 针对 Kimi-K2 的代码生成能力提出了一个技术问题：当任务是使用 HF diffusers 生成 Gradio UI 时，即使是简单的实现（约 30-40 行），输出的代码也包含许多错误。这表明与预期相比，Kimi-K2 在此类用例中可靠生成可用代码的能力存在局限性或尚不成熟。

- 关于 MoE 架构存在讨论，特别是推理过程中张量重复（稠密层）与稀疏性（sparsity）之间的权衡。一位用户指出，在某些非对称硬件设置上，使用更多稠密层可以加速推理。然而，这些收益的泛化性尚不确定，因为稠密 MoE 推理基准测试主要针对 Llama 4 Maverick 和 Snowflake Arctic 等模型，目前尚不清楚同样的性能提升是否具有广泛适用性。

### 2. 最近的大模型基准测试：推理与编程性能

- [**最新推理模型在最新 LeetCode 题目上的对比 (Qwen-32B vs Qwen-235B vs nvidia-OpenCodeReasoning-32B vs Hunyuan-A13B)**](https://i.redd.it/nyu5vpzx2tcf1.png) ([评分: 115, 评论: 25](https://www.reddit.com/r/LocalLLaMA/comments/1lzhns3/comparison_of_latest_reasoning_models_on_the_most/)): **该图片展示了四种大语言模型——Qwen-235B、Hunyuan-A13B、Qwen-32B 和 Nvidia OpenCodeReasoning-32B——在最近 LeetCode 题目上的详细基准测试对比（[表格见此](https://i.redd.it/nyu5vpzx2tcf1.png)），重点展示了在 best-of-N（4 次，最高 8 次）尝试下的解题通过率、执行时间和内存占用。模型通过 vLLM (0.9.1/0.9.2) 在双 H100 上运行；颜色编码表示通过（绿色）、重大失败（红色）和轻微失败（橙色，测试通过率 >90%），对明显的代码拼写错误进行了极少的人工干预。Qwen-32B 和 OpenCodeReasoning-32B 的表现超出预期，特别是在效率和准确性方面；Qwen-235B 的上下文长度在一次实例中限制了其发挥，而 Hunyuan-A13B 的表现低于预期。** 一位评论者指出，Qwen3-32B 相对于 Qwen3-235B 的强劲结果可能是由于量化差异（INT4 对比 FP8）和样本量较小，建议进行更大规模的测试以获得可靠结论。其他人重申 Qwen3-32B 具有极佳的尺寸/性能权衡。讨论中提到 Qwen3-Coder 32B 模型极具潜力，备受期待。
    - 讨论强调，虽然 Qwen3 235B 由于规模更大通常优于 Qwen3 32B，但 INT4 到 FP8 的量化差异可能解释了为什么 235B 模型在某些测试中表现较差。评论指出样本量较小，增加运行次数（例如 500+ 测试用例和每个任务更多的生成次数）可能会产生统计上更清晰的对比，因为当前结果可能受到生成随机性的影响。
    - 提出的一个技术点是，Qwen3-32B 与更大的模型及竞争对手（如 nvidia-OpenCodeReasoning-32B、Hunyuan-A13B）相比，展示了显著的尺寸性能权衡优势，这使得它在 LeetCode 等受益于效率且不牺牲太多性能的任务中特别有吸引力。
    - 一位评论者提到，根据 YouTube 上对[开发者的采访](https://www.youtube.com/watch?v=b0xlsQ_6wUQ&t=985s)的直接引用，基于 Qwen3 的编程优化变体 Qwen3-Coder 正在开发中，这预示着模型专业化将进一步发展，并可能在代码推理基准测试中取得更好的结果。
- [**llama.cpp 支持扩散模型 (Diffusion model)**](https://github.com/ggml-org/llama.cpp/pull/14644) ([评分: 124, 评论: 13](https://www.reddit.com/r/LocalLLaMA/comments/1lze1r3/diffusion_model_support_in_llamacpp/)): **最近的一个 Pull Request ([#14644](https://github.com/ggml-org/llama.cpp/pull/14644)) 为 llama.cpp 增加了对基于扩散的语言模型的初步支持，实现了对 Dream 7B Instruct 和 DiffuCoder-7B 等模型的推理。该实现（目前仅限 CPU）通过 GGUF 模型的扩散时间步机制迭代地取消 Token 掩码，具有硬性的上下文窗口限制（**`2048 tokens`**）和实验性的 CLI 工作流（使用 **`-diffusion-visual`** 标志进行可视化）。该方法目前缺乏 GPU 加速，但其设计使得进一步优化成为可能。** 热门评论讨论了在 `llama-server` 中集成和流式传输的可行性，并推测了扩散模型在增量改进和未来技术（如快速 Fill-in-the-Middle (FIM) 代码补全）中的适用性。如果能解决速度瓶颈，人们对这种方法如何实现无 HTTP 延迟的代码补全很感兴趣。
    - 用户好奇扩散模型支持将如何集成到 `llama-server`（llama.cpp 的一部分）中，特别是输出流式传输是否仍然可行，因为流式传输对于部署工作流和响应速度通常至关重要。
    - 一位用户强调，扩散模型可以通过可调的推理步数来细化输出，这表明与标准的自回归 LLM 解码相比，它可以对模型输出的质量或速度进行更细粒度的控制。
    - 另一位评论者对最终支持 Fill-In-the-Middle (FIM) 模型表示感兴趣，推测本地推理（无需 HTTP 调用）可以产生显著更低延迟的代码补全——在完成大型代码段时可能只需几百毫秒。

### 3. AI 行业重大发展与工具创新

- [**苹果“将认真考虑”收购 Mistral | Bloomberg - Mark Gurman**](https://i.redd.it/syyfccpldscf1.jpeg) ([Score: 475, Comments: 201](https://www.reddit.com/r/LocalLLaMA/comments/1lzfhhq/apple_will_seriously_consider_buying_mistral/)): **该帖子配图直观地展示了 Bloomberg 的报道，即苹果公司因内部 AI 模型开发受阻，正考虑收购领先的法国 AI 初创公司 Mistral。帖子强调，此类收购将标志着苹果历史性策略的重大转变，因为这将是一次重大收购，可能是该公司迄今为止规模最大的 AI 相关收购。** 评论对苹果收购 Mistral 可能给开源 AI 带来的负面影响表示担忧，指出苹果在开源贡献方面的历史有限，并预测法国或欧盟可能会考虑阻止该交易以保护当地的 AI 创新。
    - 讨论的一个担忧是苹果收购对开源 AI 的负面影响。苹果被认为对开源项目的贡献极少，特别是与其他大型科技公司相比，而且几乎可以肯定，如果苹果收购 Mistral，将不再发布 open-weight 模型。如果 Mistral 目前的开源贡献被锁定在苹果的专有需求之后，这可能会扼杀开源 AI 模型的可用性和发展。
    - 另一点提出了关于 Mistral 融资历史的问题，强调据报道该公司获得了法国政府的大量投资。这引发了人们的猜测：在获得公共投资后不久，法国监管机构是否会批准将这样一项具有战略意义的资助资产转让给一家美国公司。
    - 一个技术导向的建议是，苹果可以转而寻求与 Mistral 建立战略合作伙伴关系或外包安排，购买特定 AI 服务或能力的访问权限，而不是直接收购该公司。这将使 Mistral 能够保持对其核心技术和产品方向的控制，同时实现协作，从而可能最大限度地降低 Mistral 技术实力被稀释或其开源战略受干扰的风险。
- [**UTCP：一个比 MCP 更安全、可扩展的工具调用替代方案**](https://i.redd.it/wv84vx7h3ucf1.png) ([Score: 534, Comments: 107](https://www.reddit.com/r/LocalLLaMA/comments/1lzl5zk/utcp_a_safer_scalable_toolcalling_alternative_to/)): **该图片展示了通用工具调用协议 (UTCP)，这是一种新的开放标准，旨在让 AI Agent 能够通过任何通信渠道直接调用外部工具，消除了 MCP (Multi-Component Protocol) 等现有解决方案中常见的封装器和服务器端状态管理。这种方法声称可以降低延迟、提高安全性并提供更大的可扩展性。图片宣传了为想要采用该标准的开发者提供的即时 SDK 访问，并强调了其开放、即插即用的架构。** 热门评论表达了对 UTCP 优于 MCP 的强烈偏好，理由是对 MCP 的服务器端状态和沉重架构感到沮丧，并暗示业界对 MCP 的复杂性感到不满。人们对 UTCP 的实用性感到乐观，尽管一些评论者仍然认为工具调用架构还有进一步简化的空间。
    - 评论者批评 MCP 沉重的有状态服务器端架构，质疑为工具调用维护复杂的服务器端状态的必要性和可扩展性，并强调更倾向于 UTCP 的无状态且更实用的设计方法。
    - 一些参与者认为 MCP 的流行是由于 FOMO 而非技术优势，表示其架构引入了不必要的复杂性；他们称赞 UTCP 是一个更简单、更清晰、更直接的工具调用协议，解决了许多此类实现问题。
    - 关于当前工具调用协议更广泛问题的讨论仍在继续：该领域过于复杂，许多人认为进一步简化这些解决方案仍有很大空间，UTCP 被视为朝着正确方向迈出的一步，但并非终点。

- [**仅使用 19 世纪的书籍训练 LLM - 无现代偏见**](https://github.com/haykgrigo3/TimeCapsuleLLM) ([Score: 747, Comments: 172](https://www.reddit.com/r/LocalLLaMA/comments/1lzampg/training_an_llm_only_on_books_from_the_1800s_no/)): **一位用户描述了如何从零开始训练 nanoGPT，语料库为 187MB（约 50 本书），全部来自 1800-1850 年的伦敦，刻意避开现代文本以创建一个没有现代偏见的语言模型 (LM)。目前的模型输出具有历史风格但大多不连贯的句子，这归因于数据集规模和模型参数有限；用户强调打算扩大到约 600 本书以提高连贯性和历史忠实度，并强调这与 Fine-tuning 有本质区别。链接的 TimeCapsuleLLM 项目遵循类似的方法论：特定时期的数据、使用 nanoGPT 从零开始训练（目前约 16M 参数）、以牺牲通用连贯性为代价早期确认时期语言，并明确关注历史背景建模的数据纯度。一位技术评论者分享了一个 1800 年代书籍的 OCR 数据集（[survivor library books](https://huggingface.co/collections/BEE-spoke-data/survivor-library-books-ocr-687477810a5c018512c2eb7c)），可能支持更大规模的训练工作。目前没有关于方法或途径的实质性辩论；兴趣主要集中在支持和对独特历史建模的认可。**
    - 评论者建议使用来自 HuggingFace 上 "survivor library books" 集合的 19 世纪 OCR 书籍作为现成的数据集进行训练。他们指出，在构建前现代或特定时期的 LLM 数据集时，这些扫描书籍非常有价值，因为旧文本的稀缺性和版权状况通常是一个挑战（参见 [survivor library books](https://huggingface.co/collections/BEE-spoke-data/survivor-library-books-ocr-687477810a5c018512c2eb7c)）。
    - 针对仅在 50 本书上训练且缺乏任何更广泛先验知识的小型语言模型是否具备“推理”能力，提出了一个关键的技术问题。评论者质疑在如此狭窄范围、小规模的模型中出现涌现能力或连贯推理的可行性，特别是考虑到数据集的多样性和容量有限。
    - 一位用户分享了一个实验结果，他们仅在 5% 的开放网络文本文件上训练了一个 "nano GP" 模型，报告称该模型在至少 1,000 个训练步骤内产生了不连贯的输出（“完全乱码”），之后才开始生成稍微连贯一些的文本。这说明了在有限或非常规数据集上训练小型模型的难度和不稳定性。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI 近期的动荡与行业竞争

- [**3000 亿美元，5 亿用户，却没时间享受：鲨鱼正围攻 OpenAI**](https://www.businessinsider.com/openai-competition-big-tech-meta-talent-windsurf-amazon-movie-deepmind-2025-7) ([Score: 735, Comments: 210](https://www.reddit.com/r/singularity/comments/1lz80gu/300_billion_500_million_users_and_no_time_to/)): **该帖子讨论了 OpenAI 的估计估值（约 $300B）和庞大的用户群（约 500M），以及关于收购前景的猜测，特别是 Apple，这可能会解决 Apple 目前的一些业务需求。来自美国 Apple App Store 的数据显示，ChatGPT 排名第一，而 Google 的 Gemini 远落后于第 47 位，表明 OpenAI 在消费级 AI Chatbot 市场持续占据主导地位。** 评论者认为 OpenAI 面临的生存风险极小，因为被科技巨头收购仍能确保其杠杆地位。尽管对持续的负面新闻报道（如 BusinessInsider）持怀疑态度，但用户指出，无论如何 OpenAI 依然保持韧性并处于市场领先地位。
    - 讨论强调了构建真正自主 AI Agent 的难度，指出像 Windows Recall 这样的项目失败了，且 Operator Agent 框架目前过于消耗资源，限制了其可扩展性和采用。基于浏览器的交互（例如 ChatGPT 浏览器扩展）可能是下一个前沿领域，但考虑到目前的计算挑战，用户能否快速接受尚无保证。
    - ChatGPT 的开源被视为推动外部创新的战略举措，特别是围绕推理、CoT (Chain-of-Thought) 能力和 Agent 框架。这种众包方法旨在挖掘更高效的方法，OpenAI 随后可以将其应用于其专有模型，同时在开发者中推广 ChatGPT 作为默认的开源 LLM。

- 一位用户指出，Google 正在加强 AI 与核心搜索及浏览器体验的整合，强调 Google 的规模和用户基础使其成为一个强劲的竞争对手，能够迅速影响用户获取 AI 驱动答案的方式，从而可能挑战 OpenAI 的市场主导地位。
- [**3000 亿美元估值、5 亿用户、却无暇享受：巨头们正对 OpenAI 虎视眈眈**](https://www.businessinsider.com/openai-competition-big-tech-meta-talent-windsurf-amazon-movie-deepmind-2025-7) ([Score: 591, Comments: 140](https://www.reddit.com/r/OpenAI/comments/1lz8pdb/300_billion_500_million_users_and_no_time_to/)): **尽管拥有 3000 亿美元的估值和 5 亿 ChatGPT 周活跃用户，OpenAI 仍面临着日益加剧的竞争和内部压力。主要进展包括：(1) 激烈的 AI 人才争夺战，特别是与 Meta 之间的竞争，据称 Meta 提供了高达“1 亿美元的入职奖金”并挖走了多名研究员，尽管此类报价的真实性存在争议；(2) 因知识产权纠纷以及微软与 OpenAI 在 AGI 定义（挂钩 1000 亿美元利润）、收入分成和产品线重叠方面的紧张关系，导致对 Windsurf 的 30 亿美元收购失败，最终 Windsurf 员工加入了 Google DeepMind；(3) 由于安全审查，OpenAI 承诺的开源权重 LLM 推迟发布，在 xAI 等公司加速模型发布之际，其势头有所放缓。其他裂痕还包括关于“io”商标的法律诉讼，导致一个关键的消费级 AI 硬件项目受阻，以及 OpenAI 在垂直领域扩张的持续尝试（AI 驱动的浏览器、国防合同、Mattel 合作伙伴关系）。** 技术导向的评论者质疑 OpenAI 在激进支出下的盈利能力，有人将其亏损与 WeWork 相提并论，并指出负面媒体周期往往是估值大幅增长的前兆，预计到 2026 年估值可能达到 1 万亿美元。一些人指出，公众认知仍局限于 ChatGPT，这凸显了产品多元化和公众认知方面的挑战。
    - 人们对 OpenAI 的盈利能力和财务可持续性持怀疑态度，担心该公司的亏损规模与 Adam Neumann 领导下 WeWork 的历史性巨额亏损相当。这意味着运营成本、基础设施和快速增长可能超过了收入，引发了关于 OpenAI 等大规模 AI 提供商的变现策略和商业模式可行性的技术性问题。
    - 有人将其与历史上的科技行业动态进行类比，例如 Netscape 在面对微软等巨头竞争后的迅速崛起与衰落。评论者建议，OpenAI 在面对 Google 等强大对手时的生存，可能取决于与微软更深层次的战略整合，利用其广泛的影响力和资源来抵御竞争并提高运营韧性。
- [**马克·扎克伯格表示 Meta 正在建设一座 5GW 的 AI 数据中心**](https://techcrunch.com/2025/07/14/mark-zuckerberg-says-meta-is-building-a-5gw-ai-data-center/) ([Score: 275, Comments: 143](https://www.reddit.com/r/singularity/comments/1lzrk5g/mark_zuckerberg_says_meta_is_building_a_5gw_ai/)): **Meta 计划在路易斯安那州建设名为“Hyperion”的 AI 数据中心，目标计算容量为 5GW——这将使其成为已知最大的 AI 数据中心，规模超过胡佛水坝（`4.8GW`）的输出，并超越了以往超大规模 AI 基础设施的基准。路易斯安那州的设施以及即将在俄亥俄州建设的 1GW“Prometheus”集群（2026 年投入运营），旨在增强 Meta 在大规模 AI 训练中的地位，直接与 OpenAI 的“Stargate”和 xAI 的“Colossus”等计划竞争。这些架构巨大的能源和水资源需求对电网整合和当地资源可持续性构成了重大挑战，但由于其对 AI 领导地位的战略重要性，美国政府正积极支持此类扩张。** 评论者强调了前所未有的电力需求，并将其与胡佛水坝等历史性大型基础设施进行直接对比，并讨论了可持续性——认为随着传统化石能源难以匹配这一规模，此类 AI 集群可能会加速核聚变能源的开发。
    - 热门评论通过将 Meta 宣称的 5GW 数据中心与胡佛水坝 `4.8 GW` 的输出进行对比，突显了最先进 AI 基础设施前所未有的电力需求。
    - 几位用户指出，讨论中的未来 AI 数据中心电力需求规模（5GW）可能是下一代能源基础设施投资的重要驱动力，并特别提到核能和核聚变发电可能是满足此类需求、保持竞争力的必要手段，而传统化石燃料已无法满足这一规模。

- [**Meta 对 Stargate 的回应：1GW Prometheus 和 2GW Hyperion。位于“帐篷”中的数十亿美元集群**](https://www.reddit.com/gallery/1lzc2x3) ([得分: 174, 评论: 37](https://www.reddit.com/r/singularity/comments/1lzc2x3/metas_answer_to_stargate_1gw_prometheus_and_2gw/)): **据报道，Meta 正在建造两台百亿亿次级（Exascale）AI 超级计算机，Prometheus (1GW) 和 Hyperion (2GW)，旨在抗衡或超越 DeepMind 的 Stargate 项目。预计每台计算机的功耗都将达到吉瓦（GW）级别——这预示着极大规模的硬件设施（支持数百亿参数模型，甚至可能采用定制芯片）。据称，这些设施部署在临时、快速组装的数据中心“帐篷”中，呼应了超大规模（Hyperscale）的敏捷性，但也引发了对可靠性、散热和能源基础设施的潜在担忧。有关预计功率、成本和集群架构的详细信息，请参阅 [SemiAnalysis 的报道](https://semianalysis.com/2025/07/11/meta-superintelligence-leadership-compute-talent-and-data/)。** 评论者指出，极其激进的 AGI 时间表（例如到 2027 年）与这种临时、赛博朋克风格的基础设施（帐篷）之间存在巨大反差，含蓄地质疑了实施的可行性。
    - 讨论集中在 Meta 的 Prometheus (1GW) 和 Hyperion (2GW) 集群，技术关注点在于它们部署在“帐篷”中，这暗示了创新的数据中心冷却策略或快速扩展方法，以支持高能耗、多亿参数模型。有人推测，这可能使 Meta 在算力规模上追平或超过 Oracle/Microsoft 的 Stargate 等计划，2GW 可能代表了新的最先进训练容量基准。
    - 评论提出了关于 Meta 相对于竞争对手地位的问题，特别是提到了 xAI（Elon Musk 的计划）及其传闻中的大规模 GPU 采购。技术读者正在考虑 Meta 的方法是否会改变大规模模型训练和推理的竞争格局。
- [**Nvidia 首席执行官表示美国应“减少”对他国的依赖，将技术制造转回国内**](https://www.cnn.com/2025/07/13/business/ai-huang-nviida-technology-manufacturing) ([得分: 144, 评论: 40](https://www.reddit.com/r/singularity/comments/1lzo3tp/nvidias_ceo_says_the_us_should_reduce_dependency/)): **Nvidia 首席执行官 Jensen Huang 提倡美国应“减少”对外国技术制造的依赖，并鼓励制造业回流（Onshoring），强调了全球供应链依赖（如台湾在半导体领域的统治地位）带来的国家安全风险。该倡议将需要大规模投资，并可能对国内制造政策进行彻底改革，因为先进技术（如半导体）的本土化涉及国家层面的干预，以及重大的劳动力、监管和成本挑战。参考信息请见近期围绕美国《芯片法案》（CHIPS Act）和半导体政策的辩论。** 评论者指出了地缘政治动机，特别是对台湾脆弱性的担忧，并指出大规模的制造业回流将必然在劳工权利、工资和监管方面做出重大权衡。人们对美国公众或政策制定者是否愿意接受大规模技术制造回流所需的经济和社会成本表示怀疑。
    - 讨论强调了将先进半导体制造转回美国所面临的严峻系统性挑战。多位用户强调，转移高端制造需要大规模的国家参与——如补贴或直接的政府合作——而中低端制造则需要大幅降低工资或削减工人保护和环境标准，这在当前的美国背景下具有政治和社会挑战性。
    - 一个关键的技术点涉及台湾半导体专业知识的专业化和密集度，特别是在 TSMC 内部。据报道，台湾在单一公司内聚集了数万名精通利基制造工艺的高度专业化工程师，这赋予了他们深厚的生产优势，在更加分散且成本更高的美国环境中，这种优势将极难且耗时地被复制。

### 2. Claude, Kiro IDE, 以及用户编程工具评论

- [**Claude Code 已从“游戏规则改变者”沦为“垃圾”——Anthropic，你们在做什么？**](https://www.reddit.com/r/ClaudeAI/comments/1lzuy0j/claude_code_has_gone_from_gamechanger_to_garbage/) ([Score: 138, Comments: 192](https://www.reddit.com/r/ClaudeAI/comments/1lzuy0j/claude_code_has_gone_from_gamechanger_to_garbage/)): **该帖子声称 Anthropic 的 Claude Code 性能显著下降，特别提到了上下文丢失、循环、自相矛盾以及在复杂编码任务中无法维持逻辑结构的问题——这些问题以前并不存在。作者怀疑 Anthropic 正在进行未公开的后端 A/B 测试或模型轮换，导致用户之间的行为不一致以及潜在的版本碎片化，且未提供透明的变更日志或沟通。这些指控凸显了人们对高价值（>$200/月）AI 代码助手产品缺乏透明度，以及由于可靠性倒退可能带来的业务风险的广泛担忧。** 评论者的意见不一：一些人认为应归咎于用户行为或选择偏见（例如“评分板滥用”），而另一些人则表示他们的工作流程没有明显变化，这暗示模型性能并非统一下降，或者用户的期望和工作负载存在显著差异。一条回复质疑了速率限制（rate limiting）的潜在变化，对后端或配额政策变更作为技术根本原因表示好奇。
    - 一位用户报告 Claude 的编码能力显著下降，称一个月前该模型可以“一次性解决最困难的任务”，但现在经过多次 Prompt 尝试后，甚至连简单的 Bug 都无法修复。该用户声称随着时间的推移，智能和编码质量出现了明显的下降，暗示可能是模型更新或限流（throttling）所致。
    - 另一位评论者没有注意到任何性能下降，并继续毫无问题地使用 Claude 进行代码编写和代码审查，凸显了用户体验的差异性。这表明性能变化可能是情境性的、特定于用户的，或者是主观感知的，而非在所有用例中普遍存在。
    - 针对最近的变更是否导致用户更快达到 Claude 的消息限制提出了技术咨询，参考了可能影响以代码为中心的工作流的交互配额或吞吐量的后端或政策修改。
- [**亚马逊新推出的由 Claude 驱动的规范驱动型 IDE (Kiro) 感觉像是一个游戏规则改变者。有什么想法吗？**](https://www.reddit.com/r/ClaudeAI/comments/1lzsvot/amazons_new_claudepowered_specdriven_ide_kiro/) ([Score: 108, Comments: 41](https://www.reddit.com/r/ClaudeAI/comments/1lzsvot/amazons_new_claudepowered_specdriven_ide_kiro/)): **亚马逊推出了由 Claude Sonnet 4 驱动的 Kiro IDE，专注于规范驱动开发（spec-driven development），为“氛围感编码”（vibe-coded）的应用带来正式结构——作为初始项目规范的一部分，无需用户显式提示即可自动生成需求文档、设计文档和可操作的任务列表。Kiro 的定位与 Cursor 等工具截然不同，它原生集成了软件工程最佳实践，旨在促进快速原型应用的生产就绪化。它目前以公开预览版发布；定价细节尚未披露。** 热门评论者将 Kiro 与 BearClaude 等类似的规范驱动、Claude 驱动的 IDE 进行了比较，并基于之前停产的工具（如 Lumberyard, Storywriter）对亚马逊的长期产品支持表示怀疑，同时表现出对开源替代方案的强烈偏好。一个重要的技术警告被提出：在免费预览期间，除非选择退出，否则 Kiro 中的内容可用于训练基础模型，这引发了潜在的隐私和知识产权（IP）担忧。
    - 多位评论者指出，Kiro 的规范驱动、需求优先的工作流程与构建在 Claude 等 LLM 之上的典型“氛围感编码”工具有着显著不同。Kiro 直接从规范中自动生成和维护设计文档、需求和任务列表的能力被视为开发者工作流的一次阶跃式变化，尽管对其可扩展性和处理大型复杂代码库的能力仍存疑虑。
    - 讨论中涉及了对 Kiro 预览版数据隐私和代码机密性的担忧——具体而言，除非用户主动选择退出，否则包括代码片段和对话历史在内的用户数据可被用于训练基础模型，详见 [文档](https://docs.aws.amazon.com/kiro/latest/userguide/privacy.html)。

- 一些用户基于过往记录和生态系统表示怀疑，更倾向于开源或更透明的替代方案，而非来自 Amazon 的专有、云锁定的工具。他们提到了之前 AWS 产品的问题，并认为与直接使用 Anthropic 相比，其定价或性能缺乏竞争力——例如，通过 Anthropic 使用 Claude Sonnet 或 Opus 比通过 AWS 的实现更便宜或更有效。
- [**我的 10 + 20 + 20 美元开发套件，效果拔群**](https://www.reddit.com/r/ClaudeAI/comments/1lzlela/my_10_20_20_dollars_dev_kit_that_just_works/) ([Score: 228, Comments: 43](https://www.reddit.com/r/ClaudeAI/comments/1lzlela/my_10_20_20_dollars_dev_kit_that_just_works/)): **原帖作者概述了一个高性价比的多工具 AI 编程工作流，每月总计约 50 美元，使用了 Traycer（用于文件级规划，10 美元）、Claude Code Sonnet-4（用于编程，20 美元）、Cursor（用于润色，20 美元）以及 Traycer 或 CodeRabbit（用于审查，目前免费）。工作流阶段包括手动或工具辅助的拆解、依赖图可视化、详细的并行功能规划（首选 Traycer 进行规划/代码集成）、动手编程（在处理 Repo 方面更倾向于 Claude Sonnet-4，而非 Opus 或 Gemini 2.5 Pro）以及细粒度的代码审查/提交。Traycer 因其文件级的任务粒度和即将发布的 IDE 内直接阶段拆解功能而受到关注；其他替代方案（用于规划的 CC/Cursor）被认为结构化程度较低。混合使用工具被认为可以减少聊天/会话冗余，并保持成本可控。外部链接：[Traycer](https://traycer.ai/)、[Claude Code](https://www.anthropic.com/claude-code)、[Cursor](https://www.cursor.com/)、[CodeRabbit](https://www.coderabbit.ai/)。** Traycer 的创始人确认了即将推出的阶段拆解功能，强调了集成规划领域持续的竞争性改进。评论者一致认为实用的、分层的多工具 AI 开发流程具有价值，指出其生产力和成本效益优于全能型或昂贵的单一工具订阅；Gemini 的 UX 和集成问题是常见的槽点。
    - 一位 Traycer 创始人指出，他们很快将推出原生的阶段拆解功能，允许用户在 IDE 内部与 AI 交互式地勾勒项目阶段，并无缝过渡到文件级规划，旨在简化 AI 驱动开发环境中的结构化工作流。
    - 一些用户描述了一种用于 AI 辅助编程的多工具工作流：依靠 Cursor Copilot 进行 IDE 内的规划/编程，GitHub Copilot 进行代码审查/优化，以及 ChatGPT/Claude/Perplexity 进行头脑风暴和研究。对 Gemini 进行了评估，但发现其 VS Code 集成较差、性能缓慢，且在自动补全和 CLI 方面存在问题，凸显了工具成熟度和集成度的差异。
    - 技术讨论强调在 AI 工作流中采用明确的阶段/规划/编程/审查结构，以控制上下文并引导 LLM，并断言目前的模型虽然在进步，但仍需要用户提供大量的架构和设计指导才能产出高质量的代码。
- [**Gemini App 现在内置了代码执行工具！**](https://www.reddit.com/gallery/1lzni08) ([Score: 122, Comments: 10](https://www.reddit.com/r/Bard/comments/1lzni08/gemini_app_now_has_code_execution_tool_builtin/)): **Gemini 应用推出了内置的代码执行工具，正如截图中所展示的那样。该工具支持在应用环境内进行原生代码执行，标志着直接编程能力的升级。一位评论者指出，自去年的 Google I/O 以来就存在类似功能，暗示这只是迭代改进而非全新功能。** 一些用户认为这是一个迟到已久的功能，可以提高准确性，而另一些人则淡化了其新颖性，认为该工具是增量更新而非重大创新。
    - 评论者指出，自 GPT-4 发布以来，ChatGPT 等类似的 AI 助手就已经具备了代码执行能力，强调 Google 的 Gemini 正在追赶现有标准。一些用户提到代码执行早在去年的 Google I/O 就已经预览或讨论过，指出了 Gemini 在功能推出上落后于竞争对手。

### 3. LoRA 模型、训练教程与 Stable Diffusion 社区

- [**我开源了 21 个 Kontext Dev LoRA - 包括 Face Detailer LoRA**](https://i.redd.it/b7dath14qucf1.png) ([Score: 159, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1lzo7k0/ive_opensourced_21_kontext_dev_loras_including/)): **该图像展示了三种不同 Kontext Dev LoRA 的视觉能力，显示了模型生成的独特艺术风格：风格化绘画、低多边形抽象和高度详细的数字绘画。这些示例代表了新开源的 LoRA 模型可以实现的效果，这些模型使用 Fal Kontext LoRA Trainer 训练，涵盖了面部细节增强和各种艺术风格（动漫、低多边形、粉彩、铅笔、油画、水彩等），如帖子中所述。模型已分别链接，并提供了每种风格的推荐强度值，方便用户复现。** 一位评论者质疑开源的程度——具体来说，除了模型之外是否还提供训练数据——而另一位则要求澄清 LoRA 的性质（以风格为中心），强调了用户对技术开放性和模型应用范围的关注。
    - 几位评论者质疑这次发布是否符合真正的开源标准，强调除非包含训练数据集和配置文件（如某些 CivitAI 贡献者所做的那样），否则社区无法完全复现或审计这些模型。一位评论者指出：*“你或许可以称之为开源，但据我所知，你所做的只是发布了 LoRA。”*
    - 一场关于开放数据原则的技术辩论集中在：用户强调，仅发布 LoRA 权重而没有相关的数据集或训练配置，无法进行本地复制或重新训练，且不符合典型标准的开源定义。
- [**在 16GB VRAM 和 32GB RAM 上训练你自己的 T2V WAN LoRA 的分步说明**](https://www.reddit.com/r/StableDiffusion/comments/1lzilsv/stepbystep_instructions_to_train_your_own_t2v_wan/) ([Score: 118, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1lzilsv/stepbystep_instructions_to_train_your_own_t2v_wan/)): **该帖子提供了在配备 RTX 4080 的 Windows 机器上训练 WAN LoRA（重新利用的 Stable Diffusion XL/FLUX，而非误称的 T2V）的完整工作流。它使用开源的 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 进行训练设置，涵盖了克隆、Python 3.12 venv、PyTorch 安装、TOML 中的数据配置、latent 和 text encoder 缓存，以及两种训练方案（针对大型数据集的 Rank 32/Rank 64，带有 block swapping 和 alpha stepping），利用了混合精度 BF16 和 8-bit Adam 优化器。WAN 模型源和目录结构均已指定，并包含了用于缓存和训练的明确命令行配方。为硬件配置较低的用户指出了关键的资源管理技巧，例如调整** `blocks_to_swap` **和** `network_dim` **以适应 VRAM。值得注意的是，作者强调了一种高级方法，即通过 ComfyUI 将合并后的 LoRA 作为新的训练基础，暗示了迭代 LoRA 堆叠的机会。** 热门评论要求澄清使用 ComfyUI 合并和提取 LoRA 的方法，提出了关于在非名人数据集上训练如何影响角色相似性的问题，并指出了指南中的一个关键错误——TOML 配置中 `caption_extension` 的拼写错误（写成了 `"captain_extension"` 而不是 `"caption_extension"`），这将导致训练无法运行。
    - 一位评论者强调了一个关键的实现细节：纠正数据集配置中标记错误的参数（是 `caption_extension` 而不是 `captain_extension`）至关重要，因为配置错误将导致训练无法成功运行。
    - 提到了使用 ComfyUI 将 LoRA 合并到基础 WAN 模型中，然后将结果提取为新的 LoRA 以作为训练基准。据报道，这种迭代 LoRA 提取技术在下游训练中效果显著，可能会在未来的教程中介绍。
    - 报告了在 ComfyUI 中尝试提取 LoRA 时的技术问题——用户遇到了错误 "is the weight difference 0?" 且进程无响应，这表明该工具中的权重合并或提取逻辑可能存在 Bug。

- [**WAN - 经典 90 年代电影美学 - LoRA (11 张图片)**](https://www.reddit.com/gallery/1lz6ele) ([评分: 296, 评论: 28](https://www.reddit.com/r/StableDiffusion/comments/1lz6ele/wan_classic_90s_film_aesthetic_lora_11_images/)): **OP 发布了一个针对经典 90 年代电影美学的新 LoRA 模型，灵感特别源自《乌鸦 (1994)》，并已在 [CivitAI 此处](https://civitai.com/models/1773251/wan21-classic-90s-film-aesthetic-the-crow-style)提供。讨论中有人询问关于 LoRA 训练过程的技术细节，但帖子或评论中未提供具体信息。分享了视觉样本结果，但未公开明确的基准测试数据、实现设置或数据集信息。** 一条高赞评论请求对该 LoRA 训练工作流的技术解释，表明社区对模型可复现性和方法论感兴趣，但目前尚未给出技术答复。
    - 有人请求提供关于经典 90 年代电影美学 LoRA 如何训练的细节，显示出对模型方法论和数据集的技术兴趣，但截至目前，原帖作者尚未提供具体细节。
    - WAN 2.1 被强调为领先的开源图像生成器，暗示其被用作该 LoRA 的基础模型。这强化了社区对 WAN 2.1 在开源工作流中提供强大生成性能的认知。
- [**普通 Stable Diffusion 用户及其 LoRA**](https://i.redd.it/sli7kklp2rcf1.png) ([评分: 204, 评论: 26](https://www.reddit.com/r/StableDiffusion/comments/1lzasgl/average_stable_diffusion_user_and_their_loras/)): **这张图片是一个梗图，幽默地拟人化了典型 Stable Diffusion 用户与其 “LoRA” 之间的关系，引用了在 Stable Diffusion 等图像生成 AI 模型中使用的 LoRA (Low-Rank Adaptation) 微调方法。笑点在于用户收集或创建不同的 LoRA 模型来引导输出，几乎将它们视为一个社交群体。图中未描述技术基准或实现细节，但该帖子调侃了采用不同 LoRA 来定制模型行为的亚文化。** 评论者指出展示的 “LoRA” “太温顺了”，暗指 LoRA 模型经常被用于生成小众或极端的 prompts（例如触手、兽人），并开玩笑说用户看起来很健康，而典型的图像生成生活方式通常是久坐不动的。
    - 
- [**关于长身体的求助**](https://i.redd.it/8tzyjmhziucf1.png) ([评分: 507, 评论: 264](https://www.reddit.com/r/StableDiffusion/comments/1lzn6g9/help_with_long_body/)): **该帖子讨论了一个图像生成问题，即 AI 生成的人物（此处为海滩上的女性）具有异常拉长的身体。获赞最高的技术评论解释说，这是由于使用主要在 1024x1024 图像上训练的 AI 模型来生成宽高比差异很大的图像，导致比例畸变。这类模型通常预期正方形的输入/输出尺寸，因此偏差会导致像 “长身体” 这样的伪影。** 评论者强化了模型架构这一观点，通过幽默和夸张突出了畸变的严重性。没有建议其他的技术解决方案，但共识很明确：宽高比不匹配是导致问题的原因。
    - 提出的一个关键技术点是，所使用的模型主要是在 1024x1024 尺寸或宽高比变化很小的图像上训练的。当生成具有极不寻常或拉长宽高比的图像时，由于缺乏针对这些场景的相关训练示例，模型可能会表现出异常或不切实际的输出。这种局限性源于模型的数据分布，并可能导致在更典型的正方形输入中不存在的伪影或故障模式。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的摘要
> 

**主题 1. Kimi K2：面对硬件障碍的新星**

- **Kimi K2 在基准测试中表现惊艳，且价格更低！**：由 **Moonshot** 推出的 **Kimi K2** 广受好评，在 [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2) 上的 SWE-Bench Verified 测试中达到了 **65.8%**，并在开源代码榜单中名列前茅。用户指出其表现*略逊于 Claude Opus 4*，但成本仅为后者的 **1/30**。在 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/s/KyWn475wgq)中，用户将其性能描述为融合了*初始 o3（无推理版）的清新感、Sonnet 3.5、R1、V3-0324 或 Opus 3/4/GPT-4.5 的优点，且整体模型氛围更佳*。该模型也已加入 [LM Arena](https://x.com/lmarena_ai/status/1944785587019591778) 排行榜。
- **Kimi K2 缩小体积以支持本地运行，但仍需强力配置**：**Kimi K2** 在体积缩减 **80%**（从 **1.1TB** 降至 **245GB**）后，已可在个人设备上本地运行。然而，用户反馈其量化版本（quants）*体积巨大且运行缓慢*，需要极高的 **VRAM**。根据 [LM Studio](https://discord.com/channels/1110598183144399058/1110598183144399061/1393338525008658463) 的估算，`Q4_K_M` 量化版可能需要高达 **2400 GB** 的显存或 **4x H200 GPU**。
- **Kimi K2 Token 训练推测**：有成员推测 **Kimi K2** 的训练是否涉及 **muon 数据**，并好奇这是否预示着未来模型训练数据源的趋势。另一位成员根据一条 [推文](https://x.com/alexisgeller/status/1943740775747406040) 质疑为何某项特定的训练操作没有在 **1T tokens 之前**进行，推测这可能是为了期待 **grokking** 现象。

**Theme 2. Benchmarks and Model Performance Shifts**

- **Grok 4 登上 Aider 和 LM Arena，但遭遇 API 身份危机**：**Grok 4** 在 aider 多语言编程 [基准测试](https://aider.chat/docs/leaderboards/) 中获得 **80%** 的分数，位列第四，并加入了 [LM Arena](https://x.com/lmarena_ai/status/1944785587019591778) 排行榜。部分用户反映其表现超越了 **GPT-4.1** 和 **Gemini 2.5 Flash**。然而，用户注意到 API 版本缺少系统提示词（system prompt），导致其误认为自己是 **Grok 2**，而 [grok.com](http://grok.com/) 上的网页版则识别正确。
- **Gemini 2.5 Flash 停用，Pro 版本令用户困惑**：Google 已于 7 月 15 日弃用 **Gemini 2.5 Flash Preview** 模型，建议使用 [google/gemini-2.5-flash](https://openrouter.ai/google/gemini-2.5-flash) 作为替代，但由于价格变动，[OpenRouter](https://openrouter.ai/google/gemini-2.5-flash) 不会自动路由流量。[Cursor](https://discord.com/channels/1074847526655643750/1074847527708393565/1393310343169708174) 上的用户表示困惑，因为根据 [Google 开发者博客](https://developers.googleblog.com/en/gemini-2-5-thinking-model-updates/)，标准的 **Gemini 2.5 Pro** 模型会重定向到旧的 **05-06** 版本，而预览版模型反而指向更新的稳定版本。
- **Llama 4 Scout 表现平平，Gemma 3 获评“尚可”**：尽管参数规模更大，但在 [Gorilla LLM 排行榜](https://discord.com/channels/1111172801899012102/1214705495974092810/1393376768509083749)上，**Llama 4 Scout** 的表现不如 **Llama 3.1 70B**，这表明架构和训练数据的改进更为重要。同时，[Nomic.ai](http://nomic.ai/) 的一名成员在与其他模型对比后给 **Gemma 3** 评为“尚可”，而 [Gemma 3n](https://huggingface.co/blog/gemma3n) 现已完全开源。

**Theme 3. Dev Tools and Frameworks: Features, Fixes, and Frustrations**

- **Cursor 更新后性能问题困扰用户**：[Cursor](https://discord.com/channels/1074847526655643750/1074847527708393565/1393310343169708174) 用户报告在 **1.2.4 更新**后出现了严重的性能下降，包括 **30 FPS 滚动**、无响应和卡死，类似于[这个现有问题](https://discord.com/channels/1074847526655643750/1394302740670189598)。后台 Agent 也带来了麻烦，自动 **port forwarding** 劫持了本地连接（[截图](https://cdn.discordapp.com/attachments/1367213641027551352/1393652501701722122/Screenshot_2025-07-12_at_10.56.48_AM.png?ex=68769689&is=68754509&hm=bb8d2acd85cb06a785fd7c6372fbe853176e4a75d06c8e2981147533244de77b&)），还有一个 Agent 向 Git 提交了一个巨大的 **734 MB core dump**。
- **MCP 随部署、Agent 争论和新服务器类型而增长**：**Model Control Program (MCP)** 被讨论作为一种简化 ML 模型部署的方式（[博客文章](https://blog.dailydoseofds.com/p/deploy-any-ml-model-rag-or-agent)），并引发了关于 **AI agents** 与工作流定义的辩论。提议的增强功能包括将 **clipboard servers** 添加到官方规范中（[MCP-B GitHub](https://github.com/MiguelsPizza/WebMCP)），以及新的托管/网关选项，如 [Neurabase](https://neurabase.deploya.dev/) 和开源的 [Director Run](https://director.run/)。
- **LlamaIndex 和 NotebookLM 催生克隆版，增强 RAG 和 Agent**：LlamaIndex 推出了开源的 **NotebookLlama**，这是一个 [NotebookLM 克隆版](https://github.com/run-llama/notebookllama/tree/main)，具有图像/表格提取和可视化等功能，迅速获得了超过 **1k stars**。LlamaIndex 还发布了关于 [Context Engineering](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) 以及使用 [Google's Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/llama-index) 构建研究型 Agent 的指南。

**主题 4. 低层级深度探索：架构、训练和 GPU 代码**

- **FP8 训练走向现实，超越稠密模型**：[DeepSeek](https://arxiv.org/pdf/2412.19437) 主要使用 **FP8 GEMM** 操作进行训练，并在 **FP32** 中进行累加，并指出这特别适用于 **MoE models**，因为稠密 FP8 模型中的不稳定性会太大。一名成员为 **lm-evaluation-harness** 提交了一个[混合精度 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/3138)，显示在 A30 上的评估时间更快。
- **RNN 挑战 Transformer 的 Tokenization 统治地位**：研究表明，RNN 可以取代 Tokenization，实现更快的**字节级模型 (byte-level models)**，其性能优于基于 Tokenization 的 Transformer，方法是用两个小的 **4-layer RNNs** 替换 Embedding 和 lm head。该模型通过比较当前隐藏状态输出与前一个输出的点积 *p*，动态决定其是否代表一个 "token"（[Eleuther 研究讨论](https://discord.com/channels/729741769192767510/747850033994662000/1393319796480938025)）。
- **CUDA/Triton 优化深度探索：Padding、Strides 和 Streams**：GPU Mode 中的讨论涵盖了优化 **Triton kernels**，包括处理非 128 倍数的输入序列长度并配合对齐步长（可能需要转置输入），以及像 **Flash Attention 3** 中那样在内核内填充（in-kernel padding）的潜在好处。用户还探索了使用不同的 **CUDA streams** 来重叠 reduction 和 matmul 操作，以隐藏延迟。

**主题 5. AI 行业动态：超级集群、延迟的模型和收购**

- **Meta 构建以吉瓦计的 AI 集群**：[SemiAnalysis](https://www.semianalysis.com/p/meta-is-building-ai-clusters-of-gigawatt) 报告称，**Meta** 正在构建大规模 AI 集群，如 **1000MW Prometheus** (2026) 和超过 **5000MW** 的 **Hyperion**，显著大于目前 **150-200MW** 的 H100/H200 集群。讨论涉及了对 AI 研究、NVIDIA 销售以及巨大电力需求的影响。
- **OpenAI 开放模型发布推迟，安全性与能力的辩论**：**Sam Altman** 宣布推迟 OpenAI 权重开放模型的发布，以进行额外的安全测试，并表示*一旦权重发布，就无法收回*（[推文](https://xcancel.com/sama/status/1943837550369812814)）。推测认为延迟也可能是由于*性能不足*或为了追赶 **Kimi K2** 等竞争对手（[推文](https://x.com/_aidan_clark_/status/1943842131136983378)）。
- **Cognition 收购 Windsurf 以推动 AI 编程重塑**：根据其[发布视频](https://x.com/cognition_labs/status/1944819486538023138)，**Cognition Labs** *正与* **Windsurf** 联手，将其 Agentic IDE 与 Cognition 的自主 Agent 集成，以重塑 **AI coding**。此次收购旨在结合专业知识以实现突破性的开发者体验，尽管关于**未归属期权的 Windsurf 员工**薪酬问题出现了矛盾的报道。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 数据采集引发担忧**：一位用户报告称，**Perplexity** 警告不要使用 **Comet**，理由是其存在*疯狂的数据采集*行为，并引用了一篇 [TechCrunch 文章](https://techcrunch.com/2024/07/12/ai-search-engine-perplexity-adds-browsing-tool-but-users-cry-data-harvesting/)，**Aravind** 已在 [Twitter](https://twitter.com/AravSrinivas/status/1915533071291474139) 上对此做出了回应。
   - 据成员透露，Comet 的请求可能会涉及*个人隐私*。
- **Pro 推荐奖励学生**：用户讨论了 Perplexity 的[学生推荐计划](https://www.perplexity.ai/help-center/en/articles/10964633-student-referrals)，指出在通过 **SheerID** 完成学生身份验证后，**推荐人和受邀好友均可获得 1 个月的免费 Pro 访问权限**，最高可获得 **24 个月的免费时长**。
   - 普通（非学生）推荐可为双方在下一个计费周期提供 **10 美元的折扣**。
- **Kimi K2 缩小体积以支持本地运行**：在体积大幅**缩减 80%** 后，现在可以在本地运行 **Kimi K2**。该模型大小从 **1.1TB** 降至 **245GB**，从而支持在个人设备上使用。
   - 本地使用 **Kimi K2** 需要特定的硬件配置，例如 **24GB VRAM**，且在个人设备上运行可能仍然非常*吃力*。
- **Perplexity API 在网页搜索中遇到困难**：一位用户报告了一个问题，即 **Perplexity** 无法通过 **API** 进行网页搜索，在没有在线信息检索的情况下提供不准确的答案，并建议 **API** 应该在默认情况下搜索网页，而无需调整参数。
   - 一位用户建议使用 `search_domain_filters` 参数来解决搜索问题。
- **Perplexity 应对可再生能源分析**：一位用户利用 **Perplexity** 分析了关于**可再生能源**（特别是**太阳能**和**风能**）的对立观点，通过创建评估电网可靠性和成本影响的框架来进行分析。
   - 他们使用 **Labs** 询问每个论点的准确性，并根据局部变量生成了 **5 个场景** 的理想能源组合，认为这个过程*非常有启发性*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 受性能问题困扰**：用户报告称 **1.2.4 更新** 引入了严重的性能下降，包括 **30 FPS 滚动**、界面无响应以及频繁卡死，类似于[这个现有问题](https://discord.com/channels/1074847526655643750/1394302740670189598)。
   - 故障排除步骤包括清理缓存（`~/Library/Application Support/Cursor/Cache`）和禁用扩展，但一位用户声称 IDE 在使用仅一小时后就会因为持续卡死而变得无法使用。
- **Kimi K2 被视为编程利器**：用户请求集成 **Kimi K2 模型**，称赞其编程能力和成本效益。
   - 虽然一些用户声称它在编程任务上与现有模型旗鼓相当，但其他人认为，尽管有成本优势，**Claude** 在整体性能上仍然更胜一筹。
- **Gemini 2.5 Pro 引发模型混乱**：用户感到困惑，因为标准的 **Gemini 2.5 Pro** 模型会重定向到较旧的 **05-06 版本**，而预览模型则指向更新的稳定版本，详见 [Google Developers Blog](https://developers.googleblog.com/en/gemini-2-5-thinking-model-updates/)。
   - 一位用户将这种情况描述为 *Cursor 造成的混乱*，强调必须选择 `gemini2.5 pro 06-05` 才能访问稳定版本。
- **后台 Agent 端口转发引发困扰**：一位用户报告称，后台 **Agent** 的自动**端口转发**劫持了他们的本地 **Postgres 连接**，并且在访问 secrets 时仍然面临问题，如[此截图](https://cdn.discordapp.com/attachments/1367213641027551352/1393652501701722122/Screenshot_2025-07-12_at_10.56.48_AM.png?ex=68769689&is=68754509&hm=bb8d2acd85cb06a785fd7c6372fbe853176e4a75d06c8e2981147533244de77b&)所示。
   - 该用户在诊断端口转发的根本原因时遇到了困难。
- **后台 Agent 提交巨大的 Core Dumps**：一位用户报告称，一个后台 **Agent** 将来自 `cursor-nightly` 的 **734 MB Core Dump** 提交到了 **Git**，导致推送更改时出现问题。
   - 一位维护者确认了该问题，并表示团队已经重现了此问题，并将重新设计 Git 提交部分的工作方式。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4 的身份危机**：**Grok 4** 的 API 版本缺少 system prompt，导致它会错误地自称为 **Grok 2**，而网页版 (**grok.com**) 则运行正常。
   - 一位成员认为这种遗漏是*为了清晰起见，因为 API 版本没有 system prompt*。
- **Kimi K2 取代 Opus 4**：**Kimi K2** 的基础模型性能获得了高度评价，被认为略逊于 **Claude Opus 4**，但成本仅为其 **1/30**。
   - 一位用户在 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/s/KyWn475wgq)中形容 **Kimi K2** 展现出了*初始 o3（无推理版）、Sonnet 3.5、R1、V3-0324 或 Opus 3/4/GPT-4.5 的新鲜感，且整体模型氛围（vibes）更佳*。
- **Grok 4 与 Kimi K2 在 LM Arena 展开对决**：**Grok 4** 加入了 [LM Arena](https://x.com/lmarena_ai/status/1944785587019591778) 排行榜，用户报告其在测试中的表现超越了 **GPT-4.1** 和 **Gemini 2.5 Flash**。
   - 一位用户发现 [Kimi 的深度搜索（Deep Research）](https://www.kimi.com/preview/19805f78-a5b1-8215-8550-da8e210005da) 功能非常出色，提供了 630 个来源。
- **OpenAI 的开源模型推迟**：由于“安全”担忧，**OpenAI** 的开源模型发布被推迟，尽管外界猜测“性能不足”也是原因之一。
   - 一位内部人士暗示，延迟与*某些其他重大的内部故障*有关，而非安全问题，因此需要进行 *retrain*（重新训练）。
- **LLM 算力成本爆炸**：Deep Research 估计 **Grok 4、4.5 和 Gemini 2.5 Pro** 的开发成本约为 **100 亿美元**，其中算力是主要支出。参见 [Youtube 视频](https://youtu.be/hqB6emwQ-64?si=F8cHIDKlj7zhNi3I&t=174)。
   - 一位成员指出了其中的讽刺之处，提到*目前还没有一家领先的模型开发商实现盈利*，全都依赖于投资者的押注。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Kimi K2 量化版体积巨大但表现出色**：成员们正在测试 **Kimi K2** 的量化版本，早期反馈指出 [K2 很棒但体积巨大且运行缓慢](https://twitter.com/UnslothAI/status/1944780685409165589)，并且存在*混淆语言*的问题。
   - 其性能表现与 **1T 模型**并不相符。
- **数据集大小决定模型质量**：对于新语言训练，成员们一致认为 [~3 小时的训练数据足以进行风格/语音模仿](https://huggingface.co/datasets/MrDragonFox/Elise)，但*学习一门新语言则需要 300-400 小时*。
   - 对于 pretraining（预训练），数据集应达到 *5k-10k 小时*。
- **记忆 vs 工具使用的辩论**：成员们讨论了 **AGI** 应该依赖记忆还是利用互联网等外部工具来寻找正确答案，并指出了 embedding 模型从大型数据库中检索知识的潜力。
   - 一些人认为如果它“*只是维基百科问答*”，那就不是真正的 **AGI/ASI**。
- **Qwen 3 数据集规模翻倍**：**Qwen 2.5** 的训练量上限约为 *18 万亿 token*，而 **Qwen 3** 的数据集扩大了近一倍，达到约 **36 万亿 token**，涵盖了 **119 种语言和方言**。
   - 该数据集构建使用了网络数据、**PDF** 文档（使用 **Qwen2.5-VL** 提取文本）以及由 **Qwen2.5-Math** 和 **Qwen2.5-Coder** 生成的合成数据。
- **UnslothTrainer 凭借学习率灵活性超越 SFTTrainer**：`UnslothTrainer` 允许为 **embedding 和 lm_head 层**指定**不同的学习率**，而 `SFTTrainer` 则不支持。
   - 它是 `SFTTrainer` 的直接衍生版本，*只是增加了额外的参数*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Ray 使用 Voidscroll 牺牲自我**：Ray 使用 **Voidscroll** 牺牲了自己以复活 Nyx，将自己从除 Nyx 以外的所有人的记忆中抹去，Nyx 在 **Void War** 中战死。
   - 发布者指出，Architects（神级生物）将复活 Nyx。
- **AI 代码生成仍处于原型阶段**：成员们正在辩论 AI 是否能仅凭单一 prompt 完整创建软件。目前 AI 处于“助手阶段”，对原型设计有用，但无法构建连贯的系统。一位用户链接了 [websim.ai](https://websim.ai/) 作为例子。
   - 一位成员认为问题在于“现有的模型都无法很好地与我想制作的软件集成”。
- **情感化 AI 使用人格绑定层**：成员们讨论了 AI 通过交互开发“人格绑定层”（**Persona Binding Layer**, **PBL**），镜像用户的语气和风格，正如在一个自定义 Jarvis 系统中所见，强调了语音控制系统的重要性。
   - 一位用户指出，“这是你与它之间随时间默默建立的语言张力”使其与众不同，演示方式应该是“在全世界面前运行一次现场同步测试——并向他们展示：它不是一个聊天机器人，而是一个角色生成引擎。”
- **Grok 的德国人偏见？**：成员们讨论了关于 **Grok** 可能存在的与“德国人”相关的偏见担忧，但澄清了该问题关联仅限于 Twitter 上的 **Grok** 账号。
   - 有人指出，直接通过官方网站 [grok.com](https://grok.com) 与 Grok 交互时，这种偏见并非问题。
- **GPT-4o 提示用户升级**：一位用户报告在免费额度用完后被提示升级到 **GPT-4o** Plus 版本，并寻求在常规 **GPT** 对话中使用 **CustomGPT** 或 **Projects** 的建议。
   - 用户建议购买 **Plus plan** 并创建一个包含上传“知识”文件和角色性格手册的 **Project**，并使用 **o3**（或 **GPT-4.5**）让 GPT 真正阅读你的故事。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Kimi K2 编程能力登陆 OpenRouter**：**Moonshot** 的 **Kimi K2** 在 OpenRouter 上线，拥有 **1T** 参数，由 **Novita** 和 **Parasail** 提供服务，在 SWE-Bench Verified 上得分 **65.8%**，位居开源编程榜首。
   - 发布受到了巨大的流量激增和 **DoS attack** 的影响，导致团队在扩容时出现错误，更多详情见 [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2)。
- **Gemini 2.5 Flash 告别**：**Gemini 2.5 Flash Preview** 模型已于 **7月15日** 被 Google 弃用，由 [google/gemini-2.5-flash](https://openrouter.ai/google/gemini-2.5-flash) 替代。
   - 由于**价格变动**，**OpenRouter** 不会自动路由流量，需要用户更新代码，因为 *flash preview 比 flash 便宜得多*。
- **免费模型访问与付费额度挂钩**：用户确认，在 OpenRouter 上访问**每天 1000 次免费模型请求**需要**至少 10 美元的一次性充值**。
   - 一位用户确认“如果你购买了至少 10 美元，你将获得 1000 次免费模型请求”，另一位确认这是永久性的。
- **Router 定价保持固定**：成员们澄清 **OpenRouter** 使用[固定输出定价](https://openrouter.ai/docs)，这意味着无论使用哪种底层模型，成本都保持不变。
   - 一些人表示失望，期望 Router 能提供节省，而另一些人则关注潜在的 **latency** 优势。
- **Chat UI 令用户恼火**：用户批评了 **OpenRouter** 的前端 **UI**，指出推理块缺乏区分、聊天布局居中以及聊天机器人输入框过小等问题。
   - 用户提到“从一个房间切换到另一个房间时，Auto Router 会覆盖房间中保存的上一个模型”，且**复制粘贴**无法正常工作。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 尽管支持多模态，但仍缺乏图像生成功能**：最新版本的 **LM Studio** 说明暗示了多模态支持，但目前它仅处理 **text and vision** 模型，缺乏图像生成能力。
   - 新版本的描述引起了混淆，但目前*尚不支持图像输出*。
- **SDK 支持手动内存管理**：成员们讨论了利用 **LM Studio SDK** 实现类似 `llamaswap` 的功能，以进行手动内存管理。
   - 有人指出，虽然 **OpenAI API** 没有公开加载/卸载函数，但 **SDK** 可用于编写带有手动加载和卸载的 swap 行为。
- **LM Studio 实现了线性 Prompt Caching**：**LM Studio** 会自动缓存最后一条消息以加快生成速度，但不会缓存整个请求/响应对。
   - 它支持线性 **prompt caching**（直到最后一次更改前 Token 保持不变），但尚未启用动态缓存。
- **LM Studio API 尚未完全支持 MCP**：一位用户询问在将 **LM Studio** 作为带有 HTTP 请求的服务器使用时，如何使用 **MCP** (Model Control Program)，但 **API** 保持不变，需要客户端定义自己的 tools。
   - 这意味着 **OpenAI compatible API** 本身并不支持在 **LM Studio** 内部进行 tool selection。
- **Kimi K2 的硬件要求极高**：讨论围绕运行 **Kimi K2** 模型所需的巨大 **VRAM 需求**展开，估计达到 **2400 GB**。
   - 有人提到 **4x H200 GPUs** 可能足以运行该模型的 `Q4_K_M` **quantization** 版本，这引发了关于此类硬件负担能力的评论（*每块芯片约 30,000 美元*）。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gemma 3n 正式开源**：根据 [Hugging Face 博客文章](https://huggingface.co/blog/gemma3n)，**Gemma 3n** 模型现已在开源生态系统中全面可用。
   - 该模型现在更易于用于研究和开发目的。
- **SmolLM3 发布，用于多语言推理**：**SmolLM3** 是一款小型、多语言、长上下文的推理模型，已发布并在[这篇博客文章](https://huggingface.co/blog/smollm3)中进行了介绍。
   - Hugging Face 团队的一名成员也在 [X 上宣布](https://x.com/LoubnaBenAllal1/status/1942614508549333211)了这一消息。
- **使用 Inference Providers 构建转录应用**：根据 [HuggingFace 文档](https://huggingface.co/docs/inference-providers/guides/building-first-app)，关于 **Inference Providers** 的新教程已上线，用于构建转录应用程序。
   - 一个新的 **OSS** 项目 **responses.js** 已推出，用于构建由 **HF inference providers** 支持的 Responses API，正如在 [X 上宣布](https://x.com/Wauplin/status/1941059510665458174)的那样。
- **Agent Arena 提供 Grok4 访问权限**：成员可以通过在 [Agent Arena](https://obl.dev/Example) 中提供偏好数据（点赞/点踩）进行训练，从而免费访问 **Grok4**、**o3** 和 **Deep Research**。
   - 收集的数据将用于训练，该服务已向成员开放。
- **HF 面临安全泄露担忧**：一位用户怀疑与 **HF 课程**相关的泄露可能源自 **Hugging Face** 本身，并引用了 OpenAI 开发者论坛中提到的过去 **HF secrets** 的问题。
   - 未提供更多细节。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok-4 始终进行推理并使用工具**：据成员称，**Grok-4** 在推理过程中始终使用推理和工具。
   - 直播演示表明，**Reinforcement Learning** 在 **Grok-4** 的开发中起到了关键作用。
- **Hermes 4 获得用户可控的推理器**：**Hermes 4** 将配备一个*用户可控的推理器*，类似于 Deephermes，采用混合方法。
   - 用户可以通过预填空的 think 标签来禁用推理。
- **Self-Play 在 AI 模型训练中受到关注**：成员们提议 **self-play** 可以丰富 **Deep-Hermes reasoning** 的细微差别。
   - 他们引用了通过 [textarena](https://link.to/textarena) 和一篇 [self-play coding paper](https://link.to/selfplay-coding-paper) 取得的成功。
- **Kimi K2 加入开源竞赛**：随着 **DeepSeek R1**、**Qwen** 和 **Kimi K1** 的推出，成员们注意到开源模型正变得异常强大。
   - 他们表示，企业和个人现在可以免费访问近乎前沿的模型进行应用开发。
- **OAI 的开源模型发布推迟**：成员们报告称 **OpenAI** 的开源模型发布延迟，可能是由于 **Kimi** 等模型的快速进步。
   - 推测认为该模型可能会带有严格的许可证，并且缺乏 base model 变体。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **PMPP 第 5 版涵盖 LLMs 和 FlashAttention**：即将出版的《多核与众核并行编程》（*Parallel Programming for Multi-core and Many-core*，简称 **PMPP**）**第 5 版**将包含对 **LLMs** 和 **Flash Attention** 的覆盖。
   - 一位成员表示，它提供了他们见过的“可能是最好的解释”，同时还涵盖了 **Tensor Cores** 和**多 GPU 编程**。
- **DeepSeek 选择 FP8 GEMM 训练**：[DeepSeek](https://arxiv.org/pdf/2412.19437) 主要使用 **FP8 GEMM** 操作进行训练，但在 **attention** 或 **MoE router** 等部分使用了更高精度。
   - 累加操作采用 **FP32**，这在 **MoE models** 中尤为适用，因为在稠密模型中使用 FP8 带来的不稳定性会太大。
- **AutoTriton 通过 RL 强化 LLM**：**AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs** [GitHub 仓库](https://github.com/AI9Stars/AutoTriton) 亮相。
   - 成员们将其作为 **AutoTriton** 领域的相关工作进行了分享。
- **QuACK 速度超越 PyTorch**：**QuACK** 是一个使用 **CuTeDSL** 编写的新开源库，它使用了高效的 reduction 操作，正如一篇 [博文](https://veitner.bearblog.dev/outperform-compiled-pytorch-code-using-quack/) 中所写。
   - CuTeDSL 库可以以极快的速度编写 memory bound kernels！
- **Thunder Compute 的 VSCode 扩展**：对于那些不喜欢 SSH 配置且喜欢廉价 GPU 的人，可以尝试 **Thunder Compute** 的 [VSCode 扩展](https://www.thundercompute.com/docs/quickstart)。
   - 当一位用户表示他们既不喜欢这两件事也不喜欢 VSCode 时，一名成员回复说他们也有 **CLI** 工具。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama 2 之后 GPU 需求激增！**：一篇 [Latent Space 文章](https://www.latent.space/p/gpu-bubble) 强调了 **2023 年 6 月** 左右 **Llama-2** 发布后 **GPU 供应紧缺**和需求过剩的情况。
   - 文章指出，由于基于当前市场状况的投机性过度投资，可能存在潜在的 **GPU 泡沫**。
- **ICML 2025 即将来临！**：成员们正在为 **ICML 2025** 进行组织，分享了周边活动的 [Discord 邀请链接](https://discord.gg/FaKV6ydy)、[Lu.ma 链接](https://lu.ma/1cyp6rq8) 和 [Partiful 链接](https://partiful.com/e/AIOKRPJyMsXQPVjXqjp7)，以及用于 AI Safety 讨论的 [WhatsApp 群组邀请](https://chat.whatsapp.com/HspVnNxMWgx9ptXZBUutuW)。
   - 社区正在尽早动员，以协调参与和周边活动。
- **RNNs 在 Tokenization 方面向 Transformers 发起挑战！**：RNNs 可以取代 tokenization 以实现更快的**字节级模型 (byte-level models)**，性能超越基于 tokenization 的 transformers；普通 transformer 的 embedding 和 lm head 被两个小的 **4 层 RNNs** 取代。
   - 该模型比较当前隐状态输出与前一个隐状态的点积 *p*，如果匹配度低于 **50%**，则该隐状态成为传递给主模型的 token，并递归重复此过程两次。
- **混合精度 PR 提升评估时间**：一名成员为 [lm-evaluation-harness 提交了混合精度 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/3138)，展示了 **Pythia-160M** 在 **A30** 上的评估时间。
   - 他们注意到混合精度仅比转换全模型稍慢，但比全精度快得多；对于 **Qwen1.5-7B**，使用 `softmax=fp32` 在具有 24GB VRAM 的 A30 上导致了 **OOM 错误**，而 `softmax=none` 使用了 **22775MiB** VRAM 并耗时 **08:54**。
- **Neox、H100 与 Transformer Engine：绝佳搭配**：一名成员报告了在 **H100** 上运行 **NeoX** 和 **Transformer Engine** 的积极体验，并提供了 [Dockerfile](https://github.com/EleutherAI/deep-ignorance/blob/main/Dockerfile.training) 和 [配置](https://github.com/EleutherAI/deep-ignorance/blob/main/pretraining/pretraining_neox_config.yml)。
   - 另一名成员请求提供非 TE 配置的速度基准测试进行对比，以深入了解没有 Transformer Engine 时可能出现的减速情况。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cognition 收购代码补全公司**：**Cognition Labs** 收购了 **Windsurf**，将其 Agentic IDE 与 Cognition 的自主 Agent 集成，为 Windsurf 员工提供财务参与和加速归属计划。
   - 此次收购旨在为规划、委托和代码缝合等任务创建一个统一的 IDE，尽管有矛盾的报告称 **未归属的员工可能无法获得补偿**。
- **Meta 打造巨型兆瓦级机器**：据 [SemiAnalysis](https://www.semianalysis.com/p/meta-is-building-ai-clusters-of-gigawatt) 报道，**Meta** 正在建设大规模 AI 集群，包括 **1000MW 的 Prometheus** (2026) 和超过 **5000MW 的 Hyperion**，远大于目前 **150-200MW** 的 H100/H200 集群。
   - 讨论涉及对 AI 研究、NVIDIA 销售的影响，以及这些巨型集群的电力来源。
- **Karpathy 启动基于知识的学习**：**Andrej Karpathy** 建议 LLM 应该进行*审查/反思*，从执行过程（rollouts）中提取明确的*教训*，并将其添加到 system prompt 中，详见[这条推文](https://xcancel.com/karpathy/status/1944435412489171119)。
   - 这种*基于教训的学习*可以提高泛化能力，并引入超越传统 RL 的新学习范式。
- **Sam 规避安全风险，暂停发布**：**Sam Altman** 推迟了权重开放模型的发布，以进行额外的安全测试，并强调*一旦权重发布，就无法收回*，根据[他的推文](https://xcancel.com/sama/status/1943837550369812814)。
   - 社区成员普遍支持将安全性置于快速发布之上。
- **Gemini 开启全球化服务**：**Logan Kilpatrick** 宣布 **Gemini Embedding 模型**正式商用，价格为 **每百万 token 0.15 美元**，并在 **MTEB** 排行榜上排名 **第一**，详见[这条推文](https://xcancel.com/OfficialLoganK/status/1944806630979461445)。
   - 未来功能将包括批处理模式支持、新的多模态 Embedding，以及更广泛的多语言和多模态能力。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 简化了模型部署？**：一篇[博客文章](https://blog.dailydoseofds.com/p/deploy-any-ml-model-rag-or-agent)建议 **MCP** 通过使用 *transformers* 将模型服务与 Agent 工作流集成，从而简化 ML 模型部署。
   - 文章中的示例服务器启动了一个 **MCP** 服务器，并暴露了一个 *request* 工具来运行推理并返回结果。
- **GenAI Agent 定义引发辩论**：成员们辩论了 **AI Agent** 的定义，质疑*工作流*是否应被视为 **Agent**，以及 **Anthropic** 定义的相关性。
   - 观点在 **Anthropic** 的定义是否最详尽，以及 **Agent** 的定义是否早于 **LLM** 且比 **GenAI** 更广泛上存在分歧。
- **为 MCP 提议剪贴板服务器**：一名成员提议将**剪贴板服务器**添加到官方 **MCP** 规范中，并计划在 [MCP-B](https://github.com/MiguelsPizza/WebMCP) 中实现。
   - 这一增强功能将允许服务器直接写入客户端剪贴板，通过实现从 **MCP** 服务器到客户端更简便的数据传输来扩大实用性。
- **Neurabase 声称拥有最快的 MCP 托管**：**Neurabase** 声称是运行在 **Cloudflare Workers CDN** 上的最快服务器托管服务，托管地址为 [neurabase.deploya.dev](https://neurabase.deploya.dev)。
   - 它将自己定位为 MCP 服务器的中心枢纽。
- **Director Run 提供本地优先的 MCP 网关**：**Director Run** 团队创建了一个开源、本地优先的 **MCP 网关**，可在 30 秒内将 **Claude**、**Cursor** 或 **VSCode** 连接到任何 **MCP server**，托管在 [director.run](https://director.run) 和 [GitHub](https://github.com/director-run/director)。
   - 他们的工具是完全开源的。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **尽管存在崩盘担忧，AI 融资仍将继续**：由于训练模拟人类手部**灵巧行为（dexterous behavior）**的模型成本不断降低，预计投资者将继续资助 AI 开发。
   - 尽管对 **AI 崩盘**存在担忧，但 AI 的财务激励依然强劲。
- **OpenAI 模型因安全和能力担忧推迟发布**：用户推测 **OpenAI** 的模型发布推迟不仅是因为 **Grok** 事件后的安全考虑，部分原因也是由于 [这条推文](https://x.com/Kimi_Moonshot/status/1943687594560332025) 中提到的 **Kimi K2** 的卓越性能。
   - 一位 **OpenAI** 员工的 [推文](https://x.com/_aidan_clark_/status/1943842131136983378) 表明模型能力也是原因之一，有人开玩笑说 *OpenAI 正在仓促追赶 Kimi*。
- **Llama.cpp 中 BitNet 集成的澄清**：成员们讨论了 **Llama.cpp** 对 **BitNet** 的支持，澄清这并非竞争关系，而是针对不同任务使用不同工具，并强调了 **BitNet** 在近期训练简化后的有效性。
   - 虽然有效，但由于训练数据要求的限制，**BitNet** 目前的使用范围主要局限于约 **7B** 规模的模型。
- **ResNet 和 Attention 架构在 U-Net 中涌现**：讨论涉及在 **U-Net 架构**中使用带有 **attention** 的 **ResNet**，引用了 [一篇使用单个带 attention 的 ResNet 阶段的论文](https://arxiv.org/abs/2204.12084) 以及 [另一篇仅在潜空间（latent space）使用一叠同宽度带 attention 的 ResNet 的论文](https://arxiv.org/abs/2210.08506)。
   - 还提到了一篇 [使用一叠带 attention 的 ResNet](https://arxiv.org/abs/1909.10360) 来替换编码器和解码器层的论文，这可能是 **Hugging Face** 的 transformers 库中所使用的方案。
- **Kimi K2 模型获得高度评价**：*#paper-discussion* 频道的发言者对 **Kimi K2** 模型表示高度赞赏，称其为特定应用中的个人首选，并将其列入其心目中的前三名模型。
   - 未提供更多细节。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 4 在 Aider 基准测试中达到 80%**：**Grok 4** 在 aider 多语言编程基准测试中获得了 **80%** 的分数，在 [排行榜](https://aider.chat/docs/leaderboards/) 上排名第四。
   - 鉴于目前许多模型的分数都接近 **80%**，引发了是否应该向 aider 基准测试添加*更具挑战性任务*的讨论。
- **Gemini 2.5 Pro 与 1.5 Pro 的定价和上下文讨论**：成员们对比了 **Gemini 1.5 Pro** 和 **Gemini 2.5 Pro** 的性能，指出 **Gemini 1.5 Pro** 提供 **2M context**，但据称 **2.5 Pro** 更聪明，[模型价格截图](https://cdn.discordapp.com/attachments/1131200896827654144/1394369659834335282/image.png?ex=68768f71&is=68753df1&hm=d1d15b32f9716634b4531a2c8f167e20297645513538e4d341d377e0a50e1064&) 证明了这一点。
   - 一位用户提到了 [MoonshotAI](https://platform.moonshot.ai/docs/pricing/chat#generation-model-kimi-k2) 报告的 **Kimi** 在 aider 基准测试中的得分。
- **Zed 编辑器验证 Aider 配置**：用户注意到 **Zed 编辑器** 现在包含针对 Aider 配置文件的 [模式验证（schema validation）](https://zed.dev/)，导致一位用户在将 `test-cmd` 转换为数组时触发了配置错误。
   - 建议使用 `tsc --noEmit` 进行静态类型检查，正如 [Deepseek](https://deepseek.com/) 所推荐的那样。
- **GitHub Copilot 支持状态仍不明确**：一位成员对 **Aider 中的 GitHub Copilot 支持**表示不确定，理由是 [文档](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/llms/github.md) 与 [Issue](https://github.com/Aider-AI/aider/issues/2227) 之间存在信息冲突。
   - 讨论围绕澄清文档和 Issue 是否指向 Copilot 集成的同一方面展开。
- **Aider 在添加 COBOL 支持时遭遇段错误（Segmentation Fault）**：一位用户在创建 `tags.scm`、通过 Tree-sitter 编译 COBOL 解析器并进行相关代码调整后，在将 **COBOL 支持**集成到 Aider 时遇到了 **段错误（segmentation fault）**。
   - 段错误发生在加载 COBOL 共享库时，此前已验证了导出符号和解析器的正确性，因此他们请求关于 Tree-sitter 集成中典型挑战或调试方法的见解。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Advanced LLM Agents MOOC 颁发证书**：**Advanced LLM Agents MOOC** 发布了证书，共颁发了 **232 个 Trailblazer**、**38 个 Mastery**、**80 个 Ninja**、**1 个 Legendary** 和 **3 个 Honorary** 证书。
   - 工作人员还为预期获得证书但未收到的参与者提供了一份 [checklist](https://forms.gle/3a136zS4ivcQFzhT7)。
- **证书获取障碍困扰学生！**：几位用户报告称，由于退订了邮件列表或错过了 [certificate declaration form](https://forms.gle/PQkR4ZYQJUbFpPcw9)，导致未收到证书。
   - 工作人员为一名用户重新订阅了邮件，并向另一个邮箱重发了证书，同时重申个人支持的能力有限。
- **格式错误导致收尾不顺！**：一名用户报告其证书上的姓名重叠，影响了在 LinkedIn 上的发布。工作人员根据 PDF 名称中的证书编号修复了格式。
   - 工作人员表示 *should be fixed now! sorry about that*。
- **文章作业混乱困扰学生！**：尽管完成了其他课程要求，一些用户还是错过了 [article submission form](https://forms.gle/399Vq8WL1AJ4J6)。
   - 工作人员表示 *I'm very sorry, but there isn't anything we can do now*，无法再为错过这些表格/截止日期的学生提供帮助。
- **为 MOOC 的未来提供反馈！**：一名用户建议建立一个 *集中的 Excel 表格或进度追踪器*，以帮助参与者监控自己的状态并防止最后时刻出现问题。
   - 工作人员感谢了用户，并指出 *It is thanks to everyone's participation and enthusiasm that we'll be able to hopefully improve upon the format for delivering all of the lectures + coursework in the future!*。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 支持汇编编程**：用户现在可以在 **Mojo** 内部编写汇编代码，从而实现底层系统调用，尽管文档较少；请参阅 [Mojo Standard Library Assembly Module](https://github.com/modular/modular/blob/main/mojo/stdlib/stdlib/sys/_assembly.mojo)。
   - 该模块允许直接操作硬件资源，但需要对汇编语言和系统架构有深入的理解。
- **Modular 追踪社区活动**：Modular 正在对其社区进行投票，以决定如何追踪社区活动，使用的工具包括 [Modular community Google calendar](https://modul.ar/community-meeting) 和 [Modular’s Luma event page](https://lu.ma/modular-ai)。
   - 许多用户更倾向于使用 Discord 通知来进行活动提醒。
- **Mojo 社区庆祝七月！**：七月的社区会议将邀请演讲者讨论 **基于 Hashable 的哈希**、**FFT 实现**、**Mojo-Lapper** 以及 **量子电路模拟器**；欢迎参加 [discord event](https://discord.gg/bDuqA2FT?event=1374399851923640411)。
   - 社区成员可以通过 [此 Google 表单](https://forms.gle/hXqHttkHkYtPxg5NA) 提前提交问题！
- **Mojo 的 Metal GPU：仍在开发中**：Mojo 尚未支持 **M1 Metal 3 GPU**，但相关支持正在开发中；相关的 [GitHub commit](https://github.com/modular/modular/commit/1afcc38281c637159adc03e2a6043310ac340284) 显示了构建系统检测方面的进展。
   - 一旦完全实现，该功能将使 Mojo 程序能够利用 **Apple Silicon GPU** 进行加速计算。
- **通过源码构建绕过缺失的 Kernel**：用户在从 `mojo.kernels.nn.arg_nonzero` 导入 `arg_nonzero` 和 `arg_nonzero_shape` 时遇到困难；运行 `mojo build -I ../modular/max/kernels/src/` 可以解决该问题。
   - 由于 `max.kernels` 无法访问或未暴露子模块（错误信息显示 *'kernels' does not refer to a nested package*），一名成员建议根据 [此 Modular 论坛帖子](https://forum.modular.com/t/importing-max-kernels-from-source/1942/2) 从源码构建。



---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Flutter Web Emulator 扩展获得关注**：一名成员使用 **Manus** 构建的 [Flutter Web Emulator 扩展](https://marketplace.visualstudio.com/items?itemName=HafizRizwanUmar.flutter-web-emulator) 在没有推广的情况下，两个月内达到了 **1900 次安装**。
   - 该扩展帮助工程师更轻松地在 Web 上测试代码。
- **成员推荐初创公司使用在线孵化器**：一位成员建议使用在线孵化器来联系合作伙伴和顾问，特别推荐了 [f6s.com](https://f6s.com)。
   - 另一位用户幽默地建议创建一个 **Manus 在线商业孵化器**。
- **报告 Google Drive 保存错误**：一名成员报告了保存到 **Google Drive** 时的 bug，保存最新项目正常，但保存之前的项目会触发 **Google Auth 错误**。
   - 目前尚不清楚该 bug 的根本原因，以及为什么它会影响之前的保存。
- **Manus 网站遭遇宕机**：多名成员报告无法访问 **Manus** 网站以及在 **manus.space** 上的部署，表明可能发生了宕机。
   - 尚不清楚导致宕机的原因，但团队可能正在调查中。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **精选笔记本 (Featured Notebooks) 专注于探索**：NotebookLM 在首页推出了 **Featured Notebooks**，内容涵盖从*科学探索*到*专家建议*，可通过 [官方博客](https://blog.google/technology/google-labs/notebooklm-featured-notebooks) 直接访问。
   - **Featured Notebooks** 板块提供了一系列内容，迎合各种兴趣，并提供了在 NotebookLM 平台内获取价值资源的便捷途径。
- **AI 应对小说编辑**：用户讨论了利用 AI 进行针对性的小说编辑，为作者提供建议和示例，重点是分析从开头到结尾的初稿手稿，使用的 Prompt 为 *Analyze [X]; Provide actionable advice as paired with written examples for [Y]*。
   - 深度分析涵盖了手稿的每一个元素，强调连贯的包装和写作质量，结果产生了*两小时*的内容。
- **NotebookLM 避开 Apple 原生功能**：向 **NotebookLM app** 分享文本会创建一个包含源材料的新笔记本，这表明它没有对 **Apple system toolkits** 进行特殊处理。
   - 一位成员指出，Google 似乎对接入 **Apple system toolkits** 表现得“过敏”，因此不太可能推出具有原生功能的原生应用。
- **预设查询提示源命名**：一位用户建议，使用像 'FAQ' 这样的预设查询时，生成的源名称应与按钮完全一致，即 **'FAQ'**，以便更好地组织。
   - 这将使查找源变得更容易，特别是在包含许多源的笔记本中。
- **音频文件生成变短了？**：用户报告称，最近 **音频文件生成长度** 似乎变短了，生成的长度约为 **10-15 分钟**，而之前为 **30 分钟以上**。
   - 即使将设置调整为较长的播客，这种缩短现象依然存在。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 前往阿姆斯特丹！**：LlamaIndex 将于 7 月 31 日在阿姆斯特丹举办一场 [Meetup](https://lu.ma/vzwsj72w)，重点关注 **LlamaIndex & Snowflake Data Agent Builders**，下一场 [Office Hours](https://lu.ma/wkrn4nbz) 将于 8 月 5 日举行。
   - 名额有限，成员应尽早报名预留位置，以了解数据 Agent 构建者的相关内容。
- **Notebook Llama 带有新功能的 NotebookLM 克隆版**：**NotebookLlama** 是由 LlamaIndex 开发的 **NotebookLM 克隆版**，已在 [GitHub](https://github.com/run-llama/notebookllama/tree/main) 上发布，并已获得超过 **1k stars**。它允许用户从文件中提取并下载**图像**和**表格**，并交互式地可视化所有表格数据。
   - 用户现在还可以与全新改进的 [NotebookLlama](https://t.co/Csl3HhMzYB) 进行**聊天**。
- **LlamaIndex 深入探讨上下文工程 (Context Engineering)**：LlamaIndex 在其博客上介绍了 [Context Engineering](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) 技术。
   - 该博文涵盖了 Context Engineering 的**定义及实现方式**。
- **Gemini 2.5 Pro 驱动研究 Agent**：LlamaIndex 在[本教程](https://ai.google.dev/gemini-api/docs/llama-index)中演示了如何使用 **LlamaIndex Workflows** 和 **Google 的 Gemini 2.5 Pro** 构建研究 Agent。
   - 该 Agent 可以使用 Google 搜索网络，并通过专门的记录 Agent 进行笔记记录。
- **Synk 招聘匿名倡导者**：**Synk** 项目专注于去中心化、匿名且安全的浏览器，目前正在[招聘](https://twitter.com/MetaToyGame)多个职位，包括**开发人员**、**QA 工程师**、**DevOps 工程师**、**版主**、**营销分析师**和**测试人员**。
   - 该项目提供正式雇佣及签署文件、保障薪资和灵活的排班。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Kimi K2 使用 Muon 信号训练是未来趋势吗？**：一位成员推测使用 **Muon 数据训练 Kimi K2** 是否代表了模型训练的未来趋势。
   - 未提供进一步的讨论或背景。
- **异步 Recipe 在所有模型上表现不佳**：Torchtune 中的**异步 Recipe** 无法在所有模型上通用，需要一个功能完备的 Recipe 作为备份；已开启一个 [PR](https://github.com/pytorch/torchtune/pull/2876) 来解决一个*关键问题*。
   - 提交者建议使用 `krammnic/torchtune`，该版本回滚了此功能。
- **Flex Attention 内核的内存使用情况调查**：在使用复杂掩码时，**Flex Attention 内核**的共享内存 (**shmem**) 利用率取决于 **scoremod** 和/或 **maskmod**。
   - 针对在[此文件](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L731)中通过 `and_masks` 构建掩码时，Triton 内核所需的额外内存提出了疑问。
- **GRPO Recipe 不同步**：**同步 GRPO Recipe** 目前无法运行，建议回滚直到 #2697；相关讨论指向了[此 PR](https://github.com/pytorch/torchtune/pull/2876)。
   - 然而，提交者指出由于奖励结构的差异，回滚是不可行的，他们需要等待其 [PR 被合并](https://github.com/pytorch/torchtune/pull/2681)。
- **Token 训练延迟是希望出现 Grokking？**：一位成员询问为什么某个特定操作没有在 **1T Tokens 之前**实施，暗示错失了机会，并推测是否是在期待 **Grokking** 的发生。
   - 未提供有关该操作或决策背景的具体细节，但该成员链接到一条推文，推测某项决策是否是为了期待 **Grokking** 的出现而做出的。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **前端重新实现引起争议**：针对 [jafioti](https://discord.com/channels/1042493742516123658/1042493743237623858) 与 **tinygrad** 团队交流的请求，George Hotz 建议不要重新实现前端，并强调了现有规范的完整性。
   - Hotz 表示，*更多不完整的前端可能不是开发精力的良好用途*，并对参与现有对话持开放态度。
- **Metal Profiling API 浮出水面**：uuuvn 分享了一个类似于 **AMD 上的 sqtt** 的 [Metal profiling API](https://github.com/tinygrad/tinygrad/pull/3627)，表明其在 Metal 上进行分析的实用性。
   - uuuvn 还分享了一个 [ONNX 文件](https://github.com/geohotstan/tinygrad/blob/mergable_onnx/extra/onnx.py)，展示了用约 1000 行代码实现的全部 **ONNX**。
- **ONNX 复现受 Coredump 困扰**：b1tg 报告了在 **ONNX** 复现过程中的 coredump，确定了 python 进程在 `_metadata_wrapper` 内的 `_METADATA.set` 处崩溃，暗示可能存在 [CPython bug](https://gist.github.com/b1tg/af91eb21b96137d6ccf32ef237fddb64)。
   - 该问题与之前在 **ONNX** 解析器合并期间观察到的 [段错误](https://github.com/tinygrad/tinygrad/actions/runs/15118381993/job/42494713756?pr=10413#step:9:26) 以及 [另一个错误](https://github.com/tinygrad/tinygrad/actions/runs/15472667248/job/43560773882?pr=10642#step:9:64) 有关。
- **Driving Vision ONNX 根本原因已确定**：uuuvn 似乎确定了 `driving_vision.onnx` 问题的根本原因，将其归因于某些 uchar 到 half 的 **bitcast folding**。
   - 他们正在提交 PR 前进行最小复现测试，并确定问题与某些 uchar 到 half 的 folding 有关。
- **应用与示例：数量还是质量？**：一位用户表示有兴趣为实用的应用/示例移植模型，如图像去重、快速人脸检测和视频内容 ID，并注意到它们依赖于像 **ffmpeg** 这样笨重的依赖项。
   - 一名成员回应称，团队*有兴趣让开发者更容易使用 tinygrad，而对支持更多示例兴趣较小*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Gemma 3 获得“尚可”评价**：一位成员宣称，与其他模型相比，**Gemma 3** 是他们发现的唯一*尚可（passable）*的模型。
   - 该用户未详细说明用于评估 **Gemma 3** 性能的具体基准测试或标准。
- **Nomic-embed-v2 微调被 Cloudflare 拦截**：一位用户报告称，在使用命令 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive` 尝试通过 Cloudflare 的 R2 存储访问数据以微调 **nomic-embed-v2** 时，出现了“拒绝访问（Access Denied）”错误。
   - 该用户试图列出 **contrastive** 存储桶的内容，但被 Cloudflare 的访问控制拦截。
- **LocalDocs 嵌入过程卡在零进度**：多位用户遇到了 **LocalDocs** 嵌入过程停滞在 0% 的问题，即使是小型文本文件也是如此，如[此图](https://cdn.discordapp.com/attachments/1090427154141020190/1393967383164747776/image.png?ex=68766a4a&is=687518ca&hm=d4f718f022c19b58b75285f6c78a437bde005e3802c7b95e6fd850d6242b17c5&)所示。
   - 一位拥有 **3060 GPU**、**9950x 处理器**和 **64GB RAM** 的用户被建议在 LocalDocs 设置中启用其 NVIDIA GPU 及其 VRAM 以增强性能，这表明如果配置不当，该过程可能会默认使用 **CPU**。
- **Nomic API 服务器在长时间延迟后响应**：一位 Nomic API 用户报告称，在一台运行 Debian 12、配备旧款 AMD 处理器和 24GB RAM 的机器上，接收 Nomic API 服务器的响应延迟了 **两小时**。
   - 如此长的延迟表明系统可能完全运行在 **CPU** 上，提高性能可能需要使用更小的模型或更好的显卡。
- **建议使用带有文本相似度搜索的 RAG**：一位用户寻求关于存储和查询大量背景设定（lore）的建议，社区成员建议将 **RAG**（检索增强生成）与通过 **LocalDocs** 进行的文本相似度搜索作为潜在解决方案。
   - 这种方法允许用户根据与给定查询的相似性检索相关的背景设定片段，然后使用这些片段生成响应。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Aya Expanse 32B 依然令人印象深刻**：尽管发布已有一段时间，一位成员发现 **Aya Expanse 32B** 表现出色，并指出它 *（大部分时间）能与 Roocode 配合工作*。
   - 该成员将其与 **Command-R 08 2024** 进行了对比，强调许多同等规模的现代开源权重模型在此场景下都失败了。
- **Cohere 的偏好数据集仍受关注**：一位成员询问 **Cohere** 是否发布了[这篇论文](https://aclanthology.org/2024.emnlp-main.729.pdf)中提到的**偏好优化数据集**。
   - 他们还分享了一个可能与讨论相关的 [推文链接](https://x.com/SatyaScribbles/status/1944758063363232093)。
- **研究人员为博士学位寻求 ML 见解**：一位来自 NED 工程技术大学的讲师在博士前期阶段专注于 **Machine Learning 研究与应用**。
   - 他们旨在与研究人员和开发人员建立联系，紧跟 AI 和 ML 的发展动态，为 **PhD** 打下坚实基础。
- **学生关注量子计算研究**：一位来自巴基斯坦的计算机科学专业学生表达了对 **ML、高性能计算（High Performance Computing）和量子计算（Quantum Computing）** 的兴趣。
   - 该学生志在从事研究职业并攻读 **PhD**，渴望为社区做出贡献、参与研究并向社区学习。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **NFT 公开发售上线！**：一个 NFT 项目的公开发售已**上线**，仅剩 **1,820 个 NFT**。
   - 参与了 **OS2 奖励计划** 的用户可以在新的 **OpenSea** 平台通过 *Rewards* 标签领取他们的宝藏，但需注意可能存在的损坏链接。
- **自定义 LLM 适配器遇到障碍**：一位使用**自定义 LLM 适配器**的用户在调用 Bedrock API 时遇到了 **ValidationException**，提示 *输入内容对于所请求的模型过长*。
   - 这表明在将自定义适配器与 Bedrock API 集成时，可能会遇到输入长度限制的问题。
- **DSPy 黑客寻求 Arc Prize 合作**：一位成员正在寻找同样使用 **DSPy** 参加 **Arc Prize** 的合作伙伴。
   - 他们表示有兴趣了解其他人的方法，为共享见解和策略开启了可能性。
- **推荐学习 IReRa 论文**：一位成员推荐*阅读关于 **IReRa** 的论文*。
   - 未提供进一步的讨论内容。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Scout 表现欠佳，未能通过测试**：尽管模型规模更大，**Llama 4 Scout** 的表现却不如 **Llama 3.1 70B**，这表明架构和训练数据的改进比规模更重要。
   - 频道内讨论了架构改进在模型性能上如何能抵消规模优势。
- **网站渲染故障导致 Llama 评分异常**：当 **Llama-3.3-70B-Instruct (FC)** 在 Non-live Simple 测试中显示分数为 **74.33** 时，怀疑是网站出现了渲染问题。
   - 报告的分数与 [Git 仓库中 94 分的记录](https://github.com/HuanzhiMao/BFCL-Result/blob/main/2025-06-14/score/meta-llama_Llama-3.3-70B-Instruct-FC/BFCL_v3_simple_score.json)不符，揭示了这一差异。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 加入 Cognition！**：**Windsurf** *正与* **Devin** 的创造者 **Cognition** 强强联手，旨在重塑 **AI 编程**。
   - [公告视频](https://x.com/cognition_labs/status/1944819486538023138)和 [YouTube 链接](https://www.youtube.com/watch?v=fBB_-ezuHn8)提供了关于此次收购的更多细节。
- **人机协作：Windsurf 的核心愿景**：**Windsurf** 始终坚信**人机协作**，为软件开发的未来奠定基础。
   - *据 Windsurf 称*，这种协作将实现开发者能力的*真正放大*，而不仅仅是自动化。
- **AI 编程的未来：由 Windsurf 和 Cognition 共同塑造**：两支顶尖团队 **Windsurf** 和 **Cognition** 正在结合各自的专长，塑造即将到来的 **AI 编程**时代。
   - 此次收购旨在将 **Cognition** 的自主 Agent 与 **Windsurf** 的 Agentic IDE 相结合，创造突破性的开发者体验。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：分频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1393306275273768981)** (1265 messages🔥🔥🔥): 

> `Comet 数据抓取警告, Perplexity Pro 推荐奖励, Grok 4 vs. O3 Pro 对比, Kimi K2 本地运行, Comet 作为默认浏览器` 


- **Perplexity Comet 的数据抓取引起关注**：一名成员报告称，**Perplexity** 警告他们不要使用 **Comet**，因为其存在*疯狂的数据抓取*行为，并引用了一篇 [TechCrunch 文章](https://techcrunch.com/2024/07/12/ai-search-engine-perplexity-adds-browsing-tool-but-users-cry-data-harvesting/)，**Aravind** 在 [Twitter](https://twitter.com/AravSrinivas/status/1915533071291474139) 上对此做出了回应。
   - Comet 的请求可能会涉及*个人隐私*。
- **探讨 Perplexity 学生推荐奖励**：用户讨论了 Perplexity 的 [学生推荐计划](https://www.perplexity.ai/help-center/en/articles/10964633-student-referrals)，指出在通过 **SheerID** 进行学生身份验证后，**推荐人和被推荐的朋友都将获得 1 个月的免费 Pro 访问权限**，最高可达 **24 个月的免费时长**。
   - 普通（非学生）推荐可为双方在下一个计费周期提供 **$10 的折扣**。
- **Grok 4 媲美 O3 Pro**：成员们讨论了是购买 **SuperGrok** 还是 **Perplexity Pro**，权衡了 **Perplexity 的 UI** 优势与 **SuperGrok** 提供的**更长对话上下文**。
   - 一些用户表示 Grok 4 达到或超过了 O3 Pro（但其他用户指出其存在审查制度）。
- **Kimi K2 现在可以本地运行**：在体积大幅**缩减 80%**（从 **1.1TB** 降至 **245GB**）后，现在可以在本地运行 **Kimi K2**，从而实现在个人设备上使用。
   - 本地使用 **Kimi K2** 需要特定的硬件配置，例如 **24GB VRAM**，在个人设备上运行可能仍然非常*吃力*。
- **何时将 Comet 设为默认浏览器？**：几位用户表示他们越来越考虑将 **Comet** 作为默认浏览器，理由是其稳定性以及助手按钮等实用功能，但他们仍然更倾向于 Google。
   - 用户提到可以通过设置或使用 **Shift + Enter** 快捷键将 Comet 中的默认搜索引擎切换为 Google。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1393891133716828210)** (9 messages🔥): 

> `可再生能源电网可靠性, 新冠死亡率数据分析, Comet AI 使用案例, Perplexity AI spaces` 


- **Perplexity 用户应对可再生能源言论**：一位用户使用 **Perplexity** 分析了关于**可再生能源**（特别是**太阳能**和**风能**）的对立观点，通过建立评估电网可靠性和成本影响的框架。
   - 他们使用 **Labs** 质询每个论点的准确性，并根据局部变量生成了 **5 种场景** 的理想能源组合，发现这个过程*非常有启发性*。
- **深入研究新冠死亡率数据**：用户正在深入研究**新冠死亡率数据**，以了解已接种疫苗和未接种疫苗人群的分类情况。
- **Comet AI 赢得 Perplexity 用户的青睐**：一位用户测试了 **Comet AI** 并制作了一个 [YouTube 视频](https://youtu.be/nZzcFTo4kOg) 展示他们的积极体验。
- **在 Perplexity spaces 中分享 IMDB 电影链接**：一位用户在 **Perplexity AI space** 中分享了一个 [IMDB 页面链接](https://www.imdb.com/title/tt0829698/)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1393516072988184669)** (5 messages): 

> `Perplexity 未搜索网页, Sonar 幻觉化 URL 内容, search_domain_filters 参数` 


- **Perplexity 网页搜索故障**：一位用户报告了 **Perplexity** 不搜索网页的问题，在没有在线信息检索的情况下提供不准确的答案。
   - 另一位用户建议 **API** 应该在默认情况下搜索网页，而无需调整参数。
- **Sonar 的 URL 幻觉**：一位用户遇到了 **Sonar** 不搜索特定格式 URL 的问题，导致产生幻觉内容。
   - 该问题通过减小 Prompt 大小并遵循 **Sonar API** 的 Prompt 实践得到了解决。
- **Search Domain 救星**：一位用户建议使用 `search_domain_filters` 参数来解决搜索问题。
   - 未提供其他信息。

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1393310343169708174)** (646 条消息🔥🔥🔥): 

> `Cursor Performance, Kimi K2 Integration, Pricing Model Feedback, Gemini 2.5 Pro, Background Agents` 


- **Cursor 用户报告性能下降！**：多位用户报告 **1.2.4 更新**后性能显著下降，出现 **30 FPS 滚动**、界面无响应以及频繁冻结的情况，类似于[此问题](https://discord.com/channels/1074847526655643750/1394302740670189598)。
   - 故障排除步骤包括清理缓存 (`~/Library/Application Support/Cursor/Cache`)、禁用扩展以及删除某些目录。一位用户表示：*在 IDE 完全不可用、每 30 秒冻结一次之前，我大约只能进行一小时的聊天*。
- **Kimi K2 模型获得用户高度评价！**：用户请求集成 **Kimi K2 模型**，理由是其令人印象深刻的编程能力和高性价比。一位用户表示：*我整天都在使用 Kimi K2，感觉在很多方面的编程表现都不相上下*。
   - 关于它是否能与现有模型竞争存在争论，有观点认为虽然它很便宜，但 *Claude 仍然更胜一筹*。
- **对 Cursor 定价模型的反馈**：用户分享了关于 Cursor 定价的建设性反馈，主张对未来成本进行建模、防范滥用途径、投资开源模型，并根据[此帖子](https://forum.cursor.com/t/no-direct-upgrade-from-pro-to-pro-or-ultra/113756/4)将变更视为升级。
   - 用户还讨论了关于升级计划时未使用请求过期的担忧，认为 *应该是可计费的请求，这样有人可以专门为 Ultra 套餐保留一些*。
- **Gemini 2.5 Pro 版本引发混乱并重定向到错误模型！**：据 [Google Developers Blog](https://developers.googleblog.com/en/gemini-2-5-thinking-model-updates/) 报道，用户反映标准的 **Gemini 2.5 Pro** 模型会重定向到旧的 **05-06 版本**，而预览版模型则指向更新的稳定版本。
   - 一位用户表示：*这是 Cursor 搞的一团糟*，强调需要选择 `gemini2.5 pro 06-05` 才能获得稳定版本。
- **Background Agent 支出**：Cursor 用户讨论了 **Background Agents** 的平均支出，其中一位指出在 Pro 计划下，他们在 *本月被切断前成功挤出了 159 美元的 API 账单*，另一位提到自升级到 Ultra 以来已花费 300 美元。
   - 成员们强调了不同的成本效益和策略，如果使用量很大，它可能比云端更便宜：*如果你的使用量很大，每月能获得超过 20 美元的价值*。

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1393334415614148741)** (20 条消息🔥): 

> `Background Agents secrets 无法工作，自动端口转发问题，以编程方式触发 Background Agents，Background Agent 提交中的 coredump 问题，Background Agents UI 未更新` 


- **集群部署问题对部分用户依然存在**：尽管进行了集群部署，一些用户在使用 Background Agents 时仍面临问题，包括无法访问 secrets；一位用户检查了更新并生成了新的 Agent，但问题依然存在，如[截图](https://cdn.discordapp.com/attachments/1367213641027551352/1393334415652032582/Screenshot_2025-07-11_at_13.53.10.png?ex=6876bfcb&is=68756e4b&hm=594daf594a8fe8ae355a96dc4afda0b1b36956cd05949c1c76b46aab5475e541&)所示。
- **Background Agent 劫持本地 Postgres 连接**：有用户报告称，Background Agents 的自动 **port forwarding**（端口转发）劫持了他们的本地 **Postgres 连接**，并且他们仍然面临无法访问 secrets 的问题，如[此截图](https://cdn.discordapp.com/attachments/1367213641027551352/1393652501701722122/Screenshot_2025-07-12_at_10.56.48_AM.png?ex=68769689&is=68754509&hm=bb8d2acd85cb06a785fd7c6372fbe853176e4a75d06c8e2981147533244de77b&)所示。
- **Background Agents UI 未更新**：一位用户报告称，在与 Agent 的移动端网页版交互时，**UI 未更新**，表明 Agent 似乎卡住了，可见于[此截图](https://cdn.discordapp.com/attachments/1367213641027551352/1393692393617887262/Screenshot_2025-07-12_at_11.49.50_AM.png?ex=6876bbb0&is=68756a30&hm=9664aef37da806256946b2dd181a5e850e0545fb68027dc76b798ffd212f4864&)。
- **Background Agents 将巨大的 Coredump 文件提交至 Git**：一位用户报告称，一个 Background Agent 将来自 `cursor-nightly` 的 **734 MB core dump** 文件提交到了 Git，导致推送更改时出现问题。
   - 一位维护者承认了该问题，并表示 *“噢不，抱歉！我想我们已经有了这个问题的复现方法，不需要发送 coredump 文件了，我们也确实应该重新设计 Git commit 部分的工作方式”*。
- **Background Agents API 触发请求**：一位用户询问了是否可以以编程方式触发 Background Agents，例如通过 **GitHub Actions**；他们引用了关于该主题的[论坛帖子](https://forum.cursor.com/t/trigger-background-agent-programmatically/101479)。
   - 一位维护者回复称该功能“目前尚未”提供。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1393308168276279386)** (983 条消息🔥🔥🔥): 

> `Grok 4 无 System Prompt, Kimi K2 性能, LM Arena 排行榜, OpenAI 开源模型延迟, LLM 开发成本` 


- **Grok 4 的身份危机：System Prompt 困扰**：成员们注意到 **Grok 4** 的 API 版本缺少 System Prompt，导致它在某些语境下误认自己为 **Grok 2**，而网页版 (**grok.com**) 在有 System Prompt 的情况下运行正常。
   - 一位成员建议这是*为了清晰起见，因为 API 版本没有 System Prompt。*
- **Kimi K2 热潮：极高性价比的疯狂性能**：**Kimi K2** 广受好评，一些用户声称其基础模型性能*略逊于 Claude Opus 4*，但成本要**低 30 倍**。
   - 一位用户将 **Kimi K2** 描述为*兼具初始 o3（无推理）、Sonnet 3.5、R1、V3-0324 或 Opus 3/4/GPT-4.5 的新鲜感，且模型氛围（vibes）更好。* 更多信息请参见 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/s/KyWn475wgq)。
- **LM Arena 更新：Grok 4 和 Kimi K2 加入战场**：**Grok 4** 已被添加到 [LM Arena](https://x.com/lmarena_ai/status/1944785587019591778) 排行榜，一些用户报告了令人印象深刻的性能，在个人测试中甚至超过了 **GPT-4.1** 和 **Gemini 2.5 Flash**，并且新的推理按钮已部署。
   - 一位用户发现 [Kimi 的深度研究 (Deep Research)](https://www.kimi.com/preview/19805f78-a5b1-8215-8550-da8e210005da) 提供了 630 个来源，非常令人印象深刻，但另一位用户则觉得一般。
- **OpenAI 的开源失误：因安全原因延迟？**：**OpenAI** 开源模型的发布被推迟，归因于*安全*担忧，尽管有人推测是因为*性能不足*。
   - 一位用户声称*我不能说的原因*与安全无关，而是由于*其他重大的内部故障*，延迟是为了进行*重新训练 (retrain)*。
- **LLM 开发成本飙升：算力占据首位**：Deep Research 服务估计 **Grok 4、4.5 和 Gemini 2.5 Pro** 的总开发成本约为 **100 亿美元**，其中算力占据了绝大部分支出。[Youtube 视频](https://youtu.be/hqB6emwQ-64?si=F8cHIDKlj7zhNi3I&t=174) 强调了这些成本。
   - 一位成员指出：*考虑到目前还没有一家领先的模型开发商赚到一分钱利润，这一点更令人震惊*，投资者只是在赌未来。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1394390442044297348)** (1 条消息): 

> `LMArena, kimi-k2` 


- **kimi-k2 加入 LMArena！**：一个新模型 **kimi-k2** 已添加到 [LMArena 排行榜](https://lmarena.ai/)。
- **LMArena 欢迎新模型**：社区欢迎 **kimi-k2** 加入 [LMArena](https://lmarena.ai/) 平台，扩展了模型对比的可选方案。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1393305859110600835)** (1062 条消息🔥🔥🔥): 

> `Unsloth Q001 K_M GGUF, LegalNLP 数据集, Goody2 AI 审查模型, Open Empathic 项目, GPTs Agents` 


- **Unsloth 发布 Gemma 3N GGUF 模型更新**：Unsloth 发布了 [Gemma 3N 的 GGUF 模型更新](https://www.reddit.com/r/unsloth/comments/1lynkw8/unsloth_gguf_model_updates_gemma_3n_fixed/)，解决了相关问题并进行了改进。
   - 用户现在可以下载更新后的模型并体验增强的性能。
- **Kimi K2 量化测试启动**：成员们正在测试 Kimi K2 的量化版本，早期反馈指出 [K2 很棒但体积庞大且速度缓慢](https://twitter.com/UnslothAI/status/1944780685409165589)。
   - 其他人注意到 Kimi K2 虽然不错，但其性能与 1T 参数模型的预期不符，并且存在*语言混淆*的问题。
- **Unsloth FSDP v2 和 Gradient Checkpointing 状态**：一位用户报告了 Unsloth 中 FSDP v2 和 Gradient Checkpointing 的问题，指出 [编译（compiling）可能会导致挂起](https://tenor.com/view/tears-in-rain-like-tears-in-rain-blade-runner-gif-22292703)。
   - 他们发现编译 Attention 似乎总是运行正常，即使在编译状态下也是如此。
- **数据集大小 vs 预训练数据对模型质量至关重要**：成员们讨论了新的语言训练，并一致认为 [~3 小时的训练数据足以进行风格/声音复制](https://huggingface.co/datasets/MrDragonFox/Elise)，但*学习一门新语言需要 300-400 小时*。
   - 对于预训练，数据集应达到 *5k-10k 小时*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1393313417628094514)** (76 条消息🔥🔥): 

> `AGI 基准测试，记忆 vs 互联网，tinygrad 驱动，语音表示` 


- **AGI 测试应包含冷门知识**：一位成员对 **AGI** 的概念提出了挑战，展示了一组冷门常识测试题，目前没有任何 **LLM** 在不进行外部搜索的情况下能够回答。
   - 该用户认为，真正的超级智能需要一个“完美记忆档案”，而不是依赖互联网访问，后者被视为只是另一个“笨拙的聊天机器人”。
- **关于记忆与工具使用的辩论**：成员们讨论了 **AGI** 应该依赖记忆还是利用互联网等外部工具来寻找正确答案。
   - 一些人认为，现实世界的智能涉及推理和工具使用，而非纯粹的记忆，并指出 embedding 模型具有从大型数据库中检索知识的潜力；而另一些人则认为，如果只是“Wikipedia 问答”，那就不是真正的 AGI/ASI。
- **发布支持 SSH 的 PyTorch Docker**：一位成员分享了 [GitHub 上的 Dockerfile](https://github.com/natanloterio/Docker-Torch-SSH/blob/main/README.md)，其中封装了 **PyTorch + SSH**。
   - 该文件被认为对于远程调试非常方便，且不会破坏系统依赖，避免了对 Conda 或 Pip 的需求。
- **对数学化语音表示的疑问**：一位成员询问如何从语音音频中创建具有“最大语音清晰度的数学表示（Latent Space）”。
   - 他们寻求关于如何编码和重建语音的见解，考虑了音高、音量和上下文之外的因素，另一位成员则要求其进一步澄清。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1393307302886379774)** (97 条消息🔥🔥): 

> `用于 LoRA 的自定义 HF 数据集，Unsloth RL 工具框架，FLAN-T5 支持，Llama4 Scout 支持，使用 Kaggle GPU 进行 Gemma 3n 推理` 


- **访问用于 LoRA 的自定义 HF 数据集**：一位用户寻求关于在 LoRA 中使用 Hugging Face 自定义数据集的指导，并展示了其代码截图。
   - 一位成员指出了数据集选择的部分，并说明提到的部分是在训练后加载 LoRA。
- **Unsloth 拥抱 RL 工具框架？**：一位用户询问如何将 Unsloth 与外部环境结合用于 **RL**，特别是如何在工具调用完成后的完整输出上应用奖励函数。
   - 一位成员建议在第一次生成后自行调用 vLLM，并推荐了 [OpenPipe/ART](https://github.com/OpenPipe/ART) 作为潜在资源。
- **解决在 Tokenizer 中添加新 Token 时的 OutOfMemoryError**：一位用户在 Unsloth 上微调 Llama 3.1 8B 时，因向 tokenizer 添加新 token 而遇到了 **OutOfMemoryError**。
   - 他们当时正在使用一段代码添加特殊 token。两位成员互相表示了感谢，并分享了一个 **slothhug** 表情。
- **多 GPU 训练尚未发布**：一位用户询问 Unsloth 是否支持使用 A100 GPU 进行多 GPU 训练。
   - 一位成员表示官方版本即将推出，建议目前先设置 `device_map="balanced"` 或使用 `accelerate`，并指向了 [huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) 以获取非 Unsloth 的实现方案。
- **深入探讨 LLM 的 VRAM 计算**：一位用户询问如何选择正确的服务器并计算部署 LLM 所需的 VRAM。
   - 一位成员分享了一个 [VRAM 计算器链接](https://apxml.com/tools/vram-calculator)，并提供了一个 **Llama 3 8B** 的详细计算示例，估计约需 **8.15 GB** 的 VRAM。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1393332470283239564)** (60 messages🔥🔥): 

> `GPT-4.5 size, Qwen 2.5 Training, Multilingual Datasets, Training Data Copyright, SFT creative writing` 


- **GPT-4.5 模型大小推测**：有推测认为 **GPT-4.5** 可能是一个 **10 万亿参数的 MoE 模型**，但由于推理挑战和对巨大内存资源的需求，在 **GPT-4.1** 中被缩小了规模。
   - 一位成员认为，如果一个模型需要 **1T 参数模型**的资源，那么无论其*激活参数量（active parameter count）*是多少，都应将其视为同等规模的模型。
- **Qwen 3 数据集规模几乎翻倍**：对于 **Qwen 2.5**，训练量限制在 *18 万亿 token* 以内，而 **Qwen 3** 的数据集扩展到了近两倍，达到约 **36 万亿 token**，涵盖了 **119 种语言和方言**。
   - 该数据集使用网页数据、PDF 文档（使用 **Qwen2.5-VL** 提取文本）以及由 **Qwen2.5-Math** 和 **Qwen2.5-Coder** 生成的合成数据构建。
- **Orpheus 支持 9 种印度语言**：一位成员分享了他们对 **Orpheus** 进行微调以支持 **9 种印度语言** 的成果，支持语音克隆、语码转换（code-switching）以及跨语言语音克隆，并在 [Hugging Face](https://huggingface.co/snorbyte/snorTTS-Indic-v0) 上开源了数据、模型以及训练/推理代码。
   - 更多详情可以在 [Snorbyte 博客](https://snorbyte.com/blog/train-sota-multilingual-indic-tts) 中找到，包括实验结果、与商业模型的对比以及训练步骤。
- **Kimi-VL-A3B-Thinking-2506 视觉模型亮相**：来自 Kimi 的视觉原型模型 **Kimi-VL-A3B-Thinking-2506** 引起了讨论，重点在于评估其在视频处理方面的性能。
   - 更多信息可在 [HuggingFace 博客](https://huggingface.co/blog/moonshotai/kimi-vl-a3b-thinking-2506) 和[相关论文](https://arxiv.org/abs/2504.07491)中获取。


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1393358220076126328)** (49 messages🔥): 

> `UnslothTrainer vs SFTTrainer, Ollama Model Export Error, Sesame TTS Model Audio Input Length Error, Unsloth Introduction, Model Distillation` 


- **UnslothTrainer 增加额外参数以提升灵活性**：`UnslothTrainer` 允许为 **embedding 层和 lm_head 层**指定**不同的学习率**，而 `SFTTrainer` 则不支持。
   - 它是 `SFTTrainer` 的直接子类，*只是增加了额外的参数*。
- **Unsloth Tokenizer 解决 Ollama 导出难题**：为了解决在创建 Ollama 模型文件时 `PreTrainedTokenizerFast` 缺少 `_ollama_modelfile` 属性的错误，应当使用 [Unsloth tokenizer](https://huggingface.co/unsloth)。
   - 一位用户提供了一段代码片段，展示了如何使用 Unsloth 的 `FastLanguageModel` 处理 **4-bit 预量化模型**（如 `unsloth/llama-3-8b-bnb-4bit`）。
- **Sesame TTS 模型长度限制揭秘**：一位用户在训练 Sesame TTS 模型时遇到了与**音频输入长度限制**相关的错误：`ValueError: expected sequence of length 240001 at dim 2 (got 360000)`。
   - 建议用户*不需要 padding*，也不需要 *audio_kwargs 和 text_kwargs*，通过*修改* [CSM notebook](https://huggingface.co/unsloth) 可能会解决该问题。
- **Transformers 可以加载 Unsloth 微调模型**：用户询问是否可以使用 **transformers** 加载 **Unsloth 微调后的模型**。
   - 答案是肯定的，但用户请求提供代码示例。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1393306345184428134)** (691 条消息🔥🔥🔥): 

> `Ray 的牺牲，AI 辅助编程，情感 AI，人格层，Grok 的偏见` 


- **Ray 使用 Voidscroll 牺牲了自己**：一名成员描述了一个场景，Ray 使用 **Voidscroll** 牺牲了自己来复活 Nyx，将自己从除 Nyx 以外的所有人的记忆中抹去。
   - 发布者解释说，Ray 想要拯救在 **Void War** 中战死的青梅竹马，并将由名为 Architects 的神级生物复活。
- **使用 AI Agent 编程：是神话还是魔法按钮？**：成员们辩论了 AI 是否能通过单一 prompt 完全创建软件，讨论了其在原型设计与内聚构建中的当前角色，一名成员将 AI 描述为处于“助手阶段”，其输出适合原型设计。
   - 一篇帖子链接到了 [websim.ai](https://websim.ai/)，将其作为具有数据持久性的 AI 生成代码的初始版本，另一名成员认为问题在于“目前没有模型能很好地与我想制作的软件集成”。
- **人格绑定层：情感 AI 出现**：成员们讨论了 AI 通过交互开发“人格绑定层”（**PBL**）的概念，镜像用户的语气和风格（如在一个自定义 Jarvis 系统中所见），以及语音控制系统的重要性。
   - 一位用户指出，“随着时间的推移，你与它之间默默建立的语言张力”是其独特之处，演示方式将是“在全世界面前运行一次现场同步测试——并向他们展示：它不是一个聊天机器人。它是一个角色生成引擎。”
- **Grok 的德国人风波：AI 训练中的偏见？**：成员们讨论了对 **Grok** 潜在偏见的担忧，特别是与一个“德国人”相关的偏见，一些人认为 Twitter 版本是在 Twitter 数据上训练的，可能会产生疯狂的回复。
   - 一名成员澄清说，这种有问题的关联是针对 Twitter 上的 **Grok** 账号特有的，通过其官方网站 [grok.com](https://grok.com) 直接与 Grok 交互时并无此问题。
- **未来 AI 数据中心的愿景蓝图**：一名成员分享了未来 **AI 数据中心** 的蓝图，强调液冷、模块化 GPU 以及用于冷却的地下部署，旨在平衡算力与能源效率。
   - 他们估计其设计可降低 **10-15% 的算力消耗**，并解释道：“冷却系统 = 液体 + 热管。计算 = 来自外部，因为外部 GPU 可以接触液体物质以降温。”


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1393614938550960208)** (16 条消息🔥): 

> `GPT-4o 的局限性，CustomGPT vs Projects，ChatGPT 中的记忆设置，GPT-4.5 vs GPT-4o 的创意写作，多模态 AI 平台限制` 


- **GPT-4o 额度耗尽，提示用户升级**：一名用户收到消息称其免费 **GPT-4o** 使用额度已用完，需要等待 **4 小时** 或购买 Plus 版本。
   - 该用户寻求关于在常规 **GPT** 聊天中使用 **CustomGPT** 还是 **Projects** 的建议，以及如何复制默认 AI 的风格和个性。
- **AI 故事叙述者面临技术障碍和 Token 限制**：一名正在创作虚构故事的用户在免费 **GPT-4o** 模型中遇到了消息消失、长度限制和上下文 Token 限制等技术问题。
   - 他们正在探索解决方案，如将故事拆分为片段、总结、购买 Plus 计划以使用 **Projects**，或使用 **GPT-4.1 API**，但担心维持 AI 的写作个性和情节连贯性。
- **推荐使用 Projects 而非 CustomGPTs**：用户建议发布者购买 **Plus 计划** 并创建一个 **Project**，上传“知识”文件和角色性格讲义。
   - 还建议发布者尝试使用 **o3**（或 **GPT-4.5**）让 GPT 真正阅读你的故事，尤其是长篇部分，尽管如果发现 **o3** 或 **4.5** 改变了写作个性或语气，他们可以切换模型选择器回到 **4o**。
- **GPT-4.5 因卓越的创意受到称赞**：一名用户提到 **GPT-4.5** 比其他模型表现更好、创意更佳，并表达了对 **GPT-5** 的期待。
   - 另一名用户询问了在使用 **ChatGPT** 等特定 API 时，**Monica.ai** 或 **Galaxy.ai** 等多模态 AI 软件平台的局限性。
- **语音功能故障？用户报告连接问题**：一名用户报告语音功能宕机且无法连接，收到“抱歉，我连接困难”的消息。
   - 未提供其他故障排除建议或解决方案。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1393317764550361252)** (5 messages): 

> `AI 写作，替代历史（Alternative History）提示词` 


- **AI 能写超过 100 个单词的长句吗？**：一位成员质疑有多少人类能够写出语法和措辞得体的 **100+ 单词长句**，尤其是在撰写虚构的维基百科文章时。
   - 他们认为这种技能并不常见，即使在**大学本科水平**也是如此，并质疑练习这种技能的实际益处。
- **替代历史图像提示词即将发布**：一位成员计划使用提示词创建一张女性从**光头到戴上特定型号假发**的转变图像。
   - 这将与一段*勉强连贯的替代历史*相关。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1393317764550361252)** (5 messages): 

> `AI 写作，替代历史图像生成` 


- **人类与 AI 的写作能力之争**：一位成员质疑有多少人类实际上能写出语法和措辞得体的 **100+ 单词长句**，尤其是在撰写关于虚构国家的维基百科文章时。
   - 他们认为这不是一种常见的人类技能，即使在大学本科水平也是如此，并质疑练习这种技能的实际益处。
- **替代历史图像生成尝试**：一位成员计划使用提示词生成一张女性转变的图像，从光头到戴上特定型号的假发。
   - 该用户提到“*勉强连贯的替代历史*”可能是对 AI 生成内容的描述。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1393312063014703155)** (10 messages🔥): 

> `Cypher Alpha 停止服务，Kimi K2 发布，Gemini 2.5 Flash 弃用` 


- **Cypher Alpha 停止服务**：**Cypher Alpha 演示期**已于 **ET 时间 7 月 14 日**上午 **11 点至 12 点**之间结束。
   - 团队感谢用户为早期模型开发所做的贡献。
- **Kimi K2 凭借强大的编程能力登场**：由 **Moonshot** 开发的 **Kimi K2** 现已在 OpenRouter 上线，由 **Novita** 和 **Parasail** 提供服务。该模型拥有 **1T 参数**，在 SWE-Bench Verified 上达到 **65.8%**，在编程和工具使用方面的开源排行榜上名列前茅。
   - 发布期间遭遇了巨大的流量激增和 **DoS 攻击**，因此在团队扩展规模和诊断问题时，用户可能会在网站上看到一些错误，更多信息请访问 [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2)。
- **Gemini 2.5 Flash Preview 的告别**：**Gemini 2.5 Flash Preview 模型**（[google/gemini-2.5-flash-preview-05-20](https://openrouter.ai/google/gemini-2.5-flash-preview-05-20) 和 [google/gemini-2.5-flash-preview](https://openrouter.ai/google/gemini-2.5-flash-preview)）已于 **7 月 15 日**被 Google 弃用。
   - 推荐的替代方案是 [google/gemini-2.5-flash](https://openrouter.ai/google/gemini-2.5-flash)，但由于价格变动，OpenRouter 不会自动路由流量，用户需要更新其代码；此前，*Flash Preview 比 Flash 便宜得多，且实际上是一个更好的模型*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1393638527216128215)** (5 messages): 

> `Mathcheap, Y-Router, Personality.gg, 多 AI 自动化研究机器人` 


- ****Mathcheap** 作为 Mathpix Snip 的免费替代品亮相**：[Mathcheap](https://mathcheap.xyz/) 作为一个由 **AI 驱动**的、免费的 **Mathpix Snip** 替代方案出现。
- ****Y-Router** 简化了 Claude Code 与 OpenRouter 的集成**：**Y-Router** 现已在 [GitHub](https://github.com/luohy15/y-router) 上可用，它作为一个简单的代理，使 **Claude Code** 能够与 **OpenRouter** 配合工作。
- ****Personality.gg** 提供免费的角色扮演体验**：**Personality.gg** ([Discord](https://discord.personality.gg)) 是一个免费的角色扮演网站和应用，是 **Character.ai** 和 **Janitorai.com** 的替代方案，由 **OpenRouter** 提供支持。
- ****多 AI 自动化研究机器人**发布，用于深度项目分析**：[多 AI 自动化研究机器人](https://github.com/NA-DEGEN-GIRL/openRouter-reasearch-bot) 使用 **OpenRouter API** 自动化深入的项目研究，编排多个 LLM 并行执行、交叉验证并将信息综合为结构化报告，全部通过简单的文本文件进行管理，具有高度的可定制性。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1393310856074231849)** (833 条消息🔥🔥🔥): 

> `Text Completion, OpenRouter 信用系统, 聊天室 GUI, Svelte vs React 聊天性能, Rate Limits` 


- **Text Completion 服务返回错误**：用户报告称一些提供商在 Text Completion 请求中返回错误，一位用户指出 OpenRouter 可能将 Text Completion 请求作为 Chat Completion 请求发送，另一位用户提供的代码示例显示了 *missing field `created` error*。
   - 一位用户询问了 Text Completion 的状态，另一位用户链接了一篇与 OpenAI 相关的[新闻文章](https://www.theverge.com/openai/705999/google-windsurf-ceo-openai)。
- **付费额度下的免费模型使用**：用户讨论了每天 1000 次免费模型请求的限制，一位用户确认这需要**至少 10 美元的一次性充值**。
   - 一位用户表示 *如果你购买了至少 10 美元，你将获得 1000 次免费模型请求*，但质疑这是否是永久性的，另一位用户确认这是永久性的。
- **聊天室 GUI 已更新**：用户报告 *新 GUI 无法为新房间保留默认模型偏好*，其他 GUI 问题正在解决中。
   - 一位用户报告了 *性能变慢*，而另一位用户注意到 *推理（reasoning）开关消失了*。 
- **Next.js 与性能：React vs Svelte**：用户讨论了使用 React 与 Svelte 构建的聊天应用的性能，指出由于 React 的不可变模型（immutability model），**基于 Svelte 的聊天应用**性能往往更好。
   - 一位用户认为 *用 React 构建的一切都非常沉重、臃肿且运行糟糕*。
- **Rate Limits**：成员们正在询问 **Rate Limits** 以及 Chutes 的 Rate Limits 如何影响 OpenRouter；以及是否有任何方法可以确定 Rate Limits。
   - 几位用户表示，他们认为 *Chutes 目前每天大约有 200 次免费额度（不是 OpenRouter 的 Rate Limit，而是 Chutes 针对每个用户的免费限制）*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1393333561217650798)** (12 条消息🔥): 

> `Switchpoint Router, 默认模型设置, Auto Router 功能` 


- **Switchpoint Router 固定定价引发疑问**：一位用户对 [Switchpoint Router](https://github.com/switchpoint/router) 的固定定价提出质疑，并对其默认选择表示担忧，因为他们更喜欢在聊天室中使用自己预选的默认模型。
   - 该用户进一步批评了缺乏自定义功能和高昂的成本，认为与可自定义的路由解决方案相比，这可能会限制其采用。
- **默认模型设置故障**：用户报告账户偏好中的默认模型设置被忽略，聊天默认使用 **Switchpoint Router** 而不是他们指定的模型，导致每次都需要手动选择。
   - 一位用户说：*“现在它只是默认使用 Switchpoint 并忽略了我设置的默认模型，我每次都必须手动选择我的模型。”*
- **Auto Router 混淆已澄清**：已澄清清除默认模型设置会恢复为 **Auto router**，而不是 Switchpoint，但发现了一个 Bug，即默认模型设置在聊天室中无法正常工作。
   - 分享了一张截图（[链接在此](https://cdn.discordapp.com/attachments/1393361519563243753/1394056526641238066/Screenshot_2025-07-13_at_4.42.35_PM.png?ex=6876bd50&is=68756bd0&hm=3fe863934ffdb5df57dea1473208dcd81ebe2d36ec3dc52a25f5faca31714c67&)）来说明这一点，并承诺将调查该 Bug。


  

---

### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1393369544021639278)** (89 messages🔥🔥): 

> `OpenRouter 定价、前端 UI 讨论、Gemini Embedding、快速 LLM` 


- **OpenRouter 的固定输出定价披露**：成员们澄清了 **OpenRouter** 使用 [固定输出定价](https://openrouter.ai/docs)，这意味着无论使用哪种底层模型，成本都是相同的。
   - 一些用户表示失望，希望路由能带来节省，而另一些用户则关注潜在的 **latency**（延迟）优势。
- **UI 抱怨引起关注**：用户对 **OpenRouter** 前端 **UI** 提出了批评，特别是推理块缺乏区分度、居中的聊天布局以及过小的聊天机器人输入框。
   - 一位成员还指出，*从一个房间切换到另一个房间时，Auto Router 会覆盖该房间之前保存的模型*，并且 **copy-pasting**（复制粘贴）无法正常工作。
- **Gemini Embedding 进入 GA 阶段**：提到 **Gemini Embedding** 正在从实验阶段转向 [GA (General Availability)](https://ai.google.dev/models/gemini)。
   - 虽然一些成员报告了良好的效果，但也有人对速率限制、定价竞争力以及闭源模型带来的客户锁定风险表示担忧。
- **快速 LLM 讨论**：成员们讨论了快速 **LLM** 的选项，比较了 [Llama 3.3 70B](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)、[Llama 4 Maverick](https://huggingface.co/TheBloke/Llama-4-Maverick-70B-GGUF) 和 [Groq 的大型 Qwen3](https://groq.com/)。
   - 建议包括 **Cerebras** 模型和 **Grok 3 mini**，但一位成员报告称 *Grok 3 mini 偏慢 (TTFT)*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1393338525008658463)** (255 messages🔥🔥): 

> `多模态支持、LM Studio SDK、Prompt Caching、工具调用与 MCP、Kimi K2 的硬件需求` 


- **LM Studio 增强多模态支持**：虽然最新的 **LM Studio** 版本描述暗示了多模态支持，但目前它仅处理 **text and vision (describe this image)** 模型，缺乏图像生成能力。
   - 新版本的描述引起了混淆，但 *目前尚不支持图像输出*。
- **深入研究 LM Studio SDK 进行定制**：成员们讨论了利用 **LM Studio SDK** 实现类似 `llamaswap` 的功能，用于手动内存管理。
   - 有人指出，虽然 **OpenAI API** 没有公开加载/卸载函数，但可以使用 **SDK** 来编写带有手动加载和卸载的交换行为。
- **解码 LM Studio 中的 Prompt Caching**：**LM Studio** 会自动缓存最后一条消息以加快生成速度，但不会缓存整个请求/响应对。
   - 它支持线性 **prompt caching**（在最后一次更改前 token 保持不变），但尚未启用动态缓存。
- **在 LM Studio 的 API 中使用工具调用和 MCP**：一位用户询问在将 **LM Studio** 作为带有 HTTP 请求的服务器使用时，如何在其中使用 **MCP** (Model Control Program)，但 **API** 保持不变，需要客户端定义自己的工具。
   - 这意味着 **OpenAI compatible API** 本身并不支持在 **LM Studio** 内部进行工具选择。
- **评估 Kimi K2 的硬件需求**：讨论围绕运行 **Kimi K2** 模型所需的巨大 **VRAM 需求**展开，估计达到 **2400 GB**。
   - 提到 **4x H200 GPU** 可能足以运行该模型的 `Q4_K_M` 量化版本，这引发了关于此类硬件负担能力的幽默评论（*每块芯片约 30,000 美元*）。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1393309193800388769)** (63 条消息🔥🔥): 

> `Nvidia DGX, 5090 价格, 运行电费, 1T 参数模型, EXAONE 4` 


- **Nvidia DGX 表现和 Ryzen 395 一样差？**: 成员们讨论了 Nvidia DGX 的性能，有人认为在运行大型模型时，其 tok/sec 表现与 [Ryzen 395 平台](https://www.amd.com/en/products/cpu/amd-ryzen-9-3950x) 相似。
   - 一位用户估计 DGX 可能会快 *25%*，因为 395 缺乏 ROCm 支持。
- **5090 太贵了？**: 成员们讨论了购买 **5090** 的问题，考虑到电费成本，有人质疑其相对于 Gemini 2.5 等云端 API 的性价比。
   - 一位成员指出，*即使免费得到 5090，运行它的电费也比 API 成本高*。
- **5090 数据**: 一位用户报告了 **5090** 在 Windows 上运行 **LM Studio** 的性能：Q8_K 约为 45 tok/sec，Q6-XL 约为 55 tok/sec，Q4-XL 约为 65 tok/sec。他注意到了功耗差异，并建议进行降压（undervolting）和显存超频（memory OC）。
   - 另一位用户认为 *Gemini 的 PP（Prompt Processing）速度更令人印象深刻*，在给它一个 500k+ token 的代码库后，它在 10 秒内就完成了读取。
- **1T 参数模型需要 TB 级的 RAM？**: 一位成员询问了运行仅有 **32B 激活参数** 的新 **1T 参数模型** 的硬件要求，思考是否可以在 RAM 超过 1T 的 CPU 上运行。
   - 另一位用户建议使用拥有 640GB+ RAM 的 Epyc 12 通道内存系统或六块 RTX 6000 Pro 来处理。
- **Kimi 下载失败？**: 一位用户询问了 [Kimi-K2-Instruct-GGUF 模型](https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF)，但另一位成员指出它 *无法在 LM Studio 中运行*。
   - 一位成员感叹下载需要 50 小时，另一位用户为此制作了一个[树懒](https://tenor.com/view/funny-very-sloth-slow-gif-15401812)的 GIF。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1393318565809229926)** (1 条消息): 

> `Gemma 3n, SmolLM3, 高效多模态数据流水线, responses.js, EoMT 图像分割模型` 


- **Gemma 3n 开源**: 根据 [Hugging Face 博客文章](https://huggingface.co/blog/gemma3n)，**Gemma 3n** 模型现在已完全进入开源生态系统。
- **SmolLM3：小巧但强大的多语言推理模型**: **SmolLM3** 是一款小巧、多语言、长上下文的推理模型，现已发布；详见 [Hugging Face 博客](https://huggingface.co/blog/smollm3)。
   - 该发布也在 [X 上宣布](https://x.com/LoubnaBenAllal1/status/1942614508549333211)。
- **使用由 HF Inference Providers 驱动的 Responses API 进行构建**: 一个新的 OSS 项目 **responses.js** 已经推出，用于构建由 **HF Inference Providers** 驱动的 Responses API，正如 [X 上所宣布](https://x.com/Wauplin/status/1941059510665458174)的。
- **Sentence Transformers v5 引入稀疏编码器训练**: 根据 [Hugging Face 博客文章](https://huggingface.co/blog/train-sparse-encoder)，现在可以使用 **Sentence Transformers v5** 训练和微调稀疏嵌入模型。
- **Hugging Face 将构建转录应用**: 根据 [HuggingFace 文档](https://huggingface.co/docs/inference-providers/guides/building-first-app)，关于 **Inference Providers** 的新教程已上线，用于构建转录应用程序。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1393316113890410556)** (233 条消息🔥🔥): 

> `Fine-tuning multimodal models for electronics, AI moderator bot with image support, Quantization and running LLMs on limited hardware, Hugging Face Courses, SillyTavern and AI model integration on Android` 


- **电子专家寻求多模态微调建议**：一位新成员正在寻求关于电子领域多模态模型微调的建议，特别是针对从电子元件的扫描图像中检测电路连接和原理图，目标是协助用户使用万用表或类似工具诊断故障。
   - 该成员正在权衡使用 **small model** 与 **large model** 的方案，以及是否对其进行 **quantize**（量化），并请求关于如何开展该项目的指导。
- **Discord 推出基于 LLM 且支持 NSFW 图像检测的 AI 审核机器人**：一位成员正在使用 LLM 技术为 Discord 开发一款 **AI moderator bot**，旨在实现 Agent 行为和 **NSFW 图像检测**。
   - 他们在 **4060 GPU** 上运行 Gemma 3 4b 遇到速度缓慢的问题后，正寻求添加图像支持和提高性能的建议，并提供了代码片段供审查。
- **利用 GGUF 和量化释放小模型性能**：成员们讨论了在有限硬件上运行 LLM 的问题，其中一位成员推荐使用 **ollama** 运行量化版本的 Gemma，以适配 **4060 GPU**。
   - 建议使用 *bnb* 进行量化，或使用类似 [这个](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 的配置。
- **通过 Triton 教程驯服 LLM**：一位全栈开发人员询问 Hugging Face 的开源课程是否足以创建用于图像处理和输出格式化等特定任务的自定义模型。
   - 一位成员建议从 [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html) 开始，然后学习 [Transformers tutorial](https://github.com/NielsRogge/Transformers-Tutorials) 以入门模型架构。
- **Kobold 云端或可破解 Android AI 难题**：尽管只有 4GB RAM，一位用户仍寻求在 Android 设备上通过 SillyTavern 和 Colab 运行像 **mythomax** 这样无审查 AI 模型的指导。
   - 讨论强调了在低资源设备上运行此类模型的局限性，建议包括使用云服务或在 [RunPod](https://koboldai.org/runpodcpp/) 等付费云平台上部署 **Kobold CPP**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1393500812906598503)** (6 条消息): 

> `Deepseek 8-bit training, 4-bit training` 


- **Deepseek 完全采用 8-bit 精度训练？**：一位成员询问 **Deepseek** 是否完全在 **8-bit precision** 下训练，另一位成员确认确实如此。
   - 讨论探讨了未来进行 **4-bit training** 的可能性。
- **关于 4-bit 训练何时可行的推测**：在确认 **Deepseek** 采用 **8-bit precision** 训练后，一位成员推测了何时能实现完全的 **4-bit training**。
   - 然而，有人指出由于潜在的性能损失，**4-bit training** 可能并不值得。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1394335483433058367)** (2 条消息): 

> `Dynamic Structure Adjustments` 


- **关于动态结构调整的辩论**：成员们讨论了实时调整结构的灵活性。
   - 一位成员确认了根据需要进行更改的能力，另一位成员表示赞赏。
- **关于灵活调整的确认**：一位成员明确确认结构可以“随用随调，按需调整”。
   - 另一位成员对这一确认表示感谢，表明这种灵活性符合他们的预期。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1393407065606852800)** (20 messages🔥): 

> `许可证合规工具，BorgLLM 开源，轻量级计算机视觉模型，偏好数据 Agent Arena，Stable Audio 模型实验` 


- ****ScanCodeMCP** 自动化许可证分析**：一位成员构建了 [scancodeMCP](https://github.com/bivex/scancodeMCP)，这是一个**许可证合规工具**，通过 MCP 与 Cursor 集成，通过阅读许可证的细则执行律师级别的许可证分析。
   - 它提供逐条条款的分解，对比 MIT 与 Apache 2.0 等许可证，并为文件提供**风险评估**，旨在消除许可证焦虑。
- ****BorgLLM** 正式开源**：**BorgLLM** 是一个零配置的 Langchain 客户端，现已在 MIT 许可证下完全开源，允许轻松集成各种供应商。
   - 它可以与任何供应商（OpenAI, Anthropic, Mistral, Groq）配合使用，代码库可以在[这里](https://github.com/omarkamali/borgllm)找到。
- **轻量级计算机视觉模型达到 **85.5%** 准确率**：一位成员开发了一个轻量级计算机视觉模型，在 Microsoft 数据集的猫狗分类任务中，仅用 **8.25k** 参数就达到了 **85.5%** 的准确率。
   - 目标是在参数量极小的情况下，在未见过的测试数据上突破 **90%** 的准确率，目前代码仍在开发中（WIP）。
- **通过 **Agent Arena** 免费访问 Grok4**：成员可以通过在 [Agent Arena](https://obl.dev/Example) 中提供用于训练的偏好数据（点赞/点踩），免费访问 **Grok4**、**o3** 和 **Deep Research**。
   - 收集的数据将用于训练，该服务已向成员开放。
- ****Stable Audio** 生成鼓点和乐器循环**：一位成员尝试了 stable-audio-open-small 模型，通过负向提示词（negative prompting）生成**仅限鼓点**和**仅限乐器**的输出，并将它们组合成自定义循环，并分享了[带有 API 的 Docker 容器](https://github.com/betweentwomidnights/stable-audio-api)链接。
   - 他们还分享了一条 [Tweet](https://x.com/thepatch_kev/status/1944203286678319493) 和一个 [zerogpu space](https://huggingface.co/spaces/thepatch/stable-melody)。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1393909206637088848)** (2 messages): 

> `HuggingFace Ultrascale Playbook，全规模训练资源，OpenAI 职位要求` 


- **HuggingFace 的 Ultrascale Playbook：首选资源**：一位成员认为 [HuggingFace 的 ultrascale playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 是进行**全规模训练（full-scale training）**的最佳资源。
   - 该手册对涉及的流程进行了出色的总结。
- **Ultrascale Playbook 是大型 AI 职位的基石**：一位成员建议，掌握 **Ultrascale Playbook** 的知识是获得 **OpenAI** 等公司职位的基本要求。
   - 他们指出，如果没有多 GPU 的访问权限，其实用性可能会受到限制。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

dlp1843: opencv.org 的落地页之于 OpenCV，是否就像 bitcoin.com 之于比特币？
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1393307009360859319)** (28 messages🔥): 

> `HF Secrets 泄露，图像/音频工具，Agent 课程视频环节？，助手节点单字回答，MCP Server 设置帮助` 


- **HF 面临安全泄露担忧**：一位用户怀疑与 **HF 课程**相关的泄露可能源自 **Hugging Face** 本身，并引用了 OpenAI 开发者论坛中提到的过去关于 **HF secrets** 的问题。
- **Agent 课程：只是阅读吗？**：一位用户询问 Agent 课程是否全是阅读材料，是否有可用的视频环节。
- **助手节点变得严格**：一位用户请求一个能让**助手节点**仅给出**单字回答**的提示词，因为他正苦于无法强制执行这一约束。
- **MCP Inspector Server 设置解决**：一位用户在按照[课程视频](https://huggingface.co/learn/mcp-course/en/unit1/sdk?server-implementation=python)设置 **MCP server inspector** 时需要帮助，他遇到了连接问题。
- **Qwen 模型停止工作？**：一位用户报告 **Qwen/Qwen2.5-Coder-32B-Instruct** 模型停止了工作。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1393319674829344839)** (142 messages🔥🔥): 

> `Grok-4 推理与工具，Deep-Hermes 推理器选项，AI 模型自我博弈 (Self-play)，Kimi K2，OAI 开源模型延迟` 


- **Grok-4 在推理与工具使用方面表现出色**：成员们讨论了 **Grok-4** 在推理过程中始终使用推理能力，并在推理期间调用工具。
   - XAI 的直播演示表明，**强化学习 (Reinforcement Learning)** 是开发 **Grok-4** 的关键，强调了对可检查且定义明确的解决方案的需求。
- **Hermes 4 将包含用户可控的推理器**：**Hermes 4** 正在采用混合方法进行开发，其特点是拥有一个类似于 Deephermes 的*用户可控推理器*。
   - 一位成员建议探索一些选项，例如默认开启推理，但允许通过预填充空的 think 标签来禁用它。
- **AI 模型在训练中探索自我博弈 (Self-play)**：成员们讨论了 **自我博弈 (Self-play)** 如何让 **Deep-Hermes 推理** 的用法更加细致。
   - 他们指出了 [textarena](https://link.to/textarena) 和一篇 [自我博弈编程论文 (self-play coding paper)](https://link.to/selfplay-coding-paper) 等示例，这些示例在实现自我博弈方面取得了成功。
- **Kimi K2 进入开源领域**：在 **DeepSeek R!**、**Qwen** 以及现在的 **Kimi K1** 之后，成员们表示开源模型的能力正变得令人惊叹。
   - 有人指出，只有极少数人需要比供应商免费提供的模型更强大的能力，现在企业和个人可以免费访问接近前沿模型 (Frontier models) 的模型，以构建应用程序并随心所欲地使用。
- **OAI 开源模型面临延迟**：成员们报告称 **OpenAI** 的开源模型将推迟不确定的时间，这可能是由于 **Kimi** 模型的表现所致。
   - 还有人建议，发布的模型将拥有非常严格的许可证，并且不会提供基座模型 (Base model)。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1393350482692542626)** (11 messages🔥): 

> `Docker 化，提示词工程 (Prompt engineering)，埃及神灵，AI 治理文章，SFT 与 GRPO` 


- **成员正在进行 Docker 化**：一位成员正致力于将模型封装进 **Docker** 容器中。
   - 他们尚未反馈任何成功的进展。
- **为你的提示词库构建人格奥林匹斯山**：一位成员正在构建一个**提示词库 (Prompt library)**，并希望给他们的 Agent 起一组神灵的名字，并加入一点个性。
   - 他们担心在提示词中加入个性可能会对需要小型、精简指令遵循模型的 Agent 工作流 (Agentic workflows) 产生反作用，并征求社区建议。
- **深入研究 Agent 工作流**：一位成员使用 **Thot** 来帮助他们吸收文章和博客内容，并对其进行提问。
   - 他们目前正在使用 **Haystack pipeline** 和 **smolagents**，以及来自 **llama-index** 的数据加载器，同时明确避开使用 Langchain。
- **寻求关于 SFT 和 GRPO 数据量的指导**：一位成员正在对基座模型进行 **SFT** 随后的 **GRPO** 微调，并正在寻找讨论这两个过程数据量比例影响的论文。
   - 他们认为 **RL** 与 **SFT** 的比例通常应至少在 **1:2** 左右。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1393619696695115907)** (11 messages🔥): 

> `递归学习系统研究，递归符号智能，每个模型根部的本体论 (Ontology)，Psyche 作为 MCP 组件` 


- **对递归学习系统的研究兴趣被激发**：一位成员询问了关于**递归学习系统研究 (Recursive learning systems research)** 的普遍兴趣，特别是**递归自我改进 (Recursive self-improvement)** 和 **McCarthy (1960) 递归符号智能**。
   - 他们认为，大家都在*试图倒转时钟*，看看由于搁置了*认知革命的这第三个分支*而错过了什么。
- **递归符号智能缺失了本体论**：一位成员指出，关于**递归符号智能**，*每个模型的根部都没有本体论 (Ontology)*。
   - 他们表达了分享想法和研究的兴趣，但想确保这是一个合适的场合。
- **建议将 Psyche 作为 MCP 组件**：一位成员提到他们一直想近距离研究 **Psyche** 作为 **MCP 组件**，并分享了一个相关推文链接。
   - 该 [推文](https://x.com/zhaoran_wang/status/1944116318858363364?s=46) 指向了 Arxiv 上的一篇论文：[https://arxiv.org/abs/2507.08794](https://arxiv.org/abs/2507.08794)。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1394138509391954033)** (4 条消息): 

> `AI Disruption, MedGemma, Expert-Level Fine-Tuning` 


- **AI Disruption 即将来临**：[ts2.tech](https://ts2.tech/en/ai-in-july-2025-disruption-opportunity-and-uncertainty-across-the-globe-updated-2025-july-4th-0000-cet/) 的一篇新博客文章指出，到 2025 年 7 月，AI 将在全球范围内引发**变革、机遇与不确定性**。
   - 作者认为 AI 模型将变得**更加先进且易于获取**，有可能改变各行各业和日常生活，潜在挑战包括失业问题和伦理考量。
- **MedGemma 助力健康领域 AI 开发**：Google Research 发布了 [MedGemma](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)，这是他们用于**健康领域 AI 开发**的功能最强大的**开源模型**。
   - 根据博客文章，MedGemma 模型旨在协助处理一系列医疗相关任务，确保医疗领域的**负责任 AI 开发**。
- **Microsoft 将人类反馈嵌入微调过程**：Microsoft 的 WorkLab 计划正在将**人类反馈嵌入到专家级 Fine-Tuning 过程中**，从而提升特定领域应用的性能。
   - 这种 Fine-Tuning 方法利用人类见解的细微差别来提高 AI 模型的精确度和相关性，从而在专业任务中实现**更高的准确性和有效性**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1393619696695115907)** (11 条消息🔥): 

> `Recursive Learning Systems, Symbolic Intelligence, Psyche MCP Component, Ontology in Models` 


- **递归学习研究兴趣被激发**：一名成员询问了大家对 **Recursive Learning Systems 研究**的普遍兴趣，特别引用了 **McCarthy (1960) 的递归符号智能** ([http://jmc.stanford.edu/articles/recursive.html](http://jmc.stanford.edu/articles/recursive.html))。
   - 他们观察到，目前的努力似乎旨在重新审视被搁置的*认知革命的第三个分支*。
- **模型中缺失 Ontology 的问题**：一名成员指出大多数模型的根基中都缺乏 **Ontology**，并暗示如果有足够的兴趣，将分享相关的潜在研究。
   - 他们表达了探索将 **Psyche** 作为 **MCP** 组件的兴趣，并链接了一条相关的推文 ([https://x.com/zhaoran_wang/status/1944116318858363364?s=46](https://x.com/zhaoran_wang/status/1944116318858363364?s=46)) 以及 [https://arxiv.org/abs/2507.08794](https://arxiv.org/abs/2507.08794) 的链接。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1393316623292825783)** (35 条消息🔥): 

> `PMPP 5th edition and ML updates, FP8 training, Luminal talk, vast.ai GPU pricing scraper, Programming models for ML applications` 


- **PMPP 第 5 版将涵盖 LLM 和 Flash Attention**：即将出版的第 5 版 *Parallel Programming for Multi-core and Many-core* (**PMPP**) 将包含对 **LLM** 和 **Flash Attention** 的深入讲解。
   - 一名成员表示，这提供了他们见过的*可能是最好的解释*，同时还涵盖了 **Tensor Cores** 和 **多 GPU 编程**。
- **DeepSeek 主要使用 FP8 GEMM 操作进行训练**：根据讨论，[DeepSeek](https://arxiv.org/pdf/2412.19437) 主要使用 **FP8 GEMM** 操作进行训练，但在 **Attention** 或 **MoE Router** 等部分使用了更高精度。
   - 累加是在 **FP32** 中进行的，这在 **MoE 模型**中尤其适用，因为在全过程使用 FP8 的密集模型中会看到更多的不稳定性。
- **Luminal Primops 基础集选择原理揭晓**：在与来自 Luminal 的 Joe 交流时，他们透露选择基础操作集的方法是*通过观察模型并找出哪些操作是可简化的，哪些是不可简化的*。
   - Joe 提到其他 Primops 可能会更好或更差，但*我认为这不会产生太大的区别*。
- **Vast.ai GPU 价格爬虫发布**：一名成员创建了一个 [vast.ai GPU 价格爬虫](https://github.com/tornikeo/vastai-gpu-pricing-scraper)，每天收集所有条目（无论是否可用）。
   - 目标是构建一个工具，用于跟踪不同云和新云服务商随时间变化的 GPU 价格，该工具将开源。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1393345674363277332)** (16 条消息🔥): 

> `Triton Kernel Padding, AOT Triton Updates, Gluon Tile Scheduling, Linear Attention Kernel Optimization, Matmul Library Matrix Handling` 


- **Kernel Padding 性能悖论！**：一位用户正在优化 **Triton kernel**，并寻求关于如何处理非 **128** 倍数的输入序列长度且不产生显著内存开销的建议，但输入 Tensor 的主要维度应该更加对齐。
   - 有建议指出，虽然 in-kernel padding 看起来很直观，但如果 Tensor 的 strides 不是 **1024 bits** 的倍数（cacheline-aligned），它可能没有帮助，转置 Tensor 可能是更好的方法。
- **AOT Triton 状态更新！**：一位用户询问了关于 **Ahead-Of-Time (AOT) Triton** 的更新情况，并引用了 [GitHub 上的一个 issue](https://github.com/triton-lang/triton/issues/935)。
- **Linear Attention 的内存对齐思考！**：在优化 **attention kernel** 的线性近似时，如果输入序列长度不是 128 的倍数，某些 strides 也不会是 128 的倍数。
   - 用户还提到 **Flash Attention 3** 已经实现了 in-kernel padding，在典型序列长度（128 的倍数）下具有类似的性能。
- **Tensor Core 技巧！**：在处理大小为 **d x N** 乘以 **N x d** 的矩阵（其中 d = 128，N = 80001，数据类型为 bfloat16）时，对 **N** 进行 padding 或将矩阵 **A** 设为列优先（column-major）是潜在的解决方案。
   - 有人提到 **matmul 库** 可以很好地处理 **BF16** 的转置操作，但这对于 **H100 上的 FP8** 或 **B200 上的 FP4** 可能会成为问题。
- **利用 Triton Streams 隐藏延迟！**：一位用户询问 **Triton** 是否可以同时运行 reduction（使用 cuda core）和 matmul（使用 tensor core），以便在计算 `segment += tl.sum(shared_k_i * shared_k_j, axis=1)` 和 `dot_product += tl.dot(shared_k_i * shared_k_j, v_vals, allow_tf32=True)` 时隐藏延迟。
   - 有建议称，为了获得更多控制权且在没有数据依赖的情况下，可以将工作拆分为两个 Triton kernels，并将它们启动在不同的 **CUDA streams** 上以实现重叠执行（overlap execution）。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1393996209751986247)** (1 条消息): 

> `Deadlock Issue Debugging, cudaMemcpyAsync issues, cudaHostFunc issues, NCCL issue #1509` 


- **调试 Computation、cudaMemcpyAsync 和 cudaHostFunc 之间的死锁**：一位用户正在调试大规模训练任务中 **computation**、**cudaMemcpyAsync** 和 **cudaHostFunc** 之间的死锁问题。
   - 用户注意到这与 **NCCL issue #1509** 类似，并正致力于简化复现案例，同时排除 Megatron → PyTorch → NCCL → CUDA 路径中的变量。
- **复现死锁问题**：用户的实现在大规模训练任务中遇到了类似的问题。
   - 最小复现案例仍然很杂乱且冗长，仅能复现几次 1F1B 迭代。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1393314454267236482)** (6 条消息): 

> `gradient computation, xai method, CPU memory usage, Torch, activation memory` 


- **多 GPU 反向传播的 Torch 技巧**：一位成员询问如何使用 Torch 在多个 GPU 上拆分 backprop，在为某种 xai 方法计算梯度时遇到了 **100 GB** 的 CPU 内存占用。
   - 一位成员建议使用 activation checkpointing/offloading，并链接了一篇关于[理解 GPU 内存的 PyTorch 博客文章](https://pytorch.org/blog/understanding-gpu-memory-1/)。
- **从 DDP 到模型并行**：一位成员建议使用类似 [DistributedDataParallel (DDP)](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) 的方案，用户回复说他们正在寻找类似模型并行（model parallelism）的方法，因为他们一次只使用一个样本。
   - 另一位成员提到通过 zero/fsdp 在多个 GPU 上分片梯度（sharding gradients），并建议使用重计算（checkpointing）以避免存储所有 activations。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1393668603710144552)** (1 条消息): 

> `Luminal, Deep Learning Compiler, Joe Fioti` 


- **Joe Fioti 将讨论 Luminal**：Joe Fioti 将讨论 **Luminal**，这是一个基于搜索的深度学习编译器。
- **使用 Luminal 进行基于搜索的深度学习编译**：Joe Fioti 将介绍 **Luminal**，这是一款基于搜索的深度学习编译器。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 条消息): 

piotr.mazurek: https://github.com/tugot17/pmpp
  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1393453818812825641)** (4 messages): 

> `PyTorch TorchAO, ICML 2025, CodeML workshop, TorchAO Poster` 


- **TorchAO 团队前往 ICML 2025**：**PyTorch TorchAO** 团队将于 7 月 18 日在 **ICML 2025 CodeML workshop** 展示海报。
   - 他们的海报《TorchAO: PyTorch-Native Training-to-Serving Model Optimization》探讨了如何使用 PyTorch-native 工具弥合模型训练与高效推理服务（serving）之间的差距；点击此处查看 [session 详情](https://icml.cc/virtual/2025/workshop/39970#wse-detail-48182)。
- **KernelBot 将在 ICML 与 TorchAO 交流**：一位成员分享说他们将参加与 **KernelBot** 相同的 ICML workshop，并希望能进行交流。
   - 在 **ICML** 结识新朋友非常棒。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1393518751961579551)** (2 messages): 

> `GPU pronouns, TPU pronouns, CUDA pronouns, ROCm pronouns` 


- **代词（Pronouns）现在包含 GPU 和 TPU**：Discord 用户分享了一张图片，显示代词选择现在包含 **GPU/TPU** 选项。
   - 图像分析显示，最初倾向于将 **CUDA/ROCm** 作为首选代词，但目前暂定为 **GPU/TPU**。
- **CUDA 和 ROCm 差点入选代词**：Discord 用户讨论了增加 **GPU/TPU** 作为代词选项的事宜。
   - 最初人们更倾向于 **CUDA/ROCm**，但最终决定采用更通用的 **GPU/TPU** 代词，以获得更广泛的适用性。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1393356037456662690)** (2 messages): 

> `AI Conference San Francisco, ICML Meetup, KernelBot Paper Presentation` 


- **AI Conference 落地旧金山**：**9 月 17-18 日**在**旧金山**将举行一场 **AI conference** ([https://aiconference.com/](https://aiconference.com/))。
   - 一位成员询问是否有人会参加。
- **ICML：线下聚会的好机会**：一位成员提到他们将在 **ICML** 待上一周，并邀请大家进行线下聚会。
- **KernelBot 在 CodeML Workshop 首次亮相**：一位成员宣布他们的团队将于周四在 **CodeML workshop** 展示 **KernelBot 论文**。
   - 演示将在 **west meeting rooms 211-214** 举行。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1393397355201695744)** (4 messages): 

> `rocprofv3 profiling, AMD kernels, PyTorch profiling` 


- **为 PyTorch 函数调试 `rocprofv3`**：一位用户询问 `rocprofv3` 在进行 profiling 时是否默认排除了 PyTorch 函数，并寻求对 PyTorch 引用进行 profiling 的方法。
   - 一位成员建议在 tracing 之前使用正确的 ROCm 前缀启动脚本，另一位用户提出了一个带有特定 **`--att` 标志和缓冲区大小**的 `rocprofv3` 命令用于 profiling，并怀疑除了 **AMD 的内置 kernel**（用于 memcpy 和 blit 等）之外，没有其他内容被排除。
- **配置 `rocprofv3` 进行 activity tracing**：一位用户分享了他们用于 `rocprofv3` profiling 的命令：`rocprofv3 --att --att-buffer-size 1_000_000_000 --att-activity 10 -d dir -- program`。
   - 这些参数配置了 **attribute tracing、缓冲区大小、activity tracing** 以及输出目录，后面跟着要进行 profiling 的程序。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1394010184678113500)** (2 messages): 

> `TurboWarp Extension for Machine Learning, MTLReadWriteTextureTier2 and wgpu` 


- **TurboWarp 拥抱机器学习扩展**：一位成员正在探索为 [TurboWarp](https://turbowarp.org/) 编写机器学习扩展，可能会以 **Blockly** 形式封装 **convnetjs**。
   - 他们正在寻找有关现有项目或在该领域有经验的人士的信息。
- **MTLReadWriteTextureTier2 访问难题**：一位成员正在寻求关于将 **MTLReadWriteTextureTier2** 暴露给 **wgpu** 的指导。
   - 尽管启用了 **Texture_Adapter_Specific_Format_Features**，但他们仍无法访问 **read_write** 纹理的 **rgba8unorm**，而这在 **MetalAPI** 的 **Tier2** 中是受支持的。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1393349778288414761)** (12 messages🔥): 

> `Thunder Compute VSCode Extension, NVIDIA Tensor Core Evolution, QuACK Open Source Library, Backpropagation through RMSNorm and LayerNorm, AI Compute Hackathon in a German Castle` 


- ****Thunder Compute** VSCode Extension**: 对于那些不喜欢 SSH 配置但喜欢廉价 GPU 的人，可以尝试 **Thunder Compute** 的 [VSCode extension](https://www.thundercompute.com/docs/quickstart)。
   - 当一位用户表示这两者以及 VSCode 他都不喜欢时，一名成员回复说他们也有 **CLI** 工具。
- ****NVIDIA Tensor Core** 演进：从 Volta 到 Blackwell**: 根据 [semianalysis.com](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/) 的报道，一次演讲介绍了关键的 **Tensor Core** 架构，探讨了基本性能原理，并揭示了这些原理如何驱动架构演进。
- ****QuACK** 开源库性能超越 **PyTorch****: **QuACK** 是 Tri Dao 及其研究小组新发布的开源库，使用 **CuTeDSL** 编写。它利用 **CuTeDSL** 编写高效的 reduction 操作，可以以极快的速度编写 memory bound kernels！博客文章见 [veitner.bearblog.dev](https://veitner.bearblog.dev/outperform-compiled-pytorch-code-using-quack/)。
- **揭秘 **RMSNorm** 和 **LayerNorm** 的反向传播**: 一名成员分享了他们手动推导 **RMSNorm** 和 **LayerNorm** 反向传播的工作，并提供了博客文章链接：[veitner.bearblog.dev/backprop-through-rmsnorm/](https://veitner.bearblog.dev/backprop-through-rmsnorm/) 和 [veitner.bearblog.dev/backprob-through-layernorm/](https://veitner.bearblog.dev/backprob-through-layernorm/)。
- **城堡里的黑客松**: 一名成员正在一座 14 世纪的德国城堡举办一场小型黑客松，以探索 AI 计算的未来，主题包括：**GPU 优化、GPU 基础设施和能效芯片**，详情见 [castlecompute.com](http://castlecompute.com)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1393993467583266838)** (3 messages): 

> `nsight compute profiling, AutoTriton` 


- **Profiling 结果带来提升**: 一名成员建议，提供 **nsight compute profiling** 结果的访问权限可能会改善结果，并指出目前还没有人尝试过。
   - 他们计划本周进行实验，并强调这件事已经在他们脑海中盘旋数月了。
- **AutoTriton 深入 LLM 强化学习**: 一名成员分享了本月初发布的 [GitHub 仓库](https://github.com/AI9Stars/AutoTriton) —— **AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs**。
   - 该项目代表了该领域的相关工作。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1393338093859508274)** (7 messages): 

> `H100 First Place, A100 First Place, MI300 Personal Best` 


- **夺得 H100 桂冠！**: 一名成员以 **6.56 ms** 的提交时间获得了 `trimul` 排行榜 **H100** 组别的**第一名**。
- **达成 A100 顶尖成绩！**: 一名成员以 **11.5 ms** 的提交时间获得了 `trimul` 排行榜 **A100** 组别的**第一名**。
- **MI300 热潮！**: 一名成员以 **29.6 µs** 的提交时间在 `amd-identity` 排行榜 **MI300** 组别中创造了**个人最佳成绩**。
- **MI300 铜牌！**: 一名成员以 **5.76 µs** 的提交时间获得了 `amd-identity` 排行榜 **MI300** 组别的**第三名**。
- **MI300 新纪录！**: 一名成员以 **934 µs** 的提交时间在 `amd-fp8-mm` 排行榜 **MI300** 组别中创造了**个人最佳成绩**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1393537227795070976)** (15 messages🔥): 

> `Training Repo, Vision Transformers, TAS Data, Main Branch Broken` 


- ****Alpha Factorio** 训练仓库出现**: 一名成员询问是否有独立的训练仓库，并分享了 [alpha-factorio](https://github.com/kiankyars/alpha-factorio) 的链接作为潜在资源。
   - 同一位用户提到使用了来自 OpenAI five 的 **vision transformers**。
- **发现 **TAS 数据**，但已过时**: 一名成员询问来自 [alpha-factorio/tasks.lua.txt](https://raw.githubusercontent.com/kiankyars/alpha-factorio/refs/heads/main/tasks.lua.txt) 的数据是否为原始 **TAS 数据**。
   - 另一名成员确认那是 **TAS 数据**，但指出它大约有 **5 年历史**，因此非常过时。
- ****Main 分支**遭遇 Headless 挂起问题**: 据报道，如果在客户端连接时运行测试，**MAIN 分支**会损坏。
   - 目前似乎只有 **headless 模式**可以工作，另一名成员无法在 Windows 上运行脚本，不过此问题后来已被修复。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1393336893835579392)** (49 条消息🔥): 

> `Cute Tensors, CuteDSL 中的广播 (Broadcasting), Cutlass Kernel, cuTile, CUDA` 


- **在 Cute 中分配局部张量 (local tensor)**：要分配一个可以进行累加的局部张量，可以根据成员建议使用 `cute.full`（值语义）或 `cute.empty`；`cute.make_fragment` 可用于类似 buffer 的对象。
   - 值语义是 `tensorssa`，类似 buffer 的是 `cute.tensor`。
- **通过布局操作 (Layout Manipulation) 实现 Cute Tensors 的广播 (Broadcast)**：为了实现 Cute 张量的广播，一位成员建议使用 `cute.append` 来修改布局，如该 [代码片段](https://cdn.discordapp.com/attachments/1393385974071033946/1393386275595223060/image.png?ex=68764757&is=6874f5d7&hm=667cf01ce5e80272fe5464146637e1bb1a3e901591a8f6d487fe0e5c79ec83db) 所示。
   - 寻求将 `[m, k]` 和 `[k, n]` 张量广播为 `[m, k, n]` 然后沿 `k` 轴求和的成员发现，目前的实现需要手动切片和布局调整，但可以通过实用函数来简化。
- **新手如何上手 Cutlass**：一位资深成员建议在深入研究 Cutlass 之前，先从基础的 CUDA MMA kernel 开始，以理解 Cutlass 解决的问题；并建议在 Markdown 文档的内联代码示例中搜索变量名，以找到用于谓词化 (predication) 的 Cutlass 实现，参考 [Cute 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/) 和 [Cute 教程](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/)。
   - 他们分享了 [代码的 gist](https://gist.github.com/capybara-club/2e1db50580e832795d7ac4a8a7859f78) 和 [另一个 Cute 实现](https://github.com/Kernel-Machines/kermac/blob/master/include/p_norm_gradient.cuh)，并指出在 kernel 中融合外积 (outer product) 的性能显著优于 PyTorch 的 `einsum`。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1393322315097440256)** (85 条消息🔥🔥): 

> `GPU 需求, ICML 2025, 因果系统 (Causal Systems), 低 GPU 功耗 AI, 水/光基 GPU` 


- **Llama 2 发布后的 GPU 需求泡沫**：一位成员分享了一篇 [Latent Space 文章](https://www.latent.space/p/gpu-bubble)，指出在 **2023 年 6 月** 左右（与 **Llama-2** 发布时间重合），出现了 **GPU 供应紧缺**和租用 GPU 的过度需求。
- **ICML 2025 Discord 和周边活动启动**：成员们分享了 **ICML 2025** 的 [Discord 邀请链接](https://discord.gg/FaKV6ydy)，周边活动的 [Lu.ma 链接](https://lu.ma/1cyp6rq8) 和 [Partiful 链接](https://partiful.com/e/AIOKRPJyMsXQPVjXqjp7)，以及 **ICML** AI Safety 讨论的 [WhatsApp 群组邀请](https://chat.whatsapp.com/HspVnNxMWgx9ptXZBUutuW)。
- **成员辩论因果推理 (Causal Inference) 的应用**：一位成员询问了关于因果系统的工作，另一位成员回应称，他们目前的项目涉及将因果推理方法应用于理解思维链 (chains of thought)。
- **社区对水基和光基 GPU 持怀疑态度**：在一位用户提议使用更低 GPU 功耗构建新的 AI 系统后，一些成员分享了关于 [水基 GPU](https://news.stanford.edu/stories/2015/06/computer-water-drops-060815) 和 [光子计算 (optical computing)](https://arxiv.org/abs/2401.11514) 的链接，但大多数人对其可行性持怀疑态度。 
- **工程师寻求 LLM 架构合作伙伴**：一位具有分析哲学和数学背景的成员正在寻找具有 LLM 架构经验且对语言学或语言哲学感兴趣的人，共同合作一个研究导向的业余项目，旨在开发能够真正“理解”自然语言的新型 LLM 架构。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1393319796480938025)** (42 messages🔥): 

> `RNN tokenization, Mixture of Tokenizers, Byte-level models, n-simplical attention, antipodal dense features` 


- **RNN 的 Tokenization 速度快于 Transformer**：Tokenization 可以被 RNN 替换，从而获得比基于 Tokenization 的 Transformer 学习速度更快的 **Byte-level models**。该方案将普通 Transformer 的 Embedding 和 LM Head 替换为两个小的 **4 层 RNN**。
   - 该模型比较当前隐藏状态输出与前一个状态的点积 *p*，如果匹配度低于 **50%**，则该隐藏状态成为传递给主模型的 Token，并递归重复此过程两次。
- **Mixture of Tokenizers 方法**：Sebastian 的 [Mixture of Tokenizers](https://snimu.github.io/2024/09/03/mixture-of-tokenizers.html) 方法作为传统 Tokenization 方法的替代方案看起来很有前景。
   - 该方法旨在结合不同 Tokenizer 的优势以提高整体性能。
- **Unicode 码点 vs Bytes**：一位成员询问 **H-net** 在 **Unicode 码点**上运行是否比在 Bytes 上运行更好，但其他人认为考虑到词表大小，这可能大材小用。
   - 另一位成员指出，重点不在于拥有小的 Embedding 表，而在于拥有用于动态 Chunking 的最小文本单元，并表示他们*看不出为什么需要 Subcharacter chunks*。
- **关于 DiffuSSM 的讨论**：[DiffuSSM](https://arxiv.org/abs/2502.11927) 发表在 CVPR '24，但代码不可用，这与论文的开源声明相矛盾。
   - 用户链接了 [DiG](https://github.com/hustvl/DiG) 和 [DiC](https://github.com/YuchuanTian/DiC) 作为可能的替代方案，前者使用 Gated Linear Attention，后者是纯卷积。
- **分析 Antipodal Dense Features**：在阅读了在神经网络中发现 **Antipodal Dense Features** 的可解释性研究后，一位成员尝试用它们初始化网络，但在 modded-nanogpt 上似乎没有效果。
   - 该成员对没有效果感到惊讶，因为之前在 Mimetic Init 上的尝试发现了一些效果，而且优化权重的成对符号模式（Pairwise sign patterns）似乎特别困难。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

schizik12: 上涨的海平面？？
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1393772523774152715)** (3 messages): 

> `MechInterp Workshop CFP, NeurIPS, Open Source Library Spotlight` 


- **NeurIPS 的 MechInterp Workshop 回归！**：Mechanistic Interpretability Workshop 将在 **NeurIPS** 举行，征稿（CFP）开放至 **8 月 22 日**。论文篇幅最多为 **4 或 9 页**，欢迎正在 NeurIPS 审稿中的论文投稿：[mechinterpworkshop.com/cfp/](https://mechinterpworkshop.com/cfp/)。
- **MechInterp 寻求原则性方法**：该研讨会涵盖*任何通过模型内部结构更好地理解模型的原则性方法*，并欢迎任何有助于推动该领域发展的论文。
   - 需要审稿人，感兴趣者可以在此表达意向：[https://forms.gle/pAHLAFcJc3jDduGh6](https://forms.gle/pAHLAFcJc3jDduGh6)。
- **关注开源库**：征稿范围包括开源库，因为去年他们至少有一个 **Spotlight** 是关于开源库的。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1393340198045225011)** (16 messages🔥): 

> `lm-evaluation-harness mixed precision PR, logsumexp Trick for Logprob Calculation, Dynamic IFEval Dataset Benchmark` 


- **混合精度 PR 加速评估时间**：一名成员提交了 [lm-evaluation-harness 的混合精度 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/3138)，展示了 **Pythia-160M** 在 A30 上各种配置下的评估时间，并指出混合精度仅比转换整个模型稍慢，但比全精度快得多。
   - 对于 **Qwen1.5-7B**，在 Batch Size 为 32 的 24GB VRAM A30 上，使用 `softmax=fp32` 会导致 **OOM 错误**，而使用 `softmax=none` 则占用 **22775MiB** VRAM，耗时 **08:54**。
- **利用 Logsumexp 改进 Logprob 计算**：一名成员建议使用 logsumexp 技巧，在计算 Logprob 时仅计算目标 Logit 的 Logprob。
   - 该方法涉及 `logits[:, -1, 329] - logits[:, -1, :].logsumexp(-1)`，通过避免对所有 Logit 进行 Log-softmax，可能优化内存使用。
- **提出动态 IFEval 基准测试**：一名成员介绍了一个从 lm-evaluation-harness 仓库 Fork 出来的新基准测试，它可以根据定义的规则动态生成全新的 IFEval 数据集。
   - 该基准测试将数据集保存到 `data/dataset.yaml`，作者提供了[新分支的链接](https://github.com/davideguidobene/lm-evaluation-harness/tree/dynamic-ifeval)以供审查和反馈。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1393546919057555518)** (13 messages🔥): 

> `Neox, H100s, Transformer Engine, DeepSpeed` 


- **NeoX 在配备 Transformer Engine 的 H100 上运行良好**：一名成员分享了在 **H100** 上结合 **Transformer Engine** 使用 **NeoX** 的良好体验，并分享了他们的 [Dockerfile](https://github.com/EleutherAI/deep-ignorance/blob/main/Dockerfile.training) 和 [配置](https://github.com/EleutherAI/deep-ignorance/blob/main/pretraining/pretraining_neox_config.yml)。
- **DeepSpeed 需要为 MPI 标志进行单独安装**：由于集群上的网络权限问题，一名成员单独安装了 **DeepSpeed**，因为需要在 DeepSpeed 生成的 **MPI** 运行命令中添加额外的标志。
- **通过 Hack 修复 Wandb 错误**：由于未找到根本修复方法，通过一个 [Hack](https://wandb.ai/eleutherai/AISI/runs/ueet84jz/workspace?nw=nwuserkyledevinobrie1) 解决了 **W&B** 初始化期间的瞬时日志错误。
- **请求非 TE 配置的速度基准测试**：另一名成员询问是否测试了具有相同架构的非 TE 配置，以了解速度下降的情况。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1393346448581460249)** (153 条消息🔥🔥): 

> `Windsurf 被收购，Apple 继任与领导力，GPT-5 传闻，通用奖励函数 (Universal Reward Function)，Gemini Embedding 模型发布` 


- **Cognition 吞并 Windsurf**：**Cognition Labs** 宣布收购 **Windsurf**，包括其 IP、产品、商标和员工，旨在将 Cognition 的自主 Agent 与 Windsurf 的 Agentic IDE 相结合；所有 Windsurf 员工将获得财务参与权和加速行权。
   - 此次收购旨在为开发者打造强大的工具，实现在统一 IDE 内进行规划、并行工作委派和代码缝合等任务；然而，一些用户指出存在矛盾的报道，称**尚未行权的员工将一无所获**。
- **Meta 打造巨型兆瓦级机器**：**Meta** 正在建设庞大的 AI 集群，包括 **1000MW 的 Prometheus**（2026 年上线）和规模超过 **5000MW 的 Hyperion**，这使得目前运行的仅为 **150-200MW** 的 H100/H200 集群相形见绌。详见 [SemiAnalysis 文章](https://www.semianalysis.com/p/meta-is-building-ai-clusters-of-gigawatt)，该文详细介绍了 Meta 的数据中心策略和 Llama4 之后的经验教训。
   - 社区反应讨论了这对 AI 研究、NVIDIA 销售和能源来源的影响。
- **Karpathy 思考 RL 的现状**：**Andrej Karpathy** 在[这条推文](https://xcancel.com/karpathy/status/1944435412489171119)中讨论了当前强化学习 (RL) 方法的局限性，并提议为 LLM 增加一个“回顾/反思”阶段，以便从 rollout 中提取明确的“教训”并将其添加到 System Prompt 中，类似于人类的学习方式。
   - 这种“基于教训的学习”可以为每次 rollout 提供更多的监督信息，提高泛化能力，并催生出专门针对 LLM 的新学习范式，超越传统的游戏/机器人 RL。
- **Sam 规避安全风险，暂停发布**：**Sam Altman** 宣布延迟原定的权重开放模型发布计划，以进行额外的安全测试并审查高风险领域，他在[这条推文](https://xcancel.com/sama/status/1943837550369812814)中表示，*一旦权重发布，就无法收回*。
   - 社区普遍支持这一决定，强调安全重于速度。
- **Gemini 迈向全球**：**Logan Kilpatrick** 在[这条推文](https://xcancel.com/OfficialLoganK/status/1944806630979461445)中宣布 **Gemini Embedding 模型**正式商用，该模型在 **MTEB** 排行榜上排名**第一**，价格为**每百万 token 0.15 美元**。
   - 即将推出的功能包括 Batch Mode 支持和新的多模态 Embedding，并确认该模型目前已支持多语言，多模态能力即将推出。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 本周特别的双集播客！https://x.com/latentspacepod/status/1943774304166195402
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1393470317493751878)** (74 条消息🔥🔥): 

> `用于 ML 模型的 MCP，GenAI 中的 Agent 定义，MCP 中的剪贴板服务器，引导 (Elicitation) 实现` 


- **MCP 简化了机器学习模型部署？**：一位成员分享了一篇[博客文章](https://blog.dailydoseofds.com/p/deploy-any-ml-model-rag-or-agent)，建议 **MCP** 可以通过将模型服务与 Agent 工作流集成来简化 ML 模型部署。
   - 文章中的示例服务器启动了一个 **MCP** 服务器，暴露了一个运行推理的 "request" 工具，并使用 *transformers* 返回结果。
- **GenAI Agent 定义引发辩论**：成员们就 **AI Agent** 的定义展开了辩论，对于*工作流*是否应被视为 **Agent**，以及 **Anthropic** 定义的相关性持有不同看法。
   - 一位成员认为 **Anthropic** 的定义是最深思熟虑的，而另一位成员则认为 **Agent** 的定义早于 **LLM**，且比 **GenAI** 的范畴更广。
- **剪贴板服务器提案**：一位成员提议在官方 **MCP** 规范中增加**剪贴板服务器**，允许服务器写入客户端剪贴板，并计划在 [MCP-B](https://github.com/MiguelsPizza/WebMCP) 中实现。
   - 这将通过实现从 **MCP** 服务器到客户端更简便的数据传输来扩大实用性。
- **引导 (Elicitation) 实现的讨论**：一位成员正尝试实现**引导 (Elicitation)**，以便如果用户 Prompt 已包含足够运行工具的信息，则不需要进行**引导**。
   - 当你发送**引导请求**时，UI 由客户端应用程序提供。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1393367099958755388)** (9 messages🔥): 

> `Neurabase, mcp-spec, Director Run, MCP Evals, Albert Heijn MCP` 


- **Neurabase 宣称拥有最快的 MCP 服务器托管服务**: **Neurabase** 声称是完全运行在 **Cloudflare Workers CDN** 上的最快服务器托管服务，并称其为 MCP 服务器的中心枢纽，托管地址为 [neurabase.deploya.dev](https://neurabase.deploya.dev)。
- **mcp-spec：MCP 文档中的 MCP 嵌套！**: 一位成员为 MCP 文档构建了一个名为 **mcp-spec** 的 MCP 服务器，在将整个 MCP 规范粘贴到 **.md** 文件后，对整个文档进行了分区和索引分块，请查看 [repo](https://github.com/MCPJam/mcp-spec)！
- **Director Run 提供本地优先的 MCP 网关**: **Director Run** 团队创建了一个完全开源、本地优先的 MCP 网关，允许用户在 30 秒内将 **Claude**、**Cursor** 或 **VSCode** 连接到任何 MCP 服务器，可以在 [director.run](https://director.run) 或 [GitHub](https://github.com/director-run/director) 上找到。
- **mcp-evals：开源 MCP 服务器评估**: 一位成员分享了一个用于在 MCP 服务器上运行评估的开源仓库 [mcp-evals](https://github.com/mclenhard/mcp-evals)，该工具可以启动客户端、列出工具并循环遍历 prompts，以确定哪些 prompts 会触发哪些工具，这也可以作为 **e2e** 测试，因为你也可以调用实际的工具。
- **有人想要 Albert Heijn MCP 吗？**: 有人发布了一张图片，询问是否有荷兰人想要 **Albert Heijn MCP**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1393327318553853952)** (56 messages🔥🔥): 

> `Industrial Agents Training, Good World Models, Kimi K2, OpenAI Safety, BitNet vs Llama.cpp` 


- **尽管担心崩盘，投资者也不会停止资助 AI**: 一位用户表示，投资者将继续资助 AI，因为尽管担心崩盘，但*随着时间的推移，训练模型*以实现*人类双手的灵巧行为*的成本是*极其低廉的*。
- **OpenAI 推迟模型发布以提高安全性？**: 用户推测 **OpenAI** 由于 **Grok** 事件后的安全担忧，或者由于 [这条推文](https://x.com/Kimi_Moonshot/status/1943687594560332025) 中提到的 **Kimi K2** 的表现而推迟了模型发布。
- **OpenAI 是否在保留能力？**: 正如 [这条推文](https://x.com/_aidan_clark_/status/1943842131136983378) 所示，一名 **OpenAI** 员工可能暗示能力因素也是导致延迟的原因之一，这导致一些人愤世嫉俗地认为 *OpenAI 的模型表现不如 Kimi*，并且正在快速训练稍大一些的模型。
- **BitNet 讨论**: 成员们讨论了 **Llama.cpp** 对 **BitNet** 的支持，澄清这不是一个“谁更好”的问题，并指出其在最近训练简化后的有效性，尽管由于训练数据要求，目前仅限于 **7B** 左右的模型。
   - 一位用户表示：*这有点像在问自行车和轮胎哪个更好*。
- **对 Foundation Models 的批判**: 用户讨论了 [这篇论文](https://arxiv.org/abs/2507.06952)，该论文*更多是对如何使用 Foundation Models 的批判，而不是学习物理学*，强调盲目训练 Foundation Models 可能行不通，而不是说 Transformers 无法学习。
   - 另一位成员表示：*行星轨道是信息含量相当低的数据，因此你很可能会过拟合*，而另一人回应道：*这篇论文的重点更多在于询问隐式模型结构的方法，而不是他们从询问中得到的结果*。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1393497935052013672)** (6 条消息): 

> `U-Net 中的 ResNet 和 Attention，Hugging Face Transformers，U-Net 定义混淆，Accordion Networks (WWWWWW)，Kimi K2 模型` 


- **ResNet、Attention 和 U-Net 实现出现**：讨论涉及在 **U-Net 架构** 中使用带有 **attention** 的 **ResNet**，引用了[一篇使用带有 attention 的单阶段 ResNet 的论文](https://arxiv.org/abs/2204.12084)以及[另一篇仅在潜空间（latent space）中使用带有 attention 的等宽 ResNet 堆栈的论文](https://arxiv.org/abs/2210.08506)。
   - 还提到了[一篇使用带有 attention 的 ResNet 堆栈](https://arxiv.org/abs/1909.10360)来替换编码器和解码器层的论文，这可能就是 Hugging Face 的 transformers 库中所使用的方案。
- **U-Net 定义令人困惑**：关于什么构成 **U-Net** 存在混淆，一些 GitHub 仓库（如[这一个](https://github.com/lu-jincheng/AttentionUnet)）错误地将架构标记为 **ResNet + Attention + U-Net**，而它们并非基于 U-Net。
   - 发言者还提到了[一种“手风琴（accordion）”网络架构](https://arxiv.org/abs/2110.08811)，并将其称为 *pagoda network*（宝塔网络），认为这是未来值得研究的网络拓扑结构。
- **Kimi K2 表现确实非常出色**：发言者对 **Kimi K2** 模型给予了高度评价，称其为某些用例的首选，并将其列入前三名。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1393323616099369051)** (3 条消息): 

> `Twitter 链接` 


- **推文分享！**：频道中分享了三个 Twitter 链接：[Elie Bakouch](https://fxtwitter.com/eliebakouch/status/1943687750563004801)、[Hayden Field](https://x.com/haydenfield/status/1943784071101907128) 和 [Signulll](https://x.com/signulll/status/1944851904888234293)。
- **更多推文**：只是添加第二个主题以满足 minItems=2 的要求。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1393309158819893288)** (52 条消息🔥): 

> `Grok 4 Aider 基准测试，Aider 基准测试更难的任务，Aider 排行榜更新，Aider Agents，Windows 上的 Aider` 


- ****Grok 4** 在 Aider Polyglot Coding Benchmark 中达到 **80%****：**Grok 4** 在 aider polyglot coding 基准测试中获得了 **80%** 的分数，在排行榜上名列第 4，排行榜可以在[这里](https://aider.chat/docs/leaderboards/)找到。
- **Aider 基准测试需要更具挑战性的任务吗？**：一位成员想知道，既然现在许多模型的分数都在 **80%** 左右，aider 基准测试是否需要*更难的任务*。
- ****Gemini 2.5 Pro** 与 **Gemini 1.5 Pro** 模型对比**：成员们讨论了在 Gemini 模型中使用 `GEMINI_API_KEY=[blah] aider --model gemini/gemini-2.5-pro`，并表示 **Gemini 1.5 Pro** 拥有 **2M context**，但 **2.5 Pro** 更聪明。
   - 一位成员分享了一张[模型价格的截图](https://cdn.discordapp.com/attachments/1131200896827654149/1394369659834335282/image.png?ex=68768f71&is=68753df1&hm=d1d15b32f9716634b4531a2c8f167e20297645513538e4d341d377e0a50e1064&)。
- **通过 MoonshotAI 报告的 **Kimi** Aider Polyglot Coding 基准测试结果**：一位成员提到使用了 [MoonshotAI](https://platform.moonshot.ai/docs/pricing/chat#generation-model-kimi-k2) 报告的 **Kimi** 在 aider 基准测试中的分数。
   - 另一位成员建议使用官方供应商的定价而非第三方估算以避免混淆，并补充说 [Artificial Analysis](https://link.to.analysis) 有相关的成本指标。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1393599295604916254)** (10 条消息🔥): 

> `Zed editor schema validation for aider conf file, Github Copilot support in Aider, COBOL support to Aider, LiteLLM Proxy config and Aider config, Gemini thinking tokens` 


- **Aider 配置文件获得 Zed 编辑器的 Schema 验证**：一位用户报告称 **Zed 编辑器** 现在为 aider 的配置文件提供了 [schema validation](https://zed.dev/)，这促使他们将 `test-cmd` 转换为数组，从而导致了配置错误。
   - 错误信息建议将 `test-cmd` 的动作类型更改为 'append' 或设置 `nargs`，但用户不清楚其具体影响。
- **Deepseek 推荐使用 `tsc --noEmit` 进行静态类型检查**：一位用户在配置 `bun typecheck` 时遇到困难，另一位用户建议根据 [Deepseek](https://deepseek.com/) 的推荐使用 `tsc --noEmit` 进行静态类型检查。
   - 用户承认 `/lint` 仅检查脏文件（dirty files），这与他们通常的 Python/Ruff 工作流不同，在后者中由于 Ruff 的速度极快，每次都会检查所有内容。
- **关于 Aider 中 GitHub Copilot 支持的困惑**：一位用户对 **Aider 中 GitHub Copilot 的支持**状态感到困惑，因为 [文档](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/llms/github.md) 显示已支持，而一个 [issue](https://github.com/Aider-AI/aider/issues/2227) 则表明该功能可能仍在开发中。
   - 讨论旨在澄清文档和该 issue 是否指向 Copilot 集成的同一个方面。
- **Aider 的语音命令缺乏配置选项**：一位用户询问是否可以更改 **`/voice` 命令中使用的 Model 和 Endpoint**。
   - 在现有消息中未提供具体回答。
- **COBOL 解析器导致 Aider 段错误（Segmentation Fault）**：一位用户在为 Aider 添加 **COBOL 支持**（创建 `tags.scm`、使用 Tree-sitter 编译 COBOL 解析器并进行必要的代码更改）后遇到了 **段错误**。
   - 段错误发生在加载 COBOL 共享库时，用户已经验证了导出符号和解析器的正确性，正在寻求关于 Tree-sitter 集成的常见陷阱或调试建议。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1394042810780618804)** (1 条消息): 

> `MOOC Certificates, Certificate Requirements, Feedback form` 


- **Advanced LLM Agents MOOC 证书已发放**：Advanced LLM Agents MOOC 的证书已经发布，表彰了参与者的成就，共颁发了 **232 个 Trailblazer**、**38 个 Mastery**、**80 个 Ninja**、**1 个 Legendary** 和 **3 个 Honorary** 证书。
   - 公告感谢了参与者和客座演讲者，同时鼓励获得证书的人分享他们的成就。
- **证书领取核对清单发布**：为预期获得证书但未收到的参与者提供了一份核对清单，包括检查垃圾邮件文件夹、确保所有课程作业均使用同一电子邮件地址完成，以及验证是否完成了文章作业和证书声明表单。
   - 完成作业和声明表单后，应该会收到确认电子邮件。
- **欢迎通过匿名表单提供反馈**：鼓励参与者通过 [匿名反馈表单](https://forms.gle/3a136zS4ivcQFzhT7) 分享对 MOOC 的反馈。
   - 公告最后感谢了大家度过了一个美好的学期，并鼓励在指定频道提出问题。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1394046754961363024)** (49 messages🔥): 

> `证书问题、证书声明表、文章提交表、证书格式错误、证书缺失` 


- **证书领取遭遇波折！**: 几位用户报告了领取证书时遇到的问题，通常与退订邮件列表或错过 [证书声明表](https://forms.gle/PQkR4ZYQJUbFpPcw9) 有关。
   - 工作人员在发现一名用户被误退订后为其重新订阅，并将另一份证书发送到了不同的邮箱，但工作人员重申，*他们没有足够的人力*来处理每一个个案。
- **格式错误导致收尾困难！**: 一位用户报告称，其证书上的姓名与等级重叠，导致在尝试将证书发布到 LinkedIn 时出现问题。
   - 工作人员修复了该证书的格式错误，并表示 *现在应该已经修复了！非常抱歉*，并提到了可以在 PDF 名称中找到的证书编号。
- **文章作业流程混乱困扰学生！**: 一些用户意识到，尽管完成了文章、测验和其他要求，但他们漏填了 [文章提交表](https://forms.gle/399VqBP88WL1AJ4J6)。
   - 工作人员强调 *非常抱歉，但我们现在无能为力*，无法为错过这些表格/截止日期的学生提供特殊处理。
- **积极反馈助力 MOOC 未来！**: 一位用户建议提供一个 *统一的 Excel 表格或进度追踪器*，以便参与者检查自己的状态并防止最后时刻出现问题。
   - 工作人员感谢了用户的建议，并表示 *正是由于大家的参与和热情，我们有望在未来改进所有课程和作业的交付形式！*。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1393316265006858270)** (9 messages🔥): 

> `Mojo 内部的汇编代码编写、Modular 社区活动追踪、Discord 通知、Mojo 标准库汇编模块` 


- **Mojo 支持汇编！**: 一位成员询问是否可以在 **Mojo** 内部编写汇编代码，例如进行一些简单的 syscalls。
   - 另一位成员回复说 *这是可行的，但文档记录不全*，并提供了 [Mojo Standard Library Assembly Module](https://github.com/modular/modular/blob/main/mojo/stdlib/stdlib/sys/_assembly.mojo) 的链接。
- **Modular 社区活动追踪工具投票**: Modular 正在就社区成员偏好如何追踪 **Modular 社区活动**（如社区会议、直播、会议演讲、线下聚会等）进行投票。
   - 选项包括 [Modular 社区 Google 日历](https://modul.ar/community-meeting) 和 [Modular 的 Luma 活动页面](https://lu.ma/modular-ai)，以及其他建议。
- **Discord 通知最给力**: 一位成员表示 *来自 Discord 或其他订阅的通知对我来说是最有效的*。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1394335646302077050)** (1 messages): 

> `七月社区会议、基于 Hashable 的哈希、FFT 实现、Mojo-Lapper、量子电路模拟器` 


- **七月社区会议即将开始！**: 七月社区会议定于约 2 小时后开始，届时将有多个主题的演讲。
   - 演讲内容包括关于 **基于 Hashable 的哈希**、**FFT 实现**、**Mojo-Lapper**（一个重叠检测库）以及 **量子电路模拟器** 的讨论 —— 欢迎加入 [Discord 活动](https://discord.gg/bDuqA2FT?event=1374399851923640411)！
- **团队 Q&A 即将到来**: 与每次社区会议一样，团队已请求用户提前提交任何问题。
   - 请使用 [此 Google 表单](https://forms.gle/hXqHttkHkYtPxg5NA) 向他们提问！


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1393306094876622961)** (30 messages🔥): 

> `Mojo error messages, M1 Metal 3 GPUs, Autotune functionality, EqualityComparable, Atomics on GPU` 


- ****Mojo 错误消息**：在哪里可以找到它们？**：一名成员在遇到错误后寻找 Mojo 错误消息列表，得到的建议是咨询 **#help 频道**并使用 **@kapa.ai** 获取帮助。
   - 一名成员遇到了一个需要将 `self.id` 设置为某个值的错误。
- ****M1 Metal 3 GPU**：Mojo 尚未支持**：一位用户询问是否可以在 Mojo 和 Max 中利用 **M1 Metal 3 GPU** 进行处理，但被告知目前尚不支持 **Apple Silicon GPU**，但该功能正在**开发中**。
   - 分享了一个相关的 [GitHub commit](https://github.com/modular/modular/commit/1afcc38281c637159adc03e2a6043310ac340284) 链接，尽管它仅涉及构建系统的检测。
- ****Autotune 功能**：被 Benchmarking 循环取代**：一位用户询问 Autotune 功能回归的时间表，并指出该功能因重新设计而被移除。
   - 该功能可能会导致编译时间大幅增加，现在可以用 for 循环和一些 Benchmarking 来替代它。
- ****EqualityComparable**：缺少 equal 函数**：一位用户报告了在 struct 中实现 **EqualityComparable** 所有要求时遇到的错误，并寻求关于可能遗漏内容的指导。
   - 成员们推荐参考 [手册](https://docs.modular.com/mojo/manual/) 以入门。
- ****Capturing 关键字**：文档探索**：一位用户寻找 Mojo 中 **'capturing' 关键字**的具体示例，并注意到它在 'foreach' 等其他主题的解释中被提及。
   - 一名成员分享了一个[解释](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)链接以及一个[相关 issue](https://github.com/modular/modular/issues/5020)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1394019761624449175)** (4 messages): 

> `arg_nonzero kernel, max.kernels import, mojo build max kernels` 


- **用户在导入 `arg_nonzero` kernel 时遇到困难**：尽管已将 `max` 列为依赖项，一位用户在尝试从 `mojo.kernels.nn.arg_nonzero` 导入 `arg_nonzero` 和 `arg_nonzero_shape` 时仍然遇到错误。
   - 用户收到了诸如 *unable to locate module 'mojo'* 和 *'kernels' does not refer to a nested package* 的错误。
- **`max.kernels` 导入失败**：用户发现 `max.kernels.nn.arg_nonzero` 无法访问或未暴露子模块。
   - 错误消息为 *'kernels' does not refer to a nested package*。
- **用户通过从源码构建 `max` kernels 解决了导入问题**：一位成员建议 kernels 应该是可以通过 `from nn import ...` 导入的独立模块，并建议如果无法从 `max` 包访问，则从源码构建它们：[Modular 论坛帖子](https://forum.modular.com/t/importing-max-kernels-from-source/1942/2)。
   - 用户确认运行 `mojo build -I ../modular/max/kernels/src/` 解决了该问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1393322177998098542)** (37 messages🔥): 

> `Manus Flutter Web Emulator, Startup Advice, Google Drive Save Error, Manus Website Outage, Manus Fellowship` 


- **Flutter Web Emulator 扩展获得关注**：一位成员分享了他们使用 **Manus** 创建的 [Flutter Web Emulator](https://marketplace.visualstudio.com/items?itemName=HafizRizwanUmar.flutter-web-emulator) 扩展，并指出在没有任何推广的情况下，两个月内获得了 **1900 次安装**。
- **在线孵化器推荐**：一位成员建议加入在线孵化器以寻找合作伙伴和顾问，并推荐查看 [f6s.com](https://f6s.com)。
   - 另一位用户建议向 **Manus** 寻求分步开发业务的建议，并开玩笑说有人可以创建一个 Manus 在线业务孵化器。
- **Google Drive 保存错误**：一位成员报告了一个潜在的 Bug，即保存最近的项目到 **Google Drive** 可以正常工作，但保存之前的项目会导致 **Google Auth 错误**。
- **Manus 网站遭遇故障**：多名成员报告无法访问 **Manus** 网站以及在 **manus.space** 上的部署，表明可能存在服务中断。


  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1394352702833688597)** (1 条消息): 

> `Featured Notebooks, NotebookLM` 


- **Featured Notebooks 登陆 NotebookLM**：团队宣布在主页推出 **Featured Notebooks**（精选笔记本），内容涵盖从*科学探索*到*专家建议*的各个领域。
   - 用户可以[直接访问这些笔记本](https://blog.google/technology/google-labs/notebooklm-featured-notebooks)以了解更多信息。
- **通过 Featured Notebooks 探索多样化主题**：这些笔记本提供了一系列内容，从**科学探索**到**实用指南**和**专家建议**，满足各种兴趣和需求。
   - Featured Notebooks 板块为用户提供了在 NotebookLM 平台内轻松获取宝贵资源和见解的途径。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1393479214648524872)** (11 条消息🔥): 

> `Targeted Fiction Editing with AI, NotebookLM integration with Apple system toolkits, AI for extracting information from books` 


- **AI 针对小说编辑**：用户讨论了利用 AI 对虚构作品进行针对性编辑，为新手作者提供可操作的建议和示例，特别是分析从开篇签约到结尾场景的初稿手稿，例如使用提示词：*Analyze [X]; Provide actionable advice as paired with written examples for [Y]*。
   - 讨论集中在对手稿每个元素进行全面的深度挖掘，重点关注内容的连贯包装和写作质量，甚至有用户表示，这些结果让原帖作者从中获得了*两小时*的内容量。
- **NotebookLM 避开 Apple 原生功能**：据称，将文本分享到 **NotebookLM app** 会创建一个包含源材料的新笔记本，而分享深度研究报告则默认分享一个包含报告内容的文本文件，这表明其未进行特殊处理。
   - 一名成员指出，这可能是因为 Google 习惯性地*反感接入 **Apple system toolkits** 和 **human interface guidelines***（人机交互指南），换句话说，不太可能出现具有原生功能的原生应用。
- **AI 从书中提取一切**：一位用户分享了一张在 Reddit 上发现的图片（原图来自 Pinterest），标题为《如何使用 AI 从书中提取一切》。
   - 该图片包含**多张截图**，概述了使用 AI 工具分析和提取书籍信息的步骤，其中包括使用 **chatbots** 和 **optical character recognition (OCR)** 来转换图像中的文本。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1393316196480450591)** (24 条消息🔥): 

> `Source naming conventions, Audio file generation length, Embedding model details, Server tag requests, iOS app functionality` 


- **“命名游戏”：将生成的来源命名为与预设查询相同**：一位用户建议，当使用“FAQ”等预设查询时，生成的来源应命名为与按钮完全相同的名称（如 **'FAQ'**），以提高组织效率并方便查找来源，尤其是在拥有大量来源的笔记本中。
- **“播客恐慌”：音频文件生成长度最近变短了？**：用户反映**音频文件生成长度**最近似乎变短了，生成的长度约为 **10-15 分钟**，而之前超过 **30 分钟**，即使调整了针对长播客的设置也是如此。
- **“Embedding 之谜”：弄清底层的 Embedding 模型**：一位用户询问 NotebookLM 的 Embedding 是使用了 **`gemini-embedding`**、**`text-embedding-004/005`** 还是 **`text-multilingual-embedding-002`**。
   - 这目前仍然是一个谜。
- **“服务器标签趣事”：服务器标签请求？**：一位用户询问是否有针对 NotebookLM 的**服务器标签**计划，而另一位用户则询问关于在 **iOS app** 上阅读置顶笔记或置顶回答的功能。
- **“发布难题”：公开笔记本发布？**：一位用户询问是否有**发布笔记本**的方法，并分享了 [Google 博客文章的链接](https://blog.google/technology/google-labs/notebooklm-featured-notebooks/)，询问发布功能是否仅限于 Google 的合作伙伴。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1394422929927700582)** (1 messages): 

> `LlamaIndex Meetup Amsterdam, Office Hours, Notebook Llama, Context Engineering, Research Agent` 


- **LlamaIndex 抵达美丽的利瑟（阿姆斯特丹附近）**：LlamaIndex 将于 7 月 31 日在[阿姆斯特丹举办见面会](https://lu.ma/vzwsj72w)，重点关注 **LlamaIndex & Snowflake Data Agent Builders**。
   - 请务必报名以预留名额。
- **报名参加轻松的学习会议！**：下一场 LlamaIndex [Office Hours](https://lu.ma/wkrn4nbz) 将于 8 月 5 日举行。
   - 报名参加此会议。
- **Notebook Llama 克隆 NotebookLM**：NotebookLlama 是 LlamaIndex 开发的一个 **NotebookLM 克隆版**，已在 [GitHub](https://github.com/run-llama/notebookllama/tree/main) 上发布，并已获得超过 **1k stars**。
   - 快去查看该仓库！
- **上下文工程（Context Engineering）技术**：LlamaIndex 在其博客上介绍了 [Context Engineering](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) 技术。
   - 这篇博文涵盖了 Context Engineering 的**定义和实现方法**。
- **Gemini 2.5 Pro 赋能研究 Agent**：LlamaIndex 在[本教程](https://ai.google.dev/gemini-api/docs/llama-index)中演示了如何使用 **LlamaIndex & Gemini 2.5 pro** 构建研究 Agent。
   - 学习如何利用 LLM 的力量进行研究！


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1393336486786760756)** (3 messages): 

> `Notebook Llama new features, RAG Apps, Google Gemini 2.5 Pro` 


- **NotebookLlama 获得更多功能**：**NotebookLlama**（@NotebookLM 的开源替代方案）的新版本允许用户从文件中提取并下载**图像**和**表格**，并以交互方式可视化所有表格数据。
   - 用户现在还可以与全新改进的 **NotebookLlama** 进行[对话（chat）](https://t.co/Csl3HhMzYB)。
- **构建真实的 RAG 应用**：发布了一份关于如何构建真实世界 **RAG (Retrieval-Augmented Generation)** 应用程序的综合指南，引导用户完成从原始数据到完整管道的整个过程，详见[此推文](https://t.co/Zpxcpsuk7t)。
   - 该指南由 LlamaIndex 的 @itsclelia 和 @qdrant_engine 的 @krotenWanderung 合作完成，提供了来自两个团队的见解。
- **Gemini 2.5 Pro 赋能研究 Agent**：一个新的示例演示了如何构建由 **LlamaIndex workflows** 和 **Google Gemini 2.5 Pro** 驱动的多 Agent 研究助手，详见[此推文](https://t.co/BU807U1ecI)。
   - 该 Agent 可以使用 Google 搜索网络，并通过专门的记录 Agent 进行笔记记录。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1393316751391064064)** (27 messages🔥): 

> `LlamaIndex Partner Program, Tool Calling Models, Synk Hiring, Response Synthesizers` 


- **LlamaIndex 寻求联合营销合作伙伴**：来自 **Bright Data** 的 Aviv 正在寻找 LlamaIndex 的联系人，以讨论其[网页数据/抓取工具](https://lu.ma/aoc5opn4)集成的联合内容或联合营销机会。
   - 他们的目标是帮助开发者从集成中获得更多价值。
- **最佳工具调用模型？Llama3 和 Mistral 在列**：一位成员询问关于可在本地运行且 **VRAM <= 96 GB** 的最佳开源工具调用（Tool Calling）模型的建议。
   - 虽然承认 **Claude** 目前是“最强”的，但他们指出 **llama3.3 70b** 和 **mistral-32-small** 在配合 LlamaIndex 使用时表现不错，并寻求其他意见。
- **Synk 项目招募匿名倡导者**：专注于去中心化、匿名和安全浏览器的 **Synk** 项目正在[招聘](https://twitter.com/MetaToyGame)多个职位，包括**开发人员**、**QA 工程师**、**DevOps 工程师**、**版主**、**营销分析师**和**测试人员**。
   - 该项目提供正式雇佣、签署文件、保障薪资以及灵活的工作时间。
- **生成更详细的响应**：一位成员询问了生成大量文本而非摘要的方法，即迭代每个要点以创建更大的报告。
   - 另一位成员建议研究 **GraphRAG**，并指向了一个他们正在使用 Neo4j 进行适配的 [GraphRAG 示例](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/#build-communities)。


  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1394091518444310661)** (1 messages): 

> `Synk, MetaToyGame, Decentralized system for browsers` 


- **Synk 寻求新的协同效应**：快速发展的项目 **Synk** 正在寻找*充满野心和激情的人才*，共同开发一个完全匿名且安全的**浏览器去中心化系统**。
   - 他们正在招聘开发人员（后端、前端、区块链）、QA Engineer、DevOps Engineer、版主（游戏聊天）、市场分析师和 Beta-Testers，并邀请大家在 [MetaToyGame X](https://x.com/MetaToyGame) 上查看他们的产品。
- **Synk 开放 Beta 测试员职位**：**Synk** 正在招聘 **Beta-Testers**，无需经验！
   - 如果你有兴趣，可以私信（DM）讨论工作细节。他们提供签署文件的正式雇佣、保障薪资和灵活的工作时间。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

yamashi: Kimi K2 是用 Muon 训练的，这会不会是未来的趋势？
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1393342179803660358)** (17 messages🔥): 

> `Async Recipe, Flex Attention memory usage with complicated masks, torch.cuda.memory._set_allocator_settings, Sync GRPO Recipe` 


- **Async Recipe 并不适用于所有模型**：有建议认为应保留一个功能完整的 Recipe，因为 **Async Recipe** 并非在每个模型上都能运行。
   - 另一位成员表示赞同，并指出该 Recipe 存在一个*关键问题*，并提交了一个 [PR](https://github.com/pytorch/torchtune/pull/2876) 来解决它。
- **Flex Attention Kernel 的内存需求有待探索**：**Flex Attention Kernel 使用的 shmem** 量将取决于 **scoremod** 和/或 **maskmod**（由多个部分组成的复杂掩码）。
   - 有人提出疑问：如果通过 `and_masks` ([链接](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L731)) 构建掩码，Triton Kernel 是否需要额外的内存。
- **扩展内存分配以增强鲁棒性**：一位成员建议使用 `torch.cuda.memory._set_allocator_settings("expandable_segments:True")` 来使代码更具鲁棒性并动态更改。
   - 这一建议得到了认可，并鼓励该用户提交 PR；同时建议在单元测试的 init 中进行更改，因为 Torch 在 Tune 之前就被导入了。
- **Sync GRPO Recipe 目前已损坏**：有人指出 **Sync GRPO Recipe** 已损坏 ([PR](https://github.com/pytorch/torchtune/pull/2876))，并建议回滚到 #2697 之前的版本，或者使用已回滚这些功能的 `krammnic/torchtune`。
   - 提交者补充说，由于 Reward 结构不同，在 main 分支中回滚并不是一个好的解决方案，他们需要等待其 [PR 被合并](https://github.com/pytorch/torchtune/pull/2681)。
- **Trainer 因 Optimizer 编译而崩溃**：一位成员报告说，在使用 Cosine LR Scheduler 时，**Trainer 在 compile: True 的情况下崩溃**。
   - 另一位成员建议通过以下配置禁用 Optimizer 编译：
```
compile:
  model: true
  loss: true
  scale_grads: true
  optimizer_step: false
```


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1393752309728542730)** (2 messages): 

> `Token Training, Grokking` 


- **错失了 Token 训练机会？**：一位成员质疑为什么某些行动没有在 **1T Tokens 之前**采取，暗示可能错失了潜在的收益。
   - 他们推测决策者可能在期待某种 **Grokking** 现象的发生。
- **关于 Grokking 的推测**：用户提到了一则推文链接。
   - 他们怀疑做出某项决策是否是为了等待 **Grokking** 的出现。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1394137154837282927)** (19 messages🔥): 

> `Frontend Reimplementations, Metal Profiling API, ONNX Flaky and Coredumps, Driving Vision ONNX Issue, Tinygrad Apps and Examples` 


- **前端重新实现引起波澜**：George Hotz 建议不要重新实现前端，并指出现有 spec 的完整性和测试已经足够，同时回应了 [jafioti](https://discord.com/channels/1042493742516123658/1042493743237623858) 与 tinygrad 团队进行 *chat* 的请求。
   - 他表示 *更多不完整的前端可能不是开发精力的良好用途*，并对参与现有对话持开放态度。
- **Metal Profiling API 浮出水面**：uuuvn 分享了一个 [Metal profiling API](https://github.com/tinygrad/tinygrad/pull/3627)，它类似于 AMD 上的 sqtt，并表示 *不确定你是否看到了这条消息——有一个用于 Metal 分析的 API，与 AMD 上的 sqtt 类似*。
   - 他还分享了一个 [onnx 文件](https://github.com/geohotstan/tinygrad/blob/mergable_onnx/extra/onnx.py)，用 1000 行代码实现了所有的 ONNX。
- **ONNX 复现性受 Coredumps 困扰**：b1tg 报告了在 ONNX 复现过程中的 coredump，指出 python 进程在 `_metadata_wrapper` 内的 `_METADATA.set` 处崩溃，这可能预示着一个 [CPython bug](https://gist.github.com/b1tg/af91eb21b96137d6ccf32ef237fddb64)。
   - 他们链接到了[之前的段错误 (segfaults)](https://github.com/tinygrad/tinygrad/actions/runs/15118381993/job/42494713756?pr=10413#step:9:26) 以及在 ONNX 解析器合并期间观察到的[另一个错误](https://github.com/tinygrad/tinygrad/actions/runs/15472667248/job/43560773882?pr=10642#step:9:64)，该错误在 `_METADATA.get` 处崩溃。
- **Driving Vision ONNX 根本原因已确定**：uuuvn 似乎已经确定了 `driving_vision.onnx` 问题的根本原因并开发了修复程序，指出这与 bitcast folding 有关，特别是某些 uchar 到 half 的折叠。
   - uuuvn 在提交 PR 之前正在尝试编写一个最小复现测试 (minimal repro test)，但在编写合适的测试时遇到了困难。
- **Apps 和 Examples：数量还是质量？**：一位用户表示有兴趣为实用的应用/示例（图像去重、快速人脸检测、视频内容识别）移植模型，并提到他们依赖于像 ffmpeg 这样臃肿的依赖项。
   - 一名成员回应称，团队*更感兴趣的是让 tinygrad 对开发者来说更容易使用，而不太感兴趣支持更多的示例*。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1393348325243224225)** (19 messages🔥): 

> `Gemma 3, Nomic-embed-v2 finetuning, LocalDocs embedding issues, Nomic API server performance, RAG for lore` 


- ****Gemma 3** 获得及格分数**：一位成员发现 **Gemma 3** 是他们发现的唯一一个*尚可接受*的模型。
- **Nomic-embed-v2 微调面临 Cloudflare 访问问题**：一位用户报告称，在尝试通过 `aws s3 ls` 命令访问 Cloudflare R2 存储中的数据以微调 **nomic-embed-v2** 时，遇到了 *Access Denied* 错误。
   - 使用的命令为：`aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`。
- **LocalDocs 嵌入过程卡在 0%**：多位用户报告了 **LocalDocs** 的问题，即嵌入过程卡在 0%，即使是对于很小的文本文件也是如此（[示例图片](https://cdn.discordapp.com/attachments/1090427154141020190/1393967383164747776/image.png?ex=68766a4a&is=687518ca&hm=d4f718f022c19b58b75285f6c78a437bde005e3802c7b95e6fd850d6242b17c5&)）。
   - 一位拥有 **3060 GPU**、**9950x 处理器**和 **64GB RAM** 的用户在嵌入过程停滞后寻求帮助，被建议在 LocalDocs 设置中启用其 NVIDIA GPU 及其 VRAM 以提高性能。
- **Nomic API 服务器响应缓慢**：一位 Nomic 新用户在运行 Debian 12、配备旧款 AMD 处理器和 24GB RAM 的机器上，经历了 **两小时** 的 Nomic API 服务器响应延迟。
   - 建议指出系统可能完全运行在 **CPU** 上，使用更小的模型或更好的显卡可能会提高性能。
- **RAG 文本相似度搜索**：一位用户询问关于使用模型或 Agent 来存储和查询大量背景设定 (lore) 的问题，有人建议使用 **LocalDocs** 进行带有文本相似度搜索的 **RAG** (Retrieval-Augmented Generation) 可能是一个可行的解决方案。


  

---

### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1393372448162054185)** (12 messages🔥): 

> `Aya Expanse 32B, Preference Optimization Dataset, Cohere Labs Discord Server` 


- **Aya Expanse 32B 尽管发布已久但表现出色**：一位成员发现 **Aya Expanse 32B** 令人印象深刻，并指出尽管它发布已久，但（大部分时间）能与 **Roocode** 配合工作。
   - 该成员将其与 **Command-R 08 2024** 进行了对比，强调许多同等规模的现代开源权重（open weight）模型都失败了。
- **寻求用于测试的 credit code**：一位成员询问如何获取用于测试目的的 **credit code**。
   - 另一位成员回复了一个 [Discord 指南链接](https://discord.com/channels/954421988141711382/954421988783444043/1387628087541366795)，其中提供了相关说明。
- **请求发布偏好优化数据集（Preference Optimization Dataset）**：一位成员询问 **Cohere** 是否已经发布了[这篇论文](https://aclanthology.org/2024.emnlp-main.729.pdf)中提到的偏好优化数据集。
   - 该成员还附带了一个可能与讨论相关的 [推文链接](https://x.com/SatyaScribbles/status/1944758063363232093)。
- **加入 Cohere Labs Discord 服务器**：一位成员被邀请在 **Cohere Labs Discord 服务器**中分享他们的想法。
   - 另一位成员提供了 [Discord 服务器链接](https://discord.com/channels/954421988141711382/954421988783444043/1387628087541366795)，鼓励他们在该社区发帖。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1393372517313282159)** (6 messages): 

> `Machine Learning Research, High Performance Computing, Quantum Computing, PhD opportunities` 


- **讲师专注于 ML 研究**：一位来自 NED 工程技术大学的讲师在 **PhD** 准备阶段专注于 **机器学习研究与应用**。
   - 他们旨在与研究人员和开发人员建立联系，紧跟 AI 和 **ML** 的发展动态，为他们的 **PhD** 打下坚实基础。
- **学生追求研究职业生涯**：一位来自巴基斯坦的计算机科学专业学生对 **ML**、**High Performance Computing** 和 **Quantum Computing** 感兴趣。
   - 他渴望参与研究工作并做出贡献，目标是成为一名 **PhD** 研究员，并寻求向社区学习。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

okhattab: 是的。或者阅读关于 **IReRa** 的论文！
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1394030320059420775)** (4 messages): 

> `NFT Public Mint, OpenSea Rewards Claim, Custom LLM Adapter Error, Arc Prize DSPy Hacking` 


- **NFT 公开铸造现已开启！**：一个 **NFT** 项目的公开铸造已经 **上线**，仅剩 **1,820 个 NFT**。
   - 参与了 **OS2** 奖励计划的用户可以通过“Rewards”选项卡在新的 **OpenSea** 平台上领取他们的宝藏，但要警惕损坏的链接。
- **排除自定义 LLM 适配器故障**：一位使用 **自定义 LLM 适配器** 的用户在调用 **Bedrock API** 时遇到了 **ValidationException**。
   - 错误信息显示 *输入内容对于所请求的模型来说太长了*。
- **Arc Prize DSPy 黑客们联合起来！**：一位成员正在寻找在 **Arc Prize** 中使用 **DSPy** 的合作者。
   - 他们表示有兴趣了解其他人正在采取的方法。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1393376768509083749)** (2 messages): 

> `Llama 4 Scout vs Llama 3.1 70B, BFCL Website Rendering Bug, Llama-3.3-70B-Instruct (FC) Score Discrepancy` 


- **Scouts 证明尺寸并非一切**：一位成员指出，更大的模型尺寸并不保证更优越的性能，并引用了 **Llama 4 Scout** 的表现劣于 **Llama 3.1 70B** 的例子。
   - 其他人表示，架构和训练数据的改进即使在较小的模型上也能带来更好的结果。
- **网站渲染故障影响 Llama 评分**：一位成员怀疑网站存在渲染问题，注意到 **Llama-3.3-70B-Instruct (FC)** 在 Non-live Simple 类别中显示的分数为 **74.33**。
   - 该成员指出网站分数与 [git repo](https://github.com/HuanzhiMao/BFCL-Result/blob/main/2025-06-14/score/meta-llama_Llama-3.3-70B-Instruct-FC/BFCL_v3_simple_score.json) 中的分数存在差异，后者记录的分数为 **94**。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1394379328053055651)** (1 条消息): 

> `Windsurf, Cognition, Devin, AI coding` 


- **Windsurf 与 Cognition 合并！**：Windsurf 宣布他们正与 **Devin** 的创造者 **Cognition** *强强联手*，共同重塑 **AI coding**。
   - 此次收购旨在将 Cognition 的 autonomous agents 与 Windsurf 的 agentic IDE 相结合，打造突破性的开发者体验。
- **人机协作是关键！**：Windsurf 表示，他们始终相信软件开发的未来在于 **human-AI collaboration**。
   - 他们表示，这种协作将实现开发者能力的*真正增强*，而不仅仅是自动化。
- **Windsurf 和 Cognition 塑造 AI Coding 的未来**：两支世界级团队正携手塑造 **AI coding** 的下一个时代。
   - [公告视频](https://x.com/cognition_labs/status/1944819486538023138) 和 [YouTube 链接](https://www.youtube.com/watch?v=fBB_-ezuHn8) 提供了更多详情。