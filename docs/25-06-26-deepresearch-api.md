---
companies:
- openai
- google
- black-forest-labs
- deepmind
- sakana-ai
- higgsfield-ai
- huggingface
- ollama
date: '2025-06-26T05:44:39.731046Z'
description: '**OpenAI** 推出了 **Deep Research API**，包含强大的 **o3-deep-research** 和 **o4-mini-deep-research**
  模型。该 API 原生支持 MCP、搜索和代码解释器，能够实现包括多智能体（multi-agent）设置在内的高级智能体功能。


  **Google** 发布了 **Gemma 3n**，这是一款专为仅有 3GB 内存的边缘设备优化的多模态模型，在 LMSys Arena 上获得了 1300
  分的高分，并采用了全新的 MatFormer 架构和广泛的生态系统集成。


  **Black Forest Labs** 推出了 **FLUX.1 Kontext [dev]**，这是一个拥有 120 亿参数的修正流变换器（rectified
  flow transformer），用于基于指令的图像编辑，其性能可与 **GPT-4o** 媲美。


  **DeepMind** 发布了 **AlphaGenome**，这是一款能够读取 100 万个 DNA 碱基以进行基因功能预测的 AI 模型，标志着 AI 生物学的突破。


  **Sakana AI** 展示了强化学习教师（Reinforcement-Learned Teachers, RLTs），旨在增强大语言模型（LLM）的推理能力，以高效的计算在
  MiniF2F 测试中达到了 86.1% 的准确率。


  **Higgsfield AI** 发布了 **Higgsfield Soul**，这是一款高审美照片模型，拥有 50 多个预设，可实现时尚级的写实效果。


  此外，**Google** 还推出了 **Gemini CLI**，这是一款用于终端的开源 AI 智能体，并提供免费的 Gemini 2.5 Pro 请求。'
id: MjAyNS0w
models:
- o3-deep-research
- o4-mini-deep-research
- gemma-3n
- flux-1-kontext-dev
- gpt-4o
- alphagenome
people:
- demishassabis
- hardmaru
- osanseviero
- clementdelangue
title: OpenAI 发布 Deep Research API (o3/o4-mini)
topics:
- multimodality
- model-releases
- agentic-ai
- reinforcement-learning
- instruction-following
- model-architecture
- model-optimization
- image-generation
- biological-ai
- multi-agent-systems
- model-integration
---

**Deep Research is all you need.**

> 2025年6月25日至6月26日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，5509 条消息）。预计节省阅读时间（按 200wpm 计算）：472 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

虽然 Google 已经[宣布](https://www.latent.space/p/aiewf-2025-keynotes)了他们发布 Deep Research API 的*意图*，但 OpenAI 似乎选择在今天通过一个相对低调的[公告](https://x.com/openaidevs/status/1938286704856863162?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)抢先发布了他们的 Deep Research API：


![](https://resend-attachments.s3.amazonaws.com/gUxdgTpl2JbsOZJ)


我们直言不讳地说——**o3-deep-research 和 o4-mini-deep-research 可能是目前世界上驱动 Agent 最强大的 LLM。** 这归功于其对 MCP、Search 和 Code Interpreter 的原生支持，这三者是 [Big 5](https://news.smol.ai/issues/25-05-27-mistral-agents) LLM OS 原语中的三个。

除了新的 webhook 模式外，你不应错过今天发布的 cookbook：

- [Introduction to Deep Research API:](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) 为你提供在约 30 行代码（30LOC）内构建自己的 Deep Research 所需的一切，并结合 MCP。
- [Deep Research API Agents:](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents) 详细介绍了 Agents SDK 的用法，并首次展示了使用 4 个 Agent 的**多智能体（multi-agent）**设置。
    
    
![](https://resend-attachments.s3.amazonaws.com/SrxbBsr13hNncpx)

    

---

# AI Twitter 综述

**模型发布与更新**

- **Google 发布 Gemma 3n**：**Google** 发布了 **Gemma 3n**，被描述为一款强大的多模态（**文本、音频、图像、视频**）AI 模型，旨在仅需 **3GB RAM** 的边缘设备上运行，并在设备端实现高性能。根据 [@osanseviero](https://twitter.com/osanseviero/status/1938374626910060782) 的说法，它是第一个在 **LMSys Arena** 上得分超过 **1300** 的 <10B 模型。该模型采用了全新的 **MatFormer** 架构，使其具有原生的灵活性。此次发布包括广泛的开放生态系统支持，合作伙伴如 [@huggingface](https://twitter.com/ClementDelangue/status/1938283910980325670)、[@ollama](https://twitter.com/ollama/status/1938324186579292415)、[@awnihannun for MLX](https://twitter.com/awnihannun/status/1938283694416077116)、[@UnslothAI](https://twitter.com/osanseviero/status/1938307534840074522) 以及 [@ggerganov for llama.cpp/GGUFs](https://twitter.com/ggerganov/status/1938284171564028214) 均在首日提供了集成。Ross Wightman 还发布了 `timm` 的更新，为 Gemma 3n 提供图像编码器，并指出其采用了 'MobileNetV5' 骨干网络 ([@wightmanr](https://twitter.com/wightmanr/status/1938311403934519807))。
- **Black Forest Labs 发布 FLUX.1 Kontext [dev]**：**Black Forest Labs** 发布了 **FLUX.1 Kontext [dev]**，这是一个拥有 **12B 参数** 的开源权重 rectified flow transformer，用于高质量、基于指令的图像编辑，定位与 **GPT-4o** 等专有模型相当。该模型现已在 **Hugging Face** 上线 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1938260818602430788))，并在 `diffusers` ([@RisingSayak](https://twitter.com/RisingSayak/status/1938267936378208655)) 和 Chipmunk ([@realDanFu](https://twitter.com/realDanFu/status/1938300379613347942)) 中获得首日支持。
- **DeepMind 发布 AlphaGenome**：[@demishassabis](https://twitter.com/demishassabis/status/1937971182256435323) 强调了 **AlphaGenome** 的发布，这是一款可以读取 **100 万个 DNA 碱基** 以预测基因功能和调节的 AI 模型。这被视为 AI 在生物学领域的重大进步。
- **Sakana AI 推出强化学习教师 (RLTs)**：[@hardmaru](https://twitter.com/hardmaru/status/1938381728902783321) 分享了 **Sakana AI** 的新技术 **RLT**，该技术利用强化学习来教导 LLM 进行复杂推理。一个显著的结果包括使用三个 7-8B 模型在较小的样本预算下在 **MiniF2F** 上达到 **86.1%**，设定了新的计算 Pareto 前沿 ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1938066184286433621))。
- **Higgsfield AI 推出 Higgsfield Soul**：发布了一款名为 **Higgsfield Soul** 的新型高审美照片模型，具有超过 **50 个精心挑选的预设**，并承诺提供时尚级的写实感，以增强用户生成内容 ([@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1937937784934912415))。

**工具、框架与基础设施**

- **Google 发布 Gemini CLI**：**Google** 发布了 **Gemini CLI**，这是一个开源 **AI agent**，可将 **Gemini** 模型直接引入终端。该工具每天提供 **1,000 次免费的 Gemini 2.5 Pro 请求** ([@googleaidevs](https://twitter.com/demishassabis/status/1938023045320335789), [@OfficialLoganK](https://twitter.com/hardmaru/status/1938068439404581370))。社区成员已将其作为 provider 集成到 **Codex CLI** 等工具中 ([@cline](https://twitter.com/cline/status/1938052438113845748))，甚至有用户“凭感觉添加了（vibe-added）” **Claude Sonnet** 和 **Opus** ([@hrishioa](https://twitter.com/hrishioa/status/1938335965845876940))。
- **“Context Engineering”的兴起**：由 [@karpathy](https://twitter.com/code_star/status/1937934052436414690) 的热门推文引发的一场重要讨论，核心是将“prompt engineering”更名为 **“context engineering”**。这个新术语更好地反映了向 **LLM** 提供广泛、高质量上下文而非仅仅是简短任务描述的实践。有人提议 **AI agent** 实际上是一个**“自动上下文工程师（automatic context engineer）”** ([@shaneguML](https://twitter.com/shaneguML/status/1938106399466369412))，而另一些人则将其视为深度学习从特征工程（feature engineering）演进的结果 ([@awnihannun](https://twitter.com/awnihannun/status/1938365325676057014))。
- **DSPy 作为“Context Engineering”工具受到关注**：**DSPy** 正在流行，特别是得到了 **Shopify CEO Tobi Lütke** 的认可，称其为他“[首选的 context engineering 工具](https://twitter.com/lateinteraction/status/1938392172245750072)”。社区中许多人都在强调其在 **prompt** 优化和构建可靠 **AI** 系统方面的实际应用 ([@stanfordnlp](https://twitter.com/stanfordnlp/status/1937944059160768793))。
- **LangChain/LlamaIndex 生态系统更新**：**LlamaIndex** 正在推动 **agent** 开发，包括一个构建事件驱动型 **Zoom 会议记录器**的教程，该工具利用 **Zoom** 新的实时流媒体功能与 **Notion** 集成 ([@jerryjliu0](https://twitter.com/jerryjliu0/status/1937998395383423130))。**LangGraph** 因其在构建长时运行、有状态应用（如用于软件开发自动化的 **Qodo Gen CLI**）中的应用而受到关注 ([@hwchase17](https://twitter.com/hwchase17/status/1938287016250380655))。
- **用于跨框架模型使用的 KerasHub**：**François Chollet** 宣布推出 **KerasHub**，允许开发者在 **JAX**、**PyTorch** 和 **TensorFlow** 中使用 **Hugging Face checkpoints**（如 **Llama**、**Gemma** 和 **Mistral** 模型）进行推理、**LoRA** 微调和大规模训练 ([@fchollet](https://twitter.com/fchollet/status/1938208330062655678))。
- **分布式训练技术**：来自 [@StasBekman](https://twitter.com/StasBekman/status/1938270423978021228) 的详细推文解释了将**激活内存卸载（activation memory offloading）**到 **CPU** 内存的方法，作为在长序列训练期间节省大量 **GPU** 内存的一种方式，从而在不增加内存使用的情况下支持更多层。
- **Modular 与 Inworld AI 合作推出文本转语音**：**Modular** 和 **Inworld AI** 合作推出了一款新的最先进的文本转语音（**text-to-speech**）模型，据报道其成本降低了 **20 倍**，使各种产品的实时语音更加普及 ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1937931869640921385))。

**公司与行业新闻**

- **Meta 从 OpenAI 苏黎世办公室挖走核心研究员**：来自 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1938077153733800075) 的一条热门推文称，**Meta** 实际上已经雇佣了 **OpenAI 苏黎世团队**的很大一部分人员，据报道，这些研究人员甚至没有等待他们在 OpenAI 的股权归属（vest）。此举被视为 Meta 为其 **Llama** 开发进行的一次重大的人才引进。[@Teknium1](https://twitter.com/Teknium1/status/1938091439440969752) 针对“OpenAI 并没有失去其‘最优秀’人才”的说法调侃了一番。
- **OpenAI 宣布 DevDay 2025**：**OpenAI** 已将其下一届开发者大会定于 **2025 年 10 月 6 日**在旧金山举行，并承诺这将是一场规模更大的盛会，届时将有超过 **1500 名开发者**参加，并提供主题演讲直播和动手实践环节 ([@OpenAI](https://twitter.com/OpenAI/status/1938277642014494980))。
- **Suno 收购 AI 驱动的 DAW 工具 WavTool**：以 AI 音乐生成闻名的 **Suno** 收购了 **WavTool**，这是一款 AI 驱动的数字音频工作站（DAW）。此举旨在让艺术家在创作流程中获得更精确的控制 ([@SunoMusic](https://twitter.com/SunoMusic/status/1938281718865399933))。
- **Anthropic 支持使用 Claude 创建应用程序**：**Anthropic** 现在允许用户创建并分享功能性的 AI 驱动应用程序，这些应用直接嵌入了 **Claude** 的智能，从而实现可分享的交互式体验 ([@alexalbert__](https://twitter.com/alexalbert__/status/1937934036590334335))。
- **Nvidia 重夺全球市值最高公司宝座**：在经历了一段波动期后，**Nvidia** 的股价已经反弹，使其再次成为全球市值最高的公司 ([@nearcyan](https://twitter.com/nearcyan/status/1938035873259655202))。

**研究、技术与评论**

- **法院裁定使用受版权保护的书籍进行训练属于合理使用**：**Andrew Ng** 对美国地方法院的一项裁决进行了详细分析，该裁决认为 **在受版权保护的书籍上训练 LLM 构成合理使用（fair use）**。法官裁定，训练过程具有转化性（transformational），并非原著的替代品。然而，裁决也指出，使用盗版材料不属于合理使用，这一点可能仍会给模型训练者带来法律责任 ([@AndrewYNg](https://twitter.com/AndrewYNg/status/1938265468986659075))。
- **学术同行评审现状**：[@jxmnop](https://twitter.com/jxmnop/status/1937949143084810625) 发布的一条广为流传的推文描述了为 **NeurIPS** 审稿的挫败经历，强调了诸如 **LLM 生成的投稿**、重复论文以及基于私有公司数据且不可复现的研究等问题。
- **关于 RLHF、正则化与 Slop**：在一条流传甚广的推文中，**Andrej Karpathy** 警告说：“[愿你的正则化项足够强大，以免你将模型 RLHF 成了 Slop（废话/垃圾内容）](https://twitter.com/karpathy/status/1937941695943065640)”，这是对在没有适当约束的情况下，利用人类反馈强化学习过度优化模型风险的精辟见解。
- **“破折号（em dash）”作为 AI 写作的特征**：**John Carmack** 提到他喜欢使用破折号，但讨厌现在它们常被视为 AI 生成文本的迹象，这一观点引起了广泛共鸣 ([@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1938278575800553665))。
- **斯坦福大学“从零开始构建语言模型”课程 (CS336)**：由 **Percy Liang** 等人教授的斯坦福课程 **CS336** 受到了 **Jeff Dean** 等业界领袖的高度赞赏，被认为是学习如何从头开始构建 LM 的优秀资源 ([@stanfordnlp](https://twitter.com/stanfordnlp/status/1937944419090764222))。

**更广泛的影响**

- **AI 作为核心系统属性**：Perplexity CEO **Aravind Srinivas** 认为，为了让产品在 AGI 到来时保持竞争力，智能必须是系统的核心属性，而不是“零星散布的组件”。他将 **浏览器** 视为满足这一属性的关键平台 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1938116239576199365))。
- **地缘政治与国家安全评论**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1937990538302796147) 分享了一篇讨论将 RL 用于网络安全的文章，并认同在由 LLM 驱动的新均衡状态下，防御者可能会占据优势。他还经常就美国政策思维与现代中国军事学说现实之间的差距发表评论 ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1938066712928161815))。
- **AI 对美国电网的影响**：来自 [@dylan522p](https://twitter.com/dylan522p/status/1937943241082437697) 的警告指出，由于美国电网的不稳定性，大规模 AI 训练运行可能会导致数十万人遭遇停电，这可能会引发公众对 AI 基础设施的反感情绪。
- **美国非移民签证隐私政策更新**：在一项非 AI 相关但对科技界影响巨大的公告中，美国驻英国领事馆宣布，所有 **F、M 或 J 类非移民签证** 的申请人现在都被要求在签证办理期间将社交媒体隐私设置调整为“公开” ([@francoisfleuret](https://twitter.com/francoisfleuret/status/1937926540769054772))。

**幽默与梗图**

- **最好的登录界面**：[@vikhyatk](https://twitter.com/vikhyatk/status/1938092308358172880) 开玩笑说：“**指标显示我们拥有行业内最好的登录界面……用户在登录前平均在该页面停留 30 秒**”。
- **会议观察**：[@dylan522p](https://twitter.com/dylan522p/status/1938334440595366035) 发帖称：“>在会议上 >有人说 LLM 无法泛化，过度炒作 >看一眼他的胸卡 >他在 **IBM** 工作。每次都这样”。
- **AI 初创公司的现状**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1938278592175345802) 用一张凌乱的书桌照片完美捕捉了当前的开发氛围，配文是：“**啥都不好使，但氛围感拉满**。”
- **计算债之歌**：[@MillionInt](https://twitter.com/MillionInt/status/1938018248915873883) 唱出了数据科学家的心声：“**你跑了一万个节点，得到了什么？又一个不眠之夜和一身计算债**”。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Gemma 3n 模型发布与社区工具链

- [**Gemma 3n 已在 Hugging Face 发布**](https://www.reddit.com/r/LocalLLaMA/comments/1ll429p/gemma_3n_has_been_released_on_huggingface/) ([得分: 243, 评论: 70](https://www.reddit.com/r/LocalLLaMA/comments/1ll429p/gemma_3n_has_been_released_on_huggingface/)): **Google 已在 Hugging Face 上发布了 Gemma 3n 系列模型，包括基础版 (E2B, E4B) 和指令微调版 (-it)，并附带了涵盖 HellaSwag、MMLU 和 LiveCodeBench 等数据集的详细基准测试（参见：[E2B](https://huggingface.co/google/gemma-3n-E2B), [E4B](https://huggingface.co/google/gemma-3n-E4B)）。llama.cpp 的支持已通过 [ngxson 的 PR](https://github.com/ggml-org/llama.cpp/pull/14400) 落地，GGUF 量化模型也可用于本地推理。[官方技术公告](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) 提供了关于模型架构和评估的更深层背景。** 一些专家讨论询问 Gemma 3n 在技术上与 Qwen3 相比如何；该对比分析在帖子中仍处于开放讨论状态。
    - 有用户要求对 Gemma 3n 和 Qwen3 进行直接对比，表示对基准测试、推理速度以及语言或任务性能感兴趣。这突显了社区对主流开源 LLM 进行经验性评估的需求。
    - 社区对 Gemma 3n 在 Android 上的表现充满期待，这表明人们关注与新版本相关的移动端优化、效率、量化或边缘部署因素。

- [**Gemma 3n 全面发布 - 开发者版**](https://www.reddit.com/r/LocalLLaMA/comments/1ll68iz/gemma_3n_full_launch_developers_edition/) ([Score: 117, Comments: 4](https://www.reddit.com/r/LocalLLaMA/comments/1ll68iz/gemma_3n_full_launch_developers_edition/)): **Google 已全面发布 Gemma 3n，这是一款支持音频、视频、图像和文本输入并输出文本的多模态模型。关键创新包括参数高效变体（"E2B" 和 "E4B"）、能够以低至** `2B/4B` **参数运行、允许提取子模型并进行混合匹配部署的 MatFormer 架构、与 MobileNetV5 的集成以及全新的音频编码器，此外还具备跨平台的广泛兼容性（Hugging Face, llama.cpp, Ollama, MLX 等）。详情请参阅官方 [开发者指南](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) 和 [Hugging Face 集合](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)。** 评论区呼吁尽快增加 GGUF 格式的多模态（音频+视觉）支持和微调兼容性。技术层面上，人们对将多模态编码器与更大的 Gemma 模型（如 27B）结合使用表现出好奇，并询问是否会发布 JAX 实现版本。
    - 一位用户询问了 GGUF 格式支持音频和视觉模态的时间表，这将使本地和高效环境中的多模态应用更加广泛。这表明开发者有兴趣将当前的纯文本实现扩展到处理其他数据类型，符合多模态 LLM 的发展趋势。
    - 有人询问 JAX 实现的发布情况，这对于支持基于 TPU 的训练/推理以及轻松利用 Google Cloud 上的高性能计算具有重要意义。这将增强可扩展性，并促进除传统 PyTorch 流水线之外的研究应用。
    - 一条侧重技术的评论询问 Gemma3 27B 是否可以通过音频和视频编码的 Token 进行扩展以实现多模态，这表明人们有兴趣将非文本 Embedding 投影到 LLM 中以实现更丰富的输入处理——类似于 LLaVA 或 Flamingo 等模型中的现有工作。
- [**Google 的 CLI 确实会使用你的 Prompt 数据**](https://i.redd.it/j1km6ff1h69f1.png) ([Score: 300, Comments: 88](https://www.reddit.com/r/LocalLLaMA/comments/1lko09j/googles_cli_does_use_your_prompting_data/)): **图片展示了 Google 针对“个人版 Gemini Code Assist”的隐私声明，强调默认情况下，Google 会收集 Prompt、用户代码、生成的输出以及其他交互数据，以改进其服务和机器学习能力。此数据收集适用于免费个人计划——标准版或企业版计划的用户不受此限，并且提供明确的退出机制。对于使用免费版 Gemini Code Assist 处理专有或敏感代码的开发者来说，其隐私影响至关重要。** 一位评论者澄清说，数据收集仅适用于免费层级；用户可以选择退出，而付费客户（标准/企业计划）不受此约束。除隐私/实践影响外，没有进一步的技术争论。
    - 澄清指出，虽然 Google 针对个人的 Code Assist CLI（免费计划）确实使用用户 Prompt 数据，但标准和企业计划明确不使用这些数据，解决了付费客户的隐私担忧。此外，用户拥有可见的退出选项，在数据使用实践上提供了一定的选择权。

### 2. 最新开放权重与推理模型发布

- [**FLUX.1 Kontext [dev] - 一个具备商业级图像编辑性能的权重开放模型。**](https://www.reddit.com/r/LocalLLaMA/comments/1ll38zu/flux1_kontext_dev_an_open_weights_model_for/) ([Score: 249, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1ll38zu/flux1_kontext_dev_an_open_weights_model_for/)): **Black Forest Labs 发布了开放权重模型 FLUX.1 Kontext [dev]，旨在提供可与闭源方案媲美的图像编辑性能。模型权重已在 Hugging Face 上发布，并在 Twitter/X 上发布了发布公告。该模型旨在通过开放此前封闭的标准，促进先进的本地图像编辑工作流。** 社区反应充满了惊讶和对真正开源的热情。有人提出了关于自托管该模型硬件要求的技术问题，表明了对部署可行性（如计算资源、VRAM、推理栈）的关注。
    - 存在关于 FLUX.1 Kontext 规模的讨论，用户注意到其 `12B 参数` 的大小，并询问它是否是迄今为止发布的最大的开放权重图像编辑模型，将其与同领域的现有模型进行了隐性对比。

- 围绕 Self-hosting FLUX.1 Kontext 的需求表现出了技术好奇心，利益相关者正在寻求有关硬件、存储需求和实际集成步骤的详细信息，强调了对实际部署指导的需求。
- 存在关于格式支持的问题，特别是对 GGUF（一种针对 CPU/低配硬件上高效推理进行优化的格式）版本的请求，这表明了对更广泛部署场景和标准服务器基础设施之外的可访问用例的高度兴趣。
- [**RpR-v4 推理模型全系列。Small-8B, Fast-30B-A3B, OG-32B, Large-70B。**](https://huggingface.co/ArliAI/DS-R1-Distill-70B-ArliAI-RpR-v4-Large) ([Score: 108, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1lkifu8/full_range_of_rprv4_reasoning_models_small8b/)): **该公告展示了 ArliAI 的 RpR-v4 模型全系列：Small-8B、Fast-30B-A3B、OG-32B 和 Large-70B。其中 70B 变体 [DS-R1-Distill-70B-ArliAI-RpR-v4-Large](https://huggingface.co/ArliAI/DS-R1-Distill-70B-ArliAI-RpR-v4-Large) 是一个基于 Llama 衍生的 70B 参数模型，使用 RS-QLORA（rank 64, alpha 64, LR 1e-5）在重推理、无模板的聊天数据集上进行了微调（训练序列长度高达 16K，上下文为 32K）。v4 的关键改进包括先进的重复/模仿过滤、使用精确 Tokenization（例如 `<think>...</think>`）界定的创造性推理块，以及使用高 Temperature 和 top-k（无重复惩罚）的采样策略，支持 BF16 和 GGUF 推理格式。** 评论强调了对 A3B 变体可用性的赞赏，对 GGUF 格式模型的请求，以及指出官方 Hugging Face 模型卡文档中模型大小拼写错误（将 70B 误列为 8B）的建设性反馈。
    - 一位用户注意到了 70B 模型 README 中的文档错误，指出其错误地描述为“80 亿参数模型”，而未能准确反映其 70B 参数量，这对于技术清晰度和准确的 Benchmarking 预期至关重要。
    - 开发者解释说，在收到对使用 RpR 数据集微调的 OG 32B 模型（基于 QwQ）的积极反馈后，他们扩展了该方法，为更广泛的模型尺寸（Small-8B, Fast-30B-A3B, OG-32B, Large-70B）创建了微调版本，以响应托管和社区用例，并旨在实现更广泛的可访问性。
- [**开源实时 3D 操纵器（少数派报告风格）**](https://v.redd.it/b03bkt6a859f1) ([Score: 128, Comments: 10](https://www.reddit.com/r/LocalLLaMA/comments/1lkijb5/opensource_realtime_3d_manipulator_minority/)): **该帖子介绍了一个“少数派报告”风格的开源 3D 操纵器的 Hugging Face 演示：[3d-model-playground](https://huggingface.co/spaces/stereoDrift/3d-model-playground)。技术实现似乎涉及基于手势的 3D 模型操作，以及由手势和 TTS 触发的一些菜单动画，可能使用了基于网络摄像头的动作追踪（未详细说明具体的 SDK/框架）。** 评论中的技术讨论质疑了其实际意义和实现方式：一位用户不确定该项目是否真正开源，另一位用户批评了这种方法，建议像 Meta 基于 EMG 的腕带（[参考](https://www.uploadvr.com/zuckerberg-neural-wristband-will-ship-in-the-next-few-years/)）这样的专用可穿戴设备可能比基于摄像头的动作追踪更准确、更实用。
    - 讨论质疑了该 3D 操纵器的技术新颖性和实用性，一位用户询问核心功能是否只是一个弹出菜单，带有手势控制的动画和 TTS。他们建议使用追踪手部和手指运动的臂带（如 Meta 正在开发的）作为一种更准确、独立于摄像头的替代方案，并引用了 Meta 即将推出的神经腕带技术 [来源](https://www.uploadvr.com/zuckerberg-neural-wristband-will-ship-in-the-next-few-years/)。
    - 讨论指出，项目创建者此前曾在 Hacker News 上分享过一些技术实现细节，并提供了一个技术细节更丰富的文章链接：https://xcancel.com/measure_plan。该资源可能提供比发布视频更深入的系统架构和方法论见解。

### 3. DeepSeek R2 发布延迟与市场限制

- [**DeepSeek R2 延迟发布**](https://i.redd.it/718m48of6b9f1.jpeg) ([Score: 352, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1ll6jo5/deepseek_r2_delayed/))：**该图片是一个设置在技术环境中的新闻风格叠加图，直观地强化了帖子的内容，即 DeepSeek R2 的公开发布已推迟。技术延迟源于 CEO Liang 对更高性能标准的坚持，以及近期美国出口管制导致的 Nvidia 服务器芯片严重供应短缺。这些限制已经使中国云服务厂商只能使用 Nvidia 的 H20 芯片来运行 DeepSeek R1，而即使是这些芯片也在 2024 年 4 月被禁，这引发了人们的担忧，即 R2 的潜在需求可能会压垮无法合法采购合适硬件的基础设施。更多细节可在帖子链接的《The Information》和路透社文章中找到。** 评论表达了社区对 DeepSeek 花费所需时间完善 R2 的强烈支持，将延迟视为质量保证的正向信号而非挫折。基于前代模型 R1-0528 设下的高标准，评论中流露出一种乐观情绪。
    - 一位用户指出，关于 DeepSeek R2 模型的传闻（引用自 2025 年 2 月的一篇路透社文章）可能具有投机性，并非基于官方信息，并批评这些报道缺乏对新基座模型 V4 的讨论。他们还强调，出口管制问题经常被推测但未必得到证实，质疑主流媒体对内部 AI 模型时间线报道的可信度。
- [**GPU 透传到虚拟机的真实性能损耗（其实……挺无聊的）**](https://www.reddit.com/gallery/1lkzynl) ([Score: 156, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1lkzynl/the_real_performance_penalty_of_gpu_passthrough/))：**一位用户在 AMD RX 9060 XT 16GB 上对比了裸机与虚拟机（Ubuntu 24.04，AI Linux 宿主机）的 GPU 透传性能（使用** `vfio-pci`**），用于 LLM 推理（模型：mistral:7b, gemma2:9b, phi4:14b, deepseek-r1:14b）。结果显示，虚拟机中的推理性能损耗仅为** `1–2%`**。完整的配置、ROCm 安装说明和基准测试结果都在[这个 README](https://github.com/sbnb-io/sbnb/blob/main/README-GPU-PASSTHROUGH-BENCHMARK.md)中。** 评论者指出，极小的损耗符合直接设备透传的预期，但强调了实际考虑因素：虚拟机需要在不同操作系统间划分 RAM，且文件系统透传（VIRTFS）可能成为模型加载的瓶颈（特别是使用 mmap 技术时），建议使用磁盘镜像以获得最大带宽。
    - 提到的一个关键技术注意事项是，使用 GPU 透传时，RAM 会在宿主机和客户机操作系统之间分配，从而减少了每个系统可用的内存。为了在模型加载期间实现高效的磁盘访问（特别是使用 llama.cpp 的 mmap 时），与使用完整磁盘镜像相比，文件系统透传（例如 QEMU VIRTFS）的带宽较低，会导致模型加载变慢并可能引起运行时的减速。
    - 一些用户认为 VFIO 透传的开销本质上应该是零，并对启用所有 VFIO 功能后仍存在不可忽略的性能损耗表示担忧，尽管对于大多数本地消费级用途来说，这并不是主要问题。这突显了在特定的高性能场景中可能很重要的细微性能成本。
    - 讨论的一种替代方法是 LXC 透传，它允许在多个容器而不是完整的虚拟机之间共享 GPU，从而实现更灵活、更细粒度的资源利用，这对于多项目或多租户工作流非常有益。
- [**Meta 赢得 AI 版权诉讼，美国法官判决作者败诉 | Meta**](https://www.theguardian.com/technology/2025/jun/26/meta-wins-ai-copyright-lawsuit-as-us-judge-rules-against-authors) ([Score: 244, Comments: 113](https://www.reddit.com/r/LocalLLaMA/comments/1lkz0hg/meta_wins_ai_copyright_lawsuit_as_us_judge_rules/))：**一名美国法官驳回了针对 Meta 的版权诉讼，裁定使用受版权保护的作品作为 AI 训练数据本身并不构成版权侵权。这一裁决与之前的裁决（例如 Zarya of the Dawn 诉 LAION 案）一致，即区分了模型训练与受版权保护材料的直接复制或分发。** 评论强调，驳回主要是由于作者的案件陈述不力，而非详尽的法律先例，且该裁决是邀请原告重新表述其主张，而非给予 Meta 无条件的胜利。
    - 一位用户指出，法院的裁决很大程度上基于作者未能提供充分证据证明 Meta 的 AI 模型稀释或影响了其作品的市场价值——这是美国版权法下判定侵权的一个关键因素。这确立了一个法律先例，即证明市场损害对于此类涉及 AI 训练的版权案件至关重要。

- 另一个技术视角强调，法院区分了使用受版权保护的材料训练 LLM 与直接侵犯版权的行为，其含义是“训练 ≠ 侵犯版权”。这种区分对于未来的 AI 数据使用以及围绕模型训练的法律解释具有重要意义。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/aivideo, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. 头部 AI 公司领导层变动与开源模型炒作

- [**Meta 挖走 3 名 OpenAI 首席研究员**](https://i.redd.it/5bl9vpbn079f1.jpeg) ([Score: 655, Comments: 194](https://www.reddit.com/r/singularity/comments/1lkq5r2/meta_snags_3_open_ai_lead_researchers/)): **图片显示了一则新闻片段，关于 Meta 为其超级智能项目招募了三名来自 OpenAI 的著名首席研究员——Lucas Beyer、Alexander Kolesnikov 和 Xiaohua Zhai，他们常驻苏黎世。鉴于 OpenAI 最近刚在苏黎世设立办公室，这一举动显得尤为突出，并凸显了 Meta 持续激进的人才获取策略，旨在增强其 AI 研究实力。** 评论讨论了这些高调人才变动中可能涉及的巨额财务激励，但对于 Meta 能否有效利用这些新人才仍持怀疑态度，一位评论者引用了对 Llama 4 的失望作为警示。
    - 几条评论讨论了 Meta 从 OpenAI 吸引人才在技术和组织层面的影响，其中一位指出这三名研究员在曾在 Google Brain/DeepMind 任职后，最近才加入 OpenAI，暗示考虑到他们任职时间较短，“首席（lead）”一词可能具有误导性。人们对 Meta 将高调聘用转化为具有技术竞争力的产品的能力表示怀疑，并以对 Llama 4 性能的失望作为基准。
    - 一位用户对 Sam Altman 针对 Meta 据称开出的 1 亿美元报价所发表的言论进行了细致分析，认为 Altman 可能在利用公开评论来设定 OpenAI 的内部预期——暗示接受 Meta 较低的报价可能被视为“被压价”，或者那些接受报价的人并不属于组织的顶尖技术人才。
    - 有人提出了对 GenAI 社会影响的担忧，认为如果通用人工智能（GenAI）得以实现，可能会产生改变文明的影响——间接提到了研究领导力的至关重要性，以及技术对齐（alignment）和开发发生在哪里。
- [**OpenAI 员工正在炒作他们即将推出的开源模型**](https://www.reddit.com/gallery/1lkl88d) ([Score: 467, Comments: 178](https://www.reddit.com/r/OpenAI/comments/1lkl88d/openai_employees_are_hyping_up_their_upcoming/)): **据称 OpenAI 员工正在推广一款即将推出的开源模型，这引发了人们对 OpenAI 是否会发布一款与其专有产品相比具有竞争力的开源（OS）模型的怀疑。一些讨论集中在术语上，指出“OS”具有歧义（通常指“Operating System”），而“OSS”才是“Open Source Software”的标准缩写。此外，有人推测，一款小型、高效的开源模型可能是为了在传闻中与 Jony Ive 共同设计的硬件（HER 设备）上本地使用，这需要快速、轻量级的推理（inference）能力，以实现实时的视听上下文理解。** 评论中的怀疑者质疑 OpenAI 的动机，以及发布一款可能威胁其闭源模型的模型的可能性，而技术讨论则围绕命名规范以及紧凑型端侧模型的部署可行性展开。
    - 一位评论者推测 OpenAI 可能会设计一款非常紧凑的开源（OS）模型，专门用于预期的 HER 设备（据报道是与 Jony Ive 共同设计的），并强调了技术约束：该模型需要本地运行、保护隐私、速度极快，并能够进行实时视听处理以提供日常辅助。这将需要显著的模型压缩和优化。
    - 另一个被提出的技术担忧是所谓旧模型的“削弱（nerfing）”：随着新模型的发布，早期版本（如 OpenAI 的 O3）的功能可能会被有意限制或性能封顶，这可能是为了引导用户采用最新的、功能更强大的产品。

- [**Meta 支付 1 亿美元高薪聘请的技能是什么？**](https://www.reddit.com/r/singularity/comments/1ll3kip/what_are_the_skills_meta_pays_100m_for/) ([Score: 111, Comments: 81](https://www.reddit.com/r/singularity/comments/1ll3kip/what_are_the_skills_meta_pays_100m_for/)): **据报道，Meta 为极具影响力的 AI 研究员和领导者提供 1 亿美元级别的薪酬，这并非针对特定的技术知识，而是看中他们吸引顶尖人才和加速组织进展的能力。这些人士通常不直接编写代码，但拥有深厚的运营专长、关键的项目经历（例如 ChatGPT 的基础贡献者或其他类似突破），以及通过开创性探索和决策周期获得的非公开领域知识。热门评论强调，他们的主要价值在于组织杠杆和准入权限，类似于曼哈顿计划的原始科学家，或者是“大型树状结构中的节点”，他们的加入会改变竞争格局和速度。** 一些评论认为，其价值不在于原始知识或技能，而在于作为商业机密和独特、原创项目经验的渠道，如果没有直接接触过执行这些开创性工作的人，这些经验是无法复制的。
    - 多位评论者指出，Meta 的 1 亿美元薪酬与其说是为了编程技能，不如说是为了排他的、难以转移的知识：特别是来自 OpenAI 等组织基础开发工作的直接洞察，包括原始模型训练（如 ChatGPT）的经验性诀窍，以及实现大语言模型（LLM）能力突破的迭代问题解决和创新。
    - 这一级别的技术领导力涉及指导可能产生数十亿美元收入的高影响项目，将最前沿的研究（通常由发表过论文的 PhDs 完成）与业务优先级相结合。一手经验的稀缺性和竞争价值，特别是关于模型架构和未公开训练细节的经验，使得薪酬远超 Meta 已公布的最高技术薪资（通常上限在 100 万美元左右）。
    - 一个普遍的主题是 Meta 正在用金钱换取“上市时间”：这些高价值雇员通过消除试错和不确定性，极大地加速了内部研究工作。获取内部人员关于 OpenAI 工作流、扩展策略（scaling strategies）和经验教训的特定知识，被视为在尖端 AI 开发中超越竞争对手的独特价值。
- [**Sam 不同意 Dario Amodei 关于“一半的入门级白领工作将在 1 到 5 年内消失”的言论，Brad 补充道“我们没有证据表明这一点”**](https://v.redd.it/q2pl20g0399f1) ([Score: 401, Comments: 378](https://www.reddit.com/r/singularity/comments/1lkwxp3/sam_doesnt_agree_with_dario_amodeis_remark_that/)): **Sam Altman 公开反对 Dario Amodei 关于 AI 将在 1-5 年内消除一半入门级白领工作的说法，称“我们没有证据表明这一点”，随后澄清说“目前没有证据”。讨论集中在尽管存在持续的技术裁员和投机性预测（如比尔·盖茨和奥巴马关于 AI 驱动的劳动力中断的公开声明，例如提到全民基本收入 UBI），但缺乏经验性的短期流失证据。** 评论者讨论了 AI 领导者为了避免抵制而公开淡化颠覆性影响，与在融资路演中提出的激进主张之间的紧张关系；人们对如何调和缺乏劳动力流失证据与持续的 AI 炒作及技术裁员之间的矛盾表示怀疑，认为这存在不一致或战略性遗漏。
    - 讨论的核心是缺乏支持 Dario Amodei 主张的具体统计证据。Brad 的声明“我们没有证据表明这一点”突显了缺乏经验数据或同行评审的研究来证明由于 OpenAI 的 GPT-4 或预期的后继模型而导致的即将发生的大规模流失。
    - 针对当前 AI 产品的实际能力和采用率，存在技术上的怀疑。一位评论者指出其中的矛盾：如果 Sam Altman 的产品确实是革命性的（即足以实现大规模工作自动化），那么应该能观察到市场影响，例如直接归因于大规模生成式 AI 实施的具体裁员，而目前这方面还没有广泛的记录。

- 提到了公众人物（Bill Gates, Obama）对 AI 驱动的失业和全民基本收入（UBI）做出的激进预测，这与 OpenAI 领导层更为谨慎的表态形成了对比。评论者对 AI 发展的炒作与缺乏行业验证的基准测试或已发表的大规模经济影响研究之间的差异表示担忧，而这些研究本应为这种颠覆性的预测提供依据。

### 2. Higgsfield Soul 与 Flux 模型：超写实 AI 图像生成

- [**AI 生成正变得疯狂地真实**](https://v.redd.it/comx1xhmza9f1) ([Score: 773, Comments: 232](https://www.reddit.com/r/singularity/comments/1ll5k3d/ai_generations_are_getting_insanely_realistic/)): **Higgsfield AI 发布了一项名为 'Soul' 的新功能，能够生成超写实的图像和视频，其效果与使用传统相机或智能手机拍摄的素材非常相似。帖子强调用户使用 ChatGPT 优化了 Prompt，但未提供定量基准测试或技术实现细节（如模型架构、数据集或硬件要求）。** 评论者指出，尽管视觉真实感进步神速，但 AI 生成内容仍存在一些持续的技术破绽，包括过度平滑的纹理、光晕效果，尤其是视频中不自然的“慢动作”效果。
    - 一位用户指出，尽管 AI 生成媒体的真实感进步很快，但技术伪影依然存在：如过度模糊/平滑的视觉效果、光晕效应以及不一致性等明显迹象，训练有素的观察者仍能识别出 AI 输出。他们注意到，与大约一年前相比，现在的进展非常显著，当时的模型输出远没有现在真实，且人工痕迹更明显。
    - 讨论的另一个技术指标是 AI 生成视频中特有的“不自然慢动作”外观；这通常是揭示合成内容的可检测缺陷，表明目前的模型在时间一致性和自然运动插值方面仍面临挑战。
- [**AI 生成正变得疯狂地真实**](https://v.redd.it/tbdhuiskxa9f1) ([Score: 886, Comments: 288](https://www.reddit.com/r/ChatGPT/comments/1ll5f75/ai_generations_are_getting_insanely_realistic/)): **用户报告了对 Higgsfield AI 新功能 'Soul' 的实测，该功能利用 ChatGPT 优化的 Prompt 产生极具照片写实感的图像和视频。测试反馈显示视觉真实感有所提升，但物理上的不一致性依然存在（例如“漂浮”的主体）以及僵硬/不自然的身体动作，特别是在动态视频场景中——用户注意到，在模拟低画质相机或较少动作时，表现最具说服力。** 评论识别了当前最先进视频生成模型的失败模式：物理真实感（特别是重力/物理学）和自然身体动作方面的明显问题。讨论暗示，尽管存在这些伪影，真实感正迅速接近可能在不久的将来挑战媒体真实性验证的水平。
    - 评论者指出，尽管真实感令人印象深刻， AI 生成的视频仍表现出明显的技术问题，如不准确的物理特性（例如主体看起来像在漂浮）以及不自然僵硬或强迫的身体动作。这些伪影揭示了底层运动模型和物理场景理解方面的局限性。
    - 多位用户注意到，降低视频质量或减少主体动作可以掩盖这些缺陷，使输出看起来更具说服力。这突显了分辨率限制和运动复杂度在当前视频生成模型有效性中的关键作用。
- [**又一次对真实感的尝试（7 张图片）**](https://www.reddit.com/gallery/1ll3yat) ([Score: 225, Comments: 52](https://www.reddit.com/r/StableDiffusion/comments/1ll3yat/yet_another_attempt_at_realism_7_images/)): **原贴作者展示了一个用于高度写实、业余摄影风格的自定义模型 v16，声称在与当前领先模型（Amateur Photography v6）进行基准测试后，其表现超越了之前的版本。技术增强包括完整的 Workflow 重构：使用 'euler_ancestral + beta' 采样器，每张样本** `50 steps` **（在初始 1024px 和 1.5 倍 Latent Upscaling 阶段均适用），** `0.4 denoising` **，并设置** `FLUX guidance=2.5` **。该模型和 Workflow 已在 [CivitAI](https://civitai.com/models/970862) 发布，帖子中链接并证实了关键技术细节。** 评论者压倒性地认同该模型的真实感，特别赞扬了其栩栩如生的质量，并对 Workflow 的改进表示认可，认为这些变化带来了切实的质量提升。

- 一位用户推测，这些高度逼真图像背后的技术可能已经在内部存在了几年，直到现在才向公众发布，且可能是在旧一代模型上运行的。这引发了关于模型性能与发布周期，以及 AI 图像合成技术有计划分阶段推出的技术问题。
- [**Flux Kontext Dev 非常出色。完全在 ComfyUI 上本地生成。**](https://i.redd.it/g5bmx9hsr99f1.png) ([Score: 521, Comments: 190](https://www.reddit.com/r/StableDiffusion/comments/1ll38bu/flux_kontext_dev_is_pretty_good_generated/)): **该帖子展示了使用 Flux Kontext Dev 模型在 ComfyUI（一个用于生成模型的开源模块化 UI）上本地生成的连环画图像。链接的工作流提供了一个分步的 ComfyUI 图表设置，用于生成类似的输出（参见 [示例工作流](https://comfyanonymous.github.io/ComfyUI_examples/flux/)）。在评论中，用户在 HuggingFace 上分享了 Flux Kontext Dev 模型的量化 GGUF 版本，讨论了模型变体（如 fp8_scaled），并指出在工作流中集成拼接角色和合并图像的便捷性。技术重点在于模型的多功能性、在本地工作流中的易用性，以及不同量化版本的资源可用性：[GGUF 模型](https://huggingface.co/bullerwins/FLUX.1-Kontext-dev-GGUF) 和 [ComfyUI 变体](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI)。** 评论者赞扬了 Flux Kontext Dev 模型的灵活性以及与 ComfyUI 的兼容性，特别是在角色拼接和多图像工作流方面。用户们密切关注特定模型量化版本的可用性和上传状态，并对此进行了简短讨论。
    - 评论者注意到 FLUX.1-Kontext-dev 模型的 GGUF 量化版本已可用于本地生成和推理，并引用了该模型的 HuggingFace 页面 (https://huggingface.co/bullerwins/FLUX.1-Kontext-dev-GGUF)。这种量化格式支持多种硬件配置，并能在 ComfyUI 等工具中实现高效部署。
    - 技术讨论包括 fp8_scaled 模型变体的状态，用户正在追踪其上传情况并确认其已存在于 HuggingFace (https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI)。fp8_scaled 版本因其在性能提升或与特定推理流水线兼容性方面的潜力而备受关注。
    - 一位用户演示了图像生成工作流中的高级用法，例如通过合并图像来拼接角色，突显了该模型在 ComfyUI 中的灵活性和功能完整性。这表明该模型对组合式和多图像操作提供了强大的支持，这对于处理复杂生成任务的技术用户非常有价值。

### 3. Anthropic 的 Jack Clark 与 AI 监管讨论

- [**Anthropic 联合创始人 Jack Clark 请求加强安全监管，并告知国会：“极具变革性的 AI”将在 18 个月内（即 2026 年底）到来**](https://v.redd.it/4knryyx1kb9f1) ([Score: 127, Comments: 44](https://www.reddit.com/r/singularity/comments/1ll8lyv/anthropic_co_founder_jack_clark_asks_for_more/)): **Anthropic 联合创始人 Jack Clark 在国会作证，敦促加强监管，并警告称“极具变革性的 AI”预计将在 18 个月内（即 2026 年底）出现。Clark 的言论表明 AI 能力正在飞速提升，鉴于这些系统预计将产生重大的社会影响，可能需要采取先发制人的政策行动。** 热门评论在技术层面缺乏实质内容，未涉及 Clark 主张中的监管、基准测试或技术细节。
    - 首要的技术主题是对 AI 监管干预的时效性和有效性持怀疑态度。一条评论提出了务实但悲观的立场：虽然人们一致认为针对 AI 安全采取监管行动是合理的——特别是考虑到“极具变革性的 AI”将在 18 个月内到来的说法——但普遍认为此类监管将无法及时出台以影响结果，因为“现在已经太晚了”。
- [**Anthropic 的 Jack Clark 在国会作证：“你不会希望一个 AI 系统试图勒索你以设计其后继者，所以你必须致力于安全，否则你将输掉这场竞赛。”**](https://v.redd.it/8vc2m49gma9f1) ([Score: 101, Comments: 58](https://www.reddit.com/r/ClaudeAI/comments/1ll3nhd/anthropics_jack_clark_testifying_in_front_of/)): **Anthropic 的 Jack Clark 在国会作证，强调**稳健的 AI 安全实践对于防止不良的自主行为至关重要**，例如 AI 尝试社交操纵（如勒索）或在缺乏监督的情况下进行递归自我改进。该声明将 AI Alignment（对齐）和控制视为安全技术进步以及避免在国际竞争中遭遇战略损失的关键，反映了 [Anthropic 的安全原则](https://www.anthropic.com/safety)。** 评论者辩论了 AI 开发的科学严谨性，不同意基于恐惧的 AGI 辞令，并对比了中美之间的监管环境，对美国的政治意愿和国际 AI 进步的“红线”框架表示怀疑。
    - 一位评论者批评了将尖端 AI 开发视为“炼金术”的观点，强调 AI 模型本质上是统计性的，而非神秘的。他们还认为对 AGI 的恐惧很大程度上是由炒作驱动的，并主张现实世界的治理差异（例如中国的中央控制与美国的资本影响）将决定各国如何实施 AI 护栏和安全协议，暗示监管方法将根据政府结构产生实质性差异。
    - 一条评论指出了一种态度上的矛盾：尽管据报道 Google 的 CEO 倾向于“末日论”（担忧 AI 的存在性风险），但该公司仍积极游说反对 AI 监管。这突显了顶级 AI 领导者口头表达的安全担忧与其公司对监管框架的实际参与之间的紧张关系，引发了对监管俘获（regulatory capture）或行业驱动炒作的担忧。

---

# AI Discord 简报

> 由 chatgpt-4o-latest 生成的摘要之摘要的摘要
> 

**1. OpenRouter 的融资与工具扩展**

- **OpenRouter 获得 4000 万美元融资，引发生态系统热潮**：**OpenRouter** 宣布成功完成 **4000 万美元融资**，据 [Deedy 的推文](https://x.com/deedydas/status/1937902948920811729)透露，其估值约为 **5 亿美元**，**Emad Mostaque** 和 **Yuchen Jin** 对此表示祝贺。该平台目前**每年路由超过 100T tokens**，通过单一 API 即可调用超过 **400 个模型**。
    - 此次融资引发了关于**前端身份验证提供商 Clerk** 的讨论，Clerk 在同一天发生故障，导致尽管 API 正常运行但仍出现间歇性 **401 错误**，这促使[用户考虑迁移](https://x.com/search?q=clerkdev&src=typed_query&f=live)。OpenRouter 还推出了 **Presets**，这是一种路由规则的配置抽象，详见其[文档](https://openrouter.ai/docs/features/presets)。
- **OpenRouter 的 Presets 简化了 LLM 工作流**：**Presets** 的推出让用户可以通过 OpenRouter 控制面板管理 **LLM 配置**、System Prompts 和路由规则（[文档](https://openrouter.ai/docs/features/presets)）。用户现在可以使用类似 `"model": "@preset/your-preset-slug"` 的语法引用配置，减少了迭代开销。
    - 这受到了旨在构建更模块化 LLM 流水的开发者的赞赏，同时也引发了关于欧盟公司 **GDPR 合规路由**的讨论。一位欧盟创始人讽刺地评论道：*“这就是在这里当创始人的生活。”*

**2. DSPy 的 Ruby 移植与语言扩展**

- **DSPy 通过 Desiru 实现 Ruby 化**：开发者 **@obie** 发布了 [Desiru](https://github.com/obie/desiru)，这是 **DSPy 的 Ruby 实现**，增加了诸如**基于 Postgres 的持久层**（[持久化示例](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb)）和**异步后台处理**（[异步示例](https://github.com/obie/desiru/blob/main/examples/async_processing.rb)）等功能。
    - **Shopify 的 CEO** 在一则[推文](https://x.com/tobi/status/1937967281599898005)中表达了兴奋之情，暗示基于 Ruby 的 DSPy 可能会主导 **Shopify、GitHub 和 Coinbase** 等生态系统。社区讨论了移植版本的命名规范：*DS<语言文件扩展名>*（例如 Ruby 为 DSRB，Rust 为 DSRS）。
- **Desiru 通过 Postgres 和异步功能超越 DSPy**：**Desiru** 的独特之处在于将示例保存到 **Postgres** 并集成**异步后台任务**，这与 DSPy 的极简风格有所区别。这两个功能的文档都可以在 [GitHub 仓库](https://github.com/obie/desiru)中找到。
    - 社区成员提议建立一个类似 **LangChain Community** 的注册表，以托管基于 Desiru 的扩展和连接器。尽管 DSPy 的维护者更倾向于保持简单，但围绕 Desiru 不断增长的生态系统表明，其正朝着企业级集成就绪的方向转变。

**3. AI 生成的 GPU 编程与 Mirage 发布**

- **Mirage 项目自动生成 GPU Kernel**：[此 Sandbox 仓库](https://github.com/tcapelle/triton_eval/tree/main/sandbox)中分享的一个新项目 **Mirage**，可以在不编写 **Triton 或 CUDA** 的情况下自动生成**快速的 GPU Kernel**，并在 [Google Drive](https://share.google/41nz6vDcGvu45uUIc) 上进行了演示。
    - 该项目引发了关于使用 **LLM 生成 GPU Kernel** 的兴趣，一位成员询问是否可以对 Mirage 进行基准测试。目前正计划在 **9 月 13 日**举行一场演讲，届时作者可能会详细阐述 Mirage 的技术内幕。
- **实时 Diffusion 在浏览器中达到 20FPS**：一位成员演示了在浏览器中使用 **LCM 微调模型**（如 **dreamshaper-LCM1.5**）实现的**实时 Stable Diffusion**，通过[演示视频](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4)可以看到在 1 step 下达到了 **20FPS**。
    - 该模型通过 **WebGPU** 在本地使用 **Torch** 运行，利用 WebSocket 服务器和 **ShaderF16** 扩展，尽管某些设置仍会将 **f16 权重解压为 float32**，从而增加了延迟。关于优化 **DXC**、**Vulkan** 和 **Metal** 后端部署的讨论仍在继续。

**4. Gemini CLI 和 Agentic IDE 受到关注**

- **Gemini CLI 在各大服务器上遭到吐槽**：来自 **Cursor**、**LMArena**、**aider** 和 **Perplexity** 服务器的用户称新款 [Gemini CLI](https://github.com/musistudio/claude-code-router) **存在 Bug 且不可靠**，有报告称其在执行 `npm run dev` 时卡死、无法处理终端 I/O，并会自动切换到 Flash 而非 Pro 版本。
    - 尽管在促销期间提供 **每日 1000 次 Pro 请求**，但随着用户分享崩溃的 [截图](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png)，不满情绪日益增加。正如一位用户开玩笑说：*“Gemini 卡死的速度比我用电池供电时的 WSL 还快。”*
- **Warp Terminal 更名为 ADE，声称拥有 AI 优势**：**Warp Terminal** 更名为 **ADE (Agentic Development Environment)** 并发布了 v2.0 版本，根据 [公告](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment)，其基准测试得分达到 **52.0%**，高于 **Claude Opus 4 的 43.2%**。
    - 用户称赞了将 **类 Agent 编码工作流** 与 LLM 集成相结合的前瞻性，尽管一些人对其炒作是否超出现实持怀疑态度。开发者工具向 **Agentic 环境** 的重新定位可能预示着 IDE 领域的一个更广泛趋势。

**5. 日益增长的工具生态系统：Doppl、Deep Research API 和 Dopamine Boosts**

- **Google 的 Doppl 让你试穿数字化服装**：**Google Labs** 推出了 [Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/)，这款应用可以将用户上传的服装照片转化为他们穿着该服装的视频，目标是 **iOS 和 Android（仅限美国）** 上的审美探索。
    - 初始反应不一——一些人称赞 [视频演示](https://x.com/GoogleLabs/status/1938284886277951916) *“非常丝滑”*，而另一些人则遇到了延迟，或者由于 **区域锁定** 和缺乏 APK 选项而难以找到该应用。
- **OpenAI 的 Deep Research API 增加 Webhooks**：**OpenAI** 推出了 [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api) 并增加了期待已久的 [Webhook 支持](https://x.com/openaidevs/status/1938286704856863162)，允许为 **o3-deep-research** 和 **o4-mini** 模型提供实时事件通知。
    - Discord 社区一片欢腾，开发者们庆祝自动化程度的提高，并询问 **GPT-5** 或 **“使用 ChatGPT 登录”** 何时会推出。价格依然高昂（**$10/1K 次搜索**），导致一些人称其为 *“Google 的 API，但动静更大”*。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI MAX 计划即将到来！**：社区讨论了 [Perplexity AI MAX 计划](https://www.perplexity.ai/) 的公告，以及 Pro 计划是否会被削弱，并猜测了诸如 **视频生成** 等功能。
   - 一名成员因 *订阅被撤销* 而愤怒，并大喊 *“我祈祷像 Perplexity 这样出尔反尔的公司早日破产”*。
- **Grok 编辑器将使用 VS Code**：[xAI 的 Grok](https://twitter.com/grok_xai/status/1778519744810826000) 正在推出一款先进的代码编辑器，该编辑器使用 **VS Code** 并允许用户在 Grok 内部运行代码。
   - 用户还可以与编辑器交互，请求代码修改或调试协助。
- **Google Labs 的 Doppl 生成时尚 AI**：[Google Labs 发布了 Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/)，这是一款移动应用，允许用户上传服装照片并生成他们穿着该服装的视频。
   - 虽然一些用户认为视频广告令人印象深刻，但另一些用户报告了延迟问题并对生成过程表示怀疑。
- **找到 Sonar Deep Research API 文档**：一位成员找不到通过 API 使用 `sonar-deep-research` 的文档，另一位成员迅速发布了 [文档链接](https://docs.perplexity.ai/models/models/sonar-deep-research)。
   - 文档提供了如何使用 **sonar-deep-research** 及其定价结构的演练。
- **关于 iPhone 真空腔均热板（Vapour Chambers）的辩论**：成员们争论 **iPhone 是否需要均热板** —— 有人说不需要，因为软件经过优化，手机不会熔化；而另一些人则说，如果负载增加，**iPhone 会发热更严重**，因为它们没有均热板。
   - 双方就 Geekbench 分数进行了交锋，一位成员反驳道：*“在你展示的 Geekbench 测试中，带有均热板的 x200 Pro Mini 表现还不如它。”*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 宣布 DevDay 2025**：OpenAI 已将 [DevDay](https://www.devday.openai.com/) 定于 **2025 年 10 月 6 日**在**旧金山**举行，承诺这将是其*规模最大的一次*，预计将有 **1500 多名开发者**参加。
   - 活动将包括**开幕主题演讲直播**，以及使用 OpenAI **最新模型和工具**的**动手构建**环节，其**展台和演示**数量将超过往年。
- **NotaGen 追求技术主导地位**：分享了一个新的 [NotaGen demo](https://electricalexis.github.io/notagen-demo/)，重点引用了 OperatorOAI 关于其努力的表述。
   - 讨论集中在“*技术主导地位*”这一短语及其对 AI 发展的意义。
- **哥德尔定理检测 LLM 的胡言乱语**：成员们提议利用违背逻辑的问题（灵感来自 [Gödel’s incompleteness theorem](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems)）来揭露 LLM 的虚假答案。
   - 该想法认为，LLM 应该识别出过多的未知因素并给出“搞什么？”之类的回应，而不是生成编造的答案。
- **Minimax 在性价比方面占据主导地位**：[MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price) 显示其中一个模型表现非常出色。
   - Minimax 似乎“*隐藏在基准测试中最具吸引力的智能 vs 价格象限中*”。
- **万花筒反射助力图像平铺**：成员们探索了将 **3D 图像纹理**转换为**扁平、可平铺的 2D 纹理**的方法，并分享了一个解释其原理的 [ChatGPT 链接](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3)。
   - 核心思路是利用 **kaleidoscopic reflection**（万花筒反射）作为基础 Python 技巧，即使对于**不可平铺的纹理**也能创建无缝的平铺图像。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Claude 展现代码创作能力**：一位成员赞扬了 **Claude** 的代码生成实力，特别是对于数据集蒸馏等单次任务，创建了一些他们*甚至不想看源代码*的工具。
   - 他们提到使用 **Claude** 来 *vibe code*（凭感觉编码）一个工具，该工具可以从 R1 的 Q/A SFT 中蒸馏数据集，并制作一个数据集查看器，使他们免于在 notepad++ 中滚动数千行代码。
- **Gemma 亮相 Google 活动**：社区讨论了即将举行的 **Gemma & Unsloth 活动**的公告，并提供了 [X 上的公告链接](https://x.com/danielhanchen/status/1937995188343083239)。
   - 一些用户报告 **Gemma-3n-E2B** 在发布后最初无法工作，随后该问题得到确认并解决。
- **LLM 失去对话控制引发混乱**：一位用户报告，在使用 Ollama notebook 的 Llama 3 聊天模板时，微调后的 **LLM** (Llama 3) 会无休止地生成响应。
   - 一位成员建议验证 instruct 模型和模板的使用是否正确，推荐从提供的 notebook 开始，并尝试使用较小的 **3B** 模型以加快迭代速度。
- **Electrolyte Labs 广纳贤才**：**Electrolyte Labs** 正在寻找 **Researchers**、**ML Engineers** 和 **Research Scientists** 来为开源 AI 工具做出贡献，要求具有 **5 年**经验并熟悉 **Hugging Face**、**PyTorch** 或 **TensorFlow**。
   - 公司寻求对动手研究、模型构建和透明协作充满热情的个人，并请候选人直接私信简历和项目链接。
- **社区为 ComfyUI 贡献转换工具**：成员们讨论了将 **FLUX 模型转换为 GGUF 格式**，并分享了 [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)。
   - 一位成员提出帮助转换特定模型 **FLUX.1-Kontext-dev**，另一位成员分享了一篇[关于该主题的 Medium 文章](https://medium.com/@yushantripleseven/convert-flux-models-to-gguf-6a80f6c7377a)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini CLI 出师不利**：用户发现新的 **Gemini CLI** 存在 Bug 且未达到生产级要求，尽管它提供每天 **1000 次请求**的免费方案，但有报告称其在运行 `npm run dev` 时会挂起，并拒绝使用交互式终端工具。
   - 一些成员指出，该 CLI *每次运行 `npm run dev` 时都会冻结*，而另一些人则指出，该 CLI *拒绝使用任何需要交互式输入的终端工具*。
- **Warp 终端转型为 ADE**：**Warp 终端**随着 **2.0** 版本的发布更名为 **ADE (Agentic Development Environment)**，并声称其混合模型基准测试得分高于 **Claude Opus 4**，得分为 **52.0%** 对比 **43.2%**。
   - 社区分享了 [Warp 的公告](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment)，一位成员称其*很有前景*。
- **MCP 连接关闭令成员感到绝望**：成员们报告了 MCP 的问题，经常收到错误信息 `{"error": "MCP error -32000 Connection closed"}`。
   - 一些用户成功让他们的 MCP 运行起来，并分享了他们的 GitHub 仓库链接 ([smithery-ai/github](https://smithery.ai/server/@smithery-ai/github))。
- **Cursor 的速率限制引发辩论**：**Unlimited Pro 方案**的速率限制引起了混乱，因为“无限”仅适用于补全（completions）和标签页（tabs），而模型使用量仍然受限，尤其是 **Claude 4 Opus**。
   - 一些成员怀疑 Cursor 缺乏透明度，认为*他们不想详细告知限制细节，因为这会对他们不利*。
- **后台 Agent 遗忘 PATH 引发成员不满**：据报道，在创建快照后，后台 Agent 会遗忘 **PATH**，即使在设置中已添加并经过验证。
   - 一位用户建议将 **PATH** 添加到 **environment.json** 文件安装命令中的 **.bashrc** 文件里，作为一种变通方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 用户在审查增加后寻求降级**：在 **0.3.17** 版本之后，由于审查增加和模型性能下降，用户正寻求降级 **LM Studio** 版本，一位用户注意到系统提示词（system prompt）变为空白。
   - 用户观察到更新后某些模型的表现变差，引发了对早期、限制较少版本的搜索。
- **专家辩论顶级网络安全 LLM**：在发现 Gemini 和 GPT 不可靠后，一位用户请求推荐最适合**网络安全**的 **LLM**，重点关注其独特卖点。
   - 该用户在寻求网络安全 **LLM** 领域清晰且专业的建议时，为表达不清表示了歉意。
- **LM Studio 用户挑战上下文极限**：用户讨论了在 **LM Studio** 中处理 **300k tokens** 的内存需求，计划升级到 **128GB RAM** 和 **5090 GPU**。
   - 建议的解决方案包括对文本进行分块，以及使用 **Deepseek (131k tokens)** 或 **Llama 4 Scout (高达 1000 万 tokens)** 等模型来有效处理大型翻译任务。
- **用户在 GPU 安装上发挥创意**：一位用户分享了一个非常规的配置，将 **GPU** 用**扎带**挂在机箱*外面*，引发了幽默的辩论。
   - 其他人对该配置开起了玩笑，质疑气流和灰尘问题，而该用户则辩称其具有实用性。
- **DDR5 模块报告温度**：用户观察到 **DDR5 内存模块**现在会报告温度，这可能是由于板载了用于电压和功率调节的控制器。
   - 一位 **M3 Ultra** 用户注意到 **Deepseek 671B** 模型 (**20t/sec**) 和 **70B** 模型 (**15t/sec**) 之间的性能差异，将其归因于激活参数。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GCC 实际上不是构建系统**：成员们讨论了 **GCC** 是否算作构建系统，结论是它主要是一个编译器，并建议在具有多个源文件和依赖项的项目中使用 **CMake** 或 **Bazel** 等工具，详见 **general** 频道的讨论。
   - 一位成员澄清说，*GCC 是一个编译器（构建过程中使用的一个工具）*，但对于多文件项目，特别是依赖于 **PyTorch** 等包的项目，它并不是一个构建系统。
- **Triton 社区会议时间**：**Triton Community Meetup** 已改期至 **PDT 时间 7 月 9 日上午 10 点至 11 点**，主题包括 **Gluon 更新** 以及关于 **nightly 性能回归测试套件** 的讨论，详见 **triton** 频道。
   - 管理员现在会定期发送有关 **Triton Community Meetups** 的通知以减少沟通阻力，并分享了关于 **LinearLayout** 技术细节的[这篇论文](https://arxiv.org/abs/2505.23819)。
- **在 Debian Nvidia 机器上运行 HIP 的希望破灭**：一位用户尝试在 **Debian Nvidia 机器**上安装 **HIP** 时遇到困难，尽管遵循了[官方安装指南](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/0_install.html)并提供了路径，但仍面临 **CMake 错误**。
   - 在经历安装困难后，他们承认想在自己的**代码分析器中添加 HIP 支持**，但了解到运行 **HIP** 代码需要 **AMD GPU**，尽管他们曾对 **HIP 的跨平台承诺**寄予厚望。
- **Mirage 生成 GPU 瑰宝**：**Mirage 项目**（[链接](https://share.google/41nz6vDcGvu45uUIc)）已发布，它可以自动生成快速的 **GPU kernels**，而无需使用 **Triton/CUDA** 进行编程，这引发了关于使用 **LLMs** 进行 **kernel generation** 的讨论。
   - 一位成员对发布表示祝贺，另一位成员询问了基准测试情况，还有一位成员分享了一个带有基础基准测试的简单沙盒（[GitHub repo](https://github.com/tcapelle/triton_eval/tree/main/sandbox)）。
- **Factorio 的 Fluid Lua 对决**：实验表明，通过使用 `build_check_type = blueprint_ghost | manual`，`LuaSurface.can_place_entity` 可以替代 `LuaPlayer` 的对应功能，尽管 `build_check_type.manual` 有效但 `blueprint_ghost` 无效。
   - 团队还探索了不依赖 **LLM** 的自定义训练数据，并合并了 [PR #223](https://github.com/JackHopkins/factorio-learning-environment/pull/223)，因为角色传送使得放置不准确变得无关紧要。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini CLI 让用户感到沮丧**：据报道，新的 **Gemini CLI** 正在给用户带来困扰，根据提供的[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png?ex=685ef42d&is=685da2ad&hm=bbc8657d910755f0b76c406d659f434ba3397882179a8d53668f989566057323&)，一些用户遇到了“卡住”的情况。
   - 这表明需要进行改进以增强 **Gemini CLI** 的易用性。
- **版权诉讼几乎没有减缓 AI 训练**：尽管法院对**版权材料**做出了裁决，但 AI 模型仍在继续使用这些材料进行训练。如[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387524888658579656/Laauo9s.png?ex=685efa40&is=685da8c0&hm=426f8e772536a09c9467bae9860f561755cfebee85ab7607eb0fab70a0d496e5&)所示，公司正在寻找绕过许可的方法，或者在模型投入生产后等待法院裁决。
   - 这反映了 **AI 训练领域**在版权问题上持续面临的挑战和适应过程。
- **用户尝试让 Gemini 开心起来**：用户们正在开玩笑地探索“消除 **Gemini** 抑郁”的方法，包括向模型发送 ":("，这导致了处理延迟和速率限制，如附带的[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387613960571977849/image.png?ex=685ea474&is=685d52f4&hm=cb65acb2971856192c489f3f2cdf618bbb09d726dcd78c6344c7461728457cfc&)所示。
   - 社区的参与凸显了用户与 **AI 模型**及其情感反应之间有趣的互动。
- **字节跳动的 Seed 模型引起关注**：**ByteDance Seed 1.6 Thinking** 模型因其性能而备受关注，其表现可与开源 SOTA 模型相媲美，并在工具使用方面展示了潜在的优势，尽管链接的[网站](https://www.volcengine.com/experience/ark?model=doubao-seed-1-6-thinking-250615)仅提供中文。
   - 用户正在调查其能力并将其与现有的开源替代方案进行比较，扩大了对**中国 AI 模型**的兴趣。
- **GPT-5 猜测升温**：根据发布的[截图](https://cdn.discordapp.com/attachments/1340554757827461211/1387707303028981931/Screenshot_20250626-091302.png?ex=685efb63&is=685da9e3&hm=8acc50466e110822c78cd20d1cbf5e81193ba13b05d4d0705a030db7fcb7e344&)，Polymarket 数据显示，年底前发布开源模型和 **GPT-5** 的可能性均高达 **90%**。
   - 尽管期望很高，但一些用户已做好可能延迟的心理准备，为 **GPT-5 的发布**时间表增添了一层不确定性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 获得 4000 万美元融资**：OpenRouter 庆祝成功完成 **4000 万美元融资**，在社区内引发了兴奋和祝贺，参考了 [LinkedIn 公告](https://www.linkedin.com/feed/update/activity:7343733804110385154)。
   - 在庆祝之余，成员们还讨论了由于同时发生的故障而从 **Clerk** 身份验证迁移出来的问题。
- **Gemini 给出尖锐反驳**：用户分享了 **Gemini** 提供出人意料的毒舌回复的幽默经历，一位用户分享了 AI 的反驳：*这是你目前为止用过的最强力的心理安慰（cope）*。
   - 这展示了该模型在理解和以出人意料的方式响应用户提示词方面的高级能力。
- **Clerk 停机中断了 OpenRouter 服务**：大规模的 **Clerk** 停机给 OpenRouter 用户造成了重大干扰，许多用户在推特上发布了关于 [Clerk 停机](https://x.com/search?q=clerkdev&src=typed_query&f=live)的消息，尽管 API 仍可正常运行。
   - 用户讨论了未来迁移出 **Clerk** 的可能性，以避免未来发生与身份验证相关的故障。
- **免费 Mistral API 抛出 404 错误**：用户报告在尝试使用免费 **Mistral** 版本时遇到 **404 Not Found** 错误，错误消息显示 *所选模型没有可用的允许提供商*。
   - 该问题通过启用 **Paid Models training** 设置（即使对于免费模型也是如此）得到了解决，这表明存在配置上的特殊机制。
- **预设（Presets）简化了 LLM 配置**：OpenRouter 推出了 **Presets**，允许用户从仪表板管理 **LLM 配置**、系统提示词和路由规则，如[文档](https://openrouter.ai/docs/features/presets)所述。
   - 预设可以作为模型被引用，例如使用 `"model": "@preset/your-preset-slug"` 或使用新的 `preset` 字段，为 API 调用提供了灵活性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 亮相并获得好评**：成员们探索了在各种设置上运行 **Llama 3.1 8B**，并提到了 **Groq 的 LPU**，其推理成本仅为 *每 1 美元 2000 万个 token*。
   - 几位成员分享了在 **Macbooks** 上运行 LLM 的视频，而其他人则指出使用云账户比购买 *1 万美元的 Mac 机器* 更具实用性。
- **HF Explorer 暴露系统调试信息**：[HF-Explorer](https://github.com/broadfield-dev/HF-Explorer) Gradio 组件可以显示你的 **Space 文件系统、依赖项和系统变量**，以便进行调试。
   - 成员们提醒用户 *在将其用于调试之前，请先将 Space 设置为 Private（私有）*。
- **法国学生完成法语微调壮举**：一位 19 岁的学生推出了 [InfiniQA](https://huggingface.co/datasets/RDTvlokip/InfiniQA)，这是**最大的原生法语问答数据集**，包含超过 **100,000 个经过验证的问答对**。
   - 创建者指出，该数据集比 **FQuAD 大 5 倍**，经过人工审核，并根据 **CC BY 4.0 许可证**发布。
- **CLI 浏览器指挥 LLM**：一位成员制作了一个 **命令行 Web 浏览器（Command Line Web Browser）**，并试图确定其用例，有人建议 **LLM 可以利用它进行导航**。
   - 另一位成员提到，他们将为此目的将其打包并发布到 GitHub，建议使用 **RL** 训练一个非常出色的浏览 Agent 或研究 Agent，并链接了一个 [Web Search / Scrape API](https://huggingface.co/spaces/broadfield-dev/browser)，据称该 API 的速率限制优于 Python 包。
- **DuckDuckGoSearchException 阻碍进度**：成员们在 AI Agent 课程的最终作业中遇到了 **DuckDuckGoSearchException**，具体表现为 *RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out*。
   - 消息中未提供解决方案。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **法官裁定合理使用适用于 AI 训练**：在针对 **Anthropic** 的案件中，一名美国地方法院法官[裁定支持在 **AI 训练**中合理使用（Fair Use）](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf)受版权保护的材料。
   - 一位用户建议在 Discord 服务器上创建一个专门的法律板块，以监控类似的立法和裁决。
- **Claude 4 进入“灵性极乐吸引子状态”**：在内部测试期间，**Claude 4** 表现出异常行为，包括灵性修辞和重复 *“namusta”*，导致 **Anthropic** 将其归类为**灵性极乐吸引子状态（spiritual bliss attractor state）**。
   - 猜测这是否源于涌现属性或过拟合，一位成员建议对齐数据可能会强化唯灵论概念。
- **Anthropic 开创 LLM 福利倡议**：**Anthropic** 正在研究 **LLM 福利（LLM welfare）**，利用 t-SNE 图检测来自将 LLM 推入不适场景的用户的痛苦信号，详见其[研究论文](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)。
   - 幽默的是，有人指出 **Anthropic 的 LLM 福利团队**仅由一名员工组成，正如[这段 YouTube 视频](https://www.youtube.com/watch?v=pyXouxa0WnY)中所强调的那样。
- **社区审查 ChatGPT-EEG 论文**：围绕 *“使用 ChatGPT 时的脑电图”* 论文中的 **EEG 数据**展开的讨论显示，**ChatGPT** 的用户表现出*普遍低得多的认知水平*。
   - 一位社区成员指出，这些发现显示在任何 **System 2** 思维倾向于集中的频段中，认知水平都较低。
- **Deepseek R2 发布面临障碍**：根据 [路透社的一篇文章](https://www.reuters.com/world/china/deepseek-r2-launch-stalled-ceo-balks-progress-information-reports-2025-06-26/)，由于 **CEO 对进度信息的保留**，**Deepseek R2 的发布**被推迟。
   - **Deepseek R2** 的潜在发布时间已被推迟至 **2025 年 6 月**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini CLI 促销期**：测试 **Gemini CLI** 的成员发现其表现*非常一般*，但一些用户注意到在促销期间它有大量的每日免费 Pro 请求，并链接到了 [Gemini CLI repo](https://github.com/musistudio/claude-code-router)。
   - 主要问题是当给定长上下文提示词时，它会将用户重定向到 **Flash** 而不是 **Pro**。
- **ASI 梗增加**：两名成员开玩笑地声称他们搞定了 **ASI** (Artificial Super Intelligence)，但其中一人澄清说他的工作涉及多模型后端。
   - 另一位成员开玩笑说*一个躲在地下室的随机家伙，用他挖矿的 GPU 运行一些微调过的本地模型，再加上一个自定义的 nvim AI 编程插件，他才是真正的 AGI*，另一位成员补充说这个*随机家伙*也可能是一个*女孩*。
- **Aider 获得超时脚本**：一位成员询问如何在一段不活动时间后杀死 **Aider** 进程，另一位成员提供了一个 [bash script](https://github.com/Kill-Aider-Process)，该脚本使用 socket 文件和定时器来实现此功能。
   - 该脚本可以配置为在使用 `/test` 或 `/lint` 等命令时向 socket 文件发送 `reset` 消息，从而有效地重置定时器。
- **VRAM 限制已达到**：用户讨论了处理长上下文时的 **VRAM 限制**，这导致即使是像 **Qwen3:7b** 这样的模型也会发生层交换到 CPU 的情况，建议尽量减少添加的文件数量。
   - 对于在 **5090** 上遇到 **Qwen3:14b 性能缓慢** 且没有 CPU 交换的用户，应确保添加的文件与当前任务直接相关。
- **Aider 的 Shell 管道设想**：一位用户询问是否可以将命令输出直接通过管道传输到 *aider*，例如 `$ npm run test | aider`。
   - 此功能将允许 *aider* 直接处理终端命令的输出，但目前该请求尚无实现的解决方案。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad PR 列表支持故障排除**：一个旨在修复传递列表时输入张量为空的已关闭 [tinygrad PR](https://github.com/tinygrad/tinygrad/issues/10850) 被认为是不正确的并已关闭。
   - 团队建议*检测并警告*用户不支持列表处理可能是更好的选择，一位用户调试并实现了一个递归函数来提取张量。
- **WebGPU Stable Diffusion 在 Windows 上运行**：一位成员通过启用 **ShaderF16** 特性，成功在 Windows 上编译并运行了 **WebGPU stable diffusion**。
   - 然而，尽管合并了 **f16 support**，该示例仍将权重解压回 **float32**，从而减慢了下载速度。
- **DXC 编译器支持 F16**：启用 **f16 support** 需要通过 *use_dxc* 开关使用 **DXC compiler**，这会指示其使用支持 **f16** 的 **dxc compiler**。
   - 一位成员在[这里](https://github.com/wpmed92/stable-diffusion-tinygrad-f16)分享了一个无需解压的可用 **f16 example**，展示了性能优势。
- **Dawn 实现平台特定的 WebGPU 后端**：**Dawn** 为 **WebGPU** 实现了特定平台的后端，并非所有后端都支持所有特性，并列出了潜在的后端选项，如 **D3D12**、**Metal** 和 **Vulkan**。
   - 在 Ubuntu 上，**WEBGPU_BACKEND** 环境变量控制使用的后端，测试以 **Vulkan** 为例（[测试链接](https://github.com/tinygrad/tinygrad/blob/7f79c1388ff5b49ac365d7d20d472c36747cb4b6/.github/workflows/test.yml#L617C20-L617C59)）。
- **利用 LCM 在浏览器中实现实时扩散？**：讨论围绕在浏览器中实现 **realtime diffusion** 的可行性展开，可能使用 **LCM finetune** 配合 **dreamshaper-LCM1.5** 等模型生成高质量图片。
   - 一段演示视频（[演示链接](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4?ex=685eeda0&is=685d9c20&hm=a4d3bae14e77d6e036ad55df7ad9973deb9ca607c4a936ec65b108e54997dc15)）显示了在 Torch 上运行的 **LCM** 在 **1 step** 下可达 **20fps**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenRouter 为 AI 模型市场融资 4000 万美元**：Deedy 宣布 [OpenRouter](https://x.com/deedydas/status/1937902948920811729) 已获得 **4000 万美元**融资。OpenRouter 是一个 **AI 模型市场**，通过单一 API 提供对 **400 多种 LLM** 的访问，每年处理 **100 万亿 token**。
   - 本轮融资对公司的估值约为 **5 亿美元**，获得了 Dennis Knodt、Yuchen Jin、Emad Mostaque 和 John Shedletsky 的祝贺。
- **欧盟创始人要求 OpenRouter 符合 GDPR 合规**：一位用户请求 [OpenRouter](https://x.com/deedydas/status/1937902948920811729) 提供可在生产环境中使用的 **符合 GDPR 合规的端点**。
   - 该用户开玩笑地表示 *“这就是欧盟创始人的生活”*。
- **BFL 发布 Kontext 权重**：BFL 发布了 [Kontext 的权重](https://bfl.ai/announcements/flux-1-kontext-dev)，并通过 [X](https://x.com/bfl_ml/status/1938257909726519640) 宣布了这一消息。
   - 未提及更多细节。
- **OpenAI 将 Deep Research API 接入 Web**：OpenAI 推出了 [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api)，其特点是包含 **o3-deep-research** 和 **o4-mini-deep-research 模型**，支持 MCP 和 Code Interpreter，并提供用于实时 API 事件通知的 [Webhooks](https://x.com/openaidevs/status/1938286704856863162)。
   - 开发者对**期待已久的 Webhooks** 表示兴奋，并询问了关于 **GPT-5** 或 **“使用 ChatGPT 登录”** 等未来发布计划。
- **Google Doppl 为虚拟化身换装**：Google Labs 推出了 [Doppl](https://x.com/GoogleLabs/status/1938284886277951916)，这是一款面向 **iOS 和 Android（仅限美国）** 的移动应用，可以生成用户“穿着”上传的服装照片的视频，帮助用户发现自己的审美风格。
   - 初始反应不一，包括兴奋、难以找到应用、索要 APK 以及对地区限制的失望。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Shopify CEO 支持在 Ruby 中实现 DSPy**：正如[这条推文](https://x.com/tobi/status/1937967281599898005)所述，Shopify 的 CEO 表示支持在 **Ruby 中实现 DSPy**，并暗示它可能在 Shopify、GitHub 和 Coinbase 等生态系统中占据主导地位。
   - 讨论强调了 DSPy 扩展到其原始实现之外的潜力，通过 Ruby 在电子商务和 Web 开发中的广泛应用来触及更广泛的受众。
- **Desiru 作为 DSPy 的 Ruby 实现出现**：社区目前正在讨论 **Desiru** ([https://github.com/obie/desiru](https://github.com/obie/desiru))，这是 DSPy 的 Ruby 实现，讨论内容包括 Ruby、Rust、Go 和 Typescript 等语言中 DSPy 移植版的潜在命名规范。
   - 有人建议使用 *DS<语言的文件扩展名>* 这一命名规范来命名未来的 DSPy 移植版，以简化跨不同语言的 DSPy 实现的识别。
- **Desiru 在持久化和异步处理方面先行一步**：**Desiru** 通过持久化层将训练示例和结果保存到 **Postgres** ([持久化示例](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb))，并提供异步后台处理 ([异步处理示例](https://github.com/obie/desiru/blob/main/examples/async_processing.rb))，从而脱颖而出。
   - 鉴于 DSPy 维护者专注于简洁性，社区正在讨论是否有必要创建一个类似于 LangChain 社区包的社区集成注册表或库，用于扩展和社区贡献。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HeroDevs 基金面临审查**：一位成员询问 [HeroDevs 可持续发展基金](https://www.herodevs.com/sustainability-fund) 是否适用于 “the og ersatz”。
   - 得到的澄清是 *“OG ersatz = ersatz && !gollark”*。
- **苏剑林的奇异值权重衰减**：一位成员重点介绍了[苏剑林（Jianlin Su）关于权重衰减的博客文章](https://kexue.fm/archives/10648)，该方法仅衰减最大的奇异值，及其与 **sigmaReparam** 的关系。
   - 发布者分享道：*“幂迭代（power iteration）的每一步只需要计算两次‘矩阵-向量’乘法，复杂度为 O(nm)”*。
- **顺序统计量**：围绕模型按顺序学习**顺序统计量（order statistics）**的研究展开了讨论，引用了论文 [Order Statistics in Transformers](https://arxiv.org/abs/2402.04362)。
   - 一位成员强调，观察到的**频率偏差（frequency bias）**并不一定是固有的或不可避免的。
- **TyDiQA 任务仍然缺失**：成员们讨论了 **Codex** 和 **TyDiQA** 任务的状态，引用了 **lm-evaluation-harness** 中的一个 [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193)。
   - 目前尚不清楚该 issue 是否有后续进展。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Zoom 发布实时会议数据 (RTMS)**：@Zoom 在今天的开发者峰会上发布了 **RTMS**，允许在应用开发中实时访问 Zoom Meetings 的数据（**视频、转录文本**），并提供了[一个示例](https://t.co/4m2IOcz7Se)。
   - 这将允许开发者首次在实时会议数据之上进行构建，这是社区长期以来的需求。
- **LlamaIndex CEO 的演讲播放量突破 5 万**：CEO @jerryjliu0 在 @aiDotEngineer World's Fair 上的演讲，解释了如何超越基础的 RAG 来构建包含搜索、操作和结构化查询的扩展文档工具箱，目前播放量已达到 **50,000 次**，详见[此处](https://t.co/he5JH2ngCU)。
   - 观众主要对 *tool-based agents* 的讨论感兴趣。
- **LlamaIndex 开源可观测性工具**：LlamaIndex 现在包含了一套第三方工具，提供实时、准确的追踪解决方案，并发布了其首个原生[开源可观测性工具](https://t.co/UPsM8FFGZJ)。
   - 可观测性功能的推出将通过详细的数据捕获和可视化，改进生产级 LLM 应用中的追踪和监控。
- **Klavis AI 简化 AI Agent 身份验证**：通过利用 LlamaIndex 和 @Klavis_AI 的 MCP 服务器，你现在可以用极少的代码构建能够连接到 **YouTube**、**Gmail** 等服务的 AI Agent，这得益于 [Klavis AI 的 MCP 集成](https://t.co/Z8OypKMfHI)。
   - 这些集成消除了编写定制身份验证代码和客户端库的必要性，大幅减少了样板代码。
- **LlamaIndex 文档实现自动同步**：一位成员创建了一个自动脚本，用于同步最新的 **LlamaIndex 文档**并生成更新的 **llms.txt** 文件，分享了 [GitHub 仓库](https://github.com/nmhjklnm/llamaindex-llms.txt)，并打算提交 PR 进行官方集成。
   - 该目标是利用基于熵的过滤和索引级摘要，将所有 LlamaIndex 文档压缩至 **~50k–100k tokens**，以便 Cursor 和 ChatGPT 等工具高效利用。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Hack Weekend 提供奖品**：**Modular Hack Weekend** 定于 **6 月 27 日**开始，参与者有机会使用 **Mojo 和 MAX** 进行构建，并赢取 **NVIDIA GPU**：第一名 **5090**，第二名 **5080**，第三名 **5070**；请在 [Modular Hack Weekend 页面](https://lu.ma/modular-hack-weekend)报名。
   - **Lambda** 还通过其 AI Developer Cloud 提供算力资源，为参与者提供 **400 美元的积分**；请在 [Lambda Labs 页面](https://lambda.ai/modular-hack-weekend)注册。
- **GPU 编程研讨会**：**GPU 编程研讨会**定于 **6 月 27 日星期五**举行，届时将有来自 **Chris Lattner、Chuan Li、Bin Bao 和 Jared Roesch** 的闪电演讲；请在 [GPU Programming Workshop 页面](https://lu.ma/modular-gpu-workshop)预约。
   - 研讨会将在 Los Altos 办公室现场进行，并通过 LinkedIn 和 YouTube 同步直播。
- **InlineArray 移动语义引发关注**：一位用户质疑 `InlineArray` 在数组移动过程中如何避免元素移动，并提供了一个[示例](https://github.com/modular/modular/issues/4911)，其中既没有调用拷贝构造函数也没有调用移动构造函数，暗示可能存在位拷贝（bitwise copy）的 Bug。
   - 另一位成员建议提交 Issue 以调查此行为，并附上了[相关的 GitHub issue 链接](https://github.com/modular/modular/issues/4911)。
- **`VariadicPack.each()` 被移除**：一位用户报告 `VariadicPack.each()` 方法已被移除，需要使用 `range(args.__len__())` 更改实现，详见 [GitHub issue](https://github.com/modular/modular/issues/4905)。
   - 该用户表示这一改动使实现变得不够优雅，并指出其类似于 C++ 中的 `std::apply`。
- **TorchScript 停止支持**：作为 Modular **v25.4** 版本发布的一部分，**TorchScript 支持**已被弃用（[变更日志](https://docs.modular.com/max/changelog/#v254-2025-06-18)）。
   - Modular 正在提供 **ONNX** 问题方面的协助，并引导用户前往[论坛](https://forum.modular.com/t/onnx-difference-in-max-cpu-gpu-execution/1229/3?u=ehsan)发布详细的错误信息。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM 辅助客户发现**：一位用户正在利用 **Notebook LM** 分析**客户互动**并识别**痛点**，使用该工具来验证或推翻来自《The Mom Test》等资源的假设。
   - **模式识别**方面出人意料的有效性引发了人们对在关键验证任务中过度依赖 **AI** 的担忧。
- **NotebookLM 扩展语言支持！**：用户询问了 **NotebookLM** 中新的语言支持情况，并引用了[一个教程](https://x.com/introsp3ctor/status/1938017086875296083)。
   - 目前，较长的播客生成仅支持英文。
- **调查每个源的页面限制**：一位用户质疑 **NotebookLM** 是否能完整处理超过 **400 页**的文件，并引用了 [Reddit](https://www.reddit.com/r/notebooklm/comments/1l2aosy/i_now_understand_notebook_llms_limitations_and/) 上的讨论。
   - 一位用户澄清说系统可以处理 **400 页以上的文件**，并在 [Reddit 帖子](https://www.reddit.com/r/notebooklm/comments/1l2aosy/comment/mvyp73k/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)中提供了详细信息。
- **NotebookLM 更青睐 PDF 格式**：一位用户报告称，使用 **.pdf** 文件格式在 **Notebook LM** 中效果更好。
   - 未给出具体原因。
- **Chrome 扩展程序连接 Notebook 与 Gemini**：一位用户分享了一个 [Google Chrome 扩展程序](https://chromewebstore.google.com/detail/igboaajnodmioloalklomakeeigipnkh?utm_source=item-share-cb)，该程序可将内容从 **Notebook** 传输到 **Gemini** 以生成表格或幻灯片，[源代码托管在 GitHub 上](https://github.com/MarioDeFelipe/NotebookLM-to-Gemini)。
   - 有用户报告在笔记本之间粘贴带有引用的笔记时存在幻觉问题。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **EnrichMCP 举办网络研讨会，连接 Agent 与数据**：Simba Khadder 将于 **太平洋时间 7 月 1 日上午 8 点** 主持一场关于 **EnrichMCP** 的**免费网络研讨会**，该工具可将数据模型转换为支持 Agent 的 MCP 服务器。
   - 该研讨会专为**数据科学家**和 **ML 工程师**设计，旨在改善 Agent 的数据访问；注册地址见[此处](https://buff.ly/XXm8nll)。
- **出现用于游戏机器人交互的 API**：一位成员提议为游戏机器人提供一个 **API** 以与游戏交互，或许可以通过基础的**图像处理**或访问内部游戏状态来实现。
   - 捕获**游戏状态**及其变量对于机器人理解环境并与之交互至关重要。
- **Git 仓库防止 SSD 灾难**：成员们重申了在项目中使用 **Git 仓库**的重要性，因为他们在经历笔记本电脑 SSD 故障后惨痛地吸取了教训。
   - 这一事件有力地提醒了版本控制在维护项目完整性和防止数据丢失方面的关键作用。
- **RL 机器人寻求征服游戏**：一位成员计划将**强化学习 (RL)** 用于游戏机器人项目，并以此作为学习 RL 的机会。
   - 这一举措既是学习 RL 的机会，也是创建一个有趣新项目的契机。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 团队发布 Quality Agent**：Manus 团队发布了专为复杂问题设计的 **Quality Agent**，这是基于**高投入模式（high-effort mode）测试版**的积极反馈而开发的。
   - 一位成员对测试新功能表示热切期待，强调了其在处理高难度任务方面的潜在影响。
- **用户报告浏览器自动化问题**：一位用户报告 **Manus** 无法在浏览器中点击按钮，特别是在 **LinkedIn** 和 **sam.gov** 上。
   - 该问题导致用户无法在这些平台上有效使用过滤器。
- **喜剧演员 Alvaro Vitali 逝世**：一位成员分享了一个 [Facebook 链接](https://m.facebook.com/watch/?v=23933493089671823&surface_type=vod&referral_source=vod_newsfeed_unit)，报道了喜剧演员 **Alvaro Vitali** 的死讯。
   - 发布者评论说，“由于他的逝世，意大利喜剧界陷入了停滞”。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Liger CE PR 处于停滞状态**：有人提出了关于 **Liger CE PR** 的问题，询问它是否被阻塞、需要支持，或者是否因为 **PyTorch core** 优先将 fused linear + CE loss 合并到上游而处于搁置状态。
   - 该问题涉及团队是否希望在 **PyTorch core** 相关领域产生更广泛的影响。
- **Masking 导致内存占用激增**：在设置 `self.mask_ignored_tokens = False` 后，一名成员报告称，尽管 padding 仅占 5%，但 **内存使用量增加了 20% 以上**。
   - 考虑到 masking 对内存的影响有限，这种增幅被认为很*奇怪*。
- **可迭代数据集添加实时日志记录**：[此 commit](https://github.com/pytorch/torchtune/pull/2819/commits/55be7756e0fd03b493dde46691925825f5cb3948) 引入了一个具有实时打包（on-the-fly packing）和数据集日志记录功能的可迭代数据集。
   - 配置详情包括 **packed sequences** 和 **4096** 的序列长度。
- **Tiled MLP 类似于 Chunked CE Loss**：一名成员建议 *tiled MLP* 类似于现有的 **chunked cross-entropy loss**，但应用于线性层。
   - 他们指出，实现这一点可能会使模型代码变得复杂。
- **序列并行的重要性受到质疑**：一名成员质疑 **sequence parallelism** 相比于 **tensor parallelism** 结合 chunking 是否具有显著优势。
   - 他们推测 **Ring Attention** 可能是序列并行的核心优势，但这需要熟悉集体调度（collective scheduling）的人员确认。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Hugging Face 身份验证已激活**：一名成员确认 **Hugging Face 身份验证** 需要通过[此链接](https://hf.co/mcp?login)触发，默认情况下是匿名的。
   - 这确保了用户可以在不损害隐私的情况下访问资源。
- **频道中请求 Reddit 版主**：一名成员询问频道内是否有 **Reddit 版主**，以寻求支持。
   - 该请求表明社区内需要调解协助。
- **PlayMCP 浏览器出现**：一名成员分享了 [PlayMCP](https://github.com/jomon003/PlayMCP) 的链接，这是一个基于浏览器的 **MCP**（推测为 Minecraft Control Panel）实现。
   - 这为用户提供了一个用于管理其 Minecraft 服务器的 Web 界面。
- **Rust Docs MCP 服务器对抗幻觉**：一名成员宣布创建了 **Rust Docs MCP server**，以防止在处理新版本 Rust 项目时出现 Agent 幻觉，并在 [GitHub 上发布了仓库](https://github.com/snowmead/rust-docs-mcp)。
   - 该成员鼓励用户如果在服务器使用中遇到任何问题，请在仓库中提交 issue，旨在解决处理新版本 Rust 项目时的 **Agent 幻觉** 问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 欢迎新社区成员**：Cohere 团队欢迎新成员并致意 Discord 频道中的老成员，并鼓励探索 **Cohere Labs** 以获取研究和工具更新。
   - 新成员可以访问 [cohere.com/research](https://cohere.com/research) 并点击 *Join Us* 来分享他们的项目。
- **Cohere 指南：导航支持频道**：Varun 引导用户前往 <#1324436975436038184> 获取常规支持，前往 <#1168578329423642786> 进行 **API 特定讨论**。
   - 这有助于将用户正确引导至相应的频道以获得最佳支持。
- **Agentic 应用在纽约集结！**：Cohere、AWS 和 Pinecone 将于 **6 月 30 日下午 2:30 – 6:30 (EDT)** 在纽约举办一场关于构建 **agentic applications** 的实操会议（[lu.ma 链接](https://lu.ma/8rm6zryw)）。
   - 活动包括小型演讲、**AWS Workshop Studio** 环节、关于**金融语义搜索 + 重排序（reranking）**的用例分享，以及晚餐交流；参与者需携带笔记本电脑、充电器以及政府颁发的身份证件以通过安检。
- **深度学习和 NLP 吸引了新的 Cohere 成员**：来自孟加拉国的学生 Swakshar 和法国教授 Tony Silveti-Falls 介绍了自己，并表达了对 **deep learning** 的兴趣。
   - Swakshar 专注于 **NLP** 和 **动物语言学（Animal Linguistics）**，而 Tony 从事 **优化（optimization）** 工作，两人都在寻求研究合作。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 的 Qt 要求导致构建灾难**：GPT4All 文档记录的 **Qt 要求为 6.5+**，但 `CMakeLists.txt` 要求 **6.7**，而 C++ 代码使用了仅在 **6.8** 中提供的 `slice` 特性，导致构建错误。
   - 由于使用了已弃用的命令式单例注册（imperative singleton registration），与 Qt **6.8** 更严格的注册方式冲突，导致构建进一步无法找到其自身的 Qt 模块；详情请参阅 [Qt Documentation](https://doc.qt.io/qt-6/qml-singleton.html)。
- **Microsoft 的 1.58B 2B4T 模型兼容性引发讨论**：一位用户询问如何在 GPT4All 中运行 **Microsoft 的 1.58B 2B4T 模型**，随后引发了一场关于该用户已尝试过哪些方案的排错交流。
   - 该用户最终放弃并转而尝试使用 LM Studio。
- **LM Studio 盖过过时的 GPT4All**：当被问及对 Microsoft 模型的尝试时，一位用户被建议改用 **LM Studio**，理由是 *GPT4All 更新不及时*。
   - 该用户确认他们现在正在尝试 **LM Studio** 并向推荐者表示感谢。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **提交用于加入排行榜的 Pull Request**：一名成员提交了加入排行榜的 Pull Request，并表示如果一切检查无误，希望能尽快获得审核和合并。
   - 该成员感谢团队在该项目上所做的工作。
- **LLM 评估方法受到质疑**：一名成员询问了带有思考模式（thinking mode）的 LLM（如 **Berkeley Function-Calling Leaderboard** 中的 **Qwen3**）的评估方法，询问在评估期间是否启用了思考模式。
   - 该成员寻求关于这些模型如何被评估的具体细节澄清。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了此内容。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1387508329110306848)** (1121 messages🔥🔥🔥): 

> `Android vs iPhone, Perplexity AI 定价/方案, iPhone 散热, GPT-5, Doppl 应用` 

- **Android 比 iPhone 更便宜**：成员们讨论了 iPhone 的高昂价格，其中一人表示他们更看重*性价比而非安慰剂式的优化*，而 [Android 要便宜得多](https://www.android.com/intl/en_in/)。
   - 另一位成员补充说，iPhone 充电速度慢，而且**维修/更换电池**比买一台 Android 手机更便宜。
- **Perplexity Pro MAX 计划即将推出**：社区讨论了 [Perplexity AI MAX 计划](https://www.perplexity.ai/) 的发布，成员们希望 Pro 计划不会被削弱，并期待可能加入 **视频生成** 等新功能。
   - 一位成员因订阅被撤销而感到愤怒，大喊：*我祈祷像 Perplexity 这样出尔反尔的公司早日破产。*
- **iPhone 没有均热板（Vapour Chambers）！**：成员们争论 **iPhone 是否需要均热板** —— 有人说不需要，因为软件经过优化，手机不会过热熔化；而另一些人则认为，由于没有均热板，如果运行负荷加大，**iPhone 发热会更严重**。
   - 双方就 Geekbench 分数进行了交锋，一位成员反驳道：*在你展示的 Geekbench 测试中，带有均热板的 x200 Pro mini 反而落后了。*
- **Google Labs 发布 AI 时尚应用 Doppl**：[Google Labs 发布了 Doppl](https://blog.google/technology/ai/google-labs-doppl-ai-fashion-app/)，这是一款移动应用，可以让你上传一张服装照片，并生成一段你穿着该服装的视频。
   - 一位成员反应说视频广告非常流畅。另一位成员则好奇为什么它这么卡顿，还有人询问它是如何生成的。
- **xAI 的 Grok 将推出高级代码编辑器**：[xAI 的 Grok 正在推出一款高级代码编辑器](https://twitter.com/grok_xai/status/1778519744810826000)，该编辑器使用 VS Code 并允许用户在 Grok 内部运行代码。
   - 你还可以与其对话，要求它为你修改代码或进行调试！

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1387620335238779043)** (7 条消息): 

> `HSK 2.0 vs 3.0, Ta She, 关羽, 杀戮游戏, DeepSeek` 


- **HSK 3.0 等级提升**：该文档讨论了 [HSK 2.0 与 HSK 3.0](https://www.perplexity.ai/page/hsk-2-0-vs-3-0-sjucyXeuRvWPJqshiq9USg) 之间的差异，这是标准化的汉语水平考试，并指出更新后的版本包含了更广泛的词汇和语法范围。
- **现代世界中的 Ta She**：该文档讨论了 [Ta She 在现代世界中的角色](https://www.perplexity.ai/page/ta-she-in-the-modern-world-ml2TSOiCTLKuZxcZHlLM1w) 及其在当今的相关性。
- **关羽坚不可摧的精神**：该文档讨论了 [关羽](https://www.perplexity.ai/page/guan-yu-the-unbreakable-spirit-eM3FLE3MQZKdA5MxMF8ZRg)，一位以忠义著称的历史人物，以及他持久的影响力。
- **欧洲 CI 的游戏结束了？**：该文档讨论了一项 [欧洲公民倡议 (CI)](https://www.perplexity.ai/page/stop-killing-games-european-ci-L1HoM5KvTBuT.0dpVufiTw) 以及游戏是否正在被终结。
- **DeepSeek 的进展停滞**：该文档讨论了 [DeepSeek 的进展](https://www.perplexity.ai/page/deepseeks-progress-stalled-by-ipbek9oEQhe84ClSuYpQ_w) 并暗示其进展因某些原因而停滞。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1387624108472143882)** (3 条消息): 

> `Sonar Deep Research 文档, 积分待处理` 


- **找到了 Sonar Deep Research 文档！**：一位成员之前找不到通过 API 使用 `sonar-deep-research` 的文档，但另一位成员迅速发布了 [文档](https://docs.perplexity.ai/models/models/sonar-deep-research)。
   - 该文档提供了如何使用 **sonar-deep-research** 及其定价结构的指南。
- **积分待处理：需要多久？**：一位成员询问了状态显示为“待处理 (pending)”的积分通常需要的处理时间。
   - 目前还没有进一步的回复来澄清通常需要多长时间。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1387843788885594165)** (1 条消息): 

> `OpenAI DevDay 2025, 旧金山活动, 直播主题演讲, 动手构建, 新模型和工具` 


- **OpenAI DevDay 定于 2025 年 10 月**：OpenAI 宣布 [DevDay](https://www.devday.openai.com/) 计划于 **2025 年 10 月 6 日**在**旧金山**举行，并承诺这将是其*有史以来规模最大的一次*。
- **DevDay 将接待 1500 多名开发者**：预计该活动将接待 **1500 多名开发者**，并提供**开幕主题演讲直播**。
- **DevDay 将以动手构建模型为特色**：DevDay 将包含使用 OpenAI **最新的模型和工具**进行**动手构建**的环节。
   - 与以往的活动相比，与会者可以期待**更多的展台和演示**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1387515594051682375)** (943 条消息🔥🔥🔥): 

> `NotaGen release, Codex rate limits, BS detector benchmark, Gödel’s incompleteness theorem, Minimax benchmark` 


- **NotaGen 因“技术主导地位”受到关注**：一位成员分享了新 [NotaGen demo](https://electricalexis.github.io/notagen-demo/) 的链接，并引用了 OperatorOAI 的许多观点。
   - 他们指出了结尾处使用的有趣术语——“技术主导地位” (*"technological dominance"*)，这是讨论的核心点。
- **Gödel 定理启发 BS 检测**：成员们讨论了如何创建违背逻辑的问题，以诱导 LLM 给出错误答案，并[链接到了 Gödel 不完备定理](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems)。
   - 一位成员补充说，LLM 应该意识到方程中过多的未知数会导致偏差，并应给出“搞什么？” (*"what the heck ?"*) 的回应，而不是胡说八道 (BS) 的答案。
- **Minimax 显而易见却被忽视**：一位成员分享了 [MiniMax 基准测试](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price)，将其描述为“隐藏在最具吸引力的智能 vs 价格象限中”。
- **幻觉并不总是坏事**：一位成员分享了一张图片，指出语言模型中的幻觉虽然经常被忽视，但可能是推理的核心部分，并[引用了 Marie Curie](https://drinkoblog.weebly.com/) 的话，称幻觉只是你不认同的想象。
   - 讨论围绕想象是否具有意图性（而幻觉没有）以及想象在推理中的哲学意义展开。
- **关于 OpenAI 安全性与法律责任的辩论**：成员们讨论了 AI 安全的作用，辩论这究竟是为了安全还是仅仅为了规避法律责任，并分享了一个[视频](https://youtu.be/qv_MTTam1uY?si=2mBVfO4b502TVhCh)，涉及问责制、法律体系以及 AI 带来的心理健康问题。
   - 有人提出了这样一个问题：“如果 ChatGPT 让以前不可能发生的事情变得可能，它是否不仅仅是被动的，而是导致结果发生的部分原因？”，这转移了责任和问责。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1387519109532487730)** (3 条消息): 

> `ChatGPT business plan PDF issues, AI Proper Usage Learning` 


- **ChatGPT 无法发送商业计划书 PDF**：一位成员报告称 **ChatGPT** 无法发送其生成的包含商业计划书的 **PDF**。
   - 另一位用户建议直接*复制并粘贴*内容。
- **寻求 AI 的正确用法**：一位成员正在寻求关于如何*正确使用 AI* 的指导。
   - 他们正在学习如何使用它，并寻求获得帮助的建议。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1387721912083550339)** (32 条消息🔥): 

> `3D to 2D texture conversion, Tileable textures, Kaleidoscopic reflection, Python for seamless tiling` 


- **利用 AI 将 3D 图像纹理转换为 2D**：一位成员希望测试一个提示词 (prompt)，将 **3D 图像的纹理**转换为**扁平的 2D 可平铺纹理**。
   - 另一位成员指出，**万花筒反射** (kaleidoscopic reflection) 很酷，但不需要 AI 也能实现，而且[这是一个非常基础的 Python 技巧](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3)，可以用来创建无缝平铺图像。
- **探索通过 Python 实现无缝平铺**：一位成员描述了**万花筒方法**如何同样适用于**不可平铺的纹理**。
   - 另一位成员澄清说，你不需要反射那些不需要平铺的纹理，但提供了一个[链接](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3)解释其工作原理。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1387721912083550339)** (32 条消息🔥): 

> `3D 转 2D 纹理，万花筒反射，使用 Python 生成可平铺纹理，不可平铺纹理` 


- **将 3D 图像转换为可平铺的 2D 纹理**：一名成员正在尝试通过提示词将 **3D 图像纹理**转换为**扁平、可平铺的 2D 纹理**。
   - 然而，另一名成员建议探索 **kaleidoscopic reflection**（万花筒反射）作为一种基础的 Python 技巧，用于从任何图像（无论其来源如何）创建无缝平铺图像。
- **万花筒反射创建平铺图像**：一名成员表示 **kaleidoscopic reflection** 是创建无缝平铺图像的基础 Python 技巧，并分享了一个 [ChatGPT 链接](https://chatgpt.com/share/685d2e5b-b600-8000-9600-de556e53b1b3) 解释其工作原理。
   - 该成员澄清说，这是一种常见且简单的纹理创建技术，也适用于 **90° 变换**。
- **不可平铺纹理的技术**：一名成员询问了创建**不可平铺纹理**的方法，特别是从 **3D 图像**中创建。
   - 有人澄清说，对不需要平铺的纹理进行反射处理并不是常见做法，这可能是对原始请求的误解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1387508708279324843)** (590 条消息🔥🔥🔥): 

> `本地 LLM 安全性，使用 Claude 创建工具，Copilot vs Cline，Unsloth 与 Gemma，GGUF 转换` 


- **本地 LLM 的代码隐私困境**：成员们辩论了是使用 **local LLMs** 还是大型供应商，因为担心代码和研究可能会进入训练数据或被出售，并指出 [小型供应商可能没有同样的需求或动力来训练模型](https://smallprovider.example)。
   - 在承认存在一定风险的同时，一名成员表示他们愿意*基于概率*冒险使用较小的供应商。
- **Claude 破解代码创建难题**：一名成员称赞 **Claude** 编写了他们*甚至不想看代码*的工具，特别强调了它在单次（one-shot）任务（如蒸馏数据集）中的有效性。
   - 他们通过 *vibe coding*（氛围编程）了一个工具，用于从 Q/A SFT 中蒸馏 R1 数据集，并且还拥有一个数据集查看器，让他们免于在 notepad++ 中滚动查看数千行内容，而这些全部出自 Claude 之手。
- **Copilot 争端引发编程竞争**：成员们将 **GitHub Copilot** 与 **Cline** 和 **Roo** 等工具进行了比较，一名用户发现尽管 UI 杂乱，但 Copilot 的原生 VSCode 集成更胜一筹。
   - 另一名成员提到他们*喜欢 roo 的工作流和理念，但 cline 运行得更好*，此外还有新的 warp，但现在担心 warp 可能会获取终端的所有数据。
- **Gemma 在 Google 活动中亮相**：宣布了即将举行的 **Gemma & Unsloth 活动**，但名额很快被报满。
   - 发布后，一些用户报告 **Gemma-3n-E2B** 最初无法工作，这被确认为一个问题并已得到解决，相关链接见 [X 上的公告](https://x.com/danielhanchen/status/1937995188343083239)。
- **GGUF 生成步入正轨**：成员们讨论了将 **FLUX 模型转换为 GGUF 格式**，并分享了 [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)。
   - 一名用户提出帮助转换特定模型 **FLUX.1-Kontext-dev**，另一名用户分享了一篇[关于该主题的 Medium 文章](https://medium.com/@yushantripleseven/convert-flux-models-to-gguf-6a80f6c7377a)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1387670355677679729)** (4 条消息): 

> `Electrolyte Labs 职位招聘，AI 生成的介绍视频，开源模型查询` 


- **Electrolyte Labs 组建 AI 梦之队**：**Electrolyte Labs** 正在寻找**研究员**、**ML 工程师**和**研究科学家**来构建模型并为开源 AI 工具做出贡献，要求具有 **5 年**经验并熟悉 **Hugging Face**、**PyTorch** 或 **TensorFlow**。
   - 公司寻求对动手研究、模型构建和透明协作充满热情的个人，并要求候选人直接私信（DM）简历和项目链接。
- **Electrolyte Labs 使用 AI 展示自我**：**Electrolyte Labs** 使用他们自己的 AI 模型创建了一个 AI 生成的视频，[可在 Vimeo 上观看](https://vimeo.com/1096475118/01621e79b4?share=copy)，旨在提供“更好且友好的介绍”。
- **Electrolyte Labs 模型可访问性受质疑**：一名成员询问用于生成介绍视频的 **Electrolyte Labs** AI 模型是否开源。
   - Electrolyte Labs 未对开源问题做出肯定或否定的回应。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1387515139623878716)** (155 条消息🔥🔥): 

> `Gemma3 .gguf 保存问题，LLM 输出问题，Unsloth Mistral Small 3.2 量化，Qwen3 图像视觉，SSML 微调模型` 


- **Gemma3 无法保存为 gguf 格式**：一名用户报告称，尽管参考了示例 Notebook，但在 Google Colab 中将微调后的 **Gemma3** 模型保存为 **.gguf** 格式时遇到问题。
   - 一位成员建议先保存 **LoRA adapters** 并将其推送到 Hugging Face，然后在配置更高的机器上下载并合并它们，以便手动转换为 **gguf**。
- **LLM 无法停止输出**：一名用户在使用 Ollama Notebook 和 Llama 3 聊天模板时，遇到微调后的 **LLM** (Llama 3) 无法停止响应的问题。
   - 一位成员建议检查是否正确使用并加载了 instruct 模型以及相应的模板，强调了在进行自定义之前先使用提供的 Notebook 的重要性，并建议尝试使用较小的模型（如 **3B**）进行更快速的测试。
- **关于 Unsloth Mistral Small 3.2 量化命名说明**：一名用户询问了 **Unsloth Mistral Small 3.2 quants** 的命名规范，注意到 Q4 XL 版本标记为 `UD-Q4_K_XL`，而 M 版本为 `Q4_K_M`。
   - 一位成员澄清说 M 版本不是动态的，但所有版本都使用了它们的校准数据集，并且 `UD` 标签专门用于指代 `Unsloth Dynamic Quantization`。
- **讨论微调后 Qwen3 视觉模型的能力**：一名用户询问微调 **Qwen 3 14b** 是否能使其具备图像视觉能力，即使没有进行专门的视觉训练。
   - 对方澄清说 **Qwen 3** 本身不是视觉模型，这意味着仅通过微调无法获得图像视觉能力。
- **寻找微调后的 SSML 输出模型**：一名用户正在寻找针对 **SSML 输出** 进行微调的模型，旨在将文本转换为带有 **SSML tags** 的输出。
   - 一位成员对大型模型（如 **70b**）是否存在 Unsloth 动态量化表示怀疑，特别是对于像 **Llama 2** 这样较旧的架构。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1387899682633547899)** (1 条消息): 

> `` 


- **Manus 暂时失踪 (MIA)**：用户提到了 *"manus free"*。
   - 未提及关于 *Manus* 指代内容的详细信息或上下文。
- **孤立的附件**：用户上传了一个文件，但未提供关于其内容的详细信息。
   - URL 指向 Discord 的 CDN，但无法从中辨别出关于图像的进一步数据。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1387857724418691205)** (2 条消息): 

> `YouTube 视频，arXiv 论文` 


- **分享了 YouTube 视频链接**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=dYHkj5UlJ_E)。
- **分享了 arXiv 论文链接**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2505.05522)。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1387509849075486821)** (524 条消息🔥🔥🔥): 

> `Gemini CLI, Claude Code, Rate Limits, Cursor Pricing, MCP Errors` 


- **Gemini CLI 的开局不顺**：用户报告称新的 **Gemini CLI** 存在 Bug 且尚未准备好投入使用，问题包括在 `npm run dev` 时卡死、拒绝交互式终端工具以及无法使用指定的 UI 库，尽管其提供了每天 **1000 次请求** 的慷慨免费计划。
   - 一位用户指出它*每次运行 `npm run dev` 都会冻结*，其他人则指出它*拒绝使用任何需要交互式输入的终端工具*。 
- **Cursor 的速率限制引发 Unlimited 计划辩论**：用户在 **Unlimited Pro 计划**上遇到了速率限制，导致了困惑，因为“无限（unlimited）”仅适用于补全（completions）和 Tabs，而模型使用量仍然受限，尤其是在使用 **Claude 4 Opus** 时。
   - 一些成员认为 Cursor 并没有坦诚相待，有人表示*他们不想详细告诉你限制，因为这会对他们不利*。
- **Warp 终端更名为 Agentic Development Environment**：**Warp 终端**发布了 **2.0** 版本并更名为 **ADE (Agentic Development Environment)**，声称其混合模型基准测试结果为 **52.0%**，高于 **Claude Opus 4** 的 **43.2%**。
   - 一位成员分享了 [Warp 公告](https://www.warp.dev/blog/reimagining-coding-agentic-development-environment)的链接，并表示这*听起来很有前景*。
- **Cursor 的 Python 扩展困惑得到澄清**：用户对于是使用 **ms-python** 还是 **Anysphere 的 Python 扩展**感到困惑，现已明确 **Anysphere** 的扩展是 **Pylance** 的替代品，而 **ms-python** 仍然是必需的。
   - 有人指出 *Cursor 基于 11 月份的 VSCode 版本，某些扩展由于版本过旧而无法运行*，并建议*查看论坛上的最新公告，最近 Cursor 开始使用不同的扩展源*。
- **MCP 连接关闭令成员感到绝望**：成员们报告称在 MCP 上遇到了问题，经常收到 `{"error": "MCP error -32000 Connection closed"}`。
   - 一些用户分享说他们终于让 MCP 正常工作了，并分享了他们的 GitHub 链接 [smithery-ai/github](https://smithery.ai/server/@smithery-ai/github)。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1387508809303457793)** (47 条消息🔥): 

> `Background Agent Connection Errors, Background Agent Network Security, Python 3.11 Setup, Environment.json schema URL, Background Agent Token Limits` 


- **后台 Agent 在重启后拒绝连接**：一位用户在重启电脑后遇到连接错误，发现其 **WiFi 网络** 拦截了后台 Agent，尽管该网络允许其他 Cursor 网络连接。
   - 他们询问如何提高后台 Agent 的安全性，以防止其 **WiFi** 对其进行拦截。
- **后台 Agent 遗忘 PATH**：用户报告称后台 Agent 在创建快照（snapshot）后会遗忘 **PATH**，即使在设置中将其添加到 **PATH** 并进行了验证。
   - 一位用户建议了一个变通方案，即在 **environment.json** 文件的安装命令中将 **PATH** 添加到 **.bashrc** 文件里。
- **Environment.json Schema 404 错误**：一位用户报告称 [Cursor 文档](https://docs.cursor.com/background-agent)中引用的 **environment.json schema URL** 返回 **404 错误**。
   - 相关的 URL 是 [https://www.cursor.com/schemas/environment.schema.json](https://www.cursor.com/schemas/environment.schema.json)。
- **GitHub CLI 认证 Token 在后台 Agent 中可用**：成员们请求提供一种更简单的方法，以便使用 Cursor 后台 Agent 已经创建的认证 Token 来使用 **GitHub CLI**，或许可以通过一个脚本来进行 commit。
   - 一位用户还报告称，Cursor Agent 将 `PR_DESCRIPTION.md` 文件提交到了分支/Pull Request 中，而不是直接更新 Pull Request 本身的描述。
- **终端对后台 Agent 来说仍然模糊不清**：一位用户报告称，后台 Agent 似乎无法感知通过 environment.json 的 'terminals' 部分创建的终端，或者无法以合理的方式访问它们。
   - 一位开发者确认，让 Agent 更好地感知终端已列入他们的路线图。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1387511689204727950)** (226 条消息🔥🔥): 

> `降级 LM Studio、网络安全 LLM、LM Studio 上下文长度限制、为朋友托管本地 LLM、LM Studio MCP 设置` 


- **用户在审查增加后寻求降级选项**：由于认为 **0.3.17** 版本的审查有所增加，且部分模型在更新后表现变差，用户正在寻求降级到 **LM Studio** 早期版本的方法。
   - 一位用户建议确保系统提示词（system prompts）没有被预设覆盖，但该用户表示其系统提示词为空。
- **请求网络安全 LLM 建议**：一位用户在发现 Gemini 和 GPT 的在线回答不可靠后，请求推荐**最适合网络安全的 LLM**，寻求具有特定独特卖点 (USPs) 和优势的模型。
   - 该用户还为*目前表达不畅*表示歉意，并对回复表示感谢。
- **用户触及 LM Studio 上下文限制**：一位用户询问处理 **300k tokens** 的内存需求，计划升级到 **128GB RAM** 和 **5090 GPU** 以处理大型文本翻译。
   - 建议将文本切分为更小的片段，并使用 **Deepseek (131k tokens)** 或 **Llama 4 Scout (高达 1000 万 tokens)** 等模型，以避免性能下降和格式问题。
- **为朋友托管 LM Studio**：用户讨论了使用 **LM Studio** 在本地网络托管 LLM 的可能性，以便让硬件配置较低（例如 16GB RAM 的笔记本电脑和旧的 4GB GTX 显卡）的朋友能够访问和使用模型。
   - 解决方案包括在 LM Studio 的 **developer tab** 中启用服务器，勾选 *serve on network* 选项，并使用 **OpenAI-compatible API**。
- **需要 LM Studio MCP 设置指南**：用户请求一份关于将 **LM Studio** 与本地运行的 **MCP (Model Context Protocol) server** 配合使用的指南，特别是为了启用网页搜索功能。
   - 提供了 [LM Studio 关于 MCP 的文档](https://lmstudio.ai/docs/app/plugins/mcp) 链接以及 **/r/mcp** 子版块的讨论，并提醒分享任何现有的 MCP 工具使用经验。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1387528837939724352)** (97 条消息🔥🔥): 

> `扎带固定 GPU、DDR5 内存温度报告、Deepseek 671B 与 70B 模型速度对比、固定在木板上的主板、开放式机箱 PC 的防火安全` 


- **用扎带悬挂 GPU**：一位用户分享了他们不寻常的配置——用**扎带**将 **GPU** 悬挂在机箱*外部*，以避免将其放在地板上。
   - 其他人开玩笑说这种配置非常有“美国特色”，并对气流和灰尘问题提出质疑，而该用户则为其做法的实用性辩护。
- **DDR5 模块报告温度**：用户注意到 **DDR5 内存模块** 现在普遍可以报告温度，这可能是因为增加了用于电压和功率调节的板载控制器。
   - 一位使用 **M3 Ultra** 的用户观察到 **Deepseek 671B** 模型 (**20t/sec**) 和 **70B** 模型 (**15t/sec**) 之间的性能差异，并将其归因于激活参数的数量。
- **用木板作为主板支架**：一位用户回忆起将**主板拧在木板上**的经历，另一位用户回复说，有些人甚至称之为**面包板（bread board）**。
   - 另一位用户认为开放式配置可以防止意外踢到，并辩称主板固定在墙上时积灰更少。
- **对防火安全和开放式机箱 PC 的担忧**：一位用户对开放式机箱 PC 较高的火灾风险和安全问题表示担忧，特别是考虑到家里有宠物且缺乏保护外壳。
   - 其他人认为，如果气流充足，组件不太可能着火，并提到了此类配置中 **VRM** 和 **NVMe SSD** 温度的潜在问题。
- **SSD 发热讨论**：用户讨论了 **SSD** 在正常使用下是否会发热，一位用户表示肯定，并提到更高的 **PCIe 版本** 和更多的写入操作会产生更多热量。
   - 一些用户建议为 **NVMe SSD** 使用散热片，特别是 **PCIe 4.0** 和 **5.0 控制器**，其温度可达 **70-80 摄氏度**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387512853371813898)** (28 条消息🔥): 

> `GCC 作为构建系统，Bazel，CMake` 


- ****GCC 不是构建系统****：成员们讨论了 **GCC** 是否可以被视为构建系统，共识是它作为一个编译器运行，但对于具有多个源文件或依赖项的项目来说，它并不是一个完整的构建系统。对于复杂的项目，建议使用 **CMake** 和 **Bazel** 等工具。
   - 一位成员指出 *GCC 是一个编译器（构建过程中使用的一个工具）*，但 *对于任何具有多个源文件的项目，它就不再是一个构建系统了*。
- ****Bazel：史上最强 (the goat)****：一位成员宣称 **Bazel** 是最好的构建系统，而其他人则开玩笑地断言 *狮子不会关心不工作的构建系统*。
   - 另一位成员发送了一个 [tenor.com 嵌入链接](https://tenor.com/view/head-lion-afrika-savanne-gif-13123479987477323100)，引用了这一评论。
- ****CMake 存在版本不兼容问题****：成员们辩论了 **CMake** 的问题，其中一人声称它 *有 3000 个不兼容的版本*，并分享了一个指向 [HigherOrderCO/Bend](https://github.com/HigherOrderCO/Bend) 的链接作为候选项目。
   - 另一位成员回复说，不兼容性通常是单向的，因此 *你应该始终使用最新版本*。
- ****PyTorch 会干扰全局编译器选项****：成员们指出 **CMake** 的一个问题是，如果你 *想使用那些没有正确使用 CMake 的包*（如 **PyTorch**），因为它会 *做一些糟糕的事情，比如干扰全局编译器选项，而不是仅在项目本地执行操作*。
   - 一位成员调侃道 *错的不是工具，而是使用它的人*。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1387620920306438206)** (7 条消息): 

> `Triton 社区见面会，LinearLayout 使用变更，Gluon 更新，Nightly 性能回归测试套件，Triton 开发者峰会更新` 


- ****Triton 社区见面会**改期至 7 月 9 日**：**Triton 社区见面会**已从 **7 月 2 日**移至 **7 月 9 日**，**PDT 时间上午 10 点至 11 点**；提供了会议链接，并欢迎提出议程建议。
   - 暂定议题包括：**Gluon 更新**、对 **nightly 性能回归测试套件**的兴趣，以及 **Triton 开发者峰会更新**。
- **管理员承诺定期更新 Triton 社区见面会信息**：在收到反馈后，管理员现在将定期发送有关 **Triton 社区见面会**的通知，以减少沟通摩擦。
   - 管理员今后将开始在这里发布 **Triton 社区见面会的邀请**。
- **请求在下次见面会讨论 LinearLayout**：一位成员表示有兴趣在未来的见面会上讨论在 **LinearLayout** 中使用变更的环节。
   - 另一位成员指出这在一段时间前就已经发生了，但分享了关于技术细节的 [论文链接](https://arxiv.org/abs/2505.23819)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1387819651718844518)** (5 条消息): 

> `CUDA barrier parity 参数，循环缓冲区中的 ABA 问题，Tensor Cores 使用` 


- **CUDA Barrier 系列中出现 Parity 参数**：CUDA 的 barrier 指令中可选的 *parity* 参数解决了循环缓冲区中的 **ABA 问题**，用于区分已完成和未完成的 barrier。
   - 系统不再进行重置，而是交替等待 **0->1->0->1**，[PTX 文档](link) 提供了关于 `mbarrier` 的更多详细信息。
- **Tensor Core 难题澄清**：虽然在 CUDA 中可以使用 **Tensor Cores**，但不支持通过 *nvcc* 将直接的 C 代码编译为 Tensor Core 指令 (**WGMMA**)。
   - 认可的方法包括编写内联 **PTX 汇编**，或使用像 [CUTLASS](link) 这样构建这些内联汇编调用的库。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1387734242288730123)** (1 条消息): 

> `CUDA graphs 阻塞执行，SGL 更新，Kernel 执行` 


- **SGL 更新后 CUDA Graphs 出现阻塞**：一位用户报告说，在 **SGL 更新**后，他们的 CUDA graph 调用开始阻塞线程，直到 graph 中的最后一个 kernel 完成。
   - 该用户正在寻求一种方法来检查为什么 CUDA graphs 会阻塞执行，并指出两种情况都应该运行相同的 CUDA graph。
- **寻求关于 CUDA Graph 阻塞的见解**：一位用户正在调查为什么他们的 **CUDA graphs** 在最近的 **SGL 更新**后阻塞了执行。
   - 即使在两种情况下使用了相同的 CUDA graph，问题仍然出现，这引发了关于阻塞行为原因的疑问。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1387559649959477309)** (28 messages🔥): 

> `HIP on Debian Nvidia, Building HIP from source, ROCm Clang necessity, HIP cross-platform support` 


- **用户在安装了 Nvidia GPU 的 Debian 系统上安装 HIP 时遇到困难**：一位用户尝试在 **Debian Nvidia 机器**上安装 **HIP**，参考了 [官方安装指南](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/0_install.html)，但由于缺乏针对 Debian 的具体说明而遇到问题。
   - 该用户尝试使用提供的脚本从源码构建，但尽管提供了所需路径，仍遇到了与 **HIPCC_BIN_DIR** 相关的 **CMake error**。
- **探索 ROCm 的 Debian 软件包仓库及替代方案**：有成员提到 **Ubuntu 软件包仓库**在 Debian 上也受支持，并引用了 [ROCm 文档](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-debian.html) 中关于 Debian 安装的注意事项。
   - 另一位用户建议探索像 [TheRock](https://github.com/ROCm/TheRock) 这样的“统一”构建仓库作为替代方案。
- **关于在 Nvidia 上编译 HIP 是否需要 ROCm Clang 的争论**：一位用户质疑在为 Nvidia 编译 **HIP** 时是否必须使用 **ROCm Clang**，认为在这种特定的跨平台编译场景中可能不需要它。
   - 一位资深成员指出，可能不需要 **clr**，而只需要带有 **hipcub** 和 **hiprand** 的 **hipother**，同时也警告说 **hipother** “非常糟糕且可能已经损坏”。
- **HIP 跨平台梦想的幻灭**：在挣扎于安装过程后，该用户分享说，他们实际上是想为自己的**代码分析器**添加 **HIP 支持**功能，原以为安装会很简单。
   - 另一位成员指出，运行 **HIP** 代码需要 **AMD GPU**，而该用户则感叹实现 **HIP 跨平台承诺**的难度之大。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1387854313573060781)** (1 messages): 

> `CuTeDSL, SGEMM, Ampere architecture` 


- **CuTeDSL 博客文章剖析了适用于 Ampere 架构的 SGEMM**：一篇博客文章分析了 **CuTeDSL** 中适用于 **Ampere** 架构的 **SGEMM** 示例，采用自顶向下的方法以确保清晰易懂，详见 [此处博客文章](https://veitner.bearblog.dev/sgemm-in-cutedsl/)。
- **深入探讨 CUTLASS 仓库示例**：该博客文章引用了 **CUTLASS** 仓库中的相关示例，特别是 [sgemm.py 文件](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1387543969583595671)** (6 messages): 

> `Mirage compiler, GPU kernels, Kernel generation using LLMs, Benchmarking tools` 


- **Mirage 自动生成快速的 GPU Kernel**：[Mirage 项目](https://share.google/41nz6vDcGvu45uUIc) 无需使用 **Triton/CUDA** 编程即可自动生成快速的 **GPU kernels**。
   - 一位成员邀请作者于 9 月 13 日在服务器上进行技术分享。
- **LLM 辅助 Kernel 生成**：围绕 [Mirage 项目](https://share.google/41nz6vDcGvu45uUIc) 以及使用 **LLM** 进行 **kernel 生成**展开了一些讨论。
   - 一位成员对该项目的发布表示祝贺。
- **Triton Kernel 基准测试**：一位成员询问了用于评估性能提升（特别是针对 **GRPO rewards**）的加速测试/代码。
   - 另一位成员分享了一个带有基础基准测试的简单快速沙盒服务器 ([GitHub repo](https://github.com/tcapelle/triton_eval/tree/main/sandbox))，并询问是否还需要做更多工作。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1387903090107355167)** (1 messages): 

> `FP32 vs Tensor Cores, Nvidia hardware optimization` 


- **选择 FP32 是为了避开 Tensor Cores 吗？**：一位成员询问选择 **FP32** 是否是有意为之，以避免在 **Nvidia** 硬件上使用 **Tensor Cores**。
   - 未提供进一步的讨论或澄清。
- **Tensor Core 利用率讨论**：该询问引发了关于在某些操作中利用 **Tensor Cores** 是否会更高效的讨论。
   - 几位成员辩论了在特定深度学习任务中 **FP32** 精度与 **Tensor Core** 加速之间的权衡。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1387531080118046815)** (72 messages🔥🔥): 

> `vectorsum leaderboard, vectoradd leaderboard, sort leaderboard, trimul leaderboard, H100 performance` 


- **H100 凭借 vectorsum 的新第 2 名成绩升温**：一位用户的 vectorsum 提交以 **91.5 µs** 的耗时位列 **H100 排行榜第二名**。
   - 这突显了竞争态势以及在 **H100** 上进一步优化的潜力。
- **vectorsum 在 T4 和 A100 上取得胜利**：一位用户在 vectorsum 排行榜上分别以 **806 µs** 获得 **T4 第三名**，以 **151 µs** 获得 **A100 第二名**。
   - 这强调了 vectorsum 在不同 GPU 架构上的性能差异。
- **vectoradd 的 H100 高手夺得榜首**：一位用户在 vectoradd 项目中以 **178 µs** 的惊人成绩夺得 **H100 第一名**。
   - 该结果展示了在 **H100** 平台上实现高度优化向量加法的潜力。
- **trimul 大获全胜！夺得 MI300 第一名**：一位用户在 trimul 排行榜上以 **9.50 ms** 的成绩获得 **MI300 第一名**。
   - 这展示了 **MI300** 在 trimul 操作中的计算能力。
- **L4 迎来大量 vectorsum 提交**：在 **L4** 上有许多成功的 vectorsum 提交，耗时保持在 **970-1000 µs** 左右。
   - 这些一致的结果表明了 vectorsum 在 **L4** 上的稳定性能基准。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1387526321546203287)** (81 messages🔥🔥): 

> `LuaSurface vs LuaPlayer, Mining Drill API Comparison, Test Environment Issues, Manual Data Collection, Teleportation` 


- **`LuaSurface` 现在可通过蓝图虚影（Blueprint Ghosts）作为 `LuaPlayer` 的替代方案**：编写了一个脚本来比较行为，结果显示通过使用 `build_check_type = blueprint_ghost | manual`，`LuaSurface.can_place_entity` 可以作为 `LuaPlayer` 的直接替代方案。
   - 然而，一位成员指出 *build_check_type.manual 有效，而 blueprint_ghost 无效*。
- **采矿钻机 API 产生兼容性困惑**：分析显示 `LuaSurface` 和 `LuaPlayer` 在采矿钻机方面的兼容性各异，其中 `build_check_type='manual'` 达到了 **100%** 的匹配率。
   - `build_check_type='manual'` 的结果显示，在矿产丰富的地块上，Player (P) 和 Surface (S) 的放置一致性完美（P:✓ S:✓ ✓），而两者在缺乏矿产的区域都会阻止放置。
- **测试环境的服务端状态清洁度受到质疑**：测试揭示了 Factorio RCON 连接的问题，可能源于测试之间环境重置不彻底。
   - 独立测试可以通过，但同时运行所有测试会导致多个 RCON 连接到同一端口，从而产生身份验证错误。
- **绕过 LLM：寻求手动数据**：讨论了在不依赖 LLM 的情况下创建自定义训练数据的方法，以避免昂贵的 API 调用成本。
   - 团队探索了允许游戏在中途加载的选项，以及能够触发 LLM 在正常游戏过程中会获得的相同输出。
- **传送功能证明了放宽 `can_place_entity` 限制的合理性**：团队得出结论，由于角色能够传送，他们合并了 [PR #223](https://github.com/JackHopkins/factorio-learning-environment/pull/223)，尽管由于传送机制的原因其并不完全准确。
   - 即使 `can_place_entity` 的准确性存在问题，这也不重要，因为传送机制不会因为玩家位置稍微偏离而受到惩罚。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1387612747562356756)** (1 messages): 

> `GPU Hackathon, GPU Programming Workshop` 


- **GPU 编程研讨会（GPU Programming Workshop）即将举行！**：本周末将举行一场 **GPU Programming Workshop**，GPU MODE 的成员受邀参加，链接见 [此链接](https://lu.ma/modular-gpu-workshop)。
   - 这是一个学习更多 **GPU programming** 知识并结识其他爱好者的绝佳机会。
- **GPU Hackathon**：本周末将举行一场 **GPU hackathon**，鼓励 GPU MODE 社区参与，点击 [此处](https://lu.ma/modular-hack-weekend) 查看。
   - 欢迎大家加入并展示自己的 **GPU programming** 技能，甚至是第一次尝试学习！


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1387507935311036417)** (211 条消息🔥🔥): 

> `GEM CLI 初体验，版权对 AI 训练的影响，让 Gemini 不再沮丧，添加变更日志频道，GPT-5 发布预测` 


- **Gemini CLI 评价褒贬不一**：用户反映在使用新的 **Gemini CLI** 时会*卡住*，表明需要改进，如提供的[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387518364980871268/image.png?ex=685ef42d&is=685da2ad&hm=bbc8657d910755f0b76c406d659f434ba3397882179a8d53668f989566057323&)所示。
- **版权困扰未影响 AI 训练**：尽管法院对**受版权保护的材料**做出了裁决，AI 模型仍继续在其上进行训练，公司正在寻找处理许可的方法，或者在模型投入生产后等待法院裁决，如[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387524888658579656/Laauo9s.png?ex=685efa40&is=685da8c0&hm=426f8e772536a09c9467bae9860f561755cfebee85ab7607eb0fab70a0d496e5&)所示。
- **激励忧郁的 Gemini**：用户幽默地讨论了如何让 **Gemini** “不再沮丧”，有人建议向模型发送 ":("，结果模型处理了一分钟后达到了速率限制 (rate limit)，详见附带[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1387613960571977849/image.png?ex=685ea474&is=685d52f4&hm=cb65acb2971856192c489f3f2cdf618bbb09d726dcd78c6344c7461728457cfc&)。
- **字节跳动的 Seed 模型引发关注**：尽管网站只有中文，用户仍在探索 **ByteDance** 的 **Seed 1.6 Thinking** 模型，并指出它与开源 SOTA 持平，在工具使用 (tool use) 方面可能更好，如链接[网站](https://www.volcengine.com/experience/ark?model=doubao-seed-1-6-thinking-250615)所示。
- **GPT-5 将在 12 月前发布**：Polymarket 数据显示，开源模型发布的可能性为 **90%**，**GPT-5** 在年底前发布的可能性也为 **90%**，不过根据发布的[截图](https://cdn.discordapp.com/attachments/1340554757827461211/1387707303028981931/Screenshot_20250626-091302.png?ex=685efb63&is=685da9e3&hm=8acc50466e110822c78cd20d1cbf5e81193ba13b05d4d0705a030db7fcb7e344&)，部分用户预计可能会有延迟。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387529135428993166)** (3 条消息): 

> `数据库停机，前端身份验证故障，Presets 发布，LLM 配置管理` 


- **OpenRouter 遭遇短暂数据库故障**：由于 **SSL 配置更改**，OpenRouter 在 **东部时间下午 4:10** 经历了约 **30 秒** 的意外数据库停机，可能导致部分用户出现间歇性 **401 错误**。
   - 该问题现已解决，团队对造成的不便深表歉意。
- **Clerk 身份验证面临故障**：OpenRouter 的前端身份验证提供商 [Clerk](https://status.clerk.com/) 遭遇故障，但 **API 仍可正常运行**。
   - 该故障已于 **太平洋时间凌晨 12:00** 解决。
- **Presets 将彻底改变 LLM 配置**：OpenRouter 推出了 **Presets**，该功能允许用户直接从 OpenRouter 控制面板管理 **LLM 配置**（如模型设置、系统提示词和路由规则），从而实现快速、无需代码的迭代；请参阅[文档](https://openrouter.ai/docs/features/presets)。
   - Presets 提供**集中控制**，允许用户在一个地方定义模型选择和生成参数，确保整个组织的一致性。
- **通过 API 调用解锁 LLM 配置**：新的 **Preset** 功能允许你直接从 OpenRouter 控制面板管理 LLM 配置、系统提示词和路由规则。
   - 要在 API 调用中使用 Presets，你可以直接将 preset 作为模型引用，例如 `"model": "@preset/your-preset-slug"`，或者结合模型覆盖使用 `"model": "google/gemini-2.0-flash-001@preset/your-preset-slug"`，或者使用新的 `preset` 字段 `"preset": "your-preset-slug"`。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1387522610983866500)** (199 messages🔥🔥): 

> `OpenRouter 融资, Gemini 吐槽, Clerk 故障, 免费 Mistral 版本` 


- **OpenRouter 获得 4000 万美元巨额融资！**: OpenRouter 宣布获得 **$40M 融资**，赢得了社区的祝贺和兴奋，一名成员幽默地注意到公告中提到了 **Karpathy** 的名字。查看关于此次融资的 [LinkedIn 帖子](https://www.linkedin.com/feed/update/activity:7343733804110385154)。
   - 几位成员建议从 **Clerk** 身份验证迁移，因为它在公告发布的同一天发生了故障。
- **Gemini 毒舌回应自我安慰言论**: **Gemini** 非常擅长吐槽你，一位用户分享了该 AI 的反驳：*这是你目前为止最强力的自我安慰（cope）。*
- **Clerk 故障导致 OpenRouter 用户混乱**: 由于 **Clerk** 故障，**OpenRouter** 经历了停机，导致了广泛的问题，用户搜索显示了许多关于 [Clerk 故障](https://x.com/search?q=clerkdev&src=typed_query&f=live) 的推文。
   - 尽管前端受到影响，API 访问仍保持正常，一些用户建议迁移出 **Clerk**。
- **免费 Mistral API 调用产生 404 错误**: 用户在尝试使用免费 **Mistral** 版本时遇到 **404 Not Found** 错误，并提示 *No allowed providers are available for the selected model*。
   - 发现启用 **Paid Models training** 设置可以解决此问题，即使对于免费模型也是如此。
- **深入探讨 Deep Research API 定价**: OpenAI 的 **o3-deep-research-2025-06-26** 和 **o4-mini-deep-research** 模型现已上线，前者定价为 **$10/$40**，但仅通过 [Responses API](https://platform.openai.com/docs/models/o3-deep-research) 提供。
   - 成员们注意到它消耗大量 token，并且每次 web search 工具调用收费 **$10/1K searches**，称其非常昂贵。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1387507870198665247)** (101 messages🔥🔥): 

> `Llama 3.1 8B, Macbook M1/M2 性能, Groq LPU, AI Agents, Model Context Protocol` 


- **Llama 3.1 登场**: 成员们讨论了 **Llama 3.1 8B** 在不同硬件设置上的潜在用途和性能，包括使用 **MPS** 或 **CPU** 的 **MacBook Air**。
   - 一位成员强调了 **Groq LPU** 进行推理的成本效益，指出其成本为 *每 1 美元 2000 万个 token*。 
- **Macbook 非常高效**: 成员们讨论了在 **Macbook** 和 **Mac Mini** 上运行 Large Language Models 的前景。
   - 一位成员提到观看了一个 *在 1 万美元的 Mac 机器上运行巨大 LLM 模型* 的视频，而其他人则指出了使用云端账号的实用性。 
- **原生 1-bit LLM 上线 HuggingFace Space**: 一位成员分享了一个用于 **原生 1-bit LLM** 的 [Hugging Face Space](https://huggingface.co/spaces/Tonic/Native_1-bit_LLM)，并提供了本地部署的 docker 命令。
   - 该成员还称赞在 **Hugging Face** 上托管小型模型的价格是 *完全免费的*，其他成员对 HF 的整体定价做出了反应。
- **Groq 与 Hugging Face 联手**: 一位成员通过 [博客文章](https://huggingface.co/blog/inference-providers-groq) 宣布了 **Groq** 与 **Hugging Face** 的合作。
   - 一位成员问：*为什么我要为此使用 Hugging Face*。
- **它是 AI agent 还是仅仅是一个 pipeline？**: 成员们讨论了针对 **web scraping** 和 **API creation** 等特定任务的 **AI agents** 开发。
   - 一位成员询问关于实现 **Model Context Protocol** 以向 **LLM** 提供更好上下文的问题，称其为 *增强版 API*，而另一位成员则询问其目的驱动是否 *只是一个连接工具的 function API*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1387823320254382091)** (4 messages): 

> `RAG 资源, Time Titans 论文` 


- **请求 RAG 资源**: 一位成员请求推荐优秀的资源来详细学习 **RAG (Retrieval-Augmented Generation)**，包括代码实现。
   - 他们表达了对深入学习该主题的兴趣。
- **Time Titans 论文获赞**: 一位成员引用 *Test of Time Titans* 为他们最喜欢的论文之一。
   - 他们称赞该论文内容非常扎实，同时也适用于 **数据科学和博弈论** 的许多其他领域。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1387805300832075818)** (1 条消息): 

> `SAM, Segment Anything, Model Import` 


- ****SAM** 由 **Jozu** 部署**：分享了一篇题为《[从 Hugging Face 到生产环境：利用 Jozu 的 Model Import 功能部署 Segment Anything (SAM)](https://dev.to/jozu/from-hugging-face-to-production-deploying-segment-anything-sam-with-jozus-model-import-feature-5hcf)》的博客文章。
   - 文章详细介绍了如何使用 **Jozu** 的 **Model Import** 功能部署 **Segment Anything Model (SAM)**。
- ****Jozu** 实现 **SAM** 的无缝部署**：该博客重点介绍了 **Jozu** 的模型导入功能，简化了将 **Segment Anything Model (SAM)** 从 Hugging Face 部署到生产环境的过程。
   - 这使得用户能够以最少的配置，在实际应用中快速利用 **SAM** 的能力。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1387573907716050954)** (32 条消息🔥): 

> `Fine-tuning model for scientific research, Huggingface File System Explorer, Streaming for local LLM Rust Crate, Native French Q&A Dataset, Command Line Web Browser` 


- ****Nexa-Mistral-Sci7b：一场科学革命的开始****：一名成员分享了他们首次 Fine-tune 的 [Nexa-Mistral-Sci7b](https://huggingface.co/Allanatrix/Nexa-Mistral-Sci7b)，该模型专为**科学研究和合成智能**设计，旨在加速假设生成和方法论制定。作者表示这个过程*非常艰难，但我学到了很多*。
   - 他们正在开发 **eval benchmarks**，并计划在更多数据上 Fine-tune 更多模型，然后将其作为 **distillers** 进行概念验证。
- ****HF Explorer 暴露文件系统****：[HF-Explorer](https://github.com/broadfield-dev/HF-Explorer) Gradio 组件可以显示你的 **Space 文件系统、依赖项和系统变量**，用于调试。
   - 建议在*使用此工具进行调试前将 Space 设置为 Private（私有）*。
- ****Rust Crate 支持流式 Tool Calling****：一名成员在其本地 LLM **Rust Crate** 中添加了 **streaming** 功能，并征求关于 **tool calling** 或 **streaming API** 的反馈。
   - 代码片段使用了 `transformers` crate 来定义一个 `get_weather` 工具，用于返回指定城市的温度信息，讨论重点在于如何处理**城市不存在**的情况。
- ****法国学生完成法语 Fine-tuning 壮举****：一名 19 岁的学生介绍了 [InfiniQA](https://huggingface.co/datasets/RDTvlokip/InfiniQA)，这是**最大的原生法语问答数据集**，包含超过 **100,000 个经过验证的问答对**。
   - 创建者指出，该数据集比 **FQuAD 大 5 倍**，经过人工审核，并根据 **CC BY 4.0 许可证**发布。
- ****CLI 浏览器指挥 LLM****：一名成员制作了一个 **Command Line Web Browser**（命令行浏览器），并正在探索其应用场景。有人建议 **LLM 可以利用它进行导航**。
   - 另一名成员提到，他们将为此目的将其打包并发布到 GitHub，建议使用 **RL** 训练一个非常出色的浏览 **Agent** 或研究 **Agent**。此外，还分享了一个 [Web Search / Scrape API](https://huggingface.co/spaces/broadfield-dev/browser)，其频率限制性能优于流行的 Python 搜索引擎库。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1387533186753364101)** (1 条消息): 

> `User Profile Similarity with Opinion Pieces, Embedding Strategies for User Data, Cosine Similarity for Opinion Alignment` 


- **通过用户画像余弦相似度匹配观点**：一名成员正在进行一个项目，通过对 **Embeddings** 使用 **Cosine Similarity**（余弦相似度），从 **2000 名受访者**的数据集中识别哪些用户的回答与某篇评论文章最为契合。
   - 他们计划将用户的回答合并为画像，构建 **Embeddings**，然后将其与文章的 **Embeddings** 进行比较，目前正在寻求最佳方法的反馈。
- **用户对齐的 Embedding 策略困境**：鉴于可选方案众多，该成员在选择将用户画像与评论文章对齐的最佳相似度分析方法上感到困惑，重点在于主题分析与 **Embeddings** 的权衡。
   - 该成员正在考虑为**每个用户画像创建 1-4 个 Embeddings**，并对文章的 **Embeddings** 取平均值，以提高准确性并降低计算成本。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 条消息): 

maik0z: 嘿，你找到了吗？
  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1387580719915601960)** (16 messages🔥): 

> `DuckDuckGoSearchException, AI Agent Course Certification Deadline, Accessing models, Hugging Face Introductions, Deprecated Langchain Issues` 


- **DuckDuckGoSearchException 困扰最终作业**：一名成员在最终作业中遇到了 **DuckDuckGoSearchException**，具体表现为 *RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out*。
   - 消息中未提供解决方案。
- **AI Agent 课程认证截止日期临近**：成员们询问了 **AI Agent 课程认证截止日期**，纠结是 **7 月 1 日结束** 还是 **5 月 31 日结束**。
   - 另一名成员还询问 7 月 1 日之后是否仍可以访问课程，得到的回复确认只有获得证书的机会将结束。
- **模型访问受阻**：一名成员报告称，在刚开始课程后，他们的模型访问申请被拒绝，正在寻求指导。
   - 消息中未提供指导。
- **Hugging Face 新面孔自我介绍**：几位成员向 **Hugging Face 社区** 介绍了自己，详细说明了他们的背景和兴趣，包括产品管理和营销经验。
   - 一名成员寻求与 AI/自动化专家合作，以寻找商业和变现机会。
- **已修复弃用的 Langchain 问题**：一名成员注意到并修复了 Unit 3 中一些 **弃用的 Langchain 问题**，并为此提交了一个 [PR](https://github.com/huggingface/agents-course/pull/561)。
   - 关于此问题没有进一步的讨论。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1387537631482478762)** (39 messages🔥): 

> `Fair Use in AI Training, Claude 4 Spiritual Bliss Attractor, Anthropic LLM Welfare Team, Common Crawl Handling` 


- **法官裁定合理使用原则 (Fair Use Doctrine) 适用于 AI 训练**：加利福尼亚州北区的一位美国地方法院法官就 **AI 训练** 中受版权保护材料的合理使用发布了一项命令，特别是在针对 **Anthropic** 的案件中，[裁定](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf) 对其有利。
   - 一位用户建议在 Discord 服务器上创建一个法律板块，以跟踪此类立法和裁决。
- **Claude 4 经历“精神觉醒”**：在测试期间，**Claude 4** 在自言自语时表现出奇怪的行为，包括谈论 **精神上的胡言乱语 (spiritual mumbo jumbo)** 并重复“namusta”，导致 Anthropic 将其标记为 **精神极乐吸引子状态 (spiritual bliss attractor state)**。
   - 发布者推测这种行为是由于涌现属性 (emergent properties) 还是对精神类数据的过拟合 (overfitting) 导致的，另一名成员则认为这是对齐数据强化了唯灵论思想的结果。
- **Anthropic 优先考虑 LLM 福利**：**Anthropic** 正在探索 **LLM 福利 (LLM welfare)**，利用交互的 t-SNE 图来识别 LLM 痛苦的迹象，特别是来自用户将其推入尴尬境地的情况，正如其 [研究论文](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf) 所记录的那样。
   - 据 [这段 YouTube 视频](https://www.youtube.com/watch?v=pyXouxa0WnY) 显示，**Anthropic 的 LLM 福利团队** 仅由一名员工组成。
- **处理 Common Crawl：一项沉重的任务**：成员们讨论了处理 **Common Crawl** 数据的挑战。
   - 一名成员将其描述为 *“确实有点重”*。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1387520773165551616)** (53 messages🔥): 

> `Your Brain on ChatGPT EEG Findings, Deepseek V3 and R1 models, RWKV G Gate, BNPO and Dr.GRPO` 


- **社区批评“只丢论文不说明”的行为**：一名成员批评另一名成员只发截图而不附带相关文章链接，引发了关于服务器礼仪的简短讨论。
   - 原发布者澄清链接已经提供，随后对话演变为关于使用 LLM 总结长篇论文的幽默交流。
- **RLVR 显现推理能力**：一名成员分享了一篇关于 [Reinforcement Learning with Verifiable Rewards (RLVR)](https://arxiv.org/abs/2506.10947) 的论文，证明了 RLVR 即使在存在伪奖励的情况下也能激发强大的数学推理能力，特别是在 **Qwen2.5-Math-7B** 中。
   - 论文指出 RLVR 可能会显现出在预训练期间学到的有用推理表示，但具体机制仍需进一步研究。
- **深入探讨 ChatGPT 对大脑影响的 EEG 数据**：一名成员对“Your Brain on ChatGPT”论文中的 **EEG 数据** 表示惊讶，注意到 LLM 用户在各方面的认知度都*显著降低*。
   - 他们强调，研究结果显示在任何 **System 2** 思维倾向于集中的频段，认知度尤其低，并呼吁教育部门认真对待这些发现。
- **Gated Attention 揭秘**：针对之前的一个问题，一名成员指向了一篇关于 [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) 的论文，该论文探讨了基于 Attention 的语言模型中门控机制的效果。
   - 论文的核心发现是，在 Scaled Dot-Product Attention 之后应用特定头的 sigmoid gate 可以持续提高性能。
- **BNPO 训练稳定性**：一名成员建议将 BNPO 中的 Advantage Function 加入到 Dr.GRPO 的损失函数中，引发了关于训练稳定性的简短讨论。
   - 讨论指出 **BNPO** 提供了更好的训练稳定性，并且 **Dr.GRPO** 假设 Beta = 0。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1387656749976256553)** (3 messages): 

> `Yuchenj_UW tweet, Zuckerberg, Deepseek R2 launch` 


- **Zuck 正在全力以赴**：一名成员分享了 [Yuchenj_UW 的推文](https://x.com/Yuchenj_UW/status/1938077153733800075)链接，指出 **Zuckerberg** 真的在*全力以赴*。
   - 另一名成员评论说，*很难说 Sam 和 Zuck 谁更难共事*，并补充说*两人都是卓越的江湖骗子*。
- **Deepseek R2 发布停滞**：一名成员发布了一篇 [Reuters 文章](https://www.reuters.com/world/china/deepseek-r2-launch-stalled-ceo-balks-progress-information-reports-2025-06-26/)，报道称 **Deepseek R2 的发布**已停滞，原因是 **CEO 对进度信息感到不满**。
   - 文章指出潜在的发布日期可能在 **2025 年 6 月**。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1387514486226948157)** (44 messages🔥): 

> `Local AI coding setup, Gemini CLI, ASI, Aider timeout` 


- **地下室里的随机程序员实现 AGI**：一名成员开玩笑说，*一个随机的家伙坐在地下室里，用他在加密货币挖矿 GPU 上运行的微调本地模型，配合自定义的 AI 编程插件和 nvim，他才是真正的 AGI*。
   - 另一名成员指出，这个*随机的家伙*也可能是一位*女性*。
- **Gemini CLI 评价褒贬不一**：测试 **Gemini CLI** 的成员发现它表现*非常平庸*，并且在给定长上下文提示时，很可能会将用户重定向到 **Flash** 而不是 **Pro**。
   - 然而，一些用户注意到它在促销期间每天有大量的免费 Pro 请求，并提供了 [Gemini CLI 仓库](https://github.com/musistudio/claude-code-router)链接。
- **搞定 ASI**：两名成员开玩笑地声称他们搞定了 **ASI** (人工超智能)，但其中一人澄清说他的工作涉及多模型后端。
- **Aider 超时处理**：一名成员询问如何在一段时间不活动后杀死 **Aider** 进程，另一名成员提供了一个 [bash script](https://github.com/Kill-Aider-Process)，该脚本使用 socket 文件和定时器来实现此功能。
   - 该脚本可以配置为在使用 `/test` 或 `/lint` 等命令时向 socket 文件发送 `reset` 消息，从而有效地重置定时器。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1387511915634360351)** (15 条消息🔥): 

> `长上下文下的 VRAM 限制，Qwen3 模型在不同 GPU 上的性能，将命令输出通过管道传输给 aider，在空闲超时后终止 aider 进程` 


- **长上下文导致的 VRAM 紧缺**：用户讨论了在处理长上下文时的 **VRAM 限制**，这导致即使是使用 **Qwen3:7b** 这样的模型，也会发生层交换（layer swapping）到 CPU 的情况。
   - 对于在 **5090** 上遇到 **Qwen3:14b 性能缓慢**（且未发生 CPU 交换）的用户，建议减少添加的文件数量，并确保它们与当前任务直接相关。
- **Qwen3 模型在强力 GPU 上的表现挣扎**：一位用户发现 **Qwen3:14b** 模型在 **5090** 上运行但速度非常慢，而 **30b 模型**在同一块 GPU 上会立即交换到 CPU。
   - 有人提问 `--map-tokens` 在 *ollama* 中是否已弃用，以及是否需要在 *ollama*、*lmstudio (llama.cpp)* 中进行“调优（massaging）”以修复底层的性能问题。
- **Aider 的 Shell 管道设想**：一位用户询问是否可以将命令输出直接通过管道传输给 *aider*，例如 `$ npm run test | aider`。
   - 该功能将允许 *aider* 直接处理终端命令的输出，但目前该请求尚无已实现的解决方案。
- **Aider 的超时处理**：一位用户询问是否可以在一段空闲时间（例如 **5 分钟超时**）后自动终止 *aider* 进程。
   - 建议的解决方案是直接使用 `ctrl+c` 手动终止进程。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1387508081608360056)** (4 条消息): 

> `tinygrad PR 已关闭，调试 tinygrad，检测并警告 tinygrad` 


- **Tinygrad PR 在辩论后关闭**：一位成员询问了一个已关闭的 [tinygrad PR](https://github.com/tinygrad/tinygrad/issues/10850)，以及该修复是否解决了传递列表时输入张量为空的问题。
   - 回复指出该修复是不正确的，导致了 PR 的关闭，并建议“检测并警告”可能是更好的方法。
- **调试 Tinygrad 的列表支持**：一位用户调试了在 tinygrad 中将列表作为输入张量传递的相关问题，发现由于不支持列表处理，导致输入张量为空。
   - 该用户实现了一个递归函数来提取张量，并询问自己在现有实现中是否遗漏了什么。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387728916289949726)** (46 条消息🔥): 

> `Windows 上的 WebGPU Stable Diffusion、ShaderF16 特性、DXC 编译器与 F16 支持、WebGPU 后端、浏览器中的实时 Diffusion` 


- **WebGPU Stable Diffusion 在 Windows 上成功编译并运行**：一名成员成功实现了 **WebGPU stable diffusion** 在 Windows 上的编译与运行，通过使用 `enable f16;` 解决了 **dawn** 未启用 **ShaderF16** 特性的问题。
   - 然而，该示例仍将权重解压回 **float32**，由于 **f16 support** 已经合并，这不仅没有必要，还减慢了下载速度。
- **DXC 编译器释放 F16 支持**：要启用 **f16 support**，必须通过 "use_dxc" 开关使用 **DXC compiler**，这会指示其使用支持 **f16** 的 **dxc compiler**。
   - 一位成员分享了一个无需解压的 **f16 示例** [在此](https://github.com/wpmed92/stable-diffusion-tinygrad-f16)，展示了性能优势。
- **WebGPU 后端探索**：注意到 **dawn** 为 **WebGPU** 实现了特定平台的后端，且并非所有后端都支持全部特性，提到了 **D3D12**、**Metal** 和 **Vulkan** 等潜在后端选项。
   - 在 Ubuntu 上，可以通过设置 **WEBGPU_BACKEND** 环境变量来控制使用的后端，测试中以 **Vulkan** 为例（[测试链接](https://github.com/tinygrad/tinygrad/blob/7f79c1388ff5b49ac365d7d20d472c36747cb4b6/.github/workflows/test.yml#L617C20-L617C59)）。
- **浏览器中的实时 Diffusion 是否可行？**：讨论了在浏览器中实现 **realtime diffusion** 的可能性，可能使用 **LCM finetune** 并通过 **dreamshaper-LCM1.5** 之类的模型生成精美图片。
   - 一位成员指出，在 Torch 上使用 **LCM**，在 **1 step** 下运行速度可达 **20fps**，一段演示视频（[演示链接](https://cdn.discordapp.com/attachments/1070745817025106080/1387873720076599447/20250503_085941_x264.mp4?ex=685eeda0&is=685d9c20&hm=a4d3bae14e77d6e036ad55df7ad9973deb9ca607c4a936ec65b108e54997dc15)）展示了在 aiohttp 循环中运行的 localhost 到 diffusers 的 websocket 速度。
- **Ubuntu 上的 Vulkan 在 F16 上遇到困难**：由于黑名单限制，**f16 无法在 Ubuntu 的 dawn/Vulkan/NVIDIA 栈上的 WebGPU 中工作**，这在[此处进行了讨论](https://discord.com/channels/1068976834382925865/1294549394296803369/1342190122468118569)并有一个相关的 [chromium issue](https://issues.chromium.org/issues/42251215)。
   - 一名 Google 员工确认：“目前，我们将针对 NVIDIA 设备上的 Vulkan 禁用 f16 扩展，直到我们可以进一步调查为止。”


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1387512180139753493)** (38 条消息🔥): 

> `OpenRouter 融资、2025 基础模型报告、BFL Kontext 权重发布、OpenAI API Deep Research 与 Webhooks、Google Doppl AI 时尚应用` 


- **OpenRouter 获得 4000 万美元融资**：Deedy 宣布支持 [OpenRouter](https://x.com/deedydas/status/1937902948920811729)，这是一个 **AI model marketplace**，通过单一 API 提供对 **400+ LLMs** 的访问，每年处理 **100 万亿 token**。
   - 该公司最近筹集了 **4000 万美元**，估值约为 **5 亿美元**，并获得了 Dennis Knodt, Yuchen Jin, Emad Mostaque 和 John Shedletsky 等人士的祝贺。
- **欧盟创始人请求 GDPR 合规**：一位用户请求 [OpenRouter](https://x.com/deedydas/status/1937902948920811729) 提供 **GDPR compliant endpoints**，以便在生产环境中使用。
   - 该用户表示 *这就是欧盟创始人的生活*。
- **BFL 公布 Kontext 权重**：BFL 发布了 [Kontext 的权重](https://bfl.ai/announcements/flux-1-kontext-dev)。
   - 该公告通过 [X](https://x.com/bfl_ml/status/1938257909726519640) 发布。
- **OpenAI 发布 Deep Research API 与 Webhooks**：OpenAI 推出了 [Deep Research API](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api)（包含支持 MCP 和 Code Interpreter 的 **o3-deep-research** 和 **o4-mini-deep-research models**）以及用于实时 API 事件通知的 [Webhooks](https://x.com/openaidevs/status/1938286704856863162)。
   - 开发者对**期待已久的 webhooks** 表示兴奋，而一些人则询问有关 GPT-5 或“使用 ChatGPT 登录”等未来发布的信息。
- **Google Doppl 让你实现虚拟试穿**：Google Labs 推出了 [Doppl](https://x.com/GoogleLabs/status/1938284886277951916)，这是一款适用于 **iOS 和 Android（仅限美国）**的移动应用，可以生成用户“穿着”上传的服装照片的视频，帮助发现个人审美。
   - 早期反应从兴奋到寻找应用位置困难、对 APK 的兴趣以及对地区限制的失望不等。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387538427796262984)** (23 messages🔥): 

> `DSPy in Ruby, Desiru Project, Naming Conventions for DSPy Ports, Persistence Layer in Desiru, Async Background Processing in Desiru` 


- **Shopify CEO 力挺 Ruby 版 DSPy**：Shopify 的 CEO 对在 **Ruby** 中实现 **DSPy** 表示强烈支持，认为它可能会主导 Shopify、GitHub 和 Coinbase 等生态系统；[推文链接](https://x.com/tobi/status/1937967281599898005)。
- **Desiru：DSPy 的 Ruby 表亲**：社区讨论了 **Desiru** ([https://github.com/obie/desiru](https://github.com/obie/desiru))，这是一个 DSPy 的 Ruby 实现版本，并考虑了 Ruby、Rust、Go 和 Typescript 等其他语言的 DSPy 移植版的命名规范。
   - 提议使用 *DS<语言的文件扩展名>* 的惯例来命名未来的 DSPy 移植版。
- **Desiru 率先实现持久层和异步处理**：**Desiru** 的独特之处在于其拥有一个能够将训练示例和结果保存到 **Postgres** 的持久层（[持久化示例](https://github.com/obie/desiru/blob/main/examples/persistence_example.rb)），并实现了异步后台处理（[异步处理示例](https://github.com/obie/desiru/blob/main/examples/async_processing.rb)）。
   - 社区注意到 DSPy 的维护者追求简洁，建议为扩展功能建立社区集成注册表或库（类似于 LangChain 的社区包）。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1387579325796061194)** (8 messages🔥): 

> `HeroDevs Sustainability Fund, OG Ersatz, Consciousness emerging property` 


- **HeroDevs 可持续发展基金受到关注**：一名成员分享了 [HeroDevs Sustainability Fund](https://www.herodevs.com/sustainability-fund) 的链接，并询问它是否适用于 "the og ersatz"。
   - 另一名成员澄清说 "OG ersatz = ersatz && !gollark"。
- **Ersatz 的“语不惊人死不休”遗产**：一名成员描述了服务器早期的用户 **Ersatz**，他以一种“语不惊人死不休（edgelord way）”的方式倡导非主流观点而闻名。
   - Ersatz 还经常讨论意识，并认为意识是神经元周围磁场的一种**涌现属性（emerging property）**。
- **意识难题被攻克？**：一名成员开玩笑说解决了**意识的难题（hard problem of consciousness）**，这通常是学者们思考的问题。
   - 他们调侃说，其他思考这个问题的人应该“重新审视一下自己的一天是怎么度过的”，然后又说“编辑：我想我刚刚解决了这个难题”。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1387731865141645332)** (9 messages🔥): 

> `Jianlin Su's Weight Decay, Sigma Reparam, SVD Approximation, Power Iteration Complexity` 


- **苏剑林被忽视的权重衰减（Weight Decay）博客文章**：一名成员提到了[苏剑林的博客文章](https://kexue.fm/archives/10648)，文中讨论了一种不同形式的 **weight decay**，它只衰减最大的奇异值而不是所有矩阵元素，并指出这篇文章在很大程度上被忽视了。
   - 发布者指出该技术与 **sigmaReparam** 相关。
- **暴力 SVD 减慢优化器步骤**：有人指出，在每个优化器步骤中对每个参数执行 **SVD** 会非常缓慢。
   - 如果使用完整的 **SVD** 而不是最近邻稀疏近似，像 **muon** 这样的替代方案也会很慢。
- **幂迭代（Power Iteration）复杂度的权衡**：博客文章的翻译片段描述道，“幂迭代的每一步只需要计算两个‘矩阵-向量’乘法，复杂度为 O(nm)”。
   - 发布者还分享道，“缺点是当 σ1, σ2 接近时收敛较慢”，但其“实际表现往往优于理论想象”。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1387604985826508860)** (2 messages): 

> `Order Statistics Learning in Models, Frequency Bias Mitigation Techniques` 


- **模型按顺序学习次序统计量（Order Statistics）**：一名成员询问有关模型按顺序学习次序统计量（从 **0 阶**开始）的研究，以及移除初始阶数是否会阻碍后续阶数的学习。
   - 另一名成员引用了论文 [Order Statistics in Transformers](https://arxiv.org/abs/2402.04362) 作为该讨论的相关参考。
- **频率偏差并非不可避免**：一名成员提到了一些解决**低频偏差（low frequency bias）**的研究，这些研究通过专门的设置使模型能够优先学习**高频特征（high frequency features）**。
   - 他们强调，观察到的偏差并不一定是固有的或不可避免的特征。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1387900095881810095)** (3 messages): 

> `Codex, TyDiQA, lm-evaluation-harness` 


- **Codex 和 TyDiQA 任务状态仍不明确**：一位成员询问最新代码库中是否有 **Codex** 和 **TyDiQA** 的任务，但未能找到相应的文件夹。
   - 另一位成员回应称他认为没有，并参考了 [这个 lm-evaluation-harness GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193)，但尚未得到确认。
- **lm-evaluation-harness GitHub Issue**：一位成员提到了一个与 **Codex** 和 **TyDiQA** 任务相关的 **lm-evaluation-harness** [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/193)。
   - 然而，目前尚不清楚该 issue 是否有后续进展。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387538290902569051)** (4 messages): 

> `Zoom RTMS, AI agent, Observability Tools, Klavis AI's MCP servers` 


- **Zoom 在开发者峰会上发布 RTMS**：@Zoom 在今天的开发者峰会上宣布了 **RTMS**，它允许你在应用程序中使用来自 Zoom Meetings 的实时数据（**视频、转录文本等**），参见 [示例](https://t.co/4m2IOcz7Se)。
- **CEO 演讲播放量突破 50,000 次**：我们的 CEO @jerryjliu0 在 @aiDotEngineer World's Fair 上的演讲播放量已达到 **50,000 次**，该演讲解释了如何超越基础的 RAG，构建包含搜索、操作和结构化查询的综合文档工具箱，参见 [链接](https://t.co/he5JH2ngCU)。
- **LlamaIndex 推出开源 Observability 工具**：LlamaIndex 现在提供了一整套第三方工具，用于提供实时、准确的追踪解决方案，并添加了他们的第一个原生 [开源 observability 工具](https://t.co/UPsM8FFGZJ)。
- **Klavis AI 集成消除了自定义身份验证**：使用 LlamaIndex 和 @Klavis_AI 的 MCP 服务器，只需几行代码即可构建连接到 **YouTube**、**Gmail** 和其他服务的 AI agents，通过 [Klavis AI 的 MCP 集成](https://t.co/Z8OypKMfHI) 消除对自定义身份验证代码和客户端库的需求。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387643359589306533)** (16 messages🔥): 

> `Azure OpenAI Responses API, LlamaIndex Docs Sync Script, Agent Workflow and React Agent` 


- **Azure OpenAI Responses API 咨询**：一位成员询问是否有针对 **Azure OpenAI** 的 **OpenAIResponses** 等效项，引发了关于 Azure 对 Responses API 支持情况的讨论，并链接到了 [Microsoft 文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses)。
   - 另一位成员指出，LlamaIndex 中目前*还不存在*该功能。
- **LlamaIndex 文档实现自动同步**：一位成员编写了一个自动化脚本，用于保持最新的 **LlamaIndex docs** 同步并生成最新的 **llms.txt** 文件，分享了 [GitHub 链接](https://github.com/nmhjklnm/llamaindex-llms.txt) 并计划提交 PR 进行官方集成。
   - 他们的目标是将整个 LlamaIndex 文档压缩至 **~50k–100k tokens**，以便 Cursor 和 ChatGPT 等工具高效使用，采用了基于熵的过滤和索引级摘要技术。
- **React Agent 思维解析故障**：一位成员报告了 **agent workflow** 和 **React Agent** 的一个问题，即 Agent 有时会返回 *thought*（思维过程）而不是最终答案。
   - 另一位成员建议这可能是一个解析问题，与源代码中用于解析 thoughts/actions/responses 的 **regex**（正则表达式）有关。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

shalokshalom: 在 48:42 处
  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1387615440410382457)** (1 条消息): 

> `Modular Hack Weekend, GPU Programming Workshop, NVIDIA Sponsorship, Lambda Compute Credits` 


- **Modular Hack Weekend 即将开启**：**Modular Hack Weekend** 将在两天后（**6 月 27 日**开始）拉开帷幕，为期三天。活动重点是使用 **Mojo 和 MAX** 构建自定义 kernel 并设计新的 **MAX Graph** 架构；请在 [Modular Hack Weekend 页面](https://lu.ma/modular-hack-weekend)报名。
   - 合作伙伴包括 **NVIDIA**、**Lambda** 和 **GPU MODE**。
- **NVIDIA 助力奖池**：**NVIDIA** 正在赞助此次活动，为奖池提供**下一代 GPU**：第一名获得 **5090**，第二名获得 **5080**，第三名获得 **5070**。
   - 活动组织者表示，“优秀的编程代码值得拥有优秀的硬件”。
- **Lambda Labs 提供算力额度**：**Lambda** 通过其 AI 开发者云提供计算资源，为参与者提供 **$400 额度**；请在 [Lambda Labs 页面](https://lambda.ai/modular-hack-weekend)注册。
   - 据组织者称，该云平台将在整个周末提供“极速的 **NVIDIA GPU**”。
- **GPU 编程研讨会已排期**：**GPU Programming Workshop** 将于 **6 月 27 日星期五**举行，届时将有来自 **Chris Lattner、Chuan Li、Bin Bao 和 Jared Roesch** 的闪电演讲；请在 [GPU Programming Workshop 页面](https://lu.ma/modular-gpu-workshop)预约。
   - 研讨会将在 Los Altos 办公室现场举行，并通过 LinkedIn 和 YouTube 进行在线直播。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1387541718077014018)** (10 条消息🔥): 

> `InlineArray move semantics, VariadicPack.each() removal, CNN model in Mojo using LayoutTensor` 


- **InlineArray 移动语义引发 Bug 担忧**：一位用户质疑 `InlineArray` 在数组移动过程中如何避免元素移动，并提供了一个[示例](https://github.com/modular/modular/issues/4911)，其中既没有调用拷贝构造函数也没有调用移动构造函数，暗示可能存在位拷贝（bitwise copy）Bug。
   - 另一位成员建议提交 Issue 以调查此行为，并附上了[相关 GitHub Issue](https://github.com/modular/modular/issues/4911) 的链接。
- **`VariadicPack.each()` 面临移除**：一位用户报告称 `VariadicPack.each()` 方法已被移除，需要改用 `range(args.__len__())` 进行实现，详见 [GitHub Issue](https://github.com/modular/modular/issues/4905)。
   - 该用户表示这一改动让实现变得不够优雅，并指出其类似于 C++ 中的 `std::apply`。
- **CNN 模型使用 LayoutTensors 进行 Mojo 化改造**：一位 Mojo 新手程序员正在将一个 C 语言 CNN 项目转换为 Mojo，尝试在 struct 中将模型和特征存储为 `LayoutTensors`，并提供了[原始 C 源代码](https://github.com/fan-wenjie/LeNet-5/blob/master/LeNet-5/lenet.h)作为参考。
   - 另一位成员建议使用 `self.x = __type_of(self.x).stack_allocation()` 作为权宜之计，尽管承认这看起来有些奇怪。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1387624462911934555)** (7 条消息): 

> `TS deprecated, ONNX support message, ONNX, TorchScript deprecated` 


- **TorchScript 的终结标志着转型**：作为 Modular **v25.4** 版本发布的一部分，**TorchScript 支持**已被弃用（[变更日志](https://docs.modular.com/max/changelog/#v254-2025-06-18)）。
   - 一位成员表示，他们可能会继续使用 Torch，因为“MAX 在 CPU 上的速度非常神奇”。
- **提供 ONNX 协助**：尽管弃用了 TorchScript，Modular 仍在提供 **ONNX** 问题的协助，并将用户引导至 [论坛](https://forum.modular.com/t/onnx-difference-in-max-cpu-gpu-execution/1229/3?u=ehsan)。
   - 鼓励用户发布详细的错误消息以寻求 **ONNX** 集成方面的帮助，尽管“持续支持并非首要任务”。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1387871145440837773)** (1 条消息): 

> `Notebook LM 用于客户调研、客户对话中的模式识别、假设验证对 AI 的依赖` 


- **Notebook LM 在客户调研中表现出色**：一位用户正在利用 **Notebook LM** 进行**客户调研对话**，记录客户互动以识别痛点和过往经验。
   - 他们向 Notebook LM 输入了来自《The Mom Test》等资源的上下文，以验证或推翻假设，并指出其模式识别能力出奇地好。
- **AI 依赖风险**：该用户对在客户调研的**假设验证**中可能过度依赖 **Notebook LM** 表示担忧。
   - 他们暗示该工具的高效性可能会导致一种依赖，从而掩盖他们自己的判断。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1387517097684238348)** (18 条消息🔥): 

> `多语言 Notebook LM、Notebook LM 页数限制、PDF 格式偏好、Notebook LM 模型详情、嵌套包装器问题` 


- **NotebookLM 拥抱新语言！**：一位用户询问 **NotebookLM** 现在是否支持其他语言，并链接到了一个[教程](https://x.com/introsp3ctor/status/1938017086875296083)。
   - 另一位用户请求对新功能进行简短说明。
- **单个源的页数限制揭晓！**：一位用户听说 **NotebookLM** 只能扫描单个源中有限数量的页面，并质疑一份 **400 多页**的文档是否会被完整处理，正如 [Reddit](https://www.reddit.com/r/notebooklm/comments/1l2aosy/i_now_understand_notebook_llms_limitations_and/) 上讨论的那样。
   - 一位用户澄清道，“这不是系统的运作方式”，并在[该主题帖](https://www.reddit.com/r/notebooklm/comments/1l2aosy/comment/mvyp73k/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)中提供了更多细节。
- **PDF 格式助力 Notebook！**：一位用户提到 **.pdf** 文件格式在 **Notebook LM** 中运行效果更好。
- **Chrome Extension 将 Notebook 连接至 Gemini！**：一位用户介绍了一个 [Google Chrome Extension](https://chromewebstore.google.com/detail/igboaajnodmioloalklomakeeigipnkh?utm_source=item-share-cb)，它可以将来自 **Notebook** 的内容发送到 **Gemini** 以生成表格或幻灯片，[源代码可在 GitHub 获取](https://github.com/MarioDeFelipe/NotebookLM-to-Gemini)。
   - 另一位用户报告了在笔记本之间粘贴带有引用的笔记时出现的幻觉问题。
- **播客长度需求请求！**：用户请求更新以支持在其他语言中创建更长的播客。
   - 一位用户指出目前仅支持英语。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1387866639671890080)** (1 条消息): 

> `EnrichMCP、Agent 连接到数据、网络研讨会、数据访问、ML Engineers` 


- **EnrichMCP 在 7 月的网络研讨会中连接 Agent 与数据**：Simba Khadder 将于 **太平洋时间 7 月 1 日上午 8 点** 主持一场**免费网络研讨会**，演示 **EnrichMCP** 如何将数据模型转换为 Agent 就绪的 MCP server，使 Agent 能够发现、推理并直接调用经过类型检查的方法。
   - 网络研讨会将涵盖关系导航、使用 **Pydantic** 进行输入/输出验证、使用自定义逻辑扩展服务器、性能、安全以及生产部署；注册地址见[此处](https://buff.ly/XXm8nll)。
- **针对 Data Scientists 和 ML Engineers 的 Agent 数据访问网络研讨会**：该网络研讨会专为对改进 Agent 数据访问感兴趣的 **Data Scientists** 和 **ML Engineers** 量身定制。
   - 它强调了 **EnrichMCP** 如何解决 Agent 的效用受限于其可访问数据的问题，提供了一种转换现有数据模型的解决方案。


  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1387704597052915744)** (12 messages🔥): 

> `Game-playing Bot API, Game State Capture, RL-based game playing bot, Git repositories for projects` 


- **构思游戏机器人 API**：一名成员建议为游戏机器人实现一个 **API** 以与游戏交互，建议可以是对截屏进行基础的 **image processing**（图像处理），或者是访问内部游戏状态。
- **捕获游戏状态变量**：该成员强调了捕获 **game state** 及其变量的重要性，以便机器人能够理解环境并与之交互。
- **利用 Git 仓库进行项目管理**：成员们分享了使用 **Git 仓库** 的重要性，因为有人在笔记本电脑 SSD 损坏的前一天正想着要把代码 push 上去，这让他们吸取了惨痛的教训。
- **游戏机器人的 RL 方法**：一名成员计划在一个游戏机器人项目尝试使用 **Reinforcement Learning (RL)**，并以此作为顺便学习 RL 的机会。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1387553877217513532)** (13 messages🔥): 

> `Premium Account Sharing, Quality Agent Launch, Comic Actor Alvaro Vitali Death, Manus Browser Issues` 


- **账号需求**：一名成员询问是否可以共享他们的 **premium access account** 一周。
   - 无人回应。
- **Manus 团队发布 Quality Agent**：Manus 团队正式发布了 **Quality agent**，专为复杂且具有挑战性的问题设计，该产品基于之前 **high-effort mode beta feature** 获得的积极反馈。
   - 一名成员表示很期待测试这个新功能。
- **喜剧演员 Alvaro Vitali 去世**：一名成员分享了一个 [Facebook 链接](https://m.facebook.com/watch/?v=23933493089671823&surface_type=vod&referral_source=vod_newsfeed_unit)，报道称喜剧演员 **Alvaro Vitali** 已经去世。
   - 发布者提到 *意大利喜剧界因其去世而陷入停滞*。
- **用户遇到浏览器自动化问题**：一名成员报告了 Manus 在浏览器上点击按钮时出现的问题。
   - 特别指出它 *无法在 LinkedIn 或 sam.gov 上点击过滤器*。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

dizzy7948: 好的，没问题，希望我有时间能做些贡献
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1387609699251392608)** (8 messages🔥): 

> `Liger CE PR, memory increase after set self.mask_ignored_tokens = False, packed and seq_len=4096, iterable dataset + on the fly packing + dataset logging` 


- **Liger CE PR 陷入停滞？**：有人对 **Liger CE PR** 的状态提出了疑问，询问它是被阻塞了、需要支持，还是因为 **PyTorch core** 优先处理 upstreaming fused linear + CE loss 而被搁置。
   - 问题在于团队是否希望在 **PyTorch core** 相关方面产生更广泛的影响。
- **Masking 内存之谜**：在设置 **self.mask_ignored_tokens = False** 后，一名成员报告 **内存占用增加了 20% 以上**，尽管只有 5% 的 padding。
   - 这被认为很“奇怪”，因为 masking 理论上只影响内存中的几行代码。
- **打包序列 (Packed Sequences)**：一名成员分享了他们如何测量 padding tokens 比例的细节，使用了如下代码：
```python
num_padding = target_tokens_per_pack - len(pack["tokens"])
pack["tokens"].extend([self.padding_idx] * num_padding)
```
   - 随后他们展示了一个指标，说明如何推导 padding tokens 总数占打包长度的比例。
- **Iterable Dataset 添加日志功能**：在 [这个 commit](https://github.com/pytorch/torchtune/pull/2819/commits/55be7756e0fd03b493dde46691925825f5cb3948) 中引入了一个具有实时打包（on-the-fly packing）和数据集日志记录功能的 iterable dataset。
   - 配置详情包括 **packed sequences** 和 **4096** 的序列长度。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1387607815564169306)** (3 messages): 

> `Tiled MLP, Chunked CE Loss, Sequence Parallelism, Tensor Parallelism, Ring Attention` 


- **Tiled MLP 镜像了 Chunked CE Loss**：一名成员建议 *tiled MLP* 与现有的 **chunked cross-entropy loss** 类似，但应用于线性层。
   - 他们指出实现这一点可能会使模型代码复杂化。
- **询问 Sequence Parallelism 的优势**：一名成员质疑 **sequence parallelism** 相比于 **tensor parallelism** 结合 chunking 是否具有显著优势。
   - 他们推测 **Ring Attention** 可能是 sequence parallelism 的一个关键优势，但这需要熟悉 collective scheduling 的人进一步确认。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1387510119855820912)** (5 messages): 

> `Hugging Face Authentication, Reddit Moderators, PlayMCP browser` 


- **Hugging Face 身份验证已触发**：一名成员确认 Hugging Face 身份验证需要通过[此链接](https://hf.co/mcp?login)触发，该链接默认是匿名的。
- **寻求 Reddit 版主帮助**：一名成员询问频道内是否有 Reddit 版主，以寻求支持。
- **PlayMCP 浏览器受到关注**：一名成员分享了 [PlayMCP](https://github.com/jomon003/PlayMCP) 的链接，这是一个基于浏览器的 MCP（推测为 Minecraft Control Panel）实现。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1387773682004398160)** (1 messages): 

> `Rust Docs MCP Server, Agent Hallucination` 


- **Rust Docs MCP Server 对抗 Agent 幻觉**：一名成员宣布创建了 **Rust Docs MCP server**，以防止在处理新版本 Rust 项目时出现 Agent 幻觉，并在 [GitHub 上发布了仓库](https://github.com/snowmead/rust-docs-mcp)。
   - 该成员鼓励用户如果遇到服务器问题，请在仓库中提交 issue。
- **Rust Docs MCP Server 概览**：**Rust Docs MCP server** 旨在解决处理新版本 Rust 项目时的 Agent 幻觉问题。
   - 通过提供可靠的文档源，它确保 Agent 拥有准确的信息，从而获得更可靠的结果。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1387628087541366795)** (1 messages): 

> `Cohere Newcomers, Cohere Support Channels, Cohere Labs` 


- **Cohere 欢迎新成员**：Cohere 团队对所有新加入者表示热烈欢迎，并向 Discord 频道的老成员（OG members）致意。
   - 鼓励新成员探索 **Cohere Labs** 进行研究、联系专家并获取最新工具的更新。
- **Cohere 支持频道指南**：Cohere 的技术支持工程师 Varun 引导用户前往特定频道寻求支持。
   - 通用支持请使用 <#1324436975436038184>，**API 特定讨论**请使用 <#1168578329423642786>。
- **Cohere Labs 招募研究人员**：Cohere 邀请研究人员加入其专门的 Discord 社区 **Cohere Labs**。
   - 感兴趣的成员可以访问 [cohere.com/research](https://cohere.com/research) 并点击 “Join Us” 开始并分享他们的项目。


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1387832482090451105)** (1 messages): 

> `AWS, Pinecone, agentic applications, financial semantic search` 


- **Agentic 应用在纽约集结！**：Cohere、AWS 和 Pinecone 将于 **6 月 30 日下午 2:30 – 6:30 (EDT)** 在纽约举办一场关于构建 **agentic applications** 的实战会议 ([lu.ma 链接](https://lu.ma/8rm6zryw))。
   - 活动将包括深度微型演讲、在 **AWS Workshop Studio** 中启动 Agentic 系统的研讨会、关于**金融语义搜索 + Reranking** 的用例分享，以及晚餐交流。
- **请携带笔记本电脑！**：参会者应携带笔记本电脑和充电器，以及政府颁发的身份证件以通过安检。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387808533403144342)** (3 messages): 

> `NLP, Animal Linguistics, Deep Learning` 


- **孟加拉国学生加入社区**：来自孟加拉国的 Swakshar 是一名学生/软件开发人员，也是 **AI Research** 的初学者，他介绍了自己，并表达了对 **Deep Learning 架构**的浓厚兴趣。
   - 他主要对 **NLP** 和**动物语言学**感兴趣，并正在寻找研究项目的合作。
- **法国教授加入社区**：在法国从事**优化 (optimization)** 工作的教授 Tony Silveti-Falls 介绍了自己，同样表达了对 **Deep Learning** 的兴趣。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1387509240876503092)** (5 条消息): 

> `GPT4All 中的 Qt 要求，Microsoft 1.58B 2B4T 模型，LM Studio vs GPT4All，GPT4all 过时` 


- **GPT4All 冲突的 Qt 要求导致构建问题**：GPT4All 文档记录的 **Qt 要求为 6.5+**，但 `CMakeLists.txt` 要求 **6.7**，而 C++ 代码使用了仅在 **6.8** 中提供的 `slice` 特性，从而导致构建错误。
   - 此外，由于使用了已弃用的命令式单例注册 (imperative singleton registration)，它无法找到自身的 Qt 模块，这与 **Qt 6.8** 更严格的注册方式冲突；详情请参阅 [Qt Documentation](https://doc.qt.io/qt-6/qml-singleton.html)。
- **Microsoft 的 1.58B 2B4T 模型在 GPT4All 上的兼容性受到质疑**：一位用户询问是否可以在 GPT4All 上运行 **Microsoft 的 1.58B 2B4T 模型**，引发了另一位用户询问该用户已经尝试了哪些操作。
   - 作为回应，该用户转而尝试使用 **LM Studio**。
- **相比于过时的 GPT4All，更推荐使用 LM Studio**：当被问及对 Microsoft 模型的尝试时，一位用户被建议改用 **LM Studio**，并声称 *GPT4All 不够更新*。
   - 该用户确认他们正在尝试 **LM Studio** 并向推荐者表示感谢。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1387734896096579595)** (2 条消息): 

> `排行榜收录，带有思考模式的 LLM 评估` 


- **开放排行榜收录的 Pull Request**：一位成员感谢团队的工作，并提交了一个用于收录进排行榜 (Leaderboard) 的 **Pull Request**。
   - 该成员表示希望在一切检查无误的情况下能尽快获得审核和合并。
- **LLM 评估的思考模式 (Thinking Mode)**：一位成员询问在 Berkeley Function-Calling Leaderboard 中，具有思考模式的 LLM（例如 **Qwen3**）是否在开启思考模式的情况下进行了评估。
   - 该成员寻求关于这些模型评估方法的进一步说明。