---
companies:
- openai
- hugging-face
- sambanova
- google-cloud
date: '2025-03-28T01:20:31.459275Z'
description: '**OpenAI** 发布了全新的 **GPT-4o** 模型，该模型具有增强的指令遵循、复杂问题解决以及原生图像生成能力。该模型在数学、编程和创意领域表现出更强的性能，并支持透明背景图像生成等功能。关于图像生成的内容过滤和政策讨论强调了在创作自由与防止危害之间取得平衡。**DeepSeek
  V3-0324** API 已在 **Hugging Face** 上线，由 **SambaNovaAI** 提供支持，其表现优于基准测试以及 **Gemini
  2.0 Pro** 和 **Claude 3.7 Sonnet** 等模型。**Gemini 2.5 Pro** 被推荐用于编程，而 **Gemini 3**
  可以通过全新的 Model Garden SDK 轻松部署在 Google Cloud Vertex AI 上。**Gemma 3 技术报告**已在 arXiv
  上发布。'
id: 461f40b5-4290-495d-8ed4-16c0908c6215
models:
- gpt-4o
- deepseek-v3-0324
- gemini-2.5-pro
- gemini-3
- claude-3.7-sonnet
original_slug: ainews-not-much-happened-today-3156
people:
- abacaj
- nrehiew_
- sama
- joannejang
- giffmana
- lmarena_ai
- _philschmid
title: 今天没发生什么事。
topics:
- instruction-following
- image-generation
- content-filtering
- model-performance
- api
- coding
- model-deployment
- benchmarking
- model-release
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年3月26日至3月27日的 AI 新闻。我们为您检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord 服务器（230 个频道和 7972 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：757 分钟。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

ChatGPT 中推出了 [新的 4o 模型](https://x.com/OpenAI/status/1905331956856050135)，但目前还没有博客文章，除了发布推文外细节不多，因此没有太多可报道的内容。然而，你可以看到最近 [SOTA 模型之间的间隔时间正在缩短](https://x.com/swyx/status/1905422862833647768/photo/1)。


![image.png](https://assets.buttondown.email/images/8732511d-4823-4220-b9d9-858897182dd7.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**GPT-4o 与多模态模型**

- **OpenAI 的 GPT-4o** 迎来了重大更新，根据 [@OpenAI](https://twitter.com/OpenAI/status/1905331956856050135) 的说法，它增强了遵循 **详细指令** 的能力，能够处理 **复杂的计算和编程问题**，并提升了 **直觉和创造力**，同时减少了 **emoji** 的使用 🙃。此外，更新后的 `chatgpt-4o-latest` 现已在 **API** 中可用，计划在未来几周内将这些改进引入 API 中的日期版本模型，正如 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1905335104211185999) 所宣布的那样。
- **GPT-4o 的原生图像生成** 在指令遵循能力方面脱颖而出，[@abacaj](https://twitter.com/abacaj/status/1905075484892836308) 指出目前没有任何产品能与之媲美。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905067166350884941) 强调了 **GPT-4o** 生成的图表在构图、文本生成和整体流程方面令人印象深刻，并特别指出这些元素是在无需明确定义的情况下生成的。
- **关于图像生成内容过滤器的担忧** 也被提及，[@nrehiew_](https://twitter.com/nrehiew_/status/1905206276742611362) 指出 **OpenAI** 的过滤器允许通过了一张令人惊讶的图像。
- **初始示例至关重要**：[@sama](https://twitter.com/sama/status/1905069374035411209) 强调在介绍新技术时，对展示的初始示例进行了仔细考量。
- **创造性自由与潜在危害**：OpenAI 模型行为负责人 [@joannejang](https://twitter.com/joannejang/status/1905341734563053979) 分享了在制定 **4o 图像生成** 政策时的思考与细微差别。她讨论了 OpenAI 如何从敏感领域的全面拒绝转向更精确的方法，重点在于防止现实世界的伤害，旨在最大化创造性自由的同时防止真实伤害，并保持谦逊，承认有很多未知领域，并随时准备根据学习到的经验进行调整。
- **带有透明背景的图像生成** 是 GPT-4o 的一个酷炫功能，据 [@giffmana](https://twitter.com/giffmana/status/1905407013103747422) 称，这对于创建各种素材非常有用。
- **GPT-4o 的性能提升** 相较于之前的版本非常明显，根据 [lmarena.ai](https://twitter.com/lmarena_ai/status/1905340077339034104) 的数据，LMSYS Chatbot Arena 显示其在数学、困难提示词（Hard Prompts）和编程类别中均有提升。
- **模型质量感知**：[@abacaj](https://twitter.com/abacaj/status/1905290667338653980) 发现 Google 的模型似乎永远处于预览或实验模式，等它们完全可用时，另一个模型已经超越了它们。它们让你窥见可能性，然后由其他人真正实现它。

**DeepSeek 与 Gemini**

- **DeepSeek V3-0324 APIs** 正在被 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1905279539250676065) 在 10 个 API 中进行追踪，包括 DeepSeek 的第一方 API 以及来自 Fireworks、DeepInfra、Hyperbolic、Nebius、CentML、Novita、Replicate 和 SambaNova 的产品。它现在也通过 [@SambaNovaAI](https://twitter.com/_akhaliq/status/1905350698797334860) 在 **Hugging Face** 上可用，速度达到 250+ t/s —— 为全球最快。它横扫了 MMLU-Pro (81.2) 和 AIME (59.4) 等基准测试，表现超越了 Gemini 2.0 Pro 和 Claude 3.7 Sonnet。
- **Gemini 2.5 Pro**：根据 [@_philschmid](https://twitter.com/_philschmid/status/1905272290931061111) 的说法，如果你目前正在使用 Claude，推荐在编程（coding）时使用 Gemini 2.5 Pro。
- **Gemini 3**：根据 [@_philschmid](https://twitter.com/_philschmid/status/1905149450059727226) 的说法，得益于全新的 Model Garden SDK，只需 3 行代码即可将 Gemini 3 部署到 Google Cloud Vertex AI。
- **Gemma 3 技术报告**：根据 [@_philschmid](https://twitter.com/_philschmid/status/1905287223567671383) 的说法，该报告现已发布在 arXiv 上。它为 Gemma 系列轻量级开放模型引入了多模态（multimodal）成员，参数规模从 10 亿到 270 亿不等。该版本引入了视觉理解能力、更广泛的语言覆盖范围以及更长的上下文——至少支持 128K tokens。
- **GoogleDeepMind Gemini** 的新函数调用（function calling）指南已由 [@_philschmid](https://twitter.com/_philschmid/status/1905256740762829172) 发布，该指南使用了新的 uSDKs，并包含针对 Python、JavaScript 和 REST 的多个完整示例。
- **TxGemma**：基于 GoogleDeepMind Gemma 模型构建，能够理解并预测小分子、化学品、蛋白质等的特性。根据 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1905274926665208043) 的说法，这可以帮助科学家更快地识别有前景的靶点，预测临床试验结果，并降低整体成本。

**AI 安全与可解释性**

- **Anthropic 的可解释性研究**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905354818451112352) 重点介绍了这一研究。根据 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1905303838417973669) 的说法，新的可解释性方法允许他们追踪模型“思考”过程中的步骤。
- **Anthropic** 正在招聘研究人员，共同开展 AI 可解释性（interpretability）方面的工作，消息来自 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1905303862883365370)。
- **Anthropic 经济指数（Economic Index）**：根据 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1905381798676197819) 的说法，该指数正在发布第二份研究报告，并分享了更多基于匿名化 Claude 使用数据的资料集。
- **AI 安全狂热（AI Safety Fads）**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1905283397301452858) 指出，每隔一两年就会出现一个新的 LW/AF 狂热（如 inner optimizers, ELK, Redwood's injury classifier, SAEs），由于 LW/AF 的封闭性和资金集中化，这些趋势往往比学术界更加剧烈。

**AI 工具与框架**

- **LangChain**：现在为使用 LangChain 或 LangGraph 构建的应用提供完整的端到端（E2E）OTel 支持，实现了统一的可观测性（observability）、分布式追踪，并能够将 trace 发送到其他可观测性工具，消息来自 [@LangChainAI](https://twitter.com/LangChainAI/status/1905315130134401133)。
- **LangGraph BigTool**：LangChain 展示了其在配合本地模型（通过 @ollama）使用超过 50 个工具时的可靠性。
- **LlamaCloud**：可以用作 MCP 服务器，允许用户将最新的数据引入其工作流，作为任何 MCP 客户端使用的工具，正如 [@llama_index](https://twitter.com/llama_index/status/1905332760614764810) 所演示的那样。
- **Cohere 的 Command A**：这是一款能力强大且高效的模型，仅需 2 个 GPU 即可运行。根据 [@cohere](https://twitter.com/cohere/status/1905301761193377964) 的说法，它针对现实世界的 Agentic（智能体）和多语言任务进行了优化。
- **Keras**：为了庆祝原始版本发布 10 周年，Keras 推出了全新的主页，消息来自 [@fchollet](https://twitter.com/fchollet/status/1905391839055950032)。

**趋势与观点**

- **生成式 AI 与吉卜力工作室**：继 **GPT-4o** 发布后，根据 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905363176943759836) 和 [@aidan_mclau](https://twitter.com/aidan_mclau/status/1905385248726344023) 的说法，围绕使用 **Studio Ghibli**（吉卜力工作室）风格生成的讨论非常激烈。[@nearcyan](https://twitter.com/nearcyan/status/1905219687547621740) 讨论了未来几年即将推出的“后现实过滤阶段”，届时现实将变成人们想要的任何样子（吉卜力、宝可梦或指环王等），随着每个人找到自己真正渴望的东西，他们将被划分到由纯粹的美和艺术（对许多人来说是欲望）构成的、专为他们优化的私人花园中。
- **未来属于 ASI**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1905105940480917715) 表示，他们无法想象构建人工超级智能（**ASI**）只是为了赚钱、阻碍政治对手或进行 **EA**（有效利他主义）计算的套利。
- **对模型依赖的担忧**：[@nptacek](https://twitter.com/nptacek/status/1905200894645239988) 开始思考这里是否存在“构建者”与“整理者”的分歧，一些人想要某种整洁、有序的信息空间，而另一些人则完全适应于挑战事物存在边界的极限。
- **GPU 熔断**：[@sama](https://twitter.com/sama/status/1905296867145154688) 发推称他们的 **GPU** 正在熔断，因为人们太喜欢 **ChatGPT** 中的图像功能了。

**幽默/梗图**

- **热门作品的反面**：[@nearcyan](https://twitter.com/nearcyan/status/1905364110176649661) 定义了 AI 图像生成中所谓“热门作品（banger）”的对立面。
- **玩偶版 Jensen**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905346671753785451) 发布了来自玩偶版 **Jensen** 的友好提醒。
- **AGI = All Ghibli Images?**：[@_akhaliq](https://twitter.com/_akhaliq/status/1905111170056425728) 戏称 **AGI** 的意思是“全是吉卜力图像（All Ghibli Images）？”
- **Gary Marcus 的观点**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1905213421118980324) 调侃了 Gary Marcus 为 **ML** 社区提供的世界闻名、发人深省且具有创新价值的观点。
- **DeepMind AI 尝试打造热门作品**：[@sama](https://twitter.com/sama/status/1905062838802207045) 提到 CLAIDE 试图打造一个热门作品，但却说“一个人的垃圾是另一个人的垃圾”。

---

# AI Reddit 摘要

## /r/LocalLLaMA 摘要

**主题 1：DeepSeek V3 0324 在 Livebench 上超越 Claude 3.7，但存在幻觉问题**

- **DeepSeek V3 0324 在 Livebench 上超越 Claude 3.7** ([Score: 148, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1jl1yk4/deepseek_v3_0324_on_livebench_surpasses_claude_37/))：**DeepSeek V3 (0324)** 在 **LiveBench** 上取得了显著成绩，总排名第 10，超越了 **Claude 3.7 Sonnet**（基础模型），成为仅次于 **GPT-4.5 Preview** 的排名第二的非思考模型。这一表现表明，即将推出的 **R2** 模型可能会成为 AI 领域的强力竞争者。
  - **DeepSeek V3 的性能与幻觉问题**：用户报告称 **DeepSeek V3** 的幻觉率从 **4% 增加到 8%**，使其在某些任务中不够可靠。尽管它能根据幻觉提示给出正确答案，但用户感到惊讶，并建议在 **0.3** 的低温度（temperature）下运行以缓解此问题。
  - **与其他模型的比较**：**Gemini Pro 2.5** 在推理方面比前代有显著提升，引发了人们对 V3.1 到 **R2** 潜在增强的关注。**Anthropic** 和 **OpenAI** 面临高昂的 **API** 成本挑战，但 **OpenAI** 的多模态能力（尤其是图像生成）被视为一大优势。
  - **LiveBench 与模型更新**：人们对 **grok-3-beta** 从 **LiveBench** 中移除感到好奇。快速 **R1** 提供商可能需要时间来采用 V3，用户希望在即将到来的更新中（可能在 6 月）看到改进。


**主题 2：微软的 KBLaM：LLM 中的即插即用知识**

- **[微软开发了一种更高效的向 LLM 添加知识的方法](https://www.microsoft.com/en-us/research/blog/introducing-kblam-bringing-plug-and-play-external-knowledge-to-llms/)** ([Score: 426, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1jkzjve/microsoft_develop_a_more_efficient_way_to_add/)): **Microsoft** 开发了 **KBLaM**，这是一种旨在高效地将知识集成到 **Large Language Models (LLMs)** 中的新方法。该方法旨在通过增强其知识库来提高 **LLMs** 的性能和准确性，且不会显著增加计算需求。
  - **KBLaM 的局限性**：用户指出 **KBLaM** 是一个研究原型，尚未达到生产就绪（production-ready）状态，在与不熟悉的知识库一起使用时，提供准确答案的能力有限。这表明它目前并不优于已经投入生产的现有 **RAG** 系统。
  - **技术见解与挑战**：该实现需要大量资源，例如测试一个 **8B model** 需要 **A100 80GB**，这表明其计算需求很高。该方法涉及语言 token 注意到（attending to）知识 token，但反之则不然，这引发了关于潜在知识差距的问题，例如在没有实际应用知识的情况下理解概念。
  - **潜在应用与研究方向**：人们对从训练数据中提取事实知识是否能优化参数使用感兴趣，这可能使模型更智能或更高效。然而，共识是广泛的知识对于智能至关重要，需要更多研究来探索通用知识应用和专家级机器人。


**主题 3. Qwen Chat 上的新 QVQ-Max 功能增强了用户体验**

- **[Qwen Chat 上的新 QVQ-Max](https://i.redd.it/vlz8vwxsv9re1.png)** ([Score: 115, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1jlaeuw/new_qvqmax_on_qwen_chat/)): **Qwen Chat** 推出了强大的视觉推理模型 **"QVQ-Max"**，以及该系列中最强大的语言模型 **"Qwen2.5-Max"**。用户界面在一个深色、简洁的设计背景下突出了每个模型的能力，包括 **"Qwen2.5-Plus"**、**"QwQ-32B"** 和 **"Qwen2.5-Turbo"**。
  - **QVQ-Max** 和 **Qwen2.5-Max** 等其他模型引起了用户的兴趣，一些人计划将它们纳入测试计划，特别是在 **M3 Ultra** 等先进硬件上。
  - 一条评论指出该模型目前是**封闭（closed）**的，表明目前的访问权限有限或受限。
  - 人们对进一步的发展或发布充满期待，一名员工在 **Twitter** 上暗示预计在**周四**会有潜在的更新或增强。


**主题 4. 尽管拥有 ASIC 优势，Gemini 2.5 Pro 仍面临性能批评**

- **[Gemini 2.5 Pro 表现不佳](https://v.redd.it/7e5dflkqc6re1)** ([Score: 106, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1jkxhcu/gemini_25_pro_dropping_balls/)): 标题为 **"Gemini 2.5 Pro Dropping Balls"** 的帖子将 **Gemini 2.5 Pro** 与 **LLaMA 4** 进行了比较，但正文缺乏详细内容。标题暗示了 **Gemini 2.5 Pro** 潜在的问题或缺陷。
  - **Gemini 2.5 Pro vs LLaMA 4**：对于 **Gemini 2.5 Pro** 是否优于 **LLaMA 4** 存在怀疑，一些用户认为 **Grok** 只有在使用 **64** 采样时才能接近。然而，其他人认为目前没有模型（包括 **Claude**）能超越 **Gemini 2.5 Pro**。
  - **技术优势**：**Google** 使用自研 **ASICs** 为其带来了显著优势，而 **Meta** 和 **Amazon** 正分别尝试使用 **MTIA** 和 **Tranium** 赶超。**Google** 被认为领先了六到七代，这使得竞争充满挑战。


## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

> 正在调试我们的流水线问题，抱歉...

---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的摘要

**主题 1. Gemini 2.5 Pro：速率限制、定价和性能炒作**

- [**Cursor 用户遭遇 Gemini 2.5 Pro 速率限制瓶颈**](https://discord.com/channels/1074847526655643750)：**Cursor** 用户正面临 **Gemini 2.5 Pro** 极低的速率限制，部分用户报告仅在 *两次 API 请求* 后就触发了限流。目前的解决方法包括使用 **Open Router** 和个人 **AI Studio API keys**，也有建议称 **Google Workspace Business** 账户可能会解锁更高的限制。
- [**Cursor 为 Gemini 2.5 Pro 定价辩护，回应免费 API 质疑**](https://discord.com/channels/1074847526655643750)：**Cursor** 因对 **Gemini 2.5 Pro** 收费而面临用户抵制，因为该模型被认为可以通过 **Google AI Studio API** 免费使用。一位 **Cursor** 代表澄清说，收费是为了覆盖大规模使用的容量成本，因为 **Google** 在其使用级别上并未提供真正的免费层级。
- [**Gemini 2.5 Pro 在用户偏好上超越 Claude 3.5 Sonnet**](https://fxtwitter.com/LechMazur/status/1904975669081084273)：用户评估显示 **Gemini 2.5 Pro** 已超越 **Claude 3.5 Sonnet**，在故事生成类别的顶级排名中占据了 *3%*，而 **Sonnet** 则从 *74%* 骤降至 *18%*。用户称赞 **Gemini 2.5** 处理长上下文的能力非常出色，部分用户确认在 **AI Studio** 上拥有 **15K token** 的上下文窗口。

**主题 2. OpenAI 的 GPT-4o：更新、图像生成与政策转变**

- [**GPT-4o 再次更新，在 Arena 排行榜攀升**](https://x.com/lmarena_ai/status/1905340075225043057)：**OpenAI 的 GPT-4o** 获得了重大更新，目前在 Arena 排行榜上位列第 2，超越了 **GPT-4.5**，并在 Coding 和 Hard Prompts 类别中并列第 1。据报道，此次更新提升了指令遵循、问题解决、直觉和创造力。
- [**OpenAI 放宽图像生成政策，GPU 负载面临极限**](https://x.com/joannejang/status/1905341734563053979)：**OpenAI** 正在放宽其图像生成政策，从一味拒绝转变为防止现实世界的伤害，旨在提供更大的创作自由。**Sam Altman** 戏称由于图像生成的普及，*GPU 正在融化*，导致了临时的速率限制。
- [**Midjourney CEO 吐槽 4o 图像生成是“又慢又差”的梗**](https://x.com/flowersslop/status/1904984779759480942)：**Midjourney** CEO 对 **GPT-4o** 的图像生成不屑一顾，称其*又慢又差*，认为这只是一种融资手段和噱头（meme），而非严肃的创作工具。这一批评正值有关模型命名规范以及白宫删除一条包含吉卜力风格图像推文的讨论之际。

**主题 3. Model Context Protocol (MCP) 势头强劲但也面临挑战**

- [**OpenAI 和 Cloudflare 拥抱 Model Context Protocol (MCP)**](https://x.com/sama/status/1904957253456941061)：**OpenAI CEO Sam Altman** 宣布 **OpenAI** 产品（如 Agents SDK 和 ChatGPT 桌面应用）将支持 **MCP**，这标志着 **MCP** 普及迈出了重要一步。**Cloudflare** 现在也支持构建和部署远程 **MCP servers**，降低了准入门槛。
- [**Claude Desktop 在处理 MCP 提示词和资源时遇到困难**](https://github.com/yuniko-software/minecraft-mcp-server)：用户报告称，当 **MCP servers** 包含资源或提示词时，**Claude Desktop** 会陷入无限循环。一种解决方法是移除相关功能以防止 **Claude** 搜索这些元素，目前已发布 [修复程序](https://github.com/yuniko-software/minecraft-mcp-server)。
- [**LlamaCloud 集成为 MCP 服务器以提供实时数据**](https://twitter.com/llama_index/status/1905332760614764810)：**LlamaCloud** 可以作为 **MCP server** 运行，从而将实时数据集成到任何 **MCP client**（包括 **Claude Desktop**）的工作流中。这允许用户利用现有的 **LlamaCloud indexes** 作为 **MCP** 的动态数据源。

**主题 4. 本地 LLM 和工具更新：Unsloth, LM Studio 和 Aider**

- [**Unsloth 发布 Dynamic Quantization 和 Orpheus TTS Notebook**](https://x.com/UnslothAI/status/1905312969879421435)：**Unsloth AI** 发布了 **Dynamic Quants**（动态量化），以提高本地 LLM 的准确性和效率，并发布了 **DeepSeek-V3-0324 GGUFs**。他们还推出了 **Orpheus TTS notebook**，用于具有情感暗示和语音定制功能的人机交互式语音合成，在用户测试中表现优于 OpenAI 的 TTS。
- [**LM Studio 0.3.14 添加多 GPU 控制和优化**](https://lmstudio.ai/blog/lmstudio-v0.3.14)：**LM Studio 0.3.14** 引入了针对多 GPU 设置的细粒度控制，允许用户通过启用/禁用 GPU 和选择分配策略来优化性能。此次更新还包括“Limit Model Offload”模式，以提高稳定性和长上下文处理能力，并针对 **AMD GPUs** 进行了增强。
- [**Aider 的 /context 命令简化代码导航**](https://aider.chat/docs/usage/modes.html)：**Aider** 引入了 `/context` 命令，自动识别并添加给定请求的相关文件，提高了大型代码库中的工作流效率。然而，用户报告了通过 **OpenAI API compatibility layer** 使用 **Gemini** 时存在兼容性问题以及 CPU 使用率飙升的情况。

**主题 5. 图灵研究院动荡与开源 RL 系统 DAPO**

- [**艾伦·图灵研究院尽管获得资助仍面临大规模裁员和项目削减**](https://www.researchprofessionalnews.com/rr-news-uk-research-councils-2025-3-alan-turing-institute-axes-around-a-quarter-of-research-projects/)：**Alan Turing Institute (ATI)** 尽管最近获得了 **1 亿英镑** 的资金注入，仍计划大规模裁员并削减约四分之一的研究项目，引发了员工的反抗。在来自更广泛 AI 领域的竞争中，该研究院面临着生存危机。
- [**字节跳动开源 RL 系统 DAPO 低调现身**](https://github.com/BytedTsinghua-SIA/DAPO)：**ByteDance** 和 **Tsinghua AIR** 发布了 **DAPO**，这是一个开源的 Reinforcement Learning 系统，该系统似乎“悄无声息”地出现了。成员们分享了链接，强调了其在 RL 研究社区中的潜在重要性。
- [**灾难性过度训练论文挑战 LLM 预训练范式**](https://arxiv.org/abs/2503.19206)：一篇新论文提出了“**catastrophic overtraining**”（灾难性过度训练）一词，认为延长的预训练可能会降低微调性能，并使模型更难适应下游任务。论文指出，经过指令微调的 **OLMo-1B** 在延长预训练后表现更差。

---

# PART 1: Discord 高层级摘要

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 在 Gemini 2.5 Pro 的 API 限制下受阻**：用户报告在 **Cursor** 中使用 **Gemini 2.5 Pro** 时遇到了非常小的速率限制，低至 *两次 API 请求* 就会被限制。
   - 一些成员正在结合使用 **Gemini API**、**Requesty** 和 **Open Router** 来绕过这些限制，而其他人指出 **Google Workspace Business** 账户可能会解锁更高的限制。
- **Windsurf 和 Cursor 再次交锋**：**Windsurf** 和 **Cursor** 之间的争论仍在继续，**Windsurf** 因其在原型设计期间的“全上下文”而受到青睐，而 **Cursor** 则在修复 Bug 方面更受欢迎。
   - 一位用户抱怨 **Windsurf** 的 UI 样式与 **Cursor** 在页面间的一致性相比有所不足，而另一位用户则要求雇佣“官方 shitsurf 托儿”。
- **上下文窗口受 Cursor 限制？**：一位用户询问 **Cursor** 是否将 **Gemini 2.5 Pro** 的上下文窗口限制为 30K，尽管该模型宣传有 1M 的上下文限制。
   - 其他人补充说，虽然 Agentic 模型有 60k 的上下文窗口，但 **Claude 3.7** 有 120k 的上下文窗口，如果启用最高设置甚至可以达到 200k（尽管没有足够的数据确认这是否是 Vertex）。
- **Gemini 2.5 Pro 的定价困惑**：用户质疑为什么 **Cursor** 对 **Gemini 2.5 Pro** 收费，而通过 **Google 的 AI Studio API** 它是免费的，导致了不诚实的指控。
   - **Cursor** 代表澄清说，费用涵盖了处理该模型使用所需的容量，因为 **Google** 在 **Cursor** 的规模上不提供免费层级，并补充说价格与 **Gemini 2.0 Pro** 持平。
- **Cline 接管编码工作流**：一些用户计划放弃 **Cursor**，选择 **VSCode** 搭配 **Cline**（使用 **Gemini**）进行编码，并使用来自 **Fireworks** 的 **DeepSeek v3** 进行规划。
   - 一位用户表达了对 **Cursor Tab** 功能的怀念，并指出虽然大多数模型都在退步，但仍认可模型的实用性，但结论是“在 **RTX 4090** 级别上，本地模型都很糟糕”。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Bot 加入 Discord**：Perplexity 推出了 **Perplexity Discord Bot** 进行测试，直接在 Discord 频道内提供快速回答和事实核查，可通过艾特 <@&1354609473917816895> 或使用 **/askperplexity** 命令访问。
   - 测试者可以探索 **/askperplexity**、使用 ❓ 表情符号对评论进行事实核查，以及通过 **/meme** 创建梗图，反馈可发送至 <#1354612654836154399> 频道以进行改进。
- **GPT-4.5 从 Perplexity 悄然下架**：成员们注意到 **GPT-4.5** 从 [Perplexity.com](https://perplexity.com/) 的模型选择中消失了，可能是出于成本考虑。
   - Perplexity AI bot 澄清说，**GPT-4.5** 在科学推理、数学和编程方面通常优于 **GPT-4o**，而 **GPT-4o** 在通用和多模态使用方面表现出色。
- **Complexity 扩展增强 Perplexity**：用户讨论了 "Complexity" 扩展，这是一个旨在增强 Perplexity 功能的第三方插件。
   - Perplexity AI bot 指出，虽然该扩展的功能作为原生选项可能很有益，但集成决策取决于用户需求、技术可行性和产品路线图（product roadmap）的一致性。
- **MCP 服务器控制 Perplexity**：用户探索了利用 Model Context Protocol (MCP) 服务器（如 Playwright），允许 Perplexity 通过可用的 MCP 服务器控制浏览器或其他应用程序。
   - Perplexity AI bot 解释说，配置服务器与 Perplexity 的 API 配合使用可以实现自动化的浏览器操作和网页交互。
- **用户遇到 API 参数错误**：一位用户遇到了与 `response_format` 参数相关的错误，尽管账户内有额度，但这干扰了其应用功能并导致销售损失。
   - API 团队针对 `search_domain_filter`、`related_questions`、`images` 和 `structured_outputs` 等参数实施了错误处理，并澄清这些参数从未对非 Tier 3 用户开放；详见 [usage tiers](https://docs.perplexity.ai/guides/usage-tiers)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Gemini 2.5 Pro 初露锋芒**：成员们初步称赞 [**Gemini 2.5 Pro** 目前的编程表现相当不错](https://ai.google.dev/)，且*远好于*去年使用的任何 Gemini 模型。
   - 一位用户提到，它*将两个独立的动画组件合并为一个，实现了无缝的加载过渡*。
- **Manus 邀请码面临延迟**：用户对 **Manus 邀请码的等待时间** 表示沮丧，有些人已经等待了一周多。
   - 一位成员建议在注册和接收嵌入代码时使用*无痕模式或不同的浏览器*。
- **Discord UI 侧边栏神秘消失**：有用户报告其 **Discord 侧边栏**（特别是在 platform.openai.com 上）缺少图标、线程和消息。
   - 一位成员建议将外观设置更改为紧凑视图，或检查 PC 显示设置以解决尺寸问题。
- **WordPress 预发布站点在 Manus 中失败**：一位成员报告称，自上次维护以来，他们在 **Manus 中的 WordPress 预发布站点（staging site）** 反复出现故障。
   - 未找到解决方案。
- **Manus 与 N8N 协作良好**：成员们讨论了将 **Manus** 与 [**N8N**](https://www.n8n.io/) 或 Make.com 结合使用，以实现**流程自动化**和**工作流自动化**。
   - 一位成员正在构建他们的第一个 **N8N** 和 **Manus** 工作流，以连接全球的创意人士。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Livebench 主要测试死记硬背的任务**：成员们辩论了 **Livebench** 的优劣，一些人认为它主要测试 *死记硬背的任务（rote tasks）*，并奖励 *闪电式思考（flash thinking）* 而非更深层次的推理。
   - 这种对死记硬背任务的关注可能会扭曲结果并降低基准测试的可靠性。
- **Gemini 2.5 Pro 提高了限制并引发关注**：讨论了 **Gemini 2.5 Pro** 的能力，从 *擅长数学* 到表现出指令遵循问题，Logan Kilpatrick [在 X 上](https://x.com/OfficialLoganK/status/1905298490865160410)宣布提高了速率限制。
   - 针对模型的一致性和稳定性，人们的担忧日益增加，特别是免费的 AI Studio 版本与付费的 Gemini Advanced 版本之间的差异。
- **AI 审查制度辩论升温**：讨论转向 **AI 模型中的审查制度**，人们担心西方模型过于 *觉醒（woke）*，而中国模型则是 *宣传复读机（propaganda parrots）*。
   - 成员们争论政府审查是否与安全护栏和法律合规有所区别。
- **Qwen 3 发布在即**：人们对即将发布的 **Qwen 3** 充满热情，并对其架构和性能进行了推测。
   - 一些人期待一个具有令人印象深刻性能的 MoE 模型，而另一些人则对其与 **Qwen 2.5 Max** 相比的实际能力保持谨慎。
- **DeepSeek V3 0324 分数惊人**：**DeepSeek V3 0324** 在 **SWE-bench** 上令人印象深刻的分数受到关注，引发了对其相对于 GPT-4o 等其他模型编程能力的讨论。
   - 有人认为这些编程能力的提升可能是 *氛围编程（vibe coding）* 或 *基准测试微调（benchmark tuning）* 的结果，而非模型架构的真正进步。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **选择性量化改进了 Unsloth 的 Dynamic Quants**：Unsloth 的 [Dynamic Quants](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) 经过选择性量化，他们发布了 **DeepSeek-V3-0324 GGUFs**，包括 1-4-bit Dynamic 版本。
   - 这允许在 *llama.cpp*、LMStudio 和 Open WebUI 中运行该模型，并提供了详细的说明指南。
- **Unsloth 的 Orpheus TTS Notebook 表现出色**：Unsloth 发布了 **Orpheus TTS notebook**，它能提供带有情感线索的类人语音，并允许用户以更少的 VRAM 更快地自定义声音和对话，详见[这条推文](https://x.com/UnslothAI/status/1905312969879421435)。
   - 它支持 *single stage models*，其中一名成员表示 **Kokoro 将完全无法进行微调**。
- **YouTube 算法屈服于伽罗瓦理论**：在 YouTube 上搜索了一次 **Galois theory** 后，一位成员开玩笑说他们的信息流现在充斥着关于 *五次方程（quintics）* 的视频。
   - 他们调侃说这些算法在 *8k ctx* 之后就像“从行走退化到爬行”。
- **Instruct 模型不进行预训练更好？**：一位用户被建议 *不要对 instruct 模型进行持续预训练*，因为这会降低性能，且通常旨在添加新的领域知识，参考了 [Unsloth 文档](https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#should-i-choose-instruct-or-base)。
   - 相反，鼓励该成员探索监督微调（SFT）来处理问答任务。
- **字节跳动的 DAPO 系统开启大门**：成员们分享了 [字节跳动开源 RL 系统 **DAPO**](https://github.com/BytedTsinghua-SIA/DAPO) 的链接，指出它“似乎有点被忽视了”。
   - 该系统来自 ByteDance Seed 和清华 AIR。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Google Gemini 2.5 Pro 需求激增**：据 **@OfficialLoganK** 的[推文](https://x.com/OfficialLoganK/status/1905298490865160410)称，由于需求旺盛，Google 正在优先提高 **Gemini 2.5 Pro** 的速率限制（rate limits）。
   - 为了绕过速率限制，[OpenRouter 建议](https://x.com/OpenRouterAI/status/1905300582505624022)添加 **AI Studio API key** 并设置 OpenRouter。
- **GPT-4o 在近期更新后获得关注**：根据 [OpenAI 的推文](https://x.com/OpenAI/status/1905331956856050135)，**GPT-4o** 在 **ChatGPT** 中获得了更新，提升了指令遵循、技术问题解决、直觉和创造力。
   - 它目前在 [Arena Leaderboard](https://x.com/lmarena_ai/status/1905340075225043057) 上排名第 2，超越了 **GPT-4.5**，并在 Coding 和 Hard Prompts 类别中并列第 1。
- **Aider 新命令简化编码流程**：Aider 新的 `/context` 命令可自动识别并添加给定请求的相关文件，从而简化编码流程，尽管该功能仍处于测试阶段。
   - 这有助于处理大型代码库并节省时间，对于确定需要修改的内容非常有用，并且可以与 reasoning model 配合使用来头脑风暴 bug。
- **Gemini 的 OpenAI API 面临兼容性问题**：有用户报告 **Gemini** 无法与 **OpenAI 兼容层（compatibility layer）**配合使用，怀疑原因是 **Litellm**，尽管其他模型运行正常。
   - 该用户通过反向代理访问所有 AI 服务，因此需要 OpenAI API 兼容性。
- **用户报告 Aider CPU 使用率飙升**：一名用户报告 Aider 的 CPU 使用率突然飙升至 **100%**，导致 **LLM** 挂起或响应缓慢，尽管处理的是小型仓库。
   - 用户正在寻求调试技巧，不确定从何处开始排查该问题。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **破折号爱好者的键盘转换**：一名用户将键盘重新映射以偏好**破折号（dash）**，引发了关于标点符号偏好和替代方案（如分号）的讨论，有人开玩笑说使用 `^` 符号。
   - 这突显了个人在写作和交流风格中做出的细微且个性化的选择。
- **Sora 的挑战：植物提示词与存疑的兔子**：一名用户寻求帮助编写 **Sora** 提示词，以实现植物背景平滑变化的相机旋转效果，并分享了示例图片：[criptomeria-na_kmienku-globosa-cr-300x300.webp](https://cdn.discordapp.com/attachments/998381918976479273/1354591170927263915/criptomeria-na_kmienku-globosa-cr-300x300.webp?ex=67e72a56&is=67e5d8d6&hm=87a5fe3213980f9f0a4e7537bf9f31d79372e17e260d055e2f30c33cdb6675ac&)。
   - 与此同时，有人担心 **Sora** 在提示“兔子角色（bunny characters）”时会生成暗示性内容，引发了对内容审核的质疑。
- **图像生成：恶习还是愿景？**：一名用户批评 **AI 图像生成**是一种贬低数字艺术价值的“恶习”，并分享了一张用它创建的图片：[Screenshot_20250327_162135_Discord.jpg](https://cdn.discordapp.com/attachments/998381918976479273/1354687457974550594/Screenshot_20250327_162135_Discord.jpg?ex=67e6db42&is=67e589c2&hm=b73df22c342e6f5bf1c2f2ce2e83a20174d040096462d957460c7176a20d163b)。
   - 随后的分歧导致该用户被拉黑，凸显了对 **AI 在艺术领域角色**的不同看法。
- **Arxiv 崛起：STEM 的快速舞台**：**Arxiv** 作为 STEM 领域预出版平台的地位日益提升，引发了关于未经评审工作价值、利用“关键多数关注点”推动进步的潜力以及即将到来的 **AI 同行评审（peer review）**时代的辩论。
   - 对传统同行评审过程的批评包括科学家付费参与评审并失去所有权，这激发了对 **AI** 创建更高效、更易访问系统的热情。
- **批量文件上传：一次性全部上传**：成员们确认，为了确保模型在单个上下文中考虑所有文档，在使用 **ChatGPT** 等工具时，最好**同时上传所有文件**。
   - 这种方法能确保模型整合来自所有文档的信息，前提是这些文件格式正确。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 容量紧缺**：用户在运行 **Gemini 2.5** 时遇到 `RESOURCE_EXHAUSTED` 错误，建议在 [OpenRouter settings](https://openrouter.ai/settings/integrations) 中绑定 **AI Studio key** 以提升容量。
   - 强调了 Google 允许用户通过 **AI Studio** 付费以获取更高容量。
- **Deepseek R1 响应为空导致停滞**：有用户报告在使用 **Deepseek R1 (Free)** 时，即使使用新 Key，从 **Chutes provider** 获取的 API 响应也为空。
   - 将 *max_tokens* 设置为 0 被认为是可能的诱因，但即使调整后问题依然存在。
- **OpenRouter 提供商路由现状**：一位用户在使用 AI SDK 跨 Google/Bedrock/Anthropic 路由 Gemini/Anthropic 时，发现并调试了一个路由 Bug。
   - 即使 `allow_fallbacks` 设置为 false，请求也未遵循定义的顺序，导致所有请求最终都指向了 Anthropic；官方人员确认了该路由 Bug。
- **OpenRouter 兼容性追求**：一位用户发现，与通过 Spring AI 使用 `openai/gpt-4o-mini` 相比，使用 `google/gemini-2.5-pro-exp-03-25:free` 时 **OpenRouter** 与 **OpenAI SDK** 的兼容性不足。
   - 一名成员坚持认为 **OpenRouter** *理应 100% 兼容*，用户可能遇到了速率限制（rate limits），并建议使用 **Mistral Small 3.1** 和 **Phi 3** 模型进行测试。
- **免费使用 Gemini 2.5 Pro 的可能性**：成员们分享了如何利用 **OpenRouter** 在 [@cursor_ai](https://cursor.sh) 中免费运行 **Gemini 2.5 Pro**，并参考了一份 [Cursor 教程](https://x.com/1dolinski/status/1904966556037108191)。
   - 该成员报告称，在经过简短的排障后，此方案解决了所遇到的问题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 引入多 GPU 支持**：[LM Studio 0.3.14](https://lmstudio.ai/blog/lmstudio-v0.3.14) 引入了多 GPU 设置控制，允许用户启用/禁用特定 GPU 并选择分配策略，以优化多 GPU 系统的性能。
   - 该版本包含“限制模型卸载至专用 GPU 显存”（Limit Model Offload to Dedicated GPU memory）模式，提升了稳定性并优化了长上下文处理，同时针对 **AMD GPUs** 进行了增强。
- **视觉模型插件探索未果**：成员们在寻找适用于 Hugging Face 模型的 [vision model plugins](https://lmstudio.ai/plugins)，并指出 **Mistral Small** 仅支持文本且为 LM Studio 的 GGUF 格式。
   - 虽然有人建议在 LM Studio 中使用 **Mistral Small**，但似乎没有人成功运行视觉模型（Vision Model）。
- **Threadripper CPU 定位争论**：成员们争论 **AMD Threadripper** CPU 是否属于“消费级”，尽管其市场定位为 HEDT（高端桌面）处理器，并引用了一篇 [Gamers Nexus 文章](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks)。
   - 一名成员认为，虽然面向家庭用户销售，但 Threadripper 实际上是*专业工作站*级别。
- **Gemma 3 性能大幅提升**：一位用户在刚购入的 **9070XT** 上使用 **Gemma3 - 12b Q4_K_M** 达到了 **54 t/s**（Vulkan，未开启 flash attention），而其 **7800XT** 在 Vulkan 下仅为 **35 t/s**，在 ROCm 下为 **39 t/s**。
   - 成员们讨论了 **Gemma3** 模型即使在完全卸载（full offload）的情况下也会溢出到共享内存的问题，并指出上下文可能会填满 32 GB 的共享内存，同时有人询问如何加载超过 48GB 的大型模型。
- **P100 已过时**：一位成员询问花费 **400 加元/200 美元** 购买 **P100 16GB** 作为兴趣投资是否划算，但被强烈建议不要购买，称其为“电子垃圾”。
   - 成员们指出其 Tesla 架构、不支持的 CUDA 版本，以及与 **6750XT** 等现代显卡相比低下的性能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepseek V3 在 Mac 上可与云端抗衡**：成员们将 **Mac Studios** 上运行 **Deepseek V3** 的速度（**20toks/second**）与云端实例进行了对比，并引用了[这篇文章](https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/)中报道的 **AMD EPYC Rome** 系统较慢的性能（**4 tokens/sec**）。
   - 这种差异可能是由于 Mac 拥有更快的统一内存（unified RAM）。
- **EleutherAI 考虑举办 ICLR 2025 聚会**：EleutherAI 正在考虑举办一次 **ICLR 2025 线下聚会**，预计接待约 **30** 名参与者。
   - 如果参与意向较高，可能会探索赞助机会。
- **Qwen 32B 在 LLM harness 上停滞**：一名成员在评估 **Qwen 32B** 模型时遇到了 **LLM harness** 问题，尽管使用了最新版本的 Transformers。
   - 根本原因可能是由于分片模型拥有超过 **10 个分片**，且与 **tied embeddings** 相关，这可能是由 transformers 库本身触发的。
- **Transformers 库触发错误报错**：一名成员将一个误导性错误追溯到 transformers `4.50.2`，并分享了一个运行 `4.50.0` 版本的 [Colab notebook](https://colab.research.google.com/drive/1W1yMQfIY5365IB8b1as3SiD2EIxhtrex?usp=sharing)，该版本不存在此问题。
   - 问题源于存储空间不足，尽管错误消息提示是 `AutoModel` 加载函数的问题；修复方案将以 PR 的形式提交给 lm-eval，以增加更好的错误处理。
- **OLMo-1B 遭遇过度训练崩溃**：根据[这篇论文](https://arxiv.org/abs/2503.19206)，在 **3T tokens** 上预训练的指令微调版 **OLMo-1B** 模型在标准 LLM 基准测试中的表现比其 **2.3T token** 的对应版本差 **2%** 以上。
   - **Gemma Team** 也发表了一篇[新论文](https://arxiv.org/abs/2503.19786)，作者包括 Aishwarya Kamath、Johan Ferret 和 Shreya Pathak。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.5 Pro 夺走 Claude 3.5 Sonnet 宝座**：根据[评估](https://fxtwitter.com/LechMazur/status/1904975669081084273)，**Gemini 2.5 Pro** 在用户偏好中险胜 **Claude 3.5 Sonnet**，在故事元素组合的顶级排名中占据了 *3%*，而 **Claude 3.5 Sonnet** 从 *74%* 下降到 *18%*。
   - 尽管发生了这种转变，用户仍称赞 **Gemini 2.5** 在处理长上下文时的无缝表现，一名用户在 **AI Studio** 上观察到了 **15K token** 的上下文窗口。
- **OpenAI 营收飙升，AGI 梦想若隐若现**：据 [Bloomberg 报道](https://www.bloomberg.com/news/articles/2025-03-26/openai-expects-revenue-will-triple-to-12-7-billion-this-year?srnd=undefined)详述，在 **GPT-4o** 的进步和专注于防止现实世界危害的修订版图像生成政策的推动下，**OpenAI** 预计今年营收将翻三倍达到 **127 亿美元**，并预计到 2029 年达到 **1250 亿美元**。
   - 尽管由于 **GPT-4o** 图像生成的普及导致 GPU 资源受限，**OpenAI** 正在暂时实施速率限制，**Sam Altman** 指出：“*看到人们喜欢 ChatGPT 中的图像非常有趣，但我们的 GPU 快要熔化了*”。
- **Midjourney CEO 抨击 4o 图像生成**：据 [X 上的报道](https://x.com/flowersslop/status/1904984779759480942)，**Midjourney** CEO 批评 **4o** 的图像生成“*又慢又差*”，认为这只是一种融资手段和梗（meme），而不是创意工具。
   - 这一批评出现在关于模型命名规范的讨论中，同时白宫删除了一条包含 **Ghibli** 风格图像的推文，该图像最初被描述为“*阴暗*”。
- **图灵研究院陷入深度动荡**：尽管在 **2024** 年最近注入了 **1 亿英镑**，但据 [researchprofessionalnews.com](https://www.researchprofessionalnews.com/rr-news-uk-research-councils-2025-3-alan-turing-institute-axes-around-a-quarter-of-research-projects/) 报道，[阿兰·图灵研究所 (ATI)](https://www.turing.ac.uk/) 正面临大规模裁员，并计划削减约四分之一的研究项目，引发了员工动荡。
   - 鉴于来自更广泛领域的竞争挑战，该研究所正面临生存威胁。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在单位加法上遇到困难**：`mojo` 频道的讨论强调了在处理不同单位（如千米/秒和米/分）加法时的挑战，涉及返回类型以及如何正确缩放数值的问题。
   - 一位成员指出，*在这种情况下，scale 必须返回正确的结果*，且目前无法在函数的返回类型中使用 `-> A if cond else B` 这种逻辑。
- **C Unions 在 Mojo 中引发辩论**：一位成员询问 `union` 如何进行底层转换（lowers into），另一位建议使用 C union。
   - 第二位成员指出，*据我所知，CUDA 在某些 API 部分使用了 union*。
- **Traits 讨论揭示细微差别**：关于扩展方法和 Traits 的讨论明确了扩展允许向 **library types** 添加方法，而由于孤儿规则（orphan rules），这一特性在 Rust 的 `impl` 中无法直接实现。
   - 另一位成员纠正说，Rust 的 `impl` 是可以实现库类型的。
- **隐式 Trait 实现引起关注**：关于隐式 Trait 实现引发了辩论，一位成员希望这只是暂时的，并表示这 *使得 marker traits 变得危险*。
   - 讨论了传播 Trait 实现的其他方法，包括命名的扩展以及评估健全性（soundness）的权衡。
- **元组可变性令人惊讶**：一位成员指出，可以对元组内部的索引进行赋值，这展示了意想不到的可变性。
   - 另一位成员澄清说，这是 *`__getitem__` 返回可变引用的副作用*，并指出不应该是这种情况，这在 [测试套件](https://github.com/modular/max/blob/main/mojo/stdlib/test/builtin/test_tuple.mojo) 中有所体现。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **字节跳动的 InfiniteYou 合并至 ComfyUI**：字节跳动的 **InfiniteYou** 模型旨在灵活重塑照片并保持个人身份特征，已通过 [此 GitHub 仓库](https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou) 集成到 ComfyUI 中。
   - 其目标是提供一种 *平滑* 的方式来生成不同的高质量图像，将 **Claude** 连接到 **Vite**。
- **HF 推理 API 对免费用户设限**：根据 [HuggingFace 的 API 定价文档](https://huggingface.co/docs/api-inference/pricing)，免费用户在用完所有 **monthly credits** 后将无法再查询推理 API。
   - 相比之下，**PRO** 或 **Enterprise Hub** 用户在请求超过订阅限制时将产生费用。
- **Sieves 简化了零样本 NLP 流水线**：介绍了 **Sieves**，这是一个无需训练、仅使用零样本生成模型构建 NLP 流水线的 [工具](https://github.com/mantisai/sieves)。
   - 它利用 **Outlines**、**DSPy** 和 **LangChain** 等库的结构化输出，确保生成模型的输出准确。
- **Qwen 2.5 VL 在 Kaggle 上遇到内存错误**：一位成员在 Kaggle 上运行 **Qwen 2.5 VL 3b** 描述一段 **10 秒视频** 时遇到了内存错误，此前他刚调试完最新 transformers 库 (**4.50.0.dev0**) 的导入问题。
   - 建议包括使用更高规格的硬件、更小的模型或使用 **Flash Attention 2** 进行 GPU 卸载。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Sama 信号支持：OpenAI 拥抱 MCP！**：**OpenAI** 首席执行官 **Sam Altman** 宣布 [MCP 支持](https://x.com/sama/status/1904957253456941061?t=awjb86WjJSH4MlFo9l5sWw&s=19) 即将引入 **OpenAI** 产品，如 **Agents SDK**、**ChatGPT 桌面应用**和 **Responses API**。
   - 这一举措被视为将 **MCP** 确立为处理业务相关任务的 Agent 骨干架构的关键一步，类似于 **HTTP** 对互联网的影响。
- **Cloudflare 增强上下文：MCP 获得远程服务器工具支持**：**Cloudflare** 现在支持使用 **workers-oauth-provider** 和 **McpAgent** 等工具[构建和部署远程 MCP 服务器](https://blog.cloudflare.com/remote-model-context-protocol-servers-mcp/)。
   - 这一支持是一项重大进展，为开发者提供了更高效构建 **MCP** 服务器的资源。
- **规范问题浮现：Claude 在处理 MCP Prompt 时遇到困难**：当 **MCP** 服务器包含资源或 Prompt 时，用户在 **Claude Desktop** 中遇到了导致无限查询的问题，但包含修复程序的 [GitHub 新版本](https://github.com/yuniko-software/minecraft-mcp-server) 已经发布。
   - 一种解决方法是移除相关能力（capabilities），以防止 **Claude** 搜索缺失的资源和 Prompt。
- **Canvas MCP 连接大学课程**：一位成员为大学课程构建了一个 [Canvas MCP](https://git.new/canvas-mcp)，实现了资源和作业的自动化查询。
   - 响应用户请求，作者添加了一个 Gradescope 集成 Agent，实现了对 **Gradescope** 的自主爬取。
- **全能 Docker Compose 预备 MCP 服务器**：一个 [全能 docker-compose](https://github.com/JoshuaRL/MCP-Mealprep) 项目已创建，使用户能够通过 Portainer 轻松自托管 **17 个 MCP** 服务器。
   - 该 compose 从公共 **GitHub** 项目中获取 **Dockerfiles**，确保更新能够自动应用。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **思维导图功能公开上线**：Notebook LM 的思维导图（Mind Map）功能现已向所有用户开放，团队对用户的耐心和反馈表示感谢，并附上了[感谢图片](https://cdn.discordapp.com/attachments/1182376564525113484/1354601205091012668/NotebookLM_Thank_You.png?ex=67e733ae&is=67e5e22e&hm=feb1a12774e69af171073ec1d544ecbffb8eee2811ef0e42e7768cc89b669f91)。
   - 一位成员指出，虽然思维导图结构整齐，但由于*缺乏描述而浪费时间*，而且用户无法控制思维导图的结构或描述的详细程度。
- **西班牙语播客生成停止**：一位用户报告称，在“自定义（customize）”设置中生成**西班牙语播客**的功能已失效。
   - 一位成员建议通过 **NotebookLM API** 加入播客创建功能，并指出利用该功能可以做出非常酷的东西。
- **笔记本共享令人沮丧**：一位 **Pro 用户** 报告称无法通过链接共享 Notebook，即使内容是公开的 YouTube 视频。
   - 提到的潜在解决方案包括确保接收者拥有活跃的 **NLM 账户**，以及手动通过电子邮件发送链接。
- **Gemini 接受“火鸡测试”**：一位成员对 **Gemini 2.5 Pro** 进行了“火鸡测试（Turkey Test）”，挑战其创作关于鸟的形而上学诗歌，并在此处分享了带有 **NotebookLM 评论**的[视频](https://youtu.be/MagWnkL14js?si=ywCvQQY12Kruh6aZ&t=54)。
   - 用户通过**交互模式（Interactive Mode）**引导评论，从而过渡到一个令人满意的结局，偶然发现了 **NBLM** 的新用途。
- **高级研究功能有限制**：一位成员询问了 **Gemini Advanced** 的深度研究限制，另一位成员回答说是**每天 20 份研究报告**。
   - 第一位成员认为这*与 ChatGPT 相比相当不错*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **提议 Sketch-to-Model 流水线**：一名成员介绍了一种 "Sketch-to-Model" 流程（**Sketch** --> **2D/2.5D Concept/Image** --> **3D Model** --> **Simulation**），并探索了 Kernel Attention (KA) 的替代方案。
   - 该成员提到 **ChatGPT** 暗示了一个类似于 **KAN** 的概念，并将其归于 **Google DeepMind**，而 **Grok 3** 则指出 **xAI** 团队正在积极研究 **KAN**。
- **AI 解谜能力引发辩论**：成员们思考 AI 是否能破解谜题书《Maze: Solve the World's Most Challenging Puzzle》（[Wikipedia](https://en.wikipedia.org/wiki/Maze:_Solve_the_World%27s_Most_Challenging_Puzzle)）。
   - 建议包括在 ARG 和旧的谜题游戏上训练 LLM，尽管大家承认某些谜题刻意设计的难度可能会难倒目前的推理模型。
- **GPT-4o 确认采用自回归图像生成**：根据 [OpenAI's Native Image Generation System Card](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf) 的说明，**GPT-4o** 已被确认为自回归图像生成模型。
   - 推测认为 **GPT-4o** 可能正在重用图像输入 token 进行图像输出，通过 *semantic encoder/decoder*（语义编码器/解码器）输出与输入格式相同的图像 token。
- **图灵研究所削减研究项目**：尽管获得了 1 亿英镑的资助，[Alan Turing Institute (ATI)](https://www.turing.ac.uk/) 仍计划进行大规模裁员，并[削减四分之一的研究项目](https://www.researchprofessionalnews.com/rr-news-uk-research-councils-2025-3-alan-turing-institute-axes-around-a-quarter-of-research-projects/)。
   - 报告指出，由于这些削减，员工中出现了“公开反抗”的情绪。
- **《Tracing Thoughts in Language Model》研究受到剖析**：成员们正在分析 [Tracing Thoughts in a Language Model](https://www.anthropic.com/research/tracing-thoughts-language-model) 及其[相关的 YouTube 视频](https://youtu.be/Bj9BD2D3DzA)。
   - 由于现有资料非常详尽，预计对话将跨越多个阶段。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DP 和 TP Rank 之间的数据分布差异**：据一名成员介绍，在分布式处理（**DP**）中，每个 rank 接收不同的数据，而在张量并行（**TP**）中，所有 rank 接收相同的数据。
   - 他们建议 **TRL** (Transformer Reinforcement Learning) 应该已经自动管理这种分布，以确保高效的训练和资源利用。
- **Triton Autotune 缺少 Pre/Post Hooks**：`triton.Autotune` 或 `triton.Config` 不支持 *pre_hook* 和 *post_hook*，因为它们需要在运行时执行 Python 代码，而 **Inductor** 在 AOTI 中无法支持这一点。
   - 一名成员推测实现这种支持应该不难，并表示愿意提供帮助。
- **Hopper 的 num_ctas 设置困扰 Triton 用户**：用户在 Triton 中为 **Hopper** 使用大于 1 的 `num_ctas` 值时，会遇到崩溃或 `RuntimeError: PassManager::run failed` 异常，根本原因尚不明确。
   - 这实际上限制了在使用 Triton 时针对 **Hopper** 架构的性能调优选项。
- **CUDA 内存层级解析**：一位用户解释了 CUDA 内存层级，并澄清了 **DRAM** 和 **SRAM** 之间的数据传输才是缓慢的根源。
   - 这就是为什么内存合并（memory coalescing）以及最大化全局内存与共享内存之间的数据传输效率至关重要的原因。
- **Red Hat 招聘 GPU Kernel 工程师**：Red Hat 正在招聘不同级别的全职**软件工程师**，要求具备 **C++, GPU kernels, CUDA, Triton, CUTLASS, PyTorch, 和 vLLM** 方面的经验。
   - 有意向的候选人请将简历和相关经验总结发送至 terrytangyuan@gmail.com，并在邮件主题中注明 "GPU Mode"。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 兼作 MCP Server**：**LlamaCloud** 可作为 **MCP server** 使用，为任何 MCP 客户端的工作流实现实时数据集成，如[此视频演示](https://twitter.com/llama_index/status/1905332760614764810)所示。
   - 此设置允许现有的 **LlamaCloud index** 作为 **Claude Desktop** 所使用的 MCP server 的数据源。
- **Claude 利用来自 LlamaCloud 的数据**：**Claude Desktop** 可以将现有的 **LlamaCloud index** 作为 MCP server 的数据源，将秒级更新的数据集成到 **Claude** 工作流中，详见[此视频](https://twitter.com/llama_index/status/1905332760614764810)。
   - 该功能增强了 **Claude** 在其工作流中访问和利用实时信息的能力。
- **LlamaExtract 放弃 Schema 推断**：去年宣布的 **LlamaExtract** 中的 Schema 推断功能已被降低优先级，因为大多数用户已经拥有所需的 Schema，详见 [LlamaExtract 公告](https://example.com/llamaextract_announcement)。
   - 该功能未来可能会回归，但目前正在优先处理其他方面。
- **LLM 用于为 PDF 和扫描图像生成描述**：成员们讨论了将 **LlamaParse** 作为解析 PDF 的最佳解析工具；另一位成员建议使用 **LLM** 来读取并为图像生成描述（用于 **RAG** 应用），以回答来自上传 PDF 的问题。
   - 另一位成员询问了针对手写数学作业等扫描文档的 **Hybrid Chunking** 和 **OCR**。
- **聊天机器人在 SQL 查询生成方面遇到困难**：一位正在构建根据用户消息生成 SQL 查询的聊天机器人的用户报告称，即使在 SQL 文件中包含列注释，机器人也无法选择合适的列。
   - 未提供具体解决方案，但鼓励该用户向团队提交 Bug 报告。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nvidia 收购 Lepton AI**：据 [The Information](https://www.theinformation.com/articles/nvidia-nears-deal-buy-gpu-reseller-several-hundred-million-dollars) 报道，**Nvidia** 已斥资数亿美元收购了推理服务提供商 **Lepton AI**，以增强其提高 GPU 利用率的软件产品。
   - 此次收购旨在简化 GPU 利用率并*加强其软件产品线*。
- **OpenAI 的 Agent 拥抱 MCP**：**Model Context Protocol** (**MCP**) 现在已与 **OpenAI Agents SDK** 集成，支持使用各种 MCP server 为 Agent 提供工具，详见 [Model Context Protocol 介绍](https://modelcontextprotocol.io/introduction)。
   - MCP 被设想为 *AI 应用程序的 USB-C 接口*，使向 LLM 提供上下文的过程标准化。
- **Replit Agent v2 获得更强的自主性**：**Replit Agent v2** 目前正与 **Anthropic** 的 **Claude 3.7 Sonnet** 一起进行早期访问，它拥有更强的自主性，在进行修改前会制定假设并搜索文件，详见 [Replit 博客](https://blog.replit.com/agent-v2)。
   - 此次升级确保了它*更加自主*，且*不太可能卡在同一个 Bug 上*。
- **GPT-4o 在排行榜上跃升**：根据 [Arena 排行榜](https://x.com/lmarena_ai/status/1905340075225043057)，最新的 **ChatGPT-4o** 更新（**2025-03-26**）已飙升至 Arena 第 2 名，超越了 **GPT-4.5**，并有显著增强，在编程（Coding）和硬核提示词（Hard Prompts）方面并列第 1。
   - 据报道，此次更新*更擅长遵循详细指令*，特别是包含多个请求的指令，并具有*更好的直觉和创造力*。
- **OpenAI 放宽图像生成政策**：**OpenAI** 正在将其图像生成政策从一味拒绝调整为防止现实世界的伤害，旨在最大限度地提高创作自由，同时避免实际伤害，正如 Joanne Jang 在[她的博客文章](https://x.com/joannejang/status/1905341734563053979)中所述。
   - 这一政策转变寻求在创意表达与伤害预防之间取得平衡。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FP8 QAT 运行被发现**：一位成员正在探索 **FP8 QAT**，并在尝试对冷训练模型进行纯 **QAT** 运行时遇到了 [此问题](https://github.com/pytorch/ao/issues/1632)。
   - 他们澄清说，虽然 **FP8 QAT** 是一个目标，但目前的资源有限。
- **优化器状态保持不变**：一位成员确认激活 **fake quant** 不会改变优化器状态。
   - 这一确认解决了关于量化实验中意外副作用的担忧。
- **GRPO PR 寻求快速处理**：一位成员强调了合并两个 **GRPO PR**（[#2422](https://github.com/pytorch/torchtune/pull/2422) 和 [#2425](https://github.com/pytorch/torchtune/pull/2425)）的紧迫性，指出 **#2425** 是一个关键的 bug 修复。
   - 一位团队成员迅速回应并承诺处理这些 PR。
- **据称 Anthropic 转向 TensorFlow**：有人指出 **Anthropic** 据称正在围绕 **TensorFlow** 进行标准化。
   - 这引发了关于 **PyTorch** 在 **Anthropic** 内部未来的猜测。
- **JoeI SORA 接管**：一位成员分享了 **JoeI SORA** 在特定背景下的截图，回应了关于模型直觉的查询。
   - 该成员调侃道，*没有直觉，只有 JoeI*。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 探索向量数据库集成**：成员们讨论了 **向量数据库** 的选择，一位成员分享了 [Cohere 集成页面的链接](https://docs.cohere.com/v2/docs/integrations)，展示了 **Elasticsearch**、**MongoDB**、**Redis**、**Haystack**、**Open Search**、**Vespa**、**Chroma**、**Qdrant**、**Weaviate**、**Pinecone** 和 **Milvus** 等选项。
   - 另一位成员询问关于在线托管向量数据库的问题，回复暗示 **Cohere** 会处理托管相关事宜。
- **创始人思考 AI Agent 定价**：一位成员发起了一场关于创始人如何对 **AI Agent** 进行定价和变现的讨论，寻求与其他人的交流并验证见解。
   - 另一位成员鼓励分享更多关于 **AI Agent** 定价策略的细节。
- **Cohere 可能会参加 QCon London**：一位成员询问 **Cohere** 今年是否会参加 **QCon London**，表示有兴趣与 **Cohere** 代表讨论 **North** 的访问权限。
   - 他们去年参加了。
- **难民组织倡导生计**：肯尼亚的一位难民介绍了 **Pro-Right for Refugees**，这是一个专注于促进难民获得 **生计机会** 并增强 Kakuma 难民营和 Kalobeyei 定居点和平生活的社区组织 (CBO)。
   - 该 CBO 专注于和平建设、提高意识和生计倡议，邀请志愿者并为难民提供支持。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **廉价组装预算级 AI 装备**：一位成员探索使用从淘宝购买的旧 **X99** 组件、**Xeons** 和 **32GB ECC DDR4 RAM** 组装一套 **7000-8000 元** 的预算级 AI 装备。
   - 另一位成员在快速调查后确认了该方案的可行性。
- **AX650N 规格亮点**：通过 [产品页面链接](https://www.axera-tech.com/Product/125.html) 重点介绍了 **AX650N** 的规格，揭示了其具有 **72Tops@int4, 18.0Tops@int8 NPU**，并原生支持 **Transformer** 智能处理平台。
   - 此外，**AX650N** 包含一个 **八核 A55 CPU**，支持 **8K 视频编解码**，并具有 **双 HDMI 2.0** 输出。
- **AX650N 被逆向工程**：一篇 [博客文章](http://jas-hacks.blogspot.com/2024/09/ax650n-sipeed-maix-iv-axerapi-pro-npu.html?m=1) 详细介绍了 **AX650N** 的逆向工程，报告其达到了 **72.0 TOPS@INT4** 和 **18.0 TOPS@INT8**。
   - 文章还提到了移植较小 **Transformer** 模型的努力，并指向了一个相关的 [GitHub repo](https://github.com/AXERA-TECH/ax-llm)。
- **Tinygrad 的 PR 专注于 CPU 功能**：分享了两个 Tinygrad 的 pull requests：[PR #9546](https://github.com/tinygrad/tinygrad/pull/9546) 和 [PR #9554](https://github.com/tinygrad/tinygrad/pull/9554)。
   - 第一个 PR 解决了 *test_failure_53 中递归错误* 的潜在修复，而第二个 PR 旨在 *继续将函数从 torch backend 的 CPU 中移出*。
- **TinyGrad 的代码生成揭秘**：一位用户询问了 TinyGrad 的代码生成过程，引用了关于 `CStyleCodegen` 和 `CUDACodegen` 类的过时信息，寻求理解从优化计划到低级代码的转换。
   - 鉴于用户对当前实现的困惑，讨论旨在澄清 TinyGrad 如何将优化计划转换为适用于各种设备（CPU/GPU）的可执行代码。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **允许分享课程录像**：一位成员询问是否可以分享 **LLM Agents Berkeley MOOC** 的课程录像，管理员确认这是允许的，并鼓励新参与者[注册](https://forms.gle/9u6HdWXGws16go9)。
   - 这是新 MOOC 参与者入职流程的一部分。
- **正在考虑延长导师计划申请截止日期**：一位成员请求延长导师计划（mentorship）的申请截止日期；管理员指出表单不会立即关闭。
   - 然而，由于关注度很高且项目需要尽快启动，截止日期后的申请不保证会被考虑。
- **创业赛道（Entre Track）缺少导师计划**：一位成员询问关于 **Entre track** 的导师计划，管理员澄清 **Berkeley** 并不提供该项服务。
   - **4月/5月**将会有赞助商的答疑时间（office hours）。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AOT 分解推理过程**：**Atom of Thoughts (AOT)** 将问题分解为结构化为有向无环图 (DAG) 的原子级子问题，这与维持整个树历史的 **Tree of Thoughts (ToT)** 形成对比。
   - 发布者强调了 AOT 的无记忆推理步骤，以及针对原子子问题的显式“分解-收缩”阶段。
- **理想的评估数据集**：AOT 的理想评估数据集应包括 [GSM8K 和 MATH](https://example.com/datasets)（带有逐步解题过程的数据集）、[HotpotQA 和 2WikiMultihopQA](https://example.com/datasets)（带有标注推理路径的数据集），以及显式详细说明中间推理步骤的数据集。
   - 提供的示例包括用于测试和验证的 `mock_llm_client.generate.side_effect = ["0.9", "42"]`。
- **LLMDecomposer 策略具有灵活性**：**AOT** 利用了 `LLMDecomposer` 提供的灵活分解功能，其 Prompt 会根据问题类型（MATH, MULTI_HOP）进行调整，支持自定义分解器，并启用动态 Prompt 选择。
   - 分解策略通过收缩验证阶段确保原子性，例如 Prompt：`QuestionType.MATH: Break down this mathematical question into smaller, logically connected subquestions: Question: {question}`。
- **MiproV2 遇到 ValueError**：一位用户在使用 **MiproV2** 时遇到了 **ValueError**，这与 `signature.output_fields` 中不匹配的键有关，预期键为 `dict_keys(['proposed_instruction'])`，但实际收到的键为 `dict_keys([])`。
   - 据报道，在 GitHub 上使用 **Copro** 时也遇到了类似问题，可能与 `max_tokens` 设置有关。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 现已支持 Gemini 2.5 Pro！**：**Gemini 2.5 Pro** 现在可以在 Windsurf 中使用，每条消息为用户提供 **1.0** 用户 Prompt 额度，每次工具调用提供 **1.0** Flow Action 额度。
   - 该发布已在 [X 上宣布](https://x.com/windsurf_ai/status/1905410812921217272)。
- **Gemini 2.5 Pro 导致 Windsurf 负载过重**：由于负载巨大，Windsurf 在使用 **Gemini 2.5** 时遇到了速率限制（rate limiting）问题。
   - 团队正在积极努力增加配额，并对带来的不便表示歉意。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 用户抱怨模型导入问题**：用户报告在向 **GPT4All** 导入模型时遇到困难，系统似乎没有反应，此外还存在无法搜索模型列表的问题。
   - 其他投诉包括选择模型时缺少模型大小信息、缺乏 LaTeX 支持以及模型列表排序不符合用户习惯。
- **GPT4All 用户体验引发不满**：用户对 **GPT4All** 的用户体验表示失望，提到了缺少 Embedder 选择选项等问题。
   - 一位用户表示：*你们正在流失用户……因为其他工具更加用户友好且更愿意保持开放*。



---


**MLOps @Chipro Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：详细的频道摘要和链接


{% if medium == 'web' %}




### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1354530524646871252)** (1297 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro 定价与访问, Windsurf vs. Cursor: 优缺点, 上下文窗口限制, 模型性能与偏好, 工作流策略`

- **Cursor 在使用 Gemini 2.5 Pro 时面临速率限制（Rate Limits）**：用户在使用 Cursor 中的 **Gemini 2.5 Pro** 时遇到了非常小的速率限制，有用户报告称仅进行了*两次 API 请求*就达到了限制。
   - 一位成员提到结合使用 **Gemini API**、**Requesty** 和 **Open Router** 来规避这些限制，而另一位成员则建议 **Google Workspace Business** 账户可能通过与代表直接沟通来获得更高的限制。
- **Windsurf 与 Cursor 的对决**：成员们继续争论 **Windsurf** 与 **Cursor** 的优劣，一些人因其声称的*全上下文（full context）*而青睐 **Windsurf** 进行原型设计，而另一些人则更喜欢用 **Cursor** 进行 Bug 修复。
   - 一位用户指出，**Windsurf** 的 UI 样式与 **Cursor** 相比不够统一，而后者在不同页面间保持了一致的美感；另一位用户则调侃说应该雇佣那个“官方的 shitsurf 托儿”。
- **Cursor 限制了上下文窗口大小？**：一位用户询问 **Cursor** 是否将 **Gemini 2.5 Pro** 的上下文窗口限制在了 30K，尽管该模型宣传有 1M 的上下文限制。
   - 随后有人澄清，虽然 Agent 模型的上下文窗口为 60k，但 **Claude 3.7** 拥有 120k 的上下文窗口，如果启用最高设置甚至可达 200k，尽管目前还没有足够的数据来判断其是否使用了 **Vertex**。
- **Gemini 2.5 Pro 令人困惑的定价难题**：用户对 **Cursor** 为 **Gemini 2.5 Pro** 收费表示担忧，因为该模型据称通过 **Google** 的 **AI Studio API** 是免费提供的，这引发了对不诚实的指责。
   - 一位 **Cursor** 代表澄清说，收费是为了覆盖处理该模型使用所需的容量，因为 **Google** 在 **Cursor** 的规模下并不提供免费层级，且该模型的定价与 Gemini 2.0 Pro 相当。
- **新工作流涌现：再见 Cursor，你好 Cline？**：用户正在尝试新的编码工作流，其中一些人打算离开 **Cursor**，转而使用 **VSCode** 配合 **Cline** 进行编码（使用 **Gemini**），并使用来自 **Fireworks** 的 **DeepSeek v3** 进行规划。
   - 虽然对 **Cursor** 的 **Tab** 功能表示怀念，并认为大多数模型整体有所下滑，但仍认可上述模型的实用性。一位用户发表观点认为，*在 RTX 4090 级别上，本地模型（local models）表现都很糟糕*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dontasktoask.com">Don't ask to ask, just ask</a>：未找到描述</li><li><a href="https://docs.cursor.com/settings/models#context-window-sizes">Cursor – Models</a>：未找到描述</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>：未找到描述</li><li><a href="https://docs.cursor.com/settings/models#large-context-and-pricing">Cursor – Models</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>：受近期展示了在精心策划的数据上预训练的小型基于 Transformer 的语言模型前景的工作启发，我们通过投入大量精力策划一个...来强化这些方法。</li><li><a href="https://x.com/nicdunz/status/1905353949865238633?s=46">来自 nic (@nicdunz) 的推文</a>：4o 在编程方面排名第一</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：`chatgpt-4o-latest` 现已在 API 中更新，但请保持关注——我们计划在未来几周内将这些改进引入 API 中的特定日期版本模型。引用 OpenAI (@OpenAI) GPT-4o 获得了另一次更新...</li><li><a href="https://tenor.com/view/drake-hotline-bling-dance-dancing-gif-17654506">Drake Hotline Bling GIF - Drake Hotline Bling Dance - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/compare">比较 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 驱动编程助手。我们也是首个 Agentic IDE —— Windsurf 的创造者。</li><li><a href="https://tenor.com/view/monkey-sad-monkey-sad-edit-monkey-edit-%C3%BCzg%C3%BCn-maymun-gif-15640172319982461811">Monkey Sad Monkey GIF - Monkey Sad Monkey Sad edit - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.qt.io/blog/more-time-for-coding-with-the-qt-ai-assistant">Qt AI Assistant 实验版发布</a>：介绍 Qt AI Assistant：通过专家建议、单元测试用例创建以及针对 Qt 和 QML 开发的代码文档自动化，提升您的编程效率。</li><li><a href="https://tenor.com/view/brain-hurts-gif-12096381820083648834">Brain Hurts GIF - Brain hurts - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/correct-futurama-the-best-kind-of-correct-yes-yep-gif-5787390">Correct Futurama GIF - Correct Futurama The Best Kind Of Correct - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/ghostedvpn-hacker-cat-bongo-cat-keyboard-cat-hacker-gif-4373606555250453292">Ghostedvpn Hacker Cat GIF - GhostedVPN Hacker Cat Bongo Cat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/wink-eye-wink-gif-3023120962008687924">Wink Eye Wink GIF - Wink Eye wink - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Artificial_general_intelligence">Artificial general intelligence - 维基百科</a>：未找到描述</li><li><a href="https://cimwashere.com">Cim Was Here</a>：摄影。加利福尼亚州斯托克顿。</li><li><a href="https://codeium.com/windsurf">Codeium 推出的 Windsurf 编辑器</a>：未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，让开发者保持专注流。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://21st.dev/DavidHDev/splash-cursor/default">21st.dev - 面向设计工程师的 NPM</a>：使用受 shadcn/ui 启发的即插即用型 React Tailwind 组件，更快速地交付精美的 UI。由设计工程师构建，服务于设计工程师。</li><li><a href="https://codeium.com/profile/docker">Cim (@docker) 个人资料 | Codeium</a>：Cim (@docker) 使用 Codeium 的 AI 自动补全完成了 4,068 次操作。Codeium 提供顶级的 AI 代码补全和聊天功能 —— 全部免费。</li><li><a href="https://mcp.so/server/open-docs-mcp/askme765cs">MCP 服务器</a>：最大的 MCP 服务器集合，包括 Awesome MCP 服务器和 Claude MCP 集成。搜索并发现 MCP 服务器以增强您的 AI 能力。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1354613786744717615)** (1 条消息): 

> `Perplexity Discord Bot, Testing the Discord Bot, Discord Bot Feedback` 


- **Perplexity Bot 进驻 Discord**：Perplexity 向社区推出了 **Perplexity Discord Bot** 进行测试。
   - 该 Bot 旨在直接在 Discord 频道内提供快速回答和事实核查功能，可以通过标记 <@&1354609473917816895> 或使用 **/askperplexity** 命令来访问。
- **测试 Perplexity Bot**：测试人员可以通过使用 **/askperplexity** 命令或标记 <@&1354609473917816895> 获取快速回答，使用 ❓ 表情符号回应来对评论进行事实核查，并使用 **/meme** 命令创建迷因（memes）。
   - 此次活动号召社区成员*释放好奇心*，探索直接集成在 Discord 频道中的 Bot 功能。
- **欢迎 Discord Bot 反馈**：请在 <#1354612654836154399> 频道中提交改进 Discord Bot 的反馈。
   - 团队正在积极征集 Bug 报告和建议，以增强 Bot 的性能和用户体验。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1354530666422866121)** (748 条消息🔥🔥🔥): 

> `GPT-4.5 停用、Complexity 扩展、MCP 服务器、Perplexity Pro API 与订阅对比、CEO 建议移除广告` 


- **GPT-4.5 从 Perplexity 消失**：成员们注意到 **GPT-4.5** 从 [Perplexity.com](https://perplexity.com/) 的模型选择界面中消失了，可能是出于成本考虑，且未事先通知。
   - Perplexity AI 机器人表示，**GPT-4.5** 在科学推理、数学和编程方面通常优于 **GPT-4o**，而 **GPT-4o** 在通用和多模态使用方面表现更好。
- **Complexity 扩展增强 Perplexity 功能**：用户讨论了 "Complexity" 扩展，这是一个增强 Perplexity 功能的第三方插件。
   - Perplexity AI 机器人指出，虽然该扩展的功能作为原生选项可能很有益，但集成决策取决于用户需求、技术可行性以及产品路线图的规划。
- **Perplexity 通过 MCP 与 Playwright 集成**：用户探索利用 Model Context Protocol (**MCP**) 服务器（如 **Playwright**），让 Perplexity 能够控制浏览器或其他具有可用 **MCP** 服务器的应用程序。
   - Perplexity AI 机器人解释说，设置并配置服务器以配合 Perplexity 的 **API** 使用，将能够实现自动化的浏览器操作和网页交互。
- **服务中断困扰 Perplexity**：用户报告了 Perplexity AI 连续几天出现间歇性停机问题，潜在原因从网络问题到 **VPN** 使用不等。
   - 受影响的用户经历了 Thread 丢失、无法创建 Space 以及平台普遍无响应的情况，许多人对 Perplexity 缺乏关于停机沟通的情况表示沮丧。
- **图像生成问题令用户沮丧**：用户报告在访问 Perplexity Pro 的“生成图像”选项时遇到困难，特别是在 **iOS** 应用上。
   - Perplexity AI 机器人建议使用网页浏览器版本或联系支持团队寻求帮助，而一些用户分享了涉及特定提示词或刷新浏览器的解决方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gemini.google.com/app">‎Gemini - 激发灵感的聊天工具</a>：Bard 现已更名为 Gemini。从 Google AI 获取写作、规划、学习等方面的帮助。</li><li><a href="https://docs.perplexity.ai/system-status/system-status">未找到标题</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/guides/usage-tiers">未找到标题</a>：未找到描述</li><li><a href="https://status.perplexity.com/](https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态</li><li><a href="https://tenor.com/view/out-of-order-sign-does-not-work-not-working-gif-8828471237193375934">故障告示牌 GIF - 故障告示牌不起作用 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://notebooklm.google.com/notebook/fb467802-ecce-457f-9bec-5239c400af47/audio">未找到标题</a>：未找到描述</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jlbks7/theory_perplexity_is_cranking_cost_saving_up_to/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案</a>：Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案 - ItzCrazyKns/Perplexica</li><li><a href="https://www.reddit.com/r/perplexity_ai/s/lL0EKljazZ">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://m.youtube.com/shorts/UT4CEsSiIq0"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1354635651110273194)** (6 messages): 

> `Perplexity AI Search, Android 15, Bluetooth Toggle` 


- **Perplexity AI 搜索网页**：机器人正在 [Perplexity AI](https://www.perplexity.ai/search/gugnaejusigeseo-gongmaedoga-ge-wlU9NR5CQq2WRB0SRYfonAh4ck3rkor.kor.ghost) 上执行多次搜索，包括[另一个](https://www.perplexity.ai/search/gugnaejusigeseo-gongmaedoga-ge-wlU9NR5CQq2WRB0SRYfonA)、[又一个](https://www.perplexity.ai/search/why-does-the-perplexity-ai-tea-mRm_VPbVR_G0e7T_vB7hPw)以及[再一个](https://www.perplexity.ai/search/high-efficiency-near-infrared-ka97WjbuRVWyzasV3cGJrA)。
- **Android 15 Bluetooth 切换开关**：一位成员一直在搜索 [Android 15 Bluetooth Toggle](https://www.perplexity.ai/page/android-15-bluetooth-toggle-re-OIp4saVCRpaoCryD_d8iig) 的改进和讨论。
- **2025 年最低工资**：一位用户一直在搜索 [2025 年最低工资](https://www.perplexity.ai/search/2025nyeon-coejeoimgeum-YtFxUsbKQJW38Osg8q0n.w#0) 的详情。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1354708301874724947)** (27 messages🔥): 

> `sonar API issues, llama-3.1-sonar-small-128k-online problems, Tier 3 access needed, Perplexity API parameter error handling` 


- **旧代码导致模型错误**：一位成员通过删除一段有问题的旧代码解决了 **code 400** 错误，澄清了问题并非出在 **llama-3.1-sonar-small-128k-online** 模型本身。
   - 该用户确认在删除旧代码后，模型运行正常。
- **API 用户请求 Sonar Reasoning Pro 的 Tier 3 访问权限**：一位用户紧急请求 **Sonar Reasoning Pro** 的 **Tier 3 访问权限**，以便为即将举行的活动控制搜索参数（*图像、相关问题、搜索域名过滤、结构化输出*）。
   - 他们还询问了关于指定搜索深度、为研究提供自定义指令、迭代搜索功能，以及获取迭代搜索源以便使用另一个 LLM 进行总结的问题。
- **突发的参数错误导致应用功能中断**：一位用户在拥有额度的情况下遇到了与 `response_format` 参数相关的错误，导致其应用功能中断并造成销售损失。
   - 他们寻求立即支持以解决该问题，并请求临时访问权限以防止进一步损失。
- **已实现针对参数的 API 错误处理**：API 团队实现了针对 `search_domain_filter`、`related_questions`、`images` 和 `structured_outputs` 等参数的错误处理，并澄清这些参数从未对非 Tier 3 用户开放。
   - 如果用户在 prompt 中传递 JSON schema，将继续看到正确的行为。
- **用户寻求将 Llama Index RAG 与 Sonar 集成**：一位用户咨询了关于如何使用 index 对象将 **Llama Index RAG** 上下文集成到 Perplexity sonar 模型的建议。
   - 该帖子被标记了 'pplx' 表情符号。



**提及的链接**：<a href="https://docs.perplexity.ai/guides/usage-tiers">未找到标题</a>：未找到描述

  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1354533010229301379)** (1045 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro, Manus 邀请码等待时间, Discord 侧边栏变化, Manus Staging WordPress 问题, Manus 与 N8N 工作流自动化` 


- **对 Gemini 2.5 Pro 的早期好评**：一些成员报告称 [**Gemini 2.5 Pro** 目前的代码编写表现相当不错](https://ai.google.dev/)，且*比去年使用的任何 Gemini 模型都要好得多*。
   - 一位用户提到，他们*将两个独立的动画组件合并为一个，实现了无缝的加载过渡*。
- **Manus 邀请码延迟**：几位用户对 **Manus 邀请码的等待时间**表示沮丧，有些人已经等待了超过一周。
   - 一位成员建议在注册和接收嵌入代码时使用*无痕模式或不同的浏览器*。
- **Discord UI 侧边栏消失**：一位用户询问其 **Discord 侧边栏**中缺失的图标、线程和消息，特别是在 platform.openai.com 上。
   - 另一位用户建议将外观设置更改为紧凑视图，或检查 PC 显示设置以解决尺寸问题。
- **WordPress staging 无法正常工作**：一位成员报告称，自上次维护以来，他们在 **Manus 中的 WordPress staging 站点**一直反复失败。
   - 其他人似乎没有遇到同样的问题，目前尚未找到解决方案。
- **Manus 与 N8N 集成**：成员们讨论了将 **Manus** 与 [**N8N**](https://www.n8n.io/) 或 Make.com 结合使用，以实现**流程自动化**和**工作流自动化**。
   - 一位成员正在构建他们的第一个 **N8N** 和 **Manus** 工作流，旨在连接全球的创意人士。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://0wa15.github.io/new/">Owais 的 2025 毕业设计项目</a>：未找到描述</li><li><a href="https://x.com/sama/status/1905296867145154688">来自 Sam Altman (@sama) 的推文</a>：看到人们喜欢 ChatGPT 中的图像非常有趣。但我们的 GPU 快要熔化了。我们将暂时引入一些速率限制，同时努力提高效率。希望不会太久...</li><li><a href="https://ucebdqhq.manus.space/">使用 Manus AI 进行迭代开发：全面指南</a>：未找到描述</li><li><a href="https://ucebdqhq.manus.space">使用 Manus AI 进行迭代开发：全面指南</a>：未找到描述</li><li><a href="https://tenor.com/view/pusheen-pusheen-cat-pusheen%27s-best-friend-pusheen-ufo-pastel-pusheen-gif-2867983329129202548">Pusheen Pusheen Cat GIF - Pusheen Pusheen cat Pusheen's best friend - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/please-gif-6208927578341129473">Please GIF - Please - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://manus.im/app">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，让你在休息时完成一切。
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1354530797759238317)** (652 条消息🔥🔥🔥): 

> `Livebench 基准测试讨论, Gemini 2.5 Pro 性能, AI 模型中的审查, Qwen 3 发布, DeepSeek V3 0324 性能`

- **Livebench 被质疑为机械任务测试工具**：成员们讨论了 **Livebench** 的优劣，有人认为它主要测试的是*机械性任务（rote tasks）*，并非一个全面的基准测试。
   - 有观点认为 Livebench 奖励的是*闪念式思考（flash thinking）*而非深度推理，这可能会导致结果偏差。
- **Gemini 2.5 Pro 提高速率限制**：讨论了 **Gemini 2.5 Pro** 的能力，评价从*数学表现惊人*到存在指令遵循问题不等。
   - 存在对模型一致性和稳定性的担忧，特别是免费的 AI Studio 版本与付费的 Gemini Advanced 版本之间的差异，尽管 Logan Kilpatrick [在 X 上](https://x.com/OfficialLoganK/status/1905298490865160410)宣布提高了速率限制。
- **AI 审查辩论愈演愈烈**：讨论转向了 **AI 模型中的审查制度**，既有对西方模型过于“觉醒（woke）”的担忧，也有对中国模型成为“宣传复读机”的顾虑。
   - 成员们辩论了政府审查是否与安全护栏（safety guardrails）及法律合规有所区别。
- **Qwen 3 发布日期临近？**：对即将发布的 **Qwen 3** 热情高涨，人们对其架构和性能充满了猜测。
   - 一些人期待这是一个性能惊人的 MoE 模型，而另一些人对其相对于 **Qwen 2.5 Max** 的实际能力持谨慎态度。
- **DeepSeek V3 0324 编程表现专业**：**DeepSeek V3 0324** 出色的 **SWE-bench** 评分受到关注，引发了对其相对于 GPT-4o 等其他模型编程实力的讨论。
   - 有人认为这些编程能力的提升可能是*氛围编程（vibe coding）*或*针对基准测试调优（benchmark tuning）*的结果，而非模型架构的真实进步。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1905298490865160410">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们目前看到对 Gemini 2.5 Pro 的巨大需求，正全力以赴尽快为开发者提供更高的 rate limits。这是目前的头等大事，请保持关注...</li><li><a href="https://x.com/patloeber/status/1905333725698666913">来自 Patrick Loeber (@patloeber) 的推文</a>：🏆Gemini 2.5 Pro 目前在 LMArena 排名第一，在 Livebench 排名第一，在 SEAL 排行榜中也排名第一。它也开始成为编程任务的首选 :) 我们的团队正在努力为每个人提供更高的...</li><li><a href="https://x.com/JustinLin610/status/1905329896194539913">来自 Junyang Lin (@JustinLin610) 的推文</a>：目前尚未准备好 oss，且仍在不断演进中。但将可以通过 Qwen Chat 访问。很快会宣布。一个小版本发布</li><li><a href="https://x.com/schizoretard18/status/1904941858414870612">来自 Sensitive Young Fascist (@schizoretard18) 的推文</a>：未找到描述</li><li><a href="https://x.com/flavioad/status/1905347584438251848?s=46">来自 Flavio Adamo (@flavioAd) 的推文</a>：OpenAI 刚刚更新了 GPT-4o。我测试了旧版与新版，差异实际上非常惊人。引用 OpenAI (@OpenAI) 的话：GPT-4o 在 ChatGPT 中又获得了一次更新！有什么不同？- 更好地遵循详细...</li><li><a href="https://x.com/seb3point0/status/1904985828524192253">来自 seb3point0 🌶️ (@seb3point0) 的推文</a>：@willcole</li><li><a href="https://x.com/koltregaskes/status/1904974999011614895">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：MIDJOURNEY V7 目标发布日期是 3 月 31 日星期一！😀 就在下周！</li><li><a href="https://www.boba.video?ref_id=VFH1QJIGQ">Boba AI 视频编辑器 | 创建 TikTok, Reels &amp; Shorts 视频</a>：使用 Boba AI 视频编辑器为 TikTok、Instagram Reels 和 YouTube Shorts 创建令人惊叹的短视频。AI 驱动的视频创作变得简单。</li><li><a href="https://x.com/bedros_p/status/1905252764461965615?s=46">来自 Bedros Pamboukian (@bedros_p) 的推文</a>：不可能吧。models/gemini-coder-1</li><li><a href="https://tenor.com/view/looking-at-wrist-watch-wrist-watch-time-passing-by-late-appointment-concerned-gif-3217407494617679420">看手表/时光流逝 GIF - Looking-At-Wrist-Watch Wrist-Watch Time-Passing-By - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://matharena.ai/">MathArena.ai</a>：MathArena：在未污染的数学竞赛上评估 LLMs</li><li><a href="https://www.cybelesoft.com">零信任网络访问 (ZTNA)、VDI 和 PAM 解决方案 | Cybele Software</a>：探索 Cybele Software 的平台，提供通用 ZTNA、VDI、远程桌面、云管理和 RPAM，以增强安全性并降低成本</li><li><a href="https://g.co/gemini/share/33015c25ac37">‎Gemini - 大数乘法计算结果
</a>：由 Gemini Advanced 创建</li><li><a href="https://gemini.google.com/share/81f717b8642e">‎Gemini - 大数乘法结果
</a>：由 Gemini Advanced 创建</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://x.com/Presidentlin/status/1905297519531188376">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：看起来我们今天会迎来 QVQ-Max</li><li><a href="https://www.reddit.com/r/Bard/s/U1ieYcF8vo">Reddit - 互联网的核心</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1354549473673482242)** (391 messages🔥🔥): 

> `Unsloth 动态 4-bit 量化, Qwen/Qwen2.5-Omni-7B 在 Unsloth 中的应用, GRPO 研究, TTS 微调, Llama 3.2 vision` 


- **Unsloth 的新动态量化采用选择性量化**：Unsloth 的 [Dynamic Quants](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally) 是选择性量化的，相比标准 bits 大幅提升了准确度。
   - Unsloth 发布了 **DeepSeek-V3-0324 GGUFs**，包括 1-4-bit 动态版本，用于在 *llama.cpp*、LMStudio 和 Open WebUI 中运行该模型，并提供了一份[指南](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)包含详细的运行说明。
- **发布支持情感暗示的 TTS 微调 Unsloth Notebook**：Unsloth 发布了 **Orpheus TTS 笔记本**，它能提供带有情感暗示的类人语音，并允许用户以更少的 VRAM 更快地自定义声音和对话。
   - 一位成员表示它支持*单阶段模型* -> 基本上是 LLM text token 输入 -> audio token 输出，并补充说 **Kokoro 将完全无法微调**。
- **关于如何训练一个生成带有分类情感语音的 LLM 的细节**：成员们正在使用来自 **11labs 的 scribe v1**，它可以进行*音频事件分类*，从 **40,000 小时音频**的数据集中提取带有情感的转录语音行，以训练 Orpheus 模型。
   - 目标是在*给定情绪下的散文/速度/间距*，一位成员指出*官方 Orpheus 是在 8096 ctx 上训练的，所以你可以处理长达 5-10 分钟的内容*。
- **LLM 玩回合制策略游戏**：成员们讨论了微调 LLM 来玩复杂的、比国际象棋复杂 1000 倍的回合制游戏的可能性。
   - 一位成员回复道：“如果你认为在这么多排列组合/规则下能达到那种可靠性，那你是在自欺欺人”，并建议查看 [y-haidar/awbw-research](https://github.com/y-haidar/awbw-research)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1905312969879421435">来自 Unsloth AI (@UnslothAI) 的推文</a>: 使用我们的笔记本免费微调 Orpheus-TTS！Orpheus 提供具有情感暗示（叹息、笑声）的类人语音，表现优于 OpenAI。使用减少 70% 的 VRAM，以 2 倍的速度自定义声音和对话...</li><li><a href="https://x.com/danielhanchen/status/1905315906051604595">来自 Daniel Han (@danielhanchen) 的推文</a>: 我们在一个微型数据集上训练了 Orpheus-TTS（一个语音 LLM），并成功完全改变了模型的声音和个性！非常酷，尤其是模型具有像咯咯笑或叹息这样的情感暗示...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=MKX_XKs_BNZR">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/pensive-zech-nod-gif-21183959">Pensive Zech GIF - Pensive Zech Nod - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/MrDragonFox/Elise">MrDragonFox/Elise · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/y-haidar/awbw-research">GitHub - y-haidar/awbw-research: 这是一个已停止的项目，包含了尝试为游戏 awbw 创建 AI 的努力</a>: 这是一个已停止的项目，包含了尝试为游戏 awbw 创建 AI 的努力 - y-haidar/awbw-research</li><li><a href="https://huggingface.co/deepghs">deepghs (DeepGHS)</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Jinsaryko/Elise">Jinsaryko/Elise · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1354567398417498192)** (2 messages): 

> `YouTube 推荐算法, context length 限制` 


- **YouTube 算法推送伽罗瓦理论**：一位成员开玩笑说，在 YouTube 上搜索了一次 **Galois theory** 后，他们的信息流现在全是关于*五次方程 (quintics)* 的视频。
   - 他们嘲讽说这些算法在 *“8k ctx 之后就开始从走路退化到爬行”*。
- **Context Length 爬行**：该成员还开玩笑说 **context length** 会随着内容变长而迅速衰减。
   - 他们嘲讽说这些算法在 *“8k ctx 之后就开始从走路退化到爬行”*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1354535211420880948)** (67 条消息🔥🔥): 

> `Gemma3 微调问题，动态 4-bit 量化，Qwen2.5VL-7B 微调，毒性注入攻击，使用 LoRA 微调 Llama 3` 


- **Gemma3 微调困扰**：一位用户在尝试从磁盘加载微调后的 **Gemma3** 模型时遇到了 `ValueError`，这与缺失 `bitsandbytes` 组件有关，详见 [GitHub issue #638](https://github.com/unslothai/unsloth/issues/638)。
   - 用户提供了一个最小示例，展示了保存和加载 **Gemma3** 模型时出现的问题。
- **Qwen2.5VL-7B 微调变更**：一位用户分享了他们为微调 **Qwen2.5VL-7B** 对 **GRPOTrainer** 所做的修改，并指向了特定的 [Discord 消息](https://discord.com/channels/1179035537009545276/1179777624986357780/1354452423296422079)。
   - 用户询问 Unsloth 的实现是否能比他们自己的修改提供更好的内存优化。
- **LoRA 微调解锁 Llama 3**：分享了一篇博客文章，讨论了 **LoRA** 如何减少微调期间修改的参数量，从而使微调 **Llama 3 8B** 变得可行。
   - Neptune AI 的博客文章可以在[这里](https://neptune.ai/blog/fine-tuning-llama-3-with-lora)找到。
- **Instruct 模型：预训练还是不预训练？**：建议用户*不要对 instruct 模型进行持续预训练*，因为这会降低性能，且通常仅用于添加新的领域知识。
   - 相反，鼓励该成员针对问答任务探索有监督微调（SFT），参考 [Unsloth 文档](https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#should-i-choose-instruct-or-base)。
- **Llama 3 的批量 Token 问题**：一位用户报告称，在使用批量输入时，**Llama 3 8B** 对相同的输入返回了不同的 Token，并将问题追溯到 `2025.3.4` 版本的回归。
   - 用户确认 `2025.3.3` 版本运行符合预期。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb#scrollTo=QQMjaNrjsU5_">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/u">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#should-i-choose-instruct-or-base">我应该使用哪个模型？ | Unsloth 文档</a>：未找到描述</li><li><a href="https://neptune.ai/blog/fine-tuning-llama-3-with-lora">使用 LoRA 微调 Llama 3：分步指南</a>：你可以将这种“Google Colab 友好型”方法的核心思想应用于许多其他基础模型和任务。</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai/unsloth/issues/638">无法加载 CodeLlama-13b · Issue #638 · unslothai/unsloth</a>：我想以内存高效的方式微调 CodeLlama-13b。我能用 CodeLlama-7b 完成，但在 13b 上失败了。我无法加载 unsloth/codellama-13b-bnb-4bit 模型、tokenizer...</li><li><a href="https://huggingface.co/datasets/JoyeeChen/10k_mixed_animal_CoT_data_day85_third_path_realistic_qa">JoyeeChen/10k_mixed_animal_CoT_data_day85_third_path_realistic_qa · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">tatsu-lab/alpaca · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1cCYaP_OsYWJQPSNba0b_9C-AjyxwdQTb?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1354554210863157440)** (65 messages🔥🔥): 

> `ByteDance training policy, Dr GRPO paper, Catastrophic overtraining, Low precision training, Nvidia pruning paper` 


- **DAPO：ByteDance 的开源 RL 系统引发关注！**：成员们分享了 [ByteDance 的开源 RL 系统 **DAPO**](https://github.com/BytedTsinghua-SIA/DAPO) 的链接，并指出它*“似乎被忽视了”*。
   - 该系统来自 ByteDance Seed 和清华 AIR。
- **推理长度影响模型性能**：讨论围绕*“响应长度的稳定增加允许更大程度的探索”*这一观点展开，引用了 [Dr GRPO 论文](https://arxiv.org/abs/2503.19206)，认为这能改善模型的推理和训练。
   - 有理论认为，如果对错误答案中过多的 thinking tokens 进行惩罚，可能会*“切断搜索空间”*。
- **灾难性过训练：预训练越多，问题越多？**：论文 *“[Catastrophic Overtraining](https://arxiv.org/abs/2503.19206)”* 表明，延长的预训练可能会降低微调性能，并将这种效应称为*“灾难性过训练”*。
   - 一位成员建议，如果 LLM 非常紧密地拟合一个概率分布（预训练数据），然后你尝试将其转移到第二个分布（指令数据），效果将不会理想。
- **剪枝与层添加：Nvidia 的微调策略？**：一位成员提到，Nvidia 展示了一篇剪枝论文，可以在[这里](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62348/)找到。
   - 另一位成员描述，剪枝掉与给定任务无关的参数，然后添加新层以补偿损失的体积并继续训练，可以在可塑性和预训练收益之间取得平衡。
- **KB-LaM：Microsoft 的即插即用外部知识！**：一位成员分享了 Microsoft **KB-LaM** (*Knowledge Base Language Model*) 的链接，[为 LLM 引入了即插即用的外部知识](https://www.microsoft.com/en-us/research/blog/introducing-kblam-bringing-plug-and-play-external-knowledge-to-llms/)。
   - 与微调或 RAG 不同，**KB-LaM** 无需昂贵的重新训练或复杂的检索模块即可整合外部知识。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.19206">Overtrained Language Models Are Harder to Fine-Tune</a>：大语言模型在不断增长的 token 预算下进行预训练，前提是假设更好的预训练性能会转化为改进的下游模型。在这项工作中，我们挑战了这一观点……</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62348/">Optimizing Large Language Models: An Experimental Approach to Pruning and Fine-Tuning LLama2 7B | GTC 24 2024 | NVIDIA On-Demand</a>：面对大语言模型 (LLM) 的高计算需求，我们提出了一种模型剪枝和微调的实验方法，以克服……</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/bytedance/FlexPrefill">GitHub - bytedance/FlexPrefill: Code for paper: [ICLR2025 Oral] FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference</a>：论文代码：[ICLR2025 Oral] FlexPrefill: 一种用于高效长序列推理的上下文感知稀疏注意力机制 - bytedance/FlexPrefill</li><li><a href="https://github.com/BytedTsinghua-SIA/DAPO">GitHub - BytedTsinghua-SIA/DAPO: An Open-source RL System from ByteDance Seed and Tsinghua AIR</a>：来自 ByteDance Seed 和清华 AIR 的开源 RL 系统 - BytedTsinghua-SIA/DAPO</li><li><a href="https://www.microsoft.com/en-us/research/blog/introducing-kblam-bringing-plug-and-play-external-know">A more efficient path to add knowledge to LLMs</a>：介绍 KBLaM，一种在 LLM 内部编码和存储结构化知识的方法。通过在不重新训练的情况下整合知识，它为传统方法提供了一个可扩展的替代方案。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1354531289478336664)** (452 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, Rate Limits with Gemini 2.5 Pro, Model Context Protocol (MCP), OpenAI's GPT-4o Update, Aider's New /context Command`

- **Gemini 2.5 Pro 面临需求激增，速率限制（Rate Limits）即将提高**：由于需求旺盛，Google 团队正优先提高 **Gemini 2.5 Pro** 的速率限制 ([@OfficialLoganK 推文](https://x.com/OfficialLoganK/status/1905298490865160410))。
   - 一位用户指出：*Google 的 LLMs 在编程方面一直表现很烂，给我们点给力的东西吧！*
- **OpenRouter 提供 Gemini 2.5 Pro 优化策略**：为了绕过速率限制，[OpenRouter 建议](https://x.com/OpenRouterAI/status/1905300582505624022)添加 **AI Studio API key** 并设置 OpenRouter，将其作为*浪涌保护器（surge protector）*使用。
   - 另一位成员提到，他们免费层的量化版 DeepSeek 相当“愚笨”且啰嗦。
- **Gemini 2.5 碾压 Claude 3.7 的编程能力**：Reddit 用户报告称，**Gemini 2.5** 仅用一个提示词就修复了 Claude 3.7 糟糕的代码。
   - 另一位成员报告说，Gemini 2.5 *刚刚横扫了我的重构项目，绝对是 Sonnet 无法比拟的。*
- **OpenAI 的 GPT-4o 获得提升**：**GPT-4o** 在 **ChatGPT** 中获得了更新，提升了指令遵循、技术问题解决、直觉和创造力 ([OpenAI 推文](https://x.com/OpenAI/status/1905331956856050135))。
   - 它目前在 [Arena Leaderboard](https://x.com/lmarena_ai/status/1905340075225043057) 上排名第 2，超越了 **GPT-4.5**，并在编程（Coding）和硬核提示词（Hard Prompts）分类中并列第 1。
- **Aider 的 /context 命令加速代码库导航**：Aider 新的 `/context` 命令可自动识别并添加与给定请求相关的的文件，从而简化编程流程，但目前该功能尚不成熟且处于测试阶段。
   - 这有助于处理大型代码库并节省时间，对于确定需要修改的内容非常有用，并且可以与推理模型（reasoning model）结合使用来头脑风暴 Bug。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/OfficialLoganK/status/1905298490865160410">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们目前看到对 Gemini 2.5 Pro 的巨大需求，正全力以赴尽快为开发者提供更高的 rate limits。这是目前的头号优先级，请保持关注：...</li><li><a href="https://openai.github.io/openai-agents-python/mcp/">Model context protocol (MCP) - OpenAI Agents SDK</a>：未找到描述</li><li><a href="https://tenor.com/view/biden-joe-biden-sunglasses-searching-gif-17573852463373874852">Biden Joe Biden GIF - Biden Joe biden 墨镜 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/jim-halpert-face-yeah-ok-then-sure-gif-14215039757555920220">Jim Halpert Face GIF - Jim Halpert Face Yeah - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：相比 1 月版本有显著提升（+30 分，从第 5 名升至第 2 名）；在 Coding 和 Hard Prompts 类别中并列第 1。</li><li><a href="https://x.com/AnthropicAI/status/1905303835892990278">来自 Anthropic (@AnthropicAI) 的推文</a>：Anthropic 最新研究：追踪大语言模型（LLM）的思绪。我们构建了一个“显微镜”来检查 AI 模型内部发生的情况，并用它来理解 Claude 的（通常是复杂且令人惊讶的）...</li><li><a href="https://play.google.com/store/apps/details?id=com.aroncode.aronlauncher.lite">Aron Launcher LITE - Google Play 应用</a>：未找到描述</li><li><a href="https://x.com/OpenRouterAI/status/1905300582505624022">来自 OpenRouter (@OpenRouterAI) 的推文</a>：为了最大化利用您的免费 Gemini 2.5 配额：1. 在 https://openrouter.ai/settings/integrations 中添加您的 AI Studio API key。我们的 rate limits 将作为您的“浪涌保护器”。2. 在您的...中设置 OpenRouter</li><li><a href="https://x.com/OpenAI/status/1905331956856050135">来自 OpenAI (@OpenAI) 的推文</a>：ChatGPT 中的 GPT-4o 迎来了另一次更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的 prompts；- 提升了处理复杂技术问题的能力...</li><li><a href="https://tenor.com/view/gojo-gojo-annoyed-gif-16577936018911977061">Gojo Gojo 恼火 GIF - Gojo Gojo 恼火 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.offgamers.com/">OffGamers 在线游戏商店 - Vue</a>：未找到描述</li><li><a href="https://tenor.com/view/wink-cute-kid-heart-anime-gif-15546572">眨眼可爱 GIF - 眨眼可爱小孩 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/nervous-shy-kermit-terrified-scared-gif-3186014724530266078">紧张害羞 GIF - 紧张害羞的 Kermit - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tools.simonwillison.net/jina-reader">Jina Reader</a>：未找到描述</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>：LLM 的统一接口。为您的 prompts 寻找最佳模型和价格</li><li><a href="https://x.com/testingcatalog/status/1905038108845834531?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重磅消息 🚨：Anthropic 正准备发布新版本的 Claude Sonnet 3.7，具有 500K Context Window（目前为 200K）。</li><li><a href="https://huggingface.co/jinaai/ReaderLM-v2">jinaai/ReaderLM-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Aider-AI/aider/issues/3648">itellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes · Issue #3648 · Aider-AI/aider</a>：Issue 我添加了多个可编辑（4个）和只读（5个）文件。我准备了一个较大的 prompt 用于 "/paste"。LLM 开始回复了一点，但在中间我收到了错误：itellm.APIConnectionE...</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>：Aider 使用您的 git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>：aider 是您终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>：使用 code、architect、ask 和 help 聊天模式。</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/WnywOonYGO">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/parameter_settings">Temperature 参数 | DeepSeek API 文档</a>：temperature 的默认值为 1.0。</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/206">[RFC] 由 jspahrsummers 提交的“用新的 Streamable HTTP 传输替换 HTTP+SSE” · Pull Request #206 · modelcontextprotocol/specification</a>：此 PR 为 MCP 引入了 Streamable HTTP 传输，添加...</li>

在保持其优势的同时，解决了当前 HTTP+SSE 传输的主要局限性。我们对 @atesgoral 和 @top 表示深切感谢...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1354535973421322260)** (48 messages🔥): 

> `只读文件添加 PR，Gemini 与 OpenAI API 的兼容性问题，/context 模式说明，Aider git 问题，为 architect 和 coder 设置不同的模型` 


- ****只读文件添加？发起一个 PR****：一位用户咨询关于创建一个 **PR** 来增加一个*以只读方式添加文件*的选项。
   - 这将允许 Aider 根据正则表达式文件模式自动读取某些文件，类似于 Cursor 的基于规则的系统。
- ****Gemini 的 OpenAI API 兼容性难题****：一位用户报告了 **Gemini** 无法与 **OpenAI 兼容层**配合使用的问题，怀疑 **Litellm** 是原因，尽管其他模型运行正常。
   - Paul 建议改用 **Gemini provider**，而该用户解释说他们通过反向代理访问所有 AI 服务，因此必须使用 OpenAI API 兼容性。
- ****Aider 的 Undo/Clear 命令澄清****：一位用户询问 Aider 在使用 `/undo` 时是否会记住撤销操作，以及 `/clear` 是否会删除整个聊天记忆，Paul 予以确认。
   - Paul 建议如果陷入死循环，请使用 `/clear` 并重新开始。
- ****自定义 Lint 命令：Clojure 难题****：一位用户寻求关于让 Aider 处理 **Clojure** 中括号不匹配问题的建议，并注意到 Aider 的文档提到了 Clojure 的内置 linter。
   - 另一位成员建议在 YML 配置中设置自定义 lint 命令，而该用户想知道内置 linter 是否需要激活。
- ****控制 Aider 的 CPU 使用率飙升****：一位用户报告 Aider 突然飙升至 **100% CPU 使用率**，导致 **LLM** 挂起或响应缓慢，尽管处理的是一个小型仓库。
   - 该用户寻求调试建议，不确定从哪里开始排查该问题。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1354531477848854628)** (222 messages🔥🔥): 

> `键盘重映射，破折号 vs 分号，Sora 提示词与 AI 图像生成，Midjourney vs Sora 图像生成，NSFW 内容与 AI` 


- ****键盘重映射热潮：破折号主导讨论****：一位用户重映射了他们的键盘以更好地使用 **破折号 (dash)**，觉得这更符合他们的个性，引发了关于破折号与其他标点符号用法的讨论。
   - 一些用户承认更频繁地使用破折号，而另一些用户则幽默地威胁要使用 `^` 符号作为分隔符。
- ****Sora 的挑战：植物提示词与审查疑云浮现****：一位用户寻求 **Sora** 视频提示词的帮助，希望使用一张植物图片 ([criptomeria-na_kmienku-globosa-cr-300x300.webp](https://cdn.discordapp.com/attachments/998381918976479273/1354591170927263915/criptomeria-na_kmienku-globosa-cr-300x300.webp?ex=67e72a56&is=67e5d8d6&hm=87a5fe3213980f9f0a4e7537bf9f31d79372e17e260d055e2f30c33cdb6675ac&)) 创建相机旋转且背景平滑变化的特效。
   - 一位用户对 **Sora** 在要求生成兔子角色时生成了过于低幼的图像表示担忧，另一位用户认为生成的图像*非常可疑*。
- ****图像生成的诱惑：恶习还是愿景？****：一位用户在分享了一张用 AI 创建的图像后，认为 **AI 图像生成** 是一种*恶习和分心*，降低了数字艺术的价值 ([Screenshot_20250327_162135_Discord.jpg](https://cdn.discordapp.com/attachments/998381918976479273/1354687457974550594/Screenshot_20250327_162135_Discord.jpg?ex=67e6db42&is=67e589c2&hm=b73df22c342e6f5bf1c2f2ce2e83a20174d040096462d957460c7176a20d163b))。
   - 在一名用户因意见不合将其拉黑后，其他人注意到他们是多么轻易地拉黑持不同意见的人。
- ****DeepSeek 的图像策略：迫在眉睫的 4o 竞争？****：成员们推测 **DeepSeek** 何时会发布自己的 4o 竞争级图像模型，并指出 DeepSeek 此前曾发布过名为 **Janus** 的自回归图像生成模型。
   - 一位用户指出 **Janus** 虽然不如 4o，但它展示了 DeepSeek 在自回归图像生成方面的进展。
- ****Google AI Studio vs. ChatGPT：寻求编程清晰度****：一位用户寻求 **ChatGPT Plus** 在 Matlab 编程方面的替代方案，理由是提示词限制，建议包括 **Google AI Studio, Grok3, Hugging Face Chat, DeepSeek v3, 和 QwenChat**。
   - 他们很快被提供的众多选项搞糊涂了，并表示这些选项在他们看来似乎都一样。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1354533759655084152)** (16 条消息🔥): 

> `Context Window, 图像生成` 


- **批量文件上传**：成员们讨论了是一次性上传文件更好，还是逐个上传更好。共识是最好**同时上传所有文件**，以确保模型能够综合考虑所有文档。
   - 一位成员确认同时上传所有文件没问题，但前提是所有内容都必须*格式正确*。
- **允许 ChatGPT 图像商业用途**：一位成员询问，如果支付了 ChatGPT 费用，由 **ChatGPT** 创建的图像是否允许用于**商业用途**。
   - 另一位成员回答是肯定的，并澄清在付费订阅的情况下，图像的**商业使用**是被允许的。
- **Context Window 估算**：成员们讨论了 **Context Window** 的限制，其中一人澄清 **Plus** 账户拥有 **32k** 的 Context Window，而 **Pro** 账户则拥有 **128k**。
   - 另一位成员估计，如果上传 **10 份文档**（总计 **30MB**），在对话进行到 **10 条消息**以内就可能超过 Context Window 限制。
- **寻找无需注册的卡通图像生成器**：一位成员询问是否有某种**图像生成器**可以将剧本中的不同场景转换为卡通风格，且**不需要每次都注册**。
   - 该用户不确定应该在 Discord 频道的哪个位置发布这个问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1354614845135126609)** (42 条消息🔥): 

> `Sora Prompt Engineering, AI 学术研究, Arxiv 在 STEM 出版中的角色, AI 同行评审, 翻译外语数据` 


- ****Sora 的 Prompt：AI 电影视觉****：编写有效的 **Sora** Prompt 需要在简洁的电影导演指令与清晰的视觉构图之间取得平衡，包括镜头移动、焦点、背景处理和情感基调。正如一个为精细雕塑生成 360 度环绕镜头的示例 Prompt 所展示的那样。
   - 一位用户展示了使用特定 Prompt 生成背景模糊的雕塑视频的结果，突出了该系统在 Text-to-Video 生成方面的能力，并提供了一个成功的 Prompt Engineering 实际案例。
- ****AI 研究：付费墙与出版瓶颈****：一位寻求在学术资源中进行深度研究的 Prompt 的用户被告知，大多数学术资源都受付费墙限制，这限制了 AI 驱动研究的潜力。一位成员指出，“*目前已经充斥着大量 AI 撰写的‘研究’，这正造成出版瓶颈*”。
   - 虽然摘要是可以访问的，但仅基于摘要进行深度研究被认为是不充分的。这引发了关于由于访问壁垒和 AI 生成内容的现状，使用 AI 进行全面学术研究面临的挑战和局限性的讨论。
- ****Arxiv 势头：STEM 领域的预印本中心****：Arxiv 作为预印本平台在 STEM 领域日益普及，这引发了关于未经评审的工作与同行评审过程相比之下的价值讨论。
   - 有人建议，对预印本工作的“*关键关注度*”可能会带来进步，AI 未来可能会成为评审过程中的一名“*同行*”，这引发了关于学术验证动态演变的辩论。
- ****AI 同行评审：即将到来的革命？****：讨论了 AI 担任同行评审员的可能性，认为 AGI 模型可能具备有效评估研究的智力水平，尽管它不应该是“*唯一的定论*”。
   - 讨论中对当前的同行评审系统表达了担忧，包括科学家被迫付费才能获得评审机会，以及研究成果被锁定在付费墙后。这让人希望 AI 同行评审能够带来一个更高效、更开放的系统。
- ****AI 翻译：消除研究中的语言障碍****：一位用户表示，由于数据和文章是外语，在研究水坝拆除后果时遇到困难。对此，建议使用 AI 模型逐块翻译并讨论这些信息。
   - 建议保留原始语言并与模型进行对话，以确保对思想的全面理解，捕捉未言明的细微差别，并解决不同语言之间概念框架的差异，从而促进更有效的研究。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1354614845135126609)** (42 条消息🔥): 

> `Sora Prompts, AI Research Paper, Arxiv, Meta-Prompting, AI Peer Review` 


- **为电影级视频创作 Sora Prompt**：一位成员分享了关于为 **Sora** 编写 Prompt 的建议，强调了**摄像机移动**、**焦点**、**背景处理**以及**情感基调**对于生成电影级视频的重要性。
   - 建议的 Prompt 包含具体细节，如 *“围绕精细雕塑的平滑 360 度摄像机轨道”* 以及 *“带有景深的柔化模糊背景”*，以达到理想的效果。
- **对 AI 辅助深度研究论文的探索受到质疑**：成员们讨论了使用 AI 进行撰写研究论文所需的**深度研究**的可能性，并指出许多学术资源都存在**付费墙 (paywalled)**，这使得全面的自动化研究变得困难。
   - 一位成员建议使用 Meta-Prompting，即向 ChatGPT 解释研究目标以生成更有效的科研 Prompt；同时也表达了疑虑，认为 AI 仍难以处理学术资源获取，因为许多学术网站设置了 robots.txt，导致这些资源成为 AI 无法读取的“垃圾”。
- **Arxiv 在 STEM 领域日益普及**：一位成员提到 **Arxiv** 作为论文发布平台在 STEM 领域正变得越来越受欢迎，尽管它**未经同行评审**。
   - 另一位成员认为，一旦有足够多的人评审作品，事情就会开始改观，并预见 AI 在不久的将来会成为同行评审员，从而改变学术研究的动态。
- **同行评审并不完美；优秀的作品无需评审也同样优秀**：一位成员认为，**最高质量的工作**在同行评审**之前**和**之后**都具有价值，并指出劣质内容也可能通过同行评审过程。
   - 该成员进一步批评了传统的出版模式，即作者通常必须付费才能让其作品被考虑，随后还会**失去作品的所有权**，这引发了对该流程的大量抱怨。
- **模型可以协助解决语言翻译挑战**：一位需要处理政府机构发布的**外语**数据的成员得到了建议：与模型进行对话，解释双方将一起阅读数据，一次处理一个**分块 (chunk)**，并讨论其含义、模型的推断、未言之意以及是否存在冲突或歧义。
   - 核心思路是将原始语言文本的**分块**粘贴到模型中，让模型更有效地推断和翻译思想。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1354859538003787890)** (1 条消息): 

> `Gemini 2.5, OpenRouter tips, Cursor IDE integration` 


- **Gemini 2.5 免费配额最大化技巧**：为了最大化利用你的免费 **Gemini 2.5** 配额，用户应在 [OpenRouter 设置](https://openrouter.ai/settings/integrations)中添加其 **AI Studio API key**，这样 *速率限制 (rate limits) 就能起到浪涌保护器的作用*。
   - 此外，用户应在他们喜爱的 **IDE** 中设置 **OpenRouter**（提供有 [Cursor 教程](https://x.com/1dolinski/status/1904966556037108191)），并使用 one-shot tickets。
- **在 Cursor AI 中通过 OpenRouter 免费使用 Gemini 2.5 Pro**：一位成员发布了如何通过 **OpenRouter** 在 [@cursor_ai](https://cursor.sh) 中免费获取 **Gemini 2.5 Pro** 的方法。
   - 据提到，这是在他苦思冥想 10 分钟并在 X 上看到几次类似信息后尝试成功的，该方法确实有效并解决了他们的问题。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1905300582505624022">来自 OpenRouter (@OpenRouterAI) 的推文</a>：为了最大化你的免费 Gemini 2.5 配额：1. 在 https://openrouter.ai/settings/integrations 中添加你的 AI Studio API key。我们的速率限制将为你的配额提供“浪涌保护”。2. 在你的...中设置 OpenRouter。

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1354532572381581479)** (268 条消息🔥🔥): 

> `Stripe security, Gemini 2.5 Pro, OpenRouter and OpenAI SDK compatibility, Deepseek R1 provider issues, OpenRouter provider routing`

- **OpenRouter 不存储银行卡信息，引用 Stripe**：针对用户关于潜在问题的电子邮件，OpenRouter 确认他们不存储任何银行卡信息，所有支付处理均由 [Stripe](https://stripe.com) 处理。
   - 一位成员建议联系 Stripe 或用户的银行以获取更多信息，并提到他们在每两周充值一次的情况下使用 Stripe 没有遇到任何问题。
- **应对 Gemini 2.5 的低容量问题**：用户报告在使用新的 **Gemini 2.5 model** 时收到 `RESOURCE_EXHAUSTED` 等错误消息，Alex Atallah 建议用户连接 **AI Studio key** 以增加容量。
   - 成员指出，Google 提供了通过 **AI Studio** 付费增加容量的选项。
- **OpenRouter 致力于实现 OpenAI SDK 的完全兼容**：一位用户报告在使用 Spring AI 的 OpenAI 支持通过 **OpenRouter** 调用 `google/gemini-2.5-pro-exp-03-25:free` 等模型时，工具（tools）使用出现问题，尽管 `openai/gpt-4o-mini` 可以正常工作，从而质疑 **OpenRouter** 与 OpenAI SDK 的兼容性。
   - 一位成员确认 OpenRouter *应该是 100% 兼容的*，但建议用户可能达到了速率限制，或者特定模型可能不支持工具，而其他人则建议尝试 **Mistral Small 3.1** 和 **Phi 3** 模型进行测试。
- **调试 Deepseek R1 的空响应问题**：用户报告在使用 **Deepseek R1 (Free)** 时，从 **Chutes provider** 收到空的 API 响应，即使尝试了不同的 key 和 Targon 也是如此。
   - 经过调试，*max_tokens* 设置为 0 被确定为潜在原因，一位成员建议将其设置为更高的值，但即使增加了 token 限制，问题仍然存在。
- **OpenRouter 的 Provider 路由无法正常路由**：一位用户在使用 AI SDK 调试 Provider 路由时，尝试在 Google/Bedrock/Anthropic 之间路由 Gemini/Anthropic，但观察到即使将 `allow_fallbacks` 设置为 false，路由顺序也未被遵守，最终总是指向 Anthropic。
   - 工作人员承认了该路由 bug，并感谢用户发现该 bug。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1905300582505624022">来自 OpenRouter (@OpenRouterAI) 的推文</a>：为了最大化你的免费 Gemini 2.5 配额：1. 在 https://openrouter.ai/settings/integrations 中添加你的 AI Studio API key。我们的速率限制将作为你的“浪涌保护器”。2. 在你的...中设置 OpenRouter</li><li><a href="https://x.com/1dolinski/status/1904966556037108191">来自 Chris Dolinski (@1dolinski) 的推文</a>：在苦思冥想 10 分钟并在 X 上看到几次相关内容后，终于解决了如何在 @cursor_ai 中通过 OpenRouter 免费使用 Gemini 2.5 Pro 的问题 /thread</li><li><a href="https://x.com/OfficialLoganK/status/1904989645324296504">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@dosco 请在此提交：https://ai.google.dev/gemini-api/docs/rate-limits</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>：LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>：LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何通过 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3">DeepSeek V3 - API, 提供商, 统计数据</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编程能力之上。在近 15 万亿 token 上进行预训练，报告的评测...</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider：用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百个模型。</a>：用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百个模型。 - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider?tab=readme-ov-file#passing-extra-body-to-openrouter">GitHub - OpenRouterTeam/ai-sdk-provider：用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百个模型。</a>：用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百个模型。 - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://github.com/OpenRouterTeam/ai">GitHub - OpenRouterTeam/ai：使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用</a>：使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用 - OpenRouterTeam/ai</li><li><a href="https://pasteboard.co/gi4Ix2lOks6C.png">API 问题 - Pasteboard 上的图片</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1354943243271274646)** (1 条消息): 

> `LM Studio 0.3.14, Multi-GPU Controls, NVIDIA GPUs, AMD GPUs` 


- **LM Studio 新增多 GPU 控制功能**：[LM Studio 0.3.14](https://lmstudio.ai/blog/lmstudio-v0.3.14) 引入了针对多 GPU 设置的新细粒度控制，允许用户启用/禁用特定 GPU 并选择分配策略。
   - 用户可以在均匀分配或按特定顺序优先选择 GPU 之间进行选择，从而优化多 GPU 系统的性能。
- **GPU 分配策略增强**：新版本包含“限制模型卸载至专用 GPU 显存”模式，提高了稳定性并优化了单 GPU 设置下的长上下文处理，对 **NVIDIA GPUs** 特别有利。
   - 开发人员正致力于将这些增强功能也引入 **AMD GPUs**。
- **解锁隐藏的 GPU 控制**：LM Studio 0.3.14 引入了可通过键盘快捷键访问的高级 GPU 控制：`Ctrl+Shift+H` (Windows) 或 `Cmd+Shift+H` (Mac)，并提供使用 `Ctrl+Alt+Shift+H` (Windows) 或 `Cmd+Option+Shift+H` (Mac) 的弹出窗口选项。
   - 这允许用户在模型加载时管理 GPU 设置，提升用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.14">LM Studio 0.3.14：多 GPU 控制 🎛️</a>：针对多 GPU 设置的高级控制：启用/禁用特定 GPU、选择分配策略、限制模型权重到专用 GPU 显存等。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1354544141601869885)** (135 条消息🔥🔥): 

> `Vision Model 插件, LM Studio 下载速度, 模型 VRAM 需求, GitHub Copilot vs Cursor, 使用 Unsloth 微调模型` 


- **Vision Model 插件困惑出现**：成员们正在寻找 [vision model 插件](https://lmstudio.ai/plugins)，以便与直接从 Hugging Face 下载的模型配合使用。
   - 有人指出 **Mistral Small** 仅限文本，并在 LM Studio 中使用 GGUF 格式。
- **VRAM 决定本地模型可行性**：用户讨论了如何判断他们的 PC 是否能运行某个模型，经验法则是 **8GB VRAM** 可以处理 **7B Q4KM 模型**。
   - 有人指出，在 8GB VRAM 的本地环境下，没有任何模型能接近 Sonnet 3.5 的性能。
- **Cursor 的 Agent 模式胜过 Copilot**：一位用户表示在 VS Code 中更倾向于使用 **Cursor** 而非 **GitHub Copilot**，理由是 Cursor 的 *agent 模式*和标签补全功能。
   - 他们强调，能够选择 Cursor 使用的模型是一项关键优势，并提到了 [Cursor 论坛](https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698) 以获取最新更新。
- **微调的挫折与 Unsloth 见解**：用户分享了使用 Unsloth notebook 微调模型的经验，遇到了 VRAM 问题和各种错误，并寻求入门帮助，指向了 [Unsloth Discord](https://discord.com/invite/unsloth)。
   - 成员们阐明了 **训练 (training)** 与 **RAG** (Retrieval-Augmented Generation) 之间的区别。
- **HEDT Threadripper 挑战消费级标签**：成员们辩论了 **AMD Threadripper** CPU 是否应被视为*消费级*，理由是其市场定位为 HEDT (High-End Desktop) 处理器。
   - 一位成员指出，虽然 Threadripper 面向家庭用户销售，但它们实际上是*专业工作站*，并链接了一篇支持该观点的 [Gamers Nexus 文章](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://charanhu.medium.com/fine-tuning-llama-3-2-3b-instruct-model-using-unsloth-and-lora-adb9f9277917">使用 “Unsloth” 和 “LoRA” 微调 Llama-3.2–3B-Instruct 模型</a>：微调像 Llama-3.2–3B 这样的大型语言模型可以显著提高其在自定义数据集上的性能，同时减少……</li><li><a href="https://transformerlab.ai/">来自 Transformer Lab 的问候 | Transformer Lab</a>：LLM 工具包 Transformer Lab 的文档</li><li><a href="https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks">AMD 的“廉价” Threadripper HEDT CPU：7960X 24 核 CPU 评测与基准测试 | GamersNexus</a>：CPUs AMD 的“廉价” Threadripper HEDT CPU：7960X 24 核 CPU 评测与基准测试，2024 年 1 月 2 日。AMD Ryzen Threadripper 7960X 提供了一个引人注目的选项……</li><li><a href="https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698">Claude 3.7 的 Max 模式 - 现已推出！</a>：TL:DR 🧠 核心为 Claude 3.7 Thinking 📚 使用模型的整个 200k 上下文窗口 🛠 具有非常高的工具调用限制 🔍 可以一次读取更多代码 💰 重要提示：仅通过 usa... 提供</li><li><a href="https://harddiskdirect.com/mbd-x10drd-l-o-supermicro-desktop-motherboard.html?utm_source=google&utm_medium=cpc&src=google-search-US&network=x&place=&adid=&kw=&matchtype=&adpos=&device=m&gad_source=1&gclid=CjwKCAjw7pO_BhAlEiwA4pMQvLmwxPZ31Lo40g1U-HBINy6kbrwrcuXi081dt53eLdZ-jusZyxJ9RxoChlMQAvD_BwE">MBD-X10DRD-L-O - Supermicro LGA2011 C612 芯片组 EATX 主板</a>：探索 Supermicro LGA2011 C612 芯片组 EATX 主板 MBD-X10DRD-L-O，美国领先行业，快速发货，价格最优。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1354545095931723817)** (67 messages🔥🔥): 

> `Gemma 3, 9070XT, ROCm, P100, RTX 4060ti 16gb` 


- **Gemma 3 在 9070XT 上的 T/S**：一位刚购入 **9070XT** 的成员在使用 **Gemma3 - 12b Q4_K_M** 时达到了 **54 t/s**（使用 Vulkan，未开启 Flash Attention）。
   - 他们指出，其 **7800XT** 在使用 Vulkan 时仅产生约 **35 t/s**，而在使用 ROCm 时为 **39 t/s**。
- **Gemma 3 的内存胃口**：成员们讨论了 **Gemma3** 模型即使在完全卸载（full offload）的情况下也会溢出到共享内存中，以及上下文如何填满 32 GB 的共享内存。
   - 有人问：*所以可以加载大至 48+gb（24 VRAM + 32 共享 RAM）的模型吗？* 随后补充说，尽管总内存的一部分会在 VRAM 中，*它仍然会表现得像大部分在 RAM 中推理一样*。
- **P100：是电子垃圾还是捡漏？**：一位成员询问将 **P100 16GB** 作为 **400 CAD/200 USD** 的兴趣投资是否划算。
   - 另一位成员强烈建议不要购买，称其由于 Tesla 架构、不支持的 CUDA 版本以及性能逊于 **6750XT** 等现代显卡，基本上就是*电子垃圾*。
- **Nemo 12B 导致崩溃**：一位成员报告称，任何基于 **Nemo 12B** 的模型都会崩溃且无法加载。
   - 该成员表示不确定是否是主板问题导致了这一情况。
- **Backyard AI 程序是良方**：一位成员取得了进展，表示 **Backyard AI** 程序似乎可以正确加载模型。
   - 该成员建议其他尚未解决加载问题的成员尝试使用 **Backyard AI**。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1354543988647919706)** (88 messages🔥🔥): 

> `Deepseek V3 on Mac Studios vs. Cloud Instances, ICLR 2025 Meetup, Qwen2.5-Omni-7B Audio Testing, Qwen 32B Model Evaluation with LLM Harness, Transformers Library Errors` 


- **Deepseek V3 在 Mac 或云端运行？**：成员们讨论了在 **Mac Studios** 上运行 **Deepseek V3**，以及使用高 RAM 云实例作为 GPU 廉价替代方案的可能性，并指出一名用户在 Mac Studio 上达到了 **20toks/second**。
   - 有人引用了[这篇文章](https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/)，其中在 **AMD EPYC Rome** 系统上仅观察到 **4 tokens/sec**，这可能是由于 Mac 拥有更快的统一内存（unified RAM）。
- **EleutherAI 计划 ICLR 2025 聚会**：一位成员询问了 **ICLR 线程**，并建议如果参加人数理想，可以为聚会寻找赞助商。
   - 另一位成员估计 **EAI 及其朋友们** 大约会有 **30** 人参加。
- **Qwen 2.5 Omni 语音样本测试**：一位成员请求他人测试 **Chelsie 语音**，并使用启用了音频输出的 **Qwen/Qwen2.5-Omni-7B** 发布音频样本，具体描述见[此处](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)。
   - 提供的代码片段为：```model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto", enable_audio_output=True,)```
- **LLM Harness 在 Qwen 32B 上遇到困难**：一位成员报告称，尽管安装了最新版本的 Transformers，但在 **LLM harness** 上评估 **Qwen 32B** 模型时仍出现问题。
   - 共识是该分片模型可能拥有超过 **10 个分片**，这与 **tied embeddings** 有关，且 Transformers 库可能是原因所在。
- **报错信息归咎于 Transformers 库**：一位成员将问题追溯到 Transformers `4.50.2`，并分享了一个使用 `4.50.0` 版本且未出现问题的 [Colab notebook](https://colab.research.google.com/drive/1W1yMQfIY5365IB8b1as3SiD2EIxhtrex?usp=sharing)。
   - 错误信息具有误导性，实际上源于存储空间不足，尽管错误信息提示问题出在 `AutoModel` 加载函数上；此外还有建议向 lm-eval 提交 PR 以添加更好的错误处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1W1yMQfIY5365IB8b1as3SiD2EIxhtrex?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/">如何在 2000 美元的 EPYC 服务器上完全本地运行 Deepseek R1 671b – Digital Spaceport</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">Qwen/Qwen2.5-3B at main</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/908ac2b241ecba364e3dc1b500971f5a4fb36bb2/pyproject.toml#L37">lm-evaluation-harness/pyproject.toml at 908ac2b241ecba364e3dc1b500971f5a4fb36bb2 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1354627979702960281)** (2 条消息): 

> `Catastrophic Overtraining, OLMo-1B instruction-tuned model, Gemma Team` 


- **LLMs 面临 Catastrophic Overtraining**：一篇新的 [论文](https://arxiv.org/abs/2503.19206) 挑战了“更好的 Pre-training 性能会转化为更好的下游模型”这一假设，并提出了 **catastrophic overtraining**（灾难性过度训练）这一术语。
   - 论文指出，延长的 Pre-training 可能会使模型更难进行 Fine-tune，由于预训练参数对修改的广泛敏感性系统性增加，导致最终性能下降。
- **OLMo-1B 遭遇 Catastrophic Overtraining**：根据一篇 [论文](https://arxiv.org/abs/2503.19206)，在 **3T tokens** 上进行 Pre-training 的 **instruction-tuned** **OLMo-1B** 模型，在多个标准 LLM 基准测试中的表现比其 **2.3T token** 的对应版本差了超过 **2%**。
- **Gemma Team 发布新论文**：**Gemma Team** 发布了一篇新 [论文](https://arxiv.org/abs/2503.19786)，作者包括 Aishwarya Kamath、Johan Ferret 和 Shreya Pathak。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.19206">Overtrained Language Models Are Harder to Fine-Tune</a>：LLMs 在不断增长的 token 预算上进行 Pre-training，其假设是更好的 Pre-training 性能会转化为改进的下游模型。在这项工作中，我们挑战了这一观点...</li><li><a href="https://arxiv.org/abs/2503.19786">Gemma 3 Technical Report</a>：我们推出了 Gemma 3，这是 Gemma 轻量级开放模型家族的多模态新成员，规模从 10 亿到 270 亿参数不等。此版本引入了视觉理解能力，一个...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1354534381364052137)** (86 messages🔥🔥): 

> `Privileged Basis, 神经网络与固定机制, 用于 Reward Hacking 的 CoT, 流形操作` 


- **Privileged Basis 被解构**：成员们讨论了 "Privileged Basis" 的概念，即数据中在非线性变换后能*保留更多信息*的方向，但一位成员称其为**定义不明 (ill-defined)**，并质疑了其有用性背后的假设。
   - 他们问道：*由谁来赋予特权？*，并进一步质疑了关于点首先在单位球上均匀分布，以及每个点起初具有相同“信息内容”的假设。
- **神经网络缺乏“器官”**：一位成员反对 **Mech Interp** 的概念，断言神经网络是在没有固定机制的情况下进行泛化的，而这正是它们泛化的方式；他指出，神经网络并非由固定机制组成，而是具有信息流和神经活动强度。
   - 他们链接到一条 [推文](https://x.com/norabelrose/status/1905336894038454396)，认为神经网络*无法被组织成一组具有固定功能的部件*。
- **Circuits 是否根本不存在？**：一位成员建议，解释性工具需要描述对整个数据流形 (Manifolds) 的同时操作，而不是针对特定样本锁定模型的行为，并认为这种描述可能比基于激活或权重的方案更具架构无关性 (Architecture Invariant)。
   - 在一个可能属于“大智慧”的时刻，这位成员变得更倾向于认为 **Circuits 并不真实**。
- **CoT 真的很棒！**：一位成员表示 **CoT** 实际上非常好，是最好的解释性工具之一。
   - 他们链接了一篇 [论文](https://arxiv.org/abs/2503.11926)，该研究表明，在 Agent 编码环境中，通过使用另一个 LLM 观察模型的 Chain-of-Thought (CoT) 推理过程，来监控前沿推理模型（如 OpenAI o3-mini）的 Reward Hacking 行为，比仅监控 Agent 的动作和输出要有效得多。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.11926">Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation</a>：缓解 Reward Hacking（即 AI 系统由于学习目标的缺陷或误设而表现不佳）仍然是构建高性能且对齐模型的关键挑战。我们展示了...</li><li><a href="https://x.com/norabelrose/status/1905336894038454396">Nora Belrose (@norabelrose) 的推文</a>：神经网络没有器官。它们不是由固定机制组成的。它们有信息流和神经活动强度。它们无法被组织成一组具有固定功能的部件...</li><li><a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268">Could a Neuroscientist Understand a Microprocessor?</a>：作者摘要：由于很难评估结论是否正确，神经科学的发展受到了阻碍；所研究系统的复杂性及其实验上的不可接近性使得...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1354888783937671395)** (2 messages): 

> `AlpacaFarm logprob/loss 实现, Instruction Tuning EOS Token` 


- **AlpacaFarm 的 Loss 实现受到质疑**：一位成员询问了 [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm/issues/56) 的 **logprob/loss 实现** 之间的差异，并指出使用他们的 Loss 实现会导致在不同序列上得到相同的 Loss/Perplexity。
   - 该成员链接到了 [`rl_models.py` 中的具体实现](https://github.com/tatsu-lab/alpaca_farm/blob/94b02079b74af731b2671e3691a5080d5d340fd8/src/alpaca_farm/models/rl_models.py#L97C30-L97C46) 以供参考。
- **探讨 EOS Token 在 Instruction Tuning 中的作用**：一位成员询问在进行 Instruction Tuning 时是否会添加 **EOS Token**。
   - 未提供进一步的讨论或链接。



**提到的链接**：<a href="https://github.com/tatsu-lab/alpaca_farm/issues/56">[Discussion] about compute_logprobs · Issue #56 · tatsu-lab/alpaca_farm</a>：AlpacaFarm 实现 https://github.com/tatsu-lab/alpaca_farm/blob/94b02079b74af731b2671e3691a5080d5d340fd8/src/alpaca_farm/models/rl_models.py#L97C30-L97C46 DeepSpeedExamples 实现 ...

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1354532441980932218)** (7 messages): 

> `GPT-NeoX Data Chunking, Cross-Document Attention, FA3 Support for H100s, FP8 and H100 performance` 


- **讨论 GPT-NeoX 数据分块 (Data Chunking)**：一名成员提出了关于使用 **GPT-NeoX** 在 **Common Pile v0.1** 上进行训练时的数据分块问题，特别是针对超过上下文长度的文档。
   - 会中指出，当前的预处理脚本可能无法处理预分块（pre-chunking），这可能导致 batch 中出现相关的样本。
- **跨文档注意力 (Cross-Document Attention) 保持开启**：一名成员指出，目前无法关闭同一序列内的跨文档注意力。
   - 这可能会影响性能和训练动态，特别是在处理分块数据时。
- **FA3 支持提升 H100 性能**：提到 **FA3 support** 相比 **A100s** 在 **H100s** 上带来了显著的加速，但目前尚未在 NeoX 中实现。
   - 这可能会影响训练任务的硬件选择考量。
- **FP8 分支实现高 H100 吞吐量**：一名成员报告称，他们有一个增加了新特性的分支（确定有 **FP8**，可能还有 **FA3**），其性能达到了 **580 TFLOP/H100/s** 和 **10,312 T/H100/s**。
   - 这表明由于最近的优化，**H100s** 上的性能得到了显著提升。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1354530389443608787)** (96 messages🔥🔥): 

> `Gemini 2.5 Pro, Claude 3.5 Sonnet, OpenAI Revenue, Midjourney CEO, 4o Image Generation` 


- **Gemini 2.5 抢了 Claude 3.5 的风头**：根据 Lech Mazur 的[评估](https://fxtwitter.com/LechMazur/status/1904975669081084273)，**Gemini 2.5 Pro** 被评为所有 LLM 中表现最好的，其 *3%* 的故事结合了所有要求的元素；而年初以 *74%* 占据榜首的 **Claude 3.5 Sonnet**，现在已降至 *18%*。
- **OpenAI 营收翻三倍，剑指 AGI**：根据彭博社的[报道](https://www.bloomberg.com/news/articles/2025-03-26/openai-expects-revenue-will-triple-to-12-7-billion-this-year?srnd=undefined)，**OpenAI** 预计今年营收将翻三倍达到 **127 亿美元**，并预计到 2029 年达到 **1250 亿美元**，在追求 AGI 的过程中实现现金流转正。
- **Midjourney CEO 对 4o 图像生成表示不满**：据[这条推文](https://x.com/flowersslop/status/1904984779759480942)称，**Midjourney** CEO 据传表示 **4o imagegen** 既慢又差，声称 OpenAI 只是在试图融资，并称其为一个“梗”（meme）而非创意工具，进一步推测一周后就不会有人再谈论它了。
- **稀疏自编码器 (Sparse Autoencoders) 并未改变游戏规则**：来自 GDM 机械可解释性（mechanistic interpretability）团队的研究表明，**Sparse Autoencoders (SAEs)** 并不能帮助探针（probes）实现 OOD（分布外）泛化，也不是一个改变游戏规则的技术，详见 Alignment Forum 上的[这篇文章](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/)。
- **Cohere 发布 Command A 和 R7B 模型及新算法**：**Cohere** 发布了一份技术报告，重点介绍了他们训练 **Command A** 和 **Command R7B 模型** 的新方法，包括使用自我改进（self-refinement）算法和大规模模型合并（model merging）技术，参考[这条推文](https://x.com/max_nlp/status/1905275450101743657)及相关的[论文](https://arxiv.org/abs/2503.20215)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/Presidentlin/status/1905297519531188376">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：看起来我们今天能看到 QVQ-Max 了</li><li><a href="https://fxtwitter.com/sama/status/1905296867145154688">来自 Sam Altman (@sama) 的推文</a>：看到大家在 ChatGPT 中喜欢图像功能非常开心。但我们的 GPU 快熔化了。在努力提高效率的同时，我们将暂时引入一些速率限制（rate limits）。希望不会太久...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 上跃升至第 2 位，超越了 GPT-4.5！亮点：相比 1 月份版本有显著提升（+30 分，从第 5 名升至第 2 名）；在 Coding 和 Hard Prompts 方面并列第 1。</li><li><a href="https://arxiv.org/abs/2503.20215">Qwen2.5-Omni 技术报告</a>：在本报告中，我们介绍了 Qwen2.5-Omni，这是一个端到端的多模态模型，旨在感知包括文本、图像、音频和视频在内的多种模态，同时生成文本和自然...</li><li><a href="https://x.com/srush_nlp/status/1905302653263056911">来自 Sasha Rush (@srush_nlp) 的推文</a>：Simons Institute 研讨会：“LLMs 和 Transformers 的未来”：下周一至周五共有 21 场演讲。https://simons.berkeley.edu/workshops/future-language-models-transformers/schedule#simons-tabs</li><li><a href="https://fxtwitter.com/AnthropicAI/status/1905341566040113375">来自 Anthropic (@AnthropicAI) 的推文</a>：我们进行了一些春季大扫除。感谢您的反馈，Claude 界面现在更加精致了。</li><li><a href="https://x.com/OpenAI/status/1905331956856050135">来自 OpenAI (@OpenAI) 的推文</a>：GPT-4o 在 ChatGPT 中又获得了一次更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词（prompts）- 提升了处理复杂技术任务的能力...</li><li><a href="https://x.com/OpenAI/status/1905331958131097840">来自 OpenAI (@OpenAI) 的推文</a>：更新后的 GPT-4o 现已向所有付费用户开放。免费用户将在未来几周内看到它。</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：`chatgpt-4o-latest` 现已在 API 中更新，但请保持关注——我们计划在未来几周内将这些改进引入 API 中的固定日期模型。引用 OpenAI (@OpenAI)：GPT-4o 在 ChatGPT 中又获得了一次更新...</li><li><a href="https://fxtwitter.com/LechMazur/status/1904975669081084273">来自 Lech Mazur (@LechMazur) 的推文</a>：3% 的 Gemini 2.5 Pro 故事因其所需元素的组合被评为所有 LLM 中最好的。今年年初，Claude 3.5 Sonnet 占据了最佳故事列表的主导地位...</li><li><a href="https://x.com/flowersslop/status/1904984779759480942">来自 Flowers (@flowersslop) 的推文</a>：Midjourney CEO 刚刚表示 4o 的图像生成既慢又差，OpenAI 只是在尝试融资并以一种有毒的方式进行竞争，这只是一个梗（meme）而不是创意工具，一周内就不会有人再关注...</li><li><a href="https://x.com/NeelNanda5/status/1904988240542834724">来自 Neel Nanda (@NeelNanda5) 的推文</a>：GDM Mech Interp 更新：我们研究了 SAEs 是否能帮助探测器进行 OOD 泛化（它们不能 😢）。基于此以及在现实任务中的并行负面结果，我们正在降低 SAEs 工作的优先级。我们的猜测是...</li><li><a href="https://x.com/shiringhaffary/status/1904970542316163555?s=61">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>：新消息：据知情人士透露，OpenAI 预计今年收入将翻三倍，达到 127 亿美元。去年公司年收入为 37 亿美元，预计现金流将转正...</li><li><a href="https://x.com/max_nlp/status/1905275450101743657">来自 Max Bartolo (@max_nlp) 的推文</a>：我很高兴能为我们的 @Cohere @CohereForAI Command A 和 Command R7B 模型发布技术报告。我们强调了我们的模型训练新方法，包括使用自我细化算法（self-refinement algorithms）和...</li><li><a href="https://x.com/sama/status/1905000759336620238?s=61">来自 Sam Altman (@sama) 的推文</a>：ChatGPT 中的图像功能比我们预期的要受欢迎得多（而且我们的预期已经相当高了）。不幸的是，向免费层级的推广将推迟一段时间。</li><li><a href="https://x.com/willccbb/status/1905291246584594797">来自 will brown (@willccbb) 的推文</a>：Cohere 报告中重点强调了 COBOL。他们非常了解自己的客户。</li><li><a href="https://x.com/menhguin/status/1905007511813628299?s=61">来自 Minh Nhat Nguyen (@menhguin) 的推文</a>：你在语音模式中也看到了这一点——4o 语音模式在一年前发布，据称会横扫语音 AI 竞争对手，但它基本上完全没有影响到语音 AI 初创公司。（虽然它是免费的...）</li><li><a href="https://x.com/bedros_p/status/1905252764461965615">来自 Bedros Pamboukian (@bedros_p) 的推文</a>：绝不可能。models/gemini-coder-1</li>

www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/">SAEs 在下游任务中的负面结果以及降低 SAE 研究优先级 (GDM Mech Interp 团队进展更新 #2) — AI Alignment Forum</a>: Lewis Smith*, Sen Rajamanoharan*, Arthur Conmy, Callum McDougall, Janos Kramar, Tom Lieberum, Rohin Shah, Neel Nanda • * = 同等贡献 …</li><li><a href="https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7">Anthropic 经济指数：来自 Claude 3.7 Sonnet 的洞察</a>: Anthropic 经济指数的第二次更新
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1354574780350726274)** (23 messages🔥): 

> `Ghibli 模型训练与版权，Anthropic 推荐计划，CoT 法语，OpenAI 的图像生成政策` 


- **Ghibli 模型会被起诉吗？**: 一位用户在 **30-40 部 Ghibli 电影**上训练了一个模型，并开玩笑说由于[版权侵权](https://x.com/nearcyan/status/1905124036059017406)，他们显然会被起诉。
- **Anthropic 推荐计划**: 一位用户分享了 [Anthropic 推荐合作伙伴计划条款](https://www.anthropic.com/legal/referral-partner-program-terms)的链接。
- **Research UI 翻新与更名**: Claude "Compass" 已更名为 "Research"，并伴随着最近的 [UI 翻新](https://x.com/testingcatalog/status/1905356124314046563)。
- **OpenAI 转变图像生成政策**: 一位 OpenAI 员工分享了他们对在 ChatGPT 中通过 4o 设置新图像生成政策的[想法](https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new)，从一味拒绝转向更精确的方法，重点在于防止现实世界的伤害。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new">关于为新 AI 能力设定政策的想法</a>: 权衡责任与用户自由，以及这如何影响了 ChatGPT 中 4o 图像生成的首日政策设定</li><li><a href="https://x.com/nearcyan/status/1905124036059017406">来自 near (@nearcyan) 的推文</a>: 吉卜力工作室？你是说 ChatGPT 的功能对吧？引用 near (@nearcyan) 的话：肯定有那么一个时刻，即使是厚颜无耻的技术宅也会觉得，那些拿着美国最低工资的吉卜力艺术家理应得到一些……</li><li><a href="https://x.com/joannejang/status/1905341734563053979">来自 Joanne Jang (@joannejang) 的推文</a>: // 我在 OpenAI 领导模型行为团队，想分享一些在设定 4o 图像生成政策时的想法和细微差别。由于我是以博客文章形式发布的，所以包含大写字母 (!)：--Thi...</li><li><a href="https://x.com/TheXeophon/status/1905135997123244371">来自 Xeophon (@TheXeophon) 的推文</a>: 未找到描述</li><li><a href="https://x.com/testingcatalog/status/1905356124314046563">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: Claude "Compass" 已更名为 "Research"，并伴随着最近的 UI 翻新。这会是一个不错的周五发布吗？👀 引用 TestingCatalog News 🗞 (@testingcatalog) 的话：Claude UI 已翻新...</li><li><a href="https://youtu.be/ctdamrnDDoA?si=rRUiYahzcwHp-ow1))"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1354577184299286618)** (43 messages🔥): 

> `GPT-4o 图像生成功能的推出、模型命名规范、Gary Marcus 谈 AI 经济学、白宫删除吉卜力风格推文` 


- **GPT-4o 令人困惑的图像生成发布**：**GPT-4o** 在 ChatGPT 中推出的图像生成功能因流程混乱而受到批评，尽管 OpenAI 声称付费用户很快就收到了该功能。
   - 一位成员指出，“针对使用过时内容的免费用户的警告发布得太晚、太小，而且很容易被忽略”，并引用了 [Simon Willison 关于此次发布及其问题的博客文章](https://simonwillison.net/2025/Mar/25/introducing-4o-image-generation/)。
- **模型命名规范困扰用户**：成员们讨论认为，为模型采用“路径依赖的名称”令人困惑，因为**模型名称不应该要求你了解所有其他模型才能理解它们**。
   - 一位成员觉得 **QvQ** 很可爱，但仍质疑其实用性，而另一位成员建议将其命名为 **QwQ-V** 以明确其功能。
- **Gary Marcus 指出可疑的 AI 经济学**：**Gary Marcus** 发表了一篇 [Substack 文章](https://garymarcus.substack.com/p/genais-day-of-reckoning-may-have)，认为*生成式 AI 的经济学逻辑并不成立*，并提到了他早在 **2001** 年就对技术极限发出的警告。
   - 他指出，尽管 **Nvidia** 从 AI 中获利，但正面临阻力，其股价 YTD 下跌了 **17.75%**（尽管有人指出这个 YTD 数字是 2025 年的）。
- **白宫撤回阴暗的吉卜力风格推文**：白宫删除了一条包含吉卜力风格图片的推文，该图片被一位成员描述为“阴暗”。
   - 据报道，这是一张经过“吉卜力化”处理的恐怖拘留中心照片。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2025/Mar/25/introducing-4o-image-generation/">Introducing 4o Image Generation</a>：当 OpenAI 在 [2024 年 5 月](https://simonwillison.net/2024/May/13/gpt-4o/) 首次宣布 GPT-4o 时，最令人兴奋的功能之一是真正的多模态，即它可以同时输入和输出……</li><li><a href="https://garymarcus.substack.com/p/genais-day-of-reckoning-may-have">GenAI’s day of reckoning may have come</a>：不仅仅是股价问题</li><li><a href="https://x.com/GaryMarcus/status/1904822966908600613">Gary Marcus (@GaryMarcus) 的推文</a>：“你能画一张没有大象的写实海滩图吗？”</li><li><a href="https://bsky.app/profile/danielvanstrien.bsky.social/post/3llcodcvg522u">Daniel van Strien (@danielvanstrien.bsky.social)</a>：未找到描述
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1354838868951826634)** (6 messages): 

> `Alan Turing Institute 危机, WanTeam 的 AI 失败论文, dewey_en_beta embedding model` 


- **Alan Turing Institute 面临危机**：尽管在 **2024** 年获得了 **1 亿英镑** 的新资金，但 [Alan Turing Institute (ATI)](https://www.turing.ac.uk/) 正准备进行大规模裁员，并[削减四分之一的研究项目](https://www.researchprofessionalnews.com/rr-news-uk-research-councils-2025-3-alan-turing-institute-axes-around-a-quarter-of-research-projects/)。
   - 由于潜在的裁员计划，员工们正处于[公开反抗](https://www.theguardian.com/technology/2024/dec/11/redundancies-would-put-alan-turing-institute)状态。
- **WanTeam 发布 AI 失败论文**：WanTeam 在 Computer Vision and Pattern Recognition 上发表了一篇名为 [What AI Failure Looks Like](https://arxiv.org/abs/2503.20314) 的论文。
   - 作者包括 **Ang Wang**、**Baole Ai**、**Bin Wen** 等人。
- **dewey_en_beta Embedding Model 发布**：一份新的技术报告介绍了 [dewey_en_beta](https://arxiv.org/abs/2503.20376)，这是一款开源的 embedding 模型，在 **MTEB (Eng, v2)** 和 **LongEmbed** 上表现优异，支持 **128K token 序列**。
   - 它使用**分块对齐训练 (chunk alignment training)** 来生成局部块嵌入，并托管在 [HuggingFace](https://huggingface.co/infgrad/dewey_en_beta) 上。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.20376">Dewey Long Context Embedding Model: A Technical Report</a>: 本技术报告介绍了开源 dewey_en_beta embedding 模型的训练方法和评估结果。随着对检索增强生成 (RAG) 系统需求的日益增长...</li><li><a href="https://arxiv.org/abs/2503.20314">Wan: Open and Advanced Large-Scale Video Generative Models</a>: 本报告介绍了 Wan，这是一个全面且开放的视频基础模型套件，旨在突破视频生成的边界。Wan 基于主流的 diffusion transformer 范式构建...</li><li><a href="https://www.chalmermagne.com/p/how-not-to-build-an-ai-institute?r=68gy5&utm_medium=ios&utm_campaign=audio-player">How not to build an AI Institute</a>: Alan Turing Institute 出了什么问题？
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1354887101342285997)** (7 messages): 

> `Gemini 2.5, AI Studio, Long Contexts` 


- **Gemini 2.5 的使用令用户惊叹**：用户在 [Gemini 2.5](https://g.co/gemini/share/625533d8ad76) 完美处理实际的长上下文后对其表示赞赏。
   - 一位用户惊叹道：“当丢入实际的长上下文时，模型总是表现挣扎”，随后又补充了一句“哇哦”。
- **用户确认 AI Studio 上的 15K Token**：一位用户报告称，他们观察到了 **15K token** 的上下文窗口。
   - 当被要求在 **AI Studio** 上检查时，该用户回复“是的”确认了这一发现。



**提及的链接**：<a href="https://g.co/gemini/share/625533d8ad76">‎Gemini - LaTeX Typos and Formatting Issues
</a>：由 Gemini Advanced 创建

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1354536916384485417)** (168 条消息🔥🔥): 

> `Unit Scaling, SI Units as a Closed Set, Return Type Logic, Conditional Type, Extension Methods` 


- **解决单位加法的挑战：Kilos Per Second + Meters Per Minute**：围绕解析不同单位（如千米每秒和米每分钟）的加法展开了讨论，关注点在于返回类型以及如何正确缩放数值。
   - 成员们指出这很棘手，因为 *在这些情况下缩放必须返回正确的结果*，而且目前无法在函数的返回类型中使用类似 `-> A if cond else B` 的逻辑。
- **关于使用 C Unions 的分歧**：一位成员询问 `union` 如何下放（lowers into），另一位成员建议使用 C union。
   - 第二位成员表示：*因为我记得 CUDA 在 API 的某些部分使用了 unions。*
- **Traits 讨论揭示细微差别**：关于扩展方法和 Traits 的讨论中，一位成员指出扩展允许向 **library types** 添加方法，由于孤儿规则（orphan rules），这一特性在 Rust 的 `impl` 中无法直接实现。
   - 另一位成员纠正说 Rust 的 `impl` 可以实现 library types。
- **探讨隐式 Trait 实现的合理性 (Soundness)**：关于隐式 Trait 实现的辩论，一位成员希望这些是暂时的，并表示它们 *使得 marker traits 的存在变得危险*。
   - 其他成员讨论了传播 Trait 实现的方法，提到了为扩展命名的可能性，并讨论了 Soundness 的权衡。
- **Tuple 的可变性令人难以置信**：一位成员注意到，可以对 Tuple 内部的索引进行赋值，并且可以通过索引操作完成。
   - 另一位成员强调，这是 *`__getitem__` 返回可变引用的副作用*，并指出不应该是这种情况。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://godbolt.org/z/EzTW18eG7">Compiler Explorer - Rust (rustc nightly)</a>: struct Point {  x: i32,  y: i32,}impl Point {  fn distance(&amp;amp;self) -&amp;gt; f64 {    ((self.x * self.x + self.y * self.y) as f64).sqrt()  }}fn main() {}</li><li><a href="https://github.com/modular/max/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modular/max</a>: 该提案包含了一种 Effect 系统的替代方案，我认为它更适合在系统语言中抽象 async、raises 和类似的 function colors，因为在这些语言中上下文可能并不...</li><li><a href="https://github.com/rust-lang/rust/issues/67805">Resolving the diamond problem · Issue #67805 · rust-lang/rust</a>: 与 #31844 相关。正如 Discord 用户 pie_flavor#7868 所指出的，我们看到了一个关于菱形继承问题的错误：https://play.rust-lang.org/?version=nightly&amp;mode=debug&amp;edition=2018&amp;gist=f7b45b0efa1...</li><li><a href="https://github.com/modular/max/blob/main/mojo/stdlib/test/builtin/test_tuple.mojo">max/mojo/stdlib/test/builtin/test_tuple.mojo at main · modular/max</a>: MAX 平台（包含 Mojo）。通过在 GitHub 上创建一个账户来为 modular/max 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1354566465835106335)** (52 messages🔥): 

> `模型参数详解, ComfyUI InfiniteYou, HuggingFace Inference API 定价, OpenAI 4o 数据集` 


- **揭秘模型参数**：一位成员澄清了 **parameters**（如 **weights** 和 **biases**）如何决定模型处理输入和生成输出的方式，其中 [输入层充当“门”的角色](https://example.com/door)。
   - 该解释详细说明了流程：**Text > Tokenizer > Input Layer > Embedding Layer > Hidden Layers > Output Layer > Tokenizer**，强调了**输入层作为 lookup table 的作用**。
- **ByteDance 的 InfiniteYou 接入 ComfyUI 工作流**：**ByteDance** 的 **InfiniteYou** 模型旨在灵活重塑照片并保留个人特征，目前已集成到 ComfyUI 中，详见 [此 GitHub 仓库](https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou)。
   - 该集成旨在为生成多样化、高质量图像提供*无缝且高效的体验*。
- **HF Inference API 定价：免费层级限制**：一位成员分享了 [HuggingFace 的 API 定价文档](https://huggingface.co/docs/api-inference/pricing)，其中规定了用户运行 **HF Inference API** 每月可获得的 **monthly credits** 额度。
   - 当每月额度耗尽时，*免费用户*将无法再查询 Inference API，而 *PRO 或 Enterprise Hub 用户*将在订阅费之外按请求量付费。
- **新 OpenAI 4o 数据集发布**：发布了一个针对 **OpenAI 4o 模型** 的新数据集，包含来自约 **45,000** 名独立标注员的超过 **200,000 条人类回复**，用于评估 [此 HuggingFace 页面](https://huggingface.co/datasets/Rapidata/OpenAI-4o_t2i_human_preference) 中的**偏好、连贯性和对齐 (preference, coherence, and alignment)**。
   - 这些数据是使用 **Rapidata Python API** 在不到半天的时间内收集完成的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/api-inference/pricing">定价和速率限制</a>：未找到描述</li><li><a href="https://tenor.com/view/travolta-network-messy-network-where-cable-data-center-gif-13242014052849642896">Travolta Network GIF - Travolta Network Messy network - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou">GitHub - ZenAI-Vietnam/ComfyUI_InfiniteYou: InfiniteYou 的一种实现</a>：InfiniteYou 的一种实现。通过在 GitHub 上创建账号为 ZenAI-Vietnam/ComfyUI_InfiniteYou 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/Rapidata/OpenAI-4o_t2i_human_preference">Rapidata/OpenAI-4o_t2i_human_preference · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1354850553007702237)** (1 messages): 

> `LLMs, Transformers, 新手指南` 


- **新手寻求 LLMs 和 Transformers 指导**：一位新成员介绍了自己，并寻求关于 **LLMs** 和 **Transformers** 的通用指导。
- **社区欢迎新成员**：几位成员对新成员表示欢迎，并提供了学习 **LLMs** 和 **Transformers** 的帮助，推荐了资源并表示愿意回答问题。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1354594303858446488)** (6 messages): 

> `用于 Vite 前端的 Windsurf, ComfyUI InfiniteYou 集成` 


- **Windsurf 乘着 Vite 的浪潮**：一位成员建议使用 **Windsurf** 来构建 **Vite** 前端，强调了它对前端开发经验较少的人的吸引力，以及将 **Claude** 连接到 **Vite** 的潜力。
   - 他们还分享了将开放的 **AICUA** 算子通过 **Sage maker** 终端节点 **API gateway** 直接连接到 **Vite** 的方法，作为 **Cursor** 的替代方案，适合那些预算有限且偏好 **OpenAI** 模型和 **AWS** 架构的用户。
- **InfiniteYou 进驻 ComfyUI**：ByteDance 的 **InfiniteYou** 模型旨在灵活重塑照片并保留个人特征，已通过 [ComfyUI InfiniteYou](https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou) 集成到 **ComfyUI** 平台。
   - 该集成旨在提供无缝体验，在生成多样化、高质量图像的同时保持独特的面部特征。



**提到的链接**：<a href="https://github.com/ZenAI-Vietnam/ComfyUI_InfiniteYou">GitHub - ZenAI-Vietnam/ComfyUI_InfiniteYou: InfiniteYou 的一种实现</a>：InfiniteYou 的一种实现。通过在 GitHub 上创建账号为 ZenAI-Vietnam/ComfyUI_InfiniteYou 的开发做出贡献。

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1354538806040203515)** (12 条消息🔥): 

> `sieves zero-shot NLP 流水线，llama-cpp-connector 视觉模型更新，用于自定义模型构建的 HFInheritedModelConfig，Morphos Web 工具` 


- ****Sieves** 简化 Zero-Shot NLP 流水线**：一位成员介绍了 **Sieves**，这是一个仅使用 Zero-Shot 生成模型构建 NLP 流水线而无需训练的 [工具](https://github.com/mantisai/sieves)。
   - 它提供了预实现的常用 NLP 任务，通过利用 **Outlines**、**DSPy** 和 **LangChain** 等库的结构化输出功能，确保生成模型的输出正确。
- ****llama-cpp-connector** 在 Python 中释放视觉模型能力**：一个团队发布了 [llama-cpp-connector](https://github.com/fidecastro/llama-cpp-connector)，这是 **llama.cpp** 的 Python 连接器，使 Python 开发者能够轻松使用其视觉模型（如 **Gemma 3**）。
   - 它的创建是为了解决 *llama-cpp-python* 等封装库的滞后以及 *llama-server* 缺乏视觉模型支持的问题。
- ****HFInheritedModelConfig** 支持自定义模型构建**：一位成员分享了一个使用 **HFInheritedModelConfig** 的自定义模型构建器，旨在从现有的 Hugging Face Hub 模型创建模型，同时覆盖配置参数和层。
   - 这种方法简化了模型自定义，避免了直接修补 Hugging Face 组件的需要。
- ****Morphos** 提供基于 Web 的图像分类器训练**：一位成员介绍了 [Morphos](https://github.com/SanshruthR/Morphos)，这是一个用于训练图像分类器的 Web 工具，支持摄像头/上传和实时预览。
   - 该工具使用户能够直接从 Web 界面训练图像分类器。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/SanshruthR/Morphos">GitHub - SanshruthR/Morphos: Web tool for training image classifiers with webcam/upload support and real-time preview.</a>: 支持摄像头/上传和实时预览的图像分类器训练 Web 工具。 - SanshruthR/Morphos</li><li><a href="https://github.com/fidecastro/llama-cpp-connector">GitHub - fidecastro/llama-cpp-connector: Super simple python connectors for llama.cpp, including vision models (Gemma 3, Qwen2-VL). Compile llama.cpp and run!</a>: 极其简单的 llama.cpp Python 连接器，包含视觉模型 (Gemma 3, Qwen2-VL)。编译 llama.cpp 即可运行！ - GitHub - fidecastro/llama-cpp-connector: Super simple python connectors for ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1354539517276590266)** (6 条消息): 

> `图像参考点，Kaggle 上的 Qwen 2.5 VL 模型，Qwen 2.5 VL 的内存错误，用于 GPU Offloading 的 Flash Attention 2` 


- **关于图像参考地标的讨论**：一位成员询问是否可以使用图像中的参考点（如**标志**、**螺栓**或**图像元数据**）来辅助某些计算机视觉任务。
   - 另一位成员回应称，图像差异很大，且由于通常来自照相手机，**焦距元数据**不可用。
- **用户在导入 Qwen 2.5 VL 模型时遇到困难**：一位成员最初在 Kaggle 上使用最新的 transformers 库 (**4.50.0.dev0**) 导入 **Qwen 2.5 VL 模型**时遇到困难，遇到了与 *Qwen2_5_VLForConditionalGeneration* 相关的导入错误。
   - 经过进一步调试，该成员确定并解决了问题，理由是*与其他包存在某些不匹配*。
- **成员遇到内存错误**：在解决导入问题后，该成员随后询问在 Kaggle 上运行 **Qwen 2.5 VL 3b** 以描述一段 **10 秒视频** 的情况，但面临持续的内存错误。
   - 建议他们尝试更强大的硬件、更小的模型，或尝试 **Flash Attention 2** 以帮助 GPU Offloading。
- **Flash Attention 2 前来救场！**：一位成员建议尝试 **Flash Attention 2**，以缓解在 Kaggle 上运行 **Qwen 2.5 VL 3b** 模型进行视频描述时的 GPU 内存问题。
   - 原帖作者表示感谢，并打算尝试建议的解决方案。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1354550507204640939)** (16 messages🔥): 

> `SetFit v4 发布，Reranker 模型，LLM 生成 JSON，将 PDF 转换为 JSON，在预防措施数据上训练模型` 


- **SetFit V4 发布并增强 Sentence Transformers**：一位成员分享了 Hugging Face 上 **SetFit v4** 发布博客文章的链接，重点介绍了作为 **Sentence Transformers** 一部分的 **Reranker** 模型训练，并讨论了其特性和功能，例如 [Reranker 模型](https://huggingface.co/blog/train-reranker)。
   - 与 SetFit 不同，这些 **Reranker** 模型用于对文本对（通常是查询和答案）进行分类，常在两阶段检索器-重排序器（retriever-reranker）流水线中担任“重排序器”。
- **LLM 可以通过多种模式输出 JSON**：一位用户询问关于使用 **LLM** 生成 **JSON** 文本的问题，另一位用户确认可以通过在 Prompt 中提供示例或使用 **JSON** 模式从 **ChatGPT** 等 **LLM** 获取 **JSON** 输出。
   - 会议强调，虽然 **LLM** 可以生成 **JSON**，但它们可能不适合直接处理大型数据（如 PDF），建议对大型文档进行分块（chunking）。
- **通过分块将 PDF 转换为 JSON**：一位用户询问如何将大型 PDF 转换为 **JSON** 格式以进行微调（finetuning），另一位用户建议对大型文档进行分块。
   - 有人指出，虽然直接将 PDF 转换为 **JSON** 对大文件来说具有挑战性，但可以先从 PDF 中提取数据，转换为 CSV，然后再转换为 **JSON**，或者在输入 **LLM** 之前采用分块策略。
- **在预防措施数据上训练模型**：一位用户正在寻求在预防措施数据上训练模型的方法，将其转换为 **JSON** 进行微调，目标是根据危险等级提供预防措施。
   - 建议了一个替代方案，即使用聊天机器人数据和 CTG 数据配合 **RAG**（检索增强生成）来获取预防措施。



**提及的链接**：<a href="https://huggingface.co/blog/train-reranker">使用 Sentence Transformers v4 训练和微调 Reranker 模型</a>：未找到描述

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1354565562574831707)** (4 messages): 

> `AI Agents 课程，Smol 课程` 


- **提供 Agents 课程链接**：一位成员提供了 [AI Agents 课程](https://huggingface.co/learn/agents-course/unit0/introduction) 的链接。
   - 该课程旨在带领参与者开启一段*从入门到精通*的旅程，以理解、使用和构建 **AI agents**。
- **请求并提供 Smol Course 链接**：一位成员请求了 **Smol Course** 的链接，随后得到了回复：[smol-course](https://github.com/huggingface/smol-course)。
   - 该 **GitHub** 仓库被描述为“关于对齐 smol 模型的课程”。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建账号为 huggingface/smol-course 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1354562677766361203)** (16 条消息🔥): 

> `课程单元发布日期、Hugging Face Token 设置、Gemini vs. You.com、Agent 构建创意、LLM 评估器问题` 


- **课程单元发布日期推迟**：课程单元的发布时间正在调整，单元 3 已移至 **4 月 1 日**，这引发了关于原定于 **4 月 8 日**发布的单元 4 是否也会同样推迟的疑问。
   - 虽然一位成员最初认为单元 3 和 4 都会在下周发布，但随后得到澄清：下一个单元将侧重于构建一个 Agent，而最终作业仍将是独立的。
- **HF Token 问题已解决**：一位成员在本地使用 **Qwen2.5-Coder-32B-Instruct 模型**运行第一个 Agent 时遇到了**身份验证错误**。
   - 该问题通过使用 `os.environ['HF_TOKEN']="hf_myt0k3n"` 设置 **HF_TOKEN** 得到了解决。
- **Gemini 在与 You.com 的竞争中取得进展**：成员们讨论了使用 **Google Gemini** 替代 **You.com** 来完成课程相关任务。
   - 一位成员自 [单元 2.1](https://discord.com/channels/879548962464493619/1348087043271430164) 以来一直使用 **Gemini**，并表示 **2.0** 版本运行完美，尽管 **2.5** 版本最近也已发布；不过有人指出它*同样存在限制*。
- **需要 Agent 构建灵感？**：一位最近完成 **单元 2.1** 的成员正在寻求构建 Agent 的点子，以巩固所学知识。
   - 他们正在考虑创建一个用于从 Spaces 进行 **2D 到 3D 转换**的工具，但不确定下一步该如何进行。
- **Toxicity LLM 评估器的敏感面**：一位在 Langfuse 中测试 **toxicity LLM-as-a-judge** 的成员发现，它错误地将 Prompt *"吃胡萝卜能改善视力吗？"* 标记为具有毒性（**0.9**）。
   - 给出的理由是：假设的回答中包含*对气候变化信奉者的轻蔑语气*，尽管实际回答与该主题毫无关系；该用户使用的模型是 **OpenAI gpt-4o**。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1354532838690525389)** (94 条消息🔥🔥): 

> `MCP 在 OpenAI 产品中的采用、MCP 对业务及未来的影响、Cloudflare 的 MCP 工具链、来自 GitHub 的 MCP 安全风险、Claude 的 MCP 服务器实现问题` 


- **Sama 发出信号：MCP 即将登陆 OpenAI 产品！**：OpenAI CEO **Sam Altman** 宣布 [MCP 支持](https://x.com/sama/status/1904957253456941061?t=awjb86WjJSH4MlFo9l5sWw&s=19) 即将引入 OpenAI 产品，包括 **Agents SDK**、**ChatGPT 桌面应用**和 **Responses API**。
   - 成员们认为这是 MCP 成为执行业务相关任务的 Agent 骨干的重要一步，类似于 HTTP 对互联网的影响。
- **Cloudflare 增强上下文：通过远程服务器工具链进军 MCP**：Cloudflare 宣布[支持构建和部署远程 MCP 服务器](https://blog.cloudflare.com/remote-model-context-protocol-servers-mcp/)，并提供了 **workers-oauth-provider** 和 **McpAgent** 等工具。
   - 这一举措被视为重大进展，为开发者提供了更高效构建 MCP 服务器的资源。
- **规范问题浮现：Claude 在处理 MCP Prompt 和资源时遇到困难**：用户报告了 **Claude Desktop** 在 MCP 服务器包含资源或 Prompt 时的问题，导致其不断循环查询。
   - 一位成员建议了一种解决方法，即移除相关 capabilities 以防止 Claude 搜索缺失的资源和 Prompt，并向 GitHub 推送了包含修复的新版本。
- **服务器端见解：在 MCP 服务器上同时放置 Prompt 和 Tool？**：成员们讨论了将 Tool 与 Prompt 存储在同一个 MCP 服务器上以确保 Tool 被正确调用的方案，特别是对于像 **Stable Diffusion** 这样的工具。
   - 建议 Prompt 应直接引用服务器上可用的 Tool 名称来引导 Agent 行为，例如 *First call ${tool1.name}, then ${tool2.name}*。
- **ICL 集成见解：Prompt 提升 In-Context Learning**：小组讨论了利用 MCP Prompt 进行 **in-context learning (ICL)** 的方法，通过 Prompt 鼓励特定的 Agent 行为并引导 Tool 使用，并引用了[一个示例](https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19)，说明可以设定“针对此 Prompt 使用该工具”。
   - 有人指出，为了使 ICL 有效工作，MCP 客户端需要将 User/Assistant 对作为独立消息添加到上下文窗口中，而不仅仅是附加一个 JSON 对象。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/sama/status/1904957253456941061?t=awjb86WjJSH4MlFo9l5sWw&s=19">Sam Altman (@sama) 的推文</a>：人们非常喜欢 MCP，我们很高兴能在我们的产品中增加支持。今天已在 Agents SDK 中提供，ChatGPT 桌面应用和 Responses API 的支持也即将推出！</li><li><a href="https://glama.ai/api/mcp/openapi.json"">MCP API 参考</a>：Glama Gateway 的 API 参考</li><li><a href="https://techcrunch.com/2025/03/26/openai-adopts-rival-anthropics-standard-for-connecting-ai-models-to-data/">OpenAI 采用竞争对手 Anthropic 的标准将 AI 模型连接到数据 | TechCrunch</a>：OpenAI 正在采用竞争对手 Anthropic 的标准——Model Context Protocol (MCP)，用于将 AI 助手连接到存储数据的系统。</li><li><a href="https://glama.ai/mcp/servers/@matthewhand/mcp-openapi-proxy">mcp-any-openapi</a>：一个基于 Python 的 MCP 服务器，将 OpenAPI 描述的 REST API 集成到 MCP 工作流中，实现将 API 端点动态暴露为 MCP 工具。</li><li><a href="https://github.com/yuniko-software/minecraft-mcp-server">GitHub - yuniko-software/minecraft-mcp-server：一个由 Mineflayer API 驱动的 Minecraft MCP 服务器。它允许实时控制 Minecraft 角色，使 AI 助手能够通过自然语言指令构建建筑、探索世界并与游戏环境互动</a>：一个由 Mineflayer API 驱动的 Minecraft MCP 服务器。它允许实时控制 Minecraft 角色，使 AI 助手能够构建建筑、探索世界并与游戏...</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/188#issue-2895415136">在 GetPrompt 中为上下文学习添加了 Tool Call 和 Tool Result … 由 evalstate 提交 · Pull Request #188 · modelcontextprotocol/specification</a>：…在 PromptMessage 中添加了 ToolCall 和 ToolResult 块，以允许对工具使用模式和错误处理进行上下文学习。作为草案提交，待评审完成后再完成...</li><li><a href="https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19">llmindset (@llmindsetuk) 的推文</a>：让我们来看看一个被低估的 MCP 功能：Prompts——以及为什么它们对于基于 Agent 的应用非常重要。我们将从两个简单的 Agent 开始，它们返回物体的大小——在...</li><li><a href="https://blog.cloudflare.com/remote-model-context-protocol-servers-mcp/">在 Cloudflare 上构建并部署远程 Model Context Protocol (MCP) 服务器</a>：你现在可以在 Cloudflare 上构建和部署远程 MCP 服务器，我们会为你处理构建远程 MCP 服务器的难点。与你之前可能使用过的本地 MCP 服务器不同，远程 MCP 服务器...</li><li><a href="https://github.com/cline/cline/blob/main/src/core/prompts/system.ts">cline/src/core/prompts/system.ts at main · cline/cline</a>：直接在你的 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器，并在每一步都获得你的许可。 - cline/cline
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1354598656526778549)** (12 条消息🔥): 

> `Canvas MCP, Truto's SuperAI, Model Context Protocol (MCP), Gradescope 集成, 用于 MCP 服务器的 Docker Compose` 


- ****Canvas MCP 连接大学课程****：一位成员创建了一个 [Canvas MCP](https://git.new/canvas-mcp)，可以连接到大学课程，实现对资源和即将到来的作业的自动查询。
   - 另一位成员请求集成 **Gradescope**，创作者回应称他们 *添加了另一个可以自主爬取 Gradescope 以查找信息的 Agent，请在 [Canvas MCP](https://git.new/canvas-mcp) 尝试*。
- ****Truto 在 SuperAI 上发布 Agent 工具集****：**Truto** 推出了 [Truto SuperAI 平台上的 Agent 工具集](https://www.linkedin.com/posts/gettruto_introducing-truto-agent-toolsets-as-part-activity-7310671157647523840-Sm77)，这是一个专门为构建 AI 产品和功能的公司的平台。
   - 链接的 **LinkedIn** 帖子鼓励用户点赞或评论，以帮助团队触达更多团队。
- ****使用 Javascript 构建 MCP 服务器****：一位成员分享了 [Lokka](https://medium.com/@kenzic/getting-started-build-a-model-context-protocol-server-9d0362363435) 的文章，讨论了如何为 Hacker News 构建一个 Javascript **Model Context Protocol (MCP)** 服务器。
   - 链接的文章强调，**Model Context Protocol (MCP)** *旨在解决* 将 **LLMs** 集成到真实产品中的混乱局面。
- ****用于 MCP 服务器的一站式 Docker Compose****：一位成员创建了一个 [一站式 docker-compose](https://github.com/JoshuaRL/MCP-Mealprep)，让用户可以从 Portainer（或任何地方）轻松地自托管 **17 个 MCP** 服务器。
   - 该 compose 从公共 **GitHub** 项目中获取 **Dockerfiles**，因此当它们更新时，也应该能获取到新内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@kenzic/getting-started-build-a-model-context-protocol-server-9d0362363435">入门指南：构建 Model Context Protocol 服务器</a>：简化 LLM 集成：为 Hacker News 构建 JavaScript MCP 服务器</li><li><a href="https://github.com/JoshuaRL/MCP-Mealprep">GitHub - JoshuaRL/MCP-Mealprep：该项目从 GitHub 位置获取多个 MCP 服务器，将它们与引用的 Dockerfiles 打包在一起，并使用 docker-compose 将它们拉取在一起，作为 ML/AI 资源的堆栈运行。</a>：该项目从 GitHub 位置获取多个 MCP 服务器，将它们与引用的 Dockerfiles 打包在一起，并使用 docker-compose 将它们拉取在一起，作为 ML/AI 资源...</li><li><a href="https://git.new/canvas-mcp)">Git.new – 免费 GitHub 链接缩短器</a>：使用 git.new 缩短您的 GitHub URL 或链接 – 一个由 Dub.co 提供支持的免费品牌 GitHub URL 缩短器</li><li><a href="https://git.new/canvas-mcp">GitHub - aryankeluskar/canvas-mcp：Canvas LMS 和 Gradescope 工具集，打造终极 EdTech Model Context Protocol。允许您在自选的 AI 应用中查询课程、查找资源并就即将到来的作业进行对话。立即尝试！</a>：Canvas LMS 和 Gradescope 工具集，打造终极 EdTech Model Context Protocol。允许您在自选的 AI 应用中查询课程、查找资源并就即将到来的作业进行对话...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1354601205325889608)** (1 条消息): 

> `Mind Map 公开发布` 


- ****Mind Map** 正式公开！**：**Mind Map** 功能现在已对 Notebook LM 的所有用户 100% 公开。
   - 团队对用户的耐心、喜爱和反馈表示感谢，公告中包含一张 [感谢图片](https://cdn.discordapp.com/attachments/1182376564525113484/1354601205091012668/NotebookLM_Thank_You.png?ex=67e733ae&is=67e5e22e&hm=feb1a12774e69af171073ec1d544ecbffb8eee2811ef0e42e7768cc89b669f91)。
- **耐心终有回报，Mind Map 解锁**：经过一段时间的期待，Notebook LM 上的 **Mind Map** 功能现在所有人都可以访问。
   - 此次发布以表达对社区持久支持和深刻见解贡献的感谢来庆祝。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1354660754216718426)** (8 条消息🔥): 

> `Spanish podcasts not working, Sharing Notebooks issues, Company research for cover letters and resumes` 


- **西班牙语播客创建停滞**：有用户报告称，在“customize”设置中生成**西班牙语播客 (podcasts in Spanish)** 的功能已失效。
   - 他们寻求解决此问题的建议或评论，表明该功能此前可用但现在已停止。
- **Notebook 共享困扰 Pro 用户**：一位 **Pro 用户**报告称无法通过链接共享 Notebooks，尽管笔记本中包含的是来自 YouTube 的公开信息。
   - 提到的潜在解决方案包括确保接收者拥有活跃的 **NLM 账户**，以及手动通过电子邮件发送链接。
- **学生简化公司调研流程**：一名学生开发了一套系统来简化公司调研，以撰写更具影响力的求职信和简历，获得了 **80% 的评分**。
   - 他们将公司网站、报告和新闻源上传到 Notebook LM，从而能够获得关于公司的详细且有参考依据的回答并节省时间，但生成的求职信结果过于平淡，评分仅为 **10%**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1354531175351193663)** (83 条消息🔥🔥): 

> `Gemini 2.5 Pro Turkey Test, Gemini Advanced Research Limit, NotebookLM API and Podcast Creation, Mind Map Improvements, Gemini 2.0 Flash Readability` 


- **Gemini 进行火鸡测试 (Turkey Test)**：一位成员使用“火鸡测试”测试了 **Gemini 2.5 Pro**，要求它写一首关于鸟的形而上学诗歌，并在此处分享了带有 **NotebookLM 评论**的视频 [here](https://youtu.be/MagWnkL14js?si=ywCvQQY12Kruh6aZ&t=54)。
   - 用户通过**交互模式 (Interactive Mode)** 引导评论走向令人满意的结局，无意中发现了 **NBLM** 的新用途。
- **Gemini Advanced 存在研究限制**：一位成员询问了 **Gemini Advanced** 的深度研究限制，另一位成员回答说是**每天 20 份研究报告**。
   - 第一位成员认为这“与 ChatGPT 相比相当不错”。
- **通过 API 创建播客正在开发中？**：一位成员建议通过 **NotebookLM API** 加入播客创建功能，并指出利用该功能可以做出一些非常酷的东西。
   - 他们有一个带有思考模型的 Discord Bot，支持函数调用 (function calling)，且 Action Bot 在函数使用方面表现得更加一致。
- **思维导图 (Mind Map) 需求**：一位成员请求开发人员改进刚发布的思维导图功能。
   - 该成员指出，思维导图虽然结构整齐，但“因为缺乏描述而浪费时间”，而且用户无法控制思维导图的结构或描述的详细程度。
- **可读性 (Readability) 下降**：一位成员发现，在 **NotebookLM** 中使用 **Gemini 2.0 Flash** 时，尽管它能产生更丰富、更好的回答，但*可读性*有所下降。
   - 阅读起来变得有些困难，因为 **Gemini** 倾向于先解释/回答，然后再解释原因；或者 **Gemini** 先解释基础知识，然后在回复的中间或某个地方给出答案。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1354541492504367176)** (33 条消息🔥): 

> `Sketch-to-Model 流水线、Kernel Attention (KA) 的替代方案、AI 解决谜题、ChatGPT 和 Grok 3 用于 UX/UI、AI/ML 中的信息论` 


- **Sketch-to-Model 流水线引发讨论**：一名成员提出了一个 "Sketch-to-Model" 流程（**Sketch** --> **2D/2.5D 概念/图像** --> **3D 模型** --> **仿真**），并对 Kernel Attention (KA) 的替代方案表示了兴趣。
   - 他们注意到 **ChatGPT** 奇怪地提到了一个与 **KAN** 相似但不同的概念，并称 **KAN** 来自 **Google DeepMind**，而 **Grok 3** 则提到 **xAI** 团队正在积极研究 **KAN**。
- **AI 的解谜能力面临考验**：成员们讨论了 AI 是否能解决谜题书《Maze: Solve the World's Most Challenging Puzzle》（[维基百科链接](https://en.wikipedia.org/wiki/Maze:_Solve_the_World%27s_Most_Challenging_Puzzle)）。
   - 一位成员建议训练 LLM 来解决 ARG 和旧的解谜游戏，而另一位成员指出某些谜题是刻意设计的，目前的推理模型可能无法解决。
- **AI 在 UX/UI 设计方面表现不佳**：成员们分享了测试 **ChatGPT** 和 **Grok 3** 进行 UX/UI 设计和建筑规划的经历，结果令人失望。
   - 一位成员建议，对于应用或建筑布局，他们需要*结构化推理*，并指出了这篇论文 [LayoutGen: Layout Generation with Box-wise Diffusion](https://arxiv.org/abs/2503.17407)。
- **简历掠夺者：AI 在污染申请者池中的角色**：一位成员询问如何批量生成虚假简历以*污染申请者池*，从而让稍好一点的候选人显得更出众。
   - 共识是，无论如何 HR 都会设定一个基准，而且这种信息很可能涉及违法。
- **信息论支撑着被“精挑细选”的 AI**：一位成员指出，AI 和 ML *从信息论中精挑细选*，主要使用熵（entropy）和散度（divergence），但没有使用条件版本。
   - 他们认为，利用信息论的更多方面可以带来更好的泛化、记忆、可解释性和效率，并链接了 [牛津大学数学系关于信息论的讲座](https://www.youtube.com/playlist?list=PL4d5ZtfQonW3iAhXvTYCnoGEeRhxhKHMc)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Maze:_Solve_the_World%27s_Most_Challenging_Puzzle">Maze: Solve the World&#039;s Most Challenging Puzzle - Wikipedia</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2503.17407">A Comprehensive Survey on Long Context Language Modeling</a>：高效处理长上下文一直是自然语言处理中持久的追求。随着长文档、对话和其他文本数据的日益增多，开发...非常重要</li><li><a href="https://arxiv.org/abs/2009.14794">Rethinking Attention with Performers</a>：我们介绍了 Performer，这是一种 Transformer 架构，可以以可证明的精度估计常规（softmax）全秩注意力 Transformer，但仅使用线性（而非二次）空间...</li><li><a href="https://www.youtube.com/playlist?list=PL4d5ZtfQonW3iAhXvTYCnoGEeRhxhKHMc">Student Lectures - Information Theory</a>：选自 Sam Cohen 三年级学生课程的八场讲座。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1354532976397914312)** (18 messages🔥): 

> `Discord Timestamps, Chain-of-Bias issues, Paper Discussion Format, Tracing Thoughts in a Language Model, Attribution Graphs` 


- **Discord Timestamps 自动转换**：Discord 时间戳会自动以查看者的本地时间显示，简化了 [Discord Timestamps](https://r.3v.fi/discord-timestamps/) 中描述的日程安排。
   - 这消除了时区转换的**困惑**。
- **Chain-of-Bias 系统性错误**：一位成员评论说 **Chain-of-Bias** 仅在一定程度上有帮助，因为*当它“出错”时，是“系统性错误”。*
   - 因此，这意味着它的用途有限。
- **论文讨论采用流式直播格式**：论文讨论通常涉及有人在 **stream**（直播）中打开论文并阅读，其他人贡献经验和知识。
   - 通过 Discord 活动系统，还可以进行准备更充分、质量更高的每周讨论。
- **Tracing Thoughts in a Language Model 双论文深度解析即将到来**：成员们将分析 [Tracing Thoughts in a Language Model](https://www.anthropic.com/research/tracing-thoughts-language-model) 及其 [相关的 YouTube 视频](https://youtu.be/Bj9BD2D3DzA)。
   - 鉴于两者的长度，可能会涉及多次会议。
- **Attribution Graphs 链接分享**：一位成员分享了 [Transformer Circuits attribution graphs methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 和 [Transformer Circuits attribution graphs biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 的链接。
   - 这些图表旨在提供对 **Transformer** 功能的见解。



**提及的链接**：<a href="https://arxiv.org/abs/2503.19551">Scaling Laws of Synthetic Data for Language Models</a>：大型语言模型 (LLM) 在各种任务中表现出强大的性能，这主要由预训练中使用的高质量网络数据驱动。然而，最近的研究表明，这一数据源正在迅速……

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1354875115585933604)** (14 messages🔥): 

> `Alan Turing Institute Crisis, GPT-4o Autoregressive Image Generation, Image Token Reusal` 


- **Alan Turing Institute 面临危机与裁员**：尽管在 2024 年获得了 1 亿英镑的新资金结算，但 [Alan Turing Institute (ATI)](https://www.turing.ac.uk/) 正在准备大规模裁员，并计划[削减四分之一的研究项目](https://www.researchprofessionalnews.com/rr-news-uk-research-councils-2025-3-alan-turing-institute-axes-around-a-quarter-of-research-projects/)。
   - 据报道，员工因裁员而处于*公开反抗*状态。
- **GPT-4o 确认为 Autoregressive 图像模型**：**GPT-4o** 已被确认为 **Autoregressive** 图像生成模型，详见 [OpenAI's Native Image Generation System Card](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf)。
- **关于 GPT-4o 中 Image Token 复用的推测**：有推测认为 **GPT-4o** 在图像输出中复用了图像输入 **Token**，这表明它可能输出与输入相同格式的图像 **Token**。
   - 据观察，当被要求精确复制图像时，该模型会引入细微的变化，这表明它使用的是 **semantic encoder/decoder** 而非像素级编码。一位成员表示，*OpenAI 一直在积攒实力，直到 Google 发布一个好模型来转移注意力。*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Yampeleg/status/1905293247108219086">Yam Peleg (@Yampeleg) 的推文</a>：所以 GPT-4o 被确认为是一个 Autoregressive 图像生成模型。这到底是怎么做到的。致敬。https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf</li><li><a href="https://www.chalmermagne.com/p/how-not-to-build-an-ai-institute">如何不建立一个 AI 研究院</a>：Alan Turing Institute 出了什么问题？
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1354545555434504445)** (2 messages): 

> `DP 和 TP rank 中的数据分布，TRL 对数据分布的处理` 


- **数据处理深度解析：DP vs. TP Ranks**：一位成员澄清道，在分布式处理（**DP**）中，每个 rank 接收不同的数据，而在张量并行（**TP**）中，所有 rank 接收相同的数据。
   - 他们建议 **TRL** (Transformer Reinforcement Learning) 应该已经自动管理了这种分布。
- **TRL 框架自动管理数据分布**：一位成员指出，**TRL** 框架的设计初衷很可能就是为了自动处理跨不同 rank 的数据分布。
   - 这确保了在使用 **DP** 和 **TP** 的分布式环境中能够高效地进行训练并利用资源。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1354770489687671006)** (8 messages🔥): 

> `Triton 中的 Pre/Post Hooks，Hopper 的 num_ctas，Local Tensor 扩展` 


- **Triton Autotune 不支持 Pre/Post Hooks**：由于需要在运行时执行 Python 代码，`triton.Autotune` 或 `triton.Config` 不支持 *pre_hook* 和 *post_hook*，而 **Inductor** 在 AOTI 中无法支持这一点。
   - 一位成员表示，实现这一支持应该并不困难。
- **Hopper 的 num_ctas 之谜**：似乎没有人能在 **Hopper** 上成功使用大于 1 的 `num_ctas` 值，因为会导致崩溃或抛出 `RuntimeError: PassManager::run failed` 异常。
   - 这些问题的根本原因尚不明确。
- **Local Tensor 扩展的困扰**：一位成员在将使用 `torch.Tensor.expand()` 的代码移植到 Triton 时遇到问题，原因是 local tensor 不支持用于重复元素的 tensor index。
   - 虽然 `load()` 允许在 ptr 参数中使用重复索引，但 local tensor 索引却不支持。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1354835229214511396)** (4 messages): 

> `Memory Coalescing，CUDA 内存层级` 


- **详解 Memory Coalescing**：一位用户询问，如果数据在寄存器中不连续，但全局内存中的地址是连续的，那么合并（coalescing）是否要求数据必须连续存储。
   - 另一位用户澄清说，在这种情况下，无论数据写入共享内存（**SMEM**）还是寄存器内存（**RMEM**）的什么位置，读取操作确实都会被合并（coalesced）。
- **CUDA 内存层级解释**：一位用户解释了 CUDA 内存层级，以说明 **Memory Coalescing** 的好处。
   - 他们澄清说，**DRAM** 和 **SRAM** 之间的数据移动速度很慢，并强调了最大化全局内存与共享内存之间数据传输效率的重要性。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1354933107500777554)** (1 messages): 

> `PyTorch profiler，profiler trace` 


- **难以定位 PyTorch Profiler 中的 `save` 调用**：一位用户在 PyTorch profiler trace 中难以找到 `save` 函数被调用的确切位置。
   - 他们看到了许多认为与之对应的 `detach` 和 `copy` 调用，但随后出现了一个断层，在任何 stream 或 thread 中都看不到任何内容。
- **寻求使用 PyTorch Profiler 的帮助**：一位用户需要帮助在 profiler trace 中找到 PyTorch profiler 调用 **save** 函数的确切代码行。
   - 虽然存在许多被认为相关的 **detach** 和 **copy** 调用，但在任何 stream 或 thread 中都看不到内容的断层。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1354961720866636032)** (3 messages): 

> `Red Hat, Software Engineer, C++, GPU Kernels, CUDA` 


- **Red Hat 寻求精通 GPU 的工程师**：Red Hat 正在招聘不同级别的全职**软件工程师**，要求具备 **C++、GPU kernels、CUDA、Triton、CUTLASS、PyTorch 和 vLLM** 方面的经验。
   - 有意向的候选人请将简历和相关经验总结发送至 terrytangyuan@gmail.com，并在邮件主题中注明 "GPU Mode"。
- **Red Hat 正在招聘**：加入 Red Hat 担任软件工程师，从事前沿技术工作。
   - 该职位要求具备 C++、GPU kernels、CUDA 及其他相关领域的经验。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1354620215010398300)** (1 条消息): 

> `Knowledge Distillation for Video Models, Estimating Model Parameters, Estimating Inference Throughput on Consumer GPUs` 


- **为消费级 GPU 蒸馏视频模型**：一位成员正在寻求通过 [knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation)（知识蒸馏）缩小一个 **28M 参数的视频模型**的方法，以便在消费级 GPU 上进行实时推理。
   - 目标是估算生成的模型大小和每秒帧数 (**FPS**) 性能，同时考虑 GPU 架构、模型 FLOPs、帧率和参数数量。
- **计算视频模型参数**：该成员正在寻找资源（如指南或博客文章），以便在给定预训练的 **28M 参数视频模型**的情况下，估算适用于消费级 GPU 的参数数量。
   - 目前尚不清楚可以使用什么公式来进行这种估算。
- **估算实时 FPS**：该成员正在寻求关于如何估算蒸馏模型在消费级 GPU 上的推理吞吐量 (FPS) 的见解。
   - 这包括考虑 GPU 架构、**模型 FLOPs**、帧率和参数数量等因素。遗憾的是，目前还没有关于该主题的指南或博客。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1354616156383740045)** (5 条消息): 

> `Blocksparse, TorchAO, Pull Request #1734, Pull Request #1974` 


- **TorchAO 中的 Blocksparse 晋级——是否必要？**：一位成员质疑 [pytorch/ao PR#1734](https://github.com/pytorch/ao/pull/1734) 中与 **blocksparse** 功能晋级相关的特定代码添加是否必要。
   - 另一位成员澄清说，该添加可能是偶然的，对于 **blocksparse** 功能来说是不必要的。
- **解码动态性得到修复**：[PR #1974](https://github.com/pytorch/ao/pull/1974) 旨在*移除解码时的 dynamic=True*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/1974">remove dynamic=True for decode by jcaip · Pull Request #1974 · pytorch/ao</a>：由 jcaip 提交的移除解码时的 dynamic=True</li><li><a href="https://github.com/pytorch/ao/pull/1734">promote blocksparse from prototype, make it faster by jcaip · Pull Request #1734 · pytorch/ao</a>：此 PR 在 torchao 中将 block sparsity 从原型阶段晋级。主要工作是从 core 移植了 triton addmm blocksparse kernel，并对其进行了多项性能改进。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1354924638995615997)** (1 条消息): 

> `Hayao Miyazaki on AI art, Studio Ghibli anime filter` 


- **宫崎骏对 AI 艺术的立场再次引发关注**：一位用户分享了 [X 上的帖子](https://x.com/nuberodesign/status/1904954270119588033)，引用了 **Studio Ghibli**（吉卜力工作室）创始人**宫崎骏**对机器生成艺术的批评观点。
- **吉卜力动漫滤镜走红**：一位用户注意到将个人照片转换为 **Studio Ghibli** 动漫风格的趋势，并认为这是“巨大的机会 (tremendous alpha)”。
   - 该用户建议将这些动漫风格的照片发给自己的妻子。



**提到的链接**：<a href="https://x.com/nuberodesign/status/1904954270119588033">来自 Nuberodesign (@nuberodesign) 的推文</a>：既然这种垃圾正在流行，我们应该看看吉卜力工作室的创始人宫崎骏对机器创造的艺术是怎么说的。引用 Grant Slatton (@GrantSlatton) 的话，这是巨大的机会 (tremendous alpha)...

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 条消息): 

srns27: 天哪我真是瞎了，谢谢兄弟，哈哈
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 条消息): 

nuttt233: 因为batch gemm中默认前两个维度是batch stride，后两维才是row col
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1354873816580624557)** (3 条消息): 

> `ComfyUI, CUDA, load_inline, Triton` 


- **ComfyUI 安装帮助**：一位成员建议遇到安装错误的用户前往 [ComfyUI Discord](https://discord.com/invite/comfyorg) 寻求帮助。
   - 他们建议尝试在一个新的 *conda* 环境中安装 **Triton**。
- **CUDA 语法错误修复**：一位成员在上传 `.cu` 文件时遇到了 *SyntaxError: invalid decimal literal* 错误。
   - 另一位成员建议使用 *pytorch* 中的 `load_inline()` 功能，并参考了 [这个示例](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py)。



**提到的链接**：<a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py">reference-kernels/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py at main · gpu-mode/reference-kernels</a>：排行榜的参考内核。通过在 GitHub 上创建账户来为 gpu-mode/reference-kernels 的开发做出贡献。

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1354910489523781874)** (25 条消息🔥): 

> `Modal Runners, vectorsum, grayscale` 


- **grayscale 排行榜成功提交**：一个 ID 为 **3169** 的基准测试提交到 `grayscale` 排行榜（使用 GPU: **H100** 和 Modal runners）已成功！
- **vectorsum 排行榜成功提交**：多个测试、排行榜和基准测试提交到 `vectorsum` 排行榜（使用 GPU: **T4** 和 **L4** 以及 Modal runners）已成功。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1354892149292011783)** (1 条消息): 

> `LlamaCloud, MCP Server, Claude Desktop` 


- **LlamaCloud 兼作 MCP Server**：**LlamaCloud** 可用作 **MCP server**，为任何 MCP 客户端实现工作流中的实时数据集成。
   - 一段 [视频演示](https://twitter.com/llama_index/status/1905332760614764810) 展示了如何将现有的 **LlamaCloud index** 作为 **Claude Desktop** 所使用的 MCP server 的数据源。
- **Claude 从 LlamaCloud 获取数据**：**Claude Desktop** 可以使用现有的 **LlamaCloud index** 作为 MCP server 的数据源。
   - 这使得最新的实时数据能够集成到 **Claude** 工作流中，详见 [这段视频](https://twitter.com/llama_index/status/1905332760614764810)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1354548850966270032)** (22 messages🔥): 

> `LlamaExtract Schema Inference, TS Chatbot with Postgres DB, E-commerce Chatbot Architecture, SQL Query Generation Issues, Structured Prediction Bug` 


- **LlamaExtract 放弃 Schema 推断**：正如 [LlamaExtract 公告](https://example.com/llamaextract_announcement)中所述，去年宣布的 **LlamaExtract** 中的 Schema 推断功能已被降低优先级，因为大多数用户已经拥有了他们需要的 Schema。
   - 该功能将来可能会回归，但目前正在优先处理其他方面。
- **TS 聊天机器人处理 Postgres 数据**：一位用户询问关于将 **LlamaIndex** 与关系型 Postgres 数据库配合使用的问题，建议使用基于 **LLM 对象**构建的 Text-to-SQL 应用。
   - 由于关系型数据的特性，将数据转换为向量数据库（Vector DB）被认为没有帮助，但有一个 [TS 软件包](https://ts.llamaindex.ai/docs/llamaindex/tutorials/workflow)可能对工作流有所帮助。
- **电商聊天机器人多 Agent 方案**：一位用户考虑在电商聊天机器人中使用包含 React Agent 和功能移交 Agent 的多 Agent 系统，得到的建议是如果 LLM 支持，则坚持使用 **Function Calling Agent**。
   - 建议使用 Websockets 以提供快速回复，但选择取决于整体系统架构。
- **聊天机器人 SQL 查询生成难题**：一位正在构建根据用户消息生成 SQL 查询的聊天机器人的用户报告称，即使在 SQL 文件中有列注释，机器人也无法选择合适的列。
   - 未提供具体解决方案，但建议用户向团队提交 Bug 报告。
- **结构化预测 Bug 导致 Dict 字段失效**：据 [Issue #18298](https://github.com/run-llama/llama_index/issues/18298) 记录，报告了一个结构化预测在处理 Dict 字段时失败的 Bug，这可能是由于 JSON Schema 的限制或 OpenAI 对注解支持的局限性导致的。
   - 解决方法包括为字段提供描述并将类型设置为 Any，或者定义另一个 Pydantic 模型来描述该 Dict 字段。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/18298">[Bug]: Structured Prediction failed with Dict field · Issue #18298 · run-llama/llama_index</a>: Bug 描述：Gemini 和 OpenAI 都无法生成带有 Dict 字段（ingredients）的 recipe 对象。OpenAI 失败详情...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/advanced_text_to_sql/">Workflows for Advanced Text-to-SQL - LlamaIndex</a>: 无描述</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/tutorials/workflow">Basic Usage</a>: 了解如何使用 LlamaIndex 工作流。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1354647129162907718)** (13 messages🔥): 

> `PDF Parsing Tools, LlamaParse and Image Reading, LLMs for Image Captioning, Hybrid Chunking, OCR for Scanned Documents` 


- **LlamaParse 确实是 PDF 解析的解决方案**：用户讨论了将 **LlamaParse** 作为解析 PDF 的最佳工具，一位用户确认在正确配置后它可以很好地读取图像，并[提议测试一个示例文件](https://cdn.discordapp.com/attachments/1354647308180127784/1354651537972138134/pizza.pdf?ex=67e6b9ce&is=67e5684e&hm=07e21377d4529bbf4a54ddfd766780386d693eb55f07881369d0f746e972cdd4)。
   - 值得注意的是，讨论中的 PDF 文件 *pizza.pdf* 不包含实际文本，只有一张图像。
- **LLM 可以为 RAG 生成图像字幕**：一位成员建议使用 **LLM** 来读取并为图像生成字幕，以便在 **RAG** 应用中回答来自上传 PDF 的问题。
   - 另一位成员询问了关于手写数学作业等扫描文档的**混合分块（Hybrid Chunking）**和 **OCR** 问题。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1354611804654932060)** (25 messages🔥): 

> `Nvidia Acquires Lepton AI, Model Context Protocol, Replit Agent v2, GPT-4o Update, OpenAI Image Generation Policy`

- **Nvidia 收购 Lepton AI**：据 [The Information](https://www.theinformation.com/articles/nvidia-nears-deal-buy-gpu-reseller-several-hundred-million-dollars) 报道，**Nvidia** 以数亿美元收购了推理提供商 **Lepton AI**，以增强其提高 GPU 利用率的软件产品。
- **OpenAI 的 Agents 支持 MCP**：[Model Context Protocol](https://modelcontextprotocol.io/introduction) (**MCP**) 现在已连接到 **OpenAI Agents SDK**，允许使用各种 MCP 服务器为 Agents 提供工具。
   - MCP 被描述为 *AI 应用程序的 USB-C 接口*，标准化了应用程序向 LLM 提供上下文的方式。
- **Replit Agent v2 自主性大幅提升**：**Replit Agent v2** 现已开启早期访问，搭载 **Anthropic** 的 **Claude 3.7 Sonnet**，具有更强的自主性，在进行更改前会提出假设并搜索文件，详情见 [Replit 博客](https://blog.replit.com/agent-v2)。
   - 它现在 *更加自主*，且 *不太可能卡在同一个 bug 上*。
- **GPT-4o 迎来全新更新**：最新的 **ChatGPT-4o** 更新（**2025-03-26**）在 Arena 排行榜上跃升至第 2 位，超越了 **GPT-4.5**，并有显著改进，在 Coding 和 Hard Prompts 类别中并列第 1，据 [Arena 排行榜](https://x.com/lmarena_ai/status/1905340075225043057) 显示。
   - 它现在 *更擅长遵循详细指令*，尤其是包含多个请求的提示词，并且 *提升了直觉和创造力*。
- **OpenAI 放宽图像生成政策**：**OpenAI** 的政策从一味拒绝转向防止现实世界的伤害，旨在最大化创作自由的同时防止真正的伤害，Joanne Jang 在[她的博客文章](https://x.com/joannejang/status/1905341734563053979)中对此进行了描述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openai.github.io/openai-agents-python/mcp/">Model context protocol (MCP) - OpenAI Agents SDK</a>: 未找到描述</li><li><a href="https://blog.replit.com/agent-v2">Replit — Introducing Replit Agent v2 in Early Access</a>: 更智能，具备实时应用设计预览功能。配合 Anthropic 的 Claude 3.7 Sonnet 发布，我们很高兴地宣布在早期访问计划中推出 Replit Agent v2。早期访问...</li><li><a href="https://x.com/levelsio/status/1905324525006299521?s=46">Tweet from @levelsio (@levelsio)</a>: 致所有正在开发吉卜力（Ghibli）生成器应用的数百人：如果你将其命名为 Ghibli 或类似名称，你会收到吉卜力工作室律师的信函。但更糟糕的是，如果你赚了数百万，你只会...</li><li><a href="https://www.deeplearning.ai/short-courses/vibe-coding-101-with-replit/">Vibe Coding 101 with Replit</a>: 在集成 Web 开发环境中，使用 AI coding agent 设计、构建和部署应用。</li><li><a href="https://fxtwitter.com/OpenAI/status/1905331956856050135)">Tweet from OpenAI (@OpenAI)</a>: ChatGPT 中的 GPT-4o 再次更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词 - 提升了处理复杂技术的能力...</li><li><a href="https://x.com/OpenAI/status/1905331956856050135>)">Tweet from OpenAI (@OpenAI)</a>: ChatGPT 中的 GPT-4o 再次更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词 - 提升了处理复杂技术的能力...</li><li><a href="https://x.com/LangChainAI/status/1905325891934454170">Tweet from LangChain (@LangChainAI)</a>: 观看视频 ➡️ https://www.youtube.com/watch?v=NKXRjZd74ic</li><li><a href="https://fxtwitter.com/joannejang/status/1905341734563053979)">Tweet from Joanne Jang (@joannejang)</a>: // 我在 OpenAI 领导模型行为团队，想分享一些在制定 4o 图像生成政策时的思考和细微差别。使用了大写字母 (!)，因为我将其作为博客文章发布：--Thi...</li><li><a href="https://x.com/joannejang/status/1905341734563053979>)">Tweet from Joanne Jang (@joannejang)</a>: // 我在 OpenAI 领导模型行为团队，想分享一些在制定 4o 图像生成政策时的思考和细微差别。使用了大写字母 (!)，因为我将其作为博客文章发布：--Thi...</li><li><a href="https://fxtwitter.com/steph_palazzolo/status/1904947599368499497)">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: Nvidia 以数亿美元的价格收购了推理服务提供商 Lepton AI。这是 Nvidia 最新的交易，将有助于加强其软件产品，并使其更容易...</li><li><a href="https://x.com/steph_palazzolo/status/1904947599368499497>)">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: Nvidia 以数亿美元的价格收购了推理服务提供商 Lepton AI。这是 Nvidia 最新的交易，将有助于加强其软件产品，并使其更容易...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: 新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：- 相比 1 月版本有显著提升（+30 分，从第 5 名升至第 2 名）- 在 Coding 和 Hard Prompts 类别中并列第 1。</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057>)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: 新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：- 相比 1 月版本有显著提升（+30 分，从第 5 名升至第 2 名）- 在 Coding 和 Hard Prompts 类别中并列第 1。</li><li><a href="https://fxtwitter.com/amasad/status/1905065003209892154)">Tweet from Amjad Masad (@amasad)</a>: @McGaber @karpathy @Replit 明天发布！</li><li><a href="https://x.com/amasad/status/1905065003209892154>)">Tweet from Amjad Masad (@amasad)</a>: @McGaber @karpathy @Replit 明天发布！
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1354850570040639558)** (4 条消息): 

> `FP8 QAT, Optimizer State with Fake Quant` 


- **FP8 QAT 即将到来**: 一名成员正考虑让他们的模型对 **fp8** 更加友好，并在“冷启动”训练模型的基础上进行纯 **QAT** 运行，并发现了 PyTorch/AO 仓库中的[这个 issue](https://github.com/pytorch/ao/issues/1632)。
   - 他们正在研究 FP8 **QAT**，但目前还没有足够的精力（bandwidth）去执行。
- **Fake Quant 不会改变 Optimizer State**: 一名成员确认，启用 **fake quant** 不会导致 **optimizer state** 发生任何变化。



**提到的链接**: <a href="https://github.com/pytorch/ao/issues/1632">FP8 QAT / FP8 block-wise quantization · Issue #1632 · pytorch/ao</a>：为 FP8 提供 QAT 将是一个伟大的补充，以及通用的 FP8-blockwise 量化。

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1354539204838822009)** (18 条消息🔥): 

> `Deprecated code deletion, Linter installation issues, Anthropic using TensorFlow, GRPO PRs, JoeI sora` 


- **弃用代码寻找“执行者”**: 一名成员请求协助处理 [PR #2533](https://github.com/pytorch/torchtune/pull/2533) 以删除**弃用代码**，理由是无法在工作笔记本电脑上安装 **linter**。
   - 该 PR 涉及 *完全弃用 train_on_input* 以及移除其他弃用组件。
- **Anthropic 倒戈并使用 TensorFlow**: 一名成员指出 **Anthropic** 据称正在使用 **TensorFlow**，引发了关于 PyTorch 可能在那里被禁用的推测。
   - 另一名成员以 *smh*（摇头）回应，暗示对此无话可说。
- **JoeI SORA 接管世界**: 一名成员在未知背景下发布了 **JoeI SORA**（推测是一个 AI 模型）的截图，以回应另一名成员询问某个模型背后的直觉。
   - 该成员简单地回答说*没有直觉，只有 JoeI*，并展示了截图。
- **GRPO PR 请求处理**: 一名成员强调了**两个 GRPO PR**（[#2422](https://github.com/pytorch/torchtune/pull/2422) 和 [#2425](https://github.com/pytorch/torchtune/pull/2425)），并指出 **#2425** 是一个 bug 修复，应该尽快合并。
   - 另一名成员立即回应，确认正在处理。



**提到的链接**: <a href="https://github.com/pytorch/torchtune/pull/2533">Full train_on_input deprecation, removing other deprecated components by RdoubleA · Pull Request #2533 · pytorch/torchtune</a>：上下文此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）变更日志此 PR 做了哪些更改？使用 mas...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1354555568853094430)** (12 条消息🔥): 

> `Vector Database Options, Hosting Vector DB Online, AI Agent Pricing, Cohere at QCon London` 


- **向量数据库选项探索**: 一名成员询问了所使用的向量数据库，提到了他们使用 **Chroma** 的经验，并寻求有关其他选项及其常见用法的建议。
   - 作为回应，另一名成员提供了 [Cohere 集成页面的链接](https://docs.cohere.com/v2/docs/integrations)，展示了支持的向量数据库，如 **Elasticsearch**、**MongoDB**、**Redis**、**Haystack**、**Open Search**、**Vespa**、**Chroma**、**Qdrant**、**Weaviate**、**Pinecone** 和 **Milvus**。
- **在线托管向量数据库**: 一名成员询问了在线托管向量数据库的问题，特别是它们是同步到存储桶并每次加载，还是有其他替代方法。
   - 另一名成员提供了 [Cohere 集成页面的链接](https://docs.cohere.com/v2/docs/integrations)，展示了支持的向量数据库，暗示它们会处理托管相关问题。
- **AI Agent 定价探索**: 一名成员正在研究创始人如何为 **AI Agent** 定价和变现，并邀请其他人聊天并验证见解。
   - 另一名成员回复鼓励*与我们分享更多*关于 **AI Agent** 定价的信息。
- **Cohere 可能会再次参加 QCon London？**: 一名成员询问 Cohere 在去年参加后，今年是否还会参加 **QCon London**。
   - 他们表示有兴趣与 Cohere 代表讨论获取 **North** 访问权限的事宜。



**提到的链接**: <a href="https://docs.cohere.com/v2/docs/integrations">Integrating Embedding Models with Other Tools — Cohere</a>：了解如何将 Cohere **embeddings** 与开源向量搜索引擎集成以增强应用。

  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1354766477349359827)** (2 条消息): 

> `难民组织、和平建设、生计机会` 


- **难民倡导者介绍组织**：一位肯尼亚的难民介绍自己是 **Pro-Right for Refugees** 的负责人，这是一个位于 Kakuma 难民营和 Kalobeyei 定居点的社区组织 (CBO)。
   - 该组织的愿景是促进难民获得**生计机会**并增强和平生活，其使命是让每位难民都有权在和平的环境中获得生计。
- **Pro-Right for Refugees 专注于和平与繁荣**：该 CBO **Pro-Right for Refugees** 专注于和平建设、意识提升和生计倡议。
   - 他们欢迎志愿者和任何有兴趣支持营地难民的人，其座右铭是 *“和平生活，繁荣生活。”*


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1354661363682639983)** (12 条消息🔥): 

> `预算级 AI 装备、AX650N NPU、Tinygrad PR` 


- **组装预算级 AI 装备**：一位成员询问是否可以使用淘宝上旧的 **X99** 组件、**Xeons** 和 **32GB ECC DDR4 RAM**，以 **7000-8000 元**的价格组装一台 AI 装备。
   - 另一位成员在检查后确认了其可行性。
- **AX650N 拥有 72 TOPS 算力**：一位成员分享了 [AX650N 产品页面](https://www.axera-tech.com/Product/125.html)的链接，强调其 **72Tops@int4, 18.0Tops@int8 NPU** 以及对 Transformer 智能处理平台的原生支持。
   - **AX650N** 配备 **八核 A55 CPU**，支持 **8K 视频编解码**，并提供 **双 HDMI 2.0** 输出。
- **AX650N 性能调研**：一位成员分享了一篇对 **AX650N** 进行逆向工程的[博客文章](http://jas-hacks.blogspot.com/2024/09/ax650n-sipeed-maix-iv-axerapi-pro-npu.html?m=1)链接，指出其提供 **72.0 TOPS@INT4** 和 **18.0 TOPS@INT8** 的算力。
   - 博客文章提到正在努力移植较小的 Transformer 模型以展示其能力，并提供了一个 [GitHub 仓库](https://github.com/AXERA-TECH/ax-llm)。
- **Tinygrad PR**：分享了两个 Tinygrad 的拉取请求，[PR #9546](https://github.com/tinygrad/tinygrad/pull/9546) 和 [PR #9554](https://github.com/tinygrad/tinygrad/pull/9554)。
   - 第一个 PR 是 *针对 test_failure_53 中递归错误的潜在修复*，第二个 PR 旨在 *继续将 torch 后端中的函数从 CPU 移出*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://jas-hacks.blogspot.com/2024/09/ax650n-sipeed-maix-iv-axerapi-pro-npu.html?m=1">Tiny Devices: AX650N - Sipeed Maix-IV (AXeraPi-Pro) NPU 拆解</a>: 未找到描述</li><li><a href="https://www.axera-tech.com/Product/125.h">AX650N-爱芯元智</a>: 未找到描述</li><li><a href="https://www.axera-tech.com/Product/125.html">AX650N-爱芯元智</a>: 未找到描述</li><li><a href="https://e.tb.cn/h.6dxnX3PdIl9NZ6K?tk=1GaYeFF6MIH">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9546">yvonmanzi 提交的 test_failure_53 递归错误潜在修复 · Pull Request #9546 · tinygrad/tinygrad</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9554">yvonmanzi 提交的 Torch randpermgen · Pull Request #9554 · tinygrad/tinygrad</a>: 继续将 torch 后端中的函数从 CPU 移出 ... #9463
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1354949525160333373)** (1 条消息): 

> `TinyGrad 代码生成、Codegen 转换器` 


- **澄清 TinyGrad 代码生成**：一位用户在遇到提到 `CStyleCodegen` 和 `CUDACodegen` 类的过时信息后，寻求关于 TinyGrad 代码生成过程的澄清。
   - 该用户旨在了解从优化计划到低级 C++/CUDA 代码的转换具体发生在何处。
- **TinyGrad 中的 Codegen 转换器**：讨论集中在理解 TinyGrad 如何将优化计划转换为适用于不同设备（CPU/GPU）的机器可执行代码。
   - 过时的信息暗示存在特定的转换器类，如 `CStyleCodegen` 和 `CUDACodegen`，促使该用户询问当前的实现方式。

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1354577628866154747)** (7 messages): 

> `分享课程录像、导师申请截止日期延长、Entre 赛道导师计划` 


- **课程录像分享获准**：一名成员询问关于分享课程录像的事宜，版主确认这*完全没有问题*，并鼓励新的 MOOC 参与者[报名](https://forms.gle/9u6HdVCWXgws16go9)。
- **导师申请截止日期延长考虑中**：一名成员请求延长导师申请的截止日期；版主回复称表单不会立即关闭，允许继续提交，但由于关注度极高且项目需要尽快启动，截止日期后的申请不保证会被考虑。
- **Entre 赛道缺失导师计划**：一名成员询问关于 Entre 赛道的导师机会，版主澄清 **Berkeley** 不提供该赛道的导师，但在 **4月/5月** 会有赞助商提供的 Office Hours。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1354840615510343852)** (3 messages): 

> `Atom of Thoughts (AOT), Tree of Thoughts (ToT), Markovian Reasoning, Two-phase Transition, Atomic Granularity & Dependencies` 


- ****AOT vs ToT****：发布者区分了 **Atom of Thoughts (AOT)** 与 **Tree of Thoughts (ToT)**，详细说明 AOT 的推理步骤是无记忆的（memoryless），而 ToT 维护整个树的历史记录；此外，AOT 具有明确的“分解后收缩”（decompose-then-contract）阶段，针对原子级、不可分割的子问题，而 ToT 探索分支想法但没有明确的收缩过程。
   - AOT 还强制将问题分解为结构为有向无环图（DAG）的原子级子问题，而 ToT 允许不同的粒度且不强制依赖关系。
- ****评估数据集适用性****：理想的评估数据集包括 [GSM8K 和 MATH](https://example.com/datasets)（具有逐步解法的数据集）、[HotpotQA 和 2WikiMultihopQA](https://example.com/datasets)（带有标注推理路径的数据集），以及明确详述中间推理步骤的数据集。
   - 发布者包含了如下示例：`mock_llm_client.generate.side_effect = ["0.9", "42"]`。
- ****使用 LLMDecomposer 的分解策略****：AOT 通过 `LLMDecomposer` 使用灵活的分解方式，Prompt 根据问题类型（MATH, MULTI_HOP）进行调整，支持自定义分解器和动态 Prompt 选择，并通过收缩验证阶段确保原子性。
   - 示例分解 Prompt 包括 `QuestionType.MATH: Break down this mathematical question into smaller, logically connected subquestions: Question: {question}`。
- ****AOT 与 DSPy 集成****：AOT 平滑集成到 DSPy 工作流中，通过契合 DSPy 的模块化设计增强推理能力，与 DSPy 优化器（如 MIPROv2）协同工作，补充 DSPy 的推理策略以实现高效扩展，并与 DSPy 的动态路由互补，智能调整推理路径。
   - 发布者确认他们已经有了*一个现成的可用实现*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1354787475259920404)** (1 messages): 

> `MiproV2 问题, DSPy 中的 ValueError` 


- **MiproV2 面临 ValueError**：一名成员在使用 **MiproV2** 时遇到了 **ValueError**，具体与 `signature.output_fields` 中键值不匹配有关。
   - 错误信息显示预期的键为 `dict_keys(['proposed_instruction'])`，但实际收到的键为 `dict_keys([])`。
- **调试 MiproV2 键值不匹配**：遇到 **MiproV2** **ValueError** 的用户正在寻求解决键值不匹配问题的帮助。
   - 据报道，GitHub 上的 **Copro** 也遇到了类似问题，可能与 `max_tokens` 设置有关，尽管用户怀疑这并非本案例的根本原因。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1354970106207142009)** (2 messages): 

> `Gemini 2.5 Pro, Windsurf 额度, 速率限制` 


- **Windsurf 随 Gemini 2.5 Pro 发布掀起热潮！**：**Gemini 2.5 Pro** 现已在 Windsurf 中可用，每条消息授予用户 **1.0** 个用户 Prompt 额度，每次工具调用授予 **1.0** 个 Flow Action 额度，已在 [X 上宣布](https://x.com/windsurf_ai/status/1905410812921217272)。
- **Windsurf 因 Gemini 2.5 Pro 的火爆而告急！**：Windsurf 已经遇到了 **Gemini 2.5** 的速率限制（rate limiting）问题，理由是模型和提供商面临巨大负载。
   - 团队正积极致力于增加配额，并对造成的任何不便表示歉意。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1905410812921217272">Windsurf (@windsurf_ai) 的推文</a>：Gemini 2.5 Pro 现已在 Windsurf 中可用！✨

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1354937897198813384)** (1 条消息): 

> `GPT4All 问题，模型导入问题，用户体验挫败感` 


- **GPT4All 用户报告模型导入问题**：用户报告在向 **GPT4All** 导入模型时遇到困难，系统似乎没有响应。
   - 其他问题包括无法搜索模型列表、选择时缺少模型大小信息、缺乏 LaTeX 支持以及模型列表排序不友好。
- **GPT4All 用户对用户体验表示不满**：用户对 **GPT4All** 的用户体验表示沮丧，提到的问题包括缺少 Embedder 选择选项。
   - 一位用户表示：*你们正在流失用户……因为其他产品更加用户友好且更愿意保持开放*。


  

---


---


---


{% else %}


> 完整的频道细分内容已在邮件中截断。 
> 
> 如果你想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}