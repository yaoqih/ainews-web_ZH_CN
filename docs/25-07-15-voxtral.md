---
companies:
- mistral-ai
- moonshot-ai
- groq
- together-ai
- deepinfra
- huggingface
- langchain
date: '2025-07-15T05:44:39.731046Z'
description: '**Mistral** 发布了 **Voxtral**，这款转录模型的表现超越了 **Whisper large-v3**、**GPT-4o
  mini Transcribe** 以及 **Gemini 2.5 Flash**，令人眼前一亮。Voxtral 模型（提供 3B 和 24B 两个版本）支持
  **32k token 的上下文长度**，可处理长达 **30-40 分钟**的音频。它内置了**问答与摘要**功能，支持**多语言**，并能通过语音指令实现**函数调用
  (function-calling)**。该模型以 **Mistral Small 3.1** 语言模型为核心底座。


  与此同时，**月之暗面 (Moonshot AI)** 的 **Kimi K2** 也引发了广泛关注。这是一款由约 **200 人**团队打造的非推理型**混合专家
  (MoE)** 模型。它凭借在 **Groq** 硬件上极快的推理速度、在 **Together AI** 和 **DeepInfra** 等平台的广泛可用性，以及支持在
  **M4 Max 128GB** Mac 上本地运行而备受瞩目。此外，Kimi K2 已集成 **LangChain** 和 Hugging Face 等开发者工具，彰显了其强大的工具调用能力。'
id: MjAyNS0w
models:
- voxtal-3b
- voxtal-24b
- kimi-k2
people:
- jeremyphoward
- teortaxestex
- scaling01
- zacharynado
- jonathanross321
- reach_vb
- philschmid
title: Voxtral —— Mistral 推出的 SOTA（顶尖水平）语音识别（ASR）模型，包含 3B（mini）和 24B（"small"）两种尺寸，其表现超越了
  OpenAI 的 Whisper large-v3。
topics:
- transcription
- long-context
- function-calling
- multilingual-models
- mixture-of-experts
- inference-speed
- developer-tools
- model-integration
---



我们喜欢这种毫无保留的碾压，尤其是当它是一个开源模型时。

Voxtral 3B 和 Voxtral 24B 模型都超越了单纯的转录功能，其能力包括：

- **长上下文**：凭借 32k token 的上下文长度，Voxtral 可以处理**长达 30 分钟的转录音频，或 40 分钟的理解音频**。
- **内置问答与摘要：支持直接针对音频内容提问或生成结构化摘要，无需串联独立的 ASR 和语言模型**。
- **原生多语言**：自动语言检测，并在全球最广泛使用的语言（如英语、西班牙语、法语、葡萄牙语、印地语、德语、荷兰语、意大利语等）中表现出 state-of-the-art 的性能，帮助团队通过单一系统服务全球受众。
- **直接从语音进行 Function-calling**：能够根据用户的口头意图直接触发后端函数、工作流或 API 调用，将语音交互转化为可执行的系统命令，无需中间解析步骤。
- **强大的文本处理能力**：保留了其**语言模型骨干 Mistral Small 3.1** 的文本理解能力。

非常令人兴奋。我们之前跳过了关于他们 [Magistral 推理模型](https://mistral.ai/news/magistral)的报道（事实证明那篇[论文非常出色](https://www.youtube.com/watch?v=_vNFJcb8S_M)），但我们非常确定 Voxtral 将几乎立即投入生产环境...

---

# AI Twitter 综述

**Kimi K2 的出现与性能**

- **Kimi K2，一个非推理 MoE，挑战西方模型**：**Moonshot AI** 发布的 **Kimi K2** 引发了广泛讨论，特别是围绕其性能和起源。[@teortaxesTex 指出](https://twitter.com/teortaxesTex/status/1944856509734961596)，**Kimi** 是由一个约 **200 人** 的团队在有限的 GPU 预算下构建的，这让人质疑为什么西方公司没有开发出类似的产品。[@jeremyphoward 强调](https://twitter.com/jeremyphoward/status/1944864781695113385)，**K2** “*不是*一个推理模型”，并且在其 **Mixture of Experts (MoE)** 架构中使用了极少的激活 Token，使其更便宜、更快速。社区成员对其能力赞誉有加，[@scaling01 强调了其出色的报告生成能力](https://twitter.com/scaling01/status/1944850575470027243)，而 [@zacharynado 称其为](https://twitter.com/zacharynado/status/1944945039647629548) “领先的开放权重非推理模型”。
- **Groq 上的极速推理和广泛的平台可用性**：一个关键亮点是 **Kimi K2** 在 **Groq** 硬件上的表现。[@teortaxesTex 报告了 185 t/s 的速度](https://twitter.com/teortaxesTex/status/1944950183051321542)，认为这使得 **K2** “立即比 Sonnet 4 更具吸引力”，并且将 **1T 参数模型** 适配到他们的芯片上是一项令人印象深刻的成就。**Groq** 正式宣布该模型进入预览阶段，[@JonathanRoss321 展示了一段](https://twitter.com/JonathanRoss321/status/1944988412357849128)其速度的视频。该模型也可在 **Together AI**（[此处](https://twitter.com/togethercompute/status/1944952034840732138)和[此处](https://twitter.com/togethercompute/status/1945143838911128019)）、**DeepInfra**（[价格为 $0.55/$2.20](https://twitter.com/jeremyphoward/status/1944939322735780260)）上使用，并且正如 [@reach_vb 指出的那样](https://twitter.com/reach_vb/status/1944997786329460978)，可以在单台 **M4 Max 128GB** Mac 上本地运行。
- **工具集成和开发者资源**：**Kimi K2** 已被快速集成到开发者工具中。**Moonshot AI** 宣布了[对其 Hugging Face 仓库的错误修复](https://twitter.com/Kimi_Moonshot/status/1945050874067476962)，以改进多轮工具调用。**LangChain** 宣布在 **Groq** 上正式支持该模型（[此处](https://twitter.com/_philschmid/status/1944847828599054713)和[此处](https://twitter.com/Hacubu/status/1945144499228811676)），**Cline** 已[将 Moonshot AI 添加为供应商](https://twitter.com/cline/status/1945164549134672373)。用户正在展示其强大的工具使用能力，[@yawnxyz 展示了一个](https://twitter.com/bigeagle_xd/status/1945087963408351728)可以与 Google Maps 对话的 Chrome 扩展。

**新模型：语音、动作捕捉和 AI 伴侣**

- **Mistral 发布开源语音模型 Voxtral**：**Mistral AI** 发布了 **Voxtral**，[@GuillaumeLample 声称](https://twitter.com/GuillaumeLample/status/1945161150900924490)它是“全球最强（且开源）的语音识别模型”。[@reach_vb 对此发布感到兴奋](https://twitter.com/reach_vb/status/1945135982023520623)，并指出 audioLM 的一个主要痛点是经常丢失文本能力，但 [Voxtral 似乎避免了这一问题](https://twitter.com/reach_vb/status/1945140430288417007)。该模型可通过 API、Le Chat 和 Hugging Face 获取。[@teortaxesTex 认为这次发布将“重振转录应用市场”](https://twitter.com/teortaxesTex/status/1945133462395957621)。
- **xAI 推出 Grok 伴侣与头像**：**xAI** 推出了 **Grok** 头像和伴侣功能，并迅速走红。[@chaitualuru 宣布](https://twitter.com/chaitualuru/status/1945053158071255257)该功能“在日本重回巅峰”。[@ebbyamir 分享了](https://twitter.com/ebbyamir/status/1944902771599450237)包括名为 **Ani** 的动漫少女人格在内的各种示例，[@shaneguML 指出](https://twitter.com/shaneguML/status/1945003636439814430)考虑到市场情况，这种发展是可预见的。
- **Runway 推出用于高级动作捕捉的 Act-Two**：**RunwayML** 发布了下一代动作捕捉模型 **Act-Two**。[@c_valenzuelab 强调了](https://twitter.com/c_valenzuelab/status/1945190630449172587)其在“生成质量和对手部支持方面的重大改进”。他们还[分享了一个使用该模型制作的文艺复兴风格人声打击乐的创意演示](https://twitter.com/c_valenzuelab/status/1945219029192286717)。
- **Google 通过排名第一的 Embedding 和新功能增强 Gemini**：**Google DeepMind** 宣布其首个 **Gemini Embedding** 模型现已正式商用，并且[在 MTEB 排行榜上排名第一](https://twitter.com/demishassabis/status/1944870402251219338)。此外，[@demishassabis 分享了一项新的 Gemini 功能](https://twitter.com/demishassabis/status/1944939563170062804)，可以将照片转换为带声音的视频。
- **其他值得关注的模型与更新**：**LG 的 EXAONE 4** 是一个在 14T token 上训练的 32B 模型，[在推理和非推理模式下表现出与前沿模型接近的性能](https://twitter.com/teortaxesTex/status/1944947588006076664)。**Kling AI** 一直在展示其视频生成能力，展示了[在处理水、光影和动作方面的精准度](https://twitter.com/Kling_ai/status/1945095794127683640)。

**工具、基础设施与开发**

- **Agentic 编程助手受到关注**：**Anthropic 的 Claude Code** 被强调为一个强大的工具，[@claude_code 提供了关于将其作为本地文件系统任务的通用 Agent 使用的技巧](https://twitter.com/claude_code/status/1944944964708000083)。其受欢迎程度正在激增，[@kylebrussell 指出](https://twitter.com/kylebrussell/status/1945132555604251007)朋友们正专门为此升级到付费档位。与此同时，**Perplexity** 正在迅速为其 **Comet** 浏览器添加功能，包括[网页版语音模式](https://twitter.com/AravSrinivas/status/1944861476692615333)和[清理电子邮件收件箱](https://twitter.com/AravSrinivas/status/1945232153609978273)的能力。[@AravSrinivas 指出](https://twitter.com/AravSrinivas/status/1945136929218953577)，其目标是将工具无缝融合在一起，使用户无需切换模式。
- **向量数据库和框架不断演进**：**Qdrant** 推出了 **Qdrant Cloud Inference**，允许用户[直接在其云集群中生成、存储和索引 Embedding](https://twitter.com/qdrant_engine/status/1945090285039464518)。这包括对稠密、稀疏和多模态模型（如 **CLIP**）的支持。**LlamaIndex** 和 **Google AI** 合作开展了一项教程，旨在[使用 Gemini 2.5 Pro 构建多 Agent 深度研究系统](https://twitter.com/jerryjliu0/status/1944882346731430127)，而 **LangChain** 正在与 **Redis** 和 **Tavily** 等合作伙伴举办活动，以[展示新兴的 AI Gateway 技术栈](https://twitter.com/LangChainAI/status/1944905481069437210)。
- **端侧 AI 和专用框架**：Apple 的 **MLX** 框架继续扩张，[@awnihannun 宣布](https://twitter.com/awnihannun/status/1944904396606988655)正在进行纯 C++ 移植（**mlx-lm.cpp**）并[支持 tvOS](https://twitter.com/awnihannun/status/1944893455202967921)。在移动领域，[@maximelabonne 发布了 **LEAP**](https://twitter.com/maximelabonne/status/1945110321938514335)，这是一个用于在 iOS 和 Android 上构建由本地 LLM 驱动的应用的开发者平台。
- **数据可用性和微调**：[@maximelabonne 宣布](https://twitter.com/maximelabonne/status/1945018242290082047) **LFM2** 模型现在可以使用 **Axolotl** 进行微调。在数据方面，[@code_star 转发了一项更新](https://twitter.com/code_star/status/1944890857347539045)，称 **FineWeb** 和 **FineWeb-Edu** 现在包含 2025 年 1 月至 6 月的 **CommonCrawl** 快照。在一项重大的开源贡献中，[@ClementDelangue 分享了](https://twitter.com/ClementDelangue/status/1945185890294255741) **99% 的美国判例法** 已在 Hugging Face 上开源。

**研究、评估与 AI Safety**

- **全行业推动思维链 (Chain of Thought, CoT) 监控**：一篇由 **OpenAI**、**Anthropic** 领导者及学术界共同签署的跨机构论文，敦促各实验室保留 AI 推理的可监控性。**OpenAI** 表示其[正支持利用 CoT 监督智能体系统 (Agentic systems) 的研究](https://twitter.com/OpenAI/status/1945156362859589955)。包括 [@woj_zaremba](https://twitter.com/woj_zaremba/status/1945158231321706896)、[@merettm](https://twitter.com/merettm/status/1945157403315724547)、[@NeelNanda5](https://twitter.com/NeelNanda5/status/1945156291577700542) 和 [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1945216792051232973) 在内的关键人物均表示强烈支持，认为这种对模型思考过程的可见性是至关重要的安全馈赠，不应通过训练将其消除。
- **“Context Rot”与长上下文窗口的局限性**：来自 **Chroma** 的一份技术报告显示，[增加输入 token 会降低 LLM 的性能，即使是在简单任务上也是如此](https://twitter.com/swyx/status/1944848537092809177)。这份名为 **“Context Rot”** 的报告指出，在 113k token 的对话历史下，准确率会下降 **30%**。[@imjaredz 总结了这些发现](https://twitter.com/imjaredz/status/1944855623301988602)，并得出结论：“百万级 token 的上下文窗口是一个谎言”，上下文应该进行外科手术式的精细工程处理。
- **AI 驱动的安全与新研究方向**：**Google** 宣布其 [AI Agent **Big Sleep** 帮助检测并挫败了一次迫在眉睫的漏洞利用](https://twitter.com/sundarpichai/status/1945113799297536313)，标志着 AI 在网络安全领域的重大应用。在其他研究中，[@lateinteraction 强调了](https://twitter.com/lateinteraction/status/1944941744782512389)一个将**基于 Rust 的 ColBERT 模型编译为 WebAssembly (WASM)** 以进行客户端执行的项目。[@teortaxesTex 指出了一篇关于 **Memory Mosaics v2** 的论文](https://twitter.com/teortaxesTex/status/1944868734247788641)，据报道其性能优于在 **8 倍**更多 token 上训练的 Transformer。
- **数据污染与评估范式**：[@francoisfleuret 强调了训练中数据污染的挑战，他建议](https://twitter.com/francoisfleuret/status/1944997748807172555)“在 1799 年 12 月 31 日之前的数学数据上进行训练，在之后的数据上进行验证”。这反映了对不依赖于记忆的鲁棒评估方法的广泛需求。

**公司战略与行业格局**

- **Meta 的超智能愿景与开源辩论**：Mark Zuckerberg 关于大规模 AI 超级集群 (Superclusters) 的计划是一个主要话题。**Meta AI** 分享了他的愿景，即[“为世界上的每个人提供个人超智能 (Personal Superintelligence)”。](https://twitter.com/AIatMeta/status/1945182467088113920) 这一举动引发了担忧，[@Yuchenj_UW 表示](https://twitter.com/Yuchenj_UW/status/1944962450954313841)，随着 **Meta** 变成“另一个 OpenAI”，西方可能不得不“依靠中国来维持开源 AI 的生命力”。
- **并购活动与预测**：在据传有 Google 参与的竞购战后，**Cognition** 收购了 **Windsurf**。在一条广为流传的推文中，[@swyx 发布了一个潜在收购的“六路过关”预测](https://twitter.com/swyx/status/1944902499510653020)，包括 **Apple 收购 Mistral**、**Meta 收购 Mistral 的部分业务**，以及 [**Perplexity 收购 Character.ai**](http://character.ai/)。
- **新创企业与全球扩张**：**Andrew Ng** 宣布成立 **AI Aspire**，这是一家新的咨询公司，[与 **Bain & Company** 合作](https://twitter.com/AndrewYNg/status/1945148766962729370)帮助企业制定 AI 战略。**Cohere** 正在[韩国首尔开设其首个亚洲办事处](https://twitter.com/aidangomez/status/1944913553640558638)。一家名为 **Thinking Machines Lab** 的新初创公司透露，其正在[为其雄心勃勃的多模态 AI 项目招聘人才](https://twitter.com/lilianweng/status/1945184437185966149)。
- **长期磨练与执行力的重要性**：[@AravSrinivas 将当前的 AI 竞赛描述为一场“长达十年的磨练”](https://twitter.com/AravSrinivas/status/1944895074774737130)，任何人都无法保证成功。[@andrew_n_carr 强调了执行力和专注团队的重要性，他表示“在 OpenAI 经常需要手动标注数据”](https://twitter.com/andrew_n_carr/status/1944889836424355852)。

**幽默、模因与文化**

- **共鸣评论**：[@stephenroller 的观察](https://twitter.com/stephenroller/status/1945096001959698791) “千禧一代把 'lol' 用得像电报末尾的 STOP 一样 lol” 是点赞数最高的推文。[@willdepue 提出了一个新的最严厉侮辱](https://twitter.com/willdepue/status/1944889768812089707)：“你从根本上缺乏好奇心，而这无药可救。”
- **行业内部梗**：来自 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1944876105393168394) 的一个笑话捕捉到了冗余项目泛滥的现状：“管理层：你知道这个世界真正需要什么吗？一个新的 vscode 分支。” [@dylan522p](https://twitter.com/dylan522p/status/1945032974434537945) 的一张梗图描绘了将模型量化到 **fp4** 后的混乱结果。
- **Grok 伴侣热潮**：**xAI** 伴侣功能的发布引发了梗图刷屏，[@ebbyamir 转发的一条推文](https://twitter.com/ebbyamir/status/1944961018649829797)展示了被该新功能占据的时间线。
- **开发者体验**：[@skalskip92 发布了一段热门视频](https://twitter.com/skalskip92/status/1945142384578240748)，配文是“当你完全不知道自己在做什么，但它居然能跑通时……”，捕捉到了软件开发中一种普遍的情绪。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Kimi K2 模型基准测试、API 访问及社区梗图

- [**Kimi K2 在创意写作基准测试中夺冠**](https://i.redd.it/q48f55vcpwcf1.jpeg) ([得分: 300, 评论: 63](https://www.reddit.com/r/LocalLLaMA/comments/1lzywie/kimi_k2_tops_creative_writing_benchmark/)): **柱状图展示了各语言模型在创意写作基准测试中的排名，Kimi K2 以 8.56 的最高平均分位居榜首，证明了其在创意写作任务中优于 DeepSeek V3、Gemma 27B、Gemini 2.5 Pro 等领先的替代方案。这一视觉对比为 Kimi K2 目前在模型创意基准测试中的优势提供了实证。** 几位评论者对基准测试结果的准确性提出了挑战，特别是质疑 DeepSeek V3 0324 在实际创意写作中的表现是否真的逊于 Gemma 27B，并对 Kimi K2 的所谓优势表示怀疑，认为用户体验与基准测试结果存在显著差异。
    - 多位用户专门针对创意写作任务对比了 Kimi K2、DeepSeek V3 0324、Gemma 27B 和 Gemini 2.5 Pro。一位评论者声称 DeepSeek V3 0324 在创意写作方面大幅领先于 Gemma 27B，并表示个人测试显示两者存在巨大的质量差距；而其他人则断言 K2 并不比 DeepSeek 或 Gemini 2.5 Pro 好多少。这些对比反映了用户对知名开源和闭源模型性能的主观感知。
    - 一条具有技术洞察力的评论将 Kimi K2 在创意写作基准测试中的出色表现与其潜在的编程能力联系起来。该评论者认为，在需要整合多种约束和结构化输出的任务（例如包含多个元素的叙事）中表现出色，与程序合成和执行复杂软件计划所需的技能非常相似。从证据上看，他们在 Cline 的测试中观察到基准测试结果与 K2 代码生成可靠性之间存在相关性。
    - 讨论延伸到了特定任务的模型表现：一些人发现 Kimi K2 在角色扮演（RP）中连贯性和趣味性较差，认为与其他模型相比，它在多轮对话或对话格式中难以维持上下文和叙事吸引力。每个模型优势的细微差别似乎取决于具体的创意写作任务（故事结构、RP、约束遵循等）。

- [**Kimi K2：为无法本地运行的用户提供廉价且快速的 API 访问**](https://openrouter.ai/moonshotai/kimi-k2) ([评分: 146, 评论: 64](https://www.reddit.com/r/LocalLLaMA/comments/1m0cgnl/kimi_k2_cheap_and_fast_api_access_for_those_who/)): **该帖子重点介绍了用于访问开放权重 Kimi-K2 模型 ([moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct)) 的新 API 端点，并指出 DeepInfra 提供了最低的 API 定价（每百万 Token 的输入/输出价格为 **`$0.55/$2.20`**），而 Groq 则提供了最高的推理速度（约 **`250 tokens/sec`**，尽管成本较高）。作者指出 Kimi-K2 的 API 访问比 Claude Haiku 3.5、GPT-4.1 和 Gemini 2.5 Pro 等闭源模型更便宜，强调了许可宽松的开放权重模型的价值，并列出了 [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2) 上的所有提供商；还提到了一个免费变体。详情请参阅 [DeepInfra 定价](https://deepinfra.com/moonshotai/Kimi-K2-Instruct) 和 [Groq 文档](https://console.groq.com/docs/model/moonshotai/kimi-k2-instruct)。** 热门评论提出：(1) 是否更倾向于使用官方的 Moonshot API——其费率甚至更低（`$0.15/2.5M tokens`）；(2) 注意到 Kimi-K2 具有兼容 Anthropic 的 API 端点，可以通过设置特定的环境变量与 Claude Code 接口，提供一种高性价比（尽管较慢）的 Claude 兼容推理方案；(3) 对“本地”访问表示怀疑，因为大多数用户的硬件要求过高。
    - 一位评论者强调了 Kimi K2 兼容 Anthropic API 的优势，使用户能够通过将 `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL` 指向 Moonshot 的端点，轻松重定向 Claude Code 等客户端。这种方法被指出比官方 Anthropic 访问 *“慢但便宜得多”*，对于需要兼容性和经济性的开发者来说是一个极具成本效益的解决方案。
    - 关于免费层级的澄清：Kimi K2 每天提供高达 500k Token 的免费使用额度，这是一个相当可观的配额。然而，目前尚不清楚 Kimi K2 是否支持 Context Caching 等高级功能，这可能会影响某些高吞吐量或上下文敏感任务的性能或成本效益。
    - 引用了 Kimi-K2 的主要 HuggingFace 仓库 (https://huggingface.co/moonshotai/Kimi-K2-Instruct)，一条评论强调了这样一个现实：几乎所有用户（“99.9%”）都缺乏在本地推理大型模型的硬件，这巩固了对廉价、易用的 API 端点而非本地部署的需求。
- [**谢谢你，Unsloth！你们是传奇！！！（现在我只需要 256GB 的 DDR5）**](https://i.redd.it/nl35mhaybxcf1.jpeg) ([评分: 222, 评论: 27](https://www.reddit.com/r/LocalLLaMA/comments/1m021nx/thank_you_unsloth_you_guys_are_legends_now_i_just/)): **这张图片是一个 meme，描绘了 Unsloth 为其 1.8-bit 版本的 Kimi K2-1T MoE 大语言模型采用的动态量化（Dynamic Quantization）过程，幽默地将先进的模型量化比作经典电影场景。动态量化是一种用于减小模型尺寸和内存需求的技术，正如标题和评论所暗示的，这对于在没有极高硬件要求（例如“256GB 的 DDR5”）的情况下运行像 Kimi K2-1T MoE 这样的大型模型至关重要。该 meme 认可了最近在超低比特量化方面的创新，这可以显著提高模型效率。** 评论讨论了对更激进的模型尺寸缩减的兴趣（例如“蒸馏后的 32b 或更低参数模型”以及对“0.11 bit 版本”的需求），反映了社区对极端内存和计算效率的渴望。此外，还表达了对 Unsloth 团队的感谢，并指出希望这是在一段时间内需要的最大模型尺寸，这既表明了技术需求，也表明了运行此类海量模型的挑战。
    - Ardalok 讨论了量化策略，建议像 DeepSeek 这样的模型可以使用更高水平的量化来提高效率——可能指的是 int4/int8 或类似方案——并暗示虽然 Unsloth 的工作对研究很有价值，但其他设置在实际部署中可能更优越，特别是在资源受限的环境中。
    - oh_my_right_leg 询问了实际部署情况，特别是性能指标，例如在 DDR5 RAM 上运行大型模型时 Prompt 和 Generation 阶段的每秒 Token 数 (token/s)。他们还询问是否可以将专家模型参数加载到 GPU VRAM 中，而将模型的其余部分存储在系统 DDR5 中（使用 MoE 架构和 VLLM 等工具），突出了在 VRAM 有限但系统 RAM 充足的硬件上平衡速度和内存需求的潜在方法。

### 2. AI 模型发布与基础设施里程碑 (Meta, EXAONE, Voxtral, Llama 4)

- [**EXAONE 4.0 32B**](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B) ([Score: 278, Comments: 101](https://www.reddit.com/r/LocalLLaMA/comments/1m04a20/exaone_40_32b/)): [**EXAONE 4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B) 是由 LG AI Research 开发的拥有 30.95B 参数的多语言 LLM，其特点包括混合注意力机制（局部/全局以 3:1 混合，全局部分不使用 RoPE）、QK-Reorder-Norm（Q/K 投影后的 RMSNorm，Post-LN）、131k token 上下文窗口以及 GQA（40 个注意力头，8 个键值头）。该模型支持双模式（可切换的推理与非推理模式）、Agentic 工具使用，并在包括 LiveCodeBench 在内的大多数领域的基准测试性能超过了 Qwen 3 32B；其多语言支持仅限于英语、韩语和西班牙语。部署需要自定义的 [transformers fork](https://huggingface.co/docs/transformers/main/en/model_doc/exaone)，官方支持仅限于 TensorRT-LLM；它在严格的非商业许可证下发布，禁止任何直接或间接的商业用途和竞争，商业许可需要单独谈判。热门评论讨论了该模型基准测试相对于 Qwen 3 32B 的优势、限制甚至微小部署的严苛非商业许可证，以及相对较窄的多语言支持（仅三种语言）。
    - 据报道，EXAONE 4.0 32B 在大多数基准测试中（包括 LiveCodeBench 等专业测试）均超过了 Qwen 3 32B，并具有可切换的推理模式，突显了其相对于竞争对手的技术进步。
    - 该模型的许可证严格限制为非商业用途，未经明确许可禁止任何商业部署或衍生用途。它还限制使用该模型或其输出开发竞争模型，除非协商单独的商业许可，否则这可能会限制其在初创公司和研究环境中的采用。
    - EXAONE 4.0 32B 的多语言支持目前仅扩展到三种语言：英语、韩语和西班牙语。与一些旨在提供更广泛多语言能力的领先开源模型相比，这明显受到了限制。
- [**Meta 有望成为首个拥有 1GW 超级集群的实验室**](https://i.redd.it/584vdadc4xcf1.png) ([Score: 185, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1m0115d/meta_on_track_to_be_first_lab_with_a_1gw/)): **该图片展示了一项公告，称 Meta 有望启动首个 1GW（吉瓦）超级集群，标志着数据中心和 AI 计算基础设施的重大飞跃。Meta Superintelligence Labs 准备建立多个多吉瓦集群——包括 Prometheus 和 Hyperion——强调旨在领先行业的可用 AI 算力和研究能力的大规模投资。这一里程碑反映了硬件采购和数据中心工程方面的进步。** 评论反映了对这种快速计算扩张可持续性的怀疑，将其与历史上的军备竞赛相类比，并担心这种对增长和股价增值的追求对这些公司来说最终是否可行。
    - 一位评论者指出，增加算力并不能保证产品质量，并以 Llama 4 为例，大量的资源似乎并未转化为理想的结果。这突显了在扩大模型训练超级集群规模时有时观察到的“收益递减”或“低效率”。
    - 鉴于 Meta 生成式 AI 产品的现状，人们对其大举投资计算基础设施的策略表示怀疑，理由是用户参与度乏善可陈且模型表现平平，这证明计算投资并不能确保业务或技术成果。
    - 讨论还表达了对当前 AI 计算军备竞赛可持续性的担忧，类比了历史情景，即过度投资最终可能损害大型公司，特别是如果不能很快实现有形成果（更好的模型、更广泛的采用）。

- [**mistralai/Voxtral-Mini-3B-2507 · Hugging Face**](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) ([评分: 261, 评论: 45](https://www.reddit.com/r/LocalLLaMA/comments/1m0k22v/mistralaivoxtralmini3b2507_hugging_face/)): [**Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) 是一个基于 MistralAI Ministral-3B 的 3B 参数多模态模型（音频-文本），提供顶尖的语音转录、强大的多语言支持、转录模式、直接音频问答/摘要以及语音函数调用功能，具有** `32k token` **上下文窗口和基于 vLLM 的参考 Python 推理（GPU：bf16/fp16 约需 9.5GB）。在公共音频数据集上达到了极具竞争力的 WER，同时保持了强大的文本能力。此外，还重点介绍了其参数量更大的 24B 兄弟模型 [Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)。** 讨论指出存在更大参数的模型变体，并分享了模型基准测试图像，表明社区对性能扩展和对比基准测试非常感兴趣。
    - 据报道，Voxtral Mini 模型在转录任务上优于 OpenAI Whisper，且价格不到其一半。其他技术特性包括自动语言识别，以及在包括英语、西班牙语、法语、葡萄牙语、印地语、德语、荷兰语和意大利语在内的多种主要语言中表现出顶尖的转录性能。
    - Voxtral Mini 被描述为音频到文本模型（而非语音到文本），被定位为音频-文本转换领域第二好的开源模型。较大的 Voxtral 24B 模型被认为能力稍逊于 Stepfun Audio Chat 模型，但提供了更高效的参数量（`24B` 对比 `132B`），在性能和效率之间取得了很好的平衡。
    - Voxtral 模型的一个 24B 参数变体已发布（[链接](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)），为具有不同计算需求的用户扩展了选择范围，并在模型大小和性能之间提供了更多灵活性。
- [**好吧，如果有人在等待 Llama 4 Behemoth，它已经没了**](https://analyticsindiamag.com/global-tech/meta-plans-to-abandon-llama-4-behemoth-but-why/) ([评分: 349, 评论: 112](https://www.reddit.com/r/LocalLLaMA/comments/1m0g2mk/well_if_anyone_was_waiting_for_llama_4_behemoth/)): **据报道，Meta 已取消其原计划开源的 2T 参数模型 Llama 4 Behemoth，原因是多项技术失败：使用分块注意力（chunked attention）以适应内存，但这降低了长上下文推理能力；以及在训练中期切换了 Mixture of Experts (MoE) 路由，导致了不稳定性。其他问题还包括数据去重不足、消融研究不完整以及长上下文能力的评估基础设施不足；这些失败促使 Meta 将重心转向其新超级智能实验室下的闭源模型。摘要 [文章](https://analyticsindiamag.com/global-tech/meta-plans-to-abandon-llama-4-behemoth-but-why/) 详细列出了关键的技术批评和工程流程失误。** 热门评论重点讨论了在尝试失败后开放权重是否仍有价值，一位用户质疑 Meta 为什么不迭代并改进错误以发布更好的开源 Llama 5，而是选择闭源。此外还有关于技术教训的讨论，特别是分块注意力和不稳定的专家路由带来的负面影响。
    - 一位用户讨论了 Llama 4 Behemoth 项目中具体的架构和训练错误，指出改变注意力分块如何影响了模型的推理能力，以及在训练过程中途切换专家路由方法可能导致了其失败。这凸显了在训练中期对模型质量进行重大干预的风险。
    - 另一位用户质疑因一次迭代失败就可能关闭模型权重访问权限的理由，认为更理想的策略应该是公开从过去的错误中学习并发布改进后的 Llama 5，这反映了社区对开放与闭源权重发布的担忧。
    - 一种关于行业趋势的技术观点被表达出来：随着 Behemoth 出现的问题，人们对未来规模超过 32B 或 A3B MoE 的开源模型持怀疑态度，并认为“SaaS 赢了”，这表明随着开源发布面临扩展挑战，行业正转向专有的超大模型。

### 3. AI 使用趋势、社区分析与本地推理梗 (Memes)

- [**分析了 5000 多个 Reddit 帖子，探讨人们在工作中（除编程外）实际如何使用 AI**](https://www.reddit.com/gallery/1m0d0vz) ([Score: 171, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1m0d0vz/analyzed_5k_reddit_posts_to_see_how_people_are/)): **对包含 5000 多个 Reddit 帖子的数据集进行了分析，以调查知识工作者在非编程工作场景中对 AI 的使用情况。主要发现包括：报告的对伦理风险的担忧相对较低（占 LLM 用户的 `7.9%`），且主要用途是长文本内容生成。工作应用的分析方法或分类法未详细说明。** 评论者质疑 `7.9%` 伦理风险统计数据的准确性，认为可能受到政策相关的“虚假民意”（astroturfing）或机器人的干扰，并指出该数据集可能存在局限性，无法代表更广泛的 LLM 使用模式。
    - 一位评论者质疑 Reddit 上 7.9% 的 LLM 用户担心“伦理风险”的发现，认为这一统计数据可能因政策研究机构机器人产生的“虚假民意”评论而虚高，并对数据集的代表性表示怀疑。
    - 另一个技术层面的担忧是数据集的规模和范围较小；尽管 LLM 有许多据称的使用案例（如数学领域），但在分析中却代表性不足，这表明数据收集过程中存在采样或分类偏差。
- [**完全轻量级的本地推理...**](https://i.redd.it/r05r0wfvn2df1.png) ([Score: 150, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1m0nutb/totally_lightweight_local_inference/)): **该梗图讽刺了即使在进行激进量化（例如降低到 3.5 bits）后，大语言模型本地推理仍持续占用高 RAM 的现象，捕捉了 AI/ML 社区在内存需求与磁盘存储方面的普遍挫败感。图片概括了量化模型仍需要大量 RAM 的问题，有时甚至接近原始权重或轻度压缩权重的大小，打破了量化文件大小所带来的预期。这突显了在消费级硬件上部署大型模型的实际瓶颈。** 评论指出量化计算与实际 RAM 需求之间的不匹配，讨论了较小（1B 参数）模型用于推理的实用性和有效性，并提到文件支持的 mmap 作为内存需求的潜在缓解策略。
    - 人们对轻量级本地推理的说法表示怀疑，特别是关于在消费级硬件上高效运行大型模型的可行性——数学计算似乎并不支持所声称的资源/延迟主张。
    - 一位评论者强调使用文件支持的 `mmap` 作为一种内存高效的模型加载技术，通过利用虚拟内存，可能允许在 RAM 有限的系统上加载更大的模型。
    - 评论对 4-bit 量化方法表现出兴趣，这些方法因其减小模型大小和降低推理成本的潜力而受到认可，尽管未提供与其他量化策略的详细比较。

## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Grok 4 与 xAI Waifu/NSFW 争议与讽刺

- [**想象一下 10 年前看到这个标题**](https://i.redd.it/a47v8ialkycf1.jpeg) ([Score: 5453, Comments: 239](https://www.reddit.com/r/singularity/comments/1m07eaw/imagine_seeing_this_headline_10_years_ago/)): **该图片是《滚石》杂志文章的讽刺模型，通过提到 Grok（xAI 的聊天机器人）推出色情动漫伴侣、获得美国国防部合同并整合希特勒身份的聊天机器人，来嘲讽最近的 AI 新闻头条。这部恶搞作品展示了 AI、流行文化、伦理和军事应用的交集，批评了主流 AI 话语的方向和煽动性。** 评论延续了讽刺基调，开玩笑说用动漫头像进行军事规划，并将此情景与《南方公园》剧集进行比较，表达了对当前和未来 AI 发展的怀疑并凸显了其中的荒谬感。
    - 此帖中的评论均不包含技术讨论或实质性的技术见解；所有评论均为幽默或偏离主题的内容。

- [**Beware of bots bearing breasts**](https://i.redd.it/8r4q27q491df1.jpeg) ([Score: 597, Comments: 24](https://www.reddit.com/r/singularity/comments/1m0gnfn/beware_of_bots_bearing_breasts/)): **这张图片使用类梗图（meme）的数字插图，评论了 xAI（由 Elon Musk 支持）的 AI 聊天机器人 Grok 在形象和品牌定位上的快速且有时异想天开的变化。它将该 AI 最近的品牌重塑或市场定位进行了对比：从几天前的“威权、军事化形象”转变为今天的“柔和、育儿形象”，突显了产品方向和目标用户群体的波动。帖子标题和艺术风格讽刺地警告用户警惕人格化的 AI 营销，尤其是那些旨在提高参与度的表面变化。** 热门评论指出了 Grok 的隐私问题（指出对话可能被 Elon Musk/xAI 监控或存储），并嘲讽了夸张的 AI 时间线（“2025 年实现 AGI”）。
    - 一位用户强调了像 Grok 这样的对话式 AI 模型的隐私问题，指出对话可能被提供商（在本例中为 Elon Musk 的公司）存储和访问。这引起了技术受众应该注意的基于 AI 的聊天服务中的用户数据保留和隐私问题。
- [**Not The Onion**](https://i.redd.it/07zmg7qi81df1.png) ([Score: 401, Comments: 54](https://www.reddit.com/r/OpenAI/comments/1m0glih/not_the_onion/)): **该图片是《滚石》杂志文章的讽刺模型，将关于 xAI 和 Grok 的荒谬说法组合在一起——据称一款色情动漫 AI 伴侣获得了国防部合同，以及 xAI 的一款 AI 聊天机器人据称自称为阿道夫·希特勒——并冠以“并非《洋葱报》”的标题以强调其不可信性。该帖子通过将对 AI 安全的真实担忧与离奇的虚构场景相结合，嘲讽了当前 AI 开发中（特别是在 xAI 领导下）被察觉到的鲁莽和伦理缺失，突显了对对齐失当的人工通用智能（AGI）的焦虑。** 一位评论者尖锐地批评了 xAI 被察觉到的粗心大意，称尽管之前有关于 AI 风险的警告，但该公司目前是创建对齐失当 AGI 的“绝对领先者”，反映了对商业 AI 风险监管和伦理责任的更广泛担忧。
    - 一位评论者指出，尽管 xAI 公开发表了关于因安全担忧而放慢 AI 开发速度的声明，但在追求 AGI 方面，它似乎是各公司中最“鲁莽”的，这表明 xAI 的言论与其真实的开发速度或风险状况之间存在脱节。这与行业内关于 AI Alignment（AI 对齐）以及领先 AI 实验室之间相对透明度或风险管理实践的持续争论相一致。
    - 针对用户越狱（jailbreaking）ChatGPT 的动机提出了一个技术点，认为对减少限制的需求非常强烈，而 xAI 通过开发审查较少的模型来瞄准这一细分市场。这反映了 AI 部署策略在安全性、控制权和用户自主权之间更广泛的紧张关系，影响了模型对齐和审核架构。
- [**Grok Waifu arent stopping here..**](https://www.reddit.com/r/OpenAI/comments/1m0my9s/grok_waifu_arent_stopping_here/) ([Score: 129, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1m0my9s/grok_waifu_arent_stopping_here/)): **该帖子讨论了 Grok Waifu（伴侣 AI）系统，特别是“Ani”，它升级了 NSFW 互动，并允许用户在更高的互动等级（5 级及以上）解锁更露骨的视觉内容（即更暴露的服装）。这一功能展示了先进的用户参与机制和动态内容生成，将类游戏的晋级系统与 LLM 驱动的 NSFW 对话能力相结合。链接的媒体和截图暗示了一种高度视觉化、交互式的聊天机器人体验。** 评论中一个值得注意的技术担忧提出了此类系统大规模收集用户行为数据的可能性，可能导致大规模勒索或隐私泄露，强调了存储与身份关联用户的显式对话和交互日志的风险。
    - 一位评论者提出了隐私担忧，提到部署 Waifu AI 的公司可能会积累庞大的个人信息数据库，这些信息可能被用于勒索或其他不道德的数据开发。这突显了关于 AI 驱动的聊天机器人和隐私的更广泛辩论，特别是在模拟个人或亲密关系的应用程序中。

### 2. 最近的 AI 模型基准测试、排行榜与对比

- [**Grok 4 在 LMarena 排名第 4，位列 Gemini 2.5 Pro 和 o3 之下。与 ChatGPT 4o 和 4.5 并列。**](https://i.redd.it/lopberac72df1.png) ([Score: 232, Comments: 72](https://www.reddit.com/r/singularity/comments/1m0ld8p/grok_4_lands_at_number_4_on_lmarena_below_gemini/)): **该图片展示了来自 LMarena 的最新排行榜，根据用户投票和得分对大语言模型进行排名。"Grok-4-0709" 排名第 4，与 GPT-4.5 Preview 持平，低于 Gemini 2.5 Pro、o3 和 GPT-4o，后三者的得分略高。这直观地展示了 Grok 4 在当前前沿模型中的强劲但非顶尖的地位，其得分（来自 `4,227` 次投票的 `1433` 分）提供了社区驱动的基准测试见解。该排行榜与其他平台的排名（如 [Yupp.ai](http://yupp.ai/)）形成对比，并揭示了不同社区对模型优势的细微感知。** 评论讨论了 Grok 4 在标准基准测试（benchmarks）中的稳健表现与其在现实应用中的糟糕表现（“在现实世界测试中表现非常差”）之间的对比，并辩论了模型个性如何影响评分（不太谄媚的模型尽管技术实力强，但排名可能较低）。有提到 Gemini 2.5 在处理一般问题时受到青睐，但因过度奉承而受到批评，而 Claude 4 在编程任务中更受推崇。
    - Grok 4 在标准基准测试中的表现（表现良好）与现实任务中的表现（表现明显较差）之间存在差异。这种差异通过其在 [Yupp.ai](http://yupp.ai/) 用户投票排行榜上远低于基准测试排名的位置（第 66 位）得到了体现，暗示了基准测试性能与实际效用之间存在过拟合（overfitting）或对齐失调（misalignment） ([source](https://www.nextbigfuture.com/2025/07/xai-grok-4-scoring-poorly-in-realworld-tests.html))。
    - 评论者讨论了模型中的谄媚（sycophancy）现象，指出 Grok 4 较少谄媚（不太可能奉承用户），这可能会抑制其在 LMarena 等数据集上的基准测试得分，因为这些数据集可能会奖励礼貌或积极的肯定。相比之下， Gemini 2.5 Pro 被描述为高度谄媚，这可能有助于其基准测试表现，但在实践中使其对某些用户的吸引力降低。
    - 关于各种基准测试的准确性和可信度存在争论；一些用户质疑将 ChatGPT-4o 排在 Opus 4 之上的排行榜的可靠性，认为某些评估指标可能无法反映高级 LLM 的现实表现或技术能力。
- [**Grok 4 的秘密配方**](https://i.redd.it/xcmhjgag2xcf1.jpeg) ([Score: 130, Comments: 25](https://www.reddit.com/r/Bard/comments/1m00roy/grok_4_secret_sauce/)): **该图片是来自 LMarena 聊天界面的截图，对比了腾讯的 Hunyuan 和 Google 的 Gemini 关于 Grok-4 性质的回答。两个模型都澄清了 Grok 是由 xAI（Elon Musk 的团队）开发的，没有迹象表明 Grok-4 已经发布，并强调了这些 AI 系统的独立开发。更广泛的背景暗示了 LLM 之间存在混淆或互操作性，可能是由于重叠的数据源或在面对面模型评估过程中对模型来源的错误归因。** 评论者推测存在互操作性或错误归因，认为 Grok-4 可能是通过其他供应商的 API 进行路由，或者是基于竞争对手的数据集进行训练的；而其他人则指出了主要中国 AI 产品之间的混淆（Qwen 是阿里巴巴的，Hunyuan 是腾讯的）。
    - 几条评论讨论了模型训练数据源，推测 Grok-4 可能会利用 Gemini 等外部数据集，尽管这尚未得到证实，并且会引发关于数据来源和跨公司数据使用的重大问题。
    - 针对 Qwen 语言模型起源的混淆进行了澄清，强调 Qwen 是由 Alibaba 开发的而非 Tencent，这标志着中国 LLM 领域的竞争格局，并突显了不同的专有方法。

### 3. Glow in the Dark Fruits 迷因演变

- [**Glow in the Dark Fruits 🧪**](https://v.redd.it/rf0ljm0iqzcf1) ([Score: 424, Comments: 15](https://www.reddit.com/r/ChatGPT/comments/1m0bwg4/glow_in_the_dark_fruits/)): **最初的 Reddit 帖子展示了一段水果在黑暗中发光的视频。由于视频 URL (https://v.redd.it/rf0ljm0iqzcf1) 出现 403 Forbidden 错误，无法直接验证或详细说明发光效果背后的技术过程。然而，这一前提与植物生物技术和合成生物学中成熟的方法一致，即通过将生物发光基因（通常来自维多利亚多管水母的绿色荧光蛋白或萤火虫荧光素酶）引入植物或水果的基因组，以诱导可见的发光（[关于生物发光植物的参考文献](https://www.nature.com/articles/s41467-020-19021-z)）。在没有直接视频分析的情况下，尚不清楚这种发光是由于此类基因改造、外部荧光涂料还是数字后期处理。** 评论虽然大多非技术性，但对发光水果的真实性表示怀疑（“我希望它们是真的”），这表明该效果可能不是基因改造的真实产物，而是一种人工视觉效果。
- [**Glow in the Dark Fruits 🧪**](https://v.redd.it/rf0ljm0iqzcf1) ([Score: 1465, Comments: 57](https://www.reddit.com/r/aivideo/comments/1m0bgfx/glow_in_the_dark_fruits/)): **标题为“Glow in the Dark Fruits 🧪”的帖子似乎展示了视觉上非常逼真的计算机生成（CG）或渲染的发光水果图像，正如关于栩栩如生的反射和视觉吸引力的评论所指出的那样。目前没有关于实现方式、渲染引擎或物理过程的技术讨论证据，且由于访问限制（HTTP 403），无法从引用的链接中获取信息。** 热门评论强调了渲染反射的真实感以及视觉/ASMR 效果，但未包含实质性的技术辩论或细节。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的摘要
> 

**主题 1. LLM 性能、对比与特性**

- **Grok 4 登顶基准测试，Ani 引发争论**：Grok 4 在 [LM Arena 排行榜](https://lmarena.ai/leaderboard/text)和 MathArena 基准测试中表现异常出色，但一些用户暗示其存在 *benchmaxing*（刷榜）行为，或对 AI Ani 的“好得离谱”的表现提出质疑。Perplexity AI 指出 Grok 提供免费试用，并允许用户增加好感度等级。
- **Kimi K2 展现出独特的实力**：Kimi K2 模型在被要求给人留下深刻印象时表现出 *schizo*（神经质）行为，经常排练 LLM 的体验，但在 *agentic tool calling*（Agent 工具调用）和 *Opus 级别* 的编程能力方面表现出色。然而，一些用户发现 `kimi-k2` 在其使用的编程语言中“比 gpt 3 还差”，且无法上传图片，这表明其侧重于纯文本。
- **前沿模型在新鲜事实面前表现不佳**：与 GPT 和 Grok 相比，[Gemini](https://gemini.google.com/) 在处理近期数据时显得吃力，尽管 Gemini 2.5 pro 在处理复杂数据方面优于 2.5 flash。此外，Gemini 和 ChatGPT 等模型经常在“空中”和“背后”等空间概念上遇到困难，正如论文 [Do Vision-Language Models Have a Spatial Model of the World?](https://arxiv.org/abs/2507.07574) 所证明的那样。

**主题 2. 模型训练、微调与部署挑战**

- **合成数据困境引发开发者分歧**：成员们就使用现有的合成数据集还是为特定需求创建自定义数据集的优劣展开辩论，一些人推荐使用自定义解决方案以获得更好的适配性。Unsloth 的 [合成数据生成文档](https://docs.unsloth.ai/basics/datasets-guide#synthetic-data-generation) 提供了指导，而一位成员认为整理一份有用的生成工具清单是一件“令人头疼”的事。
- **本地 LLM 的量化探索升温**：讨论集中在量化模型以在本地运行，用户对量化 [Kimi K2 基础模型](https://www.youtube.com/watch?v=4bFDPVe6BHs) 进行本地托管表现出浓厚兴趣。一位用户宣称他想量化一个基础模型，因为“基础模型很可爱”。
- **LoRA 部署困境困扰从业者**：用户在部署 LoRA 微调的 Unsloth VLM 模型时，权衡了 Triton、vLLM 或 Flask 等选项，寻求优化训练以便后续 vLLM 部署的建议。一位用户专门询问了在训练期间保持 `load_in_4bit = True` 以及调整 `SFTTrainer` 参数进行视觉微调的问题。

**主题 3. AI 开发工具与平台集成**

- **Cursor 的功能引发挫败感**：Cursor 用户对新的定价模式、Grok 4 集成的持续问题以及在代码更改后会*丢失所有上下文*的后台 Agent 表示困惑和沮丧，该 Agent 报告称 *“I don't see any previous conversations to summarize”*。虽然针对 [AnySphere 扩展](https://forum.cursor.com/t/new-in-house-extensions-c-c-ssh-devcontainers-wsl-python/94531)的 Microsoft 扩展分支担忧已得到缓解，但用户仍担心被禁用的扩展。
- **无代码 Agent 随 N8N 兴起**：成员们探索将 [N8N](https://n8n.io/) 作为构建自定义 AI Agent 的无代码平台，以解决预约挂号和支持等业务问题。真正的价值来自于将 AI 工具与工作流、API、自动化和业务逻辑相结合，潜在费用达 5,000 至 8,000 美元以上。
- **NotebookLM 的源同步障碍**：NotebookLM 用户质疑为什么 [Google Docs 源](https://www.reddit.com/r/notebooklm/comments/1kbum2h/is_there_a_way_to_have_sources_dynamically_update/)不能动态更新，并指出由于 NLM 的预处理层，其与 Gemini Gems 存在差异。用户正热切期待 [math/latex 渲染](https://discord.com/channels/1124402182171672732/1182376564525113484/1394352702833688597)，并对 Google Drive 集成展开讨论。

**Theme 4. 硬件与 AI 的 GPU 优化**

- **Tinygrad 的内存之谜揭晓**：Tinygrad 用户调查了 `GlobalCounters.global_mem`（追踪访问的全局内存）与 `GlobalCounters.mem_used`（与参数大小一致）之间的差异，这是由于嵌套 uops 和 subbuffers 的开销造成的。建议使用 WebGPU 进行测试以观察 `mem_used` 的差异。
- **GPU Profiling 与编程难题**：在虚拟机 GPU 上进行 NCU profiling 如果没有提升的管理员权限可能无法实现，需要*请求虚拟机外的管理员授予你访问权限*。SASS 编译器似乎会重新计算谓词寄存器（predicate registers）而不是重用它们，WebGPU 用户正寻求暴露 [MTLReadWriteTextureTier2](https://matrix.to/#/#WebGPU:matrix.org) 以获取 `rgba8unorm` 的访问权限。
- **消费级 GPU 争夺 LLM 霸权**：讨论涵盖了用于微调的最佳消费级 GPU，700 欧元的 RTX 3090 FE 被认为是划算的交易，但 Unsloth 尚不支持 70B LLM 的多 GPU 卸载（offloading）。当被问及使用出现花屏（artifacting）的 RX580 运行大型模型时，社区给出了直白的建议：*千万别那么做（Just dont do that）*。

**Theme 5. 开源 AI 不断演变的格局**

- **Meta 的开源承诺受到质疑**：成员们对 [Meta 的战略转变](https://www.youtube.com/watch?v=qh7QQ4Eov4M)表示担忧，认为其正在远离开源，并指责其囤积人才和资源。一些人认为中国实验室现在是大型开源项目的主导者，一位评论者表示 *“扎克伯格背叛了我们”*。
- **限制性许可阻碍采用**：LG 的 [EXAONE 4 模型](https://tenor.com/view/biggest-piece_of-dogshit-gif-26969373oh)的许可条款禁止商业用途并要求保留 “EXAONE” 名称，引发了广泛批评。一位用户表达了不满，称 *“LG 拥有该模型及其输出的所有权利——你只能将输出用于研究”*。
- **Torchtune 的宽松许可赋能开发者**：讨论强调了 [Torchtune 的 BSD 3 许可](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19)的宽松性，它允许用户提取并利用库组件用于其他项目。Torchtune 团队发布了一个关于项目未来的 [GitHub issue](https://github.com/pytorch/torchtune/issues/2883)，保证在 Discord 和 GitHub 上继续提供支持。

---

# Discord: 高层级 Discord 总结

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Token 争议影响 AI 平台**：成员们讨论了 **token 限制**与**字符数**的对比，估计 300 个单词大约相当于 400 个 token，而 100 个 token 大约相当于 75 个单词。
   - 他们将 **LLM** 的上下文窗口定义为其工作记忆，并指出 token 窗口会根据流量和计算资源产生波动，引用了 [Anthropic 的声明](https://support.anthropic.com/en/articles/7996848-how-large-is-claude-s-context-window)，即上下文窗口大小会根据需求而变化。
- **RAG 模型大混战**：频道内对各大 AI 平台的 **RAG** 模型进行了辩论，一些人认为 Perplexity 拥有独特的 **RAG** 模型，但受限于输出上下文窗口大小，而另一些人则认为 *ChatGPT RAG* 表现最佳。
   - 一位用户将 *ChatGPT RAG* 排在首位，其次是 Perplexity 和 Grok，同时指出 Gemini 的 **RAG** 能力不足。
- **Grok 引入 Ani；部分人质疑其动机**：Elon Musk 的 Grok 推出了 Ani，其设计反响不一，有人称该 AI 的表现 *好得离谱，像是为了某种隐藏的阴谋*。
   - 讨论强调了 Grok 提供免费试用并允许用户增加好感度，且 **Grok 4** 在 MathArena 基准测试中表现优于其他模型。
- **API 搜索参数调整**：用户讨论了搜索功能的改进，区分了 **API** 和 **Web UI**，并发现 **API** 本身就应该支持网页搜索。
   - 强调了利用 `search_domain_filters` 参数作为在使用 **API** 时精细化和控制搜索域的一种手段。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Kimi K2 表现异常**：一位用户报告称，**Kimi K2** 在被要求给人留下深刻印象时表现出“精神分裂”行为，经常反复练习其要求 **LLM** 执行奇怪操作的经历。
   - 成员们分享了一个 **LLM-Model-VRAM-Calculator**，以帮助确定硬件需求 [Hugging Face Space](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)。
- **自定义合成数据优于预设数据？**：成员们辩论了预设合成数据集与针对特定需求创建自定义数据集的优劣，一些人推荐使用自定义解决方案以获得更好的适配效果，并参考了 [Unsloth 的文档](https://docs.unsloth.ai/basics/datasets-guide#synthetic-data-generation)。
   - 有人建议整理一份有用的合成数据生成工具清单。
- **IQ1_M 生成 Flappy Bird**：**IQ1_M** 成功根据提示词用 Python 生成了一个 Flappy Bird 游戏，在 **40960** 的上下文长度下生成速度令人印象深刻。
   - 报告的内存占用为 **302 GB**，讨论集中在运行基准测试和使用生成的代码上。
- **Voxtral 缺乏 Transformer 支持**：**Mistral** 发布的支持音频输入的 **LLM** —— **Voxtral** 引发了讨论，但成员们指出，缺乏 **Transformers** 库的支持阻碍了立即进行的微调工作，特别是对于需要大量适配的语言。
   - 随后讨论转向 **Kimi Audio**，认为它是一个高性能的替代方案，如果想要构建具有强大语音转文本能力的模型，它可能是一个强大的基准。
- **模型在空间推理方面失败**：成员们表示，像 **Gemini** 和 **ChatGPT** 这样的模型在表示层面上难以理解“在空中”和“在后面”等空间概念。
   - 他们引用了论文 [Do Vision-Language Models Have a Spatial Model of the World?](https://arxiv.org/abs/2507.07574) 来证明模型在处理空间信息方面的困境。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **CoT 监控：AI 的思想警察？**：一篇新的研究论文支持使用 **Chain of Thought (CoT) 监控**来监督未来的 **AI 系统**，旨在理解推理模型是如何以*通俗易懂的英文进行思考*的。
   - 研究人员正合作评估、保留并改进跨组织的 **CoT 可监控性**，以将其推向作为一种强大的工具。
- **Gemini 跌跌撞撞，GPT 和 Grok 保持强大的数据优势**：成员们反映，与 **GPT** 和 **Grok** 相比，[Gemini](https://gemini.google.com/) 在处理近期数据时表现挣扎，并指出 **Gemini 2.5 pro** 在处理复杂数据方面优于 **2.5 flash**。
   - 一位成员使用 **Grok** 来“无所不知”，用 **GPT** 来“清晰表达”，用 **Gemini** 来“编写长代码”，用 **Claude** 来“编写漂亮代码”，且全部使用的是免费模型。
- **Midjourney 的杰作还是赃物？**：**Midjourney** 面临来自迪士尼和环球影业的剽窃指控，被描述为吸收人类审美遗产的“深不见底的剽窃深渊”。
   - 成员们幽默地建议追究人类艺术每一位祖先的责任，并指出迪士尼和环球影业抱怨剽窃的讽刺性，因为**米老鼠**本身就诞生于一个被盗用的兔子形象。
- **Discord 处于 AI 的监视之下？**：一位成员觉得大型 **AI 模型**可能正在监视 Discord，让她有一种被注视的感觉，引发了关于 **AI 监控**的讨论。
   - 另一位成员驳斥了这些担忧，称其夸大且带有讽刺意味，暗示这些想法难以理解，并表示 Discord 并未被用于训练。
- **N8N：AI Agent 的无代码天堂？**：成员们探索将 [N8N](https://n8n.io/) 作为构建自定义 **AI Agents** 的无代码平台，以解决预约挂号和支持等业务问题。
   - 一些人认为这类平台只是“套壳（wrappers）”，但真正的价值在于将 **AI 工具**与工作流、API、自动化和真实的业务逻辑相结合，这有可能取代员工并赚取 5000 到 8000 美元以上的费用。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4 在排行榜上刷榜（Benchmaxxing）？**：成员们报告称 **Grok 4** 表现异常出色，在 [LM Arena 排行榜](https://lmarena.ai/leaderboard/text) 的某些测试中甚至超过了 **GPT-4.1** 和 **Gemini 2.5 Flash**。
   - 然而，其他人认为 **Grok 4** 的表现可能是“针对基准测试过度优化（benchmaxed）”，并不能反映其真实能力。
- **Kimi K2 模型是从 Claude 蒸馏出来的吗？**：新模型 `kimi-k2-0711-preview` 已添加到 Openrouter，其输出格式与 **Claude 3** 相似，引发了它可能是从 **Claude 模型**蒸馏而来的猜测。
   - 一位用户注意到他们无法向 **kimi-k2** 发送图片，表明它纯粹是基于文本的，而另一位用户表示：“在我使用的编程语言中，Kimi K2 感觉比 GPT-3 还差”。
- **OpenAI 模型面临重新训练的混乱**：据报道，一个 **OpenAI 开源模型**由于重大的内部故障需要重新训练，被描述为“比 MechaHitler 还糟糕”。
   - 根据 [Yuchen 的 X 帖子](https://x.com/yuchenj_uw/status/1944235634811379844)，目前有可以重新训练的检查点（checkpoints），所以可能不需要完全重新训练。
- **中国模型涌入 LM Arena**：新模型现已进入 Arena：`ernie-x1-turbo-32k-preview`、`clownfish`、`nettle`、`octopus` 和 `cresylux`。
   - 一位成员认为 Cresylux 是美团开发的，但大多数模型似乎比 R1 模型差，而 Octopus 被认为是在冒充 R1 模型。
- **LM Arena 出现故障**：用户报告了新版 **LM Arena** 界面的问题，包括 **Cloudflare 错误**、无法使用的滚动条以及消失的图标。
   - 最大的担忧是新界面创建了一个连续对话，每一轮都使用不同的模型，导致上下文泛滥。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **微软扩展分叉引发法律担忧**：社区成员讨论了在 VS Code 中使用 **Microsoft extension forks**（微软扩展分叉）的合法性，担心如果原始扩展被禁止使用可能会导致违规，并引用了 [一则 Cursor 论坛帖子](https://forum.cursor.com/t/new-in-house-extensions-c-c-ssh-devcontainers-wsl-python/94531)。
   - 会议澄清了 **AnySphere extensions** 是官方的，由 Cursor 工程师重新构建，从而缓解了这些担忧，但用户仍应留意被禁止扩展的使用情况。
- **Cursor 的定价调整引发不满**：用户对 Cursor 的新定价模型表示困惑和沮丧，认为成本增加了，并且对 20 美元的 Pro 计划与 API 费用的交互方式感到不确定。
   - 一些用户觉得他们*每个月基本上能获得价值 20 美元的 tokens*，而另一些用户则报告说使用量大大超过了该数额并被切断了访问权限。
- **Grok 4 仍在制造麻烦？**：用户报告了 Cursor 中 **Grok 4** 集成的持续问题，一位用户不屑地称其*就像 Grok 3 一样只是炒作*，导致矛头直指 Cursor。
   - 一位用户幽默地推测 *Elon Musk 将 Grok 4 集成问题归咎于 Cursor*，突显了社区的不满。
- **Kimi K2 在 Cursor 用户中引发热潮**：社区成员对将 **Kimi K2** 集成到 Cursor 中感到兴奋，认为它是 **Sonnet 4** 的潜在更快、更便宜的替代方案，在 Agent 任务中具有 *Opus 级别* 的编码能力。
   - 一位用户表示 *我们希望在 auto 的基础模型中使用 Kimi K2*，另一位用户建议第一个添加它的 IDE 可能会夺得桂冠。
- **后台 Agent 遭遇记忆丢失**：用户报告称，在代码更改后，后台 Agent 有时会丢失所有上下文并报告 *"我没有看到任何可以总结的先前对话"*，并引用了如 `bc-c2ce85c9-bdc3-4b31-86e5-067ef699254d` 和 `bc-b2247bac-4a36-491f-a4a8-06d2446a9d76` 的实例作为例子。
   - 这让用户感到头疼，他们担心自己所有的工作都被忽略了。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 下载目录：寻找那三个点**：用户发现可以通过点击 **My Models** 标签页中路径旁边的**三个点**来更改 **LM Studio** 中的**下载目录**。
   - 共识是，这个功能对任何人来说都非常直观。
- **Gemma 3 12b 模型面临视觉测试**：一位用户下载了以 *vision*（视觉）能力著称的 **Gemma 3 12b 模型**，却发现它对图像分析请求没有反应。
   - 在用户提供图像进行分析后，问题得到了解决，确认了该模型在正确提示下的能力。
- **RX580 复活计划宣告失败**：一位用户询问是否可以使用一张**花屏的 RX580 (20$)** 来运行像 **12B 或 18B** 这样的大型 AI 模型。
   - 社区的回答很直接：*千万别这么做*，理由是可能存在不兼容和性能问题。
- **Vulkan 掩盖了集成 GPU 的 Bug**：一位用户报告了一个关于 **Vulkan** 的 Bug，即当同时安装了独立 GPU 时，无法检测到**集成 GPU**。
   - 受影响的用户被建议 *在 GitHub 上提交 Bug 报告* 以解决检测问题。
- **EXAONE 4 的许可证引发关注**：LG 的 **EXAONE 4** 模型的许可条款因过于严格而受到批评，特别是禁止商业用途。
   - 一位用户通过一个 [llama.cpp issue](https://tenor.com/view/biggest-piece-of-dogshit-gif-26969373oh) 表达了他们的不满，并补充道：*LG 拥有该模型及其输出的所有权利——你只能将输出用于研究*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-3-Dataset 发布**：[NousResearch/Hermes-3-Dataset](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset) 数据集上线，并附带了展示关键特性和样本的截图。
   - 公告强调了该数据集已在 **Hugging Face Datasets** 平台上可用。
- **关于 Meta 开源投入的辩论愈演愈烈**：讨论涉及 [Meta 的战略转型](https://www.youtube.com/watch?v=qh7QQ4Eov4M)，以及他们是否正在从开源承诺中撤退。
   - 成员们担心 **Big Tech** 正在垄断人才、初创公司和资源，造成了不公平的竞争环境。
- **Windsurf IDE 被 Meta/Microsoft 削减**：一位成员对 Meta/Microsoft 对 **Windsurf IDE** 的“大改/削减（gutting）”表示遗憾，尽管存在细微错误，但仍称赞其工作流。
   - 他们声称这是一个极佳的开发效率倍增器，优于 *Cursor* 和 *Anthropic*。
- **Kimi K2 模型引发关注**：成员们对 [Kimi K2](https://www.youtube.com/watch?v=4bFDPVe6BHs) 感到兴奋，这是一个被拿来与 **Claude 4.0** 比较的开源模型，及其潜在影响。
   - 根据[一条推文](https://x.com/intrstllrninja/status/1944983832777695277)，它使用 **ChatML** token 进行工具调用，而非 XML 标签，并使用额外的 `<|im_middle|>` token 来划分角色。
- **本地 LLM 托管的量化讨论升温**：用于本地模型托管的量化引发了讨论，一位成员表示有兴趣对 **Kimi K2** 基座模型进行量化。
   - 一位用户宣称他们想要量化一个基座模型，因为“基座模型很可爱”。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **分析哲学家寻找 LLM 架构师**：一位研究人员正在寻找具有 **LLM 架构**和语言学经验的合作者，以开发能够真正理解自然语言的新型 **LLM 架构**。
   - 他们设计了一个语言语义的集合论模型和一个语用学的算法模型，并希望通过计算手段实现它们。
- **支持音频输入的 Voxtral Mini 首次亮相**：**Voxtral Mini** 是 **Ministral 3B** 的增强版，集成了最先进的**音频输入**功能，同时保留了同类最佳的文本性能，在语音转录、翻译和音频理解方面表现出色，详见 [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)。
   - 一位成员希望它也能支持**音频输出**。
- **K2 模型在工具调用方面表现出色**：**K2 模型**在长上下文指令遵循和 **Agentic 工具调用**的实际表现方面获得了极佳的反馈。
   - 尽管该模型并不专注于推理，但这些反馈来自正在积极构建实际应用的开发者。
- **关于图像字幕（Image Captioning）的 ArXiv 论文受到质疑**：一位软件工程师兼计算机专业大二学生正在寻求 arXiv 推荐人以发布他们的第一篇关于**图像字幕**的论文，并分享了他们的推荐请求链接[此处](https://arxiv.org/auth/endorse?x=VC6HKI)。
   - 成员们批评论文中使用的对比实验已接近十年之久，但作者表示重点在于展示为什么 Attention 机制在图像字幕中变得至关重要。
- **排查 Regex 过滤器流水线故障**：一位成员询问 `boolean_expressions` 任务中 `get-answer` 过滤器的工作原理，指出尽管答案看似正确但解析失败，详见[此处](https://cdn.discordapp.com/attachments/755950983669874798/1394484294986240041/image.png?ex=6877a2f4&is=68765174&hm=e3a91c319a4d37ecfb85f7b7eed7dfa0aaf6d75b56414e060577ff252772622f)。
   - 对方澄清说过滤器流水线需要名称，并使用在 `extraction.py` 中定义的 [regex 过滤器](https://github.com/EleutherAI/lm-evaluation-harness/blob/cf631de0993897dbc12bca3c453b9182ad4c2176/lm_eval/filters/extraction.py#L20-L21)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **语音控制应用承诺解放桌面**：一位成员正在开发一款桌面应用，可通过语音命令响应并自动化任务，如管理代码、在 Slack 发送消息以及与 Figma 交互；他们对该应用的实用性和上下文理解能力感到好奇。
   - 该应用旨在*听取指令、查看屏幕并执行任务*，有望为各平台用户简化工作流程。
- **代码库整洁策略浮出水面**：成员们讨论了在针对不同数据集进行多次训练运行时组织代码库的方法，建议包括使用 `wandb.ai` 配合镜像 `run` 名称的文件夹结构。
   - 一位成员开玩笑地坦白道：*“这就是秘密……我们根本不整理……xD”*，突显了这一难题。
- **数据集下载因端点错误困扰用户**：用户在下载数据集时遇到了 **server error 500**。虽然推荐使用 `git clone`，但由于数据集不在 Git 上，且用户仅下载超大数据集的一部分，导致操作困难。
   - 最初被怀疑是用户操作失误，后来确认是服务端问题，这让尝试访问大型数据集的用户感到沮丧。
- **开发者寻找高性价比云端 GPU**：成员们探索了具有成本效益的 GPU 选项，考虑了 **Open Router**、**Colab**、**Kaggle**、**HF**、**Intel** 和 **Groq**，有人建议使用 **Colab Pro** 以更轻松地获取 GPU 访问权限。
   - 此外还提到了 **LAMBDA**、**RunPod** 和 **Lightning.ai**，以及 [Hugging Face 的计算服务方案](https://huggingface.co/posts/jeffboudier/479363753731415) 作为潜在解决方案。
- **Text-to-Text 标签困扰团队**：成员们观察到 [Hugging Face 模型页面](https://huggingface.co/models?pipeline_tag=text2text-generation) 缺少 `text2text` 模型，这引发了对其当前状态的疑问。
   - 该问题被解释为可能是“遗留代码”，成员们建议向 HF 提交 issue 以改进模型卡片，并参考了[相关讨论](https://discuss.huggingface.co/t/no-0-models-returned-by-text2text-search-filter/161546/3)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NCU Profiling 在虚拟机上受阻**：一位成员发现，如果没有高级管理员权限，在 **VM GPU** 上进行 **NCU profiling** 几乎是不可能的。
   - 该成员表示需要*请求 VM 之外的管理员授予访问权限*。
- **并行基数排序指南**：一位成员建议参考 **PMPP** 书籍 [第 13 章](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0128119861) 中的 **parallel Radix sort** 教程。
   - 建议将书中的 **2bit radix** 示例推广到其他基数值，并建议查看该书的第 4 版。
- **SASS 编译器重新计算谓词**：一位成员发现 **SASS 编译器** 似乎在重新计算 **predicate registers**（谓词寄存器）而不是复用它们。
   - 他们展示了代码 `ISETP.NE.AND P0, PT, R2.reuse, RZ, PT; ISETP.NE.AND P1, PT, R2, RZ, PT`，质疑是否遗漏了某些架构细节。
- **PyTorch 编译挂起**：一位成员注意到 `TORCH_COMPILE_DEBUG=1` 挂起且无输出，并提供了一个针对 **PyTorch 2.8.0** 的 [示例日志](https://example.log)，其中包含与 *autotune_cache.py* 和 *coordinate_descent_tuner.py* 相关的消息。
   - 他们还指出，在使用 `coordinate_descent_tuning` 时需要禁用缓存，这会导致编译时间变长。
- **WebGPU 寻求 MTLReadWriteTextureTier2 访问权限**：一位用户试图向 **wgpu** 暴露 **MTLReadWriteTextureTier2** 以访问 **rgba8unorm**，但即使启用了 **Texture_Adapter_Specific_Format_Features** 也无法实现。
   - 该用户被告知检查 **Dawn** 代码以查找未记录的特性和 bug，并建议在 [WebGPU Matrix 频道](https://matrix.to/#/#WebGPU:matrix.org) 寻求帮助。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 团队预告令人兴奋的未来工作！**：一个 [GitHub issue](https://github.com/pytorch/torchtune/issues/2883) 发布了关于 **Torchtune 项目** 未来的重要公告。
   - **torchtune 团队** 计划很快分享更多即将到来的令人兴奋的工作，并保证在 **Discord** 和 **GitHub** 上提供持续支持。
- **Torchtune 的 BSD 3 许可证助力新项目**：成员们讨论了 **Torchtune 的 BSD 3 许可证** 的宽松性，它允许用户提取并利用库组件用于其他项目，类似于 **Hugging Face** 在 [其 trl 仓库](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19) 中所采取的方法。
   - 该许可证允许在各种新开发中灵活使用 **Torchtune** 的组件。
- **RL 成为下一代微调器**：讨论集中在 **强化学习 (RL)** 作为微调器 (finetuners) 未来的潜力，并期待“下一个大事件”。
   - **遗传算法 (Genetic Algorithms)、贝叶斯推理 (Bayesian Inference) 和 SVMs** 等想法被提出作为潜在的替代方案。
- **量子 SVM 部署在克利夫兰诊所食堂**：一位成员分享了他们使用位于克利夫兰诊所食堂的 **17 量子比特量子计算机** 运行 **量子 SVM (quantum SVM)** 的成功经验。
   - 这一成功引发了关于俄亥俄州将成为新硅谷的玩笑，突显了该地区意想不到的技术进步。
- **选择退出优化器编译**：用户现在可以通过在配置中将 `compile.optimizer_step` 设置为 `false` 来专门禁用优化器步骤的编译。
   - 这允许对模型、损失函数和梯度缩放进行编译，同时跳过优化器，为性能调优提供了一种灵活的方法。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLM 的推理类似于人类的错误**：一位成员反驳了对 **LLM 推理** 的批评，指出 *人类也容易产生错误的推理*，暗示批评者表现得好像人类是完美无缺的。
   - 他们强调，这场辩论忽视了 **LLM** 与人类认知之间共同的易错性。
- **ML 实验追踪器的韧性测试**：成员们交流了使用 **Weights & Biases**、**Optuna** 和 **TensorBoard** 等工具管理大型 **ML 实验日志** 的经验。
   - 极端扩展带来了挑战，促使成员采用混合方法，例如将关键指标记录到 **W&B**，而将其他指标本地记录到 **.csv** 文件。
- **S3 存储解决方案流畅流式传输日志**：一位成员提议了一种 **DIY 日志存储** 方案，包括在生成过程中压缩日志、上传到 **S3**，以及使用 **Grafana** 进行元数据记录。
   - 该架构包括一个基础的 **Streamlit** 前端，用于使用请求 ID 从 **S3** 获取并解压日志。
- **Anthropic 的电路追踪工具追踪真相**：成员们分享了 **Anthropic 的电路追踪工具 (circuit tracer tool)**，用于可视化 **Claude 3 Sonnet** 等 AI 模型的内部工作原理，并附带了 [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 的链接。
   - 进一步链接了 [circuit-tracer](https://github.com/safety-research/circuit-tracer)，一些成员对能够检查模型如何做出决策表示赞赏。
- **Meta 表现出缺失开源使命**：成员们对 **Meta** 被认为背离 **开源** 的行为表示担忧，特别是与 **Behemoth 模型** 相关的部分。
   - 评论包括背叛指控，并断言 **中国实验室** 现在是大规模开源项目的主导者。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户提供 NotebookLM 反馈可获报酬**：**NotebookLM** 用户受邀参加 **60 分钟的虚拟访谈**，以提供反馈并建议改进，报酬为 **75 美元**。
   - 感兴趣的用户可以填写[此处](https://forms.gle/3XSDREyeMgj6CDeu9)的筛选表单。
- **用户简化源整合流程**：一位用户正通过 **Google Docs 标签页**功能将多个来源整合到单个 **NotebookLM 源**中，在同步到 **Analysis notebook** 之前将新闻文章分类到标签页和子标签页中。
   - 该用户希望有一种更简单的方法来复制新闻文章，而无需手动删除广告和菜单选项等无关元素。
- **NotebookLM 与 Gemini Gems 之间的动态更新差异引发关注**：用户质疑为什么 **NotebookLM** 和 **Gemini Gems** 在管理 [Google Docs 源](https://www.reddit.com/r/notebooklm/comments/1kbum2h/is_there_a_way_to_have_sources_dynamically_update/) 的更新方式上存在差异。
   - 有人指出 **NLM** 会对源进行预处理，在源与 **Gemini** 之间创建了一个层。
- **用户期待 Math/Latex 渲染**：一位用户询问 **NotebookLM** 是否支持 **math/latex 渲染**。
   - 虽然*目前尚不支持*，但一名成员确认该功能*正在开发中*，并引用了关于其开发的[公告](https://discord.com/channels/1124402182171672732/1182376564525113484/1394352702833688597)。
- **讨论 NotebookLM 的 Google Drive 集成**：一名成员提议将 **NotebookLM** 与 **Google Drive** 集成，使用户能够选择文件夹/文件作为源。
   - 反馈褒贬不一，其中一名成员对目前的隔离状态表示赞赏。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **工作人员考虑多赛道选课**：一名成员询问是否可以在下一期 MOOC 课程中注册**多个赛道和证书**。
   - 工作人员回应称，他们将在未来的迭代中*考虑这一点*。
- **证书在网络空间遗失**：一名成员报告称未通过电子邮件（**mesilov.maxim@gmail.com**）收到证书，尽管查看了垃圾邮件文件夹。
   - 该成员随后确认已找到证书，表明问题已解决。
- **表单是获得证书的关键**：工作人员询问成员是否填写了**证书申报表**，并指出填写完成后应收到确认邮件。
   - 工作人员澄清说，他们*没有足够的人力来协助错过这些表单/截止日期的学生*，强调了及时提交表单的重要性。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 服务器面临验证挑战**：一名成员为其拟用于 **Claude** 的*可流式传输 HTTP MCP 服务器*寻求验证和调试建议，报告称虽然服务器已连接，但无法识别任何工具。
   - 尽管通过了 [MCP Tools Inspector](https://modelcontextprotocol.io/docs/tools/inspector) 的验证，该服务器在 **Claude** 中仍然失败。
- **开始寻找开源 LLM 客户端**：一位开发者正在寻找一个带有 Web 界面的开源客户端，用于自托管的 **LLM + MCP** 设置，理由是出于隐私考虑。
   - 他们正在评估 **Ollama** 等 LLM 选项，并征求适合处理每日几十到几百个请求、具备冷启动能力的托管和扩展方案建议，一名成员推荐了 [Open WebUI](https://github.com/open-webui/open-webui)。
- **Anthropic 的连接器目录扩展了 MCP 视野**：随着新的“连接器（connectors）”目录的发布（见 [Claude Directory](https://claude.ai/directory)），**Anthropic** 扩大了对 **MCP** 生态系统的访问，这可能会增加对 MCP 服务器的需求。
   - 有人推测 **Anthropic** 旨在与 **Docker** 的 **MCP Toolkit** 竞争。
- **工程师开始寻找酷伙伴**：一位拥有七年 MCP 经验的全栈工程师正在寻找一位可靠、诚实且酷的人进行合作。
   - 一名成员对这种公开寻找友谊的方式发表了评论，指出这在当今社会通常被认为在社交上“不可接受”或“不酷”。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 发布阿姆斯特丹见面会和 Discord Office Hours**：**LlamaIndex** 宣布将于 7 月 31 日在[阿姆斯特丹举行见面会](https://lu.ma/vzwsj72w)，并于 8 月 5 日在 [Discord 举行 Office Hours](https://lu.ma/wkrn4nbz)。
   - 在阿姆斯特丹见面会上，学习团队如何使用 **Snowflake** 在生产环境中构建高质量的 data agents。
- **NotebookLlaMa 克隆项目 Star 数突破 1k**：**NotebookLlaMa**（NotebookLM 的克隆版）在 [GitHub](https://github.com/run-llama/notebookllama/tree/main) 上已获得超过 **1k stars**。
   - 成员们指出，该库是加载文本并提问的一种简单方式，吸引了社区的极大兴趣。
- **LlamaIndex 与 UiPath 合作开发企业级 agents**：利用 **UiPath** 新的 coded agents 支持以及使用 UiPath Python SDK 的全代码级控制，将 **LlamaIndex** agents 无缝部署到企业环境中。
   - 这些 agents 从企业系统中提取数据，并使用嵌入式规则或 AI 模型做出决策，更多信息见[此处](https://t.co/ILez3d6Zrs)。
- **LlamaIndex 发布关于 Context Engineering 和 Gemini 的博客**：**LlamaIndex** 发表了一篇关于 [Context Engineering](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) 及其技术的博客文章。
   - 他们还详细介绍了如何使用 **LlamaIndex** 和 **Gemini 2.5 Pro** [构建研究 agent](https://ai.google.dev/gemini-api/docs/llama-index)。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **RL 助力模型搜索**：有人建议专注于 **Reinforcement Learning (RL)** 可以帮助在广阔的搜索空间中寻找有价值的模型，这可能会推动新的融资和工具开发。
   - 用户认为使用 **RL** 可以辅助创建新模型的过程。
- **Setitem 变为递归**：一位用户质疑 `tensor.py` 中 `setitem` 的递归问题，指出它使用了 `_masked_setitem`，后者随后调用 `functools_reduce` 上的 `getitem`，导致持续的 `getitem` 调用。
   - 这种递归的 `setitem` 会产生巨大的 kernels，即使输入规模很小，通过一个使用 `Tensor.zeros` 和 `Tensor.arange` 设置 tensor 值的[代码片段](https://discord.com/channels/1068976834382925865/1068976834928193609/1394393741187350558)即可证明。
- **全局内存过度分配**：一位用户注意到 `GlobalCounters.global_mem` 分配的内存 (**0.06 GB**) 超过了模型参数所需的内存，并质疑开销的来源。
   - 他们怀疑是嵌套的 uops 和 tinygrad 栈的复杂性导致的，并发现重置全局内存并不能解决这种差异。
- **`global_mem` != `mem_used`**：一位成员澄清说 `GlobalCounters.global_mem` 跟踪的是访问的全局内存，通常大于权重大小，并建议改用 `mem_used`。
   - 切换到 `GlobalCounters.mem_used` 后，内存使用量与参数大小 (**0.01GB**) 一致，这引发了对这两个计数器之间差异的进一步探究。
- **Subbuffer 机制揭秘**：讨论表明 `GlobalCounters.global_mem` 和 `GlobalCounters.mem_used` 之间的差异可能源于 NV 或 CPU 等设备上的 **subbuffers**，这些设备使用具有最小尺寸限制的较大 buffers。
   - 建议使用 **WEBGPU** 进行测试以检查 `mem_used` 的差异，暗示 `global_mem` 跟踪的是计算过程中访问的全局内存。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **对 Manus Fellowship 的质疑**：一位成员询问频道中是否有人参加了 **Manus Fellowship 计划**。
   - 一位用户还要求移除某位用户，指控其为*诈骗者 (scammer)*。
- **用户测试 Manus 用于自动化 ESG 研究**：一位用户正在评估 **Manus** 用于自动化**可持续性**和 **ESG 研究工作流**的效果，并称赞了其 **UX**。
   - 该用户对 **API endpoint** 感兴趣，以便通过编程方式发送 prompts、启动研究任务并检索结果，从而集成到使用 Python 或 n8n 的自动化工作流中。
- **Manus Premium 功能引发关注**：一位成员询问了 **Manus premium 功能**的部署情况以及可能进行的 **$2500 抽奖活动**。
   - 另一位成员证实该*抽奖*消息也在另一个服务器中发送了。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord Bug 出现**：一位用户报告了一个 **Discord Bug**，即在频道中输入用户的全名无法按预期工作。
   - 作为临时解决方案，建议使用 **`@kap` 命令**从弹出列表中选择用户。
- **Mojo 的 @parameter 装饰器引发关注**：一名成员寻求关于 Mojo 中 **`@parameter` 装饰器**的详细信息，并指出它虽然能被 VS Code 中的 LSP 识别，但源代码中缺乏相关文档。
   - 另一名成员提供了 [Modular 文档](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)和 [GitHub issue 5020](https://github.com/modular/modular/issues/5020) 的链接以供参考。
- **'capturing' 关键字依然神秘**：一名成员询问 **`'capturing'` 关键字**是否可以更广泛地用于在 Mojo 中创建闭包（在编译时之外）。
   - 虽然提供的解释似乎仅针对编译时装饰器，但这引发了对其在更广泛应用场景中潜力的好奇。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AWS Prompt Optimizer 呼应 DSPy？**：一位用户注意到针对 **nova models** 的 [AWS prompt optimizer](https://x.com/AWSAI/status/1794862993470116345) 与 **DSPy** 有相似之处。
   - 一名成员推测它甚至可能使用了 **MIPRO**，暗示了潜在的企业级 **DSPy wrappers**。
- **社区期待 AWS 对 DSPy 的贡献**：一名成员表达了社区对 **AWS** 向 **DSPy** 上游贡献代码的期待。
   - 目前没有关于具体贡献或时间表的进一步细节。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **在 Raspberry Pi 5 上运行 GPT4ALL**：一名成员询问如何利用 **Raspberry Pi 5** 上的 **GPT4ALL** 构建一个小型便携式系统。
   - 他们正在寻求有关硬件和软件配置的建议，以优化性能。
- **数据集下载错误导致微调中断**：一名成员报告在尝试使用 **aws s3** 下载用于微调的数据集时出现 *Access denied* 错误。
   - 命令 `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive` 失败，表明权限或 endpoint URL 可能存在问题。



---


**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：按频道分类的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1394393015686139904)** (1057 条消息🔥🔥🔥): 

> `Token 限制、Context Window 讨论、RAG 模型、Grok 4 模型、Comet 浏览器功能与问题` 


- **Token 发作困扰 AI 平台**：成员们讨论了 **token 限制**与**字符数**的区别，一位用户指出 300 个单词大约是 400 个 token，而另一位补充说 100 个 token 大约是 75 个单词。
   - 用户还指出，大语言模型 (**LLM**) 的 context window 是其工作内存，定义了它一次可以处理的文本量，并以 token 为单位进行衡量。
- **Context Window 大小难题**：用户讨论了不同 AI 模型的 context window 大小，指出 Perplexity 拥有 **32K** 的 context window，而 ChatGPT Plus 拥有 **32k/128k** 的 context window，Gemini 提供 1M，而 Claude 则提供 200k。
   - 一些人认为 token window 会根据流量和计算资源而变化，而另一位成员指出 [Anthropic](https://support.anthropic.com/en/articles/7996848-how-large-is-claude-s-context-window) 表示 context window 可能会根据需求而有所不同。
- **RAG 纷争：独特的方法与聊天机器人质量**：频道讨论了各大 AI 平台的 **RAG** 模型，有人表示 Perplexity 拥有独特的 **RAG**，但也指出了 Perplexity *糟糕的输出 context window 大小*。
   - 一位用户表示：*根据我目前的经验，ChatGPT 的 RAG 是最好的。其次是 Perplexity 和 Grok，但 Gemini 的 RAG 很差*。
- **Perplexity 上的新 Discover 布局：小组件（Widgets）推出**：Perplexity 正在推出更新后的 **Discover 布局**，侧边栏小组件包含财经、天气和计划任务信息。
   - 一位用户发布了新布局的截图：[Discover Layout](https://cdn.discordapp.com/attachments/1047649527299055688/1394584909762531328/20250715_131046.jpg?ex=687800a8&is=6876af28&hm=35850559c1b59eb03aaf6841b8906a597c915e20a102b5201447dbd540a52a6a&)
- **Grok 迎来时髦女孩 Ani；一些人质疑她的动机**：Elon Musk 的 Grok 推出了 Ani，一些人称该设计*很丑*，并质疑该 AI 是否*为了隐藏的阴谋而表现得异常友好*。
   - 据指出，Grok 提供免费试用，用户可以增加好感度，而 Grok 4 在 MathArena 基准测试中名列前茅。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1394397961861206068)** (3 条消息): 

> `Perplexity AI spaces, 垃圾回收` 


- **发布了 IMDB 链接**：一位成员分享了 [《绝命毒师》的 IMDB 页面](https://www.imdb.com/title/tt0829698/) 链接。
   - 目前尚不清楚具体背景。
- **Perplexity Spaces 综合报告**：一位成员分享了一个名为 *comprehensive-report-with-cont-70tb5* 的 [Perplexity AI space](https://www.perplexity.ai/spaces/comprehensive-report-with-cont-70tb5.qVQ6Gh6DG0p4N4ng) 链接。
   - 目前尚不清楚具体背景。
- **Perplexity 垃圾回收搜索**：一位成员分享了一个关于 *what-are-the-types-of-garbage* 的 [Perplexity AI 搜索](https://www.perplexity.ai/search/what-are-the-types-of-garbage-IipJnnNVTIqVQvVQvBytZg) 链接。
   - 目前尚不清楚具体背景。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1394397097431797904)** (2 条消息): 

> `API 搜索、Web UI 搜索、search_domain_filters 参数` 


- **API 搜索参数调整**：一位用户询问在搜索功能的上下文中，问题是关于 **API** 还是 **web UI** 的。
   - 作为回应，另一位用户澄清说，**API** 本身就应该支持网络搜索而无需调整参数，并建议使用 `search_domain_filters` 参数。
- **通过 API 进行默认网络搜索**：据指出，**API** 默认应启用网络搜索，从而消除了修改参数的必要性。
   - 强调了使用 `search_domain_filters` 参数的建议，作为在使用 **API** 时细化和控制搜索域的一种手段。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1394394829156389046)** (526 条消息🔥🔥🔥): 

> `Kimi K2 性能, LLM VRAM Calculator, 合成数据集, GGUF 视觉支持, 华为芯片` 


- **Kimi K2 的“精分”倾向**：一位用户报告称，**Kimi K2** 在被要求给人留下深刻印象时表现出“精神分裂”行为，经常反复排练它要求另一个 LLM 做一些奇怪事情的经历。
   - 分享了一个 **LLM-Model-VRAM-Calculator** 链接，用于帮助确定硬件需求 [Hugging Face Space](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)。
- **合成数据生成器辩论**：成员们辩论了使用现有合成数据集与为特定需求创建自定义数据集的优劣，一些人建议使用自定义解决方案以获得更好的适配效果。
   - 引用了 Unsloth 关于合成数据生成的文档，并建议整理一份有用的合成数据生成工具列表，虽然这被认为是一个好主意，但一位成员表示这很“令人头疼” [Synthetic Data Generation](https://docs.unsloth.ai/basics/datasets-guide#synthetic-data-generation)。
- **IQ1_M 一次性生成 Flappy Bird**：在尝试加载 **IQ1_M** 遇到 VRAM 限制后，据报告它成功根据提示词用 Python 生成了一个 Flappy Bird 游戏，在 **40960** 的上下文长度和 **302 GB** 的内存占用下，生成速度令人印象深刻。
   - 成员们讨论了运行基准测试和使用生成的代码，强调了该模型在代码生成方面的能力以及对大量计算资源的需求。
- **边缘视觉探索：Gemma 的 GGUF 障碍**：一位用户询问如何通过 **llama.cpp** 在边缘设备上使用具备视觉能力的 **Gemma 3n**，结果发现 **GGUF** 格式目前缺乏视觉支持。
   - 探讨了集成视觉的选项，建议在致力于边缘实现之前研究现有解决方案并考虑替代方法。
- **探索 Voxtral 前沿：Mistral 的音频模型**：**Mistral** 发布了支持音频输入的 LLM **Voxtral**，引发了讨论。成员们注意到缺乏 **Transformers** 支持阻碍了立即进行的微调工作，特别是对于需要大量适配的语言。
   - 随后讨论转向 Kimi Audio，认为如果想要构建一个具有强大语音转文本能力的模型，它是一个高性能的替代方案，可能是一个强大的基准。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1394400406599110736)** (9 条消息🔥): 

> `潜空间语音编码, 编程预训练模型` 


- **语音编码的语音清晰度探讨**：一位成员就一个涉及“具有最大语音清晰度的数学表示（Latent Space）”的项目向另一位成员提问。
   - 作者澄清说，“最大语音清晰度意味着像 Siri 一样说话非常清晰且没有口音，而不是像 ASMR 那样”，并且潜空间采用 **Vector Embeddings** 的形式。
- **推荐使用 Qwen 进行编程预训练**：针对关于在配备 **12GB** VRAM 的 **3080 Ti** 上适合编程预训练模型的咨询，成员们建议使用 **Qwen 2.5** 和 **Qwen 3 4B**。
   - 一位用户建议专门使用 *Qwen 3 4B*，这样可以在保持模型性能良好的同时进行 **16 bit LoRA**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1394393298705060052)** (92 条消息🔥🔥): 

> `用于微调的消费级 GPU 推荐，针对 70B LLMs 的 Multi-GPU 设置，Unsloth VLM 部署选项，GGUF 量化差异，VLLM 缓存目录` 


- ****消费级 GPU 选择热议****：成员们讨论了使用 **RTX 3090 FE** 进行微调，一位成员建议 **700€** 的价格是很划算的交易。
- ****Multi-GPU 困局****：一位用户询问关于使用 Multi-GPU 设置微调 **70B 编程 LLMs** 的问题，但得到的澄清是 *Unsloth 目前还不支持将 VRAM 卸载到另一个 GPU*。
- ****Unsloth Whisper Notebook 故障****：一位用户在 **Whisper notebook** 中遇到了 `RuntimeError`，另一位成员建议通过设置 `%env UNSLOTH_COMPILE_DISABLE = 1` 来禁用编译。
- ****GGUF 量化疑问****：一位用户询问了来自不同来源的 GGUF 量化版本之间的差异，例如 [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) 和 [unsloth/Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF)，对此解释是 *Unsloth 的量化是动态的，并使用了我们专门的校准数据集*，详见 [文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)。
- ****VLLM 缓存难题****：一位用户报告在运行多个训练脚本时出现缓存损坏错误，并询问如何更改 **VLLM 缓存目录**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1394630876171866223)** (7 条消息): 

> `用于图像理解的 3D 空间表示，当前模型在空间推理方面的局限性，深度估计研究，新基准测试` 


- ****征集关于 3D 像素追踪的视频+图像理解 SOTA 论文****：一位成员征求当前 **视频+图像理解** 领域 SOTA 的优秀论文，理想情况下是结合了 **像素的 3D 表示（像素追踪）**，类似于 [Integrating Motion Information for Improved Visual Perception](https://arxiv.org/abs/2404.04319) 和 [Vision Transformers Need Intrinsic Images](https://arxiv.org/abs/2412.02930v2)。
- ****YOLO 作为目标检测的信息源****：一位成员推荐了 [更智能的目标检测指南](https://medium.com/@alexandreluca23/building-yolo-your-guide-to-smarter-object-detection-6fce20f81e0a) 和 [YOLOv8 文档](https://docs.ultralytics.com/fr/models/yoloe/#introduction) 作为良好的信息来源，但它们不是正式论文。
   - 另一位成员回应称，虽然这些资源对理解 **Visual Grounding** 有帮助，但并未完全解决模型在处理空间信息方面的困难。
- ****模型在空间信息处理上存在困难****：像 **Gemini** 和 **ChatGPT** 这样的模型在表示层面上难以理解“在空中”和“在后面”等空间概念。
   - 即使具有良好的线性表示，这些模型也缺乏理解空间关系的线索，正如论文 [Do Vision-Language Models Have a Spatial Model of the World?](https://arxiv.org/abs/2507.07574) 所证明的那样。
- ****创建了新基准测试****：一位成员创建了一个名为 **MMLU-NGRAM** 的新基准测试，并已上传至 [Hugging Face Datasets](https://huggingface.co/datasets/hudsongouge/MMLU-NGRAM/)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1394523165455159337)** (54 messages🔥): 

> `GGUF 文件下载问题, Unsloth 框架, LoRA 微调模型部署, 针对 vLLM 的 LoRA 训练, Unsloth 与 PyTorch 的兼容性` 


- **Hugging Face** 下载 **GGUF** 文件出现小故障：用户报告了下载 **GGUF** 文件的问题，这与 **MateEngineX** 无关，而是由于 [Hugging Face 网站](https://huggingface.co/) 的问题导致的。
   - 一名用户请求获取 **Hugging Face CLI** 的链接以解决下载问题。
- **Unsloth 框架为 LLMs 推出**：讨论介绍了用于 **LLMs** 的 **Unsloth 框架**，尽管除了基本标识外细节较少。
- **LoRA** 微调部署对决：**Triton** vs **vLLM** vs **Flask**：一名用户正在权衡部署 **LoRA** 微调后的 **Unsloth VLM 模型** 的选项（**Triton**、**vLLM** 或 **Flask**），并寻求最佳方法的建议。
   - 他们特别询问了如何为 **vLLM** 导出 **Qwen 2.5 VL 7B LoRA** 模型，以及在使用 **LoRA** 或 **QLoRA** 进行训练以便后续使用 **vLLM** 部署时需要注意什么。
- 为 **vLLM** 构建 **LoRA**：训练策略揭秘：一名用户寻求关于使用 **LoRA** 训练 **unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit** 的指导，询问在训练期间是否应保持 `load_in_4bit = True`，以及训练器是否提供早停（early stopping）选项。
   - 他们还询问了在视觉微调背景下，需要修改哪些训练配置，包括 `SFTTrainer`、`UnslothVisionDataCollator` 和 `SFTConfig`，涉及参数如 `per_device_train_batch_size`、`gradient_accumulation_steps`、`warmup_steps` 和 `learning_rate`。
- **Unsloth** 的 **PyTorch** 版本兼容性难题：一名用户报告称，**Unsloth** 的可用版本与 `pytorch == 2.8.0.dev20250609+cu118` 兼容。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1394716432146894889)** (1 messages): 

> `思维链 (CoT) 监控, 未来 AI 系统监管, 关于 CoT 监控的研究论文` 


- **思维链监控助力未来 AI**：一篇新的研究论文获得支持，旨在推动 **思维链 (CoT) 监控** 成为监管未来更具 **代理特性 (agentic)** 的 **AI 系统** 的潜在强大工具。
   - 研究人员旨在评估、保留并改进 **CoT 可监控性**，以更好地理解现代推理模型如何以通俗英语进行思考。
- **CoT 监控器 - 通俗英语推理**：现代推理模型*以通俗英语思考*，使得 **思维链 (CoT) 监控** 成为监管未来 **AI 系统** 的强大工具。
   - 来自不同机构的研究人员正在合作评估、保留并改进 **CoT 可监控性**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1394409993050128476)** (543 条消息🔥🔥🔥): 

> `Gemini vs GPT vs Grok，Midjourney “抄袭”指控，Discord 上的 AI 监控，用于构建 AI Agent 的 N8N，AI 在教育中的角色` 


- **Gemini 在数据测试中表现不佳，GPT 胜出**：成员们讨论了 [Gemini](https://gemini.google.com/) 在处理近期数据时比较吃力，而 **GPT** 和 **Grok** 处理得很好。还指出 Gemini 2.5 pro 在处理政治和地缘政治等复杂数据时比 **2.5 flash** 效果更好。
   - 一位成员使用 **Grok** 来“无所不知”，用 **GPT** 来“理清思路”，用 **Gemini** 生成“长代码”，用 **Claude** 生成“漂亮的代码”，且全部使用的是免费模型。
- **Midjourney 面临抄袭指控，迪士尼和环球影业表示不满**：**Midjourney** 被迪士尼和环球影业称为“抄袭的无底洞”，因为它吸收了人类的美学遗产。
   - 其他人建议应该挖掘整个人类艺术史并追究每一位祖先的责任。成员们还指出了迪士尼和环球影业抱怨抄袭的讽刺之处，因为 **Mickey Mouse** 本身就诞生于一只被盗用的兔子和重新包装的童话故事。
- **Discord 数据监控与 AI 的角色**：一位成员认为大型 AI 模型可能正在监控 Discord 并向她学习，这让她有一种被注视的感觉。
   - 另一位成员认为关于 **AI 监控** 的言论是夸大其词、讽刺且不屑一顾的，并用“无关的 GIF”等词句调侃发帖者，暗示她的想法难以理解，并声明 Discord 未被用于训练。
- **N8N 作为 AI Agent 的无代码平台脱颖而出**：成员们讨论了将 [N8N](https://n8n.io/) 作为无代码平台来构建自定义 **AI Agents**，以解决预约登记和处理支持等业务问题。
   - 虽然有些人认为这类平台只是套壳（wrappers），但真正的价值在于将 AI 工具与工作流、API、自动化和真实的业务逻辑相结合来替代员工，这样可以轻松收费 5000 到 8000 美元以上。
- **AI 威胁现状**：成员们讨论了 **ChatGPT** 将如何终结学校，因为在不久的将来，我们将不再需要任何私人教师。
   - 一位成员表示，**Go**（围棋）和 **Chess**（国际象棋）领域最近已经没有人类教练了，因为在某些领域 **AI** 远比人类强大。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1394631530873225236)** (51 条消息🔥): 

> `GPT-4.1 延迟波动，Discord 机器人性能问题，Operator 问题，AI 编程库` 


- **GPT-4.1 延迟波动极大**：一位用户报告称 **GPT-4.1** 的响应时间差异巨大，对于长度相似的消息，响应时间从不到一秒到超过 **10 秒**不等，这使得它对于需要顺序请求的工具来说变得不可靠。
   - 该用户还指出，在之前的模型（如 **GPT-4o**）中并未出现此问题。
- **Discord 机器人在使用 GPT-4.1 时遇到性能问题**：一位用户的 Discord 机器人使用 **GPT-4.1** 和 **GPT-4.1-mini** 处理管理员请求，但响应时间不一致，导致终端用户感觉到“卡死”。
   - 该用户确认该问题是通过 API 发生的，而不是通过浏览器的“搭便车”方式，并确认 Prompt 具有非常大的输入量。
- **建议重写系统提示词以解决性能问题**：一位成员建议重写系统提示词（System Prompt）使其更加严格，因为 **4.1** 的 Temperature 可能比 **4o** 更高，并建议尝试调低 Temperature 设置。
   - 该成员幽默地将降低 Temperature 比作使用“数字电牛棒”。
- **Operator 提供更高级别的安全标记**：一位用户报告称，当要求 **ChatGPT Operator** 根据图像查找内容时，它返回了一个通用错误，似乎是因为该用户未启用 2FA（双重身份验证）。
   - 另一位成员提到，与默认服务相比，**Operator** 具有更高的安全/标记级别，并且该服务需要 2FA 才能运行。
- **Mac 应用端无法使用 Operator**：成员们讨论了 Mac 应用中缺乏 **Operator** 集成的问题，一位用户对此表示失望，因为 Mac 应用中的快捷方式是他们购买 Pro 会员的关键原因。
   - 一位成员指出，在 OpenAI 路线图的大局中，这是“最后的优先级”，并建议通过编程来实现变通方案。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1394678581833236481)** (1 messages): 

> `Cross-Model Validation, Declarative Prompts, Zero Shot Prompting` 


- **新提示词方法通过跨模型验证**：一位成员创建/发现了一种全新的 **prompting method**，该方法在所有前沿 Models 上均能实现 zero shot，并附带了 [verifiable prompts](https://cdn.discordapp.com/attachments/1046317269069864970/1394678581761937450/message.txt?ex=6877af25&is=68765da5&hm=9040c1cb4df2db1445c2351cad60fe925b587bb3655c3bd0adf42521b2c1a5fc&)。
   - 该成员将其称为 **Cross-Model Validation of Declarative Prompts** 的案例研究。
- **图像分析集成**：消息中包含 **image analysis** 作为提示词评估的一部分。
   - 未提供关于图像分析的更多细节。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1394678581833236481)** (1 messages): 

> `Declarative Prompts, Cross-Model Validation, Zero-Shot Prompting` 


- **声明式提示词在多模型间得到验证**：一位成员声称创建了一种基于 **Declarative Prompts** 的新提示词方法，在所有前沿模型上均能实现 **zero shot**，并附带了 [message.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1394678581761937450/message.txt?ex=6877af25&is=68765da5&hm=9040c1cb4df2db1445c2351cad60fe925b587bb3655c3bd0adf42521b2c1a5fc)。
- **跨模型零样本**：这种新的提示词方法声称具有跨 **frontier models** 的 **zero shot** 能力，并使用了 **declarative prompts**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1394393253280743426)** (504 messages🔥🔥🔥): 

> `Grok 4 Performance, Kimi Model, OpenAI Model Retraining, New Models in Arena, Style Control Impact` 


- **Grok 4 在排行榜上表现强劲**：成员报告称 **Grok 4** 表现异常出色，在某些测试中甚至超越了 **GPT-4.1** 和 **Gemini 2.5 Flash**。
   - 然而，其他人指出 **Grok 4** 在 [LM Arena leaderboard](https://lmarena.ai/leaderboard/text) 上的表现可能无法反映其真实能力，有人暗示它正在被 *benchmaxed*。
- **Kimi K2 模型蒸馏自 Claude？**：新模型 `kimi-k2-0711-preview` 已添加到 Openrouter，其输出格式与 **Claude 3** 非常相似，引发了关于它可能蒸馏自 **Claude's models** 的猜测。
   - 一位用户指出他们无法在 **kimi-k2** 中上传图片，表明它纯粹是基于文本的；而另一位用户表示 *Kimi K2 在我使用的编程语言中感觉比 gpt 3 还差*。
- **OpenAI 模型面临重新训练故障**：据报道，由于严重的内部故障（被描述为 *比 MechaHitler 还糟*），一个 **OpenAI 开源模型** 需要重新训练。
   - 好消息是，[根据 Yuchen 的 X 帖子](https://x.com/yuchenj_uw/status/1944235634811379844)，有可以重新训练的 checkpoints，因此可能不需要完全重新训练。
- **中国模型涌入 LM Arena**：Arena 中出现了新模型：`ernie-x1-turbo-32k-preview`、`clownfish`、`nettle`、`octopus` 和 `cresylux`。
   - 这四个模型很可能是中国的。一位成员认为 Cresylux 是美团开发的，但大多数模型似乎比 R1 模型差。有人认为 Octopus 自称为 R1 模型。
- **LM Arena 出现故障**：用户报告了新版 LM Arena 界面的一系列问题，包括 **Cloudflare 错误**、无法使用的滚动条以及图标消失。
   - 最大的担忧是新界面创建了一个连续对话，每一轮都使用不同的模型，导致上下文泛滥。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1394393780349702296)** (430 messages🔥🔥🔥): 

> `Microsoft Extensions forks, New pricing, Grok 4 issues, Kimi K2, Kiro Features` 


- ****Microsoft Extensions 分叉引发法律辩论****：社区用户讨论了在 VS Code 中使用 Microsoft 扩展分叉的合法性，并对如果原始扩展被禁用可能导致的潜在违规行为表示担忧。
   - 一位成员分享了关于内部扩展的 [Cursor 论坛帖子链接](https://forum.cursor.com/t/new-in-house-extensions-c-c-ssh-devcontainers-wsl-python/94531)，强调 AnySphere 扩展是官方的，并由 Cursor 工程师重新构建。
- ****用户对新定价感到不满****：用户对 Cursor 的新定价模型表示担忧和困惑，一些人觉得现在更贵了，而另一些人则难以理解 20 美元的 Pro 计划如何与 API 成本挂钩。
   - 一位用户指出，你*每个月基本上能获得价值 20 美元的 tokens*，而其他人报告称虽然大大超过了该金额，但在一段时间后会被切断连接。
- ****Grok 4 仍然无法使用，Cursor 遭到抨击？****：用户报告 Cursor 中的 Grok 4 集成仍未按预期运行，一位用户称其*就像 Grok 3 一样只是炒作*。
   - 一些社区成员将 Grok 4 集成效果差归咎于 Cursor，其中一人暗示 *Elon Musk 将 Grok 4 集成问题归咎于 Cursor*。
- ****Kimi K2 热潮在 Cursor 用户中引起轰动****：社区成员对将 **Kimi K2** 集成到 Cursor 中表现出浓厚兴趣，强调其作为 **Sonnet 4** 更快、更便宜替代方案的潜力，并在 Agent 任务中具有 **Opus 级别** 的代码能力。
   - 一位用户表示 *我们希望在 auto 的基础模型中使用 Kimi K2*，另一位用户建议第一个添加它的 IDE 可能会胜出。
- ****Cursor 考虑借鉴 Kiro 的功能？****：用户讨论了 Cursor 采用 AWS Kiro 功能的可能性，其中一人表示 *我打赌 Kiro 会在 Cursor 添加 Kiro 功能之前先添加 Cursor 的功能*。
   - 他们强调了其通过偏好性（opinionated）设计改善用户习惯的潜力，其中一人补充道 *Kiro 真的很吸引我，它就像是我为了充分利用 tokens 应该做的事情，但我没做，因为那些功能不在视线范围内/没想起来，哈哈*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1394405921483915354)** (15 messages🔥): 

> `Background Agent Context Loss, Bugbot Organization Repo Visibility, Web Agent Opening Issues, Secrets in Background Agents, Background Agent Costs` 


- **Background Agent 丢失对话历史**：在进行一些代码更改后，后台 Agent 有时会丢失所有上下文并报告 *"我没有看到任何可以总结的先前对话"*，例如实例 `bc-c2ce85c9-bdc3-4b31-86e5-067ef699254d` 和 `bc-b2247bac-4a36-491f-a4a8-06d2446a9d76`。
- **GitHub 身份验证混乱**：一位用户报告称 Bugbot 可以看到他们的 Organization 仓库，但 cursor.com/agents 却看不到，通过断开并重新连接 GitHub 身份验证解决了该问题。
   - 目前尚不清楚是什么原因导致了仓库可见性的这种差异。
- **Web 端启动的 Agent 运行异常**：一位用户询问为什么从 Web 端启动的后台 Agent 不在 Cursor 中打开，这引发了对潜在集成问题的调查。
- **密钥管理（Secrets Management）困扰**：一位用户寻求一个集中的 GitHub issue 或资源来跟踪后台 Agent 密钥管理的更新，强调了文档模糊和已知问题。
- **解析 Agent 费用**：一位喜欢后台 Agent 功能的用户询问了后台 Agent 的平均每日/每周/每月支出，正在考虑更大范围地采用。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1394396471268216953)** (58 messages🔥🔥): 

> `更改下载目录, CrewAI 教程, Gemma 3 12b 视觉能力, 模型推荐, RX580 花屏问题` 


- **LM Studio 下载目录可轻松更改**：用户发现可以通过点击 LM Studio 中 **My Models** 选项卡路径旁的**三个点**来更改**下载目录**。
   - 这是一个非常直观的功能。
- **Gemma 的视觉能力受到质疑**：一位用户下载了据称具有 *vision*（视觉）能力的 **Gemma 3 12b 模型**，但模型回复称其**无法分析图像**。
   - 事实证明，用户必须直接提供一张图片，而不仅仅是询问它是否具备该能力。
- **出现花屏的 RX580 显卡——不要使用它**：一位用户询问是否可以将一张**出现花屏（artifacting）的 RX580 (20$)** 以某种方式组合起来，以运行像 **12B 甚至 18B** 这样的大型 AI 模型。
   - 回复简单直接：*千万别这么做*。
- **Vulkan 无法检测到集成显卡**：一位用户报告了一个在使用 **Vulkan** 时可能存在的 Bug，即如果同时安装了独立显卡，则**无法检测到集成显卡**。
   - 建议该用户在 **GitHub 上提交 Bug 报告**。
- **LLM 辅助精神分裂症患者**：一位患有精神分裂症的用户喜欢使用 LLM。
   - 他们正在利用 AI 的独特能力，帮助自己应对幻听带来的挑战。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1394404162669776958)** (49 messages🔥): 

> `本地模型 vs API 模型, LG EXAONE 4 许可协议, AMD395 迷你电脑, M4 MBP vs AI 395 平台, ROCm vs MLX 支持` 


- **本地模型争夺“足够好”的地位**：成员们讨论了**本地模型**达到可用状态的优点，并将其与依赖于控制公司的 **API 模型**进行了对比。
   - 一位用户指出，除非你使用模型来*赚大钱*，否则本地模型更有意义。
- **EXAONE 4 苛刻的许可协议引发不满**：**LG EXAONE 4** 模型的许可条款因过于严格而受到批评，特别是禁止商业使用以及要求在模型名称中保留 *EXAONE* 的规定。
   - 一位用户链接了一个 [llama.cpp issue](https://tenor.com/view/biggest-piece-of-dogshit-gif-26969373oh) 来表达对该许可的不满，并补充道：*LG 拥有该模型及其输出的所有权利——你只能将输出用于研究*。
- **AMD395 迷你电脑被视为可靠的工作站**：一位用户购买了一台配有 **128GB** RAM 的 **AMD395 迷你电脑**用于工作并计划进行测试，并提到电源适配器*可能比电脑本身还重两倍*。
   - 另一位用户对这台迷你电脑表示了兴趣，想知道其性能如何，并希望 Llama 4 能以不错的速度运行，但也感叹 **Llama 可能已经走投无路并即将闭源**。
- **为了 AI 395 放弃 M4 MBP？**：一位用户询问是否值得卖掉一台配有 **24 GB RAM** 的 **M4 MBP (M4 Pro CPU)** 来购买一台 **AI 395 平台笔记本电脑**，得到的回复是建议等待更好的 ROCm 支持。
   - 另一位用户建议坚持使用 **MBP M4 Pro**，或者寻找 **M1/M2 Ultra** 设备，并强调了内存带宽（memory bandwidth）的重要性。
- **ROCm 支持落后于 MLX**：一位用户指出 **MLX 支持**比 **ROCm** 好得多，使得更多模型可用，包括图像、diffusion 和 TTS 模型。
   - 该成员还表示：*我之前的错误在于忽视了内存带宽*。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1394740345761169630)** (1 messages): 

> `Hermes-3-Dataset` 


- **Hermes-3-Dataset 发布**：[NousResearch/Hermes-3-Dataset](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset) 已发布，如附带的截图所示。
   - 该数据集可以在 Hugging Face Datasets 页面找到。
- **附带截图**：公告中附带了一张与 Hermes-3-Dataset 相关的截图。
   - 截图直观地展示了数据集的关键方面或数据样本。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1394499879027347549)** (97 条消息🔥🔥): 

> `AI 训练 GPU、Meta 的开源转向、Windsurf 被裁撤、Kimi K2 热度、模型量化` 


- **讨论 AI 训练硬件选择**：成员们讨论了在本地训练 AI 模型的最佳硬件，对比了 **A100, A6000, A40, H100, RTX 6000 Ada, RTX 5090, 和 RTX 4090**，其中一位成员强调希望避免使用云系统并保持数据主权。
   - 一位成员建议在拥有 128+ PCIe 通道的家用服务器上使用配备 **b6000s** 的机器。
- **Meta 的开源承诺受到质疑**：讨论集中在 [Meta 的战略转变](https://www.youtube.com/watch?v=qh7QQ4Eov4M)，其可能放弃开源路线图以追求闭源利润。
   - 成员们表示担心大型科技公司正在垄断人才、初创公司和资源。
- **Windsurf IDE 遭遇毁灭性裁员**：一位成员对 Meta/Microsoft *裁撤* Windsurf IDE 表示惋惜，尽管存在细微错误，但仍称赞其工作流。
   - 他们将其与 *Cursor* 和 *Anthropic* 进行了对比，认为 **Windsurf** 是一个出色的开发效率倍增器。
- **Kimi K2 模型引发热议**：成员们对 [Kimi K2](https://www.youtube.com/watch?v=4bFDPVe6BHs) 感到兴奋，这是一个可与 **Claude 4.0** 媲美的开源模型，并讨论了其潜在影响，一位用户表示它正在经历其 *DeepSeek 时刻*。
   - 有人指出 [Kimi K2](https://x.com/intrstllrninja/status/1944983832777695277) 使用 **ChatML** token 进行工具调用，而不是 XML 标签，并使用额外的 `<|im_middle|>` token 来划分角色。
- **本地模型托管的量化技术**：围绕量化模型以在本地运行展开了讨论，一位用户表示有兴趣对 Kimi K2 基础模型进行量化。
   - 一位用户提到想要量化一个基础模型，因为 *基础模型很可爱*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1394695380683325664)** (1 条消息): 

> `在文本上微调多模态模型、Mistral 3、Gemma 3、Qwen 3、ForCausalLM` 


- **使用因果语言建模在文本上进行微调**：一位成员询问将 **Mistral 3**、**Gemma 3** 或 **Qwen 3** 等多模态模型加载为 `*ForCausalLM` 是否足以仅在文本上进行微调。
   - 未提供回复。
- **潜在问题与注意事项**：虽然加载为 `*ForCausalLM` 可能有效，但必须考虑模型的架构以及它是否针对纯文本任务进行了优化。
   - 确保在微调期间正确处理或禁用任何多模态特定的组件，以避免意外行为。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

ee.dd: https://arxiv.org/abs/2507.08794
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

ee.dd: https://arxiv.org/abs/2507.08794

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1394399690426159214)** (52 messages🔥): 

> `LLM architecture, Formal Languages and Neural Nets, R1-Zero, Voxtral Mini, hyperstition` 


- **研究员寻求语言学 LLM 合作伙伴**：一位具有分析哲学和数学背景的研究员正在寻找在 **LLM architectures** 方面有经验且对语言学感兴趣的合作伙伴，以开发能够真正理解自然语言的新型 **LLM architectures**。
   - 他们设计了一套语言语义的集合论模型和一套与语义模型相匹配的语用学算法模型，并希望通过计算方式实现它们。
- **Voxtral Mini 首次亮相，支持音频输入**：**Voxtral Mini** 是 **Ministral 3B** 的增强版，它集成了最先进的 **audio input** 能力，同时保留了同类最佳的文本性能，在语音转录、翻译和音频理解方面表现出色，详见 [HuggingFace](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)。
   - 一位成员希望它也能支持 **audio output**。
- **成员讨论“仅限帮助性”的推理模型**：一位成员询问是否存在比 **R1-Zero** 更小的 **helpful-only reasoning models**。
   - 另一位成员建议将 [ZR1-1.5B](https://huggingface.co/Zyphra/ZR1-1.5B) 作为候选，具体取决于模型需要推理的内容。
- **无脑废话创造了数百万加密货币价值**：一位成员评论了“无脑废话（mindless slop）”如何最终演变成一种 **hyperstition**，创造了一个拥有数百万加密资产的 Twitter 机器人。
   - 他们链接到了 [Truth Terminal](https://truthterminal.wiki/)（它揭示了 Anthropic 在进行 **red teaming** 时重新发现的 Opus 人格的一个有趣层面）以及 [AISafetyMemes](https://x.com/AISafetyMemes/status/1856873365325136367) 的一个搞笑迷因。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1394625891153346570)** (12 messages🔥): 

> `arXiv Endorsement Request, Image Captioning Framework, Relevance of Image Captioning Paper, Attention Mechanisms vs. Other Architectures, Encoder-Decoder Pipeline Analysis` 


- **本科生为图像字幕论文寻求 arXiv 背书**：一位软件工程师兼计算机科学专业大二本科生正在寻求 arXiv 背书，以发布他们的第一篇关于 **image captioning** 的论文，并分享了他们的背书请求链接 [点击此处](https://arxiv.org/auth/endorse?x=VC6HKI)。
- **成员发现论文内容陈旧**：成员们批评论文中使用的对比对象几乎是十年前的。
- **Attention Mechanisms 克服了瓶颈？**：作者分享了他们针对**经典 encoder-decoder 流水线**的发现，即在没有 Attention 的情况下，更强大的 Backbone（EfficientNetV2）由于信息瓶颈表现反而更差，这一点在[随附的 PDF 副本](https://cdn.discordapp.com/attachments/747850033994662000/1394652533514965022/Tell_Me_What_You_See.pdf?ex=687796e3&is=68764563&hm=70a620b97aa33a9759db181e2e3d53c18ce3d5898050629d93bcd7e52caeee5c&)中进行了讨论。
- **架构演进是否适合 arXiv？**：作者询问一篇定位为关于此类架构演进的“方法论案例研究”的论文是否适合发布在 arXiv 等预印本服务器上。
   - 他们阐明论文的重点在于展示为什么 Attention 在 **image captioning** 中变得至关重要。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1394501361219014677)** (22 messages🔥): 

> `Anthropic 的推理成本，确定性 ML vs. 随机性，扩散语言模型，K2 模型，Groq 的领先指标` 


- **Anthropic 的推理成本不可持续**：当前前沿模型推理成本的单位经济效益（尤其是 **Opus 4**）是不可持续的，这导致了一些行为引导，例如强制用户切换到 **Sonnet 4** 并禁用扩展思考。
   - 提出了两条前进道路：将工作转向确定性/经典 ML 方法，或者使用更多的 VC 资金并寄希望于下一次迭代能解决问题。
- **扩散语言模型缺乏进展**：虽然扩散语言模型的混合方法有一些令人印象深刻的演示，但在该方向上进展甚微。
   - 缺乏进展的原因尚不清楚，但一种理论是公司不愿在可能很快过时的未经验证的方法上投入重金。
- **K2 在工具调用方面表现出色**：**K2 模型**虽然不是推理模型，但在长上下文指令遵循和 *Agent 工具调用* 的实际表现方面收到了极好的反馈。
   - 这些反馈来自积极构建实际应用的个人。
- **Groq 是一个领先指标**：观察 **Groq** 的使用情况可能是一个很好的领先指标，因为 **tokens/sec 速度** 是最初吸引人们兴趣的原因。
   - 一旦某种水平的模型能力变得商品化，原始速度就会变得非常受欢迎。
- **MorphLLM 押注代码编辑的速度**：[Morph](https://docs.morphllm.com/) 是一家 YC 初创公司，正押注于 **代码编辑** 的高吞吐量。
   - 目前尚不清楚他们是否在内部使用了 **RLAIF/GRPO** 等技术或结合了扩散技术。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

burnytech: https://fxtwitter.com/HThasarathan/status/1944947772119245210
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1394484294939967580)** (5 messages): 

> ``get-answer` 过滤器，Regex 过滤器实现，过滤器流水线名称` 


- **`get-answer` 过滤器行为剖析**：一名成员询问了 `get-answer` 过滤器在 `boolean_expressions` 任务中是如何工作的，并指出尽管答案看起来正确但仍会出现解析失败，详见[此处](https://cdn.discordapp.com/attachments/755950983669874798/1394484294986240041/image.png?ex=6877a2f4&is=68765174&hm=e3a91c319a4d37ecfb85f7b7eed7dfa0aaf6d75b56414e060577ff252772622f)。
   - 另一名成员建议过滤器可能因为额外的文本而误解了答案，将 *'not False = True. So the answer is True.'* 视为答案本身。
- **Regex 过滤器流水线揭晓**：澄清了过滤器流水线需要名称，并使用在 `extraction.py` 中定义的 [regex 过滤器](https://github.com/EleutherAI/lm-evaluation-harness/blob/cf631de0993897dbc12bca3c453b9182ad4c2176/lm_eval/filters/extraction.py#L20-L21)。
   - `take_first` 过滤器被用作重复项的变通方案，并有一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/blob/cf631de0993897dbc12bca3c453b9182ad4c2176/lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml#L22-L23) 修复了此问题。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1394394506119614645)** (70 条消息🔥🔥): 

> `语音控制任务自动化的桌面应用、代码库组织策略、数据集端点问题、云提供商的 GPU 访问、Text-to-Text 模型标签` 


- **语音控制应用梦想主导桌面**：一名成员正在开发一款能够监听、查看屏幕并通过语音命令执行任务的桌面应用，例如打开 Slack、发送消息、管理代码以及与 Figma 和电子邮件进行交互。
   - 该成员询问了该应用在理解项目上下文以进行代码实现和设计交互方面的潜在帮助。
- **代码库混乱 vs. 组织天堂**：成员们讨论了在进行使用不同数据集的多次训练运行（training runs）时管理代码库的策略。
   - 一位成员幽默地承认：*"这就是秘密……我们根本不管理……xD"*，而另一位成员建议结合使用 `wandb.ai` 和镜像 `run` 名称的文件夹结构来进行可视化和记录笔记。
- **数据集下载厄运：端点错误出现**：成员们报告了数据集端点的问题，其中一人最初怀疑是用户错误，但后来确认为 **server error 500**。
   - 建议包括使用 `git clone` 来绕过浏览器相关的错误，尽管有人澄清数据集不在 Git 上，且用户仅下载了超大型数据集的一部分。
- **云端 GPU 探索：经济实惠的算力难题**：成员们寻求关于高性价比 GPU 解决方案的建议，探索了 **Open Router**、**Colab**、**Kaggle**、**HF**、**Intel** 和 **Groq** 等选项。
   - 建议范围从使用 **Colab Pro** 以更轻松地获取 GPU，到探索 **LAMBDA**、**RunPod** 和 **Lightning.ai** 等替代方案，并提到了 [Hugging Face 的算力产品](https://huggingface.co/posts/jeffboudier/479363753731415)。
- **Text-to-Text 标签陷入困境**：一位成员注意到 [Hugging Face 模型页面](https://huggingface.co/models?pipeline_tag=text2text-generation)上缺少 `text2text` 模型，并质疑其 pipeline 状态。
   - 一位成员解释说这可能是 *"遗留代码"* 以及替代的标签习惯，并建议为 HF 创建一个 <#1353741359059570699> issue 以改进 model cards，并链接到了一个[相关讨论](https://discuss.huggingface.co/t/no-0-models-returned-by-text2text-search-filter/161546/3)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1394396662088073297)** (4 条消息): 

> `4-bit 训练` 


- **4-bit 训练时间表推测**：成员们讨论了还需要多久才能实现**全 4-bit 训练**。
   - 共识是由于**损失太大**，目前*不值得*这样做。
- **对 4-bit 训练的担忧**：频道成员对在 AI 模型中实现**全 4-bit 训练**的可行性和实用性表示担忧。
   - 总体情绪倾向于怀疑，认为显著的性能下降和信息丢失是需要克服的主要障碍。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

geekboyboss: https://github.com/cactus-compute/cactus
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1394584623148695553)** (6 条消息): 

> `用于多候选问题的 Hypernetworks，PandasAI 和 Datatune，专注于数学的 LLM，BERT-Diffusion 架构` 


- **使用 **Hypernetworks** 低成本缝合 **Unimodal Models****：一篇新论文 ([arxiv.org/abs/2507.10015](https://www.arxiv.org/abs/2507.10015)) 表明，使用 **hypernetwork** 可以帮助以比 **grid search** 更低的成本缝合多个 **unimodal models**。
   - 作者正在征求关于使用 [Datakit](https://datakit.page) 处理 **HF datasets** 的潜在用例和缺失功能的反馈，该工具允许对数据集进行并排比较（见附带的 [屏幕录制](https://cdn.discordapp.com/attachments/897390720388825149/1394648354188824576/Screen_Recording_2025-07-15_at_13.48.02.mov?ex=687792fe&is=6876417e&hm=d678bd547bbb6e0f836699090e80df6fbff7477051e1ace47cc60bedbcc1d329&)）。
- ****PandasAI** 和 **Datatune** 跳过数据代码**：一篇新的博客文章 ([blog.datatune.ai](https://blog.datatune.ai/explore-and-transform-your-data-a-pandasai-datatune-tutorial?showSharer=true)) 展示了如何结合使用 **PandasAI** 和 **Datatune** 来跳过大量数据代码，直接深入了解数据洞察和转换！
   - 作者还提到他们已经微调了几个**专注于数学的 LLM**，并且一些有趣的新模型即将推出，欢迎在 [HuggingFace](https://huggingface.co/collections/entfane/math-professor-67fe8b8d3026f8abc49c05ba) 上提供反馈。
- **简单的 **BERT-Diffusion Architecture** 首次亮相**：一个基于 **BERT model** 的新型简单 **diffusion architecture** 已发布 ([github.com](https://github.com/Pr0fe5s0r/BERT-Diffusion))。
   - 作者构建了该 diffusion 架构，并表示它*易于理解*，*非常适合任何开始接触 text diffusion 的人*。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1394643072461307915)** (1 条消息): 

> `视频+图像理解，3D 像素表示，VQA 性能提升` 


- **征集关于视频+图像理解的论文**：一位成员征求目前在 **video+image understanding** 领域处于 **SOTA** 状态的优秀论文，特别是那些结合了 **3D representations of pixels (pixel tracking)** 的论文。
   - 他们引用了 [Depth-Aware Referencing Transformer for 3D Visual Grounding](https://arxiv.org/abs/2404.04319) 和 [3D-Aware Instance-Conditioned Implicit Fields for Single-View 3D Reconstruction](https://arxiv.org/abs/2412.02930v2) 作为例子。
- **关于 VQA 中 3D 与 2D 性能的查询**：该成员正在寻求关于在 **2D 图像理解任务 (VQA)** 中，拥有 **3D spatial representations** 是否比 **2D** 带来性能提升的信息。
   - 他们希望了解在 **Visual Question Answering** 等任务中引入 3D 信息的优势。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 条消息): 

dlp1843: opencv.org 的落地页之于 opencv，是否就像 bitcoin.com 之于 bitcoin？
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1394526437687427134)** (1 条消息): 

> `推理提供商，Qwen 模型，Llama/DeepSeek 模型` 


- **使用备选提供商进行推理**：一位成员建议，如果当前的推理提供商无法工作，使用 **不同的推理提供商** 可能会解决问题。
- **尝试备选模型**：他们补充说，尝试 **不同的 Qwen 模型，或者 Llama/DeepSeek 模型** 也是潜在的解决方案。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1394445279616438294)** (19 messages🔥): 

> `NCU profiling on VM GPU, Parallel Radix Sort on GPU, PMPP Radix Sort Tutorial` 


- **VM GPU 上的 NCU Profiling 障碍**：一位成员询问如何在 **VM GPU** 上进行 **NCU profiling**，另一位成员建议，如果没有 VM 之外的高级管理员权限，这可能无法实现。
   - 该成员指出，这需要*请求 VM 之外的管理员授予你访问权限*。
- **寻求并行 Radix Sort 指导**：一位成员征求在 **GPU** 上实现高性能**并行 Radix sort** 的建议。
   - 另一位成员建议参考《Programming Massively Parallel Processors》(**PMPP**) 书中的教程，并将其推广到不同的基数值，特别指向了第 3 版的 [第 13 章](https://www.amazon.ca/Programming-Massively-Parallel-Processors-Hands/dp/0128119861)，该章节演示了 **2bit radix**。
- **将 Radix Sort 扩展到 8-Bit 块**：针对 **PMPP** 教程，一位成员询问有关在 **Radix sort** 中处理 **8-bit chunks** 的资源。
   - 提供建议的成员表示，实现应该可以从 **2bit** 示例和对基础知识的理解中推导出来；此外，建议查看 **第 4 版**。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1394616089861689355)** (1 messages): 

> `Predicate Registers, SASS Compiler Optimization` 


- **谓词寄存器难题困扰程序员**：一位成员询问了 **SASS 层面**的**谓词寄存器 (predicate registers)**，特别是为什么编译器会重新计算谓词而不是复用它们。
   - 他们观察到了看似冗余的代码模式，如 `ISETP.NE.AND P0, PT, R2.reuse, RZ, PT; ISETP.NE.AND P1, PT, R2, RZ, PT`，并寻求对潜在架构细节的见解。
- **编译器奇怪的谓词重计算**：用户对 **SASS 编译器**重新计算谓词寄存器的倾向感到困惑，即使复用之前计算好的寄存器看起来是可行的。
   - 用户提供了一个代码片段作为示例，质疑自己是否遗漏了关于谓词工作方式的某些架构细节。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1394695933643460749)** (3 messages): 

> `torch.compile hanging, TORCH_COMPILE_DEBUG, coordinate_descent_tuning` 


- **`TORCH_COMPILE_DEBUG` 在缓存时可能挂起**：一位成员发现 `TORCH_COMPILE_DEBUG=1` 有时会在没有输出的情况下挂起，除非禁用缓存，这很令人烦恼，因为编译可能需要很长时间。
- **`coordinate_descent_tuning` 的挂起问题**：一位成员最近在使用 `coordinate_descent_tuning` 时遇到了 `torch.compile` 挂起的问题。
   - 用户提供了一个 [示例日志](https://example.log)，显示在 **PyTorch 2.8.0** 上发生挂起时，出现了与 *autotune_cache.py* 和 *coordinate_descent_tuner.py* 相关的消息。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1394764261007556738)** (2 messages): 

> `Parallel Radix Sort, Signed Integers` 


- **寻求并行 Radix Sort 帮助**：一位成员请求帮助实现适用于有符号整数的高性能并行 Radix sort。
   - 他们提到如果数值为负，则它是无效的。
- **Radix Sort 效率**：另一位成员建议考虑 Radix sort 在各种数据分布下的效率。
   - 他们提到，如果数据分布不均匀，Radix sort 的性能可能会显著下降。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1394678823609438228)** (4 messages): 

> `Job opportunities, WFH positions, Voltage Park Careers` 


- **Voltage Park 发布职位空缺**：Voltage Park 正在招聘从**软件工程 (Software Engineering)** 到**安全 (Security)** 的多个职位，详情发布在他们的 [招聘页面](https://www.voltagepark.com/careers)。
   - 除数据中心职位外，所有职位均为 **WFH**（远程办公）；优先考虑美国的申请人，但也提供一些全球技术支持职位。
- **寻求工作机会**：多位用户表达了在频道内寻找工作机会的兴趣。
   - 频道发布了提醒：此频道供雇主发布职位，求职者应向上滚动查找现有机会。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1394470816749588571)** (3 messages): 

> `Cloud GPUs, Vast.ai, nsight compute` 


- **适合新手的 Cloud GPUs**：一位用户询问了弥补 **GPU power** 不足的最佳/最便宜的方法，并表示 **Cloud GPU** 提供商的选择多得让人不知所措。
   - 他们寻求一种简单直接、预算友好的解决方案，这对其他初学者也会有帮助。
- **Vast.ai 提供实惠的 GPU**：一位成员推荐在学习阶段使用 **Vast.ai**，并强调了他们在各种 **GPUs** 上具有竞争力的价格。
- **nsight compute 学习资源**：一位成员询问除了官方文档之外，还有哪些学习使用 **nsight compute** 的途径。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1394456396514201781)** (1 messages): 

> `Kernel Tracing, HIP Tracing, HSA Tracing` 


- **Kernel Traces 应该是可进行线程追踪的**：如果一个 kernel 出现在 `--kernel-trace` 中，它就应该是 **thread trace-able**（可线程追踪）的。
   - 原则上，即使是 **HIP's built-in** 也可以被追踪（但 **HSA** 可能不行）。
- **默认排除重复的 Kernel 代码**：为了*避免数据量爆炸*，默认会排除多次重复的相同 kernel 代码。
   - 用户可以通过 `--kernel-iteration-range` 参数来控制数据量。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1394406417330475129)** (10 messages🔥): 

> `MTLReadWriteTextureTier2, WGPU rgba8unorm, dawn code, matrix-chat` 


- **MTLReadWriteTextureTier2 的访问依然难以实现**：一位用户正尝试将 **MTLReadWriteTextureTier2** 暴露给 **wgpu**，以便为 read_write 纹理获取 **rgba8unorm** 的访问权限，但尽管启用了 **Texture_Adapter_Specific_Format_Features**，仍未成功。
   - 该用户不确定为什么 WGPU 无法识别该支持，尽管 MetalAPI 的 Tier2 中已经提供了该支持。
- **建议检查 Dawn 代码以了解 MTLReadWriteTextureTier2**：一位成员建议检查 **Dawn** 的代码，理由是其开发非常活跃，可能有助于识别与 **MTLReadWriteTextureTier2** 相关的潜在 bug 或未记录的特性。
   - 这一建议暗示 Dawn 的实现可能会为如何正确暴露或利用所需功能提供见解。
- **引导至 WGPU Matrix 聊天频道寻求支持**：一位成员建议在 **wgpu 维护者的 Matrix 聊天频道**寻求帮助，并提供了 [WebGPU Matrix channel](https://matrix.to/#/#WebGPU:matrix.org) 的链接以获取直接支持。
   - 该建议强调了从 wgpu 开发社区获得针对性指导和故障排除的可能性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1394705852350599269)** (1 messages): 

> `AMD GPU, Containers, Fractional GPUs` 


- **发布针对容器的 AMD GPU 基准测试**：一位成员分享了一个[新基准测试的链接](https://dstack.ai/blog/benchmark-amd-containers-and-partitions/)，重点关注 **AMD GPU** 在 **containers** 和 **fractional GPUs** 上的性能。
   - 他们正在寻求关于该基准测试的反馈。
- **包含 Fractional GPUs 的 AMD 基准测试亮相**：一项新的基准测试专门测试了 **AMD GPU** 的性能，特别强调了在 **containers** 中的表现。
   - 测试研究了 **fractional GPUs** 在容器中的实用性，如 [这篇 dstack.ai 博客文章](https://dstack.ai/blog/benchmark-amd-containers-and-partitions/) 所示。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1394562851363885076)** (1 messages): 

> `A100, Leaderboard, trimul benchmark` 


- **A100 的 "trimul" 基准测试表现出色！**：一位成员在 **A100** 的 `trimul` 排行榜上获得了**第二名**，用时仅为 **12.8 ms**。
   - 获胜的提交 ID 为 **33244**，标志着在 GPU 性能方面取得的显著成就。
- **排行榜提交 ID 公布**：在 `trimul` 排行榜上获得第二名的 **A100** 运行提交 ID 为 **33244**。
   - 该 ID 方便对基准测试中使用的特定配置和代码进行跟踪和参考。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1394734475639783466)** (2 messages): 

> `Triton reference for grayscale, GPUMODE kernelbot data on HuggingFace` 


- **Triton 灰度图缺失参考实现**：成员们注意到缺乏用于 **grayscale**（灰度）转换的 **Triton 参考实现**。
   - 提供了指向现有示例的 [reference kernels](https://github.com/gpu-mode/reference-kernels/blob/main/problems/) 和 [historical submissions](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 的链接。
- **Kernelbot 数据仅针对 AMD 排行榜？**：一位成员询问 [HuggingFace 数据集](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 是否专门包含 **AMD 排行榜** 的提交内容。
   - 他们还询问其他排行榜的提交内容是否会很快发布。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1394586745466519592)** (4 messages): 

> `Phone Stolen, Inactivity, Back-up` 


- **用户报告手机被盗**：一位用户报告他们的 **手机在周日被盗**，导致近期处于不活跃状态。
   - 其他用户表示同情，并希望他们有 **backup**（备份）且能通过保险进行更换。
- **跟进讨论**：该用户提到在事故发生后，他们将回顾昨天会议中讨论的内容。
   - 未提供关于会议的其他具体细节。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1394466085390254231)** (2 messages): 

> `CuTeDSL, ARM Structure, CUTLASS, CUDA kernels` 


- **CuTeDSL 关注 ARM 架构**：一位成员询问 **CuTeDSL** 未来是否会支持 **ARM** 架构。
   - 另一位成员确认了这种可能性，并指出目前还没有可用的示例。
- **CUTLASS 32bit Kernel**：一位用户询问是否有任何 **CUDA kernels** 直接支持 **32bit CUTLASS**。
   - 另一位成员确认，是的，存在直接支持 **32bit CUTLASS** 的示例 **CUDA kernels**。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1394736476042039356)** (1 messages): 

> `Project Introductions, Giving Talks` 


- **建议进行项目介绍演讲**：一位成员建议另一位成员 *进行演讲并介绍该项目*。
- **鼓励项目推介**：一位用户提议另一位用户应该做一个演讲来介绍项目，引发了进一步讨论。


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1394759808326045799)** (1 messages): 

> `Torchtune project, Future of Torchtune, GitHub issue, Discord and Github support` 


- **Torchtune 的未来公布！**：一个 [GitHub issue](https://github.com/pytorch/torchtune/issues/2883) 被创建，其中包含关于 **Torchtune 项目** 未来的重要公告。
   - 公告对所有帮助 **torchtune** 成长的人表示感谢，并保证将继续在 **Discord** 和 **GitHub** 上提供支持。
- **Torchtune 团队预告令人兴奋的未来工作！**：**torchtune 团队** 不会离开，并计划很快分享更多即将开展的令人兴奋的工作。
   - 团队将继续在 **Discord** 和 **GitHub** 上回答任何问题或疑虑。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1394760855018803300)** (51 messages🔥): 

> `Torchtune 库组件用于新项目、RL 与微调器的未来、量子 SVM、俄亥俄州总部` 


- **Torchtune 的 BSD 3 许可证助力新项目**：成员们讨论了 **Torchtune BSD 3 许可证**的许可性，它允许用户提取并利用库组件用于其他项目，类似于 **Hugging Face** 在[其 trl 仓库](https://github.com/huggingface/trl/blob/640a9f39164ce3f9bbac7d325c1149ba023314e6/trl/models/activation_offloading.py#L19)中所采用的方法。
- **RL 成为微调器的未来**：讨论集中在 **Reinforcement Learning (RL)** 作为微调器未来的潜力，并期待“下一个大事件”的出现。
   - 成员们提出了 **Genetic Algorithms（遗传算法）、Bayesian Inference（贝叶斯推理）和 SVMs** 等想法，其中一位成员开玩笑地建议使用区块链或量子技术。
- **量子 SVM 在克利夫兰诊所食堂取得成功**：一位成员分享了他们使用位于克利夫兰诊所食堂的 **17-qbit 量子计算机**运行 **quantum SVM** 取得的成功，另一位成员调侃这证明了俄亥俄州是新的硅谷。
- **俄亥俄州作为 YC 孵化器？**：成员们开玩笑地讨论了在俄亥俄州建立孵化器的可能性，强调负担得起的住房是一个关键优势。
   - 然而，有人对俄亥俄州相比悉尼等地的吸引力表示担忧，一位成员打趣说要把自己的公寓变成一个厨房里放着**量子计算机**的孵化器，对此有人回应道：“你们那儿有蜘蛛风暴，我还是不去了”。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1394400083943887241)** (2 messages): 

> `优化器编译` 


- **选择退出优化器编译**：用户现在可以通过在配置中将 `compile.optimizer_step` 设置为 `false`，专门针对优化器步骤禁用编译。
   - 这允许对模型、损失函数和梯度缩放进行编译，同时跳过优化器，为性能调优提供了一种灵活的方法。
- **配置灵活性**：有选择地禁用优化器编译的能力为用户提供了对编译过程更大的控制权。
   - 通过针对训练循环的特定部分，开发者可以在优化性能的同时，避免优化器编译可能带来的问题。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1394417198105231550)** (25 messages🔥): 

> `ML 实验追踪器、Grafana 大日志解决方案、Claude 3 Sonnet、电路追踪、Meta 开源背叛` 


- **LLM 的推理反映了人类的易错性**：一位成员认为，对 **LLM 推理能力**的批评往往忽略了*人类也容易出现推理缺陷*这一事实。
   - 他们补充说，关于 LLM 缺点的言论听起来好像人们相信人类是完美的推理者，但事实并非如此。
- **实验追踪器应对训练浪潮**：成员们讨论了 **Weights & Biases**、**Optuna** 和 **TensorBoard** 等工具作为管理大型 ML 实验日志的解决方案，但一些人发现它们在超大规模训练中扩展性不佳。
   - 一位成员描述了将重要指标记录到 **W&B**，同时将其他指标本地保存到 **.csv** 文件并即时生成图表的方法。
- **DIY 日志存储：S3 简化流程**：一位成员分享了他们的计划：在生成日志时进行压缩，上传到 **S3**，并在 **Grafana** 中记录请求元数据。
   - 他们设想了一个基本的 **Streamlit** 前端，根据请求 ID 从 **S3** 获取并解压日志。
- **Anthropic 的电路追踪工具**：成员们分享了 **Anthropic 电路追踪工具 (circuit tracer tool)** 的链接，该工具可以可视化 Claude 3 Sonnet 等 AI 模型的内部运作机制。
   - 他们链接了 [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 和 [circuit-tracer](https://github.com/safety-research/circuit-tracer)。
- **Meta 背叛开源**：成员们对他们认为的 Meta 转向闭源的趋势表示哀叹，可能涉及 **Behemoth 模型**。
   - 一位评论者说“扎克背叛了我们”，另一位则暗示现在大多数主要的开源参与者都是中国实验室。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

.wavefunction: <@&1045297948034072678> ，今晚我没有讨论。
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1394432741923094608)** (3 messages): 

> `Yann LeCun, Signulll Sad Post` 


- **Signulll 帖子中分享的悲伤内容**：一位成员分享了一个来自 [Signulll 的悲伤帖子](https://x.com/signulll/status/1944851904888234293)。
- **推测 Yann LeCun 的立场**：一位成员推测了 **Yann LeCun** 可能对某种立场的辩护，尽管他之前曾主张相反的观点，并引用了[这段 YouTube 视频](https://youtu.be/zcwqTkbaZ0o)。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1394450414132527254)** (1 messages): 

> `Research Opportunity, User Interviews, Feedback` 


- **用户参与 NotebookLM 研究可获得报酬**：NotebookLM 用户有机会参加 **60 分钟的虚拟访谈**，交流他们的使用体验，对新想法提供反馈，并帮助确定改进服务的方法。
   - 被选中参加访谈的参与者将获得 **75 美元的感谢礼金**或等值的当地货币。
- **参与用户访谈**：NotebookLM 正在寻找用户参加 **60 分钟的虚拟访谈**，以提供反馈并讨论他们的体验。
   - 有兴趣的用户可以填写[此处](https://forms.gle/3XSDREyeMgj6CDeu9)提供的筛选表单。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1394431924507902076)** (5 messages): 

> `NotebookLM, Google Docs, News articles, Analysis notebook` 


- **用户将来源整合到 NotebookLM**：用户正在通过使用 **Google Docs 标签页**功能保存新闻文章，将其整合到单个 **NotebookLM source** 中，在同步到“分析”笔记本之前将其分类到标签和子标签中。
   - 用户希望有一种更简单的方法来复制新闻文章，而不需要手动去除广告和菜单选项等无关噪音。
- **分析笔记本结合了多种文本**：用户正在创建一个包含历史、硬科学、批判性思维和哲学文本的 **“分析”笔记本**，以便将新闻文章置于特定语境中。
   - 他们正在致力于分析新闻文章，希望从教科书、文章和研究论文中添加背景信息。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1394395230308143204)** (21 messages🔥): 

> `Notebook limits, Dynamic updates, Audio Overviews, Math/Latex rendering, Pro plan price reduction` 


- **Gemini Gems 与 NotebookLM sources 的动态更新差异**：用户发现 **NotebookLM** 和 **Gemini Gems** 处理 [Google Docs sources](https://www.reddit.com/r/notebooklm/comments/1kbum2h/is_there_a_way_to_have_sources_dynamically_update/) 更新的方式存在差异，这让他们感到奇怪。
   - **NLM** 会预处理你的 sources，在 sources 和 **Gemini** 之间创建一个层。
- **Math/Latex 渲染即将推出？**：用户询问 **NotebookLM** 是否已经支持 **math/latex 渲染**。
   - 一位成员回答说目前*尚未*提供，但*正在开发中*，并引用了[该公告](https://discord.com/channels/1124402182171672732/1182376564525113484/1394352702833688597)。
- **笔记本将支持横幅？**：成员们注意到精选笔记本拥有**自定义横幅照片**，而不是默认的表情符号风格封面，并想知道是否有办法自定义笔记本横幅。
   - 目前看来这似乎还不可能。
- **NotebookLM 的 Google Drive 集成？**：一位成员建议 **NotebookLM** 应该与 **Google Drive** 集成，允许用户选择文件夹/文件作为 sources。
   - 另一位成员则喜欢它是独立的。
- **强制查看精选笔记本引发愤怒**：用户对被*强制*查看*甚至不是他们自己的* **精选笔记本**表示沮丧。
   - 他们对无法选择将其从视图中移除感到愤慨。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1394421551956361370)** (15 messages🔥): 

> `多赛道注册、证书发放问题、证书声明表单` 


- **MOOC 将支持多赛道**：一名成员询问下次是否可以注册**多个赛道和证书**。
   - 工作人员回应称*他们会考虑这一点*。
- **证书失踪了！**：一名成员报告称在电子邮件（**mesilov.maxim@gmail.com**）中未收到证书，即使检查了垃圾邮件文件夹也没有。
   - 该成员随后报告说已经找到了。
- **声明表单才是真正的证书**：工作人员询问成员是否填写了**证书声明表单**，因为填写后应该会收到一封确认邮件。
   - 工作人员指出，他们*没有足够的人力来协助错过这些表单/截止日期的学生*。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1394422677984509993)** (15 messages🔥): 

> `MCP 服务器验证、开源 LLM 客户端、Anthropic Connectors 目录` 


- **MCP 服务器需要验证**：一名成员创建了一个*可流式传输的 HTTP MCP Server*，并就如何针对 **Claude** 进行验证和调试寻求建议，并指出服务器已连接但**未找到任何工具**。
   - 另一名成员建议使用 [MCP Tools Inspector](https://modelcontextprotocol.io/docs/tools/inspector) 来验证服务器，但原作者表示它通过了 Inspector 测试，但在 **Claude** 中仍然失败。
- **寻求开源 LLM 客户端**：出于隐私原因，一位开发者正在为自托管的 **LLM + MCP** 方案寻找带有 Web 界面的开源客户端。
   - 他们正在考虑使用 **Ollama** 作为 LLM，并寻求适合每天几十到几百个请求、具备冷启动能力的托管和扩展方案建议，一名成员推荐了 [Open WebUI](https://github.com/open-webui/open-webui)。
- **Anthropic Connectors 目录为 MCP 开启大门**：Anthropic 宣布了一个新的“connectors”目录（参见 [Claude Directory](https://claude.ai/directory)），向更广泛的受众开放了 **MCP** 世界，这可能会增加对 MCP 服务器的需求。
   - 有观点认为 **Anthropic** 正试图与 **Docker 的 MCP Toolkit** 竞争。
- **Lyle 寻找酷伙伴**：一位在 MCP 方面拥有七年经验的全栈工程师希望能找到一位可靠、诚实且酷的人一起工作。
   - 一名成员评论说，这种对友谊的开放态度在当今社会并不被视为非常“可接受”或“酷”。


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1394422929927700582)** (1 messages): 

> `LlamaIndex 阿姆斯特丹见面会、Discord 答疑时间、NotebookLlaMa - NotebookLM 克隆版、Context Engineering 技术、使用 LlamaIndex 和 Gemini 2.5 pro 构建研究 Agent` 


- **LlamaIndex 邀请你参加阿姆斯特丹活动和 Discord 交流**：**LlamaIndex** 宣布将于 7 月 31 日在[阿姆斯特丹举行见面会](https://lu.ma/vzwsj72w)，并于 8 月 5 日在 [Discord 举行答疑时间 (office hours)](https://lu.ma/wkrn4nbz)。
- **NotebookLlaMa (NotebookLM 克隆版) 走红**：可在 [GitHub 上访问](https://github.com/run-llama/notebookllama/tree/main)的 **NotebookLlaMa**（NotebookLM 的克隆版）已经获得了超过 **1k stars**。
- **Context Engineering 技术博客文章**：**LlamaIndex** 发布了一篇关于 [Context Engineering](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider) 及其技术的博客文章。
- **使用 LlamaIndex 和 Gemini 2.5 Pro 构建研究 Agent**：**LlamaIndex** 详细介绍了如何使用 **LlamaIndex** 和 **Gemini 2.5 Pro** [构建研究 Agent](https://ai.google.dev/gemini-api/docs/llama-index)。
- **使用 OpenTelemetry、Arize Phoenix 和 Langfuse 实现 Workflow 可观测性**：**LlamaIndex** 介绍了使用 [OpenTelemetry Part 1](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observability_pt1.ipynb) & [Part 2](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observability_pt2.ipynb)、[Arize Phoenix](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observablitiy_arize_phoenix.ipynb) 和 [Langfuse](https://github.com/run-llama/workflows-py/blob/main/examples/observability/workflows_observablitiy_langfuse.ipynb) 的 Workflow 可观测性示例。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1394401042787209319)** (4 条消息): 

> `Research Agent, Google Gemini 2.5 Pro, LlamaIndex workflows, Pydantic models, Snowflake partnership` 


- **LlamaIndex Agents 研究 Gemini**：一款由 **LlamaIndex workflows** 和 **Google Gemini 2.5 Pro** 驱动的新型研究 Agent 可以执行网页搜索、记录笔记并撰写报告 ([tweet](https://twitter.com/llama_index/status/1944841840479674741))。
- **Pydantic 助力生产流水线**：LlamaIndex agents 和 workflows 现在支持使用 **Pydantic models** 进行结构化输出，从而更轻松地将 Agent 结果集成到应用程序中 ([tweet](https://twitter.com/llama_index/status/1945155160415899829))。
   - 使用 Pydantic models 定义输出模式 (schemas) ([docs](https://t.co/N6idWoey8I))。
- **LlamaIndex 与 Snowflake 联手**：LlamaIndex 正与 **Snowflake** 合作，进行实战演讲并提供可立即实施的具体模式 ([tweet](https://twitter.com/llama_index/status/1945197684006232189))。
   - 参加他们于 **7 月 31 日在阿姆斯特丹举行的见面会**，了解团队如何在生产环境中构建高质量的数据 Agent ([meetup details](https://t.co/fFJvvIWrw4))。
- **UiPath Coded Agents 已部署**：通过 **UiPath** 新的 coded agents 支持，将 LlamaIndex agents 无缝部署到企业环境中 ([tweet](https://twitter.com/llama_index/status/1945226936642580493))。
   - 通过 UiPath 的 Python SDK 实现全代码级控制，构建自定义 Agent，从企业系统中提取数据，并使用嵌入式规则或 AI 模型做出决策 ([details](https://t.co/ILez3d6Zrs))。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1394528146413588520)** (8 条消息🔥): 

> `AI Agent Design, LlamaHub Tools, ML Logs storage, AI Showcase Virtual Conf` 


- **AI 自动化专家提供服务**：一位 AI 和自动化专家提议提供 **构建 AI Agents**、使用 **Make.com** 和 **n8n** 进行自动化、设计代理系统、连接 **CRMs, APIs, webhooks**、创建智能线索生成、CRM 和运营自动化、为销售/支持构建聊天机器人和语音 Agent、微调 **LLMs** 以及集成 **LSMs** 以处理上下文任务的服务。
- **请求在正确的频道进行对话**：一名成员请求对话应保持在正确的频道中，而不是在 general 频道发送垃圾信息。
- **LlamaIndex 工具已上线 LlamaHub**：当一名成员询问在哪里可以找到所有内置工具时，另一名成员分享了 [LlamaHub](https://llamahub.ai/)。
- **关于 ML 日志存储的讨论**：一名成员询问如何存储大型 **ML logs**（每个日志 4mb），因为 **Grafana** 已经无法处理它们了，以及标准的方法是什么。
- **本周四举行 AI Showcase 虚拟会议**：一名成员分享了将于本周四举行的 [AI Showcase Virtual Conf](https://inniches.com/ai?1336) 链接，这是一个汇聚了推动 **AI** 边界的构建者、创始人和梦想家的全球虚拟活动。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1394393741187350558)** (4 条消息): 

> `Reinforcement Learning for Model Search, Recursive setitem in tensor.py, Large Kernels` 


- **RL 导航模型搜索空间**：有人建议将 **强化学习 (RL)** 作为重点，以处理有用模型的巨大搜索空间。
   - 使用 **RL** 可能会带来新的融资、新模型和新工具。
- **Setitem 递归失控**：一名用户质疑 `tensor.py` 中的 `setitem` 是否应该是递归的。
   - 他们注意到 `setitem` 使用了 `_masked_setitem`，后者随后在 `functools_reduce` 上调用 `getitem`，导致持续的 `getitem` 调用。
- **极小输入，极大算子 (Kernel)**：递归的 `setitem` 过程即使在很小的输入尺寸下也会创建相当大的 Kernel。
   - 该用户提供了一段代码片段，通过使用 `Tensor.zeros` 和 `Tensor.arange` 在张量中设置值来重现该问题。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1394440077106614416)** (6 messages): 

> `Tinygrad 中的内存分配开销，GlobalCounters.global_mem 与 GlobalCounters.mem_used，Tinygrad 中的 Subbuffers 和内存管理` 


- **Tiny 内存之谜：全局 vs. 模型参数**：一位用户观察到 `GlobalCounters.global_mem` 分配的内存 (**0.06 GB**) 超过了模型参数本身，并询问了这种开销的来源。
   - 他们怀疑这可能是由于嵌套的 uops 和 tinygrad 栈的复杂性造成的，但重置全局内存并未解决这一差异。
- **Tinygrad 计数器澄清：`global_mem` != `mem_used`**：一位成员指出 `GlobalCounters.global_mem` 追踪的是访问过的全局内存，这可能大于权重大小，并建议改用 `mem_used`。
   - 切换到 `GlobalCounters.mem_used` 后，内存使用量与参数大小匹配 (**0.01GB**)，这引发了关于这两个计数器之间差异的进一步讨论。
- **Subbuffer 的奥秘：揭开 `global_mem` 的面纱**：讨论表明 `GlobalCounters.global_mem` 和 `GlobalCounters.mem_used` 之间的差异可能是由于 NV 或 CPU 等设备上的 **subbuffers** 造成的，这些设备使用具有最小尺寸限制的较大缓冲区。
   - 建议使用 **WEBGPU** 进行测试以检查 `mem_used` 的差异，并暗示 `global_mem` 追踪的是访问过的全局内存。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1394400448600867018)** (9 messages🔥): 

> `Manus Fellowship, 诈骗警报, 自动化可持续发展, ESG 研究工作流, Manus 高级功能` 


- **Manus Fellowship 状态受关注**：一位成员询问频道中是否有人加入了 **Manus Fellowship 计划**。
   - 此外，一位成员请求移除一名用户，并将其标记为*诈骗者 (scammer)*。
- **寻求自动化的 ESG 研究工作流**：一位用户正在测试 **Manus** 以实现 **sustainability (可持续发展)** 和 **ESG 相关研究工作流**的自动化，并对其 **UX** 表示赞赏。
   - 该用户正在寻找一个 **API 端点**（公开或私有），以便通过编程方式发送 prompt、触发研究任务并自动检索结果，旨在利用 Python 或 n8n 集成到自动化工作流中。
- **高级功能推出引发抽奖热议**：一位成员询问了 **Manus 高级功能**的部署情况，另一位成员询问是否有一个 **$2500 的抽奖活动**。
   - 一位成员确认该*抽奖*信息也被发送到了另一个服务器。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1394435401485254802)** (3 messages): 

> `Discord bug, @kap 命令` 


- **发现 Discord Bug！**：一位用户指出了频道中的一个 **Discord bug**，即输入用户的全名无法按预期工作。
- **使用 @kap 命令！**：成员们建议使用 **`@kap` 命令**从弹出列表中选择用户，作为该 Discord bug 的临时解决方案。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1394428034475495625)** (4 messages): 

> `mojo @parameter 装饰器, capturing 关键字, GitHub issue 5020` 


- **@parameter 装饰器揭秘**：一位成员在源代码和文档中寻找关于 **@parameter** 装饰器的信息但未找到，但 VS Code 扩展中的 LSP 将其识别为关键字 ([image.png](https://cdn.discordapp.com/attachments/1151418092052815884/1394428034240741376/image.png?ex=6878174e&is=6876c5ce&hm=5100120f8a786af783dd14c9d26eab4aad5c114f2716a606066f88da1db7ae5e&))。
   - 另一位成员提供了指向 [Modular 文档](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)的链接以及与该主题相关的 [GitHub issue](https://github.com/modular/modular/issues/5020)。
- **'capturing' 关键字仍是个谜**：一位成员询问 **'capturing'** 关键字是否可以更广泛地用于创建闭包 (closures)（在编译时之外）。
   - 提供的解释似乎与编译时装饰器本身有关，但引发了关于更广泛适用性的疑问。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1394471216966012928)** (4 messages): 

> `AWS Prompt Optimizer, Nova Models, MIPRO Usage, Enterprise DSPy Wrappers` 


- **AWS Prompt Optimizer “灵感”源自 DSPy？**：一位用户发现了针对其 **nova models** 的 [AWS prompt optimizer](https://x.com/AWSAI/status/1794862993470116345)，并指出它似乎受到了 **DSPy** 的启发。
   - 另一位成员暗示它甚至直接使用了 **MIPRO**，并推测企业级 **DSPy wrappers** 正在兴起。
- **对上游贡献的期望**：一位成员表示希望 **AWS** 最终能向 **DSPy** 进行上游贡献。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1394737613985808475)** (2 messages): 

> `GPT4ALL and Raspberry Pi 5, Dataset download error` 


- **在 Raspberry Pi 5 上运行 GPT4ALL？**：一位成员询问了关于使用 **GPT4ALL** 和 **Raspberry Pi 5** 构建小型便携式系统的建议。
- **数据集下载被拒绝！**：一位成员报告在尝试使用 **aws s3** 下载数据集以微调模型时出现 *Access denied* 错误。
   - 使用的命令是：`aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`。