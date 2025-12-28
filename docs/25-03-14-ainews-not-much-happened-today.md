---
companies:
- google-deepmind
- cohere
- meta-ai-fair
- alibaba
- hugging-face
date: '2025-03-14T22:57:23.512875Z'
description: '**Google DeepMind** 宣布了 **Gemini 2.0** 的更新，包括升级后的 **Flash Thinking 模型**，该模型具备更强的推理能力和原生图像生成能力。**Cohere**
  推出了 **Command A**，这是一个拥有 **111B**（1110 亿）参数的稠密模型，具有 **256K 上下文窗口**和极具竞争力的定价，已在 **Hugging
  Face** 上线。**Meta AI** 提出了 **Dynamic Tanh (DyT)** 作为 Transformer 中归一化层的替代方案，并得到了
  **Yann LeCun** 的支持。**阿里巴巴**发布了 **QwQ-32B**，这是一个拥有 **32.5B** 参数的模型，在数学和编程领域表现优异，通过强化学习进行了微调，并根据
  **Apache 2.0 协议**开源。**Google DeepMind** 还发布了 **Gemma 3** 系列模型，参数范围从 **1B 到 27B**，支持
  **128K token 上下文窗口**和 **140 多种语言**，此外还推出了图像安全检查工具 **ShieldGemma 2**。基准测试显示，**Gemma
  3 27B** 具有强大的视觉能力和内存效率，但在性能上仍逊于 **Llama 3.3 70B** 和 **DeepSeek V3 671B** 等更大规模的模型。**@_lewtun**
  分享了 **Hugging Face LLM 排行榜**的历史记录。'
id: 080524db-34b6-4c7d-b8e3-fa03f4e2d105
models:
- gemini-2.0-flash-thinking
- command-a
- qwq-32b
- gemma-3-27b
- gemma-3
- shieldgemma-2
- llama-3-70b
- deepseek-r1
- o1-mini
- deepseek-v3
original_slug: ainews-not-much-happened-today-7693
people:
- yann-lecun
title: 今天没什么事发生。
topics:
- model-updates
- model-performance
- benchmarking
- reinforcement-learning
- transformers
- normalization-layers
- image-generation
- vision
- memory-efficiency
- context-windows
- fine-tuning
---

<!-- buttondown-editor-mode: plaintext -->**一个宁静的周五**

> 2025年3月14日至3月15日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**222** 个频道，**2399** 条消息）。预计节省阅读时间（按 200wpm 计算）：**240 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

祝 [GPT-4](https://en.wikipedia.org/wiki/GPT-4) 和 [Claude 1](https://www.anthropic.com/news/introducing-claude) 两周岁生日快乐。很少有人能预料到[过去一年中发生的巨大市场份额变化](https://www.latent.space/p/2024-startups)。


![image.png](https://assets.buttondown.email/images/f5e3c589-7d40-4495-ad42-3b519c21606b.png?w=960&fit=max)


---

**特别说明**：我们今天发布了 [2025 年 AI 工程现状调查](https://www.surveymonkey.com/summary/NU9euNHK_2FMmqZLGjDImPimHFO_2FbIYG7s_2Bme46v_2BeQSA_3D?ut_source=lihp)，为 6 月 3 日至 5 日举行的 AI Eng World's Fair 做准备。**请[填写调查问卷](https://www.surveymonkey.com/summary/NU9euNHK_2FMmqZLGjDImPimHFO_2FbIYG7s_2Bme46v_2BeQSA_3D?ut_source=lihp)以表达您的声音！**

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**语言模型与模型更新**

- **Google 的 Gemini 2.0 更新与功能**：[@jack_w_rae](https://twitter.com/jack_w_rae/status/1900401734046126274) 宣布，由于产品开发以及底层模型从 **1.5 Pro 更新至 2.0 Flash Thinking**，**Google Deep Research** 得到了改进。**Gemini 应用**正在推出多项改进，包括具有**更强推理能力的升级版 Flash Thinking 模型**、更深层次的应用集成、**Deep Research** 以及**个性化**功能 [@jack_w_rae](https://twitter.com/jack_w_rae/status/1900325293447061877)。此外，[@jack_w_rae](https://twitter.com/jack_w_rae/status/1900334465945395242) 指出团队在为 **Gemini 2** 创建**原生图像生成**方面取得了进展，强调了其与 text-to-image 模型的区别。
- **Cohere 的 Command A 模型**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1900606602501341518) 报道称，**Cohere** 推出了 **Command A**，这是一个拥有 **111B** 参数的稠密模型，其 **Artificial Analysis Intelligence Index 为 40**，接近 **OpenAI 最新的 GPT-4o**。该模型拥有 **256K 上下文窗口**，速度为 **185 tokens/s**，定价为**每百万输入/输出 token 2.5/10 美元**。它可在 **Hugging Face** 上用于研究，并可通过 **Cohere** 的许可进行商业使用。
- **Meta 的 Dynamic Tanh (DyT)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1900528108140372411) 报道称，**Meta AI** 提议使用 **Dynamic Tanh (DyT)** 替代 Transformer 中的归一化层，其效果相当或更好，且无需额外的计算或调优，适用于图像、语言、监督学习和自监督学习。**Yann LeCun** 也在 [Twitter](https://twitter.com/ylecun/status/1900610590315249833) 上宣布了同样的消息。
- **阿里巴巴的 QwQ-32B**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900351166086537659) 重点介绍了**阿里巴巴的 QwQ-32B**，这是一个拥有 **325 亿参数**的语言模型，在数学、编程和问题解决方面表现出色。通过强化学习进行微调，它可与 **DeepSeek-R1** 等大型模型相媲美，并在基准测试中超越了 **OpenAI 的 o1-mini**。该模型在 **Apache 2.0 许可**下免费提供。
- **Google 的 Gemma 3 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549631647367268) 宣布发布 **Gemma 3**，提供从 **1B 到 27B** 的多种尺寸，具有 **128K token 上下文窗口**，支持超过 **140 种语言** [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549635267014878)。还宣布了 **ShieldGemma 2**，这是一个基于 **Gemma 3 基础**构建的 **4B 图像安全检查器** [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549638802813312)。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1900579291404046696) 对 **Gemma 3 27B** 进行了基准测试，其 **Artificial Analysis Intelligence Index 为 38**，指出其优势包括宽松的商业许可、视觉能力和内存效率，但在竞争力上不如 **Llama 3.3 70B** 或 **DeepSeek V3 (671B)** 等大型模型。[@sirbayes](https://twitter.com/sirbayes/status/1900520172059815986) 指出，**Gemma 3** 是能在 **1 张 GPU** 上运行的最佳 **VLM**。

**模型性能与基准测试**

- **排行榜的背景与历史**：@_lewtun 分享了 **Hugging Face LLM 排行榜** 的起源故事，重点介绍了 [@edwardbeeching](https://twitter.com/_lewtun/status/1900557190722687440)、[@AiEleuther](https://twitter.com/_lewtun/status/1900557190722687440)、[@Thom_Wolf](https://twitter.com/_lewtun/status/1900557190722687440)、[@ThomasSimonini](https://twitter.com/_lewtun/status/1900557190722687440)、[@natolambert](https://twitter.com/_lewtun/status/1900557190722687440)、[@abidlabs](https://twitter.com/_lewtun/status/1900557190722687440) 和 [@clefourrier](https://twitter.com/_lewtun/status/1900557190722687440) 的贡献。该帖子强调了**小团队**、**早期发布**以及**社区参与**的影响。[@clefourrier](https://twitter.com/clefourrier/status/1900572125238378939) 对此进行了补充，指出在排行榜公开时，[@nathanhabib1011](https://twitter.com/clefourrier/status/1900572125238378939) 和他们正在开发一套内部评估套件，这促成了代码的工业化。
- **GPU 基准测试与 CPU 开销**：[@dylan522p](https://twitter.com/dylan522p/status/1900379633662779781) 表达了对测量 **CPU 开销** 的 **GPU 基准测试**（如 **vLLM** 和 **KernelBench**）的赞赏。
- **井字游戏作为基准测试**：[@scaling01](https://twitter.com/scaling01/status/1900333236565221400) 表示在 GPT-5 发布之前他都对 LLM 持悲观态度，理由是 **GPT-4.5** 和 **o1** 甚至无法稳定地玩 **井字游戏 (tic-tac-toe)**；[@scaling01](https://twitter.com/scaling01/status/1900352006641848695) 认为，如果 LLM 在看过数百万场比赛后仍无法玩好 **井字游戏**，那么在研究或业务任务中就不应该信任它们。
- **推理模型的评估脚本**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1900595120053047452) 宣布了一个 **GitHub 仓库**，提供用于测试推理模型基准性能并复现 **QwQ** 报告结果的评估脚本。

**AI 应用与工具**

- **AI 辅助编程与原型设计**：[@NandoDF](https://twitter.com/NandoDF/status/1900548832733069638) 支持现在是学习编程的大好时机的观点，因为 **AI copilot** 降低了编程门槛，可能引发一波创业浪潮。[@AndrewYNg](https://twitter.com/DeepLearningAI/status/1900593192497520842) 也表达了同样的看法，指出 AI 和 AI 辅助编程降低了原型设计的成本。
- **IDE 中的 Agentic AI**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1900321016385359958) 介绍了 **Qodo Gen 1.0**，这是由 @QodoAI 开发的一款 **IDE 插件**，它将 Agentic AI 嵌入到 **JetBrains** 和 **VS Code** 中，使用了 **LangChain 的 LangGraph** 和 **Anthropic 的 MCP**。
- **Gemini 2.0 与 OpenAI Agents SDK 的集成**：[@_philschmid](https://twitter.com/_philschmid/status/1900589029961109514) 宣布只需更改一行代码即可在 **OpenAI Agents SDK** 中使用 **Gemini 2.0**。
- **LangChain 的长期 Agentic Memory 课程**：[@LangChainAI](https://twitter.com/LangChainAI/status/1900588929772122383) 和 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900562773303554110) 宣布了一门关于 **使用 LangGraph 构建长期 Agentic Memory** 的新 **DeepLearningAI 课程**，由 [@hwchase17](https://twitter.com/LangChainAI/status/1900588929772122383) 和 [@AndrewYNg](https://twitter.com/LangChainAI/status/1900588929772122383) 授课，重点是构建具有语义、情节和程序记忆的 Agent，以创建一个个人电子邮件助手。
- **UnslothAI 更新**：[@danielhanchen](https://twitter.com/danielhanchen/status/1900592202621087944) 分享了 **UnslothAI** 的更新，包括支持全量微调 + 8bit，支持几乎所有模型（如 **Mixtral**、**Cohere**、**Granite**、**Gemma 3**），视觉微调不再出现 OOM（显存溢出），进一步降低 VRAM 占用，4-bit 速度提升，支持 Windows 等。
- **Windows 上的 Perplexity AI**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1900371155753853427) 宣布 **Perplexity 应用** 现已在 **Windows** 和 **Microsoft App Store** 上架，语音对语音模式即将推出。
- **TestFlight 上的 HuggingSnap**：[@mervenoyann](https://twitter.com/mervenoyann/status/1900492593810546774) 宣布 **HuggingSnap**（由 [@pcuenq](https://twitter.com/mervenoyann/status/1900492593810546774) 和 [@cyrilzakka](https://twitter.com/mervenoyann/status/1900492593810546774) 构建的手机端离线视觉 LM）已在 **TestFlight** 上线，正寻求反馈以进行进一步开发。
- **机器翻译新趋势**：[@_akhaliq](https://twitter.com/_akhaliq/status/1900402426886115362) 重点介绍了一篇关于 **大推理模型在现代机器翻译中的新趋势** 的论文。
- **微软与 Shopify**：[@MParakhin](https://twitter.com/MParakhin/status/1900614024116740309) 宣布 **Shopify** 已收购 **VantageAI**。

**AI 与硬件**

- **AMD 的 Radeon GPU 在 Windows 上的支持**：[@dylan522p](https://twitter.com/dylan522p/status/1900352609271300572) 报道了 **AMD 的 @AnushElangovan** 在 **ROCm 用户见面会**上讨论将 **Radeon GPU** 打造为 **Windows** 上的“一等公民”，支持多种 **GPU 架构**，并专注于 **CI 和持续交付**。
- **MLX LM 的新家**：[@awnihannun](https://twitter.com/awnihannun/status/1900311865026372032) 宣布 **MLX LM** 有了新家。

**AI 会议与活动**

- **AI Dev 25 大会**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1900594063516254299) 在旧金山启动了 **AI Dev 25**，并指出 **Agent** 是 AI 开发者最兴奋的话题。大会包括来自 **Google 的 Bill Jia** [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900596396140671194)、**Meta 的 Chaya Nayak** [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900599467822510154) 的演讲，以及关于 2025 年构建 AI 应用的小组讨论 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900610468747899142)。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900636957740323302) 分享了来自 **Nebius 的 Roman Chernin** 的观点，强调解决现实世界的问题；[@AndrewYNg](https://twitter.com/AndrewYNg/status/1900617330906067136) 强调了来自 **Replit 的 @mattppal** 关于通过理解 **LLM** 的上下文进行调试的技巧。
- **GTC 炉边对话**：[@ylecun](https://twitter.com/ylecun/status/1900298938764202154) 宣布下周二将在 **GTC** 与 **Nvidia 首席科学家 Bill Dally** 进行炉边对话。
- **Interrupt 大会**：[@LangChainAI](https://twitter.com/LangChainAI/status/1900621522475381145) 宣传了 **Interrupt 大会**，并列出了其赞助商，包括 **CiscoCX**、**TryArcade**、**Box** 等 [@LangChainAI](https://twitter.com/LangChainAI/status/1900621520508219473)。
- **智利圣地亚哥的 Khipu AI**：[@sirbayes](https://twitter.com/sirbayes/status/1900294930599121068) 在智利圣地亚哥的 **@Khipu_AI** 上分享了关于使用在线变分贝叶斯（online variational bayes）进行**序列决策（Sequential decision making）**的演讲。[@sarahookr](https://twitter.com/sarahookr/status/1900390025293942829) 提到博物馆非常好奇为什么他们最想看的物品是 **khipu**（奇普）。

**其他**

- **开源模型的价值**：[@Teknium1](https://twitter.com/Teknium1/status/1900514887413227654) 表达了担忧，认为对美国人禁用中国模型不会减缓其进展，而无法接触到全方位的模型将使美国落后。
- **AI 与电影制作**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1900610357602750619) 讨论了 **AI 视频生成**的发散特性，它允许创意冲动和对意外时刻的探索，不受物理限制的约束。
- **软件的未来**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1900540465545372102) 推测了主要上市软件公司的未来，认为专注于功能和复杂界面的公司面临风险，因为新的软件栈是意图驱动（intention-driven）的。
- **团队规模**：[@scottastevenson](https://twitter.com/scottastevenson/status/1900357184191390184) 提出小团队正在获胜，固守旧的大团队文化可能会损害你的职业生涯。

**幽默/迷因**

- **“一切皆 Transformer”**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1900376022639665640) 简单地陈述了 “everything is transformer”，并配了一张 Transformer 的图片。
- **“Midjourney 最顶尖的技术是域名无法解析”**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1900618951627075710) 开玩笑说 **Midjourney** 最顶尖的技术是 “Domain Not Resolving”，并正在寻找在该领域有至少 6 年经验的人。
- **“一百万家初创公司必须消亡”**：[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1900625829396443447) 说道 “one million startups must perish”。
- **“我将在 PlayStation 2 上对人类基因组进行氛围编辑（vibe edit）”**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1900625627302023532) 发布了 “I will vibe edit human genome on a PlayStation 2”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Gemma 3 微调革命：Unsloth 中的性能与效率**

- **Gemma 3 微调现已支持 Unsloth - 速度提升 1.6 倍，显存占用减少 60%** ([Score: 172, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1jba8c1/gemma_3_finetuning_now_in_unsloth_16x_faster_with/)): **Unsloth** 现在支持 **Gemma 3 (12B)** 的微调，与 Hugging Face + FA2 相比，**性能提升 1.6 倍**，**显存占用减少 60%**，使得 **27B** 等模型能够适配 **24GB 显存的 GPU**。该平台修复了旧款 GPU 上的**无限梯度爆炸（infinite exploding gradients）**和**双 BOS token** 等问题，并支持广泛的模型和算法，包括**全量微调（full fine-tuning）**和**动态 4-bit 量化（Dynamic 4-bit quantization）**。更多详情请访问其 [blog](https://unsloth.ai/blog/gemma3) 并通过其 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) 进行免费微调。
  - 用户对 **Unsloth** 的进展表示热烈欢迎，特别是对**全量微调**的支持以及 **8-bit 微调**的潜力。**Danielhanchen** 确认所有方法（包括 4-bit、8-bit 和全量微调）都将得到优先处理，并提到可能加入 **torchao** 以支持 **float8**。
  - 用户对更友好的界面感兴趣，有人请求提供本地运行的 **webUI** 以简化使用。**Few_Painter_5588** 预测 Unsloth 将成为 **LLM 微调**的主要工具集。
  - **FullDeer9001** 分享了在 **Radeon XTX** 上运行 **8k 上下文** **Gemma3** 的正面反馈，强调了显存占用和 prompt 统计数据，并认为其表现优于 **Deepseek R1**。用户讨论了为 **16GB RAM** 优化 **12B 模型**以提升性能的想法。


**主题 2. Sesame CSM 1B 声音克隆：预期与现实**

- **[Sesame CSM 1B 声音克隆](https://github.com/isaiahbjork/csm-voice-cloning)** ([Score: 216, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1jaxec3/sesame_csm_1b_voice_cloning/)): **Sesame CSM 1B** 是一款新发布的**声音克隆模型**。帖子中未提供更多细节。
  - **声音克隆模型的许可与使用**：讨论涉及 **Sesame**（Apache 许可）与 **F5**（Creative Commons Attribution Non Commercial 4.0）之间的许可差异，强调 **Sesame** 可用于商业用途。用户还提到将声音克隆集成到对话式语音模型（CSM）中是一项潜在的进步。
  - **性能与兼容性问题**：用户报告该声音克隆模型性能较慢，在 GPU 上处理一整段文字需要长达 **50 秒**，并指出其可能未针对 Windows 进行优化。有建议称其在 Linux 上可能运行得更好，由于 CPU 的“实验性”triton 后端，在没有独立 GPU 的迷你 PC 上运行可能会面临挑战。
  - **技术调整与 API 访问**：**Chromix_** 分享了通过升级到 **torch 2.6** 及其他包使模型在 Windows 上运行的步骤，并提到通过从镜像仓库下载文件来绕过对 **Hugging Face 账号**的需求。他们还提供了[声音克隆 API 端点](https://github.com/SesameAILabs/csm/issues/61#issuecomment-2724204772)的链接。

- **结论：Sesame 向我们展示了一个 CSM。然后 Sesame 宣布将发布……某些东西。接着 Sesame 发布了一个 TTS，但他们显然具有误导性且虚假地将其称为 CSM。我理解得对吗？** ([Score: 154, Comments: 51](https://reddit.com/r/LocalLLaMA/comments/1jb1sgv/conclusion_sesame_has_shown_us_a_csm_then_sesame/))：**Sesame 的争议** 围绕其误导性的营销策略展开，他们宣布了一个 **CSM**，但实际发布的却是 **TTS**，并错误地将其标记为 CSM。如果 Sesame 明确沟通该产品不会 **open source**，这个问题本可以得到缓解。
  - **误导性营销策略**：许多用户对 Sesame 的营销手段表示失望，指出该公司通过暗示 **open-source** 发布来制造巨大的热度，结果却交付了一个平庸的产品。**VC-backed** 公司经常使用此类策略来衡量产品市场匹配度并吸引投资者兴趣，正如 **Sesame 的领投方 a16z** 所表现的那样。
  - **技术挑战与模型性能**：大家一致认为发布的 **1B model** 性能不尽如人意，尤其是在实时应用中。用户讨论了技术细节，例如 **Mimi tokenizer** 和模型的架构，这些因素导致其速度缓慢，并建议使用 **CUDA graphs** 或 **exllamav2** 等替代模型进行优化以获得更好的性能。
  - **不完整的产品发布**：讨论强调 Sesame 的发布缺少 demo 流程中的关键组件，如 **LLM、STT 和 VAD**，迫使用户自行构建这些部分。Demo 令人印象深刻的性能与实际发布的版本形成鲜明对比，引发了人们对 demo 设置可能使用了更大的模型或更强大的硬件（如 **8xH100 nodes**）的质疑。


**主题 3. QwQ 的崛起：统治基准测试并超越预期**

- **[QwQ 在 LiveBench 上的更新 - 优于 DeepSeek R1！](https://i.redd.it/sb78tt607joe1.png)** ([Score: 256, Comments: 117](https://reddit.com/r/LocalLLaMA/comments/1jaoc8n/qwq_on_livebench_update_is_better_than_deepseek_r1/))：来自阿里巴巴的 **QwQ-32b** 在 **LiveBench** 上超越了 **DeepSeek R1**，实现了 **71.96** 的全球平均分，而 **DeepSeek R1** 为 **71.57**。如对比表所示，**QwQ-32b** 在 Reasoning、Coding、Mathematics、Data Analysis、Language 和 IF Average 等子类别中持续表现出色。
  - 对于 **QwQ-32b** 相较于 **DeepSeek R1** 的性能存在一些怀疑，部分用户指出 **Alibaba** 倾向于针对 benchmarks 而非真实场景优化模型。**QwQ-32b** 被强调为一个强大的模型，但与其 **R1** 相比，其稳定性和现实世界知识仍存疑问。
  - **Coding 性能** 是一个争议点，用户质疑 **QwQ-32b** 在 coding 能力上如何接近 **Claude 3.7**。讨论提到 **LiveBench** 主要测试 **Python** 和 **JavaScript**，而 **Aider** 测试超过 30 种语言，这表明测试环境可能存在差异。
  - 一些用户对 **QwQ-max** 的潜力表示兴奋，期待它可能在规模和性能上都超越 **R1**。还有关于设置更改对模型性能影响的讨论，并提供了进一步见解的链接（[Bindu Reddy 的推文](https://x.com/bindureddy/status/1900331870371635510)）。


- **QwQ-32b 刚刚更新了 Livebench。** ([Score: 130, Comments: 73](https://reddit.com/r/LocalLLaMA/comments/1jao3fg/qwq32b_just_got_updated_livebench/))：**QwQ 32B** 已在 **LiveBench** 上更新，为其性能提供了新的见解。完整结果可以通过 [Livebench](https://livebench.ai/#/) 链接访问。
  - **QwQ 32B** 模型因其本地 coding 能力而受到称赞，一些用户注意到它在某些任务上超越了像 **R1** 这样更大的模型。用户讨论了通过调整 **logit bias**（针对结束标签）等设置来改变模型的思考时间，一些人还尝试了最近的更新以解决无限循环等问题。
  - 讨论强调了像 **QwQ 32B** 这样的小型模型不断进化的力量，用户注意到与大型旗舰模型相比，它们在本地应用中的潜力日益增加。一些用户对该模型的创造能力及其在 benchmarks 中的出色表现感到惊讶，从而做出了取消 **OpenAI** 订阅等决定。
  - 关于 **open-source** 模型的意义存在争论，一些用户认为中国的 **open-sourcing** 策略加速了开发，这与美国专注于企业利润的方法形成对比。人们对未来 **open-source** 可用性表示担忧，特别是如果竞争优势发生转移。

- **[我制作的迷因 (Meme)](https://v.redd.it/vzku6n1lbjoe1)** ([评分: 982, 评论: 55](https://reddit.com/r/LocalLLaMA/comments/1jaoy9g/meme_i_made/)): 标题为 "Meme i made" 的帖子缺乏详细内容，仅提到了一个与 **QwQ 模型思考过程**相关的迷因创作。没有提供关于视频或迷因的其他信息或背景，因此难以提取进一步的技术见解。
  - 讨论强调了 **QwQ 模型**倾向于怀疑自己，导致低效的 Token 使用和响应时间延长。这种行为被比作过度地对自己进行“事实核查”，一些用户认为与传统的 LLM 相比，这种方式效率低下。
  - 大家的共识是，目前的推理模型（如 QwQ）仍处于早期阶段，类似于 **GPT-3** 的初始发布，预计明年其推理能力将有显著提升。用户期待向潜空间 (latent space) 推理的转变，这可能会将效率提高 **10 倍**。
  - 幽默且带有批判性的评论强调了该模型重复提问和自我怀疑的特点，并将其与过时的技术相类比，引发了关于这些模型在处理复杂推理任务时如何改进以避免过度自我质疑的讨论。


**主题 4. 去中心化 LLM 部署：Akash、IPFS 和 Pocket Network 的挑战**

- **[如何操作：在 Akash、IPFS 和 Pocket Network 上部署去中心化 LLM，这能运行 LLaMA 吗？](https://pocket.network/case-study-building-a-decentralized-deepseek-combining-open-data-compute-and-reasoning-with-pocket-network/)** ([评分: 229, 评论: 20](https://reddit.com/r/LocalLLaMA/comments/1jb1tum/howto_decentralized_llm_on_akash_ipfs_pocket/)): 标题为 **“如何在 Akash、IPFS 和 Pocket Network 上部署去中心化 LLM，这能运行 LLaMA 吗？”** 的帖子建议使用 **Akash**、**IPFS** 和 **Pocket Network** 部署去中心化大语言模型 (LLM)。它质疑了在这个去中心化基础设施上运行 **LLaMA**（一种特定的 LLM）的可行性，暗示其重点在于利用去中心化技术进行 AI 模型部署。
  - **对安全和隐私的担忧**：用户质疑 **Pocket Network** 的加密验证过程，对确保提供正确的模型以及 Prompt 的隐私表示怀疑。人们担心用户数据（如 IP 地址）是否会被记录，以及网络如何处理匿名化的延迟问题。
  - **去中心化基础设施的挑战**：评论者强调了以去中心化方式运行 LLM 的技术挑战，特别是节点之间对高带宽和低延迟的需求，这使得目前分布式 LLM 部署的可行性与单机设置相比受到限制。
  - **去中心化 vs. 中心化**：讨论对比了 **Pocket Network** 的 API 中继角色与中心化 AI 托管，指出虽然 **Pocket** 本身不运行模型，但使用 **Akash** 进行模型托管提供了诸如韧性和潜在成本节约等优势，尽管增加了一个加密层会带来复杂性。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. 使用 SDXL、Wan2.1 和长上下文微调的高级 AI 视频生成**

- **[另一个旨在追求电影级写实感的视频，这次使用了一个难度大得多的角色。SDXL + Wan 2.1 I2V](https://v.redd.it/t88g56krqnoe1)** ([评分: 1018, 评论: 123](https://reddit.com/r/StableDiffusion/comments/1jb47bs/another_video_aiming_for_cinematic_realism_this/)): 该帖子讨论了使用 **SDXL** 和 **Wan 2.1 I2V** 创建旨在实现**电影级写实感 (cinematic realism)** 视频的过程。它强调了在这种背景下处理更复杂角色的挑战。
  - **技术挑战与技巧**: **Parallax911** 分享了使用 **SDXL** 和 **Wan 2.1 I2V** 实现**电影级写实感**的复杂性，重点介绍了在 **Davinci Resolve** 中使用 **Photopea** 进行 **Inpainting** 和合成。他们提到了实现一致性和写实感的难度，特别是对于复杂的角色设计，以及使用 **Blender** 为开门等片段制作动画。
  - **项目成本与工作流**: 该项目使用 **RunPod** 的 **L40S**（价格为 **$0.84/小时**）产生了约 **$70** 的成本，耗时约 **80 小时** 的 GPU 时间。**Parallax911** 使用的工作流涉及 **RealVisXL 5.0**、**Wan 2.1** 和用于 **Upscaling** 的 **Topaz Starlight**，生成的场景为 **61 帧**、**960x544** 分辨率和 **25 steps**。
  - **社区反馈与建议**: 社区赞扬了其氛围感叙事和音效设计，并对水滴大小等元素提出了具体反馈，同时表达了对教程的需求。一些用户提出了改进建议，例如更好地整合 AI 和传统技术，并对 **Metroid** 中 **Samus Aran** 等角色的更多动作导向场景表示感兴趣。


- **[Wan2.1 中的视频扩展 - 完全在 ComfyUI 中创建 10 秒以上的高清视频](https://v.redd.it/xi58u5d3qmoe1)** ([评分: 123, 评论: 23](https://reddit.com/r/StableDiffusion/comments/1jb0h7i/video_extension_in_wan21_create_10_seconds/)): 该帖子讨论了在 **Wan2.1** 中使用 **ComfyUI** 创建 **Upscaling** 视频的**高度实验性工作流**，成功率约为 **25%**。该过程涉及从初始视频的最后一帧生成视频、合并、**Upscaling** 和 **Frame Interpolation**，具体参数如 **Sampler: UniPC**、**Steps: 18**、**CFG: 4** 和 **Shift: 11**。更多详情可以在 [工作流链接](https://civitai.com/models/1297230?modelVersionId=1531202) 中找到。
  - 用户正在询问工作流中**宽高比 (aspect ratio)** 的处理方式，质疑它是自动设置的还是需要对输入图像进行手动调整。
  - 对该工作流感兴趣的用户给出了**积极反馈**，表示对这种解决方案的期待。
  - 有人提出了对视频片段后半部分**模糊**问题的担忧，并建议这可能与输入帧的质量有关。


- **[使用 WAN 2.1 和 LTX 动画化了我的一些 AI 图片](https://v.redd.it/z5r0kyf1smoe1)** ([评分: 115, 评论: 10](https://reddit.com/r/StableDiffusion/comments/1jb0n50/animated_some_of_my_ai_pix_with_wan_21_and_ltx/)): 该帖子讨论了使用 **WAN 2.1** 和 **LTX** 创建 **AI 动画视频**。在没有进一步背景或额外细节的情况下，重点仍然是用于动画的工具。
  - **模型使用**: **LTX** 用于第一个片段（跳跃的女人和战斗机），而 **WAN** 用于奔跑的宇航员、恐怖菲比 (horror furby) 和龙。
  - **硬件详情**: 视频是使用从 **Paperspace** 租用的带有 **RTX5000** 实例的云计算机生成的。


**主题 2. OpenAI 的 Sora：将城市景观转变为反乌托邦**

- **[OpenAI 的 Sora 将旧金山的 iPhone 照片变成了反乌托邦噩梦](https://v.redd.it/y67d5ph47loe1)** ([评分: 931, 评论: 107](https://reddit.com/r/ChatGPT/comments/1jawa6c/openais_sora_turns_iphone_photos_of_san_francisco/)): **OpenAI** 的 **Sora** 是一款将**旧金山**的 **iPhone 照片**转变为具有**反乌托邦**美感图像的工具。该帖子可能讨论了使用 AI 改变现实世界图像的影响和视觉结果，尽管由于缺乏文本内容，具体细节尚不清楚。
  - 几位评论者对 **AI 生成的反乌托邦图像**的影响表示怀疑，一些人认为**旧金山**或其他城市的实际地点已经类似于这些反乌托邦视觉效果，质疑 AI 改变的必要性。
  - **iPhone** 作为拍摄原始图像的设备是一个争论点，一些人质疑它与讨论的相关性，而另一些人则强调它在理解图像来源方面的重要性。
  - 对话包含了对 **AI 能力**的钦佩和担忧，用户既对这项技术感到惊讶，又对未来区分 AI 生成和现实世界图像感到焦虑。

- **[OpenAI 的 Sora 将旧金山的 iPhone 照片变成了反乌托邦地狱景象...](https://v.redd.it/ukxvzsatzkoe1)** ([Score: 535, Comments: 58](https://reddit.com/r/OpenAI/comments/1javmkq/open_ais_sora_transformed_iphone_pics_of_san/)): **OpenAI** 的 **Sora** 将 **旧金山的 iPhone 照片** 变成了反乌托邦的地狱景象，展示了其在改变数字图像以创造未来主义、阴郁美学方面的能力。该帖子除了展示这种转变外，缺乏额外的背景或细节。
  - 评论者将 **反乌托邦图像** 与现实世界的地点进行了类比，提到了 **德里**、**底特律** 和 **印度街道**，突显了 AI 在解释城市环境时被察觉到的偏见。
  - 存在对 **AI 文本生成能力** 的担忧，一位评论者指出，图像中的 **标牌文字** 是 AI 操纵的明显迹象。
  - 用户对 **创建此类图像的过程** 表示感兴趣，并请求提供 **分步说明**，以便在自己的照片上复制这种转变。


**主题 3. OpenAI 与 DeepSeek：开源大对决**

- **[我认为太多的不安全感](https://i.redd.it/9xpl7abaoooe1.jpeg)** ([Score: 137, Comments: 58](https://reddit.com/r/ClaudeAI/comments/1jb8aj5/i_think_too_much_insecurity/)): **OpenAI** 指责 **DeepSeek** 是“受国家控制的”，并主张禁止中国 AI 模型，突显了对 AI 开发中受国家影响的担忧。图片暗示了地缘政治背景，美国和中国国旗象征着关于 AI 技术中国家控制和安全性的更广泛辩论。
  - 讨论突显了对 **OpenAI** 针对 **DeepSeek** 指控的怀疑，用户通过指出 **DeepSeek** 的模型是开源的来挑战国家控制的观点。用户质疑指控的有效性，要求提供证据，并引用了 **Sam Altman** 过去关于 **LLM** 缺乏竞争护城河的言论。
  - **DeepSeek** 被视为一个强劲的竞争对手，能够以较低的支出运营，并可能影响 **OpenAI** 的利润。一些评论建议 **DeepSeek** 的行为被视为一种经济侵略形式，等同于对美国利益的宣战。
  - 存在一股针对 **OpenAI** 和 **Sam Altman** 的强烈批评暗流，用户对他们的行为和言论表示不信任和不满。对话包括人身攻击以及对 **Altman** 公信力的怀疑，并提到了他关于开源模型的承诺尚未兑现。


- **构建了一个 AI Agent 来自动查找并申请工作** ([Score: 123, Comments: 22](https://reddit.com/r/OpenAI/comments/1jb49lo/built_an_ai_agent_to_find_and_apply_to_jobs/)): 一个名为 **SimpleApply** 的 AI Agent 通过将用户的技能和经验与相关的职位匹配，实现了求职和申请流程的自动化，提供三种使用模式：带职位评分的手动申请、选择性自动申请以及针对匹配度超过 **60%** 的职位的全自动申请。该工具旨在简化职位申请而不使雇主不堪重负，并因发现了许多用户可能无法发现的远程工作机会而受到称赞。
  - 提出了对 **数据隐私和合规性** 的担忧，涉及 **SimpleApply** 如何处理 **PII** 及其对 **GDPR** 和 **CCPA** 的遵守。开发者澄清说，他们与合规的第三方安全地存储数据，并正在制定明确的用户协议以实现完全合规。
  - 讨论了 **申请垃圾邮件风险**，并建议避免重复申请同一职位，以防止被 **ATS** 系统标记。开发者保证，该工具仅申请获得面试可能性较高的职位，以尽量减少垃圾邮件。
  - 建议了替代的 **定价策略**，例如仅在用户通过电子邮件或呼叫转移收到回访时收费。这种方法对于犹豫是否预先花钱的失业用户可能更具吸引力。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的总结

**主题 1. Google 的 Gemma 3 成为各工具关注的焦点**

- [**Unsloth 极大提升 Gemma 3 微调性能，同时支持视觉**](https://unsloth.ai/blog/gemma3)：**Unsloth AI** 现在宣布全面支持 **Gemma 3**。与 48GB GPU 上的标准 **Flash Attention 2** 设置相比，微调速度提升了 **1.6 倍**，显存（**VRAM**）占用降低了 **60%**，并将上下文长度扩展了 **6 倍**。针对全量微调、8-bit 和预训练的优化版本已在 [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b) 上发布。此外，**Gemma 3 vision** 的初步支持也已实现，不过 **Ollama** 用户目前可能会遇到兼容性问题。
- [**Gemma 3 12B 表现优于 Qwen，尚需 GPT4All 更新支持**](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b)：用户在个人测试中报告 **Gemma 3 12B** 的表现优于 **Qwen 14B** 和 **32B**，并在多语言问答方面表现出色。然而，由于架构变化和对 `mmproj` 文件的需求，**GPT4All** 需要更新以全面支持 **Gemma 3 12B**。在一次基础物理测试中，**Gemma-3-12b** 正确预测了水结冰时罐子会破碎，而 **DeepSeek-R1** 则未能做到。
- [**vLLM 和 LigerKernel 准备集成 Gemma 3**](https://github.com/linkedin/Liger-Kernel/pull/606)：**vLLM** 正在积极开发 **Gemma 3** 支持，可在 [该 GitHub issue](https://github.com/vllm-project/vllm/issues/14696) 中跟踪进度。同时，**LigerKernel** 正在进行 **Gemma 3** 的草案实现，并指出其与 **Gemma 2** 架构高度相似，仅有细微的 **RMSNorm** 调用差异；然而，一些用户报告了 **Gemma3** 在 **TGI** 上的上下文窗口大小问题。


**主题 2. 新模型涌现：OLMo 2, Command A, Jamba 1.6, PaliGemma 2 Mix**

- [**AI2 的 OLMo 2 32B 闪耀登场，被誉为开源版 GPT-3.5 杀手**](https://allenai.org/blog/olmo2-32B)：**AI2** 发布了 **OLMo 2 32B**，这是一个使用 **Tulu 3.1** 在 **6T tokens** 上训练的完全开源模型。声称其在学术基准测试中优于 **GPT3.5-Turbo** 和 **GPT-4o mini**，而训练成本仅为 **Qwen 2.5 32B** 的三分之一。该模型提供 **7B**、**13B** 和 **32B** 三种尺寸，现已在 **OpenRouter** 上线，并在 **Yannick Kilcher** 的社区中引发了关于其开源性质和性能的讨论。
- [**Cohere 的 Command A 和 AI21 的 Jamba 1.6 模型发布，具备超大上下文**](https://openrouter.ai/cohere/command-a)：**Cohere** 推出了 **Command A**，这是一个拥有 **111B 参数**和 **256k 上下文窗口**的开放权重模型，专为 **Agent**、多语言和编程任务设计。同时，**AI21** 发布了 **Jamba 1.6 Large**（**94B 激活参数**，**256K 上下文**）和 **Jamba 1.6 Mini**（**12B 激活参数**）。两者现在都支持结构化 **JSON** 输出和 **tool-use**（工具调用），所有模型均已在 **OpenRouter** 上提供。然而，**Command A** 在质数查询方面表现出一个奇怪的 bug，且据报道在没有特定补丁的情况下，本地 **API** 性能欠佳。
- [**Google 的 PaliGemma 2 Mix 系列展现视觉语言通用性**](https://huggingface.co/blog/paligemma2mix)：**Google** 发布了 **PaliGemma 2 Mix**，这是一个视觉语言模型系列，包含 **3B**、**10B** 和 **28B** 三种尺寸，支持 **224** 和 **448** 分辨率，能够处理开放式视觉语言任务和文档理解。同时，**Sebastian Raschka** 在[一篇博客文章](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html)中评测了包括 **Meta AI** 的 **Llama 3.2** 在内的多模态模型；**HuggingFace** 的用户也在寻找具有类似图像编辑能力的 **Gemini 2.0 Flash** 开源替代方案。


**主题 3. 编程工具与 IDE 随 AI 集成而演进**

- [**Cursor IDE 用户对性能下降和 Claude 3.7 降级表示不满**](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe)：**Cursor IDE** 在 **0.47.4** 等版本更新后，因在 **Linux** 和 **Windows** 上的**延迟和卡顿**遭到用户抵制；其中 **Claude 3.7** 被认为“笨拙不堪”且无视规则，同时消耗双倍积分，**Cursor agent** 也因产生过多终端而受到批评；尽管存在这些问题，**v0** 在快速 UI 原型设计方面仍广受好评，相比之下，**Cursor** 的积分系统和创意自由度较 **v0** 受到更多限制。
- [**Aider 与 Claude 联手，用户讨论 Rust 移植及 MCP Server 设置**](https://github.com/sengokudaikon/aider-mcp-server)：用户盛赞 **Claude** 与 **Aider** 的强大组合，并辅以网页搜索和 bash 脚本功能；虽然有关于将 **Aider** 移植到 **Rust** 以实现更快文件处理的讨论，但因 LLM API 瓶颈而遭到质疑；针对 **Aider MCP Server**，出现了用户改进版的 readme，但设置复杂性依然存在，**Linux** 用户正在寻找运行 **Claude Desktop** 的变通方法。
- [**“Vibe Coding” 势头强劲，在游戏开发和资源列表中展露头角**](https://github.com/filipecalegario/awesome-vibe-coding)：“Vibe Coding”（AI 辅助协作编程）的概念正受到关注，例如一位开发者利用 **Cursor** 在 20 小时内花费 20 欧元**完全使用 AI** 开发了一款多人 3D 游戏；此外，精选 AI 编程工具和资源的列表 **Awesome Vibe Coding** 已在 [GitHub](https://github.com/filipecalegario/awesome-vibe-coding) 发布，用于自动提交更改的 [GitDoc VS Code 扩展](https://github.com/lostintangent/gitdoc) 也日益流行，激发了带有可视化变更树的 “Vibe Coding” IDE 的 UI 设计灵感。


**Theme 4. 训练与优化技术进展**

- [**Unsloth 开创推理模型的 GRPO 技术，通过动态量化提升速度**](https://unsloth.ai/blog/grpo)：**Unsloth** 推出了 **GRPO**（指导偏好优化），使推理模型能够以**减少 90% VRAM** 的代价实现 **10 倍长的上下文**，并强调动态量化在质量上优于 GGUF（尤其是在 **Phi-4** 上），相关成果已展示在 [Hugging Face 排行榜](https://unsloth.ai/blog/dynamic-4bit)；同时，**Triton bitpacking** 实现了比 Pytorch 高达 **98 倍**的巨幅提速，将 **Llama3-8B** 的重新打包时间从 49 秒缩短至 1.6 秒。
- [**DeepSeek 的 Search-R1 利用 RL 实现自主查询生成，IMM 承诺更快的采样**](https://arxiv.org/abs/2503.09516)：**DeepSeek 的 Search-R1** 扩展了 **DeepSeek-R1**，利用强化学习（**RL**）在推理过程中生成搜索查询，通过检索令牌掩码实现稳定训练并增强 **LLM** 展开效果；同时，**归纳矩匹配（IMM）**作为一种新型生成模型类别出现，承诺通过单步或少步采样实现更快的推理，在无需预训练或双网络优化的情况下超越了扩散模型。
- [**Reasoning-Gym 探索 GRPO、veRL 和复合数据集以增强推理能力**](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor)：**组相对策略优化（GRPO）**在 LLM 的 RL 领域日益流行，**reasoning-gym** 证实了 **veRL 训练**在 **chain_sum** 任务上的成功，并探索通过复合数据集提升推理能力，正朝着增强“全方位”模型性能的重构迈进；该项目目前接近 500 stars，版本 *0.1.16* 已上传至 pypi。


**Theme 5. 基础设施与访问：H100、VRAM 与 API 定价**

- [**SF Compute 以低价扰乱 H100 市场，Vultr 进入推理 API 领域**](https://sfcompute.com/)：**SF Compute** 提供了令人惊讶的低 **H100** 租赁价格，特别是对于短期使用，宣传称每小时有 **128 个 H100** 可用，并即将推出另外 **2,000 个 H100**；同时 **Vultr** 宣布了推理 **API** 定价，初始阶段 **5000 万输出 token 仅需 10 美元**，随后为 **每百万 token 2 美分**，可通过兼容 **OpenAI** 的端点访问，这源于其大规模采购的 **GH200**。
- [**LM Studio 用户深入研究运行时检索和 Snapdragon 兼容性**](https://extensions.lmstudio.ai/backends-master-list-stable.json)：**LM Studio** 用户正在对应用程序进行逆向工程，以寻找离线运行时的下载 URL（此前该应用声称可离线运行），并发现了类似 [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz) 的 **CDN 'API'**；同时，**LM Studio** 对 **Snapdragon X Plus** GPU 的支持需要直接执行 *llama.cpp*，且有用户报告 **Gemini Vision** 的限制可能源于德国/欧盟的地理限制。
- [**VRAM 消耗担忧上升：讨论 Gemma 3 和 SFT**](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)：用户报告在视觉更新后 **Gemma 3 的 VRAM 使用量**增加，推测 **CLIP** 集成是原因；关于 **Gemma 的 SFT** VRAM 需求也引发了争论，认为在类似条件下其要求可能高于 **Qwen 2.5**；此外，文中还分享了估算 **LLM** 内存使用的资源，如 [Substratus AI 博客](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)和 [Hugging Face space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)。


---

# PART 1: Discord 高层摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 Gemma 3 支持势头强劲**：Unsloth 现在支持 **Gemma 3**，包括全量微调和 **8-bit**，并将 **Gemma 3 (12B) 微调优化了 1.6 倍**，**VRAM 使用量减少了 60%**，在 48GB **GPU** 上与使用 **Flash Attention 2** 的环境相比，上下文长度扩展了 **6 倍**。
   - 所有 **Gemma 3** 模型上传均可在 [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b) 上找到，包括针对全量微调、**8-bit** 和预训练优化的版本。
- **动态量化与 GGUF 质量对决**：讨论比较了**动态量化（dynamic quantization）**与 **GGUF** 模型，特别是尺寸与质量之间的权衡，Unsloth 针对 **Phi-4** 的动态量化已登上 [Hugging Face 排行榜](https://unsloth.ai/blog/dynamic-4bit)。
   - 预计将进行与 **GGUF** 基准测试的直接对比，以明确不同位宽下的性能，目前的延迟可能是因为 **llama-server** 尚不支持视觉功能。
- **GRPO 将赋予强大的推理能力**：**GRPO** (Guiding Preference Optimization) 将于*下周推出*并附带新的 notebook，现在支持 **10 倍长的上下文且 VRAM 减少 90%**，详情见[博客文章](https://unsloth.ai/blog/grpo)。
   - 团队表示，“只有让你先按照 **GRPO** 的方式推理规则”，这是专门为推理模型设计的，提供了显著的内存节省和扩展的上下文窗口。
- **视觉模型获得 Unsloth 的视觉支持**：Unsloth 实现了 *train on completions* 功能，并为 **Vision Language Models** 实现了图像缩放，模型现在可以*自动缩放图像，从而防止 OOM 并允许截断序列长度*。
   - 还分享了一个用于图像处理的 [Qwen2_VL Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_%25287B%2529-Vision.ipynb)。
- **QwQ-32B 错误修复增强模型**：**QwQ-32B 模型**已实施错误修复，如[博客文章](https://unsloth.ai/blog/qwq-32b)及相应的[模型上传](https://huggingface.co/collections/unsloth/qwen-qwq-32b-collection-676b3b29c20c09a8c71a6235)所示。
   - 这些修复提高了模型的稳定性和性能，确保更流畅的用户体验。



---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 在 Linux 和 Windows 上出现性能抖动**：用户报告 **Cursor** 在 **Linux** 和 **Windows** 上出现 **延迟（lag）** 和 **冻结**，特别是在更新到 **0.47.4** 版本之后（[下载链接](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe)）。
   - 一位用户详细说明，在 Linux 上仅发送 **20-30 条消息** 后 UI 就会冻结数秒；另一位用户指出，即使是在运行 **3.7** 版本的高性能笔记本电脑上，Windows 版也经常出现延迟。
- **Claude 3.7 被认为表现不佳且不听指令**：用户发现更新到 **0.47.4** 后，**Claude 3.7** 变得“笨得要命”，且现在使用它需要消耗双倍额度。
   - 成员们提到 **Sonnet 3.7** 会忽略全局规则，即使被提示输出这些规则时也是如此。根据[一条推文](https://x.com/kregenrek/status/1899941361908146430)，一位用户开玩笑地建议：*在你的 Prompt 中加入“做一个乖孩子”，它就能解决任何问题*。
- **Cursor Agent 触发大量终端窗口**：多位用户发现 Cursor **Agent** 正在生成过量的终端窗口，这令人感到沮丧，尤其是当它重启已经在运行的服务器时。
   - 一位成员建议，这种功能要么应该内置化，要么用户应该直接自己编写终端命令。
- **V0 因原型开发速度受到称赞**：一些用户主张在将设计转移到 Cursor 之前，使用 **v0** 进行前端原型开发，因为它具有类似 Figma 的子框架 UI 设计能力。
   - 一位用户表示：*在我看来，先构建原型和布局（更好的前端）然后本地导入到 Cursor 要好得多*，尽管其他人因为 **v0** 的额度系统和有限的创作自主权而更倾向于使用 Cursor。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LM Studio 用户寻求支持**：一位成员建议遇到 **LM Studio** 问题的用户去专门的 **LM Studio Discord** 寻求帮助。
   - 旨在为 **LM Studio** 相关问题提供更集中的帮助。
- **为立体异构体寻找 SMILES 字符串编码器**：一位成员询问是否有模型或架构可以将 **SMILES 字符串** 编码为各种 **立体异构体（stereoisomers）** 或 **ChemDraw** 输入。
   - 目标是从这些编码中提取化学描述符。
- **扩散模型在生成任务中表现出色**：分享了一篇 [Nature 文章](https://www.nature.com/articles/s41467-024-54281-3)，强调了 **扩散模型 (DMs)** 在建模复杂数据分布以及为各种媒介生成逼真样本方面的熟练程度。
   - 这些模型目前在生成图像、视频、音频和 3D 场景方面处于 SOTA 状态。
- **Search-R1 通过 RL 自主进行搜索**：介绍了 **Search-R1** 论文，详细说明了 **DeepSeek-R1** 模型的扩展，该模型利用强化学习 (**RL**) 在推理过程中生成搜索查询（参见 [论文](https://arxiv.org/abs/2503.09516)）。
   - 该模型使用检索令牌掩码（retrieved token masking）进行稳定的 **RL** 训练，通过多轮搜索交互增强了 **LLM** 的推演（rollouts）能力。
- **IMM 声称具有更快的采样时间**：分享了一篇关于 **Inductive Moment Matching (IMM)** 的[论文](https://arxiv.org/abs/2503.07565)，指出它是一类新型生成模型，承诺通过一步或几步采样实现更快的推理，超越了扩散模型。
   - 值得注意的是，与蒸馏方法不同，**IMM** 不需要预训练初始化或优化两个网络。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLM 对决：Ministral 8B 对比 Exaone 8B**：成员们建议在 LLM 任务中使用 4-bit 量化的 **Ministral 8B** 或 **Exaone 8B**。
   - 一位运行配备 24 GB RAM 的 M4 Mac mini 的用户正试图计算每秒 token 数。
- **SmolAgents 在运行 Gemma3 时遇到麻烦**：一位用户报告了在 **SmolAgents** 上运行 **Gemma3** 时出现的错误，这些错误源于代码解析和正则表达式问题，并指向了 [GitHub 上的一个潜在修复方案](https://github.com/huggingface/smolagents/pull/883)。
   - 该用户通过增加 **Ollama 上下文长度**解决了这个问题。
- **Awesome Vibe Coding 整理资源**：一个名为 [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) 的精选列表已经发布，其中包含用于 **AI 辅助编程**的工具、编辑器和资源。
   - 该列表包括 **AI 驱动的 IDE**、**基于浏览器的工具**、**插件**、**命令行工具**以及关于 *vibe coding* 的**最新动态**。
- **PaliGemma 2 模型发布**：Google 发布了 **PaliGemma 2 Mix**，这是一个视觉语言模型系列，包含三种尺寸（**3B**、**10B** 和 **28B**）以及 **224** 和 **448** 两种分辨率，*能够通过开放式提示词执行视觉语言任务*。
   - 欲了解更多信息，请查看[博客文章](https://huggingface.co/blog/paligemma2mix)。
- **象棋锦标赛模型下出违规着法？**：一位用户分享了一个名为 *Chatbot Chess Championship 2025* 的 [YouTube 播放列表](https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC)，展示了语言模型或象棋引擎对弈的过程。
   - 参与者推测这些模型是真正的语言模型还是仅仅在调用象棋引擎，其中一人注意到某个语言模型下出了违规着法。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Complexity 扩展进入全面维护模式**：由于布局更新导致 [Complexity 扩展](https://github.com/danielcardeenas/perplexity-complexity)失效，该扩展目前已进入全面维护模式。
   - 开发者感谢用户在维护期间的耐心等待。
- **锁定内核以保证安全只是白日梦**：在常规频道中，用户们争论锁定内核是否能提高安全性。
   - 其他人则认为由于 **Linux 的开源特性**，这是不可行的，一位用户还开玩笑说不如改用 Windows。
- **Perplexity 用户恳求更大的上下文**：用户请求在 Perplexity AI 中提供**更大的上下文窗口**，并愿意为此支付额外费用以避免使用 ChatGPT。
   - 一位用户列举了 Perplexity 的功能，如*一次对 50 个文件进行无限研究*、**自定义指令空间**以及**选择推理模型的能力**，作为留下的理由。
- **Grok 3 发布即充满 Bug**：据报道，新发布的 **Grok AI** 存在很多 Bug。
   - 用户报告说*聊天会突然停止工作或在中间断开*。
- **Gemini Deep Research 并不那么“深度”**：测试新的 **Gemini Deep Research** 功能的用户发现它比 **OpenAI** 的产品弱。
   - 一位用户发现，即使禁用了搜索，它保留的上下文也比普通版 Gemini 少。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude + Aider = 编程超能力**：成员们讨论了将 **Claude** 与 **Aider** 结合使用，后者通过**网页搜索/URL 抓取和运行 bash 脚本调用**来增强功能，从而实现更强大的 Prompting 能力。
   - 一位用户强调，**添加到 Claude 的每个独特工具所释放的能力远超其各部分之和**，尤其是当模型在互联网上搜索 Bug 时。
- **Rust 能否提升 Aider 的速度？**：一位用户询问是否可以将 **Aider** 移植到 **C++** 或 **Rust** 以实现更快的文件处理，特别是在为 **Gemini** 模型加载大型上下文文件时。
   - 其他人对此表示怀疑，认为瓶颈仍然在于 **LLM API**，任何改进可能都无法**量化**。
- **Linux 爱好者启动 Claude Desktop**：由于没有官方版本，用户分享了让 **Claude Desktop** 应用在 **Linux** 上运行的指南。
   - 一位用户引用了一个 [GitHub repo](https://github.com/aaddrick/claude-desktop-debian)，提供了**基于 Debian** 的安装步骤，而另一位用户分享了他们对 **Arch Linux PKGBUILD** 的修改。
- **Aider MCP Server Readme 获救**：用户讨论了 **Aider MCP Server**，有人提到另一位用户的 Readme 写得“好上 100 倍”，指的是[这个仓库](https://github.com/sengokudaikon/aider-mcp-server)。
   - 然而，另一位用户幽默地表示，尽管有 Readme，他们仍然**无法配置好 MCP**。
- **DeepSeek 模型话太多**：一位用户反映 **DeepSeek 模型** 生成了过多的输出，大约有 **20-30 行** 废话，并询问是否可以在配置中设置 `thinking-tokens` 值。
   - 有人指出，对于 **R1 模型** 来说，**20 行** 是非常标准的，一位用户分享说他们曾为了让模型思考一个 **5 个单词的 Prompt** 而等待了 **2 分钟**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OLMo 2 32B 压倒 GPT 3.5**：AI2 发布了 [**OLMo 2 32B**](https://allenai.org/blog/olmo2-32B)，这是一个使用 Tulu 3.1 训练、数据量高达 **6T tokens** 的完全开源模型，在学术基准测试中表现优于 **GPT3.5-Turbo** 和 **GPT-4o mini**。
   - 据称其训练成本仅为 **Qwen 2.5 32B** 的三分之一，同时达到了类似的性能，并提供 **7B**、**13B** 和 **32B** 参数版本。
- **Vibe Coding 利用 AI 创建游戏**：一位开发者 **100% 利用 AI** 创建了一款多人 3D 游戏，耗时 20 小时，花费 20 欧元，并将这一概念称为 *vibe coding*，同时分享了[指南](https://x.com/nicolaszu/status/1899931187398979890?s=46)。
   - 该游戏具有逼真的元素，如打击反馈、受损时的烟雾以及死亡时的爆炸，全部通过在 **Cursor** 中输入 Prompt 生成，无需手动编辑代码。
- **Levels.io 的 AI 飞行模拟器 ARR 飙升至 100 万美元**：一位成员提到了 [Levels.io 的飞行模拟器](https://x.com/levelsio/status/1893350391158292550)的成功，该模拟器使用 **Cursor** 构建，通过在游戏中出售广告，迅速达到了 **100 万美元的 ARR**（年度经常性收入）。
   - Levelsio 指出：“*AI 对我来说确实是创意和速度的放大器，让我变得更有创意、更高效*。”
- **GitDoc 扩展自动提交更改**：成员们分享了 [GitDoc VS Code 扩展](https://github.com/lostintangent/gitdoc)，它允许你编辑 **Git** 仓库并在每次更改时自动提交。
   - 一位用户建议增加分支、重启等功能，并表示“*存储很便宜，就像在每次更改时自动提交并可视化更改树一样*”。
- **Latent Space 播客深入探讨 Snipd AI 应用**：Latent Space 播客发布了关于 **Snipd** 的新[播客节目](https://x.com/latentspacepod/status/1900666708270215383)，与 Kevin Ben Smith 讨论了用于学习的 **AI 播客应用**，并在 [YouTube](https://youtu.be/FNRO_SYx68Q) 上发布了他们有史以来第一场**户外播客**。
   - 播客内容包括关于 @aidotengineer NYC 的讨论、从**金融转向技术领域**的经历、AI 如何帮助我们从播客时间中获得更多收获，并透露了 **Snipd 应用的技术栈**细节。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **通过逆向工程获取 LM Studio 的运行环境 (Runtimes)**：一位用户反编译了 **LM Studio** 以查找供离线使用的下载 URL，发现了 [后端主列表 (backends master list)](https://extensions.lmstudio.ai/backends-master-list-stable.json) 以及像 [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz) 这样的 CDN “API”。
   - 此举是在另一位用户声称 *LM Studio 不需要互联网连接即可运行* 之后进行的，显示了对离线运行环境访问的需求。
- **Snapdragon 支持需要直接执行 llama.cpp**：一位用户报告称 **LM Studio** 无法检测到其 **Snapdragon X Plus** GPU，另一位成员回复称 GPU 支持需要直接运行 *llama.cpp*。
   - 他们引导用户参考此 [github.com/ggml-org/llama.cpp/pull/10693](https://github.com/ggml-org/llama.cpp/pull/10693) Pull Request 以获取更多信息。
- **Gemini Vision 受地理限制阻碍**：用户报告了在测试 **Gemini 2.0 Flash Experimental** 的图像处理能力时遇到问题，这可能是由于德国/欧盟的地区限制。
   - 一位德国用户怀疑这些限制是由于当地法律引起的，而一位美国用户报告称 AI Studio 中的 Gemini 也未能执行图像处理操作。
- **AI 象棋锦标赛凸显模型准确率**：举办了一场包含 **15 个模型** 的 AI 象棋锦标赛，结果可在 [dubesor.de/chess/tournament](https://dubesor.de/chess/tournament) 查看，结果受对局长度和对手走法的影响。
   - 虽然 **DeepSeek-R1** 达到了 **92%** 的准确率，但组织者澄清说，准确率会根据对局长度和对手走法而变化，且普通的 O1 在锦标赛中运行成本太高。
- **视觉更新后 Gemma 3 的 VRAM 消耗激增**：在一次提升视觉速度的更新后，一位用户报告 **Gemma 3 的 VRAM 使用量** 显著增加。
   - 有推测认为，下载体积的增加可能是因为使用了 **CLIP** 进行视觉处理，且可能由于从独立文件调用，从而增加了整体内存占用 (memory footprint)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes 3 转换为 MLX**：模型 [mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit](https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit) 已使用 mlx-lm **0.21.1** 版本从 [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview) 转换为 MLX 格式。
   - 此次转换允许在 Apple Silicon 和其他兼容 MLX 的设备上高效使用。
- **深入探讨 Hermes 3 的 vLLM 参数**：成员们正在分享不同的配置以使 **vllm** 正确配合 **Hermes-3-Llama-3.1-70B-FP8** 运行，包括为 Hermes 3 70B 添加 `--enable-auto-tool-choice` 和 `--tool-call-parser` 等建议。
   - 一位成员指出分词器 (tokenizer) 中需要 `<tool_call>` 和 `</tool_call>` 标签，这些标签存在于 **Hermes 3 模型** 中，但不一定存在于 **DeepHermes** 中。
- **Vultr 公布推理定价**：来自 Vultr 的一位成员分享了其推理 API 的官方定价，最初 **10 美元可获得 5000 万输出 token**，之后为 **每百万输出 token 2 美分**，可通过位于 [https://api.vultrinference.com/](https://api.vultrinference.com/) 的 OpenAI 兼容端点访问。
   - 据一位成员称，这种定价是因为购买了“多得离谱的 GH200”，并且需要给它们找点活干。
- **动态 LoRA 对接到 vLLM**：成员们讨论了使用 **vllm** 托管动态 **LoRA** 以应对各种用例（如最新的编码风格）的可能性，并引用了 [vLLM 文档](https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters)。
   - 建议允许用户传入其 Hugging Face 仓库 ID 作为 **LoRA**，并将其提供给 **vLLM serve 命令的 CLI 参数**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Astro 客户端正准备集成 MCP**：一位成员计划为其 **Astro** 客户端使用 **MCP**，将 **AWS API Gateway** 与作为 **Lambda** 函数的每个 **MCP** 服务器结合使用，并利用带有 **SSE gateway** 的 **MCP** 桥接。
   - 目标是专门为客户启用 **MCP** 使用，并探索将 **MCP** 服务器添加到单个项目中以提高客户端可见性。
- **解码 MCP 服务器架构**：一位成员询问像 **Cursor** 和 **Cline** 这样将 **MCP** 服务器保留在客户端的客户端是如何与后端通信的。
   - 讨论涉及这些客户端使用的架构和通信方法，但为了获取详细信息，已被重定向到更具体的频道。
- **智能代理服务器转换为 Agentic MCP**：一个智能代理 **MCP** 服务器通过其自身的 **LLM**，将具有许多工具的标准 **MCP** 服务器转换为仅具有单个工具的服务器，这实际上是使用 *vector tool calling* 的子代理（sub-agent）方法。
   - **OpenAI Swarm framework** 遵循类似的过程，将工具子集分配给单个 **Agent**，现在被 **OpenAI** 更名为 **openai-agents**。
- **调试器使用 MCP 服务器调试网页**：一位成员分享了一个调试器项目 **chrome-debug-mcp** ([https://github.com/robertheadley/chrome-debug-mcp](https://github.com/robertheadley/chrome-debug-mcp))，该项目使用 **MCP** 配合 **LLMs** 调试网页，最初使用 **Puppeteer** 构建。
   - 该项目已移植到 **Playwright**，更新后的 **GitHub** 仓库在进一步测试后待发布。
- **MCP Hub 概念简化服务器管理**：为了增强企业对 **MCP** 的采用，一位成员创建了一个 **MCP Hub** 概念，其特点是拥有一个用于简化服务器连接、访问控制和跨 **MCP** 服务器可见性的仪表板，如[此视频](https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing)中演示的那样。
   - 该 **Hub** 旨在解决企业环境中管理多个 **MCP** 服务器和权限的担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 没收员工护照**：据 [Twitter 上的 Amir](https://fxtwitter.com/amir/status/1900583042659541477) 报道，**DeepSeek** 的所有者据称要求研发人员上交护照以防止出国旅行。
   - 成员们辩论这是否会导致 **DeepSeek** 开展更多开源工作，或者美国是否可能采取类似措施。
- **SF Compute H100 价格震惊市场**：一位成员指出 [SF Compute](https://sfcompute.com/) 提供的 **H100** 价格低得令人惊讶，特别是短期租赁，广告称有 **128 个 H100** 可供按小时使用。
   - **San Francisco Compute Company** [即将推出额外的 **2,000 个 H100**](https://x.com/evanjconrad/status/1884361612766896510)，并运营一个针对大规模、经过验证的 **H100** 集群的市场，同时还拥有一个[简单但强大的 **CLI**](https://docs.sfcompute.com)。
- **Gemma 3 许可证引发关注**：最近的一篇 **TechCrunch** 文章强调了对模型许可证的担忧，特别是 [Google 的 Gemma 3](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/)。
   - 文章指出，虽然 **Gemma 3** 的许可证效率很高，但其限制性和不一致的条款可能会给商业应用带来风险。
- **用户数据隐私受到威胁**：一位成员报告了他们对个人在网上发现其电话号码并提出未经请求的要求（如 *"hey nato, can you post-train my llama2 model? ty"*）的沮丧。
   - 他们推测浏览器扩展或付费服务是来源，并正在寻求从 [Xeophon](https://x.com/alexalbert__/status/1900592059364634973) 等网站删除其数据的方法。
- **Math-500 采样得到验证**：针对 **Qwen** 的 [GitHub 仓库](https://github.com/QwenLM/QwQ) 评估脚本中看似随机采样的问题，确认采样*显然*是随机的。
   - 成员们引用了 [Lightman et al 2023](https://cdn.discordapp.com/attachments/1179128538679488533/1350170687460868186/image.png?ex=67d5c3f0&is=67d47270&hm=6c771f09d27b7bad57e711e55ed2b111ac29af6a6485feb2c89103757f0771de&)，并表示长上下文评估和答案提取令人头疼，而 **Math 500** 的相关性非常好。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cohere 凭借 111B 模型引发关注**：**Cohere** 发布了 [Command A](https://openrouter.ai/cohere/command-a)，这是一款全新的 **open-weights 111B 参数模型**，拥有 **256k 上下文窗口**，专注于 Agent、多语言和编程应用。
   - 该模型旨在为各种用例提供高性能表现。
- **AI21 Jamba 推出新模型**：**AI21** 发布了拥有 **940 亿激活参数**和 **256K token 上下文窗口**的 [Jamba 1.6 Large](https://openrouter.ai/ai21/jamba-1.6-large)，以及拥有 **120 亿激活参数**的 [Jamba 1.6 Mini](https://openrouter.ai/ai21/jamba-1.6-mini)。
   - 两款模型现在都支持结构化 JSON 输出和 tool-use。
- **Gemma 3 免费开放**：**Gemma 3** 的所有变体均可免费使用：[Gemma 3 12B](https://openrouter.ai/google/gemma-3-12b-it:free) 引入了多模态功能，支持视觉-语言输入和文本输出，并能处理高达 **128k tokens** 的上下文窗口。
   - 该模型理解超过 **140 种语言**，同时还推出了 [Gemma 3 4B](https://openrouter.ai/google/gemma-3-4b-it:free) 和 [Gemma 3 1B](https://openrouter.ai/google/gemma-3-1b-it:free) 模型。
- **Anthropic API 异常已解决**：**Anthropic** 报告了一起针对 **Claude 3.7 Sonnet** 请求错误率升高的事件，更新已发布在他们的 [状态页面](https://status.anthropic.com/incidents/qtxnlg9yrwqv)。
   - 该事件现已解决。
- **象棋锦标赛让 AI 模型同台竞技**：一场 AI 象棋锦标赛（可在此处访问：[here](https://dubesor.de/chess/tournament)）让 **15 个模型**展开对决，使用标准象棋符号表示棋盘状态、游戏历史和合法步法。
   - 模型会接收到关于棋盘状态、游戏历史和合法步法列表的信息。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Go 在实际移植中胜出**：成员们讨论了在移植代码时使用 **Go** 与 **Rust** 的实用性，解释说逐个函数地移植到 **Go** 可以实现精确的行为对等，避免长达数年的代码重写。
   - 虽然 *Rust 更快且更高效*，但有成员指出 *Golang 的开发体验非常符合人体工程学 (ergonomic)*，特别是在分布式、异步或网络应用方面。
- **DeepSeek 炒作嫌疑引发讨论**：一些成员认为围绕 **DeepSeek** 的炒作是人为操纵的，且其模型经过了简化，将它们与 **frontier AI models** 相比就像是*拿菠萝比苹果*。
   - 另一些人则为 **DeepSeek** 辩护，称其*疯狂的工程师*开发出了*比生命还快*的文件系统。
- **OLMo 2 32B 完全开源**：**OLMo 2 32B** 作为[首个完全开源模型](https://allenai.org/blog/olmo2)发布，在学术基准测试中超越了 **GPT3.5-Turbo** 和 **GPT-4o mini**。
   - 据称其性能可与领先的 open-weight 模型媲美，而训练成本仅为 **Qwen 2.5 32B** 的三分之一。
- **ChatGPT 被高估，用户更青睐 Claude**：一位成员表示 **ChatGPT** 被高估了，因为*它实际上解决不了我需要解决的问题*，他更倾向于 **Mistral Small 24B**、**QwQ 32B** 和 **Claude 3.7 Sonnet**。
   - 另一位用户分享道：*我用 Claude 得到想要结果的运气更好*，而且它*似乎出于某种原因更擅长理解意图和动机*。
- **Grok 3 编写专业级代码**：成员们辩论了代码生成的质量，强调 **OpenAI** 模型经常生成过时的 (legacy) 代码，而 **Mistral** 可以将其重构为更现代的代码。
   - 还有人指出 **Grok 3** 生成的代码*看起来像是专业程序员写的*，而在 **VSCode** 中，一位成员表示相比 **Copilot** 他更喜欢使用 **Amazon Q**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Speech-to-Speech 模型引发探索**：成员正在积极寻找专注于对话式语音的 **speech-to-speech generation** 模型，将其与 **OpenAI Realtime API** 或 **Sesame AI** 等多模态模型区分开来。
   - 确定了两个潜在模型：来自 **Kyutai Labs** 的 [Moshi](https://github.com/kyutai-labs/moshi) 和来自 **Standard-Intelligence** 的 [Hertz-dev](https://github.com/Standard-Intelligence/hertz-dev)。
- **Block Diffusion 桥接自回归与扩散模型**：**Block Diffusion** 模型在 **ICLR 2025 Oral** 演讲中进行了详细介绍，它结合了自回归和扩散语言模型的优点，提供**高质量**、**任意长度生成**、**KV caching** 以及**可并行化**处理。
   - 代码可在 [GitHub](https://github.com/kuleshov-group/BD3-LMs) 和 [HuggingFace](https://huggingface.co/collections/kuleshov-group/BD3-LMs-67be95f81b96b15fec50d53f) 上找到。
- **Triton bitpacking 获得巨大提升**：**Triton** 中的 **Bitpacking** 相比 4090 上的 Pytorch 实现实现了显著加速，**32-bit packing** 达到 **98x** 加速，**8-bit packing** 达到 **26x** 加速。
   - 使用新的 bitpacking 实现，重新打包 **Llama3-8B** 模型的时间从 **49 秒缩短至 1.6 秒**，代码可在 [GitHub](https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133) 获取。
- **Gemma3 在 vLLM 和 LigerKernel 中受到关注**：成员们讨论了在 **vLLM** 中增加对 **Gemma 3** 的支持，引用了[这个 GitHub issue](https://github.com/vllm-project/vllm/issues/14696)；同时一名成员已开始起草将 **Gemma3** 集成到 **LigerKernel** 的实现，并分享了 [Pull Request 链接](https://github.com/linkedin/Liger-Kernel/pull/606)。
   - 根据 Pull Request，**Gemma3** 与 **Gemma2** 高度相似，但在 **RMSNorm Calls** 方面存在一些差异。
- **GRPO 在 LLM 训练中走红**：成员们讨论了 **Group Relative Policy Optimization (GRPO)** 如何在 LLM 的强化学习中变得流行，并引用了 [DeepSeek-R1 论文]()。
   - 分享了来自 oxen.ai 关于 [GRPO VRAM requirements](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor) 的博客文章，指出了其在训练中的有效性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **智力下降引发辩论**：讨论源于一篇[《金融时报》文章](https://www.ft.com/content/link-to-ft-article)，该文章指出发达国家的平均智力正在下降，理由是关于**认知挑战**以及**推理和问题解决**能力下降的报告增多。
   - 一位成员理论化认为这可能是由于技术（尤其是**智能手机**和**社交媒体**）导致思维外包，然而图表显示的年份实际上是在 **ChatGPT** 问世之前。
- **技术是认知能力下降的罪魁祸首吗？**：成员们辩论了认知能力下降的潜在原因，包括**技术的影响**、**移民**和**氟化水**。
   - 一位成员指出，认知挑战的比例自 20 世纪 90 年代以来稳步上升，并在 2012 年左右突然加速。
- **DeepSeek V3 蒸馏自 OpenAI 模型**：讨论提到 **Deepseek V3 (instruct 版本)** 可能是从 **OpenAI** 模型中蒸馏出来的。
   - 一位成员指出，*甚至 OpenAI 也会非正式地支持蒸馏他们的模型，他们只是似乎不喜欢 Deepseek 这么做*。
- **Claude Sonnet 3.7 在编程任务中占据主导地位**：一位成员现在专门使用 **Claude Sonnet 3.7** 进行编程，发现 **ChatGPT** 已经落后。
   - 在相关新闻中，一位成员表示 **o3-mini-high** 模型优于 **o1**。
- **食品添加剂加剧智力衰退**：成员们讨论了**超加工食品 (UPFs)** 的供应和消费在全球范围内有所增加，目前在一些高收入国家占每日能量摄入的 **50–60%**，并且与认知能力下降有关。
   - 另一位成员提到了像**雀巢 (Nestlé)** 这样在许多国家运营的跨国公司在全球范围内生产和分销，这些公司对产品添加剂的调整或改变可能会产生全球性的影响。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini 2.0 Deep Research 加入 NotebookLM？**：成员们正在探索将 **Gemini 2.0 Deep Research** 与 **NotebookLM** 结合使用，以增强文档处理能力。
   - 社区讨论了 **Deep Research** 在功能上最终是否会取代 **NotebookLM**。
- **NotebookLM 启发非洲项目 Ecokham**：一位来自非洲的成员报告称，他使用 **NotebookLM** 来连接思路、编辑路线图并为他的项目 **Ecokham** 生成音频。
   - 他对 **NotebookLM** 启发其团队所做的贡献表示感谢。
- **NotebookLM 用于 PhytoIntelligence 框架原型设计**：一位成员正利用 **NotebookLM** 整理笔记，并为用于自主营养保健品设计的 **PhytoIntelligence 框架** 制作原型，旨在缓解诊断挑战。
   - 该用户对 Google 提供的这一工具能力表示认可。
- **用户要求 NotebookLM 具备图像和表格识别能力**：用户要求 NotebookLM 支持 **图像和表格识别**，抱怨目前的状态感觉不完整，因为需要不断重新打开源文件并翻阅 Google Sheets；一位用户甚至分享了 [相关的猫咪 GIF](https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569)。
   - 社区强调图像胜过“千言万语”，而最清晰的数据通常存在于表格中。
- **NotebookLM 移动端 App 仍未上线**：用户正积极请求推出 **移动端 App 版本** 的 NotebookLM，以提高可访问性。
   - 社区感觉移动版本“似乎仍遥遥无期”。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Google Gemini 和 Vertex AI 在 LlamaIndex 中合并！**：`@googleai` 集成统一了 **Google Gemini** 和 **Google Vertex AI**，支持流式传输、异步、多模态和结构化预测，甚至支持图像，详见 [此推文](https://twitter.com/llama_index/status/1900590246070476929)。
   - 此次集成简化了利用 Google 最新模型构建应用的过程。
- **关于 LlamaIndex 优势的讨论**：一位成员寻求了解 **LlamaIndex** 相比 **Langchain** 在构建应用方面的优势。
   - 在提供的上下文范围内，该询问未得出结论性的讨论。
- **探讨 OpenAI 缺少 Delta 事件的原因**：一位成员询问为什么 **OpenAI** 模型在进行 tool calling 时不发出 delta 事件，观察到事件虽然发出了但内容为空。
   - 共识是 tool calling 无法进行流式传输，因为 LLM 需要完整的工具响应才能生成后续响应，因此建议采用自定义（DIY）方法。
- **关于 Agentic RAG 应用 API 的疑问**：有人提问是否存在专门用于构建 **Agentic RAG 应用** 的 **API**，以简化开发和管理。
   - 对话提到 LlamaIndex 中有多种构建模块可用，但缺乏清晰、明确的指南。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Gemma 3 12B 在智商测试中胜过 Qwen**：一位用户报告称，在其个人电脑上，**Gemma 3 12B 模型** 的智能表现优于 **Qwen 14B** 和 **Qwen 32B**。
   - 这是通过多语言提问测试的；**Gemma 3** 和 **DeepSeek R1** 始终能以与问题相同的语言提供正确答案。
- **Gemma 3 需要新的 GPT4All 支持**：用户注意到 **GPT4All** 可能需要更新才能完全支持 **Gemma 3 12B**，因为其架构与 **Gemma 2** 不同。
   - 具体而言，**Gemma 3** 需要一个 *mmproj* 文件才能在 **GPT4All** 中运行，这凸显了快速适应新 AI 模型开发的挑战。
- **冻水实验测试 AI 知识**：当被问及关于冻水的问题时，**DeepSeek-R1** 错误地预测罐子会破裂，而 **Gemma-3-12b** 则准确描述了由于水膨胀导致的破碎效应。
   - 这展示了模型对基础物理学理解水平的差异，表明了不同架构之间多样化的推理能力。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **显式反馈流入 Refine**：一名成员请求在 `dspy.Refine` 中重新引入**显式反馈**（explicit feedback），类似于 `dspy.Suggest`，以增强调试和理解能力。
   - 该成员强调了**显式反馈**对于识别需要改进之处的价值。
- **手动反馈在 Refine 中发挥作用**：团队宣布在 `Refine` 中增加了**手动反馈**功能。
   - 该实现涉及将反馈作为 `dspy.Prediction` 对象包含在**奖励函数（reward function）的返回值**中，其中包含分数和反馈。
- **奖励函数返回反馈**：一名团队成员询问了将反馈作为 **reward_fn 返回值**的一部分进行集成的可行性。
   - 用户给出了*肯定*答复，并表达了感谢。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A 在 OpenRouter 亮相**：**Cohere 的 Command A** 是一款拥有 **111B** 参数、**256k** 上下文窗口的开源权重模型，现已可在 [OpenRouter](https://openrouter.ai/cohere/command-a) 上访问。
   - 该模型旨在在 Agent、多语言和编程应用中提供高性能，为开源权重模型树立了新标准。
- **Command A 未能通过质数测试**：用户发现 **Command A** 中存在一个奇特的 Bug：当询问数字之和为 15 的质数时，模型要么提供错误答案，要么陷入死循环。
   - 这种意外行为凸显了模型在数学推理能力方面可能存在的缺陷。
- **本地 API 在运行 Command A 时遇到困难**：用户在本地运行 **Command A** 时遇到了性能瓶颈，报告称即使有足够的 VRAM，如果不修补 **ITOr** 中的建模或使用 **APIs**，模型也无法达到理想的速度。
   - 这表明优化 **Command A** 的本地部署可能需要进一步工作以提高其效率。
- **Cohere 发布兼容性 base_url**：一名成员建议使用 [Cohere 兼容性 API](https://api.cohere.com/compatibility/v1/chat/completions)。
   - 他们建议利用 base_url 进行集成。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord 账号冒充警报**：一名成员报告称收到一个冒充其他用户的**诈骗账号**发来的消息，被冒充的用户确认：“那不是我。这是我唯一的账号”。
   - **虚假账号** *caroline_frascaa* 已被举报至 Discord，并在用户发布[虚假账号截图](https://cdn.discordapp.com/attachments/1098765954302873621/1350124411906293851/Impersonating_Account.png?ex=67d598d7&is=67d44757&hm=bd212e9e154251a202378828ccf61282fd69df840ade2eb535738fc7d7e248cb&)后被从服务器封禁。
- **Mojo 标准库（stdlib）用途讨论**：*soracc* 在 #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/) 中提到了 **Mojo 标准库**中某些特性的使用。
   - 用户提到它被用于 `base64`。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **自我反思需要外部评估**：一名成员询问了第一节课与第二节课关于**自我评估**（self-evaluation）的表述，认为在外部反馈的作用方面存在矛盾。
   - 第一节课强调**自我反思**（self-reflection）和**自我优化**（self-refinement）受益于良好的外部评估，而没有 Oracle 反馈的**自我修正**（self-correction）可能会降低推理性能。
- **寻求对第 1 课和第 2 课中自我评估的澄清**：一名用户正在寻求关于课程中关于**自我评估**明显矛盾点的澄清。
   - 他们注意到第二节课强调了**自我评估和改进**，而第一节课则强调了外部评估的重要性以及没有 Oracle 反馈的自我修正可能带来的危害。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Vertex 为 1.6 版本做准备**：**Version 1.6** 尚未在 **Vertex** 上提供，但计划在不久的将来推出。
   - 它也将可在 **AWS** 等其他平台上提供，以实现更广泛的访问。
- **AWS 即将托管 1.6 版本**：**1.6** 版本将在不久的将来在 **AWS** 等平台上提供，从而扩大其覆盖范围。
   - 这一进展旨在让 **AWS** 客户能够访问新功能。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1350033025223622667)** (301 messages🔥🔥): 

> `Unsloth 中的 Gemma 3 支持、多 GPU 训练、动态量化 vs GGUF、GRPO 与推理、视觉模型` 


- **Unsloth 发布 Gemma 3 支持**：Unsloth 现在支持 **Gemma 3**，包括全量微调和 8-bit，支持几乎所有模型，如 **Mixtral, Cohere, Granite**。
   - 根据一条推文，Unsloth 的优化使得 **4-bit 的 VRAM 占用减少了 10%**，**速度提升了 10%**，此外还修复并改进了 **Windows 支持**和 **GGUF 转换**。
- **多 GPU 支持仍在计划中**：尽管用户很感兴趣，但 Unsloth 免费版目前**原生不支持多 GPU 训练**。
   - 然而，有推测称可以通过将组件解构到你的训练代码（FastLanguageModel）来实现，并且 **AGPL3 多 GPU** 版本和 **Enterprise** 版本即将发布。
- **动态量化与 GGUF 的质量对决**：关于**动态量化**与 **GGUF** 模型的比较正在进行中，特别是关于尺寸和质量之间的权衡。
   - Unsloth 针对 **Phi-4** 的动态量化已登上 [Hugging Face 排行榜](https://unsloth.ai/blog/dynamic-4bit)，但预计会与 GGUF 基准进行直接对比，以明确不同位宽下的性能。
- **GRPO 增强推理能力**：团队提到 **GRPO** (Guiding Preference Optimization) 将于*下周*随新的 notebook 一起推出。
   - 他们将提供一个 GRPO notebook；他们表示，*只有像 GRPO 那样先让它对规则进行推理才行*。
- **视觉模型获得 Unsloth 优化**：Unsloth 实现了 *train on completions* 功能，并针对 Vision Language Models 提供了图像缩放功能（这是一个需求量很大的功能），以减少 OOM。
   - 模型现在会*自动调整图像大小，从而防止 OOM，并允许截断序列长度。*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively">教程：如何高效运行 Gemma 3 | Unsloth 文档</a>：如何通过我们的 GGUF 在 llama.cpp、Ollama、Open WebUI、LM Studio 上高效运行 Gemma 3。</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth 基准测试 | Unsloth 文档</a>：想知道 Unsloth 有多快吗？</li><li><a href="https://unsloth.ai/blog/reintroducing">重新介绍 Unsloth</a>：为了庆祝我们成为当日 GitHub 趋势榜第一的仓库，我们回顾了我们的历程以及对开源社区的贡献。</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>：Unsloth 的动态 4-bit 量化（Dynamic 4-bit Quants）有选择性地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，极大地提高了准确性。</li><li><a href="https://huggingface.co/docs/bitsandbytes/main/en/explanations/optimizers#paged-optimizers">8-bit 优化器</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslot">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth 更新：Mistral 支持及更多</a>：我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了滑动窗口注意力（sliding window attention）、初步的 Windows 和 DPO 支持，以及 ...</li><li><a href="https://x.com/danielhanchen/status/1900592202621087944">来自 Daniel Han (@danielhanchen) 的推文</a>：很高兴分享 @UnslothAI 现在支持：• 全量微调 + 8bit • 几乎任何模型，如 Mixtral、Cohere、Granite、Gemma 3 • 视觉微调不再出现 OOM！包含详情的博客文章：https://unsl...</li><li><a href="https://github.com/unslothai/unsloth/issues/2009)">unslothai/unsloth</a>：以 2 倍的速度和减少 70% 的内存微调 Llama 3.3、DeepSeek-R1 和推理型 LLM！🦥 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/36683">AttributeError: 'Gemma3Config' 对象没有属性 'vocab_size' · Issue #36683 · huggingface/transformers</a>：系统信息 v4.50.0.dev0 谁能帮忙？@ArthurZucker @LysandreJik @xenova 信息 官方示例脚本 我自己修改的脚本 任务 示例文件夹中官方支持的任务 ...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>：未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12343#issuecomment-2718131134">llama : 由 ngxson 添加 Gemma 3 支持（+ 实验性视觉功能）· Pull Request #12343 · ggml-org/llama.cpp</a>：模型信息 官方模型页面：https://ai.google.dev/gemma/docs/core 预量化 GGUF：https://huggingface.co/collections/ggml-org/gemma-3-67d126315ac810df1ad9e913 可用尺寸：1B, 4B, 12B,...</li><li><a href="https://github.com/vllm-project/vllm/pull/14660">[Model] 由 WoosukKwon 添加对 Gemma 3 的支持 · Pull Request #14660 · vllm-project/vllm</a>：此 PR 添加了对 Gemma 3 的支持，这是来自 Google 的开源视觉语言模型。注意：此 PR 尚未实现 pan-and-scan 预处理算法。它将由后续的...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1350168485648662579)** (1 messages): 

> `Gemma 3 models, Unsloth support for models, GRPO for reasoning models, QwQ-32B bugfixes, New model uploads` 


- **Google 的 Gemma 3 已集成至 Unsloth**：Unsloth 现已支持 Google 最新的 **Gemma 3** 模型，并提供了[博客文章](https://unsloth.ai/blog/gemma3)和 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)。
   - 所有 Gemma 3 模型上传均可在 [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b) 上获取，包括针对全量微调（full finetuning）、8-bit 和预训练（pretraining）优化的版本。
- **Unsloth 提升 Gemma 3 微调速度**：在 48GB GPU 上，与使用 Flash Attention 2 的环境相比，Unsloth 将 **Gemma 3 (12B) 微调速度提升了 1.6 倍**，**VRAM 占用减少了 60%**，并将上下文长度扩展了 **6 倍**。
   - 团队修复了 **Gemma 3** 训练中的问题，并上传了所有版本，包括 2-8 bit GGUFs、dynamic 4-bit 和 16-bit 版本。
- **GRPO 在降低 VRAM 占用的同时支持超长上下文**：Unsloth 现在通过 GRPO (Generalized Rank Position Optimization) 支持 **10 倍长的上下文且减少 90% 的 VRAM 占用**，详见[博客文章](https://unsloth.ai/blog/grpo)和[教程](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)。
   - 此增强功能专为推理模型（reasoning models）设计，可显著节省显存并扩展上下文窗口。
- **QwQ-32B 模型获得更新修复**：**QwQ-32B 模型**已实施错误修复，详情见[博客文章](https://unsloth.ai/blog/qwq-32b)及相应的[模型上传](https://huggingface.co/collections/unsloth/qwen-qwq-32b-collection-676b3b29c20c09a8c71a6235)。
   - 这些修复提升了模型的稳定性和性能，确保更流畅的用户体验。
- **新模型已上传至 Hugging Face**：新上传的模型包括 **Gemma 3 GGUF** 变体（1B, 4B, 12B, 27B）、**Gemma 3 Dynamic 4-bit** 版本、**QwQ-32B** 变体以及 **Phi-4-mini** 版本，均可在 [Hugging Face](https://huggingface.co/collections) 上获取。
   - 这些模型迎合了各种硬件配置和性能需求，扩展了最前沿模型的可访问性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/gemma3#everything)">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的新型多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B、4B、12B 和 27B 尺寸。</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)">Unsloth 文档</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1350054985873887252)** (5 messages): 

> `Gemma 3, Ollama, Phi Vision, GGUFs vision` 


- **关于 Gemma 3 与 Ollama 图像兼容性的疑问**：一位成员询问 **Gemma 3** 是否可以像 **Phi Vision** 一样通过 **Ollama** 处理图像。
   - 另一位用户澄清说，他们的 **Gemma 3 GGUFs vision** 组件在除 **Ollama** 之外的所有引擎上都能运行，包括 **LM Studio** 和 **llama.cpp**；这*可能*是因为 **llama-server** *尚未*支持视觉功能。
- **Ollama 缺乏视觉支持**：**Gemma 3 GGUFs vision** 组件在除 **Ollama** 之外的所有引擎上均可运行，包括 **LM Studio** 和 **llama.cpp**。
   - 这*可能*是因为 **llama-server** *尚未*支持视觉功能。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1350042947114111016)** (51 条消息🔥): 

> `Gemma-3 GGUF 与 Ollama，Llama 3.2 推理取消，Phi-4-mini 支持，Gemma 微调错误，TurboML 持续预训练 (Continual Pre-Training)` 


- ****Gemma-3 GGUF 视觉功能在 Ollama 中失效****：由于一个影响所有上传者的广泛问题，**Ollama** 中带有视觉组件的 **Gemma-3 GGUF** 模型支持目前无法正常工作。
   - 建议在问题调试完成前，使用 **Ollama 原生**的 Gemma 模型进行文本处理。
- ****Llama 3.2 推理需要取消方法****：一位用户询问是否可以在不从内存中卸载模型的情况下，取消 **Unsloth** 中 **Llama 3.2** 模型的长时间运行推理。
   - 用户询问是否可以在超时循环中达到一定时间后停止它。
- ****Phi-4-mini 获得新的 Unsloth 更新****：升级到最新的 **Unsloth** 版本后，用户现在应该可以使用 **Phi-4-mini** 模型，此前该模型会导致 RuntimeError。
   - 用户反馈 **Phi-4-mini** 运行正常，但 `unsloth/Phi-4-mini-instruct` 会报错 `rope_scaling` 错误。
- ****Gemma 微调错误需要更新 Unsloth****：用户报告在添加新 Token 并训练新 Embedding 后微调 **Gemma 模型**时遇到错误（`only Tensors of floating point dtype can require gradients`），特别是在评估阶段。
   - 建议升级到最新版本的 **Unsloth**，尽管有用户报告更新后问题仍然存在。
- ****TurboML 寻求持续预训练 (Continual Pre-Training) 数据集格式指导****：一位使用新框架 **TurboML** 的成员就**持续预训练 (CPT)** 的正确数据集格式寻求建议，特别是针对句子补全和 SFT 任务。
   - 他们参考了一个 [Unsloth notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu)，并询问了 EOS token 的放置位置。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/huggi">huggi - Overview</a>: huggi 有 2 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个检查点继续微调 | Unsloth 文档</a>: 检查点（Checkpointing）允许你保存微调进度，以便暂停后继续。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1350123121059037237)** (9 条消息🔥): 

> `Gemma SFT，最大上下文长度，内存使用计算` 


- **关于 Gemma SFT 显存 (VRAM) 需求的讨论**：成员们讨论了 **Gemma** 在 SFT 中的 VRAM 使用情况，认为在相似的训练条件下，它可能比 **Qwen 2.5** 需要更多的 VRAM，但未给出具体数值。
   - 有人分享了一个用于图像处理的 [Qwen2_VL Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_%25287B%2529-Vision.ipynb)。
- **最大上下文长度是限制而非超参数**：成员们澄清，最大上下文长度不是模型的超参数，而是基于可用内存的**限制**。
   - 上下文越长，处理它所需的内存就越多，但精确计算并不仅仅取决于模型大小。
- **估算 LLM 的内存使用量**：提到处理上下文所需的内存量取决于模型架构，不同的层需要不同数量的内存。
   - 分享了用于估算内存需求的链接（[Substratus AI 博客](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm) 和 [Hugging Face Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)）。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb#scrollTo=idAEIeSQ3xdS">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1350033127891931137)** (263 条消息🔥🔥): 

> `Linux 和 Windows 上的 Cursor 性能问题，Claude 3.7 的问题，Cursor 中的自定义模式，Gemini API 密钥问题，Cursor agent 频繁生成终端` 


- ****Cursor 在 Linux 和 Windows 上卡顿****：用户报告 **Cursor** 在 **Linux** 和 **Windows** 上均出现 **卡顿** 和 **冻结**，即使在高性能硬件上也是如此，尤其是在 **0.47.4** 等近期更新之后（[下载链接](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe)）。
   - 一位用户提到，在 Linux 上发送 **20-30 条消息** 后，UI 会冻结数秒；而另一位用户表示，在配置精良的笔记本电脑上使用 **3.7** 版本的 Windows 版时“一直卡顿”。
- ****Claude 3.7 表现不佳且忽略全局规则****：用户在使用 **Claude 3.7** 时遇到问题，报告称在升级到 **0.47.4** 后它变得 *笨如砖头*，并指出其使用成本是原来的两倍。
   - 一些成员报告说 **Sonnet 3.7** 会忽略全局规则，即使明确要求它输出正在使用的规则也是如此；有人建议 *在 prompt 中加入“做一个乖孩子”，这能解决任何问题*。
- ****Cursor Agent 触发终端海啸****：多名用户发现 Cursor agent 过度生成终端，这被认为很烦人，尤其是当它重启已经在运行的服务器时。
   - 一位成员建议这种行为应该是内置的，或者如果用户不喜欢，就应该自己编写终端命令。
- ****V0 为快速 UI 原型设计带来乐趣****：一些用户发现 **v0** 更适合以前端为中心的原型设计，允许在导入 Cursor 之前进行类似 Figma 的子框架 UI 设计。
   - 一位用户指出：*在我看来，先构建原型和布局（更好的前端）然后本地导入到 Cursor 要好得多*，但也有人因为 v0 的积分制和较少的创意控制权而更倾向于使用 Cursor。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/context/ignore-files">Cursor – Ignore Files</a>：未找到描述</li><li><a href="https://github.com/kcolemangt/llm-router">GitHub - kcolemangt/llm-router: 通过将 llm-router 设置为 Cursor 的 Base URL 来访问来自 OpenAI、Groq、本地 Ollama 等的模型</a>：通过将 llm-router 设置为 Cursor 的 Base URL 来访问来自 OpenAI、Groq、本地 Ollama 等的模型 - kcolemangt/llm-router</li><li><a href="https://x.com/kregenrek/status/1899941361908146430">来自 Kevin Kern (@kregenrek) 的推文</a>：Sonnet 3.7 和 Cursor：我通过遵循这些规则在一定程度上控制住了它。我仍然推荐 - Edit Mode - Sonnet 3.5。引用 Jame (@Inveeest) @kregenrek 刚刚注意到你是创建者...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1350119670551679039)** (2 条消息): 

> `LM Studio, SMILES 字符串编码, ChemDraw` 


- **提倡 LM Studio 支持**：一位成员建议，遇到 **LM Studio** 相关问题的用户可能会在 **LM Studio Discord** 中找到更有针对性的帮助。
- **寻求立体异构体的 SMILES 字符串编码器**：一位成员询问是否存在能够将 **SMILES 字符串** 编码为各种 **立体异构体** 或编码 **ChemDraw** 输入的现有模型或架构，旨在实现化学描述符提取。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1350055159018815500)** (255 条消息🔥🔥): 

> `Diffusion Models 用于生成任务，Search-R1：用于自主搜索查询生成的 RL，潜空间的频谱分析，Diffusion Models 中的噪声敏感性，用于快速采样的 Inductive Moment Matching (IMM)` 


- **Diffusion Models 在生成任务中脱颖而出**：一位成员分享了一篇 [Nature 文章](https://www.nature.com/articles/s41467-024-54281-3)，强调了 **diffusion models (DMs)** 的进展，指出其在建模复杂数据分布以及为图像、视频、音频和 3D 场景生成逼真样本方面的能力。
   - 文章描述了 **diffusion models** 如何成为生成任务中的 state-of-the-art。
- **Search-R1 通过 RL 学习自主搜索**：一位成员分享了一篇 [论文](https://arxiv.org/abs/2503.09516)，介绍了 **Search-R1**，它是 **DeepSeek-R1** 模型的扩展，通过强化学习 (**RL**) 学习在推理过程中生成搜索查询。
   - 该模型通过多轮搜索交互优化 LLM rollouts，使用检索到的 token masking 来实现稳定的 **RL** 训练。
- **潜空间的频谱分析探讨**：成员们讨论了 diffusion models 潜空间的频谱分析，一位成员指出，该模型对初始噪声的扰动具有鲁棒性，即使在 **t=0**（最大噪声）时也是如此。
   - 另一位成员指出，比较使用 **flux VAE** 与 **SDXL VAE** 编码的图像的径向平均功率谱密度（radially averaged power spectral density）并没有什么实际指导意义。
- **噪声敏感性分析揭示方差**：成员们讨论了初始噪声的微小变化如何影响 diffusion models 的输出，一位成员分享了一张图表，显示了 *微小噪声导致巨大差异* 的点。
   - 他们观察到较亮的像素表示初始噪声中的敏感区域，这些区域会导致输出发生显著变化。
- **Inductive Moment Matching (IMM) 有望实现快速采样**：一位成员分享了一篇关于 **Inductive Moment Matching (IMM)** 的 [论文](https://arxiv.org/abs/2503.07565)，这是一种新型生成模型，通过单阶段训练程序实现单步或少步采样，在推理速度上比 diffusion models 更快。
   - 与蒸馏（distillation）不同，**IMM** 不需要预训练初始化和两个网络的优化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.09516">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a>: 高效获取外部知识和最新信息对于大语言模型 (LLMs) 的有效推理和文本生成至关重要。检索增强和工具使用训练...</li><li><a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://arxiv.org/abs/2503.07565">Inductive Moment Matching</a>: Diffusion models 和 Flow Matching 可以生成高质量样本，但推理速度较慢，将它们蒸馏成少步模型通常会导致不稳定和大量的调优。为了解决这些问题...</li><li><a href="https://www.nature.com/articles/s41467-024-54281-3">Dynamical regimes of diffusion models - Nature Communications</a>: Diffusion 方法广泛用于 AI 应用中的数据生成。在这里，作者展示了经过优化训练的 diffusion models 表现出三种动力学机制：从纯噪声开始，它们达到...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1350041171766022178)** (195 条消息🔥🔥): 

> `Ministral 8B, Exaone 8B, Jungle Chess AI, Stable Diffusion, Gemini 2.0` 


- **Mistral 8B 和 Exaone 8B 推荐**：对于寻找 LLM 的用户，成员们推荐使用 4-bit 量化的 **Ministral 8B** 或 **Exaone 8B**。
   - 一位拥有配备 24 GB RAM 的 M4 Mac mini 的用户询问了预期的 tokens per second，但确切的性能仍取决于硬件规格，目前处于推测阶段。
- **用户尝试训练斗兽棋 (Jungle Chess) AI**：一位用户尝试使用 o3-mini 创建一个斗兽棋 AI，但该 AI 无法理解或遵循与其 alpha-beta 算法相关的 Bug 报告，详见[此讨论帖](https://chatgpt.com/share/67d436c6-e2bc-8006-a450-41edf4acfac9)。
   - 用户指出它可以达到深度 6，但无法避开简单的开局陷阱，而深度 3 或 4 应该就足够了，这表明 [Chinese Engine](https://gitee.com/WZ403809264/animalcraftAI/releases) 在这方面表现更好。
- **聊天机器人国际象棋锦标赛出现！**：一位用户分享了一个名为 *Chatbot Chess Championship 2025* 的 [YouTube 播放列表](https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC)，展示了语言模型或国际象棋引擎对弈的情况。
   - 参与者推测这些模型是真正的语言模型还是仅仅在调用国际象棋引擎，有人注意到一个语言模型做出了非法移动。
- **用户寻找 Stable Diffusion 模型**：一位用户请求帮助寻找在 LAION 以外的数据集上进行微调的 **Stable Diffusion v1.x 模型**。
- **寻找 Gemini 2.0 Flash 开源模型**：一位成员询问是否存在类似于 **Gemini 2.0 Flash** 且具有用于图像编辑的 text-plus-image-to-image 能力的开源模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC">Chatbot Chess Championship 2025</a>: Hi</li><li><a href="https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamShapeError)">flax.errors package</a>: 未找到描述</li><li><a href="https://gitee.com/WZ403809264/animalcraftAI/releases">animalcraftAI 发行版 - Gitee.com</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

ilyachua: 大家好。我正在开始学习 Hugging Face 的 CV 课程。
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1350200457578020924)** (2 条消息): 

> `Awesome Vibe Coding, mahimairaja/awesome-csm-1b` 


- **Awesome Vibe Coding 列表发布**：一个名为 [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) 的精选列表已经发布，其中包含用于 **AI 辅助编程** 的工具、编辑器和资源。
   - 该列表包括 **AI 驱动的 IDE**、**基于浏览器的工具**、**插件**、**命令行工具**以及关于 *vibe coding* 的**最新动态**。
- **CSM 1B 使用案例精选**：一个名为 [awesome-csm-1b](https://github.com/mahimairaja/awesome-csm-1b) 的精选列表已经发布，收集了使用 **Sesame 的 CSM 1B** 构建的使用案例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: 一个精选的 vibe coding 参考列表，与 AI 协作编写代码。</a>: 一个精选的 vibe coding 参考列表，与 AI 协作编写代码。 - filipecalegario/awesome-vibe-coding</li><li><a href="https://github.com/mahimairaja/awesome-csm-1b">GitHub - mahimairaja/awesome-csm-1b: 使用 Sesame 的 CSM 1B 构建的精选使用案例列表</a>: 使用 Sesame 的 CSM 1B 构建的精选使用案例列表 - mahimairaja/awesome-csm-1b
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1350133225573453866)** (1 条消息): 

> `generate_without_kv_cache function` 


- **函数名中发现拼写错误**：一位用户指出，文章中的函数名是 **generate_without_kv_cache**，但实际使用的函数调用是 **generate_with_kv_cache**。
   - 未提供进一步的讨论。
- **函数调用不一致**：文章提到一个名为 **generate_without_kv_cache** 的函数，但实际代码使用了 **generate_with_kv_cache**，这表明可能存在错误。
   - 如果用户直接从文章中复制函数调用，这种不一致可能会导致困惑或错误使用。

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1350141638709743667)** (2 messages): 

> `PaliGemma 2 Mix, smolVLM2, QwenVL, Llama 3.2 Multimodal` 


- **Google 发布 PaliGemma 2 Mix 模型**：Google 发布了 **PaliGemma 2 Mix**，这是一个全新的多功能视觉语言模型系列，包含三种尺寸（**3B**、**10B** 和 **28B**）以及 **224** 和 **448** 两种分辨率，能够通过 *开放式提示词（open-ended prompts）执行视觉语言任务* 并理解文档 ([博客文章](https://huggingface.co/blog/paligemma2mix))。
- **Raschka 深入探讨多模态 LLM**：Sebastian Raschka 解释了多模态 LLM 的运作机制，并回顾了最近的多模态论文和模型，包括 **Meta AI 的 Llama 3.2** ([博客文章](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html))。
- **SmolVLM2 架构深度解析**：想要从根源理解 **SOTA 架构** 的用户，应在研究 **smolVLM2** 和 **QwenVL** 之前，先从 **CLIP** 和 **BLIP2** 入手。
   - 或者，也可以先学习最新的模型，然后逆向溯源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/merve">merve (Merve Noyan)</a>：未找到描述</li><li><a href="https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html">Understanding Multimodal LLMs</a>：多模态 LLM 领域出现了大量新研究，包括最新的 Llama 3.2 视觉模型，它采用了多种架构策略来整合文本等各种数据类型...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1350037931112398900)** (25 messages🔥): 

> `SerpAPI Key Errors, Deep RL Course, Interactive IDEs for Agent Code, Image to Video Loops, Gemma3 Issues with SmolAgents` 


- **SerpAPI Key 引发困扰**：一名成员报告了其 **SerpAPI key** 可能存在的错误，并询问其他人是否遇到同样的问题。
   - 另一名成员澄清道，用户需要提供自己的 **SerpAPI key**，这在课程示例中并没有立即说明清楚。
- **Deep RL 课程引发提问**：几位成员询问了 Discord 服务器中关于 **Deep RL 课程** 的专区设置，并报告 [Deep Reinforcement Learning Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard) 无法正常工作。
   - 一名成员提到他们也遇到了同样的问题，并且最近刚开始学习 Deep RL 课程。
- **寻求交互式 IDE 推荐**：一名成员征求能够为编写 Agent 代码提供建议的 **交互式 IDE** 推荐。
   - 另一名成员推荐 **VS Code** 作为一个功能强大且免费的选择。
- **图像循环生成引发模型搜索**：一名成员请求推荐能够将 **1920x1080 图像转换为 3-5 秒视频循环** 且能在 **H100** 上运行的模型。
   - 他们指出很难找到 **720p** 以上分辨率的选项。
- **Gemma3 在使用 SmolAgents 时遇到的问题**：一名成员在配合 **SmolAgents** 运行 **Gemma3** 时遇到错误，特别是与代码解析和正则表达式模式相关的错误，并链接到了 [GitHub](https://github.com/huggingface/smolagents/pull/883) 上的一个潜在修复方案。
   - 他们通过增加 **Ollama 上下文长度** 解决了该问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fhuggingface-projects%2FDeep-Reinforcement-Learning-Leaderboard">Weiterleitungshinweis</a>：未找到描述</li><li><a href="https://open.spotify.com/playlist/4J61XoHr2CINqRA1DV0ga7">Party on, Wayne!</a>：播放列表 · Music League · 17 首曲目 · 4 次保存</li><li><a href="https://github.com/huggingface/smolagents/pull/883">Update code_agent.yaml to fix persistent SyntaxErrors by toadlyBroodle · Pull Request #883 · huggingface/smolagents</a>：修复了由于 CodeAgents 在 py 代码块前添加 ``` 导致的永久性 SyntaxErrors 和代码解析错误（代码块无效）。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1350042931012177941)** (213 条消息🔥🔥): 

> `Complexity Extension 问题，Kernel 锁定，Perplexity 上下文窗口大小，Grok 3 bug，Gemini 的深度研究` 


- **Complexity Extension 进入全面维护模式**：由于新的 Perplexity 布局破坏了 [Complexity extension](https://github.com/danielcardeenas/perplexity-complexity)，它现在已进入全面维护模式。
   - 开发者感谢用户的耐心等待。
- **关于 Kernel 锁定以提高安全性的辩论**：一些用户建议为了安全而**锁定 Kernel**，但其他人认为由于 Linux 的**开源特性**，这是不可能实现的。
   - 一位用户讽刺地调侃道：*如果反作弊系统决定制作并强制执行自定义的 Linux Kernel 构建……到那时你还不如直接用 Windows 呢，哈哈*。
- **上下文窗口大小仍是 Perplexity 的痛点**：用户恳求**增加上下文窗口大小**，这样他们就可以停止为 ChatGPT 付费了。
   - 一位用户表示，他们愿意为更大的上下文窗口支付额外费用，因为 Perplexity 拥有其他产品不具备的功能：*能够同时对 50 个文件进行无限研究*、*最棒的是 Spaces，我们可以在其中提供自定义指令以及 50 个可上传的知识文件*，以及*选择 Reasoning 模型（推理模型）的选项*。
- **Grok 3 发布时饱受 Bug 困扰**：用户报告称新发布的 **Grok AI** 存在许多 Bug。
   - 报告的 Bug 包括 *聊天突然停止工作或在中间中断*。
- **Gemini 的新深度研究功能不足**：一些用户测试了新的 **Gemini Deep Research**，发现与 **OpenAI** 提供的功能相比显得较弱。
   - 一位用户发现，即使在关闭搜索的情况下，它保留的上下文也比普通 Gemini 少。



**提到的链接**：<a href="https://xkcd.com/1053/">Ten Thousand</a>：未找到描述

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1350051629554077772)** (3 条消息): 

> `OpenAI 自定义 Agent，Airpods 实时翻译，Anthropic CEO AI 退出按钮` 


- **OpenAI 首次推出自定义 Agent**：Perplexity 链接到 [OpenAI 发布新的自定义 Agent](https://www.perplexity.ai/page/openai-releases-new-custom-age-0SxjY06OTReWo9OK_gjQCA)。
   - 暂无讨论，但对于想了解的人来说是一个潜在有趣的链接。
- **Airpods 将引入实时翻译**：Perplexity 链接到 [Airpods 将引入实时翻译](https://www.perplexity.ai/page/airpods-to-introduce-live-tran-UFQuA8yaRY..k0Qwm3MOZw)。
   - 暂无讨论，但对于想了解的人来说是一个潜在有趣的链接。
- **Anthropic CEO 提出 AI 退出按钮**：Perplexity 链接到 [Anthropic CEO 提出 AI 退出按钮](https://www.perplexity.ai/page/anthropic-ceo-floats-ai-quit-b-BotCYKfST6GePBfE_Psp6w)。
   - 暂无讨论，但对于想了解的人来说是一个潜在有趣的链接。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1350035269071540235)** (133 条消息🔥🔥): 

> `Claude 配合 Aider, 为 Aider 使用 Rust, Linux 上的 Claude Desktop, Aider MCP Server, Anthropic 状态` 


- ****Claude** 与 **Aider** 强强联手助力编程！**: 成员们讨论了将 **Claude** 与 **Aider** 结合使用，这通过 *网页搜索/URL 抓取和运行 bash 脚本调用来增强它*，从而实现了更强大的提示能力。
   - 一位用户强调，*添加到 Claude 的每个独特工具所释放的潜力远超其各部分之和*，特别是当模型在互联网上搜索 bug 时。
- ****Rust** 能让 **Aider** 运行得更快吗？**: 一位用户询问是否可以将 **Aider** 移植到 **C++** 或 **Rust** 以实现更快的文件处理，特别是在为 **Gemini** 模型加载大型上下文文件时。
   - 其他人表示怀疑，认为瓶颈仍然在于 **LLM API**，任何改进可能都无法 *量化*。
- ****Linux** 用户转向 GitHub 寻求 **Claude Desktop** 应用**: 用户分享了在 **Linux** 上运行 **Claude Desktop** 应用的指令，因为目前还没有官方版本。
   - 一位用户引用了一个 [GitHub 仓库](https://github.com/aaddrick/claude-desktop-debian)，提供了基于 **Debian** 的安装步骤，而另一位用户分享了他们对 **Arch Linux PKGBUILD** 的修改。
- **Aider MCP Server 的 Readme 需要帮助！**: 用户讨论了 **Aider MCP Server**，其中一人提到另一位用户的 readme 写得 *好 100 倍*，指的是 [这个仓库](https://github.com/sengokudaikon/aider-mcp-server)。
   - 然而，另一位用户幽默地表示，尽管有 readme，他们仍然 *无法设置你的 mcp*。
- **Anthropic 3.7 Sonnet 遇到小问题**: 用户报告了来自 **Claude 3.7 Sonnet** 模型的 *空响应*，促使他们检查自己的 **Anthropic** 账户。
   - [Anthropic 状态页面](https://status.anthropic.com/) 确认了 *错误率升高*，表明问题正在调查中，并正在实施修复。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/recordings/tree-sitter-language-pack.html">通过 tree-sitter-language-pack 添加语言支持</a>：aider 是你终端里的 AI 配对编程伙伴</li><li><a href="https://asciinema.org/">以简单的方式记录并分享你的终端会话 - asciinema.org</a>：未找到描述</li><li><a href="https://aider.chat/docs/recordings/">屏幕录像</a>：aider 构建 aider 的屏幕录像。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: 全能型 LLM CLI 工具，具有 Shell 助手、Chat-REPL、RAG、AI 工具和 Agent 功能，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。</a>：sigoden/aichat</li><li><a href="https://github.com/sengokudaikon/aider-mcp-server">GitHub - sengokudaikon/aider-mcp-server</a>：通过在 GitHub 上创建账户来为 sengokudaikon/aider-mcp-server 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=NTh0hYbfpis">Claude 使用 aider mcp</a>：拒绝炒作，只用开源工具：github.com/robert-at-pretension-io/mcp</li><li><a href="https://github.com/aaddrick/claude-desktop-debian.git">GitHub - aaddrick/claude-desktop-debian: 适用于基于 Debian 的 Linux 发行版的 Claude Desktop</a>：aaddrick/claude-desktop-debian</li><li><a href="https://tenor.com/view/fedora-tipshat-mlady-melady-athiest-gif-7191305">Fedora Tipshat GIF - Fedora Tipshat Mlady - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1350039436615749694)** (44 条消息🔥): 

> `DeepSeek 模型配置，Aider 的 architect 模式行为，修改 Aider 的 completion 终端节点，Aider 配置文件` 


- **DeepSeek 模型话太多**：一位用户报告称 **DeepSeek 模型** 生成了过多的输出，大约有 **20-30 行** 短语，并询问是否可以在配置中设置 `thinking-tokens` 值。
   - 有人指出，对于 **R1 模型** 来说，**20 行** 是非常标准的，一位用户分享说，他们曾为一个 **5 个单词的提示词** 等待了 **2 分钟** 的模型思考。
- **Architect 模式无限计划**：一位用户在使用 **Aider 的 architect 模式** 时遇到了问题，模型会不断地进行计划，而不将代码更改传递给编辑器，即使在被提示 *进行代码更改* 之后也是如此。
   - 建议用户可能需要事先显式添加文件，和/或 *列出可能受影响的文件和函数*。
- **为 OpenWebUI 修改 Aider API 调用**：一位用户询问如何修改 **Aider** 调用 completions 终端节点的方式，以便与 **OpenWebUI** 的知识库（knowledge collections）集成，这需要一个带有集合 ID 的 `files` 参数，参考了 [OpenWebUI API 文档](https://docs.openwebui.com/getting-started/api-endpoints/#using-a-knowledge-collection-in-chat-completions)。
   - 建议使用 `extra_params` 或 `extra_body` 配置选项来添加必要的参数。
- **全局 vs 本地 `ai-instructions.md` 文件**：一位用户询问 `ai-instructions.md` 文件应该放在每个项目中，还是可以配置一个单一的全局文件。
   - 回复澄清说，这些文件（如 `conventions.md`）可以根据用户喜好进行处理，建议个人使用时采用全局文件，而针对特定项目的规范则采用本地文件。
- **配置 OpenRouter API Key**：一位用户需要帮助来为 **Aider** 配置 **OpenRouter API key**。
   - 一位成员展示了正确的配置格式 `api-key: - openrouter=sk-or-v1-...`


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://docs.openwebui.com/getting-started/api-endpoints/#using-a-knowledge-collection-in-chat-completions">🔗 API 终端节点 | Open WebUI</a>：本指南提供了有关如何有效与 API 终端节点交互，以使用我们的模型实现无缝集成和自动化的基本信息。请注意，这是一个实验性的...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1350034619554070560)** (17 条消息🔥): 

> `OLMo 2 32B, AI Engineer Singapore 2025, AI 游戏生成, Gemini DeepResearch with 2.0, Claude 的生日` 


- ****OLMo 2 32B** 击败了 **GPT 3.5** 和 **GPT 4o mini**！**: AI2 发布了 [**OLMo 2 32B**](https://allenai.org/blog/olmo2-32B)，这是一个完全开源的模型，使用 Tulu 3.1 在高达 **6T tokens** 的数据上训练，在学术基准测试中表现优于 **GPT3.5-Turbo** 和 **GPT-4o mini**。
   - 据称其训练成本仅为 **Qwen 2.5 32B** 的三分之一，同时达到了类似的性能，并提供 **7B**、**13B** 和 **32B** 参数版本。
- ****AI Engineer Singapore 2025** 活动宣布**: [AI Engineer Singapore 2025](https://lu.ma/aiesg?tk=fHwK70) 活动宣布举办，旨在弥合前沿 **AI research** 与实际工程应用之间的鸿沟。
   - 该活动由 **AI Eng Summit**、**World's Fair**、**JSConf Asia** 和 **GovTech Singapore** 背后的团队组织。
- **Vibe Coding：完全使用 AI 创建游戏**: 一位开发者 **100% 使用 AI** 创建了一款多人 3D 游戏，耗时 20 小时，花费 20 欧元，并称这一概念为 *vibe coding*，同时分享了[指南](https://x.com/nicolaszu/status/1899931187398979890?s=46)。
   - 该游戏具有击中反馈、受损冒烟和死亡爆炸等逼真元素，全部通过在 **Cursor** 中输入 prompt 生成，无需手动修改代码。
- ****Gemini DeepResearch 2.0** 广受好评**: 成员们反馈新的 **Gemini DeepResearch with 2.0** 模型效果非常好。
   - 一位成员指出，*与 ChatGPT deep research 相比，它非常适合公司 OSINT（开源情报），因为它拒绝回答问题的情况要少得多*。
- **Claude，生日快乐！**: 成员们庆祝了 [**Claude** 的两周岁生日](https://x.com/alexalbert__/status/1900592059364634973?s=46)。
   - 这恰好也与 **GPT-4** 的生日重合。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/joshwoodward/status/1900201110717214914?s=46">Josh Woodward (@joshwoodward) 的推文</a>: 下一批 @NotebookLM 更新正在推出：* 由 Gemini 2.0 Thinking 驱动的更智能的回答 * 在笔记中查看引用，而不仅仅是在问答中（首要需求）* 自定义用于制作...的来源</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973?s=46">Alex Albert (@alexalbert__) 的推文</a>: 两年前的今天，我们向世界宣布了 Claude。两周年生日快乐，Claude！</li><li><a href="https://x.com/natolambert/status/1900249099343192573?s=46">Nathan Lambert (@natolambert) 的推文</a>: 开源 AI 非常激动人心的一天！我们正在发布迄今为止最大的开源模型 —— OLMo 2 32B —— 它击败了最新的 GPT 3.5、GPT 4o mini 以及领先的开放权重模型如 Qwen 和...</li><li><a href="https://x.com/levelsio/status/1900235423667909041?s=46">@levelsio (@levelsio) 的推文</a>: 从一个游戏到另一个游戏的传送门！引用 Josua Sievers (@SieversJosua)：我添加了第一个传送门！你现在可以传送到 @levelsio 的游戏中！如果有一种方法可以实际立即生成一个...</li><li><a href="https://x.com/nicolaszu/status/1899931187398979890?s=46">Nicolas Zullo (@NicolasZu) 的推文</a>: 发布了！！vibe coding 游戏的终极指南。我是怎么做到的？20 小时。500 个 prompt。20 欧元。这就是 100% 使用 AI 制作多人 3D 游戏所需要的全部，0 人工代码，甚至没有一小段...</li><li><a href="https://allenai.org/blog/olmo2-32B">OLMo 2 32B：首个性能超越 GPT 3.5 和 GPT 4o mini 的完全开源模型 | Ai2</a>: 介绍 OLMo 2 32B，OLMo 2 家族中功能最强、规模最大的模型。</li><li><a href="https://lu.ma/aiesg?tk=fHwK70).">AI Engineer Singapore 2025 · Luma</a>: 加入我们的 AI Engineer Singapore 2025，这是一个权威的以行业为中心的活动，旨在补充以研究为重点的国际会议……
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1350226688595853403)** (3 messages): 

> `Snipd Podcast, AI Podcast App, Latent Space Podcast` 


- ****Snipd Podcast** 户外发布！**: Latent Space 播客发布了与 Kevin Ben Smith 合作的新一期 [Snipd podcast](https://x.com/latentspacepod/status/1900666708270215383)，讨论用于学习的 **AI Podcast App**。
   - 播客内容包括关于 aidotengineer NYC 的讨论、从 **Finance 转向 Tech** 的经历、AI 如何帮助我们从播客时间中获得更多收益，并详细介绍了 **Snipd app 的 tech stack**。
- **Latent Space Podcast 发布 Youtube 视频**: Latent Space Podcast 在 [Youtube](https://youtu.be/FNRO_SYx68Q) 上发布了他们首个 **户外（OUTDOOR）播客**。
   - 该播客讨论了 @aidotengineer NYC、从 **Finance 转向 Tech**、AI 如何提升播客效率，以及 **@snipd_app 的 tech stack** 细节。



**提及的链接**: <a href="https://x.com/latentspacepod/status/1900666708270215383">来自 Latent.Space (@latentspacepod) 的推文</a>: 🆕 Snipd: 用于学习的 AI Podcast App https://youtu.be/FNRO_SYx68Q 我们首个户外播客！@swyx 和 @KevinBenSmith 聊到了 @aidotengineer NYC，从 Finance 转向 Tech，AI 如何...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1350196967078367254)** (120 messages🔥🔥): 

> `Cursor vs Claude, Levelsio flight sim, GitDoc VS Code extension, Vibe Coding IDE UI, Auto-git commit` 


- **Claude 3.5 还是 3.7：哪种 Vibe 胜出？**: 成员们讨论了 **Claude 3.5** 和 **3.7** 之间的差异，有些人觉得 **3.7** *过于热心*，容易*多做 20 件我没要求的事*。
   - 其他人则有意识地在 coding 和 debugging 中使用这两种不同的模型和工作流——其中一人发现 *vibe debugging* 比较困难。
- **Levels.io 飞行模拟器走红**: 一位成员提到了 [Levels.io 飞行模拟器](https://x.com/levelsio/status/1893350391158292550) 的成功，该项目使用 **Cursor** 构建，通过在游戏中出售广告，迅速达到了 **100 万美元 ARR**。
   - Levelsio 指出：*AI 对我来说确实是创意和速度的放大器，让我变得更有创意、更快速*。
- **GitDoc 扩展在保存时提交**: 成员们分享了 [GitDoc VS Code 扩展](https://github.com/lostintangent/gitdoc)，它允许你编辑 Git repo 并在每次更改时 auto commit。
   - 一位用户表示 *存储很便宜，就像在每次更改时 auto commit 并可视化变更树（tree of changes）一样*，并建议增加 branching、restarting 等功能。
- **Vibe Coding IDE 需要 UI 创新**: 成员们讨论认为，传统的 IDE 可能不是 vibe coding 的理想 UI，建议需要一种由不同聊天提示触发的 **变更树可视化（visualization of the tree of changes）**。
   - 这将允许用户轻松回滚到以前的状态并尝试 branching。
- **企业级 AI 开发团队赋能**: 一位成员提议讨论 **企业级 AI 开发团队赋能**，重点关注在大型组织中采用 Cursor 等工具所涉及的“障碍”和繁琐流程（red tape）。
   - 一些人表示有兴趣了解将 AI 集成到企业开发工作流中的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/lostintangent/gitdoc">GitHub - lostintangent/gitdoc: VS Code extension that allows you to edit a Git repo, like it&#39;s a multi-file, versioned document.</a>: VS Code 扩展，允许你像编辑多文件、版本化文档一样编辑 Git repo。- lostintangent/gitdoc</li><li><a href="https://x.com/levelsio/status/1893350391158292550">来自 @levelsio (@levelsio) 的推文</a>: ✨ 今天我想，如果我让 Cursor 构建一个飞行模拟器会怎样。所以我要求“在浏览器中制作一个带有摩天大楼的 3D 飞行游戏”。经过我的多次提问和评论，我现在拥有了...</li><li><a href="https://x.com/levelsio/status/1899596115210891751">来自 @levelsio (@levelsio) 的推文</a>: ✨ http://fly.pieter.com 现在仅用 17 天就从 $0 增长到 100 万美元 ARR！💸 收入更新：$87,000 MRR（即 $1M ARR）。这是我第一个增长如此之快的项目 🤯 目前仅剩 3 个广告位：https://...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1350042362629591050)** (92 条消息🔥🔥): 

> `下载 LM Studio 运行时, Snapdragon X Plus 支持, Gemini 视觉能力, AI 象棋锦标赛, Gemma 3 的 VRAM 占用` 


- ****搜寻运行时**：用户反编译 LM Studio 以寻找下载 URL**：一名用户寻求下载 **LM Studio runtimes** 以供离线使用，询问程序从何处下载它们，此前另一名用户告诉他 *LM Studio 不需要互联网连接即可运行*。
   - 该用户反编译了应用并找到了运行时 URL，包括 [backends master list](https://extensions.lmstudio.ai/backends-master-list-stable.json) 和 CDN "APIs"，如 [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz)。
- ****Snapdragon 障碍**：LM Studio 的兼容性难题**：一名用户询问 **LM Studio 对 Snapdragon X Plus 的支持**，报告称 LM Studio 未检测到其 GPU，另一名成员回复说，为了获得 GPU 支持，*你需要直接运行 llama.cpp*，并参考此链接 [github.com/ggml-org/llama.cpp/pull/10693](https://github.com/ggml-org/llama.cpp/pull/10693)。
- ****Gemini 的地理限制**：位置锁定限制了视觉功能**：一名用户请求协助测试 **Gemini 2.0 Flash Experimental** 处理图像的能力，并指出在德国/欧盟可能存在区域限制，因为 *它在德国似乎无法工作（可能是欧盟，我猜是因为这里的法律？）*。
   - 一名美国的测试者在 AI Studio 和 Gemini 应用中测试了 Gemini，发现它未能执行请求的图像处理。
- ****将死狂热**：AI 象棋锦标赛展示模型准确率**：举办了一场 AI 象棋锦标赛，共有 **15 个模型** 相互竞争，结果和详情可在 [dubesor.de/chess/tournament](https://dubesor.de/chess/tournament) 查看。
   - 一名用户注意到 **DeepSeek-R1** 的准确率为 **92%**，但组织者澄清说，准确率取决于比赛时长和对手的走法，且普通的 O1 在锦标赛中运行成本太高。
- ****VRAM 吞噬者**：更新后 Gemma 3 的胃口增加**：一名用户报告称，在视觉速度提升更新后，其 **Gemma 3 的 VRAM 占用** 显著增加。
   - 另一名用户推测下载体积增加可能是因为用于视觉的 **CLIP** 位于独立文件中，他们认为它可能没有嵌入在下载包中，而是在上传到 LM Studio 时从独立文件调用的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dubesor.de/chess/tournament">Dubesor LLM 象棋锦标赛</a>：未找到描述</li><li><a href="https://extensions.lmstudio.ai/backend-llama.cpp-win-x86_64-nvidia-cuda-avx2-1.21.0.tar.gz">无标题</a>：未找到描述</li><li><a href="https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz">无标题</a>：未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/10693">由 lhez 引入的实验性 OpenCL 后端，支持 Qualcomm Adreno GPU · Pull Request #10693 · ggml-org/llama.cpp</a>：此 PR 为 Adreno GPU 引入了一个新的实验性 OpenCL 后端。通过 OpenCL，我们可以利用广泛应用于许多移动设备的 Adreno GPU 的计算能力，从而允许...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1350049149827022920)** (44 条消息🔥): 

> `memtest_vulkan, H100 rental t/s, Corsair product quality, 4090 vs A6000, RTX8000` 


- **使用 **Memtest Vulkan** 测试 VRAM 稳定性**：一名成员建议使用 [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan)（一个 **Vulkan 计算工具**）来测试显存稳定性。
- **租赁 **H100** 进行 Token 速度测试**：一名成员租赁了 **H100**，以测量在 **gemma3 27B** 等模型上实现的 **tokens per second**。
- **据报道 **Corsair** 的产品质量有所下降**：用户反映 **Corsair** 的产品质量近年来有所下降。
   - 一位用户报告称，在更换为 **9800X3D** 时，*一套 Corsair RAM 套条是 DOA*（到货即损）。
- **为了可靠性选择 **A6000** 而非改装版 **4090****：成员们讨论了是以 3500 美元购买本地二手 **A6000**，还是以 4100 美元在 eBay 上购买中国改装的 **4090 48GB**。
   - 共识倾向于选择 **A6000**，因为它有厂商质保和*已知的可靠性，并称 4090 是一场赌博*。
- ****RTX8000** 是一个可行的替代方案**：一名成员指出，如果只需要显存容量，可以用与 **A6000** 相同的价格购入 **两块 RTX8000 48GB**。
   - 然而，另一位成员警告说，**RTX8000** 使用的是较旧的 **Turing 架构**，可能会在较新的图像生成模型和训练中出现问题，但对于*纯 LLM 推理*可能没问题。



**提到的链接**：<a href="https://github.com/GpuZelenograd/memtest_vulkan">GitHub - GpuZelenograd/memtest_vulkan: Vulkan compute tool for testing video memory stability</a>：用于测试显存稳定性的 Vulkan 计算工具 - GpuZelenograd/memtest_vulkan

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1350040582109532181)** (120 条消息🔥🔥): 

> `ElizaOs API 框架, Helius API 密钥定价, Quicknode API 密钥定价, DeepHermes-3-Mistral-24B-Preview-4bit MLX, Hermes-3-Llama-3.1-70B-FP8 vllm 参数` 


- **DeepHermes 3 获得 MLX 转换**：模型 [mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit](https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit) 已使用 mlx-lm 版本 **0.21.1** 从 [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview) 转换为 MLX 格式。
- **深入探讨 Hermes 3 的 VLLM 参数**：成员们正在分享不同的配置，以使 **vllm** 在 **Hermes-3-Llama-3.1-70B-FP8** 上正常运行，包括建议为 Hermes 3 70B 添加 `--enable-auto-tool-choice` 和 `--tool-call-parser`。
   - 一位成员指出，tokenizer 中需要 ``<tool_call>`` 和 ``</tool_call>`` 标签，这些标签存在于 **Hermes 3 模型**中，但不一定存在于 **DeepHermes** 中。
- **Vultr 的极早期 Alpha 版推理定价**：一位来自 Vultr 的成员分享了其推理 API 的官方定价，最初包括 **10 美元 5000 万个输出 token**，之后为 **每百万输出 token 2 美分**。
   - 进一步解释称，这源于购买了“多得离谱”的 GH200 显卡并需要对其进行利用，目前在 [https://api.vultrinference.com/](https://api.vultrinference.com/) 提供了一个兼容 OpenAI 的端点。
- **动态 LoRAs 接入 VLLM**：成员们讨论了在 **vllm** 中托管动态 **LoRAs** 以用于各种用例（如保持最新的代码风格）的可能性。
   - 建议允许用户传入其 Hugging Face 仓库 ID 作为 **LoRAs**，并将其提供给 **VLLM serve 命令的 CLI 参数**，此处有指向 [vLLM 文档](https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters) 的链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit">mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters">LoRA Adapters &#8212; vLLM</a>: 未找到描述</li><li><a href="https://api.vultrinference.com/">Vultr Inference API</a>: 未找到描述</li><li><a href="https://github.com/google/minja">GitHub - google/minja: A minimalistic C++ Jinja templating engine for LLM chat templates</a>: 一个用于 LLM 聊天模板的极简 C++ Jinja 模板引擎 - google/minja</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jbbwc2/this_week_did_not_go_how_i_expected_at_all/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://hub.docker.com/r/drikster80/vllm-gh200-openai">无标题</a>: 未找到描述</li><li><a href="https://github.com/substratusai/kubeai">GitHub - substratusai/kubeai: AI Inference Operator for Kubernetes. The easiest way to serve ML models in production. Supports VLMs, LLMs, embeddings, and speech-to-text.</a>: Kubernetes 的 AI 推理算子。在生产环境中提供 ML 模型服务的最简单方法。支持 VLM, LLM, embedding 和语音转文字。 - substratusai/kubeai
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1350038946767179899)** (90 条消息🔥🔥): 

> `Astro 客户端的 MCP、MCP 服务器与架构、Windows 11 上的 Gitlab MCP 服务器、Agentic Coder 转换为 MCP、多 Agent 系统（Swarm vs Mesh vs Sequence）` 


- **Astro 客户端准备接入 MCP**：一位成员计划在其 **Astro** 客户端中使用 MCP，专门为客户提供 MCP 功能，并探索将 MCP 服务器添加到单个项目中以提高客户端可见性的可能性。
   - 他们正在考虑使用 **AWS API Gateway**，将每个 MCP 服务器作为 **Lambda** 函数，并利用带有 **SSE gateway** 的 MCP 桥接。
- **MCP 服务器架构疑问**：一位成员询问了 MCP 服务器架构，指出像 **Cursor** 和 **Cline** 这样的客户端将 MCP 服务器保留在客户端，并询问这些服务器如何与后端通信。
   - 该成员被引导至特定频道以获取更多信息。
- **智能代理服务器：MCP 的 Agent 级子路由**：成员们讨论了创建一个“智能代理” MCP 服务器，该服务器通过自然语言将具有多个工具的标准 MCP 服务器简化为仅含单个工具的服务器，并通过其自身的 LLM 将其转换为特定的工具调用。
   - 这是一种使用 *vector tool calling*（向量工具调用）的子 Agent 方法，使单个 Agent 拥有一组工具子集，**OpenAI Swarm framework** 也遵循类似的流程。
- **通过 MCP 服务器调试网页**：一位成员分享了他们的项目，这是一个使用 MCP 配合 LLM 调试网页的调试器，最初使用 **Puppeteer** 构建，后来移植到了 **Playwright**：[chrome-debug-mcp](https://github.com/robertheadley/chrome-debug-mcp)。
   - 该成员仍在测试 **Playwright** 版本，并计划随后更新 GitHub 仓库。
- **Swarm vs Mesh 多 Agent 系统**：讨论涉及了分层 Agent 系统的替代方法，用户被引导至 **Swarm** vs **Mesh** vs **Sequence** 架构等资源，重点介绍了 **Swarm framework** 如何在 Agent 之间移交单一执行线程。
   - 值得注意的是，OpenAI 现在支持并维护 *swarm* 概念，并将其更名为 **openai-agents**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/robertheadley/chrome-debug-mcp">GitHub - robertheadley/chrome-debug-mcp: An MCP server to allow you to debug webpages using LLMs</a>：一个允许你使用 LLM 调试网页的 MCP 服务器 - robertheadley/chrome-debug-mcp</li><li><a href="https://news.ycombinator.com/item?id=43177117">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1350039030573699103)** (3 条消息): 

> `MCP 服务器管理、Awesome Vibe Coding` 


- **MCP Hub 概念诞生！**：为了解决企业采用 **MCP** 的疑虑，一位成员构建了 **MCP Hub** 概念，其中包含一个仪表板，用于简化服务器连接、控制访问权限并提供跨 **MCP** 服务器的可见性，如[此视频](https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing)所示。
- **Awesome Vibe Coding 列表发布！**：一位成员宣布了 **Awesome Vibe Coding**，这是一个精选的工具、编辑器和资源列表，旨在增强 AI 辅助编程，可在 [GitHub](https://github.com/filipecalegario/awesome-vibe-coding) 上获取。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: A curated list of vibe coding references, collaborating with AI to write code.</a>：一个精选的 Vibe Coding 参考列表，与 AI 协作编写代码。 - filipecalegario/awesome-vibe-coding</li><li><a href="https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing).">Discord Demo - MCP Hub - 2025.03.13.mov</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1350031278505852929)** (57 条消息🔥🔥): 

> `ZIRP 时代遗憾、AI Startup 估值、DeepSeek 护照上缴、Long Context 评估挑战、Xet 数据分块技术` 


- **OpenAI 和 Anthropic 为 ZIRP 时代流泪**：成员们开玩笑说 **OpenAI 和 Anthropic** 正在后悔没有赶上 **ZIRP 时代**，并引用了 [Gigachad meme](https://tenor.com/view/gigachad-chad-gif-20773266)。
   - 另一位成员调侃道，所有 **AI Startup 的融资估值让 ZIRP 时代看起来就像幼儿园水平**。
- **《纽约时报》展望 AGI 未来**：一篇来自未来的 [New York Times 文章](https://www.nytimes.com/2025/03/14/technology/why-im-feeling-the-agi.html)（2025 年 3 月 14 日）指出，**AI 系统**已开始在**数学**、**编程**和**医疗诊断**等领域超越人类。
   - 文章预计，一家或多家 **AI 公司**将在 **2026 或 2027 年**实现通用的超人类智能，甚至可能就在今年。
- **DeepSeek 要求员工上交护照！**：据 [Twitter 上的 Amir](https://fxtwitter.com/amir/status/1900583042659541477) 报道，**DeepSeek** 的所有者要求研发人员上交护照以限制出境。
   - 一些成员猜测这是否会导致 **DeepSeek** 的工作保持开源，或者美国是否会对前沿公司的员工采取类似措施。
- **Xet 使用 Content-Defined Chunking (CDC)**：正如其 [HuggingFace Join 页面](https://huggingface.co/join/xet)所述，**Xet** 使用 [Content-Defined Chunking (CDC)](https://huggingface.co/blog/from-chunks-to-blocks) 技术智能地将文件拆分为唯一的块（chunks）。
   - 一位成员询问这与 fast transfer 有何不同，另一位成员回答说它们是不同的技术，fast transfer 仍在使用 Git LFS。
- **Math-500 采样受到质疑并得到验证**：一位成员询问为什么 Qwen 的 [GitHub repo](https://github.com/QwenLM/QwQ) 评估脚本中 Math-500 采样是随机的。
   - 另一位成员回复说这“显然是随机的”，并引用了 [Lightman et al 2023](https://cdn.discordapp.com/attachments/1179128538679488533/1350170687460868186/image.png?ex=67d5c3f0&is=67d47270&hm=6c771f09d27b7bad57e711e55ed2b111ac29af6a6485feb2c89103757f0771de&)，同时指出 **Long Context** 评估和答案提取非常令人头疼，而 **Math-500 的相关性非常好**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/join/xet">加入 Xet 等待名单 · Hugging Face</a>: 无描述</li><li><a href="https://fxtwitter.com/amir/status/1900583042659541477">来自 Amir Efrati (@amir) 的推文</a>: 最新消息：DeepSeek 的所有者要求研发人员上交护照，以便他们无法出国。女士们先生们……中国：</li><li><a href="https://x.com/Alibaba_Qwen/status/1900595120053047452">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 大家好，我们为 QwQ 建立了一个 GitHub repo，专门提供评估脚本，方便大家测试推理模型的基准性能，并复现我们报告的结果。我们……</li><li><a href="https://x.com/arcprize/status/1900627173280804941">来自 ARC Prize (@arcprize) 的推文</a>: 3/24/2025</li><li><a href="https://tenor.com/view/gigachad-chad-gif-20773266">Gigachad GIF - Gigachad Chad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://archive.is/jQZln">为什么我感受到了 A.G.I. - 纽约时报</a>: 无描述
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1350141493670576128)** (16 条消息🔥): 

> `隐私侵犯, Claude 生日, Claude Code Vim 模式, Gemma 3 许可问题` 


- **电话号码泄露引发挫败感**：一名成员对人们在网上找到他们的电话号码并请求帮助表示沮丧，例如询问 *"嘿 nato，你能帮我 post-train 我的 llama2 模型吗？谢谢"*。
   - 他们将此归咎于浏览器扩展或付费服务，并正在寻求从 [Xeophon](https://x.com/alexalbert__/status/1900592059364634973) 等网站删除其信息的方法。
- **Claude 庆祝又一周年**：成员们庆祝了 **Claude** 的两岁生日，并引用了[两年前](https://x.com/alexalbert__/status/1900592059364634973)的原始公告。
   - 另一位成员强调了 **Claude Code** 的新功能，包括通过输入斜杠命令 `/vim` 激活的 [Vim mode](https://x.com/_catwu/status/1900593728664035590)。
- **Gemma 3 的许可证引发担忧**：一篇 TechCrunch 文章提到了该成员关于模型许可证的工作，特别是针对 [Google 的 Gemma 3](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/)。
   - 文章讨论了 **Gemma 3 的许可证**虽然在效率方面受到称赞，但由于限制性和不一致的条款，带来了商业使用风险。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_catwu/status/1900593728664035590">来自 cat (@_catwu) 的推文</a>：Claude Code 的又一批新功能！首先是：Vim mode。这为您在 Claude Code 中编辑提示词提供了熟悉的插入/命令模式。通过输入斜杠命令 /vim 开启。但是...</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973">来自 Alex Albert (@alexalbert__) 的推文</a>：两年前的今天，我们向世界发布了 Claude。两岁生日快乐，Claude！</li><li><a href="https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/">“开源”模型许可证通常带有令人担忧的限制 | TechCrunch</a>：来自 Google、Meta 等公司的“开源”模型发布带有繁琐的条款，使得一些公司对使用它们持谨慎态度。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1350148438523773020)** (5 条消息): 

> `训练中期分析, SF Compute H100s, SF Compute CLI` 


- **深入探讨训练中期分析 (Mid-Training Analysis)**：一位成员分享了 [fxtwitter 上的链接](https://fxtwitter.com/Yuchenj_UW/status/1900589508590268485)，引发了关于 **mid-training analysis** 细微差别的讨论。
   - 讨论很简短，只有一个后续问题。
- **SF Compute 令人惊讶的低价 H100**：一位成员强调 [SF Compute](https://sfcompute.com/) 为 **H100s** 提供了惊人的低价，特别是对于短期租赁，并指出可以仅租用一小时的 **128 张 H100**。
   - 在域名正确配置之前，他们之前曾遇到过一个令人困惑的占位页面。
- **SF Compute 即将推出额外的 2,000 张 H100**：San Francisco Compute Company [即将推出额外的 **2,000 张 H100**](https://x.com/evanjconrad/status/1884361612766896510)，并运营一个针对大规模、经过审核的 **H100 clusters** 的市场。
   - SF Compute 像传统云服务商一样支持用户，并且还提供了一个[简单但强大的 CLI](https://docs.sfcompute.com)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LocoMocoBassy/status/1900191262822551986">来自 LocoMocosBasilisk (@LocoMocoBassy) 的推文</a>：嗯……</li><li><a href="https://sfcompute.com/">SF Compute | 配备 3.2Tb/s InfiniBand 的 H100s</a>：San Francisco Compute Company：您可以按小时租用的大型、低成本 GPU 集群，用于预训练、推理等。获取配备 3.2Tb/s InfiniBand、并行存储、快速网络的 H100...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1350190512946151576)** (11 条消息🔥): 

> `GRPO implementation, KL penalty, RLHF algorithms` 


- **GRPO 实现技巧：在 Loss 中应用 KL Penalty**：一位成员讨论了一个 **GRPO 实现技巧**，即直接在 Loss 中应用 **KL penalty**，而不是在计算 Reward 时应用。他指出其影响尚不完全明确，并链接到了 [RLHF Book](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1)。
   - 该成员还分享了其在 [X/Twitter 上的提问链接](https://x.com/natolambert/status/1900639281791615387)，征求关于该方法的直觉理解或消融实验结果。
- **RLHF 算法随时间的流行度演变**：用于 **RLHF** 的主流算法一直在演进；最初 **ChatGPT** 使用的是 **PPO** 的变体，但研究表明 **REINFORCE** 风格的算法也很有前景，例如 [Ahmadian et al. 2024](https://rlhfbook.com/c/11-policy-gradients.html#ref-ahmadian2024back) 和 [Wang et al. 2024](https://rlhfbook.com/c/11-policy-gradients.html#ref-wang2024helpsteer2p)。
- **通过 KL Penalty 关注 Reward 信号**：一位成员建议在 Loss 中应用 **KL penalty** 可能有助于模型专注于 Reward 信号，但该成员也指出，这在最终效果上 *应该等同于* 在计算 Reward 时应用它。
   - 另一位成员猜测 **PG term** 会使 Reward 最大化，因此两个版本的 Loss 最小值应该是相同的，但其动态过程（dynamics）可能仍然不同。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1">Policy Gradient
Algorithms | RLHF Book by Nathan Lambert</a>：Reinforcement Learning from Human Feedback 专著</li><li><a href="https://x.com/natolambert/status/1900639281791615387">Nathan Lambert (@natolambert) 的推文</a>：关于直接在 Loss 中应用 KL penalty 而不是在计算 Reward 时应用的直觉理解或消融实验。这对学习过程有何改变？normalrewards = rewards - self.beta * p...</li><li><a href="https://bsky.app/profile/natolambert.bsky.social/post/3lkeftspdzo2x">Nathan Lambert (@natolambert.bsky.social)</a>：关于直接在 Loss 中应用 KL penalty 而不是在计算 Reward 时应用的直觉理解或消融实验。这对学习过程有何改变？normalrewards = rewards - self.beta * p...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1350132490500837458)** (2 条消息): 

> `Cohere Command A, Jamba 1.6 Large, Jamba 1.6 Mini, Gemma 3 模型, Anthropic 故障` 


- ****Cohere Command A 备受瞩目****：一款全新的 **111B 参数开源权重模型**，拥有 **256k 上下文窗口**，专注于在 Agent、多语言和编程用例中提供卓越性能，[Cohere Command A](https://openrouter.ai/cohere/command-a) 现已上线。
- ****AI21 Jamba 发布新模型****：**AI21** 发布了 [Jamba 1.6 Large](https://openrouter.ai/ai21/jamba-1.6-large)，拥有 **940 亿激活参数**和 **256K Token 上下文窗口**；同时还推出了 [Jamba 1.6 Mini](https://openrouter.ai/ai21/jamba-1.6-mini)，拥有 **120 亿激活参数**，两者均支持结构化 JSON 输出和 Tool-use 能力。
- ****Gemma 3 免费开放****：**Gemma 3** 的所有版本现已免费提供：[Gemma 3 12B](https://openrouter.ai/google/gemma-3-12b-it:free) 引入了多模态能力，支持视觉语言输入和文本输出，处理高达 **128k Token** 的上下文窗口，并支持超过 **140 种语言**；此外还包括 [Gemma 3 4B](https://openrouter.ai/google/gemma-3-4b-it:free) 和 [Gemma 3 1B](https://openrouter.ai/google/gemma-3-1b-it:free)。
- ****Anthropic API 异常已解决****：**Anthropic** 宣布了针对 **Claude 3.7 Sonnet** 请求错误率升高的故障，更新已发布在他们的 [状态页面](https://status.anthropic.com/incidents/qtxnlg9yrwqv) 上。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Claude 3.7 Sonnet 请求错误率升高</a>: 未找到描述</li><li><a href="https://openrouter.ai/cohere/command-a):">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/ai21/jamba-1.6-large):">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/ai21/jamba-1.6-mini):">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/google/gemma-3-12b-it:free)">Gemma 3 12B - API, Providers, Stats</a>: Gemma 3 引入了多模态能力，支持视觉语言输入和文本输出。它处理高达 128k Token 的上下文窗口，支持超过 140 种语言，并提供改进的数学、推理...</li><li><a href="https://openrouter.ai/google/gemma-3-4b-it:free)">Gemma 3 4B - API, Providers, Stats</a>: Gemma 3 引入了多模态能力，支持视觉语言输入和文本输出。它处理高达 128k Token 的上下文窗口，支持超过 140 种语言，并提供改进的数学、推理...</li><li><a href="https://openrouter.ai/google/gemma-3-1b-it:free)">Gemma 3 1B - API, Providers, Stats</a>: Gemma 3 1B 是全新 Gemma 3 家族中最小的成员。它处理高达 32k Token 的上下文窗口，支持超过 140 种语言，并提供改进的数学、推理和聊天能力，包括...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1350039038618243092)** (67 条消息🔥🔥): 

> `OR ChatGPT 模型, OpenRouter 模型图标, Deepseek v3 问题, OLMO-2, Cohere 重复惩罚` 


- **ChatGPT-4o-latest 价格高于预期**：**chatgpt-4o-latest** 模型是最新的，但价格比普通的 **4o** 模型略贵。
- **OpenRouter 模型图标无法通过 API 获取**：模型图标在 `/api/v1/models` 响应中不可用，而是使用网站的 favicon。
- **Deepseek v3 模型出现零 Token 问题**：有时推理栈只返回 **零个补全 Token (completion tokens)**，而 OpenRouter 仍会被上游供应商收费。
- **OLMO-2 模型托管在 OpenRouter**：OLMo-2 正通过 Parasail 上线；有人会启动它并通知 OpenRouter 进行添加。
- **使用 OpenRouter 举办的 AI 象棋锦标赛**：创建了一个包含 **15 个模型** 的 AI 象棋锦标赛，模型会接收标准象棋符号表示的棋盘状态、对局历史和合法移动列表，在此进行对战，链接见[此处](https://dubesor.de/chess/tournament)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dubesor.de/chess/tournament">Dubesor LLM 象棋锦标赛</a>：未找到描述</li><li><a href="https://openrouter.ai/openai/gpt-3.5-turbo-">Discord</a>：未找到描述</li><li><a href="https://openrouter.ai/openai/gpt-3.5-turbo-instruct">GPT-3.5 Turbo Instruct - API, Providers, Stats</a>：该模型是 GPT-3.5 Turbo 的变体，针对指令性提示进行了微调，并省略了聊天相关的优化。使用 API 运行 GPT-3.5 Turbo Instruct</li><li><a href="https://parasail.canny.io/model-request">模型请求 | Parasail</a>：请求模型 - 请填写 Hugging Face 模型及任何其他信息！
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1350031425977581588)** (59 条消息🔥🔥): 

> `移植中的 Rust vs. Go, DeepSeek 炒作, OLMo 2 32B, ChatGPT 被高估, 代码生成质量：Grok 3 vs Mistral vs OpenAI` 


- **在实际移植中 Go 优于 Rust**：成员们讨论了为什么不使用 **Rust** 进行移植，一位成员解释说，逐函数移植到 **Go** 可以实现精确的行为对等，避免了多年的重写以及处理 Rust 的无 GC (no-GC) 和生命周期注解 (lifecycle annotations)。
   - 补充道：*Rust 虽然更快、更高效，但在实践中这种差异并没有人们想象的那么大，而且 Golang 在开发分布式、异步或网络应用时非常符合人体工程学 (ergonomic)*。
- **DeepSeek 的炒作是人为策划的**：一些成员认为围绕 **DeepSeek** 的炒作是人为策划的，辩称其模型被简化了，无法与前沿 AI 模型相提并论，将其比作*菠萝与苹果的对比*。
   - 另一位成员反驳说，炒作是由 **DeepSeek** 那些开发出极速文件系统的*疯狂工程师*推动的。
- **OLMo 2 32B 是一个完全开放的模型**：**OLMo 2 32B** 已发布，被描述为[第一个在学术基准测试中超越 **GPT3.5-Turbo** 和 **GPT-4o mini** 的完全开放模型](https://allenai.org/blog/olmo2)。
   - 据称它可与领先的开放权重模型相媲美，而所需的训练计算量仅为一小部分，训练成本仅为 **Qwen 2.5 32B** 的三分之一。
- **ChatGPT 被高估，建议使用 Claude**：一位成员表示 **ChatGPT** 被高估了，因为*它实际上解决不了我需要解决的问题*，他更倾向于 **Mistral Small 24B**、**QwQ 32B** 和 **Claude 3.7 Sonnet**。
   - 另一位用户分享道：*我在 Claude 上更容易得到想要的结果*，而且*由于某种原因，它似乎更擅长理解意图和动机*。
- **Grok 3 用于代码生成**：成员们辩论了代码生成的质量，提到 **OpenAI** 模型经常生成过时的代码，而 **Mistral** 可以将其重构为更现代的代码，**Grok 3** 生成的代码则*看起来像是由专业程序员编写的*。
   - 在 **VSCode** 中，一位成员更喜欢使用 **Amazon Q** 而非 **Copilot**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mgostIH/status/1900577603930444239">来自 mgostIH (@mgostIH) 的推文</a>：&gt; 是的，但软件很容易复制，你直接抄代码就行了 —— 某个从未碰过 CMake 的人</li><li><a href="https://allenai.org/blog/olmo2-32B">OLMo 2 32B：第一个超越 GPT 3.5 和 GPT 4o mini 的完全开放模型 | Ai2</a>：介绍 OLMo 2 32B，它是 OLMo 2 家族中功能最强、参数最大的模型。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

> @erkinalp: 

.ogeneral: 我会说两者都不是
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1350049237353627649)** (3 messages): 

> `Speech-to-Speech Generation, Moshi by Kyutai Labs, Hertz-dev by Standard-Intelligence` 


- **Speech-to-Speech 模型探索开启**：一名成员正在寻找 **speech-to-speech generation** 模型，重点关注无需多模态输入的对话式语音，将其与 **OpenAI Realtime API** 或 **Sesame AI** 等模型区分开来。
   - 该成员寻求的是一个独立模型，而不是一个同时接受文本和音频的多模态模型。
- **Moshi 模型发布**：来自 **Kyutai Labs** 的 [Moshi](https://github.com/kyutai-labs/moshi) 是一个语音-文本基础模型和全双工语音对话框架。
   - 它利用了 **Mimi**，一种 state-of-the-art 的流式神经音频编解码器。
- **Hertz-dev 模型开发完成**：来自 **Standard-Intelligence** 的 [Hertz-dev](https://github.com/Standard-Intelligence/hertz-dev) 是首个用于全双工对话音频的基础模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi: Moshi 是一个语音-文本基础模型和全双工语音对话框架。它使用了 Mimi，一种 state-of-the-art 的流式神经音频编解码器。</a>: Moshi is a speech-text foundation model and full-duplex spoken dialogue framework. It uses Mimi, a state-of-the-art streaming neural audio codec. - kyutai-labs/moshi</li><li><a href="https://github.com/Standard-Intelligence/hertz-dev">GitHub - Standard-Intelligence/hertz-dev: 首个用于全双工对话音频的基础模型</a>: first base model for full-duplex conversational audio - Standard-Intelligence/hertz-dev
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1350043285439254618)** (3 messages): 

> `tl.int1 masks in Triton, tl.advance negative offsets, Triton Windows upgrade to 3.2` 


- **使用 `tl.int1` 优化 Triton 掩码？**：一位成员询问在 Triton 中使用 `tl.load` 时，显式将掩码转换为 `tl.int1` 是否有任何性能或功能上的好处。
   - 未提供回复。
- **`tl.advance` 接受负偏移吗？**：一位成员询问 Triton 中的 `tl.advance` 是否接受用于指针运算的负偏移量。
   - 未提供回复。
- **Windows Triton 升级困扰？**：一位成员正在寻求在 Windows 上将 **Triton** 从 **3.1** 升级到 **3.2** 的步骤验证，特别是关于 **PyTorch** 和缓存清理方面。他们链接到了[这个仓库](https://github.com/woct0rdho/triton-windows)。
   - 他们使用的是 **Python 3.10 + CUDA 12.5** 以及 **ComfyUI 的 python_embedded: Python 3.12 + PyTorch 2.5.1+cu124 + Triton 3.1**



**提到的链接**：<a href="https://download.pytorch.org/whl/cu124">未找到标题</a>：未找到描述

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1350161350214553650)** (6 messages): 

> `cuda::memcpy_async, A100, global vs shared memory` 


- **A100 上的 `cuda::memcpy_async` 实验**：一位成员使用 CUDA 文档中的示例在 **A100** 上对 [`cuda::memcpy_async`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with-memcpy-async) 进行了实验。
   - 他们观察到使用 `memcpy_async` 的 kernel 运行时间略长，并询问这种意外行为的原因。
- **异步拷贝：全局内存 vs 共享内存**：一位成员澄清说，异步拷贝只能在**全局内存和共享内存**之间传输数据。
   - 他们解释说，异步拷贝的优势在于将内存加载与其他计算重叠，这需要从共享内存加载值才能被利用。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1350180800368218142)** (1 messages): 

> `Block Diffusion, Autoregressive Models, Diffusion Models, ICLR 2025` 


- **Block Diffusion 模型插值了 Autoregressive 和 Diffusion LMs**：新的 **Block Diffusion** 模型旨在结合自回归和扩散语言模型的优点，详见一篇被 **ICLR 2025 Oral** 接收的 [论文](https://openreview.net/forum?id=tyEyYT267x)。
   - 它实现了 **高质量**、**任意长度生成**、**KV caching** 和 **可并行化** 处理，解决了现有模型的局限性；代码已在 [GitHub](https://github.com/kuleshov-group/BD3-LMs) 和 [HuggingFace](https://huggingface.co/collections/kuleshov-group/BD3-LMs-67be95f81b96b15fec50d53f) 开源。
- **Autoregressive Models**：自回归模型具有 **高质量** 输出和 **任意长度** 生成的优势，并支持 **Key-Value (KV) caching**。
   - 然而，自回归模型存在 **不可并行化** 的缺点。



**Link mentioned**: <a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG

  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1350130082085077042)** (2 messages): 

> `Dynamic Shapes, Segmentation Fault` 


- **Dynamic Shapes 引发 Segmentation Fault**：一名成员提交了一个关于 [dynamic shapes 导致 segmentation fault](https://github.com/tile-ai/tilelang/issues/215) 的 issue。
   - 他们补充道，“也许我的理解有误”。
- **Tilelang 面临 Dynamic Shape 挑战**：有人提出了在 Tilelang 中处理 dynamic shapes 时遇到 segmentation fault 的问题。
   - 该 issue 的报告者表示不确定，称其对问题的理解可能不正确。



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang/issues/215">segmentation fault with dynamic shapes · Issue #215 · tile-ai/tilelang</a>: # Copyright (c) Microsoft Corporation. # Licensed under the MIT License. from tilelang import tvm as tvm import tilelang.language as T import tilelang.testing import tilelang import torch def matmu...

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1350051208903262242)** (1 messages): 

> `Gemma3, LigerKernel, RMSNorm` 


- **Gemma3 实现草案已提交至 LigerKernel**：一名成员发起了一项新挑战，起草了将 **Gemma3** 集成到 **LigerKernel** 的实现，并分享了 [Pull Request 链接](https://github.com/linkedin/Liger-Kernel/pull/606)。
   - 该成员作为编程新人，正在寻求对该草案的帮助和反馈。
- **Gemma3 与 Gemma2 具有相似性**：根据 Pull Request，**Gemma3** 与 **Gemma2** 高度相似，但在 **RMSNorm Calls** 方面存在一些差异。
   - 这些更改使得可以使用 **Liger kernels** 对 **Gemma3** 的文本部分进行 patch。



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel/pull/606">Adding Support for Gemma3 by DRXD1000 · Pull Request #606 · linkedin/Liger-Kernel</a>: SummaryGemma3 has high similarities to Gemma2 with some differences in RMSNorm CallsThis change enables patching the Text Parts of Gemma3 with Liger kernels.Testing DoneHardware Type: AMD ...

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1350086681461784738)** (2 条消息): 

> `Triton bitpacking, Gemlite, GTC CUDA content` 


- **Triton bitpacking 飞速提升**: **Triton** 中的 **Bitpacking** 相比 4090 上的 PyTorch 实现实现了显著的加速，**32-bit packing** 实现了 **98x** 的加速，**8-bit packing** 实现了 **26x** 的加速。
   - 使用新的 bitpacking 实现，重新打包 **Llama3-8B** 模型的时间从 **49 秒缩短至 1.6 秒**。
- **Gemlite 的 bitpack 实现释放性能**: 一位成员展示了他们在 **Gemlite** 项目中使用 **Triton** 优化 bitpacking 的工作，并附上了 [GitHub 上的相关代码链接](https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133)。
   - 该优化支持 *快速低位 matmul kernel*。
- **GTC CUDA 内容即将发布**: 一位成员分享了关于 **GTC** 上 [CUDA 内容](https://www.nvidia.com/gtc/sessions/cuda-developer/) 的信息，旨在利用 NVIDIA CUDA 构建高性能、GPU 加速的应用。
   - 他们分享了一些在会议上展示的 Plask 和 CERN 的图片。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.nvidia.com/gtc/sessions/cuda-developer/">NVIDIA GTC AI Conference 2025</a>: 2025年3月17日至21日。圣何塞。立即注册。</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133">gemlite/gemlite/bitpack.py at master · mobiusml/gemlite</a>: Triton 中的快速低位 matmul kernel。通过在 GitHub 上创建账户为 mobiusml/gemlite 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1350078770396266557)** (31 messages🔥): 

> `vLLM 对 Gemma 3 的支持、组相对策略优化 (GRPO)、针对 reasoning-gym 的 veRL 训练、reasoning-gym 中的复合配置、课程训练` 


- **vLLM 获准支持 Gemma 3**：成员们讨论了在 **vLLM** 中添加 **Gemma 3** 支持，并引用了 [此 GitHub issue](https://github.com/vllm-project/vllm/issues/14696)。
   - 一位成员报告在使用 TGI 运行 **Gemma 3** 时遇到了上下文窗口大小的问题，怀疑是底层 transformers 实现中存在问题。
- **组相对策略优化 (GRPO) 迅速普及**：成员们讨论了 **组相对策略优化 (GRPO)** 如何在 Large Language Models 的 Reinforcement Learning 中变得流行。
   - 分享了 oxen.ai 关于 [GRPO VRAM 需求](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor) 的博文，指出了其在训练中的有效性，并附带了 **DeepSeek-R1 论文** 的链接。
- **veRL 取得进展**：一位成员确认，使用 [此分支](https://github.com/open-thought/reasoning-gym/tree/ollie/verl-experiments) 中的更改，最新的 veRL 已能成功进行 **chain_sum** 的 **veRL 训练**。
   - 该更改已合并到 main 分支。
- **reasoning-gym 为推理重构做准备**：成员们讨论了是使用单个 **RG 数据集生成器** 还是多个生成器的组合来训练模型，并倾向于后者以提升“全方位”的推理能力。
   - 他们计划测试一个包含 3-5 个数据集的小型复合配置，参考了 [复合数据集代码](https://github.com/open-thought/reasoning-gym/blob/main/tests/test_composite.py) 以及 [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) 中数据集的课程状态。
- **reasoning-gym Star 数突破 500**：提到 *reasoning-gym v.0.1.16* 已 [上传至 pypi](https://pypi.org/project/reasoning-gym/)，且该项目即将达到 500 stars。
   - 一位成员发布了一张庆祝图片。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/reasoning-gym/)">Client Challenge</a>：未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/tests/test_composite.py">reasoning-gym/tests/test_composite.py (main 分支) · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建账号，为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/tree/ollie/verl-experiments">GitHub - open-thought/reasoning-gym (ollie/verl-experiments 分支)</a>：程序化推理数据集。通过在 GitHub 上创建账号，为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/vllm-project/vllm/issues/14696">[Feature]: 支持 gemma3 架构 · Issue #14696 · vllm-project/vllm</a>：🚀 功能、动机和方案。我正在使用 vLLM 托管 LLM/SLM，随着最近 Gemma 3 的发布，我希望 vLLM 能支持它。Google 表示 Gemma 3 具有首日支持......</li><li><a href="https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor">🧠 显存受限者的 GRPO VRAM 需求 | Oxen.ai</a>：自 DeepSeek-R1 发布以来，组相对策略优化 (GRPO) 因其有效性和易训练性，已成为 Large Language Models 强化学习领域的热门话题...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1350062138890129418)** (7 messages): 

> `verl 会话, tilelang 提交, pip install tilelang` 


- **对 Verl 分享会的需求**：一位成员询问了关于 **Verl** 分享会的时间表，以及提交 **tilelang** 的可能性。
- **通过 pip 安装 tilelang**：一位成员表示，用户可以通过脚本使用 **pip** 安装任何包，并提供了一个安装 **tilelang** 的示例脚本。
   - 他们警告说安装过程相当长，可能会导致超时并给机器带来不必要的工作量。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1350175035817594880)** (3 messages): 

> `排行榜提交, Grayscale 排行榜, Conv2d 排行榜, H100 GPU, Modal 运行器` 


- **Grayscale 排行榜迎来新提交**：**grayscale** 排行榜新增了三个提交。
   - ID 为 **2013** 和 **2015** 的提交使用了 **H100 GPU** 和 **Modal 运行器**。
- **Conv2d 排行榜数据更新**：**conv2d** 排行榜新增了一个提交。
   - ID 为 **2014** 的提交同样使用了 **H100 GPU** 和 **Modal 运行器**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1350036900383035442)** (46 messages🔥): 

> `智力下降，技术和智能手机的影响，食品添加剂与认知能力衰退，Deepseek 模型蒸馏，ADHD 诊断率` 


- **智力下降引发辩论！**：讨论源于一篇 [《金融时报》(Financial Times) 的文章](https://www.ft.com/content/link-to-ft-article)，该文章指出发达国家的平均智力正在下降，理由是关于**认知挑战**的报告增加，以及在**推理和解决问题**方面的表现下降。
   - 一位成员理论化认为这可能是由于技术，特别是**智能手机**和**社交媒体**导致了思考的外包，然而图表显示的时间点实际上是在 **ChatGPT** 流行之前。
- **技术是导致“脑力流失”的罪魁祸首吗？**：成员们辩论了认知能力下降的潜在原因，包括**技术的影响**、**移民**和**氟化水**。
   - 一位成员指出，认知挑战的比例自 20 世纪 90 年代以来一直在稳步上升，并从 2012 年左右开始突然加速。
- **食品添加剂与思维迟钝有关？**：成员们讨论了**超加工食品 (UPFs)** 在全球范围内的供应和消费有所增加，目前在一些高收入国家占每日能量摄入的 **50–60%**，并且这与认知能力下降有关。
   - 另一位成员提到，像 **Nestlé** 这样在许多国家运营的跨国公司在全球范围内生产和分销，这些公司之一对产品进行的添加剂调整或更改如何产生全球性影响是可以理解的。
- **DeepSeek 模型起源之谜**：讨论涵盖了 **Deepseek V3（指令版）** 可能是从 **OpenAI 模型**蒸馏而来的。
   - 一位成员指出，*甚至 OpenAI 也会非正式地支持蒸馏他们的模型，他们只是似乎不喜欢 Deepseek 这么做*。
- **TikTok 大脑缩短注意力跨度**：成员们认为像 **TikTok** 和 **Instagram** 这样的平台通过在极短的时间内传递持续的情感印象来影响我们的大脑。
   - 结果是产生了一种*成瘾，使我们不断寻求更多的刺激。*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1350105918485823509)** (1 messages): 

> `Claude Sonnet 3.7, o3-mini-high vs o1` 


- **Claude Sonnet 3.7 成为编程新宠**：一位成员现在专门使用 **Claude Sonnet 3.7** 进行编程，发现 **ChatGPT** 已经落后了。
- **o3-mini-high 模型击败 o1**：一位成员表示 **o3-mini-high** 模型优于 **o1**。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1350058306588119093)** (4 messages): 

> `Gemini 2.0 Deep Research, NotebookLM, PhytoIntelligence framework` 


- **Gemini 2.0 Deep Research 加入 NotebookLM**：成员们考虑将 **Gemini 2.0 Deep Research** 与 **NotebookLM** 结合使用，以创建一个强大的文档获取和处理工具。
   - 目标是在不超出提供边界的情况下将材料容器化，并质疑 **Deep Research** 最终是否会取代 **NotebookLM**。
- **NotebookLM 启发非洲项目 Ecokham**：一位来自非洲的成员正在使用 **NotebookLM** 来连接想法、编辑和定制路线图，并为其项目 **Ecokham** 生成鼓舞人心的音频。
   - 他对 **NotebookLM** 在激励他和他的团队方面提供的帮助表示感谢。
- **NotebookLM 原型化 PhytoIntelligence 框架**：一位成员正在使用 **NotebookLM** 整理笔记，并为自主营养保健品设计的 **PhytoIntelligence framework** 制作原型。
   - 该框架旨在减轻诊断挑战，用户感谢 Google 提供的 **NotebookLM** 功能。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1350045583200747602)** (41 messages🔥): 

> `Notebook LM 中的图像和表格识别、Notebook LM 移动端应用、Notebook LM 语言设置、公开笔记本共享、Google Sheets 集成` 


- **NotebookLM 渴望图像与表格识别能力**：用户强烈要求在 Notebook LM 中加入 **图像和表格识别** 功能，并强调由于需要不断重新打开源文件，目前的状态感觉像是“半成品”；一位用户甚至分享了 [相关的猫咪 GIF](https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569)。
   - 他们认为图像识别胜过“千言万语”，而最清晰的数据往往来自源表格和 Google Sheets。
- **NotebookLM 移动端应用**：许多用户正请求推出 NotebookLM 的 **移动端应用版本**。
   - 目前社区认为移动版本“仍未面世”。
- **用户报告 Notebook LM 反复出现系统错误**：一位用户报告了反复出现的“系统无法回答（The system was unable to answer）”错误，这是本周第二次发生。
   - 其他用户测试了该问题但无法复现。
- **修改 NotebookLM 语言的 URL 技巧**：一位用户询问如何更改 NotebookLM 的语言，另一位用户分享了在 URL 末尾添加 **?hl=LANGUAGE_CODE** 的技巧（例如，西班牙语为 `hl=es`）。
   - 一位用户确认他们身处法国。
- **笔记本公开共享**：一位用户询问了关于 **公开共享** 笔记本的计划。
   - 一名成员回应称，完全开放访问的可能性不大，但目前可以通过企业账号或 Gmail 账号进行有权限制的共享。



**提到的链接**：<a href="https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569">Cat Wait GIF - Cat Wait Im - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1350149344375996416)** (1 messages): 

> `Google Gemini, Google Vertex AI, 统一的 @googleai 集成, Streaming, Async` 


- **Google Gemini 与 Vertex AI 合并！**：根据 [这条推文](https://twitter.com/llama_index/status/1900590246070476929)，统一的 `@googleai` 集成现在同时支持 **Google Gemini** 和 **Google Vertex AI**。
   - 该集成支持 **streaming**、**async**、**multi-modal** 和 **structured prediction**，甚至支持图像。
- **更多 Google Gemini 和 Vertex AI 的益处！**：统一的 `@googleai` 集成现在支持 **Google Gemini** 和 **Google Vertex AI**，并带来更多好处！
   - 该集成支持 **streaming**、**async**、**multi-modal** 和 **structured prediction**，甚至支持图像。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1350105702131306546)** (8 messages🔥): 

> `LlamaIndex 对比 Langchain, OpenAI 工具调用的 delta 事件, Agentic RAG 应用` 


- **关于 LlamaIndex 优于 Langchain 之处的辩论爆发**：一名成员询问了 **LlamaIndex** 相对于 **Langchain** 的优势。
   - 然而，在给定的上下文中没有提供相关的讨论或答案。
- **探究 OpenAI 工具调用中缺失 Delta 事件的原因**：一名成员询问为什么 **OpenAI** 模型在进行工具调用时不发出 delta 事件，并指出虽然发出了事件但内容为空。
   - 另一名成员解释说，工具调用不能是流式的，因为 LLM 需要完整的工具响应才能生成下一个响应，建议自行构建流。
- **关于 Agentic RAG 应用 API 的咨询出现**：一名成员询问是否有任何 **专注于构建 Agentic RAG 应用的 API**，以简化创建和管理应用的过程。
   - 讨论探讨了在 LlamaIndex 中构建 Agentic 应用的首选方式，指出了其演进过程和可用的多种构建模块，但缺乏确定性的指南。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1350124926916366387)** (7 messages): 

> `Gemma 3 12B, Qwen 2.5 Coder, LM Studio, Multimodal Models, Water Freezing Experiment` 


- **Gemma 3 12B 在智能方面超越 Qwen**：一位用户发现 **Gemma 3 12B 模型** 在其电脑上的智能表现超过了 **Qwen 14B** 甚至 **Qwen 32B**。
   - 该用户指出 *Gemma 3 12b 需要另一个额外的 GGUF 文件，这可能就是它无法在 **GPT4All** 上运行的原因*。
- **Gemma 3 需要 GPT4All 更新**：一位用户指出，由于 **Gemma 3 12B** 的架构与 **Gemma 2** 不同，且需要 *mmproj* 文件，**GPT4All** 可能需要更新才能支持它。
   - 另一位用户开玩笑说 *大家都用发布才 1 天的模型，却期望一切都能正常运行*，强调了 AI 模型开发的飞速节奏。
- **语言理解测试：Gemma 3 表现出色**：一位用户通过用多种语言提问，测试了包括 **Gemma 3 12B**、**DeepSeek R1** 和 **Qwen 2.5 Coder** 在内的多种模型。
   - 用户发现 **Gemma 3** 和 **DeepSeek R1** 始终能以与问题相同的语言提供正确且详尽的回答。
- **水结冰实验产生不同的回答**：当被问及冷冻一罐水的结果时，**DeepSeek-R1** 表示罐子会破裂。
   - **Gemma-3-12b** 正确回答道：*将装满水的罐子放在室外零下温度过夜，几乎肯定会导致罐子因水结冰膨胀而破碎或产生裂纹*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1350066494473633883)** (6 messages): 

> `Explicit Feedback in dspy.Refine, Manual Feedback Implementation in Refine, Reward Function Return Value for Feedback` 


- **显式反馈召唤 Refine 的回归**：一位成员询问如何将 **显式反馈 (explicit feedback)** 集成到 `dspy.Refine` 中，类似于早期版本中的 `dspy.Suggest`，以便在超出奖励函数阈值时清晰地指出需要改进的地方。
   - 该成员指出，**显式反馈**对于调试和理解错误非常有帮助。
- **手动反馈正式引入 Refine**：一位团队成员确认正在将 **手动反馈 (manual feedback)** 添加到 `Refine` 中。
   - 拟议的实现方式是将反馈包含在 **奖励函数的返回值** 中，作为一个 `dspy.Prediction` 对象，其中包含分数和反馈。
- **奖励函数成为反馈源泉**：团队成员询问反馈作为 **reward_fn 的返回值** 的一部分是否可以接受。
   - 用户回答说这 *非常完美*，并感谢了团队成员。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1350117120519704606)** (5 messages): 

> `Command A, OpenRouter Integration, Prime Number Bug, Local API Performance` 


- **Command A 在 OpenRouter 上线**：**Cohere 的 Command A** 是一个拥有 **111B** 参数和 **256k** 上下文窗口的开放权重模型，现已在 [OpenRouter](https://openrouter.ai/cohere/command-a) 上线。
- **质数难题困扰 Command A**：用户发现，当询问关于数字之和为 15 的质数时，**Command A** 要么返回错误答案，要么进入死循环。
- **本地 API 在运行 Command A 时表现不佳**：用户报告称，需要对 **ITOr** 中的建模进行补丁或直接使用 **API**，因为即使有足够的 VRAM，本地设置也无法达到理想的速度。



**提到的链接**：<a href="https://openrouter.ai/cohere/command-a">Command A - API, Providers, Stats</a>：Command A 是一个拥有 111B 参数和 256k 上下文窗口的开放权重模型，专注于在 Agent、多语言和代码用例中提供卓越性能。与其他领先的私有模型相比...

  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

michael: 可以运行，请使用 `https://api.cohere.com/compatibility/v1/chat/completions` 作为 base_url
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1350113696415744111)** (3 messages): 

> `Discord Scam Account, Account Impersonation` 


- **Discord 账号冒充！**: 一名成员报告收到冒充另一名用户的 **scam account** 发来的消息。
   - 被冒充的用户确认：“那不是我。这是我唯一的账号”，并且该 **fake account** 已被举报至 Discord 并从服务器中封禁。
- **用户警告冒充诈骗**: 一名用户向社区警示有一个 **scam account** 正在给他们发消息，冒充 Discord 上的另一名用户。
   - 该用户澄清说，他们唯一的合法账号是 "caroline_frasca"，且该冒充账号已被举报至 Discord 并从服务器中封禁。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1350124411876937990)** (1 messages): 

> `Discord impersonation, Discord account security` 


- **Caroline Frasca 冒充者正在私信骚扰**: 一个名为 *caroline_frascaa*、昵称为 *Caroline* 的垃圾账号一直在冒充真实用户发送私信。
   - 用户发布了 [虚假账号的截图](https://cdn.discordapp.com/attachments/1098765954302873621/1350124411906293851/Impersonating_Account.png?ex=67d598d7&is=67d44757&hm=bd212e9e154251a202378828ccf61282fd69df840ade2eb535738fc7d7e248cb&)，并更新了个人资料以帮助他人轻松识别真实账号。
- **Discord 服务器封禁冒充账号**: 冒充账号 *caroline_frascaa* 已被举报至 Discord 并从本服务器封禁。
   - Discord 管理员鼓励举报任何冒充账号。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/)** (1 messages): 

soracc: 是的，我们也在 stdlib 中使用它（例如在 `base64` 中）。
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1350148608418254940)** (1 messages): 

> `Self-Evaluation, Self-Reflection, Self-Refinement, Oracle Feedback` 


- **Self-Reflection 需要外部评估**: 一名成员询问第一课中关于 **self-evaluation** 的陈述如何与第二课相对应，因为它们似乎相互矛盾。
   - 他们指出，第一课提到 **self-reflection** 和 **self-refinement** 在有良好的外部评估时有效，而没有 **oracle feedback** 的 **self-correction** 会损害推理性能。
- **关于第 1 课和第 2 课中 Self-Evaluation 的澄清**: 用户寻求澄清两节课之间关于 **self-evaluation** 明显的矛盾之处。
   - 具体而言，他们注意到第二课强调了 **self-evaluation and improvement**，而第一课则强调了外部评估的重要性以及没有 **oracle feedback** 的自我修正可能带来的危害。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1350135872153976942)** (1 messages): 

> `Vertex, AWS` 


- **Vertex 准备迎接 1.6**: **Version 1.6** 尚未在 **Vertex** 上提供，但即将推出。
   - 它也将在 **AWS** 等其他平台上提供。
- **AWS 获取 1.6 的权限**: Version **1.6** 将在不久的将来在 **AWS** 等其他平台上提供。
   - 这应该能让 **AWS** 上的客户访问新功能。


  

---


---


---


---


---


{% else %}


> 邮件中的各频道详细分析已截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}