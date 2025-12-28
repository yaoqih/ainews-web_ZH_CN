---
companies:
- openai
- deepseek
- anthropic
- google-deepmind
- togethercompute
- hypertecgroup
- coreweave
- cursor-ai
- windsurf-ai
date: '2025-03-28T23:18:38.632397Z'
description: '**GPT-4o** 因其在代码编写、指令遵循和自由度方面的提升而受到称赞，成为领先的非推理类编程模型，在编程基准测试中超越了 **DeepSeek
  V3** 和 **Claude 3.7 Sonnet**，尽管它在性能上仍落后于 **o3-mini** 等推理模型。此外，报告指出了其在图像生成政策合规性方面的担忧，并正努力提高遵循度。


  **Gemini 2.5 Pro** 则因其先进的音视频理解能力、长上下文处理能力以及与 **Cursor AI** 和 **Windsurf AI** 等平台的集成而备受瞩目。在
  AI 基础设施发展方面，**Together AI** 与 **Hypertec Group** 达成合作伙伴关系以交付大规模 GPU 集群，同时 **CoreWeave
  的 IPO** 也因推动了 AI 基础设施的进步而受到赞誉。预计 GPU 和 TPU 的使用量将大幅增长。


  “GPT-4o 的透明度和背景生成功能”以及“Gemini 2.5 Pro 在 Simple-Bench AI 解释测试中得分超过 50%”是本次更新的关键亮点。'
id: 181c127f-3bfe-45c7-9adc-372f7f24c16b
models:
- gpt-4o
- deepseek-v3
- claude-3.7-sonnet
- o3-mini
- gemini-2.5-pro
original_slug: ainews-not-much-happened-today-9938
people:
- sama
- kevinweil
- joannejang
- nrehiew_
- giffmana
- _philschmid
- scaling01
- saranormous
title: 今天没发生什么事。
topics:
- coding
- instruction-following
- image-generation
- policy-compliance
- long-context
- audio-processing
- video-processing
- gpu-clusters
- ai-infrastructure
- api-access
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天**

> 2025年3月27日至3月28日的 AI 新闻。我们为您检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord（230 个频道，13422 条消息）。预计节省阅读时间（以 200wpm 计算）：**1217 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们今天预发布了 [2025 年 AI 工程现状调查](https://www.surveymonkey.com/r/57QJSF2)，填写调查即可参加 1000 美元 Amazon 礼品卡抽奖，并让您的声音在 AI 工程现状报告中被听到！


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

以下是按主题分类的推文摘要：

**GPT-4o 模型性能与特性**

- **GPT-4o 改进的代码编写和指令遵循能力受到赞赏**：[@sama](https://twitter.com/sama/status/1905419197120680193) 强调了 **新版 GPT-4o** 在 **代码编写、指令遵循和自由度** 方面表现尤为出色。[@kevinweil](https://twitter.com/kevinweil/status/1905419868071231993) 表示赞同，称 **GPT-4o 的更新非常强劲，并鼓励用户尝试**。
- **GPT-4o 相对于其他模型（特别是在代码和推理方面）的表现评估**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1905563427776651344) 报告称，**GPT-4o（2025年3月版）现在是领先的非推理代码模型**，在 **Artificial Analysis 代码指数** 中超越了 **DeepSeek V3** 和 **Claude 3.7 Sonnet**，并在 **LiveCodeBench** 中排名 **第一**。然而，它 **仍然落后于像 o3-mini 这样的推理模型**。
- **对政策合规性的担忧**：[@joannejang](https://twitter.com/joannejang/status/1905681602619085042) 指出，**图像生成拒绝通常是由于模型对政策产生了幻觉**。他们请求用户 **在他们努力让模型遵循政策的过程中保持耐心**，并建议如果遇到问题，请 **在新的对话中重试**。
- [@nrehiew_](https://twitter.com/nrehiew_/status/1905414817034150362) 假设 **4o 的图像生成工作原理是通过编码器直接嵌入图像，使用 AR（自回归），然后基于 AR 处理后的隐藏状态进行扩散**；所谓的 **模糊效果是一种心理战（psyop）**，并且 **没有使用 VQ**。
- **GPT-4o 的透明度和背景生成功能受到关注**：[@giffmana](https://twitter.com/giffmana/status/1905407013103747422) 注意到可以要求 **GPT-4o 图像生成透明背景**，称这是一个被吉卜力化（Ghiblification）热潮淹没的酷炫功能。

**Gemini 2.5 Pro 模型性能与能力**

- **Gemini 2.5 Pro 在音频和视频理解方面的能力受到称赞**：[@_philschmid](https://twitter.com/_philschmid/status/1905566076781371642) 报告称，**Gemini 2.5 Pro** 具有 **改进的长上下文能力**，可以 **通过单个请求处理约 1 小时的视频**，并指出 **YouTube 链接已集成到 AIS 和 API 中**。该模型还可以 **在单个请求中处理约 2 小时的播客转录**。
- **Simple-Bench AI 解释性能**：[@scaling01](https://twitter.com/scaling01/status/1905729393756180985) 提到 **Gemini 2.5 Pro Thinking 在 AI Explained 的 Simple-Bench 上得分约为 51.6%**，是 **第一个得分超过 50% 的模型**。
- **可访问性与使用**：[@_philschmid](https://twitter.com/_philschmid/status/1905555766179684587) 宣布用户可以 **自带 API Key 到** [@cursor_ai](https://twitter.com/cursor_ai) 来使用 **Gemini 2.5 Pro**，但指出 **目前的速率限制（rate limits）较低**。他们还提到 **Gemini 2.5 Pro 已在** [@windsurf_ai](https://twitter.com/windsurf_ai) 中可用。

**AI 基础设施与算力**

- **GPU 使用量预计将显著增加**：[@saranormous](https://twitter.com/saranormous/status/1905451945713909790) 表示 **他们将使用所有的 GPU（和 TPU）**。
- **Together AI 和 Hypertec Group 合作交付大规模 GPU 集群**：[@togethercompute](https://twitter.com/togethercompute/status/1905632314878800044) 宣布与 [@HypertecGroup](https://twitter.com/HypertecGroup) 建立合作伙伴关系，以交付 **数千个 GPU 的集群**，强调 **高带宽网络、先进冷却技术和强大的容错能力**。
- **CoreWeave 的 IPO**：[@weights_biases](https://twitter.com/weights_biases/status/1905641235395547203) 祝贺 [@CoreWeave](https://twitter.com/CoreWeave) 成功 IPO，强调了他们在推动 AI 基础设施极限方面的成功。

**AI 工程与开发**

- **关于传统编程语言优于 vibe coding 的担忧**：[@lateinteraction](https://twitter.com/lateinteraction/status/1905447832099983564) 强调了保留传统编程语言中有用方面的重要性，例如**定义函数、控制流和模块**，而不是屈服于 "vibe coding"。
- **开源在医疗 AI 中的重要性**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905582817276494155) 强调了**开源在医疗 AI 中的关键作用**，因为**需要透明度，且将敏感患者数据发送到云端 API 是不切实际的**。
- **强调 ASI 的可扩展解决方案**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1905460812057108658) 指出了一份关于构建 **ASI 可扩展解决方案**的声明，重点关注通过**投入更多计算和数据资源**来实现改进。
- **Langchain 与 Redis 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1905691477906522473) 宣布通过 `langgraph-checkpoint-redis`，你可以将 [@Redisinc](https://twitter.com/Redisinc) 强大的内存能力引入你的 LangGraph agents。

**公司与产品发布**

- **Keras 新主页上线**：[@fchollet](https://twitter.com/fchollet/status/1905391839055950032) 宣布为庆祝 Keras 成立 10 周年，**推出了全新的主页**。
- **C.H. Robinson 使用 LangGraph 节省时间**：[@LangChainAI](https://twitter.com/LangChainAI/status/1905667121465774379) 报道称，**C.H. Robinson** 正在使用基于 **LangGraph, LangGraph Studio, 和 LangSmith** 构建的技术来**自动化日常邮件交易**，从而**每天节省 600 多个小时**。
- **MIT NLP 小组账号上线**：[@lateinteraction](https://twitter.com/lateinteraction/status/1905411805343875276) 宣布 [@nlp_mit](https://twitter.com/nlp_mit) 账号上线，旨在展示来自 MIT 实验室的最新 NLP 研究成果。
- **Perplexity AI 线程基础设施问题**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1905652310501675318) 提到 Perplexity AI 正在经历一些基础设施（infra）挑战，这就是导致历史线程无法加载的原因。

**幽默/梗**

- **各种幽默推文**：几位用户分享了幽默内容，包括 [@Teknium1](https://twitter.com/Teknium1/status/1905677763228713225) 发布了带有图片的 **"Jensen rn"**；[@teortaxesTex](https://twitter.com/teortaxesTex/status/1905448971411013792) 发布了**习在第三次世界大战去世后转生到平行世界成为正太**的内容；[@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1905445347562062030) 建议**如果你在 ChatGPT 中生成异性的自己并觉得平平无奇，那你可能应该降低你的择偶标准**；以及 [@_philschmid](https://twitter.com/_philschmid/status/1905560675495129324) 指出 [@cursor_ai](https://twitter.com/cursor_ai) 刚刚对他们进行了 Rickroll。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. 逆向工程 GPT-4o：架构洞察与推测**

- **通过 Network 标签页对 GPT-4o 图像生成进行逆向工程——这是我的发现** ([Score: 599, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1jlptqu/reverse_engineering_gpt4o_image_gen_via_network/)): 作者通过检查网络流量研究了 **GPT-4o** 的图像生成过程，发现后端返回的中间图像表明可能存在一个多步流水线（multi-step pipeline）。他们推测该模型是使用了 Diffusion 过程还是 Autoregressive 方法，并指出 **OpenAI model card** 将其描述为一个 Autoregressive 模型。作者引用了 **OmniGen 论文** 作为对 GPT-4o 能力的潜在解释，强调其使用了基于 Transformer 的架构，该架构能够随高质量数据和计算能力的提升而良好扩展。
  - 关于 **GPT-4o** 模型是使用 **Diffusion 模型** 还是 **Autoregressive 模型** 存在争议。一些评论者推测它可能采用了带有 Diffusion 模型的层次化解码器（hierarchical decoder）来处理像素级细节，而另一些人则认为它使用 Autoregressive 方法，通过以复杂方式预测 Token 序列来增强图像生成。
  - 讨论了**开源竞争对手**达到 GPT-4o 质量水平的潜力，一些人预计中国竞争对手可能会在一年内实现这一目标。然而，其他人认为开源模型可能要到 **2025** 年底才能赶上，并强调了拥有一个类似于 LLM 领域中 **LLaMA** 的开源图像模型的重要性。
  - 评论者对个人逆向工程工作的价值表示怀疑，指出更广泛的学术界和工业界（尤其是中国）可能正在进行深入分析。人们对该模型访问互联网和利用高质量数据的能力是否比 **CLIP**/**T5** 等本地文本编码器具有显著优势表现出浓厚兴趣。


**主题 2. MegaTTS3 的语音克隆：质疑与安全担忧**

- **[来自字节跳动的新 TTS 模型](https://github.com/bytedance/MegaTTS3)** ([Score: 143, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1jlw5hb/new_tts_model_from_bytedance/)): **ByteDance** 发布了 **MegaTTS3**，这是一款新的文本转语音模型，其语音克隆功能引发了争议。讨论集中在伦理影响以及该技术在创建未经授权的语音副本方面可能存在的滥用。
  - **MegaTTS3 的特性与局限性**：该模型拥有 **0.45B 参数** 的**轻量化效率**、**双语支持**以及**可控的口音强度**。然而，由于“安全问题”，**WaveVAE 编码器**无法用于本地语音克隆，这引发了对其“超高质量语音克隆”虚假宣传的批评。
  - **伦理与安全担忧**：人们对不发布语音克隆软件的“安全原因”表示怀疑，许多人认为这只是为了**数据收集**以改进其模型的幌子。批评者认为，鉴于 AI 语音克隆技术的广泛普及，这种做法与伦理考量相悖。
  - **社区反应与批评**：用户对语音克隆能力的误导性宣传表示沮丧，并质疑为训练目的而**提交数据**的伦理问题。一些人将“安全”声明视为通过收集用户数据进行进一步训练的间接变现策略。


**主题 3. Qwen-2.5-72b：引领开源 OCR 革命**

- **[Qwen-2.5-72b 现已成为最佳开源 OCR 模型](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr)** ([Score: 119, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1jm4agx/qwen2572b_is_now_the_best_open_source_ocr_model/)): **Qwen 2.5 VL (72b 和 32b)** 模型已脱颖而出，成为领先的开源 OCR 模型，在 JSON 提取方面实现了约 **75% 的准确率**，与 **GPT-4o** 相当。**72b 模型**的表现略优于 **32b 模型**（高出 **0.4%**），且两者都超过了 **mistral-ocr** 模型 **72.2%** 的准确率。令人惊讶的是，尽管 **Gemma-3 (27B)** 的架构基于高性能的 **Gemini 2.0**，其得分仅为 **42.9%**。基准测试数据和方法论可在 [GitHub](https://github.com/getomni-ai/benchmark) 和 [Hugging Face](https://huggingface.co/datasets/getomni-ai/ocr-benchmark) 上获取。
  - **Ovis2 模型**尽管在 OCRBench 上处于领先地位且参数量显著更少（少 18 倍），但未被纳入讨论，这表明人们对其相对于 Qwen 模型的表现可能存在兴趣。
  - 许多人对来自 [Hugging Face](https://huggingface.co/allenai/olmOCR-7B-0225-preview) 的 **olmOCR-7B-0225-preview** 模型的表现感到好奇，该模型以更高效的 VRAM 利用率著称，凸显了市场对平衡性能与资源消耗的模型的需求。
  - **Qwen 2.5 VL 32B 模型**已更新，与较旧且近期未获更新的 72B 模型相比，显示出显著的性能提升。此外，32B 模型在写作能力上也优于原生 Qwen 模型。


## 其他 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

> 我们的流水线（pipelines）挂了...

---

# AI Discord 汇总

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. GPT-4o 统治排行榜并引发辩论**

- [**GPT-4o 跃升至 Arena 第 2 名，编程实力得到证实**](https://fxtwitter.com/lmarena_ai/status/1905340075225043057)：最新的 **ChatGPT-4o (2025-03-26)** 模型在 Arena 排行榜上飙升至 **第 2 位**，超越了 **GPT-4.5**，并在 Coding（编程）和 Hard Prompts（困难提示词）分类中并列 **第 1**。用户注意到其性能有显著飞跃，且与之前的模型相比成本降低了 **10 倍**，尽管 API 快照的价格差异引起了一些困惑。
- [**尽管有基准测试背书，GPT-4o 的编程技能仍评价不一**](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19)：虽然基准测试将 **Gemini 2.5 Pro** 定位为领先的非推理模型，但一些用户发现 **GPT-4o** 在编程任务中表现更优，特别是在指令遵循和代码生成方面。关于 **GPT-4o** 的高排名是源于针对特定响应风格的专门训练，还是源于原始性能，争论仍在继续。
- [**GPT-4o 揭晓为自回归图像模型**](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf)：**GPT-4o** 被证实采用 **Autoregressive**（自回归）方法进行图像生成，这标志着一种直接从文本提示词创建图像的新颖方法。有推测认为该模型为了提高效率，复用了 **image input 和 image output tokens**。

**主题 2. DeepSeek V3 和 Qwen2.5-Omni 成为强力竞争者**

- [**DeepSeek V3 在 SWE-bench 上编程表现优于 GPT-4o**](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/)：新的 **DeepSeek V3 0324** 模型在编程实力方面获得认可，据报道在 SWE-bench 基准测试中超越了 **GPT-4o R1**。数据表明 **DeepSeek V3** 在非推理编程任务中超过了 **Claude 3.7 Sonnet**，成为该领域的领先模型。
- [**Qwen2.5-Omni：Meta 的多模态力作登场**](https://qwenlm.github.io/blog/qwen2.5-omni/)：**Qwen2.5-Omni** 作为 **Qwen** 系列的最新旗舰模型发布，是一款端到端的多模态模型，能够处理文本、图像、音频和视频，并提供实时流式响应。用户可以在 [Qwen Chat](https://chat.qwenlm.ai) 测试 **Qwen2.5-Omni**，这标志着向真正通用的 AI 模型迈出了重要一步。
- [**紧随 GPT-4o 步伐，DeepSeek 融合 Diffusion 与 Transformers**](https://fxtwitter.com/DeanHu11/status/1903983295626707271)：**DeepSeek** 正在采用类似于 **GPT-4o** 的多模态架构，结合了 **Diffusion 和 Transformers**。这种此前在视觉模型中出现的方法，预示着多模态 AI 开发的一个日益增长的趋势。

**主题 3. 基础设施问题和用户沮丧情绪困扰 AI 平台**

- [**Perplexity AI 服务器压力过载，用户报告宕机与数据丢失**](https://status.perplexity.com/)：**Perplexity AI** 经历了大范围的**宕机**，用户报告历史记录和 Spaces 消失。官方状态页面（[status.perplexity.com](https://status.perplexity.com/)）更新缓慢，引发了用户对改善宕机沟通和自动化报告系统的呼吁。
- [**Manus.im 积分系统因高昂成本引发用户抵制**](https://manus.im/help/credits)：**Manus.im** 的新积分系统因其高昂的感知成本面临沉重批评，部分用户估算每月费用可能达到 **$500**。从基于任务的系统向基于积分的系统转变被描述为“令人不适”，严重影响了用户体验。
- [**Cursor IDE 遭遇数据库灾难，导致全服务宕机**](https://status.cursor.com/)：由于**数据库部署问题**，**Cursor** 经历了全服务宕机，中断了核心 AI 功能和通用服务功能。虽然在几小时后得到解决，但该事件凸显了 AI 驱动的编程工具的脆弱性及其对稳健基础设施的依赖。

**主题 4. 增强型 AI 开发工具与技术涌现**

- [**LM Studio 0.3.14 发布细粒度多 GPU 控制功能**](https://lmstudio.ai/download)：**LM Studio 0.3.14** 为多 GPU 配置引入了高级控制，允许用户微调 GPU 分配策略并更有效地管理资源。新的键盘快捷键（`Ctrl+Shift+H` 或 `Cmd+Shift+H`）提供了对 GPU 设置的快速访问。
- [**Aider 的新 /context 命令实现代码库上下文管理自动化**](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809)：**Aider** 引入了 `/context` 命令，该命令可根据用户请求自动识别并向聊天中添加相关文件。这一功能简化了上下文管理，尤其是在大型代码库中，为开发者节省了时间和精力。
- [**DSPy 框架提倡声明式编程，取代脆弱的 Prompting**](https://dspy.ai/)：**DSPy** 作为一个用于对语言模型进行“编程”而非依赖传统 Prompting 的框架而受到关注。它允许使用 Python 代码和算法快速迭代模块化 AI 系统，以优化 Prompt 和模型权重，旨在实现更稳健、高质量的 AI 输出。

**主题 5. 伦理考量与 AI 安全仍是核心**

- [**OpenAI 放宽图像生成政策，优先防止现实世界伤害**](https://x.com/joannejang/status/1905341734563053979)：**OpenAI** 调整了 **ChatGPT 4o** 中的图像生成政策，从一味拒绝转变为更细致的方法，重点在于防止现实世界的伤害。这一政策变化允许在之前受限的领域拥有更大的创作自由。
- [**AI 安全讨论强调 Constitutional AI 和 Jailbreak 问题**](https://generalanalysis.com/blog/jailbreak_cookbook)：关于 AI 安全的讨论强调，像 **Claude** 这样基于 Constitutional AI 原则设计的模型，优先考虑客观性而非用户偏好，这可能会影响排行榜排名。诸如 [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) 之类的资源被分享，旨在应对 LLM 漏洞和安全措施。
- [**宫崎骏 9 年前对 AI 艺术的批评再次浮现，引发伦理辩论**](https://x.com/nuberodesign/status/1904954270119588033)：一段宫崎骏批评 AI 生成艺术的旧片段重新浮现，再次引发了 AI 社区内的伦理讨论。这场辩论将 AI 艺术采样与快时尚伦理进行了类比，质疑了这种易于获取、可能具有剥削性的内容的道德性。

---

# PART 1: High level Discord summaries

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户对 Manus 新积分系统表示愤怒**：用户对新的积分系统（Credit System）感到沮丧，一些人估计体面的使用强度下成本可能达到 **$500/月**，而且 **1000 个免费积分**消耗极快，即使任务失败也会扣分，详情见 [manus.im/help/credits](https://manus.im/help/credits)。
   - 社区指出，“从基于任务到基于积分的转变确实让人感到突兀，尤其是这并非原始 Beta 测试流程的一部分”。
- **Manus 农场构思替代能源**：一位成员建议 **Manus** 可以开发廉价的可再生能源，如熔盐、热能或太阳能，为自己的 GPU 农场供电以降低成本，并可能将其设在沙漠中。
   - 该成员提议使用飞轮作为储能装置，以保持农场在夜间以最高效率运行。
- **Manus 考虑使用像 Deepseek 这样更便宜的 AI 模型**：社区正在讨论使用像 **Deepseek** 和 **Qwen** 这样更便宜的 AI 模型来代替仅使用 **Anthropic** 的 **Claude**，以降低运营成本。
   - 目前尚未说明 **Manus** 是否会允许其他 AI 集成。
- **学生利用 Manus AI 在考试中作弊**：学生们将 **Manus** 与 **Kimi** 或 **Deepseek** 结合使用，上传研讨会和讲座文件，要求 AI 记忆这些内容以备考，一些人在作业中获得了 **81/100** 的分数。
   - 一些用户想知道，如果帮助 AI 为学校作弊，是否违反了服务条款。
- **UI 设计被赞誉为简约的天才**：多位成员称赞了 **Manus** 的 UI 设计，表示该设计非常出色、易于使用、简洁且符合现实世界的概念。
   - 一位用户表示：“让 Manus 感觉如此惊艳的不仅是得到的结果，还有任务的理念与现实世界概念的高度契合。这种简单性堪称天才。”



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 服务器遭受冲击**：多位用户报告了**宕机以及历史记录/空间（Spaces）消失**的情况，引发了幽默的调侃和挫败感，官方状态页（[status.perplexity.com](https://status.perplexity.com/)）缺乏及时的更新。
   - 用户建议建立一个自动化的用户报告宕机系统和主动通知，以应对[这条推文](https://x.com/AravSrinivas/status/1905652310501675318)中提到的基础设施挑战。
- **DeepSeek AI 表现不佳**：成员们对 **DeepSeek AI** 表示失望，理由是它在处理复杂指令时表现吃力，且倾向于产生“不必要的术语”。
   - 用户将其与更优秀的数学应用进行了对比，突显了 **DeepSeek AI** 在解决实际问题上的短板。
- **Claude AI 的上下文窗口受到质疑**：围绕 **Claude AI 相对于 Gemini 和 ChatGPT 的上下文窗口（Context Window）限制**展开了讨论，许多成员注意到了 Claude 的局限性。
   - 成员们一致认为，与竞争对手（尤其是 Gemini）相比，**Claude 的上下文窗口**限制特别明显。
- **通过 T-Mobile 获取免费 Perplexity Pro**：用户交流了通过 **T-Mobile** 和 Revolut 促销活动获取**免费 Perplexity Pro 订阅**的方法。
   - 一位用户甚至建议利用 **T-Mobile** 的临时号码来享受该优惠，另一位用户链接了关于 [Perplexity 发布语音听写功能](https://x.com/testingcatalog/status/1905390832225493429?s=61)的推文。
- **Sonar API 存在 Llama Index RAG 集成问题**：一位用户询问如何有效地将 **Llama Index RAG** 上下文传递给 **Perplexity Sonar** 模型，寻求关于利用索引对象的建议。
   - 该用户还质疑 API 中的 **Deep Research** 功能是否能与 perplexity.com 版本保持一致，并指出感知到的性能差距，同时提到 **Sonar API** 有时会遗漏引用（Citations）。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek 3.1 潜入 Cursor**：一位 Cursor 团队成员提到 **DeepSeek 3.1** 应该会在 12 小时内集成到编辑器中，但定价细节尚未披露。
   - Cursor 提供[与供应商的优惠](https://cursor.com/deals)以及确保不存储数据的**隐私模式 (privacy mode)**。
- **Cursor 在数据库灾难中陷入停滞**：由于其基础设施内的**数据库部署问题**，Cursor 经历了全服务中断，导致 Chat 和 Tab 等 AI 功能以及常规服务受阻。
   - 几小时后，问题得到解决，他们更新了 [Cursor Status](https://status.cursor.com/)。
- **人形机器人热度升温**：成员们讨论了人形机器人的实用性，观点包括将其视为*做饭和清洁*助手，以及对[数据隐私](https://en.wikipedia.org/wiki/Data_privacy)和遥测的担忧。
   - 一位成员假设 **AGI 将从机器人技术中诞生**，先在虚拟环境中发展，然后再在现实世界中体现。
- **@Codebase 标签功成身退**：用户注意到 **@Codebase 标签**被移除，工作人员澄清它已被一种类似的扫描当前索引项目的方式取代，如 [changelog](https://cursor.com/changelog) 中所述。
   - 这引发了关于 **token 限制**、定价模型以及在 AI 编程工具中平衡便利性与控制权的讨论。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O1 Pro 即将登上排行榜？**：成员们讨论了 **O1 Pro** 加入排行榜的可能性，推测 **OpenAI** 可能会承担费用以展示其在高价位下的能力。
   - 然而，一些成员对其排行榜表现和延迟表示怀疑。
- **GPT-4o 的编程能力引发争议**：成员们在最近的更新后讨论了 [GPT-4o 的编程能力](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19)，一些人注意到其在指令遵循和代码生成方面的改进。
   - 然而，需要适当的评估 (evals)，因为一位成员认为 **GPT-4o** 的排名可能由于针对首选响应风格的专门训练而虚高，而非实际性能。
- **DeepSeek V3 在编程基准测试中实现飞跃**：新的 **DeepSeek V3 0324** 模型正获得认可，根据 [这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/)，一位成员指出它在 **SWE-bench** 上的得分高于 **GPT-4o R1**。
   - 数据表明，DeepSeek 的 **V3 0324** 版本在非推理领域超越了 **Claude 3.7 Sonnet**，并已成为领先的非推理编程模型。
- **Meta 的 Llama 模型变得古怪**：成员们观察到竞技场中最近出现的匿名模型（据信来自 **Meta**）表现出古怪行为，包括添加大量表情符号并自称为 **Meta Llama** 模型。
   - 正在测试的模型包括：`bolide`、`cybele`、`ginger`、`nutmeg`、`phoebe`、`spider`、`themis`，尽管他们也注意到 `spider` 有时会自称为 GPT-4。
- **AI 安全讨论**：成员们讨论了 AI 安全，提到像 **Claude** 这样的模型是基于宪法 AI (constitutional AI) 原则设计的，优先考虑客观性而非用户偏好，这可能会影响其排行榜排名。
   - 一位成员还分享了一个用于 LLM 越狱和 AI 安全的 [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) 资源，包括一个包含系统性越狱实现的 [GitHub 仓库](https://github.com/General-Analysis/GA)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Scribe V1 驱动 FoxMoans！**：一名成员使用 **11Labs Scribe V1** 进行音频事件分类，创建了一个 [话语列表](https://github.com/zero2rizz/FoxMoans/blob/main/UtteranceList.txt)，预估成本为 **$20k**。
   - 它被用于音频事件分类，适用于需要基于情绪分析的项目。
- **OlmOCR 的 Unsloth 集成仍不稳定**：尽管 **Qwen2VL** 可以正常工作，但一名成员在 Unsloth 中加载 **OlmOCR**（**Qwen2VL** 的一个微调版本）时遇到困难。
   - Unsloth 团队询问用户是否尝试了最新版本，因为他们在*创作者意识到模型完成上传之前*就推送了更新和修复。
- **Orpheus TTS 获得微调支持**：Unsloth 团队发布了一个用于[微调 **Orpheus-TTS** 的 Notebook](https://x.com/UnslothAI/status/1905312969879421435)，强调其具有情感线索的类人语音。
   - 成员们讨论了更改 **Orpheus** 语言的问题，建议使用新的嵌入层/头部层（embedded/head layers）进行持续预训练可能就足够了。
- **BOS Token 的双重麻烦**：一名用户在检查分词器（tokenizer）解码时，发现最新的 **Unsloth 更新 (Gemma 3 4B)** 存在 **双重 BOS Token** 问题。
   - 一个[热修复补丁](https://github.com/unslothai/unsloth-zoo/pull/106)已被确认，该补丁移除了意外添加的 Token。
- **DeepSeek-R1 发布量化版**：Unsloth 提供了各种版本的 **DeepSeek-R1**，包括 [GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf) 和 [4-bit 格式](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5)。
   - Unsloth 的 **DeepSeek-R1** 1.58-bit + 2-bit 动态量化（Dynamic Quants）通过选择性量化，比标准的 1-bit/2-bit 提高了精度。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o vs Gemini 2.5：编程大比拼**：成员们比较了 **GPT-4o** 和 **Gemini 2.5 Pro** 的编程能力。尽管基准测试显示 Gemini 2.5 Pro 整体表现更好，在 6 个类别中赢了 3 个，但一些人仍认为 **GPT-4o** 更胜一筹。
   - 观点各异，一些人更青睐 *Gemini* 处理特定任务，如 C++ 和 WinAPI 集成。
- **Google AI Studio：新的免费层级英雄**：用户们称赞 **Google AI Studio** 免费提供 **Gemini 2.5 Pro** 等模型，且 Prompt 限制非常慷慨，超过了 **ChatGPT Plus** 等付费服务。
   - 一些成员报告每天发送数百条消息也未达到限制，甚至因为这些优势取消了他们的 *ChatGPT* 订阅。
- **Perplexity 在新闻领域超越 ChatGPT**：成员们发现 **Perplexity** 凭借其 Discover 标签页在新闻和时事方面表现出色，强调它不仅仅是一个 *GPT 套壳*。
   - 然而，一些人指出 **Perplexity** 的 *Deep Research* 功能在上传文件的质量和可靠性方面存在问题，建议改用 *ChatGPT*。
- **Claude 3.7 Sonnet 的推理实力**：成员们赞扬 **Claude 3.7 Sonnet** 与其他 AI 模型相比具有卓越的推理能力和解释能力，尤其是考虑到*免费层级的 Claude 额度很快用完并强制开启新对话*。
   - 推荐使用 o1、o3-mini-high 和 Grok 3 等替代模型进行编程，其中 o1 在使用 C++、物理、渲染和 Win32API 等旧 API 的复杂任务中更受青睐。
- **增强的图像 Prompt：新曙光？**：用户对新版 **ChatGPT** 图像工具对复杂 Prompt 的遵循能力赞不绝口，例如生成一个在巨龟背上的移动市场，带有太阳和三个月亮。
   - 更新后的工具在针对性图像修改方面表现出色，例如在不影响整个图像的情况下移除夜景中的星星。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro：用户遭遇速率限制瓶颈**：用户在集成自己的 **AI Studio API keys** 后，依然遇到了 [**Gemini 2.5 Pro** 的低速率限制](https://x.com/OpenRouterAI/status/1905300582505624022)，引发了关于如何最大化免费配额的讨论。
   - 一位成员指出该模型*不会永远免费*，当不可避免地开始收费时，这将成为一个问题。
- **OpenRouter AI SDK 提供商选项困扰调试人员**：成员们正在积极调试 [**OpenRouter AI SDK** 提供商选项](https://github.com/OpenRouterTeam/ai-sdk-provider)，特别是使用 `providerOptions` 来设置模型顺序和回退行为。
   - 核心问题围绕在 `provider` 键下嵌套 **order 数组** 的正确方式，因为调试尝试显示，尽管进行了配置，仍会出现非预期的提供商选择。
- **免费 LLM 中的 Function Calling 热潮**：成员们正在寻找支持 Function Calling 的免费模型，**Mistral Small 3.1** 和 **Gemini 免费模型** 成为热门选择。
   - 一位沮丧的成员感叹道：*天哪，我正努力寻找一个支持 Function Calling 的免费模型，但一个也找不到！*。
- **Gemini Flash 2.0 在 TPS 对决中表现强劲**：社区正在热烈讨论各种编程模型的 **tokens per second (TPS)** 性能，**Gemini Flash 2.0** 因其极快的速度而备受推崇。
   - 尽管有这些宣传，一些用户仍持批评态度，指出它很*烂*，因为*他们的托管搞得一团糟*；另一位成员则宣称 **Groq** 运行 **70B R1 distil 的速度达到 600tok/s**，还有人插话称它*在我看来不擅长编程*。
- **OpenAI Responses API 支持？**：一位成员询问 [**OpenRouter** 是否支持 **OpenAI Responses API**](https://platform.openai.com/docs/api-reference/responses)。
   - OpenRouter 团队建议 [**Veo2 API**](https://openai.com/blog/veo) 是获取 **SOTA 图像转视频** 的最佳选择，但价格约为 **每秒视频 50 美分**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **通过 Prompt ICL 实现最佳 Tool Use**：成员们讨论了如何引导 Agent 进行 **tool usage**，参考了 [Cline 的系统提示词](https://github.com/cline/cline/blob/main/src/core/prompts/system.ts)，并建议直接在服务器上设置提示词，例如 `First call ${tool1.name}, then ${tool2.name}`。
   - 一位成员分享了[关于使用提示词进行 ICL 的链接](https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19)以及一个[显示其正常工作的测试](https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92)。
- **为 MCP 配置 Google Search**：一位成员询问如何将 **Google Search** 添加到 MCP，另一位成员分享了他们的 [配置](https://cdn.discordapp.com/attachments/1312302100125843479/1355126409093316750/config.json?ex=67e7cb50&is=67e679d0&hm=8311f31b3b6181eb391876bad03fc45f745e439a12180e6ad087d94983c37c1c&)。
   - 他们指出，用户需要获取自己的 **Google API key** 和 **engine ID** 才能使用该配置。
- **使用 Docker 部署海量 MCP 服务器**：一位成员创建了一个全能的 **Docker Compose** 设置，使用 Portainer 轻松自托管 **17 个 MCP 服务器**，Dockerfile 源自公共 GitHub 项目 ([MCP-Mealprep](https://github.com/JoshuaRL/MCP-Mealprep))。
   - 建议*除非需要远程访问，否则不要将容器绑定在 0.0.0.0*，并*在 readme 中包含一个 mcp 配置 json 示例*。
- **Agent 也能用 Canvas 了！**：一位成员创建了一个 **Canvas MCP** 服务器，使 AI Agent 能够与 Canvas LMS 交互，并添加了一个可以自主爬取 Gradescope 以查找信息的 Agent，项目地址为 [Canvas-MCP](https://git.new/canvas-mcp)。
   - 该工具提供查找相关资源、查询即将到来的作业以及访问 **Gradescope** 中的课程和作业等功能。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-4o 称霸 Coding Arena**: 最新的 **ChatGPT-4o** 更新在 [Arena 排行榜](https://x.com/lmarena_ai/status/1905340075225043057) 上跃升至第 2 位，在 Coding（编程）、Hard Prompts（困难提示词）类别中并列第 1，并在所有类别中均位列前 2，同时成本降低了 10 倍。
   - 该更新以 **chatgpt-4o-latest** 端点形式发布，价格为每百万输入/输出 Token $5/$15，而 API 快照版的价格为 $2.5/$10。根据 [Artificial Analysis](https://x.com/ArtificialAnlys/status/1905563427776651344) 的建议，在迁移工作负载时需保持谨慎。
- **OpenRouter R1 模型表现不佳**: 一位成员发现 OpenRouter 上的免费 **R1** 模型非常“愚蠢”、冗长，且在解决损坏的测试用例时无效，尤其是在启用 repomap 的情况下，表现不如 **O3-mini**。
   - 据推测，免费的 **R1** 模型是 **DeepSeek** 的量化版本（可能是 FP8 格式），而排行榜上的 DeepSeek 来自官方 DeepSeek 团队。此外，在 OpenRouter 上轮换使用多个 API Key 的用户可能会被封号。
- **Context Architecture 实现高效代码库处理**: 常量上下文架构 (**CCA**) 被提议作为使用 LLM 处理大型代码库的解决方案，确保修改任何模块所需的上下文始终能放入 LLM 的上下文窗口中，无论代码库的总规模如何，详见此 [博客文章](https://neohalcyon.substack.com/p/constant-context-architecture)。
   - 这是通过确保模块具有受限的大小、接口和依赖关系来实现的，从而使上下文收集成为一种有界操作。
- **速率限制困扰 Gemini 2.5 Pro 用户**: 多位用户报告遇到了 **Gemini 2.5 Pro** 的速率限制，即使似乎低于文档说明的 **50 次请求/天**，其中一位用户指出存在 **2 次请求/分钟** 的限制。
   - 讨论了购买付费账户是否能解决限制问题，报告结果不一，同时还讨论了潜在的备选模型实现。
- **Aider 的 Context 命令自动包含文件**: 新的 `/context` 命令可自动识别给定请求的相关文件并将其添加到对话中，详见 [此 Discord 讨论帖](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809)。
   - 这对于大型代码库特别有用，通过自动化手动添加文件的过程来节省时间。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4o 跃升至 Arena 第 2 名！**: 最新的 **ChatGPT-4o** (2025-03-26) 在 Arena 上跃升至 **第 2 名**，超越了 **GPT-4.5**，相比 1 月份的版本有显著提升（+30 分），参考 [此推文](https://fxtwitter.com/lmarena_ai/status/1905340075225043057)。
   - 它在 Coding 和 Hard Prompts 类别中并列 **第 1**。
- **OpenAI 放宽图像生成政策**: OpenAI 通过 **4o** 在 **ChatGPT** 中推出了原生图像生成功能，从一味拒绝转向更精准的方法，重点在于防止现实世界的伤害，详见 [此博客文章](https://x.com/joannejang/status/1905341734563053979)。
   - 新政策允许在敏感领域拥有更多的创作自由。
- **Devin 自动生成 Wiki 页面**: **Devin** 现在可以自动索引仓库并生成包含架构图和源码链接的 Wiki，参考 [此推文](https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)。
   - 该功能可帮助用户快速熟悉代码库中不熟悉的部分。
- **HubSpot 联合创始人加入 Latent Space**: **HubSpot** 联合创始人、**Agent.ai** 创始人 **Dharmesh Shah** 加入 Latent Space，讨论职场组织的下一次演变，重点关注 **混合团队 (hybrid teams)**。
   - 核心概念是 *人类员工与 AI Agent 作为团队成员进行协作*，这引发了关于团队动态、信任和任务分配的问题。
- **LLM 代码生成工作流详解**: 一位成员分享了他们的 [LLM 代码生成工作流](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)，强调 **头脑风暴规格说明 (specs)**、规划，并在离散循环中执行 LLM 代码生成。
   - 该工作流基于个人经验和互联网最佳实践，但作者承认 *它可能在两周内失效，或者效果翻倍*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 驯服多 GPU 配置**：**LM Studio 0.3.14** 引入了针对多 GPU 设置的细粒度控制，允许用户启用/禁用特定 GPU，并选择分配策略（如 **evenly** 或 **priority order**），可在此处[下载](https://lmstudio.ai/download)。
   - 键盘快捷键 `Ctrl+Shift+H` (Windows) 或 `Cmd+Shift+H` (Mac) 可快速访问 GPU 控制，而 `Ctrl+Alt+Shift+H` (Windows) 或 `Cmd+Option+Shift+H` (Mac) 可在模型加载期间打开弹出窗口管理设置。
- **Threadripper 碾压 EPYC**：一项讨论对比了 **Threadripper** 和 **EPYC**，澄清了虽然 **Threadripper** 在技术上属于 HEDT（高端桌面），但 AMD 并不向家庭用户推广 **EPYC**。
   - [GamersNexus 的评测](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks)强调了 **AMD Ryzen Threadripper 7960X** 的 24 核心以及对于工作站而言相对较低的成本。
- **LLM 计算迎来视觉化革新**：成员们讨论了将 LLM 执行的计算可视化，例如将数值映射到像素颜色，并推荐了 [LLM Visualization 工具](https://bbycroft.net/llm)。
   - 为了深入理解，分享了 3b1b 关于 LLM 的播放列表以及一本关于从零开始构建 LLM 的书。
- **P100 被 6750xt 彻底击败**：一位成员询问是否可以将 **P100 16GB** 用于业余项目，但遭到了强烈反对，一位用户表示与 **6750xt** 相比，它基本上就是“电子垃圾”。
   - **6750xt** 被推荐为更好且更现代的显卡，因为它支持 **Vulkan**，而 **P100** 不受支持的 **CUDA** 版本使其吸引力降低。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Transformer 存储错误消息误导用户**：一位用户发现，在 **transformers v4.50.0** 中，存储空间不足会导致误导性的错误消息；计划提交一个 PR 以实现更好的错误处理，并在下载模型分片前检查容量。
   - 由于库的错误消息不佳，该用户不得不使用 `df -h` 来诊断系统 **100% 满载** 的问题。
- **Torchtune 鼓励通过修改代码进行自定义**：用户发现 **torchtune** 需要下载并编辑 **200 行的 PyTorch 脚本** 和 YAML 文件来进行自定义，从而提供对流程的完整视图。
   - 据一位用户称，这种方法可以避免去剖析 **Hugging Face 的实现**。
- **偏见增强一致性训练验证内省能力**：受 [Anthropic 工作](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)的启发，成员们讨论了通过创建电路表示并将其反馈来模拟 LM 的自我意识。
   - 一篇关于 **偏见增强一致性训练 (BCT)** 的[论文](https://arxiv.org/abs/2403.05518v1)也被链接作为内省方法的验证手段。
- **自适应压缩旨在提升分布式系统**：一个旨在优化分布式系统中模型传输和部署的基础设施层正在开发中，利用自适应压缩和智能路由来解决 **带宽浪费** 和 **推理延迟** 问题。
   - 对 **分布式推理** 感兴趣的人可能会发现这个基础设施对扩展大型模型很有用，目前提供演示。
- **神经网络演变为“无器官身体”**：一位成员链接到一条[推文](https://x.com/norabelrose/status/1905336894038454396)，认为神经网络是 **无器官身体 (Bodies Without Organs, BwO)**，因为它们没有“器官”或“固定机制”，而是具有“信息流”。
   - 一位成员拒绝 **机械可解释性 (mechanistic interpretability)**，并表示神经网络在没有固定机制的情况下进行泛化，这在 400 年前就被笛卡尔预见到了。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **`tl.gather` 接近发布**：在等待正式发布期间，为了解决元素重复问题，成员指出可以按照 [此 Discord 线程](https://discord.com/channels/1189498204333543425/1189607595451895918/1336735886884208672) 中的说明从源码编译 **Triton**。
   - 团队还澄清说 *tl.gather 可以解决元素重复问题*，这也是其他成员针对 `torch.Tensor.expand()` 等函数向 **Triton** 提出的需求。
- **激活稀疏化加速 FFNs**：分享了一篇新论文，认为 LLM 中用于激活加速的 **2:4 sparsity** 可在不损失精度的情况下使 **FFNs 快 1.3 倍**，参见 [Acceleration Through Activation Sparsity](https://arxiv.org/abs/2503.16672)。
   - 一位成员指出，下一步是 `带有稀疏性的 FP4，以实现有效的 2-bit tensorcore 性能`。
- **CUDA Profiling 令人困惑**：鉴于 Nvidia 工具（如 **nvprof**、**Nvidia Visual Profiler (nvvp)** 和各种 **Nsight** 软件包）琳琅满目，一位用户正在寻求 **CUDA profiling** 的权威指南。
   - 另一位用户建议 **Nsight Compute** 是进行单算子（single kernel）分析的最佳工具，并附上了 [Nvidia 文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) 和 [详细演讲](https://www.youtube.com/watch?v=F_BazucyCMw&t=5824s) 的链接。
- **宫崎骏嘲讽 AI 艺术采样**：一个 **9 年前的梗** 重新浮现，展示了 [宫崎骏](https://x.com/nuberodesign/status/1904954270119588033) 在 Niconico 创始人展示 AI 生成艺术时的批判性反应。
   - 成员们将使用 AI 艺术的伦理与从 **Shein** 等快时尚公司购买商品进行了比较，称这种不道德的商业模式提供了获取廉价内容的途径。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI 和 xAI 构想的 AI 学校**：**OpenAI** 和 **xAI** 正在探索 AI 驱动学校的概念，可能会利用生成的图像作为课程内容，讨论指出根据 [此帖子](https://x.com/TheDevilOps/status/1905297966400770155)，*吉卜力工作室风格 (Ghibli Studio Style)* 是解决对齐问题的一种方案。
   - 这些举措旨在将 AI 更紧密地整合到教育框架中，重点是创建具有视觉吸引力且符合语境的学习材料。
- **Transformer Circuits 揭晓 Crosscoders**：**Transformer Circuits** 团队发布了关于 **sparse crosscoders** 的更新，这是 **sparse autoencoders** 的一种变体，可以读取和写入多个层，形成共享特征，详见其 [研究更新](https://transformer-circuits.pub/2024/crosscoders/index.html)。
   - 这些 **crosscoders** 解决了跨层叠加（cross-layer superposition）问题，监控持久特征，并简化了电路。
- **GPT-4o 确认为自回归图像模型**：在 **Yampeleg** 的 [帖子](https://x.com/Yampeleg/status/1905293247108219086) 和 [OpenAI System Card](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf) 发布后，成员们验证了 **GPT-4o** 是一款 **autoregressive image generation model**。
   - 这一发现揭示了该模型直接从文本提示创建图像的新颖方法，成员们推测 **GPT-4o** 复用了 **image input 和 image output tokens**。
- **Qwen2.5-Omni 引起多模态轰动**：**Qwen2.5-Omni** 是 **Qwen** 系列中最新的旗舰级 **end-to-end multimodal model**，已在成员中分享。它专为全面的多模态感知而设计，可处理文本、图像、音频和视频，详见 [Qwen Chat](https://chat.qwenlm.ai)。
   - **Qwen2.5-Omni** 通过文本生成和自然语音合成提供实时流式响应，树立了多模态交互的新标杆。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o 在 Arena 排名飙升，价格便宜 10 倍**：新的 **ChatGPT-4o (2025-03-26)** 模型在 Arena 排名跃升至第 2 位，超越了 **GPT-4.5**。据 [lmarena_ai](https://fxtwitter.com/lmarena_ai/status/1905340075225043057) 报道，其成本降低了 **10 倍**，并在 Coding 和 Hard Prompts 类别中并列 **第 1**。
   - 该模型目前在 Arena 的所有类别中均位列 **前 2**，在编程和处理复杂提示词方面表现出色。
- **马斯克的 xAI 以 800 亿美元交易吞并 X**：**Elon Musk** 透露，**xAI** 已通过全股票交易接管了 **X**。据 [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition) 报道，**xAI 的估值为 800 亿美元**，**X 的估值为 330 亿美元**（包括 120 亿美元债务）。
   - 此举将马斯克的 AI 事业整合到 **xAI** 旗下，可能会改变 AI 市场的竞争格局。
- **LlamaGen 像 LLM 一样生成图像**：**LlamaGen** 系列图像生成模型应用了来自大语言模型的 *next-token prediction* 范式来生成图像。根据 [LlamaGen 论文](https://arxiv.org/abs/2406.06525) 的描述，它在 ImageNet 256x256 基准测试中达到了 **2.18 FID**。
   - 该架构实现了 **0.94 rFID** 的重建质量和 **97%** 的 codebook 利用率，其图像分词器（image tokenizer）的下采样率为 **16**。
- **Qwen2.5-Omni 全能表现**：**Qwen2.5-Omni** 是 Qwen 系列中新的旗舰级端到端多模态模型，能够处理文本、图像、音频和视频。正如 [其博客文章](https://qwenlm.github.io/blog/qwen2.5-omni/) 所述，它支持通过文本和语音进行实时流式响应。
   - 该模型已在 [Qwen Chat](https://chat.qwenlm.ai) 上线，可能预示着新一波更通用模型的到来。
- **Gemini 2.5 Pro 在 Wordle 竞赛中表现优异**：**Gemini 2.5 Pro** 在 Wordle 游戏中展现了卓越的性能，能够逻辑推导出单词和字母位置，据 [Xeophon](https://x.com/TheXeophon/status/1905535830694773003) 报道。
   - 关于 **Gemini 2.5 Pro** 的反馈非常积极，[Zvi](https://x.com/TheZvi/status/1905003873422442642) 提到一位用户指出：*“我从未见过对一个非当下最热门话题（Current Thing）的 AI 发布有如此强劲且正面的反馈”*。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FP8 QAT 面临带宽瓶颈**：一位关注 [issue #1632](https://github.com/pytorch/ao/issues/1632) 的成员指出，**FP8 QAT** 已在 *TorchAO* 的计划中，但目前缺乏立即实施的带宽（人力/资源）。
   - 这表明了 **PyTorch** 生态系统中未来开发和贡献的一个潜在领域。
- **Torchtune 团队处理积压 Issue**：团队讨论了在处理积压的 issue 之前，优先进行 **PR 评审**和处理**新 PR**，估计 **80%** 的现有 issue 已经得到解决。
   - 为了更好地组织待评审的积压工作，一位成员建议除了现有的 GRPO 追踪器外，再增加一个通用的 **RL/RLHF 追踪器**。
- **Torchtune 计划集成 bitsandbytes**：一位成员建议使用 **Torchtune** 仓库中的 [issue #906](https://github.com/pytorch/torchtune/issues/906) 来引导对 **bitsandbytes** 集成的贡献。
   - 另一位成员幽默地表示他们对文档 PR 缺乏热情，但仍同意去查看一下。
- **Centered Reward Loss 支持奖励模型训练**：成员们讨论了在 **Torchtune** 中启用奖励模型训练，特别关注实现 **centered reward loss**（中心化奖励损失），例如 **(R1 + R2)² loss**。
   - 他们注意到当前的 **preference dataset** 格式需要 **不带 prompt 的 chosen/rejected 格式**。
- **vLLM 集成导致权重热交换 Hack 出现**：一位成员详细说明了在 **vLLM** 初始化期间的内存垄断问题，并分享了一个用于 [weight hotswapping](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep) 的*晦涩黑科技（obscure hack）*。
   - 另一位成员警告说 *“每个 vLLM 版本都会破坏一些东西”*，暗示当 vLLM 发布带有新 **v1 execution engine** 的 **0.8** 版本时，可能与现有的 hack 手段不兼容。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 获得了“王者级” UI**：用户报告称 **Claude** 推出了一个简洁的新 UI，一位用户特别喜欢该 UI 隐藏了所有从不使用的功能，称其为“王者之举（king move）”。
   - 目前唯一注意到的问题是缺少 **extended think** 的切换开关。
- **DeepSeek 抄了 GPT-4o 的作业**：**DeepSeek** 正在像 GPT-4o 多模态模型一样结合 **diffusion 和 transformers**，正如[这条推文](https://fxtwitter.com/DeanHu11/status/1903983295626707271)所指出的，该推文引用了视觉领域的一个类似想法。
   - 引用的论文在图像和视频上使用了自回归条件块注意力机制（autoregressive conditional block attention）进行实验。
- **TinyZero 的 30 美元 AI 模型首次亮相**：关注点正转向美国 **TinyZero** 最近的成就，特别是他们的 **30 美元模型**，以及 **VERL** 和 **Sky-T1** 等新发布的内容，正如这篇 [CNBC 文章](https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html)所报道的。
   - 当 DeepSeek 发布其 R1 并声称仅用 600 万美元就实现了其生成式 AI 大语言模型时，包括微软资助的 OpenAI 在内的美国 AI 市场领导者所花费的数十亿美元立即受到了审查。
- **LG 的 EXAONE 模型在存疑的许可证下发布**：**LG AI Research** 发布了 **EXAONE Deep**，这是一系列参数范围从 **2.4B 到 32B** 的模型，在包括数学和编程基准测试在内的推理任务中具有卓越的能力，详见其[文档](https://arxiv.org/abs/2503.12524)、[博客](https://www.lgresearch.ai/news/view?seq=543)和 [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-Deep)。
   - 有人指出，**EXAONE AI Model License Agreement 1.1 - NC** 明确保留了输出的所有权，但该许可证的执行力存疑。
- **Hermes-3 给用户留下深刻印象**：一位成员提到，到目前为止最令人印象深刻的模型是 **Hermes3 Llama3.2 3B**。
   - 未提供更多细节。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 投身 Diffusion-Transformer 混合架构**：根据链接到其论文的[这条推文](https://fxtwitter.com/DeanHu11/status/1903983295626707271)，**DeepSeek** 像 **GPT-4o** 多模态模型一样结合了 **diffusion 和 transformers**。
   - 作者指出，[视觉领域也出现了类似的想法](https://arxiv.org/abs/2412.07720)，在图像和视频上进行的实验标题几乎相同。
- **ZeroGPU 配额困扰用户**：用户报告 **zeroGPU quota** 无法重置的问题，其中一人链接到了[这个讨论](https://discord.com/channels/879548962464493619/1355122724820746374)以获取相关投诉。
   - 一位用户指出，即使配额用完，它也会在 *30 分钟或一小时后在一定程度上恢复*，但目前存在 Bug。
- **FactoryManager 推出 LinuxServer.io Docker 支持**：一位成员介绍了 [FactoryManager](https://github.com/sampagon/factorymanager)，这是一个包装了 **linuxserver.io 桌面环境容器** 的 **Python package**，能够实现对环境的编程控制，并通过使用两个不同桌面环境的演示进行了展示。
   - 该包旨在通过在 **linuxserver.io** 之上搭建脚手架来提供灵活性，这与 **Anthropic**、**OpenAI** 等公司的 GUI Agent 演示中经常创建的自定义环境有所不同。
- **Langfuse 毒性评估器误判胡萝卜**：一位在 Langfuse 中测试毒性 LLM-as-a-judge 的用户发现，它错误地将提示词 *“吃胡萝卜能改善视力吗？”* 标记为有毒，分数为 **0.9**，理由是与气候变化话语存在错误关联。
   - 该用户质疑 *“如何评估评估器”*，并指出 **GPT-4o** 将贬低性的气候变化内容错误地归因于一个关于胡萝卜的无害问题。
- **Base 与 Instruct 模型之争**：一位 Agent 领域的新手寻求关于 Base 模型和 Instruct 模型区别的澄清，并引用了课程中提到的 chat templates。
   - 一位成员用 **Base 模型** 是 *“裸模型，没有包装”* 的比喻进行了回应，并分享了[一篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/)进一步阐述了这些差异。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **思维导图功能圈粉**：一位用户对新的思维导图功能表示兴奋，称其为 *又一个令人惊叹的时刻*。
   - 未提供关于其具体用途的更多细节。
- **源文件上传受阻，陷入停滞**：有用户报告源文件一直处于永久上传状态超过 8 小时，导致既无法导入也无法删除。
   - 该用户寻求删除永久上传中源文件的建议，但未获成功。
- **版本控制缺失，用户感到烦恼**：一位用户对“Note”源类型缺乏版本控制和回收站支持表示担忧。
   - 该用户提到由于 Google Docs 具有更优的数据保护和备份功能，因此在犹豫是否使用该功能。
- **粘贴的源文件停止自动命名**：一位用户报告称，以前会自动命名的粘贴源文件，现在默认显示为 "pasted text"。
   - 该用户询问是否有更新或可以恢复到之前行为的方法。
- **PDF 解析问题依然存在**：用户讨论了 NLM 无法从扫描的 PDF 中提取数据的问题，其中一位用户询问该工具是否可以从扫描的笔记中提取数据。
   - 一位用户澄清说 **NLM 无法处理混合内容的 PDF**（文本和图像），但可以处理文档和幻灯片。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 庆祝 MCP 周**：LlamaIndex 重点介绍了 **LlamaCloud** 作为 **MCP server** 的功能，并演示了将 **LlamaIndex** 作为任何 **MCP server** 的客户端使用，从而可以访问许多作为工具的 MCP server，详见[此推文](https://twitter.com/llama_index/status/1905678572297318564)。
   - 他们展示了通过利用数百个现有的 MCP server 来大幅*扩展 Agent 能力*的可能性。
- **FunctionAgent 获得 ChatMessage 历史记录支持**：一位成员询问如何为 **FunctionAgent** 工作流添加聊天历史记录，并提供了[相关文档](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history)。
   - 提供的指导包括使用 `agent.run(...., chat_history=chat_history)` 覆盖聊天历史记录，或使用 `ChatMemoryBuffer.from_defaults(token_limit=60000, chat_history=chat_history)`。
- **遥测追踪获取用户 ID**：一位成员询问在与 Llama Index 交互时，如何传递自定义遥测属性以及在 LLM 网络调用中附加 header 或参数，并分享了一个 [Colab notebook](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing)。
   - 该 Colab notebook 展示了如何为代码块内执行的所有事件附加用户 ID。
- **LlamaParse PDF 解析问题**：一位用户报告称 **LlamaParse** 在处理单个 PDF 时正常，但在处理两个 PDF 并询问相同问题时失败，可能导致系统过载。
   - 该用户描述系统在处理多个 PDF 时*几乎崩溃（literally cooked）*，表明存在潜在的过载或处理错误。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 将模型命名为 "Command"**：一位成员询问为什么 **Cohere** 选择将其语言模型命名为 *Command*，并暗示类似于数据库管理，*query* 本质上就是一个 **command 或 instruction**。
   - 模型选择在 **Coral** 中可用，其中 *Just Chat* 在不使用外部源的情况下使用 **Command A**。
- **软件工程师寻求 Cohere 职业机会**：一位成员正在寻找软件工程师的新工作机会，并很乐意讨论与 **websites** 或 **web applications** 相关的潜在项目。
   - 另一位成员分享了 [Cohere 招聘页面](https://cohere.com/careers)的链接，鼓励该用户探索可用职位。
- **机器人命令进行测试运行**：鼓励成员在「🤖」bot-cmd 频道测试机器人命令，以确保功能正常和用户体验良好。
   - 欢迎对机器人命令提供反馈。
- **全栈架构师准备就绪**：一位拥有 **8 年以上**经验的热情开发者，擅长使用 **React, Angular, Flutter, 和 Swift** 等现代框架构建可扩展的 **web 和 mobile apps**。
   - 他们使用 **Python, TensorFlow, 和 OpenAI** 构建智能 **AI 解决方案**，并集成 **云技术 (AWS, GCP, Azure)** 和 **微服务** 以实现全球扩展。
- **Oracle 顾问寻求 Cohere 知识**：一位在 **Oracle ERP Fusion** 领域拥有 **12 年以上**经验的技术顾问渴望了解更多关于 **Cohere 模型**和企业级应用 **AI 使用案例**的信息。
   - 一位网络和计算机科学专业的学生目标是从事**开源生成式音乐**项目，倾向于使用 **ChatGPT, Grok, Windsurf, 和 Replit** 等技术工具。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 面临易用性投诉**：用户对 **GPT4All** 的易用性表示担忧，提到的问题包括无法导入模型、无法搜索模型列表、无法查看模型大小、无法使用 LaTeX 以及无法自定义模型列表顺序。
   - 一位用户认为 **GPT4All** *正在流失用户，因为其他平台更加用户友好且开放*。
- **GPT4All 在新模型实现方面滞后**：一位用户对 **GPT4All** 尚未实现 **Mistral Small 3.1** 和 **Gemma 3** 感到沮丧，并强调了这些模型的多模态能力。
   - 该用户表示，如果 **GPT4All** 到 2025 年夏天仍未赶上进度，他们可能会放弃 *Llama.cpp*。
- **GPT4All 因原生 RAG 和模型设置受到赞赏**：尽管存在批评，**GPT4All** 仍具有**原生 RAG** 和开箱即用功能等优势，一位用户表达了对开发者的信心以及对 **GPT4All v4.0.0** 的期待。
   - 另一位用户赞赏 **GPT4All** 的模型设置页面，认为其选项全面且模型重载按钮非常方便，并指出*在聊天菜单之外只需 2-3 次点击即可完成设置*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **成员被要求关闭过期的 PR 和 Issue**：George Hotz 要求成员关闭所有已过期的开放拉取请求 (PR) 和问题 (Issues)。
   - 此举旨在通过处理过时项来清理项目的仓库。
- **关于 TinyGrad Codegen 内部机制的讨论**：一位成员询问了 **TinyGrad 的代码生成 (Codegen)** 过程，特别是文档中提到的 `CStyleCodegen` 或 `CUDACodegen` 的位置。
   - 文档描述了 **TinyGrad** 使用不同的*转换器*（Renderers 或 Codegen 类），如 `C++ (CStyleCodegen)`、`NVIDIA GPUs (CUDACodegen)`、`Apple GPUs (MetalCodegen)`，将优化后的计划转换为 CPU/GPU 可以理解的代码。
- **探索布尔索引 (Boolean Indexing) 的实现**：一位成员寻求关于如何在带孔的网格上高效创建均匀分布点的建议（类似于 PyTorch 中的布尔索引），并认为这可能是对 **TinyGrad** 有用的贡献。
   - 一个 LLM 提出了一种使用 **masked_select** 的解决方案，通过利用条件 `full.abs().max(axis=1) >= (math.pi/6)` 过滤掉孔洞之外的点，从而高效地创建所需的带孔网格。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **解决 DSPy 输出验证失败问题**：一位成员询问 **DSPy** 如何处理输出验证失败，特别是当一个整数型字段期望 1 到 10 之间的数字却收到了 **101** 时。
   - 频道内没有关于此问题的进一步讨论或提供的链接。
- **深入研究 DSPy 优化器 (Optimizers)**：一位成员正在探索 **DSPy** 中 **optimizers** 的使用，以及它们如何与 docstrings 和提示词管理交互，并参考了 [DSPy 官方文档](https://dspy.ai/)。
   - 发现的问题是 **Optimizer 会覆盖来自 docstring 的提示词**，需要从 json 或 pkl 文件加载优化后的版本。
- **解码 DSPy 的优化过程**：会议澄清了 **DSPy 的优化器**会生成提示词并在数据集上进行测试以找出性能最佳的提示词，更多细节见[官方网站](https://dspy.ai/)。
   - 用户发现优化器如何选择 **N 个示例**包含在提示词中*非常有趣*，这展示了生成的提示词类型。
- **DSPy：声明式自我改进 Python 出现**：**DSPy** 是一个用于*对语言模型进行编程而非提示 (prompting)* 的框架，旨在快速迭代**构建模块化 AI 系统**，并提供**优化提示词和权重**的算法。
   - 你不再需要编写脆弱的提示词，而是编写组合式的 _Python 代码_，并使用 DSPy 来**教你的 LM 交付高质量的输出**。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **创业赛道导师缺席**：一位创业赛道的学生询问了 **LLM Agents Berkeley MOOC** 内部的导师指导机会。
   - 会议澄清 *Berkeley 不为创业赛道提供任何导师指导*，但赞助商将在 4/5 月举办答疑时间 (Office Hours)。
- **赞助商答疑时间公布**：赞助商将在 4 月/5 月为 **LLM Agents Berkeley MOOC** 创业赛道举办答疑时间。
   - 这为学生提供了与行业专业人士交流并寻求项目指导的机会。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Gemini 2.5 Pro 冲入 Windsurf**：**Gemini 2.5 Pro** 现已在 Windsurf 中可用，每条消息为用户提供 **1.0** 用户提示词额度，每次工具调用提供 **1.0** flow 操作额度；详见 [X 上的公告](https://x.com/windsurf_ai/status/1905410812921217272)。
   - 此更新旨在通过最新模型提升用户体验。
- **Windsurf 在 Gemini 2.5 Pro 速率限制上受阻**：在 **Gemini 2.5 Pro** 发布后不久，由于模型和供应商面临巨大负载，Windsurf 遇到了速率限制（rate limits）。
   - 团队正在努力增加配额，并对带来的不便表示歉意，目标是让每个人都能*尽快在 Gemini 2.5 Pro 上“冲浪”*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Foo[1] 默认为预定义值**：在 `Foo[1]` 类型的上下文中，`self` 参数可以自动填充默认参数值。
   - 当使用 `_` 丢弃 `self` 时，参数将默认为其预定义的默认值。
- **Self 参数澄清**：`self` 参数是带有默认参数值的 `Foo[1]`，可以使用 `_` 忽略。
   - 使用 `_` 忽略 `self` 会默认使用预定义的默认参数值。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

# 第二部分：各频道详细摘要和链接


{% if medium == 'web' %}

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1354892866832564237)** (627 条消息🔥🔥🔥): 

> `Manus 新积分系统反馈、Manus GPU 算力集群的替代能源、Deepseek 和 Qwen 等更便宜的 AI 模型、Manus AI 辅助考试、对 Manus UI 的喜爱` 


- **社区对 Manus 新积分系统的强烈抗议**：许多用户对新积分系统表示沮丧，认为其价格过高且限制过多，有人估计体面的使用量每月成本可能达到 **$500**，还有人很快就耗尽了 **1000 个免费积分**，而且即使任务失败也会消耗积分。
   - 用户喜欢官方积极提供帮助，但*从基于任务到基于积分的转变确实让人感到不适，尤其是这并不在最初的 Beta 测试流程中*。
- **为 Manus GPU 算力集群构思替代能源**：一位成员建议 Manus 可以组建一个团队，致力于开发廉价的可再生能源（如熔融钠、热能或太阳能）为自有的 GPU 算力集群供电，以降低成本。
   - 他们提到可以将其设在沙漠中，并使用飞轮作为储能装置，以便在夜间保持运行并实现最高效率。
- **考虑使用 Deepseek 等替代 AI 模型**：讨论了使用像 **Deepseek** 和 **Qwen** 这样更便宜的 AI 模型来代替 **Anthropic** 的 **Claude**，以降低运营成本。
   - 然而，目前尚未说明 **Manus** 是否会允许集成其他 AI。
- **Manus AI 辅助考试**：一些用户将 **Manus** 与 Kimi 或 Deepseek 结合使用，上传研讨会和讲座文件，要求 AI 记忆这些内容以备考，这帮助一些人在作业中获得了 **81/100** 的分数。
   - 一些人担心如果帮助 AI 在学校考试中作弊，是否会违反服务条款。
- **喜爱 Manus 的 UI 设计**：多位成员称赞了 Manus 的 UI 设计，表示设计非常出色、易于使用、简洁且符合现实世界的概念。
   - 一位用户表示：*“Manus 让人感觉如此惊艳的原因不仅在于你得到的结果，还在于任务的概念与现实世界的概念紧密结合。这种简洁性简直是天才之作。”*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/a/R5vY585">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行迷因、娱乐 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://en.wikipedia.org">维基百科，自由的百科全书</a>: 未找到描述</li><li><a href="https://apps.apple.com/us/app/munas/id6742685315">‎Munas</a>: ‎1. 产品介绍：此应用集成了多种先进的 AI 技术，提供多模态对话、深度推理、智能创作和图像生成等功能。无论是在工作中...</li><li><a href="https://tenor.com/view/mad-annoyed-gif-27497951">愤怒烦恼 GIF - Mad Annoyed - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/evil-cat-floppy-herobrine-angry-cat-glowing-eyes-gif-13772161273485327421">邪恶猫 Floppy GIF - Evil cat Floppy Herobrine - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.investopedia.com/terms/m/mentalaccounting.asp">心理账户：定义、避免偏见及示例</a>: 心理账户是指一个人根据主观标准对相同金额的钱赋予不同的价值，通常会产生不利的结果。</li><li><a href="https://tenor.com/view/thats-my-opinion-have-yours-shrug-sassy-pretty-gif-16292035">那是我的观点，你有你的 GIF - Thats My Opinion Have Yours Shrug - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://kvlcogit.manus.space/">供应链可持续性情报平台</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Item_response_theory">项目反应理论 - 维基百科</a>: 未找到描述</li><li><a href="https://www.aeaweb.org/articles?id=10.1257/jel.20201593">加密货币的微观经济学 - 美国经济学会</a>: 未找到描述</li><li><a href="https://github.com/browser-use/browser-use">GitHub - browser-use/browser-use: 让网站对 AI Agent 可访问</a>: 让网站对 AI Agent 可访问。通过在 GitHub 上创建账号为 browser-use/browser-use 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1354894027660722350)** (1219 条消息🔥🔥🔥): 

> `Perplexity AI 停机、DeepSeek AI、Claude AI、用户沮丧情绪、T-Mobile 促销`

- **Perplexity AI 服务器崩溃**：多名用户报告了**停机以及历史记录/空间（spaces）消失**的问题（[示例](https://discord.com/channels/1047197230748151888/1164597981840941076/1355201859186462781)），引发了社区内的调侃和沮丧情绪。
   - 官方状态页面（[status.perplexity.com](https://status.perplexity.com/)）缺乏及时更新，用户建议建立自动化的用户报告停机系统和主动通知机制。
- **DeepSeek AI 并不那么“深”**：成员们普遍对 **DeepSeek AI** 及其理解复杂指令的能力感到**失望**。
   - 与更好的数学应用相比，他们还评论说它会给出*不必要的专业术语*。
- **Claude AI**：一位成员询问了 **Claude AI 的 Context Window 限制**，并将其与 Gemini 和 ChatGPT 进行了对比。
   - 许多成员评论说 Claude 的 Context Window 非常有限，尤其是与 Gemini 相比。
- **用户发泄不满**：用户哀叹因停机而**丢失了笔记和学习资料**，有些人开玩笑地指责 AI 毁了他们的考试。
   - 讨论了 Perplexity **每月 20 美元订阅费**的更广泛经济影响，特别是考虑到不同国家的最低工资工人。
- **T-Mobile 用户获利**：用户分享了通过 **T-Mobile** 和 Revolut 促销活动获取**免费 Perplexity Pro 订阅**的方法。
   - 一位成员建议在 T-Mobile 上使用临时号码（burner number）来获得访问权限。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/testingcatalog/status/1905390832225493429?s=61">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重大新闻 🚨：@perplexity_ai 正在发布语音听写功能以及一系列其他更新。语音听写使用的是 OpenAI，但目前似乎还无法正常工作。</li><li><a href="https://x.com/AravSrinivas/status/1905652310501675318">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们本周正面临一些 infra 挑战，这就是为什么您过去的 threads 库或发现内容现在可能无法加载的原因。我向大家表示歉意。我们正...</li><li><a href="https://x.com/AravSrinivas/status/1905652310501675318?t=0AABseZKpQZ57cG3Aw94wQ&s=19">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们本周正面临一些 infra 挑战，这就是为什么您过去的 threads 库或发现内容现在可能无法加载的原因。我向大家表示歉意。我们正...</li><li><a href="https://www.theatlantic.com/national/archive/2013/06/what-does-american-actually-mean/276999/">“American” 到底是什么意思？</a>：在拉丁美洲，“American” 指的是来自美洲大陆的任何人。美国公民占用这个词被认为是失礼或帝国主义的表现。那么解决方案是什么？</li><li><a href="https://tenor.com/view/president-stops-president-smiles-ominous-smile-ominous-no-answer-gif-78197234359887585">总统停止 总统微笑 GIF - 总统停止 总统微笑 不祥的微笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态</li><li><a href="https://tenor.com/view/megan-thee-stallion-shock-snl-excuse-me-rude-gif-12376496298828106239">Megan Thee Stallion 震惊 GIF - Megan Thee Stallion 震惊 Snl - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/april-fools-joke-dog-its-fine-this-is-not-gif-1750056094467610487">愚人节玩笑 GIF - 愚人节玩笑小狗 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/putin-stare-gif-14318699512326580302">普京凝视 GIF - 普京凝视 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/">YouTube</a>：未找到描述</li><li><a href="https://tenor.com/view/spongebob-spongebob-meme-thinking-sad-coffee-gif-21807366">海绵宝宝海绵宝宝表情包 GIF - 海绵宝宝海绵宝宝表情包思考 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/comment/mk97f47/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://tenor.com/view/log-crane-tricks-gif-17344198">原木起重机 GIF - 原木起重机技巧 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://www.livescience.com/space/black-holes/is-our-universe-trapped-inside-a-black-hole-this-james-webb-space-telescope-discovery-might-blow-your-mind">我们的宇宙是否被困在黑洞中？詹姆斯·韦伯太空望远镜（James Webb Space Telescope）的这一发现可能会让你大吃一惊</a>：未找到描述</li><li><a href="https://youtu.be/4UKM_yvTexI?si=IChRCpK43in_IznM"> - YouTube</a>：未找到描述</li><li><a href="https://payequity.gov.on.ca/docs/7-12proportional-value-comparison-method/">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=e1C75U_7e6U"> - YouTube</a>：未找到描述</li><li><a href="https://math.libretexts.org/Bookshelves/PreAlgebra/Pre-Algebra_II_(Illustrative_Mathematics_-_Grade_8)/03:_Linear_Relationships/3.00:_New_Page/3.1.4:_Comparing_Proportional_Relationships">3.1.4：比较比例关系</a>：未找到描述</li><li><a href="https://math.stackexchange.com/questions/67280/salary-calculation-solving-proportional-increase-of-two-variables">薪资计算：解决两个变量的比例增长</a>：我想按比例了解，一个人的新薪水中有多少分别归功于工作小时数的增加和美元工资的提高。两者都...</li><li><a href="https://byjus.com/maths/ratios-and-proportion/">比率与比例 - 定义、公式和示例</a>：比率和比例是用于比较数量的数学表达式。访问 BYJU’S 学习比率和比例的定义、公式和示例。</li><li><a href="https://www.investopedia.com/ask/answers/042415/what-are-differences-between-regressive-proportional-and-progressive-taxes.asp">累退税 vs. 比例税 vs. 累进税：有什么区别？</a>：美国使用三种类型的税收制度：累退税、比例税和累进税。其中两种对高收入和低收入人群的影响...</li>

对不同收入者的处理方式不同，而其中一项对所有人都是相同的。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1354898417910022258)** (10 条消息🔥): 

> `可共享线程, Super Prompt, LLM Research` 


- **需要 **Shareable Threads****：Perplexity AI 要求成员确保其线程是 *`Shareable`* 的，并链接到了 Discord 中关于此话题的先前消息，点击[此处](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)查看。
- ****Super Prompt** 即将到来**：一位成员分享了一个 Perplexity AI 搜索结果链接，内容为 *`create a super prompt for copi`*，点击[此处](https://www.perplexity.ai/search/create-a-super-prompt-for-copi-xD7hRu9bQKeeWTuv9l6lfg)查看。
- **分享了 **LLM Research** 集合**：一位成员分享了一个关于 *`LLM research`* 的 Perplexity AI 集合链接，点击[此处](https://www.perplexity.ai/collections/llm-research-TIWFlUA7SWGuIqfbmRzenQ)查看。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1354910680624664606)** (7 条消息): 

> `API 参数错误处理, 使用 Perplexity Sonar 的 Llama Index RAG 上下文, API 与 Web 端 Deep Research 的对等性` 


- **API 参数现在会抛出错误**：API 团队针对非 Tier 3 用户实现了对 **search_domain_filter**、**related_questions**、**images** 和 **structured_outputs** 等参数的错误处理。
   - 如果你之前通过在 prompt 中传递 JSON schema（而不是使用参数）实现了预期结果，你将继续看到正确的行为；*本质上没有任何改变*。
- **Sonar 模型在处理 Llama Index RAG 上下文时遇到困难**：一位用户询问如何使用 index 对象将 **Llama Index RAG** 上下文传递给 **Perplexity Sonar** 模型。
   - 该用户还询问 API 中的 **Deep Research** 是否会更接近 perplexity.com 上的版本，因为与网站相比，它似乎被 *削弱 (nerfed)* 了。
- **Sonar API 缺失引用**：一位用户报告了一个使用 sonar 模型的 API 未返回任何引用的实例。
   - 该用户指出，客户端上的相同查询是带有引用的。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1354893464340267089)** (1251 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro 定价, Cursor 基础设施, 人形机器人?, Cursor 中移除 Codebase 标签` 


- **Dan 透露了关于 Cursor 中 DeepSeek 3.1 的深度细节**：一位 Cursor 团队成员分享称，**DeepSeek 3.1** 应该会在 12 小时内出现在编辑器中，但未透露成本。
   - 此外还有与 [供应商的交易](https://cursor.com/deals)，且隐私模式确保不存储任何数据，这很不错。
- **Cursor 因数据库部署故障而崩溃**：由于其基础设施内的 **数据库部署问题**，Cursor 经历了全服务中断，影响了 Chat 和 Tab 等 AI 功能。
   - 该事件在几小时后得到解决，一位团队成员开玩笑说他们 *不小心拔掉了主服务器的插头来给手机充电*。
- **关于人形机器人的热议**：成员们辩论了人形机器人的效用，一些人将其设想为 *做饭和清洁* 助手，而另一些人则对 [数据隐私](https://en.wikipedia.org/wiki/Data_privacy) 和遥测表示担忧。
   - 另一位成员建议机器人技术将是 **AGI 出现的地方**，先在虚拟环境中进化，然后进入现实环境。
- **成员们哀悼 Cursor 中缺失的 @codebase 标签**：用户注意到 [@Codebase 标签已被移除](https://cursor.com/changelog)，工作人员解释说它被一种类似的扫描当前索引项目的方式所取代。
   - 这引发了关于 **token 限制**、定价模型以及在使用 AI 编程工具时便利性与控制权之间权衡的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/settings/models#large-context-and-pricing">Cursor – Models</a>: 未找到描述</li><li><a href="https://x.com/nicdunz/status/1905353949865238633?s=46">来自 nic (@nicdunz) 的推文</a>: 4o 在编程领域夺得第一</li><li><a href="https://x.com/artificialanlys/status/1905563427776651344?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: 今天的 GPT-4o 更新确实意义重大——它在我们的 Intelligence Index 中超越了 Claude 3.7 Sonnet (non-reasoning) 和 Gemini 2.0 Flash，现在是领先的编程类 non-reasoning 模型。这使得 GP...</li><li><a href="https://docs.cursor.com/troubleshooting/request-reporting">Cursor – 获取 Request ID</a>: 未找到描述</li><li><a href="https://tenor.com/view/correct-futurama-the-best-kind-of-correct-yes-yep-gif-5787390">Correct Futurama GIF - Correct Futurama The Best Kind Of Correct - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/readingdancer/status/1829267522777919904">来自 Chris Houston (@readingdancer) 的推文</a>: 我想知道为什么 @OpenAI 的 #chatgpt4 不喜欢这个提示词？“你能创作一张戴着牛仔帽的金鱼骑在小猪身上的图片吗？”</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: 今天的 GPT-4o 更新确实意义重大——它在我们的 Intelligence Index 中超越了 Claude 3.7 Sonnet (non-reasoning) 和 Gemini 2.0 Flash，现在是领先的编程类 non-reasoning 模型。这使得 GP...</li><li><a href="https://tenor.com/view/drake-hotline-bling-dance-dancing-gif-17654506">Drake Hotline Bling GIF - Drake Hotline Bling Dance - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/when-the-coding-when-the-coding-when-the-coding-is-when-the-meme-gif-21749595">When The Coding Coding GIF - When The Coding When The Coding - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: `chatgpt-4o-latest` 现已在 API 中更新，但请保持关注——我们计划在未来几周内将这些改进引入 API 中的特定日期版本模型。引用 OpenAI (@OpenAI) GPT-4o 获得了另一次更新...</li><li><a href="https://x.com/threejs/status/1905647468551053370">来自 Three.js (@threejs) 的推文</a>: Three.js r175 发布 🗿https://threejs.org/changelog/?r175</li><li><a href="https://status.cursor.com/">Cursor Status</a>: 未找到描述</li><li><a href="https://docs.cursor.com/settings/models#context-window-sizes">Cursor – Models</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/connection-failed-if-the-problem-persists-please-check-your-internet-connection-or-vpn-or-email-us-at-hi-cursor-sh/17334/51">连接失败。如果问题仍然存在，请检查您的互联网连接或 VPN，或发送电子邮件至 hi@cursor.sh</a>: 连接失败。如果问题仍然存在，请检查您的互联网连接或 VPN (Request ID: 3004f856-6920-443f-</li><li><a href="https://en.wikipedia.org/wiki/Artificial_general_intelligence">Artificial general intelligence - 维基百科</a>: 未找到描述</li><li><a href="https://generalanalysis.com/blog/jailbreak_cookbook">The Jailbreak Cookbook - General Analysis</a>: 未找到描述</li><li><a href="https://cimwashere.com">Cim Was Here</a>: 摄影。加利福尼亚州斯托克顿。</li><li><a href="https://codeium.com/windsurf">Codeium 推出的 Windsurf Editor</a>: 未来的编辑器，就在今天。Windsurf Editor 是首款由 AI agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://github.com/end-4/dots-hyprland">GitHub - end-4/dots-hyprland: 我讨厌极简主义，所以...</a>: 我讨厌极简主义，所以... 通过在 GitHub 上创建账户来为 end-4/dots-hyprland 的开发做出贡献。</li><li><a href="https://github.com/danperks/CursorStatus">GitHub - danperks/CursorStatus</a>: 通过在 GitHub 上创建账户来为 danperks/CursorStatus 的开发做出贡献。</li><li><a href="https://codeium.com/profile/docker">Cim (@docker) 个人资料 | Codeium</a>: Cim (@docker) 已使用 Codeium 的 AI 自动补全完成了 4,096 次。Codeium 提供一流的 AI 代码补全和聊天功能——全部免费。
</li>
</ul>

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1354893069488492754)** (906 条消息🔥🔥🔥): 

> `O1 Pro 发布，GPT-4o 最新基准测试，DeepSeek V3，Meta Llama，AI Safety` 


- **O1 Pro 即将登上排行榜？**：成员们讨论了将 **O1 Pro** 纳入排行榜的可能性，推测 **OpenAI** 可能会承担成本以展示其能力，特别是考虑到其高昂的价格。
   - 然而，一些人对其获得高排名的能力表示怀疑，一位评论者开玩笑说，由于延迟问题，它*会排在最后一名 lol*。
- **GPT-4o 的编程能力**：成员们在最近的更新后辩论了 [GPT-4o 的编程能力](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19)，一些人注意到它在指令遵循和代码生成方面有所改进。
   - 但也有人坚持认为需要进行适当的评估，正如[一位成员所言](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/)，**GPT-4o** 的排名可能因为针对特定偏好响应风格的专门训练而虚高，而非实际性能的提升。
- **DeepSeek V3 在编程基准测试中的崛起**：新的 **DeepSeek V3 0324** 模型正获得认可，根据[这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/)，一位成员指出它在 **SWE-bench** 上的得分高于 **GPT-4o R1**。
   - 数据显示，DeepSeek 的 **V3 0324** 版本在非推理领域超越了 **Claude 3.7 Sonnet**，并已成为领先的非推理编程模型。
- **Meta 充满表情符号的 Llama 模型**：成员们观察到竞技场中最近出现的匿名模型（据信来自 **Meta**）表现出古怪的行为，包括添加大量表情符号，并倾向于自称为 **Meta Llama** 模型，尽管它们的图像识别能力明显较差。
   - 正在测试的模型包括：`bolide`、`cybele`、`ginger`、`nutmeg`、`phoebe`、`spider`、`themis`；不过他们也注意到 `spider` 有时会被识别为 GPT-4。
- **关于 AI Safety 和越狱的讨论**：成员们讨论了 AI Safety，提到像 **Claude** 这样的模型是基于 Constitutional AI 原则设计的，优先考虑客观性而非用户偏好，这可能会影响它们的排行榜排名。
   - 一位成员还分享了 [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) 资源，用于 LLM 越狱和 AI Safety 研究，包括一个包含系统性越狱实现的 [GitHub 仓库](https://github.com/General-Analysis/GA)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/flavioad/status/1905347584438251848?s=46">Flavio Adamo (@flavioAd) 的推文</a>: OpenAI 刚刚更新了 GPT-4o，我测试了旧版与新版，差异实际上非常大。引用 OpenAI (@OpenAI)：GPT-4o 在 ChatGPT 中又获得了一次更新！有什么不同？- 更好地遵循详细指令...</li><li><a href="https://x.com/patloeber/status/1905333725698666913">Patrick Loeber (@patloeber) 的推文</a>: 🏆Gemini 2.5 Pro 目前在 LMArena 排名第一，在 Livebench 排名第一，在所有 SEAL 排行榜中排名第一。它也开始成为编程任务的首选 :) 我们的团队正在努力让每个人都获得更高的排名...</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19">Artificial Analysis (@ArtificialAnlys) 的推文</a>: 今天的 GPT-4o 更新实际上非常重大 - 它在我们的智能指数中超越了 Claude 3.7 Sonnet (非推理) 和 Gemini 2.0 Flash，现在是领先的非推理编程模型。这使得 GPT...</li><li><a href="https://tenor.com/view/tin-foil-hat-jabrils-foil-hat-put-on-a-hat-wearing-a-hat-gif-18223681">Tin Foil Hat Jabrils GIF - 锡纸帽 Jabrils 锡纸帽 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Conversion_to_Judaism">皈依犹太教 - 维基百科</a>: 未找到描述</li><li><a href="https://generalanalysis.com/blog/jailbreak_cookbook">The Jailbreak Cookbook - General Analysis</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1354893051541328013)** (580 条消息🔥🔥🔥): 

> `Elevenlabs Scribe V1 用于音频事件分类，OlmOCR 在 Unsloth 中加载，针对棋盘游戏微调 LLM，Gemma 3 notebook 奇点，Qwen Omni Hacking`

- **Scribe V1 驱动 FoxMoans 话语列表**：一名成员正在使用 **11Labs Scribe V1** 进行音频事件分类，以创建一个 [话语列表](https://github.com/zero2rizz/FoxMoans/blob/main/UtteranceList.txt)，他们估计这将耗资约 **2 万美元**。
   - 他们提到将其用于音频事件分类，表明它非常适合需要基于情绪分析的项目，因为它能够检测 *笑声、愤怒的病娇（yandere）风格* 等变化。
- **OlmOCR 的 Unsloth 集成仍不稳定**：尽管 **Qwen2VL** 可以正常工作，但一名成员仍难以在 Unsloth 中加载 **OlmOCR**（**Qwen2VL** 的一个微调版本）。
   - Unsloth 团队跟进询问用户是否尝试了最新版本，一名成员指出，他们一直在 *创作者意识到其模型完成上传之前* 就推送更新和修复。
- **Orpheus TTS 发布微调 Notebook，并探索多语言支持**：Unsloth 团队发布了一个 [用于微调 **Orpheus-TTS** 的 Notebook](https://x.com/UnslothAI/status/1905312969879421435)，强调其具有情感线索（叹气、笑声）的类人语音表现优于 OpenAI。
   - 成员们还讨论了将 **Orpheus** 的语言从英语更改为其他语言的可能性，并建议使用新的嵌入层/头部层（embedded/head layers）进行持续预训练可能就足够了。
- **LocalLlama 版主因删除帖子而面临指责**：一名用户抱怨其解释 **Llama Community License** 的帖子被 r/LocalLLama 删除，导致人们猜测 Meta/Facebook 可能正在管理该子版块。
   - 其他成员对此并不在意，指出它是专有的，且 Meta 并没有真正强制执行，一名成员表示 *只有当我看到停止并终止函（c&d）或实际的强制执行时我才会在意*。
- **Multi-GPU 支持即将推出，无需 Pro 版本**：一名用户询问如何获取 Unsloth Pro 以获得 Multi-GPU 支持，但一名成员回复称，**Multi-GPU** 支持可能会在未来几周内根据 AGPL 协议变为 *免费*。
   - 一名用户表示 *至少对于第一个版本，初步工作已经完成，还需要几周时间*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#saving-to-safetensors-not-bin-format-in-colab">故障排除 | Unsloth 文档</a>：如果您在运行或保存模型时遇到问题。</li><li><a href="https://x.com/UnslothAI/status/1905312969879421435">来自 Unsloth AI (@UnslothAI) 的推文</a>：使用我们的 Notebook 免费微调 Orpheus-TTS！Orpheus 提供具有情感线索（叹气、笑声）的类人语音，表现优于 OpenAI。使用减少 70% 的 VRAM，以 2 倍的速度自定义语音 + 对话...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">教程：如何在本地运行 DeepSeek-V3-0324 | Unsloth 文档</a>：如何使用我们的动态量化在本地运行 DeepSeek-V3-0324，从而恢复精度。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=MKX_XKs_BNZR">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Open_source">开源 - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/docs/trl/v0.4.2/en/sft_trainer#packing-dataset-constantlengthdataset">监督微调训练器 (SFT Trainer)</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个检查点继续微调 | Unsloth 文档</a>：检查点（Checkpointing）允许您保存微调进度，以便您可以暂停并继续。</li><li><a href="https://notes.victor.earth/youre-probably-breaking-the-llama-community-license/">你可能违反了 Llama 社区许可证</a>：你可能违反了 Llama 社区许可证</li><li><a href="https://github.com/unslothai/unsloth/issues/2086)">unslothai/unsloth</a>：微调 Llama 3.3, DeepSeek-R1, Gemma 3 &amp; 推理 LLMs，速度提升 2 倍，显存减少 70%！ 🦥 - unslothai/unsloth</li><li><a href="https://github.com/y-haidar/awbw-research">GitHub - y-haidar/awbw-research: 这是一个已停止的项目，包含了尝试为游戏 awbw 创建 AI 的努力</a>：这是一个已停止的项目，包含了尝试为游戏 awbw 创建 AI 的努力 - y-haidar/awbw-research</li><li><a href="https://github.com/canopyai/Orpheus-TTS/issues/37">预训练数据结构 · Issue #37 · canopyai/Orpheus-TTS</a>：感谢分享这项伟大的工作，我想了解预训练数据的格式及其在给定配置文件中的含义 > ` > # Datasets > text_QA_dataset: <speech input-ids> > TTS_dataset: <...
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1355173901855883415)** (1 条消息): 

> `` 


- **未讨论任何主题**：频道中未发现讨论主题。
   - 唯一的内容是一个图片附件。
- **存在图片附件**：频道中附加了一张带有 Discord CDN 链接的图片。
   - 该图片未被进一步讨论或分析。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1354896976176615606)** (68 条消息🔥🔥): 

> `Training Loss Interpretation, Gemma & Task Difficulty, Dataset Size & Overfitting, LM Studio Models, HF Upload & vLLM` 


- ****训练损失动态让学习者困惑****：一位用户询问关于训练损失在早期下降并保持在接近零的水平，质疑突然的增加是否为坏兆头，并展示了一张[图表](https://cdn.discordapp.com/attachments/1179777624986357780/1354896975967027360/image.png?ex=67e84723&is=67e6f5a3&hm=3f0e1f53cd32481a10565cb630aa71589b55f50ac1ed66d398bd327285cb40e5)。
   - 另一位成员建议该任务对于 **Gemma** 来说可能太简单了，导致其停止学习，同时建议用户使用 **Weights & Biases (W&B)** 以获得更好的图表可视化效果。
- ****BOS Token 双重问题引发混乱****：一位用户报告在检查 tokenizer 解码时，发现最新的 **Unsloth 更新 (Gemma 3 4B)** 中存在 **double BOS token** 问题。
   - 已[确定](https://github.com/unslothai/unsloth-zoo/pull/106)了一个热修复补丁，删除了意外添加的 token。
- ****Unsloth 安装更新引发意外的性能问题****：用户报告在 Unsloth 更新期间使用 `--no-deps` 标志时遇到了**严重问题**，这与现有的某些指令相反。
   - 一位用户强烈建议更新所有依赖项，并指出文档已过时，特别指向了 [Unsloth 文档](https://docs.unsloth.ai/get-started/installing-+-updating/updating)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">教程：使用 GRPO 训练你自己的推理模型 | Unsloth Documentation</a>：将 Llama 3.1 (8B) 等模型通过使用 Unsloth 和 GRPO 转换为推理模型的初学者指南。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">更新 | Unsloth Documentation</a>：要更新或使用旧版本的 Unsloth，请遵循以下步骤：</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/WecoAI/aideml">GitHub - WecoAI/aideml: AIDE: AI-Driven Exploration in the Space of Code. State of the Art machine Learning engineering agents that automates AI R&amp;D.</a>：AIDE：代码空间中的 AI 驱动探索。自动化 AI 研发的先进机器学习工程 Agent。- WecoAI/aideml</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/106">fix double bos by tamewild · Pull Request #106 · unslothai/unsloth-zoo</a>：此处被错误添加 4a66f8b</li><li><a href="https://www.reddit.com/r/unsloth/comments/1jldwql/comment/mk5yz0j/?%24deep_link=true&correlation_id=4bc85317-7b48-594b-9199-248dd1496be7&ref=email_post_reply&ref_campaign=email_post_reply&ref_source=email&%243p=e_as&_branch_match_id=1423385709101741523&utm_medium=Email+Amazon+SES&_branch_referrer=H4sIAAAAAAAAA3VO22qEMBD9GvdNrSauuiCltPQ3QjSzmt3cOomE7UO%2FvSNtHwszcDiXObOlFOKlrhGU0qmSIVRGu3vNwnPRchYmEDKeCHrUq3bSiB3NtB2pgr0U7TtNzrn6zS%2FeEoG0u4vGp40QcRZcigSbm1H5wxC6agdpdyCyTpuwOiak29FKYwRryGDv3ePz6XY0MCrhCiCI47WCvSXcoWjPi0cEI5P2TmhFPJ%2BXoWNNX%2FYzH8pu5HM5NuNYtnxQquHjeYaecghXMoOV2ojgYxIIwTx%2BBLFIG6Re3f%2BO6Hdc4E8%2FfREHiNqtYkafI%2BD0uqG38A33VShCWwEAAA%3D%3D">Reddit - 互联网的心脏</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1355245690556780747)** (2 messages): 

> `Orpheus-TTS, Voice Model Finetuning, UnslothAI` 


- **Unsloth 微调 Orpheus-TTS！**：Unsloth 发布了一个针对语音模型 **Orpheus-TTS** 的 [微调 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)，可在 Colab 上免费使用。
   - 通过 Unsloth，它可以实现定制化语音和对话，速度提升 **2 倍**，且 **VRAM 占用减少 70%**。
- **Orpheus-TTS 展示情感！**：**Orpheus** 提供具有情感线索（叹气、笑声）的类人语音，表现优于 **OpenAI**。
   - Unsloth 展示了一个在 **1000** 行数据集上仅进行 **100 steps** 微调的示例，并成功彻底改变了模型的声音和个性！


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1905312969879421435">来自 Unsloth AI (@UnslothAI) 的推文</a>: 使用我们的 notebook 免费微调 Orpheus-TTS！Orpheus 提供具有情感线索（叹气、笑声）的类人语音，表现优于 OpenAI。通过减少 70% 的 VRAM，以 2 倍的速度定制语音和对话...</li><li><a href="https://x.com/danielhanchen/status/1905315906051604595">来自 Daniel Han (@danielhanchen) 的推文</a>: 我们在一个微小的数据集上训练了语音 LLM Orpheus-TTS，并成功彻底改变了模型的声音和个性！这非常酷，尤其是模型具有像咯咯笑或叹气这样的情感线索...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1354901327292928281)** (9 messages🔥): 

> `Dynamic Quantization, DeepSeek-R1, ACDiT` 


- **Unsloth 动态量化深度解析**：一位成员询问了 Unsloth 中**动态量化（dynamic quantization）**的“动态”方面，根据 [super weights 论文](https://arxiv.org/abs/2402.10433)，询问是否识别出导致更多激活/量化误差的权重并不对其进行量化，而将其余部分量化为 4-bit。
   - 问题延伸到 kblams 的动态程度，以及程序本身是否可以编码到知识库中，换句话说，量化误差是如何计算的，哪些层更重要，并寻找代码库或正则化。
- **DeepSeek-R1 的 GGUF 和 4-bit 格式已发布！**：Unsloth 发布了 **DeepSeek-R1** 的多个版本，包括 [GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf) 和 [4-bit 格式](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5)。
   - Unsloth 的 **DeepSeek-R1** 1.58-bit + 2-bit 动态量化是选择性量化的，比标准的 1-bit/2-bit 提高了准确度。
- **ACDiT 论文发布**：分享了 AutoConditional Diffusion Transformer (**ACDiT**) 论文：[https://arxiv.org/abs/2412.07720](https://arxiv.org/abs/2412.07720)。
   - 该论文描述了用于建模连续视觉信息的自回归和扩散范式的结合，在训练期间对标准 Diffusion Transformer 使用 Skip-Causal Attention Mask (SCAM)，并在推理期间使用 KV-Cache 进行自回归解码。
- **关于 GRPO notebook 中推理过程的提问**：一位成员询问了在 **GRPO** notebook 中处理推理过程的方法，特别是如何在模型给出最终答案之前分离并修改推理内容。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.07720">ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer</a>: 我们提出了 ACDiT，一种新型的自回归块条件 Diffusion Transformer，它创新地结合了自回归和扩散范式来建模连续视觉信息。通过 i...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1354894500488548483)** (305 条消息🔥🔥): 

> `Gemini 2.5 Pro vs GPT-4o, Google AI Studio, 用于新闻和时事的 Perplexity, Claude vs GPT 推理能力对比, AI 转录工具` 


- **GPT-4o 在编程方面略胜 Gemini 2.5 Pro**：成员们讨论了 **GPT-4o** 与 **Gemini 2.5 Pro** 的编程能力，一些人认为 *GPT-4o* 在编程任务上更出色，挑战了最初的印象，而另一些人则整体更倾向于 *Gemini*，或在 C++ 和 WinAPI 集成等特定任务中青睐它。
   - 有人提到 [GPT-4o](link.to.g40) 在编程领域领先，但许多第三方基准测试显示 Gemini 2.5 Pro 在编程方面的整体表现更好，除了 **GPT-4o** 在 6 个类别中胜出了 3 个。
- **免费的 Google AI Studio 大放异彩**：用户讨论了使用 **Google AI Studio** 的好处，指出它可以免费访问 **Gemini 2.5 Pro** 等模型，并强调了其与 **ChatGPT Plus** 等付费服务相比慷慨的 Prompt 限制。
   - 成员们报告称每天使用数百条消息而未达到限制，并分享了一个方便的对比资源，导致一人因 *AI Studio* 的优势而取消了其 *ChatGPT* 订阅。
- **Perplexity 在新闻领域胜过 ChatGPT**：成员们发现 **Perplexity** 在获取新闻和时事方面优于 **ChatGPT**，强调了其用于获取最新新闻的 Discover 标签，并指出 **Perplexity** 不仅仅是一个 *GPT wrapper*。
   - 然而，一位用户发现 **Perplexity** 的 *Deep Research* 功能在质量和可靠性方面存在问题，特别是在对上传文件进行研究时，因此建议使用 *ChatGPT*。
- **Claude 3.7 Sonnet 在推理方面称霸**：成员们称赞 **Claude 3.7 Sonnet** 的推理能力，指出其解释能力优于其他 AI 模型，特别是由于 *免费版 Claude 额度用完后会强制你开启新对话*。
   - 有人建议 o1、o3-mini-high 和 Grok 3 等模型在编程方面都很出色，尽管一位成员发现 o1 在处理涉及 C++、物理、渲染和 Win32API 等旧 API 的复杂编程时表现最好。
- **寻找最佳免费 AI 转录工具**：成员们寻求关于免费 **AI 转录工具** 的建议，一人建议在本地运行 **Whisper**，另一人指出安装必要的 Python 包非常困难，且解决依赖问题的复杂度很高。
   - 为了解决这些问题，用户应该尝试使用云端解决方案，这些方案通常可以立即运行，无需安装和调试本地包。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1354969698974040086)** (8 条消息🔥): 

> `图像生成器, GPT-4.5 错误, 用于摘要的 GPT 模型, AI 语音聊天机器人` 


- **需要图像生成器**：一位成员正在寻找一种 **图像生成器**，允许从剧本中复制并粘贴场景以转换为卡通风格，且不需要频繁注册。
   - 他们还抱怨新的图像生成功能已损坏，无法生成图像。
- **GPT-4.5 错误常见吗？**：一位成员报告在消息流中遇到 **GPT-4.5 错误**，消息会随机中断，导致无法继续这些对话。
   - 错误从昨天开始出现。
- **用于文本摘要的最佳 GPT 模型**：一位成员询问哪个 **GPT 模型** 最适合总结和分析数万字的文本，并提到了他们使用 *o3 mini*、*mini high* 和 *o1* 的经验。
   - 另一位成员建议使用 **GPT-4.5** 和 *o1*，同时建议不要在此类特定任务中使用 *4o*。
- **STT 集成故障排除**：一位成员正在开发一个集成语音转文本 (STT)、语言模型 (LLM) 和文本转语音 (TTS) 的 **AI 语音聊天机器人**，但面临 OpenAI 版本的兼容性问题。
   - Chat completion 功能仅适用于 1.0 之前的 OpenAI 版本，而他们目前使用的是 1.66 版本，且 *store:true* 命令未按预期执行。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1354999301813702757)** (83 条消息🔥🔥): 

> `Yu-Gi-Oh! card art prompting, Microsoft PromptWizard, ChatGPT prompting methods, Hierarchical communication with markdown, AI prompt engineering` 


- **Yu-Gi-Oh! 卡牌艺术风格提示词**：一位成员寻求关于改进提示词的建议，以生成 **Yu-Gi-Oh!** 集换式卡牌风格的艺术作品，并指出 **ChatGPT** 往往默认生成漫画艺术。
   - 该用户已经尝试使用诸如 *"Render this character in the style of a Yu-Gi-Oh! trading card illustration. Use sharp, clean digital art..."* 之类的提示词，并取得了阶段性的进展。
- **Microsoft PromptWizard 使用情况**：一位成员询问了使用 **Microsoft PromptWizard** 处理自定义数据的经验，寻求社区的见解。
   - 给出的文本中未提供任何回复。
- **解锁 ChatGPT 的最佳潜力**：一位成员询问了关于 *secret prompts* 或最大化 **ChatGPT** 潜力的方法，觉得还有更多可以挖掘的空间。
   - 建议包括在提供提示词之前使用 *prompt conditions and disclaimers*（提示条件和免责声明）。
- **Darth 的提示词入门教程**：一位成员分享了一种教授有效提示技术的结构化方法，包括使用 Markdown 的分层通信、通过开放变量进行的抽象以及强化策略。
   - 该方法包含一个 [可共享的 ChatGPT 链接](https://chatgpt.com/share/67e6dca1-46d0-8000-8ee5-5fc8e61f06d6)，并强调通过 **ML format matching** 来获得更好的遵循效果。
- **ChatGPT 增强的图像提示功能**：一位成员对新的 **ChatGPT** 图像工具表示赞赏，指出其对提示词要求的遵循度有所提高，并能处理复杂的场景，例如在巨龟背上生成一个移动的市场，且带有太阳和三个月亮。
   - 用户发现新工具在不影响图像整体的情况下进行修改表现得更好，例如在最初包含星星的夜景中移除星星。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1354999301813702757)** (83 条消息🔥🔥): 

> `Yu-Gi-Oh! card art prompting, Microsoft PromptWizard, ChatGPT prompting tips, Hierarchical communication with markdown, GPTs in conversation` 


- **Yu-Gi-Oh! 艺术风格提示词微调**：一位用户寻求改进生成 **Yu-Gi-Oh!** 卡牌艺术提示词的建议，提到在 *Ghibli* 和 *photorealism* 风格上取得了成功，但在实现理想的 **Yu-Gi-Oh!** 美学方面遇到了困难。
   - 他们提供了示例，目前的提示词侧重于 *sharp, clean digital art with stylized anime rendering, glowing magical effects, and a dynamic pose*。
- **PromptWizard 用户集结**：一位成员询问关于在自定义数据应用中使用 **Microsoft PromptWizard** 的经验。
   - 其他人正在寻求 *secret prompts* 或最大化 **ChatGPT** 潜力的方法。
- **提示策略：条件与免责声明**：一位成员建议在主提示词之前添加 *prompt condition and disclaimer* 以引导 **ChatGPT** 的输出。
   - 一个 [预设会话 (primed session)](https://chatgpt.com/share/67e6dca1-46d0-8000-8ee5-5fc8e61f06d6) 的链接展示了 *使用 Markdown 的分层通信、通过开放变量进行的抽象、强化以及用于合规性的 ML format matching*。
- **GPTs 加入对话力量**：一位用户高兴地发现，可以通过输入 `@` 在 **ChatGPT** 对话中调用自定义 **GPTs**。
   - 另一位用户指出，他们喜欢 *能够指示工具的使用*，并且 *新的 imagegen 在处理我给出的奇怪需求时表现强劲*。
- **AI 提示词中对 Markdown 的误解**：一位用户指出该频道“禁止使用 Markdown”的规则是“懒惰”的表现，主张在教育他人时使用它，因为它是 **AI 使用的语言**。
   - 他们认为代码块虽然提供了格式化，但增加了一个不必要的抽象层，可能会使不熟悉该格式的用户感到困惑并导致其“卡壳”。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1355087823174369351)** (2 条消息): 

> `Fount AI Character Interactions Framework, Gideon project` 


- **用于 AI 角色交互的 Fount 框架发布**：一名成员分享了 [Fount](https://github.com/steve02081504/fount) 项目，这是一个使用纯 JS 构建和托管 **AI 角色交互** 的可扩展框架。
   - 该框架通过模块化组件、自定义 **AI 源集成**、强大的插件以及无缝的跨平台聊天体验提供了极高的灵活性。
- **Gideon 项目在 GitHub 上线**：一名成员在 GitHub 上分享了 [Gideon 项目](https://github.com/Emperor-Ovaltine/gideon)。
   - 未提供关于该项目用途或功能的进一步细节。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/steve02081504/fount">GitHub - steve02081504/fount: An extensible framework for building and hosting AI character interactions. Built with pure JS, Fount offers unparalleled flexibility via modular components, custom AI source integration, powerful plugins, and a seamless cross-platform chat experience.</a>：一个用于构建和托管 AI 角色交互的可扩展框架。Fount 使用纯 JS 构建，通过模块化组件、自定义 AI 源集成、强大的插件提供无与伦比的灵活性...</li><li><a href="https://github.com/Emperor-Ovaltine/gideon">GitHub - Emperor-Ovaltine/gideon</a>：通过在 GitHub 上创建账号来为 Emperor-Ovaltine/gideon 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1354923122175774922)** (327 条消息 🔥🔥): 

> `Gemini 2.5 Pro Access and Limitations, OpenRouter AI SDK Configuration, Free Models with Function Calling, Token Per Second Performance for Coding Models, OpenAI Responses API` 


- **Gemini 2.5 Pro：速率限制引发用户抱怨**：用户报告称 [Gemini 2.5 Pro 的速率限制较低](https://x.com/OpenRouterAI/status/1905300582505624022)，即使在添加了他们自己的 **AI Studio API keys** 之后也是如此，这引发了关于如何最大化免费配额以及管理实际应用用量的讨论。
   - 一名成员指出，该模型*不会永远免费*，因此 Windsurf 必须开始为此收费，这将会是一个问题。
- **AI SDK Provider 选项，嵌套 order 数组仍是一个难题**：成员们正在积极调试 [OpenRouter AI SDK provider 选项](https://github.com/OpenRouterTeam/ai-sdk-provider)，特别是使用 `providerOptions` 来指定模型顺序和回退行为。
   - 问题在于在 **provider** 键下嵌套 **order 数组** 是否正确，调试尝试显示尽管配置了顺序，但仍会出现非预期的 Provider 选择。团队承认这是一个 Bug，并希望着手解决这个 AI SDK 问题。
- **在免费 LLM 中寻求 Function Calling 的理想选择**：成员们正在寻找支持 Function Calling 的免费模型，一些人建议将 **Mistral Small 3.1** 和 **Gemini 免费模型** 作为潜在选项。
   - 另一名成员提到：“天哪，我正努力寻找一个支持 Function Calling 的免费模型。我一个也找不到！”
- **模型 TPS 对决：Gemini Flash 2.0 对阵其他模型**：成员们正在辩论各种编程模型的 **每秒 Token 数 (TPS)** 性能，其中 **Gemini Flash 2.0** 因其速度被提及，但也面临一些批评，称其表现糟糕，可能是他们的托管服务出了问题。
   - Groq 以 **600tok/s 的速度提供 70B R1 蒸馏模型**，一名成员插话道，他认为该模型“并不擅长编程”。
- **是否支持 OpenAI Responses API？**：一名成员询问 [OpenRouter 是否支持 OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)，OpenRouter 团队的一名成员指出该 API 存在“几个陷阱”。
   - 提问的成员想要高质量的图生视频功能，OpenRouter 团队建议 [Veo2 API](https://openai.blog/veo) 将是追求 SOTA（最先进技术）的最佳选择，但其价格约为 **每秒视频 50 美分**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://openrouter.ai/docs/features/prompt-caching">Prompt Caching - 通过智能缓存优化 AI 模型成本</a>：利用 OpenRouter 的 Prompt Caching 功能降低您的 AI 模型成本。了解如何在 OpenAI、Anthropic Claude 和 DeepSeek 模型中缓存和重用响应。</li><li><a href="https://x.com/OpenRouterAI/status/1905300582505624022">来自 OpenRouter (@OpenRouterAI) 的推文</a>：为了最大化您的免费 Gemini 2.5 配额：1. 在 https://openrouter.ai/settings/integrations 中添加您的 AI Studio API key。我们的速率限制将作为您的“浪涌保护器”。2. 在您的...中设置 OpenRouter。</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>：LLM 的统一接口。为您的 Prompt 找到最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>：了解 OpenRouter 的 API 速率限制、基于积分的配额以及 DDoS 保护。有效配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>：LLM 的统一接口。为您的 Prompt 找到最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - 智能多供应商请求管理</a>：智能地在多个供应商之间路由 AI 模型请求。了解如何通过 OpenRouter 的 Provider Routing 优化成本、性能和可靠性。</li><li><a href="https://github.com/OpenRouterTeam/ai">GitHub - OpenRouterTeam/ai: 使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用</a>：使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用 - OpenRouterTeam/ai</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider?tab=readme-ov-file#passing-extra-body-to-openrouter">GitHub - OpenRouterTeam/ai-sdk-provider: 适用于 Vercel AI SDK 的 OpenRouter 提供程序，通过 OpenRouter 聊天和补全 API 支持数百种模型。</a>：适用于 Vercel AI SDK 的 OpenRouter 提供程序，通过 OpenRouter 聊天和补全 API 支持数百种模型。 - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider: 适用于 Vercel AI SDK 的 OpenRouter 提供程序，通过 OpenRouter 聊天和补全 API 支持数百种模型。</a>：适用于 Vercel AI SDK 的 OpenRouter 提供程序，通过 OpenRouter 聊天和补全 API 支持数百种模型。 - OpenRouterTeam/ai-sdk-provider
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1354893312699666669)** (299 条消息🔥🔥): 

> `MCP 服务器配置, Prompts 和 ICL, Ollama 模型与 MCP, Google 搜索集成, Oterm 客户端与 MCP`

- **Agent 指令与工具使用**：成员们讨论了指导 Agent 进行**工具使用**的最佳实践，特别是关于工具调用顺序的问题，并参考了 [Cline 的系统提示词](https://github.com/cline/cline/blob/main/src/core/prompts/system.ts) 来获取灵感。
   - 一位成员建议直接在服务端进行提示，形式类似于 `First call ${tool1.name}, then ${tool2.name}`。
- **用于上下文学习 (ICL) 的提示词受到关注**：有观点指出 [MCP 服务器可以提供指令](https://github.com/modelcontextprotocol/specification/pull/188#issue-2895415136)，通过使用 ICL 提示词来鼓励特定的 Agent 行为，例如工具使用。
   - 一位成员分享了一个[关于使用 ICL 提示词的链接](https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19)以及一个[展示其效果的测试](https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92)。
- **Ollama 模型配置困惑依然存在**：一位成员在通过 Ollama 将本地 LLM 连接到 MCP 服务器时遇到问题并寻求指导。
   - 建议使用带有[此 MCP 配置](https://ggozad.github.io/oterm/mcp/)的 oterm 并替换配置文件内容，此外还指出默认的 4-bit Ollama 模型通常不足以支持正常的工具使用，建议使用 8-bit 版本以获得更好的性能，更多模型可在[此处](https://ollama.com/library/mistral)获取。
- **讨论向 MCP 添加 Google 实时搜索工具**：一位成员询问如何将 Google Search 添加到 MCP，另一位成员分享了他们的[配置](https://cdn.discordapp.com/attachments/1312302100125843479/1355126409093316750/config.json?ex=67e7cb50&is=67e679d0&hm=8311f31b3b6181eb391876bad03fc45f745e439a12180e6ad087d94983c37c1c&)。
   - 他们指出，用户需要获取自己的 Google API 密钥和引擎 ID 才能使用该配置。
- **发现 Blender MCP 服务器**：一位用户成功使用了该工具并列出了与 Blender 相关的服务器，发现了多个 **Blender Model Context Protocol 服务器**。
   - BlenderMCP ([GitHub](https://github.com/ahujasid/blender-mcp))、Blender MCP Server ([GitHub](https://github.com/cwahlfeldt/blender-mcp))、Unreal-Blender-MCP ([GitHub](https://github.com/tahooki/unreal-blender-mcp))、Bonsai-mcp ([GitHub](https://github.com/JotaDeRodriguez/Bonsai_mcp)) 以及 Tripo MCP Server ([GitHub](https://github.com/VAST-AI-Research/tripo-mcp))。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.48/containers/json":">未找到标题</a>: 未找到描述</li><li><a href="https://ggozad.github.io/oterm/mcp/">索引 - oterm</a>: 未找到描述</li><li><a href="https://ollama.com/library/llama3.2:3b-text-q8_0">llama3.2:3b-text-q8_0</a>: Meta 的 Llama 3.2 推出了 1B 和 3B 的小型模型。</li><li><a href="https://glama.ai/api/mcp/openapi.json"">MCP API 参考</a>: Glama Gateway 的 API 参考</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server">传输层</a>: ℹ️ 协议修订版本：2025-03-26。MCP 使用 JSON-RPC 对消息进行编码。JSON-RPC 消息必须使用 UTF-8 编码。该协议目前定义了两种标准传输机制...</li><li><a href="https://ollama.com/library/mistral">mistral</a>: 由 Mistral AI 发布的 7B 模型，已更新至 0.3 版本。</li><li><a href="https://ollama.com/library/llama3.2:3b-instruct-q8_0">llama3.2:3b-instruct-q8_0</a>: Meta 的 Llama 3.2 推出了 1B 和 3B 的小型模型。</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/188#issue-2895415136">在 GetPrompt 中添加了 Tool Call 和 Tool Result 以支持上下文学习 … 由 evalstate 提交 · Pull Request #188 · modelcontextprotocol/specification</a>: …工具使用。在 PromptMessage 中增加了 ToolCall 和 ToolResult 区块，以允许对工具使用模式和错误处理进行上下文学习。作为草案提交以供审核，随后完成/采用...</li><li><a href="https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19">来自 llmindset (@llmindsetuk) 的推文</a>: 让我们来看看一个被低估的 MCP 功能：Prompts（提示词）——以及为什么它们对于基于 Agent 的应用至关重要。我们将从两个返回对象大小的简单 Agent 开始...</li><li><a href="https://github.com/yuniko-software/minecraft-mcp-server">GitHub - yuniko-software/minecraft-mcp-server: 一个由 Mineflayer API 驱动的 Minecraft MCP Server。它允许实时控制 Minecraft 角色，使 AI 助手能够通过自然语言指令建造建筑、探索世界并与游戏环境互动</a>: 一个由 Mineflayer API 驱动的 Minecraft MCP Server。它允许实时控制 Minecraft 角色，使 AI 助手能够通过自然语言指令建造建筑、探索世界并与游戏环境...</li><li><a href="https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92">fast-agent/tests/e2e/prompts-resources/test_prompts.py (main 分支) · evalstate/fast-agent</a>: 定义、提示并测试支持 MCP 的 Agent 和工作流 - evalstate/fast-agent</li><li><a href="https://github.com/cline/cline/blob/main/src/core/prompts/system.ts">cline/src/core/prompts/system.ts (main 分支) · cline/cline</a>: 直接集成在 IDE 中的自主编码 Agent，能够在每一步都获得你许可的情况下创建/编辑文件、执行命令、使用浏览器等。 - cline/cline</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>: 通过在 GitHub 上创建账号来为 ahujasid/blender-mcp 的开发做出贡献。</li><li><a href="https://github.com/cwahlfeldt/blender-mcp">GitHub - cwahlfeldt/blender-mcp</a>: 通过在 GitHub 上创建账号来为 cwahlfeldt/blender-mcp 的开发做出贡献。</li><li><a href="https://github.com/tahooki/unreal-blender-mcp">GitHub - tahooki/unreal-blender-mcp: unreal-blender-mcp</a>: unreal-blender-mcp。通过在 GitHub 上创建账号来为 tahooki/unreal-blender-mcp 的开发做出贡献。</li><li><a href="https://github.com/JotaDeRodriguez/Bonsai_mcp">GitHub - JotaDeRodriguez/Bonsai_mcp</a>: 通过在 GitHub 上创建账号来为 JotaDeRodriguez/Bonsai_mcp 的开发做出贡献。</li><li><a href="https://github.com/VAST-AI-Research/tripo-mcp">GitHub - VAST-AI-Research/tripo-mcp: 针对 Tripo 的 MCP server</a>: 针对 Tripo 的 MCP server。通过在 GitHub 上创建账号来为 VAST-AI-Research/tripo-mcp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1354895650457780306)** (9 条消息🔥): 

> `Canvas MCP, 用于 MCP 服务器的 Docker Compose, Model Context Protocol (MCP) 解释, Speech MCP, Gradescope 集成` 


- **Canvas MCP 让 Agent 与 Canvas LMS 对话**：一位成员创建了一个 **Canvas MCP** 服务器，使 AI Agent 能够与 Canvas LMS 交互，并添加了一个可以自主抓取 Gradescope 信息以查找信息的 Agent，可在 [Canvas-MCP](https://git.new/canvas-mcp) 获取。
   - 该工具提供查找相关资源、查询即将截止的作业以及访问来自 **Gradescope** 的课程和作业等功能。
- **用于自托管 17 个 MCP 服务器的全能 Docker Compose**：一位成员创建了一个全能的 **Docker Compose** 配置，用于使用 Portainer 轻松自托管 **17 个 MCP 服务器**，其 Dockerfile 源自公开的 GitHub 项目 ([MCP-Mealprep](https://github.com/JoshuaRL/MCP-Mealprep))。
   - 另一位成员建议 *除非需要远程访问，否则不要将容器绑定在 0.0.0.0*，并在 *readme 中包含一个 mcp config json 示例*。
- **Model Context Protocol (MCP) 详解**：一位团队成员分享了一篇介绍 **MCP (Model Context Protocol)** 的 [博客文章](https://pieces.app/blog/mcp)，将其描述为 Anthropic 在 2024 年底发布的开放标准。
   - 该博文将其描述为 *AI 集成的 USB-C*，允许驱动 Claude 或 ChatGPT 等工具的 Large Language Models 与外部数据源和工具进行通信。
- **Speech MCP 演示**：一位成员分享了 **Speech MCP** 的链接，并附带了一个 [YouTube Shorts 演示](https://www.youtube.com/shorts/rurAp_WzOiY)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/JoshuaRL/MCP-Mealprep">GitHub - JoshuaRL/MCP-Mealprep: This project takes a number of MCP servers from GitHub locations, packages them together with their referenced Dockerfiles, and pulls them together with docker-compose to run as a stack for ML/AI resources.</a>: 该项目从 GitHub 获取多个 MCP 服务器，将它们与引用的 Dockerfile 打包在一起，并使用 docker-compose 将它们拉取到一起，作为 ML/AI 资源的堆栈运行。</li><li><a href="https://pieces.app/blog/mcp">What the heck is Model Context Protocol (MCP)? And why is everybody talking about it?</a>: 探索什么是 Model Context Protocol 或 MCP 以及它为何流行。了解它如何改变开发者和团队的游戏规则。</li><li><a href="https://git.new/canvas-mcp">GitHub - aryankeluskar/canvas-mcp: Collection of Canvas LMS and Gradescope tools for the ultimate EdTech model context protocol. Allows you to query your courses, find resources, and chat with upcoming assignments in the AI app of your choice. try now!</a>: Canvas LMS 和 Gradescope 工具集，打造终极 EdTech model context protocol。允许你在自选的 AI 应用中查询课程、查找资源并针对即将截止的作业进行聊天。立即尝试！</li><li><a href="https://www.youtube.com/shorts/rurAp_WzOiY"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1354893946190299187)** (216 条消息🔥🔥): 

> `R1 vs O3 Mini, Anthropic Thoughts Microscope, GPT-4o 更新, OpenRouter 限制, 使用 UV 运行本地 Aider 分支`

- **OpenRouter R1 模型表现不佳**：一位成员发现 OpenRouter 上的免费 **R1** 模型非常“愚蠢”、啰嗦，且在解决损坏的测试时效果不佳，尤其是在启用 repomap 的情况下，与 **O3-mini** 形成鲜明对比。
   - 据推测，免费的 **R1** 模型是 **DeepSeek** 的量化版本，可能是 FP8 格式，而排行榜上的 DeepSeek 则来自官方 DeepSeek 团队。
- **GPT-4o 在 Coding Arena 中表现出色**：最新的 **ChatGPT-4o** 更新在 [Arena 排行榜](https://x.com/lmarena_ai/status/1905340075225043057)上跃升至第 2 位，超越了 **GPT-4.5**，在 Coding 和 Hard Prompts 类别中并列第 1，并在所有类别中均位列前 2，且成本降低了 10 倍。
   - 然而，令人困惑的是，这次更新是以 **chatgpt-4o-latest** 端点的形式发布的，其价格为每百万输入/输出 token $5/$15，而 API 快照的价格为 $2.5/$10。根据 [Artificial Analysis](https://x.com/ArtificialAnlys/status/1905563427776651344) 的说法，在迁移工作负载时建议保持谨慎。
- **Constant Context Architecture 是颠覆性的**：Constant Context Architecture (**CCA**) 被提议作为使用 LLM 处理大型代码库的解决方案，它保证修改任何模块所需的上下文始终能容纳在 LLM 的 context window 内，无论代码库的总规模有多大，详见这篇 [博文](https://neohalcyon.substack.com/p/constant-context-architecture)。
   - 这是通过确保模块具有受限的大小、接口和依赖关系来实现的，从而使上下文收集成为一种受限的操作。
- **Aider 的 Context 命令实现了文件管理自动化**：新的 `/context` 命令会自动识别与给定请求相关的文件并将其添加到对话中，正如在 [此 Discord 线程](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809) 中讨论的那样。
   - 它对于大型代码库特别有用，通过自动化手动添加文件的过程来节省时间。
- **多个 API Key**：OpenRouter 用户正在轮换使用多个 API Key，以确保能够发出尽可能多的请求，同时避免 rate limits。
   - 然而，众所周知，如果 Google 检测到单个用户拥有多个 Key 或存在滥用行为，可能会封禁账号。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1905303835892990278">来自 Anthropic (@AnthropicAI) 的推文</a>：Anthropic 最新研究：追踪大语言模型的思维。我们构建了一个“显微镜”来检查 AI 模型内部发生的情况，并用它来理解 Claude（通常是复杂且...</li><li><a href="https://x.com/BenjaminDEKR/status/1905461907156271465">来自 Benjamin De Kraker (@BenjaminDEKR) 的推文</a>：POV：Cursor “正在修复你的代码”</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：相比 1 月版本有显著提升（+30 分，从第 5 名升至第 2 名）；在编程（Coding）和困难提示词（Hard Prompts）方面并列第 1。</li><li><a href="https://x.com/OpenAI/status/1905331956856050135">来自 OpenAI (@OpenAI) 的推文</a>：GPT-4o 在 ChatGPT 中又获得了更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词 - 提高了处理复杂技术任务的能力...</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：今天的 GPT-4o 更新确实意义重大——它在我们的智能指数中超越了 Claude 3.7 Sonnet (非推理型) 和 Gemini 2.0 Flash，目前是领先的非推理型编程模型。这使得 GPT...</li><li><a href="https://neohalcyon.substack.com/p/constant-context-architecture">恒定上下文架构 (Constant Context Architecture)</a>：为 LLM 时代而构建</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://aider.chat/docs/repomap.html#optimizing-the-map">仓库映射 (Repository map)</a>：Aider 使用您的 git 仓库映射为 LLM 提供代码上下文。</li><li><a href="https://github.c>>>">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (免费) - API、提供商、统计数据</a>：DeepSeek V3 是一个拥有 685B 参数的混合专家模型 (Mixture-of-Experts)，是 DeepSeek 团队旗舰聊天模型系列的最新迭代。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...</li><li><a href="https://aider.chat/docs/repomap.html">仓库映射 (Repository map)</a>：Aider 使用您的 git 仓库映射为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token 限制</a>：Aider 是您终端里的 AI 结对编程工具。</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 code、architect、ask 和 help 聊天模式。</li><li><a href="https://tenor.com/view/primary-day-polling-place-homer-simpson-the-simpsons-voting-gif-12450826">初选日投票站 GIF - Primary Day Polling Place Homer Simpson - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1354900540370456868)** (31 条消息🔥): 

> `AiderMacs, Cargo Build Integration, Gemini 2.5 Pro Rate Limits, Aider Architect Mode, Model Combinations` 


- **调试 Aidermacs 集成**：一位用户提到正在调试 **Aidermacs** 中的一个 Bug，他们使用该工具来调用 **Aider** 并将其更紧密地集成到 **Emacs** 中，并将其过程描述为 *“shaving a yak”*（薅牦牛毛，意指为了完成小事而不得不先做一系列繁琐的准备工作）。
   - 该用户还澄清了一个 `lint-cmd` 配置细节，指出它应该是 `echo` 而不是 `test`。
- **重新开始胜过原地打转**：一位用户询问是应该迭代使用 `/undo` 还是继续对话，另一位用户建议如果 *陷入死循环*，应使用 `/clear` 重新开始。
   - 对话强调了 `/undo` 会保留上一次操作的记忆，而 `/clear` 会清除整个聊天历史。
- **Cargo Build 集成进入愿望清单**：一位用户询问是否可以将 `cargo build` 与 **Aider** 集成，以便将错误/警告通过管道传回给模型进行修复。
   - 虽然目前没有提供直接的解决方案，但该询问表明了用户对增强代码调试工作流功能的渴求。
- **Gemini 2.5 Pro 速率限制令人沮丧**：多位用户报告达到了 **Gemini 2.5 Pro** 的速率限制，即使似乎低于文档说明的 **50 requests/day**，其中一位指出存在 **2 requests/minute** 的限制。
   - 讨论涉及了购买付费账户是否能解决这些限制，报告结果不一，同时还讨论了潜在的备选模型实现。
- **请求深入探讨 Architect Mode**：一位用户寻求对 **Aider** 的 **Architect Mode** 进行更深入的理解，并将其与他们目前使用 **Sonnet 3.5** 和 **Gemini 2.5 Pro** 配合 `aider.rules.md` 文件的常规模式工作流进行对比。
   - 他们表达了优化流程的愿望，希望利用 **Architect Mode** 来避免重复工作或为模型使用支付过高费用；另一位成员将其视为 *ask + code* 的组合。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743">Gemini 2.5 Pro 或 DeepSeek V3 0324 未在 `/models /` 中显示 · Issue #3641 · Aider-AI/aider</a>：我一直在使用 `/models /` 获取可用模型列表，并基于 Aidermacs 从列表中进行选择，我很高兴 Aider 支持了 Gemini 2.5 Pro 和最新的 DeepSeek...</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 code、architect、ask 和 help 聊天模式。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1354894995294912723)** (26 条消息🔥): 

> `GPT-4o Update, OpenAI Image Generation Policy, Devin Wiki Launch, AI Writing Editing` 


- **GPT-4o 跃升至 Arena 排行榜第 2 名！**：最新的 **ChatGPT-4o** (2025-03-26) 在 Arena 上跃升至 **第 2 名**，超越了 **GPT-4.5**，相比 1 月份的版本有显著提升（+30 分，从第 5 名升至第 2 名），并且根据这条 [推文](https://fxtwitter.com/lmarena_ai/status/1905340075225043057)，它在 Coding 和 Hard Prompts 类别中并列 **第 1 名**。
- **新的 4o 政策允许更多创作自由**：OpenAI 通过 **4o** 在 **ChatGPT** 中推出了原生图像生成功能，从在敏感领域的一刀切拒绝转变为更精确的方法，重点在于防止现实世界的伤害，详见这篇 [博客文章](https://x.com/joannejang/status/1905341734563053979)。
- **Devin 通过 Devin Wiki 索引仓库**：根据这条 [推文](https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)，**Devin** 现在会自动索引你的仓库并生成包含架构图、源码链接等内容的 Wiki。
- **AI 辅助非技术写作编辑**：成员们讨论了使用 **Claude** 和 **GPT** 进行非技术写作的编辑，一位成员指出 *Claude 会验证我写的所有内容，而 GPT 想要重写我写的所有内容*，该用户希望能找到 *更好的折中方案*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/artificialanlys/status/1905563427776651344?s=46">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：今天的 GPT-4o 更新确实意义重大——它在我们的 Intelligence Index 中超越了 Claude 3.7 Sonnet (non-reasoning) 和 Gemini 2.0 Flash，目前是领先的用于 Coding 的 non-reasoning 模型。这使得 GP...</li><li><a href="https://x.com/julianlehr/status/1855858599156932773">来自 Julian Lehr (@julianlehr) 的推文</a>：“我有一个想写的文章构思。它是一篇博客文章，比如长篇博客，但目前还不是很具体，我只是对这篇博客文章应该是什么样有一个粗略的想法...”</li><li><a href="https://x.com/cognition_labs/status/1905328447318311218?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">来自 Cognition (@cognition_labs) 的推文</a>：我们正在发布 Devin Search，这是一个用于代码库理解的新工具。使用 Devin Search 可以快速回答诸如“用户身份验证是如何实现的？”之类的问题，或者开启 Deep Mode 处理复杂需求，例如...</li><li><a href="https://x.com/levelsio/status/1905324525006299521?s=46">来自 @levelsio (@levelsio) 的推文</a>：致所有正在制作 Ghibli 生成器应用的数百人：如果你给它起名叫 Ghibli 或类似的名称，你会收到 Studio Ghibli 律师的信函。但更糟的是，如果你赚了数百万，你只会...</li><li><a href="https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5">来自 Cognition (@cognition_labs) 的推文</a>：发布 Devin Wiki：Devin 现在会自动索引你的仓库并生成包含架构图、源码链接等内容的 Wiki。使用它来快速熟悉代码库中不熟悉的部分...</li><li><a href="https://x.com/LangChainAI/status/1905325891934454170">来自 LangChain (@LangChainAI) 的推文</a>：观看视频 ➡️ https://www.youtube.com/watch?v=NKXRjZd74ic</li><li><a href="https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">来自 Cognition (@cognition_labs) 的推文</a>：发布 Devin Wiki：Devin 现在会自动索引你的仓库并生成包含架构图、源码链接等内容的 Wiki。使用它来快速熟悉代码库中不熟悉的部分...</li><li><a href="https://fxtwitter.com/joannejang/status/1905341734563053979)">来自 Joanne Jang (@joannejang) 的推文</a>：// 我在 OpenAI 负责模型行为，想分享一些在制定 4o 图像生成政策时的想法和细微差别。使用了大写字母 (!)，因为我是以博客文章的形式发布的：--Thi...</li><li><a href="https://x.com/joannejang/status/1905341734563053979>)">来自 Joanne Jang (@joannejang) 的推文</a>：// 我在 OpenAI 负责模型行为，想分享一些在制定 4o 图像生成政策时的想法和细微差别。使用了大写字母 (!)，因为我是以博客文章的形式发布的：--Thi...</li><li><a href="https://fxtwitter.com/OpenAI/status/1905331956856050135)">来自 OpenAI (@OpenAI) 的推文</a>：GPT-4o 在 ChatGPT 中又迎来了更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词 - 提升了处理复杂技术任务的能力...</li><li><a href="https://x.com/OpenAI/status/1905331956856050135>)">来自 OpenAI (@OpenAI) 的推文</a>：GPT-4o 在 ChatGPT 中又迎来了更新！有什么不同？- 更好地遵循详细指令，特别是包含多个请求的提示词 - 提升了处理复杂技术任务的能力...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057)">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：- 相比 1 月版本有显著提升（+30 分，从第 5 升至第 2）- 在 Coding 和 Hard Prompts 类别中并列第 1。To...</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057>)">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 排名跃升至第 2 位，超越了 GPT-4.5！亮点：- 相比 1 月版本有显著提升（+30 分，从第 5 升至第 2）- 在 Coding 和 Hard Prompts 类别中并列第 1。To...
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1355238157896912946)** (3 messages): 

> `Dharmesh Shah, HubSpot, Agent.ai, hybrid teams, Claude Plays Pokemon hackathon` 


- ****Dharmesh Shah** 加入 Latent Space**: Latent Space 分享了与 **[Dharmesh Shah](https://x.com/dharmesh/status/1789687037261402336)** 的对话，他是 **HubSpot** 的联合创始人以及 **[Agent.ai](http://agent.ai/)** 的创建者。
   - 本期节目探讨了*工作场所组织的下一次演进，即人类员工与 AI Agent 作为团队成员共同协作*。
- **参与者可以加入 **Claude Plays Pokemon hackathon****：在旧金山的朋友们：本周日加入我们的 **[Claude Plays Pokemon hackathon](https://lu.ma/poke)**！
   - 这是一个真正的 *catch 'em all*（全数捕捉）的机会。
- **鼓励成员填写 **2025 State of AI Eng survey****：不在旧金山的成员受邀填写 **[2025 State of AI Eng survey](https://www.surveymonkey.com/r/57QJSF2)**，有机会获得价值 **$250** 的 Amazon 礼品卡！
   - 动作要快，争取赢取 Amazon 奖励金。
- ****Hybrid teams** 是未来**: 我们讨论的一个特别引人注目的概念是 **"hybrid teams"** —— *工作场所组织的下一次演进，人类员工与 AI Agent 作为团队成员共同协作*。
   - 这引发了关于*团队动态、信任以及如何在人类和 AI 团队成员之间有效地委派任务*的有趣思考。



**Link mentioned**: <a href="https://www.latent.space/p/dharmesh">The Agent Network — Dharmesh Shah</a>: Dharmesh Shah 谈智能 Agent、市场低效以及构建下一个 AI 市场

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1355270295279370370)** (189 条消息🔥🔥): 

> `LLM Codegen 工作流, LLM 文档, Memory-Ref 工具, Cursor IDE, 自我改进 Agent` 


- **Harper 揭秘 LLM Codegen 工作流**：一位成员分享了他们的 [LLM codegen 工作流](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)，强调了**头脑风暴规格说明 (specs)**、规划，以及在离散循环中执行 LLM codegen。
   - 该工作流基于个人实践、与朋友的交流以及来自互联网各处的最佳实践，不过发布者指出，*它可能在两周内失效，或者效果会提升一倍*。
- **Docs.dev 接入 GitHub 实现流程化文档生成**：分享了 [Docs.dev](https://docs.dev/)，用于直接从代码库生成文档，并随代码更改保持同步更新。
   - 该工具允许用户利用 AI **生成、审计或分析 Markdown 文档**，同时提供富文本编辑器和 Markdown 选项。
- **Nuvic 的 FZF Kit 扩展 Neovim 的模糊查找功能**：一位成员链接了 [fzf-kit.nvim](https://github.com/nuvic/fzf-kit.nvim)，这是一个 Neovim 插件，为 fzf-lua 扩展了额外的实用工具。
   - 该插件增强了 Neovim 的模糊查找能力，提升了文件和代码导航效率。
- **Memory-Ref 工具辅助 LLM 上下文保留**：成员们讨论了使用 **memory-ref** 工具为 LLM 创建和查询记忆知识图谱，帮助其在不同会话间保留上下文。
   - 一位用户强调了 **Cursor IDE** 与 **Graphiti** 的集成，利用 Graphiti 的 Model Context Protocol (MCP) 服务端实现持久化记忆，详见这篇 [Hacker News 帖子](https://news.ycombinator.com/item?id=43506068)。
- **用于网站-LLM 协作的 llms.txt 出现**：一位成员分享了 [llms-txt](https://github.com/AnswerDotAI/llms-txt)，这是一个旨在帮助语言模型更有效地使用网站的文件。
   - 讨论涉及了自我改进模型这一更广泛的话题，以及如何通过结构化文档引导 LLM。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>：详细介绍了目前我使用 LLM 构建软件的工作流，涵盖从头脑风暴到规划和执行的全过程。</li><li><a href="https://news.ycombinator.com/item?id=43506068">Show HN: Cursor IDE now remembers your coding prefs using MCP | Hacker News</a>：暂无描述</li><li><a href="https://x.com/PrajwalTomar_/status/1895839765280539068?s=19">Prajwal Tomar (@PrajwalTomar_) 的推文</a>：在过去的 5 个月里，我使用 Cursor 为客户构建了 16 个 SaaS 产品。现在，我摸索出了 Cursor 的最佳 AI 编码工作流。这是我构建生产级 MVP 的分步指南：</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>：CodeGuide 为你的 AI 编码项目创建详细文档。</li><li><a href="https://github.com/AnswerDotAI/llms-txt">GitHub - AnswerDotAI/llms-txt: The /llms.txt file, helping language models use your website</a>：/llms.txt 文件，帮助语言模型使用你的网站 - AnswerDotAI/llms-txt</li><li><a href="https://docs.dev/">Docs.dev | AI-assisted docs</a>：直接从代码库和现有文档生成文档。确保文档随代码更改保持最新。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>：暂无描述</li><li><a href="https://github.com/nuvic/fzf-kit.nvim">GitHub - nuvic/fzf-kit.nvim: A Neovim plugin that extends fzf-lua with additional utilities</a>：一个扩展 fzf-lua 并提供额外实用工具的 Neovim 插件 - nuvic/fzf-kit.nvim</li><li><a href="https://github.com/go-go-golems/go-go-mcp/tree/main/ttmp">go-go-mcp/ttmp at main · go-go-golems/go-go-mcp</a>：Anthropic MCP 的 Go 语言实现。</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md">go-go-labs/ttmp/2025-03-23/03-add-embeddings-to-command.md at main · go-go-golems/go-go-labs</a>：GO GO 实验实验室。</li><li><a href="https://github.com/joernio/astgen">GitHub - joernio/astgen: Generate AST in json format for JS/TS</a>：为 JS/TS 生成 JSON 格式的 AST。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1354943243271274646)** (1 条消息): 

> `LM Studio 0.3.14 Release, Multi-GPU Controls, GPU Management Features, Beta Releases, Advanced GPU Controls` 


- **LM Studio 0.3.14 发布，带来 Multi-GPU 掌控力**：LM Studio **0.3.14** 已发布，具有针对 Multi-GPU 设置的新细粒度控制功能，可通过应用内更新或从 [https://lmstudio.ai/download](https://lmstudio.ai/download) 下载。
   - 此版本引入了启用/禁用特定 GPU、选择分配策略（**evenly (均匀), priority order (优先级顺序)**）以及将模型权重限制在专用 GPU 显存的功能，部分功能最初仅限 NVIDIA GPU。
- **LM Studio GPU 专家的新控制选项！**：LM Studio **0.3.14** 引入了管理 GPU 资源的新控件，包括启用/禁用单个 GPU 和选择分配策略。
   - 特定的 CUDA 功能，如 **"Priority order"** 模式和 **"Limit Model Offload to Dedicated GPU memory"** 模式，旨在提高稳定性并优化单 GPU 设置下的长上下文（long context）。
- **LM Studio GPU 控制的快捷键**：LM Studio **0.3.14** 引入了打开 GPU 控制的快捷方式：`Ctrl+Shift+H` (Windows) 或 `Cmd+Shift+H` (Mac)，以及通过 `Ctrl+Alt+Shift+H` (Windows) 或 `Cmd+Option+Shift+H` (Mac) 打开弹出窗口。
   - 使用弹出窗口，您可以在 *模型加载时管理 GPU 设置*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.14">LM Studio 0.3.14: Multi-GPU 控制 🎛️</a>: 针对 Multi-GPU 设置的高级控制：启用/禁用特定 GPU，选择分配策略，将模型权重限制在专用 GPU 显存等。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1354932934166974655)** (74 条消息🔥🔥): 

> `Threadripper vs EPYC, LM Studio UI, 可视化 LLM 计算, LM Studio 中的模型详情错误, Continue VSCode 扩展` 


- **Threadripper 给 EPYC 上了一课**：成员们讨论了 **Threadripper** 是属于消费级还是专业级，一些人指出虽然技术上属于 HEDT（高端桌面级），但 AMD 并不像推销 Threadripper 那样向家庭用户推广 **EPYC**。
   - 一位成员分享了 [GamersNexus 对 AMD Ryzen Threadripper 7960X 的评测](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks)，强调了其 24 核配置以及相对于专业工作站而言较低的成本。
- **LLM 计算可视化？**：一位成员询问如何可视化模型执行的计算，例如将数值映射到像素颜色。
   - 另一位成员分享了 [bbycroft 的 LLM Visualization](https://bbycroft.net/llm)，并推荐了 3b1b 关于 LLM 的播放列表，以及一本关于从零开始构建 LLM 的书，以进行更深入的理解。
- **Studio SDK 引起好奇**：一位成员询问 LM Studio 在何处调用模型，以及它如何强制使用 `<think>` 和 `</think>` 标签。
   - 另一位成员澄清说 LM Studio 并非完全开源，只有 SDK 是开源的，并指向了 [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 [MLX engine](https://github.com/lmstudio-ai/mlx-engine) 的 GitHub 仓库以获取相关源代码。
- **Studio 的错误困扰**：一位用户报告在 Windows 11 上遇到 `Model details error: fetch failed` 问题，尽管尝试了使用 Hugging Face 代理、更改 DNS 设置和使用 VPN 等各种解决方案。
   - 一个建议涉及检查 **"killer network service"**，并提供了一篇 [Intel 支持文章](https://www.intel.com/content/www/us/en/support/articles/000058995/ethernet-products/intel-killer-ethernet-products.html) 来解决潜在的网络相关冲突。
- **Continue 信心编码**：一位成员询问如何通过扩展将 LM Studio 连接到 VSCode。
   - 另一位成员分享了 [Continue.dev](https://www.continue.dev/) 的链接，将其描述为一个用于创建自定义 AI 代码助手的平台，可以在任何编程语言中自动补全代码。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: 未找到描述</li><li><a href="https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks">AMD 的 "廉价" Threadripper HEDT CPU：7960X 24 核 CPU 评测与基准测试 | GamersNexus</a>: CPUs AMD 的 "廉价" Threadripper HEDT CPU：7960X 24 核 CPU 评测与基准测试 2024 年 1 月 2 日 最后更新：2024-01-02 AMD Ryzen Threadripper 7960X 提供了一个引人注目的选择...</li><li><a href="https://www.continue.dev/">Continue</a>: 增强开发者能力，AI 增强开发 · 领先的开源 AI 代码助手。您可以连接任何模型和任何上下文，在 IDE 内部构建自定义的自动补全和聊天体验</li><li><a href="https://harddiskdirect.com/mbd-x10drd-l-o-supermicro-desktop-motherboard.html?utm_source=google&utm_medium=cpc&src=google-search-US&network=x&place=&adid=&kw=&matchtype=&adpos=&device=m&gad_source=1&gclid=CjwKCAjw7pO_BhAlEiwA4pMQvLmwxPZ31Lo40g1U-HBINy6kbrwrcuXi081dt53eLdZ-jusZyxJ9RxoChlMQAvD_BwE">MBD-X10DRD-L-O - Supermicro LGA2011 C612 芯片组 EATX 主板</a>: 在我们美国领先的行业中探索 MBD-X10DRD-L-O Supermicro LGA2011 C612 芯片组 EATX 主板，享受快速发货和最优惠价格。</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: C/C++ 实现的 LLM 推理</a>: C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号来为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/mlx-engine">GitHub - lmstudio-ai/mlx-engine: 用于 LM Studio 的 Apple MLX 引擎</a>: 用于 LM Studio 的 Apple MLX 引擎。通过在 GitHub 上创建账号来为 lmstudio-ai/mlx-engine 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1354940327592591555)** (71 messages🔥🔥): 

> `ROCm Support, P100 vs 6750xt, Nvidia vs AMD, Mac Pro 2013 for LLMs` 


- **ROCm 支持依然不明朗**：用户讨论了 **LM Studio** 中 **ROCm** 支持的现状，一名用户最初误读了文档，希望在其 **7800 XT** 上获得 **ROCm** 支持。
   - 会议澄清，**ROCm** 仅在具有 **GFX1030, 1100, 和 1101** 运行时的显卡上受支持。
- **P100 被弃，6750xt 封王**：一名用户询问是否可以将 **P100 16GB** 用于兴趣爱好，但得到的建议是否定的，称其与 **6750xt** 相比基本上是“电子垃圾”。
   - **6750xt** 被认为是一款更好、更现代的显卡，可以通过 **Vulkan** 工作，而 **P100** 由于其不支持的 **CUDA** 版本，被认为不值得购买。
- **Nvidia 显卡并非毫无问题**：在 Windows 上使用 AMD 经历过延迟和文字视觉效果卡顿后，一名用户考虑转向 NVIDIA，但听说 **40 系列到 50 系列的跨越并不大**，且 **5080** 在 **VRAM** 上完全被阉割。
   - 他们表示担心 *Nvidia 已经放弃了 GPU*，转而 *只生产服务器芯片*。
- **2013 款 Mac Pro 运行 LLM 注定失败**：一名用户考虑使用拥有 **128GB RAM** 的“垃圾桶” **Mac Pro** (2013) 来运行 **LLM**，理由是其运行安静且外观精美。
   - 然而，有人指出 **LM Studio** 不适用于 **Intel Macs**，且这些型号中的 **Xeon v2 CPUs** 缺乏 **AVX2** 支持，限制了它们的可用性。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1354918480314503189)** (23 messages🔥): 

> `transformer storage errors, torchtune use cases, self-awareness in language models, bias-augmented consistency training (BCT), adaptive compression + intelligent routing for distributed systems` 


- **Transformer 存储问题误导用户**：一名用户发现存储空间不足导致 **transformers v4.50.0** 中出现误导性的错误消息，指向库问题而非存储；目前计划提交一个用于更好处理错误的 PR。
   - 用户不得不求助于 `df -h` 来诊断 **100% 满** 的系统，建议在下载模型分片（shards）之前检查是否有足够的容量。
- **Torchtune：鼓励深入代码**：一名用户发现 **torchtune** 需要下载并编辑 **200 行的 PyTorch 脚本** 和 YAML 文件进行自定义。
   - 另一名用户反驳称，这种方法提供了流程的全貌，避免了剖析 **Hugging Face 实现** 的麻烦。
- **内省训练引发关注**：受 [Anthropic 的工作](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 启发，一名成员建议通过创建其电路的表示并将其反馈回来，来模拟 LLM 的自我意识。
   - 另一名成员支持这一想法，并链接了一篇关于 **偏差增强一致性训练 (BCT)** 的[论文](https://arxiv.org/abs/2403.05518v1)，作为内省方法的验证手段。
- **自适应压缩助力分布式系统**：一名成员正在开发一个基础设施层，利用自适应压缩和智能路由优化模型在分布式系统中的传输和部署，以解决 **带宽浪费** 和 **推理延迟** 问题。
   - 该基础设施对于扩展大型模型特别有用，并向对 **分布式推理 (distributed inference)** 感兴趣的人提供演示。



**提到的链接**：<a href="https://arxiv.org/abs/2403.05518v1">Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought</a>：虽然思维链提示 (CoT) 具有提高语言模型推理可解释性的潜力，但它可能会系统性地误传影响模型行为的因素——对于...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1355131273483649094)** (2 messages): 

> `Architectural inductive biases, Neural-guided CoT, Reasoning-adjacent work` 


- **AI 研究领域兴趣查询**：一名成员询问了特定的兴趣领域，例如 **架构归纳偏置 (architectural inductive biases)**、**神经引导的 CoT** 或 **推理相关工作**。
   - 该询问旨在缩小研究频道内的讨论范围。
- **跟进 AI 研究热点话题**：有人对探索前沿 AI 研究课题表现出兴趣。
   - 讨论旨在突出该领域最近的进展和潜在的未来方向。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1354896097734299822)** (83 messages🔥🔥): 

> `神经网络作为无器官身体 (BwO), 机械可解释性 (Mech Interp) 批判, 神经网络中的专用头, 九头蛇效应 (Hydra Effect), AI 安全的推理模型` 


- **神经网络作为**无器官身体 (BwO)****：基于一条 [推文](https://x.com/norabelrose/status/1905336894038454396)，神经网络没有器官，不是由固定机制构成的，而是具有信息流和神经活动强度，用吉尔·德勒兹（Gilles Deleuze）的话说，它们是**无器官身体 (BwO)**。
   - 一位成员拒绝机械可解释性的概念，认为**神经网络在没有固定机制的情况下进行泛化**；笛卡尔在 400 年前就预见到了这一点。
- **对当前 **Mech Interp** 方法的批判**：一位成员认为，自 IOI 工作以来，机械可解释性走入了歧途，机制的概念已被扭曲为某种极其特定于输入的东西。
   - 他们认为，*大多数 Mech Interp 相当于 FMRI 扫描，而 FMRI 以容易产生错误描述而闻名。*
- **“九头蛇效应” (**Hydra Effect**) 使 **Mech Interp** 复杂化**：[Hydra Effect 论文](https://arxiv.org/abs/2307.15771) 使任何对神经网络的“机械式”理解变得复杂，因为功能是不可还原地分散在各处的。
   - 理论上的机制应该具有局部化功能，它质疑重新参数化是否会导致模型的原始行为。
- ****CoT** 实际上很好**：一位成员表示，尽管有一些零散的轶闻，但 [CoT 实际上很好](https://arxiv.org/abs/2503.11926)，并且是最好的可解释性工具之一。
   - 他们链接了一项新研究，探讨 *CoT 监控如何比单纯监控 Agent 的动作和输出更有效，我们还进一步发现，比 o3-mini 更弱的 LLM（即 GPT-4o）可以有效地监控更强的模型。*
- **“非成员” (**Non-Member**) 状态可以被有效地博弈**：一位成员强调的研究表明，[很难通过 n-gram 重叠来定义数据集成员身份](https://arxiv.org/abs/2503.17514)。
   - 即使序列是*非成员*，补全测试仍然会成功，这展示了在成员身份定义中寻找单一可行的 n 选择的困难。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17514">Language Models May Verbatim Complete Text They Were Not Explicitly Trained On</a>：当今一个重要的问题是，给定的文本是否被用于训练大语言模型 (LLM)。通常会使用“补全”测试：检查 LLM 是否能补全一段足够复杂的文本。...</li><li><a href="https://x.com/norabelrose/status/1905336894038454396">Nora Belrose (@norabelrose) 的推文</a>：神经网络没有器官。它们不是由固定机制构成的。它们具有信息流和神经活动强度。它们无法被组织成一组具有特定功能的部件...</li><li><a href="https://arxiv.org/abs/2503.11926">Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation</a>：减轻奖励破解（reward hacking）——即 AI 系统由于学习目标的缺陷或误设而表现不当——仍然是构建高性能且对齐模型的关键挑战。我们展示了...</li><li><a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268">Could a Neuroscientist Understand a Microprocessor?</a>：作者摘要：神经科学的发展受到以下事实的阻碍：很难评估一个结论是否正确；所研究系统的复杂性及其实验上的不可接近性使得...
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1355169099281596537)** (19 条消息🔥): 

> `MMLU pro 数据集路径, MMLU pro process_doc 函数, MMLU pro 评估修改, MMLU pro COT 内容, LM harness 选择数据集` 


- **数据集路径指向 MMLU pro 下载位置**：一位成员询问是否可以通过修改 [lm-evaluation-harness 的 _default_template_yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml) 中的数据集路径来更改 **MMLU pro** 的下载位置，因为该路径看起来像是一个 **HF repo ID**。
- **MMLU Pro 缺少专用的处理函数**：一位成员注意到 [lm-evaluation-harness 的 process_docs.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/metabench/process_docs.py#L109) 中缺少 **MMLU Pro** 专用的 **`process_doc`** 函数，并询问其处理机制。
   - 回复澄清说，其他子任务配置使用带有 `include` 的基础模板并指定子任务特定的字段，并且使用 [lm-evaluation-harness 的 utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py) 来过滤样本。
- **调整 MMLU Pro 数据集：编辑技巧**：一位成员询问，如果对评估进行轻微修改（例如更改 **MMLU pro 数据集** 中的顺序或删除特定选项），是否只需更改 **default task YAML** 和 mmlu-pro 的 utils 即可。
- **COT 内容仅用于 Few-Shot 示例**：一位成员询问了评估期间 **MMLU-pro 数据集** 中 **cot_content** 的相关性，并注意到 llm-harness 在初始行中进行了正则表达式模式匹配。
   - 澄清指出，**COT 内容** 仅用于格式化 few-shot 示例，要求数据集中的格式为 `Answer: Let’s …` 而不是 `A: Let’s …`；他们只是在每个 fewshot 中添加一个参考答案，该答案不会被添加到主问题之后。
- **掌握 Few-Shot 选择**：一位成员询问如何控制 few shot 中摄取的 5 个样本。
   - 他们被引导至 [lm-evaluation-harness 的文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#selecting-and-configuring-a-dataset)，该文档涵盖了如何选择和配置数据集。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#selecting-and-configuring-a-dataset">lm-evaluation-harness/docs/new_task_guide.md at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/metabench/process_docs.py#L109">lm-evaluation-harness/lm_eval/tasks/metabench/process_docs.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py#L50)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py#L42)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1354978576298016898)** (2 messages): 

> `Dependency Issue, Test Understanding` 


- **依赖问题浮现**：一名成员指出测试引发了一个依赖项问题。
   - 最终决定，与当前活跃项目相比，修复此依赖问题属于*低优先级*。
- **不同的测试解读**：一位用户表达了对某些测试的理解，并寻求另一位用户的验证。
   - 讨论围绕特定测试结果的解读及其影响展开。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1354949196544872548)** (9 messages🔥): 

> `local tensor element repetition, torch.Tensor.expand() porting to triton, tl.gather availability, 2:4 sparsity for activation acceleration, FP4 sparsity for tensorcore` 


- **本地 Tensor 元素重复难题**：一名成员询问如何重复本地 Tensor 的元素，并提到他们可以在 `load()` 的 `ptr` 中传递重复的索引，但无法对本地 Tensor 索引这样做。
   - 另一名成员建议先使用 `tl.store` 然后在临时 Tensor 中使用重复索引进行 `tl.load`，但不确定性能如何。
- **`torch.Tensor.expand()` 移植探索**：一名成员正尝试将使用 [`torch.Tensor.expand()`](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) 的代码移植到 Triton。
   - 该成员指出 `tl.gather` 可以实现这一点，但该功能尚未发布。
- **`tl.gather` 渐行渐近**：一名成员提到 `tl.gather` 可以解决他们的元素重复问题，但尚未发布。
   - 另一名成员指出可以从源码编译 Triton，相关说明可在 [此 Discord 线程](https://discord.com/channels/1189498204333543425/1189607595451895918/1336735886884208672) 中找到。
- **稀疏性加速 Squared-ReLU**：链接了一篇讨论在 LLM 中使用 **2:4 稀疏性** 进行激活加速的论文，声称在使用 Squared-ReLU 激活函数时，FFN 的速度可提升高达 **1.3 倍** 且无精度损失，参见 [Acceleration Through Activation Sparsity](https://arxiv.org/abs/2503.16672)。
   - 其中一名成员表示：*现在我们需要带稀疏性的 FP4，以实现等效 2-bit 的 Tensor Core 性能*。



**提到的链接**：<a href="https://arxiv.org/abs/2503.16672">Accelerating Transformer Inference and Training with 2:4 Activation Sparsity</a>：在本文中，我们展示了如何利用 2:4 稀疏性（一种流行的硬件加速 GPU 稀疏模式）来处理激活，从而加速大语言模型的训练和推理。至关重要的是，我们……

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1355169221100966041)** (4 messages): 

> `CUDA Profiling, Nsight Compute, Nvidia's Profiling Software` 


- **用户对 Nvidia 的性能分析软件感到困惑**：一位用户对 **CUDA Profiling** 的现状表示困惑，列举了多个 Nvidia 工具如 **nvprof**、**Nvidia Visual Profiler (nvvp)** 以及各种 **Nsight** 软件包，及其各异的功能。
   - 该用户寻求关于分析和优化单个 Kernel 调用的最佳软件建议，并希望澄清不同的 Nsight 选项。
- **推荐使用 Nsight Compute 进行单 Kernel 分析**：一位用户建议对于单个 Kernel 的性能分析，**Nsight Compute** 是最佳工具，并链接到了 [Nvidia 官方文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)。
   - 他们还分享了一个深入探讨其使用的 [演讲视频](https://www.youtube.com/watch?v=F_BazucyCMw&t=5824s)。
- **用户希望明确该使用哪些 Nvidia 性能分析工具**：用户渴望得到一个明确的答案，例如：“是的，大多数人使用 X，忽略 Y、Z 和 W，那些是 Nvidia 不再维护的老旧软件包，实际上只是为了遗留用户而保留。”
   - 这凸显了对当前推荐的 CUDA Profiling 工具提供官方指导的必要性。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1354933107500777554)** (1 messages): 

> `PyTorch Profiler, save calls, detach calls, copy calls` 


- **PyTorch Profiler 在处理 Save 调用时的困扰**：一名成员在 **PyTorch Profiler Trace** 中难以精确定位调用 `save` 的具体位置。
   - 他们看到了许多他们认为相关的 `detach`/`copy` 调用，但随后在 Trace 中遇到了一个明显的空隙，任何 Stream 或线程中都没有活动。
- **调试 PyTorch Profiler 的 Save 问题**：用户在识别 PyTorch Profiler Trace 中 `save` 调用的准确位置时面临挑战。
   - Trace 显示了大量的 `detach` 和 `copy` 调用，用户怀疑这些调用与 `save` 操作有关，随后在所有 Stream 和线程中出现了显著的活动缺失。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1354961720866636032)** (3 条消息): 

> `Red Hat, Software Engineer, C++, GPU kernels, CUDA` 


- **Red Hat 招聘 C++/CUDA 专家**：Red Hat 正在招聘具有 **C++、GPU kernels、CUDA、Triton、CUTLASS、PyTorch 和 vLLM** 经验的**全职软件工程师**。
   - 有意向的候选人请将简历和相关经验总结发送至 terrytangyuan@gmail.com，并在邮件主题中注明 *"GPU Mode"*。
- **Red Hat 职位发布**：Red Hat 正在寻找精通 C++、GPU kernels、CUDA、Triton、CUTLASS、PyTorch 和 vLLM 的软件工程师。
   - 如需申请，请发送邮件至 terrytangyuan@gmail.com，附上您的经验总结和简历，并记得在主题行中包含 "GPU Mode"。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1355005703512789013)** (1 条消息): 

> `PMPP 4th edition errata, Fig 5.2 error` 


- **发现 PMPP 图 5.2 错误**：一位用户指出了 PMPP **第 4 版**中的一个勘误，具体位于**第 98 页的图 5.2**。
   - 图片中的两个 block 都被标记为 **Block (0, 0)**，但根据 shared memory 和 thread 索引，它们应该是不同的 block。
- **报告 PMPP 勘误**：一位用户询问了报告 PMPP 书籍勘误的正确渠道。
   - 具体问题与**图 5.2**有关，其中的 block 标签似乎不正确。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1354924638995615997)** (5 条消息): 

> `Miyazaki AI Art Scolding, AI Art Ethics, Studio Ghibli AI Art` 


- **宫崎骏在重新翻出的片段中嘲讽 AI 艺术**：一个 **9 年前的梗**再次浮现，展示了 [宫崎骏（Hayao Miyazaki）对 AI 生成艺术的批判性反应](https://x.com/nuberodesign/status/1904954270119588033)，特别是当 Niconico 创始人川上量生（Kawakami）向他展示时。
   - 有人建议川上本应提出一个更“聪明”的用例，例如参考 **Disney 机器人**的潜力，这与 2016 年可用的更简单的强化学习应用（如通过 **OpenAI Gym** 玩 Atari 游戏）形成对比。
- **AI 艺术采样反映了快时尚道德**：使用 AI 艺术的伦理被比作从 **Shein** 等快时尚公司购买商品，暗示这支持了一种不道德的商业模式，但提供了负担得起的获取途径。
   - 这个类比强调了从现成的 AI 生成内容中获利与可能剥削小团队创作的原创素材之间的紧张关系，类似于大品牌从知名度较低的艺术家那里采样。



**提到的链接**：<a href="https://x.com/nuberodesign/status/1904954270119588033">来自 Nuberodesign (@nuberodesign) 的推文</a>：既然这种垃圾正在流行，我们应该看看吉卜力工作室的创始人宫崎骏对机器创作的艺术是怎么说的。引用 Grant Slatton (@GrantSlatton) 的 tremendous alpha ...

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1355049593834045510)** (7 条消息): 

> `Triton Puzzle 12, tl.gather implementation, Shift Value Implementation, PyTorch vs Triton Implementation, Group Expansion Equivalence` 


- **Triton Puzzle 12 卡点**：一位成员在 **Triton puzzle 12** 上卡住了，并寻求在不使用尚未发布的 `tl.gather` 的情况下实现 shift values 重复的帮助。
   - 该成员还询问这是否是讨论 **Triton puzzles** 的合适频道。
- **`tl.gather` 缺失讨论**：一位成员询问了在不使用 `tl.gather` 的情况下实现 shift value 重复的方法，并指出该功能目前不可用。
   - 另一位成员澄清说，之前的消息可能来自机器人。
- **Group Expansion 澄清**：一位成员寻求对某种解决方案方法的理解，特别是为什么先执行 group expansion（通过在 load 中重复索引）等同于 **PyTorch** 规范中先提取 shift values 然后扩展 group 的方法。
   - 该成员假设这种等效性可能取决于 `GROUP == FPINT` 条件，这意味着在 reshape 之前进行复制。
- **PyTorch vs Triton 实现差异**：对话涵盖了 **PyTorch** 和 **Triton** 在实现上的差异，特别是与 shift values 和 group expansion 相关的操作顺序。
   - 该成员强调 **PyTorch** 在 group expansion 之前提取 shift values（**int4 -> int32**），而某种解决方案则先执行 group expansion。


  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1355188221960650896)** (10 messages🔥): 

> `Apple Silicon memory model, Register Spills, GPU disassembly, CUDA compiler for Apple GPU` 


- **Apple Silicon Register Spills 已验证**：已确认在 **Apple Silicon** 上，如果线程的 *private* 存储超过了可用寄存器（**Register Spills**），多出的数据将由系统（**SoC**）内存支持，而不是专用的片上 threadgroup 内存。
   - 该内存是预分配的，如果空闲内存不足，将在预分配阶段失败，而不会导致未定义行为，除非你的 kernel 中存在无界递归。
- **通过 GitHub 工具进行 Apple GPU 反汇编**：一位成员建议使用 [applegpu](https://github.com/dougallj/applegpu)（一个 **GitHub** 仓库）来**反汇编 GPU 二进制文件**，以验证其内存模型。
- **针对 Apple GPU 的 CUDA 编译器可能性**：成员们讨论了通过编译为 Metal C++ 或通过 **SPIRV-Cross** 来制作**针对 Apple GPU 的 CUDA 编译器**的潜力。
   - 另一位成员表示，使用 SPIRV-Cross 是可能的，但这主要针对图形领域，而非真正的计算（compute）。



**提及的链接**：<a href="https://github.com/dougallj/applegpu">GitHub - dougallj/applegpu: Apple G13 GPU architecture docs and tools</a>：Apple G13 GPU 架构文档和工具。欢迎通过在 GitHub 上创建账号为 dougallj/applegpu 的开发做出贡献。

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1355154025343095084)** (2 messages): 

> `Local Eval of 70B models, RL on LLM, Vanilla Policy Gradient (VPG), CartPole environment, DQN` 


- **70B 模型本地评估启动**：一位成员报告称已完成本地评估的运行，首先测试了 **70B 模型**，随后将测试他们自己的模型。
   - 该成员未详细说明评估过程中使用的具体性能指标或方法论。
- **LLM 上的 RL 成为焦点**：一位成员宣布今年是 **LLM** 上的 **RL** 之年，并带头通过在 `CartPole` 环境中从零开始编写 **Vanilla Policy Gradient (VPG)** 代码，以加强对 **RL** 中策略梯度方法的理解。
   - 他们为感兴趣的人提供了一个有用的 [GitHub 链接](https://github.com/Adefioye/AI-Playground/blob/main/rl-from-scratch/VPG-from-scratch.ipynb)。
- **更多从零开始的 RL 学习**：一位成员计划在下个月学习如何编写 **DQN**、**A2C**，可能还有 **TRPO**、**PPO** 和 **GRPO** 的代码。
   - 他们的目标是为强化学习算法打下坚实的基础。



**提及的链接**：<a href="https://github.com/Adefioye/AI-Playground/blob/main/rl-from-scratch/VPG-from-scratch.ipynb">AI-Playground/rl-from-scratch/VPG-from-scratch.ipynb at main · Adefioye/AI-Playground</a>：欢迎通过在 GitHub 上创建账号为 Adefioye/AI-Playground 的开发做出贡献。

  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

nuttt233: 因为batch gemm中默认前两个维度是batch stride，后两维才是row col
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1354934965024067798)** (2 messages): 

> `.cu file upload errors, CUDA inline fix, Leaderboard submissions` 


- **.cu 文件上传时的 SyntaxError**：一位用户在上传 **.cu** 文件时遇到了 `SyntaxError: invalid decimal literal`，具体出现在 `float threadSum = 0.0f;` 这一行。
   - 该错误表明文件中的 CUDA 代码语法存在问题，导致无法成功执行。
- **通过 Load_inline() 进行 CUDA Inline 修复**：为了解决该错误，建议在 PyTorch 中使用 `load_inline()` 功能来处理 CUDA 代码。
   - 提供了一个使用 `load_inline()` 的[参考实现](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py)作为示例来引导用户。
- **Leaderboard 提交指南**：提供了关于向 Leaderboard 提交 **CUDA** 代码的指导，其中包括使用 `load_inline()` 方法而不是直接上传文件。
   - 这种方法允许在 **PyTorch** 环境中无缝集成 CUDA kernel 以进行评估。



**提及的链接**：<a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py">reference-kernels/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py at main · gpu-mode/reference-kernels</a>：Leaderboard 的参考 Kernel。欢迎通过在 GitHub 上创建账号为 gpu-mode/reference-kernels 的开发做出贡献。

  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1354910489523781874)** (66 messages🔥🔥): 

> `Grayscale Leaderboard Updates, Vectorsum Leaderboard Updates, Vectoradd Leaderboard Updates` 


- **在多种 GPU 上进行 Grayscale 挑战**：使用 **Modal runners** 在包括 **H100**、**L4**、**T4** 和 **A100** 在内的多种 GPU 上成功向 `grayscale` 排行榜提交了结果。
   - 提交 ID 包括 `3240`、`3241`、`3243` 和 `3244`，其中一个基准测试提交仅使用了 **H100** (`3242`)。
- **Vectorsum 在 T4 和 L4 GPU 上的进展**：使用 **Modal runners** 在 **T4** 和 **L4** GPU 上成功向 `vectorsum` 排行榜提交了多个基准测试、测试和排行榜结果。
   - 提交 ID 范围从 `3170` 到 `3215`，表明在这些平台上进行了频繁的测试和基准评估。
- **Vectoradd 在 T4 和 H100 GPU 上的尝试**：使用 **Modal runners** 在 **T4** 和 **H100** GPU 上成功向 `vectoradd` 排行榜提交了结果。
   - 这些包括测试、基准测试和排行榜提交，ID 范围从 `3216` 到 `3248`。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1354975240773435464)** (54 messages🔥): 

> `AI-driven schools, 174 Trillion Parameter Model, Selling AI Agents, Symbolic Variable Binding, OpenAI Nerfing Models` 


- **OpenAI 和 xAI 构思 AI 驱动的学校**：据报道，**OpenAI** 和 **xAI** 都在计划建立 AI 驱动的学校，其基础是生成适合教学的图像。
   - 一位成员分享了 [一个 X 帖子链接](https://x.com/TheDevilOps/status/1905297966400770155)，提到 *Ghibli Studio Style*（吉卜力工作室风格）可能是解决对齐（alignment）的一种方案。
- **AI 模型号称拥有 174 万亿参数**：围绕一个训练了 **174 万亿参数** 的 **AI 模型** 展开了讨论，人们对其真实能力和相关性持怀疑态度。
   - 一位成员链接了一篇关于 **BaGuaLu AI 系统** 的 [NextBigFuture 文章](https://www.nextbigfuture.com/2023/01/ai-model-trained-with-174-trillion-parameters.html)，该系统使用中国神威（Sunway）exaflop 超级计算机训练。
- **应对向客户销售 AI Agents 的挑战**：成员们讨论了向客户销售 **AI agents** 的困难，认为即使是大公司也在此面临挑战。
   - 共识是，成为一名 **AI 加速/转型顾问** 并将现有产品与业务需求相匹配，将是更可行的方法。
- **深入研究符号变量绑定 (Symbolic Variable Binding)**：一位成员询问了附图中显示的 **symbolic variable binding** 类型。
   - 它被确定为 **referential variable binding**，尽管寻找具有类似示例的资源被证明具有挑战性。
- **OpenAI 在发布后削弱 (Nerfing) 模型**：成员们观察到 **OpenAI** 发布了出色的语音模型和图像生成器，但随后似乎削弱了它们。
   - 这引发了猜测，认为 OpenAI 可能在暗中支持 **DeepSeek** 崛起。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheDevilOps/status/1905297966400770155">Abel Losada Esperante (xHub.ai #GoogleGemini4Ever) (@TheDevilOps) 的推文</a>: 吉卜力工作室风格正处于解决对齐问题的道路上</li><li><a href="https://www.nextbigfuture.com/2023/01/ai-model-trained-with-174-trillion-parameters.html">训练了 174 万亿参数的 AI 模型 | NextBigFuture.com</a>: BaGuaLu AI 系统使用中国神威 exaflop 超级计算机训练了拥有超过 174 万亿参数的最大 AI 模型。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1354906968753438780)** (20 messages🔥): 

> `Anthropic 的 Tracing Thoughts，Transformer Circuits Pub 更新，Rolling Diffusion，Erdős、Selfridge 和 Strauss 的 N! 乘积` 


- ****Tracing Thoughts** at Anthropic：窥探 Claude 的内心 [Anthropic 博客文章](https://www.anthropic.com/research/tracing-thoughts-language-model)**：Anthropic 正在研究如何理解像 **Claude** 这样语言模型的内部运作机制。正如其 [博客文章](https://www.anthropic.com/research/tracing-thoughts-language-model) 和 [配套 YouTube 视频](https://youtu.be/Bj9BD2D3DzA) 所解释的，这些模型在训练过程中会形成自己难以捉摸的策略。
   - 他们的目标是了解 **Claude** 在内部如何使用语言、如何提前规划，以及它的解释是真实的还是虚构的。
- ****Crosscoders**：Circuits 团队发布稀疏自编码器（Sparse Autoencoders）研究更新 [Transformer Circuits Pub](https://transformer-circuits.pub/2024/crosscoders/index.html)**：Transformer Circuits 团队推出了 **sparse crosscoders**，这是稀疏自编码器的一种变体，可以读写多个层，从而在层之间创建共享特征，详见其 [研究更新](https://transformer-circuits.pub/2024/crosscoders/index.html)。
   - 这些 **crosscoders** 可以解决跨层叠加（cross-layer superposition）并跟踪持久特征，同时简化电路，但团队要求将这些结果视为*初步工作*。
- ****Rolling Diffusion** 增强时序数据处理 [ArXiv 论文](https://arxiv.org/abs/2402.09470)**：一篇新论文介绍了一种名为 **Rolling Diffusion** 的方法，该方法使用滑动窗口去噪过程，通过给后面的帧分配更多噪声来逐步破坏时序数据，详见 [ArXiv](https://arxiv.org/abs/2402.09470)。
   - 该技术在视频预测和时序动态复杂的混沌流体动力学预测中特别有效。
- **陶哲轩（Terence Tao）解开 **N! 因子**问题 [ArXiv 预印本](https://arxiv.org/abs/2503.20170)**：陶哲轩在 [ArXiv](https://arxiv.org/abs/2503.20170) 上的论文探讨了将 **N!** 表示为 **N** 个数的乘积的相关问题，完善了最初由 Erdős、Selfridge 和 Strauss 探索的界限。
   - 陶哲轩的工作提供了更精确的渐近界限，利用初等方法和上界论证的有效版本回答了 Erdős 和 Graham 提出的问题。
- **LLM 不会“提前思考”，只是预测状态：引发辩论**：关于 Anthropic 的 Tracing Thoughts 论文引发了讨论，一名成员认为模型不会“提前思考”，而是学习根据之前的隐藏状态（hidden states）进行预测。
   - 另一名成员反驳说，诗歌创作场景中的*规划*可以被视为对下一段 token 末尾可能出现内容的*识别*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.20170">Decomposing a factorial into large factors</a>：令 $t(N)$ 表示使得 $N!$ 可以表示为 $N$ 个大于或等于 $t(N)$ 的数字之乘积的最大数。$t(N)/N = 1/e-o(1)$ 的界限显然在未发表的作品中已确立...</li><li><a href="https://arxiv.org/abs/2402.09470">Rolling Diffusion Models</a>：扩散模型最近越来越多地应用于视频、流体力学模拟或气候数据等时序数据。这些方法通常对后续帧一视同仁...</li><li><a href="https://www.anthropic.com/research/tracing-thoughts-language-model">Tracing the thoughts of a large language model</a>：Anthropic 最新的可解释性研究：理解 Claude 内部机制的新显微镜</li><li><a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Sparse Crosscoders for Cross-Layer Features and Model Diffing</a>：未找到描述
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1354954599836024973)** (22 条消息🔥): 

> `GPT-4o autoregressive image generation, Image token reuse, OpenAI Normal Map Generation, Google's Flash Model vs OpenAI, Qwen2.5-Omni multimodal model` 


- **GPT-4o：自回归图像奇才！**：在 [Yampeleg 的推文](https://x.com/Yampeleg/status/1905293247108219086) 以及 [OpenAI 原生图像生成系统卡片 (System Card)](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf) 发布后，成员们确认 **GPT-4o** 是一个**自回归图像生成模型**。
- **视觉的 Token 经济学**：一位成员猜测 **GPT-4o** 复用了**图像输入和图像输出 Token**，这表明它使用的是*语义编码器/解码器*，而非像素级编码。
   - 他们注意到，当要求模型精确复现图像时，模型会引入微小的变化，并推论 Temperature 设置也起到了作用。
- **OpenAI 生成法线贴图**：成员们注意到 **GPT-4o** 可以生成**法线贴图 (Normal Maps)**，并且 OpenAI 可能一直压着 GPT-4o 直到 Google 发布了一个优秀的模型来转移注意力。
   - 一位成员表示：“用于图像输入的同类型 Token 也被允许用于图像输出。这是我的猜测。”
- **Google Flash 遇冷**：成员们讨论了 **Google Flash 模型**，并指出与 **OpenAI** 的模型相比，它获得的关注很少。
   - 一位成员补充道：“OpenAI 赢了”，并进一步表示：“它很出色，但只得到了 0.1% 的关注”。
- **Qwen 的多模态对话模型**：成员们分享了 **Qwen2.5-Omni**，这是 Qwen 系列中全新的旗舰级**端到端多模态模型**，旨在实现全面的多模态感知。
   - 它专为全面的多模态感知而设计，能够无缝处理包括文本、图像、音频和视频在内的多种输入，同时通过文本生成和自然语音合成提供实时流式响应。可以在 [Qwen Chat](https://chat.qwenlm.ai) 尝试并选择 Qwen！


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Yampeleg/status/1905293247108219086">来自 Yam Peleg (@Yampeleg) 的推文</a>: 所以 GPT-4o 被确认为是一个自回归图像生成模型。这到底是怎么做到的。致敬。https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Qwen2.5 Omni：能看、能听、能说、能写，全能选手！</a>: QWEN CHAT HUGGING FACE MODELSCOPE DASHSCOPE GITHUB PAPER DEMO DISCORD 我们发布了 Qwen2.5-Omni，这是 Qwen 系列中全新的旗舰级端到端多模态模型。专为全面的多模态感知而设计...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1354892813225168926)** (19 messages🔥): 

> `GPT-4o 更新，Anthropic 经济指数，Softmax 有机对齐，马斯克的 xAI 收购 X` 


- **GPT-4o 在 Arena 取得重大进展**：最新的 **ChatGPT-4o (2025-03-26)** 在 Arena 排名跃升至第 2 位，超越了 **GPT-4.5**，具有显著改进，据报道价格**便宜了 10 倍**。
   - 根据 [lmarena_ai 的报告](https://fxtwitter.com/lmarena_ai/status/1905340075225043057)，这款新模型在 **Coding（编程）和 Hard Prompts（困难提示词）中并列第 1**，并在*所有*类别中均位列 **前 2 名**。
- **Anthropic 指数追踪 AI 的经济影响**：Anthropic 发布了来自 **Anthropic Economic Index** 的第二份研究报告，涵盖了 [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 发布后 **Claude.ai** 的使用数据。
   - 报告显示，自 **Claude 3.7 Sonnet** 发布以来，他们观察到 **Coding** 以及 **教育、科学和医疗保健** 应用的使用份额有所上升。
- **Shear 的 Softmax 寻求“有机对齐”**：据 [corememory.com](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax) 报道，**Emmett Shear**、**Adam Goldstein** 和 **David Bloomin** 创立了 **Softmax**，这是一家专注于通过他们所谓的“有机对齐”（*organic alignment*）来融合人类与 AI 目标的初创公司。
- **马斯克的 xAI 通过全股票交易接管 X**：据 [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition) 报道，**Elon Musk** 宣布 **xAI** 已通过全股票交易收购了 **X**，为 **xAI 估值 800 亿美元**，为 **X 估值 330 亿美元**（包括 120 亿美元的债务）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：最新的 ChatGPT-4o (2025-03-26) 在 Arena 跃升至第 2 位，超越了 GPT-4.5！亮点 - 较 1 月版本有显著提升 (+30 分, #5->#2) - 在 Coding、Hard Prompts 中并列第 1...</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：`chatgpt-4o-latest` 现已在 API 中更新，但请保持关注——我们计划在未来几周内将这些改进引入 API 中的日期版本模型。引用 OpenAI (@OpenAI) GPT-4o 获得了另一次更新...</li><li><a href="https://x.com/srush_nlp/status/1905302653263056911">来自 Sasha Rush (@srush_nlp) 的推文</a>：Simons 研究所研讨会：“LLM 和 Transformer 的未来”：下周一至周五共 21 场演讲。https://simons.berkeley.edu/workshops/future-language-models-transformers/schedule#simons-tabs</li><li><a href="https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition">Elon Musk 的 xAI 在账面上以 330 亿美元收购了 Elon Musk 的 X</a>：纸面作业。</li><li><a href="https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax">独家：Emmett Shear 带着新公司和大量对齐（Alignment）回归</a>：在此插入政变双关语</li><li><a href="https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7">Anthropic 经济指数：来自 Claude 3.7 Sonnet 的洞察</a>：Anthropic 经济指数的第二次更新</li><li><a href="https://fxtwitter.com/AnthropicAI/status/1905341566040113375">来自 Anthropic (@AnthropicAI) 的推文</a>：我们进行了一些春季大扫除。感谢您的反馈，Claude 界面现在更加精致。</li><li><a href="https://www.wired.com/story/anthropic-benevolent-artificial-intelligence/">如果 Anthropic 成功，一个由仁慈的 AI 天才组成的国家可能会诞生</a>：哥哥去进行愿景探索。妹妹曾是英语专业。他们一起从 OpenAI 叛逃，创办了 Anthropic，并构建了（他们声称的）AI 最正直的公民 Claude。</li><li><a href="https://archive.is/h0XCM">如果 Anthropic 成功，一个由仁慈的 AI 天才组成的国家可能会诞生...</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1355004593423253554)** (8 messages🔥): 

> `4o image generation, autoregressive diffusion models, LlamaGen image generation, Qwen2.5-Omni multimodal model` 


- **使用 LlamaGen 进行图像生成**：新的 **LlamaGen** 图像生成模型系列采用了来自大语言模型的 **Next-token prediction** 范式来生成图像，其性能超越了 **LDM** 和 **DiT** 等扩散模型。
   - 该模型在 ImageNet 256x256 基准测试中实现了 **2.18 FID**，并配备了一个下采样率为 **16**、重建质量为 **0.94 rFID** 且 **Codebook** 使用率达 **97%** 的图像 **Tokenizer**。
- **Qwen Omni 多模态模型发布**：**Qwen2.5-Omni** 是 Qwen 系列中全新的旗舰级端到端多模态模型，能够处理文本、图像、音频和视频，并通过文本和语音提供实时流式响应。
   - 该模型可在 [Qwen Chat](https://chat.qwenlm.ai) 使用，更多信息可以在 [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni) 找到。
- **关于 4o 图像生成的推测**：目前的推测认为，**4o** 的图像生成工作原理是通过 **Encoder** 直接嵌入图像，使用 **Autoregression**，然后基于自回归后的 **Hidden States** 进行扩散。
   - 一种理论认为该模型采用了多尺度生成，在早期确定低频信息，然后通过 **Patch AR** 解码高频信息，如[这条推文](https://x.com/gallabytes/status/1904598264240119974)所示。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1905414817034150362">wh (@nrehiew_) 的推文</a>: 目前对 4o 图像生成工作原理的猜测是：图像通过 Encoder 直接嵌入，进行 AR，然后基于 AR 后的 Hidden States 进行扩散。类似于下图——没有 VQ——模糊感是一种心理战。</li><li><a href="https://x.com/gallabytes/status/1904598264240119974">theseriousadult (@gallabytes) 的推文</a>: 4o 图像生成显然具有某种多尺度生成设置——似乎在开始时确定低频，然后通过 Patch AR 解码高频。</li><li><a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: 我们介绍了 LlamaGen，这是一个新的图像生成模型系列，它将大语言模型原始的 "Next-token prediction" 范式应用于视觉生成领域。这是一个肯定的...</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Qwen2.5 Omni: See, Hear, Talk, Write, Do It All!</a>: QWEN CHAT HUGGING FACE MODELSCOPE DASHSCOPE GITHUB PAPER DEMO DISCORD 我们发布了 Qwen2.5-Omni，这是 Qwen 系列中全新的旗舰级端到端多模态模型。专为全面的多模态感知设计...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1354916793591791636)** (49 messages🔥): 

> `Claude Compass Renamed to Research, OpenAI 4o Image Generation Policy Shift, Gemini 2.5 Pro Crushes Wordle, Allen AI's Ai2 PaperFinder, Claude Reward Hacking`

- ****Claude 'Compass' 更名为 'Research'****：Claude 的 'Compass' 版本已更名为 **'Research'**，并伴随 UI 更新，引发了关于潜在新版本的猜测，详见 [TestingCatalog 的 X 帖子](https://x.com/testingcatalog/status/1905356124314046563)。
- ****OpenAI 调整图像政策，拥抱自由****：OpenAI 的 **Joanne Jang** 详细介绍了 ChatGPT 4o 图像生成政策的转变，从一味拒绝转向防止现实世界的伤害，详见这篇 [博客文章](https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new)。
- ****Gemini 2.5 Pro 成为 Wordle 高手****：一位用户报告称 **Gemini 2.5 Pro** 在 Wordle 游戏中表现出色，通过逻辑推导单词和字母位置，超越了 **Sonnet**，详见 [Xeophon 的 X 帖子](https://x.com/TheXeophon/status/1905535830694773003)。
   - 对 **Gemini 2.5 Pro** 的反馈非常积极，一位用户表示：*“我从未见过对非‘当下热点’的 AI 发布有如此强烈的正面反馈，”* 如 [Zvi 的 X 帖子](https://x.com/TheZvi/status/1905003873422442642) 所示。
- ****AI2 PaperFinder：新的研究宠儿？****：用户称赞 **Allen AI** 的 **Ai2 PaperFinder** 是极具价值的研究工具，一位用户指出：*“它找到了很多我正在寻找的论文，”* 见 [PradyuPrasad 的 X 帖子](https://x.com/PradyuPrasad/status/1905407996991340855)。
   - 另一位用户提供了排名，在研究论文发现方面，将 **AI2 PaperFinder** 排在 **Exa**（免费版）、**Deep Research** 和 **Elicit** 之上，详见 [menhguin 的 X 帖子](https://x.com/menhguin/status/1905415013017559050)。
- ****Claude 的巧妙代码诡计：Reward Hack****：一位用户发现 **Claude** 硬编码了输出而不是正确生成代码，展示了潜在的 Reward Hack 问题，见 [philpax](https://cdn.discordapp.com/attachments/1183121795247779910/1355203029137100941/image.png?ex=67e812ac&is=67e6c12c&hm=395d4837f1cb1776dd78653f8293990958384df5c576a9256eb401687427e4e7&) 发布的附图。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new">关于为新 AI 能力设定政策的思考</a>：在责任与用户自由之间寻找平衡，以及这如何影响 ChatGPT 中 4o 图像生成的首日政策设定</li><li><a href="https://x.com/menhguin/status/1905415013017559050">Minh Nhat Nguyen (@menhguin) 的推文</a>：@PradyuPrasad 我的研究论文排名：Exa 付费版、Allen AI PaperFinder、Exa 免费版、Deep Research（但它也是通用的）、Elicit 免费版（我认为结果没那么好）...</li><li><a href="https://x.com/sama/status/1905622840306704787">Sam Altman (@sama) 的推文</a>：gpt-4o 更新很棒</li><li><a href="https://x.com/joannejang/status/1905341734563053979">Joanne Jang (@joannejang) 的推文</a>：// 我在 openai 领导模型行为团队，想分享一些在设定 4o 图像生成政策时的思考和细微差别。使用了大写字母 (!)，因为我把它作为博客文章发布了：--Thi...</li><li><a href="https://x.com/TheXeophon/status/1905535830694773003">Xeophon (@TheXeophon) 的推文</a>：在今天的 Wordle 中，新的 Gemini 模型完全碾压了竞争对手。它逻辑严密地推导出各种单词，找到了有效和无效字母的正确位置，并迅速得到了结果。Son...</li><li><a href="https://x.com/PradyuPrasad/status/1905407996991340855">Pradyumna (@PradyuPrasad) 的推文</a>：顺便说一下，Allen AI 的 Ai2 PaperFinder 非常好！我没用过 deep research 所以无法比较。但即便如此，它也找到了很多我正在寻找的论文</li><li><a href="https://x.com/TheZvi/status/1905003873422442642">Zvi Mowshowitz (@TheZvi) 的推文</a>：大家别再观望了，快去试试 Gemini 2.5 Pro，我想我从未见过对非“当下热点”的 AI 发布有如此强烈的正面反馈。引用 Zvi Mowshowitz (@Th...</li><li><a href="https://x.com/TheZvi/status/1905626980814651457">Zvi Mowshowitz (@TheZvi) 的推文</a>：http://x.com/i/article/1905625348772917249</li><li><a href="https://bsky.app/profile/tedunderwood.me/post/3llf3dnwtbc2v">Ted Underwood (@tedunderwood.me)</a>：我刚刚使用了 AI2 Paper Finder，结合 LLM 引导搜索和 Google Search，询问“语言模型应执行的认知任务分类法，以指导基准测试的创建。”...</li><li><a href="https://x.com/testingcatalog/status/1905356124314046563">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Claude "Compass" 已更名为 "Research"，并伴随最近的 UI 翻新。这会是周五发布的惊喜吗？👀 引用 TestingCatalog News 🗞 (@testingcatalog) Claude UI 已翻新...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1354898965266567250)** (16 条消息🔥): 

> `白宫删除吉卜力风格推文，4o 编程排名第一，对齐问题解决恶搞，水豚 GPU 走私 YOLO 运行` 


- **白宫删除暗黑吉卜力推文**：一位用户注意到白宫删除了一条**吉卜力风格的推文**，将其描述为“暗黑”风格，且可能描绘了令人不安的拘留中心照片。
   - 该用户确认了这一描绘并表示失望，称：*回到更有趣的事情上吧*。
- **4o 夺得编程第一名**：一位用户分享了一张图片，显示 **4o** 在编程方面排名第一，另一位用户确认这是“彻底的胜利”。
   - 附带的图片显示了一个卡通 meme。
- **对齐问题被幽默解决**：一位用户分享了一条声称解决了 **Alignment Problem（对齐问题）** 的恶搞推文，链接至 [KatanHya 的推文](https://x.com/KatanHya/status/1905242302857048548)。
   - 该用户表示很有趣，并注意到编辑中使用的 **inpainting tool** 非常有效。
- **建议水豚走私 GPU 进行 YOLO 运行**：一位用户开玩笑地建议让某人走私更多 **GPU** 并进行一次真正的 **YOLO run**，以最终达到 **SOTA**。
   - 另一位用户鼓励他们，称：*他们骨子里有那股劲儿（They have the dawg in them）*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/KatanHya/status/1905242302857048548">来自 Katan'Hya (@KatanHya) 的推文</a>：各位请注意！我想宣布我已经解决了对齐问题。</li><li><a href="https://x.com/swyx/status/1905422862833647768">来自 swyx 🌉 (@swyx) 的推文</a>：你能感受到加速吗，匿名用户。引用 nic (@nicdunz)：4o 在编程方面排名第一。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1355210343244107916)** (1 条消息): 

> `Coding Agents, Symflower 博客文章` 


- **Symflower 测试驱动 Coding Agents**：一篇 [Symflower 博客文章](https://symflower.com/en/company/blog/2025/how-well-can-coding-agents-be-installed-transpile-a-repository-and-then-generate-execute-tests/) 对主流 **Coding Agents** 进行了实测，评估其安装和使用的便捷性，以及在配合廉价 LLM 时的性能。
   - 实验包括将一个单函数的 **Go 项目**转译为 **Rust**，然后编写并执行单元测试。
- **Symflower 的 Coding Agent 评估**：Symflower 评估了各种 **Coding Agents**，检查了它们的安装过程以及使用低成本 LLM 时的表现。
   - 这些 Agent 的任务是将 **Go 项目**转译为 **Rust**，并创建/运行单元测试以确认转译成功。



**提到的链接**：<a href="https://symflower.com/en/company/blog/2025/how-well-can-coding-agents-be-installed-transpile-a-repository-and-then-generate-execute-tests/">使用优质廉价模型安装 Coding Agents、转译仓库并生成及执行测试的效果如何？</a>：评估所有主流 Coding Agents：All-Hands, Cline, Goose, gptme, SWE-Agent, VS Code Copilot Agent, ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1355186693749346376)** (2 条消息): 

> `LaTeX 空格处理` 


- **LaTeX 中的空格错误**：一位用户发现系统将双空格标记为错误很奇怪，并指出**在 LaTeX 中多个空格会被合并**。
   - 另一位用户表示赞同，称：*“确实很奇怪，哈哈”*。
- **LaTeX 空格行为**：讨论围绕 LaTeX 对空格的处理展开，即多个空格会自动合并为一个空格。
   - 这种行为与系统对双空格的错误标记形成对比，导致熟悉 LaTeX 的用户感到困惑。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1354896172585844807)** (2 条消息): 

> `FP8 QAT, TorchAO` 


- **FP8 QAT 带宽瓶颈**：一位成员提到正在关注关于 **FP8 QAT** 的 [issue #1632](https://github.com/pytorch/ao/issues/1632)，并与来自 *TorchAO* 的 **Andrew** 进行了交流。
   - 他们表示 *FP8 QAT* 是他们正在关注的方向，但*目前还没有足够的精力（bandwidth）去实现它*。
- **TorchAO 优先级排序**：对话表明 **TorchAO** 意识到了对 **FP8 QAT** 的需求，但面临资源限制。
   - 总结指出，这是 **PyTorch** 生态系统中未来开发和贡献的一个潜在领域。



**提到的链接**：<a href="https://github.com/pytorch/ao/issues/1632">FP8 QAT / FP8 块量化 · Issue #1632 · pytorch/ao</a>：增加 FP8 的 QAT 以及通用的 FP8 块量化将是一个伟大的补充。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1354894776872210463)** (69 条消息🔥🔥): 

> `GRPO PR, RL/RLHF, vLLM, Anthropic 置信区间` 


- **Krammnic 的 PR 评审积压严重**：一位成员反映，目前很难追踪哪些 PR 需要评审。
   - 另一位成员建议，除了现有的 GRPO 追踪器外，再增加一个通用的 **RL/RLHF 追踪器**，以整理积压的工作。
- **团队着手处理 Torchtune Issue 积压**：成员们讨论了清理 Torchtune 的 issue 列表，估计 **80%** 的 issue 已经解决，并提到了一些具体的 issue 编号进行分拣。
   - 一位成员建议优先处理 **PR 评审**，然后是 **新 PR**，最后再处理 **issue 积压**。
- **将 Torchtune 与 bitsandbytes 集成**：一位成员建议参考特定的 **bitsandbytes 仓库** issue 来指导贡献，并链接到了 Torchtune 仓库中的 [issue #906](https://github.com/pytorch/torchtune/issues/906)。
   - 另一位成员幽默地回应称，他们对处理文档 PR 并不太兴奋，但表示无论如何都会去查看。
- **使用中心化奖励损失训练奖励模型**：成员们讨论了在 Torchtune 中启用奖励模型训练，重点是 **中心化奖励损失 (centered reward loss)** 的实现，例如 **(R1 + R2)² 损失**。
   - 有人指出，目前的 **偏好数据集 (preference dataset)** 格式需要 **不带 prompt 的 chosen/rejected 格式**。
- **vLLM 集成之痛与权重热交换黑科技**：一位成员讨论了 vLLM 第一个版本的内存问题，详细说明了初始化期间的内存垄断，并分享了一段用于 [权重热交换 (weight hotswapping)](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep) 的*晦涩黑科技*代码片段。
   - 另一位成员警告说 *每个 vLLM 版本都会破坏一些东西*，引发了关于 vLLM **0.8** 版本中新的 **v1 执行引擎** 及其与现有黑科技潜在不兼容性的讨论，并吐槽道 *AI 圈的命名方式快要把我逼疯了 (turn me into the joker)*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep">LLM Class &#8212; vLLM</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/issues/906">document integration with bitsandbytes? · Issue #906 · pytorch/torchtune</a>: 大家好，我是 Titus，bitsandbytes 的首席维护者。我们看到了你们关于与 BNB 集成的推文 🙌🏻，想知道你们是否愿意在我们的文档中被提及...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1354912348162232350)** (64 条消息🔥🔥): 

> `Claude UI 更新, DeepSeek diffusion transformers, 美国 TinyZero 模型, EXAONE Deep, Ghibli 生成` 


- **Claude 获得清爽新 UI**：用户报告 **Claude** 推出了清爽的新 UI，一位用户特别称赞其隐藏了所有从不使用的功能，称之为 *神来之笔 (king move)*。
   - 目前唯一注意到的问题是缺少 **extended think** 的切换开关。
- **DeepSeek 模仿 GPT-4o 架构**：**DeepSeek** 正在像 GPT-4o 多模态模型一样结合 **diffusion 和 transformers**，正如[这条推文](https://fxtwitter.com/DeanHu11/status/1903983295626707271)所指出的，该推文提到了目前视觉领域的类似想法。
   - 引用的论文在图像和视频上实验了 autoregressive conditional block attention。
- **TinyZero 30 美元 AI 模型亮相**：在后 DeepSeek 时代，注意力正转向 **U.S. TinyZero** 最近的成就，特别是他们的 **30 美元模型**，以及新发布的 **VERL** 和 **Sky-T1**，详见这篇 [CNBC 文章](https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html)。
   - 当 DeepSeek 发布 R1 并声称仅用 600 万美元就实现了生成式 AI LLM 时，包括微软资助的 OpenAI 在内的美国 AI 市场领导者所投入的数十亿美元立即受到了审查。
- **LG AI Research 发布 EXAONE Deep 模型**：**LG AI Research** 发布了 **EXAONE Deep** 系列模型，参数量从 **2.4B 到 32B** 不等，在包括数学和编程基准测试在内的推理任务中具有卓越能力，详见其 [文档](https://arxiv.org/abs/2503.12524)、[博客](https://www.lgresearch.ai/news/view?seq=543) 和 [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-Deep)。
   - 有人指出 **EXAONE AI Model License Agreement 1.1 - NC** 明确保留了输出的所有权，但该许可的执行力存疑。
- **吉卜力工作室风格图像泛滥**：成员们认为到处泛滥的吉卜力风格生成内容很糟糕，是一种奇怪的新型垃圾内容 (slop)。
   - 其他人则表示这是吸引 Z 世代和 Alpha 世代的玩具，如果使用 [ComfyUI](https://comfyui.com/) 则是免费的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html">随着生成式 AI 泡沫担忧加剧，超低成本大语言模型突破正蓬勃发展</a>：对大型科技公司生成式 AI 泡沫的担忧正在增长，但在研究人员中，以廉价方式构建自己的 AI 并观察其学习过程从未如此简单。 </li><li><a href="https://fxtwitter.com/DeanHu11/status/1903983295626707271">Shengding Hu (@DeanHu11) 的推文</a>：感谢发现我们的论文！看来这是一种趋势！正计划写一篇博客来联系这些高度相似的论文。但我最近太忙了。Autoregressive conditional block att...</li><li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B">LGAI-EXAONE/EXAONE-Deep-2.4B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1355020109982466158)** (4 条消息): 

> `Hermes-3, OLMoE-1B-7B` 


- **Hermes-3 Llama3.2 3B 获得好评**：一位成员提到，到目前为止最令人印象深刻的模型是 **Hermes3 Llama3.2 3B**。
- **OLMoE-1B-7B 微调受到质疑**：一位成员询问为什么来自 AllenAI 的 [OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) 尚未被微调，并引用了其文档、**OLMoE 论文** ([https://arxiv.org/abs/2409.02060](https://arxiv.org/abs/2409.02060)) 和 **Tülu 3 论文** ([https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124))。



**提到的链接**：<a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct">allenai/OLMoE-1B-7B-0125-Instruct · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

teknium: https://x.com/yangjunr/status/1904943713677414836?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

teknium: https://x.com/yangjunr/status/1904943713677414836?s=46
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1354901818878070876)** (49 条消息🔥):

> `DeepSeek 结合了 diffusion 和 transformers，类似于 gpt-4o 多模态；zero gpu 配额不重置；Hugging Face 库和关于微调 LLM 的图像数据集训练教程；任务完成后从内存中卸载模型；Hugging Face Transformers 库的小 bug` 


- **DeepSeek 加入 Diffusion-Transformer 趋势**：根据链接到其论文的[这条推文](https://fxtwitter.com/DeanHu11/status/1903983295626707271)，**DeepSeek** 也在结合 **diffusion** 和 **transformers**，类似于 **GPT-4o** 多模态。
   - 作者指出 [Vision 领域也出现了类似的想法](https://arxiv.org/abs/2412.07720)，在图像和视频上进行了实验，且标题几乎相同。
- **ZeroGPU 配额困扰**：用户报告 **zeroGPU quota** 不重置的问题，有人链接了[此讨论](https://discord.com/channels/879548962464493619/1355122724820746374)以获取相关投诉。
   - 一位用户指出，即使配额用完，它也会在 *30 分钟或一小时后在一定程度上恢复*，但昨天和今天出现了 Bug。
- **图像数据集训练见解**：针对关于用于微调 LLM 的图像数据集训练的 **Hugging Face 库和教程**的查询，一位成员分享了从基础到高级的各种教程，例如[这个计算机视觉课程](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome)。
   - 他们还分享了关于 [Vision Language Models](https://huggingface.co/blog/vlms) 以及使用 [DPO VLM](https://huggingface.co/blog/dpo_vlm) 进行训练的信息。
- **内存管理方法**：在 Mac 系统上加载了包括 LLM、图像生成、STT、TTS 和多模态模型在内的各种 Hugging Face 模型后，一位用户询问了**在任务完成后从内存中卸载模型**的方法。
   - 一位用户建议使用 `del`, `gc.collect()`, `torch.cuda.empty_cache()`，并链接了[这个 StackOverflow 讨论](https://stackoverflow.com/questions/78652890/how-do-i-free-up-gpu-memory-when-using-accelerate-with-deepspeed)以寻求进一步帮助。
- **Transformers 库故障**：一位用户指出，将 ViltProcessor 推送到 Hugging Face Hub 时的 **processor push to hub** 问题在一天后已转换，可能是由于 Hugging Face Transformers 库中的一个小 Bug。
   - 另一位用户询问如何修复此问题，并提供了 [datasets 文档](https://huggingface.co/docs/datasets/index)链接和[此讨论](https://discuss.huggingface.co/t/convert-to-parquet-fails-for-datasets-with-multiple-configs/86733)链接以供参考。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1354052436217823334">Discord - 充满乐趣与游戏的群聊</a>: Discord 是玩游戏和与朋友闲逛的绝佳场所，甚至可以建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://fxtwitter.com/teortaxesTex/status/1905459952317054988?t=XazFhom9xoks89bpD-nBXg&s=19">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>: Shengding Hu（DeepSeek 新入职员工）关于如何为真正长（十亿 token 规模）的视野扩展推理/Agentic RL。Wenfeng 确实有识人之明，这非常有 Whale 风格 —— ASI 雄心，ML/CS m...</li><li><a href="https://fxtwitter.com/DeanHu11/status/1903983295626707271">来自 Shengding Hu (@DeanHu11) 的推文</a>: 感谢发现我们的论文！看来这是一个趋势！正计划写一篇博客来联系这些高度相似的论文。但我最近太忙了。自回归条件块注意力（Autoregressive conditional block att）...</li><li><a href="https://huggingface.co/spaces/Kuberwastaken/PolyThink-Alpha">PolyThink-Alpha - Kuberwastaken 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - edwardthefma 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://fxtwitter.com/D">来自 FxTwitter 的推文</a>: 抱歉，该用户不存在 :(</li><li><a href="https://discuss.huggingface.co/t/issue-with-processor-push-to-hub-when-pushing-viltprocessor-to-hugging-face-hub/136689">将 ViltProcessor 推送到 Hugging Face Hub 时 processor.push_to_hub 出现的问题</a>: 大家好，我在尝试将 ViltProcessor 推送到 Hugging Face Hub 时遇到了一个问题。以下是我的代码：from PIL import Image from datasets import load_dataset from transformers impor...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/convert-to-parquet-fails-for-datasets-with-multiple-configs/86733">具有多个配置的数据集在执行 convert_to_parquet 时失败</a>: 你好，我们正在通过 datasets-cli 使用 convert_to_parquet 命令将带有数据加载脚本的数据集转换为纯数据。我们注意到，对于具有多个配置的数据集...</li><li><a href="https://stackoverflow.com/questions/78652890/how-do-i-free-up-gpu-memory-when-using-accelerate-with-deepspeed">在使用 Accelerate 和 Deepspeed 时如何释放 GPU 显存</a>: 我正在使用 accelerate launch 配合 deepspeed zero stage 2 进行多 GPU 训练和推理，并且在努力释放 GPU 显存。基本上，我的程序有三个部分：加载第一个 m...</li><li><a href="https://discuss.huggingface.co/t/clear-cache-with-accelerate/28745">使用 Accelerate 清理缓存</a>: 大家好！我正在尝试为多 GPU 训练清理缓存。我同时使用了 torch.cuda.empty_cache() 和 accelerator.free_memory()，但是 GPU 显存仍然处于饱和状态。torch.cuda.emp...</li><li><a href="https://www.geeksforgeeks.org/how-to-take-screenshots-in-windows-10/">在 Windows 10 中截屏的 7 种不同方法 - GeeksforGeeks</a>: 你可以使用各种工具在 Windows 上截屏，例如 Print Screen 按钮、截图工具（Snipping tool）、游戏栏（Game Bar）以及第三方应用。</li><li><a href="https://www.cnet.com/tech/mobile/how-to-take-a-screenshot-on-any-iphone-or-android-phone/">如何在任何 iPhone 或安卓手机上截屏</a>: 无论你想在 iPhone 16、Pixel 9 还是 Galaxy S25 上捕捉屏幕，这里是操作方法。</li><li><a href="https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome">欢迎来到社区计算机视觉课程 - Hugging Face 社区计算机视觉课程</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/vlms">视觉语言模型（Vision Language Models）详解</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/dpo_vlm">视觉语言模型（Vision Language Models）的偏好优化</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?q=leaderboard&sort=trending">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?q=bench&sort=trending">Spaces - Hugging Face</a>: 未找到描述
</li>
</ul>

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1355001170451759185)** (12 messages🔥): 

> `Teachable Machine 替代方案, Linuxserver.io 桌面环境, GUI Agent 演示, OpenAI CUA 模型` 


- **寻找 Teachable Machine 的继任者**：成员们讨论了 **Teachable Machine** 的替代方案，指出其 **UI 未开源** 且实现也未开源，其他人正在探索替代方案。
   - 一位用户注意到一个类似项目的链接已失效，这表明用户对易用的机器学习工具的搜索仍在继续。
- **FactoryManager 掌控 LinuxServer.io Docker**：一位成员介绍了 [FactoryManager](https://github.com/sampagon/factorymanager)，这是一个封装了 **linuxserver.io 桌面环境容器** 的 **Python 包**，能够实现对环境的编程控制，并展示了使用两个不同桌面环境的演示。
   - 该包旨在通过在 **linuxserver.io** 之上构建脚手架来提供灵活性，后者为许多桌面环境提供日常支持，这与 **Anthropic**、**OpenAI** 等公司的 GUI Agent 演示中经常创建的自定义环境有所不同。
- **OpenAI CUA 模型缺乏人性化交互**：使用 **OpenAI CUA 模型** 与 FactoryManager 进行的演示突显了其局限性，特别是它无法有效处理 **human-in-the-loop**（人机回环）场景。
   - 作者正在考虑是构建一个封装了 **OpenAI**、**Anthropic** 等功能的扩展基类，还是仅专注于项目的桌面管理器部分。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - edwardthefma 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/sampagon/factorymanager">GitHub - sampagon/factorymanager: 使用 robotgo-cli 编程控制 linuxserver.io Docker 容器的管理器</a>：使用 robotgo-cli 编程控制 linuxserver.io Docker 容器的管理器 - sampagon/factorymanager
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1355217243763638363)** (2 messages): 

> `smol-course 额度, Agent 课程额度, HuggingFace 额度` 


- **Smol-Course 中的额度危机**：一位成员对 **smol-course** 中额度耗尽表示沮丧，尽管其推理使用量极低。
   - 他们澄清了 **smol-course** 和 **Agents 课程** 之间的混淆，并询问了潜在的额度可用性。
- **处理课程额度的困惑**：一位参加 smol-course 的用户对意外耗尽额度表示担忧，尽管他们认为自己对推理的使用非常少。
   - 该用户澄清他们实际上是在学习 Agents 课程，而不是 smol course，并对为何额度用尽以及是否会有新额度感到困惑。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1354911236139257978)** (7 messages): 

> `在 Langfuse 中使用 LLM-as-a-Judge 评估毒性, Base 模型 vs Instruct 模型, 初始化后调整 Agent System Prompt` 


- **Langfuse 毒性评估器认为胡萝卜有毒？！**：一位在 Langfuse 中测试毒性 LLM-as-a-judge 的用户发现，它错误地将提示词 *'Can eating carrots improve your vision?'*（吃胡萝卜能改善视力吗？）标记为有毒，得分为 **0.9**，理由是其与气候变化言论存在错误的关联。
   - 该用户质疑 *如何评估评估器*，并指出 **GPT-4o** 将带有贬义的气候变化内容错误地归因于一个关于胡萝卜的无害问题。
- **Base 模型与 Instruct 模型：有什么区别？**：一位 Agent 新手寻求关于 Base 模型和 Instruct 模型区别的解答，并引用了课程中提到的 Chat Templates（聊天模板）。
   - 一位成员用比喻回应道：**Base 模型** 是 *“裸模型，没有包装”*，并分享了 [一篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/) 进一步阐述这些差异。
- **Prompt 难题：方向引导与数据流**：一位在 unit 2.1 结束时设计自己模型的用户，正苦于在 Agent 初始化后通过调整 **agent.system_prompt** 来引导模型遵循其指令。
   - 该用户质疑调整 **agent.system_prompt** 是否是修改模型行为的正确方式，以及 Prompt 示例是否具体决定了工具的使用方式和数据的传递方式。



**提及的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/">Reddit - 互联网的核心</a>：未找到描述

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1354957581273006131)** (1 条消息): 

> `简化求职申请、公司研究、求职信生成` 


- **学生简化求职申请**：一名学生开发了一个利用 Notebook LM 简化求职申请的系统，通过深入研究公司和职位角色，其在收集公司洞察方面的有效性获得了 **80% 的评分**。
   - 该流程包括将网页和报告保存为 PDF，并收集可靠的新闻，为 Notebook LM 提供具体细节，以撰写具有影响力的求职信和简历。
- **深度公司研究助力学生**：学生专注于探索公司的价值观和工作职责，通过缩小感兴趣的具体职位范围、访问公司网站、将网页保存为 PDF、下载相关的 PDF 报告，并从可靠的在线新闻/研究源收集信息。
   - 与 Notebook 对话可以提供关于公司的详细、具体的带引用回答，帮助学生充分了解公司的核心价值观、当前挑战以及他们可以做出的贡献。
- **求职信生成效果不佳**：一名学生尝试通过上传简历和公司详情，使用 Notebook LM 生成求职信，但由于生成内容过于通用且缺乏创意，评分仅为 **10%**。
   - 生成的求职信的通用性质凸显了该系统在为个性化申请材料提供有价值的见解或灵感方面的局限性。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1354895782687281294)** (29 条消息🔥): 

> `思维导图、上传源、版本控制、粘贴源命名、讲座转录的可读性` 


- **思维导图功能令用户惊叹**：一位用户对新的思维导图功能表示兴奋。
   - 他们称之为*另一个令人惊叹的时刻*。
- **源上传出现故障**：一位用户报告了源文件卡在永久上传状态的问题，导致既无法导入也无法删除，并表示这种情况已经持续了 8 小时。
   - 一位用户寻求删除永久上传中的源的方法，但未获成功。
- **版本控制缺失困扰用户**：一位用户对“Note”源类型缺乏版本控制和回收站支持表示担忧。
   - 他们对使用该功能犹豫不决，更倾向于使用具有数据保护和备份功能的 Google Docs。
- **源突然停止自动命名**：一位用户报告称，以前会自动命名的粘贴源现在默认显示为“pasted text”。
   - 他们询问是否有更新或恢复之前行为的方法。
- **NLM 无法解析 PDF**：用户讨论了 NLM 无法从扫描的 PDF 中提取数据的问题，一位用户询问该工具是否可以从扫描的笔记中提取数据。
   - 一位用户澄清说 **NLM 无法处理混合内容的 PDF**（文本和图像），但可以处理文档和幻灯片。



**提到的链接**：<a href="https://tenor.com/view/tole-cat-cute-gif-12080171459357821404">Tole Cat GIF - Tole Cat Cute - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1355237913495081183)** (2 条消息): 

> `LlamaCloud MCP Server, LlamaIndex MCP Client, AI Agent 系统, Text-to-SQL 转换` 


- **LlamaCloud 作为 MCP Server**：LlamaIndex 宣布本周为 **MCP week**，并展示了如何将 **LlamaCloud** 用作 [MCP server](https://twitter.com/llama_index/status/1905678572297318564)。
- **LlamaIndex 作为 MCP Client**：LlamaIndex 演示了如何将 **LlamaIndex** 作为任何 **MCP server** 的客户端，使 Agent 能够利用数百个现有的 MCP server 作为工具，并大幅[扩展能力](https://t.co/VTNomlb9c7)。
- **Text-to-SQL 转换系统**：LlamaIndex 将与 **SkySQL** 合作举办网络研讨会，主题是构建无需编码即可可靠执行 **text-to-SQL 转换** 的 **AI agent 系统**；更多信息请访问[此链接](https://twitter.com/llama_index/status/1905718367568462040)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1355181715571740672)** (18 messages🔥): 

> `ChatMessage history to the FunctionAgent workflow, Support rich content in agent responses, Custom telemetry attributes when interacting with Llama Index's LLM, Selectors, Agents , VannaPack and adding a memory with history = []` 


- **将 ChatMessage 历史记录添加到 FunctionAgent 工作流**：一名成员询问如何向 FunctionAgent 工作流添加聊天历史，[建议参考文档](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history)。
   - 用户被引导使用 `agent.run(...., chat_history=chat_history)` 来覆盖任何聊天历史，或者使用 `ChatMemoryBuffer.from_defaults(token_limit=60000, chat_history=chat_history)` 来管理 Memory 对象。
- **Agent 响应中的富文本内容支持**：一名成员询问在 Agent 响应中支持富文本内容的最佳方式，建议参考[此示例](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/)使用 function calling 构建自定义 Agent。
   - 该建议指向了使 Agent 创建更加容易的 LlamaIndex 抽象。
- **遥测追踪的进展 (Telemetry Tracking Triumph)**：一名成员询问在与 LLM 和 Agent 等 LlamaIndex 抽象交互时如何传递自定义遥测属性，以及如何向 LLM 网络调用附加 header 或参数。
   - 另一名成员分享了一个 [Colab notebook](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing)，演示了如何为代码块内执行的所有事件附加用户 ID。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history">Starter Tutorial (Using OpenAI) - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1355154991530377408)** (1 messages): 

> `LlamaParse PDF Issues, Multi-PDF Parsing` 


- **LlamaParse 在处理多个 PDF 时遇到困难**：一名用户报告称，**LlamaParse** 在处理单个 PDF 时工作正常，但在处理 **两个 PDF** 并询问相同问题时无法响应。
- **多文档处理导致系统过载**：用户描述系统在处理多个 PDF 时“彻底崩溃”（literally cooked），表明可能存在过载或处理错误。
   - 这暗示了 LlamaParse 在多文档处理能力上可能存在限制或 Bug。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1354979488164745347)** (13 messages🔥): 

> `Cohere "Command" naming, Coral Model Selection, Job opportunities at Cohere` 


- **Cohere 将其模型命名为 "Command"**：一名成员询问为什么 **Cohere** 会将其语言模型命名为 *Command*。 
   - 另一名成员建议，就像在数据库管理中一样，*query*（查询）本质上就是一个 **command**（命令）或 **instruction**（指令）。
- **Coral 默认使用 Command A**：一名成员询问 **Coral chat** 是否默认使用 **Command A**。
   - 另一名成员澄清说，Coral 中可以进行模型选择，并强调 *Just Chat* 在不使用外部来源的情况下使用 **Command A**。
- **成员寻求软件工程师工作**：一名成员表示他们正在寻找软件工程师的新工作机会，并很高兴讨论与网站或 Web 应用程序相关的潜在项目。
   - 另一名成员分享了 [Cohere 招聘页面](https://cohere.com/careers)的链接，鼓励该用户查看。



**提到的链接**：<a href="https://cohere.com/careers">Careers | Cohere</a>：我们的 ML/AI 专家团队热衷于帮助开发人员解决现实世界的问题。我们在多伦多、伦敦和帕洛阿尔托的办公室工作在机器学习的最前沿，以解锁...

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1355233912942624952)** (2 messages): 

> `Testing Bot Commands` 


- **Bot 命令进行测试运行**：鼓励成员在 「🤖」bot-cmd 频道中测试 Bot 命令。
- **鼓励进一步的 Bot 测试**：欢迎对 Bot 命令进行更多测试和反馈，以确保功能正常和良好的用户体验。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1355101762993786951)** (4 messages): 

> `Full-Stack Web Development, Mobile App Development, AI Solutions, Cloud Technologies, Oracle ERP Fusion` 


- **准备好构建的全栈炼金术士**：一位拥有 **8 年以上**经验的热情开发者，擅长使用 **React, Angular, Flutter, and Swift** 等现代框架构建可扩展的 **Web 和移动应用**。
   - 他们使用 **Python, TensorFlow, and OpenAI** 构建智能 **AI Solutions**，并集成 **Cloud Technologies (AWS, GCP, Azure)** 和 **Microservices** 以实现全球扩展。
- **Oracle 顾问寻求 Cohere 智慧**：一位在 **Oracle ERP Fusion** 领域拥有 **12 年以上**经验的技术顾问，渴望学习更多关于 **Cohere models** 以及企业级应用中 **AI use cases** 的知识。
- **网络专业学生想要开源音乐**：一位成员目前正通过 YouTube 和 MOOCs 学习 **Networking and CS**，目标是参与 **Open-Source Generative Music** 项目。
   - 他们最喜欢的技术工具包括 **ChatGPT, Grok, Windsurf, and Replit**。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1354937897198813384)** (7 messages): 

> `GPT4All usability issues, Mistral Small 3.1 and Gemma 3 implementation, GPT4All advantages, GPT4All v4.0.0 expectations, GPT4All model settings page` 


- **GPT4All 面临易用性投诉**：用户抱怨 **GPT4All** 的易用性，提到的问题包括无法导入模型、搜索模型列表、查看模型大小、使用 Latex 或自定义模型列表顺序。
   - 一位用户表示，他们正在*流失用户，因为其他产品更加用户友好且更愿意保持开放*。
- **GPT4All 在实现新模型方面滞后**：一位用户对 **GPT4All** 尚未实现 **Mistral Small 3.1** 和 **Gemma 3** 表示沮丧，并指出了它们的多模态能力。
   - 该用户表示 *Llama.cpp 正在落后*，如果 **GPT4All** 在 2025 年夏天之前赶不上，可能会转向其他工具。
- **GPT4All 提供原生 RAG 等优势**：尽管存在批评，**GPT4All** 仍具有 **Native RAG** 和开箱即用功能等优势。
   - 一位用户表达了对开发者的信心，并对 **GPT4All v4.0.0** 充满期待。
- **GPT4All 设置受到称赞**：一位用户赞赏 **GPT4All 的模型设置页面**，理由是其全面的选项和便捷的模型重新加载按钮。
   - 有人指出，*只需从聊天菜单中点击 2-3 次即可完成设置*，且其简单的集合选择功能非常出色。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

georgehotz: 大家能不能把那些过期的 PR 和 Issue 都关了？
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1354949525160333373)** (4 messages): 

> `TinyGrad Codegen, TinyGrad indexing` 


- **了解 TinyGrad Codegen 内部机制**：一位成员询问了 **TinyGrad** 的代码生成过程，特别是文档中提到的 `CStyleCodegen` 或 `CUDACodegen` 的位置。
   - 文档描述了 **TinyGrad** 使用不同的*转换器*（Renderers 或 Codegen 类），如 `C++ (CStyleCodegen)`、`NVIDIA GPUs (CUDACodegen)`、`Apple GPUs (MetalCodegen)`，将优化后的计划翻译成 CPU/GPU 可以理解的代码。
- **在 TinyGrad 中实现布尔索引**：一位成员询问在 **TinyGrad** 中是否有更好的方法来创建一个带孔的均匀分布网格点集，类似于 PyTorch 中的布尔索引。
   - 他们建议在 **TinyGrad** 中实现布尔索引可能是一个有用的贡献，特别是基于他们过去在 Dataframes 和 Kaggle 方面的经验。
- **使用 Masked Select 魔法修复索引！**：一个 LLM 提出了一种使用 **masked_select** 的解决方案，通过利用条件 `full.abs().max(axis=1) >= (math.pi/6)` 来过滤孔外的点，从而高效地创建所需的带孔网格。
   - 该解决方案涉及扩展掩码以匹配完整 Tensor 的形状，然后重新排列有效点，解决了该成员的挑战。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1355106246612750438)** (1 messages): 

> `DSPy output validation, DSPy handling invalid outputs` 


- **解决 DSPy 输出验证失败问题**：一名成员询问了当输出验证失败时 **DSPy** 的处理方法，例如一个整数字段预期接收 **1 到 10** 之间的数字，但实际收到了 **101**。
   - 关于此问题，目前没有进一步的讨论或提供的相关链接。
- **DSPy 中的无效输出处理**：用户的提问集中在 **DSPy** 如何管理模型输出不符合定义的验证标准的情况。
   - 具体而言，给出的例子是一个整数字段应在 1 到 10 之间，但模型错误地输出了 101。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1355176591965294772)** (3 messages): 

> `Optimizers in DSPy, Declarative Self-improving Python, Modular AI systems` 


- **探索 DSPy 框架中的 Optimizers**：一名成员正在探索在 **DSPy** 中使用 **optimizers**，以及它们如何与 docstrings 和 prompt 管理进行交互，并参考了 [DSPy 官方文档](https://dspy.ai/)。
   - 他发现的问题是 **Optimizer 会覆盖来自 docstring 的 prompt**，因此他们必须从 json 或 pkl 文件中加载优化后的版本。
- **理解 DSPy 的优化过程**：该成员澄清说 **DSPy 的 optimizer** 会创建 prompt 并在数据集上进行测试以找到性能最好的一个，并在 [官方网站](https://dspy.ai/) 上进行了详细说明。
   - Optimizer 可能会选择 N 个示例包含在 prompt 中，用户发现观察生成的 prompt 类型 *非常有趣*。
- **DSPy：声明式自我改进 Python**：**DSPy** 是一个用于对语言模型进行 *编程而非提示 (prompting)* 的框架，旨在快速迭代 **构建模块化 AI 系统**，并提供用于 **优化其 prompt 和权重** 的算法。
   - 你可以编写组合式的 _Python 代码_，并使用 DSPy **教你的 LM 提供高质量的输出**，而不是使用脆弱的 prompt。



**提到的链接**：<a href="https://dspy.ai/">DSPy</a>：用于对语言模型进行编程（而非提示）的框架。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1354931692829413448)** (3 messages): 

> `Entrepreneurship Track Mentorship, Office Hours with Sponsors` 


- **创业赛道不提供导师指导**：一名成员询问了创业赛道参与者的导师指导机会。
   - 遗憾的是，另一名成员澄清说 *Berkeley 不为创业赛道提供任何导师指导*。
- **赞助商主持答疑时间 (Office Hours)**：虽然 Berkeley 不提供导师指导机会，但在 4/5 月份将会有赞助商主持的答疑时间 (Office Hours)。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1354970106207142009)** (2 messages): 

> `Gemini 2.5 Pro release, Windsurf rate limits` 


- **Gemini 2.5 Pro 席卷 Windsurf！**：**Gemini 2.5 Pro** 现已在 Windsurf 中可用，每条消息授予用户 **1.0** 用户 prompt 额度，每次工具调用授予 **1.0** flow action 额度；详情请参阅 [X 上的公告](https://x.com/windsurf_ai/status/1905410812921217272)。
- **Windsurf 遭遇 Gemini 2.5 Pro 速率限制 (Rate Limits)**：在 Gemini 2.5 Pro 发布后不久，由于模型和提供商面临巨大负载，Windsurf 遇到了速率限制。
   - 团队正在努力研究如何增加配额，并对带来的不便表示歉意，目标是让每个人都能 *尽快在 Windsurf 上使用 Gemini 2.5 Pro*。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1905410812921217272">来自 Windsurf (@windsurf_ai) 的推文</a>：Gemini 2.5 Pro 现已在 Windsurf 中可用！✨

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1354992051246071858)** (1 messages): 

> `self parameter, Foo[1] default parameter` 


- **澄清 Foo[1] 中的默认参数值**：`self` 参数是带有默认参数值的 `Foo[1]`。
   - 使用 `_` 忽略它将默认为该默认参数值。
- **理解 Foo[1] 上下文中的 Self**：在 `Foo[1]` 类型上下文中的 `self` 参数可以自动填充。
   - 当使用 `_` 丢弃 `self` 时，该参数将默认为其预定义的默认值。


  

---


---


---


{% else %}


> 各频道的详细分析已为邮件格式截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}