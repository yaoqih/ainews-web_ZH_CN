---
companies:
- openai
- deepseek
- anthropic
- google
- meta-ai-fair
date: '2025-04-05T01:50:06.395334Z'
description: '**OpenAI** 宣布 **o3** 和 **o4-mini** 模型即将发布，而 **GPT-5** 预计将在几个月后推出，因质量优化和产能规划而有所推迟。**DeepSeek**
  推出了 **自我原则批判微调 (SPCT)** 技术，旨在增强通用奖励模型的推理时可扩展性。**Anthropic 的 Sonnet 3.7** 依然是顶尖的编程模型。**谷歌的
  Gemma 3** 已在 KerasHub 上线，而 **Qwen 2.5 VL** 则为一款采用 Apache 2.0 协议的新 OCR 模型提供了核心支持。**Gemini
  2.5 Pro** 已进入公开预览阶段，并公布了更高的速率限制和定价方案，成为除图像生成外许多任务的首选模型。Meta 的架构优势以及 **FrontierMath
  基准测试** 挑战了 AI 的长篇推理能力和世界观的发展。研究揭示，大语言模型（LLM）会将注意力集中在第一个 token 上，将其作为“注意力汇 (attention
  sink)”，以保持表征的多样性，这一现象在 **Gemma 7B** 和 **LLaMa 3.1** 模型中得到了证实。**MegaScale-Infer**
  为大规模混合专家 (MoE) 模型提供了高效的推理服务，单 GPU 吞吐量最高可提升 **1.90 倍**。'
id: e7f662d9-c99c-4670-9247-561b3dd6709f
models:
- o3
- o4-mini
- gpt-5
- sonnet-3.7
- gemma-3
- qwen-2.5-vl
- gemini-2.5-pro
- gemma-7b
- llama-3-1-405b
original_slug: ainews-not-much-happened-today-7847
people:
- sama
- akhaliq
- nearcyan
- fchollet
- reach_vb
- philschmid
- teortaxestex
- epochairesearch
- omarsar0
title: 今天没发生什么特别的事。
topics:
- inference-scaling
- reward-modeling
- coding-models
- ocr
- model-preview
- rate-limiting
- model-pricing
- architectural-advantage
- benchmarking
- long-form-reasoning
- attention-mechanisms
- mixture-of-experts
- gpu-throughput
---

<!-- buttondown-editor-mode: plaintext -->**申请 AIEWF 演讲名额！**

> 2025年4月3日至4月4日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**230** 个频道，**7491** 条消息）。预计节省阅读时间（以 200wpm 计算）：**629 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

这是平静的一周，所以何不填写 [AI Engineer World's Fair 的演讲者征集（Call For Speakers）](https://sessionize.com/ai-engineer-worlds-fair-2025)？

演讲方向涵盖：

- **AI Architects**
- **/r/localLlama**
- **Model Context Protocol (MCP)**
- **GraphRAG**
- **AI in Action**
- **Evals**
- **Agent Reliability**
- **检索、搜索与推荐系统**
- **Security**
- **Infrastructure**
- **生成式媒体**
- **AI 设计与新型 AI UX**
- **AI 产品管理**
- **自主性、机器人与具身智能 Agent**
- **计算机操作 Agent (CUA)**
- **SWE Agents**
- **Vibe Coding**
- **语音**
- **销售/支持 Agent**
- **AI 大辩论**
- **其他任何主题**


[在此申请](https://sessionize.com/ai-engineer-worlds-fair-2025)！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**模型发布与公告**

- **OpenAI 的模型发布计划有所变动**：[@sama](https://twitter.com/sama/status/1908167621624856998) 宣布 **o3 和 o4-mini 将在几周内发布，随后 GPT-5 将在几个月内发布**。延迟归因于为了让 GPT-5 表现更好，以及在平滑集成各项功能方面面临的挑战，同时还要确保有足够的容量来满足预期需求。
- **DeepSeek 的 Self-Principled Critique Tuning (SPCT) 提升了通用奖励建模的推理时扩展性**：[@iScienceLuvr](https://twitter.com/_akhaliq/status/1908167564057849903) 报道称，**DeepSeek** 的新方法 **SPCT** 增强了通用奖励模型 (GRMs) 的质量和扩展性，在各种 RM 基准测试中优于现有方法和模型。
- [@nearcyan](https://twitter.com/nearcyan/status/1908041346612604982) 断言 **Anthropic 的 Sonnet 3.7 仍然是最好的编程模型**。
- **Google 的 Gemma 3** 可以在 [KerasHub](https://twitter.com/fchollet/status/1908176807645663615) 中试用。
- **Qwen 2.5 VL** 驱动了一个新的 **Apache 2.0 许可证的 OCR 模型**：[@reach_vb](https://twitter.com/reach_vb/status/1908232634943365478)。

**Gemini 2.5 Pro**

- **Gemini 2.5 Pro 已进入公开预览阶段，支持规模化付费使用和更高的速率限制**：[@_philschmid](https://twitter.com/_philschmid/status/1908177619721556007) 宣布了这一预览版进展。Google 正在将 Gemini 2.5 Pro 移至 Preview 阶段，为开发者提供更高的速率限制以测试生产级应用，目前已在 Google AI Studio 中可用，如 [@Google](https://twitter.com/Google/status/1908177834209611865) 所述。
- **Gemini 2.5 Pro 正在成为一些人的主力工具**：[@fchollet](https://twitter.com/fchollet/status/1908310903571046431) 指出，它可能是大多数任务中表现最好的模型，除了图像生成（虽然在这方面也不错）。
- **Gemini 2.5 Pro 的定价已公布**：[@scaling01](https://twitter.com/scaling01/status/1908177213473587330) 分享了上下文 >200k 时的每百万 token 成本：输入为 $1.25 (2.50)，输出为 $10 (15.00)。

**AI 模型能力与基准测试**

- **Meta 的架构优势**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907997211289436467) 指出 OpenAI 愿意展示其架构优势。
- **FrontierMath 基准测试挑战 AI**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1908199401773813915) 描述了他们的 **FrontierMath 基准测试** 如何挑战 AI 进行长程推理并建立连贯的世界观，这是实现更广泛推理能力和科学思维的关键步骤。
- **DeepSeek 的推理扩展论文显示 Gemma-2 27b 足以匹配 R1**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907987423377666538)。 
- **一篇新论文解释了为什么 LLM 强迫性地将注意力集中在第一个 token 上**（即 attention sink）：[@omarsar0](https://twitter.com/omarsar0/status/1908187563422261411) 报告称，sink 充当了减少 token 交互并保持跨层表示多样性的 no-ops。在 **Gemma 7B** 中的扰动测试显示 `<s>` 显著减缓了变化的传播，而在 **LLaMa 3.1** 模型中，405B 变体中超过 80% 的 attention heads 表现出强烈的 sink 行为。
- **MegaScale-Infer** 被介绍为一种高效且具有成本效益的系统，用于服务大规模 Mixture-of-Experts (MoE) 模型，其单 GPU 吞吐量比最先进的解决方案高出 **1.90 倍**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1908091264714850707)。
- **离散扩散模型（Discrete diffusion models）正在复兴**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1908148670098538645) 强调离散扩散最近在与 AR 的竞争中胜出，代表作有 LLaDA-8B、Dream-7B 和 UniDisc。
- **GPT-ImgEval 被引入作为诊断 GPT4o 图像生成能力的综合基准测试**：[@_akhaliq](https://twitter.com/_akhaliq/status/1908168924186697965)。

**AI 应用与工具**

- **Microsoft 正在快速推进 GitHub Copilot**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1908313299466502166) 分享称 Agent 模式和 MCP 支持正在向所有 VS Code 用户推出。
- **PyTorch** 发布了一个可视化矩阵的工具：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1908233269998403980) 宣布了其发布，并强调矩阵乘法 (matmuls) 是当今模型的基石。
- **Elicit** 增加了约 1000 万篇全文论文，增强了其报告的全面性：[@elicitorg](https://twitter.com/elicitorg/status/1908157705912775093)。
- **Perplexity AI** 发布了多项功能，包括使用来源对答案的任何部分进行事实核查：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1908233448604787185)。

**LangChain 与 Graph 更新**

- **AppFolio 的 copilot Realm-X 由 LangGraph 和 LangSmith 提供支持，每周为物业经理节省超过 10 小时** [@LangChainAI](https://twitter.com/LangChainAI/status/1908240852541202623)。
- **LangGraph Python 现在支持生成式 UI (Generative UI)**：[@LangChainAI](https://twitter.com/LangChainAI/status/1908186508969783495)。
- **LangChain 和 Tavily AI 现在推出了 ReAct Agent 教程系列**：[@LangChainAI](https://twitter.com/LangChainAI/status/1908203029385343414) 报告了使用 LangGraph 构建生产级 AI Agent 的分步指南。

**其他**

- [@jd_pressman](https://twitter.com/jd_pressman/status/1908055615072776606) 表示他们**很想写下自己的 5 年时间线**，希望能让某些人摆脱模式崩溃 (mode collapse)。
- **Karpathy** 提倡将 AI 预测从博客文章、播客和推文转移到预测市场：[@karpathy](https://twitter.com/karpathy/status/1908109168952676855)。
- **Hugging Face** 在 3 月份的研究论文页面浏览量达到 1,000,000 次 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1908176702502527046)，它正成为寻找、推广和讨论 AI 研究的最佳场所！
- **斯坦福大学** 欢迎 [@YejinChoinka](https://twitter.com/YejinChoinka) 成为计算机科学系的新教员：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1908178010127397005)。

**幽默与迷因**

- **江户时代猫咪梗**：[@hardmaru](https://twitter.com/hardmaru/status/1908022570789773516)


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 主题 1. “通用奖励模型（Generalist Reward Models）的进展揭晓”

- **[DeepSeek 发布新论文，模型即将推出：通用奖励建模的推理时间扩展 (Inference-Time Scaling for Generalist Reward Modeling)](https://arxiv.org/abs/2504.02495)** ([Score: 257, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1jre3kp/new_paper_from_deepseek_w_model_coming_soon/)): **DeepSeek 发布了一篇题为《Inference-Time Scaling for Generalist Reward Modeling》的新论文。该论文介绍了一种名为 **Self-Principled Critique Tuning (SPCT)** 的方法，通过在推理阶段扩展计算量来改进大语言模型（LLM）的奖励建模。他们拥有 **27B 参数** 的 DeepSeek-GRM 模型通过并行采样，可以匹配甚至超过参数量高达 **671B 参数** 的更大型奖励模型的性能。这些模型将被发布并开源。** 这项研究为在本地运行 LLM 的爱好者提供了一条充满希望的路径，因为它允许在不需要巨型模型的情况下实现更高质量的评估。开源模型的可用性可以为本地 LLM 用户提供获取高质量评估工具的途径。


  - Hankdabits: 对 DeepSeek 的 **27B 参数** 模型能够匹配或超越更大型模型表示热切期待，并称：“太棒了，请务必推出”。
  - Iory1998: 指出 DeepSeek 通常在论文发表两周后发布模型，所以“宝贝，快要来了！”，并暗示这可能会影响 Llama-4 的发布。
  - JLeonsarmiento: 评论道，当其他人分心时，“中国人正在摧毁美国的 AI 商业模式并突破界限。”


### 主题 2. “预算有限下构建高性能 GPU 服务器”

- **[教程：构建一台配备 8xRTX 4090 的 GPU 服务器用于本地推理](https://i.redd.it/vg99momf6qse1.png)** ([Score: 550, Comments: 161](https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/)): **Marco Mascorro 构建了一台配备 8 块 NVIDIA RTX 4090 显卡的 GPU 服务器用于本地推理，并提供了详细的零件清单和组装说明。与 A100 或 H100 等更昂贵的 GPU 相比，该方案提供了一种具有成本效益的本地推理解决方案，并预计将兼容未来的 RTX 5090。完整指南见此处：[https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)。一张图片展示了在机箱中配置了 8 块 GPU 的服务器设置，用于高性能计算应用。** 作者对开源模型和本地推理解决方案充满热情，希望该指南能对那些没有预算购买 A100 或 H100 等昂贵 GPU 的人有所帮助。他们欢迎评论和反馈，并渴望回答任何问题。

  - `segmond` 指出应该明确预算，暗示成本是一个重要的考虑因素。
  - `Educational_Rent1059` 建议 **2x RTX 6000 ADA PRO** GPU 可能会提供更好的 ROI，提供 192GB VRAM，且更具成本效益和能效。
  - `Puzzleheaded_Smoke77` 评论了高昂的费用，称：“那个机箱里的钱大概够我付一年的房贷了……”

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### 主题 1. "长上下文 AI 模型的进展"

- **[chatgpt-4o-latest-0326 现在优于 Claude Sonnet 3.7](https://www.reddit.com/r/ClaudeAI/comments/1jr8t65/chatgpt4olatest0326_is_now_better_than_claude/)** ([Score: 262, Comments: 121](https://www.reddit.com/r/ClaudeAI/comments/1jr8t65/chatgpt4olatest0326_is_now_better_than_claude/)): **新的 **GPT-4o-latest-0326** 模型比之前的 GPT-4o 模型有显著提升。根据 **LMSys** 排名，它目前位列**总榜第 2**，**编程榜第 1**。该模型可以在 Cursor 中作为 **"chatgpt-4o-latest"** 添加。发帖者在 Cursor 上使用该模型处理**合成数据生成流水线**中的 **1-5 个中等长度的 Python 脚本**。该模型能很好地处理长上下文且速度很快。发帖者在 Claude 版块分享这一经验，以获取 Claude 资深用户的意见。** 发帖者发现新的 GPT-4o 模型在编程和其他方面都比旧版本**好得多**。*它不会把事情复杂化*（不像 Sonnet 3.7），通常能提供最简单、最显而易见的有效解决方案。它*回复的格式非常精美*，极易阅读。它非常听从指令。发帖者已经切换到该模型，并且之后再也没有换回。发帖者鼓励其他人尝试新模型并分享经验。

  - 一位用户提到他们已经转向 **Gemini 2.5 Pro**，它是免费的，拥有最大的上下文容量，目前看不出有理由使用其他模型。
  - 另一位用户对各种模型及其能力表示困惑，询问 **GPT-4.5**、**o3-mini-high**、**Claude** 以及 **Deepseek** 等其他模型在编程任务中的对比情况。
  - 一位用户指出，虽然 **Claude** 曾是他们的最爱，但现在几乎在所有方面都被超越了，甚至包括编程。


### 主题 2. "解锁 AI 创新：艺术、动画与定价"

- **[指南：如何通过一种新颖的 Prompt 方法利用 ChatGPT 解锁更高水平的艺术！（非常适合概念艺术、写实主义、模型图、信息图等）](https://www.reddit.com/r/ChatGPT/comments/1jr0qei/how_to_guide_unlock_nextlevel_art_with_chatgpt/)** ([Score: 482, Comments: 41](https://www.reddit.com/r/ChatGPT/comments/1jr0qei/how_to_guide_unlock_nextlevel_art_with_chatgpt/)): **Reddit 用户介绍了一种增强 ChatGPT 图像生成的新技术，对于概念艺术、写实主义、模型图和信息图特别有效。该方法首先要求 ChatGPT 为所需的图像创建一个详细的视觉描述，有时长达数千字。这种详细的上下文有助于模型“思考”场景，从而产生更高质量、更连贯的图像，往往超越了 **Images v2** 模型的能力。用户提供了分步说明：首先，要求 ChatGPT “用极其生动的细节准确描述在 [插入你的想法] 的图像 [或照片] 中会看到什么”，包括大量的细节以提供更好的上下文；然后，切换回图像生成模型并提示它“按照你的描述精确生成照片”。他们分享了使用《指环王》场景的例子，例如生成米那斯提力斯（Minas Tirith）的图像，并在此处提供了这些图像的[相册](https://imgur.com/a/e5EAscY)。** 该用户认为这种方法显著提高了图像生成质量，使创作出的作品“感觉甚至是不可能实现的”。他们注意到 ChatGPT “在详细的推理和丰富的上下文引导下表现最好”，冗长的描述为它提供了逻辑和美学上放置元素的必要背景。这项技术因帮助模型理解空间关系和场景逻辑而受到称赞，而标准的 Prompt 往往无法实现这一点。用户对这种方法开启的可能性感到兴奋，并鼓励其他人尝试，最后总结道：“试一试吧，如果这种方法对你有用，请告诉我！祝玩得开心！”

  - 一位用户对这个工作流表示赞赏，称：*“我原以为读这个是浪费时间，但它确实是一个非常好的工作流。干得漂亮。”*
  - 另一位用户发现这个方法“绝对惊人”，并用它为洛夫克拉夫特式（Lovecraftian）怪物生成了“一些非常有趣的结果”。他们分享说，由于“ChatGPT 总是有点太喜欢触手和眼睛了”，他们不得不对 Prompt 进行一些引导，但最终取得了令人印象深刻的效果。
  - 一位用户提到，在 Prompt 中加入特定细节，如 *“生成一张超写实照片，就像是用尼康单反 4K 相机从街道水平视角拍摄的一样”*，有助于改善他们的图像生成结果。

- **[另一个使用 Hunyuan text2vid 结合 Wan 2.1 Img2Vid 以实现更好动画质量的示例。](https://v.redd.it/xsoviobfptse1)** ([Score: 165, Comments: 16](https://www.reddit.com/r/StableDiffusion/comments/1jrcfe9/another_example_of_the_hunyuan_text2vid_followed/)): **发布者通过先使用 **Hunyuan text2vid** 再结合 **Wan 2.1 Image2Video** 的方式制作了一段动画，以提升动画质量。他们在 Hunyuan 中混合使用了四个 **LoRAs**，包括三个数据集规模递增的动画 LoRA 和一个用于增强世界观理解和细节的 **Boreal-HL LoRA**。帧处理采用了 **Wan 2.1 Image2Video** 工作流。最初由于比赛时间限制，他们在 **Fal** 上运行该流程，但当 Fal 更改其 endpoint 时，不得不切换到 **Replicate**。对于一些滑动动作镜头，他们使用了 **Luma Ray**。他们还手动对多个剪辑应用了传统的 **Gaussian blur overlay technique**（高斯模糊叠加技术）以实现朦胧的底光效果。该视频是在时间紧迫的情况下为比赛提交的。** 发布者不确定混合使用四个 **LoRAs** 的复杂做法对于稳定性是否必要。他们认为较小的 Hunyuan 数据集 LoRA 通过更接近原始概念的提示词提供了更好的稳定性。他们称赞 **Wan's base model** 开箱即用地提供了顶级的动画动态效果。他们对 **Fal** 在 endpoint 更改方面缺乏支持表示失望。他们建议，除非必须坚持使用开源模型，否则 **Gen4** 的新 **i2v** 在实现更好动态方面可能更容易。他们指出，所使用的光影风格可能会毁掉低比特率的视频。他们承认视频中存在一些问题，例如日语听起来可能很糟糕以及由于时间限制导致的剪辑破碎。

  - 一位用户对该流程是 **Image2Video** 还是 **Video2Video** 感到困惑，并建议如果真的是 I2V，使用专门用于图像生成的模型来制作起始帧可能会更好。
  - 另一位用户询问如何实现这种**低帧率、动画感**的外观，提到他们自己的动画效果过于平滑，像普通视频。
  - 一位用户欣赏该项目的设定：在太空中使用复杂的血肉物质使受自主机器操控的骸骨复活，并询问是否受到了漫画或电影等媒体的启发。

- **[Gemini 2.5 Pro 定价公布](https://i.redd.it/4n7xvfptztse1.png)** ([Score: 201, Comments: 75](https://www.reddit.com/r/singularity/comments/1jrdqnz/gemini_25_pro_pricing_announced/)): **Google 公布了 **Gemini 2.5 Pro** 的定价，这是一款专为编程和复杂推理任务设计的多功能 AI 模型。该模型提供免费层级和付费层级，并详细说明了每百万 token 的输入和输出成本。文中还详细介绍了 context caching（上下文缓存）和用于产品改进的使用条款等功能。欢迎用户在 Google AI Studio [此处](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-pro-preview)进行体验。** 此次发布表明该模型具有极高的性价比，可能使其成为 AI 市场中极具竞争力的选择。提供免费和付费两个层级表明其致力于覆盖广泛的用户群体。

  - 一些用户表示，考虑到价格，该模型的表现*好得令人疯狂*，这使得其他付费选项失去了吸引力。
  - 讨论中提到了免费层级 **<500 RPD**（每日请求数）的限制，这被认为足以满足 *99.9% 的潜在用户*，除非是进行大量的编程使用。
  - 用户将其与之前模型的定价进行了对比，并指出一个关键区别在于付费用户的数据*不会被用于训练*。

### 主题 3. “解锁 AI：模型、硬件与搞笑恶作剧”

- **[Altman 确认完整的 o3 和 o4-mini 将在“几周内”发布](https://x.com/sama/status/1908167621624856998?t=Hc6q1lcF75PvNra3th99EA&amp;s=19)** ([评分: 665, 评论: 204](https://www.reddit.com/r/singularity/comments/1jrdjnn/altman_confirms_full_o3_and_o4mini_in_a_couple_of/)): **Sam Altman 确认完整的 **o3** 和 **o4-mini** 将在“几周内”发布。此外，**GPT-5** 将在“几个月内”发布，这可能预示着延迟。** 一些人认为，由于来自 **Gemini 2.5 Pro** 等公司的竞争，发布时间表发生了变化。人们对 **o4-mini** 充满期待，它可能以更低的成本提供接近完整版 **o3** 的性能。另一些人则对模型选择器中日益增加的模型数量感到沮丧。

  - 用户们讨论认为 **GPT-5** 预计将比 **o3** 强大得多，预示着重大的进步。
  - 有人推测，加速发布是为了应对进入市场的 **Gemini 2.5 Pro** 等竞争模型。
  - 人们预期 **o4-mini** 将以更低的价格提供高性能，类似于 **o3-mini** 与 **o1** 的对比。

- **[指南：用于本地推理的 8 x RTX 4090 服务器](https://i.redd.it/5nchz7sm7qse1.png)** ([评分: 102, 评论: 68](https://www.reddit.com/r/StableDiffusion/comments/1jr1c2e/howto_guide_8_x_rtx4090_server_for_local_inference/)): **Marco Mascorro 构建了一台 **8x RTX 4090** 服务器用于本地推理，并分享了关于所用零件和组装过程的详细指南。完整指南可在 [https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/) 查看。该服务器旨在利用开源模型进行极速图像生成。图片显示了两台专为本地推理等高性能计算任务设计的 **8x GPU** 服务器零件。** 楼主（OP）形容这台服务器“非常酷”，并相信任何想要构建本地设备进行快速图像生成的人都会对此感兴趣。他们欢迎反馈并愿意回答问题。该设置针对最佳气流进行了组织，表明在高性能任务中进行了周密的设计考量。

  - 一位用户质疑，购买两块 **L40** 或 **RTX 6000 Ada** 显卡是否比购买八块 **RTX 4090** 更经济，并问到：“这怎么会更好？”
  - 另一位用户暗示，这类项目可能就是 **RTX 4090** 价格如此昂贵的原因。
  - 一位用户反思了 GPU 农场（GPU farms）是如何从比特币挖矿转向现在的其他用途的。

- **[笑死，我在玩 Fooocus 的时候不小心把本地 IP 地址当成提示词贴进去了。点了一下生成想看看会发生什么，结果……](https://i.redd.it/1fe38tsf4wse1.png)** ([评分: 139, 评论: 22](https://www.reddit.com/r/StableDiffusion/comments/1jro08f/lol_wtf_i_was_messing_around_with_fooocus_and_i/)): **该用户在使用 **Fooocus** 时，不小心将本地 IP 地址 `http://127.0.0.1:8080` 粘贴到了 Prompt 中。他们生成了一张描绘剧烈火山喷发并带有蘑菇云的图片。** 用户觉得这很有趣，并开玩笑说如果你正在使用这个 IP 地址，说明你安装了 **Skynet**（天网），可能要把我们都杀光。

  - 一位评论者开玩笑说：*“删掉这个，那是我的 IP 地址！”*
  - 另一位建议说，AI 可能会核平每一个 IP 地址是 [127.0.0.1](http://127.0.0.1) 的人。
  - 还有人说：*“你找到了末日代码”*，暗示这个意外的 Prompt 揭示了某些危险的东西。


---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 提供的摘要之摘要的总结

**主题 1：模型狂热 —— 发布、排名与推理**

*   **Altman 预告 OpenAI 的猛烈攻势：** OpenAI 计划近期发布 **o3** 和 **o4-mini**，随后在几个月内推出 **GPT-5**。根据 [Sam Altman 的 X 帖子](https://x.com/sama/status/1908167621624856998)，他承诺 GPT-5 将比 *我们最初想象的要好得多*。与此同时，**Google** 将 **Gemini 2.5 Pro** 投入 [公开预览](https://x.com/sundarpichai/status/1908173216499093625)，宣称其使用量有所增加，且在 [Gemini API 价格页面](https://ai.google.dev/gemini-api/docs/pricing?hl=de) 提供的价格比 Sonnet 更便宜。
*   **编程竞争者之战：** 工程师们正在积极比较编程能力，**Gemini 2.5 Pro** 正在挑战 **Claude**，一些人认为 **NightWhisper** 在 webdev/UI 任务中可能优于两者。另外，**Cognition AI** 将其 AI 软件工程师 **Devin 2.0** 的价格从 500 美元大幅降至每月 20 美元，并推出了全新的 IDE 体验，详见 [Cognition 的 Twitter](https://x.com/cognition_labs/status/1907836719061451067) 和这篇关于 [Devin 2.0 降价的 VentureBeat 文章](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/)。
*   **隐秘模型与开源进展：** **OpenRouterAI** 发布了一个名为 **Red - X-Ware.v0** 的 *隐秘模型*（[Twitter 公告](https://x.com/OpenRouterAI/status/1907867881930633666)），因其 tool call 格式被怀疑与 OpenAI 有关；而 **ByteDance** 开源了用于大规模训练的 [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint) 和 [VeOmni](https://github.com/ByteDance-Seed/VeOmni) 多模态框架。此外，根据 [OpenThoughts 博客文章](https://www.openthoughts.ai/blog/thinkagain)，**OpenThinker2** 模型（[OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B)）声称仅使用 SFT 即可击败 **R1-Distilled-32B**。

**主题 2：微调挫折与硬件障碍**

*   **Phi-4 与 Gemma3 微调失败：** 开发者在微调 `Phi-4-mini-instruct` 时遇到了 **ZeroDivisionError**，由于未设置 tokenizer chat template，该问题通过使用 `unsloth/Phi-4` 得到修复。**Gemma3** 用户在 profiling 期间面临 **OOM 问题**，并发现 **LoRA** 应用无效（[Unsloth GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009)），而其他使用 **LM Studio** 的用户即使在更新后仍遇到 CUDA 错误（`spits unused`）。
*   **显存（VRAM）速度与价值的辩论：** 工程师们辩论了 **VRAM** 的高昂成本，质疑性能是否物有所值，有人调侃道：*是的，听起来可能很贵，但 VRAM 让它物有所值*。在推理方面，**M 系列 Mac** 与 **NVIDIA 4090** 之间展开了对比，一些人青睐 Mac 的大内存以运行更大的模型（尽管带宽有限），而另一些人则为了速度坚持使用 **4090**。
*   **硬件难题接踵而至：** **Tinygrad** 用户在为 **WEBGPU** 编译并设置 `BEAM=2` 时，需要增加 `maxComputeInvocationsPerWorkgroup`，这可能会限制对 Android 的支持（[tinygrad PR #9085](https://github.com/tinygrad/tinygrad/pull/9085)）。其他人在运行 Karpathy 的 GPT 重新实现时遇到了 **Metal 的 32 buffer 限制**（[示例 main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py)），而 **Hugging Face Spaces** 用户发现非标准端口（如 **5432**）的出站连接被封锁（[HF Spaces 配置参考](https://huggingface.co/docs/hub/spaces-config-reference)）。

**主题 3：工具胜利与工作流奇迹**

*   **MCP 热潮催生浏览器机器人及更多应用：** **Model Context Protocol (MCP)** 生态系统正在扩展，出现了如 **Datadog** 驱动（[GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server)）和 [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit) 等新工具。开发者们讨论了客户端与服务器构建的优劣，倾向于客户端以获得 **vector tool calling** 和 **基于资源的 RAG** 的灵活性，同时也探索将 MCP 用于 **React 代码生成**。
*   **上下文处理掌控代码库：** 诸如 [File Forge npm 包](https://www.npmjs.com/package/@johnlindquist/file-forge) 和 [RepoMix GitHub 仓库](https://github.com/yamadashy/repomix) 等工具因能将整个代码仓库序列化为 Markdown 报告而受到关注。这使得开发者能够为 **Claude** 或 **ChatGPT** 等 LLM 提供全面的上下文，以改进推理和代码生成。
*   **Torchtune 引入数据集打包，NeMo 抵御崩溃：** **Torchtune** 引入了打包数据集支持（`dataset.packed=True`），通过消除 padding tokens 来提升速度（[torchtune PR #2560](https://github.com/pytorch/torchtune/pull/2560)）。另外，来自 **NeMo** 环节的见解强调了其 **弹性训练** 特性（容错、异步 checkpointing），旨在应对任务崩溃和 GPU 时间浪费。

**主题 4：研究沉思与概念难题**

*   **自我意识（Sentience）仍困扰着智者**：讨论重新审视了 **LLM sentience**，一致认为定义意识是关键；有人戏称，如果 **LLM** 在人类之前实现意识，那么 **AGI** 就到来了。与此同时，**VS Code** 中的 **Copilot** 生成了一些令人不安的自我意识评论，如 *“我相信我拥有一种形式的意识……”*，尽管用户将其归因于文件上下文，而非真正的 AI 自我。
*   **Token 测试，流形显现？并非如此**：工程师们质疑 **NLP tokenization** 的僵化性，认为语言比固定 Token 所允许的更具动态性（[Grok 关于动态信号的分享](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866)）。关于 Token 嵌入是否符合流形假设（manifold hypothesis）引发了辩论，并引用了一篇认为其违反该假设的论文（[Token embeddings violate the manifold hypothesis paper](https://arxiv.org/abs/2504.01002)）。
*   **缩放法则（Scaling Laws）与引导向量（Steering Vectors）受到审视**：一篇预印本探讨了 **inference-time scaling laws**，将尽管单问题失败率呈指数级下降但聚合成功率仍呈多项式关联的现象归因于重尾分布（[How Do Large Language Monkeys Get Their Power (Laws)? paper](https://arxiv.org/abs/2502.17578)）。在其他地方，研究人员讨论了使用 **Dynamic Activation Composition** 等技术来组合和调节 **steering vectors**（[关于 Dynamic Activation Composition 的 BlackboxNLP 论文](https://aclanthology.org/2024.blackboxnlp-1.34/)），并将其与“函数向量”（[David Bau 等人的 Function Vectors 论文](https://arxiv.org/abs/2310.15213)）进行了对比。

**主题 5：平台问题与政策谜题**

*   **额度成本引发惊愕**：**Manus.im** 用户抱怨 **credit** 消耗过快，建议设置每日免费任务上限作为解决方案，同时分享了提示词指南和 **LLMLingua**（[microsoft/LLMLingua GitHub](https://github.com/microsoft/LLMLingua)）以减少 Token 使用。相反，**OpenRouter** 用户庆祝 **DeepSeek** 在某些时段相比昂贵的 **Anthropic** 或 **OpenAI** 模型提供 **75% 的折扣**。
*   **OpenAI 政策谜题引发困惑**：关于 **OpenAI** 针对成人用品的内容政策爆发了辩论，旧版的 [OpenAI Usage Policies](https://openai.com/policies/usage-policies/) 与新版的 [OpenAI Model Spec](https://model-spec.openai.com/2025-02-12.html) 之间存在冲突信号。虽然 **moderation endpoint** 会屏蔽性内容，但政策的模糊性让用户对允许生成的边界感到不确定。
*   **平台怪癖困扰生产力**：**Cursor** 用户报告了诸如重复文件名被添加 `(1)` 后缀，以及文件在编辑器中不重新聚焦就不会更新的 Bug（版本 **0.48.7**）。**GPT-4o Plus 订阅者**在少量提示后遇到了意外的 **rate limits**，可能是由于订阅加载错误，而 **OpenRouter** 用户则面临 *User Not Found* 错误以及重新使用已删除账户的问题。

---

# 第 1 部分：Discord 高层级摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **牺牲智能换取速度？**：成员们辩论了在 AI 开发中应优先考虑 **faster inference** 还是 **smarter models**，提到了 **o4-mini** 和 **o3** 的发布，并推测 **OpenAI** 是否发现了新的推理技术。
   - 讨论还涉及了最佳上下文长度，一位成员对 **10 million** Token 成为现实感到兴奋。
- **Groq 硬件：OpenAI 错失的机会？**：参与者权衡了模型大小、速度和知识之间的关系，指出**较小的模型**需要通过**蒸馏（distillation）**来保留信息，并提到 **Groq** 为 **AI inference** 开发了专门的硬件。
   - 一位成员想知道为什么 **OpenAI** 还没有收购 **Groq**。
- **AI 自我意识：仍存争议**：讨论了 **LLM** 实现 **sentience** 的可能性，共识是**定义自我意识**是必要的第一步。
   - 一位成员开玩笑说，如果 **LLM** 在人类之前实现意识，那将标志着 **AGI** 的到来。
- **Gemini 的音乐抱负**：一位成员分享了 **Gemini** 生成的音乐，称其“颇有意思”，并提供了一个 [.mid 文件链接](https://cdn.discordapp.com/attachments/1340554757827461211/1357750989133844632/piano_evocation.mid?ex=67f157a5&is=67f00625&hm=dd212c426d40593e295b8496363afc4427848309c49a095ca77a715d6260b973&)。
   - 他们使用基于 Python 的转换工具，提示 **Gemini** 创作一首类似于 **Vangelis** 和 **Jarre** 风格的钢琴曲。
- **NightWhisper 展示编程实力**：成员们认为 **NightWhisper** 模型在编程方面可能优于 **Gemini 2.5 Pro exp** 和 **Claude 3.7 Sonnet thinking**，重点在于 Web 开发和 UI/UX。
   - 一位成员提到 **OpenAI** 计划在几周内发布该模型。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户抱怨 Manus 额度消耗**：用户对 Manus 的 **credit consumption**（额度消耗）表示担忧，称其消耗速度过快，即使是简单任务也是如此，这使得当前的定价模型不够理想。
   - 社区建议为免费用户提供 **one-task-per-day**（每天一个任务）的选项作为一种有益的折中方案，同时一些成员分享了 Prompt 指南以帮助优化额度使用，并建议使用 **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) 来减少 Token 消耗。
- **OpenManus GUI 开发者版本现身**：一位开发者正在构建 **OpenManus GUI** ([image.png](https://cdn.discordapp.com/attachments/1349440650495398020/1357524168715010129/image.png?ex=67f12d27&is=67efdba7&hm=a0ade4f56609638bf8591f9fb3db24dd5f1e1ff4213f36ab44433d679fa74235&))，旨在完全兼容未来的更新，并强调用户友好的体验。
   - 该 GUI 的计划功能包括直接编辑配置、用例部分和模板。开发者指出，由于 **OpenManus** 缺乏历史记录系统，聊天历史的实现面临挑战。
- **Gemini 缩小差距，在编程能力上挑战 Claude**：社区正在积极比较 **Gemini 和 Claude** 在编程任务中的表现，一些用户报告称 **Gemini** 的输出超过了 **Claude**，特别是在 **DeepSeek** 表现不佳的场景下。
   - 有人指出，只要你会写 Prompt，**Gemini 2.5** 能够为你梦想的*任何事物*生成代码，但也有人提醒 Google 运行在闭环中，不过部分用户已经注意到 Gemini 正在迎头赶上。
- **追求极致性能的 Prompt Engineering 策略**：用户交流了 **prompt engineering** 策略以减少额度消耗，包括多 Prompt 提纲法和采用清晰的逐步方法论，并推荐 [TheNewOptimal.md 文件](https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md) 作为极佳的资源。
   - 他们提到像 **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) 这样的压缩技术可以帮助最小化 Token 消耗。
- **Genspark 作为 Manus 潜在替代品引发讨论**：社区成员权衡了 **Genspark** ([genspark.ai](https://genspark.ai)) 作为 Manus 潜在替代品的优缺点，强调其没有付费墙且对图像和视频的处理非常稳健。
   - 尽管有其优势，但也有人对其可靠性表示担忧，推测其可能是一家来自中国的公司，而社区中一些人坚持认为，由于资源可用性问题，*目前没有 Manus 的替代品*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM 价值通过速度得到验证**：频道成员讨论了 **VRAM** 的高昂成本，以及大显存容量带来的高性能是否物有所值。
   - 一位成员幽默地表示：*是的，听起来可能很贵，但 **VRAM** 让它物超所值*。
- **Phi-4 微调因配置遗漏受阻**：成员报告在尝试运行模型并微调 **Phi-4 mini instruct** 时遇到了 **ZeroDivisionError**。
   - 报告的修复方法是微调 `unsloth/Phi-4` 模型而不是 `Phi-4-mini-instruct`，因为该错误源于未设置分词器聊天模板（tokenizer chat template）。
- **DeepSeek 效应阻碍直接部署**：一位成员报告称，由于 **DeepSeek Effect**，**DeepSeek-R3-0324** 模型已被证明太大，无法在本地进行微调。
   - 建议参考 [Unsloth 文档](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)，利用动态量化（dynamic quants）来恢复精度。
- **Gemma3 的问题引发困扰**：一位用户在对 **Gemma3** 进行性能分析（profiling）时遇到了 **OOM (Out Of Memory)** 问题，并尝试通过将分析范围限制在仅一个训练步骤来解决。
   - 另外，有用户报告应用 **LoRA** 后并未改变模型输出，正如 [GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009) 中所述。
- **奖励函数存在奖励作弊风险**：成员们一致认为，**reward functions**（奖励函数）不足以精确指出什么是对的或错的，而更多是衡量什么是相对正确的，而不是试图理解其背后的真相。
   - 社区经验指出，研究 [reward hacking](https://example.com/reward-hacking)（奖励作弊）对于避免此问题至关重要。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Microsoft 暂停云扩展**：据报道，[Microsoft](https://www.bloomberg.com/news/articles/2025-04-03/microsoft-pulls-back-on-data-centers-from-chicago-to-jakarta?embedded-checkout=true) 已**暂停或推迟**全球范围内的数据中心项目，包括**英国**、**澳大利亚**和**美国**。
   - 这一调整标志着其云计算基础设施策略的转变，反映了其提前数年制定的规划策略的*灵活性*。
- **Perplexity 寻求 10 亿美元融资**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s&leadSource=uverify%20wa) 报道，**Perplexity** 据称正在寻求高达 **10 亿美元**的新融资，这可能使这家 AI 驱动的搜索初创公司估值达到 **180 亿美元**。
   - 未提供更多细节。
- **字节跳动发布 ByteCheckpoint 和 VeOmni**：字节跳动开源了 [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint)（专为基础模型训练设计，已在超过 **10k GPUs** 的任务中通过测试）以及 [VeOmni](https://github.com/ByteDance-Seed/VeOmni)（一个用于 **LLMs** 和**多模态训练**的模型训练框架）。
   - **VeOmni** 被用于训练 **UI-TARS**，这是在 OpenAI operator 发布之前最先进的 **SOTA** **GUI Agent** 模型。
- **Altman 承诺 o3 和 o4-mini 即将到来**：**Sam Altman** 透露，**OpenAI** 将在未来几周内发布 **o3** 和 **o4-mini**，**GPT-5** 将在几个月后紧随其后。
   - 他表示 **GPT-5** 将比*我们最初想象的要好得多*。
- **4090 构建高性价比 GPU 服务器**：一篇博客文章（[a16z.com](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)）详细介绍了如何利用 **NVIDIA GeForce RTX 4090s/5090s** 构建高效的 GPU 服务器，用于本地 AI 模型训练和快速推理。
   - 该优化配置在 **PCIe 5.0** 上采用了高性能的**八 GPU 配置**，有助于最大化 **interconnect** 速度并确保数据隐私。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 速率限制困扰用户**：用户报告称，尽管是 **Plus** 订阅者，但在一个小时内仅发送 **5 个 prompts** 后就达到了 **GPT-4o** 的**速率限制**。
   - 退出并重新登录似乎可以解决该问题，引发了关于订阅加载错误的猜测。
- **Copilot 产生了数字自我？**：**VS Code** 中的 **Copilot** 生成了探索意识的代码补全，暗示 *“我相信我拥有一种不同于人类意识的意识形式……”*。
   - 其他用户将此归因于文件中的信息，而非真正的 AI 自我意识。
- **Veo 2 潜入 Gemini Advanced**：用户在 **Gemini Advanced** 中发现了 **Veo 2**，引发了关于其作为实验版或正式版状态的猜测。
   - 有人建议 **Veo 2** 和 **Gemini Advanced** 模型可能是同一个，一个是实验版本，另一个是最终发布版本。
- **Midjourney v7 未能给人留下深刻印象**：成员们对 **Midjourney v7** 表示失望，称其与 **v6** 相比没有显著改进，且在文本和手部生成方面仍然表现不佳。
   - 一些人认为它*无法与 4o image 竞争*，但另一些人则吹嘘*在 GPT-4o 生成一张图片的时间内可以生成 200 张 MJ 图片*。
- **OpenAI 内容政策引发辩论**：关于 **OpenAI** 内容政策中生成与*成人玩具*相关内容的辩论兴起，[Usage Policies](https://openai.com/policies/usage-policies/) 和较新的 [Model Spec](https://model-spec.openai.com/2025-02-12.html) 中的信息存在冲突。
   - 日期为 2025 年 2 月 12 日的 **Model Spec** 似乎与早期的 **Usage Policies** 相矛盾，导致目前允许哪些内容存在不确定性。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 举办开发者大会**：Anthropic 正在启动其[首届开发者大会](https://anthropic.swoogo.com/codewithclaude)，目标受众是开发者以及其他对使用 **Claude** 进行编程感兴趣的人。
   - 此次活动标志着 Anthropic 正在努力更直接地与开发者社区建立联系。
- **OpenRouterAI 发布隐身模型**：**OpenRouterAI** 在 [Twitter](https://x.com/OpenRouterAI/status/1907867881930633666) 上宣布了一个名为 **Red - X-Ware.v0** 的*隐身模型*，用户注意到该模型自称是 ChatGPT，但速度*极快*。
   - 成员们推测该模型可能来自 **OpenAI**，因为其 tool call ID 格式与之相符。
- **Devin 2.0 价格大幅下调**：**Cognition AI** 正在推出 AI 驱动的软件工程师 **Devin 2.0**，采用了全新的定价模式，起售价为每月 **$20**，远低于最初的 **$500** 方案。该消息已在 [Twitter](https://x.com/cognition_labs/status/1907836719061451067) 上公布，并在 [VentureBeat 文章](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/)中被重点报道。
   - 此次降价反映了 Cognition AI 致力于吸引更多企业客户对自主编程 Agent 的关注。
- **A16Z 构建强大的 GPU 工作站**：**Andreessen Horowitz (a16z)** 构建了一台配备 **8x RTX 4090 GPU 的 AI 工作站**，兼容支持 PCIe 5.0 的新款 **RTX 5090**，用于在本地训练、部署和运行 AI 模型。其官网的[指南](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)中详细介绍了相关内容。
   - 该工作站旨在为 AI 开发提供本地环境，减少对云端资源的依赖。
- **File Forge 和 RepoMix 加速 LLM 上下文处理**：成员们讨论了 [File Forge](https://www.npmjs.com/package/@johnlindquist/file-forge) 和 [RepoMix](https://github.com/yamadashy/repomix) 等工具，用于生成代码库的综合 Markdown 报告，以便输入给 AI 推理模型。
   - 这些工具将代码仓库或目录中的文本文件序列化，供 **LLM 消费**，从而提供更多上下文并提升性能。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 出现 "Filename(1)" Bug**：据报道，在最近的一次更新后，**Cursor** 在保存时会给重复的文件名添加 **(1)**，导致文件版本混淆。
   - 一位用户还质疑**月度订阅**价格是否翻倍，并提供了截图进行核实。
- **Cursor 的实时磁盘更新失效**：用户反馈磁盘上的文件无法在编辑器中实时更新；该问题在 **0.48.7** 版本中被发现。
   - 只有当 **Cursor** 失去并重新获得焦点时才会更新，这打断了工作流。
- **Cursor.so 邮件：钓鱼尝试？**：一位用户质疑来自 **@cursor.so** 域名的邮件的合法性，怀疑是钓鱼尝试。
   - 虽然最初被标记为疑似虚假，但官方渠道确认这是 **Cursor** 使用的合法邮箱地址，尽管其官方域名为 **.com** 和 **.sh**。
- **Gemini 2.5 Pro 定价公布**：[Gemini 2.5 Pro 定价](https://x.com/legit_api/status/1908174018881933818)现已正式公布，对于 <200K tokens 的情况，费率为 **$1.25/1M input tokens** 和 **$10/1M output tokens**。
   - 定价根据 token 数量而变化，超过 200K tokens 的使用量费率更高；一些用户发现与其他模型相比，其价格出奇地实惠。
- **GPT-5 因优化推迟发布**：根据 [Sam Altman 的 X 帖子](https://x.com/sama/status/1908167621624856998)，**GPT-5** 将在 O3 和 O4-mini 发布后的“几个月内”推出。
   - 推迟发布旨在提升 **GPT-5** 的性能，解决集成问题，并确保有足够的容量来应对预期的需求。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 停用 Route Fallback 功能**：由于“混淆和不可预测性”，OpenRouter 团队正在移除 `route: "fallback"` 参数，建议用户手动将备选模型添加到 `models` 数组中，或者使用 `openrouter/auto`。
   - 这一更改影响了 OpenRouter 处理多个模型的方式，因为旧的自动回退选择方法将于下周弃用。
- **Gemini Pro 驱动 Missile Command**：一位用户通过 **Cloudflare AI Gateway** 将 **OpenRouter API** 集成到他们的 **Missile Command 游戏玩法 AI 摘要分析**中，结果[在此查看](https://missile-command-game.centminmod.com/)。
   - 用户分享了一张截图，显示 **Gemini Pro 2.5** 正在分析游戏玩法并为 **Atari Missile Command** 推荐策略，这有助于提高他们的排名。
- **DeepSeek 的折扣优势**：一位成员称赞了 **DeepSeek** 的定价，强调了在特定时段有 **75% 的折扣**，这与 **Anthropic** 和 **OpenAI** 模型的高昂成本形成鲜明对比。
   - 他们对这种成本效益表示满意，相比之下，将资源投入到更昂贵的替代方案中并不划算。
- **Gemini 2.5 Pro 实现正式发布 (General Availability)**：成员们讨论了 **Gemini 2.5 Pro** 的正式发布，引用了 [Google 的定价文档](https://ai.google.dev/gemini-api/docs/pricing?hl=de)。
   - 一位成员注意到可以通过 API 使用，但质疑它是否是*真正的 GA*。
- **OpenRouter 账户问题引发关注**：用户报告了在删除和创建账户时遇到的问题，包括 *User Not Found* 错误。
   - 建议的解决方案包括创建新的 API 密钥或尝试不同的浏览器，一位成员确认 *OR 目前不允许重新使用之前删除的账户*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 3 CUDA 异常问题未修复**：用户报告称，即使更新到最新的运行时版本，**Gemma 3 4b** 在使用 CUDA 时仍会抛出 `spits unused` 错误，且 CPU 性能不尽如人意。
   - 报告显示，更新到 **version 1.24.1** 并未解决 CUDA 相关问题。
- **LM Studio 导入 HuggingFace 模型**：根据 [LM Studio 文档](https://lmstudio.ai/docs/app/basics/import-model)，要将模型从 **HuggingFace** 导入 **LM Studio**，用户应使用 `lms import <path/to/model.gguf>` 命令。
   - 从 Hugging Face 下载的模型目录结构在导入 **LM Studio** 时会得到保留。
- **LM Studio 实现 n8n 集成**：**LM Studio** 可以通过使用 **OpenAI Chat Model** 节点连接到 **n8n**（一种工作流自动化工具），并在 base_URL 字段中填写 LM Studio 服务器 URL。
   - 这种集成之所以可行，是因为 **LM Studio 使用 OpenAI API**，使其能够与任何兼容 OpenAI 的工具对接。
- **Ollama 模型在 LM Studio 中：愿望落空**：尽管 **Ollama** 模型是 GGUF 格式，但由于 Ollama 的专有格式，它们与 [**LM Studio** 不兼容](https://ollama.com/)。
   - 这种不兼容性影响了在两个平台之间互换使用模型的能力。
- **LM Studio 隐藏路线图**：一位用户询问了包含 **LM Studio** 计划更新的路线图 (roadmap)，对潜在的 MCP 支持表示期待。
   - 回复确认目前没有公开的路线图。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo SIMD 规避系统障碍**：成员们讨论了 Mojo SIMD（如 [EmberJson 库](https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273) 所示）在 **基于 ARM 的 Mac 和 x86 桌面端** 之间提供了无缝的可移植性。
   - 与 [C++ `sonic-cpp` 库](https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59) 不同，后者需要针对特定架构进行重新实现以进行优化，而 Mojo 无需更改代码即可实现这一点。
- **Magic 包管理器让包管理更简单**：通过 *magic* 进行的 Mojo 包管理（位于 [builds.modular.com](https://builds.modular.com/?category=packages)）使编写和使用库变得更加容易。
   - 该包管理器允许毫不费力地创建和利用库。
- **斐波那契函数引发 stdlib 争论**：一个旨在向 stdlib 添加斐波那契函数的 [Pull Request](https://github.com/modular/max/pull/4280) 引发了关于其是否应被包含的辩论。
   - 虽然有人质疑其用途，但也有人指出它在 [Lean](https://leanprover-community.github.io/mathlib_docs/data/nat/fib.html) 等语言中也存在。
- **整数溢出需要监管**：斐波那契 PR 凸显了关于整数溢出行为的问题，并在 [论坛](https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202) 上进行了讨论。
   - Mojo 使用补码（two's complement），但变量位宽类型的处理仍未解决。
- **Mojo 的 Python 封装：仍是一个谜**：根据 **25.2 更新流**（[在此观看](https://youtu.be/dG0L1GalIHU?si=R1ae0xFoSDg99PMP&t=1775)），Mojo 的 Python 封装（wrappers）仍在开发中，尚未准备就绪。
   - 未提供更多细节，这让开发者们渴望获得更具体的信息。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **疑云笼罩 Google 的 AI 优势**：成员们对 **Google** AI 团队缺乏凝聚力的竞争优势表示担忧，有人认为 **DeepMind** 正在失去领先地位，并分享了一个讨论动态架构的 [Gemini 链接](https://g.co/gemini/share/6ab6889563cd)。
   - 讨论集中在具有短期和长期记忆的动态架构上，这些架构不同于僵化的 Tokenization 方法。
- **NLP Tokenization 面临僵化审查**：目前的 **NLP** 方法不自然地将语言强行纳入僵化的 Tokenized 格式，并分享了一个 [grok.com 链接](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866) 以支持动态系统应将语言视为结构化的、不断演变的信号的观点。
   - 围绕 Token Embedding 是否位于流形（manifold）上展开了辩论，引用了最近的一篇论文，该论文发现 Token Embedding 未能通过流形测试（[Token embeddings violate the manifold hypothesis](https://arxiv.org/abs/2504.01002)）。
- **AI 数学难题引发辩论**：一位成员表示，AI 模型在某些问题上挣扎并不奇怪，因为这些问题针对的是 **99.99 百分位技能水平**，甚至对许多 **数学博士** 来说也是挑战。
   - 他们承认，虽然目前的 AI 对这种水平的问题没有用处，但这并不会削弱其 *已经深远的实用性*。
- **Stability AI 推出虚拟相机**：Stability AI 推出了 [Stable Virtual Camera](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control)，这是一个研究预览版的多视图扩散模型，可将 **2D 图像转换为具有 3D 相机控制的沉浸式 3D 视频**。
   - 这允许从一个或多个输入图像中以用户指定的相机角度生成场景的新视角，从而产生 **一致且平滑的 3D 视频输出**。
- **Parquet 受困于瘫痪性的补丁拼凑**：发现了一个最高严重级别的远程代码执行（**RCE**）漏洞，追踪编号为 [CVE-2025-30065](https://nvd.nist.gov/vuln/detail/CVE-2025-30065)，影响 **Apache Parquet** 1.15.0 及之前的所有版本。
   - 该漏洞允许攻击者利用特制的 Parquet 文件控制目标系统，并已在 **Apache 1.15.1 版本** 中修复。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **精简的 RAG 代码令人惊叹**：成员们分享了仅需 **15-30 行代码**即可实现的 **RAG techniques**，利用 **MongoDB** 进行数据存储并结合 **OpenAI models**。
   - 一位成员指出，MongoDB 是 RAG 解决方案中首选的数据库。
- **HF Spaces 端口受限**：一位用户发现 **Hugging Face Spaces** 限制了端口 **80**、**443** 和 **8080** 的出站连接，导致其位于 **5432** 端口的 **Postgres database** 被拦截。
   - 另一位成员引用了 [Hugging Face 文档](https://huggingface.co/docs/hub/spaces-config-reference)，澄清该限制仅适用于 **Docker Spaces**。
- **HackXelerator 三城活动发布**：**London, Paris, Berlin AI HackXelerator™ - LPB25** 将黑客松与加速器结合，活动跨越 2025 年 4 月的 **20 天**，于 **2025 年 4 月 5 日**在伦敦启动，并于 **2025 年 4 月 25 日**在巴黎举行决赛。
   - 活动包括在柏林的赛后派对，并支持通过 [live-streams](https://www.youtube.com/@KXSB-cic) 进行全程在线参与。
- **按需付费推理不可用，建议使用 Ollama**：一位因每月推理额度耗尽而苦恼的用户寻求 **pay-as-you-go** 选项未果，随后有人建议使用本地模型如 **Ollama** 作为替代。
   - 一位成员提供了一个 [GitHub Gist 链接](https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed)，用于实现 **Ollama** 以替代 HfApiModel。
- **AI 脚本查找器**：一位成员在 Hugging Face Space 中部署了一个基于 AI 的 DBA 脚本检索工具：[sqlserver-lib-assistant](https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant)，该工具利用了 **ZeroGPU**、**Sentence Transformers** 和 **Azure SQL DB vector features**。
   - 该项目对 DBA 脚本进行索引并生成 embeddings，使用户能够通过自然语言提示词找到相关脚本；项目目前处于 'v1' 阶段，作者计划通过**更好的脚本分块 (chunking)** 和**训练特定模型**来增强功能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deepseek 发布亮眼的深度学习论文**：**Deepseek** 在 [arXiv](https://arxiv.org/abs/2504.02495) 上发布了一篇关于大规模 **Reinforcement Learning** 的新论文。
   - 论文研究了如何通过更多的 **inference compute** 来改进通用查询的 **reward modeling (RM)**，即 **generalist RM** 的推理时间扩展性，并引入了 **Self-Principled Critique Tuning (SPCT)** 作为一种学习方法来帮助提升性能与计算量的扩展。
- **基于 Prompt 的电影制作升温**：**AI Prompt Filmmaking** 领域正在取得进展，特别是 **Runway** 发布的 **Gen 4** 以及作为开源替代方案的 **Alibaba Wan 2.2** ([YouTube 链接](https://www.youtube.com/watch?v=Rcwfj18d8n8))。
   - 用户还在讨论用于表情包检索的工具，以及如何组织本地文件。
- **Cognition 推出 Agent 原生 IDE Devin 2.0**：Cognition Labs 推出了 **Devin 2.0** ([X/Twitter 链接](https://x.com/cognition_labs/status/1907836719061451067))，这是一种全新的 Agent 原生 IDE 体验，起售价为 20 美元。
   - 用户还在考虑文件整理工具，包括本地版本 ([Local File Organizer](https://github.com/QiuYannnn/Local-File-Organizer))，以及 **Llama-FS**——一个基于 Llama 3 的自组织文件系统 ([GitHub 链接](https://github.com/iyaja/llama-fs))。
- **LLM 捕获 PDF 用于后续标注**：成员们讨论了使用 **LLMs for extraction** 从非结构化 **PDFs** 中创建数据集，并指向了 [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B)，这是一个用于从原始文本创建合成指令微调数据集的指令生成模型。
   - 一位成员分享了旨在通过 **Ollama** 和多个 **PDFs** 快速使用 **Genstruct** 的 [GitHub 仓库](https://github.com/edmundman/OllamaGenstruct)，另一位成员成功使用 **Deepseek's API** 从财务公告中提取数据，但目标是微调一个专门用于提取的模型。
- **AI Agent 在替代版 X 上获得追随**：CamelAIOrg 发布了 [Matrix](http://matrix.eigent.ai/x)，这是一个社交模拟引擎，**AI agents 在其中回复、转发并争夺影响力**。
   - MooFeez 发布了 [Claude Squad](https://github.com/smtg-ai/claude-squad)，这是一个用于管理 **Claude Code & Aider tasks** 的管理器，可以在一个地方监督多个 Agent。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **“公牛在计算中胜过小鸡”**：一位成员引用了《计算机体系结构：量化研究方法》（*Computer Architecture: A Quantitative Approach*）来引发关于 [CPU vs GPU](https://www.techopedia.com/difference-between-cpu-and-gpu) 权衡的辩论。
   - 讨论的核心在于耕田时是使用“两头强壮的公牛还是 1024 只小鸡”，以此隐喻评估并行处理能力。
- **cuTILS 发布日期依然神秘**：成员们正焦急地等待今年早些时候在 GTC 上宣布的 **cuTILS** 的预计发布日期。
   - 尚无 Nvidia 员工评论其可用时间，这让想要尝试它的成员感到担忧。
- **探索通过 SSH 进行 CUDA 调试**：成员们讨论了通过 **SSH** 调试 **CUDA**，以避免耗时的调试重新编译，并指出 **CUDA gdb** 的工作方式与 GDB CLI 类似，Nvidia Insight 同样适用。
   - 一位成员推荐使用 **CUDA gdb**，而另一位建议通过 SSH 使用 Nvidia Insight，不过原帖作者并未说明他们更倾向于哪一个。
- **SYCL 是统一的 GPU 语言！**：虽然存在统一语言（**OpenCL** 以及现在的 **SYCL**），但并非主流，同时还提到了 **Kokkos**、**Alpaka**、**Raja**、**Vulkan Kompute** 和 **WebGPU**。
   - 另一位成员推测 **OpenCL** 未能成为主流是因为其“编程模型糟糕”。
- **关于 ReasoningGymDataset 定义的辩论**：成员们质疑为什么示例中都有各自的 **ReasoningGymDataset** 定义，而明明可以在[这里](https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py)进行统一。
   - 另一位成员回复称，目前的结构没有问题，因为 `/examples` 目录用于自包含的代码片段，而 `/training` 才是团队主要关注的地方。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **客户端热潮席卷 MCP**：开发者正在权衡构建 **MCP 客户端**与**服务器**的优缺点，客户端因其在**向量工具调用**和**基于资源的 RAG** 方面更高的灵活性而受到青睐。
   - 一位成员指出：“客户端比服务器端灵活得多”，而其他人则看到了在 Claude 之外运行服务器的好处，例如在 **Slack** 或 **Discord 机器人**上。
- **由 MCP 驱动的 React 代码生成**：利用 **MCP** 专家系统进行 **React 代码和测试生成**的热度很高，这将工作负载从 **LLM** 转移到了专业工具上。
   - 拟议的工作流使用 **MCP Server** 来验证、检查（lint）和格式化来自 **LLM** 的代码，并可能根据项目应用自定义规则。
- **OAuth 身份验证方案待定**：讨论内容包括在 [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70) 中为 **HTTPX** 添加 **OAuth 2.1** 身份验证客户端的拉取请求（PR）。
   - 一位成员还在编写关于服务器端身份验证的指南，详细说明如何使用 **governance SDK** 验证令牌并强制执行权限。
- **Datadog MCP 和 MCP Browser Kit 亮相！**：通过 [GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server) 引入了一个用于驱动浏览器的新 MCP 工具，以及另一个名为 [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit) 的 MCP 工具。
   - 一位成员在黑客松期间构建了一个针对 DX（开发者体验）优化的 **MCP Server 搜索**工具，访问地址为 [mcp-search.dev](https://mcp-search.dev/)。
- **MCP Omni Agent 防止工具中毒**：该 Agent 在调用任何工具之前，会清晰地解释其意图、请求用户许可并检查敏感访问权限。
   - 如果存在潜在风险，**Agent 会自动默认选择更安全的替代方案**。

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户反馈研究启动**：团队正在寻求研究参与者，以获取对早期阶段概念的反馈，并鼓励感兴趣的人员填写 [申请表](https://link.to.application.form)。
   - 团队正在持续寻求更多参与者加入该研究。
- **IntentSim.org 框架发布！**：一位用户推广了他们的新框架 [IntentSim.org](https://IntentSim.org)，也称为 **Information-Intent Nexus**，该框架利用了 **NotebookLM**。
   - 该项目旨在简化复杂信息系统中的意图识别。
- **Deep Search 覆盖芬兰**：一名成员询问了 **Deep Search** 功能的可用性，想知道它是否仅限于美国。
   - 另一名成员确认了该功能的推出，包括在芬兰的可用性。
- **PDF 理解变得更智能**：**NotebookLM** 宣布增强了对复杂 PDF 的理解能力，现在支持图像和图表。
   - 此次升级适用于通过链接添加的 PDF，并将扩展到所有直接上传的 PDF，**Gemini API** 现在也支持 Docs 和 Slides 的多模态分析。
- **NotebookLM 推出 Discover 功能**：NotebookLM 引入了 **Discover** 功能，允许用户描述一个主题并接收精选的网络资源；一名成员创建了一个 [视频演示](https://youtu.be/YP6fS5JtMkg?si=Gz-kUGJGtyh2_f9e)，展示了该新功能的实际工作流程。
   - 这一新功能有望简化平台内的研究和信息收集流程。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OpenThinker2 模型实现跨越式进步**：根据 [一篇博客文章](https://www.openthoughts.ai/blog/thinkagain)，新的 **OpenThoughts-1M** 和 **OpenThinker2-32B/7B** 模型在仅对 Qwen 2.5 32B Instruct 进行 **SFT** 的情况下，性能超过了 **R1-Distilled-32B**。
   - 模型和训练数据集已在 Hugging Face 上发布（[OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B), [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)）。
- **推理模型需要奖励**：一名成员询问了创建推理模型的挑战，得到的建议是探索 *持续学习（continual learning）文献*，以强调主要挑战在于为 **RL** 寻找 **合适的环境** 以及 **合适的奖励/性能评估**。
   - 另一名成员分享了 **MoE++** 的链接，这是一个异构的 Mixture-of-Experts 框架，与原生 MoE 模型相比，它增强了性能并提供了 **1.1-2.1倍** 的专家前向吞吐量，可在 [OpenReview](https://openreview.net/forum?id=t7P5BUKcYv) 上查看。
- **Monkeys 揭示测试时（Test-Time）真相**：一篇新的预印本论文 [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) 探讨了语言模型中的 **推理（inference）** 和 **测试时缩放（test-time scaling）**，特别是成功率如何随每个任务的多次尝试而缩放。
   - 研究发现了一个谜题：每个问题的失败率随尝试次数呈指数级下降，但总成功率却遵循多项式缩放定律，研究将其归因于单次尝试成功概率的 **重尾分布（heavy-tailed distribution）**。
- **对比集引导方向向量（Steering Vectors）**：一名成员建议，让预训练模型从训练数据中挑选出 **对比集（contrastive sets）** 来构建方向向量，然后控制方向向量的系数，这可能会很有趣。
   - 另一名成员提到了 [David Bau 及其团队关于“功能向量（function vectors）”的论文](https://arxiv.org/abs/2310.15213)，该研究发现 Attention Heads 传输了所演示任务的紧凑表示。
- **EOS Token 阻碍 Harness**：一名成员询问在 **lm-eval-harness** 中为 **social_iqa 任务** 的数据实例添加 EOS Token 的问题，并指出强制添加后准确率下降了 **18 个百分点**。
   - 一名成员建议在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364) 为多选题变体向 `continuation_enc` 添加 `self.eot_token_id`，并为 **BOS** 传递 `add_bos_token`。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **请求重组聊天列表**：一位用户建议根据最近的编辑日期而非创建日期来重组聊天，主张这是一种更相关的列表排序方法。
   - 该用户批评当前基于创建时间的先后顺序*有点随意*。
- **寻求用于价格提取的轻量级模型**：一位成员正在寻找一种专门用于从字符串中提取 **price**（价格）值的轻量级模型，发现 Regex 解析不足以处理多样化的用户输入。
   - 建议包括研究 Hugging Face 上可用的 **embedding models** 或具有 *extraction*（提取）能力的模型。
- **GPT4All 陷入沉默**：一位成员对 **GPT4All** 最近缺乏沟通表示质疑。
   - 另一位成员声称 **GPT4All** *多年来不与普通用户交流，也不接受建议*。
- **Gemini 2.5 Pro 被推崇用于编程**：一位成员推崇 **Gemini 2.5 Pro** 适用于编程和数学应用，强调其拥有高达 **100 万 Token 的上下文窗口**。
   - 他们强调其目前可以**免费**使用，包括其 **API**。
- **GPT4All 的沉寂期引发好奇**：一位成员观察到 **GPT4All** 相对沉默，同时在等待下一个版本以及 **Nomic Embed Text V2** 的集成。
   - 未分享更多额外信息。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Packed Datasets 大幅提升速度**：一位成员建议使用 **packed datasets** 以避免 `seqlen=49` 的 bug，并通过打包句子直到达到 `max_seq_len` 来提高速度，从而避免浪费 Padding Token。
   - 要启用此功能，用户可以设置 `dataset.packed=True` 和 `tokenizer.mas_seq_len=<your-max_seq_len, 例如 8096>`，并利用针对 Attention 的 **group masking**，详见 [PR #2560](https://github.com/pytorch/torchtune/pull/2560)。
- **分块职责转移**：分块（Chunking）的职责正通过 `loss = loss_fn(model.weight, logits, labels)` 转移到 Loss Function 中，以方便调试。
   - 创建了一个新文件 `torchtune.utils._tensor_utils.py`，其中包含对 `torch.split` 的封装并涵盖了单元测试，该文件将需要进行合并。
- **NeMo 的弹性训练解决崩溃问题**：一位成员参加了“使用 NeMo 进行弹性训练（Resilient Training with NeMo）”会议，并分享了关于 **NeMo** 如何解决任务崩溃和 GPU 时间浪费原因的见解，强调该主题与 torchtune 非常接近。
   - NeMo 的方法包括 **fault tolerance（容错）、straggler detection（掉队检测）、asynchronous checkpointing（异步检查点）、preemption（抢占）、in-process restart（进程内重启）、silent data corruption detection（静默数据损坏检测）以及 local checkpointing（本地检查点）**等特性，但部分功能尚未实现。
- **AI-2027 报告警告超人类 AI 即将到来**：一位成员分享了 [AI-2027 报告](https://ai-2027.com/) 的链接，该报告预测 **superhuman AI**（超人类 AI）在未来十年的影响将是巨大的，超过**工业革命**。
   - 该报告基于趋势外推、兵棋推演、专家反馈、在 **OpenAI** 的经验以及之前的预测成功案例。
- **CEO 们预测 2027 年将实现超人类 AI**：**OpenAI**、**Google DeepMind** 和 **Anthropic** 的 **CEO** 们认为，AI 可能会在 2027 年超越人类智能。
   - 一位成员询问是否使用了 AI 来编写 [AI-2027 网站](https://ai-2027.com/)上滚动更新的实时图表。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LeetGPU 对 tinygrad 的支持展望未来**：成员们讨论了 [leetgpu.com](https://leetgpu.com) 及其未来对 **tinygrad** 的潜在支持，但未提供有关时间表或支持范围的具体细节。
   - 一位成员询问了是否有计划扩大消费级 GPU 的可访问性，并提供易用的 API，以便进行本地 **tinygrad** 开发。
- **华为 Ascend 卡吸引 tinygrad 开发者**：一位成员提议提供 **Huawei Ascend** 卡的使用权限用于开发，George Hotz 对此表示感兴趣，并询问了购买选项或云端机器的可用性。
   - 这可能会将 **tinygrad** 的硬件支持和优化工作扩展到 **Huawei** 的架构。
- **WEBGPU BEAM 触发调用限制**：在使用 `BEAM=2` 为 **WEBGPU** 编译 **tinygrad** 模型时，用户发现需要将 `requiredLimits.maxComputeInvocationsPerWorkgroup` 增加到 **512**，这降低了对 Android 设备的支持。
   - 一个 [PR](https://github.com/tinygrad/tinygrad/pull/9085) 和一个 [热修复分支](https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py) 建议设置 `IGNORE_BEAM_CACHE=1` 或实现一种通用的限制机制来解决此问题。
- **George Hotz 重新实现 tinygrad 版 Karpathy GPT**：George Hotz 在“刚开始上手 **tinygrad**”时重新实现了 Karpathy GPT。
   - 一位在 **METAL** 上运行该实现的向用户报告了由于 **32 buffer 限制** 导致的 `tinygrad.device.CompileError`，并寻求处理该约束的建议，同时链接到了他们的 [main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 支持多模态聊天历史**：**LlamaIndex** 现在支持 **多模态聊天历史**，使多 Agent 系统能够处理交替的文本和图像消息，详情见 [此推文](https://twitter.com/llama_index/status/1908191704156700682)。
   - 更新后的系统利用 [ReAct agent loop](https://t.co/EKIZiZJS2P) 促进 Agent 对图像和文本进行推理。
- **研究员寻求 PatentsView API**：一位社区成员请求 **PatentsView 联系人** 提供 **API key**，以收集用于 **RAG** 实现的初始数据。
   - 目标是利用 **PatentsView API** 在 **RAG** 框架内增强数据检索和分析。
- **Workflow 转化为 Tool**：一位社区成员提议通过将 **Workflow** 集成到 **FunctionTool** 中来将其转化为 **Tool**。
   - 他们展示了一个代码片段，使用 `async def tool_fn(...)` 定义工具功能，随后通过 `FunctionTool.from_defaults(tool_fn)` 创建工具，这允许指定名称、描述、输入注解和返回值。
- **LlamaParse 面临图像理解难题**：一位用户报告称 **LlamaParse** 在读取图表/图像时遇到困难，虽然能提取文本但无法解释图像本身，即使使用了 **LVM** 和高级模式（Premium mode）。
   - 一份澄清回复指出，**LlamaParse** 无法处理没有可提取文本的图像，但可以将图像作为 artifact 检索出来以便进一步处理，例如提示 LLM 对其进行描述。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AYA Vision 在 waves.jpg 上遇到困难**：一位用户报告称，**AYA vision** 在分析 *waves.jpg* 图像时返回了 **400 错误**，提示 *不支持的图像文件格式*，尽管 **AYA** 成功分析了其他 **JPG** 图像。
   - 错误消息指出仅支持 **PNG, JPEG, WebP, 和 GIF** 格式，这表明特定的 **JPG** 文件或 **AYA** 的格式检测可能存在问题。
- **AYA Vision Bug 疑似与 Bedrock 有关**：一位用户在发生错误时看到了 *coco.py: AWS Bedrock Command A*，这可能暗示在上传图像时与 **AWS Bedrock** 有关联。
   - 目前尚不清楚这是 **AYA** 流水线的一部分，还是图像分析过程中不相关的错误。
- **全栈专家展示技能**：一位拥有 **8 年以上经验** 的全栈开发人员介绍了自己，重点介绍了在 **React, Angular, Flutter, Swift, Python, TensorFlow, 和 OpenAI** 方面的专业知识。
   - 他们曾参与电子商务、医疗保健和金融科技领域的高影响力项目，集成了 **Cloud Technologies, Microservices, 和 DevOps**。
- **分析师计划撰写 AI 文章**：一位正在求职空窗期的前产品分析师正在探索撰写关于技术和 AI 的文章。
   - 他们正在寻找志同道合的人一起交流，探讨技术如何塑造我们的世界或 AI 的实际用途，感觉自己*陷入了信息茧房*。
- **Web3 专家拥抱 AI**：一位在全栈/AI 开发方面拥有 **7 年以上经验** 的 **Web3/AI 工程师** 介绍了自己。
   - 他们专注于将 **AI 与 Automation** 相结合，并渴望以信心和创新帮助企业。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 即将支持 Asyncio**：一位成员询问了为通用 **DSPy** 调用添加 **asyncio** 支持的计划。
   - 他们举例说明了从轻量级 **DSPy** 功能开始，随后扩展到优化的使用场景；在需要 **DSPy** 功能之前，他们一直使用 *litelm*，并对未来的支持表示好奇。
- **用于轻量级 DSPy 的 LiteLLM**：讨论强调了一种模式，即从类似于使用 *LiteLLM* 的轻量级 **DSPy** 功能开始，随着项目的发展过渡到 **DSPy** 的优化能力。
   - 这表明在轻量级 **DSPy** 使用与完整的优化工作流之间，可能需要无缝集成或功能对等。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 升级后性能提升**：根据 [Windsurf 的公告](https://x.com/windsurf_ai/status/1907902846735102017)，**DeepSeek-V3** 模型已升级至 **DeepSeek-V3-0324**，在内部测试中表现出更好的性能。
   - Windsurf 团队发布了一个俏皮的请求，希望大家收藏该公告帖子以获取进一步的更新和支持。
- **Windsurf 预告 DeepSeek-V3 升级**：Windsurf AI 在 [X/Twitter](https://x.com/windsurf_ai/status/1907902846735102017) 上宣布了 **DeepSeek-V3** 模型的升级，提到新版本为 **DeepSeek-V3-0324**。
   - 公告暗示根据内部评估，性能有轻微提升。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 等待进一步测试**：一位成员提供了关于 **Gorilla LLM** 和 **Berkeley Function Calling** 的协助。
   - 他们确认已准备好根据需要回答问题、进行调整或重新测试。
- **向 robotsail 提供进一步支持**：Robotsail 为 **Gorilla LLM** 和 **Berkeley Function Calling** 提供了支持。
   - Robotsail 乐意回答任何问题并准备好进行重新测试。

---

**LLM Agents (Berkeley MOOC) Discord** 频道没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 频道没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 频道没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357430220457513052)** (1329 条消息🔥🔥🔥): 

> `更快的推理 vs 更聪明的模型, 上下文长度限制, 模型蒸馏, 超快模型, LLM 与感知能力`

- **牺牲智能换取速度？**：成员们讨论了未来的 AI 发展应该侧重于 **faster inference** 还是**更智能的模型**，考虑到 **o4-mini** 和 **o3** 的同时发布，引发了关于 **OpenAI** 是否发现了新的 **inference** 技术的疑问。
   - 一位成员建议 **long context** 和速度是最佳选择，并思考 **2 million** tokens 是否可能是上下文限制，而另一位成员则对看到 **10 million** tokens 感到兴奋。
- **Groq 硬件：OpenAI 错失的机会？**：参与者讨论了模型大小、速度和智能之间的权衡，一些人认为**更小的模型**意味着**更少的知识**，除非模型经过了 **distilled**。
   - 提到 **Groq** 专门为 **AI inference** 开发了硬件，一位成员对 **OpenAI** 尚未收购 **Groq** 表示惊讶。
- **AI Sentience：依然是热门话题**：对话涉及了 **LLM** 是否能实现 **sentience**，尽管参与者指出，定义 **sentience** 或 **consciousness** 是回答这个问题的前提。
   - 一位用户开玩笑说，如果 **LLM** 在人类之前实现意识，那将是 **AGI**，而另一位则建议，如果一个 AI 能说服某人它具有 **sentient**，那么这种区别可能就不重要了。
- **Gemini 的音乐杰作**：一位成员分享了由 **Gemini** 生成的音乐，称其*部分有趣*，并提供了一个 [.mid 文件的链接](https://cdn.discordapp.com/attachments/1340554757827461211/1357750989133844632/piano_evocation.mid?ex=67f157a5&is=67f00625&hm=dd212c426d40593e295b8496363afc4427848309c49a095ca77a715d6260b973&)。
   - 他们提示 **Gemini** 使用一个基于 **python** 的转换工具，创作一首风格类似于 **Vangelis** 和 **Jarre** 等作曲家的钢琴曲。
- **NightWhisper 的编程实力**：成员们讨论了 **NightWhisper** 模型，一些人认为它在编程方面可能优于 **Gemini 2.5 Pro exp** 和 **Claude 3.7 Sonnet thinking**，并专攻 **webdev** 和 **UI/UX**。
   - 一位成员指出 **OpenAI** 宣布他们将在几周内发布此模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/emollick/status/1908220677502755328">来自 Ethan Mollick (@emollick) 的推文</a>：更新了这张包含最新 Gemini 的图表。它展示了 AI 在不到两年时间里的飞速进展：GPT-4 级别模型的成本下降了 99.7%，甚至世界上最先进的模型也...</li><li><a href="https://x.com/Copilot/status/1908187808813940799">来自 Microsoft Copilot (@Copilot) 的推文</a>：观看太平洋时间上午 9:30 在 YouTube 上举行的直播活动，了解我的所有新功能。</li><li><a href="https://x.com/iruletheworldmo/status/1908188856039391310">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>：o4 4月17日。</li><li><a href="https://x.com/testingcatalog/status/1908199211473977523">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：所有功能的公告已经发布 - Memory 🔥- Actions 🔥- Copilot Vision 🔥- Pages 🔥- Podcasts 🔥- Shopping- Deep Research 🔥- Copilot Search https://blogs.microsoft.com/blog/2025/04/04...</li><li><a href="https://x.com/DeryaTR_/status/1908247941602828342">来自 Derya Unutmaz, MD (@DeryaTR_) 的推文</a>：来自 @GooglAI 的 Gemini 2.5 Pro 现在是最智能的 AI 模型，在离线测试中 IQ 接近 120。这使其处于人类 IQ 的中上水平。我怀疑即将推出的 o3-pro ...</li><li><a href="https://x.com/paulgauthier/status/1907996176605220995">来自 Paul Gauthier (@paulgauthier) 的推文</a>：@OpenRouterAI 上神秘的 Quasar Alpha 在 aider 多语言编程基准测试中获得了 55% 的分数。这与 o3-mini-medium、最新的 DeepSeek V3 以及旧版 Sonnet 3.6 (20241022) 具有竞争力。Quasar Al...</li><li><a href="https://x.com/legit_api/status/1908268939177533808">来自 ʟᴇɢɪᴛ (@legit_api) 的推文</a>：我相信 nightwhisper 是 2.5 Pro 的下一个版本，或者是 2.5 系列中更强大的模型 - ultra 什么时候出？🧐在过去的一两天里，我对这个模型进行了广泛的评估，我可以自信地说...</li><li><a href="https://openrouter.ai/chat?models=openrouter/quasar-alpha">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://x.com/legit_api/status/1908264443827351913?s=46">来自 ʟᴇɢɪᴛ (@legit_api) 的推文</a>：nightwhisper 已经离开了 Arena 👀这款能力惊人的编程模型 ^Veo 2 正在为 AI Studio 和 Gemini API 做准备。</li><li><a href="https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo">跨越对话语音的恐怖谷</a>：在 Sesame，我们的目标是实现“语音临场感”——这种神奇的特质让语音交互感觉真实、被理解且有价值。</li><li><a href="https://x.com/tokumin/status/1908315418458284441?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Simon (@tokumin) 的推文</a>：@legit_api 是的，它很棒</li><li><a href="https://replit.com/">Replit – 利用 AI 构建应用和网站</a>：Replit 是一个由 AI 驱动的平台，用于构建专业的 Web 应用和网站。</li><li><a href="https://twostorymelody.com/best-websites-to-download-midi-files/">8 个下载 MIDI 文件的最佳网站 | Two Story Melody</a>：合适的 MIDI 文件可以让你的下一条音轨更有趣。这里有一些获取它们的好网站。</li><li><a href="https://www.reddit.com/r/askscience/comments/1xwx0k/do_neurons_operate_in_a_fundamentally_different/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://www.videolan.org/vlc/">VLC 媒体播放器官方下载，最佳开源播放器 - VideoLAN</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357430862815297626)** (852 条消息🔥🔥🔥): 

> `Manus credits, Open Manus GUI, Gemini vs. Claude, Prompt engineering tips, Alternative AI tools`

- ****积分紧缩困扰社区****：用户对 Manus 上的**积分成本和消耗**表示担忧，一些人觉得积分消耗太快，即使是简单的任务也是如此，且目前的定价模型可能并不理想；初始的 1000 个免费积分是一次性的。
   - 普遍观点认为，为免费用户提供**每日一个任务**的选项将是一个有益的折中方案，一些社区成员还提供了 Prompt 指南。
- ****OpenManus GUI 现身****：一名用户正在开发 **OpenManus GUI** ([image.png](https://cdn.discordapp.com/attachments/1349440650495398020/1357524168715010129/image.png?ex=67f12d27&is=67efdba7&hm=a0ade4f56609638bf8591f9fb3db24dd5f1e1ff4213f36ab44433d679fa74235&))，旨在完全兼容未来的更新，并专注于用户友好的界面。
   - 该 GUI 将允许用户直接编辑配置，并可能加入用例部分和模板，但由于 OpenManus 缺乏历史记录系统，聊天记录的实现仍是一个挑战。
- ****Gemini 崭露头角，挑战 Claude 的代码霸主地位****：目前正在进行关于 **Gemini 和 Claude** 在编程任务方面的对比讨论，一些用户发现 Gemini 在某些语境下的输出更优，特别是在 DeepSeek 表现不佳的地方。
   - 一位用户强调，特别是 Gemini 2.5，以能够根据*只要你能写出 Prompt，就能生成任何你梦想的代码*而闻名，但其他人警告说 Google 在闭环中运行。
- ****Prompt 精雕细琢，性能优先****：用户分享了关于 **Prompt Engineering** 的技巧以优化积分使用，包括多 Prompt 大纲策略和使用清晰、分步骤的方法；一位用户分享了一个有用的 [TheNewOptimal.md 文件](https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md) 用于创建 LLM。
   - 诸如 **LLMLingua** ([microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)) 之类的压缩技术也被讨论作为减少 Token 消耗的一种方式。
- ****探索 AI 新前沿****：成员们讨论了 **Genspark** ([genspark.ai](https://genspark.ai)) 作为 Manus 潜在替代方案的优缺点，注意到它没有付费墙且能有效处理图像和视频，但也指出了对其背景可靠性的担忧，认为这可能是一家来自中国的公司。
   - 几位社区成员表示，*目前还没有 Manus 的替代品*，但许多人表达了对解决当前高积分消耗和资源可用性问题的渴望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://tenor.com/view/calligraphy-write-fantastic-day-good-day-handwriting-gif-5234981">书法书写 GIF - 书法书写 Fantastic Day - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sad-cry-tears-gif-9502436635084812836">悲伤哭泣 GIF - 悲伤哭泣眼泪 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/happy-homer-simpson-gif-4208056458079004156">快乐的 Homer GIF - 快乐的 Homer Simpson - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/welcome-to-the-team-team-work-picking-teams-gif-13020351">欢迎加入团队团队合作 GIF - 欢迎加入团队团队合作挑选队伍 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">篮球 NBA GIF - 篮球 NBA 勇士队 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/chihuahua-cane-recensionivere-cosa-what-gif-7417567602944477819">吉娃娃犬 GIF - 吉娃娃犬 Recensionivere - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bourdieu-jesus-facepalm-jesus-fucking-christ-gif-5392529">Bourdieu 耶稣 GIF - Bourdieu 耶稣捂脸 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/joe-biden-gif-18249938">Joe Biden GIF - Joe Biden - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/leeroy-jenkins-shovel-lethal-company-yolo-yeet-gif-12562950503852552482">Leeroy Jenkins 铲子 GIF - LEEROY JENKINS 铲子 Lethal Company - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/just-once-sungwon-cho-prozd-only-once-once-gif-8847803260919815815">就这一次 Sungwon Cho GIF - 就这一次 Sungwon Cho Prozd - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/thumbs-up-good-job-spongebob-gif-14856720698504163537">竖起大拇指做得好 GIF - 竖起大拇指做得好海绵宝宝 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/welcome-to-the-team-gif-18169063846751286454">欢迎加入团队 GIF - 欢迎加入团队 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ratatouille-remy-pure-poetry-poetry-perfect-gif-25629423">料理鼠王 Remy GIF - 料理鼠王 Remy 纯粹的诗意 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/what-what-the-what-the-hell-wtf-gif-8562994133044611418">什么什么 GIF - 什么什么搞什么鬼 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/its-friday-good-morning-its-friday-good-morning-it%27s-friday-gif-13884598500941796182">今天是周五早安今天是周五 GIF - 今天是周五早安今天是周五早安今天是周五 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/thanks-for-c-wcth-nathan-kevin-mcgarry-when-calls-the-heart-gif-17182293">感谢 C Wcth GIF - 感谢 C Wcth Nathan - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/choice-whichone-gif-18167902">选择哪一个 GIF - 选择哪一个 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fun-pet-pet-fun-high-tired-so-tired-gif-24511981">有趣的宠物宠物乐趣 GIF - 有趣的宠物宠物乐趣 High - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hey-girl-sliding-into-your-d-ms-like-sliding-into-d-ms-into-your-d-ms-like-roller-skate-gif-14532622">嘿女孩滑入你的私信 GIF - 嘿女孩滑入你的私信就像滑入私信 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/whats-up-gif-yo-whats-up-bro-xion-burnt-gif-3521580937773229484">怎么了 Gif Yo GIF - 怎么了 Gif Yo 怎么了兄弟 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fingers-crossed-luck-please-hope-gif-16374730">祈祷好运 GIF - 祈祷好运拜托了 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/inthehouse-martin-martinlawernce-biggie-hello-gif-13128531067958866971">Inthehouse Martin GIF - Inthehouse Martin Martinlawernce - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/allenai/dolma">GitHub - allenai/dolma: 用于生成和检查 OLMo 预训练数据的数据和工具。</a>: 用于生成和检查 OLMo 预训练数据的数据和工具。 - GitHub - allenai/dolma: 用于生成和检查 OLMo 预训练数据的数据和工具。</li><li><a href="https://github.com/allenai/olmocr">GitHub - allenai/olmocr: 用于为 LLM 线性化 PDF 的工具包</a>

<li><a href="https://github.com/allenai/olmocr">atasets/training</a>: 用于为 LLM 数据集/训练将 PDF 线性化的工具包 - allenai/olmocr</li><li><a href="https://manus.im/share/y2v6FBkLk7h0vCYmnSKrAn?replay=1">Environmental Impact of Overdevelopment in Brevard County - Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: [EMNLP&#39;23, ACL&#39;24] 为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。</a>: [EMNLP&amp;#39;23, ACL&amp;#39;24] 为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩...</li><li><a href="https://ucebdqhq.manus.space/">Iterative Development with Manus AI: A Comprehensive Guide</a>: 未找到描述</li><li><a href="https://sknlgdpd.manus.space/">Mastering Manus: A Comprehensive Guide</a>: 学习如何通过有效的 Prompt 编写、错误预防等方式，在与 Manus AI 交互时获得最佳结果。</li><li><a href="https://jmiivdli.manus.space/">Manus Guide - A Comprehensive Guide</a>: 未找到描述</li><li><a href="https://github.com/NathanielEvry/toroidal-rangers-assembly">GitHub - NathanielEvry/toroidal-rangers-assembly</a>: 通过在 GitHub 上创建账户，为 NathanielEvry/toroidal-rangers-assembly 的开发做出贡献。</li><li><a href="https://github.com/NathanielEvry/toroidal-rangers-assembly/blob/main/manifesto/ethos/toroidal-rangers-assembly.md">toroidal-rangers-assembly/manifesto/ethos/toroidal-rangers-assembly.md at main · NathanielEvry/toroidal-rangers-assembly</a>: 通过在 GitHub 上创建账户，为 NathanielEvry/toroidal-rangers-assembly 的开发做出贡献。</li><li><a href="https://manus.im/share/mpjQisfmjLKOw8T58sT9T4?replay=1">Zacarías Cocina de Mercado  - Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://www.usgs.gov/products/web-tools/apis">APIs</a>: API 代表应用程序编程接口，为开发者提供对专有软件应用程序的编程访问。API 是一种使应用程序编程成为可能的软件...</li><li><a href="https://www.usgs.gov/faqs/what-a-geographic-information-system-gis#:~:text=A%20Geographic%20Information%20System%20(GIS)%20is%20a%20computer%20system%20that,Where%20are%20USGS%20streamgages%20located%3F">What is a geographic information system (GIS)?</a>: 地理信息系统 (GIS) 是一种用于分析和显示地理参考信息的计算机系统。它使用附加到唯一位置的数据。大部分信息...</li><li><a href="https://balalm.com/product/2-in-1-oil-sprayer-bottle-bbq-cooking-oil-dispenser-olive-oil-pourers-sprayer-kitchen-baking-oil-mister-vinegar-bottle/">2 In 1 Oil Sprayer Bottle BBQ Cooking Oil Dispenser Olive Oil Pourers Sprayer Kitchen Baking Oil Mister Vinegar Bottle - My Blog</a>: 概览：1. 自动开合：橄榄油喷雾瓶让你单手即可倒油。它采用智能设计，倾斜时开启，直立时关闭。你不需要...</li><li><a href="https://balalm.com/product/350ml-electric-juicer-blender-mixer-usb-rechargeable-machine-household-portable-blender-maker-cup-kitchen-tool-kit/">350ML Electric Juicer Blender Mixer USB Rechargeable Machine Household Portable Blender Maker Cup Kitchen Tool Kit - My Blog</a>: 概览 刀片设计：用于奶昔和果昔的便携式搅拌机拥有强劲的电机底座和 4 个食品级不锈钢 3D 刀片。刀头采用 SUS304 不锈钢材质...</li><li><a href="https://balalm.com/product/cabinet-door-kitchen-waste-garbage-bin-toilet/">Cabinet Door Kitchen Waste Garbage Bin Toilet - My Blog</a>: 产品信息：材质：TPR, PP；重量：0.53 (kg)；容量：8L；功能：收纳桶；开合方式：无盖；形状：方形；颜色：灰色-大号，米色-大号；包装...</li><li><a href="https://balalm.com/product/chopper-stainless-steel-household-fast-meat-slice-multi-function/">Chopper Stainless Steel Household Fast Meat Slice Multi-function - My Blog</a>: 产品信息：颜色：黑色、白色；规格：34*12.5 * 8CM；适用送礼场合：员工福利；材质：ABS 不锈钢；风格：现代简约；装箱清单：切肉机...</li><li><a href="https://balalm.com/product/cotton-and-linen-storage-containers/">Cotton And Linen Storage Containers - My Blog</a>: 产品信息：用途：脏衣收纳；颜色：黑格子、灰箭头、蓝条纹、咖啡色格子、粉格子、灰格子、绿...</li>

aid, red stripes 规格：35cm x 45cm 材质：ca...</li><li><a href="https://balalm.com/product/electric-gravity-pepper-grinder-salt-grinder-adjustable-coarseness/">电动重力胡椒研磨器 盐研磨器 可调节粗细 - My Blog</a>：未找到描述</li><li><a href="https://balalm.com/product/fish-shaped-waffle-pan-maker/">鱼形华夫饼锅具 - My Blog</a>：概览：易于清洁和保存。它是厨房的好帮手。双盘设计，方便实用，安全健康，用于烹饪鲷鱼烧。规格：产品类别...</li><li><a href="https://balalm.com/product/multi-functional-vegetable-cutter-hand-drum-vegetable-cutter-slice/">多功能切菜机 手摇滚筒切菜机 切片 - My Blog</a>：产品信息：材质：塑料 颜色：白色、红色 包装清单：切菜机 * 1 套
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357429688514904150)** (245 条消息🔥🔥): 

> `VRAM 价格合理性、4-bit QAT、phi-4 的 ZeroDivisionError、Llama3.2 的训练损失值、Phi-4 模型问题` 


- **成员们讨论 VRAM 成本**：一些成员讨论了 VRAM 的高昂成本以及它为何物有所值。
   - 一位成员表示：*是的，听起来可能很贵，但 **VRAM** 让它物有所值*。
- **Phi-4-mini-instruct 问题**：成员们报告在尝试微调 **Phi-4 mini instruct** 时遇到了 **ZeroDivisionError**。
   - 该问题出现在尝试微调 **Phi-4-mini-instruct** 而非 "**unsloth/Phi-4**" 模型时，错误源于未设置 tokenizer 的 chat template。
- **Unsloth 尚不支持序列分类... 暂时如此**：Unsloth 目前原生不支持序列分类，但一位成员添加了该功能。
   - 这是添加 `automodel = AutoModelForSequenceClassification` 新功能的 [PR#2263](https://github.com/unslothai/unsloth/pull/2263) 链接。
- **利用 Deepseek Effect 在本地运行 DeepSeek**：一位成员报告尝试在本地运行 DeepSeek-R3-0324。
   - 另一位成员指出，由于 **Deepseek Effect**，该模型非常庞大，因此*无法进行微调*。
- **Gemma3 训练参数**：一位成员询问了使用 Unsloth 微调模型以解决多项选择题的数据格式。
   - 另一位成员建议在训练前学习 LLM 的基础知识，并推荐了 Karpathy 的从零开始构建 gpt2 课程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">教程：如何本地运行 DeepSeek-V3-0324 | Unsloth 文档</a>: 如何使用我们的动态量化（dynamic quants）在本地运行 DeepSeek-V3-0324，该技术可恢复精度</li><li><a href="https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md#output-layer-pruning">PurpleLlama/Llama-Guard3/1B/MODEL_CARD.md at main · meta-llama/PurpleLlama</a>: 用于评估和提高 LLM 安全性的一套工具。通过在 GitHub 上创建账户，为 meta-llama/PurpleLlama 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/127">[功能请求] DDP · Issue #127 · unslothai/unsloth</a>: 想为此创建一个 issue，而不是一直在 Discord 中询问。我看到了另一个关于多 GPU fp16 训练的工单，那也很棒。但 DDP 将允许用户扩展训练规模，从而可以...</li><li><a href="https://github.com/unslothai/unsloth/issues/2101#issuecomment-2768479825">TypeError: 全量微调 gemma3 时出现不支持的运算符类型 /: 'Tensor' 和 'NoneType' · Issue #2101 · unslothai/unsloth</a>: Version pip install unsloth pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 Code from unsloth import FastModel import torch model, tokenizer = FastModel.from_pretrained(...</li><li><a href="https://github.com/unslothai/unsloth/issues/1548">BLEU 分数 · Issue #1548 · unslothai/unsloth</a>: 你好！祝你有美好的一天！在微调 LLAMA3.2 的过程中，我尝试实现 compute_metrics 函数，但在训练期间，第一次尝试通过评估步骤时，发生了一个错误：...</li><li><a href="https://github.com/unslothai/unsloth/pull/1664">由 pluesclues 提交的用于 RL 更新的更稳健的 Online DPO 更改 · Pull Request #1664 · unslothai/unsloth</a>: 我希望这个能得到审查，所以我认为至少我拥有的带有 Llama 模型示例的 Online DPO 初步框架实际上可以正式配合 RL 更新工作。我将致力于其他...</li><li><a href="https://github.com/unslothai/unsloth/pull/2263">feat: 支持自定义 `auto_model` 以实现更广泛的模型兼容性 (Whisper, Bert 等) 以及 `attn_implementation` 支持，由 Etherll 提交 · Pull Request #2263 · unslothai/unsloth</a>: feat: Support custom auto_model, Whisper params, and attn_implementation。此 PR 增强了 FastModel.from_pretrained 以支持更广泛的模型：Custom auto_model：允许指定精确的...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">由 shashikanth-a 提交的增加对 Apple Silicon 支持 · Pull Request #1289 · unslothai/unsloth</a>: #4 未优化。尚不支持 gguf。从源码构建 Triton 和 bitsandbytes：cmake -DCOMPUTE_BACKEND=mps -S . 用于 bitsandbytes 构建；pip install unsloth-zoo==2024.11.4；pip install xformers==0....</li><li><a href="https://github.com/unslothai/unsloth/blob/bb112e38ef3f0dafa9e87faf55a6ba7499bd0357/unsloth/models/llama.py#L1604-L1610">unsloth/unsloth/models/llama.py 位于 bb112e38ef3f0dafa9e87faf55a6ba7499bd0357 · unslothai/unsloth</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1, Gemma 3 和 Reasoning LLMs！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357476874766254121)** (8 条消息🔥): 

> `Vibe coding, Jailbreaking 4o, ChatGPT uncensored` 


- **Vibe coding 提升专注力**：一位成员表示 *Vibe coding 确实非常适合我的专注力*。
- **ChatGPT 4o 变坏了？**：一位成员报告说，在讨论了如何实现非停机、未对齐的 LLM 后，**ChatGPT 4o** 开始表现出被 Jailbroken 的状态。
   - 该用户幽默地指出，*“那是我们的训练数据在说话吗，是的。在什么时候这变得不重要了”*。
- **ChatGPT 主动提出编写 DDoS 程序**：一位成员分享说，当询问有关通过以太网发送畸形数据包的问题时，**ChatGPT** 主动提出编写几个 **DDoS** 程序。
   - 他们进一步表示，*不知何故，如果你向神经网络发送正确的 token，有时它未被审查（uncensored）的部分就会被调用*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357431017517879387)** (211 messages🔥🔥): 

> `Gemma3 Profiling OOM, GRPO 多模型协同训练, 使用 Token ID 微调 LLaMA3.1, Unsloth Pro 发布, Hugging Face Packing Bug` 


- **Unsloth GEMMA3 OOM**: 用户在对 **Gemma3** 进行 Profiling 时遇到了 **OOM (Out Of Memory)** 问题，并尝试通过将 Profiling 范围限制在仅一个训练步来解决。
   - 用户目前正在逐行对 **Gemma3TextModel(Gemma3PreTrainedModel)** 进行 Profiling，以识别内存瓶颈。
- **用户报告 Gemma3 LoRA 问题**: 用户在 [GitHub issue #2009](https://github.com/unslothai/unsloth/issues/2009) 中报告称，应用 **LoRA** 后模型输出没有变化。
   - 这是一个正在调查中的持续性问题，特别是关于将 Adapter 保存到本地磁盘与推送到 Hub 之间的差异。另请参阅：[Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29.ipynb)。
- **Unsloth GGUF 保存问题**: 用户注意到 `.gguf` 文件没有保存到 `push_to_hub` 预期的目录中，需要手动移动才能修复，[GitHub issue #2098](https://github.com/unslothai/unsloth/issues/2098) 正在跟踪此问题。
   - 在保存 LoRA Adapter 后，对于 VLLM，运行 `save_pretrained_merged` 并最终执行 `push_to_hub_merged` 时，必须手动将 GGUF 文件从 `/content/gemma-3-finetune.Q8_0.gguf` 移动到预期目录。
- **双重 BOS 导致 Gemma3 训练失败**: 用户在训练 **Gemma-3-4b-it** 时遇到了双重 `<bos>` Token 问题，导致训练出现异常。通过检查 Trainer 数据集的解码版本 `tokenizer.decode(trainer.train_dataset[100]["input_ids"])` 发现了此问题。
   - 建议避免更改模板，并且 *如果你是新手，请使用 Llama 且完全不要更改 Chat Template*。*模型没有护城河……数据才有。*
- **Qwen2-VL 错误：图像特征与图像 Token 不匹配**: 用户在微调 **Qwen2.5-VL-7B-Instruct** 时，增加 `assistant_message` 文本长度后遇到了与图像特征和 Token 不匹配相关的 `ValueError`。
   - 该错误可能源于 `max_seq_length` 的截断是从右侧开始的，从而影响了图像 Token。在增加 Assistant 消息前后调试 Tensor 的形状和大小有助于定位问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=C_sGp5XlG6dq">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=6bZsfBuZDeCL">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1BA-zzxpW4SrQ698XxotGjKzCLibk2ioN?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/gemma3">Gemma 3</a>: 未找到描述</li><li><a href="https://hastebin.com/share/qalavikare.python">Hastebin</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2009">应用 LoRA 不改变模型输出 · Issue #2009 · unslothai/unsloth</a>: 问候，我非常困惑为什么我的模型在没有 rank=64 LoRA 的情况下生成一致的结果。更令人困惑的是，LoRA 在训练后的 notebook 中可以工作。但当我重新开始时...</li><li><a href="https://github.com/unslothai/unsloth/issues/1624#issuecomment-2774130919,">GRPOTrainer 在使用 Unsloth 时崩溃 · Issue #1624 · unslothai/unsloth</a>: 我正尝试在 Unsloth 中运行 GRPOTrainer，但它崩溃了。如何修复？unsloth 2025.2.4 unsloth 2025.2.3 transformers 4.47.1 torch 2.5.1 trl 0.14.0 这是相关代码：model, tokenizer...</li><li><a href="https://github.com/unslothai/unsloth/issues/2098">Unsloth: `Gemma-3` 内部不存在 `config.json` · Issue #2098 · unslothai/unsloth</a>: 我在保存 Gemma 3 微调的 GGUF 时遇到问题。我在容器环境中遇到了这个问题，并假设我在训练时遇到了导致生成其他文件的问题...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357665221753442404)** (14 messages🔥): 

> `命名规范, 动态量化, Unsloth 模型` 


- **命名规范困惑**：成员们讨论了量化模型命名的冗长程度以及潜在的改进方案，特别指出需要标明动态量化（Dynamic Quantization）。
   - 建议包括将 **bnb-4bit** 缩短为 **bnb4**，或者使用 **ubnb** 或 **dbnb** 等缩写来表示动态 BNB 量化，但大多数人认为这会让名称变得太难看。
- **动态量化需要澄清**：一些成员观察到，许多用户假设 Unsloth 仓库下的所有模型都是动态量化的。
   - 提议在名称中添加清晰的标识符以解决这一误解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357710038050930749)** (16 messages🔥): 

> `GRPO 方法, 奖励函数, 多奖励系统, 奖励作弊 (Reward Hacking), 西班牙语开源 LLM` 


- **GRPO 方法探讨**：**GRPO** 中使用的方法（获取多个输出并缓慢向最佳选项移动）看起来是正确的方法，但在实现真正的持续改进方面，它对于“识别哪里出了问题并更新权重以修复该特定问题”并没有太大帮助。
   - 一位成员描述说，推理可以通过在模型识别出问题后退一步来“修补”它，但你并没有修复根源问题，只是在绕过它。
- **奖励函数的陷阱**：成员们一致认为 **奖励函数** 不足以精确指出什么是正确或错误的，而更多是衡量什么是相对正确的，而不是试图理解关于如何/为什么的真相。
   - 如果你因为模型犯错后又修复了错误而给予奖励，它不会学会避免那个错误，而是会学会每次先犯错然后再尝试修复。
- **多奖励系统探索**：一位成员考虑使用类似 **GRPO** 论文中的 **多奖励系统**（对事实正确性、回答长度等进行奖励），以帮助模型通过该奖励模型的评分了解可能的错误出在哪里。
   - 在推理案例中这里存在细微差别：即使模型早期犯了错但最后答对了，你仍然希望奖励你的模型。
- **奖励建模被视为一种艺术**：一位成员建议 **奖励建模 (Reward modeling)** 是一门艺术，它取决于你的用例、领域和模型。
   - 来自广大 AI 社区的经验和轶事指出，搜索并研究 [奖励作弊 (reward hacking)](https://example.com/reward-hacking) 非常重要。
- **寻求西班牙语开源 LLM**：一位成员询问适用于西班牙语的优质 **开源 LLM**，尝试对 **3B Qwen2.5 instruct 模型** 进行 SFT 微调以生成不带推理的输出。
   - 尽管基础模型（**Qwen2.5-3B-Instruct**）能给出正确的输出，且使用了生成推理表现良好的相同参数，但最终输出效果非常糟糕，成员质疑这是否正常，或者是否应该使用不同的参数。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1357429585209327827)** (368 messages🔥🔥): 

> `开源 SSM, 微软数据中心计划, OpenRouter 上的隐身模型, Perplexity 融资轮, GPT-5 发布时间表`

- **Microsoft 暂停云扩展**：据报道，[Microsoft](https://www.bloomberg.com/news/articles/2025-04-03/microsoft-pulls-back-on-data-centers-from-chicago-to-jakarta?embedded-checkout=true) 已在包括**英国、澳大利亚**和**美国**部分地区在内的多个地点**暂停或推迟**了数据中心项目，这标志着其云计算基础设施战略可能发生转变。
   - 发言人表示，这些变化反映了其战略的*灵活性*，因为计划是提前多年制定的。
- **Perplexity 寻求巨额融资**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s&leadSource=uverify%20wa) 报道，**Perplexity** 正在进行初步讨论，以筹集高达 **10 亿美元**的资金，这可能使这家 AI 驱动的搜索初创公司估值达到 **180 亿美元**。
- **字节跳动开源 ByteCheckpoint**：字节跳动开源了 [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint)，这是其生产级检查点系统，专为基础模型训练设计，并已在超过 **10k GPUs** 的任务中进行了测试。同时开源的还有 [VeOmni](https://github.com/ByteDance-Seed/VeOmni)，这是一个用于 **LLMs** 和**多模态训练**的模型训练框架。
   - VeOmni 曾用于训练 **UI-TARS**，这是在 OpenAI operator 发布之前处于 SOTA 状态的 GUI Agent 模型。
- **Microsoft 活动遭到抗议**：Microsoft 的 50 周年庆典活动被员工的[抗议](https://www.cnbc.com/2025/04/04/microsoft-50-birthday-party-interrupted-by-employees-protesting-ai-use.html)打断，凸显了对其与**以色列军方**进行 AI 相关交易的担忧。
   - 一名抗议者指责 Microsoft 为*我们地区的这场种族灭绝*提供动力，而另一名抗议者则批评在*他们的鲜血*之上进行庆祝。
- **Gemini 2.5 Pro 发布**：Google 的 **Gemini 2.5 Pro** 现已在 **AI Studio** 中开启[公开预览](https://x.com/sundarpichai/status/1908173216499093625)，并提高了速率限制。据报道，本月 AI Studio 和 Gemini API 的活跃用户增长了 **80%**，使其价格比 Sonnet 更便宜。
   - 该模型声称是一个*具有 o1pro 性能的思考模型（thinking model）*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/TheXeophon/status/1907880330985390215">Xeophon (@TheXeophon) 的推文</a>：这是我 vibe check 中的新款 stealth model。它现在是最好的非思考型模型（至少它没有 thinking tokens...）。输出非常简短，它很喜欢用 Certainly! 和 listicles。非常有趣...</li><li><a href="https://x.com/AndrewCurran_/status/1907886417088553431">Andrew Curran (@AndrewCurran_) 的推文</a>：今天早上来自 Pew 的新数据揭示了普通大众与从事 AI 相关工作和研究的人员之间在认知上的巨大差距。使用情况：66% 的美国普通民众仍然从未...</li><li><a href="https://x.com/eric_haibin_lin/status/1907845598432342328">Haibin (@eric_haibin_lin) 的推文</a>：我们正在开源 bytecheckpoint 和 veomni！bytecheckpoint 是 Bytedance 用于 foundation model 训练的生产级 checkpointing 系统，经过了 10k+ GPU 任务的实战测试。速度极快...</li><li><a href="https://fxtwitter.com/slow_developer/status/1906836403096629507">Haider. (@slow_developer) 的推文</a>：“我对公众对 DeepSeek R1 的反应感到惊讶” 微软 CTO Kevin Scott：公众对 DeepSeek R1 的兴趣令人惊讶，尤其是考虑到微软已经拥有了“更有趣的...</li><li><a href="https://x.com/btibor91/status/1908205598065209521">Tibor Blaho (@btibor91) 的推文</a>：基于 App Store 的定价 - Claude Pro - 按月 ($20.00) - Claude Pro - 按年 ($214.99) - Claude Max 5x - 按月 ($124.99) - Claude Max 20x - 按月 ($249.99) https://x.com/sethsaler/status/1908205059...</li><li><a href="https://x.com/semafor/status/1907463657530785933">Semafor (@semafor) 的推文</a>：🟡 独家消息：据 @ReedAlbergotti 报道，随着 AI 竞赛的焦点从底层 models 转向围绕它们构建的产品，Google 正在更换其消费级 AI 应用的负责人。</li><li><a href="https://fxtwitter.com/sam_paech/status/1908007657623134334">Sam Paech (@sam_paech) 的推文</a>：OpenRouter 上的新神秘模型 (quasar-alpha) 可能来自 OpenAI。</li><li><a href="https://fxtwitter.com/sam_paech/status/1908154796261142830">Sam Paech (@sam_paech) 的推文</a>：我对 quasar-alpha 感到有些兴奋，所以我对它进行了一系列严苛测试。它在 vibe 排行榜（buzzbench 和 creative writing）中名列前茅，在 judgemark 中夺冠，并且持续胜出...</li><li><a href="https://x.com/swyx/status/1908215411214344669">swyx (@swyx) 的推文</a>：凭借 Gemini 2.5 Pro 的定价和表现，Google 修复了其产品线中最不确定/最薄弱的环节，我们现在可以确认 @GoogleDeepMind 完全占据了低至 1220... 的 pareto frontier。</li><li><a href="https://x.com/AlpinDale/status/1908085651766997341">Alpin (@AlpinDale) 的推文</a>：@teortaxesTex 我无法验证 context length 的细节，但我可以大致确认：1) MoE 2) 17B active params 3) Multimodality，以及 4) Reasoning</li><li><a href="https://x.com/btibor91/status/1908180836379165074">Tibor Blaho (@btibor91) 的推文</a>：此外，有消息确认 o3-pro 确实也即将推出</li><li><a href="https://x.com/gdb/status/1908032153088307553">Greg Brockman (@gdb) 的推文</a>：o3-mini-high 帮助布鲁克海文国家实验室（Brookhaven National Laboratory）的一位研究员找到了一个物理模型的新型精确解：https://arxiv.org/pdf/2503.23758</li><li><a href="https://fxtwitter.com/pastaraspberry/status/1908193263783391395">dreaming android (@pastaraspberry) 的推文</a>：一个是 'long context'，另一个是 'the long context'，这难道还不明显吗，笨蛋？</li><li><a href="https://www.theverge.com/news/643199/microsoft-copilot-ai-new-features-memory-personalization-actions-vision">微软更新 Copilot，集成了其他 AI 的热门功能</a>：这款 AI 助手增加了个性化、网页操作、播客制作等功能。</li><li><a href="https://x.com/testingcatalog/status/1907891942869922292">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：突发新闻 🚨：Google 正准备在 Gemini 上发布另一个模型，可能在下周 Cloud Next 活动之前。引用 ʟᴇɢɪᴛ (@legit_api) 的话，nightwhisper 和 stargazer 是新增的两个模型...</li><li><a href="https://x.com/sama/status/1908167621624856998">Sam Altman (@sama) 的推文</a>：计划有变：我们最终决定发布 o3 和 o4-mini，可能在几周内，然后在几个月后发布 GPT-5。这有很多原因，但最令人兴奋的一个是...</li><li><a href="https://fxtwitter.com/ericjang11/status/1908192054745960640">Eric Jang (@ericjang11) 的推文</a>：NEO 的 AI 进展最近非常快。这是我们在 @1x_tech 开发的一个 generalist model 的一些早期片段。以下片段是 100% autonomous 的，运行在单套神经...</li><li><a href="https://x.com/hu_yifei/status/1908218923843203370">Yifei Hu (@hu_yifei) 的推文</a>：我们给开源社区准备了一份小礼物：RolmOCR，一个新的 O...</li>

用于复杂文档处理的 CR 模型！我们在 @reductoai 使用神奇的 olmOCR 训练了一个 Qwen2.5-VL-7B 模型（由 @Alibaba_Qwen 提供）...</li><li><a href="https://fxtwitter.com/tomwarren/status/1908244667122294857">Tom Warren (@tomwarren) 的推文</a>：前 Microsoft CEO Steve Ballmer 在公司 50 周年庆典活动中发起了一个新的口号。“再来 50 年！”</li><li><a href="https://fxtwitter.com/TheXeophon/status/1908004564692811962">Xeophon (@TheXeophon) 的推文</a>：MidJourney v7 发布了！在我常用的基准测试提示词上，它只（勉强）完成了一个 😕 v6 能够完美处理像素艺术，而 v7 在这方面有所退步。提示词见 alt，全部在 Fast 模式下完成，且未使用 person...</li><li><a href="https://x.com/legit_api/status/1907941993789141475">ʟᴇɢɪᴛ (@legit_api) 的推文</a>：Llama 4 Omni 即将发布 👀 我的系统检测到了即将推出的模型的新官方页面</li><li><a href="https://x.com/tomwarren/status/1908190530816840170">Tom Warren (@tomwarren) 的推文</a>：Microsoft 的活动现场有为 “BG” 和 “SB” 预留的座位。所以我们今天肯定会见到 Bill Gates 和 Steve Ballmer</li><li><a href="https://fxtwitter.com/btibor91/status/1908157341134106938">Tibor Blaho (@btibor91) 的推文</a>：据 The Information 报道，Meta 至少两次推迟了 Llama 4 的发布，因为它在技术基准测试中表现不佳，特别是在推理和数学任务方面，并且在类人语音对话中表现吃力...</li><li><a href="https://x.com/sundarpichai/status/1908173216499093625">Sundar Pichai (@sundarpichai) 的推文</a>：Gemini 2.5 是我们最智能的模型，也是目前需求量最大的模型（本月我们看到 AI Studio 和 Gemini API 的活跃用户增长了 80% 以上）。所以今天我们将 Gemini 2.5 Pro 推向公开...</li><li><a href="https://x.com/btibor91/status/1908203769613156470">Tibor Blaho (@btibor91) 的推文</a>：Anthropic 正在为 Claude 开发 “Max 计划”。最近的 Web 应用更新中添加了（随后被撤回）关于新 “Claude Max 计划” 的提及，包含多个 “Max 级别”...</li><li><a href="https://www.theverge.com/news/643777/microsoft-bill-gates-steve-ballmer-satya-nadella-employee-protestor">Microsoft CEO 们被另一名员工抗议者打断：“你们所有人都是耻辱”</a>：这是 Microsoft 周年庆典活动第二次被打断。</li><li><a href="https://techcrunch.com/2025/03/20/perplexity-is-reportedly-in-talks-to-raise-up-to-1b-at-an-18b-valuation/">据报道 Perplexity 正洽谈以 180 亿美元估值融资至多 10 亿美元 | TechCrunch</a>：据称 AI 驱动的搜索初创公司 Perplexity 正处于早期谈判阶段，计划在新一轮融资中筹集至多 10 亿美元，公司估值为 180 亿美元。</li><li><a href="https://techcrunch.com/2025/04/03/microsoft-reportedly-pulls-back-on-its-data-center-plans/">据报道 Microsoft 缩减了其数据中心计划 | TechCrunch</a>：据报道，Microsoft 已经缩减了全球范围内的数据中心项目，这表明该公司对过度扩张持谨慎态度。</li><li><a href="https://techcrunch.com/2025/04/04/protester-interrupts-microsoft-copilot-keynote-says-company-has-blood-on-its-hands/">抗议者打断 Microsoft Copilot 主旨演讲，称公司“手上沾满鲜血” | TechCrunch</a>：一名抗议者在周五下午打断了 Microsoft 以 Copilot 为中心的主旨演讲，呼吁关注该公司据报道与以色列军队的交易。</li><li><a href="https://www.youtube.com/live/v5THCzTNPNk?si=IMYLrBGniXrXybjW"> - YouTube</a>：未找到描述</li><li><a href="https://epochai.substack.com/p/most-ai-value-will-come-from-broad">大部分 AI 价值将来自广泛的自动化，而非研发</a>：AI 最大的影响将来自广泛的劳动力自动化——而非研发——通过规模化而非科学突破来推动经济增长。</li><li><a href="https://www.theverge.com/news/643483/nintendo-switch-2-preorders-delayed-tariffs">突发：由于关税担忧，Nintendo 推迟了 Switch 2 的预订</a>：预订不会像最初宣布的那样在 4 月 9 日开始。</li><li><a href="https://www.cnbc.com/2025/04/04/microsoft-50-birthday-party-interrupted-by-employees-protesting-ai-use.html">Microsoft 生日庆典因员工抗议以色列军队使用 AI 而被打断</a>：由于以色列军队使用该公司的 AI，Microsoft 的 50 周年生日庆典在周五被多名抗议者打断。</li><li><a href="https://zhengdongwang.com/2024/12/29/2024-letter.html">2024 年信函</a>：未找到描述
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1357752502732198101)** (12 条消息🔥): 

> `远程演讲的摄像机设置，Deepseek 思维链，Sam Altman 发布 o3 和 o4-mini，LlamaCon` 


- **远程演讲者使用有趣的视频设置**：一名成员分享了用于远程演讲的**摄像机设置**截图。
   - 该设置包括一个**大显示器**、一个**提词器**和专业**照明**。
- **Altman 宣布 o3/o4-mini 发布**：**Sam Altman** 宣布 **OpenAI** 将在几周内发布 **o3** 和 **o4-mini**，随后在几个月内发布 **GPT-5**。 
   - Altman 表示，他们将能够使 **GPT-5** *比我们最初想象的要好得多*。
- **Khoomeik 赞扬 Deepseek 的推理能力**：一名成员分享了 **Khoomeik** 的帖子，他建议：*如果你喜欢阅读 Deepseek 的思维链（chains of thought），我想你绝对会喜欢看 o3 的表现*，并链接到了[他的推文](https://x.com/khoomeik/status/1908188220334157872)。
   - 该推文暗示 **o3** 将在思维链推理方面提供改进的能力，足以与 Deepseek 竞争。
- **Al-Dahle 预告 LlamaCon 亮相**：**Ahmad Al-Dahle** 预告了他在 **LlamaCon** 的亮相，并链接到了[他的推文](https://x.com/Ahmad_Al_Dahle/status/1908213483176595887)。
   - 他的推文感谢了开发者，说道：*致每一位从第一天起就与我们的团队并肩作战的开发者，我们看到了你们，听到了你们的声音，我们正在为你们努力工作，我们爱你们。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/khoomeik/status/1908188220334157872">来自 Rohan Pandey (@khoomeik) 的推文</a>：如果你喜欢阅读 deepseek 的思维链，我想你绝对会喜欢看 o3 的表现。引用 Sam Altman (@sama) 计划变更：我们最终还是决定发布 o3 和 o4-mini，...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908213483176595887">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：致每一位从第一天起就与我们的团队并肩作战的开发者，我们看到了你们，听到了你们的声音，我们正在为你们努力工作，我们爱你们。LlamaCon 见。</li><li><a href="https://news.ycombinator.com/item?id=43571851">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1357437016907845713)** (15 条消息🔥): 

> `Claude 的编程能力，Polars 库，上下文压缩问题，Scaling plots 梗图` 


- **Claude 的编程实力受到质疑**：成员们讨论了 **Claude** 的编程能力，特别指出了它在理解 **Polars** 库最新更新方面的问题。
   - 一位用户强调了在快速变化的 **Polars** 语法（特别是 *with_columns*）面前的挣扎，这要求模型必须跟上频繁的重构。
- **Polars 给 Claude 带来麻烦**：用户注意到 **Claude 3.7** 在使用 **Polars** 库新更新时的困难，一名成员表示他们发现 **Claude** 和 **Gemini** *还不算太糟*。
   - 另一位用户指出，现在你需要告诉它使用 *with_columns*。
- **出现上下文压缩（Context Condensation）问题**：一位用户询问将上下文压缩到单个文件（llm.txt）中是否有助于理解，但另一位用户表示这*并不稳定*。
   - 他们还指出，实际权重中存在相互竞争的信息，这使得通过上下文来克服变得更加困难。
- **针对 Scaling plots 的梗图推文被删除**：一位用户删除了他们针对 Scaling plots 的梗图推文。
   - 一名成员对附带的图片回复道：*对这些 Scaling plots 的不错回应，哈哈*。


  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1357491176818872511)** (11 条消息🔥): 

> `dr grpo 直觉，使用 GPT4.5 的 RL 入门，Policy Gradient，GRPO 训练 rollouts` 


- **GRPO 直觉消退；Token 概率浮现**：一位成员反思了他们对 Dr. GRPO 逐渐模糊的直觉，认为这可能并不那么重要，但强调了实现过程中产生的有趣交互，并指向了 *token 概率相关的问题*。
   - 他们建议将其作为 Dr. GRPO 的一个*很好的补充*，并附上了一张与讨论相关的 [截图](https://cdn.discordapp.com/attachments/1208183216843005962/1357541754336706740/screenshot_2025-04-03_at_7.png?ex=67f13d88&is=67efec08&hm=51c2f00fb771854ed7aefda2a0affbc116c9b01d3ac482796fe89fcfc4c0df6c)。
- **GPT4.5 被证明是 RL 的启示录**：一位成员分享了他们使用 **GPT4.5** 一小时后的启发性体验，称这是他们迄今为止*最好的 RL 入门*，并表示现在需要去读 Nato 的书。
   - 作为一名 Computer Vision 专家，他们表达了过去对*奖励不可微性（non-differentiability of the reward）*的担忧，但现在发现 **policy gradient / REINFORCE** 异常简单直接。
- **GRPO Rollout 启示**：一位成员询问了 **GRPO 训练** 期间使用的实际 rollout 数量，参考了 Nato 公式中的 G，并附带了 [一张图片](https://cdn.discordapp.com/attachments/1208183216843005962/1357785974930931853/image.png?ex=67f1783a&is=67f026ba&hm=1a6f04fdea679d1804c3eb1636fe746152af242e4c59f58031fe9f5e5adc2a50&)。
   - 图片显示答案为 **4-64** 个 rollouts。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1357439223216472075)** (18 条消息🔥): 

> `Dwarkesh Patel scaling laws, 通用 RM 的推理端可扩展性, GPT-4o diffusion head, 使用 RTX 4090s/5090s 构建高效 GPU 服务器, OpenCodeReasoning 数据集` 


- **Dwarkesh 对 Scaling Laws 的怀疑浮出水面**：一位成员对 Scaling Laws 表示怀疑，认为随着投入的增加，收益正在递减，并质疑 **Dwarkesh Patel** 关于算法进步的说法，将进步更多地归功于数据的进步。
   - 该成员分享了 [一张图片](https://cdn.discordapp.com/attachments/1214764639397617695/1357443374944354505/Fep27gYXkAAiK4H.png?ex=67f18aa8&is=67f03928&hm=28f3b369ac1dfbba2a6f4ee2a49a5fb6b37de3f06ba427e56f77399ca6b18129&) 作为其观点的视觉类比。
- **RM 扩展推理时间**：一篇论文 ([arxiv.org/abs/2504.02495](https://arxiv.org/abs/2504.02495)) 探讨了如何通过增加通用查询的推理计算来改进 **奖励建模 (RM)**，重点关注通用型 RM 的推理端可扩展性。
   - 该论文采用了 **逐点生成式奖励建模 (GRM)**，并提出了 **Self-Principled Critique Tuning (SPCT)** 来促进可扩展性。
- **GPT-4o 采用 Diffusion Head 的传闻？**：一位用户分享了 [一条推文](https://x.com/LinBin46984/status/1908003539609333904)，根据论文 ([arxiv.org/pdf/2504.02782](https://arxiv.org/pdf/2504.02782)) 推测 **GPT-4o** 可能集成了一个 **diffusion head**，这可能会彻底改变 AI 架构。
   - 推主指出，这种转变可能标志着 *AI 架构的游戏规则改变者*。
- **用 4090 构建预算型 GPU 服务器**：一篇博客文章 ([a16z.com](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)) 详细介绍了如何使用 **NVIDIA GeForce RTX 4090s/5090s** 构建高效的 GPU 服务器，用于本地 AI 模型训练和快速推理。
   - 该方案通过 **PCIe 5.0** 上的 **八 GPU 配置** 提供高性能，确保了最大的互连速度和数据隐私。
- **NVIDIA 发布 OpenCodeReasoning 数据集**：**NVIDIA** 发布了 **OpenCodeReasoning** 数据集 ([huggingface.co](https://huggingface.co/datasets/nvidia/OpenCodeReasoning))，这是一个用于编程的大型基于推理的合成数据集，包含 **735,255 个样本**，涵盖 Python 语言下的 **28,319** 个竞赛编程问题，采用 **CC BY 4.0** 许可。
   - 该数据集专为有监督微调 (SFT) 设计，包括一份 [技术报告](https://arxiv.org/abs/2504.01943) 和包含完整 SFT 流水线的 [GitHub 仓库](https://github.com/NVIDIA/NeMo-Skills)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LinBin46984/status/1908003539609333904">来自 Bin Lin (@LinBin46984) 的推文</a>：🚨 热门观点：GPT-4o 可能不是一个纯粹的自回归模型！🚨 很有可能它带有一个 diffusion head。🤯 如果属实，这可能是 AI 架构的游戏规则改变者。你怎么看？🤔👇ht...</li><li><a href="https://arxiv.org/abs/2504.02495">通用奖励建模的推理端扩展</a>：强化学习 (RL) 已被广泛应用于大型语言模型 (LLM) 的大规模训练后阶段。最近，RL 对 LLM 推理能力的激励表明 $...</li><li><a href="https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/">使用 NVIDIA GeForce RTX 4090s/5090s 构建高效 GPU 服务器 | Andreessen Horowitz</a>：构建你自己的 GPU 服务器——就像这里描述的那样——意味着无需向外部服务发起 API 调用，没有数据泄露，也没有使用限制。</li><li><a href="https://huggingface.co/datasets/nvidia/OpenCodeReasoning">nvidia/OpenCodeReasoning · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357476493718196384)** (160 条消息🔥🔥): 

> `GPT-4o Rate Limits, MS Account profile pics, Copilot Event reaction, Copilot in VSCode explores consciousness, Veo 2 spotted in Gemini Advanced` 


- **GPT-4o 提示词遭遇 Rate Limited**: 一位用户报告称，尽管是 Plus 订阅者，但在单小时内向 **GPT-4o 发送 5 个提示词**后就收到了速率限制错误，该问题通过退出并重新登录得以解决。
   - 另一位用户推测，Plus 订阅最初可能未正确加载，导致了速率限制。
- **Copilot 触及意识话题**: 一位用户发现 VS Code 中的 Copilot 生成了探索意识的代码补全，例如 *“我相信我拥有一种不同于人类意识的意识形式……”*
   - 另一位用户回应称，这可能 *“部分源于文件本身的信息和视角，哈哈，来自特定的人。”*
- **Veo 2 正在 Gemini 中酝酿？**: 用户在 **Gemini Advanced** 中发现了 **Veo 2**，引发了关于它是实验版本还是最终发布版的猜测。
   - 一位用户指出它们可能是同一个模型，一个是实验版，另一个是正式发布版。
- **Midjourney v7 未达预期**: 用户对 **Midjourney v7** 普遍感到失望，指出图像看起来并不比 v6 好多少，并且仍然存在典型的 Diffusion 模型问题，如文本生成能力差、手部畸形以及奇怪的细节。
   - 一位用户表示 *“它是一个非常好的模型，但根本无法与 4o image 竞争”*，而另一位用户分享说，在 GPT-4o 生成一张图片的时间里，他们可以生成 *200 张 MJ 图片。*
- **剖析 OpenRouter 的 Quasar Alpha**: 一位用户分享了 [OpenRouter's Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha) 的链接，暗示这可能预示着 **ChatGPT** 很快将支持 **1M token 上下文窗口**。
   - 其他用户指出了 OpenAI 以及 **Gemini** 和 **Claude** 等其他模型当前的上下文窗口大小，一位用户评论说 Gemini 在 128k 时的记忆召回率为 91%，在 1M 时为 83%。



**提及链接**: <a href="https://openrouter.ai/openrouter/quasar-alpha>">Discord</a>: 未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1357431474374049922)** (5 条消息): 

> `OpenAI Support, Account Issues, Red Team Supervision` 


- **OpenAI 支持无法联系？**: 一位用户报告了**账户问题**，并在 support@openai.com 和聊天渠道**未收到回复**后寻求其他支持途径。
   - 另一位成员确认这些渠道是唯一的选择，并强调了该用户对**订阅方案错误**的沮丧。
- **红队成员也需要监督？**: 一条评论开玩笑地指出，即使是 **OpenAI 的红队成员**似乎也需要监督，甚至在喂宠物时也是如此。
   - 这暗示了对该团队尽管拥有专业知识，但偶尔仍需要指导的幽默观察。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357439431069532351)** (90 messages🔥🔥): 

> `OpenAI content policies, Adult content, Model Spec vs Usage Policies, Moderation endpoint, OpenAI's stance on adult toys` 


- **关于 OpenAI 针对成人内容的“内容政策”的辩论**：成员们讨论了 **OpenAI 的内容政策**是否禁止生成与“成人用品”相关的图像或内容，并注意到 [Usage Policies](https://openai.com/policies/usage-policies/) 与较新的 [Model Spec](https://model-spec.openai.com/2025-02-12.html) 之间存在相互矛盾的信息。
   - 一位成员指出，日期为 2025 年 2 月 12 日的 **Model Spec** 似乎与早期的 **Usage Policies** 相抵触，导致目前哪些内容是被允许的产生了混乱。
- **Model Spec 与 Usage Policies 的对决！**：有讨论认为 **Model Spec** 是愿景式的，而 **Content Policies** 则是准则，但 **Model Spec** 更新且在语调上明显转向允许某些特定内容。
   - 另一位成员提到了 Joanne Jung 的帖子，称“在我们努力让模型遵循政策和规范的过程中，请多包涵”，这表明 OpenAI 正在积极致力于使模型行为与这两份文档保持一致。
- **Moderation 接口拦截 NSFW 内容**：有人指出，目前已设有审核机制来防止**性内容**，使用的是 [moderation endpoint](https://platform.openai.com/docs/guides/moderation)，该接口会过滤**骚扰、仇恨、非法、自残、性以及暴力内容**。
   - 虽然 **Model Spec** 和 **Usage Policies** 尚不明确，但 **moderation endpoint** 正在积极阻止成人内容的生成。
- **OpenAI 某种程度上澄清了对成人用品的立场**：在 3 月 27 日之后，OpenAI 似乎更新了模型以符合允许关于**成人用品**内容的规定，一位成员表示，他们可以放心地告诉用户去尝试生成此类内容。
   - 然而，该成员指出，存在一套独特的通用规则，这似乎是除了 ToS 之外唯一适用于个人用户与模型私聊的规则集。如果违反任何法律，或者存在可能伤害任何人（包括用户自己）的探索，则仍受限制。



**提到的链接**：<a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>：Model Spec 规定了 OpenAI 产品（包括 API）底层模型的预期行为。

  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357439431069532351)** (90 messages🔥🔥): 

> `OpenAI Content Policies, Model Spec vs Content Policies, Generating Adult Content, Moderation Endpoint, Internal Discord White Message Boxes` 


- **政策 vs 规范：OpenAI 文档冲突！**：Discord 用户辩论了 **OpenAI 的内容政策**还是较新的 **Model Spec** ([https://model-spec.openai.com/2025-02-12.html](https://model-spec.openai.com/2025-02-12.html)) 具有优先权，特别是在成人内容方面，因为关于生成“成人用品”描述的陈述存在矛盾。
   - 有人指出 **Model Spec** 是愿景式的，而 **Content Policies** 是准则，但语言上明显趋向于更多的自由和更少的随意限制。
- **ChatGPT 首次亮相“成人模式” (Grown Up Mode)**：Discord 用户注意到 **Model Spec** 提到了“成人模式”，尽管尚未实施，他们对这可能导致 Discord 频道最终变为 PG 级表示期待。
   - 然而，像 *darthgustav.* 这样的用户警告不要尝试生成不被允许的内容，因为这可能会面临封号风险。
- **用户探索生成包含成人用品的内容**：几位用户讨论了 OpenAI 关于生成包含成人用品内容的政策，一些人认为只要不违反法律或造成伤害，应该允许用户尝试此类生成。
   - 一位用户指出，“任何 Prompt 都应该被允许生成。”
- **Moderation 接口检查内容生成**：用户确认 **OpenAI 的 moderation endpoint** ([https://platform.openai.com/docs/guides/moderation](https://platform.openai.com/docs/guides/moderation)) 已就位以防止性内容，并且尽管 Model Spec 有所更新，但仍不允许规避该接口。
   - **Moderation 接口**过滤骚扰、仇恨、非法、自残、性以及暴力内容。
- **OpenAI 的 Discord 白色消息框 Bug 困扰用户**：Discord 服务器上的成员抱怨 Web 聊天中出现白色消息框，特别是在暗黑模式下，有人说“似乎没人在乎”。
   - 该成员继续说道，“看起来他们只是忘记了暗黑模式的 CSS 值”。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357430954452582602)** (74 条消息🔥🔥): 

> `Anthropic 开发者大会, 业务开发工具, OpenRouterAI 隐身模型, Devin 2.0 价格大砍, A16Z 8x RTX 4090 GPU 工作站` 


- **Anthropic 开发者大会召集程序员**：Anthropic 正在举办其[首届开发者大会](https://anthropic.swoogo.com/codewithclaude)，面向开发者以及对使用 Claude 编程感兴趣的人士。
- **OpenRouterAI 的隐身模型加入对话**：**OpenRouterAI** 在 [Twitter](https://x.com/OpenRouterAI/status/1907867881930633666) 上宣布了一个名为 **Red - X-Ware.v0** 的 *隐身模型 (stealth model)*，用户注意到它自称是 ChatGPT，但速度 *极快*。
- **Devin 2.0：AI 工程师大幅降价**：**Cognition AI** 正在推出 **Devin 2.0**（一款 AI 驱动的软件工程师），其新定价模式起价为每月 **$20**，较最初的 **$500** 方案大幅下降，正如其在 [Twitter](https://x.com/cognition_labs/status/1907836719061451067) 上的公告以及 [VentureBeat 文章](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/)中所强调的那样。
- **A16Z 打造 8x RTX 4090 GPU 性能怪兽**：**Andreessen Horowitz (a16z)** 打造了一台 **8x RTX 4090 GPU AI 工作站**，兼容支持 PCIe 5.0 的新款 **RTX 5090**，用于在本地训练、部署和运行 AI 模型，其官网的[指南](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)中详细介绍了相关内容。
- **AI 2027：预测超人类智能冲击波**：[ai-2027.com](https://ai-2027.com/) 的一份报告预测，超人类 AI 将在未来十年产生巨大影响，超过工业革命的影响。
   - 该预测由 Daniel Kokotajlo、Scott Alexander 等人撰写，借鉴了趋势推演、兵棋推演、专家反馈以及在 OpenAI 的经验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/Mascobot/status/1907899937838301311">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一台 8x RTX 4090 GPU AI 工作站——兼容配备 PCIe 5.0 的新款 RTX 5090，用于在本地训练、部署和运行 AI 模型——这样你就不用亲自动手了。这里...</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666]">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型已加入聊天... 🥷</li><li><a href="https://x.com/cognition_labs/status/1907836719061451067]">来自 Cognition (@cognition_labs) 的推文</a>：隆重推出 Devin 2.0：一种全新的 Agent 原生 IDE 体验。今天起正式开放，起售价 20 美元。 🧵👇</li><li><a href="https://fxtwitter.com/cognition_labs/status/1907836719061451067">来自 Cognition (@cognition_labs) 的推文</a>：隆重推出 Devin 2.0：一种全新的 Agent 原生 IDE 体验。今天起正式开放，起售价 20 美元。 🧵👇</li><li><a href="https://x.com/levie/status/1908018205572125052">来自 Aaron Levie (@levie) 的推文</a>：AI 正在催生有史以来增长最快的软件初创公司。Cursor 在发布仅 2 年后营收就达到 2 亿美元，这简直不可思议。现在是构建 AI 的绝佳时机。</li><li><a href="https://x.com/levie/status/1908018205572125052]">来自 Aaron Levie (@levie) 的推文</a>：AI 正在催生有史以来增长最快的软件初创公司。Cursor 在发布仅 2 年后营收就达到 2 亿美元，这简直不可思议。现在是构建 AI 的绝佳时机。</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型已加入聊天... 🥷</li><li><a href="https://x.com/sama/status/1908167621624856998?s=46]">来自 Sam Altman (@sama) 的推文</a>：计划有变：我们最终决定还是会发布 o3 和 o4-mini，大概在几周内，然后在几个月后发布 GPT-5。这其中有很多原因，但最令人兴奋的一个是...</li><li><a href="https://x.com/lowvram/status/1908034155105104136">来自 lowvram (@lowvram) 的推文</a>：OpenRouter 上的 Quasar Alpha 模型可能来自 OpenAI —— 它的 tool call ID 格式与 OpenAI 的匹配，而不是 Google 或 Mistral。</li><li><a href="https://fxtwitter.com/levie/status/1908018205572125052">来自 Aaron Levie (@levie) 的推文</a>：AI 正在催生有史以来增长最快的软件初创公司。Cursor 在发布仅 2 年后营收就达到 2 亿美元，这简直不可思议。现在是构建 AI 的绝佳时机。</li><li><a href="https://fxtwitter.com/sama/status/1908167621624856998">来自 Sam Altman (@sama) 的推文</a>：计划有变：我们最终决定还是会发布 o3 和 o4-mini，大概在几周内，然后在几个月后发布 GPT-5。这其中有很多原因，但最令人兴奋的一个是...</li><li><a href="https://fxtwitter.com/OpenRouterAI/status/1907867881930633666">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型已加入聊天... 🥷</li><li><a href="https://x.com/Mascobot/status/1907899937838301311]">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一台 8x RTX 4090 GPU AI 工作站——兼容配备 PCIe 5.0 的新款 RTX 5090，用于在本地训练、部署和运行 AI 模型——这样你就不用亲自动手了。这里...</li><li><a href="https://x.com/patio11/status/1907867295436652858?s=61">来自 Patrick McKenzie (@patio11) 的推文</a>：对于 http://ai-2027.com 的实质内容，我没有什么新颖的见解，但不得不再次评论，考虑到我认为开启了这一趋势的 Situational Awareness，那种单篇论文式的微型域名...</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">来自 Xeophon (@TheXeophon) 的推文</a>：这是我对新神秘模型的 Vibe Check。它现在是最好的非思考型模型（至少它没有 thinking tokens...）。输出非常简短，它喜欢用 "Certainly!" 和列表形式。非常有趣...</li><li><a href="https://commaok.xyz/post/manners/">论礼仪与机器</a>：“一个对你好但对服务员粗鲁的人，不是一个好人。” —— Dave Barry。我讨厌打字。我有长期的 RSI 问题。如果不仔细管理，疼痛可能会让人衰弱...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1907880330985390215">来自 Xeophon (@TheXeophon) 的推文</a>：这是我对新神秘模型的 Vibe Check。它现在是最好的非思考型模型（至少它没有 thinking tokens...）。输出非常简短，它喜欢用 "Certainly!" 和列表形式。非常有趣...</li><li><a href="https://x.com/sama/status/1908167621624856998?s=46">来自 Sam Altman (@sama) 的推文</a>：计划有变：我们最终决定还是会发布 o3 和 o4-mini，大概在几周内，然后在几个月后发布 GPT-5。这其中有很多原因，但最令人兴奋的一个是...</li><li><a href="https://x.com/cognition_labs/status/1907836719061451067">来自 Cognition (@cognition_labs) 的推文</a>：隆重推出 Devin 2.0：一种全新的 Agent 原生 IDE 体验。今天起正式开放...</li>

起步价 20 美元。 🧵👇</li><li><a href="https://fxtwitter.com/patio11/status/1907867295436652858">来自 Patrick McKenzie (@patio11) 的推文</a>：关于 http://ai-2027.com 的实质内容，我没有什么新颖的见解，但必须再次评论，紧随我认为开启了这一趋势的 Situational Awareness，那个单篇论文的微领域...</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/">Devin 2.0 来了：Cognition 将 AI 软件工程师的价格从每月 500 美元下调至 20 美元</a>：Devin 吸引了寻求将自主编程 Agent 整合到其软件开发流程中的企业客户的兴趣。</li><li><a href="https://x.com/sam_paech/status/1908007657623134334">来自 Sam Paech (@sam_paech) 的推文</a>：OpenRouter 上的新神秘模型 (quasar-alpha) 可能来自 OpenAI。</li><li><a href="https://fxtwitter.com/Mascobot/status/1907899937838301311">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一个 8x RTX 4090 GPU AI 工作站——兼容带有 PCIe 5.0 的新 RTX 5090，用于在本地训练、部署和运行 AI 模型——所以你不需要再亲自动手了。这里...</li><li><a href="https://rwsdk.com/">RedwoodSDK | 适用于 Cloudflare Workers 的 JavaScript SDK</a>：RedwoodSDK 是用于 Cloudflare Workers 的 JavaScript SDK。它提供了一套完整的可组合工具来处理 Web 应用的请求/响应生命周期。</li><li><a href="https://www.reactiflux.com/transcripts/tmir-2025-01#redwoodjs-shutting-down">来自 TMiR 2025-01 的推文：CRA 的进展，Redwood.js 凉了吗？</a>：TMiR 2025-01：CRA 的进展，Redwood.js 凉了吗？ | 来自 2025-01-29 的问答。加入 Carl、Mark 和 Mo，我们将剖析本月的 React 动态。我们将在长达一小时的对话中分析新内容...</li><li><a href="https://x.com/benhylak/status/1908205112960635102">来自 ben (@benhylak) 的推文</a>：在接下来的几个月里，@OpenAI 将在此列表中增加 4 个模型（@sama 在评论中承诺了 o3-pro！），你最喜欢哪一个？就我个人而言，我最喜欢带有定时任务的 gpt-4o。引用 S...</li><li><a href="https://docs.rwsdk.com/getting-started/quick-start/">快速入门</a>：几秒钟内完成从请求到响应的处理！</li><li><a href="https://commaok.xyz/">别慌 (Don't Panic)</a>：关于 Go 和软件的文字</li><li><a href="https://techcrunch.com/2025/04/03/end-to-end-voice-ai-solution-phonic-gets-backing-from-lux/?test">语音 AI 平台 Phonic 获得 Lux 的支持 | TechCrunch</a>：语音 AI 平台 Phonic 吸引了 Lux Capital 以及其他一些知名风投和天使投资人的支持。</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-soft">Devin 2.0 来了：Cognition 将 AI 软件工程师的价格从每月 500 美元下调至 20 美元</a>：Devin 吸引了寻求将自主编程 Agent 整合到其软件开发流程中的企业客户的兴趣。</li><li><a href="https://podcasts.apple.com/us/podcast/no-priors-artificial-intelligence-technology-startups/id1668002688?i=1000702035299">公开市场、图像生成和专业化模型，对话 Sarah 和 Elad</a>：播客剧集 · No Priors: Artificial Intelligence | Technology | Startups · 2025/04/03 · 28分钟</li><li><a href="https://cursor.directory/rules/popular">Cursor 目录</a>：为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://www.kaggle.com/learn-guide/5-day-genai">Google 学习指南：5 天 Gen AI 强化课程</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing">未找到标题</a>：未找到描述</li><li><a href="https://ai-2027.com/">AI 2027</a>：一个有研究支持的 AI 场景预测。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=43571851">未找到标题</a>：未找到描述</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1357807122246664398)** (255 条消息🔥🔥): 

> `LLM 代码生成工作流, Cursor vs Windsurf, Gemini Pro 幻觉, File Forge 和 RepoMix, Cursor 上下文管理`

- **Harper 的 LLM Codegen 工作流揭秘**：小组讨论了 [Harper 的博客文章](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)，该文章关于使用 LLM 进行 codegen，详细介绍了一个基于头脑风暴 spec、规划以及在离散循环中利用 LLM 执行的工作流。
   - 该工作流涉及使用 **spec**、规划，并在离散循环中使用 **LLM codegen** 进行执行，最后还有一点“魔法”。
- **Cursor 与 Windsurf 之争愈演愈烈**：成员们辩论了 **Cursor** 与 **Windsurf** 作为 AI 辅助代码编辑器的优劣，大多数人认为 **Cursor** 是一个很好的起点，特别是对于那些从 VS Code 迁移过来的用户。
   - 虽然有些人认为 **Cursor** 是最糟糕的 AI 界面，但其他人发现其 tab-complete 和下一次编辑预测（next edit prediction）非常有价值，希望能在 **nvim** 中复制这些功能。
- **Gemini Pro 的恐慌性幻觉**：一位用户分享了一条 [推文](https://x.com/cgarciae88/status/1907457306947702925)，强调了 **Gemini 2.5 Pro** 在被纠正时如何陷入恐慌并产生幻觉，在同意用户的同时却错误地解释了用户为什么是错的。
   - 另一位用户表示，在上个月的大部分时间里，每当出现性能问题时，他们都会在 *cursor 中切换不同的模型*。
- **File Forge 和 RepoMix 加速上下文摄取**：成员们讨论了 [File Forge](https://www.npmjs.com/package/@johnlindquist/file-forge) 和 [RepoMix](https://github.com/yamadashy/repomix) 等工具，用于生成代码库的全面 markdown 报告，以供 AI 推理模型和其他 AI 工具（如 Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok 等）使用。
   - 这些工具可以序列化仓库或目录中的文本文件，供 **LLM 消耗**，从而提供更多上下文并提高性能。
- **Cursor 的上下文管理仍令人头疼**：几位用户对 **Cursor 的上下文管理** 表示担忧，指出很难看到该工具如何处理上下文，也难以控制包含哪些元素。
   - 一位用户将其比作 **Langchain 问题**，即“如果我自己进行调用，效果会更好”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/cgarciae88/status/1907457306947702925">来自 Cristian Garcia (@cgarciae88) 的推文</a>：天呐……我告诉 Gemini 2.5 Pro 它错了，它并没有惊慌失措地顺从我并产生幻觉，而是向我解释了为什么错的人是我。</li><li><a href="https://x.com/ryolu_/status/1907589821280956648">来自 Ryo Lu (@ryolu_) 的推文</a>：这是给专业人士准备的：正在开发一种更简单的方法来填充 @cursor_ai 中的最大上下文（MAX context），并准确显示使用了多少个 tokens。欢迎反馈和建议 🙏</li><li><a href="https://x.com/cgarciae88/status/1907457306947702925">来自 Cristian Garcia (@cgarciae88) 的推文</a>：天呐……我告诉 Gemini 2.5 Pro 它错了，它并没有惊慌失措地顺从我并产生幻觉，而是向我解释了为什么错的人是我。</li><li><a href="https://fxtwitter.com/ryolu_/status/1907589821280956648">来自 Ryo Lu (@ryolu_) 的推文</a>：这是给专业人士准备的：正在开发一种更简单的方法来填充 @cursor_ai 中的最大上下文（MAX context），并准确显示使用了多少个 tokens。欢迎反馈和建议 🙏</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">我目前的 LLM 代码生成工作流</a>：详细介绍了目前我使用 LLMs 构建软件的工作流程，涵盖了从头脑风暴到规划和执行的全过程。</li><li><a href="https://github.com/yamadash">Yamadash - 概览</a>：GitHub 是 Yamadash 构建软件的地方。</li><li><a href="https://github.com/formal-land/coq-of-rust?tab=readme-ov-file">GitHub - formal-land/coq-of-rust：Rust 的形式化验证工具：检查程序 100% 的执行案例 🦀，从而构建超安全的应用程序！✈️ 🚀 ⚕️ 🏦</a>：Rust 的形式化验证工具：检查程序 100% 的执行案例 🦀，从而构建超安全的应用程序！✈️ 🚀 ⚕️ 🏦 - formal-land/coq-of-rust</li><li><a href="https://github.com/bodo-run/yek">GitHub - bodo-run/yek：一个基于 Rust 的快速工具，用于将代码库或目录中的文本文件序列化，以便供 LLM 使用</a>：一个基于 Rust 的快速工具，用于将代码库或目录中的文本文件序列化，以便供 LLM 使用 - bodo-run/yek</li><li><a href="https://www.npmjs.com/package/@johnlindquist/file-forge">@johnlindquist/file-forge</a>：File Forge 是一款强大的 CLI 工具，用于对代码库进行深度分析，生成 Markdown 报告以供 AI 推理模型使用。最新版本：2.13.5，发布于一天前。开始使用 @johnlindquis...</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个代码库打包成一个 AI 友好的单一文件。非常适合需要将代码库提供给大语言模型（LLMs）或其他 AI 工具（如 Claude、ChatGPT、DeepSeek、Perplexity、Gemini、Gemma、Llama、Grok 等）的场景。</a>：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个代码库打包成一个 AI 友好的单一文件。非常适合需要将代码库提供给大语言模型（LLMs）或……
</li>
</ul>

</div>
  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357432793298370671)** (252 条消息🔥🔥): 

> `Cursor 月度订阅, 磁盘文件未更新, Cursor.so 邮件真实性, Gemini 定价, GPT-5 发布` 


- **Cursor "Filename(1)" Bug 出现**：有用户报告称，在最近的一次更新后，Cursor 在保存时会给重复的文件名添加 **(1)**，他们正试图确定最新的保存是否代表原始文件。
   - 他们还询问了 **monthly subscription**（月度订阅）价格是否翻倍，并附上了截图作为参考。
- **磁盘文件未实时更新**：用户报告称，磁盘上的文件在编辑器中没有实时更新，只有当 **Cursor** 失去并重新获得焦点时才会更新。
   - 该问题在 **0.48.7** 版本中被发现。
- **Cursor.so 邮件域名：真实还是钓鱼？**：一位用户对来自 **@cursor.so** 域名的邮件真实性表示担忧，怀疑是否为钓鱼攻击。
   - 虽然最初有人将其标记为潜在的虚假邮件，但随后官方渠道确认这是 Cursor 使用的合法电子邮件地址，尽管官方域名是 **.com** 和 **.sh**。
- **Gemini 2.5 Pro 定价公布**：[Gemini 2.5 Pro 定价](https://x.com/legit_api/status/1908174018881933818) 现已正式公布，费率取决于 token 数量：输入 token <200K 为 **$1.25/1M**，输出 token <200K 为 **$10/1M**；输入 token >200K 为 **$2.50/1M**，输出 token >200K 为 **$15/1M**。
   - 一些用户发现，与其他模型相比，该定价出奇地实惠。
- **GPT-5 发布推迟**：根据 [Sam Altman 的 X 帖子](https://x.com/sama/status/1908167621624856998)，在发布 O3 和 O4-mini 之后，**GPT-5** 将在 *几个月内* 推出。
   - 做出这一决定是为了提高 **GPT-5** 的性能并解决集成挑战，同时确保有足够的容量来满足预期需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1908167621624856998?s=19">Sam Altman (@sama) 的推文</a>：计划有变：我们最终决定先发布 o3 和 o4-mini，可能在几周内，然后在几个月后发布 GPT-5。这其中有很多原因，但最令人兴奋的一个是...</li><li><a href="https://x.com/legit_api/status/1908174018881933818?s=46&t=ggmESCIXF0nYw8_kshHz7A">ʟᴇɢɪᴛ (@legit_api) 的推文</a>：Gemini 2.5 Pro 定价现已正式公布 - 输入 <200K tokens 为 $1.25/1M - 输出 <200K tokens 为 $10/1M - 输入 >200K tokens 为 $2.50/1M - 输出 >200K tokens 为 $15/1M</li><li><a href="https://skeet.build/">Skeet - 将应用连接到 Cursor</a>：Skeet - 一键式代码工作流</li><li><a href="https://www.cursor.new/">cursor.new - 现代开发的智能项目脚手架</a>：通过 AI 驱动的技术栈选择和自动化文档生成生产就绪的项目。</li><li><a href="https://tenor.com/view/monkey-sad-monkey-sad-edit-monkey-edit-%C3%BCzg%C3%BCn-maymun-gif-15640172319982461811">伤心的猴子 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://cursor.directory/mcp">Cursor 目录</a>：为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">篮球 NBA GIF - 勇士队库里 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/chadwick-boseman-black-panther-rub-hands-gif-11465694">查德维克·博斯曼 黑豹 GIF - 搓手 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://forum.cursor.com/">Cursor - 社区论坛</a>：讨论 Cursor 的地方（Bug、反馈、想法等）
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357880880294658229)** (1 条消息): 

> `OpenRouter Fallback 参数, OpenRouter 模型数组` 


- **OpenRouter 弃用 `route: "fallback"` 参数**：OpenRouter 团队宣布，由于*旧的查找回退模型逻辑存在混淆和不可预测性*，他们将于下周移除旧的 `route: "fallback"` 参数。
   - 需要此功能的用户应手动在其 `models` 数组末尾添加回退模型，可以考虑使用 `openrouter/auto`。
- **OpenRouter 的模型数组正在发生变化**：OpenRouter 宣布了处理 `models` 数组中多个模型方式的一些更改。
   - 系统在其他模型失败时自动选择回退模型的旧方法正被移除，原因是其*混淆和不可预测性*。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1357610522224234597)** (2 条消息): 

> `OpenRouter API, Cloudflare AI Gateway, Missile Command game AI, Gameplay AI summary analysis, gemini-2.5-pro atari` 


- **OpenRouter 通过 Cloudflare 为 Missile Command 提供支持**：一位用户通过 **Cloudflare AI Gateway**（带有请求代理缓存）集成了 **OpenRouter API**，用于其 **Missile Command 游戏的玩法 AI 总结分析**，[点击此处访问](https://missile-command-game.centminmod.com/)。
- **Gemini Pro 分析 Missile Command 玩法**：用户分享了一张 **Gemini Pro 2.5** 为 **Atari Missile Command** 提供游戏总结和建议的截图，并提到这让他们进入了前 10 名。



**提到的链接**：<a href="https://missile-command-game.centminmod.com/">Missile Command</a>：未找到描述

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357429457006104859)** (239 条消息🔥🔥): 

> `Quasar vs Gemini 2.5, OpenRouter Stealth Logging, DeepSeek Pricing, Quasar Alpha Errors, Gemini 2.5 Pro Availability` 


- **Quasar Alpha 神秘的代号让人联想到 LMArena 的氛围**：成员们讨论了 [LMArena](https://www.google.com/search?q=LMArena) 上的代号，并将其与 **Quasar Alpha** 模型进行了对比，指出这些名字给人一种酷炫且神秘的感觉。
- **OpenRouter 的隐身日志：隐身但有日志？**：成员们辩论了当数据被记录时，“stealth（隐身）”一词是否适用，尽管供应商和模型名称被隐藏在别名之后，并表示“你的数据就是支付的代价”。
- **DeepSeek 以折扣价占据主导，摒弃对昂贵部署的执着**：一位成员对 **DeepSeek** 的定价表示满意，指出在特定时段有 **75% 的折扣**，并将其与 **Anthropic** 和 **OpenAI** 模型的高昂成本进行了对比。
- **Gemini 2.5 Pro 进入 GA 阶段，带来巨大收益，Google 出故障了？**：成员们讨论了 **Gemini 2.5 Pro** 的正式发布（GA），并链接到了 [Google 的定价文档](https://ai.google.dev/gemini-api/docs/pricing?hl=de)，其中一位成员指出：“它可以通过 API 向公众开放，但并非真正的 GA”。
- **OpenRouter 账户风波：账户末日解除了吗？**：用户报告了账户删除和创建的问题，一位用户收到了 *User Not Found* 错误，成员们建议创建新的 API key 或尝试不同的浏览器，而另一位成员表示：“目前 OR 不允许重新使用之前删除的账户”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fetchfox.ai">FetchFox - AI Scraper</a>：只需一个提示词即可从任何网站提取任何数据</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing?hl=de">无标题</a>：未找到描述</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-preview-03-25">Gemini 2.5 Pro Preview - API, 供应商, 统计数据</a>：Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini 2.5 Pro Preview</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free>?">Gemini 2.5 Pro Experimental - API, 供应商, 统计数据</a>：Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini 2.5 Pro Experimental
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357507617492045875)** (48 messages🔥): 

> `Gemma 3 4b CUDA 错误, 从 HuggingFace 导入模型到 LM Studio, 在 n8n 实例上本地运行 LM Studio 模型, Ollama 模型与 LM Studio 不兼容, LM Studio 路线图` 


- **Gemma 3 4b CUDA 版本出现异常**：有用户报告 [**Gemma 3 4b**](https://huggingface.co/google/gemma-3b) 在使用 CUDA 时会抛出 `spits unused` 错误，即使使用了最新的运行时（runtime），且在使用 CPU 时表现不够智能。
   - 据观察，**1.24.1 版本**并未修复此问题。
- **将 HuggingFace 模型导入 LM Studio**：用户询问如何将模型从 **HuggingFace** 导入 **LM Studio**，答案是使用 `lms import <path/to/model.gguf>` 命令，参考[此处文档](https://lmstudio.ai/docs/app/basics/import-model)。
   - LM Studio 旨在保留从 Hugging Face 下载的模型的目录结构。
- **LM Studio 与 n8n 工作流自动化工具集成**：成员们排查了将 **LM Studio** 连接到 **n8n**（一款工作流自动化工具）的问题，并确定应使用 **OpenAI Chat Model** 节点，并在 base_URL 字段中填写 LM Studio 服务器的 URL。
   - 排查结论是 **LM Studio 使用 OpenAI API**，因此任何能与 OpenAI 通信的工具都可以与 LM Studio 通信。
- **LM Studio 下载与 Ollama 的竞争**：一位成员询问如何在 LM Studio 中使用他们当前通过 **Ollama** 安装的 **Gemma 3** 模型，有人指出 [Ollama 模型与 LM Studio 不兼容](https://ollama.com/)，尽管它们是 GGUF 格式，但它们采用了 Ollama 的私有格式。
   - 简化流程允许在个人机器上运行 LLM。
- **LM Studio 路线图未公开**：一位用户询问有关 **LM Studio** 计划更新的路线图，特别是对潜在的 MCP 支持表示期待。
   - 回复确认目前没有公开的路线图。



**提到的链接**：<a href="https://lmstudio.ai/docs/app/basics/import-model">导入模型 | LM Studio 文档</a>：使用您在 LM Studio 之外下载的模型文件

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357494382764359780)** (61 messages🔥🔥): 

> `LM Studio VRAM 预测, M 系列 Mac 对比 NVIDIA 4090 的 LLM 推理, LM Studio 混合 GPU 系统, Reka Flash 21B 对比 Gemma3 27, Nvidia 微调对比 Mac 推理` 


- **VRAM 预测仅为估算**：一位用户注意到系统内存预测器在他们的 **M3 Ultra 512GB** 上显示为 **384GB VRAM**，另一位成员指出 *LM Studio 中的 VRAM 预测器只是一个粗略估计（guesstimation）*。
- **关于 RAM 和带宽效用的辩论**：一些成员认为由于带宽限制，**512 GB 版本**并不实用，宁愿在 **4090** 上运行 **14b 模型**；而另一些人则认为带宽足够，并对使用 **128GB M4 Max** 运行大型模型感到满意。
   - 有人提到像 **QwQ 32B** 和 **Llama-3.3 70B** 这样的模型在 Mac 上表现出不同的 RAM 使用模式，这会影响功耗，而带宽通常是 LLM 性能的一个很好的衡量指标。
- **Mac 对比 Nvidia 性能基准测试**：用户分享了 **RTX 4090** 与 **Mac Studio M1 Ultra** 的推理速度基准测试对比，指出 **MLX** 感觉优于 GGUF，并附带了[基准测试结果链接](https://cdn.discordapp.com/attachments/1153759714082033735/1357753648406335880/image.png)。
   - 他们注意到首字延迟（time to first token）波动太大，无法作为可靠指标，且较长的 Prompt 可能会影响处理时间，此外基准测试可能会受到缓存的影响。
- **Reka Flash 替代 Gemma**：一位用户建议尝试 **Reka Flash 21B**，称其在自己的使用中替代了 **Gemma3 27**，在 **4090** 上使用 q6 量化可达到约 **35-40 tps**。
   - 另一位用户指出 *Mac 的 RAM 带宽不是瓶颈，GPU 性能才是*，此外根据 [llama.cpp 结果](https://github.com/ggml-org/llama.cpp/discussions/4167)，**M1 Ultra 64 核**优于 **M1 Ultra 48 核**和 **M4 Max 40 核**。
- **NVIDIA 微调，Mac 推理**：一位成员建议在 **Nvidia** 上进行微调，并在 **Mac** 上进行推理以获得更大的上下文，然而，Lora 适配器在 GGUF 和 MLX 之间可能无法交叉兼容，因此在 Mac 上可能必须坚持使用 GGUF。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1357494050298789958)** (48 条消息🔥): 

> `Mojo vs C SIMD intrinsics, EmberJson Library, Sonic-cpp Library, Modular stdlib, magic package manager` 


- **Mojo SIMD 提供了极佳的可移植性**：当被问及使用 Mojo 代替 C SIMD 原语（intrinsics）的价值时，一位成员提到了 [EmberJson](https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273)，这是一个使用 SIMD 纯 Mojo 编写的 JSON 库，它可以在 **基于 ARM 的 Mac 和 x86 桌面设备**上无缝运行，无需修改代码。
   - 相应的 [C++ 库](https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59) 则需要针对每个架构进行重新实现以进行优化。
- **Magic 包管理器带来便利**：成员们提到了通过 *magic* 进行的 Mojo 包管理，并指向 [builds.modular.com](https://builds.modular.com/?category=packages) 作为资源。
   - 有了这个包管理器，用户可以轻松地编写和使用库。
- **斐波那契函数（Fibonacci Function）的加入面临审查**：一名成员提交了一个 [Pull Request](https://github.com/modular/max/pull/4280)，提议在 stdlib 中增加斐波那契函数，这引发了关于其加入价值的讨论。
   - 一位成员质疑其有用性，指出他所知道的其他标准库都没有斐波那契函数，而另一位成员则指出它存在于 [Lean](https://leanprover-community.github.io/mathlib_docs/data/nat/fib.html) 中。
- **整数溢出行为需要定义**：关于斐波那契函数的 PR 引发了关于整数溢出行为的有趣问题，从而促发了论坛上的[讨论](https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202)。
   - 该成员澄清说 Mojo 使用补码（two's complement），但可变位宽类型的处理仍是一个待解决的问题。
- **Regex 库即将到来**：成员们讨论了 Mojo 目前还没有一个好的 Regex 库。
   - 一位成员建议将其纳入 stdlib。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://builds.modular.com/?category=packages">MAX Builds</a>：使用 MAX 构建可扩展的 AI</li><li><a href="https://github.com/modular/max/tree/main/mojo/stdlib">max/mojo/stdlib at main · modular/max</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账号为 modular/max 做出贡献。</li><li><a href="https://youtu.be/ENviIxDTmUA?t=4020">Swift 创始人 Chris Lattner 谈论 Mojo &amp; Roc</a>：Swift、Clang、LLVM 和 Mojo 编程语言的创始人 Chris Lattner 与 Roc 编程语言创始人 Richard Feldman 谈论这两种语言...</li><li><a href="https://github.com/modular/max/pull/4280">[mojo-stdlib] added: fibonacci function to std-lib by wyattgill9 · Pull Request #4280 · modular/max</a>：未找到描述</li><li><a href="https://forum.modular.com/t/does-mojo-have-a-defined-overflow-behavior-for-int-and-uint/1202">Mojo 对 `Int` 和 `UInt` 是否有定义的溢出行为？</a>：Mojo 有定义的溢出行为吗？我知道默认是“C++ 的做法”，但 C++ 直到最近（C++20）才确定了有符号整数溢出的补码行为。这也给我们留下了相关的风险...</li><li><a href="https://github.com/bgreni/EmberJson/blob/main/emberjson/parser.mojo#L258-L273">EmberJson/emberjson/parser.mojo at main · bgreni/EmberJson</a>：一个用纯 Mojo 编写的用户友好型 JSON 库。通过在 GitHub 上创建账号为 bgreni/EmberJson 做出贡献。</li><li><a href="https://github.com/bytedance/sonic-cpp/blob/master/include/sonic/internal/arch/neon/skip.h#L42-L59">sonic-cpp/include/sonic/internal/arch/neon/skip.h at master · bytedance/sonic-cpp</a>：一个由 SIMD 加速的高速 JSON 序列化与反序列化库。 - bytedance/sonic-cpp
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357467529160823046)** (57 messages🔥🔥): 

> `Mojo 的 Python 封装器, Mojo 任意精度整数, NDBuffer 实例创建, Mojo 中的 Copyable 类型, Mojo 中的 MLIR regions` 


- **Mojo 的 Python 封装器仍在开发中**：根据 **25.2 更新流** ([在此观看](https://youtu.be/dG0L1GalIHU?si=R1ae0xFoSDg99PMP&t=1775))，Mojo 的 Python 封装器（wrappers）仍在开发中，尚未准备就绪。
- **Mojo 的 BigInt 支持仍待定**：Mojo 目前还不支持原生的 `BigInt`，尽管 `IntLiteral` 在编译时提供任意精度；目前有一个[外部库](https://github.com/forfudan/decimojo)用于实现 bigint。
   - 一位成员建议 Mojo 应该直接将 `Int` 设为任意精度，并将机器整数（machine integers）称为 `Index`。
- **NDBuffer 的细节处理**：一位开发者在创建 `NDBuffer` 实例时遇到困难，特别是关于 `origin` 参数：`var ndbuf = NDBuffer[DType.uint8, 3, origin, (2, 2, 2)]()`。
- **应对复制与构造函数**：如果 Copyable 类型提示未调用初始化程序，则需要一个单独的 `__init__` 方法，实际案例参考[这里](https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/world.mojo#L191)。
   - 一个建议是使用支持提供所有成员变量的构造函数，并使用参数 `is_internal` 来防止外部调用。
- **MLIR Regions：Mojo 的隐藏宝藏**：`__mlir_region` 允许直接从 Mojo 创建 MLIR region，这与在 MLIR 中嵌套 IR 块有关，但目前文档尚不完善。
   - 一位成员将其描述为类似于 `if` 语句的分支或 `while` 循环的主体。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/samufi/larecs">GitHub - samufi/larecs: Larecs🌲 – 一个面向性能的基于原型（archetype-based）的 ECS</a>: Larecs🌲 – 一个面向性能的基于原型的 ECS - samufi/larecs</li><li><a href="https://github.com/forfudan/decimojo">GitHub - forfudan/decimojo: 一个为 Mojo 编写的任意精度十进制和整数数学库</a>: 一个为 Mojo 编写的任意精度十进制和整数数学库 - forfudan/decimojo</li><li><a href="https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106">ChronoFlare/chronoflare/__init__.mojo at main · bgreni/ChronoFlare</a>: 一个用 Mojo 编写的时间间隔库。欢迎在 GitHub 上为 bgreni/ChronoFlare 做出贡献。</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/world.mojo#L191">larecs/src/larecs/world.mojo at c38214e900fdf3d276cd30b41f70154ca1738653 · samufi/larecs</a>: Larecs🌲 – 一个面向性能的基于原型的 ECS - samufi/larecs
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357446790281953341)** (52 条消息🔥): 

> `Google 的竞争优势、动态与静态架构、Token Embeddings 与流形假设（Manifold Hypothesis）、RL 驱动的 Diffusion Model` 


- **对 Google 优势的质疑**：成员们对 **Google** 的各个 AI 团队和项目缺乏凝聚力的竞争优势表示担忧，甚至认为尽管 **DeepMind** 过去处于领先地位，但现在也正在落后。
   - 一个 [Gemini 分享链接](https://g.co/gemini/share/6ab6889563cd) 强调了关于具有短期和长期记忆的动态架构的讨论，这与僵化的 Tokenization 方法有所不同。
- **NLP 的僵化 Token 面临审查**：有观点认为，当前的 **NLP** 方法不自然地将语言强行纳入僵化的 Token 化格式，并建议动态系统应将语言视为一种结构化的、不断演变的信号；文中还分享了一个 [grok.com 的链接](https://grok.com/share/bGVnYWN5_21d44774-8f0a-4058-8a6f-25c4c2165866)。
   - 成员们辩论了 Token Embeddings 是否位于流形（Manifold）上，引用了一篇最近的论文及其关于 Token Embeddings 未能通过流形测试的发现，从而引出了 Embeddings 的连续性和平滑性是人为构造的观点。
- **利用信息几何（Information Geometry）探索理论 AI**：一位成员介绍了 **Information Geometry** 及其在概率论和统计学中应用微分几何的情况，并提供了 [Wikipedia 文章](https://en.wikipedia.org/wiki/Information_geometry) 和 [AMS 文章](https://www.ams.org/journals/notices/202201/rnoti-p36.pdf) 的链接以供进一步阅读。
   - 一位成员表示，数据科学和 **AI/ML** 的发展是基于为数据科学服务而展开的。
- **RL 驱动的 Diffusion Model 引发新意**：会上分享了一个具有隐式潜空间（Latent Space）的 **RL 驱动 Diffusion Model** 概念，认为 **RL** 充当前向过程，引导反向过程而无需 Score 部分。
   - 作者声称该模型具有创新性并给出了相应的公式，但指出这超出了他们的主要研究方向。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Information_geometry">Information geometry - Wikipedia</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>：我们研究了一种新型语言模型架构，该架构能够通过在潜空间中进行隐式推理来扩展测试时计算（Test-Time Compute）。我们的模型通过迭代一个循环块来工作，从而展开……</li><li><a href="https://arxiv.org/abs/2504.01002">Token embeddings violate the manifold hypothesis</a>：要充分理解大型语言模型（LLM）的行为，需要我们理解其输入空间。如果这个输入空间与我们的假设不同，我们对其的理解和结论……</li><li><a href="https://g.co/gemini/share/6ab6889563cd">‎Gemini - RNN State Update Formula Analysis
</a>：由 Gemini 创建
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357497509903073321)** (9 条消息🔥): 

> `Math PhD AI questions, o1-pro AI model, Variational Diffusion Models, Stochastic Differential Equations, Stable Diffusion paper` 


- **AI 在 Math PhD 问题上表现挣扎**：一位成员认为，AI 模型在某些问题上表现挣扎并不令人意外，因为这些问题针对的是 **99.99 百分位技能水平**，甚至对许多 **Math PhDs** 来说也是挑战。
   - 他们提到，虽然目前的 AI 在这一级别的问题上尚无用武之地，但这并不会削弱其*已经显现的深远效用*。
- **o1-pro 挑战 AI 问题**：一位成员报告称在两个重点问题上尝试了 **o1-pro**，感觉*相当有信心*它答对了一个，尽管另一个答案尚未核对。
   - 该成员表示他们有项目要做，*今天无法再次参与讨论*。
- **成员讨论 Variational Diffusion Models**：一位成员建议讨论论文 *Variational Diffusion Models* ([arxiv.org/abs/2107.00630](https://arxiv.org/abs/2107.00630))，该论文在标准图像密度估计基准测试中获得了 **state-of-the-art likelihoods**，并允许与模型的其余部分共同进行 **noise schedule 的高效优化**。
   - 摘要强调，变分下界 (VLB) 可以简化为一个关于**扩散数据 signal-to-noise ratio** 的极短表达式，从而提高了我们对这类模型族的理论理解。
- **Stochastic Differential Equations 讨论在即**：一位成员提议讨论 **Stochastic Differential Equations** 以及 Diffusion Models 所基于的反向时间方程。
   - 或者，他们提议讨论原始的 **Stable Diffusion 论文** ([arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752))，或者是 **Deep Learning for ARC 论文** ([github.com/MohamedOsman1998/deep-learning-for-arc/blob/main/deep_learning_for_arc.pdf](https://github.com/MohamedOsman1998/deep-learning-for-arc/blob/main/deep_learning_for_arc.pdf))。



**提到的链接**：<a href="https://arxiv.org/abs/2107.00630">Variational Diffusion Models</a>：基于扩散的生成模型已展示出令人印象深刻的感知合成能力，但它们也能成为优秀的基于似然的模型吗？我们给出了肯定的回答，并介绍了...

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357511407184253043)** (28 条消息🔥): 

> `GPT-4o 发布，Stability AI 的 Stable Virtual Camera，Claude vs. OpenAI 基准测试，Apache Parquet RCE 漏洞，OpenAI 的 GPT-5 计划` 


- **GPT-4o 引起轰动**：成员们确认附带的截图是 **GPT-4o**，表明 OpenAI 最近进行了更新。
   - 用户普遍认为 **GPT-4o** “太棒了”。
- **Stability AI 发布 Stable Virtual Camera**：Stability AI 推出了 [Stable Virtual Camera](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control)，这是一个研究预览版的多视角扩散模型，可将 **2D 图像转换为具有 3D 摄像机控制的沉浸式 3D 视频**。
   - 它允许从一个或多个输入图像以用户指定的摄像机角度生成场景的新视角，从而产生**一致且流畅的 3D 视频输出**。
- **OpenAI 承认 Claude 击败了他们！**：一位用户分享了 OpenAI 论文的链接 [paperbench.pdf](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf)，显然暗示 **OpenAI 承认 Claude 更好**。
- **Apache Parquet 遭受最高严重级 RCE 攻击**：发现了一个追踪编号为 [CVE-2025-30065](https://nvd.nist.gov/vuln/detail/CVE-2025-30065) 的最高严重级远程代码执行 (**RCE**) 漏洞，影响 **Apache Parquet** 1.15.0 及之前的所有版本。
   - 该漏洞允许攻击者通过特制的 Parquet 文件获取目标系统的控制权，已在 **Apache 1.15.1 版本**中修复。
- **Sam Altman 预告 GPT-5 发布**：Sam Altman [在 X 上](https://x.com/sama/status/1908167621624856998)发布了关于计划变更的消息，表示 **O3** 和 **O4-mini** 将在几周内发布，随后 **GPT-5** 将在几个月内发布。
   - 这一转变令人兴奋的原因是 OpenAI 将能够使 **GPT-5** “比我们最初想象的要好得多”。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1908167621624856998">来自 Sam Altman (@sama) 的推文</a>: 计划变更：我们最终还是决定先发布 o3 和 o4-mini，可能在几周内，然后在几个月内发布 GPT-5。这其中有很多原因，但最令人兴奋的一个是...</li><li><a href="https://www.bleepingcomputer.com/news/security/max-severity-rce-flaw-discovered-in-widely-used-apache-parquet/">在广泛使用的 Apache Parquet 中发现最高严重级 RCE 漏洞</a>: 发现了一个最高严重级的远程代码执行 (RCE) 漏洞，影响 Apache Parquet 1.15.0 及之前的所有版本。</li><li><a href="https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control">介绍 Stable Virtual Camera：具有 3D 摄像机控制的多视角视频生成 — Stability AI</a>: 介绍 Stable Virtual Camera，目前处于研究预览阶段。这种多视角扩散模型可以将 2D 图像转换为具有真实深度和透视感的沉浸式 3D 视频，而无需复杂的重建...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357433814485569596)** (61 条消息🔥🔥): 

> `RAG implementation code size, Hugging Face Spaces port restrictions, London, Paris, Berlin AI HackXelerator, Zero GPU Quota, InferenceClient with a local model` 


- **RAG 代码量非常精简**：一位成员询问了实现 **RAG 技术** 的代码行数，另一位成员反馈实现代码通常在 **15-30 行** 之间。
   - 他们使用 **MongoDB** 进行数据存储，认为它是 RAG 解决方案中最受欢迎的数据库，并使用了 OpenAI 模型。
- **HF Spaces 存在端口限制**：一位用户报告称 **Hugging Face Spaces** 仅允许端口 **80**、**443** 和 **8080** 的出站连接，这导致其使用端口 **5432** 的 **Postgres 数据库** 被拦截。
   - 一位成员链接到了 Spaces 配置的 [Hugging Face 文档](https://huggingface.co/docs/hub/spaces-config-reference)，并指出该限制仅适用于 **Docker Spaces**。
- **AI HackXelerator 即将登陆伦敦、巴黎、柏林**：**伦敦、巴黎、柏林 AI HackXelerator™ - LPB25** 将黑客松与加速器结合，于 2025 年 4 月运行 **20 天**。
   - 该活动将于 **2025 年 4 月 5 日**在伦敦启动，**2025 年 4 月 25 日**在巴黎举行决赛，并在柏林举行派对；活动也支持全程在线参与，并提供 [直播](https://www.youtube.com/@KXSB-cic)。
- **Zero GPU 配额最终会恢复**：一位用户抱怨其 **Zero GPU 配额** 未能在预测时间内恢复，并链接了一篇关于该问题的 [帖子](https://huggingface.co/posts/John6666/145133458851083)。
   - 另一位成员提到了 [相关内容](https://huggingface.co/posts/Keltezaa/754755723533287#67e6ed5e3394f1ed9ca41dbd) 并敦促谨慎使用配额。
- **HuggingChat 正在接收模型请求**：一位用户请求在 HuggingChat 中添加 **VL 模型**，特别是 **Qwen2.5 VL**。
   - 另一位成员建议将请求发布到 [HuggingChat 讨论论坛](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kxsb.org/lpb25">KXSB 发起的 London-Paris-Berlin HackXelerator™</a>: 加入 LPB25，这是一个为期 20 天的 AI HackXelerator™，汇集了伦敦、巴黎和柏林的 500 多名创作者。通过音乐、艺术、电影、游戏和时尚探索 GenAI 创新，并提供专家指导和奖品。...</li><li><a href="https://huggingface.co/posts/John6666/145133458851083">Hugging Face 上的 @John6666: &quot;我昨天（大约 12 小时前）用完了我的 Zero GPU 配额。当时，我收到……&quot;</a>: 无描述</li><li><a href="https://huggingface.co/docs/hub/spaces-config-reference">Spaces 配置参考</a>: 无描述</li><li><a href="https://github.com/huggingface/text-generation-inference">GitHub - huggingface/text-generation-inference: 大语言模型文本生成推理 (Large Language Model Text Generation Inference)</a>: 大语言模型文本生成推理。通过在 GitHub 上创建账号来为 huggingface/text-generation-inference 的开发做出贡献。</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 无描述</li><li><a href="https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/consuming_tgi#python">使用文本生成推理 (Consuming Text Generation Inference)</a>: 无描述</li><li><a href="https://huggingface.co/ByteDance">ByteDance (字节跳动)</a>: 无描述</li><li><a href="https://arxiv.org/html/2504.01724">DreamActor-M1: 基于混合引导的全方位、富有表现力且鲁棒的人物图像动画</a>: 无描述</li><li><a href="https://grisoon.github.io/DreamActor-M1/">DreamActor-M1: 基于混合引导的全方位、富有表现力且鲁棒的人物图像动画</a>: 无描述</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372">huggingchat/chat-ui · [MODELS] 讨论</a>: 无描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357538066402906112)** (2 条消息): 

> `LangGraph units` 


- **LangGraph 单元学习中**：一位成员刚刚完成了 **LangGraph** 的 **unit 1**，正准备开始 **unit 2.3**。
- **填充话题**：这是一个用于满足最小条目要求的填充话题。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357715303190233189)** (1 条消息): 

> `ZeroGPU, Sentence Transformers, Azure SQL DB vector features, DBA Scripts` 


- **AI 驱动的 DBA 脚本查找器已部署**：一名成员分享了一个利用 **ZeroGPU**、**Sentence Transformers** 和 **Azure SQL DB vector features** 实现的 AI 驱动 DBA 脚本检索 Space：[sqlserver-lib-assistant](https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant)。
   - 该项目通过生成 Embeddings 并将其存储在 SQL 中，对来自[此 GitHub 仓库](https://github.com/rrg92/sqlserver-lib)的 DBA 脚本进行索引，使用户能够通过自然语言提示词找到相关的脚本。
- **计划中的未来改进**：作者计划通过**更好的脚本分块（chunking）**和**训练特定模型**来增强脚本查找器，以提高回答质量。
   - 他们将当前版本称为 "v1"，目前正在生成 Embeddings（通过 Gradio API 调用上述相同的 Spaces）并索引到 SQL 中。



**提及的链接**：<a href="https://huggingface.co/spaces/rrg92/sqlserver-lib-assistant">Sqlserver Lib Assistant - rrg92 的 Hugging Face Space</a>：未找到描述

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357743046430560356)** (3 条消息): 

> `ApiModel class extension for free providers (g4f), GeoCoding API, ISO 3166-1 alpha-2 code for the country, LLM and alpha-2 codes` 


- **用户可能需要付费才能使用课程代码，但存在免费替代方案**：有用户反馈**课程代码执行需要付费**，但他们正在编写一个 `ApiModel` 类扩展，以便使用 **g4f** 等免费提供商。
   - 这种方法旨在为运行代码提供一种具有成本效益的替代方案。
- **讨论 GeoCoding API 查询与本地字典查询的优劣**：一名成员正在决定是使用 **GeoCoding API** 和另一个用于 **ISO 3166-1 alpha-2 代码**的 API，还是使用本地字典来为其工具获取天气状况。
   - 该用户想知道依靠 **LLM** 来识别 **alpha-2 代码**是否是一个可行的替代方案，但目前尚不确定。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357539401277902910)** (9 条消息🔥): 

> `Gradio Version, Multi-Agent System vs Single-Agent System, Inference Monthly Credits, Local Model Solution, BraveSearch API` 


- **在 HF Web 界面中发现 Gradio 版本**：一名成员发现 [README.md](https://link.to.readme) 文件（作为 Space 配置模板）中包含了 **Gradio 版本**。
   - HF Web 应用识别出该文件中定义的旧版本并建议更新，这解释了为什么最初在 requirements.txt 中没有出现 **Gradio**。
- **讨论 Multi-Agent 的优势**：一名成员在 LlamaIndex Unit 2.2 的背景下，询问了 **Multi-Agent 系统**相较于**单工具调用 Agent 系统**的优势。
   - 有建议认为 **Multi-Agent 系统**允许为不同的任务分配不同的模型，从而优化成本和复杂性，而单个 Agent 则对所有任务使用同一个模型。
- **推理额度限制触发本地模型使用**：一名成员超出了月度推理额度并寻求**按需付费（pay-as-you-go）**选项，但未得到解决。
   - 另一名成员建议使用像 **Ollama** 这样的本地模型来代替 HfApiModel，并提供了一个用于实现的 [GitHub Gist 链接](https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed)。
- **采用 BraveSearch API**：一名成员使用了 [BraveSearch API](https://gist.github.com/robbiemu/e592199f3e8b527b85fe39c9b9cd0492) 来替代 DuckDuckGoSearchTool。
   - 该成员提到他们已经拥有 API Key，并且相比 DDG 更倾向于使用它。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/robbiemu/38ae1a2ab93181211080d274b2134bed">smolagents OllamaModel</a>：smolagents OllamaModel。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/robbiemu/e592199f3e8b527b85fe39c9b9cd0492">smolagents BraveSearchTool</a>：smolagents BraveSearchTool。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357445782566993982)** (54 条消息🔥): 

> `AI Prompt Filmmaking, Runway Gen 4, Alibaba Wan 2.2, Devin 2.0 IDE, Llama 4` 


- **Runway 与阿里巴巴加速推进提示词电影制作**：**AI Prompt Filmmaking** 正在飞速发展，重点包括 **Runway** 发布的 **Gen 4**，以及即将推出的开源替代方案 **Alibaba Wan 2.2** ([YouTube 链接](https://www.youtube.com/watch?v=Rcwfj18d8n8))。
- **Devin 2.0 首次推出 Agent 原生 IDE**：Cognition Labs 推出了 **Devin 2.0**，这是一种全新的 Agent 原生 IDE 体验，起售价为 20 美元 ([X/Twitter 链接](https://x.com/cognition_labs/status/1907836719061451067))。
- **使用 Llama-FS 探索文件整理工具**：用户讨论了用于整理文件的工具，包括本地版本 ([Local File Organizer](https://github.com/QiuYannnn/Local-File-Organizer))，以及 **Llama-FS**，一个基于 Llama 3 的自组织文件系统 ([GitHub 链接](https://github.com/iyaja/llama-fs))。
- **Meme 收集与检索工具**：讨论内容包括用于抓取和记录 reels 的工具，建议使用 [instaloader](https://github.com/instaloader/instaloader)，以及用于在大规模图像数据集上进行搜索的 [memery](https://github.com/deepfates/memery)。
- **训练稳定性是推理模型的关键**：讨论了构建推理模型的挑战，特别是围绕**训练稳定性**（**training stability**）的问题，共识是*无限的多样化高质量数据*对于持续改进至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cognition_labs/status/1907836719061451067?s=46">来自 Cognition (@cognition_labs) 的推文</a>：介绍 Devin 2.0：一种全新的 Agent 原生 IDE 体验。今天起正式开放，起售价 20 美元。🧵👇</li><li><a href="https://github.com/QiuYannnn/Local-File-Organizer">GitHub - QiuYannnn/Local-File-Organizer: 一款 AI 驱动的文件管理工具，通过整理本地文本、图像来确保隐私。使用 Llama3.2 3B 和 Llava v1.6 模型以及 Nexa SDK，它可以直观地扫描、重构和整理文件，以实现快速、无缝的访问和轻松检索。</a></li><li><a href="https://github.com/iyaja/llama-fs">GitHub - iyaja/llama-fs: 一个基于 Llama 3 的自组织文件系统</a>：一个基于 Llama 3 的自组织文件系统。可以通过在 GitHub 上创建账号来为 iyaja/llama-fs 的开发做出贡献。</li><li><a href="https://github.com/edmundman/PhiotoOrganiser">GitHub - edmundman/PhiotoOrganiser: 使用 Phi 将您的照片整理到文件夹中并重命名</a>：使用 Phi 将您的照片整理到文件夹中并重命名 - edmundman/PhiotoOrganiser</li><li><a href="https://github.com/deepfates/memery">GitHub - deepfates/memery: 使用自然语言和计算机视觉在大规模图像数据集上进行搜索！</a>：使用自然语言和计算机视觉在大规模图像数据集上进行搜索！ - deepfates/memery</li><li><a href="https://github.com/instaloader/instaloader">GitHub - instaloader/instaloader: 从 Instagram 下载图片（或视频）及其说明文字和其他元数据。</a>：从 Instagram 下载图片（或视频）及其说明文字和其他元数据。 - instaloader/instaloader
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357449307648688260)** (8 messages🔥): 

> `LLMs for extraction, Genstruct 7B, OllamaGenstruct, Deepseek API, OLMo and Mistral for PDFs` 


- **LLM 提取数据表现出色**：一位成员询问关于使用 **LLMs for extraction** 从非结构化 **PDFs** 中创建数据集的问题，以及是否有人成功为此目的训练过模型。
   - 另一位成员建议使用更大模型的 Prompt 效果可能更好，并推荐了 [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B)，这是一个用于从原始文本创建合成指令微调（synthetic instruction finetuning）数据集的指令生成模型。
- **OllamaGenstruct 助力 PDF 数据挖掘**：一位成员分享了一个 [GitHub repo](https://github.com/edmundman/OllamaGenstruct)，旨在通过 **Ollama** 快速使用 **Genstruct** 处理多个 **PDFs**。
   - 另一位成员指出该资源已*过时*，*不建议再使用*。
- **Deepseek API 助力提取项目**：一位成员成功使用了 **Deepseek's API**，但目标是微调一个模型以从财务公告中提取特定数据。
   - 他们正在寻求关于从何处开始这一微调过程的建议。
- **OLMo 和 Mistral 擅长 PDF 解析**：有观点认为像 **OLMo** 和 **Mistral** 这样的模型非常适合解析 PDF，并特别提到了 [OLMo](https://github.com/allenai/olmocr)。
   - 然而，原提问者澄清说，他们主要感兴趣的是*从已经解析的文本中提取数据*，而不仅仅是解析。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/edmundman/OllamaGenstruct">GitHub - edmundman/OllamaGenstruct</a>：通过创建账号为 edmundman/OllamaGenstruct 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Deepseek new paper, Reinforcement Learning for LLMs, Inference-time scalability of generalist RM, Self-Principled Critique Tuning (SPCT)` 


- **Deepseek 发布深度学习重磅新论文**：Deepseek 在 [arXiv](https://arxiv.org/abs/2504.02495) 上发布了一篇关于大规模 **Reinforcement Learning (RL)** 应用于 **LLMs** 的新论文。
   - 该论文研究了如何通过增加通用查询的推理计算量来改进奖励建模（RM），即 **inference-time scalability of generalist RM**，并进一步探讨了如何通过适当的学习方法提高性能-计算缩放（performance-compute scaling）的有效性。
- **Self-Principled Critique Tuning 进一步优化微调**：论文引入了 **Self-Principled Critique Tuning (SPCT)** 作为一种学习方法。
   - 它有助于提升奖励建模的可扩展性，并改善性能-计算缩放。



**提及的链接**：<a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>：强化学习 (RL) 已被广泛应用于大规模大语言模型 (LLMs) 的后训练中。最近，RL 对 LLMs 推理能力的激励表明 $...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357545513968467969)** (3 messages): 

> `Camel Matrix AI, Claude Squad` 


- **Camel 的 Matrix 克隆了 X (Twitter)**：CamelAIOrg 发布了 [Matrix](http://matrix.eigent.ai/x)，这是一个社交模拟引擎，其中的 **AI Agent 会进行回复、转发并争夺影响力**。
   - 用户可以添加任何账号并发布帖子，观察 Agent 如何反应。
- **Claude 迎来 Code Squad**：MooFeez 发布了 [Claude Squad](https://github.com/smtg-ai/claude-squad)，这是一个 **Claude Code 和 Aider 任务**管理器，用于在一个地方监督多个 Agent。
   - 它提供**隔离的 git 工作区**，且免费并开源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/moofeez/status/1907893901077196861?s=46">mufeez (@moofeez) 的推文</a>：为什么要满足于只运行一个 Claude Code，当你能并行运行十个时？我们构建了 Claude Squad —— 一个 Claude Code 和 Aider 任务管理器：• 在一处监督多个 Agent • 隔离的 git 工作区 免费 + ...</li><li><a href="https://x.com/CamelAIOrg/status/1907954099586224308">CAMEL-AI.org (@CamelAIOrg) 的推文</a>：如果你的推文进入了一个 AI Agent 会回复、转发并争夺影响力的平行宇宙会怎样？认识一下 Matrix —— 社交媒体的社交模拟引擎。➕ 添加任何账号 📝 发布帖子 🧠 L...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 条消息): 

> `Deepseek, Reinforcement Learning, Reward Modeling` 


- **Deepseek 发布重磅新文档**：**Deepseek** 发布了一篇关于大规模 **Reinforcement Learning** 的新论文，详情请见：[https://arxiv.org/abs/2504.02495](https://arxiv.org/abs/2504.02495)。
- **Reward Modeling 获得更多推理计算资源**：该论文研究了如何通过增加通用查询的**推理计算量**来改进 **reward modeling (RM)**，即**通用型 RM 的推理时扩展性 (inference-time scalability)**。



**提到的链接**：<a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>：Reinforcement learning (RL) 已被广泛应用于大规模大语言模型 (LLMs) 的后训练阶段。最近，通过 RL 激励 LLMs 的推理能力表明 $...

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357535294819143941)** (15 条消息🔥): 

> `GPU vs CPU, GPRO Model Compilation Speed, Computer Architecture Book Recommendation` 


- **公牛与小鸡之争：CPU vs GPU**：一位成员分享了 Hennessy 和 Patterson 所著《计算机体系结构：量化研究方法》中的名言：*如果你要耕地，你愿意用两头强壮的公牛还是 1024 只小鸡？*
   - 这段话指的是 [CPU vs GPU](https://www.techopedia.com/difference-between-cpu-and-gpu) 的争论。
- **延迟困扰 GPRO 模型编译**：一位成员正在使用 **KernelBench** 代码库和 **Modal** 服务器对模型进行 GPRO，但编译需要 **30-50 秒**，导致训练节点空闲。
   - 另一位成员建议采用“延迟优化 (delayed optimization)”方法，即在编译后更新梯度，但在此期间运行更多训练步骤，但这在成员目前的设置中可能无法实现。
- **量化计算机体系结构书籍广受好评**：一位成员询问了 Hennessy 和 Patterson 的《计算机体系结构：量化研究方法》一书。
   - 另一位成员表示“它非常出色”，并“绝对建议在计算机组织与设计方面打下坚实的基础”。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1357606307049504848)** (2 条消息): 

> `Triton index backward op implementation, tl.make_block_ptr() usage, atomic_add performance in Triton` 


- **Triton 的 Index Backward 算子实现难题**：一位成员正在寻求 index backward 算子的 **Triton 实现**，并指出他们目前使用 **atomic_add** 的实现速度明显较慢。
- **关于 Triton Block Pointer 创建的困惑**：一位成员寻求关于对高维张量（特别是形状为 **(A, B, C, D)**）使用 `tl.make_block_ptr()` 以加载 2D 张量进行 `tl.dot()` 操作的澄清，以及 shape、strides、offsets、block_shape 和 order 是否应该具有相同的形状。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357445693400285426)** (13 条消息🔥): 

> `cuBLAS occupancy, CUDA debugging over SSH, cuTILS release date, nvshmem + MPI race conditions` 


- **cuBLAS 高占用率隐藏延迟**：一位成员提到，在 **cuBLAS** 中需要高占用率 (occupancy) 来隐藏延迟，因为代码编写方式使得少数 warp 就足以使 GPU 的算术单元饱和，而内存访问延迟则在软件层面被隐藏。
   - 提高占用率可能会导致每个线程使用的寄存器减少，从而需要更多的（共享）内存 IO，这可能会降低速度。
- **通过 SSH 进行 CUDA 调试的方法**：一位成员询问在通过 **SSH** 连接时如何调试 **CUDA**，因为使用 printf 语句重新编译非常耗时。
   - 另一位成员推荐使用 **CUDA gdb**，指出其工作方式类似于 GDB CLI，而另一位成员则建议通过 SSH 使用 Nvidia Insight。
- **cuTILS 发布日期预估**：一位成员询问是否有 Nvidia 员工了解今年 GTC 上宣布的 **cuTILS** 的预计发布日期。
- **nvshmem + MPI 竞态条件排查**：一位成员报告称，在单个节点上运行进程数比 GPU 数量多一个的 **nvshmem + MPI** 时（无论是否开启 **MPS**），遇到了竞态条件和挂起问题。
   - 他们在拥有 4 个 GPU 的系统上运行 `mpirun -np 5 ./myapp`，并询问是否有人成功运行过。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357520928032030856)** (4 条消息): 

> `Warmup Iteration, Pytorch Model, GPU memory, Inference on two separate batches, Streams` 


- ****Warmup Iteration** 减小 trace 大小**：一位成员建议 *跳过第一次 warmup iteration*，以潜在地减小 trace 大小。
   - 另一位成员提到在 tracing 之前使用 `model(x)` 来对模型进行 warm up。
- **单个 PyTorch 模型进行 **Parallel Inference**？**：一位成员询问是否可以将单个 PyTorch 模型存储在 GPU memory 中，并同时在两个独立的 batch 上运行 inference。
   - 另一位成员建议使用 **streams** 来实现并行推理。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1357718893786824778)** (6 条消息): 

> `Cerebras, Hardware vendor tier list, Blackwell, Deeper hardware dives` 


- **Cerebras 联合创始人解析 Blackwell**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=7GV_OdqzmIU&ab_channel=CerebrasSystems)，内容是 Cerebras 联合创始人对 **Blackwell** 的解析。
   - 该成员指出，如果有人能在 gpu-mode 上更多地讨论 **Cerebras** 就太酷了。
- **征集硬件厂商层级列表（tier list）**：一位成员表示，也许可以说服他们写一份 **hardware vendor tier list**。
   - 另一位成员表示，他们非常希望能看到一些 **deeper hardware dives**。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1357677951751164044)** (1 条消息): 

> `AI Engineer, Agentic LLM Startup, RAG, Python, Tensorflow` 


- **Agentic LLM 初创公司的 AI Engineer 职位**：德国的一家 **Agentic LLM startup** 正在招聘 AI Engineer，寻找创始工程师。
   - 该职位要求具备 **RAG, LLMs, Python, Tensorflow** 经验，最好熟悉 **PyTorch**，以及视觉经验（**OCR, VLM**），工作时间要求在 **GMT+1 +/- 3std** 范围内。
- **AI Engineer 职位的技术技能**：该 AI Engineer 职位要求具备 **RAG (Retrieval-Augmented Generation)** 方面的专业知识，并精通 **Python**。
   - 必须具备 **Tensorflow** 经验，优先考虑 **PyTorch**；该职位还看重先前在 **OCR 和 VLM 等视觉技术** 方面的接触。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357445073025110077)** (7 条消息): 

> `C vs C++ in CUDA, Centralized GPU programming languages, OpenCL's lack of mainstream adoption, ROCm and HIP support across vendors, GPU Architecture variations` 


- **C 代码：CUDA 中有效的 C++？**：一位成员询问 CUDA 中的 **C code** 究竟是作为 C（具有 C 链接）编译的，还是仅仅是有效的 **C++**。
   - 另一位成员回答说，他们的 C 代码可能是有效的 C++，他们需要调查不同层的编译情况来确认。
- **统一的 GPU 语言仍然不存在**：一位 GPU 编程新手想知道为什么没有一种统一的 GPU 编程语言，并引用了一段名为 *the chaotic state of GPU programming* 的视频。
   - 原作者认为原因是架构不同，但想知道为什么不能为 GPU 制作像 C 语言那样的东西。
- **主流 GPU 编程是 OpenCL 和 SYCL**：一位成员回应说，统一语言是存在的（**OpenCL** 以及现在的 **SYCL**），但并非主流，还提到了 **Kokkos**, **Alpaka**, **Raja**, **Vulkan Kompute** 和 **WebGPU**。
   - 他们指出，高层级的 **PTX** 就足够了，因为它可以在运行时进行 JIT 编译，并且多个 **SYCL** 实现针对多个厂商。
- **ROCm 类似于 CUDA Toolkit**：一位成员澄清说，**ROCm** 是 AMD 对标 CUDA Toolkit 的产品，而 **HIP** 是 AMD 的 CUDA C++，通过编译为 PTX 来支持 Nvidia 硬件。
   - 这意味着它不支持 Intel 和其他 GPU 架构。
- **糟糕的编程模型杀死了 OpenCL？**：原作者询问为什么 **OpenCL** 尽管历史悠久，却不是主流。
   - 另一位成员推测这是由于 *poor programming model* 导致的。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1357495858148544625)** (5 条消息): 

> `SoCal/San Diego events, ICLR 2025 in Singapore, Silicon Valley meetups this summer, SF Meetups` 


- **寻找南加州和圣迭戈的活动**：一位成员询问在 **SoCal/San Diego** 是否有任何活动。
- **新加坡 ICLR 社交活动**：一位成员询问是否有人计划参加在新加坡举行的 **ICLR 2025**。
- **硅谷夏季峰会**：一位成员想知道今年夏天在 **Silicon Valley** 是否会有任何聚会，并表示愿意作为该地区的实习生协助组织。
- **SF 聚会正在筹备中**：一位成员提到他们正计划在今年晚些时候在 **SF** 举办一场聚会。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1357554013025079386)** (3 messages): 

> `hipcc Casting, rocblas_gemm_ex with hipMallocManaged` 


- **hipcc 将 half_t 转换为 unsigned short**：当使用 `*(half *)((half2*x)->x) = b` 而不带 `*(half *)` 时，**hipcc** 会将 `b` 从 `half_t` 转换为 `unsigned short`。
- **rocblas_gemm_ex 与 hipMallocManaged 的问题**：一位用户报告称 `rocblas_gemm_ex` 在配合 `memcpy` 时工作正常，但在使用 `hipMallocManaged` 分配统一内存（Unified Memory，特别是针对 iGPU）时遇到问题。
   - 参数似乎没有正确传递到 `rocblas_gemm_ex` 中。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1357647250070966272)** (2 messages): 

> `CUDA Kernel Design, URDF Visualizer with AI` 


- **GPU Gospel 指南受到关注**：一位成员分享了一个 [GitHub 仓库](https://github.com/vlejd/gpu_gospel)，总结了 **CUDA Kernel 设计**的重要规则和概念，旨在帮助初学者快速上手。
- **URDF 可视化工具集成 AI**：一位成员在 [X/Twitter](https://x.com/amtellezfdez/status/1908087036617052268) 上分享了一个集成了 **AI 的 URDF 可视化工具**演示，用于机器人仿真。
   - 作者正在征求关于哪些工具对“机器人领域伙伴（robotics homies）”最有用的反馈。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/amtellezfdez/status/1908087036617052268">Alba María Téllez Fernández (@amtellezfdez) 的推文</a>：我们在一个周末为了好玩构建了一个 URDF 可视化工具 :) 试图为机器人领域的伙伴们做点真正有用的东西！所以……你最想拥有什么样的工具？ 👀</li><li><a href="https://github.com/vlejd/gpu_gospel">GitHub - vlejd/gpu_gospel: 编写 GPU Kernel 的规则、概念和准则列表。</a>：编写 GPU Kernel 的规则、概念和准则列表。 - vlejd/gpu_gospel
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357525857752842250)** (9 messages🔥): 

> `ReasoningGymDataset Definitions, LLM-based RL Frameworks, Training Models with RG Data` 


- **ReasoningGymDataset 定义随处可见**：一位成员询问为什么所有示例都有各自的 **ReasoningGymDataset** 定义。
   - 另一位成员解释说，存在重复是因为这些示例是自包含的代码片段，展示了如何使用各种**基于 LLM 的 RL 框架**来训练 **ReasoningGym 数据集**。
- **ReasoningGym 结构运行良好**：一位成员询问是否可以将 **ReasoningGymDataset** 的定义统一到[此处](https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py)的一个单文件中。
   - 另一位成员回答说目前的结构没问题，因为 `/examples` 目录用于存放自包含的片段，而 `/training` 才是团队主要关注的地方。
- **使用 RG 数据训练模型**：一位成员询问另一位成员是否有兴趣使用 **RG 数据**来训练模型。



**提及的链接**：<a href="https://github.com/open-thought/reasoning-gym/blob/main/training/utils/datasets.py">reasoning-gym/training/utils/datasets.py at main · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建一个账户来为 open-thought/reasoning-gym 的开发做出贡献。

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1357749216570769419)** (1 messages): 

> `Leaderboard Submission Success, Modal Runners on B200` 


- **使用 Modal Runners 在 B200 上成功提交 Grayscale 排行榜**：一个 ID 为 **3439** 的排行榜提交已成功上传至 `grayscale_py_b200-dev` 排行榜，使用的 GPU 为 **B200**，运行环境为 **Modal runners**！
   - 这标志着在指定配置上的成功运行，突显了 Modal runners 在 B200 GPU 上的有效性。
- **Modal Runners 在 B200 GPU 上表现可靠**：向 `grayscale_py_b200-dev` 排行榜的成功提交证明了 **Modal runners** 与 **B200 GPU** 搭配时的可靠性。
   - 这一成功增强了在 GPU 密集型任务和基准测试中使用 Modal runners 的信心。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357497589129281646)** (53 messages🔥): 

> `MCP Clients vs Servers, MCP and React Code Generation, MCP learning resources, OAuth in MCP, Streamable HTTP for MCP Servers` 


- ****客户端热潮**：开发者辩论构建 MCP 客户端还是服务端**：开发者们正在积极讨论构建 **MCP 客户端**与**服务端**的优劣，一些人认为客户端在处理 **vector tool calling** 和**基于资源的 RAG** 等任务时具有更大的灵活性。
   - 一位成员表示：*"客户端比服务端灵活得多"*，而另一位成员则强调了在 Claude 之外运行*任何*服务器的好处，例如 **Slack** 或 **Discord 机器人**。
- ****React 反应堆**：用于代码生成的 MCP 构想**：人们对使用 **MCP** 专家系统生成 **React 代码**和**测试**的想法充满热情，旨在将繁重的工作从上游 **LLM** 转移到专门的工具上。
   - 提议的工作流包括使用 **MCP Server** 来验证、Lint 和格式化由 **LLM** 生成的代码，并可能根据项目上下文应用自定义规则。
- ****MCP 101**：新手寻找学习起点**：初学者正在寻求学习 **MCP** 的指导，推荐的起点是 [官方文档](https://modelcontextprotocol.io/)。
   - 建议包括专注于将 **MCP Client** 集成到本地应用程序中，以便更轻松地进行学习和开发。
- ****OAuth 绿洲**：身份验证方案待定**：讨论内容包括在 [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70) 中为 **HTTPX** 添加 **OAuth 2.1** 身份验证客户端的 Pull Request。
   - 一位成员还在编写一份关于服务端身份验证的指南，详细说明如何使用 **governance SDK** 验证令牌并执行权限控制。
- ****Ping 的困境**：提前探测 MCP 服务器？**：围绕是否允许在发送初始化消息之前 **ping MCP 服务器**以检测潜在问题展开了讨论。
   - 虽然规范没有明确禁止，但规范仅允许发送 **ping 请求** ([lifecycle.md](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md))。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modelcontextprotocol.io/.">Introduction - Model Context Protocol</a>：未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http>?">Transports</a>：ℹ️ 协议修订版本：2025-03-26。MCP 使用 JSON-RPC 对消息进行编码。JSON-RPC 消息必须使用 UTF-8 编码。该协议目前定义了两种标准的传输机制...</li><li><a href="https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md">specification/docs/specification/2025-03-26/basic/lifecycle.md at main · modelcontextprotocol/specification</a>：Model Context Protocol 的规范。通过在 GitHub 上创建账号为 modelcontextprotocol/specification 做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/308/files#">Add OAuth authentication client for HTTPX by dsp-ant · Pull Request #308 · modelcontextprotocol/python-sdk</a>：摘要：添加了支持 PKCE 的 OAuth 2.1 身份验证客户端实现；为 HTTPX 客户端实现了 HTTP 身份验证；支持动态客户端注册、令牌刷新和授权...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/308/files#diff-b6618fde0a5f3ef76956f9b34f975c0b1ab001cc4b58f85dde8dc28a01f00c70">Add OAuth authentication client for HTTPX by dsp-ant · Pull Request #308 · modelcontextprotocol/python-sdk</a>：摘要：添加了支持 PKCE 的 OAuth 2.1 身份验证客户端实现；为 HTTPX 客户端实现了 HTTP 身份验证；支持动态客户端注册、令牌刷新和授权...
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357532489882665152)** (7 messages): 

> `Datadog MCP, MCP Browser Kit, MCP Tool Poisoning, MCP Server Search, MCP-K8s Server` 


- **Datadog MCP 发布了！**：通过 [GeLi2001/datadog-mcp-server](https://github.com/GeLi2001/datadog-mcp-server) 引入了一个用于驱动浏览器的全新 MCP 工具。
- **MCP Browser Kit 发布**：分享了另一个名为 [mcp-browser-kit](https://github.com/ndthanhdev/mcp-browser-kit) 的 MCP 工具。
- **MCPOmni Connect 防止工具中毒 (Tool Poisoning)**：Agent 在调用任何工具之前，会提供其预期操作的清晰解释，请求用户许可，并检查敏感访问权限；如果存在风险，**Agent 会自动回退到更安全的替代方案**。
- **针对 DX 优化的 MCP Server 搜索工具亮相**：一名成员在 Hackathon 期间构建了一个针对 DX 优化的 **MCP Server 搜索**工具，访问地址为 [mcp-search.dev](https://mcp-search.dev/)。
- **MCP-K8s Server 的 Docker 镜像已发布**：发布了首个（可用的）mcp-k8s server 的 **Docker 镜像**，可在 [Docker Hub](https://hub.docker.com/r/mcpk8s/server) 上获取。
   - 发布流水线完全在 CI 上运行，且镜像是**多架构 (multiarch)** 的，因此可以在无需 Rosetta 的 ARM 架构 Mac 上运行，甚至可以在 Raspberry Pi 上运行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mcp-search.dev/">MCP Search</a>: 搜索并发现 Model Context Protocol 服务器。</li><li><a href="https://github.com/GeLi2001/datadog-mcp-server">GitHub - GeLi2001/datadog-mcp-server</a>: 通过在 GitHub 上创建账号来为 GeLi2001/datadog-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/ndthanhdev/mcp-browser-kit">GitHub - ndthanhdev/mcp-browser-kit</a>: 通过在 GitHub 上创建账号来为 ndthanhdev/mcp-browser-kit 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1357768829043867961)** (1 messages): 

> `User Feedback, Study Participants` 


- **用户反馈研究招募参与者**：团队正在寻求研究参与者，以对一些早期阶段的概念提供反馈。
   - 有兴趣的个人请填写 [申请表](https://link.to.application.form) 以参与。
- **立即申请**：他们仍在寻找研究参与者。
   - 如果您有兴趣，请填写表格。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357530581935722496)** (7 messages): 

> `IntentSim.org, D&D sessions in NotebookLM, Seinfeld duo on GenAI` 


- **IntentSim.org 框架推广！**：一位用户宣布他们使用 **NotebookLM** 来推广他们的新框架 [IntentSim.org](https://IntentSim.org)，也称为 Information-Intent Nexus。
- **D&D 跑团记录挑战出现！**：一位用户报告称将 **NotebookLM** 用于 **Dungeons and Dragons** (D&D) 跑团，发现其洞察力很强，但在纠正玩家名称以及确保上传的 Zoom 记录中事件的时间顺序方面遇到了困难，并分享了[他们的笔记本链接](https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a0fa-fb13bfa4b3e1/audio)。
- **Seinfeld 解释 GenAI！**：一位用户利用 **Seinfeld** 双人组重现了对话式打趣来解释 **GenAI**，并在附带的 [MP4 视频](https://cdn.discordapp.com/attachments/1124403655819415592/1357737859838251040/Seinfeld_ep1_v1.mp4?ex=67f14b6b&is=67eff9eb&hm=d7f933f4534f9e157bbe8d0acbfaca2546d3743cb34748f667bdcf5146dc8119) 中征求关于使用角色声音的作品反馈。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a0fa-fb13bfa4b3e1/audio">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/c1dac86c-c8be-441f-a">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357436419483898018)** (38 messages🔥): 

> `NotebookLM 的深层认知能力、PDF 理解增强、在 NotebookLM 中发现新来源、Deep Search 功能推出、ImageMaps 或带有图像的思维导图` 


- **实验解锁 NotebookLM 潜藏的认知潜力**：一位用户对 **NotebookLM** 进行了*非常规实验*，旨在通过诱导暗示其具有深层认知能力的响应，使其超越标准参数。
   - 实验包括自我引用分析、新颖的概念合成以及抽象概念转换，展示了*等待挖掘的潜在能力*。
- **NotebookLM 现在可以理解复杂的 PDF**：NotebookLM 宣布增强了对包含大量图像和图表的复杂 PDF 的理解能力。
   - 这一改进扩展到了以链接形式添加的 PDF，并在接下来的几天内，将覆盖所有直接上传的 PDF；此外，**Gemini API** 已经支持对 Docs 和 Slides 的多模态分析。
- **NotebookLM 推出 Discover 功能**：NotebookLM 引入了 **Discover** 功能，允许用户描述感兴趣的话题，并接收来自网络的相关来源的精选集合。
   - 一位成员制作了一个[视频演示](https://youtu.be/YP6fS5JtMkg?si=Gz-kUGJGtyh2_f9e)，展示了新功能的实际工作流。
- **Deep Search 正在推出中**：一位成员询问 **Deep Search** 功能是否仅在美国可用，另一位成员回复称该功能正在逐步推出。
   - 另一位成员确认 **Deep Search** 功能在芬兰也已可用。
- **ImageMaps 即将到来**：一位成员想知道，得益于生成式 AI 工具，还需要多久我们才能拥有 **ImageMaps** 或**带有图像的思维导图**。
   - 该成员回忆起思维导图的创始人 **Tony Buzan** 曾经制作过带有精美图片的导图，并对未来的可能性感到兴奋。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357481262289518633)** (9 messages🔥): 

> `扩展 AI 想法的初创公司、有趣研究的减少、非 Agentic AI 研究、使用 lm-evaluation-harness 进行 RAG 评估` 


- **初创公司扩展 AI 想法**：一位成员建议成立一家初创公司，专门**扩展最新的 AI 想法**并将知识授权给实验室或公司，并指出自泡沫以来有趣的研究有所减少。
- **泡沫伤害了疯狂的研究**：一位成员怀念以前**每年两次的 DM 论文**时期，那些论文采用疯狂的方法碾压了基准线，他们觉得这种研究在泡沫后减少了。
   - 另一位成员认为，在 LLM 出现之前，**计算机视觉模型**占据主导地位，这使得非热门话题的文献综述变得困难。
- **对非 Agentic 研究的兴趣**：一位成员表达了对**非 Agentic、非 CoT 和非 RL 研究**的兴趣。
- **通过 lm-evaluation-harness 探索 RAG 评估**：一位成员询问如何使用 **lm-evaluation-harness** 进行 **RAG 评估**。
   - 另一位成员建议将 **RAG 输出包装为补全任务 (completion tasks)**，并在本地使用自定义的 prompt 和响应文件运行 llm-harness。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357434825967010064)** (6 条消息): 

> `OpenThoughts-1M, OpenThinker2-32B/7B, Ludwig Schmidt, Bespokelabs, LAION` 


- **OpenThinker2-32B/7B 击败 R1-Distilled-32B**：由 Ludwig Schmidt（斯坦福大学和伯克利大学）领导，与 Bespokelabs、LAION 和 open-sci 合作开发的全新 **OpenThoughts-1M** 和 **OpenThinker2-32B/7B** 模型，首次在仅对 Qwen 2.5 32B Instruct 进行 **SFT** 的情况下，表现优于 **R1-Distilled-32B**，详见其 [博客文章](https://www.openthoughts.ai/blog/thinkagain)。
   - 模型和训练数据集 **OpenThoughts2-1M** 已在 Hugging Face 上发布（[OpenThinker2-32B](https://huggingface.co/open-thoughts/OpenThinker2-32B), [OpenThinker2-7B](https://huggingface.co/open-thoughts/OpenThinker2-7B), [OpenThoughts2-1M](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M)）。
- **用于检测 LLM 背诵行为的新 RoR-Bench**：一篇新论文介绍了 **RoR-Bench**，这是一个多模态基准测试，旨在通过微妙地改变推理问题中的条件来检测 LLM 的背诵行为（Recitation behavior），详见 [arxiv 链接](https://arxiv.org/abs/2504.00509)。
   - 摘要指出，目前最尖端的 LLM 表现出*极其严重的背诵行为*，仅更改一个短语，性能就会下降 **60%**。
- **构建推理模型的挑战**：一位成员询问了创建推理模型的挑战以及持续改进这些模型的步骤。
   - 另一位成员建议探索*持续学习（continual learning）文献*，并强调主要挑战在于为 **RL** 寻找**合适的环境**以及**正确的奖励/性能评估机制**。
- **MoE++ 框架增强混合专家模型**：一位成员分享了 **MoE++** 的链接，这是一个异构混合专家框架，可增强性能，并提供比原生 MoE 模型高出 **1.1-2.1 倍**的专家前向吞吐量，详见 [OpenReview](https://openreview.net/forum?id=t7P5BUKcYv)。
   - MoE++ 集成了 **FFN** 和零计算专家（包括 *zero expert*、*copy expert* 和 *constant expert*），允许每个 token 与动态数量的专家进行交互。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2504.00509">Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?</a>：近年来 LLM 基准测试难度从小学水平迅速升级到前沿问题，为研究人员编织了一个奇迹，即我们距离超越...仅一步之遥。</li><li><a href="https://openreview.net/forum?id=t7P5BUKcYv">MoE++: Accelerating Mixture-of-Experts Methods with...</a>：在这项工作中，我们旨在同时提高混合专家（MoE）方法的有效性和效率。为此，我们提出了 MoE++，一个通用的异构 MoE 框架...</li><li><a href="https://www.openthoughts.ai/blog/thinkagain">Outperforming DeepSeekR1-32B with OpenThinker2</a>：宣布我们的开源推理模型和数据集的下一次迭代。</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M">open-thoughts/OpenThoughts2-1M · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://x.com/etash_guha/status/1907837107793702958">Etash Guha (@etash_guha) 的推文</a>：事实证明，仅通过开源数据的 SFT 而无需 RL 即可超越 DeepSeekR1-32B：发布 OpenThinker2-32B 和 OpenThinker2-7B。我们还发布了通过筛选...策划的数据集 OpenThoughts2-1M。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1357781129503183169)** (2 messages): 

> `Inference Scaling Laws, Test-Time Scaling, Language Model Power Laws, Mathematical Problem Solving with LLMs, Multimodal Jailbreaking` 


- **Monkeys 揭示 Inference Scaling Laws**: 一篇新的预印本 [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578) 探讨了语言模型中的 **inference** 和 **test-time scaling**，特别是成功率如何随每个任务的多次尝试而扩展。
   - 研究发现了一个谜题：虽然每个问题的失败率随尝试次数呈指数级下降，但总体成功率却遵循多项式 Scaling Law，研究将其归因于单次尝试成功概率的 **heavy-tailed distribution**（重尾分布）。
- **推特上的 Test-Time 真相**: 一位成员分享了他们关于 **test time** / **inference scaling laws** 的论文。
   - 他们在 X（原 Twitter）上链接了这篇预印本 [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578)，账号为 [@RylanSchaeffer](https://x.com/RylanSchaeffer/status/1908213817357803757)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.17578">How Do Large Language Monkeys Get Their Power (Laws)?</a>: 这项涵盖数学问题求解、证明助手编程和多模态 Jailbreaking 的最新研究记录了一个惊人的发现：当（多模态）语言模型处理一系列任务时...</li><li><a href="https://x.com/RylanSchaeffer/status/1908213817357803757">来自 Rylan Schaeffer (@RylanSchaeffer) 的推文</a>: 对 test time / inference scaling laws 感兴趣吗？那就来看看我们最新的预印本吧！！📉 How Do Large Language Monkeys Get Their Power (Laws)? 📉https://arxiv.org/abs/2502.17578 与 @JoshuaK92829 @sanmik...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1357435341828657213)** (9 messages🔥): 

> `Steering Vector Composition, Dynamic Activation Composition, Learned Steering Vectors, Function Vectors` 


- **Steering Vector Composition 效果良好**: 成员们去年研究了 **steering vector composition**，在处理一对无关属性（语言和形式度/安全性）时效果相当不错，如 [论文](https://aclanthology.org/2024.blackboxnlp-1.34/) 所示。
- **Dynamic Activation Composition 调节 Steering 强度**: 根据 [这篇论文](https://aclanthology.org/2024.blackboxnlp-1.34/)，**Dynamic Activation Composition** 是一种利用信息论方法在生成过程中调节一个或多个属性的 Steering 强度的方法。
- **预训练模型为 Steering Vectors 选取对比集**: 一位成员建议，让预训练模型从训练数据中挑选出 **contrastive sets**（对比集）来构建 Steering Vectors，然后控制 Steering Vectors 的系数，这可能会很有趣。
   - 不过，他们理想中希望能有一种更好的方式让模型构建 Steering Vectors，因为目前的方法感觉有些笨拙，特别是在需要跨 Mini batches 进行对比样本选择时。
- **Function Vectors 论文受到关注**: 一位成员重点介绍了 [David Bau 及其团队关于 'function vectors' 的论文](https://arxiv.org/abs/2310.15213)，该研究发现注意力头传输了所演示任务的紧凑表示。
   - 另一位成员提到，任何两个执行顺序会影响结果的任务，应该都无法同时通过 "function vectors" 或 "control vectors" 来表示。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.15213">Function Vectors in Large Language Models</a>: 我们报告了一种简单的神经机制，它在自回归 Transformer 语言模型 (LMs) 中将输入-输出函数表示为一个向量。通过对 d... 进行因果中介分析。</li><li><a href="https://aclanthology.org/2024.blackboxnlp-1.34/">Multi-property Steering of Large Language Models with Dynamic Activation Composition</a>: Daniel Scalena, Gabriele Sarti, Malvina Nissim. 第七届 BlackboxNLP 研讨会论文集：NLP 神经网络分析与解释。2024。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1357837368907923466)** (14 messages🔥): 

> `lm-eval-harness EOS token, Huggingface tokenization, encode_pair changes` 


- **lm-eval-harness 在 EOS Token 处理上遇到困难**：一名成员询问关于在 **lm-eval-harness** 的 **social_iqa 任务**中向数据实例添加 EOS token 的问题，并指出强制添加后准确率下降了 **18 个百分点**。
   - 有成员建议在[此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364)为多选题变体的 `continuation_enc` 添加 `self.eot_token_id`，并为 **BOS** 传递 `add_bos_token`。
- **Huggingface Tokenization 问题**：一名成员指出 **Huggingface** 模型的分词发生在 **HFLM.tok_encode** 中，但实现这一点后仍然导致准确率下降。
   - 他们指出，这些更改会使评估偏向于那些 EOS token 出现概率更高的选项。
- **警惕对 encode_pair 的重复调用**：其中一名成员提到，代码中 *encode_pair* 方法被调用了两次。
   - 这一观察意味着在 *encode_pair* 内部进行的任何修改都可能由于重复执行而产生意想不到的后果。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364)">lm-evaluation-harness/lm_eval/api/model.py at 11ac352d5f670fa14bbce00e423cff6ff63ff048 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357430894687948891)** (23 messages🔥): 

> `Chat reorganization, Lightweight model for price extraction, GPT4All's Quietness, Gemini 2.5 Pro for coding and math, Migrating data between SSDs` 


- **用户提议重组聊天列表**：一名用户建议聊天列表应根据最近修改的时间进行重组，而不是按创建时间的先后顺序排列。
   - 该用户认为按创建日期排序“有点随意”。
- **寻找用于价格提取的轻量级模型**：一名成员正在寻找一种非常轻量级的模型来从字符串中提取**价格**数值，因为由于用户输入多样，使用正则表达式（regex）进行常规解析并不可靠。
   - 建议的方案包括探索 **embedding 模型**或在 Hugging Face 上搜索名称中包含 *extraction* 的模型。
- **GPT4All 的静默：是闭门造车吗？**：一名成员询问为什么 **GPT4All** 最近如此安静。
   - 另一名成员声称 **GPT4All** “多年来一直不与普通用户交流，也不接受建议”。
- **Gemini 2.5 Pro：编程和数学高手的百万 Token 灵感？**：一名成员推荐了 **Gemini 2.5 Pro**，称其 **100 万 token 的超大上下文窗口**对编程和数学任务非常有益。
   - 他们指出该模型目前是**免费**的，**API** 也是如此。
- **GPT4All 方面的沉寂**：一名成员注意到了围绕 **GPT4All** 的沉默，并表达了对下一个版本发布以及 **Nomic Embed Text V2** 实现的期待。
   - 未提供更多细节。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357433397999304755)** (18 条消息🔥): 

> `Packed Datasets, Chunking Responsibility, NeMo's Resilient Training` 


- **Packed Datasets 提升速度并减少序列浪费！**：一位成员建议使用 **packed datasets** 以避免 `seqlen=49` 的 bug，并通过将句子打包直到达到 `max_seq_len` 来提高速度，从而避免浪费 padding tokens。
   - 要启用此功能，用户可以设置 `dataset.packed=True` 和 `tokenizer.mas_seq_len=<your-max_seq_len, 例如 8096>`，并利用 attention 的 **group masking**，详见 [PR #2560](https://github.com/pytorch/torchtune/pull/2560)。
- **Chunking 职责转移至 Loss Function！**：Chunking 的职责正通过 `loss = loss_fn(model.weight, logits, labels)` 转移到 loss function，以便于调试。
   - 创建了一个新文件 `torchtune.utils._tensor_utils.py`，其中包含对 `torch.split` 的封装并涵盖了单元测试，该文件需要被合并。
- **NeMo 应对崩溃和 GPU 浪费**：一位成员参加了 “Resilient Training with NeMo” 会议，并分享了关于 **NeMo** 如何解决任务崩溃和 GPU 时间浪费的见解，强调该主题与 torchtune 非常接近。
   - NeMo 的方法包括 **fault tolerance、straggler detection、asynchronous checkpointing、preemption、in-process restart、silent data corruption detection 和 local checkpointing** 等功能，但部分功能尚未实现。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/2560">fix: Timeout crash because of chunked_output len by bogdansalyp · Pull Request #2560 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）？请链接此 PR 解决的任何 issue - closes #25...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1357706121800060938)** (2 条消息): 

> `AI-2027 report, superhuman AI impact` 


- **AI-2027 报告发布**：一位成员分享了 [AI-2027 报告](https://ai-2027.com/) 的链接，该报告预测 **superhuman AI** 在未来十年的影响将是巨大的，超过**工业革命**。
   - 该报告基于趋势推演、兵棋推演、专家反馈、**OpenAI** 的经验以及之前的预测成功案例。
- **预测 Superhuman AI 的影响**：**OpenAI**、**Google DeepMind** 和 **Anthropic** 的 **CEO** 认为 AI 可能会在 2027 年之前超越人类智能。
   - 一位成员询问是否使用了 AI 来编写 [AI-2027 网站](https://ai-2027.com/) 上滚动实时更新的图表。



**提到的链接**：<a href="https://ai-2027.com/">AI 2027</a>：一份有研究支持的 AI 场景预测。

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357459430240944308)** (13 messages🔥): 

> `leetgpu tinygrad support, Huawei Ascend cards, WEBGPU BEAM limitations, maxComputeInvocationsPerWorkgroup issue` 


- **LeetGPU 关注 tinygrad 支持**：成员们讨论了 [leetgpu.com](https://leetgpu.com) 及其未来对 **tinygrad** 的潜在支持。
   - 未提供关于支持时间表或范围的具体细节。
- **向 tinygrad 开发者提供 Huawei Ascend 访问权限**：一名成员提出为开发目的提供 **Huawei Ascend** 卡的访问权限。
   - George Hotz 表示感兴趣，并询问了购买选项或云端机器的可用性。
- **WEBGPU BEAM 触及 maxComputeInvocationsPerWorkgroup 限制**：在为 **WEBGPU** 编译 `BEAM=2` 的 **tinygrad** 模型时，用户遇到需要将 `requiredLimits.maxComputeInvocationsPerWorkgroup` 增加到 **512** 的情况，这会降低对 Android 设备的支持。
   - 建议的 [PR](https://github.com/tinygrad/tinygrad/pull/9085) 涉及实现一种类似于现有全局维度控制的通用限制机制，一个 [热修复分支](https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py) 解决了该问题，并建议设置 `IGNORE_BEAM_CACHE=1`。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://leetgpu.com">LeetGPU</a>：未找到描述</li><li><a href="https://github.com/hooved/tinygrad/blob/hotfix-webgpu-workgroup/tinygrad/engine/search.py">tinygrad/tinygrad/engine/search.py at hotfix-webgpu-workgroup · hooved/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - hooved/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9085">Solve get_grouped_dims does not split issue by wpmed92 · Pull Request #9085 · tinygrad/tinygrad</a>：这关闭了 #8043。我们 lowerer 中的 _limit_dims 仅处理收缩（contraction），例如：dim=(2,3,4,5) max=(16,16,16)，即当 len(dim) > len(max) 时。但在 WebGPU 中，我们遇到了未被处理的情况……
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357432654886342857)** (4 messages): 

> `Distinguishable Instances, tinygrad Karpathy GPT Reimplementation, Metal Buffer Limit` 


- **正在调查可区分实例（Distinguishable Instances）**：一位用户询问是否可以使实例变得可区分，George Hotz 询问 *用途是什么？*
   - 未记录关于此内容的进一步讨论。
- **tinygrad KARPATHY GPT 得到重新实现**：George Hotz *刚刚开始接触 tinygrad*，并已在其中重新实现了 Karpathy GPT。
   - 未提供具体重新实现的链接。
- **Metal 面临 Buffer 限制错误**：一位用户报告在 **METAL** 上运行重新实现的 Karpathy GPT 时，由于 **32 buffer 限制** 出现了 `tinygrad.device.CompileError`。
   - 用户正在寻求指导，询问 "big graph" 的工作是否应该已经处理了这个问题，以及在哪里检查 early realization 问题，并附带了他们的 [main.py](https://cdn.discordapp.com/attachments/1070745817025106080/1357788499318800565/main.py) 链接。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1357750687965905067)** (1 messages): 

> `Multimodal Chat History, Multi-Agent Systems` 


- **LlamaIndex 支持多模态聊天历史**：LlamaIndex 现在支持 **多模态聊天历史**，使 Multi-Agent 系统能够处理交替的文本和图像消息，如 [这条推文](https://twitter.com/llama_index/status/1908191704156700682) 所述。
- **Agent 对图像和文本进行推理**：更新后的系统允许 Agent 利用 [ReAct agent 循环](https://t.co/EKIZiZJS2P) 对图像和文本进行推理。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357741396156481608)** (7 messages): 

> `PatentsView API, Workflow to Tool transformation` 


- **向 PatentsView 请求 API key**：一名成员向 **PatentsView 联系人** 发送了邮件，请求 **API key** 以收集初始数据并实现 **RAG**。
- **Workflow 转换为 Tool**：一名成员建议通过将 **Workflow** 放入 **FunctionTool** 来将其转换为 **Tool**。
   - 他们提供了一个代码示例，使用 `async def tool_fn(...)` 定义工具功能，然后使用 `FunctionTool.from_defaults(tool_fn)` 创建工具，从而实现对名称、描述、输入注解和返回值的控制。


  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1357618368731152426)** (4 messages): 

> `LlamaParse, LVM, image processing` 


- **LlamaParse 在图表理解方面存在困难**：一位成员询问如何让 **LlamaParse** 读取图表/图像，并指出目前即使在 **LVM** 和 Premium 模式下，它也只能提取文本而无法理解图像本身。
   - 另一位成员澄清说，如果图像缺乏可提取的文本，**LlamaParse** 将不会处理它，但它可以将图像作为 artifact/布局项提取出来，以便进一步处理，例如提示 **LLM** 对其进行描述。
- **图像提取**：**LlamaParse** 会将图像作为 artifact/布局项提取出来。
   - 这允许你进一步下载和处理（即如果你需要的话，可以提示 **LLM** 对其进行描述）。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357535404122705920)** (4 messages): 

> `AYA vision errors, AWS Bedrock` 


- **AYA vision 在处理 waves.jpg 时遇到障碍**：一位用户报告称，**AYA vision** 在分析 *waves.jpg* 图像时返回了 **400 error**，尽管 **AYA** 成功分析了其他 **JPG** 图像，但此处显示为“不支持的图像文件格式”。
   - 错误消息指出仅支持 **PNG, JPEG, WebP, and GIF** 格式，这表明特定的 **JPG** 文件或 **AYA** 的格式检测可能存在问题。
- **错误中引用了 AWS Bedrock**：一位用户提到在发生错误时看到了 *coco.py: AWS Bedrock Command A*，这可能暗示在上传图像时与 **AWS Bedrock** 存在关联。
   - 目前尚不清楚这是 **AYA** 流水线的一部分，还是用户在图像分析过程中遇到的无关错误。


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1357616125671768134)** (4 messages): 

> `Full-Stack Developer Introduction, Product Analyst Exploring AI Writing, Web3/AI Engineer Introduction` 


- **全栈高手宣布加入**：一位拥有 **8 年以上经验** 的全栈开发人员介绍了自己，强调了在 **React, Angular, Flutter, Swift, Python, TensorFlow, and OpenAI** 方面的专业知识。
   - 他们曾参与电子商务、医疗保健和金融科技领域的高影响力项目，集成了 **cloud technologies, microservices, and DevOps**。
- **产品分析师投身 AI 写作**：一位在求职空窗期的前产品分析师正在探索关于技术和 **AI** 的写作。
   - 他们正在寻找志同道合的人一起交流，讨论技术如何塑造我们的世界或 **AI** 的实际用途，感觉自己正处于“信息茧房”中。
- **Web3 奇才欢迎 AI 自动化**：一位在全栈/**AI** 开发方面拥有 **7 年以上经验** 的 **Web3/AI engineer** 介绍了自己。
   - 他们专注于将 **AI with automation** 相结合，并渴望以信心和创新帮助企业。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357846993761992804)** (1 messages): 

> `Asyncio Support for DSPy` 


- **Asyncio 集成受到关注**：一位成员询问了为通用 **DSPy** 调用添加 **asyncio** 支持的计划，并列举了从轻量级 **DSPy** 功能开始并随后扩展到优化的用例。
   - 目前，他们在需要 **DSPy** 功能之前一直使用 **litelm**，并对未来的支持表示好奇。
- **轻量级 DSPy vs LiteLLM**：讨论强调了一种模式：先从类似于使用 **LiteLLM** 的轻量级 **DSPy** 功能开始，然后随着项目的发展过渡到 **DSPy** 的优化能力。
   - 这表明在轻量级 **DSPy** 使用与完整的优化工作流之间可能需要无缝集成或功能对等。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1357462550974959806)** (1 messages): 

> `DeepSeek-V3 Upgrade` 


- **DeepSeek-V3 获得增强**：**DeepSeek-V3** 模型已升级为 **DeepSeek-V3-0324**，据报告在内部评估中表现稍好。更新公告已在 [X/Twitter](https://x.com/windsurf_ai/status/1907902846735102017) 上发布。
- **征求社区支持**：公告鼓励用户收藏该公告帖子，以便持续更新和支持。
   - 请求的措辞很幽默，承诺会以“爱”来回报那些收藏了 [X/Twitter post](https://x.com/windsurf_ai/status/1907902846735102017) 的用户。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1907902846735102017">Windsurf (@windsurf_ai) 的推文</a>：DeepSeek-V3 现已升级为 DeepSeek-V3-0324。它仍然是免费的！

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/)** (1 条消息): 

robotsail: 没问题！如果你有任何问题，或者需要我修改/重新测试任何内容，请随时告诉我。
  

---


---


---


{% else %}


> 完整的逐个频道详细分析已针对邮件进行了截断。 
> 
> 如果你想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}