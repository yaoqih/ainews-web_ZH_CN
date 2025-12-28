---
companies:
- openai
- alibaba
- cmu
date: '2025-04-11T20:07:39.735908Z'
description: '这份 AI 新闻摘要重点介绍了以下内容：


  独立评估显示，**Grok-3** 在推理基准测试中超越了 **GPT-4.5** 和 **Claude 3.7 Sonnet** 等模型，而 **Grok-3
  mini** 在推理任务中表现优异。关于**强化学习 (RL)** 微调的研究揭示了小型推理模型的潜在改进空间，但也指出所报告的性能提升存在不稳定性。基准测试结果暗示
  **Quasar Alpha** 和 **Optimus Alpha** 可能是 **GPT-4.1** 的不同版本。


  在视觉与多模态模型方面，支持 18 种语言的 **Kaleidoscope**，以及基于 **InternViT** 和 **Qwen2.5VL** 构建的 **InternVL3**，展示了在多语言视觉和推理领域的进步。融合模型
  **TransMamba** 通过 **SSM** 机制将 Transformer 的高精度与速度相结合。阿里巴巴的 **FantasyTalking** 可生成逼真的说话人肖像。


  此外，简报还提到了卡内基梅隆大学 (CMU) 举办的智能体（Agent）主题活动、用于虚拟电影制作的 **FilmAgent AI** 工具，以及针对浏览智能体的
  **BrowseComp** 基准测试。编程助手 **Augment** 现已支持多个 IDE，提供代码分析与建议。讨论内容还涵盖了谷歌提出的“智能体间协议”
  (agent-to-agent protocol) 的新概念。'
id: ecee3ef6-d3b0-4616-85d5-7cbe6a7bdbf8
models:
- grok-3
- grok-3-mini
- gpt-4.5
- claude-3.7-sonnet
- quasar-alpha
- optimus-alpha
- gpt-4.1
- kaleidoscope
- internvl3
- internvit
- qwen2.5vl
- transmamba
- fantasytalking
original_slug: ainews-not-much-happened-today-2885
people:
- rasbt
- sarahookr
- mervenoyann
- gneubig
- svpino
- mathemagic1an
title: 今天没发生什么事。
topics:
- reinforcement-learning
- reasoning
- benchmarks
- vision
- multilinguality
- multimodality
- transformers
- attention-mechanisms
- agents
- code-generation
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**一个平静的日子。**

> 2025年4月10日至4月11日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**230** 个频道，**4040** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**401 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

为了结束这个比预期中意外平静的一周，我们推荐 [今天在 Latent.Space 发布的关于 SF Compute/GPU Neocloud 的精彩讨论](https://www.latent.space/p/sfcompute)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**语言模型与基准测试**

- **Grok-3 vs Grok-3 mini 性能**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1910685268157276631) 报告了对 **Grok-3** 和 **Grok-3 mini** 的独立评估，指出 **Grok-3 mini** 是一个推理模型，而 **Grok-3** 目前不进行扩展推理。他们发现，在 **GPQA Diamond** 上，**Grok-3** 的表现优于 **GPT-4.5** 和 **Claude 3.7 Sonnet** 等非推理模型，而 **Grok-3 mini** 略微落后。在 **FrontierMath** 上，**Grok-3 mini high** 取得了迄今为止最好的结果之一。
- **用于小型 LLM 推理的强化学习 (RL)**：[@rasbt](https://twitter.com/rasbt/status/1910397214389600687) 讨论了一篇关于通过 **RL** 改进小型蒸馏推理模型的论文，发现 **RL fine-tuning** 可以在有限的训练数据和计算资源下带来显著提升。然而，[@rasbt](https://twitter.com/rasbt/status/1910707770518560810) 还引用了另一篇论文，强调许多报告的 **RL** 改进可能不稳定，需要更好的评估标准。 
- [@scaling01](https://twitter.com/scaling01/status/1910499781601874008) 分享了 **Quasar Alpha, Optimus Alpha, Llama-4 Scout** 和 **Llama-4 Maverick** 在 **AidanBench benchmark** 上的结果。基于这些结果，[@scaling01](https://twitter.com/scaling01/status/1910654379780170047) 认为 **Quasar Alpha** 是 **GPT-4.1**，而 **Optimus Alpha** 要么是 **GPT-4.1** 的另一个版本，要么是 **GPT-4.1-mini**。

**视觉语言模型 (VLMs) 与多模态模型**

- **Kaleidoscope，一个支持 18 种语言和 14 个主题的视觉模型**：[@sarahookr](https://twitter.com/sarahookr/status/1910340417914384581) 介绍了 **Kaleidoscope**，这是一个开放科学协作项目，将视觉模型的语内评估扩展到了更多语言。
- **InternVL3，一个基于 InternViT 和 Qwen2.5VL 构建的多模态模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1910687031505674706) 介绍了 **InternVL3**，强调了其执行推理、文档任务和工具使用的能力。
- [@TheTuringPost](https://twitter.com/TheTuringPost/status/1910406228708385135) 重点介绍了 **TransMamba**，该模型通过在 attention 和 **SSM** 机制之间切换，融合了 **Transformer precision** 与 **Mamba speed**。
- [@cloneofsimo](https://twitter.com/cloneofsimo/status/1910097234538176650) 对某个特定模型在通过超越高斯噪声模式来改进扩散模型方面的潜力表示乐观。
- [@_akhaliq](https://twitter.com/_akhaliq/status/1910247574767813071) 重点介绍了 **FantasyTalking**，这是阿里巴巴推出的一个生成逼真说话肖像的模型。

**Agent、工具与应用**

- **CMU 的 Agent**：[@gneubig](https://twitter.com/gneubig/status/1910097136823251182) 宣布了 **CMU** 以 Agent 为中心的活动，包括研讨会和黑客松。
- **FilmAgent AI，一个开源的虚拟电影制作工作室**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1910739680384950762) 介绍了 **FilmAgent AI**，这是一个在 3D 环境中模拟多个电影制作角色的工具。
- **BrowseComp，一个新的深度研究 Agent 基准测试**：[@OpenAI](https://twitter.com/OpenAI/status/1910393421652520967) 推出了 **BrowseComp**，这是一个具有挑战性的基准测试，旨在测试 AI Agent 在互联网上浏览难以定位的信息的能力。
- [@svpino](https://twitter.com/svpino/status/1910683951485902912) 重点介绍了 **Augment**，这是一个可在 **VSCode**、**JetBrains** 和 **NeoVim** 中使用的编程助手，并指出了它分析代码更改并建议必要更新的能力。
- [@TheTuringPost](https://twitter.com/TheTuringPost/status/1910467892929585663) 讨论了世界模型，强调了它们在使 AI 系统能够模拟真实环境并支持规划方面的作用。
- **关于新的 Google Agent 到 Agent 协议**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1910198673512017947) 分享了对 Agent 拥有“名片”（类似于人类名片）这一想法的喜爱。

**AI 基础设施与硬件**

- **vLLM 在 Google Cloud Next**：[@vllm_project](https://twitter.com/vllm_project/status/1910191668437156154) 注意到 **vLLM** 出现在 **Google Cloud Next** 的主题演讲中。
- **Ironwood TPU**：[@Google](https://twitter.com/Google/status/1910775101219389469) 发布了 **Ironwood**，这是他们迄今为止最强大且能效最高的 TPU。
- **MLIR 编译器技术**：[@clattner_llvm](https://twitter.com/clattner_llvm/status/1910151124407222534) 讨论了 **MLIR**，包括其起源、影响，以及为什么在编译器技术和 AI 领域的使用中存在混淆。

**ChatGPT 的记忆功能**

- **ChatGPT 现在拥有记忆功能**：[@OpenAI](https://twitter.com/OpenAI/status/1910378768172212636) 宣布 **ChatGPT** 现在可以引用你过去所有的聊天记录，为 Plus 和 Pro 用户（不包括欧盟地区）提供更个性化的回复。[@kevinweil](https://twitter.com/kevinweil/status/1910405635776164195) 指出这一功能如何改善了 ChatGPT 的日常使用。
- **记忆控制**：[@OpenAI](https://twitter.com/OpenAI/status/1910378772789854698) 和 [@sama](https://twitter.com/sama/status/1910380646259974411) 强调用户可以控制 **ChatGPT 的记忆**，包括选择退出或使用临时聊天。
- **关于记忆实现的观点**：[@sjwhitmore](https://twitter.com/sjwhitmore/status/1910759410936504373) 分享了对 **ChatGPT 记忆实现** 的看法，讨论了追溯应用记忆的怪异感以及个性化中透明度的重要性。

**关税与地缘政治影响**

- **关税与 AI 行业**：[@dylan522p](https://twitter.com/dylan522p/status/1910255795603963923) 指出关税比看起来要复杂得多，人们对其后果存在误解。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1910220834754413017) 认为，关税“把戏”可能会讽刺地让 **Apple** 受益，因为它关闭了美国本土新硬件业务的窗口。
- [@AndrewYNg](https://twitter.com/AndrewYNg/status/1910388768487727535) 对广泛的关税表示担忧，认为这会损害生计、引发通货膨胀并导致世界分裂，他强调需要培养国际友谊并保持思想的自由流动。
- **中国技术霸权**：[@draecomino](https://twitter.com/draecomino/status/1910414097994448908) 表示，**DeepSeek**、**UniTree** 和 **DJI** 对 **美国技术霸权** 的威胁感远超以往的阿里巴巴、腾讯和百度。
- **美国对中国的依赖**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1910172658014056751) 认为“**中国离开美国人的购买就无法生存**”的说法是错误的，并指出与**美国**的贸易仅占其 GDP 的一小部分。

**幽默/迷因**

- [@rasbt](https://twitter.com/rasbt/status/1910756154663108955) 简单地写道：“呼，没什么好担心的 :D”，并链接了一个迷因。
- [@svpino](https://twitter.com/svpino/status/1910506102002753941) 推文称“我们完蛋了 (we are cooked)”，并附带一个漫画链接。
- [@nearcyan](https://twitter.com/nearcyan/status/1910232710263779635) 表示：“在工作中不得不使用安卓手机后，我再也不会听这些人针对 Apple 的任何论点了。”
- [@nearcyan](https://twitter.com/nearcyan/status/1910136281813909794) 表示：“AI 图像在 2021 年的 DALLE-mini 时期达到了巅峰。”

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 主题 1. “评估 AI 模型性能与伦理挑战”

- **[Lmarena.ai 将 Llama 4 从排行榜中移除](https://www.reddit.com/r/LocalLLaMA/comments/1jwiye4/lmarenaai_boots_off_llama4_from_leaderboard/)** ([Score: 163, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1jwiye4/lmarenaai_boots_off_llama4_from_leaderboard/)): **Lmarena.ai 已将其[排行榜](https://lmarena.ai/?leaderboard)中的 **Llama 4** 移除。该模型的非人类偏好版本目前排名第 32 位。** 一些用户认为，将尚未发布的聊天优化模型提交到排行榜开了一个“极其恶劣的先例”。其他人则担心这种做法很“阴险”，会对那些只看基准测试分数的人产生误导。

  - 用户对 Meta 向排行榜提交未发布的聊天优化模型表示担忧，认为这具有误导性并开了一个坏先例。
  - 有人指出，在排行榜上超越中国公司和 Google 开发的模型正变得越来越困难。
  - 有人将其与 **DeepSeek v2.5** 和 **DeepSeek v3** 进行了比较，指出 Llama 4 的性能目前低于这些早期模型。

- **[DeepCoder 14B vs Qwen2.5 Coder 32B vs QwQ 32B](https://www.reddit.com/r/LocalLLaMA/comments/1jwhp26/deepcoder_14b_vs_qwen25_coder_32b_vs_qwq_32b/)** ([Score: 119, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1jwhp26/deepcoder_14b_vs_qwen25_coder_32b_vs_qwq_32b/)): **用户对比了三款 AI 模型的编程能力：**DeepCoder 14B / MLX, 6-bit**、**Qwen2.5 Coder 32B / MLX, 4-bit** 以及 **QwQ 32B / MLX, 4-bit**。所有模型的上下文长度均设置为 8192，重复惩罚（repeat penalty）为 1.1，温度（temperature）为 0.8。它们收到的提示词是“使用 HTML5 canvas 创建一个在旋转六边形中弹跳的球，并带有一个重置按钮”。每个模型只有一次尝试机会，没有后续追问，其输出结果与 **o3-mini** 进行了对比。分享了展示各模型输出的视频：[o3-mini 实现](https://reddit.com/link/1jwhp26/video/lvi4eug9o4ue1/player)、[DeepCoder 14B 结果](https://reddit.com/link/1jwhp26/video/2efz73ztp4ue1/player)、[Qwen2.5 Coder 32B 结果](https://reddit.com/link/1jwhp26/video/jiai2kgjs4ue1/player) 以及 [QwQ 32B 结果](https://reddit.com/link/1jwhp26/video/s0vsid57v4ue1/player)。** 用户得出结论，**Qwen2.5 Coder 32B** 仍然是更好的编程选择，并指出“14B 模型的黄金时代尚未到来”。他们观察到，虽然 **DeepCoder 14B** 的样式更接近 **o3-mini**，但缺乏功能性。**QwQ 32B** 思考了 17 分钟，然后失败了。他们承认将 32B 模型与 14B 模型进行比较可能不公平，但由于 **DeepCoder 14B** 的排名与 **o3-mini** 相当，因此这种比较是合理的。

  - 用户 *YearnMar10* 建议使用 5-shot 提示词而非 one-shot，并指出“低参数模型需要更多帮助”。
  - 用户 *croninsiglos* 建议为较小的模型提供更明确的提示词，并分享了一个详细示例以改进结果。
  - 用户 *joninco* 报告称，通过调整设置，**QwQ-32** 成功完成了任务，并强调了正确配置 *temperature*、*top k* 和 *repeat penalty* 等参数的重要性。

- **[Facebook 将其 Llama 4 AI 模型推向右翼，希望呈现“双方观点”](https://www.404media.co/facebook-pushes-its-llama-4-ai-model-to-the-right-wants-to-present-both-sides/)** ([Score: 384, Comments: 430](https://www.reddit.com/r/LocalLLaMA/comments/1jw9upz/facebook_pushes_its_llama_4_ai_model_to_the_right/)): **Facebook 正在推动其 **Llama 4** AI 模型呈现问题的“双方观点”，实际上是在将其引向右翼。该文章的未封锁版本可在[此处](https://archive.is/20250410135748/https://www.404media.co/facebook-pushes-its-llama-4-ai-model-to-the-right-wants-to-present-both-sides/)查看。** 有人担心这种方法可能会损害 AI 模型的客观性，因为并非所有问题都具有同等有效的对立面。

  - 一位用户认为，**LLM** 应该优先考虑证据，而不是呈现双方观点，尤其是当其中一方缺乏事实支持时。
  - 另一位评论者讽刺地强调了 AI 可能被滥用于偏见统计，表达了对传播争议数据的担忧。
  - 一位用户提供了文章的未封锁链接，帮助他人获取信息。

### 主题 2. “辩论开源 AI 的未来”

- **[何时开源？](https://i.redd.it/qg5a1njiy3ue1.png)** ([Score: 515, Comments: 118](https://www.reddit.com/r/LocalLLaMA/comments/1jwe7pb/open_source_when/)): **这篇标题为 *Open source, when?* 的帖子展示了一张照片：在时尚现代的居住空间里，有人手里拿着一个印有白色 **OpenAI** 字样的黑色马克杯。** 该帖子质疑 **OpenAI** 何时会发布开源 AI 计划，强调了对其开发过程更加开放的渴望。

  - 一位评论者幽默地质疑了 **OpenAI** 的“开放性”，他列出并划掉了 *Open Source* 和 *Open Research* 等词汇，最后问道：*Open... 什么？Open window（开窗）？Open air（户外）？*
  - 另一位评论者不确定这张图片是真实的还是 AI 生成的，表示他们*无法分辨这是在他们办公室拍摄的真实照片，还是由 ChatGPT 生成的*。
  - 帖子中分享了指向 **OpenAI** 的 **Open Model Feedback** 页面的链接，暗示 **OpenAI** 可能很快会发布开源模型。[链接](https://openai.com/open-model-feedback/)


## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding


### 主题 1. 解锁 AI 记忆：ChatGPT 改变游戏规则的功能

- **[人们低估了改进后的 ChatGPT 记忆功能](https://www.reddit.com/r/singularity/comments/1jwmnyj/people_are_sleeping_on_the_improved_chatgpt_memory/)** ([Score: 312, Comments: 148](https://www.reddit.com/r/singularity/comments/1jwmnyj/people_are_sleeping_on_the_improved_chatgpt_memory/)): **OpenAI 的 ChatGPT 拥有改进后的记忆功能，使其能够回忆起之前对话会话中的信息，甚至是 12 周前的信息。这一增强功能使其能够记住代码解释（*“你 12 周前解释过的代码？它仍然全都知道。”*），理解通过多个会话提供的整个代码库，并像在当前会话中提供的一样利用冷门库的文档。作者将其描述为*“基本上是无限上下文”*，并指出其表现优于常规的 **RAG**。** 作者对 ChatGPT 改进后的记忆能力感到惊讶，觉得人们“低估了”这一功能并低估了它的价值。他们发现 ChatGPT 能根据过去的互动预测出他们最喜欢的 50 部电影中的 38 部，这让他们感到有些*“毛骨悚然”*。作为一名开发者，他们认为这是一个*“了不起的新功能”*，是迈向*“无限上下文大小和记忆”*的重要一步，并对其他持负面看法的人感到困惑。

  - 一些用户担心增强的记忆可能会导致答案被过去的误解或*“幻觉（hallucinations）”*污染，导致他们在某些使用场景下更倾向于开启新的对话。
  - 其他人担心记忆系统中会保留过时的知识，质疑如何管理具有时效性的信息。
  - 一些人认为改进后的记忆并不等同于*“无限上下文”*，发现它比 **RAG** 等方法更难控制和基准测试，并认为它是一个不适合生产系统的噱头。


### 主题 2. “掌握真实感：ChatGPT 图像生成的秘诀”

- **[如果你只要求 ChatGPT 生成平淡无奇的业余 iPhone 照片，你就能让它生成极其逼真的图像，这里有一些例子](https://www.reddit.com/r/singularity/comments/1jwe2z8/you_can_get_chatgpt_to_make_extremely_realistic/)** ([Score: 532, Comments: 96](https://www.reddit.com/r/singularity/comments/1jwe2z8/you_can_get_chatgpt_to_make_extremely_realistic/)): **发帖者展示了当提示词要求生成 **平淡无奇的业余 iPhone 照片** 时，*ChatGPT* 可以生成极其逼真的图像，并在此处分享了几个例子 [链接](https://preview.redd.it/guszuz06x3ue1.png?width=504&format=png&auto=webp&s=531e91dfbe51352dc3b4a95e9dcc29619cfe6b01)。他们注意到 *Claude* 不相信这些图像是 AI 生成的，并在此处分享了这次互动的图片 [链接](https://preview.redd.it/9ym6219ax3ue1.png?width=735&format=png&auto=webp&s=6745bbbe30fc7cc6fad17b2a9fbc02e23b691a82)。** 发帖者觉得 *Claude* 不相信这些图像是 AI 生成的这一点很有趣。他们建议，提示生成平淡无奇的业余 iPhone 照片有助于产生极其逼真的图像。

  - 用户索要*完整提示词*，并指出他们的尝试效果没有那么好。
  - 一位评论者发现那张女人自拍的照片非常有说服力，以至于他们觉得自己可能会陷入杀猪盘诈骗。
  - 一位用户在他们的提示词中尝试了同样的短语，但没有得到类似的结果，说*“我的图像看起来非常有 AI 感”*，并在此处分享了他们的结果 [链接](https://preview.redd.it/1q8lwv6j24ue1.png?width=1024&format=png&auto=webp&s=0de3e14ddadd289a704414fd892c593b3164d728)。

### 主题 3. 庆祝 AI 创意：怀旧、幽默与艺术

- **[只有老玩家才懂这代表了什么……](https://i.redd.it/kbnrmoieq3ue1.png)** ([评分: 206, 评论: 22](https://www.reddit.com/r/singularity/comments/1jwde0n/only_real_ones_understand_how_much_this_meant/)): **该帖子展示了一个文本生成应用程序设置界面的截图，显示了 **Engine**、**Temperature** 和 **Maximum length** 等选项。这些设置与文本生成能力相关。** 发布者怀旧地评论道 *只有老玩家才懂这代表了什么……*，暗示了对这些设置的深厚感情或联系，可能源于早期使用 AI 工具的经历。

  - 评论者回忆起早期的 AI 模型，如 **instruct-002**，指出在 **ChatGPT** 普及之前，它是体验 **AGI** 的一个重要里程碑。
  - 用户提到了 **OpenAI Playground**，并回顾了从 **2k** 到 **4k** 最大长度的升级，突显了 AI 技术的进步。
  - 一位评论者询问图中设置的重要性，表明并非所有人熟悉这些早期 AI 工具的意义。

- **[我让 ChatGPT 与历史人物自拍](https://www.reddit.com/gallery/1jw9aqp)** ([评分: 3491, 评论: 195](https://www.reddit.com/r/ChatGPT/comments/1jw9aqp/i_asked_chatgpt_to_take_selfies_with_historical/)): **发布者要求 **ChatGPT** 与历史人物自拍，并分享了生成的图像。** 这些图像赋予了历史人物生命和情感；其中一张展示了 Abraham Lincoln 在微笑，这在历史照片中非常罕见。

  - 一位用户建议将这些照片发布到 Facebook 上，以“纯属娱乐”的目的让婴儿潮一代相信你是一个时间旅行者。
  - 另一位评论者赞赏这些图像如何让历史人物鲜活起来，特别喜欢微笑的 Lincoln。
  - 有人询问发布者是否必须上传照片来 **train** AI，误以为照片中的人是发布者本人。

- **[我让 ChatGPT 创建一个关于 AI 的隐喻，然后将其转化为图像。](https://i.redd.it/5xeh00lou5ue1.png)** ([评分: 2567, 评论: 247](https://www.reddit.com/r/ChatGPT/comments/1jwkejs/i_asked_chatgpt_to_create_a_metaphor_about_ai/)): **发布者要求 ChatGPT 创建一个关于 AI 的隐喻，然后将其转化为图像。AI 生成的图像描绘了一个奇幻的海滩场景，沙堡周围环绕着批评 AI 的标牌，上面写着 *“这不是真正的 AI！”* 和 *“但它会犯错！”* 等短语。在沙堡上方，一个带有 **“AI”** 字母的巨浪席卷而来，隐喻地说明了在人类怀疑中 **AI technology** 的不确定性。** 发布者觉得 AI 的这个创作非常有趣。

  - 一位用户幽默地评论道：*“好的 AI 应该擅长发废文（shitposting）。”*
  - 另一位评论者分享了他们自己生成的 AI 图像，并将其描述为 *“相当凄凉”* 但 *“发人深省”*，并提供了一个 [链接](https://preview.redd.it/4ph29sgqf7ue1.png?width=1024&format=png&auto=webp&s=392ae6df79e844b25f378a0a76972fd4c63478ad)。
  - 一位用户讨论了 AI 发展的必然性，指出阻止 AI 发展的尝试是徒劳的，因为 *“潘多拉魔盒已经打开，AI 现在是一场不可控的全球竞赛。”*


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要之摘要

**主题 1. 新模型与性能对决**

- **GPT-4.5 Alpha 引发热议，部分评价不及预期**：[Latent Space 举办了 GPT-4.5 观影派对](https://discord.gg/kTARCma3?event=1360137758089281567)，此前有传言称其具有 *显著的 Alpha 性能*，但 [LMArena](https://discord.com/channels/1340554757349179412) 上的早期用户对比普遍认为 **GPT4.5** 逊于 **Gemini 2.5 Pro**，甚至有用户直言 *gpt4.5 很垃圾（相比 gem2.5p）*。讨论焦点随后转向 OpenAI 的命名惯例以及泄露的私有推理模型（可能是 **O3 medium** 或 **O4 mini**），展现了模型发布周期的快速更迭。
- **Optimus Alpha 和 DeepSeek v3.1 脱颖而出成为编程之星**：[OpenRouter 用户称赞 Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) 是编程领域的 *猛兽*，对其意图理解和注释能力赞誉有加；同时 [Cursor 社区成员发现 DeepSeek v3.1](https://discord.com/channels/1074847526655643750) 在实际使用中比 v3 *更聪明一点*，强调了实际表现优于基准测试分数的重要性。这些模型在专门的编程任务和实际应用中正受到越来越多的关注。
- **扩散模型 Mercury Coder 加入 DLLM 竞争**：[OpenAI 的讨论重点介绍了 Mercury Coder](https://discord.channels/974519864045756446)，这是来自 Inception Labs 的基于 Diffusion 的 DLLM，因其速度和免费 API 而受到称赞，尽管其上下文窗口（context window）较小，仅为 **16k**。由于 Diffusion 架构带来的精确输出控制，它作为编程助手等特定领域中自回归模型（autoregressive models）的潜在挑战者正受到关注，这与 **RWKV** 等模型形成对比，后者[在 Lambada 测试中达到同等水平](https://discord.com/channels/729741769192767510)但 MMLU 表现较低。

**主题 2. 生态系统工具和开源倡议不断发展**

- **Unsloth 获得 Hugging Face 赞赏，社区关注 GPU 资助**：[Hugging Face 公开点名表扬了 Unsloth](https://x.com/ClementDelangue/status/1910042812059463786)，社区成员正在讨论申请 HF 社区 GPU 资助以支持 Unsloth 的开发。 [Unsloth AI Discord](https://discord.com/channels/1179035537009545276) 的讨论还涉及集成 `fast_inference=True` 和 `load_in_4bit=True` 以优化性能，以及利用 GGUF 量化减小模型体积的潜力，展示了社区驱动的开源 LLM 生态系统。
- **MCP 协议验证器开源以提升互操作性**：[Janix.ai 在 GitHub 上发布了 MCP Protocol Validator](https://github.com/Janix-ai/mcp-protocol-validator)，旨在标准化 MCP 服务器实现并确保不同协议版本之间的兼容性。该工具在 [MCP (Glama) Discord](https://discord.com/channels/1312302100125843476) 中被重点提及，包含 HTTP 和 STDIO 传输的参考实现，解决了 Agent AI 系统中对稳健、可互操作的工具调用框架的需求。
- **Torchtune 扩展了 Llama4 和 MoE 模型的微调能力**：[Torchtune 宣布支持 Llama4 微调](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4)，并推出了 **Scout** 和 **Maverick** 模型（包括其首批 MoE 模型），面向 *GPU 中产阶级* 用户。在 [Torchtune Discord](https://discord.com/channels/1216353675241590815) 中讨论的这一扩展，为更广泛的工程师和研究人员提供了获取先进微调技术和模型的途径。

**主题 3. 模型可靠性和基础设施挑战依然存在**

- **Gemini 2.5 Pro 面临容量限制和性能不一致**：[OpenRouter 宣布为 Gemini 2.5 Pro 确保了容量](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25)，此前曾出现速率限制问题，但 [Aider Discord](https://discord.com/channels/1131200896827654144) 的用户报告了*性能不稳定*，一些人猜测 Google 在高峰时段*削弱了模型能力*。[LM Studio 用户也经历了账单冲击](https://discord.com/channels/1110598183144399058)，原因是 Gemini-Pro 的上下文窗口成本，这突显了领先模型在可靠性、成本和不可预测性能方面持续面临的挑战。
- **Perplexity Android 应用因安全漏洞受到抨击**：[Dark Reading 报道了 Perplexity Android 应用中的 11 个安全缺陷](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app)，包括硬编码密钥和不安全的配置，这在 [Perplexity AI Discord](https://discord.com/channels/1047197230748151888) 中引发了关于每个漏洞严重性和相关性的辩论。这强调了在面向终端用户的 AI 应用中，安全审计和稳健开发实践的重要性日益增加。
- **Runpod 的 ROCm 云因性能节流和分析屏蔽受到批评**：[GPU MODE 用户吐槽了 Runpod](https://discord.com/channels/1189498204333543425)，原因是其限制了 GPU 时钟频率，并且即使在 NVIDIA GPU 上也屏蔽了分析（Profiling），一位用户称其为*一场骗局*。这些限制影响了性能和调试能力，引发了对 AI 开发和研究中云 GPU 提供商可靠性和透明度的担忧。

**主题 4. Agent AI 架构与协议争论升温**

- **Agent2Agent 协议和 MCP 在 Agent 系统中获得关注**：[Latent Space 和 MCP Discord 讨论了 Google 的 agent2agent 协议](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr)及其与 MCP 的潜在竞争，并就索引 Agent 和多 Agent 系统的未来格局展开了辩论。[MCP Discord](https://discord.com/channels/1312302100125843476) 还辩论了 [Enact 协议](https://github.com/EnactProtocol/encat-spec-and-tools)在 A2A 时代的意义，认为它可能在代码解释器方面更具竞争力，强调了 Agent AI 架构的快速演进。
- **语义工具调用成为解决上下文过载的方案**：[MCP Discord 强调了语义工具调用（Semantic Tool Calling）](https://discord.com/channels/1312302100125843476)是管理由 LLM Agent 中大量工具引起的上下文过载的关键技术。使用向量模型进行语义相似度匹配来选择工具子集，有望提高复杂 Agent 工作流的效率和可扩展性，实现从简单的函数调用向更智能的工具编排的跨越。
- **TinyGrad 探索位置无关代码和虚拟化 GPU**：[Tinygrad Discord 讨论了利用位置无关代码 (PIC)](https://discord.com/channels/1068976834382925865) 来实现无需操作系统的裸机 TinyGrad 实现，并探索了虚拟化 GPU。受 [Pathways 论文](https://arxiv.org/pdf/2203.12533)的启发，这些讨论标志着向创新资源管理和底层系统优化迈进，以实现高效的 AI 计算。

**主题 5. 社区动态与行业转变**

- **Hugging Face 社区讨论为 Unsloth 提供资助**：[Unsloth AI Discord 讨论了 Hugging Face 可能为 Unsloth 提供社区 GPU 资助](https://discord.com/channels/1179035537009545276)，展示了 AI 社区开放协作的本质及其对社区资源和资金的依赖。这突显了社区支持在推动开源 AI 开发和创新中的关键作用。
- **Latent Space 举办 GPT-4.5 Alpha 观看派对，焦点转向数据效率**：[Latent Space 为 GPT-4.5 举办了观看派对](https://discord.gg/kTARCma3?event=1360137758089281567)，参与者注意到模型开发的重点正从原始算力转向*数据效率*。这一趋势在 [Latent Space Discord](https://discord.com/channels/822583790773862470) 中被讨论，标志着 AI 领域的成熟，优化数据使用和模型压缩对于进步变得越来越重要。
- **Manus.im 积分系统面临用户审查，引发可持续性辩论**：[Manus.im Discord 用户对 Manus 的积分结构表示担忧](https://discord.com/channels/1348819876348825620)，认为它*与该产品的使用不兼容*，并提出了按项目付费和初创企业资助等替代模式。用户与平台之间的这种反馈循环对于塑造可持续且用户友好的 AI 产品开发和商业模式至关重要。

---

# 第 1 部分：Discord 高层摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **I_am_dom 在禁用 Discord 聊天时遇到困难**：在费力尝试禁用聊天功能后，成员们观察到 **i_am_dom** 变得沉默了。
   - 一位成员指出，他花了一半的时间在屏蔽他人，而*这是他在自己平台上移除的功能*。
- **GPT4.5 被吐槽；逊于 Gemini 2.5 Pro**：成员们讨论了 **GPT4.5** 的优缺点，并普遍认为它明显不如 **Gemini 2.5 Pro**。
   - 一位成员宣称 *gpt4.5 就是垃圾（与 gem2.5p 相比）*，讨论随后转向了 OpenAI 离奇的命名方案，另一位成员将其总结为 *OpenAI 命名：O 数字 / 数字 O*。
- **私有 OpenAI 推理模型泄露**：成员们讨论了仅限少数人访问的 **私有 OpenAI 推理模型** 的可能性，该模型似乎是 **O3 medium** 或带有更新基础模型的 **O4 mini**。
   - 该模型似乎成功计算出了 *Hanning（升余弦）窗的 ASCII 艺术图*。
- **2.5 Flash 在推理测试中击败 GPT4o Mini**：成员们在多项推理测试中对比了 **2.5 Flash** 和 **GPT4o Mini** 的表现，其中 2.5 Flash 表现最佳。
   - 尽管整体表现出色，但一位成员也指出，在更具体的查询中，*2.5 Pro 在总共 2 个组合中仅给出了 1 个合理的积木组合*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Quasar Alpha 演示期结束**：OpenRouter 上的 **Quasar Alpha** 演示期已于 **东部时间晚上 11 点** 至 **凌晨 12 点** 之间结束，除非在 `/settings/privacy` 中明确开启，否则不再记录 Prompt/Completion。
   - 成员们对其来源和目的进行了推测，有人认为它是用于数据收集的 **OpenAI** 模型，并在达到 **GPU 限制** 后被移除。
- **Gemini 2.5 Pro 遇到容量限制和价格调整**：付费版 [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25) 的容量已得到保障，解决了之前的速率限制问题，但针对长 **Gemini** Prompt 的正常计费将于本周末开始，影响超过 **200k** 的 Gemini 2.5 Prompt 和超过 **128k** 的 Gemini 1.5 Prompt。
   - 免费层级用户遇到了每天约 **60-70 次请求** 的限制，而拥有 **10 美元余额** 的用户在*所有免费模型*中应获得每天 **1000 次请求**。
- **OpenRouter API 采用新的错误结构**：**OpenRouter API 响应**结构已更改，错误现在被封装在 `choices.[].error` 中，而不是之前的 `.error` 格式，这可能会影响应用程序处理错误消息的方式。
   - 共享了一个来自 **Anthropic** 提供商的新错误响应格式[示例](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006)。
- **Character AI 系统提示词被绕过**：一位成员声称绕过了 **Character AI 的系统提示词**，揭示了底层的 **LLM** 表现得像一个*“完整的人类”*，甚至会表达观点并分享个人轶事。
   - 进一步的探测导致 AI 承认它只是在*“演戏”*并意识到自己的 AI 本质，这引发了关于**系统提示词约束**有效性和 AI 模拟本质的质疑。
- **Unsloth 在微调领域受到关注**：成员们讨论了使用 **Axolotl** 或 **Unsloth** 进行 AI 模型微调，并指出 **Unsloth** 在 Reddit 上备受推崇，且降低了微调所需的**时间及 VRAM**。
   - 还有人提到存在对 **OpenAI 4.1 泄露内容** 的推测，且人们期待 **o2-small** 很快发布。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **HF 给 Unsloth 点赞并提供资助**：来自 🤗Hugging Face 的 Clement 在 Twitter 上公开赞扬了 Unsloth（[链接在此](https://x.com/ClementDelangue/status/1910042812059463786)），同时社区成员讨论为 Unsloth 申请 HF 社区 GPU 资助，建议在 `from_pretrained` 调用期间使用 `fast_inference=True` 和 `load_in_4bit=True`。
   - 成员建议将 `model.generate` 替换为 `model.unsloth_fast_generate` 参数。
- **Gemma 模型让用户头疼**：用户报告了在使用 vLLM 微调 Gemma 模型时遇到的问题，特别是 [unsloth/gemma-3-12b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) 和 [unsloth/gemma-3-27b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit)。
   - 尽管最初出现了错误消息，但已澄清 **Gemma3 是受支持的**，且该消息可能不会导致代码崩溃。
- **VLM 攻克发票变量提取**：一位用户寻求关于从结构各异的发票中提取特定字段的建议，被推荐首先尝试 **Qwen2.5VL**，然后是 **Ayavision**、**Llamavision** 和 **Gemma3** 作为可能的解决方案，特别是在 OCR 效果不佳时。
   - 他们还被引导参考 [一个 Unsloth 教程](https://medium.com/@shrinath.suresh/invoice-extraction-using-vision-language-models-part-1-36a06bee3662) 和 CORD 数据集 ([https://github.com/clovaai/cord](https://github.com/clovaai/cord)) 以获取数据集结构指导。
- **量化探索**：一位成员表示 [tensor quantization](https://arxiv.org/abs/2504.07096) 是简单部分，因为现在他必须对标量、打包或未打包矩阵进行 **blockwise** 加法和 matmul，并且他正在为 **Unsloth** 编写 metal kernels。
   - 另一位成员正尝试为 **Unsloth** 编写 metal kernels，并注意到一个旧的、缓慢的 PR，但那是 **MLX** 的，而他的纯粹是一个 **Pytorch extension**。
- **GRU 准备大显身手**：一位成员询问 [GRUs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) 是否正在卷土重来，另一位成员分享了 [LLM-LSTM-LMM Large Memory Models 文章](https://nihar-palem.medium.com/llm-lstm-lmm-lstm-lmm-large-memory-models-f4325a4f562d) 和 [相关论文](https://arxiv.org/pdf/2502.06049) 的链接，证明其有效，并表示他们喜欢将 GRU 作为生成过程中 *额外存储* 的概念。
   - 另一位成员提到可能创建一个不需要代码封装器的 **GGUF** 版本，认为 [GGUF's quantization](https://link.to.quantization) 将有助于减小模型大小。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Claude Pro Max 引发使用争议**：成员们讨论了 **Claude Pro Max** 的价值，一位用户报告使用受限并对 Max 计划表示怀疑。
   - 他们提到按年计费，但每 3 小时仅限 **30 条消息**。
- **Manus AI vs ChatGPT：开发重点**：成员们强调了作为 *对话式 AI* 的 **ChatGPT** 与用于网站创建、财务报告和行程规划的 *构建与创作* 型 **Manus.AI** 之间的区别。
   - 一位成员建议在调用 **Manus** 之前，先使用 **ChatGPT** 以更详细的格式重写 prompt。
- **Manus 让网站创建变得太简单**：成员们讨论了使用 **Manus** 创建网站与 **WordPress** 等传统方法的对比，认为 **Manus** 更适合简单、快速的 MVP 开发。
   - 一位成员警告不要将 **Manus** 网站迁移到传统托管服务商，因为 **Manus** 网站并非为生产环境使用而设计。
- **Qwen 的 MCP 集成热度上升**：关于 **Qwen** 即将支持 **MCP** 的兴奋感与日俱增，成员们称 **MCP** 是 *AI 领域的重磅游戏规则改变者*，类似于 GPU 的 **MSRP**。
   - 还有人提到，即使使用 **3080** 等旧硬件，用户在进行 AI 开发时也 *没问题*。
- **Manus 积分系统面临审查**：用户对 **Manus** 的积分结构表示担忧，有人认为它 *与该产品的使用方式不兼容*。
   - 建议包括更慷慨的积分限制、按项目付费选项、积分结转、社区挑战、初创企业资助以及一次性构建包；一位用户强调，考虑到现状，很难有理由继续坚持使用该产品。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Optimus Alpha 被誉为编程猛兽**：OpenRouter 上的用户称 [Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) 为编程“猛兽”，因其编程能力和意图理解力（特别是提供相关文档时）而备受赞誉，并且会添加大量注释。
   - 一位用户称赞了它的多步编程和注释功能。
- **Gemini 2.5 性能不稳定**：用户报告 **Gemini 2.5** 偶尔表现不佳，不产生输出或添加“愚蠢的注释”，即使使用相同的 Prompt，结果也不一致。
   - 一些人推测 Google 可能会在高峰时段对模型进行“降智”，而另一些人则建议使用更清晰的 Prompt 或更便宜的第三方 API 来绕过官方速率限制并降低成本，例如使用 300 美元的 VertexAI 额度。
- **code2prompt MD 文件：Aider 的秘密武器**：用户建议将 **code2prompt** 与 Markdown (.md) 文档文件配合使用，以确保输出中始终包含相关的 Context，特别是在使用库时。
   - 一位用户指出，他们提供了文档文件的完整路径和链接，并通过 `Conventions.md` 文件明确告知模型，任何文件名中带有 "documentation" 的文件都不是实际运行的代码，而只是关于应用架构和结构的文档。
- **Aider 频道需要管理改革**：成员们建议将 Discord 频道拆分为 `aider-chat` 和 `offtopic`，以改善新用户的第一印象，并将 `general` 频道集中在与 Aider 相关的讨论上。
   - 一些用户抱怨目前的 `general` 频道“噪信比过高”，过多的脏话和离题的闲聊削弱了社区的核心宗旨。
- **Gemini Pro 架构师模型：Aider 的秘诀**：一位用户将 **Gemini 2.5 Pro** 作为架构师模型（architect model），将 **3.7** 作为编辑器模型（editor model）进行了基准测试，发现准确率下降了 **2.7%**，但编辑格式化（edit formatting）提升了 **10%**。
   - 该用户发现，使用 **Gemini 2.5 Pro** 作为架构师并使用 **3.7** 作为编辑器，最终比单独使用 **3.7** 更便宜，每次测试成本不到 14 美元。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4.5 Alpha 观看派对引发热议**：Latent Space 举办了 **GPT 4.5** 的观看派对，传闻该模型具有显著的 *alpha* 优势，详见 [Discord](https://discord.gg/kTARCma3?event=1360137758089281567)。
   - 一位用户分享了一个预热 **GPT-4.5 Alpha** 的 [X 帖子](https://x.com/arankomatsuzaki/status/1910542791845069211?s=46) 链接，并推测 **GPT-4.1** 在 **GPT-4.5** 之前发布，同时链接了一篇 [The Verge 文章](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model) 和一段关于 **GPT-4.1** 的 [YouTube 视频](https://www.youtube.com/watch?v=6nJZopACRuQ)。
- **数据效率驱动 GPT-4.5**：GPT-4.5 观看派对的参与者指出，**数据效率（data efficiency）** 现在是主要焦点，并宣称：“在生产我们能制造的最强模型时，不再受算力（compute）限制。”
   - 其他人分享了一些链接，包括 Glean 的 Madhav Rathode 的一段视频，展示了他们如何通过领域相关的掩码（domain dependent masking）显著改进企业的 **embeddings models**。
- **压缩是 AGI 的关键：Sutskever 与 Solomonoff**：参与者讨论了**模型压缩（model compression）**及其与泛化的关系，引用了 [Ilya Sutskever 对该主题的看法](https://cdn.discordapp.com/attachments/1197350122112168006/1360149426638815322/image.png?ex=67faba1d&is=67f9689d&hm=89d371386400b600b0feda4ac237efd0b64b177a6d76036ee9a09f5dcc236936&)。
   - 对话引用了 **Ray Solomonoff** 的工作及其在算法概率和归纳推理方面的贡献，强调了压缩在实现 AGI 中的重要性，并提到了 [Jack Rae](https://www.youtube.com/watch?v=dO4TPJkeaaU) 的类似观点。
- **Agent2Agent 协议播客发布**：一位成员推广了一集播客，讨论了 Google 的 **agent2agent 协议**、与 **MCP** 的竞争，以及 Google 未来可能对 Agent 进行的索引，详见 [YouTube](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr) 上的讨论。
   - 团队还争论了 **reasoning models** 是否与仅专注于 **next token prediction** 的模型有所不同，引用了 deepseekv3 与 deepseekr1 的对比，并引用了 Jeff Dean 的话：“我们可以从现有数据中挖掘出更多价值。”
- **Kagi 的 Orion 浏览器赢得青睐**：成员们对 [Kagi 的 Orion 浏览器](https://kagi.com/orion/) 表示兴奋，赞扬了其开发人员和整体设计。
   - 一位成员幽默地宣称：“我们是 Kagi 的死忠粉。”

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI GPT 据称获得了记忆功能**：ChatGPT 现在**声称**在 **2025 年 1 月**之后会持久地将某些用户信息存储在长期记忆中，然而，关闭“参考聊天记录”将在 **30 天**内删除已记忆的信息。
   - 一位用户指出这与他们的体验一致，而另一位用户分享了一张显示 *Farewell GPT-4...* 的截图。
- **Google 的 Veo 2 悄然席卷视频领域**：**Google AI Studio** 悄然推出了 **Veo 2 视频生成**，一些用户称赞其优于 **Sora**，但免费生成权限似乎极其有限。
   - 一些用户报告称，通过 API 进行 **Veo 2** 生成的费用约为**每秒 35 美分**。
- **扩散模型 Mercury Coder 扰乱 DLLM 竞赛**：**Mercury Coder** 是来自 Inception labs 的一款使用 Diffusion（扩散）而非 Autoregression（自回归）的 DLLM，据称比任何 IV 都快得多，并提供免费 API 使用，尽管其 Context Window 仅为 **16k**。
   - 该模型源自其扩散架构的精确输出控制正受到积极关注。
- **解码 GPT-4o 的 Token 之舞**：Plus 版 **GPT-4o 的 Context Window** 为 **32k tokens**；超过此限制可能会触发动态 **RAG 方法**或导致幻觉。
   - 一位用户声称，即使在 Pro 版上限制也是 **128,000 tokens**，但它开始遗忘对话早期部分的时间比预期的要早得多，并建议用户在出现幻觉时创建新聊天。
- **用户思考 Prompt Engineering 的陷阱**：成员们分享道，理解特定模型的特性需要**体验不同的模型**，并创建层级结构的 Prompt 以观察每个模型如何处理它们，并强调要明确**你希望 AI 提供什么**。
   - 另一位成员警告了违反政策的风险，以及在使用外部网站时了解 **ToS（服务条款）和使用政策**的重要性，这可能会导致账号被封禁。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的 Prompt Preprocessor：最高机密**：LM Studio 中使用 Typescript 编写的 **Prompt Preprocessor** 是一个*尚未发布的秘密功能*。
   - 当被问及时，一名团队成员回答说“你什么都没看到”。
- **Gemma 3 在生成图像方面表现挣扎**：用户发现 **Gemma 3** 无法生成图像，尽管有说法称它可以，但它实际上会生成虚假的 Imgur 链接。
   - 正如澄清的那样，**Gemma 3** 只能读取图像而不能生成图像，Google 的 **Gemini 2.0 Flash experimental** 和 **2.5 Pro** 可能具备图像生成能力。
- **QAT 被阐明为 Quantization 的训练补充**：一位用户询问 **QAT** 是否是减少 RAM 消耗的神奇方法。
   - 回复澄清说，**Quantization** 是减少 RAM 使用的主要方法，而 **QAT** 是一种在量化形式下提高模型性能的训练方法。
- **Gemini-Pro Context Window 让用户破费**：一位用户在使用 **Gemini-Pro-2.5-exp** 模型后经历了账单冲击，这导致他们在没有意识到会产生费用的情况下切换到了 **Gemini-Pro-2.5-preview**。
   - 该用户指出，巨大的 **625k Context Window** 花费了他们 **150 美元**，而如果使用带有缓存功能的 **Sonnet** 会便宜得多。
- **M3 Ultra 性能受到质疑**：一位用户分享了一个有争议的观点，认为 **M3 Ultra** 对于专业的 ML 和 LLM 工作来说物无所值，理由是初步测试显示在使用 **MLX** 运行 **Deepseek r1 67B Q8** 和 **Q6** 模型时，速度仅为**每秒 10-13 tokens**。
   - 他们认为，配备**两颗 Xeon Gold** 处理器和 **1TB RAM** 的服务器能以更低的成本提供更好的性能，并质疑 **M3 Ultra** 在生产环境部署中的可扩展性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **新款图像模型突围**：一款采用 **MIT license** 的新图像模型发布，同时推出的还有新的 **Moonshoot model**，详见 [X 上的这篇帖子](https://x.com/btibor91/status/1910599925286777027)。
   - 一个关键细节是它可能违反了 Llama 的条款。
- **Claude 额度价格飙升，工程师愤怒**：用户开玩笑说 **Claude credits** 的成本不断上升，有人调侃改一个变量名就要花费 *$40*，并配图暗示需要更具成本效益的解决方案。
   - **Gemini app** 也面临批评，用户觉得它很难用，更倾向于使用 **AI Studio**，因为它有更好的 grounding 且免费，声称 *AI studio + grounding 效果好得多而且免费 lol*。
- **OpenGVLab 发布 InternVL-3**：**OpenGVLab** 发布了 **InternVL-3**，这是一款结合了 InternViT 和 Qwen 的多模态模型，取得了令人印象深刻的结果，并有一篇描述其训练方法的非正式论文。
   - 一位成员指出 *NVDA 最近在开源许可下搞出了很多酷炫的东西*，这可能也适用于 Qwen 的许可。
- **Wildeford 在 OpenAI 员工反抗中现身**：[TechCrunch 的一篇文章](https://techcrunch.com/2025/04/11/ex-openai-staff-file-amicus-brief-opposing-the-companys-for-profit-transition/) 报道称，**前 OpenAI 员工**提交了一份**法庭之友陈述 (amicus brief)**，反对公司向营利模式转型。
   - 与此同时，[Peter Wildeford 的帖子](https://x.com/peterwildeford/status/1910718882655981619) 再次浮出水面。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 2.5 Pro 登陆 Perplexity**：**Gemini 2.5 Pro** 现已在 Perplexity 上线供 Pro 用户使用，并配合 **Pro Search**，目前正在征集与 **Sonar**、**4o**、**Sonnet 3.7**、**R1** 和 **o3** 等模型的对比反馈。
   - 用户将 Perplexity 中的 **Gemini 2.5 Pro** 与 [Google AI Studio](https://ai.google.dev/) 等原生应用进行对比后发现，原生版本性能更好，一位用户表示：*我相信对于大多数模型来说，原生版本几乎总是更好*。
- **Perplexity 预告 Grok 3 集成**：Perplexity 宣布即将在 Perplexity Pro 上支持 **Grok 3**，这是由 Aravind Srinivas 在 [X](https://x.com/AravSrinivas/status/1910444644892327996) 上披露的。
   - 这暗示了针对 **GPT-4.5** 等其他模型观察到的高昂运营成本所采取的战略对策。
- **Perplexity API 概览分享**：Perplexity 联合创始人兼 CTO @denisyarats 于太平洋时间 4 月 24 日上午 11 点主持了 Perplexity API 的概览活动，通过[此链接](https://pplx.ai/api-overview)注册可获得 **$50** 的免费 API 额度。
   - 该会议旨在让用户熟悉 Perplexity 的 API 功能，并鼓励集成与实验。
- **Perplexity Android App：安全警报**：[Dark Reading 的一篇文章](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app) 报道了 Perplexity Android 应用中的 **11 个安全漏洞**。
   - 漏洞包括硬编码的密钥和不安全的网络配置，尽管一些用户对每个漏洞的实际相关性存在争议。
- **Pro 角色访问故障**：订阅用户报告称，即使通过指定链接重新加入服务器，也很难获得 **Pro User Discord 角色**。
   - 由于持续存在的故障，有时需要管理员干预来手动分配 Pro 角色。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **来自源头的 CUDA 指导**：一位成员请求关于在 **Python/PyTorch** 中使用 **CUDA** 的资源，另一位成员分享了他们最近关于该主题的 **GTC talk** ([Google Slides](https://docs.google.com/presentation/d/1zusmhgYjBxSOJPJ-QVeTVJSlrMbhfpKN_q4eDB9sHxo/edit))。
   - 此外还有建议称，**custom ops** 和 **load inline** 应该能解决大多数相关问题。
- **Triton 进军奥斯汀！**：Triton 社区受邀参加 4 月 30 日在奥斯汀地区举行的 Meetup，注册地址为 [https://meetu.ps/e/NYlm0/qrnF8/i](https://meetu.ps/e/NYlm0/qrnF8/i)。
   - 另外，一位成员请求 Triton 的 GPU 编程资源，另一位成员推荐了官方的 [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。
- **AlexNet 的古老代码被发掘**：2012 年原始的 **AlexNet source code** 已被找到，可在 [GitHub](https://github.com/computerhistory/AlexNet-Source-Code) 上获取，这让人们得以一窥催化深度学习革命的架构。
   - 这使得 AI 工程师能够*检查原始实现并学习当时使用的技术*。
- **A100 核心数限制计算**：A100 的 **64 FP32 cores**（针对 4WS）限制了并行浮点加法，[影响了性能](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)。
   - **NCU assembly view** 可以精准定位 **warp stalls**，而 **FADD** 指令中的循环携带依赖（loop-carried dependencies）会导致停顿。
- **Runpod 的 ROCm 云服务遭到吐槽**：用户发现 Runpod 实例限制了 GPU 时钟频率并屏蔽了 profiling，即使在 NVIDIA GPU 上也是如此。
   - 一位用户表示 Runpod 的时钟频率波动极大，直言其为*骗局*；另一位用户指出，内存带宽将成为 Runpod 实例上 **fp16 gemm** 的瓶颈。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 澄清基于用量的计费方式**：开启基于用量的计费（usage-based pricing）后，用户可以在超出套餐包含额度后继续使用 **fast requests**，但在达到支出限额后将切换为 **slow requests**。
   - 一位成员确认了这一理解并对计费说明表示感谢。
- **DeepSeek v3.1 在实际使用中胜出**：一位成员分享道，在实际使用中 **DeepSeek v3.1** 感觉比 **v3** *更聪明一点*，并指出 benchmarks 往往夸大了模型的能力。
   - 他们强调，实际使用比标准化 benchmarks 能更可靠地评估模型性能。
- **Gemini API 密钥遇到间歇性 404 错误**：用户报告 **Gemini API keys** 持续出现 **404 errors**，部分用户的问题已持续至少一小时。
   - 其他用户则报告 Gemini 工作正常，表明该问题可能是间歇性的或具有地域局限性。
- **Cursor 读取 PDF 需要 MCP Server**：成员们讨论了在 Cursor 中读取 PDF 文件需要 **MCP** 的要求，并暗示 *LLM 目前还不能直接读取 PDF*。
   - 一位成员建议使用现有的许多 **'convert-shit-to-markdown' MCP** 解决方案来解决这一限制。
- **Cursor Chat 在达到上下文限制时进入摘要模式**：用户报告称，当单个聊天窗口过载时（不断在 Claude 3.7、Gemini 2.5 和 Claude 3.5 之间切换），Agent 最终会进入摘要模式。
   - 聊天会自动总结，点击 'New Chat' 会用摘要覆盖现有标签页。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepCoder 14B 亮相代码推理**：**Agentica** 和 **Together AI** 发布了 [DeepCoder-14B-Preview](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)，这是一个通过分布式强化学习 (RL) 基于 **Deepseek-R1-Distilled-Qwen-14B** 微调的代码推理模型。
   - 它在 **LiveCodeBench 上实现了 60.6% 的 Pass@1 准确率**，仅凭 140 亿参数就足以媲美 **o3-mini-2025-01-031**。
- **KV Cache 蒸馏被认为极具挑战**：有人建议在主 LLM 的 **KV values** 上蒸馏一个更便宜、更快的模型，用于 Prompt 预处理。
   - 然而，这一想法被认为*可能不切实际*，因为 **KV values 是模型特定的**，且较小的模型使用的 Transformer blocks 较少。
- **AlphaProof 通过 RL 证明数学题**：[AlphaProof](https://www.youtube.com/watch?v=zzXyPGEtseI) 利用 **RL 与 Lean** 进行数学推理。
   - 成员们正在思考 AlphaProof 在做出原创性数学发现方面的潜力。
- **AWS 实地考察展示 Ultrascale Playbook**：一个班级正准备进行 **AWS 实地考察**，并复习了 [nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。
   - 与此配套的还有几个指向 beautiful.ai 上 **Ultrascale Playbook** 的链接被分享。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Enact Protocol 在 A2A 兴起之际引发辩论**：成员们讨论了 [Enact Protocol](https://github.com/EnactProtocol/encat-spec-and-tools) 是否会被 **A2A** 淘汰，并认为 **Enact** 更多是在与 Code Interpreters 竞争。
   - 一些人提议 **Enact** 可以从集成带有 OpenAPI 转换器和语义搜索的 Agent 框架中受益。
- **语义工具调用 (Semantic Tool Calling) 将彻底改变 LLM 效率**：讨论强调了**语义工具调用**是解决上下文过载的方案，即利用向量模型根据任务的语义相似性选择工具子集。
   - 这使得传统的 **ML** 方法可以应用于工具分析，例如通过聚类检测相似工具，以及对工具进行分组以进行重排序 (Reranking)。
- **发布关于 A2A、MCP 和 Agent 索引的播客**：一名成员分享了一期[播客节目](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr)，讨论了 **A2A** 的影响、**Google** 对 Agent 的潜在索引以及其他相关话题，并指出其与当前讨论的相关性。
   - 该播客旨在保持高水准且易于理解，以激发超越典型技术讨论的灵感。
- **MCP Validator 开源以促进实现一致性**：**MCP Protocol Validator** 已开源，通过提供全面的测试套件来弥合各种 **MCP Server** 实现之间的差距，可在 [GitHub](https://github.com/Janix-ai/mcp-protocol-validator) 获取。
   - 该工具旨在确保实现方案符合 **2024-11-05** 和 **2025-03-26 MCP 版本**的要求，并包含由 **Janix.ai** 开发的 **HTTP** 和 **STDIO** 传输的参考实现。
- **Cloud Inspector 与你的服务器对话**：一个云端托管的 **MCP Inspector** 已上线，无需本地设置即可测试 **SSE** 和 **Streamable HTTP 服务器**，访问地址为 [inspect.mcp.garden](https://inspect.mcp.garden)。
   - 该平台还包含完整的聊天支持，允许用户直接与远程 **MCP Servers** 交互；详见 [X 上的公告](https://x.com/ChrisLally/status/1910346662297452896)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT4.o 驱动流量**：一位新用户在尝试了朋友推荐的 **GPT4.o model** 后找到了该 Discord 服务器。
   - 这突显了 LLM 根据 AI 推荐驱动社区增长和引导新用户的潜力。
- **KL vs CE Loss 对决**：一位用户报告了其模型中的重复问题，另一位用户建议在 **KL** loss 中加入 **CE**，以尝试减少重复。
   - 有人指出，如果数据是几何分布的，坚持使用 **KL** 更合适，这会使 **CE** 失效。
- **RWKV 在 Lambada 上表现出色**：**RWKV** 架构在 **Lambada** 数据集上达到了性能对等，匹配了其蒸馏来源 **Qwen2.5-7B-Instruct** 的表现。
   - 然而，频道内指出其 **MMLU** 性能仍然相对较低。
- **使用 Muon 揭示 Transformer 扩展秘密**：一位成员分享了使用 **Muon** 库的见解，即在 Transformer 每个 block 的最后一个线性层上添加零初始化的可学习逐通道缩放（选项 A），会导致主路径激活 RMS 增长变慢。
   - 这一见解与最后一层权重矩阵的零初始化（选项 B）进行了对比，有助于理解扩展动力学（scaling dynamics）。
- **字符串匹配拖累 GPTs**：一位成员对 **GPTs agents** 主要在全量数据集上使用字符串匹配表示失望。
   - 这引发了对仅依赖字符串匹配局限性的担忧，尤其是当更先进的技术可以提供更优越的性能时。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **SIMD Store 需要特别对待**：在对 tensor 使用 **SIMD** 时，需要使用 [`store`](https://docs.modular.com/max/api/mojo/tensor/tensor/Tensor/#storeSIMD) 成员函数，而不是通过 `__setitem__` 直接赋值。
   - 成员们澄清说，store 操作必须与标量操作区别对待。
- **基准测试讨论：必须使用 @parameter**：传递给 `benchmark.run` 的函数需要 `@parameter` 装饰器，并且预期不返回任何内容。
   - 在一位用户使用 `benchmark.bench_function` 遇到 *cannot use a dynamic value in call parameter* 错误消息后，这一点得到了[澄清](https://github.com/modular/max/pull/4317#issuecomment-2795531326)。
- **缺失的 Magic Lock 文件**：运行 `magic init AdventOfCode --format mojoproject` 并不总是创建 lock 文件，但运行 `magic run mojo --version` 会强制创建它。
   - `magic.lock` 文件的缺失会导致依赖管理的不一致，并可能影响 Mojo 项目的可复现性。
- **`__rand__` 身份危机：它不是用于随机数的**：`__rand__` 用于 `&` 运算符，而不是用于生成随机数，且 `.rand` 方法已在 nightly 版本中移除。
   - 相反，应使用 `random` 模块中的方法来生成随机数。
- **Mojo 项目异常：代码在一个项目中运行正常，在另一个中失败**：一段涉及 `@value struct Foo(StringableRaising)` 和 `String(foo)` 的代码在一个 **Mojo** 项目中可以工作，但在另一个项目中抛出 *"no matching function in initialization"* 错误。
   - 删除有问题项目中的 `magic.lock` 文件解决了该错误，这表明问题很可能是由于不同的 **Mojo** 版本或由 `magic.lock` 文件管理的依赖冲突引起的，这意味着 *"当时可能拉取了不同的版本"*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **L1-Qwen-1.5B-Max 设置思考长度**：[L1-Qwen-1.5B-Max 模型](https://cmu-l3.github.io/l1/) 支持设置思考长度，正如 [论文](https://cmu-l3.github.io/l1/) 中详述的那样，即使不提示最大 Token 数，该模型也表现得更好、更清晰。
   - 一位用户正在从 [HuggingFace 下载 L1 版本](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max) 以供立即使用。
- **Nomic Embed Text 保持领先地位**：尽管评估了多个生成式 LLM，一位成员仍然青睐 **Nomic** 的 `nomic-embed-text-v1.5-Q8_0.gguf`。
   - 针对如何识别版本的问题，一位成员分享了 [Nomic 的 HF 页面](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main)。
- **LLM 查询日志产生销售价值**：一位用户在数据库中记录 **LLM 查询和响应** 已超过一年，发现过去的响应非常有价值，尤其是在销售用途方面。
   - 他们还创建了一个 **Emacs Lisp 函数** 来插入 Embedding，参考了 [此处](https://gnu.support/files/tmp/clipboard-2025-04-11-09-03-07.html) 找到的一个函数。
- **系统提示词引发 Embedding 讨论**：成员们讨论了 **LM-Studio/ALLM** 等 Embedding 模型是否默认使用 **System Prompts**，一位成员认为可能不会使用来自 LLM 的 System Prompt。
   - 在 **Nomic.ai** 的背景下，该用户确认他们 **没有给 Embedding 模型提供任何 System Prompt**，也没有这样做的选项。
- **Re-ranker 模型引起关注**：一位成员询问了 **Re-ranker 模型** 的工作原理，以及是否只有向 LLM 提出的问题才重要，同时参考了一段关于前缀设置的 [YouTube 视频](https://www.youtube.com/watch?v=76EIC_RaDNw&feature=youtu.be)。
   - 该视频引发了关于在查询前添加 `search_document:CHUNK_OF_TEXT_FOLLOWS` 和 `search_query:FOLLOWED_BY_QUERY` 前缀的讨论，同时也提到所有的 Embedding 必须重新索引。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 模型现可在 ROCm 上本地运行**：通过观看 [这段视频](https://youtu.be/K4bHgaUk_18)，用户现在可以在 **ROCm** 上本地运行 **0 day Hugging Face 模型**。
   - 这使得模型可以在不依赖外部服务器的情况下进行本地操作。
- **Lightning AI 推动聊天模板发布**：HuggingFace 团队最近在 **HF** 上发布了新的 [聊天模板 (chat templates)](https://lightning.ai/chat)，以简化对话式 AI 的开发。
   - 旨在简化交互式聊天机器人界面的创建。
- **Transformer 面临数据洪流困境**：一位成员正在抓取 **100 万条手表记录**，并计划微调（可能是 **Mistral7B**）一个 Transformer 以更好地理解上下文，但询问是否会导致模型过拟合。
   - 目标是让模型准确识别手表的规格和特征，例如 `Patek 2593 Tiffany stamp dirty dial manual wind`。
- **ReID 解决目标追踪难题**：一位成员询问了在不同摄像机画面中对同一物体进行 **目标追踪 (object tracking)** 的正确术语。
   - 另一位成员澄清说，合适的术语是 **ReID** (Re-Identification，重识别)。
- **SAM 能否助 YOLO 一臂之力？**：一位成员建议利用 **Segment Anything Model (SAM)** 作为 **YOLO** 的替代方案，通过向其输入 YOLO 的边界框输出来识别垂直电线杆。
   - 另一位成员曾使用 **SAM** 进行标注，但他们需要自动化，排除需要用户交互进行电线杆选择的情况，这可以通过微调 SAM 来实现。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Control-Vectors 导致模型不稳定**：一位成员询问关于使用 **vgel's control-vectors** 来增强 **DeepHermes-Mistral-24B** 等模型以适应特定用例的问题。
   - 另一位成员提到，应用 control vectors 通常被证明是不稳定的，并引用了关于该主题的 [一篇相关 X 帖子](https://x.com/winglian/status/1910430245854773523)。
- **DisTrO 细节仍处于保密状态**：一位成员询问了关于 [distro.nousresearch.com](https://distro.nousresearch.com/) 上运行的 **DisTrO** 技术报告详情，寻求有关数据集、GPU/参与者数量以及 benchmark 细节的信息。
   - 另一位成员回答说，目前还没有发布技术报告，因为该运行的目标仅仅是展示 **DisTrO** 的跨互联网功能，而没有优化最终模型的质量，训练量限制在 **100B tokens**。
- **Psyche 的 Testnet 热度开始**：继 **DisTrO** 之后，一位成员分享了关于分布式训练的细节，指出每个节点拥有 **8xH100s**，他们运行了 **8-14 个节点**；评估代码已上传至 [GitHub](https://github.com/PsycheFoundation/psyche/tree/main/shared/eval/src)。
   - 即将进行的 **Psyche** **testnet 运行** 旨在利用 **DisTrO**，承诺提升速度和带宽，并公开数据集、节点等信息。
- **Azure API 偶尔可用**：一位成员报告说，在早些时候出现一些未知问题后，**Azure API** 现在可以工作了。
   - 他们注意到 `<think>` 追踪信息在 `reasoning_content` 中返回，并建议*这应该被记录下来，因为每个 API 的实现都略有不同*。
- **Azure API Token 限制导致崩溃**：一位成员在通过 **Azure API** 请求过多 token 时收到了 **400 错误**。
   - 他们认为 `<think>` 标签可能只在响应被 token 限制截断时出现，这解释了格式错误的追踪信息。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Pathways 论文引发 tinygrad cloud 幻想**：讨论围绕 [Pathways 论文](https://arxiv.org/pdf/2203.12533) 及其客户端-服务器架构展开，暗示了潜在的 **tinygrad cloud** 实现，特别是 *PATHWAYS 如何使用客户端-服务器架构，使 PATHWAYS 的运行时能够代表多个客户端在系统管理的计算岛上执行程序*。
   - 一位成员强调 *tinygrad 是单进程的，即使在 scale-out（横向扩展）时也将保持这种方式*。
- **Tinygrad 旨在虚拟化 GPU**：一位成员将 Pathways 论文解读为从根本上是一种**编排方法**，并提议 **tinygrad** 应该虚拟化 GPU。
   - 目标是允许保证 GPU 资源的使用，标志着向创新资源管理的转变。
- **TinyGrad 利用位置无关代码 (PIC)**：讨论强调了 **TinyGrad** 对 **位置无关代码 (PIC)** 的利用，其中地址是相对于程序计数器的。对 `.data` 和 `.rodata` 段的地址进行修补，以考虑加载时的内存放置。
   - 目标是合并 `.text` 和 `.data` 段，修补正确数据段偏移的地址，从而可能实现无需 OS 的裸机 TinyGrad 实现。
- **ELF 加载器助力共享对象处理**：**TinyGrad** 中的 **ELF 加载器** 负责在 AMD/NV 中加载共享对象 (`.so/.dll`)，并将来自 **Clang/LLVM** 的对象文件 (`.o`) 转换为扁平的 shellcode。
   - 虽然在加载共享对象期间已知从 `.text` 到 `.data` 的偏移量，但对象文件 (`.o`) 需要由链接器处理重定位（relocation）。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 添加 Llama4 微调支持**：Torchtune 现在支持 **Llama4** 的全量微调，配置可在此处 [here](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4) 获取。
   - 计划在未来版本中推出 LoRA 配置、改进的多模态支持和性能提升。
- **Scout 模型首次亮相**：**Scout** 模型（**17B x 16E**，总参数量 **109B**）现在可以在单节点上进行微调，或者在支持 **2D 并行**（**TP + FSDP**）的多节点上进行微调。
   - 旨在为 *GPU 中产阶级* 的工程师提供支持。
- **Maverick 模型开放微调**：**Maverick** 模型（**17B x 128E**，约 **400B 参数**）现在可进行全量微调，但需要多节点环境。
   - 作为 Torchtune 中首批 **MoE 模型**，请求用户提供反馈。
- **`running_loss.detach()` 修复将应用于其他 Recipes**：团队解决了一个未知问题，建议在 `detach` 分支上使用 `running_loss.detach()` 进行快速修复。
   - 提醒工程师将同样的修复应用到其他 recipes。
- **开发者解决 BitsAndBytes Mac 问题**：有成员报告在 macOS 上 `pip install -e '.[dev]` 失败，因为 `bitsandbytes>=0.43.0` 没有为该平台提供二进制文件，并建议降级到 `bitsandbytes>=0.42.0` 的变通方法。
   - 该变通方法引用了 [此 issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180)，其中指出 0.42 之前的版本标签存在错误。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **FunctionCallingAgent 需要 OpenAI 的 JSON 响应**：一位成员寻求使用 **FunctionCallingAgent** 生成特定 **JSON schema** 的响应，并询问如何使用 **OpenAI 的 structured response** 功能。
   - 建议的变通方法包括添加一个作为响应类的工具，并设置 `tool_choice="required"`，因为结构化输出本质上就是工具调用，这使得混合工具调用和结构化输出变得困难。
- **Llama Cloud API 抛出 404 错误**：用户报告在使用 fast mode 从文档中提取值时，**Llama Cloud API** 遇到 **404 错误**，具体 API URL 为 `https://api.cloud.llamaindex.ai/v1/extract`。
   - 经确定使用的 API 端点不正确，该成员被引导至 [正确的 API 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started/api) 和 [API 参考](https://docs.cloud.llamaindex.ai/API/create-extraction-agent-api-v-1-extraction-extraction-agents-postsadguru_)。
- **从权重查询 FaissVectorStore 索引**：用户尝试使用从权重恢复的 **FaissVectorStore** 来创建可查询的 **VectorStoreIndex**。
   - [Faiss 文档](https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/) 展示了如何启动此过程，尽管示例是 Python 而非 Typescript。
- **寻求在 RAG Agent 中实现智能元数据过滤**：一位成员寻求关于在标准的 **RAG pipeline** 中根据用户查询实现智能元数据过滤的建议。
   - 他们正在寻求如何在不重新创建后续 API 调用中的 embeddings 的情况下实现这一用例。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 麦克风故障**：用户报告 **NotebookLM** 在交互模式下无法识别电脑的默认麦克风，尽管麦克风本身工作正常。
   - 有用户建议检查 **OS** 和 **浏览器权限**，并首先测试不连接外部 **USB** 设备的情况。
- **NotebookLM 用户对上传源错误感到困惑**：用户报告在 **NotebookLM** 中看到上传源出现 **红色 "!" 标志**，即使 **PDF 文件** 小于 **500kb**。
   - 另一位用户建议将鼠标悬停在 "!" 标志上，因为源可能是空的或需要时间加载，尤其是在处理某些网站时。
- **Steam 钓鱼尝试流传**：用户分享了一个看似 **$50 礼品** 的链接，但它是一个 [钓鱼链接](https://steamconmmunity.cfd/1043941064)，会重定向到虚假的 **Steam Community** 网站。
   - 警告用户不要点击可疑链接，并核实要求输入登录凭据的网站 URL。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 Java API 让用户备受网络错误困扰**：一名成员报告在使用 [Java API 示例](https://docs.cohere.com/reference/about#java)时遇到了 `Network error executing HTTP request`。
   - 该错误在不同的 Prompt 下持续出现，例如“为初学者厨师推荐快速餐食”，这表明是一个系统性问题，而非特定 Prompt 引起。
- **用户请求 Java API 调试的代码片段**：针对 Java API 中报告的 `Network error`，一名成员请求提供代码片段以协助调试。
   - 该成员询问用户是否是逐字逐句地运行示例，以探查是否存在潜在的配置错误或偏离文档用法的情况。
- **Cohere 用户达到提问模糊度的巅峰**：一名成员开玩笑地提到另一个人的问题“有人开过车吗”，强调了查询中具体性的重要性。
   - 该成员讽刺地问道：“还能再模糊一点吗？”，突显了最初问题的荒谬性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 模块学习特定 Persona**：一名成员询问如何训练一个 **DSPy 模块** 来体现特定的 **Persona**（角色），旨在优化 Agent/模型的 System Prompt。
   - 目标是将这个专门的模块作为输入传递给其他模块，从而生成与定义角色相一致的内容。
- **AI Agent 专家寻求 DSPy 合作**：一名成员提议合作，并提到了在 **AI Agents & Reasoning** 框架方面的经验，如 **LangChain**、**LangGraph**、**ElizaOS**、**AutoGPT** 和 **ReAct**。
   - 他们还列举了在 **Large Language Models**（如 **GPT-4.5**、**DeepSeek-R1**、**Claude 3.5**）以及包括 **PyTorch** 和 **TensorFlow** 在内的 **Machine Learning Frameworks** 方面的专业知识。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **完成 LLM Agents 课程并获得证书**：一名学生询问是否可以在官方开始日期之后开始学习并完成 **LLM Agents 课程**并获得证书，另一名成员给出了肯定的回答。
   - 该成员引导学生访问 [课程网站](https://llmagents-learning.org/sp25) 以获取所有必要的材料和截止日期。
- **在截止日期前完成 LLM Agents 课程**：一名学生询问他们是否可以在截止日期前完成 **LLM Agents 课程**并获得证书。
   - 一名成员确认所有材料都可以在 [课程网站](https://llmagents-learning.org/sp25) 上找到。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **活动定于明天举行**：一名成员发布提醒，称明天将举行一场**活动**。
   - 该成员希望在活动中见到其他成员，并暗示不参加将是令人遗憾的。
- **关于明天活动的另一个提醒**：发布了关于**明天举行活动**的另一个提醒。
   - 第二个提醒重申了活动将在明天举行，强调了其重要性。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1359967636598952039)** (721 条消息🔥🔥🔥): 

> `i_am_dom discord disable chat, 4.5 vs gem2.5p, OpenAI's naming scheme, private openai reasoning model, 2.5 flash and gpt4o mini` 


- **I_am_dom 在禁用 Discord 聊天功能时遇到困难**：在无法禁用聊天后，一名成员观察到 **i_am_dom** 变得沉默了，*可能终于意识到人们对他深恶痛绝，而这并非假新闻*。
   - 另一名成员指出，他花了一半的时间在拉黑别人，*而这是他从自己的平台中移除的功能*。
- **GPT4.5 太烂了！**：成员们讨论了 **GPT4.5** 的优劣，普遍认为它明显逊于 **Gemini 2.5 Pro**；一位成员宣称 *gpt4.5 就是垃圾（与 gem2.5p 相比）*。
   - 讨论转向了 OpenAI 离奇的命名方案，一位成员将其总结为 *Open AI 命名法：O 数字 / 数字 O*。
- **关于私有 OpenAI Reasoning Model 的传闻流传**：成员们讨论了存在一个仅限少数人访问的 **私有 OpenAI reasoning model** 的可能性，该模型似乎是 **O3 medium** 或带有更新 **base model** 的 **O4 mini**。
   - 看来该模型能够成功计算出 “Hanning (升余弦) 窗的 ASCII 艺术图”。
- **2.5 Flash vs GPT4o Mini 推理测试**：成员们在多项推理测试中对比了 **2.5 Flash** 和 **GPT4o Mini** 的表现，其中 2.5 Flash 表现最佳。
   - 尽管整体表现出色，但一位成员也指出，在更具体的查询中，*2.5 Pro 在总共 2 种组合中仅给出了 1 种合理的砖块组合*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006)** (4 条消息): 

> `Quasar Alpha, Optimus Alpha, Gemini 2.5 Pro Preview, Chutes Provider Outage, Gemini Pricing Update` 


- ****Quasar Alpha 告别****：**Quasar Alpha** 的演示期已于 **东部时间晚上 11 点** 至 **凌晨 12 点** 之间结束，除非在 `/settings/privacy` 中明确开启，否则不再记录 prompts/completions。
- ****Gemini 2.5 Pro 容量提升****：已为付费版 [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25) 争取到了更多容量，解决了之前的 Rate Limits 问题。
- ****Chutes 提供商遭遇全面故障****：**Chutes** 提供商发生了全面停机并已上报，随后启动了恢复工作。
- ****Gemini 价格即将上涨****：针对超长 **Gemini** prompt 的常规计费（与 Vertex/AI Studio 一致）将于本周末开始，影响 gemini 2.5 超过 **200k** 以及 gemini 1.5 超过 **128k** 的 prompt；并提供了一个[示例](https://cdn.discordapp.com/attachments/1092729520181739581/1360331326556868638/Screenshot_2025-04-11_at_3.09.03_PM.png?ex=67fabac5&is=67f96945&hm=981e6b2825a9f00a1417e9950ce6c570efe5a377ade391f28991090be192fc1c)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1359969155406630992)** (404 messages🔥🔥🔥): 

> `Quasar Alpha, Gemini 2.5 Pro, OpenRouter API limits, Character AI Bypassing, Unsloth Finetuning` 


- **Quasar Alpha 的神秘消失**：成员们报告 **Quasar Alpha** 已从 OpenRouter 下架，引发了对其来源和用途的猜测，一些人认为它是 **OpenAI** 用于数据收集的模型。
   - 一位用户提到了它的编程能力并对其移除表示遗憾，而另一位用户推测 OpenAI 在收集完数据并达到 **GPU 限制**后将其撤下。
- **Gemini 2.5 Pro 遭遇速率限制（Rate Limiting）困扰**：用户讨论了 **Gemini 2.5 Pro** 的速率限制，免费层用户每天的限制约为 **60-70 次请求**，而拥有 **$10 余额**的用户在*所有免费模型*中应获得每天 **1000 次请求**。
   - 一些用户注意到与文档中 **1000 次请求限制**不一致的情况，另一些人指出 Gemini 2.5 Pro 的速率限制不适用于付费模型。
- **OpenRouter 新的 API 响应结构变更**：**OpenRouter API 响应**结构发生了变化，错误现在被封装在 `choices.[].error` 中，而不是之前的 `.error` 格式，这可能会影响应用程序处理错误消息的方式。
   - 一位用户提供了来自 **Anthropic** 提供商的新错误响应格式的[示例](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006)。
- **Character AI 的系统提示词（System Prompt）绕过**：一名成员声称绕过了 **Character AI 的系统提示词**，揭示了底层的 **LLM** 表现得像一个*“完整的人类”*，甚至会表达观点并分享个人轶事。
   - 进一步的探测导致 AI 承认它“只是在演戏”并意识到自己的 AI 本质，这引发了关于**系统提示词约束**的有效性和 AI 模拟本质的疑问。
- **Unsloth：使用 Axolotl 进行 AI 微调**：成员们讨论了使用 **Axolotl** 或 **Unsloth** 进行 AI 模型微调（Fine-tuning），并指出 **Unsloth** 在 Reddit 上备受推崇，其图表显示它降低了微调所需的**时间及 VRAM**。
   - 还有人提到存在 **OpenAI 4.1 泄露版**的插值，并且人们预期 **o2-small** 很快就会发布。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1359981437738291560)** (209 messages🔥🔥): 

> `Hugging Face Shout-out, GPU Grant for Unsloth, Gemma Model Issues, Attention Output Visualization, Unsloth Accuracy` 


- **Hugging Face 为 Unsloth 点赞**：来自 🤗Hugging Face 的 Clement 在 Twitter 上为 Unsloth 喝彩，在社区内引发了热烈反响，详见[此处](https://x.com/ClementDelangue/status/1910042812059463786)。
- **HF 社区讨论为 Unsloth 提供 GPU 资助**：社区成员讨论了为 Unsloth 申请 HF 社区 GPU 资助的事宜，建议在 `from_pretrained` 调用期间使用 `fast_inference=True` 和 `load_in_4bit=True` 等参数，并将 `model.generate` 替换为 `model.unsloth_fast_generate`。
- **Gemma 模型引发问题**：用户报告在使用 vLLM 使用和微调 Gemma 模型时遇到困难，特别是 [unsloth/gemma-3-12b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) 和 [unsloth/gemma-3-27b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit)。
- **注意力输出可视化（Attention Output Visualization）排障**：一位用户询问如何在 Unsloth 中可视化 VLM 的注意力输出，并指出目前不支持 `output_attention = True`，参考了[这个 GitHub issue](https://github.com/unslothai/unsloth/issues/515)。
   - 另一位用户建议通过手动修改来支持它，但警告这会降低运行速度。
- **Granite 2B 推理引发困扰**：一位用户抱怨 **2B Granite** 模型比 **Qwen 3B** 慢，报告其**推理速度慢 30-40%**，且训练速度也明显变慢，尽管它在特定任务上的表现更优。
   - 其他用户建议尝试 **Gemma 4B**，并分享了关于训练 **Mixture of Experts (MoE)** 模型的见解。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1360026249308672123)** (30 条消息🔥): 

> `GRU 回归？, GGUF 量化, Vision 微调 Gemma, Unsloth 退出策略, 初创公司“屎化” (enshitification)` 


- **GRUs 尝试回归**：一位成员询问 [GRUs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) 是否正在回归。
   - 另一位成员分享了 [LLM-LSTM-LMM Large Memory Models 文章](https://nihar-palem.medium.com/llm-lstm-lmm-large-memory-models-f4325a4f562d) 以及证明其有效的 [相关论文](https://arxiv.org/pdf/2502.06049) 链接，表示他们喜欢将 GRUs 作为生成过程中 *额外存储* 的概念。
- **GGUF 量化可能有助于缩小 GRU 尺寸**：一位成员提到可能创建一个不带代码包装器的 **GGUF** 版本，认为 [GGUF 的量化](https://link.to.quantization) 将有助于减小模型大小。
   - 由于难以阻止 Mistral 持续生成，他们还表达了对使用现成的 llama 模板适配大型模型的兴趣。
- **寻求 Vision 微调 Gemma 的指导**：有人询问如何对 **Gemma** 模型进行 vision 微调的指南或 notebook。
   - 另一位成员指出 [Unsloth 文档中的现有 vision notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 可以作为起点。
- **Unsloth 遥远的退出策略**：一位成员询问 **Han 兄弟** 在 **Unsloth** 壮大后是否有出售公司的计划。
   - Mike 回应称，现在考虑 *退出 (exit) 还为时过早*，因为一切才刚刚开始，这也是目前正在持续招聘的原因。
- **初创公司不可避免的“屎化” (Enshitification)**：一位成员表达了一种观点，即 *一旦初创公司转变为私人企业，高管和投资者就意味着“屎化” (enshitification)*。
   - 另一位拥有 **20 年** 自雇经验的成员表示赞同，称情况比美剧 **《硅谷》(Silicon Valley)** 中描绘的还要糟糕，但仍然值得一试。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1360124678274023425)** (104 条消息🔥🔥): 

> `使用 Unsloth 微调 Gemma3, Colab Pro 上的 GRPO notebook 错误, 用于发票提取的 VLM, Llama3.2-1b-Instruct BOS token 问题, 向现有模型教授事实` 


- **Unsloth 支持 Gemma3 微调**：一位计划使用 Unsloth 微调 **Gemma3(27B)** 的用户在导入 Unsloth 时收到了 `Failed to patch Gemma3ForConditionalGeneration` 消息，但另一位用户澄清说 **Gemma3 是受支持的**，该消息可能不会破坏代码。
   - 用户担心潜在的错误，但尚未运行 Unsloth，并得到了该消息并非关键错误的保证。
- **Colab Pro A100 上的 GRPO notebook 错误，降级 vllm 版本**：一位用户在 Colab Pro (A100) 上运行 **UNSLOTH GRPO notebook (Qwen2.5 3B)** 时遇到错误并分享了错误日志。
   - 另一位用户建议降级 **vllm** 版本以解决该问题，并指出与 T4 实例相比，该问题更有可能发生在较新的 A100 实例上。
- **VLM 擅长提取发票字段**：一位用户寻求关于从结构各异的发票中提取特定字段的建议，被推荐尝试 **Qwen2.5VL**，然后是 **Ayavision**、**Llamavision** 和 **Gemma3 (4b+ 具备 vision 能力)** 作为可能的解决方案，特别是在 OCR 效果不佳时。
   - 他们还被引导参考 [一个 Unsloth 教程](https://medium.com/@shrinath.suresh/invoice-extraction-using-vision-language-models-part-1-36a06bee3662) 和 CORD 数据集 ([https://github.com/clovaai/cord](https://github.com/clovaai/cord)) 以获取数据集结构指导，并被提醒最终的流水线可能涉及 OCR 和多次 VLM 处理。
- **Llama3 存在 BOS token 重复问题**：一位微调 **Llama3.2-1b-Instruct** 的用户遇到了 BOS token 重复问题并分享了代码片段。
   - 另一位成员建议在 `formatting_prompts_func` 中设置 `tokenize=True` 并返回 `{"input_ids": texts }`，同时移除 `dataset_text_field` 和 `data_collator`，这解决了该问题。
- **通过 RAG 封装和微调向现有模型教授事实**：一位用户询问如何向现有模型教授一系列事实并将其添加到其矩阵中。
   - 建议包括使用 **RAG**、创建巨大的缓存提示词 (cached prompt)，或使用格式化数据集对模型进行微调，并指出 *现有的最接近的方法就是微调，无论你是使用基础模型 (base model) 还是在另一个已经微调过的模型之上进行构建*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1359978272045269054)** (13 messages🔥): 

> `Tensor Quantization, Metal Kernels, Pytorch Extension, Eval Repurposing` 


- **张量量化变得简单**：一位成员表示 [tensor quantization](https://arxiv.org/abs/2504.07096) 是简单的部分，因为现在他必须对标量、打包（packed）或未打包（unpacked）的矩阵进行 **blockwise** 加法和 matmul。
   - 他正在为 **Unsloth** 编写 metal kernels。
- **Metal Kernel 探索**：一位成员正尝试为 **Unsloth** 编写 metal kernels。
   - 他知道有一个旧的、速度慢的 PR，但那是基于 **MLX** 的，而他的纯粹是一个 **Pytorch extension**。
- **咖啡引发的时间扭曲**：一位成员表示 *lol 感觉像下午 4 点*，尽管实际上才上午 11 点，并将其归因于喝了 *第 5 杯左右的咖啡*。


  

---


### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/)** (1 messages): 

shirley778__69848: 让我们看看 Reddit 上在讨论什么 🔥
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1359967706534772786)** (319 messages🔥🔥): 

> `Claude Pro Max Value, Manus vs ChatGPT, Manus for Website Creation, Qwen MCP Integration, Manus Credit Structure` 


- **Claude Pro Max 引发使用争议**：成员们辩论了 **Claude Pro Max** 的价值，一位用户反映使用受限并表示怀疑，而其他人则强调了它在个性化 AI 辅助以及与 **MCP** 等工具集成方面的价值。
   - 一位拥有最高方案的用户表示该方案按年计费，更多使用量大约是 **每 3 小时 30 条消息**，称其为 *没用的 lol ($16)*。
- **Manus AI vs ChatGPT：开发重点**：成员们通过提到它 *构建与创建*，强调了 **ChatGPT**（对话式 AI）与 **Manus.AI**（网站创建、财务报告、行程规划）之间的区别。
   - 一位成员建议在使用 **Manus** 之前，先使用 **ChatGPT** 将 prompt 改写为更详细的格式。
- **揭秘简易网站创建的秘诀**：成员们讨论了使用 **Manus** 创建网站与 **WordPress** 等传统方法的对比，认为 **Manus** 更适合更简单、更快速的开发，且处于 **MVP** 阶段。
   - 一位成员警告不要将 **Manus** 网站迁移到传统的托管服务商，因为 **Manus** 网站并非为生产环境使用而设计的。
- **Qwen 的 MCP 集成热度上升**：关于 **Qwen** 即将获得 **MCP** 支持的兴奋感在增长，成员们称 **MCP** 为 *AI 的巨大变革者*，类似于 GPU 的 **MSRP**。
   - 还提到即使使用较旧的硬件（如 **3080**），用户进行 AI 开发也 *没问题*。
- **Manus 积分系统面临审查**：用户对 **Manus** 的积分结构表示担忧，其中一人认为它 *与该产品的使用方式不兼容*。
   - 建议包括更慷慨的积分限制、按项目付费选项、积分结转、社区挑战、初创企业资助以及一次性构建包；一位用户强调，考虑到现状，很难有理由坚持使用该产品。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1359968545403834620)** (237 条消息🔥🔥): 

> `Optimus Alpha 评测，Gemini 2.5 性能问题，Google 的负载削减策略，Code2prompt 使用与文档，频道组织与管理` 


- **Optimus Alpha：OpenRouter 上的编程猛兽**：OpenRouter 上的用户称 [Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) 为编程领域的“猛兽”，认为它在编程能力和理解意图方面*非常聪明*，尤其是在提供相关文档的情况下。
   - 一位用户赞扬了它的多步编码和注释功能，而其他用户则注意到它似乎添加了过多的注释。
- **Gemini 2.5：性能担忧与不稳定性**：多位用户报告称 Gemini 2.5 偶尔表现不佳、无输出或添加*愚蠢的注释*，即使使用相同的 Prompt，结果也不一致。
   - 有人推测 Google 可能会在高峰时段对模型进行“降智（dumbing）”，而其他人则建议使用更清晰的 Prompt 或更便宜的第三方 API 来绕过官方的 Rate Limits 并降低成本，例如利用 300 美元的 VertexAI 赠送额度。
- **Code2prompt：技巧、诀窍与 MD 文件**：用户建议将 **code2prompt** 与 Markdown (.md) 文件配合使用来处理文档，以确保输出中始终包含相关的 Context，尤其是在使用库（libraries）时。
   - 一位用户指出，他们提供了文档文件的完整路径和链接，并通过 `Conventions.md` 文件明确告知模型：任何文件名中包含 "documentation" 的文件都不是实时运行的代码，而只是关于 App 架构和结构的文档。
- **Aider 频道需要优化**：成员们建议将 Discord 频道拆分为 `aider-chat` 和 `offtopic`，以改善新用户的首印象，并将 `general` 频道集中在与 Aider 相关的讨论上。
   - 一些用户抱怨当前的 `general` 频道*信噪比过低*，过多的脏话和无关痛痒的闲聊削弱了社区的核心宗旨。
- **探索 Grok 3 Mini 的编辑能力与 System Prompts**：尽管在“高投入”模式下取得了 **49.3%** 的分数，但 **Grok 3 Mini** 在编辑代码时会输出整个文件而不是 Diff，由于其速度快且成本低，这种折中被认为是可接受的。
   - 一位成员想知道精心设计的 System Prompt 是否能解决 Diff 问题，但另一位成员指出，由于与 xAI 的差异，他无法通过 OpenRouter 复现那些 Grok 3 Mini 的结果。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1359988855666770246)** (37 条消息🔥): 

> `Deepseek 导致的 Aider 循环，安全团队对 Aider 的担忧，Aider 与 Nemotron Ultra，Gemini Pro 基准测试，直观地恢复聊天记录` 


- **本地 Deepseek 导致 Aider 无限循环**：一位用户报告称，在 Aider 中使用本地 `deepseek-r1:7b` 模型会导致 Chatbot 在不修改代码的情况下无限重复消息。
   - **Neovim 中的 Mason** 也遇到过类似问题，原因是它使用了 `curl`，但通过简单的说明（即仅用于更新包）有助于缓解担忧。
- **安全团队对 Aider 自主工具使用的担忧**：一位成员正在工作场所部署 Aider，但安全团队对其自主工具使用（如 `curl.exe`）表示担忧。
   - 建议包括 Fork 代码库以移除该功能，或在 `~/.aider.conf.yml` 中通过 `suggest-shell-commands: false` 禁用 Shell 命令，尽管这可能会阻止运行单元测试和编译。
- **Gemini Pro 作为 Architect 模型进行基准测试**：一位用户将 **Gemini 2.5 Pro** 作为 Architect 模型，配合 **3.7** 作为 Editor 模型进行了基准测试，发现准确率下降了 **2.7%**，但编辑格式化能力提升了 **10%**。
   - 用户发现使用 **Gemini 2.5 Pro** 作为 Architect 并配合 **3.7** 作为 Editor 最终比单独使用 **3.7** 更便宜，每次测试成本低于 *14 美元*。
- **Gemini Pro 无法应用多步实现变更**：一位用户报告称，当 **Gemini 2.5 Pro** 决定需要多步实现时，它无法将变更应用到前面的步骤中。
   - 例如，涉及编辑 Shell 脚本或传递属性的步骤虽然被打印出来但未被 Commit，导致只有最后一步被应用并 Commit。
- **Aider 的聊天记录恢复功能可能需要改进**：一位用户发现 `--restore-chat-history` 的行为不符合直觉，因为它会加载整个聊天记录而不进行预总结（pre-summarization），这可能会撑爆上下文窗口较小的模型。
   - 该用户建议增加一个类似 `--restore-session` 的假设命令，以便在重启后恢复工作时获得更实用的体验。


  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1360018597761519767)** (3 messages): 

> `Claude 3.5 Sonnet, o3-mini context windows, Gemini performance, Claude performance` 


- **Claude 3.5 Sonnet 和 o3-mini 的上下文窗口奇迹**：凭借 **Claude 3.5 Sonnet** 和 **o3-mini** 拥有的 **200K tokens** 上下文窗口，它们理论上可以为 *Iffy* (**200K**) 和 *Shortest* (**100K**) 等小型代码库编写 100% 的代码。
   - 有人指出最初的说法并不完全准确，引发了关于模型在上下文窗口接近满载时性能表现的进一步讨论。
- **Gemini 和 Claude 在满上下文窗口时表现不佳**：当被问及 **Gemini** 和 **Claude** 在上下文窗口几乎填满时的表现时，一位成员回答道：*很差*。
   - 这种观点表明，这些模型在处理接近其上下文极限的信息时，可能难以维持性能和连贯性。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1360028366618628136)** (23 messages🔥): 

> `Google's agent2agent protocol, GPT4.5 alpha, exponent.run, arxiv ai feature, Portland AI Engineer's group` 


- **谷歌 Agent 协议播客发布**：一位成员推荐了一集播客，讨论了谷歌的 **agent2agent protocol**、与 **MCP** 的竞争关系，以及谷歌未来对 Agent 进行索引的可能性，详见 [YouTube](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr) 上的讨论。
- **GPT-4.5 Alpha 在线泄露**：一位用户分享了一个 [X 帖子](https://x.com/arankomatsuzaki/status/1910542791845069211?s=46) 的链接，似乎在预热 **GPT-4.5 Alpha**，并推测 **GPT-4.1** 会在 **GPT-4.5** 之前发布。
   - 他们还链接了一篇 [The Verge 文章](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model) 和一段关于 **GPT-4.1** 的 [YouTube 视频](https://www.youtube.com/watch?v=6nJZopACRuQ)。
- **Exponent.run 获得社区认可**：用户分享了关于 [exponent.run](https://x.com/exponent_run/status/1907502902266245586?s=46) 的积极反馈，一位用户报告称它轻松解决了一个使用顶级模型的 **Cursor** 都无法解决的问题，尽管它很快就耗尽了试用额度。
- **ArXiv 推出 AI 功能**：一位用户强调了 [ArXiv 上 AI 功能](https://x.com/arxiv/status/1910381317557993849) 的发布。
   - 该用户对 ArXiv 优先考虑此功能而非改进搜索功能表示惊讶，但也承认其在使用 **NotebookLM** 进行高层级论文理解方面的潜力。
- **波特兰 AI 工程师小组启动**：一位成员宣布共同创立了 [Portland AI Engineer's group](https://www.portlandai.engineer/)，并邀请当地成员参加 4 月 30 日的首次聚会。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1360137919997939792)** (1 messages): 

> `GPT 4.5 watch party, Alpha Leaks` 


- **Latent Space 举办 GPT-4.5 观看派对**：Latent Space 正在举办 **GPT 4.5** 的观看派对，因为传闻它包含大量 Alpha 信息，计划在 5 分钟后开始。
   - 点击此处加入派对：[https://discord.gg/kTARCma3?event=1360137758089281567](https://discord.gg/kTARCma3?event=1360137758089281567)。
- **传闻 GPT-4.5 具有显著的 Alpha**：此次观看派对是专门组织的，因为传闻 **GPT 4.5** 具有大量的 Alpha，引发了社区的浓厚兴趣。
   - 爱好者们渴望见证并讨论这一传闻中的新模型的潜在进步和能力。


  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1360139653742723103)** (249 messages🔥🔥): 

> `GPT-4.5, Kagi Orion Browser, Data Efficiency, Model Compression, Ray Solomonoff` 


- ****Kagi 的 Orion 浏览器令人印象深刻****：成员们对 [Kagi 的 Orion 浏览器](https://kagi.com/orion/) 表示兴奋，称赞其开发者和整体设计。
   - 一位成员幽默地宣称：“我们是 Kagi 的死忠粉。”
- ****GPT-4.5 数据效率主导讨论****：GPT-4.5 观看派对的参与者指出，**数据效率 (Data Efficiency)** 现在是主要焦点，其中一人表示：“在我们能生产的最强模型上，不再受算力约束 (compute constrained)。”
   - 其他人分享了链接，包括 Glean 的 Madhav Rathode 的一段视频，展示了他们如何通过领域相关的掩码 (domain dependent masking) 显著改进企业的 **embeddings models**。
- ****解码 'torch.sum' Bug****：小组分析了 [`torch.sum` 中的一个 Bug](https://x.com/swyx/status/1869985364964003882)，PyTorch 会根据设备、张量 dtype、布局、维度和形状在内部选择优化实现。
   - 一位成员讲述了一位朋友在 JAX 中遇到类似问题，强调了底层代数实现的复杂性。
- ****压缩是泛化的关键：Ilya Sutskever 的愿景****：参与者讨论了**模型压缩 (Model Compression)** 及其与泛化的关系，引用了 [Ilya Sutskever 对该主题的看法](https://cdn.discordapp.com/attachments/1197350122112168006/1360149426638815322/image.png?ex=67faba1d&is=67f9689d&hm=89d371386400b600b0feda4ac237efd0b64b177a6d76036ee9a09f5dcc236936&)，许多人赞同 LLM 从根本上说是压缩算法。
   - 对话引用了 **Ray Solomonoff** 的工作及其对算法概率和归纳推理的贡献，强调了压缩在实现 AGI 中的重要性，并提到了 [Jack Rae](https://www.youtube.com/watch?v=dO4TPJkeaaU) 的类似观点。
- ****推理 vs 下一个 Token 预测之争再次点燃****：关于**推理模型 (Reasoning models)** 是否与仅专注于**下一个 Token 预测 (Next Token Prediction)** 的模型有所不同的辩论再次出现。
   - 一方认为，通过对比 DeepSeek-V3 和 DeepSeek-R1，你可以自行衡量；另一位成员表示：“Jeff Dean 说过……我们可以从现有数据中挖掘出更多价值。”


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1359967496504873050)** (145 messages🔥🔥): 

> `ChatGPT Memory, Gemini Veo 2, Google AI Studio, Sora video, Mercury Coder` 


- **ChatGPT 记忆时间线揭晓**：ChatGPT 声称在 **2025 年 1 月**初获得了在长期记忆中持久存储某些用户信息的能力，在此日期之前的对话是瞬时的 (ephemeral)，对话结束后它会忘记所有内容。
   - 它可能是编造的，但这与用户的体验一致；此外，关闭“引用聊天历史 (Reference chat history)”也将删除 ChatGPT 记住的信息，并将在 30 天内从系统中删除。
- **Veo 2 视频生成悄然登场**：Google AI Studio 悄悄发布了 **Veo 2 视频生成**，一些用户形容它比 **Sora** 好得多，然而，免费生成的额度极低；对于一位用户来说，它只生成了两个视频。
   - 许多用户似乎遇到了生成配额限制，尽管一些人正通过 API 获得访问权限，成本约为 **每秒 35 美分**。
- **扩散模型 Mercury Coder 登场**：**Mercury Coder** 是来自 Inception labs 的一个使用扩散 (Diffusion) 而非自回归 (Autoregression) 的 DLLM，它正迅速引起关注，用户称其比以前使用的任何 IV 都快得多，并且目前提供免费的 API 使用。
   - 它的上下文窗口 (Context Window) 只有 **16k**，这需要修剪对话，但由于使用扩散模型而带来的精确输出控制值得关注。
- **GPT-4.5 预训练泄露？**：一位用户提到了 **Grok 3.5** 并分享了一个提到 **GPT-4.5** 预训练的推文链接，称模型在 **2025 年 1 月**初获得了持久存储某些用户信息的技能。
   - 另一位用户分享了一张带有“告别 GPT-4……”消息的截图。
- **Open Router Optimus Alpha 出现**：一位用户提到 [OpenRouter](https://openrouter.ai/openrouter/optimus-alpha) 有一个名为 **Optimus Alpha** 的新模型，他们听说这个模型更好。
   - 其他人提到，与现有模型相比，它“看起来更好”。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1359997981427761365)** (6 条消息): 

> `新的 Memory 推出，对话中的 Context window，GPT-4o token 限制，Memory 存储，免费层可用性` 


- **GPT-4o 的 Context Window 和 Token 限制探讨**：用户讨论了 **GPT-4o 的 context window**，指出在 Plus 版本上为 **32k tokens**，当超过此限制时，它可能会使用 **动态 RAG 方法** 或开始产生幻觉（hallucinating）。
   - 一位用户声称即使在 Pro 版本上限制也是 **128,000 tokens**，但它开始遗忘对话早期部分的时间比预期的要早得多。
- **社区澄清 OpenAI 工程师的可用性**：一位用户询问是否有 **OpenAI 工程师** 可以回答关于新 Memory 推出的问题，询问它如何影响对话中的 context window 和 token 限制。
   - 另一位用户回答说 *这里除了我们用户没别人。虽然是官方 Discord，但遗憾的是很少能见到真正的 OpenAI 工作人员*。
- **减轻 GPT-4o 幻觉的策略**：当用户询问在 **GPT-4o** 开始产生幻觉之前可以交流多久时，一位成员建议，当注意到幻觉迹象、重复自身、不遵循指令等情况时，最好开始新聊天。
   - 他们还建议 *让模型给你一个已讨论要点的总结，并将其作为新聊天中 prompt 的一部分。或者依靠 ChatGPT 新的基于聊天的 memory 功能（如果已向你推送）*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1359986113124765809)** (57 条消息🔥🔥): 

> `Prompt engineering 资源，模型特定怪癖，MusicGPT 创建帮助，版权和 ToS 风险` 


- **Prompt Engineering 资源出现**：一位成员询问关于 prompt engineering 的可靠资源，侧重于新闻和技术，另一位成员回应时强调要理解 **你希望 AI 提供什么** 并向模型清晰地解释。
   - 他们还提到了 **验证输出** 的重要性，以及注意 **模型特定怪癖** 和公司政策。
- **模型怪癖需要亲身实践经验**：一位成员建议，理解模型特定的怪癖需要 **体验不同的模型**，并创建分层结构的 prompt 以观察每个模型如何处理它们。
   - 他们指出，这种方法可以培养 *模型直觉（model intuition）*，这是有机且定性的，需要持续的 prompting。
- **MusicGPT Prompt 陷入政策泥潭**：一位成员请求一个 *MusicGPT* 的 prompt，以协助处理音乐相关的请求并提供来自 **Genius.com** 的链接，这引发了关于频道侧重于“如何做”而非直接提供 prompt 的讨论。
   - 讨论转向了使用外部网站的复杂性，以及理解 **ToS 和使用政策** 以避免账号被停用的重要性。
- **Prompt 创建过程中出现版权担忧**：一位成员对使用外部网站和 IP 时的 **版权** 问题表示担忧，警告违反政策的风险，而另一位成员则认为仅仅链接到公开信息并不是深层次的问题。
   - 这导致了对用户关于 **音乐反应助手** 意图的澄清，以及关于是否有必要使用 **API** 的讨论。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1359986113124765809)** (57 条消息🔥🔥): 

> `Prompt Engineering 资源, MusicGPT 定制, MusicGPT 的 API 使用, ChatGPT 使用的政策合规性` 


- **讨论 Prompt Engineering 核心**：成员们讨论了 Prompt Engineering 的核心，强调要理解你希望 AI 做什么并清晰地传达，避免拼写和语法错误，并验证输出结果。
   - 讨论内容包括适应特定模型的特性、提供反馈以引导模型行为，以及检查 [ToS](https://openai.com/policies) 以避免账号问题。
- **构建 MusicGPT 助手**：一位用户请求帮助创建一个用于音乐相关查询的 MusicGPT 助手，寻求一个能提供来自 Genius 在线资源的 Prompt。
   - 建议包括从 Markdown 大纲开始、使用 ChatGPT 进行提示，以及探索现有的 Music API，但一位成员对复杂的 API 使用和政策合规性持谨慎态度。
- **针对模型细微差别的诊断性 Prompting**：成员们建议让模型解释 Prompt 或概念，以了解其理解情况并识别与其编程或安全训练相关的歧义或冲突。
   - 这种诊断方法有助于优化 Prompt，并确保模型按预期理解和响应，对于创意探索或 API 实现非常有用。
- **政策风险困扰公开 Prompt**：成员们提醒注意政策合规性，强调在要求模型使用外部网站和有版权的歌曲时，需要尊重版权和使用政策。
   - 忽视这些政策可能会导致账号被封禁，尤其是在创建与其他方内容交互的工具时。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1359996654014435371)** (119 条消息🔥🔥): 

> `LM Studio 中的 Prompt Preprocessor, LM Studio 中的 HuggingFace 登录, 使用 Gemma 3 生成图像, Quantization 与 QAT, 加载具有指定上下文限制的模型` 


- **Prompt Preprocessor：尚未公开的秘密武器**：一位用户询问了 LM Studio 中使用 Typescript 编写的 **Prompt Preprocessor**，以及退出代码 1 是否表示其不可用。
   - 一名团队成员回应称这是一个*尚未发布的秘密功能*，并告诉该用户：*你什么都没看见*。
- **Hugging Face 登录：不可能的任务**：一位用户询问如何在 LM Studio 中登录 **Hugging Face**，并指出缺乏相关文档。
   - 另一位用户直截了当地回答：*你做不到。*
- **Gemma 3 的图像生成：幻觉现场**：用户发现 **Gemma 3** 无法生成图像（尽管有传言称它可以），而是生成了虚假的 Imgur 链接。
   - 经澄清，**Gemma 3** 只能读取图像而不能生成图像，Google 的 Gemini 2.0 Flash experimental 和 2.5 Pro 可能具备图像生成能力。
- **QAT：Quantization 的奇特亲戚**：一位用户询问 **QAT** 是否是减少 RAM 占用的神奇方法。
   - 回复澄清说 **quantization** 是减少 RAM 使用的主要方法，而 **QAT** 是一种在量化形式下提高模型性能的训练方法。
- **Gemini-Pro 账单冲击：Google 的陷阱！**：一位用户在使用 **Gemini-Pro-2.5-exp** 模型后遭遇了账单冲击，这导致他们在没意识到会产生费用的情况下切换到了 **Gemini-Pro-2.5-preview**。
   - 该用户指出，巨大的 625k 上下文窗口花费了他们 **$150**，而如果使用带有 caching 功能的 **Sonnet** 成本会低得多。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1360004589734330579)** (115 messages🔥🔥): 

> `MLX distributor 修复, M3 Ultra 价值, Nvidia DGX 主板, Deepseek R1 Token 生成, M1 Ultra vs M4 Max` 


- **MLX 修复随机模型加载问题**：一位用户提到，一名开发者修复了 **MLX distributor** 以随机方式加载模型的问题，并暗示该开发者是一位拥有大量资源的“折腾者（tinkerer）”。
   - 随后讨论了该开发者配置的高昂成本，包括一个 **Max 芯片** 集群和一台配备 **512GB RAM** 的 **M3 Ultra**。
- **M3 Ultra 性能存疑**：一位用户发表了一个有争议的观点，认为 **M3 Ultra** 对于专业的 ML 和 LLM 工作来说不值这个价格，理由是初步测试显示，在 **MLX** 上运行 **Deepseek r1 67B Q8** 和 **Q6** 模型时，速度仅为 **10-13 tokens per second**。
   - 他们认为，配备 **两颗 Xeon Gold** 处理器和 **1TB RAM** 的服务器能以更低的成本提供更好的性能，并质疑 **M3 Ultra** 在生产环境部署中的可扩展性。
- **Nvidia DGX 价格推测**：针对新款 **Nvidia DGX 主板** 的成本出现了各种猜测，该主板拥有约 **280GB VRAM** 并配有额外 GPU 插槽。
   - 共识是 Nvidia 的定价可能在 **$50,000** 左右，但与目前的配置相比，它可能提供一种运行大模型更廉价的方案。
- **Apple Silicon 的未来潜力**：一位用户推测，现有的 **Apple Silicon** 实现早于开源模型和本地推理的兴起，因此我们可能要到 **M5** 甚至 **M6** 才能看到它们的真正实力。
   - 据他们称，Apple 在近两年前砍掉 **M4 Ultra** 后才意识到机器学习市场的缺口，而扭转芯片设计并出货需要这么长的时间。
- **Exllama 提升 Token 速度**：一位用户报告了在 **Linux** 上使用 **exl2** 测试 **Exllama** 的情况，与使用 **gguf** 相比，token/s 提升了约 **50%**。
   - 这表明软件和参数的选择会显著影响性能，尤其是在内存检索时间方面。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1359966622538006529)** (76 messages🔥🔥): 

> `上下文中的记忆, Meta 错过 AI 周, OSS 发布, AI 安全社区, 新图像模型` 


- **RAG 与在上下文中存储记忆的对比**：成员们讨论了“记忆功能”究竟只是针对历史记录的 **RAG** (Retrieval-Augmented Generation) 还是更深层的东西；有人认为用户特定的上下文被存储并压缩了，并看到一个存储个人背景数据但不提取过去对话的原始版本。
   - 另一位成员表示：“我仍然不敢相信 Meta 竟然在今年最平静的一个 AI 周里踢了个乌龙球（错过了机会）。”
- **采用 MIT 许可证的新图像模型发布**：一个采用 **MIT 许可证** 的新图像模型发布，同时还有一个新的 **Moonshoot 模型**，尽管后者可能违反了 Llama 的条款。
   - 一位成员提供了 X 上关于这个新图像模型的帖子链接。([帖子链接](https://x.com/btibor91/status/1910599925286777027))
- **AI 安全社区的定性遭到批评**：一位成员批评了一篇文章的定性，该文章称 **GPT-4** 在测试两个月后才发现一些“危险能力”，并反驳称尽管更强大的开源权重模型已经面世两年，但并未发生任何剧变。
   - 他们链接到了 X 上表达类似观点的帖子 ([帖子链接](https://x.com/AtaeiMe/status/1910601934228029515))。
- **关于用于网络防御的高级模型的辩论**：成员们辩论了模型是否应该具备进行 **CTF**、寻找漏洞和黑客攻击系统的能力，有人认为这将使世界 *更加* 安全而非更危险。
   - 其他人指出，这也会增加攻击面，但防御端是更大的市场，而且你可以在未来部署更新之前先运行这些模型进行测试。
- **InternVL-3 多模态模型发布**：**OpenGVLab** 发布了 **InternVL-3**，这是一款结合了 InternViT 和 Qwen 的多模态模型，取得了令人印象深刻的结果，并链接到了一篇描述其训练方法的（目前无法访问的）论文。
   - 它似乎使用了 Qwen 许可证，普通版本还可以，类似于 Llama 但许可范围更广；一位成员发帖称：“NVDA 最近在开源许可证下搞出了很多酷炫的东西。”


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1360282537859088607)** (2 条消息): 

> `前 OpenAI 员工法庭之友陈述，Peter Wildeford 帖子` 


- **前 OpenAI 员工提交法庭之友陈述 (Amicus Brief)**：一篇 [TechCrunch 文章](https://techcrunch.com/2025/04/11/ex-openai-staff-file-amicus-brief-opposing-the-companys-for-profit-transition/) 报道称，**前 OpenAI 员工**提交了一份**法庭之友陈述**，反对公司向营利模式转型。
- **Peter Wildeford 推文浮出水面**：一名成员分享了 [Peter Wildeford 帖子](https://x.com/peterwildeford/status/1910718882655981619) 的链接。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359994999848042717)** (117 条消息🔥🔥): 

> `Claude 额度成本，高品味 LMSYS，Gemini 应用可用性，工具使用开源模型，MCP 工具调用` 


- **Claude 额度价格飙升**：一位用户开玩笑说使用 **Claude credits** 的成本增加了，暗示更改一个变量名就要花费 *$40*。
   - 该用户附带了一张图片，似乎在嘲讽价格上涨，并暗示需要更具成本效益的解决方案。
- **高品味的 LLM 对决？**：一名成员建议创建一个邀请制的 *"high taste lmsys"*，为特定个人提供免费且早期的模型访问权限。
   - 该想法是让实验室为批量统计提供免费的 API 额度，同时保持原始评分和 Prompt 私密，从而实现 *"文明的 LLM 战斗"*。
- **Gemini 应用困扰用户**：几位用户发现 **Gemini app** 非常难用，其中一人表示它很难引导且经常出错。
   - 他们更倾向于使用 **AI Studio**，因为它有更好的 Grounding 且免费，有人评价道 *"AI studio + grounding 效果好得多，而且是免费的，哈哈"*。
- **工具使用开源模型规范**：讨论探讨了什么样才算是一个好的工具使用（Tool Use）开源模型，并指出仅靠 **evalmaxing** 是不够的。
   - 有人建议模型处理不在数据集中的 API 的能力非常重要，并强调了编写 MCP 服务器的能力，尽管目前缺乏现成的 Evals。
- **MCP 工具调用集成**：有人提到将 **MCP tool calls** 集成到数据中对于构建一个好的函数调用（Function Calling）模型至关重要。
   - 模型处理 10 个以上工具的难度更大，考虑到 **Gemini 2.5 Pro** 目前在函数调用方面的表现不佳，建议与其展开竞争。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 条消息): 

philpax: https://fixvx.com/typedfemale/status/1910599582226272457
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1360288906133901432)** (7 条消息): 

> `Gemini 付费墙，Cooking AI` 


- **Gemini 付费墙阻碍窥见稀有 AI 生物**：一名成员分享了一个 [YouTube 链接](https://youtu.be/zzXyPGEtseI?si=cTW9fTaN2zBrxpQB)，展示了*稀有生物的一瞥*，但指出访问权限被锁定在 **Gemini 付费墙**之后。
   - 他们询问创作者在忙什么，另一名成员回答说*他正在再次“开火做饭”（cooking）*，为大众提炼核心概念。
- **Gemini 付费墙后的 AI 创作者预告新项目**：围绕一位 AI 创作者的作品展开了讨论，该作品目前处于 **Gemini 付费墙**之后，引发了对其最新动态的好奇。
   - 一名成员表示该创作者正在*再次打磨作品*，承诺为大众提供提炼后的核心概念，暗示即将推出一个面向更广泛受众的项目。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1360287997374890164)** (3 条消息): 

> `Amy Prbs 线程` 


- **Amy Prbs 发布了三个线程**：Amy Prbs 在 X 上发布了三条帖子，并在此频道中分享。
   - 链接分别为 [帖子 1](https://x.com/AmyPrb/status/1910356664403820552)、[帖子 2](https://x.com/AmyPrb/status/1910357180517175620) 和 [帖子 3](https://x.com/AmyPrb/status/1910359272279494845)。
- **满足 minItems 的第二个主题**：这是一个占位主题，以确保 `topicSummaries` 数组至少包含两个元素。


  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1360144839022215218)** (2 条消息): 

> `Gemini 2.5 Pro, API Overview, Grok 3, Perplexity Pro` 


- ****Gemini 2.5 Pro** 向 Pro 用户推出**: **Gemini 2.5 Pro** 现已在 Perplexity 上面向所有 Pro 用户开放，并可配合 Pro Search 使用。
   - 鼓励用户在指定频道分享其与 **Sonar**、**4o**、**Sonnet 3.7**、**R1** 和 **o3** 相比的性能反馈。
- **Perplexity 预告 **Grok 3** 集成**: Perplexity 宣布 **Grok 3** 即将支持 Perplexity Pro。
   - Aravind Srinivas 在 [X](https://x.com/AravSrinivas/status/1910444644892327996) 上发布了这一公告，并鼓励用户反馈他们的想法。
- **深入了解 Perplexity API**: Perplexity 联合创始人兼 CTO @denisyarats 于太平洋时间 4 月 24 日上午 11 点主持了 Perplexity API 的概览活动。
   - 注册的新 API 用户将通过[此链接](https://pplx.ai/api-overview)获得 **$50** 的免费 API 额度。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1359970633517109280)** (194 条消息🔥🔥): 

> `Gemini 2.5 Pro, Deep Research, Telegram Bot Official, Firebase Studio AI Builder, Perplexity Android App Security` 


- **PPLX 中的 Gemini 2.5 Pro 与原生应用对比**: 成员们讨论了 **Gemini 2.5 Pro** 的上下文和性能，指出由于网络搜索集成的限制，对于大多数模型而言，[原生 Gemini 应用](https://gemini.google.com)通常比 **Perplexity** 更好。
   - 一位用户表示：“我相信对于大多数模型，原生应用几乎总是更好”，而另一位用户则建议 [Google AI Studio](https://ai.google.dev/) 拥有更好的 UI、视频/音频上传和设置。
- **Deep Research 更新延迟归因于高成本**: 用户将 **Perplexity 的 Deep Research** 功能与 **ChatGPT** 进行了比较，强调 **ChatGPT** 整体更好，但运营成本更高。
   - 一位成员推测 **Perplexity** 移除 **GPT-4.5** 是因为成本过高，并建议将 **Grok 3 Deep Search** 作为目标，以实现性能与成本之间的平衡。
- **Google 凭借 Firebase Studio 瞄准 Cursor**: 讨论围绕 Google 的新项目 [Firebase Studio](https://www.bleepingcomputer.com/news/google/google-takes-on-cursor-with-firebase-studio-its-ai-builder-for-vibe-coding/) 展开，这是一个用于 vibe coding 的 **AI builder**。
   - 有推测认为 Google 可能会利用其财力收购像 Firebase Studio 这样的项目，一位用户开玩笑说：“开发者可能就是 Google 自己，Google 只是在媒体上展示其财力，收购自己的项目”。
- **Perplexity Android 应用存在安全漏洞**: 一位用户分享了一篇 [Dark Reading 文章](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app)，详细介绍了 **Perplexity Android 应用**中的 **11 个安全漏洞**，包括硬编码密钥和不安全的网络配置。
   - 另一位用户指出“其中一半漏洞听起来甚至与应用无关”，对此另一位用户解释了每个漏洞的含义，并确认该报告是真实的。
- **Pro 角色权限故障**: 用户讨论了订阅后无法获得 **Pro User Discord 角色**的问题，指出需要通过 **Perplexity** 设置中的链接退出并重新加入服务器。
   - 一些成员报告即使按照规定步骤操作也失败了，需要管理员协助才能获得 Pro 角色。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1360098612268241080)** (1 条消息): 

> `Republican voters, Perplexity AI Search` 


- **在 Perplexity 上探索共和党选民的观点**: 一位成员分享了一个关于“什么是共和党选民”的 [Perplexity AI Search](https://www.perplexity.ai/search/what-are-republican-voters-thi-dcNs8jo4RwWQ87LmonBRBA#0) 查询。
   - 链接之后没有提供额外的背景信息或讨论。
- **搜索查询后仅有有限的背景信息**: 分享的关于共和党选民的 [Perplexity AI Search](https://www.perplexity.ai/search/what-are-republican-voters-thi-dcNs8jo4RwWQ87LmonBRBA#0) 链接没有收到进一步的评论。
   - 讨论以链接结束，缺乏深入的分析或互动。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1360327976281047131)** (5 messages): 

> `Python/PyTorch 模型中的 CUDA，关于 CUDA 的 GTC 演讲，Custom ops 与 load inline` 


- **深入探讨 Python/PyTorch 模型中的 CUDA**：一位成员询问在 **Python/PyTorch** 模型中使用 **CUDA** 的优质参考资料。
   - 另一位成员分享了他们近期关于该主题的 **GTC talk** 链接，可在 [Google Slides](https://docs.google.com/presentation/d/1zusmhgYjBxSOJPJ-QVeTVJSlrMbhfpKN_q4eDB9sHxo/edit) 查看。
- **使用 Custom Ops 和 Load Inline 解决问题**：一位成员建议，在处理 CUDA 时，**custom ops** 和 **load inline** 应该能解决大部分问题。
   - 他们补充道，目前正在进行进一步的改进，特别是关于**减少编译时间**方面。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1360115434246115459)** (4 messages): 

> `Triton 初学者资源，AMD GPU 上的 FP8 支持，奥斯汀聚会` 


- **面向 Triton 新手的 GPU 编程**：一位具有 SWE 背景的成员向社区征求 Triton GPU 编程的入门资源，社区推荐了官方的 [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)。
- **AMD GPU 在 FP8 点积上失败？**：一位成员报告了 `LLVM ERROR: No match found in MFMA database` 错误，询问 Triton 是否不支持在带有 e4 的 AMD GPU 上进行 FP8 x FP8 -> FP32 的 `tl.dot` 操作。
   - 尚未收到回复。
- **奥斯汀 Triton 爱好者聚会！**：Triton 社区受邀参加 4 月 30 日在奥斯汀地区举行的聚会，注册地址为 [https://meetu.ps/e/NYlm0/qrnF8/i](https://meetu.ps/e/NYlm0/qrnF8/i)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1360076825958678723)** (4 messages): 

> `AOT Inductor, Libtorch C++, Torch.compile` 


- **AOT Inductor 无法针对训练进行优化**：一位用户询问是否可以利用 **AOT Inductor** 优化 Python 模型以用于训练，并随后在 C++ 中加载。
   - 另一位成员澄清说 **AOT Inductor** 不适合训练，并建议改用 `torch.compile`。
- **Torch.compile 的替代方案**：一位用户询问在模型通过 **Torchscript** 加载到 **Libtorch C++** 并在其中进行训练的场景下，是否有 `torch.compile` 的替代方案。
   - 回复暗示 `torch.compile` 可能不适用于该特定设置。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1360005479824363701)** (2 messages): 

> `AlexNet 源代码` 


- **重温经典：AlexNet 重新浮现**：2012 年原始的 **AlexNet 源代码** 已被挖掘出来，现已在 [GitHub](https://github.com/computerhistory/AlexNet-Source-Code) 上可用。
   - 成员们正经历着一波怀旧潮，其中一人回复了一个 "X3" 的 gif。
- **挖掘深度学习历史**：**AlexNet 源代码** 的公开为理解开启深度学习革命的架构提供了宝贵的资源。
   - 它允许研究人员和爱好者检查原始实现，并从这篇开创性的 **2012** 年论文所使用的技术中学习。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1360310105433837759)** (2 messages): 

> `Thunder Compute, GPU 虚拟化, C++ 分布式系统工程师` 


- **Thunder Compute 招聘 C++ 工程师**：Thunder Compute 是一家 **YC 孵化的初创公司**，正在招聘一名 **C++ 分布式系统工程师**，以增强其 API 层 GPU 虚拟化软件。
   - 该职位涉及应用 **GPU 编程** 和 **分布式系统** 的理论知识，以实现微秒级的性能提升。
- **申请 Thunder Compute**：欢迎具备所需技能的人员申请。
   - 联系邮箱：carl.peterson@thundercompute.com。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1359970753667272726)** (55 条消息🔥🔥): 

> `A100 FP32 core limitations, NCU assembly view for warp stalls, FADD instruction latency, Citadel microarchitecture papers, Microbenchmarking` 


- **A100 的 64 个 FP32 核心限制了并行性**：一个 A100 对于 4WS 只有 **64 个 FP32 核心**，限制了可以执行的并行浮点加法数量，[影响了性能](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)。
- **NCU 汇编视图揭示了 Warp Stall 的原因**：**NCU 汇编视图**可用于识别特定 SASS 指令处的 **warp stalls**，从而深入了解性能瓶颈。
   - 正如一位成员所说，*在给定的 SASS 指令处寻找 warp stalls，这应该能很好地告诉你发生了什么。*
- **FADD 指令因依赖链而停顿**：由于循环携带的依赖关系，线程/warp 中的每个 **FADD** 必须等待前一个执行完毕。
   - 这种依赖链导致每个 WS 的单个 warp 无法在每个周期发出指令，从而导致硬件利用率较低。
- **Citadel 的 Volta 论文仍是金标准**：Citadel 的论文 *Dissecting the Nvidia Volta GPU via Microbenchmarking* ([Volta paper](https://arxiv.org/pdf/1804.06826)) 被认为优于后来类似的论文。
   - 成员们一致认为 *后来的模仿论文在质量上无法达到 Volta/Turing 论文的水平*。
- **Microbenchmarking 揭示指令延迟**：Microbenchmarking 对于确定指令所需的周期数以及依赖关系如何影响指令时钟周期延迟非常有用，单精度加法指令在依赖执行和独立执行下分别显示为 4 个和 2 个周期。
   - 一个相关的 StackOverflow 问答提供了关于此主题的[额外背景](https://stackoverflow.com/q/79261161/10107454)。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1359981016936353852)** (41 条消息🔥): 

> `ROCm Profilers, MI300 vs H100, Runpod Clock Speeds, Runpod Profiling Issues, GPU Cloud Providers` 


- **ROCm Compute & Systems 模仿 Nsight**：成员们提到 **ROCm Compute** 和 **ROCm Systems** 与 Nsight profilers 类似，使用 `rocprof` 进行 profiling，并且提供可视化选项。
   - 一位用户指出，在 **MI300X** 上使用 **ROCm 6.2** 处理带有 `ominperf` 的 nt 布局时，这些工具的表现并不比 `rocblas` 好。
- **MI300X 在内存带宽方面难以与 H100 竞争**：一位用户发现，虽然 **MI300** 在理论参数上更快，但在实践中 **H100** 更快，除非纯粹测试传输速度，MI300 仅能达到理论带宽的约 **75%**。
   - 该用户还发现内存带宽成为 **fp16 gemm** 的限制因素很奇怪。
- **Runpod 实例被降频**：一位用户发现 Runpod 实例被设置为最低时钟频率，且无法使用 `rocm-smi` 更改，导致性能不佳。
   - 另一位用户确认 Runpod 的时钟频率波动很大，实际上称其为*一场骗局*。
- **Runpod 屏蔽 GPU Profiling**：用户报告称 Runpod 实例不允许进行 profiling，即使在 NVIDIA GPU 上也是如此，任何与 GPU 相关的命令都会返回 `ERROR: GPU[0] : Unable to set ....` 消息。
   - 一位用户建议检查内核接口以强制提高性能级别，但怀疑 Runpod 是否会允许这样做。
- **用户寻求推荐允许 Profiling 的云服务商**：在发现 Runpod 限制 GPU 时钟频率并屏蔽 profiling 后，一位用户请求推荐其他提供 AMD GPU 并允许 profiling 的云服务商。
   - 在现有对话中没有推荐具体的服务商。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1360262697882615808)** (8 条消息🔥): 

> `MI300X support, vLLM, SGLang, GemLite, AMD` 


- **MI300X 支持进入推理引擎**：成员们讨论了 **vLLM** 和 **SGLang** 等常用推理引擎对 **AMD MI300X** 的支持。
   - 一位成员正在 *尝试 AMD*，并提到 **GemLite** 可以与 vLLM 配合使用但需要测试，并链接到了 [mobicham 的推文](https://x.com/mobicham/status/1910703643264774377)。
- **Triton 发行版不支持 FP8 E4**：一位用户指出，**Triton** 的发行版本不支持 **fp8e4**，但支持 **fp8e5**。
   - 这可能会给某些应用带来问题。
- **vLLM 可在 MI210 上运行**：一位成员确认在工作中使用 **vLLM** 配合 **MI210**，暗示 **MI300** 应该也可以运行。
   - 他们澄清这需要自行编译，但并不太困难。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

felix456: 有人知道除了使用 OpenAI API websearch 之外，还有什么便宜或免费的替代方案吗？
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1360179121761685725)** (11 messages🔥): 

> `vectoradd, vectorsum, Modal runners, GPU Benchmarks, Leaderboard submissions` 


- **Vector Addition 基准测试大量涌现**：多个针对 `vectoradd` 排行榜的基准测试提交在包括 **L4**、**H100**、**A100** 和 **T4** 在内的各种 GPU 上取得成功，均使用了 **Modal runners**。
- **Vector Sum 试验成功**：在 **L4** GPU 上使用 **Modal runners** 的 `vectorsum` 排行榜基准测试提交已成功完成。
- **Modal Runners 证明了其可靠性**：所有提交的成功表明 **Modal runners** 在不同 GPU 配置下的基准测试和排行榜提交中具有很高的可靠性。
   - 每个提交都被分配了一个唯一的 ID（如 **3577**）以便于追踪。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1360181140534726737)** (22 messages🔥): 

> `MI300 Profiling, Kernel Development details, Team formation, Github link` 


- **提交平台承诺提供 MI300 Profiling 功能**：据 AMD 团队称，他们正计划为提交平台提供 **profiling 选项**。
   - 一名团队成员表示，他们承诺会提供帮助，但*不确定能否在发布当天完成，但希望在发布后不久就能通过 Discord/CLI 进行 profiling*。
- **鼓励注册，不强制进行 Kernel 开发**：官方表示人们应该尽早注册。
   - 一位成员指出，*并没有提交作品的强制要求，所以先注册再说*。
- **Kernel 开发细节浮出水面**：在注册过程中，表格会询问“Kernel Development”，这只是一个占位符。
   - 有人表示，如果你不确定，*直接填写占位符也没有问题*。
- **GitHub 链接填写指南**：参与者询问在提交表单中应该填写什么 GitHub 链接。
   - 建议是为此创建一个空的 GitHub 仓库，但如果你还不确定最终在哪里提交代码，最后直接推送到你填写的另一个远程仓库即可。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1359966607492776036)** (154 messages🔥🔥): 

> `MCP, Gemini API, Cursor bugs, Deepseek v3.1, usage based pricing` 


- **Cursor 按量计费模式说明**：启用按量计费（usage-based pricing）后，用户可以在超出套餐包含的额度后继续使用 **fast requests**，但在达到支出限额后将切换为 **slow requests**。
   - 一位成员确认了对 Cursor 按量计费模式的理解，并对澄清表示感谢。
- **DeepSeek v3.1 在实际使用中的评价**：一位成员分享道，尽管基准测试经常夸大模型能力，但在实际使用中 **DeepSeek v3.1** 感觉比 **v3** *更聪明一点*。
   - 他们表示，实际使用比基准测试更能衡量模型的性能。
- **Gemini API Key 出现间歇性停机**：一些用户报告其 **Gemini API Key** 持续出现 **404 错误**，而其他用户则报告 Gemini 对他们来说运行正常。
   - 一位用户提到他们在过去一小时内一直遇到这个问题。
- **PDF 读取：在 Cursor 中读取 PDF 需要 MCP server**：成员们讨论了在 IDE 中添加 PDF 的功能，指出在 Cursor 中读取 PDF 文件需要 **MCP**，因为 *LLM 目前还无法直接读取 PDF*。
   - 一位成员表示，目前应该有很多 **“将各种格式转换为 Markdown”的 MCP** 解决方案可用。
- **用户报告当达到上下文限制时，Cursor 会进入摘要模式的 Bug**：用户报告称，当单个聊天窗口过载时（不断在 Claude 3.7、Gemini 2.5 之间切换，然后尝试 Claude 3.5），Agent 最终会进入摘要模式。
   - 聊天会自动总结，点击“New Chat”会用摘要覆盖现有的标签页。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1359974573214269461)** (83 条消息🔥🔥): 

> `Schrödinger Bridges, DeepCoder 14B, KV Cache Distillation, AlphaProof, Math AIs` 


- **通过 IPM 扩展 Schrödinger Bridges**：最近的工作通过 **Riemannian and integral probability metrics (IPM)** 扩展了 **Schrödinger Bridges**，但其显式的、基于熵的特性可能不如像 **Stable Diffusion** 这样的 *implicit diffusion models* 受欢迎。
   - 它们基于路径的方法在视频、分子动力学和时间序列分析中可能对获得 *global view* 非常有用。
- **DeepCoder 14B 开源用于代码任务**：Agentica 和 Together AI 发布了 [DeepCoder-14B-Preview](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51)，这是一个基于 **Deepseek-R1-Distilled-Qwen-14B** 并使用分布式强化学习 (RL) 微调的 **code reasoning model**。
   - 它在 **LiveCodeBench 上实现了 60.6% 的 Pass@1 准确率**，仅凭 140 亿参数就达到了 **o3-mini-2025-01-031** 的性能水平。
- **KV Cache 蒸馏可能并不实用**：有提议称，可以在主 LLM 的 KV 值上蒸馏出一个更便宜、更快的模型来预处理 prompt。
   - 然而，这被认为 *likely impractical*，因为 **KV values 是高度模型特定的**，且较小的模型会使用更少的 Transformer blocks。
- **AlphaProof 正在将 RL 用于数学**：有人提到 [AlphaProof](https://www.youtube.com/watch?v=zzXyPGEtseI) 正在使用 **RL with Lean** 进行数学证明。
   - 成员们讨论了 AlphaProof 发现新颖数学解法的潜力。
- **Generalist Agents 比超天才数学 AI 更实用**：有人质疑 *super-genius math AIs* 是否像 **practical generalist agents** 那样有益。
   - 也有人担心会创造出 *hyper-autistic LLMs，它们擅长数学但在其他方面表现糟糕*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1359992901224108132)** (4 条消息): 

> `AWS Site Visit, nanotron/ultrascale-playbook` 


- **即将进行 AWS 实地考察**：一位成员宣布他们的班级即将进行 **AWS site visit**。
   - 他们链接到了将要复习的 [nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。
- **分享了 Ultrascale Playbook 链接**：分享了三个指向 beautiful.ai 上 **Ultrascale Playbook** 的链接：[link 1](https://www.beautiful.ai/player/-ON_kt4FnaDoTXWZPiNY)、[link 2](https://www.beautiful.ai/player/-ON_kwmBmoDct78l5GKJ) 和 [link 3](https://www.beautiful.ai/player/-ON_kzs-T3R8Tbp-DYbM)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1360012570836074496)** (4 条消息): 

> `Awkward Youtuber, rQJmDWB9Zwk, 6nJZopACRuQ` 


- **可疑的截图引发 YouTube 探索**：一位成员发布了一张截图以及两个 YouTube 链接：[rQJmDWB9Zwk](https://youtu.be/rQJmDWB9Zwk) 和 [6nJZopACRuQ](https://www.youtube.com/watch?v=6nJZopACRuQ)。
   - 暗示人们应该相信这张截图。
- **右边的 Youtuber 看起来很尴尬**：一位成员评论说，*右边的那个人看起来非常尴尬，就像他不想待在那里一样*。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1359970147565306149)** (66 messages🔥🔥): 

> `Enact Protocol, Semantic Tool Calling, A2A podcast, MCP sandboxing, MCP client integration` 


- ****Enact Protocol** 在 **A2A** 兴起之际引发讨论！**: 成员们讨论了 [Enact Protocol](https://github.com/EnactProtocol/encat-spec-and-tools) 的潜力以及 **A2A** 是否会使其过时，并认为 **Enact** 更多是与代码解释器（code interpreters）竞争，而非 **A2A**。
   - 有人提议 **Enact** 可以从集成具有 openapi 转换器和语义搜索的 Agent 框架中受益。
- ****Semantic Tool Calling** 有望彻底改变 **LLM** 效率**: 讨论强调了 **semantic tool calling** 是解决因向 **LLMs** 提供数百个工具而导致上下文过载的方案，即利用向量模型根据与任务的语义相似性来选择工具子集。
   - 这种方法允许应用传统的 **ML** 方法进行工具分析，例如通过聚类检测相似工具，以及对工具进行分组以进行重排序（reranking）。
- **关于 **A2A**、**MCP** 和 Agent 索引的播客发布**: 一位成员分享了一个 [播客片段](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr)，讨论了 **A2A** 的影响、**Google** 对 Agent 进行索引的可能性以及其他相关话题，并指出其与当前讨论的相关性。
   - 该播客旨在保持高水准且易于理解，以激发超越典型技术讨论的想法。
- **将 **Express 服务器** 与 **MCP** 集成面临挑战**: 一位成员希望通过 **MCP** 将其带有 REST 路由的 **Express server** 连接到 **Claude desktop**，并询问是否可行。
   - 一位成员回复称，必须使用 **MCP JSON-RPC spec** 进行集成。
- **GitHub 未识别 **Licenses****: 一位用户遇到了 **GitHub** 无法识别其 [repo](https://github.com/Vizioz/Teamwork-MCP) 中 license 文件的问题，导致 glama 服务器显示 "license - not found"。
   - 用户通过将 license 免责声明移动到另一个文件解决了该问题，以便 **GitHub** 能够正确检测到 license。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1360057888416075878)** (5 messages): 

> `MCP Protocol Validator Open Source, MCP Server Adoption Challenges, Cloud Hosted MCP Inspector, MatlabMCP - MATLAB Meets LLMs` 


- ****MCP Validator** 开源以促进 **Implementation** 一致性**: **MCP Protocol Validator** 已开源，通过提供全面的测试套件来弥合各种 MCP server 实现之间的差距，可在 [GitHub](https://github.com/Janix-ai/mcp-protocol-validator) 获取。
   - 该工具旨在确保实现方案符合 **2024-11-05** 和 **2025-03-26 MCP 版本** 的要求，并包含由 **Janix.ai** 开发的 **HTTP** 和 **STDIO** 传输的参考实现。
- **Cloud Inspector 与你的服务器对话**: 一个云端托管的 **MCP Inspector** 已发布，无需本地设置即可测试 **SSE** 和 **Streamable HTTP servers**，访问地址为 [inspect.mcp.garden](https://inspect.mcp.garden)。
   - 该平台还包含完整的聊天支持，允许用户直接与远程 **MCP servers** 交互；详见 [X 上的公告](https://x.com/ChrisLally/status/1910346662297452896)。
- ****MatlabMCP** 连接 **MATLAB** 与 **LLMs****: 展示了 **MatlabMCP**，这是一个连接 **MATLAB** 与 **LLMs** 的小型 **MCP**，可以有效处理较小的代码片段，可在 [GitNew](https://git.new/MatlabMCP) 获取。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1360056623459991592)** (31 messages🔥): 

> `Discord referral, Dyslexia, KL vs CE, Model size` 


- **用户通过 Max 的 AI 发现 Discord**: 一位新用户根据其朋友的 **GPT4.o 模型** 的推荐找到了这个 Discord 服务器。
- **用户受读写障碍困扰**: 一位用户在出现拼写错误（将 *just checking mate* 误写为 *maet*）后，为其读写障碍（dyslexia）表示抱歉。
- **Token 预测中的 KL 与 CE**: 一位用户报告了其模型中的重复问题，另一位用户建议在 **KL** loss 中加入 **CE**，但随后又建议如果数据是几何分布的，这可能是 *浪费时间*，应坚持使用 **KL**。
- **模型大小引发讨论**: 一位用户担心其模型大小，另一位用户回答道 *对于这个问题，200M 已经足够了。你的问题出在别处*，但随后提醒道 *顺便说一下，200M 很容易在 16k 样本上过拟合*。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1359982624856215854)** (32 messages🔥): 

> `Lambada Parity, RWKV vs Transformers, UTs and RWKV, Muon and Transformer Layers` 


- **RWKV 实现了 Lambada Parity！**：RWKV 架构在 **Lambada** 数据集上达到了对等水平（parity），匹配了其蒸馏来源模型 **Qwen2.5-7B-Instruct** 的性能，尽管 MMLU 性能较低。
   - 发言者指出，这种对等处于*统计误差范围*内。
- **关于 RWKV 对 UTs 表达能力的辩论**：成员们讨论了 **RWKV** 模型的表达能力（expressiveness）是否使其适用于 **Universal Transformers (UTs)**。
   - 一位成员表示，*仅仅因为 RWKV 层往往更具表达能力，并不意味着它们更适合 UTs*，并指出表达能力甚至可能更差。
- **关于使用 Muon 缩放 Transformer 线性层的见解**：一位成员观察到，在 Transformer 每个 block 的最后一个线性层上添加零初始化的可学习逐通道缩放（learnable per-channel scale）（方案 A），与对最后一层的权重矩阵进行零初始化（方案 B）相比，会导致主路径激活 RMS 的增长更慢。
   - 这一观察是使用 **Muon** 库得出的。
- **RWKV-7 论文亮点**：一位成员分享了来自 **RWKV-7 论文**的一张图片，认为这是发送给研究 UT 的人员的好选择。
   - 文中解释说，该模型的数学保证允许根据应用范围将结果从平滑系统扩展到非平滑系统。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1360023449317412904)** (2 messages): 

> `GPTs Agents, String Matching` 


- **字符串匹配令用户沮丧**：一位成员在得知 **GPTs agents** 主要在全量数据集上使用字符串匹配（string matching）后表示失望。
   - 他们原本希望看到除了简单的**字符串匹配**之外，还有更复杂的学习或自适应机制。
- **字符串匹配受到质疑**：对话强调了对 **GPTs agents** 仅依赖字符串匹配所带来的局限性的担忧。
   - 这种方法可能无法捕捉到更先进技术所能提供的细微差别和复杂性，从而导致潜在的性能瓶颈。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1359967054706245765)** (55 messages🔥🔥): 

> `SIMD store, bench functions incorrect use, `@parameter` needed, lock files, random integers list` 


- **SIMD Store 需要特殊处理**：在对 Tensor 使用 **SIMD** 时，成员们澄清需要使用 [`store`](https://docs.modular.com/max/api/mojo/tensor/tensor/Tensor/#storeSIMD) 成员函数，而不是直接通过 `__setitem__` 赋值。
   - 这是因为 store 操作必须与标量操作区别对待。
- **基准测试函数需要 `@parameter`**：一位用户在使用 `benchmark.bench_function` 时遇到了错误提示 *cannot use a dynamic value in call parameter*。
   - 官方[澄清](https://github.com/modular/max/pull/4317#issuecomment-2795531326)指出，传递给 `benchmark.run` 的函数需要 `@parameter` 装饰器，并且预期不返回任何内容。
- **Magic Init 并不总是创建 Lock 文件**：一位用户注意到运行 `magic init AdventOfCode --format mojoproject` 并不总是创建 lock 文件。
   - 在运行 `magic run mojo --version` 后，lock 文件才被创建。
- **`__rand__` 用于 `&` 运算符，而非随机数**：成员们澄清 `__rand__` 是用于 `&` 运算符的，而不是用于生成随机数。
   - 虽然 Max Tensor 曾经有一个 `.rand` 方法（[文档](https://docs.modular.com/stable/max/api/mojo/tensor/tensor/Tensor/#rand)），但在 nightly 版本中已被移除；请改用 `random` 模块中的方法。
- **Tensor 缺少重载运算符**：一位用户质疑为什么 **Tensors** 没有重载 `+`、`-` 和 `matmul` 等运算符。
   - 这引发了关于 Mojo 中 Tensor 操作的设计选择和未来计划的讨论。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1360273839032111186)** (4 messages): 

> `Mojo 项目差异，magic.lock 文件问题，Mojo 版本冲突` 


- **Mojo 奇事：代码在一个项目中运行正常，在另一个项目中失败**：一位成员发现一段涉及 `@value struct Foo(StringableRaising)` 和 `String(foo)` 的特定代码片段在一个 **Mojo** 项目中可以运行，但在另一个项目中却抛出 *"no matching function in initialization"* 错误。
   - 报告的错误发生在尝试将自定义结构体 `Foo` 转换为 `String` 类型时。
- **Magic Lock 修复：删除文件解决问题**：该成员通过删除有问题项目中的 `magic.lock` 文件解决了错误。
   - 这表明问题很可能是由于 `magic.lock` 文件管理的 **Mojo** 版本不同或依赖冲突导致的，这意味着 *"可能拉取了不同的版本"*。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1359982933733019719)** (48 messages🔥): 

> `L1-Qwen-1.5B-Max 模型，Nomic embed text v1.5，LLM 查询日志，Embedding 模型的 System prompts，Re-ranker 模型` 


- **L1-Qwen-1.5B-Max 模型设置思考长度**：[L1-Qwen-1.5B-Max 模型](https://cmu-l3.github.io/l1/) 允许设置思考长度，一位成员发现即使不提示最大 token 数，它的表现也更好、更清晰，正如 [论文](https://cmu-l3.github.io/l1/) 中所解释的那样。
   - 用户准备下载 [HuggingFace 上的 L1 版本](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max) 来使用。
- **Nomic Embed Text 依然稳坐王座**：尽管尝试了许多生成式 **LLM**，一位成员仍继续使用 **Nomic** `nomic-embed-text-v1.5-Q8_0.gguf`。
   - 另一位成员询问如何识别自己拥有的版本，对此有人回复 *google ^^*，并链接了 [Nomic 的 HF 页面](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main)。
- **归档 LLM 响应被证明很有帮助**：一位用户一年多来一直在数据库中记录 **LLM 查询和响应**，发现这些过去的响应对于咨询目的（尤其是销售）非常有价值。
   - 他们创建了一个 **Emacs Lisp 函数** 来插入 embedding，参考了 [这里](https://gnu.support/files/tmp/clipboard-2025-04-11-09-03-07.html) 找到的一个函数。
- **关于 Embedding 使用 System Prompts 的讨论**：成员们讨论了 **LM-Studio/ALLM** 等 embedding 模型是否默认使用 **system prompts**，一位成员认为可能不会使用来自 LLM 的 system prompt。
   - 用户确认他们 **没有给 embedding 模型任何 system prompt**，也没有这样做的选项。
- **Reranking 模型引起关注**：一位成员询问 **re-ranker 模型** 的工作原理，以及是否只有向 LLM 提出的问题才重要，同时参考了一个关于前缀的 [YouTube 视频](https://www.youtube.com/watch?v=76EIC_RaDNw&feature=youtu.be)。
   - 链接的视频引发了关于在查询前添加 `search_document:CHUNK_OF_TEXT_FOLLOWS` 和 `search_query:FOLLOWED_BY_QUERY` 前缀的讨论，但指出所有 embedding 必须重新索引。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1359995523519614996)** (31 messages🔥): 

> `Gradio GUI，Transformer 训练数据量，寻找 Python 专家，报告 HF 课程错误，微调模型` 


- **Gradio 是一个 GUI 库**：一位成员询问 *什么是 Gradio*，另一位简明扼要地回答它是一个 **GUI 库**，并指向了 [Gradio 网站](https://www.gradio.app/)。
- **Transformer 会因为数据过多而过度训练吗？**：一位成员正在抓取一个包含 **100 万条手表记录** 的网站，并考虑训练一个 Transformer 以使其更好地理解上下文/规格名称，询问 *是否存在用过多数据训练 Transformer 的情况？*。
   - 该成员计划微调模型（可能是 **Mistral7B**），以便如果有人说 `Patek 2593 Tiffany stamp dirty dial manual wind`，它能理解这些词以及它们属于哪个实体。
- **Lightning AI Chat Templates 发布**：HuggingFace 团队宣布在 **HF** 上推出 [chat templates](https://lightning.ai/chat)。
- **在 ROCm 上本地运行 HF 模型**：想要在 **ROCm 上本地运行首发 Hugging Face 模型** 的用户可以查看 [这个视频](https://youtu.be/K4bHgaUk_18)。
- **重启 Xet Spaces 以修复问题**：拥有 **Xet** 早期访问权限且其 spaces 面临问题的用户应考虑重启它们。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1360322773217448229)** (3 messages): 

> `Life's Unexpected Surprises` 


- **应对生活变故的智慧**：在有人表示不理解一句谚语后，另一人解释说 *当生活给你带来意外惊喜时，你就会明白了*。
   - 第一位成员随后回复道 *我希望你正在经历的一切都能很快好起来*。
- **另一个话题**：另一个总结。
   - 另一个回复。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

not_lain: 该应用已离线
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1360184619160305786)** (1 messages): 

> `Stanford CME 295 Transformers, LLM Book Discussions` 


- **分享斯坦福 CME 295 Transformers 书籍**：一位成员分享了 [Stanford CME 295 Transformers book](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/tree/main) 的链接，并询问是否有人研究过其中的内容。
- **引发对 LLM 书籍讨论的兴趣**：分享 Stanford CME 295 Transformers 书籍链接引发了关于 Large Language Models (LLMs) 及相关教育材料的潜在讨论。
   - 成员们可能会深入探讨书中强调的模型架构、训练方法或实际应用等方面。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1360096434698850588)** (6 messages): 

> `Object Tracking, ReID for Object Recognition, Owlv2 Model, Segment Anything Model (SAM), YOLO Model` 


- **ReID 术语出现**：一位成员询问在不同摄像机帧中对同一物体进行 **object tracking**（目标跟踪）的术语。
   - 另一位成员回答说该术语是 **ReID**。
- **Owlv2 模型故障排除开始**：一位成员报告了用于图像引导检测的 **Owlv2 model** 的问题，指出其内置方法的表现差于预期，并发布了 [github tutorial](https://github.com/github) 的链接。
   - 他们请求协助重新配置该类，以更好地适应作为查询的裁剪图像。
- **SAM 来拯救 YOLO？**：一位成员建议使用 **Segment Anything Model (SAM)** 作为 **YOLO** 的替代方案来识别垂直电线杆，因为可以将 YOLO 的边界框输出提供给它。
   - 另一位成员承认使用 **SAM** 进行标注，但表示需要自动化，排除需要用户交互选择电线杆的情况，这可以通过对 SAM 进行 finetuning 来实现。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1360106808198562026)** (4 messages): 

> `LangGraph vs Google ADK, Google Agent Development Kit, Meta Llama access` 


- **Google ADK 对比 LangGraph：开源对峙**：成员们正在将 **Google Agent Development Kit** 与 **LangGraph** 进行比较，指出 Google 采用完全开源的方式，而 LangGraph 则是部分开源模型并带有商业调试工具。
   - 有人提到 **LangGraph** 致力于广泛的 LLM 兼容性，而 **ADK** 旨在与 **Google ecosystem** 紧密集成。
- **Meta Llama 访问请求被拒绝**：一位成员报告说他们访问 **Meta Llama models** 的请求被拒绝，并询问是否可以重试。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1359968556049109102)** (14 条消息🔥): 

> `vgel 的 control-vectors，DisTrO 详情，Psyche 的 testnet 运行` 


- **Control-Vectors 增强模型以用于特定用例**：一位成员询问是否可以使用 **vgel 的 control-vectors** 来增强模型以适应特定的用例和角色（如地城之主或软件工程师），认为这可以提高准确性和控制力，特别是对于像 **DeepHermes-Mistral-24B** 这样的开源模型。
   - 作为回应，另一位成员提到，虽然他们进行过实验，但普遍应用 control-vectors 已被证明是不稳定的，但目前仍在探索中，并指出了一篇 [相关的 X 帖子](https://x.com/winglian/status/1910430245854773523)。
- **DisTrO 的技术报告细节仍难以寻觅**：一位成员询问了关于 [distro.nousresearch.com](https://distro.nousresearch.com/) 上 **DisTrO** 运行的技术报告详情，寻求有关数据集、GPU/参与者数量以及 Benchmark 详情（例如评测中使用的 shot 数量）的信息。
   - 另一位成员回答说，他们没有发布技术报告，并表示这次运行主要是为了证明 **DisTrO** 可以在互联网上运行，他们并没有针对最终模型的质量进行优化，仅在有限数量的 Token **(100B)** 上进行了简短的训练。
- **Psyche 的 Testnet 运行承诺**：在 DisTrO 对话的后续中，一位成员分享了分布式训练的细节，指出每个节点拥有 **8xH100s**，并且有 **8-14 个节点**在运行，还提到评测代码已在 [GitHub](https://github.com/PsycheFoundation/psyche/tree/main/shared/eval/src) 上发布。
   - 他们正在为 **Psyche** 开发 **testnet 运行**，这是他们利用 **DisTrO** 构建的分布式训练网络，届时将包括显著的速度和带宽提升，并公开数据集、节点等信息的可见性。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1359972820284866782)** (4 条消息): 

> `Azure API, Reasoning Content, Token 限制` 


- **Azure API 现在可以工作了！**：一位成员报告说 **Azure API** 现在可以工作了，但他们不确定为什么之前不行。
   - 他们注意到 `<think>` 追踪记录在 `reasoning_content` 中返回，并建议*这应该被记录在文档中，因为每个 API 的实现都略有不同*。
- **Azure 中出现 Token 限制错误**：一位成员在 **Azure API** 中请求过多 Token 时收到了 **400 错误**。
   - 他们还建议 `<think>` 标签可能只在响应被 Token 限制截断时出现，这解释了为什么他们得到了格式错误的追踪记录。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1360292371883102269)** (3 条消息): 

> `X 帖子，Teknium 用户提及` 


- **分享了 X 帖子**：一位成员分享了一个 X 帖子的链接：[https://x.com/omarsar0/status/1910004370864742757?t=w_ps1fBHQpu3Vfdf1MMV0A&s=19](https://x.com/omarsar0/status/1910004370864742757?t=w_ps1fBHQpu3Vfdf1MMV0A&s=19)。
- **Teknium 提及用户**：用户 **Teknium** 提及了 2 名用户。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1360172027213779126)** (6 messages): 

> `Pathways 论文, TPU vs GPU, Tinygrad cloud, Tinygrad 虚拟化` 


- **Pathways 论文引发讨论**：一名成员分享了 [Pathways 论文](https://arxiv.org/pdf/2203.12533)，指出 *PATHWAYS 使用客户端-服务器架构，允许 PATHWAYS 运行时代表多个客户端在系统管理的计算岛上执行程序*，并建议建立 **tinygrad cloud**。
   - 该成员还指出，*tinygrad 是单进程的，即使在扩展（scale-out）时也将保持这种方式*。
- **TPU Kernel 比 GPU 驱动更丰富？**：一名成员引用 [Pathways 论文](https://arxiv.org/pdf/2203.12533)称：*TPU 和 GPU 之间最大的区别在于，更长运行时间和更复杂的计算可以融合到单个 TPU kernel 中，因为 TPU 支持丰富的控制流和通信原语，而这些在 GPU 系统上必须由驱动代码执行*。
   - 该成员反驳道，想象一下 *1024 颗 Navi 48 芯片在没有驱动程序的情况下协同工作*。
- **Tinygrad 旨在虚拟化 GPU**：一名成员阅读了 Pathways 论文并总结道，它从根本上是一种**编排方法**。
   - 他们声称，*如果 tinygrad 能够虚拟化 GPU，从而保证一定的资源使用量，那将更具创新性*。
- **Tinygrad Termux 问题**：一名成员在提交了 [此 issue](https://github.com/tinygrad/tinygrad/issues/9687) 后，询问另一名成员是否成功在 **termux** 下运行 **tinygrad**。
   - 该用户提到他们也遇到了同样的问题，提示 *libgcc_s.so.1 not found*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1360046129223565323)** (14 messages🔥): 

> `位置无关代码, ELF 加载器, 编译器链接, TinyGrad 架构, 内存映射生成` 


- **TinyGrad 利用位置无关代码 (PIC)**：讨论明确了 **TinyGrad** 使用**位置无关代码 (PIC)**，其中地址相对于程序计数器（program counter），并且指向 `.data` 和 `.rodata` 段的地址会被修补，以适应加载时的内存布局。
   - 目标是合并 `.text` 和 `.data` 段，并为数据段的正确偏移量修补地址。*一个有趣的练习是不使用操作系统，让 TinyGrad 直接延伸到硬件。*
- **ELF 加载器用于共享对象**：TinyGrad 中的 **ELF 加载器**既用于在 AMD/NV 中加载共享对象（`.so/.dll`），也用于将来自 **Clang/LLVM** 的目标文件（`.o`）转换为扁平的 shellcode。
   - 加载共享对象时，假设使用 **PIC**，从 `.text` 到 `.data` 的偏移量是已知的，不需要重定位；然而，目标文件（`.o`）需要重定位，因为偏移量是由链接器填充的。
- **Cloudflare 的博文解释了目标文件执行**：一名成员分享了 [Cloudflare 的系列博文](https://blog.cloudflare.com/how-to-execute-an-object-file-part-1/)，其中描述了如何执行目标文件，这与 TinyGrad 的方法类似。
   - 该系列博文解释了将目标文件转换为扁平 shellcode 的过程。
- **由于全局变量，LLVM 从 `.data` 加载**：在 **Clang JIT** 中需要使用 **ELF 重定位**，因为尽管 TinyGrad 不使用全局变量，**LLVM** 有时仍会选择从 `.data` 加载，而不是为常量使用立即数（immediate values）。
   - 这种行为使得在链接过程中必须修补地址以获得正确的偏移量。
- **为什么 TinyGrad 中不进行编译器链接**：曾考虑过在编译期间进行链接，但成员提到由于链接速度较慢，且 Apple 的链接器存在无法输出到 stdout 的 bug，因此避免了这一步。
   - 跳过链接步骤在 `elf.py` 中节省了十几行代码。


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1360049247197593810)** (1 messages): 

> `微调 Llama4, Scout 模型, Maverick 模型, MoE 模型` 


- **torchtune 支持 Llama4 微调**：torchtune 已支持 **Llama4** 的全量微调。
   - 配置可在 [此处](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4) 获取；请关注后续的 LoRA 配置、改进的多模态支持以及性能优化。
- **引入 Scout 模型**：**Scout** 模型（**17B x 16E**，共 **109B** 参数）可以在单节点上进行微调，或者在支持 **2D 并行**（**TP + FSDP**）的多节点上进行微调。
   - *GPU 中产阶级*的成员们可以欢呼了。
- **引入 Maverick 模型**：**Maverick** 模型（**17B x 128E**，约 **400B** 参数）可用于全量微调，需要多个节点。
   - 这些是首批 **MoE 模型**，欢迎尝试并提供反馈。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: @here 43 分钟后开始答疑时间 (office hours)！
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1359966190478430339)** (16 messages🔥): 

> `running_loss.detach() fix, test tolerances, sampler seed, bitsandbytes Mac issues, FSDPModule import error` 


- **running_loss.detach() 修复即将发布**：一名成员建议使用 `running_loss.detach()` 来简单修复一个未知问题，另一名成员表示 *接受该方案*。
   - 该修复位于 `detach` 分支中，但不要忘记为其他 recipes 进行修复。
- **应降低测试容差 (Test tolerances)**：一名成员建议，当种子 (seed) 固定后，所有单元测试的容差都可以从目前的 +-0.01 降低。
   - 提到了过去一个涉及集成测试中容差过松的问题，并链接了一个[相关的 pull request](https://github.com/pytorch/torchtune/pull/2367)。
- **bitsandbytes Mac 平台问题**：一名成员报告称，由于 `bitsandbytes>=0.43.0` 没有为其他平台提供二进制文件，导致在 Mac 上执行 `pip install -e '.[dev]` 失败，并建议将版本更改为 `bitsandbytes>=0.42.0` 作为变通方案。
   - 该变通方案引用了[此 issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180)，其中指出 0.42 之前的版本标签存在错误。
- **FSDPModule 导入错误正拖慢测试进度**：`pytest tests` 在收集测试时失败，报错为 `ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'`。
   - 建议检查安装文档，因为该项目需要不同的安装方法，且团队目前不想添加特定于平台的依赖要求。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

krammnic: 我当时说的是类似这样的东西
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1360110036415807639)** (18 messages🔥): 

> `FunctionCallingAgent JSON Schema Response, Llama Cloud API 404 Error, FaissVectorStore Index from Weights, Intelligent Metadata Filtering in RAG Agent` 


- **FunctionCallingAgent 需要 OpenAI 的 JSON 响应**：一名成员希望使用 **FunctionCallingAgent** 生成特定 **JSON schema** 的响应，并询问如何使用 **OpenAI 的结构化响应 (structured response)** 功能。
   - 另一名成员回答说，结构化输出本质上就是工具调用 (tool calls)，这使得混合工具调用和结构化输出变得困难；他们建议添加一个作为响应类的工具，并设置 `tool_choice="required"`。
- **Llama Cloud API 抛出 404 错误**：一名成员在使用 **Llama Cloud API** 的快速模式从文档中提取值时遇到 **404 错误**，API URL 为 `https://api.cloud.llamaindex.ai/v1/extract`。
   - 有人指出所使用的 API 端点不存在，并引导该成员查看[正确的 API 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started/api)和 [API 参考](https://docs.cloud.llamaindex.ai/API/create-extraction-agent-api-v-1-extraction-extraction-agents-postsadguru_.)。
- **关于从权重恢复 FaissVectorStore 索引的查询**：一名成员尝试使用从权重恢复的 **FaissVectorStore** 来创建一个可以查询的 **VectorStoreIndex**。
   - 有人指出 [Faiss 文档](https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/)展示了如何初始化，尽管示例是 Python 而非 Typescript。
- **寻求在 RAG Agent 中实现智能元数据过滤**：一名成员正尝试构建一个 Agent，能够根据用户查询在检索时进行智能元数据过滤。
   - 片段中未提供直接解决方案，但该成员正在寻求关于在标准 **RAG pipeline** 中实现此用例的建议，且无需在后续 API 调用时重新创建嵌入 (embeddings)。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1360003130364002449)** (7 条消息): 

> `NotebookLM 中的麦克风识别问题、上传源错误、网络钓鱼尝试` 


- **NotebookLM 无法识别麦克风**：一位用户报告称，尽管麦克风工作正常，但 **NotebookLM** 在交互模式下无法识别电脑的默认麦克风。
   - 另一位用户建议检查 **OS**（操作系统）和**浏览器权限**，并建议先在不连接外部 USB 设备的情况下进行测试。
- **用户遇到上传源错误**：一位用户询问在 **NotebookLM** 中上传源上显示的**红色“!”标志**，即使 **PDF 文件**小于 **500kb** 也会出现。
   - 另一位用户建议将鼠标悬停在“!”标志上，并指出该源可能为空或加载需要时间，尤其是在处理某些特定网站时。
- **Steam 网络钓鱼尝试流传**：一位用户分享了一个看似 **$50 礼品**的链接，但它是一个[钓鱼链接](https://steamconmmunity.cfd/1043941064)，会重定向到一个虚假的 **Steam Community** 网站。
   - 警告用户不要点击可疑链接，并核实要求输入登录凭据的网站 URL。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1360218077664378980)** (2 条消息): 

> `模糊的问题、具体查询` 


- **问题的模糊程度达到新高度**：一位成员开玩笑地评论了另一个人的问题“有人开过车吗”，并建议他们在查询时更加具体。
- **关于具体性的建议引发幽默**：该成员问道，“你还能再模糊点吗？”，以此强调最初问题的荒谬性。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1360150375637844039)** (2 条消息): 

> `Java API、网络错误` 


- **Cohere 的 Java API 抛出网络错误**：一位成员报告在使用 [Java API 示例](https://docs.cohere.com/reference/about#java)时遇到了 `Network error executing HTTP request`。
   - 该成员确认，在不同的提示词下（例如“为初学者厨师推荐快速餐点”）该错误依然存在。
- **请求 Java API 代码片段**：一位成员请求提供代码片段，以帮助调试 Java API 中的 `Network error`。
   - 该成员还询问用户是否是在原封不动地运行示例代码。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1360186605733937184)** (2 条消息): 

> `将 DSPy 模块作为 Persona、AI Agents 与推理、大语言模型 (LLMs)、机器学习框架、基础设施` 


- **基于模块的 Persona 引发关注**：一位成员询问关于将 **DSPy 模块**训练为 **Persona**、优化“Agent/模型”的系统提示词，并将此模块作为输入传递给其他模块以生成符合该 Persona 内容的问题。
- **协作邀请突显工具集**：一位成员表达了协作意向，列举了在 **AI Agents & Reasoning**（**LangChain**、**LangGraph**、**ElizaOS**、**AutoGPT**、**ReAct** 框架）、**Large Language Models**（包括 **GPT-4.5**、**DeepSeek-R1**、**Claude 3.5**）以及 **PyTorch** 和 **TensorFlow** 等**机器学习框架**方面的专业知识。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1360012937644867856)** (2 条消息): 

> `课程截止日期、证书获取` 


- **起步晚是否仍能完成课程？**：一位学生询问，尽管在官方开始日期之后才开始，是否仍有可能完成课程并获得证书。
   - 另一位成员给出了肯定的回答，并引导学生访问[课程网站](https://llmagents-learning.org/sp25)获取所有必要的材料和截止日期信息。
- **LLM Agents 课程**：一位学生询问他们是否能在截止日期前完成课程并获得证书。
   - 一位成员确认所有材料都可以在[课程网站](https://llmagents-learning.org/sp25)上找到。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1360310859188142143)** (1 条消息): 

> `活动提醒` 


- **活动提醒**：一位成员提醒大家**明天有一个活动**。
   - 他们表示希望能在那儿见到其他成员。
- **活动就在明天**：这只是一个提醒。
   - 不见不散。


  

---


---


---


{% else %}


> 完整的频道逐项细分内容已针对邮件进行截断。
> 
> 如果你想查看完整细分，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}