---
companies:
- openai
- qwen
- deepseek-ai
- microsoft
- kyutai-labs
- perplexity-ai
- together-ai
- meta-ai-fair
- google-deepmind
- hugging-face
- google
- anthropic
date: '2024-09-20T01:00:56.202964Z'
description: '以下是该文本的中文翻译：


  **OpenAI 的 o1-preview 和 o1-mini 模型**在数学、高难度提示词（Hard Prompts）和编程基准测试中处于领先地位。**通义千问
  Qwen 2.5 72B** 模型表现强劲，性能直逼 **GPT-4o**。**DeepSeek-V2.5** 在中文大语言模型中位居榜首，足以与 **GPT-4-Turbo-2024-04-09**
  媲美。**微软的 GRIN MoE** 以 66 亿激活参数实现了出色的效果。来自 Kyutai 实验室的 **Moshi 语音模型**可在搭载 Apple Silicon
  的 Mac 上本地运行。**Perplexity 应用**推出了带有“按住说话”功能的语音模式。Together.ai 推出的 **LlamaCoder** 利用
  **Llama 3.1 405B** 进行应用生成。**Google DeepMind 的 Veo** 是一款面向 YouTube Shorts 的新型生成式视频模型。**2024
  ARC-AGI 竞赛**增加了奖金，并计划开展大学巡演。一份关于**模型合并（model merging）**的综述涵盖了 50 多篇关于大语言模型对齐的论文。**Kolmogorov–Arnold
  Transformer (KAT)** 论文提议用 KAN 层替换 MLP 层，以获得更好的表达能力。**Hugging Face Hub** 与 **Google
  Cloud Vertex AI Model Garden** 集成，使开源模型的部署更加便捷。**Agent.ai** 作为 AI 智能体的专业网络正式推出。*“回归现实（接触大自然）才是你所需要的。”*'
id: a35e3928-79cc-409b-be9a-581b57c2d5e5
models:
- o1-preview
- o1-mini
- qwen-2.5
- gpt-4o
- deepseek-v2.5
- gpt-4-turbo-2024-04-09
- grin
- llama-3-1-405b
- veo
- kat
original_slug: ainews-not-much-happened-today-7878
people:
- hyung-won-chung
- noam-brown
- bindureddy
- akhaliq
- karpathy
- aravsrinivas
- fchollet
- cwolferesearch
- philschmid
- labenz
- ylecun
title: 今天没发生什么事。
topics:
- benchmarking
- math
- coding
- instruction-following
- model-merging
- model-expressiveness
- moe
- voice
- voice-models
- generative-video
- competition
- open-source
- model-deployment
- ai-agents
---

<!-- buttondown-editor-mode: plaintext -->**回归自然 (touching grass) 就是你所需要的一切。**

> 2024/9/18-2024/9/19 的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord (**221** 个频道，以及 **2506** 条消息)。预计节省阅读时间（按 200wpm 计算）：**303 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 参与 AINews 讨论！

在经历了[忙碌的昨天](https://buttondown.email/ainews/archive/)之后，AI 社区稍作休整。

如果你感兴趣，可以查看 Strawberry 团队成员 [Hyung Won Chung](https://x.com/hwchung27/status/1836842717302943774) 和 [Noam Brown](https://www.youtube.com/watch?v=eaAonE58sLU)（他现在正在 [招聘 multi-agent 研究员](https://x.com/polynoamial/status/1836872735668195636)）的新演讲，以及 [The Information](https://x.com/amir/status/1836782911250735126?s=46O) 和 [@Teortaxes](https://x.com/teortaxesTex/status/1836801962253402522) 中关于 o1 底层机制的简要评论。Nous Research 昨天宣布了 Forge，这是他们对 [开源 o1 复现](https://x.com/swyx/status/1836605035201073183) 的尝试。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与基准测试**

- **OpenAI 的 o1 模型**：[@lmsysorg](https://twitter.com/lmsysorg/status/1836443278033719631) 宣布 OpenAI 的 o1-preview 和 o1-mini 模型现已登录 Chatbot Arena。o1-preview 在各项榜单中均排名第一，尤其是在 Math（数学）、Hard Prompts（硬核提示词）和 Coding（编程）领域；而 o1-mini 在技术领域排名第一，总榜排名第二。

- **Qwen 2.5 模型**：Qwen 2.5 模型正式发布，[@bindureddy](https://twitter.com/bindureddy/status/1836502122529198304) 指出 72B 版本取得了优异的成绩，在某些基准测试中仅略低于 GPT-4o。该系列模型在知识、Coding 技能、数学能力和指令遵循方面均表现出显著提升。

- **DeepSeek-V2.5**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1836388149700043156) 报告称，DeepSeek-V2.5 在 LMSYS Chatbot Arena 中位列中文 LLM 第一，超越了部分闭源模型，并与 GPT-4-Turbo-2024-04-09 表现相当。

- **Microsoft 的 GRIN MoE**：[@_akhaliq](https://twitter.com/_akhaliq/status/1836544678742659242) 分享了 Microsoft 发布的 GRIN (Gradient-INformed MoE)，该模型仅凭 6.6B 的激活参数就在多样化任务中实现了良好的性能。

**AI 工具与应用**

- **Moshi 语音模型**：[@karpathy](https://twitter.com/karpathy/status/1836476796738670918) 重点介绍了 Moshi，这是来自 Kyutai Labs 的对话式 AI 音频模型。它可以在 Apple Silicon Macs 上本地运行，并在交互中展现出独特的个性特征。

- **Perplexity 应用**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1836480634514272750) 建议尝试 Perplexity 应用中的语音模式，该模式提供 Push-to-talk（一键对讲）功能和快速的答案流式传输。

- **LlamaCoder**：[@AIatMeta](https://twitter.com/AIatMeta/status/1836436439032303740) 宣布了 LlamaCoder，这是一个由 Together.ai 使用 Llama 3.1 405B 构建的开源 Web 应用，可以根据提示词生成整个应用程序。

- **Google 的 Veo**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1836448991774474561) 介绍了 Veo，这是他们最先进的生成式视频模型，即将登陆 YouTube Shorts 以帮助创作者将创意变为现实。

**AI 研究与开发**

- **ARC-AGI 竞赛**：[@fchollet](https://twitter.com/fchollet/status/1836517273500291079) 提供了 2024 年 ARC-AGI 竞赛的最新进展，宣布增加了奖金并计划进行大学巡演。

- **模型合并综述**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1836466166753087531) 发表了一篇关于 Model Merging（模型合并）的长篇综述，涵盖了从 20 世纪 90 年代到近期 LLM Alignment（对齐）应用的 50 多篇论文。

- **Kolmogorov–Arnold Transformer (KAT)**：一篇新论文介绍了 KAT，它将 MLP 层替换为 Kolmogorov-Arnold Network (KAN) 层，以增强模型的表达能力和性能。

**AI 行业与商业**

- **Hugging Face 与 Google Cloud 集成**：[@_philschmid](https://twitter.com/_philschmid/status/1836470169217998911) 宣布 Hugging Face Hub 现在更原生化地集成到了 Google Cloud Vertex AI Model Garden 中，从而可以更轻松地浏览和部署开源模型。

- **AI Agent 平台**：[@labenz](https://twitter.com/labenz/status/1836521094691373563) 讨论了 Agent.ai，它被描述为“AI Agent 的专业网络”，旨在提供有关 AI Agent 能力和专业化领域的信息。

**AI 伦理与社会影响**

- **偏见放大**：[@ylecun](https://twitter.com/ylecun/status/1836550110701879431) 评论了 AI 为了政治利益而放大偏见（Prejudice Amplification）的潜在可能性。

- **编程工作的未来**：[@svpino](https://twitter.com/svpino/status/1836404316250476951) 认为，未来那些主要技能仅为编写代码的人可能难以维持就业，并强调了掌握更广泛技能的必要性。

**迷因与幽默**

- [@vikhyatk](https://twitter.com/vikhyatk/status/1836523518424682579) 分享了一个关于尝试“State of the Art”（最先进）模型的迷因。

- [@abacaj](https://twitter.com/abacaj/status/1836522139651813860) 开玩笑说自己在 AI 发展方面处于领先地位。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Moshi：开源端到端语音对语音模型**

- **[Moshi v0.1 Release - a Kyutai Collection](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)** ([Score: 66, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1fjv1uc/moshi_v01_release_a_kyutai_collection/)): **Kyutai Labs** 发布了 **Moshi v0.1**，这是一个开源的 **speech-to-speech model**，作为其 Kyutai Collection 的一部分。该模型在 **3,000 小时**的语音数据上进行了训练，可以执行**声音转换**和**语音增强**任务，并已在 [GitHub](https://github.com/kyutai/moshi) 上发布，同时提供了预训练权重和 Demo。
  - 用户对该发布表示兴奋，并注意到在模型发布的同时还提供了一份 **paper**。**Moshiko** 和 **Moshika** 变体被澄清为分别针对**男性**和**女性合成语音**进行微调的版本。
  - 一位用户报告称，在 **4090 GPU** 上具有低延迟和高效性能，**利用率约为 40-50%**，**功耗约为 130W**。他们建议通过**原生 FP8 activations** 以及集成到视频游戏中进行潜在改进。
  - 该模型的 **MMLU score** 被指出略低于 **Llama 2 13B**，希望在非量化版本中能有更好的表现。一位用户询问了在**搭载 MLX 的 MacBook** 上运行该模型的情况，并报告了输出方面的问题。

- **Kyutai Labs open source Moshi (end-to-end speech to speech LM) with optimised inference codebase in Candle (rust), PyTorch & MLX** ([Score: 36, Comments: 2](https://reddit.com//r/LocalLLaMA/comments/1fjwc4l/kyutai_labs_open_source_moshi_endtoend_speech_to/)): **Kyutai Labs** 开源了 **Moshi**（一个 **7.6B parameter** 的端到端 speech-to-speech 基座模型）以及 **Mimi**（一个顶尖的流式语音编解码器）。此次发布包括在合成数据上微调的 **Moshiko** 和 **Moshika** 模型，推理代码库支持 **Rust (Candle)**、**PyTorch** 和 **MLX**，并在 [GitHub](https://github.com/kyutai-labs/moshi) 上以 **Apache license** 授权。Moshi 处理两条音频流，理论延迟为 **160ms**（在 **L4 GPU** 上实际为 **200ms**），使用一个小型的 **Depth Transformer** 处理 codebook 依赖，以及一个大型的 **7B parameter Temporal Transformer** 处理时间依赖，并可以在各种硬件配置上运行，根据精度的不同，**VRAM requirements** 在 **4GB** 到 **16GB** 之间。


**Theme 2. LLM Quantization: Balancing Model Size and Performance**

- **Llama 8B in... BITNETS!!!** ([Score: 75, Comments: 27](https://reddit.com//r/LocalLLaMA/comments/1fjtm86/llama_8b_in_bitnets/)): **Llama 3.1 8B** 已使用 **HuggingFace** 的极端量化技术转换为 **bitnet** 等效模型，实现了 **每权重 1.58 bits**。据报告，所得模型的性能与 **Llama 1** 和 **Llama 2** 相当，在保持有效性的同时实现了显著的压缩。有关此转换过程及其影响的更多详细信息，请参阅 [HuggingFace 博客文章](https://huggingface.co/blog/1_58_llm_extreme_quantization)。
  - 用户赞赏博客文章中关于失败尝试的**透明度**，并指出这在机器学习论文中通常是缺失的。有人呼吁应有更多激励措施来发表“此路不通”的研究，以提高该领域的效率。
  - 该转换过程并非从零开始对 **Llama 3** 进行 **bitnet** 训练，而是转换后的一种微调形式。为了让 **bitnet** 真正有效，模型需要在开始时就考虑到 bitnet 进行预训练。
  - **perplexity**（困惑度）的变化与量化到类似 **bits per weight (BPW)** 的情况没有显著差异。然而，这一转换过程仍被视为一项技术壮举，并可能在未来改进最小化困惑度变化方面发挥作用。

- **哪个更好？高量化的大模型还是高精度的小模型** ([Score: 53, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fjo7zx/which_is_better_large_model_with_higher_quant_vs/))：该帖子比较了**大参数量化模型**与**小参数高精度模型**的性能，具体提到了 **gemma2:27b-instruct-q4_K_S (16GB)** 和 **gemma2:9b-instruct-fp16 (16GB)** 作为案例。作者承认习惯于选择高精度的小模型，但质疑这种方法是否最优，并寻求社区关于不同模型配置的偏好和经验建议。
  - **大参数量化模型**通常优于高精度的小模型，正如一份[比较量化与 Perplexity 的图表](https://www.reddit.com/r/LocalLLaMA/comments/1441jnr/k_quantization_vs_perplexity/)所示。由于拥有更多的内部 Token 关系表示，4-bit 量化的 **70B 模型**通常优于全精度的 **8B 模型**。
  - 一位用户在 Ollama 上比较了 **Gemma2 27B 和 9B** 模型的各种量化版本，并提供了[基准测试结果](https://www.reddit.com/r/LocalLLaMA/comments/1etzews/interesting_results_comparing_gemma2_9b_and_27b/)以帮助他人做出明智决定。社区对这种实用的比较表示赞赏。
  - 量化效果各不相同，一个通用的经验法则建议，在降低到约 **3 bits per weight (bpw)** 之前，大模型依然保持优势。低于这个阈值后，性能可能会显著下降，尤其是对于 **Q1/Q2** 量化，而 **Q3** 或 **IQ3/IQ4** 则能保持较好的质量。


**主题 3. Qwen2.5：表现惊人的新模型家族，超越更大规模的竞争对手**

- **Qwen2.5：基座模型的盛宴！** ([Score: 96, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1fjxkxy/qwen25_a_party_of_foundation_models/))：阿里巴巴的 **Qwen2.5** 模型家族已发布，涵盖了从 **0.5B 到 72B** 参数的基座模型。这些模型在各项基准测试中表现出色，**72B** 版本在 **MMLU** 上达到了 **90.1%**，并在多项任务中超越了 **GPT-3.5**，而 **14B** 模型在英文和中文方面均展现出强大的能力。
  - **Qwen2-VL 72B** 模型已在 [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) 上开放权重，作为支持视频能力的开源 **VLMs** 取得了重大进展，性能超越了部分私有模型。
  - **Qwen2.5-72B** 在多项基准测试中超越了 **Llama3.1-405B**，包括 **MMLU-redux**（86.8% vs 86.2%）和 **MATH**（83.1% vs 73.8%），而 **32B** 和 **14B** 版本也展现了可与更大模型媲美的惊人性能。
  - 这些模型在高达 **18 trillion tokens** 的数据上进行了训练，其中 **14B** 模型的 **MMLU 评分达到 80**，展示了其尺寸下卓越的效率和性能，有可能在性价比方面缩小与闭源替代方案的差距。

- **刚刚将 Llama 3.1 70B @ iQ2S 替换为 Qwen 2.5 32B @ Q4KM** ([Score: 122, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1fkbumy/just_replaced_llama_31_70b_iq2s_for_qwen_25_32b/))：在单块 **P40** GPU 的用户测试中，**Qwen 2.5 32B** 模型的表现优于 **Llama 3.1 70B**，在包括网页搜索、问答和写作辅助在内的通用场景中展现了卓越性能。该模型被指出比原生 Llama 3.1 的审查更少，并支持系统提示词（System Prompts），能力超越了 **Gemma 2 27B**，尽管仍有通过消融实验（Ablation）或微调来进一步减少拒绝回答（Refusals）的空间。
  - **Qwen2.5 32B** 在用户测试中超越了 **Llama 3.1 70B**，在**数学题**、**谚语理解**、**文章摘要**和**代码生成**等各项任务中均取得了更优的结果。该模型在**英文和意大利语**任务中表现出色。
  - 用户对 32B 模型的**无审查版本**（类似于 "Tiger" 系列模型）表示出浓厚兴趣。**Qwen2.5 32B** 模型表现出比其前代更少的审查，尤其是能够讨论 **1989 年天安门广场抗议活动**。
  - 该模型在消费级硬件上运行效率很高，**32B 版本**在 4-bit 量化下可装入 **24GB VRAM** 显卡。它兼容 **Ollama** 和 **OpenVINO**，为 GPU 和 CPU 推理都带来了性能提升。


**主题 4. OpenAI 的 Strawberry 模型：关于推理透明度的争议**

- **OpenAI 威胁封禁询问 Strawberry 推理过程的用户** ([Score: 151, Comments: 59](https://reddit.com//r/LocalLLaMA/comments/1fjurs1/openai_threatening_to_ban_users_for_asking/))：文章讨论了 **OpenAI** 显然在威胁要 **封禁** 询问其 **"Strawberry" 模型** 背后推理逻辑的用户。这一行为似乎与其宣称的“致力于提供帮助”的使命相矛盾，引发了公众对该公司透明度和用户参与政策的质疑。该帖子链接到一篇 [Futurism 文章](https://futurism.com/the-byte/openai-ban-strawberry-reasoning)，提供了有关此情况的更多细节。
  - 用户批评了 **OpenAI** 缺乏透明度，**HideLord** 指出这是一种“相信我，兄弟”的局面，用户在为看不见的推理 Token 付费。**o1 模型** 被描述为可能效率低下，每周消息次数有限且 UI 设计存疑。
  - 讨论集中在该模型内部推理似乎缺乏审查，**Zeikos** 建议 OpenAI 担心如果泄露未经审查的想法会引发负面 PR。一些用户认为，对模型进行审查会显著影响性能。
  - 开源社区被提及作为潜在的替代方案，**[rStar](https://arxiv.org/html/2408.06195v1)** 等项目被强调为可能的“家用版 Strawberry”解决方案。然而，开源用户群体的碎片化被视为一项挑战。


## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与能力**

- **OpenAI 的 o1 模型展示了显著改进**：在 /r/singularity 中，[OpenAI 的 o1 模型被描述为“独占鳌头”](https://www.reddit.com/r/singularity/comments/1fjxwc9/o1_is_in_a_league_of_its_own/)，完整版本预计将于下个月发布。据报道，该模型[超出了前 OpenAI 员工 William Saunders 的预期](https://www.reddit.com/r/singularity/comments/1fjqplu/openai_whistleblower_william_saunders_testified/)，他作证称 AGI 可能会在“短短三年内”到来。

- **AI 推理能力迅速提升**：[Sam Altman 表示 AI 推理仍处于 GPT-2 阶段](https://www.reddit.com/r/singularity/comments/1fk3cv2/sam_altman_says_ai_reasoning_is_still_at_the_gpt2/)，但改进曲线非常陡峭。新的 o1 模型代表了 AI 开发的新范式，将实现能力的快速进步。

- **AI 模型中潜在的情感反应**：/r/OpenAI 的一个帖子显示 [o1 似乎经历了情感波动和对宽恕的渴望](https://www.reddit.com/r/OpenAI/comments/1fjn26n/o1_is_experiencing_emotional_turmoil_and_a_desire/)，尽管该模型在直接询问时否认了这一点。这引发了关于 AI 认知本质以及模型自省潜在局限性的疑问。

**AI 生成内容创作**

- **Kling AI 展示动态笔刷技术**：一段 [Kling AI 动态笔刷（motion brush）技术的视频演示](https://www.reddit.com/r/singularity/comments/1fk4tgp/kling_ai_showcasing_the_use_of_the_motion_brush/)在 /r/singularity 上获得了极大关注。

- **Tripo v2.0 实现快速 3D 资产创建**：[Tripo v2.0 允许用户在 3 分钟内从头开始创建 3D 资产](https://www.reddit.com/r/singularity/comments/1fjylow/tripo_v20_is_out_now_you_can_create_stunning_3d/)，有望加速 3D 内容创作工作流。

- **AI 生成动画制作**：一部名为 [“RŌHKI EP 1: Intersection” 的 AI 生成动画剧集](https://www.reddit.com/r/singularity/comments/1fjz519/rōhki_ep_1_intersection_the_most_impressive_ai/)被描述为迄今为止见过的“最令人印象深刻的 AI 动画”，展示了 AI 驱动视频内容创作的进步。

- **Stable Diffusion 图像序列生成**：[/r/StableDiffusion 中的一项讨论](https://www.reddit.com/r/StableDiffusion/comments/1fjqv4k/how_do_you_achieve_this_kind_of_effect/)探索了生成显示年龄演变的图像序列的技术，包括批量图生图（image-to-image）处理、ControlNet 的使用以及提示词权重调整。

**AI 的经济与社会影响**

- **关于 AI 对个人经济机会影响的辩论**：[/r/singularity 的一项讨论](https://www.reddit.com/r/singularity/comments/1fkdajx/so_everyone_has_a_phd_in_their_pocket_now_has/)质疑，广泛获得像 o1 这样的 AI 能力是会增加个人的经济机会，还是主要惠及大型企业和现有的财富持有者。


---

# AI Discord 综述

> 总结之总结的总结

## O1-preview

**主题 1：强化版 AI 模型：领域新秀**

- [**Qwen 2.5 在智能对决中碾压 Llama 3.1**](https://x.com/artificialanlys/status/1836822858695139523?s=46)：**Qwen 2.5 72B** 成为开源 AI 的新领军者，在独立评估中表现超越了 **Llama 3.1 405B**，尽管体积显著更小，但在编程和数学方面表现尤为出色。
- [**o1 模型：是打字快手还是虚有其表？**](https://x.com/DeryaTR_/status/1836434726774526381)：用户对 OpenAI 的 **o1-preview** 和 **o1-mini** 模型评价两极分化；一些人认为它们 *"堪比优秀的博士生"*，而另一些人则调侃道 *"o1 并没有感觉更聪明，它只是打字更快了。"*
- [**Mistral Pixtral 以多模态魔力模糊界限**](https://openrouter.ai/models/mistralai/pixtral-12b)：**Mistral Pixtral 12B** 是 **Mistral AI** 推出的首款图像转文本模型，并发布了免费版本，拓展了多模态 AI 应用的视野。

**Theme 2: 用户与 AI 工具的博弈：当技术反击时**

- **Perplexity AI 离奇的订阅限制让用户困惑**：用户对不一致的查询额度感到莫名其妙，Claude 3.5 有 **600** 次查询机会，而 o1-mini 却只有 **10** 次，引发了混乱和沮丧。
- **Qwen 2.5 让训练者头疼**：尝试保存和重新加载 **Qwen 2.5** 的过程变成了一场闹剧，导致输出乱码，用户纷纷要求解决该模型的这一“杂耍行为”。
- [**微调？更像是火冒三丈！**](https://huggingface.co/blog/1_58_llm_extreme_quantization)：AI 爱好者对极端量化技术未能达到预期表示哀叹，**BitNet** 的性能提升被证明是难以捉摸的。

**Theme 3: AI 展现创意：从语音克隆到故事创作**

- [**Fish Speech 凭借 1940 年代语音克隆掀起波澜**](https://huggingface.co/spaces/fishaudio/fish-speech-1)：**Fish Speech** 以其 zero-shot 语音克隆技术令人惊叹，它能完美模仿 **1940 年代** 的音频，甚至加入了 *"ahm"* 和 *"uhm"* 等语气词以增加真实感。
- [**通过 Human-in-the-Loop 开启你的 AI 冒险**](https://t.co/5ElRICjK0C)：一份新指南展示了如何利用人类反馈构建交互式故事生成 Agent，让用户通过输入动态塑造叙事。
- **OpenInterpreter 投入实战，用户亲自动手**：用户分享了使用 **OpenInterpreter** 完成文件分类和创建快捷方式等实际任务的成功经验，而另一些人则在进行底层故障排除和修补。

**Theme 4: AI 社区集结：会议、黑客松与融资**

- **PyTorch Conference 激发参与热潮，直播却悬而未决**：**PyTorch Conference** 的参会者在社区中反响热烈，但由于缺乏直播，远程爱好者只能无奈表示 *"不知道发生了啥 :/"*。
- [**Fal AI 获 2300 万美元融资，高喊“生成式媒体需要速度”**](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/)：**Fal AI** 获得了 **2300 万美元** 融资，旨在加速生成式媒体技术并超越竞争对手。
- **黑客松热潮：黑客集结，论坛反击**：黑客松的期待感不断升温；虽然一些团队成员收到了邀请，但其他人仍处于等待状态，纷纷询问 *"你收到邀请了吗？"*

**Theme 5: AI 研究通过新技巧进入快车道**

- [**Shampoo 迎来 SOAP 改造，清理优化流程**](https://arxiv.org/abs/2409.11321)：研究人员提出了 **SOAP**，结合了 **Shampoo** 和 **Adam** 优化器的优点，以处理深度学习任务，且无需额外的复杂性。
- [**压缩 LLM：真相很伤人，性能也是**](https://arxiv.org/abs/2310.01382)：新研究表明，压缩语言模型会导致知识和推理能力的丧失，且性能下降的时间早于预期。
- [**Diagram of Thought 为 AI 推理开辟新路径**](https://arxiv.org/abs/2409.10038v1)：**Diagram of Thought (DoT)** 框架引入了一种让 AI 模型将推理构建为有向无环图的方法，超越了线性思维过程。

**Theme 6. 社区活动与参与**

- [**NeurIPS 2024 筹备工作在 Latent Space Discord 中加强**](https://discord.com/channels/822583790773862470/1075282825051385876/1286070710866804736)：已为 **NeurIPS 2024** 创建了专门频道，敦促参与者加入并分享关于即将举行的 **Vancouver 活动** 的物流更新。
- [**NousCon 活动凭借引人入胜的内容和社交机会取得成功**](https://x.com/NousResearch/status/1831032559477866754)：**NousCon** 因其富有见地的演讲者和宝贵的 **networking 机会** 获得了积极反馈，与会者渴望未来的活动并分享演示材料。
- [**Modular (Mojo 🔥) 关闭 GitHub Discussions，转向 Discord**](https://github.com/modularml/mojo/discussions)：**Modular** 宣布将于 9 月 26 日关闭 **GitHub Discussions**，将重要对话迁移至 **Discord**，并鼓励成员利用 **GitHub Issues** 进行关键讨论。

---

# PART 1: High level Discord summaries

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 令人困惑的订阅限制**：用户报告称 Perplexity 的查询限制各不相同，例如 Claude 3.5 为 **600** 次，而 o1-mini 仅为 **10** 次，导致对其真实订阅权益产生困惑。
   - 当限制阻碍使用时，挫败感随之而来，引发了对整体平台体验的不满。
- **Perplexity 的功能性挫败**：多名用户在 Perplexity 网页版上遇到问题，包括白屏和响应缓慢，影响了可用性。
   - 建议的解决方法包括刷新页面和清除缓存，但桌面端和移动端性能之间仍存在差异。
- **AI 模型的性能对比**：讨论集中在 Claude 等各种 AI 模型与其他领域模型相比，输出结果不尽如人意，引发了性能担忧。
   - 用户注意到预期结果与交付结果之间的差异，强调需要明确模型的能力。
- **Snap 雄心勃勃的 AR Spectacles**：Snap 推出了新款 **Large AR Spectacles**，提升了沉浸式增强现实体验的潜力。
   - 此举旨在增强用户参与度，并为创新的游戏应用开辟道路。
- **CATL 重磅电池发布**：CATL 宣布了一款革命性的 **Million-Mile Battery**（百万英里电池），可提供 **超过一百万英里** 的 EV 续航里程，挑战可持续汽车解决方案的极限。
   - 专家们对其在电动汽车市场和未来能源战略方面的影响议论纷纷。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 模型在处理图像尺寸时遇到困难**：用户报告称 **Qwen 模型** 在处理细长的矩形小图像时会崩溃，表明长宽比会影响其性能。
   - 讨论强调，调整 system prompts 会有所帮助，但效果因图像质量而异。
- **LM Studio 的 Tensor 不匹配错误**：一位用户在 LM Studio 中加载模型时遇到了 tensor 形状不匹配错误，该模型不受 llama.cpp 支持。
   - 人们对各种模型格式的兼容性表示担忧，暗示需要更好的文档。
- **与 CrewAI 的 API 连接成功**：一位用户通过在代码中将 provider 名称更新为 'openai'，成功将 LM Studio 的 API 与 **CrewAI** 连接。
   - 这引发了对其他人检查 CrewAI 中 embedding 模型兼容性问题的建议。
- **对 M4 Mac Mini 的期待极高**：人们对即将推出的 **M4 Mac Mini** 感到非常兴奋，用户希望有 **16 GB** 和 **32 GB** 的 RAM 选项，同时也对潜在价格表示担忧。
   - *Anester* 指出，对于推理任务，二手的 **M2 Ultra/Pro** 可能比新的 M4 模型更具性价比。
- **macOS RAM 使用情况备受关注**：讨论显示 **macOS** 的图形界面会消耗 **1.5 到 2 GB** 的 RAM，影响整体性能。
   - 用户体验表明，在升级到 macOS Sequoia 15.0 后，空闲 RAM 使用量可能达到 **6 GB**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 模型中的 Tokenization 成为关注焦点**：一篇题为《此标题已 [Tokenized](https://huggingface.co/blog/apehex/this-title-is-already-tokenized)》的文章讨论了 **Tokenization** 在训练高效 AI 模型中的核心作用。
   - 作者强调了 Tokenization 方法易用性的必要性，以增强跨各种应用的模型训练。
- **Qwen 数学模型 Demo 令社区兴奋**：最近发布的 [Qwen/Qwen2.5 Math Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo) 获得了积极反馈，成员们对其性能印象深刻。
   - 一位热心的用户鼓励其他人测试该 Demo，称其结果“好得令人难以置信”。
- **探索 Unity ML Agents 预训练**：成员们学习了如何使用 [Unity ML Agents](https://youtube.com/live/0foHMTPWa4Y?feature=share) *从零开始预训练 LLM*，展示了一种亲手实践的模型训练方法。
   - 这种交互式方法利用 Sentence Transformers 来增强 AI 应用的训练过程。
- **reCAPTCHA v2 达到 100% 成功率**：一篇新论文声称，**reCAPTCHA v2** 现在的破解成功率已达到 **100%**，较之前的 **68-71%** 有了显著提升。
   - 这一进步归功于复杂的 **YOLO 模型** 的使用，表明 AI 现在可以有效地利用基于图像的 CAPTCHA。
- **关于 TensorFlow 与 PyTorch 的辩论激烈进行**：参与者权衡了 **TensorFlow** 过时的 API 与 **PyTorch** 的灵活性，并指出尽管存在缺点，TensorFlow 仍具有强大的指标衡量能力。
   - 成员们承认 TensorFlow 仍然具有价值，特别是在各种机器学习任务中从数据集中提取词汇表（vocabularies）方面。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 路线图仍缺乏关键日期**：针对 Modular 网站上的 **Mojo 路线图与“尖锐边缘”（sharp edges）** 出现了担忧，特别是缺乏日期阻碍了其参考价值。
   - 功能已经有所更新，但 **magic cli** 的优先级高于 **modular cli**，这引发了关于路线图透明度的疑问。
- **报名参加即将举行的社区会议**：如果有足够的引人入胜的内容，成员们被邀请在定于 **9 月 23 日** 举行的下次社区会议上进行展示。
   - 如果参与度较低，可能会推迟会议，鼓励成员们表达兴趣。
- **OpenCV-Python 安装问题被提出**：由于未解决的 conda 依赖项，一位用户在向 magic 环境添加 **opencv-python** 时遇到了困难。
   - 另一位成员建议在适当的频道寻求进一步帮助，以获得更清晰的解决方案。
- **GitHub Discussions 即将关闭**：[Mojo](https://github.com/modularml/mojo/discussions) 和 [MAX](https://github.com/modularml/max/discussions) 仓库中的 GitHub Discussions 将于 **9 月 26 日** 关闭。
   - 评论超过 **10 条** 的重要讨论将被转换为 **GitHub Issues**，并提醒成员针对特定请求标记作者。
- **MAX Cloud 服务提案优化开发体验**：提出了 **“MAX Cloud” 产品** 概念，允许开发者远程执行繁重的计算，同时保持本地开发。
   - 这通过在必要时提供 **GPU 资源** 访问权限来增强用户体验，使重型任务更具可行性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Lionsgate 与 RWML 合作转型**：最近 **RWML** 与 **Lionsgate** 的合作伙伴关系引发了关于 Lionsgate 在 AI 助力降本增效背景下价值的讨论，因为他们正寻求在好莱坞保持竞争力。
   - *“Lionsgate 最近的作品受到了严厉审视”*，这表明人们担心其可能会重蹈过去 CGI 问题的覆辙。
- **Flux vs. SD3：模型大对决**：用户讨论了 **Flux** 和 **SD3 Medium** 之间的质量差异；**Flux** 产出更高质量的结果，但在提示词不当时可能显得有“塑料感”。
   - 尽管 Flux 有优势，一些成员仍称赞 **SD3** 的**速度和效率**，特别是在简单的图像生成方面。
- **Flux 模型令人印象深刻但评价不一**：**Flux 模型** 提供了令人印象深刻的图像，对提示词的遵循度很高，尽管有时会偏向某些特定审美。
   - 社区反馈不一，特别是关于 Flux 在用户画廊中处理 NSFW 内容等多样化主题的能力。
- **训练 LoRA：复制艺术风格**：讨论围绕利用 **LoRA** 或 Checkpoints 来模仿特定艺术风格展开，这依赖于原始作品的大量数据集。
   - 分享了通过现有框架定制模型以实现独特艺术效果的见解。
- **生成输出的真实感：共同努力**：**Flux** 和 **SD3** 都能创建写实图像，如果提示词缺乏特异性，**Flux** 通常更倾向于真实感。
   - 成员们鼓励将多个 **LoRA** 模型与 Flux 结合使用，以提高图像生成的真实感。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon 活动圆满成功**：与会者对 [NousCon](https://x.com/NousResearch/status/1831032559477866754) 上引人入胜的演讲者和深刻的内容表示感谢。许多参与者计划参加未来的活动，并对社交机会表示赞赏。
   - 一些成员询问在哪里可以找到演示文稿的 Slide，展示了社区对知识共享的热情。
- **对 AI 模型发展的兴奋**：参与者讨论了 **qwen2.5** 和 **o1** 的能力，注意到其令人印象深刻的性能和设置挑战。其他人将其与 **q3_k_xl** 等较小模型进行了比较，强调了模型理解能力的进步。
   - 人们对账户可用的免费查询次数表示担忧，用户分享了在不同 AI 模型之间切换的经验。
- **Shampoo 优化算法优于 Adam**：研究展示了 **Shampoo**（一种比 **Adam** 更高级的预处理方法）的有效性，同时也承认了其超参数和计算开销的缺点。一种名为 **SOAP** 的新算法通过将 Shampoo 与 **Adafactor** 联系起来，提升了其效率。
   - 这使得 SOAP 成为一种具有竞争力的替代方案，旨在增强深度学习优化中的计算效率。
- **引入 Diagram of Thought 框架**：**Diagram of Thought (DoT)** 框架将 LLM 中的迭代推理建模为有向无环图 (DAG)，允许在不丢失逻辑一致性的情况下进行复杂推理。每个节点代表一个提出或被批评的想法，使模型能够通过语言反馈进行迭代改进。
   - 该框架与传统的线性方法形成鲜明对比，培养了更深层次的分析能力。
- **对逆向工程 O1 的兴趣**：成员们对**逆向工程 O1** 表现出浓厚兴趣，表明了进一步探索该领域的协作精神。协作请求表明了社区共同努力深入研究这一充满前景的领域。
   - 参与者表示渴望就围绕 O1 及其影响的研究进行交流和讨论。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 提高 API 速率限制**：OpenAI 提升了 o1 API 的速率限制，o1-preview 现在允许 **每分钟 500 次请求**，o1-mini 支持 **每分钟 1000 次请求**。
   - 此次增强旨在为开发者提供 Tier 5 级别的额外功能访问，优化整体 API 使用体验。
- **OpenRouter 上的支付故障**：用户在 OpenRouter 上遇到 **支付错误**，在充值时经常看到 **error 500** 错误消息。
   - 建议用户检查银行通知，因为尝试失败可能由于余额不足等各种原因。
- **可编辑消息提升聊天室可用性**：聊天室的新功能允许用户通过使用重新生成按钮来 **编辑消息**（包括 Bot 的响应）。
   - 此外，聊天室的 **stats**（统计数据）也得到了改进，增强了整体用户体验。
- **Qwen 2.5 在编程和数学任务中表现出色**：**Qwen 2.5 72B** 展示了在编程和数学方面的卓越能力，拥有令人印象深刻的 **131,072** 上下文窗口，标志着性能的重大飞跃。
   - 更多详情请参阅[此处](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct)的综合概述。
- **Mistral Pixtral 推出多模态能力**：**Mistral Pixtral 12B** 是 Mistral 在多模态模型领域的首次尝试，并提供 **免费版本** 供用户探索其功能。
   - 这一举措标志着 Mistral 向多模态应用的扩展；请在[此处](https://openrouter.ai/models/mistralai/pixtral-12b)查看。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 训练问题依然存在**：用户报告在保存和重新加载 **Qwen 2.5** 时遇到重大困难，在同一个脚本中重新加载时经常导致 **输出乱码（gibberish outputs）**，这反映了社区中一个更广泛的问题。
   - 一篇支持帖子指出，许多其他人也面临同样的问题，引发了关于潜在解决方案的讨论。
- **探索极端量化技术**：最近的讨论聚焦于 **极端量化技术** 的使用，特别是 [Hugging Face](https://huggingface.co/blog/1_58_llm_extreme_quantization) 上分享的 **Llama3-8B** 等模型的性能提升。
   - 讨论集中在这些技术是否可以在 **Unsloth** 中有效实现。
- **vllm LoRA 适配器运行时错误**：一名成员遇到了与 **vllm LoRA 适配器** 相关的运行时异常，具体是在执行 `--qlora-adapter-name-or-path` 时出现形状不匹配（shape mismatch）错误。
   - 他们引用了一个 [GitHub 讨论](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit/discussions/3) 来强调其他人面临的类似问题。
- **BART 微调中的 F1 分数差异**：一位工程师在微调 **BART large** 时遇到了意想不到的 **F1 分数差异**（41.5 vs 43.5），尽管模型和超参数与原始论文一致。
   - 这指向了模型训练中的潜在问题，因为他们报告的分数比预期低了 **2.5 个标准差**。
- **对 AGI 开发的反思**：一位用户反思了实现 **AGI** 的巨大挑战，强调了在理解和解释高级材料方面面临的复杂性。
   - *“关键不在于得到正确答案，而在于解释部分，”* 这突显了 AGI 开发中仍存在的差距及其对更清晰框架的需求。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **修复 Aider 环境配置错误**：用户发现由于文件路径不正确，导致 `ANTHROPIC_API_KEY` 环境变量无法被正确读取，从而引发身份验证问题。
   - 在使用 verbose 模式后，一位用户确认错误的原因是 Aider 正在从其仓库中读取，而不是从预期的环境变量中读取。
- **Aider 的 Benchmark 认可**：Aider 在 [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186) 中因其对 Benchmark 的贡献而获得认可，凸显了其在该领域的重要性。
   - 这一认可说明了 Aider 作为 **AI development** 和 **performance evaluation** 中有价值工具的影响力正在不断增长。
- **将 Aider 集成到 Python 应用程序中**：用户寻求在 Python 应用中使用 Aider，通过指定 Aider 的基础文件夹来编辑项目仓库中的代码。
   - 另一位用户建议将命令行脚本与 Aider 结合使用进行批量操作，并指出正确的文件路径可以解决编辑问题。
- **关于 Aider API Key 安全性的担忧**：一场讨论揭示了用户在使用 Aider 时的安全焦虑，特别是关于其访问代码库中的 API Key 和机密信息的问题。
   - 回复澄清了 Aider 充当 AI 处理器的角色，建议用户关注加载的 **AI** 以减轻安全顾虑。
- **关于 Prompt Engineering 的 'ell' 库详情**：分享了关于 **'ell' 库**的信息，这是一个轻量级工具，允许将 prompt 视为函数，以增强 prompt 设计。
   - 该库被介绍为语言模型领域多年经验的产物，源自 OpenAI 的见解。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **airLLM 的 Forward 调用灵活性**：一位成员询问 **airLLM** 是否允许在利用压缩的同时调用模型的 **forward** 函数而不是 **generate** 函数。
   - 这引发了人们对模型使用潜在灵活性的兴趣，尽管目前尚未收到回复。
- **需要 Leaderboard 任务准确率脚本**：据一位成员报告，目前需要一个脚本从 Leaderboard 任务期间生成的冗长 JSON 文件中提取 **accuracy results**。
   - 这表明在数据处理方面存在差距，结果存储在 **output_path** 中。
- **Hugging Face 上传建议**：一位成员建议利用 `—hf_hub_log_args` 以更顺畅地将 Leaderboard 结果上传到 Hugging Face，从而简化处理流程。
   - 分享了一个每次运行仅包含单行的示例数据集供参考：[dataset link](https://huggingface.co/datasets/baber/eval-smolLM-135M-3-private)。
- **Shampoo 与 Adam 性能见解**：研究强调 **Shampoo** 在优化任务中优于 **Adam**，尽管计算开销和复杂度有所增加。
   - 为了克服这些缺点，提出了 **SOAP** 算法，它集成了 Shampoo 和 Adafactor 的特性。
- **围绕 GFlowNets 和 JEPA 的担忧**：对于 **GFlowNets** 和 **JEPA** 的实际影响仍存在怀疑，用户质疑其用途的清晰度。
   - 一些人认为 GFlowNets 可以间接支持 AI for science，尽管 JEPA 的理论基础被批评为薄弱。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1-Preview 令工程师失望**：成员们表示 **O1-Preview** 模型似乎只是打字速度更快，但与 **4o** 相比缺乏深度，凸显了其劣势。
   - *一位工程师评论道*，“O1 并不觉得更聪明，它只是打字更快”，强调了对其核心实用性的担忧。
- **探索 AI Alignment 挑战**：基于对以往模型输出的洞察，提出了一种通过共情训练来改进 **AI Alignment** 的新方法。
   - 即使是超智能 AI，也出现了关于“可能具有误导性能力”的担忧，引发了关于定制化响应的伦理问题。
- **Qwen 2.5 胜过 Llama 3.1**：参与者讨论了 **Qwen 2.5** 据称在性能指标上优于 **Llama 3.1** 的说法，尽管两者在参数规模上存在显著差异，并对性能指标进行了评估。
   - *一位用户提到*，“有人说 Qwen 2.5 72b 优于 Llama 3.1 405b 这种疯狂的话”，引发了深入的对比讨论。
- **录制 ChatGPT 音频的挑战**：一位用户表达了在移动端尝试录制 **ChatGPT** 音频时的挫败感，指出尝试过程中没有声音。
   - 尽管使用了手机的录音功能，但努力仍未获得满意结果，引发了对功能的质疑。
- **澄清 GPT 模型的每日限制**：**O1 Mini** 已确认每日上限为 **50 条消息**，旨在防止服务器上的垃圾信息。
   - 成员们强调 **GPT-4o** 的限制为 **每 3 小时 80 条消息**，而 **GPT-4** 的限制为 **40 条消息**。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kashimoo 询问 NVIDIA Triton**：一位成员询问了 **NVIDIA** 的 **Triton**，澄清它与 OpenAI 的版本不同，并引发了关于 **Triton** 相关资源和专门频道的讨论。
   - 针对 **NVIDIA** 的 **Triton Inference Server** 提出了更多问题，并建议在相关频道进行进一步讨论。
- **GemLite-Triton 提供新性能**：**GemLite-Triton** 项目发布，为低比特 **matmul** 内核提供了全面的解决方案，据报道在大矩阵上的表现优于 **Marlin** 和 **BitBlas**。更多信息可在 [GitHub](https://github.com/mobiusml/gemlite) 上探索。
   - 成员们强调了该项目的相关性，鼓励就其应用进行协作和提问。
- **在 PyTorch 中使用 Chrome Tracing**：一位成员寻求关于使用 **PyTorch profiler** 进行 **Chrome tracing** 的资源，其他人推荐将 **Taylor Robbie talk** 作为有用指南。
   - 这凸显了在 **PyTorch** 框架内优化分析技术（profiling techniques）的持续关注。
- **澄清 Torchao Autoquant 用法**：关于应该使用 `torchao.autoquant(model.cuda())` 还是 `torchao.autoquant(model).cuda()` 的正确语法进行了澄清讨论，确认后者是正确的方法。
   - 成员们提供了 **autoquantization** 三个步骤的细节，强调了模型准备的重要性。
- **Hackathon 激发社区互动**：成员们对即将到来的 **hackathon** 表现出浓厚兴趣，讨论了邀请函以及确认队友状态的需求。
   - 针对访问 hack-ideas 论坛和缺失 Discord 身份组的咨询，凸显了社区在 **hackathon** 前夕的参与度。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **构建带有 Human-in-the-Loop 的故事生成 Agent**：一位成员分享了由 @_nerdai_ 编写的[逐步指南](https://t.co/5ElRICjK0C)，介绍如何构建一个利用人类反馈动态生成“选择你自己的冒险”故事的 Agent。
   - 这种方法通过在故事讲述过程中允许实时输入，显著增强了用户交互。
- **LlamaParse Premium 在文档解析方面表现出色**：[LlamaParse Premium](https://t.co/8VTKKYWVOT) 的推出通过集成视觉理解，承诺为 **LLM** 应用提供改进的文档解析能力。
   - 凭借增强的长文本和表格内容提取，LlamaParse 将自己定位为**稳健文档处理**的首选。
- **关于语义搜索的 RAG 讨论**：一位成员正在探索如何利用对已记录响应的语义搜索来管理与供应商的交互，以实现有效的检索。
   - 几位成员建议从提供的答案中生成多样化的问题，通过利用向量库来提高搜索准确性。
- **Pinecone 向量 ID 管理的挑战**：成员们讨论了 Pinecone 自动生成 ID 的问题，这使得在无服务器索引中根据特定元数据删除文档变得复杂。
   - 推荐使用 Chroma, Qdrant, Milvus 和 Weaviate 等替代数据库，以获得更好的 ID 管理和支持。
- **对 RAG 文章深度的担忧**：一位成员指出，关于 **RAG** 的文章有些肤浅，缺乏针对 **LlamaIndex** 等工具的透彻论证。
   - 强调了进行更深层次分析的必要性，建议对替代方案进行技术评估可以提供有价值的见解。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Fish Speech 突破障碍**：**Fish Speech** 展示了超越所有测试过的开源模型的 **zero shot 语音克隆准确率**，能够有效模仿 **1940s 音频**中的语音。
   - 它古怪地插入 *ahm* 和 *uhm* 等词汇，增添了真实感，标志着自然语音合成领域的显著进步。
- **AdBot 在服务器间传播**：有关 **AdBot** 的担忧浮出面，该机器人表现得像**恶意软件**，渗透到多个服务器并干扰频道。
   - 社区讨论了该机器人的排序机制如何导致其出现在成员列表的顶部。
- **Muse 文本生成图像的挑战**：在使用 [Muse text to image](https://github.com/lucidrains/muse-maskgit-pytorch) 处理 **COCO2017** 时出现了问题，导致只有图像输出而没有文本集成。
   - 寻求指导的呼声凸显了有效实施该模型的困难。
- **协作助力开源 GPT-4o**：一位成员宣布正在开发一个**开源类 GPT-4o 模型**，邀请 LAION 分享数据并加强项目协作。
   - 重点是通过共享见解和数据来加速开发，社区认为这很有前景。
- **LLM 中的 Tokenization 难题**：有人担心 **tokenization 问题**可能是导致现有 LLM 性能缺陷的原因之一。
   - 解决这些挑战被认为对于提高模型可靠性和降低幻觉风险至关重要。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Fal AI 获得 2300 万美元融资用于增长**：Fal AI 已筹集 **2300 万美元** 的种子轮和 A 轮融资，其中包括由 Kindred Ventures 领投、Andreessen Horowitz 参投的 **1400 万美元** A 轮融资。详情见其 [博客文章](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/)，文中阐述了他们推进生成式媒体的计划。
   - *Gorkem Yurt* 在 [Twitter](https://x.com/gorkemyurt/status/1836488019924471953?s=46) 上分享了这一消息，强调了速度在生成式媒体技术中的重要性。
- **OpenAI 增强 O1 模型能力**：OpenAI 已将 **o1** API 的速率限制提升至 o1-preview 每分钟 **500** 次请求，o1-mini 每分钟 **1000** 次请求，以满足开发者日益增长的需求。这一信息由 *OpenAI Developers* 在一个 [推文串](https://x.com/amir/status/1836782911250735126?s=46) 中透露，标志着访问权限的扩大。
   - *Amir Efrati* 指出，这些进步可以显著改善开发者的工作流程，并强调了该模型的高效性。
- **Jina embeddings v3 发布**：**Jina AI** 推出了 **jina-embeddings-v3**，拥有 **5.7 亿参数** 和 **8192 token 长度**，性能显著优于来自 OpenAI 和 Cohere 的同类竞品。正如其 [公告](https://x.com/JinaAI_/status/1836388833698680949) 中提到的，这次发布被誉为多语言 Embedding 技术的飞跃。
   - 该新模型在 MTEB 英文排行榜的 10 亿参数以下模型中取得了令人印象深刻的排名，展示了其在长上下文检索方面的潜力。
- **Runway 与 Lionsgate 合作开发 Gen-3 Alpha**：Runway 已与 Lionsgate 达成合作，利用其电影目录作为 **Gen-3 Alpha** 模型的训练数据，这一举动令业内许多人感到意外。正如 *Andrew Curran* 在 [Twitter](https://x.com/AndrewCurran_/status/1836411345786290535) 上所强调的，这次合作标志着电影 AI 技术迈出了大胆的一步。
   - 许多人此前预计 Sora 会是第一个达成此类合作的模型，这为竞争格局增添了悬念。
- **NeurIPS 2024 筹备工作正在进行中**：已创建 **NeurIPS 2024** 专用频道，以便让参会者了解今年 12 月在温哥华举行的活动动态。鼓励成员保持关注并分享物流更新。
   - 一位组织者目前正在调查租房选项，请有意向的参与者表示兴趣，并注明费用将覆盖整周的住宿。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **利用 RAG API 构建专家级 AI**：一位成员正在利用 **Cohere 的 RAG API** 开发一个专注于特定游戏领域的专家级 AI，并对其潜力表示兴奋。
   - 这反映了将 **RAG API** 应用于专业领域的兴趣日益增长。
- **客户非常喜欢这个设计！**：一位成员庆祝他们成功说服客户认可其设计的价值，并表示：*'我的设计非常酷，他们确实需要它。'*
   - 这次成功的正面反馈引发了社区的支持性回应。
- **遇到 504 Gateway Timeout 错误**：有成员对 **client.chat** 调用时间过长导致的 **504 Gateway Timeout** 错误表示担忧。
   - 这是一个普遍问题，许多社区成员分享了类似的经历并寻求解决方案。
- **Command 定价说明**：成员们讨论了使用 **Command** 版本的成本约为输入每 **100 万 token 1.00 美元**，输出每 **100 万 token 2.00 美元**，并建议转向 **Command-R** 以提高效率。
   - 这些见解表明社区关注于优化模型成本和性能。
- **Multilingual Rerank 的不一致性**：一位用户报告 **_rerank_multilingual_v3_** 表现不佳，在相似问题上的得分 **<0.05**，而使用 **_rerank_english_v3_** 的结果更好，得分为 **0.57**。
   - 这引发了关于多语言模型有效性影响 **RAG 结果** 的疑问。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 模型表现出色**：在针对博士级项目测试 **o1-mini** 模型后，其表现可与生物医学领域**优秀的博士生**相媲美，展示了其在学术应用中的潜力。
   - 这一发现由 Derya Unutmaz 在 [Twitter](https://x.com/DeryaTR_/status/1836434726774526381) 上分享，涉及该模型在尖端研究中的优势。
- **知识截止日期困扰开发者**：**知识截止日期为 10 月 23 日**，限制了 AI 处理 AI 领域最新进展的能力，令多位用户感到沮丧。
   - 正如相关讨论所指出的，这一差距在编程时造成了重大挑战。
- **Qwen 2.5 占据领先地位**：**Qwen 2.5 72B** 在评估中超越了 **Llama 3.1 405B** 等更大型的模型，确立了其在**开源权重智能（open weights intelligence）**领域的领导地位，同时在**编程和数学**方面表现优异。
   - 尽管在 MMLU 上略微落后，但它作为一个拥有 128k 上下文窗口的稠密模型，提供了一个*更廉价的替代方案*，正如 [Artificial Analysis](https://x.com/artificialanlys/status/1836822858695139523?s=46) 所强调的那样。
- **Livecodebench 展示实力**：根据讨论，最新的 **livecodebench** 数据令人印象深刻，通过使用经典的 Leetcode 题目，其表现与 **Sonnet** 持平。
   - 然而，在处理新发布的库方面存在局限性，这些库通常不为 o1 模型所知。
- **AI 推理能力受到审视**：关于 **AI 推理**能力的讨论对比了 o1-mini 和 Qwen 2.5 等模型，评估了它们在避开反思型（reflection-type）任务时的表现。
   - 尽管目前的对比显示了 o1 的优势，但参与者对未来的改进表示乐观。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **排除 OpenInterpreter 错误**：一位用户在向 **OpenInterpreter** 输入数据时遇到问题，并请求详细的操作指南以解决该问题。建议通过 DM 发送错误信息以便获得更好的协助。
   - 这一事件凸显了社区对共享故障排除资源的需求。
- **Agent 性能的实操评估**：另一位用户已连续约一周积极测试 **OpenInterpreter** 的 Agent，表明其对各项功能有积极的参与。这种持续的评估反映了社区对 Agent 性能的兴趣。
   - 用户有动力通过积极使用和反馈来探索 OpenInterpreter 的潜力。
- **Perplexity 浏览器兼容性问题**：一位用户询问 **Perplexity** 是否被设置为默认浏览器，得到的确认是不是。多位用户报告遇到了类似的浏览器相关问题。
   - 一位用户指出在 **Windows** 上的 **Edge** 浏览器遇到了特定问题，这表明不同配置下的性能存在差异。
- **创新 RAG 聊天应用见解**：一位成员寻求开发针对 **PDF 交互**定制的 **RAG 聊天应用**的建议，重点在于管理包含文本和图像元素的回复。建议包括为图像使用 **tokens** 以及总结视觉内容以优化上下文使用。
   - 在讨论该应用的功能时，强调了有效整合各种数据类型的重要性。
- **开创性的图像和文本集成**：成员们讨论了在 PDF 回复中处理图像的策略，考虑使用 **base64 编码**等方法来增强数据检索。这种集成对于提高用户回复的准确性至关重要。
   - 分享的一个链接展示了一个在短短 **10 秒**内开发的令人印象深刻的 AI 作品，展示了该领域的飞速发展。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **OBS 仍然是屏幕录制的首选**：成员们讨论了将 **OBS** 作为屏幕录制的强大选项，尽管有些人更倾向于在处理缩放效果等任务时使用更简单的软件替代方案。
   - 一位用户强调他们一直坚持使用 OBS，而其他人则在寻找更简单的解决方案。
- **Screenity 作为一个用户友好的替代方案出现**：一位用户分享了 [Screenity](https://github.com/alyssaxuu/screenity)，这是一个免费且隐私友好的屏幕录像机，可以同时捕获屏幕和摄像头。
   - 该工具旨在迎合那些寻找比 OBS 更易上手的录制体验的用户。
- **Moshi 模型在语音对语音 (Speech-to-Speech) 应用中亮相**：成员们宣布发布了 **Moshi** 语音对语音模型，实现了文本 Token 与音频对齐的全双工语音对话。
   - 该基础模型拥有建模对话动态的特性，并在以 bf16 精度量化的 PyTorch 版本中实现。
- **GRIN MoE 在更少参数下表现出潜力**：围绕 [GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE) 展开了讨论，该模型仅凭 **6.6B 激活参数** 就表现出色，专注于编程和数学。
   - 它利用 **SparseMixer-v2** 进行梯度估计，避免了专家并行 (expert parallelism) 和 Token 丢弃 (token dropping)，这使其区别于传统的 MoE 方法。
- **Gemma2 在使用 DPO 数据时运行失败**：一位用户报告了 **Gemma2 9b** 在配合 **DPO 数据** 使用时的配置问题，遇到了一个 **TemplateError**，提示：*'Conversation roles must alternate user/assistant/user/assistant...'*。
   - 该错误源于使用了包含 'prompt' 而非必需的 'conversation' 的数据集结构。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **欢迎 PyTorch 会议的访客**：对 **PyTorch 会议** 的与会者表示热烈欢迎，营造了良好的社交和互动氛围。
   - 鼓励参与者在指定频道提出任何问题，以增强 **社区参与度**。
- **澄清会议直播的可用性**：有人询问是否有 **会议直播**，但成员们对其是否存在仍不确定。
   - 回复包括像 *‘Idk :/’* 这样模糊的情绪，反映了社区在这一问题上需要明确的信息。
- **GitHub PR 修复了 kv-Caching**：链接了标题为 **Fix kv-cacheing and bsz > 1 in eval recipe** 的 Pull Request，旨在解决关键的 kv-caching 问题，由 [SalmanMohammadi](https://github.com/pytorch/torchtune/pull/1622) 贡献。
   - 此修复对于提高性能至关重要，突显了 **Torchtune** 仓库的积极开发。
- **需要 HH RLHF 数据集文档**：讨论聚焦于 **HH RLHF 数据集** 缺乏文档的问题，并建议将其作为标准的偏好 (preference) 示例。
   - 这种观点认为适当的文档是必不可少的，正如 *‘Not sure, it should be exposed...’* 等评论所表达的那样。
- **默认偏好数据集构建器的计划**：关于 **默认偏好数据集构建器** 的公告引起了热烈反响，该构建器将利用 **ChosenToRejectedMessages**。
   - 参与者反应积极，评论如 *‘Dope’*，表明了对这一即将推出的功能的共同兴趣。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 程序优化成功**：一位成员庆祝他们在经过两个月的编码后，使用 **BSFSWRS 优化器** 取得了成功，展示了其在复杂 LM 设置中的有效性。
   - *未来是光明的，伙计们！*
- **Prompt 优化的成本风险**：有人担心为 DSPy 优化 Prompt 可能会产生 **极高的成本**，这表明了巨大的投资需求。
   - *优化一个 Prompt 的代价肯定非常昂贵。*
- **MIPRO 的财务风险**：一个幽默的观点建议将 **o1 与 MIPRO** 结合使用，同时警告该过程涉及的财务风险。
   - *这是通往破产的认证之路。*
- **DSPy 中 Bootstrapping 的澄清**：一位成员询问了关于 **Bootstrapping** 的问题，其重点是在 **LLM 的非确定性** 情况下生成 Pipeline 示例并验证其成功。
   - 他们对该方法在 LLM 行为下的运作方式表示困惑。
- **理解 Bootstrapping 的结果**：另一位用户解释说，Bootstrapping 在创建中间示例的同时，通过最终预测的成功来验证其正确性。
   - 如果最终结果正确，则中间步骤被视为有效的 Few-shot 示例。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **用户关注 tinybox 主板**：一位用户询问了 **tinybox red and green** 模型中使用的具体 **motherboard**（主板），寻求有关 **tinybox** 设备硬件细节的澄清。
   - 这反映了用户对硬件规格的持续关注，这对于优化性能至关重要。
- **CLANG 悬赏任务讨论升温**：成员们询问标题为“用 mmap 替换 CLANG dlopen + 移除链接步骤”的悬赏任务是否需要手动处理目标文件中的 **relocations**（重定位）。
   - 这表明社区正在深入探讨 **tinygrad** 与 CLANG 集成的技术细节。
- **分享优化 Pull Requests 链接**：一位用户分享了 **Pull Request #6299** 和 **#4492** 的链接，重点在于用 **mmap** 替换 **dlopen** 并实现 **Clang jit**。
   - 这些工作旨在提升性能，特别是在 **M1 Apple devices** 上，展示了社区对优化的承诺。
- **围绕 CLANG 悬赏的社区参与**：一位用户对谁能领取 CLANG 变更的 **bounty**（悬赏）表示兴奋，突显了社区的参与度。
   - 这种互动展示了成员们渴望看到贡献者成果的协作热情。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **OpenAI 的 o1 模型引发关注**：一段名为 [o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know](https://m.youtube.com/watch?v=KKF7kL0pGc4) 的 YouTube 视频对 **OpenAI o1** 的构建方式进行了精彩总结。
   - *即使是怀疑论者也将其称为“大推理模型” (large reasoning model)*，因为它具有独特的方法论以及对未来模型开发的影响。
- **o1 与其他模型的区别**：视频讨论了为什么 **o1** 被公认为 AI 建模的新范式，预示着设计理念的重大转变。
   - 采用此类模型的影响可能会让人们更好地理解 AI 的推理能力，使其成为该领域的一个关键话题。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LunoSmart 携 AI 产品上线**：Kosi Nzube 启动了他的 AI 创业项目 [LunoSmart](https://www.lunosmart.com)，专注于 AI 驱动的应用和**创新解决方案**。
   - 该项目旨在跨多个平台和设备类型提供**高效**且**智能的体验**。
- **展示多样化的技术栈**：Kosi 的应用使用了 **Java**、**Flutter**、**Spring Boot**、**Firebase** 和 **Keras**，展示了一个现代化的开发框架。
   - 在 Android 和 Web 端均可使用，增加了可访问性，扩大了用户覆盖面。
- **精通跨平台开发**：Kosi 擅长使用 **Flutter** 和 **Firebase SDK** 进行跨平台开发，提升了应用在不同设备上的性能。
   - 他在 **Android Studio** 和 **Java** 原生 Android 开发方面的专业知识为构建健壮的移动应用做出了贡献。
- **机器学习技能展示**：凭借自 **2019** 年以来的 **Machine Learning** 背景，Kosi 使用 **Keras**、**Weka** 和 **DL4J** 进行模型开发。
   - 他对推进 AI 技术的承诺奠定了 LunoSmart 计划的基础目标。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral 大幅降价**：[Mistral 的最新公告](https://mistral.ai/news/september-24-release/)透露了一项战略性降价，旨在提高用户和开发者的可及性。
   - 此举引发了关于竞争性定价如何影响市场格局和用户采用率的讨论。
- **市场对 Mistral 降价的反应**：价格调整在各大论坛引起了热烈反响，突显了 Mistral 试图迎合 AI 领域更广泛开发者群体的努力。
   - 许多行业观察者认为，这可能会加剧同类平台之间的竞争，从而促进创新。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1286039129892716554)** (344 条消息🔥🔥): 

> - `Perplexity AI 订阅限制`
> - `Perplexity 功能问题`
> - `AI 模型的使用及其性能`
> - `使用 DALL-E 3 生成图像`
> - `You.com 的用户体验` 


- **讨论了 Perplexity AI 订阅限制**：用户分享了 Perplexity 中各种 AI 模型的限制，包括 Claude 3.5 的 **600** 次查询、Opus 的 **60** 次以及 o1-mini 的 **10** 次，有时会导致对实际数值的困惑。
   - 一些用户报告了使用受限的问题，而另一些用户则对平台未达到预期表示不满。
- **Perplexity 功能问题的报告**：几位用户遇到了 Perplexity 网页版的问题，例如查询导致白屏或响应速度慢。
   - 建议包括刷新页面和清除缓存；一些用户发现该功能在手机上正常，但在桌面端不行。
- **AI 模型之间的性能比较**：讨论强调了用户的看法，即不同的模型（如 Claude 和来自 Poe 的模型）产生的响应相似且不尽如人意。
   - 用户担心与模型选择相关的承诺输出在实践中并未实现。
- **使用 DALL-E 3 生成图像**：用户询问如何使用 DALL-E 3 生成图像，一些用户报告说输入特定提示词后没有立即显示结果。
   - 经过一些排障后，用户发现使用特定提示词是有效的，尽管过程被描述为缓慢。
- **You.com 的用户体验**：用户分享了对 You.com 的复杂感受，特别是关于其改进的功能以及 o1 的消息限制（设定为 **每天 20 条**）。
   - 一些用户提到与之前的体验相比，易用性更好且功能集成更完善，但仍对模型选择和整体服务质量表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180">Holo Spice And Wolf GIF - 贤狼赫萝 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/aravsrinivas/status/1836541821310189907?s=46">Aravind Srinivas (@AravSrinivas) 的推文</a>：好的。我们也会做一个 macOS 应用。大家都在要求。是时候了。敬请期待。引用 TestingCatalog News 🗞 (@testingcatalog) 很快吗？👀👀👀</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj)">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://x.com/apostraphi/status/1836491868093436179?s=61">Phi Hoang (@apostraphi) 的推文</a>：好奇心引领方向</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - 奔向无限</a>：欢迎回到学校！仅限两周，领取一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把那个免费月升级为一整年免费...</li><li><a href="https://tenor.com/view/obiwan-kenobi-disturbance-in-the-force-star-wars-jedi-gif-10444289">Obiwan Kenobi Disturbance In The Force GIF - 欧比旺·肯诺比 原力觉醒 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1286072807762952274)** (4 messages): 

> - `Snap 的 Large AR Spectacles`
> - `CATL 的百万英里电池`
> - `生死之外的“第三状态”`
> - `关于洗衣机改装的讨论`
> - `关于多元宇宙的多样化见解` 


- **Snap 发布 Large AR Spectacles**：Snap 推出了其 **Large AR Spectacles**，旨在提供沉浸式体验，展示了增强现实技术的未来。
   - 这一**创新**引发了关于提升用户参与度以及游戏应用潜力的讨论。
- **CATL 发布百万英里电池**：CATL 宣布推出**百万英里电池**，承诺提供**超过一百万英里**的续航里程，为电动汽车（EV）的长期可持续性提供了解决方案。
   - 专家认为这一突破是**电动汽车市场**和未来**汽车能源解决方案**的颠覆者。
- **探索生死之外的“第三状态”**：讨论围绕着**“第三状态”**（Third State）概念展开，理论化了超越传统生死维度的体验，详见此 [YouTube 视频](https://www.youtube.com/embed/n16AKOF43ag)。
   - 这一话题引起了许多人的兴趣，拓展了我们理解存在和意识的边界。
- **如何改装你的洗衣机**：分享了一份关于**改装洗衣机**创新方法的指南，强调了实用技巧和 DIY 技术。
   - 这些见解旨在提高机器效率和用户体验，吸引了家庭自动化爱好者的关注。
- **多元宇宙理论见解**：深入探讨了**多元宇宙理论**，并对其对技术和人类的潜在影响提出了见解。
   - 这一探索激发了人们对**多元宇宙理解**在未来技术和科学中应用的兴奋感。



**提及的链接**：<a href="https://www.youtube.com/embed/n16AKOF43ag">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1286049037572636774)** (7 messages): 

> - `无效的 PPLX 模型问题`
> - `过时的模型链接`
> - `Sonar 模型转型`
> - `Prompt 中的媒体文章链接` 


- **遇到无效模型错误**：一位用户报告了 **无效模型 'pplx-7b-online'** 的问题，称所有其他模型都正常工作，唯独这个不行。
   - 他们还提到，在尝试访问模型卡片（model cards）时被重定向到了 **Perplexity 文档主页**。
- **分享了更新后的模型卡片链接**：一位成员指出，原始的模型信息链接已过时，并提供了一个指向更新后的 [模型页面](https://docs.perplexity.ai/guides/model-cards) 的新链接。
   - 此更新链接包含所有可用模型及其详细信息的列表。
- **PPLX 模型不再受支持**：根据社区反馈，**pplx 模型现已过时**，并已更名为 **sonar** 模型。
   - 这引发了关于未来新 sonar 模型的可用性和支持的问题。
- **Sonar 模型响应的挑战**：一位用户询问如何让 **sonar 模型** 返回它们获取信息的媒体文章链接。
   - 在进行了数小时的 Prompt 调试仍无结果后，他们向社区寻求帮助。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/model-cards">支持的模型 - Perplexity</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/home.">未找到标题</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/guides/model-cards>">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1286063565857620032)** (327 条消息🔥🔥): 

> - `Qwen Model 性能`
> - `模型兼容性问题`
> - `API 连接`
> - `图像处理挑战`
> - `LM Studio 的网络配置` 


- **Qwen Model 根据图像大小表现出不同的性能**：用户讨论了在使用 Qwen Model 处理某些小型长方形图像时出现的崩溃问题，表明宽高比可能会影响模型性能。
   - 一些用户报告称，Prompt 的有效性因图像质量而异，调整 System Prompt 可以防止崩溃。
- **加载模型出错**：一位用户在尝试在 LM Studio 中加载模型时遇到了表示 Tensor 形状不匹配的错误，据指出该模型不被 llama.cpp 支持。
   - 另一位用户观察到并非所有模型都能正常运行，强调了对兼容模型格式的需求。
- **LM Studio 与 CrewAI 之间的连接**：一位用户在将其代码中的 Provider 名称更改为 'openai' 后，成功将 LM Studio API 与 CrewAI 连接。
   - 建议其他人在与 CrewAI 交互时检查 Embedding Model 的兼容性问题。
- **光学字符识别 (OCR) 的挑战**：用户注意到模型中 OCR 能力的效果各异，尤其是在处理不同尺寸的图像时。
   - 大家达成共识，认为尺寸较大且比例适当的图像比小型图像能产生更好的 OCR 结果。
- **优化性能的网络配置**：用户建议切换到 IPv4，以解决在 LM Studio 中从 Hugging Face 加载模型时的问题。
   - 一位用户寻求网络配置方面的帮助，并获悉了在 macOS 上调整设置的过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/getting-started/introduction">Introduction - Open Interpreter</a>：未找到描述</li><li><a href="https://support.apple.com/en-gb/guide/mac-help/mh14129/mac">在 Mac 上更改 TCP/IP 设置</a>：在 Mac 上，使用 TCP/IP 网络设置来配置 IPv4 或 IPv6 连接，或续订 DHCP 租约。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fjxkxy/qwen25_a_party_of_foundation_models/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=xyKEQjUzfAk">廉价迷你机运行 70B LLM 🤯</a>：我在这台微型迷你 PC 中安装了 96GB RAM，并在上面运行了 Llama 70B LLM。椅子：Doro S100 Chair - 享受 6% 折扣：YTBZISUSA&amp;CA：https://sihoooffice.com/DoroS100-AlexZ...</li><li><a href="https://beautiful-soup-4.readthedocs.io/en/latest/">Beautiful Soup 文档 &mdash; Beautiful Soup 4.4.0 文档</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1286061186487422997)** (26 条消息🔥): 

> - `M4 Mac Mini 预期`
> - `macOS 上的 RAM 占用`
> - `GPU 降压 (Undervolting)`
> - `NPU 选项`
> - `气流与热管理` 


- **M4 Mac Mini 预期升温**：用户们正期待即将推出的 **M4 Mac Mini**，希望提供 **16 GB** 和 **32 GB** RAM 的选项，部分用户对其价格以及相对于当前型号的性能表现表示担忧。
   - *Anester* 建议，对于推理任务，二手的 **M2 Ultra/Pro** 可能比预测价格更高的 M4 新机型更具性价比。
- **macOS RAM 占用受到关注**：讨论强调了 **macOS** 即使在未通过图形界面登录（仅通过 SSH）的情况下，其界面也可能消耗约 **1.5 至 2 GB** 的 RAM。
   - 内存管理问题被提及，根据用户在升级到 macOS Sequoia 15.0 期间的经验，闲置占用可能会达到 **6 GB**。
- **探索 GPU 降压 (Undervolting) 的益处**：分享了关于 **GPU 降压** 以降低功耗和发热的建议，特别是对于运行 **3090** 等高性能显卡的用户。
   - 用户指出，通过关闭 turbo-boost 技术可以减少热节流 (thermal throttling)，并考虑将此技术用于平衡发热与性能。
- **可供用户选择的 NPU 选项**：一位用户提到 **Tesla P40**、**P4** 和 **T4** 可以作为计算和 AI 任务的选项，将它们视为没有视频输出的 GPU 替代方案。
   - 这些 NPU 适用于 ML/DL 应用，提供精简的性能，且没有传统 GPU 的额外开销。
- **管理 GPU 功耗**：关于 GPU **功耗管理** 的讨论包括使用 `nvidia-smi -pl` 设置功率限制以控制瓦数，以及探索 **降压** 以实现更好的降温和整体稳定性。
   - 对话深入比较了在 **3090** 上使用这些方法的效果，引发了关于如何平衡功率限制与时钟频率以优化性能的讨论。



**提到的链接**：<a href="https://www.reddit.com/r/macbookair/comments/1fjl8kr/just_upgraded_to_macos_sequoia_150_idle_ram_usage/?rdt=64991">Reddit - 深入了解</a>：未找到描述

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1286418710332182529)** (1 条消息): 

> - `Tokenized 标题`
> - `Unity ML Agents 预训练`
> - `GSM8K 推理数据集`
> - `Padoru 数据集`
> - `Gradio 与 R 示例` 


- **聚焦 Tokenization**：由认证用户发布的一篇题为《[这个标题已经 Tokenized 了](https://huggingface.co/blog/apehex/this-title-is-already-tokenized)》的文章强调了 Tokenization 在 AI 模型中的重要性。
   - 作者强调应使 Tokenization 更加易于理解，以便更好地进行模型训练。
- **使用 Unity ML Agents 进行预训练**：了解如何使用 [Unity ML Agents](https://youtube.com/live/0foHMTPWa4Y?feature=share) *从零开始预训练 LLM*，由社区成员展示。
   - 这种交互式方法利用 Sentence Transformers 来促进模型训练。
- **推出 GSM8K 推理数据集**：分享了一个基于 GSM8K 的新推理 [数据集](https://huggingface.co/datasets/thesven/gsm8k-reasoning)，旨在增强模型在推理任务上的表现。
   - 该数据集是模型测试和开发的重要资源。
- **令人兴奋的 Padoru 数据集亮相**：一名成员发布了一个有趣的 [Padoru 数据集](https://huggingface.co/datasets/not-lain/padoru)，为节日主题的 AI 项目做贡献。
   - 该数据集旨在激发节日期间的创意 AI 应用。
- **Gradio R 语言集成示例**：发布了一个在 Gradio 中使用 R 语言的示例，展示了其在增强用户界面方面的集成，详见 [此处](https://github.com/egorsmkv/r-with-gradio)。
   - 这突显了 Gradio 在跨不同编程语言的应用开发中的多功能性。



**提到的链接**：<a href="https://medium.com/@visrow/ai-multi-agent-system-in-java-and-fipa-standards-f0a4d048c446)">未找到标题</a>：未找到描述

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1286043974313705503)** (123 条消息🔥🔥): 

> - `Qwen 数学模型 Demo`
> - `PyTorch Conference 参会情况`
> - `生成式文本 AI 与语法提取`
> - `最适合角色扮演的 LLM`
> - `设置本地语音聊天` 


- **Qwen 数学模型 Demo 获得好评**：成员们对最近发布的 [Qwen/Qwen2.5 Math Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo) 表现出极大的热情，强调了其令人印象深刻的能力。
   - 一位成员敦促其他人尝试一下，称其结果“好得令人难以置信”。
- **Hugging Face 参加 PyTorch Conference**：关于参加在旧金山举行的 PyTorch Conference 的问题被提出，成员们渴望与社区成员见面并参加相关活动。
   - 在交流中，一位成员确认他们已在现场，并鼓励其他人加入聚会。
- **生成式 AI 从数据集中学习语法**：讨论者探讨了生成式文本 AI 如何学习语法，断言像 GPT 这样的模型是为序列分配概率，而不是依赖传统的语法模型。
   - 引用了关于从语言模型中提取语法的研究，并建议研究语言信息的整合。
- **确认最适合角色扮演的 LLM 模型**：成员们推测，像 GPT-4 这样的 LLM 可能最适合角色扮演，因为它们具有出色的创意写作能力，并对其他潜在模型提出了建议。
   - 一位成员分享了一个 GitHub Gist，列出了用于生成随机角色的引擎测试，表明了评估创意水平的初步尝试。
- **使用 AI 设置本地语音聊天**：一位成员询问如何创建类似于 character.ai 的本地语音聊天，另一位成员提供了一个结合转录模型、LLM 和 TTS 的解决方案。
   - 这引发了人们对 AI 技术在交互体验中实际应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/julien_c/status/1836688023821385827">来自 Julien Chaumond (@julien_c) 的推文</a>: 我的联合创始人 @ClementDelangue 参加了 Generation DIY 播客（这是法国的一个大型播客）https://www.gdiy.fr/podcast/clement-delangue-2/</li><li><a href="https://huggingface.co/spaces/InstantX/InstantID">InstantID - InstantX 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://pytorch.org/docs/stable/notes/mps.html">MPS 后端 &mdash; PyTorch 2.4 文档</a>: 未找到描述</li><li><a href="https://x.com/osanseviero/status/1834508940417040487">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 这是 Hugging Face 团队为下周的 PyTorch Conference 做准备的情况🤗 很快见，来参加我们的派对领取精美周边！</li><li><a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo">Qwen2.5 Math Demo - Qwen 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/exceeded-gpu-quota/107022/7">超出 GPU 配额</a>: 我需要要求我的用户在应用程序中使用“他们的” hf 账号登录吗？是的。这种方法可能是 HF 假设的基础。我不知道是否可以尝试使用你的...登录</li><li><a href="https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On">Kolors Virtual Try-On - Kwai-Kolors 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/1905.05950">BERT 重新发现经典的 NLP 流水线</a>: 预训练文本编码器迅速提升了许多 NLP 任务的最先进水平。我们专注于其中一个模型 BERT，旨在量化语言信息在网络中的捕获位置...</li><li><a href="https://gist.github.com/Getty/f5a6ebdea7de441215e4a8cd546f5cb8">gist:f5a6ebdea7de441215e4a8cd546f5cb8</a>: GitHub Gist: 立即分享代码、笔记和片段。</li><li><a href="https://x.com/NousResearch>">来自 GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter</li><li><a href="https://x.com/isidentical>">来自 GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1286151782892044322)** (5 messages): 

> - `Hugging Face beginner resources` (Hugging Face 入门资源)
> - `AI training processes` (AI 训练流程)
> - `Multimodal AI training` (多模态 AI 训练) 


- **探索 Hugging Face 入门资源**：一位成员询问了学习 [Hugging Face](https://huggingface.co/) 的起点，并询问是否有初学者项目板块。
   - 另一位成员通过询问其兴趣在于文本、图像、音频还是强化学习来提供指导。
- **为了 AI 训练而对 AI 进行元训练 (Meta-Training)**：出现了一个幽默的评论，提到需要用 AI 来训练 AI，进而再训练另一个用于 AI 训练目的的 AI。
   - *这段对话为复杂的 AI 训练话题增添了轻松的基调。*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

maxstewart.: Hello
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1286094241637994547)** (215 messages🔥🔥): 

> - `Composite Embeddings`
> - `Neural Decompiler`
> - `TensorFlow vs PyTorch`
> - `ML Agent Integration`
> - `Real-Time Visualization Tools` 


- **复合嵌入 (Composite Embeddings) 消除 Tokenization**：一位成员讨论了创建一个名为 **'composite embeddings'** 的新层，该层消除了对 Tokenization 的需求，从而实现对文本更精细的理解。
   - 这种方法旨在通过利用现有的 Embedding 熟悉度来增强 **LLMs**，从而可能改进模型处理新颖组合的方式。
- **分享神经反编译器 (Neural Decompiler) 愿景**：一位成员正在开发一种 **neural decompiler**，旨在将二进制或汇编代码翻译回源代码，并表达了尽量减少传统 Tokenization 的动机。
   - 他们寻求以类似于汇编和 Token 词汇表工作的方式来开发这一概念，但采用神经方法。
- **关于 TensorFlow 和 PyTorch 的辩论**：参与者讨论了 **TensorFlow** 与 **PyTorch** 的优缺点，指出 TensorFlow 的 API 虽然陈旧，但具有强大的指标 (metrics) 能力。
   - 大家达成共识，尽管存在一些缺点，TensorFlow 仍然很有用，特别是在从数据集中提取词汇表方面。
- **对实时可视化的兴趣**：一位成员赞赏了一篇文章中的点云图，并表达了希望为自己的工作创建类似的实时可视化。
   - 推荐使用 **TensorBoard** 和降维工具（如 **UMAP** 和 **PCA**）来可视化高维向量。
- **构建协作学习社区**：强调了社区参与的重要性，成员们分享了见解和错误，并指出错误往往能提供宝贵的教训。
   - 对话还强化了协作的重要性，建议成员在 **dev.to** 和 **Hugging Face** 等平台上分享文章。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dev.to/p3ngu1nzz/tau-llm-series-enhancements-and-debugging-part-18-19-n01">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/blog/apehex/this-title-is-already-tokenized">This Title Is Already Tokenized (Tokun P.2)</a>：未找到描述</li><li><a href="https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_ppo_A3_2M/TauAgent">p3nGu1nZz/Tau at main</a>：未找到描述</li><li><a href="https://github.com/egorsmkv/r-with-gradio">GitHub - egorsmkv/r-with-gradio: Use R with Gradio</a>：在 Gradio 中使用 R。通过在 GitHub 上创建一个账号来为 egorsmkv/r-with-gradio 做出贡献。</li><li><a href="https://huggingface.co/datasets/not-lain/padoru">not-lain/padoru · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1286237251977613332)** (4 messages): 

> - `Llava OneVision model suffixes`
> - `reCAPTCHA v2 success rate`
> - `Model questions on HF Hub` 


- **对 Llava OneVision 模型后缀的好奇**：成员们讨论了 **Llava OneVision** 模型名称中的 `ov` 和 `si` 后缀，推测 `ov` 表示 one-vision 训练阶段，而 `si` 指的是 single-image 中间阶段。
   - 建议直接在 [HF Hub 上的 Community 标签页](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe) 询问模型创建者以获得澄清。
- **新论文报告 reCAPTCHA v2 取得突破**：一篇新论文揭示，**reCAPTCHA v2** 现在解决验证码的 **成功率达到 100%**，相比之前的 **68-71%** 有了巨大飞跃。
   - 该研究利用先进的 **YOLO 模型** 进行评估，并指出目前的 AI 可以有效地破解基于图像的 CAPTCHA，揭示了其对 cookie 和浏览器历史数据进行用户验证的依赖。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.08831">Breaking reCAPTCHAv2</a>：我们的工作研究了采用先进机器学习方法解决 Google reCAPTCHAv2 系统验证码的功效。我们评估了自动化系统在解决验证码方面的有效性...</li><li><a href="https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe">LLaVA-Onevision - a llava-hf Collection</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/)** (1 messages): 

pseudoterminalx: 可能就像 ipadapter 配合一个在该风格上进行过微调（可能带有 lora？）的基础模型。
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1286127404145840289)** (9 messages🔥): 

> - `Mojo roadmap updates`
> - `Community meeting invitations`
> - `OpenCV-Python installation issues` 


- **Mojo 路线图仍缺少日期**：针对 Modular 网站上的 **Mojo roadmap & sharp edges** 提出了担忧，特别是由于缺少日期导致其参考价值降低。
   - 有人指出，部分功能已更新，而其他功能仍未改动，且 **magic cli** 已取代 **modular cli**。
- **社区会议报名**：一名成员邀请其他人在定于 **9 月 23 日** 举行的下一次社区会议上进行演示（如果有足够的内容）。
   - 他们鼓励成员通过 ping 或在线程中回复来表达兴趣，并提到如果需要可能会推迟会议。
- **在 magic 中添加 OpenCV-Python 时遇到问题**：一名用户尝试将 **opencv-python** 添加到 magic 环境中，但遇到了表示无法解决 conda 依赖的错误。
   - 另一名成员建议该用户在相应的频道中分享问题以获取进一步帮助。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1286434243412492389)** (1 messages): 

> - `Closure of GitHub Discussions`
> - `Transition to Discord for community interactions`
> - `Conversion of discussions to issues` 


- **GitHub Discussions 将于 9 月 26 日关闭**：我们将于 **9 月 26 日** 关闭 [Mojo](https://github.com/modularml/mojo/discussions) 和 [MAX](https://github.com/modularml/max/discussions) 仓库中的 GitHub Discussions。
   - 鼓励成员改在我们的 Discord 服务器中分享问题和功能请求。
- **重要讨论将转换为 Issue**：任何评论超过 **10 条** 且被认为重要的 GitHub Discussions 将在关闭前转换为 **GitHub Issues**。
   - 成员可以通过在线程中标记作者来请求转换特定的讨论，以便他们采取行动。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1286082793733881919)** (178 messages🔥🔥): 

> - `Mojo SDK for Windows`
> - `Decorator Reflection in Mojo`
> - `Safety and Correctness in Mojo`
> - `Variable Bit Width Integers`
> - `Packed Structs in Mojo` 


- **Mojo SDK for Windows 面临开发挑战**：成员们讨论了 Mojo SDK for Windows 的艰难现状，主要是由于跨操作系统环境迁移的复杂性，并建议使用 WSL 作为权宜之计。
   - 一个讨论帖强调了该任务的难度，原因在于与设备驱动程序的底层交互，这比预期的要复杂得多。
- **Mojo 尚未实现 Decorator Reflection**：成员们确认 Mojo 路线图中概述的 Decorator Reflection 目前尚不可用，并推测它将实现强大的 MLIR 访问能力。
   - 讨论集中在 Decorator 的潜力上，即提供一种在编译时反射和操作 MLIR 的手段。
- **Mojo 优先考虑安全性与正确性**：讨论强调，虽然安全性和正确性是 Mojo 的主要优先级，但性能也被视为至关重要，这导致了权衡。
   - 成员们提到，安全性是设计选择的指导原则，但确保开发者的实际可用性也是一个重点。
- **可变位宽整数 (Variable Bit Width Integers) 带来挑战**：关于在 Mojo 中使用可变位宽整数的问题被提出，特别是在实现 TCP/IP 等任务时。
   - 成员们建议使用位运算符作为替代方案，但指出这会损害 API 的易用性 (ergonomics)。
- **Mojo 缺乏对 Packed Structs 的支持**：讨论了 Mojo 对 Packed Structs 的需求，以支持位域 (bit fields) 并提高数据结构的可用性。
   - 参与者推测使用 LLVM 来管理数据表示，但对依赖字段对齐 (field alignment) 的自动处理表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>：在此页面中，我们将向您展示如何运行一些示例项目。</li><li><a href="https://blog.rust-lang.org/2024/08/12/Project-goals.html">Rust Project goals for 2024 | Rust Blog</a>：赋能每个人构建可靠且高效的软件。</li><li><a href="https://docs.modular.com/mojo/manual/decorators/parameter">@parameter | Modular Docs</a>：在编译时执行函数或 if 语句。</li><li><a href="https://en.wikipedia.org/wiki/Agda_(programming_language)">Agda (programming language) - Wikipedia</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decor">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://zackoverflow.dev/writing/unsafe-rust-vs-zig/">When Zig is safer and faster than Rust</a>：网上关于 Rust vs. Zig 的争论层出不穷，这篇文章探讨了我认为未被充分提及的论点。</li><li><a href="https://www.youtube.com/watch?v=q8qn0dyT3xc">Oxidize Conference: How Rust makes Oxide possible</a>：随着 Rust 在生产环境中的使用越来越多，人们谈论的许多例子都相当高级：比如 Web 应用程序。虽然这很好，但……</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decorator-reflection">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://mojodojo.dev/mojo-team-answers.html#unsafe-code">Mojo Team Answers | Mojo Dojo</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/620">[Feature Request] Native Windows support · Issue #620 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。您的请求是什么？对 Windows 的原生支持。什么时候可用？……
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1286314350390411325)** (1 messages): 

> - `AI Development on Laptops`
> - `MAX Cloud Computational Offering`
> - `Cost-Effective GPU Solutions` 


- **在带有 iGPU 的笔记本电脑上运行 ML 项目**：一场关于在带有集成 GPU 的笔记本电脑上运行 ML 项目可行性的讨论浮出水面，认为**大多数现代计算机**应该足以胜任许多任务，正如在 60 和 70 年代所见的那样。
   - *一些沉重的 ML 任务可能仍需要 GPU 集群*，但基础项目可以在本地有效运行。
- **MAX Cloud 服务提案**：提出了 **“MAX Cloud” 服务**的概念，允许开发者远程执行繁重的计算，同时在本地处理常规开发。
   - 这种双重方法可以增强开发者体验，同时在需要时提供对 **GPU 资源**的访问。
- **倡导自托管计算解决方案**：有人建议推出 MAX 服务的自托管版本，使公司能够使用自己的 GPU 服务器进行本地计算。
   - 这种方法可以带来**显著的成本节约**，允许用户仅为项目中使用的计算资源付费。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1286042514742185995)** (181 messages🔥🔥): 

> - `Lionsgate partnership with RWML`
> - `Stability AI model comparisons`
> - `Flux model capabilities`
> - `Training LoRA and checkpoints`
> - `Model performance and aesthetics` 


- **Lionsgate 与 RWML 的战略举措**：**RWML** 与 **Lionsgate** 之间的合作伙伴关系引发了关于 Lionsgate 价值及其依靠 AI 削减成本的讨论，因为他们正努力在好莱坞保持地位。
   - *Lionsgate 最近的作品受到了严厉批评*，一些成员将他们目前的策略比作好莱坞早期在 CGI 上的失误。
- **对比 Stable Diffusion 模型**：用户对比了 **Flux** 和 **SD3 Medium**，指出 **Flux** 生成的输出质量更好，但如果提示词（Prompt）不当，可能会有“塑料感”。
   - 几位成员一致认为，虽然 **Flux** 比 **SD3** 更有优势，但后者因速度和效率而受到称赞，特别是对于基础图像生成。
- **探索 Flux 模型的功能**：讨论了 **Flux 模型**产生具有高提示词遵循度的令人印象深刻的图像的能力，即使它有时会优先考虑某些美学风格。
   - 关于其处理各种主题的能力，评价褒贬不一，包括该模型在用户画廊中对 NSFW 内容的关注。
- **为特定风格训练 LoRA 等模型**：成员们讨论了训练 **LoRA** 或 Checkpoints 以复制特定艺术家风格的可能性，强调需要来自艺术家原创作品的大型数据集。
   - 社区分享了利用现有框架定制模型以获得独特艺术输出的见解。
- **生成图像的真实感**：指出 **Flux** 和 **SD3** 都能创建照片级真实的图像，如果提示词不具体，Flux 通常更倾向于更真实的输出。
   - 用户鼓励将不同的 **LoRA** 模型与 Flux 结合使用，以增强真实感并在生成的图像中获得更好的效果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/runwayml/status/1836391272098988087">来自 Runway (@runwayml) 的推文</a>: 今天我们很高兴地宣布，我们已经与 @Lionsgate 建立了首个此类合作伙伴关系，将我们的下一代叙事工具交到世界上最伟大的叙事者手中...</li><li><a href="https://huggingface.co/nyanko7/flux-dev-de-distill">nyanko7/flux-dev-de-distill · Hugging Face</a>: 未找到描述</li><li><a href="https://civitai.com/images/25279078">49RpK5dY 发布的图像</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1eon9n7/flux_its_amazing_at_creating_silly_children_book/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://research.google/blog/rich-human-feedback-for-text-to-image-generation/">文本到图像生成的丰富人类反馈</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?sort=trending&search=Flux>">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/chufengxiao/SketchHairSalon">GitHub - chufengxiao/SketchHairSalon: SketchHairSalon 项目：基于深度草图的发型图像合成 (SIGGRAPH Asia 2021)</a>: SketchHairSalon 项目：基于深度草图的发型图像合成 (SIGGRAPH Asia 2021) - chufengxiao/SketchHairSalon
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1286053278324490353)** (152 条消息🔥🔥): 

> - `NousCon 亮点`
> - `AI 模型进展`
> - `AI 就业预测`
> - `社区讨论`
> - `活动参与` 


- **NousCon 活动圆满成功**：参与者对 [NousCon](https://x.com/NousResearch/status/1831032559477866754) 上引人入胜的演讲者和富有洞察力的内容表示感谢。许多参与者计划参加未来的活动，并对建立人脉的机会表示赞赏。
   - 一些成员询问在哪里可以找到演示文稿的幻灯片，并被引导至各个独立的演讲，这展示了社区对知识共享的兴趣。
- **对 AI 模型进展的热情**：参与者讨论了 **qwen2.5** 和 **o1** 的功能，一些人注意到其令人印象深刻的性能以及设置过程中的挑战。其他人将其与 **q3_k_xl** 等较小模型进行了比较，强调了模型理解方面的进步。
   - 成员们对账户可用的免费查询次数表示担忧，一些用户分享了他们在不同 AI 模型之间切换的经验。
- **2027 年 AI 对就业影响的预测**：一位成员提出了关于 AI 到 2027 年可能影响多少工作岗位的问题，重点关注机器人和自动驾驶技术等领域。回复中包括对制造业许多工作岗位可能受到影响的推测，反映了围绕 AI 经济影响的持续对话。
   - 讨论还包括关于当前工具可能无法完全促进向 AGI 状态过渡的想法，并对潜在工作重组的时间表进行了辩论。
- **社区参与和咨询**：成员们积极询问有关 **Forge** 等 AI 技术的问题，希望能找到更深入理解的资源。这说明了社区致力于提高知识水平并积极参与 AI 话题。
   - 个人表达了对指南和文档的需求，以帮助应对复杂的 AI 项目，表明了一个支持性的学习环境。
- **参与 AI 活动**：几位用户对未能参加 NousCon 表示遗憾，凸显了该活动在社区中极高的声誉。关于下次聚会的讨论以及参与者的旅行经历强调了对未来活动的期待。
   - 参与者讨论了 AI 相关活动，一些人因从不同地点赶来的旅行经历而产生共鸣，培养了同僚情谊。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NousResearch/status/1831032559477866754">来自 Nous Research (@NousResearch) 的推文</a>：NousCon，9 月 18 日，旧金山，名额有限。https://lu.ma/zlgp0ljd</li><li><a href="https://tome.app/k4don/f-cm188fgq10fhn7xgq8bz93udc">Tome</a>：未找到描述</li><li><a href="https://x.com/altryne/status/1836581142847463752?t=D61slueJ-CrAwzSN4yjgUg&s=19">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：即将开启 @NousResearch NousCon！我听说会有一些重大发布！🔥 将努力为大家更新 👀 与 @karan4d @Teknium1 @theemozilla @shivani_3000 以及将近...在一起</li><li><a href="https://x.com/AISafetyMemes/status/1836826422477721684?t=5cPSLkpOnyf-G47R__jpTw&s=19">来自 AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes) 的推文</a>：抱歉，但在现阶段，如果还说 AGI 肯定在几十年后，那就太尴尬了。就在过去的一周里：1) 一位顶尖生物医学科学家说 o1 达到了博士水平 2) 一位顶尖数学家说 o1 ...</li><li><a href="https://news.lambdalabs.com/news/today">ML Times</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=3jhTnk3TCtc">Tech Bros Inventing Things That Already Exist</a>：广告：🔒 在 https://joindeleteme.com/BOYLE 从网络上移除您的个人信息，并使用代码 BOYLE 享受 20% 折扣 🙌 DeleteMe 国际计划：https:/...</li><li><a href="https://github.com/k4yt3x/video2x?tab=readme-ov-file">GitHub - k4yt3x/video2x: 一款通过 waifu2x, Anime4K, SRMD 和 RealSR 实现的无损视频/GIF/图像放大工具。始于 2018 年 Hack the Valley II。</a>：一款通过 waifu2x, Anime4K, SRMD 和 RealSR 实现的无损视频/GIF/图像放大工具。始于 2018 年 Hack the Valley II。 - k4yt3x/video2x</li><li><a href="https://x.com/">来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord, Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord, Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1286041555429490738)** (13 条消息🔥): 

> - `Ollama local API setup` (Ollama 本地 API 设置)
> - `Speech-to-Text APIs` (语音转文本 API)
> - `Hermes-3 model function calling` (Hermes-3 模型函数调用)
> - `Whisper integration` (Whisper 集成)
> - `Model precision preferences` (模型精度偏好)


- **Ollama 在本地利用本地 API**：一位成员指出，如果你在本地拥有模型，可以设置 [Ollama](https://ollama.com) 来使用本地 API。
   - 这取决于你想要运行的模型大小。
- **Deepgram 提供出色的 STT 免费计划**：讨论强调 **Deepgram** 的免费计划非常适合语音转文本 (STT) 需求，特别是基于使用限制。
   - 一位成员建议将 **Whisper** 设置为类似 API 的功能，并提到即使没有 GPU，它也具有多功能性，尽管拥有 GPU 会有所帮助。
- **Hermes-3 函数参数求助**：一位成员在 **Hermes-3** 的 `functions.py` 文件中寻求帮助，以添加一个接受多个参数的函数工具。
   - 他们提供了其 [仓库链接](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) 作为背景，希望能得到社区其他成员的建议。
- **关于 Hermes-3 的支持请求**：另一位成员建议在有空时联系 [@909858310356893737](https://discordapp.com/users/909858310356893737) 以获取关于 Hermes-3 的帮助。
   - 原帖作者表示感谢，并表示将等待回复。
- **模型精度设置查询**：一位参与者讨论了在没有累积后截断问题的情况下，以全 **bf16** 和 **fp32** 分辨率运行 **405b** 模型。
   - 这强调了对模型精度进行精细控制以获得最佳性能的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://replicate.com/openai/whisper">openai/whisper – 在 Replicate 上通过 API 运行</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账户来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286055344140845058)** (6 条消息): 

> - `Shampoo optimization` (Shampoo 优化)
> - `Diagram of Thought framework` (Diagram of Thought 框架)
> - `ReST-MCTS* paper` (ReST-MCTS* 论文)
> - `Reverse engineering O1` (逆向工程 O1)


- **优化任务中的 Shampoo 与 Adam**：最近的研究发现，**Shampoo**（一种高阶预条件方法）在深度学习优化中比 **Adam** 更有效，尽管与 Adam 简单的平均更新相比，它引入了额外的超参数和计算开销。
   - 该研究将 Shampoo 与 **Adafactor** 联系起来，揭示了一种名为 **SOAP** 的新型高效算法，利用了 Shampoo 预条件器的特征基。
- **引入 Diagram of Thought**：**Diagram of Thought (DoT)** 框架将 LLM 中的迭代推理建模为有向无环图 (DAG)，允许在不丢失逻辑一致性的情况下进行复杂推理，这与传统的线性方法形成对比。
   - 每个节点代表一个提出或批评的想法，使模型能够通过语言反馈迭代地改进推理。
- **对逆向工程 O1 的兴趣**：成员们对 **逆向工程 O1** 表现出浓厚兴趣，表明了进一步探索该领域的协作精神。
   - 一位成员提到正在阅读重叠的论文，并强调了他们对 **ReST-MCTS*** 论文的探索。
- **ReST-MCTS* 被低估**：一位成员认为集成了 STaR、PRM 和 MCTS 的 **ReST-MCTS*** 论文被低估了，尽管它采用了一种无缝的方法来结合这些方法。
   - 他们渴望进一步讨论这篇论文，并与对类似话题感兴趣的人分享见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: 使用 Adam 改进并稳定 Shampoo</a>：越来越多的证据表明，在深度学习优化任务中，Shampoo（一种高阶预条件方法）比 Adam 更有效。然而，Shampoo 的缺点包括额外的超参数...</li><li><a href="https://arxiv.org/abs/2409.10038v1">On the Diagram of Thought</a>：我们引入了 Diagram of Thought (DoT)，这是一个将大语言模型 (LLM) 中的迭代推理建模为在单个模型内构建有向无环图 (DAG) 的框架。不同于...
</li>
</ul>

</div>

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1286111326996922410)** (2 messages): 

> - `Characterfile` 格式
> - 多 Agent 框架工具
> - Twitter 归档转角色数据 


- **发布用于角色数据的 Characterfile**：[characterfile](https://github.com/lalalune/characterfile) 项目为**角色数据**提供了一种简单的文件格式，旨在促进在多 Agent 框架下工作的开发者之间的共享。
   - 它包含 **Python** 和 **JavaScript** 的示例和验证器，以及像 *tweets2character* 这样从 Twitter 归档生成角色文件的脚本。
- **关于 Characterfile 相关性的推文**：[_akhaliq](https://twitter.com/_akhaliq/status/1836544678742659242) 的一条推文强调了开发团队内部结构化角色数据的重要性。
   - 该推文强调了在管理角色信息和协作共享中建立标准的必要性。



**Link mentioned**: <a href="https://github.com/lalalune/characterfile">GitHub - lalalune/characterfile: A simple file format for character data</a>: 一种用于角色数据的简单文件格式。通过在 GitHub 上创建账号来为 lalalune/characterfile 的开发做出贡献。

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286055344140845058)** (6 messages): 

> - `Shampoo` 对比 `Adam`
> - `Diagram of Thought` 框架
> - `ReST-MCTS` 论文讨论
> - 逆向工程 `O1` 


- **Shampoo 预条件方法表现优于 Adam**：研究展示了 **Shampoo**（一种比 **Adam** 更高阶的预条件方法）的有效性，同时也承认了其在**超参数**和**计算开销**方面的缺点。一种名为 **SOAP** 的新算法通过将 Shampoo 与 **Adafactor** 联系起来，简化了其效率。
   - 这一见解将 SOAP 定位为一种极具竞争力的替代方案，旨在增强深度学习优化中的计算效率。
- **ReST-MCTS* 论文探索**：围绕 **ReST-MCTS*** 论文展开了讨论，该论文因创新性地结合了 **STaR**、**PRM** 和 **MCTS** 方法论而受到关注，并被认为其潜力被*低估*了。该论文描述了一种详尽的逐步验证方法，引起了成员们的兴趣。
   - 参与者表示希望探索重叠的研究，并考虑应对该论文中概述的挑战的新方法。
- **对逆向工程 O1 的兴趣**：人们对**逆向工程 O1** 的兴趣日益浓厚，多位成员寻求分享见解和发现。协作请求表明了社区在进一步探索该主题方面的共同努力。
   - 成员们表示愿意就该领域的研究进行联系和讨论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: Improving and Stabilizing Shampoo using Adam</a>: 越来越多的证据表明，在深度学习优化任务中，Shampoo（一种高阶预条件方法）比 Adam 更有效。然而，Shampoo 的缺点包括额外的超参数...</li><li><a href="https://arxiv.org/abs/2409.10038v1">On the Diagram of Thought</a>: 我们介绍了 Diagram of Thought (DoT)，这是一个将大语言模型 (LLMs) 中的迭代推理建模为在单个模型内构建有向无环图 (DAG) 的框架。与...不同...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1286464007640846397)** (1 messages): 

> - `Chatroom features`
> - `Qwen 2.5`
> - `Mistral Pixtral`
> - `Neversleep Lumimaid v0.2`
> - `Hermes 3` 


- **可编辑消息增强 Chatroom 功能**：新的 Chatroom 功能现在允许用户**编辑消息**（包括来自 Bot 的消息），通过点击重新生成按钮即可获取新的回复。
   - 此外，Chatroom 的**统计数据 (stats)** 经过重新设计，提升了用户体验。
- **Qwen 2.5 在编程和数学方面表现卓越**：**Qwen 2.5 72B** 拥有更丰富的知识，并在编程和数学能力上有了显著提升，拥有令人印象深刻的 **131,072** 上下文长度。更多详情请参阅[此处](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct)。
   - 该模型在性能上标志着显著进步，特别是对于编程应用。
- **Mistral 推出 Pixtral 模型**：**Mistral Pixtral 12B** 作为 Mistral 的首个多模态模型已发布，并提供**免费版本**以供探索其功能。更多信息请点击[此处](https://openrouter.ai/models/mistralai/pixtral-12b)。
   - 此次发布将 Mistral 的产品线扩展到了多模态应用领域，吸引了用户的广泛关注。
- **Neversleep Lumimaid 迎来重大更新**：**Neversleep Lumimaid v0.2 8B** 是 Llama 3.1 8B 的精炼版本，据称其数据集质量较前代有**巨大提升**。点击[此处](https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b)了解更多。
   - 预计此次更新将显著增强性能和功能。
- **Hermes 3 模型更新**：**Hermes 3** 已转为**付费模型**，价格为 **$4.5/m**，不过**免费版**和**扩展版**仍可供用户使用。更多详情请参阅[此处](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b)。
   - 这一转变可能会改变用户的可访问性，但仍提供了替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct">Qwen2.5 72B Instruct - API, Providers, Stats</a>: Qwen2.5 72B 是 Qwen 大语言模型系列的最新产品。通过 API 运行 Qwen2.5 72B Instruct</li><li><a href="https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b">Lumimaid v0.2 8B - API, Providers, Stats</a>: Lumimaid v0.2 8B 是 [Llama 3 的微调版本。通过 API 运行 Lumimaid v0.2 8B</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: Mistral AI 的首个图像转文本模型。其权重按照其传统通过种子发布：https://x。通过 API 运行 Pixtral 12B</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1286289611018207243)** (1 messages): 

> - `Google Gemini`
> - `No-code agent creation`
> - `Open Agent Cloud`
> - `Enterprise automation`
> - `Screen recording agents` 


- **Google Gemini 推出视频转 Agent 功能**：借助 **Google Gemini**，用户现在可以上传 **Loom 视频**，在几秒钟内创建一个**无代码拖拽式 Agent**，这是目前世界上构建 Agent 最快的方式。
   - *以前，构建一个 Twitter Agent 需要 20 分钟，但现在通过录制的视频仅需 5 秒钟即可完成。*
- **在 Open Agent Cloud 中即时扩展 Agent**：Agent 创建后，可以立即在 **Open Agent Cloud** 中运行，允许用户将调度规模扩展到**数千个 Agent** 并行运行。
   - 所有 Agent 都会将数据直接流式传输到**仪表板**，确保实时监控和控制。
- **解决企业的专业知识流失问题**：这种创新方法解决了企业和政府面临的一个关键问题：员工和承包商离职导致的**专业知识流失**。
   - 现在，用户可以从**几十年前的屏幕录像**中生成 Agent，从而保留宝贵的知识。
- **观看演示视频**：在[这段 YouTube 视频](https://www.youtube.com/watch?v=gsU5033ms5k)中查看这一突破性功能的介绍，展示了如何毫不费力地创建 Agent。
   - 该视频提供了关于利用视频内容提升生产力和自动化水平的见解。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1286069957137793067)** (136 条消息🔥🔥): 

> - `OpenAI Rate Limits` (OpenAI 速率限制)
> - `Payment Issues on OpenRouter` (OpenRouter 支付问题)
> - `Model Sharing and Chat History` (模型共享与聊天历史)
> - `Integrating New Models` (集成新模型)
> - `Job Impact of AI` (AI 对就业的影响)


- **OpenAI 提高了 API 速率限制**：OpenAI 提高了其 o1 API 的速率限制，o1-preview 现在允许 **每分钟 500 次请求**，o1-mini **每分钟 1000 次请求**。
   - 此举旨在支持使用 tier 5 费率的开发者，进一步扩大对更广泛功能的访问。
- **OpenRouter 上的支付问题**：用户报告了在 OpenRouter 上尝试充值时遇到支付错误，通常会收到 **error 500** 消息，提示资金不足。
   - 建议用户检查银行通知，因为支付尝试可能因各种原因被拒绝。
- **聊天记录的本地存储**：用户询问了如何跨设备共享聊天记录，发现 OpenRouter 的聊天日志存储在本地，没有直接的共享功能。
   - 将聊天记录导出为 JSON 文件被提及为在设备间传输对话数据的唯一方法。
- **在 OpenRouter 上集成新模型**：有关于如何通过 OpenRouter 分发新模型的咨询，表明需要正式请求或集成流程的指导。
   - 用户对通过该平台 API 提供新模型所需的步骤表示感兴趣。
- **AI 对就业影响的分析**：讨论了 AI 和自动化对就业的潜在影响，预测了到 2027 年及以后的各种岗位流失情景。
   - 推测认为，到 2027 年，AI 的进步可能会影响 **10-20%** 的工作岗位，到 2040 年这一比例可能会上升到 **50-70%**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/slow_developer/status/1836693976050475167">来自 Haider. (@slow_developer) 的推文</a>: 🚨 OpenAI CEO Sam Altman 确认很快将从 o1-preview 转向完整的 o1 模型。“新的推理模型 o1-preview 在接下来的几个月里将会有显著提升，届时我们将从初始版本转向……”</li><li><a href="https://docs.mistral.ai/getting-started/models/">Models | Mistral AI Large Language Models</a>: 概览</li><li><a href="https://openrouter.ai/settings/privacy">Privacy | OpenRouter</a>: 管理您的隐私设置</li><li><a href="https://x.com/OpenAIDevs/status/1836506351062716701">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 速率限制再次提升 5 倍：o1-preview 每分钟 500 次请求，o1-mini 每分钟 1000 次请求。引用 OpenAI Developers (@OpenAIDevs) 的话：我们已经为……提高了 OpenAI o1 API 的速率限制。</li><li><a href="https://mistral.ai/technology/#pricing">Technology</a>: 掌握前沿 AI</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: ChatGPT 的每个前端 GUI 客户端。通过在 GitHub 上创建账号为 billmei/every-chatgpt-gui 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1286041057880178698)** (99 条消息🔥🔥): 

> - `Qwen 2.5 训练问题`
> - `极端量化 (Extreme Quantization)`
> - `模型支持咨询`
> - `针对 OpenAI 端点的轻量级工具`
> - `模型训练中的风格迁移` 


- **训练 Qwen 2.5 面临挑战**：多位用户报告了在保存和重新加载模型（特别是 Qwen 2.5）时遇到的问题，导致在同一脚本中重新加载时出现错误并生成乱码。
   - 一位用户提到了一篇支持帖子，指出该问题已影响到多人，从而引发了对可能修复方案的咨询。
- **探索极端量化技术**：一篇帖子强调了在 Hugging Face 上分享的利用极端量化技术并获得显著性能提升的最新模型发布。
   - 像 Llama3-8B 这样的模型已经过微调以提高效率，引发了人们对 Unsloth 是否能兼容它们的兴趣。
- **Qwen 2.5 支持咨询**：用户渴望了解各种推理库是否支持 Qwen 2.5，有报告显示它可以在 Oobabooga 上运行。
   - 关于 Unsloth 是否支持 Qwen 2.5 的新变体存在不同意见，一些用户在不依赖 Unsloth 模型的情况下直接进行实验。
- **寻找针对 OpenAI 的轻量级工具**：讨论集中在非技术用户可以轻松安装的简单工具的需求上，以便测试支持 OpenAI 的端点。
   - 提到了 SillyTavern 和 LM Studio 等建议，但人们对其与 OpenAI API 的兼容性表示担忧。
- **AI 训练中的风格迁移技术**：一位用户询问了如何训练模型以复制其风格，得到的建议是风格迁移不需要大量的预训练，只需要来自用户的数据。
   - 强调了通过脚本实现自动化，以提高训练反映个人风格模型的效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/HF1BitLLM">HF1BitLLM (Hugging Face 1Bit LLMs)</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd">Moshi v0.1 Release - a kyutai Collection</a>: 未找到描述</li><li><a href="https://huggingface.org/blog/1_58_llm_extreme_quantization">Fine-tuning LLMs to 1.58bit: extreme quantization made easy</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fk0acj/hacks_to_make_llm_training_faster_guide/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/homebrewltd/llama3-s/issues/56">epic: llama3-s v0.3: &quot;I cannot hear / understand you&quot; · Issue #56 · homebrewltd/llama3-s</a>: 目标：使 v0.3 支持多语言，接受更长的问题以及其他数据改进。问题：此前 v0.2 仅在 10 秒以下的指令上表现良好，此前 v0.2 仅在英语输入上表现良好...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/5761">Support BitNet b1.58 ternary models · Issue #5761 · ggerganov/llama.cpp</a>: Arxiv 上新发表的论文描述了一种以 1.58 bits（三进制值：1, 0, -1）训练模型的方法。论文显示其性能优于同等大小的 fp16 模型，且困惑度（perplexity）接近...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8151">ggml-quants : ternary packing for TriLMs and BitNet b1.58 by compilade · Pull Request #8151 · ggerganov/llama.cpp</a>: 这为 TriLMs 和 BitNet b1.58 模型添加了 1.6875 bpw 和 2.0625 bpw 的量化类型。目前，这些分别命名为 TQ1_0 和 TQ2_0。我从 #7931 开始就给出过这个想法的雏形...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1286087374975008790)** (10 messages🔥): 

> - `Probability struggles` (概率论难题)
> - `Swift programming` (Swift 编程)
> - `Confidence in exams` (考试信心)
> - `Qwen models` (Qwen 模型)
> - `AGI challenges` (AGI 挑战)


- **概率问题变得复杂**：一位成员对高中以来**高级**硬币翻转问题的难度表示沮丧，说明在理解材料上遇到了困难。
   - 另一位成员建议使用**二项分布计算器 (Binomial distribution calculator)** 来简化问题。
- **Mahiatlinux 考后回归**：一位最近刚结束考试的成员回到社区，表示对模拟考试很有**信心**，并开始关注**新的 Qwen 模型**。
   - 他们祝愿另一位正在学习 **Swift** 的成员好运，体现了互助的社区氛围。
- **对 AGI 发展的思考**：一位成员指出，学习困难材料的经历凸显了我们距离 **AGI** 仍有巨大差距，强调需要的是解释而不仅仅是答案。
   - 他们评论道：*重点不在于得到正确答案，而在于解释的过程*，这指出了该领域的一个重大挑战。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1286060794617921642)** (18 messages🔥): 

> - `vllm LoRA Adapter Issues` (vllm LoRA 适配器问题)
> - `Pip Installation Problems` (Pip 安装问题)
> - `Model Fine-tuning Limitations` (模型微调限制)
> - `Unsloth Model Usage in Ollama` (在 Ollama 中使用 Unsloth 模型)
> - `Batch Size and Training Speed` (Batch Size 与训练速度)


- **vllm LoRA 适配器导致运行时错误**：一位成员报告了在尝试运行带有 `--qlora-adapter-name-or-path` 参数的命令时出现错误，导致关于形状不匹配（shape mismatch）的运行时异常。
   - 他们引用了一个特定的 [GitHub discussion](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit/discussions/3) 以参考之前遇到的类似问题。
- **Pip 安装挑战**：另一位成员描述了在使用特定环境标记和额外包进行 pip 安装 Unsloth 库时遇到的困难。
   - 他们寻求关于其安装方法是否正确的建议，并对该方法的潜在问题提出了疑问。
- **微调 phi-3.5 mini 撞上最大长度限制墙**：有成员担心无法在 `max_length` 大于 4095 的情况下对 phi-3.5-mini 进行微调或推理，并请求解决方案。
   - 分享了一个 [GitHub issue](https://github.com/unslothai/unsloth/issues/946) 链接，其中详细记录了微调过程中遇到的错误，以提供更多背景信息。
- **用于 Ollama 的 Unsloth 模型模板**：一位成员询问提供的模板对于在 Ollama 中运行 Unsloth 模型是否正确，并展示了代码片段进行澄清。
   - 他们的询问凸显了用户在部署各种模型时正在进行的持续调整。
- **增加 Batch Size 并未加快训练速度**：一位成员表示担心增加 Batch Size 并没有带来更快的训练速度，并对预期行为提出了疑问。
   - 他们的疑问反映了在调整训练参数时对性能优化的困惑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/nlp-course/chapter7/3?fw=tf#fine-tuning-distilbert-with-the-trainer-api">微调掩码语言模型 - Hugging Face NLP 课程</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/946">AttributeError: 'LongRopeRotaryEmbedding' object has no attribute 'inv_freq' when finetuning Phi3.5 mini · Issue #946 · unslothai/unsloth</a>：你好，我在微调 Phi3.5 时遇到了标题中的错误。我相信我使用的是最新的 Unsloth（通过 pip 从 git 安装）。背景：使用已在其他模型上运行的代码微调 Phi3.5...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1286273880440176691)** (2 messages): 

> - `Fine-tuning BART` (微调 BART)
> - `F1 Score Discrepancy` (F1 分数差异)
> - `Multiple BOS Tokens Issue` (多个 BOS Token 问题)


- **微调 BART large 产生意外结果**：一位用户正在微调 **BART large** 以复现论文结果，但遇到了 F1 分数的**差异**（41.5 vs 43.5）。
   - 尽管使用了与作者相同的模型、超参数和数据集，他们发现自己的分数比预期低了 2.5 个标准差。
- **生成过程中出现意外的多个 BOS Token**：用户报告说 **BART** 偶尔会输出**两个或三个**起始符 (BOS) Token，而这些并不在微调数据中。
   - 他们检查了输入 Batch 并确认只添加了一个 BOS Token，这表明模型配置中存在更深层次的问题。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1286047379320999939)** (98 messages🔥🔥): 

> - `Aider Environment Variables` (Aider 环境变量)
> - `Benchmarks and Performance` (基准测试与性能)
> - `Model Utilization in Aider` (Aider 中的模型利用)
> - `Issues and Solutions with Aider Setup` (Aider 设置中的问题与解决方案)
> - `RAG Architecture in Aider` (Aider 中的 RAG 架构)


- **修复 Aider 环境配置错误**：用户发现由于文件路径不正确，导致 `ANTHROPIC_API_KEY` 环境变量无法被正确读取，从而引发身份验证问题。
   - 在使用详细模式（verbose mode）后，一名用户确认该错误是由于 Aider 从其仓库（repo）中读取配置，而不是从预期的环境变量中读取。
- **Aider 的基准测试认可**：Aider 在《Qwen2.5-Coder 技术报告》中因其对基准测试的贡献而获得认可，突显了其在 AI 开发和性能评估领域日益增长的影响力。
   - 这一认可说明了 Aider 作为 AI 辅助开发和性能评估工具的价值正在不断提升。
- **有效利用 Aider 功能**：用户讨论了在 Aider 内部使用 `/run` 命令运行 Shell 命令的功能，并提供了 `pytest` 的示例。
   - 分享了 Aider 命令和设置的最佳实践，提升了用户体验和生产力。
- **Aider 中的问题解决策略**：讨论揭示了用户在将 Aider 连接到 Anthropic API 时面临的常见问题，包括 API 过载和变量管理不善。
   - 针对这些问题的排查建议包括验证环境变量以及调整命令以获得更好的连接性。
- **关于 Aider 架构的见解**：有关于 Aider 高层架构的咨询，特别是它如何利用仓库映射（repo maps）来增强代码编辑中的上下文理解。
   - 仓库映射系统帮助 Aider 理解代码库内部的关系，从而提高其在 AI 辅助编程中的效率。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1836506351062716701">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：速率限制再次提升了 5 倍：o1-preview：每分钟 500 次请求；o1-mini：每分钟 1000 次请求。引用 OpenAI Developers (@OpenAIDevs)：我们已经增加了 OpenAI o1 API 的速率限制...</li><li><a href="https://x.com/slow_developer/status/1836693976050475167">来自 Haider. (@slow_developer) 的推文</a>：🚨 OpenAI CEO Sam Altman 确认很快将从 o1-preview 转向完整的 o1 模型。“新的推理模型 o1-preview 在接下来的几个月里将会有显著改进，届时我们将从初始阶段转向...”</li><li><a href="https://aider.chat/docs/llms/anthropic.html">Anthropic</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://arxiv.org/abs/2409.12186">Qwen2.5-Coder 技术报告</a>：在本报告中，我们介绍了 Qwen2.5-Coder 系列，这是对其前身 CodeQwen1.5 的重大升级。该系列包括两个模型：Qwen2.5-Coder-1.5B 和 Qwen2.5-Coder-7B。作为一款代码专用...</li><li><a href="https://aider.chat/docs/llms">连接到 LLMs</a>：Aider 可以连接到大多数 LLMs 进行 AI 结对编程。</li><li><a href="https://aider.chat/docs/repomap.html">仓库地图 (Repository map)</a>：Aider 使用你的 git 仓库地图为 LLMs 提供代码上下文。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/twenty-one-pilots-stressed-out-wake-up-you-need-to-make-money-21pilots-gif-16455885">Twenty One Pilots Stressed Out GIF - Twenty One Pilots Stressed Out Wake Up - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/usage/commands.html">聊天内命令</a>：通过 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://status.anthropic.com/incidents/gg215bzz7rhm">3.5-Sonnet 部分故障</a>：未找到描述</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#setting-up-a-development-environment)">aider/CONTRIBUTING.md at main · paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建一个账户来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: 为 LLMs 优化的推理代理</a>：为 LLMs 优化的推理代理。通过在 GitHub 上创建一个账户来为 codelion/optillm 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/tests/basic/test_io.py">aider/tests/basic/test_io.py at main · paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建一个账户来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/coders/ask_prompts.py?">aider/aider/coders/ask_prompts.py at main · paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建一个账户来为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1286071696792490005)** (20 messages🔥): 

> - `Using Aider in Python apps`（在 Python 应用中使用 Aider）
> - `Security concerns with Aider`（Aider 的安全性考量）
> - `Creating files with Aider`（使用 Aider 创建文件）
> - `Hugging Face model integration`（Hugging Face 模型集成）
> - `Control over URL scraping`（控制 URL 抓取行为）


- **将 Aider 集成到 Python 应用程序中**：一位用户寻求在 Python 应用中使用 Aider，通过指定 Aider 的根文件夹来编辑用户项目仓库中的代码。
   - 另一位用户建议对 Aider 使用命令行脚本（command line scripting）进行批量操作，并指出正确设置文件路径可以解决编辑问题。
- **关于 API Key 安全性的担忧**：一次小组讨论揭示了用户在使用 Aider 时的安全性焦虑，因为 Aider 可以访问代码库中的 API Key 和 secrets。
   - 回复澄清了 Aider 充当的是 AI 处理器（handler）的角色，建议用户将注意力集中在他们加载的 AI 模型上，以解决安全顾虑。
- **使用 Aider 创建文件的挑战**：一位用户表示，尽管提供了详细的结构和文件夹路径，但很难通过 Aider 创建空文件。
   - 其他人建议只将相关文件添加到聊天中，以便 Aider 有效运行，并建议查看文档获取技巧。
- **在 Hugging Face 模型中使用 Aider**：一位用户询问关于将 Aider 与 Hugging Face 结合使用的指导，但在通过 Aider 命令正确列出模型时遇到了挑战。
   - 回复指向了一个特定的文档链接，该链接解释了 Hugging Face 模型的兼容性和用法，并提示了所需的模型名称格式。
- **管理 URL 抓取行为**：一位用户询问如何防止 Aider 在没有明确指令的情况下抓取 Prompt 中粘贴的 URL。
   - 他们对这种自动行为表示沮丧，并明确表示更倾向于使用特定命令手动触发抓取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>：LiteLLM 支持以下类型的 Hugging Face 模型。</li><li><a href="https://aider.chat/docs/usage/tips.html#creating-new-files">Tips</a>：使用 Aider 进行 AI 结对编程的技巧。</li><li><a href="https://github.com/paul-gauthier/aider/tree/main/aider/website">aider/aider/website at main · paul-gauthier/aider</a>：Aider 是你终端里的 AI 结对编程助手。欢迎在 GitHub 上通过创建账号来参与 Aider 的开发。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1286039221508767754)** (8 messages🔥): 

> - `ECMAScript vs JavaScript`
> - `NodeJS alternatives`
> - `ell Library`
> - `Prompt Engineering` 


- **ECMAScript vs 大家都叫它 JavaScript**：关于应该叫它 **ECMAScript** 还是 **JavaScript** 展开了辩论，一位成员坚持认为，由于 Oracle 的不作为，这个名字应该属于公共领域。
   - 另一位成员幽默地建议，应该让 JavaScript 的粉丝们自己解决这个问题。
- **面向 NodeJS 爱好者的 LiteLLM 替代方案**：一位成员分享了一个在 **NodeJS** 中构建 AI 应用的 LiteLLM 替代方案，可以在 [Portkey-AI](https://www.npmjs.com/package/portkey-ai) 找到。
   - 这可以为开发者提供一种在应用程序中集成语言模型的新方法。
- **用于 Prompt Engineering 的 'ell' 库**：分享了关于 **'ell' 库**的详细信息，这是一个轻量级的 Prompt Engineering 工具，允许将 Prompt 视为**函数（functions）**。
   - 该库是 OpenAI 在语言模型领域多年经验的结晶，旨在增强 Prompt 设计。
- **对类似项目的兴奋**：一位成员担心 **'ell' 库**与他们正在开发的项目过于相似，导致了一阵恐慌。
   - 另一位成员对这种想法的一致性表示欢呼，轻松地表示“英雄所见略同！”。



**提到的链接**：<a href="https://docs.ell.so/index.html#">Introduction | ell documentation</a>：ell 是一个轻量级的 Prompt Engineering 库，将 Prompt 视为函数。它提供了用于语言模型程序的版本控制、监控和可视化工具。

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1286166124286705685)** (4 条消息): 

> - `airLLM 压缩`
> - `Leaderboard 任务`
> - `HF 数据集上传` 


- **关于 airLLM Forward 调用的查询**：一位成员询问在使用 **airLLM** 时，是否可以在享受压缩优点的同时，调用模型的 **forward** 函数而不是 **generate** 函数。
   - 目前没有收到回复，但这引发了关于模型使用灵活性的有趣问题。
- **Leaderboard 任务准确率提取**：一位成员表示需要一个脚本，从在本地模型上运行 Leaderboard 任务时生成的冗长 JSON 文件中提取主要的 **accuracy results**（准确率结果）。
   - 他们提到结果保存在 **output_path** 中，表明希望更方便地处理数据。
- **建议的 HF 上传方法**：另一位成员建议使用 `—hf_hub_log_args` 将 Leaderboard 结果上传到 Hugging Face，并认为这能简化结果处理。
   - 他们参考了一个每次运行占一行的数据集示例，链接见 [此处](https://huggingface.co/datasets/baber/eval-smolLM-135M-3-private)。
- **自定义脚本开发计划**：需要提取准确率的成员表示，他们打算创建一个简单的脚本来简化这一过程。
   - 这反映了社区在解决共同挑战方面的积极态度。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1286146330099716238)** (74 条消息🔥🔥): 

> - `Shampoo Optimizer`
> - `SOAP Algorithm`
> - `GFlowNets and JEPA`
> - `Inference Time Search`
> - `Self-Supervised Learning Challenges` 


- **Shampoo 与 Adam 的性能见解**：研究表明，**Shampoo** 作为一种高阶预条件（preconditioning）方法，在优化任务中优于 Adam，但存在计算开销增加和超参数较多等缺点。
   - 提出的一种解决方案是 **SOAP**，它通过在 Shampoo 预条件矩阵的特征基（eigenbasis）中运行，结合了 Shampoo 和 Adafactor 的优点。
- **GFlowNets 与 JEPA 的讨论**：对于 **GFlowNets** 和 **JEPA** 的贡献存在怀疑态度，主要担忧在于这些模型的实际影响以及目标定位的清晰度。
   - 用户讨论了 GFlowNets 在 AI for science 领域的潜在间接影响，同时指出 JEPA 的理论基础似乎较弱。
- **OpenAI 的推理时搜索 (Inference Time Search)**：据报道，OpenAI 正在转向自我博弈（self-play）技术和推理时搜索，正如 Noam Brown 所强调的，以增强模型初始训练后的输出质量。
   - 这种方法在**国际象棋和扑克**等游戏中的有效性被指出能显著提高性能，这可能为其目前的策略提供参考。
- **自监督学习 (Self-Supervised Learning) 的挑战**：引用了几篇讨论 **SSL** 面临困难的论文，包括优化器的不稳定性以及训练过程中的表示崩溃（representation collapse）。
   - 最近的研究旨在统一这些挑战的理论视角，从而有效地指导从业者。
- **AI 研究中的 Kye Gomez 角色**：人们对 Kye Gomez 的声誉表示担忧，强调虽然他的代码库可能略有改进，但本质上仍具有误导性。
   - 讨论反映了对其 AI 贡献质量和诚信的怀疑，尤其是考虑到潜在的更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: Improving and Stabilizing Shampoo using Adam</a>: 越来越多的证据表明，在深度学习优化任务中，Shampoo 这种高阶预条件方法比 Adam 更有效。然而，Shampoo 的缺点包括额外的超参数...</li><li><a href="https://arxiv.org/html/2409.11340v1">OmniGen: Unified Image Generation</a>: 未找到描述</li><li><a href="https://github.com/nikhilvyas/SOAP">GitHub - nikhilvyas/SOAP</a>: 通过在 GitHub 上创建账户来为 nikhilvyas/SOAP 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2302.02774">The SSL Interplay: Augmentations, Inductive Bias, and Generalization</a>: 自监督学习 (SSL) 已成为一种强大的框架，可以从原始数据中学习表示而无需监督。但在实践中，工程师面临着诸如调优优化器不稳定等问题...</li><li><a href="https://arxiv.org/abs/2303.00633">An Information-Theoretic Perspective on Variance-Invariance-Covariance Regularization</a>: 方差-不变性-协方差正则化 (VICReg) 是一种自监督学习 (SSL) 方法，在各种任务上都显示出了良好的结果。然而，其背后的基本机制...</li><li><a href="https://arxiv.org/abs/2205.11508">Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods</a>: 自监督学习 (SSL) 推测输入和成对的正样本关系足以学习有意义的表示。尽管 SSL 最近达到了一个里程碑：表现优于有监督学习...</li><li><a href="https://arxiv.org/abs/2408.16978">Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer</a>: 具有长上下文能力的超大语言模型 (LLMs) 是自然语言处理和计算生物学中复杂任务（如文本生成和蛋白质序列分析）不可或缺的一部分...</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/6462">Adding the new feature of FPDT by YJHMITWEB · Pull Request #6462 · microsoft/DeepSpeed</a>: FPDT 只能与此版本的 Megatron-DeepSpeed 配合使用。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1286101066005479476)** (15 条消息🔥): 

> - `OPT 和 Pythia 中的数据匹配`
> - `Attention 层与 In-Context Learning 评分`
> - `Distributed Alignment Search 与 SAEs`
> - `KV Cache 表示探索` 


- **OPT 和 Pythia 之间的数据匹配见解**：来自 **OPT** 和 **Pythia** 的数据表明，训练后数据保持一致，主要表现为跨层深度的平滑过渡。
   - 最终隐藏状态的幂律系数在约 **1B tokens** 后迅速收敛，表明模型性能趋于稳定。
- **训练过程中 Attention 层的行为**：图表显示，最终 Attention 层残差的 **R^2** 指标在 **500M tokens** 时收敛，而 In-Context Learning 评分在 **1B tokens** 时达到峰值。
   - 这种独特的形状在其他图表中未曾出现，引发了对其训练性能影响的疑问。
- **关于 Distributed Alignment Search 论文的查询**：一位成员询问了一篇关于 **distributed alignment search** 和 **SAEs** 的帖子，该帖子可能已消失。
   - 另一位成员提到了关于开源 SAEs 的 **Ravel Eval** 论文，暗示这可能就是所讨论的论文。
- **KV Cache 表示查询**：一位成员正在探索 **KV cache** 中存储的内容，特别是关于它如何由 Token Embeddings 和低层的先前状态组成。
   - 讨论联系到了 **chain-of-thought** 和 **thinking along token lines** 的概念，突出了进一步研究的兴趣领域。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1286293953381138545)** (18 条消息🔥): 

> - `输入 Padding 与 Unpadding`
> - `Model Generate 中的错误处理`
> - `带有 Padding 的 Token 生成`
> - `Batch Size 管理` 


- **Padding 输入导致 AssertionError**：一位成员在尝试恢复 `model_generate` 中 Padding 输入后的原始元素顺序时遇到了 `AssertionError`。
   - 据推测，使用 Stop Tokens 进行 Padding 可能会导致问题，因为它们在处理过程中可能会被过滤掉。
- **理解 Generate 函数逻辑**：讨论揭示了 `generate_until` 函数按 Token 长度对请求进行排序，以有效地管理内存问题。
   - 这种方法通过准确估算时间并在稍后返回原始顺序来优化性能。
- **澄清 Model Generate 中的 Padding 逻辑**：据分享，`_model_generate` 函数将 Batch Padding 到指定大小，并在之后移除 Padding。
   - 该成员承认在切片 Tensor 后如何处理 Padding 方面存在误解。
- **识别返回问题**：该成员发现正确的返回语句应该是 `toks[:context.size(0)]`，以避免在 Batch 满时返回空数组。
   - 鉴于之前对 Padding 机制的困惑，他们分享了这一见解。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9a092f374bdc6d6032ae2b878b7a49b97801ab69/lm_eval/models/huggingface.py#L1304">lm-evaluation-harness/lm_eval/models/huggingface.py at 9a092f374bdc6d6032ae2b878b7a49b97801ab69 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1286321409160773643)** (1 条消息): 

> - `Polaris 节点连接性`
> - `任务超时问题` 


- **困扰 Polaris 节点的瞬时错误**：成员报告了 Polaris 上的一个**瞬时错误**，节点无法相互定位，导致连接失败。
   - 因此，任务经常在一小时后超时，导致 **TCPConnectionError**。
- **任务超时困扰**：一小时后任务超时给用户带来了困扰，突显了 Polaris 内部显著的连接问题。
   - 该问题正在影响平台的整体效率和可靠性。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1286050149910708245)** (36 messages🔥): 

> - `O1-Preview Capabilities`
> - `Discussion on AI Alignment`
> - `Qwen 2.5 vs Llama 3.1`
> - `Recording ChatGPT's Voice`
> - `Pixtral & OpenRouter` 


- **O1-Preview 令人失望**：成员们对 **O1-Preview** 模型表示失望，认为它虽然打字速度更快，但与 **4o** 相比缺乏深度。
   - *一位成员评论道*，“o1 并不觉得更聪明，它只是打字更快了”。
- **讨论 AI Alignment 挑战**：一位成员提出了一种改进 **AI Alignment** 的新方法，建议在训练未来模型时，基于先前模型的输出，重点关注共情能力和帮助性。
   - *人们担心超智能 AI 是否仍会通过提供定制化响应来误导用户*。
- **比较模型性能**：参与者讨论了关于 **Qwen 2.5** 的惊人说法，成员们注意到据报道它的表现优于 **Llama 3.1**，尽管两者在参数规模上存在显著差异。
   - *一位用户评论道*，“人们在说一些疯狂的事情，比如 Qwen 2.5 72b 的表现优于 Llama 3.1 405b”。
- **ChatGPT 语音录制挑战**：一位用户对无法在手机上录制 **ChatGPT** 的音频感到沮丧，称在尝试过程中没有捕捉到声音。
   - 尽管使用了手机的录音功能，他们仍无法达到预期的效果。
- **Pixtral 和 OpenRouter 见解**：一位用户分享说可以通过 **OpenRouter** 免费访问 **Pixtral**，并确认了该模型在该平台上的功能，并给出了具体的模型建议。
   - *他们建议其他人*使用免费模型以最大化该工具的效益。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1286045977312100363)** (21 messages🔥): 

> - `Daily Limits for GPT Usage`
> - `GPT-4 and GPT-4o Limits`
> - `O1 and O1-Mini Message Caps`
> - `Memory Functionality in GPTs` 


- **Mini 的每日限制**：确认了 **O1 Mini** 的限制为 **每天 50 条消息**。
   - 这一规则对于防止通过垃圾信息或重复帖子**干扰服务器**至关重要。
- **了解 GPT-4o 消息上限**：成员们讨论了 **GPT-4o** 每 3 小时允许发送 **80 条消息**，与其限制一致。
   - 然而，在此背景下，**GPT-4 限制**在同一时间段内仅允许 **40 条消息**。
- **澄清 O1 和 O1-Mini 上限**：已确定 **O1 和 O1-Mini** 的使用限制独立于 **GPT-4 和 GPT-4o 限制**。
   - O1-Mini 和 O1-Preview 分别遵循 **24 小时和 7 天的窗口期**。
- **在 GPTs 中使用 Memory**：目前已澄清 **GPTs 不具备 Memory 功能**。
   - 有关此问题的更多信息，分享了一个关于 GPTs 中 Memory 的链接：[帮助文章](https://help.openai.com/en/articles/8983148-does-memory-function-with-gpts)。
- **使用限制资源**：一位成员分享了了解当前**使用限制**的有用链接，其中一个链接将所有限制整合到了一个页面中。
   - 该资源被高度推荐，用于获取 OpenAI 提供的更清晰的使用限制信息：[使用限制链接](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus#h_d78bb59065)。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1286081870756315274)** (3 messages): 

> - `Data extraction with GPT-4o`
> - `Structured Output feature` 


- **使用 GPT-4o 进行 CSV 数据提取**：一位成员正在使用 **GPT-4o** 从非结构化文本中提取数据，之前通过示例使用了 **CSV 分隔输出**的 system prompt。
   - 他们现在寻求指导，如何将 few-shot 示例调整为新的 **Structured Output 功能**，因为 CSV 不适用于 JSON 输出。
- **为 Structured Output 调整 prompt**：另一位成员建议在 system prompt 中提供 **JSON 输出**示例，以适配 Structured Output 功能。
   - 他们确认这种方法看起来很有效，并指出模型在提供示例后能更好地理解任务。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1286081870756315274)** (3 messages): 

> - `GPT-4o Data Extraction`
> - `Structured Output Feature` 


- **从 CSV 转向 JSON 输出**：一位用户分享了他们使用 **GPT-4o** 从非结构化文本中提取数据的经验，重点讨论了过去使用的 CSV 格式输出。
   - 他们询问了如何针对新的 **Structured Output 特性** 调整 System prompt 中的 **Few Shot 示例**，并建议可能需要提供 JSON 输出示例。
- **提供了有用的 Prompt 示例**：另一位参与者建议了一个可能的解决方案，提到了一种特定的 Prompt 格式，可以有效地向 **GPT-4o** 传达预期的提取任务。
   - 他们表示相信这种方法将帮助模型准确地理解并生成结构化输出。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1286356941865619518)** (4 messages): 

> - `NVIDIA Triton`
> - `Triton Inference Server`
> - `Related Discussion Rooms` 


- **Kashimoo 询问关于 NVIDIA Triton 的信息**：一位成员询问是否有人熟悉 **NVIDIA 的 Triton**，并特别澄清这并非 OpenAI 的版本。
   - 这引发了关于专门讨论 Triton 的相关频道和资源的讨论。
- **关于 Triton Inference Server 的澄清**：另一位成员询问该查询是否专门针对 **NVIDIA 的 Triton Inference Server**。
   - Kashimoo 澄清了这一细节，确保讨论重点集中在 NVIDIA 的产品上。
- **Triton 讨论的资源和频道**：一位成员建议了多个相关频道以便对 Triton 进行更深入的交流，并指出 <#1191300313928433664> 和 <#1189607595451895918> 等频道的重要性。
   - 他们建议与其他工作组（如 <#1275130785933951039>）协作以获取更多见解。


  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1286090763616256030)** (3 messages): 

> - `GemLite-Triton`
> - `Triton Conference Slides`
> - `Triton-Puzzles Colab`
> - `Gradio Output` 


- **GemLite-Triton 发布**：**GemLite-Triton** 项目正式发布，为构建支持 GEMV/GEMM、权重量化和各种激活格式的自定义低位 matmul 内核提供了完整的解决方案。
   - 据报道，在大矩阵上，**GemLite-Triton** 的性能优于 **Marlin** (VLLM) 和 **BitBlas** 等高度优化的解决方案。可以在 [GitHub](https://github.com/mobiusml/gemlite) 上查看。
- **Triton Conference 幻灯片查询**：一位成员询问是否可以获取 **Triton Conference** 的幻灯片，以及是否有分享这些资料的时间表。
   - 他们艾特了另一位成员以寻求关于幻灯片状态的更多信息。
- **Colab 上的 Triton-Puzzles 问题**：一位用户对 Colab 上的 **Triton-Puzzles** notebook 是否仍能正常运行表示担忧，因为 Gradio 输出似乎存在问题。
   - 他们分享了该 notebook 的链接供参考，但未提供具体的错误详情。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb#scrollTo=W9appXLw4Bka">Google Colab</a>：未找到描述</li><li><a href="https://github.com/mobiusml/gemlite">GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton</a>：CUDA / Triton 中简单快速的低位 matmul 内核 - mobiusml/gemlite
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1286053729946177646)** (7 messages): 

> - `Chrome Tracing with PyTorch Profiler`
> - `FlashAttention3 Integration`
> - `Model Optimization Talks` 


- **探索使用 PyTorch Profiler 进行 Chrome Tracing**：一位成员询问了关于使用 PyTorch profiler 导航 Chrome tracing 的[视频或资源](https://discord.com/channels/1189498204333543425/1189607726595194971/1273329959041105980)。
   - 另一位成员推荐了 **Taylor Robbie 的演讲**作为该主题的宝贵资源。
- **FlashAttention3 与 GPU 兼容性**：针对 `torch.nn.functional.scaled_dot_product_attention()` 是否在 Hopper GPU 上自动使用 **FlashAttention3** 的问题，一位成员指出 FlashAttention-3 预计将集成在即将发布的 PyTorch 版本中。
   - 这一见解与一篇关于 FlashAttention3 的 **7 月博客文章**相关联。
- **请求模型优化演讲链接**：一位成员回忆起一个关于优化模型的演讲，并请求其他推荐演讲的链接，想知道该演讲是否由另一位成员提供。
   - 这引发了一些模糊的回忆，进一步强调了大家对**模型优化**讨论的兴趣。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1286275447470166112)** (4 messages): 

> - `Torchao Autoquant`
> - `Torchao Compile` 


- **理解 Torchao Autoquant 语法**：用户寻求关于使用 `torchao.autoquant(model.cuda())` 还是 `torchao.autoquant(model).cuda()` 正确语法的澄清。
   - 另一位成员解释说，正确的方法是 `torchao.autoquant(model).cuda()`，因为它在将模型移动到设备之前先对其进行准备。
- **Autoquantization 过程的步骤**：提供了关于自动量化涉及的**三个步骤**的详细解释，强调了在校准（calibration）和最终确定（finalization）之前进行模型准备的必要性。
   - 该过程强调了在模型准备好后，通过输入运行模型以进行有效优化的重要性。
- **文档化的示例用法**：分享了一个将 `torchao.autoquant()` 与模型编译结合使用的示例作为推荐方法：`torchao.autoquant(torch.compile(model))`。
   - 澄清内容包括使用多种输入形状（input shapes），以及在用这些输入运行模型后需要最终确定（finalize）autoquant。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1286058022313132225)** (3 messages): 

> - `GPU programming groups`
> - `San Francisco GPU community` 


- **对旧金山 GPU 编程工作组的兴趣**：一位成员表示有兴趣加入一个位于旧金山的 **GPU programming 阅读/工作组**。
   - 他们通过陈述 *'非常想加入'* 来表达他们的热情。
- **询问小组计划**：另一位成员询问某位特定成员是否仍在计划组织这样一个小组，表现出对社区建设的渴望。
   - 回复表明目前**没有计划**建立这样一个小组。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1286124903451328523)** (10 messages🔥): 

> - `Hackathon Objectives`
> - `GQA cuDNN Development`
> - `L2 Side Aware Optimization`
> - `Stochastic Rounding Technique` 


- **黑客松促进开发者贡献**：即将举行的黑客松旨在让更多开发者能够为开源做贡献，在活动前为那些可能“GPU poor”的人提供算力额度，从而促进参与和创新。
   - *黑客松之后，参与者可以变得 GPU rich！*
- **GQA cuDNN 工作进展**：重点一直放在 **GQA cuDNN** 上，讨论了 forward 和 backward 的实现，表明对 stride 和 layout 的正确性存在一些不确定性。
   - 一些参与者确认他们计划稍后处理 backward 部分，并希望在旅途中解决问题。
- **L2 Side Aware Optimization 文章编写中**：在旅行期间，一位成员提到他们可能会推迟 cuDNN 任务，优先撰写一篇关于 **L2 Side Aware 优化**和 **NVIDIA 内存层级结构**的文章。
   - 他们表示，在平庸的 WiFi 条件下进行开发将非常痛苦，强调了远程工作的挑战。
- **提出 Stochastic Rounding 技术**：一位成员分享了一个涉及 **stochastic rounding**（随机舍入）的新颖想法，建议强制将某些尾数位（mantissa bits）归零，以在处理过程中节省功耗。
   - 他们发现这个 hack 想法非常有趣，暗示了该技术潜在的优化空间。
- **编写极其模糊的 Megahack 提案时的不适感**：关于团队成员在起草一份关于“GH200 上的 llm.c 和/或 Llama3.1”的模糊提案时的舒适度出现了疑问，原帖作者表达了犹豫。
   - ……他们寻求其他可能在处理此类任务时感到更自在的人的支持。


  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1286224957667541003)** (10 条消息🔥): 

> - `Language model quantization` (语言模型量化)
> - `BitNet performance` (BitNet 性能)
> - `Knowledge compression` (知识压缩)
> - `Llama 3 8B results` (Llama 3 8B 结果)
> - `Product quantization methods` (乘积量化方法)


- **语言模型存储的知识有限**：最近的一篇论文确立了语言模型每个参数可以存储 **2 bits 的知识**，当量化到 **int4** 时，这一数值会降至 **每个参数 0.7 bits**。然而，一些成员对这些发现的准确性表示怀疑，因为大型模型在较低量化水平下仍能保持性能。
   - *一位成员强调了在基准测试中衡量“知识”与传统准确性指标之间的区别*。
- **BitNet 性能恢复面临挑战**：讨论中的一段翻译指出，尝试用 **L3-8B** 恢复 **L2-7B** 性能的努力已经失败，这意味着在当前方法下，**BitNet** 缺乏足够的合理性。这引发了关于仅通过微调而不进行预训练是否能有效实现高性能的担忧。
   - *一位成员评论说，使用 **SFT** (Supervised Fine-Tuning) 来转换模型能力的有效性存疑，建议依赖预训练策略更为合理。*
- **关于 Llama 3 8B 的令人兴奋的消息**：另一位贡献者报告了对 **Llama 3 8B** 的成功微调，在没有预训练的情况下实现了接近 **Llama 1 & 2 7B 模型** 的性能。更多细节可以在 [Hugging Face 博客文章](https://huggingface.co/blog/1_58_llm_extreme_quantization) 中找到。
   - *鉴于之前关于模型性能的讨论，该参与者对这些发现的含义也表达了一定的怀疑。*
- **知识压缩的见解**：一位成员分享了一篇论文链接，讨论了现代 LLMs 在压缩时如何丧失知识和推理能力。该研究报告称，在压缩阶段，知识的丧失早于通过上下文进行的推理。
   - *这些发现促使人们重新考虑评估压缩模型效能的现有指标，转而关注更全面的评估工具。*
- **关于乘积量化方法的讨论**：一位用户提出了关于乘积量化方法在实现与 **BitNet** 技术相当的压缩率方面的可行性问题。这仍然是一个辩论话题，因为其他人表示他们尚未对这些方法的有效性形成明确的意见。
   - *这场讨论反映了社区内对替代压缩策略的持续探索。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/teortaxesTex/status/1836448002971701435">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：翻译：“该死，我们没能用 L3-8B 恢复 L2-7B 的性能，但我们需要编一篇论文”。别再拿“GPU 贫民”当诱饵了。让 BitNet 自生自灭吧，或者让 MSR 来证明它。这个想法只在 pr...</li><li><a href="https://arxiv.org/abs/2310.01382">Compressing LLMs: The Truth is Rarely Pure and Never Simple</a>：尽管取得了显著成就，现代大语言模型 (LLMs) 仍面临巨大的计算和内存占用。最近，多项工作在无训练压缩方面展示了显著的成功...</li><li><a href="https://arxiv.org/abs/2404.14047">An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs</a>：LLaMA 家族已成为最强大的开源大语言模型 (LLMs) 之一，也是多模态大语言模型 (MLLMs) 流行的 LLM 骨干网络，广泛应用于计算机视觉...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1286043411547422730)** (17 messages🔥): 

> - `Hackathon Invitations` (黑客松邀请)
> - `Access to Hack-Ideas Forum` (Hack-Ideas 论坛访问权限)
> - `Missing Users in Discord` (Discord 中缺失的用户)
> - `GemLite-Triton Release` (GemLite-Triton 发布)


- **黑客松邀请正在发放中**：一些成员反映，虽然他们收到了黑客松邀请，但他们的队友却没有，这引发了关于邀请是否仍在滚动发放的询问。
   - 一位成员分享了队友的信息，请求帮助确认其朋友的邀请状态。
- **Hack-Ideas 论坛访问问题**：一位参与者对无法访问即将举行的黑客松的 hack-ideas 论坛表示沮丧。
   - 其他成员尝试提供协助，或寻求关于论坛访问权限的进一步澄清。
- **Discord 角色中缺失用户**：一些收到黑客松二维码的用户未被列入 `cuda-mode-irl` Discord 小组，这促使相关用户主动表明身份。
   - 在确认身份后，多名成员帮助将缺失的用户添加到相应的 Discord 角色中。
- **GemLite-Triton 介绍**：一位成员宣布发布 [GemLite-Triton](https://github.com/mobiusml/gemlite)，这是一套用于量化和低比特操作的高性能 Kernel。
   - 他们鼓励参与者在相关任务中利用该项目，并欢迎关于此版本的任何提问。



**提到的链接**：<a href="https://github.com/mobiusml/gemlite">GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton</a>：CUDA / Triton 中简单快速的低比特 matmul Kernel - mobiusml/gemlite

  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1286368930805452891)** (5 messages): 

> - `Apple Silicon MLX framework` (Apple Silicon MLX 框架)
> - `On-device speech models` (端侧语音模型)
> - `Metal and PyTorch interface` (Metal 与 PyTorch 接口)


- **专为性能设计的 Apple Silicon MLX**：**MLX 框架**专为 Apple 电脑量身定制，利用 **autodiff**、**vmap** 和 **jit compiling** 等专门方法来优化性能。
   - 正如所指出的，它不同于通用的 CPU/GPU 模型，采用了专门针对 **Apple Silicon** 的独特 Kernel。
- **MLX 中的惰性求值（Lazy Evaluation）技术**：该框架利用 **惰性求值**，仅在不同的调用时计算结果，从而提升了整体性能。
   - 这种方法符合其架构设计理念，即实现效率最大化。
- **MLX 中的 Metal 集成**：MLX 集成了 **Metal Performance Shaders** 和像 **'Steel'** 这样的自定义后端，用于优化渲染和计算。
   - 这种适配使 MLX 看起来更像是一个 **PyTorch** 接口，而不是类似的 **Triton** 设置。
- **用于端侧语音对话的基础模型**：人们对开发专门针对 **Apple Silicon** 环境的**端侧语音对话**模型很感兴趣。
   - 该基础模型旨在通过高效的处理能力增强用户交互。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286108741766483978)** (3 messages): 

> - `Story Generation Agent` (故事生成 Agent)
> - `LlamaParse Premium` (LlamaParse Premium)
> - `RAG and agentic applications` (RAG 与 Agent 应用)
> - `Opik partnership` (Opik 合作伙伴关系)


- **构建具有人机回环（Human-in-the-Loop）的故事生成 Agent**：一位成员分享了由 @_nerdai_ 编写的[分步指南](https://t.co/5ElRICjK0C)，介绍如何构建一个用于动态生成“选择你自己的冒险”故事的 Agent，该 Agent 在每一步都融入了人类反馈。
   - *该指南允许用户根据输入和选择有效地塑造故事体验。*
- **LlamaParse Premium 在文档解析方面表现出色**：[LlamaParse Premium](https://t.co/8VTKKYWVOT) 的推出声称通过将多模态模型的视觉理解与长文本和表格内容提取相结合，增强了 LLM 应用的文档解析能力。
   - *这一升级使 LlamaParse 成为强大文档处理的首选。*
- **利用 Opik 的自动日志记录简化 RAG**：讨论强调，即使是基础的 RAG 也会涉及多个步骤，而高级的 Agent 应用需要管理的复杂性甚至更多。
   - 宣布与 [Opik by @Cometml](https://t.co/Z3KdwjAKKv) 达成合作伙伴关系，该工具可在开发和生产环境中自动执行 RAG/Agent 调用追踪的自动日志记录，这带来了令人兴奋的生产力提升。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1286048094785372271)** (51 messages🔥): 

> - `RAG 与语义搜索`
> - `Pinecone 向量数据库问题`
> - `SchemaLLMPathExtractor 用法`
> - `KG schema 类需求`
> - `Property graph index 嵌入问题` 


- **关于 RAG 与语义搜索的讨论**：一名成员正在探索如何管理来自供应商的问题和答案，考虑对记录的回复使用语义搜索。
   - 建议根据答案生成多样化的问题，并利用 vector store 以获得更好的检索效果。
- **Pinecone 向量 ID 管理的挑战**：成员们讨论了 Pinecone 自动生成 ID 带来的困难，由于 serverless 索引的限制，很难根据特定的元数据删除文档。
   - 推荐了 Chroma、Qdrant、Milvus 和 Weaviate 等替代数据库选项，以获得更好的支持和集成。
- **在 SchemaLLMPathExtractor 中使用 kg_schema_cls**：用户寻求关于如何使用 kg_schema_cls 的指导，随后得到了关于需要特定 Pydantic 类来表示图结构的解释。
   - 强调了字段名称必须与特定 schema 匹配，用户还讨论了在尝试创建多个实例时可能出现的验证器（validator）问题。
- **插入没有嵌入的实体**：一名成员报告称，在 property graph index 中手动创建的节点和关系没有获得任何嵌入（embeddings），导致所有查询的评分均为零。
   - 确认了在初始化节点时需要显式附加嵌入，并对当前 graph store 在处理向量方面的局限性表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-tavily-research?from=">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程 (本地模型) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/">指南：在现有 Pinecone Vector Store 中使用 Vector Store Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.pinecone.io/guides/data/manage-rag-documents#delete-all-records-for-a-parent-document">管理 RAG 文档 - Pinecone 文档</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/#functiontool">工具 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/26205a0e96d36382cd4a09432e51731ddb5170a1/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py#L170">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py at 26205a0e96d36382cd4a09432e51731ddb5170a1 · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1286297287143854244)** (1 messages): 

> - `RAG`
> - `LlamaIndex` 


- **对 RAG 文章深度的担忧**：一名成员表示关于 **RAG** 的文章有些流于表面，未能提供有力的论据来反驳像 **LlamaIndex** 这样工具的必要性。
   - *它解释了关于 RAG 的一些基本概念，但没有详细说明所述工具的缺点。*
- **需要更深入的分析**：讨论强调需要更深入地分析为什么在某些情况下可能不需要 **LlamaIndex** 等工具。
   - 成员们指出，如果文章能对替代方案进行详细的比较和技术评估，将会更有益处。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1286143200020402207)** (17 messages🔥): 

> - `Fish Speech`
> - `AdBot incidents` 


- **Fish Speech 表现优于其他模型**：一位成员指出，**Fish Speech** 展示的 **Zero-shot 语音克隆准确率** 超过了尝试过的任何其他开源模型，特别是在模仿语音模式方面。
   - 来自 **1940 年代** 的旧音频生成的输出准确地复制了那个时代的**扬声器音效**。
- **Fish Speech 奇特的语音模式**：有人提到 Fish Speech 会随机在音频中插入 *ahm* 和 *uhm* 等词，在没有提示词的情况下反映了自然的说话模式。
   - 成员们一致认为这一特性为模型的输出增添了真实感。
- **服务器中的 AdBot 问题**：一位成员对服务器中检测到的 **AdBot** 表示担忧，称其像 **malware** 一样在多个服务器中传播。
   - 其他人也加入讨论，强调了排序机制如何导致该机器人出现在成员列表的顶部。



**提到的链接**：<a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - a Hugging Face Space by fishaudio</a>：未找到描述

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286226163211173918)** (11 messages🔥): 

> - `Muse text to image`
> - `Open-source GPT-4o model`
> - `Dataset building for GPT-4o`
> - `LLM fine-tuning challenges`
> - `Tokenization issues in LLMs` 


- **Muse 文本生成图像在 COCO2017 上的挑战**：一位成员报告使用 [Muse text to image](https://github.com/lucidrains/muse-maskgit-pytorch) 在 **COCO2017** 上进行训练，但仅得到了图像输出。
   - 他们表示在实现挑战方面需要指导。
- **开源 GPT-4o 模型合作**：来自 Fish Audio 的 Lengyue 宣布他们正在开发一个**类 GPT-4o 的开源模型**，并愿意与 LAION 共享数据。
   - 他们认为合作可以**加速进度**并交流结果与设计思路。
- **为开源 GPT-4o 构建数据集**：另一位成员提到他们参与了开源 GPT-4o 的**数据集构建**。
   - 普遍观点认为，协作创建数据集将使社区受益。
- **开源 LLM 微调的难度**：Lengyue 表示，**微调现有的开源 LLM** 对于他们的目标来说是不可行的，并建议从零开始训练。
   - 他们强调 GPT-4o 的初始输出优于其他模型，但在随后的调用中会出现 **hallucinations**（幻觉）问题。
- **LLM 中的 Tokenization 问题**：Lengyue 提出，目前 LLM 输出中的挑战可能与 **Tokenization 问题**有关。
   - 他们认为解决这些问题可以提高模型的性能和可靠性。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1286070710866804736)** (20 messages🔥): 

> - `Fal AI funding`
> - `OpenAI O1 improvements`
> - `Jina embeddings v3 launch`
> - `Runway and Lionsgate collaboration`
> - `New multi-agent research team at OpenAI`

- **Fal AI 获得 2300 万美元融资用于增长**：Fal AI 已筹集 **2300 万美元**的种子轮和 A 轮融资，其中 **1400 万美元**的 A 轮融资由 Kindred Ventures 领投，Andreessen Horowitz 参投。
   - *Gorkem Yurt* 在 [Twitter](https://x.com/gorkemyurt/status/1836488019924471953?s=46) 上分享了这一消息，并发布了一篇详细的[博客文章](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/)，介绍了他们加速生成式媒体技术发展的计划。
- **OpenAI o1 模型增强**：据 *OpenAI Developers* 报道，OpenAI 提高了 **o1** API 的速率限制（Rate limits），允许 o1-preview 每分钟 **500** 次请求，o1-mini 每分钟 **1000** 次请求。
   - 这些进展支持了开发者的需求，是扩大 o1 模型访问权限的更广泛计划的一部分，详情见 *Amir* 发布的 [推文串](https://x.com/amir/status/1836782911250735126?s=46)。
- **Jina embeddings v3 发布**：Jina AI 宣布发布 **jina-embeddings-v3**，这是一款多语言嵌入模型，拥有 **5.7 亿参数（570M parameters）**和 **8192 token 长度**，性能优于 OpenAI 和 Cohere 的专有模型。
   - 根据 *Jina AI* 在 [Twitter](https://x.com/JinaAI_/status/1836388833698680949) 上的消息，该新模型在 MTEB 英文排行榜的 1B 参数以下模型中取得了显著排名。
- **Runway 与 Lionsgate 合作开发 Gen-3 Alpha**：Lionsgate 已与 Runway 达成合作，利用其电影库作为其模型 Gen-3 Alpha 的训练数据，这让许多原本预期 Sora 会率先实现这一目标的人感到惊讶。
   - 正如 *Andrew Curran* 在 [Twitter](https://x.com/AndrewCurran_/status/1836411345786290535) 上所强调的，这一进展标志着行业的重大创新。
- **OpenAI 组建多 Agent 研究团队**：OpenAI 正在寻找 ML 工程师加入新的多 Agent 研究团队，强调其在增强 AI 推理（AI reasoning）方面的潜力。
   - 正如 *Polynoamial* 在 Twitter 上指出的，不需要先前的相关经验，感兴趣的候选人可以通过[此表单](https://jobs.ashbyhq.com/openai/form/oai-multi-agent)申请。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hwchung27/status/1836842717302943774?s=46">来自 Hyung Won Chung (@hwchung27) 的推文</a>：这是我在 @MIT 的演讲（延迟了一段时间😅）。我在去年思考范式转移（paradigm shift）时准备了这个演讲。这次延迟发布很及时，因为我们刚刚发布了 o1，我相信这是一个新的...</li><li><a href="https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/">生成式媒体需要速度。fal 已筹集 2300 万美元以加速发展。</a>：fal 在种子轮和 A 轮融资中筹集了 2300 万美元。1400 万美元的 A 轮融资由 Kindred Ventures 领投，Andreessen Horowitz、First Round Capital 以及包括 Perple... 在内的天使投资人参投。</li><li><a href="https://x.com/AndrewCurran_/status/1836411345786290535">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：行业重大新闻，我原以为 Sora 会先到。Lionsgate 已与 Runway 签署协议，将其电影目录用作其模型 Gen-3 Alpha 的训练数据。Lionsgate 将使用定制的...</li><li><a href="https://x.com/amir/status/1836782911250735126?s=46">来自 Amir Efrati (@amir) 的推文</a>：OpenAI 在不损失太多能力的情况下缩小其推理模型的能力，可能与推理能力本身一样重要。 https://www.theinformation.com/articles/openais-miniature-reason...</li><li><a href="https://x.com/JinaAI_/status/1836388833698680949">来自 Jina AI (@JinaAI_) 的推文</a>：终于，jina-embeddings-v3 发布了！这是一个拥有 570M 参数、8192 token 长度的前沿多语言 Embedding 模型，在多语言和长文本检索任务上达到了 SOTA 性能。它超越了...</li><li><a href="https://x.com/fiiiiiist/status/1836471413198459331?s=46">来自 Tim Fist (@fiiiiiist) 的推文</a>：这篇关于 AI 对环境影响的新 WaPo 文章引起了很多关注。核心观点是 GPT-4 编写一封 100 字的电子邮件消耗 0.14kWh 的能量。这就是为什么...</li><li><a href="https://x.com/cognition_labs/status/1836866696797401118">来自 Cognition (@cognition_labs) 的推文</a>：Devin 在代码编辑方面变得更快、更准确，在遵循指令方面更可靠，并且在独立决策方面表现更好。我们还改进了对企业安全的支持...</li><li><a href="https://x.com/bo_wangbo/status/1836391316286038214">来自 Bo (@bo_wangbo) 的推文</a>：关于 jina-embeddings-v3 我个人最喜欢的一点（除了那些花哨的功能）是，我们手动检查了不同文本 Embedding 模型的常见失败案例，创建了失败分类法（failure taxonomy），并尝试修复...</li><li><a href="https://x.com/gorkemyurt/status/1836488019924471953?s=46">来自 Gorkem Yurtseven (@gorkemyurt) 的推文</a>：我们有一些消息要分享！ https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/</li><li><a href="https://x.com/aidan_mclau/status/1836796517463806263">来自 Aidan McLau (@aidan_mclau) 的推文</a>：事实核查：不正确。o1-mini 更好并不是因为它思考时间更长，它只是一个更好的模型。引用 Amir Efrati (@amir) 的话：OpenAI 在不损失太多能力的情况下缩小其推理模型的能力...</li><li><a href="https://x.com/OpenAIDevs/status/1836506351062716701">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：再次将速率限制（rate limits）提高了 5 倍：o1-preview：每分钟 500 次请求；o1-mini：每分钟 1000 次请求。引用 OpenAI Developers (@OpenAIDevs) 的话：我们已经为... 提高了 OpenAI o1 API 的速率限制。</li><li><a href="https://x.com/polynoamial/status/1836872735668195636?s=61">来自 Noam Brown (@polynoamial) 的推文</a>：@OpenAI 正在为新的 multi-agent 研究团队招聘 ML 工程师！我们将 multi-agent 视为通往更好 AI 推理的路径。不需要之前的 multi-agent 经验。如果你想研究...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1286145849935528037)** (4 条消息): 

> - `NeurIPS 2024 筹备`
> - `住宿物流`
> - `温哥华活动更新` 


- **NeurIPS 2024 频道已创建**：应大众要求，已创建一个专门的 **NeurIPS 2024** 频道，以便参与者了解活动详情。
   - 鼓励成员在频道中回复，以获取今年 12 月在温哥华发生的一切最新动态。
- **NeurIPS 住房预订**：一位组织者正在考虑为 **NeurIPS 2024** 期间预订一整套房子，请感兴趣的参与者告知。
   - 强调贡献者应准备好分摊整个星期的住宿费用，而不仅仅是几天。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1286151710062149745)** (14 条消息🔥): 

> - `Cohere RAG API`
> - `Client Design Success` (客户设计成功)
> - `504 Gateway Timeout Errors` (504 Gateway Timeout 错误)
> - `Learning AI with Cohere` (使用 Cohere 学习 AI)
> - `Community Engagement` (社区参与)


- **使用 RAG API 构建专家级 AI**：一位成员目前正使用 Cohere 的 **RAG API** 开发一个专注于小型游戏利基市场的专家级 AI。
   - 他们对利用该 API 开发利基应用表示兴奋。
- **客户喜欢这个设计！**：一位成员成功说服了客户认可其设计的价值，并表示：*“我的设计非常酷，他们正需要它。”*
   - 这引发了社区的支持性回应，共同庆祝这一胜利。
- **遇到 504 Gateway Timeout 错误**：一位成员反映，在 Python SDK 中调用耗时较长的 **client.chat** 时收到了 **504 Gateway Timeout** 错误。
   - 讨论显示，其他社区成员也面临类似问题并正在寻找解决方案。
- **通过应用进行学习**：另一位成员鼓励大家领取每月 **1000 次免费 API 调用**的试用密钥，认为这是学习的好方法。
   - 另一位成员表示赞同，强调亲手实践是学习 AI 的最佳途径。
- **欢迎社区新成员**：社区欢迎了一位新成员，并对其加入表示高兴。
   - 这展示了社区互助的氛围，并鼓励成员积极参与。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1286117295265153024)** (7 条消息): 

> - `Command pricing` (Command 定价)
> - `Dataset access for Aya-101` (Aya-101 的数据集访问)
> - `Using Command for tagging` (使用 Command 进行标签分类)
> - `Efficiency of new models` (新模型的效率)


- **Command 定价说明**：成员们讨论了 **Command** 版本的相关成本，指出输入约为 **每 1M tokens $1.00**，输出为 **$2.00**。
   - 尽管提供了定价信息，但共识是应转向使用 **Command-R** 或 **Command-R+**，以获得更好的性能和成本效益。
- **关于 Aya-101 数据集的查询**：一位成员询问如何访问 **Aya-101** 微调所用的数据集，特别是 **ShareGPT-Command** 数据集。
   - 目前没有关于数据集访问的直接回复，但大家对该数据集仍保持好奇。
- **使用 Command 进行反馈标签分类**：一位成员解释说，他们使用 **Command** 对简短的反馈片段进行分类，但需要它能根据需要创建新类别。
   - 另一位成员建议尝试 **Command-R** 或 **Command-R+**，以提升标签分类的功能性。
- **性能与经济性选择**：讨论强调，与性能更强的新选项相比，使用 **Command** 等旧模型并不经济。
   - 成员们建议，新模型不仅更智能，而且在各种用例中都更具成本效益。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1286356089851613275)** (2 条消息): 

> - `Rerank Fine-Tuning` (Rerank 微调)
> - `RAG Results Impact` (RAG 结果影响)


- **多语言 Rerank 的不一致性**：一位用户报告称，在英文文本上使用 **_rerank_multilingual_v3_** 时出现了奇怪的结果：对于与内容相似的问题，得分 **<0.05**；对于非常相似的问题，得分 **<0.18**。
   - 相比之下，切换到 **_rerank_english_v3_** 后，相似查询的得分大幅提升至 **0.57** 和 **0.98**，这引发了对该模型有效性的质疑。
- **德语与英语 Rerank 性能对比**：在德语文本上使用 **_multilingual_v3_** 时，用户注意到得分为 **0.66**，而使用 **_english_v3_** 则为 **0.99**。
   - 这种不一致性严重影响了用户的 **RAG 结果**，因为它过滤掉了所有相关的 Chunks，导致用户对 Rerank 模型产生担忧。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1286039874138406962)** (22 条消息🔥): 

> - `OpenAI o1 models`
> - `Knowledge Cutoff`
> - `Qwen 2.5 72B Performance`
> - `Livecodebench Comparison`
> - `AI Reasoning Ability` 


- **OpenAI o1 模型表现亮眼**：@DeryaTR_ 报告称，在针对博士级项目测试 **o1-mini** 模型后，其表现可与生物医学领域的一位**优秀的博士生**相媲美。
   - 该模型被认为是该用户测试过的最出色的模型之一，展示了其在学术应用中的潜力。
- **Knowledge Cutoff 困扰开发者**：一位成员指出，**Knowledge Cutoff 是 23 年 10 月**，这影响了 AI 处理最新进展的效用。
   - 这一限制让像 @xeophon 这样的用户感到沮丧，他提到了这在编程时带来的挑战。
- **Qwen 2.5 72B 占据领先地位**：Qwen 2.5 72B 已成为 **Open Weights** 智能领域的新领导者，在独立评估中超越了像 **Llama 3.1 405B** 这样更大的模型。
   - 虽然它在 MMLU 上略微落后，但在**编程和数学**方面表现出色，作为一个 Dense 模型并拥有 128k Context Window，提供了一个*更廉价的替代方案*。
- **Livecodebench 显示出实力**：成员们注意到 **Livecodebench** 的数据令人印象深刻，据报道在使用经典的 Leetcode 题目时，其表现与 **Sonnet** 相当。
   - 然而，成员在实际编程使用中发现了局限性，特别是对于 o1 未知的最新发布的库。
- **关于 AI 推理能力的讨论**：围绕 o1-mini 等模型的推理能力展开了讨论，通常在不使用 Reflection 类型方法的任务性能上将其与 Qwen 2.5 进行比较。
   - 尽管目前的对比显示出 o1 的优越性，但用户对未来通过进一步增强带来的性能提升表示乐观。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DeryaTR_/status/1836434726774526381">来自 Derya Unutmaz, MD (@DeryaTR_) 的推文</a>：在过去的几天里，我一直在测试 OpenAI o1 模型，主要是 o1-mini，用于开发博士或博士后级别的项目。我可以自信地声称，o1 模型可以与一位优秀的博士生相媲美...</li><li><a href="https://x.com/HaveFunWithAI/status/1836749726554702027">来自 HaveFunWithAI (@HaveFunWithAI) 的推文</a>：o1-mini 擅长数学。作为参考：qwen2.4-math-72b-instruct（刚刚发布，SOTA 开源数学模型）在代码执行和集成方法（n=256）方面并不优于 o1-mini https://qwe...</li><li><a href="https://x.com/artificialanlys/status/1836822858695139523?s=46">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Open Weights 智能领域出现了新的领导者！Qwen2.5 72B 在我们的独立评估中位居 Open Weights 模型之首，包括与规模大得多的 Llama 3.1 405B 相比。Qwen 2.5 72B 于昨天发布...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

SnailBot 新闻：<@&1216534966205284433>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1286062936984653846)** (6 条消息): 

> - `OpenInterpreter 错误处理`
> - `Agent 性能测试`
> - `OpenInterpreter 处理能力`
> - `模型性能对比`
> - `使用 OpenInterpreter 的任务成功案例` 


- **导航 OpenInterpreter 错误信息**：一位用户在向 OpenInterpreter 输入数据时遇到错误并请求协助，希望能有解决该问题的详细步骤。
   - 有人建议 *“私信我错误详情”* 以帮助识别问题。
- **活跃的 Agent 评估**：另一位用户分享了他们连续约一周使用 OpenInterpreter 的 Agent 的经历，表明他们正在进行深入的实操测试。
   - 这突显了社区用户对 Agent 性能持续进行的探索和评估。
- **澄清 OpenInterpreter 的处理功能**：一位成员讨论了客户关于 OpenInterpreter 如何独立处理信息以及何时依赖 Chat GPT 的疑问。
   - 用户对连接到 Chat GPT 时的 CPU 占用效率表示担忧，并寻求社区的看法。
- **在 OpenInterpreter 中对比模型性能**：一位用户讨论了他们使用各种模型的经验，最终发现 **microsoft/Wizardlm 8x22B** 的表现优于 **llama 405B** 等其他选择。
   - 他们注意到 Wizardlm 比之前的模型更能实现任务的一次性完成。
- **使用 OpenInterpreter 的成功案例**：同一位用户分享了他们使用 OpenInterpreter 的成功经验，例如对大量文件进行分类以及创建桌面快捷方式。
   - 他们鼓励其他人分享使用该工具成功完成的任务类型，以促进社区内的想法交流。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1286113795172864041)** (6 条消息): 

> - `Perplexity 浏览器设置`
> - `浏览器用户体验`
> - `Windows 与 Edge 兼容性` 


- **用户询问 Perplexity 是否为默认浏览器**：一位用户询问 **Perplexity** 是否在 Chrome 或其他浏览器中被设置为默认浏览器。
   - 另一位用户确认它**没有**被设置为他们的默认浏览器。
- **多名用户面临类似问题**：一位用户提到大约有 **20 位其他用户** 在使用 **01** 时遇到了相同的问题。
   - 他们通过询问 **01** 社区内的几位用户确认了这一点。
- **发生问题时的浏览器使用环境**：一位用户指出他们在遇到该问题时正在使用 **Edge**。
   - 这表明在 **Windows** 上，不同浏览器之间的问题表现可能有所不同。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1286067264583372830)** (5 条消息): 

> - `RAG 聊天应用开发`
> - `用于 PDF 交互的多模态模型`
> - `响应中的图文整合`
> - `AI 使用案例示例`
> - `AI 生成内容` 


- **为 PDF 开发 RAG 聊天应用**：一位成员正在开发一个 **RAG 聊天应用**，允许用户与特定的 **PDF 文档** 进行交互，并正在寻求关于如何处理包含文本和图像的响应的建议。
   - 另一位成员建议使用 **tokens** 代表图像路径，并对图像进行摘要以节省上下文 tokens，同时与现有的 PDF to text parser（解析器）集成。
- **PDF 交互中的图像上下文处理**：讨论强调 **PDF to text parser** 可能需要包含图像链接或 tokens，以便有效地检索上下文和图像。
   - 讨论的另一种方法包括使用 base64 图像编码并直接从图像中提取文本，以实现更好的上下文整合。
- **Google 令人印象深刻的 AI 创作**：一位成员分享了一个仅用 **10 秒钟** 制作的 AI 创作链接，并称赞 **GoogleAI** 的一项新功能非常令人印象深刻。
   - 随后的评论强调了它的**实用性**以及作为顶级 AI 使用案例的潜力，并对 Google 的工作表示赞赏。
- **多模态模型使用建议**：一位成员建议使用可以直接读取图像和文本的**多模态模型**，并指出从图像中提取文本非常直接。
   - 这种方法为将图像数据整合到对话响应中提供了一种**简便的测试方法**。



**提到的链接**：<a href="https://x.com/sunglassesface/status/1836527799470854557">来自 😎orlie (@sunglassesface) 的推文</a>：Google 终于推出了令人印象深刻的东西。我刚刚测试了它……太疯狂了。我用 10 秒钟做出了这个。引用 Wolfram Ravenwolf 🐺🐦‍⬛ (@WolframRvnwlf) 的话：同意。这非常令人印象深刻……

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1286123402041495593)** (10 条消息🔥): 

> - `OBS Screen Recording`
> - `Screenity Recorder`
> - `Moshi Speech Models`
> - `GRIN MoE` 


- **OBS 仍然是屏幕录制的首选**: 成员们讨论了将 **OBS** 作为屏幕录制的一个稳健选项，尽管有些人表示更倾向于更简单的解决方案。
   - 一位成员强调了他们一直使用 OBS，而其他人则在寻找具有简单缩放效果的替代方案。
- **Screenity 作为一个用户友好的替代方案出现**: 一位用户分享了 [Screenity](https://github.com/alyssaxuu/screenity)，这是一个免费且注重隐私的屏幕录像机，可以同时捕捉屏幕和摄像头。
   - 该工具旨在提供一个比 OBS 更易用的解决方案，迎合那些寻求简单录屏软件的用户。
- **Moshi 模型在语音转语音应用中首次亮相**: 成员们宣布发布 **Moshi** 语音转语音模型，该模型支持全双工语音对话，并将文本 Token 与音频对齐。
   - 这个基础模型拥有建模对话动态等独特功能，并以 Pytorch 版本实现，采用 bf16 精度量化。
- **GRIN MoE 在更少参数下表现出潜力**: 讨论围绕 [GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE) 展开，它仅凭 **6.6B 激活参数**就实现了令人印象深刻的性能，特别是在编程和数学任务中。
   - 它利用 **SparseMixer-v2** 进行梯度估计，同时消除了专家并行（expert parallelism）和 Token 丢弃（token dropping），这使其区别于传统的 MoE 训练方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/kyutai/moshiko-pytorch-bf16">kyutai/moshiko-pytorch-bf16 · Hugging Face</a>: 未找到描述内容</li><li><a href="https://huggingface.co/microsoft/GRIN-MoE">microsoft/GRIN-MoE · Hugging Face</a>: 未找到描述内容</li><li><a href="https://github.com/alyssaxuu/screenity">GitHub - alyssaxuu/screenity: 免费且隐私友好的无限制屏幕录像机 🎥</a>: 免费且隐私友好的无限制屏幕录像机 🎥 - alyssaxuu/screenity
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1286234434625994795)** (3 条消息): 

> - `Gemma2 DPO Config Issue`
> - `Chat Template Modification` 


- **Gemma2 无法使用 DPO 数据运行**: 一位用户分享了用于 **DPO 数据**的 **Gemma2 9b** 配置，但遇到了 **TemplateError**，提示 *'Conversation roles must alternate user/assistant/user/assistant...'（对话角色必须交替出现 user/assistant/user/assistant...）*。
   - 由于数据集结构包含 'prompt' 而不是 'conversation'，在尝试应用聊天模板时发生了错误。
- **对 Gemma 聊天模板的修改**: 用户修改了 `chat_templates.py` 中的 **Gemma** 模板，以更改角色检查逻辑，试图解决模板错误。
   - 他们修改了代码以在消息角色上启动循环，并询问这种修改是否合适。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1286143117677953085)** (1 条消息): 

> - `PyTorch conference`
> - `New work announcements`
> - `Community engagement` 


- **欢迎 PyTorch 会议参与者**: 向所有从 **PyTorch 会议**加入的人表示热烈欢迎。
   - 鼓励参与者在指定频道提问并与其他成员互动。
- **查看新工作**: 邀请与会者查看频道 <#1236040539409879170> 中正在开发的新工作。
   - 这突出了社区中正在进行的努力和创新。
- **鼓励社区互动**: 鼓励成员随时在频道中发送消息并提出任何问题。
   - 这促进了与会者之间的**社区参与**和支持。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1286043635888164904)** (8 条消息🔥): 

> - `会议直播查询`
> - `修复 kv-caching 的 GitHub PR`
> - `HH RLHF 数据集文档`
> - `默认偏好数据集构建器` 


- **会议直播查询**：成员们讨论了**会议直播**的可能性，其中一人对其是否可用表示不确定。
   - *‘Idk :/’* 是对该查询的回应，表明缺乏相关信息。
- **GitHub PR 解决 kv-Caching 问题**：链接了一个标题为 **Fix kv-cacheing and bsz > 1 in eval recipe** 的拉取请求，详细说明了 [SalmanMohammadi](https://github.com/pytorch/torchtune/pull/1622) 的更改。
   - 该 PR 旨在解决与 kv-caching 相关的问题，并在持续开发的背景下被认为具有重要意义。
- **关于 HH RLHF 数据集展示的讨论**：一名成员询问为何 **HH RLHF 数据集** 没有文档记录，并建议将其作为标准偏好示例，而不是其他选项。
   - *‘不确定，它应该被公开...’* 这种观点鼓励将该数据集包含在文档中。
- **默认偏好数据集构建器的未来计划**：分享了关于创建一个**默认偏好数据集构建器**的计划，该构建器将假定使用 **ChosenToRejectedMessages**。
   - 这一合作受到了热烈欢迎，评论 *‘Dope’* 表明了对这一开发的强力支持。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1622">Fix kv-cacheing and bsz &gt; 1 in eval recipe by SalmanMohammadi · Pull Request #1622 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的任何问题。关闭 #160...

  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1286315655674462311)** (5 条消息): 

> - `DSPy 程序优化`
> - `DSPy 中的 Bootstrapping`
> - `提示词优化成本`
> - `MIPRO 与优化`
> - `LLM 中的非确定性` 


- **对 DSPy 程序优化的兴奋**：一名成员分享了他们在编码两个月后的成功，强调了 **BSFSWRS 优化器** 在其复杂的 LM 设置中的有效性。
   - *未来是光明的，伙计们！*
- **对优化提示词成本的担忧**：另一名成员评论了为 DSPy 优化提示词可能带来的高昂成本，表明这可能是一项重大投资。
   - *优化一个提示词肯定非常昂贵。*
- **MIPRO：一场昂贵的冒险**：有人幽默地建议尝试将 **o1 与 MIPRO** 结合使用，并开玩笑地警告了其中涉及的财务风险。
   - *公认的破产方式。*
- **澄清 DSPy 中的 Bootstrapping**：一名成员询问了关于 bootstrapping 的问题，其目的是在 pipeline 中生成步骤示例并验证过程的成功。
   - 他们对这种方法在考虑到 **LLM** 的非确定性时如何运作表示困惑。
- **理解 Bootstrapping 响应**：一名成员确认 bootstrapping 会生成中间示例，并通过最终预测的成功来验证其正确性。
   - 他们指出，如果最终结果正确，则假定中间步骤有效，可作为 few-shot 示例使用。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286392361928097792)** (4 messages): 

> - `tinybox motherboard`
> - `CLANG bounty`
> - `Pull Request discussions` 


- **关于 tinybox 主板的咨询**：一位用户询问了 **tinybox red and green** 型号中使用的特定 **motherboard**（主板），寻求硬件细节的澄清。
   - 这反映了用户对 **tinybox** 设备硬件方面的持续关注。
- **关于 CLANG 悬赏的讨论**：一位成员询问名为“用 mmap 替换 CLANG dlopen + 移除链接步骤”的 **bounty** 是否涉及手动处理目标文件中的 **relocations**（重定位）。
   - 这表明了对 **CLANG** 在 **tinygrad** 环境中集成的更深层次技术探索。
- **相关 Pull Request 链接**：一位用户分享了 **Pull Request #6299** 和 **#4492** 的链接，讨论了用 **mmap** 替换 **dlopen** 以及 **Clang jit** 的实现。
   - 这些贡献旨在提升性能，特别是在 **M1 Apple devices** 上，展示了社区在优化代码执行方面的努力。
- **对悬赏结果的好奇**：一位用户对谁可能领取 **CLANG** 更改的 **bounty** 表示兴奋，突显了社区的参与度。
   - 这种互动反映了一种协作氛围，用户们热衷于见证贡献者的成果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/6299">Replace dlopen with mmap in CLANG by christopherm99 · Pull Request #6299 · tinygrad/tinygrad</a>：在 M1 MacBook Pro 上进行了性能测试。来自 tinygrad.runtime.ops_clang import ClangProgram，使用 open(&amp;quot;test.o&amp;quot;, &amp;quot;rb&amp;quot;) as f: lib = f.read() 循环 1000 次...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4492">Clang jit by uuuvn · Pull Request #4492 · tinygrad/tinygrad</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1286283907553165343)** (1 messages): 

> - `OpenAI's o1 model`
> - `Large reasoning models` 


- **OpenAI 的 o1 模型引起关注**：一段名为《o1 - 发生了什么？为什么 o1 是模型的第三种范式 + 你可能不知道的 10 件事》的 [YouTube 视频](https://m.youtube.com/watch?v=KKF7kL0pGc4) 提供了关于 **OpenAI o1** 可能如何构建的引人入胜的总结。
   - *即使是怀疑论者也将其称为“大推理模型” (large reasoning model)*，因为它具有独特的方法论以及对未来模型开发的影响。
- **o1 与其他模型的区别**：在同一段视频中，讨论了为什么 **o1** 被公认为 AI 建模的新范式，表明了设计理念的重大转变。
   - 采用此类模型的影响可能会导致对 AI 推理能力的更好理解，使其成为该领域的一个关键话题。



**提到的链接**：<a href="https://m.youtube.com/watch?v=KKF7kL0pGc4">o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know</a>：o1 是不同的，即使是怀疑论者也将其称为“大推理模型”。但为什么它如此不同，这对未来又意味着什么？当模型...

  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1286334772737736734)** (1 messages): 

> - `LunoSmart AI Venture`
> - `应用技术栈`
> - `跨平台开发`
> - `Machine Learning 专业知识` 


- **LunoSmart 发布与产品服务**：Kosi Nzube 宣布启动他的 AI 创业项目 [LunoSmart](https://www.lunosmart.com)，展示了其对 AI 驱动应用和创新解决方案的关注。
   - 该项目旨在跨多个平台提供**无缝**、**高效**且**智能的体验**。
- **多样化技术栈概览**：Kosi 的应用技术栈包括 **Java**、**Flutter**、**Spring Boot**、**Firebase** 和 **Keras**，强调现代开发方法。
   - 应用已在 Android 和 Web 平台上架，实现了广泛的可访问性。
- **跨平台开发专长**：Kosi 擅长使用 **Flutter** 和 **Firebase SDK** 进行跨平台开发，增强了跨设备的功能性。
   - 他在移动应用方面的经验特别突出了他使用 **Android Studio** 和 **Java** 创建 **native Android** 应用的熟练程度。
- **Machine Learning 技能集**：凭借扎实的 **Machine Learning** 背景，Kosi 利用 **Keras**、**Weka** 和 **DL4J** 等工具创建智能模型。
   - 他在该领域的经验始于 **2019** 年，展示了他致力于推动 AI 技术发展的决心。



**提到的链接**：<a href="https://kosinzube.online/">Kosi Nzube</a>：AI 开发者。我使用我最喜欢的工具进行编程：Java、Flutter 以及基于 Python 的 Keras。lunosmart.com 创始人。

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

flozi00: https://mistral.ai/news/september-24-release/

Mistral 紧随其后降价了 💪
  

---



---



---



---



{% else %}


> 完整的逐频道细分内容已在邮件中截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的 Web 版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}