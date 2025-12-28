---
companies:
- meta-ai-fair
- ai2
- qualcomm
- mediatek
- arm
- ollama
- together-ai
- fireworks-ai
- weights-biases
- cohere
- weaviate
date: '2024-09-25T23:54:30.322812Z'
description: '**Meta** 发布了 **Llama 3.2**，其中包括基于冻结版 Llama 3.1 构建的 **3B** 和 **20B**
  视觉适配器等新型多模态版本，其性能足以与 **Claude Haiku** 和 **GPT-4o-mini** 相媲美。与此同时，**AI2** 推出了多模态
  **Molmo 72B** 和 **7B** 模型，在视觉任务上的表现优于 Llama 3.2。


  Meta 还推出了具备 **128k 上下文窗口的 1B 和 3B 模型**，旨在与 **Gemma 2** 和 **Phi 3.5** 竞争，并暗示将与 **高通
  (Qualcomm)**、**联发科 (Mediatek)** 和 **Arm** 展开合作，推动端侧 AI 的发展。此次发布的 Llama 1B 和 3B 模型训练数据量高达
  **9 万亿 token**。


  合作伙伴的同步发布包括 **Ollama**、提供免费 11B 模型访问权限的 **Together AI** 以及 **Fireworks AI**。此外，由
  **Weights & Biases**、**Cohere** 和 **Weaviate** 联合推出的全新 **RAG++ 课程**，基于丰富的生产实践经验，为检索增强生成（RAG）系统提供了系统的评估和部署指导。'
id: be2493bf-e679-4d29-adbb-b68878bb4ddc
models:
- llama-3-2
- llama-3-1
- claude-3-haiku
- gpt-4o-mini
- molmo-72b
- molmo-7b
- gemma-2
- phi-3-5
- llama-3-2-vision
- llama-3-2-3b
- llama-3-2-20b
original_slug: ainews-llama-32-on-device-1b3b-and-multimodal
people:
- mira-murati
- daniel-han
title: Llama 3.2：1B/3B 端侧模型与 11B/90B 多模态模型（附带 AI2 Molmo 亮点）
topics:
- multimodality
- vision
- context-windows
- quantization
- model-release
- tokenization
- model-performance
- model-optimization
- rag
- model-training
- instruction-following
---

<!-- buttondown-editor-mode: plaintext -->**9000:1 的 token:param 比例就是你所需要的一切。**

> 2024年9月24日至9月25日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 服务端（**223** 个频道，**3218** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**316 分钟**。你现在可以艾特 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天来自 [Mira Murati](https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 和 [FB Reality Labs](https://news.ycombinator.com/item?id=41650047) 的消息很大，但今天你可以真正使用的技术新闻是 Llama 3.2：


![image.png](https://assets.buttondown.email/images/057082e6-48ab-452c-9b88-c98e14477edb.png?w=960&fit=max)


正如 Zuck 所预告以及在 Llama 3 论文中预览的那样（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-llama-31-the-synthetic-data-model/)），Llama 3.2 的多模态版本如期发布，在冻结的 Llama 3.1 基础上增加了 3B 和 20B 的视觉适配器（vision adapter）：


![image.png](https://assets.buttondown.email/images/7eb80330-6b79-4854-864c-b9fe52dec70c.png?w=960&fit=max)


11B 模型与 Claude Haiku 相当或略好，90B 模型与 GPT-4o-mini 相当或略好，尽管你可能需要更深入地挖掘才能发现它在 MMMU 上以 60.3 的得分[落后于 4o、3.5 Sonnet、1.5 Pro 和 Qwen2-VL 多少](https://mmmu-benchmark.github.io/#leaderboard)。

Meta 因其开源行为受到称赞，但不要错过 AI2 今天同样发布的 [Molmo 72B 和 7B 多模态模型](https://x.com/allen_ai/status/1838956313902219595)。[/r/localLlama 已经注意到](https://www.reddit.com/r/LocalLLaMA/comments/1fpb4m3/molmo_models_outperform_llama_32_in_most_vision/) Molmo 在视觉方面的表现优于 3.2：


![image.png](https://assets.buttondown.email/images/4564aac0-56b5-470f-817b-04aadf92003a.png?w=960&fit=max)


Meta 带来的更大、更令人愉悦且印象深刻的惊喜是全新的支持 128k 上下文的 1B 和 3B 模型，它们现在正与 Gemma 2 和 Phi 3.5 竞争：


![image.png](https://assets.buttondown.email/images/c01faa12-3547-4f58-93ab-70b28c745f3e.png?w=960&fit=max)


发布说明暗示了与 Qualcomm、Mediatek 和 Arm 在设备端（on-device）进行的非常紧密的合作：

> 今天发布的权重基于 BFloat16 数值。我们的团队正在积极探索运行速度更快的量化变体，我们希望很快能分享更多相关信息。

不要错过：

- [发布博客文章](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- 来自 [@AIatMeta](https://x.com/AIatMeta/status/1839018076446294060) 的后续技术细节，披露了 Llama 1B 和 3B 的 **9 万亿 token 数量**，以及 [Daniel Han 的快速架构分析](https://x.com/danielhanchen/status/1839009095883567520?s=46)
- 更新后的 [HuggingFace 集合](https://huggingface.co/meta-llama)，包括 [Evals](https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0)
- [Llama Stack 发布](https://github.com/meta-llama/llama-stack)（参见 [RFC 在此](https://github.com/meta-llama/llama-stack/issues/6)）

合作伙伴发布：

- [Ollama](https://ollama.com/blog/llama3.2)
- [Together AI](https://x.com/togethercompute/status/1839013617817309563)（提供 **免费** 的 11B 模型访问，限速 5 rpm 直至年底）
- [Fireworks AI](https://www.linkedin.com/posts/fireworks-ai_genai-llama32-atatmeta-activity-7244771399779721219-A0du?utm_source=share&utm_medium=member_ios)

---

**本期由 RAG++ 赞助：来自 Weights & Biases 的新课程**。超越 RAG POC，学习如何进行系统评估、正确使用混合搜索，并让你的 RAG 系统具备工具调用（tool calling）能力。基于**在生产环境中运行客服机器人 18 个月**的经验，来自 Weights & Biases、Cohere 和 Weaviate 的行业专家将展示如何构建达到部署级别的 RAG 应用。包含来自 Cohere 的免费额度，助你开启旅程！

[
![image.png](https://assets.buttondown.email/images/547df650-7220-4489-ac2a-ca08c08b42df.png?w=960&fit=max)
](http://wandb.me/ainews-course)

> **Swyx 评论**：哇，2 小时内有 74 节课。我以前参与过这种剪辑非常紧凑的课程内容制作，这竟然是免费的，太令人惊讶了！第 1-2 章涵盖了一些必要的 RAG 基础知识，但随后很高兴看到第 3 章教授了重要的 ETL 和 IR 概念，并在第 4 和 5 章中学习了关于交叉编码（cross encoding）、排名融合（rank fusion）和查询转换（query translation）的新知识。我们很快会在直播中涵盖这些内容！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**高级语音模式（Advanced Voice Mode）发布**

- OpenAI 正在一周内向 ChatGPT Plus 和 Team 用户推行高级语音模式。
- [@sama](https://twitter.com/sama/status/1838644910985003126) 宣布：“高级语音模式今天开始推出！（将在本周内完成）希望你们觉得这值得等待 🥺🫶”
- [@miramurati](https://twitter.com/miramurati/status/1838642696111689788) 确认：“ChatGPT 中的所有 Plus 和 Team 用户”
- [@gdb](https://twitter.com/gdb/status/1838662392970150023) 指出：“Advanced Voice 正在广泛推出，实现了与 ChatGPT 流畅的语音对话。这让你意识到在电脑上打字是多么不自然：”

新的语音模型具有更低的延迟、打断长回答的能力，并支持通过记忆（memory）来实现个性化回答。它还包括了新的语音和改进的口音。

**Google Gemini 1.5 Pro 与 Flash 更新**

Google 宣布了其 Gemini 模型的重大更新：

- [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1838618720677302560) 推特表示：“今天，我们很高兴发布两个全新的、生产就绪的 Gemini 1.5 Pro 和 Flash 版本。🚢 它们基于我们最新的实验版本，并在长上下文理解、视觉和数学方面有显著改进。”
- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838613238088634835) 总结了关键改进：“MMLU-Pro 基准测试提升 7%，MATH 和 HiddenMath 提升 20%，视觉和代码任务提升 2-7%”
- Gemini 1.5 Pro 降价超过 50%
- 输出速度提升 2 倍，延迟降低 3 倍
- 提高速率限制：Flash 为 2,000 RPM，Pro 为 1,000 RPM

这些模型现在可以处理 1000 页的 PDF、1 万行以上的代码以及长达一小时的视频。为了提高效率，输出内容缩短了 5-20%，且开发者可以自定义安全过滤器。

**AI 模型性能与基准测试**

- OpenAI 的模型在各项基准测试中处于领先地位：
  - [@alexandr_wang](https://twitter.com/alexandr_wang/status/1838637233169211838) 报告称：“OpenAI 的 o1 在 SEAL 排名中占据主导地位！🥇 o1-preview 在关键类别中领先：- Agentic Tool Use (Enterprise) 排名第 1 - 指令遵循（Instruction Following）排名第 1 - 西班牙语排名第 1 👑 o1-mini 在 Coding 领域领先”
- 不同模型之间的对比：
  - [@bindureddy](https://twitter.com/bindureddy/status/1838723326895886618) 指出：“Gemini 的真正超能力——它比 o1 便宜 10 倍！如果你想体验，新的 Gemini 已在 ChatLLM 团队版上线。”

**AI 开发与研究**

- [@alexandr_wang](https://twitter.com/alexandr_wang/status/1838706686837821941) 讨论了 LLM 开发的阶段：“我们正在进入 LLM 开发的第 3 阶段。第 1 阶段是早期探索，从 Transformer 到 GPT-3；第 2 阶段是规模化（scaling）；第 3 阶段是创新阶段：除了 o1 之外，还有哪些突破能让我们进入新的准 AGI（proto-AGI）范式。”
- [@JayAlammar](https://twitter.com/JayAlammar/status/1838720544352686414) 分享了关于 LLM 概念的见解：“第一章为理解 LLM 铺平了道路，提供了相关概念的历史和概述。公众应该了解的一个核心概念是，语言模型不仅仅是文本生成器，它们还可以形成其他对解决问题有用的系统（embedding、分类）。”

**AI 工具与应用**

- [@svpino](https://twitter.com/svpino/status/1838550186756366678) 讨论了 AI 驱动的代码审查：“不受欢迎的观点：代码审查（Code reviews）很愚蠢，我迫不及待地想让 AI 完全接管。”
- [@_nerdai_](https://twitter.com/_nerdai_/status/1838706149178126394) 分享了一个 ARC 任务求解器，允许人类与 LLM 协作：“利用便捷的 @llama_index Workflows，我们构建了一个 ARC 任务求解器，允许人类与 LLM 协作解决这些 ARC 任务。”

**梗与幽默**

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1838695343351042483) 开玩笑说：“我该发布一个壁纸 App 吗？”
- [@swyx](https://twitter.com/swyx/status/1838722558285484054) 幽默地评论了这一情况：“伙计们别吵了，mkbhd 只是把错误的 .IPA 文件上传到了 App Store。耐心点，他正在从头重新编译代码。与此同时，他私下给我发了一个真实 mkbhd app 的 TestFlight 链接。作为壁纸社区自封的傲罗（auror），我会调查并查明真相。”


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 高速推理平台：Cerebras 与 MLX**

- **刚刚获得了 Cerebras 的访问权限。每秒 2,000 个 token。** ([Score: 99, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1fosxwt/just_got_access_to_cerebras_2000_token_per_second/))：**Cerebras 平台**展示了令人印象深刻的推理速度，使用 **Llama3.1-8B** 模型达到了 **每秒 2,010 个 token**，使用 **Llama3.1-70B** 模型达到了 **每秒 560 个 token**。用户对这种性能表示惊讶，并表示他们仍在探索这种高速推理能力的潜在应用场景。
  - 原帖确认，Cerebras 平台支持 **JSON outputs**。该平台的访问权限通过 **注册和邀请系统** 授予，用户可前往 [inference.cerebras.ai](https://inference.cerebras.ai/)。
  - 讨论的潜在应用包括 **Chain of Thought (CoT) + RAG 结合语音**，可能创建一个能够实时提供专家级回答的 **Siri/Google Voice 竞争对手**。Cerebras 上的 **语音 demo** 可以在 [cerebras.vercel.app](https://cerebras.vercel.app/) 体验。
  - 该平台被拿来与 **Groq** 进行比较，据报道 Cerebras 甚至更快。**SambaNova APIs** 被提作为一种替代方案，提供类似的推理速度（**1500 tokens/second**）且无需排队等待，同时用户也注意到了这种高速推理在实时应用和安全性方面的潜力。
- **MLX 批量生成非常酷！** ([Score: 42, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fodyal/mlx_batch_generation_is_pretty_cool/))：**MLX paraLLM 库**使 **Mistral-22b** 的生成速度提升了 **5.8 倍**，在 **Batch Size 为 31** 时，速度从 **每秒 17.3 个 token** 增加到 **101.4 tps**。峰值内存占用从 **12.66GB** 增加到 **17.01GB**，每个额外的并发生成大约需要 **150MB**，作者成功在 **64GB M1 Max 设备**上运行了 **22b-4bit 模型**的 **100 个并发 Batch**，且未超过 **41GB** 的 Wired Memory。
  - **能源效率**测试显示，在低功耗模式下，**Batch Size 为 100** 时，**Mistral-7b** 为 **每瓦 10 个 token**，**22b** 为 **每瓦 3.5 个 token**。这种效率在每瓦单词数方面可与人类大脑的性能相媲美。
  - 该库是 **Apple-only** 的，但对于 **NVIDIA/CUDA**，通过 **vLLM**、**Aphrodite** 和 **MLC** 等工具也存在类似的 Batching 能力，尽管设置过程可能更复杂。
  - 虽然这项技术不适用于提高普通聊天场景的速度，但对于**合成数据生成 (synthetic data generation)** 和**数据集蒸馏 (dataset distillation)** 非常有价值。


**主题 2. Qwen 2.5：在消费级硬件上的突破性性能**

- **[Qwen2-VL-72B-Instruct-GPTQ-Int4 在 4x P100 上达到 24 tok/s](https://i.redd.it/qzshr7c9vqqd1.png)** ([Score: 37, Comments: 52](https://reddit.com//r/LocalLLaMA/comments/1foae69/qwen2vl72binstructgptqint4_on_4x_p100_24_toks/)): **Qwen2-VL-72B-Instruct-GPTQ-Int4** 作为一个大型多模态模型，据报道在 **4x P100 GPU** 上的运行速度达到 **每秒 24 tokens**。该实现利用了 **GPTQ quantization** 和 **Int4 precision**，使得在显存有限的旧款 GPU 硬件上部署 **720 亿参数** 模型成为可能。
  - **DeltaSqueezer** 提供了一个 **GitHub** 仓库和 **Docker** 命令，用于在 **Pascal GPUs** 上运行 **Qwen2-VL-72B-Instruct-GPTQ-Int4**。该配置包含对 **P40 GPUs** 的支持，但由于 **FP16** 处理，可能会遇到加载缓慢的问题。
  - 该模型在测试政治图像时展示了合理的视觉和推理能力。文中提供了与 **Pixtral** 模型在同一图像上的输出对比，显示出类似的解释能力。
  - 关于视频处理的讨论显示，**7B VL** 版本消耗大量 VRAM。该模型在 **P100 GPUs** 上的性能被指出比 **3x 3090s** 更快，因为 **P100** 的 **HBM** 与 3090 的内存带宽相当。
- **Qwen 2.5 改变了游戏规则。** ([Score: 524, Comments: 121](https://reddit.com//r/LocalLLaMA/comments/1fohil2/qwen_25_is_a_gamechanger/)): **Qwen 2.5 72B** 模型在双 **RTX 3090s** 上高效运行，其中 **Q4_K_S (44GB)** 版本达到约 **16.7 T/s**，**Q4_0 (41GB)** 版本达到约 **18 T/s**。该帖子包含了用于设置 **Tailscale**、**Ollama** 和 **Open WebUI** 的 **Docker compose** 配置，以及用于更新和下载多个 AI 模型的 bash 脚本，包括 **Llama 3.1**、**Qwen 2.5**、**Gemma 2** 和 **Mistral** 的变体。
  - 设置中的 **Tailscale** 集成允许通过移动设备和 iPad 远程访问 **OpenWebUI**，从而能够通过浏览器随时随地使用 AI 模型。
  - 用户讨论了模型性能，建议尝试由 **lmdeploy** 提供的 **AWQ** (4-bit quantization)，以在 **70B models** 上获得潜在的更快性能。**32B** 和 **7B** 模型的对比显示，大模型在复杂任务上表现更好。
  - 讨论中表达了对硬件需求的兴趣，原帖作者指出选择 **双 RTX 3090s** 是为了高效运行 **70B models**，预计 **ROI** 为 6 个月。此外还提出了关于在 **Apple M1/M3** 硬件上运行模型的问题。


**主题 3. Gemini 1.5 Pro 002: Google 的最新模型令人印象深刻**

- **[Gemini 1.5 Pro 002 展现出令人印象深刻的基准测试数据](https://i.redd.it/75b3u6g8vvqd1.png)** ([Score: 102, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1fow9a9/gemini_15_pro_002_putting_up_some_impressive/)): Gemini 1.5 Pro 002 在各项基准测试中展现了**令人印象深刻的性能**。该模型在 **MMLU** 上达到了 **97.8%**，在 **HumanEval** 上达到了 **90.0%**，在 **MATH** 上达到了 **82.6%**，超越了之前的 SOTA 结果，并较其前代产品 Gemini 1.0 Pro 有了显著提升。
  - **Google 的 Gemini 1.5 Pro 002** 带来了重大改进，包括**价格降低 50% 以上**、**速率限制提高 2-3 倍**，以及 **2-3 倍的输出速度提升和更低的延迟**。该模型在 **MMLU (97.8%)** 和 **HumanEval (90.0%)** 等基准测试中的表现令人印象深刻。
  - 用户称赞了 Google 最近的进展，指出了他们**研究论文的发表**和 **AI Studio 实验场**。一些人将 Google 与其他 AI 公司进行了正面对比，其中 **Meta** 因其权重开放模型和详尽的论文而受到关注。
  - 讨论中提到了 **Gemini 的消费者版本**，一些用户发现其能力不如竞争对手。关于更新后的模型何时向消费者开放的推测从几天内到最迟 **10 月 8 日**不等。
- **[更新后的 Gemini 模型被声称是性价比最高的智能模型*](https://i.redd.it/a0txrr8w8sqd1.png)** ([Score: 291, Comments: 184](https://reddit.com//r/LocalLLaMA/comments/1fogic7/updated_gemini_models_are_claimed_to_be_the_most/)): Google 发布了 **Gemini 1.5 Pro 002**，声称它是**每美元最智能的 AI 模型**。该模型在各项基准测试中表现出**显著改进**，包括 **MMLU 上的 90% 评分**和 **HumanEval 上的 93.2%**，同时提供极具竞争力的定价：**每 1k 输入 token 0.0025 美元**，**每 1k 输出 token 0.00875 美元**。这些性能提升和极具成本效益的定价使 Gemini 1.5 Pro 002 成为 AI 模型市场中的强力竞争者。
  - **Mistral** 每月免费提供 **10 亿 token** 的 **Large v2**，用户注意到其强劲的性能。这与 Google 对 Gemini 1.5 Pro 002 的定价策略形成对比。
  - 用户批评了 Google 对 Gemini 模型的**命名方案**，建议采用基于日期的版本管理等替代方案。公告还透露了 API 用户将获得 **2-3 倍的速率限制提升**和**更快的性能**。
  - 讨论强调了**成本**、**性能**和**数据隐私**之间的权衡。一些用户为了数据控制更倾向于自托管，而另一些用户则欣赏 Google 的免费层级和 [AI Studio](https://aistudio.google.com/app/prompts/new_chat?pli=1) 提供的无限免费使用。


**主题 4. Apple Silicon vs NVIDIA GPUs 在 LLM 推理方面的对比**

- **HF 发布 Hugging Chat Mac App - 免费运行 Qwen 2.5 72B、Command R+ 等模型！** ([Score: 54, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fohtov/hf_releases_hugging_chat_mac_app_run_qwen_25_72b/))：Hugging Face 发布了 **Hugging Chat Mac App**，允许用户在 Mac 上免费本地运行 **Qwen 2.5 72B**、**Command R+**、**Phi 3.5** 和 **Mistral 12B** 等最先进的开源语言模型。该应用包含 **web search**（网页搜索）和 **code highlighting**（代码高亮）等功能，并计划推出更多功能，还包含 Macintosh、404 和 Pixel pals 主题等**隐藏彩蛋**；用户可以从 [GitHub](https://github.com/huggingface/chat-macOS) 下载并为未来的改进提供反馈。
- **低上下文速度对比：Macbook、Mac Studios 和 RTX 4090** ([Score: 33, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1fovw8h/low_context_speed_comparison_macbook_mac_studios/))：该帖子对比了 **RTX 4090**、**M2 Max Macbook Pro**、**M1 Ultra Mac Studio** 和 **M2 Ultra Mac Studio** 在运行 **Llama 3.1 8b q8**、**Nemo 12b q8** 和 **Mistral Small 22b q6_K** 模型时的性能。在所有测试中，**RTX 4090** 的表现始终优于 Mac 设备，**M2 Ultra Mac Studio** 通常位居第二，随后是 **M1 Ultra Mac Studio** 和 **M2 Max Macbook Pro**。作者指出，这些测试是在模型**刚加载**且未启用 **flash attention** 的情况下运行的，并对测试未实现确定性表示歉意。
  - 用户建议在 **RTX 4090** 上使用 **exllamav2** 以获得更好的性能，一名用户报告在 **RTX 3090** 上运行 **Llama 3.1 8b** 的生成速度达到 **104.81 T/s**。一些人指出，与 **gguf** 模型相比，**exl2** 过去存在质量问题。
  - 关于 Apple Silicon 的 **prompt processing speed**（提示词处理速度）的讨论，用户强调了由于缓存原因，初始 prompt 和后续 prompt 之间存在显著差异。**M2 Ultra** 处理 4000 个 **token** 需要 **16.7 秒**，而 **RTX 4090** 仅需 **5.6 秒**。
  - 用户探讨了提高 Mac 性能的选项，包括启用 **flash attention** 以及在运行 Linux 的 Mac 上添加 GPU 进行 prompt 处理的理论可能性，尽管驱动支持仍然有限。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与改进**

- **OpenAI 为 ChatGPT 发布高级语音模式**：OpenAI 已经推出了 [ChatGPT 的高级语音模式 (advanced voice mode)](https://www.reddit.com/r/OpenAI/comments/1fou4vi/openais_advanced_voice_mode_is_shockingly_good/)，该模式支持更自然的对话，包括中断和继续思路的能力。用户反馈称其有显著改进，但在允许用户完成思考方面仍存在一些限制。

- **Google 更新 Gemini 模型**：Google 宣布了 [更新的可用于生产环境的 Gemini 模型](https://www.reddit.com/r/singularity/comments/1fog8fd/updated_productionready_gemini_models_reduced_15/)，包括 Gemini-1.5-Pro-002 和 Gemini-1.5-Flash-002。此次更新包括降低价格、提高速率限制以及在各项基准测试中的性能提升。

- **新的 Flux 模型发布**：Realistic Vision 的创建者 [发布了一个名为 RealFlux 的 Flux 模型](https://www.reddit.com/r/StableDiffusion/comments/1fotv20/the_creator_of_realistic_vision_released_a_flux/)，可在 Civitai 上获取。用户指出它能产生不错的结果，但在面部特征方面仍存在一些局限性。

**AI 能力与基准测试**

- **Gemini 1.5 002 性能表现**：报告显示 [Gemini 1.5 002 在 MATH 基准测试中优于 OpenAI 的 o1-preview](https://www.reddit.com/r/singularity/comments/1fohi2z/gemini_15_002_beats_o1preview_on_math_and_it_does/)，且成本仅为 1/10，并且没有思考时间。

- **o1 的能力**：一名 OpenAI 员工表示 [o1 的表现能够达到顶尖博士生的水平](https://www.reddit.com/r/singularity/comments/1fo7rvl/openais_dane_vahey_says_gpt3_was_as_smart_as_a/)，在某些任务中超过人类的比例超过 50%。然而，一些用户对这一说法表示质疑，指出与人类相比，o1 在学习和适应能力方面存在局限。

**AI 开发工具与界面**

- **Invoke 5.0 更新**：[Invoke AI 工具迎来了重大更新](https://www.reddit.com/r/StableDiffusion/comments/1focbhe/invoke_50_massive_update_introducing_a_new_canvas/)，引入了带有图层的新 Canvas、Flux 支持和提示词模板。此次更新旨在为结合各种 AI 图像生成技术提供更强大的界面。

**AI 对社会和工作的影响**

- **职位取代预测**：Vinod Khosla 预测 [AI 将接管 80% 职业中 80% 的工作](https://www.reddit.com/r/OpenAI/comments/1fos72b/vinod_khosla_says_ai_will_take_over_80_of_work_in/)，引发了关于潜在经济影响和全民基本收入必要性的讨论。

- **AI 在执法领域的应用**：一款新的 [用于警察工作的 AI 工具](https://www.reddit.com/r/singularity/comments/1fo9gc4/ai_tool_that_can_do_81_years_of_detective_work_in/) 声称能在 30 小时内完成“81 年的侦探工作”，这既引发了对效率提高的兴奋，也引发了对潜在滥用的担忧。

**新兴 AI 研究与应用**

- **MIT 疫苗技术**：MIT 的研究人员开发了一种 [新的疫苗技术，可能仅需两针即可消除 HIV](https://www.reddit.com/r/singularity/comments/1foq5ab/new_mit_vaccine_technology_could_wipe_out_hiv_in/)，展示了 AI 加速医学突破的潜力。


---

# AI Discord 回顾

> 由 O1-mini 生成的摘要之摘要

**主题 1. 新 AI 模型发布与多模态增强**

- [**Llama 3.2 发布，具备多模态和边缘计算能力**](https://x.com/danielhanchen/status/1838987356810199153)：**Llama 3.2** 推出了多种模型规格，包括 **1B, 3B, 11B** 和 **90B**，支持多模态和 **128K 上下文长度**，并针对 **移动和边缘设备** 的部署进行了优化。
- [**Molmo 72B 在基准测试中超越竞争对手**](https://x.com/osanseviero/status/1838939324651299235?s=46)：来自 **Allen Institute for AI** 的 **Molmo 72B** 模型在 **AI2D** 和 **ChatQA** 等基准测试中优于 **Llama 3.2 V 90B** 等模型，以 **Apache 许可证** 提供 **SOTA (州级) 性能**。
- [**Hermes 3 在 HuggingChat 上增强了指令遵循能力**](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B)：可在 **HuggingChat** 上使用的 **Hermes 3** 展示了改进的 **指令遵循 (instruction adherence)** 能力，与之前的版本相比，提供了更 **准确且符合上下文** 的回答。

**主题 2. 模型性能、量化与优化**

- [**MaskBit 与 MonoFormer 在图像生成领域的创新**](https://arxiv.org/abs/2409.16211)：**MaskBit** 模型在不使用 embeddings 的情况下，在 ImageNet **256 × 256** 上实现了 **1.52 的 FID**，而 **MonoFormer** 统一了 autoregressive 文本和基于 diffusion 的图像生成，通过利用类似的训练方法达到了 **state-of-the-art performance**。
- [**量化技术提升模型效率**](https://github.com/pytorch/torchtune/pull/930/files)：关于 **quantization vs distillation** 的讨论揭示了每种方法的互补优势，在 **Setfit** 和 **TorchAO** 中的实现解决了 **Llama 3.2** 等模型的内存和计算优化问题。
- [**提升性能的 GPU 优化策略**](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)：成员们探索了 **TF32** 和 **float8** 表示法以加速矩阵运算，并利用 **Torch Profiler** 和 **Compute Sanitizer** 等工具来识别和解决性能瓶颈。

**主题 3. API 定价、集成与部署挑战**

- [**为开发者澄清 Cohere API 定价**](https://discord.com/channels/954421988141711382/1168578329423642786/1288429856132038668)：开发者了解到，虽然 **rate-limited Trial-Keys** 是免费的，但转向 **Production-Keys** 会产生商业应用费用，强调了 API 使用需与项目预算保持一致。
- [**OpenAI 的 API 与数据访问审查**](https://x.com/morqon/status/1838891125492355280?s=46)：**OpenAI** 宣布限度开放用于审查目的的训练数据访问，托管在 **secured server** 上，这引发了工程界对 **transparency** 和 **licensing compliance** 的关注。
- [**集成多个工具与平台**](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)：讨论了将 **SillyTavern, Forge, Langtrace** 和 **Zapier** 与各种 API 集成的挑战，突显了维护无缝 **deployment pipelines** 和 **compatibility across tools** 的复杂性。

**主题 4. AI 安全、审查与许可问题**

- [**关于模型审查与去审查技术的辩论**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：社区成员讨论了 **Phi-3.5** 等模型的 **over-censorship** 问题，并致力于通过工具和在 **Hugging Face** 等平台上分享 **uncensored versions** 来实现模型的 **uncensor**。
- [**MetaAI 在欧盟的许可限制**](https://github.com/pytorch/torchtune/issues/1675)：**MetaAI** 在 **EU** 面临 **licensing challenges**，限制了对 **Llama 3.2** 等 **multimodal models** 的访问，并引发了关于遵守 **regional laws** 的讨论。
- [**OpenAI 的公司转型与团队离职潮**](https://x.com/miramurati/status/1839025700009030027)：**Mira Murati** 及其他核心团队成员从 **OpenAI** 辞职，引发了对 **organizational stability**、**corporate culture changes** 以及对 **AI model development** 和 **safety protocols** 潜在影响的猜测。

**主题 5. 硬件基础设施与 AI 的 GPU 优化**

- [**使用 Lambda Labs 获得高性价比的 GPU 访问**](https://www.diffchecker.com/O4ijl7QY/)：成员们讨论了以约 **$2/hour** 的价格利用 **Lambda Labs** 进行 GPU 访问，强调了其在运行 **benchmarks** 和 **fine-tuning models** 方面的灵活性，且无需高昂的前期成本。
- [**排除 Run Pod 上的 CUDA 错误**](https://x.com/Seshubon/status/1838527532972359882)：用户在 **Run Pod** 等平台上遇到 **illegal CUDA memory access errors**，解决方案包括 **switching machines**、**updating drivers** 以及修改 **CUDA code** 以防止内存溢出。
- [**在边缘设备上部署多模态模型**](https://github.com/Nutlope/napkins/blob/f6c89c76b07b234c7ec690195df278db355f18fc/app/api/generateCode/route.ts)：讨论了将 **Llama 3.2** 模型集成到 **GroqCloud** 等 **edge platforms** 中，强调了 **optimized inference kernels** 和 **minimal latency** 对于实时 AI 应用的重要性。

---

# 第 1 部分：Discord 高层摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.2 发布并带来新特性**: [Llama 3.2](https://x.com/danielhanchen/status/1838987356810199153) 已经发布，推出了新的文本模型（1B 和 3B）以及视觉模型（11B 和 90B），支持 **128K Context length** 并处理了 **9 trillion Tokens**。
   - 该版本带来了对 **GGUF** 和 **BNB** 等 **Quantization** 格式的支持，增强了其在各种场景中的应用。
- **模型使用成本效益对比**: 讨论集中在小型模型是否能在保证质量的同时节省成本，一位成员提到他们尽管有支出，但创建了一个价值 **$15-20k** 的数据集。
   - 矛盾的观点引发了关于 **GPU 成本** 是否最终比订阅 **APIs** 更经济的辩论，特别是在 **Token** 消耗巨大的情况下。
- **Llama 模型 Fine-tuning 咨询**: 成员们对 [在本地 Fine-tuning Llama 3.1](https://github.com/unslothai/unsloth/issues/418) 感兴趣，并推荐了为此过程量身定制的 Unsloth 工具和脚本。
   - 对 **Llama Vision** 模型支持的期待日益增高，预示着未来增强功能的路线图。
- **OpenAI 的反馈流程受到审查**: 参与者讨论了 OpenAI 通过 **Reinforcement Learning from Human Feedback (RLHF)** 进行改进的方法，寻求实现细节的澄清。
   - 对话强调了其反馈机制的模糊性，指出了流程透明度的必要性。
- **高 Token 使用量引发关注**: 据报道，密集的 AI 流水线每次生成平均消耗 **10-15M Tokens**，这强调了资深开发者所理解的复杂性。
   - 一位成员对同行对其硬件设置的误解表示沮丧。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 模型发布提供更好支持**: Hugging Face 最近的公告包括 **Mistral Small (22B)** 和 [Qwen models](https://qwenlm.github.io/blog/qwen2.5/) 的更新（已开放探索），以及用于 ML 应用开发的 **Gradio 5** 新特性。
   - **FinePersonas-v0.1** 的发布引入了 2100 万个用于合成数据生成的 Personas，同时 Hugging Face 与 [Google Cloud's Vertex AI](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_exciting-update-for-ai-developers-the-hugging-activity-7242235533236609025-w2FA?utm_source=share&utm_medium=member_desktop) 的深度集成增强了 AI 开发者的可访问性。
- **Llama 3.2 提供 Multimodal 能力**: 新发布的 **Llama 3.2** 拥有 **Multimodal** 支持，模型能够处理文本和图像数据，并包含高达 **128k Token Context length**。
   - 这些模型专为移动端和边缘设备部署而设计，促进了多样化的应用场景，有可能彻底改变本地 **Inferencing** 性能。
- **训练主题聚类面临的挑战**: 成员们在聚合合理数量的主题进行训练时遇到了困难，且不希望进行过多的手动合并，因此将重点转向 **Zero-shot** 系统作为解决方案。
   - 讨论围绕使用灵活的主题管理技术来简化生产流程展开。
- **对 Diffusion Models 的见解**: 使用 **Google Colab** 运行 **Diffusion Models** 的有效性引发了讨论，特别是关于在使用免费层级时的模型性能标准。
   - 成员们讨论了 **Flux** 作为一个强大的开源 **Diffusion Model**，并提出了 **SDXL Lightning** 等替代方案，以便在不牺牲太多质量的情况下实现更快的图像生成。
- **探索 Fine-tuning 和优化技术**: **Fine-tuning Token embeddings** 的技术和其他优化是核心话题，重点是在集成新添加的 **Embeddings** 时保持预先存在的 **Token** 功能。
   - 还讨论了由于内存限制导致的 **Setfit 序列化** 问题，强调了在训练阶段进行更好 **Checkpoint** 管理的策略。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 在 HuggingChat 上线**：最新发布的 **Hermes 3** **8B** 版本现已在 [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B) 可用，展示了改进的指令遵循能力。
   - **Hermes 3** 显著增强了其遵循指令的能力，承诺比之前的版本提供更准确且更具上下文相关性的响应。
- **Llama 3.2 性能见解**：包含多个尺寸的 **Llama 3.2** 发布引发了关于其性能的讨论，特别是与 Llama 1B 和 3B 等较小模型相比。
   - 用户注意到了特定的能力和局限性，包括改进的代码生成能力，引发了广泛的好奇心。
- **讨论样本打包技术**：一场关于训练小型 **GPT-2** 模型的 *sample packing* 讨论引发了对如果执行不当可能导致性能下降的担忧。
   - 一位参与者强调，尽管在理论上有好处，但简单的实现可能会导致次优结果。
- **MIMO 框架彻底改变视频合成**：[MIMO 框架](https://huggingface.co/papers/2409.16160) 提出了一种基于简单用户输入合成具有可控属性的逼真人物视频的方法。
   - **MIMO** 旨在克服现有 3D 方法的局限性，并增强视频合成任务的可扩展性和交互性。
- **寻求职位推荐系统研究**：一位成员详细说明了他们在寻找与构建 **resume ATS** 生成器和职位推荐系统相关的优质研究时面临的挑战。
   - *寻求建议*以有效地在广泛的现有文献中导航。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Llama 3.2 发布**：Meta 在 **Meta Connect** 期间宣布发布 **Llama 3.2**，其特点是包含中小型视觉 LLM 以及适用于边缘和移动设备的轻量级模型。
   - 正如在最新模型进展背景下所讨论的，这些模型旨在提高资源有限的开发者的可访问性。
- **Aider 面临功能挑战**：用户报告了 **Aider** 的局限性，特别是缺乏内置翻译和文档索引不足，推动了对潜在增强功能的讨论。
   - 想法包括加入语音反馈和自动文档搜索，以改善用户体验。
- **切换 LLM 以获得更好性能**：报告显示，用户正在从 **Claude Sonnet 3.5** 切换到 **Gemini Pro 1.5** 等模型，以提高代码理解和性能。
   - 使用 Aider 的基准测试套件进行模型性能跟踪被认为是确保准确结果的关键。
- **本地向量数据库探索**：一场围绕本地向量数据库的讨论显示了对 **Chroma**、**Qdrant** 和 **PostgreSQL** 向量扩展以高效处理复杂数据的兴趣。
   - 虽然 SQLite 可以管理向量数据库任务，但专门的数据库被认为更适合处理沉重的负载。
- **介绍 par_scrape 工具**：一位成员在 GitHub 上展示了 **par_scrape** 工具，作为一种高效的网络爬虫解决方案，因其与替代方案相比的能力而受到称赞。
   - 它的利用可以显著简化社区的爬虫任务。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 数据库升级计划**：**数据库升级**定于**东部时间周五上午 10 点**进行，届时将有 **5-10 分钟**的简短停机。用户应为潜在的服务中断做好准备。
   - 此次升级旨在提高整体系统性能，与最近的 API 更改保持一致。
- **API 输出增强功能发布**：OpenRouter 现在在 **completion response** 中包含 **provider**，以提高数据检索的清晰度。
   - 这一更改旨在简化信息处理并增强用户体验。
- **Gemini 模型路由升级**：**Gemini-1.5-flash** 和 **Gemini-1.5-pro** 已重新路由以使用最新的 **002 version**，从而获得更好的性能。
   - 鼓励社区测试这些更新的模型，以衡量它们在各种应用中的效率。
- **Llama 3.2 发布引发期待**：即将发布的 **Llama 3.2** 包含较小的模型，以便更容易地集成到移动和边缘部署中。
   - 关于 **OpenRouter** 是否会托管新模型的咨询引发了开发者的兴奋。
- **本地服务器支持面临限制**：由于**受限的外部访问**阻碍了协助能力，对本地服务器的支持仍然是一个挑战。
   - 如果端点满足特定的 **OpenAI-style schema** 要求，未来的 API 支持可能会扩展，从而为合作打开大门。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **高级语音模式 (Advanced Voice Mode) 分发引发的不满**：成员们对 **Advanced Voice mode** 的**有限推送**感到沮丧，特别是在欧盟地区，尽管官方宣布已全面开放，但访问权限仍然受限。
   - 他们指出欧盟用户面临**功能延迟**已成趋势，并提到了之前的记忆功能 (memory functionality) 等先例。
- **Meta AI 在欧盟面临许可限制**：会议澄清，由于多模态模型 (multimodal models) 受到严格的许可规则限制，**Meta AI** 目前**无法**供欧盟和英国用户使用，这直接与 **Llama 3.2** 的许可问题挂钩。
   - 成员们注意到 **Llama 3.2** 虽然提升了多模态能力，但仍受困于这些复杂的许可问题。
- **论文评分需要更严格的反馈**：讨论集中在如何对论文提供**诚实的反馈**，强调了模型往往倾向于给出过于温和的评价。
   - 成员们建议使用详细的评分标准 (rubrics) 和示例，但也指出模型固有的正向强化倾向使这一问题变得复杂。
- **优化 Minecraft API 提示词 (Prompts)**：成员们提出了增强 **Minecraft API** 提示词的策略，旨在通过改变主题和复杂度来减少重复查询。
   - 针对如何引导 AI 执行结构化的响应格式并避免重复提问，大家表达了关注。
- **处理复杂任务时的挣扎**：用户表达了对 **GPT** 处理复杂任务能力的沮丧，提到在处理写书请求时，往往需要长时间等待却只能得到极少的产出。
   - 一些人建议使用 **Claude** 和 **o1-preview** 等替代模型，认为得益于更长的记忆窗口 (memory windows)，这些模型的能力更强。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 发布引发热潮**：近期发布的 **Llama 3.2**（包括 **1B** 和 **3B** 模型）因其在各种硬件配置上的表现而引起了广泛关注。
   - 用户特别渴望获得对 **11B** 多模态模型的支持，但由于视觉集成方面的复杂性，其可用性可能会推迟。
- **与 SillyTavern 的集成问题**：用户在使用 LM Studio 时遇到了 **SillyTavern 的集成问题**，主要涉及服务器通信和响应生成。
   - 故障排除建议指出，**任务输入**可能需要更加具体，而不是依赖自由格式的文本提示词。
- **对多模态模型能力的关注**：虽然 **Llama 3.2** 包含视觉模型，但用户要求拥有类似 **GPT-4** 的真正多模态能力，以实现更广泛的用途。
   - 会议澄清 **11B 模型仅限于视觉任务**，目前缺乏语音或视频功能。
- **价格差异引起不满**：用户分享了对**欧盟地区科技产品价格更高**的沮丧，其价格有时可能是美国的**两倍**。
   - 许多人强调 **VAT**（增值税）是导致这些差异的重要因素。
- **对 RTX 3090 TPS 的预期**：关于 RTX 3090 的讨论强调，在 Q4 **8B 模型**上预期的 **每秒事务数 (TPS)** 约为 **60-70 TPS**。
   - 澄清了该指标主要用于**推理训练 (inference training)**，而非简单的查询处理。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **yt-dlp 成为必备工具**：一名成员重点推荐了 [yt-dlp](https://github.com/yt-dlp/yt-dlp)，展示了它作为一个强大的音频/视频下载器的功能，虽然引发了对恶意软件的担忧，但确认了来源的安全性。
   - 该工具可以简化开发者的内容下载流程，但由于潜在的安全风险，评估其在工程师中的使用情况至关重要。
- **PyTorch 训练属性 Bug 引发挫败感**：讨论了 PyTorch 中的一个已知 Bug，即执行 `.eval()` 或 `.train()` 时无法更新 `torch.compile()` 模块的 `.training` 属性，详见 [此 GitHub issue](https://github.com/pytorch/pytorch/issues/132986)。
   - 成员们对该问题缺乏透明度表示失望，同时集思广益探讨了变通方案，例如修改 `mod.compile()`。
- **需要本地 LLM 基准测试工具**：针对本地 LLM 测试的开源基准测试套件的推荐请求指向了 MMLU 和 GSM8K 等成熟指标，并提到了用于评估模型的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
   - 这一需求强调了 AI 社区需要全面的评估框架来验证本地模型的性能。
- **关于众包数据的 NAIV3 技术报告**：发布的 [NAIV3 技术报告](https://arxiv.org/abs/2409.15997) 包含一个拥有 **600 万** 张众包图像的数据集，重点关注打标签实践和图像管理。
   - 讨论围绕在文档中加入幽默感展开，表明了对技术报告风格偏好的分歧。
- **BERT 掩码率显示对性能的影响**：对 BERT 模型高掩码率的调查显示，掩码率超过 **15%**（特别是高达 **40%**）可以提升性能，这表明在大型模型中具有显著优势。
   - 这意味着可能需要重新评估训练方法，以整合近期关于掩码策略研究的发现。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 在上下文保留方面表现不佳**：用户对 **Perplexity AI** 无法保留后续问题的上下文表示沮丧，这一趋势近期有所恶化。一些成员注意到平台性能下降，影响了其效用。
   - 针对可能影响 Perplexity 能力的潜在财务问题提出了担忧，引发了关于可行替代方案的讨论。
- **Merlin.ai 提供带联网功能的 O1**：**Merlin.ai** 被推荐作为 Perplexity 的替代方案，因为它提供带联网功能的 **O1 能力**，允许用户绕过每日消息限制。参与者表现出探索 Merlin 扩展功能的兴趣。
   - 讨论强调了用户认为 Merlin 比 Perplexity 更具功能性，这可能会重塑他们的工具选择。
- **Wolfram Alpha 与 Perplexity API 的集成**：一位用户询问是否可以像在 Web 端那样在 **Perplexity API** 中使用 **Wolfram Alpha**，得到的确认是目前**无法实现**。强调了 API 与 Web 界面的独立性。
   - 进一步询问了 **API** 在解决数学和科学问题方面是否能像 Web 界面一样高效，但未得到确切答案。
- **用户评价用于教育的 AI 工具**：许多用户分享了他们使用各种 **AI 工具** 完成学术任务的观点，在替代方案中，偏好在 **GPT-4o** 和 **Claude** 之间摇摆。反馈表明，不同的 AI 工具对学校相关需求提供的协助程度各不相同。
   - 这一交流凸显了 AI 在教育环境中的重要作用，并强调了用户体验如何塑造这些偏好。
- **评估空气炸锅：值得吗？**：一位用户分享了一个讨论 [空气炸锅是否值得购买](https://www.perplexity.ai/search/are-air-fryers-worth-it-5Ylk154lSZyKHan.UxR2UA) 的链接，重点关注其健康益处与传统油炸方法的对比以及烹饪效率。对话包含了消费者对该设备实用性的各种观点。
   - 讨论的核心结论集中在空气炸锅积极的烹饪属性，以及对其与传统方法相比实际健康益处的怀疑。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Anthropic 目标年营收达 10 亿美元**：据 [CNBC](https://x.com/tanayj/status/1838623375167574421?s=46) 报道，**Anthropic** 预计今年营收将突破 **10 亿美元**，实现令人惊叹的 **1000% 同比增长**。
   - 营收来源包括 **60-75%** 来自第三方 API，**15%** 来自 Claude 订阅，这标志着公司业务的重大转变。
- **OpenAI 提供训练数据访问权限**：在一次显著的转变中，OpenAI 宣布将允许访问其 [训练数据进行审查](https://x.com/morqon/status/1838891125492355280?s=46)，以应对涉及受版权保护作品的使用问题。
   - 这种访问权限仅限于 OpenAI 旧金山办公室的一台安全计算机，在社区中引发了不同的反应。
- **Molmo 模型超出预期**：**Molmo 模型** 引发了热烈讨论，有说法称其 **pointing feature** 可能比更高的 AIME 分数更具意义，并在与 **Llama 3.2 V 90B** 的基准测试对比中获得了积极评价。
   - 评论指出 Molmo 在 AI2D 和 ChatQA 等指标上表现出色，展示了其相对于竞争对手的强劲性能。
- **Curriculum Learning 提升 RL 效率**：研究表明，实施 **curriculum learning** 可以通过利用先前的演示数据来实现更好的探索，从而显著提升 **Reinforcement Learning (RL)** 的效率。
   - 该方法包括一种极具创意的 **reverse and forward curriculum** 策略，与 **DeepMind** 类似的 **Demostart** 相比，突显了机器人在收益和挑战方面的并存。
- **Llama 3.2 发布引发社区热议**：**Llama 3.2** 已正式发布，包含 **1B, 3B, 11B** 和 **90B** 等多种模型尺寸，旨在增强文本和多模态能力。
   - 最初的反应交织着兴奋与对其成熟度的怀疑，关于未来改进和更新的暗示进一步推动了讨论。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **探索将 SAM2-fast 与 Diffusion Transformers 结合**：成员们讨论了在 ***Diffusion Transformer Policy*** 中使用 **SAM2-fast**，将摄像头传感器数据映射到机械臂位置，并建议在此用例中使用 *图像/视频分割*。
   - 对话强调了通过先进的 ML 技术将快速传感器数据处理与机器人控制相结合的潜力。
- **Torch Profiler 的文件大小问题**：***Torch profiler*** 生成的文件体积过大（高达 **7GB**），引发了关于仅对必要项进行分析并导出为 **.json.gz** 以进行压缩的建议。
   - 成员们强调了高效的分析策略，以保持文件大小可控并确保性能追踪的可用性。
- **RoPE Cache 应始终保持 FP32**：关于 Torchao Llama 模型中 **RoPE cache** 的讨论指出，为了保证准确性，应始终采用 **FP32** 格式。
   - 成员们指出了 [模型代码库](https://github.com/pytorch/ao/blob/7dff17a0e6880cdbeed1a14f92846fac33717b75/torchao/_models/llama/model.py#L186-L192) 中的特定代码行以进一步澄清。
- **Lambda Labs 高性价比的 GPU 访问**：使用 **Lambda Labs** 获取 GPU 访问权限（价格约为 **$2/小时**）被强调为运行基准测试和微调的灵活选择。
   - 用户分享了关于无缝 SSH 访问和按需付费结构的体验，这使其对许多 ML 应用具有吸引力。
- **Metal Atomics 需要原子加载/存储**：为了实现工作组（workgroups）之间的消息传递，一位成员建议在 Metal Atomics 操作中使用 **atomic bytes** 数组。
   - 强调了结合原子操作和非原子加载的高效标志位（flag）使用，以改进数据处理。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Run Pod 问题令用户沮丧**：用户报告在 Run Pod 上遇到 **illegal CUDA errors**，一些人建议通过更换机器来解决此问题。
   - 一位用户幽默地建议不要使用 Run Pod，因为问题频发，并强调了其中的挫败感。
- **Molmo 72B 成为焦点**：由 Allen Institute for AI 开发的 **Molmo 72B** 拥有最先进的基准测试表现，并基于图像-文本对的 PixMo 数据集构建。
   - 该模型采用 **Apache licensed**，旨在与包括 GPT-4o 在内的领先多模态模型竞争。
- **OpenAI 领导层变动震惊社区**：OpenAI CTO 的辞职是一个引人注目的时刻，引发了对组织未来方向的猜测。
   - 成员们讨论了对 OpenAI 战略的潜在影响，暗示了有趣的内部动态。
- **Llama 3.2 发布令人兴奋**：**Llama 3.2** 的推出引入了适用于边缘设备的轻量级模型，引发了关于 1B 到 90B 不同规模的热议。
   - 多个来源确认了分阶段发布，人们对新模型的性能验证感到兴奋。
- **Meta 的欧盟合规困境**：对话揭示了 Meta 在欧盟法规方面的挣扎，导致欧洲用户的访问受限。
   - 讨论提到了可能影响模型可用性的许可证变更，引发了对公司动机的辩论。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API Key 定价说明**：成员们强调了用于免费使用的 **rate-limited Trial-Key**，但指出商业应用需要会产生费用的 **Production-Key**。
   - 这强调了在规划 API key 资源时需要仔细考虑预期用途。
- **测试递归迭代模型假设**：一位用户提出疑问，如果在多个 LLM 中获得相似的结果，是否意味着他们的 **Recursive Iterative model** 运行正常。
   - 建议包括针对基准测试进行进一步评估，以确保结果可靠。
- **新 RAG 课程发布**：宣布了与 **Weights&Biases** 合作制作的新 [RAG course](https://www.wandb.courses/courses/rag-in-production)，在 2 小时内涵盖评估和流水线。
   - 参与者可获得 **Cohere credits**，并可在课程期间向 Cohere 团队成员提问。
- **令人兴奋的智能望远镜项目**：一位成员分享了他们对 **智能望远镜支架** 项目的热情，该项目旨在自动定位 Messier catalog 中的 **110 个物体**。
   - 社区提供了支持，鼓励为该项目进行协作和资源共享。
- **Cohere Cookbook 现已上线**：**Cohere Cookbook** 被强调为包含有效使用 Cohere 生成式 AI 平台指南的资源。
   - 成员们被引导去探索针对其 AI 项目需求的特定章节，包括 embedding 和语义搜索。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 欺诈警报**：发布了关于欺诈网站 [llamaparse dot cloud](https://twitter.com/llama_index/status/1838699883756466512) 的警告，该网站试图冒充 LlamaIndex 产品；官方 **LlamaParse** 可通过 [cloud.llamaindex.ai](https://t.co/jM9ioNJuv3) 访问。
   - 对冒充合法服务并给用户带来风险的诈骗行为*保持警惕*。
- **AWS Gen AI Loft 的精彩演讲**：LlamaIndex 的开发者将在 2024 年 3 月 21 日的 AWS Gen AI Loft（与 ElasticON 会议同期举行）分享关于 RAG 和 Agent 的见解 ([来源](https://twitter.com/llama_index/status/1838714867697803526))。
   - 与会者将了解 Fiber AI 如何将 **Elasticsearch** 集成到高性能 B2B 拓客中。
- **Pixtral 12B 模型发布**：来自 @MistralAI 的 **Pixtral 12B 模型** 现已与 LlamaIndex 集成，在涉及图表和图像理解的多模态任务中表现出色 ([来源](https://twitter.com/llama_index/status/1838970087354798492))。
   - 该模型在与同类尺寸模型的对比中展示了令人印象深刻的性能。
- **加入 LlamaIndex 团队！**：LlamaIndex 正在为其旧金山团队积极招聘工程师；职位涵盖从全栈到专业角色的各种岗位 ([链接](https://twitter.com/llama_index/status/1839055997291344050))。
   - 团队寻求渴望从事 **ML/AI** 技术工作的热心人士。
- **关于 VectorStoreIndex 使用的澄清**：用户讨论了如何使用 `VectorStoreIndex` 正确访问底层向量存储，特别是通过 `index.vector_store`。针对 **SimpleVectorStore** 的局限性进行了澄清，并引发了关于替代存储方案的讨论。
   - 对话强调了可调用方法和属性的技术层面，有助于更好地理解 Python 装饰器。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 定价巩固竞争地位**：近期 **Gemini Pro** 的降价与其基于 **Elo** 分数的对数线性定价曲线一致，优化了其针对其他模型的竞争策略。
   - 随着价格调整，**OpenAI** 继续主导高端市场，而 **Gemini Pro** 和 **Flash** 则在类似“iPhone vs Android”的生动框架中占据了较低层级。
- **Anthropic 达到营收里程碑**：根据 CNBC 的报告，**Anthropic** 今年有望实现 **10 亿美元** 的营收，同比增长高达 **1000%**。
   - 营收细分显示其严重依赖**第三方 API** 销售（贡献了 **60-75%** 的收入），API 和聊天机器人订阅也发挥了关键作用。
- **Llama 3.2 模型增强边缘能力**：**Llama 3.2** 的发布引入了针对边缘设备优化的轻量级模型，配置包括 **1B, 3B, 11B** 和 **90B 视觉模型**。
   - 这些新产品强调了多模态能力，鼓励开发者通过开源访问探索增强的功能。
- **Mira Murati 告别 OpenAI**：在社区分享的告别信中，Mira Murati 从 **OpenAI** 离职引发了对其任职期间重大贡献的回顾讨论。
   - Sam Altman 认可了她所经历的情感历程，强调了她在面临挑战时为团队提供的支持。
- **Meta 的 Orion 眼镜原型首次亮相**：经过近十年的开发，**Meta** 揭晓了其 **Orion** AR 眼镜原型，尽管最初存在质疑，但这标志着重大进步。
   - 该眼镜旨在通过内部使用来优化用户体验，具有宽广的视野和轻量化的特性，为最终的消费者发布做准备。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Basicsr 安装困扰简化处理**：为了解决 **Forge** 中 **ComfyUI** 的问题，用户应在激活虚拟环境后，进入 Forge 文件夹并运行 `pip install basicsr`。
   - 关于安装过程存在越来越多的困惑，一些用户希望扩展在安装后能以标签页的形式出现。
- **界面之争：ComfyUI vs Forge**：成员们分享了他们的偏好，其中一位表示，与 **ComfyUI** 相比，他们发现 **Invoke** 使用起来要容易得多。
   - 许多人选择继续忠于 **ComfyUI**，原因是 **Forge** 内部的版本过旧且存在兼容性问题。
- **3D 模型生成器：哪些好用？**：对 **3D 模型生成器** 的咨询揭示了 **TripoSR** 的问题，暗示许多开源工具似乎已失效。
   - 尽管对 **Luma Genie** 和 **Hyperhuman** 的功能仍持高度怀疑态度，但人们对其表现出了兴趣。
- **在没有 GPU 的情况下运行 Stable Diffusion**：对于那些希望在没有 GPU 的情况下运行 **Stable Diffusion** 的用户，使用 **Google Colab** 或 **Kaggle** 可以提供免费的 GPU 资源访问。
   - 大家一致认为，这些平台是初学者接触 Stable Diffusion 的绝佳起点。
- **玩转 ControlNet OpenPose**：成员们学习了如何使用 **ControlNet OpenPose** 预处理器在平台内生成和编辑预览图像。
   - 探索这一功能显然令人兴奋，它允许对生成的输出进行**详细调整**。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.2 发布，支持多模态功能**：**Llama 3.2** 的发布引入了支持长上下文的 **1B** 和 **3B 文本模型**，允许用户在长上下文数据集上尝试 `enable_activation_offloading=True`。
   - 此外，**11B 多模态模型**支持 **The Cauldron 数据集**和自定义多模态数据集，以增强生成能力。
- **对绿卡的渴望**：一位成员幽默地表达了对**绿卡**的极度渴望，暗示由于目前的处境，他们可能会让欧洲的生活变得艰难。
   - *“为了换取绿卡，我不会告诉任何人”* 突显了他们的沮丧和谈判意愿。
- **为 FP32 用户考虑 TF32**：围绕为仍在使用 **FP32** 的用户启用 **TF32** 选项展开了讨论，因为它可以加速矩阵乘法 (matmul)。
   - 观点认为，如果已经在使用 **FP16/BF16**，TF32 可能不会带来额外好处，一位成员幽默地指出：*“我想知道谁会放着 FP16/BF16 不用而更倾向于它”*。
- **关于 KV-cache 切换的 RFC 提案**：[一项关于 KV-cache 切换的 RFC](https://github.com/pytorch/torchtune/issues/1675) 已被提出，旨在改进模型前向传播 (forward passes) 期间缓存的处理方式。
   - 该提案解决了目前缓存总是被不必要更新的限制，引发了关于必要性和可用性的进一步讨论。
- **关于处理 Tensor 尺寸的建议**：有人询问除了使用 **Tensor item()** 方法之外，如何改进对 Tensor 尺寸的处理。
   - 一位成员承认需要更好的解决方案，并承诺会进一步思考。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MOToMGP 错误调查**：一位用户询问关于“failed to run the MOToMGP pass manager”的错误，寻求复现案例以改进相关的错误提示信息。
   - 社区成员被鼓励分享与此特定问题相关的见解或经验。
- **为 MAX 优化调整 Linux 机器配置**：一位成员询问在运行带有 **ollama3.1** 的 **MAX** 时如何调整 Linux 机器的配置，开启了关于最佳配置的讨论。
   - 成员们贡献了关于资源分配的技巧以增强性能。
- **GitHub Discussions 转移**：由于参与度较低，**Mojo GitHub Discussions** 将于 **9 月 26 日**禁用新评论，以将社区互动集中在 Discord 上。
   - 此举旨在简化讨论流程，并反思了将过去的讨论转换为 Issue 的低效性。
- **Mojo 与 C 的通信速度**：参与者想知道 Mojo 与 C 的通信是否比与 Python 更快，并指出这可能取决于具体的实现。
   - 大家一致认为，Python 与 C 的交互会根据上下文而有所不同。
- **Evan 实现关联别名 (Associated Aliases)**：Evan 正在 Mojo 中推出**关联别名**，允许使用类似于所提供示例的 traits 和类型别名 (type aliases)，这令社区感到兴奋。
   - 成员们认为该功能有望改进代码的组织结构和清晰度。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **对 o1 Preview 和 Mini API 的兴奋感**：成员们对接入 **o1 Preview 和 Mini API** 表示兴奋，并思考其在 **Lite LLM** 中的能力以及通过 **Open Interpreter** 获得的响应。
   - 一位成员幽默地提到，尽管缺乏 **tier 5** 访问权限，他们仍渴望对其进行测试。
- **Llama 3.2 发布轻量级边缘模型**：Meta 的 **Llama 3.2** 已经发布，推出了用于边缘侧的 **1B & 3B 模型**，并针对 **Arm**、**MediaTek** 和 **Qualcomm** 进行了优化。
   - 开发者可以通过 Meta 和 [Hugging Face](https://go.fb.me/w63yfd) 获取这些模型，其中 **11B & 90B vision models** 旨在与闭源模型竞争。
- **Tool Use 剧集涵盖开源 AI**：最新的 [Tool Use 剧集](https://www.youtube.com/watch?v=-To_ZIynjIk) 讨论了开源编程工具以及围绕 **AI** 的基础设施。
   - 该剧集聚焦于社区驱动的创新，与频道内之前分享的想法产生了共鸣。
- **Llama 3.2 现已在 GroqCloud 上线**：Groq 宣布在 **GroqCloud** 中提供 **Llama 3.2** 预览版，通过其基础设施增强了开发者的可访问性。
   - 成员们注意到了对 Groq 速度的积极反馈，并评论道任何与 Groq 相关的东西运行速度都极快。
- **Logo 设计选择引发讨论**：一位成员分享了他们的 Logo 设计历程，指出虽然他们考虑过 GitHub 的 Logo，但觉得目前的选择更胜一筹。
   - 另一位成员轻松地对他们设计选择的**力量**发表了评论，为讨论增添了幽默感。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Gemini 1.5 展示了强劲的基准测试结果**：**Gemini 1.5 Flash** 在 2024 年 9 月达到了 **67.3%** 的分数，而 **Gemini 1.5 Pro** 的表现更佳，达到 **85.4%**，标志着性能的重大提升。
   - 这一进步突显了模型在各种数据集上能力的持续增强。
- **MMLU-Pro 数据集发布**：新的 **MMLU-Pro** 数据集包含来自 **57 个学科** 的问题，难度有所增加，旨在有效地挑战模型评估。
   - 这个更新的数据集对于评估模型在 **STEM** 和人文科学等复杂领域的表现至关重要。
- **质疑 Chain of Thought (CoT) 的实用性**：最近一项包含 **300 多次实验** 的研究表明，**Chain of Thought (CoT)** 仅对数学和符号推理有益，在大多数任务中的表现与直接回答相似。
   - 分析表明，对于 **95%** 的 MMLU 任务，CoT 是不必要的，应将重点重新转向其在**符号计算**方面的优势。
- **AutoGen 在研究中证明了其价值**：研究强调了 **AutoGen** 的使用日益增长，反映了其在当前 AI 领域的相关性。
   - 这一趋势指向了自动化模型生成的重大发展，影响了性能和研究进展。
- **Quiz 3 详情已公布**：关于 **Quiz 3** 的询问引导成员确认其可在**课程网站**的教学大纲（syllabus）部分找到。
   - 强调定期检查教学大纲更新，以便及时了解评估安排。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 发布酷炫新功能！**：本周 [Langtrace](https://x.com/karthikkalyan90/status/1838809430009299240?s=46&t=XrJJzmievg67l3JcMEEDEw) 上推出了针对 **DSPy** 的新功能，包括受 **MLFlow** 启发的新项目类型和自动实验追踪。
   - 这些功能包括自动 Checkpoint 状态追踪、**评估分数趋势线 (eval score trendlines)** 以及对 **litellm** 的支持。
- **文本分类瞄准欺诈检测**：用户正在使用 **DSPy** 将文本分类为**三种欺诈类型**，并寻求关于最佳 Claude 模型的建议。
   - 讨论指出 **Sonnet 3.5** 是领先模型，而 **Haiku** 提供了一个高性价比的替代方案。
- **DSPy 作为用户查询的编排器**：一名成员正在探索将 **DSPy** 作为将用户查询路由到子 Agent 的工具，并评估其直接交互能力。
   - 对话涵盖了集成工具的潜力，并质疑了 Memory 与**独立对话历史 (standalone conversation history)** 相比的有效性。
- **澄清文本分类中的复杂类别**：成员们讨论了在将文本分类为复杂类别（特别是包括**美国政治 (US politics)** 和**国际政治 (International Politics)**）时，需要精确的定义。
   - 一位成员指出，这些定义在很大程度上取决于业务上下文，强调了需要细致处理的方法。
- **分类任务协作教程**：正在进行的讨论恰逢一位成员正在编写关于分类任务的教程，旨在提高清晰度。
   - 这标志着在提高分类领域理解方面所做的努力。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 贡献的基本资源**：一位成员分享了一系列关于 [tinygrad 的教程](https://mesozoic-egg.github.io/tinygrad-notes/)，涵盖了内部原理，以帮助新贡献者掌握该框架。
   - 他们强调 [快速入门指南 (quickstart guide)](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) 和 [抽象指南 (abstraction guide)](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py) 是入门的首选资源。
- **Tinygrad 训练循环太慢**：一位用户在开发字符模型时抱怨 tinygrad 0.9.2 版本中的**训练速度缓慢**，形容其“慢得离谱 (slow as balls)”。
   - 他们租用了一块 **4090 GPU** 来提升性能，但报告称改进微乎其微。
- **采样代码中的 Bug 影响输出质量**：该用户在最初将训练缓慢归咎于通用性能问题后，发现其**采样代码 (sampling code)** 中存在 **Bug**。
   - 他们澄清问题专门源于采样实现，而非训练代码，这影响了模型推理 (Inference) 的质量。
- **通过代码高效学习**：成员们建议通过阅读代码并让产生的问题引导在 tinygrad 中的学习。
   - 使用 ChatGPT 等工具可以辅助排查问题并促进高效的反馈循环。
- **使用 DEBUG 了解 Tinygrad 的流程**：一位成员建议在执行简单操作时使用 `DEBUG=4`，以查看生成的代码并理解 tinygrad 中的流程。
   - 这种技术提供了对框架内部运作机制的实用见解。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **寻求开源聊天 UI**：一名成员正在寻找专门为**编程任务**定制的**开源 UI** 聊天界面，并寻求社区对可用选项的见解。
   - 讨论中欢迎分享部署类似系统的经验，以帮助优化选择。
- **点赞/点踩反馈机制**：成员们正在探索为聊天机器人实现**点赞/点踩评价选项**，其中一人分享了他们排除 **Streamlit** 的自定义前端方案。
   - 这反映了通过反馈系统增强用户参与度的共同兴趣。
- **Azure Chat OpenAI 集成细节**：一位开发者透露了他们集成 **Azure Chat OpenAI** 以实现聊天机器人功能的细节，并强调其是类似项目的可行平台。
   - 他们鼓励其他人就此集成的想法和挑战进行交流。
- **构建 Agentic RAG 应用的经验**：一位用户详细介绍了他们使用 **LangGraph**、**Ollama** 和 **Streamlit** 开发 **agentic RAG** 应用的过程，旨在检索相关的研究数据。
   - 他们通过 [Lightning Studios](https://lightning.ai/maxidiazbattan/studios/langgraph-agenticrag-with-streamlit) 成功部署了解决方案，并在 [LinkedIn 帖子](https://www.linkedin.com/posts/maxidiazbattan_last-weekend-i-decided-to-put-the-tool-calling-activity-7244692826754629632-Um7w?utm_source=share&utm_medium=member_ios)中分享了过程心得。
- **使用 Lightning Studios 进行实验**：开发者利用 **Lightning Studios** 进行高效的应用部署，并对其 **Streamlit** 应用进行实验，优化了技术栈。
   - 这强调了该平台在增强不同工具应用性能方面的能力。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GANs、CNNs 和 ViTs 作为顶级图像算法**：成员们注意到 **GANs**、**CNNs** 和 **ViTs** 经常在**图像任务**的顶级算法中交替领先，并请求一个视觉**时间线**来展示这一演变过程。
   - 对时间线的兴趣凸显了在图像处理算法领域对历史背景的需求。
- **MaskBit 彻底改变图像生成**：关于 [MaskBit](https://arxiv.org/abs/2409.16211) 的论文介绍了一种无嵌入（embedding-free）模型，该模型通过 bit tokens 生成图像，在 ImageNet **256 × 256** 上达到了 **1.52** 的 SOTA FID。
   - 这项工作还增强了对 **VQGANs** 的理解，创建了一个提高可访问性并揭示新细节的模型。
- **MonoFormer 融合了自回归和扩散**：[MonoFormer 论文](https://arxiv.org/abs/2409.16280)提出了一种统一的 Transformer，用于自回归文本和基于扩散的图像生成，性能达到了 SOTA 水平。
   - 这是通过利用训练相似性实现的，主要区别在于所使用的 attention masks。
- **滑动窗口注意力（Sliding window attention）仍依赖位置编码**：成员们讨论到，虽然**滑动窗口注意力**带来了优势，但它仍然依赖于**位置编码**机制。
   - 这一讨论强调了在模型效率和保留位置感知之间持续的平衡。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1288218181596614729)** (510 条消息 🔥🔥🔥): 

> - `Llama 3.2 发布`
> - `微调模型`
> - `模型性能比较`
> - `视觉模型`
> - `模型推理与兼容性`

- **Llama 3.2 发布**：Llama 3.2 已经发布，包含新的文本模型（1B, 3B）和视觉模型（11B, 90B），具备 128K 上下文长度和 9 万亿 tokens 等特性。
   - 它支持多种量化格式，如 GGUF 和 BNB。
- **大模型微调指南**：建议初学者从较小的模型开始，以熟悉微调流程，然后再转向更大的模型，如 Llama 70B 模型。
   - 对于较小的模型，建议使用 Colab，而较大的模型则需要大量的硬件资源。
- **模型性能与比较**：讨论了在不同数据集上训练的模型的性能，并指出数据质量对模型效能的影响。
   - 对 Llama 模型与其他模型进行了比较，强调了数据质量比单纯的数据量更重要。
- **视觉模型支持**：针对 Llama 3.2 的视觉支持提出了疑问，并对模型的能力和适配进行了说明。
   - Unsloth 被提及为未来支持视觉模型的潜在解决方案。
- **在不同系统上运行 Llama**：用户讨论了在各种硬件配置上运行 Llama 及其变体的经验，特别关注 Windows 和 ROCm 版本的兼容性问题。
   - 对于尝试运行这些模型的 AMD GPU 用户，建议参考 Llama.cpp。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jsonlint.com/">JSON Online Validator and Formatter - JSON Lint</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1839009095883567520">来自 Daniel Han (@danielhanchen) 的推文</a>: 我对 Llama 3.2 的分析：1. 新的 1B 和 3B 纯文本 LLM，9 万亿 token 2. 新的 11B 和 90B 视觉多模态模型 3. 128K 上下文长度 4. 1B 和 3B 使用了来自 8B 和 70B 的一些蒸馏 5. VL...</li><li><a href="https://huggingface.co/papers/2308.05884">Paper 页面 - PIPPA: A Partially Synthetic Conversational Dataset</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF">unsloth/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/mattkdouglas/status/1838403695605690444">来自 Matthew Douglas (@mattkdouglas) 的推文</a>: 宣布 bitsandbytes 0.44.0 发布！我们实现了由 @Apple 研究员 @MatPagliardini, @GrangierDavid, 和 @PierreAblin 提出的 AdEMAMix 优化器的 8-bit 版本。</li><li><a href="https://x.com/danielhanchen/status/1838994357728506121">来自 Daniel Han (@danielhanchen) 的推文</a>: Llama 3.2 小型模型 1B 和 3B 纯文本 LLM 基准测试 - 或许也是独立的 LLM？1B MMLU 49.3 3B MMLU 63.4</li><li><a href="https://www.reddit.com/r/Loc">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1838991771948425652">来自 Daniel Han (@danielhanchen) 的推文</a>: Llama 3.2 多模态基准测试：11B 的 MMMU 为 60.3，对比 Claude Haiku 的 50.2；90B 的 MMMU 为 60.3，对比 GPT 4o mini 的 59.4。90B 看起来非常强大！</li><li><a href="https://x.com/danielhanchen/status/1838987356810199153">来自 Daniel Han (@danielhanchen) 的推文</a>: Llama 3.2 多模态发布了！模型尺寸涵盖 1B, 3B 到 11B 和 90B！</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-bnb-4bit">unsloth/Llama-3.2-3B-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit">unsloth/Llama-3.2-1B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/fullstackwebdev/81e64e8faca496e5390d09a4756d8db4">llama32_3b_failwhale.py</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpeb5g/llama_32_versions_gguf_4bit_bnb_more/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf">Llama 3.2 - meta-llama 集合</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_ggufs_4bit_bitsandbytes/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct">meta-llama/Llama-3.2-90B-Vision-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/UWF6dxQYcbU?feature=shared">仅需 5 步，使用 Unsloth + Ollama 免费微调 AI 模型！</a>: 你准备好训练自己的大语言模型 (LLM) 了吗，但觉得太复杂？再想想！在这段视频中，我将向你展示任何人都可以如何...</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2-5 倍的速度、减少 80% 的显存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM</a>: 以 2-5 倍的速度、减少 80% 的显存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/meta-llama/llama-models/pull/150/files#diff-245e85fd8ab6d944f46d6d3d30b45e8f9fca75ec047781ef27d5c48129044c55">ashwinb 提交的对 Llama 3.2 系列模型的支持 · Pull Request #150 · meta-llama/llama-models</a>: 未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1288423494404804620)** (42 messages🔥): 

> - `Model Costs Comparison` (模型成本对比)
> - `API vs Local GPU Usage` (API 与本地 GPU 使用对比)
> - `OpenAI's Corporate Transition` (OpenAI 的企业转型)
> - `Token Usage in Generation` (生成过程中的 Token 使用量)


- **关于模型成本效益的辩论**：讨论了使用较小模型是否能节省成本，但一些人坚持认为**高质量**应优先于**廉价方案**。
   - 一名成员强调，尽管成本很高，但他们从创建价值 **$15-20k** 的数据集中看到了利润。
- **API 使用与自托管 GPU 的对比**：对话转向使用 API 是否比运行 **8 台 H100 GPU** 更便宜，一名成员声称 24 小时的成本约为 **$384**。
   - 另一个观点是，在大量使用 Token 的情况下，**GPU 成本**可能比使用可能达到 **2-5k** 的 API 成本更低。
- **Token 过载讨论**：成员们讨论了他们设置中惊人的 **Token 使用量**，对于密集型流水线，单次生成的估计量在 **10-15M Token** 之间。
   - 一位参与者对那些不了解其复杂设置的人所做的假设表示沮丧。
- **OpenAI 向企业文化的转变**：一名成员分享了一个链接，提到 **Mira Murati** 关于 OpenAI 近期变化的笔记，预示着其企业文化可能发生转变。
   - 成员们对 OpenAI 变得不再像一个令人兴奋的 Startup，并可能进入**企业模式**（corporate mode）表示担忧，这可能是由于管理层变动所致。



**提到的链接**：<a href="https://x.com/miramurati/status/1839025700009030027">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下笔记。

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1288255425648656475)** (52 messages🔥): 

> - `KTO and ORPO methods` (KTO 和 ORPO 方法)
> - `OpenAI's Feedback Mechanism` (OpenAI 的反馈机制)
> - `Llama fine-tuning inquiries` (Llama 微调咨询)
> - `Issues with Llama model` (Llama 模型问题)
> - `Spider Dataset and SQL Assistants` (Spider 数据集与 SQL 助手)


- **关于引导模型的 KTO 和 ORPO 方法的讨论**：一名成员就如何利用过去的错误案例教导模型寻求建议，并对 KTO 的二元真/假结构表示沮丧。
   - 他们对训练模型的不同方法以及如何将反馈整合到性能增强中表现出兴趣。
- **对 OpenAI 反馈过程的好奇**：成员们讨论了 OpenAI 如何使用 RLHF（来自人类反馈的强化学习）来改进他们的模型，一些人对具体的实施细节提出了疑问。
   - 有人询问了 OpenAI 整合反馈的确切方法，但结论性的细节仍未明确。
- **关于微调 Llama 模型的咨询**：几位成员询问了在本地微调 Llama 3.1 模型的可能性，建议指向安装 Unsloth 并使用提供的脚本。
   - 讨论还涉及了 Llama Vision 模型的微调状态，并提到了未来支持的预期时间表。
- **Llama 模型遇到的问题**：一名成员将从 Unsloth 下载的 Llama 3.1 8B 模型与来自 ollama.ai 的模型进行了性能对比，注意到输出质量存在差异。
   - 他们提出了关于模型能力的潜在差异问题，这引发了关于不同来源输出有效性的讨论。
- **利用 Spider 数据集开发 Text-to-SQL 应用**：一名成员详细说明了他们利用 Spider text-to-SQL 数据集构建 MySQL 助手的意图，并分享了使用现有模型的经验。
   - 他们强调了 Llama 3.1 的积极结果，但表示希望通过专业数据集的微调进一步提高模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.themoviedb.org/tv/45318-ohay-spank">Hello! Spank</a>：森村爱子是一名初中生，身高比同龄人矮。她的父亲十年前出海后失踪。她的母亲是一名帽子设计师，前往了巴黎...</li><li><a href="https://github.com/unslothai/unsloth/issues/418">phi3 playbook gguf: llama_model_load: error loading model: vocab size mismatch · Issue #418 · unslothai/unsloth</a>：playbook 中的 llama.cpp 集成无法工作，无论如何我手动创建了 gguf 文件，但当我尝试使用 llama.cpp server 提供模型服务时，遇到了以下错误...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1288231645341548668)** (1 messages): 

> - `New Model Releases` (新模型发布)
> - `Gradio 5 Launch` (Gradio 5 发布)
> - `FinePersonas Data Set` (FinePersonas 数据集)
> - `HF Hub Google Cloud Integration` (HF Hub 与 Google Cloud 集成)
> - `Wikimedia Dataset Release` (Wikimedia 数据集发布)

- **Mistral Small 和新 Qwen 模型发布**：**Mistral Small (22B)** 和最新迭代的 [Qwen 系列模型](https://qwenlm.github.io/blog/qwen2.5/) 现已可在 Hugging Face 上探索，同时还发布了用于视频生成任务的 [CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V)。
   - 探索 [HF 集合](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) 并在 [Hugging Chat](https://huggingface.co/chat/settings/Qwen/Qwen2.5-72B-Instruct) 中试用模型。
- **Gradio 5 简化 ML 应用演示**：[Gradio 5](https://5-0-dev.gradio-website.pages.dev/playground) 的发布承诺通过快速简便的设置，提升构建和共享机器学习应用的用户体验。
   - 通过简单的 Python 函数，用户可以创建在任何平台上运行的界面，使其成为协作和演示的理想选择。
- **推出用于合成数据的 FinePersonas**：[FinePersonas-v0.1](https://x.com/reach_vb/status/1836882281434165629) 正式发布，提供 2100 万个许可宽松的角色（Personas），用于为各种应用生成多样化且可控的合成数据。
   - 该数据集允许用户创建逼真的指令、用户查询和特定领域的问题，以提升 LLM 的能力。
- **HF Hub 加强与 Google Cloud 的集成**：Hugging Face Hub 深化了与 Google Cloud 的 [Vertex AI Model Garden](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_exciting-update-for-ai-developers-the-hugging-activity-7242235533236609025-w2FA?utm_source=share&utm_medium=member_desktop) 的集成，提高了 AI 开发者的可访问性。
   - 此次更新为在 Google Cloud 生态系统内无缝部署模型和数据集带来了可能。
- **Wikimedia 发布结构化 Wikipedia 数据集**：Wikimedia 公布了一个供公众使用的早期测试版 [数据集](https://enterprise.wikimedia.com/blog/hugging-face-dataset/)，该数据集源自 Snapshot API，重点关注英语和法语的 Wikipedia 文章。
   - 该数据集旨在提供更多机器可读的响应，增强研究人员和开发者的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground)">Gradio</a>: 构建并分享令人愉悦的 Machine Learning 应用</li><li><a href="https://x.com/reach_vb/status/1836882281434165629)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: 介绍 FinePersonas-v0.1 —— 拥有宽松许可的 2100 万个 Personas（人物角色），用于生成大规模（多样化且可控）的合成数据！🔥 使用 @AIatMeta Llama 3.1 70B Instruct, @arg... 制作。</li><li><a href="https://x.com/mattkdouglas/status/1838403695605690444)">Matthew Douglas (@mattkdouglas) 的推文</a>: 宣布 bitsandbytes 0.44.0！我们实现了由 @Apple 研究员 @MatPagliardini, @GrangierDavid, 和 @PierreAblin 提出的 AdEMAMix 优化器的 8-bit 版本。</li><li><a href="https://x.com/micuelll/status/1838244638873809125)">Miquel Farré (@micuelll) 的推文</a>: 好奇 FineVideo 是如何构建的吗？🍿 我们开源了完整的抓取和处理脚本，将约 200 万个 YouTube 视频转换为用于训练视频基础模型的丰富、带标注的数据集。R...</li><li><a href="https://x.com/tomaarsen/status/1837132943728209921)">tomaarsen (@tomaarsen) 的推文</a>: 我刚刚发布了 Sentence Transformers v3.1.1 补丁版本，修复了某些模型的 hard negatives 挖掘工具。这个工具对于从你的 embedding 中获得更高性能非常有用...</li><li><a href="https://x.com/davidberenstei/status/1838482286523601339)">David Berenstein (@davidberenstei) 的推文</a>: 为什么即使在使用合成数据时，观察你的合成数据也很重要？DataCraft UX 更新。数据可能包含一些怪癖，比如重复的 prompts、过于复杂的措辞和 Markdown 格式...</li><li><a href="https://x.com/gabrielmbmb_/status/1838239658737549797)">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>: 好奇你可以用 FinePersonas 中的 2100 万个 Personas 做什么吗？一个用例是创建全新的数据集 —— 就像我刚才做的那样！FinePersonas 合成电子邮件对话 ✉️ 使用 distilab...</li><li><a href="https://x.com/Gradio/status/1838210842497560971)">Gradio (@Gradio) 的推文</a>: 🔥 由 @OzzyGT 开发的 Diffusers 快速 Inpaint（局部重绘）。在你想擦除或更改的主体上绘制 mask（遮罩），并写下你想用来 Inpaint 的内容。使用 Diffusers 和 Gradio 创作有趣的艺术作品 😎</li><li><a href="https://enterprise.wikimedia.com/blog/hugging-face-dataset/)">Hugging Face 上的 Wikipedia 数据集：AI/ML 的结构化内容</a>: Wikimedia Enterprise 在 Hugging Face 上发布 Wikipedia 数据集，包含来自 Snapshot API 的结构化内容测试版，适用于 AI 和 Machine Learning 应用</li><li><a href="https://x.com/qlhoest/status/1837179483201147279)">Quentin Lhoest 🤗 (@qlhoest) 的推文</a>: FinePersonas 是最丰富的 Personas 数据集。现在你可以通过 ReWrite 来调整这些 Personas 以适应你的需求（适用于 HF 上的任何数据集！）。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1288215967427461181)** (432 messages🔥🔥🔥): 

> - `Llama 3.2 release`
> - `vLLM memory management`
> - `Quantization techniques`
> - `Hugging Face libraries`
> - `Machine learning optimization` 


- **Llama 3.2 引入新模型**：Llama 3.2 已经发布，具有多模态支持和用于本地部署的小型模型，在文本和图像处理方面具有显著能力。
   - 这些更新包括 128k token 的上下文长度，以及专门为移动和边缘设备部署设计的模型。
- **在 Tesla T4 上使用 vLLM 的内存挑战**：用户报告由于 VRAM 限制，在 Tesla T4 GPU 上使用 vLLM 运行 Llama 3.1 存在困难，主要是当同时加载多个模型时。
   - 一位用户成功地单独运行了模型，但在尝试同时运行其他模型时遇到了 VRAM 耗尽的问题。
- **探索模型效率的量化技术**：讨论了量化技术（如将模型转换为 4-bit 或 8-bit 表示）对于使大型模型在有限的 VRAM 容量内运行至关重要。
   - Hugging Face 文档提供了关于应用这些量化策略以优化模型性能并减少内存负载的指导。
- **代码执行的差异**：两个代码片段之间的比较表明，一个成功加载了模型而没有内存问题，而另一个则没有，这引发了关于代码优化的讨论。
   - 有人指出，第二个代码片段逐个处理模型，这可能解释了它相比第一个片段执行成功的原因。
- **从 AI 交互中学习**：用户强调他们希望通过积极询问来学习 AI，利用 ChatGPT 和 Claude 等工具来增强理解。
   - 在认识到自己编程局限性的同时，他们对利用现有资源更有效地掌握 AI 概念和实践表示乐观。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/seshubon/status/1838527532972359882">来自 seshu bonam (@seshubon) 的推文</a>: 今天在 workflow builder 中添加了 3 个强大的节点：📽️Text to Video 🏄Realism LoRA 或导入任何 LoRA url ✍️使用自然语言输入编辑图像。你可以从主页尝试这些并查看...</li><li><a href="https://x._philschmid/status/1838998169293615318">来自 Philipp Schmid (@_philschmid) 的推文</a>: Llama 现在可以看东西并在你的手机上运行了！👀🖼️ Llama 3.2 发布，在 Llama Vision 中支持 Multimodal，并提供用于设备端使用的 tiny llamas。@AIatMeta 发布了 10 个新的 llama，从 1B 纯文本到 90...</li><li><a href="https://tenor.com/view/hug-gif-27703442">Hug GIF - Hug - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://graphic.so/workflows/editor/d4462ed0-c2b2-4a6c-8fb3-f9b08e8e7df0/chat">graphic : AI 驱动的 Workflow Automation</a>: 未找到描述</li><li><a href="https://tenor.com/view/wink-wink-agnes-agatha-harkness-kathryn-hahn-wandavision-gif-22927975">Wink Wink Agnes GIF - Wink Wink Agnes Agatha Harkness - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-simpsons-moe-syzlak-get-a-load-of-this-guy-shrug-talking-crazy-gif-6439732525981754175">The Simpsons Moe Syzlak GIF - The simpsons Moe syzlak Get a load of this guy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_pro/">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=GVsUOuSjvcg&t=1067s">未来的计算机将截然不同 (Analog Computing)</a>: 访问 https://brilliant.org/Veritasium/ 免费开始学习 STEM，前 200 名用户将获得年度高级订阅的 8 折优惠。D...</li><li><a href="https://www.youtube.com/watch?v=QLGlrY7cooY">The Tax Breaks (Twilight) [15.ai]</a>: 多年来，小马谷面临过许多巨大的威胁，但面对可怕的税务小马的造访，他们将如何应对？AI 工具：15.ai - https://15.ai/Adapte...</li><li><a href="https://x.com/sama/status/1756089361609981993">来自 Sam Altman (@sama) 的推文</a>: OpenAI 现在每天生成约 1000 亿个单词。地球上所有人每天生成约 100 万亿个单词。</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/llm_tutorial_optimization">优化 LLM 的速度和内存</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/main_classes/quantization">Quantization</a>: 未找到描述</li><li><a href="https://github.com/DigitalPhonetics/IMS-Toucan">GitHub - DigitalPhonetics/IMS-Toucan: 斯图加特大学语音与语言技术小组的多语言且可控的 Text-to-Speech 工具包。</a>: 斯图加特大学语音与语言技术小组的多语言且可控的 Text-to-Speech 工具包。 - DigitalPhonetics/IMS-Toucan</li><li><a href="https://github.com/ai-graphic/Graphic-so">GitHub - ai-graphic/Graphic-so: Graphic.so 是一个 Multi Modal AI 游乐场，通过自然语言界面将构建 AI 应用和自动化的速度提高 10 倍。</a>: Graphic.so 是一个 Multi Modal AI 游乐场，通过自然语言界面将构建 AI 应用和自动化的速度提高 10 倍。 - ai-graphic/Graphic-so
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1288454425899175956)** (7 messages): 

> - `Fine-tuning LLMs`
> - `Stable Diffusion Models`
> - `Diffusion Models from Scratch`
> - `Optimizing Embeddings with PCA` 


- **使用另一个 LLM 对 LLM 进行 Fine-tuning**：一位成员澄清说，可以使用另一个 **LLM** 来创建与输出关联的输入，从而对 **LLM** 进行 **Fine-tuning**，而不是仅仅依赖人工输入。
   - 这种方法可以为训练目的提供更高效的数据生成。
- **运行 Stable Diffusion 模型**：一位用户分享了他们在学习 **Hugging Face 课程**中运行 **Stable Diffusion** 模型的经验。
   - 他们提到他们的活动非常基础，针对的是该领域的新手。
- **从零开始构建 Diffusion Models**：一位成员报告了他们从零开始开发 **diffusion models** 的进展，重点是使用 **Rust** 和 **WGSL** 实现 **convolutional neural networks**。
   - 他们的初步里程碑包括成功运行高斯模糊核（Gaussian blur kernels）进行测试。
- **使用 PCA 优化 Embeddings**：一位用户正在学习如何使用 **sklearn** 通过**主成分分析 (PCA)** 来优化 embeddings。
   - 这个话题表明了对降维技术（dimensionality reduction techniques）的重视，以实现更高效的数据处理。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288222828184735754)** (12 messages🔥): 

> - `Comic Sans FLUX Model`
> - `Neural Network Learning Constraints`
> - `Digital Processing of Postal Codes`
> - `DHL Manual Sorting`
> - `Deutsche Post Operations` 


- **Comic Sans FLUX Model 助力文本生成**：[Comic Sans FLUX model](https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572) 的发布旨在增强 Text-Tacular Showdown 竞赛，允许用户使用这种标志性字体生成带有大量文本的图像。
   - *趣味事实：* 在 Google 搜索 "comic sans"，搜索结果将全部以 Comic Sans 字体渲染。
- **受领域约束增强的学习网络**：一篇引用的论文强调了增加任务领域约束如何显著提高学习网络的泛化能力，特别是对于手写数字识别。
   - 这种方法展示了一种用于识别邮政服务中邮政编码数字的统一方法，证明了处理效率。
- **各国邮政服务对比**：讨论涉及 **德国** 如何成为早期采用邮政编码数字化处理的国家，但对其与其他国家相比的性能仍存在疑问。
   - 一位成员指出邮政服务中仍然依赖人工分拣（manual sorting），引发了关于运营效率的讨论。
- **DHL 的分拣实践受到关注**：对一段视频的回应显示，尽管技术有所进步，**DHL** 在包裹处理中仍涉及大量人工分拣，正如一段 [YouTube 视频](https://youtu.be/tLqjlTKiR9o) 中所提到的。
   - 有人担心这是否反映了邮政服务中一个更广泛的问题，即自动化尚未完全取代人工方法。
- **走进 Deutsche Post 运营内部**：一位成员分享的视频提供了对 **Deutsche Post** 运营的见解，展示了每天处理约一百万件的大量邮件是自动完成的。
   - 然而，观察表明仍然存在*大量的手工分拣*，强调了自动化和人工流程的混合。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u-x6o8ySG0sC">Backpropagation applied to handwritten zip code recognition</a>：Y LeCun, B Boser, JS Denker, D Henderson, RE Howard, W Hubbard, LD Jackel, Neural computation, 1989 - 被引用 17,411 次</li><li><a href="https://youtu.be/tLqjlTKiR9o>">AirTags Expose Dodgy Postal Industry (DHL Responds)</a>：感谢 Brilliant 赞助本视频！免费试用 Brilliant：https://brilliant.org/MegaLagAirTagAlex's Video: https://www.youtube.com/watch?v=tRIdo...</li><li><a href="https://www.youtube.com/watch?v=m9g8Fn9EvGQ">BRIEFZENTRUM BERLIN: Hier geht mit täglich einer Million Sendungen mächtig die Post ab! | Magazin</a>：Deutsche Post 可能很快会发生一些变化，这将影响您的钱包。您愿意为更快的速度支付更多邮资吗...</li><li><a href="https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572">Comic Sans Font for Flux - V1 | Stable Diffusion LoRA | Civitai</a>：赶在 Text-Tacular Showdown 竞赛之前，通过使用此 FLUX 模型准确生成带有文本的图像，在竞争中获得优势...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1288216764051620022)** (125 条消息🔥🔥): 

> - `Google Gemini demo`
> - `Zeromagic API platform`
> - `Transcription tools`
> - `PCA optimization project`
> - `Game development and quantum programming` 


- **Google Gemini 演示展示了 bounding box 功能**：一位成员使用 **Google Gemini** 创建了一个演示，可以从图像中提供 **bounding boxes** 的坐标，这引发了关于其性能的讨论。
   - *Qtjack* 对该演示表示赞赏，而其他人则讨论了将 **YOLO** 作为基准的替代方案。
- **Zeromagic 加速 API 构建**：一位成员介绍了 **Zeromagic**，这是一个 AI 驱动的低代码平台，可显著加快 REST 和 GraphQL API 的创建速度。
   - 他们分享了项目链接并鼓励反馈，强调了该平台对中小企业的益处。
- **转录工具开发趋势**：一位成员讨论了开发一种工具的计划，该工具可以从 **YouTube** 视频中提取音频、进行转录，并将其对齐用于博客文章和社交媒体。
   - 反馈建议增加审查复选框并提高可用性，特别是针对教育内容。
- **创新的 PCA 优化项目启动**：一位成员详细介绍了他们新的 **PCA optimization** Python 包，旨在减少数据维度的同时保持 Embedding 之间的关系。
   - 讨论包括未来在 **PyPI** 上市的计划，强调了 PCA 在机器学习和数据科学中的相关性。
- **游戏开发与学习理念**：一位成员分享了他们在游戏开发创业中的经验，讨论了 3D **Tetris** 游戏等项目以及在量子编程方面的努力。
   - 他们提到了持续学习的重要性，并应用 **80/20 Pareto principle** 来平衡工作与娱乐。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/p3nGu1nZz/Kyle-b0a">p3nGu1nZz/Kyle-b0a · Hugging Face</a>: 未发现描述</li><li><a href="https://huggingface.co/spaces/qamarsidd/FreeTranscriptMaker">FreeTranscriptMaker - a Hugging Face Space by qamarsidd</a>: 未发现描述</li><li><a href="https://www.youtube.com/@BatCountryEnt/videos#:~:text=Share%20your%20videos%20with%20friends,%20family,%20and%20the%20world">BatCountryEnt</a>: 未发现描述</li><li><a href="https://huggingface.co/spaces/saq1b/gemini-object-detection">Gemini Object Detection - a Hugging Face Space by saq1b</a>: 未发现描述</li><li><a href="https://github.com/p3nGu1nZz/Tau/blob/dev-pca-optimization-script/MLAgentsProject/Scripts/optimizer.py">Tau/MLAgentsProject/Scripts/optimizer.py at dev-pca-optimization-script · p3nGu1nZz/Tau</a>: 使用 Unity 6 ML Agents 制作的 Tau LLM。通过在 GitHub 上创建账号为 p3nGu1nZz/Tau 开发做贡献。</li><li><a href="https://github.com/p3nGu1nZz/oproof/blob/main/oproof/main.py">oproof/oproof/main.py at main · p3nGu1nZz/oproof</a>: 使用 Ollama 和 Python 验证 Prompt-Response 对。 - p3nGu1nZz/oproof</li><li><a href="https://github.com/ytdl-org/youtube-dl">GitHub - ytdl-org/youtube-dl: Command-line program to download videos from YouTube.com and other video sites</a>: 用于从 YouTube.com 和其他视频网站下载视频的命令行程序 - ytdl-org/youtube-dl</li><li><a href="https://zeromagic.cloud/">ZeroMagic - Build, Deploy, and Scale Your Application Faster</a>: Zeromagic 是一个 AI 驱动的低代码 SAAS 平台，帮助开发者和企业以 10 倍的速度构建 REST 和 GraphQL API，并实现即时部署，一切仅需几分钟。</li><li><a href="https://docs.zeromagic.cloud/">Zeromagic Documentation</a>: 未发现描述</li><li><a href="https://docs.zeromagic.cloud/blog/">Zeromagic Documentation</a>: 未发现描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288532340855017554)** (3 条消息): 

> - `Molmo models`
> - `Multimodal comparison`
> - `Graphs and resources` 


- **对多模态 Molmo 模型的期待**：一位成员表达了对分享新型多模态 **Molmo models** 见解的兴趣，并重点推荐了[这篇博客文章](https://molmo.allenai.org/blog)，其中包含许多图表和对比。
   - *是的！如果有人感兴趣，这里有博客文章，* 另一位成员指出，表示该文章提供了丰富的信息。
- **请求更多背景信息**：一位成员请求其他人**分享资源和背景信息**相关的讨论，为进一步的见解交流拉开了序幕。
   - *“分享资源/背景”* 成为了成员间协作学习的号召。



**提到的链接**: <a href="https://molmo.allenai.org/blog">未发现标题</a>: 未发现描述

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1288226007899312293)** (12 messages🔥): 

> - `Training Topic Clusters` (训练主题聚类)
> - `Setfit Serialization Issue` (Setfit 序列化问题)
> - `vLLM Deployment for NER` (用于 NER 的 vLLM 部署)
> - `Fine-tuning Token Embeddings` (微调 Token Embeddings)


- **训练主题聚类的挑战**：一位成员表示，在不进行大量手动合并的情况下，很难获得合理数量的主题，认为这在生产环境中是不可行的。
   - 作为回应，另一位成员分享了他们部署 zero-shot 系统的经验，并建议灵活处理新主题。
- **Setfit 序列化参数查询**：一位成员询问了 Setfit 中的一个参数，该参数由于集群上的内存限制而阻止保存序列化的 safetensors。
   - 其他人提到 'save_strategy' 可能有助于在训练期间管理 Checkpoint 的保存。
- **使用 vLLM 优化 NER**：一位成员讨论了使用 vLLM 部署 LLM，并探索了如何提高 **bert-base-NER** 模型在命名实体识别（Named Entity Recognition）方面的性能。
   - 他们询问了如何将 Pipeline 与 vLLM 结合用于 NER 任务，并在 Triton 上成功设置后寻求更简单的打包替代方案。
- **微调新 Token Embeddings**：一位成员询问如何仅微调模型中新添加 token 的 embeddings，同时保留现有 token 的 embeddings。
   - 这个问题表明了关于如何在不破坏已建立的 token 功能的情况下管理模型更新的持续讨论。



**提到的链接**：<a href="https://huggingface.co/dslim/bert-base-NER">dslim/bert-base-NER · Hugging Face</a>：未找到描述

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1288459206629916733)** (4 messages): 

> - `Google Colab`
> - `Diffusion Models`
> - `Flux Model`
> - `SDXL Lightning`
> - `Würstchen Model` 


- **Google Colab 免费层支持多种模型**：成员们讨论了在 **Google Colab 免费层**运行 Diffusion 模型的可能性，并指出大多数模型都可以被有效利用。
   - 一位成员强调，在寻求建议时，需要明确定义什么是**“性能相对较好”**的模型。
- **Flux 模型作为开源竞争者脱颖而出**：一位成员推荐使用 **Flux** 模型，称其是目前最好的开源 Diffusion 模型，并且可以在 Colab 中无障碍运行。
   - 然而，他们提醒说，使用该模型生成图像可能需要相当长的时间。
- **SDXL Lightning 用于更快速的图像生成**：为了更快速地生成图像，成员们建议将 **SDXL Lightning 类型模型**作为 Flux 的可靠替代方案，它们能提供质量尚可的图像。
   - 这一选项被认为对于那些优先考虑速度而非高保真度的用户特别有利。


  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1288615728127148032)** (1 messages): 

> - `Hermes 3`
> - `Llama-3.1 8B`
> - `HuggingChat` 


- **Hermes 3 在 HuggingChat 上线**：最新发布的 **8B** 规格 **Hermes 3** 现已在 [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B) 上线，展示了改进的指令遵循能力。
   - 该模型是 **Nous Research** 提供的产品之一，旨在增强用户交互体验。
- **指令遵循能力增强**：**Hermes 3** 显著提高了遵循指令的能力，从而在对话交互中表现更佳。
   - 与之前的版本相比，用户可以期待更准确、更具上下文相关性的回答。



**提到的链接**：<a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>：让社区最好的 AI 对话模型惠及每一个人。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1288213368523722866)** (214 messages🔥🔥): 

> - `Llama 3.2 Model Release` (Llama 3.2 模型发布)
> - `Quantization vs. Distillation` (量化 vs. 蒸馏)
> - `New Innovations in AI Training` (AI 训练的新创新)
> - `Hermes 3 Update` (Hermes 3 更新)
> - `Localization of Models` (模型本地化)

- **Llama 3.2 模型性能洞察**：Llama 3.2 已经发布了多种尺寸，引发了关于其性能的讨论，特别是与 Llama 1B 和 3B 等较小模型的比较。
   - 用户注意到了这些模型的具体能力和局限性，一些人称赞它们生成功能性代码的能力。
- **关于 Quantization 与 Distillation 的辩论**：对话强调，虽然 Meta 在模型压缩方面更倾向于 Distillation，但由于 Quantization 在各种硬件上的互补优势和效率，它仍然具有相关性。
   - 参与者指出，这两种技术在解决不同应用的内存和计算需求方面都非常重要。
- **训练创新与改进**：关于 DPO 之后的 AI 训练技术进展出现了疑问，重点关注模型 Distillation 的有效性。
   - 社区对优化模型训练的持续创新表达了好奇。
- **HuggingChat 上的 Hermes 3**：注意到 Hermes 3 在 HuggingChat 上的发布，其采用了 Llama 3.1 的 8B 配置，旨在紧密遵循指令。
   - 这一更新反映了高级语言模型应用方面的持续发展。
- **AI 模型的本地化挑战**：人们对使用 Llama 3.2 等低精度模型处理复杂任务（特别是在各种语言中）的限制表示担忧。
   - 讨论强调，虽然较小的模型可以生成英语，但在处理代码编写和外语生成等更细致的任务时会感到吃力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1838987356810199153">来自 Daniel Han (@danielhanchen) 的推文</a>：Llama 3.2 多模态发布了！模型尺寸涵盖 1B, 3B 到 11B 和 90B！</li><li><a href="https://molmo.allenai.org/blog">未找到标题</a>：未找到描述</li><li><a href="https://x.com/AIatMeta/status/1838993953502515702?t=Lva0weiqBSGrpNXp02yz9Q&s=19">来自 AI at Meta (@AIatMeta) 的推文</a>：📣 隆重推出 Llama 3.2：适用于边缘设备的轻量级模型、视觉模型等！有哪些新特性？• Llama 3.2 1B 和 3B 模型在多项设备端应用中展现了同类产品中最先进的能力...</li><li><a href="https://huggingface.co/collections/alpindale/llama-32-re-upload-66f463d7940e8a6c7f5b7bbc">Llama 3.2 重新上传 - alpindale 收藏集</a>：未找到描述</li><li><a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>：让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF">bartowski/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/llama-3.2-3B-Instruct">Llama 3.2 3B Instruct - 由 huggingface-projects 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF">bartowski/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=ezXhxAkhJfk">Meta Connect 2024 全程直播。Meta Quest 3S 及更多</a>：Meta Connect 2024 全程直播揭晓了 Meta Quest 3S 等。这是年度 Meta Connect 大会，我们终于等到了 Meta 的发布...</li><li><a href="https://apkpure.com/layla-lite/com.laylalite">Layla Lite APK Android 版下载</a>：未找到描述</li><li><a href="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF">hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fp9wem/llama_32_1b_3b_benchmarks/)">Llama 3.2 1B &amp; 3B 基准测试</a>：由 u/TKGaming_11 发布于 r/LocalLLaMA • 67 分和 5 条评论</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpb4m3/molmo_models_outperform_llama_32_in_most_vision/">Molmo 模型在大多数视觉基准测试中优于 Llama 3.2 🌟</a>：由 u/shrewdeenger 发布于 r/LocalLLaMA • 55 分和 7 条评论</li><li><a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi</a>：通过在 GitHub 上创建账户，为 kyutai-labs/moshi 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf">未找到标题</a>：未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-llm/">Qwen2.5-LLM：扩展 LLM 的边界</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 在这篇博客中，我们将深入探讨最新的 Qwen2.5 系列语言模型的细节。我们开发了一系列仅 decoder-only 的稠密模型...</li><li><a href="https://www.llama.com/">Llama 3.2</a>：您可以随处进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B, 70B 和 405B 版本。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1288215366371119236)** (25 messages🔥): 

> - `Sample Packing Techniques`
> - `Tokenizers in Model Training`
> - `LLaMA 3.2 Vision Encoder` 


- **创新的 Sample Packing 引发性能疑问**：在训练小型 **GPT-2** 模型时，围绕 **Sample Packing** 的实现展开了讨论，成员们强调如果处理不当，可能会导致性能下降。
   - 一位参与者建议，尽管可以将 Attention 分数置零，但使用所提议的方法仍可能导致结果不理想。
- **Special Tokens 在句子上下文中的作用**：有人建议添加像 **'endoftext'** 这样的 Special Tokens 可以帮助在训练中明确句子边界，尽管有些人认为这并非严格必要。
   - 另一位成员提到，大多数现成的 Tokenizers 如果在配置中指定，会自动包含 Special Tokens。
- **Tokenizer 替换的成功率仍不确定**：跨模型成功替换 Tokenizer 的潜力引发了询问，特别是关于 Huggingface Tokenizers 管理 Special Tokens 的能力。
   - 对话显示，与现有 Tokenizers 相比，自定义 Tokenizers 在这方面的能力尚存在不确定性。
- **LLaMA 3.2 Vision Encoder 惊人的尺寸**：一位成员分享了关于 **LLaMA 3.2** Vision Encoder 的见解，指出不同模型变体的尺寸高达 **18B** 和 **3B**。
   - 讨论强调文本 Decoder 与之前的版本（L3.1）保持一致，这引发了人们对 Encoder 尺寸所带来影响的好奇。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288381544985006160)** (2 messages): 

> - `Character Video Synthesis`
> - `Resume ATS Builder`
> - `Job Recommendation Systems` 


- **MIMO 框架革新角色视频合成**：一个名为 **MIMO** 的新颖框架为逼真的角色视频合成提供了解决方案，通过简单的用户输入生成具有可控属性（如角色、动作和场景）的视频。
   - 它旨在克服需要多视角捕捉的 **3D methods** 的局限性，并利用对任意角色的**高级可扩展性**（advanced scalability）增强了姿态通用性和场景交互。
- **简历 ATS 和职位推荐需要建议**：一位成员分享了他们在开发**简历 ATS 构建器**以及**职位匹配和推荐系统**方面的经验，但在寻找高质量研究论文时感到迷茫。
   - 他们寻求他人的指导，以了解如何高效地开展该领域的研究工作。



**提及的链接**：<a href="https://huggingface.co/papers/2409.16160">Paper page - MIMO: Controllable Character Video Synthesis with Spatial Decomposed
  Modeling</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1288590690539929713)** (2 messages): 

> - `Opus Insight updates`
> - `Sonnet 3.5 review process`
> - `ThinkLab exploration`
> - `O1-preview functionality` 


- **Opus Insight 引入 O1mini**：**Opus Insight** 的最新更新集成了 **O1mini**，通过全面的模型调整和排名评审增强了其多模型模板。
   - 在此过程中，**Sonnet 3.5** 处理初始评审，随后由 **O1mini** 进行最终评审和模型排名。
- **由 Sonar Huge 405b 驱动的 ThinkLab**：**ThinkLab** 利用 **Sonar Huge 405b** 模型进行网络搜索，强调其在 Scratchpad 使用和后续搜索以扩大探索范围方面的实用性。
   - 这种方法旨在简化探索过程并增强用户与内容的交互。
- **O1-preview 功能的 Rate limits**：**O1-preview** 在 Wordware 应用中作为一个选项可用，但它受到 **Rate limited**（速率限制），如果不返回数据可能会导致停顿。
   - 该功能已添加到应用中，但默认禁用；用户可以通过点击“create my own version”来启用它，以测试专用的 O1-preview 流程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://app.wordware.ai/explore/apps/aa2996a0-93c9-4c19-ade2-1796c5c8a409">OPUS Insight : Latest Model Ranking - o1mini</a>：此 Prompt 使用最新的模型处理问题，并提供全面的评审和排名。更新：2024/9/25 - 已添加：o1mini, Gemini 1.5 Flash, Command R+。注意：o1-preview 是...的一部分</li><li><a href="https://app.wordware.ai/explore/apps/999cc252-5181-42b9-a6d3-060b4e9f858d">_Think-Lab Revised - o1mini</a>：(版本 1.10) 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精炼的搜索查询。Scratchpad 是一个强大的工具，可以帮助您保持连贯性和准确性...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288381544985006160)** (2 条消息): 

> - `Character video synthesis`
> - `Resume ATS builder`
> - `Job recommendation systems` 


- **用于 Character Video Synthesis 的 MIMO 框架**：[MIMO 框架](https://huggingface.co/papers/2409.16160)提出了一种从简单的用户输入中合成具有可控属性（如角色、动作和场景）的真实角色视频的方法。
   - 它旨在通过在一个统一的框架中实现 **scalability**、**generality** 和 **interactivity**，来解决传统 3D 方法和现有 2D 方法的局限性。
- **在简历系统中寻找高质量研究**：一位成员表示，在搜索研究论文以协助构建 **resume ATS** 构建器和职位推荐系统时感到迷茫。
   - *寻求建议*关于如何在海量的现有研究中有效地进行此类搜索。



**提到的链接**：<a href="https://huggingface.co/papers/2409.16160">Paper page - MIMO: Controllable Character Video Synthesis with Spatial Decomposed
  Modeling</a>：未找到描述

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1288265040327348236)** (95 条消息🔥🔥): 

> - `Llama 3.2 发布`
> - `Aider 功能`
> - `向量数据库`
> - `Meta Connect`
> - `Sonnet 3.5 错误` 


- **Llama 3.2 发布**: Meta 宣布发布 **Llama 3.2**，包括中小型视觉 LLM 以及适用于边缘和移动设备的轻量级模型。
   - 正如 **Meta Connect** 期间所强调的，这些新模型旨在为没有大量资源的开发者提供更易获得的 AI 能力。
- **Aider 的潜在扩展**: 用户讨论了 **Aider** 在缺乏内置翻译功能时的局限性，并强调了对更好文档索引的需求。
   - 用户对添加语音反馈和自动文档搜索等功能以增强用户体验表现出浓厚兴趣。
- **向量数据库选项**: 成员们分享了对本地向量数据库的看法，提到了 **Chroma**、**Qdrant** 以及具有向量扩展功能的 **PostgreSQL** 的潜力。
   - 虽然 SQLite 和 **PostgreSQL** 可以执行向量数据库任务，但专用向量数据库被认为在处理重负载时效率更高。
- **持续的 Sonnet 3.5 问题**: 一起关于 **Sonnet 3.5** 的事件导致用户遇到的错误率上升，该问题已在 Anthropic 状态页面上报告并解决。
   - 这一事件表明 Anthropic 模型的可用性和性能存在持续波动，影响了用户的工作流。
- **Meta Connect 期间的参与度**: 成员们积极参与了关于 **Meta Connect** 的讨论，分享了对新发布模型（如 **Llama 3.2**）的见解。
   - 讨论内容包括这些发布可能如何影响未来的使用和开发考量。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://scale.com/leaderboard/coding">Scale | SEAL 排行榜：编程评估</a>: Scale 的 SEAL 编程排行榜对顶级 LLM 在编程语言、学科和任务方面进行评估和排名。</li><li><a href="https://aider.chat/docs/repomap.html">仓库地图 (Repository map)</a>: Aider 使用 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://www.facebook.com/MetaforDevelopers/videos/449444780818091/">22M 观看 · 33K 评论 | Meta Connect 2024 | 加入 Mark Zuckerberg，分享 Meta 对 AI 和元宇宙的愿景，包括 Meta 最新的产品发布。

然后继续关注... | By Meta for DevelopersFacebook</a>: 加入 Mark Zuckerberg，分享 Meta 对 AI 和元宇宙的愿景，包括 Meta 最新的产品发布。然后继续关注...</li><li><a href="https://www.answer.ai/posts/2024-09-03-llmstxt.html">/llms.txt — 一个为 LLM 使用网站提供信息的提案 – Answer.AI</a>: 我们建议有兴趣提供 LLM 友好型内容的人在他们的网站中添加一个 /llms.txt 文件。这是一个 Markdown 文件，提供简要的背景信息和指导，以及链接...</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">未找到标题</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fp9had/llama_32_multimodal/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://status.anthropic.com/incidents/rzbypbn1f7lf">Claude 3.5 Sonnet 错误率上升</a>: 未找到描述</li><li><a href="https://github.com/mendableai/firecrawl?tab=readme-ov-file)">GitHub - mendableai/firecrawl: 🔥 将整个网站转换为 LLM 就绪的 Markdown 或结构化数据。通过单个 API 进行抓取、爬取和提取。</a>: 🔥 将整个网站转换为 LLM 就绪的 Markdown 或结构化数据。通过单个 API 进行抓取、爬取和提取。 - mendableai/firecrawl
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1288221842321969255)** (104 条消息🔥🔥): 

> - `Aider Model Issues` (Aider 模型问题)
> - `Using Aider with PDFs` (在 Aider 中使用 PDF)
> - `Testing Aider with Various Models` (使用各种模型测试 Aider)
> - `Aider's Token Usage` (Aider 的 Token 使用情况)
> - `Integrating External Libraries and Analyzers` (集成外部库和分析器)


- **对 Claude Sonnet 性能的担忧**：用户报告了 **Claude Sonnet 3.5** 在性能和代码库理解能力方面有所下降，导致出现错误和不正确的假设。
   - 不过，一些用户通过在特定提示词下切换到 **Gemini Pro 1.5** 等模型，然后再切回 Sonnet，成功改善了结果。
- **Aider 处理 PDF 文件的功能**：讨论强调 **PDF 文件** 是二进制格式，LLM 无法直接读取，用户应将其转换为文本格式以便更好地理解。
   - 建议使用 Jina Reader 等工具将 URL 转换为文本，但也有人对 LaTeX 中方程式的解析表示担忧。
- **切换 Aider 模型以获得更好的输出**：几位用户注意到不同模型的成功程度各不相同，这促使一些人探索切换到其他 LLM 以获得更好的编码结果。
   - 普遍共识是使用 Aider 的 benchmark 套件监控并报告模型性能，以确保评估的准确性。
- **了解 Aider 的 Token 管理**：用户经常使用 `/tokens` 命令检查其 Token 使用情况，以管理上下文并避免过多的内存错误。
   - 保持精简的上下文对于高效性能至关重要，因为较大的 Token 使用量往往会导致对代码的误解或理解不完整。
- **在 Aider 中集成外部库**：一位用户对 Aider 处理特定 **Rust 库**和其他外部分析器的能力表示担忧，这可能会阻碍其生成正确代码的效率。
   - 这些资源的集成被认为是提高 LLM 在处理复杂语言规则时准确编码性能的关键。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://jina.ai/reader/">Reader API</a>: 读取 URL 或搜索网络，为 LLM 提供更好的基础。</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: 由 aider 用户制作的介绍和教程视频。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages)">In-chat commands</a>: 使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://github.com/zed-industries/zed/discussions/18149)">How to forward option-enter (alt-enter) to terminal? · zed-industries/zed · Discussion #18149</a>: 我在向终端发送 option-enter 按键时遇到困难，我已经阅读了 https://zed.dev/docs/key-bindings#forward-keys-to-terminal 并在我的键位映射中尝试了以下操作：[ {...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1288284061688795177)** (4 messages): 

> - `par_scrape 工具`
> - `shot-scraper 实用程序`
> - `Llama 3.2 Multimodal`
> - `o1-preview Bug 修复` 


- **探索 par_scrape 工具**：一位成员在 GitHub 上分享了一个名为 [par_scrape](https://github.com/paulrobello/par_scrape) 的有趣工具，旨在简化网页抓取任务。
   - 据指出，与替代方案相比，该工具可能效率更高。
- **Shot-scraper 超出预期**：另一位成员强调 [shot-scraper](https://shot-scraper.datasette.io/en/stable/) 实用程序是一个不错的替代方案，并表示在某些任务上它比 par_scrape 更简单。
   - *“天哪，这真是个好工具”* 强调了它作为自动化网站截图命令行实用程序的能力。
- **Llama 3.2 Multimodal 备受关注**：分享了一个标题为“Llama 3.2 Multimodal”的 Reddit 帖子链接，正等待版主批准以进行进一步讨论。
   - 它激发了人们对 Llama 模型最新进展的兴趣。
- **使用 o1-preview 修复 LLM Bug**：一篇帖子讨论了使用 [o1-preview](https://simonwillison.net/2024/Sep/25/o1-preview-llm/) 解决 Bug 的过程，并提到为 DJP 添加了一个用于插件元数据定位的新功能。
   - 作者分享了他们结合使用 [files-to-prompt](https://github.com/simonw/files-to-prompt) 和 OpenAI 的 LLM 来解决测试失败的经验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2024/Sep/25/o1-preview-llm/">使用 o1-preview、files-to-prompt 和 LLM 解决 Bug</a>：今天早上我为 DJP 添加了<a href="https://github.com/simonw/djp/issues/10">一个新功能</a>：现在你可以让插件指定其元数据相对于其他元数据的定位方式...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fp9had/llama_32_multimodal/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/paulrobello/par_scrape">GitHub - paulrobello/par_scrape</a>：通过在 GitHub 上创建账户来为 paulrobello/par_scrape 的开发做出贡献。</li><li><a href="https://shot-scraper.datasette.io/en/stable/">shot-scraper</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288268772817244292)** (2 条消息): 

> - `数据库升级`
> - `API Completion 响应变更`
> - `Gemini 模型更新`
> - `新的 Vision Language Models`
> - `Cohere 模型折扣` 


- **数据库升级计划将导致停机**：数据库升级定于 **美国东部时间周五上午 10 点**，预计将导致 **5-10 分钟的停机**。
   - 建议用户针对潜在的服务中断做好相应准备。
- **响应输出中的 API 增强**：用于处理请求的 **provider** 现在直接包含在 **completion response** 中。
   - 此更新旨在简化返回给用户的信息量。
- **Gemini 模型路由已更新**：**Gemini-1.5-flash** 和 **Gemini-1.5-pro** 现在路由到最新的 **002 版本**。
   - 此更改是 Gemini 模型系列持续改进的一部分。
- **令人兴奋的新 Vision Language Models 发布**：OpenRouter 现在上线了一系列新的 **开源 Vision Language Models**，可供交互。
   - 模型包括 **Mistral Pixtral 12B** 和 **Qwen2-VL** 系列，鼓励用户在聊天室中使用。
- **所有 Cohere 模型 5% 折扣**：OpenRouter 独家推出了所有 **Cohere 模型** 的 **5% 折扣**。
   - 用户可以通过提供的链接探索具有 **128k context** 的旗舰模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ - API, Providers, Stats</a>: Command R+ 是来自 Cohere 的新型 104B 参数 LLM。它适用于角色扮演、通用消费者用例和 Retrieval Augmented Generation (RAG)。使用 API 运行 Command R+</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: 来自 Mistral AI 的首个 image to text 模型。其权重按照其传统通过种子（torrent）发布：https://x。使用 API 运行 Pixtral 12B</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct">Qwen2-VL 7B Instruct - API, Providers, Stats</a>: Qwen2 VL 7B 是来自 Qwen 团队的多模态 LLM，具有以下关键增强：- 对各种分辨率和比例的图像具有 SoTA 级别的理解：Qwen2-VL 实现了最先进的性能...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-72b-instruct">Qwen2-VL 72B Instruct - API, Providers, Stats</a>: Qwen2 VL 72B 是来自 Qwen 团队的多模态 LLM，具有以下关键增强：- 对各种分辨率和比例的图像具有 SoTA 级别的理解：Qwen2-VL 实现了最先进的性能...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1288214663686783048)** (169 条消息🔥🔥): 

> - `GPTs Agents 性能`
> - `OpenRouter 模型可用性`
> - `Mistral 图像识别问题`
> - `Llama 3.2 发布`
> - `OpenRouter API 速率限制` 


- **GPT4o Mini vs Gemini 1.5 Flash**：成员们测试了 **GPT4o Mini** 和 **Gemini 1.5 Flash**，指出 GPT4o Mini 表现尚可但未严格遵守约束条件，而 Flash 速度更快但输出结果不可靠。
   - 针对速度和对要求的遵循情况提出了担忧，特别是与 context size 相关的部分。
- **Mistral Pixtral 模型性能**：有成员报告 **mistralai/pixtral-12b** 存在幻觉且输出质量较差，而其他模型在图像识别方面表现良好。
   - 有建议指出，虽然 Pixtral 不是最好的，但 **Gemini Flash 模型** 在类似任务中更具性价比。
- **Llama 3.2 发布的热度**：即将发布的 **Llama 3.2** 公告中包含了更小的模型，旨在简化移动端和边缘设备上的开发部署。
   - 社区对 Llama 3.2 是否会很快在 **OpenRouter** 上线表示关注。
- **OpenRouter API 的速率限制**：讨论表明，用户在向 OpenRouter API 发送连续请求后遇到了 **rate limits**。
   - 澄清了速率限制与 credits 使用情况及请求频率挂钩。
- **翻译模型建议**：一位用户询问了最适合 **translation** 的模型，引发了关于模型大小、准确性和语言通用性之间权衡的讨论。
   - 建议包括根据具体的用例和需求测试不同的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在浏览器中。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.llama.com/llama3_2/use-policy/">Llama 3.2 Acceptable Use Policy</a>: Llama 3.2 可接受使用政策</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: 为模型消费转换数据</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">no title found</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://ollama.com/library/phind-codellama">phind-codellama</a>: 基于 Code Llama 的代码生成模型。</li><li><a href="https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp">Gemini Flash 8B 1.5 Experimental - API, Providers, Stats</a>: Gemini 1.5 Flash 8B Experimental 是 [Gemini 1. 的 8B 参数实验版本。通过 API 运行 Gemini Flash 8B 1.5 Experimental</li><li><a href="https://status.anthropic.com/incidents/rzbypbn1f7lf">Elevated Errors on Claude 3.5 Sonnet</a>: 未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1288238285948190844)** (2 条消息): 

> - `本地服务器支持`
> - `API 访问条件` 


- **支持本地服务器的挑战**：有人指出，在没有外部访问权限的情况下本地运行服务会使支持变得困难，这表明在不久的将来协助有限。
   - *如果你在本地运行，* 支持可能无法实现。
- **未来 API 支持的可能性**：对于可通过 HTTPS 访问并遵循带有 API key 的 **OpenAI-style schema** 的端点，未来可能会提供支持。
   - 如果满足这些标准，这为以后的潜在合作打开了大门。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1288213453193875619)** (110 messages🔥🔥): 

> - `Advanced Voice Mode Availability` (高级语音模式可用性)
> - `Meta AI Restrictions` (Meta AI 限制)
> - `Llama 3.2 Licensing` (Llama 3.2 许可)
> - `Game Development with AI` (使用 AI 进行游戏开发)
> - `AI Character Interactions` (AI 角色交互)


- **Advanced Voice Mode 分发争议**：成员们对 **Advanced Voice 模式** 的 **有限推送** 表示沮丧，尽管官方声称已向所有用户开放，但欧盟（EU）的许多用户仍无法访问。
   - 评论强调了不同地区功能可用性的差异，特别指出之前的 Memory 等功能在欧盟用户中也遭遇了类似的延迟。
- **Meta AI 的欧盟冲突**：讨论明确了由于多模态模型的许可限制，**Meta AI** 暂**不适用于**欧盟、英国和其他国家的用户。
   - 成员们指出 **Llama 3.2 的许可证** 明确与欧盟法规不兼容，限制了该地区开发者对其的访问。
- **Llama 3.2 模型特性**：**Llama 3.2** 的发布备受关注，重点在于其增强的多模态能力和改进的模型，但对欧盟用户仍有严格的许可限制。
   - 分享的细节包括该模型在 **Hugging Face** 上的发布及其技术规格，强调尽管存在许可障碍，它仍是该模型系列的一次飞跃。
- **用于游戏开发的 AI IDE 工具**：一位用户寻求关于游戏开发最佳 **AI IDE** 方案的建议，希望找到能够编写并执行代码进行审计的工具。
   - 建议包括 **Cursor** 等付费云服务以及社区驱动的工具，一些用户还利用 **ChatGPT** 进行文件系统修改。
- **AI 与包裹递送的未来**：一个引发思考的观点被提出，即利用 **脑机接口** 和 **加密邮寄地址** 来简化包裹递送，从而摆脱传统的地址系统。
   - 虽然构思引发了兴趣，但参与者指出，将想法转化为实际应用至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1fp5gut/molmo_a_family_of_open_stateoftheart_multimodal/">Reddit - Dive into anything</a>: 无描述</li><li><a href="https://huggingface.co/blog/llama32">Llama can now see and run on your device - welcome Llama 3.2</a>: 无描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1288228466143006865)** (8 messages🔥): 

> - `Voice Functionality with GPT` (GPT 的语音功能)
> - `Complex Tasks Handling by GPT` (GPT 处理复杂任务)
> - `Comparative Efficiency of AI Models` (AI 模型效率对比)


- **GPT 无法进行调用**：一位成员澄清说，ChatGPT 目前无法调用其他 GPTs，用户必须亲自执行操作。
   - 已提交一项建议，希望未来能允许此类功能。
- **复杂任务导致延迟**：一位用户对 GPT 无法按时完成复杂的书籍写作任务表示沮丧，仅收到“我正在处理”之类的反馈。
   - 在等待数天后，该用户担心 GPT 的响应无法产生令人满意的结果，在提示后仅交付了两页内容。
- **对 GPT 能力的怀疑**：一位成员对 GPT 在没有持续提示和调整的情况下处理复杂任务的能力表示怀疑。
   - 另一位成员建议，虽然 GPT 可能会感到吃力，但使用 **o1-preview** 和 **Claude** 等模型可能会获得更好的结果。
- **对替代 AI 模型的偏好**：一位用户提到，由于拥有更长的记忆窗口，他们发现 **Claude** 和 **o1-preview** 模型更适合处理复杂任务。
   - 他们分享了大量使用付费模型的经验，并表示对对比反馈感兴趣。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1288241293817807032)** (24 messages🔥): 

> - `Minecraft API 提示词编写`
> - `AI 中的反馈机制`
> - `论文评分挑战`
> - `模型输出的可变性`
> - `Prompt Engineering 最佳实践` 


- **优化 Minecraft API 提示词**：成员们讨论了如何优化 Minecraft API 的提示词，通过在查询中指定不同的相关主题和复杂度级别，来减少重复问题并提高多样性。
   - 建议包括引导 AI 避免重复之前的问题，并强制执行结构化的 JSON 输出格式。
- **论文评分的挑战**：一位成员寻求微调模型的方法，以便根据特定的评分标准对论文提供“直言不讳”的反馈，并指出模型回应过于宽大是一个主要问题。
   - 建议提供详细的评分细则（rubrics）和样本，同时指出模型的内在设计倾向于正向激励，这可能与严厉批评的需求相冲突。
- **提示词和模型引导的有效性**：讨论强调了在与 AI 交流时，使用精确且积极的语言可以产生更好的结果，尤其是在定义“直言不讳的反馈”等预期时。
   - 成员们注意到提示词清晰度的重要性，以及用户预期与模型所理解的“有帮助”之间可能存在的错位。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1288241293817807032)** (24 messages🔥): 

> - `针对 Minecraft 问题的 Prompt Engineering`
> - `反馈模型的挑战`
> - `为论文反馈微调模型` 


- **改进 Minecraft 问题提示词**：成员们讨论了增强生成独特且有趣的 Minecraft 相关问题的提示词的方法，建议增加更具体的问题主题，如机制和生物（mobs）。
   - 一位成员对输出的重复性表示沮丧，寻求关于如何修改提示词以确保多样性和一致性的建议。
- **AI 反馈系统的困境**：讨论了 AI 在论文反馈中过于宽大的倾向，并建议鼓励更直言不讳的批评。
   - 一位成员指出，模型被设计为支持性的，这可能与对写作进行严厉评估的期望相冲突。
- **为诚实的论文反馈进行微调**：关于微调模型是否能产生直言不讳的论文反馈进行了对话，成员们辩论了各种 Prompting 技术的有效性。
   - 建议包括提供带有评分标准的示例论文来引导 AI 的反馈，并强调模型的目标通常是建设性的，即使在追求诚实时也是如此。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1288219278213906543)** (145 messages🔥🔥): 

> - `Llama 3.2 模型`
> - `SillyTavern 集成`
> - `多模态能力`
> - `模型性能基准测试`
> - `对新模型的支持` 


- **Llama 3.2 发布引发关注**：最近发布的 **Llama 3.2**（包括 **1B** 和 **3B** 模型）引起了极大兴趣，特别是其在各种硬件配置上的运行速度。
   - 用户渴望对 **11B** 多模态模型的支持，但指出由于其视觉模型的特性，支持可能需要一些时间。
- **SillyTavern 的集成问题**：用户在使用 LM Studio 时遇到了 **SillyTavern 的集成问题**，主要与服务器通信和响应生成有关。
   - 故障排除建议 **SillyTavern** 可能需要特定的任务输入，而不是自由格式的文本提示。
- **对多模态模型能力的担忧**：讨论强调虽然 **Llama 3.2** 包含视觉模型，但用户在寻求类似于 **GPT-4** 的真正多模态能力。
   - 澄清指出 **Llama 3.2 的 11B 模型仅用于视觉任务，目前还不包含语音或视频功能。
- **模型性能基准测试**：**Llama 3.2** 的基准测试结果显示性能各异，**1B** 和 **3B** 分别获得了 **49.3%** 和 **63.4%** 的分数。
   - 与 **Qwen2.5** 模型的对比显示出类似的性能，表明不同语言模型之间具有竞争力的实力。
- **未来支持和开发计划**：对**新模型**的额外支持抱有期待，并正在讨论实施各种量化级别的可行性。
   - 随着技术的不断进步，用户对未来版本集成 **NPU 能力**和更快的推理速度表示乐观。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://program.pinokio.computer/#/?id=mac">program.pinokio</a>: Pinokio 编程手册</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/hugging-quants/llama-32-3b-and-1b-gguf-quants-66f43204a559009763c009a5">Llama 3.2 3B &amp; 1B GGUF Quants - hugging-quants 集合</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/SillyTavernAI/comments/1fnv8ts/comment/lopt8p5/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fp5gut/molmo_a_family_of_open_stateoftheart_m">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fp5gut/molmo_a_family_of_open_stateoftheart_multimodal/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpckrw/qwen25_selfreported_now_on_official_mmlupro/">Qwen2.5（自报数据）现已登上官方 MMLU-Pro 排行榜，超越 Gemini 1.5 Pro 和 Claude 3 Opus</a>: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro</li><li><a href="https://www.youtube.com/watch?v=uTFdl5s6Vb0">Forge 中的新 Flux。操作指南。Flux Img2Img + Inpainting</a>: Flux Forge 教程指南 https://www.patreon.com/posts/110007661 在我们的社区 Discord 中与我交流：https://discord.gg/dFB7zuXyFY 适合初学者的 Stable Diffusion...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/omaJSVPpXe">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/qGIUKYWeYe">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/THUDM/LongWriter">GitHub - THUDM/LongWriter: LongWriter: 释放长上下文 LLM 的 10,000+ 字生成能力</a>: LongWriter: 释放长上下文 LLM 的 10,000+ 字生成能力 - THUDM/LongWriter</li><li><a href="https://github.com/chigkim/Ollama-MMLU-Pro">GitHub - chigkim/Ollama-MMLU-Pro</a>: 通过在 GitHub 上创建账号来为 chigkim/Ollama-MMLU-Pro 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/scripts/gguf_new_metadata.py">llama.cpp/gguf-py/scripts/gguf_new_metadata.py at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama/pull/6963">pdevine 为 llama3.2 提供的图像处理 · Pull Request #6963 · ollama/ollama</a>: 用于运行 llama3.2 的图像处理例程。未来需要进行重构以支持其他多模态模型。</li><li><a href="https://colab.research.google.com/drive/1lW6aQW77NDttBQ2Mk5M_OZrp-ZjIaFEt>)">Google Colab</a>: 未找到描述</li><li><a href="https://x.com/omarsar0/status/1761037006505722340>)">elvis (@omarsar0) 的推文</a>: Gemma 7B Instruct 的提示词指南已上线！我开始记录一些如何为 Gemma 编写提示词的示例。根据一些测试，它感觉是一个有趣且能力很强的模型。思维链 (Chain-of-thought)...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1288226537254031514)** (17 条消息🔥): 

> - `科技产品价格差异`
> - `LM Studio 中的双 GPU 配置`
> - `RTX 3090 的性能`
> - `Vulkan 与混合 GPU 类型` 


- **价格差异引发不满**：用户对 **欧盟地区较高的科技产品价格** 表示沮丧，指出其价格有时可能比美国高出 **一倍**。
   - 一位成员表示“这不公平！”，而另一位成员则指出 **VAT**（增值税）是造成这种差异的主要原因。
- **LM Studio 支持双 GPU 配置**：有成员询问是否可以同时使用 RTX 4070ti 和 RTX 3080，得到的确认是，如果 GPU 类型相同，**LM Studio 支持双 GPU 配置**。
   - 其他人讨论了通过 **Vulkan** 使用不同类型 GPU 的潜力，认为这可能是一个不错的实验。
- **RTX 3090 的 TPS 预期**：关于 RTX 3090 的 **TPS**（每秒处理数）的讨论显示，在 Q4 8B 模型上预期性能约为 **60-70 TPS**。
   - 讨论中澄清了这种性能更适合 **inference training**（推理训练）而非简单的查询。
- **在 Vulkan 中尝试混合 GPU**：一位用户分享了同时使用 **RTX 2070 和 RX 7800** 配合 **Vulkan** 的经验，实现了总计 **24GB VRAM 来加载 LLM**。
   - 他们指出不同 LLM 的性能表现各异，当加载到两块 GPU 上时，有些运行较慢，而有些则较快。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/income-tax-tax-taxes-gif-11011288">Income Taxes GIF - Income Tax Tax Taxes - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f7n4d7/has_anyone_mixed_nvidia_and_amd/">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1288215589776654346)** (19 messages🔥): 

> - `yt-dlp 工具`
> - `PyTorch 训练属性问题`
> - `本地 LLM 基准测试套件`
> - `基础模型工程师招聘`
> - `控制扩散模型` 


- **介绍用于音视频下载的 yt-dlp**：一位成员分享了 [yt-dlp](https://github.com/yt-dlp/yt-dlp) 的 GitHub 链接，这是一个功能丰富的命令行音视频下载器，并建议将其作为现成的工具使用。
   - 虽然有人对恶意软件表示担忧，但指出源仓库看起来是安全的。
- **PyTorch 训练属性 Bug**：讨论了一个关于 PyTorch 的已知问题：调用 `.eval()` 或 `.train()` 不会改变 `torch.compile()` 模块的 `.training` 属性，详见 [此 GitHub issue](https://github.com/pytorch/pytorch/issues/132986)。
   - 成员们表达了对这类问题缺乏可见性的沮丧，并讨论了诸如修改 `mod.compile()` 等潜在的变通方法。
- **寻求本地 LLM 基准测试套件**：一位成员请求推荐用于本地 LLM 测试的开源基准测试套件，并提到了 MMLU 和 GSM8K 等特定基准。
   - 另一位成员提供了 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的链接，这是一个用于语言模型 few-shot 评估的框架。
- **招聘基础模型工程师**：一位成员正在推广基础模型工程师的职位空缺，并分享了社交媒体链接以增加互动。
   - 该成员鼓励在 LinkedIn 和 X 上转发或点赞他们的帖子，以提高曝光度。
- **控制扩散模型的挑战**：讨论集中在控制用于图像生成的潜在扩散模型（latent diffusion models）的困难上，引用了 Advex 关于保真度（fidelity）与忠实度（faithfulness）权衡的推文。
   - 他们提到了像 DALLE-3 和 ControlNet 这样在平衡这两者时遇到困难的现有模型，并强调了 Advex 的研究目标。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/qasim31wani/status/1839045889790456010">来自 Qasim Wani (@qasim31wani) 的推文</a>：控制 LDM（潜在扩散模型）很难。😓 能够精确控制想要生成的图像是图像生成的终极目标。许多论文已经从不同角度研究了这个问题……</li><li><a href="https://discuss.pytorch.org/t/training-property-of-complied-models-is-always-true/209992">已编译模型的 `training` 属性始终为 `True`</a>：我注意到我无法为已编译的模块将 training 属性设置为 False，无论编译时该属性的状态如何。这是预期行为吗？演示如下：...</li><li><a href="https://github.com/pytorch/pytorch/issues/132986">`.eval()` 和 `.train()` 在 `torch.compile()` 模块上无法正确设置 `.training` 的值 · Issue #132986 · pytorch/pytorch</a>：🐛 Bug 描述：调用 .eval() 或 .train() 不会改变 torch.compile() 模块的 .training 值，它只改变底层模块的值。复现代码：import tor...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness：一个用于语言模型 few-shot 评估的框架。</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/yt-dlp/yt-dlp">GitHub - yt-dlp/yt-dlp：一个功能丰富的命令行音视频下载器</a>：一个功能丰富的命令行音视频下载器 - yt-dlp/yt-dlp
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1288480557524717630)** (70 messages🔥🔥): 

> - `NAIV3 技术报告`
> - `BERT 掩码率`
> - `模型训练技术`
> - `众包数据集`
> - `SUNDAE 模型见解` 


- **NAIV3 技术报告发布**：[NAIV3 技术报告](https://arxiv.org/abs/2409.15997) 讨论了一个包含来自众包平台的 **600 万** 张图像的数据集，提高了图像打标（tagging）方法等细节的透明度。
   - 讨论内容包括是否根据之前项目的文档加入幽默元素（如皇冠图像）。
- **探索更高的 BERT 掩码率**：一位成员询问是否有人训练过掩码率超过 **15%** 的类 BERT 模型，随后引出的见解表明，更高的掩码率可以提高性能，特别是在大型模型中。
   - 另一项研究指出 **40%** 的掩码率优于 **15%**，并指出极高的掩码率仍能保持微调（fine-tuning）性能。
- **开源社区中的训练技术**：成员们对当前项目工作的感知简单性表示不满，暗示在社区反馈中**复杂性**被低估了。
   - 评论暗示对“新颖见解”的期望可能会削弱项目的认可度，担心微小的改进可能无法满足评审员。
- **关于众包数据使用的见解**：对话强调了利用来自 Danbooru 等平台的数据时需要谨慎措辞，以避免激怒潜在的批评者。
   - 提到针对特定图像格式训练实践的标准化，为学术讨论增加了专业发展的维度。
- **SUNDAE 训练方法的后续跟进**：关于 SUNDAE 模型的讨论揭示了其在 **0** 到 **1** 范围内使用随机掩码，尽管其主要关注点是机器翻译。
   - 有人询问 SUNDAE 的后续研究，指出由于兴趣转向文本扩散（text diffusion）技术，目前缺乏新的进展。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/lexin_zhou/status/1838961179936293098">来自 Lexin Zhou (@lexin_zhou) 的推文</a>: 1/ Nature 新论文！人类对任务难度的预期与 LLM 错误之间的差异损害了可靠性。在 2022 年，Ilya Sutskever @ilyasut 预测：&#34;也许随着时间的推移，这种差异会...</li><li><a href="https://arxiv.org/abs/2409.15997">NovelAI Diffusion V3 中对 SDXL 的改进</a>: 在这份技术报告中，我们记录了在训练 NovelAI Diffusion V3（我们最先进的动漫图像生成模型）过程中对 SDXL 所做的更改。</li><li><a href="https://arxiv.org/abs/2202.08005">在掩码语言建模中你应该掩码 15% 吗？</a>: 掩码语言模型 (MLMs) 习惯上掩码 15% 的 token，因为人们认为更多的掩码会导致缺乏足够的上下文来学习良好的表示；这种掩码率已被广泛使用...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1288260104659140690)** (9 messages🔥): 

> - `Decomposing models`
> - `Scaling laws datapoints`
> - `Broken neural scaling laws dataset`
> - `Google research findings` 


- **使用受 Strassen 启发的算法分解模型**：一名成员建议，可以尝试以 **Strassen-inspired** 的方式对模型进行**分解**，以减少加法和减法操作的数量。
   - 该方法可能允许模型在不具备完整复杂性的情况下，逼近与完整模型相似的结果。
- **寻求 Scaling Laws 的数据点**：有人请求提供关于 **scaling laws** 的 **datapoints** 来源，特别是针对使用不同参数和数据集大小训练的架构。
   - 该查询旨在测试在 Chinchilla 的 scaling laws 中加入缺失的低阶项是否会影响计算最优 (compute-optimal) 的选择。
- **使用 Broken Neural Scaling Laws 数据集**：一位成员推荐使用 **broken neural scaling laws dataset**，称其内容**详尽**但缺乏架构细节。
   - 由于缺乏架构细节，训练新模型并收集针对个人需求定制的数据点变得具有挑战性。
- **Google 研究论文引用**：讨论中引用了一篇 **Google research paper**，该论文在 GitHub 上提供了与神经网络 scaling laws 相关的结果，但主要包含原始数据点。
   - 成员们指出，尽管这些数据点正是所需要的，但目前还没有简单的方法来复制或扩展这些结果。



**Link mentioned**: <a href="https://github.com/google-research/google-research/blob/master/revisiting_neural_scaling_laws/README.md">google-research/revisiting_neural_scaling_laws/README.md at master · google-research/google-research</a>: Google Research. Contribute to google-research/google-research development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1288551611240808498)** (1 messages): 

> - `Chain-of-thought responses`
> - `Filler tokens efficacy`
> - `Human-like task decomposition`
> - `Transformers and algorithmic tasks` 


- **Chain-of-thought 响应提升性能**：最近的研究表明，语言模型的 **chain-of-thought 响应** 显著增强了在大多数基准测试中的表现，如论文 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2404.15758) 所述。
   - 然而，该研究质疑性能提升是源于**类人的任务分解 (human-like task decomposition)**，还是仅仅因为额外 Token 带来的计算量增加。
- **填充 Token 可以替代思维链**：实验表明，Transformer 可以利用**无意义的填充 Token (filler tokens)**（例如 '......'）来处理复杂的算法任务，而如果没有这些中间 Token，它们将无法解决这些任务。
   - 尽管如此，研究强调，教导模型使用这些填充 Token 需要*特定的密集监督 (dense supervision)* 才能实现有效的学习。
- **填充 Token 与量词深度**：该论文提供了一个理论框架，识别了一类 **filler tokens** 具有优势的问题，并将其有效性与**一阶公式的量词深度 (quantifier depth of first-order formulas)** 联系起来。
   - 在该框架内，研究表明对于某些问题，chain-of-thought Token 不需要传达有关中间计算的信息。
- **隐藏的思考嵌入仍存争议**：有人对 Token 文本背后是否存在“隐藏的思考嵌入 (hidden thinking embedding)”提出了疑问，表明其确定性尚不明确。
   - 这突显了社区中关于模型响应背后的机制以及**任务分解 (task decomposition)** 有效性的持续讨论。 



**Link mentioned**: <a href="https://arxiv.org/abs/2404.15758">Let&#39;s Think Dot by Dot: Hidden Computation in Transformer Language Models</a>: Chain-of-thought responses from language models improve performance across most benchmarks. However, it remains unclear to what extent these performance gains can be attributed to human-like task deco...

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1288256063719870464)** (40 messages🔥): 

> - `Pile model formatting issues` (Pile 模型格式化问题)
> - `lm_eval usage with OpenAI API` (在 OpenAI API 中使用 lm_eval)
> - `Aexams task evaluation` (Aexams 任务评估)
> - `Exact match metrics` (精确匹配指标)
> - `Debugging sequence length errors` (调试序列长度错误)


- **关于 Pile 模型格式化问题的参考**：有成员询问关于“Pile 模型在格式化方面表现糟糕”的相关参考文献，随后有人建议引用论文 [Lessons from the Trenches](https://arxiv.org/abs/2405.14782) 以获取相关数据。
   - 另一位成员确认，在该论文目前的表格中，ARC-Challenge 任务也能观察到相同的结果。
- **在 OpenAI API 中使用 lm_eval 进行纯文本测试**：一位用户询问如何在针对 OpenAI API 的 `lm_eval` 中运行不带预分词（pre-tokenization）的测试，另一位成员确认在使用 `--apply_chat_template` 时，`openai-chat-completions` 不会进行分词。
   - 对于 `openai-completions`，建议在 `--model_args` 中添加 `tokenized_requests=False` 作为解决方案。
- **评估 aexams 任务的问题**：一位成员表示，在使用 Claude 3.5 Sonnet 对 aexams 任务进行评估时，尽管子任务有结果，但 Groups 表格却是空的，并对此表示担忧。
   - 另一位成员建议应更新 `_aexams.yaml` 配置，使用 `exact_match` 聚合代替 `acc`。
- **调试 generate_until 中的序列长度错误**：一位成员在调试 harness 中的 `generate_until` 函数时，遇到了超过 `self.max_length` 的序列长度错误。
   - 他们怀疑问题可能源于重写了 `tok_batch_encode`，这可能导致了非预期的上下文编码结果。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2353).">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/1644">[BUG] Eval recipe not using max_seq_length · Issue #1644 · pytorch/torchtune</a>: 2024-09-21:20:19:56,843 INFO [_logging.py:101] 正在运行 EleutherEvalRecipe，解析配置为：batch_size: 1 checkpointer: _component_: torchtune.training.FullModelHFCheckpointer checkpoint_dir: ....</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/bc50a9aa6d6149f6702889b4ef341b83d0304f85/lm_eval/models/huggingface.py#L465)">lm-evaluation-harness/lm_eval/models/huggingface.py at bc50a9aa6d6149f6702889b4ef341b83d0304f85 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/bc50a9aa6d6149f6702889b4ef341b83d0304f85/lm_eval/models/huggingface.py#L1270-L1278)">lm-evaluation-harness/lm_eval/models/huggingface.py at bc50a9aa6d6149f6702889b4ef341b83d0304f85 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1288531371501031566)** (1 messages): 

> - `Non-casual masking` (非因果掩码)
> - `AR masks and positional information` (AR 掩码与位置信息)


- **AR 掩码辅助位置学习**：一位成员澄清说，该断言仅适用于**非因果掩码 (non-casual masking)**；使用 **AR 掩码** 能让模型获取**位置信息**。
   - 这种区别对于理解掩码如何影响模型训练和输入处理至关重要。
- **理解掩码效果**：另一位成员讨论了各种掩码技术对模型的总体影响，强调了明确定义的需求。
   - 他们指出，在比较因果 (casual) 和非因果 (non-casual) 掩码策略时经常会产生混淆。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1288213405169090684)** (123 条消息🔥🔥): 

> - `Perplexity AI 更新`
> - `ChatGPT 语音模式功能`
> - `O1 预览版访问权限`
> - `上下文保留问题`
> - `在学校使用 AI 工具` 


- **Perplexity AI 在上下文保留方面表现不佳**：用户对 Perplexity 无法保留后续问题的上下文表示沮丧，并指出这一问题最近变得更加频繁。
   - 几位成员讨论了他们的经历，认为该平台的性能可能有所下降，影响了其实用性。
- **Merlin.ai 提供带网络访问权限的 O1**：Merlin.ai 被推荐作为一个替代方案，因为它提供带网络访问的 O1 功能，且没有每日消息限制，仅有速率限制（rate limit）。
   - 用户表现出探索 Merlin 的兴趣，因为它比 Perplexity 具有更扩展的功能。
- **用户权衡用于教育的 AI 工具**：几位用户讨论了他们在学校相关任务中对 AI 工具的依赖，一些人青睐 GPT-4o，另一些人则考虑将 Claude 作为替代方案。
   - 反馈包括承认各种 AI 工具提供了不同程度的帮助，特别是在学术需求方面。
- **对 Perplexity 性能下降的担忧**：用户注意到 Perplexity 的性能有所下降，与以往相比，详细回复更少。
   - 用户对可能影响平台能力的潜在财务问题提出了担忧，并分享了对替代方案的看法。
- **Llama 3.2 发布新闻**：Llama 3.2 模型的发布引起了用户的兴奋，提到了它的可用性和功能。
   - 然而，一些用户注意到 Perplexity 上缺少 Llama 3.2 的信息，表明可用资源中可能存在缺口。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/sir-sir-sir-give-me-that-get-me-that-pervert-gif-20490657">Sir Sir Sir GIF - Sir Sir Sir Give Me That - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/T5N4ZSlmh9k">Perplexity Full Tutorial: WILD Ai Research Tool! (A-Z Guide)</a>：⚡加入 Ai Foundations：https://swiy.co/aif-0 在今天的视频中，我将教你关于 Perplexity 需要了解的一切。这个 AI 工具带来了...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1288242017343897680)** (6 条消息): 

> - `空气炸锅是否值得`
> - `4 天健身计划`
> - `Jony Ive 的 OpenAI 项目`
> - `南极洲臭氧层空洞`
> - `时尚行业研究` 


- **评估空气炸锅**：一位用户分享了一个讨论[空气炸锅是否值得](https://www.perplexity.ai/search/are-air-fryers-worth-it-5Ylk154lSZyKHan.UxR2UA)的链接，重点关注其优缺点。
   - 关键点包括与传统油炸方法相比的健康益处以及烹饪效率。
- **针对初学者的 4 天健身计划**：一位成员提供了一个针对健身新手的 [4 天健身计划](https://www.perplexity.ai/page/4-day-workout-plan-for-beginne-dAahwrI0SPC46B2CAGI1MQ)链接。
   - 该计划侧重于建立稳固的常规训练，同时对新手来说易于管理。
- **Jony Ive 的 OpenAI 项目讨论**：一段名为 *YouTube* 的精彩视频涵盖了 [Jony Ive 的 OpenAI 项目](https://www.youtube.com/embed/8bGKu8UVvcM)，详细介绍了设计与 AI 的创新交集。
   - 视频还涉及了南极洲臭氧层空洞和古老宇宙信号等显著话题。
- **调查时尚行业趋势**：一位用户寻求关于[时尚行业](https://www.perplexity.ai/search/can-you-explain-the-total-amou-YyYhBQLLQIWSVTrz4.jLbQ)的信息，引发了关于当前趋势和可持续性的讨论。
   - 核心讨论围绕该行业对环境的影响及其不断演变的实践。
- **板载焊接 RAM 过时解释**：围绕[板载焊接 RAM 的过时性](https://www.perplexity.ai/page/soldered-ram-obsolescence-stra-xib3eRUHQxGZ7ujeAJR9Gg)展开了讨论，分析了其对系统升级的影响。
   - 参与者对设备在 RAM 变得越来越不可升级的情况下的未来适用性（future-proofing）表示担忧。



**提到的链接**：<a href="https://www.youtube.com/embed/8bGKu8UVvcM">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1288399114161225789)** (9 条消息🔥): 

> - `Wolfram Alpha 集成`
> - `Perplexity API 能力`
> - `Webhook Zap 结构`
> - `脚本化 Web 界面使用`
> - `外部模型的可用性` 


- **Wolfram Alpha 与 Perplexity API 的集成**：一位用户询问是否可以像在 Web 端应用中那样，在 **Perplexity API** 中使用 **Wolfram Alpha**。
   - 另一位成员确认目前**无法实现**，并强调了 API 与 Web 界面之间的独立性。
- **数学和科学问题的 API 与 Web 界面对比**：有人提问 **API** 在解决数学和科学问题时是否与 **Web 界面** 一样强大。
   - 在围绕该能力的讨论中，没有提供确切的答案。
- **Webhook Zap 结构的澄清**：一位在 **Zapier** Webhook 中使用 **Perplexity** 的用户寻求关于其结构的澄清，特别是关于 System 和 User 内容的部分。
   - 他们提出 System 内容是给 AI 的指令，而 User 内容是提供给 AI 的输入。
- **外部模型的可用性**：一位用户注意到外部模型似乎只能通过 **Web 界面** 访问，而不能通过 API 访问。
   - 他们询问了如何以 **Perplexity** 支持的方式对 Web 界面进行脚本化使用，但未收到任何回复。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1288237554813894697)** (32 条消息🔥): 

> - `Anthropic Financials` (Anthropic 财务状况)
> - `OpenAI Training Data Access` (OpenAI 训练数据访问)
> - `Molmo Model Performance` (Molmo 模型性能)
> - `Molmo Model Capabilities` (Molmo 模型能力)
> - `Molmo License Changes` (Molmo 许可证变更)


- **Anthropic 预计营收将突破 10 亿美元**：据 [CNBC](https://x.com/tanayj/status/1838623375167574421?s=46) 报道，**Anthropic** 今年有望实现 **10 亿美元**的营收，标志着惊人的 **1000% 同比增长**。
   - 营收细分显示：**60-75%** 来自第三方 API，**10-25%** 来自直接 API，**15%** 来自 Claude 订阅，另有 **2%** 的小份额来自专业服务。
- **OpenAI 提供训练数据访问权限**：OpenAI 将首次提供其[训练数据的审查权限](https://x.com/morqon/status/1838891125492355280?s=46)，以检查是否使用了任何受版权保护的作品。
   - 访问将在 OpenAI 旧金山总部的一台没有互联网或网络连接的安全计算机上提供，这在社区中引发了褒贬不一的反应。
- **对 Molmo 模型能力的兴奋**：新的 **Molmo 模型**引发了热议，一位成员评论称其 **pointing feature**（指向功能）是他们近期见过的最令人兴奋的 AI 能力之一，并声称这比更高的 AIME 分数更具影响力。
   - 该模型在各种对比中的表现显示其确实令人印象深刻，反馈表明它“通过了氛围感测试 (passes the vibe check)”。
- **关于 Molmo 模型基准测试的讨论**：在将最大的 **Molmo 模型** (72B) 与 **Llama 3.2 V 90B** 进行对比时，成员们分享了具体的基准测试结果，表明 Molmo 在 AI2D 和 ChatQA 等多个领域具有优势。
   - 总体而言，讨论指向了有利的指标，一位成员幽默地表示，尽管他们在开发过程中参与有限，但对模型如此优秀的表现感到惊讶。
- **关于 Molmo 许可证变更的担忧**：一位成员对 **3.2 许可证**与 **3.1** 相比的重大差异提出了疑问，促使其他人对这些变更进行分析。
   - 他们分享了一个 [diff checker 链接](https://www.diffchecker.com/O4ijl7QY/)，以便对许可文本进行更深入的检查。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下说明。</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917">来自 Colin Fraser (@colin_fraser) 的推文</a>：我从未如此感到被证明是正确的。引用 Colin Fraser (@colin_fraser)：如果它实际上看起来像这样呢？</li><li><a href="https://x.com/osanseviero/status/1838939324651299235?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：@allenai 推出的 Molmo - 一个 SOTA 多模态模型 🤗开放模型和部分开放数据 🤏7B 和 72B 模型尺寸（+ 带有 1B 激活参数的 7B MoE） 🤯基准测试高于 GPT-4V, Flash 等 🗣️人类偏好...</li><li><a href="https://x.com/tanayj/status/1838623375167574421?s=46">来自 Tanay Jaipuria (@tanayj) 的推文</a>：据 CNBC 报道，Anthropic 今年营收预计将达到 10 亿美元，同比增长约 1000%。营收细分为：• 第三方 API（通过 Amazon 等）：60-75% • 直接 API：10-25% • Claude 聊天...</li><li><a href="https://x.com/andersonbcdefg/status/1839030313659564424">来自 Ben (e/treats) (@andersonbcdefg) 的推文</a>：我已经很久没有像对 Molmo 的指向功能那样对新的 AI 模型能力感到如此兴奋了。对于想要构建产品（而不是神）的人来说，我认为这可能更具影响力...</li><li><a href="https://x.com/morqon/status/1838891125492355280?s=46">来自 morgan — (@morqon) 的推文</a>：“OpenAI 将首次提供其训练数据的访问权限，以审查是否使用了受版权保护的作品”</li><li><a href="https://x.com/morqon/status/1838891975841366437?s=46">来自 morgan — (@morqon) 的推文</a>：“训练数据集将在 OpenAI 旧金山办公室的一台没有互联网或网络连接的安全计算机上提供”</li><li><a href="https://x.com/natolambert/status/1838991810846502938">来自 Nathan Lambert (@natolambert) 的推文</a>：将最大的 Molmo 模型 (72B) 与 Llama 3.2 V 90B 进行比较。MMMU，Llama 高出 6 个百分点；MathVista，Molmo 高出 1 个百分点；ChatQA Molmo 高出 2 个百分点；AI2D Molmo 高出 4 个百分点；DocVQA 高出 3 个百分点；VQAv2 大致持平或 Molmo 更好...</li><li><a href="https://www.diffchecker.com/O4ijl7QY/">llama 3.2 vs 3.1 - Diffchecker</a>：llama 3.2 vs 3.1 - LLAMA 3.1 社区许可协议 Llama 3.1 版本发布日期：2024 年 7 月 23 日。“协议”指...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1288213939502584024)** (15 messages🔥): 

> - `Twitter 与博客`
> - `OpenAI 最近的公告`
> - `Hugging Face 集合`
> - `视觉语言模型中的 Late Fusion`
> - `试用 GPT-4` 


- **Twitter 与博客认可**：一位成员认可了他们的 Twitter 和博客以一种非讽刺的方式呈现，强调了他们对这些平台的使用。
   - 他们还暗示了个人在组织一场讲座方面的贡献，但未提供更多细节。
- **OpenAI 的语音公告引发讨论**：一位成员指出，OpenAI 最近的语音公告引发了对除了单纯更新之外潜在模型发布的思考，并参考了更复杂的系统。
   - 他们建议在 Interconnects Artifacts、Models、数据集以及提到的系统之间建立潜在联系。
- **Hugging Face 集合概览**：分享了关于该组织 Hugging Face 集合的细节，特别是关于 **2024 Interconnects Artifacts**。
   - 该集合包含诸如 **argilla/notux-8x7b-v1** 之类的模型，最近更新于 3 月 4 日。
- **调研 Late Fusion 视觉 LM**：有成员提出了关于 Late Fusion 视觉语言模型在文本基准测试上的表现，以及与 Tulu 方案相比潜在收益的问题。
   - 该成员对带有视觉输入的模型性能可能出现的退化表示好奇。
- **试用 GPT-4 的图像输入**：一位成员分享了他们对 **GPT-4** 的实验，测试了在有和没有图像输入的情况下的表现，以分析其性能。
   - 他们观察到，当提供图像时，模型会路由到不同的模型，并思考了其相对智能程度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing">[2024年4月18日] 对齐开源语言模型</a>：对齐开源语言模型 Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS25: Transformers United V4</li><li><a href="https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988">2024 Interconnects Artifacts - natolambert 集合</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1288586685629857943)** (24 messages🔥): 

> - `OpenAI 团队变动`
> - `AI 公司的纷争`
> - `Anthropic 的立场`
> - `投资者信心`
> - `OpenAI 最新动态` 


- **OpenAI 的人才流失引起关注**：成员们指出，除了 **Sam Altman** 之外，几乎所有的 **OpenAI 创始团队** 都已离开，引发了关于这次大规模离职影响的讨论。
   - *一位成员断言，没有比这更强烈的信号了*，暗示了组织内部的不稳定性。
- **对领导层适配度的担忧**：一位成员质疑某位知名领导人最近的离职是否是一个**负面信号**，暗示她可能并不适合该职位。
   - 另一位成员指出，虽然她的离开令人担忧，但 OpenAI 在**模型开发和投资者支持**方面仍处于领先地位。
- **纷争引发猜测**：围绕 OpenAI 的持续纷争似乎成为了焦点，讨论暗示该公司策划了一些事件来掩盖像 **Molmo** 这样的竞争对手。
   - *一位成员评论说这一切感觉多么不靠谱 (sketchy)*，反映了对 OpenAI 现状的广泛担忧。
- **跳槽到 Anthropic 的可能性**：关于前 OpenAI 领导人加入 **Anthropic** 是否会进一步改变行业格局的猜测浮出水面，有人表示这将意味着重大颠覆。
   - 另一位成员将 Anthropic 的严肃态度与 OpenAI 的纷争进行了对比，暗示前者的运作使命更加明确。
- **动荡中 Greg 面临的挑战**：OpenAI 周围的氛围被描述为**对 Greg 来说非常残酷**，这可能表明领导层在动荡中面临的挑战。
   - 这一观点得到了多位成员的回应，他们对公司目前的发展轨迹表示怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vxtwitter.com/Yuchenj_UW/status/1839030011376054454">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下便条。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1288214160424960031)** (22 条消息🔥): 

> - `UV Python 工具增强`
> - `FTC 宣布打击 AI 诈骗`
> - `脚本中的数据依赖`
> - `UV 与 Docker 的集成`
> - `LLaMA Stack 讨论` 


- **UV Python 工具增强包管理**：一位用户建议开始使用 `uv pip install XY` 来获得**极速提升**，特别建议使用 `pyproject.toml` 管理依赖，并使用 `uv run myfile.py` 执行脚本。
   - *“这简直是游戏规则改变者”*，新支持的内联脚本依赖显著增强了易用性。
- **FTC 打击 AI 诈骗**：FTC 宣布了一项与 AI 相关的打击行动，由于其对什么是*真正的 AI Inside*标准模糊，引发了人们的关注。
   - 正如一位评论者所指出的，*“到底什么是 AI？我不知道，但 FCC 会在起诉你之后告诉你！”*
- **UV 的 Docker 集成技巧**：关于在 Docker 中使用 UV，一位用户参考了 [文档](https://docs.astral.sh/uv/guides/integration/docker/#caching) 并提到了一个展示最佳实践的 [GitHub 示例](https://github.com/astral-sh/uv-docker-example)。
   - 现有的 Docker 镜像（包括 **distroless** 和 **alpine** 变体）通过运行 `docker run ghcr.io/astral-sh/uv --help` 简化了命令执行。
- **关于 LLaMA Stack 重要性的讨论**：一位用户询问了 “LLaMA Stack” 相对于近期更新的重要性，推测它可能只是工具集成。
   - 另一位成员表示赞同，根据他们的发现，他们觉得这 *“在我看来只是集成。不重要”*。
- **用户对 PyCharm 的 UV 集成看法**：由于 PyCharm 处理虚拟环境的方式，用户对 UV 与 PyCharm 的功能兼容性表示担忧，促使一位用户分享了一个 [GitHub 仓库](https://github.com/InSyncWithFoo/ryecharm) 作为潜在的解决方案。
   - 尽管集成过程有些笨拙，但体验总结为 *“uv 的 UX 非常出色”*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1838642453391511892">来自 OpenAI (@OpenAI) 的推文</a>：Advanced Voice 尚未在欧盟、英国、瑞士、冰岛、挪威和列支敦士登提供。</li><li><a href="https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>.">运行脚本 | uv</a>：未找到描述</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917?s=46">来自 Colin Fraser (@colin_fraser) 的推文</a>：我从未感到如此被证明是正确的。引用 Colin Fraser (@colin_fraser) —— 如果它实际上长这样呢？</li><li><a href="https://docs.astral.sh/uv/guides/integration/docker/#caching>)">Docker | uv</a>：未找到描述</li><li><a href="https://fxtwitter.com/cdolan92/status/1839024340689371356">来自 Charlie Dolan (@cdolan92) 的推文</a>：FTC 宣布了与 AI 相关的打击行动。热门观点：好！有很多诈骗。进一步阅读后：什么鬼！？例如：这到底是什么意思？“……我们的技术人员可以弄清楚 [如果你的……”</li><li><a href="https://x.com/hanchunglee/status/1838793147163513190?s=46">来自 Han (@HanchungLee) 的推文</a>：o1-preview 推理能力：1. 在 9.11 与 9.8 的比较中，准确率收敛至约 85%。2. 并非 100% 遵循指令。3. 推理输出约为 10 TPS。4. 当 max_completion_token < reason... 时返回空响应。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1288569842462494752)** (8 条消息🔥): 

> - `Artificial General Intelligence (AGI)`
> - `社交媒体影响`
> - `Xeophon 语录`
> - `变革性机遇`
> - `公众反应` 


- **Leopold Aschenbrenner 对 AGI 的预测**：Leopold Aschenbrenner 的《SITUATIONAL AWARENESS》预测，我们正处于 2027 年实现 **Artificial General Intelligence (AGI)** 的轨道上，随后不久将实现超级智能，这带来了重大的机遇与风险。Ivanka Trump 分享的一条 [推文](https://x.com/IvankaTrump/status/1839002887600370145) 强调了这一点。
   - 该消息强调了理解这些技术能力潜在转变的重要性。
- **关于推文和粉丝的辩论**：关于是否要追求与 Interconnects 主题相关的推文和粉丝，进行了一场幽默的对话，其中一名成员质疑这是否真的是大家想要的。Xeophon 带着怀疑回应道：“**你真的想要那些推文和粉丝吗？**”。
   - 这种轻松的互动引发了对社交媒体参与价值的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/IvankaTrump/status/1839002887600370145">来自 Ivanka Trump (@IvankaTrump) 的推文</a>：Leopold Aschenbrenner 的 SITUATIONAL AWARENESS 预测我们正处于 2027 年实现 Artificial General Intelligence (AGI) 的轨道上，随后不久将实现超级智能，带来了变革性的机遇...</li><li><a href="https://x.com/unccrypto/status/1839010928937021643?s=46">来自 Crypto Unc (@UncCrypto) 的推文</a>：引用 Xeophon (@TheXeophon) 确实彻底完了
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1288553546408529931)** (2 条消息): 

> - `Reinforcement Learning`
> - `Curriculum Learning`
> - `演示数据`
> - `机器人学中的仿真` 


- **通过 Curriculum Learning 增强 RL**：最近的研究建议使用 **curriculum learning**，通过利用来自先前任务的离线演示数据来提高 **Reinforcement Learning (RL)** 的效率，这可以缓解复杂环境中的探索问题。他们的方法由 **reverse curriculum**（逆向课程）和随后的 **forward curriculum**（正向课程）组成，从而在窄状态分布上训练出有效的策略。
   - 这些方法与 **DeepMind** 的 **Demostart** 方法论的类似工作进行了比较，强调了获取高质量演示数据的挑战，特别是在 **robotics** 领域。
- **Curriculum Learning 的利基应用**：**Curriculum learning** 的效率归功于其利用 **演示数据** 并将环境重置为先前见过的状态的能力，从而优化了训练过程。然而，对 **simulation**（仿真）的依赖意味着现实世界的应用受到限制，这可能会阻碍一些研究人员的参与。
   - 这种利基需求既突显了 **Reinforcement Learning** 策略的创新，也突显了仿真环境必要性带来的固有挑战。



**提到的链接**：<a href="https://arxiv.org/abs/2405.03379">Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in Reinforcement Learning</a>：**Reinforcement Learning (RL)** 提供了一个通过环境交互学习策略的有前景的框架，但通常需要极其大量的交互数据来解决来自 sp... 的复杂任务。

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1288527907207118954)** (25 条消息🔥): 

> - `Llama 3.2 Release`
> - `Multimodal Models`
> - `Molmo Model Performance`
> - `Meta's Edge Device Integration`
> - `Community Reactions` 


- **Llama 3.2 正式发布**：Llama 3.2 已经发布，包含 **1B, 3B, 11B,** 和 **90B** 多个模型尺寸，旨在增强文本和 Multimodal 能力。
   - 初步反应表明它在细节上可能还有待完善，但现在发布总比在更繁忙的时段发布要好。
- **Multimodal 模型颠覆格局**：成员们注意到，一个新的 Multimodal 模型 **Molmo** 据报道在某些 Benchmark 上表现优于 **Llama 3.2 90B**，突显了竞争性的进步。
   - 评论建议向 **Llama** 和 **Molmo 团队** 表示祝贺，因为他们提供了令人印象深刻的产品。
- **Meta 为 Edge 设备集成 Llama**：Meta 推出 **Llama 3.2**，适用于 Edge 设备，并获得了 **Arm** 和 **Qualcomm** 等主要合作伙伴的支持，展示了其 On-device 能力。
   - 该公告强调了在各种平台和合作伙伴关系中扩大 **Open-source AI** 使用的意图。
- **社区对 Llama 就绪情况的热议**：讨论显示了对 **Llama 3.2** 立即部署的复杂情绪，一些人猜测其发布时机策略，同时热切期待官方更新。
   - 一位成员提到有 **3.5 小时的会议**，开玩笑地在繁忙的日程和对新发布的兴奋之间寻找平衡。
- **暗示和删除的推文引发猜测**：Hugging Face 成员暗示了关于 Llama 的潜在更新，特别是 **Julien Chaumond**，他的推文随后被删除，增加了神秘感。
   - 成员们积极刷新 **Llama** 的网站，希望能获得新信息，展现了一个专注且投入的社区。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1838988924259025040?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>: 他们似乎删除了预发布内容 😅 - 几分钟前它被上传到了 https://www.llama.com/ 和 https://ai.meta.com/ (一直在刷新 哈哈)</li><li><a href="https://x.com/danielhanchen/status/1838987356810199153?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>: Llama 3.2 Multimodal 来了！模型尺寸从 1B, 3B 到 11B 和 90B！</li><li><a href="https://x.com/aiatmeta/status/1838993953502515702?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>: 📣 隆重推出 Llama 3.2：适用于 Edge 设备、Vision 模型等的轻量级模型！有什么新变化？ • Llama 3.2 1B 和 3B 模型为多款 On-device 设备提供了同类产品中领先的能力...</li><li><a href="https://fxtwitter.com/_xjdr/status/1838993256925061342?s=46">来自 xjdr (@_xjdr) 的推文</a>: 有趣的背景。哇，祝贺 Llama 和 Molmo 团队</li><li><a href="https://x.com/andrewcurran_/status/1838992493066789254?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>: 这个小小的 1B 很强。扎克伯格先生在舞台上说，他们正在新款眼镜中嵌入本地模型，我猜可能就是这个。</li><li><a href="https://x.com/altryne/status/1838945025818062910?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 我们这里有什么？正值 Meta 即将颠覆 Multimodal 模型之际（即将报道！），@allen_ai 的优秀团队发布了 Multimodal MOLMO！一天内发布两个 SOTA Vision 模型？？ > 凭借...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1288241261387710595)** (22 条消息🔥): 

> - `Diffusion Transformer Policy for Imitation Learning`
> - `Torch Profiler File Size Issues`
> - `Pytorch Kernel Performance Improvement`
> - `Understanding torch.float8_e4m3fn`
> - `Embedding Data Loading from GPU` 


- **探索结合 Diffusion Transformers 的 SAM2-fast**：讨论集中在将 **SAM2-fast** 作为 ***Diffusion Transformer Policy*** 的输入用于 **imitation learning**，特别是将摄像头传感器数据映射到机器人手臂关节位置。
   - *图像/视频分割标注*被认为是该方法的一个潜在需求。
- **Torch Profiler 生成巨型文件**：***Torch profiler*** 正在生成过大的文件（高达 **7GB**），导致无法在 Chrome Tracer 中加载，引发了关于如何减小文件体积的讨论。
   - 建议包括仅对必要项进行 profiling，并将 trace 导出为 **.json.gz** 以压缩输出。
- **提升 Pytorch Kernel 性能**：一位用户寻求在不编写 ***CUDA*** kernel 的情况下提升 PyTorch 中 kernel 性能的建议，并分享了一个相关的**讨论链接**。
   - 成员们提供了提升性能的一般策略，尽管具体细节并未详尽展开。
- **澄清 torch.float8_e4m3fn 术语**：关于 ***torch.float8_e4m3fn*** 中 `fn` 的含义以及 `torch.finfo(torch.float8_e4m3fn).max` 为 **448** 的原因提出了疑问。
   - 对话澄清了 `0 1111 111` 代表 NaN，从而解释了最大值的限制。
- **理解 GPU 中 float8 的资源**：一位用户分享了一个 GitHub 资源链接，该资源提供了关于 ***float8*** 表示及其在机器学习算子中使用的详细信息。
   - 这为那些寻求澄清该主题（特别是关于 **fp8** 及其规范）的人提供了一个参考。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discuss.pytorch.org/t/improving-pytorch-kernel-performance/210020">Improving Pytorch kernel performance</a>: My code (for a somewhat degenerate matmul) looks like this:  def opt_foo(out, i1, i2, j1, j2, idx):     out[i1*512:i2*512].view(i2-i1, 512).add_(torch.mm(A[idx:idx+(i2-i1)*(j2-j1)].view(j2-j1,i2-i1).t...</li><li><a href="https://discuss.pytorch.org/t/training-property-of-complied-models-is-always-true/209992">`training` property of complied models is always `True`</a>: 我注意到我无法为已编译的模块将 training 属性设置为 False，无论编译时该属性的状态如何。这是预期行为吗？演示如下：...</li><li><a href="https://github.com/openxla/stablehlo/blob/main/rfcs/20230321-fp8_fnuz.md">stablehlo/rfcs/20230321-fp8_fnuz.md at main · openxla/stablehlo</a>: 受 HLO/MHLO 启发的向后兼容机器学习计算算子集 - openxla/stablehlo
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1288224449891405914)** (17 条消息🔥): 

> - `新课程发布`
> - `Pranav Marla 的 O1 替代方案`
> - `NVIDIA ASR 模型优化`
> - `Llama 3.2 发布公告`
> - `本地 AI 模型性能` 


- **新课程刚刚上线！**：“很棒；是我开发的”——成员们确认今天发布的新课程非常成功。
   - 一位成员在听到正面反馈后表示有兴趣去看看。
- **Pranav Marla 的 O1 替代方案表现出色**：查看 [Pranav Marla 的 O1 替代方案](https://x.com/pranavmarla/status/1838590157265539307)，它是**透明、自修复**的，并且在推理问题上表现得惊人地好。
   - 该模型设计了*无限递归*和 **Python 解释器**等功能，在各种应用中极具前景。
- **使用 NVIDIA 优化 ASR 模型**：一篇有趣的博客文章讨论了 [NVIDIA 的 ASR 模型](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)，这些模型在速度和准确性方面树立了基准。
   - 一个关键点是*自动混合精度*在推理过程中的表现未达预期，这引起了 ML 研究人员的关注。
- **Meta 发布 Llama 3.2 模型**：Meta 宣布推出 [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)，其特点是专为边缘和移动设备设计的较小视觉和文本模型。
   - 这些模型包括 **11B 和 90B** 视觉 LLM 以及 **1B 和 3B** 文本模型，旨在为开发者扩大可访问性。
- **讨论本地 AI 模型**：有建议认为在本地运行像 **Phi** 这样的模型可能更实用，特别是采用特定的微调方法。
   - 讨论了使用模型的蒸馏版本，强调了提升本地 AI 性能的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/pranavmarla/status/1838590157265539307">来自 trees of thought (@pranavmarla) 的推文</a>：我构建了一个 O1 替代方案，它是：1) 完全透明，视觉可追踪 2) 无限递归 3) 自修复，每一步都有测试 4) 能够使用 Python 解释器。它的表现...</li><li><a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">未找到标题</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/">使用 NVIDIA NeMo 将排行榜顶尖的 ASR 模型加速 10 倍 | NVIDIA 技术博客</a>：NVIDIA NeMo 一直在开发自动语音识别 (ASR) 模型，这些模型树立了行业基准，特别是那些在 Hugging Face Open ASR 排行榜上名列前茅的模型。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1288253857121570827)** (2 条消息): 

> - `Luma 工程岗位`
> - `Meta 社区项目` 


- **Luma 寻求顶尖工程师进行优化**：Luma 正在寻找优秀的工程师来负责其 [Dream Machine](https://lumalabs.ai/dream-machine) 和其他多模态基础模型的性能优化，并为实地办公的候选人提供签证赞助。
   - 他们要求具备大规模分布式训练、底层内核 (kernels) 以及优化分布式推理工作负载的经验，强调快速交付和极简的官僚主义。
- **加入 Meta 参与激动人心的社区项目**：Meta 邀请申请参与与 PyTorch 相关的社区项目，包括 GPU MODE 和 NeurIPS LLM 效率竞赛，强调该职位的自由度和灵活性。
   - 感兴趣的候选人可以通过 [Meta 招聘页面](https://www.metacareers.com/jobs/537331065442341/)进行申请，并鼓励就该职位提出任何问题。



**提到的链接**: <a href="https://www.metacareers.com/jobs/537331065442341/">软件工程师，系统 ML - PyTorch 性能与参与</a>：Meta 的使命是赋予人们建立社区的力量，让世界联系更紧密。齐心协力，我们可以帮助人们建立更强大的社区——加入我们。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1288269278528929933)** (12 messages🔥): 

> - `CUDA code in Python`
> - `Custom Ops in PyTorch`
> - `Beginner Projects` 


- **轻松移植 CUDA 代码**：为了在 Python 中封装独立的 **CUDA code**，成员们建议使用 [PyTorch 的 load_inline](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline)，正如第一课中所讨论的那样。
   - 另一位成员提到，虽然这种方法很直接，但自定义 **ops** 才是集成的“最佳实践”解决方案。
- **无 Graph Break 的自定义算子**：讨论显示，你可以在 PyTorch 中集成自定义 **kernels** 而不产生 **graph break**，从而简化开发。
   - 然而，成员们承认 **packaging process**（打包过程）可能有点烦人，希望提供的资源能让它变得更容易。
- **寻求初学者项目建议**：一位成员表示希望获得一份 **beginner-friendly projects**（初学者友好项目）清单，寻求视频课程中演示的算法之外的选项。
   - 他们指出，工作组中的项目对于新人来说似乎有些**令人生畏**。
- **定期活动安排**：一位成员更新道，周五的活动已排定，并表示目前正在**联系演讲者**。
   - 活动选项卡将随着新会议的安排而更新，方便成员跟踪。
- **文章讨论频道**：一位成员告知，已创建一个新的频道用于文章讨论，以保持其他频道专注于对话。
   - 另一位成员表示感谢，并同意在适当的频道中**重新发布他们的内容**。



**提及的链接**：<a href="https://github.com/pytorch/ao/tree/main/torchao/csrc">ao/torchao/csrc at main · pytorch/ao</a>：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288313223895978044)** (1 messages): 

> - `RoPE cache in FP32`
> - `Torchao Llama` 


- **RoPE 缓存应始终为 FP32**：一位成员就 **Torchao Llama** 模型中的 **RoPE cache** 提出了疑问，建议它应该始终采用 **FP32** 格式。
   - 他们引用了仓库中[特定的代码行](https://github.com/pytorch/ao/blob/7dff17a0e6880cdbeed1a14f92846fac33717b75/torchao/_models/llama/model.py#L186-L192)，这可能有助于澄清原因。
- **关于 Torchao Llama 代码行为的讨论**：对话包含了关于 **Torchao Llama 代码**行为的细节，特别是 **RoPE cache** 的实现。
   - 参与者对缓存机制中精度的处理表示关注，强调了其对性能的重要性。



**提及的链接**：<a href="https://github.com/pytorch/ao/blob/7dff17a0e6880cdbeed1a14f92846fac33717b75/torchao/_models/llama/model.py#L186-L192">ao/torchao/_models/llama/model.py at 7dff17a0e6880cdbeed1a14f92846fac33717b75 · pytorch/ao</a>：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1288225030454513815)** (16 messages🔥): 

> - `GPU Mode Name Ideas`
> - `Lambda Labs Credits`
> - `Mascot Suggestions` 


- **GPU Mode 的创意命名**：关于 GPU mode 的潜在名称展开了讨论，提出了 **Accelerator Mode** 和 **Accel Mode** 等想法，同时也对 GPU mode 这个名字表示了赞赏。
   - 有人建议将 *Goku*（悟空）作为吉祥物示例，突显了有趣的社区精神。
- **关于 Lambda Labs 额度的疑问**：一位成员询问了 Lambda Labs 为线下参与者提供的 **$300 credits** 状态，注意到他们的账户中没有显示。
   - 其他人指出 Lambda 额度是一次性的，但提到像 **Prime Intellect** 和 **Modal** 这样的服务应该仍会显示额度活动。
- **关于 Lambda 和 Prime Intellect 易用性的反馈**：成员们分享了他们使用 Lambda Labs 的经验，其中一位指出它是最容易使用的，尤其是从 **3070** 显卡过渡过来时。
   - 有人指出 **Prime Intellect** 运行在 Lambda 之上，使其成为一个无缝的过渡选项。


  

---

### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1288237546698182707)** (3 条消息): 

> - `CUDA puzzles`
> - `PMP 书籍中的 Machine Learning` 


- **寻找 CUDA Puzzles**: *Locknit3* 询问是否有类似于 **Triton Puzzles** 但专门针对原生 **CUDA** 的平台。
   - 有人分享了一个资源建议：**[GPU Puzzles](https://github.com/srush/GPU-Puzzles)**，它提供了一些谜题来帮助学习 **CUDA**。
- **PMP 书籍中缺乏 Machine Learning 内容**: *Locknit3* 表达了失望，指出 **PMP** 书籍中没有包含任何关于 **machine learning** 的引用。
   - 这表明在传统项目管理文献中整合 AI 主题可能存在空白。



**提到的链接**: <a href="https://github.com/srush/GPU-Puzzles">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>: 解决谜题。学习 CUDA。通过在 GitHub 上创建账户为 srush/GPU-Puzzles 的开发做出贡献。

  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288248066251423814)** (13 条消息🔥): 

> - `RoPE 实现`
> - `Fused RMSNorm 集成`
> - `Llama 3.1 和 3.2 支持`
> - `CUDA kernel 开发` 


- **RoPE 实现与 3.1 Scaling 不匹配**: 一位成员指出，当前的 **RoPE** 实现不支持 **3.1** 版本中引入的 **RoPE scaling**。
   - 另一位成员分享了他们的进展，表示他们已经添加了 **RoPE 代码**，该代码调整了 PyTorch 以进行实值计算，并为 CUDA 实现做准备。
- **发现 Fused RMSNorm 参考**: 一位成员分享了一个 [GitHub pull request](https://github.com/karpathy/llm.c/pull/769) 作为 **fused RMSNorm** 的参考，标志着正在进行的开发工作。
   - 建议的参考资料概述了与 RoPE 实现相关的数学属性。
- **RoPE Forward 集成成功**: 同一位成员确认 **RoPE forward** 功能已添加并测试成功。
   - 他们还提到，下一步是在完成 RoPE 更新后集成 **fused RMSNorm**。
- **Fused RMSNorm 集成成功**: **fused RMSNorm forward** 的集成已成功完成，并计划下一步实现 **SwiGLU**。
   - 这一集成获得了点赞，表明整体开发取得了进展。
- **对 Llama 3.2 的兴奋**: 一位成员表达了对新 **Llama 3.2** 模型的狂热，特别提到了 **1B** 变体，展示了社区的兴奋之情。
   - 这为该模型的未来功能增添了期待和兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/769">gordicaleksa 提供的 Fused rmsnorm 参考 · Pull Request #769 · karpathy/llm.c</a>: 供 @karpathy 参考</li><li><a href="https://github.com/karpathy/llm.c/pull/754/commits/026e4ed323fe87004f3a5af6c95e17894cfc5032">karpathy 为 llm.c 添加 llama 3 支持 · Pull Request #754 · karpathy/llm.c</a>: 此分支从复制粘贴 train_gpt2.cu 和 test_gpt2.cu 开始，但在合并回 master 之前，这两个文件（以及其他文件）将进行更改以整合 Llama 3.1 支持。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1288264547538571375)** (6 条消息): 

> - `MI250x`
> - `GroupedGemm examples`
> - `AMD Instinct™ GPU Training`
> - `Architecture Development` 


- **MI250x 目标硬件**：讨论中针对的硬件包括 **MI250x** GPU，正如一位参与者所提到的。
   - 这突显了在计算任务中对 AMD 先进 GPU 产品的关注。
- **AMD Instinct™ GPU 训练详情**：分享了由 *HLRS* 提供的 **AMD Instinct™ GPU Training** 信息，包括即将举行的课程材料链接。
   - 该课程涵盖了各种主题，例如 AMD GPU 上的编程模型和 OpenMP offloading 策略。
- **GroupedGemm 示例的澄清**：一位用户对在多个 **GroupedGemm examples** 中进行选择表示不确定，需要指导以选择合适的示例。
   - *这表明在浏览示例实现时存在挑战，特别是对于该领域的新手。*
- **为 MI250x 构建架构**：讨论中包括了可能构建一个小架构以开始在 **MI250x** 上进行实验的建议。
   - 这意味着在硬件上进行初步实验和开发的一种战略性方法。
- **对示例运行的信心**：参与者一致认为，如果一个示例运行成功，就确认了在 **MI250x** 上进一步使用的准备就绪。
   - 这反映了在硬件利用过程中故障排除和知识共享的协作性质。



**提到的链接**：<a href="https://fs.hlrs.de/projects/par/events/2024/GPU-AMD/">AMD Instinct™ GPU Training</a>：未找到描述

  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1288491302803210292)** (2 条消息): 

> - `TorchAO PR for BitNet`
> - `Optimized Inference Kernel`
> - `Model Training Plans` 


- **针对 BitNet b1.58 的 TorchAO Pull Request 已就绪**：一位成员宣布 [TorchAO PR #930](https://github.com/pytorch/ao/pull/930) 已就绪，增加了使用 **ternary weights**（三值权重）的 **BitNet b1.58** 训练代码，并以 tensor subclass 形式实现。
   - 该版本是之前 **binary weights**（二值权重）实现的显著升级，并与 quantization 框架平滑集成。
- **呼吁开发优化的推理内核**：成员们讨论了开发 **optimized inference kernel**（优化推理内核）的重要性，建议以类似 **gemlite A8W2** 的基准作为参考。
   - 这被定位为补充新实现的训练代码所必需的增强功能。
- **玩具模型训练计划**：一位成员表达了训练一个处理 **10B-100B tokens** 之间的 **toy model** 的意图，倾向于该范围的下限。
   - 他们鼓励其他人贡献额外的计算资源来支持这一努力。



**提到的链接**：<a href="https://github.com/pytorch/ao/pull/930">BitNet b1.58 training by gau-nernst · Pull Request #930 · pytorch/ao</a>：此 PR 增加了 BitNet b1.58 的训练代码（三值权重 - 1.58 bit。BitNet 的第一个版本是二值权重）。这被实现为 tensor subclass，并能很好地与 quantiz... 结合。

  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1288315357765046342)** (4 条消息): 

> - `iPhone 16 performance`
> - `SME on A18 chip`
> - `Metal benchmarks on GitHub` 


- **对 iPhone 16 SME 性能的好奇**：一位成员询问是否有人在 **iPhone 16** 上实验过 **SME**，并指出有说法称它可以处理 14 或 16 个 int8 操作。这也引发了对新 **A18** 芯片能力的兴趣。
   - 另一位用户澄清了性能背景，将其与 **M4 chip** 联系起来，同时引发了关于 benchmark 的更深层讨论。
- **一位新的 iPhone 16 Pro 用户加入**：一位成员宣布他们刚拿到 **iPhone 16 Pro** 并提出测试 SME 性能。他们分享了一个 [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) 链接，以便进一步探索 **Apple GPU microarchitecture**。
   - 该链接为任何有兴趣更详细了解 Apple GPU 进步能力的人提供了资源。



**提到的链接**：<a href="https://github.com/philipturner/metal-benchmarks">GitHub - philipturner/metal-benchmarks: Apple GPU microarchitecture</a>：Apple GPU 微架构。通过创建一个账号为 philipturner/metal-benchmarks 的开发做出贡献。

  

---

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1288324671346839592)** (1 messages): 

> - `Compute Pass 流水线行为`
> - `Workgroup 调度可见性` 


- **带有管道切换的 Compute Pass 行为**：一位用户询问在同一个 Compute Pass 序列中设置时，**pipe1** 调度的所有写入是否在随后的 **pipe2** 调度中可见。
   - 这与 **CUDA streams** 中观察到的行为类似，其中操作通常是序列化的，并且在不同流水线之间维持可见性。
- **寻求管道写入的清晰度**：一名成员提出了关于在 Compute Pass 内切换流水线时写入可见性的问题，特别是其中一个的输出是否可以在下一个中看到。
   - 这涉及到 **CUDA streams** 的运作方式，其中一个操作的写入可以被显式管理，以确保在后续操作中的可见性。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1288470823614677066)** (23 messages🔥): 

> - `Jensen-Shannon 散度基准测试`
> - `Lambda Labs 使用`
> - `SyncBatchNorm 实现` 


- **讨论 JSD Kernel Pull Request**：一名成员分享了 GitHub 上 [JSD kernel pull request](https://github.com/linkedin/Liger-Kernel/pull/264) 的链接，该 PR 解决了 issue #252，重点关注对数空间（log-space）中的分布。
   - 细节表明使用了 Jenson-Shannon 散度来比较两个分布。
- **通过 Lambda Labs 获取高性价比 GPU 访问**：一名成员提到使用 **Lambda Labs** 获取 GPU 访问权限，价格约为 **$2/小时**，使其成为运行基准测试和最终微调的经济实惠选择。
   - 他们强调了 SSH 访问的易用性和按需付费的灵活性，适用于各种项目。
- **基准测试后可查看图表可视化**：一名成员确认在运行单个基准测试后，结果将保存到 CSV 中，并且可以在 visualizations 文件夹中查看图表。
   - 这提供了一种可视化和分析基准测试结果的简便方法。
- **SyncBatchNorm 实现工作**：一名成员宣布计划 **实现 SyncBatchNorm**，并参考官方 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) 进行指导。
   - 他们强调了其在 N 维输入上应用 Batch Normalization 的作用，以及原始 Batch Normalization 论文中描述的核心功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.lambdalabs.com/software/virtual-environments-and-docker-containers">虚拟环境和 Docker 容器 | Lambda 文档</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm">SyncBatchNorm &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/264">由 Tcc0403 添加 JSD kernel · Pull Request #264 · linkedin/Liger-Kernel</a>：摘要 解决 #252 详情 JSD 我们期望输入 $X$ 和目标 $Y$ 是对数空间中的分布，即 $X = log Q$ 且 $Y = log P$。两个分布 $P$ 和 $Q$ 之间的 Jenson-Shannon 散度...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1288556891001065522)** (2 messages): 

> - `Metal 原子操作`
> - `内存屏障语义`
> - `Workgroup 调度` 


- **Metal 原子操作限制了解耦消息传递**：对 **metal atomics** 和 **内存屏障语义** 的理解表明，要实现 Workgroup 之间的解耦回溯（decoupled look-back）或消息传递，需要构建一个包含 **原子字节（atomic bytes）** 的数组，并对所有操作使用原子加载/存储（atomic load/stores）。
   - 理想情况下，只有标志位（flag）使用原子操作，从而允许通过 **快速非原子加载** 来读取数据。
- **观察到随机的 Workgroup 调度**：在实践中，Block 似乎是以本质上 **随机** 的方式调度的，这降低了近期添加消息传递功能的可能性。
   - 这种调度行为引发了对在 Workgroup 之间实现更高效通信实用性的担忧。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1288619866445582366)** (1 条消息): 

> - `GPU Performance Optimization`
> - `Deep Learning Applications with GPUs`
> - `NVIDIA Nsight`
> - `GPU Programming Languages` 


- **博客发布：优化 GPU 性能**：一篇新的 [博客文章](https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization) 探讨了如何提升 Deep Learning 中的 GPU 性能，重点关注 GPU 架构以及 **NVIDIA Nsight** 等性能监控工具。
   - 文章强调，实验和 Benchmarking 是提高 Deep Learning 应用中硬件利用率的关键。
- **GPU 驱动 Deep Learning 创新**：讨论强调了得益于高速并行处理，GPU 计算如何促进 **autonomous vehicles** 和 **robotics** 等多个领域的进步。
   - 深入理解 GPU 性能优化，对于实现更快速、更具成本效益的神经网络 Training 和 Inference 至关重要。



**提到的链接**：<a href="https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization">An Introduction to GPU Performance Optimization for Deep Learning | DigitalOcean</a>：未找到描述

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1288255401132953632)** (82 messages🔥🔥): 

> - `Run Pod 问题`
> - `Molmo 72B 发布`
> - `OpenAI 的最新动态`
> - `Llama 3.2 发布`
> - `Meta 与欧盟的争议` 


- **Run Pod 问题持续存在**：用户报告在 Run Pod 上遇到 **illegal CUDA errors**，一些人建议通过更换机器来解决此问题。
   - 一位用户幽默地建议不要使用 Run Pod，因为问题频发，凸显了其带来的挫败感。
- **Molmo 72B：新的多模态竞争者**：由 Allen Institute for AI 开发的 **Molmo 72B** 拥有最先进的基准测试表现，并基于 PixMo 图像-文本对数据集构建。
   - 该模型被强调为采用 **Apache 许可证**，支持多种模态，旨在与 GPT-4o 等领先的多模态模型竞争。
- **OpenAI 面临组织变革**：OpenAI 的 CTO 辞职引发了重磅关注，引发了对公司营销和未来方向的猜测。
   - 成员们讨论了对 OpenAI 的潜在影响，并调侃内部动态可能会演变成一部 Netflix 迷你剧。
- **Llama 3.2 推出引发关注**：Llama 3.2 的推出以适用于边缘设备的轻量级模型为特色，引发了关于其从 1B 到 90B 不同模型尺寸的讨论。
   - 多个消息来源确认了分阶段推出的消息，一些用户对新模型的性能表示兴奋和好奇。
- **Meta 在欧盟的合规问题**：对话揭示了 Meta 对遵守欧盟法律的担忧，导致欧洲用户的访问受限。
   - 讨论提到了可能影响模型可用性的许可证变更，引发了对公司动机的深入探讨。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aiatmeta/status/1838993953502515702?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>：📣 隆重推出 Llama 3.2：适用于边缘设备的轻量级模型、视觉模型等！有什么新变化？• Llama 3.2 1B & 3B 模型在同类产品中为多种设备端应用提供了最先进的能力...</li><li><a href="https://x.com/danielhanchen/status/1838987356810199153?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>：Llama 3.2 多模态来了！模型尺寸涵盖 1B, 3B 到 11B 和 90B！</li><li><a href="https://molmo.allenai.org/blog">未找到标题</a>：未找到描述</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py">axolotl/src/axolotl/utils/chat_templates.py at main · axolotl-ai-cloud/axolotl</a>：尽管向 axolotl 提问。通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/allenai/Molmo-72B-0924">allenai/Molmo-72B-0924 · Hugging Face</a>：未找到描述</li><li><a href="https://www.llama.com/">Llama 3.2</a>：您可以随处进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 变体。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1927">由 pandora-s-git 修复/添加 Mistral 模板 · Pull Request #1927 · axolotl-ai-cloud/axolotl</a>：描述：此 PR 的目标是修复模板，并使它们尽可能匹配真实情况（Mistral Common），一份文档在此深入探讨了此问题。动力...</li><li><a href="https://www.llama.com/docs/how-to-guides/fine-tuning#hugging-face-peft-lora-(link))">微调 | 操作指南</a>：全参数微调是一种对预训练模型所有层的所有参数进行微调的方法。</li><li><a href="https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/MODEL_CARD.md">PurpleLlama/Llama-Guard3/1B/MODEL_CARD.md at main · meta-llama/PurpleLlama</a>：用于评估和提高 LLM 安全性的一套工具。通过在 GitHub 上创建账号为 meta-llama/PurpleLlama 的开发做出贡献。</li><li><a href="https://x.com/miramurati/status/1839025700009030027?t=pVYyCN8C7RnV0UruM9H2Lg&s=19">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下简报。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1288372813442453536)** (9 条消息🔥): 

> - `Axolotl 与 Metaflow`
> - `使用 rope_theta`
> - `用于微调的 Tool calling` 


- **Axolotl 在 Metaflow 流水线中遇到困难**：一位成员尝试在 [Metaflow pipeline](https://metaflow.org/) 中运行 **Axolotl**，但在使用 Axolotl Docker 镜像加载步骤时遇到问题，而使用更通用的 Python 镜像则运行正常。
   - 他们怀疑 **Metaflow** 构建了一个与 Axolotl 镜像不兼容的自定义 Docker entrypoint，并指出系统设置阻止了拉取其他公共仓库。
- **关于工具替代方案的讨论**：一位成员建议目前的工具共识是使用 **rope_theta**，但承认还没有时间阅读其设置说明。
   - 另一位成员询问他们是否尝试过其他镜像，以验证问题是否专门出在 Axolotl 上。
- **Tool calling 微调的注意事项**：一位成员寻求关于如何针对 **tool calling 进行微调** 的建议，并想知道为此使用 **Alpaca** 是否会有所帮助。
   - 他们随后分享了一个涉及激活或停用夜间模式功能的配置示例。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1288466913198080052)** (5 条消息): 

> - `非法 CUDA 内存访问`
> - `CUDA 版本兼容性`
> - `NVIDIA 驱动更新`
> - `Compute Sanitizer 使用`
> - `CUDA 中的错误检查` 


- **修复非法 CUDA 内存访问的步骤**：要修复 **非法 CUDA 内存访问** 错误，用户应使用 `nvcc --version` 检查 **CUDA 版本** 与 PyTorch 等库的兼容性。
   - 更新 **NVIDIA 驱动程序** 和 CUDA toolkit 通常可以解决这类兼容性问题。
- **使用环境变量进行调试**：设置 `CUDA_LAUNCH_BLOCKING=1` 可以让 CUDA 操作同步运行，有助于确定导致错误的具体行。
   - 用户可以使用该环境变量运行脚本，以便更清晰地检测错误。
- **监控 GPU 显存使用情况**：像 `nvidia-smi` 这样的工具可以帮助追踪 **GPU 显存使用情况**，以避免因超出可用内存限制而导致的非法访问。
   - 根据观察到的内存情况调整 batch size 和模型大小可以防止非法内存访问。
- **验证 CUDA 中的内存操作**：确保所有 CUDA **内存操作** 都是有效的，并避免使用未初始化或已释放的内存以防止错误。
   - 在 CUDA API 调用周围添加错误检查可以帮助在代码早期发现问题。
- **检查自定义 CUDA Kernel**：用户应仔细检查 **自定义 CUDA kernel** 中的数组索引，以防止越界内存访问。
   - 在调试期间使用 **Compute Sanitizer** 可以更有效地识别内存访问问题。



**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=93a77c80-1b4e-4bd5-aab7-e104b89668a5)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1288234446301958328)** (52 条消息🔥): 

> - `Recursive Iterative Models`
> - `Multi-step Tool Applications`
> - `RAG Models`
> - `Cohere Course on RAG`
> - `Using Cohere Models in Projects` 


- **在 LLM 上测试假设**：一位用户询问，如果在多个 LLM 上测试一个假设并得到相似的结果，是否表明他们的 Recursive Iterative 模型有效。
   - 建议通过 benchmarks 和 evaluation harnesses 进一步评估，以确保准确性。
- **探索 Multi-step Tools**：一位成员对其他人最喜欢的 Multi-step Tools 应用表示兴趣，并提到了之前在 Agent 构建日竞赛中的获奖应用。
   - 另一位成员分享了一个包含多个多步用例示例的 GitHub 链接，引发了关于最受欢迎应用的讨论。
- **新的 Cohere RAG 课程发布**：发布了关于与 Weights&Biases 合作的生产级 RAG 新课程的公告，涵盖了评估和 pipelines 等多个重要方面。
   - 该课程时长不到 2 小时，并为参与者提供 Cohere 积分，Cohere 团队成员可随时回答问题。
- **集成 Cohere 模型**：一位用户寻求在 GitHub 项目中添加或测试 Cohere 模型的兴趣，引发了关于更成熟框架和可适配 proxies 优势的讨论。
   - 成员们辩论了他们项目的固有目标以及高效集成工具所需的灵活性。
- **欢迎 Cohere 新成员**：新用户表达了学习 AI 和 Cohere 应用的热情，收到了社区的热烈欢迎。
   - 成员们鼓励提问，并提醒他们在 Cohere Discord 中有可用的资源提供帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/codelion/optillm/tree/main">GitHub - codelion/optillm: Optimizing inference proxy for LLMs</a>：针对 LLMs 的推理优化代理。可以通过在 GitHub 上创建账户为 codelion/optillm 的开发做出贡献。</li><li><a href="https://github.com/MadcowD/ell">GitHub - MadcowD/ell: A language model programming library.</a>：一个语言模型编程库。可以通过在 GitHub 上创建账户为 MadcowD/ell 的开发做出贡献。</li><li><a href="https://github.com/cohere-ai/notebooks">GitHub - cohere-ai/notebooks: Code examples and jupyter notebooks for the Cohere Platform</a>：Cohere 平台的代码示例和 Jupyter Notebooks - cohere-ai/notebooks</li><li><a href="https://www.wandb.courses/courses/rag-in-production">高级 RAG 课程 </a>：面向工程师的实用 RAG 技术：向行业专家学习生产就绪的解决方案，以优化性能、降低成本并提高应用程序的准确性和相关性。</li><li><a href="https://github.com/ack-sec/toyberry">GitHub - ack-sec/toyberry: Toy implementation of Strawberry</a>：Strawberry 的玩具实现。可以通过在 GitHub 上创建账户为 ack-sec/toyberry 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1288226191022620781)** (6 messages): 

> - `Cohere API Key Pricing`
> - `Citations for Unstructured Text Data`
> - `Rerank Fine-tuning Expectations` 


- **Cohere API Key 定价说明**：一位成员解释说，用户可以免费使用**受速率限制的 Trial-Key**，但对于商业应用，则需要切换到**Production-Key**，这是需要付费的。
   - 这一见解强调了在规划资源时需要考虑 API Key 的预期用途。
- **分享引用的最佳实践**：另一位成员推荐参考 **LLM University RAG 模块**，以获取关于非结构化文本数据引用的最佳实践，详情请访问 [cohere.com](https://cohere.com/llmu/rag-start#generate-the-response-with-citations)。
   - 还提到了其他文档，强调了关于如何在模型中有效利用引用的资源。
- **对 Rerank 微调时长的担忧**：一位成员表示，他们的 Rerank 微调任务在一个仅包含 **1711 个查询**（且每个查询的正负样本极少）的数据集上运行了超过 **2 小时**，对此感到担忧。
   - 他们提到没有提供验证集，并且不确定对于他们的配置来说，这个时长是否属于正常范围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/llmu/rag-start#generate-the-response-with-citations">Getting Started with Retrieval-Augmented Generation</a>：LLM University 关于 Retrieval-Augmented Generation 模块的第一部分。</li><li><a href="https://docs.cohere.com/v1/reference/chat">Chat — Cohere</a>：对用户消息生成文本响应。要了解如何使用 Chat API 和 RAG，请参考我们的 Text Generation 指南。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1288429856132038668)** (15 messages🔥): 

> - `NotFoundError with multilingual Embed`
> - `404 error on documentation link`
> - `Switching to newer Embed model`
> - `Response from Cohere Support` 


- **使用 multilingual Embed 时遇到 NotFoundError**：一个团队报告了模型 'packed-embed-multilingual-v2.0' 出现 `cohere.errors.not_found_error.NotFoundError`，暗示该模型可能无法通过当前的 API keys 访问。
   - 另一位成员提到他们切换到了 **embed-multilingual-v3.0**，从而解决了问题，并建议其他人也考虑同样的方案。
- **文档链接出现 404 错误**：一位用户对之前可以访问的 Cohere 文档链接 (https://docs.cohere.com/docs/structure-of-the-course) 现在返回 **404 错误**表示担忧。
   - Cohere 支持团队确认了该问题，并承诺会进一步调查。
- **Embed 模型托管的潜在变化**：成员们讨论了 **multilingual Embed** 模型的托管是否发生了变化，从而导致了这些错误。
   - 他们分享了在使用同一个 API Key 时，**Chat** 功能可以正常使用，但 Embed 模型却出现问题的经历。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1288271754191573032)** (11 条消息🔥): 

> - `Cohere 在 Embedded Systems 中的应用`
> - `智能望远镜支架/追踪器`
> - `Cohere Cookbook`
> - `Embedding 使用案例`
> - `GitHub Notebooks` 


- **探索 Cohere 在 Embedded Systems 中的应用**：一位用户询问了 **Cohere** 在 **embedded systems** 中使用的示例，并表示有兴趣将其集成到其毕业设计项目（capstone project）的智能望远镜支架中。
   - 随后讨论了利用来自 **Messier catalog** 的 **embeddings** 来寻找天体的潜力。
- **智能望远镜项目引发社区关注**：该用户分享了其项目的兴奋点，该项目旨在自动定位 **Messier catalog** 中的 **110 个天体**，并计划在此基础上进一步扩展。
   - 社区成员热情地支持了这一想法，鼓励合作并提供资源。
- **Cohere Cookbook 作为资源**：成员们强调了其网站上提供的 **Cohere Cookbook**，它为使用 Cohere 的生成式 AI 平台提供了现成的指南。
   - 这些指南涵盖了一系列使用案例，例如构建强大的 **Agents** 以及与开源软件集成。
- **Cohere 的使用案例**：讨论中提到了 **Cohere Cookbook** 中的多个类别，包括对于 AI 项目至关重要的 **embedding** 和语义搜索。
   - 鼓励成员探索与其项目需求相关的特定章节。
- **GitHub 上的代码示例**：一位用户分享了 **GitHub notebooks** 的链接，其中包含用于探索 Cohere 平台的代码示例和 **Jupyter notebooks**。
   - 该资源旨在帮助用户进行 Cohere 功能的实际落地和实验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/page/cookbooks">Cookbooks — Cohere</a>：探索一系列 AI 指南并开始使用 Cohere 的生成式平台，这些指南已预先制作并经过最佳实践优化。</li><li><a href="https://github.com/cohere-ai/notebooks">GitHub - cohere-ai/notebooks: Cohere 平台的代码示例和 Jupyter notebooks</a>：Cohere 平台的代码示例和 Jupyter notebooks - cohere-ai/notebooks
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1288258847303008297)** (1 条消息): 

> - `LlamaParse 欺诈警报`
> - `LlamaIndex 产品澄清` 


- **警惕虚假 LlamaParse 网站**：发布了一个关于伪装成 **LlamaIndex 产品**的网站警报：llamaparse dot cloud（我们不提供其链接！）。
   - 建议用户忽略它，因为真正的 **LlamaParse** 可以在 [cloud.llamaindex.ai](https://cloud.llamaindex.ai) 找到。
- **关于 LlamaParse 正当性的澄清**：社区被告知合法的 **LlamaParse** 服务托管在 cloud.llamaindex.ai，以确保用户访问正确的产品。
   - 这一澄清对于防止混淆和潜在的虚假网站误用至关重要。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1288259270004965530)** (5 条消息): 

> - `LlamaParse 诈骗警告`
> - `AWS Gen AI Loft 活动`
> - `针对 Excel 的高级 RAG`
> - `Pixtral 12B 模型发布`
> - `LlamaIndex 招聘公告` 


- **警惕 LlamaParse 冒充者！**：官方发布了关于 [llamaparse dot cloud](https://twitter.com/llama_index/status/1838699883756466512) 的警告，这是一个冒充 LlamaIndex 产品的欺诈网站；真实的 LlamaParse 位于[此链接](https://t.co/jM9ioNJuv3)。
   - *保持警惕*，防范伪装成知名服务的诈骗。
- **AWS Gen AI Loft 的精彩演讲**：我们的 @seldo 将在 AWS Gen AI Loft 上分享关于 RAG 和 Agents 的内容，该活动是 2024 年 3 月 21 日与 @elastic 共同举办的 ElasticON 大会的前奏 ([来源](https://twitter.com/llama_index/status/1838714867697803526))。
   - 与会者将了解 Fiber AI 如何利用 **Elasticsearch** 进行高性能的 B2B 潜在客户开发。
- **发布多工作表 Excel 的 RAG 指南**：一份新指南即将发布，详细介绍了如何使用 OpenAI **o1** 模型对具有多个工作表的 Excel 文件进行高级 RAG 分析 ([链接](https://twitter.com/llama_index/status/1838733053491057029))。
   - 该指南解决了与多工作表 Excel 文件相关的复杂性，以实现更好的分析方法。
- **介绍 Pixtral 12B：多模态奇迹**：来自 @MistralAI 的 **Pixtral 12B 模型** 现在已与 LlamaIndex 兼容，在图表和图像理解方面表现出惊人的能力 ([来源](https://twitter.com/llama_index/status/1838970087354798492))。
   - 在多模态任务中，Pixtral 的表现优于同等规模的模型。
- **加入 LlamaIndex 不断壮大的团队！**：LlamaIndex 正在旧金山寻找充满激情的工程师，以扩展我们充满活力的团队 ([链接](https://twitter.com/llama_index/status/1839055997291344050))。
   - 职位涵盖从 Full-stack 到专业岗位，目标是渴望投身于 **ML/AI** 技术的积极人士。



**提到的链接**：<a href="https://t.co/jM9ioNJuv3">LlamaCloud</a>：未找到描述

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1288256888936595527)** (74 条消息🔥🔥): 

> - `在 Ollama 中使用 LlamaIndex`
> - `访问 VectorStoreIndex`
> - `处理 LlamaTrace 项目错误`
> - `Notebook 中的错误解决`
> - `向 ReAct Agent 传递消息` 


- **Llama Index Notebook 的问题**：用户报告在 LlamaIndex Notebook ([链接](https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook_ollama_replicate/)) 中遇到错误。对类似问题的确认引发了频道内持续的排查讨论。
- **理解 VectorStoreIndex**：对 `VectorStoreIndex` 进行了澄清，特别是如何通过 `index.vector_store` 访问底层的 vector store。用户讨论了 `SimpleVectorStore` 的存储限制，促使考虑替代的 vector stores。
- **解决 LlamaTrace 项目错误**：一位用户表达了在登录其 LlamaTrace 项目后遇到错误的沮丧。最终，他们指出清除 cookies 可以解决该问题，并考虑设置个人实例以避免未来的复杂情况。
- **访问 ReAct Agent 功能**：参与者探索了如何有效地将多个示例作为输入传递给 `ReActAgent`，并使用 `PydanticOutputParser` 获取结构化输出。他们讨论了对用户和系统消息进行正确格式化和处理的需求。
- **关于函数的术语困惑**：在讨论中，参与者强调了对 `VectorStoreIndex` 中可调用方法和属性的混淆，例如 Python 装饰器的使用。误解得到了澄清，从而更好地理解了类属性及其功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.pandas-ai.com/intro">PandasAI 介绍 - PandasAI</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/groq/">Groq - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/">Ollama - Llama 3.1 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent/#customizing-the-prompt>).">ReAct Agent - 带有计算器工具的简单介绍 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/usecases/email_data_extraction/#use-llm-function-to-extract-content-in-json-format>).">邮件数据提取 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#view-prompts>)">Function Calling NVIDIA Agent - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.ReActChatFormatter>)">React - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.output_parser.ReActOutputParser>)">React - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#customizing-the-prompt>)">Function Calling NVIDIA Agent - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic/#llama_index.core.output_parsers.PydanticOutputParser>)">Pydantic - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/output_parsing/llm_program/#initialize-with-pydantic-output-parser>)">LLM Pydantic Program - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1288222695716294696)** (76 条消息🔥🔥): 

> - `Gemini 定价策略`
> - `Anthropic 营收明细`
> - `Llama 3.2 发布`
> - `Mira Murati 离职`
> - `Meta 的 Orion 眼镜`

- **Gemini 定价符合预期**：今日 **Gemini Pro** 的降价与基于其 **Elo** 分数的 loglinear 定价曲线完美契合，表明其在竞争格局中采取了有效的策略。
   - **OpenAI** 模型占据了较高的定价层级，而 **Gemini Pro** 和 **Flash** 则锁定了低端市场，这种科技竞争态势类似于 'iPhone vs Android'。
- **Anthropic 营收预测飙升**：据 CNBC 报道，**Anthropic** 预计今年营收将达到 **$1B**，同比增长高达 **1000%**。
   - 营收构成显示其高度依赖 **Third Party API**（占比 **60-75%**），直接 API 销售和 chatbot 订阅也贡献显著。
- **Llama 3.2 模型发布**：**Llama 3.2** 推出了适用于 edge devices 的轻量级模型，提供 **1B, 3B, 11B,** 和 **90B vision models**，并承诺具备竞争力的性能。
   - 值得注意的是，新模型支持 multimodal 用例，开发者可以免费测试最新功能，从而促进 open-source AI 的发展。
- **Mira Murati 离开 OpenAI**：Mira Murati 在离开 **OpenAI** 时分享了一封感人至深的告别信，这在社区内引发了广泛讨论。
   - Sam Altman 对她的贡献表示感谢，认可了她在任职期间面临的审视，并回顾了她在困难时期提供的精神支持。
- **Meta Orion 眼镜原型亮相**：Meta 展示了其 **Orion** AR 眼镜原型，这标志着经过近十年的研发（最初对其成功的预期较低）后取得的一个重要里程碑。
   - 这些眼镜旨在实现宽广的 field of view 和轻量化设计，将用于内部开发 user experiences，为未来的消费者版本发布做准备。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/Yuchenj_UW/status/1839030011376054454">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：起步与现状。引用 Sam Altman (@sama) 我这样回复。Mira，感谢你所做的一切。很难言表 Mira 对 OpenAI、我们的使命以及我们所有人的意义...</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下便签。</li><li><a href="https://x.com/tanayj/status/1838623375167574421">来自 Tanay Jaipuria (@tanayj) 的推文</a>：据 CNBC 报道，Anthropic 今年营收预计将达到 10 亿美元，同比增长约 1000%。营收细分为：• 第三方 API（通过 Amazon 等）：60-75% • 直接 API：10-25% • Claude chat...</li><li><a href="https://x.com/boztank/status/1838999636402647453">来自 Boz (@boztank) 的推文</a>：我们刚刚发布了 Orion，这是我们研发了近十年的全功能 AR 眼镜原型。当我们开始这段旅程时，我们的团队预测成功的概率（最高）只有 10%...</li><li><a href="https://x.com/AIatMeta/status/1838993953502515702">来自 AI at Meta (@AIatMeta) 的推文</a>：📣 隆重推出 Llama 3.2：适用于边缘设备的轻量级模型、视觉模型等！有什么新变化？• Llama 3.2 1B & 3B 模型在多项设备端任务中提供了同类产品中最先进的能力...</li><li><a href="https://x.com/allen_ai/status/1838956313902219595">来自 Ai2 (@allen_ai) 的推文</a>：认识 Molmo：一系列开放且最先进的多模态 AI 模型。我们最好的模型仅使用 1/1000 的数据就超越了专有系统。Molmo 不仅仅理解多模态数据——它还能执行...</li><li><a href="https://x.com/Smol_AI/status/1838663719536201790">来自 AI News by Smol AI (@Smol_AI) 的推文</a>：值得注意的是 Lmsys Elo 与价格曲线的预测性是多么强，以及这一策略是如何奏效的。今天 Gemini Pro 的降价使其正好符合对数线性定价曲线...</li><li><a href="https://x.com/soldni/status/1839015117587099892">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>：绿色是我最喜欢的颜色</li><li><a href="https://www.interconnects.ai/p/molmo-and-llama-3-vision">Llama 3.2 Vision 和 Molmo：多模态开源生态系统的基石</a>：开放模型、工具、示例、限制以及多模态模型训练的现状。</li><li><a href="https://x.com/nutlope/status/1839016682729226699">来自 Hassan (@nutlope) 的推文</a>：宣布 http://napkins.dev！一个由 Llama 3.2 vision 驱动的开源线框图转应用工具。上传简单网站/设计的截图即可获取代码。100% 免费且开源。</li><li><a href="https://x.com/ggerganov/status/1839009849805291667?s=46">来自 Georgi Gerganov (@ggerganov) 的推文</a>：在你的 Mac 上轻松尝试 4-bit 模型（甚至在欧盟地区）：引用 Georgi Gerganov (@ggerganov) Llama 3.2 3B & 1B GGUF https://huggingface.co/collections/hugging-quants/llama-32-3b-and-1b-gguf-quants-66...</li><li><a href="https://x.com/danielhanchen/status/1839009095883567520?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>：我对 Llama 3.2 的分析：1. 新的 1B 和 3B 纯文本 LLM，9 万亿 token 2. 新的 11B 和 90B 视觉多模态模型 3. 128K 上下文长度 4. 1B 和 3B 使用了来自 8B 和 70B 的蒸馏 5. VL...</li><li><a href="https://x.com/shishirpatil_/status/1839007216407556467?s=46">来自 Shishir Patil (@shishirpatil_) 的推文</a>：💥 LLAMA 模型：1B 是新的 8B 💥 📢 很高兴今天开源 LLAMA-1B 和 LLAMA-3B 模型。在高达 9T token 上进行训练，我们凭借新的 LLAMA 模型系列打破了许多新的基准。跳过...</li><li><a href="https://stability.ai/news/james-cameron-joins-stability-ai-board-of-directors">奥斯卡获奖导演 James Cameron 加入 Stability AI 董事会 — Stability AI</a>：今天我们宣布，传奇电影制作人、技术创新者和视觉特效先驱 James Cameron 已加入我们的董事会。</li><li><a href="https://x.com/andrewcurran_/status/1839037623756796196?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：哇。</li><li><a href="https://x.com/togethercompute/status/1839013617817309563">来自 Together AI (@togethercompute) 的推文</a>：🚀 重大新闻！我们很高兴宣布在 Together AI 上推出 Llama 3.2 Vision 模型和 Llama Stack。🎉 开发者可免费访问 Llama 3.2 Vision 模型，利用开源进行构建和创新...</li><li><a href="https://x.com/natolambert/status/1838991810846502938?s=61">来自 Nathan Lambert (@natolambert) 的推文</a>：将最大的 Molmo 模型 (72B) 与 Llama 3.2 V 90B 进行比较：MMMU，Llama 高出 6 个百分点；MathVista，Molmo 高出 1 点；ChatQA，Molmo 高出 2 点；AI2D，Molmo 高出 4 点；DocVQA 高出 3 点；VQAv2 基本持平或 Molmo 更好...</li><li><a href="https://x.com/vikhyatk/status/1839030970340741408">来自 vik (@vikhyatk) 的推文</a>：molmo > gemini 1.5 flash（在计数方面）</li><li><a href="https://x.com/RihardJar">来自 RihardJar 的推文</a>：...</li>

c/status/1839014234266755473">来自 Rihard Jarc (@RihardJarc) 的推文</a>：$META 的 iPhone 时刻</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpb4m3/molmo_models_outperform_llama_32_in_most_vision/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/Nutlope/napkins/blob/f6c89c76b07b234c7ec690195df278db355f18fc/app/api">napkins/app/api (位于 f6c89c76b07b234c7ec690195df278db355f18fc) · Nutlope/napkins</a>：napkins.dev – 从截图到应用。通过在 GitHub 上创建账号为 Nutlope/napkins 的开发做出贡献。</li><li><a href="https://youtu.be/y8INJQQ96YU?feature=shared">导演 James Cameron 解释他为何加入 Stability AI 董事会</a>：奥斯卡获奖电影制作人 James Cameron 和 Stability AI CEO Prem Akkaraju 加入 'Closing Bell Overtime'，与 CNBC 的 Julia Boorstin 讨论 AI 的影响...</li><li><a href="https://youtu.be/Y5-FeaFOEFM?si=tDZN338_r5nRwIPg">[Paper Club] 🍓 关于推理：Q-STaR 及其相关研究！</a>：在 Strawberry 发布后，我们将调研几篇传闻相关的论文：STaR: Boostrapping Reasoning with Reasoning (https://arxiv.org/abs...</li><li><a href="https://www.diffchecker.com/O4ijl7QY/">Llama 3.2 vs 3.1 - Diffchecker</a>：Llama 3.2 vs 3.1 - LLAMA 3.1 社区许可协议 Llama 3.1 版本发布日期：2024 年 7 月 23 日 “协议”指</li><li><a href="https://www.llama.com/">Llama 3.2</a>：您可以随处进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。</li><li><a href="https://github.com/Nutlope/napkins/blob/f6c89c76b07b234c7ec690195df278db355f18fc/app/api/generateCode/route.ts#L102">napkins/app/api/generateCode/route.ts (位于 f6c89c76b07b234c7ec690195df278db355f18fc) · Nutlope/napkins</a>：napkins.dev – 从截图到应用。通过在 GitHub 上创建账号为 Nutlope/napkins 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 由 <@656968717883670570> 领导的 <@&1284244976024424630> 新见面会！ https://lu.ma/i8ulstlw
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1288243277010833430)** (66 条消息 🔥🔥): 

> - `安装 Basicsr`
> - `ComfyUI vs Forge`
> - `3D 模型生成器`
> - `Stable Diffusion 用法`
> - `ControlNet OpenPose` 


- **Basicsr 安装故障排除**：为了解决 Forge 中 ComfyUI 的问题，@cs1o 建议在激活虚拟环境后，进入 Forge 文件夹并在命令提示符中运行 `pip install basicsr`。
   - 用户表示困惑并计划再次尝试，希望安装后扩展能以标签页形式显示。
- **ComfyUI 与 Forge 的偏好**：成员们讨论了他们对 ComfyUI 和 Forge 的偏好，@emperatorzacksweden 提到他们发现 Invoke 使用起来要容易得多。
   - 一些用户主张坚持使用 ComfyUI，而不是使用 Forge 中的扩展，因为其版本陈旧且存在兼容性问题。
- **对 3D 模型生成器的兴趣**：@placebo_yue 询问了在本地运行的 3D 生成器，提到了 TripoSR 的问题，并表示开源选项似乎已损坏。
   - 虽然对 Luma Genie 和 Hyperhuman 等工具有兴趣，但对其功能表示怀疑。
- **在没有 GPU 的情况下学习 Stable Diffusion**：一位寻求在没有 GPU 的情况下使用 Stable Diffusion 建议的用户被引导使用 Google Colab 或 Kaggle 以获取免费的 GPU 资源。
   - 大家达成共识，认为对于学习 Stable Diffusion 的初学者来说，使用这些平台运行脚本是可以接受的。
- **使用 ControlNet OpenPose 编辑器**：@cs1o 解释了如何利用 ControlNet OpenPose 预处理器在平台内生成和编辑预览图像。
   - 用户对探索此功能很感兴趣，有迹象表明它允许在生成的输出中进行更详细的调整。


  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1288562394552860775)** (1 条消息): 

> - `Llama 3.2 Release` (Llama 3.2 发布)
> - `Multimodal Support` (多模态支持)
> - `Long-context Models` (长上下文模型)
> - `Finetuning Options` (微调选项)


- **Llama 3.2 发布并具备多模态特性**：**Llama 3.2** 的发布引入了具备长上下文支持的 **1B** 和 **3B 文本模型**，允许用户在长上下文数据集上尝试使用 `enable_activation_offloading=True`。
   - 此外，**11B 多模态模型**支持 **The Cauldron 数据集**和自定义多模态数据集，以增强生成能力。
- **提供灵活的微调选项**：**Llama 3.2** 提供多种微调方法，包括全量微调、**LoRA** 和 **QLoRA**，并即将支持 **DoRA** 微调。
   - 请在接下来的几天内关注有关配置详情的更多更新。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1288581828231561258)** (29 条消息🔥): 

> - `Green Card Concerns` (绿卡担忧)
> - `Pen and Paper Return` (回归纸笔)
> - `Llama 3.2 Access` (Llama 3.2 访问权限)
> - `MetaAI Restrictions` (MetaAI 限制)
> - `Legal Department Reactions` (法务部门反应) 


- **对绿卡的渴望**：一位成员幽默地表达了对**绿卡**的极度渴望，并暗示由于自己的处境，他们可能会让欧洲的生活变得更加艰难。
   - “我不会告诉任何人，以此换取一张绿卡”凸显了他们的沮丧和谈判意愿。
- **回归纸笔**：一位成员以幽默的口吻表示要回归像 80 年代那样**在纸上绘制手术报告**，分享了对传统方法的怀念。
   - 这反映了在当前的挫折中对现代流程的不满。
- **对 Llama 3.2 访问权限的困惑**：对于无法直接**下载 Llama 3.2** 存在困惑，并有推测认为使用它的服务可能仍然可以访问。
   - 一位成员指出：“也许就像如果一家美国公司用 Llama 3.2 构建了一项服务，我仍然可以使用该服务。”
- **MetaAI 的限制**：围绕 **MetaAI** 展开了讨论，一些成员确认由于地理限制导致欧盟用户无法登录，他们无法访问该平台。
   - 有人评论道：“我不认为我们这里能访问 MetaAI”，这说明了访问障碍如何影响用户。
- **关于 Llama 3.2 发布的不确定性**：成员们对 **Llama 3.2 405B** 的发布状态提出质疑，对它是否开源表示困惑。
   - 一位成员承认：“是的，我在 Hugging Face 上找不到它，哈哈”，展示了在寻找模型时的挫败感。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1288275906032435280)** (9 条消息🔥): 

> - `PyTorch 中的 TF32 使用`
> - `KV-cache 切换 RFC`
> - `处理 Tensor 尺寸调整` 


- **为 FP32 用户考虑 TF32**：讨论围绕着为仍在使用 **FP32** 的用户启用 **TF32** 作为一个选项展开，因为它可以加速矩阵乘法（matmul）。普遍观点认为，如果已经在利用 **FP16/BF16**，TF32 可能不会带来额外收益。
   - 有人幽默地提到：*“我想知道谁会放着 FP16/BF16 不直接用，而更倾向于它（TF32）”*。
- **关于 KV-cache 切换的 RFC 提案**：[一项关于 KV-cache 切换的 RFC](https://github.com/pytorch/torchtune/issues/1675) 已被提出，旨在改进模型前向传递（forward passes）期间缓存的处理方式。该提案解决了当前缓存总是被不必要更新的局限性。
   - 针对在特定模型设置中这种缓存机制的必要性和可用性提出了疑虑，引发了进一步讨论。
- **在 PR 中合并 compile 支持**：一名成员询问是否应该合并一个相关的 **pull request**，该 PR 修复了 KV-cache 中的重新编译问题并添加了 compile 支持。这引出了将 compile 支持整合进该 PR 以实现更好集成的建议。
   - 对话表明，大家正在共同努力优化代码并提高整个项目的性能。
- **关于处理 Tensor 尺寸调整的建议**：有人询问除了使用 **Tensor item()** 方法之外，如何改进对 Tensor 尺寸调整的处理。请求其他成员提供见解或替代方案。
   - 一名成员承认需要更好的解决方案，并承诺会进一步思考。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1675">[RFC] Supporting KV-cache toggling · Issue #1675 · pytorch/torchtune</a>：问题：目前，当我们使用 model.setup_caches() 时，KV-caches 在模型的后续每一次前向传递中都会更新。我们有使用 model.setup_caches() 的有效用例，但随后没有...</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices">CUDA 语义 &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.cfjs9aarcek">torch.compile，缺失的手册</a>：torch.compile，缺失的手册。你来到这里是因为你想使用 torch.compile 让你的 PyTorch 模型运行得更快。torch.compile 是一个复杂且相对较新的软件，因此你...</li><li><a href="https://github.com/pytorch/torchtune/pull/1663">由 SalmanMohammadi 提交的修复 KV-cache + compile 中的重新编译问题 · Pull Request #1663 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档，还是其他（请在此处添加）。我之前没意识到，当 #1449 合并时，它破坏了兼容性...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1288216454667440303)** (4 条消息): 

> - `MOToMGP 错误`
> - `为 MAX 调整 Linux 机器配置`
> - `Mojo 生日庆祝` 


- **MOToMGP 错误调查**：一位用户询问关于“failed to run the MOToMGP pass manager”的错误，并寻求任何复现案例，以便预防或提供更好的错误提示。
   - 社区成员被鼓励分享关于这一特定问题的经验或见解。
- **为 MAX 和 Ollama 3.1 调整 Linux 机器配置**：一名成员就运行带有 **ollama3.1** 的 **MAX** 时如何调整 Linux 机器的规格寻求建议。
   - 这个问题开启了关于资源分配的最佳配置以确保性能的讨论。
- **祝 Mojo 生日快乐！**：分享了一条庆祝 Mojo 生日的短消息，幽默地提到 AI 几乎数对了蜡烛的数量。
   - 欢快的公告为聊天带来了节日气氛，突显了社区的友谊。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1288555804143911025)** (1 messages): 

> - `Mojo GitHub Discussions`
> - `Community Q&A on Magic` 


- **Mojo GitHub Discussions 变更公告**：从 **9 月 26 日**开始，**Mojo GitHub Discussions** 标签页将保持可访问状态，但将禁用新的讨论和评论，以便将社区对话集中在 Discord 中。
   - 这一决定源于意识到将具有历史意义的重要讨论转换为 Issues 的价值不大，因此由于活跃度较低，决定关闭 **MAX repo** 上的 GitHub Discussions。
- **计划举行 Magic 问答社区会议**：社区会议将于 **9 月 30 日**举行，届时 Zac 将回答关于 **Magic 🪄** 的问题，鼓励参与者通过链接的 [Google form](https://forms.gle/hyXTJJz1dyXNsD5M8) 提交问题。
   - 会议录像将上传至 **YouTube**，供无法实时参加的人员观看。



**Link mentioned**: <a href="https://forms.gle/hyXTJJz1dyXNsD5M8">Community Magic Questions</a>: 请分享您与 Magic 相关的问题！Zac 将在 9 月 30 日的社区会议 Magic 问答环节中回答这些问题。一如既往，录像将发布到 YouTube。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1288251555753099304)** (21 messages🔥): 

> - `Mojo and C communication speed`
> - `Implementation of associated aliases`
> - `Potential for chainable iterables in Mojo`
> - `Ownership mechanisms in Mojo`
> - `UnsafeMaybeUninitialized wrapper` 


- **Mojo 与 C 相比 Python 的通信速度**：一位参与者询问 Mojo 与 C（通过 DLHandle）的通信是否比与 Python 的通信更快，并认为这可能取决于 Python 是否调用了 C。
   - 另一位成员表示赞同，认为答案可能因具体的实现而异。
- **Evan 正在实现关联别名（associated aliases）**：Evan 正在 Mojo 中实现 **associated aliases**，其定义方式类似于给出的包含 traits 和类型别名的代码片段。
   - 成员们对这一进展及其对代码组织和清晰度的潜在影响表示兴奋。
- **对 Mojo 链式可迭代对象的兴趣**：一位成员表示有兴趣在 Mojo 中创建 **chainable iterables**，参考了类似于 **torchdata** 的转换流水线（transformation pipeline）。
   - 他们推测关联别名可能会为 list、set 和 dict 启用适当的可迭代 trait。
- **Mojo 的所有权初始化机制**：讨论围绕 Mojo 是否有方法指示变量已初始化或未初始化展开，参考了 `lit.ownership.mark_destroyed` 和 `lit.ownership.mark_initialized`。
   - 虽然这得到了确认，但有人指出目前缺乏相关文档，用户应查看标准库以获取示例。
- **Mojo 中的所有权操作封装**：讨论了 **UnsafeMaybeUninitialized** 这一术语，一位成员将其描述为 Mojo 中所有权操作（ownership operations）的封装。
   - 随着 Mojo 趋于稳定，成员们呼吁增加更多像关联别名这样的特性，以简化编码和调试工作。



**Link mentioned**: <a href="https://github.com/modularml/mojo/pull/3453/files">[stdlib] Use ownership ops in `Uninit` to remove pointer indirection. by helehex · Pull Request #3453 · modularml/mojo</a>: 所有权操作提供了比 pop.array 更清晰的实现，并允许移除指针间接寻址。这使得 UnsafeMaybeUninitialized 的内部字段仅为一个 ElementType ins...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1288234789509533789)** (12 条消息🔥): 

> - `o1 Preview 和 Mini API 访问`
> - `Llama 3.2 实验结果`
> - `Logo 设计选择`
> - `Open Interpreter 损坏问题` 


- **围绕 o1 Preview 和 Mini API 的兴奋**：成员们正在讨论访问 **o1 Preview 和 Mini API**，其中一人对 Lite LLM 的支持表示好奇，因为他们收到了来自 Open Interpreter 的响应。
   - 另一位成员幽默地提到准备测试它，但没有 Tier 5 的访问权限。
- **Llama 3.2 实验**：关于即将进行的 **Llama 3.2** 实验的讨论，引发了关于人们计划进行哪些测试的问题。
   - 一位成员分享说，他们尝试让 Llama 3.2 统计桌面上的文件，结果失败了。
- **Logo 设计选择**：一位成员分享了他们的设计进展，表示最初尝试了 GitHub logo，但发现当前的选择更好。
   - 另一位成员插话，开玩笑地质疑他们设计决策的 **power**（力量），为讨论增添了轻松的基调。
- **Open Interpreter 损坏问题**：一位成员分享了一个轻松的个人问题，称他们弄坏了自己的 Open Interpreter 设置，但正在恢复正轨。
   - 对话中包括提供共享访问权限以进行故障排除，社区成员纷纷表示支持。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1288502577214001223)** (5 条消息): 

> - `Llama 3.2 发布`
> - `Tool Use YouTube 剧集`
> - `GroqCloud 可用性` 


- **Llama 3.2 发布轻量级边缘模型**：Meta 推出了 **Llama 3.2**，其特点是包含用于设备端用例的 1B 和 3B 模型，并在发布首日支持 **Arm**、**MediaTek** 和 **Qualcomm**。公告强调了新的 **11B & 90B vision models** 在与领先的闭源模型竞争中表现出色。
   - 开发者可以直接从 Meta 和 [Hugging Face](https://go.fb.me/w63yfd) 获取这些模型，它们正陆续在包括 **AWS**、**Google Cloud** 和 **NVIDIA** 在内的 **25+ 合作伙伴**平台上推出。
- **Tool Use 剧集涵盖开源 AI**：[观看最新的 Tool Use 剧集](https://www.youtube.com/watch?v=-To_ZIynjIk)，其中 **AJ (@techfren)** 讨论了开源编程工具和基础设施项目。该剧集强调了社区驱动的开源创新在 AI 领域的重要性。
   - 讨论强调了利用开源工具的日益增长的趋势，这与频道中分享的情绪一致。
- **Llama 3.2 现已在 GroqCloud 上线**：Groq 宣布在 **GroqCloud** 中提供 **Llama 3.2** 的预览版，展示了其与基础设施的集成。此次发布旨在增强开发者和企业获取 Llama 模型的便利性。
   - **Mikebirdtech** 注意到了热烈的反响，评论说与 Groq 相关的一切都很快，进一步证实了部署的速度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/groqinc/status/1839002579511968113?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 Groq Inc (@GroqInc) 的推文</a>：💥 GroqCloud 提供 Llama 3.2 预览版。</li><li><a href="https://x.com/aiatmeta/status/1838993953502515702?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 AI at Meta (@AIatMeta) 的推文</a>：📣 介绍 Llama 3.2：用于边缘设备的轻量级模型、视觉模型等！有什么新变化？• Llama 3.2 1B & 3B 模型在同类产品中为多种设备端应用提供了最先进的能力...</li><li><a href="https://www.youtube.com/watch?v=-To_ZIynjIk">Techfren 最喜欢的 AI 工具有哪些？- 第 6 集 - Tool Use</a>：我们邀请到了科技界的 AJ（也被称为 @techfren）。我们讨论了开源编程工具，并演示了一些开源基础设施项目...
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1288452362062790656)** (2 条消息): 

> - `Quiz 3 信息` 


- **Quiz 3 等待查询**：一位成员询问了 **Quiz 3** 的状态，表示他们正在等待相关信息。
   - 另一位成员迅速回应称，可以在 **course website** 的 syllabus（教学大纲）部分找到它。
- **课程网站资源**：一位成员指导提问者查看 **course website** 以获取预定评估的详细信息。
   - 这强调了经常检查 syllabus 更新以获取及时考试信息的重要性。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1288224458221420606)** (13 messages🔥): 

> - `Gemini 1.5 Flash performance`
> - `New MMLU-Pro dataset`
> - `Chain of Thought effectiveness`
> - `AutoGen usage in research` 


- **Gemini 1.5 Flash 展示了令人印象深刻的基准测试结果**：据报道，截至 2024 年 9 月，**Gemini 1.5 Flash** 模型的得分达到了 **67.3%**，并在各种数据集上都有所提升。
   - 据指出，**Gemini 1.5 Pro** 的得分更高，达到了 **85.4%**，展示了模型性能的显著进步。
- **引入 MMLU-Pro 数据集**：**MMLU** 数据集的一个新增强版本，称为 **MMLU-Pro**，包含跨 **57 个学科**且难度更高的题目。
   - 该数据集旨在为评估模型能力提供更好的挑战，特别是在 **STEM** 和**人文科学**等领域。
- **Chain of Thought 的有效性受到质疑**：一项新研究探讨了 **Chain of Thought (CoT)** 何时有益，指出除了数学和符号推理任务外，直接回答的表现与之相似。
   - 该分析涉及 **300 多项实验**，并表明对于 **95%** 的 MMLU 任务，CoT 是不必要的；它的主要用途在于**符号计算**。
- **研究强调了 AutoGen 的使用**：另一个研究项目显著地利用了 **AutoGen**，展示了其在当前 AI 发展中的相关性。
   - 这指向了利用自动化模型生成技术来增强性能和研究产出的持续趋势。
- **关于 Flash 模型的讨论**：成员们参与了围绕 **Flash 模型** 的讨论，该模型最近进行了更新，其性能与之前的 Pro 级模型相当。
   - 有人强调，其定价证明了其在性价比领先的模型类别中的地位。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/ZayneSprague/status/1836784332704215519">来自 Zayne Sprague (@ZayneSprague) 的推文</a>: To CoT or not to CoT?🤔 14 个 LLM 的 300 多项实验以及对 100 多篇近期论文的系统性元分析 🤯 除了数学和符号推理外，直接回答与 CoT 一样好 🤯 你不需要 Co...</li><li><a href="https://threadreaderapp.com/thread/1836784332704215519.html">Thread Reader App 上 @ZayneSprague 的线程</a>: @ZayneSprague: To CoT or not to CoT?🤔 14 个 LLM 的 300 多项实验以及对 100 多篇近期论文的系统性元分析 🤯 直接回答与 CoT 一样好，除了数学和符号推理 🤯...</li><li><a href="https://arxiv.org/abs/2409.12183">To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning</a>: 通过 Prompt 使用 Chain-of-thought (CoT) 是激发大语言模型 (LLM) 推理能力的事实上的方法。但这种额外的“思考”究竟对哪类任务真正有帮助...</li><li><a href="https://deepmind.google/technologies/gemini/flash/">Gemini Flash</a>: 我们的轻量级模型，针对速度和效率至关重要的场景进行了优化，上下文窗口高达 100 万个 token。
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1288369279972020285)** (2 messages): 

> - `Langtrace Updates`
> - `DSPy Features`
> - `Automatic Experiment Tracking`
> - `Checkpoint State Tracking` 


- **令人兴奋的 DSPy 特性即将发布！**：本周将在 [Langtrace](https://x.com/karthikkalyan90/status/1838809430009299240?s=46&t=XrJJzmievg67l3JcMEEDEw) 上发布针对 **DSPy** 的新特性，包括新的项目类型和受 **MLFlow** 启发的自动实验追踪。
   - 还将包括**自动 Checkpoint 状态追踪**、**评估分数趋势线**和 **Span 图表**，并支持 **litellm**。
- **Typescript 版本 Ax 将支持新特性**：这些新的 DSPy 特性很快将在 Langtrace 的 **Typescript 版本 Ax** 中可用。
   - 这有望增强使用 **DSPy** 的用户的可用性和功能。



**提及的链接**: <a href="https://x.com/karthikkalyan90/status/1838809430009299240?s=46&t=XrJJzmievg67l3JcMEEDEw">来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文</a>: @langtrace_ai 本周将发布一些 DSPy 特有的原生特性。- 新项目类型 - DSPy - 自动实验追踪 (灵感来自 MLFlow) - 自动 Checkpoint 状态追踪 - 评估 ...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1288228392910590106)** (5 条消息): 

> - `DSPy 文本分类`
> - `Claude 模型`
> - `DSPy 用于对话式 Agent`
> - `编排用户查询`
> - `DSPy 记忆与对话历史` 


- **DSPy 在欺诈检测的文本分类方面表现出色**：一位用户正使用 **DSPy** 根据语义和上下文将文本分类为**三种欺诈类型**，并寻求关于该任务最佳 Claude 模型的建议。
   - 另一位成员指出，**Sonnet 3.5** 是目前最顶尖的 Anthropic 模型，而 **Haiku** 则是更具性价比的选择。
- **DSPy 作为对话式 Agent 编排工具**：一位成员正在探索将 **DSPy** 作为编排器，用于将用户查询路由到子 Agent，并询问其处理与用户直接对话的能力。
   - 他们讨论了为其提供可调用工具的潜力，并提到了记忆的概念，质疑其效果是否优于使用**带有摘要功能的独立对话历史**。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1288502464328237178)** (3 条消息): 

> - `文本分类`
> - `分类中的复杂类别`
> - `分类任务教程` 


- **探索文本分类中的复杂类别**：成员们讨论了“将文本分类为复杂类别”的概念，强调了精确定义的必要性。
   - 一位成员指出了**美国政治**、**国际政治**、**模糊查询**和**超出范围查询**之间的区别，强调了它们的定义如何取决于业务上下文。
- **分类教程的时机**：一位成员指出，当前的讨论非常及时，因为他们正在编写关于分类任务的教程。
   - 这表明社区正在共同努力，以增强在分类领域的理解和清晰度。


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1288557746454790234)** (9 条消息🔥): 

> - `tinygrad 资源`
> - `调试 tinygrad`
> - `为 tinygrad 做贡献` 


- **为 tinygrad 贡献的核心资源**：一位成员分享了一系列 [tinygrad 教程](https://mesozoic-egg.github.io/tinygrad-notes/)，涵盖了内部机制，将帮助新贡献者理解该框架。
   - 他们强调 [快速入门指南](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) 和 [抽象指南](https://github.com/tinygrad/tinygrad/blob/master/docs/abstractions2.py) 是非常棒的资源。
- **通过代码高效学习**：为了掌握 tinygrad 的各种概念，一位成员建议阅读代码，并让产生的问题引导进一步的学习。
   - *使用搜索引擎或 ChatGPT 等工具*可以帮助回答疑问并增强理解，从而创建一个高效的学习反馈循环。
- **追踪讨论以获取见解**：建议关注 GitHub 上的 Pull Requests 和 Discord 频道中的讨论，以此来跟进贡献者们在 tinygrad 中开展的工作。
   - 这提供了背景知识，并能深入了解社区内正在进行的项目。
- **使用 DEBUG 理解 tinygrad 的流程**：另一位成员提到，在简单操作上使用 `DEBUG=4` 可以显示生成的代码，有助于理解从前端到后端的流程。
   - 这种方法是剖析和理解 tinygrad 内部运行机制的实用手段。
- **学习需要坚持**：一条轻松的评论指出，由于 tinygrad 的复杂性，熟悉它的过程可能感觉像是在*拿头撞墙*。
   - 这反映了深入研究复杂系统的挑战性，但也鼓励大家持之以恒。



**提到的链接**：<a href="https://mesozoic-egg.github.io/tinygrad-notes/">Tinygrad 教程</a>：关于 tinygrad 的教程

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1288613336568692766)** (1 条消息): 

> - `训练速度问题`
> - `采样代码 Bug`
> - `模型输出质量` 


- **tinygrad 中的训练循环太慢**：一位用户对 tinygrad（版本 0.9.2）在训练字符模型时的**缓慢训练**表示沮丧，称该过程“慢得离谱”。
   - 他们提到租用了 **4090 GPU** 来提高性能，但并未看到显著提升。
- **采样代码中的 Bug 影响输出质量**：在最初怀疑是训练缓慢之后，用户在他们的**采样代码**中发现了一个 **Bug**，该 Bug 导致推理期间的输出质量很差。
   - 他们澄清说问题不在于训练代码，而是专门出在采样实现上。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1288442460388986940)** (8 messages🔥): 

> - `Open Source Chat UI for Programming Tasks`
> - `Thumbs Up/Down Review Options for Chatbots`
> - `Azure Chat OpenAI Integration` 


- **寻找用于编程的开源聊天 UI**：一位成员询问是否有专门为**编程任务**设计的聊天界面**开源 UI**。
   - 他们表示这个问题比较宽泛，欢迎任何有相关经验的人提供建议。
- **寻找聊天机器人反馈功能**：另一位成员询问是否有人为他们的聊天机器人实现了**点赞/点踩评价选项**。
   - 他们分享说自己创建了一个**自定义前端**，并排除了使用 **Streamlit** 的选项。
- **关于聊天机器人增强功能的讨论线程**：出现了一个专注于**点赞/点踩评价选项**的讨论线程，引发了进一步的参与。
   - 这表明社区对增强聊天机器人设计中的用户反馈机制很感兴趣。
- **与 Azure Chat OpenAI 的集成**：一位成员透露他们正在利用 **Azure Chat OpenAI** 进行聊天机器人开发。
   - 这一提法突出了其他人在类似用例中可能会考虑的平台选择。
- **需要关于聊天机器人实现的建议**：使用 **Azure Chat OpenAI** 的开发者向社区征求了关于其项目的**想法和建议**。
   - 这强调了成员之间相互支持、共同应对开发挑战的协作努力。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1288491956179304530)** (1 messages): 

> - `Agentic RAG application`
> - `LangGraph`
> - `Lightning Studios`
> - `Streamlit`
> - `Research Assistant` 


- **构建 Agentic RAG 应用**：一位用户分享了他们使用 **LangGraph**、**Ollama** 和 **Streamlit** 构建 **Agentic RAG 应用**的经验，该应用作为一个研究助手，可以从论文和网络搜索中检索有价值的信息。
   - 他们通过 [Lightning Studios](https://lightning.ai/maxidiazbattan/studios/langgraph-agenticrag-with-streamlit) 成功部署了该应用，并在 [LinkedIn 帖子](https://www.linkedin.com/posts/maxidiazbattan_last-weekend-i-decided-to-put-the-tool-calling-activity-7244692826754629632-Um7w?utm_source=share&utm_medium=member_ios) 中记录了他们的历程。
- **利用 Lightning Studios 进行实验**：用户利用 **Lightning Studios** 运行实验并为其研究应用部署 **Streamlit app**。
   - 通过这个平台，他们优化了应用设置，结合了不同的技术以增强功能。



**提到的链接**：<a href="https://lightning.ai/maxidiazbattan/studios/langgraph-agenticrag-with-streamlit">LangGraph-AgenticRAG with Streamlit - a Lightning Studio by maxidiazbattan</a>：该 Studio 提供了关于如何集成和使用结合了 LangGraph、Ollama 和 Streamlit 的 Agentic-RAG 指南。

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1288274983339954198)** (1 messages): 

> - `GANs`
> - `CNNs`
> - `ViTs`
> - `Image tasks`
> - `Algorithm comparison` 


- **GANs, CNNs 和 ViTs 作为顶尖图像算法**：一位成员表示，**GANs**、**CNNs** 和 **ViTs** 经常在**图像任务**的顶尖算法位置上交替领先。
   - 他们寻求确认，并希望能有一个可视化的**时间线**来展示这种演变。
- **请求时间线可视化图表**：一位成员请求提供一个**可视化时间线**，以说明 **GANs**、**CNNs** 和 **ViTs** 主导地位的变化。
   - 这一请求突显了人们对了解这些算法在图像处理领域的历史背景的兴趣。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1288214389261996133)** (7 messages): 

> - `MaskBit 图像生成`
> - `MonoFormer 多模态 Transformer`
> - `Sliding window attention`
> - `VQGAN 现代化`
> - `Embedding-free 生成` 


- **MaskBit 彻底改变了图像生成**：关于 [MaskBit](https://arxiv.org/abs/2409.16211) 的论文介绍了一种基于 bit tokens 的 Embedding-free 图像生成模型，在 ImageNet **256 × 256** 基准测试中实现了 **1.52** 的 state-of-the-art FID。
   - 该研究还对 **VQGANs** 进行了深入探讨，得到了一个高性能模型，在提高可访问性的同时揭示了此前未知的细节。
- **MonoFormer 融合了自回归与扩散**：[MonoFormer 论文](https://arxiv.org/abs/2409.16280) 提出了一种统一的 Transformer 架构，可同时用于自回归文本生成和基于扩散的图像生成，其性能可与 state-of-the-art 模型相媲美。
   - 这是通过利用两种方法在训练上的相似性实现的，区别仅在于训练期间使用的 attention masks。
- **Sliding window attention 仍依赖 positional encoding**：一位成员指出，虽然 **sliding window attention**（类似于 Longformer）带来了好处，但它仍然包含 **positional encoding** 机制。
   - 这引发了关于模型架构中效率与维持位置感知需求之间平衡的进一步讨论。


  

---



---



---



---



---



---



---



{% else %}


> 完整的逐频道详情已在邮件中截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}