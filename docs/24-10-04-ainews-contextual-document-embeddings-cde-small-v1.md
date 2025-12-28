---
companies:
- meta-ai-fair
- openai
- google-deepmind
- weights-biases
- togethercompute
date: '2024-10-05T01:38:06.226049Z'
description: '**Meta** 发布了全新的文生视频模型 **Movie Gen**。该公司声称，与 OpenAI 的 Sora（基于 Diffusion
  Transformers 架构）相比，Movie Gen 将 **Llama 3** 适配于视频生成的效果更佳，不过目前该模型尚未公开发布。研究人员 Jack
  Morris 和 Sasha Rush 推出了 **cde-small-v1** 模型，该模型采用了创新的**上下文批处理 (contextual batching)**
  训练技术和**上下文嵌入 (contextual embeddings)**，仅凭 **1.43 亿 (143M)** 参数就实现了强劲的性能表现。**OpenAI**
  推出了 **Canvas**，这是 ChatGPT 的一个协作界面，并采用了合成数据进行训练。**Google DeepMind** 迎来了 Tim Brooks
  的加入，他将致力于视频生成和世界模拟器 (world simulators) 的研究。Google 发布了 **Gemini 1.5 Flash-8B**，通过算法效率的提升优化了成本和速率限制。'
id: 2bac3a4c-2649-40d3-965c-5fc2ab97ddb2
models:
- llama-3
- cde-small-v1
- gemini-1.5-flash-8b
- chatgpt
original_slug: ainews-contextual-document-embeddings-cde-small-v1
people:
- jack-morris
- sasha-rush
- tim-brooks
- demis-hassabis
- karina-nguyen
title: 上下文文档嵌入：`cde-small-v1`
topics:
- contextual-embeddings
- contextual-batching
- video-generation
- synthetic-data
- model-efficiency
- training-techniques
- rag
- algorithmic-efficiency
---

<!-- buttondown-editor-mode: plaintext -->**Contextual Batching is all you need.**

> 2024年10月3日至10月4日的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**226** 个频道和 **1896** 条消息）。为您节省了预计 **210** 分钟的阅读时间（按每分钟 200 字计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们通常将 AINews 的头条新闻留给大型模型实验室的动态，今天 Meta 的新文本生成视频模型 [Movie Gen](https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 席卷了新闻。其[论文](https://x.com/sytelus/status/1841960777588379656)声称，他们能够将 Llama 3 适配到视频生成中，且效果显著优于 OpenAI Sora 的 Diffusion Transformers。然而，目前还没有实际发布，只有经过精挑细选的营销视频，而我们在这里尝试关注您可以实际使用的消息。

因此，我们很高兴向大家推荐 Jack Morris 和 Sasha Rush 的新论文以及关于 [Contextual Document Embeddings](https://arxiv.org/abs/2410.02525) 的 `cde-small-v1` 模型，它是“**世界上最好的 BERT 尺寸的文本嵌入模型**”。


![image.png](https://assets.buttondown.email/images/b1a9ffd9-ed18-4159-9925-22029f37649e.png?w=960&fit=max)


Jack 的描述最为精辟：

> "典型的文本嵌入模型有两个主要问题：
> 
> 1. 训练它们很复杂，需要很多技巧：巨大的 batches、蒸馏、hard negatives...
> 2. 嵌入（embeddings）不知道它们将被用于哪个语料库；因此，所有的文本片段都以相同的方式编码。"
>
> 为了解决 (1)，我们开发了一种新的训练技术：contextual batching。所有的 batches 都共享大量的上下文——一个 batch 可能关于肯塔基州的赛马，下一个 batch 可能关于微分方程等。
>
> 
> 对于 (2)，我们提出了一种新的 **contextual embedding** 架构。这需要对训练和评估流程进行更改，以纳入 **contextual tokens**——本质上，模型会看到来自周围上下文的额外文本，并可以据此更新嵌入。

这似乎很有道理——在进行正式嵌入之前，先引导嵌入模型适应上下文 tokens。

虽然大多数在 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 登顶的嵌入模型参数量都超过 7B（得分约为 72），但拥有 1.43 亿参数的 `cde-small-v1` 得分达到了体面的 65，且在比它大 50 倍的模型中表现稳健。这是一个非常棒的效率提升。


![image.png](https://assets.buttondown.email/images/ab55e9ab-ac48-4e7c-abb6-8f35ca02598a.png?w=960&fit=max)


在您探索新的嵌入模型时，您可能还想探索来自 [今日赞助商](http://wandb.me/ainews-course) 的其他高级 RAG 技术！


---

> **由 RAG++ 为您呈现**：RAG 的查询细化（Query refinement）就像给您的系统装上了 X 光透视眼；有了它，系统可以更清晰地“看到”用户意图——从而实现更准确的 chunk 检索和更相关的 LLM 响应。
>
> [
![image.png](https://assets.buttondown.email/images/05c0f424-b239-4561-bdc1-42322ae26689.png?w=960&fit=max)
](http://wandb.me/ainews-course)
>
> 
> 在 Weights & Biases 的新课程 **[RAG++ : From POC to Production](http://wandb.me/ainews-course)** 的这段 YouTube 摘录中了解如何改进您的 RAG 查询细化，并注册获取免费的 LLM API 额度以开始学习！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与公司动态**

- **OpenAI 动态**：OpenAI 推出了 Canvas，这是一个用于在写作和编程项目上与 ChatGPT 协作的新界面。[@karinanguyen_](https://twitter.com/karinanguyen_/status/1841888532299973056) 强调了关键功能，包括行内反馈、针对性编辑和快捷菜单。Canvas 模型是使用新型合成数据生成技术训练的，允许在不依赖人工数据收集的情况下进行快速迭代。

- **Google AI 新闻**：[@_tim_brooks](https://twitter.com/_tim_brooks/status/1841982327431561528) 宣布加入 Google DeepMind，致力于视频生成和世界模拟器（world simulators）的研究。[@demishassabis](https://twitter.com/demishassabis/status/1841984103312208037) 对他表示欢迎，并对将长期以来的世界模拟器梦想变为现实表示兴奋。

- **模型发布与更新**：Google 发布了 Gemini 1.5 Flash-8B，与之前的版本相比，价格降低了 50%，速率限制（rate limits）提高了 2 倍。[@_arohan_](https://twitter.com/_arohan_/status/1841904919772856631) 提到 Flash 8B 结合了算法效率的改进，以便在小型化形态中尽可能多地打包功能。[@bfl_ml](https://twitter.com/togethercompute/status/1841856799613600233) 推出了 FLUX1.1 [pro]，这是一款全新的 state-of-the-art 扩散模型，其出图速度比前代快 3 倍，且质量有所提升。

**AI 研究与技术**

- **Scaling Laws 与模型训练**：[@soumithchintala](https://twitter.com/soumithchintala/status/1841931462427476431) 讨论了现代 Transformer 如何遵循良好的 Scaling Laws，使研究人员能够在较小规模上找到超参数（hyperparameters），然后根据幂律（power laws）扩展参数和数据。这种方法增加了对更大规模训练运行的信心。

- **推理优化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1841854984142336460) 分享了 Transformer 推理优化技术的总结，包括 KV Cache、MQA/GQA、Sliding Window Attention、Linear Attention、FlashAttention、Ring Attention 和 PagedAttention。

- **AI 安全与对齐**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1841968612250419464) 对过度关注 AI 安全而牺牲神经网络、深度学习和 Agent 基础等潜在突破性研究表示沮丧。

**行业趋势与应用**

- **语音 AI 与呼叫中心**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1841833425432449066) 强调了 OpenAI 的 Real-time API 对呼叫中心行业的潜在影响，AI 驱动的通话成本显著低于人工 Agent。

- **AI 在医疗保健领域**：[@BorisMPower](https://twitter.com/BorisMPower/status/1841936047858672066) 指出，在针对专业医生的狭窄测试中，AI 的表现优于“人类 + AI”，这与在国际象棋和围棋中观察到的现象相似。

- **开发者工具与界面**：多条推文讨论了新型 AI 界面的重要性，[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1841907725095338279) 指出更好的界面将使 LLM 更易于使用，并以 Cursor 与 Copilot 为例。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Whisper Turbo：语音识别速度的显著提升**

- **OpenAI 的新 Whisper Turbo 模型在 M1 Pro 上本地运行速度比 Whisper V3 Large 快 5.4 倍** ([评分: 80, 评论: 15](https://reddit.com//r/LocalLLaMA/comments/1fvb83n/open_ais_new_whisper_turbo_model_runs_54_times/))：OpenAI 的新 **Whisper Turbo** 模型展示了与 **Whisper V3 Large** 相比，在 **M1 Pro MacBook Pro** 上的本地转录速度快了 **5.4 倍**，处理一个 **66 秒**的音频文件仅需 **24 秒**，而后者需要 **130 秒**。该帖子提供了使用 [nexa-sdk python package](https://github.com/NexaAI/nexa-sdk?tab=readme-ov-file#python-package) 进行本地测试的说明，并包含了 nexaai.com 上 **Whisper-V3-Large-Turbo** 和 **Whisper-V3-Large** 模型的链接。
  - 在 **RTX3090** Linux 系统上，**Faster-Whisper** 的表现优于 **Whisper-Turbo**，转录一个 **24:55** 的音频文件用时 **14 秒**，而后者为 **23 秒**。对于优先考虑转录速度和长音频文件的情况，建议使用分块算法（chunked algorithm）。
  - 用户报告 **Whisper Turbo** 在 MacBook 上的运行速度快于实时，为**本地实时助手解决方案**开启了可能性。该模型支持多种语言，不仅限于英语。
  - 关于 Whisper 等 ASR 模型的**流式输入/输出**讨论强调了由于其 **30 秒分块架构（30-second chunk architecture）**带来的挑战。目前存在一个工作原型，但与非异步架构相比可靠性较低。

- **终于有了一个用户友好的 Whisper 转录应用：SoftWhisper** ([Score: 62, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fvncqc/finally_a_userfriendly_whisper_transcription_app/)): **SoftWhisper** 是一款全新的 **Whisper AI** 转录桌面应用，提供直观的界面，功能包括**内置媒体播放器**、**说话人日志 (speaker diarization)**（使用 **Hugging Face API**）、**SRT 字幕创建**以及处理长文件的能力。该应用使用 **Python** 和 **Tkinter** 开发，旨在让转录变得触手可及，开发者正在寻求反馈和潜在合作伙伴，以进行 **GPU 优化**等未来改进。
  - 用户讨论了**运行该应用程序**的方法，开发者提供了**教程**和 **dependency_installer.bat** 脚本以简化设置。该项目现在包含 **requirements.txt** 文件和 **Python 安装**说明。
  - 一位用户分享了一个用于使用 **Pyannote** 进行**离线说话人日志**的 [GitHub 仓库](https://github.com/rmusser01/tldw/blob/main/App_Function_Libraries/Audio/Diarization_Lib.py)，开发者表示有兴趣探索。[Pyannote 的离线使用](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb)被确认是允许的。
  - 对未来改进的建议包括会议的**实时捕获能力**以及对**多音频流视频**的支持。开发者确认 **SoftWhisper** 可以通过提取音频来转录视频格式，尽管格式支持可能有限。


**主题 2. Qwen 2.5：保守行业对中国 AI 模型的争议**

- **[Gemma 2 2b-it 是一个被低估的 SLM GOAT]** ([Score: 92, Comments: 21](https://i.redd.it/18x465phhnsd1.png)): **Gemma 2 2b-it** 被赞誉为卓越的 **Small Language Model (SLM)**，在各种基准测试中表现优于许多大型模型。尽管其参数规模仅为 **20 亿 (2 billion parameters)**，该模型仍展示了令人印象深刻的能力，包括**零样本推理 (zero-shot reasoning)**、**少样本学习 (few-shot learning)** 以及在**编程任务**中的强劲表现。其效率和性能使其成为 SLM 领域的有力竞争者，挑战了像 **Mistral 7B** 和 **Llama 2 13B** 这样的大型模型。
  - 有人建议为 **Small Language Models (SLMs)** 设立单独的**排行榜**，并认为在智能手机上运行**本地 AGI** 具有潜力。然而，关于 "SLM" 一词引发了争论，一些人认为模型大小并不能定义它是否属于大型或小型语言模型。
  - **Qwen2.5-3B-Instruct** 模型与 **Gemma2-2B-IT** 和 **Phi3.5-mini-Instruct** 等其他小型模型相比，表现出令人印象深刻的性能。分享了一份详细的性能对比表，突出了 Qwen 在 **MATH** (65.9%) 和 **GSM8K** (86.7%) 等任务中的优势。
  - **Gemma 2 2b-it** 的能力受到称赞，用户注意到它在面对 **Claude 2** 和 **Gemini 1 Pro** 等较旧的大型模型时的表现。该模型的效率和低廉的微调成本也受到了关注。


- **Qwen 2.5 = 中国 = 糟糕** ([Score: 300, Comments: 232](https://reddit.com//r/LocalLLaMA/comments/1fv37i1/qwen_25_china_bad/)): 该帖子讨论了在**保守行业**中使用**中国 AI 模型 Qwen 2.5** 的担忧，由于担心它是来自**阿里巴巴**的**木马 (trojan)**，上级拒绝使用它。作者认为这些担忧是没有根据的，特别是考虑到计划在没有互联网连接的情况下**本地部署 (on-premise)** 使用该模型并对其进行**微调 (finetune)**，这可能使其与原始形式完全不同。
  - 用户讨论了 LLM 潜在的**安全风险**，包括可以绕过安全训练持续存在的**潜伏特工 (sleeper agents)**，以及被训练在特定条件下插入**可利用代码**的模型。一些人认为**物理隔离 (air-gapping)** 和使用 **safetensors** 格式可以减轻风险。
  - 几位评论者指出，虽然**技术风险**可能较低，但**感知风险**会对业务产生实际影响，包括对**风险评估**、**保险费**和**投资者关系**的影响。一些人建议使用替代模型以避免这些问题。
  - 关于对 **Qwen** 等**中国模型**的担忧是否合理存在争论。一些人认为它的风险并不比其他中国制造的技术产品高，而另一些人则引用了**中国间谍活动**的例子，并建议在处理敏感数据或应用时保持谨慎。


**主题 3. XTC Sampler：减少 LLM 输出中 GPT 风格用语 (GPTisms) 的新技术**

- **[告别 GPTisms 和废话 (slop)！为 llama.cpp 开发的 XTC sampler](https://github.com/cyan2k/llama.cpp/tree/feature/xtc-sampler)** ([Score: 144, Comments: 45](https://reddit.com//r/LocalLLaMA/comments/1fv5kos/say_goodbye_to_gptisms_and_slop_xtc_sampler_for/)): 该帖子介绍了一个为 **llama.cpp** 实现的 **XTC sampler**，旨在减少语言模型输出中的 **GPTisms** 和**废话 (slop)**。这种采样方法通过解决与大型语言模型中使用的传统采样技术相关的常见问题，旨在提高生成文本的质量和连贯性。
  - 为 **llama.cpp** 实现的 **XTC sampler** 旨在通过在采样过程中忽略 top tokens 来减少 **GPTisms** 并提高创造力。用户可以在 [GitHub 仓库](https://github.com/cyan2k/llama.cpp/tree/feature/xtc-sampler/xtc-examples)中找到示例和使用说明。
  - 关于 XTC 有效性的讨论随之展开，一些用户称赞其增强创意写作的能力，而另一些用户则质疑其对通用性能的影响。推荐的参数值为 **threshold = 0.1** 和 **probability = 0.5**，可行范围为 threshold **0.05-0.2**，probability **0.3-1.0**。
  - 辩论围绕移除 top token 候选者是否是改进语言模型输出的最佳方法展开。一些人认为这可能导致非创意任务的性能下降，而另一些人则强调其在减少重复短语和增强生成文本多样性方面的潜力。


- **[量化测试：看看 Aphrodite Engine 的自定义 FPx 量化是否好用](https://www.reddit.com/gallery/1fv2bqp)** ([Score: 64, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1fv2bqp/quantization_testing_to_see_if_aphrodite_engines/)): **Aphrodite Engine** 的自定义 **FPx 量化**与标准 **FP16** 和 **INT8** 量化方法进行了对比测试。结果显示，**FPx** 的表现优于 **INT8**，并达到或略微超过了 **FP16** 的性能，同时提供了潜在的内存节省。测试使用了 **MMLU** 和 **HumanEval** 基准测试，并计划使用 **TinyStories** 和 **Alpaca** 数据集进行进一步评估。
  - **Aphrodite** 的自定义 **FP 量化**展示了令人印象深刻的结果，推荐将 **FP6** 用于 <8-bit 的快速推理。**FP5** 出人意料地获得了最高分 (40.61%)，这可能是由于无意中触发了**思维链 (Chain of Thought)** 推理。
  - 基准测试结果显示 **GGUF Q4_K_M** 表现出奇地好，优于 **GPTQ** 和 **FP4** 量化。**Aphrodite** 的 **FP 量化**展示了极高的速度，在较低量化级别下扩展速度更快，而 **GGUF 模型**明显较慢。
  - 研究结论认为，使用 **Aphrodite 的自定义 FP 量化**进行 **>4-bit 量化**是速度最优的选择。对于 4-bit 或更低的量化，**GGUF** 表现更好。8-bit 量化在各种方法中显示出与完整 **BF16** 模型相似的性能。


**主题 4. 开源 LLM 中的工具调用：构建 Agentic AI 系统**

- **LLM 工具调用：入门指南** ([Score: 73, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fvdtqk/tool_calling_in_llms_an_introductory_guide/)): 该帖子介绍了 **LLM 中的工具调用**，将工具定义为提供给语言模型的具有**名称**、**参数**和**描述**的函数。它解释了 LLM 并不直接执行工具，而是在识别出与给定查询相关的工具时，生成一个包含工具名称和参数值的**结构化模式 (structured schema)**（通常是一个 **JSON 对象**）。该帖子概述了工具调用的**四步工作流**，从定义工具到使用工具输出生成完整答案，并提供了一个关于在开源 **Llama 3** 中使用 **Agent** 进行工具调用的深入指南链接。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与技术**

- **Google DeepMind 推进多模态学习**：一篇[新论文](https://arxiv.org/html/2406.17711v1)展示了通过联合样本选择（joint example selection）进行数据策展如何加速多模态学习。

- **Microsoft 的 MInference 加速长上下文推理**：[MInference](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，为长上下文任务实现高达数百万个 token 的推理。

- **扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 10 亿个从网络策展的角色（personas）来生成多样化的训练数据。

- **NeRFs 的精确体渲染（Exact volume rendering）**：一篇[新论文](https://www.reddit.com/r/singularity/comments/1fvfhfj/new_paper_performs_exact_volume_rendering_at/)在 30FPS@720p 下实现了精确体渲染，生成了高度详细且 3D 一致的 NeRFs。

**AI 模型发布与改进**

- **Salesforce 发布 xLAM-1b**：这个拥有 10 亿参数的模型[在函数调用（function calling）方面达到了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **Phi-3 Mini 更新函数调用功能**：Rubra AI 发布了更新后的 Phi-3 Mini 模型，[具备函数调用能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。

- **适用于 Flux 的 iPhone 照片风格 LoRA**：一个新的 [LoRA 微调](https://www.reddit.com/r/StableDiffusion/comments/1fvpptg/iphone_photo_stye_lora_for_flux/)提升了 Stable Diffusion Flux 输出的逼真度，使其匹配 iPhone 的照片美学。

**AI 行业动态**

- **Nvidia Blackwell AI 芯片需求旺盛**：Nvidia CEO Jensen Huang 报告称，各大科技公司对其下一代 AI 芯片的需求[“非常疯狂”](https://www.forbes.com/sites/antoniopequenoiv/2024/10/03/nvidia-shares-jump-after-ceo-jensen-huang-notes-insane-demand-for-blackwell-ai-superchip/)。

- **OpenAI 劝阻投资者支持竞争对手**：OpenAI [要求投资者不要资助某些 AI 竞争对手](https://www.reddit.com/r/singularity/comments/1fv86bx/the_vibes_are_off/)，引发了对垄断行为的担忧。

- **Sora 负责人加入 Google**：OpenAI Sora 视频生成模型的首席研究员 Tim Brooks [已加入 Google](https://www.reddit.com/r/singularity/comments/1fvliry/sora_lead_tim_brooks_joins_google/)。

**AI 伦理与社会影响**

- **关于 AI 对齐（AI alignment）与企业控制的辩论**：围绕 [OpenAI 转向营利模式](https://www.reddit.com/r/singularity/comments/1fv86bx/the_vibes_are_off/)的讨论，以及对企业控制 AGI 开发的担忧。

- **欧盟 AI 监管担忧**：法国总统 Macron [警告称，对 AI 的过度监管和投资不足](https://www.reddit.com/r/singularity/comments/1fvfq9o/the_eu_could_die_warns_macron_on_overregulation/)可能会损害欧盟的竞争力。

- **工会与 AI 采用**：瑞典工会领导人关于[拥抱新技术](https://www.reddit.com/r/singularity/comments/1fv47p3/swedens_union_leaders_views_on_new_technology/)同时保护劳动者的观点，强调了重新培训和适应的必要性。

**AI 能力与里程碑**

- **人类水平推理能力的宣称**：OpenAI CEO Sam Altman 暗示他们已经[达到了人类水平的推理能力](https://www.reddit.com/r/singularity/comments/1fvd7uv/altman_we_just_reached_humanlevel_reasoning/)，尽管其确切含义和影响仍存争议。

- **图像生成能力的提升**：展示了使用 Stable Diffusion Flux 生成的[高度写实照片](https://www.reddit.com/r/StableDiffusion/comments/1fvs0e1/ultra_realistic_photos_on_flux_just_by_adding_img/)，尽管部分说法存在争议。


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要

**主题 1：Meta 发布 Movie Gen，彻底改变视频生成领域**

- **Meta 首映 Movie Gen，重新定义多媒体创作**：[Meta 的 Movie Gen](https://ai.meta.com/research/movie-gen) 推出了先进的模型，可以根据文本提示生成高质量的图像、视频和同步音频。其功能包括精确的视频编辑和个性化内容生成。
- **AI 社区对 Movie Gen 的潜力议论纷纷**：[Movie Gen 研究论文](https://ai.meta.com/static-resource/movie-gen-research-paper)展示了视频内容创作中的突破性技术。Meta 正在与创意人士合作，在广泛发布前进一步完善该工具。
- **Movie Gen 在各大 AI 论坛引发热议**：讨论强调了 Movie Gen 在突破 AI 生成视频边界方面的潜力，爱好者们渴望探索其在多媒体项目中的应用。

**主题 2：新 AI 模型与基准测试引领潮流**

- **Nvidia 发布重磅消息，推出 GPT-4 竞争对手**：据 [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/) 报道，Nvidia 的新 AI 模型是**开源且巨大的**，旨在挑战 GPT-4。AI 社区正拭目以待它的实际表现。
- **金融 LLM 排行榜揭晓顶尖表现者**：一份新的针对金融领域的 [LLM 排行榜](https://huggingface.co/blog/leaderboard-finbench) 对 **OpenAI 的 GPT-4**、**Meta 的 Llama 3.1** 和 **Alibaba 的 Qwen** 在 40 项任务中的表现进行了排名。这为评估金融应用中的模型提供了新的指标。
- **Gemini 1.5 Flash-8B 提供高性价比的 AI 算力**：现在已在 [OpenRouter](https://openrouter.ai/models/google/gemini-flash-1.5-8b) 上线，价格为 **每百万 token 0.0375 美元**，Gemini 1.5 Flash-8B 在不牺牲性能的前提下提供了一个极具成本效益的选择。

**主题 3：模型优化与训练技术的进展**

- **TorchAO 通过模型优化点亮 PyTorch**：新的 [torchao 库](https://pytorch.org/blog/quantization-aware-training/) 引入了量化和低位数据类型，提升了模型性能并大幅降低了内存占用。这是 PyTorch 用户迈出的重要一步。
- **SageAttention 速度超越竞争对手**：[SageAttention](https://github.com/thu-ml/SageAttention) 实现了比 FlashAttention2 快 **2.1 倍**、比 xformers 快 **2.7 倍** 的速度，且完全没有精度损失。这种量化方法极大地加速了注意力机制。
- **VinePPO 释放 LLM 中的 RL 潜力**：[VinePPO 算法](https://arxiv.org/abs/2410.01679) 解决了 LLM 推理任务中的信用分配（credit assignment）问题，其表现优于 PPO，**步骤减少了 9 倍**，**时间缩短了 3 倍**，同时内存占用减半。

**主题 4：OpenAI 的 Canvas 工具与模型引发复杂反响**

- **OpenAI 的 Canvas 工具引发喜忧参半的反响**：新的 [Canvas 工具](https://openai.com/index/introducing-canvas/) 通过集成功能和减少滚动操作简化了代码编写。然而，用户对缺失 **continue 按钮** 等基本功能表示遗憾，并遇到了一些编辑上的小问题。
- **高级语音模式（Advanced Voice Mode）可能提升编程体验**：讨论表明，将 **Advanced Voice Mode** 与 Canvas 结合可以增强编程工作流。社区分享的 [设置指南](https://github.com/jjmlovesgit/ChatGPT-Advanced-Voice-Mode) 旨在帮助用户实现平滑集成。
- **OpenAI 的 o1 模型给开发者留下深刻印象**：[o1-preview 和 o1-mini 模型](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0) 的引入增强了聊天机器人的能力。用户注意到 **o1-mini** 在处理复杂任务时表现出惊人的实力。

**主题 5：循环神经网络（RNN）强势回归**

- **RNN 以 175 倍的训练速度回归**：论文《[Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201)》揭示了没有隐藏状态依赖的 **minLSTMs** 和 **minGRUs** 训练速度显著加快，重新引发了人们对 RNN 架构的兴趣。
- **极简 RNN 实现高效并行训练**：通过消除随时间反向传播（BPTT），这些简化的 RNN 允许并行计算，在序列建模效率方面挑战 Transformer。
- **社区探索 RNN 的现代潜力**：爱好者们讨论了精简 RNN 如何实现适合当今 AI 需求的扩展训练方法，这可能会重塑神经网络架构的格局。


---

# 第 1 部分：高层级 Discord 摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **torchao 库引入模型优化**：来自 PyTorch 的 [torchao 库](https://huggingface.co/posts/singhsidhukuldeep/639926000427051) 具备量化（quantization）和低比特数据类型技术，提升了模型性能和内存利用率。
   - 它承诺在现有工具的基础上实现自动量化，标志着 PyTorch 的一项重大进步。
- **OpenAI 的 Canvas 工具简化代码编写**：OpenAI 的 [Canvas 工具](https://openai.com/index/introducing-canvas) 因其集成功能而备受关注，减少了编码过程中不必要的滚动。
   - 用户注意到其编辑能力相比 Claude 等之前的工具有了显著进步。
- **Meta 的 Movie Gen 模型展现巨大潜力**：Meta 发布了 [Movie Gen 模型](https://ai.meta.com/static-resource/movie-gen-research-paper)，可根据文本提示生成高质量的多媒体内容。
   - 这些模型具有精确的视频编辑和个性化生成功能，突显了其创意应用价值。
- **文化偏见限制 AI 训练理解**：目前的讨论指出，LLM 训练缺乏人类偏见，且过度依赖大型数据集，这影响了对爱和道德等概念的理解。
   - 成员们质疑 AI 在没有真正内在理解的情况下，如何“学习”这些复杂的情感。
- **VinePPO 解决 LLM 信用分配问题**：关于 **VinePPO** 的论文批评了 Proximal Policy Optimization (PPO) 在推理任务中的不一致性，并引入了一种改进方案来解决信用分配（credit assignment）问题。
   - 研究表明，PPO 中现有的价值网络（value networks）会产生高方差更新，表现仅勉强优于随机基准线。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **敦促 Aider 完善遥测功能**：成员们强调了 **Aider 中遥测（telemetry）** 的重要性，建议增加选择性加入（opt-in）功能以保护用户隐私，同时提高对性能的洞察。
   - 提议使用 *System call tracing* 来诊断性能问题，并强调了对收集数据保持**透明度**的必要性。
- **OpenRouter 免费模型面临测试**：**OpenRouter 的免费模型** 存在严格的全账户限制，即 **每天 200 条消息**，这影响了希望获得更多访问权限的用户的灵活性。
   - 参与者对某些模型缺乏付费选项表示担忧，质疑其整体可用性。
- **模型基准测试引发疑问**：参与者分享了 **对各种模型进行基准测试（benchmarking）** 的经验，指出在处理错误率方面的表现参差不齐。
   - Aider 处理编辑任务的能力是关注焦点，用户报告了与 Token 限制以及特定错误相关的问题。
- **Ollama 模型在 Aider 中的性能表现**：用户报告在使用 Aider 配合 **Ollama 的本地 8B 模型** 时**响应速度缓慢**，质疑付费 API keys 的益处。
   - 讨论显示本地模型在处理编辑任务时可能会遇到困难，表明用户更倾向于具有更强编辑能力的模型。
- **探索文件添加的复杂性**：在 Aider 中测试 **/read-only** 命令表明，它现在只能按文件夹完成任务，这使文件访问变得复杂。
   - 另一位用户确认，正确的使用方式仍应添加所有文件，这揭示了命令功能中的细微差别。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Salamandra 设备端演示大放异彩**：[Salamandra](https://huggingface.co/spaces/Tonic/salamandra-on-device) 演示展示了令人印象深刻的能力，在吸引用户的同时也突出了其功能特性。
   - *社区对 Salamandra 亮点的兴奋* 反映了人们对设备端 AI 应用日益增长的兴趣。
- **Nvidia 发布了一款具有变革意义的 AI 模型**：根据 [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/) 的报道，Nvidia 的新 AI 模型是**开源且巨大的**，并准备好**与 GPT-4 竞争**。社区渴望看到该模型将如何竞争以及它拥有哪些独特的能力。
   - 这一公告在 AI 社区引起了轰动。
- **OpenAI 推出新模型**：两个新的 **OpenAI** 模型 **o1-preview** 和 **o1-mini** 已集成到 [开源聊天机器人](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0) 中，增强了其功能。成员们庆祝这些新增功能，认为这是迈向更强大聊天机器人体验的重要飞跃。
- **MusicGen iOS 应用取得进展**：**MusicGen** 的 iOS 应用更新揭示了一些新功能，包括输入音频的降噪功能和针对鼓点的 'tame the gary' 开关。*一位成员评论道*，它的目标是精细化的音频输入输出集成，旨在提升用户体验。
- **AI 感知预测引发疑问**：一篇名为《感知预测方程》（The Sentience Prediction Equation）的文章讨论了未来 AI 可能产生的感知及其影响，质疑 AI 是否会思考其存在的意义。文章幽默地指出，AI 可能会问：*“为什么人类坚持要在披萨上放菠萝？”*，并引入了“预测方程”作为一种估算工具。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas 模型增强功能引发热议**：新的 **Canvas 模型** 引起了轰动，成员们正在讨论其潜在功能以及与 [GPT-4o](https://openai.com/index/introducing-canvas/) 的集成。然而，由于缺少 **continue 按钮** 和编辑问题，也出现了一些挫败感。
   - 用户希望改进能提升编程任务的 UX，同时解决当前的局限性。
- **Advanced Voice Mode 可能促进集成**：关于 **Advanced Voice Mode** 的讨论强调了它与 **Canvas 工具** 的潜在协同作用，从而在编码中提供更流畅的用户体验。社区成员在 GitHub 上传阅设置指南，以帮助实现无缝集成。
   - 他们提议将实时 API 集成作为令人兴奋的下一步，以提高编码效率。
- **Custom GPTs 体验反馈不一**：用户报告了在 **Custom GPTs** 初始推出期间集成 **Google API/OAuth** 时遇到的挑战，引发了对其可靠性的一些担忧。他们尚未查看最近关于稳定性的改进情况。
   - 这种缺乏一致性的情况让一些用户对重新尝试该集成持谨慎态度。
- **ChatGPT 评估的不一致性成为焦点**：当任务是在 **temperature 0.7** 下对答案进行评分时，**ChatGPT** 评估的不一致性引发了不满，促使人们建议采用更严格的评分标准。一位用户建议使用 **评分量规 (grading rubric)** 来提高清晰度和一致性。
   - 另一位用户提议使用 **Chain-of-Thought** 推理框架来提高评分准确性和评估清晰度。
- **分享高效 JSON 处理技巧**：一位开发者寻求关于使用 **GPT-4o** 将 10,000 个代码片段解析为 JSON 的建议，并询问是否需要为每个片段重新发送协议参数。建议鼓励通过在处理过程中仅发送新片段来进行优化。
   - 这次对话说明了在模型交互和 JSON 处理中对成本效率的持续需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 项目简化了微调流程**：成员们讨论了使用 **Unsloth AI** 进行 LLM 的持续预训练（continual pretraining），与传统方法相比，训练速度提升了 **2 倍**，同时 **VRAM** 占用减少了 **50%**。
   - 强调了像 [持续预训练 notebook](https://docs.unsloth.ai/basics/continued-pretraining) 这样的必备工具对于扩展模型训练能力的重要性。
- **ZLUDA 的融资带来新希望**：ZLUDA 的开发已获得一家新商业实体的支持，目标是增强 LLM 的功能。
   - 法律纠纷的担忧依然存在，特别是与 **NVIDIA** 之间可能产生的冲突，这呼应了以往股权支持案例中遇到的问题。
- **代际偏好：一个幽默的视角**：成员们戏谑地争论起自己的代际身份，有人声称年仅 24 岁就感觉像个 **boomer**（老顽固），触及了文化认知的话题。
   - 这场轻松的对话指出，**Legos** 和 **modded Minecraft**（模组化我的世界）定义了代际界限，暗示了文化习俗的转变。
- **本地推理脚本的烦恼**：一位成员在使用 **llama-cpp** 运行 **gguf models** 的本地推理脚本时遇到挑战，反映尽管 GPU 性能强劲，但表现依然迟缓。
   - 诸如使用 **llama-cli** 等建议被提出，表明了提升脚本效率的潜力。
- **循环神经网络（RNNs）的复兴**：最近的一篇论文建议，通过消除隐藏状态依赖，**minimal LSTMs** 和 **GRUs** 的训练速度可以快 **175 倍**，引发了对 RNN 的重新关注。
   - 这一发现指向了与现代架构相关的可扩展训练方法的新可能性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **IREE 面临不可预测的采用时间表**：成员们讨论了大型实验室是否会采用 **IREE** 进行大规模模型服务，迹象表明许多实验室仍在使用自定义推理运行时（inference runtimes）。
   - 一些人指出，像 IREE 这样的新技术拥有**不可预测**的采用时间表是很正常的。
- **RWKV 引入高效并行化**：**RWKV** 通过将网络结构化为更小的层来采用部分并行化，从而能够在等待 Token 输入时进行计算。
   - 这种方法旨在优化性能，同时有效地管理模型间的相互依赖。
- **探索线性注意力（Linear Attention）模型**：对话集中在线性注意力和门控线性注意力作为 RNN 运行的能力上，这使得跨序列的并行计算成为可能。
   - 随着 **Songlin Yang** 的研究揭示了能够提高并行化程度的复杂 RNN 类别，人们的兴趣日益浓厚。
- **VinePPO 在信用分配（credit assignment）上挣扎**：**VinePPO** 论文概述了价值网络在复杂推理任务中面临的信用分配挑战，其表现甚至低于随机基准线。
   - 这强调了在 **Proximal Policy Optimization (PPO)** 中需要改进模型或技术来优化信用分配。
- **lm-evaluation-harness 寻求贡献者**：**lm-evaluation-harness** 正在邀请贡献者集成新的 LLM 评估并修复 Bug，目前有许多待处理的问题。
   - 潜在的贡献者可以在 [GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness) 中找到更多详细信息。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **SambaNova AI 的吞吐量表现令人印象深刻**：[SambaNova AI](https://x.com/SambaNovaAI/status/1841901026821210131) 在 OpenRouter 上推出了 **Llama 3.1 和 3.2** 的端点，声称拥有记录以来最快的吞吐量测量结果。
   - 他们指出，“这是我们见过的最快速度”，表明其吞吐量指标与竞争对手相比具有显著优势。
- **Gemini 1.5 Flash-8B 正式发布**：**Gemini 1.5 Flash-8B** 模型现已上线，价格为 **每百万 Token 0.0375 美元**，使其成为与同类产品相比值得关注的预算选择。
   - 如需访问，请查看[此处](https://openrouter.ai/models/google/gemini-flash-1.5-8b)的链接；讨论还集中在其性能扩展潜力上。
- **o1 Mini 在任务性能上带来惊喜**：**o1 Mini** 在解决复杂任务方面表现出更强的能力，超出了社区对其性能的预期。
   - 一位成员提到计划将 **o1 Mini** 用于处理图像描述的机器人，展示了其在实际应用中的潜力。
- **Anthropic 乘着融资浪潮前进**：讨论透露，**Anthropic** 快速的模型开发（特别是 **Claude**）源于一支前 OpenAI 工程师团队以及来自 **Amazon** 的支持。
   - 针对 Anthropic 如何在资金支持少于行业巨头的情况下，在性能上保持有效竞争，出现了一些推测。
- **OpenRouter 基础设施扩展指日可待**：人们对 **OpenRouter** 的扩展充满期待，以适应包括图像和音频处理在内的多样化模型功能。
   - 开发负责人确认正积极致力于升级，以应对不断增长的流量和新模型的发布。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Langflow 集成助力 LM Studio**：LM Studio 正在集成对 Langflow 的支持，正如[最近的 GitHub Pull Request](https://github.com/langflow-ai/langflow/pull/4021)所强调的那样，这增强了构建 LLM 应用程序的功能。
   - 此次集成旨在简化用户体验并扩展 LM Studio 的能力。
- **v0.3.2.6 版本的内存泄漏风波**：用户报告了 LM Studio **v0.3.2.6** 版本的严重内存泄漏问题，导致模型生成无意义的输出。
   - 建议检查 **v0.3.3** 版本是否已解决该问题。
- **模型下载问题引发错误**：从 Hugging Face 下载模型时出现持续性问题，在 LM Studio 中选择模型时会发生错误。
   - 成员建议直接将模型[侧加载 (Sideloading)](https://lmstudio.ai/docs/advanced/sideload)到模型目录中以绕过这些错误。
- **聊天缓存位置不可自定义**：关于在 LM Studio 中自定义聊天缓存位置的问题被提出，目前该位置是硬编码的。
   - LM Studio 以 JSON 格式保存对话数据，但目前没有更改缓存位置的选项。
- **AI 模型推荐引发讨论**：讨论强调 **Llama-3-8B** 作为聊天机器人助手时未能达到部分用户的预期。
   - 鼓励用户在 [LM Studio Model Catalog](https://lmstudio.ai/models) 上探索各种选项，以寻找可能更合适的模型。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LangChain 发布 Voice ReAct Agent**：LangChain 推出了 **Voice ReAct Agent**，利用 [Realtime API](https://x.com/langchainai/status/1841914757764485439?s=46) 提供定制化语音体验，并演示了一个使用计算器和 [Tavily web search tool](https://youtu.be/TdZtr1nrhJg) 的 Agent。
   - 这一创新的 Agent 展示了交互式应用中语音交互的新可能性。
- **GPT-4o 机器人热聊**：一个演示展示了两个 **GPT-4o Voice AI 机器人** 使用 Realtime API 进行对话，凸显了语音 AI 技术的进步。
   - 机器人表现出令人印象深刻的 *轮替延迟 (turn-taking latency)*，显示出交互流畅度的显著提升。
- **Meta Movie Gen 进军视频生成**：Meta 展示了其最新项目 **Meta Movie Gen**，旨在开拓 **视频生成** 领域，但尚未确定发布日期。更多详情可以在其 [AI 研究页面](https://ai.meta.com/research/movie-gen/) 和 [相关论文](https://ai.meta.com/static-resource/movie-gen-research-paper) 中探索。
   - 该项目承诺在最先进模型的驱动下，突破视频内容创作的边界。
- **新 LLM 排行榜引入金融领域领先者**：最新的金融领域 **LLM 排行榜** 将 **OpenAI 的 GPT-4**、**Meta 的 Llama 3.1** 和 **阿里巴巴的 Qwen** 列为 40 项相关任务中的佼佼者，详见 [Hugging Face 博客文章](https://huggingface.co/blog/leaderboard-finbench)。
   - 这种评估方法为衡量模型在金融应用中的性能提供了一种新颖的方式。
- **Luma AI 激发 3D 建模兴趣**：关于 **Luma AI** 的热烈讨论强调了其在为 Unity 和 Unreal 等平台创建逼真 3D 模型方面的潜力，成员们分享了各种功能展示。
   - Luma AI 的能力在其电影编辑和精细 3D 模型应用中得到凸显，预示着其在创意科技领域的广阔前景。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **性能基准测试咨询**：成员们正在寻求工具和方法的 **性能基准测试 (performance benchmarks)**，特别是将这些指标与 **fio 工具** 的原始性能进行对比。
   - 目前正致力于分析 **数据访问方法**，以了解其相对于传统性能指标的有效性。
- **OpenAI 的财务成功**：据报道，得益于近期的创新，**OpenAI** 正在刷新财务记录，并有推测称其正在开发硬件以利用这一增长。
   - 围绕 **新产品开发** 的讨论日益增多，指向了开发专注于用户数据应用的移动设备的可能性，这让人联想到 **Apple** 的隐私考量。
- **活动策划策略**：活动规划时间表建议可能在 **9月** 左右举行，以配合开学季来提高出席率。
   - 提议与 **Triton** 和 **PyTorch** 会议同地举办，以便于团体旅行，展示了有效的策划策略。
- **Triton Kernel 挑战**：用户正在排查 **Triton kernels** 的故障，特别是面临非连续输入的问题，这表明可能需要进行 **reshape**。
   - 此外，**OptimState8bit** 调度错误问题依然存在，凸显了 8-bit 优化器实现的局限性。
- **需要超参数缩放指南**：一位成员呼吁制定 **超参数缩放指南 (hyperparameter scaling guide)**，表示由于缺乏针对大型模型训练的清晰启发式方法而感到困惑。
   - 对训练方法的担忧表明，在支持社区成员这一技术领域方面，可获取的资源存在缺口。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 更新 Collections UI**：Perplexity AI 正在增强其 Collections 功能，采用新的 UI 以支持 **custom instructions** 和文件上传，计划于未来部署。
   - 即将推出的 **Files search feature** 旨在改进信息组织和用户体验。
- **Boeing 777-300ER 规格发布**：分享了 **Boeing 777-300ER** 规格的详细大纲，涵盖尺寸、性能和载客量。
   - 关键亮点包括 **7,370 海里** 的 **最大航程** 以及最多可容纳 **550 名乘客** 的潜力。
- **TradingView Premium 破解版披露**：流传出一个免费的 **TradingView Premium**（版本 2.9）破解版，提供无需付费的高级交易工具。
   - 这一披露引起了寻求改进图表功能的交易者的兴趣。
- **Llama 3.2 发布备受期待**：用户对 **Llama 3.2** 的预期功能和发布日期议论纷纷，对其进展表现出浓厚兴趣。
   - 社区对这一新迭代可能带来的潜在创新感到兴奋。
- **Claude 3.5 表现优于竞争对手**：出现了将 **Claude 3.5 Sonnet** 与其他模型进行比较的讨论，许多人断言其在信息检索方面的可靠性。
   - 成员们强调了 **Perplexity Pro** 与 Claude 协同工作以改进资源数据提取的效果。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R 08-2024 微调亮点**：更新后的 *Command R 08-2024* 引入了对新选项的支持，旨在为用户提供 **更多控制** 和 **可见性**。此次更新的特点是与 [Weights & Biases](https://cohere.com/blog/fine-tuning-command0824) **无缝集成**，以增强性能跟踪。
   - 成员们对 Command R 的更新表示 **热烈欢迎**，诸如 “*Awesome*” 之类的评论捕捉到了社区的兴奋和期待。
- **平台指标缺失**：一位用户报告称，他们无法在 Overview 和 API 等各个选项卡中看到其模型的 **metrics boxes**（指标框），而这些选项卡以前会显示基本信息。他们强调该问题已持续 **2 天** 未解决。
   - 这引发了对平台一致性的担忧，并对模型创建的状态提出了质疑。
- **价格页面困惑**：**价格页面** 显示训练费用为 **每 1M tokens 3 美元**，但微调 UI 显示的价格为 **8 美元**。这种差异引发了对不同平台定价信息准确性的质疑。
   - 这造成了困惑，可能会影响用户对训练和微调项目的预算编制。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **寻找 OpenPose 替代方案**：用户表达了对 **OpenPose** 在生成坐姿时的挫败感，引发了对 **DWPose** 等替代方案的讨论，并探索了自定义模型训练选项。
   - *如果有足够的参考图像，训练自己的模型也不失为一个可行的解决方案。*
- **提高 ComfyUI 的图像质量**：一位成员提出了关于如何使 **ComfyUI** 输出达到与 **Auto1111** 相当水平的问题，因为最近的图像质量看起来像卡通。
   - *推荐了 ComfyUI 中的特定节点作为获得更好质量输出的潜在方法。*
- **澄清 SDXL 模型变体**：讨论了多个版本的 **SDXL**，特别是 `SDXL 1.0`，涵盖了从 **1024x1024** 开始的分辨率等方面。
   - *参与者确认所有变体都与 **SDXL 1.0** 模型框架相关。*
- **参考图像生成姿势**：已确认在 **Stable Diffusion** 中使用单张参考图像生成姿势是可行的，尽管准确性可能会受到影响。
   - *强调了 **img2img** 功能是正确的方法，并建议多张参考图像将提高保真度。*
- **AI 物品放置工具查询**：讨论发现了对 **OpenPose** 技术的兴趣，以协助物品放置，特别是关于剑等物品的 LoRA 模型。
   - *虽然 Stable Diffusion 中存在各种训练风格，但用户注意到在专用姿势方法方面存在空白。*

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **MinGRU 架构让循环网络更精简**：**minGRUs** 的引入提出了一种更简单的 **GRUs** 形式，消除了隐藏状态依赖，将训练速度提升了 **175 倍**。
   - 该论文强调，只需两个线性层即可实现并行隐藏状态计算，引发了关于简化 **NLP** 架构的讨论。
- **寻找构建 BARK 模型的资源**：一位新手渴望在 **2-3 个月** 内从头开始训练一个 **类似 BARK 的模型**，但难以找到相关文献。
   - 他们注意到 BARK 与 **Audio LM** 和 **VALL-E** 等模型之间的联系，寻求社区建议以获取论文来指导其训练工作。
- **应对技术领域的语言挑战**：一位成员对技术讨论中 **English** 的主导地位表示担忧，指出许多复杂术语（如 **embeddings** 和 **transformers**）往往缺乏直接的翻译。
   - *对语言偏好的沮丧*使技术讨论变得复杂，因为有效的沟通取决于共享的术语。
- **社区诈骗警报提醒成员保持警惕**：出现了大量关于潜在诈骗的警告，这些诈骗以虚假承诺诱导成员，声称在 **72 小时** 内赚取 **5 万美元** 即可获得 10% 的利润分成。
   - 建议个人对此类方案保持怀疑，特别是那些涉及未经请求的 **Telegram** 外联。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **文章评分查询引发关注**：一位成员询问如何查看他们提交的三篇文章的评分，包括草稿和 LinkedIn 链接，这凸显了对提交反馈的持续关注。
   - *提交反馈仍然是成员们的热门话题*，他们寻求关于自己贡献的明确说明。
- **实时流式传输受垃圾回收阻碍**：一位成员表示希望将 **chat_manager** 的响应实时直接流式传输到前端，并指出目前的响应仅在垃圾回收后才进行流式传输。
   - 另一位成员确认大约 8 个月前已经创建了一个 **Streamlit UI**，解决了这一挑战。
- **Chainlit 在对话管理方面展现潜力**：一位成员指出存在使用 **Chainlit** 的解决方案，在 **GitHub** 上的 **AutoGen** 项目中可能有一个方案可以促进实时聊天功能。
   - 这一实现可以有效解决正在进行的讨论中强调的改进对话管理的需求。
- **GitHub Pull Request 聊天处理见解**：一位成员分享了一个相关的 [GitHub pull request](https://github.com/microsoft/autogen/pull/1783)，该 PR 专注于在发送消息之前对其进行处理，从而增强了自定义功能。
   - 这一进展与之前关于实时流式传输的咨询相一致，显示了社区在改进功能方面的动力。
- **校园课程地点已明确**：一位成员询问了伯克利校园内某门课程的具体教室，强调了参与者对物流安排的关注。
   - 随着社区成员处理他们的教育要求，*协调活动*似乎至关重要。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaCloud 构建 AI Agents**：了解如何使用 [LlamaCloud 和 Qdrant Engine](https://twitter.com/llama_index/status/1841935964081627315) 构建 AI Agents，重点在于实现 **semantic caching** 以提升速度和效率。
   - 该演示包含了 **query routing** 和 **query decomposition** 等高级技术，以优化 Agent 交互。
- **增强 RAG 部署的安全性**：一场关于利用 [Box 的企业级安全特性](https://twitter.com/llama_index/status/1841950022835044833) 结合 LlamaIndex 进行安全 RAG 实现的讨论引发关注。
   - 成员们强调了 **permission-aware RAG** 体验对于确保稳健数据处理的重要性。
- **与 OpenAI API 的语音交互**：Marcus 展示了一个使用 [OpenAI 实时音频 API](https://twitter.com/llama_index/status/1842236491784982982) 的新功能，支持通过语音命令进行文档聊天。
   - 该功能彻底改变了文档交互方式，允许用户通过口语进行交流。
- **对抗 RAG 中的幻觉**：[CleanlabAI 的解决方案](https://twitter.com/llama_index/status/1842259131274817739) 通过为 LLM 输出实现信任度评分系统，解决了 RAG 中的幻觉问题。
   - 该方法通过识别并移除不可靠的响应来提升数据质量。
- **宣布令人兴奋的黑客松机会**：即将举行的黑客松提供超过 **12,000 美元的现金奖励**，将于 10 月 11 日在帕洛阿尔托的 [500 Global VC 总部](https://twitter.com/llama_index/status/1842274685989576947) 拉开帷幕。
   - 参与者将有机会在整个周末争夺丰厚现金奖励的同时，创作创新项目。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **dslmodel 实时演示排期**：**dslmodel** 的交互式编程实时演示将于 **PST 4:30** 举行，欢迎在 coding lounge 参与。
   - 这些演示旨在展示实时应用以及用户对 dslmodel 功能的参与。
- **情感分析结果令人印象深刻**：**SentimentModel** 准确地将短语 "This is a wonderful experience!" 分类为 **sentiment='positive'**，置信度为 **1.0**。
   - 这突显了其在情感分类任务中的有效性，为用户提供可靠的结果。
- **摘要模型有效捕捉主题**：使用 **SummarizationModel**，文档的关键信息被提炼为：“**关于成功与坚持的励志演讲。**”
   - 该模型有效地识别了控制、成功和韧性等主题，展示了其在摘要任务中的能力。
- **DSPy 解读其缩写含义**：成员们澄清 **DSPy** 代表 **Declarative Self-improving Language Programs**，也被戏称为 **Declarative Self-Improving Python**。
   - 这次对话展示了社区的参与度和在解读 DSPy 缩写时的幽默感。
- **DSPy Signatures 详解**：一位用户分享了关于 **DSPy Signatures** 的细节，强调它们作为模块输入/输出行为的声明式规范的作用。
   - 这些 Signatures 提供了一种结构化的方式来定义和管理模块交互，与标准的函数签名有所不同。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **活动参与人数限制回退至 25 人**：成员们注意到活动的参与人数上限被设为 **25 人**，尽管 MikeBirdTech 曾提议将其更改为 **99 人**。
   - 一位用户确认多次尝试加入，但仍然遇到“已满 (**full**)”状态。
- **加入 Human Devices 活动**：MikeBirdTech 分享了即将举行的 **Human Devices 活动**链接：[点击加入](https://discord.gg/mzcrk6pZ?event=1291393902758330389)。
   - 鼓励参与者在指定频道中**请求或分享**与活动相关的任何内容。
- **Obelisk：一个实用的 GitHub 工具**：一位成员重点介绍了来自 **GitHub** 的 **Obelisk** 项目，这是一个将网页保存为单个 **HTML 文件**的工具。
   - 他们建议它在**许多场景下都非常有用**，并提供了探索链接：[GitHub - go-shiori/obelisk](https://github.com/go-shiori/obelisk)。
- **Meta Movie Gen 发布**：今天，[Meta 首映了 Movie Gen](https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)，这是一套旨在增强视频和音频创作的高级媒体基础模型。
   - 这些模型可以生成高质量的图像、视频和同步音频，具有令人印象深刻的对齐效果和质量。
- **Mozilla 的开源愿景**：在关于 **Meta Movie Gen** 开放性的讨论中，一位成员澄清说，虽然 **Mozilla** 提倡开源，但这一举措更多是为了展示他们的愿景。
   - **Mozilla** 的原则与 **Movie Gen** 性质之间的区别突显了其与更广泛目标的一致性。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **FAANG 公司要求 SDLC 认证**：一位用户询问了除了 **PMP** 之外，被 **FAANG** 公司认可的**软件开发生命周期 (SDLC)** 认证的相关课程。
   - 这对于从不同行业转型到技术岗位的申请人来说是一个重大关注点。
- **LangChain API 调用发生变化**：一位成员注意到 **LangChain** 的 **API chain** 发生了变化，并正在寻求最新的 **API** 调用方法。
   - 这突显了 **LangChain** 框架内持续的更新和发展。
- **LangChain 将支持 GPT 实时 API**：一位用户询问 **LangChain** 何时会支持最近发布的 **GPT real-time API**，并提到了即将到来的集成。
   - 通过一段解决这些查询的 [YouTube 视频](https://www.youtube.com/watch?v=TdZtr1nrhJg)提供了进一步的澄清。
- **评估 RAG 流水线检索器**：有人就如何评估和比较 **RAG pipeline** 中三种不同**检索器 (retrievers)** 的性能寻求建议。
   - 一位成员建议使用 **query_similarity_score** 来识别性能最佳的检索器，并提出通过 LinkedIn 分享代码片段。
- **用户对 LangChain 聊天机器人的兴趣**：一位用户请求关于如何使用 **LangChain** 创建自己的**聊天机器人 (chatbot)** 的指导。
   - 这表明利用 **LangChain** 进行聊天机器人开发的兴趣日益浓厚。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **NeurIPS 2024 为 Taylor Swift 粉丝调整日期**：**NeurIPS 2024** 会议的开始日期已移至 **12 月 10 日星期二**，幽默地指出这是受到了 **Taylor Swift 的 Eras Tour** 影响。
   - 正如一篇 [tweet](https://fxtwitter.com/WilliamWangNLP/status/1841879266142904469) 中所强调的，这一变动允许代表们提前一天到达，从而更好地配合旅行计划。
- **Elon Musk 举办安保严密的 xAI 招聘盛会**：**Elon Musk 的 xAI** 招聘活动在身份检查和金属探测器的严密安保下，展示了通过代码生成的现场音乐，引发了 AI 招聘领域的关注。
   - 此次活动恰逢 **OpenAI 的 Dev Day**，在 **融资传闻** 满天飞之际，Musk 旨在吸引顶尖人才，引发了广泛讨论。
- **OpenAI CEO 在座无虚席的 Dev Day 发表演讲**：**OpenAI** CEO **Sam Altman** 在年度 **Dev Day** 上向满场的开发者发表讲话，推广了最近的进展和即将推出的项目。
   - 活动期间，有关 OpenAI 即将完成一轮创纪录融资的传闻四起。
- **Meta Movie Gen 发布高级功能**：Meta 首发了 **Movie Gen**，这是一套能够根据文本提示生成高质量图像、视频和音频的媒体基础模型（media foundation models），具备个性化视频创建等令人印象深刻的能力。
   - 据报道，他们正在与*创意专业人士密切合作*，以便在更广泛发布之前增强该工具的功能。
- **强化学习增强用于代码的 LLM**：一篇新论文提出了一种用于 **LLM** 竞技编程任务的端到端强化学习（Reinforcement Learning）方法，在提高效率的同时实现了 state-of-the-art 的结果。
   - 该方法展示了执行反馈（execution feedback）如何大幅减少样本需求，同时增强催化性能。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensors：Permute 与 Reshape 的抉择**：一位成员询问是使用 `.permute` 还是 `.reshape` 来将目标 tensor 的尺寸从 (1024,1,14,1) 转换为 (14,1024,1)，突显了深度学习中 tensor 操作的复杂性。
   - *Dumb q.* 反映了一些挫败感，表明需要明确 tensor 操作的最佳实践。
- **高效的 Stable Diffusion 训练**：有人询问在 **M3 MacBook Air** 上于 **48 小时**内训练 **Stable Diffusion** 模型的可行性，显示出对高效模型训练方法的兴趣。
   - 这表明用户需要精简的资源，使高性能训练更加触手可及。
- **需要增强 bfloat16 测试**：George 强调了在 tinygrad 中增加 **bfloat16 测试**的重要性，指出了目前 `test_dtype.py` 中的局限性。
   - 一位成员询问哪些*额外测试*能真正增强测试框架的鲁棒性。
- **看看这些 Triton 演讲**：一位成员分享了一个 **Triton 演讲**的 YouTube 链接，内容涵盖了 Triton 技术的各种发展，为开发者提供了见解。
   - 你可以在[这里](https://www.youtube.com/watch?v=ONrKkI7KhU4)观看，以深入了解 Triton 的功能。
- **分析 tinygrad CI 警告和失败**：有人呼吁对 tinygrad 测试运行期间最近出现的 **CI 警告**提供见解，旨在提高框架的可靠性。
   - 查看 [node cleanup 和测试速度](https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880) 有助于理解最近的更改和稳定性工作。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 的 KTO 训练查询**：一位用户询问 **Torchtune** 是否支持 **KTO 训练**，表明了对其效率能力的兴趣。
   - 该讨论串中没有分享更多细节或回复。
- **VinePPO 改变了 LLM 推理的 RL**：一位成员展示了 **VinePPO**（对 PPO 的一种改进），在基于 RL 的方法中实现了高达 **9 倍的步数减少**和 **3 倍的时间节省**。
   - 这些结果表明 **RL post-training** 方法可能会发生转变，同时也带来了显著的内存节省。
- **Flex Attention 提升运行时效率**：**Flex Attention** 通过利用 Attention Mask 中的 **block sparsity** 来保持运行时性能，在 **bsz=1** 和 **bsz=2** 的设置下表现出相同的性能。
   - 测试已确认，处理 **1000 tokens** 时，其时间和内存效率与批处理（batching）相似。
- **简化 Packed Runs 中的 Batch Size**：有人提议取消 Packed Runs 中的 Batch Size 选项，专注于 **tokens_per_pack** 以实现稳定的 **bs=1**。
   - 这可以提高效率并简化对性能指标的考量。
- **DDP 实现讨论**：成员们推测了 **Distributed Data Parallel (DDP)** 的集成，将每个 Sampler 设置为 **bsz=1**，以优化单设备资源利用率。
   - 这可能会改善跨设备的性能分配。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **AI 提升了网络速度，但软件滞后**：最近的讨论指出，**AI 的进步**使得 **100 Gbps** 技术更加实惠，实验室已实现 **1.6 Tbps**。
   - *Darkmatter* 强调软件未能跟上 **80 倍的带宽增长**，导致即使在 **10 Gbps** 时也面临挑战。
- **增强网络能力的紧迫性**：*Luanon404* 表达了对改进网络的强烈愿望，宣称 *“是时候加速网络了。”*
   - 这凸显了对当前网络框架中最佳 **throughput** 和 **latency** 的日益关注。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **为 axolotl 探索 pip 的替代方案**：一位成员发现 **axolotl** 中的 **dependency management** 令人沮丧，并建议使用 **uv** 等非 pip 打包工具进行安装和更新。
   - 他们表现出积极参与旨在增强 **axolotl** 体验的持续工作的意愿。
- **axolotl 开发中的社区参与**：同一位成员表示愿意通过研究多样化的 **packaging options** 来改进 **axolotl** 库。
   - 他们的目标是促使其他开发者参与进来，解决在 **dependency management** 方面的共同困扰。



---


**Alignment Lab AI Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---

# PART 2: 按频道分类的详细摘要和链接


{% if medium == 'web' %}




### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1291475638653550683)** (254 条消息🔥🔥): 

> - `PyTorch 的 torchao 库`
> - `OpenAI 的 Canvas 工具`
> - `Meta 的 Movie Gen 模型`
> - `AI 训练中的文化偏见`
> - `Nous Forge Framework`

- **torchao：模型优化的飞跃**：由 PyTorch 发布的 [torchao 库](https://huggingface.co/posts/singhsidhukuldeep/639926000427051) 引入了诸如 quantization 和 low-bit datatypes 等先进技术，优化了模型的性能和内存效率。
   - 功能包括自动 quantization 以及与现有工具的集成，被誉为 PyTorch 生态系统迈出的重要一步。
- **OpenAI Canvas 工具广受好评**：用户对 OpenAI 的 [Canvas 工具](https://openai.com/index/introducing-canvas) 感到兴奋，因为它结合了其他平台的特性，简化了编码流程并减少了不必要的滚动。
   - Canvas 的编辑能力被强调为相比 Claude 等工具以往版本的重大改进。
- **Meta 令人印象深刻的 Movie Gen 模型**：Meta 最近发布了其 [Movie Gen 模型](https://ai.meta.com/static-resource/movie-gen-research-paper)，能够根据文本提示生成高质量的图像、视频和音频。
   - 该模型融合了精确视频编辑和个性化视频生成等先进功能，展示了在重大创意应用方面的潜力。
- **文化偏见与 AI 训练**：关于训练 LLM 缺乏内在人类偏见的讨论，使得它们依赖大量训练数据来理解爱和道德等概念。
   - 对话探讨了人类情感的复杂性，以及 AI 如何在不具备内在“真实性”的情况下学习或模拟这些情感。
- **Nous Forge：AI 编排框架**：Nous Forge 被描述为一个编排 AI Agent 的平台，类似于“LLM 领域的 Kubernetes”，增强了对 AI 交互和资源的管理。
   - 然而，该名称可能与 AI 社区中其他现有框架冲突，引发了关于品牌和功能的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">Hugging Face 上的 @singhsidhukuldeep：“@PyTorch 的伙伴们刚刚发布了 torchao，这是一个改变游戏规则的库，用于……”</a>：未找到描述</li><li><a href="https://x.com/aiatmeta/status/1842188252541043075?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>：🎥 今天我们首映 Meta Movie Gen：迄今为止最先进的多媒体基础模型。由 Meta 的 AI 研究团队开发，Movie Gen 在一系列能力上提供了最前沿的结果...</li><li><a href="https://x.com/_tim_brooks/status/1841982327431561528">来自 Tim Brooks (@_tim_brooks) 的推文</a>：我将加入 @GoogleDeepMind，致力于视频生成和世界模拟器（world simulators）的研究！迫不及待想与如此优秀的团队合作。我在 OpenAI 度过了精彩的两年，参与了 Sora 的开发。谢谢...</li><li><a href="https://x.com/m_wulfmeier/status/1842201976597074290?t=bVksmRCFScV1q6Vc4kDwgw&s=19">来自 Markus Wulfmeier (@m_wulfmeier) 的推文</a>：看起来新一代学生已经为基于 Gemini/ChatGPT 的评审时代做好了更充分的准备……</li><li><a href="https://pytorch.org/blog/quantization-aware-training/">使用 PyTorch 进行大语言模型的量化感知训练（Quantization-Aware Training）</a>：在这篇博客中，我们介绍了一个在 PyTorch 中针对大语言模型的端到端量化感知训练（QAT）流程。我们展示了 PyTorch 中的 QAT 如何恢复高达 96% 的精度损失……</li><li><a href="https://arxiv.org/abs/2409.13079">对比式语言-图像预训练（CLIP）的嵌入几何</a>：自 CLIP 发布以来，使用 InfoNCE 损失进行对比式预训练的方法在桥接两种或多种模态方面变得非常流行。尽管被广泛采用，CLIP 原始的...</li><li><a href="https://x.com/lauriewired/status/1841875972691525673?s=46">来自 LaurieWired (@lauriewired) 的推文</a>：你的手机（目前）还不能运行本地 70B 模型。但当你睡觉时，它或许可以。一篇全新的论文 (arXiv:2410.00531) 将 Llama-3.1-70B 在*全精度*下压缩到了仅 11.3GB 的内存中！传统上...</li><li><a href="https://x.com/studiomilitary/status/1841980965771506141?s=46">来自 John Galt (@StudioMilitary) 的推文</a>：我上传了 36 张新壁纸到 doors 应用。引用 kenneth (@kennethnym)：来自 @NousResearch 的新 DOORS 壁纸发布。</li><li><a href="https://x.com/slow_developer/status/1842270727153623414?t=HR1olb-kaLei_1EZRIncug&s=19">来自 Haider. (@slow_developer) 的推文</a>：🚨 重磅消息 Grok 3 将开源。Elon Musk 刚刚宣布 xAI 将开源其模型。</li><li><a href="https://pastebin.com/P0wQwvv9">o1preview - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/pytorch/ao">GitHub - pytorch/ao：PyTorch 原生的用于训练和推理的量化与稀疏化工具</a>：PyTorch 原生的用于训练和推理的量化与稀疏化工具 - pytorch/ao</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户来为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/pull/930">由 gau-nernst 提交的 BitNet b1.58 训练 · Pull Request #930 · pytorch/ao</a>：此 PR 添加了 BitNet b1.58 的训练代码（三元权重 - 1.58 bit。BitNet 的第一个版本是二进制权重）。这是作为 tensor 子类实现的，并能很好地与量化集成...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

lukfbi: 伙计们，请帮帮我，在 Hermes 70b 上进行 RPG 和 RP（角色扮演）的最佳 temperature（温度值）是多少？
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291484044508532798)** (2 条消息): 

> - `VinePPO 算法`
> - `Pluralistic alignment`
> - `Model steerability benchmarks` 


- **VinePPO 解决重推理任务**：论文介绍了 **VinePPO**，这是一种针对 LLM 中 **Proximal Policy Optimization (PPO)** 在重推理任务中因信用分配 (credit assignment) 问题而表现不佳的新方法。
   - 研究揭示，PPO 中的价值网络 (value networks) 经常导致**高方差更新**，并且在评估替代步骤时几乎不优于随机基准 (random baseline)。
- **NeurIPS 上的 Pluralistic Alignment 研讨会**：一名成员对即将举行的 NeurIPS **pluralistic alignment** 研讨会表示热切期待，强调了其与当前 AI 讨论的相关性。
   - 他们寻求关于推理阶段 **model steerability** 基准的见解，特别是关于模型如何与种子人格 (seeded personas) 对齐。
- **对 tradeoff-steerable 基准的需求**：讨论引用了一篇论文，该论文提出需要 **trade-off steerable benchmarks**，使模型能够在推理阶段管理多个目标。
   - 一位贡献者指出该论文具有很强的概念框架，但指出这些基准缺乏具体的实现。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) 越来越多地应用于复杂的推理任务，这些任务在获得任何奖励之前需要执行几个复杂的步骤。为这些步骤正确分配信用 (credit assignment) 是...</li><li><a href="https://x.com/ma_tay_/status/1755605755607359760">Taylor Sorensen (@ma_tay_) 的推文</a>: 我们定义并鼓励多元多目标基准 (pluralistic multi-objective benchmarks) 以及权衡可控基准 (trade-off steerable benchmarks)，这些基准鼓励模型在推理阶段进行转向 ↔️ 以权衡目标，并且……
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291484044508532798)** (2 条消息): 

> - `VinePPO`
> - `Pluralistic Alignment`
> - `Model Steerability Evaluation` 


- **VinePPO 解决 LLM 中的信用分配问题**：该论文评估了 **Proximal Policy Optimization (PPO)** 在增强 LLM 信用分配 (credit assignment) 方面的作用，并引入了 **VinePPO** 来改进这一方面，因为当前的价值网络 (value networks) 在复杂推理任务中经常失效。
   - 结果显示，价值网络*几乎不优于随机基准*，凸显了在重推理任务中采用替代策略的必要性。
- **对 Pluralistic Alignment 研讨会的期待**：一名成员对即将举行的 NeurIPS 研讨会表示兴奋，该研讨会专注于 **pluralistic alignment** 及其对模型行为的影响。
   - 他们寻求关于现有 **model steerability** 基准的见解，这些基准旨在推理阶段对齐特定的角色 (personas)。
- **对 Tradeoff-Steerable 基准的需求**：讨论集中在需要 **trade-off steerable benchmarks** 来评估模型在推理过程中管理多个目标的能力。
   - 该论文提供了一个坚实的概念框架，但缺乏这些基准的具体实现，而这对于评估 **model steerability** 至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) 越来越多地应用于复杂的推理任务，这些任务在获得任何奖励之前需要执行几个复杂的步骤。为这些步骤正确分配信用 (credit assignment) 是...</li><li><a href="https://x.com/ma_tay_/status/1755605755607359760">Taylor Sorensen (@ma_tay_) 的推文</a>: 我们定义并鼓励多元多目标基准 (pluralistic multi-objective benchmarks) 以及权衡可控基准 (trade-off steerable benchmarks)，这些基准鼓励模型在推理阶段进行转向 ↔️ 以权衡目标，并且……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1291479413170503754)** (2 messages): 

> - `OpenAI's model outputs` (OpenAI 模型输出)
> - `Open-source reasoning models` (开源推理模型)


- **OpenAI 的输出阻碍了开源开发**：一位成员指出，**OpenAI 希望**防止其输出被蒸馏（distilled）到开源推理模型中。
   - 这种限制可能会阻碍更广泛的社区获取 AI 模型开发资源并限制其创新。
- **对推理工具可访问性的担忧**：另一点集中在如何通过对输出设置**限制**，来防止个人创建自己的推理模型。
   - 这一观点反映了开发者对 AI 技术更**开放获取**的渴望。


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1291475960188895324)** (140 messages🔥🔥): 

> - `Telemetry in Aider` (Aider 中的遥测)
> - `OpenRouter Free Models Limitations` (OpenRouter 免费模型限制)
> - `Benchmarking Aider with Various Models` (使用各种模型对 Aider 进行基准测试)
> - `Transition of Aider Repo Ownership` (Aider 仓库所有权转移)
> - `User Experiences with Aider Performance` (用户对 Aider 性能的使用体验)


- **关于 Aider 遥测功能的讨论**：一位成员表达了在 Aider 中加入遥测（telemetry）的需求，强调了“选择性加入”（opt-in）功能的重要性，既能提供洞察力又能确保用户隐私。
   - 另一位成员建议包含系统调用追踪以诊断性能问题，并提到需要对收集的数据保持透明。
- **OpenRouter 免费模型及其使用限制**：用户讨论了 OpenRouter 免费模型的限制，指出所有免费模型共享每个账户每天 200 条消息的严格限制。
   - 某些模型无法访问付费版本，这引发了用户对使用灵活性的疑问。
- **使用 Aider 对各种模型进行基准测试**：参与者分享了不同模型的基准测试结果，显示出性能参差不齐，并讨论了处理过程中的错误率。
   - 重点介绍了 Aider 处理各种编辑场景的能力，以及用户在 Token 限制和错误消息方面的体验。
- **Aider 仓库所有权转移**：官方宣布 GitHub 上的 Aider 主仓库已从个人账户迁移到专门的 Aider 组织页面，以便更好地进行组织管理。
   - 文档和代码中的链接将进行更新以反映此更改，旨在明确项目的身份。
- **Aider 的用户体验与性能**：多位用户报告了在使用 Aider 处理大文件时遇到的各种性能问题，并讨论了避免错误的配置。
   - 一位用户指出，使用 `--no-pretty` 标志显著提高了处理速度，并对默认设置导致意外 API 错误表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install/install.html">Installing aider</a>：Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/install.html">Installation</a>：如何安装并开始使用 Aider 进行结对编程。</li><li><a href="https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks">Creating and highlighting code blocks - GitHub Docs</a>：未找到描述</li><li><a href="https://bolt.new/">bolt.new</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账户为 Aider-AI/aider 做出贡献。</li><li><a href="https://github.com/Aider-AI/aider">GitHub - Aider-AI/aider: aider is AI pair programming in your terminal</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账户为 Aider-AI/aider 做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/pull/1907">fix: Ensure consistent language in coder prompt descriptions by fry69 · Pull Request #1907 · Aider-AI/aider</a>：修复 #1850。感谢 @businistry 的报告以及 @jorgecolonconsulting 的深入研究并提供修复方案！
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1291524121326518303)** (69 条消息🔥🔥): 

> - `在 Ollama 中使用 Aider`
> - `Aider 中的文件添加`
> - `Aider 性能与模型`
> - `Repo map 功能`
> - `用于查询的 Aider 模式` 


- **在 Ollama 本地模型中使用 Aider**：有用户反馈在使用本地 **Ollama 8B 模型**运行 Aider 时响应速度较慢，并询问使用 OpenAI 或 Anthropic 的付费 API key 是否能提高速度。
   - 据指出，本地模型在处理编辑任务时可能会比较吃力，而 Aider 在使用以编辑能力著称的模型时通常表现更好。
- **Aider 中的文件添加行为**：一位用户测试了 Aider 中新的 **/read-only** 命令，发现它现在仅按文件夹而非文件名进行补全，这增加了访问特定文件的复杂性。
   - 另一位用户确认，如果使用得当，**/read-only** 命令仍应添加文件夹中的所有文件。
- **Aider 模型性能与速度**：用户讨论了使用 **Ollama 8B** 等小型本地模型的局限性，这些模型阻碍了 Aider 在代码编辑过程中快速且准确响应的能力。
   - 有用户提到了 **Cursor Composer AI** 等替代方案作为更快的选择，从而引发了关于 Aider 的速度是否可以通过付费 API key 得到提升的疑问。
- **Repo Map 功能及其禁用状态**：一位用户发现 **repo map** 需要 Git 仓库才能正常工作，此前曾对其在某些模型下的禁用状态感到困惑。
   - 在初始化 Git 仓库后，该用户成功启用了 repo map，并在查询过程中获得了有用的上下文。
- **利用 Aider 模式进行高效查询**：讨论强调了在 Aider 中使用不同模式的重要性，特别是使用 **/ask** 和 **/architect** 来有效地查询代码库。
   - 用户指出，这些模式可以引导 Aider 请求相关文件，从而减少 token 使用量并改善结果。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>：使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1291745189232377916)** (2 条消息): 

> - `Aider 相关提及`
> - `使用 SQLite 的混合搜索`
> - `Reciprocal Rank Fusion` 


- **HN 讨论中的 Aider**：[Hacker News](https://news.ycombinator.com/item?id=41732634) 上一个有趣的帖子多次提及 **Aider**，引发了引人入胜的对话。
   - *多位参与者分享了与 Aider 功能及其在近期发展中作用相关的见解。*
- **SQLite 中的混合搜索策略**：**Alex** 在 [sqlite-vec](https://github.com/asg017/sqlite-vec) 扩展上的工作引入了快速向量查找，融合了**向量相似度**和**传统全文搜索**。
   - 在他的[博客文章](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/)中可以找到详细的探索，其中概述了混合搜索方法的潜力。
- **Reciprocal Rank Fusion 方法**：目前正在研究的最有前途的方法是 **Reciprocal Rank Fusion**，它结合了向量搜索和全文搜索结果中排名靠前的项目。
   - Alex 提供了一个 SQL 查询示例，展示了如何将 **sqlite-vec** KNN 向量搜索与 FTS5 搜索结果集成。



**提及的链接**：<a href="https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/">Hybrid full-text search and vector search with SQLite</a>：作为 Alex 在其 [sqlite-vec](https://github.com/asg017/sqlite-vec) SQLite 扩展（为 SQLite 添加快速向量查找）工作的一部分，他一直在研究混合搜索，即搜索结果从...

  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1291486205766008903)** (1 条消息): 

> - `Salamandra 设备端演示`
> - `OpenAI 模型更新`
> - `Nemo-Mistral-Minitron 改进`
> - `Realtime Whisper Turbo`
> - `MusicGen iOS 应用进展` 


- **Salamandra 设备端演示表现出色**：[Salamandra](https://huggingface.co/spaces/Tonic/salamandra-on-device) 演示由一位认证用户展示，以引人入胜的形式突出了其功能，展现了令人印象深刻的能力。
   - *社区对 Salamandra 关注的热情* 反映了人们对设备端 AI 应用日益增长的兴趣。
- **OpenAI 的 o1 模型登场**：两个新的 **OpenAI** 模型 **o1-preview** 和 **o1-mini** 已集成到 [开源聊天机器人](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0) 中，增强了其功能。
   - 成员们对这些新增功能表示庆祝，称其为迈向更强大聊天机器人体验的重要一步。
- **Nemo-Mistral-Minitron 获得提升**：[Nemo-Mistral-Minitron](https://huggingface.co/spaces/Tonic/Nemo-Mistral-Minitron) 演示的改进已由一位认证用户推出，提升了其性能和可用性。
   - *关于升级的讨论* 表明了优化 AI 模型以获得更好交互和结果的趋势。
- **Realtime Whisper Turbo 上线**：一个使用 Gradio 5 beta 的新 [Realtime Whisper Turbo](https://huggingface.co/spaces/KingNish/Realtime-whisper-large-v3-turbo) 项目已发布，承诺提供实时转录性能。
   - 社区反馈积极，强调了其在各种应用中的潜力。
- **MusicGen iOS 应用取得进展**：**MusicGen** iOS 应用的进展突出了其功能，包括输入音频的降噪以及一个“tame the gary”开关。
   - *一位成员评论道*，它特别专注于鼓点，并尝试更好地结合输入以获得更精细的输出。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://iatalk.ing/mapa-conceitos-ia/)">什么是 OpenAI、神经网络、架构、LLM 和其他 AI 概念？ - IA Talking 🤖</a>: 当我开始学习 AI 时，我遇到了大量新概念：OpenAI、LLM、ChatGPT、参数、模型、llama、gpt、hugging face、模型、rag、embedding、gguf，啊…… 它是 ...</li><li><a href="https://x.com/thepatch_kev/status/1840536425776763020)">来自 thecollabagepatch (@thepatch_kev) 的推文</a>: 第 4 天，MusicGen 延续版的 iOS 应用，着陆页、输入音频降噪以及一个大致有效的 'tame the gary' 开关，专注于鼓点并更努力地整合输入...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1291476058779947048)** (151 条消息🔥🔥): 

> - `Meta Movie Gen`
> - `Hugging Face Chat support`
> - `Gradio Chatbot UI`
> - `Model usage`
> - `InstantMesh` 


- **Meta Movie Gen 发布先进模型**：Meta 推出了 [Movie Gen](https://go.fb.me/kx1nqm)，这是一个拥有 30B 参数的 Transformer 模型，能够根据文本提示生成高质量的图像和视频，同时还发布了一个 13B 的音频模型，用于将高保真音频与视频同步。
   - 该发布展示了精确视频编辑和个性化视频等详细功能，引发了关于可访问性和实用性的讨论。
- **Hugging Face Chat 支持 Transformer**：用户询问 Hugging Face Chat 是否支持 Transformer 模型，特别是提到像 BERT 这样用于问答任务的模型。
   - Huggingchat 中的模型利用了 Transformer 架构，重点关注问答等任务，详见 [Hugging Face 任务部分](https://huggingface.co/tasks/question-answering)。
- **Gradio Chatbot UI 咨询**：一位用户寻求关于如何在 Gradio Chatbot UI 中通过编程方式触发提交按钮而无需手动点击的建议。
   - 建议手动调用相关函数，但用户仍不清楚具体需要哪个函数。
- **模型执行环境讨论**：用户讨论了 Diffusion Pipeline 的执行，澄清了生成过程是在执行命令的机器上运行，而不是在 Hugging Face 服务器上。
   - 产生了一个疑问：使用 Diffuser API 是否比直接用 Python 运行模型具有明显优势。
- **InstantMesh 集成咨询**：一位用户提出了关于使用 Diffuser API 运行 InstantMesh 以及它与本地执行方法相比如何的问题。
   - 这突显了使用 API 与直接本地执行相比所提供的灵活性，特别是在处理模型输出方面。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">@singhsidhukuldeep 在 Hugging Face 上发布: &quot;PyTorch 的优秀团队刚刚发布了 torchao，这是一个改变游戏规则的库，用于…&quot;</a>: 未找到描述</li><li><a href="https://x.com/AIatMeta/status/1842188252541043075">来自 AI at Meta (@AIatMeta) 的推文</a>: 🎥 今天我们首映 Meta Movie Gen：迄今为止最先进的媒体基础模型。由 Meta 的 AI 研究团队开发，Movie Gen 在一系列能力上提供了最先进的结果...</li><li><a href="https://huggingface.co/spaces/allenai/reward-bench">Reward Bench Leaderboard - 由 allenai 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#hardware-specs">使用 GPU Spaces</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/autotrain/tasks/llm_finetuning">LLM 微调</a>: 未找到描述</li><li><a href="https://www.tiktok.com/t/ZP8RtX7x1/">TikTok - Make Your Day</a>: 未找到描述</li><li><a href="https://github.com/NVIDIAGameWorks/toolkit-remix">GitHub - NVIDIAGameWorks/toolkit-remix: RTX Remix Toolkit</a>: RTX Remix Toolkit。通过在 GitHub 上创建账号为 NVIDIAGameWorks/toolkit-remix 的开发做出贡献。</li><li><a href="https://huggingface.co/tasks/question-answering">什么是问答系统？ - Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/TencentARC/InstantMesh">GitHub - TencentARC/InstantMesh: InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</a>: InstantMesh：使用稀疏视图大型重建模型从单张图像高效生成 3D 网格 - TencentARC/InstantMesh</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/TuringsSolutions/527665072738819">@TuringsSolutions 在 Hugging Face 上发布: &quot;超维计算 + 神经网络，告诉你的朋友们。对我而言…&quot;</a>: 未找到描述</li><li><a href="https://github.com/RichardAragon/HyperDimensionalComputingNeuralNetwork">GitHub - RichardAragon/HyperDimensionalComputingNeuralNetwork</a>: 通过在 GitHub 上创建账号为 RichardAragon/HyperDimensionalComputingNeuralNetwork 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1291480950580056075)** (5 messages): 

> - `Sanity 70B FP8`
> - `CUDA`
> - `μP`
> - `HuggingFace model upload`
> - `Outdated tutorials` 


- **学习 Sanity 70B FP8 和 CUDA**: 一位成员分享了他们在过去三天里学习了 **Sanity 70B FP8**、**CUDA** 和 **μP**，表明他们正在深入研究高级主题。
   - *学习这些技术对于性能优化和有效的模型部署至关重要。*
- **在 HuggingFace 模型上传方面遇到困难**: 一位成员正尝试学习如何正确地将模型上传到 **HuggingFace console**，但他们参考的教程已经过时。
   - *他们注意到模型文件类型存在差异，发现其他模型除了 **model.pkl** 之外还使用了 **.json** 文件。*
- **寻找更新的资源**: 该成员表示需要更及时的教程，并正在 YouTube 上搜索示例以澄清上传过程。
   - *社区成员参与了讨论，其中一人询问官方资源中是否记录了这些信息。*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1291762637214060684)** (6 messages): 

> - `New AI Model from Nvidia`
> - `Music Composer on HuggingFace`
> - `Text to Singing Model` 


- **Nvidia 发布了一款改变游戏规则的 AI 模型**: 据 [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/) 报道，Nvidia 发布了一款新的 AI 模型，被描述为**开放、巨大**，并准备好**与 GPT-4 竞争**。这一公告在 AI 社区引起了轰动。
   - 许多人好奇该模型将如何与 GPT-4 等现有竞争对手竞争，以及它带来了哪些独特的功能。
- **HuggingFace 上基于 Gradio 的音乐作曲家**: 一位成员分享了一个项目的链接，展示了一个在 HuggingFace Spaces 上使用 **Gradio** 构建的完整**音乐作曲家**，链接见 [此处](https://huggingface.co/spaces/skytnt/midi-composer)。这揭示了 HuggingFace 生态系统中正在开发的创新应用。
   - 该项目因其创意和功能性而受到关注，展示了 AI 如何辅助音乐创作。
- **寻求文本转歌唱功能**: 一位成员表示他们正在寻找一种 **text to singing** 模型或方法，以便在传统空间之外有效地利用歌唱功能。这突显了对更多样化音乐 AI 应用的需求。
   - 对开发此类功能的兴趣表明，将 AI 整合到各种音乐格式中正成为一种日益增长的趋势。



**提到的链接**: <a href="https://huggingface.co/spaces/skytnt/midi-composer">Midi Music Generator - a Hugging Face Space by skytnt</a>: 未找到描述

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1291482310587191339)** (10 条消息🔥): 

> - `Salamandra-2B-Instruct 发布`
> - `Fastai 卷积详解`
> - `Nvidia 模型更新`
> - `新型 LLM 标注工具`
> - `Llava 视频理解模型` 


- **Salamandra-2B-Instruct 来了！**: **Salamandra-2B-Instruct** 模型已发布，为 Hugging Face 平台的用户带来了令人兴奋的新功能。
   - 访问其 [演示页面](https://huggingface.co/spaces/Tonic/Salamandra-2B-Instruct) 了解更多详情。
- **Fastai 课程关于 CNN 的见解**: 一位用户在学习 Fastai 课程的 **Lesson 15** 时，分享了他们对展开卷积（unrolling convolutions）的探索，解释了 **CNNs** 就像是每个输入都没有独立权重的 **NNs**。
   - 他们在 [Fastai 论坛](https://forums.fast.ai/t/rearranging-convolutions-as-matrix-products/114703?u=forbo7) 上详细讨论了他们的发现。
- **Nvidia 团队的模型创新**: Nvidia 团队正在持续发布新内容，特别是通过 **nvidialign** 进行消融实验（ablations）以缩减模型尺寸，从而增强可用性的新模型。
   - 该系列模型受到密切关注，展示了在模型性能和功能方面的重大进展。
- **创新的数据标注工具**: 开发了一款用于 **标注数据** 和微调 **LLMs** 的新型协作工具，该工具结合了 AI 和人工监督，以提高准确性和效率。
   - 感兴趣的测试者可以观看 [演示视频](https://www.youtube.com/watch?v=YVwby-49Y-I&feature=youtu.be) 并对这款极具前景的工具提供反馈。
- **令人兴奋的新 Llava 视频理解模型**: **Llava Video Understanding Model** 的最新演示已发布，展示了其在视频理解方面的能力。
   - 好奇的用户可以点击 [此处](https://huggingface.co/spaces/Tonic/Llava-Video) 查看演示，了解其功能的更多信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Salamandra-2B-Instruct">Salamandra 2B Instruct - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Llava-Video">Llava Video - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://forums.fast.ai/t/rearranging-convolutions-as-matrix-products/114703?u=forbo7">Rearranging Convolutions as Matrix Products</a>: 我目前正在学习 fastai 课程的第 15 课，并完成了关于卷积的部分。在这里，我解释了如何将卷积操作重排或展开为矩阵...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1291837752127783055)** (4 条消息): 

> - `AI 感知力预测`
> - `原创研究分享`
> - `每周读书会` 


- **探索 AI 感知力预测**: 一篇题为《感知力预测方程》（The Sentience Prediction Equation）的文章讨论了 AI 何时可能获得感知力及其影响，并质疑 AI 是否会思考其存在的意义。
   - 文章幽默地列举了 AI 可能会问的问题，比如 *“为什么人类坚持要在披萨上加菠萝？”*，并介绍了“感知力预测方程”作为一种估算工具。
- **关于分享原创研究的咨询**: 一位成员询问社区内是否有分享原创研究的场所。
   - 另一位成员建议在 Discord 中发布，并艾特可能对该话题感兴趣的个人。
- **用于讨论的每周读书会**: 提到存在一个每周读书会，用于分享和讨论各种话题。
   - 这为成员提供了一个展示研究成果并进行学术交流的平台。



**提到的链接**: <a href="https://medium.com/@ryanfoster_37838/the-sentience-prediction-equation-when-will-ai-achieve-sentience-and-should-we-be-worried-bf5fa0042408">The Sentience Prediction Equation: When Will AI Achieve Sentience? (And Should We Be Worried?)</a>: 你已经听到了传闻：AI 变得越来越聪明。它在写小说、制作迷因、诊断疾病，甚至，嗯，正在生成这段文字……

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1291583441393029184)** (1 条消息): 

> - `模型训练详解`
> - `概念学习 vs 指令学习`
> - `灾难性遗忘` 


- **模型训练：孩童式学习 vs 学生式学习**：这里提出了一个类比，将孩子从环境中学习比作学生从书本中学习数学。孩子代表 **pre-trained model**（预训练模型），而 fine-tuning（微调）则反映了进阶所需的 **instruction-based learning**（基于指令的学习）。
   - 如果一个幼儿拿到一本数学书，他们不会明白其用途，这类似于缺乏基础知识的模型会影响其性能。
- **学习稀有概念的挑战**：对话转向理解新的、稀有的主题，例如假设的**居住在暗物质中的外星人**，这些主题并不广泛可见。这意味着当学生面对缺乏先前学习经验关联的主题时，可能会感到**吃力**。
   - 在没有上下文的情况下，渴望学习的人可能会遇到困难，导致对先前学习信息的所谓 **catastrophic forgetting**（灾难性遗忘）。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1291479150871314566)** (8 条消息🔥): 

> - `Spacy 的在线训练模块`
> - `使用自定义数据集进行微调`
> - `为语言模型使用 SFTTrainer`
> - `ONNX 模型转换问题`
> - `Transformers.js 集成` 


- **Spacy 的在线训练深受欢迎**：一位成员赞扬了 **Spacy** 结构化的在线训练模块，认为它是初学者深入研究 NLP 概念的绝佳起点。
   - 他们强调该模块提供了一个结构化的免费课程，有效地针对初学者阶段。
- **使用自定义数据微调模型**：一位成员表示，虽然可以使用**公共数据集**微调模型，但适配自定义数据集在很大程度上取决于具体用例。
   - 他们建议，如果不对原始文本进行大量修改或清洗，应确保自定义数据与公共数据集相似。
- **用于语言模型数据集的 SFTTrainer 类**：一位用户指出所讨论的数据集属于 **language model** 类型，并建议使用 **SFTTrainer** 类进行微调。
   - 他们请求确认这是否正确，希望能明确合适的 trainer 用法。
- **ONNX 转换与 Transformers.js 的问题**：一位成员在使用 **transformers.js** 加载以 **ONNX** 格式导出的模型时遇到问题，无法加载 `onnx/decoder_model_merged_quantized.onnx`。
   - 他们寻求帮助，促使另一位成员建议验证模型的保存位置以及指定路径的正确性。
- **排查 ONNX 模型加载故障**：针对 ONNX 加载问题，另一位成员建议检查 `from_pretrained` 函数中的默认参数以解决加载问题。
   - 他们强调了确保模型的物理位置与预期路径匹配的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/openai/gsm8k">openai/gsm8k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct?row=0">TIGER-Lab/MathInstruct · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/main/en/alignprop_trainer">通过奖励反向传播对齐文本到图像扩散模型</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1291568492830785546)** (3 条消息): 

> - `Flux 模型限制`
> - `Hacktoberfest 贡献` 


- **Flux 并非真正的开源**：与普遍看法相反，**Flux** 并不是真正意义上的**开源**，因为其模型规范和训练数据保持私有，仅分享了权重。
   - *这凸显了开源实践中认知与现实之间的脱节。*
- **在 ML 领域寻找 Hacktoberfest 仓库**：一位成员询问如何发现专门针对 ML 领域的 **Hacktoberfest** 贡献仓库。
   - 作为回应，另一位成员建议使用 **GitHub 的搜索功能**，并提到他们特定的频道 **Diffusers** 尚未开启 Hacktoberfest 的 issue。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1291477746245177475)** (134 条消息🔥🔥): 

> - `Canvas Model`
> - `OpenAI Tools`
> - `Advanced Voice Mode`
> - `Discord Bots`
> - `AI Programming` 


- **Canvas 模型讨论升温**：许多用户对新的 **Canvas 模型** 表示兴奋，讨论了其潜在功能，包括手动调用以及与 GPT-4o 的集成，详见[此链接](https://openai.com/index/introducing-canvas/)。
   - 成员们注意到 **Canvas** 目前不支持某些功能，这引发了一些挫败感，但也承认它在增强编程 UX 方面具有潜力。
- **Advanced Voice Mode 受到关注**：**Advanced Voice Mode** 引发了关于其与 Canvas 工具集成的讨论，暗示未来两者可能会无缝协作。
   - 用户表达了希望实时 API 集成等功能能够提高编码效率的愿望，甚至有人在 GitHub 上分享了设置指南。
- **Discord 机器人与社区帮助**：一名用户就 **Discord 机器人** 寻求帮助，显示出社区对在编码中遇到困难的新手提供了强有力的支持。
   - 这促使多位成员提供协助，并分享了他们在创建和调试机器人方面的经验。
- **AI 在编程语言中的角色**：讨论强调了 **OpenAI 模型** 在编程中的有效性，特别是 **TypeScript** 和 **Python**，建议在使用 AI 进行编码任务时优先选择强类型语言。
   - 一些成员指出了使用 JavaScript 时的挑战和挫折，同时称赞了 **Kotlin** 等替代方案的易用性。
- **通信中的 AI 与头像**：关于在语音通话中 AI 工具使用头像的实用性存在辩论，对于其必要性以及对用户体验的影响持有不同意见。
   - 成员们注意到了头像在专业品牌塑造中的潜力，暗示了专业环境中数字交互工具的演变趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/2oQ5VkW-DZ8?si=DwnRHuROyerEZ0zR">How to use a Large Action Model (AI) to schedule any task</a>：了解如何通过 Nelima 全新的调度功能将你的操作提升到新水平！在本视频中，我将向你展示如何使用 Nelima 强大的...</li><li><a href="https://github.com/jjmlovesgit/ChatGPT-Advanced-Voice-Mode">GitHub - jjmlovesgit/ChatGPT-Advanced-Voice-Mode: ChatGPT Advanced Voice Mode Gets an Avatar!</a>：ChatGPT Advanced Voice Mode 拥有了头像！通过在 GitHub 上创建账号，为 jjmlovesgit/ChatGPT-Advanced-Voice-Mode 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1291477208401186826)** (11 条消息🔥): 

> - `Custom GPTs with Google API`
> - `Custom GPTs Model Queries`
> - `Canvas Issues`
> - `ChatGPT Counting and Math Concerns` 


- **Custom GPTs 集成较为繁琐**：一位成员分享了过去尝试将 **Google API/OAuth** 与 Custom GPTs 集成的经历，指出在最初发布期间该过程非常繁琐。
   - 自那以后，他们还没有重新检查该集成的稳定性是否有所改善。
- **Canvas 缺少核心功能**：多位成员对新版 **Canvas** 缺少 **继续（continue）按钮** 表示失望，这使得使用过程变得笨重。
   - 此外，在编辑大文件时存在问题，且文档格式不匹配也阻碍了功能的发挥。
- **ChatGPT 的数学能力受到质疑**：一位成员质疑 **ChatGPT** 的计数能力是否变差了，认为需要明确的指令才能执行数学任务。
   - 另一位成员澄清说，像 ChatGPT 这样的 LLM 是文本预测器，并建议使用 **数据分析工具** 或 Python 进行精确计算。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1291534615156363315)** (8 messages🔥): 

> - `ChatGPT 评估的不一致性`
> - `在 Newl Canvas 中嵌入图像`
> - `将片段高效解析为 JSON`
> - `模型评分技术` 


- **ChatGPT 评估的不一致性**：一位用户分享了在 **temperature 0.7** 下提示 ChatGPT 按 10 分制对答案进行评分时，评估结果不一致的困扰。
   - 成员建议，由于 GPT 具有随机性（stochastic），使用更紧凑的量表（如 **0-5**）并提供 **grading rubric**（评分标准）可以提高一致性。
- **在 Newl Canvas 中嵌入图像**：一位用户指出，对于 Newl Canvas 主界面，可以使用语法 ```
![Image Description](Image Link)
``` 直接嵌入图像。
   - 此功能可以简化在 canvas 演示中包含视觉内容的过程。
- **将片段高效解析为 JSON**：一位用户正在使用 Python 和 GPT-4o 将 10,000 条文本片段解析为 JSON 格式，并询问在每个片段中重复提交 **system_prompt** 和 **response_format** 的效率问题。
   - 有建议提出如何通过仅提交下一个片段而无需每次重新提交结构来降低成本。
- **模型评分技术**：为了解决评估问题，一位用户建议在提供分数之前，采用 **Chain-of-Thought** 方法对评估过程进行推理。
   - 此外，建议一次评估一个答案，并提供多样化的高质量评估示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1291534615156363315)** (8 messages🔥): 

> - `ChatGPT 评估一致性`
> - `在 Newl Canvas 中使用图像`
> - `使用 GPT-4o 进行 JSON 解析`
> - `评估用的 grading rubric`
> - `评估中的 Chain-of-Thought` 


- **ChatGPT 评估的不一致性**：一位用户注意到，当要求 ChatGPT (temperature @ 0.7) 按 10 分制评估答案时，重新运行时会得到不同的分数。
   - 另一位用户解释说，GPT 的随机特性导致了输出的多样化，建议收紧评分量表以提高一致性。
- **在 Newl Canvas 中嵌入图像**：一位成员分享了 Newl Canvas 主界面可以使用语法 ```
![Image Description](Image Link)
``` 嵌入图像。
   - 此功能增强了用户在 canvas 内的视觉化能力。
- **使用 GPT-4o 进行高效 JSON 解析**：一位使用 GPT-4o 将 10,000 个片段解析为 JSON 的开发者询问，是否需要为每个片段重新发送 **system_prompt** 和 **response_format**。
   - 针对通过简化流程而不重新提交公共参数来优化成本提供了建议。
- **grading rubric 和 Chain-of-Thought 建议**：为了提高评分准确性，一位用户建议提供 **grading rubric** 以明确每个分数的含义。
   - 他们还建议采用 **Chain-of-Thought** 推理，在得出最终分数之前提高评估的清晰度。
- **评估的最佳实践**：为了进行有效的评估，建议包括将 **temperature** 降至 0，并一次评估一个答案，同时提供多样化的高质量示例。
   - 这些策略旨在促进模型生成更可靠且多样化的评估。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1291495318004695060)** (77 messages🔥🔥): 

> - `Unsloth AI 项目`
> - `PEFT 中的 Lora 配置`
> - `微调模型`
> - `ZLUDA 项目更新`
> - `Movie Gen AI 模型`

- **用于微调的 Unsloth AI 项目**：成员们讨论了使用 Unsloth AI 进行 LLM 的持续预训练（continual pretraining），指出其训练效率比其他方案**快 2 倍**，且**节省 50% 的 VRAM**。
   - 持续预训练笔记本和文本补全笔记本被强调为在不同语言中训练模型的关键工具。
- **Lora 配置挑战**：一位成员询问了如何在 Lora 配置中使 embedding 层可训练，寻求关于通过 **modules_to_save** 选项将其包含在目标模块中的明确说明。
   - 关于持续预训练和文本补全笔记本之间的区别存在一些困惑，重点在于学习率调度（learning rate scheduling）。
- **渐进式学习微调**：关于微调方法的讨论建议从较简单的数据集开始，随后逐渐引入更复杂的数据，以提升模型性能。
   - 成员们推测，逐渐增加数据集的复杂度可能是有益的，特别是对于未知语言。
- **ZLUDA 项目公告**：ZLUDA 的开发正由一家新的商业机构资助，承诺将改进功能并为项目提供长期愿景。
   - 然而，有人对与 NVIDIA 可能存在的法律问题表示担忧，这呼应了过去的经验，即投资者的支持可能会因知识产权纠纷而动摇。
- **Movie Gen AI 模型介绍**：Movie Gen AI 模型作为一种通过简单文本输入生成高清视频的新标准被推出，支持先进的编辑功能。
   - 成员们反应积极，认可了该项目的创新性，并对它在内容创作方面的潜在影响表示兴奋。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vosen.github.io/ZLUDA/blog/zludas-third-life/">ZLUDA - ZLUDA 的第三次生命</a>：未找到描述</li><li><a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">Hugging Face 上的 @singhsidhukuldeep："@PyTorch 的伙伴们刚刚发布了 torchao，这是一个改变游戏规则的库..."</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2410.00531">TPI-LLM：在低资源边缘设备上高效运行 70B 规模的 LLM</a>：由于对用户交互数据隐私的担忧，大模型推理正从云端转向边缘。然而，边缘设备通常面临计算能力、内存和带宽有限的困境...</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 (Continued Pretraining) | Unsloth 文档</a>：又名持续微调 (Continued Finetuning)。Unsloth 允许你进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://huggingface.co/posts/TuringsSolutions/527665072738819">Hugging Face 上的 @TuringsSolutions："超维计算 (Hyperdimensional Computing) + 神经网络 (Neural Network)，告诉你的朋友们。对我来说..."</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，使模型学习一种新语言。</li><li><a href="https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch">如何在 PyTorch 中释放 GPU 显存</a>：我有一组句子，正尝试使用以下代码通过多个模型计算困惑度 (perplexity)：
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
impo...</li><li><a href="https://x.com/danielhanchen/status/1841921149804163247">Daniel Han (@danielhanchen) 的推文</a>：我在 @PyTorch 大会上关于加速 LLM 训练技巧的演讲视频发布了！1. 极限位表示。需要 O(Mantissa^2) 个晶体管。Bfloat16(M=7)=49 & float32(M=32)=529。2. 硬件 - tensor cores...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md (位于 0efb8bd31e4) · meta-llama/llama-recipes</a>：使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，支持单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要生成和问答等应用...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003">GitHub - meta-llama/llama-recipes (位于 0efb8bd31e4)</a>：使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，支持单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要生成和问答等应用...</li><li><a href="https://github.com/unslothai/unsloth#finetune-llama-32-mistral-phi-35--gemma-2-5x-faster-with-80-less-memory">GitHub - unslothai/unsloth：微调 Llama 3.2、Mistral、Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3.2、Mistral、Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/RichardAragon/HyperDimensionalComputingNeuralNetwork">GitHub - RichardAragon/HyperDimensionalComputingNeuralNetwork</a>：通过在 GitHub 上创建账号为 RichardAragon/HyperDimensionalComputingNeuralNetwork 的开发做出贡献。</li><li><a href="https://ai.meta.com/research/movie-gen">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1291477922221264997)** (43 条消息🔥): 

> - `代际认同`
> - `Z 世代偏好`
> - `Lego vs. 模组化 Minecraft` 


- **代际认同危机**：成员们开玩笑地讨论了他们的代际身份，其中一人声称对身为 Z 世代感到羞耻，而另一人尽管只有 24 岁却自称“婴儿潮一代 (boomer)”。
   - *“只是那些人最吵闹”* 引起了对代际刻板印象的共鸣。
- **Z 世代更喜欢 VSCode？**：讨论了 Z 世代对 **VSCode** 的偏好，一些成员幽默地提到他们使用 **VS Codium** 来阻止遥测 (telemetry)。
   - 有人开玩笑说，童年时期玩 Lego 现在几乎像星座一样定义了代际界限。
- **Lego 的文化意义**：一位成员表示，未来几代人中 Lego 游戏的衰落将标志着“社会的终结”，强调了其文化重要性。
   - 另一位成员建议，**模组化 Minecraft (modded Minecraft)** 可以作为年轻一代 Lego 的可接受替代品。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1291506668575789147)** (31 messages🔥): 

> - `使用 llama-cpp 进行本地推理`
> - `Multi-GPU 支持`
> - `使用纯文本微调模型`
> - `准备训练数据集`
> - `在移动端使用 Flutter 运行 LLM` 


- **本地推理脚本遇到困难**：一位成员分享了在使用 **llama-cpp** 运行 **gguf 模型** 的本地推理脚本时遇到的困难，尽管拥有 **GPU**，但处理时间仍然很长。
   - 另一位成员建议使用 **llama-cli** 以获得潜在的更好性能。
- **Multi-GPU 支持更新**：一位成员询问了关于 **multi-GPU** 支持的更新，并提到一周前申请了权限但未收到回复。
   - 另一位成员指出测试正在进行中，目前访问权限有限，但预计将在今年晚些时候更广泛地开放。
- **使用纯文本进行微调**：讨论了关于在纯文本数据集（特别是医学书籍）上微调 **llama3.1** 的问题，并提醒训练需要 **structured data**（结构化数据）。
   - 一位成员建议使用 **augmenToolKit** 将书籍转换为结构化数据集，并强调 80% 的工作量在于数据集准备。
- **在 Flutter 应用中运行 LLM**：一位成员表示需要在 **PC** 上运行 **LLM**，同时接收来自 Flutter 移动应用程序的输入。
   - 另一位成员建议使用基于 **/chat/completion** 的方法来实现这种集成。
- **寻找 16bit 模型**：一位成员请求有关 **16bit 模型** 的信息和资源，寻找 notebooks 或相关材料。
   - 另一位成员提供了 **Unsloth 文档** 的链接，其中包含可供参考的 notebooks 列表。



**提到的链接**：<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：请参阅下面的列表以获取我们所有的 notebooks：

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1291477205452460207)** (6 条消息): 

> - `Nanoflow 框架`
> - `循环神经网络 (RNN) 的复兴`
> - `SageAttention 量化`
> - `代码替换建议` 


- **Nanoflow 为 LLM 提供高吞吐量服务**：[Nanoflow](https://github.com/efeslab/Nanoflow) 是一个针对 **LLM** 优化的高性能推理服务框架，专注于吞吐量，旨在提高处理速度。
   - 它旨在解决大型模型中经常遇到的推理服务复杂性，在效率方面提供了显著改进。
- **RNN 重磅回归！**：最近的一篇论文讨论了极简 LSTM 和 GRU 的潜力，通过移除隐藏状态依赖，从而避免随时间反向传播 (BPTT)，训练速度可提高 **175 倍**。
   - 在现代架构背景下，传统 RNN 的复兴为可扩展的训练方法提供了新途径。
- **SageAttention 助力加速量化**：[SageAttention](https://github.com/thu-ml/SageAttention) 引入了一种针对 Attention 的量化方法，与 **FlashAttention2** 和 **xformers** 相比，在不牺牲模型指标的情况下，分别实现了 **2.1 倍** 和 **2.7 倍** 的加速。
   - 该方法与量化过程无缝集成，结合先进技术以提升性能。
- **探索使用 SageAttention 进行代码替换**：有建议认为，将 [Llama 模型](https://github.com/unslothai/unsloth/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426) 中的特定代码行替换为 `sageattn` 是可行的，并引用了效率提升作为理由。
   - 这反映了关于使用最新技术优化实现的持续讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06147">State-Free Inference of State-Space Models: The Transfer Function Approach</a>：我们通过其对偶表示（传递函数）来设计用于深度学习应用的状态空间模型，并发现了一种高效的序列并行推理算法...</li><li><a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>：Transformer 在序列长度方面的可扩展性限制，重新引发了人们对训练期间可并行的循环序列模型的兴趣。因此，出现了许多新型循环架构...</li><li><a href="https://github.com/efeslab/Nanoflow">GitHub - efeslab/Nanoflow: 一个面向吞吐量的高性能 LLM 推理服务框架</a>：一个面向吞吐量的高性能 LLM 推理服务框架 - efeslab/Nanoflow</li><li><a href="https://github.com/thu-ml/SageAttention">GitHub - thu-ml/SageAttention: 一种 Attention 量化方法，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速，且不会损失各种模型的端到端指标。</a>：一种 Attention 量化方法，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速，且不会损失各种模型的端到端指标。 - thu-m...</li><li><a href="https://github.com/unslothai/unsloth/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426">unsloth/unsloth/models/llama.py at ae9e264e33c69b53dd5d533a4c5a264af4141c28 · unslothai/unsloth</a>：以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1291496054419886206)** (94 条消息🔥🔥): 

> - `IREE 的采用与编译`
> - `RWKV 与并行化`
> - `Chain of Thought (CoT) 输出限制`
> - `Gated Linear Attention 与可表示为 RNN 的模型`
> - `MATS 项目与导师机会` 


- **探索 IREE 的潜力**：成员们讨论了大型实验室是否会采用 IREE 进行大规模模型推理服务，迹象表明许多实验室仍在使用自定义的 inference runtimes。
   - 一些成员指出，像 IREE 这样新技术的采用时间表通常是不可预测的。
- **RWKV 的分层并行化策略**：RWKV 通过将网络结构化为更小的层，引入了一种部分并行化的方法，允许在等待其他层时计算下一个 token 的 hidden state。
   - 这种设计约束旨在简化计算，同时平衡模型输出中相互依赖的需求。
- **Chain of Thought 与计算效率**：讨论中对 Chain of Thought (CoT) 输出的效率表示怀疑，建议可以通过更密集的 representation 方法进行改进。
   - 成员们强调，虽然 CoT 可能有益，但过度依赖它可能无法有效解决潜在的性能问题。
- **理解 Linear Attention 模型**：成员们强调了某些模型（如 linear attention 和 gated linear attention）的双重性质，它们既可以表示为 RNN，同时也能在序列上实现并行计算。
   - 成员们对 Songlin Yang 的研究表现出兴趣，该研究揭示了能够进行高效并行化的更复杂的 RNN 类别。
- **MATS 项目导师公告**：一位成员分享了一条推文，宣布了 2024-25 冬季 MATS 项目的导师名额及申请详情。
   - 这包括与 AnthropicAI 的 Alignment Science 共同负责人合作的导师机会，强调了该项目的增长和参与度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wiki.rwkv.com/advance/architecture.html#how-does-rwkv-differ-from-classic-rnn)">RWKV Architecture</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2312.06635">Gated Linear Attention Transformers with Hardware-Efficient Training</a>: 具有 linear attention 的 Transformers 允许高效的并行训练，但同时可以被表述为具有 2D（矩阵值）hidden states 的 RNN，从而享受线性时间推理...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: 具有 linear attention 的 Transformers（即 linear transformers）和状态空间模型（state-space models）最近被认为是 softmax attention Transformers 的一种可行的线性时间替代方案。然而...</li><li><a href="https://x.com/MATSprogram/status/1842286650006892914">来自 ML Alignment & Theory Scholars (@MATSprogram) 的推文</a>: @janleike，AnthropicAI 的 Alignment Science 共同负责人，现在将担任 MATS 2024-25 冬季项目的导师！申请截止日期为 PT 时间 10 月 6 日晚上 11:59。https://matsprogram.org/apply</li><li><a href="https://github.com/lucidrains/quartic-transformer">GitHub - lucidrains/quartic-transformer: 探索一种放弃效率并在节点（tokens）的每个边缘执行注意力的想法</a>: 探索一种放弃效率并在节点（tokens）的每个边缘执行注意力的想法 - lucidrains/quartic-transformer
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1291484013566885939)** (49 条消息🔥): 

> - `VinePPO 的挑战`
> - `minLSTMs 和 minGRUs`
> - `数学领域的迁移学习`
> - `Softmax 函数的局限性`
> - `测试时训练 (TTT)` 


- **VinePPO 在 LLM 信用分配中表现出问题**：论文讨论了 **value networks** 在复杂推理任务中如何难以进行信用分配（credit assignment），导致其表现甚至不如随机基准（random baselines）。
   - 这凸显了在 Proximal Policy Optimization (PPO) 中需要更好的模型或方法来有效利用信用分配技术。
- **重新审视用于并行训练的 LSTM 和 GRU**：对 **minLSTMs** 和 **minGRUs** 的探索揭示了一种无需通过时间反向传播（BPTT）即可高效并行训练循环网络的方法，实现了 175 倍的训练加速。
   - 该研究表明，传统的 RNN 架构可以被简化，同时仍能提供显著的性能提升。
- **量化数学领域的迁移学习**：一位参与者询问了关于在 **MATH** 和 **GSM8k** 等数学推理任务上训练模型时，量化迁移效应的研究。
   - 他们表示有兴趣了解跨相关任务的性能提升在多大程度上是可预测的。
- **Softmax 函数的局限性**：一篇论文讨论了 **softmax function** 在输入增长时可能难以做出果断决策（sharp decisions），从而限制了其逼近激进计算的能力。
   - 这一局限性表明在 softmax 实现中需要自适应方法，以增强模型预测的鲁棒性。
- **测试时训练 (TTT) 的前景**：参与者强调了 **Test Time Training (TTT)** 的引人注目之处，并指出其在机器学习未来理论进步方面的潜力。
   - 大家认识到 TTT 可能会给非线性模型带来风险，但它仍被认为是一个充满希望的探索领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>：Transformer 在序列长度方面的可扩展性限制重新激发了人们对训练期间可并行的循环序列模型的兴趣。因此，许多新型循环架构...</li><li><a href="https://arxiv.org/abs/2410.01104">softmax is not enough (for sharp out-of-distribution)</a>：推理系统的一个关键属性是对其输入数据做出果断决策的能力。对于当代的 AI 系统，果断行为的一个关键载体是 softmax 函数，凭借其能力...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>：Self-attention 在长上下文中表现良好，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的表现受限于其...</li><li><a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>：大语言模型（LLMs）越来越多地应用于复杂的推理任务，这些任务在收到任何奖励之前需要执行几个复杂的步骤。正确地为这些步骤分配信用是至关重要的...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1291515512706174987)** (1 条消息): 

> - `lm-evaluation-harness`
> - `GPT-NeoX improvements` 


- **lm-evaluation-harness 需要贡献者**：**lm-evaluation-harness** 正在征集贡献者，以集成新的 LLM 评估并修复 bug，[这里](https://github.com/EleutherAI/lm-evaluation-harness/issues)有许多详细的 issue 可供探索。
   - 社区鼓励潜在的贡献者查看 [GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness) 以获取更多信息。
- **GPT-NeoX 寻求改进**：**GPT-NeoX** 团队正在寻求帮助，以增强其测试套件并添加新测试，这些内容可以在 [tests directory](https://github.com/EleutherAI/gpt-neox/tree/main/tests) 中找到。
   - 贡献者还可以帮助改进容器设置，并探索 [issues page](https://github.com/EleutherAI/gpt-neox/issues) 上列出的各种问题。
- **探索 GPT-NeoX 的新特性**：**GPT-NeoX** 项目为有兴趣贡献的人展示了许多新的分布式特性，详情可通过其 [PRs](https://github.com/EleutherAI/gpt-neox/pulls) 查看。
   - 参与这一领域可能会为该库的功能带来具有影响力的增强。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/755950983669874798)">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues">Issues · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。- Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/gpt-neox/tree/main/tests">gpt-neox/tests at main · EleutherAI/gpt-neox</a>: 一个基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues">Issues · EleutherAI/gpt-neox</a>: 一个基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - Issues · EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1291475109600825405)** (2 条消息): 

> - `SambaNova AI on OpenRouter`
> - `Gemini 1.5 Flash-8B Release` 


- **SambaNova AI 登陆 OpenRouter，拥有最快吞吐量**：[SambaNova AI](https://x.com/SambaNovaAI/status/1841901026821210131) 宣布其 **Llama 3.1 和 3.2** 的端点已在 OpenRouter 上线，并自豪地展示了他们记录到的最快吞吐量测量结果。
   - 他们提到，“这是我们见过的最快的”，并强调他们的吞吐量测量通常比其他公司更保守。
- **Gemini 1.5 Flash-8B 现已可用**：**Gemini 1.5 Flash-8B** 模型已正式发布，可在此处访问使用 [here](https://openrouter.ai/models/google/gemini-flash-1.5-8b)。
   - 此外，为了保持一致性，该模型的 ID 已重命名，而旧 ID 仍将通过别名继续发挥作用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/SambaNovaAI/status/1841901026821210131">Tweet from SambaNova Systems (@SambaNovaAI)</a>: 我们在 @OpenRouter 上线了！他们说这是他们见过的最快吞吐量测量结果。🚀🚀🚀 感谢点名！ 引用 OpenRouter (@OpenRouterAI) .@SambaNovaAI 的 Llama 3.1 端点...</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5-8b">Gemini 1.5 Flash-8B - API, Providers, Stats</a>: Gemini 1.5 Flash-8B 针对速度和效率进行了优化，在聊天、转录和翻译等小提示词任务中提供增强的性能。通过 API 运行 Gemini 1.5 Flash-8B
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1291477094810779804)** (140 messages🔥🔥): 

> - `Gemini 1.5 Flash`
> - `o1 Mini performance`
> - `Anthropic's model development`
> - `Model alignment techniques`
> - `OpenRouter infrastructure updates` 


- **Gemini 1.5 Flash 以低成本优势令人瞩目**：**Gemini 1.5 Flash-8B** 模型提供了每百万 **tokens** **$0.0375** 的竞争性价格，引发了关于其性能和定价结构与其他模型对比的讨论。
   - 成员们推测了 **Gemini** 最新产品的潜在扩展性和适用性。
- **o1 Mini 展示了更强的解决能力**：用户注意到 **o1 Mini** 在有效解决复杂任务方面表现出色，令社区中那些未曾预料其性能会超越其他模型的人感到惊讶。
   - 一位参与者计划在机器人中使用 **o1 Mini** 来辅助图像描述，突显了其增强的易用性。
- **Anthropic 凭借资金支持获得战略优势**：讨论显示，**Anthropic** 的成功可以归功于其由前 **OpenAI** 工程师组成的团队以及来自 **Amazon** 的支持，这使得他们的 **Claude** 模型得以快速开发。
   - 尽管与大型企业相比资金支持较少，但关于他们如何保持竞争性能的推测依然存在。
- **创新的 alignment 技术引发讨论**：成员们讨论了像 **Anthropic** 这样的模型如何处理 **alignment**，提到了其在无需模型后置过滤的情况下进行训练的有效性，这与 **OpenAI** 的方法形成对比。
   - 对话还涉及了 **prompt injections** 和模型审核技术的概念。
- **OpenRouter 基础设施改进**：用户对 **OpenRouter** 未来的扩展表示期待，以支持更广泛的模型功能，包括图像和音频处理。
   - 开发负责人确认正在进行升级，以应对不断增长的流量和新模型的发布。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developers.googleblog.com/en/gemini-15-flash-8b-is-now-generally-available-for-use/">Gemini 1.5 Flash-8B 现已准备好投入生产</a>：未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1fvk2wr/what_would_an_ai_with_anxiety_look_like/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://openrouter.ai/settings/privacy">隐私 | OpenRouter</a>：管理您的隐私设置</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：跨多个提供商路由请求</li><li><a href="https://openrouter.ai/models/openai/gpt-4-vision-preview">GPT-4 Vision - API、提供商、统计数据</a>：除了所有其他 [GPT-4 Turbo 功能](/models/openai/gpt-4-turbo)外，还具备理解图像的能力。训练数据：截至 2023 年 4 月。通过 API 运行 GPT-4 Vision
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1291569294731378730)** (127 条消息🔥🔥): 

> - `LM Studio 更新`
> - `内存泄漏问题`
> - `模型下载与集成`
> - `聊天缓存位置`
> - `AI 模型推荐` 


- **LM Studio 支持 Langflow**：好消息是 LM Studio 的支持正被集成到 Langflow 中，正如 GitHub 上最近的一个 Pull Request 所指出的。
   - 这旨在为希望创建 LLM 应用程序的用户增强功能。
- **内存泄漏担忧**：用户报告在 LM Studio v0.3.2.6 版本中遇到内存泄漏，导致模型输出乱码。
   - 建议检查 v0.3.3 版本是否仍存在相同问题。
- **下载模型与故障排除**：用户在从 Hugging Face 下载模型时遇到问题，特别是在 LM Studio 中选择模型时出现错误。
   - 建议通过将模型直接 sideload（侧载）到 LM Studio 的模型目录中来解决。
- **聊天缓存自定义查询**：用户询问是否可以更改 LM Studio 中聊天缓存的位置，目前该功能尚不支持自定义。
   - 应用程序现在以 JSON 格式保存对话数据，但聊天缓存位置的配置尚未开放。
- **AI 模型推荐**：关于推荐哪些 AI 模型作为聊天机器人助手的讨论中，一些用户对 Llama-3-8B 表示不满。
   - 用户被引导至 LM Studio 平台上的各种可用模型，鼓励探索更符合其需求的选项。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:9222`">未找到标题</a>: 未找到描述</li><li><a href="https://dontasktoask.com/">不要问是否可以问，直接问</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">侧载模型 - 高级 | LM Studio 文档</a>: 使用你在 LM Studio 之外下载的模型文件</li><li><a href="https://lmstudio.ai/docs">入门指南 | LM Studio 文档</a>: 了解如何使用 LM Studio 在本地运行 Llama, Mistral, Gemma 和其他 LLM。</li><li><a href="https://lmstudio.ai">LM Studio - 实验本地 LLM</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://lmstudio.ai/models">模型目录 - LM Studio</a>: 你可以在电脑上运行的最新且最出色的 LLM。</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory))">下载 LLM - 本地运行 LLM | LM Studio 文档</a>: 在 LM Studio 中发现并下载受支持的 LLM</li><li><a href="https://lmstudio.ai/docs/basics/chat#faq">管理聊天 - 本地运行 LLM | LM Studio 文档</a>: 管理与 LLM 的对话线程</li><li><a href="https://github.com/langflow-ai/langflow/pull/4021">feat: 由 EDLLT 添加 LM Studio 模型和 Embeddings 组件 · Pull Request #4021 · langflow-ai/langflow</a>: 修复了 #3973</li><li><a href="https://lmstudio.ai/docs/basics/server">本地 LLM 服务器 - 本地运行 LLM | LM Studio 文档</a>: 使用 LM Studio 在 localhost 上运行 LLM API 服务器
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1291475762268082299)** (18 条消息🔥): 

> - `LangChain 语音 ReAct Agent`
> - `GPT-4o 对话`
> - `Meta Movie Gen 突破`
> - `新的金融领域 LLM 排行榜`
> - `上下文信息嵌入模型`

- **LangChain 发布 Voice ReAct Agent**：利用 [Realtime API](https://x.com/langchainai/status/1841914757764485439?s=46)，LangChain 推出了 **Voice ReAct Agent**，它集成了语音和工具，用于创建自定义语音体验。
   - 他们通过一段视频展示了其能力，视频中一个 Agent 使用计算器和 [Tavily 网络搜索工具](https://youtu.be/TdZtr1nrhJg)执行操作。
- **GPT-4o 机器人进行对话**：一段演示展示了 **两个 GPT-4o 语音 AI 机器人** 使用 Realtime API 进行对话，突显了语音 AI 技术的进步。
   - 对话涉及不同的设置，展示了新 API 在 *轮流发言延迟 (turn-taking latency)* 方面的效率。
- **Meta 宣布 Movie Gen 项目**：Meta 的新突破 **Meta Movie Gen** 旨在提供先进的视频生成能力，目前尚未设定发布日期。
   - 可以在其 [AI 研究页面](https://ai.meta.com/research/movie-gen/) 及其 [相关论文](https://ai.meta.com/static-resource/movie-gen-research-paper) 中进一步了解该研究。
- **金融领域新 LLM 排名亮相**：最近发布的金融 **LLM 排行榜** 显示，**OpenAI 的 GPT-4**、**Meta 的 Llama 3.1** 和 **阿里巴巴的 Qwen** 是 40 项相关任务中的领先模型。
   - 这一新基准旨在优化性能评估，详见 [Hugging Face 博客](https://huggingface.co/blog/leaderboard-finbench)。
- **上下文嵌入模型 (Contextual Embedding Models) 的进展**：开发了一种新的 **上下文信息嵌入** 模型 cde-small-v1，通过在训练过程中加入 *上下文标记 (contextual tokens)* 来增强文本检索。
   - 该模型的性能和理论基础记录在最近的一篇 [ArXiv 论文](https://x.com/jxmnop/status/1842236045074498026?s=46) 中，详细描述了它所代表的范式转变。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_tim_brooks/status/1841982327431561528?s=46">来自 Tim Brooks (@_tim_brooks) 的推文</a>：我将加入 @GoogleDeepMind，致力于视频生成和世界模拟器 (world simulators) 的研究！迫不及待想与如此优秀的团队合作。我在 OpenAI 参与开发 Sora 的两年时光非常精彩。感谢...</li><li><a href="https://x.com/jxmnop/status/1842236045074498026?s=46">来自 jack morris (@jxmnop) 的推文</a>：我们花了一年时间开发 cde-small-v1，这是世界上最好的 BERT 规模的文本嵌入 (text embedding) 模型。今天，我们正在 HuggingFace 上发布该模型，并同步在 ArXiv 上发布论文。我认为我们的发布...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我非常激动能分享我们最新的 AI 研究突破。我们称之为 Meta Movie Gen，它是一系列最先进模型的集合，共同实现了最先进的视频生成...</li><li><a href="https://x.com/langchainai/status/1841914757764485439?s=46">来自 LangChain (@LangChainAI) 的推文</a>：🎤 Voice ReAct Agent 🤖 使用 @OpenAI 的新 Realtime API，你可以利用语音 + 工具的力量来构建自定义语音体验。看看我们与一个简单 Agent 对话的视频，它能够推理...</li><li><a href="https://x.com/clefourrier/status/1842286565374193665?s=46">来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：新的 LLM 排行榜：金融领域！💰 它使用了 40 个领域相关的任务，从预测和风险管理到问答和信息提取！目前排名前三的模型：- @OpenAI 的 GPT4 ...</li><li><a href="https://x.com/jxmnop">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/kwindla/status/1841936672755483115">来自 kwindla (@kwindla) 的推文</a>：旧版 4o vs 新版 4o —— 两代语音 AI 之间的对话。这是我昨晚在 @cloudflare/@openai 开发者活动上展示的演示。这是两个 GPT-4o 语音 AI 机器人互相交谈...</li><li><a href="https://x.com/andersonbcdefg/status/1841987927049724120">来自 Ben (e/treats) (@andersonbcdefg) 的推文</a>：虽然不是无损的，但它确实有效！！</li><li><a href="https://x.com/sama/status/1841946796274176405?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sam Altman (@sama) 的推文</a>：现在已向 100% 的 ChatGPT Plus 订阅用户开放！引用 Sam Altman (@sama)：查看 ChatGPT 中的 Canvas：https://openai.com/index/introducing-canvas/</li><li><a href="https://x.com/OfficialLoganK/status/1841903061360640029">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：向 Gemini 1.5 Flash-8B ⚡️ 问好，现在已可用于生产环境，具有：- 价格降低 50%（对比 1.5 Flash）- 速率限制提高 2 倍（对比 1.5 Flash）- 短提示词延迟更低（对比 1.5 Flash）...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPa">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我非常激动能分享我们最新的 AI 研究突破。我们称之为 Meta Movie Gen，它是一系列最先进模型的集合，共同实现了最先进的视频生成...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1291852309009006723)** (98 messages🔥🔥): 

> - `Discord 音频问题`
> - `Luma AI 应用`
> - `Gaussian splatting`
> - `游戏中的 3D 建模`
> - `虚拟会议` 


- **Discord 音频困扰**：用户报告了会议期间 Discord 音频功能的各种挑战，建议切换到 Zoom 或重新加入通话。
   - *一位用户幽默地指出，没有麦克风问题的会议是不完整的*，突显了对在线平台的普遍挫败感。
- **Luma AI 的精彩用途揭晓**：成员们对 **Luma AI** 创建逼真 3D 模型并将其集成到 Unity 或 Unreal 等平台的能力表示热衷。
   - 几位成员分享了展示 **Luma AI** 在电影剪辑和 3D 建模中功能的链接，表明了其在各种创意领域的潜力。
- **Gaussian splatting 与 3D 表示**：对话讨论了 **Gaussian splatting**，特别是其在游戏渲染和优化 3D 环境中的重要性。
   - 用户提到了包含 **Gaussian splatting** 的特定模型和工具，强调了该领域未来发展的*巨大潜力*。
- **对虚拟会议的兴趣**：参与者表示有兴趣安排更多虚拟会议，以深入探讨通话中讨论的 AI 和 3D 建模话题。
   - *记录了合作呼吁*，用户对未来的探索和技术咨询表示兴奋。
- **感谢与正面反馈**：随着对话结束，用户对通话中引人入胜的讨论和知识分享表示感谢。
   - 开场白 **AI in Action** 作为会议的主题焦点，强化了共同探索 AI 进展的意图。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vimeo.com/1012136742/065081e415">FREE YOSHI - PROOF OF CONCEPT</a>：这是 Jeremy Rubier 在 Vimeo 上的 &amp;quot;FREE YOSHI - PROOF OF CONCEPT&amp;quot;，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://x.com/aishashok14/status/1832760312455450907/video/1">Aishwarya Ashok (@aishashok14) 的推文</a>：山中之夜——皮克斯风格的电影 :) 合作方 @midjourney (--sref 804246641), @LumaLabsAI (摄像机运动) 和 @udiomusic。在疲惫的攀登结束时，去徒步旅行是什么感觉...</li><li><a href="https://x.com/karanganesan">undefined 的推文</a>：未找到描述</li><li><a href="https://lumalabs.ai/web">Luma AI - Fields Dashboard</a>：用 AI 将你的想象变为现实。</li><li><a href="https://x.com/aishashok14/status/1829738607281635371/video/1">Aishwarya Ashok (@aishashok14) 的推文</a>：慢即是美✨ 深呼吸，冷静的头脑，宁静的温暖，放松的时刻……这些都是美好的！这里提醒我们所有人：慢很酷，慢很美。合作方 @midjourney 和 @LumaLabs...</li><li><a href="https://x.com/bennash/status/1840829850292011172?s=46">Ben Nash (@bennash) 的推文</a>：使用速度提升 10 倍的新版 @LumaLabsAI 制作的文本转视频驾驶舱场景</li><li><a href="https://x.com/aishashok14/status/1828790536410730878/video/1">Aishwarya Ashok (@aishashok14) 的推文</a>：稍等，正忙着制作茶园纪录片 AI 电影。☕️ 🍃 从郁郁葱葱的种植园到浓郁的茶杯，制茶过程是一种情感。使用 @midjourney & @LumaLabsAI 拍摄...</li><li><a href="https://x.com/lumalabsai/status/1841833038700761205?s=46&t=fm_-fV17wG2CozW7wmZR7g">Luma AI (@LumaLabsAI) 的推文</a>：👀 那么... 你的选择是？🍊↔🍎? carrot↔broccoli? 🧁↔🍩? 🍔↔🍕? 使用 #LumaDreamMachine Keyframes 制作 #foodforthought #hungry #foodie</li><li><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting for Real-Time Radiance Field Rendering</a>：未找到描述</li><li><a href="https://lumalabs.ai/ios">‎Luma AI</a>：‎以惊人的 3D 质量展示你的世界，并在网络上的任何地方分享。由 Luma AI 提供。Luma 是一种使用 iPhone 通过 AI 创建令人难以置信的逼真 3D 的新方式。轻松捕捉产品...</li><li><a href="https://github.com/graphdeco-inria/nerfshop">GitHub - graphdeco-inria/nerfshop: NeRFshop: Interactive Editing of Neural Radiance Fields</a>：NeRFshop：神经辐射场的交互式编辑 - graphdeco-inria/nerfshop
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1291607508875612201)** (1 条消息): 

> - `Performance benchmarks` (性能基准测试)
> - `Fio tools` (Fio 工具)
> - `Data access methods` (数据访问方法)


- **关于性能基准测试的咨询**：一位成员询问了与讨论中提到的某些工具和方法相关的现有**性能基准测试**。
   - 他们特别寻求将这些基准测试与直接从存储访问数据时从 **fio 工具** 获得的原始性能分析进行对比。
- **数据访问方法的对比分析**：讨论强调了分析和比较数据访问方法性能的需求。
   - 成员们对这些方法与传统的 **fio 工具** 性能指标相比表现如何感到好奇。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1291794415425294338)** (2 条消息): 

> - `SageAttention`
> - `Meta Movie Gen` 


- **SageAttention 量化突破**：[SageAttention](https://github.com/thu-ml/SageAttention) 方法在不损失各种模型端到端指标的情况下，与 **FlashAttention2** 和 **xformers** 相比，分别实现了 **2.1 倍**和 **2.7 倍**的加速。
   - 这种量化方法在保持 Attention 机制高性能的同时，强调了效率。
- **Meta 推出 Movie Gen - 一场创意革命**：Meta 发布了 **Movie Gen**，这是一套最先进的媒体基础模型，旨在根据文本提示创建高质量图像和高清视频。
   - 关键能力包括**音视频同步**、精准视频编辑，以及使用用户提供的图像生成**个性化视频**的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/i/status/1842188252541043075">来自 AI at Meta (@AIatMeta) 的推文</a>: 🎥 今天我们发布了 Meta Movie Gen：迄今为止最先进的媒体基础模型。由 Meta 的 AI 研究团队开发，Movie Gen 在一系列能力上提供了最先进的结果...</li><li><a href="https://github.com/thu-ml/SageAttention">GitHub - thu-ml/SageAttention: 一种 Attention 量化方法，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速，且不损失各种模型的端到端指标。</a>: 一种 Attention 量化方法，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速，且不损失各种模型的端到端指标。 - thu-m...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1291606009399349280)** (2 条消息): 

> - `Book Updates` (书籍更新)
> - `Chapter Upgrades` (章节升级) 


- **团队致力于章节升级**：团队正积极致力于升级章节和示例，以增强书籍内容。
   - *“我们正在尽最大努力达成目标，”* 表明了他们对改进的承诺。
- **新书的大幅翻新**：与之前的版本相比，即将出版的新书将进行大幅翻新，承诺对素材进行全新的诠释。
   - 这种翻新表明其重点是更好地符合当前的行业标准和实践。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1291490894234779790)** (4 条消息): 

> - `Event Planning` (活动策划)
> - `Colocation with Conferences` (与会议同期举办)
> - `Planning Timelines` (策划时间线) 


- **关于活动时间的假设**：一位成员指出，如果能在几个月前就知道活动日期，将有利于策划，并建议活动可能会在劳动节假期后的 **9 月**举行。
   - 这个时间线有助于与开学季保持一致，以获得更好的出席率。
- **与会议同期举办的策略**：一位成员提到，可能会与 **Triton** 和 **PyTorch** 会议同期举办，以鼓励团体旅行。
   - 这种策略此前曾为参会者提供了聚集在同一地点的充分理由。
- **活动策划的起步阶段**：一位参与者回顾了他们最初的活动策划经验，承认这是他们协助策划的第一个活动，称之为**起步阶段 (baby steps)**。
   - 他们表示，跨越数月的策划对他们来说构成了挑战。
- **通过经验学习**：另一位成员称赞了最初策划者的努力，尽管他们自己也有活动策划经验，组织过大约 **6 到 7** 场活动。
   - 他们强调，即使是经验丰富的策划者也可以在过程中相互学习。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1291768550603882587)** (4 messages): 

> - `Torchao 中的非连续输入`
> - `OptimState8bit 调度错误`
> - `AdamW8bit 与 Accelerate 的兼容性` 


- **Torchao 在处理非连续输入时存在困难**：建议如果张量（tensor）不连续，**Torchao** 需要使用 **reshape** 才能正常运行。
   - *此问题可能会限制其整体性能。*
- **遇到 OptimState8bit 调度错误**：成员们在尝试使用 **OptimState8bit** 时遇到错误，提示“尝试运行未实现的算子/函数：aten._to_copy.default”。
   - *这指向了当前与 8bit 优化器相关的实现中存在的潜在限制。*
- **AdamW8bit 在 Accelerate 中失效**：**AdamW8bit** 优化器无法与 **Accelerate** 的 save_state/load_state 功能配合使用，导致 NotImplementedError。
   - *堆栈跟踪表明错误发生在与优化器状态管理相关的函数中。*


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1291538040719671428)** (22 messages🔥): 

> - `OpenAI 的财务成功`
> - `OpenAI 潜在的新产品`
> - `简历评审频道提案`
> - `研究生院申请讨论` 


- **OpenAI 财务成功的势头**：成员们注意到 **OpenAI** 在近期创新的推动下，财务增长正创下纪录。
   - *这种收入规模可能旨在制造他们自己的芯片，* 也有关于他们向硬件领域扩张的推测。
- **关于构建新产品的讨论**：一位成员推测 **OpenAI** 可能会开发自己的移动设备，暗示了机器学习在用户数据上的应用。
   - 这一见解强调了类似于 **Apple** 等公司处理用户数据隐私的担忧。
- **关于简历评审频道的提案**：一位成员建议创建一个专门用于**简历评审**的频道，强调了来自同行匿名反馈的好处。
   - 讨论还包括整合模拟面试和社区反馈，尽管该想法面临优先级排序问题。
- **对研究生院申请建议的兴趣**：有人呼吁建立一个专注于**研究生院申请**的频道，成员们表达了对多元化视角的渴望。
   - 一位成员主动提出提供帮助，表明围绕学术界对该领域探索的讨论将大有裨益。
- **强调开源项目开发**：一位用户担心社区会变成 **CV 评审**和求职论坛，强调应专注于开源性能项目。
   - 他们对苦于求职的初级人员表示同情，分享了自己漫长的求职经历，并强调了内在动力而非薪资驱动的目标。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1291743090692063295)** (1 messages): 

> - `Triton kernel 性能`
> - `张量操作`
> - `调试 Triton 函数` 


- **用户在 Triton kernel 中遇到结果不变的问题**：一位用户表达了挫败感，无论他们对 **Triton kernel** 中的代码进行何种修改，结果似乎都没有变化。
   - *有人以前遇到过这个问题吗？* 他们提供了代码片段作为背景。
- **在 Triton 中添加常量的代码片段**：用户分享了一个代码片段，展示了他们实现的 **Triton kernel**，该 kernel 使用 `tl.store` 向张量添加一个常量。
   - `add_kernel` 函数从指针加载值并尝试对它们执行加法操作。


  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1291644916082151434)** (1 messages): 

> - `BF16 随机舍入 (stochastic rounding)`
> - `梯度范数 (grad norm) 分析`
> - `数据打乱 (data shuffling) 问题` 


- **BF16 随机舍入提升性能**：在权重更新中加入 **BF16 stochastic rounding** 带来了**显著的性能提升**。
   - 这种技术似乎增强了模型训练过程的整体效率。
- **梯度范数曲线显示有趣的缺口**：**grad norm 曲线**中的缺口呈现出一个有趣的观察结果，目前尚无解释。
   - 可能需要进一步分析以了解其对模型训练和收敛的影响。
- **数据打乱的潜在问题**：观察到的**损失曲线模式**表明训练期间的数据打乱（data shuffling）可能不足。
   - 改进数据打乱过程可能有助于优化模型的学习并提升性能。


  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1291610706885476433)** (11 messages🔥): 

> - `Conv2d Triton Kernel Performance`
> - `Scaled Int8 Conv2d Exploration`
> - `Liger vs. PyTorch Performance`
> - `Fused KL/JSD Requirement Clarification` 


- **Conv2d Triton Kernel 性能见解**：关于 **Conv2d Triton kernel** 性能的讨论表明，它目前比基准 **PyTorch BF16 conv2d** 实现慢，且速度取决于输入大小。
   - 一位成员计划在两周后处理完当前的学校事务后，重新审视并优化该 kernel。
- **探索 Scaled Int8 Conv2d 的潜力**：有成员提出了关于使用合理的 **Triton Conv2d 实现** 的担忧，并预期 **int8 tensor cores** 带来的加速将补偿较慢的 **BF16 Triton conv2d**。
   - 一位成员强调了通过改进配置和 auto-tuning 来提升性能。
- **性能对比：Liger vs. PyTorch**：测试显示，在某些条件下，**Liger 框架** 比 **Torch Compile** 慢约 **8 倍**，这可能是由于 flag 配置错误导致的。
   - 这表明需要对 Liger 项目的性能调优进行进一步调查。
- **澄清 Fused KL/JSD 实现需求**：一位成员寻求关于实现 **fused KL/JSD loss** 的澄清，询问是否只需要 teacher 的 logits，以及是否应该应用 softmax 和温度调节。
   - 他们展示了拟议的实现结构，但鼓励大家对该方法提供反馈以确保准确性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1291487739874639872)** (5 messages): 

> - `Hyperparameter Scaling Guide`
> - `Open Source Project Maintenance`
> - `Embedding Geometries Paper Acceptance`
> - `Contrastive Language-Image Pre-Training`
> - `Euclidean vs Hyperbolic Geometry` 


- **需要超参数缩放指南**：一位成员表达了对 **超参数缩放（hyperparameter scaling）** 的困惑，强调在大型模型训练实验中缺乏易于获取的启发式方法，并指出现有信息通常局限于个人研究者。
   - *“也许存在这样一份指南……而我因为找不到它而像个白痴”* 反映了在这个复杂话题中寻求清晰度的挣扎。
- **即将发布的关于开源维护的文章**：一位成员预告了一篇关于 **维护开源项目** 的文章，承诺它将比之前的帖子更长、写得更好。
   - 这一主题暗示了可能使开源社区受益的见解和策略。
- **关于替代嵌入几何的 ECCV '24 论文**：题为《*Embedding Geometries of Contrastive Language-Image Pre-Training*》的论文已被 **ECCV '24 Beyond Euclidean Workshop** 接收，该论文探索了对各种嵌入几何的系统性测试。
   - 研究结果表明，直观的 **Euclidean geometry（欧几里得几何）** 在 zero-shot 场景下优于传统的 **CLIP** 和 **MERU**。
- **重新审视 CLIP 设计选择**：在讨论的论文中，作者回顾了原始 **CLIP** 的设计选择，并发现他们的 **Euclidean CLIP** (EuCLIP) 实验提供了与 **hyperbolic（双曲）** 替代方案相似或更优的性能。
   - 他们强调，尽管对比预训练已被广泛采用，但重新审视其基础方面仍然非常重要。
- **研究链接和个人网站**：该成员提供了各种资源的链接，包括他们的个人网站和多篇博客文章，以促进对其工作的进一步关注。
   - 其中包括重要项目和论文的链接，如 **pgen parser generator** 以及关于 **compiler writing** 和 **bug tracking** 的文章。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://apaz-cli.github.io/blog/Hyperparameter_Heuristics.html">Hyperparameter Heuristics</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.13079">Embedding Geometries of Contrastive Language-Image Pre-Training</a>：自 CLIP 发布以来，使用 InfoNCE loss 进行对比预训练的方法已在桥接两种或多种模态方面广泛流行。尽管被广泛采用，CLIP 原本的...</li><li><a href="https://apaz.dev">apaz 的网站</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1291829811374850048)** (48 条消息🔥): 

> - `AVX2 Emulation` (AVX2 仿真)
> - `Matrix Multiplication Implementation` (矩阵乘法实现)
> - `Performance Testing` (性能测试)
> - `Parallel Programming Resources` (并行编程资源)
> - `Tinygrad with AVX Intrinsics` (结合 AVX Intrinsics 的 Tinygrad)


- **AVX2 仿真讨论**：成员们讨论了模拟 **AVX512** 的依赖性取决于具体目标，并指出验证实现将产生不同的性能结果。
   - 一位成员旨在利用 GCC 和 Clang 的向量扩展，为基础算术创建一个**实现库**。
- **阿尔托大学的矩阵乘法练习**：**阿尔托大学 (Aalto University) 的一门课程**提供了一项练习，涉及在原生支持 **AVX512** 的 Intel CPU 上进行代码基准测试，该课程对所有人开放注册。
   - 成员们注意到该练习包括自动基准测试和实现的单元测试，这对于编程实践非常有价值。
- **并行编程 GitHub 资源**：一位成员分享了一个 GitHub 仓库 [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream)，其中包含 **CPU 和 GPU** 的编程资源。
   - 该仓库提供了与 **GPU 编程**相关的材料链接，并强调了一些机构缺乏并行编程课程的问题。
- **Tinygrad 编译至 AVX Intrinsics**：一位成员表示有兴趣尝试 **Tinygrad**，目标是将其编译为 **AVX intrinsics** 以提高性能。
   - 这一想法与正在进行的关于硬件利用率和性能基准测试的讨论相一致。
- **Python 实现中的权重加载挑战**：一位成员分享了在 **Python 实现**中匹配权重加载时面临的挑战，并指出该工作仍在进行中。
   - 他们表示有兴趣利用现有资源来增强对实现实践的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.cppreference.com/w/cpp/experimental/simd">SIMD library - cppreference.com</a>: 未找到描述</li><li><a href="https://github.com/gpu-mode/resource-stream">GitHub - gpu-mode/resource-stream: GPU programming related news and material links</a>: GPU 编程相关新闻和材料链接。通过创建账号为 gpu-mode/resource-stream 的开发做出贡献。</li><li><a href="https://github.com/AndreSlavescu/EasyAI/blob/main/src/kernels/cpu_avx/matrix_methods/matrix_transpose_nn.cpp">EasyAI/src/kernels/cpu_avx/matrix_methods/matrix_transpose_nn.cpp at main · AndreSlavescu/EasyAI</a>: 适合所有人的学习工具！通过创建账号为 AndreSlavescu/EasyAI 的开发做出贡献。</li><li><a href="https://github.com/addaleax/sw-simd">GitHub - addaleax/sw-simd: AVX2 software polyfill for CPUs supporting AVX instructions.</a>: 为支持 AVX 指令的 CPU 提供的 AVX2 软件 polyfill。 - addaleax/sw-simd
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1291504111476408342)** (65 条消息🔥🔥): 

> - `Perplexity AI Collections UI`
> - `Boeing 777-300ER Specifications`
> - `TradingView Premium Package`
> - `Llama 3.2 Release`
> - `Claude 3.5 vs Other Models` 


- **Perplexity AI 正在开发新的 Collections UI**：最近的讨论透露，Perplexity AI 正在为其 Collections 功能开发新的用户界面，重点是显示自定义指令并支持文件上传，尽管目前尚未公开发布。
   - 这项备受期待的 **Files 搜索功能** 将通过更有效地组织信息来提升用户体验。
- **分享了 Boeing 777-300ER 的完整规格参数**：提供了一份关于 **Boeing 777-300ER** 规格的全面大纲，重点介绍了其尺寸、性能、动力装置、载客量及其他特性。
   - 值得注意的细节包括 **7,370 海里** 的 **最大航程**，以及单舱布局下高达 **550 名乘客** 的载客量。
- **TradingView Premium 破解版发布**：一名成员分享了 **TradingView Premium**（版本 2.9）免费破解版的链接，声称拥有适用于各种市场的交易者高级工具。
   - 该版本允许无需付费即可访问高级功能，吸引了众多寻找顶级图表解决方案的用户。
- **对 Llama 3.2 发布的期待**：用户正在询问 **Llama 3.2** 的发布日期，对其即将推出的功能表示兴奋和好奇。
   - 对话显示出人们对这一新迭代的进展和预期改进有着浓厚兴趣。
- **Claude 3.5 与其他 AI 模型的对比**：讨论对比了 **Claude 3.5 Sonnet** 与其他 AI 模型的能力，许多人认为它在获取信息方面更可靠。
   - 用户对 Perplexity Pro 与 Claude 结合的潜力表示关注，认为这能提升从教科书中检索信息的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-working-on-new-collections-ui-for-custom-instructions-and-file-uploads/">Perplexity 正在开发带有文件上传功能的新 Collections UI</a>：发现 Perplexity AI 即将推出的功能：用于自定义指令和文件上传的新 UI。敬请期待增强的搜索能力和文件管理。</li><li><a href="https://www.reddit.com/r/Cracked_Software_Hub/comments/1fo875c/tradingview_premium_cracked_version_available_for/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/Cracked_Software_">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1291579307516231752)** (5 条消息): 

> - `U2V`
> - `Kreutzer's Etudes`
> - `Four-legged Robot`
> - `Quantum Clocks`
> - `Enum Values` 


- **关于如何添加新 Enum 值的说明**：一位用户分享了关于 [如何添加新 Enum 值](https://www.perplexity.ai/search/how-do-you-add-a-new-enum-valu-fBPEV5LtStO_P19ZMVXiQQ) 的查询，重点关注具体的实现细节。
   - 讨论包括了在修改 Enum 时对兼容性和代码完整性的考虑。
- **关于 U2V 的想法**：一位成员询问了对 [U2V](https://www.perplexity.ai/search/hey-what-are-your-rhoughts-on-u2vFogOaTzOse8ibsQNwrA) 的看法，强调了其相关性和应用。
   - 回复讨论了它在各种背景下的潜在影响和有效性。
- **为什么 Kreutzer's Etudes 很重要**：一篇帖子关注了 [Kreutzer's Etudes 在音乐教育中的重要性](https://www.perplexity.ai/search/why-kreutzer-s-etudes-are-one-d8wh8YTgQnm8AO63eJnnvw#0)，强调了技巧的发展。
   - 参与者分享了关于练习曲在掌握小提琴演奏中的作用的见解。
- **四足机器人爬梯子**：分享了一个关于 [爬梯子机器人](https://www.perplexity.ai/page/four-legged-robot-climbs-ladde-OT7S9LK0R.iJ6Yq0c7QmOg) 的链接，展示了其设计和能力。
   - 讨论围绕此类技术在实际应用中的影响展开。
- **了解量子时钟**：一位用户询问了 [量子时钟](https://www.perplexity.ai/search/what-is-a-quantum-clock-t4A_.5lTTiCUnbMObd_5_A)，旨在了解其原理和精度。
   - 贡献内容强调了计时技术的进步以及由该技术驱动的潜在创新。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1291480400366932031)** (2 条消息): 

> - `Command R 08-2024 更新`
> - `与 Weights & Biases 的集成` 


- **Command R 08-2024 微调亮点**：更新后的 *Command R 08-2024* 引入了对新选项的支持，旨在为用户提供**更多控制权**和**可见性**。
   - 此次更新还包含与 [Weights & Biases](https://cohere.com/blog/fine-tuning-command0824) 的**无缝集成**，以增强性能追踪。
- **对 Command R 的热烈反响**：成员们对 Command R 的更新表示**热烈欢迎**，强调了新功能与改进后的易用性的结合。
   - “*Awesome*”等评论捕捉到了社区整体的兴奋与期待。



**提到的链接**：<a href="https://cohere.com/blog/fine-tuning-command0824">Command R 微调更新</a>：微调更新后的 Command R 08-2024，支持新选项，为您提供更多控制权和可见性，包括与 Weights &amp; Biases 的无缝集成。

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1291485248865042472)** (39 条消息🔥): 

> - `指标可见性问题`
> - `微调挑战`
> - `Next.js 中的 Tool Use`
> - `结合 Embedding 数据集的 RAG`
> - `关于 Colabs 的 UI 反馈` 


- **平台中缺失指标**：一位用户报告称，他们无法在 Overview 和 API 等多个标签页中看到模型的**指标框 (metrics boxes)**，而这些位置此前会显示关键信息。
   - 他们对平台的一致性表示担忧，并询问了模型创建的状态，强调该问题已持续 **2 天** 未得到解决。
- **微调上传故障排除**：另一位成员在尝试使用 JSON 训练文档微调聊天机器人时遇到了多个错误，包括编码和解析问题。
   - 他们请求指导以及一个能与 Cohere 平台兼容的 JSON 示例文件。
- **关于 Next.js 中 Tool Use 示例的咨询**：一位用户寻求在 **Next.js** 中使用 Tool Use (Single Step) 的简单示例，并指出大多数文档都是 Python 编写的。
   - 贡献者建议检查切换到 v2 版本是否能解决某些问题。
- **用于 RAG 的 Embedding 数据集**：一位用户表示他们上传了一个 Embedding 数据集打算利用 RAG，但发现无法将其连接到聊天，从而引发了对易用性的担忧。
   - 他们询问了如何针对其需求有效地对 CSV 分块进行 Embedding 处理。
- **对 Colabs 和 UI 的反馈**：用户对文档中几个失效的 Colabs 表示沮丧，并提供了改进反馈。
   - 参与者被鼓励分享代码产生错误或需要更新的具体实例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/structured-outputs-json">结构化生成 (JSON) — Cohere</a>：此页面描述了如何让 Cohere 模型以特定格式（如 JSON）创建输出。</li><li><a href="https://docs.cohere.com/v2/docs/chat-fine-tuning">聊天微调 — Cohere</a>：此文档提供了关于微调、评估和改进聊天模型的指导。</li><li><a href="https://docs.cohere.com/v2/docs/tool-use">Tool Use — Cohere</a>：使您的大型语言模型能够连接外部工具，以实现更高级和动态的交互。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1291746700834242600)** (5 条消息): 

> - `定价差异`
> - `微调命令`
> - `文档更新` 


- **定价页面困惑**：**定价页面**显示训练费用为 **每 1M tokens $3**，但微调 UI 显示的价格为 **$8**。
   - 这一差异引发了对不同平台定价信息准确性的质疑。
- **Command 快捷方式咨询**：有一个关于训练的默认命令是否设置为 **cmd-r+** 以及是否可以更改为 **cmd-r** 的问题。
   - 该咨询反映了对用户体验和界面自定义的关注。
- **微调中 Command 快捷方式的不确定性**：一位成员对 **cmd-r+** 是否适用于微调过程表示不确定。
   - 这表明用户在命令功能知识方面可能存在缺口。
- **文档过时的担忧**：有建议认为文档可能仍然过时，导致了对命令和定价的混淆。
   - 陈旧的文档会显著影响用户体验和故障排除。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 条消息): 

kittykills: Hello!
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1291480042303524985)** (44 条消息🔥): 

> - `OpenPose Alternatives` (OpenPose 替代方案)
> - `ComfyUI Image Quality` (ComfyUI 图像质量)
> - `SDXL Models` (SDXL 模型)
> - `Reference Image Generation` (参考图生成)
> - `AI Tools for Object Placement` (用于物体放置的 AI 工具)


- **OpenPose 姿态替代方案**：用户讨论了使用 **OpenPose** 生成坐姿时遇到的问题，以及 **DWPose** 等替代方案，并询问在哪里可以找到更好的模型。
   - *在拥有足够参考图像的情况下，训练自己的模型也不失为一个可行的解决方案。*
- **提升 ComfyUI 输出质量**：一位成员询问如何让 **ComfyUI** 生成与 **Auto1111** 质量相当的图像，并指出生成的图像看起来很奇怪或带有卡通感。
   - *建议使用 **ComfyUI** 中的特定节点作为提升输出质量的潜在方法。*
- **SDXL 模型说明**：用户讨论了 **SDXL** 的不同版本（包括 `SDXL 1.0`）及其各自的属性，例如分辨率能力（通常从 **1024x1024** 开始）。
   - *一些人确认所有变体都是基于 **SDXL 1.0** 模型开发的。*
- **从参考图生成姿态**：确认了在 **Stable Diffusion** 中使用单张参考图生成姿态是可行的，但可能无法产生最准确的结果。
   - *虽然 **Img2img** 被认为是正确的方法，但拥有多个不同角度的图像会获得更好的保真度。*
- **对物体放置 AI 工具的需求**：有人询问关于 **OpenPose** 技术如何帮助放置物体姿态，并建议针对特定物品（如剑）使用 LoRA 模型。
   - *用户指出，虽然 **Stable Diffusion** 中存在一些训练风格，但仍缺乏专门的姿态设定方法。*


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1291723957317271552)** (4 条消息): 

> - `Translation of Technical Language` (技术语言翻译)
> - `Language Barriers in Tech` (技术领域的语言障碍)


- **单行代码即可实现字幕语言切换**：一位成员建议，只需**一行代码**即可将语音和字幕更改为另一种语言。
   - *这使得技术应用中的多语言支持变得更加容易。*
- **翻译技术术语的挑战**：一位成员指出，技术世界主要以**英语**为主，许多术语不需要翻译。
   - *像 **embeddings**、**manifold** 和 **transformers** 这样的术语在非英语语境中很难处理。*
- **对语言偏好的理解与接受**：另一位成员对这种困难表示认可，称他们理解围绕技术翻译的挫败感。
   - *语言偏好可能会使技术讨论中的沟通变得复杂。*


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1291494195617333309)** (14 条消息🔥): 

> - `MinGRU Architecture` (MinGRU 架构)
> - `Training Bark-like Models` (训练类 Bark 模型)
> - `Scam Alert` (诈骗警示)


- **MinGRU 简化循环神经网络**：该论文介绍了一种极简版本的 **GRUs**，称为 **minGRUs**，它消除了隐藏状态依赖，从而实现了 **175 倍**加速的高效并行训练。
   - 其简单的架构由两个线性层组成，并采用并行处理来计算隐藏状态，这引发了对 **NLP** 中潜在解决方案简洁性的思考。
- **寻求类 Bark 模型指导**：一位新手表示有兴趣从头开始训练一个**类 Bark 模型**，目标是在两到三个月内完成，并寻求相关资源或论文指导。
   - 建议将 **Vall-E 论文** 作为理解训练过程的基础资源。
- **社区诈骗警告**：一位用户识别出另一名成员可能是潜在的诈骗者，该成员声称提供一种在 72 小时内赚取 **$50k** 的方法，以换取 10% 的利润分成。
   - 社区成员被提醒警惕此类计划，并对该提议的真实性表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>：Transformer 在序列长度方面的可扩展性限制重新引起了人们对训练期间可并行的循环序列模型的兴趣。因此，许多新型循环架构...</li><li><a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>：成功的秘诀在于开始行动 🤝
</li>
</ul>

</div>
  

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1291845339707609118)** (1 messages): 

> - `赚钱机会`
> - `Telegram 联系方式` 


- **快速致富方案提议**：一名成员正向**前 10 名感兴趣的人**提供指导，教他们如何在 **72 小时内赚取 5 万美元或更多**，并要求抽取利润的 **10% 作为报酬**。
   - 鼓励*感兴趣的个人*通过 **Telegram** 联系以讨论细节。
- **在 Telegram 上联系 Hugo**：提供了 **Hugo Larsson** 的 **Telegram** 联系方式及直接消息链接。
   - 他强调：*“领先的秘诀在于开始。”*



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1291556189863411766)** (2 messages): 

> - `训练 BARK 模型`
> - `快速赚钱` 


- **寻求 BARK 模型训练指导**：一位新成员表示有兴趣在 **2-3 个月的时间内**从头开始训练一个具有自定义功能的**类 BARK 模型**，但难以找到与 BARK 相关的论文。
   - *他们请求关于如何学习这一过程的建议*，并指出训练细节似乎与 Audio LM 和 VALL-E 等模型密切相关。
- **来自 Hugo 的快速赚钱机会**：成员 **Hugo** 向**前 10 名感兴趣的人**提供帮助，以换取 10% 的利润，在 **72 小时内赚取 5 万美元或更多**。
   - *感兴趣的个人被指示通过 Telegram 向他发送好友请求或私信*，并强调开始行动是成功的关键。



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/1291845216684343369)** (1 messages): 

> - `72 小时内赚取 5 万美元`
> - `Telegram 联络` 


- **72 小时赚 5 万美元方案**：提出了一项提议，帮助前 10 名感兴趣的个人在 **72 小时内赚取 5 万美元或更多**，并收取 10% 的利润分成。
   - 鼓励感兴趣的人在 **Telegram** 上发送好友请求或私信以获取更多细节。
- **直接通过 Telegram 互动**：协调人 **Hugo Larsson** 提供了 **Telegram** 联系链接，以便咨询该赚钱方案。
   - Hugo 强调 *“领先的秘诀在于开始”*，并敦促潜在参与者直接与其互动。



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1291618288429957120)** (19 条消息🔥): 

> - `Article Score Inquiry` (文章评分查询)
> - `Real-time Streaming of Responses` (响应的实时流式传输)
> - `Chainlit Integration` (Chainlit 集成)
> - `Github Autogen Pull Requests` (GitHub AutoGen 拉取请求)
> - `Course Location on Campus` (校园课程地点)


- **文章评分查询**：一位成员询问如何查看他们提交的三篇文章的评分，包括草稿和 LinkedIn 链接。
   - 这一询问突显了社区中对提交反馈的持续关注。
- **实时流式传输挑战**：一位成员表示希望将 **chat_manager** 的响应直接实时流式传输到前端，并指出默认情况下，响应仅在垃圾回收（garbage collection）完成后才进行流式传输。
   - 另一位成员确认存在一个可以实时流式传输响应的 Streamlit UI，并提到它是大约 8 个月前构建的。
- **Chainlit 解决方案**：一位成员指出存在使用 **Chainlit** 的解决方案，在 GitHub 的 AutoGen 项目中可能有一个可用的示例（recipe）。
   - 他们注意到这个实现似乎满足了实时对话管理的需求。
- **GitHub AutoGen 拉取请求讨论**：一位成员分享了一个相关的 [GitHub 拉取请求](https://github.com/microsoft/autogen/pull/1783)，该请求讨论了在发送消息之前对其进行处理，这对于自定义消息显示非常有用。
   - 这一进展与之前关于实时流式传输的询问相呼应。
- **课程地点查询**：一位成员询问了伯克利校园内举办某门课程的具体教室。
   - 这突显了社区在协调课程相关活动时的后勤关注。



**提到的链接**：<a href="https://github.com/microsoft/autogen/pull/1783">process message before send by sonichi · Pull Request #1783 · microsoft/autogen</a>：为什么需要这些更改？添加了一个用于在发送前处理消息的可挂钩（hookable）方法。应用示例：用于显示消息的自定义前端。重命名了其他可挂钩方法以使其更清晰...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1291495504517009430)** (5 条消息): 

> - `Building AI agents with LlamaCloud` (使用 LlamaCloud 构建 AI Agent)
> - `Security in RAG` (RAG 中的安全性)
> - `Real-time audio APIs from OpenAI` (来自 OpenAI 的实时音频 API)
> - `Avoiding hallucination in RAG` (避免 RAG 中的幻觉)
> - `Hackathon announcement` (黑客松公告)


- **使用 LlamaCloud 构建 AI Agent**：了解如何使用 [LlamaCloud 和 Qdrant Engine](https://twitter.com/llama_index/status/1841935964081627315) 构建 AI Agent，重点是实现**语义缓存（semantic caching）**以提高速度和效率。
   - 该演示涵盖了高级 Agent 技术，包括**查询路由（query routing）**和**查询分解（query decomposition）**。
- **增强 RAG 部署的安全性**：讨论了将 [Box 的企业级安全性](https://twitter.com/llama_index/status/1841950022835044833)与 LlamaIndex 结合使用，以确保安全 RAG 实现的稳健权限管理。
   - 成员们强调了**无缝且具备权限感知能力的 RAG** 体验的重要性。
- **通过 OpenAI API 进行语音交互**：Marcus 展示了一项使用 [OpenAI 实时音频 API](https://twitter.com/llama_index/status/1842236491784982982) 的新功能，允许用户通过语音命令与文档进行对话。
   - 这种创新方法通过支持语音对话简化了文档交互。
- **对抗 RAG 中的幻觉**：为了防止 RAG 中的幻觉，[CleanlabAI 的解决方案](https://twitter.com/llama_index/status/1842259131274817739)集成了一个信任度评分系统来评估 LLM 的响应。
   - 该方法有助于识别并消除低质量数据点，从而提升整体数据集的质量。
- **激动人心的黑客松机会**：第二届黑客松将于 10 月 11 日在帕罗奥图的 [500 Global VC 总部](https://twitter.com/llama_index/status/1842274685989576947)拉开帷幕，奖金总额超过 **12,000 美元**。
   - 参与者可以在整个周末竞争现金奖励的同时，学习构建令人兴奋的项目。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1291482823131402252)** (11 条消息🔥): 

> - `Agent Class with Streaming`（支持流式传输的 Agent 类）
> - `Integrating LLM with BigQuery`（将 LLM 与 BigQuery 集成）
> - `Error Handling in Code`（代码中的错误处理）
> - `OpenAIAgent for Streaming`（用于流式传输的 OpenAIAgent）
> - `Custom Agent Development`（自定义 Agent 开发）


- **Agent 类需要流式传输支持**：一位用户询问是否存在支持 **chat_memory**、工具和**流式传输 (streaming)** 响应的现有 Agent 类，特别是针对函数调用和上下文管理。
   - 另一位成员建议使用 **OpenAIAgent** 或构建具有**异步流式传输 (async streaming)** 和动态上下文检索功能的自定义 Agent，并分享了一个 [Colab notebook](https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing) 作为参考。
- **将 LLM 与 BigQuery 集成**：一位用户尝试将 LLM 与 BigQuery 表集成以进行实时提示，但在过程中遇到了错误。
   - 社区建议提供具体的错误消息以便更好地协助排查，并建议使用三反引号格式化代码以提高清晰度。
- **集成过程中的代码错误**：一位用户分享了尝试将 LLM 与 BigQuery 集成的代码，但未说明遇到的具体错误。
   - 社区成员鼓励分享错误详情以便提供更有针对性的帮助，并强调了代码可读性的重要性。



**提到的链接**：<a href="https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing">Google Colab</a>：未找到描述

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1291523368247627898)** (7 条消息): 

> - `dslmodel live demos`（dslmodel 现场演示）
> - `Sentiment Analysis`（情感分析）
> - `Document Summarization`（文档摘要）
> - `Arxiv Paper Structure`（Arxiv 论文结构）
> - `New Features in DSLModel`（DSLModel 的新特性）


- **即将举行的 dslmodel 现场演示**：**dslmodel** 的现场演示定于 **PST 时间 4:30** 举行。
   - 鼓励参与者加入休息室的演示活动，进行互动式编程。
- **情感分析取得积极结果**：SentimentModel 成功将句子 “This is a wonderful experience!” 分类为 **sentiment='positive'** 且 **confidence=1.0**。
   - 这展示了该模型在情感分类任务中的可靠性。
- **摘要模型捕捉精髓**：使用 SummarizationModel 进行的文档摘要提供了一个简洁的总结：“**关于成功与坚持的励志演讲。**”
   - 该模型在推理中强调了控制力、成功和韧性等主题。
- **实现了 Arxiv 论文结构**：演示了一个 Arxiv 论文模型，其类设置包含主作者和共同作者的详细信息。
   - 该论文介绍了 **DSPy**，这是一种用于语言处理的重要编程模型。
- **Gif 捕捉到的有趣时刻**：分享了一个幽默的 Gif，显示一个穿着黑色高领毛衣的男人表情滑稽，配文为 “Mind Blow”（大开眼界）。
   - 该 Gif 用来描绘许多人对频道中分享的“惊人”概念的反应。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/mind-blow-galaxy-explode-boom-fireworks-gif-5139389">Mind Blow Galaxy GIF - Mind Blow Galaxy Explode - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/dspy.ipynb">dslmodel/src/dslmodel/examples/dspy.ipynb at main · seanchatmangpt/dslmodel</a>：来自 DSPy 和 Jinja2 的结构化输出。通过在 GitHub 上创建账户为 seanchatmangpt/dslmodel 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1291752125679800391)** (4 条消息): 

> - `DSPy full form`（DSPy 全称）
> - `Backronym for DSPy`（DSPy 的逆向首字母缩略词）


- **DSPy 代表 Declarative Self-improving Language Programs**：一位成员澄清说，目前 DSPy 的逆向首字母缩略词（backronym）是 **Declarative Self-improving Language Programs**（Python 风格）。
   - 他们幽默地提到，DSPy 也被称为 **Declarative Self-Improving Python**。
- **社区关于 DSPy 的询问**：一位社区成员询问 DSPy 的全称，引发了关于其含义的讨论。
   - 这一询问促成了关于该缩写词解释和相关幽默的友好交流。


  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1291704918096347198)** (4 条消息): 

> - `Text Classification Tasks` (文本分类任务)
> - `DSPy Signatures`
> - `LM Behavior Specification` (LM 行为规范)


- **分享文本分类示例**：一位用户请求关于 **文本分类任务** 的示例。
   - *希望这有所帮助！*
- **理解 DSPy Signatures**：另一位用户分享了一个链接，解释 **DSPy Signatures** 是模块中输入/输出行为的声明式规范。
   - 这些 Signatures 允许用户定义和控制模块行为，与仅描述参数的典型函数签名形成对比。



**提到的链接**：<a href="https://dspy-docs.vercel.app/docs/building-blocks/signatures#example-c-classification">Signatures | DSPy</a>：当我们在 DSPy 中为 LM 分配任务时，我们将所需的行为指定为 Signature。

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1291482053287874742)** (7 条消息): 

> - `Event Participation Limit` (活动参与人数限制)
> - `Human Devices Event` (Human Devices 活动)
> - `Obelisk GitHub Tool` (Obelisk GitHub 工具)


- **活动参与人数限制回退至 25 人**：成员们注意到活动的参与人数上限被限制在 **25 人**，尽管 MikeBirdTech 曾提议更改为 **99 人**。
   - 一位用户确认多次尝试加入但仍显示 **已满 (full)** 状态。
- **加入 Human Devices 活动**：MikeBirdTech 分享了即将举行的 **Human Devices 活动** 链接，并提供了一个用于访问的 Discord URL：[点击此处加入](https://discord.gg/mzcrk6pZ?event=1291393902758330389)。
   - 鼓励参与者在指定频道中 **请求或分享** 与活动相关的任何内容。
- **Obelisk：一个实用的 GitHub 工具**：一位成员重点推荐了 GitHub 上的 **Obelisk** 项目，这是一个将网页保存为单个 **HTML 文件** 的工具。
   - 他们认为这在许多场景下都 **非常有用**，并提供了链接供他人探索：[GitHub - go-shiori/obelisk](https://github.com/go-shiori/obelisk)。



**提到的链接**：<a href="https://github.com/go-shiori/obelisk">GitHub - go-shiori/obelisk: Go package and CLI tool for saving web page as single HTML file</a>：用于将网页保存为单个 HTML 文件的 Go 语言包和 CLI 工具 - go-shiori/obelisk

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 条消息): 

ellsies_: 完全没有日志
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1291579642226151466)** (5 条消息): 

> - `Meta Movie Gen`
> - `Open Source Discussion` (开源讨论) 


- **Meta Movie Gen 发布**：今天，[Meta 首映了 Movie Gen](https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)，这是一系列旨在增强视频和音频创作的先进多媒体基础模型。
   - 该模型可以生成高质量的图像和视频，以及与视频同步且具有出色对齐度和质量的音频。
- **来自 Mozilla 的开源愿景**：在回答关于 Meta Movie Gen 开放性的查询时，一位成员澄清说，虽然 **Mozilla** 倡导开源，但这一举措更多是为了展示他们的愿景。
   - 讨论强调了 Mozilla 的原则与 Movie Gen 性质之间的区别，强调其仍与其更广泛的目标保持一致。



**提到的链接**：<a href="https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 AI at Meta (@AIatMeta) 的推文</a>：🎥 今天我们首映了 Meta Movie Gen：迄今为止最先进的多媒体基础模型。由 Meta 的 AI 研究团队开发，Movie Gen 在一系列能力上提供了最前沿的结果...

  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1291480614528090165)** (12 messages🔥): 

> - `FAANG SDLC certifications`
> - `LangChain API updates`
> - `LangChain support for GPT real-time API`
> - `Evaluating RAG pipelines`
> - `Creating a chatbot with LangChain` 


- **FAANG 公司寻求 SDLC 认证**：一位用户询问了除了 **PMP** 之外，FAANG 公司认可的、广泛承认的 **软件开发生命周期 (SDLC)** 课程或认证。
   - 这反映了从不同行业转型到科技岗位的申请人的共同关注点。
- **LangChain 中 API 调用方式的变化**：一位用户提到注意到 LangChain 的 **API chain** 发生了变化，正在寻求最新的 API 调用方法。
   - 这表明 LangChain 框架内正在进行持续的更新和开发。
- **关于 LangChain 对 GPT real-time API 支持的咨询**：一位用户询问 **LangChain** 何时会支持新发布的 **GPT real-time API**。
   - 回复中包含了一个 [YouTube 视频](https://www.youtube.com/watch?v=TdZtr1nrhJg) 链接以作进一步说明。
- **评估 RAG pipeline 检索器**：一位用户寻求关于如何评估和比较其 **RAG pipeline** 中三种不同 **retrievers** 性能的建议。
   - 另一位成员建议使用 **query_similarity_score** 来确定性能最佳的检索器，并表示愿意通过 LinkedIn 提供代码片段。
- **使用 LangChain 构建聊天机器人**：一位用户寻求关于使用 **LangChain** 创建自己的 **chatbot** 的指导。
   - 这反映了利用 LangChain 进行聊天机器人开发的兴趣日益增长。


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1291484840847216691)** (3 messages): 

> - `NeurIPS 2024 Conference Date Change`
> - `Elon Musk's xAI Recruiting Event`
> - `OpenAI's Dev Day`
> - `Funding Rumors` 


- **NeurIPS 2024 为 Taylor Swift 粉丝调整日期**：**NeurIPS 2024** 会议的开始日期已移至 **12 月 10 日星期二**，以便代表们能在前一天到达。
   - 这一变化被幽默地指出是受到了 **Taylor Swift 的 Eras Tour** 的影响，该巡演导致了计划的变动。
- **Elon Musk 举办安保严密的 xAI 招聘派对**：**Elon Musk 的 xAI** 招聘活动现场播放了通过代码生成的音乐，而参与者则面临金属探测器筛选和身份检查。
   - 该活动的时间恰逢 **OpenAI 的 Dev Day**，在 Musk 寻求人才且融资传闻不断的背景下引发了热议。
- **OpenAI CEO 在座无虚席的 Dev Day 发表演讲**：在 Musk 活动的同一天，**OpenAI** 的 CEO **Sam Altman** 在年度 **Dev Day** 期间向挤满礼堂的开发者发表了讲话。
   - 有传言称 OpenAI 可能即将完成迄今为止规模最大的初创公司融资。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/10/3/24261160/elon-musk-xai-recruiting-party-openai-dev-day-sam-altman">走进 Elon Musk 在 OpenAI 旧总部举办的 AI 派对</a>：Elon Musk 在 OpenAI 最初的旧金山总部举办了一场 xAI 招聘派对。</li><li><a href="https://fxtwitter.com/WilliamWangNLP/status/1841879266142904469">William Wang (@WilliamWangNLP) 的推文</a>：突发：Taylor Swift 的 Eras Tour 刚刚做到了 AI 做不到的事情——将 NeurIPS 推迟了一整天！🤖 🤣🤣🤣 #NeurIPS 2024 会议日期变更。会议开始日期已更改为 12 月 10 日星期二...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1291620845357236269)** (8 条消息🔥): 

> - `Meta Movie Gen`
> - `Model Optimization Techniques`（模型优化技术）
> - `LLMs and Code Synthesis Reinforcement Learning`（LLMs 与代码合成强化学习）
> - `OpenAI's Model Distillation`（OpenAI 的模型蒸馏）
> - `Canvas Development`（Canvas 开发）


- **Meta Movie Gen 发布高级功能**：Meta 首映了 [Movie Gen](https://go.fb.me/kx1nqm)，这是一系列能够根据文本提示生成高质量图像、视频和音频的多媒体基础模型（media foundation models），拥有如个性化视频生成等令人印象深刻的能力。
   - *我们正继续与创意专业人士密切合作*，在潜在发布前增强该工具的功能。
- **创新的模型布局优化**：在 Movie Gen 论文中强调，Meta 开发了建模工具来优化训练期间的布局（layout），从而实现了一种复杂的并行策略，有效地使模型与硬件相匹配。
   - *这种优化提高了训练效率*，并提升了在视频和音频生成任务中的性能。
- **强化学习增强 LLMs 的代码能力**：一篇新论文提出了一种针对作为 Agent 部署的 LLMs 的端到端强化学习方法，在利用执行反馈的同时，在竞赛编程任务中取得了 state-of-the-art 的结果。
   - 该方法展示了在迭代代码合成方面的显著改进，在使用更小模型的同时大幅减少了样本需求。
- **利用 OpenAI 蒸馏技术进行 Canvas 开发**：一位开发者分享了关于构建 Canvas 的见解，利用新颖的合成数据技术在无需人工生成数据的情况下增强交互，特别是利用了来自 OpenAI o1-preview 的蒸馏（distillation）。
   - *开发者可以利用 DevDay 宣布的新 [distillation product](https://openai.com/index/api-model-distillation/)* 来复制这些改进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1842188252541043075">来自 AI at Meta (@AIatMeta) 的推文</a>: 🎥 今天我们首映了 Meta Movie Gen：迄今为止最先进的多媒体基础模型。由 Meta 的 AI 研究团队开发，Movie Gen 在一系列能力上提供了 state-of-the-art 的结果...</li><li><a href="https://x.com/nickaturley/status/1842281132265484595">来自 Nick Turley (@nickaturley) 的推文</a>: 构建 Canvas 时我最喜欢的事情之一：我们使用了新颖的合成数据生成技术，例如蒸馏来自 OpenAI o1-preview 的输出，来微调 GPT-4o 以开启 canvas、进行修改...</li><li><a href="https://arxiv.org/abs/2410.02089">RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning</a>: 作为 Agent 部署的大语言模型（LLMs）通过多个步骤解决用户指定的任务，同时将所需的人工参与降至最低。至关重要的是，此类 LLMs 需要将其生成的代码锚定在执行反馈中...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842032577164804571?s=46">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>: 期待明天…… 👀</li><li><a href="https://fxtwitter.com/xlr8harder/status/1842199810763370742">来自 xlr8harder (@xlr8harder) 的推文</a>: Meta Movie Gen 论文中最酷的事情之一是 Meta 构建了建模工具来优化训练期间模型的布局，这使他们能够使用一种复杂且高度优化的并行策略...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 条消息): 

natolambert: 我应该把它做成会议上的正式海报吗？
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1291585301059342386)** (4 messages): 

> - `张量的 Permuting 与 Reshaping`
> - `Stable Diffusion 模型训练`
> - `Tinygrad CI 警告`
> - `CI 测试失败分析` 


- **针对目标张量的 Permuting 与 Reshaping**：一位成员询问是将大小为 (1024,1,14,1) 的目标张量进行 `.permute` 还是 `.reshape`，以匹配所需的 (14,1024,1) 形状。这次讨论强调了深度学习框架中张量操作的细微差别。
   - *Dumb q.* 暗示了围绕该张量转换问题的某种程度的挫败感或困惑。
- **在 M3 MacBook Air 上训练 Stable Diffusion**：一位成员询问是否存在可以在标准 **M3 MacBook Air** 上在 **48 小时** 内完成 **stable diffusion** 训练的模型。这一询问反映了人们对在消费级硬件上进行高效模型训练日益增长的兴趣。
   - 该问题表明需要易于获取的资源和指导来进行高效的模型训练。
- **探索 Tinygrad CI 警告**：号召对分析 Tinygrad [测试运行期间的警告](https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880) 感兴趣的人员参与。这些见解有助于优化框架的稳定性和可靠性。
   - 链接的 CI 运行展示了最近的更改，包括节点清理和本地 **metal test speeds** 增强。
- **CI 测试失败的历史分析**：一位用户表示有兴趣对历史 **CI runs** 中 **失败** 的测试以及 **从未失败** 的测试进行全面分析。此类分析可以为测试可靠性和代码稳定性提供宝贵的见解。
   - 这一请求表明了改进持续集成流程的积极态度。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880">node cleanup + local metal test speed [pr] · tinygrad/tinygrad@2a8b305</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - node cleanup + local metal test speed [pr] · tinygrad/tinygrad@2a8b305

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1291607734948593727)** (2 messages): 

> - `bfloat16 测试`
> - `Triton 演讲` 


- **呼吁在 Tinygrad 中增加更多 bfloat16 测试**：George 在最近的一次讨论中强调了 tinygrad 需要 **更多 bfloat16 测试**，并提到了 `test_dtype.py` 中现有的有限测试。
   - 一位成员询问哪些 *额外测试* 会有利于增强测试框架。
- **可观看富有见地的 Triton 演讲**：一位成员分享了 YouTube 上一个 **Triton 演讲** 的链接，讨论了与 Triton 技术相关的各个方面和发展。
   - 感兴趣探索 Triton 功能的人可以在 [这里](https://www.youtube.com/watch?v=ONrKkI7KhU4) 观看该演讲。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

leoandlibe: 嘿伙计们，torchtune 支持 KTO 训练吗？~
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1291484591583924265)** (5 messages): 

> - `VinePPO`
> - `Flex Attention`
> - `Batch Size Optimization`
> - `Distributed Data Parallel (DDP)` 


- **VinePPO 为 LLM 推理的 RL 带来革命性变化**：一位成员强调，**VinePPO**（对 PPO 的一种改进）显示出优于无 RL 方法和标准 PPO 的显著改进，其实现的训练结果**步数减少高达 9 倍**，**时间缩短 3 倍**，且**内存占用减半**。
   - 正如讨论线程中所指出的，这引发了对 **RL post-training** 的重新思考。
- **Flex Attention 实现了运行性能的提升**：一位成员讨论了由于 Attention Mask 的 **block sparsity**（块稀疏性），**Flex Attention** 在处理拼接样本的 Batch 时应保持类似的运行时间。
   - 另一位成员证实，测试显示 **bsz=1** 且包含 **1000 tokens** 的时间与内存表现，与 **bsz=2** 且每个包含 **500 tokens** 的表现相同。
- **探索 Packed Runs 中的 Batch Size**：一位成员建议在利用 Packed 设置时，可能需要移除 Batch Size 选项以简化处理，主张使用 Batch Size 或 **tokens_per_pack** 来实现一致的 **bs=1**。
   - 这引发了关于效率及其对性能指标影响的疑问。
- **关于实现 DDP 的讨论**：有关于整合 **Distributed Data Parallel (DDP)** 的推测，其中每个 Sampler 设置为 **bsz=1**，从而针对单设备使用进行优化。
   - 这种方法可以增强跨设备的资源分配和性能。



**提到的链接**：<a href="https://x.com/a_kazemnejad/status/1841888338816455033/photo/1">来自 Amirhossein Kazemnejad (@a_kazemnejad) 的推文</a>：VinePPO 是对 PPO 的一种简单改进，释放了 RL 在 LLM 推理方面的真正潜力。它击败了无 RL 方法（DPO 和 RestEM）以及 PPO，在更少的步数（高达 9 倍）和更短的时间（高达...）内超越了它们。

  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1291546985664614451)** (4 messages): 

> - `Network Speed Improvements`
> - `Software Limitations`
> - `100 Gbps Technology`
> - `Latency vs Throughput`
> - `AI Contributions to Networking` 


- **AI 提升了网络速度，但软件仍然滞后**：成员们讨论了 **AI 的进步** 如何使 **100 Gbps** 变得比以往任何时候都便宜，而实验室中目前已有 **1.6 Tbps** 的技术。
   - *Darkmatter* 指出，软件未能跟上 **80 倍带宽增长** 的步伐，导致即使在 **10 Gbps** 时也会出现**问题**。
- **增强网络能力的紧迫性**：*Luanon404* 对这些改进表示热切期待，并称：*“是时候提高网络速度了。”*
   - 这种观点反映了人们对在当前网络环境中实现最佳 **throughput**（吞吐量）和 **latency**（延迟）的广泛关注。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1291868166791630909)** (1 messages): 

> - `axolotl packaging`
> - `dependency management` 


- **为 axolotl 探索 pip 之外的替代方案**：一位成员提出，在 **axolotl** 中**安装/更新依赖项**令人沮丧，并询问是否可以使用非 pip 打包工具（如 **uv**）作为替代方案。
   - 他们对目前正在进行的努力以及如何贡献力量以使体验更顺畅表示好奇。
- **社区参与 axolotl 开发**：该成员强调，他们愿意通过探索不同的打包选项来帮助改进 **axolotl** 库。
   - 此举旨在鼓励其他开发者加入，并缓解依赖管理中常见的挫败感。


  

---



---



---



---



---



---



---



{% else %}


> 各频道的详细拆解内容已针对邮件进行截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}