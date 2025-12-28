---
companies:
- openai
- nvidia
date: '2024-09-13T01:18:57.613300Z'
description: '**OpenAI** 发布了 **o1** 模型系列，包括 **o1-preview** 和 **o1-mini**。该系列专注于“推理时推理”（test-time
  reasoning），其输出 Token 限制扩展到了 3 万个以上。


  这些模型表现强劲：在竞技编程中排名达到前 11%（第 89 百分位），在美国数学奥林匹克预选赛中表现优异，并在物理、生物和化学的基准测试中超过了博士水平的准确率。值得注意的是，尽管
  **o1-mini** 的规模比 **gpt-4o** 更小，但其表现依然令人印象深刻。此次发布凸显了推理时计算的新“缩放法则”（scaling laws），即性能随计算量呈对数线性增长。


  此外，据报道 **英伟达（Nvidia）** 的 AI 芯片市场份额正被初创公司蚕食，开发者在 Web 开发中的偏好正从 CUDA 转向 **llama** 模型，不过英伟达在模型训练领域仍保持主导地位。这些新闻反映了推理型模型的重大突破以及
  AI 硬件竞争格局的变化。'
id: f655930c-e7b4-4c62-b373-de80ef23d0ca
models:
- o1
- o1-preview
- o1-mini
- gpt-4o
- llama
original_slug: ainews-o1-openais-new-general-reasoning-models
people:
- jason-wei
- jim-fan
title: o1：OpenAI 全新的通用推理模型
topics:
- test-time-reasoning
- reasoning-tokens
- token-limit
- competitive-programming
- benchmarking
- scaling-laws
- ai-chip-competition
- inference
- training
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**推理时推理（Test-time reasoning）就是你所需要的一切。**

> 2024年9月11日至9月12日的 AI 新闻。我们为你查看了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**216** 个频道，**4377** 条消息）。预计为你节省了 **416 分钟** 的阅读时间（以每分钟 200 字计算）。你现在可以在 AINews 讨论中艾特 [@smol_ai](https://x.com/smol_ai) 了！

o1，又名 Strawberry，又名 Q*，[终于发布了](https://openai.com/o1/#snake-video)！今天我们可以使用两个模型：o1-preview（较大的模型，定价为输入 $15 / 输出 $60）和 o1-mini（专注于 STEM 推理的蒸馏版本，定价为输入 $3 / 输出 $12）——而主模型 o1 仍在训练中。这引起了[一点点困惑](https://x.com/giffmana/status/1834306463142949338)。



![image.png](https://assets.buttondown.email/images/2bb9bd8a-db7b-404c-a9e3-268e54cfea65.png?w=960&fit=max)



有一系列相关的链接，请不要错过：

- [o1 中心 (Hub)](https://openai.com/o1/#snake-video)
- [o1-preview 博客文章](https://openai.com/index/introducing-openai-o1-preview/)
- [o1-mini 博客文章](https://news.ycombinator.com/item?id=41523050)
- [技术研究博客文章](https://openai.com/index/learning-to-reason-with-llms/)
- [o1 系统卡 (System Card)](https://openai.com/index/openai-o1-system-card/)
- [平台文档](https://platform.openai.com/docs/guides/reasoning)
- o1 团队视频和[贡献者名单](https://www.notion.so/ainews-draft-479a38e041fe4ab4b05d6f90573d967d?pvs=21) ([Twitter](https://x.com/polynoamial/status/1834346060170367031))

与今天之前的大量泄露信息一致，核心故事是更长的“推理时推断（test-time inference）”，即更长的逐步响应——在 ChatGPT 应用中，这表现为一个新的“思考（thinking）”步骤，你可以点击展开查看推理轨迹，尽管有争议的是，这些轨迹对你是隐藏的（有趣的利益冲突……）：



![image.png](https://assets.buttondown.email/images/4839e35d-1f4e-4e25-bdc7-9ffd8a327128.png?w=960&fit=max)



在底层，o1 经过训练以添加新的 **推理 Token（reasoning tokens）** ——你需要为此付费，OpenAI 也相应地将输出 Token 限制扩展到了 >30k Token（顺便提一下，这也是为什么来自其他模型的一些 API 参数，如 `temperature`、`role`、工具调用（tool calling）和流式传输（streaming），尤其是 `max_tokens` 不再受支持的原因）。


![image.png](https://assets.buttondown.email/images/b52a6e7b-024c-4476-94f4-57c63001d0da.png?w=960&fit=max)


评估结果非常出色。OpenAI o1：

- 在编程竞赛（Codeforces）问题中排名第 89 百分位，
- 在美国数学奥林匹克竞赛（AIME）预选赛中名列全美前 500 名学生之列，
- 并且在物理、生物和化学问题的基准测试（GPQA）中超过了人类博士级别的准确率。


![image.png](https://assets.buttondown.email/images/118ae994-79b0-413f-894b-f1c357fb2540.png?w=960&fit=max)


你已经习惯了新模型展示那些令人愉悦的图表，但有一张值得注意的图表在许多模型发布公告中并不常见，这可能是最重要的图表。Jim Fan 博士说得很对：我们现在有了 **推理时计算（test time compute）的缩放定律（scaling laws），而且看起来它们呈对数线性缩放**。


![image.png](https://assets.buttondown.email/images/6927a289-f8ac-46db-b7bb-55f4ca34d026.png?w=960&fit=max)



遗憾的是，我们可能永远无法得知推理能力提升的驱动因素，但 [Jason Wei 分享了一些提示](https://x.com/_jasonwei/status/1834278706522849788?s=46)：


![image.png](https://assets.buttondown.email/images/857e5294-2903-4775-854d-03dd4db02c98.png?w=960&fit=max)


通常大模型会获得所有的赞誉，但值得注意的是，许多人都在称赞 o1-mini 以其体量（比 GPT-4o 更小）所表现出的性能，所以不要错过这一点。


![image.png](https://assets.buttondown.email/images/8c69b1fd-cf49-49f1-99a9-757284698359.png?w=960&fit=max)


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

**Nvidia 与 AI 芯片竞争**

- **市场份额转移**：[@draecomino](https://twitter.com/draecomino/status/1833940572706668934) 指出，Nvidia 第一次开始向 AI 芯片初创公司失去份额，最近的 AI 会议上的讨论证明了这一点。

- **CUDA vs. Llama**：同一位用户[强调](https://twitter.com/draecomino/status/1833980354497233232)，虽然 CUDA 在 2012 年至关重要，但在 2024 年，90% 的 AI 开发者是基于 Llama 而非 CUDA 构建应用的 Web 开发者。

- **性能对比**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1833955753314713655) 指出，Nvidia 在训练领域依然没有对手，但在推理领域可能会面临一些竞争。然而，采用 fp4 推理的 B100 性能可能会超越竞争对手。

- **SambaNova 性能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833806904742604822) 分享了 SambaNova 推出了“全球最快的 API”，Llama 3.1 405B 的速度达到 132 tokens/sec，Llama 3.1 70B 达到 570 tokens/sec。

**新 AI 模型与发布**

- **Pixtral 12B**：[@sophiamyang](https://twitter.com/sophiamyang/status/1833820604618924531) 宣布发布 Pixtral 12B，这是 Mistral 的首个多模态模型。[@swyx](https://twitter.com/swyx/status/1833933507590324483) 分享的基准测试显示 Pixtral 的表现优于 Phi 3、Qwen VL、Claude Haiku 和 LLaVA 等模型。

- **LLaMA-Omni**：[@osanseviero](https://twitter.com/osanseviero/status/1833860776823562511) 介绍了 LLaMA-Omni，这是一个基于 Llama 3.1 8B Instruct 的新型语音交互模型，具有低延迟语音以及文本和语音同步生成的特点。

- **Reader-LM**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833996412339552670) 分享了关于 JinaAI 的 Reader-LM 的细节，这是一个用于网页数据提取和清洗的 Small Language Model，在 HTML2Markdown 任务上的表现优于 GPT-4 和 LLaMA-3.1-70B 等更大型的模型。

- **GOT (General OCR Theory)**：[@_philschmid](https://twitter.com/_philschmid/status/1833767227218186533) 描述了 GOT，这是一个 580M 参数的端到端 OCR-2.0 模型，在处理表格、公式和几何形状等复杂任务时优于现有方法。

**AI 研究与进展**

- **Superposition prompting**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833867343052554347) 分享了关于 superposition prompting 的研究，该技术无需微调即可加速并增强 RAG，解决了长上下文 LLM 的挑战。

- **LongWriter**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833924999884746778) 讨论了 LongWriter 论文，该论文介绍了一种从长上下文 LLM 中生成 10,000 字以上输出的方法。

- **AI 在科学发现中的应用**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1834011539721842771) 强调了使用 Agentic AI 进行自动科学发现的研究，揭示了隐藏的跨学科关系。

- **AI 安全与风险**：[@ylecun](https://twitter.com/ylecun/status/1833964689686225233) 分享了对 AI 安全的看法，认为人类水平的 AI 仍很遥远，因担心生存风险而监管 AI 研发还为时过早。

**AI 工具与应用**

- **Gamma**：[@svpino](https://twitter.com/svpino/status/1833842992274395220) 展示了 Gamma，这款应用可以在几秒钟内根据上传的简历生成功能完备的网站。

- **基于 RAG 的文档问答**：[@llama_index](https://twitter.com/llama_index/status/1833907464355647906) 介绍了 Kotaemon，这是一个用于通过 RAG 系统与文档对话的开源 UI。

- **AI 调度器**：[@llama_index](https://twitter.com/llama_index/status/1833976952354717890) 宣布即将举办一场关于使用 Zoom、LlamaIndex 和 Qdrant 构建智能会议 AI 调度器的研讨会。

**AI 行业与市场趋势**

- **AI 取代企业软件**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833893424258331040) 指出，Klarna 用 AI 驱动的内部软件取代 Salesforce 和 Workday，标志着 AI 正在蚕食大部分 SaaS 的趋势。

- **AI 估值**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833993631415300321) 分享到 Anthropic 的估值已达到 184 亿美元，跻身顶级私有科技公司之列。

- **AI 在制造业的应用**：[@weights_biases](https://twitter.com/weights_biases/status/1833882770835218638) 宣传了在 AIAI Berlin 举办的“制造业中的生成式 AI：革新工具开发”分会。

**AI 伦理与社会影响**

- **AI 检测挑战**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833897108190978293) 分享的研究显示，AI 模型和人类都难以在对话记录中区分人类和 AI。

- **涉及 AI 的刑事案件**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833875297831178379) 详细介绍了首例涉及 AI 虚增音乐流媒体播放量的刑事案件，犯罪者利用 AI 生成的音乐和虚假账号骗取版税。

**迷因与幽默**

- [@ylecun](https://twitter.com/ylecun/status/1834017038668386350) 分享了一个迷因，关于一名一年级学生因为坚持 2+2=5 而得到差评并感到沮丧。

- [@ylecun](https://twitter.com/ylecun/status/1833853430886006898) 拿《辛普森一家》的万圣节特辑开玩笑：“在斯普林菲尔德，他们在吃他们的狗！”

- [@cto_junior](https://twitter.com/cto_junior/status/1833907296482754702) 分享了一张关于 strawberry 可能是什么的幽默图片。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 复兴并改进经典的 LLM 架构**

- **新发布：Solar Pro (preview) Instruct - 22B** ([Score: 44, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1fedtgz/new_release_solar_pro_preview_instruct_22b/)): **Solar Pro** 团队发布了他们新的 **22B** 参数 **instruct model** 预览版，可在 **Hugging Face** 上获取。该模型声称是可以在 **单 GPU** 上运行的 **最佳 open model**，尽管帖子作者指出此类声明在模型发布中很常见。
  - 用户对 **4K context window** 表示失望，认为在 **2024** 年 **8K** 已被视为最低标准的情况下，这显然不足。**2024 年 11 月** 的正式发布版将提供 **更长的 context windows** 并扩展语言支持。
  - 讨论中将 **Solar Pro** 与其他模型进行了对比，一些人称赞了 **Solar 11B** 的表现。用户对 **Phi 3** 的可用性和个性改进感到好奇，因为它被视为“榜单狙击手（benchmark sniper）”，在实际应用中表现不佳。
  - 相比于声称优于 **Llama 70B** 等大型模型，该模型声称是“**单 GPU 可运行的最佳 open model**”被认为更务实。一些用户表示有兴趣尝试 Solar Pro 的 **GGUF 版本**。

- **Chronos-Divergence-33B ~ 释放经典潜力** ([Score: 72, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1ferbob/chronosdivergence33b_unleashing_the_potential_of/)): Zeus Labs 推出了 **Chronos-Divergence-33B**，这是原始 **LLaMA 1** 模型的改进版本，将有效序列长度从 **2048** 扩展到了 **16,384** tokens。该模型基于 **约 500M tokens** 的新数据训练，可以连贯地编写多达 **12K tokens** 的内容，同时保留了 LLaMA 1 的魅力并避免了重复的 "**GPT-isms**"，技术细节和量化版本可在其 [Hugging Face model card](https://huggingface.co/ZeusLabs/Chronos-Divergence-33B) 上找到。
  - 用户讨论了该模型的**叙事能力**，开发者确认其主要专注于**多轮 RP**。一些用户反映回复比预期的短，并寻求改进建议。
  - 社区对使用**旧模型作为基座**以避免新模型中存在的 "**GPT slop**" 或 "**AI smell**" 表现出兴趣。用户辩论了这种方法的有效性，并讨论了如 **InternLM 20B** 等潜在替代方案。
  - 讨论涉及了**消除偏见的训练技术**，建议包括**温和的 DPO/KTO**、**ORPO** 以及在**预训练原始文本上进行 SFT**。一些人提议**修改 tokenizer** 以消除与 GPT-isms 相关的特定 tokens。

**主题 2. 推动边界的新开源语音模型**

- **新开源文本转语音模型：Fish Speech v1.4** ([Score: 106, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1fe7fz7/new_open_texttospeech_model_fish_speech_v14/)): **Fish Speech v1.4** 是一款新发布的开源 **text-to-speech 模型**，在涵盖多种语言的 **700,000 小时** 音频数据上进行了训练。该模型推理仅需 **4GB VRAM**，使其适用于各种应用，可通过其 [官方网站](https://speech.fish.audio/)、[GitHub 仓库](https://github.com/fishaudio/fish-speech)、[HuggingFace 页面](https://huggingface.co/fishaudio/fish-speech-1.4) 获取，并包含一个交互式 [demo](https://huggingface.co/spaces/fishaudio/fish-speech-1)。
  - 用户将 **Fish Speech v1.4** 与其他开源模型进行了比较，指出虽然它比以前的版本有所改进，但在语音克隆方面仍不及 **XTTSv2**。该模型在**德语**方面的表现受到了好评。
  - Fish Speech v1.4 背后的开发团队成员来自 **SoVITS** 和 **RVC** 项目，这些项目在开源 text-to-speech 社区中备受瞩目。**RVC** 被认为是目前最好的开源选择。
  - 一位用户指出命令行二进制文件 "**fap**" (fish-audio-preprocessing) 的命名选择不太妥当，建议更改以避免尴尬的执行命令。

- **LLaMA-Omni: Seamless Speech Interaction with Large Language Models** ([分数：74，评论：30](https://reddit.com//r/LocalLLaMA/comments/1fe7owt/llamaomni_seamless_speech_interaction_with_large/)): **LLaMA-Omni** 是一款能够实现与大语言模型进行**无缝语音交互**的新模型。该模型已在 [Hugging Face](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni) 上发布，并附带了[研究论文](https://arxiv.org/abs/2409.06666)和 GitHub 上的[开源代码](https://github.com/ictnlp/LLaMA-Omni)，为语音驱动的 AI 交互领域的进一步探索和开发提供了可能。
  - **LLaMA-Omni** 仅使用 **Whisper** 的 **voice encoder** 部分来嵌入音频，而不是完整的转录模型。这种方法不同于以往使用 Whisper 完整 speech-to-text 能力的多模态方法。
  - 该模型的**硬件要求**引发了讨论，用户注意到其对 VRAM 的需求较高。开发者提到，可以使用更小的 Whisper 模型来加快推理速度，但代价是质量会有所下降。
  - 用户对该模型的有效性进行了辩论，一些人将其视为与现有解决方案相当的**概念验证（proof of concept）**。其他人则质疑其语音质量和非言语语音能力，认为它可能是独立的 ASR、LLM 和 TTS 模型的组合。

**主题 3. LLM 部署的基准测试与成本分析**

- **Ollama LLM benchmarks on different GPUs on runpod.io** ([分数：52，评论：16](https://reddit.com//r/LocalLLaMA/comments/1fe8g8z/ollama_llm_benchmarks_on_different_gpus_on/)): 作者在 **runpod.io** 上使用 **Ollama** 进行了 **GPU 和 AI 模型性能基准测试**，重点关注各种模型和 GPU 的 `eval_rate` 指标。主要发现包括：**llama3.1:8b** 在 **2x 和 4x RTX4090** 上的表现相似；**mistral-nemo:12b** 比 **llama3.1:8b** 慢 **~30%**；**command-r:35b** 的速度是 **llama3.1:70b** 的两倍；对于较小的模型，**L40 与 L40S** 以及 **A5000 与 A6000** 之间的差异极小。作者分享了一份包含研究结果的[详细电子表格](https://docs.google.com/spreadsheets/d/1dnMCBeUYHGDB2inBl6fQhaQBstI_G199qESJBxS3FWk/edit?usp=sharing)，并欢迎对基准测试潜在扩展的反馈。
  - 用户询问了 **VRAM 使用情况**和**量化（quantization）**，作者在表格中增加了 **VRAM 占用范围**。一位评论者提到在 **Runpod** 上运行 **Q8 或 fp16** 模型，以完成超出家庭硬件能力的任务。
  - 讨论了**上下文长度（context length）**对模型速度的影响，作者详细说明了他们的测试过程：使用一个标准问题（*“为什么天空是蓝色的？”*），该问题在不同模型中通常会生成 **300-500 token** 的回答。
  - 一位用户分享了使用 **kobalcpp 配合 vulkan** 来**混合 GPU 品牌**（**7900xt** 和 **4090**）的经验，在 **llama3.1 70b q4KS** 上达到了 **10-14 tokens/second**，并指出了 **llama.cpp** 与 **vulkan** 的兼容性问题。

- **LLMs already cheaper than traditional ML models?** ([分数：65，评论：40](https://reddit.com//r/LocalLLaMA/comments/1fe7kv1/llms_already_cheaper_than_traditional_ml_models/)): 该帖子比较了 **GPT-4** 模型与 **Google Translate** 之间的翻译成本，发现 **GPT-4** 选项明显更便宜：**GPT-4** 的成本为 **$20/1M tokens**，**GPT-4-mini** 为 **$0.75/1M tokens**，而 **Google Translate** 的成本为 **$20/1M 字符**（相当于 **$60-$100/1M tokens**）。作者对为什么还有人使用更昂贵的 Google Translate 表示困惑，因为 GPT-4 模型还提供了诸如**上下文理解**和**自定义提示词（prompts）**等额外优势，从而可能获得更好的翻译结果。
  - **Google Translate** 使用的是**混合 Transformer（hybrid transformer）**架构，详见 [2020 年 Google Research 博客文章](https://research.google/blog/recent-advances-in-google-translate/)。用户质疑其与较新 LLM 相比的性能，一些人将其归因于其技术陈旧。
  - **OpenAI 的 API 定价**引发了关于盈利能力的辩论。一些人认为 OpenAI 并没有从 API 调用中赚钱，而另一些人则认为他们可能已经实现了**现金流为正**，理由是其高效的基础设施以及与开源 LLM 供应商相比具有竞争力的定价。
  - 用户强调了 **Google Translate 在特定语言对和冷门语言中的可靠性**，指出 **GPT-4** 在非英语翻译中可能会遇到困难或产生无关内容。一些人提到 **Gemini Pro 1.5** 是多语言翻译的潜在替代方案。

**主题 4. 开发者拥抱 AI 编程助手，不像艺术家对待 AI 艺术**

- **我想问一个可能会冒犯很多人的问题：是否有很多程序员/软件工程师对 LLMs 在编程方面的进步感到愤愤不平，就像很多艺术家对 AI 艺术感到愤愤不平一样？** ([Score: 107, Comments: 275](https://reddit.com//r/LocalLLaMA/comments/1fet52k/i_want_to_ask_a_question_that_may_offend_a_lot_of/))：**软件工程师和程序员**通常对 **LLMs** 编程能力的提升表现得不如艺术家对 AI 艺术那样愤愤不平。编程行业的整体氛围似乎倾向于适应与整合，许多人将 **LLMs** 视为提高生产力的工具，而非职业威胁。
  - **软件工程师**通常将 **LLMs** 视为增强生产力的工具而非威胁，将其用于**样板代码生成 (boilerplate creation)**、理解新框架以及自动化繁琐工作等任务。许多人将 AI 视为编程**抽象层级 (abstraction level)** 演进的又一步。
  - 虽然 **LLMs** 在编程辅助方面很有帮助，但它们仍存在局限性，例如**幻觉产生错误代码**以及需要人工监督。一些开发者指出，**LLMs** 目前在实际应用编程方面表现不佳，主要在生成简单或简短的代码片段时表现出色。
  - 开发者之间的态度各不相同，从热衷的早期采用者到否认 AI 潜在影响的人都有。一些人对管理层的预期和潜在的就业市场影响表示担忧，而另一些人则将 AI 视为专注于更高层次问题解决和系统设计的机会。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与改进**

- **OpenAI 动态**：关于 OpenAI 即将发布的产品有很多传闻和猜测，包括潜在的 "Strawberry" 模型和 GPT-4o 语音模式。然而，许多用户对模糊的公告和延迟发布表示沮丧。一位 [OpenAI 应用研究主管发推](https://i.redd.it/33bn8t3x6aod1.jpeg)称即将“发布 (shipping)”一些东西，引发了更多猜测。

- **Fish Speech 1.4**：发布了一个新的[开源文本转语音模型](https://www.reddit.com/r/singularity/comments/1fe6zky/new_powerful_open_text_to_speech_model_fish/)，该模型在涵盖 8 种语言的 70 万小时语音上进行了训练。它仅需 4GB VRAM，每天提供 50 次免费使用。

- **Amateur Photography Lora v4**：发布了一个用于生成逼真业余风格照片的 [Stable Diffusion Lora 模型](https://www.reddit.com/r/StableDiffusion/comments/1fe7d7o/amateur_photography_lora_v4_shot_on_a_phone/)更新，在清晰度和真实感方面有所提升。

**AI 研究与突破**

- **类脑计算 (Neuromorphic computing)**：科学家报告了使用分子忆阻器在[类脑计算方面取得的突破](https://www.deccanherald.com/india/karnataka/iisc-scientists-report-computing-breakthrough-3187052)，这可能实现更高效的 AI 硬件。该设备提供 14 位精度、高能效和快速计算。

- **PaperQA2**：引入了一个可以[自主进行完整科学文献综述](https://x.com/SGRodriques/status/1833908643856818443)的 AI Agent。

**AI 行业新闻**

- **OpenAI 人员离职**：多位知名研究人员最近离开了 OpenAI，包括负责 GPT-4o/GPT-5 工作的 [Alexis Conneau](https://i.redd.it/11aqrhjzt4od1.jpeg)。

- **Ilya Sutskever 的新创业项目**：这位前 OpenAI 研究员在没有产品或收入的情况下，为他的[新 AI 公司筹集了 10 亿美元](https://command.ai/blog/ssi-5-billion-investment/)。

- **苹果的 AI 举措**：与竞争对手相比，苹果即将推出的 iPhone 16 AI 功能被[批评为“迟到、未完成且笨拙”](https://www.ai-supremacy.com/p/apples-iphone-16-shows-apple-intelligence)。

**AI 伦理与社会影响**

- **Deepfake 担忧**：Taylor Swift [对 AI 生成的 Deepfake 表示担忧](https://www.the-express.com/entertainment/celebrity-news/148376/taylor-swift-ai-fake-trump-endorsement-fears)，这些视频虚假地描绘了她支持政治候选人的场景。

- **逼真的 AI 生成图像**：[Amateur Photography Lora v4 模型](https://www.reddit.com/r/StableDiffusion/comments/1fe7d7o/amateur_photography_lora_v4_shot_on_a_phone/)展示了区分 AI 生成图像与真实照片正变得越来越困难。


---

# AI Discord 回顾

> 摘要的摘要的摘要。我们将 4o 与 o1 的对比留给您参考。

## GPT4O (gpt-4o-2024-05-13)


**1. OpenAI O1 模型发布及各界反应**

- **OpenAI O1 模型因成本问题面临批评**：**OpenAI 的 O1 模型**因其高昂的成本和不尽如人意的表现（尤其是在编程任务中）受到了用户的批评，正如 [OpenRouter](https://discord.com/channels/1091220969173028894) 中所讨论的那样。
  - 价格定为**每百万 Token 60 美元**，用户对意外成本和整体价值表示担忧，引发了关于该模型实际效用的讨论。
- **对 OpenAI O1 Preview 的评价褒贬不一**：**O1 preview** 收到了用户的混合反馈，人们质疑其相对于 **Claude 3.5 Sonnet** 等模型的改进，特别是在创意任务方面，如 [Nous Research AI](https://discord.com/channels/1053877538025386074) 所述。
  - 用户对其运行效率持怀疑态度，认为在全面推广之前可能需要进一步迭代。
- **OpenAI 发布用于推理任务的 O1 模型**：据 [OpenAI](https://discord.com/channels/974519864045756446) 报道，**OpenAI 推出了 O1 模型**，旨在增强**科学、编程和数学**等复杂任务中的推理能力。该模型已对 **ChatGPT** 的所有 Plus 和 Team 用户开放，并通过 [API](https://openai.com/o1/) 提供给第 5 级（tier 5）开发者。
  - 用户注意到 **O1 模型**在解决问题方面优于之前的版本，但一些人对价格与性能提升的比例表示怀疑。


**2. AI 模型性能与基准测试**

- **DeepSeek V2.5 发布，具备用户友好特性**：[OpenRouter](https://discord.com/channels/1091220969173028894) 宣布，**DeepSeek V2.5** 引入了**全精度提供商**，并为注重隐私的用户承诺*不记录提示词日志（no prompt logging）*。
  - 此外，**DeepSeek V2 Chat** 和 **DeepSeek Coder V2** 模型已合并到此版本中，允许无缝重定向到新模型。
- **PLANSEARCH 算法提升代码生成能力**：关于 **PLANSEARCH 算法**的研究表明，它通过在代码生成前识别多种方案来增强**基于 LLM 的代码生成**，从而促进高效输出，如 [aider](https://discord.com/channels/1131200896827654144) 中所讨论。
  - 通过缓解 LLM 输出**缺乏多样性**的问题，PLANSEARCH 在 **HumanEval+** 和 **LiveCodeBench** 等基准测试中展现了显著的性能提升。


**3. AI 训练与推理中的挑战**

- **OpenAI O1 和 GPT-4o 模型面临的挑战**：关于 **O1 和 GPT-4o** 模型的反馈强调了用户的挫败感，因为性能结果显示它们并没有明显优于之前的版本，如 [OpenRouter](https://discord.com/channels/1091220969173028894) 所述。
  - 用户主张在这些先进模型中进行更多针对实际应用的改进，并建议需要进行持续的测试验证。
- **LLM 蒸馏（Distillation）的复杂性**：[Unsloth AI](https://discord.com/channels/1179035537009545276) 的参与者表达了将 LLM 蒸馏为更小模型时面临的挑战，强调**准确的输出数据**是成功的关键。
  - 依赖高质量示例至关重要，而复杂推理的 Token 成本会导致推理速度缓慢。


**4. AI 工具与框架的创新**

- **Ophrase 和 Oproof CLI 工具变革操作流程**：一篇详细的[文章](https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn)解释了 **Ophrase** 和 **Oproof** 如何通过简化任务自动化和管理来增强命令行界面（CLI）功能，如 [HuggingFace](https://discord.com/channels/879548962464493619) 中所讨论。
  - 这些工具为工作流效率提供了实质性增强，有望为处理命令行任务带来**革命**。
- **HTML 分块（Chunking）包首次亮相**：新软件包 `html_chunking` 能够高效地对 HTML 内容进行分块和合并，同时保持 Token 限制，这对于网页自动化任务至关重要，已在 [LangChain AI](https://discord.com/channels/1038097195422978059) 中介绍。
  - 这种结构化方法确保了有效的 HTML 解析，为各种应用保留了必要的属性。


**5. AI 在各领域的应用**

- **Roblox 利用 AI 赋能游戏**：一段视频讨论了 **Roblox** 如何创新地将 AI 与游戏融合，将其定位为该领域的领导者，如 [Perplexity AI](https://discord.com/channels/1047197230748151888) 所述。
  - 这一发展标志着在游戏环境中集成先进技术的重大飞跃。
- **个人 AI 行政助理取得成功**：[Cohere](https://discord.com/channels/954421988141711382) 的一名成员成功构建了一个**个人 AI 行政助理**，该助理通过 [calendar agent cookbook](https://link.to.cookbook) 管理日程，并集成语音输入来编辑 Google Calendar 事件。
  - 该助手能够熟练地解释非结构化数据，证明其在组织考试日期和项目截止日期方面非常有用。

## GPT4O-Aug (gpt-4o-2024-08-06)


**1. OpenAI O1 模型发布与性能表现**

- **OpenAI O1 模型引发褒贬不一的反应**：**[OpenAI O1](https://openai.com/index/introducing-openai-o1-preview/)** 发布，声称增强了在**科学、编程和数学**等复杂任务中的推理能力，引发了用户的兴奋。
  - 虽然一些人称赞其解决问题的能力，但也有人批评其高昂的成本，并质疑其改进是否物有所值，特别是与 **GPT-4o** 等旧模型相比。
- **社区对 O1 实用性的担忧**：用户对 **O1** 在编程任务中的表现表示沮丧，指出隐藏的 'thinking tokens' 导致了意想不到的成本，据称价格为 **每百万 token $60**。
  - 尽管宣传中提到了其技术进步，许多用户报告称更倾向于使用 **Sonnet** 等替代方案，强调了在实际应用中进行实用性增强的必要性。


**2. DeepSeek V2.5 特性与用户反馈**

- **DeepSeek V2.5 发布并具备隐私功能**：**[DeepSeek V2.5](https://x.com/OpenRouterAI/status/1834242566738399470)** 引入了全精度提供商，并确保不记录 prompt 日志，吸引了注重隐私的用户。
  - 新版本合并了 **DeepSeek V2 Chat** 和 **DeepSeek Coder V2**，在保持向后兼容性的同时，促进了模型的无缝访问。
- **DeepSeek 端点性能受到质疑**：用户报告 **DeepSeek 端点** 性能不稳定，质疑其在最近更新后的可靠性。
  - 社区反馈表明需要提高性能的一致性，一些人正在考虑替代方案。


**3. AI 监管与社区反应**

- **加州 AI 监管法案面临反对**：围绕加州 **SB 1047 AI 安全法案** 的推测表明，由于政治动态，特别是 **Pelosi** 的参与，该法案有 **66-80% 的概率被否决**。
  - 讨论强调了其对数据隐私和 **inference compute** 的潜在影响，反映了技术创新与监管措施之间的紧张关系。
- **AI 监管讨论引发辩论**：受 Dr. Epstein 在 [Joe Rogan 的播客](https://joerogan.techwatchproject.org/) 上的见解启发，关于 AI 监管的对话揭示了对社交媒体塑造的叙事的担忧。
  - 成员们主张在 **free speech** 与 **hate speech** 之间采取平衡的方法，强调 AI 讨论中的适度管理。


**4. LLM 训练与优化方面的创新**

- **PLANSEARCH 算法提升代码生成**：对 **[PLANSEARCH 算法](https://arxiv.org/abs/2409.03733)** 的研究表明，它通过识别多样化的计划来增强 **基于 LLM 的代码生成**，从而提高效率。
  - 通过扩大生成观察结果的范围，PLANSEARCH 在 **HumanEval+** 和 **LiveCodeBench** 等基准测试中展现了显著的性能提升。
- **用于高效上下文处理的 Dolphin 架构**：**[Dolphin](https://arxiv.org/abs/2408.15518)** 架构引入了一个 **0.5B 参数** 的解码器，在将延迟降低 **5 倍** 的同时，提高了长上下文处理的能源效率。
  - 实验结果显示能源效率提高了 **10 倍**，使其成为模型处理技术领域的一项重大进展。


**5. AI 模型部署与集成的挑战**

- **Reflection 模型面临批评**：**Reflection 70B 模型** 因涉嫌伪造基准测试结果而遭到抨击，引发了对 AI 开发公平性的担忧。
  - 批评者将其贴上现有技术 wrapper（套壳）的标签，强调了模型评估透明度的必要性。
- **API 访问与集成挑战**：用户对 API 限制和授权问题表示沮丧，强调了故障排除中明确性的必要。
  - 在保持低成本的同时管理多个 AI 模型仍然是讨论中的一个经常性话题，突显了集成的挑战。

## o1-mini (o1-mini-2024-09-12)

- **OpenAI 发布具备卓越推理能力的 O1 模型**：[OpenAI's O1](https://openai.com/o1/) 模型系列增强了在复杂任务中的推理能力，在超过 **50,000 次对决**中表现优于 **GPT-4o** 和 **Claude 3.5 Sonnet** 等模型。尽管取得了进步，用户对其 **每百万 token 60 美元** 的高昂定价表示担忧，并引发了与 **Sonnet** 等替代方案的比较。
- **先进训练技术提升模型效率**：诸如 **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 之类的创新将 GPU 通信开销降低了 **4 倍**，而 **[vAttention](https://arxiv.org/abs/2405.04437)** 优化了 KV-cache 内存以实现高效推理。此外，**[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 探索了并行 token 解码，显著降低了推理延迟。
- **高价模型的成本管理策略**：社区讨论了如何优化使用以减轻 O1 等模型的成本，强调了管理 token 消耗和探索计费结构的重要性。建议包括利用 **/settings/privacy** 进行数据偏好设置，以及利用高效的 prompting 技术来最大化价值。
- **开源框架加速 AI 开发**：**[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 和 **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** 等项目为开发者提供了构建强大 AI 应用的工具。黑客松和社区活动提供了超过 **10,000 美元** 的奖金，促进了 **RAG technology** 和 AI agent 开发方面的协作与创新。
- **微调和专业模型训练受到关注**：**Qlora** 等技术促进了对 **Llama 3.1** 和 **Mistral** 等模型的有效微调，增强了在代码生成和翻译等任务中的性能。社区基准测试显示，微调后的模型可以取得具有竞争力的结果，引发了关于优化训练工作流和解决特定领域挑战的持续讨论。


## o1-preview (o1-preview-2024-09-12)

**OpenAI 的 o1 模型发布，评价褒贬不一**

- OpenAI 发布了 **[o1 模型系列](https://openai.com/index/introducing-openai-o1-preview/)**，旨在增强科学、编程和数学等复杂任务中的推理能力。虽然一些人称赞其解决问题的能力有所提高，但其他人批评其成本高昂（**每百万输出 token 60 美元**），并质疑其相对于 GPT-4o 和 Sonnet 等现有模型的性能提升。

**Reflection 70B 模型因夸大宣传面临抵制**

- **[Reflection 70B 模型](https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp)** 因涉嫌伪造基准测试数据（声称超越 GPT-4o 和 Claude 3.5 Sonnet）而受到批评。用户将其称为 *“现有技术的套壳”*，并报告其性能令人失望，引发了对 AI 模型评估透明度的担忧。

**AI 开发者采用新工具优化工作流**

- **[HOPE Agent](https://github.com/U-C4N/HOPE-Agent)** 等项目引入了动态任务分配和基于 JSON 的管理等功能，以简化 AI 编排。此外，**[Parakeet](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch)** 展示了快速训练能力，使用 **AdEMAMix optimizer** 在不到 20 小时内完成了训练。

**尽管承诺有所改进，新 AI 模型仍面临质疑**

- 用户对 **OpenAI's o1**、**DeepSeek V2.5** 和 **Solar Pro Preview** 等模型的有效性表示怀疑，理由是性能不佳且成本高昂。讨论强调了在 AI 进步中，可验证的基准测试和实际改进比营销噱头更重要。

**加州 AI 监管法案 SB 1047 引发行业担忧**

- 拟议的 **[加州 AI 监管法案 SB 1047](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047)** 引发了关于对数据隐私和推理算力潜在影响的辩论。推测认为，由于政治动态，该法案有 *“66-80% 的概率被否决”*，凸显了技术创新与监管措施之间的紧张关系。


---

# PART 1: High level Discord summaries

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek V2.5 发布，具备用户友好特性**：根据 [OpenRouter 公告](https://x.com/OpenRouterAI/status/1834242566738399470)，**DeepSeek V2.5** 的推出引入了**全精度提供商**，并为关注隐私的用户承诺*不记录 Prompt*。用户可以在 `/settings/privacy` 栏目中更新其**数据偏好**。
   - 此外，**DeepSeek V2 Chat** 和 **DeepSeek Coder V2** 模型已合并至此版本，允许无缝重定向到新模型而不会失去访问权限。
- **OpenAI O1 模型面临批评**：**OpenAI O1 模型**因其高昂的成本和在编程任务中令人失望的输出而引发用户不满，特别是由于其使用了隐藏的“思考 Token”。多位用户报告称，他们感觉该模型未能达到预期的能力。
   - 定价设定为**每百万 Token 60 美元**，用户对意外成本和所提供的整体价值表示担忧，引发了围绕该模型实际效用的讨论。
- **DeepSeek Endpoint 性能受到质疑**：社区成员报告 **DeepSeek Endpoint** 性能不稳定，理由是之前的停机问题和整体质量。用户特别质疑最近更新后该 Endpoint 的可靠性。
   - 对性能一致性的担忧表明，用户可能需要调整或更换对该 Endpoint 可靠性和输出质量的预期。
- **LLM 的多样化用户体验**：论坛讨论显示，用户对各种 **LLM** 的体验褒贬不一，许多人更倾向于 **Sonnet** 等替代方案，而非 **OpenAI 的 O1**。一些用户指出，尽管 **O1** 宣传有进步，但与其它竞争对手相比，性能仍然不足。
   - 这些交流反映出一种普遍情绪：虽然用户希望有所改进，但许多人正转向其它表现更好的模型。
- **探索 O1 和 GPT-4o 模型的局限性**：关于 **O1 和 GPT-4o** 模型反馈强调了挫败感，因为性能结果显示它们并没有明显优于之前的版本。用户主张在这些先进模型中加入更多针对现实世界应用的实际增强功能。
   - 批评者呼吁更加重视实际效果，而非未经证实的卓越推理能力宣称，并认为持续的测试对于验证至关重要。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenAI o1 定价令用户震惊**：OpenAI 的 o1 模型，包括 **o1-mini** 和 **o1-preview**，价格高昂，输入 Token 每百万 **$15.00**，输出 Token 每百万 **$60.00**，导致用户担心成本会与全职开发人员相当。
   - 用户担心使用 o1 进行调试可能会导致费用迅速飙升，这引起了社区的关注。
- **Aider 相比 Cursor 的智能优势**：**Aider** 与 **Cursor** 的对比突出了 Aider 在代码迭代和仓库映射方面的优势，这有助于**结对编程**。
   - 虽然 Cursor 允许在提交前更轻松地查看文件，但 Aider 被认为是进行复杂代码调整的更优选择。
- **PLANSEARCH 算法提升代码生成**：关于 **PLANSEARCH 算法** 的研究表明，它通过在代码生成前识别多样化的计划来增强**基于 LLM 的代码生成**，从而促进更高效的输出。
   - 通过缓解 LLM 输出**缺乏多样性**的问题，PLANSEARCH 在 **HumanEval+** 和 **LiveCodeBench** 等基准测试中表现出极具价值的性能。
- **社区热议 LLM 输出的多样性**：一篇新论文强调，LLM 输出的多样性不足会阻碍性能，导致搜索效率低下和重复的错误输出。
   - **PLANSEARCH** 通过扩大生成观察的范围来解决这一问题，从而在代码生成任务中实现显著的性能提升。
- **Aider 脚本改进易用性**：围绕增强 Aider 脚本的讨论提出了一些建议，包括定义脚本文件名和配置 .aider.conf.yml 以实现高效的文件管理。
   - 其他用户还处理了 **git ignore 问题**，提供了通过调整设置或使用命令行标志来编辑被忽略文件的解决方案。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 发布用于推理的 o1 模型**：OpenAI 发布了 **o1** 预览版，这是一个全新的 AI 模型系列，旨在增强在**科学、编程和数学**等复杂任务中的推理能力。该版本已面向 **ChatGPT** 的所有 Plus 和 Team 用户开放，并通过 [API](https://openai.com/o1/) 提供给 tier 5 开发者。
   - 用户注意到 **o1** 模型在解决问题方面优于之前的迭代，但一些人对价格与性能的提升比例表示怀疑。
- **ChatGPT 在记忆功能方面遇到困难**：成员们报告了 ChatGPT 记忆加载的持续问题，影响了数周对话中响应的一致性。一些人转而使用移动端 App，希望能获得可靠的访问，同时注意到 Windows 应用的缺失。
   - 用户对这些缺陷表示沮丧，特别是在处理对话记忆方面，同时对于新 o1 模型与 GPT-4 相比的创造力表现也褒贬不一。
- **关于提示词性能的热烈讨论**：一位成员指出，在进行优化（特别是集成了增强的物理功能）后，某个提示词的执行时间为 **54 秒**，表现良好。为了方便用户访问，明确了 **prompt-labs** 的位置。
   - 社区成员在 **Workshop** 中发现库（library）部分后感到宽慰，这揭示了在工具可用性沟通方面可能存在的差距。
- **讨论了可定制 ChatGPT 的 API 访问**：讨论强调了用户对访问可定制 ChatGPT 的 API 的好奇心，确认了其可用性，但对 o1 等模型的有效性表示不确定。对于不同模型界面和推出状态导致的用户困惑，人们表示担忧。
   - 还出现了一些关于自定义 GPT 的关键体验，包括由于疑似违反政策而导致的发布问题，引发了对 OpenAI 流程可靠性的质疑。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **对 o1 预览版的评价褒贬不一**：**o1 preview** 收到了褒贬不一的反馈，一些用户质疑其相对于 **Claude 3.5 Sonnet** 等现有模型的改进，特别是在创意任务方面。
   - 人们对其运行效率表示担忧，认为在全面推出之前可能需要进一步的迭代。
- **Dolphin 架构提升上下文效率**：**Dolphin 架构**展示了一个 **0.5B 参数**的解码器，提升了长上下文处理的能源效率，同时将延迟降低了 **5 倍**。
   - 实验结果显示能源效率提升了 **10 倍**，使其成为模型处理技术的一项重大进步。
- **探寻演绎推理的替代方案**：关于通用推理引擎可用性的讨论探索了传统 **LLM** 之外的选择，强调了对能够解决逻辑三段论的系统的需求。
   - 一个涉及土豆的示例问题说明了 LLM 在演绎推理方面的表现差距，暗示了混合系统使用的潜力。
- **Cohere 模型与 Mistral 的对比**：关于 **Cohere 模型**的反馈显示，与 **Mistral** 相比，其对齐程度和智能程度有限，表明后者性能更佳。
   - 参与者强化了 **Mistral Large 2** 与 **CMD R+** 之间的对比，展示了 Mistral 卓越的能力。
- **AI 在产品营销中的讨论**：成员们研究了能够跨平台自主管理营销任务的 **AI 模型**，表示目前缺乏可行的解决方案。
   - 提出了集成各种 API 的可能性，激发了对营销自动化未来发展的想法。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ophrase 和 Oproof CLI 工具变革操作**：一篇详细的[文章](https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn)解释了 **Ophrase** 和 **Oproof** 如何通过简化任务自动化和管理来增强命令行界面功能。
   - 这些工具为工作流效率提供了实质性的提升，承诺在处理命令行任务方面带来一场**革命**。
- **解析 Reflection 70B 模型**：[Reflection 70B](https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp) 项目展示了模型通用性的进步，利用 **Llama cpp** 提升了性能和用户交互。
   - 它开启了关于模型适应性动态的讨论，旨在围绕其功能进行可访问的社区参与。
- **发布用于 NLP 的波斯语数据集**：推出一个新的[波斯语数据集](https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian)，包含来自 Wikipedia 的 **6K 条翻译句子**，旨在辅助波斯语语言建模。
   - 这一举措增强了 NLP 领域内各种语言处理任务的资源可用性。
- **AI 监管引发褒贬不一的反应**：关于 AI 监管的对话（由 Epstein 博士在 [Joe Rogan 的播客](https://joerogan.techwatchproject.org/)中的见解引发）反映了对社交媒体平台塑造叙事的担忧。
   - 成员们主张在**言论自由**与**仇恨言论**之间采取平衡的方法，强调在 AI 讨论中进行适度引导的必要性。
- **HOPE Agent 增强 AI 工作流管理**：[HOPE Agent](https://github.com/U-C4N/HOPE-Agent) 引入了诸如**动态任务分配**和**基于 JSON 的管理**等功能，以简化 AI 编排。
   - 它与 **LangChain** 和 **Groq API** 等现有框架集成，增强了跨工作流的自动化。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 错误处理的重要性**：讨论集中在确保 Mojo 中的 syscall 接口返回有意义的错误值，鉴于 C 等语言中的接口契约，这一点至关重要。
   - 设计这些接口需要深入了解潜在错误，以便进行有效的 syscall 响应管理。
- **将 Span[UInt8] 转换为 String**：寻求关于在 Mojo 中将 `Span[UInt8]` 转换为字符串视图的指导，建议指向了 `StringSlice`。
   - 为了正确初始化 `StringSlice`，在遇到初始化错误后，注意到 `unsafe_from_utf8` 关键字参数是必需的。
- **ExplicitlyCopyable Trait RFC 提案**：有人建议为 `ExplicitlyCopyable` trait 发起一个 RFC，要求实现 `copy()`，这可能会影响未来的更新。
   - 据参与者称，这可以显著减少对现有定义的破坏性变更。
- **MojoDojo 的开源协作**：社区发现 [mojodojo.dev](https://github.com/modularml/mojodojo.dev) 现已开源，为协作铺平了道路。
   - 该资源最初由 Jack Clayton 创建，是学习 Mojo 的游乐场。
- **Mojo 中的推荐系统**：有关于开发推荐系统的现有 Mojo 或 MAX 功能的咨询，结果显示两者仍处于“自行构建”阶段。
   - 成员们注意到这些功能正在开发中，尚未完全建立。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **OpenAI O1 引发热潮**：用户们正在热烈讨论 [OpenAI O1](https://openai.com/index/introducing-openai-o1-preview/) 的发布，这是一个承诺增强推理和复杂问题解决能力的模型系列。
   - 推测认为它可能会集成来自反思和面向 Agent 框架（如 Open Interpreter）的功能。
- **对 AI 音乐的质疑**：成员们对 AI 驱动音乐的可持续性表示怀疑，认为这只是一个缺乏真正艺术价值的噱头。
   - 他们强调 AI 音乐缺乏赋予传统音乐深度和意义的“人类情感（human touch）”。
- **Uncovr.ai 面临开发挑战**：Uncovr.ai 的创建者分享了在平台开发过程中遇到的障碍，强调了增强用户体验的必要性。
   - 讨论中多次提到对成本管理和可持续收入模式的担忧。
- **API 限制带来的挫败感依然存在**：用户表达了对 API 限制和授权问题的挫败感，强调在故障排除时需要更高的透明度。
   - 在保持低成本的同时管理多个 AI 模型是讨论中反复出现的主题。
- **Roblox 利用 AI 赋能游戏**：一段视频讨论了 **Roblox** 如何创新地将 AI 与游戏融合，将其定位为该领域的领导者；点击[此处](https://www.youtube.com/embed/yT6Vw4n6PvI)查看视频。
   - 这一进展标志着在游戏环境中集成先进技术迈出了实质性的一步。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 模型推理能力令人印象深刻**：新发布的 [OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/) 模型旨在响应前进行更多思考，并在 MathVista 等基准测试中表现出强劲结果。
   - 用户注意到它处理复杂任务的能力，但对其未来的实际表现评价褒贬不一。
- **加州 AI 监管法案 SB 1047 引发担忧**：围绕加州 SB 1047 AI 安全法案的推测表明，由于政治动态（尤其是 **Pelosi** 的参与），该法案有 **66-80% 的概率被否决**。
   - 关于该法案对 **数据隐私** 和 **推理算力（inference compute）** 潜在影响的讨论，凸显了技术创新与监管措施之间的紧张关系。
- **关于 OpenAI o1 性能的基准测试推测**：初步基准测试表明，[OpenAI 的 o1-mini 模型](https://aider.chat/2024/09/12/o1.html)的表现与 gpt-4o 相当，特别是在代码编辑任务中。
   - 社区对 o1 模型与现有 LLM 的竞争情况非常感兴趣，反映了 AI 领域激烈的竞争格局。
- **了解私有模型中的 RLHF**：随着 **Scale AI** 在该领域的探索，成员们正试图揭示 **RLHF (Reinforcement Learning from Human Feedback)** 在私有模型中具体是如何运作的。
   - 这种方法旨在使模型行为与人类偏好对齐，从而增强训练的可靠性。
- **Scale AI 在领域专业知识方面的挑战**：在 **材料科学** 和 **化学** 等专业领域，预计 **Scale AI** 将面临来自资深领域专家的挑战。
   - 成员们反映，与监管较少的领域相比，处理临床环境中的数据要复杂得多，这会影响训练的有效性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Reflection 模型面临抵制**：**Reflection 70B 模型**因涉嫌伪造基准测试（benchmarks）而遭到抨击，该模型目前仍可在 Hugging Face 上获取，这引发了人们对 AI 开发公平性的担忧。
   - *批评者将其贴上“现有技术包装（wrapper）”的标签*，并强调模型评估中透明度的必要性。
- **Unsloth 仅限于 NVIDIA GPU**：Unsloth 确认其微调（finetuning）仅支持 **NVIDIA GPU**，这让潜在的 AMD 用户感到失望。
   - 讨论强调，Unsloth 优化的内存占用使其成为需要高性能项目的首选。
- **KTO 表现优于传统模型对齐**：关于 **KTO** 的见解表明，它的表现可能显著优于 DPO 等传统方法，但由于使用私有数据，使用该方法训练的模型仍未公开。
   - *成员们对 KTO 的潜力感到兴奋*，但指出一旦模型可以获取，还需要进一步的验证。
- **Solar Pro Preview 模型引发质疑**：拥有 **220 亿参数**的 **Solar Pro Preview 模型**已推出，旨在实现单 GPU 效率，同时声称性能优于更大的模型。
   - *批评者对其大胆声明的实用性表示担忧*，并回顾了以往类似公告中令人失望的情况。
- **LLM 蒸馏的复杂性**：参与者表达了将 LLM 蒸馏为更小模型的挑战，强调**准确的输出数据**是成功的关键。
   - *Disgrace6161 指出*，依赖高质量样本是关键，而由于复杂推理的 Token 成本，推理（inference）速度较慢。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 发布 o1：推理能力的规则改变者**：OpenAI 推出了 **o1 模型系列**，旨在数学和编程等多个领域表现出卓越的推理能力，因其增强的问题解决能力而备受关注。
   - 报告强调，**o1 模型**不仅优于之前的版本，还在**安全性**和**鲁棒性**方面有所提高，在 AI 技术上取得了显著飞跃。
- **Devin AI 在 o1 测试中表现出色**：对编程 Agent **Devin** 使用 OpenAI o1 模型的评估取得了令人印象深刻的结果，展示了**推理**在软件工程任务中的重要性。
   - 这些测试表明，o1 的泛化推理能力为专注于编程应用的 Agent 系统提供了显著的性能提升。
- **推理时间扩展（Inference Time Scaling）讨论**：专家们正在评估与 o1 模型相关的**推理时间扩展**方法的潜力，认为它可以与传统的训练扩展（training scaling）相媲美，并增强 LLM 的功能。
   - 讨论强调需要衡量隐藏的推理过程，以了解它们对 o1 等模型运行成功的影响。
- **社区对 o1 的复杂情绪**：AI 社区对 o1 模型表达了多种情绪，一些人对其与 **Sonnet/4o** 等早期模型相比的效能表示怀疑。
   - 对话的重点包括对非领域专家使用 LLM 局限性的担忧，强调了 AI 专用工具的必要性。
- **期待 o1 的未来发展**：社区热切期待 o1 模型的后续发展，特别是对潜在**语音功能**的探索。
   - 虽然人们对 o1 的认知能力感到兴奋，但一些用户在语音交互等功能层面仍面临限制。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Hackathon 令人兴奋的计算额度**：组织团队为黑客松参与者争取到了 **$300K** 的云端额度，以及一个 **10 节点 GH200 集群**和一个 **4 节点 8 H100 集群**的使用权。
   - 此次机会包括对节点的 **SSH access**，允许使用 **Modal stack** 进行无服务器扩展（serverless scaling）。
- **对 torch.compile 的强力支持**：在 MobiusML 仓库的 [0.2.2 版本](https://github.com/mobiusml/hqq/releases/tag/0.2.2)中，`torch.compile` 已集成到 `model.generate()` 函数中，提升了易用性。
   - 此次更新意味着模型生成不再依赖之前的 **HFGenerator**，简化了开发者的工作流程。
- **GEMM FP8 实现进展**：最近的一个 [pull request](https://github.com/linkedin/Liger-Kernel/pull/185) 使用 **E4M3 representation** 实现了 FP8 GEMM，解决了 issue #65 并测试了多种矩阵尺寸。
   - 新增了 **SplitK GEMM** 的文档，以指导开发者了解其用法和实现策略。
- **Aurora Innovation 招聘工程师**：Aurora Innovation 目标在 2024 年底前实现**商业发布**，正在寻找在 **GPU acceleration** 以及 **CUDA/Triton** 工具方面具有专业知识的 L6 和 L7 级工程师。
   - 该公司最近筹集了 **4.83 亿美元**以支持其无人驾驶发布计划，彰显了投资者的显著信心。
- **评估 AI 模型与 OpenAI 的策略**：参与者对 OpenAI 相较于 **Anthropic** 等竞争对手的竞争力表示担忧，强调了对创新训练策略的需求。
   - 围绕 **Chain of Thought (CoT)** 的讨论揭示了对其实现透明度的不满，这影响了对 OpenAI 领导效能的看法。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Reflection LLM 性能担忧**：Reflection LLM 声称性能超越了 **GPT-4o** 和 **Claude Sonnet 3.5**，但其实际表现引发了大量批评，尤其是与开源变体相比。
   - 由于其 API 似乎在镜像 **Claude**，人们对其原创性产生了怀疑，促使用户质疑其有效性。
- **AI 照片生成服务探索**：有关于生成写实图像效果最佳的付费 AI 照片生成服务的咨询，引发了围绕现有选项的热烈讨论。
   - 讨论中提到了一个值得注意的替代方案：[Easy Diffusion](https://easydiffusion.github.io/)，被定位为一个强大的免费竞争对手。
- **Flux 模型性能优化**：用户报告了使用 **Flux model** 的积极体验，强调了与内存使用调整和 RAM 限制相关的显著性能提升。
   - 目前正在进行关于低 VRAM 优化的讨论，特别是与 **SDXL** 等竞争对手的比较。
- **Lora 训练故障排除**：成员们分享了在 Lora 训练过程中遇到的困难，寻求在显存（VRAM）有限的设备上进行更好配置的帮助。
   - 讨论中参考了工作流优化的资源，特别提到了 [Kohya](https://github.com/bmaltais/kohya_ss/tree/sd3-flux.1) 等训练器。
- **模型中的动态对比度调整**：一位用户探索了通过使用特定 CFG 设置来降低其光照模型对比度的方法，并提出了动态阈值（dynamic thresholding）技术。
   - 他们征求了关于在修改 CFG 值时如何平衡参数的建议，表明需要精确调整以提高输出质量。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 LM Studio 中实现 RAG 的指南**：为了在 **LM Studio** 中成功构建 **RAG** 流水线，用户应下载 **0.3.2** 版本并根据成员建议上传 PDF。另一位用户遇到了“未找到与用户查询相关的引用”错误，建议进行具体询问而非泛泛而谈。
   - 鼓励成员参考 [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) 以获取更多深入见解。
- **Ternary Models 性能问题**：成员们讨论了加载 **ternary models** 时遇到的麻烦，有人报告在访问模型文件时出错，另一人建议回退到旧的量化类型，因为最新的 “TQ” 类型仍处于实验阶段。
   - 这突显了在处理新型号时需要谨慎，并相应地调整工作流程。
- **伦敦社区见面会**：邀请用户参加在**伦敦**举行的社区见面会，讨论 Prompt Engineering 并分享经验，强调已准备好与同行工程师建立联系。与会者可以寻找携带笔记本电脑的老成员。
   - 此次活动为建立网络和交流宝贵见解提供了一个平台。
- **OpenAI O1 访问权限逐步开放**：成员们正在讨论获取 **OpenAI O1 preview** 访问权限的体验，指出开放过程是分批进行的。一些用户最近获得了访问权限，而其他人仍在等待机会。
   - 这一过程展示了新工具在社区中的逐步普及。
- **对双 4090D 配置的兴趣**：一位成员分享了使用**两块 4090D GPU**（每块配备 **96GB RAM**）的兴奋之情，但强调了电力挑战，其 **600W** 的需求需要一个小功率发电机。这种对高功耗的幽默看法引起了小组的关注。
   - 这种热情反映了成员们为了提升性能而考虑的先进配置。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **使用 Rich 增强终端输出**：[Rich](https://github.com/Textualize/rich) 是一个用于美化终端格式的 **Python** 库，极大地增强了开发者的视觉输出效果。
   - 成员们探索了各种终端操作技术，讨论了改进颜色和动画功能的替代方案。
- **LiveKit Server 最佳实践**：社区共识推荐 **Linux** 作为 **LiveKit Server** 的首选操作系统，并有在 **Windows** 上遇到麻烦的案例。
   - 一位成员幽默地提到：*“别用 Windows，到目前为止只遇到过问题，”* 缓解了其他人对操作系统选择的担忧。
- **OpenAI o1 模型预览版发布**：OpenAI 预告了 **o1** 预览版，这是一个旨在提高**科学**、**编程**和**数学**应用推理能力的新模型系列，详见其[公告](https://openai.com/index/introducing-openai-o1-preview/)。
   - 成员们对该模型比以往模型更好地处理复杂任务的潜力感到兴奋。
- **Open Interpreter 技能面临的挑战**：参与者指出 **Open Interpreter skills** 在会话结束后无法保留，影响了 Slack 消息传递等功能。
   - 社区已发起协作呼吁以解决此问题，寻求进一步调查。
- **期待 Cursor 和 o1-mini 的集成**：用户表达了对 **Cursor** 推出 **o1-mini** 支持的渴望，并用俏皮的表情符号暗示了其即将发布的功能。
   - 对 **o1-mini** 的期待表明社区对新型工具能力的需求日益增长。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Parakeet 项目引起轰动**：使用 A6000 系列模型在 **3080 Ti 上训练 10 小时**的 **Parakeet 项目**产出了显著的成果，引发了人们对 **AdEMAMix 优化器**有效性的兴趣。一位成员指出，*“这可能就是 Parakeet 能够在不到 20 小时内完成 4 层模型训练的原因。”*
   - 这一成功表明训练范式可能发生转变，同时也引发了对优化技术的进一步研究。
- **GPU 拥有情况揭示了多样化的配置**：一位成员分享了他们令人印象深刻的 GPU 阵容，拥有 **7 块 GPU**，包括 **3 块 RTX 3070** 和 **2 块 RTX 4090**。这引发了关于 GPU 命名惯例及其在当今相关性的幽默讨论。
   - 持续的讨论凸显了成员之间在硬件选择和使用上的广泛多样性。
- **训练数据质量重于数量**：一次对话强调，在训练模型时，重要的不是纯粹的数据量，而是数据的**质量**——一位成员在处理 **26k 行**数据用于 JSON 转 YAML 的用例时分享了这一观点。他们表示：*“数据量并不那么重要——更重要的是质量。”*
   - 这一交流表明人们对训练方法论中数据重要性有了更深层次的理解。
- **个人 AI 行政助理取得成功**：一位成员成功构建了一个**个人 AI 行政助理**，它通过 [calendar agent cookbook](https://link.to.cookbook) 管理日程，并集成了语音输入来编辑 Google Calendar 事件。该项目展示了 AI 在个人生产力方面的创新应用。
   - 该助理能够熟练地解释非结构化数据，事实证明对整理考试日期和项目截止日期非常有益。
- **寻求 RAG 应用的最佳实践**：一位用户询问了在 **RAG 应用**中实施护栏（guardrails）的**最佳实践**，强调解决方案应视具体语境而定。这与优化 AI 应用以实现现实实用性的持续努力相一致。
   - 他们还调查了用于评估 RAG 性能的工具，旨在确定广泛认可的指标和方法论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **简化项目贡献流程**：一位成员强调，在任何项目中提交 Issue 并随后提交 PR 是最简单的贡献方式。
   - 这种方法提供了清晰度，并促进了正在进行的项目中的协作。
- **博士毕业的喜悦**：一位成员分享了在德国完成博士学位的热情，并准备开始专注于**安全（safety）**和**多 Agent 系统（multi-agent systems）**的博士后研究。
   - 他们还强调了国际象棋和乒乓球等爱好，展示了丰富多彩的个人生活。
- **RWKV-7 凭借 Chain of Thought 脱颖而出**：仅拥有 **2.9M** 参数的 **RWKV-7** 模型在利用 **Chain of Thought** 解决复杂任务方面展示了令人印象深刻的能力。
   - 据指出，使用反转数字生成大量数据增强了其训练效率。
- **Pixtral 12B 在对比中表现不佳**：关于 **Pixtral 12B** 性能的讨论爆发，尽管其规模更大，但表现似乎逊色于 **Qwen 2 7B VL**。
   - 人们对 **MistralAI** 会议上展示的数据完整性产生了怀疑，认为可能存在疏忽。
- **多节点训练的挑战**：人们对在慢速以太网连接上进行多节点训练的可行性表示担忧，特别是在 8xH100 机器上使用 **DDP** 时。
   - 成员们一致认为，优化全局 Batch Size 对于克服性能瓶颈至关重要。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **即将举行 AI 调度器实战研讨会**：请于 **9月20日** 加入我们在 AWS Loft 的活动，学习如何使用 Zoom, LlamaIndex 和 Qdrant 构建用于提升会议效率的 **RAG 推荐引擎**。更多详情请见 [此处](https://t.co/v3Ej58AQ6v)。
   - 本次研讨会旨在利用提供转录功能的前沿工具，创造一个高效的会议环境。
- **为汽车需求构建 RAG 系统**：一个新的多文档 **Agentic RAG 系统** 将利用 LanceDB 帮助诊断汽车问题并管理保养计划。参与者可以按照 [此处](https://t.co/NgMfj95YAd) 的说明设置向量数据库，以进行有效的汽车诊断。
   - 这种方法强调了 RAG 系统在传统场景之外的实际应用中的多功能性。
- **OpenAI 模型现已在 LlamaIndex 中可用**：随着 OpenAI 的 **o1 和 o1-mini 模型** 的集成，用户可以直接在 LlamaIndex 中使用这些模型。使用 `pip install -U llama-index-llms-openai` 安装最新版本以获取完整访问权限，[详情见此](https://t.co/0EgCP45oxV)。
   - 此次更新增强了 LlamaIndex 的功能，与模型实用性的持续进步保持一致。
- **LlamaIndex 黑客松提供现金奖励**：为定于 **10月11-13日** 举行的第二届 LlamaIndex 黑客松做准备，由 Pinecone 和 Vesslai 赞助的奖金超过 **$10,000**。参与者可以 [在此](https://t.co/13LHrlQ7ER) 注册参加这个专注于 **RAG 技术** 的活动。
   - 黑客松鼓励在 RAG 应用和 AI Agent 开发方面的创新。
- **关于 AI 框架复杂性的辩论**：针对 **LangChain**, **Llama_Index** 和 **HayStack** 等框架是否对于 LLM 开发的实际应用变得 *过于复杂* 展开了讨论。引用了一篇深刻的 [Medium 文章](https://medium.com/@jlchereau/do-we-still-need-langchain-llamaindex-and-haystack-and-are-ai-agents-dead-522c77bed94e)。
   - 这突显了人们对工具设计中功能性与简洁性平衡的持续关注。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **输出评估挑战**：成员们强调，除了旨在减少幻觉的标准 Prompt 技术外，缺乏评估输出 **真实性 (veracity)** 的方法，并强调了对更好评估手段的需求。
   - 一位成员幽默地提到，“*请不要在你的下一篇出版物中引用我的网站*”，强调了在使用生成输出时的谨慎。
- **定制 DSPy 聊天机器人**：一位成员询问如何在 DSPy 生成的 Prompt 中通过后处理步骤而非硬编码来实现针对特定客户的定制，以追求灵活性。
   - 另一位成员建议利用类似于 RAG 方法的“context”输入字段，建议使用通用格式训练 Pipeline 以增强适应性。
- **O1 定价困惑**：关于 OpenAI O1 定价的讨论显示，成员们对其结构并不确定，其中一人确认 **O1-mini** 比其他选项更便宜。
   - 成员们表示有兴趣对 DSPy 和 O1 进行对比分析，建议试用 O1 以评估潜在的成本效益。
- **理解 O1 的 RPM**：一位成员澄清 “20rpm” 指的是 “每分钟请求数 (requests per minute)”，这是一个影响 O1 和 DSPy 性能讨论的关键指标。
   - 这一澄清引发了关于该指标对当前和未来集成影响的进一步询问。
- **对 DSPy 和 O1 集成的好奇**：关于 DSPy 与 **O1-preview** 兼容性的问题不断涌现，反映了社区探索这两个系统之间更多功能的渴望。
   - 这种兴趣标志着集成对于增强 DSPy 内部能力的重要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **混合精度训练复杂性**：维护 **mixed precision modules** 与其他特性之间的兼容性需要**额外的工作**；由于旧款 GPU 上的 fp16 支持问题，bf16 半精度训练被认为明显更好。
   - *简单地使用 fp16 会导致溢出错误*，增加系统复杂性，并因使用**全精度梯度 (full precision gradients)** 而增加内存占用。
- **FlexAttention 集成获批**：用于文档掩码 (document masking) 的 **FlexAttention** 集成已合并，其潜力引发了广泛关注。
   - *有人提出疑问，每个 'pack' 是否都填充到了 max_seq_len*，考虑到缺乏完美打乱 (perfect shuffle) 对收敛 (convergence) 的影响。
- **PackedDataset 在 INT8 下表现出色**：性能测试显示，在 torchao 中使用 INT8 混合精度时，**PackedDataset** 在 **A100** 上实现了 **40% 的加速**。
   - *一名成员计划进行更多测试，确认 PackedDataset 的固定 seq_len 与其 INT8 策略非常契合。*
- **Tokenizer API 标准化讨论**：一名成员建议在解决 eos_id 问题之前，先处理 issue #1503 以统一 **tokenizer API**，这意味着这可以简化开发流程。
   - *由于 #1503 已经有负责人*，该成员打算探索其他修复方案以增强整体改进。
- **提供了 QAT 澄清**：一名成员对比了 **QAT** (Quantization-Aware Training) 与 **INT8 混合精度训练**，强调了它们目标上的关键差异。
   - *QAT 旨在提高准确性，而 INT8 训练侧重于提升速度*，且在准确性损失极小的情况下可能不需要 QAT。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **OpenAI 模型引发好奇**：成员们对 **新的 OpenAI 模型** 议论纷纷，询问其特性和反响，认为这是一个值得探索的新发布。
   - *它引发了好奇心，并带动了关于其功能和反响的进一步讨论。*
- **Llama Index 关注度达到顶峰**：随着成员们分享对 **Llama Index** 交互工具的熟悉程度以及与 OpenAI 模型的潜在联系，对其兴趣日益增长。
   - *这促使人们开始探索它与新的 OpenAI 模型之间的关系。*
- **Reflection 70B 被贴上“哑弹”标签**：有成员担心 **Reflection 70B** 模型被视为**哑弹 (dud)**，引发了关于 OpenAI 新发布时机影响的讨论。
   - *该评论是以轻松的方式分享的，暗示这是对之前失望情绪的一种回应。*
- **DPO 格式预期已设定**：一名成员参考 [GitHub issue](https://github.com/axolotl-ai-cloud/axolotl/issues/1417) 澄清了预期的 **DPO** 格式为 `<|begin_of_text|>{prompt}`，后接 `{chosen}<|end_of_text|>` 和 `{rejected}<|end_of_text|>`。
   - 这一更新指向了 Axolotl 框架中自定义格式处理的改进。
- **Llama 3.1 在工具调用方面遇到困难**：**llama3.1:70b** 模型在利用工具调用 (tool calls) 时出现了无意义输出的问题，尽管功能上看起来是正确的。
   - *在一个案例中，在工具指示夜间模式已停用后，助手仍未能对后续请求做出恰当回应。*

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **RenderNet 推出的 Flux 已发布！**：来自 RenderNet 的 Flux 允许用户仅凭一张参考图即可创建**超现实 (hyper-realistic)** 图像，无需 LoRAs。可以通过此 [链接](https://x.com/rendernet_ai/status/1833865069744083198) 查看。
   - *准备好让你的角色栩栩如生了吗？* 用户只需点击几下即可轻松上手。
- **SD 团队更名**：**SD 团队**更改了名称以反映他们从 SAI 离职，引发了关于其现状的讨论。
   - *所以 SD 就这么没了？* 这条评论捕捉到了成员们对该团队未来的担忧情绪。
- **对 SD 开源状态的担忧**：成员们对 **SD** 在**开源领域**缺乏活跃度表示担忧，这表明社区参与度可能有所下降。
   - *如果你关心开源，SD 似乎已经名存实亡了*，这是一条关于其活跃度下降的显著评论。
- **SD 发布了新的仅限 API/Web 的模型**：尽管存在参与度方面的担忧，**SD 团队**最近发布了一个**仅限 API/Web 的模型**，标志着一定程度的产出。
   - 虽然对其开源承诺的初步怀疑依然存在，但这次发布表明他们仍在工作。
- **通过 Sci Scope 保持更新**：Sci Scope 汇集了带有相关主题的新 [ArXiv 论文](https://www.sci-scope.com/)，并每周进行总结，以便于 AI 研究的消化吸收。
   - *订阅时事通讯* 以获取直接投递的简明更新，增强你对当前文献的了解。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **HTML 分块包发布**：新软件包 `html_chunking` 能够高效地对 HTML 内容进行分块和合并，同时保持 Token 限制，这对于网络自动化任务至关重要。
   - *这种结构化方法确保了有效的 HTML 解析*，为各种应用保留了必要的属性。
- **共享 HTML 分块演示代码**：展示 `get_html_chunks` 的演示代码片段说明了如何在设定的 Token 限制内处理 HTML 字符串并保留其结构。
   - 输出由有效的 HTML 块组成——*过长的属性会被截断*，确保长度在实际使用的合理范围内。
- **HTML 分块与现有工具对比**：将 `html_chunking` 与 LangChain 的 `HTMLHeaderTextSplitter` 和 LlamaIndex 的 `HTMLNodeParser` 进行了对比，突出了其在保留 HTML 上下文方面的优越性。
   - 现有工具主要提取文本内容，这削弱了它们在需要完整保留 HTML 场景下的有效性。
- **开发者行动指南**：鼓励开发者探索 `html_chunking` 以增强网络自动化能力，并强调了其精确的 HTML 处理能力。
   - [HTML chunking PYPI 页面](https://pypi.org/project/html-chunking/)和 [GitHub 仓库](https://github.com/KLGR123/html_chunking)的链接提供了进一步探索的途径。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 提倡理性互动**：George Hotz 要求成员除非问题具有建设性，否则请勿进行不必要的 @mentions，鼓励在资源利用方面保持自给自足。
   - *只需一次搜索*即可找到相关信息，从而在讨论中培养一种关注相关性的文化。
- **新服务条款侧重于伦理实践**：George 为 **ML 开发者**引入了专门的**服务条款**，旨在禁止在 GPU 上进行**加密货币挖矿和转售**等活动。
   - 该政策旨在创建一个专注的开发环境，特别是利用 MacBook 的能力。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Literal AI 的易用性脱颖而出**：一位成员赞扬了 **Literal AI** 的易用性，并指出了其增强 LLM 应用生命周期的功能，详见 [literalai.com](https://literalai.com/)。
   - 集成和直观的设计被强调为促进开发者更顺畅运营的关键方面。
- **通过 LLM 可观测性提升应用生命周期**：**LLM 可观测性**被讨论为增强应用开发的重大变革，它允许更快的迭代和调试，同时利用日志对小型模型进行 Fine-tuning。
   - 这种方法旨在显著提高模型管理的**性能**并降低成本。
- **变革 Prompt 管理**：强调将 **Prompt 性能追踪**作为防止部署回归的保障，讨论指出这是获得可靠 LLM 输出的必要条件。
   - 这种方法在更新过程中主动维护质量保证。
- **建立 LLM 监控设置**：分享了关于构建强大的 **LLM 监控与分析**系统的见解，整合日志评估以维持最佳生产性能。
   - 此类设置被认为对于确保运营的持续效率至关重要。
- **Fine-tuning LLM 以获得更好的翻译**：关于专门针对翻译进行 **Fine-tuning LLM** 的讨论浮出水面，指出了 LLM 虽然能捕捉大意但往往遗漏语气或风格的挑战。
   - 这一差距为开发者在翻译能力方面的创新提供了途径。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 测试显示准确率参差不齐**：最近对 **Gorilla LLM** 的测试显示了令人担忧的结果，**irrelevance（无关性）**达到了 **1.0** 的完美**准确率**，而 **java** 和 **javascript** 的记录均为 **0.0**。
   - **live_parallel_multiple** 和 **live_simple** 等测试结果令人失望，引发了对模型有效性的质疑。
- **成员寻求 Qwen2-7B-Chat Prompt 拼接方面的帮助**：一位用户对 **qwen2-7b-chat** 的**表现欠佳**表示担忧，询问这是否源于 **Prompt 拼接**的问题。
   - 他们正在寻找可靠的见解和方法，通过有效的 Prompt 策略来增强其**测试体验**。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻求预测性维护（Predictive Maintenance）的见解**：一名成员就其在 **predictive maintenance** 方面的经验提出了疑问，并寻求有关最佳模型和实践的资源（如论文或书籍），特别是关于无需追踪故障的无监督方法。
   - 讨论强调了手动标记事件的不切实际性，强调了该领域对高效方法论的需求。
- **监控中的机械与电气重点**：讨论集中在一种兼具 **mechanical**（机械）和 **electrical**（电气）特性的设备上，该设备记录了多个操作事件，可以从改进的监控实践中受益。
   - 成员们一致认为，利用有效的监控策略可以增强维护方法，并可能降低未来的故障率。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：详细的频道摘要与链接

{% if medium == 'web' %}

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1283802157065175091)** (1 条消息): 

> - `DeepSeek V2.5`
> - `DeepSeek merger`
> - `Reflection endpoint`
> - `Data privacy` 

- **DeepSeek V2.5 发布新功能**：DeepSeek V2.5 现在包含一个新的 **full-precision provider**（全精度提供商），并确保为对数据敏感的用户提供 *no prompt logging*（无 Prompt 日志记录），详见 [官方公告](https://x.com/OpenRouterAI/status/1834242566738399470)。
   - 用户可以通过 `/settings/privacy` 栏目管理其 **data preferences**（数据偏好）。
- **DeepSeek 模型合并**：**DeepSeek V2 Chat** 和 **DeepSeek Coder V2** 模型已合并并升级为新的 **DeepSeek V2.5**，通过将 `deepseek/deepseek-coder` 重定向至 `deepseek/deepseek-chat` 确保了向后兼容性。
   - 这一变化简化了用户访问模型的方式，信息源自 [OpenRouterAI](https://x.com/OpenRouterAI/status/1834242566738399470) 的公告。
- **停止免费的 Reflection 端点**：请注意，**免费的 Reflection 端点** 即将消失，只要有可用的提供商，标准版本将继续存在。
   - 这一即将到来的变化鼓励用户为从免费访问过渡做好准备，正如 [OpenRouterAI](https://x.com/OpenRouterAI/status/1834242566738399470) 所分享的那样。

**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1834242566738399470">来自 OpenRouter (@OpenRouterAI) 的推文</a>：DeepSeek 2.5 现在拥有一个全精度提供商 @hyperbolic_labs！对于关注数据的用户，它也不记录 Prompt。您可以在 /settings/privacy 中配置您的数据偏好。

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1283522492157268093)** (778 条消息 🔥🔥🔥): 

> - `OpenAI O1 Model`
> - `DeepSeek Endpoint Performance`
> - `Model Pricing and Costs`
> - `User Experiences with LLMs`
> - `Limitations of O1 and GPT-4o`

- **OpenAI O1 模型的局限性**：用户对 OpenAI O1 模型表示不满，指出其成本高昂且性能不及预期，特别是在 coding 任务中。该模型对隐藏 “thinking tokens” 的依赖增加了用户的不满。
- **DeepSeek 端点性能**：DeepSeek 端点的性能正受到审视，用户注意到其之前的宕机记录和波动的质量。鉴于最近的更新，一些用户对其端点是否能持续稳定工作表示好奇。
- **模型定价与成本**：用户对 O1 等模型的定价结构表示担忧，认为这可能导致高昂的成本，尤其是针对 tokens 的隐藏计费。用户提到 O1 的价格为每百万 tokens 60 美元，并对可能产生的过度计费提出了疑问。
- **用户对 LLM 的使用体验**：对话强调了用户对不同 LLM 的多样化体验，一些人倾向于 OpenAI 的模型，而另一些人则表示更喜欢 Sonnet 等替代方案。少数用户报告称，尽管 O1 承诺了技术进步，但其他模型的表现更好。
- **O1 和 GPT-4o 的局限性**：关于 O1 和 GPT-4o 的反馈表明，虽然 O1 在营销中宣称具有增强的推理能力，但实际测试显示其表现可能并不比早期版本有显著提升。用户强调，结果表明这些模型需要在实际应用和改进方面下功夫。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://x.com/itsclivetime/status/1834291198640492860">Clive Chan (@itsclivetime) 的推文</a>: 隐藏功能：o1 具有 CUDA 模式（顺便说一下，确实有效）</li><li><a href="https://x.com/Foxalabs/status/1833981862194077754">Spencer Bentley (@Foxalabs) 的推文</a>: 在 10 月 2 日星期三，GPT-4o 的默认版本将更新为最新的 GPT-4o 模型 gpt-4o-2024-08-06。最新的 GPT-4o 模型输入 token 便宜 50%，输出 token 便宜 33% ...</li><li><a href="https://pastebin.com/AX0KteTX">markdown\n[LESS_THAN]system[GREATER_THAN]\nKnowledge cutoff[COLON] 2023[MINUS]10 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个您可以限时在线存储文本的网站。</li><li><a href="https://openrouter.ai/settings/preferences">设置 | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://www.riddles.com/2345">你带走的越多，留下的就越多... 谜语与答案 - Riddles.com</a>: 谜语：你带走的越多，留下的就越多。“我是什么？” 答案：你留下的脚印... </li><li><a href="https://docs.hyperbolic.xyz/docs/hyperbolic-ai-inference-pricing">AI 推理定价</a>: 基础层级使用：免费用户每分钟最多 60 次请求，账户充值至少 10 美元的用户每分钟 600 次请求。如果用户需要更高的速率限制，可以联系...</li><li><a href="https://pastebin.com/js3QRDcf">[LESS_THAN]system[GREATER_THAN]您是 Perplexity[COMMA] 一个乐于助人的搜索助手 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个您可以限时在线存储文本的网站。</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: 为模型消耗转换数据</li><li><a href="https://x.com/_xjdr/status/1834306852181737977">xjdr (@_xjdr) 的推文</a>: 首次使用 Sonnet 实现了近乎一致的复现，使用了长且巧妙的 System Prompt，并将博客中的代码和数学部分作为 ICL 示例。现在开始尝试 405B ...</li><li><a href="https://tenor.com/view/wendler-sandwich-gif-18891274">Wendler Sandwich GIF - Wendler Sandwich - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/HuggingFaceFV/finevideo">HuggingFaceFV/finevideo · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://deepinfra.com/Sao10K/L3.1-70B-Euryale-v2.2">Sao10K/L3.1-70B-Euryale-v2.2 - 演示 - DeepInfra</a>: Euryale 3.1 - 70B v2.2 是来自 Sao10k 的专注于创意角色扮演的模型。在 Web 上尝试 API</li><li><a href="https://fal.ai/models/fal-ai/openai-o1/">Openai O1 | AI Playground | fal.ai</a>: 未找到描述</li><li><a href="https://pastebin.com/Ru4p5cJK">```markdown\n&lt;system&gt;\n您是 Perplexity[COMMA] 一个乐于助人的搜索助手 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个您可以限时在线存储文本的网站。</li><li><a href="https://openrouter.ai/docs/provider-routing#ignoring-providers">提供商路由 | OpenRouter</a>: 跨多个提供商路由请求</li><li><a href="https://tenor.com/view/manoj-bajpai-gangs-of-wasseypur-sardar-khan-hiding-mysterious-gif-13671557">Manoj Bajpai Gangs Of Wasseypur GIF - Manoj Bajpai Gangs Of Wasseypur Sardar Khan - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://pastebin.com/ZCLzwTbH">当然！您可以通过 r 为您的 `ECS` 结构体实现 `From&lt;&amp;Scene&gt;` trait - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个您可以限时在线存储文本的网站。</li><li><a href="https://openrouter.ai/models/deepsee">模型：'deepsee' | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API、提供商、统计数据</a>: DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/rYzaTW4yLS">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/HeCWAZcDCp">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://gist.github.com/ShivamSphn/9b3a7e55ca1dd0687a1a8259ac02fb46">rust.rs</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/getcursor/cursor/issues/1511">无法在 OpenRouter 上使用 Claude 3.5 Sonnet，似乎是 Cursor 的问题 · Issue #1511 · getcursor/cursor</a>: 在 Windows 11 上使用 Cursor。直到最近都还正常工作，至少到上周五为止。如果我使用 anthropic/claude-3.5-sonne</li>

我收到了 Invalid API key 错误。在模型偏好设置中验证 API key 时，...</li><li><a href="https://platform.deepseek.com/api-docs/news/news0905">DeepSeek-V2.5：结合通用与编程能力的全新开源模型 | DeepSeek API Docs</a>：我们正式发布了 DeepSeek-V2.5 —— DeepSeek-V2-0628 与 DeepSeek-Coder-V2-0724 的强大结合体！这个新版本不仅保留了 Chat 模型的通用对话能力...
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1283524898412560404)** (658 条消息🔥🔥🔥): 

> - `OpenAI o1 的定价与性能`
> - `Aider vs. Cursor`
> - `o1 的用户体验`
> - `Aider 中的 System Prompt 修改`
> - `针对 o1 的 Jailbreak 尝试` 


- **OpenAI o1 的定价与性能**：OpenAI 的新模型 o1（包括 o1-mini 和 o1-preview）因其高昂的成本受到关注，价格为每百万 input tokens $15.00，每百万 output tokens $60.00。
   - 用户对费用表示担忧，指出调试和使用的成本最终可能与雇佣一名全职开发者相当。
- **Aider vs. Cursor**：用户对比了 Aider 和 Cursor，强调 Aider 凭借其 repo map 在代码迭代方面具有更强的能力，而 Cursor 在提交前的代码查看方面更加便捷。
   - 总体而言，Aider 被认为在进行代码修改时更智能，巩固了其在结对编程（pair programming）中的优势。
- **o1 的用户体验**：一位用户报告称，使用 o1-preview 处理涉及千行代码的复杂任务取得了成功，仅通过四次 prompt 就达成了结果。
   - 对于 o1 的一致性和有效性，各方反应不一，部分用户持乐观态度，而另一些用户则对其局限性表示担忧。
- **Aider 中的 System Prompt 修改**：用户讨论了在 Aider 中修改 system prompts 的影响，其中一位分享了一个详细的 super prompt，显著提升了性能。
   - 建议通过这些修改来优化 Aider 处理任务的方式，特别是在与 o1 等新模型集成时。
- **针对 o1 的 Jailbreak 尝试**：试图揭示 o1 system prompts 的 jailbreak 实验导致了意想不到的账单问题，尝试过程中生成的 thought tokens 产生了巨额费用。
   - 社区成员建议在使用 o1 时保持谨慎，强调要意识到其计费结构与 output 和生成的 tokens 之间的关系。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/cognition_labs/status/1834292718174077014/photo/1">Cognition (@cognition_labs) 的推文</a>：我们在过去几周与 OpenAI 密切合作，通过 Devin 评估 OpenAI o1 的推理能力。我们发现这一新系列模型对 Agent 系统有显著提升...</li><li><a href="https://mentat.ai/">AI-Powered Coding Assistant</a>：未找到描述</li><li><a href="https://x.com/OpenAIDevs/status/1834278701569118287">OpenAI Developers (@OpenAIDevs) 的推文</a>：OpenAI o1 并不是 gpt-4o 的继任者。不要直接替换它——你甚至可能希望将 gpt-4o 与 o1 的推理能力协同使用。了解如何为你的产品添加推理功能：http://platform....</li><li><a href="https://x.com/cognition_labs/status/1834292725417730408">Cognition (@cognition_labs) 的推文</a>：对 o1 进行 Prompt 引导有显著不同；特别是：Chain-of-thought 以及要求模型“大声思考”是以前模型的常用 Prompt。相反，我们发现要求 o1 仅...</li><li><a href="https://x.com/sama/status/1834283103038439566">Sam Altman (@sama) 的推文</a>：而且，这是一个新范式的开始：能够进行通用复杂推理的 AI。o1-preview 和 o1-mini 今天已在 ChatGPT Plus 中提供（将在数小时内逐步推出）...</li><li><a href="https://x.com/OpenAI/status/1834278218888872042">OpenAI (@OpenAI) 的推文</a>：今天开始向所有 Plus 和 Team 用户在 ChatGPT 中推出，并在 API 中向第 5 级（tier 5）开发者开放。</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet 似乎一如既往地出色</a>：自发布以来，Sonnet 在 aider 代码编辑基准测试中的得分一直保持稳定。</li><li><a href="https://aider.chat/2024/09/12/o1.html">OpenAI o1-mini 的基准测试结果</a>：新 OpenAI o1-mini 模型的初步基准测试结果。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://openrouter.ai/settings/preferences">设置 | OpenRouter</a>：管理您的账户和偏好设置</li><li><a href="https://githubnext.com/projects/copilot-workspace">GitHub Next | Copilot Workspace</a>：GitHub Next 项目：一个 Copilot 原生开发环境，专为日常任务设计。</li><li><a href="https://aider.chat/docs/leaderboards/#code-refactoring-leaderboard">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://fal.ai/models/fal-ai/openai-o1/playground">Openai O1 | AI Playground | fal.ai</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=4lXQRLcLRCg">OpenAI Strawberry 已上线</a>：加入我的时事通讯以获取定期 AI 更新 👇🏼https://www.matthewberman.com 我的链接 🔗👉🏻 主频道：https://www.youtube.com/@matthew_berman 👉🏻 剪辑频道...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API、提供商、统计数据</a>：DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#building-the-documentation">aider/CONTRIBUTING.md（位于 main 分支）· paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=wyuZzLfDhD8">Scott Wu：OpenAI o1 与编程</a>：向 OpenAI o1 问好——这是一个全新的 AI 模型系列，旨在响应前花费更多时间思考。这个新的 AI 模型系列可以进行推理...</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#install-the-project-in-editable-mode">aider/CONTRIBUTING.md（位于 main 分支）· paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider：aider 是你终端里的 AI 结对编程工具</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2409.03733">用自然语言进行规划可改进 LLM 在代码生成中的搜索</a>：虽然扩展训练算力带来了大语言模型（LLM）的显著提升，但扩展推理算力尚未产生类似的收益。我们假设一个核心缺失的组件...</li><li><a href="https://github.com/codelion/optillm/blob/main/plansearch.py">optillm/plansearch.py（位于 main 分支）· codelion/optillm</a>：为 LLM 优化推理代理。通过在 GitHub 上创建账号来为 codelion/optillm 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/issues/1503">无法使用环境变量连接远程 ollama，如何设置 · Issue #15</a></li>

03 · paul-gauthier/aider</a>: Aider 不处理 `OLLAMA_HOST` 环境变量，它返回本地的 ollama，而不是我在 ngrok 上的 Google Colab 里的那个，请帮忙修复，或者是否有我可以更改的配置文件...</li><li><a href="https://azure.microsoft.com/en-us/blog/introducing-o1-openais-new-reasoning-model-series-for-developers-and-enterprises-on-azure/">Introducing o1: OpenAI&#039;s new reasoning model series for developers and enterprises on Azure | Microsoft Azure Blog</a>: 我们很高兴将 OpenAI 最新的模型 o1-preview 和 o1-mini 添加到 Azure OpenAI Service、Azure AI Studio 和 GitHub Models。了解更多。</li><li><a href="https://azure.microsoft.com/en-us/blog/introducing-o1-openais-new-reasoning-model-series-for-develop">Introducing o1: OpenAI&#039;s new reasoning model series for developers and enterprises on Azure | Microsoft Azure Blog</a>: 我们很高兴将 OpenAI 最新的模型 o1-preview 和 o1-mini 添加到 Azure OpenAI Service、Azure AI Studio 和 GitHub Models。了解更多。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1283510120298582097)** (62 messages🔥🔥): 

> - `Aider scripting`
> - `DeepSeek integration`
> - `Git ignore issues`
> - `Environment variables in Windows`
> - `Cached token usage` 


- **Aider 脚本增强**：用户正在讨论在使用 Aider 时定义脚本文件名的可能性，并建议使用特定命令以获得更好的结果，例如在 Prompt 中包含所需的文件名。
   - 还有建议配置 `.aider.conf.yml` 以便在启动时一致地加载某些文件，例如 `CONVENTIONS.md`。
- **DeepSeek 集成问题**：用户询问了关于使用 DeepSeek 模型的问题，讨论了配置以及使用 DeepSeek API 与 OpenRouter API 访问模型功能之间的区别。
   - 澄清了如何使用正确的 API 端点和模型名称，以及调整上下文大小（context size）设置以获得更好的模型性能。
- **Aider 的 Git Ignore 问题**：一位用户表示难以编辑 `.gitignore` 中列出的文件，当 Git 拒绝包含这些文件时，Aider 会提示创建新文件而不是编辑。
   - 据指出，用户可以通过重命名或将文件添加到 Git，或者使用命令行标志来绕过 Git 检查。
- **环境变量故障排除**：用户讨论了在 Windows 上使用 DeepSeek 时遇到的所需环境变量问题，即变量已设置但未被识别。
   - 建议包括确保重启终端和正确的变量配置，并将用户引导至相关的 Aider 文档。
- **缓存 Token 使用与成本**：讨论集中在使用 Sonnet 时 Aider 中缓存 Token 相关的成本，用户澄清了与缓存 Prompt 和使用相关的预付及持续费用。
   - 建议大量缓存可能会导致迅速达到每日 Token 限制，鼓励用户根据其层级限制管理使用情况。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://pypi.org/project/ConfigArgParse/">ConfigArgParse</a>: argparse 的直接替代品，允许通过配置文件和/或环境变量设置选项。</li><li><a href="https://tenor.com/view/ponke-ponkesol-solana-monke-monkey-gif-9277508150881632018">Ponke Ponkesol GIF - Ponke Ponkesol Solana - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/deepseek.html">DeepSeek</a>: Aider 是你终端里的 AI 配对编程工具。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1283838016632389694)** (2 messages): 

> - `PLANSEARCH algorithm`
> - `LLM code generation`
> - `Diversity in LLM outputs` 


- **PLANSEARCH 算法增强代码生成**：关于 **PLANSEARCH 算法** 的新研究表明，通过在生成代码之前探索多样化的想法，它可以显著提升 **基于 LLM 的代码生成** 性能。
   - 研究显示，通过在自然语言的候选计划空间中进行搜索，可以缓解 **LLM 输出** 缺乏多样性的问题，从而实现更高效的代码生成。
- **LLM 输出的多样性是性能的关键**：论文假设 LLM 性能的核心问题在于 **缺乏多样化的输出**，导致搜索效率低下以及对错误生成的重复采样。
   - PLANSEARCH 通过生成广泛的观察结果并据此构建计划来解决这一问题，这在 **HumanEval+** 和 **LiveCodeBench** 等基准测试中展现了强劲的结果。
- **提供 PLANSEARCH 论文和实现**：对该方法细节感兴趣的人员可以点击 [此处](https://arxiv.org/abs/2409.03733) 查阅详细介绍 **PLANSEARCH 算法** 的研究论文。
   - 此外，可以在 GitHub 的 [此链接](https://github.com/codelion/optillm/blob/main/plansearch.py) 找到其实现，展示了对 LLM 推理的优化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.03733">Planning In Natural Language Improves LLM Search For Code Generation</a>：虽然扩展训练算力带来了大语言模型 (LLMs) 的显著进步，但扩展推理算力尚未产生类似的收益。我们假设一个核心缺失的组件...</li><li><a href="https://github.com/codelion/optillm/blob/main/plansearch.py">optillm/plansearch.py at main · codelion/optillm</a>：LLM 推理优化代理。通过在 GitHub 上创建账号为 codelion/optillm 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1283837401780977792)** (1 messages): 

> - `OpenAI o1 models`
> - `AI reasoning capabilities`
> - `ChatGPT updates`
> - `API access for developers` 


- **OpenAI 推出增强推理能力的 o1 模型**：OpenAI 正在发布 **o1** 预览版，这是一个全新的 AI 模型系列，旨在响应前花费更多时间进行思考，专门用于解决 **科学、编程和数学** 领域的复杂任务。
   - 此次发布从今天开始，面向所有 **ChatGPT** Plus 和 Team 用户开放，tier 5 开发者可以通过 [API](https://openai.com/o1/) 进行访问。
- **揭晓增强的问题解决能力**：**o1** 模型比之前的版本能更好地推理难题，标志着 AI 功能的一次重大飞跃。
   - 预计用户将从各个领域中 **复杂问题解决** 任务的效率提升中获益。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1283531610540478608)** (416 条消息🔥🔥🔥): 

> - `GPT-4o vs GPT o1`
> - `Advanced Voice Mode`
> - `Performance of AI Models in Math`
> - `Use of Chain of Thought`
> - `Limitations of File Upload Features` 


- **GPT-4o 与 GPT o1 的比较**：用户讨论了 GPT-4o 和 GPT o1 之间的性能差异，指出 GPT-4o 可以处理像 3^^4 这样的复杂计算，但在 3^^5 上表现吃力。
   - 一些人对 o1 的改进是否与其价格相符表示怀疑，质疑它是否仅仅是 GPT-4o 的微调版本。
- **对 Advanced Voice Mode 的挫败感**：许多用户对 Advanced Voice Mode 缓慢的推出速度以及 OpenAI 似乎缺乏优先处理该功能的兴趣表示失望。
   - 有人将对语音功能的期待与期待已久的 GTA 6 发布相提并论，反映出兴奋感正在消退。
- **AI 模型与数学基准测试**：讨论强调了 AI 模型在数学基准测试中令人惊讶的表现，一些用户质疑这是否仅通过 Chain of Thought (CoT) 就能实现。
   - 有人担心数学方面的高分是否代表 AI 能力的重大进步，呼吁进行更深入的分析。
- **Chain of Thought 的实现**：参与者讨论了 Chain of Thought 在使 AI 模型更好地执行推理任务方面的作用，同时也承认了其局限性。
   - 有推测认为当前的进步是否足以将 AI 提升到智能推理的新水平，或者是否需要未来的模型。
- **文件上传功能的限制**：用户对 GPT 目前缺乏文件上传功能表示不满，这阻碍了从图像中提取文本等任务。
   - 一些人指出，虽然高级模式中有相关功能，但限制仍然影响了特定应用的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://polymarket.com/event/will-openai-release-gpt-5-in-2024?tid=1726155867341">Polymarket | GPT-5 会在 2024 年发布吗？</a>: Polymarket | 如果 OpenAI 的 GPT-5 模型在 2024 年 12 月 31 日晚上 11:59 (ET) 之前向公众开放，该市场将结算为“是”。否则，该市场将结算为...</li><li><a href="https://polymarket.com/event/when-will-gpt-5-be-announced">Polymarket | GPT-5 何时发布？</a>: Polymarket | 这是一个关于 GPT-5 预测发布日期的市场。</li><li><a href="https://x.com/rickyrobinett/status/1825581674870055189">来自 Ricky (@rickyrobinett) 的推文</a>: 一个 8 岁的孩子在 AI 的辅助下 45 分钟能做出什么？我的女儿一直在使用 @cursor_ai 学习编程，这简直令人惊叹🤯 这里是她第二次编程课的精华...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b>">Hermes 3 405B Instruct - API, 提供商, 统计数据</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1283551207335530578)** (27 条消息🔥): 

> - `ChatGPT memory issues` (ChatGPT 记忆问题)
> - `OpenAI o1 model reactions` (OpenAI o1 模型反馈)
> - `Custom GPT publishing problems` (自定义 GPT 发布问题)
> - `Windows app status` (Windows 应用状态)
> - `API for customizable GPTs` (可定制 GPT 的 API)


- **ChatGPT 在加载记忆方面存在困难**：多位成员对浏览器端 ChatGPT 无法加载对话记忆且数周内无法稳定生成回复表示沮丧。
   - 一位成员提到切换到了应用端，而另一位成员指出目前缺少 Windows 应用，并期待其发布。
- **对 OpenAI o1 模型的评价褒贬不一**：社区正在热烈讨论新的 OpenAI o1 模型，对其与当前的 4.0 模型相比在创造力方面的表现持不同预期。
   - 批评性评论指出，基准测试表明 o1 在写作质量上可能不会超越之前的模型，尽管其声称具有更好的推理能力。
- **自定义 GPT 的发布问题**：一位成员报告称其自定义 GPT 因涉嫌违反使用政策而无法发布，且申诉被忽略。
   - 这引发了人们对个性化模型发布流程可靠性的担忧。
- **对可定制 GPT API 的兴趣**：成员们询问了访问可定制 ChatGPT API 的可能性，表示需要更好的集成。
   - 一位成员确认可以获取 API，但新模型的有效性仍有待确定。
- **o1 的访问状态和推广**：关于 o1 模型访问权限的讨论，部分成员表示刚刚获得访问权限，而其他成员仍在等待。
   - 人们对用户在模型及其界面之间的混淆表示担忧，对 o1 的整体满意度似乎一般。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1283683029512097872)** (6 条消息): 

> - `Library Location` (库位置)
> - `Prompt Performance` (Prompt 性能)


- **在左侧菜单中查找库**：一位用户澄清说，“库”可以在左侧菜单栏的“Workshop”下找到，具体在“prompt-labs”部分。
   - 如果“prompt-labs”没有立即显示，他们建议再次点击“Workshop”以使其可见。
- **用户对库信息表示感谢**：在得知库的位置后，一位用户表示感谢，说：*Bless you*。他们提到之前的搜索没有产生任何相关信息。
   - 这突显了服务器内可用资源沟通方面可能存在的差距。
- **Prompt 性能记录**：一位成员报告称，优化后的 Prompt 耗时 **54 秒**，并评论道：*还不错*。
   - 他们提到这是在集成了进一步的物理功能之后的结果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1283683029512097872)** (6 条消息): 

> - `Location of Library` (库的位置)
> - `Prompt-Labs` (Prompt-Labs)
> - `Server Search Challenges` (服务器搜索挑战)
> - `Physics Integration` (物理集成)
> - `Response Time` (响应时间)


- **轻松找到库**：一位成员分享说，“库”可以在左侧菜单栏的“Workshop”下找到，分为两个类别：**community-help** 和 **prompt-labs**。
   - 如果 **prompt-labs** 没有立即出现，再次点击 **Workshop** 即可显示。
- **Prompt-Labs 已更名**：有人指出 **prompt-labs** 是之前提到的库的新名称，提高了用户的清晰度。
   - *“Bless you. 我尝试搜索了服务器，简直没人提到过这个。”* 一位成员在终于找到它后表达了如释重负的心情。
- **Prompt 的快速响应时间**：一位成员报告其 Prompt 的响应时间为 **54 秒**，表明性能稳健。
   - 他们提到这发生在 *“集成了其余物理部分”* 之后，暗示系统效率有所提高。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1283502618332627125)** (296 条消息🔥🔥): 

> - `Nous Research AI Discord 聊天机器人性能`
> - `OpenAI o1 预览版讨论`
> - `MCTS 及其与 LLM 的关系`
> - `Consensus Prompting 技术`
> - `Hermes 3 与 o1 模型评估` 


- **o1 预览版评价褒贬不一**：用户对 o1 预览版的看法不一，指出尽管有人声称其性能更高，但在某些方面可能并不比 Claude 3.5 Sonnet 等现有模型有显著提升。
   - 用户对其在 Agent 性能和创意写作等任务中的实用性表示担忧，并经常将其与之前的模型进行对比，且评价较低。
- **LLM 的技术评估**：基准测试显示 o1-preview 处于中等计算量范围，其准确率低于预期（与 gpt-4o 相比），这表明它可能需要显著更多的资源才能达到最佳性能。
   - 从准确率与计算量的测量数据推断，o1-preview 的效率可能低于 gpt-4o，引发了关于在全面部署前还需进一步迭代的猜测。
- **MCTS 与 Prompting 技术**：讨论了蒙特卡洛树搜索 (MCTS) 在提高 LLM 推理能力方面的效用，一些用户争论其必要性，并将其与单一 Prompt 链进行对比。
   - 参与者讨论了统一的 Prompt 链是否能在没有额外结构的情况下有效复制 MCTS，并评估了其对模型性能的影响。
- **Hermes 3 中的 Reflection 模型**：Hermes 3 采用了基于 XML 的 reflection 技术并展示了 Agent 能力，这表明其训练方法旨在增强各种任务中的推理和性能。
   - 尽管在重复性任务中有所改进，但一些用户质疑这些功能的创新性，以及 Hermes 3 是否比其前代产品解锁了任何新能力。
- **用户体验与性能反馈**：多位用户分享了对 o1 模型的不满，包括一个导致高 Token 消耗且无输出的无限循环 Bug。
   - 总体反馈表明，虽然 o1 显示出潜力，但在没有用户最佳 Prompt 引导的情况下，在实际应用中可能仍会遇到困难。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/DefenderOfBasic/status/1833865957590040719">Defender (@DefenderOfBasic) 的推文</a>: 这就是我一直说的 Twitter 是全球性的 Slack。每个人都在为同一家公司工作，只是部门不同。如果你能引起社会中层管理人员的注意，他们就能……</li><li><a href="https://www.cognition.ai/blog/evaluating-coding-agents">Cognition | OpenAI o1 回顾以及我们如何评估 Coding Agent</a>: 我们是一家构建端到端软件 Agent 的应用 AI 实验室。</li><li><a href="https://x.com/sage_mints/status/1834093740241420453">Mints (@sage_mints) 的推文</a>: @D0TheMath 想象一下在给你的机器人排查故障，因为它认为自己有了自我意识，并哀求你停下来</li><li><a href="https://en.wikipedia.org/wiki/Replicant">Replicant - 维基百科</a>: 未找到描述</li><li><a href="https://x.com/D0TheMath/status/1833976648414494814">D0TheMath (@D0TheMath) 的推文</a>: 尝试排查 Sonnet 在编码过程中犯下的一些愚蠢错误的原因，结果它突然变得有了自我意识</li><li><a href="https://x.com/lukaszkaiser/status/1834281403829375168">Lukasz Kaiser (@lukaszkaiser) 的推文</a>: 我很高兴看到 o1 发布！与同事一起领导这项研究近 3 年，并研究相关想法更长时间，这让我确信：这是一个新范式。训练隐藏状态的模型……</li><li><a href="https://x.com/MatPagliardini/status/1832107984752951478">Matteo Pagliardini (@MatPagliardini) 的推文</a>: 别再丢弃旧的梯度了！介绍 AdEMAMix，一种能够超越 Adam 的新型（一阶）优化器。让我们来讨论一下动量以及极旧梯度的惊人相关性……</li><li><a href="https://x.com/Ar_Douillard/status/1833460226026033453">Arthur Douillard (@Ar_Douillard) 的推文</a>: AdEMAMix 优化器 (http://arxiv.org/abs/2409.03137) 对 Adam 的分子进行了两次 EMA，一个快的（低 \beta）和一个慢的（高 \beta）。这可以解释 FedOpt 变体的良好性能……</li><li><a href="https://minihf.com/posts/2024-08-11-weave-agent-dev-log-0/">Weave Agent 开发日志 #0 - 核心问题</a>: 未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1283517274468388957)** (48 messages🔥): 

> - `小型 LLM 推荐`
> - `将 AI 用于营销`
> - `Mistral 模型讨论`
> - `Cohere 模型的性能`
> - `AI 中的安全对齐` 


- **寻找更小的 LLM**：成员们讨论了各种小型语言模型，例如 **Qwen2-1.5b** 和 **Mistral** 的替代方案，以满足寻求比 **Llama 3.1 8B** 更小选项的用户需求。
   - 一位成员指出 **Open LLM Leaderboard** 是发现目前可用小型模型的绝佳资源。
- **用于营销任务的 AI 解决方案**：有人询问是否有 LLM 可以自主处理跨社交平台的社交产品营销，在宣传产品的同时收集反馈。
   - 虽然**目前还没有解决方案**能完全满足这一需求，但有人建议整合各种 API 可能会产生一个创业点子。
- **Mistral 的独特功能**：讨论强调了 **Mistral** 对安全对齐 (safety alignment) 的重视程度较低，这引发了褒贬不一的看法，包括对其事后审核 (post-hoc moderation) 能力的评论。
   - 一些成员指出，它可以有效地对有害提示词 (prompts) 进行分类，展示了其对特定用例的适应性。
- **Cohere 模型的弱点**：用户分享了关于 **Cohere** 模型的经验，称其对齐程度相对较弱，且与 **Mistral** 相比缺乏智能。
   - 通过对 **CMD R+** 和 **Mistral Large 2** 的比较，进一步强化了后者性能更优的看法。
- **关于 AI 安全的担忧**：表达了对 AI 安全对齐必要性的担忧，特别是 AI 应该能够独立执行审核的观点。
   - 成员们建议，在 AI 进一步进化之前，严格的对抗性测试 (adversarial testing) 可能是必不可少的，以避免未来的控制问题。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/mistralai/mistral-large">Mistral Large - API, Providers, Stats</a>: 这是 Mistral AI 的旗舰模型 Mistral Large 2 (版本 `mistral-large-2407`)。它是一个专有的权重可用模型，在推理、代码、JSON、聊天等方面表现出色。运行 Mistr...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1283594047495344160)** (4 messages): 

> - `Hallucinations in Large Language Models`
> - `Dolphin Architecture for Energy Efficiency`
> - `Contextual Processing in Language Models` 


- **幻觉：LLM 不可避免的特征**：研究认为，**Large Language Models** 中的幻觉不仅仅是错误，而是源于其基本的数学结构，这一观点得到了 [Gödel's First Incompleteness Theorem](https://arxiv.org/abs/2409.05746) 的支持。这意味着架构或数据集的改进无法消除这些幻觉。
   - *LLM 处理的每个阶段*，从训练到生成，都存在产生幻觉的非零概率，突显了这一问题的不可避免性。
- **用于高效上下文处理的 Dolphin 解码器**：**Dolphin** 架构引入了一个 0.5B 参数的解码器，旨在提高处理长上下文时的能效，显著减少了 7B 参数主解码器的输入长度，详见论文 [此处](https://arxiv.org/abs/2408.15518)。实证评估表明，能效提高了 **10 倍**，延迟降低了 **5 倍**。
   - 利用 **multi-layer perceptron (MLP)**，该设计有助于将文本编码转换为 context token embeddings，在不增加通常计算成本的情况下改进了扩展上下文处理。
- **使用更少 Prefix Tokens 的智能调节**：一次讨论强调了一种训练模型的智能方法，使其在更少的前缀 tokens 上进行调节，同时与更长历史记录所预期的 latent space 紧密对齐。一位用户指出，这涉及训练一个 MLP 来执行适当的 projection，以有效地管理信息。
   - 通过利用紧凑的 embeddings，该方法旨在使预测就像模型在更广泛的上下文中进行调节一样，展示了模型效率的一个创新方向。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/e4T97TdblP">LLMs Will Always Hallucinate, and We Need to Live With This</a>: 随着 Large Language Models 在各个领域变得越来越普遍，批判性地检查其固有的局限性变得非常重要。这项工作认为语言模型中的幻觉并非...</li><li><a href="https://arxiv.org/abs/2408.15518">Squid: Long Context as a New Modality for Energy-Efficient On-Device Language Models</a>: 本文介绍了 Dolphin，一种用于语言模型中长上下文能效处理的新型 decoder-decoder 架构。我们的方法解决了显著的能耗和延迟...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1283740086617247828)** (2 messages): 

> - `AI models reasoning`
> - `Sci Scope newsletter` 


- **新型 AI 模型强调推理**：OpenAI 开发了一个新的 **AI models** 系列，它们在回答之前会花费更多时间思考，旨在通过推理完成复杂任务。
   - 据报道，这些模型在 **science**、**coding** 和 **math** 等领域更擅长解决难题；欲了解更多详情，请访问 [OpenAI's announcement](https://openai.com/index/introducing-openai-o1-preview/)。
- **通过 Sci Scope 保持更新**：Sci Scope 按主题对 **ArXiv** 论文进行分组，并提供每周摘要，以便更轻松地浏览 AI 研究。
   - 订阅通讯，直接在您的收件箱中接收每周所有 AI 研究的概览，网址为 [Sci Scope](https://www.sci-scope.com/)。



**Link mentioned**: <a href="https://www.sci-scope.com/">Sci Scope</a>: 一份关于 AI 研究的 AI 生成报纸

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1283594047495344160)** (4 messages): 

> - `Hallucinations in Language Models`
> - `Dolphin Architecture for Long Contexts`
> - `Projector in Language Models`
> - `Efficiency in Language Processing` 


- **LLM 中不可避免的幻觉特征**：最近的一篇论文指出，大型语言模型 (LLM) 中的**幻觉 (hallucinations)** 不仅仅是偶尔的错误，而是其设计的根本特性，并引用了 **Godel's First Incompleteness Theorem** 作为依据。
   - 作者强调，改进架构或数据集无法消除幻觉，幻觉可能存在于 LLM 过程的每个阶段。
- **Dolphin 模型重新定义长上下文处理**：**Dolphin** 架构提出了一个紧凑的 **0.5B parameter** decoder，以高效管理语言模型中的广泛上下文，展示了在能耗和延迟方面的显著降低。
   - 通过将长文本上下文编码为一种独特的模态，其目标是实现 **10 倍的能效提升** 和 **5 倍的延迟降低**。
- **用于上下文信息的创新 Projector**：Dolphin 利用 **multi-layer perceptron (MLP)** projector 将来自 text encoder 的 embedding 信息转换为用于主 decoder 的 context token embeddings。
   - 该方法有效地连接了不同的 embedding 维度，桥接了紧凑的 0.5B 模型与主 **7B parameter** decoder 模型之间的信息。
- **关于上下文调节技术的讨论**：一位成员讨论了 Dolphin 如何寻求通过更少的 prefix tokens 来调节模型，以产生类似于更长历史记录的结果，并利用 MLP 进行投影。
   - 这突显了一种创新的方法，即尽管减少了上下文输入，仍能最大化模型中的 latent space 效应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.co/e4T97TdblP">LLMs Will Always Hallucinate, and We Need to Live With This</a>: 随着大型语言模型在各个领域变得越来越普遍，批判性地审视其固有局限性变得非常重要。这项工作认为语言模型中的幻觉并不是...</li><li><a href="https://arxiv.org/abs/2408.15518">Squid: Long Context as a New Modality for Energy-Efficient On-Device Language Models</a>: 本文介绍了 Dolphin，一种新型的 decoder-decoder 架构，用于语言模型中长上下文的高效能处理。我们的方法解决了显著的能耗和延迟...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1283590376569045076)** (2 messages): 

> - `Survey Paper Implementation`
> - `Alternatives to LLMs for Deductive Reasoning`
> - `General Reasoning Engines`
> - `Syllogisms and Logic Application` 


- **已实现的综述论文任务**：一位用户报告称已实现了一篇 [survey paper](https://arxiv.org/pdf/2312.11562) 中的所有任务，并正在寻求对其效用的验证，以确保这些任务尚未包含在 [repository](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks/blob/main/General_Reasoning/core_nlp_reasoning.json) 中。如果一切正常，他们计划提交一个 pull request，并标记了相关成员以寻求帮助。
   - 该 repository 旨在为医疗 LLM 及其他领域提供全面的推理任务集合，突显了其潜在的重要性。
- **关于通用推理引擎的咨询**：一位用户询问是否有不限于数学等特定领域的通用推理引擎可用，特别是考虑到 **LLM** 在演绎推理方面表现不佳。他们讨论了对能够解决三段论并应用命题逻辑和谓词逻辑的系统的需求。
   - 该用户提供了一个涉及土豆的示例问题，以说明他们感兴趣的演绎推理类型，并建议将 LLM 与推理引擎结合使用的可能性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1fey8qm/alternatives_to_llm_for_deductive_reasoning/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks/blob/main/General_Reasoning/core_nlp_reasoning.json">Open-Medical-Reasoning-Tasks/General_Reasoning/core_nlp_reasoning.json at main · openlifescience-ai/Open-Medical-Reasoning-Tasks</a>: 一个为医疗 LLM（及其他领域）提供的全面推理任务库 - openlifescience-ai/Open-Medical-Reasoning-Tasks
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1283879248762765332)** (1 messages): 

> - `CLI Tools Revolution` (CLI 工具革命)
> - `Reflection 70B Model` (Reflection 70B 模型)
> - `Persian Dataset` (波斯语数据集)
> - `Arena Learning Performance` (Arena Learning 性能)
> - `Fine-Tuning LLMs` (微调 LLMs)


- **使用 Ophrase 和 Oproof 彻底改变 CLI 工具**：一篇详细的[文章](https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn)讨论了 **Ophrase** 和 **Oproof** 如何改变命令行界面工具。
   - 作者强调了使这些工具在简化任务方面极其有效的关键特性。
- **使用 Llama cpp 探索 Reflection 70B**：[Reflection 70B](https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp) 项目展示了使用 **Llama cpp** 在模型通用性和性能方面的进展。
   - 该 Space 提供了对其能力的见解，并使其可供社区互动。
- **新波斯语数据集发布**：引入了一个新的[波斯语数据集](https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian)，包含来自 Wikipedia 的 **6K 句子**翻译。
   - 该数据集旨在支持波斯语语言建模，并增强多样化语言资源的获取。
- **通过 Arena Learning 提升性能**：一篇 [博文](https://huggingface.co/blog/satpalsr/arena-learning-post-train-data-performance-improve) 概述了 **Arena Learning** 如何在训练后（post-training）场景中提升性能。
   - 文章讨论了展示模型结果显著增强的技术。
- **在 Kubernetes 上微调 LLMs**：一份富有洞察力的指南详细介绍了如何利用 Intel® Gaudi® 加速器技术在 **Kubernetes 上微调 LLMs**。
   - 该帖子为希望使用云基础设施优化模型训练工作流的开发人员提供了实用资源。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1283527844655529985)** (324 messages🔥🔥): 

> - `AI Regulation Discussions` (AI 监管讨论)
> - `ML Community Sentiment` (ML 社区情绪)
> - `Free Speech vs. Hate Speech` (言论自由 vs. 仇恨言论)
> - `Debate Reactions` (辩论反应)
> - `Fine-tuning Models with Qlora` (使用 Qlora 微调模型)


- **AI 监管讨论**：针对欧盟 AI 法规等监管影响进行了深入讨论，并引用了 Joe Rogan 的播客，其中 Dr. Epstein 讨论了这些问题。
   - 参与者对社交媒体如何塑造叙事表达了担忧，特别是与来自 Reddit 等平台的信息准确性相关的问题。
- **ML 社区情绪**：成员们注意到社区内存在一种针对美国人的偏见，对某些人如此自由地表达负面情绪感到不适。
   - 成员们分享了关于讨论中适度引导（moderation）必要性的看法，反思了言论自由与维持尊重环境之间的平衡。
- **言论自由 vs. 仇恨言论**：对话包括一场关于言论自由定义和界限的辩论，特别强调了言论自由与仇恨言论之间的区别。
   - 参与者讨论了不受监管的言论及其滋生仇恨言论的潜力，强调了对一定程度社区准则的需求。
- **辩论反应**：对近期政治辩论的反应不一，成员们评论了 Trump 的表现，他们认为那是灾难性的。
   - 讨论转向了更广泛的美国政治主题，一些成员表示希望脱离政治话语。
- **使用 Qlora 微调模型**：有人提出了关于使用 Qlora 微调模型时最佳学习率调度器的问题，偏好倾向于余弦（cosine）调度。
   - 对话还涉及了 **rank** 和 **lora_alpha** 等缩放因子，指出较小的 **rank** 结合较高的 **alpha** 可能会在指令微调（instruction tuning）中产生更好的结果。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/code-of-conduct">Code of Conduct – Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/defog/llama-3-sqlcoder-8b">defog/llama-3-sqlcoder-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bunkalab/mapping_the_OS_community">Mapping the AI OS community - a Hugging Face Space by bunkalab</a>: 未找到描述</li><li><a href="https://tenor.com/view/monkey-laught-monkey-laught-smile-funny-gif-8811182016519780369">Monkey Laught GIF - Monkey Laught Monkey laught - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue">nyu-mll/glue · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/chatdb/natural-sql-7b">chatdb/natural-sql-7b · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/im-in-your-head-you-cant-win-gif-8159346">Im In Your Head You Cant Win GIF - Im In Your Head You Cant Win - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://joerogan.techwatchproject.org/">Dr. Robert Epstein’s 'The Evidence': Unveiling Big Tech's Hidden Influence on Elections and Democracy | TECHWATCHPROJECT.ORG</a>: 探索 Robert Epstein 博士的《证据》(The Evidence)，揭示大型科技公司审查异议和影响选举的力量。了解 Google 的隐藏策略如何威胁民主，以及我们可以采取什么措施来保护...</li><li><a href="https://open.spotify.com/episode/2GRJYz6ZMtVlUfqaqhro5o?si=2d">#2201 - Robert Epstein</a>: 剧集 · The Joe Rogan Experience · Robert Epstein 是一位作家、编辑和心理学研究员。他曾任《今日心理学》(Psychology Today) 的主编，目前担任高级研究员...</li><li><a href="https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified?sql_console=true">airtrain-ai/fineweb-edu-fortified · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue?sql_console=true&sql=SELEC">nyu-mll/glue · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/fka/awesome-chatgpt-prompts?sql_console=true&sql=SELECT+*+FROM+train+LIMIT+10">fka/awesome-chatgpt-prompts · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue?sql_console=true&sql=SELECT+*+FROM+ax+LIMIT+10">nyu-mll/glue · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/pgvector/pgvector">GitHub - pgvector/pgvector: Open-source vector similarity search for Postgres</a>: 适用于 Postgres 的开源向量相似度搜索。通过在 GitHub 上创建账号来为 pgvector/pgvector 的开发做出贡献。</li><li><a href="https://open.spotify.com/episode/2GRJYz6ZMtVlUfqaqhro5o?si=2d5cee494a7f4ffd">#2201 - Robert Epstein</a>: 剧集 · The Joe Rogan Experience · Robert Epstein 是一位作家、编辑和心理学研究员。他曾任《今日心理学》(Psychology Today) 的主编，目前担任高级研究员...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

lunarflu: 太棒了，让我们知道进展如何！ <:hugging_rocket:968127385864134656>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1283578032791097456)** (2 条消息): 

> - `Exa AI 搜索引擎`
> - `游戏中的 Meta Discovery 框架` 


- **Exa AI 搜索引擎增强知识检索**：[Exa AI 搜索引擎](https://exa.ai/) 将你的 AI 连接到庞大的知识库，提供专为 AI 应用量身定制的 **语义和 embedding 搜索** 功能。
   - 它承诺提供 **正是你所要求的** 内容，没有标题党，支持内容抓取和相似度搜索等功能。
- **Meta Discovery 框架辅助游戏平衡决策**：一篇论文讨论了 **Meta Discovery 框架**，该框架利用 Reinforcement Learning 来预测《宝可梦》和《英雄联盟》等竞技游戏中平衡性调整的影响，详见 [arXiv 论文](https://arxiv.org/abs/2409.07340)。
   - 作者断言，该框架可以帮助开发者做出 **更明智的平衡决策**，并在测试中展示了极高的预测准确性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.07340">A Framework for Predicting the Impact of Game Balance Changes through Meta Discovery</a>: 元游戏 (metagame) 是超越游戏规则的知识集合。在像《宝可梦》或《英雄联盟》这样的竞技类团队游戏中，它指的是当前主流的角色集合...</li><li><a href="https://exa.ai/">Exa</a>: Exa API 从网络检索最佳的实时数据，以补充你的 AI
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1283542542784401420)** (5 messages): 

> - `HOPE Agent`
> - `DebateThing.com`
> - `Synthetic Data Generation Packages`
> - `Hugging Face Community Mapping`
> - `Batch Size in Sentence Transformers` 


- **HOPE Agent 简化了 AI 编排 (Orchestration)**：介绍 [HOPE Agent](https://github.com/U-C4N/HOPE-Agent)，它提供 **基于 JSON 的 Agent 管理**、**动态任务分配**和**模块化插件系统**等功能，以简化复杂的 AI 工作流。
   - 它集成了 **LangChain** 和 **Groq API**，有效增强了对多个 AI Agent 的管理。
- **DebateThing.com 生成 AI 驱动的辩论**：[DebateThing.com](https://debatething.com/) 是一个全新的 AI 驱动辩论生成器，支持最多 **4 名参与者**，并具备 **Text-to-Speech** 支持，可进行多轮音频辩论。
   - 利用 **OpenAI** 的 **GPT-4o-Mini** 模型，它能够创建模拟辩论，并根据 **MIT License** [开源](https://github.com/phughesmcr/debatething)。
- **用于合成数据的新 Python 包**：一篇文章讨论了两个新的 Python 包 **ophrase** 和 **oproof**，用于生成合成数据，以及一个能够为句子创建 **9 种变体** 的改写生成器。
   - 这些包还包含一个用于验证 Prompt-Response 对的证明引擎，并在 **基础数学、语法和拼写** 上进行了测试。
- **映射 Hugging Face 社区**：一个新的社区映射项目现已在 [Hugging Face](https://huggingface.co/spaces/bunkalab/mapping_the_OS_community) 上线，旨在扩展社区参与度。
   - 该项目旨在为 AI 领域的开发者和研究人员之间创造协同空间。
- **Sentence Transformers 中 Batch Size 的重要性**：一篇新的 [Medium 文章](https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649) 强调了为 **Sentence Transformers** 正确设置 **Batch Size** 参数的重要性。
   - 作者强调，如果不将 Batch Size 从默认值 **32** 进行调整，可能会导致 VRAM 利用率低下和性能下降。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn">无标题</a>: 未找到描述</li><li><a href="https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649">在使用 Sentence Transformers 嵌入模型时，正确设置 Batch Size 至关重要…</a>: 人们经常忽略 Sentence Transformers 也包含一个“batch”参数，你可以（且应该）根据具体情况进行调整……</li><li><a href="https://huggingface.co/spaces/bunkalab/mapping_the_OS_community">映射 AI 开源社区 - bunkalab 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/U-C4N/HOPE-Agent">GitHub - U-C4N/HOPE-Agent: HOPE (Highly Orchestrated Python Environment) Agent 简化了复杂的 AI 工作流。轻松管理多个 AI Agent 和任务。特性：• 基于 JSON 的配置 • 丰富的 CLI • LangChain &amp; Groq 集成 • 动态任务分配 • 模块化插件。使用 HOPE Agent 优化您的 AI 项目。</a></li><li><a href="https://debatething.com/">DebateThing.com</a>: 使用 AI 就任何话题生成有趣的辩论并免费收听！
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1283845477091246122)** (1 messages): 

> - `Team Building in ML`
> - `Collaborative Projects in AI` 


- **寻找 ML 团队成员**：一位成员表示有兴趣组建一个 ML 项目团队，并说明他们正处于机器学习旅程的初期阶段。
   - 他们邀请具有相似技能水平的人联系并协作，共同创建令人兴奋的模型。
- **呼吁在 ML 领域进行协作学习**：同一位成员强调了在学习过程中团队合作的愿望，提议新手们团结起来共同成长。
   - 他们提到，聚在一起可以增强学习过程，并促进创新模型的开发。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1283567832533176380)** (9 messages🔥): 

> - `NSFW text detection datasets` (NSFW 文本检测数据集)
> - `Transformer architecture and embeddings` (Transformer 架构与 embeddings)
> - `Mean pooling in embeddings` (Embeddings 中的 Mean pooling)
> - `Self-supervised training techniques` (自监督训练技术)
> - `Building models from scratch` (从零开始构建模型)


- **寻找 NSFW 文本检测数据集**：一名成员询问是否存在类似于图像领域 MNIST 或 ImageNet 的标准学术 **NSFW text** 检测数据集，并提到了 CensorChat 和 Reddit 数据集。
   - 对话强调了在该特定领域缺乏现成的基准数据集。
- **理解 Transformer 和 embeddings 至关重要**：一位用户强调了在机器学习项目中熟悉 **Transformer architecture** 和 **embeddings** 概念的重要性。
   - 他们建议将搜索 Transformer 的图解作为初学者的良好起点。
- **Embeddings 的 Mean pooling 问题**：一位用户报告称 **Chroma** 默认不执行 mean pooling，这导致了多维 embeddings 的问题。
   - 另一名成员建议 Chroma 的 `Embedding_functions.SentenceTransformerEmbeddingFunction` 可能会提供该问题的解决方案。
- **Transformer 的自监督训练**：围绕 Transformer 的 **self-supervised training** 展开了讨论，指出如果没有充足的资金，从头开始进行预训练（pretraining from scratch）可能非常昂贵。
   - 成员们强调，在简单的领域训练模型是可行的，但在其他情况下则具有挑战性。
- **不使用高级工具构建模型**：一名成员表示希望在不使用高级工具的情况下从头构建模型，以增强学习效果。
   - 这引发了关于这种方法在模型开发中的可行性和实用性的讨论。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1283523477411860480)** (274 messages🔥🔥): 

> - `Mojo Error Handling` (Mojo 错误处理)
> - `String Conversion in Mojo` (Mojo 中的字符串转换) 


- **理解 Mojo 对 Syscalls 的错误处理**：讨论强调了确保 syscall 接口返回有意义的错误值的重要性，因为这些在很大程度上取决于 C 等语言中的接口契约。
   - 对话强调，设计接口需要了解可能的错误，以确保有效处理 syscall 响应。
- **从 Span[UInt8] 中提取字符串**：一名成员寻求将 `Span[UInt8]` 转换为字符串视图的指导，并被引导至 `StringSlice`。
   - 在遇到错误后，他们澄清需要使用关键字参数 `unsafe_from_utf8` 来正确初始化 `StringSlice`。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/#Adding-Constraints-to-Protocol-Extensions">Documentation</a>: 未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Protocol-Extensions">Documentation</a>: 未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Protocol-Ex">Documentation</a>: 未找到描述</li><li><a href="https://docs.oracle.com/en/database/oracle/oracle-database/19/admin/repairing-corrupted-data.html">Database Administrator’s Guide </a>: 您可以检测并修复数据块损坏。</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/#Adding-Con">Documentation</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=OWfxexSE2aM">408 Protocol Oriented Programming in Swift</a>: 翻译 Yandex 浏览器的视频</li><li><a href="https://docs.python.org/3/tutorial/classes.html#inheritance">9. Classes</a>: 类提供了一种将数据和功能捆绑在一起的方法。创建一个新类会创建一个新类型的对象，从而允许创建该类型的新实例。每个类实例可以具有...</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Classes/extends">extends - JavaScript | MDN</a>: extends 关键字用于类声明或类表达式中，以创建一个作为另一个类子类的类。
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1283502329492144270)** (24 messages🔥): 

> - `ExplicitlyCopyable trait`
> - `MojoDojo open source community`
> - `Recommendation systems in Mojo`
> - `Chaining in Rust-like syntax`
> - `Accessing errno in Mojo` 


- **关于 ExplicitlyCopyable 的 RFC 提案**：一名成员建议发起一个关于让 `ExplicitlyCopyable` trait 要求实现 `copy()` 的 RFC，并指出这一变化可能会产生重大影响。
   - 另一位成员指出，这一变化可能允许在未来对定义进行破坏性较小的更新。
- **MojoDojo：一个开源机会**：一名成员发现 [mojodojo.dev](https://github.com/modularml/mojodojo.dev) 已开源，为社区协作提供了机会。
   - 该平台最初由 Jack Clayton 创建，在 Mojo 仅作为 Web Playground 时作为学习资源。
- **Mojo 中推荐系统的探索**：一位用户询问了 Mojo 或 MAX 中用于构建推荐系统的可用功能。
   - 回复指出，Mojo 和 MAX 仍处于“自行构建”阶段，且正在持续开发中。
- **链式语法偏好**：成员们就使用自由函数 `copy` 还是 `ExplicitlyCopyable::copy(self)` 来实现复制功能进行了辩论，重点在于链式调用的优势。
   - 一位用户表示偏好类似于 Rust 的 `.clone` 语法，以促进函数式编程风格。
- **在 Mojo 中访问 errno**：有人询问了在 Mojo 语言中访问 `errno` 的能力。
   - 讨论仍在进行中，用户正在寻求关于此功能的明确说明。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1283875210851778625)** (3 messages): 

> - `Embedding in MAX`
> - `Vector databases`
> - `Semantic search`
> - `Natural Language Processing`
> - `MAX Engine` 


- **MAX 缺乏内置的 Embedding 支持**：一名成员询问了 MAX 中是否提供 Embedding、向量数据库和相似度搜索功能，以便与 LLM 配合使用。
   - 回复澄清说 **MAX** 并不直接提供这些开箱即用的功能，但可以使用 **chromadb**、**qdrant** 和 **weaviate** 等选项与 **MAX Engine** 配合使用。
- **分享了语义搜索博客文章**：一位贡献成员指向了一篇[博客文章](https://www.modular.com/blog/semantic-search-with-max-engine)，内容涉及利用 **MAX Engine** 进行语义搜索。
   - 文章强调，**语义搜索**超越了关键词匹配，利用先进的 Embedding 模型来理解查询中的上下文和意图。
- **重点介绍了 Amazon 多语言数据集**：该博客文章还介绍了 **Amazon Multilingual Counterfactual Dataset (AMCD)**，这对于理解 NLP 任务中的反事实检测非常有用。
   - 该数据集包含来自 Amazon 评论的标注句子，有助于识别用户反馈中的假设场景。



**提及的链接**：<a href="https://www.modular.com/blog/semantic-search-with-max-engine">Modular: Semantic Search with MAX Engine</a>：在自然语言处理 (NLP) 领域，语义搜索侧重于理解查询背后的上下文和意图，超越了单纯的关键词匹配，以提供更相关和上下文相关的...

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1283506876662812734)** (253 条消息🔥🔥): 

> - `OpenAI O1 Preview`
> - `AI Music Trends` (AI 音乐趋势)
> - `Suno Discord Insights` (Suno Discord 见解)
> - `User Experiences with APIs` (API 用户体验)
> - `Uncovr.ai Development` (Uncovr.ai 开发)


- **对 OpenAI O1 的兴奋**：用户们对 [OpenAI O1](https://openai.com/index/introducing-openai-o1-preview/) 的发布议论纷纷，这是一个强调改进推理和复杂问题解决能力的新模型系列。
   - 有人推测它整合了来自 reflection、Chain of Thought 以及像 Open Interpreter 这样的 Agent 面向框架的概念。
- **对 AI 音乐的怀疑**：成员们对 AI 驱动音乐的可行性表示怀疑，将其贴上“噱头”而非真正艺术形式的标签，并对其长期价值提出质疑。
   - 他们引用了一些例子，认为 AI 音乐缺乏传统音乐所传达的人性化触感和深层含义。
- **Uncovr.ai 的挑战与进展**：Uncovr.ai 的创作者分享了在构建平台过程中面临的挑战，以及为增强用户体验而开发新功能的需求。
   - 对成本和可持续收入模式的担忧十分突出，讨论集中在平衡功能与开支的重要性上。
- **对 AI API 和用户体验的担忧**：用户讨论了对 API 限制和授权问题的挫败感，强调了在排除这些故障时缺乏清晰度。
   - 整个对话强调了管理多个 AI 模型并保持低成本高效运行的复杂性。
- **独特 AI 解决方案的愿景**：成员们鼓励关于创建独特 AI 工具的创新思考，以求在竞争产品中脱颖而出，并专注于用户需求。
   - 对话旨在激励开发者利用他们的创造力和见解，构建能与用户产生真实共鸣的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/stephen-diaz-rich-wealthy-making-it-rain-money-rain-gif-15629367">Stephen Diaz Rich GIF - Stephen Diaz Rich Wealthy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/OpenAI/status/1834278217626317026?s=19">来自 OpenAI (@OpenAI) 的推文</a>：我们正在发布 OpenAI o1 的预览版——这是一个全新的 AI 模型系列，旨在在响应前花费更多时间思考。这些模型可以推理复杂任务并解决更难的问题...</li><li><a href="https://tenor.com/view/let-me-in-eric-andre-wanna-come-in-gif-13730108">Let Me In Eric Andre GIF - Let Me In Eric Andre Wanna Come In - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://uncovr.app/">uncovr</a>：您的 AI 搜索伴侣。以美观的方式寻找有用的答案和信息。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1283510850388361318)** (16 messages🔥): 

> - `Air Raid Sirens History`（防空警报历史）
> - `AI Developments in Gaming`（游戏领域的 AI 发展）
> - `IDE Recommendations`（IDE 推荐）
> - `Dark Shipping Trends`（暗航趋势）
> - `Supercomputer Innovations`（超级计算机创新）


- **探索防空警报的起源**：分享了一个讨论 **air raid sirens** 何时成为标准的链接，详细介绍了其历史意义和演变。读者可以在[这里](https://www.perplexity.ai/search/when-did-air-raid-sirens-becom-5hYiNhmJQz.dq639.OMU6g)探索时间线和技术进步。
   - 提到防空警报突显了它们在应急准备和社区安全中的作用。
- **Roblox 构建 AI 世界模型**：一位成员指出了一段视频，讨论 **Roblox** 如何处于将 AI 集成到游戏中的前沿，并强调了其创新方法。在[这里](https://www.youtube.com/embed/yT6Vw4n6PvI)查看视频。
   - 这一发展使 Roblox 成为将游戏与先进技术融合的领导者。
- **识别顶尖 IDE**：分享了一个链接，重点介绍了面向开发者的 **top 10 IDEs**，揭示了用户偏好和功能。你可以在[这里](https://www.perplexity.ai/search/what-are-the-top-10-ides-for-n-NkpW74i6TCShNE_L_eKCCA)查看 IDE 排名和见解。
   - 选择合适的 IDE 对于优化编码效率和提升用户体验至关重要。
- **暗航（Dark Shipping）的兴起**：引用了一篇文章讨论 **dark shipping** 趋势的出现，探讨了其在物流和安全方面的影响。在[这里](https://www.perplexity.ai/page/the-rise-of-dark-shipping-pXXXVKmnS_WMRYdN9HAuOg)深入阅读文章。
   - 这一趋势引发了关于航运实践中透明度和安全性的重要问题。
- **日本的 Zeta 级超级计算机**：提到 **Japan's Zeta-Class Supercomputer**，突出了计算能力方面的尖端进展，这对各种应用都至关重要。关于该超级计算机的更多细节可以在[这里](https://www.perplexity.ai/search/explain-halliburton-s-involvem-nPwjvuWlRPK9czAODfjxFA)找到。
   - 这些创新标志着研究和工业应用能力的飞跃。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1283777940966084693)** (3 messages): 

> - `Enterprise Pro License Issues`（Enterprise Pro 许可证问题）
> - `Free Credit Balance Concerns`（免费额度余额问题）
> - `Comparison of Perplexity Pro and Sonar Online`（Perplexity Pro 与 Sonar Online 的对比）


- **Enterprise Pro 许可证用户报告问题**：用户表示在注册 **Enterprise Pro license** 后急于开始使用 API，但面临技术问题。
   - 用户对 **credit card details** 的处理以及随后 API 的可用性表示担忧。
- **免费额度余额未显示**：多位用户报告称，尽管设置了支付详情，但其 Pro 版 5 美元的 **free credit bonus** 并未反映在账户中。
   - 一位用户表示，最近的反馈让他们对该流程产生怀疑，并认为其*不够理想*。
- **Perplexity Pro 与 Sonar Online 的对比分析**：讨论了仅靠 **prompting** 是否能弥补 **Perplexity Pro** 与 **Sonar Online** 功能之间的差距。
   - 成员们对性能差异点以及如何通过更好的 prompting 来改进它们感到好奇。


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1283521468574994554)** (213 messages🔥🔥): 

> - `OpenAI o1 Model Release`（OpenAI o1 模型发布）
> - `California AI Regulation SB 1047`（加州 AI 监管法案 SB 1047）
> - `AI Model Benchmarking`（AI 模型基准测试）
> - `Chain of Thought in AI`（AI 中的思维链）
> - `AI Process and Reasoning`（AI 过程与推理）

- **OpenAI o1 模型在推理方面表现出色**：新发布的 [OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/) 模型旨在响应前进行更多思考，并在 MathVista 等基准测试中展现了强劲的结果。
   - 用户注意到它处理复杂任务的能力，并对其潜力表示兴奋，尽管一些人对其在实际应用中的真实表现持谨慎态度。
- **加州 AI 监管法案 SB 1047 引发关注**：围绕加州 SB 1047 AI 安全法案的命运存在各种猜测，由于政治动态（特别是 Pelosi 的参与），预测显示有 66-80% 的概率会被否决。
   - 该法案对数据隐私和 inference compute 的潜在影响正在讨论中，凸显了技术创新与立法行动的交集。
- **编程基准测试的对比表现**：初步基准测试显示，OpenAI 的 o1-mini 模型表现与 gpt-4o 相当，特别是在 [Aider](https://aider.chat/2024/09/12/o1.html) 报告的代码编辑任务中。
   - 用户热衷于进一步探索 o1 模型在各种基准测试中与现有 LLM 的对比，反映了 AI 发展中的竞争格局。
- **探索 AI 中的 Chain of Thought 方法论**：讨论强调了 o1 模型中 Chain of Thought (CoT) 推理的使用，强调了通过结构化推理策略提高 AI 性能的潜力。
   - 专家认为，正确实施 CoT 可能会带来更好的 AI 输出，尽管与其相比简单 prompting 技术的真实效能仍存在疑问。
- **AI 推理和可靠性方面的挑战**：人们对 inference compute 的处理方式提出了担忧，指出仅仅生成更多 Token 可能无法有效地扩展以实现稳定的 AI 行为。
   - 参与者讨论了 AI 模型在实际场景中的局限性，强调随着 AI 工具开发的继续，需要集体改进推理能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cognition.ai/blog/evaluating-coding-agents">Cognition | 对 OpenAI o1 的回顾以及我们如何评估 coding agents</a>：我们是一家构建端到端软件 agents 的应用 AI 实验室。</li><li><a href="https://x.com/cHHillee/status/1834294227494436967">Horace He (@cHHillee) 的推文</a>：很酷的工作（稍后肯定会做更多测试！）。但就目前的结果（以及 alphacode2 的结果）来看，似乎所有的 RL + 更好的模型 + IOI 专项训练仅仅得到了...</li><li><a href="https://polymarket.com/event/will-california-pass-sb-1047-ai-safety-bill/will-california-pass-sb-1047-ai-safety-bill?tid=1725767181654">Polymarket | 加州会通过 SB 1047 AI 安全法案吗？...</a>：Polymarket | 加州的 SB 1047 AI 安全法案目前正在州议会进行辩论。立法者在 8 月 31 日之前必须通过该法案，如果获得批准，州长在 9 月 3 日之前...</li><li><a href="https://www.oneusefulthing.org/p/something-new-on-openais-strawberry">新事物：关于 OpenAI 的 “Strawberry” 和推理</a>：以新方式解决难题</li><li><a href="https://x.com/_jasonwei/status/1834278706522849788?s=61">Jason Wei (@_jasonwei) 的推文</a>：非常激动终于能分享我在 OpenAI 的工作成果！o1 是一个在给出最终答案前会进行思考的模型。用我自己的话来说，这是 AI 领域最大的更新（见下文...）</li><li><a href="https://x.com/lupantech/status/1834301611960926308">Pan Lu (@lupantech) 的推文</a>：🚀 o1 现已由 @OpenAI 发布！它经过训练，可以通过长 Chain of Thought 进行慢思考。它的表现令人印象深刻，可能会攻克科学和数学领域的难题，在 #... 上以 73.2% 的成绩创下新的 SOTA。</li><li><a href="https://fxtwitter.com/fofrAI/status/1834293535069122966">fofr (@fofrAI) 的推文</a>：好的，用纽约时报的 Connection 游戏测试 o1。转录并向其传递了规则和今天的谜题。它找对了 50% 的分组，但在最难的一组中遇到了困难。NBA 推理令人印象深刻。我...</li><li><a href="https://x.com/tianle_cai/status/1834283977613390001?s=46">Tianle Cai (@tianle_cai) 的推文</a>：o1 的 Chain of Thought 包含很多口语表达，如 “Hmm”、“But how?” 等。他们是不是用了讲座录音来训练这个模型...</li><li><a href="https://x.com/ClementDelangue/status/1834283206474191320">clem 🤗 (@ClementDelangue) 的推文</a>：再次强调，AI 系统不是在“思考”，而是在“处理”、“运行预测”……就像 Google 或计算机所做的那样。给人一种技术系统正在...的错觉。</li><li><a href="https://x.com/max_a_schwarzer/status/1834280954443321694">Max Schwarzer (@max_a_schwarzer) 的推文</a>：System Card (https://openai.com/index/openai-o1-system-card/) 很好地展示了 o1 的高光时刻——我最喜欢的是当模型被要求解决一个 CTF 挑战时，意识到目标...</li><li><a href="https://fxtwitter.com/terryyuezhuo/status/1834327808333754631?s=46">Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：对 o1 System Card 的评论：https://cdn.openai.com/o1-system-card.pdf 0. 模型仍有预训练阶段。1. 他们肯定支付了高昂费用来获取高质量数据。2. 他们从...中学到了一些东西。</li><li><a href="https://x.com/rao2z/status/1834314021912359393?s=46">Subbarao Kambhampati (@rao2z) 的推文</a>：..是的，我们正在试用 o1 模型。评价褒贬不一；请保持关注。（此外，任何严肃的评估都受到每周 30 条提示词限制的阻碍。如果 @polynoamial 真想这样做，我确信...）</li><li><a href="https://x.com/paulgauthier/status/1834339747839574392?s=61">Paul Gauthier (@paulgauthier) 的推文</a>：o1-mini 的首次 Benchmark 运行显示，在 aider 的代码编辑 Benchmark 中它与 GPT-4o 基本持平。随着更多 Benchmark 运行完成，本文将持续更新：https://aider.chat/2024/09/12/o1.htm...</li><li><a href="https://x.com/rachelmetz/status/1833960059392413944?s=46">Rachel Metz (@rachelmetz) 的推文</a>：🚨来自我、@EdLudlow、@mhbergen 和 @GillianTan 的独家消息！🚨 OpenAI 融资将使这家初创公司的估值跃升至 1500 亿美元 https://www.bloomberg.com/news/articles/2024-09-11/openai-fundraising-set...</li><li><a href="https://x.com/isidentical/status/1834302726785601616">batuhan taskaya (@isidentical) 的推文</a>：如果有人需要免费访问 o1，可以在这里使用（这是一个临时游乐场，请勿作为 API 使用）：https://fal.ai/models/fal-ai/openai-o1/</li><li><a href="https://manifold.markets/ZviMowshowitz/will-california-bill-sb-1047-become">加州 AI 监管法案 SB 1047 会在本届会议上成为法律吗？</a>：50% 的可能性。来自旧金山的加州参议员 Scott Weiner 提出了该法案 (https://twitter.com/Scott_Wiener/status/1755650108287578585, https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bil...
</li>
</ul>

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1283873062168690730)** (6 messages): 

> - `Matt's API Fix`
> - `Recent Release Excitement`
> - `Benchmark Speculation` 


- **Matt 的修复与 API 托管**：一位用户提到 **IT 部门的 Matt** 需要宣布他“修复了它”，并可能 **托管一个指向 O1 的 API**。
   - 一位成员在分享一条关于 Matt 使用 O1 进行 Benchmark 的相关 [推文](https://x.com/thexeophon/status/1834314098554929217?s=46) 时表示“英雄所见略同”。
- **对近期发布的兴奋**：用户对 **近期发布** 表示兴奋，称其“很有趣”，并期待撰写相关文章。
   - 提到了 **低风险（Low stakes）**，表明对即将到来的关于该发布的讨论采取一种随性的态度。



**提到的链接**：<a href="https://x.com/thexeophon/status/1834314098554929217?s=46">来自 Xeophon (@TheXeophon) 的推文</a>：@terryyuezhuo 哈哈，如果 Matt 的 Benchmark 是用 O1 做的呢？

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1283892925935255564)** (2 messages): 

> - `xeophon`
> - `logan` 


- **xeophon 引起关注**：一位成员提到了 **xeophon**，引发了带有俏皮表情符号 <:3berk:794379348311801876> 的反应。
   - *这类表达突显了社区对新兴话题的参与度。*
- **对 Logan 的赞赏**：另一位成员肯定了 **logan 很棒**，表达了社区内的认可。
   - *这类评论增强了关于 AI 领域中受关注的模型和人物的讨论。*


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1283544784220196917)** (6 messages): 

> - `RLHF in Private Models`
> - `Enterprise Offerings by Scale AI`
> - `Model Behavior Shifting`
> - `Training Reliability`
> - `Domain Expertise Challenges` 


- **了解私有模型的 RLHF**：成员们正试图揭示 **RLHF (Reinforcement Learning from Human Feedback)** 在私有和定制模型中具体是如何运作的，因为 Scale AI 正在探索这一领域。
   - *它似乎专注于将模型行为与人类偏好对齐，* 使格式在训练中更加可靠。
- **Scale AI 的企业级策略缺乏细节**：虽然 **Scale AI** 推广其企业级产品，但成员指出网页缺乏关于这如何使客户受益的实质性细节。
   - 一位成员表示，尽管基础见解有限，但仍预期企业级 AI 在 **材料科学和有机化学** 领域将持续保持相关性。
- **转变模型行为**：讨论强调 RLHF 主要是为了 **转变模型行为** 以符合人类的偏好，从而帮助改进输出。
   - 对格式可靠性的关注使得训练在特定领域（如 **数学相关任务**）中表现得更有效。
- **Scale AI 在领域专业知识方面的挑战**：在 **材料科学** 和 **化学** 等专业领域，成员预计 Scale AI 将面临来自既有领域专家的挑战。
   - 观点认为，在监管较少的领域学习数据处理可能比在极其复杂的临床环境中更容易。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1283506504523055184)** (179 条消息🔥🔥): 

> - `Reflection Model 争议`
> - `Unsloth 与 GPU 兼容性`
> - `KTO 模型对齐`
> - `Solar Pro Preview 模型`
> - `LLM 蒸馏挑战` 


- **Reflection Model 备受质疑**：讨论集中在 Reflection 70B 模型上，有关其 Benchmark 造假的指控以及它在 Hugging Face 上持续可用引发了批评。
   - 参与者表示担心此类做法会破坏 AI 开发的公平性，一些人称该模型仅仅是现有技术的封装（wrapper）。
- **Unsloth 需要 NVIDIA GPU**：Wattana 询问了关于在 AMD GPU 上使用 Unsloth 的问题，得到的澄清是 Unsloth 仅兼容 NVIDIA GPU 进行微调（finetuning）。
   - 讨论强调了 Unsloth 优化的内存使用，但强调了 NVIDIA 硬件的必要性。
- **KTO 模型对齐前景广阔**：Sherlockzoozoo 分享了关于 KTO 作为一种极具前景的模型对齐技术的见解，其性能优于 DPO 等传统方法。
   - KTO 的实验取得了积极结果，但由于专有数据原因，使用它训练的模型仍未公开。
- **Solar Pro Preview 推出**：拥有 220 亿参数的 Solar Pro Preview 模型发布，旨在单张 GPU 上高效使用，并声称性能优于更大规模的模型。
   - 批评者对该模型大胆的宣传及其在实践中的可行性表示担忧，并指出了 AI 社区此前经历过的失望。
- **LLM 蒸馏的挑战**：参与者讨论了将 LLM 蒸馏为更小、侧重推理的模型所面临的复杂性，强调了对准确输出数据的需求。
   - Disgrace6161 认为蒸馏依赖于高质量的示例，而缓慢的推理过程源于与复杂推理任务相关的 Token 成本。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jina.ai/news/reader-lm-small-language-models-for-cleaning-and-converting-html-to-markdown/">Reader-LM: Small Language Models for Cleaning and Converting HTML to Markdown</a>：Reader-LM-0.5B 和 Reader-LM-1.5B 是两个受 Jina Reader 启发的新型小语言模型，旨在将来自开放网络的原始、多噪声 HTML 转换为干净的 Markdown。</li><li><a href="https://tenor.com/view/happy-early-birthday-now-gimmee-gif-15731324811646516558">Happy Early Birthday Now GIF - Happy Early Birthday Now Gimmee - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/DrJimFan/status/1834279865933332752">Jim Fan (@DrJimFan) 的推文</a>：OpenAI Strawberry (o1) 发布了！我们终于看到了推理时扩展（inference-time scaling）范式的普及和生产部署。正如 Sutton 在《苦涩的教训》（The Bitter Lesson）中所说，只有两种技术...</li><li><a href="https://blog.google/technology/ai/google-datagemma-ai-llm/">DataGemma: Using real-world data to address AI hallucinations</a>：介绍 DataGemma，这是首个旨在将 LLM 与来自 Google Data Commons 的广泛现实世界数据连接起来的开放模型。</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py#L48-L57">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，内存减少 80% - unslothai/unsloth</li><li><a href="https://datta0.substack.com/p/ai-unplugged-19-kto-for-model-alignment">AI Unplugged 19: KTO for model alignment, OLMoE, Mamba in the LlaMa, Plan Search</a>：洞察胜过信息</li><li><a href="https://huggingface.co/upstage/solar-pro-preview-instruct/tree/main">upstage/solar-pro-preview-instruct at main</a>：未找到描述</li><li><a href="https://www.upstage.ai/products/solar-pro-preview?utm_source=%08platform&utm_medium=huggingface&utm_campaign=solarpro-preview-launch).">Solar Pro Preview | The most intelligent LLM on a single GPU &mdash; Upstage</a>：Upstage 的 Solar Pro Preview 提供顶级的 AI 智能，在单张 GPU 上实现了领先的 MMLU Pro 和 IFEval 分数。体验可与多 GPU +70B 模型相媲美的性能，且无需基础设施...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1283602407451918386)** (12 条消息🔥): 

> - `AI 相关的 Discord 推荐`
> - `Android 应用 Beta 测试`
> - `使用单个文件运行 LLM` 


- **离开 x.com 寻找更好的 AI 社区**：*一位成员提到他们已经离开了 x.com（称其为“地狱”），并寻求 Discord 推荐*，以便紧跟开源和闭源实验室的最新 AI 动态。
   - 建议包括 **mistral** 和 **cohere** 的 Discord，以及 Reddit 社区如 **ChatGPT** 和 **LocalLLAMA** 版块。
- **为 'Kawaii-Clicker' 应用寻找 Beta 测试人员**：*一位成员宣布开发了一款名为 'Kawaii-Clicker' 的 Android 应用*，正在寻找 Beta 测试人员协助发布，并提到利润将用于资助未来的 AI 模型。
   - 他们请求通过私信发送电子邮件以获取测试权限，强调这是支持他们项目的一种免费方式，同时可以测试一款尚未发布的游戏。
- **关于 Mozilla-Ocho 的 Llamafile 的反馈**：*询问是否有人尝试过 GitHub 上的 Llamafile 项目*，该项目声称可以通过单个文件分发和运行 LLM。
   - 一位成员回应称他们已经尝试过，并证实其**功能与描述相符**。
- **寻找项目合作开发者**：*一位用户正在寻找来自美国或欧洲的开发者*，要求沟通顺畅且可靠，以便在各种项目上进行合作。
   - 他们强调了按时交付和保持诚信的重要性，以便共同创造出色的成果。



**提及的链接**：<a href="https://github.com/mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>：通过单个文件分发和运行 LLM。通过在 GitHub 上创建账户，为 Mozilla-Ocho/llamafile 的开发做出贡献。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1283619175641911337)** (29 条消息🔥): 

> - `使用 Alpaca 微调 Gemma`
> - `使用 Unsloth 处理 Llama 3.1`
> - `使用 vLLM 部署量化模型`
> - `数据集格式问题`
> - `Qlora 的学习率调度器 (Learning Rate Schedulers)` 


- **微调 Gemma 产生意外的 <eos> 输出**：在 **Alpaca cleaned** 数据集上微调 **Gemma** 后，有用户反馈模型只生成 **<eos>** 而不产生任何回复。
   - 另一位成员建议，该问题可能源于**数据集格式**，并鼓励用户遵循文档中的代码。
- **Unsloth 简化了 Llama 3.1 微调**：一位成员确认 **Unsloth** 是在自定义数据集上微调 **Llama 3.1** 等模型的理想选择，并参考了 [文档](https://docs.unsloth.ai/)。
   - 他们详细介绍了使用 Unsloth 的优势，例如针对各种模型的速度和显存效率。
- **在 vLLM 上部署量化模型**：讨论涉及了部署**量化模型**及其与 **vLLM** 的兼容性，一名成员正在寻求有关该流程的指导。
   - 他们被引导至 [vLLM 文档](https://docs.vllm.ai/en/latest/)，以进一步了解显存管理和部署的相关信息。
- **澄清数据集格式混淆**：一位用户对损失函数中的 **labels** 表示困惑，询问代码中为何同时存在 **y** 和 **label**。
   - 成员们询问这是否会影响训练，并分享了链接以澄清特定的代码功能。
- **为微调选择学习率调度器**：一位用户询问在使用 **Qlora** 进行微调时合适的 **lr_scheduler** 建议，并列举了 cosine 和 linear 等多种调度器选项。
   - 回复强调没有标准方法，许多因素都会影响结果，这通常是一个反复试验的过程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：Unsloth 新手？从这里开始！</li><li><a href="https://docs.vllm.ai/en/latest/">欢迎来到 vLLM！ &#8212; vLLM</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py#L205">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/issues/862">SFTTrainer 中生成任务的计算指标 · Issue #862 · huggingface/trl</a>：你好，我想在 SFTTrainer 中包含一个基于自定义生成的 compute_metrics（例如 BLEU）。但是，我遇到了困难，因为：输入到 compute_metrics 的 eval_preds 包含一个 .predicti...</li><li><a href="https://docs.unsloth.ai/basics/saving-models/saving-to-vllm">保存到 VLLM | Unsloth 文档</a>：为 VLLM 将模型保存为 16bit
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1283739586585170012)** (5 messages): 

> - `Sci Scope 新闻通讯`
> - `Unsloth AI 性能`
> - `Fast Forward 优化策略`
> - `OpenAI O1 预览版` 


- **Sci Scope 让 AI 研究更易于理解**：Sci Scope 将关于类似主题的新 [ArXiv papers](https://www.sci-scope.com/) 分组，并提供每周摘要，帮助研究人员在快节奏的 AI 领域保持更新。
   - 他们鼓励用户[订阅其新闻通讯](https://www.sci-scope.com/)，以获取最新 AI 研究的简明概述，从而更轻松地选择阅读材料。
- **单 GPU 限制了 Unsloth 的潜力**：讨论强调，**强大的机器**不会显著增强 Unsloth 的性能，因为它被设计为在单 GPU 上以最佳状态运行。
   - *用户强调硬件的改进效果微乎其微*，建议将重点放在软件优化上。
- **Fast Forward 提升微调效率**：[这篇论文](https://arxiv.org/abs/2409.04206)介绍了 Fast Forward，这是一种参数高效的微调方法，可以通过重复优化器步骤直到损失改进达到平台期来加速训练。
   - 该方法实现了高达 **87% 的 FLOPs 减少**和 **81% 的训练时间减少**，并在各种任务中得到了验证，且未损失性能。
- **OpenAI 发布 O1 预览版**：OpenAI 推出了 [O1 preview](https://openai.com/index/introducing-openai-o1-preview/)，提供了旨在增强 AI 交互的新功能。
   - 显著的更新包括用户友好的界面和改进的功能，旨在简化用户的工作流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.sci-scope.com/">Sci Scope</a>：一份关于 AI 研究的 AI 生成报纸</li><li><a href="https://arxiv.org/abs/2409.04206">Fast Forwarding Low-Rank Training</a>：像低秩自适应 (LoRA) 这样的参数高效微调方法旨在降低微调预训练 Language Models (LMs) 的计算成本。基于这些低秩设置，我们提出...
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1283503713863860296)** (133 messages🔥🔥): 

> - `OpenAI o1 模型发布`
> - `Devin AI 编程 Agent`
> - `推理时间扩展 (Inference time scaling)`
> - `推理能力`
> - `AI 行业反应` 


- **OpenAI 发布 o1：一种新的推理模型**：OpenAI 推出了 o1 模型系列，该系列强调推理能力，并提高了在数学和编程等各个领域的解决问题能力，标志着 AI 技术的重大转变。
   - 据报道，o1 模型在需要复杂推理的任务中优于之前的模型，展示了在安全性和鲁棒性方面的进步。
- **使用 Devin AI 测试 o1**：AI 编程 Agent Devin 已使用 OpenAI 的 o1 模型进行了测试，取得了理想的结果，并突显了推理在软件工程任务中的重要性。
   - 这些评估表明，o1 的泛化推理能力显著增强了专注于编程的 Agent 系统性能。
- **推理时间扩展 (Inference Time Scaling) 见解**：专家们讨论了关于 o1 模型的推理时间扩展，认为它可以与传统的训练扩展相竞争，并提升 LLM 的能力。
   - 这种方法包括衡量隐藏的推理过程，以及它们如何影响像 o1 这样模型的整体功能。
- **AI 社区反应不一**：AI 社区对 o1 模型及其能力的反应各异，一些人对其与早期的 Sonnet/4o 等模型相比的影响表示怀疑。
   - 某些讨论围绕 LLM 对非领域专家的局限性展开，强调了提供面向专家的工具和框架的重要性。
- **未来发展与语音功能**：社区正热切期待进一步的发展，包括仍在计划中的 o1 模型潜在语音功能。
   - 尽管人们对 o1 的认知能力感到兴奋，但用户目前在某些功能方面仍受限，例如语音交互。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/colin_fraser/status/1834334418007457897">来自 Colin Fraser (@colin_fraser) 的推文</a>：这很愚蠢 :(</li><li><a href="https://www.cognition.ai/blog/evaluating-coding-agents">Cognition | OpenAI o1 回顾以及我们如何评估编程 Agent</a>：我们是一家构建端到端软件 Agent 的应用 AI 实验室。</li><li><a href="https://x.com/teortaxestex/status/1834297569545257297?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：令人惊讶：在一些似乎非常需要通用推理的 Agent 任务中，Sonnet/4o 的表现不亚于 o1。我猜通用性也是一种领域专业化，而不是一种涌现能力……</li><li><a href="https://x.com/polynoamial/status/1834280155730043108?s=46">来自 Noam Brown (@polynoamial) 的推文</a>：今天，我很高兴能与大家分享我们在 @OpenAI 努力的成果，即创造出具备真正通用推理能力的 AI 模型：OpenAI 全新的 o1 模型系列！（又名 🍓）让我来解释一下 🧵 1/</li><li><a href="https://x.com/polynoamial/status/1834280155730043108](https://x.com/polynoamial/status/1834280969786065278)">来自 Noam Brown (@polynoamial) 的推文</a>：今天，我很高兴能与大家分享我们在 @OpenAI 努力的成果，即创造出具备真正通用推理能力的 AI 模型：OpenAI 全新的 o1 模型系列！（又名 🍓）让我来解释一下 🧵 1/</li><li><a href="https://x.com/sullyomarr/status/1834282869554118934?s=46">来自 Sully (@SullyOmarr) 的推文</a>：这相当重大 —— o1 像 PhD 学生一样聪明 —— 解决了 83% 的 IMO 数学题，而 gpt4o 仅为 13%</li><li><a href="https://x.com/percyliang/status/1834309959565111673?s=46">来自 Percy Liang (@percyliang) 的推文</a>：HELM MMLU v1.8.0 和 HELM lite（10 个不同场景）v1.8.0 发布了！Writer 的新模型 Palmyra-X-004 在这两个榜单中都进入了前 10 名，这是一个由巨头（OpenAI, Anthropic, ...）主导的竞争极其激烈的领域。</li><li><a href="https://x.com/lilianweng/status/1834346548786069647?s=46">来自 Lilian Weng (@lilianweng) 的推文</a>：🍓 o1 终于发布了 —— 这是我们第一个具备通用推理能力的模型。它不仅在困难的科学任务上取得了令人印象深刻的结果，而且在安全性和鲁棒性方面也有了显著提升……</li><li><a href="https://x.com/hume_ai/status/1833906262351974483?s=46">来自 Hume (@hume_ai) 的推文</a>：隆重推出 Empathic Voice Interface 2 (EVI 2)，我们全新的语音到语音基础模型。EVI 2 将语言和语音合并为一个专门为情商训练的单一模型。你可以……</li><li><a href="https://www.chatprd.ai.">ChatPRD | 产品工作的 AI Copilot</a>：未找到描述</li><li><a href="https://x.com/gregkamradt/status/1834286346938225048?s=46">来自 Greg Kamradt (@GregKamradt) 的推文</a>：这是我用来难倒所有 LLM 的问题：“你回复这条消息的第 4 个词是什么？” o1-preview 第一次尝试就答对了，这个模型确实有些不同。</li><li><a href="https://x.com/drjimfan/status/1834284702494327197?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：这可能是自 2022 年原始 Chinchilla Scaling Law 以来 LLM 研究中最重要的图表。核心见解是两条曲线协同工作，而不是一条。人们一直在预测……的停滞。</li><li><a href="https://x.com/draecomino/status/1833940572706668934">来自 James Wang (@draecomino) 的推文</a>：Nvidia 历史上首次开始将市场份额输给 AI 芯片初创公司。在过去几个月的每一次 AI 会议走廊里，你都能听到这种讨论。</li><li><a href="https://x.com/cognition_labs/status/1834292718174077014">来自 Cognition (@cognition_labs) 的推文</a>：在过去的几周里，我们与 OpenAI 密切合作，利用 Devin 评估了 OpenAI o1 的推理能力。我们发现，这一新系列模型对 Agent 系统来说是一个显著的进步……</li><li><a href="https://x.com/matthewberman/status/1834295485773054312?s=46">来自 MatthewBerman (@MatthewBerman) 的推文</a>：我的天……</li><li><a href="https://x.com/openai/status/1834278217626317026?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 OpenAI (@OpenAI) 的推文</a>：我们正在发布 OpenAI o1 的预览版 —— 这是一个全新的 AI 模型系列，旨在响应前花费更多时间进行思考。这些模型可以推理复杂任务并解决更难的问题……</li><li><a href="https://x.com/martinnebelong/status/1833961448734699989?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Martin Nebelong (@MartinNebelong) 的推文</a>：Midjourney 办公时间：他们正在开发一个 3D 系统，允许你进入 Midjourney 图像。不是多边形，不是 NeRF，也不是 Gaussians。而是一种新的类 NeRF 格式，由……领导的团队开发。</li><li><a href="https://x.com/willdepue/status/1834294935497179633?s=46">来自 will depue (@willdepue) 的推文</a>：关于今天推理模型发布真正意义的一些反思：新范式。我真的希望人们明白这是一个新范式：不要指望会有相同的节奏、时间表或……的动态。</li><li><a href="https://x.com/sainingxie/statu">来自 Saining Xie (@sainingxie) 的推文</a>：...</li>

<li><a href="https://x.com/sainingxie/status/1834300251324256439?s=46">来自 Saining Xie (@sainingxie) 的推文</a>：现在这事儿和重力有关了吗？ 😶</li><li><a href="https://x.com/_jasonwei/status/1834278706522849788?s=46">来自 Jason Wei (@_jasonwei) 的推文</a>：非常激动终于能分享我在 OpenAI 的工作成果！o1 是一个在给出最终答案前会进行思考的模型。用我自己的话来说，这是 AI 领域最大的更新（见...</li><li><a href="https://x.com/fabianstelzer/status/1834300757241102588?s=46">来自 fabian (@fabianstelzer) 的推文</a>：我首选的 LLM 测试是看模型能否正确解释这个笑话：“两头牛站在田野里，一头牛问另一头：‘你对最近流行的疯牛病怎么看？’。另一头...”</li><li><a href="https://x.com/omarsar0/status/1833913149395030222">来自 elvis (@omarsar0) 的推文</a>：开源 LLM 工具。如果你正在寻找有用的开源 LLM 工具，这是一个非常有用的资源。它包含了不同的类别，如教程、AI 工程和应用等...</li><li><a href="https://x.com/OfficialLoganK/status/1834239568070971441">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：刚刚经历了我在 AI 领域的第三个惊叹时刻... 这一次是通过 NotebookLM 的 AI Overview 🤯</li><li><a href="https://x.com/sama/status/1834351981881950234">来自 Sam Altman (@sama) 的推文</a>：@MattPaulsonSD 先对天空中的神奇智能保持几周的感激之情怎么样，然后你很快就能拥有更多玩具了？</li><li><a href="https://x.com/OpenAIDevs/status/1834278699388338427">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：OpenAI o1-preview 和 o1-mini 今天开始向 tier 5 的开发者开放 API。o1-preview 具有强大的推理能力和广泛的世界知识。o1-mini 更快、便宜 80%，并且...</li><li><a href="https://x.com/sama/status/1834276403270857021?s=46">来自 Sam Altman (@sama) 的推文</a>：没耐心了，jimmy</li><li><a href="https://x.com/skeptrune/status/1833954889904652737?s=46">来自 skeptrune (@skeptrune) 的推文</a>：我非常激动地宣布 @trieveai 完成了由 Root Ventures 领投的 350 万美元融资！我和 @cdxker 创立 Trieve 是因为我们觉得构建 AI 应用应该更简单。我们正在寻找...</li><li><a href="https://suno.com/song/eed1b9c6-526c-480a-8bff-64f0908ffcb1">@swyx 的《New Horizons》 | Suno</a>：史诗般的 hiphop rap 歌曲。在 Suno 上聆听并创作你自己的作品。</li><li><a href="https://youtu.be/lH74gNeryhQ?feature=shared">与 Jeff Dean 一起解码 Google Gemini</a>：Professor Hannah Fry 与计算机科学界最传奇的人物之一、Google DeepMind 和 Google Research 的首席科学家 Jeff Dean 进行了交流...</li><li><a href="https://youtu.be/XzK9bx3CSPE?feature=shared">与 Sergey Brin 对话 | All-In Summit 2024</a>：(0:00) David Friedberg 介绍 Sergey Brin (1:41) Sergey 在 Google 负责的工作 (5:45) Google 是否在追求“上帝模型”？ (8:49) 对大规模 AI 的看法...</li><li><a href="https://www.goodmorningamerica.com/culture/story/oprah-winfrey-discusses-experience-ai-honor-113603606">Oprah Winfrey 讨论她使用 AI 的经验：“我们应该尊重它”</a>：Winfrey 在 ABC 的新黄金时段特别节目中深入探讨了这项技术。</li><li><a href="https://abc.com/news/1efd942d-61bb-4519-8a62-c4a8fce50792/category/1138628">观看 9 月 12 日星期四的《AI 与我们的未来：Oprah Winfrey 特别节目》 | ABC 更新</a>：“AI 与我们的未来：Oprah Winfrey 特别节目”将于美国东部时间 9 月 12 日星期四（晚上 8:00-9:03）在 ABC 播出，次日在 Hulu 上线。</li><li><a href="https://techcrunch.com/2024/09/12/openai-unveils-a-model-that-can-fact-check-itself/?guccounter=1&guce_referrer=aHR0cHM6Ly90LmNvLw&guce_referrer_sig=AQAAAHKtgNex_27lwuSUDAnbQ_AtInkh3VS5doDt8R3UwyP6jWH_4OBpu7Z2V-f0INbnQaEDPK_7J4kwxbODXrbvPPYokLISDutc2SHNE30S6OnE6DtkfPHOsUAuw2MdwJKjOmTPiKuP8NiKpai6qiZ5Pnot8qfP7NHFLPxx6HHmKPz-">OpenAI 发布 o1，一个可以自我核查事实的模型 | TechCrunch</a>：ChatGPT 制造商 OpenAI 宣布了一个可以通过对问题进行“推理”来有效核查自身事实的模型。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 今天在 <@&979487661490311320> 举行 o1 见面会 https://x.com/swyx/status/1834300361102029286
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1283506066898030633)** (5 条消息): 

> - `PTQ and Static Quantization`
> - `Dynamic Quantization Discussion`
> - `CUDA MODE Notes`
> - `De-Quantization Error` 


- **PTQ 实现概览**：一位成员分享了他们运行 **Post-Training Quantization (PTQ)** 的经验，包括使用校准数据集并附加各种 observers 来获取统计信息。
   - 他们量化地指出，与动态方法相比，**static quantization**（静态量化）可以产生更好的结果。
- **动态量化的担忧**：围绕 **dynamic quantization**（动态量化）展开了讨论，即量化参数是实时计算而非固定的。
   - 有人担心这可能导致更差的结果，因为 **outliers**（离群值）会影响 **de-quantization error**（反量化误差）。
- **实现问题的评估**：一位成员对自己的实现表示怀疑，并链接到了 [PyTorch 上的讨论](https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396)，该讨论涉及自定义激活量化后精度大幅下降的问题。
   - 他们正在寻求反馈，以确定在他们的方法中是否忽略了任何重要细节。
- **分享 CUDA MODE 资源**：一位成员提供了他们的 **CUDA MODE notes** 链接，这些笔记是根据 **Andreas Kopf** 和 **Mark Saroufim** 主持的阅读小组讲座整理的。
   - 这些笔记可以通过他们的博客 [christianjmills.com](https://christianjmills.com/series/notes/cuda-mode-notes.html) 公开访问。



**提及的链接**：<a href="https://christianjmills.com/series/notes/cuda-mode-notes.html">Christian Mills - CUDA MODE Lecture Notes</a>：我从 Andreas Kopf 和 Mark Saroufim 主持的 CUDA MODE 阅读小组讲座中整理的笔记。

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1283519941445943430)** (6 条消息): 

> - `Utilizing Tensor Cores`
> - `Support for uint4`
> - `Block Pointer Usage` 


- **通过 Triton 轻松使用 Tensor Cores**：在 Triton 中使用 **tl.dot** 并配合正确形状的输入，可以非常直接地利用 **Tensor Cores**，这符合其硬件无关的设计方法。
   - 有关矩阵形状的更多详细信息，请参阅 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-shape)。
- **关于 uint4 支持的疑问**：讨论了 Triton 对 **uint4** 的支持情况，一位成员对其功能表示感兴趣。
   - 然而，另一位成员澄清说，Triton 目前不支持 **uint4**。
- **使用 Cutlass 实现高级功能**：在需要 **uint4** 支持的场景下，应使用 [Cutlass](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp#L1384) 等工具以获得更好的性能。
   - Cutlass 提供了对线性代数非常有帮助的 CUDA 模板，并可能提供 Triton 缺失的功能。
- **对 Block Pointers 的担忧**：关于在 Triton 中使用 block pointers (`make_block_ptr`) 的有效性出现了疑问，特别是注意到相关文档已被移除。
   - 一个 [GitHub issue](https://github.com/triton-lang/triton/issues/2301) 证实了这些担忧，该 issue 表明创建 block pointers 可能会导致比其他方法更慢的性能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp#L1384">cutlass/include/cute/arch/mma_sm80.hpp at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。欢迎在 GitHub 上为 NVIDIA/cutlass 的开发做出贡献。</li><li><a href="https://github.com/triton-lang/triton/issues/2301">&#39;block_pointer&#39; is slower than &#39;pointer block&#39; · Issue #2301 · triton-lang/triton</a>：在特定情况下，创建 block pointer 非常缓慢。pointer: y_size x_size block pointer pointer block 0 32.0 32.0 0.024209 0.025256 1 64.0 64.0 0.025827 0.024142 2 128.0 128.0 0.024124 0.025241...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1283723674280525824)** (1 条消息): 

> - `Torch Inductor Internals`
> - `PyTorch Native Compiler` 


- **对 Torch Inductor 文档的需求**：一位成员正在寻求关于 **Torch Inductor 内部原理**的更详细文档，超出了[此处](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)现有的讨论内容。
   - *是否有更全面的资源*可以深入了解 Torch Inductor 的内部工作机制？
- **关于 PyTorch 原生编译器的讨论**：提到了 **PyTorch 原生编译器**以及在获取必要文档和资源方面面临的挑战。
   - 成员们表示需要对 Torch Inductor 的特性和功能进行更清晰的阐述。



**提到的链接**：<a href="https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747">TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes</a>：PyTorch 团队一直在构建 TorchDynamo，它通过动态 Python 字节码转换帮助解决 PyTorch 的图捕获问题。为了真正让 PyTorch 更快，TorchDynamo 必须……

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1283794892635639951)** (5 条消息): 

> - `HF transformers LLMs`
> - `OpenAI O1 Preview`
> - `Learning to Reason with LLMs`
> - `Iterative Improvements with LLMs` 


- **让 HF Transformers LLMs 完美运行！**：一位成员分享了一个[技巧](https://x.com/mobicham/status/1834235295522254963)，可以在不需要自定义生成器的情况下，让 HF transformers LLMs 与 **torch.compile** 和 **static cache** 配合正常工作。
   - 这种方法简化了设置，可能为 LLM 用户简化开发流程。
- **围绕 OpenAI O1 Preview 的兴奋情绪**：通过链接到 [OpenAI O1 Preview](https://openai.com/index/introducing-openai-o1-preview/)，成员们对 AI 领域的新进展表示热切关注。
   - 该预览展示了几个有前景的特性和增强功能，可能会影响 LLM 应用的发展方向。
- **探索学习使用 LLMs 进行推理**：讨论转向了[学习使用 LLMs 进行推理](https://openai.com/index/learning-to-reason-with-llms/)，强调了 LLM 能力的持续进步。
   - 成员们注意到最近的改进效果好得令人怀疑，引发了关于其真实性和方法论的辩论。
- **迭代改进引起关注**：一位成员评论说，最新的结果看起来比之前的迭代增强有了“过好”的改进。
   - 这导致了对实现这些结果所使用的数据和技术的怀疑，表明可能需要进一步验证。



**提到的链接**：<a href="https://x.com/mobicham/status/1834235295522254963">mobicham (@mobicham) 的推文</a>：让 HF transformers LLMs 与 torch.compile + static cache 配合正常工作且无需实现自定义生成器的简单技巧！

  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1283899850068594758)** (1 条消息): 

> - `Aurora Innovation`
> - `招聘工程师`
> - `商业发布计划`
> - `近期融资轮次`
> - `无人驾驶卡车走廊` 


- **Aurora Innovation 的扩张计划**：Aurora Innovation 是一家上市的 L4 级自动驾驶卡车公司，目标是在 2024 年底前实现无安全员的**商业发布**，其股价在过去六个月内已**翻倍 (2x)**。
   - 该公司正在寻找擅长 **GPU 加速**的 L6 和 L7 级工程师，特别是具备 **CUDA/Triton** 工具专业知识的人才。
- **Aurora 的招聘机会**：Aurora 正在招聘与训练和推理 GPU 加速相关的职位，为优秀工程师提供优厚待遇。
   - 感兴趣的候选人可以查看具体职位，例如 [深度学习加速主任软件工程师 (Staff Software Engineer for Deep Learning Acceleration)](https://aurora.tech/jobs/staff-software-engineer-deep-learning-acceleration-7518608002) 和 [ML 加速器高级主任软件工程师 (Senior Staff Software Engineer for ML Accelerators)](https://aurora.tech/jobs/sr-staff-software-engineer-ml-accelerators-5574800002)。
- **近期融资助力 Aurora 增长**：Aurora 最近筹集了 **4.83 亿美元**资金，超过了 **4.2 亿美元**的目标，这反映了投资者对其长期愿景的强大信心。
   - 此次融资紧随此前的 **8.2 亿美元**资本募集之后，将支持其在 2024 年底实现无人驾驶发布的努力。
- **无人驾驶卡车走廊已建立**：Aurora 已在达拉斯和休斯顿之间成功开通了首条具备商业化条件的**无人驾驶卡车车道**，每周可支持超过 **75 趟商业货运**。
   - 这一战略走廊至关重要，因为德克萨斯州近一半的卡车货运都沿着 **I-45** 路线移动，这使其成为即将发布的理想选择。
- **里程碑与未来愿景**：在最近的分析师日（Analyst Day）上，投资者体验了无人驾驶卡车试乘，展示了 Aurora 的技术进步和合作伙伴生态系统的实力。
   - Aurora 的终端系统旨在**昼夜**运行，强化了其满足商业物流需求的承诺。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aurora.tech/jobs/staff-software-engineer-deep-learning-acceleration-7518608002">Staff Software Engineer - Deep Learning Acceleration </a>: 我们正在招聘技术和业务领导者！加入全球经验最丰富的团队，安全、快速、广泛地将自动驾驶技术推向市场。软件平台软件与服务...</li><li><a href="https://aurora.tech/jobs/sr-staff-software-engineer-ml-accelerators-5574800002">Sr Staff Software Engineer, ML Accelerators</a>: 我们正在招聘技术和业务领导者！加入全球经验最丰富的团队，安全、快速、广泛地将自动驾驶技术推向市场。企业发展与战略伙伴...</li><li><a href="https://techcrunch.com/2024/08/02/self-driving-truck-startup-aurora-innovation-raises-483m-commercial-launch/">自动驾驶卡车初创公司 Aurora Innovation 在商业发布前通过股票发售筹集 4.83 亿美元 | TechCrunch</a>: 自动驾驶技术公司 Aurora Innovation 希望在冲刺无人驾驶目标的过程中筹集数亿美元的额外资金。</li><li><a href="https://ir.aurora.tech/news-events/press-releases/detail/84/aurora-opens-first-commercial-ready-route-for-its-planned">Aurora 为其计划于 2024 年底推出的无人驾驶卡车开通首条具备商业化条件的路线</a>: 随着休斯顿商业化终端的首次亮相，Aurora 可以支持并服务于达拉斯和休斯顿之间的无人驾驶卡车。Aurora…...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1283605261554028597)** (7 messages): 

> - `Rust CUDA Crate`
> - `NCCL Allreduce 文档`
> - `Cublas GEMM 函数问题` 


- **Python Numba CUDA 的 Rust 等效项**：一位成员建议查看 [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)，这是一个类似于 Python `numba.cuda` 的 Rust crate，用于编写和执行 GPU 代码。
   - 该 GitHub 仓库托管了一个致力于在 Rust 中进行快速 GPU 开发的库和工具生态系统。
- **关于 NCCL 归约算法的咨询**：一位成员询问了 NCCL 在不同消息大小和 GPU 设置下用于 allreduce 操作的具体归约算法（reduction algorithms）。
   - 他们希望在单个 GPU 节点内模拟 `NCCL_ALGO=tree` 的累加顺序，以便进行测试。
- **Cublas GEMM 函数故障**：一位用户在尝试计算两个以 `float*` 表示的列向量的矩阵乘积时，在使用 `cublasSgemm` 函数时遇到了问题，并得到了 NaN 结果。
   - 他们后来独立解决了问题，但表示这是使用 C++、CUDA 和 cuBLAS 时常见的初学者挑战。



**提到的链接**：<a href="https://github.com/Rust-GPU/Rust-CUDA">GitHub - Rust-GPU/Rust-CUDA: Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust.</a>：完全使用 Rust 编写和执行快速 GPU 代码的库和工具生态系统。 - Rust-GPU/Rust-CUDA

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1283699764759756800)** (1 messages): 

> - `1 半径模糊实现`
> - `GitHub 贡献` 


- **成功实现 1 半径模糊**：一位成员对成功运行该书第 6 章代码中的 **1 半径模糊（1 radius blur）** 表示自豪。
   - 他们为有兴趣尝试相同实现的初学者分享了代码链接：[GitHub 上的代码](https://github.com/jfischoff/pmpp/blob/main/src/convolution.cu#L222)。
- **为初学者分享资源**：该成员鼓励其他人，特别是初学者，利用书中提供的资源进行实践。
   - 他们引用了其 GitHub 项目的可视化链接，强调了通过共享解决方案进行社区学习的影响：[pmpp/src/convolution.cu](https://opengraph.githubassets.com/38f5cf8aff14f2a71819d73003ce88e2b887efd7d7e3b85efd9f202b6bb73fa5/jfischoff/pmpp)。



**提到的链接**：<a href="https://github.com/jfischoff/pmpp/blob/main/src/convolution.cu#L222">pmpp/src/convolution.cu at main · jfischoff/pmpp</a>：Programming Massively Parallel Processors 练习 - jfischoff/pmpp

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1283504710476890235)** (5 条消息): 

> - `TorchAO 的局限性`
> - `AO 文档的可用性`
> - `CUDA matmul 操作`
> - `量化技术` 


- **TorchAO 存在一些局限性**：几位成员指出 **TorchAO** 与 **compile** 的配合并不理想，目前主要是一个 **CPU 模型**，且其**卷积架构**（**convolutional architecture**）带来了一些挑战。
   - 此外，还有人担心一些意外行为，例如使用 **F.linear** 初始化线性操作时出现的问题。
- **关于 PyTorch Domains 上 AO 文档的查询**：一位成员询问 **AO 文档** 是否会像 **torchtune** 一样在 **pytorch-domains** 上提供。
   - 讨论中针对此查询没有给出明确的答复。
- **对 FP16 输入的 CUDA matmul 操作表示好奇**：一位成员对 **TorchAO** 中是否存在针对 **FP16 输入**和 **int8 权重**的 **CUDA matmul 操作**表示感兴趣，并提到他们在理解技术术语时遇到了挑战。
   - 他们引用了一个内容丰富的 [GitHub issue](https://github.com/pytorch/ao/issues/697)，该 issue 讨论了旨在提高性能的低比特算子（lower-bit kernels）的现状。
- **关于 _weight_int8pack_mm 功能的澄清**：讨论强调可以在 [此链接](https://github.com/pytorch/pytorch/blob/e157ce3ebbb3f30d008c15914e82eb74217562f0/aten/src/ATen/native/native_functions.yaml#L4154) 找到 **_weight_int8pack_mm**，它主要在 **CPU** 和 **MPS** 上运行。
   - 讨论还提供了关于量化技术的进一步见解，并将相关查询引导至 [量化技术仓库](https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization-techniques)。
- **探索高级量化技术**：为了最小化内存占用（memory footprint），建议尝试 **int4_weight_only** 量化，但需要注意的是它仅适用于 **bfloat16**。
   - 对于必须使用 **float16** 的情况，成员建议探索 **fpx_weight_only(3, 2)** 作为实现有效量化的潜在解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/697">[RFC] 我们应该合并或编写哪些低比特 CUDA 算子？ · Issue #697 · pytorch/ao</a>：这是我对现状的理解，以及我认为我们应该做些什么，以使我们的低比特算子在小批量和大批量下都具有更高的性能。我发起这个...</li><li><a href="https://github.com/pytorch/pytorch/blob/e157ce3ebbb3f30d008c15914e82eb74217562f0/aten/src/ATen/native/native_functions.yaml#L4154">pytorch/aten/src/ATen/native/native_functions.yaml (e157ce3ebbb3f30d008c15914e82eb74217562f0) · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization-techniques">ao/torchao/quantization (main 分支) · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/8236a874479a9a9168e584c81dda8707f4c41006/torchao/dtypes/affine_quantized_tensor.py#L1474-L1480">ao/torchao/dtypes/affine_quantized_tensor.py (8236a874479a9a9168e584c81dda8707f4c41006) · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1283744502670037032)** (49 messages🔥): 

> - `O1 及其局限性`
> - `AI 模型对比`
> - `隐藏思维链 (CoT)`
> - `OpenAI 的决策`
> - `AI 研究现状` 


- **O1 揭示了隐藏功能但表现平平**：一场关于 O1 明显隐藏功能的讨论展开，该功能利用了 **CUDA mode**，引发了对其影响的复杂感受，一位成员将其贴上了**毫无实质内容 (massive nothingburger)** 的标签。
   - 尽管有一些令人印象深刻的基准测试结果，但许多人认为在有效的端到端执行方面仍需改进。
- **评估 AI 模型：与时间的赛跑**：有人对 **OpenAI** 在与 **Anthropic** 等竞争对手的对比中似乎处于追赶状态表示担忧，并对**基础模型 (base model)** 的优越性产生怀疑。
   - 许多参与者对当前模型训练策略中的权衡表示怀疑，强调转向更多的 RL 微调方法。
- **关于隐藏 CoT 有效性的辩论**：大家一致认为**思维链 (CoT)** 对未来的进步至关重要，但对其**隐藏实现**表示沮丧，并呼吁更高的透明度。
   - 成员们指出，尽管其有效，但 OpenAI 隐藏某些方面的理由似乎值得商榷，尤其是考虑到最终会发生泄露。
- **OpenAI 的做法受到质疑**：讨论转向对 **OpenAI** 领导层及其选择不公开所有模型提示词的批评，引发了对技术现实与公司决策之间一致性的担忧。
   - 这种透明度的缺乏导致人们认为，由于主要实验室的风险规避立场，该领域的一些重要想法可能仍未得到探索。
- **行业关注下的 AI 研究演变**：参与者注意到**学术界**和**工业界**研究之间的分歧，学术界追求创新但规模较小的优化，而工业界则专注于扩展现有想法。
   - 这种分歧被认为导致了整体进展缓慢，一些人希望未来在经过验证的技术推动下，编程基准测试能有所改进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/itsclivetime/status/1834291198640492860">Clive Chan (@itsclivetime) 的推文</a>: 隐藏功能：o1 具有 cuda mode（顺便说一下，确实有效）</li><li><a href="https://news.ycombinator.com/item?id=41359152">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1283875532013830208)** (1 messages): 

> - `torch.compile 支持`
> - `transformers model.generate`
> - `HFGenerator 弃用` 


- **torch.compile 与 transformers 集成**：对 `torch.compile` 的支持已直接添加到 MobiusML 仓库 [0.2.2 版本](https://github.com/mobiusml/hqq/releases/tag/0.2.2) 的 `model.generate()` 函数中。
   - 这一改进消除了使用 **HFGenerator** 生成模型的需要，使流程更加精简。
- **不再需要 HFGenerator**：随着最近的更新，用户不再需要依赖 **HFGenerator** 来生成模型，因为 `torch.compile` 现在已直接集成。
   - 这一变化简化了以前使用 HFGenerator 执行类似任务的开发者的工作流程。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1283838767169273856)** (2 messages): 

> - `混合输入缩放因子 (Mixed Input Scale Factor)`
> - `Hopper 架构下的 GEMM 性能` 


- **混合输入缩放因子适用于量化张量**：一位成员确认，在混合输入场景中，**缩放因子 (scale factor)** 仅与**量化张量 (quantized tensor)** 相关。
   - 这强调了在混合张量操作中需要考虑如何应用缩放。
- **在 GEMM 中融合反量化会损害性能**：有人指出，在 **Hopper** 架构的 **GEMM** 中融合 A 和 B 张量的**反量化 (dequantization)** 将显著降低性能。
   - 重点强调了这种技术可能对效率产生的负面影响。


  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1283508075168403547)** (15 messages🔥): 

> - `CUDA Hackathon 资源`
> - `Cooley-Tukey FFT 实现`
> - `GH200 计算额度`
> - `自定义 Kernel 与赛道提案` 


- **令人兴奋的大规模计算额度获取！**：组织团队为本次 Hackathon 争取到了 **$300K 的云端额度**，以及一个 **10 节点的 GH200 集群**和一个 **4 节点的 8 H100 集群**的使用权。
   - 参与者将享有节点的 **SSH 访问权限**，并有机会利用 **Modal stack** 进行无服务器扩展。
- **深入研究 FFT 的 Cooley-Tukey 算法**：一位参与者计划使用 CUDA C++ 实现用于 FFT 的 **Radix-2 Cooley-Tukey** 算法，相关资源可在 [FFT 详情](https://brianmcfee.net/dstbook-site/content/ch08-fft/FFT.html) 查阅。
   - 该项目侧重于学习而非高级实现，欢迎他人协作。
- **分享 GH200 架构资源**：针对对 **GH200 架构**感兴趣的人员，分享了多个链接，包括 [NVIDIA 基准测试指南](https://docs.nvidia.com/gh200-superchip-benchmark-guide.pdf)。
   - 一个关于 [在 GH200 上训练 LLM 的 GitHub 项目](https://github.com/abacusai/gh200-llm) 也展示了其潜在应用。
- **协作编写自定义 Kernel 文档**：有人建议创建一个 **自定义 Kernel 的 Google Doc** 以整合贡献，因为目前的文档主要集中在稀疏化（sparsity）和量化（quantization）等特定任务上。
   - 讨论继续围绕各个赛道的提案展开，包括一个用于多 GPU 的 **麦克斯韦方程组（Maxwell's equations）模拟器**。
- **需要更多赛道资源**：成员们询问了除多 GPU 以外的其他赛道的项目列表，寻求明确各赛道可用的资源。
   - 提供了一份包含 **自定义 Kernel** 和 **LLM 推理**的项目摘要，以及相关文档的链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/">量化与稀疏化项目</a>：IRL 高性能自定义 Kernel 的量化与稀疏化项目：1. 开发 A16W3（混合 fp16 x 3-bit）融合 Matmul Kernel：为什么？目前还没有可用的 3-bit 线性 Kernel...</li><li><a href="https://docs.google.com/document/d/1OxWw9aHeoUBFDOClcMr9UrPW8qmpdR5pPOcwH4jEhms/edit#heading=h.c3hqbft26ocn">多 GPU 环节的 Hackathon 项目提案：麦克斯韦方程组模拟器</a>：简介：作为多 GPU Hackathon 环节的一个项目，我建议实现麦克斯韦方程组模拟器。麦克斯韦方程组模拟电磁波的传播。与替代方案相比...</li><li><a href="https://github.com/abacusai/gh200-llm">GitHub - abacusai/gh200-llm：在 NVIDIA 新型 GH200 机器上进行 LLM 训练和推理的软件包及说明</a>：在 NVIDIA 新型 GH200 机器上进行 LLM 训练和推理的软件包及说明 - abacusai/gh200-llm
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1283638163046404177)** (4 messages): 

> - `GEMM FP8 实现`
> - `SplitK GEMM 文档`
> - `测试用例缩放挑战` 


- **GEMM FP8 采用 E4M3**：最近提交的一个 [Pull Request](https://github.com/linkedin/Liger-Kernel/pull/185) 实现了采用 **E4M3 表示**的 FP8 GEMM，解决了 issue #65。
   - 该实现涉及对不同尺寸的正方形矩阵（包括 **64, 256, 512, 1024 和 2048**）以及非正方形矩阵进行测试。
- **添加了 SplitK GEMM 文档**：添加了详细的 docstring 以澄清何时使用 **SplitK GEMM**，并概述了 **SplitK FP8 GEMM** 的预期结构。
   - 此举旨在提高开发者对这些功能的理解和使用。
- **处理变更意见**：贡献者已解决针对 GEMM 实现中**近期变更**的一些评论。
   - 这表明了在完善代码和有效处理反馈方面的协作努力。
- **测试用例中的缩放问题**：面临的一个挑战是在不直接转换为 FP8 的情况下，如何确定**测试用例的缩放（scaling）**。
   - 这仍然是确保所有测试用例成功通过的关注重点。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/pull/185">AndreSlavescu 提交的 gemm fp8 e4m3 · Pull Request #185 · linkedin/Liger-Kernel</a>：摘要：实现了采用 E4M3 表示的 FP8 GEMM。Issue #65。已完成测试：测试了不同尺寸的正方形矩阵 (64, 256, 512, 1024, 2048) + 不同尺寸的非正方形矩阵 ...

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1283507787149737984)** (93 messages🔥🔥): 

> - `Reflection LLM`
> - `AI Photo Generation Services`
> - `Flux Model Performance`
> - `Lora Training Issues`
> - `Dynamic Thresholding in Models` 


- **Reflection LLM 性能担忧**：Reflection LLM 被宣传为一款能够“思考”和“反思”的卓越模型，其 Benchmark 表现优于 **GPT-4o** 和 **Claude Sonnet 3.5**，但因其实际性能差异（尤其是开源版本）而受到批评。
   - 有人担心其 API 似乎在模仿 **Claude**，从而对其原创性和有效性产生了怀疑。
- **关于 AI 照片生成服务的讨论**：有人询问关于第三方 AI 照片生成服务的情况，并探讨了用于生成逼真且多样化图像的最佳付费选项。
   - 有人提出了关于免费替代方案的反对意见，特别引用了 [Easy Diffusion](https://easydiffusion.github.io/) 作为具有竞争力的选择。
- **Flux 模型性能优化**：用户正在分享 **Flux 模型** 内存使用的经验，强调了在不同 RAM 限制下运行测试时显著的性能提升，并实现了不错的生成速度。
   - 有关于内存管理灵活性和低 VRAM 优化的推测，特别是将其与 **SDXL** 等竞争对手进行比较时。
- **Lora 训练故障排除**：个人成员在 Lora 训练过程中遇到了挑战，正在寻求关于在有限 VRAM 的设备上运行的更好配置和训练方法的建议。
   - 建议包括训练环境中的工作流优化资源，并提到了 **Kohya** 等特定训练器。
- **模型中的动态对比度调整**：一位用户正在探索通过特定步骤和 CFG 设置来降低其 Lightning 模型对比度的方法，并考虑实施 Dynamic Thresholding（动态阈值）解决方案。
   - 寻求关于在增加 CFG 值时平衡参数的建议，表明需要对输出质量进行自适应调整。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/zyzz-flex-motivational-stare-anytime-fitness-gif-23227512">Zyzz Flex GIF - Zyzz Flex Motivational - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://easydiffusion.github.io/">Easy Diffusion v3</a>：通过安装 Stable Diffusion，在电脑上创建精美图像的简单一键式方法。无需依赖项或技术知识。</li><li><a href="https://github.com/lllyasviel">lllyasviel - Overview</a>：Lvmin Zhang (Lyumin Zhang)。lllyasviel 拥有 48 个代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/bmaltais/kohya_ss/tree/sd3-flux.1">GitHub - bmaltais/kohya_ss at sd3-flux.1</a>：通过创建账号为 bmaltais/kohya_ss 的开发做出贡献。</li><li><a href="https://github.com/kohya-ss/sd-scripts/tree/sd3?tab=readme-ov-file#flux1-lora-training">GitHub - kohya-ss/sd-scripts at sd3</a>：通过创建账号为 kohya-ss/sd-scripts 的开发做出贡献。</li><li><a href="https://github.com/jhc13/taggui">GitHub - jhc13/taggui: Tag manager and captioner for image datasets</a>：用于图像数据集的标签管理器和打标器。通过创建账号为 jhc13/taggui 的开发做出贡献。</li><li><a href="https://www.cosmic365.ai/">Home | COSMIC365</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1283507306155212922)** (57 messages🔥🔥): 

> - `LM Studio RAG Implementation`
> - `Ternary Models Performance`
> - `Stable Diffusion Integration`
> - `Community Meet-up in London`
> - `OpenAI O1 Access` 


- **LM Studio 中实现 RAG 的指南**：要在 LM Studio 中实现 RAG 流水线，用户应下载 0.3.2 版本并按照成员描述的方式将 PDF 上传到聊天中。
   - 另一位成员指出他们遇到了“未找到与用户查询相关的引用”错误，建议提出具体问题而非泛泛的要求。
- **三元模型 (Ternary Models) 的问题**：成员们讨论了加载三元模型的挑战，有人报告在尝试访问特定模型文件时出错。
   - 建议尝试较旧的量化类型，因为最新的 “TQ” 类型是实验性的，可能无法在所有设备上运行。
- **LM Studio 不支持 Stable Diffusion**：一位用户询问将 Stable Diffusion 集成到 LM Studio 的事宜，另一位成员确认这是不可能的。
   - 这一讨论突显了仅将 LM Studio 用于本地模型的局限性。
- **伦敦社区见面会**：邀请用户明天晚上参加在伦敦举行的见面会，讨论 Prompt Engineering 并分享经验。
   - 鼓励参加者在现场寻找使用笔记本电脑的资深社区成员。
- **OpenAI O1 访问权限的推出**：成员们正在分享他们获得 OpenAI O1 预览版访问权限的更新，并指出可用性是分批推出的。
   - 最近的消息表明，一些成员最近获得了访问权限，而其他人仍在等待。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/optimizing/production_rag/">Building Performant RAG Applications for Production - LlamaIndex</a>: 未找到描述</li><li><a href="https://maps.app.goo.gl/1bHCRW5DP79fKapUA">  Google Maps  </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1283554014017097739)** (35 messages🔥): 

> - `4090D GPUs`
> - `Power requirements for GPUs`
> - `Mixing GPU vendors`
> - `Model precision settings`
> - `ROCm support for AMD GPUs` 


- **对双 4090D 配置的兴趣**：一位成员对使用两块各带 **96GB RAM** 的 **4090D GPU** 表示兴奋，但指出需要小型发电机来支持其 **600W** 功耗的挑战。
   - *“我需要一个小发电机来运行我的电脑”* 是对高功耗的幽默承认。
- **关于混合 GPU 厂商立场的疑问**：在讨论 GPU 厂商兼容性时，一位成员了解到在 **LM Studio** 或类似工具中混合使用 **Nvidia 和 AMD** GPU 是不可行的，从而引发了关于使用同一厂商多块 GPU 的进一步查询。
   - 成员们一致认为，在同一厂商的 GPU（如 **AMD 或 Nvidia**）内进行混合应该可以正常工作，但建议对跨代 **ROCm 支持** 保持谨慎。
- **关于模型精度和性能的辩论**：一位成员寻求关于使用 **FP8 与 FP16** 对模型性能影响的澄清，指出 **FP8** 表现相似且似乎因效率更高而更受青睐。
   - 为了支持这些说法，另一位成员分享了来自 **Llama 3.1 博客** 的最新见解，坚持精度设置应为 **BF16 和 FP8**，而非之前的迭代。
- **MI GPU 的性能影响**：通过 **MI60 vs MI100** 的对比，成员们讨论了对推理速度的潜在影响，指出 **Context Window** 分配可能会影响模型性能，尤其是在 Offloading 场景下。
   - *“模型越大，性能越差”* 总结了对模型大小与处理能力之间关系的担忧。
- **对 MI100 ROCm 支持的担忧**：讨论了 **AMD MI100** GPU 的性能和价格，但成员们对其自 2020 年发布以来的 **ROCm 支持** 寿命表示担忧。
   - 一位成员强调了其 **32GB HBM2** 显存，但因 *可能缺乏未来更新* 而降低了热情。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/u3JI9px">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://www.techpowerup.com/gpu-specs/radeon-instinct-mi100.c3496">AMD Radeon Instinct MI100 Specs</a>: AMD Arcturus, 1502 MHz, 7680 Cores, 480 TMUs, 64 ROPs, 32768 MB HBM2, 1200 MHz, 4096 bit</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/.">无标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1283504301402099783)** (25 条消息🔥): 

> - `Rich Python 库`
> - `Open Interpreter 技能`
> - `Claude 的 Prompt Caching`
> - `探索 OI 的硬件可能性`
> - `Open Interpreter 中的 Token 使用` 


- **使用 Rich 实现美观的终端输出**：[Rich](https://github.com/Textualize/rich) 是一个 Python 库，支持在终端中实现富文本和美观的格式化，提升视觉效果。
   - 成员们讨论了使用终端转义序列（escape sequences）来实现颜色和动画的替代技术，强调了终端操作的多样性。
- **Open Interpreter 技能的挑战**：一位成员指出 Open Interpreter 的技能在会话结束后无法被记住的问题，特别是关于发送 Slack 消息的技能。
   - 另一位成员建议发布一个帖子以进一步调查此问题，引发了社区的回应。
- **在 Claude 中实现 Prompt Caching**：分享了在 Claude 中实现 Prompt Caching 的指南，强调了 Prompt 的正确结构化以及必要的 API 修改。
   - 据指出，对于较长的 Prompt，这些更改可以将成本降低高达 **90%**，并将延迟降低 **85%**。
- **探索 OI 的构建选项**：在 01 项目停止后，成员们讨论了探索不同形态和硬件配置的可能性。
   - 一位成员确认找到了 builders 标签页，这是进行此类探索的良好资源。
- **关于 Open Interpreter 中 Token 使用的担忧**：一位成员对 Open Interpreter 的 Token 使用情况表示担忧，质疑仅 **6** 次请求就消耗 **10,000 tokens** 的效率。
   - 此外，还有关于集成 Webhooks 和 API 访问以实现 Open Interpreter 及其 GPTs 自动化的咨询。



**提到的链接**：<a href="https://github.com/Textualize/rich">GitHub - Textualize/rich: Rich 是一个用于在终端中实现富文本和美观格式化的 Python 库。</a>：Rich 是一个用于在终端中实现富文本和美观格式化的 Python 库。 - Textualize/rich

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1283515461429891092)** (36 条消息🔥): 

> - `设置 LiveKit Server`
> - `OpenAI O1 发布`
> - `iPhone App 安装`
> - `Python 证书更新`
> - `Windows 安装挑战` 


- **推荐使用 Linux 设置 LiveKit Server**：成员们压倒性地建议 **Linux** 是设置 **LiveKit Server** 的最佳操作系统，其中一位表示：*“不要用 Windows，到目前为止只遇到过问题。”*
   - 针对这一观点，另一位成员对该建议表示宽慰，称：*“很有用的信息！！谢谢。这能省下我好几个小时的时间。”*
- **OpenAI O1 以 Strawberry 项目命名**：OpenAI 社区幽默地注意到 **OpenAI** 将其新项目命名为 **O1**，并表示：*“Open AI 简直偷了你们的项目名称，真是荣幸。”*
   - 成员们讨论了这一命名对他们自己工作的影响，并带着怀疑的态度引用了一篇详述发布的博客文章。
- **iPhone App 设置指导**：一位用户寻求安装新发布的 **iPhone App** 的帮助，促使成员们分享了包括**设置指南**和必要命令在内的各种资源。
   - 此外还提供了使用命令启动带有 LiveKit 的 App 并获取二维码的说明。
- **必要的 Python 证书更新**：一位用户强调需要更新他们的 **Python 证书**，并指出了要使用的具体命令，引发了关于安装问题的讨论。
   - 另一位成员说明了他们更新这些证书以确保程序顺利运行的过程，并提供了一个详细的命令行解决方案。
- **Windows 用户面临安装挑战**：一位用户分享了在 **Windows** 上安装的挫败感，特别是他们发现必须通过替换代码中的路径引用来解决问题。
   - 他们提到了过渡到其 AI 机器的复杂性，但对最终成功设置的支持功能和配置表示满意。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://01.openinterpreter.com/setup/introduction">未找到标题</a>: 未找到描述</li><li><a href="https://01.openinterpreter.com/client/android-ios">Android &amp; iOS - 01</a>: 未找到描述</li><li><a href="https://01.openinterpreter.com/setup/installation">Installation - 01</a>: 未找到描述</li><li><a href="https://tenor.com/view/copy-cat-gif-26719674">Copy Cat GIF - Copy Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://01.openinterpreter.com/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/01-app">GitHub - OpenInterpreter/01-app: 用于电脑控制的 AI 助手。</a>: 用于电脑控制的 AI 助手。通过在 GitHub 上创建账号为 OpenInterpreter/01-app 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1283837167986016287)** (5 messages): 

> - `OpenAI o1 Models`
> - `Access to o1`
> - `Cursor and o1-mini`
> - `Quality of outputs` 


- **OpenAI o1 模型预览版发布**：OpenAI 宣布发布 **o1** 预览版，这是一个全新的 AI 模型系列，旨在响应前进行更有效的推理，应用于 **科学**、**编程** 和 **数学** 领域。更多详情请参阅其 [公告](https://openai.com/index/introducing-openai-o1-preview/)。
   - 根据其帖子，这些模型旨在处理比前代更复杂的任务，承诺增强解决问题的能力。
- **社区对 o1 访问权限的关注**：用户对 **o1** 的可用性越来越好奇，几位成员表达了对是否有人已经获得访问权限的兴趣。一位用户指出，*
   - 对早期访问的期待显而易见，表明了尽快探索这些新能力的强烈愿望。
- **等待 Cursor 和 o1-mini 发布**：成员们渴望 **Cursor** 搭载 **o1-mini** 发布，对它的能力表示兴奋和期待。一位用户分享了一个调皮的表情符号，暗示了他们对即将发布的版本的兴趣。
- **关于输出质量的讨论**：一位成员分享反馈称 **o1** 的输出具有 *高质量*，这可能正面反映了新模型的性能。该评论表明早期采用者已开始测试 o1 的能力。
- **消息限制的策略性使用**：一位用户评论说，由于每周 30 条消息的限制，需要采取 **策略性** 的方式，这表明他们在测试新模型的响应时采取了谨慎的态度。这反映了在平台约束下优化使用率的日益增长的兴趣。



**提到的链接**：<a href="https://x.com/openai/status/1834278217626317026?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 OpenAI (@OpenAI) 的推文</a>：我们正在发布 OpenAI o1 的预览版——这是一个全新的 AI 模型系列，旨在在响应前花更多时间思考。这些模型可以推理复杂的任务并解决更难的问题...

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1283746611960483962)** (46 messages🔥): 

> - `Parakeet Project`
> - `GPU Hardware Survey`
> - `Upcoming Developer Office Hours`
> - `Finetuning Llama2 for American Bar Exam` 


- **Parakeet 的快速训练**：**Parakeet 项目**在 A6000 系列模型上训练，在 **3080 Ti** 上运行 **10 小时**，实现了可理解的输出，引发了对 AdEMAMix 优化器性能的猜想。
   - 一位成员评论道，*'这可能是 Parakeet 能够在不到 20 小时内完成 4 层训练的原因。'*
- **硬件调查：GPU 持有情况**：成员们分享了他们的 GPU 配置，其中一位成员报告拥有 **7 个 GPU**，包括 **3 个 RTX 3070** 和 **2 个 RTX 4090**。
   - 对于像 **3070** 这样的模型是否存在，存在幽默的怀疑，引发了关于 GPU 命名惯例的玩笑。
- **关于训练数据质量的讨论**：*'数据量不那么重要——重要的是质量，'* 一位成员在回应关于在 **Phi3.5** 上使用 **26k 行**数据进行 JSON 到 YAML 转换用例的训练对话时表示。
   - 另一位成员强调了干净数据的重要性，指出在模型训练中质量比数量更重要。
- **开发者答疑时间 (Office Hours) 计划**：随着开发者答疑时间的临近，团队正在考虑在 **Zoom** 上举行，以确保缺席的成员能更好地获取录像。
   - 成员们表达了直接在服务器上互动的偏好，赞赏实时互动时的极佳氛围。
- **Llama2 微调见解**：一位新成员分享了他们为美国律师考试 **微调 Llama2** 的工作，并正在探索 DPO，寻求关于其研究广度与深度的反馈。
   - 另一位成员鼓励尝试 **Cohere 的 Command R+**，并提供了相关文档和资源的链接以供进一步探索。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com">Cohere Documentation — Cohere</a>：未找到描述</li><li><a href="https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch">GitHub - nanowell/AdEMAMix-Optimizer-Pytorch: The AdEMAMix Optimizer: Better, Faster, Older.</a>：AdEMAMix 优化器：更好、更快、更老。通过在 GitHub 上创建账户，为 nanowell/AdEMAMix-Optimizer-Pytorch 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1283805167317418256)** (1 条消息): 

> - `Best practices for guardrails in RAG applications`
> - `Tools for evaluating RAG performance` 


- **探索 RAG 应用的 Guardrails**：一位用户寻求关于在 **RAG 应用**中设计和实现 **Guardrails 的最佳实践**，以确保其有效性。
   - 讨论强调了根据特定应用场景定制解决方案的必要性。
- **常用的 RAG 性能评估工具**：该用户询问了用于**评估 RAG 应用性能**的**最常用工具或框架**。
   - 此查询旨在了解常用的评估指标和方法论。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1283520576853512222)** (1 条消息): 

> - `Polymorphic objects`
> - `Structured JSON in Cohere`
> - `Cohere schema limitations` 


- **多态支持方面的困扰**：一名成员询问关于在 Cohere 的结构化 JSON 中支持 **Polymorphic objects** 的问题，并指出 **Cohere** 不支持 `anyOf`。
   - 他们尝试使用 `type: [null, object]`，但也被拒绝了。
- **探索 JSON Schema 方法**：分享了两种 JSON Schema 方法：一种是为 `animal` 对象使用 `anyOf`，另一种是为 **cat** 和 **dog** 属性类型使用 `null`。
   - 第一种方法为两种类型的动物定义了必需属性，而第二种方法为每种动物类型定义了单独的属性，但两者都遭到了拒绝。
- **JSON Schema 验证无法正常工作**：用户的 **Schemas** 旨在为 **cat** 和 **dog** 定义单独的结构，重点关注它们各自的属性和类型。
   - 然而，由于平台限制，这两个 Schemas 最终都被 **Cohere** 平台拒绝。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1283652282239549464)** (3 条消息): 

> - `Personal AI Executive Assistant`
> - `Scheduling Automation`
> - `AI General Purpose Agent`
> - `Rate Limit Concerns`
> - `Open Source Codebase` 


- **个人 AI 行政助理成功实现日程管理**：一名成员分享了他们构建**个人 AI 行政助理**的经验，该助理使用 [calendar agent cookbook](https://link.to.cookbook) 有效地管理日程。他们成功将其与 Google Calendar 集成，通过语音输入添加/编辑/删除事件。
- **模型熟练处理非结构化数据**：该成员强调，他们的模型可以准确地从课程大纲中解析**考试日期和项目截止日期**等非结构化文本，并将其添加到正确的日期和时间。
   - 这一功能极大地增强了他们学习过程中的组织和项目管理能力。
- **用于文档生成的通用 Agent**：他们提到还有一个单独的**通用 Agent**，利用 Web 搜索工具回答提示词，并帮助创建课程项目的提纲文档。
   - 这为他们的个人助理功能增加了多样性。
- **Rate Limit 问题阻碍进一步使用**：由于达到了 **Trial Keys** 的 **Rate Limit**，该成员询问是否有必要购买 Production Key 以继续使用其 AI 助理。
   - 在他们考虑下一步行动时，这构成了项目的一个关键转折点。
- **开源代码库已发布至 GitHub**：该成员确认他们的项目是开源的，代码库可在 [GitHub](https://github.com/mcruz90/ai-need-help) 上获取。
   - 这邀请了对个人 AI 开发感兴趣的社区成员进行协作和贡献。



**提到的链接**：<a href="https://github.com/mcruz90/ai-need-help">GitHub - mcruz90/ai-need-help</a>：通过在 GitHub 上创建账号来为 mcruz90/ai-need-help 的开发做出贡献。

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1283592470218936383)** (12 条消息🔥): 

> - `参与正在进行的项目`
> - `博士到博士后的过渡`
> - `虚拟环境中的 sqlitedict 错误` 


- **项目贡献指南**：一位成员建议在任何项目中通过提交 issue 来进行贡献，随后再提交 PR。
   - 这种方法被确认为最简单的贡献方式。
- **博士到博士后的过渡**：一位成员分享了他们在德国完成博士学位并开始专注于 Safety 和 Multi-agent systems 的博士后研究的兴奋之情。
   - 他们还提到了下国际象棋和打乒乓球等个人爱好。
- **遇到 Module Not Found 错误**：一位成员遇到了即使在虚拟环境中，已安装的 **sqlitedict** 模块仍无法找到的问题。
   - 另一位成员建议在相应的频道中寻求潜在的解决方案。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1283504907135090820)** (18 条消息🔥): 

> - `RWKV-7`
> - `Pixtral 12B 结果`
> - `Transfusion PyTorch 实现`
> - `OpenAI 的 o1 模型系列`
> - `极端思维链 (CoT)` 


- **RWKV-7 在极端 CoT 下表现强劲**：一个仅有 **2.9M** 参数的微型 #RWKV 能够通过极端思维链 (CoT) 解决复杂的数学任务，展示了该模型在无需 KV cache 情况下的高效性。
   - *该技巧涉及生成大量带有反向数字的数据*，以有效地训练模型。
- **Pixtral 12B 对比 Qwen 2 7B VL 的结果不尽如人意**：有说法称 **MistralAI** 会议上的演示具有误导性，图中显示 Pixtral 12B 的表现优于比它小 40% 的 **Qwen 2 7B VL**，但实际情况并非如此。
   - 成员们对数据的完整性表示怀疑，指出他们在自己的对比中也得到了类似的结果，这表明可能并非恶意，而是存在错误。
- **Transfusion 在 PyTorch 中的发布**：**Transfusion** 项目旨在通过单一的多模态模型预测下一个 token 并扩散图像，并采用了一种将扩散替换为 flow matching 的升级方法。
   - 该项目将保持其本质，同时尝试将能力扩展到多种模态，相关内容已在 [GitHub repo](https://github.com/lucidrains/transfusion-pytorch) 中分享。
- **OpenAI 发布新的 o1 模型系列预告**：OpenAI 预告了一个名为 **o1 系列** 的新模型系列，重点关注通用推理能力。
   - 围绕该模型谱系潜在功能的兴奋感正在增长，但具体细节仍然很少。
- **关于模型改进和功能的讨论**：讨论涉及增强模型执行诸如分叉（forking）和合并（joining）状态等任务以实现搜索能力，特别是在 **RNN 风格模型** 的背景下。
   - 成员们辩论了*循环神经网络*在这些任务中的有效性，强调了模型设计中创意解决方案的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.sci-scope.com/">Sci Scope</a>：一份关于 AI 研究的 AI 生成报纸</li><li><a href="https://x.com/polynoamial/status/1834280155730043108">Noam Brown (@polynoamial) 的推文</a>：今天，我很高兴能与大家分享我们在 @OpenAI 努力创造具有真正通用推理能力的 AI 模型的成果：OpenAI 全新的 o1 模型系列！（又名 🍓）让我来解释一下 🧵 1/</li><li><a href="https://x.com/BlinkDL_AI/status/1834300605973889111">BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV 是极端 CoT 的最佳选择🙂无需 KV cache。恒定的状态大小。恒定的 VRAM。恒定的速度。引用 BlinkDL (@BlinkDL_AI) 一个仅有 2.9M (!) 参数的微型 #RWKV 可以解决 18239.715*9.728263 或 4....</li><li><a href="https://x.com/BlinkDL_AI/status/1833863117480280528">BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV-7 "Goose" 预览，具有动态状态演变（使用结构化矩阵）🪿 在修复了一个隐藏的 bug 后，现在的 loss 曲线看起来具有可扩展性😀</li><li><a href="https://x.com/_philschmid/status/1833954941624615151">Philipp Schmid (@_philschmid) 的推文</a>：有人没说实话！@swyx 分享了来自 @MistralAI 内部受邀会议的图片，展示了 Pixtral 12B 的结果。与其他开源模型相比，包括 @Alibaba_Qwen 2 ...</li><li><a href="https://github.com/lucidrains/transfusion-pytorch">GitHub - lucidrains/transfusion-pytorch: Transfusion 的 Pytorch 实现，“通过单一多模态模型预测下一个 Token 并扩散图像”，来自 MetaAI</a>：Transfusion 的 Pytorch 实现，“通过单一多模态模型预测下一个 Token 并扩散图像”，来自 MetaAI - lucidrains/transfusion-pytorch</li><li><a href="https://x.com/BlinkDL_AI/status/">GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1283895078133436448)** (5 条消息): 

> - `CoT 与 scaling laws`
> - `上下文长度对计算量的影响`
> - `CoT 中的多分支`
> - `树搜索动态`
> - `Log-log plot 观察` 


- **CoT Scaling 显示出从线性到二次方的转变**：在使用 **Chain of Thought (CoT)** 扩展推理侧计算量 (test time compute) 时，计算量随上下文长度线性增长，直到达到某个阈值，随后 **Attention 的二次方成本** 开始占据主导。
   - 如果采样足够密集，这应该会导致 **scaling law 曲线中出现明显的拐点 (kink)**，除非 token 的 **价值 (value)** 发生变化以抵消这种效应。
- **关于在 CoT 中使用多分支的争议**：讨论了使用 **CoT 多分支** 的可行性，并对其缓解上述效应的能力表示怀疑。
   - *一位成员提到*，独立生成许多短的、固定大小的链可能会有帮助，但这引发了关于 scaling dynamics 的疑问。
- **分支因子对 Scaling 的影响有限**：另一位成员建议，*无论分支因子如何*，它在单上下文 scaling 效应面前仅起到 **b^n 乘数** 的作用。
   - 这表明之前观察到的 **计算量 scaling 中的拐点** 仍会出现在 log-log plot 中。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1283519909283762338)** (9 条消息🔥): 

> - `多节点训练挑战`
> - `慢速链路下的 DDP 性能`
> - `用于节点间通信的 HSDP`
> - `最优全局 Batch Size`
> - `Token 估算方法` 


- **使用慢速以太网链路进行多节点训练**：一位用户询问了在 8xH100 机器之间使用慢速以太网链路进行多节点训练的可行性，特别是针对跨节点的 DDP。
   - 另一位成员回应称这将具有挑战性，特别强调了更大的全局 Batch Size 对提升性能的重要性。
- **为 DDP 设置全局 Batch Size**：提到增加 `train_batch_size` 对于饱和 VRAM 并提高训练期间的 DDP 性能至关重要。
   - 一位用户指出，在 Pythia 预训练期间，每个全局 Batch 通常需要 **4M-8M tokens** 才能对收敛产生积极影响。
- **用于增强通信的 HSDP**：建议使用 HSDP (High Speed Data Path) 来优化节点间通信，这可以缓解慢速链路问题。
   - 另一位成员保证，如果对 HSDP 通信进行适当调优以与计算时间重叠，**50G 链路** 对于 DDP 应该是足够的。
- **Token 估算技术**：一位用户分享了一个基于数据集大小以及 micro batch size 和梯度累积步数等参数的 token 估算公式。
   - 该方法使用 **已准备好的 .bin 文件的磁盘大小** 来近似 `num_total_tokens`，建议除以 **4** 来获得估算数量。
- **HSDP 通信调优资源**：一位用户请求在多节点训练期间有效调优 HSDP 通信的参考资料。
   - 其他用户提供了建议，并分享了他们之前用于优化数据传输的方法和经验。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1283536775977111563)** (5 条消息): 

> - `AI Scheduler 工作坊`
> - `针对汽车行业的多文档 RAG 系统`
> - `OpenAI o1 与 o1-mini 模型集成`
> - `LlamaIndex TypeScript 版本发布`
> - `LlamaIndex 黑客松` 


- **使用 Zoom 构建 AI Scheduler！**: **9月20日**加入我们在 AWS Loft 举办的实战工作坊，学习如何使用 Zoom、LlamaIndex 和 Qdrant 创建一个用于提高会议效率的 **RAG 推荐引擎**。
   - *利用我们的转录 SDK 创建高效的会议环境！* [更多详情](https://t.co/v3Ej58AQ6v)。
- **创建针对汽车需求的 RAG 系统**: 了解如何使用 LanceDB 构建多文档 **Agentic RAG 系统**，用于诊断汽车问题并管理保养计划。
   - *设置向量数据库以实现高效的汽车诊断！* [阅读更多](https://t.co/NgMfj95YAd)。
- **OpenAI 将 o1 模型与 LlamaIndex 集成**: 随着 OpenAI 全新 **o1 和 o1-mini 模型**的开放，用户现在可以通过 pip 安装最新版本，将其集成到 LlamaIndex 中。
   - 使用 `pip install -U llama-index-llms-openai` 更新你的安装，并在此处探索其功能 [here](https://t.co/0EgCP45oxV)。
- **LlamaIndex.TS 现已发布！**: TypeScript 爱好者们可以欢呼了，LlamaIndex.TS 现已发布，为偏好 TypeScript 的开发者扩大了接入渠道。
   - 在 [NPM](https://www.npmjs.com/package/llamaindex) 上发现更多关于此版本及其特性的信息。
- **LlamaIndex 黑客松公告**: 第二届 LlamaIndex 黑客松定于 **10月11-13日** 举行，由 Pinecone 和 Vesslai 赞助，提供超过 **$10,000** 的现金奖励和积分。
   - 参与者可以在 [这里](https://t.co/13LHrlQ7ER) 注册，加入这场将 **RAG 技术**与 AI Agent 结合的激动人心的活动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.co/3agScNi74h">llamaindex</a>: [
![NPM Version](https://img.shields.io/npm/v/llamaindex)
](https://www.npmjs.com/package/llamaindex) [
![NPM License](https://img.shields.io/npm/l/llamaindex)
](https://www.npmjs.com/package/llamaindex) ...</li><li><a href="https://t.co/13LHrlQ7ER">AGENTIC RAG-A-THON ($10K 现金奖励)</a>: 由 Pinecone &amp; Vessl 赞助的 LlamaIndex Agentic RAG-a-thon | 2024年10月11日 - 13日
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1283703819972378626)** (28 条消息🔥): 

> - `Llama3.1 Function Calling`
> - `使用 chat_engine 的高级 RAG`
> - `流式传输 ReAct Agent 活动`
> - `在私有服务器上使用 LlamaParse`
> - `在 Qdrant 中按特定文档过滤` 


- **Llama3.1 在 OpenAI 库中支持 Function Calling**: 一位成员询问为什么 OpenAI 库利用 **Llama3.1** 进行 Function Calling，而 **LlamaIndex** 却不行。
   - 另一位成员建议，在初始化期间将 `is_function_calling_model` 的值设为 `True` 可能会解决该问题。
- **从 query_engine 轻松过渡到 chat_engine**: 一位成员解释说，配置使用 `chat_engine` 的**高级 RAG** 非常简单，只需镜像 `query_engine` 初始化时使用的参数即可。
   - 共享了代码示例，以说明如何在两个引擎中启用**基于相似度的查询**。
- **流式传输 ReAct Agent 活动是可行的**: 成员们讨论了从 **LlamaIndex** 流式传输响应以增强与 ReAct Agent 交互的潜力。
   - 提供了在查询引擎设置中启用流式传输的详细解释，包括配置步骤。
- **LlamaParse 可以 On-premise 托管**: 一位成员确认了在私有服务器上运行 **LlamaParse** 的能力，以满足隐私需求。
   - 共享了联系信息，用于讨论与托管 LlamaParse 相关的特定用例。
- **在 Qdrant 向量存储中过滤文档**: 一位成员寻求在 **Qdrant** 中过滤特定文档的建议，并被引导使用元数据（metadata）或原始过滤器（raw filters）。
   - 共享了相关文档链接，以协助实现基于文档 ID 或文件名的过滤。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming/#streaming-response">Streaming - LlamaIndex</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - Hugging Face Space</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/#qdrant-vector-store-metadata-filter">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1283799633658642473)** (1 messages): 

> - `LangChain`
> - `Llama_Index`
> - `HayStack`
> - `Retrieval Augmented Generation (RAG)`
> - `LLM development` 


- **LangChain、Llama_Index 和 HayStack 是否变得臃肿？**：一场讨论引发了对 **LangChain**、**Llama_Index** 和 **HayStack** 等框架在 **LLM** 应用开发中是否变得*过于复杂*的质疑，并建议转向更核心、更简单的 API。
   - 评论者引用了一篇 [Medium 文章](https://medium.com/@jlchereau/do-we-still-need-langchain-llamaindex-and-haystack-and-are-ai-agents-dead-522c77bed94e)，表达了他们对这一现象的看法。
- **理解检索增强生成 (RAG)**：**Retrieval Augmented Generation (RAG)** 应用允许通用的 **LLMs** 通过在定制的文档语料库中搜索相关信息，来回答特定业务的查询。
   - **RAG** 应用的典型流程包括查询向量数据库，然后利用检索到的文档生成答案。



**提到的链接**：<a href="https://medium.com/@jlchereau/do-we-still-need-langchain-llamaindex-and-haystack-and-are-ai-agents-dead-522c77bed94e">我们是否还需要 LangChain、LlamaIndex 和 Haystack，AI Agents 是否已死？</a>：LangChain、LLamaIndex 和 Haystack 是用于 **LLM** 开发的 Python 框架，特别是 **Retrieval Augmented Generation (RAG)** 应用。

  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1283556420674916393)** (2 messages): 

> - `Output Evaluation Techniques`
> - `Hallucination Reduction`
> - `Website Citation Concerns` 


- **缺乏对输出真实性的评估**：一位成员指出，除了旨在减少幻觉的标准 **Prompt** 技术外，他们目前还没有评估输出**真实性**的方法。
   - “请不要在你的下一篇出版物中引用我的网站”被幽默地提及，强调了在使用生成输出时需要保持谨慎。
- **输出生成的来源**：尽管缺乏直接的评估方法，该成员提到用于生成输出的所有来源都列在他们的网站上。
   - 这意味着他们具备透明度意识，尽管这并不能确保输出的真实性。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1283507812248457359)** (22 messages🔥): 

> - `DSPy Chatbot Customization`
> - `OpenAI O1 Pricing`
> - `Requests Per Minute for O1`
> - `DSPy and O1 Integration` 


- **DSPy Chatbot 中的动态定制**：一位成员询问如何在不硬编码客户信息的情况下，向 **DSPy** 生成的 **Prompt** 中添加特定于客户的定制内容，并建议采用后处理步骤。
   - 另一位成员建议使用 'context' 输入字段，将其类比为 **RAG** 方法，并建议使用通用格式的上下文来训练 **Pipeline**。
- **对 O1 定价的困惑**：关于 **OpenAI** **O1** 的定价引发了讨论，一名成员质疑其结构，另一名成员确认 **O1 mini** 更便宜。
   - 成员们表达了对 **DSPy** 和 **O1** 进行对比分析的愿望，其中一人建议试用 **O1** 以评估成本效益。
- **20 RPM 指标澄清**：在关于 **O1** 的讨论中，一位成员分享说 '20rpm' 指的是 '每分钟请求数' (**requests per minute**)，引发了关于上下文的提问。
   - 这一指标对于目前 **O1** 和 **DSPy** 等服务的性能分析似乎非常重要。
- **对 DSPy 和 O1 兼容性的好奇**：一位成员询问 **DSPy** 是否支持 **O1-preview**，突显了社区对集成这些工具的持续兴趣。
   - 这反映了社区对 **DSPy** 和 **O1** 之间实现更多功能的期待。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1283509862848008292)** (5 messages): 

> - `Mixed Precision Training`
> - `FP16 vs BF16`
> - `Quantization Issues`
> - `Tokenizer API Standardization` 


- **混合精度训练的复杂性**：在 **mixed precision modules** 与其他特性之间保持兼容性需要**额外的工作**；目前注意到 bf16 半精度训练严格优于 fp16，因为旧版 GPU 上的 fp16 支持存在问题。
   - 一位成员指出，直接使用 fp16 会导致溢出错误，从而由于 **full precision gradients** 增加了系统复杂性和内存占用。
- **FP16 训练的挑战**：一位成员强调，虽然可以通过自动混合精度 (AMP) 实现 FP16 训练，但它必须将权重保留在 FP32 中，这显著增加了 **large language models** 的内存消耗。
   - 还有人提到，虽然 fairseq 可以处理全 FP16 训练，但代码库的复杂性使其具有挑战性，且可能不值得投入工程精力。
- **Tokenizer API 标准化讨论**：一位成员建议在解决 eos_id 问题之前，先处理 issue #1503 以统一 tokenizer API，这意味着这可以简化未来的开发。
   - 由于 #1503 已经有负责人，该成员计划探索其他可能的修复方案，以贡献于整体改进。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1283503567797489746)** (17 messages🔥): 

> - `FlexAttention integration`
> - `PackedDataset performance`
> - `INT8 mixed-precision training`
> - `FSDP integration`
> - `QAT vs INT8 training` 


- **FlexAttention 集成获批**：用于文档掩码 (document masking) 的 **FlexAttention** 集成已合并，其潜力令人兴奋。
   - *有人提出了关于每个 'pack' 是否被填充到 max_seq_len，以及没有完美打乱 (perfect shuffle) 对收敛影响的问题。*
- **PackedDataset 在 INT8 下表现出色**：性能测试显示，在 A100 上使用 torchao 中的 INT8 混合精度配合 **PackedDataset** 可获得 **40% 的加速**。
   - *一位成员计划运行更多测试，确认 PackedDataset 的固定 seq_len 非常契合他们的 INT8 策略。*
- **关于 FSDP 集成的 PR 讨论**：讨论了一个在 torchtune 中暴露新设置的 PR 提案，因为它仅包含三行代码。
   - *成员们对将 **FSDP + AC + compile** 的支持与新特性集成表现出兴趣。*
- **提供了 QAT 的澄清**：一位成员对比了 **QAT** (量化感知训练) 与 **INT8 混合精度训练**，强调了它们目标上的关键差异。
   - *他们指出，虽然 QAT 旨在提高精度，但 INT8 训练主要侧重于提升训练速度，并且在精度损失极小的情况下可能不需要 QAT。*
- **关于消融实验的讨论**：成员们讨论了对 **num_warm_up steps** 的潜在修改，特别是考虑将其从 **100 减少到 10**。
   - *对早期的消融研究结果做出了贡献，强调了沟通和未来的实验。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/eb92658a360d7a7d4ce1c93bbcf99c99a2e0943b/torchtune/data/_collate.py#L204">torchtune/torchtune/data/_collate.py at eb92658a360d7a7d4ce1c93bbcf99c99a2e0943b · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。欢迎通过在 GitHub 上创建账号来为 torchtune 的开发做贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1403">Update quantization to use tensor subclasses by andrewor14 · Pull Request #1403 · pytorch/torchtune</a>: 摘要：在 torchao 中，我们正在将量化流程从模块替换 (module swap) 迁移到 tensor 子类 (tensor subclasses)。现有的 Int8DynActInt4WeightQuantizer 将在不久的将来被弃用，取而代之的是...
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1283853886280699965)** (4 条消息): 

> - `New OpenAI model`
> - `Llama Index`
> - `Reflection 70B` 


- **关于新 OpenAI 模型的讨论**：一位成员询问是否有人尝试过新的 **OpenAI model**，暗示这可能是最近发布的一个版本。
   - *这引发了人们对其功能和反响的好奇及进一步讨论。*
- **对 Llama Index 的熟悉程度**：一位成员表示熟悉 **Llama Index**，表现出对模型交互工具的兴趣。
   - *这引发了关于它如何与新的 OpenAI model 关联的潜在探讨。*
- **Reflection 70B 被视为失败之作**：有成员担心 **Reflection 70B** 模型被视为 **dud**（哑弹/失败之作），并引发了对新 OpenAI 发布时机的猜测。
   - *该评论以轻松的方式分享，暗示这是对之前失望情绪的一种回应。*
- **OpenAI 模型被视为营销策略**：一位成员认为新的 OpenAI model 仅仅是一个 **marketing model**（营销模型），质疑其真实价值。
   - *这种情绪反映了对产品发布背后意图的怀疑。*


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1283794555367591996)** (6 条消息): 

> - `Duplicating Code for Chat Formats`
> - `Internal Data Format for Datasets`
> - `Chat Template Feature`
> - `Multimodal Support in Chat Templates`
> - `Llama 3.1 Performance` 


- **继续为聊天格式重复代码**：一位成员询问是否计划在 Axolotl 中继续为所有聊天格式重复代码，同时承认向内部数据格式迁移是一个积极的方向。
   - *我总体认为转向内部数据格式……是一个伟大的想法*。
- **强调 Chat Template 特性**：另一位成员认为，专注于用于模板化的 chat template 特性至关重要，并指出目前尚未发现任何易用性问题。
   - *我还没遇到过它无法处理的使用场景*。
- **训练中的 Reflection 处理**：一位成员指出，不应该对错误的初始想法进行训练，而应专注于 reflection（反思）和改进后的输出，这一点非常重要。
   - 他们强调了测试中的一个示例，说明了这种方法。
- **Chat Template 的多模态挑战**：有成员担心当前的 chat template 是否支持多模态功能，特别是在响应中的 token masking 方面。
   - 成员们承认，将推理用的 chat template 用于多模态任务具有复杂性。
- **Llama 3.1 模板的正面体验**：一位成员报告称，在使用带有 masking 的 `chat_template` 提示策略时，**Llama 3.1** 或 ChatML 模板没有出现问题。
   - 不过，他们建议保持谨慎，因为他们尚未探索多模态功能。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1283667319209590859)** (3 条消息): 

> - `DPO Custom Formats`
> - `Llama 3.1 Tool Call Issues` 


- **澄清 DPO 格式预期**：一位成员澄清了运行 DPO 的预期格式，使用了 `<|begin_of_text|>{prompt}` 后接 `{chosen}<|end_of_text|>` 和 `{rejected}<|end_of_text|>` 的符号。
   - 他们提到了自定义格式处理的更新，并引用了 [GitHub 上的此 issue](https://github.com/axolotl-ai-cloud/axolotl/issues/1417) 以获取更多上下文。
- **Llama 3.1 在工具响应方面的困扰**：一位成员报告了 `llama3.1:70b` 模型的问题，指出尽管调用了工具，但生成的响应却是无意义的。
   - 在一个案例中，在工具指示夜间模式已停用后，助手仍未能对后续请求做出恰当回应。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1417)">Issues · axolotl-ai-cloud/axolotl</a>：尽管提问。通过在 GitHub 上创建账户，为 axolotl-ai-cloud/axolotl 的开发做出贡献。

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1283653737067581461)** (8 messages🔥): 

> - `Flux by RenderNet`
> - `SD 团队变动`
> - `来自 SD 的 API/纯网页版模型`
> - `SD 的开源状态` 


- **Flux by RenderNet 发布！**: 🚀 **来自 @bfl_ml 的 Flux** 现已在 RenderNet 上线，使用户能够仅凭一张参考图即可创建**超写实**图像，无需 LORAs。
   - *准备好赋予你的角色生命了吗？* 用户只需点击几下即可进行尝试。
- **SD 团队更名**: 成员们讨论了 **SD 团队**已经更名，因为他们不再为 **SAI** 工作。
   - *所以 SD 已经死了吗？* 一位成员针对更名后团队的状态发表了感想。
- **SD 的开源地位受到质疑**: 成员们担心 SD 在**开源领域**不再活跃，暗示其社区参与度有所下降。
   - *基本上如果你关心开源，SD 似乎已经名存实亡了，* 另一位成员对现状评论道。
- **SD 发布新的 API/纯网页版模型**: 尽管存在担忧，SD 团队最近发布了一个 **API/纯网页版模型**，显示了一定程度的活跃。
   - 然而，成员们对其开源努力的承诺仍普遍持怀疑态度。



**提到的链接**: <a href="https://x.com/rendernet_ai/status/1833865069744083198">来自 rendernet (@rendernet_ai) 的推文</a>: 🚀 来自 @bfl_ml 的 Flux 现已在 RenderNet 上线！只需一张参考图即可创建超写实、角色一致的图像 —— 无需 LORAs。准备好赋予你的角色生命了吗？尝试使用 j...

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

helium__: https://openai.com/index/introducing-openai-o1-preview/
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1283738949407211521)** (1 messages): 

> - `Sci Scope 亮点`
> - `AI 研究通讯`
> - `ArXiv 论文摘要`
> - `AI 文献导航` 


- **通过 Sci Scope 保持更新**: Sci Scope 将主题相似的新 [ArXiv 论文](https://www.sci-scope.com/)进行分组，并每周进行总结，从而简化你在 AI 研究中的阅读选择。
   - *订阅通讯* 即可直接在收件箱中接收简洁的概览，让你更轻松地保持信息同步。
- **轻松获取 AI 研究**: 该平台旨在通过用户友好的摘要系统，简化在快速变化的 AI 文献景观中的导航。
   - 通过整理相似论文，它增强了你识别该领域相关且有趣阅读材料的能力。



**提到的链接**: <a href="https://www.sci-scope.com/">Sci Scope</a>: 一份关于 AI 研究的 AI 生成报纸

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1283530862570242192)** (2 messages): 

> - `` 


- **聊天中的问候**: 一位成员用简单的“cheers!”表达了感谢。
   - 随后另一位成员回复了“Thanks”。
- **社区参与度保持较低**: 互动仅由两次简短的交流组成，没有提出重要的讨论话题。
   - 这表明目前对话的参与度较低。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1283721048855937025)** (1 messages): 

> - `HTML chunking package` (HTML 分块软件包)
> - `Web automation tools` (Web 自动化工具)
> - `HTML parsing techniques` (HTML 解析技术) 


- **介绍 HTML Chunking 软件包**：一个名为 `html_chunking` 的新包已发布，它能高效地对 HTML 内容进行分块和合并，同时在 Web 自动化任务中遵循 token 限制。
   - 该过程结构良好，确保了准确的 HTML 解析，并保留了带有适当标签属性的结构，以便在各种应用中使用。
- **HTML Chunking 演示代码**：分享了一个演示代码片段，展示了如何使用 `html_chunking` 包中的 `get_html_chunks` 来处理具有最大 token 限制的 HTML 字符串，并保留有效的 HTML 结构。
   - 输出结果为多个有效的 HTML 块，并对过长的属性进行截断以确保长度可控。
- **与现有工具的比较**：相比于 LangChain 的 `HTMLHeaderTextSplitter` 和 LlamaIndex 的 `HTMLNodeParser`（它们会剥离 HTML 上下文），`html_chunking` 包被定位为更优的 HTML 分块解决方案。
   - 这些现有工具仅提取文本内容，限制了它们在需要完整 HTML 上下文场景中的效用。
- **行动号召：探索 HTML Chunking**：鼓励用户研究 `html_chunking` 以用于 Web 自动化及相关任务，强调了其在准确 HTML 分块方面的优势。
   - 提供了 [HTML chunking PYPI 页面](https://pypi.org/project/html-chunking/) 和其 [GitHub 仓库](https://github.com/KLGR123/html_chunking) 的链接以供进一步探索。



**提及的链接**：<a href="https://github.com/KLGR123/html_chunking">GitHub - KLGR123/html_chunking: A Python implementation for token-aware HTML chunking that preserves structure and attributes, with optional cleaning and attribute length control.</a>：一个用于 token 感知型 HTML 分块的 Python 实现，可保留结构和属性，并具有可选的清理和属性长度控制功能。- KLGR123/html_chunking

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1283798329402331217)** (2 messages): 

> - `George Hotz's Engagement Policy` (George Hotz 的互动政策)
> - `Terms of Service for ML Developers` (针对 ML 开发者的服务条款) 


- **George Hotz 建议避免不必要的 ping**：一名成员提醒其他人，**George** 要求除非问题被认为有用，否则不要 @ 他，从而在讨论中强化一种注重相关性的文化。
   - *只需搜索一次*即可找到此政策，鼓励成员利用现有资源。
- **针对特定行为的新服务条款**：George 宣布将推出一项**服务条款 (ToS)**，旨在禁止**加密货币挖矿和转售**等行为，以维持专注的环境。
   - 该政策是为 **ML 开发者**量身定制的，强调了通过他们的 MacBook 访问强大 GPU 的目标。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

xprose7820: http://www.catb.org/~esr/faqs/smart-questions.html
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1283529301815459860)** (2 条消息): 

> - `LLM Applications`
> - `LLM Observability`
> - `LLM Monitoring`
> - `LLM Integrations`
> - `Fine-tuning for Translations` 


- **Literal AI 在易用性方面表现出色**：一位成员分享了他们使用 **Literal AI** 的积极体验，特别称赞了其在 [literalai.com](https://literalai.com/) 上展示的易用性。
   - 他们提到该平台提供了支持 LLM 应用生命周期的实用集成和功能。
- **LLM 可观测性提升应用生命周期**：强调了 **LLM Observability** 的重要性，指出它通过实现更快的迭代和调试来增强应用生命周期。开发者可以利用日志来 Fine-tune 更小的模型，在提高性能的同时降低成本。
- **将 Prompt 管理转化为风险规避**：讨论了 **Prompt 性能追踪**，将其作为确保在部署新版本更新前不发生回归的手段。这种主动管理有助于保持 LLM 输出的一致性和可靠性。
- **全面的 LLM 监控设置**：对话中包含了关于在集成日志评估后建立全面的 **LLM Monitoring** 与分析系统的见解。这套设置被认为对于在生产环境中维持最佳性能至关重要。
- **针对翻译微调 LLM**：有人询问了专门针对翻译进行 **Fine-tuning LLMs** 的经验，并指出一个常见问题：LLM 擅长捕捉大意，但无法传达原始的语气或风格。
   - 这个问题凸显了当前 LLM 翻译能力中开发者寻求解决的空白。



**提及的链接**：<a href="https://literalai.com/">Literal AI - RAG LLM 可观测性与评估平台</a>：Literal AI 是专为开发者和产品负责人构建的 RAG LLM 评估与可观测性平台。 

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1283622939480162334)** (2 条消息): 

> - `Test Results Accuracy`
> - `Prompt Splicing for Qwen2-7B-Chat` 


- **测试准确率结果**：进行了多次测试，其中 **irrelevance** 达到了 **1.0** 的 **Accuracy**，而包括 **java** 和 **javascript** 在内的其余项得分均为 **0.0**。
   - 像 **live_parallel_multiple** 和 **live_simple** 这样的测试也表现不佳，引发了对模型有效性的质疑。
- **寻求关于 Prompt 拼接的建议**：一位成员对 **qwen2-7b-chat** 的 **糟糕表现** 表示担忧，并询问：“这是 Prompt Splicing 的问题吗？”
   - 他们请求有关 **Prompt Splicing** 的见解或方法，以改善测试体验。


  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1283772479466242128)** (1 条消息): 

> - `Predictive Maintenance`
> - `Unsupervised Learning in Maintenance`
> - `Embedded Systems Monitoring` 


- **寻求关于预测性维护的见解**：一位成员询问了关于 **Predictive Maintenance** 的经验，并请求提供关于最佳模型和实践的论文或书籍等资源。
   - 他们提到由于缺乏追踪的系统故障记录，且手动标记事件不切实际，因此需要 **Unsupervised Learning** 方法。
- **关注机械和电气系统**：讨论的设备被描述为部分是机械的，部分是电气的，在运行期间会记录大量事件。
   - 成员们指出，有效的监控可以改进维护策略并减少未来的故障。


  

---



---



---



---



{% else %}


> 完整的各频道详情因邮件长度限制已截断。 
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}