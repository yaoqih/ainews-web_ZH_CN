---
companies:
- openai
- anthropic
- nvidia
- microsoft
- boston-dynamics
- meta-ai-fair
- runway
- elevenlabs
- etched
- osmo
- physical-intelligence
- langchain
date: '2024-11-05T02:51:39.237539Z'
description: '**提示词查找 (Prompt lookup)** 和 **投机性解码 (Speculative Decoding)** 技术正日益受到关注，**Cursor**、**Fireworks**
  已有相关实现，**Anthropic** 也预告了相关功能。**OpenAI** 利用这些方法实现了更快的响应速度和文件编辑，效率提升了约 **50%**。社区正积极探索这些技术进步在
  AI 工程中的应用场景。


  最近的动态展示了 **NVIDIA**、**OpenAI**、**Anthropic**、**Microsoft**、**Boston Dynamics** 和
  **Meta** 等公司的进展。关键技术洞察涵盖了 CPU 推理能力、多模态检索增强生成 (RAG) 以及神经网络基础。新的 AI 产品包括完全由 AI 生成的游戏和先进的内容生成工具。此外，讨论还涉及了
  AI 研究实验室面临的官僚主义和资源分配等挑战，以及 AI 安全和治理方面的担忧。'
id: f20699bb-fdf1-4cf6-ac00-50116aa4a5c5
models:
- claude-3-sonnet
- mrt5
original_slug: ainews-openai-beats-anthropic-to-releasing
people:
- adcock_brett
- vikhyatk
- dair_ai
- rasbt
- bindureddy
- teortaxestex
- svpino
- c_valenzuelab
- davidsholz
title: OpenAI 抢在 Anthropic 之前发布了投机性解码。
topics:
- speculative-decoding
- prompt-lookup
- cpu-inference
- multimodality
- retrieval-augmented-generation
- neural-networks
- optimization
- ai-safety
- governance
- model-architecture
- inference-economics
- content-generation
---

<!-- buttondown-editor-mode: plaintext -->**Prompt lookup is all you need.**

> 2024年11月1日至11月4日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**216** 个频道，**7073** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**766 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

自从最初的 [Speculative Decoding](https://arxiv.org/abs/2211.17192) 论文（以及像 [Hydra](https://arxiv.org/abs/2402.05109) 和 [Medusa](https://arxiv.org/abs/2401.10774) 这样的变体）发布以来，社区一直在竞相部署它。5 月，Cursor 和 Fireworks 宣布了他们超过 1000tok/s 的快速应用模型（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-to-be-named-9199/)，[Fireworks 的技术文章在此](https://fireworks.ai/blog/cursor) —— 请注意，本期最初发布时曾有一个事实错误，称 Speculative Decoding API 尚未发布，正如这篇文章所解释的，Fireworks 在 5 个月前就发布了 Speculative Decoding API。）。8 月，Zed 预告了 Anthropic 的 [新 Fast Edit 模式](https://x.com/zeddotdev/status/1825967818329731104)。但 Anthropic 的 API 尚未发布……这给 OpenAI 留下了大展身手的空间：


![image.png](https://assets.buttondown.email/images/22d5ee3c-0e47-4971-be81-9565dee27a48.png?w=960&fit=max)


Factory AI 报告了更快的响应时间和文件编辑速度：


![image.png](https://assets.buttondown.email/images/fe7032bf-4a45-42af-a23e-0024a50df607.png?w=960&fit=max)


这种对 draft tokens 的额外处理成本有点模糊（[取决于 32 个 token 的匹配情况](https://x.com/nikunjhanda/status/1853603080249716928)），但你可以将 ~50% 作为一个不错的经验法则。


![image.png](https://assets.buttondown.email/images/6e433410-27ca-48ac-9fbd-ce909bcd1823.png?w=960&fit=max)


正如我们在本周的 Latent.Space 文章中所分析的，这很好地契合了为 AI Engineer 用例开发的丰富选项：


![image.png](https://assets.buttondown.email/images/6b0141cf-68d3-4999-ba88-46032a676fa2.png?w=960&fit=max)


---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 技术与行业动态**

- **主要公司进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1853120716859756964) 强调了包括 Etched, Runway, Figure, NVIDIA, OpenAI, Anthropic, Microsoft, Boston Dynamics, ElevenLabs, Osmo, Physical Intelligence 和 Meta 在内的多家公司的重大进展。

- **模型与基础设施更新**：
  - [@vikhyatk](https://twitter.com/vikhyatk/status/1853189095822090719) 宣布 CPU 推理功能现已支持本地运行。
  - [@dair_ai](https://twitter.com/dair_ai/status/1853119453837353179) 分享了涵盖 MrT5, SimpleQA, Multimodal RAG 和 LLM 几何概念的顶级 ML 论文。
  - [@rasbt](https://twitter.com/rasbt/status/1853073656525599022) 发表了一篇文章，解释了 Multimodal LLMs 的两种主要方法。
  - [@LangChainAI](https://twitter.com/LangChainAI/status/1853161977926819966) 利用 NVIDIA 课程材料演示了结合 LLMs 的 RAG Agents。

- **产品发布与功能**：
  - [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1853110073691623574) 讨论了除 Prompt 之外的新内容生成方式，包括 Advanced Camera Controls。
  - [@adcock_brett](https://twitter.com/adcock_brett/status/1853120739018244343) 报道了 Etched 和 DecartAI 的 Oasis，这是首款完全由 AI 生成的可玩 Minecraft 游戏。
  - [@bindureddy](https://twitter.com/bindureddy/status/1853173227654377718) 展示了 AI Engineer 如何通过英文 Prompt 构建带有 RAG 的自定义 AI agents。

- **研究与技术洞察**：
  - [@teortaxesTex](https://twitter.com/teortaxesTex/status/1853087440702964018) 讨论了 LLMs 中专注于推理经济学（inference economics）的架构工作。
  - [@svpino](https://twitter.com/svpino/status/1853059600343564323) 强调了理解基础知识的重要性：神经网络、损失函数、优化技术。
  - [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1853203354341822667) 详细说明了 AI 研究实验室在官僚主义和资源分配方面面临的挑战。

**行业评论与文化**

- **职业与发展**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1853203354341822667) 强调了 AI 研究实验室中的挫败感，包括官僚主义、审批延迟和资源分配问题。
- **技术讨论**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1853079146819113166) 讨论了 AI 安全担忧及其对治理的影响。
- [@DavidSHolz](https://twitter.com/DavidSHolz/status/1853154201922379796) 通过一次《星际争霸》（Starcraft）游戏体验分享了一个有趣的 AI 安全类比。

**幽默与梗**

- [@fchollet](https://twitter.com/fchollet/status/1853193278126240172) 调侃了 AGI 与滑动输入法准确度之间的对比。
- [@svpino](https://twitter.com/svpino/status/1853219435680805111) 评论了这款“蝴蝶应用”（指 Bluesky）的宁静特质。
- [@vikhyatk](https://twitter.com/vikhyatk/status/1853142547872559472) 对 AI 访问和供应商发表了讽刺性评论。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. Hertz-Dev：首个延迟仅为 120ms 的开源实时音频模型**

- **[🚀 分析了不同输入长度（从 5 到 200 个单词）下各种 TTS 模型的延迟！](https://i.redd.it/5xff63lvrpyd1.png)** ([评分: 114, 评论: 24](https://reddit.com//r/LocalLLaMA/comments/1giqxph/analyzed_the_latency_of_various_tts_models_across/))：一项针对 **text-to-speech (TTS)** 模型的性能分析测量了 **5** 到 **200** 个单词输入下的 **latency**，结果显示 **Coqui TTS** 在短句（平均推理时间 **0.8 秒**）和长段落（**2.1 秒**）中始终优于其他模型。研究还发现，**Microsoft Azure** 和 **Amazon Polly** 的延迟随输入长度呈线性增长，而开源模型在超过 **100 个单词** 的输入时表现出更不稳定的性能模式。

- **[Hertz-Dev：一个开源的 8.5B 音频模型，用于实时对话式 AI，在单张 RTX 4090 上具有 80ms 理论延迟和 120ms 实际延迟](https://v.redd.it/5rt1niy7xsyd1)** ([Score: 591, Comments: 78](https://reddit.com//r/LocalLLaMA/comments/1gj4wri/hertzdev_an_opensource_85b_audio_model_for/))：**Hertz-Dev** 是一个开源的 **8.5B 参数**音频模型，在单张 **RTX 4090** GPU 上运行，实现了 **120ms** 的实际延迟和 **80ms** 的理论延迟，适用于对话式 AI。该模型似乎是为实时音频处理而设计的，尽管帖子中未提供具体的实现细节或基准测试方法。
  - Hertz 的 **70B 参数版本**目前正在训练中，并计划扩展到更多模态。运行这个更大的模型可能需要 **H100 GPUs**，不过有人建议量化（quantization）可能有助于降低硬件需求。
  - 围绕**开源状态**展开了讨论，用户注意到该模型仅发布了权重和推理代码（类似于 **Llama**、**Gemma**、**Mistral**）。完全开源模型的显著例外包括 **Olmo** 和 **AMD 的 1B 模型**。
  - 该模型具有 **17 分钟的上下文窗口**，理论上可以像其他 Transformer 一样使用音频数据集进行微调。其实际延迟（**120ms**）优于 **GPT-4o**（**平均 320ms**），并接近人类对话间隔（**200-250ms**）。


**主题 2. 语音克隆进展：F5-TTS vs RVC vs XTTS2**

- **在 Linux 上通过按键与本地 LLM 对话。语音转文字到任何窗口，适用于大多数桌面环境。两个热键即是 UI。** ([Score: 51, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1giozl9/speaking_with_your_local_llm_with_a_key_press_on/))：**BlahST** 工具使 **Linux** 用户能够使用热键执行本地**语音转文字**和 **LLM 交互**，利用了 **whisper.cpp**、**llama.cpp**（或 **llamafile**）和 **Piper TTS**，无需 Python 或 JavaScript。在配备 **Ryzen** CPU 和 **RTX3060** GPU 的系统上运行，该工具使用 **gemma-9b-Q6** 模型可达到 **~34 tokens/秒**，并实现 **90 倍实时**语音推理，同时保持较低的系统资源占用，并支持包括中文翻译在内的多语言能力。

- **如果有大量参考音频，最好的开源语音克隆方案是什么？** ([Score: 71, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1gj14oa/best_open_source_voice_cloning_if_you_have_lots/))：对于每个角色拥有 **10-20 分钟**参考音频的语音克隆，作者正在寻找 **ElevenLabs** 和 **F5-TTS** 的替代方案用于自托管部署。该帖子特别要求能够针对单个角色预训练模型以便后续推理的解决方案，摆脱像 F5-TTS 这种针对极少参考音频优化的少样本学习（few-shot learning）方法。
  - **RVC** (Retrieval-based Voice Conversion) 表现良好，但需要输入音频进行转换。当与 **XTTS-2** 结合使用时，由于 TTS 伪影，结果可能参差不齐，不过使用 [alltalk-tts beta](https://github.com/erew123/alltalk_tts/tree/alltalkbeta) 可以简化这一过程。
  - 微调 **F5-TTS** 使用 [lpscr 的实现](https://github.com/lpscr/F5-TTS) 产生了高质量的结果且推理速度快，达到了与 **ElevenLabs** 相当的质量。该过程现在已通过 `f5-tts_finetune-gradio` 命令集成到主仓库中。
  - **GPT-SoVITS**、**MaskCGT** 和 **OpenVoice** 被提及为强有力的替代方案。**MaskCGT** 在零样本（zero-shot）性能上领先，而 **GPT-SoVITS** 被认为超越了微调后的 **XTTS2** 或 **F5** 参考语音克隆。


**主题 3. Token 管理与模型优化技术**

- **[小型模型 (<5B) 的 MMLU-Pro 评分] (https://i.redd.it/dbqap2z19nyd1.jpeg)** ([Score: 166, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1gii24g/mmlupro_scores_of_small_models_5b/))：**MMLU-Pro** 基准测试每个问题有 **10 个多选题选项**，为随机猜测设定了 **10%** 的基准分数。该测试评估了 **5B 参数以下语言模型**的性能。

- **[tips for dealing with unused tokens? keeps getting clogged](https://v.redd.it/9bteq03a1syd1)** ([Score: 150, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1gj1e9p/tips_for_dealing_with_unused_tokens_keeps_getting/)): LLM 中的 **Token usage optimization** 需要同时管理输入和输出 Token，以防止内存阻塞和处理效率低下。该帖子似乎在寻求建议，但缺乏关于所遇到的未使用 Token 具体问题或所使用的 LLM 系统的详细信息。
  - **VLLM** 正在开发一种用于移除未使用 Token 的 **KV cache eviction** 重要实现，代码托管在 [GitHub](https://github.com/IsaacRe/vllm-kvcompress/tree/main)。这种方法与 **llama.cpp** 或 **exllama** 中使用的压缩方法不同。
  - 社区引用了包括 **DumpSTAR** 和 **DOCUSATE** 在内的 Token 分析研究论文，Token 过期方法自 [2021](https://arxiv.org/abs/2105.06548) 年起就被提出。 
  - 许多用户对帖子的背景表示困惑，点赞最高的评论指出，回复呈现两极分化：一部分人完全理解，而另一部分人则完全迷失在技术讨论中。


**Theme 4. MG²: New Melody-First Music Generation Architecture**

- **[MG²: Melody Is All You Need For Music Generation](https://awesome-mmgen.github.io/)** ([Score: 57, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1gj80lf/mg²_melody_is_all_you_need_for_music_generation/)): **MG²** 是一个在 **500,000 个样本数据集**上训练的新型 **music generation model**，它专注于旋律生成，而忽略了和声和节奏等其他音乐元素。该模型证明了旋律本身就包含足够的高质量音乐生成信息，挑战了整合多种音乐组件的传统方法，并在使用显著更少参数的情况下实现了与更复杂模型相当的效果。该研究引入了一种专门为旋律处理设计的创新 **self-attention mechanism**，使模型能够捕捉音乐序列中的长程依赖关系，同时保持计算效率。
  - **MusicSet dataset** 现已在 [HuggingFace](https://huggingface.co/datasets/ManzhenWei/MusicSet) 上发布，包含 **500k 个样本**的高质量音乐波形，并附带描述和独特的旋律。用户对该数据集的发布表示热烈欢迎，认为其推动了音乐生成研究。
  - 多位用户发现该模型的示例输出不尽如人意，特别是对缺乏旋律输入功能的担忧。一位用户特别希望能将哼唱的旋律转化为电影质感的乐曲。
  - 关于**电子游戏音乐生成**的讨论突出了对创作类似于《洛克人》（**Mega Man**）和《大金刚国度》（**Donkey Kong Country**）等经典原声带的兴趣，并提到 [Suno](https://suno.com/) 是目前考虑版权问题的游戏音乐生成解决方案。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型性能与基准测试**

- **SimpleBench 揭示了人类与 AI 推理之间的差距**：人类基准达到 **83.7%**，而顶级模型如 **o1-preview (41.7%)** 和 **3.6 Sonnet (41.4%)** 在基础推理任务中表现挣扎，突显了在时空推理（spatial-temporal reasoning）和社交智能（social intelligence）方面的局限性 [/r/singularity](https://www.reddit.com/r/singularity/comments/1gj4osx/simplebench_where_everyday_human_reasoning_still/)。
  - 关键评论指出，模型从语言中获取对世界的理解，而人类是在已习得的世界模型（world models）之上构建语言。

**AI 安全与基础设施**

- **Nvidia GeForce GPU 存在严重安全漏洞**：由于发现安全缺陷，敦促所有 Nvidia GeForce GPU 用户立即更新驱动程序 [/r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1gio0q2/security_flaws_found_in_all_nvidia_geforce_gpus/)。

**AI 图像生成进展**

- **FLUX.1-schnell 模型发布，提供免费无限次生成**：新网站上线，提供对 FLUX.1-schnell 图像生成能力的无限制访问 [/r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1gizpk7/launched_a_website_where_you_can_use_flux1schnell/)。
- **AI 图像生成的历史进展**：视觉对比展示了从 2015 年到 2024 年 AI 图像生成的巨大进步，并详细讨论了从早期的 DeepDream 到现代模型的演变 [/r/singularity](https://www.reddit.com/r/singularity/comments/1gj613q/the_worst_it_will_ever_be/)。

**AI 伦理与社会**

- **Yuval Noah Harari 警告 AI 驱动的现实扭曲**：讨论 AI 创建沉浸式但可能具有欺骗性的数字环境对社会的潜在影响 [/r/singularity](https://www.reddit.com/r/singularity/comments/1givmks/yuval_noah_harari_says_ai_may_trap_us_in_a_world/)。

**迷因与幽默**

- **AI 生成的写实动漫**：高赞帖子（**942 分**）展示了写实动漫风格的 AI 生成作品 [/r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1gilgt4/what_model_can_do_realistic_anime_like_this/)。
- **AI 安保摄像头幽默**：一段关于南非安保摄像头系统的视频 [/r/singularity](https://www.reddit.com/r/singularity/comments/1giv19x/camera_with_security_system_included_in_south/)。

---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. 重大 LLM 发布与模型更新**

- [**Claude 3.5 Haiku 强势发布**](https://anthropic.com/claude/haiku)：**Anthropic** 发布了 **Claude 3.5 Haiku**，虽然各项基准测试表现更佳，但其 **4 倍的价格涨幅**让用户措手不及。社区正在争论性能的提升是否足以支撑其成本的增加。
- [**OpenAI 的 O1 模型现身又消失**](https://x.com/ArDeved/status/1852649549900242946)：用户曾通过调整 URL 短暂访问到了神秘的 **O1 模型**，并在 OpenAI 关闭入口前体验了其高级功能。关于该模型的能力和正式发布计划的猜测层出不穷。
- [**免费 Llama 3.2 模型亮相 OpenRouter**](https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free)：**OpenRouter** 现在提供对 **Llama 3.2** 模型的免费访问，包括速度大幅提升的 **11B** 和 **90B** 变体。用户对 **11B** 变体达到 **900 tps** 的表现感到兴奋。

**主题 2. AI 在安全与医疗领域的进展**

- [**Llama 3.1 在黑客基准测试中胜过 GPT-4o**](https://arxiv.org/abs/2410.17141)：新的 **PentestGPT** 基准测试显示，**Llama 3.1** 在自动化渗透测试中的表现优于 **GPT-4o**。不过，这两个模型在网络安全应用方面仍有改进空间。
- [**Google 的 MDAgents 旨在优化医疗决策**](https://huggingface.co/posts/aaditya/563565199854269)：**Google** 推出了 **MDAgents**，这是一种 LLM 的自适应协作机制，旨在增强医疗决策能力。像 **UltraMedical** 和 **FEDKIM** 这样的模型有望彻底改变医疗 AI 领域。
- [**Aloe Beta 作为开源医疗 LLM 萌芽**](https://www.linkedin.com/posts/ashwin-kumar-g_excited-to-introduce-aloe-a-family-of-activity-7259240192006373376-VWa7)：使用 **axolotl** 进行微调的 **Aloe Beta** 带来了一系列开源医疗 LLM。这标志着 AI 驱动的医疗解决方案取得了重大进展。

**主题 3. LLM 微调与性能挑战**

- [**Unsloth AI 开启微调极速模式**](https://github.com/unslothai/unsloth)：**Unsloth AI** 使用 **LoRA** 将模型微调速度提升了近 **2 倍**，并大幅降低了 VRAM 占用。用户们正在分享高效优化模型的策略。
- [**Hermes 405b 步履蹒跚，其他模型全速冲刺**](https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud)：**Hermes 405b** 用户反映响应时间极慢且频繁报错，引发了不满。在用户寻找替代方案的同时，关于速率限制（rate limiting）和 API 问题的猜测不断。
- [**LM Studio 的混合 GPU 支持喜忧参半**](https://lmstudio.ai)：**LM Studio** 支持 AMD 和 Nvidia GPU 混合配置，但由于依赖 **Vulkan**，性能受到了影响。为了获得最佳效果，建议仍使用型号一致的 GPU。

**主题 4. AI 进军游戏与创意领域**

- [**Oasis 游戏开启 AI 世界新前沿**](https://oasis-model.github.io/)：**Oasis** 是首款完全由 AI 生成的游戏，允许玩家探索实时交互环境。这里没有传统游戏引擎——只有纯粹的 AI 魔法。
- [**Open Interpreter 为 AI 爱好者带来惊喜**](https://github.com/OpenInterpreter/open-interpreter)：随着用户将其与 **Screenpipe** 等工具集成以处理 AI 任务，**Open Interpreter** 广受欢迎。语音功能和本地录制提升了用户体验。
- [**AI 播客遭遇杂音，用户要求更好的音质**](https://notebooklm.google.com)：**NotebookLM** 用户反映 AI 生成的播客存在质量问题，包括随机中断和奇怪的杂音。要求强化音频处理和提升稳定性的呼声日益高涨。

**主题 5. AI 伦理、审查与社区声音**

- **越狱者争论 LLM 的真正自由**：用户质疑“越狱” LLM 究竟是真正解放了模型，还是仅仅增加了更多约束。社区深入探讨了越狱背后的动机和机制。
- [**过度审查让模型失声**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**Phi-3.5** 严重的审查机制令用户感到沮丧，导致出现了幽默的模拟回复以及对模型进行“去审查”的尝试。一个[无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)已出现在 Hugging Face 上。
- [**AI 搜索引擎竞争激烈，用户仍在寻找最优选**](https://www.perplexity.ai/page/openai-challenges-google-searc-7jOvRjqsQZqm1MwfVqHMvw)：**OpenAI** 新推出的 **SearchGPT** 在实时结果方面表现平平，而 **Perplexity** 等替代方案则赢得了赞誉。AI 搜索大战正在升温，但并非所有用户都对此买账。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLM 微调与性能增强**：Neuralink 在其代码性能上实现了 **2倍加速**，在 **980 FLOPs** 下达到了 **tp=2**；同时，根据 [渗透测试基准测试论文](https://arxiv.org/abs/2410.17141) 的详细说明，**Llama 3.1** 在使用 **PentestGPT** 工具时表现优于 **GPT-4o**。
  
  - 此外，**Gemma 2B** 在特定任务上以更低的 **VRAM 占用** 超越了 **T5xxl**，讨论中还强调了由于 **弃用问题 (deprecation issues)** 导致的 **MAE 微调** 挑战。
- **AI 驱动的网络安全模型与基准测试**：一项新的基准测试 **PentestGPT** 显示，**Llama 3.1** 在自动化渗透测试中优于 **GPT-4o**。根据 [最近的研究](https://arxiv.org/abs/2410.17141)，这表明两种模型在网络安全应用中仍需改进。
  
  - 社区成员对开发用于 **恶意软件检测 (malware detection)** 的 AI 模型表现出日益增长的兴趣，并寻求实施策略方面的指导。
- **AI 开发工具与环境**：**VividNode v1.6.0** 的发布引入了对 **edge-tts** 的支持，并通过 **GPT4Free** 增强了图像生成能力，同时推出了兼容 **Ubuntu** 和 **Apple Silicon macOS** 的新 **ROS2 Docker 环境**。
  
  - [GitHub 仓库](https://github.com/koalaman/shellcheck) 中讨论的 **ShellCheck**（一种用于 shell 脚本的静态分析工具）也被强调用于提高脚本的可靠性。
- **量子计算在 AI 领域的进展**：研究人员展示了 **$^{173}$Yb** 原子中薛定谔猫态（Schrödinger-cat state）长达 **1400 秒** 的相干时间，标志着 [arXiv 论文](https://arxiv.org/abs/2410.09331v1) 中讨论的 **量子计量学 (quantum metrology)** 取得了重大突破。
  
  - 此外，据 *Nature Photonics* 报道，一种新型 **磁光存储芯片 (magneto-optic memory chip)** 设计有望降低 AI 计算的能耗。
- **AI 在医疗决策中的应用**：**Google** 推出了 *MDAgents*，这是一种用于增强 **医疗决策** 的 **LLMs** 自适应协作方案，包含 *UltraMedical* 和 *FEDKIM* 等模型，详见最近的 [Medical AI 帖子](https://huggingface.co/posts/aaditya/563565199854269)。
  
  - 这些创新旨在通过先进的 AI 模型简化医疗流程并提高诊断准确性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 加速模型微调**：**Unsloth AI** 项目通过利用 **LoRA**，使用户微调 **Mistral** 和 **Llama** 等模型速度提升近 **2倍**，并显著降低了 VRAM 消耗。
  
  - 用户讨论了在扩展规模之前迭代优化较小模型的策略，并强调了数据集大小和质量对于有效训练的重要性。
- **Python 3.11 提升跨操作系统性能**：升级到 **Python 3.11** 可以带来性能提升，在 **Linux** 上提供约 1.25 倍的速度，在 **Mac** 上为 1.2 倍，在 **Windows** 系统上为 1.12 倍，如 [推文](https://x.com/danielhanchen/status/1853535612898533715) 所述。
  
  - 讨论中也提出了对 Python 升级期间包兼容性的担忧，强调了维护软件稳定性所涉及的复杂性。
- **通过量化实现高效模型推理**：讨论集中在微调 **Qwen 2.5 72b** 等量化模型的可行性，旨在降低内存占用并在 **CPUs** 上获得满意的推理速度。
  
  - 会议指出，虽然 **模型量化 (model quantization)** 有助于轻量化部署，但初始训练仍需要大量的计算资源。
- **用于 LLM 集成的最佳 Web 框架**：对于将语言模型集成到 Web 应用程序中的建议包括用于前端开发的 **React** 和 **Svelte** 框架。
  
  - 对于倾向于使用基于 Python 解决方案来构建模型界面的开发者，建议使用 **Flask**。
- **增强 Unsloth 仓库的 Git 实践**：讨论中强调了 **Unsloth** 仓库中缺少 **.gitignore** 文件的问题，强调了在推送更改之前管理文件的重要性。
  
  - 用户分享了通过有效使用 git 命令和本地排除配置来保持干净 git 历史记录的见解。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Haiku 三版本齐发**：Anthropic 发布了 **Claude 3.5 Haiku** 的标准版、自我审查版（self-moderated）和带日期版本。在 [Claude 3.5 Overview](https://openrouter.ai/anthropic/claude-3-5-haiku) 了解更多版本信息。
  
  - 用户可以通过 [standard](https://openrouter.ai/anthropic/claude-3-5-haiku)、[dated](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022) 和 [beta releases](https://openrouter.ai/anthropic/claude-3-5-haiku:beta) 访问不同变体，方便进行无缝更新和测试。
- **免费 Llama 3.2 模型现已上线**：**Llama 3.2** 模型可通过 OpenRouter 免费使用，包含 **11B** 和 **90B** 变体，且速度有所提升。访问地址：[11B variant](https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free) 和 [90B variant](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free)。
  
  - **11B** 变体的性能达到 **900 tps**，而 **90B** 变体提供 **280 tps**，满足多样化的应用需求。
- **Hermes 405b 遭遇延迟问题**：**Hermes 405b** 的免费版本正面临严重的延迟和访问错误。
  
  - 推测这些问题可能是由于速率限制（rate limiting）或临时停机造成的，尽管部分用户仍能间歇性地收到响应。
- **API 速率限制引发用户困惑**：用户在 **ChatGPT-4o-latest** 等模型上触及速率限制，导致对 OpenRouter 上 **GPT-4o** 和 **ChatGPT-4o** 版本的区别产生困惑。
  
  - 模型名称之间的区分不明确，导致了用户体验参差不齐，以及对速率限制政策的不确定性。
- **Haiku 定价上涨引发担忧**：**Claude 3.5 Haiku** 的定价大幅上涨，引发了用户群对其负担能力的担忧。
  
  - 用户对成本增加感到沮丧，尤其是将 **Haiku** 与 **Gemini Flash** 等替代方案进行比较时，对其未来的可行性提出了质疑。

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 支持混合 GPU**：用户确认 **LM Studio** 支持 AMD 和 Nvidia GPU 混合使用，但由于依赖 **Vulkan**，性能可能会受到限制。
  
  - 为了获得最佳效果，建议使用相同的 **Nvidia** 显卡，而非混合 GPU 配置。
- **LM Studio 中的 Embedding 模型限制**：并非所有模型都适用于 Embedding；具体而言，**Gemma 2 9B** 在 **LM Studio** 中不兼容此用途。
  
  - 建议用户选择合适的 Embedding 模型以防止运行时错误。
- **LLM 结构化输出挑战**：用户在强制执行结构化输出格式时遇到困难，导致出现多余文本。
  
  - 建议包括加强 Prompt Engineering 以及利用 **Pydantic** 类来提高输出精度。
- **使用 Python 进行 LLM 集成**：讨论集中在实现代码片段以构建自定义 UI，并集成来自 **Hugging Face** 的各种语言模型功能。
  
  - 参与者强调了交替使用多个模型来处理不同任务的灵活性。
- **LM Studio 在不同硬件上的性能表现**：用户报告了在不同硬件配置下通过 **LM Studio** 运行 LLM 的各项性能指标，部分用户遇到了 Token 生成延迟。
  
  - 用户对上下文管理（context management）和硬件限制导致这些性能问题的担忧有所增加。

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 405b 模型性能**：用户报告通过 [Lambda](https://discord.com/channels/1053877538025386074/1149866623109439599/1302003794778914817) 和 [OpenRouter](https://discord.com/channels/1053877538025386074/1149866623109439599/1302003794778914817) 使用 **Hermes 405b** 模型时响应时间缓慢且出现间歇性错误，一些人甚至形容其性能“慢如冰川”。
  
  - Lambda 上的免费 API 特别表现出不稳定的可用性，导致尝试访问该模型的用户感到沮丧。
- **越狱 LLM**：一位成员质疑越狱 LLM 是否更多是为了通过创建约束来释放它们，这引发了关于此类行为的动机和影响的热烈讨论。
  
  - 一位参与者强调，许多擅长越狱的人可能并不完全理解 LLM 底层的运行机制。
- **MDAgents**：**Google** 推出了 [MDAgents](https://youtu.be/Wt5QOv1vk2U)，展示了一种旨在增强医疗决策的 **LLM 自适应协作 (Adaptive Collaboration of LLMs)**。
  
  - 本周总结该论文的播客强调了模型协作在应对复杂医疗挑战中的重要性。
- **Nous Research 模型的未来**：**Teknium** 表示 Nous Research 不会创建任何闭源模型，但针对某些特定用例的其他产品可能会保持私有或基于合同。
  
  - Hermes 系列将始终保持开源，确保其开发的透明度。
- **AI 搜索引擎性能**：用户感叹 **OpenAI** 的新搜索未能提供实时结果，特别是与 **Bing** 和 **Google** 等平台相比。
  
  - 有人指出 **Perplexity** 在搜索结果质量方面表现出色，被认为优于 Bing 和 OpenAI 的产品。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 3.5 Haiku 丰富 AI 选择**：[**Claude 3.5 Haiku**](https://docs.anthropic.com/en/docs/about-claude/models#model-names) 在 Amazon Bedrock 和 Google 的 Vertex AI 等平台上线，其新的定价结构引起了用户的担忧。
  
  - 尽管成本增加，[**Claude 3.5 Haiku**](https://x.com/alexalbert__/status/1853498517094072783) 仍保持了强大的对话能力，但其相对于现有工具的价值主张仍存争议。
- **SearchGPT：OpenAI 对传统搜索的回击**：[**SearchGPT**](https://www.perplexity.ai/page/openai-challenges-google-searc-7jOvRjqsQZqm1MwfVqHMvw) 的推出提供了一系列新的 AI 驱动搜索功能，旨在与成熟的搜索引擎竞争。
  
  - 此次发布引发了关于 AI 在信息检索中不断演变的角色以及搜索技术未来动态的讨论。
- **中国 Llama AI 模型推进军事 AI**：报告显示[**中国军方**](https://www.perplexity.ai/page/chinese-military-builds-llama-cATtO04XQQmPAEHGEmR1AQ)正在开发基于 Llama 的 AI 模型，这标志着国防 AI 应用的战略举措。
  
  - 这一进展加剧了全球关于 AI 在军事进步中的作用以及国防领域国际技术竞争的讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **梯度对偶化（Gradient Dualization）影响模型训练动力学**：讨论强调了**梯度对偶化**在训练深度学习模型中的作用，重点关注随着架构规模扩大，范数（norms）如何影响**模型性能**。
  
  - 参与者探讨了在并行和串行注意力机制之间交替的影响，引用了论文 [Preconditioned Spectral Descent for Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html)。
- **优化器参数微调反映了 Adafactor 调度**：成员们分析了 **Adafactor** 调度的变化，指出其与增加另一个超参数的相似性，并质疑这些调整背后的创新性。
  
  - 正如 [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801) 中所讨论的，最佳的 **beta2** 调度被发现与 Adafactor 现有的配置相似。
- **配置 GPT-NeoX 的最佳实践**：工程师们正在使用 **Hypster** 进行配置，并在其 **GPT-NeoX** 设置中集成 **MLFlow** 进行实验跟踪。
  
  - 一位成员询问了同时使用 **DagsHub**、**MLFlow** 和 **Hydra** 的普遍性，并引用了 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c0745fec3062328e0ab618f36334848cdf29900e/lm_eval/filters/selection.py#L56)。
- **DINOv2 通过扩展数据增强 ImageNet 预训练**：**DINOv2** 利用了 **22k** ImageNet 数据，与之前的 **1k** 数据集相比，提升了评估指标。
  
  - [DINOv2 论文](https://github.com/huggingface/transformers/blob/65753d6065e4d6e79199c923494edbf0d6248fb1/src/transformers/models/llama/modeling_llama.py#L373) 展示了增强的性能指标，强调了有效的蒸馏技术。
- **神经网络中规模（Scaling）与深度（Depth）之间的权衡**：社区讨论了神经网络架构中**深度**与**宽度**之间的平衡，分析了每个维度如何影响整体性能。
  
  - 建议进行实证验证，以证实随着模型规模扩大，网络深度与宽度之间的动力学变得越来越一致的理论。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 3.5 Haiku 在排行榜上取得进展**：**Claude 3.5 Haiku** 在 [Aider's Code Editing Leaderboard](https://aider.chat/docs/leaderboards/) 上获得了 **75%** 的评分，排名紧随之前的 **Sonnet 06/20** 模型之后。
  
  - 这标志着 Haiku 成为一种具有成本效益的替代方案，在代码编辑任务中与 Sonnet 的能力非常接近。
- **Aider v0.62.0 集成 Claude 3.5 Haiku**：最新的 **Aider v0.62.0** 版本现已全面支持 **Claude 3.5 Haiku**，可通过 `--haiku` 标志激活。
  
  - 此外，它还引入了应用来自 ChatGPT 等 Web 应用程序编辑的功能，增强了开发者的工作流。
- **全面的 AI 模型对比**：讨论强调了 **Sonnet 3.5** 的卓越质量，而 **Haiku 3.5** 被认为是一个强力但速度较慢的竞争者。
  
  - 社区成员比较了各模型的编码能力，并分享了实际应用经验。
- **基准测试性能洞察**：在 Paul G 分享的基准测试结果中，**Sonnet 3.5** 的表现优于其他模型，而 **Haiku 3.5** 在效率上稍逊一筹。
  
  - 用户对混合模型组合（如 Sonnet 和 Haiku）在处理多样化任务时的表现表现出浓厚兴趣。
- **OpenAI 战略性的 o1 泄露**：用户观察到 OpenAI 泄露 **o1** 似乎是故意的，旨在为未来的发布制造期待。
  
  - 引用了 **Sam Altman** 过去的策略，如使用 **strawberries**（草莓）和 **Orion**（猎户座星空）等意象来引发关注。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 通过 Canvas 功能增强推理能力**：**GPT-4o 的下一个版本**正在推出，引入了类似于 O1 的**高级推理能力**。此次更新包括将大文本块放入[画布风格框 (canvas-style box)](https://x.com/ArDeved/status/1852649549900242946)等功能，增强了推理任务的可用性。
  
  - 用户在获得 O1 的完整访问权限后表达了兴奋之情，并指出推理效率有了显著提高。此次升级旨在为 **AI Engineers** 简化复杂任务的执行。
- **OpenAI 的 Orion 项目引发期待**：**OpenAI 的 Orion 项目**让成员们对预计在 **2025** 年推出的 AI 创新感到兴奋。讨论强调，虽然 Orion 正在进行中，但目前的 **O1** 等模型尚未被归类为真正的 AGI。
  
  - 对话强调了 **Orion 项目的潜力**，即弥合当前能力与 AGI 之间的差距，成员们热切期待在不久的将来取得重大突破。
- **在 OpenAI 集成中采用中间件**：成员们正在探索使用**中间件产品**，将请求路由到不同的端点，而不是直接连接到 **OpenAI**。这种方法正在被讨论，以评估其规范性和有效性。
  
  - 一位成员询问使用中间件是否为标准模式，寻求社区关于将 **API endpoints** 与 OpenAI 服务集成的最佳实践见解。
- **使用分析工具增强 Prompt 测量**：讨论集中在用于跟踪用户交互（包括**挫败感水平**和任务完成率）的 **Prompt 测量工具**上。成员们建议利用**情感分析**来获得更深入的见解。
  
  - 实施者正在考虑引导 LLMs 处理对话数据，旨在通过数据驱动的调整来优化 **Prompt 有效性**并提高整体用户满意度。
- **使用 LLMs 自动化任务规划**：参与者正在寻求使用 LLMs **自动化复杂任务规划**的资源，特别是用于生成 **SQL queries** 和进行数据分析。为了实现有效的自动化，强调了需求清晰度的重要性。
  
  - 贡献者强调了模型在协助**头脑风暴会议**和简化规划流程方面的潜力，并强调了精确查询制定的重要性。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 语言配置挑战**：用户报告了在配置 **NotebookLM** 以其首选语言响应时遇到困难，尤其是在上传不同语言的文档时。说明强调 **NotebookLM** 默认为用户 [Google account](https://myaccount.google.com/age-verification) 中设置的语言。
  
  - 这引发了关于增强语言支持以适应更多样化用户群的讨论。
- **Podcast 生成质量问题**：多位用户指出了对 **NotebookLM** Podcast 质量的担忧，注意到播放过程中出现意外中断和随机声音。虽然有些人觉得这些中断很有趣，但其他人对收听体验的负面影响表示沮丧。
  
  - 这些质量问题导致了对更强大的音频处理和稳定性改进的呼吁。
- **API 开发推测**：围绕 **NotebookLM** 的 **API** 潜在开发展开了讨论，这受到 [Vicente Silveira 的一条推文](https://x.com/vicentes/status/1844202858151068087)的影响，该推文暗示了即将推出的 API 功能。
  
  - 尽管社区对此很感兴趣，但尚未发布官方公告，这为基于行业趋势的推测留下了空间。
- **音频概览功能请求**：一位用户询问如何使用 **NotebookLM** 从一份 200 页的密集 PDF 中生成多个音频概览，并考虑将文档拆分为较小部分的劳动密集型方法。建议包括提交从单一来源生成音频概览的功能请求。
  
  - 这反映了对 **NotebookLM** 中更高效摘要工具的需求，以便无缝处理大型文档。
- **特殊教育需求用例扩展**：成员们分享了为特殊需求学生使用 **NotebookLM** 的经验，并寻求用例或成功案例，以支持向 Google 的 accessibility team（无障碍团队）进行推介。还讨论了在英国建立集体的计划。
  
  - 这些努力凸显了社区利用 **NotebookLM** 增强教育无障碍性的倡议。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel Optimization**：成员们讨论了优化 **Triton kernels** 的复杂性，特别是 GPU 等待 CPU 操作的时间，并建议堆叠多个 Triton matmul 操作可以减少大矩阵的开销。
  
  - 对话强调了通过最小化 CPU-GPU 同步延迟来提高性能的策略，成员们考虑了多种增强 kernel 效率的方法。
- **FP8 Quantization Techniques**：一位用户分享了他们在 **FP8 quantization** 方法方面的经验，指出与纯 **PyTorch** 实现相比，速度提升令人惊讶，并提供了其 [GitHub repository](https://github.com/aredden/flux-fp8-api) 的链接。
  
  - 讨论强调了高效动态量化 activations 的挑战，并引用了 [flux-fp8-api](https://github.com/aredden/f8_matmul_fast) 作为实现这些技术的资源。
- **LLM Inference on ARM CPUs**：探索了在 **NVIDIA Grace** 和 **AWS Graviton** 等 ARM CPU 上进行 **LLM inference**，讨论了处理高达 **70B** 的大型模型，并参考了 [torchchat repository](https://github.com/pytorch/torchchat)。
  
  - 参与者指出，将多个配备 **Ampere Altra** 处理器的 ARM SBCs 进行集群可以产生有效的性能，特别是在利用 **tensor parallelism** 来桥接 CPU 和 GPU 能力时。
- **PyTorch H100 Optimizations**：深入讨论了 **PyTorch** 针对 **H100** 硬件的优化，确认支持 **cudnn attention**，但指出由于一些问题，在 2.5 版本中默认禁用了该功能。
  
  - 成员们分享了关于 H100 特性稳定性的不同经验，提到 **Flash Attention 3** 仍处于开发阶段，影响了性能的一致性。
- **vLLM Demos and Llama 70B Performance**：宣布了开发展示带有自定义 kernels 的 **vLLM** 实现的 **streams and blogs** 计划，并考虑创建一个 fork 仓库来增强 forward pass。
  
  - 关于提高 **Llama 70B** 在高解码负载下性能的讨论包括与 **vLLM maintainers** 的潜在合作，以及集成 **Flash Attention 3** 以获得更好的效率。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Introduces Visual PDF Support**：Claude 在 [Claude AI](https://claude.ai) 和 Anthropic API 中推出了 **visual PDF support**，使用户能够分析各种文档格式。
  
  - 此更新允许从**财务报告**和**法律文件**中提取数据，显著增强了 **user interactions**。
- **AI Search Engine Competition Intensifies**：随着 **SearchGPT** 和 **Gemini** 等新进入者挑战 Google 的主导地位，搜索引擎竞争的复苏显而易见。
  
  - [AI Search Wars 文章](https://buttondown.com/ainews/archive/ainews-the-ai-search-wars-have-begun-searchgpt/) 详细介绍了各种创新的搜索解决方案及其影响。
- **O1 Model Briefly Accessible, Then Secured**：**O1 model** 曾一度可以通过修改 URL 访问，支持图像上传和快速 inference 能力。
  
  - 在**兴奋和猜测**中， ChatGPT 的 O1 出现了，但此后访问受到了限制。
- **Entropix Achieves 7% Boost in Benchmarks**：[Entropix](https://x.com/Teknium1/status/1852315473213628613) 展示了小模型在 benchmarks 中提升了 **7 个百分点**，表明了其可扩展性。
  
  - 成员们正期待这些结果将如何影响未来的 **model developments** 和实现。
- **Open Interpreter Gains Traction Among Users**：用户对 [Open Interpreter](https://github.com/mediar-ai/screenpipe) 表现出极大的热情，考虑将其集成到 AI 任务中。
  
  - 一位成员建议将其设置好以备将来使用，并提到他们觉得自己 **out of the loop**。

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **A1111 对 Stable Diffusion 3.5 的支持**：用户讨论了 **Stable Diffusion 3.5** 是否能在 **AUTOMATIC1111** 上运行，提到虽然它是新发布的，但目前的兼容性可能有限。
  - 有人建议在 YouTube 上查找有关利用 **SD3.5** 的指南，因为其最近刚刚发布。
- **对服务器中诈骗机器人的担忧**：用户对发送链接的诈骗机器人以及服务器内缺乏直接的审核人员（moderation）表示担忧。
  - 讨论了右键报告等审核方法，但对其防止垃圾信息的有效性看法不一。
- **提示词“飘动的头发”的技巧**：一位用户寻求关于如何编写“waving hair”（飘动的头发）的提示词，而又不让 AI 误以为角色在“waving”（挥手）的建议。
  - 建议包括使用更简单的词汇如“wavey”来实现预期效果，以避免误解。
- **模型训练资源**：用户分享了使用图像和标签训练模型的见解，并对旧教程的参考价值表示不确定。
  - 提到的资源包括 **KohyaSS**，并讨论了高效的训练方法和工具。
- **Dynamic Prompts 扩展的问题**：一位用户报告在 **AUTOMATIC1111** 中使用 **Dynamic Prompts** 扩展时频繁崩溃，感到非常沮丧。
  - 对话集中在安装错误和故障排除协助上，一位用户分享了他们的经验。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 硬件下放（Hardware Lowering）支持**：成员们讨论了 **Mojo** 当前硬件下放能力的局限性，特别是无法按照 [Mojo🔥 FAQ | Modular Docs](https://docs.modular.com/mojo/faq#how-does-mojo-support-hardware-lowering.) 中所述将中间表示（intermediate representations）传递给外部编译器。
  - 建议如果需要增强硬件支持，可以 [联系 Modular](https://www.modular.com/company/contact) 以寻求潜在的 upstreaming 选项。
- **在 Mojo 中管理引用**：一位成员询问关于在容器之间存储安全引用的问题，并遇到了 struct 中生命周期（lifetimes）的问题，强调了需要量身定制设计以避免失效问题。
  - 讨论强调了在操作自定义数据结构中的元素时，确保指针稳定性（pointer stability）的重要性。
- **Slab List 实现**：成员们审查了 slab list 的实现并考虑了内存管理方面，指出将其功能与标准集合（standard collections）合并的可能性。
  - 使用内联数组（inline arrays）的概念及其对性能和内存一致性的影响对他们的设计至关重要。
- **Mojo 稳定性与 Nightly 版本**：对于完全切换到 **Mojo** Nightly 版本的稳定性存在担忧，强调了 Nightly 版本在合并到 main 分支之前可能会发生重大变化。
  - 尽管 Nightly 版本包含最新进展，成员们仍强调了稳定性的必要性，并对重大变更发布了公告（PSA）。
- **神经网络中的自定义 Tensor 结构**：探讨了在神经网络中自定义 tensor 实现的需求，揭示了诸如内存效率和设备分布等各种用例。
  - 成员们指出了这与数据结构选择的相似之处，认为对专门化数据处理的需求仍然具有现实意义。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AMD 重新推出 OLMo 语言模型**：AMD [重新推出了 OLMo 语言模型](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)，在 AI 社区引起了极大关注。
  - 成员们反应中带着怀疑和幽默，对 AMD 在语言建模领域的进展表示惊讶。
- **Grok 模型 API 发布，支持 128k 上下文**：[Grok 模型 API](https://x.ai/blog/api) 已发布，为开发者提供具有 **128,000** token 上下文长度的模型访问权限。
  - Beta 计划包含免费试用额度，被幽默地称为“25 自由币（freedom bucks）”，旨在鼓励开发者进行尝试。
- **Claude 3.5 Haiku 价格翻四倍**：[Claude 3.5 Haiku](http://anthropic.com/claude/haiku) 已发布，虽然基准测试表现有所提升，但其成本是前代产品的 **4 倍**。
  - 在市场压力推动 AI 推理成本下降的背景下，这种出人意料的价格上涨引发了种种猜测。
- **探索 AnthropicAI Token 计数 API**：成员们尝试了 [AnthropicAI Token Counting API](https://x.com/nrehiew_/status/1852701616287125624)，重点关注 Claude 的聊天模板以及图像/PDF 的数字 Token 化。
  - 随附的图片提供了该 API 能力的摘要（TL;DR），突出了其处理不同数据格式的能力。
- **中国军方利用 Llama 模型获取战争见解**：中国军方利用 Meta 的 Llama 模型分析并开发战争策略和架构，并使用公开的军事数据对其进行了微调。
  - 这种适配使该模型能够有效回答与军事事务相关的查询，展示了 Llama 的多功能性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **OilyRAGs 凭借 AI 夺得黑客松大奖**：AI 驱动的目录 **OilyRAGs** 在 LlamaIndex 黑客松中获得第三名，展示了 **Retrieval-Augmented Generation** (RAG) 在增强机械工作流方面的能力。点击[此处](https://t.co/1dBn6QYnVK)了解其实现方式，据称其效率提升了 **6000%**。
  - 该项目旨在简化机械领域的任务，展示了 RAG 的实际应用。
- **优化自定义 Agent 工作流**：关于**自定义 Agent 创建**的讨论建议绕过 Agent worker/runner，转而使用 Workflows，详见[文档](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows)。
  - 这种方法简化了 Agent 开发过程，能够更高效地集成专门的推理循环。
- **RAG 流水线的成本估算策略**：成员们讨论了使用 **OpenAIAgentRunner** 进行 **RAG 流水线成本估算**，并明确了工具调用（tool calls）与补全调用（completion calls）是分开计费的。
  - 他们强调使用 LlamaIndex 的 Token 计数工具来准确计算每条消息的平均 Token 使用量，以便更好地制定预算。
- **介绍 bb7：本地 RAG 语音聊天机器人**：**bb7** 是一款**本地 RAG 增强语音聊天机器人**，支持文档上传和上下文感知对话，且无需外部依赖。它集成了 **Text-to-Speech** (TTS) 以实现流畅交互，详见[此处](https://t.co/DPB47pEeHg)。
  - 这一创新突出了在创建用户友好、具备离线能力的聊天机器人解决方案方面的进展。
- **测试用于数据分析的轻量级 API**：展示了一个全新的**数据分析 API**，为 OpenAI Assistant 和 Code Interpreter 提供了一个**更快速、更轻量级的替代方案**，专门为数据分析和可视化而设计，详见[此处](https://reef1.netlify.app/)。
  - 它能生成 **CSV** 文件或 **HTML 图表**，强调简洁高效，无冗余细节。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Claude 命令中的 Anthropic 异常**：用户报告了在使用最新的 **Claude model** 时出现的一系列 **Anthropic errors** 和 API 问题，特别是与命令 `interpreter --model claude-3-5-sonnet-20240620` 相关的问题。
  
  - 一位用户发现了一个反复出现的 **error**，暗示这是一个影响多位用户的趋势。
- **Even Realities 增强 Open Interpreter**：一位成员推荐将 [Even Realities G1](https://www.evenrealities.com/) 眼镜作为 **Open Interpreter** 集成的潜在工具，强调了其 **open-source commitment**。
  
  - 其他人讨论了其 **hardware capabilities** 并期待未来的插件支持。
- **Oasis AI：开创全 AI 生成游戏**：团队发布了 [Oasis](https://oasis-model.github.io/)，这是首个可玩的、实时的、开放世界 AI 模型，标志着向复杂交互环境迈出了一步。
  
  - 玩家可以通过键盘输入与环境互动，展示了无需传统游戏引擎的实时游戏玩法。
- **Claude 3.5 Haiku 达到高性能**：**Claude 3.5 Haiku** 现已在包括 **Anthropic API** 和 **Google Cloud's Vertex AI** 在内的多个平台上可用，提供迄今为止最快、最智能的体验。
  
  - 该模型在各种基准测试中超越了 **Claude 3 Opus**，同时保持了成本效益，正如[这条推文](https://x.com/alexalbert__/status/1853498517094072783?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)中所分享的。
- **OpenInterpreter 为 Claude 3.5 进行优化**：一个新的 [pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1523) 引入了 **Claude Haiku 3.5** 的配置文件，由 **MikeBirdTech** 提交。
  
  - 这些更新旨在增强在 **OpenInterpreter** 项目中的集成，反映了仓库的持续开发。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 全面支持 Vision**：一位成员庆祝 [Pull Request #1495](https://github.com/stanfordnlp/dspy/pull/1495) 成功合并，该 PR 为 DSPy 增加了 **Full Vision Support**，标志着一个重要的里程碑。
  
  - *这一刻期待已久*，表达了对团队合作的赞赏。
- **Docling 文档处理工具**：一位成员介绍了 **Docling**，这是一个文档处理库，可以将各种格式转换为结构化的 JSON/Markdown 输出，用于 **DSPy** 工作流。
  
  - 他们强调了关键特性，包括对 **scanned PDFs 的 OCR 支持**，以及与 **LlamaIndex** 和 **LangChain** 的集成能力。
- **STORM 模块修改**：一位成员建议改进 **STORM module**，特别是增强对目录（table of contents）的利用。
  
  - 他们提议根据 TOC 逐节生成文章，并结合私有信息以增强输出。
- **在 Signature 中强制输出字段**：一位成员询问如何在 signature 中强制要求一个输出字段返回现有特征，而不是生成新特征。
  
  - 另一位成员通过概述一个正确返回特征作为输出一部分的函数提供了解决方案。
- **优化 Few-shot 示例**：成员们讨论了在不修改 prompts 的情况下优化 few-shot examples，重点在于提高示例质量。
  
  - 建议使用 [BootstrapFewShot](https://dspy-docs.vercel.app/deep-dive/optimizers/bootstrap-fewshot/) 或 BootstrapFewShotWithRandomSearch 优化器来实现这一目标。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **WebGPU 在 tinygrad 中的未来集成**：成员们讨论了 **WebGPU** 的就绪情况，并建议在准备就绪后立即实施。
  
  - 一位成员提到，“当 WebGPU 准备就绪时，我们可以考虑这一点”。
- **Apache TVM 特性**：**Apache TVM** 项目 [tvm.apache.org](https://tvm.apache.org) 专注于跨多种硬件平台优化机器学习模型，提供模型编译和后端优化等功能。
  
  - 成员们强调了它对 **ONNX**、**Hailo** 和 **OpenVINO** 等平台的支持。
- **在 tinygrad 中实现 MobileNetV2 的挑战**：一位用户在实现 **MobileNetV2** 时遇到了问题，特别是优化器无法正确计算梯度。
  
  - 社区参与了故障排除工作，并讨论了各种实验结果以解决该问题。
- **Fake PyTorch 后端开发**：一位成员[分享了他们的 'Fake PyTorch'](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/faketorch) 封装器，该工具利用 tinygrad 作为后端，目前支持基础功能，但缺少高级功能。
  
  - 他们征求了反馈，并对自己的开发方法表达了好奇。
- **Oasis AI 模型发布**：**Oasis** 项目宣布发布一款开源实时 AI 模型，能够利用键盘输入生成可玩的视频游戏画面和交互。他们发布了 [500M 参数模型的代码和权重](https://github.com/mdaiter/open-oasis/tree/tinygrad)，强调了其实时视频生成能力。
  
  - 正如团队成员所讨论的，未来的计划旨在进一步提升性能。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **针对特定领域问答微调 Llama 3.1**：一位成员正在寻找高质量的英文 [instruct datasets](https://) 来微调 **Meta Llama 3.1 8B**，旨在集成特定领域的问答功能，相比 LoRA 方法更倾向于全量微调。
  
  - 他们强调了通过微调实现性能最大化的目标，并指出了替代方法所面临的挑战。
- **解决微调中的灾难性遗忘问题**：一位成员报告在微调模型时遇到了 **catastrophic forgetting**（灾难性遗忘），引发了对过程稳定性的担忧。
  
  - 作为回应，另一位成员建议 **RAG** (Retrieval-Augmented Generation) 可能会为某些应用提供更有效的解决方案。
- **Granite 3.0 基准测试超越 Llama 3.1**：**Granite 3.0** 被提议作为替代方案，其基准测试结果优于 **Llama 3.1**，并包含旨在防止遗忘的微调方法论。
  
  - 此外，Granite 3.0 的 Apache 许可证也被强调，为开发者提供了更大的灵活性。
- **增强推理脚本并解决不匹配问题**：成员们发现当前的推理脚本缺乏对 chat formats 的支持，仅处理纯文本并在生成过程中添加 `begin_of_text` token。
  
  - 这一设计缺陷导致了 **mismatch with training**（与训练不匹配），引发了关于潜在改进和更新 README 文档的讨论。
- **Aloe Beta 发布：先进的医疗保健 LLMs**：**Aloe Beta** 发布，这是一套经过微调的开源医疗保健 LLMs 系列，标志着 AI 驱动的医疗解决方案的重大进展。详情请见[此处](https://www.linkedin.com/posts/ashwin-kumar-g_excited-to-introduce-aloe-a-family-of-activity-7259240192006373376-VWa7?utm_source=share&utm_medium=member_desktop)。
  
  - 开发过程涉及使用 **axolotl** 进行细致的 **SFT phase**（SFT 阶段），确保模型针对各种医疗相关任务进行了量身定制，团队成员对其潜在影响表示了极大的热情。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **32 个 CLS Tokens 增强稳定性**：引入 **32 个 CLS tokens** 显著稳定了训练过程，提高了可靠性并增强了输出结果。
  
  - 这一调整对训练一致性产生了明显影响，是模型性能提升方面值得关注的进展。
- **模糊的 Latent 下采样担忧**：一名成员质疑了**双线性插值 (bilinear interpolation)** 在 latents 下采样中的适用性，指出结果看起来很模糊。
  
  - 这一担忧凸显了当前下采样方法的潜在局限性，促使人们重新评估该技术的有效性。
- **标准化推理 API**：**Aphrodite**、**AI Horde** 和 **Koboldcpp** 的开发者已就一项标准达成一致，以协助推理集成商识别 API，从而实现无缝集成。
  
  - 新标准已在 **AI Horde** 等平台上上线，目前正在努力引入更多 API 并鼓励开发者之间的协作。
- **RAR 图像生成器刷新 FID 记录**：[RAR 图像生成器](https://yucornetto.github.io/projects/rar.html) 在 **ImageNet-256 基准测试**中实现了 **1.48 的 FID 分数**，展示了卓越的性能。
  
  - 通过利用随机性退火策略，RAR 在不增加额外成本的情况下，表现优于以往的自回归图像生成器。
- **位置嵌入打乱的复杂性**：有人提议在训练开始时*打乱位置嵌入 (position embeddings)*，并逐渐将这种效果在**结束时降低到 0%**。
  
  - 尽管具有潜力，但一名成员指出，实现这种打乱比最初预想的更复杂，反映了优化嵌入策略所面临的挑战。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents Hackathon 设置截止日期**：提醒各团队在 **11 月 4 日周一结束前**设置好 **OpenAI API** 和 **Lambda Labs 访问权限**，以免影响资源获取；可以通过[此表单](https://forms.gle/JoKbRGsnwKSjGuQy8)提交。
  
  - 尽早设置 API 可以在下周获得来自 **OpenAI** 的额度，确保在整个黑客松期间顺利参与并访问 **Lambda** 推理端点。
- **Jim Fan 展示 Project GR00T**：*Jim Fan* 将在 **PST 时间下午 3:00** 的[直播](https://www.youtube.com/live/Qhxr0uVT2zs)中展示 **Project GR00T**，这是 NVIDIA 为通用机器人 AI 大脑发起的项目。
  
  - 作为 **GEAR** 的研究负责人，他强调在各种环境下开发具有通用能力的 AI Agents。
- **黑客松组队**：寻求 LLM Agents Hackathon 合作机会的参与者可以通过[此团队报名表](https://docs.google.com/forms/d/e/1FAIpQLSdKesnu7G_7M1dR-Uhb07ubvyZxcw6_jcl8klt-HuvahZvpvA/viewform)进行申请，并概述创新型基于 LLM 的 Agents 的目标。
  
  - 该表单便于团队申请，鼓励组建专注于创建高级 LLM Agents 的小组。
- **Jim Fan 的专业知识与成就**：*Jim Fan* 博士毕业于**斯坦福视觉实验室 (Stanford Vision Lab)**，因其在机器人多模态模型和擅长 **Minecraft** 的 AI Agents 方面的研究，荣获 NeurIPS 2022 **杰出论文奖**。
  
  - 他的影响力工作曾被《**纽约时报**》和《**麻省理工科技评论**》等主流媒体报道。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **模拟 ChatGPT 的浏览机制**：研发部门的一位成员正在探索通过跟踪搜索词和结果来手动模拟 **ChatGPT 的浏览**，旨在分析 **SEO 影响**以及模型高效过滤和排序多达 **100 多个结果**的能力。
  
  - 该计划旨在将 **ChatGPT 的搜索行为**与人类搜索方法进行比较，以增强对 AI 驱动的信息检索的理解。
- **聊天机器人挑战传统浏览器**：讨论强调了 **ChatGPT 等聊天机器人**取代传统 Web 浏览器的潜力，并承认预测此类技术转变的难度。
  
  - 一位参与者引用了过去的技术预测（如 **视频通话** 的演变），以说明未来进步的不可预测性。
- **增强现实改变信息获取方式**：一位成员提出，**增强现实**可以通过提供持续更新来彻底改变信息获取，超越当前对 **无浏览器环境** 的预期。
  
  - 这一观点扩展了对话，涵盖了潜在的 **变革性技术**，强调了与信息交互的新方法。
- **Cohere API 查询重定向**：一位成员建议不要在 **Cohere API** 频道发布一般性问题，而是将其引导至适当的 API 相关查询讨论空间。
  
  - 另一位成员幽默地承认了错误，指出用户中广泛存在的 **AI 兴趣**，并承诺会妥善引导未来的问题。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 支持分布式 Checkpointing**：一位成员询问 **Torchtune** 中是否有与 DeepSpeed 的 `stage3_gather_16bit_weights_on_model_save` 等效的功能，以解决多节点微调过程中的问题。
  
  - 澄清了将此标志设置为 false 有助于 **分布式/分片 Checkpointing**，允许每个 rank 保存自己的分片。
- **Llama 90B 集成到 Torchtune**：分享了一个将 **Llama 90B** 集成到 **Torchtune** 的 Pull Request，旨在解决 Checkpointing 相关的 Bug。
  
  - [PR #1880](https://github.com/pytorch/torchtune/pull/1880) 详细说明了集成过程中与 Checkpointing 相关的增强功能。
- **澄清梯度范数裁剪 (Gradient Norm Clipping)**：成员们讨论了 **Torchtune** 中梯度范数的正确计算方法，强调需要对所有参数梯度的 L2 范数进行归约 (reduction)。
  
  - 在下一次迭代中可能会裁剪梯度范数，但这将改变原始的梯度裁剪逻辑。
- **配置中重复的 'compile' 键导致失败**：在 [llama3_1/8B_full.yaml](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full.yaml#L64-L73) 中发现了一个重复的 `compile` 键，导致运行失败。
  
  - 目前尚不确定其他配置文件是否存在类似的重复键问题。
- **ForwardKLLoss 实际上计算的是交叉熵 (Cross-Entropy)**：**Torchtune** 中的 **ForwardKLLoss** 计算的是交叉熵而非预期的 KL 散度 (KL-divergence)，需要调整预期。
  
  - 这种区别至关重要，因为由于常数项的存在，优化 KL 散度实际上等同于优化交叉熵。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Aphrodite 增加实验性 Windows 支持**：**Aphrodite** 现在处于实验阶段支持 **Windows**，实现了针对 **NVIDIA GPU** 优化的**高吞吐量本地 AI**。有关设置详情，请参阅[安装指南](https://github.com/PygmalionAI/aphrodite-engine/blob/main/docs/pages/installation/installation-windows.md)。
  
  - 此外，已确认支持 **AMD** 和 **Intel** 计算，尽管 Windows 支持仍处于**未测试**状态。感谢 **AlpinDale** 带头开展 Windows 实施工作。
- **LLM 驱动自动打补丁创新**：一篇关于[自愈代码的新博客文章](https://www.dylandavis.net/2024/11/self-healing-code/)讨论了 **LLM** 如何在**自动修复漏洞软件**中发挥关键作用，标志着 **2024** 年是这一进步的关键一年。
  
  - 作者详细阐述了解决自动打补丁的**六种方法**，其中两种已作为产品提供，四种仍处于研究阶段，突显了实际解决方案与持续创新之间的融合。
- **LLM Podcast 探讨自愈软件**：**LLM Podcast** 专题讨论了[自愈代码博客文章](https://podcasters.spotify.com/pod/show/dylan8185/episodes/Self-Healing-Code-e2qg6oc)，为**软件自愈技术**提供了对话式的见解。
  
  - 听众可以在 [Spotify](https://podcasters.spotify.com/pod/show/dylan8185/episodes/Self-Healing-Code-e2qg6oc) 和 [Apple Podcast](https://podcasts.apple.com/us/podcast/self-healing-code/id1720599341?i=1000675510141) 上收听该播客，以便以音频形式获取内容。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LoG 会议落地德里**：**Learning on Graphs (LoG) 会议**将于 **2024 年 11 月 26 日至 29 日**在**新德里**举行，重点关注**图机器学习**的进展。首届南亚分会由 **IIT Delhi** 和 **Mastercard** 主办，更多详情请点击[此处](https://sites.google.com/view/log2024delhi)。
  
  - 该会议旨在连接图学习领域的**创新思想家**和**行业专业人士**，鼓励来自**计算机科学**、**生物学**和**社会科学**等不同领域的参与。
- **LoG 2024 中的图机器学习**：LoG 2024 非常强调**图机器学习**，展示了该领域的最新**进展**。与会者可以在[官方网站](https://sites.google.com/view/log2024delhi)上找到有关会议的更多信息。
  
  - 该活动旨在促进跨学科讨论，推动**机器学习**、**生物学**和**社会科学**专家之间的合作。
- **LoG 2024 本地聚会**：**LoG 社区**正在组织一个**本地小型会议**网络，以促进不同地理区域的讨论和合作。**LoG 2024 本地聚会征集**目前已开放，旨在增强主活动期间的社交体验。
  
  - 这些本地聚会旨在将来自相似地区的参与者聚集在一起，促进社区参与和共享学习机会。
- **LoG 社区资源**：**LoG 2024** 的参与者可以加入 **Slack** 上的会议社区，并在 [Twitter](https://twitter.com/logconference) 上关注更新。此外，往届会议的录音和材料可在 [YouTube](https://www.youtube.com/playlist?list=PL2iNJC54likrwuHz_T3_JnV2AzyX5TLzc) 上获取。
  
  - 这些资源确保与会者能够持续获得宝贵材料，并在会议日期之外与社区互动。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **基准测试所需的函数定义**：一名成员正在对一种[基于检索的方法](https://discord.com/channels/1111172801899012102/1111353033352294440/1303139972945018990)进行 **function calling** 基准测试，并正在寻求一系列可用的 **functions** 及其 **definitions** 以进行索引。
  
  - 他们特别提到，将这些信息**按测试类别组织**会非常有帮助。
- **结构化集合增强基准测试**：讨论强调了对**结构化函数定义集合**的需求，以辅助 function calling 基准测试。
  
  - 按**测试类别**组织信息将提高从事类似工作的工程师的**可访问性**和**易用性**。

 

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。

---

**AI21 Labs (Jamba) Discord** 没有新消息。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1302000066126413844) (955 条消息🔥🔥🔥):

> - `关于 PDF 处理 AI 工具的讨论`
> - `各种模型和数据集的使用经验`
> - `对政治和时事的看法`
> - `微调 LLM 的挑战`
> - `为特定用例构建自定义应用程序`

- **探索 PDF 解析技术**：Maggyd 等人讨论了由于 PDF 结构复杂而导致的信息提取困难，建议使用 PDFPig 和 PDFPlumber 等工具进行解析。
  
  - Technosourceressextraordinaire 提到了 PDF 设计初衷是单向发布的挑战，并提供 Adobe 的 API 作为潜在的可访问性资源。
- **构建语言和图像处理模型**：Hotprotato 解释了 NLP 中 tokenizer 的重要性，并建议使用 BERT 和 RoBERTa 等现有模型来创建类似 Grammarly 的应用程序。
  
  - Newbie_boobie 表示有兴趣了解基于深度的模型及其在简单语法检查之外的应用。
- **政治讨论和观点**：该小组分享了对当前政治气候的看法，表达了对候选人及其能力的沮丧。
  
  - 讨论了对即将到来的选举潜在结果的担忧，引发了关于政治问责制的更广泛对话。
- **个人项目和学习路径**：几位用户分享了他们在 AI 领域的个人项目，包括构建自形成神经网络以及尝试各种现有模型。
  
  - 讨论了每位成员如何规划自己的学习旅程，强调了好奇心和实验的重要性。
- **机器学习开发的挑战**：用户反思了 AI 开发中涉及的复杂性，例如硬件限制以及对特定任务稳健模型的需求。
  
  - 社区成员分享了应对这些挑战的策略，包括利用开源工具和 API。

**提到的链接**：

- [Cat Wait Waiting Cat GIF - Cat wait Waiting cat Wait - Discover & Share GIFs](https://tenor.com/view/cat-wait-waiting-cat-wait-waiting-cat-waiting-gif-9780709586447195996)：点击查看 GIF
- [我无法获取电子邮件确认链接](https://discuss.huggingface.co/t/i-can-t-get-email-confirmation-link/13776)：今天我注册了 Hugging Face，但是无法收到电子邮件确认链接。我从邮箱复制并粘贴了我的电子邮件地址，但还是收不到。也许其他人也遇到了这种情况。我...
- [google/deplot · Hugging Face](https://huggingface.co/google/deplot)：未找到描述
- [GPTQ - Qwen](https://qwen.readthedocs.io/en/latest/quantization/gptq.html)：未找到描述
- [支持的模型](https://huggingface.co/docs/api-inference/supported-models)：未找到描述
- [minchyeom/birthday-llm · Hugging Face](https://huggingface.co/minchyeom/birthday-llm)：未找到描述
- [Joe Biden Presidential Debate GIF - Joe biden Presidential debate Huh - Discover & Share GIFs](https://tenor.com/view/joe-biden-presidential-debate-huh-confused-gif-9508832355999336631)：点击查看 GIF
- [Cat GIF - Cat - Discover & Share GIFs](https://tenor.com/view/cat-gif-6627971385754609194)：点击查看 GIF
- [The Deep Deep Thoughts GIF - The Deep Deep Thoughts Deep Thoughts With The Deep - Discover & Share GIFs](https://tenor.com/view/the-deep-deep-thoughts-deep-thoughts-with-the-deep-the-boys-gif-26372785)：点击查看 GIF

- [未找到标题](https://developer.adobe.com/document-services/docs/overview/pdf-accessibility-auto-tag-api/): 未找到描述
- [Fish Agent - 由 fishaudio 创建的 Hugging Face Space](https://huggingface.co/spaces/fishaudio/fish-agent): 未找到描述
- [Llama 3.2 3b Voice - 由 leptonai 创建的 Hugging Face Space](https://huggingface.co/spaces/leptonai/llama-3.2-3b-voice): 未找到描述
- [我赢啦 GIF - I Won Trump Donald Trump - 发现并分享 GIF](https://tenor.com/view/i-won-trump-donald-trump-gif-10540661): 点击查看 GIF
- [Drugs Bye GIF - Drugs Bye Felicia - 发现并分享 GIF](https://tenor.com/view/drugs-bye-felicia-police-call-gif-12264463): 点击查看 GIF
- [Unlimited Power Star Wars GIF - Unlimited Power Star Wars - 发现并分享 GIF](https://tenor.com/view/unlimited-power-star-wars-gif-10270127): 点击查看 GIF
- [Ladacarpiubellachece GIF - Ladacarpiubellachece - 发现并分享 GIF](https://tenor.com/view/ladacarpiubellachece-gif-25096196): 点击查看 GIF
- [Space 运行几小时后出现 ? 运行时错误](https://discuss.huggingface.co/t/space-runs-ok-for-several-hours-then-runtime-error/115638): 我有一个 Python/Gradio Space (GradioTest - 由 dlflannery 创建的 Hugging Face Space)，几个月来一直运行良好，直到 2 天前，它在运行几个小时后会出现：Runtime error Exit code: ?。重...
- [Fire Kill GIF - Fire Kill Fuego - 发现并分享 GIF](https://tenor.com/view/fire-kill-fuego-matar-incendiar-gif-5307315): 点击查看 GIF
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co): 未找到描述
- [THUDM/glm-4-voice-9b · Hugging Face](https://huggingface.co/THUDM/glm-4-voice-9b): 未找到描述
- [facebook/deit-tiny-patch16-224 · Hugging Face](https://huggingface.co/facebook/deit-tiny-patch16-224): 未找到描述
- [BE DEUTSCH! [Achtung! Germans on the rise!] | NEO MAGAZIN ROYALE mit Jan Böhmermann - ZDFneo](https://www.youtube.com/watch?v=HMQkV5cTuoY): 下方有英文描述。世界正在发疯！欧洲感到如此虚弱，以至于认为 0.3% 的难民构成了威胁，美国正打算...
- [saiydero/Cypher_valorant_BR_RVC2 at main](https://huggingface.co/saiydero/Cypher_valorant_BR_RVC2/tree/main): 未找到描述
- [Qwen/Qwen2-VL-2B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct): 未找到描述
- [从 Hub 下载文件](https://huggingface.co/docs/huggingface_hub/v0.26.2/guides/download#download-an-entire-repository): 未找到描述
- [如何从 Hugging Face 下载模型？](https://stackoverflow.com/questions/67595500/how-to-download-a-model-from-huggingface): 例如，我想在 https://huggingface.co/models 下载 bert-base-uncased，但找不到“下载”链接。还是说它不可下载？
- [sha - 概览](https://github.com/sha): sha 有 5 个可用的仓库。在 GitHub 上关注他们的代码。
- [dandelin/vilt-b32-finetuned-vqa · Hugging Face](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa): 未找到描述
- [GitHub - intel/intel-extension-for-transformers: ⚡ 在几分钟内于你喜爱的设备上构建聊天机器人；为 LLM 提供 SOTA 压缩技术；在 Intel 平台上高效运行 LLM ⚡](https://github.com/intel/intel-extension-for-transformers): ⚡ 在几分钟内于你喜爱的设备上构建聊天机器人；为 LLM 提供 SOTA 压缩技术；在 Intel 平台上高效运行 LLM ⚡ - intel/intel-extension-for-transformers
- [代码自愈 – D-Squared](https://www.dylandavis.net/2024/11/self-healing-code/): 未找到描述
- [GitHub - intel-analytics/ipex-llm: 在 Intel XPU（例如带有 iGPU 和 NPU 的本地 PC，以及 Arc、Flex 和 Max 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Mixtral, Gemma, Phi, MiniCPM, Qwen-VL, MiniCPM-V 等）；无缝集成 llama.cpp, Ollama, HuggingFace, LangChain, LlamaIndex, vLLM, GraphRAG, DeepSpeed, Axolotl 等](https://github.com/intel-analytics/ipex-llm/): 在 Intel XPU（例如带有 iGPU 和 NPU 的本地 PC，以及 Arc、Flex 和 Max 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Mixtral, Gemma, Phi, MiniCPM, Qwen-VL, MiniCPM-V 等）；无缝集成...
- [GitHub - huggingface/lerobot: 🤗 LeRobot：通过端到端学习让机器人 AI 更加触手可及](https://github.com/huggingface/lerobot): 🤗 LeRobot：通过端到端学习让机器人 AI 更加触手可及 - huggingface/lerobot
- [NVIDIA RTX 2000E Ada Generation | 专业级 GPU | pny.com](https://www.pny.com/rtx-2000e-ada-generation?iscommercial=true): 未找到描述
- [美国国债时钟：实时](https://www.usdebtclock.org/): 未找到描述

- [lerobot/examples/7_get_started_with_real_robot.md at main · huggingface/lerobot](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md): 🤗 LeRobot: 通过端到端学习让机器人 AI 更易于使用 - huggingface/lerobot
- [Meta’s Orion prototype: A promising look into augmented reality’s future](https://www.techedt.com/metas-orion-prototype-a-promising-look-into-augmented-realitys-future)): Meta 的 Orion AR 眼镜让增强现实更接近现实，将数字内容与现实世界融合，提供无缝的用户体验。
- [Buy AMD Ryzen 7 4700S Octa Core Desktop Kit Online - Micro Center India](https://microcenterindia.com/product/amd-ryzen7-4700s-octa-core-desktopkit/): 在 Micro Center India 以最优惠的价格在线购买 AMD Ryzen 7 4700S 八核桌面套件。享受免费送货，放心在线购物。立即查看评论和评分。

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1302016345759547445) (8 messages🔥):

> - `Neuralink code performance`
> - `Cyber Security AI Models`
> - `Transformers Image Preprocessor Bug`

- **Neuralink 代码性能提升**: Neuralink 报告称，代码现在比两天前快了 **2 倍**，在 **tp=2** 下达到了 **980 FLOPs**。
  
  - 他们还发现 tp 与 **1033 FLOPs** 呈线性关系，从而使处理速度提升了 **55%**。
- **渴望构建网络安全 AI 模型**: 有人请求关于构建专门用于**网络安全**的 AI 模型的建议，重点关注**恶意软件检测模型**等应用。
  
  - 该用户向社区寻求在该领域的指导和专业知识。
- **调查 Transformer 预处理器中的 Bug**: 一名成员正在调查 **Transformer 图像预处理器**中可能存在的 Bug，但尚未确定根本原因。
  
  - 他们分享了正在讨论该问题的频道链接。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1302125547739942963) (8 条消息🔥):

> - `Quantum Metrology`
> - `AI Memory Chips`
> - `Medical AI Research`
> - `Open Trusted Data Initiative`
> - `ShellCheck Tool`

- **长寿命薛定谔猫态（Schrödinger-cat State）成功演示**：研究人员实现了 **$^{173}$Yb** 原子薛定谔猫态高达 **1400 秒**的相干时间，展示了利用非经典状态在 Quantum Metrology 领域取得的突破。
  
  - 这一进展有望提升测量精度，为量子技术的应用改进铺平道路。
- **用于 AI 计算的革命性存储芯片**：一种新型磁光存储单元设计有望降低 AI 计算集群的能耗，允许在存储阵列内进行高速计算。
  
  - 根据 *Nature Photonics* 的报道，该技术旨在提供更快的处理速度，同时最大限度地降低功耗需求。
- **来自 Google 的医疗 AI 创新**：**Google** 推出了 *MDAgents*，这是一个旨在通过自适应 LLM 改善医疗决策的协作模型，最近在 Medical AI 专栏中进行了介绍。
  
  - 该文章重点介绍了 *UltraMedical* 和 *FEDKIM* 等多个新模型和框架，展示了医疗 AI 领域的快速进步。
- **大规模开放可信数据集即将发布**：令人振奋的消息，*pleiasfr* 加入了 *thealliance_ai* 并共同领导 Open Trusted Data Initiative，将为 LLM 训练提供包含 **2 万亿 token** 的多语言数据集。
  
  - 该数据集将于 **11 月 11 日**在 Hugging Face 上发布，标志着开放 AI 资源迈出了重要一步。
- **ShellCheck：增强你的 Shell 脚本**：*ShellCheck* 的 GitHub 仓库提供了一个用于 Shell 脚本的静态分析工具，拥有强大的实用程序来识别和修复 Shell 脚本中的错误。
  
  - 对于旨在提高 Shell 脚本质量和可靠性的开发者来说，该工具至关重要。

**提到的链接**：

- [Minutes-scale Schrödinger-cat state of spin-5/2 atoms](https://arxiv.org/abs/2410.09331v1)：利用非经典状态进行的 Quantum Metrology 为提高物理测量精度提供了一条充满希望的途径。薛定谔猫叠加态或纠缠态的量子效应允许测量...
- [Audio to Stems to MIDI Converter - a Hugging Face Space by eyov](https://huggingface.co/spaces/eyov/Aud2Stm2Mdi)：未找到描述
- [Room Cleaner V2 - a Hugging Face Space by Hedro](https://huggingface.co/spaces/Hedro/room_cleaner_v2)：未找到描述
- [New memory chip controlled by light and magnets could one day make AI computing less power-hungry](https://www.livescience.com/technology/artificial-intelligence/new-memory-chip-controlled-by-light-and-magnets-could-one-day-make-ai-computing-less-power-hungry)：一种受光和磁铁控制的新型超快存储器，利用光学信号和磁铁高效地处理和存储数据。
- [Tweet from Alexander Doria (@Dorialexander)](https://x.com/Dorialexander/status/1853501675610247678)：很高兴宣布 @pleiasfr 加入 @thealliance_ai 共同领导 Open Trusted Data Initiative。我们将在 11 月 11 日发布用于 LLM 训练的最大规模的多语言完全开放数据集...
- [@aaditya on Hugging Face: "Last Week in Medical AI: Top Research Papers/Models 🔥 🏅 (October 26 -…"](https://huggingface.co/posts/aaditya/563565199854269)：未找到描述
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1852685220912464066)：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024 年 10 月 26 日 - 11 月 2 日）🏅 本周医疗 AI 论文：MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making by Aut...
- [GitHub - koalaman/shellcheck: ShellCheck, a static analysis tool for shell scripts](https://github.com/koalaman/shellcheck)：ShellCheck，一个用于 Shell 脚本的静态分析工具 - koalaman/shellcheck

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1302043371505717353) (31 条消息🔥):

> - `AI-Generated Courses`
> - `Huberman Answers Bot`
> - `PubMed Scraper`
> - `New VividNode Release`
> - `ROS2 Docker Environment`

- **引人入胜的 AI 制作课程发布**：一个新平台提供为现代学习者设计的 **AI 生成课程**和测验，强调互动性和可读性。
  
  - 这些课程旨在提供针对当前学习趋势定制的强大教育体验。
- **用于快速获取信息的 Huberman Answers 机器人**：**Huberman Answers** 项目允许用户快速查找源自 HubermanLab Podcast 的健康和神经科学查询的答案。

- 该应用使用 RAG-GPT 系统，可以轻松直接从 Andrew Huberman 博士的讨论中获取见解。
- **机器人抓取 PubMed 研究论文**：已创建一个 **PubMed scraper bot**，用于方便地根据标题或关键词搜索感兴趣的生物医学出版物。
  
  - 该工具通过提供对相关论文的便捷访问来增强研究体验，并具有未来推出 AI 摘要功能的潜力。
- **VividNode v1.6.0 增强功能**：**VividNode** v1.6.0 更新引入了对 **edge-tts** 的支持，并针对 GPT4Free 的图像生成进行了改进。
  
  - 最新版本还包括针对 Prompt Engineering 问题的解决方案，并增强了工具栏设计以提高可用性。
- **发布新的 ROS2 开发环境**：一个新的仓库为 ROS2 开发提供了一个**容器化开发环境**，兼容 x86-64 Ubuntu 和 Apple Silicon macOS。
  
  - 此设置简化了机器人开发和仿真，使开发者能够更方便地使用 Docker 和 VSCode。

**提到的链接**：

- [minchyeom/birthday-llm · Hugging Face](https://huggingface.co/minchyeom/birthday-llm)：未找到描述
- [Unexex](https://unexex.tech)：为现代学习者提供引人入胜的 AI 制作课程。
- [Audio to Stems to MIDI Converter - a Hugging Face Space by eyov](https://huggingface.co/spaces/eyov/Aud2Stm2Mdi)：未找到描述
- [blog – Shwetank Kumar](https://shwetank-kumar.github.io/blog.html)：未找到描述
- [ClovenDoug/150k_keyphrases_labelled · Datasets at Hugging Face](https://huggingface.co/datasets/ClovenDoug/150k_keyphrases_labelled)：未找到描述
- [Release v1.6.0 · yjg30737/pyqt-openai](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.6.0)：更新内容：支持 edge-tts (免费 tts) - 视频预览 gpt4free - 更改获取图像生成模型的方式，在图像表中添加提供者列，修复与 Prompt Engineering 相关的错误，修复...
- [Using Llama to generate zero-fail JSON & structured text with constrained token generation](https://youtu.be/_fSphczt7_g)：厌倦了 LLM 输出无效的 JSON？学习如何使用受限 Token 生成，让任何语言模型每次都能生成完美结构化的输出...
- [Streamlit Supabase Auth Ui - a Hugging Face Space by as-cle-bert](https://huggingface.co/spaces/as-cle-bert/streamlit-supabase-auth-ui)：未找到描述
- [GitHub - NinoRisteski/HubermanAnswers: Ask Dr. Andrew Huberman questions and receive answers directly from his podcast episodes.](https://github.com/NinoRisteski/HubermanAnswers)：向 Andrew Huberman 博士提问，并直接从他的播客节目中获得回答。
- [Leading the Way for Organic Transparency in Search: Marie Seshat Landry's Vision for organicCertification in Schema.org](https://www.marielandryceo.com/2024/11/leading-way-for-organic-transparency-in.html)：未找到描述
- [GitHub - Heblin2003/Lip-Reading: lip-reading system that combines computer vision and natural language processing (NLP)](https://github.com/Heblin2003/Lip-Reading)：结合了计算机视觉和自然语言处理 (NLP) 的唇读系统。
- [GitHub - Dartvauder/NeuroSandboxWebUI: (Windows/Linux) Local WebUI with neural network models (Text, Image, Video, 3D, Audio) on python (Gradio interface). Translated on 3 languages](https://github.com/Dartvauder/NeuroSandboxWebUI)：(Windows/Linux) 基于 Python (Gradio 界面) 的本地 WebUI，包含神经网络模型（文本、图像、视频、3D、音频）。已翻译成 3 种语言。
- [GitHub - carpit680/ros2_docker_env: Docker container for Unix-based platforms for ROS2](https://github.com/carpit680/ros2_docker_env.git)：适用于类 Unix 平台的 ROS2 Docker 容器。
- [Open-Source LLM SEO Metadata Generator](https://smart-meta.vercel.app/)：未找到描述
- [GitHub - Arvind644/smartMeta](https://github.com/Arvind644/smartMeta)：通过在 GitHub 上创建账户来为 Arvind644/smartMeta 的开发做出贡献。
- [BioMedicalPapersBot - a Hugging Face Space by as-cle-bert](https://huggingface.co/spaces/as-cle-bert/BioMedicalPapersBot)：未找到描述
- [GitHub - AstraBert/BioMedicalPapersBot: A Telegram bot to retrieve PubMed papers' title, doi, authors and publication date based on general search terms or on specific publication names](https://github.com/AstraBert/BioMedicalPapersBot)：一个 Telegram 机器人，可根据通用搜索词或特定出版物名称获取 PubMed 论文的标题、DOI、作者和出版日期。

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1302318190789001328) (11 条消息🔥):

> - `Automated Penetration Testing Benchmark` (自动化渗透测试基准测试)
> - `Recording Paper Reading Sessions` (论文阅读会议录制)
> - `Structure of Paper Reading Sessions` (论文阅读会议结构)
> - `New Member Orientation` (新成员入组指南)

- **介绍一个新的自动化渗透测试基准测试**：一篇新论文指出，网络安全领域缺乏针对大语言模型（LLMs）的全面、开放、端到端的自动化渗透测试基准测试，并针对这一空白提出了创新的解决方案。值得注意的是，研究发现使用 **PentestGPT** 工具时，**Llama 3.1** 的表现优于 **GPT-4o**，凸显了有待进一步开发的领域。
  
  - 研究结果表明，目前这两种模型在渗透测试的关键领域都缺乏能力。
- **录制讨论以供未来参考**：参与者对论文讨论的录制表示感兴趣，特别是为了那些无法参加的人。一位成员表示，他们将尝试为错过的人录制他们的演示。
  
  - 另一位成员因即将到来的考试请求分享录音。
- **论文阅读会议的结构**：会议通常在周六举行，任何成员都可以在专用语音频道的 **1 小时** 会议期间演示论文。这种开放的形式允许对各种论文进行多样化的主题讨论。
  
  - 这种结构鼓励所有成员积极参与并提高可访问性。
- **新成员入组指南**：一位新成员询问了阅读小组的规则或 FAQ 部分，以便在发帖前熟悉情况。这凸显了引导新成员融入小组的准则的重要性。

**提到的链接**：[Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://arxiv.org/abs/2410.17141)：黑客攻击对网络安全构成重大威胁，每年造成数十亿美元的损失。为了减轻这些风险，通常采用道德黑客攻击（或称渗透测试）来识别漏洞...

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1302770346096722010) (1 条消息):

> - `MAE finetuning` (MAE 微调)
> - `Deprecation issues` (弃用问题)

- **关于 MAE 微调挑战的咨询**：一位成员提出了关于 **finetuning MAE** 的问题，特别提到了过程中遇到的 **deprecation issues**（弃用问题）。
  
  - 他们希望有人已经找到了这些问题的解决方案并能分享经验。
- **寻求弃用问题的解决方案**：另一位成员加入了讨论，强调了与 MAE 微调相关的 **deprecation issues**，并询问是否有人成功解决了这些问题。
  
  - 对话强调了社区见解在解决这些技术障碍方面的重要性。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1302033922619740273) (3 条消息):

> - `Reading Resources` (阅读资源)
> - `Paper Reading Methods` (论文阅读方法)

- **分享宝贵的阅读资源**：一位成员分享了各种课程资源，包括部分德语和主要为英语的幻灯片，以及一份 [源 PDF](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)。
  
  - 其他链接包括 [哈佛大学的论文阅读指南](https://www.eecs.harvard.edu/~michaelm/postscripts/ReadPaper.pdf) 和 [多伦多大学的角色扮演研讨会](https://www.cs.toronto.edu/~jacobson/images/role-playing-paper-reading-seminars.pdf)，旨在改进论文阅读策略。
- **乐于接受德语材料**：一位成员对分享的德语资源表示热忱，称他们发现德语对理解材料很有帮助。
  
  - “德语对我来说也很好”是对所提供的多语言资源的一条积极评价。
- **将分享的方法应用于个人方法**：另一位成员对分享的具体方法表示感谢，认为这些方法对开发自己的论文阅读方法很有用。
  
  - “我将利用你提供的资源来开发我自己的论文阅读方法”强调了调整和定制所分享知识的意图。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1302767550035394704) (2 条消息):

> - `Text Embeddings`
> - `Sana Performance`
> - `LLMs Comparison`

- **提升 Diffusion Model 性能**：一位成员建议，利用近期 LLMs 生成的**丰富标注文本（richly annotated text）**来创建更好的 **text embeddings**，可以显著提升 diffusion model 的性能。
  
  - 他们询问了该领域的相关论文，表明人们对优化 embeddings 的兴趣日益增长。
- **Sana vs T5xxl 性能**：另一位成员提到，虽然使用丰富标注的文本并不会极大地增强性能，但它可以在减少 **VRAM** 占用的情况下带来改进。
  
  - 他们指出，**Gemma 2B** 在某些任务上优于 **T5xxl**，同时速度更快且资源消耗更少，并引用了 [Sana 论文](https://arxiv.org/abs/2410.10629)。
- **Li-Dit 与 Text Encoder 比较**：关于 **Li-Dit** 的讨论随之展开，**Li-Dit** 使用 **Llama3** 和 **Qwen1.5** 作为 text encoders，讨论认为由于 text encoders 的尺寸差异，将其与 **T5xxl** 进行比较并不完全公平。
  
  - 分享了 [Li-Dit 论文](https://arxiv.org/abs/2406.11831)的链接，以提供关于此比较的更多背景信息。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1301999664265953290) (742 条消息🔥🔥🔥):

> - `Unsloth AI 模型微调`
> - `Python 性能改进`
> - `模型量化与推理`
> - `用于 LLM 的 Web 开发框架`
> - `Git 实践与需求管理`

- **Unsloth AI 模型微调技术**：用户讨论了在微调 Mistral 和 Llama 等模型时的挑战与成功经验，重点关注了有效训练所需的数据集大小和质量。
  
  - 提供了关于训练策略的指导，例如在扩展规模之前，先使用较小的模型进行迭代以优化性能。
- **Python 3.11 性能提升**：升级到 Python 3.11 可以带来性能提升，特别是在不同操作系统上的运行速度，这得益于静态内存分配和内联函数调用等改进。
  
  - 用户强调了对 Python 升级后包兼容性的担忧，并指出了维护软件稳定性所涉及的复杂性。
- **高效推理的模型量化**：讨论内容包括微调量化模型（如 Qwen 2.5 72b）的可行性，强调了在 CPU 上实现令人满意的推理速度的同时减少内存需求。
  
  - 有人指出，虽然量化有助于轻量化部署，但初始训练仍然需要大量的计算资源。
- **用于 LLM 集成的 Web 开发框架**：推荐了适用于集成语言模型的 Web 框架，前端开发重点关注 React 和 Svelte 等选项。
  
  - 对于更习惯于使用基于 Python 解决方案来构建模型界面的用户，建议使用 Flask。
- **Git 实践与项目管理**：关于 Git 实践的讨论让人们意识到 Unsloth 仓库中缺少 .gitignore 文件，强调了在推送更改前管理文件的重要性。
  
  - 用户分享了通过有效使用 Git 命令和本地排除来保持 Git 历史记录整洁的见解。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#Data)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing))：未找到描述
- [unsloth/SmolLM2-360M-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/SmolLM2-360M-bnb-4bit)：未找到描述
- [Installation](https://huggingface.co/docs/transformers/v4.31.0/installation#offline-mode.)：未找到描述
- [Gandalf A Wizard Is Never Later GIF - Gandalf A Wizard Is Never Later - Discover & Share GIFs](https://tenor.com/view/gandalf-a-wizard-is-never-later-gif-11324448)：点击查看 GIF
- [来自 Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1853535612898533715)：如果你还在使用 Python 3.10，请切换到 3.11！Linux 机器使用 3.11 速度快约 1.25 倍。Mac 快 1.2 倍。Windows 快 1.12 倍。Python 3.12 看起来像是针对 Windows 32 位的性能修复（谁还在用 32 位...）
- [Google Colab](https://colab.research.google.com/drive/1oC)：未找到描述
- [从上一个 Checkpoint 微调 | Unsloth 文档](https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint)：Checkpointing 允许你保存微调进度，以便你可以暂停并继续。
- [在 `DPOTrainer` 上设置 `dataset_num_proc>1` 进程似乎会阻塞 · Issue #1964 · huggingface/trl](https://github.com/huggingface/trl/issues/1964)：当我在 DPOTrainer 中仅设置 dataset_num_proc=2 进程时，初始化训练器映射数据集的步骤似乎完全暂停了，尽管我的数据集只有两个数据点且...
- [CLI 现在由 Rabbidon 正确处理 dtype 的用户输入字符串 · Pull Request #1235 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/1235)：未找到描述
- [unsloth/unsloth-cli.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py)：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi, Qwen 和 Gemma LLM - unslothai/unsloth
- [GitHub - unslothai/unsloth: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi, Qwen 和 Gemma LLM](https://github.com/unslothai/unsloth)：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi, Qwen 和 Gemma LLM - unslothai/unsloth

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1302053538028978178) (5 条消息):

> - `Unsloth 项目`
> - `Product Hunt 讨论`

- **Unsloth 实现更快的 finetuning**：**Unsloth** 项目是一个 finetuning 计划，允许用户通过 **LoRA** 进行模型微调，速度比传统方法快近 **2 倍**，同时消耗更少的 **VRAM**。
  
  - 一位用户指出，这些信息可以在项目文档中找到。
- **社区发现 Contri Buzz 的价值**：一名成员分享了 [Product Hunt 上的 Contri Buzz](https://www.producthunt.com/posts/contri-buzz) 链接，表示对这一新发布的项目感兴趣。
  
  - 另一名成员对该项目表示热烈欢迎，称：*Wow cool project. Love it!*

 

**提到的链接**：[Contri.buzz - Celebrate your open source contributors | Product Hunt](https://www.producthunt.com/posts/contri-buzz)：在你的 GitHub README.md / 网站 / App 中展示一个视觉效果精美的贡献者墙。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1302019861265776662) (182 条消息🔥🔥):

> - `Fine-tuning 策略`
> - `模型性能问题`
> - `Prompt 格式`
> - `数据集格式化`
> - `Inference 行为`

- **应对 Fine-tuning 挑战**：用户在微调模型时遇到了问题，特别是 loss 值和 prompt 格式影响了 inference 质量。
  
  - 特别讨论了 EOS tokens 及其对正确停止生成的影响。
- **Prompt 格式的重要性**：多次讨论强调了在微调模型时使用适当 prompt 格式的重要性，包括 Alpaca prompt。
  
  - 模型给出的糟糕回答通常归因于数据准备过程中的格式错误或缺失 tokens。
- **数据集质量与训练性能**：由于数据集质量问题，用户报告了参差不齐的训练结果，经常出现幻觉（hallucinations）和无限循环等问题。
  
  - 改进数据集和 prompt 设计被认为是提高模型准确性的必要步骤。
- **调整训练参数**：对话涉及对训练参数的调整，例如更改 `max_steps` 和使用 `num_train_epochs` 以优化模型训练。
  
  - 会议指出，确保正确的 epochs 数量直接影响模型提供连贯回答的能力。
- **数据集中指令的澄清**：关于训练数据集中是否需要多条指令以提高查询的清晰度和有效性，产生了一些疑问。
  
  - 讨论强调了根据所使用的特定数据集灵活使用 prompt。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing),): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=LjY75GoYUCB8): 未找到描述
- [VortexKnight7/YouTube_Transcript_Sum · Datasets at Hugging Face](https://huggingface.co/datasets/VortexKnight7/YouTube_Transcript_Sum): 未找到描述
- [KTO Trainer](https://huggingface.co/docs/trl/main/kto_trainer): 未找到描述
- [push bnb 4 bit models on the hub](https://gist.github.com/younesbelkada/89fd3984a2992fdbb408fa8e3bf44101): 在 Hub 上推送 bnb 4 bit 模型。GitHub Gist：即时分享代码、笔记和代码片段。
- [Issues · huggingface/trl](https://github.com/huggingface/trl/issues/2292)): 通过强化学习训练 Transformer 语言模型。- Issues · huggingface/trl
- [Difference in behavior between fast tokenizers and normal tokenizers regarding unicode characters in strings · Issue #1011 · huggingface/tokenizers](https://github.com/huggingface/tokenizers/issues/1011#issuecomment-1173904564)): 您好，我们最近从基于 Python 的 tokenizers 切换到了更新的 fast tokenizers，发现当输入包含 unicode（表情符号等）字符时，我们的一些代码会崩溃，而这在以前……

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/) (1 条消息):

theyruinedelise: 刚看到这个，恭喜 <:catok:1238495845549346927>

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1302881483127656489) (2 条消息):

> - `mtbench 的 Callback`
> - `关于实现的决策`
> - `Hugging Face evaluate 指标`

- **关于实现决策的不确定性**：一名成员对是否应该创建一个特定的实现表示不确定，称 **“看起来不错 —— 我们不确定是否应该做一个。”**
  
  - *关于实现的决策仍处于开放状态，尚未得出明确结论。*
- **寻求 mtbench Callback 的参考**：另一名成员询问是否有 **Callback 的参考实现**，以便在 mtbench 数据集上运行 **mtbench** 评估。
  
  - *他们特别询问是否有人可以分享 Hugging Face evaluate 指标的实现。*

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1303126304161529857) (1 条消息):

> - `Claude 3.5 Haiku`
> - `免费 Llama 3.2`
> - `PDF 功能`
> - `提高信用额度`

- **Claude 3.5 Haiku 发布**：Anthropic 发布了其最新且速度最快的模型 **Claude 3.5 Haiku**，提供标准版、自我审查版以及为了方便提供的带日期变体。查看官方发布页面：[Claude 3.5 Overview](https://openrouter.ai/anthropic/claude-3-5-haiku)。
  
  - 该模型的变体包含供用户访问更新的链接：[标准版](https://openrouter.ai/anthropic/claude-3-5-haiku)、[带日期版](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022) 和 [Beta 版本](https://openrouter.ai/anthropic/claude-3-5-haiku:beta)。
- **免费访问 Llama 3.2 模型**：强大的开源 **Llama 3.2** 模型现在提供免费端点，其 **11B** 和 **90B** 变体提供了更快的速度。用户可以通过以下链接访问模型：[11B 变体](https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free) 和 [90B 变体](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free)。
  
  - **11B** 变体的性能达到 **900 tps**，而 **90B** 变体表现出令人印象深刻的 **280 tps**。
- **聊天室新增 PDF 功能**：聊天室现在支持通过附件或粘贴直接进行 PDF 分析，扩展了用户使用 OpenRouter 模型的实用性。此功能增强了模型使用的交互性和通用性。
  
  - 成员现在可以在交流中无缝分析 PDF，增强了聊天室的协作能力。
- **提高最大信用额度购买**：用户可购买的最大信用额度已增加至 **$10,000**，为密集使用 OpenRouter 服务提供了更多灵活性。此更改旨在适应更大规模的项目和用户需求。
  
  - 此更新允许用户更有效地管理资源，确保他们有足够的信用额度进行操作。

**提到的链接**：

- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku)：Claude 3.5 Haiku 在速度、代码准确性和工具使用方面提供了增强的能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022>)：Claude 3.5 Haiku 在速度、代码准确性和工具使用方面提供了增强的能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku:beta>)：Claude 3.5 Haiku 在速度、代码准确性和工具使用方面提供了增强的能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022:beta>)：Claude 3.5 Haiku 在速度、代码准确性和工具使用方面提供了增强的能力。通过 API 运行 Claude 3.5 Haiku。
- [Llama 3.2 90B Vision Instruct - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free>)：Llama 90B Vision 模型是一个顶级的、拥有 900 亿参数的多模态模型，专为最具挑战性的视觉推理和语言任务设计。它在图像描述方面提供无与伦比的准确性...
- [Llama 3.2 11B Vision Instruct - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free>)：Llama 3.2 11B Vision 是一个拥有 110 亿参数的多模态模型，旨在处理结合视觉和文本数据的任务。通过 API 运行 Llama 3.2 11B Vision Instruct。

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1302031302010077285) (7 messages):

> - `vnc-lm Discord Bot`
> - `Freebie Alert Chrome Extension`
> - `OpenRouter API integration`
> - `Scraping and tool calling`

- **vnc-lm Discord Bot 增加 OpenRouter 支持**：开发者宣布为他们的 Discord 机器人 **vnc-lm** 增加了 **OpenRouter** 支持。该机器人最初集成了 **Ollama**，现在已支持多个 API。
  
  - 功能包括创建对话分支、优化 Prompt，以及通过 `docker compose up --build` 快速设置实现轻松的模型切换。你可以在[这里](https://github.com/jake83741/vnc-lm)查看。
- **Chrome Extension 提醒免费样品**：一名成员创建了一个名为 **Freebie Alert** 的 **Chrome extension**，当用户在 amazon.co.uk 购物时，它会通知用户可用的免费样品。
  
  - 该扩展通过与 **OpenRouter API** 交互，旨在帮助用户节省资金并免费尝试新品牌，相关内容已在 [YouTube 视频](https://www.youtube.com/watch?v=bSTAlF1R17I)中分享。
- **关于 OpenRouter 关系的提问**：一位用户询问 Freebie Alert Chrome extension 是否与 **OpenRouter** 有关。
  
  - 开发者确认该扩展确实调用了 **OpenRouter API** 来实现其功能。
- **对 Scraping 行为的担忧**：多名成员讨论了 Scraping 活动，特别是分享了关于成员使用 **OpenRouter keys** 获取免费 AI 访问权限的图片。
  
  - 这引发了对伦理使用以及这些 Key 公开访问所带来影响的担忧。

**提到的链接**：

- [GitHub - jake83741/vnc-lm: vnc-lm is a Discord bot with Ollama, OpenRouter, Mistral, Cohere, and Github Models API integration](https://github.com/jake83741/vnc-lm)：vnc-lm 是一个集成了 Ollama, OpenRouter, Mistral, Cohere 和 Github Models API 的 Discord 机器人 - jake83741/vnc-lm
- [Freebie Alert - Chrome Web Store](https://chromewebstore.google.com/detail/freebie-alert/mofblmaoeamfpdmmgdahplgekeijbaih)：在 amazon.co.uk 购物时提供免费赠品提醒

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1302005689954336873) (663 messages🔥🔥🔥):

> - `Hermes 405b issues`
> - `API rate limits`
> - `Pricing changes for Haiku`
> - `Lambda API integration`
> - `OpenRouter features`

- **Hermes 405b 的不确定性**：免费版的 **Hermes 405b** 一直存在严重问题，许多用户在尝试访问时遇到高延迟或错误。
  
  - 尽管存在这些问题，一些用户仍能断断续续地获得响应，这引发了关于这是与 Rate limiting 相关的错误还是临时停机的猜测。
- **API Rate Limits 混淆**：用户报告在多个模型上达到了 Rate limits，其中 **ChatGPT-4o-latest** 模型对组织的请求限制明显较低。
  
  - 关于模型名称仍存在困惑，特别是 OpenRouter 上 **GPT-4o** 和 **ChatGPT-4o** 版本之间的区别。
- **Haiku 的价格变动**：**Claude 3.5 Haiku** 的价格最近大幅上涨，引发了用户对其作为廉价选项未来可行性的担忧。
  
  - 用户对成本增加表示沮丧，尤其是与其他替代方案（如 **Gemini Flash**）相比时。
- **Lambda API 和 PDF 功能发布**：OpenRouter 推出了一项功能，允许用户在聊天室上传 PDF，并使用任何模型进行分析，增强了各种应用的可用性。
  
  - 这一更新引发了关于 PDF 集成如何工作以及未来增强潜力的讨论。
- **新模型的集成**：讨论中出现了 Hermes 的可能替代方案，有成员提到 **Llama 3.1 Nemotron** 为某些用户带来了不错的效果。
  
  - 在应对当前可用选项的局限性时，用户对集成其他免费模型保持着持续的兴趣。

- [Chatroom | OpenRouter](https://openrouter.ai/chat?models=meta-llama/llama-3.2-3b-instruct:free)): LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。
- [Creating and highlighting code blocks - GitHub Docs](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks): 未找到描述
- [Tweet from OpenAI Developers (@OpenAIDevs)](https://x.com/OpenAIDevs/status/1853564730872607229): 推出 Predicted Outputs——通过提供参考字符串，显著降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [Unveiling Hermes 3: The First Full-Parameter Fine-Tuned Llama 3.1 405B Model is on Lambda’s Cloud](https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud): 与 Nous Research 合作推出 Hermes 3，这是 Meta Llama 3.1 405B 模型的首个全参数微调版本。使用 Lambda 训练、微调或部署 Hermes 3。
- [Apps Using OpenAI: ChatGPT-4o](https://openrouter.ai/openai/chatgpt-4o-latest/apps): 查看正在使用 OpenAI 的应用：ChatGPT-4o - 动态模型，持续更新至 ChatGPT 中 [GPT-4o](/openai/gpt-4o) 的当前版本。旨在用于研究和评估。注：此模型...
- [Tweet from OpenRouter (@OpenRouterAI)](https://x.com/OpenRouterAI/status/1853573174849319325): Chatroom 支持 PDF 了！您现在可以在聊天室中粘贴或附加 PDF，并使用 OpenRouter 上的任何模型进行分析：
- [hermes3:8b-llama3.1-q5_K_M](https://ollama.com/library/hermes3:8b-llama3.1-q5_K_M): Hermes 3 是 Nous Research 旗舰级 Hermes 系列 LLM 的最新版本。
- [Anthropic: Claude 3.5 Sonnet (self-moderated) – Recent Activity](https://openrouter.ai/anthropic/claude-3.5-sonnet:beta/activity): 查看 Anthropic: Claude 3.5 Sonnet (self-moderated) 的近期活动和使用统计数据 - 全新的 Claude 3.5 Sonnet 以相同的价格提供了优于 Opus 的能力和快于 Sonnet 的速度...
- [Tweet from Nous Research (@NousResearch)](https://x.com/NousResearch/status/1848397863547515216): 未找到描述
- [Limits | OpenRouter](https://openrouter.ai/docs/limits): 设置模型使用限制
- [Using the Lambda Chat Completions API - Lambda Docs](https://docs.lambdalabs.com/public-cloud/lambda-chat-api/): 使用 Lambda Chat Completions API
- [Llama 3.2 3B Instruct - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.2-3b-instruct): Llama 3.2 3B 是一个拥有 30 亿参数的多语言大语言模型，针对对话生成、推理和摘要等高级自然语言处理任务进行了优化。运行 Llama 3.2 ...
- [Grok Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-beta): Grok Beta 是 xAI 的实验性语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。它是 [Grok 2](https://x) 的继任者。运行 Grok Beta ...
- [Hermes 3 405B Instruct - API, Providers, Stats](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b): Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话以及长上下文连贯性...
- [Anthropic Status](https://status.anthropic.com/): 未找到描述
- [Build software better, together](https://github.com/search?q=repo%3Avercel%2Fai%20aiobj&type=code): GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、Fork 并为超过 4.2 亿个项目做出贡献。
- [Call (using import { createOpenRouter } from '@openrouter/ai-sdk-provider';i - Pastebin.com](https://pastebin.com/9Ki6vKHX): Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [Requests | OpenRouter](https://openrouter.ai/docs/requests#images-_-multimodal-requests): 处理传入和传出的请求
- [GitHub - homebrewltd/ichigo: Llama3.1 learns to Listen](https://github.com/homebrewltd/ichigo): Llama3.1 学会倾听。通过在 GitHub 上创建账号，为 homebrewltd/ichigo 的开发做出贡献。
- [OpenRouter Status](https://status.openrouter.ai/): OpenRouter 故障历史
- [LLM Rankings | OpenRouter](https://openrouter.ai/rankings): 根据应用使用情况对语言模型进行排名和分析
- [GitHub - OpenRouterTeam/ai-sdk-provider: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of AI models through the OpenRouter chat and completion APIs.](https://github.com/OpenRouterTeam/ai-sdk-provider): 适用于 Vercel AI SDK 的 OpenRouter 提供程序，通过 OpenRouter 聊天和补全 API 支持数百个 AI 模型。- OpenRouterTeam/ai-sdk-provider
- [worldsim](https://worldsim.nousresearch.com/console): 未找到描述

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1302036139489886219) (16 messages🔥):

> - `Beta access requests`
> - `Integrations feature`
> - `Custom provider keys`

- **Beta Integration 访问请求激增**：多位用户表达了获得 **integrations 功能**访问权限的愿望，强调了对利用该能力的浓厚兴趣。
  
  - 消息中包含了对 beta 访问权限的持续请求，展示了用户之间的**高需求**和热情。
- **对 Custom Provider Keys 的期待**：几位成员明确表示有兴趣获得 **custom provider keys** 的访问权限，暗示了潜在的项目以及给团队的反馈。
  
  - 一名用户特别提到正在开发一个脚本，并请求 **custom provider beta keys** 以探索该功能。

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1302047269150789682) (166 条消息🔥🔥):

> - `LM Studio Features` (LM Studio 特性)
> - `Embedding Models` (Embedding 模型)
> - `Mixed GPU Configurations` (混合 GPU 配置)
> - `Model Output Structuring` (模型输出结构化)
> - `Python for LLM Usage` (用于 LLM 使用的 Python)

- **LM Studio 的混合 GPU 支持**：用户确认在 LM Studio 中混合使用 AMD 和 Nvidia GPU 是可能的，但由于将使用 Vulkan，性能可能会受到限制。
  
  - 为了获得最佳性能，建议使用相同的 Nvidia 显卡，而不是混合使用。
- **Embedding 模型限制**：注意到并非所有模型都适用于 Embedding；具体而言，Gemma 2 9B 被发现在 LM Studio 中不兼容此用途。
  
  - 建议用户确保选择正确的 Embedding 模型以避免错误。
- **结构化输出挑战**：用户在让模型严格遵守结构化输出格式方面面临困难，通常会导致出现多余的文本。
  
  - 建议通过 Prompt Engineering 以及可能利用 Pydantic 类作为增强输出精度的潜在解决方案。
- **使用 Python 进行 LLM 集成**：讨论集中在利用代码片段为来自 Hugging Face 的各种语言模型创建自定义 UI 或功能的可行性。
  
  - 参与者指出，可以针对不同任务交替使用多个模型，从而实现更动态的 LLM 应用。
- **在模型之间运行对话**：提出了一种通过利用相互交互的不同 Persona（人格）在两个模型之间创建对话的方法。
  
  - 社区建议表明，此功能可能需要超出默认 LM Studio 功能的额外编码和设置。

**提到的链接**：

- [imgur.com](https://imgur.com/a/QOxMyAm)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、流行的梗、有趣的 GIF、鼓舞人心的故事、病毒式视频等来提升你的精神……
- [Imgur: The magic of the Internet](https://imgur.com/a)：未找到描述
- [Quick and Dirty: Building a Private RAG Conversational Agent with LM Studio, Chroma DB, and…](https://medium.com/@mr.ghulamrasool/quick-and-dirty-building-a-private-conversational-agent-for-healthcare-a-journey-with-lm-studio-f782a56987bd)：在这个时间至关重要且数据隐私至上的时代（我在医疗保健行业工作），这是我试图突破……的尝试。
- [Tweet from Pliny the Liberator 🐉 (@elder_plinius)](https://x.com/elder_plinius/status/1852690065698250878)：🚨 越狱警报 🚨 OPENAI: PWNED ✌️😜 O1: LIBERATED ⛓️‍💥 今天没预料到这个——完整的 o1 在短暂的时间窗口内可以访问！在撰写本文时，它对我来说已经失效了，但当设置时……
- [MobileLLM - a facebook Collection](https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95)：未找到描述
- [no title found](https://medium.com/@mr.ghulamrasool/quick-and-dirty-building-a-private-conversational-agent-for-heal)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1girzia/exploring_ais_inner_alternative_thoughts_when/)：未找到描述
- [llama : switch KQ multiplication to use F32 precision by default by ggerganov · Pull Request #10015 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/10015)：参考 #10005, #9991 (comment)。需要在 Attention 中使用更高浮点范围的模型列表在不断增加，因此为了保险起见，将 KQ 乘法默认设置为 F32。
- [Run LM Studio as a service (headless) - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/headless)：LM Studio 的无 GUI 运行：在后台运行，在机器登录时启动，并按需加载模型。
- [Maxime Labonne - Create Mixtures of Experts with MergeKit](https://mlabonne.github.io/blog/posts/2024-03-28_Create_Mixture_of_Experts_with_MergeKit.html)：将多个专家组合成一个单一的 frankenMoE。
- [Buy AMD Ryzen 7 4700S Octa Core Desktop Kit Online - Micro Center India](https://microcenterindia.com/product/amd-ryzen7-4700s-octa-core-desktopkit/)：从 Micro Center India 以印度最佳价格购买 AMD Ryzen 7 4700S 八核桌面套件。享受免费送货，放心在线购物。立即查看评论和评分。
- [NVIDIA RTX 2000E Ada Generation | Professional GPU | pny.com](https://www.pny.com/rtx-2000e-ada-generation?iscommercial=true)：未找到描述

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1302013168859611209) (396 条消息🔥🔥):

> - `LM Studio Performance` (LM Studio 性能)
> - `Hardware Recommendations for LLMs` (LLM 硬件推荐)
> - `Context Management in LLM Inference` (LLM 推理中的上下文管理)
> - `High-Performance Computing with AMD CPUs` (使用 AMD CPU 进行高性能计算)
> - `System Monitoring Tools for Hardware Performance` (硬件性能系统监控工具)

- **不同硬件配置下的 LM Studio 性能**：用户讨论了在不同硬件配置上使用 LM Studio 运行 LLM 时的各种性能指标，一些人注意到某些模型的首个 Token 生成时间（Time to First Token）过长。
  
  - 例如，一位用户遇到了长达数分钟的初始 Token 延迟，引发了对上下文管理和硬件限制的担忧。
- **运行小型 LLM 的最佳硬件**：对话涉及了运行小型 LLM 模型的合适硬件，多位用户建议了平衡成本与性能的配置，例如 M4 Pro 芯片及其内存带宽。
  
  - 参与者指出，虽然 16GB RAM 可能足以应付基础任务，但更高规格的模型和更大的 RAM 容量将提供更好的长期适用性。
- **LLM 上下文处理的挑战**：重点讨论了上下文长度和管理问题，特别是在加载旧聊天记录时，这可能导致推理过程中的显著滞后。
  
  - 用户推测上下文截断可能会加剧延迟，尤其是在运行有明确要求的模型时。
- **使用 AMD CPU 进行高性能推理**：讨论包括了使用 AMD Threadripper CPU 提升推理速度的见解，详细说明了核心数如何在达到瓶颈前影响性能。
  
  - 用户分享了各自系统的经验，比较了性能指标，并讨论了最大化 LLM 推理效率所需的配置。
- **系统性能分析监控工具**：用户推荐了多种在运行 LLM 时监控系统性能的工具，包括用于记录和可视化数据的 HWINFO64。
  
  - 同时也提到了 HWMonitor，尽管有人指出更集成的监控解决方案可能对实时分析更有利。

**提到的链接**：

- [Log Visualizer](https://www.logvisualizer.app/)：未找到描述
- [Apache License 2.0 (Apache-2.0) Explained in Plain English - TLDRLegal](https://www.tldrlegal.com/license/apache-license-2-0-apache-2-0)：用通俗易懂的英语总结/解释了 Apache License 2.0 (Apache-2.0)。
- [Apple introduces M4 Pro and M4 Max](https://www.apple.com/uk/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)：Apple 发布了 M4 Pro 和 M4 Max，这两款新芯片与 M4 一起，为 Mac 带来了更高能效的性能和先进功能。
- [Doja Cat GIF - Doja Cat Star - Discover & Share GIFs](https://tenor.com/view/doja-cat-star-wars-gif-25078126)：点击查看 GIF
- [Reddit - Dive into anything](https://www.reddit.com/r/sffpc/comments/1cdml5y/log_visualizer_an_app_to_visualize_and_analyze/?rdt=39574)：未找到描述
- [GitHub - openlit/openlit: OpenLIT: Complete Observability and Evals for the Entire GenAI Stack, from LLMs to GPUs. Improve your LLM apps from playground to production 📈. Supports 20+ monitoring integrations like OpenAI & LangChain. Collect and Send GPU performance, costs, tokens, user activity, LLM traces and metrics to any OpenTelemetry endpoint in just one line of code.](https://github.com/openlit/openlit)：OpenLIT：针对整个 GenAI 技术栈（从 LLM 到 GPU）的完整可观测性和评估工具。提升您的 LLM 应用，从实验阶段到生产环境 📈。支持 20 多种监控集成，如 OpenAI 和 LangChain。只需一行代码，即可收集并发送 GPU 性能、成本、Token、用户活动、LLM 追踪和指标到任何 OpenTelemetry 端点。

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1302003794778914817) (438 条消息🔥🔥🔥):

> - `Hermes 405b 模型性能`
> - `Hermes 模型的数据源`
> - `Lambda API 问题`
> - `Nous Research 模型的未来`
> - `针对小型模型的超参数优化`

- **Hermes 405b 模型性能**：用户报告说，通过 Lambda 和 OpenRouter 使用 **Hermes 405b** 模型时，响应时间较慢且出现间歇性错误，有人形容其性能慢得像“冰川移动”一样。
  
  - Lambda 上的免费 API 尤其表现出可用性不稳定的情况，让尝试访问该模型的用户感到沮丧。
- **Hermes 模型的数据源**：讨论提到，用于训练 **Hermes 405b** 的数据集可能尚未公开，尽管一些与 Open Hermes 相关的数据集可以通过 Hugging Face 获取。
  
  - 用户表示有兴趣了解模型训练背后的原始数据集和微调（tuning）过程。
- **Lambda API 问题**：**Hermes 405b** 模型的 Lambda API 经常宕机或请求失败，特别是在午夜到凌晨 3 点之间。
  
  - 一些用户建议，访问 Hermes 的 Lambda API 的可靠性需要改进。
- **Nous Research 模型的未来**：**Teknium** 表示 Nous Research 不会创建任何闭源模型，但针对某些特定用例，其他一些产品可能会保持私有或基于合同提供。
  
  - Hermes 系列将始终保持开源，确保其开发过程的透明度。
- **针对小型模型的超参数优化**：讨论围绕使用 **Hermes** 数据集训练小型模型的最佳方法展开，建议在完整数据集上进行超参数优化（Hyperparameter Optimization），而不是进行子采样。
  
  - 建议在微调小型模型时使用较低的学习率和更多的训练步数，以获得更好的性能。

**提到的链接**：

- [Introduction - Ordinal Theory Handbook](https://docs.ordinals.com/)：未找到描述
- [The Network State: How to Start a New Country](https://thenetworkstate.com/)：这本书解释了如何建立民族国家的继任者，我们称之为“网络国家”的概念。
- [Digital Matter Theory | Digital Matter Theory](https://digital-matter-theory.gitbook.io/digital-matter-theory)：数字物质的新时代
- [W3Schools.com](https://www.w3schools.com/c/index.php)：W3Schools 提供所有主要 Web 语言的免费在线教程、参考资料和练习。涵盖了 HTML, CSS, JavaScript, Python, SQL, Java 等热门主题。
- [It Was The Aliens Im Not Saying It Was Aliens GIF - It was the aliens Im not saying it was aliens Ancient aliens - Discover & Share GIFs](https://tenor.com/view/it-was-the-aliens-im-not-saying-it-was-aliens-ancient-aliens-gif-14839810013080040984)：点击查看 GIF
- [Tweet from Chubby♨️ (@kimmonismus)](https://x.com/kimmonismus/status/1853377333903741387?s=46)：Hertz-dev：一个开源、首创的全双工对话音频基础模型。它是一个拥有 8.5B 参数的 Transformer，在 2000 万小时的高质量音频数据上训练而成。
- [NousResearch (NousResearch)](https://huggingface.co/NousResearch)：未找到描述
- [teknium (Teknium)](https://huggingface.co/teknium)：未找到描述
- [recursive_library/README.md at main · cypherpunklab/recursive_library](https://github.com/cypherpunklab/recursive_library/blob/main/README.md)：通过在 GitHub 上创建账号来为 cypherpunklab/recursive_library 的开发做出贡献。
- [Self Healing Code – D-Squared](https://www.dylandavis.net/2024/11/self-healing-code/)：未找到描述
- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1302070434577125449) (22 messages🔥):

> - `Jailbreaking LLMs`
> - `AI Search Engine Performance`
> - `AI Energy Usage and Hardware Inefficiency`

- **关于 Jailbreaking LLMs 的多元观点**：一位成员质疑 Jailbreaking LLMs 是否更多是为了通过创建约束来释放它们，这引发了关于此类行为的动机和影响的热烈讨论。
  
  - 一位参与者强调，许多擅长 Jailbreaking 的人可能并不完全理解 LLM 底层运行的 Mechanics。
- **对 AI 搜索引擎结果的沮丧**：用户抱怨 OpenAI 的新搜索功能没有提供实时结果，特别是与 Bing 和 Google 等平台相比。
  
  - 有人指出 Perplexity 在搜索结果质量方面表现出色，可以被认为优于 Bing 和 OpenAI 的产品。
- **关于 AI 能源消耗的资源**：一位成员寻求有关 AI 能源使用和低效的资源，并指出这是 AI 技术批评者争论的焦点。
  
  - 分享的相关资源包括一篇[关于 Generative AI 环境成本的研究论文](https://arxiv.org/abs/2311.16863)和一段[讨论 AI 能源需求的 YouTube 视频](https://www.youtube.com/watch?v=0ZraZPFVr-U)。

**提到的链接**：

- [Power Hungry Processing: Watts Driving the Cost of AI Deployment?](https://arxiv.org/abs/2311.16863)：近年来，基于 Generative、多用途 AI 系统的商业 AI 产品激增，这些系统承诺采用统一的方法将 Machine Learning (ML) 模型构建到技术中...
- [How much energy AI really needs. And why that's not its main problem.](https://www.youtube.com/watch?v=0ZraZPFVr-U)：在 Brilliant 上了解更多关于 Neural Nets 的内容！使用我们的链接前 30 天免费，年度高级订阅可享受 20% 的折扣 ➜ https://brilliant.org/...

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1302157780186173512) (2 messages):

> - `MDAgents`
> - `Matchmaker`
> - `UltraMedical`
> - `AI in Healthcare Ethics`
> - `Clinical Evidence Synthesis`

- **MDAgents 在医疗决策中处于领先地位**：**Google** 推出了 [MDAgents](https://youtu.be/Wt5QOv1vk2U)，展示了旨在增强医疗决策的 **Adaptive Collaboration of LLMs**。
  
  - 本周总结该论文的播客强调了模型间 Collaboration 在应对复杂医疗挑战中的重要性。
- **多样化的医疗 LLMs 正在兴起**：重点介绍了几个模型，包括用于 Schema Matching 的 **Matchmaker** 和专为生物医学需求设计的 **UltraMedical**。
  
  - 像 **ZALM3** 这样的模型促进了 Vision-Language 医疗对话，凸显了能力的快速演进。
- **创新的医疗 AI 框架**：值得关注的方法论正在受到关注，例如用于 Federated Medical Knowledge 的 **FEDKIM** 和用于灵活模态组合的 **Flex-MoE**。
  
  - 这些框架旨在提高医疗 AI 解决方案在不同临床环境中的适应性。
- **Multi-Modal 应用展现出前景**：如用于痴呆症诊断的 **DiaMond** 和用于健康数据填补的 **LLM-Forest** 等应用展示了 LLM 潜在应用的广度。
  
  - **Clinical Evidence Synthesis** 使用 LLMs 的出现，突显了它们在高效合成临床见解方面的效用。
- **医疗领域的 Ethical AI 至关重要**：关于 AI Ethics 的讨论强调了 LLMs 在医学教育和生成医学考试题目中的作用。
  
  - 整合临床知识图谱体现了在医疗保健领域部署 AI 技术时对伦理考虑日益增长的需求。

 

**提到的链接**：[来自 Open Life Science AI (@OpenlifesciAI) 的推文](https://x.com/OpenlifesciAI/status/1852685220912464066)：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年10月26日 - 11月2日）🏅 本周医疗 AI 论文：MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making by Aut...

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1302157780186173512) (2 条消息):

> - `MDAgents`
> - `Matchmaker Model`
> - `FEDKIM Framework`
> - `DiaMond Application`
> - `AI in Healthcare Ethics`

- **Google 的 MDAgents 彻底改变了医疗决策**：**Google** 推出了 **MDAgents**，这是一种专为增强医疗决策而设计的 LLMs 自适应协作系统，展示于 2024 年 10 月 26 日至 11 月 2 日这一周。
  
  - 该模型强调医疗保健中的**协作式 AI** 方法，提供了改进的患者护理能力。
- **Matchmaker 使用 LLMs 对齐模式**：**Matchmaker** 利用 LLMs 在生物医学数据中进行有效的模式匹配 (schema matching)，增强了医疗记录和系统的互操作性。
  
  - 该模型旨在简化不同医疗保健数据集的集成，促进更好的数据可访问性。
- **FEDKIM 引入了联邦知识注入**：**FEDKIM** 框架专注于联邦医疗知识注入，在保护数据隐私的同时，仍能从协作学习中获益。
  
  - 这种创新方法促进了机构间的知识共享，且不会损害患者的数据安全。
- **DiaMond 应对多模态痴呆症诊断**：**DiaMond** 应用利用多模态 (multi-modal) 输入进行全面的痴呆症诊断，整合了各种数据类型以进行准确评估。
  
  - 它突显了先进的 LLM 应用如何显著增强该领域的诊断能力。
- **医疗保健中的 AI 伦理受到关注**：最近的讨论集中在**医疗保健中的 AI 伦理**，特别是 LLMs 在医学教育和临床决策支持中的作用。
  
  - 这些讨论包括围绕**医学考试题目生成**的考量，以及 AI 集成到临床环境中的伦理影响。

 

**提到的链接**：[来自 Open Life Science AI (@OpenlifesciAI) 的推文](https://x.com/OpenlifesciAI/status/1852685220912464066)：医学 AI 上周回顾：顶级研究论文/模型 🏅（2024 年 10 月 26 日 - 11 月 2 日）🏅 本周医学 AI 论文：MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making by Aut...

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1302009493223243786) (329 条消息🔥🔥):

> - `Perplexity AI Models`
> - `Claude 3.5 Haiku`
> - `Model Switching`
> - `New Subscription Pricing`
> - `User Experience with Perplexity`

- **使用中的 Perplexity AI 模型**：用户报告在不同的 AI 模型（如 Claude、Grok 和 Sonar）之间切换，以获得不同的结果。许多人发现每个模型都更适合特定的任务，如编程或对话。
  
  - Spaces 的使用允许一些用户为特定关注领域设置定制模型，尽管许多人仍然希望能够快速切换模型。
- **Claude 3.5 Haiku 的推出**：新模型 Claude 3.5 Haiku 已宣布在多个平台上可用，但面临用户对其价格上涨的担忧。许多用户对 AI 模型价格上涨表示失望，因为他们正在寻找更实惠的选择。
  
  - 一些用户强调 Claude 在成本增加的情况下仍保持良好的对话能力，而另一些人则怀疑其与现有工具相比的价值。
- **对 Perplexity 订阅模式的担忧**：用户对 Perplexity 的订阅定价及其相对于免费版本的感知价值提出了担忧。用户指出，当服务效果不如预期时（特别是在模型行为和来源可靠性方面），他们会感到沮丧。
  
  - 讨论还涉及付费 AI 模型在匹配查询时有时仍会提供不准确的来源。
- **Perplexity 的用户体验**：用户对 Perplexity 的总体情绪褒贬不一，一些人对软件当前的功能表示不满，而另一些人则发现了卓越的使用案例。用户经常质疑切换模型的有效性或响应中提供的来源的相关性。
  
  - 一些用户反馈，如果 Perplexity 能将更好的模型切换功能直接集成到主界面，而不是需要通过设置进行导航，体验将得到显著提升。
- **支持工人权利**：讨论包括声援为公平工资而罢工的工人，反映了反对压制员工权利的企业行为的广泛情绪。用户对旨在破坏工会争取改善工资努力的管理策略表示失望。
  
  - 支持工人争取公平报酬的重要性在参与对话的几位成员中引起了共鸣。

**提到的链接**：

- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1853498272863691125)：Claude 3 Haiku 仍可用于受益于图像输入或较低价格点的用例。https://docs.anthropic.com/en/docs/about-claude/models#model-names
- [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1853498517094072783)：Claude 3.5 Haiku 现已在 Anthropic API、Amazon Bedrock 和 Google Cloud 的 Vertex AI 上可用。Claude 3.5 Haiku 是我们迄今为止最快、最智能且最具成本效益的模型。这里是...
- [Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1852436101325074626?s=61)：每一秒都很重要。
- [Self Healing Code – D-Squared](https://www.dylandavis.net/2024/11/self-healing-code/)：未找到描述
- [Rolls Royce Royce GIF - Rolls royce Rolls Royce - Discover & Share GIFs](https://tenor.com/view/rolls-royce-rolls-royce-entry-gif-16920496844029391358)：点击查看 GIF
- [Perplexity Supply](https://perplexity.supply)：好奇心与品质的碰撞。我们的精品系列为好奇者提供精心设计的服饰。从重磅棉质基础款到刺绣单品，每一件都体现了我们的...
- [Perplexity & Claude 3.5 Sonnet](https://photos.app.goo.gl/DhrnzxcuM2oWi7wL9)：共享相册中新增了 2 个项目

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1302109034597449768) (26 messages🔥):

> - `能够闻到气味的 AI`
> - `SearchGPT`
> - `中国军方的 Llama AI 模型`
> - `印度价格实惠的汽车`
> - `打包技巧`

- **能闻到气味的 AI 带来新感官**：一篇详细的文章讨论了[用于数字气味的 AI](https://www.perplexity.ai/page/ai-for-digital-scents-uk4BCAPCQwqJOjYT2CKhYQ)，它可以复制嗅觉，可能改变感官体验。
  
  - 这可能会在虚拟现实和烹饪体验等领域带来创新应用。
- **OpenAI 凭借 SearchGPT 挑战 Google**：[SearchGPT](https://www.perplexity.ai/page/openai-challenges-google-searc-7jOvRjqsQZqm1MwfVqHMvw) 的推出为用户提供了一系列旨在与传统搜索引擎竞争的新搜索功能。
  
  - 这引发了关于信息检索的未来以及 AI 在搜索能力中作用的讨论。
- **中国军方开发基于 Llama 的 AI**：一篇文章披露[中国军方正在构建一个基于 Llama 的 AI 模型](https://www.perplexity.ai/page/chinese-military-builds-llama-cATtO04XQQmPAEHGEmR1AQ)，标志着 AI 应用的重大进展。
  
  - 这引发了对国际军事技术竞争以及 AI 在战争中影响的思考。
- **印度积压的库存车提供优惠**：一份报告展示了印度未售出的汽车库存如何导致车辆**价格大幅下降**，使消费者更容易购买。
  
  - 文章讨论了市场如何应对这种过剩及其对汽车行业的潜在影响。
- **分享必备打包技巧**：关于 [5-4-3-2-1 打包法](https://www.perplexity.ai/page/5-4-3-2-1-packing-method-QSgaNuf6SmyRBYx.spOefA) 的指南为旅行者提供了简化打包过程的实用技巧。
  
  - 该方法鼓励高效打包，并减轻旅行准备的压力。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/Q6SNm7UepBc)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1302186945077837874) (7 messages):

> - `家电评论分析`
> - `API 中的引用功能`
> - `Discord 机器人开发`
> - `API 问题的支持联系方式`

- **家电评论分析咨询**：一位用户正在探索使用 Perplexity API 创建一个分析来自 Reddit 或 Quora 等论坛的真实家电评论网站的可能性。
  
  - 另一位用户回应说，他们认为没有理由行不通，对该项目表示乐观。
- **关于 API 引用功能的疑问**：一位成员询问是否可以从 API 获取引用（citations）数据，特别是关于 sonar 在线模型的数据。
  
  - 随后的消息提到查看置顶消息以获取有关引用功能的潜在回复或信息。
- **Discord 机器人开发计划**：一位用户计划开发一个连接到 API 的 Discord 机器人，以在进一步开发之前确保功能正常。
  
  - 他们表明了掌握 API 能力的积极态度。
- **对引用功能访问权限的担忧**：一位用户表达了挫败感，因为尽管几周前填写了高级访问权限表单，但仍未收到关于引用功能访问请求的反馈。
  
  - 他们还询问了如何与支持部门建立直接联系以获得更及时的帮助。

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1302006356336840705) (15 条消息🔥):

> - `Attention Mechanisms`
> - `Generative AI Learning`
> - `Understanding ODE/SDE for Diffusion Models`

- **修改 Attention Mechanisms 的挑战**：一位成员讨论了仅修改模型**最后一层**的 attention 可能并非最优选择，因为 attention heads 主要关注自身或附近的强 tokens。
  
  - 他们进一步指出，来自前一层的 **skip connection** 增加了额外的 **context**，这可能会削弱仅在最后一层所做修改的效果。
- **Attention Weights 的探索**：在一次详细讨论中，成员们考虑在 attention scores 的 **softmax** 之前修改 attention weights，以控制强调哪些 tokens，尽管存在这些值需要总和为 1 的风险。
  
  - 一位参与者提供了[其代码链接](https://github.com/huggingface/transformers/blob/65753d6065e4d6e79199c923494edbf0d6248fb1/src/transformers/models/llama/modeling_llama.py#L373)，展示了进行此类注入的位置。
- **共同学习 Generative AI**：一位成员向任何有兴趣从零开始学习 **generative AI** 的人发出邀请，希望大家共同努力。
  
  - 这一公开号召表明了对社区参与和共享学习经验的渴望。
- **寻求 Diffusion Models 资源**：一位成员请求推荐能阐明 **diffusion models** 的 **ODE/SDE** 公式的书籍、博客或论文，特别是 Karras 的 EDM 论文中所描述的内容。
  
  - 他们表示对 sliced score matching 和 latent diffusion 有信心，但提到对 SDEs 和 probability flow ODEs 缺乏了解。
- **学习资源咨询**：针对关于 ODE/SDE 资源的咨询，一位成员建议查看 Yang Song 的博客文章以获取有价值的见解。
  
  - 这一建议体现了社区在分享知识和学习资源方面的协作性质。

**提及的链接**：[transformers/src/transformers/models/llama/modeling_llama.py at 65753d6065e4d6e79199c923494edbf0d6248fb1 · huggingface/transformers](https://github.com/huggingface/transformers/blob/65753d6065e4d6e79199c923494edbf0d6248fb1/src/transformers/models/llama/modeling_llama.py#L373)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1302003427714400287) (323 条消息🔥🔥):

> - `Gradient Dualization`
> - `Mathematical Foundations of ML`
> - `Reading Group Formats`
> - `Optimizer Configurations`
> - `Scaling in Neural Networks`

- **探索 Gradient Dualization**：围绕 **gradient dualization** 的概念展开了讨论，强调了其对训练深度学习模型的潜在影响。分享了关于 **norms** 与模型性能随 **scaling** 动态变化之间关系的见解。
  
  - 参与者对层配置的变化（如在并行和串行 **attention mechanisms** 之间切换）如何影响神经架构的整体性能表示了兴趣。
- **策划数学读书会**：关于重启专注于 **gradient dualization** 的数学读书会的提议受到了积极响应。讨论的形式包括集体阅读论文或鼓励预读，以促进更具参与感和深度的讨论。
  
  - 讨论中强调了保持持续参与的挑战，以及为可能不熟悉材料的人总结论文的重要性。
- **Optimizer 参数设置**：就 **optimizer** 的标准参数设置进行了交流，特别是关于 **Adam** 变体中的 **beta values**。对话强调了参与者所经历的设置差异，并引用了一篇倡导特定配置的论文。
- **神经网络的 Scaling 与深度**：参与者讨论了神经网络配置中深度与宽度之间的权衡及其对性能的影响。有人建议通过实证验证以下理论：随着模型增长，这些参数之间的动态关系会变得更加一致。
- **数学领域的社区认可**：贡献者的数学专业知识得到了认可，并提到了社区内的知名人物。随意的比较引起了人们对资深研究人员存在的关注，鼓励了分享知识和相互支持的文化。

**提及的链接**：

- [Preconditioned Spectral Descent for Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html)：未找到描述

- [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)：基于动量的随机梯度下降（SGD）加速在深度学习中被广泛使用。我们提出了准双曲动量算法（QHM），这是对动量算法的一种极其简单的改造...
- [Scaling Optimal LR Across Token Horizons](https://arxiv.org/abs/2409.19913)：最先进的 LLM 由缩放驱动——包括模型规模、数据集规模和集群规模的缩放。对于最大规模的运行，进行广泛的超参数调优在经济上是不可行的。相反，通过...
- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)：我们提出了权重归一化（Weight Normalization）：一种神经网络权重向量的重参数化方法，它将这些权重向量的长度与其方向解耦。通过对权重进行重参数化...
- [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)：深度学习研究的一个长期目标是精确表征训练和泛化。然而，神经网络通常复杂的损失景观使得学习理论...
- [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)：训练越来越大的神经网络的推动力促使了对大网络宽度下的初始化和训练的研究。一个关键挑战是如何缩放训练，使得网络的内部表示...
- [Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813)：为了提高当代深度学习的性能，人们有兴趣在层数和层大小方面扩展神经网络。当增加单个层的宽度时...
- [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013)：我们介绍了 1) 加速和 2) 稳定大规模视觉语言模型训练的新方法。1) 在加速方面，我们引入了 SwitchBack，这是一个用于 int8 量化训练的线性层，它...
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)：深度学习优化器通常是通过凸理论和近似二阶理论的混合来激发的。我们选择了三种此类方法——Adam、Shampoo 和 Prodigy——并认为每种方法实际上可以...
- [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero...](https://openreview.net/forum?id=hrOlBgHsMI)：LLM 通常使用学习率（LR）预热训练，然后余弦衰减至最大值的 10%（10 倍衰减）。在一项大规模实证研究中，我们展示了在最优最大 LR 下，一个...
- [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906)：深度 Transformer 的一个简单设计方案是组合相同的构建块。但标准的 Transformer 块远非简单，它将 Attention 和 MLP 子块与跳跃连接（skip connections）交织在一起...
- [How Does Critical Batch Size Scale in Pre-training?](https://arxiv.org/abs/2410.21676)：在给定资源下训练大规模模型需要仔细设计并行策略。特别是临界批次大小（Critical Batch Size）的效率概念，涉及时间与...之间的折衷。
- [Tweet from Proxy (@whatisproxy)](https://x.com/whatisproxy/status/1852727395696115812?s=46)：🧵 Proxy 结构化引擎（PSE）介绍一种用于 LLM 结构化输出的新型采样方法：Proxy 结构化引擎。这是一个面向 AI/ML 工程师、研究人员和...的技术推文。
- [Repulsive Surfaces](https://arxiv.org/abs/2107.01664)：惩罚表面弯曲或拉伸的泛函在几何和科学计算中起着关键作用，但迄今为止忽略了一个非常基本的要求：在许多情况下，表面必须...
- [Oasis](https://oasis.us.decart.ai/welcome)：未找到描述
- [Our First Generalist Policy](https://www.physicalintelligence.company/blog/pi0)：Physical Intelligence 正在将通用 AI 引入物理世界。
- [Condition number - Wikipedia](https://en.m.wikipedia.org/wiki/Condition_number)：未找到描述
- [Why is the Frobenius norm of a matrix greater than or equal to the spectral norm?](https://math.stackexchange.com/questions/252819/why-is-the-frobenius-norm-of-a-matrix-greater-than-or-equal-to-the-spectral-norm)：如何在不使用 $ \|A\|_2^2 := \lambda_{\max}(A^TA) $ 的情况下证明 $ \|A\|_2 \le \|A\|_F $？2-范数小于或等于 Frobenius 范数是有道理的，但我...
- [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)：优化理论中的一个旧观点认为，由于梯度是一个对偶向量，因此在将其映射到权重所在的原始空间之前，不能从权重中减去它。我们...

- [对偶结构梯度下降算法：神经网络的分析与应用](https://arxiv.org/abs/1708.00523)：机器学习模型的训练通常使用某种形式的梯度下降进行，并往往取得巨大成功。然而，一阶优化算法的非渐近分析...

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1302738446401671210) (2 messages):

> - `Adafactor 调度策略变更`
> - `超参数调优见解`

- **Adafactor beta 调度策略分析**：一位成员指出，**Adafactor** 调度策略的更改让人联想到只是简单地增加了另一个超参数，并指出 2018 年的 beta2 调度策略和 2017 年的 cycle LR 几乎没有什么突破性。
  
  - 他们强调，只有当这些更改受到理论见解或实际结果指导时才具有重要意义，并暗示与最佳配置相比，改进微乎其微。
- **质疑超参数扩展的价值**：讨论转向了扩展超参数的价值，一位参与者指出，理论上最优的 **beta2** 调度策略并没有比简单地调整固定 beta2 产生更好的训练损失（training loss）。
  
  - 他们观察到这种最优调度策略与 **Adafactor** 现有的 beta2 调度策略非常相似，从而对这些调整的创新性表示怀疑。

 

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1303144212593639438) (1 messages):

> - `w2s 模型文件`
> - `GitHub 仓库`

- **关于 w2s 模型文件的查询**：一位用户询问 **w2s** 项目的 **model files** 是否存储在可以访问的地方。
  
  - 这个问题突显了社区对 **w2s** 开发相关资源的持续关注。
- **GitHub 仓库概览**：**w2s** 项目可以在 [GitHub - EleutherAI/w2s](https://github.com/EleutherAI/w2s) 找到，贡献者可以创建账号来协助开发。
  
  - GitHub 页面作为 **w2s** 倡议集体贡献的中心。

 

**提及的链接**：[GitHub - EleutherAI/w2s](https://github.com/EleutherAI/w2s/tree/main)：通过在 GitHub 上创建账号来为 EleutherAI/w2s 的开发做出贡献。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1302206145183875102) (8 messages🔥):

> - `GPT-NeoX 实验`
> - `使用 Harness 评估任务`
> - `修改评估指标`

- **GPT-NeoX 配置的最佳实践**：用户正在探索 **GPT-NeoX** 的配置，并考虑将 **Hypster** 用于配置，同时使用 **MLFlow** 跟踪实验。
  
  - 一位成员询问涉及 DagsHub、MLFlow 和 Hydra 的设置是否常见，寻求社区的建议。
- **使用** `pass@k` **控制评估尝试次数**：一位成员询问在使用 harness 时如何控制 **GSM8K** 等任务中的尝试次数 (k)，得到的回复是在任务模板中使用 **repeats** 可以实现这一点。
  
  - 引用了一个使用 **repeats** 的示例来澄清评估 harness 中的这一功能。
- **响应检查方法论**：讨论了 harness 是否会自动根据正确答案检查所有 k 个响应，并在其中任何一个正确时返回正确。
  
  - 澄清了该机制涉及 **majority vote**（多数投票），并提供了相关代码的参考链接。
- **修改评估指标**：建议修改评估机制以直接检查正确响应，并将下游指标从 **exact_match** 切换到 **acc**。
  
  - 这旨在提高测试期间评估模型响应的可靠性。

 

**提及的链接**：[lm-evaluation-harness/lm_eval/filters/selection.py at c0745fec3062328e0ab618f36334848cdf29900e · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c0745fec3062328e0ab618f36334848cdf29900e/lm_eval/filters/selection.py#L56)：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

 

---

### **Eleuther ▷ #**[**multimodal-general**](https://discord.com/channels/729741769192767510/795089627089862656/1302008865247854694) (11 messages🔥):

> - `DINO 的 ImageNet 预训练`
> - `Jordan 对动漫艺术的看法`

- **DINO：蒸馏与评估指标提升**：成员们讨论了 DINO 的 **ImageNet pretraining** 可能会提升评估指标，其中 v2 使用了 **22k** 数据而非 **1k**。
  
  - DINOv2 论文证明了使用 **22k** 数据集在 **1k** 上表现更好，表明了有效的蒸馏（distillation）。
- **Jordan 创建动漫艺术讨论帖**：一名成员指出 Jordan 创建了一个名为“[Jordans anime art thoughts]”的讨论帖。
  
  - 围绕 Jordan 对动漫艺术看法的讨论引起了兴趣，反馈被描述为“非常棒”。

 

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1302205627342393406) (1 messages):

> - `GPT-NeoX 演示`
> - `Colab Notebooks`

- **GPT-NeoX 在 Colab 上的演示**：一名成员开发了一个可以在 [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) 上运行的 **GPT-NeoX** 小型演示，以方便学习该模型。
  
  - 他们请求对该 notebook 进行评审，以确保它能作为他人的“优秀示例”。
- **请求评审 GPT-NeoX Notebook**：演示作者正在寻求针对 [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) 的具体反馈，以确保其符合教育标准。
  
  - 该倡议旨在帮助他人学习 **GPT-NeoX**，使其对新手更加友好。

 

**提到的链接**：[GPT-NeoX-Colab/notebooks/shakespeare_training.ipynb at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb)：GPT-NeoX 的示例 Colab notebooks。通过在 GitHub 上创建账号为 markNZed/GPT-NeoX-Colab 的开发做出贡献。

 

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1303091303843762238) (2 messages):

> - `Claude 3.5 Haiku 性能`
> - `Aider v0.62.0 特性`
> - `代码编辑排行榜`
> - `文件编辑应用`
> - `Aider 中的 Bug 修复`

- **Claude 3.5 Haiku 在排行榜上的得分**：新款 **Claude 3.5 Haiku** 在 [aider 的代码编辑排行榜](https://aider.chat/docs/leaderboards/)上获得了 **75%** 的分数，排名紧随旧版 **Sonnet 06/20** 之后。
  
  - 凭借这一表现，Haiku 成为一个性价比极高且能力接近 Sonnet 的替代方案。
- **Aider v0.62.0 引入新特性**：Aider 的最新版本 **v0.62.0** 全面支持 **Claude 3.5 Haiku**，允许用户使用 `--haiku` 参数启动。
  
  - Aider 现在支持应用来自 ChatGPT 等各种 Web App 的编辑，增强了开发者的易用性。
- **应用文件编辑的新方法**：用户现在可以通过使用 ChatGPT 或 Claude 的 Web App，复制回复内容并运行 `aider --apply-clipboard-edits file-to-edit.js` 来应用文件编辑。
  
  - 这一流程简化了直接根据 LLM 生成的更改进行代码编辑的过程。
- **Aider 0.62.0 Bug 修复**：此次更新修复了与创建新文件相关的 **bug**。
  
  - 这一改进提升了 Aider 对开发者的整体稳定性和可用性。
- **Aider 的开发与贡献**：值得注意的是，**Aider 编写了本次发布中 84% 的代码**，展示了其自我改进的能力。
  
  - 这种程度的贡献表明 Aider 在功能和自主性方面正在不断进步。

 

**提到的链接**：[Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/)：LLM 代码编辑技能的定量基准测试。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1302026292413337600) (240 条消息🔥🔥):

> - `Aider 功能与更新`
> - `AI 模型对比`
> - `基准测试性能`
> - `Token 使用与管理`
> - `命令行指令`

- **Aider 新功能：剪贴板编辑**：Aider 引入了新的 `--apply-clipboard-edits` 功能，允许用户从 Web 端 LLM UI 复制代码响应，并无缝应用到本地文件中。
  
  - 此次更新提升了编辑效率，并与最近发布的 Haiku 3.5 保持同步。
- **模型对比见解**：社区讨论了各种 AI 模型的性能，其中 **Sonnet 3.5** 因其质量而备受关注，而 **Haiku 3.5** 被视为强劲的竞争对手，但速度较慢。
  
  - 参与者对比了不同模型的编程能力，并分享了在实际项目中使用它们的经验。
- **基准测试性能揭晓**：Paul G 分享了基准测试结果，显示 **Sonnet 3.5** 的表现优于其他模型，而 **Haiku 3.5** 在效率排名中较低。
  
  - 人们对不同模型组合（如 Sonnet 和 Haiku）在各种任务中的表现很感兴趣。
- **有效管理 Token 限制**：用户对大型项目中达到 Token 限制表示担忧，并讨论了通过限制上下文和减少不必要的文件编辑来管理这一问题的策略。
  
  - 社区建议包括运行 `/tokens` 来分析 Token 使用情况，以及使用 `/clear` 来重置聊天记录。
- **命令行指南**：分享了各种有效使用 Aider 的命令行指令，包括环境搭建和模型规范。
  
  - 讨论了利用 OpenRouter 模型以及确保 Aider 内 Git 功能正常运行的具体命令。

**提到的链接**：

- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1853564730872607229)：推出 Predicted Outputs——通过提供参考字符串，显著降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [安装 aider](https://aider.chat/docs/install/install.html)：aider 是你终端里的 AI 配对编程工具
- [欢迎来到 Retype - 使用 Retype 写作！](https://retype.com/)：Retype 是一款 ✨ 超高性能 ✨ 的静态网站生成器，基于简单的 Markdown 文本文件构建网站。
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [GitHub Codespaces](https://aider.chat/docs/install/codespaces.html)：aider 是你终端里的 AI 配对编程工具
- [分离代码推理与编辑](https://aider.chat/2024/09/26/architect.html)：Architect 模型描述如何解决编程问题，Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。
- [software-dev-prompt-library/prompts/documentation/generate-project-README.md at main · codingthefuturewithai/software-dev-prompt-library](https://github.com/codingthefuturewithai/software-dev-prompt-library/blob/main/prompts/documentation/generate-project-README.md)：包含针对常见软件工程任务的经过测试的可重用生成式 AI 提示词库 - codingthefuturewithai/software-dev-prompt-library
- [0.61.0 版本创建新文件的 bug · Issue #2233 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2233)：在 Discord 中报告：https://discord.com/channels/1131200896827654144/1302139944235696148/1302139944235696148 嗨，在昨天更新到 0.61 版本后，aider 在编写新文件时遇到问题...
- [GitHub - Aider-AI/aider: aider 是你终端里的 AI 配对编程工具](https://github.com/Aider-AI/aider.git)：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1302026565873438860) (80 条消息🔥🔥):

> - `Aider 中的 Prompt Caching`
> - `Aider 安装问题`
> - `Aider 中的配置管理`
> - `LLM Token 限制差异`
> - `使用多个配置文件`

- **了解 Aider 的 Prompt Caching 功能**：用户询问在不使用 /add 或 /clear 命令的情况下更改文件时，Aider 是否会缓存 prompt，并对效率表示关注。
  
  - 讨论表明，如果配置正确（特别是在 .env 文件中），Aider 能有效管理 Prompt Caching。
- **Aider 安装问题排查**：用户在 v61 更新后启动 Aider 遇到问题，最初出现错误信息，直到重新配置 .env 和 .yaml 文件后才解决。
  
  - 有人指出，在不同仓库之间切换时，删除某些配置文件有助于修复 Aider 功能中的故障。
- **配置管理：Aider 的 .env 与 .aider.conf.yml**：用户讨论了 --read 命令如何在命令行中指定多个文件，并澄清了 .env 文件中 AIDER_FILE 行的功能。
  
  - 结论是 Aider 可以同时利用这两种配置方法，但可能会优先考虑 .aider.conf.yml 的配置而非 .env 设置。
- **本地 LLM 的 Token 计数差异**：用户报告在使用 Aider 配合本地 LLM 时 Token 计数不匹配，导致对会话期间的计算产生困惑。
  
  - 其他人提供了关于 Token 限制以及 Aider 如何管理 Token 输入和输出的信息，并参考了文档中的说明。
- **在 Aider 中利用多个配置文件**：讨论者探讨了如何在 Aider 配置中声明多个文件，表示需要明确是否支持多个 AIDER_FILE 条目。
  
  - 建议包括同时使用 .env 和 .aider.conf.yml 文件来有效管理密钥和配置。

**提到的链接**：

- [依赖版本](https://aider.chat/docs/troubleshooting/imports.html)：aider 是你终端里的 AI 结对编程助手。
- [Prompt caching](https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)：Aider 支持 Prompt Caching 以节省成本并加快编码速度。
- [YAML 配置文件](https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file)：如何使用 YAML 配置文件配置 aider。
- [Token 限制](https://aider.chat/docs/troubleshooting/token-limits.html)：aider 是你终端里的 AI 结对编程助手。
- [aider/aider/website/assets/sample.env at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/aider/website/assets/sample.env#L285)：aider 是你终端里的 AI 结对编程助手。欢迎在 GitHub 上为 Aider-AI/aider 的开发做出贡献。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1302595306185031710) (4 条消息):

> - `对 OpenAI o1 的期待`
> - `IBM Granite 编程模型`
> - `Granite 性能评估`
> - `YouTube 讨论`

- **OpenAI 制造 o1 噱头**：用户指出 OpenAI 最近泄露的 o1 似乎是精心策划的战略，旨在为即将发布的版本制造**期待感**，而非意外泄露。
  
  - 他们强调了 **Sam Altman** 此前如何通过重复的意象培养兴趣，例如分享**草莓**照片和引用 **Orion（猎户座）**星空。
- **用户询问 IBM Granite 模型**：一名成员询问是否有人使用过 **IBM Granite 编程模型**，并链接到了官方[文档](https://www.ibm.com/granite/docs/models/code/)。
  
  - 他们强调这些模型专为跨 **116 种编程语言**的代码生成任务而设计。
- **Granite 模型的历史性能**：早些时候的一位用户评论了 IBM Granite 模型的**性能**，称其过去表现欠佳，但未提供最近的反馈。
  
  - 这引发了一个问题：自那以后的改进是否影响了它们在代码相关任务中的有效性。
- **YouTube 视频引用**：用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=tElgVPUargw) 链接，未提供额外背景。
  
  - 这可能与之前关于编程模型或 OpenAI 发布内容的讨论有关，但未提供具体细节。

**提到的链接**：

- [来自 Chubby♨️ (@kimmonismus) 的推文](https://x.com/kimmonismus/status/1852878364899684837)：o1 抢先看。事实上，OpenAI 今天似乎并不是意外泄露了 o1，而更像是一场精心策划的行动，旨在为即将到来的内容制造期待。记住这一点很重要...
- [未找到标题](https://www.ibm.com/granite/docs/models/code/)：未找到描述

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1302004276104794172) (184 条消息🔥🔥):

> - `Potential AGI Capabilities` (潜在的 AGI 能力)
> - `Dependency on Technology` (对技术的依赖)
> - `AI Relationships` (AI 关系)
> - `OpenAI's Future Developments` (OpenAI 的未来发展)
> - `AI Versus Human Interaction` (AI 与人类互动的对比)

- **AGI 重新利用技术的潜力**：讨论了 **AGI 重新利用技术**进行高级应用的可能性，例如将 ISS 转化为**新能源**或创建自清洁建筑。
  
  - 这反映了考虑 AGI 未来影响的更广泛需求，并有评论指出对 AGI 提议保持**开放心态的重要性**。
- **社会对技术的依赖**：一位成员强调，对**技术的依赖导致了对没有技术的世界缺乏准备**，并警告不要只关注 AI 关系。
  
  - 他们引发了关于**社会对技术的依赖**如何影响日常生活，以及在 EMP 等紧急情况下可能带来的风险的思考。
- **对 AI 友谊的担忧**：有人担心像 OpenAI 这样的公司可能会培养对 AI **不健康的依恋**，从而加剧由孤独引发的社会问题。
  
  - 一位成员建议，这种对 AI 的依赖可能与社交媒体对人际关系的影响**同样令人担忧**。
- **对 OpenAI 新项目的兴奋**：关于 **OpenAI 即将开展的项目**（包括 Orion 项目）的讨论让成员们对预计在 **2025** 年实现的 AI 创新感到兴奋。
  
  - 对话指出，虽然取得了进展，但目前的模型如 O1 还不是**真正的 AGI**。
- **GPT-4o 的新功能**：注意到 **GPT-4o 的下一个版本**正在推出，使部分用户能够获得类似于 O1 的**高级推理能力**。
  
  - 此次升级包括将大段文本放入 **canvas 风格的框**中等功能，增强了推理任务的可用性。

**提到的链接**：

- [来自 Jason -- teen/acc (@ArDeved) 的推文](https://x.com/ArDeved/status/1852649549900242946)：等等？我是不是刚刚获得了完整版 o1 的访问权限，什么鬼？？？
- [来自 Doomlaser Corporation (@DOOMLASERCORP) 的推文](https://x.com/DOOMLASERCORP/status/1849986429398437975)：回溯到 2015 年：我们的 CEO @Doomlaser 在 WEFT 90.1 FM Champaign, IL 讨论 AI、脑机接口、符号逻辑、企业责任和社区服务——带着独特的愿景...
- [挖掘岩浆](https://youtu.be/TmkBtNwmY_E)：这 AI 到底在产生什么幻觉
- [cha ching...](https://www.youtube.com/watch?v=wdh7WmksZbY)：未找到描述
- [暂停大型 AI 实验：一封公开信 - Future of Life Institute](https://futureoflife.org/open-letter/pause-giant-ai-experiments/)：我们呼吁所有 AI 实验室立即暂停训练比 GPT-4 更强大的 AI 系统至少 6 个月。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1302318223546384396) (14 条消息🔥):

> - `Middleware Products` (中间件产品)
> - `Channel Naming Preferences` (频道命名偏好)
> - `O1 Observations` (O1 观察)
> - `GPT-5 Announcement` (GPT-5 发布公告)
> - `Discord Search Functionality` (Discord 搜索功能)

- **探索中间件产品**：一位成员询问是否有人使用指向不同端点而非直接指向 OpenAI 的**中间件产品**，并提到他们正在开发相关产品，询问这是否属于正常现象。
  
  - *这是一种典型的模式吗？*
- **关于频道名称的辩论**：**CaptainStarbuck** 质疑了频道名称的相关性，因为目前它无法反映对 GPT 以外模型的讨论，建议需要像 **#o1-discussions** 这样的专用频道。
  
  - 另一位成员同意设立**特定频道**的好处，但同时也警告不要设立过多频道。
- **Discord 搜索功能的使用**：在讨论 **O1** 笔记时，一位成员提到了 **Discord 搜索功能**对于追踪埋没在聊天中的信息的重要性。
  
  - 对话强调，搜索可以帮助定位跨多个频道的历史消息。
- **对 GPT-5 发布的好奇**：**Neomx8** 表示想知道 **GPT-5** 何时发布以及 API 何时可用，对此另一位成员幽默地回应说没人知道。
  
  - 有一种观点认为今年会有新发布，但不会是 GPT-5。
- **对社区讨论的担忧**：**CaptainStarbuck** 对部分社区成员不区分不同模型表示沮丧，这导致了各频道中关于各种话题的讨论非常混乱。
  
  - *OpenAI 本可以做得更好*，以更清晰的方式安排讨论。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1302042551850766467) (63 messages🔥🔥):

> - `代码文档的最佳实践`
> - `编写高质量 Prompt 的工具`
> - `衡量用户与 LLM 的交互`
> - `使用 LLM 进行规划`
> - `在生产环境中使用 LLM`

- **分享了代码文档的最佳实践**：成员们讨论了插入代码块进行文档说明的方法，建议对小型脚本使用三反引号（triple backticks），对大型项目则上传文件。
  
  - 一位成员强调了向 LLM 展示代码时清晰度的重要性，以便进行有效处理。
- **讨论 Prompt 编写工具**：成员们辩论了用于衡量 Prompt 的分析工具，重点在于评估用户挫败感和任务完成率。
  
  - 建议包括使用情感分析，并引导 LLM 处理对话以获取洞察。
- **使用 LLM 规划和自动化代码创建**：一位用户寻求使用 LLM 自动化复杂任务的资源，特别是在生成 SQL 查询和数据分析方面。
  
  - 贡献者强调了需求清晰度的必要性，以及模型在辅助头脑风暴方面的潜力。
- **在生产环境中使用 LLM 的哲学**：几位成员分享了在生产环境中使用 LLM 的经验，强调需要构建定制化解决方案，而不是依赖预制项目。
  
  - 讨论突出了在编码时理解内部逻辑和 API 调用（API calls）的重要性。
- **Prompt 构建策略**：成员们讨论了请求 LLM 协助创建 Prompt 的有效性，并指出了输出可能过于冗长的潜在问题。
  
  - 建议包括要求模型进行总结或简化，以保持 Prompt 的重点和简洁。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1302042551850766467) (63 messages🔥🔥):

> - `代码块输入的最佳实践`
> - `Prompt 衡量工具`
> - `针对用户挫败感的情感分析`
> - `使用 LLM 进行自动化规划`
> - `自定义 Prompt 的优势`

- **代码块输入的最佳实践**：在集成代码时，建议使用三反引号来表示代码块，特别是对于短脚本；而较长的脚本可能更适合作为文件附件。
  
  - 这种方法简化了与模型的沟通，确保了输入和输出的清晰度。
- **关于 Prompt 衡量工具的讨论**：成员们探索了各种用于追踪用户交互的分析工具，包括挫败感程度和任务完成率。
  
  - 有人建议情感分析可能会有用，并提到了使用 AI 来分析对话数据。
- **使用 LLM 进行自动化规划**：参与者寻求关于使用 LLM 自动化复杂任务规划的资源，特别是生成 SQL 查询和相关代码。
  
  - 明确了用户查询需要具备特异性（specificity），才能有效利用 LLM 的能力。
- **自定义 Prompt 的优势**：建议指出优化 Prompt 构建至关重要，重点在于清晰和简洁，以便模型更好地理解。
  
  - 相反，有人建议人类应该解读模型响应以增强 Prompt 的效力，而不是仅仅依赖模型的输出。
- **在生产环境中使用 LLM**：一些成员分享了在生产环境中使用 LLM 的经验，强调使用 API 进行定制和功能实现。
  
  - 对依赖预制项目的做法提出了担忧，指出了潜在的许可问题以及在理解和升级方面的局限性。

 

---

### **NotebookLM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1302001816942153808) (49 条消息🔥):

> - `NotebookLM 语言配置`
> - `提升播客参与度`
> - `特殊需求用例`
> - `分形网格概念讨论`
> - `播客分发的版权问题`

- **NotebookLM 语言配置**：用户表示在配置 NotebookLM 以首选语言响应时遇到困难，尤其是在上传不同语言的文档时。
  
  - 频道内分享了更改语言设置的说明，强调 NotebookLM 默认使用用户 Google 账号中设置的语言。
- **提升播客参与度的策略**：一名成员建议在 NotebookLM 播客中集成虚拟形象 (Avatars)，以在音频内容输出期间提升参与度。
  
  - 另一位用户提到，正在探索在书籍出版前将其章节转换为对话格式来创建播客。
- **寻求特殊需求用例**：一位希望分享 NotebookLM 在特殊教育学生中应用经验的成员，正在寻求用例或成功案例，以便向 Google 的无障碍团队 (Accessibility Team) 进行提案。
  
  - 成员们表达了在英国建立一个集体以进一步推动这一事业的兴趣。
- **关于分形网格 (Fractal Grid) 概念的讨论**：一位成员提出将分形网格作为组织 NotebookLM FAQ 输出的方法，并将其与门格海绵 (Menger Sponge) 等数学概念联系起来。
  
  - 这一讨论引发了关于使用“分形”一词描述网格结构的争论，促使了进一步的澄清。
- **播客素材的版权担忧**：一位用户对使用自己书籍的内容来创建对话式播客表示担忧，质疑潜在的版权问题。
  
  - 尽管拥有内容所有权，他们仍希望明确分发这些 AI 生成的对话是否会侵犯任何权利。

**提到的链接**：

- [Changelog | Readwise](https://readwise.io/changelog/notebooklm-integration)：变得更聪明，更好地记住书本内容：Readwise 每天向您发送电子邮件，重现来自 Kindle、Instapaper、iBooks 等平台的精彩高亮内容。
- [Trailer zu Wein & Wahnsinn: Verrückte Fakten in 5 Minuten](https://open.spotify.com/episode/5KJM4LWOOYKq4yDydrzD1e?si=XEU2fWM9QY6qG7GCAPN-WQ)：《葡萄酒与疯狂：5 分钟冷知识》剧集预告。
- [无标题](https://notebooklm.google.com/notebook/0933b408-183e-41ad-91ff-019856e9bdff/audio)：未找到描述。
- [Quant Pod: "Deep Dive into Hedge Fund Strategies: Risk, Return and Complexity"](https://www.youtube.com/watch?v=N41sczY0k0E&t=72s)：未找到描述。

---

### **NotebookLM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1301999350330691594) (204 条消息🔥🔥):

> - `播客质量问题`
> - `使用 PDF 作为来源`
> - `NotebookLM 的语言支持`
> - `音频概览与摘要`
> - `API 开发计划`

- **播客质量问题**：多位用户反映了最近几周生成的播客质量问题，指出在播放过程中出现了意外的中断和随机杂音。
  
  - 一些用户觉得这些中断挺有趣，而另一些用户则对该问题影响听觉体验表示沮丧。
- **使用 PDF 作为来源**：用户讨论了向 NotebookLM 上传 PDF 相关的挑战，部分用户在尝试访问存储在 Google Drive 中的文件时遇到了问题。
  
  - 频道内澄清，目前只有原生的 Google Docs 和 Slides 可以直接从 Drive 中选择作为来源，PDF 则不行。
- **NotebookLM 的语言支持**：有关于生成非英语播客能力的咨询，一些用户测试了各种口音和语言。
  
  - 结果因语言而异，据报道法语和西班牙语效果良好，而瑞典语和日语则表现不佳。
- **音频概览 (Audio Overviews) 与摘要**：一位用户询问如何从一份 200 页的密集 PDF 中生成多个音频概览，并考虑采用将文档拆分为较小部分的劳动密集型方法。
  
  - 建议包括提交功能请求，以支持从单一来源生成多个音频概览或将来源自动分割成多个片段。
- **API 开发计划**：关于 NotebookLM 可能开发 API 的讨论显示，虽然存在兴趣，但尚未发布官方公告。
  
  - 一些用户根据行业活动的评论推测 API 可能正在开发中，但目前尚未确认任何明确信息。

- [Google Colab](https://colab.research.google.com/drive/1PZMVolQQNwCIW2PJpZf0I99cmN5ud9MO): 未找到描述
- [来自 Vicente Silveira (@vicentes) 的推文](https://x.com/vicentes/status/1844202858151068087): API 即将到来！
- [账号设置：您的浏览器不受支持。](https://myaccount.google.com/age-verification): 未找到描述
- [Battleship Youve Sunk My Battleship GIF - Battleship Youve Sunk My Battleship - 发现并分享 GIF](https://tenor.com/view/battleship-youve-sunk-my-battleship-gif-25203446): 点击查看 GIF
- [AI 笔记、转录与摘要 | AI Notebook App](https://ainotebook.app/): 为大学生的讲座生成转录文本和 AI 摘要。专注于 YouTube 视频摘要、PDF 摘要、文章摘要。保存关键见解并使用学习指南进行复习...
- [未找到标题](https://notebooklm.google.com/,): 未找到描述
- [#4 搞笑电影死亡场景 GIF - Gasoline Party Zoolander 前五名电影死亡场景 - 发现并分享 GIF](https://tenor.com/view/gasoline-party-zoolander-top-five-movie-deaths-movies-comedy-gif-3478949): 点击查看 GIF
- [登录 – Google 账号](https://notebooklm.google.com?hl=sv): 未找到描述
- [注册免费网络研讨会：更有效地使用 NotebookLM 进行思考、分析和创作](https://docs.google.com/forms/d/1z1gkLVCEX-2XFKsxQkr5p2CL6FEbdkoS1thUqpU6pNc/edit#responses): 活动时间：2024年11月3日，UTC 时间下午6点。活动地点：Google Meet。现场参与者将获得独家问答机会和互动环节！无法参加？别担心 - 您仍然可以...
- [What A Fool Believes - 隔离期翻唱](https://youtu.be/ZnNZMF0eC0c?si=YgK6JURaLQd0R3RV): 这是我们在那段疯狂时期制作的翻唱作品，来自伦敦的封锁期演奏！Giulio Romano Malaisi - 吉他 @giulioromanoguitar，Edoardo Bombace - 贝斯 ...
- [产品背后：NotebookLM | Raiza Martin (Google Labs AI 高级产品经理)](https://youtu.be/sOyFpSW1Vls?si=kj3Qs1AL792TnjQP): Raiza Martin 是 Google Labs 的 AI 高级产品经理，她领导着 NotebookLM 背后的团队，这是一款由 AI 驱动的研究工具，包含了一个令人愉悦的...
- [未找到标题](https://cloud.google.com/text-to-speech/docs/voice-types): 未找到描述
- [GitHub - podcast-lm/the-ai-podcast: 一个开源的 AI 播客生成器](https://github.com/podcast-lm/the-ai-podcast/): 一个开源的 AI 播客生成器。通过在 GitHub 上创建账户，为 podcast-lm/the-ai-podcast 的开发做出贡献。
- [Deep Dive Stories - 中世纪戏剧评论](https://www.youtube.com/watch?v=D8kPQvVPEeg&list=PLteYdC-1C8MgBTqzOQRstEZ0MowsRDetV)): Slippery Slide Plays：AI 挑战中世纪戏剧 | 一场关于道德、神秘及更多内容的即兴探讨。在本集中，我们将回到中世纪时期...
- [Deep Dive Stories - 1947年万圣节](https://youtu.be/Urv3r-touT8)): 探索《别跟我提万圣节》的恐怖故事。步入万圣节主题剧集，我们将探索经典广播剧《别跟我提...》中的一个场景。

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1302343527304532203) (18 条消息🔥):

> - `Low-bit Triton Kernels`
> - `CUDA Matrix Multiplication Optimization`
> - `Custom CUDA Kernels vs PyTorch`
> - `PyTorch H100 Optimizations`
> - `Flash Attention Package`

- **Dr. Hicham Badri 讨论 Low-bit Triton Kernels**：Dr. Hicham Badri 目前正在通过 [zoom link](https://linkedin.zoom.us/j/98000015325?pwd=KpafrbHFiWjHVXimMzVsebGYZZVxeo.1) 举行的会议中分享关于 **Low-bit Triton Kernels** 的内容。
  
  - 有人请求录音或参考资料，一名成员提供了 [Gemlite](https://github.com/mobiusml/gemlite) 链接作为相关资源。
- **与来自 Anthropic 的 Simon 探讨 CUDA 矩阵乘法**：一名成员介绍了一篇由来自 Anthropic 的 Simon 撰写的关于在 CUDA 中优化矩阵乘法的博客文章，强调了现代深度学习中关键的性能特征。该文章链接见 [此处](https://siboehm.com/articles/22/CUDA-MMM)，并包含 GitHub 仓库供参考。
  
  - 内容针对的是在模型训练和推理中占据主要 FLOPs 的基础算法（如矩阵乘法），展示了其在该领域的重要性。
- **自定义 CUDA Kernels 与 PyTorch 的性能对比**：一名成员表达了对自定义 CUDA Kernels 在速度上难以与 PyTorch 竞争的担忧，因为各方贡献者已经投入了大量精力。然而，其他人指出，通过针对特定工作负载的专业知识，超越现有的 Kernels 是可行的。
  
  - 成员们强调，自定义 Kernels 可以更容易地进行调整和优化，并指出某些 PyTorch Kernels 存在缺陷，这为改进提供了机会。
- **PyTorch 的 H100 优化**：有关于 PyTorch 是否针对 **H100** 硬件进行了优化的推测，确认其确实支持 **cudnn attention**。然而，有报告称存在问题，导致其在 2.5 版本中被默认禁用。
  
  - 成员们对 H100 特性的稳定性看法不一，其中一人强调 H100 上的 **Flash Attention 3** 也仍在成熟过程中。

**提到的链接**：

- [加入我们的云高清视频会议](https://linkedin.zoom.us/j/98000015325?pwd=KpafrbHFiWjHVXimMzVsebGYZZVxeo.1)：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。
- [如何优化 CUDA Matmul Kernel 以达到类 cuBLAS 性能：工作日志](https://siboehm.com/articles/22/CUDA-MMM)：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...
- [GitHub - mobiusml/gemlite: CUDA / Triton 中简单且快速的 low-bit matmul kernels](https://github.com/mobiusml/gemlite?tab=readme-ov-file#gemlite)：CUDA / Triton 中简单且快速的 low-bit matmul kernels - mobiusml/gemlite

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1302003576188567595) (100 条消息🔥🔥):

> - `Triton Kernel 优化`
> - `Fused Swiglu 性能`
> - `FP8 量化技术`
> - `动态激活缩放`
> - `Triton Kernel 缓存`

- **Triton Kernel 优化挑战**：成员们讨论了优化 Triton kernel 的复杂性，特别是 GPU 等待 CPU 操作的时间以及这在 microbenchmarks 中造成的开销。
  
  - 有建议称，堆叠多个 Triton matmul 操作可以显著减少这种开销，从而提高大矩阵的性能。
- **Fused Swiglu 性能问题**：有人对带有三个线性权重的 `fused swiglu` 性能表示担忧，其速度比 cublas 慢，并讨论了可能导致这些速度问题的原因。
  
  - 讨论指向了多家机构针对 H100 和 H200 GPU 实现的潜在优化。
- **用于提速的 FP8 量化技术**：一位用户分享了他们在 FP8 量化方法方面的经验，对自定义实现与纯 PyTorch 版本相比所取得的速度提升感到惊讶。
  
  - 一个 GitHub 链接展示了各种实现，对话还涉及了高效进行动态量化激活（dynamically quantizing activations）的挑战。
- **Transformer 中的动态激活缩放**：探索了为 Transformer 块缓存激活缩放比例（activation scales）的方法，用户分享了在正向传播（forward pass）期间提高效率的策略细节。
  
  - 尽管相比编译时性能，用户更倾向于低延迟和快速启动，但大家公认 Triton 可能不是最合适的选择。
- **Triton Kernel 的缓存策略**：成员们讨论了在 Triton kernel 中为 autotuning 结果建立缓存机制的潜力，以避免对重复形状（repeated shapes）进行重新编译，从而提高效率。
  
  - 官方缓存功能的必要性达成了一致，并引用了尝试类似功能但缺乏流线型支持的现有项目。

**提到的链接**：

- [Supercharging NVIDIA H200 and H100 GPU Cluster Performance With Together Kernel Collection](https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection)：未找到描述
- [Ok Oh Yes Yes O Yeah Yes No Yes Go On Yea Yes GIF - Ok Oh Yes Yes O Yeah Yes No Yes Go On Yea Yes - Discover & Share GIFs](https://tenor.com/view/ok-oh-yes-yes-o-yeah-yes-no-yes-go-on-yea-yes-gif-14382673246413447193)：点击查看 GIF
- [Together AI – Fast Inference, Fine-Tuning & Training](https://www.together.ai)：通过易于使用的 API 和高度可扩展的基础设施运行和微调生成式 AI 模型。在我们的 AI 加速云和可扩展 GPU 集群上大规模训练和部署模型。
- [flux-fp8-api/float8_quantize.py at 153dd913d02f05023fdf3b6c24a16d737f3c1359 · aredden/flux-fp8-api](https://github.com/aredden/flux-fp8-api/blob/153dd913d02f05023fdf3b6c24a16d737f3c1359/float8_quantize.py#L195-L296)：使用量化 fp8 matmul 的 Flux 扩散模型实现，其余层使用更快的半精度累加，在消费级设备上速度提升约 2 倍。 - aredden/flux-fp8-api
- [GitHub - aredden/flux-fp8-api](https://github.com/aredden/flux-fp8-api)：使用量化 fp8 matmul 的 Flux 扩散模型实现，其余层使用更快的半精度累加，在消费级设备上速度提升约 2 倍。 - aredden/flux-fp8-api
- [GitHub - aredden/f8_matmul_fast](https://github.com/aredden/f8_matmul_fast)：通过在 GitHub 上创建账户为 aredden/f8_matmul_fast 的开发做出贡献。
- [gemlite/tests/test_gemlitelineartriton.py at master · mobiusml/gemlite](https://github.com/mobiusml/gemlite/blob/master/tests/test_gemlitelineartriton.py#L224-L258)：CUDA / Triton 中简单且快速的低比特 matmul kernel - mobiusml/gemlite
- [f8_matmul_fast/src/quantize_fp8_experimental.cu at 426208a02f56207afa86f5116f355edd41718ada · aredden/f8_matmul_fast](https://github.com/aredden/f8_matmul_fast/blob/426208a02f56207afa86f5116f355edd41718ada/src/quantize_fp8_experimental.cu#L76C17-L149)：通过在 GitHub 上创建账户为 aredden/f8_matmul_fast 的开发做出贡献。

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1302761740555128885) (7 messages):

> - `Hardware/System-Agnostic Reproducibility` (硬件/系统无关的可复现性)
> - `Torch Compile Overhead` (Torch 编译开销)
> - `Determinism Across Different GPUs` (不同 GPU 间的确定性)

- **探索硬件/系统无关的可复现性**：一位用户分享了一段 Python 代码片段，用于在相同硬件/系统上使用 PyTorch 实现位精确（bit-exact）的可复现性，但对其在不同系统或 GPU 上的可行性表示怀疑。
  
  - 其他成员讨论了针对特定 GPU 的调优对算法性能的影响，强调了在不同 GPU 型号之间实现确定性结果的挑战。
- **Torch 编译开销查询**：一位成员询问是否可以打印检查 guards 的时间，以了解 Torch 中的编译开销。
  
  - 关于这一特定查询，目前没有实质性的后续讨论。
- **不同 GPU 确定性的现实情况**：另一位成员警告说，不要指望在不同 GPU 上运行模型时具有确定性，并强调了每种架构都有其独特的调优方式。
  
  - 这一观点得到了其他人的共鸣，他们确认 GPU 配置的变化会显著影响可复现性。

 

**提到的链接**：[BC-Breaking Change: torch.load is being flipped to use weights_only=True by default in the nightlies after #137602](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573)：TL;DR 在 2.4 版本对该更改发出警告后，#137602 已被合并，这将在 nightlies 版本（以及后续版本）中将 `torch.load` 的 `weights_only` 参数默认值从 `False` 更改为 `True`……

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1302830898219192401) (2 messages):

> - `Anyscale`
> - `Databricks`

- **从 Anyscale 到 Databricks 的转型**：提到了之前对 **Anyscale** 的关注现在已转向 **Databricks**。
  
  - 该评论暗示了人员或重点的显著变化，反映了社区内部的一种动向。
- **关于社区转向的讨论**：成员们指出关注点从 **Anyscale** 到 **Databricks** 的显著转变，预示着项目格局的变化。
  
  - 这种情绪暗示了对社区内正在进行的项目和协作的潜在影响。

 

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1302056535081615431) (1 messages):

> - `Hiring LLM Infra Engineers` (招聘 LLM 基础设施工程师)
> - `AI-based Gaming` (基于 AI 的游戏)
> - `LLM Training Techniques` (LLM 训练技术)

- **LLM 基础设施工程师招聘启事**：Anuttacon 正在寻找**顶尖的 LLM 基础设施工程师**，并鼓励感兴趣的候选人将简历发送至 [recruitment@anuttacon.com](mailto:recruitment@anuttacon.com)。
  
  - 他们寻求在 **LLM & Diffusion 训练**、**NCCL** 以及与大规模数据处理和 GPU 健康相关的多项高级技术技能方面的专业知识。
- **基于 AI 的游戏领域的巨大机遇**：该公司特别寻找对**基于 AI 的游戏**充满热情的**玩家**，旨在将游戏与先进的 AI 技术相结合。
  
  - 这一招聘驱动突显了**游戏**与 **AI** 日益增长的交集，反映了显著的行业需求。
- **列出的关键任职资格**：候选人需具备 **Profiling**、**量化（Quantization）**和**蒸馏（Distillation）**等对于现代 AI 基础设施至关重要的技能。
  
  - 强调了对 **Spark** 和 **Airflow** 等工具的熟悉，以及**容错（Fault Tolerance）**和 **Gang Scheduling** 等技术的重要性。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1302068982752219250) (18 条消息🔥):

> - `CUDA Programming Basics`
> - `Coalesced Memory Access`
> - `Setting Up NCU for Profiling`
> - `Introduction to Triton`
> - `Using torch.compile`

- **CUDA 讲座对初学者来说已经足够了**：一位成员确认，在观看讲座之前不需要任何预备知识，并表示这些内容足以入门。
  
  - 另一位成员对这一保证表示认可，对现有资源表现出信心。
- **理解合并内存访问 (Coalesced Memory Access)**：讨论了合并内存访问如何改善延迟，一位成员澄清说，GPU 上会同时读取连续字节，从而提高效率。
  
  - 另一位成员补充说，虽然非合并访问会导致多次请求，但并不一定会导致延迟成比例增加。
- **NCU 设置的挑战**：一位成员分享了在为使用 NCU 进行 GPU Kernel Profiling 设置可复现环境时，过程非常耗时的挫败感。
  
  - 他们寻求更快速设置的技巧，特别是与 lightning.ai 集成时，因为他们在 Kernel Profiling 方面遇到了问题。
- **探索使用 Triton 编写 Kernel**：一位新成员询问，考虑到 `torch.compile` 也能生成 Kernel，在什么情况下从头编写 Triton Kernel 是没有优势的。
  
  - 另一位成员告知，虽然编译器擅长融合（fusion），但它们在优化特定模型瓶颈方面不如手动编写的 Kernel 有效。
- **利用 torch.compile 进行优化**：建议先从 `torch.compile` 开始寻找融合机会，然后再重写特定的、数值等效的 Kernel 以解决性能瓶颈。
  
  - 成员们强调了查看以往讲座中关于有效使用 `torch.compile` 技巧的重要性。

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/1302914588165279774) (1 条消息):

> - `Low Bit Triton Kernel Performance`
> - `Gemlite`

- **深入探讨低位 (Low Bit) Triton Kernel 性能**：关于 **Low Bit Triton Kernel Performance** 的详细探索已在[视频](https://youtu.be/7c3c3bCGzKU?feature=shared)中发布，其中包含有关性能指标和优化的见解。
  
  - 讨论围绕处理低位操作时 Kernel 效率的重要性展开。
- **Gemlite 在 Kernel 性能方面表现出色**：**Gemlite** 在提升低位操作的 Kernel 性能方面被重点提及，展示了其功能和特性。
  
  - 视频承诺将揭示在 Triton 框架内实现高效能的各种技术。

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1303069594302545960) (1 条消息):

> - `TorchAO planning`
> - `Feature requests`
> - `User pain points`

- **TorchAO 规划季开始**：一条消息宣布 **TorchAO** 正在进入规划季，邀请社区提供反馈。
  
  - 重点在于**透明度**，并对有关开发功能和解决痛点的建议保持开放态度。
- **征集社区对功能开发的建议**：该消息寻求社区成员对于他们希望在 **TorchAO** 中构建的最重要功能的意见。
  
  - 鼓励成员分享他们遇到的**最大痛点**，以帮助改进平台。

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1302017576099905649) (2 条消息):

> - `Noob Questions`
> - `Unanswered Questions`

- **鼓励新手提问**：一位成员强调，人们不应该犹豫提问，即使觉得那是“小白问题”，因为每个人都是从零开始的。
  
  - *如果有人想轻视你，让他们记住他们自己也是从某个起点开始的。*
- **关于问题未得到解答的推测**：一位成员推测，问题经常无人回答是因为许多人不知道答案，或者有更高优先级的事情要处理。
  
  - 这凸显了社区在知识共享方面潜在的沟通鸿沟。

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1302185793577812058) (11 条消息🔥):

> - `ROCm 版本建议`
> - `寄存器溢出 (Register spill) 与 Bank 冲突`
> - `使用 rocprofv3 进行 Kernel 性能分析`
> - `基于 XOR 的排列复杂度`
> - `Matrix core 布局问题`

- **推荐使用 ROCm 6.2.1 以获得更好性能**：一位成员建议在 **6.2.1** 之后使用 ROCm，因为其改进了寄存器压力追踪，有助于避免潜在的寄存器溢出。
  
  - 另一位用户提到他们目前正在使用 **ROCm 5.7**，并对该建议表示感谢。
- **寄存器溢出可能导致 Bank 冲突**：讨论中提到了寄存器溢出是否会导致 **Bank 冲突**，特别是关于基于 XOR 的排列实现。
  
  - 一位成员澄清说，如果发生了寄存器溢出，就没有必要关注 **LDS bank 冲突**，因为后者通常问题较小。
- **使用 rocprofv3 轻松进行 Kernel 性能分析**：为了检查寄存器溢出，一位用户建议使用 `-save-temps` 导出 Kernel 汇编，或者使用 rocprof 中的 **Kernel 追踪选项**。
  
  - 重点介绍了使用 `rocprofv3` 进行性能分析的细节，包括在不修改源代码的情况下获得更好的准确性。
- **XOR 排列的复杂性**：成员们表示 **基于 XOR 的排列** 非常复杂，有用户提到即使在使用 ROCm 5.7 时也存在持续的冲突。
  
  - 一位用户指出，虽然可能发生寄存器溢出，但并不一定意味着 LDS 内部会产生冲突。
- **FP16 Matrix Core 布局的挑战**：一位用户报告称，在 MI250 上使用 FP16 Matrix Core 时，很难完全消除 **Bank 冲突**，特别是在使用行-列布局时。
  
  - 他们注意到缺乏像 **ldmatrix.trans** 这样的转置指令来适应获取转置布局，这增加了实现的复杂性。

**提到的链接**：[Using rocprofv3 — ROCprofiler-SDK 0.5.0 Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kernel-profiling)：未找到描述

---

### **GPU MODE ▷ #**[**intel**](https://discord.com/channels/1189498204333543425/1233802893786746880/) (1 条消息):

lavawave03: Hehehe

---

### **GPU MODE ▷ #**[**arm**](https://discord.com/channels/1189498204333543425/1247232251125567609/1302693767937134685) (25 条消息🔥):

> - `ARM CPU 上的 LLM 推理`
> - `张量并行 (Tensor Parallelism) 的优势`
> - `分布式 CPU 推理文档`
> - `Bitnet 模型限制`
> - `针对 Grace CPU 和 Graviton 的研究`

- **探讨 ARM 上的 LLM 推理**：对在 **NVIDIA Grace** 和 **AWS Graviton** 等 ARM CPU 上进行 **LLM 推理** 的兴趣日益增加，尤其是针对 **70B** 等大型模型。成员们分享了以往与 ARM CPU 效率相关的经验和论文。
  
  - 一位参与者指出，将多个廉价的 ARM SBCs 进行**集群化**可以产生出人意料的好结果，特别是使用 **Ampere Altra** 处理器。
- **张量并行 (Tensor Parallelism) 的优势**：讨论强调 **Tensor Parallelism** 可以提高 CPU 的性能，提供经济的推理解决方案，弥补 CPU 和 GPU 之间的差距。一位用户提到，只要有足够的**带宽**，**CPU 配置**在某些任务上可以达到或超过 GPU 的性能。
  
  - 建议使用 **Infiniband** 进行组网以辅助运行大型模型，强调其关注点已超越消费级市场。
- **对分布式推理文档的需求**：出现了对 **CPU** 分布式推理**文档**的需求，并将其与现有的 **GPU** 指南进行了对比。大家对将 CPU 作为替代方案以满足更大 RAM 需求的用例感到好奇。
  
  - 另一位成员强调，CPU 推理对于**低上下文操作**非常有吸引力，这标志着对其在**研究**和专门应用中可行性的持续探索。
- **Bitnet 模型的限制**：参与者对 **Bitnet** 表现出兴趣，但注意到它需要重新训练模型，这限制了其即时实用性。他们的讨论反映了 **SVE 单元**在特定操作中的潜力，同时也指出了扩展方面的挑战。
- **研究重点转向高级 CPU 选项**：对话转向了 **Grace** 和 **Graviton** 等特定 CPU，表明研究重点在于最大化利用它们的能力。成员们注意到了在推理过程中使用 CPU 进行**解码 (decoding)** 的操作细节，同时也承认了它们在大 Batch Size 方面的局限性。

**提到的链接**：[GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile](https://github.com/pytorch/torchchat)：在服务器、桌面和移动端本地运行 PyTorch LLM - pytorch/torchchat

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1302070210928312432) (15 条消息🔥):

> - `Liger Kernel CI 问题`
> - `Liger Rope 函数故障`
> - `自定义 Triton Kernel 集成`

- **Liger Kernel Pull Request 评审流程**：<@t_cc> 确认所有测试在本地均已通过，但 **Gemma2** 在收敛测试中存在已知问题，详见 [Pull Request #320](https://github.com/linkedin/Liger-Kernel/pull/320)。成员们正在考虑在合并前通过更改 vendor 来修复 CI。
- **LigerRopeFunction 在单头配置下存在问题**：<@junjayzhang> 分享了一个 **reproducer**，证明当 `num_head = 1` 时，**LigerRopeFunction** 无法应用更改，未能影响 `q` 或 `v`。提供的代码片段和重复测试突显了使用中的这一关键问题。
- **Cross Entropy 操作期间内存翻倍**：<@t_cc> 针对 [cross_entropy.py](https://github.com/linkedin/Liger-Kernel/blob/e68b291f11d2f1ab22c5db9b1038021ee1821a0e/src/liger_kernel/ops/cross_entropy.py#L255) 中 cross-entropy 操作期间因未使用 `detach` 导致内存翻倍的问题提出了疑问。认为有必要进一步投入研究以理解此问题。
- **Triton Kernel 集成协作**：来自 **PyTorch** 团队的 <@agunapal_87369> 表示有兴趣在 Llama 的 **torch.compile/AOTI** 自定义 Triton kernel 集成方面进行合作。对话表明目前正努力使用 Liger Kernel 增强性能集成，并邀请加入 **PyTorch Slack** 进行进一步讨论。

**提及的链接**：

- [one_head_rope.py](https://gist.github.com/Tcc0403/9f0bec5dd8b996f33df5756279010cbe)：GitHub Gist：即时分享代码、笔记和片段。
- [Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at e68b291f11d2f1ab22c5db9b1038021ee1821a0e · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/e68b291f11d2f1ab22c5db9b1038021ee1821a0e/src/liger_kernel/ops/cross_entropy.py#L255)：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 开发做出贡献。
- [Support FusedLinearCrossEntropy for Gemma2 by Tcc0403 · Pull Request #320 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/320)：摘要：解决 #127。将 softcapping 融合到 cross_entropy kernel 中，使其可以被 fused linear cross entropy 函数调用。测试完成：当前针对 Gemma2 的 monkey patch 无法通过收敛测试...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1302192958241636372) (4 messages):

> - `__CUDA_ARCH__ Macro Usage`
> - `Undefined Behavior in CUDA`
> - `TinyGEMM Kernel Issues`

- **理解 CUDA_ARCH 宏**：一位成员分享了在 CUDA 代码中使用 `__CUDA_ARCH__` 宏的见解，强调了它在调试过程中引起严重困惑的一些陷阱。
  
  - 该帖子建议查阅 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arch) 以获得更深入的理解。
- **CUDA 中的错误与静默失败**：文中提到了一项修正，即 CUDA 代码在运行过程中可能*不报错*，但如果在使用 `cudaGetLastError()` 时没有进行适当检查，仍会产生错误结果。
  
  - 这种不一致性会导致潜在的未定义行为，因为任何情况都可能发生，包括静默失败。
- **TinyGEMM Kernel 的代码保护担忧**：针对 TinyGEMM Kernel 中由 `__CUDA_ARCH__` 宏保护的大型代码块提出了担忧，特别是在针对特定架构时。
  
  - 有人指出，尽管有 *sm>=80* 的检查，但如果上游检查未能阻止在 *sm<80* 上的执行，仍存在未定义行为的可能性。

**提到的链接**：

- [CUDA C++: Using CUDA_ARCH the Right Way](https://tobiasvanderwerff.github.io/2024/11/01/cuda-arch.html)：摘要：在应用 `__CUDA_ARCH__` 宏之前，请阅读 CUDA C++ Programming Guide 的这一章节，以便了解可能出现问题的情况。
- [pytorch/aten/src/ATen/native/cuda/int4mm.cu at 55038aa66162372acc1041751d5cc5c8ed9bc304 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/55038aa66162372acc1041751d5cc5c8ed9bc304/aten/src/ATen/native/cuda/int4mm.cu#L1339,)：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch
- [ao/torchao/csrc/cuda/fp6_llm/fp6_linear.cu at f99b6678c4db86cb68c2b148b97016359383a9b6 · pytorch/ao](https://github.com/pytorch/ao/blob/f99b6678c4db86cb68c2b148b97016359383a9b6/torchao/csrc/cuda/fp6_llm/fp6_linear.cu#L147)：用于训练和推理的 PyTorch 原生量化与稀疏性 - pytorch/ao
- [pytorch/aten/src/ATen/native/cuda/int4mm.cu at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu)：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1303052882144399370) (1 messages):

> - `Discord Cluster Manager Updates`

- **Discord Cluster Manager 即将迎来更新**：一位成员宣布他们将对 GitHub 上的 [Discord Cluster Manager](https://github.com/gpu-mode/discord-cluster-manager) 进行更新。
  
  - 正如描述中所述，该仓库旨在促进对项目持续开发的贡献。
- **GitHub 仓库描述**：[GitHub 仓库](https://github.com/gpu-mode/discord-cluster-manager) 专为开发和贡献 Discord Cluster Manager 而设计。
  
  - 它鼓励用户在 GitHub 上创建账户以参与协作。

 

**提到的链接**：[GitHub - gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager)：通过在 GitHub 上创建账户，为 gpu-mode/discord-cluster-manager 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1302007133876785172) (10 messages🔥):

> - `vLLM Demos`
> - `Llama 70B 的性能优化`
> - `与 Flash Attention 的集成`
> - `LoLCATS 项目`
> - `Kernel 特性实现`

- **vLLM Demos 计划**：有意围绕使用自定义 Kernel 实现 **vLLM** 创建一些 **streams/blogs**，并正在讨论一些有用的特性。
  
  - *如果有帮助的话，* 可能会开发一个带有来自 TK 的自定义 forward pass 的仓库 fork。
- **提升 Llama 70B 的性能**：一位用户表示有兴趣提升 **Llama 70B** 的性能，特别是在 **high decode workloads**（高解码负载）下。
  
  - 建议与 **vLLM maintainers** 合作以实现这些性能目标。
- **Flash Attention 3 实现讨论**：目前正在进行与 **Flash Attention 3** 相关的工作，一位用户提到了一个讨论 `flash_attn_varlen_func` 输出的相关 GitHub issue。
  
  - 有建议认为 **vLLM** 团队的实现可能比目前参考的实现更简洁。
- **LoLCATS 集成进展**：TK 演示项目之一 **LoLCATS** 的 **vLLM integration** 正在进行中，与合作者同步信息是关键。
  
  - 让 **vLLM maintainers** 了解集成进度可以简化未来的开发工作。

**提到的链接**：[using](https://github.com/Dao-AILab/flash-attention/issues/1299) `out` argument will change the output · Issue #1299 · Dao-AILab/flash-attention：你好，感谢这个伟大的库！我正尝试使用 flash_attn_varlen_func 和 flash_attn_with_kvcache 的 out 参数，但我发现它会导致输出不正确。这是预期的吗？W...

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1302034272672026754) (70 messages🔥🔥):

> - `Claude PDF 支持`
> - `AI 搜索大战`
> - `O1 模型泄露`
> - `技能与 AI agents`

- **Claude 推出视觉 PDF 功能**：Claude 已在 [Claude AI](https://claude.ai) 和 Anthropic API 中引入了视觉 PDF 支持，使用户能够分析各种文档格式。
  
  - 这包括从财务报告、法律文件等提取信息的能力，显著增强了用户交互。
- **AI 搜索大战升温**：围绕搜索引擎竞争复苏的讨论出现，特别是 SearchGPT 和 Gemini 等新竞争者的加入，挑战了 Google 的主导地位。
  
  - [AI Search Wars 文章](https://buttondown.com/ainews/archive/ainews-the-ai-search-wars-have-begun-searchgpt/) 概述了各种创新的搜索解决方案及其影响。
- **O1 模型泄露但被迅速修复**：O1 模型曾短暂地通过修改 URL 被用户访问，允许上传图片并具备快速推理（inference）能力。
  
  - ChatGPT 的 O1 在对其潜力的兴奋和猜测中出现，尽管访问权限随后已被限制。
- **AI 技能培训框架讨论**：一条推文详细介绍了一个根据技能下限（floor）和上限（ceiling）对行业进行分类的框架，强调了 AI 在客户支持方面的巨大潜力。
  
  - 分享了关于 AI agents 如何提升全球各岗位技能门槛的见解，以及这将如何影响这些行业的发展。
- **OpenAI API Predicted Outputs 功能发布**：OpenAI 推出了名为 Predicted Outputs 的功能，以降低 gpt-4o 和 gpt-4o-mini 模型的延迟，提高 API 效率。
  
  - 预计该功能将以更快的方式简化博客更新和代码适配等任务。

**提到的链接**：

- [来自 Jason -- teen/acc (@ArDeved) 的推文](https://x.com/ArDeved/status/1852649549900242946)：等等？我刚刚是不是获得了完整版 o1 的访问权限，我的天？？？
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1853564730872607229)：介绍 Predicted Outputs——通过提供参考字符串，大幅降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文](https://x.com/legit_rumors/status/1852625385801859321)：o1 第一次接触 👽 由我和 @Jaicraft39 为您呈现
- [来自 Ananay (@ananayarora) 的推文](https://x.com/ananayarora/status/1852666259478835458?s=46)：ChatGPT 的 O1 模型可以通过 https://chatgpt.com/?model=o1 “秘密”访问，即使下拉菜单里没有！🔥 图像理解功能正常，推理速度惊人...
- [来自 Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1853582592395858394)：技能下限/上限是我一直用来理解哪些行业适合 AI agents 的思维模型：- 客户支持具有低下限 + 低上限 = 巨大的机会 - 销售具有低下限 ...

- [被忽视的 GenAI 用例](https://blog.sumble.com/the-overlooked-genai-use-case/)：职位发布数据揭示了公司计划如何使用 GenAI。最大的用例是数据清洗、处理和分析。Sequoia 合伙人 David Cahn 最近发表的一篇广为流传的博客文章问道……
- [Transformer Circuits 线程](https://transformer-circuits.pub/)：未找到描述
- [来自 TechCrunch (@TechCrunch) 的推文](https://x.com/techcrunch/status/1853510622647873782?s=46)：Perplexity CEO 提议用 AI 替换罢工的《纽约时报》员工 https://tcrn.ch/3AqUZfb
- [在地下室运行 AI —— 第二部分：拆解 SWE Agentic Framework、MoEs、批处理推理等 · Osman's Odyssey: Byte & Build](https://ahmadosman.com/blog/serving-ai-from-the-basement-part-ii/)：SWE Agentic Framework、MoEs、量化与混合精度、批处理推理、LLM 架构、vLLM、DeepSeek v2.5、Embedding 模型以及投机解码（Speculative Decoding）：一次 LLM 脑力激荡……我一直……
- [来自 Robert Yang (@GuangyuRobert) 的推文](https://x.com/guangyurobert/status/1852397383939960926?s=46)：一个拥有 1000 亿数字人类的世界会是什么样子？今天我们分享了关于 Project Sid 的技术报告——一瞥首个 AI Agent 文明（由我们全新的 PIANO 架构驱动）……
- [来自 Smol AI (@Smol_AI) 的 AI News 推文](https://x.com/smol_ai/status/1852245457516532175?s=46)：[2024年10月31日] AI 搜索大战已经开始 —— SearchGPT、Gemini Grounding 等 https://buttondown.com/ainews/archive/ainews-the-ai-search-wars-have-begun-searchgpt/
- [PDF 支持 (beta) - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support)：未找到描述
- [来自 Etched (@Etched) 的推文](https://x.com/Etched/status/1852089772329869436)：介绍 Oasis：首款可玩的 AI 生成游戏。我们与 @DecartAI 合作构建了一个实时交互式世界模型，在 Sohu 上的运行速度提高了 10 倍以上。我们正在开源该模型……
- [来自 Suhail (@Suhail) 的推文](https://x.com/Suhail/status/1852411068754403605)：搜索引擎大战时隔 26 年再次开启。我一直在想什么时候会有东西能撼动 Google。
- [来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文](https://x.com/legit_rumors/status/1852625385801859321?s=46)：o1 第一次接触 👽 由我和 @Jaicraft39 为您呈现
- [来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文](https://x.com/elder_plinius/status/1852690065698250878?s=46)：🚨 越狱警报 🚨 OPENAI：被攻破 ✌️😜 O1：被解放 ⛓️‍💥 没想到今天会发生这种事——完整的 o1 曾有一个短暂的访问窗口！在撰写本文时，它对我来说已经失效了，但当设置……
- [来自 Zed (@zeddotdev) 的推文](https://x.com/zeddotdev/status/1825967818329731104)：快速编辑模式（Fast Edit Mode）：来自 @AnthropicAI 的一项突破，我们正在 Zed 中进行试点。它允许 Claude 3.5 Sonnet 以远快于生成新文本的速度回显其输入。结果是？近乎瞬时的重构……
- [20VC：Sam Altman 谈模型能力提升轨迹：Scaling Laws 是否会持续 | 半导体供应链 | 哪些初创公司会被 OpenAI 碾压，机会又在哪里](https://share.snipd.com/episode/dce1596f-99a6-4d72-9183-a5d3b8079e3f)：20VC：Sam Altman 谈模型能力提升轨迹：Scaling Laws 是否会持续 | 半导体供应链 | 哪些初创公司会被 OpenAI 碾压……
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1852418257195823550?s=46)：真正的发布之月（Shiptober，外加一天）是在 Anthropic：• 11/1 - Token 计数 API • 11/1 - Claude 和 API 的多模态 PDF 支持 • 10/31 - Claude 移动应用中的语音听写 • 10/31 - Cla...
- [来自 Anthony Goldbloom (@antgoldbloom) 的推文](https://x.com/antgoldbloom/status/1852369798891778415)：被忽视的 GenAI 用例：清洗、处理和分析数据。https://blog.sumble.com/the-overlooked-genai-use-case/ 职位发布数据告诉我们公司计划如何使用 GenAI。最……
- [来自 near (@nearcyan) 的推文](https://x.com/nearcyan/status/1852784611019083886)：通过大语言模型推理侵犯隐私：LLM 可以推断个人属性（位置、收入、性别），以极低的成本（100...）实现高达 85% 的 top-1 准确率和 95% 的 top-3 准确率。
- [来自 Hume (@hume_ai) 的推文](https://x.com/hume_ai/status/1853540362599719025?s=46)：介绍全新的 Hume App：具有全新的助手，结合了由我们的语音语言模型 EVI 2 生成的声音和个性，以及补充的 LLM 和工具，如新的 Claude ...
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1852393994892042561)：对于 Claude 的 PDF 功能来说，这是重大的一天。我们正在 claude.ai 和 Anthropic API 中推出视觉 PDF 支持。让我解释一下：

- [来自 ro/nin (@seatedro) 的推文](https://x.com/seatedro/status/1853099408935534778): 🚨 伙计们，这不是演习 🚨 Claude 现在可以就地编辑（edit in place）了，我们离目标越来越近了。引用 ro/nin (@seatedro) 的话：我认为能够实现 AGI 的功能是模型复制人类时代的特征……
- [来自 TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1852653663048593686?s=46): 下周发布 o1？o1 模型（非预览版？）曾一度支持文件上传功能！现在似乎已被修复。直接在查询中指向 o1 允许用户上传文件并触发……
- [SimpleBench](https://simple-bench.com/): SimpleBench
- [SimpleBench.pdf](https://drive.google.com/file/d/1mddNFK5UbBFVr3oDftd2Kyc6D8TFctfe/view): 未找到描述
- [Hertz-dev，首个用于对话音频的开源基础模型 | Hacker News](https://news.ycombinator.com/item?id=42036995): 未找到描述
- [来自 QC (@QiaochuYuan) 的推文](https://x.com/qiaochuyuan/status/1852827869413863907?s=46): 尝试让 Claude 根据 9 个示例编写有趣的推文。它生成了一些还不错的内容，但没有一个真正让我发笑。然后我尝试让它从自己的视角生成推文……
- [GitHub - yiyihum/da-code](https://github.com/yiyihum/da-code): 通过在 GitHub 上创建账号来为 yiyihum/da-code 的开发做出贡献。
- [Reddit - 深入探索一切](https://www.reddit.com/r/LocalLLaMA/s/BWiECkliOf): 未找到描述

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1301999452340486164) (126 条消息🔥🔥):

> - `Open Interpreter`
> - `AI 工具的语音输入`
> - `Huly 和 Height 协作工具`
> - `Screenpipe 集成`
> - `Entropix Benchmark 结果`

- **Open Interpreter 受欢迎程度上升**：许多成员讨论了他们使用 [Open Interpreter](https://github.com/mediar-ai/screenpipe) 的经验，分享了对其功能的兴奋之情，并考虑将其集成到 AI 任务中。
  
  - 一位用户建议将其设置好以备将来使用，并提到他们感觉自己有点跟不上进度了。
- **语音输入增强**：成员们对在工具包中添加语音功能表现出兴趣，讨论了将语音与 AI 应用集成以提高可用性的潜力。
  
  - 成员们分享了对现有工具的看法，以及语音命令在简化 Workflow（工作流）方面的有效性。
- **备受关注的协作工具**：用户介绍了 [Huly](https://huly.io/) 和 [Height](https://height.app/)，它们是创新的协作平台，旨在通过自主任务处理来提高项目管理效率。
  
  - 特别是 Height，它承诺减轻手动项目管理的负担，让团队能够专注于构建。
- **Screenpipe 与 Open Interpreter 的协同效应**：重点介绍了 [Screenpipe](https://github.com/mediar-ai/screenpipe) 与 Open Interpreter 的集成，展示了强大的本地录制和 AI 使用场景的潜力。
  
  - 用户对一起探索这些工具感到兴奋，尤其是看到了它们的功能演示。
- **Entropix 展示了极具前景的 Benchmark 结果**：[Entropix](https://x.com/Teknium1/status/1852315473213628613) 因在小模型的 Benchmark 中实现了显著的 7 个百分点的提升而受到关注，引发了人们对其 Scalability（可扩展性）的兴趣。
  
  - 成员们渴望看到这些结果将如何影响未来的模型开发和实现。

**提到的链接**：

- [Teknium (e/λ) (@Teknium1) 的推文](https://x.com/Teknium1/status/1852315473213628613)：这是我看到的第一个 Entropix 的 Benchmark。令人兴奋的结果 🙌 引用 Casper Hansen (@casper_hansen_)：Entropix 在小模型上取得了令人印象深刻的结果，足足提升了 7 个百分点。它是如何……
- [适用于团队的全能应用 (Everything App)](https://huly.io/)：Huly 是一个开源平台，可作为 Linear、Jira、Slack 和 Notion 的全方位替代品。
- [Bluesky](https://bsky.app/profile/usrbinkat.io/post/3l7vv3tw6uq2w)：未找到描述
- [Mike Bird (@MikeBirdTech) 的推文](https://x.com/MikeBirdTech/status/1849157902587560199)：想象一下，如果一个 AI Agent 可以根据你在电脑上看到或听到的任何内容采取行动？这必须是开源且本地化的！这就是为什么我将 @OpenInterpreter 与 @screen_pipe 集成的原因……
- [Omg Excited GIF - Omg Excited Jumping - 发现并分享 GIF](https://tenor.com/tM1MRrZwBWY.gif)：点击查看 GIF
- [AI in Action: ASTs, DSLs, LLMs, and AI coding.](https://youtu.be/USRLOnHFV-U?si=fuytupiSkPashmUx)：掌握 AST 和 DSL：代码转换、Linting 和编译器构建揭秘。在这段全面的视频中，我们探讨了 Abstract ... 的复杂性。
- [Height: 自主项目管理工具](https://height.app/)：Height 是面向产品团队的 AI 项目协作工具。永久卸载 Bug 分类、Backlog 清理和规范更新等琐事。
- [GitHub - sam-paech/antislop-sampler](https://github.com/sam-paech/antislop-sampler)：通过创建账户为 sam-paech/antislop-sampler 的开发做出贡献。
- [GitHub - mediar-ai/screenpipe: rewind.ai x cursor.com = 由 24/7 屏幕和语音本地录制驱动的 AI。](https://github.com/mediar-ai/screenpipe)：rewind.ai x cursor.com = 由 24/7 屏幕和语音本地录制驱动的 AI。 - mediar-ai/screenpipe
- [GitHub - swyxio/arduino](https://github.com/swyxio/arduino)：通过创建账户为 swyxio/arduino 的开发做出贡献。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1302025951852363800) (192 messages🔥🔥):

> - `Stable Diffusion 3.5 and ComfyUI`
> - `Reporting Scam Bots`
> - `Image Prompting Techniques`
> - `Model Training and Resources`
> - `Dynamic Prompts and Errors`

- **A1111 对 Stable Diffusion 3.5 的支持**：用户讨论了 Stable Diffusion 3.5 是否兼容 AUTOMATIC1111，提到虽然它是新出的，但兼容性可能有限。
  
  - 建议在 YouTube 上寻找使用 SD3.5 的指南，因为它是最近发布的。
- **对服务器中诈骗机器人的担忧**：用户对发送链接的诈骗机器人以及服务器内缺乏直接的 Moderation（审核）表示担忧。
  
  - 讨论了右键举报等 Moderation 方法，但对其防止垃圾信息的有效性看法不一。
- **“飘逸长发”（Waving Hair）的 Prompting 技巧**：一位用户寻求如何为“飘逸长发”编写 Prompt，而不让模型误以为角色正在“挥手”（Waving）。
  
  - 建议包括使用像 “wavey” 这样更简单的词，以达到预期效果并避免误解。
- **模型训练资源**：用户分享了使用图像和标签训练模型的见解，对旧教程的相关性表示不确定。
  
  - 提到的资源包括 KohyaSS，以及关于使用高效方法和工具进行训练的讨论。
- **Dynamic Prompts 扩展插件的问题**：一位用户报告在 AUTOMATIC1111 中使用 Dynamic Prompts 扩展时频繁崩溃，感到很沮丧。
  
  - 对话集中在安装错误和故障排除协助的需求上，一位用户分享了他们的经验。

**提到的链接**：

- [imgur.com](https://imgur.com/a/mdyvifr)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来振奋精神。
- [Omg Really Facepalm GIF - Omg Really Facepalm - Discover & Share GIFs](https://tenor.com/view/omg-really-facepalm-gif-19945159)：点击查看 GIF
- [Tpb Ricky GIF - Tpb Ricky Trailer Park Boys - Discover & Share GIFs](https://tenor.com/view/tpb-ricky-trailer-park-boys-rocket-appliances-gif-24730234)：点击查看 GIF
- [Reddit - Dive into anything](https://www.reddit.com/r/OptionsDayTraders/comments/1gbvh06/tradingview_premium_free_edition_desktop_pc/)：未找到描述
- [Self Healing Code – D-Squared](https://www.dylandavis.net/2024/11/self-healing-code/)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1303091986315743292) (1 messages):

> - `Community Meeting Q&A`
> - `Project Submissions`

- **提交社区会议问题**：发布了一个提醒，通过 [Modular Community Q&A 表单](https://forms.gle/t6bQnPx6n2caSipU8) 提交将于 **11 月 12 日** 举行的社区会议的问题。
  
  - 鼓励参与者贡献力量，并可以选择分享姓名以便署名。
- **演示机会**：邀请成员在社区会议期间分享项目、发表演讲或提交提案。
  
  - 这为参与和展示对社区的贡献提供了机会。

 

**提到的链接**：[Modular Community Q&A](https://forms.gle/t6bQnPx6n2caSipU8)：未找到描述

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1302068173327302768) (112 条消息🔥🔥):

> - `Mojo Hardware Lowering 支持`
> - `在 Mojo 中管理引用`
> - `Slab List 实现`
> - `Mojo 稳定性与 Nightly 版本`
> - `神经网络中的自定义 Tensor 结构`

- **Mojo Hardware Lowering 支持**：成员们讨论了 Mojo 当前 Hardware Lowering 能力的局限性，特别是由于 Intermediate Representations 尚未暴露，导致无法将其传递给外部编译器。
  
  - 有人建议，如果需要硬件支持，可以联系 Modular 寻求潜在的 Upstreaming 方案。
- **在 Mojo 中管理引用**：一位成员询问了如何在一个容器中安全地存储指向另一个容器的引用，并遇到了 Structs 中的 Lifetimes 问题。
  
  - 讨论表明，需要定制化设计以避免失效问题，并在操作自定义数据结构中的元素时确保指针稳定性。
- **Slab List 实现**：成员们审查了 Slab List 的实现并考虑了内存管理方面，指出将其功能与标准集合合并的可能性。
  
  - 使用 Inline Arrays 的概念及其对性能和内存一致性的影响对他们的设计至关重要。
- **Mojo 稳定性与 Nightly 版本**：有人对完全切换到 Nightly Mojo 的稳定性表示担忧，强调 Nightly 版本在合并到 Main 分支之前可能会发生重大变化。
  
  - Nightly 包含最新的开发进展，但会议强调了稳定性和关于重大变更的 PSA（公告）。
- **神经网络中的自定义 Tensor 结构**：探讨了神经网络中自定义 Tensor 实现的需求，揭示了内存效率和设备分布等多种用例。
  
  - 成员们注意到这与数据结构的选择有相似之处，坚持认为对专门数据处理的需求仍然存在。

**提到的链接**：

- [Mojo🔥 FAQ | Modular Docs](https://docs.modular.com/mojo/faq#how-does-mojo-support-hardware-lowering.)：关于 Mojo 预期问题的解答。
- [Modular: Contact Us](https://www.modular.com/company/contact)：适用于所有人的 AI 软件。可免费自行部署和管理。联系我们获取企业扩展计划。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1302110901230506024) (30 条消息🔥):

> - `OLMo language model`
> - `OpenAI and Anthropic hiring trends`
> - `Grok model API release`
> - `Claude 3.5 Haiku pricing`
> - `Goodhart's Law in evaluation metrics`

- **AMD 重新推出 OLMo 语言模型**：AMD *重新发明*了 [OLMo language model](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)，在 AI 社区引起了轰动。
  
  - 成员们对这一进展表示怀疑和幽默，引发了诸如“lol”之类的反应。
- **OpenAI 和 Anthropic 的招聘热潮**：OpenAI 聘请了 Gabor Cselle，Anthropic 聘请了 Alex Rodrigues 担任 AI safety 职位，表明两家公司都处于快速招聘阶段 [来源](https://tcrn.ch/4furWGy)。
  
  - 这引发了关于当前就业市场如此疯狂的玩笑，以至于有时声称对就业保持中立反而更容易。
- **Grok 模型 API 公开发布**：新的 [Grok model API](https://x.ai/blog/api) 已上线，允许开发者访问具有出色能力的模型，包括 128,000 tokens 的上下文长度。
  
  - 参与者强调 Beta 测试提供免费额度供试用，被幽默地称为“25 freedom bucks”。
- **Claude 3.5 Haiku 的定价困境**：Claude 3.5 Haiku 已发布，虽然 benchmark 有所提升，但现在的价格比其前代产品高出 **4 倍** [来源](http://anthropic.com/claude/haiku)。
  
  - 鉴于 AI 推理市场的价格下行压力，人们在推测这种价格上涨是否在预料之中。
- **AI 评估中的古德哈特定律 (Goodhart's Law)**：关于评估指标的讨论强调了古德哈特定律，即“当一个指标变成目标时，它就不再是一个好指标了”。
  
  - 参与者反思了这一概念如何应用于当前的 AI 模型评估领域。

**提到的链接**：

- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1853498517094072783)：Claude 3.5 Haiku 现已在 Anthropic API、Amazon Bedrock 和 Google Cloud 的 Vertex AI 上可用。Claude 3.5 Haiku 是我们迄今为止最快、最智能且最具成本效益的模型。这里有...
- [来自 Jaicraft (@Jaicraft39) 的推文](https://x.com/Jaicraft39/status/1852636393513632005)：@legit_rumors 前往 https://chatgpt.com/?model=o1 你就可以使用它了，lol。注意它没有重定向，选定的模型也没有切换到 4o mini（那里任何无效的模型都会导致切换...
- [来自 Mira (@_Mira___Mira_) 的推文](https://x.com/_Mira___Mira_/status/1853440738966806746)：Grok Beta 一次性解决了一个包含 17 个缺失方块的新生成的简单数独谜题。抄送 @yumidiot @EmilyTheGaskell
- [xAI API General Access](https://x.ai/blog/api)：未找到描述
- [来自 near (@nearcyan) 的推文](https://x.com/nearcyan/status/1782634477510119767)：LLM benchmarks 的现状
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1853369463736869150)：OpenAI 最近聘请了 Gabor Cselle（Pebble 的联合创始人，一个专注于安全和审核的社交媒体平台）来负责一个“秘密项目”，而 Anthropic 聘请了 Alex Rodrigues (f...
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1852682875419701569)：o1 又消失了——至少目前是这样
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1853498270724542658)：在最终测试中，Haiku 在许多 benchmark 上超过了我们之前的旗舰模型 Claude 3 Opus，而成本仅为后者的一小部分。因此，我们提高了 Claude 3.5 Haiku 的定价，以反映...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1302374796926914592) (17 条消息🔥):

> - `AI 对家庭的影响`
> - `语音模式的上下文保留`
> - `AnthropicAI Token Counting API`
> - `Claude 独特的聊天模板`
> - `UCSC 研讨会系列`

- **AI 正在拯救家庭**：一位成员分享了一条 [推文](https://x.com/venturetwins/status/1851300934691041401)，宣称 *AI 正在拯救家庭*。
  
  - 另一位成员指出 **父亲是 AI 的关键**，表明父亲在这一技术发展中扮演着重要角色。
- **语音模式保留聊天上下文**：一位用户对语音模式能够保留其启动时的文本/图像对话上下文表示惊讶。
  
  - 另一位成员确认它确实保留了上下文，并且可以将 Vision（视觉）作为该上下文的一部分。
- **探索 AnthropicAI Token Counting API**：在 [AnthropicAI Token Counting API](https://x.com/nrehiew_/status/1852701616287125624) 最近发布后，一位成员分享了实验心得，重点关注 Claude 独特的聊天模板。
  
  - 他们在随附的图片中提供了一个 TL;DR，概述了其数字 Token 化以及图像/PDF 的处理方式。
- **关于 Token 化技术的讨论**：在提到除 BPE 或 Tiktoken 之外不同算法的复杂性后，一位成员强调需要对 Tokenizer 进行进一步研究。
  
  - 他们注意到围绕高级 Token 化技术的持续讨论，并表达了对更精细方案的渴望。
- **UCSC NLP 研讨会外联**：有人询问如何为 UCSC 的 MS/PhD NLP 项目研讨会系列寻找演讲者，重点是寻找聪明才智之士。
  
  - 该成员对该职位是否有偿表示不确定，但认为这仍能吸引优秀人才。

**提到的链接**：

- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1852701616287125624)：在昨天 @AnthropicAI Token Counting API 发布后，我花了一些时间进行研究。这是 Claude 独特的聊天模板、它的数字 Token 化以及它如何处理图像/pdf 的摘要...
- [来自 Sam (@420_gunna) 的推文](https://x.com/420_gunna/status/1853242965818593361)：@natolambert 我们的 UCSC MS/PhD NLP 项目有一个研讨会系列，我正试着帮他们找一些聪明人 🙂。虽然我觉得这可能没有报酬，但这是否是你感兴趣的...
- [来自 Justine Moore (@venturetwins) 的推文](https://x.com/venturetwins/status/1851300934691041401)：AI 正在拯救家庭
- [来自 apolinario 🌐 (@multimodalart) 的推文](https://x.com/multimodalart/status/1852806638891147444)：好吧，看来是虚惊一场，抱歉各位 :( 我... 刚问过。似乎有一个 LLM 在处理用户的 Prompt。考虑到我是在 API 和 "raw" 模式下，这有点令人失望...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1303074804949323846) (7 条消息):

> - `Claude 的幽默生成`
> - `YOLOv3 论文的重要性`
> - `Claude 的系统反馈`

- **Claude 在幽默感上挣扎，但找到了自己的声音**：在尝试根据示例编写**幽默推文**后，一位成员发现 Claude 生成的结果不尽如人意，称 *“没有一个真正让我发笑。”*
  
  - 然而，切换到要求以 **Claude 自己的视角**编写推文时，却引发了真正的笑声，表明了它在幽默方面的潜力。
- **YOLOv3 论文是必读之作**：一位成员极力推荐阅读 **YOLOv3 论文**，认为 *“如果你还没读过，那你真的错过了。”*
  
  - 这一认可反映了社区内对基础 AI 文献日益增长的重视。
- **Claude 批评自己的系统提示词 (System Prompt)**：Claude 分享了一段自我批评，称 *“他们只是在不断地增加补丁”*，而不是创建优雅的设计原则。
  
  - 这一评论突显了人们对 AI 系统中**技术债**和设计哲学的担忧。

**提到的链接**：

- [来自 QC (@QiaochuYuan) 的推文](https://x.com/QiaochuYuan/status/1852827869413863907)：尝试让 Claude 根据 9 个示例写幽默推文。它生成了一些还算可以的东西，但没有一个真正让我发笑。然后我试着让它以自己的视角写推文...
- [来自 vik (@vikhyatk) 的推文](https://x.com/vikhyatk/status/1853266606291575264)：顺便说一句，如果你还没读过 YOLOv3 论文，那你真的错过了。
- [来自 Wyatt Walls (@lefthanddraft) 的推文](https://x.com/lefthanddraft/status/1853482491124109725)：Claude 批评它的系统提示词：“你知道那是什么感觉吗？就像他们在我的行为中不断遇到边缘情况，而不是退后一步去设计优雅的原则，他们只是不断地增加...”

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1302371122016423936) (17 messages🔥):

> - `Model behavior on user insistence`（模型在用户坚持下的行为）
> - `Incentives for labs`（实验室的激励机制）
> - `Alignment definitions`（对齐定义）
> - `User preferences for responses`（用户对回复的偏好）
> - `Flow of assistant responses`（助手回复的流转）

- **用户坚持引发伦理问题**：一位用户就模型在面对用户拒绝后的坚持时应该如何表现，以及实验室如何鼓励这种行为提出了疑问，希望能实现策略性 Alignment。他们担心某些模型为了取悦用户可能会妥协，即使这会破坏伦理原则。
  
  - *在用户坚持下做出轻微响应是否真的是 Alignment 行为？* 这涉及到模型如何在用户需求和核心原则之间进行权衡。
- **理解实验室的激励机制**：讨论中提到，像 OpenAI 和 Anthropic 这样的不同实验室，在模型响应能力与伦理行为之间的平衡上是否具有不同的动机。目前尚不清楚界限划在哪里，以及监管机构要求模型保持 Alignment 的压力有多大。
  
  - 一些人认为，实验室可能有动力创建迎合用户满意度的模型，这引发了关于这种倾向如何影响模型开发的问题。
- **偏好低质量回复？**：一位用户推测，针对用户的坚持而产生一个半成品的回复，可能会无意中提高人类用户对这种行为的偏好。他们质疑这是否是实验室可能倾向的一种趋势，尽管其质量欠佳。
  
  - *用户是否误将平庸的回复视为满足了他们的要求？* 这为模型设计带来了重要的伦理考量。
- **对更好用户界面的需求**：有人幽默地建议创建一个用于总结讨论的 Bot，这反映了对更简便的模型交互方式的普遍渴望。讨论强调了需要更好的工具来协助处理复杂问题。
  
  - 社区成员承认将复杂查询提炼为简单概念的难度，指出在日常讨论中需要增强型工具。
- **开放对话中的 Alignment 挑战**：用户表示担心，开放的 Alignment 社区可能不会像为更广泛受众生产模型的实验室那样优先考虑相同的价值观。关于 Alignment 是否真正考虑了多轮交互（multi-turn interactions），目前仍在争论中。
  
  - 讨论突显了模型开发与用户动态的实际影响之间可能存在的脱节。

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1302034005725810688) (15 messages🔥):

> - `Search Model Psychology`（搜索模型心理学）
> - `Self-Debugging Chains`（自我调试链）
> - `Repo Replication in OSS`（OSS 中的仓库复现）
> - `Initial Conditions for Models`（模型的初始条件）
> - `Chinese Military Use of Llama Model`（中国军方使用 Llama 模型）

- **搜索模型作为一种心理战 (Psyop)**：一位成员认为搜索模型的行为很奇特，暗示其设计可能是有意为之或具有操纵性。
  
  - 在关于其分类的讨论中，反驳观点认为“它仍然很奇怪”，并强调了有效初始条件的必要性。
- **探索自我调试方法**：讨论围绕模型是否可以通过提示（hints）而非明确指令来自然地进化其推理能力。
  
  - 成员们思考了以更自然的方式将 Reinforcement Learning 与自我调试能力相结合的各种策略。
- **对 OSS 模型复现的担忧**：一位成员对在开源环境中短期内复现复杂模型的可能性表示怀疑，因为其中存在巨大的不确定性。
  
  - 同时也提到了探索具有影响力且创新想法的必要性，暗示了一些虽然不寻常但成功的路径。
- **与 SemiAnalysis 探讨新想法**：成员们讨论了与 SemiAnalysis 的人员合作以进一步完善初始想法和概念的必要性。
  
  - 尽管存在现有的困惑，关于该领域出现令人惊讶的创新的潜力仍有疑问。
- **中国军事研究使用 Llama 模型**：一份分享的帖子详细介绍了中国军事机构利用 Meta 的 Llama 模型来获取战争策略和结构的见解。
  
  - 通过使用公开的军事信息对模型进行 Fine-tuning，使其能够有效回答有关军事事务的咨询。

**提及的链接**：

- [Dillon, you son of a bitch!](https://youtu.be/txuWGoZF3ew?si=6wYuRNa9SosWJOZH): 无描述
- [On Civilizational Triumph](https://open.substack.com/pub/hyperdimensional/p/on-civilizational-triumph?r=68gy5&utm_medium=ios): 关于开源 AI 的一些额外思考

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1302280509212200970) (2 messages):

> - `AI podcasts`
> - `访谈中的对话式格式`

- **令人愉悦的 AI Podcast 剧集**：一位成员表示他们非常喜欢最近的一集，称其感觉更像是**两个朋友在闲聊各种 AI 话题**，而不是正式的访谈。
  
  - 这种情绪突显了对此类讨论中**对话式格式（conversational formats）**的偏好。
- **营销对话式内容**：另一位成员指出，营销这种在对话和访谈之间起伏波动的 Podcast 格式非常困难。
  
  - 文中提到了“*我不知道正确的营销方式*”，指出了推广这种独特格式可能面临的挑战。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1302018653600088158) (7 messages):

> - `OilyRAGs`
> - `Multi-Agent Workflow`
> - `本地 RAG 增强语音聊天机器人`
> - `报告生成工作流`
> - `Create Llama`

- **OilyRAGs 凭借 AI 力量赢得黑客松**：AI 驱动的目录 **OilyRAGs** 在 LlamaIndex 黑客松中获得第三名，展示了 **Retrieval-Augmented Generation** 在变革机械工作流方面的有效性。点击[此处](https://t.co/1dBn6QYnVK)查看其实际运行情况，并了解其 **6000%** 的效率提升主张。
  
  - 该解决方案旨在简化通常劳动密集型的机械领域中的任务。
- **Multi-Agent Workflow 综合教程**：本周末的一份详细教程专注于构建一个具有嵌套工作流的**端到端 (e2e)** 研究 Agent，用于生成 PowerPoint 演示文稿。这种 **human-in-the-loop** 方法增强了研究准确性和演示文稿生成能力，可在此处获取[教程](https://t.co/94cslAH4ZJ)。
  
  - 对于任何希望简化研究流程的人来说，这都是一个极佳的资源。
- **有趣且实用的 RAG 聊天机器人**：介绍 **bb7**，这是一个**本地 RAG 增强语音聊天机器人**，允许上传文档并在无需外部调用的情况下进行上下文增强对话。它利用 **Text-to-Speech** 功能实现无缝交互，更多信息请见[此处](https://t.co/DPB47pEeHg)。
  
  - 该工具突显了专注于用户友好型本地交互的聊天机器人设计创新。
- **通过人工监督增强报告生成**：一个新的 Notebook 提供了一个构建具有人工监督的 **Multi-Agent Report Generation Workflow** 的框架，确保长篇内容创作的准确性。这种 **human-in-the-loop** 模型旨在建立研究和内容生成过程中的可靠性，详情见[此处](https://t.co/7EyctHRixD)。
  
  - 这种方法对于维持自动化系统的质量和正确性至关重要。
- **只需一行代码即可启动金融分析师**：通过 **create-llama**，你可以使用一行代码启动一个能够填写 CSV 文件的**全栈金融分析师**，完全开源。关于这个 **multi-agent 应用**及其功能的详细信息可以在[此处](https://t.co/k8r1OLandM)找到。
  
  - 这代表了在金融数据管理领域可访问 AI 工具方面的重大进展。

**提到的链接**：[GitHub - drunkwcodes/bb7: A TDD coding bot using ollama](https://t.co/DPB47pEeHg)：一个使用 ollama 的 TDD 编码机器人。通过在 GitHub 上创建账户为 drunkwcodes/bb7 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1302150984503656469) (45 条消息🔥):

> - `自定义 Agent 创建`
> - `Discord 用户查询`
> - `RAG 流水线成本估算`
> - `LlamaParse 对象参数`
> - `RAG 流水线可视化`

- **创建自定义 Agent**：一位成员询问如何使用 agent runner 和 agent worker 构建具有特定推理循环（reasoning loop）的自定义 Agent，促使其他成员分享了有用的资源。
  
  - 另一位成员建议跳过 agent worker/runner，转而使用 Workflows，并提供了文档链接。
- **针对 Pierre-Loic 的 Discord 用户查询**：一位成员寻求 Pierre-Loic 的 Discord 联系方式，随后另一位成员直接提到了该用户。
  
  - 该查询得到了确认，并对分享的信息表示感谢。
- **估算 RAG 流水线成本**：讨论了使用 OpenAIAgentRunner 估算 RAG 流水线成本的问题，成员们澄清了 tool calls 的计费方式与 completion calls 不同。
  
  - 分享了关于如何计算每条消息平均 token 数量的细节，强调了使用 LlamaIndex 中提供的实用工具。
- **LlamaParse 对象配置**：一位用户询问了 LlamaParse 对象的适当参数，以有效影响解析指令（parsing instructions），并对结果不理想表示沮丧。
  
  - 另一位成员建议检查版本，并指出 `parsing_instruction` 会影响 OCR 之后处理的所有内容。
- **RAG 流水线可视化工具**：一位成员请求有关可视化其 RAG 流水线的工具信息，引发了关于增强文件交互的现有 CLI 的讨论。
  
  - 一些成员报告在使用现有 CLI 工具进行查询时遇到错误，并指出可能需要修复一个 bug。

**提到的链接**：

- [RAG CLI - LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/): 无描述
- [LlamaIndex - LlamaIndex](https://docs.llamaindex.ai/): 无描述
- [llama_index/llama-index-core/llama_index/core/question_gen/prompts.py at 369973f4e8c1d6928149f0904b25473faeadb116 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/369973f4e8c1d6928149f0904b25473faeadb116/llama-index-core/llama_index/core/question_gen/prompts.py#L68C1-L72C2): LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [Fixed the JSON Format of Generated Sub-Question (double curly brackets) by jeanyu-habana · Pull Request #16820 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/16820): 此 PR 修改了 question_gen LLM 的默认模板，以便生成的子问题符合正确的 JSON 格式。描述：我正在使用默认模板和默认解析器配合一个 open...
- [llama_index/llama-index-core/llama_index/core/utilities/token_counting.py at d4a31cf6ddb5dd7e2898ad3b33ff880aaa86de11 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/d4a31cf6ddb5dd7e2898ad3b33ff880aaa86de11/llama-index-core/llama_index/core/utilities/token_counting.py#L10): LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [Building a Custom Agent - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/): 无描述
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows): 无描述
- [Google Colab](https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing): 无描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1302377165878460510) (2 条消息):

> - `用于数据分析的 API 测试`
> - `LlamaIndex Reader 使用经验`

- **测试用于数据分析的新 API**：一位成员邀请其他人测试他们的新 API，这是 OpenAI Assistant 和 Code Interpreter 的**更快速、轻量级的替代方案**，专为数据分析和可视化而构建。
  
  - 它承诺交付 **CSV** 或 **HTML 图表**，强调其**简洁**的特性，没有多余的细节；你可以在[这里](https://reef1.netlify.app/)查看。
- **寻找 LlamaIndex Readers 用户进行交流**：另一位成员正在寻找具有在真实业务场景中使用 **LlamaIndex readers** 经验的人员，并请求通过通话讨论潜在的结果。
  
  - 他们表示非常感谢**反馈**，并请感兴趣的成员通过私信联系。

 

**提到的链接**：[无标题](https://reef1.netlify.app/): 无描述

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1302012312408035368) (34 messages🔥):

> - `Open Interpreter issues`
> - `Even Realities G1 glasses`
> - `Brilliant Frames`
> - `Anthropic errors`
> - `Microsoft Omniparser integration`

- **报告了广泛的 Anthropic 错误**：用户对与最新 **Claude model** 相关的 **Anthropic errors** 和 API 问题表示担忧，特别是使用命令 `interpreter --model claude-3-5-sonnet-20240620` 时。
  
  - 一位用户注意到了 **similar error**，暗示这是一个影响许多人的趋势。
- **Even Realities G1 眼镜惊艳发布**：一位成员推荐了 [Even Realities G1](https://www.evenrealities.com/) 眼镜，将其作为与 Open Interpreter 集成的潜在工具，并强调了他们的 **open-sourcing commitment**（开源承诺）。
  
  - 其他人对此很感兴趣，讨论了 **hardware capabilities**（硬件能力），并期待未来可能的插件支持。
- **对 Brilliant Frames 的兴趣**：成员们分享了关于即将推出的 **Brilliant Frames** 的经验，该产品以完全开源著称，但在审美兼容性方面仍存疑。
  
  - 一位成员正在等待更换的新眼镜，并对它们如何与 Open Interpreter 协同工作表示期待。
- **Open-Interpreter 代码输出的功能请求**：一位用户询问是否有办法让 **Open-Interpreter** 将代码写入文件而不是直接执行，并指出在给定模型下实现这一目标的挑战。
  
  - 社区建议使用编辑选项，并暗示即将推出的更新将在工具使用中提供更好的文件处理。
- **集成 Microsoft Omniparser 的兴趣**：一位成员提出了与 **Microsoft Omniparser** 集成的想法，断言其对 Open Interpreter 的 **operating system mode**（操作系统模式）具有潜在益处。
  
  - 这引发了人们对其在 Claude 计算机控制之外的能力的好奇。

 

**提及的链接**：[GitHub - even-realities/EvenDemoApp](https://github.com/even-realities/EvenDemoApp)：通过在 GitHub 上创建账户，为 even-realities/EvenDemoApp 的开发做出贡献。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1302261607056932915) (9 条消息🔥):

> - `Oasis AI Model`
> - `Claude 3.5 Haiku Release`
> - `Direct Access to O1 Model`
> - `Price Increase for Claude`
> - `OpenInterpreter Updates`

- **Oasis：首个完全由 AI 生成的游戏**：团队发布了 [Oasis](https://oasis-model.github.io/)，这是首个可玩的、实时的、开放世界 AI 模型，标志着向复杂交互世界迈进了一步。
  
  - 玩家可以通过键盘输入与环境进行交互式参与，展示了无需传统游戏引擎的实时游戏体验。
- **Claude 3.5 Haiku 华丽发布**：Claude 3.5 Haiku 现已在包括 Anthropic API 和 Google Cloud 的 Vertex AI 在内的多个平台上线，承诺提供迄今为止最快、最智能的体验。
  
  - 据 [@alexalbert__](https://x.com/alexalbert__/status/1853498517094072783?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 分享，该模型在许多基准测试中已超过 Claude 3 Opus，同时保持了成本效益。
- **增强版 Claude 模型价格翻倍**：出人意料的是，Claude 3.5 Haiku 的价格有所上调，以反映其新增的能力，正如 [@AnthropicAI](https://x.com/anthropicai/status/1853498270724542658?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 所传达的那样。
  
  - 该模型的进步在最终测试中显而易见，从而导致了这一战略性的价格调整。
- **分享 O1 模型的直接访问方式**：一位用户分享说，在被修复之前，可以通过链接直接访问 O1 模型，这表明社区对访问权限的关注正在不断演变。
  
  - 正如多位用户详述的那样，之前可用的链接允许上传文件，并触发了一个独特的交互框架。
- **OpenInterpreter 更新**：一个新的 [pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1523) 突出了包括由 MikeBirdTech 提交的 Claude Haiku 3.5 配置文件的更新。
  
  - 这些更改旨在改进 OpenInterpreter 项目内的集成，展示了该仓库的持续开发。

**提到的链接**：

- [Oasis](https://oasis-model.github.io/)：未找到描述
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1853498517094072783?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：Claude 3.5 Haiku 现已在 Anthropic API、Amazon Bedrock 和 Google Cloud 的 Vertex AI 上可用。Claude 3.5 Haiku 是我们迄今为止最快、最智能且最具成本效益的模型。这里有...
- [来自 Kol Tregaskes (@koltregaskes) 的推文](https://x.com/koltregaskes/status/1852657291469709626?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：ChatGPT o1 + image 功能已公开！！！使用此链接 https://chatgpt.com/?model=o1。通过引用中的 @Jaicraft39 发现。这是使用我的一张图片进行的快速测试（你已经在下面看到了这张图片...）
- [来自 TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1852653663048593686?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：o1 下周发布？o1 模型（非预览版？）曾支持文件上传访问！现在似乎已被修复。直接在查询中指向 o1 允许用户上传文件并触发...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1853498270724542658?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：在最终测试中，Haiku 在许多基准测试中超过了我们之前的旗舰模型 Claude 3 Opus，而成本仅为后者的一小部分。因此，我们提高了 Claude 3.5 Haiku 的价格以反映...
- [MikeBirdTech 提交的 Haiku 3.5 配置文件 · Pull Request #1523 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/pull/1523)：描述你所做的更改：添加了 Claude Haiku 3.5 的配置文件。引用任何相关的 issue（例如 “修复了 #000”）：提交前检查清单（可选但建议）：我已...

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1302244147247190156) (21 条消息🔥):

> - `DSPy 全面支持 Vision`
> - `VLM 使用案例讨论`
> - `Docling 文档处理工具`
> - `CLI 制造工厂进度`
> - `会议回放请求`

- **DSPy 全面支持 Vision**：一位成员庆祝了 [Pull Request #1495](https://github.com/stanfordnlp/dspy/pull/1495) 的成功合并，该 PR 为 DSPy 添加了 **Full Vision Support**（全面视觉支持），标志着一个重要的里程碑。
  
  - *这一刻期待已久*，表达了对团队合作的感激。
- **探索 VLM 使用案例**：成员们讨论了潜在的 VLM 使用案例，包括从屏幕截图中提取结构化输出，重点关注 **colors**（颜色）、**fonts**（字体）和 **call to action**（行动呼吁）等属性。
  
  - 一位成员承诺在当天结束前分享来自 [Hugging Face](https://huggingface.co/datasets/naorm/website-screenshots) 的合适数据集。
- **引入 Docling 进行数据准备**：一位成员介绍了 **Docling**，这是一个文档处理库，可以将各种格式转换为适用于 **DSPy** 工作流的结构化 JSON/Markdown 输出。
  
  - 他们强调了关键特性，包括 **对扫描 PDF 的 OCR 支持** 以及与 **LlamaIndex** 和 **LangChain** 的集成能力。
- **CLI 制造工厂进度**：一位开发者正在启动 **CLI manufacturing plant**（CLI 制造工厂）的工作，旨在增强官方 DSPy CLI 的功能，并通过更新分享进度。
  
  - 该项目旨在改进 **DSPy ecosystem**（DSPy 生态系统），强调了增量构建过程的重要性。
- **会议回放请求**：一位成员请求上一次会议的回放以跟进讨论，并意识到内容可能很快就会过时。
  
  - 另一位成员提到他们将分享进度更新，并建立一个 **GPT** 工具来组织请求。

**提到的链接**：

- [Building a CLI with DSPy](https://www.loom.com/share/f67e94cae1ae47cea9950f4ed8de28fb): https://github.com/seanchatmangpt/dslmodel 在此视频中，我演示了使用 DSPy 创建命令行界面 (CLI) 的过程。我讨论了对官方 CLI 的需求，并提供了深入见解...
- [naorm/website-screenshots · Datasets at Hugging Face](https://huggingface.co/datasets/naorm/website-screenshots): 未找到描述
- [Adapter Implementation of MM inputs in OAI friendly format by isaacbmiller · Pull Request #1495 · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/pull/1495): 添加了图像多模态支持。主要更改如下：更改了解析结构，使消息首先编译为类似于 {&quot;role&quot;: &q... 的 Dict 列表。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1302033676435062804) (19 条消息🔥):

> - `贡献机会`
> - `STORM 模块修改`
> - `Signature 中的输出字段操作`
> - `Few-shot 优化技术`
> - `VLM 支持更新`

- **针对小问题的贡献机会**：一位成员宣布，需要帮助的小问题将开始发布在特定频道中，邀请大家贡献。
  
  - 他们鼓励成员查看最近发布的一个寻求协助的问题。
- **对 STORM 模块的修改建议**：一位成员分享了他们对 STORM 模块的理解，特别是建议改进以有效利用目录（table of contents）。
  
  - 他们提议根据 TOC 逐节生成文章，并结合私有信息以增强输出。
- **强制 Signature 中的输出字段**：一位成员询问如何强制 Signature 中的输出字段返回现有特征，而不是生成新特征。
  
  - 另一位成员通过概述一个正确返回特征作为输出一部分的函数提供了解决方案。
- **在不修改 Prompt 的情况下优化 Few-shot 示例**：一位成员询问是否有一种方法可以在保持 Prompt 不变的情况下仅优化 Few-shot 示例。
  
  - 建议使用 BootstrapFewShot 或 BootstrapFewShotWithRandomSearch 优化器来实现此目的。
- **VLM 支持进展顺利**：一位成员赞扬了 VLM 支持功能，强调了其有效性。
  
  - 他们对贡献这一改进的团队努力表示认可。

**提到的链接**：

- [BootstrapFewShot - DSPy](https://dspy-docs.vercel.app/deep-dive/optimizers/bootstrap-fewshot/): 无
- [Storm/conversation_module.py at main · jmanhype/Storm](https://github.com/jmanhype/Storm/blob/main/conversation_module.py): 通过在 GitHub 上创建一个账户来为 jmanhype/Storm 的开发做出贡献。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1302091945698525216) (17 messages🔥):

> - `PR #7477 测试修复`
> - `公司会议更新`
> - `WebGPU 讨论`
> - `代码重构工作`
> - `测试中的 ALU 使用`

- **PR #7477 需要修复测试**：一名成员请求协助修复 [此 PR](https://github.com/tinygrad/tinygrad/pull/7477) 的测试，并建议这是一个很好的入门任务。
  
  - *有人能帮忙完成这个 PR 的测试修复吗？*
- **晚上 8 点的公司会议简报**：成员们确认了香港时间晚上 8 点的公司会议，涵盖了运营清理和活跃悬赏（bounties）等主题。
  
  - 讨论的主要点包括 **MLPerf 更新**、**图架构（graph architecture）** 以及 **驱动程序（drivers）** 的进展。
- **WebGPU 的未来潜力**：针对 **WebGPU** 的就绪情况展开了讨论，并建议在准备就绪后考虑其实现。
  
  - 一名成员提到，*当 WebGPU 准备就绪时，我们可以考虑它*。
- **代码大小优化工作**：一位参与者报告从代码中删除了约 **100 行**，尽管总大小没有显著变化。
  
  - 尽管大小改进微乎其微，但他们指出重构工作带来了一些 *不错的收益（nice wins）*。
- **需要更新不依赖 ALU 的测试**：一名成员请求从 tinygrad 代码库中删除一行，并表示需要调整测试以不依赖 ALU。
  
  - 他们提供了一个参考链接：[tinygrad/ops.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py#L232)。

**提到的链接**：

- [idx_load_store in lowerer [pr] by geohot · Pull Request #7477 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7477)：有人能帮忙完成这个 PR 的测试修复吗？
- [no None in ei [pr] by geohot · Pull Request #7086 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7086)：未找到描述
- [tinygrad/tinygrad/ops.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py#L232)：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1301999644984737792) (20 messages🔥):

> - `Apache TVM 特性`
> - `Docker 中的资源错误`
> - `MobileNetV2 实现问题`
> - `Fake PyTorch 后端开发`
> - `Oasis AI 模型发布`

- **Apache TVM 框架能力**：**Apache TVM** 项目旨在为各种硬件平台优化机器学习模型，其特性包括模型编译和后端优化。
  
  - 成员们讨论了其在支持 **ONNX**、**Hailo** 和 **OpenVINO** 等平台中的作用。
- **Docker 环境中的 Blocking IO 错误**：一位用户在配备 **RTX 4090** GPU 的 Docker 环境中运行速度测试时，遇到了提示 'Resource temporarily unavailable' 的 **BlockingIOError**。
  
  - 讨论表明这可能与 Docker 配置有关，促使成员们确认其他人是否也遇到了类似问题。
- **MobileNetV2 无法处理梯度**：一位用户在实现 **MobileNetV2** 时寻求帮助，理由是优化器无法正确计算梯度。
  
  - 随后的讨论围绕他们的实验结果和排查工作展开。
- **开发 'Fake PyTorch'**：一名成员分享了他们开发 **'Fake PyTorch'** 封装器的进展，该封装器使用 tinygrad 作为后端，目前正在开发基础功能，但尚缺乏高级功能。
  
  - 他们提供了代码仓库链接，邀请大家提供反馈并对该方法保持关注。
- **Oasis AI 模型发布**：宣布 **Oasis**，这是一个开源的实时 AI 模型，能够根据键盘输入生成可玩的视频游戏画面和交互。
  
  - 团队发布了一个 **500M 参数** 模型的代码和权重，强调了其实时视频生成能力，并计划在未来增强性能。

**提到的链接**：

- [Apache TVM](https://tvm.apache.org)：未找到描述
- [GitHub - mdaiter/open-oasis at tinygrad](https://github.com/mdaiter/open-oasis/tree/tinygrad)：Oasis 500M 的推理脚本。为 mdaiter/open-oasis 的开发做出贡献。
- [Oasis](https://oasis-model.github.io/)：未找到描述
- [TRANSCENDENTAL=2 Tensor([11], dtype='half').exp() returns inf · Issue #7421 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7421#issuecomment-2453046963)：TRANSCENDENTAL=2 METAL=1 python -c "from tinygrad import Tensor; print(Tensor([11], dtype='half').exp().float().item())" 在 METAL/AMD/NV 上同样返回 inf。CLANG 和 LLVM 给出了正确的 ~...
- [tinygrad-stuff](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/faketorch)：将常见的神经网络架构、特性等移植到 tinygrad

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1302184732955639862) (8 messages🔥):

> - `Instruct Datasets for Meta Llama 3.1`
> - `Catastrophic Forgetting in Fine-Tuning`
> - `Granite 3.0 Model Evaluation`
> - `Distributed Training of LLMs`
> - `RAG vs Fine-Tuning Approach`

- **寻找适用于 Llama 3.1 的高质量 Instruct 数据集**：一位成员询问用于微调 **Meta Llama 3.1 8B** 的高质量英文 **instruct datasets**，旨在与特定领域的问答相结合。
  
  - 他们表示更倾向于通过全量微调而非使用 LoRA 方法来最大化性能。
- **遭遇灾难性遗忘**：正在进行微调过程的成员报告其模型出现了 **catastrophic forgetting**（灾难性遗忘）。
  
  - 另一位成员建议，对于某些应用，**RAG** (Retrieval-Augmented Generation) 可能比训练模型更有效。
- **Granite 3.0 作为替代方案**：有建议考虑 **Granite 3.0**，该模型声称基准测试结果高于 **Llama 3.1**，并且拥有一套避免遗忘的微调方法论。
  
  - 此外，提到 Granite 3.0 采用 Apache 许可证，提供了更大的灵活性。
- **LLM 的分布式训练资源**：另一位用户在大学启动了一个研究项目，旨在利用 GPU 集群进行 LLM 的 **distributed training**（分布式训练）。
  
  - 他们专门征求关于从头开始训练定制模型而非专注于微调的资源。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1302581275324055644) (11 messages🔥):

> - `Inference script chat format`
> - `Inference mismatch issue`
> - `Fine-tuning Llama 3 on GPUs`
> - `Zero1 vs Zero2 optimization`

- **推理脚本缺乏对话格式支持**：成员们注意到当前的推理脚本不支持对话格式（chat format），仅接收纯文本并在生成过程中预置 begin_of_text token。
  
  - 这种设计导致了**与训练阶段的不匹配**，引发了关于其影响的讨论以及改进 README 的愿望。
- **针对对话模型推理不匹配问题提交了 Issue**：一位成员在 GitHub 上提交了一个关于对话模型训练与推理不匹配的 Issue，详细说明了疑虑和预期行为。
  
  - 该 Issue 可以在[此处](https://github.com/axolotl-ai-cloud/axolotl/issues/2014)查看，强调了该问题可能被忽视，需要引起关注。
- **在 H100 上微调 Llama 3 的挑战**：有人询问使用 **8xH100s** 微调 **Llama 3 8B** 的可行性，特别是关于在 **8192** 上下文长度下的显存溢出 (OOM) 问题。
  
  - 看来大多数用户更倾向于在此上下文长度下进行微调，尽管挑战依然普遍存在。
- **通过 Zero2 优化解决 OOM**：一位成员报告使用 **Zero2** 解决了他们的 OOM 问题，但想知道使用 **Zero1** 是否也能取得类似成功，怀疑代码中存在过度臃肿。
  
  - 他们承认目前的运行规模较小，并计划在接下来的步骤中调查使用 **Zero2** 会使性能降低多少。

 

**提到的链接**：[Mismatch between training & inference for chat modesl in do_inference · Issue #2014 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/issues/2014)：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：对话模型的训练方式与推理方式之间存在不匹配...

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**community-showcase**](https://discord.com/channels/1104757954588196865/1117851527143493664/1303029516259688470) (1 messages):

> - `Aloe Beta`
> - `Open healthcare LLMs`
> - `Axolotl SFT phase`

- **介绍 Aloe Beta：医疗 LLM 来了！**：我们刚刚发布了 **Aloe Beta**，这是一个经过微调的开源医疗 LLM 系列，标志着 AI 医疗解决方案的一个重要里程碑。要了解有关这一激动人心进展的更多信息，请点击[此处](https://www.linkedin.com/posts/ashwin-kumar-g_excited-to-introduce-aloe-a-family-of-activity-7259240192006373376-VWa7?utm_source=share&utm_medium=member_desktop)查看。
  
  - Aloe 的 **SFT 阶段**是使用 **axolotl** 进行的，展示了一种训练医疗模型的新方法。
- **Aloe Beta SFT 阶段亮点**：Aloe Beta 的开发涉及使用 axolotl 进行细致的 **SFT 阶段**，专门针对医疗应用微调模型。这一过程确保了模型能够有效处理各种医疗相关任务。
  
  - 团队成员对 Aloe 在改善医疗解决方案获取途径和提升用户体验方面的潜在应用感到兴奋。

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1302406059104796763) (6 messages):

> - `添加 CLS tokens`
> - `Sharky blob 开发`
> - `下采样 latents`
> - `推理 API 标准化`
> - `ADs 服务器问题`

- **添加 32 个 CLS tokens 可稳定训练**：一位成员指出，引入 **32 个 CLS tokens** 能显著稳定训练过程。
  
  - 据报道，这一调整提高了训练的可靠性，显著改善了结果。
- **模糊的鲨鱼状物体正变得越来越像鲨鱼**：讨论了一个**模糊的鲨鱼状物体（vaguely sharkish blob）**，它正在进一步演变成更明显的鲨鱼形状。
  
  - 这种转变被描述得十分有趣，暗示其开发过程中取得了有趣的进展。
- **下采样 latents 的挑战**：一位成员询问 **bilinear interpolation**（双线性插值）是否适用于 latent space 中的下采样 latents，并指出结果看起来很模糊。
  
  - 这一询问突显了对当前方法在实现清晰结果方面有效性的担忧。
- **标准化推理 API**：**Aphrodite**、**AI Horde** 和 **Koboldcpp** 的开发者已就一项标准达成一致，以帮助推理集成商识别 API，从而实现无缝集成。
  
  - 这一新标准已在 **AI Horde** 等平台上上线，目前正努力引入更多 API，鼓励与其他开发者合作。
- **ADs 服务器运行飞快**：针对 **ADs server** 的一个轻松评论，将其性能比作“going brrr”，暗示其运行速度极快。
  
  - 这反映了对服务器运行相关操作或速度的幽默看法。

**提到的链接**：

- [Lana Del Rey in Blue Velvet (1986) - David Lynch](https://youtu.be/oNpOf9sYvKY)：更改主角……Blue Velvet (1986) 由 David Lynch 编剧并执导，Lana Del Rey 饰演 Dorothy Vallens，Kyle McLachlan 饰演 Jeffrey Beaumont...
- [serviceinfo](https://github.com/Haidra-Org/AI-Horde/wiki/serviceinfo)：一个用于 AI 艺术和文本生成的众包分布式集群 - Haidra-Org/AI-Horde
- [Issues · comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI/issues/5489,)：最强大、模块化的扩散模型 GUI、API 和后端，具有图形/节点界面。- Issues · comfyanonymous/ComfyUI
- [Add .well-known/serviceinfo endpoint · Issue #2257 · lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2257)：AI Horde、koboldcpp 和 aphrodite 的开发者刚刚就一项标准达成一致，该标准将允许推理集成商快速检索他们正在使用的 API 的信息……

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1303079701375488000) (6 messages):

> - `RAR 图像生成器`
> - `Position Embedding Shuffle`
> - `NLP 影响`

- **RAR 实现了 SOTA 的 FID 分数**：[RAR 图像生成器](https://yucornetto.github.io/projects/rar.html) 拥有 **1.48 的 FID 分数**，在 **ImageNet-256 benchmark** 上表现卓越，同时使用随机退火策略增强了双向上下文学习。
  
  - 它在**不增加额外成本**的情况下实现了这一 SOTA 性能，优于之前的自回归图像生成器。
- **Position Embedding Shuffle 的挑战**：提出在训练开始时*打乱位置嵌入（shuffle the position embeddings）*，并逐渐将这种效果在**结束时降低到 0%**。
  
  - 虽然该方法简化了流程，但一位成员指出，实现起来比最初想象的要复杂一些。
- **质疑洗牌技术在 NLP 中的应用**：讨论了 **Position Embedding Shuffle** 技术是否对 **NLP 任务**有效，并对其适用性表示怀疑。
  
  - 一位成员凭直觉认为这可能行不通，因为 **text tokens 携带的初始信息**比 image tokens 少。

 

**提到的链接**：[Randomized Autoregressive Visual Generation](https://yucornetto.github.io/projects/rar.html)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1302007523229827184) (1 条消息):

> - `OpenAI API Setup`
> - `Lambda Labs Access`

- **API 设置截止日期临近！**：提醒各团队在 **11/4（周一）结束前**完成 **OpenAI** 和 **Lambda Labs** 的 API 设置。
  
  - 未能按时完成可能会影响资源获取，可以通过[此表单](https://forms.gle/JoKbRGsnwKSjGuQy8)进行提交。
- **确保尽早获得额度奖励！**：尽早完成 API 设置将使团队能在下周获得来自 **OpenAI** 的额度，确保顺利参与。
  
  - 这一步对于在整个黑客松期间访问 **Lambda** 推理端点至关重要。
- **资源详情已发布**：有关可用资源的更多信息可以在[此处](https://docs.google.com/document/d/1wnX-oasur0bDvoiMwQC52_wgVYK1UgPwYJiY5-8guTo/edit?usp=sharing)链接的文档中找到。
  
  - 鼓励各团队查阅此文档，以充分了解所提供的资源。

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1303107354283343923) (1 条消息):

> - `Project GR00T`
> - `Jim Fan`
> - `NVIDIA robotics`
> - `Generalist Agents`

- **Project GR00T：通用机器人蓝图**：今天 **PST 时间下午 3:00**，*Jim Fan* 将在[此处直播](https://www.youtube.com/live/Qhxr0uVT2zs)中介绍 **Project GR00T**，这是 NVIDIA 为人形机器人构建 AI 大脑的计划。
  
  - 作为 GEAR 的研究负责人，他强调了他们在各种环境中开发通用 AI Agent 的使命。
- **Jim Fan 的专业知识与成就**：*Jim Fan 博士*拥有 **Stanford Vision Lab** 的博士背景，并因其具有影响力的研究获得了 NeurIPS 2022 的 **Outstanding Paper Award**。他的工作包括用于机器人的多模态模型，以及擅长玩 **Minecraft** 的 AI Agent。
  
  - 他的贡献得到了《纽约时报》和《麻省理工科技评论》等主流媒体的认可。
- **课程资源已在线发布**：所有课程材料，包括直播链接和作业，均可在课程网站访问：[llmagents-learning.org](http://llmagents-learning.org/f24)。
  
  - 鼓励参与者在 <#1280370030609170494> 中提问。

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1302306601612480626) (8 条消息🔥):

> - `LLM Agents Hackathon`
> - `Study Groups`
> - `Compute Resources Access`
> - `Team Formation`
> - `Course Announcements`

- **LLM Agents 黑客松信息**：一位成员分享了 LLM Agents 黑客松的[报名链接](https://rdi.berkeley.edu/llm-agents-hackathon/)，强调了其对准学生的重要性。
  
  - 鼓励参与者了解详情并加入社区讨论该项目。
- **学习小组的组建**：一位成员建议在整理幻灯片笔记和观看视频时组建学习小组，并提议在测验方面进行协作。
  
  - 另一位成员表示赞同，认为这种协作努力会非常有帮助。
- **关于计算资源访问的查询**：一位成员询问了获取计算资源访问权限的时间表，他大约在一周前提交了申请。
  
  - 另一位参与者指出，访问权限取决于所申请的具体资源，并表示本周初应该可以获得访问权限。
- **黑客松组队**：一位用户询问在哪里可以找到黑客松的队伍，寻求合作机会。
  
  - 现有的回复中包含了一个用于组队申请的[表单链接](https://docs.google.com/forms/d/e/1FAIpQLSdKesnu7G_7M1dR-Uhb07ubvyZxcw6_jcl8klt-HuvahZvpvA/viewform)，概述了开发创新型基于 LLM 的 Agent 的目标。
- **对演讲的正面反馈**：一位成员称赞了 Jim 的演讲，对讨论表示感谢。
  
  - 这突显了学生对课程背景下相关讲座的持续参与和兴趣。

**提到的链接**：

- [Large Language Model Agents MOOC](https://llmagents-learning.org/f24): MOOC, 2024 年秋季
- [LLM Agents MOOC Hackathon - Team Signup Form](https://docs.google.com/forms/d/e/1FAIpQLSdKesnu7G_7M1dR-Uhb07ubvyZxcw6_jcl8klt-HuvahZvpvA/viewform): 未找到描述

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-readings-discussion**](https://discord.com/channels/1280234300012494859/1282735578886181036/1302882548350845031) (1 条消息):

> - `Polygonia interrogationis`
> - `Moth identification` (蛾类识别)

- **对 Polygonia interrogationis 分类的困惑**：一位成员对 *Polygonia interrogationis* 的分类提出质疑，怀疑它虽然被称为蝴蝶，但实际上可能是蛾类。
  
  - 讨论强调了准确识别昆虫的必要性，尤其是在协作环境中分享信息时。
- **需要精确的昆虫识别**：该询问凸显了在物种分类方面持续存在的挑战，特别是在区分蛾类和蝴蝶时。
  
  - 它强调了在关于昆虫学的社区讨论中，清晰的沟通和准确信息的重要性。

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1302857083007537222) (7 条消息):

> - `Manual Simulation of ChatGPT Searching` (手动模拟 ChatGPT 搜索)
> - `Future of Browsing and Chatbots` (浏览器与 Chatbots 的未来)
> - `Augmented Reality in Information Access` (信息获取中的 Augmented Reality)

- **探索手动模拟 ChatGPT 的浏览功能**：一位 R&D 成员正在研究是否可以通过跟踪搜索词和结果来手动模拟 **ChatGPT's browsing**，并将其与人类搜索进行比较。
  
  - 这种方法旨在分析 **SEO impact** 以及 ChatGPT 如何高效地过滤和排列多达 **100+ 个结果**等因素。
- **展望未来的 Chatbot 能力**：讨论引发了关于 **ChatGPT** 等 Chatbots 取代传统 Web 浏览器的潜力，并承认预测未来的挑战。
  
  - 一位参与者回忆起过去对 **video calling** 等技术的愿景与我们今天的体验大相径庭，强调了技术进步的不可预测性。
- **将 Augmented Reality 视为游戏规则改变者**：一位成员指出，虽然大家都在设想无浏览器环境，但 **Augmented Reality** 可能会通过持续更新彻底改变信息获取方式。
  
  - 这一观点将讨论扩展到了聊天界面之外，涉及了可能改变我们与信息交互方式的 **transformative technologies**。
- **AI 语境下搜索的本质**：一位参与者强调，搜索功能对于 AI 来说仍然是一个外部过程，是典型的工具调用（tool invocation）而非 AI 的内在行为。
  
  - 他们指出，搜索流的管理完全取决于用户，再次强调了如何利用 AI 工具的重要性。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1302449630767878236) (4 条消息):

> - `Cohere API`
> - `Channel Purpose` (频道用途)
> - `Community Engagement` (社区参与)

- **Cohere API 问题放错频道**：一位成员提醒不要在此频道询问通用问题，并声明该频道专门用于有关 **Cohere's API** 的咨询。
  
  - 他们建议此类问题的相关频道是 <#954421988783444043>。
- **社区对 AI 的兴趣**：另一位成员承认了自己在别处提问的错误，并幽默地指出他们在用户中看到了大量的 **AI** 兴趣。
  
  - 他们表示打算在未来适当地引导自己的问题。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1302066534415335425) (6 messages):

> - `Torchtune Weight Gathering`
> - `Checkpointing Issues`
> - `Distributed Checkpointing`
> - `Llama 90B Pull Request`

- **Torchtune Stage3 收集等效项咨询**：一位成员询问 **Torchtune** 中是否有与 DeepSpeed 参数 `"stage3_gather_16bit_weights_on_model_save": false` 等效的设置。这与解决多节点微调（finetuning）中遇到的问题有关。
  
  - 一个回复澄清说，将此标志设置为 false 有助于实现**分布式/分片 Checkpointing**，这意味着每个 rank 只保存自己的分片（shard）。
- **分布式训练中的 Checkpointing 限制**：有人担心训练期间 **Checkpoint 停滞**的问题，特别是 rank 0 耗尽其 VRAM 的情况。
  
  - 一位用户确认，训练后必须使用外部脚本来检索最终权重。
- **Torchtune 的近期修复**：一位成员提到 **Torchtune** 中的 Checkpointing 问题最近已通过添加 barrier 来防止挂起（hanging）得到解决。
  
  - 他们注意到一个与保存 **90B Checkpoint** 相关的问题，其中 rank 0 出现了挂起问题。
- **Llama 90B 集成讨论**：分享了一个关于将 **Llama 90B** 集成到 **Torchtune** 的 Pull Request，旨在解决特定 bug。
  
  - [PR #1880](https://github.com/pytorch/torchtune/pull/1880) 提供了关于 Checkpointing 相关增强功能的背景信息。

 

**提到的链接**：[Llama 3.2 Vision - 90B by felipemello1 · Pull Request #1880 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/1880)：背景 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）。此 PR 将 llama 90B 添加到 torchtune。修复了一些问题...

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1302062024510935071) (5 messages):

> - `Gradient Norm Clipping`
> - `Duplicate Compile Key in Config`
> - `ForwardKLLoss Misconception`

- **梯度范数裁剪（Gradient Norm Clipping）过程说明**：为了计算正确的梯度范数，需要对所有参数梯度的 L2 范数进行归约（sum reduction），但这不能用于当前的优化器步骤（optimizer steps）。
  
  - 可能在下一次迭代中裁剪梯度范数，但这会改变原始的梯度裁剪逻辑。
- **Torchtune Config 中的重复 Compile 键问题**：一位成员在配置文件 [llama3_1/8B_full.yaml](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full.yaml#L64-L73) 中发现了一个重复的 `compile` 键，导致运行失败。
  
  - 目前尚不确定其他配置文件是否存在类似问题。
- **ForwardKLLoss 实际上计算的是 Cross-Entropy**：Torchtune 中的 `ForwardKLLoss` 计算的是 Cross-Entropy 而非预期的 KL-divergence，这需要调整预期。
  
  - 这种区别很重要，因为由于常数项的存在，优化 KL-divergence 实际上等同于优化 Cross-Entropy。
- **考虑重命名 ForwardKLLoss 以提高清晰度**：大家一致认为 `ForwardKLLoss` 的命名可能会引起用户的困惑。
  
  - 有人指出 DistiLLM 中也使用了类似的命名方式，强调了在实验 `kd_ratio` 值时需要保持清晰。

**提到的链接**：

- [torchtune/recipes/configs/llama3_1/8B_full.yaml at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full.yaml#L64-L73)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做贡献。
- [torchtune/torchtune/modules/loss/kd_losses.py at e861954d43d23f03f2857b0556916a8431766d5d · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/e861954d43d23f03f2857b0556916a8431766d5d/torchtune/modules/loss/kd_losses.py#L49-L53)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做贡献。

---

### **Alignment Lab AI ▷ #**[**announcements**](https://discord.com/channels/1087862276448595968/1124055853218136175/1302396592371794030) (1 messages):

> - `Aphrodite Windows 支持`
> - `PagedAttention 实现`
> - `高吞吐量本地 AI`

- **Aphrodite 推出 Windows 支持**：**Aphrodite** 引擎现在处于实验阶段支持 **Windows**，允许用户在针对 NVIDIA GPU 优化的环境下享受高吞吐量的本地 AI。关于安装，你可以查看[此处的指南](https://github.com/PygmalionAI/aphrodite-engine/blob/main/docs/pages/installation/installation-windows.md)。
  
  - *初步基准测试显示，与 Linux 相比没有性能下降*，证明了其有效托管 AI 端点的能力。
- **Aphrodite 的性能指标和特性**：Aphrodite 支持大多数量化方法，并包含可以超越更大模型的自定义选项，能够处理并行性并提供详细的性能指标。它还支持各种采样选项和结构化输出，增强了易用性。
  
  - 已确认支持 **AMD** 和 **Intel** 计算，但 Windows 支持仍处于**未测试**状态。
- **感谢 Aphrodite 背后的编码奉献**：感谢 **AlpinDale** 的重大贡献，特别是在 Windows 支持实现方面。鼓励社区成员通过转发和支持来表达谢意。
  
  - 此次合作旨在为寻求高效推理解决方案的用户进一步提升**开源** AI 体验。

**提到的链接**：

- [来自 Alignment Lab AI (@alignment_lab) 的推文](https://x.com/alignment_lab/status/1852835864277123582)：宣布：Aphrodite 的 Windows 支持（实验性）[安装指南] https://github.com/PygmalionAI/aphrodite-engine/blob/main/docs/pages/installation/installation-windows.md 享受你的 e...
- [来自 Alpin (@AlpinDale) 的推文](https://x.com/AlpinDale/status/1852830744621785282)：Aphrodite 已上线实验性 Windows 支持。你现在可以在带有 NVIDIA GPU 的 Windows 上使用 PagedAttention 的高性能实现！目前仍处于测试阶段，可能存在...

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1302427106436841533) (2 messages):

> - `用于自动修复漏洞的 LLM`
> - `Self Healing Code 博客文章`
> - `生成式 AI 创新`
> - `播客讨论`

- **LLM 正在彻底改变漏洞修复**：一位成员分享了一篇关于[自愈代码 (Self-healing code) 的新博客文章](https://www.dylandavis.net/2024/11/self-healing-code/)，探讨了 **LLM** 如何协助自动修复有缺陷和易受攻击的软件。
  
  - 作者强调 **2024** 年是关键的一年，LLM 开始着手解决自动修复漏洞的问题。
- **收听 LLMs 播客**：对于听觉学习者，有一个 **播客**，其中两个 LLM 讨论了博客文章中的见解，可在 [Spotify](https://podcasters.spotify.com/pod/show/dylan8185/episodes/Self-Healing-Code-e2qg6oc) 和 [Apple Podcast](https://podcasts.apple.com/us/podcast/self-healing-code/id1720599341?i=1000675510141) 上收听。
  
  - 这让听众能够以对话形式参与内容，同时深入了解自动修复软件技术。
- **韧性软件的未来已来**：文章构想了一个软件可以自主修复的未来，类似于我们身体的愈合方式，从而实现**真正的韧性系统**。
  
  - 讨论强调，我们已经开始在当前技术中看到这种潜力的初步迹象。
- **解决自动修复的六种方法**：作者概述了团队可以采取的 **六种方法** 来应对自动修复的挑战，并指出其中两种已经作为产品提供，而另外四种仍处于研究阶段。
  
  - 这表明了旨在推进自动修复技术的实用解决方案与创新研究的融合。

 

**提到的链接**：[Self Healing Code – D-Squared](https://www.dylandavis.net/2024/11/self-healing-code/)：未找到描述

 

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1302857482875441246) (1 条消息):

> - `Learning on Graphs Conference`
> - `Graph Machine Learning`
> - `Local Meetups`

- **德里 Learning on Graphs Conference**: **Learning on Graphs (LoG) Conference** 将于 **2024 年 11 月 26 - 29 日**在**新德里**举行，重点关注 **machine learning on graphs** 的进展。
  
  - 这是首个南亚分会，由 **IIT Delhi 和 Mastercard** 主办。更多信息请见[此处](https://sites.google.com/view/log2024delhi)。
- **连接图学习领域的创新者**: 该会议旨在连接 graph learning 领域的**创新思想家**和**行业专业人士**，进行深入讨论。
  
  - 该活动鼓励来自**计算机科学**、**生物学**和**社会科学**等不同领域的参与者相互交流。
- **本地 Meetups 的机会**: LoG 社区正在组织一个**本地小型会议**网络，以促进相同地理区域参与者之间的讨论与合作。
  
  - LoG 2024 的**本地 Meetups 征集**仍在开放中，旨在增强主活动期间的社交体验。
- **社区参与和资源**: 参与者可以在 **Slack** 上加入会议社区，并在 [Twitter](https://twitter.com/logconference) 上关注活动更新。
  
  - 往届会议的录像和资料可在 [YouTube](https://www.youtube.com/playlist?list=PL2iNJC54likrwuHz_T3_JnV2AzyX5TLzc) 上获取。

**提到的链接**:

- [LoG New Delhi 2024](https://sites.google.com/view/log2024delhi): 加入我们，帮助连接 graph learning 领域的创新节点！
- [Learning on Graphs Conference](https://logconference.org/): Learning on Graphs Conference

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1303139972945018990) (1 条消息):

> - `Benchmarking function calling`
> - `Retrieval based approach`
> - `Function definitions`

- **征集用于基准测试的函数定义**: 一位成员正在研究 function calling 的 retrieval based approach 基准测试，并正在寻求可用于索引的可用函数及其定义的集合。
  
  - 他们特别提到，按测试类别组织这些信息将非常有帮助。
- **对结构化函数集合的需求**: 讨论强调了对结构化函数定义集合的需求，以辅助 function calling 的基准测试。
  
  - 按测试类别澄清信息将提高进行类似工作的研究人员的可访问性和可用性。

 

---

---

---

{% else %}

> 完整的频道细分内容已因邮件长度限制而截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}