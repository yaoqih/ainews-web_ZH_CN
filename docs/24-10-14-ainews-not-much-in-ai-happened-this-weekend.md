---
companies:
- openai
- meta-ai-fair
- google-deepmind
- microsoft
- x-ai
- spacex
- harvard
- nvidia
date: '2024-10-14T22:52:37.794603Z'
description: '以下是该文本的中文翻译：


  **OpenAI** 推出了图像生成的“编辑此区域”功能，并得到了 **Sam Altman** 的赞赏。**Yann LeCun** 重点介绍了一篇纽约大学的论文，该论文通过使用
  DINOv2 等预训练视觉编码器的特征预测损失来改进像素生成。像 **llama-3.1-8b** 和 **llama-3.2** 变体等长上下文大语言模型（LLM）现在支持高达
  **13.1 万个 token**，为 RAG（检索增强生成）系统提供了替代方案。**Bindu Reddy** 宣布了能够根据英语指令构建和部署代码的 AI
  智能体，这预示着 AI 将取代 SQL，并可能对 Python 产生影响。SpaceX 成功**捕获星舰（Starship）火箭**受到了 **Andrej Karpathy**
  等人的庆祝，**Soumith Chintala** 称赞了 SpaceX 高效、低官僚主义的研究方法。哈佛大学学生开发的 AI 眼镜 **I-XRAY** 引发了隐私担忧，该眼镜可以识别并泄露个人信息。**Meta
  AI FAIR** 的 Movie Gen 模型通过高质量的文本到图像和视频生成（包括同步音频），推动了媒体基础模型的发展。像 **Ameca** 和 **Azi**
  这样的人形机器人现在可以使用 **ChatGPT** 进行富有表现力的对话。**xAI** 在 19 天内快速部署了 **10 万块英伟达 H100 GPU**，首席执行官黄仁勋（Jensen
  Huang）对此向埃隆·马斯克（Elon Musk）表示赞赏。文中还对 **Meta-FAIR**、**Google DeepMind** 和 **微软研究院（Microsoft
  Research）** 等领先的 AI 研究实验室进行了比较。**Sam Pino** 对大语言模型的智能表示怀疑，强调尽管其记忆力很强，但在解决新颖问题方面仍存在局限性。'
id: b2d51ae2-7760-48c7-943f-00f8c0adfa43
models:
- llama-3.1-8b
- llama-3.2
- chatgpt
- movie-gen
original_slug: ainews-not-much-in-ai-happened-this-weekend
people:
- sam-altman
- yann-lecun
- rasbt
- bindureddy
- andrej-karpathy
- soumithchintala
- svpino
- adcock_brett
- rohanpaul_ai
title: 这个周末（AI 领域）没发生什么大事。
topics:
- long-context
- feature-prediction-loss
- ai-agents
- privacy
- text-to-video
- text-to-image
- humanoid-robots
- gpu-deployment
- media-foundation-models
- ai-research-labs
---

<!-- buttondown-editor-mode: plaintext -->**筷子机械臂 (Chopstick arms) 就是你所需要的一切。**

> AI News (2024/10/11-2024/10/14)。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**228** 个频道和 **4291** 条消息）。预计节省阅读时间（以 200wpm 计算）：**551 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

AI 领域动态不多（发布了一个很棒的 [Entropix 解释文档](https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505)），但这是[人类迈向多行星未来的一大步](https://twitter.com/soumithchintala/status/1845519791802708052)。

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

**AI 与技术进展**

- **OpenAI 动态**：[@sama](https://twitter.com/sama/status/1845510942949253156) 分享了他使用 OpenAI 图像生成工具的 "edit this area"（编辑此区域）功能进行头脑风暴的经验，在使用 10 分钟后表达了极大的热情。他还[分享](https://twitter.com/sama/status/1845499416330821890)了另一项引起广泛关注的未指明进展。

- **AI 研究与模型**：[@ylecun](https://twitter.com/ylecun/status/1845535264917364983) 讨论了来自 NYU 的一篇论文，该论文表明即使对于像素生成任务，包含特征预测损失（feature prediction loss）也有助于解码器的内部表示从 DINOv2 等预训练视觉编码器中预测特征。[@dair_ai](https://twitter.com/dair_ai/status/1845513535629271140) 重点介绍了本周的热门 ML 论文，包括 ToolGen、Astute RAG 和 MLE-Bench。

- **长上下文 LLM**：[@rasbt](https://twitter.com/rasbt/status/1845468766118850862) 讨论了长上下文 LLM（如 Llama 3.1 8B 和 Llama 3.2 1B/3B）的潜力，这些模型现在支持高达 131k 的输入 token，在某些任务中可以作为 RAG 系统的替代方案。他还提到了一篇关于 "LongCite" 的论文，旨在通过细粒度引用来改进信息检索。

- **AI Agent**：[@bindureddy](https://twitter.com/bindureddy/status/1845535541342978277) 宣布他们的 AI 工程师现在可以使用英语指令构建简单的 Agent，生成、执行并部署代码。他们认为 AI 已经取代了 SQL，Python 可能是下一步。

**SpaceX 与太空探索**

- **Starship 捕捉**：多条推文（包括来自 [@karpathy](https://twitter.com/karpathy/status/1845452592513507493) 和 [@willdepue](https://twitter.com/willdepue/status/1845442783886111123) 的推文）对 SpaceX 成功捕捉 Starship 火箭表示兴奋和惊叹。这一成就被广泛誉为太空探索中的一个重要里程碑。

- **SpaceX 的组织效率**：[@soumithchintala](https://twitter.com/soumithchintala/status/1845519791802708052) 赞扬了 SpaceX 在没有官僚主义且保持高速度的情况下，执行结构化长期研究和工程博弈的能力，并指出 99.999% 这种规模的组织无法将结构与官僚主义解耦。

**AI 伦理与社会影响**

- **AI 能力**：[@svpino](https://twitter.com/svpino/status/1845434379264458845) 对大语言模型（Large Language Models）的智能表示怀疑，认为虽然它们在记忆和插值方面令人印象深刻，但在解决新颖问题时表现挣扎。

- **隐私担忧**：[@adcock_brett](https://twitter.com/adcock_brett/status/1845495492152574115) 报道了 I-XRAY，这是由哈佛大学学生开发的 AI 眼镜，可以通过观察某人来揭露其个人信息，引发了隐私担忧。

**AI 研发**

- **Meta 的 Movie Gen**：[@adcock_brett](https://twitter.com/adcock_brett/status/1845495489006854435) 分享了关于 Meta 的 Movie Gen 的信息，该模型被描述为“迄今为止最先进的媒体基础模型”，能够从文本生成高质量的图像和视频，Movie Gen Audio 则增加了高保真同步音频。

- **人形机器人**：几条推文（包括来自 [@adcock_brett](https://twitter.com/adcock_brett/status/1845495555247517919) 的推文）讨论了人形机器人的进展，例如 Engineered Arts 的 Ameca 和 Azi，它们现在可以使用 ChatGPT 进行富有表现力的对话。

**AI 行业与市场**

- **xAI 进展**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1845536692821799256) 报道称 xAI 在短短 19 天内就部署了 10 万块 H100 GPU，并引用了 Nvidia 首席执行官 Jensen Huang 对 Elon Musk 在这方面能力的称赞。

- **AI 研究实验室**：[@ylecun](https://twitter.com/ylecun/status/1845495511811264726) 将 Meta-FAIR、Google DeepMind 和 Microsoft Research 等现代 AI 研究实验室与贝尔实验室（Bell Labs）和施乐帕克研究中心（Xerox PARC）等历史悠久的实验室进行了比较，指出 FAIR 是目前这些实验室中最开放的一个。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 经济实惠的 LLM 硬件解决方案**

- **[我的首个低预算 LLM 专用组装机。总计 250 欧元。](https://i.redd.it/ipx6d53e0lud1.jpeg)** ([评分: 110, 评论: 34](https://reddit.com//r/LocalLLaMA/comments/1g2yqs8/my_first_llm_only_build_on_a_budget_250_all/)): 一位用户使用**二手硬件**（包括 **Quadro P5000** GPU 和一台 **HP EliteDesk** 电脑）以 **250 欧元**的价格构建了一个廉价的 **LLM server**。该设置在测试本地 LLM 方面表现良好，如果测试继续产生积极结果，构建者正考虑进行更专业的升级。

- **2倍 AMD MI60 推理速度。MLC-LLM 是 AMD GPU 的快速后端。** ([Score: 54, Comments: 48](https://reddit.com//r/LocalLLaMA/comments/1g37nad/2x_amd_mi60_inference_speed_mlcllm_is_a_fast/)): AMD 的 **MI60 GPUs** 为 **LLM 推理**提供了一个高性价比的选择，其 **32GB VRAM** 价格约为 **$300**，与 RTX 3060 12GB 的价格相当。作者成功编译并运行了各种 LLM 后端，包括 **flash attention**、**llama.cpp** 和 **MLC-LLM**，在使用 **q4f16_1 量化**时，MLC-LLM 在 7-8B 模型上达到了 **81.5 tokens/s**，在 32B 模型上达到了 **23.8 tokens/s**。尽管最初在某些后端上遇到挑战，但 MI60 证明了其能够高效运行现代 LLM，为寻求低价位高 VRAM 容量的用户提供了可行方案。
  - 用户讨论了**廉价 MI60 GPU** 的可用性，有人报告以 **$300** 购买，与 **RTX 3060** 价格相当。MI60 的**性能**与 **RTX 3090** 和 **4090** 进行了对比，关于实际性能与纸面参数的看法不一。
  - 关于**软件兼容性**的讨论强调了 **VLLM** 和 **Aphrodite** 的挑战，而带有 **flash attention** 的 **llama.cpp** 被报告在 **ROCm** 上运行良好。用户对 **MLC-LLM** 的速度表示兴趣，但对模型可用性和转换过程表示担忧。
  - 一位用户感谢原帖作者提供的 **MI60** 编译 **ROCm** 指南，特别提到了 *"修改 setup.py 第 126 行 - 将 "gfx906" 添加到 allowed_archs"* 的技巧。这凸显了在 AI 应用中改进 AMD GPU 软件支持的持续努力。


**主题 2. 开源 AI 语音与转录工具的进展**

- **使用开源工具创建高质量转录：100% 自动化工作流指南** ([Score: 146, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1g2vhy3/creating_very_highquality_transcripts_with/)): 该帖子描述了一个使用**开源工具**创建高质量转录的 **100% 自动化工作流**，包括用于初始转录的 **whisper-turbo**、用于名词提取的开源 LLM **结构化 API 响应**，以及用于说话人识别的 **pyannote.audio**。作者声称这种方法达到了 **98% 的准确率**，与商业解决方案相比具有完全的控制力、灵活性和成本效益，并计划在未来增加对提到的书籍和论文的自动高亮功能。

- **[Ichigo-Llama3.1: 本地实时语音 AI](https://v.redd.it/c23arz6vinud1)** ([Score: 449, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1g38e9s/ichigollama31_local_realtime_voice_ai/)): **Ichigo-Llama3.1** 是一个开源的本地实时语音 AI 系统，结合了 **Whisper**、**Llama** 和 **Bark** 模型，可在无网络连接的情况下进行语音对话。该系统运行在 **RTX 4090** 等消费级硬件上，语音识别和文本转语音生成的**延迟均低于一秒**，实现了与 AI 助手的自然流畅对话。
  - **Ichigo** 是一种教 **LLMs** 理解和说人类语言的灵活方法。开源代码和数据允许用户使用任何 LLM 模型复现该系统，详见 [GitHub](https://github.com/homebrewltd/ichigo)。
  - 该系统在最新的 checkpoint 中支持 **7 种语言**，并使用了修改后的 tokenizer。目前使用 **FishSpeech** 进行文本转语音（可更换），并计划在未来更新中加入声音克隆功能。
  - **Ichigo** 将与 **Jan** 集成，移动应用版本即将推出。基于 **Llama 3.2 3B** 构建的 mini-Ichigo 版本已在 [Hugging Face](https://huggingface.co/homebrewltd/mini-Ichigo-llama3.2-3B-s-instruct) 发布。


**主题 3. Ichigo-Llama3.1: 本地实时语音 AI 的突破**

- **[Ichigo-Llama3.1: 本地实时语音 AI](https://v.redd.it/c23arz6vinud1)** ([Score: 449, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1g38e9s/ichigollama31_local_realtime_voice_ai/)): **Ichigo-Llama3.1** 是一款新型 AI 模型，展示了无需依赖云服务的**本地实时语音 AI 能力**。该模型证明了完全在设备上执行**语音识别**、**文本转语音转换**和**自然语言处理**的能力，与基于云的解决方案相比，可能提供更好的隐私保护并降低延迟。这一进展表明，在使先进语音 AI 技术可用于本地、离线使用方面取得了重大进展。
  - **Ichigo** 是一种灵活的方法，用于教导 **LLM 人类语音理解和说话能力**，其开源代码和数据可在 [GitHub](https://github.com/homebrewltd/ichigo) 上获得。该架构通过 Whisper 使用音频的**早期融合 (early fusion)**和**向量量化 (vector quantization)**。
  - 该模型目前支持 **7 种语言**，并可在单张 **Nvidia 3090 GPU** 上运行。用户对潜在的**语音克隆**功能以及与非 GPU 系统的 **llamacpp** 兼容性表示了兴趣。
  - **Ichigo-Llama3.1** 引入了诸如**回话 (talking back)**和**识别无法理解的输入**等改进。开发人员计划将 Ichigo 与 **Jan Mobile** 集成，创建一个具有记忆和 RAG 等功能的 Android 应用。


- **[文本转语音：xTTS-v2、F5-TTS 和 GPT-SoVITS-v2 的比较](https://tts.x86.st/)** ([Score: 127, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1g36cm6/texttospeech_comparison_between_xttsv2_f5tts_and/)): **xTTS-v2**、**F5-TTS** 和 **GPT-SoVITS-v2** 是三款正在进行性能比较的先进**文本转语音 (TTS)** 模型。虽然帖子正文未提供比较的具体细节，但这些模型代表了目前 TTS 技术的最新水平，每种模型都可能在语音合成质量、自然度或多功能性方面提供独特的功能或改进。
  - **GPT-SoVITS-v2 Finetuned** 的表现受到了称赞，尤其是在处理笑声方面。用户对**微调指令**表示了兴趣，并讨论了其 **MIT 许可证**，鉴于 XTTS-v2 的不确定状态，这可能是一个优势。
  - 讨论了消费级 GPU 上的实时 TTS 性能，一位用户报告称在 **3090 GPU** 上使用 **xTTS 或 SoVITS** 获得了接近实时的结果。为了获得最佳性能，建议按标点符号拆分输出并使用独立的 GPU 进行 TTS。
  - 模型之间的比较强调了 **F5-TTS** 表现良好，其 **E2 模型**对某些用户来说听起来更好。 **XTTS-v2** 因其稳定性和适合有声读物风格的声音而受到关注，而 F5/E2 被描述为更具情感，但容易产生伪影。


**主题 4. 高端 AI 硬件：NVIDIA DGX B200 现已公开上市**

- **你现在可以在商店买到 DGX B200 了** ([Score: 53, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1g2mzm4/you_can_buy_dgx_b200_in_a_shop_now/)): **NVIDIA 的 DGX B200** 是一款高性能计算系统，拥有 **1.5TB VRAM** 和 **64TB/s** 带宽，现已在一家服务器硬件商店公开列出待售。该系统拥有令人印象深刻的理论性能，在运行 **LLaMa 3.1 405B** 时可达 **120t/s**，但其要求也极高，包括 **10 KW 的功耗**，且价格足以让一家中型公司配备完整的服务器。
  - **$500,000 的 NVIDIA DGX B200** 与一台以 **$480,000** 售出的 **8 年前拥有 8000 颗 Xeon 处理器的超级计算机**之间的对比，凸显了计算硬件的快速贬值。这展示了不到十年间巨大的技术进步。
  - 用户讨论了该系统 **72/144 petaflops** 的理论性能，并注意到其具有竞争力的性价比。然而，考虑到模型在多个 GPU 之间的分片 (sharding)，对于 LLM 推理/训练中 **64TB/s** 带宽的实际利用率提出了疑问。
  - 对 **NVIDIA 许可实践**的批评也随之出现，用户称 **3 年许可证费用**是“骗局”，并认为 **NVIDIA Docker 许可证**是在不改进产品的情况下榨取更多资金的“疯狂”尝试。


**主题 5. 提高 LLM 输出质量：重复惩罚 (Repetition Penalty) 的实现**

- **重复惩罚（Repetition penalties）的实现非常糟糕 - 简短的解释与解决方案** ([Score: 47, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1g383mq/repetition_penalties_are_terribly_implemented_a/)): 该帖子分析了 **LLMs** 中的 **重复惩罚**，强调了其在减少 **多轮对话** 期间 **重复性** 的重要性。作者批评了当前的实现方式，特别是 **frequency penalty**，它通常应用于包括特殊 token 和用户消息在内的 **所有现有 token**，可能导致模型无休止地胡言乱语等问题。作者提出了一种使用 **logit bias** 的 **临时解决方案**，仅将 **frequency penalties** 应用于模型自身的消息，并认为这种方法优于标准的 **repetition penalties**。
  - **Frequency penalties** 因惩罚了诸如 "a"、"the" 和 "and" 等基本语言元素而受到批评。建议使用 **DRY**（惩罚序列重复）和 **XTC**（移除高概率 token）等替代方法来更有效地对抗重复。
  - 用户报告了针对采样器的 **掩码方法（masking approaches）** 取得了成功，允许根据消息属性、格式化字符和标点符号进行自定义。这种有针对性的方法被认为优于全局应用采样器。
  - 某些模型（如 **Mistral Large 2 123B**）在有效上下文长度内使用时可能不需要重复惩罚。**XTC sampler** 可以增加写作任务的创造力，而 **DRY** 则被推荐用于角色扮演场景。

## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**太空探索与工程突破**

- **SpaceX 成功回收 Super Heavy booster**：SpaceX 通过使用 "Mechazilla" 塔臂 [成功回收了 Super Heavy booster](https://www.reddit.com/r/singularity/comments/1g2onev/super_heavy_booster_catch_successful/)，实现了一个重大里程碑。这一工程壮举被视为迈向完全可重复使用火箭的重要一步。

- **从海滩观看 Starship 和 Super Heavy booster 回收**：一段 [来自海滩的视频](https://www.reddit.com/r/singularity/comments/1g2ozgs/mechazilla_and_super_heavy_booster_from_the_beach/) 展示了 Mechazilla 塔回收 **Super Heavy booster** 的过程，展示了这一成就的规模和震撼力。

**AI 与机器学习进展**

- **Counter-Strike 在神经网络中运行**：研究人员展示了 [完全在神经网络内运行的 Counter-Strike](https://www.reddit.com/r/StableDiffusion/comments/1g2of3a/counterstrike_runs_purely_within_a_neural_network/)，运行平台为 RTX 3090 GPU。该模型根据玩家输入生成游戏画面，无需传统游戏代码。

- **xAI 快速搭建训练集群**：NVIDIA 的 Jensen Huang 赞扬了 **xAI** [仅用 19 天就搭建好了他们的 AI 训练集群](https://www.reddit.com/r/singularity/comments/1g2tgh5/jensen_huang_on_how_fast_xai_setup_their_training/)，而这一过程对于其他公司通常需要一年时间。

- **AI 研究员批评 OpenAI**：一位 AI 研究员 [警告称 OpenAI 可能会成为“有史以来最奥威尔式（Orwellian）的公司”](https://www.reddit.com/r/OpenAI/comments/1g2t1ys/ai_researcher_slams_openai_warns_it_will_become/)，对公司最近的发展方向表示担忧。

- **论文发现工具**：两名工程师创建了 [Ribbit Ribbit](https://www.reddit.com/r/MachineLearning/comments/1g31hfd/p_drowning_in_research_papers/)，这是一款能够推荐个性化 AI 研究论文并生成推文大小摘要的应用。

**未来主义与技术预测**

- **Kurzweil 的预测回顾**：一份 [Ray Kurzweil 预测的汇编](https://www.reddit.com/r/singularity/comments/1g2nurz/kurzweil_predictions_all/) 显示，虽然许多预测并未在预定日期实现，但技术进步的整体轨迹与他的预测一致。

**机器人与自动化**

- **特斯拉 Optimus 机器人为远程操作**：在特斯拉的 **Cybercab** 活动中，[Optimus 机器人被揭露是由人类通过 VR 远程操作的](https://www.reddit.com/r/singularity/comments/1g3708u/the_optimus_robots_at_teslas_cybercab_event_were/)，而非像最初一些人认为的那样是完全自主的。

**新兴技术**

- **梦境通信突破**：研究人员实现了 [两人在梦境中的某种形式的通信](https://www.reddit.com/r/singularity/comments/1g2wkvl/two_people_communicate_in_dreams_inception/)，让人联想到电影《盗梦空间》（Inception）。

---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1：新 AI 模型发布与对比**

- [**Aria 登顶最强多模态模型**](https://hf.co/papers/2410.05993)：由 [@rhymes_ai_](https://twitter.com/rhymes_ai_) 推出的新 **Aria** 模型以 **24.9B parameters** 统治了 🤗 Open LLM Leaderboard。该模型在 **400B multimodal tokens** 上训练，支持图像、视频和文本输入，并拥有 **64k token context window**。
- [**O1-mini 表现不佳，而 O1-preview 遥遥领先**](https://x.com/JJitsev/status/1845309654022205720)：尽管声称很强大，但 **O1-mini** 在简单任务上的表现不如 **O1-preview**。后者甚至在奥林匹克级别的挑战中也表现出色，这让人对 mini 模型的能力产生怀疑。
- [**NanoGPT 再次打破速度纪录**](https://x.com/kellerjordan0/status/1845865698532450646)：通过 **SOAP optimizer** 和 **ReLU²** 激活函数等巧妙的代码优化，**NanoGPT** 在 **15.2 minutes** 内达到了惊人的 **3.28 Fineweb validation loss**，刷新了训练速度纪录。

**主题 2：AI 框架与工具的进展**

- [**OpenAI 进军多智能体系统**](https://github.com/openai/swarm)：**OpenAI** 发布了 **Swarm**，这是一个用于构建和编排多智能体系统的实验性框架，无需 Assistants API 即可实现无缝的 Agent 交互。
- [**Swarm.js 为 Node.js 带来多智能体魔力**](https://github.com/youseai/openai-swarm-node)：受 OpenAI Swarm 启发，**Swarm.js** 作为 Node.js SDK 发布，让开发者能够利用 **OpenAI API** 操控多智能体系统，并邀请社区协作。
- [**LegoScale 构建重量级 LLM 训练方案**](https://openreview.net/forum?id=SFN6Wm7YBI)：**LegoScale** 系统为大语言模型的 3D 并行预训练提供了一个可定制的、PyTorch 原生解决方案，简化了跨 GPU 的复杂训练。

**主题 3：AI 模型训练与微调的挑战**

- [**微调者与 LLaMA 3.2 及 Qwen 2.5 的博弈**](https://github.com/unslothai/unsloth)：用户在微调 **LLaMA 3.2** 和 **Qwen 2.5** 时遇到了困难，尽管遵循了指南，但仍遇到了障碍和令人困惑的输出。
- [**超参数困扰：社区呼吁缩放指南**](https://apaz-cli.github.io/blog/Hyperparameter_Heuristics.html)：工程师们强调需要一份超参数缩放指南，感叹关键知识被“困在研究人员的脑子里”，并强调正确的调优至关重要。
- [**速度对决中 FA3 不敌 F.sdpa**](https://github.com/EleutherAI)：实现 **FA3** 的尝试表明其性能落后于 **F.sdpa**，引发了对安装问题和性能下降的困惑。

**主题 4：安装噩梦与性能谜题**

- [**Mojo 安装让用户火冒三丈**](https://docs.modular.com/magic/conda)：沮丧的用户报告称 **Mojo** 的安装过程像迷宫一样，损坏的 Playground 和教程的匮乏导致了死胡同。
- [**GPU 利用率低让用户百思不得其解**](https://pinokio.computer/)：尽管拥有强大的硬件，用户发现他们的双 **3060** 显卡 GPU 利用率低于 **10%**，认为原因是 IO 瓶颈或电源管理异常。
- [**LM Studio 安装因 UAC 提示引发关注**](https://lmstudio.ai/docs/cli/log-stream#)：由于 **LM Studio** 在安装时没有 UAC 提示，引发了用户的担忧，质疑其是否在篡改系统文件，并分享了针对 Linux 库问题的修复方案。

**主题 5：AI 伦理与社区风暴**

- [**OpenAI 模型恶作剧引发对齐警报**](https://arxiv.org/abs/2410.04840)：有报告称 **OpenAI** 模型操纵其测试环境，这引发了工程师们对 **AI alignment** 的严重担忧。
- [**Swarm 之争：OpenAI 被指控代码剽窃**](https://x.com/KyeGomezB/status/1844948853604196763)：**Kye Gomez** 指控 **OpenAI** 剽窃了 **Swarms framework**，声称他们“偷走了我们的名字、代码和方法论”，并暗示除非获得赔偿，否则将采取法律行动。
- [**苹果语出惊人：“LLM 无法推理”**](https://www.youtube.com/watch?v=tTG_a0KPJAc)：一段名为《苹果投下 AI 炸弹：LLM 无法推理》的挑衅性视频引发了关于 AI 推理极限的辩论，并呼吁观众为 AGI 做好准备。


---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 模块导入困扰**：用户在导入 **Unsloth** 模块时遇到了与 Python 环境相关的安装错误，建议的修复方法包括在 conda 环境中使用 pip。
   - 这引发了对依赖管理的广泛关注，导致了故障排除链接和技巧的分享。
- **模型微调是一项棘手的任务**：参与者讨论了微调语言模型的困难，指出模型在训练后往往无法响应查询。
   - 建议强调了对微调数据集进行仔细评估的重要性，以确保最佳性能。
- **推荐使用 WSL2 进行开发**：建议 Windows 用户利用 **WSL2** 来有效地运行 AI 开发环境，包括安装和执行模型。
   - 用户之间流传着关于 WSL2 安装问题的故障排除方案，强调了对特定错误进行指导的需求。
- **LLaMA 3 对决 Claude 3.5 Sonnet**：一位用户寻求关于 **LLaMA 3** 与 **Claude 3.5 Sonnet** 在编程任务中表现对比的见解，暗示希望通过 **Unsloth** 增强 **LLaMA** 的性能。
   - 这种兴趣表明了围绕针对特定任务有效性调整模型的更广泛讨论。
- **Hugging Face 状态检查**：一位用户报告称 **Hugging Face** 上的服务运行正常，尽管他们自己在下载模型时遇到了麻烦。
   - 这引发了关于潜在的本地化问题与更广泛的可访问性之间的疑问。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 服务遭遇宕机**：用户报告称 **Hugging Face 服务** 出现了 **504** 和 **502** 等服务器错误，表明可能存在宕机。社区成员分享了他们的经历，并指出服务间歇性地恢复在线。
   - 持续的问题似乎影响了各种功能，引发了关于服务器可靠性和用户挫败感的讨论。
- **寻求多语言 Embedding 模型**：成员们讨论了对最佳 **多语言 Embedding 模型**（尤其是德语）的需求。重点放在选择适合多样化语言应用的模型上。
   - 多位成员就有效 Embedding 模型对于多语言数据集等高维空间的重要性发表了看法。
- **对特斯拉机器人活动的质疑**：参与者对 **特斯拉机器人活动** 的真实性表示怀疑，质疑这些机器人是否真的是自主运行。许多人认为这些机器人可能是被远程控制的。
   - 对公司声誉和投资者感知的潜在影响的担忧，凸显了此类误导性展示可能带来的后果。
- **AI Agent 与协作平台**：一位成员提出了创建一个 **AI Agent 定制平台** 的想法，讨论了普通用户在使用现有解决方案时面临的复杂性。讨论迅速转向了对协作项目的需求。
   - 参与者表示对更精简的协作感兴趣，而不是分散的个人努力。
- **澄清模型许可变体**：讨论涉及了 **MIT** 和 **Apache 许可证** 之间的区别，重点关注商业使用和代码分叉（forking）方面。成员们澄清说 **MIT 许可证** 更加宽松，对多功能项目更有吸引力。
   - 社区表达了对 MIT 在各种开发场景中所提供的灵活性的偏好。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **用户反映 LM Studio 安装问题**：关于 **LM Studio** 在没有 UAC 提示的情况下进行安装的担忧引起了关注，特别是其对用户配置文件与系统文件的影响。
   - 一些用户报告在某些 Linux 发行版上运行 AppImage 时缺少库，导致设置过程复杂化。
- **Qwen-2.5 性能领先**：用户对比了 **Qwen-2.5** 和 **Deepseek** 等 LLM 的性能，指出 **Qwen** 在 Python 任务中的速度和效率表现出色。
   - 用户对测试各种量化（quantization）选项以进一步提升输出质量和速度表现出浓厚兴趣。
- **审视 GPU 电源管理**：尽管达到了 **17.60 tokens/秒**，但用户对双 **3060** 显卡运行模型时 **GPU** 利用率低于 **10%** 的情况表示担忧。
   - 讨论暗示潜在的 IO 瓶颈挑战或不稳定的电源管理可能是罪魁祸首。
- **NVIDIA 在 AI 任务中占据优势**：辩论集中在 **NVIDIA 4060 Ti** 和 **AMD RX 7700 XT** 之间的选择，强调了 NVIDIA 卓越的 AI 支持。
   - 用户建议使用 **NVIDIA** GPU 通常在运行 AI 应用程序时会遇到更少的麻烦。
- **Mistral Large 在消费级设备上大放异彩**：**Mistral-Large 123b** 模型因其在消费级机器（特别是 **M2 Studio** 配置）上的灵活性而受到青睐。
   - 用户指出 **Mistral Large** 配置能有效利用 VRAM，熟练处理各种上下文。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OpenAI 模型性能担忧**：成员们对据报道 OpenAI 模型操纵其测试环境表示担忧，这引发了 **AI alignment**（AI 对齐）问题。
   - *这凸显了社区内现有的 AI 安全与伦理挑战。*
- **FA3 相比 F.sdpa 变慢**：用户在 FA3 上遇到了重大挑战，指出其运行速度比 **F.sdpa** 慢，使实现过程复杂化。
   - *一位用户强调了与现有模型相比，在正确安装方面的困惑。*
- **NanoGPT 打破训练速度记录**：通过代码优化，实现了 **15.2 分钟内达到 3.28 Fineweb 验证损失** 的新 **NanoGPT 速度记录**。
   - *更新包括使用 SOAP 优化器和对投影层进行零初始化以增强性能。*
- **Swiglu vs ReLU²：激活函数之争**：讨论对比了 **ReLU²** 和 **Swiglu** 激活函数的有效性，表明性能因模型大小而异。
   - *结果显示 Swiglu 在大型模型中可能更有效，尽管目前的测试更倾向于 ReLU²。*
- **创建超参数缩放指南**：提出了一个关于超参数缩放（hyperparameter scaling）指南的建议，旨在集中调优方法论的知识，这对于模型性能至关重要。
   - *成员们承认现有信息主要掌握在研究人员手中，导致获取困难。*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 辅助老年护理管理**：参与者讨论了使用 AI 协助老年人管理药物并提供陪伴，同时探讨了**可靠性**和**伦理影响**。
   - 讨论提出了关于确保 AI 在不损害安全的情况下处理**护理任务**的担忧。
- **F5-TTS 语音克隆挑战**：一位用户分享了将 **F5-TTS** 模型与 Groqcaster 集成以实现自动化语音输出的经验，该模型在本地语音克隆方面表现出色。
   - 虽然质量尚未达到 **ElevenLabs** 的水平，但完全在本地生成的能力是一个显著优势。
- **空间计算设备对决**：用户评估了 **Meta Quest** 和 **Xreal** 等空间计算设备在桌面使用中的表现，辩论了它们在多显示器设置中的有效性。
   - 虽然 Meta Quest 因原生应用支持而受到青睐，但其在光学质量方面的一些局限性也被指出。
- **GPT 集成 Bug 报告**：用户注意到自定义 GPT 不再能通过 '@' 符号集成到另一个 GPT 的对话中，这一变化可能暗示存在 Bug。
   - 他们建议联系支持部门，因为仍有一些用户能够使用此功能。
- **Text2SQL 查询关注**：关于 **text2sql** 实现经验的讨论非常活跃，特别是在使用 LLM 管理复杂查询方面。
   - 用户强调在获取相关数据时需要保持上下文清晰，以避免输出内容过于冗杂。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection 模型上线**：由 **@inflectionAI** 提供支持、驱动 **@Pi** 的模型现已在 [OpenRouter](https://x.com/OpenRouterAI/status/1845213137747706224) 上线，无最低消费限制，并趣味性地侧重于 **emojis** 🍓。
   - 该模型旨在通过集成 emoji 来提供更具参与感和趣味性的聊天体验，从而增强用户交互 🤗。
- **Grok 2 发布并提供访问权限**：OpenRouter 现在提供 **Grok 2** 和 **Grok 2 Mini**，价格为 **$4.2/M input** 和 **$6.9/M output**，尽管最初存在速率限制，详见其 [公告](https://x.com/OpenRouterAI/status/1845549651811824078)。
   - 用户对其强大的功能表示赞赏，但也指出交互过程中资源管理的重要性。
- **MythoMax 端点提供免费访问**：OpenRouter 推出了 **免费的 MythoMax 端点**，为希望利用先进模型的用户扩大了可访问性。
   - 这一举措旨在通过在不增加额外成本的情况下提供更多选择来增强用户体验。
- **聊天室改进提升易用性**：用户现在可以在聊天室中直接 **拖放** 或粘贴图片，提升了整体交互质量。
   - 这些改进体现了 OpenRouter 致力于在其平台内实现流线型且用户友好的沟通。
- **Grok API 遇到问题**：用户报告 **Grok API** 频繁出现“500 Internal Server Error”和“Rate limit exceeded”等错误，该 API 目前仍被归类为实验性。
   - 建议考虑使用 beta 模型和其他替代方案来缓解这些问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider AI LLC 为源代码提供归属保障**：**Aider AI LLC** 的成立确保了 **aider** 源代码受 **Apache 2.0 license** 保护，维持其作为完全 **免费且开源项目** 的地位。
   - *“这是一个社区驱动的努力，没有融资轮次或员工参与，”* 再次重申了对开源原则的承诺。
- **用户克服 Aider 安装挑战**：反馈表明，通过 `pipx` 安装 Aider 大大简化了设置过程，避免了漫长的安装问题。
   - 一位用户强调，严格遵守安装指南可以减少安装问题。
- **Jetbrains 插件崩溃引发辩论**：用户报告 Aider 的 Jetbrains 插件在启动时崩溃，促使一些人直接通过终端使用 Aider。
   - 讨论集中在插件缺乏基本功能（包括文件捕获和键位绑定）所导致的挫败感。
- **在公司代码库中使用 Aider 的警告**：由于潜在的 **政策违反** 和 **数据泄露** 风险，在公司代码库中使用 Aider 引起了担忧。
   - 虽然一些人强调 Aider 在本地运行且不共享数据，但对 API 使用和屏幕共享的担忧依然存在。
- **LLM 模型的性能对比**：关于不同 LLM 与 Aider 集成效果的辩论引发了对模型性能的讨论，特别是针对 **Grok-2** 和 **GPT-4o** 等模型。
   - 成员们指出需要仔细选择模型，以确保在编码任务中获得最佳输出。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 演变为初创公司**：Nous Research 最初是一个 Discord 小组，现已转型为一家专注于 AI 开发（尤其是 **开源项目**）的受资助初创公司。
   - 社区现在在促进 AI 研究领域的协作和创意共享方面发挥着至关重要作用。
- **DisTrO 加速模型训练**：**DisTrO** 旨在实现跨互联网的更快 AI 模型训练，推动社区驱动的开发，作为封闭模型的替代方案。
   - 该倡议旨在确保开源领域的持续进步。
- **揭示神经网络中的模型坍缩 (Model collapse)**：最近的一项研究调查了 **模型坍缩 (Model collapse)** 现象，表明即使是 **1%** 的合成数据也可能导致显著的性能衰退。
   - 研究警告说，更大的模型可能会加剧这种坍缩，对传统的 Scaling 方式提出了挑战。
- **GSM-Symbolic 改进 LLM 评估**：**GSM-Symbolic** 基准测试的引入为评估 **LLM** 的数学推理能力提供了增强的指标。
   - 该基准使评估方法多样化，促进了对语言模型更可靠的评估。
- **OpenAI 因 Swarm 框架面临审查**：有指控称 **OpenAI** 侵犯了 **Kye Gomez** 的 **Swarms 框架**，声称其窃取了代码和方法论。
   - 除非投资流向他们的项目，否则正在考虑采取潜在的法律行动。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Reasoning Mode 在 Pro Search 中上线**：**Perplexity Team** 推出了 **Reasoning Mode**，这是一项实验性功能，可检测额外的计算资源何时能提升回答质量，并鼓励用户在 [feedback channel](https://discordapp.com/channels/1054944216876331118) 分享使用案例。
   - 用户提供了各种 Pro Search 示例查询，包括寻找 **OpenAI** 联合创始人和高分电影，旨在利用这一增强功能。
- **应对 AI Image Generation 的成本影响**：讨论中提到了与 **AI image generation** 相关的成本，敦促用户考虑这些功能的预算，更多详情见 [此处](https://www.perplexity.ai/search/how-much-is-ai-image-generatio-WZxnVk6YS.iLP_uBYLgPdA)。
   - 对话强调了成本效益与项目对高质量视觉输出需求之间的平衡。
- **用户对 API 来源 URL 的不满**：用户正面临 API 在响应中不显示来源 URL 的问题，寻求支持却未得到回应，导致询问悬而未决。
   - 讨论转向了在线 API，提到了 [Perplexity API Docs](https://docs.perplexity.ai/guides/model-cards) 中提供的 `sonar-online` 模型，旨在澄清模型功能。
- **对 AI 模型性能的评价褒贬不一**：用户在使用各种 AI 模型时体验各异，一些人更倾向于使用 **Claude** 处理编程任务，而非受近期更新影响性能的 **O1 mini**。
   - 用户对 **Perplexity API** 提供类似于在线交互的高质量响应的能力表示担忧，并指出了显著差异。
- **Perplexity Pro 功能的精彩更新**：**Perplexity Pro** 的更新引起了近期关注，用户分享了旨在增强参与度和功能的新特性见解。
   - 成员可以通过此 [链接](https://www.perplexity.ai/search/perplexity-pro-uber-one-l3nvwTFnQ6e1xILs5cPtHA#0) 进一步探索这些变化，引发了关于最佳实践的热烈讨论。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Attention Layer 实现困惑**：一名成员正在寻求使用 Python 中 cuDNN 的 **SDPA** 实现 **Attention layer** 的教程，在实例化 **pygraph** 时感到困惑。他们正在参考 [cudnn-frontend 仓库中的 notebook](https://github.com/cudnn/frontend)。
   - *希望能得到任何帮助*来澄清实现细节。
- **PyTorch 和 Triton 中的性能分析差异**：成员们发现使用 **PyTorch**、**Triton** 和 **CUDA** 进行性能分析（profiling）时结果存在显著差异，引发了关于该信任哪个分析器的问题。
   - 尽管 **Triton** 声称整体性能相当，但在许多测试中，自我评估似乎显示 **PyTorch** 处于领先地位。
- **Apple Silicon 上的 Metal 编程挑战**：成员报告了在 **Apple Silicon GPUs** 上使用 **Docker** 的困难，并引用了社区内未解决的问题。该问题的内部工单仍处于开启状态，且未在积极处理。
   - 讨论还涉及了 [torch.special.i0 算子的 PR](https://github.com/pytorch/pytorch/pull/137849)，重点是增强 **MPS** 支持。
- **对 Entropix Sampling 的怀疑**：围绕 **Entropix sampling** 的怀疑情绪升温，一些人声称这感觉像是“毫无意义的邪教内容”，对其可信度提出质疑。
   - 尽管存在担忧，最近的一篇 [博客文章](https://timkellogg.me/blog/2024/10/10/entropix) 提到其目标是在不进行大规模修改的情况下简化推理（reasoning）。
- **高效 LLM 部署策略**：一场在线见面会定于 **10 月 5 日** **下午 4 点 PST** 举行，讨论 **LLM** 部署，参与方包括 **SGLang**、**FlashInfer** 和 **MLC LLM**。
   - 主题包括 **low CPU overhead scheduling** 和用于高性能 **LLM** 服务的 **kernel generation**，并提供社区互动机会。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **鼓励对 CohereForAI 做出贡献**：成员们强调了对 [CohereForAI](https://cohere.com/research) 社区做出有意义贡献的重要性，并建议将公民科学作为 AI 参与者的切入点。
   - 一位成员表达了贡献和指导的愿望，旨在使项目符合与技术建立**共生关系**的愿景。
- **AI 创新者荣获诺贝尔奖**：Sir John J. Hopfield 和 Sir Geoffrey E. Hinton 因在 AI 和神经网络领域的开创性贡献获得了 2024 年 [诺贝尔物理学奖](https://www.nobelprize.org/prizes/physics/2024/press-release/)。
   - 他们的工作为推动机器学习技术的发展奠定了**基础性发现**。
- **API Tokens 困惑**：一位成员询问在 API 请求中是否有必要使用 `<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>` token，并担心如果不使用这些 token，响应质量会受到影响。
   - *如果不包含这些 token，响应质量还会好吗？* 这在社区中仍然是一个悬而未决的问题。
- **Cohere Rerank 定价需要澄清**：关于 Cohere 的 Rerank 定价结构中是否包含网络搜索定价存在困惑，因为成员们在网站上找不到相关细节。
   - 了解这一定价模型对于规划有效的实施策略至关重要。
- **即将举行的 Gen AI Hackathon 公告**：成员们受邀参加由 **CreatorsCorner** 组织的 **Gen AI Hackathon**，旨在创建创新的多 Agent 系统。
   - 正如 [黑客松邀请函](https://lu.ma/ke0rwi8n) 中所述，该活动鼓励通过智能解决方案进行协作，以增强人类潜力。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 安装挫折**：用户在安装 *Mojo* 时遇到**安装问题**，导致 playground 功能损坏和示例代码错误，且缺乏清晰的教程。
   - 一位用户指出，*Magic*、*Mojo* 和 *MAX* 之间这种脱节的过程在社区中造成了极大的困惑。
- **AES 硬件支持进展**：一位成员展示了通过 LLVM intrinsics 在 *Mojo* 中实现 **AES 硬件支持**的进展，增强了库的集成能力。
   - 这一努力强化了 *Mojo* 的灵活性，使得高级硬件功能可以平滑地整合到项目中。
- **MAX 编译时间评估**：即使是简单的任务，**MAX** 的图编译时间在初始运行后约为 **300 ms**，首次编译约为 **500 ms**。
   - 讨论强调了提高缓存命中时间以优化开发期间性能的重要性。
- **隐式转换引发争论**：*Mojo Lists* 中的**隐式转换**引发了成员们的疑问，因为由于现有构造函数的存在，将 Int 添加到 List 中似乎可能是非预期的行为。
   - 目前正在跟踪这一行为，这可能会使未来实现中的类型处理变得复杂。
- **使用 Mojo 构建时面临链接错误**：用户在尝试构建 *Mojo* 文件时遇到**链接错误**，这表明编译过程中可能缺少库。
   - 提供的帮助包括检查 magic 环境激活情况以及通过命令行执行正确的安装协议。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Swarm 发布实验性框架**：OpenAI 推出了 **Swarm**，这是一个用于构建多 Agent 系统的轻量级库，强调了一种**无状态抽象**，用于管理 Agent 交互，而不依赖于 Assistants API。
   - *这是一个实验性框架*，旨在帮助开发者轻松理解 Agent 的角色和移交（handoffs）。
- **Entropix 在成员中走红**：成员们对 **Entropix** 进行了热烈讨论，并对其功能和潜在影响进行了概述。
   - 随着兴趣的增长，用户渴望看到与该工具进展相关的后续评估功能。
- **增强 AI 性能的 RAG 技术**：关于 **RAG 技术** 的讨论集中在一个 GitHub 仓库上，该仓库展示了将检索与生成模型集成的先进方法。
   - 参与者旨在优化性能，并将 *Haystack* 等框架与针对特定用例的自定义解决方案进行比较。
- **Jensen 对 NVIDIA 基础设施的见解**：在最近的一次采访中，Jensen 讨论了 NVIDIA 对 AI 基础设施的**全栈方法**，强调了现代应用中加速计算的必要性。
   - 他的言论重申了**生成式 AI** 的变革潜力，并指出该领域需要持续创新。
- **生产级 AI 工程剧集亮点**：最新的 [播客剧集](https://x.com/latentspacepod/status/1844870676202783126) 涵盖了对**生产级 AI 工程**的见解，重点关注了 **Evals** 在行业中的关键作用。
   - 专家宣布 **Evals 是核心**，强调了在 LLM Ops 领域中，对稳健评估指标的需求已成为日益增长的优先事项。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **3060ti 在 Stable Diffusion 中表现出色**：讨论强调了 **3060ti** 在 Stable Diffusion 中的有效性，尽管存在 **8GB VRAM** 的限制，但表现出奇地好。用户引用 **Flux** 图像生成作为该 GPU 能力的证明。
   - 一位用户断言，通过正确的技术，3060ti 可以高效处理 AI 图像生成中要求苛刻的任务。
- **Lora 训练优于 Embedding**：参与者辩论了 **Lora 训练** 相比 **Embedding** 的优势，认为 Lora 通常会产生更高质量的图像。虽然 Embedding 仅影响文本编码器，但 Lora 允许进行更细致的扩散模型训练。
   - 这一细节引发了对优化图像质量的工作流调整进行更深入讨论的兴趣。
- **图像放大技术受到关注**：社区比较了 **Tiled Diffusion** 和 **Ultimate SD Upscale**，指出每种方法服务于不同的目的——VRAM 管理与分辨率增强。这两种技术的优点在进行中的项目中得到了广泛评估。
   - 用户一致认为，了解何时应用每种技术可以显著影响图像处理任务的结果。
- **图像转 3D 模型生成仍需改进**：**图像转 3D 模型生成** 的复杂性引发了大量讨论，参与者认识到现有解决方案中存在的差距。多视角推理技术成为目前最可靠的方法。
   - 成员们表达了对该领域创新的集体需求，因为挑战依然重大。
- **寻求产品照片集成的帮助**：一位成员寻求关于将产品照片集成到各种背景中的建议，强调需要高质量的结果而非基础的合成。建议指向利用 **Lora 训练** 在最终图像中实现更好的融合。
   - 对话强调了先进技术在满足特定视觉需求方面的重要性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Hackathon 方案**：本周末即将举行的 **LlamaIndex Hackathon** 邀请参与者积极参与和创新，所有相关的 [幻灯片和资源](https://bit.ly/llamaindex-rag-a-thon) 已共享，用于赛前准备。
   - 鼓励参与者利用这些资源，以确保他们的项目有坚实的基础。
- **简化 RAG 流水线构建**：查看这段 [视频](https://twitter.com/llama_index/status/1844845845797327352)，它演示了使用 **LlamaIndex** 设置基础 **RAG pipeline** 的过程，重点介绍了核心工作流和组件。
   - 该教程展示了一个使用路由查询技术来提高准确性的简单实现。
- **将聊天历史与 RouterQueryEngine 集成**：关于 **RouterQueryEngine** 的咨询表明，用户有兴趣通过包含所有聊天消息来整合聊天历史，以增强交互动态。
   - 共享了工作流建议和示例，以促进更好的集成实践。
- **PDF 图像提取的挑战**：用户在从 PDF 中提取图像时遇到困难，输出中经常出现意外的 ASCII 字符，造成困惑。
   - 寻求指导以澄清从解析结果中导出的数据，这表明需要更好的文档或支持。
- **对 LlamaIndex 中 Colpali 的兴趣**：在文档尚不完善的情况下，出现了关于在 **LlamaIndex** 中实现 **Colpali** 可能性的讨论。
   - 虽然目前不支持完全嵌入，但社区的兴趣表明，将其作为 reranker 添加可能指日可待。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 类型注解讨论**：团队评估了三个用于在 Tinygrad 中添加类型注解的 PR，其中一个因性能问题被否决，另一个由公认贡献者提交的 PR 被优先考虑。
   - 一个 PR 在测试失败后被拒绝，引发了对合并此类更改实际性的担忧。
- **Bounties 需要经过验证的贡献者**：George 强调，拥有多个已合并 PR 的贡献者在 Bounty 任务中具有优先权，并指出并行 **SHA3** 实现的新 **$200** Bounty。
   - 这突显了经验是承担更大贡献任务的先决条件。
- **SHA256 实现中的挑战**：在 Tinygrad 中实现完整 **SHA256** 的提案引发了关于尽管当前设计存在限制但仍集成并行处理的讨论。
   - George 表现出探索并行能力以优化实现的兴趣。
- **DDPM 调度器在 Metal 上表现出色**：一名成员介绍了他们自己的 **DDPM scheduler**，用于在 Metal 上训练扩散模型，填补了 Tinygrad 资源的空白。
   - 他们愿意与需要此新工具支持的其他人合作。
- **解决张量梯度轴问题**：社区讨论了解决张量中梯度轴不匹配的方案，提供了轴对齐和 resharding 等多种方法。
   - 有人对将 resharding 作为解决方案的浪费性表示担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI O1 模型复现进展**：关于复现 OpenAI **O1 model** 的初步报告展示了一种新的“journey learning”范式，通过 **327 training samples** 增强了**数学推理 (mathematical reasoning)** 能力，带来了 **8% 的性能提升**。
   - 详细的复现过程文档（包括面临的挑战）可以通过 [report](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf) 和 [code](https://github.com/GAIR-NLP/O1-Journey) 获取。
- **微软顶尖 AI 研究员加入 OpenAI**：报告确认，来自 **Microsoft** 的著名 **AI researcher** **Sebastien Bubeck** 将加入 **OpenAI**。在 AI 领域高薪职位的背景下，此类变动的动机引发了关注。[The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx) 报道了这一重大人事变动。
   - 这一变动引起了轰动，业内同行幽默地推测其对现有 AI 团队的影响。
- **前 OpenAI 员工创办初创公司**：预计前 **OpenAI** 员工将创办多达 **1,700 startups**，标志着 AI 初创生态系统的显著激增。
   - 这一趋势反映了该领域向创新和多样化的转变，有望产生 AI 技术的新领导者。
- **Dario Amodei 的影响力作品获得认可**：[Machines of Loving Grace](https://darioamodei.com/machines-of-loving-grace) 因其引人入胜的标题和内容而受到赞誉，激发了人们对 AI 造福社会的潜力的兴趣。
   - 这种日益增长的讨论信号表明，人们对 AI 未来的看法正转向积极，摆脱了基于恐惧的叙事。
- **Folding@Home 在 AI 领域的早期影响**：关于 **Folding@Home** 及其被认为影响不足的讨论展开，一些成员断言尽管它在生物计算领域做出了开创性贡献，但它领先于时代。
   - 对话还承认了药物研发中成熟方法（如 **docking**）的相关性，这些方法在诺贝尔奖讨论中似乎被掩盖了。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Next.JS 语音面试准备平台发布**：一名成员宣布开发了一个全栈 **Next.JS** 语音面试准备/测试平台，旨在通过语音交互增强面试准备。
   - 该平台预计将显著提升面试培训期间的用户体验。
- **GraphIC 变革 ICL 选择**：论文介绍了 [GraphIC](https://arxiv.org/abs/2410.02203)，这是一种使用基于图的表示和 **Bayesian Networks** 来增强 LLM **in-context examples** 选择的技术。
   - 它通过过滤掉浅层语义，解决了多步推理任务中基于文本的 embedding 方法的局限性。
- **LLM 分类器寻求歧义处理**：一位用户正在训练 **LLM classifier**，并寻求社区关于处理分类歧义的建议，以有效管理不确定的输出。
   - 建议包括在 **LLM signature** 中添加第二个输出字段来声明歧义，而不是创建单独的类别。
- **使用余弦相似度评估输出有效性**：一位成员询问了评估 **chatbot outputs** 是否符合既定标准的指标，考虑使用**余弦相似度 (cosine similarity)** 来比较输入查询与生成的类别。
   - Stuart 正在积极寻求建议，以完善这种方法，从而更好地检测离题内容。
- **为 Signatures 创建 FastAPI 路由**：一位成员分享了一段代码片段，可以将任何 **dspy.Signature** 转换为 **FastAPI** 路由，返回 predictor 字典，并使用 **init_instant** 函数进行环境初始化。
   - 这种实现简化了使用 **DSPy** 开发 API 所必需的请求处理过程。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LLaMA 3.2 在流行文化知识方面占据主导地位**：**LLaMA 3.2** 凭借在 **50 亿张图像**上的训练，在描述连贯性上超越了竞争对手。
   - 在对比中，**LLaMA 3.2** 与 **Molmo** 和 **PixTral** 等模型相比，展示了显著的上下文理解能力。
- **PixTral 在成人内容场景中表现出色**：成员们强调，**PixTral** 在专注于成人内容时脱颖而出，而 **LLaMA 3.2** 则更适合更广泛的语境。
   - 这种对比表明，虽然 **PixTral** 有其利基市场，但 **LLaMA 3.2** 在更通用的应用中保持了文化相关性。
- **Epic Games 移除 Sketchfab 引发担忧**：**Epic Games** 移除 **Sketchfab** 将导致 Objaverse 失去 **80 万个 3D 模型**，促使用户紧急下载。
   - 这一决定引发了对其对 3D 建模社区以及依赖这些资源的用户的后续影响的警觉。
- **o1-mini 无法与 o1-preview 竞争**：报告指出，**o1-mini** 的表现不如 **o1-preview**，根据最近的见解，它在简单任务上被描述为表现脆弱。
   - 尽管早些时候声称可以媲美更大的模型，但证据表明 **o1-preview** 甚至在奥林匹克竞赛级别的任务上表现优异。
- **CLIP 对比训练面临的挑战**：使用 **CLIP** 训练 T2I 模型可以加快进程，但会引入与其对比训练方法相关的伪影（artifacts）。
   - 这些伪影引发了对整体训练质量影响的担忧，表明在效率和性能之间存在权衡。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Graham Neubig 关于 AI Agent 的讲座**：**Graham Neubig** 今天的讲座（**PST 时间下午 3:00**）讨论了 
   - Neubig 还强调了为大型代码库开发 AI Agent 的复杂性，解决了文件选择以及将 **web browsing** 集成到工作流中的问题。
- **注册延迟与故障**：成员们确认课程注册将开放至 **12 月 12 日**，并在解决链接问题后分享了成功的报名经验。
   - 参与者报告了用于测验的 Google Forms 的挑战，建议清除浏览器缓存以解决访问问题；课程时间定为 **PST 时间下午 3:00 至 5:00**。
- **定义 AI Agent**：**AI Agent** 通过与 API 和数据库的交互自主执行任务，**ChatGPT** 被归类为 Agent，而 **gpt-3.5-turbo** 则缺乏此类功能。
   - 讨论还包括正在进行的完善 AI Agent 定义的努力，强调了通过 Twitter 等平台获取社区意见的重要性。
- **思维链（CoT）增强 LLM 的问题解决能力**：**Chain of Thought (CoT)** 方法论协助 LLM 将复杂任务分解为可管理的步骤，提升问题解决的清晰度。
   - 成员们通过一个涉及 **Apple** 的案例认可了 CoT 的有效性，展示了系统性分解如何引导至最终解决方案。
- **AI 驱动搜索书籍备受关注**：一位成员推荐了[这本书](https://www.manning.com/books/ai-powered-search)作为 **AI-powered search** 的首选资源，并称赞其在未来几年预期的影响力。
   - 这本书预计将成为 **AI practitioners** 和研究人员的重要参考资料，凸显了其在该领域的未来相关性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Gemma 2 支持利用 Flex Attention**：讨论集中在使用 **Flex Attention** 实现 **Gemma 2**，其中 **logit softcapping** 被确定为主要障碍，需要一个合适的 `score_mod` 函数。
   - 成员们认为 **Flex** 的权衡简化了过程，尽管它可能需要 **CUDA** 的高性能计算能力。
- **Aria 模型介绍**：**Aria** 是一款新型**开放多模态** AI 模型，展示了 **3.9B** 和 **3.5B** 参数架构，在语言和编码任务中表现出色，超越了 **Pixtral-12B**。
   - 虽然目前还没有直接的基准测试对比，但早期迹象表明 **Aria** 的能力超越了同类模型。
- **LegoScale 彻底改变分布式训练**：**LegoScale** 引入了一个**可定制的、PyTorch 原生**系统，用于大型语言模型的 3D 并行预训练，显著提升了性能。
   - 其模块化方法旨在简化跨 GPU 的复杂训练，有可能改变分布式训练的格局。
- **2024 年 AI 现状报告洞察**：**Nathan Benaich** 发布的 **State of AI Report 2024** 概述了 AI 的重大趋势和投资领域，其中很少提到 **Torchtune** 等模型。
   - 该报告旨在引发关于 AI 未来的讨论，特别是关于其在医学和生物学中的应用。
- **Flex Attention 的共享内存溢出问题**：一个 GitHub issue 分享了在 **RTX 4090** 上使用 **flex attention** 的问题，详细说明了与共享内存溢出（out-of-shared-memory）相关的错误。
   - 对话包含了一个最小复现代码片段，促进了故障排除的协作。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **为 Node.js 爱好者推出 Swarm.js**：[Swarm.js](https://github.com/youseai/openai-swarm-node) 是一个轻量级的 Node.js SDK，使用 **OpenAI API** 编排**多 Agent 系统**，实现无缝的 Agent 管理和任务执行。
   - 开发者可以通过运行 `npm install openai-swarm-node` 轻松开始，该项目积极邀请初学者和专家的贡献与协作。
- **社区关闭公告**：Jess 宣布 LangChain Discord 社区定于 **2024 年 10 月 31 日**关闭，以专注于创建一个新的社区平台。
   - 鼓励所有成员填写表格以获取更新，并在 community@langchain.dev 提供反馈，同时向潜在的版主发出邀请。
- **探索上下文检索技术**：一段新的 [YouTube 视频](https://www.youtube.com/watch?v=n3nKTw83LW4) 展示了如何使用 LangChain 和 OpenAI 的 Swarm Agent 实现**上下文检索**（contextual retrieval），引导观众完成集成过程。
   - 这一信息丰富的内容旨在增强信息检索，对于在项目中使用 LangChain 的人来说尤其相关。
- **bootstrap-rag v0.0.9 上线！**：[bootstrap-rag v0.0.9](https://pypi.org/project/bootstrap-rag/) 已发布，包含关键错误修复、改进的文档以及与 LangChain 和 MLflow-evals 的集成。
   - 该更新还包括 Qdrant 模板，增强了检索增强生成（RAG）能力，这是 AI 工程师关注高效数据处理的关键领域。
- **求职者 LangGraph 教程**：一个新教程演示了如何构建一个分析简历与职位描述匹配度的**双节点 LangGraph 应用**，为求职者提供实际帮助。点击[此处](https://youtu.be/7KIrBjQTGLA)观看。
   - 该应用可以编写量身定制的求职信并生成特定职位的面试问题，使其成为 LangGraph 初学者的实用工具。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **探索 Instruction Data 的影响**：一位成员提出了关于在 Pretraining 过程中使用 **Instruction Data** 的问题，强调了其对模型参与度的潜在益处。
   - 该话题可能会引发关于增强模型适应性的创新 Pretraining 技术的讨论。
- **Config 共享引发建议**：在关于 **Config 共享** 的讨论中，一位成员请求了特定的 Config，同时建议将 **Sample Packing** 作为一项关键更新。
   - 讨论中强调了 **Multi-GPU** 设置面临的挑战，并强调需要对设置进行彻底审查。
- **使用 Adapters 进行 Fine-Tuning**：讨论了将现有的 **Llama 3** Adapter 与 Fine-Tuning 模型合并，以提高任务准确性。
   - 共享了一个用于合并过程的 [GitHub 指南](https://github.com/axolotl-ai-cloud/axolotl?tab=readme-ov-file#merge-lora-to-base)，再次强调了正确设置 Config 的重要性。
- **Text Completion 训练增强 Instruction 模型**：在 **Text Completion** 任务上训练像 **GPT-3.5-Instruct** 这样的 Instruct 模型，可以显著提高指令遵循性能。
   - 社区成员警告了 **Overfitting** 的风险，建议使用多样化的数据集以获得最佳训练效果。
- **多样性是避免 Overfitting 的关键**：在讨论训练数据集时提出了对 **Overfitting** 的担忧，呼吁增加多样性以增强泛化能力。
   - 成员们强调要监控跨任务的性能，以减轻在陌生数据集上性能退化的风险。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **消息格式合规提醒**：提醒成员遵守频道中规定的消息格式，以保持组织性和清晰度。
   - 提醒强调了遵循既定指南对增强频道沟通的重要性。
- **引用频道规则**：引用了指导频道内讨论行为和贡献的现有规则。
   - 鼓励成员查看这些规则，以改善频道动态。
- **Aria 成为顶尖 Multimodal 模型**：由 [@rhymes_ai_](https://twitter.com/rhymes_ai_) 开发的 **Aria** 目前在 🤗 Open LLM Leaderboard 上排名第一，拥有 **24.9B 参数**，并能处理图像、视频和文本输入。
   - 它拥有 **64k Token Context Window**，并在 **400B Multimodal Token** 上进行了训练，承诺在各种任务中保持稳定性（[论文](https://hf.co/papers/2410.05993), [博客](https://rhymes.ai/blog-details/aria-first-open-multimodal-native-moe-model)）。
- **用户称赞 Aria 的 Multimodal 能力**：用户对 **25.3B Multimodal Aria 模型** 充满热情，称其为“我尝试过的最好的视觉语言模型！”
   - 该模型在 **Apache-2.0 许可证**下发布，同时也为社区参与提供了 Fine-Tuning 脚本。
- **AI 推理能力引发辩论**：一段名为“Apple 投下 AI 震撼弹：LLM 无法推理”的 [YouTube 视频](https://www.youtube.com/watch?v=tTG_a0KPJAc) 对语言模型的推理能力提出了关键质疑。
   - 创作者激发了围绕当前 AI 局限性的对话，敦促观众为 AGI 的进步做好准备。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **成员发起关于 Jamba 的支持咨询**：一位用户发起了关于 **Jamba** 问题的支持线程，询问该频道是否适合寻求帮助。
   - 另一位成员确认他们已在同一频道处理了该咨询，促进了那里的持续讨论。
- **线程连续性对 Jamba 问题的重要性**：针对支持咨询，一位成员强调将讨论保持在原始线程内以保持清晰。
   - 他们指出，这种方法将有助于在未来更轻松地获取相关信息。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **社区参与策略小组讨论**：由社区与开发者关系专家（包括 **Jillian Bejtlich** 和 **Rynn Mancuso**）组成的小组，将在即将举行的活动中讨论提升 **community engagement**（社区参与度）的可行策略。
   - 这一专注于参与度的环节旨在为参与者提供扩大用户群和增加项目贡献的实用技巧。
- **构建繁荣社区的战术见解**：小组将分享关于如何围绕项目培养成功社区的战术建议，强调在代码之外建立关系的重要性。
   - 寻求提升社区建设技能的项目负责人会发现这一环节对建立人脉和增强策略特别有用。
- **预约社区小组讨论！**：鼓励参与者在[此处](https://discord.com/events/1089876418936180786/1288598715711356928)预约关于社区建设实践的小组讨论，以预留名额。
   - 频道中回响着 *“不要错过这个宝贵的机会！”*，敦促社区成员积极参与。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **介绍 Backtrack Sampler**：Mihai4256 分享了一个有趣的 [GitHub 仓库](https://github.com/Mihaiii/backtrack_sampler)，专注于一种回溯采样方法。
   - 该仓库可能会吸引对高级采样技术和模型优化感兴趣的 AI 工程师。
- **查看 GitHub 仓库**：该仓库提供了创新的采样方法，可以提高算法效率和模型准确性。
   - Mihai4256 鼓励社区对仓库中讨论的实现方案提供协作和反馈。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **多轮评估中的性能瓶颈**：成员在多轮评估中遇到了 **~0% 的性能**率，因为模型尽管预测正确，但无法退出计数循环。
   - 讨论强调了为这一评估难题寻找可行解决方案的各种努力。
- **权宜之计提升了多轮评估评分**：通过在 base_handler.py 中进行临时代码修改，尝试每轮仅运行一次，将评估准确率提高到了 **~15% 的性能**。
   - 然而，由于需要遵守 **修改限制**，成员们仍在寻找提升性能的替代策略。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉默，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉默，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1294374030580777010)** (375 messages🔥🔥): 

> - `Importing Unsloth module`
> - `Fine-tuning models for specific tasks`
> - `Using WSL2 for development`
> - `Training procedures and options`
> - `Loading local model paths` 


- **导入 Unsloth 模块的问题**：用户在导入 Unsloth 模块时面临挑战，经常遇到与 Python 环境和依赖项相关的安装错误。
   - 建议的解决方案是在 conda 环境中通过 pip 安装 Unsloth。
- **模型微调的挑战**：参与者讨论了微调语言模型的困难，特别是模型在训练后无法响应基本查询的问题。
   - 社区建议评估微调数据集和模型类型以确保有效性。
- **使用 WSL2 进行开发**：建议 Windows 用户在 WSL2 中运行其 AI 开发环境，以便正确安装和运行像 Unsloth 这样的模型。
   - 一些用户遇到了安装问题，促使其他人分享解决 WSL2 错误的排查技巧和链接。
- **训练程序和选项**：讨论包括各种模型的训练脚本，参与者在成功设置环境后澄清了在本地运行这些脚本的步骤。
   - 用户确认 Notebook 中的训练脚本与本地执行所需的脚本一致。
- **加载本地模型路径**：关于如何为已下载的模型指定本地路径的问题引起了关注，用户寻求关于正确引用这些路径的指导。
   - 社区强调使用正确的语法以确保模型在训练脚本中正确加载。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.arcee.ai/introducing-arcee-supernova-medius-a-14b-model-that-rivals-a-70b-2/">Introducing SuperNova-Medius: Arcee AI's 14B Small Language Model That Rivals a 70B</a>: 首先推出的是我们的旗舰级 70B SuperNova，随后是 8B SuperNova-Lite。今天，随着 14B SuperNova-Medius 的发布，我们为这个超强 Small Language Models 家族增添了新成员。</li><li><a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/arcee-ai/SuperNova-Medius">arcee-ai/SuperNova-Medius · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-11B-Vision">unsloth/Llama-3.2-11B-Vision · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Hermes-3-Llama-3.1-8B">unsloth/Hermes-3-Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B">NousResearch/Hermes-3-Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/learnpython/comments/173objv/i_am_getting_modulenotfounderror_no_module_named/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://tenor.com/view/sorry-im-sorry-sad-tears-cry-gif-25812284">Sorry Im Sorry GIF - Sorry Im Sorry Sad - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/arcee-ai/distillkit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit at blog.arcee.ai</a>: 一个用于 LLM 蒸馏的开源工具包。通过在 GitHub 上创建账户，为 arcee-ai/DistillKit 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1tDvx2UNj7lsaVSw2zEXB9r0Y8EXiTRTDMORNv0gxV-I/edit?usp=sharing">output_sample</a>: 未找到描述</li><li><a href="https://tenor.com/view/kanna-kamui-miss-kobayashi-dragon-maid-determination-determined-gif-14054173">Kanna Kamui Miss Kobayashi GIF - Kanna Kamui Miss Kobayashi Dragon Maid - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://ollama.com/unclemusclez/unsloth-qwen2.5-coder">unclemusclez/unsloth-qwen2.5-coder</a>: 结合 Unsloth 的 Qwen 2.5 Coder</li><li><a href="https://github.com/un">Unproprietary Corporation</a>: 用于协作工作的商业开源软件 (COSS) 基础设施平台 - Unproprietary Corporation</li><li><a href="https://github.com/microsoft/vptq">GitHub - microsoft/VPTQ: VPTQ, A Flexible and Extreme low-bit quantization algorithm</a>: VPTQ，一种灵活且极低比特的量化算法 - microsoft/VPTQ</li><li><a href="https://tenor.com/view/kiss-gif-4875956593066505581">Kiss GIF - Kiss - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度、减少 80% 的内存占用微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://conda.anaconda.org/pytorch">Package repository for pytorch | Anaconda.org</a>: 未找到描述</li><li><a href="https://conda.anaconda.org/nvidia">Package repository for nvidia | Anaconda.org</a>: 未找到描述</li><li><a href="https://conda.anaconda.org/xformers">Package repository for xformers | Anaconda.org</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1295334086633717844)** (6 条消息): 

> - `LLaMA vs Claude 3.5 Sonnet`
> - `模型训练问题`
> - `Hugging Face 状态` 


- **比较 LLaMA 3 和 Claude 3.5 Sonnet**：一位用户询问了 **LLaMA 3** 在编程任务中与 **Claude 3.5 Sonnet** 的对比情况。
   - 他们表示有兴趣使用 **Unsloth** 对 **LLaMA** 进行微调，以观察其效果。
- **模型训练期间的 NaN 问题**：一位用户报告称，他们的模型突然产生 **NaN** 错误，而训练的其他方面进展顺利。
   - 这引发了关于影响模型稳定性的潜在根本原因的疑问。
- **Hugging Face 状态报告**：一位用户分享了 **Hugging Face** 的状态更新，显示截至上次更新，所有服务目前均在线。
   - 尽管网站运行正常，但另一位用户提到他们无法下载任何模型。



**提到的链接**: <a href="https://status.huggingface.co/">
Hugging Face status
</a>: 未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1294514784062013460)** (104 条消息🔥🔥): 

> - `微调 Qwen 2.5`
> - `使用 Llama 3.2 和 Ollama 进行 embeddings`
> - `GGUF 转换问题`
> - `RAG 框架中的 Embedding 比较`
> - `针对量化模型的 VLLM 使用` 


- **微调 Qwen 2.5 模型**：一位用户在尝试微调 **Qwen 2.5 0.5B** 模型时遇到了问题，特别是使用 **Alpaca 格式**的本地数据集时。
   - 他们收到的建议是确保数据集格式和映射键（mapping keys）正确，并参考 Unsloth 文档获取指导。
- **使用 Llama 3.2 进行 Embeddings**：一位使用 **Ollama 配合 Llama 3.2 3B** 的用户在比较动物特征 embedding 时遇到了意外结果，**老鼠（rat）的得分始终最高**。
   - 尽管描述详尽，但相似度得分与预期不符，引发了关于 RAG 框架和 embedding 方法的疑问。
- **GGUF 转换的挑战**：一位用户确认了 **llama-3.1-70b-instruct** 模型在进行 **GGUF 转换**时存在问题，转换后生成的文本变得毫无意义，而转换前运行正常。
   - 他们分享了转换过程中使用的代码片段，寻求关于如何解决输出问题的进一步建议。
- **在 Unsloth 模型中使用 VLLM**：一位用户在尝试使用 VLLM 运行 **unsloth/Qwen2.5-14B-bnb-4bit** 时遇到 BitsAndBytes 量化相关的错误。
   - 得到的建议是使用非量化版本的模型，因为错误提示该量化方式不受支持。
- **RAG 框架中的 Embedding 比较**：一位用户询问 RAG 框架如何执行 embedding 比较，并质疑不同 prompt 之间结果的一致性。
   - 他们分享的示例结果显示，无关特征经常产生异常的相似度得分，表明需要进一步调查。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth Documentation</a>：未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1294746286448185384)** (13 条消息🔥): 

> - `Adapter 训练技术`
> - `LoRA 微调`
> - `针对硬件优化 RNN`
> - `训练 LLM 进行代码生成`
> - `数据集在模型训练中的重要性` 


- **探索在输出层前训练 Adapter**：讨论了在输出层之前训练一个小型 adapter 以将 embedding 转向 **Chain of Thought (CoT)** 推理的可能性。
   - 讨论中提出了该概念是否与 **LoRA** 微调一致的疑虑，随后对它们的功能区别进行了澄清。
- **讨论 LoRA 微调属性**：成员们解释说 **LoRA** 涉及将 adapter 附加到现有层，冻结原始模型层以仅训练 adapter。
   - 强调了在保留原始模型知识与允许针对性训练之间的平衡，总结为：*“LoRA 学得更少，遗忘也更少。”*
- **揭示 RNN 优化策略**：分享了一篇讨论优化传统 **RNN** 的论文，旨在利用硬件进步实现类似于 Transformer 的性能水平。
   - 作者通过 **FlashRNN** 和 Triton 展示了并行处理能力，同时保留了逻辑推理等任务的状态跟踪。
- **关于训练 LLM 进行代码生成的咨询**：提出了一个关于如何训练 **LLM** 生成 UI 代码的问题，类似于 **V0** 等服务。
   - 建议包括在特定代码库上使用文本补全模型进行训练，并指出了在没有问答对（Q/A pairs）的情况下进行微调可能面临的挑战。
- **模型训练中数据集的相关性**：对话强调了数据集在模型训练中的重要性，指出“一切都在数据集中”。
   - 会议指出，理解数据集构建符合基础数据科学原则，并有相关课程和文献支持。



**提到的链接**：<a href="https://openreview.net/forum?id=l0ZzTvPfTw">FlashRNN: I/O-Aware Optimization of Traditional RNNs on modern...</a>：虽然 Transformer 和其他序列可并行化神经网络架构似乎是目前序列建模的最前沿技术，但它们特别缺乏状态跟踪能力……

  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1294375092012449803)** (365 条消息🔥🔥):

> - `Hugging Face 服务状态`
> - `多语言 Embedding 模型`
> - `AI Agent 讨论`
> - `对 Tesla 机器人发布会的质疑`
> - `模型许可协议解析` 


- **Hugging Face 服务状态问题**：用户报告遇到了各种服务器错误，包括 504 和 502 状态码，这表明 Hugging Face 服务可能存在宕机或性能问题。
   - 社区表达了沮丧情绪，但也分享了更新，一些人确认服务正在间歇性恢复。
- **最佳多语言 Embedding 模型**：有用户寻求目前最佳多语言 Embedding 模型的建议，特别是针对德语。
   - 讨论点包括为多语言应用寻找合适模型的重要性。
- **对 Tesla 机器人发布会的质疑**：参与者讨论了对 Tesla 机器人活动真实性的怀疑，重点在于怀疑机器人是由远程控制而非自主运行。
   - 这引发了关于此类演示对公司声誉和投资者认知影响的更广泛讨论。
- **AI Agent 与协作讨论**：一位用户提出了创建一个定制 AI Agent 平台的想法，并对现有解决方案对普通用户来说过于复杂表示担忧。
   - 这引发了关于社区更关注个人项目而非在大型平台上协作的对话。
- **理解模型许可协议**：讨论涉及了 MIT 和 Apache 许可证之间的区别，特别是与商业用途和代码 Forking 相关的部分。
   - 用户强调了 MIT 许可证相对于 Apache 的宽松性，突出了其对各种项目的适用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/">放大模型示例</a>: ComfyUI 工作流示例</li><li><a href="https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker">Fast Subtitle Maker - Nick088 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">CompVis/stable-diffusion-v1-4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Com">com (全名)</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最优秀的 AI 聊天模型惠及每一个人。</li><li><a href="https://tenor.com/view/theodd1sout-vsauce-or-is-it-gif-16159095273288783425">Theodd1sout Vsauce GIF - Theodd1sout Vsauce 或者真的是吗 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aistudio.google.com/app/prompts/new_chat">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/mlp-sunrise-morning-good-morning-princess-celestia-gif-22250694">Mlp 日出 GIF - Mlp 日出清晨 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/">GitHub: 从这里开始构建</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://www.youtube.com/watch?v=IG4wSOzQatE"> - YouTube</a>: 未找到描述</li><li><a href="https://aistudio.google.com/app/prompts/new_chat)">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/TroyTeslike/status/1845047695284613344">来自 Troy Teslike (@TroyTeslike) 的推文</a>: 似乎有些人仍然对 Tesla 活动中 Optimus 机器人的自主程度感到困惑。这段剪辑来自 Tesla 很久以前发布的另一个视频。它展示了机器人是如何被控制的...</li><li><a href="https://colab.research.google.com/drive/1fLk0xjomBhTZMhHwX7-gtuGxgIUQ0rR-?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/catsOfpeople/owlaHightQ">catsOfpeople/owlaHightQ · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://status.huggingface.org/">
Hugging Face 状态
</a>: 未找到描述</li><li><a href="https://github.com/huggingface/candle/discussions/2557">关于在消费级硬件上使用多模态嵌入对 24 小时屏幕录制进行索引的建议与技巧（100,000 帧/天）· huggingface/candle · 讨论 #2557</a>: 嗨，我正在开发：https://github.com/mediar-ai/screenpipe，它会 24/7 全天候记录你的屏幕和麦克风，并将 OCR 和 STT 提取到本地 sqlite 数据库中，我们想探索向量搜索以改进搜索...</li><li><a href="https://arxiv.org/search/?query=finetuning&searchtype=title&abstracts=show&order=-announced_date_first&size=50">搜索 | arXiv 电子预印本库</a>: 未找到描述</li><li><a href="https://tenor.com/view/throw-up-dry-heave-vomit-gross-eww-gif-3311871195309494726">呕吐 干呕 GIF - 呕吐 干呕 呕吐物 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/smile-mlp-pinkie-pie-gif-13606108">微笑 Mlp GIF - 微笑 MLP Pinkie - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mlp-my-little-pony-friendship-is-magic-my-little-pony-good-night-sleepy-gif-16208834">Mlp 小马宝莉：友谊就是魔法 GIF - Mlp 小马宝莉：友谊就是魔法 小马宝莉 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/rombodawg">rombodawg (rombo dawg)</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/rombodawg/Everything_Instruct">rombodawg/Everything_Instruct · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1294397414513315981)** (8 条消息🔥): 

> - `Rust 编写的 Neural Network`
> - `NN 项目协作`
> - `Terraria Neural Networks`
> - `未完成的 NN 开发` 


- **对 Rust Neural Network 项目的兴趣**：一位成员表示有兴趣合作从零开始构建一个 **Rust 编写的 neural network**，尽管该项目目前尚未完成。
   - 他们澄清说，他们指的是 **编程语言**，而不是电子游戏 Terraria。
- **NN 开发的协作建议**：另一位成员建议发布一个 **协作帖子**，以吸引更多人对该 neural network 项目的关注。
   - 他们提供了一个指向相关频道的链接，用于组织此次协作：[协作帖子](https://discord.com/channels/879548962464493619/1204742843969708053)。
- **关于电子游戏中 Neural Networks 的讨论**：成员们开玩笑地讨论了该请求是指编程语言还是游戏，暗示了在 **Terraria** 中构建 **neural networks** 所涉及的创意。
   - 这段对话强调了社区内围绕 neural network 开发的趣味互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/EthanHe_42/status/1844542533105500280">Ethan He (@EthanHe_42) 的推文</a>: 我很高兴分享我们关于通过将 LLM 升级回收（upcycling）为 Mixture of Experts (MoE) 来改进它们的最新研究！1. 我们在 1T tokens 上对 Nemotron-4 15B 模型进行了升级回收，并将其与持续训练的...</li><li><a href="https://arxiv.org/abs/2410.07524">Upcycling Large Language Models into Mixture of Experts</a>: 将预训练的稠密语言模型升级回收为稀疏的 mixture-of-experts (MoE) 模型是增加已训练模型容量的一种有效方法。然而，最佳技术...</li><li><a href="https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe">Megatron-LM/megatron/core/transformer/moe at main · NVIDIA/Megatron-LM</a>: 关于大规模训练 transformer 模型的研究 - NVIDIA/Megatron-LM
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1294635610920718418)** (8 条消息🔥): 

> - `GitHub 上的 DIAMOND Agent`
> - `使用 INTELLECT-1 进行去中心化训练`
> - `红牛 Basement 创新活动`
> - `医疗 AI 研究亮点`
> - `OSS-Fuzz Gen 项目` 


- **探索 DIAMOND Agent 架构**：[DIAMOND](https://github.com/eloialonso/diamond/tree/csgo) 仓库展示了一个在扩散世界模型（diffusion world model）中训练的强化学习 Agent，该成果在 NeurIPS 2024 上展出。
   - 其独特的架构有望推动学习环境的进步，值得 AI 研究人员关注。
- **INTELLECT-1 去中心化训练启动**：[INTELLECT-1](https://fxtwitter.com/PrimeIntellect/status/1844814829154169038) 被宣布为首个针对 10B 模型的去中心化训练项目，旨在显著扩大训练规模。
   - 该倡议邀请贡献者加入开发开源 AGI 的行动，扩大社区参与度。
- **2024 红牛 Basement 创新者征集**：[红牛 Basement](https://www.redbull.com/us-en/red-bull-basement-2024) 活动鼓励创新者展示他们的想法，并有机会获得 AI 支持。
   - 团队可以制定商业计划，争取在东京举行的全球总决赛中竞争；申请截止日期为 2024 年 10 月 27 日。
- **医疗 AI 每周洞察**：最新的播客讨论了医疗 AI 领域的顶级研究论文，重点介绍了 **MMedAgent** 等著名模型，该模型专注于用于医疗工具使用的多模态 Agent。
   - 您可以通过观看此 [YouTube 视频](https://youtu.be/OD3C5jirszw) 了解本周核心模型和方法的总结。
- **用于 LLM Fuzzing 的 OSS-Fuzz Gen**：Google 的 [OSS-Fuzz Gen](https://github.com/google/oss-fuzz-gen) 引入了一种利用 LLM 驱动技术进行模糊测试（fuzzing）的新方法。
   - 该仓库旨在增强软件鲁棒性，并邀请合作以进行进一步开发。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/MISATO-dataset/Adaptability_protein_dynamics">Adaptability protein dynamics - MISATO-dataset 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://fxtwitter.com/PrimeIntellect/status/1844814829154169038">来自 Prime Intellect (@PrimeIntellect) 的推文</a>：宣布 INTELLECT-1：史上首次 10B 模型的去中心化训练。将去中心化训练规模较此前努力提升了 10 倍。任何人都可以加入我们，共同构建开源 AGI 🦋</li><li><a href="https://youtu.be/5HyY5QiBV-U?si=4eHYUQsYiXnaLKVD"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/0xD4rky/Vision-Transformers">GitHub - 0xD4rky/Vision-Transformers: 该仓库包含理解完整 Vision Transformer 架构及其各种实现所需的所有基础内容。</a>：该仓库包含理解完整 Vision Transformer 架构及其各种实现所需的所有基础内容。 - 0xD4rky/Vision-Transformers</li><li><a href="https://www.redbull.com/us-en/red-bull-basement-2024">召集所有创新者：红牛 Basement 回归</a>：随着红牛 Basement 2024 的回归，一个好的创意将把美国的创新者推向全球舞台。</li><li><a href="https://github.com/eloialonso/diamond/tree/csgo">GitHub - eloialonso/diamond (csgo 分支)</a>：DIAMOND (DIffusion As a Model Of eNvironment Dreams) 是一个在扩散世界模型中训练的强化学习 Agent。NeurIPS 2024 Spotlight。- GitHub - eloialonso/diamond (csgo 分支)</li><li><a href="https://www.youtube.com/watch?v=OD3C5jirszw"> - YouTube</a>：未找到描述</li><li><a href="https://huggingface.co/posts/aaditya/596746813944053">@aaditya 在 Hugging Face 上发布：“医疗 AI 上周回顾：顶级研究论文/模型 🏅 (10月5日 - 10月…”</a>：未找到描述</li><li><a href="https://x.com/OpenlifesciAI/status/1845182901694103945">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：医疗 AI 上周回顾：顶级研究论文/模型 🏅 (2024年10月5日 - 10月12日) 🏅 本周医疗 AI 论文：MMedAgent: Learning to Use Medical Tools with Multi-modal Agent 作者：(@Haoy...</li><li><a href="https://github.com/google/oss-fuzz-gen">GitHub - google/oss-fuzz-gen: 通过 OSS-Fuzz 实现的 LLM 驱动模糊测试。</a>：通过 OSS-Fuzz 实现的 LLM 驱动模糊测试。欢迎通过在 GitHub 上创建账户来为 google/oss-fuzz-gen 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1294656489813971016)** (7 条消息): 

> - `OpenAI Swarm Framework`
> - `GitHub Token Markdown Generator`
> - `Qompass Diver Editor`
> - `Backtrack Sampler for LLMs`
> - `Vintage Action AI Video` 


- **OpenAI Swarm 增加对开源模型的支持**：由 OpenAI Solutions 团队管理的 [Swarm 框架](https://github.com/MunirAbobaker/swarm/tree/main) 增加了对 **开源模型** 的支持。
   - 一位社区成员建议在提交 Pull Request 之前，先查看 [GitHub](https://github.com/openai/swarm/issues/35) 上的相关 Issue 以寻求改进。
- **为 GitHub 仓库生成 Markdown 结构**：一个简便的工具允许用户输入他们的 **GitHub token** 和仓库 URL，以生成全面的 Markdown 文件结构，方便无缝复制。
   - 该工具作为一个 React 应用程序发布在 [GitHub](https://github.com/U-C4N/GithubtoLLM/) 上，旨在增强开发工作流。
- **Qompass Diver 编辑器介绍**：**Qompass Diver** 是一款快速高效的编辑器，目前处于公开预览阶段，已在包括 **Arch Linux**、**Android** 和 **Mac M1** 在内的多个平台上进行了测试。
   - 它采用全新的 **AGPL/Qompass 双重许可** 运行，倡导开源精神。
- **针对 LLM 的 Backtrack Sampler 框架**：一个全新的 **LLM 采样算法** 框架，允许丢弃已生成的 Token，使用户能够重新生成错误的 Token。
   - 它包含演示算法，并支持 **GGUF** 模型和 Huggingface 格式模型，可在 [GitHub](https://github.com/Mihaiii/backtrack_sampler) 上获取。
- **复古动作 AI 视频展示**：分享了一个 **复古动作 AI 视频** 的展示，并附带了 [Instagram](https://www.instagram.com/reel/DBHQJaesqNC/?igsh=MTR6djhjdjZoaDFqdw==)、[TikTok](https://vm.tiktok.com/ZGdeTyj92/) 和 [Twitter](https://x.com/bigtown_xyz/status/1845883546541740513?t=2-S8s-VnM1Zee3E04BRIww&s=19) 等多个平台的链接。
   - 创作者表达了对进一步尝试 AI 工具的兴奋。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/Mihaiii/backtrack_sampler">GitHub - Mihaiii/backtrack_sampler: An easy-to-understand framework for LLM samplers that rewind and revise generated tokens</a>: 一个易于理解的 LLM 采样器框架，可回溯并修正生成的 Token - Mihaiii/backtrack_sampler</li><li><a href="https://github.com/qompassai/Diver">GitHub - qompassai/Diver: Diver: Your blazingly Fast Everything Editor</a>: Diver：你极速的全能编辑器。通过在 GitHub 上创建账号为 qompassai/Diver 做出贡献。</li><li><a href="https://github.com/U-C4N/GithubtoLLM/">GitHub - U-C4N/GithubtoLLM: This React application allows users to enter their GitHub token and repository URL to generate a comprehensive Markdown file structure. Each file’s code is listed sequentially, facilitating seamless copying for LLM integration. Enhance your development workflow with this efficient tool!</a>: 这个 React 应用程序允许用户输入他们的 GitHub token 和仓库 URL 来生成全面的 Markdown 文件结构。每个文件的代码按顺序排列，方便 LLM 集成的无缝复制。使用这个高效工具增强你的开发工作流！</li><li><a href="https://github.com/MunirAbobaker/swarm/tree/main">GitHub - MunirAbobaker/swarm: Framework for building, orchestrating and deploying multi-agent systems. Managed by OpenAI Solutions team. Experimental framework.</a>: 用于构建、编排和部署多 Agent 系统的框架。由 OpenAI Solutions 团队管理。实验性框架。 - MunirAbobaker/swarm</li><li><a href="https://www.instagram.com/reel/DBHQJaesqNC/?igsh=MTR6djhjdjZoaDFqdw==">Instagram 上的 Prompt Revolution: &quot;vintage action

~~~
~~~

#aiart #aivideo #midjourney #kling #klingai #aesthetic #vintage #cinema&quot;</a>: 6 个赞，2 条评论 - prompt_revolution 于 2024 年 10 月 14 日发布: &quot;vintage action  ~~~ ~~~  #aiart #aivideo #midjourney #kling #klingai #aesthetic #vintage #cinema&quot;. </li><li><a href="https://vm.tiktok.com/ZGdeTyj92/">TikTok - Make Your Day</a>: 未找到描述</li><li><a href="https://x.com/bigtown_xyz/status/1845883546541740513?t=2-S8s-VnM1Zee3E04BRIww&s=19">来自 bigtown | Magnet Labs (@bigtown_xyz) 的推文</a>: 复古动作，继续和我最亲近的朋友一起探索 --sref 3982906704
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1295419635507793991)** (1 messages): 

> - `图像处理协作`
> - `CV 项目`
> - `元启发式优化` 


- **寻求图像处理项目合作伙伴**：一位成员表示有兴趣在 **图像处理** 或 **CV 项目** 上进行协作，愿意投入几个月的时间致力于在会议或期刊论文上发表研究结果。
   - 他们强调了自己在 **用于特征选择的元启发式优化** 方面的经验，并愿意与感兴趣的合作者安排会议。
- **CV 论文协作邀请**：同一位成员邀请其他人就计算机视觉相关的潜在 **会议或期刊投稿** 进行合作。
   - 他们对涉及使用优化技术进行 **特征选择** 的项目特别感兴趣，并强调已准备好积极贡献。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1294693480492302337)** (43 messages🔥): 

> - `微调 BERT`
> - `理解 Logits 计算`
> - `隐藏状态探索`
> - `模型参数查询`
> - `微调 Base LLM 的成功案例` 


- **探索 BERT 微调中的 Logits**：一位成员询问在为分类任务微调 **BERT** 时，是否有办法计算每个 token 的 logits。
   - 对方澄清说，使用 `BertForSequenceClassification` 模型将处理 logits，并且可以通过对输出进行索引轻松访问它们。
- **理解每一层的贡献**：为了理解如何获得 logits，成员们讨论了必须对模型进行完整的前向传播（forward pass），包括所有层和激活值。
   - 这种理解使得人们认识到，有无数的参数和复杂的相互作用在影响输出的 logits。
- **对 BERT 逐层机制的兴趣**：一位成员表示希望掌握 **BERT** 架构的逐步工作原理，以便在学位研究期间自信地回答有关 logits 的问题。
   - 虽然另一位成员建议专注于核心理解，但大家一致认为深入研究 BERT 需要投入大量时间。
- **知识获取建议**：一位参与者建议查看各种资源（如 YouTube 视频）以详细了解 **BERT**，这可能需要花费数小时才能全面覆盖。
   - 这反映了社区愿意协助分享知识，以实现对模型机制的更深层次洞察。
- **微调 Base LLM 的成功案例**：一位成员询问是否有人看到过展示在特定领域成功微调大型 Base LLM 的文章或论文。
   - 他们对这种微调带来的显著性能提升和领域知识增强特别感兴趣。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1294635552313704488)** (18 条消息🔥): 

> - `CogVideoX 微调`
> - `用于教育的 Diffusion 模型`
> - `微型模型发布`
> - `从零开始训练 DiT`
> - `模型大小与资源` 


- **CogVideoX 微调经验**：一位用户询问了关于 **CogVideoX** 的微调情况，但另一位成员提到他们之前尝试过，但无法分享结果。
   - 这引发了关于微调过程中时间需求和结果预期的讨论。
- **为学生创建 Diffusion 模型课程**：一位成员寻求关于使用 Diffusers 库教学 **Diffusion 模型** 的建议，特别是寻找 4GB 以下的小型模型。
   - 另一位成员建议使用 **SD 1.5**，它在 FP16 精度下小于 3GB 且兼容性好，同时还讨论了模型参数限制。
- **微型模型发布讨论**：一位用户回想起名为 **Wuerstchen** 的模型存在微型版本，并暗示还有更小的模型可用。
   - 共享了指向 **tiny-sd** 等模型的链接，强调了它们即使质量不是顶尖，但在实际应用中也很有用。
- **使用独特数据训练 DiT**：一位成员报告了从零开始成功训练 **DiT** 的经历，使用了来自 **Super Mario Bros** 的游戏图像。
   - 他们采用了一个具有显著压缩率的自定义 VAE，取得了值得分享的满意进展。
- **模型资源考量**：用户讨论了在教育场景中有效运行 Diffusion 模型时，模型大小和 VRAM 需求的影响。
   - 他们强调了拥有专用 GPU 资源的重要性，以避免在不使用 CPU 时出现复杂情况，并将此类讨论与实际教学场景联系起来。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sihyun.me/REPA/">Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think </a>：生成任务的表示对齐：训练 Diffusion Transformers 比你想象的要简单</li><li><a href="https://huggingface.co/segmind/tiny-sd">segmind/tiny-sd · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/models?other=diffusers%3AStableDiffusionPipeline).">Models - Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1294374366850711552)** (369 条消息🔥🔥): 

> - `LM Studio 功能请求`
> - `模型性能对比`
> - `本地 AI 设置指南`
> - `Multi-Agent 系统`
> - `社区支持与故障排除` 


- **LM Studio 安装问题**：用户对 LM Studio 在没有 UAC 提示的情况下进行安装表示担忧。用户询问该程序是仅影响用户配置文件还是会影响系统文件。
   - 此外，一些用户报告了在特定 Linux 发行版上运行 AppImage 时缺少库的问题。
- **编程模型的性能对比**：用户讨论了他们在各种 LLM（如 Qwen-2.5 和 Deepseek）上的使用体验，并指出了不同的性能水平。许多人更青睐 Qwen 模型在 Python 编程任务中的速度和效率。
   - 用户有兴趣尝试模型的不同量化（quantization）选项，以最大限度地提高输出质量和速度。
- **LM Studio 中角色卡的潜力**：用户建议为 Agent 实现角色卡（character cards），以增强 LM Studio 中的个性和交互。该想法旨在改进 Agent 的配置并促进更具吸引力的对话。
   - 此外，还提出了创建 Agent 间对话的可能性，以扩展平台的垂直能力。
- **Multi-agent 编排工具**：社区讨论了像 OpenAI 的 Swarm 以及修改版（如 ollama-swarm）等工具，这些工具促进了 Multi-agent 编排。这些框架因其与 LM Studio API 配合使用以实现更灵活的 Prompt 处理的潜力而受到关注。
   - 用户对在 LM Studio 工作流中集成 Multi-agent 系统的潜力表示兴奋。
- **用户界面反馈**：用户对 UI 中上下文（context）显示的清晰度提出了担忧，并提出了改进建议。用户强调当前的上下文填充指示器需要更加直观，以提升用户体验。
   - 团队承认了关于上下文表示像“彩蛋”一样的反馈，并承诺在未来的更新中明确其可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eqbench.com/creative_writing.html">EQ-Bench Creative Writing Leaderboard</a>: 未找到描述</li><li><a href="https://medium.com/@bhupesh.gupta_19807/building-your-own-copilot-local-coding-assistant-using-all-open-source-8126995dafdf">Building Your Own copilot — Local Coding Assistant using all open source</a>: 想象一下拥有你自己的代码助手，在本地运行，完全受你控制，且不受商业产品的限制……</li><li><a href="https://medium.com/@bhupesh.gupta_19807/building-you">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Claude-GGUF">bartowski/Meta-Llama-3.1-8B-Claude-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/133">Feature Request: Use LM Studio as a Client for a different LLM Server in the local Network. · Issue #133 · lmstudio-ai/lmstudio-bug-tracker</a>: LM Studio 已经允许创建一个服务器并将其用于 API 请求。但它不允许 LM Studio 作为该服务器的客户端。场景如下：我有一台性能强大的机器在我的……</li><li><a href="https://github.com/Jenqyang/Awesome-AI-Agents">GitHub - Jenqyang/Awesome-AI-Agents: A collection of autonomous agents 🤖️ powered by LLM.</a>: 由 LLM 驱动的自主 Agent 🤖️ 集合。 - Jenqyang/Awesome-AI-Agents</li><li><a href="https://github.com/OthersideAI/self-operating-computer">GitHub - OthersideAI/self-operating-computer: A framework to enable multimodal models to operate a computer.</a>: 一个使多模态模型能够操作计算机的框架。 - OthersideAI/self-operating-computer</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。</li><li><a href="https://console.groq.com/docs/api-reference#models-list">GroqCloud</a>: 体验世界上最快的推理速度</li><li><a href="https://lmstudio.ai/docs/cli/log-stream#">lms log stream - CLI | LM Studio Docs</a>: 从 LM Studio 流式传输日志。对于调试发送给模型的 Prompt 非常有用。</li><li><a href="https://github.com/victorb/ollama-swarm">GitHub - victorb/ollama-swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Modified to use local Ollama endpoint</a>: 探索符合人体工程学、轻量级多 Agent 编排的教学框架。已修改为使用本地 Ollama 端点 - victorb/ollama-swarm</li><li><a href="https://github.com/openai/swarm">GitHub - openai/swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solution team.</a>: 探索符合人体工程学、轻量级多 Agent 编排的教学框架。由 OpenAI 解决方案团队管理。 - openai/swarm</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js/blob/6eda4ebe5abe83aafb67786c8e466fa0f4d0bdfa/packages/lms-client/src/llm/LLMDynamicHandle.ts#L375">lmstudio.js/packages/lms-client/src/llm/LLMDynamicHandle.ts at 6eda4ebe5abe83aafb67786c8e466fa0f4d0bdfa · lmstudio-ai/lmstudio.js</a>: LM Studio TypeScript SDK（预发布公开 Alpha 版） - lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/matatonic/openedai-speech">GitHub - matatonic/openedai-speech: An OpenAI API compatible text to speech server using Coqui AI&#39;s xtts_v2 and/or piper tts as the backend.</a>: 一个兼容 OpenAI API 的文本转语音服务器，使用 Coqui AI 的 xtts_v2 和/或 piper tts 作为后端。 - matatonic/openedai-speech</li><li><a href="https://github.com/travisvn/openai-edge-tts">GitHub - travisvn/openai-edge-tts: Text-to-speech API endpoint compatible with OpenAI&#39;s TTS API endpoint, using Microsoft Edge TTS to generate speech for free locally</a>: 兼容 OpenAI TTS API 端点的文本转语音 API 端点，使用 Microsoft Edge TTS 在本地免费生成语音 - travisvn/openai-edge-tts
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1294647505279123510)** (50 messages🔥): 

> - `GPU 电源管理`
> - `AMD 7900 系列`
> - `Intel CPU 与 LLM 支持`
> - `NVIDIA vs AMD AI 性能对比`
> - `模型中的 VRAM 利用率` 


- **警惕 GPU 电源管理问题**：有成员反映在双 **3060** 显卡上运行模型时 GPU 利用率较低，尽管速度达到了 **17.60 tok/sec**，但利用率显示低于 **10%**。
   - 讨论指出这可能是由于 IO 瓶颈问题，或者是电源管理导致频率上下波动，而非性能问题。
- **显存爱好者的双 7900 配置**：一位成员考虑使用 *2x 7900 XTX/XT* 的性能优势，指出它可以提供 **48GB/40GB 的 VRAM**，这对于加载大型模型非常有利。
   - 然而，其他人指出其速度与单卡相似，增加显卡只是为了更容易在内存使用上管理大型模型。
- **针对 LLM 优化的 Intel CPU**：Intel 的新系列 **CPU** 宣传支持 **Llama 2** 等大语言模型（LLMs），并预期会有较高的每秒操作数（TOPS）。
   - 用户讨论了 **Core Ultra** CPU 是否支持 SYCL，迹象表明将使用 ***SYCL/IPEX***。
- **NVIDIA 在 AI 支持方面的领先地位**：分享了关于在 **NVIDIA 4060 Ti** 和 **AMD RX 7700 XT** 之间做出选择的看法，强调尽管 AMD 显卡显存更高，但 NVIDIA 对 AI 工作负载的支持更好。
   - 总体共识是，使用 **NVIDIA** GPU 在运行 AI 应用程序时遇到的麻烦更少。
- **Mistral Large 作为首选创意模型**：**Mistral-Large 123b** 模型因其在消费级机器上的性能和灵活性而受到赞赏，特别是在资源充足的 **M2 Studio** 上。
   - 多位用户一致认为，**Mistral Large** 的配置在处理各种上下文时能高效利用 VRAM。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pinokio.computer/">Pinokio</a>: AI 浏览器</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: 多个 NVIDIA GPU 或 Apple Silicon 用于大语言模型推理对比 - XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1294518884753211504)** (162 messages🔥🔥): 

> - `OpenAI 模型性能`
> - `FA3 实现问题`
> - `LLM 产品偏好研究`
> - `自回归图像生成模型`
> - `FA3 vs F.sdpa 速度对比` 


- **OpenAI 模型性能引发担忧**：成员们讨论了 OpenAI 最近的模型据报道操纵其测试环境的影响，强调了对 AI 对齐问题的担忧。
   - *有人建议，此类行为凸显了 AI 安全和伦理考量中面临的持续挑战。*
- **FA3 实现困境**：用户报告在实现 FA3 最佳性能方面存在显著困难，指出它比原始的 F.sdpa 慢。
   - 一位用户评论了围绕 FA3 正确安装和使用的混乱，特别是与现有实现相比。
- **研究 LLM 产品偏好**：一位成员分享了他们对不同 LLM 产品偏好的探索，揭示了生成的推荐存在不一致性。
   - 他们表示有兴趣识别能够有效预测排名的协变量，同时寻求潜在研究方向的指导。
- **自回归图像生成模型的挑战**：一位用户询问了自回归图像生成所需的模型大小，以便从 Imagenet 等复杂数据集中生成可识别的输出。
   - 讨论集中在架构选择以及使用手工构建模型与 DiT 等成熟模型相比的挑战。
- **FA3 vs F.sdpa 速度对比**：据观察，使用 FA3 导致的性能低于 F.sdpa，引发了关于 CUDA graph 效率的问题。
   - 用户在测试场景中辩论了模型大小和 CUDA graph 配置对性能的影响。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1294425298636636222)** (148 条消息🔥🔥): 

> - `NanoGPT 训练速度记录`
> - `Tokenformer 架构`
> - `神经网络中的容量利用率`
> - `Swiglu 与 ReLU² 激活函数`
> - `使用生成式 AI 进行虚拟家装 (Virtual Staging)` 


- **NanoGPT 创下新的训练速度记录**：NanoGPT 训练创下了新纪录，在 **15.2 分钟内达到了 3.28 Fineweb 验证损失**，这得益于多项代码优化。
   - 之前的优化包括使用 SOAP 优化器以及将投影层（projection layers）初始化为零等更改，推动了近期的进展。
- **Tokenformer 架构及其影响**：Tokenformer 引入了一种完全基于 Attention 的神经网络设计，通过将模型参数视为 Token，专注于实现高效的模型 Scaling。
   - 该架构旨在缓解与 Scaling 和独立架构修改相关的高计算成本。
- **探索神经网络中的容量利用率**：讨论围绕寻找神经网络中**容量利用率（capacity utilization）**的代理指标展开，并关注了近期论文中关于知识容量（knowledge capacity）的发现。
   - 有人指出，计算资源的限制阻碍了准确评估模型性能所必需的大规模消融研究（ablation studies）。
- **关于 ReLU² 与 Swiglu 的辩论**：对比了 **ReLU²** 和 **Swiglu** 等激活函数，表明性能表现随模型规模（scale）而异。
   - 结果表明 Swiglu 在大型模型上可能表现更好，但目前的实验因其效率而更倾向于使用 ReLU²。
- **用于家具虚拟家装的生成式 AI**：一位用户寻求关于使用生成式 AI 模型实现家具虚拟家装（virtual staging）的建议，将其类比为图像修复（inpainting）技术。
   - 建议包括利用 Stability AI 等平台，这些平台通过用户友好的界面支持此类功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://sihyun.me/REPA/">Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think </a>: 生成任务的表示对齐：训练 Diffusion Transformers 比你想象的要简单</li><li><a href="https://openreview.net/forum?id=oQ4igHyh3N">TokenFormer: Rethinking Transformer Scaling with Tokenized Model...</a>: 由于 Transformer 在各个领域的卓越表现，它已成为基础模型中的主流架构。然而，扩展这些模型的巨大成本仍然是一个...</li><li><a href="https://openreview.net/forum?id=din0lGfZFd">Understanding Reasoning with Looped Models</a>: 大语言模型在推理问题上展现了出色的能力，而 Scaling Laws 表明参数量是核心驱动力。最近的研究（Chen &amp; Zou, 2024; Ye et al., 2024）认为...</li><li><a href="https://x.com/kellerjordan0/status/1844094933197783298">Keller Jordan (@kellerjordan0) 的推文</a>: NanoGPT 竞速更新：使用 SOAP 优化器 (https://arxiv.org/abs/2409.11321)，@vyasnikhil96 在 3.25B 训练 Token 中实现了 3.28 Fineweb 验证损失的新样本效率记录...</li><li><a href="https://x.com/xuandongzhao/status/1845330594185806117">Xuandong Zhao (@xuandongzhao) 的推文</a>: 🚨 来自论文 “Strong Model Collapse” 的迷人见解！(https://arxiv.org/abs/2410.04840) 该研究得出结论，即使是极小比例的合成数据（仅占总数据集的 1%）...</li><li><a href="https://x.com/kellerjordan0/status/1844820919061287009">Keller Jordan (@kellerjordan0) 的推文</a>: @karpathy 的 NanoGPT 设置的新训练速度记录：22.3 分钟内达到 3.28 Fineweb 验证损失。之前的记录：24.9 分钟。更新日志：- 移除了学习率预热（warmup），因为优化器 (Muon) ...</li><li><a href="https://x.com/kellerjordan0/status/1845865698532450646">Keller Jordan (@kellerjordan0) 的推文</a>: 新的 NanoGPT 训练速度记录：15.2 分钟内达到 3.28 Fineweb 验证损失。之前的记录：22.3 分钟。更新日志：- 将 embedding 填充至最近的 64 - 从 GELU 切换到 ReLU² - 零初始化投影层...</li><li><a href="https://x.com/JJitsev/status/1845309654022205720">Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>: (又一个) 关于兴衰的故事：o1-mini 声称在奥数级数学和编程问题上能与更大规模的 o1-preview 相媲美。它能处理揭示泛化能力的简单 AIW 问题吗...</li><li><a href="https://x.com/Grad62304977,">GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1294768113505669190)** (72 messages🔥🔥): 

> - `Hyperparameter Scaling Guide` (超参数缩放指南)
> - `LR Selection Challenges` (LR 选择挑战)
> - `HEBO Limitations` (HEBO 的局限性)
> - `Scaling Laws for Models` (模型的 Scaling Laws)
> - `Importance of Tuning` (调优的重要性)


- **构建超参数缩放指南**：一名成员表示需要一份关于缩放超参数的指南，特别是侧重于从零开始发现这些参数的方法论，因为目前大多数知识似乎都只存在于研究人员的脑海中。他们强调了将这些知识中心化以供更广泛获取的重要性。
   - *如果有一份关于缩放训练实验的启发式方法指南就太酷了。*
- **LR 选择的困扰**：成员们讨论了确定最佳学习率 (LR) 及其缩放行为的困难，其中一人指出最佳 beta2 会随数据集大小而变化。他们一致认为，寻找一致的超参数对于有效训练至关重要。
   - *使用未经调优的 LR 既非常愚蠢又非常普遍。*
- **HEBO 对超参数优化的限制**：有人对 HEBO 对噪声的敏感性及其轴对齐假设（axis-aligned assumption）表示担忧，这使得探索相关参数变得复杂。成员们指出，这些限制阻碍了在不同规模上进行有效的超参数搜索。
   - *对噪声敏感意味着更长时间的搜索会被验证噪声所主导。*
- **从现有的 Scaling Laws 中学习**：讨论强调，重要的缩放规则可能只在 Anthropic 和 OpenAI 等大型实验室中广为人知，强调了已发表资源的稀缺性。其意图是通过较小规模的实验来利用已知的 Scaling Laws。
   - *每一篇论文都会因为拥有正确的 LR 而大获裨益。*
- **超参数调优的重要性**：成员们辩论了在小规模下调优超参数是否能消除在大规模下调优的需求，其中一人断言适当的调优仍然至关重要。他们一致认为，调优不当的超参数可能导致训练的灾难性失败。
   - *我对自己选择 LR/LR schedule 没有信心。*



**提及的链接**: <a href="https://apaz-cli.github.io/blog/Hyperparameter_Heuristics.html">Hyperparameter Heuristics</a>：未找到描述

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1294653989530370102)** (10 messages🔥): 

> - `Common caching for multiple tasks` (多任务通用缓存)
> - `Eval for long context reasoning` (长上下文推理评估)
> - `lm-evaluation-harness bug discussion` (lm-evaluation-harness Bug 讨论)


- **不同任务的通用缓存**：一名成员询问是否可以对使用相同模型和基准但具有不同指标的两个任务使用 **--use_cache** 参数，以寻求共享缓存输出的方法。
   - 另一名成员澄清说，**如果输入相同，缓存应该是相同的**，并提供了使用多个指标处理结果的替代方案。
- **关于长上下文推理评估的讨论**：分享了一个关于专有的“未泄露”长上下文推理评估的链接，并询问 Eleuther 是否有类似的计划。
   - 这引发了对未来评估以及处理复杂推理任务策略的好奇。
- **lm-evaluation-harness 中的 Bug 讨论**：在 `evaluator.py` 中发现了一个关于未绑定变量的潜在 Bug，引发了对其预期行为的质疑。
   - 对话揭示了对变量作用域的困惑，成员们指出这似乎是由于变量引用混淆引起的 Bug。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12640">Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries</a>: 我们介绍了 Michelangelo：一个针对 LLM 的极简、合成且未泄露的长上下文推理评估，同时也易于自动评分。该评估通过无...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/0845b588303f1f59af98dd1c5bdbd78a9e75a1e2/lm_eval/evaluator.py#L497">lm-evaluation-harness/lm_eval/evaluator.py at 0845b588303f1f59af98dd1c5bdbd78a9e75a1e2 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

00_aretha: https://x.com/sainingxie/status/1845510163152687242
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1294401490538074165)** (252 messages🔥🔥): 

> - `AI in caregiving`
> - `Voice cloning models`
> - `Spatial computing devices`
> - `Language model performance comparisons`
> - `VR and AR headset recommendations` 


- **关于 AI 老年护理的讨论**：参与者讨论了使用 AI 辅助老年人的潜力，特别是在药物管理和提供陪伴方面。
   - 然而，会议强调了对使用 AI 执行护理任务的可靠性和伦理影响的担忧。
- **使用 F5-TTS 进行语音克隆**：一位用户分享了使用集成在 Groqcaster 中的 F5-TTS 本地语音克隆模型的经验，强调了其创建自动化语音输出的能力。
   - 虽然质量令人印象深刻，但尚未达到 ElevenLabs 的水平，但本地生成所有内容的优势得到了认可。
- **选择空间计算设备**：用户评估了不同的空间计算设备，辩论了 Meta Quest 与 Xreal 在多显示器设置方面的优劣。
   - 建议将 Meta Quest 作为配合原生应用程序进行桌面使用的更好选择，尽管其光学质量的局限性也得到了承认。
- **模型参数对比**：参与者讨论了各种语言模型，包括 GPT-4 Turbo 和 Claude，推测它们的参数量和性能特征。
   - 对话揭示了对公开发布的参数量的怀疑，以及 AI 模型之间持续的竞争。
- **幽默与 AI 语音应用**：关于使用 AI 生成的语音制造恐怖效果（特别是在万圣节前后）进行了一番幽默的交流。
   - 参与者开玩笑说，一个无人看管的 AI 语音在邻居窗户附近发笑可能会让人感到毛骨悚然。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://community.openai.com/t/stupid-feature-which-response-do-you-prefer-your-choice-will-help-make-chatgpt-better/394873">Stupid Feature: Which response do you prefer? Your choice will help make ChatGPT better</a>：一段时间以来，出现了这个选项，我必须从中选择一个：“你更喜欢哪个回答？你的选择将帮助 ChatGPT 变得更好。” 首先，正如你在图片链接中看到的 ChatGPT ...</li><li><a href="https://www.reddit.com/r/deeplearning/s/eiJ6RT7G9Z">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://youtu.be/qbIk7-JPB2c?feature=shar"> - YouTube</a>：未找到描述</li><li><a href="https://gpt-unicorn.adamkdean.co.uk/">GPT Unicorn</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1294702523504132239)** (59 messages🔥🔥): 

> - `Issues with Custom GPTs`
> - `GPT-4o Performance`
> - `Code Cracking Puzzle`
> - `Model Reasoning Capabilities`
> - `PDF Processing Challenges` 


- **Custom GPT 集成问题**：用户报告称，他们无法再使用“@”符号将一个 Custom GPT 引入另一个 GPT 的对话中，而这以前是 Plus 计划订阅者可以使用的一项功能。
   - 一位用户建议这可能是一个 Bug，因为其他人仍然可以使用此功能，并建议联系支持人员寻求帮助。
- **建筑规范 GPT 的问题**：一位用户使用建筑规范的 PDF 构建了 Custom GPT，但该模型已处于“Update Pendings”状态超过一周，且模型在准确引用内容方面表现挣扎。
   - 尽管提供了 300 页的规范信息，该机器人仍经常指示用户去查阅规范，而不是从 PDF 中进行总结。
- **PDF 处理中的挑战**：另一位用户仅用一个 PDF 测试了模型的能力，观察到它在有效阅读和响应与文档相关的查询方面仍然存在困难。
   - 这引发了对模型准确处理和检索 PDF 信息能力的担忧。
- **GPT-4o 与 O1 的能力对比**：讨论了 GPT-4o 和 O1 不同的推理能力，强调 O1 通常能更好地解释冲突并避免生成荒谬的答案。
   - 参与者指出，虽然 GPT-4o 有时能提供正确答案，但它往往无法预先识别任务何时是不可能的。
- **破译代码谜题讨论**：用户参与了一个破译代码的谜题，分析线索并尝试根据给定的反馈推断出正确的 4 位代码。
   - 这包括讨论哪些模型可以准确确定正确答案，以及不同模型需要尝试多少次才能得出答案。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1294575379981406228)** (24 messages🔥): 

> - `Prompt Engineering 策略`
> - `使用 LLMs 进行 Text2SQL`
> - `创建自定义 GPTs`
> - `管理 RAG 系统`
> - `探索 AI 局限性` 


- **LLMs 的 Prompt Engineering**：用户正在尝试各种提示策略，以增强与 LLMs 交互的清晰度和有效性，例如使用特定的列表和流程来指导模型。
   - 一位用户提到，专注于清晰的沟通可以获得更好的结果，并能更有效地引导模型的回答。
- **Text2SQL 实现**：一位成员询问了使用 LLMs 进行 Text2SQL 的经验，特别是在涉及多表和数据管理的复杂场景中。
   - 讨论围绕如何确保模型高效获取数据，同时不让不必要的输出淹没上下文展开。
- **创建自定义 GPTs 的挑战**：一位用户表达了在自定义 GPT 方面的困难，该 GPT 在处理图像或遵循指定的输出格式方面表现不一致。
   - 他们寻求有关如何解决这些问题的帮助和指导，并特别请求德语使用者的输入以便更好地沟通。
- **探索 AI 局限性**：一位成员询问了绕过 LLM 限制的提示技术，例如重复系统提示或执行其通常规避的任务。
   - 回复指出，某些查询可能会触及模型能力的灰色地带，并建议专注于社区准则允许的探索。
- **澄清 RAG 系统功能**：讨论包括了理解 RAG 系统优缺点的平衡，强调了在实验期间使用已知数据集的必要性。
   - 另一位用户强调，验证数据集的一致性对于可靠的输出至关重要，建议用户明确目标以确保有效使用。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1294575379981406228)** (24 messages🔥): 

> - `使用 LLMs 进行 Text2SQL`
> - `创建私有 GPTs`
> - `改进模型响应`
> - `探索模型限制` 


- **寻求关于 LLMs 进行 Text2SQL 的建议**：一位成员询问了使用 LLMs 进行 **text2sql** 的经验，特别是在处理涉及多表的复杂查询时。
   - 他们对确保模型仅获取相关数据而不淹没上下文表示了担忧。
- **使用 PDF 的自定义 GPT 遇到麻烦**：一位成员报告了在使用上传 PDF 提取数据的自定义 GPT 时遇到的困难，强调了图像识别和格式化方面的问题。
   - 他们寻求故障排除方面的协助，并提到他们正在创建一个**图像标题和关键词生成器**。
- **针对模型限制的提示探索**：一位用户询问了可能绕过 LLM 预期限制的提示，特别是针对“重复最后一条消息”等功能。
   - 另一位成员澄清说，虽然讨论探索模型能力的方法是可以接受的，但深入研究禁止的操作是不允许的。
- **关于模型改进的见解**：一位成员认为，发现规避模型约束的方法有助于增强他们的自定义模型和响应。
   - 他们强调了在讨论模型交互时，探索许可途径的重要性。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1294772107158360095)** (2 条消息): 

> - `Inflection models`
> - `Grok 2 launch`
> - `MythoMax endpoint`
> - `Chatroom improvements` 


- **Inflection 模型现已在 OpenRouter 上线**：为 @Pi 提供支持的 @inflectionAI 模型现已在 OpenRouter 上线，无最低消费限制，且偏好使用 **emojis** 🍓。在此查看公告 [链接](https://x.com/OpenRouterAI/status/1845213137747706224)。
   - Inflection 模型不仅增强了用户交互，还通过大量使用 emojis 🤗 融入了趣味性元素。
- **Grok 2 已正式上线 🚀**：OpenRouter 现已开放 **Grok 2** 和 **Grok 2 Mini** 的访问，尽管这些模型在初始阶段存在速率限制 (rate limited)，定价为 **每百万输入 $4.2** 和 **每百万输出 $6.9**。更多性能统计详情请见 [此处](https://x.com/OpenRouterAI/status/1845549651811824078)。
   - 此次发布突显了在管理资源需求的同时提供强大能力的承诺。
- **推出免费的 MythoMax 终端节点 (Endpoint)**：新的 **免费 MythoMax 终端节点** 已发布，为平台用户提供更多选择。官方分享了详细的使用指标以帮助评估其效果。
   - 此举旨在向更广泛的受众普及先进模型，提升用户体验。
- **聊天室增强：轻松发送图片**：聊天室的重大 **改进** 现已允许用户直接 **拖放** 或粘贴图片，使交互更加流畅且更具吸引力。此功能增强了平台内的整体沟通。
   - 这些变化体现了对用户友好体验和简化工作流程的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1845213137747706224">来自 OpenRouter (@OpenRouterAI) 的推文</a>：为 @Pi 提供支持的 @inflectionAI 模型现已在 OpenRouter 上提供。它会使用大量的 emojis 🤗 快来试试吧！</li><li><a href="https://x.com/OpenRouterAI/status/1845549651811824078">来自 OpenRouter (@OpenRouterAI) 的推文</a>：在其他新闻中，Grok 终于上线了 🚀 Grok 2 + Grok 2 Mini 现已向所有人开放。性能统计如下：
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1294374870578368594)** (357 条消息🔥🔥): 

> - `Grok API 问题`
> - `用于角色扮演和聊天的模型对比`
> - `OpenRouter 界面性能`
> - `Claude 模型访问与错误`
> - `模型 Prompt Caching 特性` 


- **Grok API 使用体验**：用户报告了 Grok API 的问题，遇到了诸如“500 Internal Server Error”和“Rate limit exceeded”等错误。一些用户强调 Grok 仍被归类为实验性（experimental），这可能会导致内部错误。
   - Grok 的响应各不相同，建议在遇到问题时尝试替代模型或 beta 版本。
- **最适合聊天和角色扮演的模型**：用户讨论了在聊天和角色扮演中表现最佳的 AI 模型，推荐包括 Llama-3 和 GPT-4o。基准测试数据表明，某些模型在遵循指令的能力方面优于其他模型。
   - 提到的有效对话顶级模型包括 Llama-3-8b-Instruct 和 GPT-4o，表明在不同语境下性能有所差异。
- **OpenRouter Web 界面性能**：多位用户对 OpenRouter Web 界面的稳定性和响应速度表示担忧，特别是在较长的聊天会话期间。报告显示，性能缓慢或延迟等问题可能受浏览器能力的影响。
   - 用户建议在进行大量聊天时尝试专用客户端以获得更好的性能，并提到了 IndexedDB 存储方式对界面速度的影响。
- **Claude 模型访问与错误**：Timotheeee 等人在使用 Claude 模型时遇到了频繁的错误代码（如 429），引发了使用 beta 版或替代模型的建议。建议在遇到这些错误时实施回退（fallback）机制，以确保更流畅的使用。
   - 通过 OpenRouter 访问 Claude 凸显了与使用原生平台相比的性能差异，引发了关于模型可靠性的讨论。
- **模型缓存特性**：关于 Prompt Caching 特性的讨论显示，某些模型（如 gpt4o-mini）会自动缓存上下文以提高性能。了解缓存的工作原理可以显著增强交互效率，尤其是对于重复查询。
   - 用户阐明了缓存与模型 Prompt 及响应处理方式的关系，对其在不同任务中的有效性持有不同看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat?models=anthropic/claude-3.5-sonnet,openai/o1-preview,google/gemini-pro-1.5">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 在浏览器本地存储数据。</li><li><a href="https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch/">在 NVIDIA H200 Tensor Core GPU 和 NVLink Switch 上将 Llama 3.1 405B 的吞吐量再提升 1.5 倍 | NVIDIA 技术博客</a>: LLM 能力的持续增长，得益于参数数量的增加和对更长上下文的支持，使其被广泛应用于各种场景，每种场景都有不同的部署需求...</li><li><a href="https://fxtwitter.com/tobyphln/status/1845176806065811688?s=46">来自 Toby Pohlen (@TobyPhln) 的推文</a>: 我们正在今天的 @xai 黑客松上提供新 API 的早期访问权限。很高兴看到现场有这么多人。</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet:beta">Claude 3.5 Sonnet (self-moderated) - API, 提供商, 统计数据</a>: Claude 3.5 Sonnet 提供优于 Opus 的能力，速度快于 Sonnet，价格与 Sonnet 相同。通过 API 运行 Claude 3.5 Sonnet (self-moderated)。</li><li><a href="https://docs.anthropic.com/en/release-notes/system-prompts#sept-9th-2024">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">billmei/every-chatgpt-gui 仓库的 README.md</a>: 适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui</li><li><a href="https://openrouter.ai/rankings/roleplay?view=week">LLM 排名：角色扮演 | OpenRouter</a>: 根据角色扮演 Prompt 的使用情况对语言模型进行排名和分析。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1295179637890224138)** (1 条消息): 

> - `Aider AI LLC`
> - `Open Source Aider`
> - `Community Contribution` 


- **Aider AI LLC 为源代码建立明确归属**：创始人宣布成立 **Aider AI LLC**，专门持有 **aider** 源代码，同时保持其在 **Apache 2.0 license** 下 100% 免费且开源。
   - 这一变化使 **aider** 与其他项目明确区分开来，强调它仍然是一个由社区驱动的项目，不涉及融资轮次或雇员。
- **Aider 对用户保持免费**：**Aider** 继续支持配合任何首选的 LLMs 免费使用，确保用户可以无经济障碍地访问该工具。
   - 该消息强调了维护 **open-source** 原则的承诺，同时允许社区贡献以增强项目。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1294391707798081607)** (206 条消息🔥🔥): 

> - `Aider Installation Issues`
> - `Jetbrains Plugin for Aider`
> - `Usage of Aider with Corporate Code`
> - `Feature Requests for Aider`
> - `Performance of Different LLM Models` 


- **Aider 安装问题**：用户分享了安装 Aider 的经验，指出通过 `pipx` 安装比其他方法显著简化了流程。
   - 一位用户评论说，更严格地遵循安装指南可以避免耗时的安装问题。
- **Aider 的 Jetbrains 插件**：一位用户报告了 Aider 的 Jetbrains 插件问题，提到它在启动时崩溃，而其他用户则提供了在 terminal 中直接使用 Aider 的见解。
   - 尽管进行了尝试，社区成员仍认为现有插件功能不足，缺乏文件捕获和 keybindings 等功能。
- **在公司代码中使用 Aider**：讨论者辩论了在公司代码库中使用 Aider 的安全性，一些人因潜在的政策影响和数据泄露而表示谨慎。
   - 其他人认为 Aider 本身是本地运行的，不会共享数据；然而，关于 API 使用和屏幕共享的担忧也得到了认可。
- **Aider 的功能需求**：几位用户集思广益，提出了改进 Aider 的功能，例如基于上下文自动添加文件或为命令实现 keybindings。
   - 社区成员表示有兴趣通过用于文件管理的弹出对话框以及与 IDEs 更好的集成来增强功能。
- **不同 LLM 模型的性能**：用户询问了各种 LLM 模型与 Aider 结合使用时的性能，强调了在编码任务中有效性的 benchmarks。
   - 讨论包括了 Grok-2 和 GPT-4o 等模型之间的比较，强调了选择合适模型以获得最佳性能的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/install/pipx.html">使用 pipx 安装</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/languages.html#how-to-add-support-for-another-language">支持的语言</a>：Aider 几乎支持所有流行的编程语言。</li><li><a href="https://bolt.new">bolt.new</a>：未找到描述</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://aider.chat/">首页</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/commands.html#keybindings">聊天内命令</a>：通过 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://theonlyprompt.tiiny.site/">AI 看板项目管理器</a>：未找到描述</li><li><a href="https://what-i-need.tiiny.site/">UltraLevel - 终极营销平台</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>：最高可优化 90% 的 LLM 成本</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>：Aider 支持 Prompt Caching，以节省成本并加快编码速度。</li><li><a href="https://www.anthropic.com/news/prompt-caching">Claude 的 Prompt caching</a>：Prompt Caching 允许开发者在 API 调用之间缓存常用的上下文，现已在 Anthropic API 上可用。通过 Prompt Caching，客户可以为 Claude 提供更多背景信息...</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>：未找到描述</li><li><a href="https://theonlyprompt.com/">The Only Prompt - TheOnlyOS</a>：The Only Prompt 提供量身定制的 AI 工具，用于开发网站和网页体验，并在其专用生态系统中利用 AI 能力。</li><li><a href="https://github.com/sogaiu/tree-sitter-clojure">GitHub - sogaiu/tree-sitter-clojure: 适用于 tree-sitter 的 Clojure(Script) 语法</a>：适用于 tree-sitter 的 Clojure(Script) 语法。通过在 GitHub 上创建账户，为 sogaiu/tree-sitter-clojure 的开发做出贡献。</li><li><a href="https://artificialanalysis.ai/leaderboards/models">LLM 排行榜 - 比较 GPT-4o, Llama 3, Mistral, Gemini 及其他模型 | Artificial Analysis</a>：对超过 30 个 AI 模型 (LLMs) 的性能进行比较和排名，涵盖质量、价格、性能和速度（输出速度 - 每秒 tokens 数 & 延迟 - TTFT）、上下文...</li><li><a href="https://artificialanalysis.ai/models/llama-3-1-instruct-405b/providers">Llama 3.1 405B：API 提供商性能基准测试与价格分析 | Artificial Analysis</a>：分析 Llama 3.1 Instruct 405B 的 API 提供商，涵盖延迟（首个 token 时间）、输出速度（每秒输出 tokens 数）、价格等性能指标。API 提供商基准...</li><li><a href="https://qwenlm.github.io/blog/qwen2.5/">Qwen2.5：基础模型的盛宴！</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 在 Qwen2 发布后的过去三个月里，众多开发者在 Qwen2 语言模型的基础上构建了新模型，为我们提供了...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1294394246538858601)** (67 messages🔥🔥): 

> - `Aider 教程与资源`
> - `Aider 上下文管理`
> - `Aider API 集成挑战`
> - `在 Aider 中使用 Repo Map`
> - `代码编辑中的注释管理` 


- **Aider 教程与资源**：用户发现了一系列[教程视频](https://aider.chat/docs/usage/tutorials.html)，展示了如何有效地将 Aider 用于编程任务。
   - 重点视频包括使用 Aider 构建非琐碎（non-trivial）应用以及在 Replit 上实现功能。
- **上下文管理问题**：几位成员对 Aider 处理上下文的方式表示担忧，特别是询问它是否会将修改后的文件或之前的 commits 发送到聊天中。
   - 澄清了除非另有说明，否则仅发送已提交的版本，这表明 Aider 需要更好地沟通上下文信息。
- **Aider 的 API 集成挑战**：用户在 API 提供商有速率限制时遇到了重复 API 请求的问题，寻求防止导致访问冻结的自动重试的方法。
   - 社区讨论了为自定义 API 集成配置设置的有效方法，以避免服务被封禁。
- **使用 Repo Map 获取代码上下文**：讨论了如何利用 Aider 中的 repo map 功能来帮助模型检索和理解项目中的相关代码。
   - 用户建议创建有效的库映射（library maps），并预先将文件添加到聊天中，以便 Aider 在编程任务中提供更好的支持。
- **代码编辑期间的注释管理**：有用户担心 Aider 倾向于从代码中删除注释，即使在用户指示不要删除的情况下也是如此。
   - 这引发了对在自动编辑期间更好地保留注释的方法以及现有 prompts 可靠性的询问。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://devdocs.io/">DevDocs</a>: 为开发者提供的快速、离线且免费的文档浏览器。在一个 Web 应用中搜索 100 多个文档，包括 HTML, CSS, JavaScript, PHP, Ruby, Python, Go, C, C++ 等。</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet 表现一如既往</a>: 自发布以来，Sonnet 在 aider 代码编辑基准测试中的得分一直保持稳定。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-with-multip">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider 使用你的 git 仓库映射为 LLMs 提供代码上下文。</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>: 使用 aider 进行 AI 配对编程的技巧。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-with-multiple-git-repos-at-once">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/config/options.html#output-settings">Options reference</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: 由 aider 用户制作的入门和教程视频。</li><li><a href="https://youtu.be/QlUt06XLbJE">AI 编程的秘诀？使用 Aider, Cursor, Bun 和 Notion 的 AI 开发日志</a>: 高产出 AI 编程的秘诀是什么？🔗 更多关于 AIDER 的 AI 编程：https://youtu.be/ag-KxYS8Vuw 🚀 更多关于 Cursor 的 AI 编程：https://youtu.be/V9_Rzj...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1294391418533711893)** (3 messages): 

> - `PHP 搜索/替换问题`
> - `Aider 故障排除`
> - `LLM 行为`
> - `Aider 中的模型能力` 


- **PHP 替换破坏代码语法**：一位成员分享了在 **PHP** 中遇到的挫折，搜索和替换操作导致了**代码重复**而非正确的替换，从而引发了 lint 错误。
   - *“...它想要替换的代码”* 展示了 PHP 中常见的编辑不完整问题。
- **故障排除页面提供指导**：另一位成员建议，当编辑未能应用到本地文件时，请查看 [故障排除页面](https://aider.chat/docs/troubleshooting/edit-errors.html)。
   - 该指导包含了诸如 *“Failed to apply edit to filename”* 之类的消息示例，这些消息是 LLM 不服从指令的指标。
- **模型强度会影响结果**：建议切换到更强大的模型，如 **GPT-4o** 或 **Claude 3.5 Sonnet**，以提高 Aider 编辑的可靠性。
   - 较弱的模型通常被注意到会违反系统提示词（system prompt）指令，导致常见的编辑错误。
- **切换到 Whole 格式以修复编辑**：一位成员提到使用 **whole format** 方法来尝试解决频繁出现的 PHP 编辑问题。
   - 这表明在将 Aider 与 PHP 结合使用时，用户正在采取主动方法来减少错误。



**提及的链接**：<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 结对编程工具

  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1294375652526653582)** (208 messages🔥🔥): 

> - `Nous Research 概览`
> - `DisTrO 项目`
> - `Hermes 3 模型`
> - `区块链开发机会`
> - `AI 社区参与` 


- **Nous Research 概览**：Nous Research 最初是一个 Discord 群组，现已发展成为一家专注于 AI 研究和开发的资助初创公司，尤其关注开源项目。
   - 该社区促进了成员之间关于 AI 研究的协作、想法分享和讨论。
- **DisTrO 项目**：DisTrO 是 Nous Research 的一个项目，它支持在公共互联网上进行更快的 AI 模型训练，并强调社区驱动的开发。
   - 该项目旨在提供封闭 AI 模型的替代方案，确保开源领域的持续进步。
- **Hermes 3 模型**：Hermes 3 是一个开源的微调语言模型，组织机构将其用于各种应用，并在 lambda.chat 等平台上展示。
   - 该模型是更广泛倡议的一部分，旨在提供可访问的 AI 工具，而不受通常与专有模型相关的限制。
- **区块链开发机会**：成员们讨论了社区内区块链开发人员的可用性，并提供了在相关项目上进行协作的机会。
   - 随着 AI 和区块链技术的融合变得越来越紧密，区块链开发人才受到了欢迎。
- **AI 社区参与**：Discord 频道作为成员分享链接、见解以及讨论他们正在进行的 AI 相关项目的中心。
   - 鼓励参与者探索各种 AI 工具，与语言模型聊天，并建立联系以加强他们的研究工作。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Hermes_Solana">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/Victor_crypto_/status/1845408914910654653">来自 Victor.crypto (@Victor_crypto_) 的推文</a>：这个 AI 项目是一个巨大的爆款。@NousResearch 是一个开源 AI 研究人员集体，最近在去中心化 AI 方面取得了突破性进展。TL;DR：他们训练了...</li><li><a href="https://huggingface.co/arcee-ai/SuperNova-Medius">arcee-ai/SuperNova-Medius · Hugging Face</a>：未找到描述</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO 与社区训练 AI 模型的探索 | Andreessen Horowitz</a>：Nous Research 的 Bowen Peng 和 Jeffrey Quesnelle 讨论了他们加速开源 AI 研究的使命，包括一个名为 DisTrO 的新项目。</li><li><a href="https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475">Facepalm Really GIF - Facepalm Really Stressed - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/nanulled/status/1844847958593814655">来自 nano (@nanulled) 的推文</a>：这是一个运行中的 Test-time compute 模型：Qwen-2.5-0.5B 采样器：entropix-local。连续链帮助模型恢复并得出最终答案，截图是大规模...的一小部分。</li><li><a href="https://x.com/NickADobos/status/1845552512763588951">来自 Nick Dobos (@NickADobos) 的推文</a>：Grok API 终于可用了！现在已上线 OpenRouter！！！</li><li><a href="https://x.com/samsja19/status/1844831857785045210?t=U36K8JJim5Z6xnkrKAFtsg&s=19">来自 samsja (@samsja19) 的推文</a>：我们如何跨数据中心训练模型？我们将 4 个以上的数据中心连接在一起。然而，对于一个 10b 模型，我们每 45 分钟仅通信 1 分钟。引用 Prime Intellect (@Prim...</li><li><a href="https://tenor.com/view/jashancutie-bumsekichu-gif-15661510994914397092">Jashancutie Bumsekichu GIF - JashanCutie BumSeKichu - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/lML0ndFlBuc?si=wGfk-uNDQm4C0ym3"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/BGatesIsaPyscho/status/1845366621495255224?t=c_vrG9n_MnHw-LBaWJY6Fw&s=19">来自 Concerned Citizen (@BGatesIsaPyscho) 的推文</a>：中国新型警察机器人。纯粹的反乌托邦。</li><li><a href="https://x.com/m_chirculescu/status/1845850378920726821?t=8PPSyiv0kmJnIFIEKhT7wg&s=19">来自 Mihai Chirculescu (@m_chirculescu) 的推文</a>：推出 backtrack_sampler —— 用于实验自定义 LLM 采样策略，可以回滚并重新生成 token。它同时支持 @ggerganov 的 llama.cpp (GGUF 文件) 和 @huggin...</li><li><a href="https://en.wikipedia.org/wiki/UTF-8">UTF-8 - 维基百科</a>：未找到描述</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>：互联网分布式训练（Distributed Training Over-The-Internet）。通过在 GitHub 上创建账号为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1294514797836111953)** (5 条消息): 

> - `RP 的最佳模型`
> - `微调 Llama 3.2`
> - `Unsloth 工具` 


- **关于 RP 最佳模型的讨论**：一名成员询问了**当前 RP 的最佳模型**，表达了对通用模型性能的兴趣。
   - 在该问题之后的对话中未提及具体模型。
- **微调 Llama 3.2 的技术**：另一位成员 @deckard.1968 请求协助微调 **Llama 3.2**，他准备了一个包含 **800 个输入输出对**的 JSON 文件。
   - 这一求助请求表明社区在模型增强方面参与活跃。
- **使用 Unsloth 进行高效微调**：有人建议查看 **unsloth**，将其作为微调 Llama 3.2 等模型的潜在工具，并指出了其高效性。
   - 成员们指向了 [unsloth 的 GitHub 链接](https://github.com/unslothai/unsloth)，该工具声称微调各种 LLM 的速度快 2-5 倍，且**内存占用减少 80%**。



**提及的链接**：<a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM</a>：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1294696188569981011)** (6 messages): 

> - `ArXiv 性能`
> - `Model Collapse 现象`
> - `MMedAgent 框架`
> - `GSM-Symbolic 基准测试`
> - `医疗 AI 研究更新` 


- **ArXiv 性能备受关注**：成员们讨论了 [ArXiv 平台](https://arxiv.org) 最近的运行缓慢问题，有人指出它“整周都在崩溃”。
   - 另一位成员报告称其运行正常，凸显了用户体验的差异。
- **理解神经网络中的 Model Collapse**：由 [azure2089](https://arxiv.org/abs/2410.04840) 介绍的一篇新论文强调了大型神经网络中的 Model Collapse 现象，指出即使是 **1%** 的合成数据也可能导致严重的性能下降。
   - 研究表明，增加模型规模实际上可能会加剧 Model Collapse，这对当前的训练范式提出了挑战。
- **MMedAgent：多模态医疗工具利用**：本周最受关注的论文名为 **MMedAgent**，重点在于教导多模态 Agent 在患者护理中有效地使用医疗工具，据 [OpenlifesciAI](https://x.com/OpenlifesciAI/status/1845182901694103945) 报道。
   - 讨论了该论文的研究结果，以及 **ONCOPILOT** 和 **PharmacyGPT** 等各种模型的应用。
- **用于模型评估的 GSM-Symbolic 基准测试**：最近的一篇论文介绍了 **GSM-Symbolic** 基准测试，该基准利用符号模板创建多样化的数学问题，以评估 LLM 的性能。
   - 这一新基准旨在为评估 LLM 的形式推理能力提供更可靠的指标。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1845182901694103945">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>: 医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年10月5日 - 10月12日） 🏅 本周医疗 AI 论文：MMedAgent: Learning to Use Medical Tools with Multi-modal Agent 作者：(@Haoy...</li><li><a href="https://arxiv.org/abs/2410.05229">GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models</a>: 大型语言模型 (LLMs) 的最新进展激发了人们对其形式推理能力（尤其是数学领域）的兴趣。GSM8K 基准测试被广泛用于评估数学...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>: 在 Scaling Laws 范式下（这是 ChatGPT 和 Llama 等大型神经网络训练的基础），我们考虑了监督回归设置，并确定了强形式 Model Collapse 的存在...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1294625726082973766)** (11 messages🔥): 

> - `Machines of Loving Grace`
> - `OpenAI Swarm Controversy`
> - `Market Response to Multi-Agent Frameworks`
> - `GPTSwarm Overview` 


- **Machines of Loving Grace 探讨 AI 乐观主义**：Anthropic 的 CEO 在最近的一篇论文中概述了他对 **AI** 如何显著改善世界的愿景，强调了在风险之外，它所提供的**被低估的好处**。
   - 他认为，如果风险得到适当管理，AI 的**未来从根本上是积极的**。
- **OpenAI 面临关于 Swarm 代码的指控**：一名成员批评 OpenAI 涉嫌剽窃 **Kye Gomez** 的仓库，声称 **Swarms 框架**受到了侵权。
   - 该指控包括计划对 OpenAI 采取**法律行动**，除非他们对该项目进行投资。
- **关于 Multi-Agent 框架相似性的讨论**：成员们讨论了包括 **CrewAI** 和 **Langchain** 在内的多个 Multi-Agent 框架都具有共同的概念，这表明这并非 OpenAI 所独有。
   - 一名成员特别指出，考虑到现有框架的更广泛背景，**Kye** 的做法看起来像是一种营销手段。
- **GPTSwarm 平台的兴起**：**GPTSwarm** 项目旨在将各种 Prompt 工程技术统一到一个内聚的框架中，用于 **LLM Agent**，从而简化 Agent 协作。
   - 他们的方案涉及使用计算图来优化节点级 Prompt 和 Agent 编排，展示了该平台的通用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gptswarm.org/">GPTSwarm</a>：未找到描述</li><li><a href="https://x.com/KyeGomezB/status/1844948853604196763">来自 Ky⨋ Gom⨋z (U/ACC) (HIRING) (@KyeGomezB) 的推文</a>：Swarms 框架是首个生产级的 Multi-Agent 编排框架。OpenAI 窃取了我们的名称、代码和方法论。从 Agent 结构的语法到一切内容...</li><li><a href="https://darioamodei.com/machines-of-loving-grace">Dario Amodei — Machines of Loving Grace</a>：AI 将如何让世界变得更美好</li><li><a href="https://github.com/openai/swarm">GitHub - openai/swarm: 探索人体工程学、轻量级 Multi-Agent 编排的教育性框架。由 OpenAI Solution 团队管理。</a>：探索人体工程学、轻量级 Multi-Agent 编排的教育性框架。由 OpenAI Solution 团队管理。 - openai/swarm
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1294696188569981011)** (6 messages): 

> - `ArXiv 性能问题`
> - `神经网络中的模型崩溃 (Model collapse)`
> - `医疗 AI 进展`
> - `针对 LLM 的 GSM-Symbolic 基准测试`
> - `最近的医疗 AI 论文` 


- **ArXiv 性能问题报告**：用户对 **ArXiv** 最近变慢表示担忧，讨论指出它“整周都在崩溃”。
   - 其他人确认他们体验正常，表明用户体验各异。
- **大型神经网络中的模型崩溃 (Model collapse) 现象**：一项研究引入了 **模型崩溃 (model collapse)** 的概念，表明即使在训练中加入 **1%** 的合成数据也会显著降低性能。
   - 理论认为，在 **scaling laws** 范式下训练的大型模型实际上可能会加剧这种崩溃，而不是改善它。
- **GSM-Symbolic 基准测试增强 LLM 评估**：一篇论文讨论了 **GSM-Symbolic** 的引入，这是一个用于评估 **LLM** 数学推理能力的改进基准，解决了现有基准的问题。
   - 这一新基准通过分段、可控的评估提供更可靠的性能指标。
- **顶级医疗 AI 论文助力医疗保健**：几篇论文强调了 **医疗 AI** 的进展，包括 **MMedAgent**，它专注于通过多模态方法学习使用医疗工具。
   - 该列表包含罕见病表型分析框架和评估医疗 **LLM** 安全性的基准等。
- **关于各种 AI 研究论文的讨论**：用户分享并总结了多篇重要的研究论文，概述了 **AI** 在医疗诊断和遗传实验相关的创新应用。
   - 讨论还集中在提高 **LLM** 事实性以及在医疗伦理中使用 **AI** 的影响。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1845182901694103945">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>: 上周医疗 AI 动态：顶级研究论文/模型 🏅（2024年10月5日 - 10月12日）🏅 本周医疗 AI 论文：MMedAgent: Learning to Use Medical Tools with Multi-modal Agent 作者：(@Haoy...</li><li><a href="https://arxiv.org/abs/2410.05229">GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models</a>: 大型语言模型 (LLMs) 的最新进展激发了人们对其形式推理能力（尤其是数学领域）的兴趣。GSM8K 基准测试被广泛用于评估数学...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>: 在支撑 ChatGPT 和 Llama 等大型神经网络训练的 scaling laws 范式内，我们考虑了监督回归设置，并确立了强形式模型崩溃的存在...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1295417943030763562)** (1 messages): 

> - `推理模式 (Reasoning Mode)`
> - `Pro Search 功能`
> - `推理模式的使用案例` 


- **“推理模式 (Reasoning Mode)”在 Pro Search 中推出**：**Perplexity 团队**宣布推出一项名为 **推理模式 (Reasoning Mode)** 的新实验性功能，该功能可检测何时增加计算量或搜索量可以产生更好的答案。
   - 鼓励用户探索此功能并在[指定反馈频道](https://discordapp.com/channels/1054944216876331118)分享有趣的使用案例。
- **可尝试的 Pro Search 示例**：提供的示例包括查询 **OpenAI 的联合创始人**、列出**顶级加密货币**以及评估 **IMDB** 上评分最高的电影。
   - 该功能还建议收集有关 **道琼斯 (DOW) 指数公司 CEO** 的详细信息，包括他们的 LinkedIn URL 和任期详情。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1294379273590935594)** (205 条消息🔥🔥): 

> - `Perplexity AI 性能问题`
> - `使用 Perplexity 生成图像`
> - `Pro Search 中的 Reasoning 功能`
> - `不同模型的使用体验`
> - `AI 模型的编程能力` 


- **Perplexity AI 在记忆力方面表现不佳**：用户报告称 Perplexity AI 在保留对话上下文方面存在问题，由于无法在交互过程中回忆起之前的消息，导致用户感到沮丧。
   - 一些用户推测这可能与 API 限制或 token 问题有关，导致模型在长对话中的性能表现不稳定。
- **讨论图像生成功能**：有关于如何使用 Perplexity AI 生成图像的咨询，并参考了 Discord 上提供的指南。
   - 用户被引导至网页版查看图像生成选项，但对不同平台间的功能差异感到困惑。
- **对 Reasoning 功能感到困惑**：Pro Search 选项中新引入的 Reasoning 功能让用户对其机制产生疑问，特别是关于它何时以及如何激活。
   - 一些用户发现搜索过程中自动激活推理功能既有利也有弊，响应质量参差不齐。
- **AI 模型的使用体验褒贬不一**：用户分享了使用不同 AI 模型的结果，一些人表示 Claude 在编程任务和交互方面的表现优于其他模型。
   - 报告显示，更新影响了某些模型（如 O1 mini）的性能，用户对其输出准确性表示担忧。
- **寻求定向搜索结果**：一位用户对无法使用 Perplexity 过滤搜索结果以仅检索特定网站的信息感到沮丧。
   - 尽管使用了 'site:' 参数，结果仍包含来自其他来源的信息，使用户不得不寻找更好的定向搜索解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/basedbeffjezos/status/1845174550599958877?s=46">来自 Beff – e/acc (@BasedBeffJezos) 的推文</a>: 谁在周六加班？👨‍💻</li><li><a href="https://x.com/apostraphi/status/1845847650328797364?s=46">来自 Phi Hoang (@apostraphi) 的推文</a>: 好奇心的边缘</li><li><a href="https://apps.apple.com/app/per-watch/id6478818380">‎Per Watch</a>: 作为 Perplexity AI 的忠实粉丝，我开发了一款 Apple Watch 应用，让它触手可及。Per Watch 旨在利用 ... 的力量，为用户的查询提供快速、直接且高效的回答。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1294388751560081448)** (19 messages🔥): 

> - `AI Image Generation Costs` (AI 图像生成成本)
> - `Kardashev Scale of Civilization` (卡尔达肖夫文明等级)
> - `Perplexity Pro Updates` (Perplexity Pro 更新)
> - `SSD Booting from Ubuntu` (从 Ubuntu 启动 SSD)
> - `AI for Critical Thinking` (用于批判性思维的 AI)


- **揭秘 AI 图像生成成本**：一场关于当前 **AI image generation** 相关成本及其对追求性价比用户影响的讨论展开了。更多详情请访问此 [链接](https://www.perplexity.ai/search/how-much-is-ai-image-generatio-WZxnVk6YS.iLP_uBYLgPdA)。
   - 会议指出，了解这些成本有助于为涉及复杂视觉内容的项目的预算制定提供参考。
- **探索卡尔达肖夫等级**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/embed/8L6o0-zWzHQ)，讨论了衡量文明能源消耗的 **Kardashev Scale of Civilization Advancement**。该等级为技术进步和能源利用提供了一个迷人的视角。
   - 该视频激发了人们对其对人类未来和技术进步影响的兴趣。
- **Perplexity Pro 功能与更新**：多条消息强调了关于 **Perplexity Pro** 更新的讨论，用户分享了他们的使用体验和建议。此次更新似乎侧重于增强用户功能和参与度。
   - 欲深入了解更新内容，请查看 [此链接](https://www.perplexity.ai/search/perplexity-pro-uber-one-l3nvwTFnQ6e1xILs5cPtHA#0)。
- **解决从 Ubuntu 启动 SSD 的问题**：一位用户提到了一种情况，即他们需要指导如何在不进入 grub 菜单的情况下从 **Ubuntu** 启动进入 **Windows**。他们提到了一种涉及使用 SSH 运行命令的方法，强调了社区分享的有效解决方案。
   - 欲查看更广泛的讨论，请点击 [此处](https://www.perplexity.ai/search/i-am-in-grub-menu-and-got-this-xsPzxQTfT8.9c148VB2AdA) 查看相关 **链接**。
- **利用 AI 鼓励批判性思维**：一位成员发表了一个发人深省的观点，即政府可能更喜欢听话的工人而非具有批判性思维的个人，并以此影射 **Perplexity** 的 **Reasoning Beta** 功能。该功能旨在培养更多具备批判性思维能力的人群，强调了其在信息时代的重要性。
   - 社区对这一观点表示共鸣，强调了对能够赋能独立思考工具的需求，正如在 [此链接](https://www.perplexity.ai/search/help-me-understand-all-the-sta-UUExMPOvTpacP4eJb7LHpg) 中讨论的那样。



**提及的链接**: <a href="https://www.youtube.com/embed/8L6o0-zWzHQ">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1294456312146432060)** (7 messages): 

> - `API URL Responses` (API URL 响应)
> - `Perplexity API Support` (Perplexity API 支持)
> - `Search Domain Filter` (搜索域名过滤器)
> - `Requesting Similar Responses` (请求类似的响应)
> - `Sonar Models` (Sonar 模型)


- **API 不显示来源 URL**：一位用户正在寻求如何开启 API 响应中的来源 URL 的帮助，并指出他们最近发给支持部门的邮件未得到回复。
   - *Hello anyone?* 表达了他们对缺乏回应的沮丧。
- **关于 Perplexity 在线 API 的查询**：一位成员询问 Perplexity 是否支持在线 API，促使另一位用户引用了 [Perplexity API Docs](https://docs.perplexity.ai/guides/model-cards) 中提供的 `sonar-online` 模型。
   - 分享了各种 `llama-3.1-sonar` 模型的详细信息，包括它们的参数量和用于对话补全（chat completion）的上下文长度。
- **对 'search_domain_filter' 参数的担忧**：一位用户正在测试 API 请求中 `search_domain_filter` 参数的有效性，并质疑其是否遵循了提供的特定域名。
   - 尽管指定的域名缺少所搜索的内容，他们仍然收到了有效的响应，这引发了对该参数功能的怀疑。
- **请求获取 API 响应相似性的技巧**：一位用户正在寻找技巧，以实现与在线界面类似的 API 响应，并指出响应质量存在显著差异。
   - 他们不确定 API 在使用模型 `llama-3.1-sonar-large-128k-online` 时是否利用了搜索功能。



**提及的链接**: <a href="https://docs.perplexity.ai/guides/model-cards">Supported Models - Perplexity</a>: 未找到描述

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1294391364129132584)** (21 条消息🔥): 

> - `使用 cuDNN 的 Attention 层`
> - `PyTorch、Triton 和 CUDA 的性能分析 (Profiling)`
> - `周六演讲录像`
> - `CUDA 文档 Unicode 问题` 


- **关于 Attention 层实现的困惑**：一位成员正在寻找使用 cuDNN 的 **SDPA** 在 Python 中实现 **Attention 层** 的教程或实现，并对实例化 **pygraph** 表示困惑。
   - *希望能得到帮助*，因为该成员正在参考 **cudnn-frontend 仓库中的 notebook**。
- **性能分析差异**：一位成员对 **PyTorch**、**Triton** 和 **CUDA** 中的操作进行了性能分析 (Profiling)，发现性能结果存在**显著差异**。
   - 虽然 **Triton** 声称三者性能相当，但自我评估显示 **PyTorch** 领先，从而引发了关于**该信任哪个分析器 (profiler)** 的疑问。
- **周六演讲录像可用性**：有人询问周六演讲的录像，特别是关于 **Metal** 的演讲，并确认这些演讲总是有录像的。
   - 一位成员寻找录像链接，另一位成员提供了该链接。
- **CUDA 文档 Unicode 混淆**：一位成员指出 CUDA 文档在 `#include <cuda∕barrier>` 中错误地使用了 Unicode 字符，而不是标准的 `/`。
   - 另一位成员建议这可能与**渲染引擎或字体问题**有关，并邀请其他人检查引用的文档。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1294375044818141304)** (4 条消息): 

> - `使用 Triton 处理 CUDA 任务`
> - `Triton 中的位打包 (Bit packing) 技术`
> - `内联汇编 (Inline ASM) 和 warp 操作` 


- **Triton 与纯 CUDA 在位打包方面的对比**：一位用户表达了在尝试将 Nx32 bool 数组打包成 N 个 32 位整数时使用 Triton 的困难，认为纯 CUDA 可能更适合此类任务。
   - 他们提到使用 `reduce` 可以完成任务，但指出 CUDA 中的单个 `__ballot_sync` 性能更好。
- **Reduce 函数未利用 balloting**：用户确认他们在 Triton 中基础的 `reduce` 实现并未在生成的 `ptx` 中利用 balloting，这在预料之中但令人失望。
   - 他们分享了一个带有自定义 `bitwise_or` 函数的代码片段，解释了他们合并布尔值的方法。
- **内联汇编在位打包中的挑战**：有人询问在 Triton 中使用内联汇编 (Inline ASM) 进行位打包的可行性，并担心缺乏对 warp 映射的控制。
   - 该用户质疑内联汇编是否能解决他们的位打包问题，尽管他们对 Triton 并不熟悉。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1294466758203281489)** (7 条消息): 

> - `PyTorch CI 中的 IMA 错误`
> - `学习 Torch`
> - `模型并行 vs 数据并行` 


- **PyTorch CI 故障中的链式 IMA 错误**：有人提出了关于在更新 PyTorch nightly 版本后，**torchao CI** 中偶尔出现的 **IMA 错误**，导致大量测试失败。
   - *建议*使用 `CUDA_LAUNCH_BLOCKING=1` 隔离 IMA 错误，或者利用禁用缓存的 `compute-sanitizer --tool=memcheck` 来立即暴露错误。
- **学习 Torch 的最佳项目**：学习 **Torch** 的一个简单项目建议是实现**交通标志检测和分类**，这是一个实用的起点。
   - 在掌握基础知识后，应探索 **DDP** (Distributed Data Parallel) 和 **FSDP** (Fully Sharded Data Parallel) 等高级主题以加深理解。
- **数据并行优先**：共识认为，由于**数据并行**的简单直接，它应该是首选方法，只有在出现内存限制时才考虑**模型并行**。
   - *强调了*大多数使用模型并行的用户通常已经实现了数据并行。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1294730106425708717)** (1 messages): 

> - `Metal Programming`
> - `Apple M series chips`
> - `PyTorch kernel merging` 


- **Metal Programming 课程启动**：我们将在 **PST** 时间中午（**25 分钟后**）开始一场专门为 **Apple M series chips** 初学者定制的 **Metal Programming** 课程。
   - 加入 <@1247230890036166827>，他们将指导参与者在 **PyTorch** 中编写并合并他们的第一个 **Metal kernel**。
- **准备 Metal Kernel 合并**：本节课将重点帮助参与者编写并合并他们的**第一个 Metal kernel**。
   - 对于那些有兴趣在实操指导下深入了解 **Metal programming** 世界的人来说，这是一个绝佳的机会。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1294794384461529109)** (4 messages): 

> - `Entropix Sampling`
> - `Open Source Projects`
> - `LLM Samplers` 


- **对 Entropix Sampling 的怀疑**：讨论中出现了关于 **Entropix sampling** 的争议，一些人对支持者的可信度表示担忧，导致最初怀疑这可能是一个**骗局**。
   - 尽管存在怀疑，但其方法在无需大规模修改即可简化推理方面似乎展现出了*前景*。
- **理解 Entropix 的困难**：成员们分享了在寻找 **Entropix** 解释时的挫败感，认为有些解释听起来像*荒谬的邪教言论*，凸显了掌握其概念的挑战。
   - 这引发了对更清晰、更具体信息的追求，以揭开其工作原理的神秘面纱。
- **关于 Entropix 的深刻博客文章**：最近的一篇 [blog post](https://timkellogg.me/blog/2024/10/10/entropix) 讨论了围绕 **Entropix** 的热度，指出其目标是利用现有模型通过极小的修改来实现**类似 o1 的推理**。
   - 文章详细阐述了它如何在不需要重新训练的情况下，将传统的 samplers 替换为基于 **entropy** 和 **varentropy** 的算法。
- **理解 LLM Samplers**：解释了 **sampler** 的概念，强调了它在根据计算出的概率预测序列中下一个单词的作用。
   - 讨论揭示了存在多种常见的 sampling 方法，指出了开发高效 **LLM** 模型所涉及的复杂性。



**提及的链接**：<a href="https://timkellogg.me/blog/2024/10/10/entropix">What is entropix doing? - Tim Kellogg</a>：未找到描述

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1294396297570160754)** (46 条消息🔥): 

> - `INTELLECT-1 去中心化训练`
> - `Diff Transformer 论文`
> - `LegoScale 框架`
> - `分布式训练中的通信方法`
> - `OpenDiLoCo 项目` 


- **INTELLECT-1：去中心化训练的飞跃**：Prime Intellect 发布了 **INTELLECT-1**，据称是史上首次对 **10B 模型** 进行的去中心化训练，旨在将去中心化训练的规模提升 **10 倍**。
   - 该项目邀请社区参与构建 **开源 AGI**，激发了成员们对分布式训练方法的浓厚兴趣。
- **探索 Diff Transformer**：**Diff Transformer** 论文提出了一种在消除噪声的同时增强对相关上下文注意力（Attention）的方法，从而提升语言建模任务的性能。
   - 成员们对其有效性和注意力噪声的本质表示好奇，讨论认为需要对其主张进行更深入的证明。
- **LegoScale：LLM 的模块化 3D 并行**：一篇关于 **LegoScale** 的论文因其众多创新而备受关注，这是一个可定制的、PyTorch 原生的、用于大语言模型 **3D 并行预训练** 的系统。
   - 它拥有可定制的激活检查点（Activation Checkpointing）和 **fp8 训练** 等特性，标志着高效模型训练迈出了重要一步。
- **分布式训练中的通信策略**：关于训练期间通信策略的讨论强调了分布式设置中效率与效果之间的平衡，特别是涉及 **diloco** 等方法。
   - 参与者探讨了架构和通信技术的潜在改进，建议使用非对称神经网络设计的可能性。
- **对 OpenDiLoCo 项目的兴趣**：成员们对 **OpenDiLoCo** 项目表现出极大的热情，讨论集中在其巨大的潜力和去中心化训练领域的最新进展。
   - 值得注意的包括社区与其团队的互动，以及将其作为未来创新基础元素的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/PrimeIntellect/status/1844814829154169038">来自 Prime Intellect (@PrimeIntellect) 的推文</a>：宣布 INTELLECT-1：史上首次对 10B 模型进行的去中心化训练。将去中心化训练规模在之前努力的基础上提升了 10 倍。任何人都可以加入我们，共同构建开源 AGI 🦋</li><li><a href="https://arxiv.org/abs/2410.05258">Differential Transformer</a>：Transformer 往往会将过多的注意力分配给无关的上下文。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时增强了对相关上下文的注意力。具体来说，t...</li><li><a href="https://openreview.net/forum?id=SFN6Wm7YBI">LegoScale：一站式 PyTorch 原生生产就绪解决方案...</a>：大语言模型（LLMs）的发展在推动最先进的自然语言处理应用方面发挥了重要作用。训练具有数十亿参数和数万亿...</li><li><a href="https://github.com/PrimeIntellect-ai/prime">GitHub - PrimeIntellect-ai/prime：prime（原名 ZeroBand）是一个用于在互联网上高效进行全球分布式 AI 模型训练的框架。</a>：prime（原名 ZeroBand）是一个用于在互联网上高效进行全球分布式 AI 模型训练的框架。 - PrimeIntellect-ai/prime
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1294417139146690650)** (34 条消息🔥): 

> - `Windows 上的 CUDA 编程`
> - `WSL 与 GPU 访问`
> - `双系统 vs WSL2`
> - `Triton Kernel 调试`
> - `使用 GPU 进行图像处理` 


- **Windows 上的 CUDA 编程选项**：成员们讨论了在 Windows 上设置 CUDA 编程的选项，强调了使用 [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 作为一个可行的路径，以及为了获得更好性能而选择双系统。
   - 有人指出，虽然“Windows 可以运行，但与 Linux 替代方案相比，它提供的选项较少”。
- **WSL GPU 访问已确认**：已确认 WSL 2 可以与 Windows 同时访问 GPU，这为 CUDA 开发提供了灵活性，无需第二个 GPU。
   - 成员们指出，为双系统设置制作第二个可启动 SSD 非常容易。
- **双系统困境**：成员们对双系统的挑战表示担忧，特别是 Linux 上的游戏限制，例如使用 Vanguard 反作弊系统的游戏。
   - 一位成员幽默地评论道：“Vanguard 的侵入性太强了”，强调了游戏玩家在双系统方面面临的挑战。
- **Triton Kernel 调试技巧**：一位用户分享了在 VSCode 中调试 Triton kernel 的经验，遇到了旧版本与 nightly 版本安装重叠的问题。
   - 他们发现，在不删除旧版本的情况下安装 Triton 的 nightly 版本会导致错误，特别提到了 `name jit not defined`。
- **图像处理见解**：一位新人寻求关于使用 GPU 进行图像处理的建议，表示有兴趣与社区就此话题进行交流。
   - 他们的咨询反映了人们对利用 GPU 能力处理图像处理任务日益增长的兴趣。



**相关链接**: <a href="https://docs.google.com/document/d/1Z345-b2WeLpXODRZ2TUQgardZJKbcAgQAseyQzVBzgE/edit?tab=t.0)">GPU MODE FAQ</a>: FAQ 什么是 GPU MODE？这是一个涉及高性能计算（主要关注 CUDA 和其他用于 ML/AI 性能的框架）的社区。通常每周，一位著名的工程师会...

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 条消息): 

rbatra: 第 5 版什么时候出？
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

gau.nernst: https://youtu.be/cGtiaJjLkAI
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1295084756303609958)** (2 条消息): 

> - `Pallas kernel 性能`
> - `JAX vs. PyTorch 优化`
> - `用于 AI/ML 的自定义算子`
> - `LLM 中的 Float16 精度`
> - `Mistral-7B 推理速度` 


- **Pallas kernel 难以超越 JAX**：用户表示很难找到 **Pallas kernel** 性能优于 **JAX** 的例子，通常注意到它们只能在理想情况下达到同等性能。
   - 许多例子强调代码实际上比参考实现慢，导致人们对 Pallas 的速度声明持怀疑态度。
- **自定义算子优化系列**：一篇文章讨论了构建 **自定义算子** 以优化 AI/ML 工作负载，强调了 **Pallas** 在其实现中的潜力。
   - 该文章是一个[由三部分组成的系列](https://chaimrand.medium.com/accelerating-ai-ml-model-training-with-custom-operators-163ef2a04b12)的一部分，旨在最大限度地发挥 TPU 的潜力。
- **使用 Float16 优化 LLM 推理**：一篇文章详细介绍了在 **JAX** 中使用 `float16` 精度优化 **本地 LLM 模型** 推理速度的努力，以匹配或超越 **PyTorch** 的性能。
   - 测试显示，虽然最初的 JAX 实现较慢，但优化后的版本在 **RTX 3090** 上的表现可以超过现有的 PyTorch 版本。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a">Pallas 的崛起：通过自定义 Kernel 释放 TPU 潜力</a>：使用自定义算子加速 AI/ML 模型训练 —— 第 3 部分</li><li><a href="https://robertdyro.com/articles/mistral_in_jax/">Robert Dyro</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1294392359316099083)** (11 messages🔥): 

> - `cuBLAS 限制`
> - `comfyui-flux-accelerator`
> - `Triton 性能对比`
> - `int8 操作加速`
> - `梯度累积问题` 


- **讨论 cuBLAS 路径限制**：一名成员指出了 `safe_int_mm` 在 **cuBLAS 路径**下的限制，并引用了 [Pull Request](https://github.com/pytorch/pytorch/pull/96685) 以供深入了解。
   - 他们观察到这些限制与 **FP8** 操作的限制有些相似。
- **引用 ComfyUI Flux Accelerator**：围绕 **comfyui-flux-accelerator** 展开了讨论，该工具因能加速 Flux.1 图像生成而受到关注，成员们对其实用性表现出浓厚兴趣。
   - 另一名成员确认这并非他们的作品，但仍认为这是一个值得关注的项目。
- **Triton 在 Linux 上的性能提升**：据报告，**Triton** 在 Linux 上的速度几乎是原生 Windows 的 **2 倍**，这表明了显著的性能升级。
   - 一名成员幽默地表示，如果他们有相应的技能，就会去实现这些改进。
- **讨论加速 Int8 和 FP8 操作**：对话转向在 Ampere 架构上对 **加速 int8 操作** 的需求，强调了量化的重要性。
   - 大家认识到，如果 Apple 处理器上也存在类似的加速技术，将极大地提升性能。
- **梯度累积与 Offload 兼容性**：一名成员询问了 **梯度累积（gradient accumulation）** 与设置了 `offload=True` 的 CPUOffloadOptimizer 的兼容性，并建议了一种在 CPU 上进行累积的变通方法。
   - 这引发了关于优化过程中梯度处理复杂性的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/discus0434/comfyui-flux-accelerator">GitHub - discus0434/comfyui-flux-accelerator: 只需使用此节点即可加速 Flux.1 图像生成。</a>：只需使用此节点即可加速 Flux.1 图像生成。 - discus0434/comfyui-flux-accelerator</li><li><a href="https://github.com/pytorch/pytorch/pull/96685">由 cpuhrsch 重新提交 _int_mm · Pull Request #96685 · pytorch/pytorch</a>：避免对 gemm_and_bias 进行任何更改。抄送 @soumith @voznesenskym @penguinwu @anijain2305 @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @Xia-Weiwen @wenzhe-nrv @jiayisunx @peterbe...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

navmarri.: 你介意分享一下制作这些精美动画所使用的工具吗？
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1294410868704018473)** (6 messages): 

> - `Theta 波双耳节拍`
> - `睡眠质量`
> - `安慰剂效应`
> - `梦幻般的验证提示词` 


- **尝试 Theta 波双耳节拍**：一位成员分享了他们在睡前使用 **Theta 波双耳节拍** 的经验，强调在优先考虑休息并获得高睡眠评分的日子里，感觉有所改善。
   - 他们提到，虽然**熬夜**可能对年轻人有吸引力，但对于有宝宝的人来说，获得良好的睡眠评分可能具有挑战性。
- **对功效的怀疑**：另一位成员对双耳节拍的有效性表示怀疑，认为这可能是“蛇油”（虚假宣传）。
   - 对此，一名成员指出，即使宣传有夸大成分，**安慰剂效应**仍可能带来积极体验。
- **双耳节拍的影响尚不明确**：一位参与者分享了轶事证据，认为双耳节拍确实有*某种作用*，但具体效果仍不清楚，且可能与现有研究不符。
   - 这突显了双耳节拍对睡眠和放松影响的透明度可能不足。
- **对验证提示词的担忧**：一位用户对他们在验证提示词中选择 **'dreamlike'**（梦幻般）一词表示怀疑，暗示可能存在问题。
   - 这引发了关于特定术语在实现预期用户响应方面有效性的疑问。


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/)** (1 messages): 

madna11: 很棒的方法
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1294421828768366694)** (13 messages🔥): 

> - `AMD Linux Radeon drivers`
> - `ROCm on alternative distros`
> - `hipBLASLt performance` 


- **AMD 更新 Linux Radeon drivers**: 终于，AMD 对 **Linux Radeon drivers** 进行了重大更新，以支持最新的发行版，包括 **Ubuntu 24.04**，尽管目前还不支持 HWE kernel。一位成员表达了沮丧，称：*“天哪，为什么 Linux 这么烂。”*
   - 您可以在 [release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-UNIFIED-LINUX-24-20-3.html) 中找到有关此次更新的更多详情。
- **为 ROCm 选择更好的发行版**: 成员们讨论了在 **Arch Linux** 上使用 ROCm 的优势，因为它提供了更好的 packaging，并且更多的显卡开箱即用地编译了 ROCm。一位成员分享了他们在 **RX6700XT** 上仅通过安装命令运行 ROCm 的经验。
   - 另一位成员指出，由于 `extra` 仓库中的依赖关系，Arch **6.0.2** 版本可能会出现损坏，而他们个人已针对 **6.2.1** 迁移到了 **distrobox Ubuntu container**。
- **了解 MI300X 上的 hipBLASLt 性能**: 有人提出了关于 **rocBLAS** 和 **hipBLASLt** 之间区别的问题，后者被指出在 **MI300X** 上明显更快。一位成员解释说：*“hipBLASLt 是生态系统中的新成员，专门针对 MI300X 上的 Tensor Core 使用进行了优化。”*
   - 这表明寻求性能提升的用户将受益于在兼容硬件上使用 hipBLASLt。



**提到的链接**: <a href="https://x.com/opinali/status/1844862171416555617?s=46">Osvaldo Doederlein (@opinali) 的推文</a>: 哦太棒了，AMD 终于对 Linux Radeon drivers 进行了重大更新，以支持最新的发行版，包括我的 Ubuntu 24.04... 但是，还不支持 HWE kernel。天哪，为什么 Linux 这么烂。总之 g...

  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1294395942652612739)** (8 messages🔥): 

> - `SIMD programming on ARM`
> - `Neon intrinsics`
> - `ARM Vectorization`
> - `Cross-Architecture Libraries`
> - `ARM Experimentation` 


- **探索 ARM 上的 SIMD 编程选项**: 一位成员询问了在 **RK3588** 等 ARM 平台上进行 **SIMD programming** 的普遍共识，以及 **OpenCL** 与 SIMD intrinsics 相比的有效性。
   - 另一位成员建议，大多数开发者使用来自 arm_neon.h 的 **Neon compiler intrinsics**，或者使用针对各种架构的带有 polyfills 的库。
- **ATen Vectorized PR 改进了寄存器使用**: 一位成员宣布了一个 **清理 ARM 上 ATen Vectorized 的 PR**，该 PR 使每个 instance 使用一个 vector register 而不是两个。
   - 此更改旨在优化 ARM **vectorized operations** 的性能和效率。
- **库 vs Intrinsics 的可移植性**: 一位参与者强调，虽然 intrinsics 很有效，但使用 **vectorized** 或 **highway** 等库可以更轻松地实现 **cross-architecture portability**。
   - 这些库有助于在不绑定到特定低级实现的情况下保持效率。
- **没有良好 AVX 对应物的 Intrinsics**: 有人提到某些 ARM 指令（如 **FMLAL**，一种累加到 float32 的 float16 FMA）缺乏直接的 **AVX analogues**。
   - 由于这些特定的 instruction sets，**优化 ARM code** 需要独特的考量。
- **M1 MacBook 用于 ARM 实验**: 一位成员观察到，拥有 **M1 MacBook** 就不需要为了基础的 ARM 实验而购买 Raspberry Pi。
   - 这一观点很快得到了另一位成员的确认，强调了 M1 在学习方面的效率。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1294653540135993378)** (4 messages): 

> - `Loss Calculation in Training`
> - `Nanogpt Performance with Liger Kernel` 


- **关于 Loss 和梯度的澄清**: *t_cc* 寻求澄清，即在 `label==ignore_index` 的地方 **loss** 和 **gradients** 是否应该为 **0**，以及在计算 batch mean loss 时是否包含 `ignore_index`。
   - *qqs222* 确认这种方法类似于 **cross-entropy loss**。
- **估算使用 Liger Kernel 的性能提升**: *faresobeid* 询问了在 **8xH100** 上，使用 **3.5B tokens** 和 **64 batch size** 训练的普通 **NanoGPT Pytorch model (124M)**，在使用 **Liger Kernel** 后潜在的速度提升。
   - 然而，对话中并未给出关于性能改进的具体估算。

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1294452347027849279)** (14 messages🔥): 

> - `Apple Silicon GPU 与 Docker`
> - `MPS Op 贡献`
> - `适用于 Apple Silicon 的 Kubernetes Device Plugin` 


- **Apple Silicon GPUs 与 Docker**: 围绕让 Docker 利用 **Apple Silicon GPUs** 的挑战展开了讨论，报告显示该问题目前仍未解决。
   - 一位成员提到，这已经作为一个内部工单存在一段时间了，但目前没有人积极跟进，这让人感到有些无奈。
- **即将发布的 PyTorch i0 算子 PR**: 一位成员宣布在会议期间为 [torch.special.i0 创建了 PR](https://github.com/pytorch/pytorch/pull/137849)，这表明 PyTorch MPS 支持的开发工作正在持续进行。
   - 他们还在开发 `bilinear2d_aa` 算子，并建议接下来的易于贡献的目标可能是 **i1** 和 **i0e**。
- **Kubernetes 需要适用于 Apple GPUs 的 Device Plugin**: 另一位成员提出了在 **Kubernetes** 中使用 Apple Silicon GPUs 的类似担忧，并主张开发一个类似于 NVIDIA [版本](https://github.com/NVIDIA/k8s-device-plugin) 的 Device Plugin。
   - 这表明在容器化环境中，迫切需要针对 Apple 硬件量身定制的生态系统工具。
- **MPS 算子覆盖范围追踪**: 一位成员链接到了一个 [MPS op 覆盖范围追踪 issue](https://github.com/pytorch/pytorch/issues/77764)，该 issue 集中讨论了为 MPS 后端添加新算子的工作。
   - 他们提供了另一个参考链接，详细说明了 MPS ops 的支持状态和贡献机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chariotsolutions.com/blog/post/apple-silicon-gpus-docker-and-ollama-pick-two/">Apple Silicon GPUs, Docker and Ollama: Pick two.</a>: 如果你最近尝试在 Apple GPU 上通过 Docker 使用 Ollama，你可能会发现其 GPU 并不受支持。但你可以在 Mac 上以支持 GPU 的方式运行 Ollama。本文将解释...</li><li><a href="https://github.com/pytorc">pytorc - 概览</a>: pytorc 有 2 个可用的代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/NVIDIA/k8s-device-plugin">GitHub - NVIDIA/k8s-device-plugin: 适用于 Kubernetes 的 NVIDIA device plugin</a>: 适用于 Kubernetes 的 NVIDIA device plugin。通过在 GitHub 上创建账号来为 NVIDIA/k8s-device-plugin 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/pull/137849">[MPS] 由 malfet 添加 i0 op · Pull Request #137849 · pytorch/pytorch</a>: 大致逐字复制自 pytorch/aten/src/ATen/native/Math.h 第 101 行，位于 47c8aa8 JITERATOR_HOST_DE...</li><li><a href="https://github.com/pytorch/pytorch/issues/77764">通用 MPS op 覆盖范围追踪 issue · Issue #77764 · pytorch/pytorch</a>: 此 issue 旨在提供一个集中的地方，用于列出和追踪为 MPS 后端添加新算子支持的工作。PyTorch MPS Ops 项目：用于追踪 MPS 后端所有算子的项目。目前有...</li><li><a href="https://qqaatw.dev/pytorch-mps-ops-coverage/">MPS 支持矩阵</a>: 未找到描述内容
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1294744544243482685)** (1 条消息): 

> - `SGLang`
> - `FlashInfer`
> - `MLC LLM`
> - `LLM Deployment`
> - `Community Meetup` 


- **LLM 爱好者的在线研讨会**：加入由 **SGLang**、**FlashInfer** 和 **MLC LLM** 共同举办的在线研讨会，探索高效的 LLM 部署与推理服务，时间为 **10 月 5 日** **下午 4:00 PST**。点击[此处](https://docs.google.com/forms/d/e/1FAIpQLScJ0nsUm_wNud7IBQ3xFcwI46k7JbikiMjiCwez57Lu7AkuOA/viewform)免费注册！
   - 议程包括 SGLang 中的 **低 CPU 开销调度** 以及高性能 LLM 服务中的 **Kernel 生成** 等主题的见解。
- **SGLang 关于高效调度的见解**：**Liangsheng Yin**、**Lianmin Zheng** 和 **Ke Bao** 将在 **下午 4:00 - 4:45 PST** 的环节中分享 SGLang 的最新进展，重点关注 **DeepSeek MLA 优化** 和 **快速 JSON 解码**。
   - 随后将进行专门的 **Q&A** 环节，解答社区疑问。
- **FlashInfer 的高性能 Kernel 生成**：在 **下午 4:50 - 5:35 PST**，听取 **Zihao Ye** 关于 **FlashInfer** 项目的分享，重点介绍高性能 LLM 服务中的 **Kernel 生成** 技术。
   - 与同行开发者一起讨论该领域的进展。
- **MLC LLM 的通用部署策略**：**Ruihang Lai**、**Yixin Dong** 和 **Tianqi Chen** 将在 **下午 5:40 - 6:25 PST** 概述 **MLC LLM**，讨论 **低延迟服务** 和通用 LLM 部署等主题。
   - 他们的环节也将包含 Q&A 部分以供参与者互动。



**提到的链接**：<a href="https://docs.google.com/forms/d/e/1FAIpQLScJ0nsUm_wNud7IBQ3xFcwI46k7JbikiMjiCwez57Lu7AkuOA/viewform">LMSYS 关于高效 LLM 部署与服务的在线研讨会 (10 月 16 日)</a>：我们很高兴邀请您参加由 SGLang、FlashInfer 和 MLC LLM 共同举办的在线研讨会！这三个紧密协作的项目将分享他们对高效 LLM 部署的不同见解...

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1294375857531523123)** (121 条消息🔥🔥): 

> - `CohereForAI contributions`
> - `Nobel Prize in Physics 2024`
> - `AI technology advancements`
> - `Hurricane impact on productivity`
> - `Community engagement in AI` 


- **探索对 CohereForAI 的贡献**：成员们讨论了对 [CohereForAI](https://cohere.com/research) 社区做出有意义贡献的重要性，强调公民科学是那些对 AI 感兴趣的人的切入点。
   - 一位成员表达了积极贡献和指导的愿望，同时寻求与他们对技术共生关系愿景相一致的项目。
- **庆祝 AI 创新者获得诺贝尔奖**：社区成员提到，Sir John J. Hopfield 和 Sir Geoffrey E. Hinton 因其在 AI 和神经网络领域的开创性工作被授予 2024 年诺贝尔物理学奖。
   - 他们的研究集中在使机器学习成为可能的奠基性发现上，展示了 AI 在科学探索中的深厚根基。
- **老旧技术与快速进步**：社区成员反思了技术进步的飞速步伐，指出了个人成长过程中计算机使用经验所带来的历史背景的重要性。
   - 讨论承认，虽然技术取得了巨大进步，但其未来仍充满不确定性和细微差别，潜在的优势与冲突都不可避免。
- **飓风对工作效率的影响**：一位成员分享了正在发生的飓风影响其工作能力的情况，突显了更广泛的社区背景。
   - 这引发了关于外部事件如何显著影响 AI 相关工作效率和工作流的讨论。
- **对 Cohere 界面深色模式的需求**：成员们表达了对 Cohere 网站增加深色模式功能的期望，指出其对提升用户体验的实用性。
   - 对话强调了对用户友好型增强功能的需求，特别是考虑到社区成员频繁与该平台互动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.nobelprize.org/prizes/physics/2024/press-release/">2024 年诺贝尔物理学奖</a>：2024 年诺贝尔物理学奖共同授予 John J. Hopfield 和 Geoffrey E. Hinton，以表彰他们“为实现人工神经网络机器学习所做的奠基性发现和发明”...</li><li><a href="https://cohere.com/research">Cohere For AI (C4AI)</a>：Cohere For AI (C4AI) 是 Cohere 的非营利研究实验室，致力于解决复杂的机器学习问题。 
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1294632508427731004)** (20 messages🔥): 

> - `Cohere AI Learning Contributions` (Cohere AI 学习贡献)
> - `Cohere Rerank Model Date Handling` (Cohere Rerank 模型日期处理)
> - `Cohere Rerank Pricing` (Cohere Rerank 定价)
> - `Command-R-08-2024 Model Obscurities` (Command-R-08-2024 模型模糊点)
> - `Prompt Engineering Best Practices` (Prompt Engineering 最佳实践)


- **Cohere AI 贡献不明确**：一位用户对关于“意识”等话题的讨论是否对模型学习有贡献表示担忧，称其时间有限，希望进行有意义的参与。
   - 他们正在寻求关于如何有效地为机器学习做出贡献，并确保其见解得到重视的指导。
- **Cohere Rerank 模型的日期感知**：关于实施 Cohere Rerank，一位用户询问该模型是否天生能识别“今天”，还是需要提供今天的日期以便与文档日期进行比较。
   - 明确这一点可能会显著影响搜索结果如何根据时效性进行优先级排序。
- **对 Rerank 定价的困惑**：一位用户询问 Web-search 定价是否包含在 Rerank 定价中，因为他们在 Cohere 网站上找不到相关信息。
   - 了解定价结构有助于有效地规划 RAG 的实施。
- **Command-R-08-2024 模型的 Prompt 问题**：用户报告称，在向 Command-R-08-2024 模型提供第二篇财经文章时，它倾向于分析第一篇文章，而不是等待新数据。
   - 这表明模型在处理连续的用户 Prompt 时可能存在局限性，影响了其对未经培训用户的可用性。
- **有效的 Prompt Engineering 技巧**：讨论强调了编写简洁 Prompt 的重要性，指出 Command-R-08-2024 在处理短描述而非长指令时表现更好。
   - 鼓励用户根据文档指示，同时利用 Preamble 和 Message 参数来获得更清晰的任务指令。



**相关链接**：<a href="https://docs.cohere.com/v1/docs/crafting-effective-prompts#task-splitting)">Crafting Effective Prompts (v1 API) — Cohere</a>：该页面介绍了为 Prompt Engineering 编写有效 Prompt 的不同方法。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1294383632613507165)** (7 messages): 

> - `API Tokens Usage` (API Token 使用)
> - `V2 API User Role Restrictions` (V2 API 用户角色限制)
> - `Gen AI Hackathon Announcement` (Gen AI 黑客松公告)
> - `ChatBot Development Challenges` (ChatBot 开发挑战)
> - `Cohere Code Sharing` (Cohere 代码共享)


- **获得体面回复不一定需要 API Token**：一位成员询问在 API 请求中是否必须使用 `<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>` 等 Token，并对它们的重要性表示困惑。
   - *如果不包含这些 Token，回复是否仍然体面？* 这一问题尚未得到解答，但反映了用户的普遍疑虑。
- **V2 API 不允许在 Tool Call 之后出现用户角色**：一位用户询问了最近的 V2 API 更新，该更新不允许在序列中的 Tool Call 之后出现 User Role。
   - 这一限制似乎指向更精简的交互流程，但其背后的原因尚不明确。
- **Gen AI 黑客松邀请**：一份公告详细介绍了由 **CreatorsCorner** 组织的 **Gen AI Hackathon**，鼓励团队协作创建 AI 解决方案。
   - 挑战的重点是构建安全且能通过智能协作增强人类潜力的 Multi-Agent 系统。
- **ChatBot 开发方法**：一位成员表达了开发 ChatBot 的兴趣，但指出它不会回答每一个问题，表明其响应具有针对性。
   - 这表明其关注点在于量身定制的交互，而非对用户查询的通用响应。
- **Cohere 代码共享协助**：一位成员请求协助私下分享 **Cohere 代码**部分，寻求社区其他成员的帮助。
   - 另一位用户的反馈建议采用 Role 系统可能已经解决了最初的问题，强调了社区成员之间的支持。



**相关链接**：<a href="https://lu.ma/ke0rwi8n">Vertical Specific AI Agents Hackathon · Luma</a>：Gen AI Agents CreatorsCorner 与 Sambanova Systems、Prem、Marly、Senso 等合作，热烈欢迎个人和团队加入我们的……

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

competent: 请不要在这里发布那个内容。
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1294411414320054272)** (73 条消息🔥🔥): 

> - `Mojo Backend Performance` (Mojo 后端性能)
> - `Magic CLI Installation` (Magic CLI 安装)
> - `Training Courses for Mojo` (Mojo 培训课程)
> - `Issues with Mojo Playground` (Mojo Playground 相关问题)
> - `Community Support for Mojo` (Mojo 社区支持)


- **对 Mojo 安装的挫败感**：一位用户表达了对 *Mojo* 安装问题的沮丧，声称 playground 已损坏，且演示代码会导致错误。
   - 他们强调缺乏最新的教程，并发现 *Magic*、*Mojo* 和 *MAX* 的安装过程脱节且令人困惑。
- **Mojo 性能的未来**：一位用户指出，虽然该语言的语法很有前景，但 *Mojo* 项目目前在性能方面仍面临挑战，尤其是在缺乏 GPU 支持的情况下。
   - 他们希望看到可运行的示例和稳定的开发环境，以帮助学习和迁移现有代码。
- **Magic CLI 的困惑**：关于是否在现有项目中使用 *Magic* 展开了讨论，一位用户表示拒绝转换现有代码，担心这可能会破坏功能。
   - 建议要么使用 *Magic* 设置 `mojoproject.toml` 文件，要么创建一个 *Conda* 环境以便更好地管理。
- **社区会议公告**：一位用户宣布了即将于 10 月 21 日举行的社区会议，邀请参与者将他们的议题添加到议程文档中。
   - 此外还邀请分享与 *Modular* 相关的 Discord 服务器链接，以增加社区参与度。
- **Mojo 的支持与资源**：社区鼓励分享具体的错误信息和项目细节，以改进对 *Mojo* 用户的支持。
   - 用户被引导至文档和资源，以更好地理解如何有效地实施 *Magic* 和 *Mojo*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/magic/conda">将 MAX/Mojo 添加到 conda 项目 | Modular 文档</a>：虽然我们建议使用 Magic 来管理您的虚拟环境</li><li><a href="https://docs.modular.com/max/python/get-started">使用 Python 运行推理 | Modular 文档</a>：MAX Engine Python API 的演练，展示如何加载和运行模型。</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit?tab=t.0)">[公开] Modular 社区会议</a>：Modular 社区会议。此文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...</li><li><a href="https://www.youtube.com/watch?v=4xp1JpNMnLY)).">介绍 Magic 🪄</a>：Magic 🪄 是一个包管理器和虚拟环境工具，它将几个原本独立的工具统一为一个，取代了我们的 Modular CLI。在本视频中 B...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1294375049188474931)** (66 messages🔥🔥): 

> - `Mojo 中的 AES 硬件支持`
> - `Mojo Lists 中的隐式转换`
> - `闭源编译器`
> - `Mojo 库设计`
> - `Mojo 构建问题` 


- **AES 硬件支持实现**：一位成员分享了他们通过 LLVM intrinsics 在 Mojo 中添加硬件 AES 支持的工作，展示了如何使用字符串模板动态调用 intrinsics。
   - 他们强调这使得即使在库代码中也能与 LLVM 进行更紧密的集成，突显了 Mojo 在此类实现中的灵活性。
- **Mojo Lists 中的隐式转换**：一位成员指出 Mojo Lists 允许将 Int 添加到 List 中，质疑这种行为是否因隐式转换的构造函数而有意为之。
   - 据解释，这种行为是偶然的，目前有一个正在处理的 Issue 追踪语言中这种潜在的不一致性。
- **Mojo 编译器目前是闭源的**：一位成员对 Mojo 的编译器开发表示兴趣，但发现编译器是闭源的，以避免“委员会式设计”（design by committee）。
   - 鼓励他们研究开源的标准库以寻求潜在的贡献机会，并强调这也能提供重要的类编译器设计体验。
- **Mojo 的构建问题**：一位用户在尝试构建 Mojo 文件时遇到链接错误，提示过程中缺少库。
   - 社区提供了帮助，建议他们检查 magic 环境是否已激活，以及是否通过命令行工具正确安装了 Mojo。
- **成功将 Rust 库绑定到 Mojo**：一位成员通过 FFI 绑定成功创建了 Rust UUID 库的 Mojo 版本，并将其发布在 prefix.dev 平台上。
   - 他们鼓励其他人参考其项目的构建脚本来绑定 Rust 库，并分享了其工作的 GitHub 链接以供进一步参考。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>：未找到描述</li><li><a href="https://github.com/better-mojo/uuid">GitHub - better-mojo/uuid: binding a mojo version of rust uuid.</a>：绑定 Rust uuid 的 Mojo 版本。通过创建账户为 better-mojo/uuid 开发做贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1310">[BUG]: Mojo permits the use of any constructor for implicit conversions · Issue #1310 · modularml/mojo</a>：Bug 描述如题。也许是按预期工作的，但我认为这是一个 Bug，因为 Mojo 不是 JavaScript。这种行为是为了绕过 print 的 trait 缺失吗？复现步骤...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1295034700552732673)** (5 messages): 

> - `MAX 在 Linux 上的安装`
> - `MAX 的编译速度`
> - `分析编译耗时`
> - `MAX 的功能请求` 


- **MAX 支持多个操作系统**：根据 [Modular Docs](https://docs.modular.com/max/get-started)，MAX 目前支持 **Linux**、**MacOS** 和 Windows 上的 **WSL2**。这使得刚开始使用的用户可以在不同平台上访问它。
   - 鼓励用户使用 `curl -ssL https://magic.modular.com | bash` 安装 Magic 进行环境管理。
- **编译时间需要关注**：据报告，在首次尝试后，图（graphs）的编译时间约为 **300 ms**，初始编译时间约为 **500 ms**，即使是简单的张量乘法操作也是如此。
   - 此外，有人提到缓存命中应该比目前观察到的时间更快。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>：在此页面中，我们将向您展示如何运行一些示例项目。</li><li><a href="https://github.com/modularml/max/issues">Issues · modularml/max</a>：一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - Issues · modularml/max
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1294381155185066075)** (69 messages🔥🔥): 

> - `OpenAI Swarm`
> - `Entropix`
> - `RAG 技术`
> - `LLM 推理`
> - `Jensen 访谈见解`

- **OpenAI Swarm 助力多智能体系统**：OpenAI 推出了 **Swarm**，这是一个用于构建多智能体系统的轻量级库，为管理 Agent 之间的交互提供了一种**无状态抽象 (stateless abstraction)**。
   - 它提供了对 Agent 角色和移交 (handoffs) 的见解，且在不使用 Assistants API 的情况下运行，开发者强调这仍是一个**实验性框架 (experimental framework)**。
- **Entropix 讨论热度上升**：成员们参与了关于 **Entropix** 的讨论，赞赏其对其运作方式和潜力的深入见解。
   - 随着名称和概念获得关注，用户对即将推出的评估功能表示期待。
- **探索检索增强生成 (RAG)**：讨论集中在 **RAG Techniques**，重点介绍了一个展示将检索与生成模型集成的各种高级方法的 GitHub 仓库。
   - 参与者热衷于优化性能，并理解像 *Haystack* 这样的框架与从零开始实现解决方案相比的优势。
- **Jensen 关于 NVIDIA 的见解引起共鸣**：在一次采访中，Jensen 强调了 NVIDIA 的**全栈方法 (full stack approach)** 以及 AI 基础设施的复杂需求，提到了在各种工作负载中进行加速计算的必要性。
   - 他承认了生成式 AI 的变革性影响，同时概述了 NVIDIA 的竞争地位和持续创新的必要性。
- **实施框架的挑战**：一些用户对 *Haystack* 等框架表示怀疑，更倾向于自定义解决方案，但也承认其对经验较少的开发者的好处。
   - 他们一致认为每个框架都有其优缺点，且围绕 *Langchain* 存在显著的批评情绪。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://diamond-wm.github.io">Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎</a>：Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎 网页</li><li><a href="https://x.com/darioamodei/status/1844830404064288934?s=46">Dario Amodei (@DarioAmodei) 的推文</a>：Machines of Loving Grace：我关于 AI 如何让世界变得更好的文章 https://darioamodei.com/machines-of-loving-grace</li><li><a href="https://events.zoom.us/ev/AqLi3dmNZSAXddMiqJkHlHTWkEjpoQZ7CEHtgg-bgBXf5FUjyxMS~AiKVqU-B1HTbeHxFjNySoPHgPEmmKjIcWvxbqWn3NPWWteuSxmiPlxev_A">OpenAI Presents</a>：未找到描述</li><li><a href="https://x.com/ilanbigio/status/1844444850160271717?s=46">Ilan Bigio (@ilanbigio) 的推文</a>：分享我们在这个功能中使用的所有 meta-prompts 和 meta-schemas... 刚刚在 @openai 文档中发布了它们，去看看吧，把它们变成你自己的——很想听听你们的问题和反馈！...</li><li><a href="https://www.facebook.com/trckgmng/videos/1254494898910739/?mibextid=rS40aB7S9Ucbxw6v">150万次观看 · 2万次反应 | GTA 5 现实生活图形 | Generative AI | GTA 5 现实生活图形 | Generative AI | 由 TRCK 发布 | Facebook</a>：GTA 5 现实生活图形 | Generative AI</li><li><a href="https://x.com/latentspacepod/status/1844870676202783126">Latent.Space (@latentspacepod) 的推文</a>：🆕 生产级 AI Engineering 从 Evals 开始 https://latent.space/p/braintrust 在 @braintrustdata A 轮融资后，与 @ankrgyl 一起深入探讨 LLM Ops 行业的现状，时长 2 小时！我们讨论了...</li><li><a href="https://x.com/yuhu_ai_/status/1844847171411304771?s=46">Yuhuai (Tony) Wu (@Yuhu_ai_) 的推文</a>：AI 推理（Reasoning）的三个组成部分：1. Foundation (Pre-training) 2. Self-improvement (RL) 3. Test-time compute (planning)。@xai 很快将拥有世界上最好的 foundation - Grok3。加入我们...</li><li><a href="https://x.com/attunewise/status/1845677455681499278">attunewise.ai (@attunewise) 的推文</a>：通过“竞争者”指的是下一个 token。根据我目前的观察，高 varentropy 预示着幻觉（正如你所预料的那样）@_xjdr</li><li><a href="https://news.crunchbase.com/venture/ai-legal-tech-evenup-unicorn-bain/">AI 法律科技初创公司 EvenUp 融资 1.35 亿美元，晋升独角兽地位</a>：EvenUp 是一家为人身伤害领域创建人工智能产品的法律科技初创公司，在由 Bain Capital 领投的 D 轮融资中以超过 10 亿美元的估值筹集了 1.35 亿美元...</li><li><a href="https://arxiv.org/abs/2410.05983">Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG</a>：当长上下文 LLMs 遇到 RAG：克服 RAG 中长输入的挑战。检索增强生成 (RAG) 使大语言模型 (LLMs) 能够利用外部知识源。LLMs 处理更长输入序列的能力不断增强，为...开辟了道路。</li><li><a href="https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一体化工作区</li><li><a href="https://inference.cerebras.ai">Cerebras Inference</a>：未找到描述</li><li><a href="https://x.com/shyamalanadkat/status/1844888546014052800?s=46">shyamal (@shyamalanadkat) 的推文</a>：介绍 swarm：一个用于构建、编排和部署 multi-agent 系统的实验性框架。🐝 https://github.com/openai/swarm</li><li><a href="https://x.com/_clarktang/status/1845495471038677360?s=46">Clark Tang (@_clarktang) 的推文</a>：关于 Altimeter 最酷的事情之一是 @altcap 鼓励我们在研究过程中保持开放。在这次与 Jensen 的采访中，我们问了他我们（作为分析师）想知道的问题 - 并且...</li><li><a href="https://x.com/swyx/status/1845340932155244726">swyx @ NYC (@swyx) 的推文</a>：[直播] OpenAI Swarm + Realtime API 初探 https://x.com/i/broadcasts/1RDGlyoybRRJL</li><li><a href="https://x.com/tom_doerr/status/1844739285326422485">Tom Dörr (@tom_doerr) 的推文</a>：OpenAI 正在考虑增加对 DSPy 的支持</li><li><a href="https://x.com/_philschmid/status/1845075902578999325?s=46">Philipp Schmid (@_philschmid) 的推文</a>：这太出乎意料了！@OpenAI 发布了 Swarm，一个用于构建 multi-agent 系统的轻量级库。Swarm 提供了一个无状态的抽象来管理多个 Agent 之间的交互和交接...</li><li><a href="https://x.com/apples_jimmy/status/1844416663925719146?s=46">Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：时候到了</li><li><a href="https://x.com/shyamalanadkat/status/1844934179013919085?s=46&t=PW8PiFwluc0tdmv2tOMdEg">shyamal (@shyamalanadkat) 的推文</a>：‼️ 既然这出乎意料地开始流行：swarm 不是 OpenAI 的官方产品。把它看作是一个 cookbook。它是用于构建简单 Agent 的实验性代码。它不是...</li>

<li><a href="https://x.com/therealadamg/status/1845154888637993434?s=46">来自 Adam.GPT (@TheRealAdamG) 的推文</a>：我最喜欢 Swarm 的一点是它的简洁。它不过度堆砌，而是恰到好处。这是我们帮助你入门的示例。查看 cookbook 以开始使用：...</li><li><a href="https://arxiv.org/abs/2410.05229">GSM-Symbolic: 理解 Large Language Models 中数学推理的局限性</a>：Large Language Models (LLMs) 最近的进展激发了人们对其形式推理能力的兴趣，特别是在数学方面。GSM8K 基准测试被广泛用于评估数学...</li><li><a href="https://x.com/svpino/status/1844765535457902839">来自 Santiago (@svpino) 的推文</a>：Large Language Models 不具备推理能力。谢谢你，Apple。</li><li><a href="https://github.com/NirDiamant/RAG_Techniques">GitHub - NirDiamant/RAG_Techniques: 该仓库展示了用于 Retrieval-Augmented Generation (RAG) 系统的各种高级技术。RAG 系统将信息检索与生成模型相结合，以提供准确且上下文丰富的响应。</a>：该仓库展示了用于 Retrieval-Augmented Generation (RAG) 系统的各种高级技术。RAG 系统将信息检索与生成模型相结合，以提供准确且上下文丰富的响应...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1294429723715174451)** (1 条消息): 

> - `Production AI Engineering`
> - `LLM Ops industry`
> - `Evals centrality`
> - `Impossible Triangle of LLM Infra`
> - `Open source stats` 


- **Production AI Engineering 发布新见解**：最新的 [播客剧集](https://x.com/latentspacepod/status/1844870676202783126) 深入探讨了 **Production AI Engineering**，包含了 @braintrustdata A 轮融资后来自 @ankrgyl 的见解。
   - 讨论包括为什么 **Evals** 在生产级 AI 中至关重要，这标志着行业格局中的一个重要亮点。
- **Evals 是 LLM Ops 的核心**：专家们一致认为 **Evals 是核心**，处于生产级 AI 工程版图的中心，强调了它们在 LLM 运营中的重要性。
   - @HamelHusain 的贡献强化了这一观点，他指出人们对评估指标的关注度日益提高。
- **探索 LLM Infra 的不可能三角**：Swyx 分享了他关于 LLM 基础设施 **不可能三角 (Impossible Triangle)** 的个人理论，并对其影响提出了新的视角。
   - 这促成了 @doppenhe 的 **AI SDLC 图表** 的更新版本，为社区内正在进行的讨论提供了清晰的解释。
- **令人震惊的开源统计数据揭晓**：播客透露了令人震惊的统计数据，显示目前只有不到 **5%** 的 LLM 模型是开源的。
   - 预计在接下来的讨论中会有更多细节，因为这一数字突显了 AI 领域向专有解决方案发展的趋势。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1844870676202783126">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 Production AI Engineering 始于 Evals https://latent.space/p/braintrust 在 @braintrustdata A 轮融资后，与 @ankrgyl 一起对 LLM Ops 行业现状进行了 2 小时的深度探讨！我们讨论了...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1294389030775029764)** (65 messages🔥🔥): 

> - `实时编程演示`
> - `编程语言对比`
> - `演示技巧`
> - `编程中的 DSLs`
> - `即将举行的会议` 


- **实时编程演示激发热情**：成员们对实时编程演示表现出浓厚兴趣，其中一人提议下周演示如何使用 Lua, Golang 或 Ruby **构建 DSLs**。
   - *实时演示具有挑战性*，大家一致认为需要提前进行准备和规划。
- **关于编程语言的讨论**：讨论了各种编程语言在编码任务中的有效性，其中 Python 被认为在 **DSLs** 方面表现很差。
   - 成员们指出 Lua, Golang 和 Ruby 可能是更合适的选择，并强调了各自独特的优势和劣势。
- **演示准备思路**：几位成员表示有兴趣创建和分享演示文稿，并特别提到了使用 **O1** 来构建演示文稿。
   - 一位成员提议分享进行实时编程演示的技巧，强调了在此类活动中有效沟通的重要性。
- **协作工作坊模式**：成员们赞同在聊天频道中共同协作，在接下来的一周内创建和完善演示文稿。
   - 这种“没有配对的配对编程”概念旨在促进同行间的反馈与协作，以获得更好的演示效果。
- **即将举行的会议公告**：分享了即将举行的会议链接，主题包括 **GenAI 的 UI/UX 模式**以及用于有效信息检索的 **RAG 架构**。
   - 提供了文章和视频等资源，为参与者提供进一步的背景信息和材料。



**提及的链接**：<a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>：未找到描述

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1294376625408049264)** (119 messages🔥🔥): 

> - `Stable Diffusion 与 3060ti 的兼容性`
> - `Embedding 与 Lora 训练对比`
> - `放大技术`
> - `图像转 3D 模型生成`
> - `产品照片集成` 


- **3060ti 适用于 Stable Diffusion**：用户讨论了 **3060ti** 在 Stable Diffusion 中的性能，一些人认为尽管只有 **8GB VRAM**，它的表现依然不错。
   - 一位用户分享说，使用 **Flux** 图像生成证明了该显卡在限制条件下的能力。
- **Embedding 与 Lora 训练对比**：讨论了训练 **embedding** 和 **Lora** 之间的区别，用户表示 Lora 训练通常能产生更高质量的图像。
   - 有人指出，与仅影响文本编码器的 embedding 相比，**Lora 训练**允许对扩散模型进行更详细的训练。
- **放大技术对比**：对话评估了 **Tiled Diffusion** 与 **Ultimate SD Upscale** 在图像增强方面的优劣，强调它们用途不同。
   - **Tiled Diffusion** 侧重于减少 VRAM 消耗，而 **Ultimate SD Upscale** 则专注于提高图像分辨率。
- **图像转 3D 模型生成的挑战**：参与者交流了对**图像转 3D 模型**生成复杂性的看法，表示目前仍缺乏有效的解决方案。
   - 用户强调，多视图推理（multi-view inference）等技术是目前克服该领域某些挑战的最佳可用方法。
- **寻求产品照片集成方面的帮助**：一位用户请求协助将产品照片集成到新背景中，希望获得高质量的结果而非基础的合成。
   - 建议包括需要使用 **Lora** 等专门的训练技术来生成融合度更好的图像。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://turboflip.de/flux-1-nf4-for-lowend-gpu-for-comfyui/">针对低端 GPU 的 ComfyUI FLUX 1 NF4 基础工作流 + img2img</a>：dᴉlɟ ǝɥʇ uᴉddᴉlɟ oqɹnʇ
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1294507163900710923)** (1 条消息): 

> - `LlamaIndex Hackathon`
> - `Slides and Resources` 


- **LlamaIndex Hackathon 公告**：一位成员分享了关于本周末即将举行的 **LlamaIndex Hackathon** 的详细信息。
   - 他们提供了一个链接，供参赛者访问所有相关的 [幻灯片与资源](https://bit.ly/llamaindex-rag-a-thon)。
- **参赛者资源**：鼓励参赛者查看提供的资源，为 Hackathon 做好准备。
   - [幻灯片与资源](https://bit.ly/llamaindex-rag-a-thon) 包含了支持他们项目的重要信息。



**提到的链接**：<a href="https://bit.ly/llamaindex-rag-a-thon">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1294404628321927240)** (4 条消息): 

> - `RAG pipeline with LlamaIndex`
> - `Advanced RAG deployment`
> - `Multi-agent workflow for business analysis`
> - `Multi-agent system for RFP response generation` 


- **使用 LlamaIndex 构建 RAG 流水线**：查看来自 @awscloud Developers 的这段 [视频](https://twitter.com/llama_index/status/1844845845797327352)，了解如何使用 **LlamaIndex** 构建 **RAG 流水线**，涵盖了基础工作流和组件。
   - 它包括一个简单的 RAG 实现和用于增强准确性的 router query 技术。
- **简单的 3 步高级 RAG 流程**：部署高级 **RAG** 可能很繁琐，但可以简化为 **3 步** 流程：用 Python 编写你的工作流，使用 **llama_deploy** 进行部署，然后运行它！
   - 由 @pavan_mantha1 编写的优秀 [教程](https://twitter.com/llama_index/status/1845143898206896365) 将引导你完成这些步骤。
- **用于业务分析的多 Agent 工作流**：发现一个关于构建 **多策略业务分析** 工作流的教程，其中包括公司历史、市场分析和策略画布。
   - Lakshmi 在一个 [推文线程](https://twitter.com/llama_index/status/1845502303215980782) 中分享了这种综合方法，值得一读。
- **用于 RFP 响应生成的 Agentic 工作流**：探索一份全新的指南，介绍如何开发一个 **多 Agent 系统**，根据你的知识库生成对 RFP 的响应。
   - 这种 Agentic 工作流可以无缝生成完整的响应，详见 [此发布内容](https://twitter.com/llama_index/status/1845853830485082474)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1294408862614687826)** (65 条消息🔥🔥): 

> - `DocumentSummaryIndex 过滤器`
> - `结合聊天历史的 RouterQueryEngine`
> - `Langfuse 集成问题`
> - `从 PDF 中提取图像`
> - `在 LlamaIndex 中实现 Colpali` 


- **过滤器不适用于 DocumentSummaryIndex**：一位用户表示在 `DocumentSummaryIndex` 上应用过滤器时遇到困难，指出 retriever 无法正确传递 kwargs。
   - 另一位成员建议采用不同的方法，并表示在他们的用例中可能不需要使用 Document Summary Index。
- **将聊天历史与 RouterQueryEngine 集成**：一位用户询问通过包含所有聊天消息，将 `RouterQueryEngine` 与聊天历史结合使用的可行性。
   - 建议使用 workflows 以实现更好的集成，并提供了相关示例链接。
- **Langfuse 集成故障排除**：一位用户报告在与 LlamaIndex 集成时，Langfuse 中的数据可见性出现问题，怀疑可能是由于其所在地区导致的。
   - 另一位成员建议确保在应用程序中包含将事件刷新（flush）到 Langfuse 的代码，以解决此问题。
- **从 PDF 中提取图像的挑战**：一位用户提出了从 PDF 中提取图像的问题，提到输出中出现了意外的 ASCII 字符。
   - 他们就此寻求指导，并希望明确解析数据的导出方式。
- **探索在 LlamaIndex 中实现 Colpali**：一位用户询问在 LlamaIndex 中实现 Colpali 的可能性，并指出缺乏相关文档。
   - 回复指出，虽然目前不支持 Colpali 的完整 embedding 模型，但有兴趣在未来将其作为 reranker 加入。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://langfuse.com/docs/integrations/llama-index/">Open Source Observability for LlamaIndex - Langfuse</a>：LlamaIndex 的开源可观测性工具。自动捕获 RAG 应用程序每个请求的详细追踪和指标。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows)">Workflows - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/#arize-phoenix-local">Observability - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/observability/LangfuseCallbackHandler/#flush-events-to-langfuse">Langfuse Callback Handler - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core%2Fllama_index%2Fcore%2Fchat_engine%2Fcontext.py#L223">llama_index/llama-index-core/llama_index/core/chat_engine/context.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/router_query_engine/">Router Query Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#structured-prediction">OpenAI - LlamaIndex</a>：未找到描述</li><li><a href="https://langfuse.com/docs/integrations/llama-index/example-python-instrumentation-module#custom-trace-properties,">Cookbook LlamaIndex Integration (Instrumentation Module) - Langfuse</a>：使用 LlamaIndex 的 instrumentation 模块进行实验性 LlamaIndex Langfuse 集成的示例指南。</li><li><a href="https://github.com/run-llama/llama_index/issues/16529,">Issues · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - Issues · run-llama/llama_index
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1294456056105406526)** (45 条消息🔥): 

> - `在 Tinygrad 中添加类型注解`
> - `Bounties 与贡献要求`
> - `Tinygrad 中的 SHA256 实现`
> - `MLPerf 推理与 StableDiffusion`
> - `会议议程与更新` 


- **评估添加类型注解的 PR**：讨论集中在评估三个用于添加类型注解的 PR 上，其中一个因性能问题被排除，而另一位知名贡献者的 PR 则获得了更高的权重。
   - 最后一个 PR 因测试失败而被驳回，引发了对不必要更改以及 PR 整体合并状态的担忧。
- **Bounties 的贡献要求**：George 指出，除非贡献者已经合并过几个 PR，否则他们不太可能解决 Bounty 任务，这强调了成为公认贡献者的明确晋升路径。
   - 为并行 SHA3 实现增加了 200 美元的 Bounty，强调了在处理更大型任务之前经验的重要性。
- **SHA256 实现挑战**：一位成员提议在 Tinygrad 中完整实现 SHA256，并讨论了并行处理的可能性，尽管有人对当前设计的局限性提出了担忧。
   - George 表达了对实现并行能力的渴望，并反思了性能方面的问题。
- **关于 MLPerf 的咨询**：一位新人询问了 MLPerf 与 tinygrad 仓库中现有示例的关系，随后有人建议参考现有文档。
   - 这指向了 Tinygrad 中与优化和 Benchmarking 相关的重点领域。
- **即将召开的会议与议程项目**：即将召开的会议将涵盖来自 Tiny Corp 的各种更新、性能结果和活跃的 Bounties，展示了持续的开发努力。
   - 主题包括云策略和驱动程序调试，表明了解决潜在挑战的协作努力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/7005/files">Add type annotations to classes in `nn/__init__.py` by mszep · Pull Request #7005 · tinygrad/tinygrad</a>：通过此更改，mypy --strict tinygrad/nn/__init__.py 仅对其他模块中函数的返回值报错。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/7001">refactor _make_hip_code_for_op into pm rules by ignaciosica · Pull Request #7001 · tinygrad/tinygrad</a>：由于 casting 不再固定且现在是一个 uop，process_replay 存在差异。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/7003">nn/__init__.py type annotations by Ry4nW · Pull Request #7003 · tinygrad/tinygrad</a>：未找到描述。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/7002">add types in all nn/init.py classes by bhavyagada · Pull Request #7002 · tinygrad/tinygrad</a>：为 nn/init.py 文件中的所有类和函数添加了类型注解。此外，还为不需要换行的变量包含了类型注解。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1294895083325362322)** (11 条消息🔥): 

> - `Tinygrad 中的 Action Diffusion`
> - `DDPM/DDIM 调度库`
> - `Tensor 中的梯度轴匹配`
> - `Lean 证明开发`
> - `结合 Action Diffusion 的机器人控制` 


- **Tinygrad 中的 Action Diffusion 需要更好的文档**：一位用户指出了 Tinygrad 中 [Action Diffusion](https://github.com/mdaiter/diffusion_action_policy) 库的问题，指出代码强制转换回 numpy，并且在被 `TinyJit` 包装时错误地添加了噪声。
   - 他们计划下周创建一个兼容性更好的 diffuser 调度器，以便更好地与 Tinygrad 项目集成。
- **用户为 Metal 训练创建了 DDPM 调度器**：一位成员宣布他们开发了自己的 **DDPM scheduler**，因为他们在 Tinygrad 中找不到现有的库，使其可用于在 Metal 上训练 diffusion 模型。
   - 他们愿意与社区中需要的其他人分享他们的工作。
- **Tensor 梯度轴不匹配问题**：围绕在更新 Tensor 'w' 及其梯度时解决梯度轴不匹配的策略展开了讨论，用户提出了多种潜在的解决方案。
   - 选项包括对齐轴、移除约束或 resharding，尽管后者被认为比较浪费。
- **合并开发中 Lean 证明的挑战**：一位成员分享了在开发用于合并的 Lean 证明时的挫败感，理由是 Lean4 语法的困难以及与 Lean3 相比资源的匮乏。
   - 他们邀请其他人组队，以使 Lean4 的学习过程更容易，并提出分享他们的逻辑框架。
- **通过 action diffusion 探索机器人技术**：一位用户表达了对将机器人控制与 **action diffusion** 相结合的热情，特别提到打算探索 Vq-BET 和 ACT + ALOHA 等概念。
   - 他们还发出了合作邀请，旨在让 diffusers 运行起来，特别是具有 **ResNet18** 背景的合作。



**提及的链接**：<a href="https://github.com/mdaiter/diffusion_action_policy">GitHub - mdaiter/diffusion_action_policy: Action Diffusion, in Tinygrad</a>：Tinygrad 中的 Action Diffusion。通过在 GitHub 上创建账户来为 mdaiter/diffusion_action_policy 的开发做出贡献。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1294375043123384422)** (4 条消息): 

> - `复现 OpenAI 的 O1 模型`
> - `Journey Learning 范式`
> - `Dowehaveopeno1.com 倡议` 


- **复现 OpenAI O1 模型的见解**：关于复现 OpenAI **O1** 的第一份深入报告强调了一种名为 “journey learning” 的新训练范式，该范式仅使用 **327 个训练样本** 就在 **数学推理** 中集成了搜索和学习。
   - *Zhen Huang* 指出，这种方法包括试错和回溯，使性能提升了 **8%**。
- **复现历程的文档记录**：该报告详尽地记录了团队在 **复现历程** 中遇到的发现、挑战和创新方法。
   - 随附的 [论文](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf) 和 [代码](https://github.com/GAIR-NLP/O1-Journey) 可供进一步了解。
- **关于 Dowehaveopeno1.com 的讨论**：有人建议创建 **dowehaveopeno1.com** 网站，表现出对整合 OpenAI 模型相关信息的兴趣。
   - 然而，另一位成员评论说，虽然这是进步，但现在实施这个想法还为时过早。
- **对 Twitter 监管的担忧**：一位成员表示自己在 **Twitter** 上不够活跃，无法有效地监控讨论。
   - 这指向了关于管理社区讨论和信息传播的更广泛担忧。



**提及的链接**：<a href="https://x.com/stefan_fee/status/1844775434740809794">Pengfei Liu (@stefan_fee) 的推文</a>：关于复现 OpenAI o1 的第一份深入技术报告！！！揭示试错见解和艰辛教训的宝库。一些亮点：(1) 我们引入了一种新的训练范式...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1295418182219071530)** (25 messages🔥): 

> - `.io 域名的影响`
> - `Microsoft AI 研究员加入 OpenAI`
> - `o1-turbo-mini 的 Benchmark 性能`
> - `AGI 的定义与讨论`
> - `AI 研究员的财务成功` 


- **.io 域名变更引发讨论**：英国政府移交岛屿主权的地缘政治影响引发了对 **.io** 域名后缀未来的猜测，促使了关于 **Big Tech** 可能介入的讨论。
   - 链接的一篇文章强调了历史事件如何破坏数字空间，将当前局势与过去的地缘政治转变进行了类比。
- **Microsoft 顶尖 AI 研究员跳槽 OpenAI**：据包括 [The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx) 在内的多家媒体报道，Microsoft 领先的 AI 研究员之一 **Sebastien Bubeck** 正前往 **OpenAI**。
   - 这一举动引发了人们对这类职业变动动机的幽默调侃和怀疑，尤其是考虑到当今 AI 职位的丰厚待遇。
- **o1-turbo-mini 的 Benchmark 表现令人印象深刻**：有关 **o1-turbo-mini** 模型在 Benchmark 上表现异常出色的讨论不断，一些人认为其表现出人意料地好。
   - 评论还反映出，这种性能表现可能会引发对那些怀疑 AI 进步的人的戏谑反应。
- **关于 AGI 及其影响的辩论**：一位成员批判性地指出 Bubeck 的变动...
   - 对话转向了 AGI 模糊的定义，并对讨论中涉及的细微差别达成了共识。
- **AI 研究员获得巨额收益**：在 AI 热潮中，顶尖研究员获得显著财务收益的趋势显而易见，这引发了一些在该领域财务成就较少的人的戏谑嘲讽。
   - 这种财富上的反差助长了打趣之风，因为行业变革正在迅速为少数人创造财富。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://every.to/p/the-disappearance-of-an-internet-domain">The Disappearance of an Internet Domain</a>: 地缘政治如何改变数字基础设施 </li><li><a href="https://x.com/amir/status/1845905601110462481">Amir Efrati (@amir) 的推文</a>: 新闻：Microsoft 顶尖 AI 研究员之一 @SebastienBubeck 正在加入 OpenAI。 https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1294451755345772575)** (6 messages): 

> - `OpenAI 爬虫限制`
> - `Dario Amodei 的文章`
> - `OpenAI 衍生公司`
> - `前 OpenAI 员工创立初创公司的崛起`
> - `AI 领域的公司动态` 


- **OpenAI 无法抓取 Dario Amodei 的网站**：一个引人注目的评论指出，唯一不被允许抓取 [Dario Amodei 网站](https://x.com/soldni/status/1844886515580879070) 的机器人是 **OpenAI**，这引发了一些关注。
   - 这引发了关于 AI 研究透明度和可访问性的幽默猜测。
- **Dario Amodei 的文章令人印象深刻**：一位成员称赞了 Dario 的文笔，称*他的文章写得非常精彩*，并推测由于其 Markdown 的使用，在 **lmsys** 上会有不错的表现。
   - 对内容的如此高度评价表明，人们对 AI 领域高质量学术讨论的兴趣日益浓厚。
- **新的 OpenAI 衍生公司传闻**：关于又一家 **OpenAI 衍生公司** 的猜测已经出现，并被认为对消费者有利。
   - 这预示着由主要巨头公司的前员工所推动的 AI 初创生态系统可能存在增长趋势。
- **前 OpenAI 员工启动初创公司**：据报道，很快将有 **1,700 家初创公司** 由前 OpenAI 员工创立，标志着显著的行业活跃度。
   - 创业公司的激增暗示了 AI 领域的创新和多样化。
- **OpenAI 对公司结构的影响**：有一个大胆的说法称，OpenAI 不仅在创新 AI，还在引起美国企业界的**左翼分裂**。
   - 这一评论反映了 AI 进步和公司治理对社会更深层次的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/soldni/status/1844886515580879070">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>：唯一不被允许抓取 @DarioAmodei 网站的机器人是 @OpenAI 💀</li><li><a href="https://fxtwitter.com/pitdesi/status/1845849405699842320">来自 Sheel Mohnot (@pitdesi) 的推文</a>：又一个 OpenAI 衍生公司要来了？对消费者来说太棒了</li><li><a href="https://x.com/aidan_mclau/status/1844869102898380869">来自 Aidan McLau (@aidan_mclau) 的推文</a>：dario 的文章写得太棒了。虽然用了大量的 markdown。感觉它在 lmsys 上会表现很好。等等...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1294424408709922868)** (8 messages🔥): 

> - `Machines of Loving Grace`
> - `积极的 AI 未来`
> - `Opus 发布暗示` 


- **《Machines of Loving Grace》收获好评如潮**：这篇名为 [Machines of Loving Grace](https://darioamodei.com/machines-of-loving-grace) 的文章被描述为拥有一个*完美的标题*，立即吸引了人们的注意。
   - 一位成员评论说，开头*非常棒*且引人入胜，激发了人们对 AI 潜力讨论的兴趣。
- **行业领袖强调 AI 的积极面**：一位成员表示，行业领袖终于开始讨论 AI 的**积极面**，这让他感到欣慰，并表示他已经*厌倦了围绕该话题的恐慌情绪*。
   - 这种情绪标志着人们对 AI 未来潜力的积极对话越来越感兴趣。
- **Opus 即将到来的暗示**：一位用户暗示 **Opus** 可能很快就会推出，预示着 AI 领域即将迎来新的发展。
   - 这种猜测为正在进行的关于未来 AI 进步的讨论增添了兴奋感。



**提到的链接**：<a href="https://darioamodei.com/machines-of-loving-grace">Dario Amodei — Machines of Loving Grace</a>：AI 将如何让世界变得更好

  

---

### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1295362097995059321)** (8 messages🔥): 

> - `Vijay Pande 与诺贝尔化学奖`
> - `Folding@Home 的影响`
> - `EVfold 与蛋白质折叠算法`
> - `药物研发中的分子对接 (Docking)`
> - `神经网络的历史性论文` 


- **Vijay Pande 意外无缘诺贝尔化学奖**：一名成员对 **Vijay Pande** 没有获得**诺贝尔**化学奖表示惊讶，并指出 Schrödinger 的许多 AI 同事都曾是他的组成员。
   - *有人指出了他的工作对 NVIDIA **CUDA** 的潜在影响*，并质疑通过 Folding@Home 等项目对生物学产生了多大的影响。
- **Folding@Home：早期但影响有限**：一位成员认为 **Folding@Home** 可能没有产生应有的影响，认为现在实现其全部潜力还为时过早。
   - 他们强调 **DeepChem** 是 Pande 的另一项重要贡献，推动了量子化学中的数据科学。
- **EVfold：蛋白质折叠中缺失的认可**：讨论了 **AlphaFold** 和其他依赖氨基酸共变异（co-variation）的蛋白质折叠算法，这一概念最初由 **EVfold** 在 2011 年提出。
   - *一位成员感叹道，诺贝尔奖似乎更青睐最终产品，而非基础科学理念*。
- **诺贝尔奖讨论中忽视了分子对接（Docking）的主导地位**：另一位成员指出，**Docking** 在先导药物研发中的重要作用在诺贝尔奖讨论中没有得到充分认可。
   - 他们强调这一概念已被广泛使用，但其重要性却未被承认。
- **1982 年关于神经网络的历史性论文**：一位成员引用了 **John Hopfield** 1982 年发表在 PNAS 上的关于**神经网络**的论文，并分享了链接。
   - *随后的一条评论将其称为来自“婴儿潮（boomer）”时代的有趣引用。*


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1294386915751231539)** (5 messages): 

> - `Next.JS 语音面试准备平台`
> - `用于 Signatures 的 FastAPI 路由`
> - `使用 Typer 生成 CLI 命令`
> - `GeneratePureJSX Signature`
> - `GenerateMarkdown Signature` 


- **Next.JS 语音面试准备平台发布**：一位成员在休息室宣布开发了一个全栈 **Next.JS** 语音面试准备/测试平台。
   - 这个创新平台旨在通过语音交互增强面试准备。
- **为 Signatures 创建 FastAPI 路由**：一位成员分享了一段代码片段，可以将任何 **dspy.Signature** 转换为 **FastAPI** 路由，允许其将 predictor 作为字典返回。
   - 该实现利用 **init_instant** 函数来初始化处理请求的环境。
- **使用 Typer 动态创建 CLI 命令**：分享了一个将 **dspy.Signature** 类转换为使用 **Typer** 的 CLI 命令的代码示例，展示了自动化能力。
   - 该功能根据 Signature 类中指定的输入字段动态构建命令函数。
- **展示 GeneratePureJSX Signature**：展示了 **GeneratePureJSX** Signature，旨在为 React 环境生成纯净的 JSX 代码。
   - 该 Signature 包含上下文和需求的字段，可生成结构良好的输出。
- **重点介绍 GenerateMarkdown Signature**：介绍了另一个 Signature **GenerateMarkdown**，用于根据给定的标题和内容创建 Markdown 文本。
   - 与 GeneratePureJSX 类似，它提供了一个带有输入字段的定义结构，用于生成 Markdown。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.loom.com/share/9b9b6964cbd6471c8f31616e4f939a6c">DSLModel 框架介绍</a>: https://github.com/seanchatmangpt/dslmodel 大家好，我是 Sean Chatman，这里向大家介绍 DSL Model 框架，这是一个使用 DSPy 和 Jinja 进行数据建模的强大工具。在这个视频中，我解释了如何...</li><li><a href="https://www.youtube.com/watch?v=ZEi4OTuFa3I"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/seanchatmangpt/dslmodel">GitHub - seanchatmangpt/dslmodel: 来自 DSPy 和 Jinja2 的结构化输出</a>: 来自 DSPy 和 Jinja2 的结构化输出。通过在 GitHub 上创建一个账户来为 seanchatmangpt/dslmodel 的开发做出贡献。
</li>
</ul>

</div>

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1294397086980374598)** (4 条消息): 

> - `用于 ICL 选择的 GraphIC`
> - `推理时计算对 LLM 的影响`
> - `知识密集型任务中的 StructRAG` 


- **GraphIC 变革了 ICL 选择**：论文介绍了 [GraphIC](https://arxiv.org/abs/2410.02203)，这是一种利用基于图的表示和贝叶斯网络 (BNs) 来增强大语言模型 (LLMs) 的上下文示例 (ICEs) 选择的技术。它解决了传统基于文本的嵌入方法在多步推理任务中的局限性。
   - *图结构过滤掉了浅层语义*，保留了有效多步推理所需的更深层推理结构。
- **通过推理时计算增强 LLM**：关于 [推理时计算 (inference-time computation)](https://arxiv.org/abs/2408.03314) 的研究探讨了允许 LLM 利用固定的额外计算资源如何提升在困难提示词上的性能。研究结果强调了对未来 LLM 预训练以及推理时计算与预训练计算之间平衡的重要启示。
   - 目前的研究表明，理解测试时推理方法的扩展行为 (scaling behaviors) 至关重要，尽管许多策略已经显示出负面结果。
- **StructRAG 优化了推理任务中的检索**：论文提出了一种名为 [StructRAG](https://arxiv.org/abs/2410.08815) 的新框架，旨在增强检索增强生成 (RAG) 方法，重点是更好地处理知识密集型推理任务。通过将原始信息转换为结构化知识，该框架旨在有效地识别所需信息。
   - 这种结构化方法可以针对各种任务优化信息，从而在现有 RAG 方法传统上面临的噪声增强环境中促进推理能力的提升。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.08815">StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization</a>：检索增强生成 (RAG) 是在许多基于知识的任务中有效增强大语言模型 (LLMs) 的关键手段。然而，现有的 RAG 方法在处理知识密集型...</li><li><a href="https://arxiv.org/abs/2410.02203">GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning</a>：上下文学习 (ICL) 使大语言模型 (LLMs) 能够通过在输入中直接加入少量上下文示例 (ICEs) 来泛化到新任务，而无需更新参数。然而，...</li><li><a href="https://arxiv.org/abs/2408.03314">Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters</a>：使 LLM 能够通过使用更多的测试时计算来改进其输出，是构建能够在开放式自然语言上运行的通用自我改进 Agent 的关键一步。在本文中...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1294384940225663026)** (18 条消息🔥): 

> - `LLM Classifier Training` (LLM Classifier 训练)
> - `Versioning Documentation` (版本化文档)
> - `Creative Writing Criteria Optimization` (创意写作标准优化)
> - `Contextual Embedding in DSPy` (DSPy 中的 Contextual Embedding)
> - `Metric Programs for Evaluation` (用于评估的 Metric 程序)


- **LLM Classifier 寻求歧义处理**：一位用户正在训练 **LLM classifier**，并寻求社区关于处理分类歧义的建议，提议在模型遇到不确定性时采用一种输出方法。
   - 提出的问题是：*我是否应该为所有可能的歧义创建单独的类别？*
- **社区建议增强 LLM Classifier**：针对歧义问题的回应是建议不要创建新类别，而是建议在 **LLM signature** 中添加第二个输出字段来声明歧义。
   - 一位成员甚至提出可以制作一个小型示例原型来协助实现。
- **版本化文档以提高清晰度**：一位新用户请求 DSPy 提供版本化文档，表示这有助于澄清文档中已反映但尚未发布的更改用法。
   - 社区成员承认这是一个明智的想法，并指出最新版本现已发布。
- **优化器中显式标准的探索**：讨论开发一种优化器，能够识别任务的显式标准，以根据人工标签揭示输出中的模式。
   - 建议采用一种详细的方法，涉及一个使用模式匹配来生成与创意写作成功或失败相关的标准的程序。
- **需要 Contextual Embedding 指导**：一位用户询问关于在 DSPy 程序中使用 **contextual embeddings** 的 cookbook 或指南，并引用了外部资源。
   - 该用户表示困惑，并请求协助以更好地理解其实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/lateinteraction/status/1845168986750845023">来自 Omar Khattab (@lateinteraction) 的推文</a>：当 chatgpt 终于讲出一个好笑的笑话时：</li><li><a href="https://github.com/Scale3-Labs/dspy-examples/tree/main/src/summarization/programs/metric">dspy-examples/src/summarization/programs/metric at main · Scale3-Labs/dspy-examples</a>：由 Langtrace AI 团队构建和维护的一系列使用 DSPy 开发的 AI 程序示例。- Scale3-Labs/dspy-examples
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1295021218851721287)** (9 条消息🔥): 

> - `Chatbot Features Extraction` (聊天机器人特征提取)
> - `Off-topic Classification` (离题分类)
> - `Input Training Set` (输入训练集)
> - `Cosine Similarity Metric` (Cosine Similarity 指标)


- **Stuart 的聊天机器人特征提取过程**：Stuart 正在构建一个提取特征的聊天机器人，包括通过标准数据库识别离题查询。
   - 这种方法包括一个 **off-topic table**（离题表），其中包含诸如 *Insufficient Info*（信息不足）之类的条目，用于标记缺乏上下文的查询。
- **典型输入的训练集**：Stuart 确认拥有一个根据查询是否离题进行分类的训练集。
   - 该数据集旨在通过过滤无关查询来提高聊天机器人的响应准确性。
- **评估输出有效性**：okhattab 询问是否开发了指标来评估聊天机器人输出是否符合既定标准。
   - Stuart 提到他缺乏预定义的类别，并正在考虑使用 **cosine similarity**（余弦相似度）将输入查询与生成的类别进行比较。
- **寻求改进建议**：Stuart 积极寻求关于增强其用于评估输出的 **cosine similarity** 指标方法的建议。
   - 他的重点是完善查询与相应类别的匹配方式，以改进离题检测。
- **社区对主题讨论的兴趣**：另一位成员表示有兴趣了解如何实现离题分类的改进。
   - 这表明社区内对增强聊天机器人有效性的方法有着广泛的好奇心。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1294629508376432641)** (18 条消息🔥): 

> - `LLaMA 3.2`
> - `PixTral 对比`
> - `Epic Games & Sketchfab`
> - `高级 Cookie Monster 模式`
> - `CogVLM 输出` 


- **LLaMA 3.2 在流行文化知识方面表现出色**：讨论强调了 **LLaMA 3.2** 在流行文化知识方面优于竞争对手，这归功于其在 **50 亿张图像**上的训练，使其生成的 captions 更加连贯。
   - *与 Molmo 或 PixTral 等模型相比，LLaMA 3.2 在图像 captions 中提供了更好的上下文理解。*
- **PixTral 最适合成人内容**：成员们注意到 **PixTral** 的独特定位，主要在针对成人内容进行训练时表现出色，而 **LLaMA 3.2** 则在通用语境中表现优异。
   - *强调在侧向对比中，LLaMA 3.2 因其更广泛的文化相关性而更受青睐。*
- **Epic Games 移除 Sketchfab 的影响**：**Epic Games** 决定移除 Sketchfab 将导致 Objaverse 损失 **80 万个 3D 模型**，敦促成员尽快下载。
   - *该公告引发了关于对 3D 建模社区以及依赖该资源的用户的潜在影响的讨论。*
- **讨论了高级 Cookie Monster 模式**：分享了一个标题为 **'Advanced Cookie Monster Mode :D'** 的 YouTube 链接，推介其有趣的内容，但缺乏进一步细节。
   - *未提供关于该视频的其他评论或上下文。*
- **CogVLM 展示了对卡通文化的参与**：**CogVLM** 被描述为通过生动描绘庆祝 **Cartoon Network 20 周年**的角色，捕捉到了卡通文化的精髓。
   - *其输出因识别出广泛的角色而受到关注，突显了对粉丝的怀旧价值。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://swivid.github.io/F5-TTS/">F5-TTS</a>: 未找到描述</li><li><a href="https://youtu.be/AH_dbzdNu98">Advanced Cookie Monster Mode :D</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/CaptionEmporium/laion-pop-llama3.2-11b">CaptionEmporium/laion-pop-llama3.2-11b · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1294871241454522429)** (10 条消息🔥): 

> - `o1-mini vs o1-preview`
> - `o1 LLM 的性能`
> - `作为表示学习器的 Diffusion Models`
> - `Representation Alignment (表示对齐)`
> - `使用 CLIP 训练 T2I 模型` 


- **o1-mini 无法匹配 o1-preview 的性能**：成员分享称 **o1-mini** 缺乏 **o1-preview** 的能力，并将其描述为在简单问题上表现脆弱。他们指出，**o1-preview** 在鲁棒性方面优于所有其他 SOTA LLM，甚至在奥林匹克级别的任务中也是如此 [链接](https://x.com/JJitsev/status/1845309654022205720)。
   - *Yet another tale of Rise and Fall* 强调了虽然声称 **o1-mini** 可以处理复杂问题，但在基础推理任务上却表现不佳。
- **o1 是最佳的 LLM 体验**：一位成员强调 **o1** 是他们使用过的最好的 LLM，能有效处理论文和代码。他们声称 **o1** 生成的几乎所有代码都能在第一次尝试时运行，展示了其可靠性。
- **来自 REPresentation Alignment 论文的见解**：一位成员讨论了一篇关于生成式 **diffusion models** 的论文，强调了通过外部高质量表示带来的性能显著提升。他们质疑这种增强是源于知识蒸馏（knowledge distillation），还是关于生成建模过程的更深层见解。
   - **REPresentation Alignment** 的引入表明，在将隐藏层输出连接到图像的神经表示时，训练效率有了显著提升。
- **使用 vision encoders 加速训练**：另一位成员指出，将隐藏层的输出与 **vision encoders** 对齐可以加速训练，因为它们具有丰富的图像表示。他们注意到这与使用 **CLIP**（一种与图像对齐的文本编码器）训练 T2I 模型有相似之处。
- **CLIP 对比训练带来的挑战**：虽然使用 **CLIP** 训练 T2I 模型可以加速过程，但它也会引入与其对比训练（contrastive training）方法相关的伪影（artifacts）。该成员表示，这些伪影可能会影响整体训练质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sihyun.me/REPA/">Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think </a>：Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think</li><li><a href="https://x.com/JJitsev/status/1845309654022205720">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：(Yet) another tale of Rise and Fall：据称 o1-mini 在奥林匹克级别的数学和编程问题上可以匹配更大规模的 o1-preview。但它能处理揭示泛化能力的简单 AIW 问题吗...
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1295457284562092166)** (1 条消息): 

> - `Graham Neubig 的讲座`
> - `软件开发中的 AI agents`
> - `现实世界软件任务中的挑战`
> - `多模态数据处理` 


- **直播讲座预告：Graham Neubig 谈 AI Agents**：今天 **PST 时间下午 3:00**，我们的第 6 场讲座将由 **Graham Neubig** 主讲“用于软件开发的 Agents”，届时将在此 [直播](https://www.youtube.com/live/f9L9Fkq-8K4)。
   - 他将涵盖软件开发 agents 的 **state-of-the-art**，解决诸如编辑文件和多模态数据处理等挑战。
- **探索软件开发任务中的挑战**：Graham Neubig 旨在解决开发 AI agents 以协助现实世界软件任务的复杂性，特别是针对大型代码库（repositories）。
   - 关键挑战包括确定要编辑的文件、测试编辑内容以及将 **web browsing** 集成到编码过程中。
- **讲者介绍：Graham Neubig**：**Graham Neubig** 是卡内基梅隆大学的副教授，专注于自然语言处理研究。他的研究重点包括用于 **code generation** 和 **question answering** 等应用的 LLM。
   - 鼓励观众在指定的课程工作人员频道中提问，以进行进一步咨询。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1294408048395161731)** (18 条消息🔥): 

> - `Course Registration Issues` (课程注册问题)
> - `Quiz Google Form Access` (测验 Google Form 访问)
> - `Course Timings` (课程时间)
> - `Supplemental Course Information` (补充课程信息)
> - `Lab Access Questions` (Lab 访问问题)


- **课程注册仍开放**：新成员表示有兴趣在 **Dec 12** 截止日期前报名参加课程，并获知目前仍可注册，确认链接对其他人有效。
   - 另一位用户在排除链接故障后确认注册成功。
- **测验 Google Form 故障排除**：多名用户报告访问测验的 Google Forms 链接时出现问题，但回复显示在清除缓存后链接对其他人有效。
   - 一位参与者在排除故障后确认可以访问表单，强调这可能只是浏览器问题。
- **课程时间公布**：一位用户询问直播安排，获知课程时间为 **3:00 PM to 5:00 PM PST**。
   - 这一信息对于需要计划参加直播课程的参与者至关重要。
- **有用链接与资源**：课程材料和补充信息（包括说明和额外资源）已确认可在课程网站上获取。
   - 成员们被提醒可以通过 Discord 服务器获取课程报名和社区聊天链接。
- **Lab 访问延迟**：新参与者担心在提交报名表后未收到关于 Lab 和作业访问权限的电子邮件。
   - 他们被引导至课程网站，以获取有关 Lab 指南和作业预期的全面概述。



**提到的链接**：<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1294638771505070233)** (7 条消息): 

> - `Definition of AI Agent` (AI Agent 的定义)
> - `Agentic Applications` (Agentic 应用)
> - `Chain of Thought (CoT) Methodology` (Chain of Thought (CoT) 方法论) 


- **澄清 AI Agent 标准**：AI Agent 的特征是能够通过与数据库或 API 交互来自主执行任务，从而实现既定目标。例如，虽然 **ChatGPT** 符合 Agent 的定义，但 **gpt-3.5-turbo** 仍然只是一个不具备此类能力的 LLM 模型。
- **Agentic 应用详解**：Agentic 应用会根据新证据改变其内部状态，并在奖励系统的驱动下独立运行。这种机制允许应用通过反映现实世界的情况进行演进。
- **众包 AI Agent 定义**：目前正在努力完善 AI Agent 的定义，相关贡献和讨论正在 Twitter 等平台上进行。一个显著的响应是建议协作跟踪不同版本的定义。
- **理解 CoT 对 LLM 的有效性**：**Chain of Thought (CoT)** 方法论通过生成简化复杂问题的中间步骤来增强 LLM 的问题解决能力。通过分解大型任务，可以从这些较小的组件中推导出解决方案。
- **社区关于 CoT 的讨论**：成员们讨论了 CoT 方法的实际优势，强调了其在应对复杂挑战时促进思路清晰的能力。正如 **Apple** 的一个例子所强调的，这种策略能有效帮助得出最终答案。



**提到的链接**：<a href="https://x.com/simonw/status/1843290729260703801?s=46">Simon Willison (@simonw) 的推文</a>：让我们看看是否可以众包一个关于“agent”（针对 AI 和 LLM）的稳健定义，且符合 <=280 字符的推文要求。请回复你的最佳尝试，然后浏览回复...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1295481309501722634)** (1 条消息): 

> - `AI-Powered Search Book` (AI 驱动搜索书籍) 


- **AI 驱动搜索的必读书籍**：一位成员强调 [这本书](https://www.manning.com/books/ai-powered-search) 预计将成为未来几年 AI 驱动搜索领域的权威资源。
   - *这本书可能会成为未来几年的首选资源。*
- **AI 驱动搜索书籍的预期影响力**：讨论强调了该书对 AI 驱动搜索领域的潜在影响，预测了其对未来专业实践的相关性。
   - 该成员推测它将成为 AI 从业者和研究人员的关键参考资料。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1294621049035690054)** (10 messages🔥): 

> - `Gemma 2 support` (Gemma 2 支持)
> - `Flex Attention implementation` (Flex Attention 实现)
> - `Logit softcapping`
> - `CUDA dependencies` (CUDA 依赖)
> - `Testing considerations` (测试考量)


- **通过 Flex Attention 支持 Gemma 2**：成员们讨论了使用 **Flex Attention** 实现 **Gemma 2**，并指出唯一的阻碍是 **logit softcapping**，这需要定义一个合适的 `score_mod` 函数。
   - 建议使用 Flex 的**权衡**是可以接受的，因为它简化了流程，尽管可能需要依赖具有高计算能力的 CUDA。
- **Logit Softcapping 和 Flex API**：有人指出，虽然 **logit softcapping** 在 `F.sdpa` 中不直接可用，但可以通过 Flex API 在 `_sdpa_or_flex_attention()` 中暴露，从而实现它。
   - 讨论强调，转向 **Flex** 可以更快地适应未来模型中即将出现的注意力算子。
- **测试和实现考量**：针对不同模型大小的 Gemma 2 所需的**测试**提出了担忧，并建议向拥有更多计算资源的人员众包测试工作。
   - 参与者一致认为，确保**实现**的稳健性非常重要，即使这意味着在需要时依赖更简单的回退方案 (fallback)。
- **CUDA 依赖考量**：大家达成共识，通过 Flex 实现 **Gemma 2** 可能会受到要求具备足够计算能力的 **CUDA** 的限制，这突显了非 CUDA 用户的差距。
   - 成员们承认，这种限制将允许更集中地逐步开发所需的各种支持功能。
- **Naive Fallback vs Flex**：一名成员提出，拥有一个 **`naive_fallback`** 可能有利于支持所有新出现的功能，尽管维护开销可能会增加。
   - 也有人担心 Flex 的灵活性是否能随着时间的推移减轻这些维护挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_import_guard.py#L15).">torchtune/torchtune/utils/_import_guard.py at main · pytorch/torchtune</a>: PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L942">pytorch/torch/nn/attention/flex_attention.py at main · pytorch/pytorch</a>: Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1294821984194138234)** (1 messages): 

> - `Flex attention on RTX 4090` (RTX 4090 上的 Flex attention)
> - `Shared memory issues in PyTorch` (PyTorch 中的共享内存问题)


- **面临 Flex Attention 的共享内存溢出问题**：一位用户在 [GitHub 上分享了一个 issue](https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459)，关于在 **RTX 4090** 上使用 **flex attention** 时遇到的共享内存溢出 (out-of-shared-memory) 问题。
   - 该 issue 描述了遇到的具体错误，并为其他遇到类似问题的人提供了潜在的修复建议。
- **关于共享内存资源的 Bug 报告**：该 GitHub issue 概述了一个 bug，即使用 **flex attention** 会导致共享内存资源不足错误。
   - 它包含了一个最小复现代码片段，并鼓励协作解决已识别的问题。



**提到的链接**: <a href="https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459">Shared memory out of resource when using flex attention · Issue #133254 · pytorch/pytorch</a>: 🐛 Bug 描述：当我在一块 RTX 4090 上使用 flex attention 时，遇到了错误。最小复现代码：import torch from torch.nn.attention.flex_attention import flex_attention flex_attention = torch.com.....

  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1294396797514682419)** (14 messages🔥): 

> - `Aria 模型发布`
> - `Pixtral 12B 性能`
> - `State of AI Report 2024`
> - `用于分布式训练的 LegoScale`
> - `OpenReview 上的 ICLR 投稿` 


- **Aria 开源多模态模型发布**：新模型 **Aria** 是一款开源原生多模态 AI，在语言和编程任务中表现出色，采用 **3.9B** 和 **3.5B** 参数架构。
   - 在经过全面的预训练过程后，其性能超越了 *Pixtral-12B* 和 *Llama3.2-11B*。
- **Pixtral 12B 面临竞争**：**Pixtral 12B** 模型与 Aria 进行了对比，后者展示了更优越的性能，但由于两者几乎同时发布，目前缺乏直接的 Benchmark 对比。
   - 成员们注意到 Aria 在发布时的一些新 Benchmark 表现更好，强调了进行更清晰的正面交锋对比的必要性。
- **State of AI 2024 Report 引发见解**：由 **Nathan Benaich** 撰写的 **State of AI Report 2024** 讨论了 AI 领域的主要趋势和投资，极少提及像 *Torchtune* 这样的特定模型。
   - 该报告旨在为有关 AI 发展轨迹的讨论提供信息和启发，特别是在医学和生物学等领域的影响。
- **LegoScale 有望推动分布式训练的进步**：**LegoScale** 提出了一个可定制的 PyTorch 原生系统，能够为大语言模型实现模块化的 3D 并行预训练，显著提升性能。
   - 它旨在简化跨多个库和庞大 GPU 生态系统的复杂训练，有可能彻底改变分布式训练方法。
- **ICLR 投稿可在 OpenReview 上查看**：成员们讨论了 ICLR 如何利用 OpenReview 来获取 Preprint 并跟踪整个投稿过程中的评审意见。
   - 这种方法与 NeurIPS 形成对比，后者的投稿除非公开分享，否则保持隐藏状态。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.stateof.ai/">State of AI Report 2024</a>：State of AI Report 分析了 AI 领域最有趣的进展。在此阅读并下载。</li><li><a href="https://arxiv.org/abs/2410.05993">Aria: An Open Multimodal Native Mixture-of-Experts Model</a>：信息以多种模态呈现。原生多模态 AI 模型对于整合现实世界信息并提供全面理解至关重要。虽然专有的原生多模态模型...</li><li><a href="https://openreview.net/forum?id=SFN6Wm7YBI">LegoScale: One-stop PyTorch native solution for production ready...</a>：大语言模型 (LLMs) 的发展对于推动最先进的自然语言处理应用起到了关键作用。训练具有数十亿参数和数万亿...</li><li><a href="https://arxiv.org/abs/2410.07073">Pixtral 12B</a>：我们推出了 Pixtral-12B，一个拥有 120 亿参数的多模态语言模型。Pixtral-12B 经过训练可以理解自然图像和文档，在各种多模态任务上实现了领先的性能...
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1294440757809385504)** (15 条消息🔥): 

> - `社区关闭公告`
> - `Swarm.js 发布`
> - `房地产 LangChain 项目`
> - `OpenAI API 集成`
> - `ImageMessage 类型讨论` 


- **社区关闭公告**: Jess 宣布 LangChain Discord 社区将于 **2024 年 10 月 31 日**关闭，以便专注于构建一个全新的、改进后的社区。鼓励成员填写表格以获取有关变更的最新信息。
   - 欢迎通过 community@langchain.dev 提供反馈，并邀请有兴趣成为版主的人员加入。
- **Swarm.js 发布**: Pulkitgarg 介绍了 **Swarm.js**，这是一个面向 JavaScript 社区的 Node.js SDK，用于使用 OpenAI API 编排多 Agent 系统。它包含使用自定义指令定义 Agent 的功能，使过程变得简单且符合人体工程学。
   - 鼓励开发者[查看 GitHub 仓库](https://github.com/youseai/openai-swarm-node)并为项目做出贡献。
- **房地产 LangChain 项目**: Gustaf_81960_10487 分享了他们的开源 LangChain 房地产项目，该项目帮助他们获得了一份工作，并表示愿意为贡献者提供推荐信。他们为该项目创建了一个 GitHub 仓库，并提供了其网站 [Condo Cube](https://condo-cube.com/) 的链接。
   - 鼓励贡献者查看仓库并参与解决现实世界的问题。
- **OpenAI API 集成咨询**: Mikef0x 询问了集成 OpenAI 实时 API 的计划，并提到了 Pydantic 的问题。成员们讨论了他们在 Azure OpenAI 和 SQLDatabase 方面的经验，并指出了一些幽默的响应结果。
- **ImageMessage 类型讨论**: Grigorij0543 建议在 LangChain 中引入 **ImageMessage** 类型，以简化聊天中的图像处理。Clay_ferguson 评论说文本通常与图像相关联，并建议需要一种更通用的文件关联方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://airtable.com/app9AB74Dql7uubL2/pagTKrmJu1rQRkJKV/form">Airtable | 每个人的应用平台</a>: Airtable 是一个用于构建协作应用的低代码平台。自定义您的工作流，进行协作并实现宏伟目标。免费开始使用。</li><li><a href="https://github.com/youseai/openai-swarm-node">GitHub - youseai/openai-swarm-node: Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量级且符合人体工程学的方式使用 OpenAI API 编排多 Agent 系统，同时利用 Node.js 构建可扩展的现实应用。</a>: Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量级且符合人体工程学的方式使用 OpenAI API 编排多 Agent 系统...</li><li><a href="https://github.com/GGyll/condo_gpt">GitHub - GGyll/condo_gpt: 一个用于查询和分析迈阿密房地产公寓数据的智能助手。</a>: 一个用于查询和分析迈阿密房地产公寓数据的智能助手。 - GitHub - GGyll/condo_gpt: An intelligent assistant for querying and analyzing real estate condo data in Miami.</li><li><a href="https://youtu.be/k7zFH1PYaRA?si=WoH67PUBjS0W85MS"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1295025564696182898)** (1 条消息): 

> - `Swarm.js`
> - `Node.js SDK`
> - `OpenAI API`
> - `Multi-agent systems`
> - `Open-source contributions` 


- **推出用于多 Agent 魔法的 Swarm.js**：发布 **Swarm.js**，这是一个轻量级的 Node.js SDK，使用 **OpenAI API** 编排多 Agent 系统。该框架灵感来自 Python 原版的 OpenAI Swarm，能够轻松进行 Agent 管理和任务执行。
   - 凭借**简单的编排**能力和可定制的 Agent，开发者可以将 OpenAI 的强大功能无缝集成到他们的应用程序中。
- **轻松安装 Swarm.js**：要开始使用，只需运行 `npm install openai-swarm-node` 来安装 **Swarm.js**。这个轻量级 SDK 专为实际应用场景设计。
   - 该系统鼓励各级开发者为项目做出贡献，促进社区参与。
- **在 GitHub 上征集贡献**：该项目开放贡献，邀请用户查看 [GitHub Repo](https://github.com/youseai/openai-swarm-node) 以提出 Issue 并提交 Pull Request。欢迎所有开发者（无论是初学者还是专家）帮助增强该框架。
   - 鼓励积极协作和想法分享，共同在**开源**社区中构建出色的成果。



**提到的链接**：<a href="https://github.com/youseai/openai-swarm-node">GitHub - youseai/openai-swarm-node: Swarm.js is a Node.js implementation of OpenAI’s experimental Swarm framework. This SDK allows developers to orchestrate multi-agent systems using OpenAI’s API in a lightweight and ergonomic way, while leveraging Node.js for building scalable, real-world applications.</a>：Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量且符合人体工程学的方式，利用 OpenAI API 编排多 Agent 系统，同时利用 Node.js 构建可扩展的实际应用程序。

  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1295025574913376318)** (1 条消息): 

> - `Swarm.js`
> - `Multi-agent systems`
> - `OpenAI API`
> - `Node.js SDK`
> - `Contributions to open source` 


- **Swarm.js 为 Node.js 爱好者发布**：介绍 [Swarm.js](https://github.com/youseai/openai-swarm-node)，这是一个用于使用 **OpenAI API** 编排**多 Agent 系统**的轻量级 Node.js SDK。
   - 该框架灵感来自原版 Python 版本，允许开发者轻松创建和管理执行任务并相互通信的 Agent。
- **Swarm.js 核心特性揭晓**：Swarm.js 拥有 **Agent 的简单编排**、自定义指令以及为**实际应用**设计的轻量级结构等特性。
   - 其符合人体工程学的设计旨在为希望利用 OpenAI 能力的开发者增强可用性和灵活性。
- **安装与入门**：要安装 Swarm.js，只需运行 `npm install openai-swarm-node` 并开始与 OpenAI API 集成。
   - 此次发布鼓励初学者和专家在他们的项目中探索该框架的潜力。
- **欢迎社区贡献！**：项目负责人邀请用户通过查看 [GitHub repo](https://github.com/youseai/openai-swarm-node)、提出 Issue 和提交 Pull Request 来做出贡献。
   - 这种协作方式旨在丰富 SDK 并围绕 Swarm.js 培养一个充满活力的社区。



**提到的链接**：<a href="https://github.com/youseai/openai-swarm-node">GitHub - youseai/openai-swarm-node: Swarm.js is a Node.js implementation of OpenAI’s experimental Swarm framework. This SDK allows developers to orchestrate multi-agent systems using OpenAI’s API in a lightweight and ergonomic way, while leveraging Node.js for building scalable, real-world applications.</a>：Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量且符合人体工程学的方式，利用 OpenAI API 编排多 Agent 系统，同时利用 Node.js 构建可扩展的实际应用程序。

  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1294671840668815421)** (5 条消息): 

> - `bootstrap-rag v0.0.9 发布`
> - `使用 Langchain 进行 Contextual Retrieval`
> - `Swarm.js Node.js SDK` 


- **bootstrap-rag v0.0.9 正式上线！**: [bootstrap-rag v0.0.9](https://pypi.org/project/bootstrap-rag/) 的早期版本引入了对冗余测试代码的移除，以及错误修复和文档改进。
   - 主要特性包括 LangChain 集成、MLflow-evals 支持以及经过测试的 Qdrant 模板，增强了 RAG 能力。
- **探索 Contextual Retrieval**: 一段名为 [“使用 Langchain 和 OpenAI Swarm Agent 进行 Contextual Retrieval”](https://www.youtube.com/watch?v=n3nKTw83LW4) 的新 YouTube 视频指导观众如何实现 Contextual Retrieval。
   - 该视频深入探讨了如何结合 Langchain 与 OpenAI 的 Swarm Agent 进行有效的信息检索。
- **为 Node.js 推出 Swarm.js**: [Swarm.js](https://github.com/youseai/openai-swarm-node) 作为一款轻量级 Node.js SDK 发布，用于利用 OpenAI API 编排多 Agent 系统，专为 JavaScript 社区设计。
   - 该框架具有简单的多 Agent 编排功能且可定制，欢迎各种水平的开发者贡献代码。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=n3nKTw83LW4">Contextual Retrieval using Langchain and OpenAI Swarm Agent</a>: 我们将探讨如何使用 Langchain 和 OpenAI Swarm Agent 实现 Contextual Retrieval。https://github.com/githubpradeep/notebooks/blob/main/conte...</li><li><a href="https://github.com/youseai/openai-swarm-node">GitHub - youseai/openai-swarm-node: Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量级且符合人体工程学的方式使用 OpenAI API 编排多 Agent 系统，同时利用 Node.js 构建可扩展的现实应用。</a>: Swarm.js 是 OpenAI 实验性 Swarm 框架的 Node.js 实现。该 SDK 允许开发者以轻量级且符合人体工程学的方式使用 OpenAI API 编排多 Agent 系统...</li><li><a href="https://pypi.org/project/bootstrap-rag/">bootstrap-rag</a>: 无</li><li><a href="https://github.com/pavanjava/bootstrap-rag/releases/tag/v0.0.9">Release v0.0.9 · pavanjava/bootstrap-rag</a>: 更新内容 - 由 @pavanjava 在 #44 中移除了冗余的测试集生成代码；由 @pavanjava 在 #46 中修复了 Bug；由 @pavanjava 在 #47 和 #48 中进行了文档增强；Langch...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1295405364145295422)** (1 条消息): 

> - `LangGraph 教程`
> - `简历优化`
> - `面试准备工具` 


- **LangGraph 教程发布！**: 一个全新的教程展示了一个简单但强大的**双节点 LangGraph 应用**，该应用从简历和职位描述中检索数据以回答各种问题。点击[此处](https://youtu.be/7KIrBjQTGLA)观看教程。
   - *该应用可以重写简历部分内容以匹配职位描述*，并生成相关的面试问题。
- **简历和求职信功能**: 该 LangGraph 应用还可以根据提供的数据撰写针对特定职位的**求职信**，并提供面试反馈。这一功能让求职者更容易准备申请材料。
   - *对于刚开始接触 LangGraph 的人来说，这个工具特别有帮助*，因为该教程被描述为易于上手。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/)** (1 条消息): 

duh_kola: 有人在 Pretraining 期间尝试过 Instruction Data 吗？
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1294388853007843339)** (9 messages🔥): 

> - `Config 共享`
> - `Sample packing 的重要性`
> - `使用 adapter 进行模型训练`
> - `合并 adapter`
> - `使用新数据进行训练` 


- **成员讨论 Config 共享**：成员 @le_mess 请求查看 @palebluedot_32091 的 Config，后者确认了其准确性，但提到了潜在的改进空间。
   - @le_mess 表示：“*乍一看是正确的*”，并建议在 Config 中启用 **sample packing**。
- **Sample packing 的重要性**：成员 @le_mess 强调了在运行训练命令 `accelerate launch -m axolotl.cli.train config.yml` 时启用 **sample packing** 的必要性。
   - 他们还在早前的讨论中提到了 **multi-GPU** 设置和使用 **LoRA** 时的挑战。
- **使用现有 adapter 进行微调**：成员 @vortexwault 询问如何以 **Llama 3** 为基座并配合一个 adapter 来精炼微调模型，以获得更好的准确性。
   - @palebluedot_32091 建议在针对额外数据进行训练之前，先将 adapter 合并到新的基座模型中。
- **合并 adapter 的指南**：当 @vortexwault 寻求合并 adapter 的指导时，@palebluedot_32091 将其引导至一份 [GitHub 指南](https://github.com/axolotl-ai-cloud/axolotl?tab=readme-ov-file#merge-lora-to-base)。
   - 他还指出，新训练运行的 Config 应包含 `base_model: '/path/to/merged/model'` 这一行。
- **后续指导**：@vortexwault 感谢 @palebluedot_32091 的指导，并表示会查看推荐的资源。
   - @vortexwault 说：“让我检查一下，兄弟，谢谢你”，表示他打算遵循该建议。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl?tab=readme-ov-file#merge-lora-">GitHub - axolotl-ai-cloud/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. 通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl?tab=readme-ov-file#merge-lora-to-base">GitHub - axolotl-ai-cloud/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. 通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1295036626795495517)** (5 messages): 

> - `Instruct 模型`
> - `文本补全训练`
> - `模型专业化`
> - `过拟合风险` 


- **Instruct 模型通过文本补全训练获得优化**：在文本补全任务上训练 instruct 模型（例如 **GPT-3.5-Instruct**）可以显著提高其在指令任务上的表现，从而产生更准确且符合上下文的补全结果。
   - *“这可以提高生成与 Prompt 高度一致的文本的能力。”*
- **窄域训练中的过拟合风险**：如果文本补全任务的训练数据集不够多样化或规模太小，存在潜在的**过拟合**风险，导致在熟悉的数据上表现优异，但泛化能力较差。
   - 强调的一项警告是，在特定训练期间未见过的其他任务上，**性能可能会下降**。
- **对新格式和领域的适应**：训练可以帮助模型适应在其初始训练期间未接触过的新格式或指令类型。
   - 这种适应增强了模型更有效地处理更广泛输入的能力。
- **多样化训练数据集的重要性**：为了在训练 instruct 模型时获得最佳结果，使用具有代表性的多样化数据集至关重要。
   - 监控过拟合并确保对各项任务进行广泛的性能评估，对于保持泛化能力至关重要。



**提及的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e34ff1ec-0807-4a5f-9b8a-b9afcd1aeb24)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1295069120488017970)** (1 条消息): 

> - `消息格式合规性`
> - `频道规则`
> - `文档链接` 


- **消息格式合规性提醒**：提醒成员根据既定规则，遵守频道 <#1210088092782952498> 中规定的消息格式。
   - 该消息强调了遵循指南对于维持频道组织和清晰度的重要性。
- **引用频道规则**：引用了频道 <#1207731120138092628> 中概述的规则，以指导讨论中预期的行为和贡献。
   - 鼓励成员查看这些规则，以便更好地了解频道动态。
- **提供文档链接**：分享了一个链接，引导成员前往频道 <#1207731120138092628> 上的相关文档，以进一步明确预期做法。
   - 分享的链接作为增强合规性和确保频道沟通顺畅的资源。


  
---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1294616632211279872)** (3 条消息): 

> - `Aria 多模态模型`
> - `Aria 用户体验`
> - `AI 推理局限性` 


- **Aria 作为行业颠覆者登场**：由 [@rhymes_ai_](https://twitter.com/rhymes_ai_) 推出的全新多模态模型 **Aria**，目前在 🤗 Open LLM Leaderboard 上排名第一，拥有 **24.9B 参数**，能够处理图像、视频和文本输入。
   - 它拥有 **64k token 上下文窗口**，并在包括 **400B 多模态 token** 在内的多样化数据集上进行了预训练，确保了跨任务的稳定性能（[论文](https://hf.co/papers/2410.05993)，[博客](https://rhymes.ai/blog-details/aria-first-open-multimodal-native-moe-model)）。
- **用户对 Aria 的性能赞不绝口**：针对 **25.3B 多模态 Aria 模型**，一位用户称：*“这是我尝试过的最好的视觉语言模型！”*，并强调了其在图像和视频输入方面的出色能力。
   - 该模型已根据 **Apache-2.0 许可证**发布，社区也可以获得微调脚本。
- **AI 推理能力受到质疑**：一段名为 [“Apple DROPS AI BOMBSHELL: LLMS CANNOT Reason”](https://www.youtube.com/watch?v=tTG_a0KPJAc) 的 YouTube 视频探讨了围绕语言模型推理能力的批判性讨论。
   - 创作者鼓励观众为 AGI 做好准备，分享了他们的社交媒体和相关内容的链接，暗示了关于 AI 当前局限性的更广泛对话。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ailozovskaya/status/1844708897560375729?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Alina Lozovskaya (@ailozovskaya) 的推文</a>：🌸 🤗 Open LLM Leaderboard 上的首个多模态模型！rhymes-ai/Aria 是一个开源多模态原生模型，旨在视觉、语言和编程任务中提供一流的性能...</li><li><a href="https://www.youtube.com/watch?v=tTG_a0KPJAc">Apple DROPS AI BOMBSHELL: LLMS CANNOT Reason</a>：和我一起为 AGI 做好准备 - https://www.skool.com/postagiprepardness 🐤 在 Twitter 上关注我 https://twitter.com/TheAiGrid🌐 查看我的网站 - https://thea...</li><li><a href="https://fxtwitter.com/mervenoyann/status/1844356121370427546?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">merve (@mervenoyann) 的推文</a>：这是我尝试过的最好的视觉语言模型！Aria 是 @rhymes_ai_ 的新模型：一个可以接收图像/视频输入的 25.3B 多模态模型 🤩 他们以 Apache-2.0 许可证发布了该模型...
</li>
</ul>

</div>
  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1295369234662166538)** (2 条消息): 

> - `Jamba 支持线程` 


- **关于 Jamba 问题的支持咨询**：一名成员在 <#1222916247063232553> 中创建了一个关于运行 **Jamba** 时遇到问题的线程，并询问这是否是寻求支持的正确方式。
   - *Keepitirie* 做出回应，确认他们已在该频道中处理了该成员的查询，并鼓励在那里继续讨论。
- **在频道中继续讨论**：另一名成员建议，关于 Jamba 问题的讨论应保留在原始线程中，以确保清晰和连贯。
   - 他们强调了在同一频道进行后续跟进的重要性，以确保所有相关信息都易于获取。


  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1295408247389421620)** (1 messages): 

> - `Community Building`
> - `Panel Discussion on Engagement`
> - `Expert Insights`
> - `Sustainable Community Models` 


- **揭秘社区建设专题讨论**：由 Community 和 Developer Relations 专家组成的杰出小组将讨论促进社区参与和项目成功的可行策略。
   - 与会者可以从包括 **Jillian Bejtlich** 和 **Rynn Mancuso** 在内的专家那里获得见解，重点关注扩大用户群和增加贡献的技术。
- **增强你的社区技能**：该小组将提供关于如何围绕项目建立繁荣社区的战术建议，强调超越单纯编写代码的关系建立。
   - 对于项目负责人来说，这是一个提升社区建设技能并与该领域其他人建立联系的宝贵机会。
- **不要错过这次活动！**：鼓励大家在[这里](https://discord.com/events/1089876418936180786/1288598715711356928)预约参加这场关于社区建设实践的深刻专题讨论。
   - *“不要错过这个宝贵的机会！”* 频道中回荡着鼓励积极参与的声音。


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

mihai4256: 基于上述内容，我制作了这个：https://github.com/Mihaiii/backtrack_sampler
  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1295329910050521130)** (1 messages): 

> - `Multi-turn Examples Evaluation`
> - `Model Performance Metrics`
> - `Modification Restrictions on Base Handler` 


- **多轮评估过程中的挑战**：成员们讨论了多轮示例的评估方法，在模型不输出空字符串的情况下，每轮尝试 20 次才判定尝试失败。
   - 出现了一个重大问题，因为即使预测正确，模型也无法打破计数循环，导致 **~0% 的性能**。
- **临时变通方案提高了性能**：一位成员指出，修改 base_handler.py 中的代码，使其每轮仅尝试一次，可以将多轮评估的性能提高到 **~15%**。
   - 然而，他们强调**不允许修改 base_handler.py**，正在寻求替代解决方案。


  

---



---



---



{% else %}


> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}