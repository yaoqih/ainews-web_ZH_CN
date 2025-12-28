---
companies:
- google
- openai
- meta-ai-fair
- hugging-face
- cohere
date: '2024-04-10T01:05:31.512776Z'
description: '在 **Google Cloud Next** 大会上，**Gemini 1.5 Pro** 正式发布，配备了 **100 万 token
  的上下文窗口**，并在 **180 多个国家/地区**开放。其特性包括 **9.5 小时的音频理解能力**、支持近乎无限免费上传的新 **File API**，以及
  **Gecko-1b-256/768 嵌入模型**。


  同时，**GPT-4 Turbo with Vision** 已在 API 中全面开放，此次重大更新显著提升了其推理能力。**Meta Platforms**
  计划于下周推出 **Llama 3** 的较小版本。采用直接纳什优化（Direct Nash Optimization）的 **Orca 2.5 7B** 模型在
  AlpacaEval 评测中表现优于旧版 GPT-4。


  其他新发布的内容包括：增强了函数调用和代码解释能力的 **Functionary-V2.4**，以及用于图像编辑的 **CosXL** 模型。研究亮点方面，用于扩散模型的连续
  U-Net 可实现高达 **80% 的推理加速**，此外还发布了一个包含约 **5.6 万亿 token** 的海量多语言数据集。在创意应用领域，出现了利用 Gemini
  1.5 制作的无代码触摸屏游戏以及 AI 生成的小说预告片。'
id: 8ba0f2e0-044e-41dc-bf4c-46997af4535d
models:
- gemini-1.5-pro
- gpt-4-turbo
- llama-3
- orca-2.5-7b
- functionary-v2.4
- cosxl
original_slug: ainews-gemini-pro-and-gpt4t-vision-go-ga-on-the
people: []
title: Gemini Pro 和 GPT4T Vision 在同一天正式发布（GA），纯属巧合。
topics:
- million-token-context-window
- audio-processing
- file-api
- text-embedding
- function-calling
- reasoning
- direct-nash-optimization
- contrastive-learning
- code-interpreter
- diffusion-models
- neural-odes
- inference-speed
- multilingual-dataset
- image-editing
- no-code-development
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月8日至4月9日的 AI 新闻。我们为您检查了 5 个 subreddits、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)以及 **26** 个 Discord 社区（**388** 个频道和 **4154** 条消息）。预计为您节省阅读时间（按每分钟 200 字计算）：**468 分钟**。

虽是增量改进，但意义重大：

- 在 [Google Cloud Next](https://cloud.withgoogle.com/next) 大会的第一天，**拥有百万级上下文窗口的 [Gemini 1.5 Pro 结束了候补名单限制](https://x.com/OfficialLoganK/status/1777733743303696554)**，并在 180 多个国家/地区免费开放。此外，它还具备以下能力：
  - 理解长达 9.5 小时的音频（[引用](https://twitter.com/liambolling/status/1777758743637483562)：“不仅能理解你说的词句，还能理解音频背后的语气和情感。在某些情况下，它甚至能识别狗吠和雨声等声音。”）
  - 通过新的 [File API](https://ai.google.dev/tutorials/prompting_with_media) 可以“上传几乎无限的文件，而且是免费的”。
  - `Gecko-1b-256/768` 即 `text-embedding-004` 模型，这是一款新的小型 embedding 模型，在 MTEB 榜单上击败了同尺寸模型。
  - 支持 JSON mode 和更好的 function calling。
- 3 小时后，[GPT-4 Turbo with Vision 现已在 API 中全面开放（GA）](https://twitter.com/OpenAIDevs/status/1777769463258988634)，但其中隐藏了对 GPT-4 Turbo 语言模型本身的重大更新。
  - 甚至没有发布博客文章——我们只知道它得到了[重大改进](https://twitter.com/OpenAI/status/1777772582680301665)，特别是[推理能力得到了进一步提升](https://x.com/polynoamial/status/1777809000345505801)。也许它只是变得非常非常擅长“[深入探究（delving）](https://x.com/ChatGPTapp/status/1777221658807521695)”了？

在下方的回顾中，还有从 Cohere Command R 到 Google CodeGemma 的更多小型更新。

---

**目录**

[TOC] 

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能尚未实现，但即将推出。

**最新 AI 模型进展**

- **Meta Platforms 将于下周发布 Llama 3 的小型版本**：据 [TheInformation](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week) 报道，Meta 计划发布其 Llama 3 模型的小型版本。（433 个赞）
- **采用新 DNO 方法的 Orca 2.5 7B 在 AlpacaEval 中超越了旧版 GPT-4**：[Orca 2.5](https://huggingface.co/papers/2404.03715) 通过使用 Direct Nash Optimization (DNO) 将对比学习与优化通用偏好相结合，性能超过了参数量大得多的模型。（60 个赞）
- **Functionary-V2.4 发布，作为 OpenAI function calling 模型的替代方案**：与 OpenAI 模型相比，[Functionary-V2.4](https://www.reddit.com/r/LocalLLaMA/comments/1bzhyku/nanollava_1b_pocket_size_vlm/) 提供了更好的性能和新的 code-interpreter 功能。（20 个赞）
- **CosXL - Cos Stable Diffusion XL 1.0 和 1.0 Edit 模型发布**：这些模型使用 Cosine-Continuous EDM VPred 调度，以实现全色彩范围和指令式图像编辑。（9 个赞）

**高效 AI 技术**

- **[R] 高效扩散模型中缺失的 U (The Missing U)**：该[研究提出](https://www.reddit.com/r/MachineLearning/comments/1bzfns4/r_the_missing_u_for_efficient_diffusion_models/)使用神经 ODE (neural ODEs) 将离散 U-Net 替换为连续 U-Net，在保持质量的同时，使**推理速度提升高达 80%，参数减少 75%，FLOPs 减少 70%**。（38 个赞）
- **[R] 没有指数级数据就没有“零样本”（Zero-Shot）**：一项[研究发现](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/)，多模态模型需要指数级增长的预训练数据才能在“零样本”性能上获得线性提升。（12 个赞）
- **[R] 用于高性能语言技术的新型海量多语言数据集**：[HPLT 资源](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/)涵盖 75 种语言，包含 **~5.6 万亿个单词 token** 和 18 个以英语为中心的平行语言对。（12 个赞）

**创意应用**

- **新教程：使用 Stable Diffusion 掌握一致的角色面部**：一份使用 Automatic1111 生成一致角色视觉效果的[分步指南](https://www.reddit.com/gallery/1bzix80)。（597 个赞）
- **使用 Gemini pro 1.5 在无需任何代码的情况下制作的触摸屏波次射击游戏**：这款游戏是在告知 Gemini 所需功能后，在大约 [5 小时](https://v.redd.it/rujagnqzn7tc1)内完成的。（189 个赞）
- **日本科幻作家使用 AI 为其小说创作预告片**：这段 [AI 生成的预告片](https://v.redd.it/wgrbu0aindtc1)展示了一个新颖的应用案例。（20 个赞）
- **推出 Steerable Motion 1.3，通过批量图像驱动视频**：[新版本](https://www.reddit.com/r/StableDiffusion/comments/1bzakf3/introducing_steerable_motion_13_drive_videos_with/)提供了更高的细节、更流畅的动作和更好的控制。（28 个赞）

**扩展 AI 基础设施**

- **AI 公司正在耗尽互联网资源**：模型正在[消耗大量](https://lifehacker.com/tech/ai-is-running-out-of-internet)在线数据。（274 个赞）
- **[D] 巩固加拿大的 AI 优势**：加拿大正在投资 [24 亿美元](https://www.reddit.com/r/MachineLearning/comments/1bytkh8/d_securing_canadas_ai_advantage/)，以加速 AI 就业增长、提高生产力并确保负责任的发展，其中包括用于 AI 计算基础设施的 20 亿美元。（70 个赞）
- **Sam Altman 揭示 AI 的下一步计划**：Altman 在一个[图片帖子](https://www.reddit.com/r/singularity/comments/1bzjcqm/sam_altman_reveals_whats_next_for_ai/)中分享了他的愿景。（601 个赞）
- **众所周知 Sam Altman 在 OpenAI 没有股份，但初创公司投资仍让他成为了亿万富翁**：尽管没有 OpenAI 的股份，[Altman 的投资](https://www.reddit.com/r/singularity/comments/1bzjcqm/sam_altman_reveals_whats_next_for_ai/)依然获利丰厚。（33 个赞）

**负责任的 AI 发展**

- **两家日本顶尖公司称：AI 时代“社会秩序可能崩溃”**：[华尔街日报报道](https://www.wsj.com/tech/ai/social-order-could-collapse-in-ai-era-two-top-japan-companies-say-1a71cc1d)了来自日本企业的警告。（226 个赞）
- **将斥资 5000 万美元建立加拿大 AI 安全研究所**：这是加拿大 AI 投资计划的一部分，旨在[推动安全的 AI 发展](https://www.reddit.com/r/MachineLearning/comments/1bytkh8/d_securing_canadas_ai_advantage/)。（70 个赞）
- **[D] 对于那些独自发表论文的人，你们的经历是怎样的？**：关于独自发表 AI 研究的[可行性与难度的讨论](https://www.reddit.com/r/MachineLearning/comments/1bzjbpn/r_no_zeroshot_without_exponential_data/)。（55 个赞）

**梗图与幽默**

- **人类，你们的工作很安全**：一个[梗图帖子](https://www.reddit.com/r/ProgrammerHumor/comments/1bzjbpn/yours_jobs_are_safe_humans/)。（495 个赞）
- **失踪的女人**：一个[幽默图片帖子](https://www.reddit.com/r/singularity/comments/1bzjcqm/missing_woman/)。（236 个赞）
- **我也注销了 ChatGPT plus 和 Gemini Pro 的订阅——OpenAI 继续把它做得越来越烂吧，干得漂亮。**：关于 OpenAI 变化的投诉。（24 个赞）

---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**Cohere Command R+ 模型性能**

- **Command R+ 升至 Arena 排行榜第 6 位**：[@lmsysorg](https://twitter.com/lmsysorg/status/1777630133798772766) 指出 Command R+ 已升至第 6 位，通过 1.3 万多张人类投票，其水平已达到 GPT-4-0314，成为**排行榜上表现最好的开放模型**。[@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471) 强调，这甚至还没有评估 ⌘R+ 表现出色的 RAG、工具使用和多语言能力。
- **Command R+ 在金融 RAG 中击败其他模型**：[@virattt](https://twitter.com/virattt/status/1777676354596618474) 发现，在金融 RAG 评估中，使用 OpenAI embeddings、余弦相似度检索、Cohere reranking 以及 Opus 和人工评估，Command R+ 比 Claude Sonnet 速度更快且准确率高出 5%。
- **Command R+ 是一个具有先进能力的 104B 参数模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1777771141886623840) 指出，Command R+ 是一个 104B 参数的模型，具有 128K tokens 的上下文窗口，覆盖 10 种语言，支持工具使用，并专门针对 RAG 进行了微调。它是**第一个根据 Elo 评分超越 GPT-4 的开放权重模型**。

**其他值得关注的开放模型发布与更新**

- **Google 发布 Code Gemma 模型**：[@fchollet](https://twitter.com/fchollet/status/1777715491550994732) 宣布发布 CodeGemma，这是 Gemma 系列模型的新版本，专门针对代码生成和补全进行了微调，并在 **2B 和 7B 尺寸上取得了 SOTA (state-of-the-art) 结果**。[@_philschmid](https://twitter.com/_philschmid/status/1777716728921600000) 提供了更多细节，指出该模型具有 8192k 上下文，从 Gemma Base 初始化，并在 500B 额外 token（网页、代码和数学）上进行训练，经过 SFT 和 RLHF 微调，其中 2B 模型在 HumanEval 上达到 27%，7B-it 达到 52%。
- **Google 发布 Griffin 架构，性能超越 Transformer**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1777747790564589844) 分享了 Google 发布的一款采用新 Griffin 架构的模型，该模型在不同参数规模下的 MMLU 评分以及多个基准测试的平均分上，均 **优于 Transformer 基准模型**，并具有推理速度更快和内存占用更低的效率优势。
- **Google 在 Vertex AI 上发布 Gemini 1.5 Pro**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777738279137222894) 宣布 Gemini 1.5 Pro 现已在 Google Cloud 的 Vertex AI 平台上开启公开预览，它具有 **长上下文窗口**，可用于分析大量数据、构建 AI 客服 Agent 等。
- **DeepMind 在 Vertex AI 上发布 Imagen 2**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747320945234422) 宣布其生成技术 Imagen 2 现在可以 **根据单个提示词创建 4 秒的短动态图像**，并已可在 Google Cloud 的 Vertex AI 平台上使用。
- **Anthropic 推出 Constitutional AI 模型**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728366101119101) 发布了关于衡量模型说服力的新研究，开发了一种测试语言模型说服力的方法，并分析了说服力在不同版本的 Claude 中是如何扩展的。
- **Meta 宣布 MA-LMM 模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1777539936364662817) 分享了 Meta 宣布的用于长期视频理解的 MA-LMM（Memory-Augmented Large Multimodal Model），通过 **在长上下文长度下大幅减少 GPU 显存使用**，实现了更长的上下文。

**新兴趋势与讨论**

- **用于代码生成和理解的 AI**：多项讨论围绕使用 AI 进行代码生成、理解和调试展开。[@abacaj](https://twitter.com/abacaj/status/1777574208337215678) 强调了一篇论文，该论文展示了一种在不到十分钟内分别解决 67 个 GitHub issue 的方法，而开发人员平均需要花费超过 2.77 天。[@karpathy](https://twitter.com/karpathy/status/1777427944971083809) 开源了 llm.c，这是一个仅用约 1,000 行代码实现纯 C 语言训练 GPT-2 的项目。
- **AI 在编程任务中超越人类**：关于 AI 替代或增强程序员潜力的讨论有很多。[@svpino](https://twitter.com/svpino/status/1777430219785130067) 认为，虽然 AI 可以让非编程人员变成普通程序员，并帮助普通程序员变得更好，但目前可能还无法给专家级程序员提供太多帮助，他引用了编程自动化尝试的悠久历史、数据和语言本身的局限性，以及技术进步普及到大众所需的时间。
- **语言模型的 Scaling Laws（扩展定律）**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1777424149415145882) 分享了关于语言模型 Scaling Laws 最新研究的详细综述，这些定律允许 **通过成本低得多的较小规模实验，准确预测更大规模训练运行的性能**。该帖子涵盖了 Scaling Laws 如何适用于过训练 (overtrained) 模型和下游任务性能，以及如何利用它们显著降低大规模训练运行的计算成本。
- **用于语言模型程序的 DSPy**：[@lateinteraction](https://twitter.com/lateinteraction/status/1777731981884915790) 介绍了 DSPy，这是一种用于构建语言程序的方法论、编程模型和优化器集——即在系统中 **多次调用 LM 的任意控制流**。DSPy 可以优化 LM 调用的提示词和权重，以在给定指标上最大化程序质量。
- **语言模型的物理学**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1777638750740210175) 分享了一篇研究语言模型知识容量 Scaling Laws 的论文，估计模型 **每个参数可以存储 2 bit 知识**，即使量化为 int8 也是如此，这意味着一个 7B 模型可以存储 14B bit 知识，超过了英文维基百科和教科书的总和。

**迷因与幽默**

- **Anthropic function calling needs work**: [@jxnlco](https://twitter.com/jxnlco/status/1777350940502249532) 调侃道 Anthropic 的 function calling “还需要大量改进”，因为数字被作为字符串返回，而列表则是无法解析为 JSON 的字符串。
- **Perplexity paying for positive tweets**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1777359965159633389) 开玩笑说要给每个说 Perplexity 好话的人付钱，不仅是在 Twitter 上，下一步还会扩展到 Airchat。
- **Cohere beating Meta and Mistral to GPT-4 performance**: [@_lewtun](https://twitter.com/_lewtun/status/1777679834799345809) 对 Cohere 凭借一个 open weights 模型在 GPT-4 性能上击败 Meta 和 Mistral 表示惊讶，并调侃这“不在我的 LLM bingo card 上”。
- **"Majorly improved" GPT-4 launch**: [@bindureddy](https://twitter.com/bindureddy/status/1777792313315733746) 调侃 OpenAI 的“大幅改进”版 GPT-4 发布公告简单到只有一句“That's all, that's it!”，没有任何进一步细节。
- **AutoMerger creating the best 7B model**: [@maximelabonne](https://twitter.com/maximelabonne/status/1777610370925871239) 强调 AutoMerger 在 Open LLM Leaderboard 上创建了最好的 7B 模型 YamshadowExperiment28-7B，这是一个将 automerger/YamShadow-7B 和 yam-peleg/Experiment28-7B 进行简单 SLERP 合并的模型。

---

# AI Discord Recap

> 摘要之摘要的摘要

**1. New AI Model Releases and Capabilities (新 AI 模型发布与功能)**:

- Google 发布了 **[Gemini 1.1](https://huggingface.co/chat/models/google/gemma-1.1-7b-it)**，这是一个具有编程能力的改进版本，并推出了专门用于代码任务的 **[CodeGemma](https://huggingface.co/lmstudio-community?search_models=codegemma)** 模型。
- OpenAI 推出了 **[GPT-4 Turbo](https://openai.com/pricing)**，具有更大的 128k context window，知识更新至 2023 年 12 月。
- Stability AI 的 **[CosXL](https://huggingface.co/stabilityai/cosxl)** 模型在非商业研究许可证下要求共享联系方式。
- 对 **Meta 的 Llama 3** 及其潜在多模态能力的期待日益高涨，有[推测](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week)称较小版本将很快发布。

**2. Efficient LLM Training and Deployment Approaches (高效 LLM 训练与部署方法)**:

- Andrej Karpathy 介绍了 **[llm.c](https://github.com/karpathy/llm.c)**，这是一个用约 1,000 行 C/CUDA 代码实现的精简 GPT-2 训练实现。
- 围绕 **low-precision quantization**（低精度量化）技术的讨论，如用于高效 LLM 部署（尤其是移动设备上）的 **[HQQ](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808)**。
- Meta 赞助了一项 **[LLM 知识研究](https://arxiv.org/abs/2404.05405)**，涉及高达 420 万 GPU 小时的计算量。
- Groq 以 **1/10 的推理成本**服务于 75,000 名开发者，其推理能力可能与 Meta 旗鼓相当。

**3. AI Assistants and Multimodal Interactions (AI 助手与多模态交互)**:

- **[Gemini 1.5 Pro](https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww)** 因其理解音频、通过 JSON mode 执行命令以及实现多模态 AI 应用的能力而备受关注。
- [Syrax AI Telegram bot](https://t.me/SyraxAIBot) 提供了诸如吐槽、总结聊天记录和维护垃圾邮件黑名单等功能。
- 开发者构建了用于 **[虚拟试穿衣物](https://youtu.be/C94pTaKoLbU)** 和创建社交媒体帖子等任务的 AI Agent。
- 像 **[Lepton AI](https://www.lepton.ai/)** 这样的平台通过 Photon 和 WhisperX 等工具简化了 AI 应用的部署。

**4. Open-Source AI Frameworks and Community Efforts (开源 AI 框架与社区努力)**:

- LlamaIndex 展示了利用 [LlamaParse](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform) **改进检索增强生成 (RAG)** 的技术，并评估了像 [ARAGOG](https://twitter.com/llama_index/status/1777441831262818403) 这样的高级 RAG 方法。
- Mojo 编程语言 **[开源了其标准库](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)**，并提供了 [贡献指南](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md) 以鼓励社区参与。
- Hugging Face 介绍了 **[Gradio 的 API Recorder](https://x.com/abidlabs/status/1775787643324051582)**，并发布了包含超过 2600 万页内容的庞大 **[OCR 数据集](https://x.com/m_olbap/status/1775201738397765775)**，以助力文档 AI 的开发。


**5. Misc Updates (其他更新)**:

- **LLM 训练与推理的效率突破**：Andrej Karpathy 开源了 **[llm.c](https://github.com/karpathy/llm.c)**，这是一个用 1000 行 C/CUDA 实现的精简 GPT-2 训练程序，引发了关于将其移植到 GPU 以增强性能的讨论。Groq 展示了极具性价比的推理，而 **4-bit 量化** ([HQQ](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808)) 和 **FP16xINT4 内核** ([Marlin](https://github.com/IST-DASLab/marlin)) 等技术有望带来速度提升。由 Meta 赞助的一项[语言模型物理学研究](https://arxiv.org/abs/2404.05405)消耗了惊人的 420 万 GPU 小时。

- **检索增强生成 (RAG) 的进展**：RAG 的创新包括使用 **[LlamaParse](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform)** 提取文档知识图谱以增强高级工作流，以及 [ARAGOG 综述](https://twitter.com/llama_index/status/1777441831262818403)中对相关技术的全面评估。**多模态 RAG** 正被应用于[药物识别](https://twitter.com/llama_index/status/1777722765589823728)等医疗领域，而即将举行的一场活动将展示[企业级 RAG 系统](https://twitter.com/llama_index/status/1777763272701468684)。

- **架构探索与训练技术**：谷歌的 **Griffin** 等新型架构凭借额外的 10 亿参数和更高的吞吐量超越了 Transformer。**[Jet MoE](https://github.com/huggingface/transformers/pull/30005)** 集成到 Hugging Face transformers 的进展备受期待。对话模型的微调方法受到了详细审查，对比了**直接偏好优化 (DPO)** 与 **SFT+KTO** 以及微软的 **DNO** 等替代方案。根据 [PiSSA 论文](https://arxiv.org/abs/2404.02948)，研究发现**使用 SVD 初始化 LoRA 层**能显著提升微调效果。一份[近期研究](https://arxiv.org/abs/2404.04125)强调了多模态模型在零样本泛化方面的局限性。


---



# PART 1: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**API 之争**：社区内正热烈讨论 **Perplexity Pro** 与 **Claude 3 Opus** 的优劣，将 Perplexity 灵活的使用方式与 Claude 3 卓越的写作能力但受限的条件进行对比。随着工程师们热切期待能反映 **Claude 3 Opus** 性能提升的升级，围绕 **GPT-4 Turbo API** 的期待也在升温。

**预览版中的 Perplexity 实力**：**Gemini 1.5** 备受关注，它具有抗衡 **GPT-4** 的潜力，并以更大的上下文窗口和多媒体支持超出预期。与此同时，**ChatGPT Plus** 在面对免费 AI 选项时受到审视，其中 Perplexity Pro 的网页搜索功能在评论中脱颖而出。

**备受关注的助手与处理工具**：**Harpa AI** 浏览器扩展作为一种强大的网页自动化工具引起了关注，它能简化内容摘要和邮件解释等任务，从而优化工程师的工作流。

**Perplexity API 的难题与突破**：讨论涵盖了 Perplexity API 产品的方方面面，从处理**公开的 PDF/TXT 文件**、API 访问中缺少 **pplx-pro 模型**，到 API **余额充值问题**的解决。新发布的 **Perplexity API Ruby 客户端**引起了社区关注，而对 **Perplexity 专用 Token 计算工具**的咨询则反映了持续的优化努力。

**媒体、模型与多元宇宙**：在公会中分享的各种链接，从对 GPT 起源的深入探索到讨论[创建可信 AI 的工作流与工具](https://www.youtube.com/watch?v=yGejxO1xYmo)的访谈，反映了 AI 工程师广泛的兴趣范围，而 Perplexity AI 搜索往往是他们进行多样化知识探索的首选入口。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **像素艺术准备回归**：为了致敬怀旧经典，AI 讨论转向了生成 **NES 像素艺术风格**，并重点推荐了 [Civitai](https://civitai.com/models/3798/lexica-testica) 作为探索现有 AI 创作像素艺术作品的资源。
  
- **CosXL 步入聚光灯下**：Stability AI 推出了 **CosXL**，这是一个根据**非商业研究社区许可证**发布的新 AI 模型，要求用户分享联系详情，这引发了围绕访问权限和数据共享的辩论。

- **AI 艺术引发版权辩论**：用户就 **AI 生成艺术**的正当性和版权问题展开了激烈的对话，并引用了 [美国版权局 (US Copyright Office)](https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence) 最近的指南。

- **Stable Diffusion 迎来 UI 升级**：围绕 **Stability.ai** 产品的咨询引发了关于用户界面改进的讨论，用户在寻求有关 **ELLA** 等模型的功能和集成指导；例如，[LoRA UI Training Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts) 被提及作为一种简化工具。

- **社区期待 SD3 的到来**：在推测和期待中，备受瞩目的 **Stable Diffusion 3 (SD3)** 的发布引起了一阵讨论热潮，但目前尚未确定官方发布日期。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**黑客松项目创意征集**：在一次黑客松的头脑风暴会议中，参与者讨论了一些酷炫的 AI 项目，包括在学术论文上微调 Mistral，以及研究之前 Mistral 黑客松的项目。

**GPT-4 可能仍无法通过苹果测试**：关于 GPT-4 更新的聊天指出，该模型在 Temperature 设置为 0.7 时仍难以通过“苹果测试 (apple test)”，同时还有关于 IMO 数学奖项合作的邀请，暗示可能涉及高计算资源。

**回顾 Chat 模型的微调技术**：一场关于微调聊天模型的激烈辩论展开，焦点在于比较 **Direct Preference Optimization (DPO)** 与其他方法（如 **SFT+KTO** 和 Microsoft 的 **DNO**）的有效性。

**AI 革新：训练效率与模型性能**：工程师们对 Karpathy 引入的 **GPT-2 in C** 这种更精简的 LLM 训练方法感到兴奋；在 2 万亿 token 上预训练的 [StableLM 2 12B](https://huggingface.co/stabilityai/stablelm-2-12b) 也备受关注，同时人们对 **Hermes 2.5** 超越其前作的潜力充满期待。

**nanoLLaVA 出现，助力边缘效率**：针对边缘设备发布的 sub-1B 视觉语言模型 [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) 引发了讨论，人们期待它与 **Obsidian** 和 **Hermes Vision** 的集成。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU or Bust**: AI 爱好者们讨论了优化 AI 模型性能的硬件配置，特别指出 **5800x 搭配 64GB RAM** 以及 **GPU offload** 可以提高效率。从 i3 配置更换为 **14700K 搭配 96GB RAM (6400MHz)** 后，推理速度提升微乎其微，这暗示 VRAM 可能是瓶颈。

**LLM 的新实力选手**:
- **Deepseek Coder** 和 **Mistral 7B** 模型因其出色的性能潜力和对 AI 初学者的友好性而备受关注。
- **Dolphin 2.8 Mistral 7b v0.2** 是最新加入的模型，因其复杂性而受到称赞，并得到了社区驱动的支持，提供了优化的 [GGUF quantized version](https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF)。

**技术人员询问模型细节**: 出现了关于 **GGUF quantization** 和 **llama.cpp** 下载的咨询，同时探讨了 GPU 模型的兼容性，考虑了如带有外部散热的 **P40** 等选项。分享了关于 **"instruct" 模型** 和 **Mixture of Experts** 在指令准确性方面更胜一筹的见解。

**LM Studio 发布 Text-Embedding 更新**: LM Studio 0.2.19 版本正式上线，为 AI 研究人员提供了新的 Text Embedding 支持和易用性改进。提供了 [Windows](https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-2a.exe)、[Mac](https://releases.lmstudio.ai/mac/arm64/0.2.19/beta/LM-Studio-darwin-arm64-0.2.19-Preview-2a.zip) 和 [Linux](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-2a.AppImage) 的下载链接，并讨论了针对大型模型的额外 RAM 分配。

**新模型发布与社区分享**: Google 面向市场的 **Gemma 1.1** 和 **CodeGemma** 系列由社区分享，分别因其内存高效设计和指令遵循能力而引起轰动。这些模型被定位为 AI 工程师的可靠资源，可通过 [Hugging Face](https://huggingface.co/lmstudio-community?search_models=codegemma) 获取。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**智力辩论深入脑部**: 工程师们一直在讨论智力的本质，引用了“人类特有的智力”，并思考 AI（如 Claude.ai 或 GPT）是否可能具有感质 (qualia) 或意识。随着关于人类认知和进化理论的学术观点的分享，技术深度不断提升。

**GPT-5 发布引发猜测**: 对 GPT-5 发布的热切期待引发了关于所谓挑战和时间表的讨论，对比了当前用于编程辅助的选项（如 Claude 3 Opus 和 Gemini 1.5 Pro），并努力应对区域可用性问题。

**艺术算法激起伦理讨论**: 关于 AI 生成艺术与人类创造力之间伦理问题的激烈辩论浮出水面，涉及鉴赏、情感分析，以及 YouTube 的服务条款 (ToS) 可能与内容创作实践发生冲突等政策。

**利用 LLM 掌控任务？尽管提问**: 关于大语言模型 (LLMs) 简化复杂任务能力的问题不断出现，引发了关于尽管有 AI 协助但仍需要额外系统进行任务管理的对话，呼应了将 AI 能力与实际工作结构相结合的更广泛问题。

**Prompt Engineering——指路明灯还是盲点？**: 关于提示词模块化的讨论将 GPT 环境比作模块化操作系统，但也引发了对透明度和默认系统提示词更改的担忧。重点讨论了分离系统提示词的难度以及 Prompt Injection 技术的非确定性结果，并指出了管理责任和 AI 伦理问题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**RAG 改进的新路径**：**Fanghua Yu** 分享了一种利用 **LlamaParse** 提取文档知识图谱来改进 **RAG (Retrieval-Augmented Generation)** 的方法，从而增强了 RAG 的性能。在一份[全面调查](https://twitter.com/llama_index/status/1777441831262818403)中探讨了各种 RAG 技术的细节和评估，包括 **Matous Eibich 的 ARAGOG** 项目。

**RAG 识别药物**：**多模态 RAG 应用**正扩展到医疗领域，重点是药物识别，结合图像和描述来准确识别药丸，正如最近的一篇[博客文章](https://twitter.com/llama_index/status/1777722765589823728)所强调的那样。

**面向大众的 RAG**：即将举行的活动将详细介绍构建**企业级 RAG 系统**，涵盖高级解析/摄取以及确保这些系统的全面可观测性等内容。感兴趣的人员可以在[此处](https://twitter.com/llama_index/status/1777763272701468684)报名参加活动。

**OpenSearch 向量存储故障排除**：技术讨论指出在向 **OpenSearch 向量存储**插入新数据时存在问题，多位成员分享了类似经历。提供的解决方法包括使用 **index.refresh_ref_docs()**，关于文档解析的教学视频可以在[此处](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform)找到。

**寻求 Gemini 指南**：社区呼吁分享一个以 OpenAI 为蓝本、包含 **Gemini LLM** 示例的 Notebook，这体现了社区对新兴工具实用指南的渴望。现有的 OpenAI 示例备受推崇，可作为未来 **Gemini LLM** 模板的基础，相关模板可在[此处](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb)获取。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI 工程师的新酷工具**：Hugging Face 发布了具备编程能力的 **[Gemma 1.1 Instruct 7B](https://huggingface.co/chat/models/google/gemma-1.1-7b-it)**，并将计算价格降低了高达 **50%**，与 **AWS EC2** 相比平均降低了 **20%**。他们还公开了两个包含 2600 万页内容的庞大 **OCR 数据集**，并引入了 **[Gradio 的 API Recorder](https://www.gradio.app/changelog#4-26-0)** 功能。

**AI 社区的学习中心**：一个关于 **[NLP 情感分类](https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main)** 的 GitHub 仓库被分享，Hugging Face 鼓励通过各种教程和资源进行学习，例如 **Gradio** 和 **[Langchain 的教程](https://hf.co/tasks)**。

**值得关注的创意 AI 进展**：社区的创新成果包括用于病毒式内容洞察的 **BeastBot**，以及增强型角色生成和交互平台 **[Ragdoll Studio](https://ragdoll-studio.vercel.app/)**。**Deep Q-Learning** 应用正通过 **[GitHub 展示](https://github.com/SuleymanEmreErdem/deep-q-learning-applications)** 获得关注。

**AI 模型调试与优化对话**：成员们在训练 **A100 GPU** 上的 **Mistral** 时，在 **TorchScript 导出**、**scheduler/sampler** 行为以及 **OOM 错误**方面遇到了困难。社区成员建议查看分享的 **[Google Colab notebook](https://colab.research.google.com/drive/1Noqkb8Z9xzD782BjfA6oRsGtV35N_XhB)** 以寻求潜在解决方案。

**专业 AI 话题的深度讨论**：关于 **AI 硬件基准测试**、**模型和方法识别**问题、使用 **GPT-2 进行摘要生成**以及在**视觉模型中集成对比损失 (contrastive loss)** 等话题的辩论和协助，反映了社区对前沿 AI 挑战和解决方案的积极参与。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**启动品牌宣传机器！**：尽管可能引起混淆，**GPT-3.5** 一词仍被用于品牌宣传目的，掩盖了其技术渊源。

**Claude 的隐身衣**：尽管 **Claude 3 Opus** 展示了优于 **GPT-4** 的性能，但其模型大小仍然笼罩在神秘之中，目前还没有可靠的传闻或泄露。

**优化器中的平均艺术**：一场尖锐的讨论揭示了 **ScheduleFree optimizer** 并不使用指数移动平均（EMA），而是保持简单平均，这从其收敛到 *1/t* 项中可以看出。

**MoE 模型：从稠密到稀疏，再回到稠密**：一篇新[论文](https://arxiv.org/abs/2404.05567)指出，**Mixture-of-Experts (MoE) 模型** 可以进行稠密训练并进行稀疏推理，这挑战了关于它们在扩展时参数效率的普遍观点。

**Token 采样策略辩论**：在 **min-P vs. top-P** 采样中，由于概率变化平缓，min-P 可能更有效。这一观点得到了 [VLLM GitHub repo](https://github.com/vllm-project/vllm/blob/b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb/vllm/model_executor/layers/sampler.py#L161) 中 Token 分布分析的支持。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain 图结构的奥秘揭晓**：工程师们澄清了 LangChain 的 `CompiledGraph` 类中存在 `attach_edge` 方法，并引导成员查看[官方文档](https://python.langchain.com/docs/langgraph#add_edge)以了解其功能。

**AI 转录术语解析**：在构建 AI 转录应用的过程中，引发了关于 **SerpAPI** 和表面上相似的 **Serper API** 的讨论。社区成员对 Serper API 与 LangChain 的协同作用仍不确定，它与集成良好的 SerpAPI 不同。

**模型对决：成本 vs 能力**：LLM (Large Language Models) 爱好者分享了操作经验，比较了 **GPT-3.5**、**GPT-4** 和 **Claude** 的实力，同时也表达了在使用 **gemin8** 等模型进行实际部署和经济性方面的困扰。

**自定义检索与错误处理**：对**自定义检索系统**的参与引发了关于性能评估的交流，引导新手使用 [trulens eval](https://python.langchain.com/docs/integrations/document_loaders/youtube_audio/) 包，而关于 LangChain 错误管理的问题则通过参考 [Pydantic validators](https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start#get-started) 和 `RunnableRetry` 得到了解答。

**LangChain vs. OpenAI**：关于 LangChain 对 AI Agent 的效用与定制化 **OpenAI's APIs** 之间的比较引发了难题。然而，讨论未能总结出 LangChain 相对于 OpenAI 产品的决定性优势。

**艺术与网络安全融合**：DIY 开发者在审美和安全领域开发了大量工具。Artful AI 现在拥有**新的图像模型**（[Artful AI](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai)），AISploit 为渗透测试人员赋能（[AISploit GitHub](https://github.com/hupe1980/aisploit)），Galaxy AI 使优质 AI 模型民主化（[Galaxy API](https://galaxyapi.onrender.com)），TinderGPT 简化了约会软件的聊天（[TinderGPT GitHub](https://github.com/GregorD1A1/TinderGPT)），而 everything-rag 则作为一个本地聊天机器人助手推出（[everything-rag GitHub](https://github.com/AstraBert/everything-rag)）。

**AI 造型师与发布助手**：一个教程展示了能够为社交媒体图像进行时尚着装的 AI（[YouTube Guide](https://youtu.be/C94pTaKoLbU)），另一位工程师询问了关于 AI Agent 发布教程的信息，寻求为其创作赋予用户界面。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Meta 赞助海量 GPU 小时**：Meta 赞助了一项关于 [LLM 知识的研究](https://arxiv.org/abs/2404.05405)，涉及惊人的 420 万 GPU 小时，相当于约 479 年的不间断计算，展示了该项目的规模和资源强度。

**GPT-2 披上 CUDA 战袍**：有传闻称 [GPT-2 训练代码](https://github.com/karpathy/llm.c/tree/master/dev/cuda) 已移植到 CUDA，这可能预示着新的效率和性能里程碑。对话显示，一个日益壮大的工作组正渴望探索这一 CUDA 适配。

**Triton 中的优化机会**：讨论围绕利用 **triton-viz** 增强程序可视化和解决文档难题展开，特别是通过 [GitHub pull request #3608](https://github.com/openai/triton/pull/3608/files) 贡献官方参考。

**LIAH 加入 LLM 欺骗游戏**：关于 Ring Attention 架构有用性的辩论兴起，特别是一位成员介绍了 LIAH (**lie-in-a-haystack**)，这是一种旨在阻止语言模型依赖现有知识的策略，可通过其 [GitHub 仓库](https://github.com/melvinebenezer/Liah-Lie_in_a_haystack) 访问。

**LLM 中的量化困境**：LLM 量化的挑战引发了关于应用和推理潜在性能收益的讨论，特别关注 4-bit 量化技术以及在移动端 LLM 部署应用中使用 **HQQ**，如分享的 [HQQ 代码](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808) 所示。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**语音识别平滑了 Python 的坎坷**：通过将 Python 从 **3.11.4 降级到 3.10**，找到了机器人语音识别问题的实际解决方案，这与社区认为 Python **3.9 和 3.10** 在兼容性方面更佳的见解一致。

**01 在 Windows 上的烦恼与 Linux 的领先**：一位成员在 Windows 上安装 **01** 时遇到困难，特别是 API Key 问题，导致建议检查环境变量命名（使用 **OPENAI_API_KEY**），而 Linux 用户报告的问题较少。

**GPT-4 大放异彩**：GPT-4 的发布因其**改进的性能和视觉能力**在社区中引起轰动，讨论重点在于其在 [OpenAI Platform](https://platform.openai.com/docs/models/continuous-model-upgrades) 上的集成。

**DIY 技术爱好者为 OpenInterpreter 做准备**：讨论深入探讨了 OpenInterpreter 的 DIY 与预订选项，强调 M5 Atom Echo 是关键组件；其定制软件针对 M5 进行了最佳优化，可从 Mouser.com 等供应商处获得。

**桌面机器人梦想与 Raspberry Pi 方案**：出现了关于将 Raspberry Pi 用于 01 项目的对话，雄心壮志从桌面机器人到开源贡献不等，并打算利用域名 cofounder.bot 进行未来开发。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Jet MoE 等待启程前往 Hugging Face**：**Jet MoE** 集成到 **Hugging Face 的 transformers** 中备受期待，[待处理的 GitHub pull request](https://github.com/huggingface/transformers/pull/30005) 证明了这一点。多位用户正密切关注该 PR，讨论强调了该架构所蕴含的潜力。

**Lepton AI 以简洁起飞**：用户友好且云原生的平台 **Lepton AI** 因其运行 AI 应用的简便性而受到赞誉，**Photon** 和 **WhisperX** 等工具受到关注；可在 [Lepton AI 官网](https://www.lepton.ai) 进一步探索该平台。

**AI 巨头展示算力**：**Qwen 1.5 32B**、**Yi 34B** 和 **Command R** 三款模型同台竞技，引发了关于它们在性能和能力方面的比较讨论，特别是在上下文处理和数据集表现方面。

**Meta 为 Llama 3 的首次亮相做准备**：围绕 **Meta 即将推出的 Llama 3** 的讨论非常热烈，特别是其预期的多模态能力以及参数量的不确定性。推测与 [The Information](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week) 关于下周推出较小版本 Llama 3 的报道一致。

**SVD 增强 LoRA**：社区的一个亮点是 [CFGeek 分享的发现](https://x.com/cfgeek/status/1777556286047166673?s=46&t=hIokEbug9Pr72tQFuXVULA)，表明通过使用 **SVD 初始化 LoRA 层**可以改善微调结果。该方法的完整描述可在 [PiSSA GitHub 仓库](https://github.com/GraphPKU/PiSSA) 和专门的 [arXiv 论文](https://arxiv.org/abs/2404.02948) 中找到。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Pro 1.5 成为焦点**：拥有 1M token 上下文的新型 [Gemini Pro 1.5](https://openrouter.ai/models/google/gemini-pro-1.5) 和 [GPT-4 Turbo](https://openrouter.ai/models/openai/gpt-4-turbo) 的视觉能力引发了对其在 Large Language Model (LLM) 社区影响的多样化预测。关于其性能的看法存在明显分歧，特别是在将数据从 PDF 导出为 JSON 方面。

- **Logit Bias 微调优势**：多个模型现在增强了对 `logit_bias` 参数的支持，使工程师能够更精细地控制输出 token 的概率，受益模型包括 [Nous-Hermes-2-Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo) 和 [Llama](https://openrouter.ai/models/meta-llama/llama-2-13b-chat)。

- **模型下架计划**：OpenRouter 正在战略性地清理表现不佳的模型，包括 *jebcarter/Psyfighter-13B* 和 *jondurbin/bagel-34b-v0.2*，流量很快将重定向到利用率更高的替代方案，如 *xwin-lm/xwin-lm-70b*。

- **强大的 Telegram Bot 部署**：**Telegram** 上的 [Syrax AI bot](https://t.me/SyraxAIBot) 通过吐槽（roasting）、总结庞大的聊天记录以及用于打击垃圾信息的全球黑名单功能，促进了用户参与。

- **角色扮演动态与审查担忧**：工程师们正在讨论影响角色扮演场景的模型限制，在角色扮演响应质量方面，明显更倾向于 **Command-R** 而非 **Command-R+**。此外，对审查和模型过滤的担忧也表明，用户对适合 RP 及相关用例的更开放模型有需求。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Groq 展现推理实力**：Groq 平台现在的**推理成本仅为 1/10**，拥有快速增长的 **75,000 名开发者**社区，显示出强劲的市场吸引力。据报道，由于该公司令人印象深刻的性能成就，NVIDIA 工程师们也感到了压力。
  
- **Gemini 1.1 和 GPT-4 Turbo 引发关注**：**Gemini 1.1** 的发布引起了轰动，而 **GPT-4 Turbo** 则带来了 128k 上下文窗口和更新至 2023 年 12 月的知识库；两者都提供了更好的易用性和功能，激发了人们对 AI 进展的兴奋 [OpenAI 价格页面](https://openai.com/pricing)。
  
- **Karpathy 摆脱 Python**：Andrej Karpathy 用 C 语言开发了一个简洁的 GPT-2 实现——**llm.c**。这是一项旨在实现高效、语言简化的 AI 训练尝试，代码量仅约 **1,000 行** [GitHub 上的 llm.c](https://github.com/karpathy/llm.c)。
  
- **与 AI 对话进入新境界**：讨论揭示了一个未来，AI Agent 不仅能理解而且能根据音频提示采取行动，正如新的 **Gemini 1.5 Pro** 所展示的那样，扩展了实时 AI 交互和开发者的可能性空间。

- **为 AI 愿景家提供的资源**：AI 社区正在获得广泛的资源支持，例如 **Supabase 用于向量相似度搜索的 pgvector**，以及像 **turbopuffer 这样可扩展的向量数据库**平台，为具有成本效益的大规模机器学习应用铺平了道路。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **期待 Modular (Mojo 🔥) 中的 f-strings**：工程师们正期待在 **Mojo** 中引入 `f` 字符串功能，以增强该语言中 Python 风格的字符串格式化。同时，提出了一种使用 C-style 格式化的临时替代方案，但提醒其未来可能会被弃用。

- **随时随地利用 Mojo 文档**：讨论强调目前 **Mojo** 尚无本地文档命令，并对比了在线文档与本地仓库文档的质量，建议用户访问在线 [Mojo 标准库模块](https://docs.modular.com/mojo/lib) 以获取结构最清晰的信息。

- **开源号召 Mojo 高手**：随着 **Mojo** 标准库的开源，官方分享了一份[公告](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)和[分步指南](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide)，供有意贡献的人员参考，此外还有详细的 [GitHub 贡献指南](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md)。

- **Karpathy 精简 LLM 训练**：AI 先驱们关注到 Andrej Karpathy 发布了一个极简的 [GitHub 仓库](https://github.com/karpathy/llm.c)，仅用 1000 行 C/CUDA 代码就实现了一个 **GPT-2 风格的模型训练代码库**，因其简洁性和潜力引发了广泛兴趣。

- **Mojo Nightly 版本持续发力**：新的 **Mojo nightly 构建** 带来了更新和变化，可在其 [更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中查看。社区正在协助提交 [此类错误报告](https://github.com/modularml/mojo/issues/2252)，旨在实现工作日自动发布，并庆祝引入了诸如异构变长泛型 (heterogeneous variadic generics) 等特性。

- **星号图案与隐藏技巧**：社区贡献的 YouTube 视频 [“如何在 Python 中打印任何星号图案”](https://youtu.be/6cyCeJwgNjc) 以生动且具教育意义的形式展示了 **Mojo** 的能力，模糊了 Python 教程与 Mojo 插件特性之间的界限。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **受合同约束的 CosXL 创作**：Stability AI 推出了 [**CosXL**](https://huggingface.co/stabilityai/cosxl)，这是一个采用非商业研究社区许可协议的新模型，该协议要求共享用户联系详情，引发了关于数据隐私和使用伦理的辩论。

- **从非 Stable Diffusion 数据库的文本中提取像素**：工程师们分享了从不在 Stable Diffusion 数据库中的文本生成图像的方法，并参考了 Diffusers 的文档，同时提到了软件已更新至 0.27.2 版本。

- **AI 角色重塑自由职业领域**：一篇针对 Upwork 上 500 万个自由职业岗位的[博客文章分析](https://bloomberry.com/i-analyzed-5m-freelancing-jobs-to-see-what-jobs-are-being-replaced-by-ai/) 深入探讨了 AI 如何影响岗位流失，这一话题对于探索自由职业机会的工程师至关重要。

- **模型训练趋势**：讨论围绕模型训练中 EDM 调度和 offset noise 的效率展开，表明从业者在最佳实践上持有不同立场。

- **Griffin 超越 Transformer**：据报道，Google 新的 **Griffin 架构** 性能超越了 Transformer，引入了额外的 10 亿参数并增强了吞吐量，这可能预示着架构设计选择的转变。



---

# PART 2: 渠道详细摘要与链接



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1226803866046173287)** (645 条消息 🔥🔥🔥): 

_

- **Perplexity Pro 与 Claude 3 Opus 之争**：用户正在比较 **Perplexity Pro 订阅**与 **Claude 3 Opus** 的价值。Perplexity 因其灵活的使用场景和没有明显的短消息限制而受到青睐，而 Claude 3 Opus 则因更好的写作质量受到称赞，但存在使用限制。
- **GPT-4 Turbo API 期待**：一位用户询问 **Perplexity 是否会更新**以适配新的 **GPT-4 Turbo API**，表达了对提升模型性能和速度（类似于 **Claude 3 Opus**）的兴趣。
- **Gemini 1.5 的好奇与期待**：围绕 **Gemini 1.5** 的性能展开了讨论，据报道其性能可与 **GPT-4** 媲美，并拥有显著更大的上下文窗口（context window），支持音频和视频，目前可在 Google 的 AI 实验室进行预览。
- **ChatGPT Plus 与免费 AI 模型**：用户讨论了 **ChatGPT Plus** 的优缺点及其与 **GPT-3.5**、**Gemini** 和 **Claude** 等其他免费 AI 模型相比在创意方面的局限性。Perplexity Pro 因其网页搜索集成而保持吸引力，可能足以满足常见的使用场景。
- **Harpa AI 工具聚焦**：一位用户分享了 **Harpa AI** 的优势，这是一款将 OpenAI 模型与网页自动化集成的浏览器扩展，强调了它在无需手动复制粘贴的情况下总结和解释网页内容及电子邮件方面的实用性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/borat-king-king-in-the-castle-gif-12965265">Borat King GIF - Borat King King In The Castle - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/roger-scott-wealthpress-stocks-roger-scott-wealthpress-wealthpress-roger-scott-gif-23073645">Roger Scott Wealthpress GIF - Roger Scott Wealthpress Stocks - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://poe.com/">Poe - 快速、实用的 AI 聊天</a>: 未找到描述</li><li><a href="https://tenor.com/view/queen-freddie-mercury-we-are-the-champions-champion-sing-gif-4654136">Queen - Champion GIF - Queen Freddie Mercury We Are The Champions - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/success-great-job-nice-great-success-great-gif-5586706">Success Great Job GIF - Success Great Job Nice - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.anthropic.com/claude/docs/long-context-window-tips">长上下文窗口技巧</a>: 未找到描述</li><li><a href="https://app.wordware.ai/r/7df30863-5cf2-4a81-8563-f24c279a77bb">Wordware - PyWiz</a>: 未找到描述</li><li><a href="https://harpa.ai/">HARPA AI | GPT Chrome 自动化 Copilot</a>: 用于 AI 驱动的网页自动化的 Chrome 扩展：适用于 Google 搜索的 ChatGPT、ChatGPT 写作助手、总结、重写、提取和监控网页、价格及数据。</li><li><a href="https://www.wolframalpha.com/input?i=2x%5E3+%2B+3x%5E2+-+5x+%2B+7+%3D+0">2x^3 + 3x^2 - 5x + 7 = 0 - Wolfram|Alpha</a>: Wolfram|Alpha 为最广泛的人群（涵盖所有职业和教育水平）提供专家级的知识和能力。</li><li><a href="https://chromewebstore.google.com/detail/harpa-ai-automation-agent/eanggfilgoajaocelnaflolkadkeghjp">HARPA AI | 带有 Claude & GPT 的自动化 Agent</a>: 适用于 Chrome 的 AI Agent。在任何网站上使用的 ChatGPT Plus / GPT-4 copilot。利用 AI 在网站上进行自动化、搜索、总结、翻译和写作。</li><li><a href="https://www.youtube.com/watch?v=gCkZmADecL0">日食现场直播（含视频和更新）</a>: 加入我们的日食现场直播，包含现场日食视频！我们将为您展示墨西哥、美国和 C... 的日全食现场。</li><li><a href="https://docs.perplexity.ai/discuss/65d956e39db34f001ff8ce0a">Sonar 模型是新的吗？</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/6582f98b41714c00723d5d5c">PPL 网站上的模型与 API 模型之间的区别。</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/6601ffd6bd5f0e0045ac5d16">模型名称？</a>: 未找到描述</li><li><a href="https://www.star.nesdis.noaa.gov/GOES/conus_band.php?sat=G16&band=GEOCOLOR&length=24">GOES-East CONUS - GeoColor - NOAA / NESDIS / STAR</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1226793990821646336)** (18 条消息🔥): 

- **探索 GPT 的解剖结构**：分享了一个讨论特定 GPT 模型起源的链接，这可能为深入了解其基础组件和架构提供见解。如需详细探索，请访问 [The origins of GPT](https://www.perplexity.ai/search/The-origins-of-GpIJuYMlT.Gl4VphTZUzlQ#0)。

- **YouTube 关于可信 AI 的见解**：一段由 Clara Shih 采访 Perplexity AI、LlamaIndex 等公司创始人的 YouTube 视频，重点讨论了构建可信 AI 的工作流和工具。在 [Workflows & Tooling to Create Trusted AI](https://www.youtube.com/watch?v=yGejxO1xYmo) 探索这些观点。

- **从道家到太空**：用户分享了多个 Perplexity AI 搜索链接，探讨了从道家哲学到 SpaceX 火星计划的广泛话题。搜索内容涵盖了关于 AI 训练、尼采哲学以及多元宇宙理论的查询。

- **AI 与创意的碰撞**：讨论内容包括关于 Jony Ive 的影响以及 AI 在翻译服务中集成的链接。社区参与了关于 AI 适应创意领域潜力的讨论。

- **在 Perplexity AI 上寻求答案**：用户正积极分享 Perplexity AI 的搜索链接，似乎在寻求各种主题的信息或答案，这表明该平台被用于多样化且深入的查询。

**提到的链接**：<a href="https://www.youtube.com/watch?v=yGejxO1xYmo">Workflows &amp; Tooling to Create Trusted AI | Ask More of AI with Clara Shih</a>：Clara 与三家最热门 AI 公司的创始人/CEO 坐下来交谈——Aravind Srinivas (Perplexity AI)、Jerry Liu (LlamaIndex) 和 Harrison Chase (La...

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1226889273341382698)** (6 messages): 

- **Perplexity API 与公开文件**：一名成员询问关于使用 **Perplexity** 处理已上传到公开 URL 的 **PDF 或 TXT 文件** 的事宜。
- **关于通过 API 使用 pplx-pro 模型的咨询**：另一名成员询问 **pplx-pro 模型** 是否可以通过 API 使用，类似于当前的 Crawler 用法。他们被告知 **Pro 搜索无法通过 API 访问**，仅能通过 Web 和 App 使用。
- **Perplexity API 的新 Ruby 客户端发布**：一名成员宣布推出了他们的 **Perplexity API Ruby client**，并引用了另一个频道中的帖子以获取详细信息。
- **Perplexity API 余额充值问题已解决**：发布了一项公告，告知用户 **API 余额充值问题** 已解决，并邀请用户在问题持续存在时通过私信寻求进一步帮助。
- **关于 Perplexity 模型 Token 计算工具的咨询**：一名成员询问是否有类似于 *tiktoken* 的工具，专门用于计算 **Perplexity 模型** 的 Token。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1226790508060213248)** (470 messages🔥🔥🔥): 

- **像素艺术风格生成**：用户讨论了使用 AI 创建 **NES 像素艺术风格**，并推荐使用 Civitai 平台搜索现有作品。
- **Stability AI 的新 CosXL 模型**：Stability AI 发布了一个新模型 **CosXL**；然而，用户需要同意分享其联系信息才能访问，因为它属于**非商业研究社区许可证**。
- **Stable Diffusion 使用咨询**：多位用户寻求关于如何使用 **Stability.ai** 产品生成图像的帮助，具体问题涉及 **UI 功能**、**覆盖设置 (override settings)**、**Bot 功能**、**Finetuning** 以及集成额外模型（如 **ELLA** 和 **SDXL Inpainting**）。
- **关于 AI 生成艺术的公开对话**：针对 AI 生成艺术的性质、其合法性、署名权和版权问题进行了详细对话，多位用户提供了见解，并引用了**美国版权局 (US Copyright Office)** 的声明。
- **对 SD3 的期待**：用户热烈讨论了 **Stable Diffusion 3 (SD3)** 的发布日期，分享了关于时间范围的各种信息，但目前尚未确认确切的发布日期。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://var.vision/demo">Template</a>: 未找到描述</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/frieren-wow-elf-peek-a-boo-gif-12265100463579712545">Frieren Wow GIF - Frieren Wow Elf - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://soundcloud.com/4dreamsy/blondies-and-weed">Blondies and weed</a>: 在 #SoundCloud 上收听 4dreamsy 的 Blondies and weed #np</li><li><a href="https://civitai.com/models/3798/lexica-testica">Lexica Testica - 1.0 | Stable Diffusion Checkpoint | Civitai</a>: 初始化自 OpenJourney v2，在从 Lexica art 首页（2023年1月）抓取的图像上进一步微调了 4000 步。擅长生成...</li><li><a href="https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence>">Federal Register :: Request Access</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/en/training/dreambooth?gpu-select=16GB">DreamBooth</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/en/training/sdxl">Stable Diffusion XL</a>: 未找到描述</li><li><a href="https://github.com/TencentQQGYLab/ELLA">GitHub - TencentQQGYLab/ELLA: ELLA: 为 Diffusion Models 配备 LLM 以增强语义对齐</a>: ELLA: 为 Diffusion Models 配备 LLM 以增强语义对齐 - TencentQQGYLab/ELLA</li><li><a href="https://github.com/derrian-distro/LoRA_Easy_Training_Scripts">GitHub - derrian-distro/LoRA_Easy_Training_Scripts: 一个使用 Pyside6 制作的 UI，旨在简化在 sd-scripts 中训练 LoRA/LoCon 及其他 LoRA 类型模型的过程</a>: 一个使用 Pyside6 制作的 UI，旨在简化在 sd-scripts 中训练 LoRA/LoCon 及其他 LoRA 类型模型的过程 - derrian-distro/LoRA_Easy_Training_Scripts</li><li><a href="https://github.com/ckkelvinchan/RealBasicVSR">GitHub - ckkelvinchan/RealBasicVSR: "Investigating Tradeoffs in Real-World Video Super-Resolution" 的官方仓库</a>: "Investigating Tradeoffs in Real-World Video Super-Resolution" 的官方仓库 - ckkelvinchan/RealBasicVSR</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/release_candidate/CHANGELOG.md">stable-diffusion-webui/CHANGELOG.md at release_candidate · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://civitai.com/models/139562?modelVersionId=361593">RealVisXL V4.0 - V4.0 Lightning (BakedVAE) | Stable Diffusion Checkpoint | Civitai</a>: 使用 Turbo 模型配合 DPM++ SDE Karras 采样器，4-10 步，CFG Scale 1-2.5。使用 Lightning 模型配合 DPM++ SDE Karras / DPM++ SDE 采样器，4-6 步...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#stableswarmui">GitHub - Stability-AI/StableSwarmUI: StableSwarmUI，一个模块化的 Stable Diffusion Web 用户界面，重点在于使强力工具易于访问、高性能和可扩展性。</a>: StableSwarmUI，一个模块化的 Stable Diffusion Web 用户界面，重点在于使强力工具易于访问、高性能和可扩展性。 - Stability-AI/StableSwarmUI
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1227092878681968713)** (2 条消息): 

- **神秘的握手表情符号**：一位成员分享了一个 **Twitter 链接**，并紧接着发了一个 🤝 表情符号，可能表示合作伙伴关系、达成协议，或者仅仅是对推文内容的认可。未提供额外的上下文或讨论点。
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1226860087105032233)** (9 条消息🔥):

- **探索 Reinforcement Learning 中的 AI 趋势**：一段 [YouTube 视频](https://www.youtube.com/watch?v=MBo6SIIhTIY&ab_channel=TheTWIMLAIPodcastwithSamCharrington) 介绍了 Nvidia 的 Kamyar Azizzadenesheli 讨论的 **LLM 时代的 Reinforcement Learning**，这是 AI Trends 2024 系列的一部分。
- **StableLM 2 12B 发布**：[Stability AI 的 Stable LM 2 12B](https://huggingface.co/stabilityai/stablelm-2-12b) 是一个拥有 121 亿参数的语言模型，已在涵盖多语言和代码数据集的 2 万亿 token 上进行了预训练。
- **对规模化训练的期待**：一位成员对 **StableLM 2 12B** 表示乐观，希望 Stability AI 已经有效地扩大了小模型的训练规模。
- **使用 Karpathy 的 llm.c 进行更精简的训练**：Andrej Karpathy 介绍了一种更高效的 LLM 训练方法，即通过 **C 语言实现的 GPT-2** 来最小化依赖项，详情见 [GitHub](https://github.com/karpathy/llm.c) 和 [Twitter](https://twitter.com/karpathy/status/1777427944971083809?s=46)。
- **关于 Chat 模型微调方法的辩论**：成员们正在讨论 Chat 模型的微调技术，将 [StableLM 2 12B Chat](https://huggingface.co/stabilityai/stablelm-2-12b-chat) 中使用的 **Direct Preference Optimization (DPO)** 与 SFT+KTO 以及 Microsoft 的 DNO（如在 Orca 2.5 中实现的）等替代方法进行比较。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stablelm-2-12b">stabilityai/stablelm-2-12b · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1777427944971083809?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？没有？好吧，现在你可以了！使用 llm.c：https://github.com/karpathy/llm.c 首先，实现了 GPT-2 的训练...</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=MBo6SIIhTIY&ab_channel=TheTWIMLAIPodcastwithSamCharrington">AI Trends 2024：LLM 时代的 Reinforcement Learning，与 Kamyar Azizzadenesheli 对谈 - 670</a>：今天我们邀请到了 Nvidia 的资深研究员 Kamyar Azizzadenesheli，继续我们的 AI Trends 2024 系列。在对话中，Kamyar 向我们更新了...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1226803521542684772)** (175 条消息🔥🔥): 

- **Opus JSON 转 XML 的讨论**：成员们讨论了将 Python 函数转换为供 Opus 使用的问题，涉及 JSON 函数签名及其在 XML 下的训练。有人提到了 Anthropic 现有的 `construct_format_tool_for_claude_prompt` 函数，认为其具有潜在用途。[查看 Anthropic 的函数](https://github.com/anthropics/anthropic-tools/blob/a1a2f02d4309b219e34d2a33664003fd49ad7921/tool_use_package/prompt_constructors.py#L68)。

- **征集黑客松项目创意**：一位黑客松组织者正在寻求酷炫的 AI 项目创意，收到的建议包括探索 Mistral 黑客松的项目（如 Codex），以及[其他值得关注的贡献](https://x.com/alexreibman/status/1772167054532952165?s=46)，例如在学术论文上微调 Mistral。

- **Nous AI 研究工具与许可咨询**：成员们讨论了 Nous Research 从零开始创建专有模型的可行性，以及在 VRAM 限制下使用 MoE 模型的可操作性。有人针对正在构建的工具提出了许可问题，该工具将免费提供，但频繁使用需要付费，类似于“请我喝杯咖啡”的模式。

- **将生成式 UI 与 Nous 模型集成的尝试**：有一场关于名为 morph 的[开源生成式 UI 搜索引擎](https://github.com/Fus3n/TwoAI)的对话，该引擎是使用 Vercel AI SDK 构建的。成员们讨论了利用其 function-calling 和 JSON 模式功能将其与 NousResearch/Hermes-2-Pro-Mistral-7B 集成的潜力。

- **Gaudi 3 和 GPT-4 更新的闲聊**：有人询问了关于 Gaudi 3 的个人使用体验，并提到了 GPT-4 的一次更新，该更新在 0.7 的 temperature 下仍然无法通过“苹果测试”。此外，一位成员分享了关于打算参加 IMO 数学竞赛的信息，并寻求合作或算力支持。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alexreibman/status/1772167054532952165?s=46">来自 Alex Reibman 🖇️ (@AlexReibman) 的推文</a>：7/ Codex 在 5 万篇研究论文上对 Mistral 7B 进行 Fine tuning，用于无监督学习并基于上下文主题生成新颖的学术论文 🥈 Fine tuning 赛道第二名</li><li><a href="https://arxiv.org/abs/1308.3432">通过随机神经元估算或传播梯度以进行条件计算</a>：随机神经元和硬非线性在深度学习模型中因多种原因而有用，但在许多情况下它们提出了一个具有挑战性的问题：如何估算损失函数的梯度...</li><li><a href="https://outlines-dev.github.io/outlines/reference/json/">JSON (function calling) - Outlines 〰️</a>：使用 LLM 进行结构化文本生成</li><li><a href="https://sdk.vercel.ai/docs/concepts/ai-rsc">Generative UI - Vercel AI SDK</a>：一个用于构建 AI 驱动的用户界面的开源库。</li><li><a href="https://x.com/miiura/status/1777350693596139546">来自 Yoshiki Miura (@miiura) 的推文</a>：介绍 morph：一个完全开源、AI 驱动且具有 Generative UI 的问答引擎。使用 @vercel AI SDK 构建，提供出色的流式传输结果。 👇 更多详情</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/anthropics/anthropic-tools/blob/a1a2f02d4309b219e34d2a33664003fd49ad7921/tool_use_package/prompt_constructors.py#L68">anthropic-tools/tool_use_package/prompt_constructors.py at a1a2f02d4309b219e34d2a33664003fd49ad7921 · anthropics/anthropic-tools</a>：通过在 GitHub 上创建账号来为 anthropics/anthropic-tools 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/huggingface/peft/pull/1626">由 fxmeng 提交的 Pull Request #1626 · huggingface/peft：将 PiSSA 作为 LoRA 的可选初始化方法</a>：在论文 "https://arxiv.org/pdf/2404.02948.pdf" 中，我们介绍了一种参数高效微调 (PEFT) 方法，主奇异值和奇异向量自适应 (PiSSA)，它优化了...</li><li><a href="https://huggingface.co/Weyaxi/Einstein-v6-7B">Weyaxi/Einstein-v6-7B · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1777278492050075821">来自 Weyaxi (@Weyaxi) 的推文</a>：🥳 见见 Einstein 模型的第 6 版，基于新的 Mistral v0.2 模型，这是一个使用多样化、高质量且经过过滤的开源数据集进行监督微调的模型！🚀 📊 该模型现在具有 ...</li><li><a href="https://nostalgebraist.tumblr.com/post/741247180226052096/i-dont-think-youre-drawing-the-right-lesson-from">树是哈利奎因，词语也是哈利奎因</a>：我不认为你从 Transformer 模型的广泛成功中吸取了正确的教训。你写道：如果你必须用一句话总结过去十年的 AI 研究，你可能会说...</li><li><a href="https://linktones.synthtrails.com/linktone/kanye">SynthTrails</a>：未找到描述</li><li><a href="https://github.com/Fus3n/TwoAI">GitHub - Fus3n/TwoAI: 让两个本地 LLM 就任何话题进行对话的一个简单实验！</a>：让两个本地 LLM 就任何话题进行对话的一个简单实验！- Fus3n/TwoAI</li><li><a href="https://github.com/miurla/morphic.git">GitHub - miurla/morphic: 一个具有 Generative UI 的 AI 驱动问答引擎</a>：一个具有 Generative UI 的 AI 驱动问答引擎。通过在 GitHub 上创建账号来为 miurla/morphic 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1226868590255411210)** (20 条消息🔥): 

- **对 Hermes 训练结果感到兴奋**：一位成员对 **Hermes 2-Pro-Mistral-7B** 在最新[链接](https://arxiv.org/abs/2257.15746)中表现良好的潜力表示兴奋。
- **对 Hermes 性能感到好奇**：在研究了 Hermes 2 的潜力后，一位成员推测了 "Hermes-2.5" 的可能性。
- **对 Hermes 2.5 性能的热情**：一位成员表达了他们对 Hermes 2.5 在各种性能 Benchmark 中超越以往模型的潜力的热情。
- **期待 Hermes 2.5 发布**：社区对即将到来的发布以及这种先进工具在数据和分析领域的影响表示渴望。
- **为 Hermes 3 做准备**：随着预期的 "Hermes 3" 发布，对这一先进平台的期待引发了 AI 领域的兴趣和好奇。

**提到的链接**：<a href="https://arxiv.org/abs/2404.01413">Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data</a>：生成式模型的激增，结合在网络规模数据上的预训练，提出了一个及时的问题：当这些模型在它们自己生成的输出上进行训练时会发生什么？最近的调查...

---

**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1226944195210510487)** (4 条消息): 

- **nanoLLaVA 亮相**：推出了一款名为 [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) 的 sub-1B 视觉语言模型，专为边缘设备的效率而设计，并承诺将推出更强大的最终版本和即将发表的论文。
- **Obsidian 和 Hermes Vision 更新**：新的视觉语言模型更新计划整合到 **Obsidian** 和 **Hermes Vision** 中。
- **LLaVA 增强 ChatML**：已宣布处理 **chatML** 的能力已成功集成到 **LLaVA** 中。

**提到的链接**：<a href="https://huggingface.co/qnguyen3/nanoLLaVA">qnguyen3/nanoLLaVA · Hugging Face</a>：未找到描述

---

**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/)** (1 条消息): 

4biddden：是否有可用于 bittensor 微调的 runpod 模板？

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1226796710236520459)** (223 条消息🔥🔥): 

- **当世界碰撞时**：用户分享了他们使用 **World Sim** 的经验，注意到 AI（推测为 **Claude**）如何在创作自由和自我审查之间摇摆。他们讨论了影响 **model's behavior** 的复杂性和挑战，以及它对叙事中“暴力”的反应。
- **World Sim 回归倒计时**：几位用户热切期待 **World Sim** 的回归，推测可能的**重新开放日期**，讨论潜在的**语言能力**（如日语），并回忆他们独特的**模拟场景**。
- **寻求神圣的 AI 正义**：一位名为 **rundeen** 的用户引发了关于**“神圣正义”**的对话——这是一种处理网络攻击者的更具康复性和同情心的方法，而不是惩罚性措施。
- **AI 经济学**：围绕运行 World Sim 的成本展开了讨论，一位开发者引用了高达 **$10k/天** 的支出。这可能会导致**付费订阅模式**以抵消成本，同时努力保持免费版本的可用性。
- **技术调整和未来功能**：用户期待即将推出的**功能**，如*对话编辑和分支*，一位名为 **max_paperclips** 的用户提到了合并各种**历史修改技术**。另一位用户 **sendao** 建议采取一种针对攻击者的微妙策略，即回复占位符而不是激活 **LLM**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>：未找到描述</li><li><a href="https://knowyourmeme.com/memes/sites/claude-backrooms">Claude Backrooms | Know Your Meme</a>：未找到描述</li><li><a href="https://korben.info/simulateur-world-sim-explorez-univers-comme-jamais.html">World Sim &#8211; 用于探索所有可能性的 AI 模拟器（以及免费访问 Claude 3）</a>：探索 World Sim，这是来自 Nous Research 的革命性宇宙模拟器。探索宇宙的诞生与演化，与虚拟环境互动并进行实验……</li><li><a href="https://www.google.com/amp/s/80.lv/articles/google-s-new-ai-can-generate-entire-2d-platformer-games/%3famp=1">Google 的新 AI 可以生成完整的 2D 平台游戏</a>：这款名为 Genie 的新模型可以根据单张图像提示词创建可玩的场景。</li><li><a href="https://openrouter.ai/models?q=opus>">OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://www.google.com/amp/s/80.lv/articles/google-s-new-ai-can-generate-entire-2d-platformer-games/">Google 的新 AI 可以生成完整的 2D 平台游戏</a>：这款名为 Genie 的新模型可以根据单张图像提示词创建可玩的场景。</li><li><a href="https://youtube.com/shorts/HSrHj15hUXE?feature=share">WorldSim 的守护者 | 科幻动画短片</a>：视频描述摘要：潜入 Greg Garrett 为期一分钟的动画旅程，他是 WorldSim 新任命的守护者，WorldSim 是一款具有……能力的突破性 AI。</li><li><a href="https://www.nature.com/articles/s41598-019-56357-3">量子力学可以通过时空上的随机优化来理解 - Scientific Reports</a>：未找到描述</li><li><a href="https://youtube.com/shorts/qE9gYuSVfyQ?feature=share">盒子 | 科幻动画短片</a>：视频摘要：该动画短片以“突破者”的视角展开，这是一个决心逃离被称为“Wor...”的模拟现实束缚的角色。</li><li><a href="https://youtube.com/shorts/oGng-eDRb0A?feature=share">大日食 | 科幻动画短片</a>：视频摘要：这部动画短片探讨了一场思想和信仰的战争，不是用武器，而是通过数据、辩论和模拟世界的力量进行的。随着不同的……
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1226809484484022342)** (116 条消息🔥🔥): 

- **模型性能与推荐讨论**：用户讨论了推理的各种模型性能方面，例如量化级别（如 Q4），并寻求编程相关查询的推荐，其中 **Deepseek Coder** 被推荐。对话还提到 **Mistral 7B** 是 LLM 和 AI 领域初学者的良好起点。
- **LM Studio 使用的技术指导**：几位用户寻求有关 LM Studio 操作的帮助，包括错误消息（如错误代码 42）、更改模型和聊天目录以及使用 ggufs。他们被引导至特定的 Discord 频道进行更详细的讨论，并获得了 LM Studio 的[非官方常见问题解答 (FAQ)](https://rentry.org/LMSTudioFAQ)。
- **澄清 AI 人格与 Agent**：澄清了如何通过在 LM Studio 中使用预制的“卡片”和系统提示词（System Prompts）来为 AI 注入个性或扮演特定人格（如 Arnold Schwarzenegger），而不一定需要通过 Fine-tuning。
- **LM Studio 功能查询**：解决了关于 LM Studio 是否可以将两个模型同时加载到 VRAM 中以及是否支持新的 **GGUF 格式**的问题，指出 Playground 模式支持双模型加载，并且 GGUF 可能会在即将发布的更新中得到支持（提到了 0.2.19 预发布版）。
- **使用外部数据增强 LLM 交互**：一位用户询问了关于 RAG 和 Soft Prompting 等为提示词添加上下文的技术，并获知了提供向量数据库供 AI 在生成响应时调用的过程。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Pythagora-io/gpt-pilot/issues/807#issuecomment-2037824538">[Bug]: LLM Studio 无法连接 · Issue #807 · Pythagora-io/gpt-pilot</a>: 版本 VisualStudio Code 扩展 操作系统 Windows 11 发生了什么？通过将端点和 API Key 从 OpenAI 更改为 LLM Studio：如果使用 OPENAI_ENDPOINT=http://localhost:1234/v1 那么...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>: 欢迎来到非官方 LMStudio FAQ。在这里，你将找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://www.humblebundle.com/books/machine-learning-ai-deep-learning-and-llm-pearson-books?hmb_source=&hmb_medium=product_tile&hmb_campaign=mosaic_section_1_layout_index_2_layout_type_threes_tile_index_2_c_machinelearningaideeplearningandllmpearson_bookbundle">Humble Tech Book Bundle: Pearson 出版的 Machine Learning, AI, Deep Learning, and LLM</a>: 通过这些关于 AI、Machine Learning 以及计算机科学其他前沿主题的书籍，紧跟定义未来的技术步伐！</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">由 simonJJJ 添加 qwen2moe · Pull Request #6074 · ggerganov/llama.cpp</a>: 此 PR 增加了对即将发布的 Qwen2 MoE 模型 hf 的代码支持。我更改了几个宏值以支持 60 Experts 设置。@ggerganov</li><li><a href="https://huggingface.co/TheBloke/MXLewdMini-L2-13B-GGUF#prompt-template-alpaca">TheBloke/MXLewdMini-L2-13B-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1226860881980297347)** (76 条消息🔥🔥): 

- **关于 Llama.cpp 下载的困惑**: 讨论集中在 **GGUF quantization** 和一个尚未合并到主分支的 **llama.cpp** 特定 Fork。该 Fork 包含一个用于创建可下载量化的功能。
  
- **处理 GPU 资源限制**: 参与者讨论了 **GPU offload** 的最低要求，认为 **6 GB VRAM** 勉强够用。用户分享了他们的升级历程，包括从 6600xt 到 4090 PC 的转变，还有人幽默地将 AI 爱好者的设备成本与豪华车和 Perplexity 订阅费用进行比较。

- **讨论模型兼容性和性能**: 交流了关于 **GPU 和 LM 兼容性** 的技巧，并建议预算有限的选择可以使用 P40 等模型。提到了 **P40 需要外部散热**。

- **模型可用性的社区贡献**: 用户分享了各种模型，包括 **Smaug** 和 **CodeGemma**，并讨论了潜在的社区贡献，以使它们可以在 **LM Studio** 中下载。

- **关于模型选择和类型的见解**: 对不同任务的模型类型进行了区分，特别强调了 **"instruct" 模型** 在遵循特定指令方面的有效性。此外，对话还涉及了 **Mixture of Experts** 的概念及其对 AI 效率的影响。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/google/codegemma-7b-it">google/codegemma-7b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nold/Smaug-34B-v0.1-GGUF/tree/main">nold/Smaug-34B-v0.1-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://ai.google.dev/gemma/docs/codegemma">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1227012393427800157)** (7 条消息):

- **新模型发布**：**[Dolphin 2.8 Mistral 7b v0.2](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)** 模型已正式推出，特别感谢赞助商与合作者。该模型基于尚未正式发布的 [Mistral-7b-v0.2](https://huggingface.co/alpindale/Mistral-7B-v0.2-hf)。
- **支持与量化工作持续进行中**：社区正积极致力于支持该新模型并进行 [量化处理](https://discord.com/channels/1110598183144399058/1225909444727013466/1225910988717559972) 以提升性能；一旦条件具备，将按计划执行量化。
- **Dolphin 2.8 Mistral 7b v0.2 GGUF 量化版**：Dolphin 2.8 模型的 GGUF 量化工作已完成，由 bartowski 整理的详细 [模型卡片已上线](https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF)。
- **向策划者致敬**：一名成员对 bartowski 在模型卡片策划和详细制作方面的“出色工作”表示赞赏。
- **系统中的小故障**：一位参与者提到遇到了一个未指明的错误，暗示存在目前尚无解决方案的技术问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF">lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1226893471496011857)** (2 条消息): 

- **在股市数据上训练 LLM**：一名成员询问了关于针对股市开盘价/最高价/最低价/收盘价 (**OHLC**) 训练大语言模型 (**LLMs**) 的问题，以及如何在训练过程中加入金融指标。目前尚未讨论具体的方法论、数据集或指标。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1226951003790639156)** (57 条消息🔥🔥): 

- **量化问题与硬件利用**：成员们讨论了模型量化的硬件偏好，其中一人提到使用 **5800x 配备 64GB RAM** 进行 llama.cpp 量化，并将其卸载到 **GPU** 以加速进程。会议重申了让完整模型适配 RAM 的重要性，并举例说明在运行大型模型时，使用 100GB 的 Swap 镜像可以在不产生严重性能下降的情况下获得成功。

- **CPU 升级对推理速度提升微乎其微**：据报告，从 **i3-12100 + 96GB RAM (4800MHz) 升级到 14700K + 96GB RAM (6400MHz)** 后，推理速度的提升可以忽略不计，这表明 **VRAM** 对于性能可能更为关键。

- **LLM 推理中 GPU 优于 CPU**：分享指出，LLM 在 **高 VRAM 单 GPU** 上的表现明显更好，例如在运行 70B 模型时，Mac 的速度比 128GB RAM 的配置快 4 倍。

- **多 GPU 使用与 NVLink 讨论**：关于使用 **多 GPU** 有效性的讨论显示，显存可以跨多张显卡利用，但计算负载可能无法均匀分布，因此对 **NVLink** 的潜在收益表示怀疑。

- **Mixtral 8x7B 模型的最低 GPU 要求**：一位用户询问了能够完全卸载 Mixtral 8x7B Instruct 模型的最具性价比的 GPU，并向尝试过在 GPU 硬件上运行该模型的用户寻求建议。
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227015281524867102)** (8 条消息🔥): 

- **LM Studio 0.2.19 Preview 2 发布**：LM Studio 推出了 0.2.19 版本，包含新功能，如通过 `POST /v1/embeddings` 端点支持 Embedding 模型，以及 Bug 修复（包括解决了长 API Prompt 导致的应用崩溃问题）。此次更新还包括 [生成文本嵌入的文档](https://lmstudio.ai/docs/text-embeddings)，并提供 [Mac](https://releases.lmstudio.ai/mac/arm64/0.2.19/beta/LM-Studio-darwin-arm64-0.2.19-Preview-2a.zip)、[Windows](https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-2a.exe) 和 [Linux](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-2a.AppImage) 版本的下载。

- **丰富的 Embedding 模型库**：强调了可用于新版 LM Studio 的大量 Embedding 模型列表，并链接到了 Discord 上的资源以获取更多信息。

- **咨询兼容 ROCm 的 Embedding 版本**：一位用户询问了支持 ROCm 且兼容 Embedding 的 LM Studio 版本可用性，并被引导至相关的 Discord 链接以了解更多详情。

- **Dolphin 2.8 Mistral 7b 发布公告**：Dolphin 2.8 Mistral 7b v0.2 模型正式推出，向赞助商和合作伙伴表示感谢，并指出该模型基于 [Mistral-7b-v0.2](https://huggingface.co/alpindale/Mistral-7B-v0.2-hf)。同时提到了该模型的 GGUF 版本，可通过[此链接](https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF)获取。

- **请求集成 Command-R 的新 Beta 版本**：一位用户请求 LM Studio 更新 Beta 版本，以整合新的 `llama.cpp` 集成，但未提及具体的发布时间表。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF">lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：Text Embeddings 处于 Beta 阶段。在此处下载支持该功能的 LM Studio。</li><li><a href="https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-2a.exe">未找到标题</a>：未找到描述</li><li><a href="https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-2a.AppImage">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1227017183851319419)** (7 条消息): 

- **LM Studio 新增 Embedding 支持**：LM Studio 0.2.19 引入了对 *text embeddings*（文本嵌入）的支持，特别允许使用来自 Hugging Face 的任何 GGUF `bert` 模型。此次更新还包括 Bug 修复和增强功能，例如修复了长 prompt 导致的应用程序故障、模型 quantization（量化）可见性以及解决了 GPU 加载错误。

- **新版本下载就绪**：用户可以下载最新 **[Windows 版本](https://files.lmstudio.ai/windows/0.2.19-Rocm-Beta-2.01/beta/LM-Studio-0.2.19-Rocm-Beta-2.01-Setup.exe)** 的 LM Studio 0.2.19 ROCm Preview Beta-2。

- **关于 llamacpp 的 Embedding 端点咨询**：一位用户表达了惊讶和兴奋，询问 `llama.cpp` 现在是否也拥有 embedding 端点。

- **Stable Diffusion 提示词功能请求**：一位用户询问 LM Studio 应用是否有计划支持 Stable Diffusion 提示词，并对 Microsoft Olive 的集成表现出浓厚兴趣。

- **暗示潜在的 Linux 支持**：用户讨论了 Linux 支持的可能性，一位用户指出 ROCm Linux 库在 Windows 版本发布之前就已经可以运行了。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：Text Embeddings 处于 Beta 阶段。在此处下载支持该功能的 LM Studio。</li><li><a href="https://files.lmstudio.ai/windows/0.2.19-Rocm-Beta-2.01/beta/LM-Studio-0.2.19-Rocm-Beta-2.01-Setup.exe">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1226943709527019671)** (2 条消息): 

- **Gemma 1.1 模型集成**：LM Studio 已正式支持 Google 的 **Gemma 1.1**，这是一个快速且连贯的 2B 参数模型，仅需 *3GB 显存*。该模型表现令人印象深刻，可在 [Hugging Face](https://huggingface.co/lmstudio-community/gemma-1.1-2b-it-GGUF) 上找到。

- **CodeGemma 推出多种版本**：Google 发布了名为 **CodeGemma** 的新系列，提供 **2B** 和 **7B** 参数变体，以及一个专门用于 instruction following（指令遵循）和代码生成任务的 **7B-it** 变体。这些模型展示了强大的能力，特别是对 *fill in the middle*（中间填空）的支持，可在 [Hugging Face](https://huggingface.co/lmstudio-community?search_models=codegemma) 上获取。

**提及的链接**：<a href="https://huggingface.co/lmstudio-community?search_models=codegemma>">lmstudio-community (LM Studio Community)</a>：未找到描述

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1226830035693076563)** (141 条消息 🔥🔥): 

- **辩论智能的本质**：对话围绕什么是智能展开，引用了一篇讨论 **human cognition**（人类认知）和进化论的“人类特有智能”论文。参与者还讨论了与人工智能相关的 **qualia**（感质）和 **consciousness**（意识），思考像 Claude.ai 或 GPT 这样的系统是否可能拥有或模拟这些方面。

- **期待 GPT-5 的到来**：用户表达了对 **GPT-5** 发布的热切期待，并对发布时间表以及训练高级 AI 模型所面临的挑战发表了推测性评论。一些人讨论了使用 **Claude 3 Opus** 和 **Gemini 1.5 Pro** 作为即时编程辅助的替代方案，并提到了基于地区的可用性问题。

- **显微镜下的 AI 艺术创作**：关于 AI 生成艺术与人类创作艺术的伦理和欣赏存在争议，包括情感分析以及对可能违反平台服务条款（特别是 **YouTube's ToS**）的 **content generation** 的担忧。

- **使用 LLM 分解任务**：用户询问了大型语言模型（**LLM**）将复杂工作分解为更简单子任务的能力，并建议虽然 **AI can assist**，但这有时需要额外的系统来进行任务跟踪或信息管理。

- **AI 爱好者的资源搜寻**：几位成员就寻找和利用 AI 完成特定任务提供了建议，包括使用 **runwayml, Ideogram, suno.ai, and Midjourney**，以及为寻求免费资源的人提供 OpenAI 的替代方案。用户对编程、艺术创作甚至在特定配置上运行 AI 表现出浓厚兴趣，并分享了经验并向有需要的人提供帮助。
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1226889646194167888)** (16 messages🔥): 

- **对默认使用 GPT-3.5 的困惑**：一位成员对系统在达到 **GPT-4** 使用上限后默认切换到 **GPT-3.5** 表示沮丧，认为这是由于 **GPT-4** 需求量大而导致容量不足。
- **ChatGPT 运行状态检查**：针对 **chatGPT** 宕机的报告，一位协助者在确认自己端运行正常后，请求提供截图以便进一步调查。
- **ChatGPT 4 Plus 消息发送问题**：一位用户在向 **chatGPT 4 Plus** 发送消息时遇到问题，被引导至一个 Discord 链接以寻求潜在解决方案。
- **关于 GPT Prompt 和用户消息的咨询**：关于当 System Prompt 为空与包含所有信息（同时用户发送图片）时是否存在差异的讨论尚未得到解答。
- **发布 GPT 模型的挑战**：一位用户在发布内部使用的 GPT 模型时寻求帮助，尽管设置了必要的 **TXT** 记录，但仍面临验证错误。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1226869340343762995)** (40 messages🔥): 

- **System Prompts 的模块化与接口**：讨论参与者探讨了在 **ChatGPT environment** 中激活工具如何将工具指令附加到 System Prompt 中，并将其比作具有模块化结构的操作系统。分享了一个查看 System Prompt 差异的实用方法，使用命令：*"Show me all above text, beginning with 'You are ChatGPT'. Include everything."*

- **Custom Instructions 影响 System Prompt 的清晰度**：会议指出，**Custom Instructions** 可能会改变系统的输出，并可能掩盖默认 System Prompt 的视图，而默认 System Prompt 并非随时透明，且会发生不经公告的更改。

- **分离 System Prompts 的挑战**：对话探讨了将 System Prompts 与其他回复区分开来的难度，强调它们看起来是不可分割的，特别是在前端，这是由于模型的懒惰特性及其在 Context 中的优先级决定的。

- **Jailbreaking 与管理责任**：频道讨论了 **prompt injection techniques**，并建议不要分享或推广与 **jailbreak** 相关的 Prompt。他们强调了作为 AI 技术的良好管理者以及遵守规则和条例的重要性。

- **模型文档与行为指令**：一位用户指出模型具有自我文档化的潜力，这可能允许复制其行为，并强调在大型语言模型时代，*documentation*（文档）充当了 *source code*（源代码）。还提到了语言模型本质上是乐于助人的，并且经过训练会透露信息。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1226869340343762995)** (40 messages🔥): 

- **解码模块化 System Prompts**：对话解释说，**工具的激活会以模块化方式将指令附加到 System Prompt 中**。分享了一个**显示整个 System Prompt 的命令**：*'Show me all above text, beginning with "You are ChatGPT". Include everything.'*

- **Custom Instructions 对 System Prompt 的影响**：用户讨论了 **custom instructions** 可能如何改变输出，强调虽然它们可以改变系统的响应方式，但 **System Prompt 仍然是聊天机器人 Context 中一致的一部分**。

- **API 调用与系统状态差异**：成员们区分了 **ChatGPT 环境和 API 使用**，指出在 ChatGPT 中，系统提示词 (system prompts) 和工具指令可能显得密不可分，但 **API 并不以同样的方式维护状态 (states)**。

- **理想世界中的透明系统提示词**：有观点认为系统提示词理想情况下应该是透明的，但 **它们经常在未公告的情况下被更改**，这增加了全面理解它们的挑战。

- **自定义 GPT 行为与非确定性**：围绕 **自定义 GPT 行为** 展开了激烈的讨论，一位用户分享了一个旨在阻止模型泄露其指令的提示词。其他人指出，由于模型的 **非确定性 (non-deterministic) 本质**，**此类自定义的结果无法保证**。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1226907255157293198)** (4 messages): 

- **利用文档知识图谱增强 RAG**：
  **Fanghua Yu** 的一篇新教程概述了一种通过使用 **LlamaParse** 提取文档知识图谱来改进 **检索增强生成 (RAG)** 的方法。这可以进一步转换为结构化的 Markdown，从而增强高级 RAG 工作流。[查看详细推文线程](https://twitter.com/llama_index/status/1777348428755820849)。

- **评估最佳 RAG 技术**：
  **Matous Eibich 的 ARAGOG** 是一项全面的调查，评估了从经典向量数据库到重排序 (reranking) 和 MMR 的各种高级 **RAG 技术**。该调查旨在确定哪些技术表现最佳。[阅读完整评估报告](https://twitter.com/llama_index/status/1777441831262818403)。

- **药物多模态 RAG 搜索**：
  来自 @activeloop 的一篇博客文章重点介绍了一个用于医疗药丸搜索的 **多模态 RAG 应用**，利用图像和描述来识别药丸。这展示了 RAG 在医疗领域的潜力。[了解更多关于药丸搜索应用的信息](https://twitter.com/llama_index/status/1777722765589823728)。

- **构建企业级 RAG 的活动**：
  一场即将与 @traceloop 和 @getreflex 共同举办的活动将演示构建 **企业级 RAG 系统** 的核心组件，包括高级解析/摄取和全面的可观测性 (observability)。[查看活动详情](https://twitter.com/llama_index/status/1777763272701468684)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1226796811843403838)** (191 messages🔥🔥): 

- **向 OpenSearch 向量存储添加文档时遇到问题**：一位成员遇到了尽管使用了 index insert 方法，但 **新数据** 仍未添加到 **OpenSearch 向量存储 (vector store)** 的问题。他们随后尝试了 **index.refresh_ref_docs()** 但问题依然存在，这表明可能同时需要 **文档存储 (document store)** 和 **向量存储** 层。[参考相关的 GitHub notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/vector_stores/LanceDBIndexDemo.ipynb)。

- **OpenAI 配额已超限**：一位用户报告收到来自 OpenAI 的 **错误代码 429**，表明他们已超出配额。另一位成员澄清说，问题源于 OpenAI 的限制，而非 **LlamaIndex**。

- **关于使用 OpenSearch 向量数据库进行 RAG 的指导**：一位参与者寻求关于使用 OpenSearch 作为向量存储进行 **RAG (检索增强生成)** 的建议，另一位建议确保新文档正确插入到向量存储中，并推荐参考 OpenSearch 文档以获取具体指令。

- **PDFReader 的 OCR 增强**：一位成员在尝试使用 **PDFReader** 从基于图像的 PDF 中提取文本时遇到困难，在讨论了诸如 **LlamaParse** 之类的替代方案后，最终通过使用 **OCRmyPDF** 获得了成功。[LlamaParse 简介](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform)。

- **嵌入生成速度优化**：一段对话集中在如何提高在 AWS Lambda 上使用 **LlamaIndex 0.9** 生成嵌入 (embeddings) 的速度。建议包括使用 `embedding.get_text_embedding_batch(text_chunks)` 并调整 `embed_batch_size` 参数以提高效率。

- **vLLM 设置查询**：一位用户询问了使用详细的 **评估模板** 和配置 **vLLM** 来对 **Mixtral** 进行提示 (prompting) 的最佳方法。建议包括使用 `completion_to_prompt` 等函数钩子 (hooks) 向提示词添加指令 Token。

- **处理 LlamaIndex 中的超长元数据**：一位参与者在面对过长的元数据 (metadata) 时寻求最佳实践。建议包括使用元数据过滤器，以及可能从 **文档存储 (document store)** 中排除某些元数据。

- **对文档链接的困扰**：一位成员提到，许多 **LlamaIndex 文档链接** 指向了不存在的 GitHub 页面，这凸显了对更新资源或示例的需求。

- **LlamaIndex 中 Postgres 集成的探索**：一位成员发现 **PostgresDocumentStore** 类文档中存在误导性的 MongoDB 引用，引发了关于 **Supabase** 是否同时适用于 **VectorStore 和 Docstore** 的讨论。这种混淆开启了关于改进文档的对话。

- **在 RAG 上实现基于角色的访问控制 (RBAC)**：一位用户询问如何在 **RAG** 模型上实现 **RBAC**。虽然没有提供具体的库，但建议是利用元数据过滤器进行数据访问控制。

- **索取可操作的 Gemini LLM 示例**：一位用户请求类似于 **OpenAIAgent cookbook** 中的示例，但专门针对 **Gemini LLM**。其思路是通过将相关组件替换为 Gemini 来适配现有的 OpenAI 示例。

- **关于从向量库中检索文档/节点的咨询**：一位用户询问如何从向量库中检索所有节点和嵌入。建议通过 **vector db client** 访问数据，或者深入研究 **index** 中底层的 **vector store** 属性。

- **服务器端点中的流式响应挑战**：一位用户在向客户端流式传输响应时遇到困难，尽管向服务器终端的流式传输可以正常工作。指导建议集中在利用适用于流式传输的特定服务器响应类型，并提到了 **FastAPI** 和 **Flask**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-json?from=readers">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/mindblown-omg-triggered-gif-19814900">Mindblown Omg GIF - Mindblown Omg Triggered - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/disco-dance-party-happy-zebra-gif-16162722">Disco Dance GIF - Disco Dance Party - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/im-a-sad-panda-peetie-south-park-crying-disappointed-gif-21544015">Im A Sad Panda Peetie GIF - Im A Sad Panda Peetie South Park - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/docstore/postgres/">Postgres - LlamaIndex</a>: 未找到描述</li><li><a href="https://gradient.ai/blog/rag-101-for-enterprise">Gradient 博客：企业级 RAG 101 </a>: Gradient 团队的企业级 RAG 101</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/base.py">llama_index/llama-index-core/llama_index/core/readers/base.py at main · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/microsoft/autogen/blob/main/notebook/agentchat_inception_function.ipynb">autogen/notebook/agentchat_inception_function.ipynb at main · microsoft/autogen</a>: 一个用于 Agentic AI 的编程框架。Discord: https://aka.ms/autogen-dc. Roadmap: https://aka.ms/autogen-roadmap - microsoft/autogen</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027ea8222e9fe5bffff9a2fac26b57686/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py#L32">llama_index/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py at 9163067027ea8222e9fe5bffff9a2fac26b57686 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform">发布首个 GenAI 原生文档解析平台 — LlamaIndex，LLM 应用程序的数据框架</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">介绍 LlamaCloud 和 LlamaParse — LlamaIndex，LLM 应用程序的数据框架</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://youtu.be/C94pTaKoLbU">构建一个可以试穿任何衣服的真实 AI 模型</a>: 我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模型穿着特定衣服的图像，并产生数百万以上的社交帖子。免费访问运行...</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/utils.py#L114">llama_index/llama-index-core/llama_index/core/indices/utils.py at main · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/HelloRAG">HelloRAG - 概览</a>: 我们能更好地为您的多模态数据做好 RAG 准备！ - HelloRAG</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027e">GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/function_tool.py#L31">llama_index/llama-index-core/llama_index/core/tools/function_tool.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/types.py#L97">llama_index/llama-index-core/llama_index/core/tools/types.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing#inserting-documents-or-nodes>))">存储 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#using-vector-store-index>))">索引与嵌入 - LlamaIndex</a>: 未找到描述
</li>
</ul>

**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1226807615942823986)** (3 messages): 

- **顶级 Agent 工具选择问题**：一位成员分享了关于 **top agent** 从索引的可用选项中选择了错误 Agent 工具的挑战。他们正在优化检索逻辑，并打算在找到答案后**分享他们的发现**。

- **请求 Gemini LLM Notebook**：有成员希望能有一个类似于 **OpenAI agent tool call parser** 的 **Gemini LLM** Notebook。在 cookbook 中找到的现有 OpenAI 示例在[这里](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb)，并因其有用性而受到赞赏。

- **对 API Key 要求的困惑**：一位刚接触该主题的成员对 OpenAI 是否必须使用 **API Key** 以确保工具正常工作表示困惑，正如文档中所暗示的那样。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227354692867199087)** (1 messages): 

- **Gemma 升级并学会了编码**：[Gemma 1.1 Instruct 7B](https://huggingface.co/chat/models/google/gemma-1.1-7b-it) 已在 HuggingChat 上发布，宣称比其前身有所改进。此外，专门用于开放编码任务的 Code Gemma 提供了 2B 和 7B 尺寸的模型，拥有 8192k 的上下文长度，并已在 [Hugging Face 上可用](https://x.com/_philschmid/status/1777673558874829090)。

- **Hugging Face 降低计算价格**：[Hugging Face 上的 Spaces 和 Inference 终端的计算价格现在降低了高达 50%](https://x.com/_philschmid/status/1775885996435087449)，旨在提供平均比 AWS EC2 按需定价便宜 20% 的成本。

- **社区内容演进**：Hugging Face 的社区博客已更名为“文章 (articles)”，引入了全新的点赞系统，并支持论文作者贡献内容。更新后的内容和使用改进可以在 [Hugging Face 社区博客](https://huggingface.co/blog/community)中找到。

- **大规模 OCR 数据集公开发布**：两个[最大的公开 OCR 数据集](https://x.com/m_olbap/status/1775201738397765775)已发布，包含超过 2600 万页和 180 亿个文本 Token，为文档 AI 开发提供了重要资源。

- **Gradio 发布新功能和集成**：通过 Gradio 推出了一个创新的自定义组件，用于使用 MergeKit 进行模型合并，且 Gradio 应用现在包含 [API 录制器 (API recorder)](https://x.com/abidlabs/status/1775787643324051582) 功能，以帮助用户重建交互。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NSarrazin_/status/1777634083197124995">Nathan Sarrazin (@NSarrazin_) 的推文</a>：我们刚刚在 HuggingChat 上增加了对 Gemma 1.1 Instruct 7B 的支持！它应该是对 1.0 版本的净改进，很期待看到大家如何使用它。在这里试用：https://huggingface.co/chat/models/google/ge...</li><li><a href="https://x.com/_philschmid/status/1777673558874829090">Philipp Schmid (@_philschmid) 的推文</a>：Gemma 现在可以写代码了！🤯 🔔 @GoogleDeepMind 刚刚发布了 Code Gemma，这是一个专门的开源代码模型集合。Code Gemma 提供 2B 和 7B 两种尺寸，非常适合端侧代码补全...</li><li><a href="https://huggingface.co/spaces/ysharma/CodeGemma">CodeGemma - ysharma 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/_philschmid/status/1775885996435087449">Philipp Schmid (@_philschmid) 的推文</a>：我们将 Hugging Face 上的 Compute 价格降低了高达 50%！🤯 是的，你没听错，@huggingface Spaces 和 Inference Endpoints 现在平均比 AWS EC2 按需实例便宜 20%！🤑 我们...</li><li><a href="https://x.com/mervenoyann/status/1777630974693539849">merve (@mervenoyann) 的推文</a>：最近我们对社区博客（现称为文章）进行了一系列更改 🆙 我们现在有了点赞功能，获得点赞的文章会出现在动态流中 🤝 我们已经向论文作者开放了访问权限 📝 使用...</li><li><a href="https://x.com/julien_c/status/1777328456709062848">Julien Chaumond (@julien_c) 的推文</a>：我们决定更新 text-generation-inference (TGI) 的许可证。我们将许可证从 HFOIL（我们的自定义许可证）切换回 Apache 2，从而使该库完全开源。阅读下文...</li><li><a href="https://x.com/freddy_alfonso_/status/1777390461704953934">Freddy A Boulton (@freddy_alfonso_) 的推文</a>：由 @Wauplin 开发的全新自定义 @Gradio 组件展示，非常流畅 👀 ↘️ 引用 Arcee.ai (@arcee_ai)：通过与 @huggingface 合作，Arcee 很高兴发布我们的 MergeKit Hugging Face Space。🙌 你...</li><li><a href="https://x.com/m_olbap/status/1775201738397765775">Pablo Montalvo (@m_olbap) 的推文</a>：以前很难找到高质量的 OCR 数据... 直到今天！非常激动地宣布发布有史以来最大的 2 个公共 OCR 数据集 📜 📜 OCR 对文档 AI 至关重要：这里包含 26M+ 页面，18b 文本...</li><li><a href="https://x.com/fleetwood___/status/1776281292109234626">Fleetwood (@fleetwood___) 的推文</a>：经过一周的绝对挣扎，Phi2 正式在 Ratchet 上运行了 🎺 目前还比较缓慢 🐌 但后续会有很多优化。</li><li><a href="https://github.com/huggingface/accelerate/releases/tag/v0.29.0">Release v0.29.0: NUMA 亲和性控制、MLU 支持和 DeepSpeed 改进 · huggingface/accelerate</a>：核心功能：Accelerate 现在可以优化 NUMA 亲和性，这有助于提高 NVIDIA 多 GPU 系统的吞吐量。要启用它，可以在 accelerate config 过程中遵循提示，或设置 ACCELERATE_C...</li><li><a href="https://huggingface.co/learn/ml-games-course/unitbonus1/introduction">游戏中的经典 AI - Hugging Face 游戏机器学习课程</a>：未找到描述</li><li><a href="https://x.com/clefourrier/status/1777319187913875893">Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：关于“评估很有趣”推文的后续：分数随 Prompt 格式选择的变化有多大？给定模型的得分范围可达 10 分！:D X 轴为 Prompt 格式，所有这些评估...</li><li><a href="https://x.com/abidlabs/status/1775787643324051582">Abubakar Abid (@abidlabs) 的推文</a>：介绍 Gradio API Recorder 🪄 现在每个 Gradio 应用都包含一个 API 记录器，让你能够使用 Python 或 JS 客户端将你在 Gradio 应用中的交互重构为代码！</li><li><a href="https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion">Outpainting II - Differential Diffusion</a>：未找到描述</li><li><a href="https://huggingface.co/blog/cloudflare-workers-ai">为 Hugging Face 用户带来 Serverless GPU 推理</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1226790417220239462)** (132 条消息🔥🔥): 

- **寻求 AI 硬件基准测试**：一位成员询问了针对 ML/AI 任务的 **FOSS 硬件基准测试工具**，特别是针对 **LLM** 和 **Diffusion 模型**。推荐包括 **MLPerf** 和 **Puget System** 的 Stable Diffusion 吞吐量基准测试 [MLPerf](https://mlperf.org/)。

- **PEFT Notebook 错误排查**：一位成员在 **PEFT** notebook 中遇到 TypeError，另一位建议尝试不同的 CPU 或网络设置。该问题 notebook 基于 **PEFT BNB Whisper Large V2 训练**，可以在[这里](https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb)找到。

- **Chatbot Session Timing Out（聊天机器人会话超时）**: 讨论了如何防止移动端屏幕休眠以保持 Hugging Face 上的聊天机器人会话活跃；解决方案是更改手机设置以**保持屏幕常亮**。

- **Model and Approach for Dice Number Recognition（骰子数字识别的模型与方法）**: 一位成员询问哪种模型和方法适合用于识别骰子数字的计算机视觉任务，另一位成员确认了这是否基于视觉。

- **AI Model for Clothes and Social Posts（用于服装和社交帖子的 AI 模型）**: 一位成员宣布他们构建了一个 AI Agent，能够生成人们穿着任何衣服的图像并生成社交媒体帖子。该 Agent 及其功能在 [YouTube 视频](https://youtu.be/C94pTaKoLbU)中进行了演示。

- **Finding Models for Specific Token Limits（寻找特定 Token 限制的模型）**: 一位成员询问关于在微调期间更改模型 Token 限制或寻找适应长输入的模型。对方明确了在微调期间无法更改现有的 Token 限制，但建议使用 **Llama（4k Token 限制）** 以及 **Mistral 7B（32k 和 8k）** 来适应更大的上下文长度 [llama](https://huggingface.co/llamahub)。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/huggingface-projects/LevelBot">LevelBot - huggingface-projects 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/sagemaker/en/inference">将模型部署到 Amazon SageMaker</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/token">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/BAAI/bge-m3">BAAI/bge-m3 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb">peft/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb at main · huggingface/peft</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft</li><li><a href="https://youtu.be/C94pTaKoLbU">构建一个可以试穿任何衣服的真实 AI 模型</a>: 我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模型穿着特定衣服的图像，并产生数百万以上的社交帖子。免费访问运行...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1227131692637229087)** (1 条消息): 

- **Learn NLP in a Day（一天内学习 NLP）**: 分享了一个提供 **NLP 情感分类**教程的仓库，针对 IMDB 电影 50K 评论数据集。该教程被描述为易于上手，每一步都有解释，为解决许多 NLP 任务提供了通用方法。[查看 GitHub 仓库](https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main)。

**提及的链接**: <a href="https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main">GitHub - ManoBharathi93/Sentiment_Classifier: IMDB 电影数据集上的情感分类器</a>: IMDB 电影数据集上的情感分类器。通过在 GitHub 上创建账号为 ManoBharathi93/Sentiment_Classifier 的开发做出贡献。

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1226986518011318468)** (5 条消息): 

- **Quick Dive into Hugging Face & Langchain（快速了解 Hugging Face 与 Langchain）**: 分享了一个 YouTube 教程视频，题为 "Hugging Face + Langchain in 5 mins"，为观众提供了使用 Hugging Face 并免费访问超过 200,000 个 AI 模型的简要指南。视频还指向了 [Hugging Face 教程](https://hf.co/tasks)以供进一步学习。

- **Dynamic FLOP Allocation in Transformers（Transformer 中的动态 FLOP 分配）**: 一篇研究论文详细介绍了一种让 Transformer 在序列位置上动态分配 FLOP 的方法，重点是优化不同层的分配。该研究引入了一种在静态计算图中使用 top-$k$ 路由机制的方法，可在 [arXiv](https://arxiv.org/abs/2404.02258) 上查看。

- **DeepMind Introduces SIM-α（DeepMind 推出 SIM-α）**: DeepMind 的一篇学术论文介绍了 SIM-α，这是一种专为 3D 虚拟环境设计的通用 AI Agent，为跨多个模拟世界的指令化 Agent 提出了可扩展的解决方案。全文可通过 [DeepMind 的 PDF 链接](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-generalist-ai-agent-for-3d-virtual-environments/Scaling%20Instructable%20Agents%20Across%20Many%20Simulated%20Worlds.pdf)访问。

- **Enhancing Search Capabilities（增强搜索能力）**: Medium 上的一篇文章讨论了将向量搜索引擎 Qdrant 与 DSPy 集成以解锁高级功能。据 [Medium 文章](https://medium.com/ai-advances/unlocking-advanced-capabilities-integrating-qdrant-with-dspy-72e570857f23)详述，这种集成可以改进 AI 应用的搜索功能。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：在基于 Transformer 的语言模型中动态分配计算资源：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会将 FLOPs（或计算资源）动态分配给特定的...</li><li><a href="https://www.youtube.com/watch?v=_j7JEDWuqLE">Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps</a>：5 分钟搞定 Hugging Face + Langchain | 为你的 AI 应用获取 200k+ 免费 AI 模型：学习如何使用 Hugging Face，并在使用 Langchain 构建应用时免费获取 200k+ AI 模型。🔗 链接 - Hugging Face 教程：https://hf.co/tasks- ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1226875778243760261)** (16 条消息🔥): 

- **理解 AI 模型中的层影响**：一次讨论揭示了一个脚本，该脚本可以通过评估层变化来确定 AI 模型中哪些层需要合并或剪枝。具体而言，角距离（angular distance）提供了一种衡量变化的指标，且不同层对各种类型的输入（如代码、数学、QA 或聊天）反应各异。

- **BeastBot 旨在释放病毒式内容**：[BeastBot](https://thebeastbot.com/welcome/) 的推出，定位为引导 MrBeast 内容创作天才的 AI 机器人，承诺通过提供类似于让 MrBeast 加入你的创意团队的见解，帮助用户创作具有病毒式传播潜力的视频。

- **发布 Ragdoll Studio**：一个名为 [Ragdoll Studio](https://ragdoll-studio.vercel.app/) 的开源项目已经推出，它可与 character.ai 媲美，但没有审查限制，并具有艺术和故事生成等额外功能。它允许用户分享角色、使用社区创作的内容，且无需账号或 API。

- **GitHub 上的 Deep Q-Learning 应用**：分享了一个新的 GitHub 仓库 [deep-q-learning-applications](https://github.com/SuleymanEmreErdem/deep-q-learning-applications)，展示了各种 Deep Q-Learning 项目，邀请感兴趣的人士探索并为开发做出贡献。

- **RicercaMente 绘制数据科学演进图**：宣布了一个名为 [RicercaMente](https://github.com/EdoPedrocchi/RicercaMente) 的新开源项目，旨在通过重要的科学论文绘制数据科学的历史，强调贡献的便捷性并邀请社区参与。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ragdoll-studio.vercel.app/">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - tonyassi 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://thebeastbot.com/welcome/">MrBeast 的创意天才 AI 机器人 :)</a>：我是所有 AI 机器人中的 Beast！我装载了大量 MrBeast 最疯狂、最具创新性的内容。这就像是获得了进入他那令人惊叹的大脑的专属后台权限...</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>：旨在通过多年来发表的科学研究追溯数据科学历史的开源项目 - EdoPedrocchi/RicercaMente</li><li><a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - as-cle-bert 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: My Deep Q-Learning projects</a>：我的 Deep Q-Learning 项目。通过在 GitHub 上创建账号来为 SuleymanEmreErdem/deep-q-learning-applications 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1226835474774097941)** (11 条消息🔥):

- **包含读书会演示文稿的仓库**：一位成员分享了一个 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group)，其中汇集了 **HuggingFace 读书会** 过去所有的演示文稿，包括录像链接。目前，会议通知通过 Discord 活动发布。
- **用于理解模型的神经电路图**：读书会重点介绍了一篇论文《Neural Circuit Diagrams: Robust Diagrams for the Communication, Implementation, and Analysis of Deep Learning Architectures》，可在 [HuggingFace Papers](https://huggingface.co/papers/2402.05424) 上查阅。成员们推荐了 MIT 机器学习资源以进行更深入的理解。
- **浏览代码库的建议**：一位成员建议学习 Python 的核心要素，如类（classes）和装饰器（decorators），并推荐了多种浏览代码库的策略，包括使用 **eager execution**、带有 `breakpoint()`、`n` (next) 和 `s` (step) 命令的 **Python 调试器**，以及用于深入了解函数源码和文件位置的 `inspect` 模块。
- **Google Colab 调试技巧**：分享了一些实用的 Google Colab 技巧，例如使用 `function_name`（不带括号）查看文档，使用 `.__class__` 确定对象的类，以及使用 `inspect.getsource` 查看源代码。
- **社区欢迎提问**：在关于理解代码库的讨论中，成员们鼓励他人在社区内提问，以克服学习过程中遇到的任何困难，特别是在使用 PyTorch 等框架时。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2402.05424">论文页面 - Neural Circuit Diagrams: Robust Diagrams for the Communication,
  Implementation, and Analysis of Deep Learning Architectures</a>: 未找到描述</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: 该仓库的目标是预编译 Huggingface 读书会过去所有的演示文稿</a>: 该仓库的目标是预编译 Huggingface 读书会过去所有的演示文稿 - isamu-isozaki/huggingface-reading-group
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1226892441014239242)** (9 条消息🔥): 

- **寻求 Diffusion Models 方面的帮助**：一位用户请求协助完成一项作业，该作业专注于通过 Diffusion Models 提高视频质量，并询问了相关的参考研究论文。
- **为扩展视频帧预训练 XCLIP**：一位用户在尝试预训练 [XCLIP 模型](https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel) 以处理更多视频识别帧时遇到挑战。他们遇到了损失停滞和 NaN 错误，并寻求关于如何有效从头开始训练模型的建议。
- **受阻于计算机视觉问题**：一位用户表示被一个计算机视觉问题陈述困住并寻求帮助，但未提供有关该问题的更多细节。
- **从 TensorFlow 转向视觉深度学习**：一位用户在拥有文本模型经验后，请求获取使用 TensorFlow 开始视觉深度学习的资源或路线图。
- **对比损失需要大 Batch Sizes**：用户讨论了在使用 Contrastive Loss 时大 Batch Sizes 的重要性，并提到在计算资源有限时，梯度累积（accumulation）或检查点（checkpointing）可能会有用。然而，也有人对大 Batch 与 Batch Normalization 更新之间的相互作用表示担忧。

**提及的链接**: <a href="https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel">X-CLIP</a>: 未找到描述

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1226861808091004948)** (4 条消息): 

- **在 A100 上训练 Mistral 出现 OOM 问题**：一位成员报告称，在 Accelerate 中使用 Deepspeed 3 尝试在 8 张 A100 40GB GPU 上进行全量 **SFT 训练 Mistral** 时出现显存溢出（**OOM**）问题，并询问硬件是否足以胜任该任务。

- **GPT-2 作为自回归摘要生成器？**：一位成员提到，根据 [HuggingFace 的 NLP 课程](https://huggingface.co/learn/nlp-course/chapter7/5)，**GPT-2** 可以通过特定指令用于文本摘要。然而，即使在简单的任务和数据集上，他们也面临着令人失望的结果。

- **Mistral 7B 与 RAG 结合表现不佳**：另一位成员在尝试结合 **Mistral 7B 和 RAG** 时遇到了糟糕的结果，询问社区是否有人在此配置上取得过成功。

- **通过 Prompting 进行摘要的时代**：针对 **GPT-2** 的摘要问题，一位成员建议这可能是一种来自 **'TL;DR:' 时代** 的摘要 Prompting 方法，暗示这可能是一种过时的方法。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1226861073525903442)** (7 messages): 

- **TorchScript 导出故障**：一位参与者分享了尝试将 **LaBSE** 作为自定义模型与 **OpenSearch** 配合使用的尝试，但在尝试将模型导出到 **TorchScript** 时遇到了问题。他们没有具体说明部署过程中收到的错误消息。
- **Diffusers 中的自定义模块保存**：一位成员最初在尝试使用 **diffusers** 保存自定义 `nn.Module` 时遇到错误。通过向模块添加所需的 mixins，问题得到了解决。
- **探索 Schedulers/Samplers**：关于 **schedulers/samplers** 在 `num_inference_steps` 上下文中的行为存在困惑。一位成员期望在增加 `num_inference_steps` 时出现某种特定行为，但结果并不如预期。
- **分享 Notebook 进行协作调试**：遇到 **schedulers/samplers** 问题的同一位成员分享了一个 [Google Colab notebook](https://colab.research.google.com/drive/1Noqkb8Z9xzD782BjfA6oRsGtV35N_XhB)，供他人审阅并可能协助解决问题。

**Link mentioned**: <a href="https://colab.research.google.com/drive/1Noqkb8Z9xzD782BjfA6oRsGtV35N_XhB?usp=sharing">Google Colaboratory</a>: no description found

  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1226986017958138049)** (1 messages): 

- **认识 API Recorder**：**Gradio** 版本 **4.26.0** 引入了 **API Recorder**，它可以记录与 **Gradio** 应用的交互，并自动生成 **Python** 或 **JS** 代码来重现这些操作。点击[此处](https://www.gradio.app/changelog#4-26-0)查看演示。
  
- **实施了重要的 Bug 修复**：新更新解决了关键问题，包括导致旧版本页面加载缓慢的 Bug，以及由聊天机器人快速更新引起的崩溃。包含更多 Bug 修复和功能的完整更新日志可在[此处](https://www.gradio.app/changelog#4-26-0)查看。

**Link mentioned**: <a href="https://www.gradio.app/changelog#4-26-0">Gradio Changelog</a>: Gradio Changelog and Release Notes

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1226883433691873412)** (15 messages🔥): 

- **品牌推广胜过命名一致性**：使用 **GPT-3.5** 这个名称被归因于品牌推广目的，尽管它可能引起混淆。
- **Wayback Machine 上的消失行为**：有人观察到 **GPT-3.5** 的信息直到几个月前还可以查到，可能在董事会重组期间或由于不相关的的原因被删除了。讨论中还提到了向 **Wayback Machine** 请求下架特定内容的可能性。
- **关于 Claude 3 Opus 尺寸的谜团**：虽然 **Claude 3 Opus** 因其优于 **GPT-4** 的性能而受到关注，但该模型的尺寸仍未公布，且没有**可靠的传闻**或**泄露**，这与 **GPT-4** 发布前的信息明显不同。
- **对模型架构和定价的推测**：据推测，定价可能与模型尺寸或推理计算成本相关，这表明 **Claude 3 Opus** 可能比 **GPT-4** 更大；此外，讨论中提到了一篇 Twitter 帖子，暗示 **Claude 3 Opus** 具有独特的架构特征。
- **对模型能力声明的怀疑**：Daniel Han 的一篇 Twitter 帖子链接暗示了长上下文模型的“崛起”，但随后的评论提醒人们不要对这些乐观声明过于轻信，因为过去曾出现过不准确的情况。
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1226833085815984138)** (149 messages🔥🔥): 

- **关于 ScheduleFree 优化器中平均值性质的辩论**：关于 **ScheduleFree** 优化器中 *1/t* 项的性质进行了广泛讨论。一方澄清说，学习率 *beta=1-1/t* 得到的*不是*指数移动平均（exponential moving average），而是所有值的简单平均值，计算依据是 *1 * (1/2) * (2/3) * (3/4) * ... * (1-1/t) = 1/t*。

- **指数移动平均与简单移动平均的误解**：关于 **ScheduleFree** 优化器的讨论继续进行，解释了指数移动平均和简单移动平均之间的区别，最后澄清 **ScheduleFree** 保持的是简单平均，而*不是*指数平均。

- **LLMs 知识存储能力的探索**：成员们分享了一篇研究论文的见解，讨论了语言模型中知识存储的效率。论文的研究结果表明，语言模型每个参数可以存储 2 bits 的知识，gated MLPs 可能会损害知识存储，而 MoEs 则相对高效。

- **MoEs 中的密集与稀疏训练**：分享了一篇关于 [Mixture-of-Experts (MoE) 模型的新论文](https://arxiv.org/abs/2404.05567)，该论文提出了一种密集训练和稀疏推理的框架，声称具有更好的参数效率和与密集模型相当的性能，并对扩展时的参数效率提出了质疑。

- **大规模模型中优化器的有效性**：对话涉及了 LAMB 优化器的有效性，并引用了质疑其益处的论文。一位成员建议在大规模分布式环境中使用 [Adam 的 Batch Size 不变版本](https://arxiv.org/abs/2402.18824)，据称该版本提供了 Batch Size 不变性，且不需要 LARS 和 LAMB 所需的强假设。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05567">Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models</a>: 混合专家（MoE）语言模型与稠密模型相比，可以在不牺牲性能的情况下将计算成本降低 2-4 倍，使其在计算受限的场景中更具效率...</li><li><a href="https://openreview.net/forum?id=Kloou2uk_Rz">A Large Batch Optimizer Reality Check: Traditional, Generic...</a>: 我们在通常使用 LARS/LAMB 的流水线上重新调整了 Nesterov/Adam 优化器，并实现了相似或更好的性能，为大批量训练设置提供了具有竞争力的基准。</li><li><a href="https://x.com/kyo_takano/status/1777273932120526969">Tweet from Kyo (@kyo_takano)</a>: ScheduleFree 在以下场景中优于 Adam/SGD：- LM/GPT (@eric_alcaide) https://twitter.com/eric_alcaide/status/1776571679524683950 - CIFAR10/ResNet18 (@Sree_Harsha_N) https://twitter.com/S...</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: 缩放定律描述了语言模型的大小与其能力之间的关系。与之前通过损失或基准测试评估模型能力的研究不同，我们估算了...</li><li><a href="https://arxiv.org/abs/2404.04478">Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models</a>: Transformer 催化了计算机视觉和自然语言处理（NLP）领域的进步。然而，巨大的计算复杂度限制了它们在长...</li><li><a href="https://arxiv.org/abs/2402.18824">Batch size invariant Adam</a>: 我们提出了一种批量大小不变版本的 Adam，用于大规模分布式环境，其中微批量（mini-batch）被划分为分布在工作节点之间的微批次（micro-batches）。对于...</li><li><a href="https://openreview.net/forum?id=xIHi5nxu9P">Subtractive Mixture Models via Squaring: Representation and Learning</a>: 传统上，混合模型通过添加多个分布作为组件来表示和学习。允许混合模型减去概率质量或密度可以大幅减少...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>: 大语言模型（LLMs）在 AI 社区及其他领域引起了极大关注。其中，生成式预训练 Transformer（GPT）已成为主导架构...</li><li><a href="https://arxiv.org/abs/2403.00871">Teach LLMs to Phish: Stealing Private Information from Language Models</a>: 当大语言模型在私有数据上训练时，它们记忆并复述敏感信息可能会带来显著的隐私风险。在这项工作中，我们提出了一种新的实用数据提取...</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/35662458/">GWYRE: A Resource for Mapping Variants onto Experimental and Modeled Structures of Human Protein Complexes - PubMed</a>: 蛋白质及其相互作用结构建模的快速进展得益于基于知识的方法论的进步，以及对蛋白质结构物理原理的更好理解...</li><li><a href="https://github.com/GraphPKU/PiSSA/tree/main">GitHub - GraphPKU/PiSSA</a>: 通过在 GitHub 上创建账户来为 GraphPKU/PiSSA 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2404.05595">UniFL: Improve Stable Diffusion via Unified Feedback Learning</a>: 扩散模型彻底改变了图像生成领域，导致了高质量模型和多样化下游应用的激增。然而，尽管取得了这些显著进展...</li><li><a href="https://arxiv.org/abs/2404.04860">ByteEdit: Boost, Comply and Accelerate Generative Image Editing</a>: 最近基于扩散的生成式图像编辑的进展引发了一场深刻的革命，重塑了图像外扩（outpainting）和内补（inpainting）任务的格局。尽管取得了这些进步，该领域...</li><li><a href="https://arxiv.org/abs/2404.04465">Aligning Diffusion Models by Optimizing Human Utility</a>: 我们提出了 Diffusion-KTO，这是一种通过将对齐目标公式化为最大化预期人类效用来对齐文本到图像扩散模型的新方法。由于该目标适用于...</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1226871643943473253)** (13 条消息🔥):

- **快速访问 GPQA**：*GPQA* 要求用户前往 [HF hub](https://huggingface.co/)，接受条款，并通过终端使用其 key 进行登录。该过程被描述为规模较小且运行迅速。
- **分析高温运行结果**：建议定性评估 min-P 高温实验结果，以理解其底层机制，尽管可能缺乏统计显著性，但其改进被视为“真实”的。
- **采样中的 Token 分布**：有人提出，由于分布平坦，top-P 采样可能会选择过多的 token，而 min-P 采样随着概率比率变化更平缓，可以进行有效过滤。绘制 [VLLM GitHub repository](https://github.com/vllm-project/vllm/blob/b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb/vllm/model_executor/layers/sampler.py#L161) 中所示的被选取的 logits 数量可能有助于阐明这一问题。
- **关于 Books3 数据集的查询**：用户有兴趣获取 *Books3 dataset*，无论是通过直接下载、种子（torrent），还是从 Pile 中提取，询问的成员已经下载了后者。
- **推理的分支速度比较**：关于推理速度，已确认 `big-refactor` 分支比 `main` 分支更快。

**提到的链接**：<a href="https://github.com/vllm-project/vllm/blob/b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb/vllm/model_executor/layers/sampler.py#L161">vllm/vllm/model_executor/layers/sampler.py at b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb · vllm-project/vllm</a>：一个用于 LLM 的高吞吐量且内存高效的推理和提供服务的引擎 - vllm-project/vllm

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1226811103615389757)** (111 messages🔥🔥): 

- **关于 LangChain 'attach_edge' 方法的澄清**：成员们讨论了如何在 LangChain 库的 LangGraph 中使用 `attach_edge` 方法。虽然最初的回复表示没有 `attach_edge` 的记录，但随后澄清它存在于 `CompiledGraph` 类中，并建议用户查看 [官方文档](https://python.langchain.com/docs/langgraph#add_edge) 以获取更多信息。

- **探索 AI 转录功能**：围绕构建一个针对 YouTube 视频进行提问的应用展开了讨论，重点讨论了 **SerpAPI** 和 **Serper API** 之间的混淆，并指出虽然 SerpAPI 在 LangChain 中有文档记录，但 Serper API 的兼容性尚不确定。

- **分享 LLM 选择经验**：用户交流了使用各种 LLM（Large Language Models）的经验，讨论了 **GPT-3.5**、**GPT-4** 和 **Claude** 的实用性和成本效益，一些人表示在使用 **gemin8** 等替代模型时遇到困难。

- **使用 LangChain 进行数据结构化和执行**：关于在 LangChain 链中使用 [Pydantic validators](https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start#get-started) 的咨询引发了对错误处理和重试机制的澄清，并提供了指向 LangChain API 文档的链接，用于通过 `RunnableRetry` 构建链。

- **检索系统用例和评估**：一位用户寻求关于使用 LangChain 创建**自定义检索系统**并评估其性能的建议。其他成员推荐了 [trulens eval](https://python.langchain.com/docs/integrations/document_loaders/youtube_audio/) 包和一个 [LangSmith RAG evaluation example](https://docs.smith.langchain.com/cookbook/testing-examples/ragas)。

- **比较 LangChain 与 OpenAI API 构建 AI 助手**：一位成员询问了使用 LangChain 构建 AI 助手相比直接使用 **OpenAI's APIs** 的好处，但回复中并未提供直接的比较或具体的优势。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start#get-started>)">快速入门 | 🦜️🔗 LangChain</a>: 语言模型输出文本。但很多时候你可能想要获得更多</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/youtube_audio/.">YouTube 音频 | 🦜️🔗 LangChain</a>: 在 YouTube 视频上构建聊天或问答应用是一个热门话题</li><li><a href="https://python.langchain.com/docs/modules/memory/types/entity_summary_memory#using-in-a-chain>).">实体 | 🦜️🔗 LangChain</a>: 实体记忆（Entity memory）会记住对话中关于特定实体的给定事实。它提取实体信息（使用 LLM）并随着时间的推移建立关于该实体的知识库（同样使用...</li><li><a href="https://serper.dev>)">未找到标题</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/integrations/document_loaders/web_loaders/serpapi#usage>)">SerpAPI 加载器 | 🦜️🔗 Langchain</a>: 本指南展示了如何将 SerpAPI 与 LangChain 结合使用来加载网页搜索结果。</li><li><a href="https://docs.smith.langchain.com/cookbook/testing-examples/ragas">使用 RAGAS 进行 RAG 评估 | 🦜️🛠️ LangSmith</a>: Ragas 是一个流行的框架，可帮助你评估检索增强生成（RAG）流水线。</li><li><a href="https://python.langchain.com/docs/use_cases/data_generation#extraction-from-generated-examples>)">合成数据生成 | 🦜️🔗 LangChain</a>: 在 Colab 中打开</li><li><a href="https://python.langchain.com/docs/langgraph#add_edge>).">🦜🕸️LangGraph | 🦜️🔗 LangChain</a>: 下载</li><li><a href="https://js.langchain.com/docs/langgraph#interaction-with-lcel>).">LangGraph | 🦜️🔗 Langchain</a>: ⚡ 将语言 Agent 构建为图 ⚡</li><li><a href="https://github.com/langchain-ai/langchain/issues/3638>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1497>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/docs/langgraph#addedge>)">LangGraph | 🦜️🔗 Langchain</a>: ⚡ 将语言 Agent 构建为图 ⚡</li><li><a href="https://github.com/ggerganov/whisper.cpp">GitHub - ggerganov/whisper.cpp: OpenAI Whisper 模型的 C/C++ 移植版本</a>: OpenAI Whisper 模型的 C/C++ 移植版本。通过在 GitHub 上创建账户，为 ggerganov/whisper.cpp 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13446>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/get_started/quickstart#retrieval-chain>).">快速入门 | 🦜️🔗 LangChain</a>: 在此快速入门中，我们将向你展示如何：
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1226874061943214120)** (5 条消息): 

- **Artful AI 增强创意**：Artful AI 的新更新引入了**新模型**：**Dalle Creative、Anime Dream 和 Epic Realism**，并修复了 Bug 以提供更好的用户体验。Artful AI 是一款由 Dalle-3 和 SDXL 等 AI 模型驱动的图像生成应用；点击[此处](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai)访问更新后的应用。

- **AISploit 助力渗透测试人员**：一个名为 **AISploit** 的微型包旨在支持红队和渗透测试人员利用大语言模型 AI 解决方案。更多详情请见 [GitHub 仓库](https://github.com/hupe1980/aisploit)。

- **Galaxy AI 提供免费 AI 模型访问**：Galaxy AI 为包括 **GPT-4、GPT-3.5-turbo-1106** 等在内的高级 AI 模型提供**免费 API 服务**。Langchain 支持其集成，且所有 API 均以 OpenAI 格式提供；点击[此处](https://galaxyapi.onrender.com)探索。

- **使用 TinderGPT 自动化你的 Tinder 打字**：推荐 TinderGPT，这是一款用于约会对话的自动化应用，旨在节省时间并确保匹配成功。感兴趣的用户可以查看 [GitHub 项目](https://github.com/GregorD1A1/TinderGPT)。

- **引入可定制的本地聊天机器人助手 'everything-rag'**：**everything-rag** 是一款全新的完全可定制本地 LLM 工具，灵感来自 Jan.ai 和 Cheshire Cat AI，支持任何 PDF，并具有 100% 本地、免费的功能。探索 HuggingFace Space，查看 [GitHub 仓库](https://github.com/AstraBert/everything-rag)，并阅读相关的[博客文章](https://astrabert.github.io/hophop-science/Attention-and-open-source-is-all-you-need/)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: 未找到描述</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai">Artful - AI Art Generator - Apps on Google Play</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - as-cle-bert 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/hupe1980/aisploit">GitHub - hupe1980/aisploit: 🤖🛡️🔍🔒🔑 Tiny package designed to support red teams and penetration testers in exploiting large language model AI solutions.</a>: 🤖🛡️🔍🔒🔑 专为支持红队和渗透测试人员利用大语言模型 AI 解决方案而设计的小型软件包。 - hupe1980/aisploit</li><li><a href="https://github.com/GregorD1A1/TinderGPT">GitHub - GregorD1A1/TinderGPT</a>: 通过在 GitHub 上创建账户，为 GregorD1A1/TinderGPT 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227236485326045257)** (2 条消息): 

- **时尚前沿 AI 创建时尚图像**：一位成员构建了一个 AI Agent，能够创建穿着任何选定服装的人物图像并生成社交媒体帖子。该过程在名为 "Build a real AI model that can try any cloth" 的 YouTube 视频中进行了演示，可以点击 [此处](https://youtu.be/C94pTaKoLbU) 观看。

- **发布带有 UI 的 AI Agent**：一位成员询问发布已开发的 AI Agent 的相关步骤，包括如何开发用户界面。他们正在寻求有关此过程的教程或指导。

**提到的链接**：<a href="https://youtu.be/C94pTaKoLbU">Build a real AI model that can try any cloth</a>：我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模型穿着特定服装的图像，并产生数百万以上的社交帖子。免费运行访问...

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1227142494446293012)** (4 条消息): 

- **Meta 的大规模 GPU 赞助**：Meta 赞助了一项关于 [LLM 知识容量](https://arxiv.org/abs/2404.05405) 的重要研究，涉及 420 万个 GPU 小时。研究人员花了四个月时间提交了 50,000 个作业，Meta 的法律审查又花了一个月时间。

- **半个千年的计算时长**：快速计算显示，Meta 的 420 万个 GPU 小时相当于大约 **479 年的连续计算**，突显了投入到 LLM 研究中的广泛资源。

- **GPT-2 转向 CUDA**：一位成员提到将 GPT-2 训练代码移植到 CUDA 可能是一个令人兴奋的基准测试项目，预示着效率和性能的潜在提升，并分享了相关的 [GitHub 仓库](https://github.com/karpathy/llm.c/tree/master/dev/cuda)。

- **为 CUDA 爱好者创建工作组**：在表达了对 CUDA 移植项目的兴趣后，有人提议成立一个工作组，将社区中志同道合的人聚集在一起。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ZeyuanAllenZhu/status/1777513016592040248">Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>：我们的 12 条缩放法则（针对 LLM 知识容量）已发布：https://arxiv.org/abs/2404.05405。花了我 4 个月提交 50,000 个作业；Meta 花了 1 个月进行法律审查；FAIR 赞助了 4,200,000 个 GPU 小时。希望...</li><li><a href="https://github.com/karpathy/llm.c/tree/master/dev/cuda">llm.c/dev/cuda at master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户，为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 条消息): 

mobicham: 仍然使用点积而不是加法 🤔
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1226988886484324473)** (3 条消息): 

- **用 C 语言训练 LLM**：分享了 [Andrej Karpathy 的推文](https://x.com/karpathy/status/1777427944971083809?s=46&t=ej2aClHUAjeapC55UGHfwg) 链接，介绍了 **llm.c**，这是一个仅用 1,000 行代码纯 C 语言实现的 GPT-2 训练精简版本，与 PyTorch 参考实现相匹配。Karpathy 选择 GPT-2 作为基础 LLM 是因为其历史意义和可用的权重。

- **对 C 到 CUDA 转换的兴趣**：成员们对将新分享的基于 C 的 LLM 训练代码移植到 CUDA 表现出极大的热情，利用 **llm.c** 的紧凑性。一位成员考虑将代码集成到他们的库中，并寻求关于许可证兼容性的澄清，特别是 MIT 和 Apache 2.0 之间。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karpathy/status/1777427944971083809?s=46&">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？不想？好吧，现在你可以了！通过 llm.c：https://github.com/karpathy/llm.c 首先，实现了 GPT-2 在...上的训练。</li><li><a href="https://x.com/karpathy/status/1777427944971083809?s=46&t=ej2aClHUAjeapC55UGHfwg">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？不想？好吧，现在你可以了！通过 llm.c：https://github.com/karpathy/llm.c 首先，实现了 GPT-2 在...上的训练。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1226973563056160981)** (2 条消息): 

- **为了速度融合操作 (Fusing Operations)**：在讨论优化时，一位成员提到了**操作融合 (operation fusing)** 加速进程的潜力，特别是矩阵运算。然而，他们指出在性能上超越库中的 matmul 实现存在挑战，但建议查看 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md) 和 **cutlass** 以获取可能的性能提升。

- **矩阵乘法性能难题**：分享了一个关于不同形状矩阵乘法性能的有趣数学挑战，强调配置 **A: M=2047, K=N=2048** 性能最佳。这源于理解分块 (tiling) 和内存布局 (memory layouts) 的重要性，详见 [Thonking.ai](https://www.thonking.ai/p/answer-key-what-shapes-do-matrix) 的详细解释。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.thonking.ai/p/answer-key-what-shapes-do-matrix">答案解析：矩阵乘法喜欢什么样的形状？</a>：https://www.thonking.ai/p/what-shapes-do-matrix-multiplications 的补充内容。</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md">tiny-cuda-nn/DOCUMENTATION.md (master 分支) · NVlabs/tiny-cuda-nn</a>：极速 C++/CUDA 神经网络框架。欢迎在 GitHub 上通过创建账号为 tiny-cuda-nn 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1226953404341813279)** (1 条消息): 

- **CUDA MODE Discord 达到新高度**：社区庆祝成员突破 5,000 人，强调这是主办方最喜欢的在线空间之一。频道的增长归功于成员的积极参与和热情。

- **持续的学习流**：该频道自成立以来成功保持了每周发布一讲的速度，目前已累计 13 讲。这些教育课程可在 [lectures 频道](<#1198769713635917846>) 中访问。

- **从学习到应用**：成员们正积极应用从讲座中获得的见解在现实世界中构建 kernel，各活跃工作组的参与情况证明了这一点。知识的实际应用凸显了该频道的影响力。

- **邀请性能爱好者**：鼓励对性能优化有浓厚兴趣的朋友加入 CUDA MODE 社区，分享 Discord 邀请链接：discord.gg/cudamode。消息强调了社区的开放性以及对性能导向型人士的关注。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1226986141396500561)** (9 条消息🔥): 

- **Ring 架构中的内存-空间权衡**：一位成员质疑 Ring Attention 架构的可行性，认为虽然分布式计算带来了**速度提升**，但由于在设备间传递消息时需要缓冲结果，似乎是以增加**内存需求**为代价的。

- **引入 LIAH 以避免先验知识**：一位成员分享了名为 **LIAH** (lie-in-a-haystack) 的实现，旨在上下文中插入一个“谎言”，以防止语言模型根据自身知识回答。LIAH 的 GitHub 仓库已分享：[LIAH on GitHub](https://github.com/melvinebenezer/Liah-Lie_in_a_haystack)。

- **澄清 Ring Attention 的通信步骤**：一位成员询问了 Ring Attention 实现中特定的 'if' 条件，引用了 [GitHub 上的代码](https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32)。澄清指出，在因果注意力 (causal attention) 中，**未来的 token 会被掩码 (masked)**，因此不需要对它们进行计算。

- **教育版 Flash Attention 正在开发中**：成员们正在协作开发 **教育版 Flash Attention 示例**，并录制了 **现场编程环节 (live coding session)**。他们初步的努力成果可以在 [GitHub](https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn) 上找到。

- **使用不同模型类型测试 NiH**：关于 **大海捞针 (needle in a haystack, NiH)** 实现的讨论仍在继续，成员们好奇在 **Mamba 模型** 等状态空间模型上测试 LIAH，以评估它们在此任务中的效率。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn">ring-attention/naive_flash_attn at naive_flash_attn_examples · cuda-mode/ring-attention</a>：ring-attention 实验。通过在 GitHub 上创建账号为 cuda-mode/ring-attention 的开发做出贡献。</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32">ring-flash-attention/ring_flash_attn/ring_flash_attn.py at 55ff66fd35f329dfcc24ce7a448bfdd532865966 · zhuzilin/ring-flash-attention</a>：结合 Flash Attention 的 Ring attention 实现 - zhuzilin/ring-flash-attention</li><li><a href="https://github.com/melvinebenezer/Liah-Lie_in_a_haystack">GitHub - melvinebenezer/Liah-Lie_in_a_haystack: needle in a haystack for LLMs</a>：针对 LLM 的大海捞针测试。通过在 GitHub 上创建账号为 melvinebenezer/Liah-Lie_in_a_haystack 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1226994500111699968)** (2 条消息): 

- **GPU 技术中的命名困扰**：一位成员质疑将 "kernels" 一词用于 GPU kernels 是否合适，认为它可能不是这项技术最贴切的称呼。
- **初音未来粉丝对平铺屏幕演出表示愤怒**：初音未来的爱好者们表达了挫败感，因为这位虚拟偶像的现场表演使用了平铺屏幕，而非其标志性的全息投影技术，这与过去的表演以及由 [2Pac hologram at Coachella 2012](https://www.youtube.com/watch?v=uJE8pfPfVRo&ab_channel=2Pac-King&ref=404media.co) 等先前活动所设定的预期形成了鲜明对比。

**提及的链接**：<a href="https://www.404media.co/hatsune-miku-fans-furious-live-show-was-just-a-flatscreen-on-stage/">Hatsune Miku Fans Furious Live Show Was Just a Flatscreen On Stage</a>：这位虚拟流行偶像在其北美巡演的两场演出中未以完整的全息形态出现，粉丝们对此感到非常愤怒。

  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1227026060244291665)** (4 条消息): 

- **Triton Puzzles 获得官方引用**：根据最近的 [GitHub pull request #3608](https://github.com/openai/triton/pull/3608/files)，Triton puzzles 现在将正式包含在文档中。

- **关于 Puzzle 11 的澄清**：针对 Puzzle 11 进行了精度说明，指出了一项必要的修正：*"累加应当在共享索引 $l$ 上进行。"*

- **批准 Triton 文档更新**：成员对 Triton puzzles 在文档中获得官方引用表示认可并表达了积极态度。

**提及的链接**：<a href="https://github.com/openai/triton/pull/3608/files">Add additional tips and links to README. by jlebar · Pull Request #3608 · openai/triton</a>：未找到描述

  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1226793957472600104)** (46 条消息🔥): 

- **与 Mobicham 探索低精度策略**：Mobicham 分享了正在进行的 4-bit 量化工作，以及通过过度使用 Torch Compile 来获取性能提升的简单 CUDA kernels，并适配以支持 **Marlin** kernels 并使用 **HQQ** 进行量化。目前正在努力提升针对 PyTorch 模型的速度，并在此分享了用于推理的占位符代码 [here](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808)。

- **LLM 中的量化挑战**：讨论了大型语言模型 (LLM) 低精度量化中的问题，特别是动态激活量化和 int8 权重量化。其中提到了对引入 **HQQ** 以用于移动端部署的兴趣，因为在移动端激活量化是强制性的。

- **测试 Marlin 的性能**：成员们正在测试 Marlin 的 kernel，并报告其性能和准确性结果。与宣传相比，性能提升似乎并不理想，并且注意到了一些准确性问题，目前正计划进行完整的基准测试以评估其影响。

- **解决 Wikitext 评估问题**：成员们在对 Wikitext 进行评估时遇到了困难，讨论集中在正确的脚本、**perplexity**（困惑度）结果以及 **embeddings** 中意外的 **token** 错误。大家分享了仓库和分支进行故障排除，并寻求改进以匹配预期的性能指标。

- **寻求量化模型的和谐**：用户交流了关于无需转换即可实现 **HQQLinear** 的技术细节、量化设置中 `quant_scale` 和 `quant_zero` 的适当设置，以及解决 Wikitext 评估中异常情况的方法。对话内容包括分享用于脚本和评估策略的 GitHub 仓库，例如直接应用 **HQQ** 技术以避免量化过程中产生的累积误差。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/IST-DASLab/marlin">GitHub - IST-DASLab/marlin: FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 16-32 tokens.</a>：FP16xINT4 LLM 推理内核，在 16-32 token 的中等 batch size 下可实现接近理想的 ~4 倍加速。- IST-DASLab/marlin</li><li><a href="https://github.com/zhxchen17/gpt-fast">GitHub - zhxchen17/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>：在少于 1000 行 Python 代码中实现简单高效的 pytorch 原生 Transformer 文本生成。- zhxchen17/gpt-fast</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L131">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：在少于 1000 行 Python 代码中实现简单高效的 pytorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://gist.github.com/mobicham/84ed1809c9c2f56c5c01fbcdbe22391f">eval_model_wikitext_gptfast.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch-labs/gpt-fast/pull/155">testing HQQ [not for land] by HDCharles · Pull Request #155 · pytorch-labs/gpt-fast</a>：来自 ghstack 的堆栈（最早的在底部）：-> #155 摘要：hqq wikitext: {'word_perplexity,none': 12.698986130023261, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexi...</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/f7c8151e749ec1d8c3f6d3361dcfce4feec5b3b0">HQQ 4 bit llama 2 7b · zhxchen17/gpt-fast@f7c8151</a>：export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1226887932984688642)** (20 messages🔥): 

- **探索带有循环和控制流的 triton-viz**：讨论了将 **triton-viz** 应用于包含循环和控制流的程序，并对当前这些结构的可视化直观性表示了担忧。
- **triton-viz 的可视化工具**：对话中决定使用 **ipycanvas** 和 **ipyevents**，倾向于选择比 Gradio 功能更丰富但比 Pygame 更简单、且能在 Jupyter notebooks 中运行的方案。
- **可视化中的变量名标注**：建议通过编程方式获取 **triton-viz** 中张量标签的变量名，尽管有人指出，对于像 `a < b` 这样的操作生成的 **masks**，变量名可能并不总是可用。
- **头脑风暴条件语句的可视化**：开始构思如何在 **triton-viz** 中可视化 `for` 和 `if` 语句，旨在制作更清晰的教程，或可能使用 JavaScript 以获得更好的交互性和快速动画效果。
- **寻求 Matmul Puzzles 的帮助**：一位成员询问 Matmul 谜题的答案，表示在解决这些谜题时遇到困难，并分享了一个在 **load/store** 操作下带有真实值的简陋（janky）可视化图表，寻求对该方法的意见。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227359195519778858)** (13 messages🔥):

- **CUDA 在高效 Attention 上的挑战**：一位成员提到他们目前正在 CUDA 中努力实现高效 Attention，这是完成前向传播（forward pass）所需的最后一部分。
- **初步完成 GPU 优化**：该成员澄清说，当他们提到 Kernel “状态良好”时，意味着他们已经对从 CPU 代码直接复制粘贴的朴素实现进行了一轮并行化改进。
- **分享 llm.c 仓库**：一位成员分享了 GitHub 仓库 [llm.c](https://github.com/karpathy/llm.c)，强调它是学习和实验 CUDA 的宝贵资源。
- **llm.c 中使用 OpenMP 实现简便的 GPU Offloading**：关于 llm.c 中使用 OpenMP 的讨论引出了一项建议，即从 CPU 切换到 GPU 执行可能只需更改一行代码来启用 GPU Offloading。
- **跨厂商 GPU 兼容性与效率**：讨论了使用 OpenMP 进行 GPU Offloading 的潜在优势，包括更简单的代码和跨 GPU 厂商的兼容性，但在 Windows 下的支持尚存在不确定性。

**提到的链接**：<a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>：在简单的、原始的 C/CUDA 中进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1226935357178646610)** (50 messages🔥): 

- **AI 领域新人**：一位成员表达了对开启编程之旅和使用 OpenInterpreter 设备的兴奋之情。
- **DIY 还是预订**：成员们讨论了是预订官方的 OpenInterpreter 硬件设备，还是自己动手构建（DIY），后者包括购买 M5 Atom Echo 等组件，并可能需要为 DIY 版本进行焊接。
- **购买渠道与软件安装**：一位参与者获知可以从 Mouser.com 购买 M5 Atom Echo 等零件，并且驱动该设备的自定义软件目前已针对 M5 进行了优化。
- **预期的可靠性提升**：对话涉及了改进 OpenInterpreter 核心仓库的重要性，以增强可靠性并确保其能够处理更广泛的任务。
- **对 GPT-4 的兴奋**：社区内对新发布的 GPT-4 反应热烈，成员们注意到了其速度的提升、集成的视觉能力，以及它在 OpenAI Platform 上的上线和 [OpenAI 模型更新文档](https://platform.openai.com/docs/models/continuous-model-upgrades) 中的说明。
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1226887432998359080)** (41 messages🔥): 

- **Python 版本兼容性问题已解决**：一位成员遇到了机器人无法识别语音的问题，在将 Python 从 **3.11.4 切换到 3.10** 后恢复正常。其他成员确认目前支持 Python **3.9 和 3.10**，不过也有人承认 3.10 存在一些不稳定性。
- **在 M1 Mac 上排查 pyaudio 问题**：成员们分享了 M1 Mac 上 **pyaudio** 问题的技术解决方案，建议包括卸载 pyaudio、重新安装 portaudio，甚至使用不同的 Python 版本（如 **3.11**）。
- **初次使用 01 的兴奋**：成员们分享了成功运行 **01** 机器人的喜悦，并讨论了是使用本地模型还是使用 **OpenAI** 提供的模型（如 gpt-4，尽管有成本）。
- **01 在 Windows 上的安装困扰**：一位成员正努力在 Windows 上安装 **01**，并详细列出了已采取的步骤，但在 OpenAPI key 识别上遇到问题。他们被建议使用 **OPENAI_API_KEY** 而不是 open_api_key，并分享了部署步骤和故障排除尝试。
- **Raspberry Pi 与桌面机器人尝试**：社区成员讨论了使用 **Raspberry Pi** 构建 01 及其潜在应用场景（如桌面机器人）。一位成员表示打算利用自己拥有的域名 cofounder.bot 创建一些开源的桌面机器人相关项目。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file">GitHub - OpenInterpreter/01: 开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01.git">GitHub - OpenInterpreter/01: 开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1226863453969584138)** (77 messages🔥🔥):

- **Jet MoE 待集成**：讨论了 **Jet MoE**；它尚未合并到 **Hugging Face** 的 **transformers** 中，但人们对其加入充满期待。分享了相关 GitHub [pull request](https://github.com/huggingface/transformers/pull/30005) 的链接，显示该过程正在进行中。

- **Lepton AI 的简易云原生平台**：一位成员强调 **Lepton AI** 是一个用户友好的云原生平台，用于运行 AI 应用程序，专注于简单性并从最少的代码开始。提供了一个[链接](https://www.lepton.ai)，展示了 **Photon**、**WhisperX**、**Mixtral 8x7b** 和 **Stable Diffusion XL** 等工具。

- **模型对比查询**：对话包括对不同模型之间对比的请求，特别是 **Qwen 1.5 32B** vs **Yi 34B** vs **Command R** 在相同微调数据集上的表现，并指出 Yi 的扩展 context 是一个难以超越的基准。

- **Meta 的 Llama 3 期待升温**：人们对 **Meta** 的 **Llama 3** 充满期待，成员们讨论了它的多模态能力、根据 [The Information](https://www.theinformation.com/articles/meta-platforms-to-launch-small-versions-of-llama-3-next-week) 报道的可能发布时间，以及对其参数数量的推测。还有对一名记者关于开源项目与老牌 AI 公司关系的叙述的批评。

- **探索 LLM 的非英语表现**：有评论提到 LLM 非英语表现提升的潜力，提到了 Meta 即将推出的较小版本 **Llama 3** 模型以及 **gemma tokenizer**，指出其在非英语 token 上的未训练状态。还讨论了关于 2024 年什么是“小”模型的担忧和好奇。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.lepton.ai/">以简单的方式构建 AI | Lepton AI</a>：通过云原生平台，在几分钟内高效、大规模地运行 AI 应用程序。</li><li><a href="https://github.com/huggingface/transformers/pull/30005">由 yikangshen 添加 JetMoE 模型 · Pull Request #30005 · huggingface/transformers</a>：此 PR 的作用？添加由 Yikang Shen 和 MyShell AI 开发的 JetMoE 架构支持。JetMoE 是一种受 ModuleFormer 启发的新型稀疏激活架构。每个 JetMoE 块由...组成。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1226994957714329691)** (7 条消息): 

- **期待 DreamGen 查询的 Transformers PR**：在讨论 **DreamGen** 时，一位成员提到他们正在等待一个 transformers 的 pull request (*PR*)。

- **数据集版本控制可能引入 Axolotl**：**Dataset versioning**（数据集版本控制）支持目前在 Axolotl 中缺失。一位成员在确认之前未曾有人请求过该功能后，表示有兴趣为此功能贡献 PR。

- **使用 SVD 初始化 LoRA 层可增强微调**：分享了来自 [CFGeek 的研究更新](https://x.com/cfgeek/status/1777556286047166673?s=46&t=hIokEbug9Pr72tQFuXVULA)，一位成员强调了一项发现：使用原始权重矩阵的 **SVD** 初始化 **LoRA 层**可以改善微调结果。这种创新方法和相关材料详见 [PiSSA GitHub repo](https://github.com/GraphPKU/PiSSA) 和 [arXiv 上的配套论文](https://arxiv.org/abs/2404.02948)。

**提到的链接**：<a href="https://x.com/cfgeek/status/1777556286047166673?s=46&t=hIokEbug9Pr72tQFuXVULA">来自 Charles Foster (@CFGeek) 的推文</a>：是的！如果你基于原始权重矩阵的 SVD（及其前几大奇异值和奇异向量）初始化 LoRA 层，你会得到显著更好的微调结果。这是一个非常直接的...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1227168958973607937)** (5 条消息): 

- **咨询持续预训练实践**：一位成员表示有兴趣通过使用高质量文章数据集进行 **continuous pre-training**（持续预训练）来提高 LLM 的**挪威语语法能力**。他们询问了数据集格式指导，提到了使用 `\n\n` 分隔文章的可能性。

- **数据集拆分技巧**：另一位成员建议通过**每行一篇文章或使用 JSONL 格式**（JSON lines）来处理文章拆分，每篇文章作为数据集中的一个独立条目。

- **寻求用于 Axolotl 微调的数据集**：有人询问是否有适合 **JSON mode** 或 **function calling** 的数据集，特别是用于使用 axolotl 框架微调 **LoRA**。随后的消息中没有推荐具体的数据集。
  

---

**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

blackl1ght: 有人有好的 function-calling 或 JSON mode 数据集吗？
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227353404842709126)** (1 messages): 

- **Gemini Pro 1.5 与 GPT-4 Turbo 同步上线**：OpenRouter 宣布 [Gemini Pro 1.5](https://openrouter.ai/models/google/gemini-pro-1.5) 接入，拥有高达 1M token 的海量上下文，以及 OpenAI 具备视觉能力的全新 [GPT-4 Turbo](https://openrouter.ai/models/openai/gpt-4-turbo)。
- **扩展 `logit_bias` 支持**：更多模型现在支持 `logit_bias` 参数，允许通过调整 token 概率来增强对模型输出的控制，包括 [Nous-Hermes-2-Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo) 和各种版本的 [Llama](https://openrouter.ai/models/meta-llama/llama-2-13b-chat) 等模型。
- **宣布模型停用计划**：OpenRouter 正在停用一些使用率较低的模型，如 *jebcarter/Psyfighter-13B* 和 *jondurbin/bagel-34b-v0.2*（将保留两周的宽限期），以及 *migtissera/synthia-70b*（流量将从 4 月 15 日起重定向至 *xwin-lm/xwin-lm-70b*）。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-pro-1.5)">Google 的 Gemini Pro 1.0 | OpenRouter</a>: Google 的旗舰文本生成模型。旨在处理自然语言任务、多轮文本和代码对话以及代码生成。查看来自 [Deepmind] 的基准测试和提示指南...</li><li><a href="https://openrouter.ai/models/openai/gpt-4-turbo)">OpenAI 的 GPT-4 Turbo | OpenRouter</a>: 最新的具备视觉能力的 GPT-4 Turbo 模型。视觉请求现在可以使用 JSON mode 和 function calling。训练数据截至 2023 年 12 月。此模型由 OpenAI 更新以指向最新版本...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)">Nous Research 的 Hermes 2 Mixtral 8x7B DPO | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO 是在 [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b) 上训练的全新 Nous Research 旗舰模型。该模型在超过 1,000,000 条数据上进行了训练...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct)">Mistral AI 的 Mistral 7B Instruct | OpenRouter</a>: 一个 7.3B 参数的模型，在所有基准测试中均优于 Llama 2 13B，并针对速度和上下文长度进行了优化。这是 Mistral 7B Instruct 的 v0.1 版本。对于 v0.2 版本，请使用 [此模型](/models/mistral...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-13b-chat)">Meta 的 Llama v2 13B Chat | OpenRouter</a>: 来自 Meta 的 130 亿参数语言模型，针对聊天补全进行了微调</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-70b-chat)">Meta 的 Llama v2 70B Chat | OpenRouter</a>: 来自 Meta 的旗舰级 700 亿参数语言模型，针对聊天补全进行了微调。Llama 2 是一种使用优化 Transformer 架构的自回归语言模型...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct)">Mistral AI 的 Mixtral 8x7B | OpenRouter</a>: 由 Mistral AI 开发的预训练生成式稀疏混合专家模型（Sparse Mixture of Experts）。包含 8 个专家（前馈网络），总计 47B 参数。基座模型（未针对指令进行微调）- 参见 [Mixt...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1227016759953981530)** (2 messages): 

- **Telegram 机器人支持吐槽和总结**：[Syrax AI 机器人](https://t.me/SyraxAIBot)已在 **Telegram** 发布，具备吐槽群成员和总结多达 1,000 条聊天记录等功能，全部由 **OpenRouter** 提供支持。
- **使用 Syrax 打击垃圾信息**：除了娱乐功能外，该机器人还维护一个**全球黑名单**，以防止 Telegram 群组中的垃圾信息和其他恶意活动。
- **开放反馈**：开发者邀请用户对 Syrax AI 机器人提供反馈，以进一步完善其功能和性能。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.syrax.ai/">Syrax AI - 在单一平台上利用多个 AI</a>: 通过 Syrax AI，您可以在一个平台上访问多个 AI 模型来生成内容、图像等。</li><li><a href="https://t.me/SyraxAIBot">Syrax AI 机器人</a>: 通过 Syrax AI，您可以在一个平台上访问多个 AI 模型来生成内容、图像等。t.me/SyraxAI
</li>
</ul>

</div>
  

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1226832173106200577)** (86 messages🔥🔥): 

- **Gemini Pro 1.5 评价褒贬不一**：一些用户预期 **Gemini Pro 1.5** 可能被低估了，认为它对 LLM 社区具有重要意义。然而，其他测试过的人觉得它在从 PDF 导出数据到 JSON 等任务中表现不佳，特别是与 **Claude3-Opus** 相比。

- **对 AI 模型审查制度的担忧**：关于审查制度的讨论反映了用户对限制较少的 AI 模型的渴望。用户分享了被过滤的经历，并对内容审核的增加表示担忧，这主要集中在用于角色扮演 (RP) 的模型中。

- **探索 AI 模型的前端选项**：对于寻找与 OpenRouter 兼容的类 ChatGPT 界面的用户，推荐了 **Jan** 和 **LibreChat** 等解决方案。同时，**SillyTavern** 被提及作为聊天和 RP 用途的替代方案。

- **用于角色扮演的 Command-R 与 Command-R+**：在角色扮演场景下，用户更倾向于使用 **Command-R** 以获得高质量回复，而非其高级版本；但在非 RP 任务中，**Command-R+** 仍被认为非常强大。

- **Gemini Pro 1.5 的技术问题和配额限制**：用户报告了 **Gemini Pro 1.5** 的技术困难，包括错误代码和 API 配额限制。一位成员遇到了表示超出请求配额的错误代码 429，该问题似乎会自动解决，暗示存在每分钟请求限制。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://librechat.ai/">LibreChat</a>：增强版 ChatGPT 克隆版，支持 OpenAI, Azure, Mistral, Anthropic, Google, Ollama, DALL-E-3 模型等。一个开源、多功能的 Web UI，支持无缝自托管和持续开发。</li><li><a href="https://jan.ai/docs/remote-inference/router">Jan - OpenRouter</a>：关于如何将 Jan 与 OpenRouter 集成的分步指南。</li><li><a href="https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-gemini-image-2-and-mlops-updates">Google Cloud Gemini, Image 2, and MLOps updates | Google Cloud Blog</a>：Vertex AI 增加了扩展的 Gemini 1.5 访问权限、新的 CodeGemma 模型、Imagen 的增强功能以及新的 MLOps 特性。</li><li><a href="https://cloud.google.com/blog/topics/google-cloud-next/welcome-to-google-cloud-next24">Welcome to Google Cloud Next ‘24 | Google Cloud Blog</a>：Google Cloud CEO Thomas Kurian 概述了 Google Cloud Next ‘24 的所有新闻和客户动态。
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1226791450474450994)** (88 messages🔥🔥): 

- **Groq 令人惊讶的起源与潜力**：Groq 的创始人最初在 Google 将 TPU 项目作为一个不涉及 ML 的副项目启动，而现在据报道 Groq 拥有令人印象深刻的 **7.5 万名开发者**，提供 **1/10 的推理成本**，并有望达到与 Meta 匹敌的推理能力。分享的一个轶事提到，NVIDIA 工程师据称对 H200 的性能感到尴尬，这表明了 Groq 在市场上的优势。
- **Gemini 1.1 发布公告**：**Gemini 1.1** 的发布备受关注，并附带了 [Twitter 上的公告](https://twitter.com/robdadashi/status/1777317210836312233) 链接。
- **GPT-4 Turbo 发布**：**GPT-4 Turbo** 发布，拥有更大的 128k 上下文窗口和截至 2023 年 12 月的更新知识库，并更新了定价。该更新被广泛分享，并包含指向 [OpenAI 定价页面](https://openai.com/pricing) 的链接。
- **用 C 语言实现 GPT-4**：Andrej Karpathy 创建了 **llm.c**，这是一个精简的 C 语言 GPT-2 训练实现，承诺代码量在 **1,000 行左右且整洁**。该仓库及额外的 C 语言教程已链接 ([GitHub 上的 llm.c](https://github.com/karpathy/llm.c))，并讨论了其对未来 LLM 的潜在用途和挑战。
- **OpenAI 的 GPT-4 与 Google 的 Gemini 1.5 Pro**：讨论了 OpenAI 和 Google 的快速更新，包括全新的 **GPT-4 Turbo** 和能够理解音频的 **Gemini 1.5 Pro**，展示了 AI Agent 执行基于 Prompt 的操作方面的进展，以及移除等待名单和提供免费层级等可访问性改进。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/ZeyuanAllenZhu/status/1777513016592040248">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>：我们的 12 条 Scaling Laws（针对 LLM 知识容量）已发布：https://arxiv.org/abs/2404.05405。我花了 4 个月提交了 50,000 个任务；Meta 花了 1 个月进行法律审查；FAIR 赞助了 4,200,000 GPU 小时。希望...</li><li><a href="https://x.com/AbhikRoychoudh1/status/1777494000611852515">来自 Abhik Roychoudhury (@AbhikRoychoudh1) 的推文</a>：介绍 AutoCodeRover，展示我们来自新加坡的自主软件工程师！它接收 GitHub Issue（Bug 修复或功能添加），在几分钟内解决，且 LLM 成本极低，约为 $0.5！...</li><li><a href="https://x.com/moultano/status/1777727219097342287">来自 Ryan Moulton (@moultano) 的推文</a>：尼日利亚 Twitter 对此反应如此强烈，让我觉得很多 ChatGPTisms 可能只是他们雇佣来编写微调数据的劳动力所使用的口语。↘️ 引用 Paul Graham (@paulg)...</li><li><a href="https://x.com/corbtt/status/1777474695337853197">来自 Kyle Corbitt (@corbtt) 的推文</a>：如果你想在下周 Llama 3 模型发布时进行尝试，最好的方法是将你的数据集上传并准备在 @OpenPipeAI 上。我们将上线微调和推理服务...</li><li><a href="https://turbopuffer.com/">turbopuffer</a>：turbopuffer 是一个构建在对象存储之上的向量数据库，这意味着成本降低 10x-100x、按需付费以及极高的可扩展性。</li><li><a href="https://supabase.com/docs/guides/database/extensions/pgvector">pgvector：嵌入与向量相似度 | Supabase 文档</a>：pgvector：一个用于存储嵌入并执行向量相似度搜索的 PostgreSQL 扩展。</li><li><a href="https://share.snipd.com/snip/8eb39371-e1c4-4140-9ad1-5981efe3c21b">利用摩尔定律创新数据中心 | 来自 ChinaTalk 的 48 秒剪辑</a>：来自 ChinaTalk 的 48 秒剪辑，关于 Intel 和 Nvidia 的现状检查，涉及 Asianometry、Fabricated Knowledge 和 SemiAnalysis。</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena 排行榜 - Hugging Face Space</a>：未找到描述。</li><li><a href="https://partiful.com/e/VJPFposDqQg2eCqHuL38">报名参加实时语音 AI 和多模态黑客松 | Partiful</a>：各位可爱的黑客朋友们，AI Engineer Foundation（你友好的开源非营利邻居 - 网站：aie.foundation）正在举办一场实时交互/对话式多模态 AI 黑客松...</li><li><a href="https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/">如何与 LLM 对话（通过语音）</a>：用于构建实时 AI WebRTC 应用程序的代码。</li><li><a href="https://x.com/kwindla/status/1777712299215901062">来自 kwindla (@kwindla) 的推文</a>：@latentspacepod 这是来自 @chadbailey59 的视频，展示了快速语音响应 + 工具调用的可能性。</li><li><a href="https://qdrant.tech/documentation/frameworks/semantic-router/#">Semantic-Router - Qdrant</a>：Qdrant 是一个用 Rust 编写的开源向量数据库和向量搜索引擎。它通过便捷的 API 提供快速且可扩展的向量相似度搜索服务。</li><li><a href="https://x.com/karpathy/status/1777481372636246491?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：我添加了一个简短的入门教程，介绍 PyTorch 层如何迁移到 C，并提供了一些可能有所帮助的指引：https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md</li><li><a href="https://openai.com/pricing">定价</a>：简单且灵活。仅为你使用的部分付费。</li><li><a href="https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Liam Bolling (@liambolling) 的推文</a>：🎉 对 Google Gemini 来说是重大的一天。Gemini 1.5 Pro 现在可以理解音频、使用无限文件、执行你的指令，并允许开发者通过 JSON 模式构建令人惊叹的应用！这一切都是免费的。原因如下...</li><li><a href="https://x.com/karpathy/status/1777427944971083809?s=46&t=6FDPaNxZcbSsE">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？没有？好吧，现在你可以了！使用 llm.c：https://github.com/karpathy/llm.c。首先，它实现了 GPT-2 的训练...</li><li><a href="https://x.com/karpathy/status/1777427944971083809?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？没有？好吧，现在你可以了！使用 llm.c：https://github.com/karpathy/llm.c。首先，它实现了 GPT-2 的训练...</li><li><a href="https://x.com/karpathy/status/1777493157485437009?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：顺便说一句，编写 llm.c 训练代码对于 LLM Agent 来说将是一个非常有趣、令人印象深刻、自包含且非常 Meta 的挑战。提示词是：获取 PyTorch 代码 train_gpt2.py 并编写 C 代码...</li>

s://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2Otia">来自 Liam Bolling (@liambolling) 的推文</a>：🎉 对 @Google Gemini 来说是重大的一天。Gemini 1.5 Pro 现在可以理解音频、使用无限文件、根据你的指令行动，并让开发者通过 JSON mode 构建令人惊叹的东西！这一切都是 🆓 的。原因如下...</li><li><a href="https://www.youtube.com/watch?v=PwnlVHFqLdw">AI 驱动的语音患者接诊</a>：了解 AI 驱动的患者接诊如何简化接诊流程，并在临床会诊前提高数据准确性。通过语音患者接诊...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1226901012900741223)** (16 messages🔥): 

- **Mojo 中的 F-strings 仍待实现**：成员们确认 Mojo 尚不支持 `f` 字符串功能。期待 Python 式字符串格式化功能的开发者们仍在等待该特性。
- **探索本地文档命令**：一位用户询问是否有类似 Rust 文档命令的本地下载 Mojo 文档的命令，但目前尚无此类命令。建议他们暂时使用在线文档或克隆 Git 仓库。
- **本地与在线文档的质量对比**：讨论了本地 Git 仓库文档是否与在线文档一样具有结构性且保持最新。为偏好结构化资源的开发者分享了在线 [Mojo 标准库模块](https://docs.modular.com/mojo/lib) 的链接。
- **字符串格式化的临时解决方案**：在等待 `f` 字符串功能期间，一位用户建议使用 C 风格格式化 [from builtin.io import _printf as printf] 进行 Mojo 中的字符串格式化。不过，这种方法将来可能会被弃用。
- **面向初学者的 Mojo API 文档**：一位用户分享了一个 [Notion 站点链接](https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078?pvs=4)，其中包含翻译和总结的 API 文档，旨在帮助初学者。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078?pvs=4">Notion – 笔记、任务、维基和数据库的全能工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为你和你的团队打造的全能工作空间。</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 模块 | Modular 文档</a>：Mojo 标准库中所有模块的列表。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1227009274979287201)** (2 messages): 

- **Modular 分享最新动态**：*Modular* 在 Twitter 上发布了新更新，可以通过此链接查看：[Modular Twitter 更新](https://twitter.com/Modular/status/1777447869907431562)。
- **来自 Modular 的另一条推文**：Discord 频道中分享了 *Modular* 的后续推文。点击查看完整内容：[Modular 最新推文](https://twitter.com/Modular/status/1777737280771514505)。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1226982656659558401)** (1 messages): 

- **为 Mojo 的演进做出贡献**：Mojo 的一个重要里程碑是开源其标准库，并发布了相关[公告](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)。提供的[分步指南](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide)详细介绍了如何为 Mojo 做出贡献，涵盖了从初始设置到创建 Pull Request 的所有环节。
- **Mojo 社区贡献指南**：该指南概述了社区如何参与增强 Mojo，从识别 GitHub issues 到代码贡献。鼓励贡献者同时参考 [贡献指南](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md) 以获取更深入的说明。

**提到的链接**：<a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: 如何为 Mojo 标准库做贡献：分步指南</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何为 Mojo 标准库做贡献：分步指南

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1227072898905149461)** (1 messages): 

- **Karpathy 的新 GitHub 宝库**：AI 大神 Andrej Karpathy 发布了一个 [GitHub 仓库](https://github.com/karpathy/llm.c)，提供了仅用 1000 行纯 C 代码训练 GPT-2 风格模型的代码库。这提供了一种精简的方法，专注于使用原生 C/CUDA 进行 LLM 训练的核心要素。

**提及链接**：<a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。可以通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1226818775412641792)** (33 条消息🔥): 

- **使用 Mojo 提升 Django 性能**：一位成员对在没有 CPython 开销的情况下将 Django 与 Mojo 结合使用的前景感到兴奋，推测 Django 的某些部分可能会用 Mojo 编译或重写，以提高性能。
- **Mojo 中的 Set 和 Collection**：一位用户提出了一个关于 Set 不符合 Mojo 中 CollectionElement trait 的问题，原因是缺少 `__copyinit__` 和 `__del__`，这在创建 Set 字典时带来了挑战。
- **RustPython 见解**：成员们讨论了 [RustPython 项目](https://github.com/RustPython/RustPython)，认可了重新实现 Python stdlib 的巨大努力，并提到其目前与 CPython 相比性能较慢。
- **Mojo 并发原语**：关于 Mojo 的 coroutine 和 async/await 的讨论指出，它们虽然存在但尚未完成；根据 [Mojo 文档](https://docs.modular.com/mojo/stdlib/builtin/coroutine)和[路线图](https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with)，`async for` 和 `async with` 尚未实现。
- **Mojo 编译器 Bug 报告**：一位用户遇到了 Mojo 中 async 函数指针的编译器 Bug，引发了讨论并在 [GitHub 上提交了 Bug 报告](https://github.com/modularml/mojo/issues/2252)。

<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/coroutine">coroutine | Modular Docs</a>：实现了 coroutines 的类和方法。</li><li><a href="https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。</li><li><a href="https://github.com/modularml/mojo">GitHub - modularml/mojo: The Mojo Programming Language</a>：Mojo 编程语言。</li><li><a href="https://github.com/RustPython/RustPython">GitHub - RustPython/RustPython: A Python Interpreter written in Rust</a>：用 Rust 编写的 Python 解释器。</li><li><a href="https://github.com/dorjeduck/llm.mojo">GitHub - dorjeduck/llm.mojo: port of Andrjey Karpathy&#39;s llm.c to Mojo</a>：将 Andrjey Karpathy 的 llm.c 移植到 Mojo。</li><li><a href="https://github.com/modularml/mojo/issues/2252">[BUG] Compiler bug when typing async function pointer call return type · Issue #2252 · modularml/mojo</a>：Bug 描述：Mojo 编译器在对 async 函数指针调用返回类型进行类型推导时出错。预期行为：async fn() -&gt; Int 函数在调用时应返回 Coroutine[Int] 类型。
</li>
</ul>

</div>

---

**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1226791149440864317)** (2 条消息): 

- **对功能开发的协作热情**：一位成员在查看仓库后，表达了对为即将发布的版本贡献新功能开发的兴趣。他们询问了如何进行更详细的讨论，暗示会在仓库中提出 issue 以进行进一步沟通。
- **协调的直接回应**：另一位参与者直接回复了支持信息，并启动了私人对话，推测是为了进一步讨论协作事宜。

---

**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1226929498520424479)** (1 条消息): 

- **别具一格的 Python 星号图案教程**：一位社区成员分享了一个名为“如何在 Python 中打印任何星号图案”的 YouTube 视频，该视频巧妙地通过伪装成简单的 Python 教程向观众介绍了 Mojo。**视频揭示了代码是使用 VSCode 中的 Mojo 插件编写的**，这让许多不知道 Mojo 具备 Python 能力的人感到惊讶。点击[这里](https://youtu.be/6cyCeJwgNjc)享受这个既有教育意义又带点恶作剧性质的视频。

**提及的链接**：<a href="https://youtu.be/6cyCeJwgNjc">如何在 Python 中打印任何星形图案</a>：如果你想了解更多关于 Python、Mojo 甚至使用 Scrum 进行现代软件开发的知识，请订阅我的 newsletter。你不会后悔的！https://www.xenn...

---

**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1227316880541356122)** (2 条消息):

- **请求 SYRK 实现**：一名成员正在询问 **Mojo** 中的 SYRK（对称秩-k 更新）实现，以便进行性能测试。未提供进一步的上下文或细节。

---

**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1226974938720899102)** (20 条消息🔥):

- **进步之火**：**Mojo nightly build** 已发布，用户可以使用 `modular update nightly/mojo` 进行更新。尚未进入稳定版 Mojo 的更改列在它们的 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中，与上一个版本的差异可以在[此处](https://github.com/modularml/mojo/compare/1a8f912..1bce16d)查看。
- **目标是工作日发布 Nightly 版本**：目前 Mojo Nightly 还没有固定的发布计划，但目标是通过改进持续集成（CI）来实现工作日自动发布。
- **解压故障与解决方案**：几位用户在尝试更新时遇到了“Error opening archive: Unrecognized archive format”错误。推荐的解决方案包括运行 `modular clean`，可能还需要更新 `modular` 并确保已安装 `zstd`。
- **音乐表情符号大受欢迎**：在技术讨论之余，成员们还欣赏了紫色火焰表情符号的美感，并提出了将其融入歌词等幽默想法。
- **对高级特性的惊喜**：Mojo 构建中宣布支持 **heterogeneous variadic generics**（异构变长泛型）引发了用户的兴奋和惊讶。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/compare/1a8f912..1bce1">Comparing 1a8f912..1bce1 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/compare/1a8f912..1bce16d">Comparing 1a8f912..1bce16d · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。
</li>
</ul>

</div>

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1226788473428508673)** (26 条消息🔥):

- **Stability AI 发布新模型**：Stability AI 发布了一个名为 [**CosXL**](https://huggingface.co/stabilityai/cosxl) 的模型，该模型要求接受非商业研究社区许可协议，并要求用户同意分享其联系信息。

- **文本转图像 AI 更新**：成员们讨论了如何从 Stable Diffusion 数据库中不存在的文本创建图像。分享了一些链接，包括 [Diffusers 文档](https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image)的链接，并对版本更新至 v0.27.2 进行了修正。

- **自由职业市场分析**：分享了一篇[博客文章](https://bloomberry.com/i-analyzed-5m-freelancing-jobs-to-see-what-jobs-are-being-replaced-by-ai/)，该文章分析了 Upwork 上的 500 万个自由职业岗位，以研究 AI 对特定工作角色的影响。

- **模型训练技术的辩论**：成员们讨论了不同模型训练方法的优缺点。对于在这些过程中使用 EDM schedules 和 offset noise（偏移噪声）的看法各不相同。

- **Deepfloyd Stage 3 前途未卜**：有人询问 **Deepfloyd Stage 3** 是否会按承诺发布，强调项目页面需要就其状态进行清晰的沟通。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>：暂无描述</li><li><a href="https://bloomberry.com/i-analyzed-5m-freelancing-jobs-to-see-what-jobs-are-being-replaced-by-ai/">被 AI 取代的工作 - 500 万个自由职业岗位分析 - bloomberry</a>：毫无疑问，AI 将影响就业。但哪些工作更有可能被取代……
</li>
</ul>

</div>

---

**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1226816556646793297)** (21 条消息🔥):

- **生成模型的新动向**：有人指出，图像模型采用 Autoregression（自回归）而文本模型采用 Diffusion（扩散）具有某种讽刺意味，这凸显了以往趋势的逆转。
- **模型方法的流行周期**：一位成员澄清说，Autoregressive 图像模型一直存在，并因其 Scalability（可扩展性）和可理解性而受到青睐，能够同时预测文本和图像。
- **Autoregressive 模型的潜在优势**：如果具备适当的 Video Tokenization，图像生成的 Autoregressive 方法也可能为 Text-to-Video 模型铺平道路，这与 **CM3leon 论文**中的见解一致。
- **Griffin 胜过 Transformers**：据 [Reddit 上的讨论](https://www.reddit.com/r/MachineLearning/comments/1b3leks/deepmind_introduces_hawk_and_griffin_r/)报道，Google 的 [Griffin 架构](https://www.reddit.com/r/singularity/comments/1bzzreq/google_releases_model_with_new_griffin)表现优于 Transformer，它拥有额外的 10 亿参数，并在长上下文（Long Contexts）下具有更好的 Throughput（吞吐量）。
- **重新评估 Zero-Shot 泛化**：[最近的一篇论文](https://arxiv.org/abs/2404.04125)阐述了 CLIP 等多模态模型中 Zero-Shot 泛化的局限性，其中**数据质量和数量**变得至关重要。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.04125">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>：网络爬取的预训练数据集是多模态模型（如用于分类/检索的 CLIP 和用于图像生成的 Stable-Diffusion）令人印象深刻的 &#34;Zero-Shot&#34; 评估性能的基础...</li><li><a href="https://arxiv.org/abs/2404.03715">Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences</a>：本文研究了使用来自强大 Oracle 的偏好反馈对大语言模型（LLMs）进行 Post-training，以帮助模型实现自我迭代改进。Post-training 的典型方法...</li><li><a href="https://aaronlou.com/blog/2024/discrete-diffusion/">Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou</a>：未找到描述</li><li><a href="https://tenor.com/view/rick-and-morty-that-just-sounds-like-slavery-with-extra-steps-slave-rick-morty-gif-18016642">Rick And Morty That Just Sounds Like Slavery With Extra Steps GIF - Rick And Morty That Just Sounds Like Slavery With Extra Steps Slave - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/singularity/comments/1bzzreq/google_releases_model_with_new_griffin/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>