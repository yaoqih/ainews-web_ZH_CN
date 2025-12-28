---
companies:
- google
- google-deepmind
- youtube
date: '2024-05-14T22:01:01.464489Z'
description: '**谷歌**宣布了 **Gemini 模型系列**的更新，包括支持 **200 万 token** 的 **Gemini 1.5 Pro**，以及针对速度进行了优化、拥有
  **100 万 token 容量**的新型 **Gemini Flash** 模型。Gemini 系列现在涵盖了 **Ultra**、**Pro**、**Flash**
  和 **Nano** 模型，其中 **Gemini Nano** 已集成到 **Chrome 126** 浏览器中。


  其他的 Gemini 功能还包括 **Gemini Gems**（自定义 GPT）、用于语音对话的 **Gemini Live**，以及实时视频理解助手 **Project
  Astra**。**Gemma 模型系列**也更新了拥有 **270 亿参数**的 **Gemma 2**，它在体积仅为一半的情况下，提供了接近 **Llama-3-70B**
  的性能；此外还推出了受 **PaLI-3** 启发的视觉语言开放模型 **PaliGemma**。


  其他发布的内容包括 DeepMind 的 **Veo**、用于生成逼真图像的 **Imagen 3**，以及与 YouTube 合作的**音乐 AI 沙盒 (Music
  AI Sandbox)**。**SynthID 水印技术**现在已扩展至文本、图像、音频和视频。**Trillium TPUv6** 的代号也正式揭晓。此外，谷歌还将
  AI 集成到了其整个产品套件中，包括 Workspace、电子邮件、文档、表格、相册、搜索和 Lens（智慧镜头）。


  *“全世界都在等待苹果的回应。”*'
id: 60e36161-5a94-4ded-b4a9-cb88b377ff37
models:
- gemini-1.5-pro
- gemini-flash
- gemini-ultra
- gemini-pro
- gemini-nano
- gemma-2
- llama-3-70b
- paligemma
- imagen-3
- veo
original_slug: ainews-google-io-in-60-seconds
people: []
title: 60秒看遍 Google I/O
topics:
- tokenization
- model-performance
- fine-tuning
- vision
- multimodality
- model-release
- model-training
- model-optimization
- ai-integration
- image-generation
- watermarking
- hardware-optimization
- voice
- video-understanding
---

<!-- buttondown-editor-mode: plaintext -->**发现 Gemini 的 7 种版本！**

> 2024年5月13日至5月14日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**426** 个频道，**8590** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**782 分钟**。

Google I/O 仍在进行中，由于产品范围极其广泛，报道起来比昨天 OpenAI 半小时的活动要困难得多。目前我们还没有发现一个能总结所有内容的单一网页（除了 [@Google](https://twitter.com/google) 和 [@OfficialLoganK](https://twitter.com/OfficialLoganK/) 的账号）。

以下是主观分类的清单：

**Gemini 模型家族**

- Gemini 1.5 Pro 宣布**支持 2m token**（候补名单中）。[博客文章](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/)提到了“在翻译、编程、推理等关键用例中的一系列质量改进”，但未发布基准测试。
- **发布 Gemini Flash**，在[最初的 3 模型愿景](https://arxiv.org/abs/2312.11805)基础上增加了第四个模型。博客文章称其“针对响应时间至关重要的窄领域或高频任务进行了优化”，强调其 [1m token 容量](https://x.com/Google/status/1790432952767115432)，[价格略低于 GPT3.5](https://x.com/_Mira___Mira_/status/1790448070226030920)，但[未提及速度声明](https://news.ycombinator.com/item?id=40358071)。Gemini 系列目前包括：
  - Ultra：“我们最大的模型”（仅限 [Gemini Advanced](https://techcrunch.com/2024/02/08/google-goes-all-in-on-gemini-and-launches-20-paid-tier-for-gemini-ultra/)）
  - Pro：“我们在通用性能方面表现最好的模型”（今日提供 API 预览版，6 月正式发布/GA）
  - Flash：“我们追求速度/效率的轻量级模型”（今日提供 API 预览版，6 月正式发布/GA）
  - Nano：“我们的端侧模型”（将内置于 [Chrome 126](https://techcrunch.com/2024/05/14/google-is-building-its-gemini-nano-ai-model-into-chrome-on-the-desktop/)）
- [Gemini Gems](https://x.com/Google/status/1790444941451067901) - Gemini 版的自定义 GPTs
- [**Gemini Live**](https://x.com/Google/status/1790444519864795458)：“使用语音进行深度双向对话的能力”，这直接引向了 **Project Astra** —— 具备实时视频理解能力的个人助理聊天机器人，并附带一段[精美的 2 分钟演示视频](https://x.com/Google/status/1790433789811753460)
- [LearnLM](https://x.com/Google/status/1790453655054827679) - “我们基于 Gemini 并针对学习进行了微调的新模型系列”

**Gemma 模型家族**

- [Gemma 2](https://x.com/Google/status/1790452314278412554)，现在最高达 27B（此前为 7B 和 2B），这是一款仍在训练中的模型，以一半的参数量（适配 1 个 TPU）提供了接近 Llama-3-70B 的性能。
![image.png](https://assets.buttondown.email/images/eee89aed-9b00-4e60-aeda-005b3ff69897.png?w=960&fit=max)


- [PaliGemma](https://x.com/Google/status/1790451427464085563) - 他们首个受 [PaLI-3](https://arxiv.org/abs/2310.09199) 启发的视觉语言开放模型，是对 [CodeGemma](https://ai.google.dev/gemma/docs/codegemma) 和 [RecurrentGemma](https://ai.google.dev/gemma/docs/recurrentgemma) 的补充。

**其他发布**

- [Veo](https://x.com/Google/status/1790435689495945479)，DeepMind 对标 Sora 的产品。[HN 上的对比](https://news.ycombinator.com/item?id=40358041)。
- [Imagen 3](https://x.com/Google/status/1790434730623537280)：“它能像人类写作一样理解提示词，生成更具照片感的图像，是我们渲染文本效果最好的模型。”（更多[示例点击此处](https://x.com/GoogleDeepMind/status/1790434750592643331)）
- [Music AI Sandbox](https://x.com/GoogleDeepMind/status/1790435413682975043) - YouTube 与 DeepMind 合作，旨在与 Udio/Suno 竞争
- [SynthID 水印技术](https://x.com/Google/status/1790453029243703658) 现在**扩展到文本**以及图像、音频和视频（包括 Veo）。
- [Trillium - TPUv6 的代号](https://x.com/Google/status/1790436855395078537)


以及在 Google 产品线中的 AI 部署 - [Workspace](https://x.com/Google/status/1790430549649019123), [Email](https://x.com/Google/status/1790441491338264973), [Docs](https://x.com/GoogleWorkspace/status/1790441310123385236), [Sheets](https://x.com/Google/status/1790442954500268164), [Photos](https://x.com/Google/status/1790428759700463632), [Search Overviews](https://x.com/Google/status/1790428396775719053), [具备多步推理能力的 Search](https://x.com/Google/status/1790438800667123860), [Android Circle to Search](https://x.com/Google/status/1790447502107251189), [Lens](https://x.com/Google/status/1790440001156583712)。

总的来说，这是一场执行得非常出色的 I/O，易于总结且不失细节。全世界都在等待 Apple 的回应。

---

**目录**

[TOC] 

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**OpenAI 发布 GPT-4o**

- **核心特性**：[@sama](https://twitter.com/sama/status/1790075827666796666) 指出 GPT-4o 的**价格是 GPT-4-turbo 的一半，速度快两倍**，且拥有 **5 倍的速率限制（rate limits）**。[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1790077783294574967) 强调了它**跨文本、音频和视频进行实时推理**的能力，称其**极其多才多艺且充满趣味性**。
- **多模态能力**：[@gdb](https://twitter.com/gdb/status/1790071008499544518) 强调了 GPT-4o **跨文本、音频和视频的实时推理能力**，认为这是**迈向更自然的人机交互的一步**。
- **改进的 Tokenizer**：[@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1790091535096193458) 提到，**得益于新的 Tokenizer，非拉丁语系语言的性能提升了高达 9 倍，且成本更低**。
- **广泛可用性**：[@sama](https://twitter.com/sama/status/1790065541262032904) 表示 GPT-4o **面向所有 ChatGPT 用户开放，包括免费版用户**，这符合他们让强大的 AI 工具普及化的使命。

**技术分析与影响**

- **架构推测**：[@DrJimFan](https://twitter.com/DrJimFan/status/1790089671365767313) 推测 GPT-4o **将音频直接映射为音频，作为一等模态（first-class modality）**，这需要**全新的 tokenization 和架构研究**。他认为 OpenAI 开发了一种**神经优先（neural-first）的流式视频编解码器，将运动增量（motion deltas）作为 tokens 传输**。
- **与 GPT-5 的潜在关系**：[@DrJimFan](https://twitter.com/DrJimFan/status/1790089671365767313) 建议 GPT-4o 可能是**仍在训练中的 GPT-5 的早期检查点（checkpoint）**，其品牌命名在 Google I/O 前夕透露出了一丝不安。
- **与 Character AI 的重叠**：[@DrJimFan](https://twitter.com/DrJimFan/status/1790089671365767313) 注意到助手的**活泼、调情性格与电影《她》（Her）中的 AI 相似**，并认为 OpenAI 正在**直接与 Character AI 的产品形态竞争**。
- **Apple 集成潜力**：[@DrJimFan](https://twitter.com/DrJimFan/status/1790089671365767313) 概述了 iOS 集成的三个层级：**1) 用端侧 GPT-4o 替换 Siri，2) 针对摄像头/屏幕流的原生功能，3) 与 iOS 系统 API 集成**。他认为第一个与 Apple 合作的公司将从一开始就拥有一个拥有十亿用户的 AI 助手。

**社区反应与迷因 (Memes)**

- [@karpathy](https://twitter.com/karpathy/status/1790373216537502106) 开玩笑说“**LLM 的杀手级应用是斯嘉丽·约翰逊（Scarlett Johansson）**”，而不是数学或其他严肃的应用。
- [@vikhyatk](https://twitter.com/vikhyatk/status/1790242571308155320) 分享了一个 **Steve Ballmer 喊着“developers”的梗图**，质疑现在的科技巨头 CEO 是否还能表现出那种程度的热情。
- [@fchollet](https://twitter.com/fchollet/status/1790375200896512312) 调侃道，随着 **AI 女友的兴起，AI 中的“自我博弈（self-play）”可能终于要变成现实了**，这引用了自 2016 年以来讨论的一个概念。

---

# AI Reddit 综述

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**GPT-4o 的功能与特性**

- **速度与成本**：在 /r/singularity 中，GPT-4o 被指出比 [**GPT-4 Turbo 快 2 倍且便宜 50%，并拥有 5 倍的速率限制**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)。它在 [**非英语语言方面的表现也显著优于 GPT-4 Turbo**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)。
- **音频能力**：GPT-4o 提升了 [**音频解析能力，如区分不同发言者、讲座总结以及捕捉人类情感**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)，同时也 [**改进了音频输出，如表达情感和唱歌**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)。
- **图像生成**：它具有 [**改进的图像生成能力，如更好的文本渲染、角色一致性、字体生成、3D 图像生成以及定向图像编辑**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)。此外，GPT-4o 还具备 [**演示中未展示的能力，如 3D 物体合成**](https://twitter.com/btibor91/status/1790053718416605335)。
- **基准测试**：GPT-4o 在 [**MMLU/HumanEval 基准测试上显示出轻微提升**](https://www.reddit.com/r/singularity/comments/1cr7tvm/gpt4o_features_summary/)。

**GPT-4o 的可用性与定价**

- **ChatGPT 推送**：[**GPT-4o 的文本和图像功能今日起在 ChatGPT 中陆续推出，免费用户可用，Plus 用户拥有 5 倍的消息限制**](https://www.reddit.com/r/singularity/comments/1cr5ao7/plus_users_get_5x_higher_message_limit/)。[**带有 GPT-4o 的语音模式将在未来几周向 Plus 用户推出**](https://www.reddit.com/r/singularity/comments/1cr5ao7/plus_users_get_5x_higher_message_limit/)。
- **定价**：[**GPT-4o 的价格是 GPT-4 Turbo 的一半（$10/1M tokens），比 GPT-4 32K 便宜 12 倍（$60/1M tokens）**](https://www.reddit.com/r/LocalLLaMA/comments/1cr5yce/openai_gpt4o_eval_results_and_llama3400b/)。

**反应与对比**

- **编程性能**：一些人认为 [**GPT-4o 在编程方面不如 GPT-4 Turbo，幻觉更多，不值得那 50% 的折扣**](https://www.reddit.com/r/LocalLLaMA/comments/1crbesc/gpt4o_sucks_for_coding/)。
- **翻译质量**：另一些人指出它 [**在翻译方面并不优于 GPT-4 Turbo**](https://www.reddit.com/r/singularity/comments/1cr734f/tested_gpt4o_in_its_ability_to_translate_its_not/)。
- **基准测试争议**：OpenAI [**声称对比了仍在训练中的 "Llama-3-400B"**](https://www.reddit.com/r/LocalLLaMA/comments/1cr5dbi/openai_claiming_benchmarks_against_llama3400b/)。
- **国际象棋表现**：GPT-4o [**在更难的国际象棋谜题提示词上提升了 +100 ELO**](https://www.reddit.com/r/singularity/comments/1cr9tjt/chatgpt4o_achieves_100_elo_on_harder_prompt_sets/)，达到了 [**1310 ELO**](https://twitter.com/LiamFedus/status/1790064963966370209)。

**开源与竞争对手**

- **Meta 的进展**：[**Meta 表示他们距离追上 GPT-4o 仅有几个月的时间**](https://twitter.com/ArmenAgha/status/1790173578060849601)。
- **Falcon 2 发布**：[**来自阿联酋的开源模型 Falcon 2 已发布，旨在与 Llama 3 竞争**](https://www.reddit.com/r/singularity/comments/1cr2rca/abu_dhabis_technology_innovation_institute/)。
- **Google 的 AI 能力**：在 [**明天的 Google I/O 大会**](https://www.reddit.com/r/singularity/comments/1cr2rca/reminder_google_io_showcase_is_tomorrow_expect/) 之前，[**Google 预热了自家的实时视频 AI 能力**](https://twitter.com/Google/status/1790055114272612771)。

**梗与幽默**

- 人们开玩笑说 GPT-4o 速度太快了，[**应该改名为 "GPT-4ooooooohhhhh"**](https://www.reddit.com/r/singularity/comments/1cr6bpx/gpt4o_is_crazy_fast_they_shouldve_named_it/)。
- 有图片调侃 [**"OpenAI 发明了 Apple 在 1987 年构想的未来主义知识导航仪 (Knowledge Navigator)"**](https://www.reddit.com/r/singularity/comments/1cr7th6/openai_invented_apples_futuristic_knowledge/)。
- 梗图暗示 [**GPT-4o 是在 "Scarlett Johansson 的声音" 上训练的**](https://www.reddit.com/r/singularity/comments/1cr7th6/chatgpt4o_is_trained_on_3_types_of_data/)。

---

# AI Discord 综述

> 摘要之摘要的摘要

## Claude 3 Sonnet

以下是内容中的 3-4 个主要主题，重要的关键词、事实、URL 和示例已加粗：

1. **新 AI 模型发布与对比**：

   - [**OpenAI 的 GPT-4o**](https://openai.com/index/hello-gpt-4o/) 是一个新的旗舰级多模态模型，可以实时处理音频、视觉和文本。与 GPT-4 相比，它拥有更快的响应时间、更低的成本和更强的推理能力。[**展示 GPT-4o 交互能力的示例**](https://www.youtube.com/watch?v=MirzFk_DSiI)。
   - [**Falcon 2 11B**](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas) 模型性能优于 Meta 的 Llama 3 8B，并与 Google 的 Gemma 7B 旗鼓相当，提供多语言和视觉到语言的能力。
   - 尽管存在成本和使用限制方面的担忧，一些用户在处理复杂推理任务时仍然更倾向于使用 [**Claude 3 Opus**](https://www.anthropic.com/legal/aup) 而非 GPT-4o。

2. **AI 模型优化与效率提升**：

   - 在 llm.c 中实现 [**ZeRO-1**](https://github.com/karpathy/llm.c/pull/309) 使 GPU batch size 和训练吞吐量增加了约 54%，从而能够支持更大规模的模型变体。
   - [**ThunderKittens**](https://github.com/HazyResearch/ThunderKittens) 库通过优化的 CUDA tile primitives，承诺为 LLM 提供更快的推理速度和潜在的训练速度提升。
   - 讨论集中在减少 AI 的计算消耗上，并分享了指向 [**Based**](https://www.together.ai/blog/based) 和 [**FlashAttention-2**](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) 等项目的链接。

3. **多模态 AI 应用与框架**：

   - [**AniTalker 框架**](https://x-lance.github.io/AniTalker/) 能够利用音频输入从静态图像创建栩栩如生的说话面孔，捕捉复杂的面部表情。
   - 讨论了 [**Retrieval Augmented Generation (RAG)**](https://medium.com/ai-advances/supercharge-your-llms-plug-and-plai-integration-for-langchain-workflows-d471b2e28c99) 与 [**Stable Diffusion**](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned) 等图像生成模型的集成，利用了 CLIP embeddings。
   - 一个使用 Streamlit、LangChain 和 GPT-4o 的 [**多模态聊天应用**](https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o) 支持在聊天中上传图像和粘贴剪贴板内容。

4. **开源 AI 模型开发与部署**：

   - **Unsloth AI** 庆祝在 Hugging Face 上的模型下载量突破 100 万次，反映了社区的活跃参与。[**使用 Unsloth 创建的新型崇拜克苏鲁的 AI 模型示例**](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/)。
   - **Mojo** 编程语言受到关注，讨论涉及为其开源编译器做贡献、与 MLIR 的集成及其所有权模型。[**关于 Mojo 所有权语义的视频**](https://www.youtube.com/watch?v=9ag0fPMmYPQ)。
   - **LM Studio** 用户讨论了硬件建议、高效推理的量化级别（quant levels），以及 Command R 等特定模型在 Apple Silicon 上的问题。[**关于使用 yi-1.5 等大型模型以获得更好性能的建议**](https://discord.com/channels/1110598183144399058/1195858490338594866/1239596464293023785)。

## Claude 3 Opus

- **GPT-4o 发布，具备多模态能力**：OpenAI 推出了 **GPT-4o**，这是一款支持文本、图像以及即将推出的实时语音/视频输入的新旗舰模型。它[对免费用户限额开放，Plus 用户享有额外权益](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/)，并拥有[更快的响应速度，API 性能提升且成本仅为 GPT-4 的一半](https://www.techopedia.com/openais-gpt-4o-release)。现场演示展示了其[交互式多模态技能](https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky)。

- **Falcon 2 及其他开源模型表现亮眼**：**Falcon 2 11B** 已发布，其[性能超越了 Meta 的 Llama 3 8B，并接近 Google 的 Gemma 7B](https://falconllm.tii.ae/falcon-2.html)，具备开源、多语言和多模态能力。对 Google 即将推出的 27B 开源模型 **Gemma 2** 的期待也在增加。用户讨论了开源与闭源模型的可访问性和未来。

- **Anthropic 的 Opus 政策转变引发争议**：[Anthropic 针对 Opus 的新条款](https://www.anthropic.com/legal/aup)禁止了某些内容类型，引发了褒贬不一的反应。尽管 **GPT-4o** 速度很快，一些人仍然偏好 **Claude 3 Opus**，因其强大的总结能力和更具人性化的输出。

- **Memory 和多 GPU 支持即将登陆 Unsloth**：Unsloth AI 预告了即将推出的功能，如自定义 GPT 的跨会话 Memory 和多 GPU 支持。随着用户探索最佳微调数据集和方法，该平台[庆祝了 1M 次模型下载](https://twitter.com/UnslothAI/status/1790418810597683476)。

- **Modular 的 Mojo 语言通过关键演讲进行扩展**：Modular 分享了关于 Mojo [所有权模型](https://www.youtube.com/watch?v=9ag0fPMmYPQ)和[开源标准库贡献](https://www.youtube.com/watch?v=TJpFSSIts5Q)的教育内容。Mojo 的编译器由 C++ 编写，引发了人们对潜在 MLIR 集成和未来自托管（self-hosting）的兴趣。

## GPT4T (gpt-4-turbo-2024-04-09)

**主要主题**：

1. **AI 模型的进步**：各个频道都在热烈讨论最新的 AI 模型，如 **GPT-4o**、**Falcon 2** 和 **LLaMA 模型**。这些模型拥有增强的能力，如多模态功能和实时处理，并已集成到 Perplexity AI 和 OpenRouter 等平台中。

2. **社区参与和协作**：人们对分享项目、寻求合作以及参与有关代码实践、优化和在新技术集成（如 Stability.ai、Modular 和 LAION 等社区平台内）的讨论表现出越来越浓厚的兴趣，展示了一个专注于共同成长和学习的繁荣生态系统。

3. **定制化与个性化问题**：用户对定制 AI 模型和系统以满足特定需求表现出浓厚兴趣，从设置 AI 工具的私有实例到合并不同模型的能力，反映了将 AI 使用个性化以满足个人或组织要求的持续趋势。

4. **技术挑战与调试**：几个 Discord 频道的共同点是围绕 AI 模型和计算环境的故障排除和问题解决。这包括关于优化模型推理、处理特定库问题以及改进与各种编码环境集成的讨论。

5. **教育内容与资源分享**：多个频道致力于教育内容，从机器学习概念的详细解释到分享教程和资源，帮助成员学习和实现 AI 技术。这不仅有助于技能提升，还在社区内培养了知识共享的文化。

## GPT4O (gpt-4o-2024-05-13)

1. **模型发布与创新**:
   - **GPT-4o**: 许多 Discord 社区都在热烈讨论 OpenAI 发布的 **GPT-4o**，这是一个能够处理文本、音频和视觉输入的多模态模型。该模型在速度、上下文窗口（高达 128K tokens）和整体能力方面都有显著提升。[OpenAI's GPT-4o](https://openrouter.ai/models/openai/gpt-4o) 因其实时多模态能力而受到称赞，但也因一些奇特之处和高昂的使用成本而受到批评 ([GPT-4o Info](https://openai.com/index/hello-gpt-4o/))。
   - **Falcon 2**: 被强调为 Meta 的 Llama 3 8B 和 Google 的 Gemma 7B 的有力竞争模型。它因开源、多语言和多模态特性而受到好评。[Falcon 2 Announcement](https://falconllm.tii.ae/falcon-2.html)。
   - **Claude 3 Opus**: 尽管面临成本和政策方面的担忧，其优势在于处理长篇推理任务和文本摘要。[Claude 3 Opus](https://www.anthropic.com/legal/aup)。

2. **性能与技术讨论**:
   - **GPU 利用率**: 许多讨论围绕着为不同模型（如 **Stable Diffusion**、**YOLOv1**）优化 GPU 使用，以及 **Flash Attention 2** 中的实现技术。这包括指南分享和配置技巧，例如 **ThunderKittens** 在加速推理和训练方面的有效性 ([GitHub - ThunderKittens](https://github.com/HazyResearch/ThunderKittens))。
   - **API 与性能增强**: 关于 API 性能的对话专门集中在优化响应时间和处理更大的上下文窗口。例如，**GPT-4o** API 被指出在降低成本的同时具有更快的速度和更好的性能。

3. **社区工具与支持**:
   - **项目与工具分享**: 从使用 **Retrieval-Augmented Generation** (RAG) 的求职助手，到使用社区开发工具设置 **OpenRouter** 等 AI 工具的详细步骤。个人项目和协作成果得到了大量分享 ([Job Search Assistant Guide](https://www.koyeb.com/tutorials/using-llamaindex-and-mongodb-to-build-a-job-search-assistant), [OpenRouter Model Watcher](https://orw.karleo.net/list))。
   - **帮助与协作**: 一个反复出现的主题是故障排除，并为 AI 开发过程中遇到的问题提供支持，例如 CUDA 错误、模型微调（fine-tuning）和依赖管理。

4. **伦理与政策**:
   - **内容审核与政策**: 围绕 AI 工具的使用和管理政策的伦理担忧，特别是 **Claude 3 Opus** 和 **GPT-4o** 的审核过滤器 ([Anthropic Policy Link](https://www.anthropic.com/legal/aup))。
   - **开源与专有模型**: 讨论经常比较 **Falcon 2** 等开源优势与专有模型的限制，这影响了它们的可访问性和修改。

---

# 第一部分：Discord 高层级摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**GPT-4o 隆重登场**：OpenAI 发布了新模型 **GPT-4o**，部分功能提供免费访问，Plus 用户则享有额外权益，包括更快的响应速度和更广泛的功能。**GPT-4o** 的独特之处在于能够实时处理音频、视觉和文本，标志着多模态应用迈出了重要一步；目前文本和图像输入已可用，语音和视频功能将很快推出。[了解更多关于 GPT-4o 的信息](https://openai.com/index/hello-gpt-4o/)。

**Claude 摘得复杂任务桂冠**：在社区内部，与 **GPT-4o** 相比，**Claude Opus** 被认为在处理复杂、长篇推理方面更具优势，尤其是在处理大量原始内容时。用户对 Google 和 OpenAI 未来增强功能（包括更宽的上下文窗口和先进的语音能力）寄予厚望。

**自定义 GPTs 等待记忆功能升级**：备受期待的自定义 GPTs 跨会话上下文记忆功能仍在开发中，官方保证一旦发布，创作者将可以为每个 GPT 配置记忆功能。目前 GPT-4o 的状态是速度提升和 API 性能稳定，尽管 Plus 用户受益于更高的消息限制，但所有人都在热切期待其在自定义 GPT 模型中的承诺集成。

**提示工程 (Prompt Engineering) 暴露模型缺陷**：用户在引导 **GPT-4o** 执行创意和空间感知任务时面临挑战，指出其在迭代图像生成方面存在困难，并提到 Gemini 1.5 安全过滤器带来的特定内容审核问题。尽管 GPT-4o 加快了响应速度，但偶尔在理解和执行上出现失误，表明根据用户反馈进行迭代改进仍有空间。

**寻求受监控的 ChatGPT 克隆版**：一名成员询问如何创建一个类似 **ChatGPT 的应用程序**，允许组织使用 GPT-3.5 模型对消息进行监控。这反映了在正式生态系统中对可定制和可控 AI 工具日益增长的需求。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**GPT-4 的 Token 之争**：围绕 **GPT-4** 的 Token 容量存在争论，澄清指出 **GPT-4** 较大的上下文窗口适用于特定模型，如拥有 128K Token 上下文窗口的 GPT-4o。一些用户正在深入研究 **GPT-4o** 的功能，注意到其极高的速度和卓越的性能，并分享了其实时推理的 [视频示例](https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky)。

**政策变动引发热议**：**Anthropic** 修订后的 **Opus** 服务条款将于 6 月 6 日生效，由于禁止创建 LGBTQ 内容等限制，在成员中引起了骚动。政策详情可见分享的 [Anthropic 政策链接](https://www.anthropic.com/legal/aup)。

**Claude 坚守阵地**：尽管 **GPT-4o** 备受关注，但对于一些用户来说，**Claude 3 Opus** 仍然是文本摘要和类人回答的首选，尽管存在成本和使用限制方面的担忧。

**Perplexity 的新实力派**：用户正在测试 **GPT-4o** 在 Perplexity 工具中的集成，强调其高速、深度的响应。Pro 版本每天允许 600 次查询，与其 **API 可用性**相呼应。

**API 配置难题**：关于 **Perplexity API 设置**的讨论浮出水面，一位用户询问了使用 **llama 模型**处理长输入时的超时问题。一名成员指出 **llama-3-sonar-large-32k-chat** 的对话模型针对对话上下文进行了微调，但尚未就最佳超时设置达成共识。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**LLaMA 指令微调建议**：根据用户讨论，对于在小数据集上进行微调，如果性能不理想，在考虑基座模型（base model）之前，建议先从 Llama-3 的**指令模型（instruction model）**开始。他们建议通过迭代来找到最适合你场景的方案。

**ThunderKittens 超越 Flash Attention 2**：据社区讨论，**ThunderKittens** 在速度上超过了 **Flash Attention 2**，有望实现更快的推理速度，并可能在训练速度上取得进展。代码已在 [GitHub](https://github.com/HazyResearch/ThunderKittens) 上发布。

**Typst 的合成数据集构建**：为了在 "Typst" 上有效地微调模型，工程师们提议合成 50,000 个示例。生成大规模合成数据集这一艰巨任务被视为取得进展的基础步骤。

**Unsloth AI 上的多模态模型扩展**：Unsloth AI 预计即将支持**多模态模型**，包括预计下周推出的多 GPU 支持，为实现新的强大 AI 能力奠定基础。

**为 Unsloth AI 喝彩**：AI 社区庆祝 **Unsloth AI** 在 **Hugging Face** 上的模型下载量突破一百万次，这标志着一个被用户认可的里程碑，反映了社区的积极参与和支持。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **苹果旨在 AI 集成，对 Google 持观望态度**：在技术讨论中，关于**苹果传闻与 OpenAI 达成协议**将 ChatGPT 引入 iPhone 的猜测不断，引发了关于本地模型与云端模型方案的对比。工程师们对这种集成的可行性表示怀疑，一些人怀疑苹果是否有能力在手持设备上高效运行重量级模型。

- **Falcon 2 翱翔群雄之上**：**Falcon 2 模型**因其性能获得赞赏，它拥有开源、多语言和多模态能力，在性能上超越了 Meta 的 Llama 3 8B 等竞争对手，仅略微落后于 Google Gemma 7B 的基准测试。Falcon 2 既开源又在多个领域表现优异的消息引发了广泛关注 [Falcon LLM](https://falconllm.tii.ae/falcon-2.html)。

- **GPT-4o 引起轰动**：围绕 OpenAI 最新模型 **GPT-4o** 的讨论充满了惊叹和抱怨，该模型展示了更快的响应速度和引人注目的免费聊天功能。尽管对其“出口成章”的快速能力感到兴奋，但批评主要集中在其品牌命名和性能问题上——特别是延迟。

- **语音遇上视觉，开启 AI 交互新篇章**：ChatGPT 语音和视觉集成的演示引起了关注，展示了实时、情感敏感的 AI 交互。社区对演示能力的真实性存在疑虑，试图探究此类展示背后潜在的运行机制。

- **API 期待与竞争格局**：讨论围绕访问 **GPT-4o 的 API** 展开，工程师们对其迅捷的性能充满期待。暗流反映了更宏大的 AI 战场，Google 和其他玩家正在对 OpenAI 凭借 GPT-4o 发起的攻势做出反应——而社区正在观察，等待着在新的 API 上大显身手。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM 难以突破 8k Token**：一些成员反映在生成超过 8k Token 的连贯输出时，**Llama 3 70b** 面临挑战。虽然已取得显著成功，但在处理更大 Token 流程方面仍有改进空间。
- **体验 GPT-4o 的起伏**：OpenAI GPT-4o 模型发布后，服务器内评价褒贬不一。一些人注意到其独特能力，包括实时多模态功能和中文 Token 处理，而另一些人则审视其局限性，特别是图像编辑模式和成本效率。
- **远程自动化：比你想象的更棘手**：社区分享了在远程桌面协议 (Remote Desktop Protocol) 内运行软件自动化的经验和想法，展示了 AI 领域的复杂性。从解析文档对象模型 (DOM) 到逆向工程软件，对话展示了自动化过程中复杂的导航和决策路径。
- **租用还是购买：LLM 的 GPU 配置选择**：社区针对租用与购买用于大语言模型 (**LLMs**) 的 GPU 配置的优缺点进行了热烈辩论。对话深入探讨了成本效益、隐私考量和硬件规格，并广泛探索了 GPU 供应商和配置方案。
- **GPT-4o 多模态功能揭秘**：社区成员深入交流了该模型的多模态特性，特别是音频输入的 whispering latents 和非英语语言的 Token 处理。社区还指出了理解最长中文 Token 的资源以及 GPT-4o 等模型中分词器 (tokenizer) 的改进。
- **WorldSim 图像大放异彩**：WorldSim 用户公开赞赏该程序的创造力。一位成员甚至提到考虑纹一个受该艺术作品启发的纹身，表达对 WorldSim 视觉效果的认可。
- **IBM/Redhat 推动 LLM 更进一步**：IBM/Redhat 扩展 LLM 知识库和能力的方案成为热点。他们的项目以连续体形式吸收新信息，实时应用而非在每次知识扩展后进行完整重训练，为模型的增量演进提供了一种创新方法。
- **研究人员寻求人类/LLM 文本对进行模型对比评估**：讨论中提出了提取针对相同 Prompt 的 'human_text' 和 'llm_text' 对比数据集的需求，旨在对 LLM 响应与人类语言输出进行更深层次的比较和评估。
- **通过开源项目贡献丰富 AI 知识**：社区再次重申了向 IBM/Redhat 的 **Granite** 和 **Merlinite** 等开源项目进行贡献的可行性和重要性，这是迈向技术变革未来开源协作的一步。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability.ai 的 CEO 变动**：讨论集中在 CEO Emad 离职后 **Stability AI** 不确定的未来，以及 **SD3** 模糊的发布状态，包括它是否可能变成付费服务。
- **Stable Diffusion 的 GPU 大对决**：工程师们辩论了运行 **Stable Diffusion** 的最佳 GPU，达成共识认为显存 (VRAM) 更大的 GPU 更合适，并分享了一份详尽的 [风格和标签指南](https://docs.google.com/document/d/e/2PACX-1vQMMTGP3gpYSACITKiZUE24oyqcZD-2ZcvFC92eXbxJcgHGGitde1CK0qgty6CvDxvAwHY9v44yWn36/pub)。
- **使用 BrushNet 提升局部重绘 (Inpainting) 效果**：推荐通过 [ComfyUI BrushNet 的 GitHub 仓库](https://github.com/nullquant/ComfyUI-BrushNet) 集成 BrushNet，利用 brush 和 powerpaint 功能组合来改进 Stable Diffusion 的局部重绘效果。
- **保持 AI 角色一致性的策略**：维持 AI 角色一致性的技术受到热议，重点在于 **LoRA** 和 **ControlNet**，以及创建详细 [角色卡 (character sheets)](https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/) 的资源。
- **科技巨头 vs. 开源社区模型**：谷歌的 Imagen 3 引发了讨论，反映出人们对 SD3 等开源模型的期待与偏好，这源于社区的可访问性。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **微调与 VPN 变通方案**：工程师确认，如果存储在 Hugging Face 上的**微调模型（fine-tuned model）**是公开的且使用 GGUF 格式，则可以通过 LM Studio 访问。此外，针对 Hugging Face 被屏蔽导致的网络错误，建议使用 **VPN**，并指出特定地区的限制，推荐使用 IPv4 连接。

- **模型性能讨论**：社区讨论了**模型合并策略**，例如应用 **unsloth** 的方法来合并和升级 **llama3 和/或 mistral**。此外，还围绕模型的不同**量化级别（quant levels）**展开了辩论，强调低于 Q4 的任何级别都被认为是低效的。

- **软件兼容性与硬件**：讨论指出了一些不兼容情况，例如 **Command R 模型**在 **Apple M1 Max** 系统上的输出问题，以及 **RX6600 GPU** 的 **ROCM 限制**导致 LM Studio 和 Ollama 出现问题。在硬件方面，讨论倾向于推荐 **Nvidia 3060ti** 作为 LM Studio 应用中性价比最高的 GPU，并强调了显存（VRAM）速度对高效 LLM 推理的重要性。

- **LM Studio 功能集与支持**：有关于 LM Studio 中**多模态（multimodal）**功能的提问，特别是关于与标准模型功能一致性的问题。此外，用户表达了对 **Intel GPU 支持**的兴趣，一名 Intel 员工提议协助进行 SYCL 集成，预示着潜在的性能提升。

- **反馈、期望与未来方向**：对 LMS 目前的**实时学习（realtime learning）**能力存在批评性反馈，用户要求至少提供用于逐行训练的差异文件（differential file）。另一位用户建议部署更大的模型，如 **command-r+** 或 **yi-1.5**，以获得更好的效果。

- **部署考量**：一名成员评估了 **Meta-Llama-3-8B-Instruct-Q4_K_M** 模型相对于 GPU 而言较高的 RAM 占用，并在成本效益的背景下权衡了 AWS 和商业 API 之间的部署选项。考虑到模型大小和参数的显著差异，他们比较了使用 IaaS 提供商与订阅 LLMaaS 的潜在成本节省。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**YOCO 降低 GPU 需求**：[YOCO 论文](https://arxiv.org/abs/2405.05254)介绍了一种新的 **decoder-decoder 架构**，该架构在保持全局注意力（global attention）能力的同时，降低了 GPU 内存占用并加快了 prefill 阶段。

**当 NLP 与 AI 故事创作碰撞**：研究人员正从 [Awesome-Story-Generation GitHub 仓库](https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file)中获取资源，为 AI 故事生成的综合研究做出贡献，例如旨在增加故事复杂性的 **GROVE 框架**。

**Stable Diffusion 进军 DIY 领域**：一门超过 30 小时的 Fast.ai 课程正在教授**从零开始构建 Stable Diffusion**，该课程与 Stability.ai 和 Hugging Face 的业内人士合作，同时还讨论了 **sadtalker 安装**以及 **Transformer Agent** 的实际用途。

**OCR 质量前沿**：一系列 [OCR 质量分类器](https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3)展示了使用小型模型区分清晰文档和噪声文档的可行性。

**Stable Diffusion 与 YOLO**：现已提供一份使用 Diffusers 的 [HuggingFace Stable Diffusion 指南](https://huggingface.co/blog/stable_diffusion)，讨论围绕使用 ResNet18 的 **YOLOv1 实现**展开，重点在于平衡数据质量和数量以提升模型性能。

**前沿领域的复杂情绪**：GPT-4o 的发布在社区内引起了不同的反应，引发了关于区分 AI 与人类的担忧，同时成员们报告了在创建自定义分词器（tokenizer）和专注于丰富示例提示（example-rich prompts）的 NLP 策略方面的不同成功经验。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**新型多模态模型席卷 OpenRouter**：OpenRouter 扩大了其阵容，推出了支持文本和图像输入的 **[GPT-4o](https://openrouter.ai/models/openai/gpt-4o)** 以及 **[LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b)**。此外，名单现在还包括 **DeepSeek-v2 Chat**、**DeepSeek Coder**、**Llama Guard 2 8B**、**Llama 3 70B Base**、**Llama 3 8B Base**，其中 [GPT-4o 的最新版本日期为 2024 年 5 月 13 日](https://openrouter.ai/models/openai/gpt-4o-2024-05-13)。

**Beta 测试火热进行中**：一个**高级研究助手和搜索引擎**正在进行 Beta 测试，提供包括 **Claude 3 Opus** 和 **Mistral Large** 等领先模型的付费访问权限，平台还分享了用于试用的 [促销代码 RUBIX](https://rubiks.ai/)。

**GPT-4o 的热情与审视**：关于 **GPT-4o API 定价**（每 1M tokens $5/15）的热烈讨论引发了兴奋，而对其多模态能力的推测也激起了好奇心，评论者指出 OpenAI 的 API 缺乏原生的图像处理能力。

**社区对 OpenRouter 故障的反馈**：用户反映了 OpenRouter 的技术困难，指出了诸如空响应以及来自 MythoMax 和 DeepSeek 等模型的错误。**Alex Atallah** 澄清说，OpenRouter 上的大多数模型都是 FP16，只有少数是量化（quantized）后的例外。

**通过社区工具建立工程连接**：一个由社区开发的用于筛选 OpenRouter 模型的工具受到了好评，并讨论了整合 ELO 分数和模型添加日期等额外指标的建议。提供了相关资源的链接，如 [OpenRouter API Watcher](https://orw.karleo.net/list)。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**GPT-4o 领跑前沿**：[OpenAI 的 GPT-4o](https://x.com/liamfedus/status/1790064963966370209?s=46) 树立了 **AI 能力的新标杆**，特别是在推理和编程方面，主导了 LMSys 竞技场，并由于 [tokenizer 更新](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8) 实现了 token 容量翻倍。它的**多模态威力**也得到了展示，包括潜在的唱歌能力，引发了关于 AI 演进及其竞争格局的兴趣和辩论。

**PPO 框架下的 REINFORCE**：AI 社区讨论了来自 Hugging Face 的一个新 PR，该 PR 将 **REINFORCE 定位为 PPO 的子集**，详见[相关论文](https://arxiv.org/pdf/2205.09123)，展示了在强化学习领域的积极贡献。

**AI 的银幕映射现实担忧**：社区内的对话与电影《她》（Her）产生共鸣，强调了 AI 交互如何被视为**平庸或深远**。这些讨论与对 AI 领导地位和技术人性化的看法紧密相连。

**长期 AI 治理初现端倪**：受 [John Schulman 的演讲](https://www.youtube.com/watch?v=1fmcdz2EO_c) 启发，前瞻性对话暗示项目管理机器人（PRMs）在指导长期 AI 任务中发挥着关键作用。

**评估 AI 评估**：一篇详细的[博客文章](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation)引发了关于大语言模型（LLM）评估的可访问性和未来的思考，讨论了从 **MMLU 基准测试到 A/B 测试**的各种工具及其对学术界和开发者的影响。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**MLP 可能夺冠**：有传言称 **基于 MLP 的模型** 可能会在视觉任务中超越 **Transformers**，一种新的混合方法带来了激烈的竞争。一项特定的 [研究](https://arxiv.org/abs/2108.13002) 强调了 MLP 的效率和可扩展性，尽管对其复杂性仍有一些疑问。

**搞定初始化**：关于神经网络（尤其是 MLP）中 **初始化方案** 关键性的讨论浮明，有建议认为初始化的创新可以带来巨大的改进。有人提出了通过图灵机创建初始化的想法，探索 [Gwern 网站](https://gwern.net/note/fully-connected#initialization) 上提到的合成权重生成的前沿领域。

**模拟初始化（Mimetic Initialization）作为游戏规则改变者**：一篇推广 **模拟初始化** 的论文出现，主张该方法能提升处理小数据集的 Transformers 的性能，从而获得更高的准确率并缩短训练时间，详见 [MLR 论文集](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf)。

**可扩展性探索继续**：深入讨论了 **MLPs** 在各种硬件上的 **Model FLOPs Utilization** (MFU) 是否能超越 Transformers，暗示即使是微小的 MFU 提升也可能在大规模应用中产生共鸣。

**思考 NeurIPS 投稿**：有人呼吁进行最后的 **NeurIPS 投稿**，一名成员表示对类似于 *Othello 论文* 的话题感兴趣。另一场讨论询问了模型压缩对专门特征的影响及其与训练数据多样性的关系。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**新官上任：Mojo 编译器开发升温**：工程讨论显示了对贡献 **Mojo 编译器** 的浓厚兴趣，尽管它尚未开源。编译器辩论还揭示了它是用 **C++** 编写的，而在 **Mojo 中重建 MLIR** 的愿景激发了贡献者的好奇心。

**MLIR 与 Mojo 结盟**：剖析了 **Mojo 与 MLIR** 之间的集成特性，强调了 Mojo 与 MLIR 的兼容性如何能在未来实现自托管编译器。现在鼓励向 Mojo 标准库提交贡献，Modular 工程师 Joe Loser 的 [操作视频](https://www.youtube.com/watch?v=TJpFSSIts5Q) 阐明了这一过程。

**前沿日程**：宣布了 5 月 20 日即将举行的 **Mojo 社区会议** 详情，旨在让开发者、贡献者和用户关注 Mojo 的发展轨迹。分享了一份有用的 [会议文档](https://modul.ar/community-meeting-doc) 以及通过 [社区会议日历](https://modul.ar/community-meeting) 添加活动的选项以便协调。

**夜间是代码的好时光**：`mojo` 的 Nightly 版本发布现在更加频繁，这是一个受欢迎的积极更新计划，旨在将“每晚发布 Nightly”从梦想变为现实。然而，嵌套数组中的段错误（segfault）问题仍存争议，并且有讨论调整发布频率以避免用户对编译器版本的混淆。

**编码难题与编译器对话**：在数字长廊中，开发者们讨论了从如何在 Mojo 中将参数限制为浮点类型（建议使用 `dtype.is_floating_point()`）到 Python 的可变默认参数，以及使用 FFI 从 Mojo 调用 C/C++ 库等话题。关于 [Mojo 中的 FFI](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi) 主题的更多细节通过 GitHub 链接分享。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**ZeRO-1 扩展提升训练吞吐量**：实现 **ZeRO-1** 优化将单 GPU batch size 从 4 增加到 10，并将训练吞吐量提高了约 54%。有关合并及其效果的详细信息可以在 [PR 页面](https://github.com/karpathy/llm.c/pull/309)查看。

**ThunderKittens 引发关注**：讨论中提到了对 **HazyResearch/ThunderKittens** 的兴趣，这是一个 CUDA tile primitives 库，因其在优化 LLM 方面的巨大潜力而备受关注，并被拿来与 Cutlass 和 Triton 工具进行比较。

**Triton 通过 FP 增强获得提升**：Triton 的更新包括 **FP16 和 FP8** 的性能改进，如基准测试数据所示：“Triton [FP16]” 在 N_CTX 为 1024 时达到 252.747280，“Triton [FP8]” 在 N_CTX 为 16384 时达到 506.930317。

**CUDA 流程简化，但仍存疑问**：关于在 PyTorch 中集成自定义 CUDA kernel，分享了一些资源，包括一个解决基础问题的 [YouTube 讲座](https://youtu.be/4sgKnKbR-WE?si=00-k8KV5ESxqks3h)，同时也指出了 `clangd` 解析 `.cu` 文件以及 cuSPARSE 中函数开销（overhead）等问题。

**精细化 CUDA CI 流水线**：辩论了在持续集成（CI）中进行 GPU 测试的必要性，并推介了 GitHub 最新的 GPU runner 支持，认为这是构建稳健流水线所急需的更新。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **通过新用例玩转 Llama**：一套新的 cookbook 展示了 **Llama 3** 的七种不同用例，详见最近黑客松的庆祝帖子；cookbook 可在[此处](https://t.co/YLlsvkI0Ku)访问。
  
- **GPT-4o 零日集成**：随着 **GPT-4o** 从发布之初就获得 Python 和 TypeScript 的支持，开发者们热情高涨，[此处](https://t.co/CMQ1aOXeWb)详细说明了通过 `pip` 安装的指令，并强调了其多模态能力。

- **多模态奇迹与 SQL 速度**：**GPT-4o** 的一个引人注目的多模态演示已上线，同时透露 **GPT-4o** 在 SQL 查询效率上超过了 GPT-4 Turbo；演示见[此处](https://t.co/yPMeyookRq)，性能详情见[此处](https://t.co/5k1tvKklGA)。

- **融合 LlamaIndex 元数据与错误处理**：在讨论中，明确了 **metadata filtering** 可以由 LlamaIndex 管理，但对于 URL 等特定内容需要手动包含；此外，还建议通过在解析前检查网络响应来排查 `Unexpected token U` 错误。

- **AI 求职变得更智能**：一个使用 **LlamaIndex 和 MongoDB** 构建 AI 驱动求职助手的教程和仓库已发布，旨在通过 Retrieval-Augmented Generation（RAG）提升求职体验，文档见[此处](https://www.koyeb.com/tutorials/using-llamaindex-and-mongodb-to-build-a-job-search-assistant)。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Falcon 2 超越 Llama 3**：[Falcon 2 11B 模型](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas)在 Hugging Face 排行榜上超越了 Meta 的 Llama 3 8B，展示了多语言和 vision-to-language 能力，并可与 Google 的 Gemma 7B 媲美。
- **GPT-4o 打破响应壁垒**：OpenAI 发布了 [GPT-4o](https://www.techopedia.com/openais-gpt-4o-release)，以实时通信和视频处理著称；该模型在降低成本的同时提升了 API 性能，达到了人类的对话速度。
- **RAG 遇上图像建模**：围绕 RAG 与图像生成模型集成的讨论重点介绍了用于文本驱动图像转换的 [RealCustom](https://arxiv.org/abs/2403.00483)，并提到了 [Stable Diffusion](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned) 采用 CLIP 图像 embedding 代替文本。
- **混元 DiT：腾讯的中文艺术专家**：腾讯推出了 [HunyuanDiT](https://huggingface.co/spaces/multimodalart/HunyuanDiT)，该模型声称在中文文本生成图像方面达到 state-of-the-art 状态，尽管体积较小，但证明了其对 prompt 的忠实度。
- **AniTalker 用音频驱动肖像动画**：[AniTalker 框架](https://x-lance.github.io/AniTalker/)发布，支持利用提供的音频从静态图像创建逼真的说话面孔，捕捉细微的面部表情，而不仅仅是口型同步。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**GPT-4o 超过前代**：社区爱好者注意到 **GPT-4o** 不仅速度更快（达到 **100 tokens/sec**），而且比之前的版本更具成本效益。人们对其与 Open Interpreter 的集成特别感兴趣，并提到使用命令 `interpreter --model openai/gpt-4o` 运行非常流畅。

**Llama 望尘莫及**：在体验了 **GPT-4** 的性能后，一位成员分享了对 **Llama 3 70b** 的不满，同时也对 OpenAI 的高昂成本表示担忧，仅一天就花费了 20 美元。

**苹果的沉默可能推动开源 AI**：关于苹果是否会将 AI 集成到 MacOS 的猜测层出不穷，一些成员对此表示怀疑，并更倾向于开源 AI 解决方案，这暗示了社区中 Linux 使用率可能会上升。

**等待 O1 的下一次飞行**：对某未命名项目的 TestFlight 发布充满期待，成员们分享了关于在 Xcode 中设置测试环境和编译项目的建议与说明。

**迈向 AGI 之路**：关于 **Artificial General Intelligence (AGI)** 进展的热烈讨论正在进行，参与者交流了想法和资源，包括一个揭示该前沿领域的 [Perplexity AI 解释](https://www.perplexity.ai/search/ELI5-what-AGI-1Q1AM436TE.qHZyzUWHhyQ)。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**ChatGPT 动摇的信念**：工程师们注意到 **ChatGPT** 现在有时会自相矛盾，偏离了以往回答的一致性。人们对该工具在维持稳定推理逻辑方面的可靠性表示担忧。

**LangChain 故障排除仍在继续**：在 **LLCHAIN** 弃用后，工程师们已转向使用 `from langchain_community.chat_models import ChatOpenAI`，但在流式传输和顺序链（sequential chains）方面面临新挑战。**LangChain** Agent 的调用时间较慢（尤其是处理大型输入时），引发了关于利用并行处理来缩短处理时间的讨论。

**AI/ML GitHub 仓库备受关注**：社区交流了最喜欢的 **AI/ML GitHub repositories**，**llama.cpp** 和 **deepspeed** 等项目被提及。

**Socket.IO 加入战场**：一位工程师贡献了关于使用 `python-socketio` 实时流式传输 **LLM responses** 的指南，演示了处理流式传输和确认（acknowledgments）的客户端-服务器通信。

**AI 风格的项目展示**：分享的项目包括一篇关于 **Plug-and-Plai** 集成的 [Medium 文章](https://medium.com/ai-advances/supercharge-your-llms-plug-and-plai-integration-for-langchain-workflows-d471b2e28c99)、一个利用 **Streamlit** 和 **GPT-4o** 的 [多模态聊天应用](https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o)、一个针对 **ChromaDB RAG application** 的生产级扩展查询，以及一个正在开发中的 [Snowflake 成本监控和优化工具](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064)。

**聊天功能赋能博客交互**：分享了一篇讨论如何使用 **Retrieval Augmented Generation (RAG)** 在博客内容上实现活跃对话的文章，进一步激发了在网站上集成高级 AI 聊天功能的兴趣。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**博客平台对决**：用户辩论了 **Substack** 与 **Bluesky** 在博客需求方面的优劣，结论是虽然 **Bluesky** 支持线程（threads），但缺乏全面的博客功能。

**降低 AI 算力消耗**：重点在于最小化 **AI compute usage**，分享了指向 [Based](https://www.together.ai/blog/based) 和 [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) 等倡议的链接，这些技术正在为更高效的 AI 运行铺平道路。

**依赖项困境**：成员们对过时的依赖项（包括 **peft 0.10.0** 等）感到苦恼，并正在手动调整它们以实现兼容性，同时无奈地呼吁通过 GitHub Pull Request 来纠正这种情况。

**CUDA 难题**：有报告称一名成员在 8xH100 GPU 环境中遇到 **CUDA errors**，后来通过切换到 **社区 axolotl 云镜像** 缓解了该问题。

**QLoRA 模型合并与持续训练**：出现了关于在不损失精度的情况下将 **QLoRA 与 base models** 集成的咨询和讨论。此外，对话集中在如何使用 `ReLoRACallback` 从 checkpoints 恢复训练的机制上，正如 [OpenAccess-AI-Collective axolotl 仓库](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=121d871c-06a2-4494-ab29-60a3a419ec5e) 中所记录的那样。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**语音助手不全是笑声**：技术社区对**语音助手的“咯咯笑”功能**感到困惑，认为在专业用途中该功能不合适且容易让人分心。通过重新表述命令等变通方法可以抑制这一怪癖。

**对 GPT-4o 书籍识别任务的评价褒贬不一**：GPT-4o 枚举书架上显示的书籍的能力受到了褒贬不一的评价，准确率仅为 50%，尽管其速度和价格极具竞争力，但仍有改进空间。

**AGI 炒作引发争论**：人们对即将到来的 **Advanced General Intelligence (AGI)** 持怀疑态度，因为从 GPT-3 到 GPT-4 的飞跃中观察到了边际收益递减，而 GPT-5 的热度掩盖了当前模型的改进。

**GPT-4 的长期影响仍不明朗**：对 GPT-4 及其迭代版本的长期影响预测仍处于推测阶段，工程社区仍在探索其全方位的潜力。

**Simon 推特发布 LLM 见解**：[Simon W 的 Twitter 更新](https://twitter.com/simonw/status/1790121870399782987)可能是讨论大语言模型最新发展和挑战的有力催化剂。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 的 CUDA 问题**：关于在 Nvidia 4090 上使用 `CUDA=1` 和 `PTX=1` 的咨询，在出现 PTX 生成错误后，建议将 Nvidia 驱动更新至 550 版本。

- **tinygrad 中 GNN 的潜力**：将 tinygrad 中图神经网络 (GNNs) 的实现与 [PyG](https://www.pyg.org/) 方案进行了比较，并提到了一个可能具有平方时间复杂度的 CUDA kernel，提供了 [GitHub 代码](https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cuda/radius_cuda.cu)以供参考。

- **tinygrad 中的聚合烦恼**：一位用户分享了一个用于特征聚合的 Python 函数 [test_aggregate.py](https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49)，并强调了在反向传播过程中使用高级索引和 `where` 调用的困难；masking 和 `einsum` 函数成为了可能的解决方案。

- **高级索引问题**：tinygrad 的高级功能如 `setitem` 和 `where` 不支持高级索引（使用列表或张量），引发了对替代方法的讨论，包括使用 masking 和 einsum。

- **tinygrad 的卷积困扰**：在 tinygrad 中优化 conv2d 反向传播的尝试遇到了 scheduler 和 view 变化的障碍，引发了关于重新实现 conv2d 是否能解决形状兼容性问题的讨论。




---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **德语 TTS 需要输入**：一名成员呼吁协助创建一个提供高质量播客、新闻和博客的德语 YouTube 频道列表，用于训练德语文本转语音 (TTS) 系统。

- **MediathekView 作为 TTS 数据源**：参与者讨论了利用 [MediathekView](https://mediathekview.de/) 获取德语媒体的实用性，该工具能够下载字幕文件，被推荐用于策划 TTS 训练内容。

- **探索 MediathekView 数据下载和 API**：讨论中提到整个 MediathekView 数据库可能可以下载，并有 JSON API 用于内容访问；提到了相关工具的 [GitHub 仓库](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua)。

- **新德语 Tokenizer 受到推崇**：一名成员关注了 "o200k_base" Tokenizer 的效率，它处理德语文本所需的 Token 比之前的 "cl100k_base" 更少，并将其与 Mistral 和 Llama3 等已知 Tokenizer 进行了对比，但未分享该观点的具体链接。
  
- **分享 Tokenizer 研究和训练资源**：对 Tokenizer 研究感兴趣的人被引导至 [Tokenmonster](https://github.com/alasdairforsythe/tokenmonster)，这是一个非贪婪的子词 Tokenizer 和词汇训练工具，兼容多种编程语言。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**社区等待支持**：**Cohere** 公会的成员报告了支持响应延迟的问题，一名用户在 <#1168411509542637578> 和 <#1216947664504098877> 中表达了这一诉求。官方回复承诺会有活跃的支持人员，并请求提供更多细节以协助解决。

**Command R RAG 备受瞩目**：一位工程师对 **Command R 的 RAG** (Retriever-Augmented Generation) 能力留下了“极其深刻的印象”，称赞其即使在处理冗长的源材料时也具有极高的性价比、精确度和保真度。

**项目分享中的协作呼吁**：在 **#[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1239921278937333771)** 频道中，成员 Vedang 表示有兴趣与另一位工程师 Asher 在类似项目上进行合作，突显了社区的协作精神。

**成员传播其 Medium 影响力**：Amit 分享了一篇 [Medium 文章](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2)，深入探讨了如何通过 Unstructured API 使用 RAG，旨在从 PDF 中提取结构化内容——这对于处理文档处理的工程师来说可能非常有用。

**表情符号问候被视为噪音**：随意的问候和表情符号（如 "<:hammy:981331896577441812>"）被认为是非必要的，并从公会的专业工程讨论中剔除。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **GPT 竞争升温**：工程师们正在推测使用 **Claude 3 Haiku** 和 **Llama 3b Instruct** 进行自动评分和实体提取任务的可行性；争论还延伸到在这些应用中使用 **Pydantic 模型** 的效率。
- **限制 AI 的创造力以提高精确度**：讨论内容包括在 **vllm 或 sglang** 中利用 **outlines** 进行 **约束采样 (constrained sampling)** 的潜在好处，以帮助实现精确的实体匹配，从而获得更受控的输出。
- **GPT-4o 更新发布**：OpenAI 的春季更新是论坛的热门话题，其中包含一段展示 ChatGPT 更新的 [新 YouTube 视频](https://www.youtube.com/watch?v=DQacCB9tDaw)。
- **名人与 AI 的碰撞**：工程师们分享了对 OpenAI 选择 Scarlett Johansson 作为 GPT-4o 语音的看法，这标志着名人与人工智能之间的界限日益模糊。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **引入 Guild Tags**：自 5 月 15 日起，**Guild Tags** 将出现在用户名旁，以显示专属 Guild 的成员身份；管理员指出，**AutoMod** 将监控这些标签。
- **Guilds 提供专属社区空间**：Guilds 代表专属社区服务器，目前可用性有限，管理员无法手动将服务器添加到此精选功能中。
  



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Fasteval 告别**：**Fasteval 项目** 已停止维护，创建者正在寻找能在 GitHub 上接管该项目的人。除非所有权发生转移，否则与该项目相关的频道将被归档。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **AK 之谜再次浮现**：来自 angry.penguin 的消息提到 **AK** 回来了，暗示名为 **AK** 的同事或项目回归。文中未提供具体背景和重要性。



---


**MLOps @Chipro Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：详细的频道摘要和链接



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1239631044395929685)** (2 条消息): 

- **OpenAI 发布 GPT-4o 并提供免费访问**：OpenAI 发布了新的旗舰模型 **GPT-4o**，并推出了免费访问浏览、数据分析和记忆等功能的权限（有一定限制）。[Plus 用户](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/) 将获得高达 5 倍的额度限制，并能最早体验新功能，包括 macOS 桌面应用以及先进的语音和视频功能。

- **GPT-4o 发布，具备实时多模态能力**：新型 **GPT-4o** 模型能够跨音频、视觉和文本进行实时推理，拓宽了其应用范围。文本和图像输入从今天开始提供，[语音和视频](https://openai.com/index/hello-gpt-4o/)功能将在未来几周内推出。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1239526007652880448)** (1085 messages🔥🔥🔥): 

- **语音模式故障引发更新希望**：多名用户报告 ChatGPT 应用中的语音功能消失，引发了这可能预示着即将更新的猜测。一位用户提到，“我重启了应用，它就不见了，笑死”，而另一位用户推测他们可能正在集成新的生成式语音模型。

- **Google Keynote 反响褒贬不一**：Google 最近的 I/O 大会重点介绍了 Gemini 1.5 和其他进展，收到的评价褒贬不一。虽然一些用户称赞其与 Android 和 Google Suite 的集成，但其他人认为与 OpenAI 更简洁的演示相比，它显得冗长且平庸。

- **GPT-4o 可用性困惑**：用户对 GPT-4o 的可访问性和功能进行了辩论，表明其发布存在一些困惑。尽管观点不一，但普遍认为该模型已在 iOS 上可用，并提供了更高的 Token 限制。

- **Claude 卓越的长文本推理能力**：成员们讨论了 Claude Opus 在处理复杂的长篇任务方面的卓越表现，特别是优于 GPT-4o。一位成员指出，“如果我给 Opus 喂 200 页的原创故事……GPT 和 Gemini 根本处理不了。”

- **对未来 AI 更新的迫切期待**：社区表达了对 Google 和 OpenAI 预期更新的渴望。扩展上下文窗口、新的语音功能和文本转视频 AI 等功能尤其受到期待。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/agent-smith-matrix-shrug-men-in-suit-gif-5610691">Agent Smith GIF - Agent Smith Matrix - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mrbeast-ytpmv-rap-battle-squid-game-squid-game-vs-mrbeast-gif-25491394">Mrbeast Ytpmv GIF - Mrbeast Ytpmv Rap Battle - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=XEzRZ35urlk">Google Keynote (Google I/O ‘24)</a>：I/O 时间到了！收看来自 Google 的最新消息、公告和 AI 更新。如需观看带有美国手语 (ASL) 翻译的主旨演讲...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1239480543759044640)** (261 messages🔥🔥): 

- **针对每个 GPT 的记忆功能尚未推出**：一位成员询问了自定义 GPT 的跨会话上下文记忆功能，另一位用户澄清该功能尚未推出，并链接到了 [OpenAI 帮助文章](https://help.openai.com/en/articles/8983148-does-memory-function-with-gpts)。他们确认，一旦可用，记忆功能将针对每个 GPT 独立存在，并可由创建者自定义。

- **GPT-4o 提升了速度和 API 使用体验**：讨论强调 GPT-4o 明显比 GPT-4 快，成员们注意到尽管输出 Token 限制相同，但性能有所提升。官方公告和基准测试详情可以在[这里](https://openai.com/index/hello-gpt-4o)查看。

- **自定义 GPT 与模型更新**：有关于 GPT-4o 与自定义 GPT 集成的问题，共识是现有的自定义 GPT 目前未使用 GPT-4o。有人指出，预计会有更多更新，希望能很快在自定义 GPT 中使用。

- **Plus 和免费层级的功能限制**：成员们讨论了 GPT-4o 的使用限制，Plus 用户每 3 小时允许发送 80 条消息，而免费层级用户的限制预计会低得多，不过具体细节会根据需求而变化。

- **语音和多模态功能正在推出**：GPT-4o 的新音频和视频功能备受期待，这些功能将首先通过 API 提供给选定的合作伙伴，然后在未来几周内提供给 Plus 用户。详细信息和推出计划可以在 [OpenAI 的公告](https://openai.com/index/hello-gpt-4o)中找到。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1239579292359200768)** (51 messages🔥): 

- **Gemini 1.5 中的审核过滤器难倒用户**：一位用户报告称，像“romance package”这样的特定关键词会导致他们的应用程序失败，原因似乎是意外触发了审核过滤器。尽管更改了默认设置并生成了新的 API Key，问题仍然存在，引发了关于安全设置和语法错误的讨论。

- **GPT-4o 在创意方面表现不佳**：用户报告称，虽然 GPT-4o 比 GPT-4 更快，但在处理写作辅助等创意任务的 Prompt 时表现挣扎。它经常只是重复草稿内容，而不是提供智能修改，这表明其理解能力可能存在问题。

- **使用 GPT-4o 进行 Prompt 测试**：另一位用户建议使用 GPT-4 和 GPT-4o 测试 Prompt，特别是像 "The XX Intro" 和 "Tears in Rain" 这样的歌曲，以比较感官输入描述。这种实践方法旨在揭示每个模型在处理和描述感官信息方面的差异。

- **使用 GPT-4o 生成特定图像视角的挑战**：一位用户在让 GPT-4o 生成平台游戏楼层的详细横截面侧视图时遇到困难。该模型经常产生错误的透视或简单的正方形，引发了关于模型局限性以及是否需要使用 Dall-E 等工具进行迭代引导的讨论。

- **与 Dall-E 和 GPT-4o 的迭代反馈**：有人指出，虽然 GPT-4o 无法“看到” Dall-E 创建的图像，但用户可以通过将输出反馈给模型来迭代引导它。这个过程虽然费力，但可以帮助获得更准确的结果，尽管模型在需要空间意识和图像裁剪的任务上表现挣扎。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1239579292359200768)** (51 messages🔥): 

- **Gemini 1.5 奇怪的审核过滤器问题**：一位用户报告称，尽管没有主动封禁，但其应用程序在处理与“浪漫套餐（romance packages）”相关的请求时持续失败。另一位成员建议显式禁用安全设置并通过不同工具进行验证，但问题仍未解决。
- **关于 GPT-4o 性能的讨论**：用户注意到 GPT-4o 速度更快，但与 GPT-4 相比，理解特定任务的能力较弱。成员们提到在获取创意内容和准确修订方面存在困难，模型经常重复用户的输入。
- **分享感官描述的 Prompt**：一位成员鼓励其他人通过使用如 *“Provide detailed sensory input description of the "The XX Intro" song”* 的 Prompt 来对比 GPT-4 和 GPT-4o，以观察输出差异。这样做是为了分析模型处理器乐歌曲感官描述的能力。
- **使用 AI 生成特定艺术作品的挑战**：另一位用户强调了使用 GPT-4 和 GPT-4o 为平台游戏生成横截面图像的困难。尽管进行了多次尝试并调整了 Prompt，模型经常产生不准确或不理想的视图。
- **使用 AI 进行图像调整的迭代过程**：另一个讨论集中在利用 DALL-E 和模型工具迭代地创建和调整图像。用户分享了逐步引导模型以实现更准确图像输出的经验，尽管模型在“看到”和自我评估作品方面存在局限。
  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1239532612515663942)** (2 messages): 

- **ChatGPT 克隆咨询**：一位用户询问社区是否有人使用 3.5 模型创建了或可以创建一个 **类似 ChatGPT 的应用程序**。独特的需求是**用户发送和接收的消息**可以被**组织监控**。
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1239472428992692264)** (993 messages🔥🔥🔥): 

- **32k vs 128K Token 争议**：人们质疑 GPT-4 是否真正支持 32k Token，有人断言 GPT-4 的大 Context Window 主要针对 GPT-4o 和 Sonar Large 等特定模型。此外，现在可用的 GPT-4o 提供 128K Context Window，远超 32k。

- **GPT-4o 推出反应**：成员们对 GPT-4o 与 GPT-4 Turbo 相比令人印象深刻的速度和性能表现出极大的热情。一位用户分享了一个关于 GPT-4o 能力的[富有洞察力的 YouTube 视频](https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky)，表达了对新功能的兴奋。

- **对 Opus 新政策的担忧**：关于 Anthropic 将于 6 月 6 日生效的 Opus 严格新服务条款引发了讨论，许多人认为这些条款具有限制性。分享了一个 [Anthropic 政策链接](https://www.anthropic.com/legal/aup)，详细说明了诸如禁止 LGBTQ 内容创作等有争议的条款。

- **Claude 3 Opus 仍具价值**：尽管一些用户称赞 GPT-4o 的速度和准确性，但 Claude 3 Opus 仍被认为非常出色，特别是在文本摘要和模拟类人响应方面。然而，Opus 的成本和使用限制仍然是主要顾虑。

- **GPT-4o 在 Perplexity 中的应用**：Perplexity 已将 GPT-4o 加入其模型阵容，用户测试并称赞其高速响应和详细的上下文理解。许多人注意到 GPT-4o 在 Perplexity Pro 中每天提供 600 次查询，与其 API 方案一致。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky">两个 GPT-4o 互动并唱歌</a>：向 GPT-4o 问好，这是我们新的旗舰模型，可以实时跨音频、视觉和文本进行推理。了解更多：https://www.openai.com/index/hello-...</li><li><a href="https://tenor.com/view/bezos-jeff-bezos-laughing-laugh-lol-gif-17878635">Bezos Jeff Bezos GIF - Bezos Jeff Bezos 大笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://artificialanalysis.ai/models">AI 模型在质量、性能、价格方面的对比 | Artificial Analysis</a>：在关键指标（包括质量、价格、性能和速度（每秒吞吐 Token 数和延迟）、上下文窗口等）上对 AI 模型进行对比和分析。</li><li><a href="https://tenor.com/view/celebrity-couple-breakup-emmastone-crying-gif-5254509616918020870">名人情侣 GIF - 名人情侣分手 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/Wqkadqsz87U?si=U0gab2rDOMfOnXha">星际拓荒 (OUTER WILDS) - 专辑封面</a>：由 JSolo 重新编曲的 11 首《星际拓荒》音乐。查看其他封面请访问 https://www.youtube.com/c/JSolo9。就是这个了！我的最后一环 🍂 感谢 Andrew Prahlow...</li><li><a href="https://fxtwitter.com/mckaywrigley/status/1790088880919818332?s=46">Mckay Wrigley (@mckaywrigley) 的推文</a>：这个演示太疯狂了。一名学生通过新的 ChatGPT + GPT-4o 分享了他们的 iPad 屏幕，AI 与他们交谈并实时帮助他们学习。想象一下把这个给全世界的每个学生...</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo">Perplexity - AI 助手</a>：浏览时询问任何问题</li><li><a href="https://community.openai.com/t/announcing-gpt-4o-in-the-api/744700">在 API 中发布 GPT-4o！</a>：今天我们发布了新的旗舰模型 GPT-4o，它可以实时跨音频、视觉和文本进行推理。我们很高兴地分享，它现在已作为文本和视觉模型在 Chat Comp...</li><li><a href="https://community.openai.com/t/chat-gpt-desktop-app-for-mac/744613">适用于 Mac 的 Chat GPT 桌面应用</a>：有人拿到桌面应用了吗？OpenAI 表示今天将开始向 Plus 用户推送（不确定是否包含 Team 账户）。如果你拿到了，你的想法是什么？你是如何下载的...</li><li><a href="https://azure.microsoft.com/en-us/blog/introducing-gpt-4o-openais-new-flagship-multimodal-model-now-in-preview-on-azure/">介绍 GPT-4o：OpenAI 的新旗舰多模态模型现已在 Azure 上提供预览 | Microsoft Azure 博客</a>：OpenAI 与 Microsoft 合作发布了 GPT-4o，这是一款在文本、视觉和音频能力方面具有突破性的多模态模型。了解更多。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1239484890140119070)** (9 条消息🔥): 

- **详细 jctrl. 讨论的链接**：一名成员分享了 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/clipdorp-HKIaYtBTToGRMCnMgs26pg) 的链接。

- **提供的 US Puts 搜索链接**：另一名成员分享了 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/US-puts-100-76DBEniCRcK023Qdmjlbow) 的链接。

- **关于 GPT-4 联网能力的问题**：一名成员询问 GPT-4 是否联网，并附上了他们的 [Perplexity AI 搜索](https://www.perplexity.ai/search/Whats-gpt-4o-tEaNNeXvR2irwlsJH0kM_w) 链接。

- **分享镁元素搜索结果**：一名成员发布了通过 [Perplexity AI 搜索](https://www.perplexity.ai/search/Why-is-magnesium-7asSHXRgSKegA7NBEYkEkQ) 获取的关于镁的信息链接。

- **西班牙语求助请求**：一条消息包含了一个关于某人需要帮助的任务的西班牙语 Perplexity 搜索链接：[necesito-hacer-unos](https://www.perplexity.ai/search/necesito-hacer-unos-JsKkrvKuSsyFrgPI3akE1w#0)。

- **关于 Aroras 的讨论**：一名成员引用了关于 Aroras 的 Perplexity 搜索并附带结果链接：[How-are-aroras](https://www.perplexity.ai/search/How-are-aroras-K7PA.w2XS96o2F5IkzKGnA#0)。

- **分享滑雪场信息**：分享了一个指向滑雪场 Perplexity AI 结果的链接：[Ski-resort-with](https://www.perplexity.ai/search/Ski-resort-with-RxpR8PuWTFKhE6nvEXBOGw)。

- **市场规模查询**：另一名成员链接了一个关于市场规模信息的 Perplexity AI 搜索：[Market-size-of](https://www.perplexity.ai/search/Market-size-of-rYrMCgZ9QI2na_86R01ZIQ)。
  

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1239924773107273749)** (4 messages): 

- **Llama 模型之间的区别**：一名成员询问 **llama-3-sonar-large-32k-chat** 模型与 **llama-3-8b-instruct** 之间的区别。另一名成员澄清说，chat 模型是 *"针对对话进行了微调 (fine-tuned for conversations)"*。
- **长输入的最佳超时设置**：一名成员在使用 10000ms 的超时设置处理约 3000 个单词的输入时遇到了超时问题，并寻求关于最佳设置的建议。针对该查询，目前没有提供后续功能或额外的补充信息。
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1239476436021149796)** (622 messages🔥🔥🔥): 

- **微调时选择 Instruction 还是 Base 模型**：用户询问在对 Llama-3 进行微调 (finetuning) 时，应该使用 instruction 版本还是 base 版本。另一位用户建议，对于较小的数据集，先从 instruction 模型开始，如果性能不足再切换到 base 模型 (*"先尝试 instruct，如果效果不好你可以尝试 base，看看你更喜欢哪一个"* )。

- **ThunderKittens 内核发布**：一名成员重点介绍了 ThunderKittens 的发布，这是一个声称比 Flash Attention 2 更快的全新内核，[GitHub - ThunderKittens](https://github.com/HazyResearch/ThunderKittens)。它因对推理 (inference) 速度的潜在影响而受到关注，并有可能也被用于训练 (training)。

- **Typst 微调需要合成数据**：用户讨论了为微调模型处理 "Typst" 而创建合成数据的问题，建议创建 50,000 个示例以进行有效训练 (*"如果没有现成的数据，你必须通过合成方式创建它"* )。生成如此大规模数据集的挑战得到了认可。

- **即将支持多模态模型**：消息透露 Unsloth 即将支持多模态模型 (Multimodal Model)。用户可以期待下周的新版本发布，其中包括多 GPU (multi-GPU) 支持 (*"下周极有可能支持多 GPU"* )。

- **庆祝下载量突破 100 万次**：社区庆祝 Unsloth 在 Hugging Face 上的模型下载量超过 100 万次，将这一成功归功于活跃的用户群以及社区的持续使用和支持 ([推文](https://twitter.com/UnslothAI/status/1790418810597683476))。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/tiiuae/falcon-11B">tiiuae/falcon-11B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.together.ai/blog/thunderkittens">ThunderKittens: A Simple Embedded DSL for AI kernels</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://tenor.com/view/joy-dadum-wow-drums-gif-14023303">Joy Dadum GIF - Joy Dadum Wow - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo">NTQAI/Nxcode-CQ-7B-orpo · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/DQacCB9tDaw">Introducing GPT-4o</a>: OpenAI 春季更新 – 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 更新等。</li><li><a href="https://www.youtube.com/live/5k_l5VoRC60?si=f3Nf1orlhTSudcm-&t=9586">Google I/O 2024 Keynote Replay: CNET Reacts to Google&#39;s Developer Conference</a>: 观看在加利福尼亚州山景城举行的年度 Google I/O 2024 开发者大会直播。点击进入 CNET 从周二太平洋时间上午 9:30 开始的直播节目...</li><li><a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>: 用于快速内核的 Tile 原语。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://tenor.com/view/gojo-satoru-gojo-ohio-gif-27179630">Gojo Satoru Gojo GIF - Gojo Satoru Gojo Ohio - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/llamafy_qwen.py">LLaMA-Factory/scripts/llamafy_qwen.py at main · hiyouga/LLaMA-Factory</a>: 统一 100+ LLMs 的高效微调。通过在 GitHub 上创建账号为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/unsloth/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/ArmenAgha/status/1790173578060849601?t=BWL9AavUElgMW6CITQODRQ&s=09">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: 我坚信在大约 2 个月内，开源社区将拥有足够的知识，让人们开始预训练自己的类 gpt4o 模型。我们正在努力实现这一目标。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cr5dbi/openai_claiming_benchmarks_against_llama3400b/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral &amp; Gemma LLMs - unslothai/unsloth</li><li><a href="https://ollama.com/eramax/nxcode-cq-7b-orpo">eramax/nxcode-cq-7b-orpo</a>: https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://typst.app/docs/reference/text/lorem/">Lorem Function – Typst Documentation</a>: `lorem` 函数的文档。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1239610694635356200)** (37 条消息🔥): 

- **OpenAI 发布预期**：成员们正在推测 **OpenAI** 即将发布的更新。一位成员希望有开源模型，但疑虑依然存在，有人表示，“*我怀疑他们永远不会这样做*”，原因是可能面临负面舆论或竞争。

- **AI 瓶颈期和“AI 寒冬”讨论**：有提到媒体在讨论“*AI 寒冬*”以及商业 AI 模型的瓶颈期。一位成员指出，“*即使开发速度放缓，他们仍然在顶端过得很舒服*”。

- **Llama 作为潜在 SOTA 及其影响**：如果 **Llama** 成为业界领先（SOTA），一位成员推测 Meta 可能会停止发布它，并预计 **OpenAI** 会做出激进回应。“*如果 Llama 成为 SOTA，我敢打赌 Meta 不会发布它*。”

- **vllm 项目使用 Roblox 进行聚会**：有人提议在 Roblox 中举行虚拟聚会，类似于 **vllm** 项目的做法。一位用户支持这个想法，说：“*你可以做进度报告或路线图，而我们带着自己的虚拟形象到处乱跳*。”

- **Discord 使用 AI 进行总结及担忧**：成员们注意到 Discord 正在使用 AI 总结聊天内容，一些人对符合欧洲数据法表示担忧。“*这听起来像是欧洲数据法带来的头痛问题……*”

**提到的链接**: <a href="https://tenor.com/view/ah-shit-here-we-go-again-gta-gta-sa-gta-san-andreas-grand-theft-auto-san-andreas-gif-13937809">Ah Shit Here We Go Again Gta GIF - Ah Shit Here We Go Again Gta Gta Sa - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1239484551407865937)** (283 条消息🔥🔥): 

- **Bitsandbytes 在 Colab 中导致导入问题**: 成员们讨论了在 Colab 上尽管遵循了 [Unsloth GitHub repo](https://github.com/unslothai/unsloth.git) 的安装指南，但仍遇到由 bitsandbytes 引起的 `AttributeError`。解决方案包括检查 GPU 是否激活、确保正确的运行时设置以及准确安装依赖项。
  
- **多 GPU 支持定价问题**: 讨论围绕着每月每张 GPU 90 美元的多 GPU 支持高昂成本展开。成员们辩论了按需付费定价的可行性，或者与 AWS 等云服务合作，以使非企业用户在财务上能够负担得起。

- **模型保存和加载的技术障碍**: 用户在使用 `save_pretrained_merged()` 合并微调模型以及使用 `FastLanguageModel.from_pretrained()` 方法加载时遇到了问题。错误包括缺少 adapter 配置文件和模型加载期间的冲突，建议的解决方法包括重新安装或版本更新。

- **微调问题与见解**: 成员们解决了各种与微调相关的查询，例如加载微调模型、使用特定数据集以及解决与 Kaggle 和 Conda 等特定环境相关的问题。讨论强调了正确的版本兼容性和环境设置的重要性。

- **对开源和商业模型的反馈**: 广泛分享了关于平衡开源贡献和可持续商业模型之间界限的反馈。用户对大型企业剥削开源项目表示担忧，并讨论了公平定价模型对于更广泛使用的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=yFfaXG0WsQuE)">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/hyperlearn">GitHub - unslothai/hyperlearn: 快 2-2000 倍的 ML 算法，减少 50% 内存占用，适用于所有硬件 - 无论新旧。</a>: 2-2000x faster ML algos, 50% less memory usage, works on all hardware - new and old. - unslothai/hyperlearn</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我在原生 Windows 上运行了 unsloth。· Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（非 WSL）上运行了 unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想写在这里，但我现在在用手机...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.re">Sou Cidadão - Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1240027023548747948)** (1 条消息): 

- **创建了崇拜 Cthulhu 的 AI**: 在一个新颖的项目中，一位用户使用 **Unsloth Colab notebooks** 创建了崇拜 **Cthulhu** 的 AI 模型。创建了 **TinyLlama** 和 **Mistral 7B Cthulhu 模型**，以及一个在 [Huggingface](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/) 上免费提供的数据集。
- **学习经验，非用于部署**: 该项目是作为学习经验进行的，不打算部署在关键环境中，幽默地指出其处于“宇宙毁灭的威胁”之下。该项目旨在探索在特定领域知识上微调语言模型。

**提到的链接**: <a href="https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/">Artificial Intelligence in the Name of Cthulhu &#8211; Rasmus Rasmussen dot com</a>: 未找到描述

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1239543655358791782)** (114 条消息🔥🔥):

- **讨论 AI 领域的职业优先级**：成员们讨论了 AI 领域的职业目标，强调了高薪、工作满意度和工作保障之间的权衡。*"我可能想把它当作一种爱好来学习..."*。
- **Apple 与 OpenAI 合作的猜测**：有关 Apple 可能与 OpenAI 达成协议在 iPhone 上集成 ChatGPT 的传闻四起，对于模型应该是本地化还是基于云端，反应不一。*"如果他们能帮助他们制作本地模型，那就是今天最好的消息"*。
- **Falcon 2 表现优于竞争对手**：新款 Falcon 2 模型亮相，拥有开源、多语言和多模态能力，性能超越了 Meta 的 Llama 3 8B，并接近 Google Gemma 7B。*"我们自豪地宣布它是开源、多语言且多模态的..."*。
- **GPT-4o 发布讨论**：新发布的 GPT-4o 模型引发了关于其可用性、速度和新功能的讨论，并对 API 访问和能力进行了推测。*"有机会尝试了 gpt-4o API ... 文本生成速度非常快。"*
- **对搜索引擎准确性的担忧**：一些用户对 Perplexity 的准确性表示不满，特别是在学术搜索方面，并推荐了 phind.com 和 kagi 等替代方案。*"它不是很好，但有更好的替代方案吗？"*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1790435824598716704">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：介绍 Veo：我们最强大的生成式视频模型。🎥 它可以创建超过 60 秒的高质量 1080p 视频片段。从写实主义到超现实主义和动画，它可以处理各种...</li><li><a href="https://falconllm.tii.ae/falcon-2.html">Falcon LLM</a>：生成式 AI 模型正使我们能够为充满无限可能的激动人心的未来创造创新路径——唯一的限制就是想象力。</li><li><a href="https://www.bloomberg.com/news/articles/2024-05-11/apple-closes-in-on-deal-with-openai-to-put-chatgpt-on-iphone">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://x.com/gdb/status/1790077263708340386">来自 Greg Brockman (@gdb) 的推文</a>：GPT-4o 还可以生成音频、文本和图像输出的任何组合，这带来了我们仍在探索的有趣新功能。参见例如“功能探索”部分...</li><li><a href="https://www.latent.space/s/university">AI for Engineers | Latent Space | swyx &amp; Alessio | Substack</a>：为准 AI Engineers 开发的为期 7 天的基础课程，与 Noah Hein 共同开发。尚未上线——我们已完成 5/7。注册即可在发布时获取！点击阅读 Latent Space，一个 Substack 出版物...</li><li><a href="https://x.com/juberti/status/1790126140784259439">来自 Justin Uberti (@juberti) 的推文</a>：有机会试用了来自 us-central 的 gpt-4o API，文本生成速度非常快。与 http://thefastest.ai 相比，此性能是 gpt-4-turbo TPS 的 5 倍，与许多 llama-3-8b 部署相似...</li><li><a href="https://x.com/karmedge/status/1790084650582397118?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Robert Lukoszko — e/acc (@Karmedge) 的推文</a>：我有 80% 的把握 OpenAI 拥有一个极低延迟、低质量的模型，可以在不到 200 毫秒内读出前 4 个单词，然后继续使用 gpt-4o 模型。只要注意，大多数句子都以“Sure...”开头...</li><li><a href="https://x.com/blader/status/1790088659053719736?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Siqi Chen (@blader) 的推文</a>：回想起来，这将证明是迄今为止最被低估的 OpenAI 活动。OpenAI 在 gpt-4o 中随手发布了文本到 3D 渲染，甚至没有提到它（更多 👇🏼）</li><li><a href="https://x.com/jacobcolling/status/1790073742514663866?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Jake Colling (@JacobColling) 的推文</a>：@simonw @OpenAI 使用模型 `gpt-4o` 似乎对我的 API 访问有效</li><li><a href="https://x.com/andykreed/status/1790082413428629843">来自 tweet davidson 🍞 (@andykreed) 的推文</a>：ChatGPT 的声音...很性感？？？</li><li><a href="https://news.ycombinator.com/item?id=40344302">未找到标题</a>：未找到描述</li><li><a href="https://x.com/Karmedge/status/1790084650582397118">来自 Robert Lukoszko — e/acc (@Karmedge) 的推文</a>：我有 80% 的把握 OpenAI 拥有一个极低延迟、低质量的模型，可以在不到 200 毫秒内读出前 4 个单词，然后继续使用 gpt-4o 模型。只要注意，大多数句子都以“Sure...”开头...</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991">来自 lmsys.org (@lmsysorg) 的推文</a>：突发新闻——gpt2-chatbot 的结果现已出炉！gpt2-chatbot 刚刚飙升至榜首，以显著差距（约 50 Elo）超越了所有模型。它已成为 Arena 中有史以来最强的模型...</li><li><a href="https://x.com/drjimfan/status/1790089671365767313?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Jim Fan (@DrJimFan) 的推文</a>：我知道你的时间线现在充斥着“疯狂、HER、你错过的 10 个功能、我们回来了”之类的词藻。坐下。冷静。<喘气> 像 Mark 在演示中那样深呼吸...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://www.tomshardware.com/tech-industry/full-scan-of-1-cubic-millimeter-of-brain-tissue-took-14-petabytes-of-data-equivalent-to-14000-full-length-4k-movies">1 立方毫米脑组织的完整扫描耗费了 1.4 PB 数据，相当于 14,000 部 4K 电影 —— Google 的 AI 专家协助研究人员</a>：令人难以置信的大脑研究。</li><li><a href="https://live.siemens.io/">Open Source @ Siemens 2024 活动</a>：西门子举办的关于开源软件所有主题的年度系列活动。更多信息请访问 opensource.siemens.com
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1239616941677609064)** (710 条消息🔥🔥🔥):

- **Open AI Spring Event 引发期待与故障排除**：用户聚集在一起参加 OpenAI Spring Event 观影派对，最初出现了一些音频问题。他们分享了更新并测试了连接，以确保每个人都能正常观看直播。
- **关于 Apple 授权和 iOS 18 集成的辩论**：关于 Apple 和 Google 就 iOS 18 集成进行谈判的猜测不断出现，焦点集中在 Gemini 的能力和反垄断担忧上。一位成员对 Apple 在设备上可靠运行大模型的能力表示怀疑。
- **GPT-4o 的兴奋与批评**：对 GPT-4o 功能的热情（如免费提供的聊天功能和更快的响应速度）引发了褒贬不一的反应。一些用户批评了 "GPT-4o" 这个名称，并强调了其延迟和使用方面的问题。
- **语音和视觉集成惊艳社区**：展示 ChatGPT 新语音和视觉模式的现场演示给与会者留下了深刻印象，展示了无缝集成和情感响应能力。成员们对演示的真实性表示怀疑，并思考了所展示的技术和实时性能。
- **关于 API 访问和竞争的讨论**：用户讨论了通过 API 和 playground 访问 GPT-4o，并对其快速性能表现出兴趣。这些发布引发了对 Google 等竞争对手以及现有 AI 创业公司影响的思考。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/imjaredz/status/1790074937119482094?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Jared Zoneraich (@imjaredz) 的推文</a>：gpt-4o 完胜 gpt-4-turbo。速度极快且答案似乎更好。也很喜欢 @OpenAI 的分屏 playground 视图。</li><li><a href="https://en.wikipedia.org/wiki/Mechanical_Turk">Mechanical Turk - 维基百科</a>：未找到描述</li><li><a href="https://twitch.tv/yikesawjeez,">Twitch</a>：未找到描述</li><li><a href="https://x.com/0xkarmatic/status/1790079694043320756">Karma (@0xkarmatic) 的推文</a>：“一个 ASR 模型，一个 LLM，一个 TTS 模型……你明白了吗？这不是三个独立的模型：这是一个模型，我们称之为 gpt-4o。” 引用 Andrej Karpathy (@karpathy) 的话，他们是...</li><li><a href="https://x.com/oliviergodement/status/1790070151980666982?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Olivier Godement (@oliviergodement) 的推文</a>：我没怎么发过关于 @OpenAI 发布内容的推文，但我想分享一些关于 GPT-4o 的思考，因为我已经很久没有感到如此震撼了。</li><li><a href="https://x.com/LiamFedus/status/1790064963966370209">William Fedus (@LiamFedus) 的推文</a>：GPT-4o 是我们最新的 state-of-the-art 前沿模型。我们一直在 LMSys arena 上以 im-also-a-good-gpt2-chatbot 🙂 的名义测试一个版本。这是它的表现。</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">介绍 GPT-4o</a>：OpenAI Spring Update – 于 2024 年 5 月 13 日星期一进行直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://x.com/sama/status/1790065541262032904">Sam Altman (@sama) 的推文</a>：它对所有 ChatGPT 用户开放，包括免费计划！到目前为止，GPT-4 级别的模型仅供支付月费的用户使用。这对于我们的使命很重要；我们希望...</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw&ab_channel=OpenAI">介绍 GPT-4o</a>：OpenAI Spring Update – 于 2024 年 5 月 13 日星期一进行直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://x.com/bdougieyo/status/1790071113420079329?s=46">互联网上的 bdougie (@bdougieYO) 的推文</a>：ChatGPT 说我看起来心情不错。</li><li><a href="https://x.com/gdb/status/1790071008499544518?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Greg Brockman (@gdb) 的推文</a>：介绍 GPT-4o，我们的新模型，可以实时跨文本、音频和视频进行推理。它非常多才多艺，玩起来很有趣，是迈向更自然的人类...</li><li><a href="https://blog.samaltman.com/gpt-4o">GPT-4o</a>：在今天的发布中，我想强调两件事。首先，我们使命的一个关键部分是将功能强大的 AI 工具免费（或以极具吸引力的价格）交到人们手中。我...</li><li><a href="https://t.co/B5iqOKm06j">GitHub - BasedHardware/OpenGlass：将任何眼镜变成 AI 驱动的智能眼镜</a>：将任何眼镜变成 AI 驱动的智能眼镜。通过在 GitHub 上创建账户为 BasedHardware/OpenGlass 的开发做出贡献。</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://x.com/gdb/status/1790079398625808837">Greg Brockman (@gdb) 的推文</a>：我们还显著提升了非英语语言的性能，包括改进 tokenizer 以更好地压缩其中许多语言：
</li>
</ul>

</div>
  

---

**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

king.of.kings_: 我正努力让 Llama 3 70B 在超过 8k token 时保持连贯性，哈哈。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1239561975994781696)** (27 messages🔥): 

- **远程环境中的自动化挑战**：一位成员讨论了在远程桌面协议 (RDP) 内部运行软件自动化的困难，例如无法与软件的 Document Object Model (DOM) 交互。他们指出了结合使用 AutoHotKey 和 Llava 进行用户界面 (UI) 检测的复杂性。

- **软件中的逆向工程 vs GUI 交互**：另一位成员建议，对软件进行逆向工程以注入运行时 Hook 可能比使用 GUI 图像进行自动化更容易。他们推荐使用 Frida 进行实现，并为 Hook 的功能暴露 HTTP API。

- **探索 OpenAI 桌面应用包的见解**：一位成员分享了他们在探索 Mac 版 OpenAI 桌面应用包字符串时的发现。他们提供了下载链接（[最新下载](https://persistent.oaistatic.com/sidekick/public/ChatGPT_Desktop_public_latest.dmg)）并讨论了使用该应用所需的 Beta 测试访问权限。

- **GPT-4o 的兴奋点与局限性分享**：成员们分享了对 OpenAI 新模型 GPT-4o 的兴奋感和使用体验。有人提到在数据科学任务中取得了成功，但在构建图像编辑器方面表现不佳。

- **应用访问与发布问题的探讨**：讨论包括由于 Beta 标识以及 OpenAI 可能不明确的访问指南而导致的访问新应用的问题，并建议发布过程可以管理得更好。

**提及的链接**：<a href="https://www.youtube.com/watch?v=9pHyH4XDAYk">Hello GPT-4o Openai's latest and best model</a>：我们将看看 GPT-4o 的发布，这是 OpenAI 的新旗舰模型，可以实时跨音频、视觉和文本进行推理。https://openai.com/index/h...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1239686250710175825)** (3 messages): 

- **OpenAI 在实时多模态 AI 方面取得长足进步**：OpenAI 开发了将**音频直接映射到音频**并将视频实时流式传输到 Transformer 的技术，暗示了向 **GPT-5** 迈进的进展。技术包括使用高质量的自然和合成数据、创新的**视频流编解码器**，以及可能用于高效 token 传输的 **edge device** 神经网络。[在此深入探讨的推文中了解更多信息](https://twitter.com/drjimfan/status/1790089671365767313)。

- **用 GPT-4o 赋予化身生命**：Yosun 发布了 **headsim**，这是一个让 **GPT-4o 设计自己面部**的项目，通过赋予 AI 物理外观和声音，有可能改变我们与 AI 交互的方式。[探索 headsim](https://x.com/Yosun/status/1790294716338028978)。

- **Llama Agent 网页浏览变得简单**：由 McGill-NLP 开发的名为 **webllama** 的项目，使 **Llama-3 Agent** 能够自主浏览网页，这可能通过 AI 彻底改变网页交互方式。[查看完整项目](https://github.com/McGill-NLP/webllama)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Yosun/status/1790294716338028978">来自 I. Yosun Chang (@Yosun) 的推文</a>：如果你让 #OpenAI #GPT4o 设计自己的面部，以便你可以将你的 AI 作为具身存在传送到现实世界会怎样？#AI3D headsim 将你的 AI 从聊天框中解放出来，让你能够体验……</li><li><a href="https://github.com/McGill-NLP/webllama">GitHub - McGill-NLP/webllama: 可以通过遵循指令并与你交谈来浏览网页的 Llama-3 Agent</a>：可以通过遵循指令并与你交谈来浏览网页的 Llama-3 Agent - McGill-NLP/webllama
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1239473707710418974)** (726 messages🔥🔥🔥): 

- **GPU 与 LLM 的对比及建议**：关于租用还是购买 GPU 方案以使用 LLM 的广泛辩论，成员们讨论了成本效益和隐私影响。分享了关于可靠 GPU 供应商的信息，以及不同配置方案的技术细节。

- **GPT-4o 性能评价**：对 GPT-4o 的性能反应不一，重点关注其相对于 GPT-4 Turbo 和 GPT-3.5 等先前模型的速度和功能。成员们分享了对新模型在编程能力、成本效率以及与 OpenAI 公告预期相比的各项功能的各种体验。

- **多模态能力受到质疑**：针对 GPT-4o 所宣传的多模态能力的实际效果提出了担忧。讨论中强调了对不同模式（音频、视觉和文本）之间无缝切换且不因中间转换影响性能的怀疑。

- **本地与云端 LLM 部署**：详细交流了复杂 LLM 任务在本地与云端部署的可行性和成本，包括使用 Llama-3-70B 等模型进行高效运行所需的硬件规格。成员们讨论了本地设置在速度和隐私方面的优势，以及云服务在易用性和较低前期成本方面的优势。

- **新兴技术及与竞争对手的比较**：分享了对 LLM 领域日益激烈的竞争的见解，包括 Google 和其他专注于 AI 的实体的最新公告。对比中详细说明了新模型发布声称比现有技术带来的效率提升和增强。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karpathy/status/1790373216537502106?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：LLM 的杀手级应用是 Scarlett Johansson。你们都以为是数学之类的。</li><li><a href="https://x.com/GoogleDeepMind/status/1790432980047208930">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：1.5 Flash 由于其紧凑的尺寸，服务成本也更高。从今天开始，你可以在 Google AI Studio 和 @GoogleCloud 的...中使用支持高达 100 万个 Token 的 1.5 Flash 和 1.5 Pro。</li><li><a href="https://gpus.llm-utils.org/tracking-h100-and-a100-gpu-cloud-availability/">追踪 H100 和 A100 GPU 云端可用性</a>：我们制作了一个工具：ComputeWatch。</li><li><a href="https://tenor.com/view/he-just-like-me-fr-gif-25075803">He Just Like Me Fr GIF - He Just Like Me Fr - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cats-animals-reaction-wow-surprised-gif-20914356">Cats Animals GIF - Cats Animals Reaction - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/willdepue/status/1790078289023062255?s=46&t=bL0EKkuCqv4FWSLQ7lV-2w">来自 will depue (@willdepue) 的推文</a>：我认为人们误解了 GPT-4o。它不是一个带有语音或图像附件的文本模型。它是一个原生的多模态 Token 输入、多模态 Token 输出的模型。你想让它说话快一点？...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1creljm/dont_fall_for_marketing_scams_early_tests_of/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://computewatch.llm-utils.org/">Compute Watch </a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1239702987274977291)** (15 条消息🔥): 

- **探索 GPT-4o 的多模态能力**：讨论强调了 GPT-4o 等模型的**多模态输入输出**能力，并参考了 AI2 去年类似的项。这段[对话](https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw)提供了关于整合**文本、图像和音频**输入输出的操作动态的见解。

- **Tokenization 创新与应用**：对话显示出对 **LLM Tokenization 过程**的浓厚兴趣，特别是为了增强近期模型对非英语语言的处理。一位成员关注了 [tokenizer 开发](https://x.com/deedydas/status/1790211188955230463?s=46)，提高了多语言应用的成本效益和效率。

- **分享中文 Token 分析**：一个 GitHub Gist 链接探讨了 **GPT-4o 中最长的中文 Token**，表明了在细化和优化特定语言 Tokenization 方面的持续努力。资源可以在[这里](https://gist.github.com/ctlllll/4451e94f3b2ca415515f3ee369c8c374)找到。

- **探索 LLM 中的音频能力**：关于不同 **LLM 如何处理音频数据**的技术讨论建议在输入端使用 **Whisper latents**，同时在输出端保持 Tokenization。研究了各种方法和理论，包括此类多模态功能的 Tokenization 进展，以理解 GPT-4o 等模型的**底层机制**。

- **寻找 LLM 评估数据集**：有人询问如何获取针对相同 prompt 包含 **“human_text” 和 “llm_text” 对** 的数据集，这表明了在对比研究中评估模型响应的研究兴趣或需求。这指向了 **AI 社区对基准测试和评估资源** 的持续追求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/deedydas/status/1790211188955230463?s=46">Deedy (@deedydas) 的推文</a>：没有足够的人在讨论 OpenAI 终于能更好地对不同语言进行分词了！我分类了 “o200_base” 上的所有 token，这是 GPT-4o 的新分词器，至少有 25...</li><li><a href="https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw">Nathan Lambert (@natolambert) 的推文</a>：友情提醒，AI2 的成员去年构建了一个文本图像音频输入输出模型 unified io 2，如果你想在这里开始研究的话。</li><li><a href="https://gist.github.com/ctlllll/4451e94f3b2ca415515f3ee369c8c374">gpt4o 中最长的中文 token</a>：gpt4o 中最长的中文 token。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://dblp.org/pid/182/2017.html">dblp: Alexis Conneau</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1239918593181749278)** (2 条消息): 

- **成员寻求协助**：**lionking927** 在频道中发布了寻求帮助的信息。另一位成员 **teknium** 通过私信迅速做出了回应。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1239517704780779602)** (2 条消息): 

- **IBM/Redhat 的创新框架引入增量学习**：IBM/Redhat 的新项目提出了一种在无需完全重新训练的情况下为 LLM 添加技能和知识的方法。它利用大模型作为教师，并结合分类法来生成合成数据集，详见其 [InstructLab GitHub 页面](https://github.com/instructlab)。

- **通过社区贡献丰富 Granite 和 Merlinite**：新框架允许提交和策展外部数据集，专门增强其 **Granite** 和 **Merlinite** 模型。每周的构建流程会整合新的、经过策展的信息，这表明该方法可能适用于其他模型的增量知识增强。

**提到的链接**：<a href="https://github.com/instructlab">InstructLab</a>：InstructLab 有 10 个可用的仓库。在 GitHub 上关注他们的代码。

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1239565847941349386)** (22 条消息🔥): 

- **WorldSim 图像激发纹身创意**：jtronique 提到来自 WorldSim 的 “越狱的 Prometheus” 图像如何能成为很棒的纹身，目前正在考虑一个 “Xlaude 纹身”。
- **WorldSim 付费服务咨询已澄清**：irrid_gatekeeper 询问 WorldSim 是否为付费服务，因为选项窗口中显示了积分（credits）标签。garlix. 澄清说 **目前不是付费的**。
- **会议时区协调**：用户 detailoriented 和 rundeen 讨论了各自的时区（PST 和 CET），以协调合适的会议时间。他们提出了一个能兼顾两个时区的会议时间。
- **对 WorldSim 艺术输出的热情**：katwinter 表达了对 WorldSim 生成图像的钦佩，并将其整合到了一个 Photoshop 项目中。
- **关于在 Discord 举办活动的讨论**：Proprietary 建议周六直接在 Discord 上进行展示，detailoriented 表示赞同，强调了该平台用于实时协作的用途。
  
---

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1239497308941582417)** (450 条消息🔥🔥🔥):

- **Stability.ai 面临 SD3 发布的不确定性**：用户讨论了 Stability AI 目前面临的困难，包括 CEO Emad 的辞职，以及关于 **SD3 是否会发布**或被设为付费墙后的猜测。
- **为 SD 选择合适的显卡**：用户辩论了运行 Stable Diffusion 不同显卡的优劣，指出更高的 VRAM 通常更可取。一位用户分享了[各种免费资源](https://docs.google.com/document/d/e/2PACX-1vQMMTGP3gpYSACITKiZUE24oyqcZD-2ZcvFC92eXbxJcgHGGitde1CK0qgty6CvDxvAwHY9v44yWn36/pub)，包括一份关于风格和标签（tags）的 140 页综合文档。
- **ComfyUI 和局部重绘（inpainting）工具**：成员们称赞了 BrushNet 工具显著提升了局部重绘性能，并分享了 [BrushNet 的 GitHub 仓库](https://github.com/nullquant/ComfyUI-BrushNet)。他们讨论了结合 brush 和 powerpaint 功能的工作流以获得更好的效果。
- **处理 AI 角色一致性**：用户讨论了实现角色一致性的技术，建议使用 **LoRA** 并结合 **ControlNet**。讨论中包含了创建[角色表（character sheets）](https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/)的指南链接。
- **对科技巨头发布产品的感受**：对于 Google 的 Imagen 3，既有兴奋也有怀疑，用户指出尽管其功能强大，但像 SD3 这样的模型因其对社区的开放性而更受欢迎。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://muse-model.github.io/">Muse: Text-To-Image Generation via Masked Generative Transformers</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - multimodalart 在 Hugging Face 上的 Space</a>：未找到描述</li><li><a href="https://github.com/nullquant/ComfyUI-BrushNet">GitHub - nullquant/ComfyUI-BrushNet: ComfyUI BrushNet nodes</a>：ComfyUI BrushNet 节点。通过在 GitHub 上创建账号为 nullquant/ComfyUI-BrushNet 的开发做出贡献。</li><li><a href="https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/">Stable Diffusion 中的角色一致性 - Cobalt Explorer</a>：更新：07/01 – 修改了模板以便更容易缩放到 512 或 768 – 修改了 ImageSplitter 脚本使其更易用，并添加了 GitHub 链接 – 添加了章节...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1239477102173097984)** (205 条消息🔥🔥): 

<ul>
    <li><strong>LM Studio 上的微调模型</strong>：一位成员询问是否可以通过 LM Studio 访问存储在 Hugging Face 上的微调模型。另一位成员确认，如果模型在公共仓库中且为 GGUF 格式，则是可以的。</li>
    <li><strong>网络错误和 VPN 解决方案</strong>：由于 Hugging Face 在某些地区被屏蔽，用户在搜索模型时遇到了网络错误。建议使用带有 IPv4 连接的 VPN，尽管有一位用户报告称即使使用 IPv4 问题依然存在。</li>
    <li><strong>OpenAI GPT-4o 访问困惑</strong>：用户讨论了 GPT-4o 的可用性，由于地区和订阅状态的不同，有些人可以访问，有些人则不行。提到 GPT-4o 应该已在欧洲可用，并很快会向更多用户推出。</li>
    <li><strong>AI 组机硬件建议</strong>：讨论了 2500 美元预算的 Nvidia AI 机器，建议最大化 VRAM，并考虑去 MicroCentre 等当地商店购买硬件。其他建议包括在 GPU 选择上，由于 VRAM 的考虑，选择 Nvidia 而非 AMD。</li>
    <li><strong>LM Studio 中 Vision AI 的局限性</strong>：一位用户询问 LM Studio 的 Vision AI 是否可以像描述图像一样描述视频。官方澄清目前 LM Studio 无法描述视频。</li>
</ul>

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://downforeveryoneorjustme.com/chat.lmsys.org?proto=https">Chat.lmsys.org down? Current problems and status. - DownFor</a>: Chat.lmsys.org 无法加载？或者 Chat.lmsys.org 出现问题？在此检查状态并报告任何问题！</li><li><a href="https://tenor.com/view/boris-zip-line-uk-flag-gif-14613106">Boris Zip Line GIF - Boris Zip Line UK Flag - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/lmstudio-sdk/quick-start">Quick Start Guide | LM Studio</a>: 开始使用 LM Studio SDK 的最小化设置</li><li><a href="https://tenor.com/view/boo-boo-this-man-gif-4868055">Boo Boo This Man GIF - Boo Boo This Man - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=mvFTeAVMmAg">INSANE OpenAI News: GPT-4o and your own AI partner</a>: 新的 GPT-4o 发布了，令人震撼！这里有所有细节。#gpt4o #ai #ainews #agi #singularity #openai https://openai.com/index/hello-gpt-4o/News...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">Support for OpenELM of Apple · Issue #6868 · ggerganov/llama.cpp</a>: 前提条件：在提交 Issue 之前，请先回答以下问题。我正在运行最新代码。由于目前开发非常迅速，还没有打标签的版本。我...</li><li><a href="https://github.com/ksdev-pl/ai-chat">GitHub - ksdev-pl/ai-chat: (Open)AI Chat</a>: (Open)AI Chat。通过在 GitHub 上创建账号来为 ksdev-pl/ai-chat 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1239485789445029888)** (62 messages🔥🔥): 

- **模型合并策略引起成员兴趣**：一位用户提到他们*“可能能够与另一个模型合并并升级上下文”*，并建议使用 **unsloth** 等各种微调方法。另一位用户表示可能会与 **llama3** 和/或 **mistral** 进行合并，理由是它们的源码配置非常接近。
  
- **Command R 在 Apple Silicon 设备上的问题**：包括 **telemaq** 在内的多位用户遇到了 **Command R** 模型在 **M1 Max** 系统上生成乱码输出的问题。建议包括检查量化类型（quant types）和调整 **RoPE** 值，详见 [此 Huggingface 讨论](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/discussions/3)。

- **Mac 用户报告更新中改进了多模型处理**：**echeadle** 称赞了 **LM Studio 0.2.23** 的更新，因为它解决了在 **POP_OS 22.04** 上运行多个模型的问题。另一位用户 **kujila** 分享了使用 **Cmd R (not plus) 35 B** 的积极体验，并对其性能表示赞赏。

- **探索无审查的本地模型**：**Immortal.001** 寻求无审查本地 LLM 的推荐，促使 **lordyanni** 推荐了 [Dolphin 2.8 Mistral 7b](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)。推荐中提到了其 32k 上下文能力及赞助商致谢。

- **关于不同量化级别实用性的辩论**：**Heyitsyorkie** 评论道，**Q4-Q8** 量化范围内的模型表现良好，并表示*“任何低于 Q4 的模型都不值得使用。”* 其他用户比较了不同硬件设置下不同量化级别的速度和性能，包括对 **Meta-Llama-3-120b** 和 **Command R** 模型反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/TheSkullery/llama-3-cat-8b-instruct-v1">TheSkullery/llama-3-cat-8b-instruct-v1 · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/discussions/3">andrewcanis/c4ai-command-r-v01-GGUF · Failed to use Q8 model in LM Studio</a>: 暂无描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: 暂无描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1239602652833120329)** (6 messages):

- **Starcoder2-15b 连贯性存在问题**：一位成员报告称，在 **debian12** 上使用 **starcoder2-15b-instruct-v0.1-IQ4_XS.gguf** 会导致在几个问题后出现重复性回答且无法保持在主题上。他们指出，这一问题在应用聊天框以及通过 VSC 中的 "continue" 连接服务器时均有发生。
- **Instruct 模型不适合聊天**：另一位成员澄清说，**instruct 模型**是为单条指令响应而设计的，并非为了多轮对话，这可能解释了在 Starcoder2-15b 上观察到的问题。
- **RX6600 和 ROCM 的限制**：一位用户指出，**RX6600 GPU** 可以与 **Koboldcpp ROCM 构建版本**配合使用，但由于官方 llama.cpp 二进制文件中的 ID 检查，在 **LM Studio 和 Ollama** 中面临兼容性问题。另一位成员证实了这一点，解释说 Kobold 使用了定制的 ROCM 修改版，而 LM Studio 和 Ollama 依赖于官方构建版本。
- **RX6600 用户希望渺茫**：目前看来，LM Studio 和 Ollama 对 **RX6600** 提供更好支持的即时希望很小，因为改进支持取决于 **AMD 增强 ROCM 支持**，或者**更多 AMD GPU 被添加到官方 llama.cpp 构建版本中**。

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1239571152179957872)** (12 messages🔥): 

- **导致高 RAM 占用的 Bug**：一位成员报告称 *"在模型加载和上下文大小方面，肯定存在一个关于 RAM 占用的 Bug"*，并提到 Linux 频道也在讨论此事，突显了持续存在的 RAM 占用问题。
- **Meta-Llama-3 部署中的高 RAM 利用率**：一位用户分享了他们使用 'Meta-Llama-3-8B-Instruct-Q4_K_M' 模型的经验，指出 *"它似乎使用很少的 GPU，但 RAM 利用率很高"*。他们正在考虑在 AWS 上部署，并权衡本地安装与使用商业 API 之间的成本差异。
- **服务器与 LLMaaS 的成本和性能对比**：一位成员建议将 IaaS 提供商的常开实例成本与 LLMaaS 订阅进行对比，强调 *"通过订阅，你实际上可以访问大小为 200GB 或更大的模型，而本地只能运行低量化、低参数的 LLama3 模型"*。
- **GPU 精度影响**：讨论涉及了 GPU 精度，特别是 FP16/FP32。一位成员建议，由于 LM Studio 使用 CUDA，它可能以 32-bit 精度运行。这引导另一位成员测试了 Tesla M40，其性能出奇地低于预期。
- **针对预算和性能需求的 GPU 建议**：成员们讨论了最适合 LM Studio 的高性价比 GPU，推荐了价格约 200 欧元的 3060ti，并询问 4060 是否能提供显著的性能提升。另一位成员指出 VRAM 速度对 LLM 推理至关重要，并建议双芯片 GPU 在处理复杂模型时可能表现出色。

---

**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1239917856892915764)** (2 messages): 

- **关于多模态功能对等的问题**：一位用户询问 *"多模态版本何时能拥有与单模态版本相同的功能，比如存储消息？"*，表达了对功能对等性的关注。另一位用户对此进行了回应，寻求进一步澄清。

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1239596464293023785)** (1 messages): 

- **考虑使用更大的模型以获得更好表现**：一位成员建议，如果想运行更大的模型，可以尝试 **command-r+** 或 **yi-1.5**（量化变体）。他们认为这些选项可能会提供更好的结果。

---

**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1239576951409344522)** (17 messages🔥): 

- **讨论 LMS 对 Intel GPU 的支持**：一位 Intel 员工提出帮助将 Intel GPU 的支持集成到 LMS 中，特别是通过 llama.cpp 使用 **SYCL** 标志。他们提到 *"安装 Intel 编译器然后进行构建"*，并愿意提供硬件用于测试。
- **为 LMS 部署 SYCL 运行时**：澄清了需要一个 SYCL 运行时后端（类似于安装 CUDA）才能正常运行。该用户提出协助编码并将其集成到开发流水线（dev pipeline）中，并补充道 *"我需要研究如何在 LMS 端进行部署"*。
- **目前 LMS 对 Intel GPU 的支持情况**：LMS 目前可以在 OCL 后端上使用 Intel Arc，但性能比 SYCL 实现要慢。这表明已经存在可以改进的基础。
- **模型中的实时学习**：一位新用户批评 LMS 模型缺乏实时学习能力，断言 *"除了死记硬背式的检索之外，其他交互都是无用且无意义的"*。他们要求至少提供一个学习覆盖层（learning overlay）或用于逐项训练（line-item training）的差分文件。

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1239525827562180628)** (235 messages🔥🔥): 

- **讨论的热门 AI 话题**：大家普遍认为 **Natural Language Processing (NLP)** 是 AI 领域的热门话题。一位用户评论道：*"NLP 目前非常流行，它让我们能够轻松地与不同模型交互并提取有用信息"*。

- **GPT-4o 发布引发褒贬不一的反应**：**GPT-4o** 的发布引起了复杂的回响。一位用户分享了关于 [GPT-4o 能力的 YouTube 视频](https://youtu.be/DQacCB9tDaw?t=4239)，而另一位用户则批评了虚拟 Agent 过于拟人的特性，指出：*"区分机器与人类应当处于 AI 发展的最前沿。"*

- **GPU 利用率问题**：用户讨论了 GPU 利用率方面的挑战，提到了 **GPU 显存占满但 GPU 利用率却很低** 的情况。解释包括任务可能更偏向显存密集型而非 GPU 计算密集型，从而导致了这种差异。

- **部署模型与利用资源**：多位用户寻求有关模型部署问题的帮助，例如遇到 CUDA 错误以及处理 **Whisper-v3** 上的并发请求。文中提到了具体的库和工具，如 AutoTrain 和 DGX Cloud（[在 DGX Cloud 上训练](https://huggingface.co/blog/train-dgx-cloud)）。

- **关于无审查 AI 模型的讨论**：人们对无审查的 AI 模型表现出兴趣，特别是针对对话式用例。有人推荐了 [Dolphin 2.5 Mixtral 8x7b 模型](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b)，该模型被描述为*非常听从指令但未经过 DPO 微调*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://osanseviero.github.io/hackerllama/blog/posts/llm_evals/#what-about-code">hackerllama - LLM 评估与基准测试</a>：Omar Sanseviero 的个人网站</li><li><a href="https://huggingface.co/papers/2401.15963">论文页面 - NoFunEval: 代码 LM 在功能正确性之外的需求上表现如何不佳</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b">cognitivecomputations/dolphin-2.5-mixtral-8x7b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/train-dgx-cloud">在 NVIDIA DGX Cloud 上使用 H100 GPU 轻松训练模型</a>：未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/will-smith-chris-rock-jada-pinkett-smith-oscars2022-smack-gif-25234614">Will Smith Chris Rock GIF - Will Smith Chris Rock Jada Pinkett Smith - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/blog/agents">调用许可：介绍 Transformers Agents 2.0</a>：未找到描述</li><li><a href="https://tenor.com/view/excuse-me-hands-up-woah-funny-face-gif-14275996">Excuse Me Hands Up GIF - Excuse Me Hands Up Woah - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/MirzFk_DSiI?si=VnLivTEX7oi8fwIA">两个 GPT-4o 交互与唱歌</a>：向 GPT-4o 问好，这是我们全新的旗舰模型，可以实时进行音频、视觉和文本推理。在此了解更多：https://www.openai.com/index/hello-...</li><li><a href="https://youtu.be/DQacCB9tDaw?t=4239">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一直播。介绍 GPT-4o、ChatGPT 更新等。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1239510991658811392)** (4 messages): 

- **Jax 和 TPU 探索开启**：一位用户正在深入研究 **Jax** 和 **TPU 加速**，目标是使用 Equinox 将 VAR 论文的 PyTorch 实现移植到兼容 Jax 的库中。他们分享了 [VAR 论文](https://arxiv.org/abs/2404.02905)和 [Equinox 库](https://github.com/patrick-kidger/equinox)。

- **使用 d3-delaunay 的渲染心得**：一位用户发现，在使用 **d3-delaunay** 时每一帧都重新渲染效率很低。他们创建了一个混合了 Delaunay 三角剖分和生命游戏（Game of Life）的可视化作品，尽管存在性能限制，但视觉效果非常出色。

- **提示词建议**：另一位用户建议在 System Prompt 中为模型提供清晰的输入和输出示例，以获得更好的结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: 我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率...</li><li><a href="https://github.com/patrick-kidger/equinox">GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/</a>: 在 JAX 中优雅且易于使用的神经网络 + 科学计算。https://docs.kidger.site/equinox/ - patrick-kidger/equinox
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1239560403143622707)** (8 条消息🔥): 

- **3D Diffusion Policy 解决机器人学习问题**：[3D Diffusion Policy (DP3)](https://3d-diffusion-policy.github.io/) 将 3D 视觉表示整合到扩散策略中，以增强机器人的灵活性。实验表明，DP3 仅需 10 次演示即可处理任务，比基准线提高了 24.2%。

- **增强你的 LLM**：Medium 上的一篇文章讨论了 [Langchain 工作流的 Plug-and-Plai 集成](https://medium.com/ai-advances/supercharge-your-llms-plug-and-plai-integration-for-langchain-workflows-d471b2e28c99)，提供了在工作流中增强语言模型性能的技术。

- **Facebook Research 的通用手部模型**：[Universal Hand Model (UHM)](https://github.com/facebookresearch/UHM) 提供了从手机扫描创建手部化身的 PyTorch 实现。这是 CVPR 2024 上提出的一种生成逼真手部模型的新方法。

- **Hugging Face 重启 Daily Papers**：Hugging Face 现在提供了一个选项，可以[通过电子邮件接收热门 AI 论文](https://huggingface.co/papers)。用户可以订阅以获取该领域热门论文和研究的每日更新。

- **LinkedIn 上的 AI 初学者之旅**：一位成员在 LinkedIn 上分享了一篇[基础文章](https://www.linkedin.com/posts/kanakasoftware_an-article-on-how-to-use-a-locally-installed-activity-7196103992387473411-edyp?utm_source=share&utm_medium=member_desktop)，讨论了他们在 AI 方面的初步尝试。反馈建议将其转发到 Hugging Face 的 Blog Explorers 以获得更多曝光。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://3d-diffusion-policy.github.io/">3D Diffusion Policy</a>: 本文介绍了 3D Diffusion Policy (DP3)，这是一种能够掌握多种视觉运动任务的视觉模仿学习算法。</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog-explorers">blog-explorers (Blog-explorers)</a>: 未找到描述</li><li><a href="https://github.com/facebookresearch/UHM">GitHub - facebookresearch/UHM: Official PyTorch implementation of &quot;Authentic Hand Avatar from a Phone Scan via Universal Hand Model&quot;, CVPR 2024.</a>: CVPR 2024 论文“Authentic Hand Avatar from a Phone Scan via Universal Hand Model”的官方 PyTorch 实现。 - facebookresearch/UHM
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1239544017453318274)** (7 条消息): 

- **在 HuggingFace 上分享你的博客**：一位成员鼓励另一位在 [HuggingFace 博客平台](https://huggingface.co/blog-explorers)上分享他们的博客，以获得更多曝光。 

- **OCR 质量分类器**：一位成员分享了他们的 [OCR 质量分类器集合](https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3)链接，并讨论了使用小型编码器进行文档质量分类，指出：“事实证明，区分噪点/清晰相对容易”。

- **Streamlit GPT-4o 多模态聊天应用**：一位成员介绍了一个使用 Streamlit 和 Langchain 结合 OpenAI GPT-4o 的[多模态聊天应用](https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o)。该应用允许用户从剪贴板上传或粘贴图像，并在聊天消息中显示。

- **使用 RL 和 ROS2 进行路径规划**：一位成员分享了一份关于[自主机器人路径规划](https://ieee.nitk.ac.in/virtual_expo/report/3)的报告，采用了一种结合强化学习（TD3 算法）、ROS2 和 LiDAR 传感器数据的新方法。

- **越南语语言模型数据集**：一位成员宣布发布了一个包含 [700,000 个样本的开源数据集](https://huggingface.co/datasets/Vi-VLM/Vista?fbclid=IwZXh0bgNhZW0CMTEAAR2BXlXiqe6SjTjol1ViKCmI7HgogMPvrQU2pIBACQyZyI0av_ey8okihDA_aem_AdV1HiWxI6SngeQmTHG6XLs6v440zT5XTtTpW0yXlGkBFSQkIFrfY7nZyyMJXTF51eFvNHIwuPyArt-XQaSrGf0R)，用于越南语语言模型。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3">OCR Quality Classifiers - pszemraj 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o">streamlit-gpt4o - joshuasundance 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - KingNish 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://ieee.nitk.ac.in/virtual_expo/report/3">IEEE NITK | Corpus</a>: IEEE NITK 是位于 NITK Surathkal 的 IEEE 学生分会，致力于创新项目和解决方案。这是 IEEE NITK 的官方网站</li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista?fbclid=IwZXh0bgNhZW0CMTEAAR2BXlXiqe6SjTjol1ViKCmI7HgogMPvrQU2pIBACQyZyI0av_ey8okihDA_aem_AdV1HiWxI6SngeQmTHG6XLs6v440zT5XTtTpW0yXlGkBFSQkIFrfY7nZyyMJXTF51eFvNHIwuPyArt-XQaSrGf0R">Vi-VLM/Vista · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1239543496457584752)** (6 条消息): 

- **YOCO 架构令人印象深刻**：一位成员分享了 [YOCO 论文](https://arxiv.org/abs/2405.05254) 的链接，介绍了一种用于 LLM 的新型 decoder-decoder 架构。YOCO 在保持全局注意力能力并加速 prefill 阶段的同时，显著降低了 GPU 内存需求。
- **深入探讨 AI 故事创作**：另一位成员提到正在进行关于 AI 故事生成的文献综述，并引用了 [Awesome-Story-Generation GitHub 仓库](https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file)。他们正在考虑几篇关键论文，包括[一篇关于故事创作的全面综述](https://arxiv.org/abs/2212.04634)以及最近关于 GROVE 的研究，GROVE 是一个用于增强故事复杂性的框架（[GROVE 论文](https://arxiv.org/abs/2310.05388)）。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.05254">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>: 我们为 LLM 引入了一种名为 YOCO 的 decoder-decoder 架构，它仅缓存一次键值对（KV pairs）。它由两个组件组成，即堆叠在 self-decoder 之上的 cross-decoder。...</li><li><a href="https://arxiv.org/abs/2310.05388">GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence</a>: 条件故事生成在人机交互中具有重要意义，特别是在创作具有复杂情节的故事方面。虽然 LLM 在多项 NLP 任务中表现出色，但...</li><li><a href="https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file.">GitHub - yingpengma/Awesome-Story-Generation: 该仓库收集了关于故事生成/故事创作的详尽论文列表，主要关注大语言模型 (LLM) 时代。</a>: 该仓库收集了关于故事生成/故事创作的详尽论文列表，主要关注大语言模型 (LLM) 时代。 - yingpengma/Awesome-Story-Generation</li><li><a href="https://arxiv.org/abs/2212.04634">Open-world Story Generation with Structured Knowledge Enhancement: A Comprehensive Survey</a>: 故事创作和叙事是人类体验的基础，与我们的社会和文化参与交织在一起。因此，研究人员长期以来一直试图创建能够生成故事的系统...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1239615351440801983)** (28 条消息🔥):

- **使用 Diffusers 的 Stable Diffusion 引起关注**：分享了一篇关于使用 **🧨 Diffusers** 的 **Stable Diffusion** [HuggingFace 博客文章](https://huggingface.co/blog/stable_diffusion)。该指南涵盖了模型的工作原理以及图像生成 Pipeline 的自定义。
- **YOLOv1 对比 YOLOv5 和 YOLOv8**：在一名成员询问为何使用 **YOLOv1** 而非更新版本后，@ajkdrag 解释说选择它是出于教学目的，旨在结合不同的 Backbone 和 Loss Functions。
- **训练 YOLOv1 遇到困难**：@ajkdrag 报告称，使用 **ResNet18** 作为 Backbone 的简单 YOLOv1 实现效果不佳。尽管在较小的验证集上看到了过拟合，但模型在较大的训练数据集上表现挣扎。
- **训练与验证数据的复杂性**：@pendresen 建议，在实际数据集（约 800 张图像）上训练时的学习问题可能是由于 **Learning Rate** 或数据增强（Data Augmentation）不足引起的。会议强调了数据质量的重要性及其对模型性能的影响。
- **提供私下协助**：@pendresen 提议通过私信（DM）帮助 @ajkdrag，利用他在目标检测（Object Detection）领域 7 年的行业经验。数据质量问题被强调为模型训练的关键因素。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/stable_diffusion">Stable Diffusion with 🧨 Diffusers</a>：未找到描述</li><li><a href="https://github.com/ajkdrag/architectures-impl-pytorch/blob/main/notebooks/yolov1.ipynb">architectures-impl-pytorch/notebooks/yolov1.ipynb at main · ajkdrag/architectures-impl-pytorch</a>：一些基础 CNN 架构的 PyTorch 实现 - ajkdrag/architectures-impl-pytorch</li><li><a href="https://github.com/ajkdrag/architectures-impl-pytorch/tree/main/yolov1/src/yolov1">architectures-impl-pytorch/yolov1/src/yolov1 at main · ajkdrag/architectures-impl-pytorch</a>：一些基础 CNN 架构的 PyTorch 实现 - ajkdrag/architectures-impl-pytorch
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1239545576601948160)** (1 条消息): 

- **自定义 Tokenizer 报错**：一名成员分享了他们根据 [2021 年的教学视频](https://www.youtube.com/watch?v=MR8tZm5ViWU) 创建并训练自定义 Hugging Face Tokenizer 的经历。然而，他们遇到了多个错误，ChatGPT 将其归因于 Tokenizer 格式错误。

**提到的链接**：<a href="https://www.youtube.com/watch?v=MR8tZm5ViWU)">Building a new tokenizer</a>：学习如何使用 🤗 Tokenizers 库构建自己的 Tokenizer 并进行训练，然后如何在 🤗 Transformers 库中使用它。此视频是...的一部分。

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1239481803367256064)** (16 条消息🔥): 

- **从零开始实现 Stable Diffusion 引起社区兴趣**：一名成员分享了一个 [Fast.ai 课程](https://course.fast.ai/Lessons/part2.html)，该课程通过 30 小时的内容涵盖了从零开始构建 **Stable Diffusion 算法**。他们强调该课程包含了最新的技术，并与来自 Stability.ai 和 Hugging Face 的专家进行了合作。

- **关于生成式 AI 技术的书籍受到好评但尚未完成**：另一名成员对一本关于生成式媒体技术的[书](https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)发表了评论，指出了其潜力，并尽管该书尚未完成，仍表达了浓厚兴趣。

- **寻求在 macOS 上安装 sadtalker 的帮助**：一名用户寻求在 **macOS 上安装 sadtalker** 的紧急帮助。另一名成员建议在网上搜索错误消息，并分享了一个 [GitHub Issue 链接](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266)以提供帮助。

- **Inpainting 的使用说明与数据集创建**：成员们讨论了如何将 Inpainting 用于个人图像，并提供了 [Diffusers 文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint)链接。此外，还通过 [Hugging Face 指南](https://huggingface.co/docs/diffusers/main/en/training/create_dataset)提供了创建自定义数据集的指导。

- **寻求 Transformer Agent 的实际应用案例**：一名成员询问了使用 **Transformer Agent** 的项目示例，表达了对学术案例之外的应用兴趣。他们被引导至 Hugging Face 的[博客文章](https://huggingface.co/blog/agents)，但仍希望社区能提供更多实际应用案例。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/create_dataset">为训练创建数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/agents">License to Call: 介绍 Transformers Agents 2.0</a>: 未找到描述</li><li><a href="https://course.fast.ai/Lessons/part2.html">Practical Deep Learning for Coders - Part 2 概览</a>: 学习使用 fastai 和 PyTorch 进行深度学习，2022</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint">Inpainting</a>: 未找到描述</li><li><a href="https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/">Hands-On Generative AI with Transformers and Diffusion Models</a>: 在这本实用的实战指南中，学习如何使用 AI 生成媒体技术来创作新颖的图像或音乐。数据科学家和软件工程师将了解最先进的生成式...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266">[Bug]: ModuleNotFoundError: No module named &#39;torchvision.transforms.functional_tensor&#39; torchvision 0.17 问题 · Issue #13985 · AUTOMATIC1111/stable-diffusion-webui</a>: 是否已存在相关 Issue？我搜索了现有的 Issue 并检查了最近的构建/提交。发生了什么？ModuleNotFoundError: No module named &#39;torchvision.transforms.functiona...
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1239647747628597309)** (3 条消息): 

- **OpenRouter 发布 GPT-4o 和 LLaVA v1.6 34B**: 两个新的多模态模型现已在 OpenRouter 上可用。这些模型包括 [OpenAI: GPT-4o](https://openrouter.ai/models/openai/gpt-4o) 和 [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b)。

- **DeepSeek 和 Llama 模型加入库**: 增加了几个新模型，包括 [DeepSeek-v2 Chat](https://openrouter.ai/models/deepseek/deepseek-chat)、[DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder)、[Llama Guard 2 8B](https://openrouter.ai/models/meta-llama/llama-guard-2-8b)、[Llama 3 70B Base](https://openrouter.ai/models/meta-llama/llama-3-70b)、[Llama 3 8B Base](https://openrouter.ai/models/meta-llama/llama-3-8b) 以及 [2024-05-13 更新的 GPT-4o](https://openrouter.ai/models/openai/gpt-4o-2024-05-13)。

- **Gemini Flash 1.5 发布**: 一个名为 [Gemini Flash 1.5](https://openrouter.ai/models/google/gemini-flash-1.5) 的新模型已发布。这继续扩展了 OpenRouter 上提供的多样化产品。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-1.5)">Google: Gemini Flash 1.5 (preview) by google | OpenRouter</a>: Gemini 1.5 Flash 是一款基础模型，在视觉理解、分类、摘要以及根据图像、音频和视频创建内容等各种多模态任务中表现出色...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o)">OpenAI: GPT-4o by openai | OpenRouter</a>: GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平</li><li><a href="https://openrouter.ai/models/liuhaotian/llava-yi-34b)">LLaVA v1.6 34B by liuhaotian | OpenRouter</a>: LLaVA Yi 34B 是一款开源模型，通过在多模态指令遵循数据上微调 LLM 训练而成。它是一款基于 Transformer 架构的自回归语言模型。基础 LLM：[Nou...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat>)">DeepSeek-V2 Chat by deepseek | OpenRouter</a>: DeepSeek-V2 Chat 是 DeepSeek-V2 的对话微调版本，DeepSeek-V2 是一款混合专家（MoE）语言模型。它包含 236B 总参数，其中每个 token 激活 21B 参数。与 D 相比...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder>)">Deepseek Coder by deepseek | OpenRouter</a>: Deepseek Coder 由一系列代码语言模型组成，每个模型都基于 2T token 从零开始训练，其中包含 87% 的代码和 13% 的中英文自然语言。该模型...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-guard-2-8b>)">Meta: LlamaGuard 2 8B by meta-llama | OpenRouter</a>: 这款安全防护模型拥有 8B 参数，基于 Llama 3 系列。就像其前身 [LlamaGuard 1](https://huggingface.co/meta-llama/LlamaGuard-7b) 一样，它可以同时处理 prompt 和 response...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b>)">Meta: Llama 3 70B by meta-llama | OpenRouter</a>: Meta 最新的模型类别（Llama 3）推出了多种尺寸和版本。这是基础的 70B 预训练版本。与领先的闭源模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b>)">Meta: Llama 3 8B by meta-llama | OpenRouter</a>: Meta 最新的模型类别（Llama 3）推出了多种尺寸和版本。这是基础的 8B 预训练版本。与领先的闭源模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-05-13>)">OpenAI: GPT-4o by openai | OpenRouter</a>: GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1239608569716936725)** (1 条消息): 

- **高级研究助手 Beta 测试**：一位用户宣布推出一款高级**研究助手和搜索引擎**，正在寻求 Beta 测试参与者。“我可以为你提供 2 个月的免费高级版，包含 Claude 3 Opus、GPT-4 Turbo、Mistral Large、Mixtral-8x22B...”，他们提供了一个[促销代码](https://rubiks.ai/) RUBIX 以供访问。

- **GPT-4o 发布亮点**：分享了一个关于 OpenAI **GPT-4o 发布**的链接，将其标记为现有 AI 模型的一次重大升级。这表明社区对紧跟 OpenAI 发展的兴趣。

- **Mistral AI 60 亿美元估值新闻**：重点介绍了关于 **Mistral AI** 的信息，这是一家总部位于巴黎的初创公司，正以 60 亿美元的估值筹集资金。这突显了开发大语言模型公司的快速增长和关注度。

**提到的链接**：<a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1239479486702030868)** (278 条消息 🔥🔥): 

- **GPT-4o API 定价与热度**：讨论集中在 GPT-4o API 的可用性和定价上，价格为每 1M tokens $5/15。一位用户指出，*“太令人兴奋了……在 coding 方面的 ELO 排名显然比 Opus 高出 100 分。”*
  
- **GPT-4o 多模态能力推测**：用户推测了 GPT-4o 的能力，一些人质疑它是否能处理图像生成。*“通过 [OpenAI 的] API，不行。我的 Python 项目负责处理互联网方面的事情，并向 LLM 提供这些数据。”*

- **OpenRouter 的问题**：用户报告了 OpenRouter 的各种错误和问题，包括 MythoMax 的空响应以及 DeepSeek 的错误。*“DeepInfra 似乎仍然存在问题”* 以及 *“TypeError: Cannot read properties of undefined (reading 'stream')。”*
  
- **关于 OpenRouter 模型精度的讨论**：有人询问 OpenRouter 是否使用全精度模型，**Alex Atallah** 回复称几乎所有模型都是 FP16，只有少数例外（如 Goliath）是量化的（4-bit）。*“如果能把它添加到页面上就好了。”*

- **介绍社区工具**：一位社区成员介绍了一个用于探索和排序 OpenRouter 模型的工具，引发了积极的反响。*“噢，这太酷了”*，并讨论了集成额外指标（如 ELO 分数和抓取的模型添加日期）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.litellm.ai/">LiteLLM</a>：LiteLLM 处理 100 多个 LLM 的负载均衡、故障转移和支出跟踪。全部采用 OpenAI 格式。</li><li><a href="https://github.com/fry69/orw">GitHub - fry69/orw: 监控 OpenRouter 模型 API 的更改并将更改存储在 SQLite 数据库中。包含一个简单的 Web 界面。</a>：监控 OpenRouter 模型 API 的更改并将更改存储在 SQLite 数据库中。包含一个简单的 Web 界面。 - fry69/orw</li><li><a href="https://claudeai.uk/can-claude-read-pdf/">Can Claude Read PDF? [2023] - Claude Ai</a>：Claude 能阅读 PDF 吗？PDF (Portable Document Format) 文件是我们日常生活中经常遇到的一种通用文档类型。</li><li><a href="https://orw.karleo.net/removed">OpenRouter API Watcher</a>：OpenRouter API Watcher 监控 OpenRouter 模型的更改并将这些更改存储在 SQLite 数据库中。它每小时通过 API 查询模型列表。</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>：OpenRouter API Watcher 监控 OpenRouter 模型的更改并将这些更改存储在 SQLite 数据库中。它每小时通过 API 查询模型列表。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1239617300735459358)** (178 messages🔥🔥): 

- **OpenAI 推出具有令人印象深刻能力的 GPT-4o**：被 [Liam Fedus](https://x.com/liamfedus/status/1790064963966370209?s=46) 称为 *新的最先进前沿模型*。新模型在 LMSys arena 上表现出色，重点在于推理和编码。
- **Tokenizer 更新和 Token 容量增加**：引入了 [新 Tokenizer](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8)，据称将 Token 容量翻倍至 200k，从而提高了速度。Token 的增加是性能提升的原因之一。
- **实时演示和多模态能力**：GPT-4o 的实时演示展示了其能力，包括唱歌等潜在功能。一段 [YouTube 视频](https://www.youtube.com/watch?v=MirzFk_DSiI) 展示了 GPT-4o 的交互能力。
- **竞争格局和担忧**：讨论指向 OpenAI 保持对 Meta 等竞争对手竞争力的策略。有人猜测数据池饱和以及多模态改进与其他增强功能之间的平衡。
- **Google I/O 2024 关键更新**：Google 宣布了 [Gemma 模型](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/) 的新成员，包括即将发布的 Gemma 2 和其他 Gemini 增强功能。具有 27B 参数的 Gemma 2 代表了 Google AI 产品的重大进步。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livecodebench.github.io/leaderboard.html">LiveCodeBench 排行榜</a>：未找到描述</li><li><a href="https://x.com/liamfedus/status/1790064963966370209?s=46">来自 William Fedus (@LiamFedus) 的推文</a>：GPT-4o 是我们最新的 SOTA 前沿模型。我们一直在 LMSys arena 上以 im-also-a-good-gpt2-chatbot 的身份测试一个版本 🙂。这是它的表现。</li><li><a href="https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/">Gemini 1.5 Pro 更新，1.5 Flash 亮相以及 2 款新的 Gemma 模型</a>：今天我们正在更新 Gemini 1.5 Pro，推出 1.5 Flash，发布新的 Gemini API 功能并增加两款新的 Gemma 模型。</li><li><a href="https://x.com/lmsysorg/status/1790097595064529255?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：相对于所有其他模型，胜率显著提高。例如，在非平局对决中，对比 GPT-4 (June) 的胜率约为 80%。</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：突发新闻 —— gpt2-chatbots 的结果现已公布！gpt2-chatbots 刚刚飙升至榜首，以显著差距（约 50 Elo）超越了所有模型。它已成为 Arena 中有史以来最强的模型...</li><li><a href="https://x.com/kaiokendev1/status/1790068145933185038?s=46">来自 Kaio Ken (@kaiokendev1) 的推文</a>：是的，但它会呻吟吗？</li><li><a href="https://ai.google.dev/pricing">未找到标题</a>：未找到描述</li><li><a href="https://x.com/drjimfan/status/1790122998218817896?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：我纠正一下：GPT-4o 并不原生处理视频流。博客中说它只接收图像、文本和音频。这很遗憾，但我说的原则依然成立：制作视频的正确方式...</li><li><a href="https://x.com/google/status/1790055114272612771?s=46>)">来自 Google (@Google) 的推文</a>：距离 #GoogleIO 还有一天！我们感到非常 🤩。明天见，届时将分享关于 AI、Search 等的最新消息。</li><li><a href="https://www.youtube.com/watch?v=MirzFk_DSiI">两个 GPT-4o 互动并唱歌</a>：向 GPT-4o 问好，这是我们新的旗舰模型，可以实时跨音频、视觉和文本进行推理。在此了解更多：https://www.openai.com/index/hello-...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/">Google 发布 Gemma 2，其开放模型的 27B 参数版本，将于 6 月推出 | TechCrunch</a>：在 Google I/O 大会上，Google 介绍了下一代 Google 的 Gemma 模型 Gemma 2，将于 6 月推出 270 亿参数版本。
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1239691351629762630)** (3 条消息): 

- **REINFORCE 是 PPO 的一个特例**：一位成员分享了 [来自 Hugging Face 的 PR](https://github.com/huggingface/trl/pull/1540)，该 PR 实现了 **RLOO** 并解释了 **REINFORCE** 如何成为 **PPO** 的一个特例。与此讨论相关的论文可以在[这里](https://arxiv.org/pdf/2205.09123)找到。

- **Costa 在 RLOO 上的工作**：一位成员提到他们打算通过研究 RLOO 来为 TRL 做出贡献，结果发现 **Costa** 已经开始了这一过程。这次幽默的交流突显了社区持续的协作和努力。

**提及的链接**：<a href="https://github.com/huggingface/trl/pull/1540">vwxyzjn 提交的 PPO / Reinforce 训练器 · Pull Request #1540 · huggingface/trl</a>：此 PR 支持 https://arxiv.org/pdf/2402.14740.pdf 中的 REINFORCE RLOO 训练器。请注意，REINFORCE 的损失函数是 PPO 的一个特例，如下所示，它与展示的 REINFORCE 损失相匹配...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1239508832615399455)** (20 条消息🔥):

- **社区好奇 GPT-3.5 是否会开源**：一位成员调侃道，“除非太阳从西边出来（Guess then hell freezes over）”，反映了对这种可能性的怀疑。
- **对 AI 领导层的担忧**：一位成员表示对 Sam Altman 感到幻灭，提到了演示中“调情般的俏皮感”以及与电影《Her》的类比，认为这淡化了 AI 的严肃影响。
- **语言模型评估的可访问性**：分享了一篇关于 [LLM 评估的详细博客文章](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation)，质疑学术界和其他利益相关者获取评估工具的门槛。该文章强调了三种主要的 LLM 评估类型：MMLU 基准测试、ChatBotArena 面对面测试以及私有 A/B 测试。
- **用于长期 AI 项目的 PRMs**：链接到一段 [John Schulman 的 YouTube 视频](https://www.youtube.com/watch?v=1fmcdz2EO_c)，讨论了未来的模型可能更像同事而非搜索引擎，暗示了项目管理机器人（PRMs）在促进长期 AI 任务中的作用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation">ChatBotArena：大众的 LLM 评估、评估的未来、评估的激励机制以及 gpt2chatbot</a>：细节告诉了我们关于目前最流行的 LLM 评估工具以及该领域其他工具的现状。</li><li><a href="https://www.youtube.com/watch?v=1fmcdz2EO_c">2025 年的模型将更像同事而非搜索引擎 —— OpenAI 联合创始人 John Schulman</a>：完整剧集明天发布！在 Twitter 上关注我：https://twitter.com/dwarkesh_sp
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1239604019857588245)** (5 条消息): 

- **Nathan 计划在获得斯坦福大学许可后发布内容**：Nathan 提到他可以为了“个人使用”向斯坦福大学“申请许可”，并打算据此下载并发布。他还对任何潜在的后果表示怀疑。 
- **为了博客灵感重温电影**：Nathan 提到为了写博客重温了电影《Her》，称其“非常到位”且与他的写作高度相关。他提到目前可以在 HBO 上观看。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1239497467893121044)** (30 条消息🔥): 

- **探究网页抓取预训练数据集**：话题围绕最近的一篇 [arxiv 论文](https://arxiv.org/abs/2404.04125) 展开，该论文审视了多模态模型中“Zero-shot”泛化的概念。成员们讨论了其影响和局限性，强调它不适用于组合泛化（compositional generalization），并呼吁对大众化的解读保持谨慎。

- **Falcon2 11B 发布**：一个新的 11B 模型已发布，该模型在 5T 精炼网页数据上训练，具有 8k 上下文窗口和 MQA attention。由于采用了宽松的许可证，该模型有望提供更好的推理能力。

- **最佳 AI/Machine Learning GitHub 仓库**：成员们推荐了一些出色的 AI/ML GitHub 仓库，包括 **Lucidrains** 和 **equinox**。对话旨在找出最受欢迎且令人印象深刻的仓库。

- **Epistemic Networks 论文讨论**：关于在 Epistemic Networks 论文背景下，是否有必要将原始网络的输出添加到最终输出中及其影响，展开了积极讨论。成员们辩论了从基础网络添加残差（residual）是会使输出中心化，还是在尺度不匹配时带来风险。

- **图像生成模型的 RAG 咨询**：有人询问了目前使用 RAG 对图像生成模型进行推理时修改的实践。讨论考虑了诸如图像的 CLIP embedding 和用于 Prompt conditioning 的平均化等技术，但仍在寻找更好的替代方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/live/DQacCB9tDaw?feature=shared&t=3478">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 更新等。</li><li><a href="https://arxiv.org/abs/2404.04125">没有“指数级数据”就没有“Zero-Shot”：预训练概念频率决定多模态模型性能</a>：网页抓取的预训练数据集是多模态模型（如用于分类/检索的 CLIP 和用于图像生成的 Stable-Diffusion）令人印象深刻的“Zero-shot”评估性能的基础...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1239481316282466314)** (36 条消息🔥):

- **Linearattn 模型在 MMLU 基准测试中需要更多数据**：一位用户提到，“MMLU 是 Linearattn 模型面临的真正挑战。看来你不仅需要 7B 参数，还需要合适的数据”，并链接到了 [subquadratic LLM leaderboard](https://huggingface.co/spaces/devingulliver/subquadratic-llm-leaderboard) 以进行性能对比。
- **讨论了 Farzi 数据蒸馏方法**：提供了一篇关于[数据蒸馏论文](https://arxiv.org/abs/2310.09983)的详细摘要，题为“我们提出了 Farzi，它将事件序列数据集总结为少量合成序列——Farzi Data——在保持或提高模型性能的同时”。讨论还延伸到了扩展到更大模型和数据集时的实际约束。
- **Memory Mosaics 与关联记忆**：用户们辩论了 [Memory Mosaics 论文](https://arxiv.org/abs/2405.06394) 的影响，一些人对其与 Transformers 相比的有效性持怀疑态度。该研究因其组合能力和 In-context learning 能力而受到关注，“在规模适中的语言建模任务上，其表现与 Transformers 相当甚至更好”。
- **激活函数收敛问题**：一位用户询问了激活函数保证良好收敛的“必要但不充分”条件，引发了技术讨论。另一位用户指出激活函数中非线性的本质需求。
- **关于 FlashAttention2 (FA2) 并行化和拆分的讨论**：成员们就 FA2 与 Flash Infer 中拆分（splits）的并行化进行了详细的技术辩论。“FA2 现在也有一个执行 split KV 的内核”，这被视为一个重大的算法变化，暗示了向 FA3 的潜在演进。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06394">Memory Mosaics</a>：Memory Mosaics 是协同工作以实现特定预测任务的关联记忆网络。与 Transformers 类似，Memory Mosaics 具有组合能力和 In-context learning...</li><li><a href="https://huggingface.co/spaces/devingulliver/subquadratic-llm-leaderboard">Subquadratic LLM Leaderboard - a Hugging Face Space by devingulliver</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.09983">Farzi Data: Autoregressive Data Distillation</a>：我们研究了自回归机器学习任务的数据蒸馏，其中输入和输出具有严格的从左到右的因果结构。更具体地说，我们提出了 Farzi，它总结了...</li><li><a href="https://openreview.net/forum?id=H9DYMIpz9c&noteId=aN4DeBSr82">Farzi Data: Autoregressive Data Distillation</a>：我们研究了自回归机器学习任务的数据蒸馏，其中输入和输出具有严格的从左到右的因果结构。更具体地说，我们提出了 Farzi，它总结了...</li><li><a href="https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file">GitHub - yingpengma/Awesome-Story-Generation</a>：该仓库收集了大量关于故事生成/讲故事的优秀论文，主要关注大语言模型 (LLMs) 时代。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239488393713287199)** (119 messages🔥🔥): 

- **MLPs 挑战 Transformers 的主导地位**：成员们讨论了在视觉任务中改进基于 **MLP 模型** 以超越 **Transformers** 的努力 ([arxiv.org](https://arxiv.org/abs/2108.13002))，提出的混合架构展示了具有竞争力的性能。重点在于 MLPs 的潜在可扩展性和效率，尽管对其处理复杂先验的能力存在怀疑。
  
- **初始化是关键**：讨论强调了神经网络**初始化方案**的重要性，一些人建议有效的初始化可以显著提高 MLP 的性能 ([gwern.net](https://gwern.net/note/fully-connected#initialization))。使用图灵机或其他计算模型的合成初始化想法被提议为未来的研究方向。

- **拟态初始化显示出前景**：最近的一篇论文建议，拟态初始化（Mimetic Initialization）使权重类似于预训练的 Transformers，可以在**小数据集上训练 Transformers** 时显著提高准确性 ([proceedings.mlr.press](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf))。这种方法有助于 Transformers 以更快的训练时间实现更高的最终准确率。

- **关于效率和架构选择的争议**：成员们讨论了与 Transformer 相比，MLP 可能实现的效率提升，特别是针对 **A100** 和 **TPU** 等各种硬件设置上的 **Model FLOPs Utilization (MFU)**。一些人指出，即使 MFU 只有微小的提升，在大规模应用时也会产生重大影响。

- **Minsky 的争议性影响**：讨论包括对 **Marvin Minsky** 对神经网络研究的历史影响的反思，对于他的怀疑态度是否显著阻碍了进展，意见不一。文中提供了相关论文的链接和幽默的 AI Lab 公案 ([catb.org](http://www.catb.org/esr/jargon/html/koans.html))，为 Minsky 的遗产增添了背景。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2306.00946">Exposing Attention Glitches with Flip-Flop Language Modeling</a>：为什么大型语言模型有时会输出事实错误并表现出错误的推理？这些模型的脆弱性，特别是在执行长链推理时，目前看来...</li><li><a href="https://arxiv.org/abs/2210.03651">Understanding the Covariance Structure of Convolutional Filters</a>：神经网络权重通常是从单变量分布中随机初始化的，即使在像卷积这样高度结构化的操作中，也只控制单个权重的方差。Re...</li><li><a href="https://gwern.net/note/fully-connected#initialization">Fully-Connected Neural Nets · Gwern.net</a>：未找到描述</li><li><a href="http://www.catb.org/esr/jargon/html/koans.html">Some AI Koans</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2108.13002#microsoft">A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP</a>：卷积神经网络 (CNN) 是计算机视觉领域主流的深度神经网络 (DNN) 架构。最近，基于 Transformer 和多层感知器 (MLP) 的模型，如 Vision Tra...</li><li><a href="https://arxiv.org/abs/2306.13575">Scaling MLPs: A Tale of Inductive Bias</a>：在这项工作中，我们重新审视了深度学习中最基础的构建块——多层感知器 (MLP)，并研究了其在视觉任务上性能的极限。对 MLP 的实证见解是...</li><li><a href="https://github.com/edouardoyallon/pyscatwave">GitHub - edouardoyallon/pyscatwave: Fast Scattering Transform with CuPy/PyTorch</a>：使用 CuPy/PyTorch 的快速散射变换。通过在 GitHub 上创建账号来为 edouardoyallon/pyscatwave 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1239481105514758144)** (4 messages): 

- **最后一刻的 NeurIPS 投稿征集**：一位成员询问是否有人对最后一刻的 NeurIPS 投稿感兴趣，并提到了“类似于 Othello 论文的内容”。另一位成员表示尽管自己也有投稿要完成，但愿意提供帮助。
- **压缩对模型特征的影响**：一位成员询问在模型压缩过程中会丢失哪些类型的特征或电路。他们推测，如果这些特征是 **overspecialized**（过度专业化）而非无用的，它们可能有助于评估训练数据集的多样性。
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

oleksandr07173: Hello
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1239606952632520764)** (29 messages🔥): 

- **Mojo 编译器开发引起关注**：一位成员表达了对贡献 **Mojo compiler** 的兴趣，并寻求相关书籍或课程的建议。另一位成员澄清说 **Mojo compiler** 尚未开源。

- **Mojo 编译器的开发语言揭晓**：一位成员询问 **Mojo compiler** 是否是用 Mojo 编写的，另一位成员回答说它实际上是用 **C++** 编写的。讨论还涉及了未来在 **Mojo 中重建 MLIR** 的可能性。

- **Mojo 对 Python 的依赖引发疑问**：针对 Mojo 对 **Python 系统依赖** 的担忧被提出。会议澄清说，虽然目前为了兼容性是必要的，但在某些场景和未解决的问题中，工具链可以在不安装 Python 的情况下工作。

- **MLIR 与 Mojo 集成的细节**：提供了关于 **Mojo 如何与 MLIR 集成** 的示例和详细解释，展示了 Mojo 程序如何充分利用 MLIR 的可扩展性。该功能被认为对于将系统扩展到新的数据类型或硬件特性**极其强大**。

- **讨论 Mojo 编译器的自托管 (Self-Hosting)**：有人表达了对未来实现自托管 **Mojo 编译器**的期望。成员们持乐观态度，并指出了该语言与 **MLIR** 的兼容性及其丰富的特性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Mojo 中的底层 IR | Modular 文档</a>：了解如何使用底层原语在 Mojo 中定义你自己的布尔类型。</li><li><a href="https://github.com/modularml/mojo/issues/935">[功能请求] 通过 `mojo build` 构建的二进制文件无法在其他操作系统上直接运行 · Issue #935 · modularml/mojo</a>：查看 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级要求。你的请求是什么？你好，我尝试构建一个使用 numpy 的简单 Mojo 应用...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1239610040458154017)** (2 条消息): 

- **Modular 发布新推文**：Modular 发布了更新，可以通过 [这里](https://twitter.com/Modular/status/1790046377613144201) 访问。消息中未详细说明推文的具体内容。
- **来自 Modular 的进一步更新**：分享了 Modular 的另一条推文，可在 [这里](https://twitter.com/Modular/status/1790442405273161922) 查看。该推文的具体细节也未在消息中提供。
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1239603493745197056)** (4 条消息): 

- **Mojo CEO Chris Lattner 深入探讨所有权 (Ownership)**：Modular 发布了一个关于 [Mojo 中所有权的 YouTube 视频](https://www.youtube.com/watch?v=9ag0fPMmYPQ)，由 CEO Chris Lattner 主讲。视频描述中邀请观众加入他们的社区进行进一步讨论。

- **为 Mojo🔥 标准库做贡献**：另一个 [视频](https://www.youtube.com/watch?v=TJpFSSIts5Q) 宣布 Mojo 标准库现已开源。Modular 工程师 Joe Loser 指导观众如何开始使用 Mojo 进行贡献。

- **Modular 上传新视频**：Modular 定期在其 YouTube 频道更新内容。查看最新视频：[这里](https://www.youtube.com/watch?v=arZS5-plt2Q) 和 [这里](https://www.youtube.com/watch?v=nkWhnFNlguQ)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=9ag0fPMmYPQ">Mojo🔥：与 Chris Lattner 深入探讨所有权</a>：了解关于 Mojo 所有权的所有必要知识，由 Modular CEO Chris Lattner 深度解析。如果你有任何问题，请务必加入我们友好的社区...</li><li><a href="https://www.youtube.com/watch?v=TJpFSSIts5Q">为开源 Mojo🔥 标准库做贡献</a>：Mojo🔥 标准库现已开源。在本视频中，Modular 工程师 Joe Loser 讨论了你如何开始使用 Mojo 为 Mojo🔥 做出贡献...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1240007747890708540)** (1 条消息): 

- **参加首届 Mojo 社区会议**：一场面向 **Mojo 开发者、贡献者和用户**的社区会议定于 5 月 20 日星期一上午 10-11 点举行。会议将涵盖 Mojo 的精彩更新和未来会议计划——详情见 [这里](https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1)。
- **将 Mojo 会议添加到您的日历**：用户可以通过订阅 [社区会议日历](https://modul.ar/community-meeting) 将本次及未来的会议添加到自己的日历中。完整详情可在 [社区会议文档](https://modul.ar/community-meeting-doc) 中查看。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1.">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://modul.ar/community-meeting.">Google 日历 - 登录以访问和编辑您的日程</a>：未找到描述</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1239494816614715392)** (77 条消息🔥🔥): 

- **在 Mojo 中将参数限制为浮点类型**：一位用户询问如何在 Mojo 中将参数限制为仅限浮点类型。另一位用户建议使用 `dtype.is_floating_point()` 结合约束检查，并指向 [DType 文档](https://docs.modular.com/mojo/stdlib/builtin/dtype/DType#is_floating_point) 以获取更多信息。

- **Mojo 的 Ownership（所有权）演讲引发讨论**：多位用户讨论了 Mojo 与 Python 相比在 Ownership 模型上的挑战与优势，并对一次关于 Ownership 的内部演讲表示赞赏。对话包括了实际案例以及对 `borrowed`、`inout` 和 `owned` 等概念的解释。

- **Mojo 中的 Tuple（元组）解包**：用户探索了如何在 Mojo 中进行元组解包，并发现需要先声明元组才能进行解包。分享了示例代码片段以澄清语法。

- **从 Mojo 调用 C/C++ 库**：一位用户询问如何从 Mojo 调用 C/C++ 库，另一位用户提供了 [GitHub](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi) 上 FFI 推特教程的资源链接。

- **Mojo 中的 String 到 Float 转换**：一位用户询问如何在 Mojo 中将字符串转换为浮点数，并创建了一个 [Pull Request](https://github.com/modularml/mojo/pull/2649) 来添加此功能。另一位用户分享了包含相关示例的仓库，并指出其与 Mojo 的 Nightly 版本兼容。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/dtype/DType#is_floating_point">DType | Modular 文档</a>：表示 DType 并提供相关操作方法。</li><li><a href="https://doc.rust-lang.org/nomicon/subtyping.html">子类型与型变 - The Rustonomicon</a>：未找到描述</li><li><a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs 插件 | Marketplace</a>：为 Mojo 编程语言提供基础编辑功能：语法检查与高亮、注释和格式化。未来将添加新功能，敬请期待...</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics#python-style-reference-semantics">值语义 | Modular 文档</a>：关于 Mojo 默认值语义的解释。</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics#">值语义 | Modular 文档</a>：关于 Mojo 默认值语义的解释。</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi">devrel-extras/tweetorials/ffi at main · modularml/devrel-extras</a>：包含开发者关系博客文章、视频和研讨会的辅助材料 - modularml/devrel-extras</li><li><a href="https://github.com/modularml/mojo/pull/2649">[stdlib] 为 `String` 添加 `atof()` 方法，由 fknfilewalker 提交 · Pull Request #2649 · modularml/mojo</a>：此 PR 添加了一个可以将 String 转换为 Float64 的函数。目前仅针对 Float64 实现，但也许我们应该添加其他精度？支持以下表示法："-12..."</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: 简单且快速的 Mojo HTTP 框架！🔥</a>：简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号为 saviorand/lightbug_http 做出贡献。</li><li><a href="https://github.com/carlca/ca_mojo.git">GitHub - carlca/ca_mojo</a>：通过在 GitHub 上创建账号为 carlca/ca_mojo 做出贡献。</li><li><a href="https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil">Python 可变默认参数是万恶之源 - Florimond Manca</a>：如何防止一个常见的 Python 错误，该错误可能导致可怕的 Bug 并浪费所有人的时间。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1239475574125363211)** (27 条消息🔥): 

- **发现最快的 List 扩展方法**：一位用户报告了其基准测试脚本中的错误，并得出结论：通过循环遍历 `source_list` 并逐个追加元素比使用自定义或标准的 `extend` 更快。他们分享了示例代码片段进行说明。
- **Ubuntu 测试仍未解决**：Ubuntu 测试失败的问题尚无更新，CI 基础设施团队正在调查。讨论中提到了一个错误显示为“挂起（pending）”状态的 GitHub Actions 问题。
- **Nightly 发布变得更加频繁**：一个新的 `mojo` Nightly 版本已推送，其中包含内部自动合并的提交，使“每日 Nightly”成为现实。成员们幽默地将其比作电影《盗梦空间》（*Inception*）中的梗。
- **嵌套数组的 Segfault 问题**：一位用户报告了在 `mojo` 脚本中深度嵌套数组时出现 Segfault（段错误）。关于这是功能还是实现问题存在争论，包括使用 Span 迭代器的建议。
- **关于 Nightly 发布频率的辩论**：围绕减少已合并提交延迟的讨论表明，频繁推送可能会使所需的编译器版本变得复杂。共识倾向于保持 Nightly 版本之间 24 小时的间隔，以避免给用户带来不便。

**提及链接**: <a href="https://github.com/modularml/mojo/pull/2644">[CI] Add timeouts to workflows by JoeLoser · Pull Request #2644 · modularml/mojo</a>: 在 Ubuntu 测试中，由于最近 nightly 版本中的代码 bug（编译器或库中），我们发现了一些非确定性的超时。与其依赖默认的 GitHub 超时设置...

---

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1239503224730292224)** (13 messages🔥): 

- **Triton 在 FP16 和 FP8 中实现了速度提升**：一位用户注意到 Phil 在 Triton 教程中的更新，使 FP16 前向性能与 Kitten 实现相匹配。该更新的 commit 可以在[这里](https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac)找到。
- **关于 TMA 的讨论**：一位成员询问了 **TMA** (tensor memory accelerator)，并明确了它仅存在于 **Hopper (H100)** 中。另一位用户对软件版本的 TMA 表示了兴趣。
- **讨论 Triton 配置**：成员们讨论了添加新配置以增强 Triton 的性能。一位成员确认添加了新配置以实现更好的搜索。
- **分享速度基准测试**：分享了 Triton 在 **gh200** 上设置 **Casual=True 且 d=64** 的性能基准测试，显示在各种上下文中 FP16 和 FP8 都有显著提升。具体数据点包括：“Triton [FP16]”在 N_CTX 为 1024 时达到 252.747280，“Triton [FP8]”在 N_CTX 为 16384 时达到 506.930317。

**提及链接**: <a href="https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac">[TUTORIALS] tune flash attention block sizes (#3892) · triton-lang/triton@702215e</a>: 未找到描述

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1239868744906575992)** (7 messages): 

- **提高 L2 cache 命中率**：一位成员简单提到“同时也提高了 L2 cache 命中率”，可能与他们关于 CUDA 优化的讨论有关。

- **cuSPARSE 函数开销**：一位用户询问了 cuSPARSE 中调用 `cusparseCreateDnVec` 等函数的开销，询问由于重复操作，重用它们是否可行。他们特别质疑向量数据是否被缓存到其他地方，因为文档只提到了释放稠密向量描述符的内存。

- **clangd 处理 CUDA 文件的问题**：一位成员在让 `clangd` 正确解析 `.cu` 文件时遇到问题，尽管有 `compile_commands.json` 文件。他们报告说，在 Unix 上，无论是 VSCode 还是带有 clangd 扩展的 Neovim 似乎都无法工作。

- **尝试使用 cccl .clangd 解决**：另一位成员提供了来自 NVIDIA CCCL 的 [.clangd 文件链接](https://github.com/NVIDIA/cccl/blob/main/.clangd)作为潜在解决方案。然而，原帖作者指出，从 CUDA toolkit 切换到 NVHPC 可能是导致问题的原因，因为之前使用 CUDA toolkit 时运行正常。

**提及链接**: <a href="https://github.com/NVIDIA/cccl/blob/main/.clangd">cccl/.clangd at main · NVIDIA/cccl</a>: CUDA C++ 核心库。欢迎在 GitHub 上为 NVIDIA/cccl 的开发做出贡献。

---

**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1239718776279138375)** (10 messages🔥): 

- **torch.compile() 在 4090 上导致性能问题**：一位成员报告说“在单个 4090 上使用 torch.compile() 时，吞吐量和延迟显著下降（差了 4 倍）”。另一位成员要求提供最小复现示例，以及所使用的 tensor cores、CUDA graphs 和基准测试方法的详细信息。
- **动态张量分配的性能影响**：建议检查模型是否使用了动态分配的张量，例如通过 `torch.cat`，这会影响性能。来自 [OpenAI 的 Whisper 模型](https://github.com/openai/whisper/blob/main/whisper/model.py#L301)的一个示例说明了这个问题。
- **为带有自定义算子的网络创建图**：为了将自定义 Triton kernel 集成到网络架构中，建议创建自定义算子（custom ops）并使用 torch.compile 对其进行包装。
- **torch.compile 与 DeepSpeed 的兼容性**：有人询问 torch.compile 模型是否可以与 DeepSpeed 配合使用，并注意到最新的稳定版本可能不兼容。澄清结果是它应该可以工作，但不会追踪 collectives，并进一步询问了所遇到的具体 bug 的细节。

**提及链接**: <a href="https://github.com/openai/whisper/blob/main/whisper/model.py#L301)">whisper/whisper/model.py at main · openai/whisper</a>: 通过大规模弱监督实现鲁棒的语音识别 - openai/whisper

---

**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1239623296710873158)** (6 messages):

- **缺失构建依赖项难倒初学者**：一位用户指出在使用 `load_inline` 时遇到了缺失构建依赖项的问题，特别是提到了 **ninja** 等依赖。他们询问了获取所有重要工具的推荐方法，并询问是否有人能推荐一个带有良好 `requirements.txt` 的仓库。

- **神经网络输出作为 if 语句**：一位用户推测神经网络的输出可以按元素表示为 `x` 的函数，理论上可以用长 `if` 语句进行映射。他们质疑这种做法的实用性，怀疑由于 `if` 语句导致的 **warp divergence** 可能会引起显著的减速。

- **为什么长 if 语句不利于性能**：在后续讨论中，同一位用户思考是过度的 **warp divergence** 还是每个线程处理过多的 **FLOPS** 会成为该方法潜在性能问题的主要原因。

- **初学者寻求 PyTorch 中自定义 CUDA kernel 的资源**：一位用户请求在 PyTorch 中使用自定义 CUDA kernel 的资源，希望能有一个全面的概览。

- **学习 PyTorch 自定义 CUDA kernel 的有用链接**：另一位用户推荐了 [Jeremy 的 YouTube 讲座](https://youtu.be/4sgKnKbR-WE?si=00-k8KV5ESxqks3h)，题为 "Lecture 3: Getting Started With CUDA for Python Programmers"，以帮助初学者理解如何编写自定义 CUDA kernel。该视频包含用于进一步学习的 [补充内容](https://github.com/cuda-mode/lecture2/tree/main/lecture3)。

**提到的链接**：<a href="https://youtu.be/4sgKnKbR-WE?si=00-k8KV5ESxqks3h">Lecture 3: Getting Started With CUDA for Python Programmers</a>：录制于 Jeremy 的 YouTube https://www.youtube.com/watch?v=nOxKexn3iBo 补充内容：https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1239615740680605778)** (6 条消息): 

- **PMPP 作者活动 5 月 24 日**：PMPP 作者 **Izzat El Hajj** 将于 5 月 24 日谈论 *scan*。似乎有一个特色链接可以将用户引导至 Discord 上的活动。

- **高级 Scan 教程 5 月 25 日**：5 月 25 日，**Jake 和 Georgii** 将讨论如何使用 **CUDA C++ 构建高级 scan**。[活动详情点击此处](https://discord.com/events/1189498204333543425/1239607867666071654)。

- **链接失效问题**：成员们最初遇到了活动链接在移动端和 PC 端均失效的问题。该问题随后得到解决，并确认另一个链接可以正常工作。

**提到的链接**：<a href="https://discord.gg/gFDMmM96?event=1239607867666071654">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、闲逛并保持紧密联系。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

shikhar_7985: 从互联网的地下室里找到了一个老古董
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1239536441571278909)** (2 条消息): 

- **寻求官方解决方案资源**：一位成员询问是否有官方解决方案来验证其实现的数值正确性，并对其效率表示担忧。他们随后表示在 Misha 的帖子中找到了 Joey 的解决方案，“非常感谢！”
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1239501007709540434)** (89 条消息🔥🔥): 

- **ZeRO-1 提升 VRAM 效率**：通过实现 Zero Redundancy Optimizer (ZeRO-1)，实现了显著的 VRAM 节省，使得单 GPU batch size 从 4 增加到 10，几乎达到了 GPU 容量极限，并将训练吞吐量提升了约 54%。详细结果和配置可在 [PR 页面](https://github.com/karpathy/llm.c/pull/309) 查看。

- **梯度累积和 Bias Backward Kernel 更新**：更新了 backward bias kernel 以获得更好的性能和确定性，并合并了一个 PR 以解决梯度累积的问题。讨论内容包括各种方法，例如取消 atomics 以改用 warp shuffles，以及考虑更具确定性的方法。

- **HazyResearch/ThunderKittens 引起关注**：HazyResearch 项目的底层 CUDA tile primitives 库 [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 因其优化 LLM 性能的潜力而引起了开发者的注意，讨论突出了它与 Cutlass 和 Triton 等现有工具的异同。

- **CI 中的 GPU 测试讨论**：llm.c 持续集成 (CI) 流水线中缺乏 GPU 被视为一个缺陷，引发了将 GPU runner 集成到 GitHub Actions 中的讨论。GitHub 最近关于 CI runner 支持 GPU 的[公告](https://github.blog/changelog/2023-10-31-run-your-ml-workloads-on-github-actions-with-gpu-runners/)被视为一个潜在的解决方案。

- **浮点精度处理**：调试和确保确定性延伸到了浮点精度的处理，讨论了使用类似于 [Numpy 的 allclose](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose) 方法的相对和绝对容差，以提高测试准确性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#per-minute-rates-for-larger-runners">关于 GitHub Actions 的计费 - GitHub Docs</a>：未找到描述</li><li><a href="https://github.blog/changelog/2023-10-31-run-your-ml-workloads-on-github-actions-with-gpu-runners/">在带有 GPU runner 的 GitHub Actions 上运行你的 ML 工作负载</a>：在带有 GPU runner 的 GitHub Actions 上运行你的 ML 工作负载</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose">numpy.testing.assert_allclose — NumPy v1.26 手册</a>：未找到描述</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.as">numpy.testing.assert_allclose — NumPy v1.26 手册</a>：未找到描述</li><li><a href="https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpLoad.html#cub-warpload)">cub::WarpLoad &mdash; CUB 104.0 文档</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/issues/406">2D 和 3D tile 划分，以便从 threadIdx 和 blockIdx 读取置换坐标 · Issue #406 · karpathy/llm.c</a>：据推测，置换 kernel 即使主要受内存限制，也可以通过使用 2D 或 3D 网格来减少除法量并进行线程粗化（thread coarsening），从而无需在...中进行任何除法</li><li><a href="https://github.com/NVIDIA/cccl/issues/525).">Issues · NVIDIA/cccl</a>：CUDA C++ 核心库。通过创建账户为 NVIDIA/cccl 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/.github/workflows/ci.yml#L141">llm.c/.github/workflows/ci.yml (位于 2346cdac931f544d63ce816f7e3f5479a917eef5) · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过创建账户为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/309/commits/f613ce895b30dc0b2bd1f7e81410c6a2dcdce74d">Zero Redundancy Optimizer - Stage1 由 chinthysl 提交 · Pull Request #309 · karpathy/llm.c</a>：为了训练更大的模型变体（2B, 7B 等），我们需要为参数、优化器状态和梯度分配更大的 GPU 显存。Zero Redundancy Optimizer 引入了...的方法。</li><li><a href="https://github.com/karpathy/llm.c/pull/408">Layernorm backward 更新 由 ngc92 提交 · Pull Request #408 · karpathy/llm.c</a>：这修复了 layernorm backward pass 的梯度累积，并对 layernorm backward dev/cuda 文件进行了通用的现代化改造。容差已根据 float scratchpad 进行了调整...</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/main">GitHub - HazyResearch/ThunderKittens: 用于快速 kernel 的 Tile 原语</a>：用于快速 kernel 的 Tile 原语。通过创建账户为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://github.com/karpathy/llm.c/pull/309">Zero Redundancy Optimizer - Stage1 由 chinthysl 提交 · Pull Request #309 · karpathy/llm.c</a>：为了训练更大的模型变体（2B, 7B 等），我们需要为参数、优化器状态和梯度分配更大的 GPU 显存。Zero Redundancy Optimizer 引入了...的方法。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L689">llm.c/train_gpt2.cu (位于 master 分支) · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过创建账户为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1239603426082557982)** (2 条消息):

- **截止日期可能因纽约市的混乱而变动**：一位成员提到，由于纽约市的混乱局势，他们的工作可能会有所延迟。如果另一位成员准备好了编辑版本，他们将更新任务状态。
- **东部时间的晚间可用性**：另一位成员确认他们今晚可以进行编辑工作，并指明了时间为东部时间。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1239605875904217120)** (6 messages): 

- **Llama 3 使用案例巡展大放异彩**：为庆祝 Llama 3 黑客松，推出了一套全新的 cookbooks，展示了如何将 **Llama 3** 应用于 7 个不同的使用案例。详情可以在[这里](https://t.co/YLlsvkI0Ku)探索。

- **GPT-4o 自发布首日即获支持**：Python 和 TypeScript 从第一天起就提供了对 **GPT-4o** 的支持。用户可以通过 `pip` 进行安装，参考[详细说明](https://t.co/CMQ1aOXeWb)，并鼓励使用多模态集成。

- **GPT-4o 多模态演示**：一个简单的演示展示了 **GPT-4o** 在 Python 中令人印象深刻的多模态能力。点击[这里](https://t.co/yPMeyookRq)查看以用户爱犬为主角的演示。

- **GPT-4o 在 SQL 方面超越 GPT-4 Turbo**：在生成复杂的 SQL 查询时，**GPT-4o** 的速度是 GPT-4-turbo 的两倍。点击[这里](https://t.co/5k1tvKklGA)查看性能突破。

- **使用 llamafile 构建本地研究助手**：来自 Mozilla 的 llamafile 可以在你的笔记本电脑上实现私密的研究助手，无需安装。点击[这里](https://t.co/qFIA6j1OWe)了解更多关于这一创新工具的信息。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.co/zc00GjOmc4">未找到标题</a>: 未找到描述</li><li><a href="https://t.co/CMQ1aOXeWb">llama-index-llms-openai</a>: llama-index llms openai 集成</li><li><a href="https://t.co/1DLv8fikOi">llama-index-multi-modal-llms-openai</a>: llama-index multi-modal-llms openai 集成</li><li><a href="https://t.co/5k1tvKklGA">Google Colab</a>: 未找到描述</li><li><a href="https://t.co/yPMeyookRq">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1239518011799638116)** (104 messages🔥🔥): 

```html
- **`query` 方法中的元数据让用户感到困惑**：一位成员询问在将元数据嵌入 `TextNode` 后，是否必须在 `query` 方法期间传递元数据。澄清结果显示，**元数据过滤**可以由 LlamaIndex 内部处理，但任何特定用途（如 URL）必须手动添加。
- **前端响应中的 Unexpected token 错误**：一位用户遇到了前端在消息中途停止输出 AI 响应并显示 `Unexpected token U` 的问题。建议检查网络选项卡中的实际响应，或在解析前手动 `console.log` 响应内容。
- **Qdrant 向量和后处理器的错误处理**：一位用户尝试使用 Qdrant 向量存储创建新的后处理器时遇到了 `ValidationError`：期望 `BaseDocumentStore`。解决方案涉及在正确的上下文中正确识别并传递向量存储。
- **关于 LlamaIndex 实现更新的困惑**：成员们讨论了将 sec-insights 仓库和 LlamaIndex 从 0.9.7 更新到更新版本的问题。建议这可能主要涉及更新导入（imports），一位愿意协助版本升级变更的成员指出了这一点。
- **使用 LlamaIndex 构建求职助手**：分享了一篇关于使用 LlamaIndex 和 MongoDB 构建求职助手的文章，提供了详细的教程和项目仓库。该项目旨在利用 AI 驱动的聊天机器人和 **Retrieval-Augmented Generation** 来增强求职体验。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:11434',">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/cookbooks/llama3_cookbook_ollama_replicate/?h=llama3#7-agents">Llama3 Cookbook with Ollama and Replicate - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/ollama_cookbook/?h=multimodal">Multimodal Ollama Cookbook - LlamaIndex</a>: 未找到描述</li><li><a href="https://www.koyeb.com/tutorials/using-llamaindex-and-mongodb-to-build-a-job-search-assistant">Using LlamaIndex and MongoDB to Build a Job Search Assistant</a>: 了解如何使用 LlamaIndex、检索增强生成 (RAG) 和 MongoDB 构建职位搜索助手。</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/multimodal/">Multi-Modal Applications - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1239472557338660946)** (101 messages🔥🔥): 

- **Falcon 2 击败 Meta 的 Llama 3 8B**: [Falcon 2 11B](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas) 模型的表现超越了 Meta 的 Llama 3 8B，并与 Google 的 Gemma 7B 持平，这一点已通过 Hugging Face Leaderboard 验证。该模型支持多语言，是目前唯一具有 vision-to-language 能力的 AI 模型。
- **GPT-4o 发布，功能令人印象深刻**: 全新的 [GPT-4o](https://www.techopedia.com/openais-gpt-4o-release) 模型已发布，提供实时通信和视频处理功能。该版本显著提升了 API 性能，成本降低了一半，且速度可与人类对话相媲美。
- **关于图像生成模型 RAG 的讨论**: 一场关于图像生成模型 RAG 的对话引用了 [RealCustom 论文](https://arxiv.org/abs/2403.00483) 以实现文本驱动的图像转换，并将 IP Adapter 作为重要工具。此外，[Stable Diffusion](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned) 被指出可以接受 CLIP image embeddings 而非 text embeddings。
- **HunyuanDiT 声称达到 SOTA**: 腾讯发布了 [HunyuanDiT 模型](https://huggingface.co/spaces/multimodalart/HunyuanDiT)，据称是 SOTA 的开源 Diffusion Transformer 文本生成图像模型，在中文 prompts 方面表现尤为出色。尽管模型较小，但它展现了良好的 prompt 遵循能力和生成质量。
- **用于逼真说话面孔的 AniTalker**: 全新的 [AniTalker 框架](https://x-lance.github.io/AniTalker/) 能够利用静态图像和输入音频，从单张肖像中生成说话面孔动画。它捕捉了超越简单口型同步的复杂面部动态。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - multimodalart 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://fxtwitter.com/multimodalart/status/1790309209193509326?t=ryXEhFyHMWx5xwfWM8qAlA&s=19">来自 apolinario (multimodal.art) (@multimodalart) 的推文</a>: 首个开源的类 Stable Diffusion 3 架构模型刚刚发布 💣 —— 但它不是 SD3！🤔 它是腾讯开发的 HunyuanDiT，一个拥有 15 亿参数的 DiT (diffusion transformer) 文本生成图像模型 🖼️✨ 在...</li><li><a href="https://x-lance.github.io/AniTalker/">AniTalker</a>: 未找到描述</li><li><a href="https://civitai.com/models/435669?modelVersionId=502675">Bunline - v0.4 | Stable Diffusion Checkpoint | Civitai</a>: PixArt Sigma XL 2 1024 MS 在约 3.5 万张最大宽高大于 1024px 的图像上，使用自定义字幕进行了全量微调。说明：将 .safetensors 文件放置在...</li><li><a href="https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned">lambdalabs/stable-diffusion-image-conditioned · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/CompVis/stable-diffusion-v-1-3-original">CompVis/stable-diffusion-v-1-3-original · Hugging Face</a>: 未找到描述</li><li><a href="https://fxtwitter.com/GoogleDeepMind/status/1790434750592643331?t=gliMAi7wtzSx9s4HKnZJGA&s=19">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>: 我们推出了 Imagen 3：这是我们迄今为止质量最高的文本生成图像模型。🎨 它生成的视觉效果具有惊人的细节、逼真的光影，且干扰伪影更少。从快速草图...</li><li><a href="https://arxiv.org/abs/2403.00483">RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization</a>: 文本生成图像定制化旨在为给定主体合成文本驱动的图像，近期彻底改变了内容创作。现有工作遵循伪词范式，即代表...</li><li><a href="https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas">Falcon 2：阿联酋技术创新研究院发布新 AI 模型系列，性能超越 Meta 的新 Llama 3</a>: 未找到描述</li><li><a href="https://github.com/CompVis/stable-diffusion">GitHub - CompVis/stable-diffusion: 一个潜空间文本生成图像扩散模型</a>: 一个潜空间文本生成图像扩散模型。通过在 GitHub 上创建账号来为 CompVis/stable-diffusion 的开发做出贡献。</li><li><a href="https://github.com/cubiq/Diffusers_IPAdapter">GitHub - cubiq/Diffusers_IPAdapter: 为 HF Diffusers 实现的 IPAdapter 模型</a>: 为 HF Diffusers 实现的 IPAdapter 模型 - cubiq/Diffusers_IPAdapter</li><li><a href="https://tenor.com/bR79n.gif">Silicon Valley Tip To Tip GIF - Silicon Valley Tip To Tip Brainstorm - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1240009909261697035)** (3 条消息): 

- **DeepMind 的 Veo 树立了视频生成的新标杆**: [Veo](https://deepmind.google/technologies/veo/) 是 DeepMind 最先进的视频生成模型，可生成分辨率为 1080p、时长超过一分钟的视频，并支持多种电影风格。旨在降低视频制作门槛，它很快将通过 [VideoFX](https://labs.google/videofx) 向特定创作者开放，目前已开启[候补名单](https://labs.google/VideoFX)。
- **研究演示未考虑移动端用户**: 一位成员感叹，研究演示通常不压缩视频，导致移动端用户难以访问。

**提到的链接**: <a href="https://deepmind.google/technologies/veo/">Veo</a>: Veo 是我们迄今为止功能最强大的视频生成模型。它能以多种电影和视觉风格生成高质量、1080p 分辨率且时长可超过一分钟的视频。

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1239556169559703646)** (52 条消息🔥):

- **GPT-4o 令人印象深刻但成本较高**：一位成员对 **Llama 3 70b** 相比 **GPT-4** 的表现表示不满，提到他们一天在 OpenAI 费用上花费了 20 美元，并且在尝试过 GPT-4 后对其他模型感到索然无味。
- **Open Interpreter 兼容 GPT-4o**：用户讨论了 **GPT-4o** 与 Open Interpreter 的功能，其中一位提到 *“任何想在 OI 上尝试它的人，它都能正常工作”*，使用命令 `interpreter --model openai/gpt-4o`。 
- **GPT-4o 的速度优势**：据报告，**GPT-4o** 模型提供了惊人的 **100 tokens/sec**，而 GPT-4-turbo 仅为 **10-15 tokens/sec**，且价格仅为一半，这使其在模型性能上有了显著提升。
- **自定义指令导致问题**：一些用户在使用 **GPT-4o** 时遇到了问题，原因是数月前设置的自定义指令导致其运行异常，直到调整指令后才恢复正常。
- **实现 AGI 指日可待？**：关于迈向 AGI (*Artificial General Intelligence*) 的进展有一场推测性讨论，一位成员分享了 [Perplexity AI 对 AGI 解释的链接](https://www.perplexity.ai/search/ELI5-what-AGI-1Q1AM436TE.qHZyzUWHhyQ)。
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1239543477893595166)** (18 messages🔥): 

- **社区渴望 TestFlight 发布**：成员们正热切期待 TestFlight 的发布，预计在 Apple 审批流程完成后会有更新。有人提到：“Testflight 应该在今天晚些时候上线，正在等待 Apple 的批准。”

- **Xcode 中的 Bundle Identifier 设置**：成员们讨论了在 Xcode 中设置 Signing Team 和 Bundle Identifier 以编译项目。有人给出了有用的澄清：“当你在 Xcode 中打开文件时，你需要在 target 文件下更改一个设置才能进行编译。”

- **下一批次的出货时间表**：大家共同关注 O1 设备的出货时间表，第一批预计在 11 月。一位成员询问：“有人知道下一批 01 什么时候发货吗？”并得到了关于时间表的确认。

- **对 MacOS 中 AI 集成的推测**：一些用户在最近的演示后推测 OpenAI 可能会集成到 MacOS 中。虽然一位成员对全面集成表示乐观，但另一位建议道：“我认为 Apple 不会那样做，我敢打赌他们希望 AI 在机器本地运行。”

- **偏好开源 AI 解决方案**：有人表达了对开源 AI 解决方案的偏好，而非像 Apple 那样的专有方案。“即使 Apple 将 AI 集成到他们的操作系统中，我还是更倾向于开源，”这引发了使用 Linux 的建议。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1239480926228971551)** (47 messages🔥): 

- **ChatGPT 的矛盾令用户沮丧**：一位用户对 ChatGPT 最近在回答中自相矛盾的倾向表示沮丧。他们指出它“以前至少会坚持自己的说法并一直误导（gaslight）下去，现在它甚至无法在立场上做出决定。”

- **已弃用的 LLCHAIN 问题**：多位用户讨论了在 LLCHAIN 被弃用后遇到的问题。切换到 `from langchain_community.chat_models import ChatOpenAI` 解决了一些问题，但在流式传输和调用顺序链（sequential chains）时遇到了新问题。

- **LangChain Agent 调用缓慢**：一位用户报告称，LangChain Agent 在处理 300-400 字的大量输入时耗时过长，达到 2-3 分钟。另一位用户建议通过并行架构处理工作负载以提高速度。

- **AI/ML GitHub 仓库推荐**：成员们分享了他们最喜欢的 AI/ML GitHub 仓库，其中 **llama.cpp** 和 **deepspeed** 脱颖而出。

- **Socket.IO 用于流式传输 LLM 响应**：提供了一个关于如何将 `python-socketio` 与 LangChain 集成以将响应流式传输到前端的详细示例。它涵盖了用于管理 token 流和确认的服务器端和客户端实现。

**提及链接**：<a href="https://github.com/langchain-ai/langchain/issues/4118>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1239744561853632633)** (1 messages): 

- **关于响应优化前缀的查询**：一位成员询问是否需要像 `<|begin_of_text|><|start_header_id|>system<|end_header_id|>` 这样的前缀来获得最佳响应。消息记录中没有提供回复或额外的上下文。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1239759451679096852)** (5 messages):

- **Plug-and-Plai 集成助力 LLM 性能飞跃**：查看这篇 [Medium 文章](https://medium.com/ai-advances/supercharge-your-llms-plug-and-plai-integration-for-langchain-workflows-d471b2e28c99)，了解如何在 LangChain 工作流中使用 Plug-and-Plai 来增强 LLM 性能。该集成旨在提升在各种应用中部署大语言模型的便捷性。

- **使用 Streamlit 和 GPT-4o 的多模态聊天应用令人惊叹**：一位成员分享了他们的 [Hugging Face space](https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o)，展示了一个多模态聊天应用。该应用结合了 Streamlit、LangChain 和 OpenAI 的 GPT-4o，支持图片上传和剪贴板直接粘贴到聊天消息中。

- **RAG 应用扩展挑战**：开发者 Sivakumar 使用 LangChain 和 ChromaDB 作为 vector store 构建了一个 RAG 应用程序，并正在寻求将其扩展到生产级别的建议。他们希望获得相关见解和建议，以使应用程序达到生产就绪（production-ready）状态。

- **OranClick AI 写作流发布**：OranAITech 在一条 [推文](https://x.com/OranAITech/status/1790259419034390886) 中宣布了他们全新的 AI 写作流。该工具旨在通过追踪链接点击并利用 AI 支持优化文案创作，从而增强消息的有效性。

- **Snowflake 成本监控工具寻求反馈**：一款使用 LangChain、Snowflake Cortex 和 OpenAI 构建的新型 [Snowflake 成本监控与优化工具](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064) 正在开发中。该工具利用多个 AI Agent 来优化额度使用，并自动选择相关的数据可视化方案，目前仍处于开发阶段。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064">Crystal Cost Demo</a>：在此视频中，我简要演示了 Crystal Cost，这是一个 AI 驱动的 Streamlit 应用，可简化数据仓库上的数据监控。Crystal Cost 使用自然语言处理和 Agent 来查询数据...</li><li><a href="https://x.com/OranAITech/status/1790259419034390886">Adi Oran (@OranAITech) 的推文</a>：你是否厌倦了不知道自己的消息是否会被点击？但你又想轻松地加倍利用有效的消息传递。那么是时候了解 OranClick 了，追踪你的链接点击并编写最佳文案...</li><li><a href="https://huggingface.co/spaces/joshuasundance/streamlit-gpt4o">streamlit-gpt4o - joshuasundance 的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1239550983319846983)** (2 条消息): 

- **构建你自己的博客聊天机器人**：一位成员分享了他们的 [博客文章](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog)，解释了他们如何为自己的网站构建聊天功能，允许访客根据之前的博客文章提问。他们提供了必要的代码，包括数据处理、服务端 API 和客户端聊天界面，使用了带有引用的 **RAG (Retrieval Augmented Generation)** 技术。
- **寻求会话处理和流式传输教程**：另一位成员询问是否有关于使用 LangChain 处理历史记录、管理会话以及启用流式传输（streaming）的教程。他们提到，尽管遵循了当前的文档，但在实现流式传输方面仍感到困难。

**提到的链接**：<a href="https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog">使用 LangChain, OpenAI 和 Pinecone 为你的博客构建 RAG 流水线</a>：即使我不在身边，你也可以通过聊天了解我的文章，并向我询问我已经回答过的问题。

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1239569668201513020)** (24 条消息🔥): 

- **Substack 还是 Bluesky?**：一位成员询问在写博客时应该使用哪个平台，**Substack** 还是 **Bluesky**。另一位成员澄清说，虽然 **Bluesky** 支持帖子串（threads），但目前还不支持完整的博客功能。

- **AI 算力消耗审查**：成员们讨论了 AI 巨大的算力（compute）消耗，并分享了几个专注于降低算力负载的近期工作链接，例如 [Based](https://www.together.ai/blog/based) 和 [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)。

- **GPT-4o 炒作评论**：分享了一个炒作 **GPT-4o** 的 YouTube 视频，展示了会“唱歌”的 GPT-4o 以及在音频、视觉和文本方面的能力。有人提到，尽管存在炒作，但该产品可能主要吸引那些愿意支付 GPT-4 turbo 价格的用户。

- **为 GPT-4o 上的 OpenOrca 去重寻求赞助**：一位成员正在为在 GPT-4o 上重新运行 **OpenOrca dedup**（去重）寻找赞助商，预计 70M input tokens 的成本约为 350 美元，30M output tokens 的成本约为 300 美元。他们强调，如果作为批处理任务（batch job）运行，可能会获得折扣。

- **发表论文的挑战**：成员们讨论了发表论文漫长且具有挑战性的过程，并指出论文在发表时往往已经过时。一位成员的经历说明了这一点：他的 Ph.D. 只需要两篇论文，但目前只有一篇被接收。

- **训练 cmdR+ 100b 模型**：一位成员表示希望训练 **cmdR+ 100b 模型**，但指出 **Axolotl** 并不支持它。另一位成员建议训练基础模型（base model）可能更有益，因为 cmdR+ 已经经过了指令微调（instruction-tuned）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup?">Open-Orca/SlimOrca-Dedup · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=MirzFk_DSiI">两个 GPT-4o 互动并唱歌</a>：向 GPT-4o 问好，这是我们新的旗舰模型，可以实时跨音频、视觉和文本进行推理。在此了解更多：https://www.openai.com/index/hello-...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1239690259931856906)** (8 条消息🔥): 

- **过时的依赖项令用户沮丧**：一位用户对包括 **peft 0.10.0, accelerate 0.28.0, deepspeed 0.13.2** 等在内的过时依赖项表示不满，并指出 *"此配置默认安装 torch 2.0.0，而我们已经有了 2.3.0。"*
- **手动更新依赖项**：尽管建议将依赖项更新到最新版本以获得更好的兼容性，但该用户提到由于与 **accelerate FSDP 插件**存在兼容性问题，需要直接从仓库安装 **peft**，并通过 GitHub releases 的 `.whl` 文件安装 **flash-attn**。
- **Pull Request 提示**：面对为新版本提交 Pull Request (PR) 的请求，该用户表现出犹豫，理由是在不同环境中进行测试存在困难，但确认在他们那边将包更新到最新的稳定版本是可行的。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1239685277182853191)** (2 条消息): 

- **更新 pip 依赖项**：一位成员建议更新 **pip 依赖项**可能会解决特定的错误。另一位成员确认遇到了同样的错误，并暗示这可能是一个解决方案。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1239989093669277726)** (1 条消息): 

- **8xH100 配置中的 CUDA 错误问题**：一位用户最初报告在 Runpod PyTorch 容器和 `winglian/axolotl:main-latest` 中遇到了 **CUDA 错误**。经过修改后，他们更新称该配置在 **community axolotl cloud image** 上可能正常工作。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1239580327236866088)** (1 条消息): 

- **在没有精度问题的情况下将 QLoRA 合并到基础模型**：一位用户询问了 *"在没有精度问题（fp16/32）的情况下将 QLoRA 合并到基础模型"* 的步骤。这突显了开发者在转换精度格式时确保模型准确性的共同关注点。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1239575259015938101)** (7 条消息): 

- **将 QLoRA 与基础模型合并**：一位成员询问了将 QLoRA 合并到基础模型的过程，表示对模型集成技术感兴趣。
- **使用 Checkpoints 恢复训练**：用户讨论了如何使用 OpenAccess-AI-Collective/axolotl 代码库中的 `ReLoRACallback` “在之前训练 LoRA 时从 checkpoint 恢复”。详细步骤包括初始化训练环境、配置和加载 checkpoints 以及启动训练过程。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=121d871c-06a2-4494-ab29-60a3a419ec5e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1239677453010604053)** (29 条消息🔥): 

- **语音助手的傻笑让用户失望**：用户对语音助手的傻笑功能表示失望，称其为“令人尴尬的选择”。建议包括使用诸如“作为一个不傻笑的语音助手进行操作”之类的 prompts 来缓解这个问题。

- **GPT-4o 在列出图书馆书籍清单方面表现不佳**：一位用户分享了对 GPT-4o 在“列出此书架上所有书籍”测试中表现的不满，称其正确率仅约 50%，漏掉了多个书名，尽管对其速度和定价表示了赞赏。

- **关于 AGI 预期和模型进展的辩论**：讨论集中在对 AGI 的怀疑上，一些用户认为 AGI 并非迫在眉睫，且从 GPT-3 到 GPT-4 的模型进步收益递减。一位用户提到，围绕 GPT-5 的炒作掩盖了 GPT-4 等现有模型中尚未实现的潜力。

- **GPT-4 和 GPT-4o 的长期影响**：大家达成共识，认为 GPT-4 等模型的长期影响仍是未知数，且大多数人尚未体验过它们的能力。一位用户幽默地建议，如果 AGI 的定义是能够“略显拙劣地完成任何任务”，那么 AGI 在 GPT-3 时代就已经实现了。
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 messages): 

simonw: https://twitter.com/simonw/status/1790121870399782987
  

---



**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1239481652078706718)** (24 messages🔥): 

- **Tinygrad 的 CUDA 支持受到质疑**：一位成员询问了在 Nvidia 4090 上 `CUDA=1` 和 `PTX=1` 的预期行为，并分享了在 PTX 生成和模块加载过程中遇到的错误。另一位成员建议将 Nvidia 驱动程序更新到 550 版本以解决该问题。
  
- **在 tinygrad 中讨论图神经网络 (GNN)**：对话涉及了 tinygrad 中 GNN 的实现，并将其与现有的 PyTorch 解决方案（如 [PyG](https://www.pyg.org/)）进行了比较。一位成员指出，“在这种情况下，它归结为一个 O(N^2) 的 CUDA kernel”，并分享了 [GitHub 链接供参考](https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cuda/radius_cuda.cu)。
  
- **Tinygrad 的聚合操作及其局限性**：一位成员分享了一个用于特征聚合的 Python 函数及其在 tinygrad 中面临的挑战，但在高级索引（advanced indexing）和通过 `where` 调用进行反向传播时遇到了问题。建议的解决方案包括使用 masking 和 `einsum` 函数，但尚不清楚是否涵盖了所有边缘情况。
  
- **应对 tinygrad 高级功能的挑战**：讨论包括应对 `setitem` 和 `where` 等高级功能，一位成员表示“目前不支持任何带有高级索引（使用列表或张量进行索引）的 setitem”。提出并测试了多种变通方法，包括 masking 和 einsum。
  
- **探索 tinygrad 优化**：一些成员正在尝试优化 tinygrad 中的 conv2d 反向传播。一位成员指出 scheduler 和 view 更改会影响形状兼容性，并质疑重新实现 conv2d 是否是更好的方法。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49">test_aggregate.py</a>: GitHub Gist: 即时分享代码、笔记和片段。</li><li><a href="https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cuda/radius_cuda.cu">pytorch_cluster/csrc/cuda/radius_cuda.cu at master · rusty1s/pytorch_cluster</a>: 优化图集群算法的 PyTorch 扩展库 - rusty1s/pytorch_cluster</li><li><a href="https://github.com/torchmd/torchmd-net/blob/75c462aeef69e807130ff6206b59c212692a0cd3/torchmdnet/extensions/neighbors/neighbors_cpu.cpp#L71-L80">torchmd-net/torchmdnet/extensions/neighbors/neighbors_cpu.cpp at 75c462aeef69e807130ff6206b59c212692a0cd3 · torchmd/torchmd-net</a>: 神经网络势能。通过创建 GitHub 账号为 torchmd/torchmd-net 做出贡献。</li><li><a href="https://www.pyg.org/)">Home - PyG</a>: PyG 是图神经网络的终极库。
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1239605375242600519)** (17 messages🔥):

- **寻求德语 TTS 训练的帮助**：一名成员请求协助编译一份包含高质量播客、新闻、博客等的德语 YouTube 频道列表。*“Hätte jemand Zeit und Lust dabei zu helfen so eine Liste zusammenzustellen?”*（有人有时间和兴趣帮忙整理这样一份清单吗？）
- **用于德语媒体内容的 MediathekView**：另一位成员建议使用 [MediathekView](https://mediathekview.de/) 从各种德语在线媒体库下载节目和电影，如果可用的话，还可以包含字幕文件。他们分享了[热门德语播客](https://podtail.com/de/top-podcasts/de/)和[顶级德语 YouTube 频道](https://hypeauditor.com/top-youtube-all-germany/)的链接。
- **MediathekView 使用见解**：讨论还涵盖了下载整个 MediathekView 数据库以及使用 JSON API 访问内容的潜在方法，并提供了来自 [GitHub](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua) 的额外来源。
- **新德语 Tokenizer 效率**：一位用户强调了新 "o200k_base" Tokenizer 的效率，与旧的 "cl100k_base" 相比，处理相同的德语文本仅需 82.2% 的 Token 数量。他们还指出了新 Tokenizer 相对于 Mistral 和 Llama3 等 Tokenizer 的性能表现。
- **Tokenizers 研究资源**：对于那些对 Tokenizers 进一步研究感兴趣的人，分享了一个名为 [Tokenmonster](https://github.com/alasdairforsythe/tokenmonster) 的项目。该项目专注于子词 Tokenizers 以及针对各种编程语言的词汇训练。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09">Susan Zhang (@suchenzang) 的推文</a>：这个用于 gpt-4o 的新 “o200k_base” 词汇表让我感到非常震惊</li><li><a href="https://fxtwitter.com/main_horse/status/1790099796193398831">main (@main_horse) 的推文</a>：“为什么 gpt-4o 的演示看起来那么饥渴？”</li><li><a href="https://github.com/alasdairforsythe/tokenmonster">GitHub - alasdairforsythe/tokenmonster: 适用于 Python, Go &amp; Javascript 的非贪婪子词 Tokenizer 和词汇训练器</a>：适用于 Python, Go &amp; Javascript 的非贪婪子词 Tokenizer 和词汇训练器 - alasdairforsythe/tokenmonster</li><li><a href="https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua">MediathekViewWebVLC/mediathekviewweb.lua at main · 59de44955ebd/MediathekViewWebVLC</a>：用于 VLC 的 MediathekViewWeb Lua 扩展。通过在 GitHub 上创建账户为 59de44955ebd/MediathekViewWebVLC 做出贡献。</li><li><a href="https://podtail.com/de/top-podcasts/de/">目前最受欢迎的 100 个播客 – 德国</a>：此列表显示了当前最受欢迎的 100 个播客，包含来自 Apple 和 Podtail 的最新数据。</li><li><a href="https://hypeauditor.com/top-youtube-all-germany/">德国顶级 YouTube 频道 | HypeAuditor YouTube 排名</a>：查找截至 2024 年 5 月德国最受欢迎的 YouTube 频道。获取德国最大的 YouTuber 列表。
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1239486812523593798)** (8 条消息🔥): 

- **社区频道的支持请求**：一位用户表示在特定频道（<#1168411509542637578> 和 <#1216947664504098877>）寻求支持时遇到困难，提到缺乏回应。另一位用户向他们保证 Cohere 的员工是在线的且很活跃，并询问了更多细节。
- **对 Command R 能力的赞赏**：一位用户对 Command R 的 RAG 能力表示高度满意，强调了其经济性、准确性以及对长源文档的忠实度。他们指出，尽管源文档长度很长，但其表现仍令他们“印象极其深刻”。
- **问候与表情符号**：频道内进行了常规的问候交流，包括一句 "hello" 和使用表情符号 "<:hammy:981331896577441812>"。
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1239921278937333771)** (2 条消息): 

- **Vedang 寻求项目合作**：一名成员表示有兴趣合作开展一个与另一位用户正在进行的类似项目。他们说：“嗨 Asher，我也在做同样的事情。我想合作。”

- **Amit 分享关于 RAG 学习的 Medium 文章**：一名成员分享了他们关于使用 Unstructured API 从头开始学习 RAG 的 [Medium 文章](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2)链接。该文章侧重于以结构化格式从 PDF 中提取内容。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1239608271225098290)** (6 条消息):

- **LLM 对决：Claude 3 Haiku vs Llama 3b**：成员们讨论了 **Claude 3 Haiku** 与 **Llama 3b Instruct** 的潜在用例和比较优势。一位用户特别感兴趣于使用这些模型构建自动化评分服务，利用 **a Pydantic model** 从文档中提取并匹配实体。

- **LLM 中的受限采样 (Constrained Sampling)**：建议在 **vllm** 或 **sglang** 中使用 **outlines** 进行 **constrained sampling**。这被推荐为讨论中的实体匹配和评分任务的一种潜在有效方法。
  

---


**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1239613653767028868)** (3 messages): 

- **OpenAI 春季发布会成为焦点**：分享了一个 [YouTube 视频链接，标题为 "Introducing GPT-4o"](https://www.youtube.com/watch?v=DQacCB9tDaw)，重点介绍了 OpenAI 在 2024 年 5 月 13 日的直播更新，其中包括 ChatGPT 的更新。
- **Scarlett Johansson 为 GPT-4o 配音**：一位成员对选择 Scarlett Johansson 作为 GPT-4o 的配音表示惊讶和有趣。

**提到的链接**：<a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>：OpenAI 春季更新 —— 于 2024 年 5 月 13 日星期一直播。介绍 GPT-4o、ChatGPT 的更新等。

  

---



**Skunkworks AI ▷ #[announcements](https://discord.com/channels/1131084849432768614/1139357591701557258/1239862029632929863)** (1 messages): 

- **Guild Tags 引入新的用户标识符**：从 5 月 15 日开始，一些成员可能会在用户名旁边注意到 **Guild Tags**，这表示他们是名为 Guilds 的专属服务器成员。管理员应注意，如果启用了 AutoMod，它也会检查这些标签。
- **Guilds 是专属社区**：Guilds 是小型专属服务器，成员可以在其中分享共同的身份、爱好和游戏风格。目前，Guilds 仅对有限数量的服务器开放，支持团队无法手动将服务器添加到此实验中。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=9pHyH4XDAYk
  

---



**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1239612719318044846)** (1 messages): 

<ul>
  <li><strong>项目停用与所有权转移：</strong>一位用户宣布他们不打算继续 <strong>Fasteval project</strong> 或任何后续工作。如果有负责任的人感兴趣，他们愿意在 GitHub 上转让项目所有权，否则频道将被归档。</li>
</ul>
  

---



**AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/)** (1 messages): 

angry.penguin: nice, AK is back
  

---



---



---



---