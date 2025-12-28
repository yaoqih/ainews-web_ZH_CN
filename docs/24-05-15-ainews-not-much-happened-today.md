---
companies:
- openai
- google-deepmind
- anthropic
- rekailabs
- alibaba
- salesforce
date: '2024-05-15T21:20:08.374758Z'
description: '**伊利亚·苏茨克维（Ilya Sutskever）**在 **OpenAI** 任职近十年后卸任首席科学家一职，**雅库布·帕乔基（Jakub
  Pachocki）**被任命为其继任者。**Google DeepMind** 发布了 **Gemini 1.5 Pro** 和 **Gemini 1.5 Flash**
  模型，其特点是拥有 200 万 token 的上下文窗口和增强的多模态能力；同时还展示了 **Project Astra** AI 助手、**Imagen 3**
  文本生成图像模型以及 **Veo** 生成式视频模型。**GPT-4o** 在 VHELM 排行榜上名列前茅，并在 LMSYS Chatbot Arena 中表现优于竞争对手。拥有
  128K 上下文窗口的 **Reka Core** 多模态模型以及阿里巴巴的 **Qwen1.5-110B** 开源模型正式发布。**Salesforce**
  分享了一种在线 RLHF（基于人类反馈的强化学习）方案。'
id: 9d073ccd-a2aa-4c56-95e2-6966c0386805
models:
- gpt-4o
- gemini-1.5-pro
- gemini-1.5-flash
- imagen-3
- veo
- reka-core
- qwen-1.5-110b
original_slug: ainews-to-be-named-3669
people:
- ilya-sutskever
- jakub-pachocki
- mike-krieger
- sama
title: 今天没什么事。
topics:
- multimodality
- long-context
- model-releases
- reinforcement-learning
- model-benchmarking
- text-to-image
- video-generation
- ai-assistants
---

<!-- buttondown-editor-mode: plaintext -->*GPT-4o 和 Gemini 的余波。*

> 2024/5/14-2024/5/15 的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**427** 个频道，**6455** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**686 分钟**。

```
'Twas the night after I/O, when all through AI
Not a startup was posting, not even on LI
The UBI research was studied by e/accs with care
In hopes that AGI soon would be there
```

你可以祝愿 [Ilya](https://twitter.com/ilyasut/status/1790517455628198322)、[Jan](https://news.ycombinator.com/item?id=40363273) 和 [Evan](https://x.com/E0M/status/1790814866695143696) 一切顺利（[离职时间线](https://twitter.com/liron/status/1790773952811545051)是否有蹊跷？），阅读关于 GPT-4o [令人惊叹的多重“大海捞针”（multi-Needlestack）性能](https://news.ycombinator.com/item?id=40348947)的文章，或者如果你是 OpenAI 阵营的，可以观看 [John Schulman](https://www.youtube.com/watch?v=Wo95ob_s_NI) 或 [Sama](https://www.youtube.com/watch?v=fMtbrKhXMWc) 的最新访谈；你也可以[祝贺 Mike Krieger 加入 Anthropic](https://twitter.com/i/trending/1790766332885299320)，或者阅读在我们之后发布的[所有 Google I/O 综述](https://twitter.com/i/trending/1790833517636764082)（看来我们最初低估了 [PaliGemma](https://news.ycombinator.com/item?id=40371237)）。

---

**目录**

[TOC] 

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Ilya Sutskever 离开 OpenAI**

- **Ilya Sutskever 在任职近十年后卸任首席科学家**：[@sama](https://twitter.com/sama/status/1790518031640347056) 称赞 Ilya 是“我们这一代最伟大的思想家之一”，是 OpenAI 成功的基石。[@ilyasut](https://twitter.com/ilyasut/status/1790517455628198322) 表示，共同工作是一种“荣幸和特权”，在他追求一个对他个人有意义的项目之际，他会想念大家。
- **Jakub Pachocki 被任命为新任首席科学家**：[@sama](https://twitter.com/sama/status/1790518031640347056) 表示相信 Jakub 作为另一位“我们这一代最伟大的思想家之一”，将在新岗位上领导 OpenAI 在实现 AGI 的道路上取得快速且安全的进展。
- **Ilya 在塑造 OpenAI 使命和战略方面的关键早期作用**：[@gdb](https://twitter.com/gdb/status/1790519014562898012) 回忆起他和 Ilya 在早期非营利阶段度过的无数时光，他们在文化、技术方向和构建 OpenAI 的战略上达成一致，即使当时其他人怀疑 AGI 是否能在短期内实现。

**Google I/O AI 发布会**

- **Gemini 1.5 Pro 和 Flash 语言模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790432978126139411) 推出了具有 200 万 token 上下文窗口、并改进了代码、推理和多模态能力的 Gemini 1.5 Pro，以及针对低延迟和成本优化的 Gemini 1.5 Flash。两者现已在 Google AI Studio 和 Vertex AI 中可用。
- **Project Astra AI 助手原型演示**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790433540548558853) 分享了 Project Astra 的视频，这是一个未来的 AI 助手，可以与世界互动、记住上下文并协助日常生活。许多人将其能力与 GPT-4 进行了比较。
- **Imagen 3 文本生成图像模型发布**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790434750592643331) 介绍了 Imagen 3，这是他们迄今为止最先进的文本生成图像模型，具有更强的细节和现实感。
- **Veo 生成式视频模型预告**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435824598716704) 展示了 Veo，它能够生成各种风格的 1080p、60 秒以上的视频片段。目前已通过 Labs 候补名单开放。
- **面向创作者的 Music AI Sandbox 工具**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435413682975043) 与音乐人合作开发了一系列 AI 工具，旨在改变音乐创作过程，并通过新的演示录音进行了展示。

**AI 模型发布与基准测试**

- **GPT-4o 登顶排行榜**：[@percyliang](https://twitter.com/percyliang/status/1790622792347701432) 指出 **GPT-4o 登顶 VHELM 排行榜**。[@maximelabonne](https://twitter.com/maximelabonne/status/1790519226677026831) 分享了根据 [@LiamFedus](https://twitter.com/LiamFedus) 的数据，GPT-4o **在 LMSYS Chatbot Arena 上显著超越竞争对手**。
- **Reka Core 和 Qwen 模型**：[@maximelabonne](https://twitter.com/maximelabonne/status/1790519226677026831) 提到 @RekaAILabs 发布了一个强大的多模态模型 **具有 128K context 的 Reka Core**，而 @Alibaba_Qwen 发布了开源的 **Qwen1.5-110B** 和闭源的 **Qwen Max**。
- **Salesforce 在线 RLHF 方案**：[@_philschmid](https://twitter.com/_philschmid/status/1790747448807215428) 分享了 Salesforce 的 **在线迭代 RLHF 可复现方案**，表明迭代 DPO 等在线方法优于离线方法。代码、模型、数据集和训练细节均已开源。

**多模态 AI 和视频模型**

- **Imagen 3 和 Veo**：据 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790434750592643331) 称，Google 推出了 **Imagen 3**，这是他们迄今为止质量最高的文本生成图像模型，具有惊人的细节和逼真的光影效果。他们还展示了 **Veo**，这是一个强大的视频模型，可以创建各种风格的 1080p 60秒以上剪辑，目前可在 VideoFX Labs 申请候补名单体验，消息来自 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435824598716704)。
- **Music AI Sandbox**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435413682975043) 与 @YouTube 合作开发了 **Music AI Sandbox**，这是一套旨在改变音乐创作的 AI 工具套件，与音乐家和制作人紧密合作。新的演示录音已在 YouTube 上发布。

**梗与幽默**

- **Scarlett Johansson AI**：[@karpathy](https://twitter.com/karpathy/status/1790373216537502106) 开玩笑说 LLM 的杀手级应用是 Scarlett Johansson，而不是数学之类的。人们原以为是数学，结果是 ScarJo。
- **Gemini Flash 命名**：[@agihippo](https://twitter.com/agihippo/status/1790435129577599188) 提到作为前 Google 员工，他们仍在为 Google 的命名做贡献，指的是 Gemini Flash 这个名字。
- **Gemini 观看 I/O**：[@zacharynado](https://twitter.com/zacharynado/status/1790474150345081123) 调侃 Gemini 观看了 Google I/O，这是指 Project Astra 演示中 AI 观看主题演讲的情节。

---

# AI Reddit 综述

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**OpenAI 领导层变动与内部动态**

- **OpenAI Superalignment 团队共同负责人 Jan Leike 辞职，并推文表达对“幕后发生的一切”的担忧**：在 /r/singularity 中，有人分享了 [Jan Leike 推文的截图](https://i.redd.it/ztfqaypt0j0d1.png)，宣布他从 OpenAI 辞职，并对内部动态表示不安。这暗示了 OpenAI 内部在公司发展方向和 AI 安全方法上可能存在分歧。

- **OpenAI 联合创始人兼首席科学家 Ilya Sutskever 宣布在工作近十年后离职**：OpenAI CEO Sam Altman [发推](https://twitter.com/sama/status/1790518031640347056?t=0fsBJjGOiJzFcDK1_oqdPQ&s=19) 确认 Sutskever 即将离职，这引发了 /r/singularity 对领导层在 OpenAI 发展轨迹和 AI 安全优先级上存在不一致的猜测。

- **前 OpenAI 员工 Logan Kilpatrick 在回应 Leike 辞职时暗示了更多内部戏码**：Kilpatrick [回复](https://twitter.com/OfficialLoganK/status/1790604996641472987) Leike 的推文说“继续打好这场仗 🫡”，暗示 OpenAI 幕后可能存在更多即将浮出水面的紧张局势。

**GPT-4o 的能力与局限性**

- **GPT-4o 作为 OpenAI 的新旗舰模型推出，比 GPT-4 有显著效率提升**：在 /r/singularity 中，分享了一份 [OpenAI 电子邮件公告](https://www.reddit.com/r/singularity/comments/1crv0ri/new_openai_email/)，详细介绍了 GPT-4o 的能力——声称在性能匹配 GPT-4 的同时，价格降低 50%，latency 提升 2 倍，rate limits 提高 5 倍。

- **一些用户发现 GPT-4o 在基础推理测试中仍然失败，且在编程方面表现不如替代方案**：/r/OpenAI 的帖子显示 GPT-4o [在处理初级逻辑谜题时感到吃力](https://www.reddit.com/r/OpenAI/comments/1cs11b1/chat_gpt4o_still_fails_my_very_basic_intelligence/)，并且与 Anthropic 的 Claude 模型等竞争对手相比，表现出[令人失望的代码生成能力](https://www.reddit.com/r/OpenAI/comments/1cs210q/gpt4o_disappointing_performance_for_programming/)。

- **GPT-4o 的图像生成能力在发布会中被“奇怪地低估了”**：/r/singularity 的一个[帖子](https://www.reddit.com/r/singularity/comments/1crto0m/gpt4o_was_bizarrely_underpresented/)认为，GPT-4o 的视觉技能（包括生成独立对象和用于 3D 重建的图像）值得更多的强调和演示。

**Google I/O AI 发布**

- **Google 在 I/O 大会上宣布了多项新的 AI 计划，但与 GPT-4o 相比，反响褒贬不一**：一些 /r/singularity 的评论者认为，与 GPT-4o 的发布相比，Google 的 AI 演示和 Demo [显得平淡无奇](https://www.reddit.com/r/singularity/comments/1cs22sj/the_contrast_in_openai_versus_googles_approach/)。

- **Google 的新 AI 产品包括 Gemini 1.5 Flash、Imagen 3 和 Project Astra**：[Gemini 1.5 Flash](https://www.reddit.com/r/singularity/comments/1cs2v7h/gemini_15_flash_is_very_price_effective_relative/) 是一款高效的语言模型，[Imagen 3](https://deepmind.google/technologies/imagen-3/) 提升了图像生成能力，而 [Project Astra](https://www.youtube.com/watch?v=nXVvvRhiGjI) 则专注于 AI 助手。

**开源替代方案与担忧** 

- **一些人主张开源 AI 是 OpenAI 和 Google 闭源模型的重要替代方案**：一篇[评论文章](https://open.substack.com/pub/molbal94/p/opinion-the-case-for-open-source?utm_source=share&utm_medium=android&r=1wdxxq)认为，随着大型科技公司 AI 竞赛的加速，开源 AI 可以更好地优先考虑用户隐私、定制化和可访问性。

- **Meta 的 Llama-3 模型展现了潜力，但其限制性许可引发了对衍生作品的担忧**：/r/LocalLLaMA 的一个[帖子](https://www.reddit.com/r/LocalLLaMA/comments/1csctvt/we_need_to_have_a_serious_conversation_about_the/)呼吁讨论 Llama-3 许可对衍生模型（如 Salesforce 最近发布的模型）的法律影响。

**影响与社会冲击**

- **预测像 GPT-4o 这样能力日益增强的 AI 将颠覆教育、创意领域和编程岗位**：/r/singularity 的帖子推测，随着 AI 的快速进步，[学校教育](https://www.reddit.com/r/singularity/comments/1crqogx/with_the_recent_gpt4o_release_how_will_the_future/)、[娱乐](https://www.reddit.com/r/singularity/comments/1cs8r9q/im_excited_for_ai_generated_movies_made_by_great/)和软件工程将发生重大变化。

- **民意调查显示，大多数美国人支持监管以防止开发超智能 AI**：在 /r/singularity 分享的一项调查发现，63% 的人支持采取措施限制创建超越人类智能水平的 AI 系统。

**迷因与幽默**

- **迷因和笑话对 AI 进步的飞速发展做出反应**：/r/OpenAI 的帖子幽默地[预测人们会尝试与 ChatGPT 结婚](https://www.reddit.com/r/OpenAI/comments/1crxc89/i_bet_5_that_someone_will_really_try_marry_gpt/)，以及 [AI 让“受折磨的艺术家”](https://www.reddit.com/gallery/1cs2cwa) 能够应对令人不安的 AI 进步速度。

---

# AI Discord 摘要

> 摘要的摘要的摘要。我们的结论是 Claude 仍然是最好的摘要模型，因此我们放弃了 GPT-4 Turbo 和 4o 的对比。

1. **新 AI 模型与能力的揭晓**：
   - Google 在 Google I/O 上推出了多款新 AI 模型，包括用于高质量视频生成的 [**Veo**](https://deepmind.google/technologies/veo/)、用于提升文本生成图像能力的 [**Imagen 3**](https://deepmind.google/technologies/imagen-3/)，以及拥有 27B 参数的模型 [**Gemma 2**](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/)。[来源](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)
   - OpenAI 的 [**GPT-4o**](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/) 在发布前曾以秘密名称登顶 LMSYS 的 Chatbot Arena 排行榜。[来源](https://discord.com/channels/879548962464493619/879548962464493622/1239846551526965259)
   - Nous Research 发布了 [**Hermes 2 Θ**](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B)，这是一个融合了 Hermes 2 Pro 和 Llama-3 Instruct 的实验性模型，在保留 function calling 能力的同时，在基准测试中超越了之前的模型。[来源](https://discord.com/channels/1053877538025386074/1145143867818119272/1240351762863493150)

2. **多模态 AI 与统一模型的进展**：
   - 讨论集中在**多模态模型**的挑战与潜力，成员们探讨了像 [**ImageBind**](https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/) 这样的统一模型，该模型利用联合嵌入（joint embeddings）将跨多个模态的信息绑定在一起。[来源](https://discord.com/channels/1053877538025386074/1154120232051408927/1239836871887159306)
   - Google 推出了 [**Gemini 1.5 Flash**](https://openrouter.ai/models/google/gemini-flash-1.5) 和 [**Gemini 1.5 Pro**](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/)，为视觉理解、分类、摘要以及基于各种输入的内创创作提供了多模态能力。[来源](https://discord.com/channels/1091220969173028894/1092729520181739581/1239890486387281920)
   - 成员们讨论了将**多模态模型直接集成到智能手机和边缘设备**中的潜力，以实现低延迟和增强的多模态功能。[来源](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)

3. **LLM 的优化与效率提升**：
   - 讨论了 [**Gemini 的上下文缓存（context caching）**](https://ai.google.dev/gemini-api/docs/caching) 和 [**llama.cpp 的提示词缓存（prompt caching）**](https://github.com/ggerganov/llama.cpp/blob/e1b40ac3b94824d761b5e26ea1bc5692706029d9/examples/main/main.cpp#L225-L245) 等技术，这些技术通过减少长提示词的 Token 使用量，使 LLM 工作流更加高效且具成本效益。[来源](https://discord.com/channels/823971286308356157/1097032579812687943/1240332490154053725)
   - 成员们探索了提高 **L2 缓存命中率**以获得更好性能的策略，并参考了关于块级乘法和指针运算的 [**Triton 矩阵乘法教程**](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 等资源。[来源](https://discord.com/channels/1189498204333543425/1189607726595194971/1239868744906575992)
   - 讨论围绕在使用 `torch.compile` 时优化**张量分配（tensor allocations）和缓存**展开，建议使用预分配张量代替动态分配，并利用静态缓存来减少开销。[来源](https://discord.com/channels/1189498204333543425/1189607750876008468/1239840938080342057)

4. **关于 LLM 评估与行业动态的辩论**：
   - 一篇 [博客文章](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation) 强调了当前 LLM 评估实践的封闭性（由学术基准和私有 A/B 测试主导），呼吁在评估中实现更广泛的可访问性。[来源](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)
   - 成员们讨论了 Anthropic 向产品型公司的转型、OpenAI 通过关键招聘进军搜索领域的潜在尝试，以及 AI 公司提供终端用户产品而非仅仅是 API 或服务的战略需求。[来源](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)
   - Ilya Sutskever 从 OpenAI 离职引发了关于公司内部潜在人事重组的讨论，[Sam Altman](https://x.com/ilyasut/status/1790517455628198322) 等人对这一变动发表了评论。[来源](https://discord.com/channels/822583790773862470/1075282825051385876/1239868524688576552)

---

# 第 1 部分：Discord 高层级摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**GPT-4o 遭遇创作瓶颈**：与 **GPT-4** 相比，**GPT-4o** 更快的响应速度在创意写作任务中带来了一些权衡，它往往只是重复草稿，而不是通过智能修订来增强内容。这引起了一些试图利用 AI 进行写作增强的用户的不满。

**模型通过听觉描述走向音乐化**：社区成员创意性地使用提示词，要求 **GPT-4** 和 **GPT-4o** 描述器乐歌曲（如 *"The XX Intro"* 和 *"Tears in Rain"*），以衡量模型在听觉感知方面的描述能力。这些提示词的结果可能为每个模型的解释技巧提供见解。

**图像生成中的想象力挑战**：一位用户在请求平台游戏开发的特定侧视图时，在 **GPT-4** 和 **GPT-4o** 上都遇到了困难——AI 倾向于提供不想要的等轴测视图（isometric perspectives）和多余的细节，这表明在透视理解和上下文遵循方面存在差距。

**访问和功能的分阶段路径**：**GPT-4o** 功能的推出是循序渐进的，**Voice Mode** 和多模态（multimodal）等功能首先提供给 **API** 合作伙伴，然后是 **Plus** 用户。这种逐步部署导致了一些成员的困惑和访问问题。

**定制模型激发协作**：关于将自定义 **GPTs** 与 **GPT-4o** 集成的讨论，以及对 **OptimusGPT** 等自定义模型的反馈会议，凸显了社区对改进和协作的热情。用户已被建议在未来几周内将其自定义 **GPTs** 迁移到 **GPT-4o**，以获得更好的性能。

**AI 语音助手的简便性**：一位成员强调了 **Plug & Play AI Voice Assistant**，它可以快速设置，并邀请社区试用。尽管有重复的公告，但重点在于该助手的易用性以及用户对其效能反馈的价值。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**机器学习神话**：一位公会成员使用 **Unsloth** 创建和微调数据，开发了一个面向 **Cthulhu** 崇拜的 AI，从而创建了 **TinyLlama** 和 **Mistral 7B Cthulhu** 模型，资源可在 [Huggingface](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/) 上获得。

**航行在量化之海**：对话探讨了 **quantization**（量化）和 **model merging**（模型合并）中的挑战，成员们分享了诸如在合并前手动上采样到 16-bit 以及使用 notebook 促进转换过程等技巧，展示了优化 AI 模型以获得更好性能的复杂领域。

**全球模型推广**：**Unsloth** 在 [AI News](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord) 的专题报道中因其在开源 AI 开发方面的进步而获得认可，社区成员也团结起来支持一项提案，旨在即将举行的纽约市开源数据管道聚会上展示 **Unsloth**。

**GPT-4 救生员值班**：为一位在 **Trigonometry**（三角学）问题上挣扎的公会成员提供了帮助，证明了社区在提供 **ChatGPT** 和 **Claude** 等资源进行学术援助方面的快速响应。

**AI 摘要受到审查**：使用 AI 总结 **Discord** 互动的做法被指出可能与欧洲数据隐私法冲突，这标志着在平衡技术创新与法律合规方面需要持续保持警惕。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **转录困扰与极速 GPT-4o**：工程师们讨论了 Perplexity 应用转录长达一小时会议的能力，并称赞 GPT-4o 在速度上超越了 Turbo。然而，人们对目前录音功能的局限性表示担忧，特别是完整的转录功能尚未全面推出。

- **解析问题依然存在**：用户指出 Perplexity 在解析 URL 内容时存在不准确的问题，暗示它是在进行猜测而非分析实际的网页内容。这表明内容解析算法仍有提升空间。

- **LLaMA-3 技能集多样化**：*LLaMA-3-sonar-large-32k-chat* 模型针对对话细微差别进行了优化，而 *LLaMA-3-8b-instruct* 则旨在提供更全面的指令覆盖。此外，人们对 *LLaMA-3-sonar-large-32k-online* 模型的网页搜索能力表现出兴趣，该模型类似于 Perplexity.com 等平台上的 RAG 模型。

- **API 访问与延迟问题浮出水面**：申请引用 API 的测试权限以及观察到 Perplexity API 调用延迟增加的情况，反映了社区对 API 功能的积极开发和利用。目前正在积极考虑 API 超时效率，特别是针对较长的 3000 字输入，这些输入在 10000ms 的设置下目前会面临超时。

- **Perplexity 搜索发现宝库**：成员们分享了各种 [Perplexity.ai](https://www.perplexity.ai) 搜索案例，从市场规模的详细分析、对正念练习的深刻见解，到模型微调（finetuning）的全面资源。这些搜索证明了该平台在 AI 探索和模型调优方面拥有丰富的信息生态系统。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**LLM 性能的重大突破**：新推出的 **Hermes 2 Θ** 在基准测试中表现优于 Hermes 2 Pro 和 Llama-3 Instruct，在保持函数调用（function calling）能力的同时展现了卓越的性能，正如 **[公告](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B)** 中所宣布的那样。

**Discord 与创新的碰撞**：**[off-topic](https://autocompressor.net/av1?s=sznVX9AV)** 中讨论了一个利用 Discord 漏洞的工具，该工具允许嵌入大于 500MB 的 **AV1 视频**，这些视频也可以在 Twitter 等平台上分享。

**GPT-4 评价褒贬不一**：尽管 GPT-4 在数据科学任务中表现出色，但 **[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1239651072503709797)** 频道的讨论揭示了它在复杂任务中表现不佳以及容易丢失上下文的问题，暗示了在速度与准确性之间存在权衡。

**北欧 AI 语言模型发布**：**[interesting-links](https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages)** 展示了 Viking 7B，这是由 Silo AI 和图尔库大学（University of Turku）的 TurkuNLP 为北欧语言设计的领先多语言 LLM，增强了语言 AI 的可访问性。

**AI 怀疑论与热情交织**：**[general](https://github.com/NousResearch/Hermes-Function-Calling/tree/main)** 和 **[ask-about-llms](https://arxiv.org/abs/2305.05665)** 等各个频道的总体情绪依然复杂，既有对 Hermes 2 Θ 等新模型的热情，也对多模态能力以及从零开始构建 LLM 面临的障碍持怀疑态度。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**与 LLM 进行语音交互**：关于将语音交互与本地大语言模型（LLM）集成的讨论突出了 *AnythingLLM* 等工具的使用。社区讨论了涉及 Whisper.cpp 和 Coqui TTS 的资源密集型解决方案，尽管其复杂性较高且体验尚不理想。

**强化硬件装备**：关于 AI 模型硬件偏好的辩论在 3060Ti GPU 与双 16 核 Xeon V4 CPU 之间展开。爱好者们讨论了 VRAM 的关键作用，并倾向于选择 Nvidia 显卡以获得顶级的 AI 性能。提到 4060 也引发了对其预期收益的关注。

**PrivateGPT vs. AnythingLLM - 文档查询之战**：**PrivateGPT** 与 **AnythingLLM** 在使用 LLM 查询文档方面的竞争引发了技术分析。讨论强调了每个平台的设置复杂性和用户友好性。

**MacOS 优先引发不满**：一场关于 Mac 级别的辩论浮出水面，主要针对应用发布优先级的不满，特别是 OpenAI 的 MacOS 优先策略。这演变成了关于 MacOS 与 Windows 应用开发复杂性和差异的对话。

**模型竞技场中的巨头之战**：从不受限的本地 LLM 推荐（特别是 [Dolphin 2.8 Mistral 7B v0.2](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)），到量化和模型性能的细微差别，社区剖析了各种 AI 范式。上述内容还包括 Command R 模型的对比以及与 GPU 相关的谜团。

**攻克硬件前沿**：通常不被 AMD 的 ROCM 构建支持的 ROGUE RX6600 在 Koboldcpp 中运行良好，而官方的 llama.cpp 二进制文件由于 GPU ID 验证过程限制了其使用。用户指出了 LM Studio 设置中用户界面（UI）的复杂性。

**搜集 GPU 优化秘籍**：关于使用 Windows 任务管理器优化 GPU 资源的技巧层出不穷，例如禁用硬件加速以增强资源可见性等奇特建议。然而，在某些笔记本电脑上配置 CUDA 仍然存在困难，导致 LM Studio 中持续出现模型加载错误。

**GPU 争斗中的老将与新兵**：Tesla M40 在 LLM 任务中与 GeForce 1060 的对决表现令人失望，显存（VRAM）速度的重要性备受关注。财务限制困扰着用户，低端 PC 在适度的本地模型中找到了慰藉，而 APU 在 llama.cpp 中显示出相比 CPU 并没有性能优势。

**Beta 版本的忧伤**：在 Beta 测试领域，关于多模态功能对等性的思考与 LM Studio 因缺乏 AVX2 支持而导致的启动问题报告并存。一位用户对无法启动 LM Studio 的愤怒在确定 AGX 指令集对运行至关重要后得到了平息。

**开发者摘要**：Intel 提议使用 SYCL 为 llama.cpp 提供 Intel GPU 支持，这拓宽了 LM Studio 的前景。围绕深度学习（DL）模型适配、对 AGI 的追求以及社区要求将开发讨论集中在 LM Studio 的 API 和软件构建上的呼声日益高涨。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**GPT-4o 隐秘冠军**：OpenAI 的 [GPT-4o](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/) 已被确认为 LMSYS Chatbot Arena 中以神秘名称位居榜首的模型，展现了未公开的卓越性能。

**数据集与模型效能增强**：一个团队发布了一个包含 [700,000 个样本的越南语数据集](https://huggingface.co/datasets/Vi-VLM/Vista) 用于开源语言建模；同时 AutoTrain 扩展了其工具包，增加了 Object Detection 功能；[Diarizers](https://x.com/kamilakesbi/status/1790025563631132940) 作为一个新库出现，用于在 Hugging Face Hub 上微调支持多语言的 speaker diarization 系统。

**AI 驱动的故事创作**：一个阅读小组对 [AI story generation](https://arxiv.org/abs/2310.05388) 进行了全面回顾，讨论重点转向完善 [GROVE framework 论文](https://arxiv.org/abs/2310.05388)，社区成员通过 [Medium](https://isamu-website.medium.com/understanding-ai-for-stories-d0c1cd7b7bdc) 分享了相关尝试和心得。

**从视觉数据到营收洞察**：[#computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1239859086409863271) 频道的咨询引发了关于训练一个将图像转换为销售数据输出的模型的可行性讨论；发帖者提供了一个相关的 [数据集链接](https://huggingface.co/datasets/tonyassi/sales1) 供参考。

**使用 LangChain 增强聊天机器人**：在 [#NLP](https://discord.com/channels/879548962464493619/922424173916196955/1240103986791452758) 频道中，一位成员寻求使用 LangChain 改进聊天机器人对话的方法，建议指向了一个使用本地 LLM 和 embedding 模型的 [入门示例](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA 训练首战告捷**：一位工程师成功完成了他们的首次 LoRA 训练，仅耗时 90 分钟，并计划在 [Civitai](https://civit.ai/) 上分享结果。
- **使用 Powerpaint 聚焦细节**：用户间的技术讨论集中在利用 inpainting 和 Powerpaint 增强图像细节，特别是 1.5 版本能够改进眼睛等细节特征。
- **工作流专家**：对于那些对使用 ComfyUI 进行 outpainting 技术感兴趣的人，一位热心的工程师链接了一个 [GitHub 工作流指南](https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md)，帮助其他用户掌握 inpainting 和 outpainting。
- **Google Imagen 对阵大众之选**：Google Imagen 3 与 Stable Diffusion 的对比反映出社区更倾向于后者，理由是相比这家科技巨头的产品，后者具有更好的可访问性和易用性。
- **GPU 闲谈**：工程师们讨论了 AI 相关任务的 GPU 偏好，强调了 VRAM 对长期效用的重要性。共识建议等待定于 11 月发布的 50xx 系列 GPU，可能会获得更好的性价比。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 掀起模型发布狂潮**：OpenRouter 发布了多款新 AI 模型，如 [DeepSeek-v2 Chat](https://openrouter.ai/models/deepseek/deepseek-chat) 和 [Llama 3 70B Base](https://openrouter.ai/models/meta-llama/llama-3-70b)，以及来自 Google 的 [Gemini Flash 1.5](https://openrouter.ai/models/google/gemini-flash-1.5) 和 [Perplexity 的多款模型](https://docs.perplexity.ai/changelog/new-models-llama-3-sonar-family) 以扩充其阵容。尽管使用 DeepSeek 模型需要开启日志记录，但 OpenRouter 仍强调了这些创新。

- **性能需求导致 WizardLM-2 8x22B Nitro 停运**：OpenRouter 砍掉了 WizardLM-2 8x22B Nitro 变体，原因是供应商无法维持每秒 100 tokens 的预期吞吐量，这体现了其严格的性能标准。
  
- **加密货币确认延迟疑问获解答**：加密货币余额到账延迟归因于 Coinbase 等平台的网络确认要求，例如在 Polygon 上需要 128 个区块确认，其他网络也有类似标准。

- **掌握模型的 API 工具**：一位用户贡献了一个基于 API 的工具，用于跟踪 OpenRouter 模型更新，通过 [GitHub repository](https://github.com/fry69/orw) 提供每小时刷新的列表，标志着一种社区驱动的技术监控方式。
  
- **对 Google Gemini 大会的反应褒贬不一**：Google 的 Gemini 活动引起了不同的反响，虽然推出了 Gemini 1.5 Flash 等新模型，但在 OpenAI 备受瞩目的活动背景下，似乎并未给所有人留下深刻印象，展现了社区预期的差异。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 集成 MLIR**：工程师们讨论了 **Mojo** 通过微小的语法调整执行 **MLIR** 代码的能力，这增强了 Mojo 的通用性并使其能够访问更底层的特性。

**掌握 Mojo 的策略**：社区推荐了多种学习 **Mojo** 的资源，包括 [Mojo SDK manual](https://docs.modular.com/mojo/manual/get-started/) 和 [Mandelbrot notebook](https://docs.modular.com/mojo/notebooks/Mandelbrot)，并强调了该语言在跨厂商 GPU 代码移植性方面的优势。

**摆脱 Python 依赖的便利性**：社区正在探索在 **Mojo** 工具链中替代 Python 依赖项的方案，表明其正致力于构建一个更具语言无关性的生态系统。可通过 [GitHub 上的功能请求](https://github.com/modularml/mojo/issues/935) 关注进展。

**Mojo 与 C/C++ 及 Python 的互操作性引发热议**：关于使用 ffi 调用 C/C++ 库以及处理 Python 互操作性问题的讨论非常活跃，反映了人们对 **Mojo 跨语言能力** 的浓厚兴趣。工程师们正在分享相关机制的见解，共享的 [tweetorial](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi) 和问题解决线程就是证明。

**Modular 的多媒体 Mojo 动态**：Modular 通过关于 Mojo nightly 构建和 MAX Graph API 的新视频，以及 [MAX Graph API 博客教程](https://www.modular.com/blog/max-graph-api-tutorial) 提供了更新和教程。此外，还发布了两条预告更新和社区会议的推文，但具体细节尚未明确。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Mimetic Initialization 展现前景**：根据一份分享的 [论文](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf)，将 **mimetic initialization**（模拟初始化）引入 Transformers 在 CIFAR-10 和 ImageNet 等数据集上取得了显著的准确率提升。该技术模仿了预训练模型的权重模式，预示着更高效训练的潜力。

**利用 Sakuga-42M 实现数据集多样化**：新的 **Sakuga-42M 数据集** 发布，包含 4200 万个卡通动画关键帧，旨在减少在自然图像上训练的模型的偏差。该数据集的 [arXiv 链接](https://arxiv.org/abs/2405.07425) 为进一步探索提供了入口。

**Hypernetworks 引发初始化研究兴趣**：围绕使用 Hypernetworks 进行权重初始化的讨论不断涌现，暗示了利用符号回归（symbolic regression）来构建创新初始化技术的可能性。

**在神经网络中利用点积**：一场热烈的讨论认可了点积在神经网络中的有效性，一位成员链接了一篇 [文章](https://archive.is/GfliU)，探讨了点积与傅里叶变换的联系及其对认知处理的影响。

**增强多选题分析**：关于优化模型处理多选题方式的辩论十分激烈，重点介绍了 **lm-evaluation-harness** 处理每个答案请求的方法，并考虑增加输出导出功能以进行准确率分析，参考了 [GitHub 代码](https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485)。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **LLVM 中的 CUDA 奇特性与集成**：参与者们正在解决 CUDA streams 中的同步异常问题，怀疑这会影响梯度累积和 GPU 活动。观察到 *Stream misordering*（流排序错误）可能会引入竞态条件。建议推动更显式的流处理并重新思考梯度逻辑（[PR #417](https://github.com/karpathy/llm.c/pull/417)）。梯度检查中的容差水平也引发了激烈辩论，支持者主张采用相对于量级的实用阈值。

- **Triton 教程吸引矩阵乘法关注者**：一个 *Triton 教程* 受到关注，它深入探讨了通过块级矩阵乘法和优化的指针运算来提高 L2 cache 命中率的方法（[教程链接](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)）。这与阐明 CUDA 问题的讨论相吻合，例如由于 FP32 精度限制导致的 *naive dot product implementation*（朴素点积实现）错误（[点积难题 Issue](https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8)）。

- **PyTorch torch.compile 特性揭秘**：用户发现了 torch.compile 在处理动态张量分配时的特性及其性能开销，并与静态分配进行了对比。建议甚至包括在编译期间使用 `torch._dynamo` 装饰器进行调试。DeepSpeed 最近的发布引发了关于其与 `torch.compile` 兼容性的疑问，引起了对一个 [建议增加编译标志的 GitHub PR](https://github.com/microsoft/DeepSpeed/pull/4878) 的关注。

- **讲座与指南指引方向**：一位寻求 CUDA kernels 指导的新手被引导至一个对 Python 程序员非常有用的 [CUDA YouTube 讲座](https://youtu.be/4sgKnKbR-WE)；另一方面，一位成员为 GPU 编程倡导者提供了 [NVIDIA GPU Programming Guide](https://download.nvidia.com/developer/GPU_Programming_Guide/GPU_Programming_Guide.pdf)。

- **云端推文趣闻**：在技术讨论之余，一位成员幽默地推荐查看 cloud 的 Twitter 而未给出背景（[推文链接](https://twitter.com/cloud11665/status/1790776040681271583)），展示了社区偶尔轻松互动的一面。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 Mozilla 的 llamefile 运行你自己的 LLM**：Mozilla 新推出的 **llamefile** 让工程师能够轻松建立本地、私密的科研助手。只需简单的下载和执行，即可直接使用来自 LlamaIndex 的本地 LLM 和 embedding 模型，增强了数据隐私和控制力。[点击此处了解更多](https://t.co/qFIA6j1OWe)。

- **Navarasa 的认可与 LlamaIndex 的新合作伙伴关系**：**Navarasa** 是一款支持 15 种印度语言的模型，在 Google I/O 上备受关注。此外，LlamaIndex 与 **Vertex AI** 合作推出的 RAG API 标志着简化复杂 AI 系统集成的趋势。[Google I/O 上的 Navarasa](https://t.co/zc00GjOmc4) | [Vertex AI 上的 LlamaIndex](https://t.co/ekAQ24hNWr)。

- **使用 GPT-4o 轻松创建聊天机器人**：**create-llama** 的推出让技术背景较弱的用户也能通过精简的问答设置流程，利用 **GPT-4o** 构建聊天机器人。这是迈向 AI 驱动的对话式 Agent 普及化的重要一步。[探索如何实现](https://t.co/wtcaWdrB7H)。

- **各类技术辩论与澄清**：社区成员围绕“从细粒度到粗粒度检索（small to big retrieval）”的效率、**sec-insights repo** 的更新流程、模型性能差异（特别是 **Meta-Llama** 与量化后的 **Ollama** 之间），以及 **GPT-4o** 与 **LlamaIndex** 的集成展开了技术讨论。

- **LlamaParse 的安全协议受到质疑**：对 **LlamaParse** 安全性的担忧促使官方对数据保留政策进行了澄清，例如 48 小时缓存政策，并为优先考虑严格数据安全措施的用户提供了 on-premise（本地部署）选项。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Nathan Lambert 引发 AI 讨论**：Nathan Lambert 在一条[推文](https://twitter.com/natolambert/status/1790393805633712246)中批评了 OpenAI 以用户为中心的方法，并认为 Google I/O 上展示的生成式视频进展令人印象深刻，但也指出 Gemini 1.5 Ultra 等一些发布被忽视了。

**Google 发布 Gemma 2**：据 [TechCrunch](https://techcrunch.com/2024/05/14/google-i-o-2024-everything-announced-so-far/) 报道，Google 在 Google I/O 上宣布了拥有 270 亿参数的模型 [Gemma 2](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/)，并对其 AI 套件进行了更新，包括 Gemini 1.5 Pro 和 Flash。

**Tokenizer 调整困扰工程师**：讨论集中在 **OpenAI** 是使用新的 tokenizer 重新预训练，还是为 LLM 扩展当前的 tokenizer，同时分享了 [arXiv 论文](https://arxiv.org/abs/2405.07883v1)中讨论的 **Zero-Shot Tokenizer Transfer (ZeTT)** 这一新颖概念。

**观察到神经网络的收敛现象**：一项新兴研究表明，跨模态的神经网络正趋向于一个共同的现实统计模型，正如一篇[论文](https://phillipi.github.io/prh/)所提出的，并得到了 [Phillip Isola 的提及](https://x.com/phillip_isola/status/1790488967827108304?s=46)支持。

**AI 评估与行业转型受到关注**：一篇分享的[博客文章](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation)强调了当前 LLM 评估实践的封闭性，讨论还涉及 Anthropic 向产品公司转型的趋势、OpenAI 的重要招聘暗示其可能进军搜索领域，以及 AI 公司提供产品的战略必要性（参考[推文和文章](https://x.com/theinformation/status/1790467870545027186?s=46)）。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**AI 真的在等待**：用户对 **LangChain agents** 缓慢的响应时间表示沮丧，处理大型输入和调用工具需要 2-3 分钟，他们正在寻求快速解决建议。活跃的讨论围绕使用 `python-socketio` 来流式传输 LLM 响应展开，参与者们交换了 [代码片段和故障排除建议](https://github.com/langchain-ai/langchain/issues/4118)。

**服务器，快醒醒！**：对于托管版 Langserve 的用户来说，服务器不活动和速率限制错误等间歇性问题导致服务可用性难以预测。用户提出了关于升级到 Pro 计划是否能缓解这些麻烦以及如何访问更详细日志的问题。

**AI 优化下的 Snowflake 成本关注**：演示了一个创新的 **Snowflake 成本监控工具**，该工具集成了 LangChain 的功能与 Snowflake 和 OpenAI，旨在简化数据可视化和分析。这个正在开发中的工具功能在 [Loom 视频演示](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064) 中进行了展示。

**AI 变现，Java 风格**：一位 Langserve 用户正在尝试使用 `py4j` 库，通过 JVM 促进 AI 交互的小额支付功能，目标是集成加密货币 SDK。该设置旨在通过跟踪 prompt/response token 计数并在 OpenAI API 密钥对使用中增加利润空间，来创新小额支付结构。

**数据库难题与 Embedding 效率**：讨论线程涉及在 pgvector 和 Qdrant 等向量数据库之间进行 embedding 迁移。成员们分享了并行传输和优化检索速度的策略，并引用了 [Supabase 关于 Matryoshka Embeddings 的博客](https://supabase.com/blog/matryoshka-embeddings) 等参考资料。此外，在 API 调整停滞的背景下，用户寻求关于弃用 `LLMChain` 转而支持 `MultiQueryRetriever` 的 `RunnableSequence` 的澄清。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **腾讯 HunyuanDiT 反应平平**：工程师们探讨了腾讯 [HunyuanDiT 模型](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 的优缺点，指出其在中文 prompt 遵循方面表现强劲，但在处理直线方面存在挑战，显示出它可能仍未超越现有的 stable cascade 模型。

- **AniTalker 连接音频与动画**：[AniTalker](https://x-lance.github.io/AniTalker/) 因其能够使用音频输入使静态肖像动起来的能力而受到关注，即使在给定类似的控制信号时，也能提供一种创建逼真说话视频的方法。

- **DeepMind 发布 Imagen 3 和 Veo**：[Google DeepMind 的 Imagen 3](https://deepmind.google/technologies/imagen-3) 因在文本生成图像的细节和真实光影方面设定了新基准而获得认可；同时，DeepMind 的 Veo 被介绍为一个强大的工具，能够根据文本 prompt 生成详细的 1080p 视频，目前正通过 [VideoFX](https://labs.google/videofx) 等待早期访问。

- **depyf 简化深度学习性能调优**：PyTorch 宣布了一个名为 [depyf](https://pytorch.org/blog/introducing-depyf) 的新工具，旨在解码 `torch.compile` 的复杂性以进行性能优化；这是一个积极的进展，同时也凸显了对改进错误消息传递的需求。

- **AI 对能源和 GPU 算力的渴求**：对话从可持续性和效率的角度转向了 AI 巨大的能源消耗和对 GPU 的依赖，例如，有人指出 8x H100 GPU 配置的高待机功耗。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **MacOS 用户尝试本地 AI**：工程师们正在 macOS 本地部署 **ollama/dolphin-mixtral:8x7b-v2.6** 等模型，主要是为了规避高昂的成本。集成本地模型的技巧包括使用 **OpenRouter** 和 **Groq**，并针对 **llama3** 和 **Mixtral** 等模型提供了特定命令。

- **在运行 OpenInterpreter 方面，Ubuntu 优于 Windows**：一场热烈的讨论认为在运行 OpenInterpreter 时 **Ubuntu** 比 **Windows** 更好，特别是在 GPU 兼容性方面。建议很明确：使用 Ubuntu 的命令，而不是 macOS 的，一位直言不讳的用户强调道：*“记住你是在 UBUNTU 上运行！！！请使用 Ubuntu 命令。”*

- **指示灯闪烁与调试乐趣**：当一些人在为 **Light 设备预订** 的发货更新而苦恼时，另一些人发现了如何激活 **01 terminal** 的调试模式。调试 01 的 interpreter 的关键？在 i.py 脚本中设置 `"interpreter.debug = True"`，以获得更高的系统运行可见性。

- **开源 AI 拥护者的选择**：开源 AI 的支持者宣扬其相对于潜在 Apple OS 集成的价值，转而选择 Linux 的开放性。与此同时，固件方面的困扰得到了重新刷机的建议，并伴有关于 **OpenRouter** 的 Groq 兼容性文档不准确的警告。

- **AI 领域中创意胜过控制**：分享的 [播客见解](https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e) 强调了在 AI 创业中，创意、质量以及关注客户而非控制客户对成功的重要性。历史上的失败被引用为教训，证明控制通常会导致垮台，而对 Linus Torvalds 的致敬则凸显了趣味和开放协作如何促进创新。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Google 在 LLM 可靠性上失策**：讨论批评了 **Google I/O 主旨演讲** 掩盖了 LLM 的缺陷，并将 Google 的做法与 OpenAI 更为谨慎的态度进行了对比，后者公开承认了 LLM 输出存在错误的可能。
  
- **Meta 低调行事**：工程师们对 **Meta** 低调但高效的 AI 产品表示赞赏，特别提到了其 Wayfarer 眼镜的多样化功能。

- **将 AI 落地于日常工作**：展示 AI 实际落地的概念（被称为 "Sober AI"）得到了用户的认可，重点关注实用主义的 AI 工具，而非那些被过度炒作的用途。

- **新闻业加入 AI 浪潮**：正在讨论 AI 的实用化应用，如 **MuckRock** 的 AI 用于自动化 FOIA 任务，同时对 Zach Seward 在 SXSW 上关于 AI 在新闻业中角色的精彩演讲表示赞同。

- **让 LLM 更省钱**：对话转向通过 **Gemini 的 context caching** 和 **llama.cpp 的 prompt caching** 等策略来提高 AI 的成本效益，旨在减少与长 prompt 相关的 Token 消耗。

- **上下文一致性问题**：一名成员提出了在对话过程中切换不同模型时保持上下文的问题，强调了提取和传输 JSON 日志以实现无缝过渡的重要性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **使用 ChatGPT 解锁设备控制**：一项提案建议将设备控制提升为 ChatGPT 集成中的主要模态，建议从基于文本的命令转向直接操作，因为观察到像 PyAutoGUI 这样的中间工具有局限性。

- **大脑研究：存储奥德赛**：哈佛大学和 Google AI 的研究人员面临着巨大的存储需求。根据 [Google 的研究博客](https://blog.google/technology/research/google-ai-research-new-images-human-brain/)，仅 1 立方毫米的大脑组织成像就需要 1.4 petabytes 的存储空间。

- **Google AI 最新模型评价褒贬不一**：Google 发布的 AI 模型 **Veo** 和 **Project Astra** 在性能评价上褒贬不一。根据 [Google DeepMind](https://x.com/GoogleDeepMind/status/1790435824598716704) 和 [其他来源](https://x.com/0xgaut/status/1790428601789067614) 的讨论和推文，它们与 GPT-4o 现场演示的对比评价各异。

- **寻找更好的 AI 替代方案**：对 Perplexity AI 的不可靠性及其 "Pro" 账户限制感到沮丧，促使了对替代资源的讨论，例如用于编程查询的 **Phind.com** 和用于高效搜索的 **Kagi**。

- **OpenAI 的重大离职**：**Ilya Sutskever** 从 OpenAI 离职在 AI 社区引起了复杂反应，从 [Sam Altman](https://x.com/ilyasut/status/1790517455628198322) 等人的推文中可见一斑，这暗示了组织高层的重组。

- **与 Eugene 一起准备 Evals**：一场关于 Evals 的活动已排期，由 Eugene 主持讨论，准备材料和讨论见 [此处](https://eugeneyan.com/writing/evals/)。建议参与者订阅 iCal 以获取活动更新。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **多语言建模的规模挑战**：工程师们讨论了训练 cmdR+ 100b 模型的困难，原因是 **VRAM 需求极高**，并强调了该模型作为顶级多语言选项的独特性。一些人建议利用 **FSDP (Fully Sharded Data Parallelism)** 来管理多个 GPU 上的权重分布。

- **Llama3 的数据获取**：用户在 **Llama3** 上的成功取决于添加更多数据，这引发了社区对实现这些结果所使用的具体配置设置的兴趣。

- **TinyLlama 的路径查找问题**：解决 **TinyLlama** 的 `No such file or directory` 错误需要手动干预，解决方案包括删除目录并在 **RunPod** 上执行特定命令。

- **Falcon 11b 与 LLaMA 3 的对峙**：社区对 **Falcon 11b** 和 **LLaMA 3** 进行了对比，考虑了许可等方面；Falcon 的许可证包含潜在的 **不可执行条款**，导致尽管 Falcon 是开放的（虽然有问题），人们仍更倾向于 LLaMA 3。

- **快速 LoRA 训练咨询**：一位成员请求快速 YAML 配置微调的技巧，比起结果质量更看重速度，社区建议强调了 **禁用梯度检查点 (gradient checkpointing)** 与运行时间改进之间的权衡。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R 的 RAG 大获成功**：一位用户称赞 **Command R 的 RAG** 在处理长源码时的准确性和忠实度，称其不仅具有成本效益，而且在检索增强生成任务中表现出色。

- **Preamble 是 System Message 的一部分**：参与者区分了 *'Preamble'* 和 *'System Message'*，解释说 Preamble 包含在系统消息中，并由 `<|SYSTEM_TOKEN|>` 和 `<|END_OF_TURN_TOKEN|>` 等 Token 标记，以改进模型的对话处理。

- **Cohere 模型的特殊 Token 说明**：解释了在 Cohere 的语言模型中如何使用特殊 Token 来界定系统消息的开始和结束，这对于对话式 AI 中正确的响应生成至关重要。

- **使用 Reranker 模型探索 Token 相关性**：一位用户询问了 **Cohere reranker 模型** 在突出显示相关 Token 方面的能力，并将其与 ColBERT 的功能进行了比较，后者可以指示单词的重要性以促进更好的用户交互。

- **RAG 解析与合作邀请**：一篇 [Medium 文章](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2) 解释了如何使用 **@UnstructuredIO API** 从头开始学习 **RAG**，而另一份合作邀请则表明了对类似项目工作的共同兴趣。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **将 Tinygrad 移植到 Urbit**：一位用户启动了将 **tinygrad** 移植到 **Urbit/Nock** 的项目，正在处理 `forward()` 函数并展示了 [项目仓库](https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon)。他们指出需要一个转换层来桥接 tinygrad 风格的 Python 与 Urbit 系统。

- **优质初学者议题警报**：对于 tinygrad 社区的新手，**George Hotz** 强调了一个对初学者友好的 GitHub issue：[BEAM kernel count number is wrong](https://github.com/tinygrad/tinygrad/issues/4595)，鼓励大家贡献代码。

- **在尖端硬件上排查 CUDA 故障**：在 GeForce 4090 上处理 PTX=1 的 **CUDA 错误** 时需要更新驱动程序；虽然 Titan V 没有出现类似问题，但仍强调了安装最新驱动的必要性。

- **Shape-Stride Visualizer 简化张量重塑**：引入了一个创新的可视化工具 [Shape-Stride Visualizer](https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx)，帮助用户更直观地理解 tinygrad 中复杂的重塑（reshaping）操作。

- **TACO 助力张量理解**：讨论了 **Tensor Algebra Compiler (TACO)** 及其丰富的张量格式可视化功能，支持深入研究张量操作，并重点介绍了其 [在线文档](http://tensor-compiler.org/codegen.html) 以供进一步探索。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **AI Town 现已在 Hugging Face Spaces 运行**：[AI Town 已在 Hugging Face Spaces 上线](https://huggingface.co/spaces/radames/ai-town)，提供了一个在 CPU 上运行的模拟环境，这对容器化 AI 应用具有前景。
- **通过优化交互增强 AI Town**：为了提升 **AI Town 的性能**，工程师建议减少非玩家角色（NPC）的数量并调整交互“冷却时间”的计时器，旨在更有效地管理 NPC 活动和对话频率。
- **对 AI Town 自定义 Agent 控制的兴趣**：成员们正在评估 **AI Town** 如何通过 API 允许 Agent 控制，目前单个语言模型 Agent 尚不支持此功能；讨论暗示未来可能会涉及 LLamaFarm 的功能。
- **深入探讨 AI Town API 能力**：AI 工程师们集思广益，探讨了 **AI Town** 与 API 集成的潜力，考虑使用 API 获取补全（completions）、嵌入（embeddings）以及处理语义交互，并提到将包含用于状态监控的 webhook 支持。
- **Tommy1901 预告 Raspberry Pi 项目**：虽然细节较少，但 tommy1901 表示打算在未来分享与 **Raspberry Pi** 相关的“酷玩意”，引发了大家对 **#[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** 频道中即将推出的项目或黑客技巧的好奇。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Token 的烦恼与胜利**：工程师们就 [德语的 vocab_size 与 tokens/byte](https://discord.com/channels/1178995845727785010/1182877486854451271/1239880142181105664) 数据缺失问题展开争论，强调了偏向混合语言的 tokenizer 数据集存在的空白。

**非贪婪 Tokenizer 问世**：分词领域的新工具 [TokenMonster 项目](https://github.com/alasdairforsythe/tokenmonster)（一个“非贪婪子词分词器和词汇训练器”）因其在 Python、Go 和 Javascript 中的实用性而备受关注。

**调情风格的 AI 演示走红**：GPT-4o 最近的演示因其带有暗示性的幽默而引发了一阵笑声和关注，正如一位用户在幽默的 [推文](https://fxtwitter.com/main_horse/status/1790099796193398831) 中所指出的那样。

**词表冲击波席卷 Twitter**：GPT-4o 的“o200k_base”词表在技术社区引起了惊讶甚至可能是沮丧的反应，正如这篇 [推文](https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09) 所传达的那样。

**Ilya 从 OpenAI 离职**：Ilya Sutskever 离开 OpenAI，引发了关于这家研究巨头未来的讨论，该离职消息通过 [Twitter](https://twitter.com/ilyasut/status/1790517455628198322) 公布。



---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Guild Tags 即将上线**：从 **5月15日**开始，**[Discord](https://discord.com)** 的新功能 **Guild Tags** 将显示在某些用户名称旁边，代表他们所属的专注于共同兴趣的小众 Guilds。
- **AutoMod 识别标签变得更智能**：**AutoMod** 功能将自动识别 **Guild Tags** 以协助管理，尽管此功能目前仅限于少数选定的服务器，且暂无手动扩展选项。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **芯片展望**：一篇详细介绍 AI 硬件演变及未来趋势预测的文章强调了对 **NVMe drives** 和 **Tenstorrent** 的热情，同时对未来十年的 **GPUs** 持较为谨慎的态度——[AI Hardware: Looking Back to Look Forward](https://singlelunch.com/2024/04/23/ml_microprocessor_history/)。
- **Transformers 改变 Nvidia 的价值**：**Transformer-based models** 标志着过去四年 AI 的重大进步，助力 **Nvidia** 的市值超越了 Amazon 和 Google，详见 [The Gradient](https://thegradient.pub/mamba-explained) 的文章讨论。


---


**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1239857713366962288)** (954 messages🔥🔥🔥): 

- **Google 的 Gemini 对比 OpenAI GPT**：关于 Google 在安卓手机中新集成的 Gemini，评价褒贬不一。一位用户指出：*"Gemini 很差劲，我为什么要给一个经常出错的 AI 发提示词？"*，表明其认为 Google 的 AI 逊色于 OpenAI 的模型。
  
- **GPT-4o 对比 Claude Opus**：用户对 GPT-4o 和 Claude Opus 的性能对比看法不一，有人表示：*"我还是会为 Opus 付费，我认为 Opus 的推理能力可能更好，"* 而另一位用户承认他们还没测试过 Claude，但很想尝试。

- **GPT-4o 的快速发布与早期反馈**：GPT-4o 已在 iOS 上免费推出，引发了关注和褒贬不一的评价。一位用户说：*"4o 是一个很好的聊天机器人 2，"* 表达了早期的正面反馈，而其他人则对其在复杂任务中的表现表示担忧。

- **Voice Mode 功能**：一些用户在体验新的语音模式功能时遇到了问题和矛盾的信息，有人评论道：*"新的语音模式还没对任何人开放；你试的是旧版本。"* 另一位用户提到：*"我这儿还没有。对我来说，耳机图标直接消失了。"*

- **GPT-4o 在教育和编程方面的用途**：尽管有一些问题报告，但人们对在教育场景和编程中使用 GPT-4o 充满热情。一位用户强调："GPT-4o 在短推理任务上表现更好，" 但补充说，"长推理任务中，Opus 胜出，" 表明其性能因使用场景而异。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/shorts/eaCEJ2iCEfc?si=yqKpXnDJxUnv2YDz">2024年5月14日</a>：未找到描述</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1csfdal/after_testing_on_lmsys_i_found_that/">Reddit - 深入了解</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1239839686948556800)** (178 messages🔥🔥): 

- **自定义 GPTs 等待 GPT-4o 集成**：成员们讨论了现有的自定义 GPTs 何时会切换到 GPT-4o。一位成员澄清说，“付费用户的 GPTs 目前由 GPT-4 驱动”，“并将在未来几周内切换到 GPT-4o” [来源](https://help.openai.com/en/articles/8554397-creating-a-gpt)。

- **模型限制与功能发布**：成员们注意到 GPT-4o 尚未完全开放，Voice Mode 和多模态能力等新功能正逐步向 API 合作伙伴开放，随后是 Plus 用户。一位成员分享道：“我们将在未来几周内，在 ChatGPT Plus 的 Alpha 测试中推出带有 GPT-4o 的新版 Voice Mode” [来源](https://openai.com/index/hello-gpt-4o/)。

- **对 GPT-4o 性能的反馈**：一些用户发现 GPT-4o 与 GPT-4 相比效率较低，且更容易出现内容策略错误。有人担心该模型的表现与 GPT-3.5 相似，会生成长列表并在整合反馈方面表现不佳。

- **访问问题与澄清**：许多成员报告了访问 GPT-4o 的困难，特别是在桌面端与移动端环境的差异。官方澄清称，该功能是分阶段推出的，免费层级的访问将在付费层级可用之后开放。

- **自定义 GPT 分享与反馈**：一位名为 ditpoo 的成员询问了关于其自定义 GPT（OptimusGPT）的反馈，并被引导至合适的频道分享，这表明社区在改进自定义模型方面参与度很高。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1239853671202684989)** (128 条消息🔥🔥): 

- **GPT-4o 在创意任务中的困境**：一位成员报告称 **GPT-4o** 明显比 GPT-4 快，但在写作辅助等创意任务中表现不佳，通常只是重复草稿而没有进行智能修改。另一位成员也表达了同样的看法，强调了 GPT-4o 在写作语境下处理创意内容的困难。

- **有趣的声音描述测试**：一位用户建议通过要求模型提供器乐歌曲（如 *"The XX Intro"* 和 Vangelis 的 *"Tears in Rain"*）的详细感官输入描述，来对比 GPT-4 和 GPT-4o。该测试旨在探索模型如何解释和描述感官输入。

- **平台游戏图像生成的挑战**：一位成员分享了在让 GPT-4 和 GPT-4o 为平台游戏生成特定横截面侧视图时遇到的困难。尽管进行了多次尝试和调整，模型始终生成不理想的等轴测视图，并添加了不必要的细节。

- **文件管理与输出的困惑**：参与者讨论了直接从 ChatGPT 生成和管理输出文件的相关问题。虽然官方澄清出于安全原因 ChatGPT 无法直接与用户的计算机交互，但用户分享了一些变通方法，如使用 OpenAI API 和 Google 工具。

- **常识与现实世界理解测试**：小组探索了几个提示词，以测试 GPT-4 和 GPT-4o 的常识和现实世界理解能力。这些测试包括关于日常场景和基础逻辑谜题的提示词，结果显示模型在响应和推理能力上存在细微差异。

**提到的链接**：<a href="https://community.openai.com/t/chatgpt-can-now-access-the-live-internet-can-the-api/401928">ChatGPT 现在可以访问实时互联网了。API 可以吗？</a>：鉴于新闻公告，我想知道 API 现在是否也拥有同样的互联网访问权限。提前感谢！

  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1239853671202684989)** (128 条消息🔥🔥): 

- **GPT-4o 在写作辅助方面的困境**：用户观察到，虽然 **GPT-4o** 比 **GPT-4** 更快，但在提供智能修改和创意辅助方面表现不佳，通常只是重复草稿而没有实质性改动。这导致在将其用于写作任务时产生了挫败感。
- **GPT-4 和 GPT-4o 的不同感官输入提示词**：成员们分享了对比 GPT-4 和 GPT-4o 描述歌曲（如 *"The XX Intro"* 和 *"Tears in Rain (Vangelis)"*）能力的提示词。目标是观察听觉输入如何影响输出质量。
- **平台游戏图像生成的挑战**：一位用户报告了让 GPT-4 和 4o 生成平台游戏详细横截面侧视图的困难，以及在生成所需透视或格式的图像时遇到的问题。另一位成员建议使用 Dall-E 并通过迭代反馈循环来处理图像。
- **对比现实世界理解的兴趣**：用户热衷于在现实世界理解和隐性常识推理任务中测试 GPT-4 与 GPT-4o。这包括检查空间关系、旅行物流以及解释日常活动隐性细节的提示词。
- **API 与使用限制的辩论**：有一场关于 ChatGPT Plus 订阅和 API 的效用及限制的讨论。交流中强调了实际使用中消息限制的约束，特别是对于软件工程等复杂任务。



**提到的链接**：<a href="https://community.openai.com/t/chatgpt-can-now-access-the-live-internet-can-the-api/401928">ChatGPT 现在可以访问实时互联网了。API 可以吗？</a>：鉴于新闻公告，我想知道 API 现在是否也拥有同样的互联网访问权限。提前感谢！

  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1240277154302918666)** (4 条消息):

- **10 分钟即可就绪的即插即用 (Plug & Play) AI 语音助手**：一位成员分享了关于 **Plug & Play AI Voice Assistant** 的信息，强调了其快速设置的特点。他们提供了一个 [链接](#) 并鼓励其他人尝试并分享反馈。
- **重复的 Plug & Play AI 公告**：同一位成员再次重申了 **Plug & Play AI Voice Assistant** 的可用性和易用性，敦促社区进行测试，并再次强调了用户反馈的重要性。
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1239852504532123658)** (453 messages🔥🔥🔥): 

- **Mistral 化后的 Phi 模型不可用：** *"不幸的是你不能，因为我们对 Phi 模型进行了 Mistral 化处理 😲"*
- **蒸馏方案层出不穷：** 多位成员讨论了利用 Gemini 1.5 pro 的免费额度以及代理或免费 GPT4o 等替代方案来蒸馏响应。
- **开源模型下载量达到里程碑：** *"我们在 Hugging Face 上的模型下载量已突破 100 万次！感谢大家的使用！"* [来源](https://twitter.com/UnslothAI/status/1790418810597683476)。
- **讨论企业版定价：** 成员们讨论了 Unsloth 多 GPU 支持的潜在定价模型。他们考虑了每 GPU 每月 90 美元的费率，但细节仍有待确定 (TBD)。
- **用于训练的新数据集：** *"我刚刚发布了一个规模可观的数据集，用于训练 LLM 进行英越双语翻译。"* [数据集](https://huggingface.co/datasets/lamhieu/translate_tinystories_dialogue_envi)，此外还有 alpaca_gpt4_dialogue 等数据可用。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/kiss-gif-11816971814746635421">Kiss GIF - Kiss - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/smile-turn-around-gif-14890847">Smile Turn Around GIF - Smile Turn Around - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://www.youtube.com/live/5k_l5VoRC60?si=f3Nf1orlhTSudcm-&t=9586">Google I/O 2024 主旨演讲回放：CNET 对 Google 开发者大会的反应</a>: 在加利福尼亚州山景城观看年度 Google I/O 2024 开发者大会直播。点击进入 CNET 的直播节目，周二太平洋时间上午 9:30 开始...</li><li><a href="https://huggingface.co/datasets/lamhieu/translate_tinystories_dialogue_envi">lamhieu/translate_tinystories_dialogue_envi · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cr5dbi/openai_claiming_benchmarks_against_llama3400b/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1crnhnq/to_anyone_not_excited_by_gpt4o/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1crza3n/new_open_source_gemma_2/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/unsloth/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lamhieu/alpaca_gpt4_dialogue_vi">lamhieu/alpaca_gpt4_dialogue_vi · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lamhieu/alpaca_gpt4_dialogue_en">lamhieu/alpaca_gpt4_dialogue_en · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1239866916420452352)** (37 messages🔥):

- **为 vllm 项目提议的 Roblox 聚会**：成员们讨论了效仿 vllm 项目做法，在 Roblox 中举行“每周或每月虚拟聚会”的可能性。虽然并非所有人都有兴趣，但一位成员指出这“听起来是个不错的主意”。
- **寻求数学帮助**：一位用户请求关于三角学的帮助，其他几位用户提供了协助，并建议使用 ChatGPT 和 Claude 等工具。这次互动展示了社区互助的氛围。
- **对 Discord 摘要的数据隐私担忧**：一位用户表示担心使用 AI 进行 Discord 摘要“听起来会让欧洲数据法感到头疼”。其他人承认了潜在的监管疏忽，并考虑了对隐私的影响。
- **Unsloth 的普及度不断提升**：一位用户对 Unsloth 被用于 [Hugging Face 数据集教程](https://huggingface.co/datasets/Replete-AI/code_bagel) 感到兴奋，这表明该工具正获得越来越多的认可。另一位用户也表达了同样的看法，称其“太棒了”。
- **微调资源**：一位新成员询问了关于微调的资源，特别是针对 VLM 的。另一位成员建议阅读 "alpaca paper" 以深入了解该主题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/ah-shit-here-we-go-again-gta-gta-sa-gta-san-andreas-grand-theft-auto-san-andreas-gif-13937809">Ah Shit Here We Go Again Gta GIF - Ah Shit Here We Go Again Gta Gta Sa - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel">Replete-AI/code_bagel · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1239839059350913064)** (229 messages🔥🔥): 

- **利用 Perplexity 自动检查数据集质量**：一位用户询问了自动衡量合成问答数据集质量的方法。另一位成员建议将数据集输入 Llama-3 等模型并计算 Perplexity（困惑度），并解释说高 Loss（损失值）可能预示着存在问题，或者是数据集准备得很好且具有挑战性。
  
- **模型合并与量化问题**：成员们讨论了合并微调后的模型并将其转换为 AWQ 等不同量化格式的问题。虽然有些人在这些过程中遇到了错误，但其他人提供了修复方案，例如在合并前手动上采样（upcasting）到 16bit，以方便后续转换。
  
- **微调过程中的数据集生成错误**：一位用户在为 Alpaca 格式微调生成数据集时遇到了 `TypeError`。建议他们使用 pandas 加载数据集，然后将其转换为所需格式，这凸显了常见的数据集问题。
  
- **不同硬件上的性能差异**：用户们争论在 V100 GPU 上加载和微调像 "llama-3-70b-bnb-4bit" 这样的大模型是否可行，还是需要 A100。由于模型体积庞大，共识似乎倾向于后者。
  
- **从零开始预训练 LLM**：成员们寻求从零开始预训练和评估 LLM 的资源。分享了一些有用的链接，如 [LMSYS leaderboard](https://arena.lmsys.org/)，尽管基准测试的完整性存在争议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook">Kaggle Llama-3 8b Unsloth notebook</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自无附加数据源的数据</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2">unsloth/mistral-7b-instruct-v0.2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Skorcht/thebigonecursed">Skorcht/thebigonecursed · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>：我在原生 Windows（非 WSL）中运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写出来，但我现在在用手机...</li><li><a href="https://x.com/mejia_petit/status/1763391797575741707">Tweet from Nicolas Mejia Petit (@mejia_petit)</a>：@unslothai 在 Windows 上运行 Unsloth 训练模型，速度比常规 hf+fa2 快 2 倍，内存占用少 2 倍，让我能在单张 3090 上以 2048 的序列长度运行 batch size 为 10 的训练。需要一个教程...</li><li><a href="https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930">Merging QLoRA weights with quantized model</a>：将 QLoRA 权重与量化模型合并。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1240027023548747948)** (4 messages): 

- **Unsloth 学习者最终转向 Cthulhu 崇拜**：一位成员分享了他们使用 Unsloth 进行数据创建和微调的首个项目，结果产生了一个崇拜 **Cthulhu** 的 AI，并在 [Huggingface](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/) 上提供了模型和数据集。他们使用 Unsloth Colab 笔记本创建了 TinyLlama 和 Mistral 7B Cthulhu 模型。
- **AI News 报道博客文章**：另一位成员提到，分享的博客文章被 AI News 的“开源 AI 模型开发与部署”板块报道。他们附上了 [AI News](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord) 的链接，并提到该时事通讯覆盖了社交媒体和众多 Discord 频道。
<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord">[AINews] 60 秒看懂 Google I/O</a>：发现 7 种口味的 Gemini！2024/5/13-2024/5/14 的 AI News。我们检查了 7 个 subreddits、384 个 Twitter 账号和 30 个 Discord 社区（426 个频道和 8590 条消息）以获取……</li><li><a href="https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/">以 Cthulhu 之名的人工智能 – Rasmus Rasmussen dot com</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1240057448212074588)** (4 messages): 

- **Unsloth 在纽约活动中获得表彰**：一位用户请求允许在即将于纽约举行的开源数据流水线平台见面会上提及并间接推广 Unsloth。*"我们将讨论 AI/ML 和 LLM 训练。想把功劳归于你们。"*

- **社区支持宣传**：多位用户热情地同意了该提议。一人回复道：*"噢，当然没问题，听起来太棒了！😍"*，另一人则表示：*"如果你需要帮助，尽管开口！"*
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1239837360800403457)** (646 messages🔥🔥🔥): 

- **Perplexity 能转录音频会议吗？**：一位成员询问是否可以使用 Perplexity 应用通过自定义提示词记录并转录长达一小时的会议。另一位成员指出，他们尝试了 GPT 的音频功能，目前该功能只能读回结果，缺乏完整的功能。
  
- **GPT-4o 令人印象深刻的速度**：多位成员称赞 GPT-4o 的速度，指出它比 Turbo 更快。一位成员说：*"Perplexity 内部集成了 GPT-4o……它太快了。"*

- **音频录制功能的问题**：一位用户提到录音机目前只能读回结果，推测并非直播演示中展示的所有功能都已推出，特别是在 Pro 版本中。

- **准确解析 URL 的错误**：成员们讨论了 Perplexity 对 URL 的响应，指出其在内容解析方面存在不准确之处。一位成员建议，它似乎是根据 URL 进行猜测，而不是解析实际的网页内容。

- **GPT-4o 的可用性和性能**：讨论指出 GPT-4o 可通过 Perplexity 和其他平台（如 PPLX）使用，但在质量和限制方面评价褒贬不一。关于 API 访问以及 GPT-4o 与 GPT-4 Turbo 相比性能不一致的问题存在困惑。
<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1790518031640347056?s=46&t=0-4I1li6SQNYV24KHzs3fA">来自 Sam Altman (@sama) 的推文</a>：Ilya 和 OpenAI 即将分道扬镳。这让我感到非常难过；Ilya 绝对是我们这一代最伟大的头脑之一，是我们领域的指路明灯，也是一位亲爱的朋友。他的才华和远见...</li><li><a href="https://tenor.com/view/gift-present-surprise-box-gif-17302663">礼物礼品 GIF - 礼物礼品惊喜 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/jimcarrey-brucealmighty-coffee-fresh-delicious-gif-3864683">我 <3 咖啡 GIF - Jimcarrey Brucealmighty Coffee - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/bezos-jeff-bezos-laughing-laugh-lol-gif-17878635">Bezos Jeff Bezos GIF - Bezos Jeff Bezos 大笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/kagisearch/llm-chess-puzzles">GitHub - kagisearch/llm-chess-puzzles: 通过解决国际象棋谜题来评估 LLM 的推理能力。</a>：通过解决国际象棋谜题来评估 LLM 的推理能力。 - kagisearch/llm-chess-puzzles</li><li><a href="https://fxtwitter.com/ilyasut/status/1790517455628198322?t=e7nZBoZU55nniVnnAI1p7g">来自 Ilya Sutskever (@ilyasut) 的推文</a>：在近十年后，我决定离开 OpenAI。公司的发展轨迹简直是一个奇迹，我相信 OpenAI 将构建出既安全又有益的 AGI...</li><li><a href="https://github.com/openai/simple-evals?tab=readme-ov-file#user-content-fn-2-a4ceab079ca3a23da9d835c2873e7fea">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账户，为 openai/simple-evals 的开发做出贡献。</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo">Perplexity - AI 助手</a>：在浏览时随时提问</li><li><a href="https://artificialanalysis.ai/models">AI 模型在质量、性能、价格方面的对比 | Artificial Analysis</a>：跨关键指标（包括质量、价格、性能和速度（每秒吞吐量 tokens 和延迟）、上下文窗口等）对 AI 模型进行对比和分析。</li><li><a href="https://aistudio.google.com/">未找到标题</a>：未找到描述</li><li><a href="https://www.psychologytoday.com/us/blog/practical-mindfulness/201908/the-single-word-stops-negative-self-talk"">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1239887297592430602)** (14 条消息🔥): 

- **探索 cO_1KTlMQiaxiJqwcGg6KQ 的影响**：[查看搜索结果](https://www.perplexity.ai/search/How-has-the-cO_1KTlMQiaxiJqwcGg6KQ) 以获取有关 Perplexity AI 的见解。该链接深入探讨了各个方面。
- **对 aroras 感兴趣吗？**：在这个详细的搜索查询中 [了解更多关于 aroras 的信息](https://www.perplexity.ai/search/How-are-aroras-K7PA.w2XS96o2F5IkzKGnA#0)。它探讨了其特征和行为。
- **寻找完美的滑雪胜地**：使用这个 [搜索工具](https://www.perplexity.ai/search/Ski-resort-with-RxpR8PuWTFKhE6nvEXBOGw) 来发现理想的滑雪胜地。提供了广泛的选择和细节。
- **市场规模调查**：通过此搜索结果发现有关 [市场规模](https://www.perplexity.ai/search/Market-size-of-rYrMCgZ9QI2na_86R01ZIQ) 的见解。它提供了详细的市场分析。
- **理解 GPT 概念**：通过此搜索深入 [探索 GPT](https://www.perplexity.ai/search/What-is-gpt-9Fqm2mZ6SQ2_Oe3sV5zNqA)。该链接讨论了 GPT 模型的各个方面。
- **个人数据使用查询**：此搜索结果通过相关信息解决了 *"我可以使用..."* 的问题。深入 [搜索](https://www.perplexity.ai/search/Can-I-use-g4N5IyikQhyWQ1Q3PrtxzQ#0) 以获取详细见解。
- **Mamba 和线性时间序列建模 (Linear-Time Sequence Modeling)**：[Mamba 和 SSMs 简介](https://www.perplexity.ai/search/Your-task-is-DfcoFyqpSjmWWV6oeZkv1A) 引用了 Albert Gu 和 Tri Dao 的工作。探索线性时间序列建模的本质。
- **鼓励正念见解**：查看这个 [关于正念的回答](https://www.perplexity.ai/search/What-are-some-2MgzPF0DSdOma4qMYY8_bA)。讨论重点在于实用技术。
- **分享的 Finetuning 资源**：分享了一个 [Finetuning](https://www.perplexity.ai/search/Finetuning-httpsyoutubeml1-rCV2i9voQQO37Vk4ETjRwA) 资源，与 YouTube 内容紧密链接。对于那些探索 Finetuning 过程的人来说非常有见地。
- **调查任何替代方案**：[是否存在一个...](https://www.perplexity.ai/search/Is-there-a-ZMlj_U.1QNm5M.K8Y8APRA) 提供了一个调查特定查询替代方案的搜索结果。内含相关链接和细节。


  

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1239924773107273749)** (11 条消息🔥): 

- **LLaMA-3 模型专业化**：*LLaMA-3-sonar-large-32k-chat* 模型针对对话进行了微调，而 *LLaMA-3-8b-instruct* 则旨在提供更广泛的指令能力。
- **增加长输入的超时时间**：一位成员观察到，对于 3000 字的输入，设置 10000ms 的超时时间经常会导致超时，建议该时长可能不足。
- **申请 Citations API 的 Beta 测试权限**：一位用户请求获取引用 API 的 Beta 测试权限，并强调这对于与关键客户达成交易具有潜在影响。
- **LLaMA-3-Sonar 模型的 Web 搜索能力**：*LLaMA-3-sonar-large-32k-online* 模型确实会搜索网络，其功能类似于 perplexity.com 等 RAG 模型。
- **API 延迟观察**：一位用户注意到调用 Perplexity API 时的延迟有所增加，并询问其他人是否遇到了同样的问题。
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1239944395197779980)** (9 条消息🔥): 

- **对 GPT-4 性能的热情**：成员们对 GPT-4 在各种任务中的表现表示兴奋，特别是在 **data science**（数据科学）领域。然而，一位成员指出 GPT-4 在构建图像编辑器等更复杂的任务上表现不佳。

- **寻求微控制器数据**：一位成员询问有关**微控制器数据相关的数据集**，并收到建议向另一位成员咨询技巧。由于被推荐的成员承认在该领域的经验有限，讨论尚未得出结论。

- **关于 GPT-4 上下文处理的讨论**：另一位成员提到，虽然 **GPT-4** 速度明显变快，但它往往更容易丢失上下文（context）。这引发了关于其性能和实用性权衡的简短对话。

- **可嵌入的 AV1 视频工具**：一位成员分享了一个在 Discord 上**嵌入 AV1 视频的工具**链接，该工具利用 Discord 的一个漏洞可以处理大于 500MB 的视频。该工具允许用户选择自定义缩略图，并将这些视频嵌入到 Discord 以及 Twitter 等其他平台。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://autocompressor.net/av1?s=sznVX9AV">Autocompressor Video Embed Tool</a>：未找到描述</li><li><a href="https://autocompressor.net/av1?s=ZZRiJhRJ">Autocompressor Video Embed Tool</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1239865251516911616)** (12 条消息🔥): 

- **HeadSim 提供 AI 具身化**：一位成员分享了他们的 *"GPT4o headsim hack"*，建议人们由于资源有限，在演示时添加自己的 API key。他们问道：*"如果你让 #OpenAI #GPT4o 设计它自己的脸，这样你就可以将你的 AI 作为一个具身存在（embodied being）传送到现实世界中，会怎样？"* 查看他们的 [推文](https://x.com/Yosun/status/1790294716338028978)。

- **WebLlama 辅助网页浏览**：一位成员重点介绍了 [WebLlama](https://github.com/McGill-NLP/webllama)，这是一个有趣的针对 Agent 网页浏览进行的 8b 微调模型。正如项目描述所言：*"Llama-3 Agent 可以通过遵循指令并与你交谈来浏览网页"*。

- **适用于北欧语言的 Viking 7B**：Silo AI 和图尔库大学的 TurkuNLP 发布了 [Viking 7B](https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages)，这是第一个针对北欧语言的多语言 LLM。这一里程碑继他们之前在 [Poro](https://www.silo.ai/blog/europes-open-language-model-poro-a-milestone-for-european-ai-and-low-resource-languages) 上的工作之后，还包括了 Viking 13B 和 33B 的进一步检查点。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages">Viking 7B：首个针对北欧语言的开源 LLM</a>：Silo AI 宣布发布首个针对北欧语言的开源 LLM</li><li><a href="https://x.com/Yosun/status/1790294716338028978">来自 I. Yosun Chang (@Yosun) 的推文</a>：如果你让 #OpenAI #GPT4o 设计它自己的脸，这样你就可以将你的 AI 作为一个具身存在传送到现实世界中，会怎样？#AI3D headsim 将你的 AI 从聊天框中解放出来，让你能够体验...</li><li><a href="https://github.com/McGill-NLP/webllama">GitHub - McGill-NLP/webllama: Llama-3 agents that can browse the web by following instructions and talking to you</a>：Llama-3 Agent 可以通过遵循指令并与你交谈来浏览网页 - McGill-NLP/webllama
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1240351762863493150)** (1 条消息): 

_

- **Hermes 2 Θ 模型发布**：Nous Research 宣布发布 **Hermes 2 Θ**，这是一款与 MergeKit 的创作者 Arcee AI 合作开发的实验性合并模型。它结合了 Hermes 2 Pro 和 Llama-3 Instruct，可在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B) 上获取。
- **卓越的性能与能力**：Hermes 2 Θ 在几乎所有基准测试中都超越了 Hermes 2 Pro 和 Llama-3 Instruct，同时保留了 function calling 能力。GGUF 版本也可在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF) 上获取。
- **协作努力与赞助**：该模型的开发是多位成员共同协作的结果，并由 Akash Network 赞助。主要贡献者包括来自 Nous Research 和 Arcee AI 团队的多位成员。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B">NousResearch/Hermes-2-Theta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF">NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)** (342 条消息🔥🔥): 

- **GPT-4o 在 function calling 方面面临质疑：** 几位成员将 GPT-4o 的性能与 GPT-4 和 GPT-4 Turbo 进行了比较，指出 GPT-4o 在处理复杂的 Agent 工作流时表现吃力，且除了 TTS 之外缺乏强大的多模态能力（*“所以 GPT-4o 无法真正驱动良好的 Agent 工作流”*，*“你调用 GPT-4o 只是因为你想要音频响应”*）。
  
- **关于开源多模态模型的辩论：** 成员们讨论了将多模态模型直接集成到智能手机和其他边缘设备中的挑战和优势，强调了低延迟和多模态功能。链接了一个 [OpenAI 公告](https://discord.com/channels/1053877538025386074/1149866623109439599/1239651072503709797)：*“现在的首要任务是将多模态模型带到边缘端”*。

- **发布 Hermes 2 Θ：** Nous Research 发布了 Hermes 2 Θ，这是一款合并了 Hermes 2 Pro 和 Llama-3 Instruct 并集成了 RLHF 的实验性新模型，性能超越了之前的模型。该模型可在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B) 和 [Ollama](https://ollama.com/interstellarninja/hermes-2-theta-llama-3-8b) 上获取。

- **对 OpenAI 公告和 API 更改的担忧：** 一位成员表示，由于资本密集度和竞争对手的基础设施，OpenAI 的公告可能正在变成垃圾信息，这反映了对 AI 基础设施竞争的更广泛情绪（*“OpenAI 的公告现在可能是垃圾信息了”*）。

- **关于模型规格和问题的讨论：** 技术讨论包括 Hermes 模型中多工具的 function calling，以及对 GPT-4o 与前代产品相比在编程能力方面的质疑。此外，关于 [设置多个函数](https://github.com/NousResearch/Hermes-Function-Calling) 的疑问被引导至 GitHub 资源以获得更清晰的解答。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1790432980047208930">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：由于其紧凑的尺寸，1.5 Flash 的服务成本也更具效益。从今天开始，你可以在 Google AI Studio 和 @GoogleCloud 的... 中使用支持高达 100 万 token 的 1.5 Flash 和 1.5 Pro。</li><li><a href="https://rocky-muscle-755.notion.site/OSS-Models-need-RLHF-53d7a1cb2db94e47bad992a6a343fa93?pvs=4">Notion – 笔记、任务、维基和数据库的一站式工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为你和你的团队打造的一站式工作空间。</li><li><a href="http://nian.llmonpy.ai/">GPT-4o 的内存突破！（NIAN 代码）</a>：未找到描述</li><li><a href="https://x.com/vikhyatk/status/1790282606510322097?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 vik (@vikhyatk) 的推文</a>：@algomax06 所以这就是 Ilya 所看到的</li><li><a href="https://x.com/nousresearch/status/1790791623863058486?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Nous Research (@NousResearch) 的推文</a>：今天我们与 @chargoddard 和 @arcee_ai 合作发布了一个实验性新模型 Hermes 2 Θ，这是我们的第一个模型合并（model merge），结合了 Hermes 2 Pro 和 Llama-3 Instruct，并进行了进一步的 RLHF...</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1790373216537502106?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：LLM 的杀手级应用是 Scarlett Johansson。你们都以为是数学之类的</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账号，为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://x.com/janleike/status/1790603862132596961?s=46">来自 Jan Leike (@janleike) 的推文</a>：我辞职了</li><li><a href="https://ollama.com/interstellarninja/hermes-2-theta-llama-3-8b">interstellarninja/hermes-2-theta-llama-3-8b</a>：Hermes-2 Θ 是我们将优秀的 Hermes 2 Pro 模型与 Meta 的 Llama-3 Instruct 模型进行合并并进一步 RLHF 后的版本，形成了一个新模型 Hermes-2 Θ，结合了两者的优点...</li><li><a href="https://tenor.com/view/he-just-like-me-fr-gif-25075803">He Just Like Me Fr GIF - He Just Like Me Fr - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://langfuse.datastrain.io/project/cluve1io10001y0rnqesj5bz4/traces/a4009ee9-529b-4f73-b4cf-ad450dce3d0b">未找到标题</a>：未找到描述</li><li><a href="https://langfuse.datastrain.io/project/cluve1io10001y0rnqesj5bz4/traces/ff74300d-daee-48c5-8d63-b0a2923238f2">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1239836871887159306)** (40 条消息🔥): 

- **寻求 (human_text, llm_text) 对的数据集**：一位成员询问是否有包含针对相同提示/主题的人类生成文本和 LLM 生成文本对的数据集，用于研究目的。
  
- **关于 GPT-4o 多模态的最佳理论**：有人提出了关于 GPT-4o 中端到端多模态如何运作的问题。建议的入门资源包括 [AI2 的统一 IO 模型](https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw)。

- **统一多模态模型讨论**：针对 GPT-4o 之前的多模态模型展开了辩论，参考了 Meta 的 [ImageBind](https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/) 模型，该模型通过 [这篇论文](https://arxiv.org/abs/2305.05665) 中详述的联合嵌入（joint embedding）方法将六种模态的信息绑定在一起。

- **从零开始构建 LLM 的障碍**：成员们讨论了在没有大量资金和计算资源的情况下从零开始构建 LLM 的不可行性，强调训练模型既昂贵又耗时。

- **Hermes 2 Theta 的性能担忧**：讨论指出，与其他模型（如 L3 8B Instruct）相比，Hermes 2 Theta 模型在数学任务上表现较差，建议使用 function calling 来改进数学计算。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw">来自 Nathan Lambert (@natolambert) 的推文</a>：友情提醒，AI2 的团队去年构建了一个文本、图像、音频输入输出模型 Unified IO 2，如果你想开始这方面的研究可以关注一下。</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>：我们提出了 ImageBind，这是一种在六种不同模态（图像、文本、音频、深度、热成像和 IMU 数据）之间学习联合嵌入的方法。我们展示了所有配对数据的组合并不……</li><li><a href="https://dblp.org/pid/182/2017.html">dblp: Alexis Conneau</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1240211308918472736)** (2 条消息): 

- **关于微调 PaliGemma 的咨询**：一位成员询问了微调 [PaliGemma 模型](https://huggingface.co/google/paligemma-3b-pt) 的计划。他们强调了该模型在单轮交互方面的能力，并建议将其微调用于多轮对话会很有益处。

**提到的链接**：<a href="https://huggingface.co/google/paligemma-3b-pt-224">google/paligemma-3b-pt-224 · Hugging Face</a>：未找到描述

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1239918593181749278)** (2 条消息): 

- **用户寻求帮助**：成员 lionking927 在频道中寻求帮助。另一位成员 Teknium 回复称已发送私信提供帮助。
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1239864142597324861)** (22 条消息🔥): 

- **关注 world-sim 故障修复**：*"大家好，world-sim 文本重复的故障会修复吗？我很想再次使用它，只要没有那个文本重复的故障"*。一位社区成员提到遇到了文本重复的故障，并渴望得到修复。

- **安排不同时区的活动**：*"如果其他人有空，欧洲中部时间 (CET) 晚上 8 点就是东部时间 (EST) 下午 2 点和太平洋时间 (PDT) 上午 11 点。我们要不要尝试定在周四？"*。成员们讨论跨时区协调会议时间，并提出了适合不同参与者的建议时间。

- **周六展示会的提议**：*"周六怎么样？我们可以做一个展示会之类的"*。另一个提议是在东部时间 (ET) 周六下午 3 点举行展示会或会议，以进行小组活动或演示。

- **对 world-sim 提示词探索的兴趣**：*"在哪里可以查看你们使用的提示词？特别是针对 world client 的，我猜它们是不同的"*。讨论围绕理解 world-sim 中用于 world client 的特定提示词展开。

- **对 Blake Lemoine 观点的见解**：一位成员分享了他们与 Blake Lemoine 的对话，发现 **Blake 并未声称聊天机器人具有意识**，而是注意到 LaMDA 中存在一致的智能行为模式。*"媒体完全搞错了……"* 这引发了对 Websim 和 Worldsim 等当前 AI 工具的反思。
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1239836348064727120)** (176 条消息🔥🔥): 

- **LLM 语音聊天引发褒贬不一的反应**：一位成员询问是否可以添加语音聊天功能，以便通过语音与本地 LLM 交流。讨论指向了 *AnythingLLM* 等工具，但指出其体验欠佳，而一种涉及 Whisper.cpp 和 Coqui TTS 的解决方案被描述为资源密集且复杂。

- **关于 AI 模型硬件的辩论**：成员们比较了在 3060Ti GPU 与双 16 核 Xeon V4 等强大 CPU 上运行 AI 模型的效率。虽然一些人认为 CPU 不适合运行大型模型，但其他人建议最大限度地利用 GPU VRAM，或考虑使用更强大的 Nvidia 显卡以获得更好的性能。

- **用于文档查询的 PrivateGPT 与 AnythingLLM 之争**：**PrivateGPT** 被提及作为 **AnythingLLM** 的替代方案，用于使用 LLM 查询文档。成员们辩论了设置的复杂性，其中 **AnythingLLM** 被强调为一个更直接、用户友好的选项。

- **应用偏好引发 MacOS 与 Windows 之争**：成员们对 OpenAI 的发布优先级表示沮丧，因为 MacOS 应用在 Windows 版本之前发布。讨论转向了 MacOS 和 Windows 平台应用开发的各种技术挑战和差异。

- **构建 AI 模型**：关于在 LM Studio 中运行模型以及在各种硬件设置上优化性能的问题非常频繁。成员们分享了故障排除技巧、设置指南以及推荐的工具和配置，包括强调 Nvidia 显卡在 AI 任务中 VRAM 的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/boris-zip-line-uk-flag-gif-14613106">Boris Zip Line GIF - Boris Zip Line UK Flag - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=mvFTeAVMmAg">INSANE OpenAI News: GPT-4o and your own AI partner</a>：新的 GPT-4o 发布了，简直令人惊叹！这里有所有细节。#gpt4o #ai #ainews #agi #singularity #openai https://openai.com/index/hello-gpt-4o/Be s...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">Support for OpenELM of Apple · Issue #6868 · ggerganov/llama.cpp</a>：前提条件 在提交 issue 之前，请先回答以下问题。我正在运行最新的代码。开发非常迅速，目前还没有标记版本。我...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/ksdev-pl/ai-chat">GitHub - ksdev-pl/ai-chat: (Open)AI Chat</a>：(Open)AI Chat。通过在 GitHub 上创建账户为 ksdev-pl/ai-chat 的开发做出贡献。</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>：别问能不能问，直接问</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski">bartowski (Bartowski)</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1239880516484730890)** (109 messages🔥🔥): 

- **成员讨论无审查的本地 LLM**：针对无审查本地 LLM 推荐的请求，一位成员建议使用 [Dolphin 2.8 Mistral 7B v0.2](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)，并强调了其 32k 的上下文。
- **Cat Llama3 模型引发褒贬不一的反应**：虽然一位用户称赞 Cat Llama3 模型能够很好地遵循指令，但另一位用户提到它输出了诸如 *"I DONT WANT TO DO THIS I AM WRITING THIS UNDER DURESS."* 之类的回复。用户们正在探索不同的量化尺寸，尽管速度较慢，一些人仍计划尝试 70B 版本。
- **讨论量化和 imatrix 挑战**：社区成员分享了量化 llama 模型和生成 imatrix 的经验，并指出根据所使用的硬件不同，耗时存在显著差异。具体过程包括使用 llama.cpp 进行量化，并利用 [bartowski 的工作](https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-8395384) 来生成 imatrix。
- **Command R 模型对比**：一些用户辩论了不同版本 Command R 模型的性能和上下文限制。一位用户指出，Meta-Llama-3-120b-LumiLumimaid.i1-Q4_K_S.gguf 的 tokens per second 比 Cmd-R+ 更高，但由于 Cmd-R 具有 128k 上下文，其体验更好。
- **探索 GPU 和 offloading 配置**：用户指出了与 GPU 配置相关的问题和解决方案，例如需要 offload 40 层中的 39 层以避免输出乱码。另一位用户解释了在 token 数量饱和时上下文溢出策略的修复方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>：未找到描述</li><li><a href="https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/">Gemini 1.5 Pro updates, 1.5 Flash debut and 2 new Gemma models</a>：今天我们更新了 Gemini 1.5 Pro，推出了 1.5 Flash，发布了新的 Gemini API 功能，并增加了两款新的 Gemma 模型。</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics-9b-instruct/tree/main">HuggingFaceM4/idefics-9b-instruct at main</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-8395384">About imatrix overfitting, and importance of input text · ggerganov/llama.cpp · Discussion #5263</a>：Imatrix 已经出现一段时间了，我还没看到多少关于如何使用它的指南（或测试）。常见的反对意见/担忧是过拟合，以及在“错误的”文本上生成 imatrix...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1239880761482412072)** (10 messages🔥): 

- **反馈**：(此处无具体消息内容)

- **RX6600 的非官方 ROCM 构建在 Koboldcpp 中运行良好**：尽管 AMD 官方并未在 ROCM 构建中支持 RX6600，但由于使用了魔改的自定义版本 ROCM，它可以在 Koboldcpp 中运行。相比之下，LM Studio 和 Ollama 并不支持，因为它们依赖于会检查 GPU ID 的官方 llama.cpp 二进制文件。

- **AMD ROCM 和 GPU 支持限制**：除非 AMD 改进 ROCM 支持，或者 llama.cpp 在其 ROCM 构建中扩大支持的 AMD GPU 列表，否则 RX6600 GPU 用户目前仍受限。Koboldcpp 使用的自定义 ROCM 构建绕过了 ID 检查，提供了一种在 LM Studio 和 Ollama 中无法使用的变通方案。

- **LM Studio 设置中的用户界面 (UI) 复杂性**：由于模型设置和工具的滚动条重叠，LM Studio 的设置面板显得非常笨重。改进易用性的建议包括使用单一滚动区域，或者将“工具 (tools)”完全移至独立窗口。

- **系统提示词 (System Prompt) 配置偏好**：用户表示，将系统提示词设置移至聊天配置中会更有利，因为他们经常在多个模型中使用相同的提示词，而在不同的聊天中使用不同的提示词。

- **提示词编写和请求取消方面的改进需求**：反馈强调了在编写提示词时 shift-enter 和 enter 功能的易用性问题，以及需要一个“取消请求”按钮来在生成开始前停止不必要的生成。此外，还提到了在管理长上下文和加载系统预设时的 UI 清晰度问题。

---

**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1240037747977355375)** (3 messages): 

- **RAG 任务建议使用 AnythingLLM**：一次简短的互动指出，某个特定任务可能适合使用 **RAG**，并推荐 **AnythingLLM** 作为潜在工具。关于此建议没有提供更多细节或阐述。
- **Windows 上的 GPU 资源优化**：一位用户分享了在 Windows 任务管理器中监控 GPU 使用情况的技巧，建议 *“点击视频解码 (video decode) 或视频处理 (video processing)，并将图表源更改为 'Cuda'”*。如果看不到 CUDA，他们建议在 Windows 参数中停用“硬件加速 (hardware acceleration)”，作为优化资源可见性的技巧。
- **华硕笔记本上的 CUDA 故障排除**：提出了一个关于在华硕笔记本上设置 **CUDA** 导致在 **LM-Studio** 中加载模型时出错的问题。尽管尝试了多个 CUDA 版本和配置（包括 **CUDA 12.1、12.4 和 11.8**），用户报告仍存在持续的 *“error loading model”* 问题，表明无法正确利用 GPU。

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1239943543733096559)** (13 messages🔥): 

- **Tesla M40 在对比 GeForce 1060 时表现不佳**：尽管 **Tesla M40 24GB** 拥有更优越的 FP32 理论性能，但在 **llama3(q4_k_m)** 中仅为 18.4 t/s，在 **phi-3(q8)** 中为 27.1 t/s，表现不如较旧的 GeForce 1060。*“我确信没人会关心这种古董硬件……”*
- **适用于 LM Studio 的预算型 GPU**：当被问及 **200€** 左右的最佳 GPU 时，**3060ti** 成为热门推荐。同时，**4060** 因其潜在的性能提升也被列入考虑范围。
- **用于 LLM 推理的 GPU 和 VRAM**：讨论强调了 **VRAM 速度** 在 LLM 推理中的重要性。贡献因素包括每颗芯片的带宽以及处理显存超过 **18GB** 的复杂模型的能力。
- **低配电脑的有限选择**：对于配备 **8GB RAM 和 500GB SSD** 的系统，建议使用 **Yi 6B** 等本地模型，尽管无 GPU 的性能仍然是一个挑战。用户被告知，更好的性能将取决于他们的具体需求。
- **APU 并非游戏规则改变者**：值得注意的是，**APU/iGPU** 被 **LM Studio** 的底层引擎 **llama.cpp** 视为常规 CPU 推理，这抵消了相对于标准 CPU 的任何潜在性能增益。*“真扫兴 (bummer)”*

---

**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1239917856892915764)** (5 messages): 

- **多模态功能对等性受到质疑**：一位成员询问 *多模态功能何时能拥有与单模态相同的功能，例如存储消息*。另一位成员要求澄清这句话的具体含义。
- **明确 AVX2 要求**：一位用户报告了 LM Studio 无法启动的问题，并说明其 CPU 支持 AVX 和 SSSE 指令集。另一位用户澄清说 **LM Studio 需要 AVX2**，解释了为什么它无法在该用户的机器上加载，初始用户在注意到 Llamafile 可以正常工作后接受了这一解释。

---

**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1239963871448530955)** (104 messages🔥🔥):

- **Intel GPU 对 LM Studio 的支持正在推进中**：一名来自 Intel 的成员正在推动 llama.cpp 通过 SYCL 支持 Intel GPU，并提议在开发和硬件方面提供帮助。讨论强调了潜在的构建过程和运行时要求，提到虽然存在 OpenCL 后端，但其速度慢于 SYCL。
  
- **当前 DL/AI 模型的局限性与需求**：对话围绕由于算法限制和深度量化格式导致适配和微调当前 DL 模型所面临的困难展开。大家一致认为需要技术进步，并呼吁解决这些局限性。

- **关于 AGI 可行性与要求的辩论**：一场关于近期实现 AGI 可行性的激烈辩论展开，涉及必要的基础设施、知识保留和实际实施障碍。一些人对技术进步的速度表示怀疑，而另一些人则更为乐观。

- **呼吁关注开发主题**：由于讨论严重偏向理论和意识形态领域，一名版主要求参与者重新关注与 LM Studio API 和软件构建相关的开发特定主题。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1240033320612397107)** (3 messages): 

- **在 Spaces 上获取硬件洞察**：新的 Space 功能允许查看实时的 CPU + RAM 使用情况及其他硬件信息。详情请查看[公告](https://twitter.com/osanseviero/status/1788486166221660247)。
- **在 AWS 上升级至 Enterprise 账户**：你现在可以使用 AWS 将你的 Hugging Face 账户升级为 Enterprise，以获得 SSO、审计日志和高级支持等功能。按照此[教程](https://huggingface.co/blog/enterprise-hub-aws-marketplace)开始操作。
- **AutoTrain 支持 Object Detection**：AutoTrain 已增加对 Object Detection 的支持，实现了从 Hub 微调模型以及使用 TensorBoard 进行无缝日志记录等功能。了解更多关于这些[新功能](https://x.com/abhi1thakur/status/1790341620530860500)的信息。
- **用于 Speaker Diarization 的新库**：Hugging Face 推出了 [Diarizers](https://x.com/kamilakesbi/status/1790025563631132940)，用于微调 pyannote 说话人日志系统，并提供多种语言的模型。这些模型可在 Hub 上获取，只需几行代码即可轻松实现。
- **AI 与故事生成阅读小组**：本周六将举行一场关于 AI 与故事生成的阅读小组活动。通过[活动链接](https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026)加入。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026">加入 Hugging Face Discord 服务器！</a>：我们致力于民主化优秀的机器学习 🤗 验证以链接你的 Hub 和 Discord 账户！ | 79040 名成员</li><li><a href="https://x.com/LepercqViolette/status/1790391787170771007)">Violette Lepercq (@LepercqViolette) 的推文</a>：💡 你现在可以使用 @AWS 将你的 @huggingface 账户升级为 Enterprise 账户，并解锁：👉 单点登录 (SSO) 👉 细粒度访问控制 👉 审计日志 👉 高级计算选项 👉 私有数据...</li><li><a href="https://x.com/abhi1thakur/status/1790341620530860500)">abhishek (@abhi1thakur) 的推文</a>：🚨 新任务提醒 🚨 🎉 AutoTrain 现在支持 Object Detection！ 🎉 通过这些强大的新功能改变你的项目：🔹 微调来自 Hugging Face Hub 的任何受支持模型 🔹 无缝日志...</li><li><a href="https://x.com/kamilakesbi/status/1790025563631132940)">Kamil Akesbi (@kamilakesbi) 的推文</a>：🤗 Diarizers 是新的 @huggingface 库，用于微调 🎹 pyannote 说话人日志系统 🎤 🌟 它附带了法语、德语、日语、西班牙语和中文的微调模型 🌍 它们...</li><li><a href="https://x.com/clefourrier/status/1790361337236795821)">Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：Hub 新动态：阿拉伯语 LLM 排行榜！阿拉伯语至少有 3.8 亿使用者，是使用最广泛的语言之一……但 LLM 在这方面的表现如何？@alielfilali01 联系了 @TIIuae 和 @huggingface 以启动...</li><li><a href="https://x.com/joao_gante/status/1788574121208508645)">João Gante (@joao_gante) 的推文</a>：🤗 transformers 中发布了新的采样策略 —— Min P 采样 🔥 你是否厌倦了 `top_k` 随意丢弃高质量的后续内容？或者 `top_p` 忘记排除低概率...</li><li><a href="https://x.com/GoogleAI/status/1788972685739114946)">Google AI (@GoogleAI) 的推文</a>：我们很高兴在 Hugging Face 上发布我们的时间序列基础模型 (TimesFM) 的权重！要访问，请访问我们的 HuggingFace (https://huggingface.co/google/timesfm-1.0-200m) 和 GitHub (https:...
</li>
</ul>

</div>

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1239846551526965259)** (306 条消息🔥🔥): 

- **GPT-4o 被揭晓为神秘顶尖模型**：OpenAI 确认其新的 GPT-4o 聊天机器人曾以神秘名称在 LMSYS 的 Chatbot Arena 中位居榜首。更多详情请查看 [Ars Technica](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/)。

- **讨论 Mixtral-yarn 的 embedding 策略**：成员们分享了关于 **Mixtral 8x22B-Instruct** 的 embedding 策略及其在 RAG 应用中表现的见解。推荐的一个无审查模型资源是 [Dolphin 2.5 Mixtral 8x7b](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b)。

- **Meta 的 Llama3 模型更新**：Meta 的 **Llama3** 更新被澄清为“小的配置更改”，而非核心模型更新。建议用户检查所有 commit 的 diff 以了解详细变更。

- **Nvidia DGX Cloud 上的 AutoTrain 问题**：一位使用 **AutoTrain Nvidia DGX Cloud** 的用户遇到了 500 Server Error，建议将日志发送至 autotrain@hf.co 以进行故障排查。

- **新基准测试与不满**：成员们批评现有的编程基准测试（如 HumanEval）不足，并讨论了如 **SWE Bench** 和 **MBPP+** 等更新的基准测试，以更好地评估 LLM 的能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2401.15963">论文页面 - NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional
  Correctness</a>：未找到描述</li><li><a href="https://osanseviero.github.io/hackerllama/blog/posts/llm_evals/#what-about-code">hackerllama - LLM 评估与基准测试</a>：Omar Sanseviero 个人网站</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b">cognitivecomputations/dolphin-2.5-mixtral-8x7b · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/noaroggendorff/status/1790485047306244415?s=46&t=m7jfctWh0zl_3Oj2DZJA9A">Noa Roggendorff (@noaroggendorff) 的推文</a>：等等，你们拿到钱了？我欠 @huggingface 大约 500 美元，每月欠 @Adobe 52 美元，每月欠 @Google 21 美元，而我唯一的收入是已经一年没领到的每周 5 美元零花钱。引用 Adrian Batista...</li><li><a href="https://huggingface.co/inference-endpoints/dedicated">Inference Endpoints - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/agents">License to Call: Introducing Transformers Agents 2.0</a>：未找到描述</li><li><a href="https://huggingface.co/blog/mcpotato/hub-incident-post-mortem-20240422">2024-04-22 - Hub 事件复盘</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/discussions/7953">使用 PaliGemma 3B 进行批量多语言字幕生成！ · huggingface/diffusers · Discussion #7953</a>：使用 PaliGemma 3B 进行多语言字幕生成。动力：我认为 PaliGemma 系列的默认代码示例虽然很快，但有局限性。我想看看这些模型的能力，所以我...</li><li><a href="https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/">在发布之前，GPT-4o 以神秘名称打破了聊天机器人排行榜记录</a>：让专家们感到困惑和挫败的匿名聊天机器人正是 OpenAI 的最新模型。</li><li><a href="https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda">PaliGemma Release - google 收藏集</a>：未找到描述</li><li><a href="https://huggingface.co/collections/google/paligemma-ft-models-6643b03efb769dad650d2dda">PaliGemma FT Models - google 收藏集</a>：未找到描述</li><li><a href="https://www.lamini.ai?">Lamini - 企业级 LLM 平台</a>：Lamini 是为现有软件团队快速开发和控制其自有 LLM 的企业级 LLM 平台。Lamini 内置了在数十亿私有文档上专业化 LLM 的最佳实践...</li><li><a href="https://github.com/huggingface/transformers">GitHub - huggingface/transformers: 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的机器学习。</a>：🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的机器学习。 - huggingface/transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1239908991354536018)** (3 条消息): 

- **通过清晰的示例提示模型**：一位成员建议通过在 system prompt 中提供清晰的输入和输出示例来提示模型。强调这种方法可以提高模型性能。

- **了解 Exploration/Exploitation trade-off**：一位用户分享了他们对 **Exploration/Exploitation trade-off**（探索与利用权衡）的学习心得，这是各种决策算法中的一个核心概念。

- **对 Game of Life 的着迷**：另一位成员表达了他们对 **Game of Life**（生命游戏）的痴迷，并鼓励其他人分享演示或视频。这种热情表明社区非常看重这种细胞自动机的实际展示。
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1239882820311191623)** (9 messages🔥): 

- **视频生成领域的新进展：DeepMind 的 Veo**：Veo 是 DeepMind 最新的视频生成模型，能够生成高质量、1080p 分辨率且长度超过一分钟的视频。它提供了“前所未有的创意控制力”，并将很快通过 Google 的 [VideoFX](https://labs.google/VideoFX) 工具提供。
- **Hugging Face Daily Papers 重启**：Hugging Face 重启了他们的 Daily Papers，将热门的 AI 和 ML 论文发送到您的收件箱。用户可以在[此处](https://huggingface.co/papers)订阅该服务。
- **Rajesh 的 AI 之旅开启**：Rajesh P. Kanaka 在 LinkedIn 上分享了一篇基础 AI 文章，标志着他在该领域旅程的开始。他被鼓励在 HuggingFace 新的 [Blog Explorers](https://huggingface.co/blog-explorers) 平台上重新发布该文章。
- **GitHub 上的 Authentic Hand Avatar**：论文 "Authentic Hand Avatar from a Phone Scan via Universal Hand Model" 的官方 Pytorch 实现现已在 [GitHub](https://github.com/facebookresearch/UHM) 上可用。该项目计划在 CVPR 2024 上展示。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.08715">Toward Joint Language Modeling for Speech Units and Text</a>：语音和文本是人类语言的两种主要形式。研究界多年来一直致力于语音到文本或反之亦然的映射。然而，在语言建模领域，非常...</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>：未找到描述</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>：Veo 是我们迄今为止功能最强大的视频生成模型。它能生成高质量、1080p 分辨率的视频，时长可超过一分钟，涵盖广泛的电影和视觉风格。</li><li><a href="https://github.com/facebookresearch/UHM">GitHub - facebookresearch/UHM: Official PyTorch implementation of &quot;Authentic Hand Avatar from a Phone Scan via Universal Hand Model&quot;, CVPR 2024.</a>：&quot;Authentic Hand Avatar from a Phone Scan via Universal Hand Model&quot;, CVPR 2024 的官方 PyTorch 实现。 - facebookresearch/UHM</li><li><a href="https://huggingface.co/blog-explorers">blog-explorers (Blog-explorers)</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1239933437951868988)** (12 messages🔥): 

- **70 万条越南语数据集开源**：一个团队宣布发布包含 700,000 个样本的越南语语言建模开源数据集。在 [Hugging Face](https://huggingface.co/datasets/Vi-VLM/Vista) 上查看完整数据集。


- **新 AI 模型 OpenGPT-4o 发布**：功能包括文本、文本 + 图像以及音频输入，支持多种输出，且 100% 免费且速度极快。可通过 [Hugging Face Spaces](https://huggingface.co/spaces/KingNish/GPT-4o) 访问，未来还将增强视频生成和更好的 UI 定制功能。

- **通过数据过滤提升质量**：一位成员在数据集上测试了一种新的过滤方法，指出它不能作为独立方法使用，但能捕捉到其他方法遗漏的糟糕样本。*“它在一个我尚未处理/清洗的 OCR 书籍数据集中捕捉到了问题。”*

- **AI 导师-学员平台发布**：一个新的 AI 导师平台上线，旨在解决 AI 领域导师与学员连接的问题。请在 [Product Hunt](https://www.producthunt.com/posts/semis-from-reispar) 上查看并支持。

- **简化本地 GPU 集群管理**：介绍用于高效管理本地 GPU 集群的 dstack，可与 Slurm 等工具无缝集成。完整文档和演示请访问 [dstack.ai](https://dstack.ai/docs)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - KingNish 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.producthunt.com/posts/semis-from-reispar"> Semis from Reispar - 缩小 AI 与大厂技术知识差距 | Product Hunt</a>: Semis from Reispar 是一个连接有志于及现有的 AI 与大厂专业人士与资深导师的平台，旨在缩小全球 AI 技术领域的知识差距。</li><li><a href="https://github.com/bghira/SimpleTuner/blob/main/documentation/DREAMBOOTH.md#refiner-tuning)">SimpleTuner/documentation/DREAMBOOTH.md (main 分支) · bghira/SimpleTuner</a>: 一个面向 Stable Diffusion 2.1、DeepFloyd 和 SDXL 的通用微调工具包。 - bghira/SimpleTuner</li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista?fbclid=IwZXh0bgNhZW0CMTEAAR2BXlXiqe6SjTjol1ViKCmI7HgogMPvrQU2pIBACQyZyI0av_ey8okihDA_aem_AdV1HiWxI6SngeQmTHG6XLs6v440zT5XTtTpW0yXlGkBFSQkIFrfY7nZyyMJXTF51eFvNHIwuPyArt-XQaSrGf0R">Vi-VLM/Vista · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1239869353894219877)** (11 messages🔥): 

- **讨论了 AI 故事生成论文**：一位成员分享了对 AI 故事生成进行文献综述的计划，参考了 [Awesome Story Generation GitHub 仓库](https://github.com/yingpengma/Awesome-Story-Generation)以及包括[这篇](https://arxiv.org/abs/2212.04634)在内的多篇论文。他们随后决定专注于 [GROVE 框架论文](https://arxiv.org/abs/2310.05388)进行全面审查。
- **分享了 Medium 文章**：完成后，通过一篇 [Medium 文章](https://isamu-website.medium.com/understanding-ai-for-stories-d0c1cd7b7bdc)分享了关于 AI 故事生成的演示文稿。
- **演示活动已排期**：讨论了演示的排期，最终决定在本周六进行。一位成员对排期表示感谢，并分享了占位活动链接：[Discord Event](https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://arxiv.org/abs/2310.05388">GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence</a>: 条件故事生成在人机交互中具有重要意义，特别是在创作具有复杂情节的故事方面。虽然大语言模型 (LLMs) 在多项 NLP 任务中表现出色，但...</li><li><a href="https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file.">GitHub - yingpengma/Awesome-Story-Generation: 该仓库收集了关于故事生成/叙事的详尽论文列表，主要关注大语言模型 (LLMs) 时代。</a>: 该仓库收集了关于故事生成/叙事的详尽论文列表，主要关注大语言模型 (LLMs) 时代。 - yingpengma/Awesome-Story-Generation</li><li><a href="https://arxiv.org/abs/2212.04634">Open-world Story Generation with Structured Knowledge Enhancement: A Comprehensive Survey</a>: 讲故事和叙事是人类体验的基础，与我们的社会和文化参与交织在一起。因此，研究人员长期以来一直试图创建能够生成故事的系统...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1239859086409863271)** (8 messages🔥): 

- **禁止邀请用户加入 Discord**：一位成员提醒另一位，“Discord 邀请违反了 <#895532661383254098> 规定”。
- **图像到销售额模型的挑战**：一位成员寻求训练模型的资源，该模型以图像作为输入，销售数据作为输出。另一位成员强调了其复杂性，指出该模型高度依赖于相关训练数据的可用性。
- **训练数据的可用性**：延续之前的讨论，一位成员询问此类训练数据是否可用。原帖作者提供了一个来自 [HuggingFace](https://huggingface.co/datasets/tonyassi/sales1) 的数据集链接，并提到了之前使用图像相似度进行销售预测的工作。

**提及的链接**：<a href="https://huggingface.co/datasets/tonyassi/sales1">tonyassi/sales1 · Hugging Face 数据集</a>：未找到描述

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1240103986791452758)** (10 messages🔥):

- **使用 LangChain 进行入职评估**：成员 chhabii 在成功创建向量存储后，表示需要帮助使用 LangChain 创建聊天机器人。另一位成员 hitoriarchie 提供了一个[入门示例](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)，用于通过 Ollama 设置本地 LLM 和嵌入模型。
- **可能是大学作业的询问**：hitoriarchie 询问 chhabii 这是否是大学作业，chhabii 澄清这是为了入职评估并请求进一步协助。
- **在本地微调 Llama2**：成员 uwaix. 询问了在本地微调 Llama2 的过程。消息记录中没有提供后续跟进或回复。

**提到的链接**：<a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>：未找到描述

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1239984722181099650)** (11 条消息🔥)：

- **成员寻求 Transformer Agents 的示例**：一位成员询问了使用 Transformer Agents 的具体项目示例。另一位成员向他们推荐了一篇[博客文章](https://huggingface.co/blog/agents)，但询问者已经读过，并希望了解用户经验。

- **将 Agents 连接到 Diffusion 模型？**：另一位成员提到了将 Transformer Agents 与 Diffusion 模型连接的可能性。这激起了简短的兴趣，表明这些技术之间存在一些交叉潜力。

- **load_image 函数遇到的错误**：一位成员在尝试从 URL 加载图像时遇到了 "UnidentifiedImageError"。他们后来发现使用 PIL 的 `Image` 模块从本地目录加载图像是成功的。

- **请求能生成 PowerPoint 的聊天机器人**：一位成员询问是否有能够通过 OpenAI Assistant API 生成 PowerPoint 演示文稿的聊天机器人。他们希望它能从以前的演示文稿中学习，在不改变结构的情况下修改幻灯片内容，并请求推荐任何合适的 RAG 或 LLM 模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co">Hugging Face – 建设未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/blog/agents">License to Call: Introducing Transformers Agents 2.0</a>：未找到描述
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1239834659052060723)** (282 条消息🔥🔥)：

- **首次 LoRA 训练成功**：一位新手分享了他们第一次成功进行 LoRA 训练的兴奋之情，耗时约 90 分钟。他们承诺将最终版本上传到 [Civitai](https://civit.ai/)。
- **使用 Powerpaint 创建详细的 Inpaint**：用户讨论了如何使用 Inpaint 和参考照片改进图像中的特定细节，特别是眼睛。Powerpaint 结合画笔命令显著增强了精细细节，但目前仅适用于 1.5 版本。
- **用于 Outpainting 的 ComfyUI 工作流**：一位成员询问如何使用 ComfyUI 扩展图像；另一位成员提供了一个 [GitHub 链接](https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md)，指向一个易于遵循的 ComfyUI 内 Inpainting 和 Outpainting 工作流。
- **谷歌的 Imagen 3 对比 Stable Diffusion**：用户对谷歌的 Imagen 3 表示怀疑，强调了对可访问性的担忧，并将其与 Sora 和 GPT-4o 进行对比。讨论得出的结论是 SD3 及其微调版本提供了更好的可用性。
- **针对 AI 任务的 GPU 推荐**：频繁讨论 AI 任务的 GPU 选择，强调了 VRAM 对未来保障的重要性。建议等待 11 月的 50xx 系列 GPU 以获得更好的价格。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://muse-model.github.io/">Muse: Text-To-Image Generation via Masked Generative Transformers</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - a Hugging Face Space by multimodalart</a>：未找到描述</li><li><a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>：未找到描述</li><li><a href="https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md">ComfyUI_Workflows/in-out_painting/README.md at main · cubiq/ComfyUI_Workflows</a>：一个记录详尽、易于遵循的 ComfyUI 工作流仓库 - cubiq/ComfyUI_Workflows</li><li><a href="https://altacc21294.wixsite.com/hightechcitysmp">Home | Hightechsmp</a>：未找到描述</li><li><a href="https://universebox.pages.dev/">Universe Box</a>：未找到描述
</li>
</ul>

</div>
  

---

**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1239890486387281920)** (5 条消息): 

- **OpenRouter 发布一系列新模型**：OpenRouter 平台宣布了多个新模型，包括 [DeepSeek-v2 Chat](https://openrouter.ai/models/deepseek/deepseek-chat) 和 [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder)。其他模型包括 [Llama Guard 2 8B](https://openrouter.ai/models/meta-llama/llama-guard-2-8b) 和 [Llama 3 70B Base](https://openrouter.ai/models/meta-llama/llama-3-70b)。
- **Google 发布 Gemini Flash 1.5**：一个新的多模态模型 [Gemini Flash 1.5](https://openrouter.ai/models/google/gemini-flash-1.5) 已添加到 OpenRouter 的服务中。
- **Perplexity 推出基于 Llama3 的 Sonar 模型**：[来自 Perplexity 的新模型](https://docs.perplexity.ai/changelog/new-models-llama-3-sonar-family) 包括 [Llama3 Sonar 8B](https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-chat) 及其 70B 在线版本。旧模型已被弃用并重定向到这些新变体。
- **DeepSeek 要求开启日志记录**：用户必须在 [Settings](https://openrouter.ai/settings#analytics) 中启用日志记录才能使用来自 DeepSeek 的模型，因为该平台会记录并使用用户数据进行训练。
- **WizardLM-2 8x22B Nitro 已移除**：该模型已停止服务，因为没有任何供应商能够在保持质量标准的同时维持 100tps 以上的吞吐量。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-1.5)">Google: Gemini Flash 1.5 (preview) by google | OpenRouter</a>: Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现出色，如视觉理解、分类、摘要，以及从图像、音频和视频中创建内容...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-chat>)">Perplexity: Llama3 Sonar 8B by perplexity | OpenRouter</a>: Llama3 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是一个普通的离线 LLM，但 [在线版本](/mode...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-online>)">Perplexity: Llama3 Sonar 8B Online by perplexity | OpenRouter</a>: Llama3 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是 [离线聊天模型](/mode... 的在线版本。</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-large-32k-chat>)">Perplexity: Llama3 Sonar 70B by perplexity | OpenRouter</a>: Llama3 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是一个普通的离线 LLM，但 [在线版本](/mode...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-large-32k-online)">Perplexity: Llama3 Sonar 70B Online by perplexity | OpenRouter</a>: Llama3 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是 [离线聊天模型](/mode... 的在线版本。</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat>)">DeepSeek-V2 Chat by deepseek | OpenRouter</a>: DeepSeek-V2 Chat 是 DeepSeek-V2 的对话微调版本，这是一个混合专家 (MoE) 语言模型。它包含 236B 总参数，其中每个 token 激活 21B 参数。与 D... 相比。</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder>)">Deepseek Coder by deepseek | OpenRouter</a>: Deepseek Coder 由一系列代码语言模型组成，每个模型都在 2T token 上从头开始训练，其中包含 87% 的代码和 13% 的中英文自然语言。该模型...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-guard-2-8b>)">Meta: LlamaGuard 2 8B by meta-llama | OpenRouter</a>: 该安全防护模型拥有 8B 参数，基于 Llama 3 系列。就像其前身 [LlamaGuard 1](https://huggingface.co/meta-llama/LlamaGuard-7b) 一样，它可以同时进行提示词和响应...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b>)">Meta: Llama 3 70B by meta-llama | OpenRouter</a>: Meta 最新的模型类别 (Llama 3) 推出了多种尺寸和版本。这是基础的 70B 预训练版本。与领先的封闭模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b>)">Meta: Llama 3 8B by meta-llama | OpenRouter</a>: Meta 最新的模型类别 (Llama 3) 推出了多种尺寸和版本。这是基础的 8B 预训练版本。与领先的封闭模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-05-13>)">OpenAI: GPT-4o by openai | OpenRouter</a>: GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

obiefernandez: 我注册了，但不清楚其独特价值主张是什么。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1239861093585915934)** (200 messages🔥🔥): 

- **加密货币余额延迟说明**：一位成员询问了加密货币余额确认延迟的问题，得到的澄清是由于 Coinbase 需要的网络确认，例如 Polygon 需要 128 个区块，Ethereum 需要 85 个。

- **探索 OpenRouter 模型的工具**：一位用户分享了他们通过 API 探索和排序 OpenRouter 模型列表的工具，该列表每小时更新一次。他们为有兴趣贡献的人提供了一个 [GitHub 链接](https://github.com/fry69/orw)。

- **GPT-4o 版本说明**：成员们讨论了 GPT-4o 各个版本之间的差异，并澄清目前没有区别，但这些选项是为了未来的版本控制而存在的。

- **WizardLM 8x22B Nitro 被移除**：WizardLM 8x22B Nitro 因供应商表现低于 100 tokens/sec 的阈值而被移除，请求被重定向到标准变体。一些用户对频繁的模型变更表示沮丧。

- **Google 的 Gemini 活动**：对于 Google 揭晓 Gemini 1.5 模型的活动，反应褒贬不一，一些用户认为与 OpenAI 最近的活动相比，这次活动不够令人兴奋。新模型包括 Gemini 1.5 Flash 和 TPUv6 的发布。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.litellm.ai/">LiteLLM</a>：LiteLLM 处理 100 多个 LLM 的负载均衡、故障转移和支出跟踪，全部采用 OpenAI 格式。</li><li><a href="https://huggingface.co/Salesforce/SFR-Iterative-DPO-LLaMA-3-8B-R">Salesforce/SFR-Iterative-DPO-LLaMA-3-8B-R · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct">mlabonne/Meta-Llama-3-120B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/fry69/orw">GitHub - fry69/orw: Watch for changes in OpenRouter models API and store changes in a SQLite database. Includes a simple web interface.</a>：监控 OpenRouter 模型 API 的变化并将更改存储在 SQLite 数据库中。包含一个简单的 Web 界面。 - fry69/orw</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>：OpenRouter API Watcher 监控 OpenRouter 模型的变化并将这些更改存储在 SQLite 数据库中。它每小时通过 API 查询模型列表。</li><li><a href="https://orw.karleo.net/removed">OpenRouter API Watcher</a>：OpenRouter API Watcher 监控 OpenRouter 模型的变化并将这些更改存储在 SQLite 数据库中。它每小时通过 API 查询模型列表。
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1239895427336372284)** (41 messages🔥): 

- **Mojo 通过少量语法调整原生运行 MLIR**：成员们讨论了 **Mojo** 如何仅需少量额外语法即可原生运行 **MLIR**。一位用户分享了一个[链接](https://docs.modular.com/mojo/notebooks/BoolMLIR)，解释了 Mojo 访问底层 MLIR 特性的优势。
- **Mojo 将拥有 Python 依赖替代方案**：线程中的对话提出了整个 Mojo 工具链可以在没有 Python 的情况下工作的场景。引用了一个相关的 [GitHub issue](https://github.com/modularml/mojo/issues/935) 来跟踪此功能请求。
- **学习 Mojo 的策略**：寻求学习 Mojo 建议的新成员被引导至 [Mojo SDK manual](https://docs.modular.com/mojo/manual/get-started/) 和其他有用的资源，如 [Mandelbrot notebook](https://docs.modular.com/mojo/notebooks/Mandelbrot)。
- **在 GPU 市场倡导 Mojo**：用户辩论了 **Mojo** 相比 **CUDA** 的可移植性优势，强调 Mojo 的 GPU 代码可移植性可以创造一个更具竞争力的硬件市场。有人指出 CUDA 的供应商锁定目前使 Nvidia 受益，但 Mojo 的跨供应商潜力充满前景。
- **鼓励社区精神**：成员们讨论了社区讨论和倡导对推广 Mojo 的重要性。他们还注意到新编程语言的采用周期很长，建议早期讨论有助于完善和普及 Mojo。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>：获取 Mojo SDK 或在 Mojo Playground 中尝试编码。</li><li><a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>：了解如何使用底层原语在 Mojo 中定义你自己的布尔类型。</li><li><a href="https://docs.modular.com/mojo/notebooks/Mandelbrot">Mandelbrot in Mojo with Python plots | Modular Docs</a>：了解如何编写高性能 Mojo 代码并导入 Python 包。</li><li><a href="https://docs.modular.com/mojo/manual/basics">Introduction to Mojo | Modular Docs</a>：Mojo 基本语言特性介绍。</li><li><a href="https://github.com/modularml/mojo/issues/935">[Feature Request] binary build via `mojo build` could not run directly on other os · Issue #935 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？嗨，我尝试构建一个使用 numpy 的简单 mojo 应用...</li><li><a href="https://modul.ar/community-meeting">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1240005922076758159)** (2 messages):

- **Modular 在 Twitter 上分享了更新**：[Modular 发推](https://twitter.com/Modular/status/1790442405273161922) 了一则更新，推测是关于他们正在进行的项目或与其平台相关的公告。分享的链接中未提供更多细节。
- **分享了另一条 Modular 推文**：[Modular 发布](https://twitter.com/Modular/status/1790774045581152561) 了另一条推文，可能在讨论近期进展或社区新闻。分享的链接中未透露推文的具体内容。
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1239985859370160190)** (3 条消息): 

- **Mojo🔥 Nightly Build 视频发布**：[Modular 的新视频](https://www.youtube.com/watch?v=arZS5-plt2Q) 讨论了 **Mojo🔥 nightly build** 和 nightly Visual Studio Code 扩展。Modular 工程师 Brian Gesiak 介绍了名为 Nightly 的新分支，该分支与 Mojo nightly build 同步。
- **开源 Mojo🔥 标准库贡献**：[Modular 的一段视频](https://www.youtube.com/watch?v=TJpFSSIts5Q) 解释了如何为 **开源 Mojo🔥 标准库** 做出贡献。Modular 工程师 Joe Loser 提供了关于如何开始使用 Mojo 进行贡献的指导。
- **MAX Graph API 和自定义算子介绍**：[Modular 的最新视频](https://www.youtube.com/watch?v=nkWhnFNlguQ) 探讨了用于在 Mojo 中构建 AI 推理流水线的 **MAX Graph API**。Ehsan Kermani 解释了如何开始使用 Mojo 中的 MAX Graph 和自定义算子（custom operators）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=arZS5-plt2Q">使用 Mojo🔥 nightly build 和 nightly Visual Studio Code 扩展</a>：Mojo🔥 公共仓库现在有一个名为 Nightly 的新分支，它与 Mojo nightly build 同步。在这段视频中，Modular 工程师 Brian Gesiak 讨论了...</li><li><a href="https://www.youtube.com/watch?v=TJpFSSIts5Q">为开源 Mojo🔥 标准库做贡献</a>：Mojo🔥 标准库现已开源。在这段视频中，Modular 工程师 Joe Loser 讨论了你如何开始使用 Mojo 为 Mojo🔥 做出贡献...</li><li><a href="https://www.youtube.com/watch?v=nkWhnFNlguQ">MAX Graph API 和自定义算子介绍</a>：MAX Graph API 允许你在 Mojo 中构建整个 AI 推理流水线。在这段视频中，Ehsan Kermani 讨论了你如何开始使用 MAX Graph...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/)** (1 条消息): 

Zapier: Modular: MAX Graph API 教程
https://www.modular.com/blog/max-graph-api-tutorial
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1240007747890708540)** (1 条消息): 

- **加入 Mojo 社区会议！**：Mojo 团队将于 5 月 20 日星期一上午 10-11 点通过 [Zoom](https://modul.ar/community-meeting-zoom) 为开发者、贡献者和用户举办社区会议。会议将分享 Mojo 的未来计划并讨论即将举行的活动；也可以通过此 [链接](https://modul.ar/community-meeting) 将详情添加到你的日历。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom.">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://modul.ar/community-meeting.">Google Calendar - 登录以访问和编辑您的日程</a>：未找到描述</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1239838353806069790)** (120 条消息🔥🔥): 

- **从 Mojo 调用 C/C++ 库**：成员们讨论了从 Mojo 调用 C/C++ 库的可能性。经确认，可以通过使用 ffi 和 external_call [tweetorial](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi) 来实现。

- **添加了字符串转浮点数函数**：一位成员分享说，他们创建了一个 PR，为 `String` 添加了 `atof()` 方法，用于在 Mojo 中将字符串转换为浮点数。该 PR 可以在 [此处](https://github.com/modularml/mojo/pull/2649) 查看。

- **Mojo 与 CLion 的兼容性**：有人询问关于在 CLion 中使用 Mojo 的问题，并分享了所需插件的链接，见 [此处](https://plugins.jetbrains.com/plugin/23371-mojo)。

- **在 Mojo 中创建 HTTP 客户端**：对于创建 HTTP 客户端，建议使用 [lightbug_http](https://github.com/saviorand/lightbug_http) 框架，因为 Mojo 文档目前缺少 HTTP 模块。

- **Python 互操作性问题**：一位成员分享了一个 Python 互操作性问题，该问题在更新到最新版本的 Mojo nightly 后得到解决。初始问题描述及后续解决讨论可以点击[这里](https://discord.com/channels/1087530497313357884/1151418579913277450/1239755238924095579)访问。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/values/value-semantics#">Value semantics | Modular Docs</a>：关于 Mojo 值语义默认设置的解释。</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics#python-style-reference-semantics">Value semantics | Modular Docs</a>：关于 Mojo 值语义默认设置的解释。</li><li><a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs Plugin | Marketplace</a>：为 Mojo 编程语言提供基础编辑功能：语法检查和高亮、注释和格式化。未来将添加新功能，敬请...</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi">devrel-extras/tweetorials/ffi at main · modularml/devrel-extras</a>：包含开发者关系博客文章、视频和研讨会的配套材料 - modularml/devrel-extras</li><li><a href="https://github.com/modularml/mojo/issues/2653">[BUG] NumPy array in-place operation over a copied object reference does not modify the original Python object in-place · Issue #2653 · modularml/mojo</a>：Bug 描述：from python import Python def do_numpy_stuff(ar: PythonObject) -&gt; PythonObject: ar.__iadd__(3) print(&quot;inside function:\n&quot;, ar) return ar fn main() raises: var np = Python....</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>：简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2649">[stdlib] Add method `atof()` to `String`  by fknfilewalker · Pull Request #2649 · modularml/mojo</a>：此 PR 添加了一个可以将 String 转换为 Float64 的函数。目前仅针对 Float64 实现，但也许我们应该添加其他精度？支持以下表示法：&quot;-12...</li><li><a href="https://a.co/d/6dK6Xzl">未找到标题</a>：未找到描述</li><li><a href="https://ivellapillil.github.io/mojo">Learn Mojo Programming Language</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/builtin/tuple.mojo#L100>">mojo/stdlib/src/builtin/tuple.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/carlca/ca_mojo.git">GitHub - carlca/ca_mojo</a>：通过在 GitHub 上创建账号来为 carlca/ca_mojo 的开发做出贡献。</li><li><a href="https://github.com/dimitrilw/toybox/issues/9>">Issues · dimitrilw/toybox</a>：在 Mojo🔥 中实现的各种数据结构和其他玩具。- Issues · dimitrilw/toybox
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1239996921435918576)** (17 条消息🔥): 

- **Modular 机器人通知正在开发中**：成员们讨论了让 Modular 机器人在此 nightly 版本发布时通知他们的想法。一位开发者暗示这可能会通过 GitHub Actions 来组织。

- **Nightly 构建与每周更新**：有人建议 nightly 构建仅包含非编译器更改，并将编译器更改的更新保留给每周构建。目前尚未对此达成进一步共识。

- **新的 Nightly Mojo 编译器已发布**：最新的 nightly 版本 `2024.5.1515` 已可用，可以通过 `modular update nightly/mojo` 进行更新。[自上次发布以来的更改链接](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **macOS 上潜在的自检失败**：由于 LLDB 初始化问题，今天的 nightly 版本在 Mac 上可能会出现非确定性的自检（self-test）失败，目前正在调查中。用户已获悉该持续存在的问题。

- **重点提交回顾**：分享了两个值得关注的提交：一个是使 `Tuple` 的构造函数移动其输入元素，另一个是将 `Reference.is_mutable` 更改为 `Bool`。[Tuple 构造函数更改](https://github.com/modularml/mojo/commit/f05749db22548f61c17e6cb0db938e22f092341f) 和 [Reference.is_mutable 更改](https://github.com/modularml/mojo/commit/09db8f3bcb0d783078a63241cda57047a850e05e)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/commit/f05749db22548f61c17e6cb0db938e22f092341f">[mojo-stdlib] 使 `Tuple` 的构造函数移动其输入元素。 (#3… · modularml/mojo@f05749d</a>: …9904) 此更改使 `Tuple` 将其输入 pack 视为 'owned'，然后从该 pack 移动到其存储中。这发现了一些处理 owned packs 时的 bug，这些 bug 导致了多个 d...</li><li><a href="https://github.com/modularml/mojo/commit/09db8f3bcb0d783078a63241cda57047a850e05e">[stdlib] 将 `Reference.is_mutable` 更改为 `Bool`（原为 `i1`） · modularml/mojo@09db8f3</a>: 随着最近将 `Bool` 更改为使用 `i1` 作为其表示形式，许多阻碍将 `Reference.is_mutable` 移至 `Bool` 的错误已得到解决。 Co-authored-by: Chris Lattner &amp;lt;clatt...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1239968856722636962)** (23 messages🔥): 

- **Austin 介绍了自己，他来自 AWS GenAI Innovation Center**：Austin 加入聊天以探索开源研究机会，并受到了热烈欢迎，成员建议他查看[其他频道](https://discord.com/channels/747850033994662000)以获取更多资源。
- **epinet 与 JARVIS 角色的比较**：成员们辩论了为什么相比于 "JARVIS"，"Her" 中的角色对于 AGI 开发更具吸引力。一位成员指出：*"大众市场想和 'HER' 发生关系多过想和 JARVIS 发生关系。"*
- **关于 epinet 有效性和局限性的讨论**：小组探讨了 epinets 作为模拟集成模型以估计认知不确定性（epistemic uncertainty）的方法的残余效应和实用性。尽管喜欢 epinets，但有人指出：*"它不像其他确定不确定性的 Bayesian 方法那样具有具体的理论基础。"*
- **对带有 epinets 的神经网络稳定性的担忧**：针对在原始模型中添加 epinet 时神经网络可能出现的棘手问题提供了详细解释，这可能导致不准确的不确定性预测。*"预测的方差可能很低，因为模型确定它所做的是正确的……尽管模型不确定某事是否正确，方差却很高。"*
- **关于残差及其对 epinets 影响的困惑**：对话包括对 epinet 是否需要残差（residuals）以更有效地调整基础模型输出的推测。一位成员建议：*"从这个意义上说，残差只是让训练变得更容易。"*
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1239940002998255666)** (79 messages🔥🔥): 

- **关于模型收敛的激活函数的讨论**：成员们辩论了激活函数确保良好收敛的要求，强调其需要是非线性的。一位成员分享了关于在预训练模型中对两个函数进行参数化而不出错的担忧。

- **围绕 FlashInfer 和 FA2 的微调对话**：随后对 FlashInfer 和 FA2 等 AI 模型中的不同方法论进行了详细讨论。针对跨 K 序列长度的拆分以及 FA2 中包含的 reduction 步骤提供了澄清。

- **对神经网络中点积的见解**：成员们分享了对神经网络中点积的简化和鲁棒性的看法，其中一人链接到一篇[文章](https://archive.is/GfliU)，以扩展傅里叶变换及其认知失调的影响。

- **Sakuga-42M 数据集介绍**：介绍了一个名为 Sakuga-42M 的新型大规模卡通动画数据集，包含 4200 万个关键帧，旨在解决在自然视频上训练的模型中存在的偏差。分享了相应的 [arXiv 链接](https://arxiv.org/abs/2405.07425)以供进一步阅读。

- **视觉问答与专用模型**：成员们讨论了视觉问答（VQA）模型和 BLIP3 等多模态模型的局限性和应用，并参考了 [Hugging Face](https://huggingface.co/tasks/visual-question-answering) 上的资源，用于辅助视障人士和图像检索等特定用例。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07518">SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts</a>: 像 GPT-4 这样的单体大语言模型（LLMs）为现代生成式 AI 应用铺平了道路。然而，大规模训练、部署和维护单体 LLMs 仍然面临极高的成本……</li><li><a href="https://arxiv.org/abs/2405.08707">Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory</a>: 增加 Transformer 模型的大小并不总是能提升性能。这种现象无法用经验性的 Scaling Laws 来解释。此外，改进的泛化能力……</li><li><a href="https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1">Salesforce/xgen-mm-phi3-mini-instruct-r-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.07425">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: 手绘卡通动画利用草图和色块来创造运动的错觉。虽然 CLIP、SVD 和 Sora 等近期进展在理解和……方面展示了令人印象深刻的结果。</li><li><a href="https://arxiv.org/abs/2405.08644">Thinking Tokens for Language Modeling</a>: 56 乘以 37 等于多少？语言模型在这些类型的困难计算中经常出错。这通常被解释为它们无法进行复杂的推理。由于语言模型……</li><li><a href="https://huggingface.co/tasks/visual-question-answering">What is Visual Question Answering? - Hugging Face</a>: 未找到描述</li><li><a href="http://arxiv.org/abs/2405.08553">Improving Transformers with Dynamically Composable Multi-Head Attention</a>: 多头注意力（MHA）是 Transformer 的核心组件。在 MHA 中，注意力头独立工作，导致了诸如注意力评分矩阵的低秩瓶颈和头部冗余等问题。……</li><li><a href="https://github.com/caiyun-ai/dcformer">GitHub - Caiyun-AI/DCFormer</a>: 通过在 GitHub 上创建账号来为 Caiyun-AI/DCFormer 的开发做出贡献。</li><li><a href="http://arxiv.org/abs/2112.00114">Show Your Work: Scratchpads for Intermediate Computation with Language Models</a>: 大型预训练语言模型在可以“一次性完成”的任务上表现出色，例如生成逼真的文本或合成计算机程序。然而，它们在……方面表现不佳。</li><li><a href="http://arxiv.org/abs/2305.00833">Learning to Reason and Memorize with Self-Notes</a>: 大语言模型已被证明在多步推理方面存在困难，并且不会保留之前的推理步骤供将来使用。我们提出了一种简单的方法来解决这两个问题，通过……</li><li><a href="http://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。尽管以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理对于……是重要的。</li><li><a href="https://archive.is/GfliU">The Power of the Dot Product in Artificial Intelligence | by Manuel B&#x2026;</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239948993925222444)** (45 条消息🔥):

- **模拟初始化（Mimetic initialization）可提升 Transformer 训练效果**：一位成员分享了一篇[论文](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf)，研究表明模仿预训练 Transformer 的权重模式（被称为“模拟初始化”）可以显著提高在 CIFAR-10 和 ImageNet 等小数据集上的训练效率和准确率，准确率分别提升了 5% 和 4% 以上。这涉及到将 query 和 key 的权重初始化为接近单位矩阵，将 value 和 projection 的权重初始化为接近负单位矩阵。
- **关于使用 Hypernetworks 进行初始化的讨论**：在利用 Hypernetworks 进行权重初始化的对话中，结论是虽然这本质上是元学习（meta-learning），但寻找一种低维、简单的符号初始化可能会大有裨益。一位成员思考通过符号回归（symbolic regression）来推导和逆向工程新的初始化原理。
- **对 Minsky 对神经网络影响的反思**：对话中表达了一种观点，即 Marvin Minsky 在神经网络早期失败期间，因阻碍了人们对神经网络的兴趣而获得了过多的“功劳”，尽管他本人也有神经网络背景。有一种论点认为，如果神经网络从一开始就取得成功，Minsky 的影响就不会如此显著。
- **小数据集训练的挑战**：成员们分享了在小数据集上训练 Transformer 的实际困难，一些成员表示此前并不知道这是一个问题，这暗示了心理上的“无知”可能会降低解决问题的门槛。
- **想法分享与社区项目提议**：成员们讨论了想法很多但缺乏时间去实现的问题。有人呼吁为社区驱动的项目创建一个“idea-dump”频道，这反映了人力有限但热情高涨的普遍瓶颈。

**提到的链接**：<a href="https://arxiv.org/abs/2210.03651">Understanding the Covariance Structure of Convolutional Filters</a>：神经网络权重通常从单变量分布中随机初始化，即使在像卷积这样高度结构化的操作中，也仅控制单个权重的方差。Re...

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

ocg6377: 我也可能有兴趣提供帮助，具体取决于需要什么。

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1240087798841606187)** (5 messages): 

- **Harness 对每个 Token 调用处理多选题**：成员们讨论了 MMLU 中的单个多选题如何导致每个答案（A, B, C, D）产生一个请求，即使它们只是一个 token。一位成员确认，正如 [harness 代码](https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485)所示，harness 确实通过单次调用处理每个答案选项。
- **导出多选题答案以进行准确率分析**：一位用户询问如何从多选题中导出单个答案，以辨别模型的回答正确与否。这将有助于比较正确/错误回答的分布。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485)">lm-evaluation-harness/lm_eval/models/utils.py at a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

---

**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1240176018799464478)** (3 messages): 

- **Thunder Kittens 创作者会议请求**：一位成员询问是否可以让 **Thunder Kittens 创作者**在 **CUDA MODE** 中进行演示。另一位成员给出了积极回应，说：“我可以去问问。”

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1239868744906575992)** (19 messages🔥): 

- **CuSPARSE 向量重用机制不明确**：一位成员对 `cusparseCreateDnVec` 调用的开销和可重用性提出疑问，对文档中缺乏内存分配细节表示困惑。他们询问是仅影响描述符内存，还是向量数据被缓存到了其他地方。

- **clangd 与 CUDA 文件配合出现问题**：尽管有 `compile_commands.json` 文件并使用了 VSCode 和 Neovim，一位成员在让 clangd 正确解析 `.cu` 文件时仍遇到困难。他们指出之前使用 CUDA Toolkit 时可以正常工作，但在切换到 NVHPC 后出现了问题。

- **Torch Tensor Accessor 讨论**：一位成员就应当按照文档使用 C++ 中 Torch tensor 的 accessor，还是直接将 `tensor.data_ptr` 传递给 kernel 寻求建议，并提出了关于在这些 tensor 中使用 unsigned char 指针的问题。他们请求获取更多关于该主题的文档。

- **点积谜题浮点数问题**：一位成员在大型数组点积的朴素实现中遇到了浮点数溢出错误，其结果与使用 reduction 方法时不同。另一位成员解释说，该问题与 FP32 精度限制有关，并建议合并他们的 kernel 代码以获得更好的准确性。

- **提高 L2 Cache 命中率**：有人提出了关于提高 L2 Cache 命中率的建议，并引用了 Triton 讲座和 [Triton 矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。该教程涵盖了块级矩阵乘法、指针算术以及为了获得更好 cache 性能的程序重排序。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html">Matrix Multiplication &mdash; Triton  documentation</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/cccl/blob/main/.clangd">cccl/.clangd at main · NVIDIA/cccl</a>：CUDA C++ 核心库。通过在 GitHub 上创建账号为 NVIDIA/cccl 的开发做出贡献。</li><li><a href="https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product).">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>：解决谜题。学习 CUDA。通过在 GitHub 上创建账号为 srush/GPU-Puzzles 的开发做出贡献。</li><li><a href="https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8">puzzle10_dotproduct floating point overflow error</a>：puzzle10_dotproduct 浮点数溢出错误。GitHub Gist：立即分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1239840938080342057)** (39 条消息🔥): 

- **使用 `torch.compile` 时检查你的 tensor 分配**：一位成员建议将 `torch.cat` 实现中常见的动态分配 tensor 替换为预分配 tensor，以提高性能（[示例在此](https://github.com/openai/whisper/blob/main/whisper/model.py#L301)）。
  
- **在 Triton kernel 中使用 `torch.compile`**：一位成员询问了关于使用 Triton kernel 创建网络图的建议。建议是“创建一个 custom op 并使用 torch.compile 包装它们”，并链接了一个详细讨论以获取进一步帮助。

- **处理 DeepSpeed 兼容性问题**：一位用户询问 DeepSpeed 的最新版本是否与 `torch.compile` 兼容。一位成员引用了 [GitHub 上的 PR](https://github.com/microsoft/DeepSpeed/pull/4878)，强调应当在 DeepSpeed 配置中添加一个编译标志（compile flag）。

- **使用 `torch.compile` 优化自定义操作**：针对在自定义操作和外部库中使用 `torch.compile` 的优势和挑战展开了详细讨论。建议包括使用 `torch._dynamo` 装饰器来正确追踪（trace）和调试编译过程中的 tensor 问题（[示例项目](https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/bindings.py)）。

- **`torch.compile` 中的静态 vs 动态 tensor 分配**：一位成员解释了为什么动态分配 tensor 会损害性能，特别是在使用 `torch.cat` 时，因为重新分配和复制 cache 的开销很大。强调了静态 cache 的重要性以及减少开销的技术（[博客链接在此](https://pytorch.org/blog/accelerating-generative-ai-2/)）。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-generative-ai-2/">Accelerating Generative AI with PyTorch II: GPT, Fast</a>: 这篇文章是一个系列博客的第二部分，重点介绍如何使用纯原生 PyTorch 加速生成式 AI 模型。我们很高兴能分享一系列新发布的 PyTorch 性能...</li><li><a href="https://pytorch.org/blog/introducing-depyf">Introducing depyf: mastering torch.compile with ease</a>: 介绍 depyf：轻松掌握 torch.compile</li><li><a href="https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp#L218">cuda-sample/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp at master · zchee/cuda-sample</a>: CUDA 官方示例代码。通过在 GitHub 上创建账号为 zchee/cuda-sample 的开发做出贡献。</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/bindings.py?ref_type=heads#L50">src/python/bindings.py · v043 · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto 版本控制系统</li><li><a href="https://github.com/openai/whisper/blob/main/whisper/model.py#L301)">whisper/whisper/model.py at main · openai/whisper</a>: 通过大规模弱监督实现鲁棒的语音识别 - openai/whisper</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/4878">Enable torch.compile with ZeRO (Experimental) by tohtana · Pull Request #4878 · microsoft/DeepSpeed</a>: 此 PR 支持在 ZeRO 阶段 1/2/3 中使用 torch.compile。你需要在 DeepSpeed 配置中添加 compile 部分。该部分中的字段将传递给 torch.compile。   &quot;compile&quot;: {     &quo...</li><li><a href="https://pastebin.com/XHwFwDLx">compile problem - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/jobs/86868">manylinux-cu121: [cp310, 2.3] (#86868) · Jobs · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto 版本控制系统</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/ops.py?ref_type=heads#L41">src/python/ops.py · v043 · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto 版本控制系统</li><li><a href="https://github.com/pytorch/ao/pull/184/files#diff-3444226e1dc5947e486c918c8d57b8742bbcd9af6b4f5a599e0443b08bd7164aR222">[wip] fast semi-sparse sparse training  by jcaip · Pull Request #184 · pytorch/ao</a>: 在 HuggingFace BERT 上测试时没有看到加速——因为我被其他一堆东西卡住了（bf16, compile, adamw, dataloader, batchsize）。bf16 + compil...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1239840680256213022)** (3 messages): 

- **初学者寻求在 PyTorch 中使用自定义 CUDA kernels 的建议**：一位初学者请求获取学习如何在 **PyTorch** 中使用自定义 **CUDA kernels** 的资源。
- **热心回复提供了讲座链接**：一位成员分享了 [Jeremy 的 YouTube 讲座](https://youtu.be/4sgKnKbR-WE) 链接，标题为 *“Lecture 3: Getting Started With CUDA for Python Programmers”*，该讲座演示了如何编写自定义 CUDA kernels 并在 PyTorch 中使用它们。

**提到的链接**：<a href="https://youtu.be/4sgKnKbR-WE?si=00-k8KV5ESxqks3h">Lecture 3: Getting Started With CUDA for Python Programmers</a>：Jeremy 的 YouTube 录像 https://www.youtube.com/watch?v=nOxKexn3iBo 补充内容：https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1239966431060295712)** (4 messages): 

- **移动端问题导致用户困惑**：一位用户提到在移动端遇到“奇怪的问题”，但表示现在“看起来正常了”。另一位用户报告称该问题在 PC 和移动端均会出现，确认问题尚未解决。
- **活动链接故障排除继续**：用户讨论了一个指向 Discord 活动的超链接，以及通过“活动选项卡 (events tab)”访问活动是否能解决问题。经检查，一位用户确认通过活动选项卡访问链接是有效的。
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1240353611956097074)** (3 messages): 

- **查看 Twitter 上的 cloud**：一位成员分享了 [X 上的 cloud (@cloud11665) 链接](https://twitter.com/cloud11665/status/1790776040681271583)。该链接在消息中提供，没有太多上下文。
- **NVIDIA GPU 编程指南**：另一位成员链接到了 [NVIDIA GPU Programming Guide](https://download.nvidia.com/developer/GPU_Programming_Guide/GPU_Programming_Guide.pdf)，为那些对 GPU 编程感兴趣的人提供了资源。
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1240324433667686453)** (1 messages):

- **CUDA Puzzle 10 的解答与问题**：一位成员分享了他们使用朴素方法和归约（reduction）方法解决 [CUDA Puzzle 10 - Dot Product](https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product) 的方案。他们在处理大小为 **20480000** 的数组时，朴素实现在输出为 `16777216` 而非 `20480000` 时遇到了浮点数溢出错误，而归约方法则运行正常。他们正在寻求对此行为的解释。你可以在[这里](https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8)查看他们的代码。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product).">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>：解决谜题，学习 CUDA。通过在 GitHub 上创建账号为 srush/GPU-Puzzles 的开发做出贡献。</li><li><a href="https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8">puzzle10_dotproduct floating point overflow error</a>：puzzle10_dotproduct 浮点数溢出错误。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1239877561010618419)** (65 条消息🔥🔥): 

- **梯度累积与 `layernorm_backward_kernel8` 修复**：关于重新审视梯度累积和验证 Kernel 计算的讨论发现，`layernorm_backward_kernel8` 存在未累积偏置（biases）和权重（weights）的问题。通过建议更新以确保正确的梯度调整解决了此问题（[PR #408](https://github.com/karpathy/llm.c/pull/408)）。

- **CUDA 流同步 Bug 排查**：调试工作发现，CUDA 流（streams）和事件（events）的顺序错误可能导致同步问题，从而影响梯度累积和 GPU 操作。值得注意的是，使用 `parallel_streams[0]` 和默认流同步行为是否会导致潜在的竞态条件（race conditions）受到了质疑。

- **提议的修复与简化**：几位成员建议重置与 CUDA 流相关的代码，并从更简单、更受控的方法开始。提议包括通过向 Kernel 启动器传递流参数来使流管理更加显式，并从头重做梯度累积逻辑（[PR #417](https://github.com/karpathy/llm.c/pull/417)）。

- **关于测试容差的辩论**：团队讨论了梯度检查中的相对容差和绝对容差，并将当前实践与 NumPy 的 `assert_allclose` 等标准进行了比较。大家一致认为需要合理的容差参数，且该参数应随被比较值的量级而缩放。

- **使用 Stream Guards 维持并行性**：确保流依赖性的想法包括使用 CUDA 事件回调以及为 `cpu_losses` 等元素创建保护机制（guard mechanism）。尽管取得了一些进展，团队成员承认工作量巨大，完全解决这些并行性问题可能会有延迟。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html">CUDA Runtime API :: CUDA Toolkit Documentation</a>：未找到描述</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose">numpy.testing.assert_allclose &#8212; NumPy v1.26 Manual</a>：未找到描述</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.as">numpy.testing.assert_allclose &#8212; NumPy v1.26 Manual</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/412">[wip] gradient accumulation, another attempt by karpathy · Pull Request #412 · karpathy/llm.c</a>：无法运行。在 master 分支上，我们通过运行以下命令（几乎）精确地复现了 Python 脚本：make train_gpt2cu NO_MULTI_GPU=1 USE_CUDNN=1 ./train_gpt2cu -b 4 -t 64 -l 1e-4 -v 200 -s 200 -a 1 -x 10 但是...</li><li><a href="https://github.com/karpathy/llm.c/pull/408">Layernorm backward updates by ngc92 · Pull Request #408 · karpathy/llm.c</a>：这修复了 layernorm 反向传播的梯度累积，并对 layernorm backward dev/cuda 文件进行了通用的现代化改造。容差已适应 float 暂存器...</li><li><a href="https://github.com/karpathy/llm.c/pull/417">Remove parallel CUDA streams while keeping main_stream and loss_event(?) by ademeure · Pull Request #417 · karpathy/llm.c</a>：参见 Discord 上的讨论，我认为无论我们最终构建出什么比我最初的尝试更好的架构，可能仍然需要类似于 "main_stream" 的默认流...
</li>
</ul>

</div>

**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1240008727445242017)** (4 messages): 

- **使用 llamafile 运行本地 LLM**：Mozilla 推出的新 llamafile 让你能轻松在笔记本电脑上构建本地、私密的调研助手。只需下载文件并运行，即可直接从 LlamaIndex 使用本地 LLM 和 embedding 模型。[来源](https://t.co/qFIA6j1OWe) 

- **Navarasa 在 Google I/O 大放异彩**：向 Navarasa 的共同创作者 @ravithejads 致敬，他的作品在 Google I/O 上展出。Navarasa 是 Google Gemma 模型的微调版本，支持 15 种印度语言。[视频](https://t.co/zc00GjOmc4)

- **LlamaIndex 与 Vertex AI 合作**：LlamaIndex 现已上线 Google Cloud 的 Vertex AI，推出了由高级模块驱动的新 RAG API，用于端到端的索引（indexing）、嵌入（embedding）、检索（retrieval）和生成（generation）。此次合作旨在简化复杂的集成和流程。[来源](https://t.co/ekAQ24hNWr)

- **在 create-llama 中使用 GPT-4o 构建聊天机器人**：现在，只需通过 create-llama 回答几个问题，即可完成使用 GPT-4o 构建聊天机器人 90% 的工作。此次更新显著简化了聊天机器人的创建过程。[来源](https://t.co/wtcaWdrB7H)


  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1239847969101185134)** (130 messages🔥🔥): 

- **关于“从小到大检索”（Small to Big Retrieval）的辩论**：一位成员对“从小到大检索”的工作原理表示好奇，质疑在不同分块大小（chunk sizes）中包含相同信息是否冗余。另一位成员澄清说，在检索过程中，“实际上只检索底层分块”，并在必要时向上合并形成更大的分块。
  
- **将 sec-insights 升级到新版 LlamaIndex**：一位用户询问将 **sec-insights repo** 从 **llamaindex 0.9.7** 升级到新版本的难度。回复各不相同，有的承诺提供帮助，有的建议这可能只是更新 import 语句的问题。

- **Meta-Llama 与 Ollama 的性能问题**：一位成员报告了在解析 JSON 对象时，来自 Hugging Face 的 **Meta-Llama 3-8B** 与 **Ollama** 模型之间的差异。“4-bit 量化”的 Ollama 模型失败了，而 Meta-Llama 处理得非常完美，这引发了对模型量化（quantization）的担忧。

- **在 LlamaIndex 中处理 GPT-4o**：讨论了在 **LlamaIndex** 中使用 **GPT-4o** 的情况，一位成员确认从发布第一天起就已支持。他们分享了成功使用 **GPT-4o** 的代码片段。

- **对 LlamaParse 安全性的担忧**：多位用户对 **LlamaParse** 的安全性和数据保留政策表示担忧。团队澄清说，数据会为了缓存而暂存 48 小时，之后会被删除，并且他们还为注重安全性的客户提供本地部署（on-premise）模式。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - Hugging Face 空间</a>: 暂无描述</li><li><a href="https://www.llamaindex.ai/contact">联系我们 — LlamaIndex，LLM 应用的数据框架</a>: 如果您对 LlamaIndex 有任何疑问，请联系我们，我们将尽快安排通话。</li><li><a href="https://www.youtube.com/watch?v=q7_PLPmQDnc&t=3s">房地产中的人工智能？ 🏘️😱</a>: 暂无描述</li><li><a href="https://www.koyeb.com/tutorials/using-llamaindex-and-mongodb-to-build-a-job-search-assistant">使用 LlamaIndex 和 MongoDB 构建职位搜索助手</a>: 了解如何使用 LlamaIndex 通过检索增强生成 (RAG) 和 MongoDB 构建职位搜索助手。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/document_management_pipeline#ingestion-pipeline-document-management>)">Ingestion Pipeline + 文档管理 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/ingestion_pipeline#document-management>)">Ingestion Pipeline - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_documents#customizing-the-id>)">使用 Documents - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/loading/loading#adding-metadata>).">加载数据 (Ingestion) - LlamaIndex</a>: 暂无描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 messages): 

pier1337: 五月份 RAG 的最前沿技术（SOTA）是什么？
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1239836028915683380)** (25 messages🔥):

- **Nathan Lambert 分享对 OpenAI 优先事项的看法**：Nathan Lambert 参与了 Twitter 上的讨论，表示“OpenAI 是一家拥有强大研究团队的产品公司，用户是最重要的”。[查看推文](https://twitter.com/natolambert/status/1790393805633712246)。

- **Google IO 的生成式视频令人兴奋**：Nathan Lambert 对“google io is good”和“let's gooooo generative video”表达了热情。尽管错过了一些公告（如 Gemini 1.5 Ultra），但他称赞这次活动不是一场“我的模型比你的大”的竞赛。

- **Google 在 Google I/O 2024 上发布 Gemma 2**：Google 揭晓了 [Gemma 2](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/)，这是一个拥有 270 亿参数的模型，同时发布的还有用于视觉语言任务的 PaliGemma。详情通过一篇 [TechCrunch 文章](https://techcrunch.com/2024/05/14/google-i-o-2024-everything-announced-so-far/)分享。

- **Google 的 Gemini 1.5 和 AI Studio 更新**：Google 更新了其 AI 套件，在 AI Studio 中提供了 [Gemini 1.5 Pro 和 Flash](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/)。公告中提到了 API 密钥的试用以及帮助开发者入门的 API Cookbook。

- **AI Studio 的定价和区域可用性**：一位成员强调 Google 的 AI Studio 现已在英国可用，并可能进入 EEA。他们还根据基准测试指出，与 Pro 相比，Flash 服务的性价比更高。[查看定价](https://ai.google.dev/pricing)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/">Gemini 1.5 Pro 更新、1.5 Flash 亮相以及 2 个新的 Gemma 模型</a>：今天我们正在更新 Gemini 1.5 Pro，推出 1.5 Flash，发布新的 Gemini API 功能并增加两个新的 Gemma 模型。</li><li><a href="https://ai.google.dev/pricing">未找到标题</a>：未找到描述词</li><li><a href="https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/">Google 发布 Gemma 2，这是其开源模型的 27B 参数版本，将于 6 月推出 | TechCrunch</a>：在 Google I/O 大会上，Google 推出了 Gemma 2，这是 Google 下一代 Gemma 模型，将于 6 月推出 270 亿参数版本。
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1240206213107548170)** (10 条消息🔥): 

- **跨学科神经网络向现实收敛**：*“在不同数据和模态上以不同目标训练的神经网络，正在其表示空间中收敛到一个共享的现实统计模型。”* 引用了一篇 [论文](https://phillipi.github.io/prh/) 和一篇 [arXiv 文章](https://arxiv.org/abs/2405.07987) 来探讨这一现象。
- **文献综述强调模型收敛**：[Phillip Isola](https://x.com/phillip_isola/status/1790488967827108304?s=46) 描述了随着 LLM 变得更大更强，它们学到的表示越来越类似于视觉模型学习到的表示，反之亦然。这种收敛在过去的研究和新证据中都得到了强调。

**提到的链接**：<a href="https://x.com/phillip_isola/status/1790488967827108304?s=46">来自 Phillip Isola (@phillip_isola) 的推文</a>：我们调查了文献中的证据，然后提供了几个“新”结果，包括：随着 LLM 变得更大更强，它们学习到的表示与视觉模型学习到的表示越来越相似...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1240097783566827600)** (42 条消息🔥):

- **Tokenizer 变更之谜**：一位成员质疑更改 LLM 的 **tokenizer** 是否需要重新训练，并对 **OpenAI** 是否重新进行 pretraining 的争议表示困惑。他们讨论了**扩展 tokenizer** 的可能性更大，尽管这可能会带来挑战。
- **OpenAI 模型推测**：成员们推测 **OpenAI** 是否预训练了多个模型以通过排名衡量用户偏好，或者他们是否重新训练了现有模型。他们讨论了这种可能性，一致认为 OpenAI 依赖公共排名而非内部评估似乎效率低下。
- **零样本 Tokenizer 迁移 (ZeTT)**：一位成员分享了一篇关于 **Zero-Shot Tokenizer Transfer (ZeTT)** 概念的论文，该技术允许 LM 在不损失性能的情况下切换 tokenizer。[论文链接](https://arxiv.org/abs/2405.07883v1) 讨论了为此目的使用 **hypernetwork**，引发了关注并因其技术意义获得认可。
- **OpenAI 的 Tokenization 策略**：成员们想知道 OpenAI 是否在没有新标识符的情况下从头开始训练模型，以及每个模态的 tokenization 是伪造的还是真实的。他们思考了各模态是否共享 tokenizer 以及不同 tokenization 策略的影响。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/bminixhofer/status/1790267652587258343?s=46">Benjamin Minixhofer (@bminixhofer) 的推文</a>：介绍 Zero-Shot Tokenizer Transfer (ZeTT) ⚡ ZeTT 将语言模型从其 tokenizer 中解放出来，允许你在几乎不需要额外训练的情况下，将任何模型与任何 tokenizer 配合使用。非常激动...</li><li><a href="https://arxiv.org/abs/2405.07883v1#page11">Zero-Shot Tokenizer Transfer</a>：语言模型 (LMs) 受限于其 tokenizer，后者将原始文本映射为词汇表项（tokens）序列。这限制了它们的灵活性：例如，主要在英语上训练的 LMs 可能...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)** (24 messages🔥): 

- **评估瓶颈阻碍开放获取**：一位成员分享了一篇 [博客文章](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation)，强调了当前语言模型评估中的问题。他们指出了学术基准（如 [MMLU](https://arxiv.org/abs/2009.03300)）和私有 A/B testing 的主导地位，并指出评估需要更广泛的可访问性。

- **Anthropic 寻求产品转型**：Xeophon 链接到一篇 [Anthropic 新闻文章](https://www.anthropic.com/news/mike-krieger-joins-anthropic)，宣布转向成为一家产品公司。他们讨论了由于竞争压力，行业正从 API 服务转型。

- **OpenAI 招聘引发推测**：Xeophon 提到 OpenAI 聘请了一位前 Google 高管，这与其开发搜索引擎以挑战 Google 的计划有关，并引用了一条 [推文](https://x.com/theinformation/status/1790467870545027186?s=46)。此举引发了关于 OpenAI 市场策略和潜在 IPO 的推测。

- **AI 领域的持久护城河需要产品**：DN123456789 认为 AI 公司需要提供产品以保持竞争优势。他们将此与计算机行业的发展进行了类比，强调了拥有终端用户产品比单纯提供硬件或 API 服务更重要。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=1fmcdz2EO_c">2025 年的模型将更像同事而非搜索引擎 – OpenAI 联合创始人 John Schulman</a>：完整剧集：https://youtu.be/Wo95ob_s_NI Apple Podcasts：https://podcasts.apple.com/us/podcast/john-schulman-openai-cofounder-reasoning-rlhf-plan/id15160933...</li><li><a href="https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation">ChatBotArena：大众的 LLM 评估、评估的未来、评估的激励机制以及 gpt2chatbot</a>：细节告诉了我们关于目前最流行的 LLM 评估工具以及该领域其他工具的哪些信息。</li><li><a href="https://x.com/theinformation/status/1790467870545027186?s=46">The Information (@theinformation) 的推文</a>：OpenAI 聘请了在 Google 工作 21 年的资深人士 Shivakumar Venkataraman，他此前领导该公司的搜索广告业务。此举正值 OpenAI 开发搜索引擎与 Google 竞争之际...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1240288331674423336)** (3 messages):

- **对 OpenAI 的复杂情感**：一位用户指出了近期帖子的反差，从“赞扬 OpenAI 的技术领导地位转变为全面吐槽其文化表现”。他们认为这种情绪转变非常经典。
- **对帖子的正面反馈**：另一位成员在回应持续的讨论时简单评论道：“好帖”。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1239848641640796190)** (73 messages🔥🔥): 

- **LangChain Agent 响应缓慢**：一位用户抱怨他们的 LangChain Agent 在调用工具和处理大量输入时耗时过长（2-3 分钟），寻求他人的解决方案。
  
- **使用 SocketIO 流式传输 LLM 响应**：几位成员讨论了使用 `python-socketio` 将 LLM 响应流式传输到前端，分享了详细的代码示例并引用了[相关的 GitHub issues](https://github.com/langchain-ai/langchain/issues/4118)。

- **自主 AI Agent 活动**：来自 Olas / Autonolas 的成员邀请 LangChain 演讲者参加由 NEAR 和 Celo 主办的讨论 AI Agent 在 web3 中作用的活动。

- **向量数据库 Embedding 迁移**：成员们讨论了在 pgvector 和 Qdrant 等向量数据库之间迁移 Embedding 的方法，并探索了如并行化和 Matryoshka Embeddings 等优化检索速度的策略，并附带了 [Supabase 博客](https://supabase.com/blog/matryoshka-embeddings)链接。

- **关于已弃用的 LLMChain 的担忧**：成员们澄清了关于 `LLMChain` 弃用的困惑，以及如何使用 `RunnableSequence` 代替 `MultiQueryRetriever`。他们指出 `MultiQueryRetriever` 可能尚未反映最新的 API 更改。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://supabase.com/blog/matryoshka-embeddings">Matryoshka embeddings: faster OpenAI vector search using Adaptive Retrieval</a>: 使用 Adaptive Retrieval 提升 OpenAI 新 Embedding 模型的查询性能。</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/vectorstores/pgembedding/">Postgres Embedding | 🦜️🔗 LangChain</a>: Postgres Embedding 是 Postgres 的开源向量相似度搜索，使用 Hierarchical Navigable Small Worlds (HNSW) 进行近似最近邻搜索。</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/">MultiQueryRetriever | 🦜️🔗 LangChain</a>: 基于距离的向量数据库检索将查询嵌入（表示）在高维空间中，并根据“距离”查找相似的嵌入文档。但是，检索可能会产生不同的...</li><li><a href="https://github.com/langchain-ai/langchain/issues/21658">DOC: Jsonloader uses jq schema to parse Json files which cannot be installed on windows 11 · Issue #21658 · langchain-ai/langchain</a>: Checklist 我为此 issue 添加了一个非常详细的标题。我包含了指向我所指文档页面的链接（如果适用）。当前文档的问题：文档：https://python....</li><li><a href="https://github.com/langchain-ai/langchain/issues/4118>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 开发做贡献。</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#langchain.chains.llm.LLMChain">langchain.chains.llm.LLMChain &mdash; 🦜🔗 LangChain 0.2.0rc2</a>: 未找到描述</li><li><a href="https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html">langchain.retrievers.multi_query.MultiQueryRetriever &mdash; 🦜🔗 LangChain 0.2.0rc2</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1240228553895968848)** (2 messages): 

- **LangServe 面临速率限制问题**：通过 LangSmith 部署部分部署的托管 LangServe 在访问带有 "/docs" 的服务器 URL 时遇到 "rate exceeded" 错误，导致工作流中断。用户好奇升级到 Pro 计划是否能解决此问题，以及是否可以查看除构建日志之外的已部署 Revision 日志。
  
- **服务器不活跃影响稳定性**：服务器断断续续进入睡眠模式或变得不活跃，影响了服务的持续使用。用户寻求原因分析及可能的解决方案。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1239915041927725077)** (2 messages):

- **基于 LangChain 的 Snowflake 成本监控器**：一位成员分享了使用 LangChain, Snowflake Cortex 和 OpenAI 构建的 **Snowflake 成本监控与优化工具**。可以查看此 [Loom 视频](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064) 中的演示，该工具具有 AI 选择的数据可视化功能，并指出目前仍在开发中。
- **集成 JVM 用于微支付**：一位用户描述了利用 py4j 库从 Langserve 后端与 JVM 中的 JAR 进行交互。该设置旨在进行 **加密 SDK 交互**，以便为 prompt/response 的 token 计数实现微支付，其中包括在 OpenAI API 密钥对之上设置的可调节利润空间。

**提及的链接**：<a href="https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064">Crystal Cost Demo</a>：在此视频中，我简要演示了 Crystal Cost，这是一个 AI 驱动的 Streamlit 应用，可简化数据仓库上的数据监控。Crystal Cost 使用自然语言处理和 Agent 来查询数据...

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1239933506600173608)** (51 messages🔥): 

- **HunyuanDiT 引发褒贬不一的反应**：腾讯推出的 [HunyuanDiT 模型](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 被誉为中文 prompt 遵循方面的 SOTA，但评价褒贬不一。一些人称赞其输出质量，而另一些人则指出，与 stable cascade 模型相比，它在处理直线方面表现吃力。

- **AniTalker 利用音频驱动静态肖像动画**：[AniTalker](https://x-lance.github.io/AniTalker/) 旨在将单张静态肖像结合输入音频转换为动画说话视频。尽管控制信号相似，它仍能生成多样且逼真的面部动画。

- **Google DeepMind 的 Imagen 3 亮相**：[Google DeepMind 推出了 Imagen 3](https://fxtwitter.com/GoogleDeepMind/status/1790434750592643331)，这是一款高质量的文本生成图像模型，拥有令人惊叹的细节和逼真的光影效果。然而，一些人对其可访问性和潜在限制表示担忧。

- **depyf 首次亮相以辅助 PyTorch 用户**：PyTorch 推出了 [depyf](https://pytorch.org/blog/introducing-depyf)，这是一个用于理解 `torch.compile` 的新项目，旨在简化深度学习性能优化的复杂性。虽然受到欢迎，但也有人呼吁提供更好的错误信息。

- **AI 竞赛由能源和 GPU 需求驱动**：讨论强调了 AI 对巨大能源消耗的依赖以及 GPU 的关键作用。引用的一个例子是 8x H100 设备在每张 GPU 待机功率为 75W 时，会导致巨大的电力需求和可持续性担忧。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/introducing-depyf">Introducing depyf: mastering torch.compile with ease</a>：未找到描述</li><li><a href="https://x-lance.github.io/AniTalker/">AniTalker</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - a Hugging Face Space by multimodalart</a>：未找到描述</li><li><a href="https://fxtwitter.com/GoogleDeepMind/status/1790434750592643331?t=gliMAi7wtzSx9s4HKnZJGA&s=19">Tweet from Google DeepMind (@GoogleDeepMind)</a>：我们推出了 Imagen 3：这是我们迄今为止质量最高的文本生成图像模型。🎨 它生成的视觉效果具有令人难以置信的细节、逼真的光影和更少的干扰伪影。从快速草图...</li><li><a href="https://fxtwitter.com/multimodalart/status/1790309209193509326?t=ryXEhFyHMWx5xwfWM8qAlA&s=19">Tweet from apolinario (multimodal.art) (@multimodalart)</a>：第一个开放的类 Stable Diffusion 3 架构模型刚刚发布 💣 - 但它不是 SD3！🤔 它是腾讯的 HunyuanDiT，一个 15 亿参数的 DiT (diffusion transformer) 文本生成图像模型 🖼️✨ 在...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1240009909261697035)** (16 messages🔥): 

- **DeepMind 的 Veo 将使视频制作大众化**：提到 [DeepMind 的 Veo](https://deepmind.google/technologies/veo)，成员们强调了它能够根据 prompt 生成超过一分钟的高质量 1080p 视频，捕捉细微差别和电影级效果。相关功能很快将通过 [VideoFX](https://labs.google/videofx) 向选定的创作者开放，目前已开启等候名单。

- **VidProM 数据集：文本生成视频的新资源**：一篇新论文介绍了 [VidProM](https://arxiv.org/abs/2403.06098)，这是一个大规模数据集，包含 167 万个独特的文本生成视频 prompt 和 669 万个由扩散模型生成的视频。该资源解决了缺乏特定 prompt 数据集的问题，并与 DiffusionDB 形成对比。

- **神经网络图像采样的挑战**：成员们讨论了双线性图像采样中梯度的局部性以及傅里叶变换中正则化效果差的问题。有人建议训练一个小网络来近似双线性采样，这可能会提供更平滑、全局优化的梯度。

- **Google Imagen 3 树立新标准**：[Google Imagen 3](https://deepmind.google/technologies/imagen-3) 被誉为最高质量的文本生成图像（text-to-image）模型，具有更好的细节和更丰富的光影。目前已向选定的创作者开放私人预览，它有望为社区数据集和项目生成合成数据提供帮助。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.06098">VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models</a>：Sora 的到来标志着文本生成视频（text-to-video）扩散模型进入了一个新时代，在视频生成和潜在应用方面带来了显著进步。然而，Sora 以及其他文本生成视频模型...</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>：Veo 是我们迄今为止功能最强大的视频生成模型。它可以生成高质量、1080p 分辨率、时长可超过一分钟的视频，涵盖广泛的电影和视觉风格。</li><li><a href="https://deepmind.google/technologies/imagen-3/">Imagen 3</a>：Imagen 3 是我们最高质量的文本生成图像模型，与之前的模型相比，能够生成细节更出色、光影更丰富且干扰伪影更少的图像。
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1239882886606229616)** (35 messages🔥): 

- **在 macOS 上管理本地模型**：一位用户分享了他们在 macOS 上本地运行 **ollama/dolphin-mixtral:8x7b-v2.6** 以避免高昂费用的经验。他们还讨论了为模型尝试自定义指令。
- **零成本运行模型**：用户讨论了运行本地模型的方法，其中一人表示他们更喜欢本地设置以避免高额开支，另一人提到了 **OpenRouter** 和 **Groq** 作为替代方案。一位用户分享了将 **Groq** 与 **llama3** 和 **Mixtral** 等模型集成的命令。
- **OpenInterpreter 的操作系统偏好**：辩论集中在运行 OpenInterpreter 时使用 **Windows 还是 Ubuntu**。多位用户指出 Ubuntu 运行效果更好，尤其是对于带有 GPU 的本地模型，但也提到了针对 Ubuntu 的特定自定义指令，以避免使用 macOS 特有的命令。
- **Ubuntu 的自定义指令**：一位用户分享了专为 Ubuntu 定制的系统指令，以避免与 macOS 命令相关的问题。他们强调了“记住你是在 UBUNTU 上运行！！！请使用 Ubuntu 命令”的要求。
- **持久化 Sudo 密码**：用户讨论了如何处理 OpenInterpreter 中的 sudo 密码请求。一位用户建议直接将 sudo 密码嵌入到系统消息（system message）中。

**提到的链接**：<a href="https://tenor.com/view/thank-you-sticker-thanks-sticker-line-sticker-bear-sticker-bear-gif-26476682">Thank You Sticker Thanks Sticker GIF - Thank You Sticker Thanks Sticker Line Sticker - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1239958830193053768)** (23 messages🔥): 

- **偏好开源 AI 而非 Apple 的集成**：一位成员表示，相比 **Apple 将 AI 集成到其操作系统**，他更倾向于 **开源 AI**。另一位成员回应道：“那就选 Linux 吧！”。

- **Light 预订发货咨询**：包括 **yikesawjeez** 和 **maxpetrusenko** 在内的多位用户在延迟数月后，正在询问其 **Light 设备预订** 的发货状态。

- **固件更新和重新刷机方案**：一位成员提供了解决设备无法更新问题的方案，建议在重新刷机前升级固件或在 Arduino 工具菜单中启用“刷机前擦除（erase before flashing）”。

- **01 Terminal 的调试模式**：**.merlinvt** 发现了如何通过在 i.py 脚本中设置 `"interpreter.debug = True"` 来启用 01 的调试模式。该模式允许用户查看底层操作并改进系统消息。

- **OpenRouter 与 Groq 集成的问题**：用户讨论了让 **OpenRouter 与 Groq 协同工作** 的困难，一位成员指出 OpenRouter 文档存在历史性错误，另一位成员则遇到了重复提示词导致的循环问题。
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1240064323829633184)** (2 messages): 

_

- **创新和以客户为中心驱动成功**：分享的一集 [播客要点](https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e) 强调，成功源于创新、质量和以客户为中心，而非控制。“差异化和价值源于在某些独特且富有创意的事情上做到极致”，而创意是稀缺且有利可图的。
- **从历史案例中学习**：另一个关键要点指出，历史证明控制会导致失败，并以专制作为负面案例。“生存和成功需要制造最好的产品并满足客户需求，而不是试图控制他们。”
- **Linus Torvalds 与 Open Interpreter 的相似之处**：一位成员对关于 Linus Torvalds 的创始人播客集表示热烈赞赏，并指出 Linux 项目与 Open Interpreter 之间存在相似之处。他们还称赞了 Torvalds 的书名 *《Just for Fun》*（只是为了好玩）非常贴切。

**提到的链接**：<a href="https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e">Jack Mielke 关于 #176 Linus Torvalds（Linux 创始人）的 AI 播客笔记</a>：查看使用 Snipd 创建的 AI 播客笔记。

---

**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1240332490154053725)** (58 条消息🔥🔥): 

- **Google 未能强调 LLM 的不准确性**：讨论集中在 **Google I/O 主题演讲** 如何未能解决 LLM 不可靠的本质，考虑到风险，这使得演示的吸引力大打折扣。一位用户指出：“OpenAI 在这方面做得更好，他们至少承认了这些东西可能出错的一些方式。”
  
- **Meta 低调发布的 AI 令人印象深刻**：Meta 在没有媒体宣传的情况下缓慢推出多模态 AI 的策略给一些用户留下了深刻印象。一位成员评论道：“即使是 AI 设备，他们的 Wayfarer 眼镜作为耳机和摄像头本身就非常棒。”
  
- **提议为实用 AI 用途建立 “Sober AI”**：一位成员提议创建一个 “Sober AI” 展示区，以突出那些相对平淡但有效的 AI 驱动工具。Simon Willison 支持这个想法，说：“老实说，这是一个很棒的主意。”
  
- **AI 在新闻业的实际应用**：用户讨论了 AI 的实际应用，例如 **MuckRock** 使用 AI 进行 FOIA（信息自由法）任务自动化，以及一位成员的 AI 新闻课程专注于数据提取和实现媒体“采访”。Zach Seward 在 SXSW 上关于新闻业 AI 的演讲也因展示了无炒作的有价值 AI 应用而受到称赞。
  
- **优化 LLM 效率的努力**：讨论了 **Gemini 的 context caching** 和 **llama.cpp 的 prompt caching** 等技术，作为使 LLM 工作流更便宜、更高效的方法。一位用户指出，“长 Prompt 可能消耗了大部分的 Token 使用量”，强调了这些策略的潜在好处。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.boundaryml.com/">Boundary | AI 工程师的一站式工具包</a>：未找到描述</li><li><a href="https://www.amazon.com/Edisons-Eve-Magical-History-Mechanical/dp/1400031583">未找到标题</a>：未找到描述</li><li><a href="https://www.theverge.com/2024/5/15/24154808/ai-chatgpt-google-gemini-microsoft-copilot-hallucination-wrong">我们必须停止忽视 AI 的幻觉问题</a>：AI 可能很酷，但它也是个十足的骗子。</li><li><a href="https://www.zachseward.com/ai-news-thats-fit-to-print-sxsw-2024/">适合印刷的 AI 新闻</a>：新闻机构如何以好坏参半的方式使用 AI。</li><li><a href="https://github.com/MuckRock/muckrock/blob/11eb9a155fd52140184d1ed4f88bf5097eb5e785/muckrock/foia/tasks.py#L388">MuckRock 源代码 (GitHub)</a>：MuckRock 的源代码 - 请向 info@muckrock.com 报告错误、问题和功能请求。</li><li><a href="https://ai.google.dev/gemini-api/docs/caching">未找到标题</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/e1b40ac3b94824d761b5e26ea1bc5692706029d9/examples/main/main.cpp#L225-L245">llama.cpp 示例代码 (GitHub)</a>：C/C++ 中的 LLM 推理。在 GitHub 上参与 llama.cpp 的开发。
</li>
</ul>

</div>

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1240390214241751060)** (1 条消息): 

- **在模型之间切换上下文引发担忧**：一位成员对使用不同模型（“`4o`”）继续对话表示担忧，担心这可能会破坏对话。他们建议从 SQLite 表的最新条目中提取 JSON 日志，并将其提供给另一个模型作为变通方案。

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1239868524688576552)** (56 条消息🔥🔥): 

- **ChatGPT 的设备集成**：一名成员建议将“动作（action）”提升为像音频和视觉一样的一等公民模态，以实现 ChatGPT 与设备的全面集成。他们指出，目前大多数初创公司使用文本作为中间媒介，通过 PyAutoGUI 等工具来控制设备。

- **大规模脑成像项目凸显数据挑战**：最近一个对一立方毫米人类大脑组织进行全成像的项目需要 1.4 PB 的存储空间。这项由哈佛大学研究人员和 Google AI 合作完成的[研究](https://blog.google/technology/research/google-ai-research-new-images-human-brain/)展示了在扩展此类实验时面临的巨大数据挑战。

- **Google 的新 AI 模型及褒贬不一的反应**：Google 推出了几款新的 AI 模型，包括 **Veo** 和 **Project Astra**。虽然一些用户对 Veo 的功能印象深刻，但也有人发现其质量参差不齐，而 **Project Astra** 在与 GPT-4o 的现场演示对比中评价褒贬不一。

- **对 Perplexity AI 的批评及替代方案**：成员们报告了 Perplexity AI 提供虚假来源以及在没有“Pro”账户的情况下无法使用的问题。讨论中提到了针对代码问题的 **Phind.com** 和具有强大搜索功能的 **Kagi** 等替代方案。

- **Ilya Sutskever 从 OpenAI 离职**：Ilya Sutskever 宣布从 OpenAI 离职，引发了社区的各种反应。[Sam Altman](https://x.com/ilyasut/status/1790517455628198322) 和其他关键人物对这一变动发表了评论，预示着公司内部将进行重大重组。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/theinformation/status/1790467870545027186?s=46">来自 The Information (@theinformation) 的推文</a>：OpenAI 聘请了在 Google 工作 21 年的资深人士 Shivakumar Venkataraman，他此前领导该公司的搜索广告业务。此举正值 OpenAI 开发将与 Google 竞争的搜索引擎之际...</li><li><a href="https://x.com/nearcyan/status/1790533418658455688">来自 near (@nearcyan) 的推文</a>：.@janleike（RLHF 的共同发明人）也将离开 OpenAI</li><li><a href="https://x.com/GoogleDeepMind/status/1790435824598716704">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：介绍 Veo：我们功能最强大的生成式视频模型。🎥 它能够创建高质量的 1080p 片段，时长可超过 60 秒。从写实主义到超现实主义和动画，它可以应对各种风格...</li><li><a href="https://x.com/0xgaut/status/1790428601789067614">来自 gaut (@0xgaut) 的推文</a>：OpenAI：这是 GPT-4o；Google：</li><li><a href="https://x.com/GoogleDeepMind/status/1790434750592643331">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：我们正在推出 Imagen 3：我们迄今为止质量最高的文本生成图像模型。🎨 它能产生具有惊人细节、真实光影且干扰瑕疵更少的视觉效果。从快速草图到...</li><li><a href="https://x.com/ilyasut/status/1790517455628198322">来自 Ilya Sutskever (@ilyasut) 的推文</a>：在近十年后，我决定离开 OpenAI。公司的发展轨迹堪称奇迹，我相信 OpenAI 将构建出既安全又造福人类的 AGI...</li><li><a href="https://x.com/dwarkesh_sp/status/1790765691496460460?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：这是我与 @johnschulman2（OpenAI 联合创始人，领导了 ChatGPT 的创建）的一期节目。关于后训练如何驯服 Shoggoth，以及未来进步的本质... 链接见下文。请欣赏！</li><li><a href="https://www.tomshardware.com/tech-industry/full-scan-of-1-cubic-millimeter-of-brain-tissue-took-14-petabytes-of-data-equivalent-to-14000-full-length-4k-movies">对 1 立方毫米脑组织的完整扫描耗费了 1.4 PB 数据，相当于 14,000 部 4K 电影 —— Google 的 AI 专家协助研究人员</a>：令人惊叹的大脑研究。</li><li><a href="https://live.siemens.io/">2024 年西门子开源活动</a>：西门子举办的关于开源软件所有主题的年度系列活动。欲了解更多信息，请访问 opensource.siemens.com</li>
</ul>

</div>

---

**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1240076681390325790)** (1 条消息): 

- **明天关于 Evals 的精彩讨论**：发布了关于即将举行的 Evals 活动的公告，感谢 Eugene 自愿主持该会议。鼓励大家在[此处](https://eugeneyan.com/writing/evals/)阅读相关资料并准备问题。
- **通过 iCal 订阅保持通知**：提供了如何将活动日历添加到自己的日历中的说明，只需点击 RSS 标志并选择“添加 iCal 订阅”。这被推荐为接收 Latent.Space 新活动通知的主要方式。

**提到的链接**：<a href="https://lu.ma/1hoagv05">LLM Paper Club (Eugene on Evals) · Zoom · Luma</a>：Eugene 正在带我们深入了解所有的评估（evals）：https://eugeneyan.com/writing/evals/。同时欢迎为我们的下一篇论文提交建议并投票：…

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1239998575648444496)** (33 条消息🔥): 

- **训练 cmdR+ 100b 模型似乎不受支持**：一位成员表达了训练 cmdR+ 100b 模型的强烈需求，称其是目前唯一高质量的多语言模型。随后讨论了由于巨大的 VRAM 需求，如何在 GPU 之间分配权重以及使用 FSDP 的可行性。
  
- **Llama3 随着数据量增加而获得关注**：一位用户报告了 Llama3 的成功结果，并将其归功于使用了更多数据。另一位成员对所使用的配置细节表现出兴趣。

- **TinyLlama 模型的目录问题**：一位用户在尝试使用 TinyLlama 时遇到了 `No such file or directory` 错误，但在使用 Mistral 模型时未遇到此问题。排查尝试包括删除目录和手动干预，最终通过在 RunPod 中执行特定命令解决了该问题。

- **关于 Falcon 11b 与 LLaMA 3 的辩论**：成员们讨论了 Falcon 11b 和 LLaMA 3 的优缺点，并考虑了许可协议问题。一位成员指出 Falcon 2 的许可证包含一个有问题的条款，暗示虽然该许可证并非完全开放，但可能无法强制执行。

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1240100543100227706)** (6 条消息): 

- **PEFT 需要从仓库安装**：一位成员指出 [peft](https://github.com/huggingface/peft/releases) 自三月以来一直没有更新，建议直接从 GitHub 仓库安装。
- **Xformers 版本问题**：`requirements.txt` 中将 xformers 版本硬性设定为 0.0.22，导致在更新其他包时出现冲突。这种版本锁定旨在支持旧版 PyTorch，但被认为会导致兼容性问题。
- **多 GPU 配置的手动测试**：成员们讨论了对 deepspeed 等某些组件的更新需要进行广泛的手动测试，特别是针对多 GPU 配置，以确保它们在各种设置下保持功能正常。
- **多 GPU 设置的验证**：一位用户确认他们的多 GPU 设置在 Nvidia 环境下工作正常，这意味着所讨论的配置和版本在他们的环境中是可运行的。

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1240262005370388581)** (2 条消息): 

- **成员寻求关于 LORA 训练提示词的指导**：一位成员询问遵循基础模型训练时的提示词风格（例如 llama3 的 `<|eot|>` token）是否能为 LORA 带来更好的效果。他们询问将 alpaca-formatted 数据集重新格式化为 llama3 风格是否能改善结果。

---

**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1240157572485480458)** (3 条消息): 

- **Tiger Lab 发布具有挑战性的 MMLU-Pro 数据集**：[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) 被介绍为一个用于基准测试大型语言模型的**鲁棒**且**具有挑战性**的数据集。它包含 1.2 万个复杂问题，并将多选题选项从 4 个增加到 10 个。

**提到的链接**：<a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro">TIGER-Lab/MMLU-Pro · Hugging Face 数据集</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1239989093669277726)** (1 条消息): 

- **Axolotl 在 Runpod 上的初始问题**：一位成员报告在使用提供的容器在 Runpod 上尝试启动 8xH100s 的 axolotl 运行任务时遇到了 **CUDA 错误**。他们提到使用 **`winglian/axolotl:main-latest`** 镜像也无法正常启动 pod。

- **找到潜在解决方案**：该成员随后编辑了消息，表示通过使用 **community axolotl cloud image** 可能会解决该问题。这暗示了这种替代方案可能取得成功。

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1240092491789897748)** (8 条消息🔥):

- **寻求 YAML 优化以实现更快的 Fine-Tuning**：一位成员寻求关于最小化 YAML 配置运行时间的建议，以便检查 Fine-Tuning 过程的系统设置。他们更关心速度而非结果质量。
- **禁用 Gradient Checkpointing 的影响**：针对上述优化查询，另一位成员询问“禁用 `gradient_checkpointing`”是否真的会对运行速度产生影响。讨论强调了调整设置以在内存节省和计算速度之间取得平衡。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b05e7d25-cd93-40be-a6b1-05f9a8ed5f77)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

---

**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1239867085350502400)** (16 messages🔥): 

- **对 Command R 的 RAG 能力印象深刻**：一位用户对 Command R 的 RAG 能力表示高度赞赏，称其“不仅价格便宜，而且极其准确，即使在源内容非常长的情况下也能忠实于给定源”。

- **澄清 Preamble 与 System Message 的区别**：针对 Cohere 模型的 “Preamble” 和 “System Message” 之间的区别展开了讨论。用户解释说，Preamble 是 System Message 的一部分，并包含在由 `<|SYSTEM_TOKEN|>` 和 `<|END_OF_TURN_TOKEN|>` 划定的特殊 Token 中。

- **理解示例中的 Token 划分**：一位用户阐明了 Token 划分在 System 部分的工作原理，使用特殊 Token 来指示系统指令的开始和结束。这一细节有助于模型在对话期间识别并做出适当响应。

- **Reranker 模型高亮功能咨询**：一位用户分享了使用 Cohere Reranker 的成功经验，但询问它是否能提供相关 Token 的高亮显示。他们提到在 ColBERT 中使用类似功能来计算词相关性，这有助于向用户高亮显示重要词汇。

- **成员介绍**：新成员包括 Nedal（工程师/供应链经理）等进行了简短的自我介绍。多位用户之间进行了常规的问候和欢迎。

---

**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1239921278937333771)** (2 messages): 

- **类似工作的协作邀请**：一位成员表达了对某个项目进行协作的兴趣，提到：*“嗨 Asher，我也在做同样的事情。我想合作。”*

- **分享 RAG 学习文章**：一位成员分享了一篇关于使用 **@UnstructuredIO API** **从零开始学习 RAG** 的 [Medium 文章](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2)。文章的重点在于如何以结构化方式从 PDF 中提取内容。

---

**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1240043290053705838)** (2 messages): 

- **tinygrad 探索 Urbit/Nock 移植**：一位用户正致力于将 tinygrad 移植到 **Urbit/Nock**，并已实现了一些 opcodes，初步目标是 `forward()` 函数。他们分享了[项目链接](https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon)，并提到需要一个与 tinygrad 风格 Python 代码兼容的转换层。
- **新贡献者的第一个 Issue**：George Hotz 为新贡献者介绍了一个适合入门的 Issue。该 Issue 标题为 [BEAM kernel count number is wrong](https://github.com/tinygrad/tinygrad/issues/4595)，详情已在 tinygrad GitHub 仓库中列出。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/4595">BEAM kernel count number is wrong · Issue #4595 · tinygrad/tinygrad</a>：beam2 : 16 31 31 16 2 3 2 4 3 2 : 817.92 us &lt; hc : 4 31 31 32 4 3 3 4 2 2 : 1000.83 us *** GPU 9 r_16_31_31_16_2_3_2_4_3_2 arg 3 mem 0.87 GB tm 1244.89us/ 4.99ms ( 113.83 GFLOPS, 24.03 GB/s) 0.00s:...</li><li><a href="https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon">numerics/maroon/desk/lib/tinygrad.hoon at main · urbit/numerics</a>：通过在 GitHub 上创建账号来为 urbit/numerics 的开发做出贡献。
</li>
</ul>

</div>

---

**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1239937625293586463)** (14 messages🔥): 

- **PTX=1 时 CUDA 错误困扰 GeForce 4090**：一位用户在 CUDA 12.4 的 GeForce 4090 上运行 **tinygrad** 时遇到了多个错误，并发现有必要更新驱动程序。他们随后确认 CUDA 在 Titan V 上可以工作，但 PTX=1 仍然报错，这表明驱动程序更新至关重要。

- **Shape-Stride Visualizer 工具简化了重塑（reshaping）**：分享了一个创新工具，用于可视化 tinygrad 中 **view** 和 **shapetracker** 使用的 **shape index expression**，帮助用户理解旧数据布局与新数据布局之间复杂的映射关系。[Shape-Stride Visualizer](https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx) 通过展示维度在内存中的布局方式，辅助解释表达式。

- **TACO 展示张量格式可视化**：Tensor Algebra Compiler (TACO) 通过 Dense, Compressed, 和 Singleton 等不同层级表示张量格式，并将其转换为生成的代码。该工具捕获了各种格式，包括非行优先（non-row major）张量格式，提供了对张量操作的深入见解 [TACO documentation](http://tensor-compiler.org/codegen.html)。

- **重排序 Reduce 与展开（Expanding）优化**：一位用户分享了 tinygrad 中的优化技巧建议，建议在 expand 之前进行 realize，以避免重复工作。他们还提到了管理 reduce 操作的重要性，以便在某些情况下绕过展开的需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx">Shape & Stride Visualizer</a>：未找到描述</li><li><a href="http://tensor-compiler.org/codegen.html">Web Tool</a>：TACO 项目网站
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1240035893600059412)** (5 条消息): 

- **Hugging Face AI Town 在 CPU 上运行**：分享了在 CPU 上运行的 [Hugging Face's AI Town](https://huggingface.co/spaces/radames/ai-town)。它被强调为目前在容器中使用 Hugging Face 的“最佳选择”。

- **对用于 Agent 控制的 AI Town API 的兴趣**：一位成员询问 AI Town 是否提供通过 API 进行 Agent 控制，以便与自定义代码集成。虽然目前不支持为每个 Agent 设置单独的 LLM，但有关于通过 LLamaFarm 工作提供潜在支持的讨论。

- **AI Town API 集成的可能性**：另一位成员详细阐述了 AI Town 可能实现的 API 集成层级。建议包括调用用于 completions 和 embeddings 的 API，或者用于交互控制和内存管理的更具语义化的 API，并支持用于状态查询订阅的 webhook。

**提到的链接**：<a href="https://huggingface.co/spaces/radames/ai-town">AI Town on HuggingFace - a Hugging Face Space by radames</a>：未找到描述

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1240035763412799572)** (4 条消息): 

- **AI Town 在 Hugging Face Spaces 上线**：一位用户兴奋地宣布 **AI Town** 现已在 [Hugging Face Spaces](https://huggingface.co/spaces/radames/ai-town) 上可用。此消息包含了该 Space 的链接及其运行环境的详细信息。
- **优化 NPC 交互的建议**：一位成员建议减少 NPC 的数量并调整“cooldown”时间的常量，以优化 AI Town 的性能。这些调整有助于管理 NPC 在开始新对话前的等待时间以及他们参与活动的方式。

**提到的链接**：<a href="https://huggingface.co/spaces/radames/ai-town">AI Town on HuggingFace - a Hugging Face Space by radames</a>：未找到描述

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** (1 条消息): 

tommy1901: 准备在这里发一些酷炫的东西
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1239880142181105664)** (9 条消息🔥): 

- **关于德语词表大小（vocab size）与 tokens/byte 的辩论缺乏数据**：一位成员对“德语的 vocab_size vs. tokens/byte 图表”表示感兴趣。另一位成员回应称此类数据目前无法直接获得，并强调了 tokenizer 数据集中**语言混合比例的重要性**。

- **分享 TokenMonster 项目**：在对 tokenizer 的研究背景下，一位成员分享了一个 [GitHub 上的项目](https://github.com/alasdairforsythe/tokenmonster)，将其描述为“适用于 Python, Go & Javascript 的非贪婪子词（Ungreedy subword）分词器和词汇训练器”。

- **GPT-4o 演示被指过于挑逗**：分享了一条嘲讽 GPT-4o 演示过于暗示性的推文。推文可以在[这里](https://fxtwitter.com/main_horse/status/1790099796193398831)查看。

- **GPT-4o 的新词表令用户震惊**：分享的另一条推文对 GPT-4o 的 "o200k_base" 词表表示难以置信，显示出惊讶或不赞同。推文可在[这里](https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09)查看。

- **Ilya 离开 OpenAI**：一个分享的重大更新是 Ilya Sutskever 在 Twitter 上宣布他将从 OpenAI 离职。公告可以在 [这里](https://twitter.com/ilyasut/status/1790517455628198322) 找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/main_horse/status/1790099796193398831">来自 main (@main_horse) 的推文</a>：&#34;为什么 gpt-4o 的演示看起来那么有挑逗感？&#34;</li><li><a href="https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09">来自 Susan Zhang (@suchenzang) 的推文</a>：gpt-4o 这个新的 &#34;o200k_base&#34; 词表让我大吃一惊</li><li><a href="https://github.com/alasdairforsythe/tokenmonster">GitHub - alasdairforsythe/tokenmonster: 适用于 Python, Go 和 Javascript 的非贪婪子词分词器和词表训练器</a>：适用于 Python, Go 和 Javascript 的非贪婪子词分词器和词表训练器 - alasdairforsythe/tokenmonster
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[announcements](https://discord.com/channels/1131084849432768614/1139357591701557258/1239862029632929863)** (1 条消息): 

- **Guild Tags 首次亮相用于用户识别**：**Discord** 宣布从 **5月15日** 开始，用户可能会在某些成员的用户名和个人资料旁边看到新的 **Guild Tags**。这些标签表示该成员属于名为 Guilds 的小型专属服务器，这些服务器专注于共同的身份和爱好。

- **AutoMod 整合 Guild Tags**：启用了 AutoMod 的管理员和版主现在也可以让它检查这些 **Guild Tags**。此功能目前仅限于少数服务器，目前没有手动添加更多服务器到该实验的方法。
  

---



**MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1240307668493799455)** (1 条消息): 

- **预测未来 AI 硬件趋势**：一位用户分享了一篇关于 AI 硬件历史和未来趋势预测的详尽文章，可在 [这里](https://singlelunch.com/2024/04/23/ml_microprocessor_history/) 阅读。该用户在短期内看好 **NVMe 驱动器** 和 **Tenstorrent**，但在未来 5-10 年内对 **GPUs** 的热情较低。
- **Transformer 驱动 AI 突破**：如 [这篇文章](https://thegradient.pub/mamba-explained) 中所讨论的，用户强调基于 *Transformer* 的模型几乎是过去四年中所有重大 AI 突破的关键。他们指出 **Nvidia 的估值** 已经超过了 Amazon 和 Google，这主要归功于 Transformer 技术的进步。

**提到的链接**：<a href="https://singlelunch.com/2024/04/23/ml_microprocessor_history/">AI 硬件的过去、现在和未来 - SingleLunch</a>：未找到描述

  

---



---



---



---



---