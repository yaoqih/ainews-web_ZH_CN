---
companies:
- google
- google-deepmind
date: '2025-05-20T05:44:39.731046Z'
description: '**Google I/O 2024** 展示了来自 **google-deepmind** 的 **Gemini 2.5 Pro** 和
  **Deep Think** 推理模式的重大进展，强调了 AI 驱动的变革以及为开发者带来的机遇。**GeminiApp** 旨在成为通往 **AGI**（通用人工智能）道路上的全能
  **AI 助手**，而 Google 搜索中推出的 **AI 模式**等新功能进一步扩大了生成式 AI 的应用范围。


  此次大会包含多场主题演讲，发布了十多款模型和 20 多种 AI 产品的更新，彰显了 **Google** 在 AI 创新领域的领导地位。**Demis Hassabis**
  和 **Phil Schmid** 等业内大咖分享了深刻见解与总结，同时，作为 Codex 和 Devin 竞争对手的 **Jules** 的发布也备受关注。'
id: MjAyNS0w
models:
- gemini-2.5-pro
- gemini-2.5
people:
- demishassabis
- philschmid
- jack_w_rae
title: Google I/O：Gemini 新原生语音、Flash、DeepThink、AI 模式（DeepSearch+Mariner+Astra）
topics:
- ai-assistants
- reasoning
- generative-ai
- developer-tools
- ai-integration
- model-optimization
- ai-application
- model-updates
- ai-deployment
- model-performance
---

**Gemini is all you need.**

> 2025年5月19日至5月20日的 AI 新闻。我们为您检查了 9 个 Reddit 子版块、449 个 Twitter 账号和 29 个 Discord 社区（215 个频道，7031 条消息）。预计节省阅读时间（以 200wpm 计算）：622 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 呈现方式，涵盖所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

十二个月前，我们报道了 [Google I/O](https://news.smol.ai/issues/24-05-14-ainews-google-io-in-60-seconds)，但坦率地说，当时的 Gemini 还不完全算得上是 frontier 模型，并且在某种程度上被 [4o 的发布](https://news.smol.ai/issues/24-05-13-ainews-gpt-4o-the-new-sota-everything-frontier-model-gpt4t-version) 抢了风头。

六个月前，我们写道 [Google 凭借 Gemini 2.0 觉醒](https://news.smol.ai/issues/24-12-11-ainews-google-wakes-up-gemini-20-et-al)，这开启了长达数月、史诗般的 [Gemini 统治力提升](https://news.smol.ai/issues/?pattern=gemini#2024-May)（甚至采用了 [AINews 图表](https://x.com/swyx/status/1924886631020167300)）：


![gemini](https://resend-attachments.s3.amazonaws.com/O7aiLUgaq2ZuFxS)


而今天，Gemini 的官方数据 [证实](https://x.com/natolambert/status/1924916998133129716?s=46) 了这一点（尽管这在很大程度上得益于其拥有全球最慷慨的免费层级）：


![gemini](https://resend-attachments.s3.amazonaws.com/mWfaoYj1MF4isyk)


下方的 AI Twitter 摘要很好地回顾了主要的发布内容，因此我们不再赘述，但我们肯定要说它错过了 [Jules 的发布](https://news.ycombinator.com/item?id=44034918)（Gemini 的 Codex/Devin 竞争对手），因为 Jules 在某种程度上被提前泄露了。

一如既往，The Verge 在将 3 小时的主题演讲浓缩为 [30 分钟](https://www.youtube.com/watch?v=bDVpI23q8Zg) 方面做得非常出色：


![verge](https://resend-attachments.s3.amazonaws.com/fHSpFg5Q6F6XcT0)


---

# AI Twitter Recap

**Google I/O 2024 活动与公告**

- **Google I/O 活动详情与主题演讲**：[*philschmid](https://twitter.com/_philschmid/status/1924480973049098625) 提供了一个推文串，总结了在线观看和参与所需的链接。**主主题演讲**于 **太平洋时间 5 月 20 日上午 10 点** 举行，**开发者主题演讲**于 **下午 1:30** 举行。**AI Stage** 活动安排在 **5 月 20 日和 21 日**。此外，[philschmid](https://twitter.com/_philschmid/status/1924480975712530607) 列出了活动时间。活动将以 **太平洋时间上午 10 点（中欧夏令时晚上 7 点）** 的 **主主题演讲** 开始。
- **自上次 I/O 以来的 Google AI 进展**：[@Google](https://twitter.com/Google/status/1924873673556594915) 指出，自上次 **Google I/O** 以来，他们已经发布了十几个模型和研究突破，并推出了 20 多个主要的 **AI 产品和功能**。他们去年还展示了 **Project Astra** ([@Google](https://twitter.com/Google/status/1924883454992252993))。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924872748536656214) 也进行了分享。
- **AI 驱动的转型与开发者机遇**：[@Google](https://twitter.com/Google/status/1924901038424781307) 引用了 **Sundar Pichai** 的话，指出 **AI** 带来的重大机遇以及开发者在确保其惠及大众方面的作用。[_philschmid](https://twitter.com/_philschmid/status/1924389434838962444) 表示，他们发布了 **AI Studio 的两个次要 QoL 更新**：一个内置的使用情况仪表板和一个新的 **GenMedia "Playground"**。
- **AI Overviews 与搜索改进**：[@Google](https://twitter.com/Google/status/1924906327190434212) 宣布向美国所有用户推出 **AI Mode**，这是基于 **AI Overviews** 的成功，后者带来了更高的用户满意度和更频繁的搜索。此外，Gemini 2.5 即将接入搜索 ([@Google](https://twitter.com/Google/status/1924885533609599187))。Google 搜索正在将生成式 AI 带给比世界上任何其他产品都更多的人 ([@Google](https://twitter.com/Google/status/1924874920871526830))。[@Google](https://twitter.com/Google/status/1924886582479171927) 预告了 AI Mode 即将推出的功能，例如个性化建议、复杂分析、深度搜索、Agent 能力和 Search Live。
- **Gemini 更新**：[@demishassabis](https://twitter.com/demishassabis/status/1924882592236540085) 表示，他们对 **GeminiApp** 的最终愿景是将它转变为一个通用的 **AI 助手**，这是通往 **AGI** 道路上的一个重要里程碑。一些用户在升级时遇到了问题 ([@hkproj](https://twitter.com/hkproj/status/1924905149916401748))。

- **Gemini 2.5 Pro 和 Flash 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924881598102839373) 在 **2.5 Pro** 中发布了 **Deep Think**，这是一种使用并行思维技术的增强推理模式。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1924897579122491523) 强调 **Deep Think** 标志着 **Gemini** 向更强大的测试时计算（test-time compute）和更强推理能力的演进。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924879639253500361) 提到了更强的安全性以及模型思维过程更高的透明度。研究发现 Gemini 2.5 目前是领先的学习模型 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924878252172353851))。[@omarsar0](https://twitter.com/omarsar0/status/1924885378692972554) 指出 Gemini 2.5 Flash 现在的表现非常出色。此外，Flash 版本在相同性能下使用的 Token 更少 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924879639253500361))。
- **Gemini Diffusion 模型**：GoogleDeepMind 发布了 Gemini Diffusion ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924888095448825893))，该模型的生成速度比 2.0 Flash Light 快 5 倍 ([@omarsar0](https://twitter.com/omarsar0/status/1924882868477563141))。它目前作为实验性 Demo 提供，旨在帮助塑造未来的模型 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924888098019905606))。
- **Veo 3 视频生成模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924893528062140417) 推出了 **Veo 3**，这是一款全新的生成式视频模型，可以为你制作的剪辑添加配乐，允许你创建会说话的角色并包含音效。[*tim*brooks](https://twitter.com/_tim_brooks/status/1924895946967810234) 分享了这一消息。
- **Imagen 4 图像生成模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924892780876464362) 发布了 **Imagen 4**，这是一款图像生成模型，具有更丰富的图像表现、细腻的色彩、复杂的细节和卓越的排版能力。它可以创作漫画、风格化印章、包装等，并改进了拼写能力 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924892789638070732))。
- **Project Astra 和 Gemini Live**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924883244459425797) 回顾了 **Project Astra** 的改进，包括更好的语音输出、记忆能力和计算机控制，使其更加个性化和主动。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924876301573239061) 确认 **GeminiApp** 中的 **Gemini Live** 摄像头和屏幕共享功能已在 **Android** 上可用，并正在向 **iOS** 推广。
- **Agent 模式**：[@Google](https://twitter.com/Google/status/1924877381853978790) 指出，他们正开始在其所有产品中集成 Agent 能力，包括 **GoogleChrome**、**Search** 和 **GeminiApp**。GeminiApp 中的 Agent 模式将允许你将复杂的规划和任务委托给 Gemini ([@Google](https://twitter.com/Google/status/1924877422761005352))。
- **Google Beam（原 Project Starline）**：这个全新的 AI 优先视频通信平台使用最先进的 AI 视频模型，将 2D 视频流转化为逼真的 3D 体验 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924875328037466302))。
- **Android XR 与合作伙伴关系**：Google 宣布搭载 **Android XR** 的眼镜非常轻便，专为全天佩戴设计 ([@Google](https://twitter.com/Google/status/1924899930109575474))。他们正与 **Samsung** 合作开发软件和参考硬件 ([@Google](https://twitter.com/Google/status/1924899932949070065))。
- **定价与可用性**：全新的 **Google AI Ultra 订阅** 级别将提供对 **Gemini 2.5 Pro Deep Think**、**Veo 3** 和 **Project Mariner** 的访问权限 ([@scaling01](https://twitter.com/scaling01/status/1924891236109799838))。Google 正在升级其 **AI 订阅计划** ([@Google](https://twitter.com/Google/status/1924897418249687494))。
- **对发布会的不同观点**：[c_valenzuelab](https://twitter.com/c_valenzuelab/status/1924901633588429311) 认为这些发布内容乏善可陈，甚至令人感到乏味。

**AI 模型发布、评估与分析**

- **DeepSeek V3 详情**：来自 [@deepseek_ai](https://twitter.com/TheTuringPost/status/1924631209050833205) 的研究阐明了 **DeepSeek-V3** 如何利用其关键创新进行工作。
- **Hugging Face 的 Tiny Agents**：Hugging Face 已将其 Tiny Agents 发布到自有的 NPM 软件包中，其特点是基于 Hugging Face 的 Inference Client 和 MCP 栈构建的轻量级可组合 Agent ([@_akhaliq](https://twitter.com/_akhaliq/status/1924871432816783681))。
- **预训练中的模型合并**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1924804322568896800) 强调了一项关于 LLM 预训练期间模型合并的研究，证明合并来自稳定训练阶段的 Checkpoint 可以产生一致且显著的性能提升。
- **Meta 的 Adjoint Sampling**：Meta AI 引入了 Adjoint Sampling，这是一种基于标量奖励（scalar rewards）训练生成模型的新学习算法，可为进一步研究高度可扩展的采样方法奠定基础 ([@AIatMeta](https://twitter.com/AIatMeta/status/1924858860936069503))。
- **Meta 的 KernelLLM 8B 在 Single-Shot 性能上超越 GPT-4o**：Meta 在 Hugging Face 上发布了 KernelLLM 8B，其在 Single-Shot 性能上超过了 GPT-4o 和 DeepSeek V3 等模型 ([@reach_vb](https://twitter.com/reach_vb/status/1924478755898085552))。
- **NVIDIA 的 Cosmos-Reason1-7B**：NVIDIA 发布了 Cosmos-Reason1-7B，这是一款用于机器人领域的新型视觉推理模型。它是首个基于 Qwen 2.5-VL-7B 的机器人推理模型 ([@mervenoyann](https://twitter.com/_akhaliq/status/1924817927561183498))。
- **AniSORA 模型**：Bilibili 在 Hugging Face 上发布了 AniSORA，这是一款动漫视频生成模型，采用 Apache 2.0 协议授权 ([@reach_vb](https://twitter.com/reach_vb/status/1924425789774123316))。
- **Stability AI 发布 Stable Audio Open Small**：[mervenoyann](https://twitter.com/_akhaliq/status/1924430139242283172) 注意到新款文本转音频模型 stable-audio-open-small 的发布。
- **用于长上下文视觉语言模型的 MMLongBench**：MMLongBench 有效且彻底地对长上下文视觉语言模型进行了基准测试 ([@_akhaliq](https://twitter.com/_akhaliq/status/1924477810430624017))。
- **Marin：AI 开发的开放实验室**：[@percyliang](https://twitter.com/percyliang/status/1924527490351169964) 介绍了开放实验室 Marin，以实现开源 AI 的愿景，[@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232110383960163) 进行了分享。Marin 重新利用了在开源软件领域取得成功的 GitHub 来服务于 AI ([@percyliang](https://twitter.com/percyliang/status/1924527492649648393))。
- **OMol25 和 UMA**：Meta AI 发布了 Open Molecules 2025 (OMol25) ([@AIatMeta](https://twitter.com/AIatMeta/status/1924502785028190366))。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1924836697373565191) 表示 Meta AI 刚刚在 @huggingface 上发布了 OMol25。
- **DeepSeek-V3 洞察**：本文介绍了关于 DeepSeek-V3 AI 架构硬件的见解 ([@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232113101890003))。DeepSeek 现在已成为 Nvidia 的基准 ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1924588309688267139))。
- **LMEval 排行榜**：新版本的 Gemini-2.5-Flash 在聊天类别中攀升至总榜第 2 位 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1924894101918646510))。同时发现 Mistral Medium 3 在社区中强势亮相 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1924482515244622120))。
- **代码生成模型排行榜**：DeepCoder-14B-Preview 是一款代码生成模型，可与 OpenAI 的 o1 和 DeepSeek-R1 等顶级推理模型竞争，但体积仅为它们的一小部分 ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1924570759793369303))。
- **Runway References 与图像生成**：这是 Gen-4 References 的新工作流：元素提取与合成 ([@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1924596075568222654))。
- **通过 RL 利用 LLM 提高汇编代码性能**：本文讨论了如何通过强化学习利用 LLM 提高汇编代码性能 ([@_akhaliq](https://twitter.com/_akhaliq/status/1924505603403047208))。
- **Group Think 论文**：该论文关于多个并发推理 Agent 在 Token 级粒度上进行协作 ([@_akhaliq](https://twitter.com/_akhaliq/status/1924504013963173961))。
- **提高 LLM 的事实性**：扩展推理（Scaling Reasoning）可以提高大语言模型的事实性 ([@_akhaliq](https://twitter.com/_akhaliq/status/1924477447120068895))。
- **数据增强研究**：TeortaxesTex 注意到另一篇 Seed 论文，现在是关于数据增强的。

**机器人、Agent 与自动化中的 AI**

- **NVIDIA 的 Physical AI 模型推理**：NVIDIA 开源了 Physical AI 推理模型，这些模型能够理解物理常识并生成适当的具身决策 ([@reach_vb](https://twitter.com/reach_vb/status/1924525937443365193))。
- **Project Mariner**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1924876541147709897) 更新了 Project Mariner，这是他们的研究原型，可以与网页交互并完成任务。
- **用于机器人学习的 DreamGen**：NVIDIA GEAR Lab 推出了 DreamGen，这是一款新型引擎，它不依赖大量人类操作员，而是通过像素级的数字梦境来扩展机器人学习 ([@DrJimFan](https://twitter.com/DrJimFan/status/1924819887139987855))。
- **结合 GitHub Copilot 的 Agentic DevOps**：GitHub Copilot 现在支持整个软件开发生命周期——从规划和实施到更新、测试和调试 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1924495827999031709))。
- **Azure AI Foundry Agent Service**：Azure AI Foundry Agent Service 现在已正式发布 (GA)，并提供一流的 LlamaIndex 支持 ([@llama_index](https://twitter.com/llama_index/status/1924502129974411504))。

**公司合作伙伴、投资与业务应用**

- **Cohere 与 Dell 及 SAP 的合作伙伴关系**：Cohere 宣布与 Dell 合作提供本地部署的 Cohere North ([@cohere](https://twitter.com/cohere/status/1924512634373865950))，并与 SAP 合作推动企业自动化 ([@cohere](https://twitter.com/cohere/status/1924858543716630644))。
- **Sakana AI 与三菱日联银行 (MUFG Bank)**：Sakana AI 与日本最大的银行三菱日联银行签署了全面合作伙伴协议 ([@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1924442310210678974))。三菱日联金融集团的银行部门已聘请初创公司 Sakana AI 为其系统提供 AI 动力 ([@hardmaru](https://twitter.com/hardmaru/status/1924480171606003841))。
- **Klarna 与 OpenAI**：[@kevinweil](https://twitter.com/kevinweil/status/1924285487491240162) 提到了与 OpenAI 和 Box 的合作伙伴关系。

**技术、工具与教程**

- **数据质量与 LLM 训练**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1924482811899064451) 分享了调试 LLM 训练数据集的实用指南，强调了数据质量的重要性。
- **Hugging Face Hub 增强功能**：Hugging Face Hub 现在可以在交互式查看器中自动格式化对话/推理消息 ([@_lewtun](https://twitter.com/_lewtun/status/1924492654282207368))。
- **LlamaIndex 更新**：LlamaIndex 团队正在举办其首个 Discord 答疑时间 (office hours) 活动 ([@llama_index](https://twitter.com/llama_index/status/1924527932258845178))。
- **微软的开源工作**：微软开源了 Visual Studio Code 中的 GitHub Copilot、Natural Language Web (NL Web)、TypeAgent、Windows Subsystem for Linux (WSL) 以及 Edit 命令行文本编辑器 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1924598434507743728))。
- **Together AI 的代码执行产品**：Together AI 推出了两款新产品，为 AI 应用带来代码执行和开发环境：Together Code Sandbox 和 Together Code Interpreter ([@togethercompute](https://twitter.com/togethercompute/status/1924860124436238532))。
- **Hugging Face Hub 与 MLX 集成**：两项全新的 MLX + Hugging Face Hub 集成使得在本地运行模型变得前所未有的简单 ([@awnihannun](https://twitter.com/awnihannun/status/1924512714287939816))。
- **API 中的结构化输出**：在 LLM API 中，结构化输出变得更加规范——包括对正则表达式 (regex) 的支持！ ([@stevenheidel](https://twitter.com/stevenheidel/status/1924924775266144565))。
- **LangGraph 平台支持 MCP**：LangGraph 平台现在支持 MCP。LangGraph 平台上部署的每个 Agent 现在都会暴露其自身的 MCP 端点 ([@LangChainAI](https://twitter.com/LangChainAI/status/1924863441862562279))。
- **我是唯一一个希望有方法能在一行中调用这两个函数的人吗？**：[@gabriberton](https://twitter.com/gabriberton/status/1924322645627269630) 指出了出于效率考虑，对 `optimizer.step_and_zero_grad()` 的普遍需求。
- **使用 Google Gemini 构建 AI Agents**：[@_philschmid](https://twitter.com/_philschmid/status/1924886346444710135) 推荐了一篇关于如何开始使用 Google Gemini 构建 AI Agents 的博客。
- **学习如何为医疗和法律任务构建轻量级、实时的 AgnoAgi agents**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1924348846647259626) 的教程涵盖了模块化 agents 以及保持低资源消耗的技术。

**政治、伦理与哲学思考**

- **美国与中国的负和博弈**：TeortaxesTex 讨论了美国如何试图与中国进行一场负和博弈，通过剥夺中国规模化所需的资源，而美国自身也在损失金钱和人才。其目标是在绝对能力水平足够的情况下，建立充分的相对差距，从而使美国的霸权地位持久化。
- **AI 的开放性**：美国的历史性技术领导地位并非建立在保护主义和封闭系统之上，而是建立在创造一个充满选择权的动态市场，包括全世界都可以构建其上的开放平台 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1924578324392587385))。

**幽默与杂项**

- **“no discourse”**：[@EigenGender](https://twitter.com/EigenGender/status/1924239911630356970) 分享说，他们痴迷于说 “no discourse” 这个想法，就像 2025 年版的 “no homo”。
- **“煮过头的汉堡烧焦了奶酪”**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1924857866621501841) 说，这大概是一句关于对失败者展现风度的古代中国谚语。
- **测量主旨演讲中提到 “AI” 次数的排行榜**：看来他们有了新的领跑者 ([@Google](https://twitter.com/Google/status/1924901352070676540))。
- [@jxmnop](https://twitter.com/jxmnop/status/1924578141688791392) 分享了一个关于在一家法国 LLM 初创公司工作的朋友的趣事。
- “«说的是基本、明智的话，而不是像个垂涎欲滴、只会教条式复读的平庸之辈。«投资研发以创新»。«雇佣聪明人»。«硅谷很酷，它在创新且无所畏惧»。«洗手»» 真是个疯子 🤯 怎么能指望美国参与竞争呢 😩” [@teortaxesTex](https://twitter.com/teortaxesTex/status/1924586798455390373) 写道。
- 根据 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1924505705400160652) 的说法，“有趣的是，政治光谱实际上更像是一个球体”。
- [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1924495495516573749) 开玩笑说，需要咖啡因是一种能力问题（skill issue）。
- [@demishassabis](https://twitter.com/demishassabis/status/1924501631972057186) 说正在为明天准备一些好东西……

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Gemma 3n 模型发布与社区反应

- [**Gemma 3n 预览版**](https://huggingface.co/collections/google/gemma-3n-preview-682ca41097a31e5ac804d57b) ([Score: 333, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1kr8s40/gemma_3n_preview/)): **Google 新发布的 Gemma 3n 系列预览版 ([Hugging Face 发布页面](https://huggingface.co/collections/google/gemma-3n-preview-682ca41097a31e5ac804d57b)) 推出了专为边缘和低资源设备设计的高效多模态模型。它们利用选择性参数激活技术（类似于 Mixture-of-Experts，简称 MoE），使模型能够以 2B 或 4B 的有效参数量运行——尽管总参数集更大——从而在受限硬件上优化推理。这些模型支持文本、图像、视频和音频输入，经过指令微调（instruction-tuned），并涵盖 140 多种语言；更多技术细节托管在 [Gemma 3n 文档页面](https://ai.google.dev/gemma/docs/gemma-3n)。** 评论者指出，该架构偏离了典型的 Transformer 布局，推测其与 Gemini 架构相似（特别是在多模态和扩展上下文方面），并强调了其在保护隐私的本地部署（例如 HomeAssistant 替代方案）中的潜力。
    - Gemma 3n 使用“选择性参数激活技术”，这是 MoE 的一种形式，允许在边缘设备上部署，其资源占用相当于有效参数大小为 2B 和 4B 的模型，尽管总数更高（例如，E2B 运行时约有 5B 激活参数）。该技术旨在平衡效率和性能，实现多模态能力——包括文本、图像、视频和音频输入——并在 140 多种语言中运行。
    - 架构设计似乎深受 Gemini Nano 系列的启发或直接基于该系列，后者以强大的多模态和扩展上下文处理能力而闻名。技术推测认为，Gemma 3n 的 MoE 门控不仅按通用用途划分，还可能按*模态*划分（即根据输入类型激活专家子网络），这可以提高处理不同输入类型的效率和专业化程度。
    - 官方 [Gemma 3n 文档](https://ai.google.dev/gemma/docs/gemma-3n#parameters) 确认了 E2B 的典型激活参数数量，并强化了该模型独特的基于模态的结构。该模型的边缘导向设计（适用于 Home Assistant 或 DIY Alexa 等系统）强调设备端隐私和数据本地化，与传统的云端模型不同。

- [**Announcing Gemma 3n preview: powerful, efficient, mobile-first AI**](https://developers.googleblog.com/en/introducing-gemma-3n/) ([Score: 159, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1krc35x/announcing_gemma_3n_preview_powerful_efficient/)): **Google 发布了 Gemma 3n 的预览版，这是一款专为移动级硬件上的高效、实时设备端推理而设计的 AI 模型。该模型采用 5B/8B 参数设计，但利用了 DeepMind 的 Per-Layer Embeddings (PLE)、KVC 共享和先进的量化技术，使 Gemma 3n 实现了与更小模型相当的 RAM 效率，支持动态子模型 (MatFormer)，并能处理多模态任务（音频、图像、视频、文本）。基准测试显示了强大的多语言结果（例如 WMT24++ 上的 ChrF 为 50.1%），其隐私优先的设计允许在 Android 和 Chrome 上离线使用；早期访问通过 Google AI Studio 和 Google AI Edge 提供。[官方博客](https://developers.googleblog.com/en/introducing-gemma-3n/)。** 热门技术评论对 Google 声称其基准测试可比肩 Claude Sonnet 3.7 表示怀疑，并质疑 Arena Score 在此类比较中的可靠性。其他人讨论了实际应用场景，如本地智能音箱推理，强调延迟比准确性更重要，以及进一步缩短响应时间甚至消除对 Whisper 等独立 Speech-to-Text (STT) 组件需求的潜力。
    - 针对 4B 参数的 Gemma 3n 预览版能与 Claude Sonnet 3.7 媲美的说法存在质疑。评论者强调需要独立的基准测试，并提到 Google 之前的演示在实时性能和统计代表性方面具有误导性。
    - 一位用户报告使用 Whisper 加上 Gemma3 4B 用于本地智能音箱应用，实现了约 3 秒的响应时间，并强调在这种情况下速度优于最终的准确性。他们推测新的 Gemma 3n 可能会进一步改善延迟，或在没有 Whisper 等独立 Speech-to-Text (STT) 组件的情况下实现直接推理，从而简化边缘部署。
    - 一些人批评将 Chatbot Arena 分数作为质量指标，认为“将 [Gemma 3n 预览版] 与 Claude Sonnet 3.7 进行比较是荒谬的”，因为预期的性能存在差距，并认为 Gemma 3n 的真正优势可能在于移动或边缘应用，而不是与领先的前沿模型进行正面交锋。
- [**ok google, next time mention llama.cpp too!**](https://i.redd.it/ml66h5yxj02f1.png) ([Score: 142, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1kri7ik/ok_google_next_time_mention_llamacpp_too/)): **该图片来自 Google 展示“Gemma 3n 预览版”的演示文稿。屏幕上的字幕提到了“Ollama、UnSloth 等”，显然是在公开致敬开源语言模型生态系统中这些流行的框架/工具。该帖子的背景是，虽然 Google 提到了 Ollama 和 UnSloth（两者都广泛用于运行和微调 Gemma 等模型），但遗漏了另一个主要的推理库 llama.cpp，这引发了讨论。其技术意义在于社区驱动的开源工具的重要性，以及 Google 在主流发布期间对特定项目的认可。** 评论者强调 UnSloth 的提及实至名归，因为它具有实用性和开发者质量，而其他人则对缺少对 llama.cpp 的认可展开辩论，反映了社区中关于开源项目认可和功劳的持续讨论。
    - 一些参与者注意到 Google 的致谢中遗漏了 `llama.cpp`，尽管它具有技术重要性，并暗示了可能与项目知名度或行业关系有关的原因，同时强调了它如何支撑本地高效运行 LLM 的重大推理进展。
    - 关于 Google 纳入 UnSloth 存在技术讨论，用户强调 UnSloth 使得在消费级硬件上进行模型训练变得更加容易和快速，使其成为寻求微调模型的从业者的杰出工具。
    - 有人提出了关于 Ollama 中 `gemma 3n` 的可用性和标记问题，并对 Checkpoint 术语以及寻求使用最新模型的用户的清晰度进行了审查，表明该平台需要更好的版本透明度。

### 2. Gemma 3 在 llama.cpp 中的技术更新与优化

- [**滑动窗口注意力（Sliding Window Attention）支持已合并至 llama.cpp，显著降低了运行 Gemma 3 的内存需求**](https://github.com/ggml-org/llama.cpp/pull/13194) ([Score: 469, Comments: 76](https://www.reddit.com/r/LocalLLaMA/comments/1kqye2t/sliding_window_attention_support_merged_into/)): **最新的 llama.cpp PR 合并了滑动窗口注意力（SWA）支持，大幅降低了 Gemma 3 等模型的 KV cache 内存需求。该实现引入了** `llama_kv_cache_unified_iswa`**，通过 SWA/非 SWA 层拆分 KV cache，其中 SWA 缓存会在每个 batch 后积极清理陈旧 token，并限制高级缓存操作（如 context shifting）以最小化 token 丢失。Gemma 3 的实际 VRAM 减少量估计在 75-80% 之间，使得在消费级硬件上实现更长的上下文长度成为可能，但高级 KV-cache 功能会回退到全上下文模式，代价是更高的内存占用。核心 KV-cache 逻辑经过重构以提高模块化和可维护性，并简化了整个代码库的 attention 方法。** 热门评论指出 SWA 带来了显著（约 `75-80%`）的 VRAM 减少，并强调了对于 Gemma 等高缓存占用模型的性能提升，尽管由于 iSWA 固有的设计权衡，他们警告 context-shifting 能力有限——这使其最适合非长上下文偏移（non-long-context-shift）的工作负载（如 RAG）。
    - 为 llama.cpp 开发的滑动窗口注意力（SWA）实现将 KV cache 的 VRAM 需求降低了约 `75-80%`（根据初步估计和 PR 评论），这是运行 Gemma 等大模型的一项重大优化。这一变化支持了更长的上下文窗口，且内存占用不会成比例增加。
    - 一个关键的技术警告是，针对 Gemma 的 iSWA 方法目前不支持 KV cache context shifting，这可能会影响需要动态上下文窗口的任务。然而，对于检索增强生成（RAG）等应用，内存占用的减少显著提升了性能。
    - 实际基准测试显示，用户现在可以 offload 更多层（例如在 27B q4 量化模型上从 27 层增加到 39 层），在相同的硬件限制下实现更高的速度和更大的 token 窗口，这实质性地增强了高上下文工作负载的可用性。
- [**Google MedGemma**](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4) ([Score: 173, Comments: 51](https://www.reddit.com/r/LocalLLaMA/comments/1krb6uu/google_medgemma/)): **Google 发布了 MedGemma，这是一系列专门用于医疗 AI 任务的 Gemma 3 模型变体，详见其官方 [Hugging Face 发布页面](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4)。该系列包括一个 4B 多模态模型（集成了 SigLIP 图像编码器，预训练于放射学、组织病理学、皮肤病学和眼科学的脱敏医疗图像）和一个 27B 纯文本模型，两者均针对临床数据进行了微调，并在开放和策划的医疗基准测试中进行了评估。这些模型专为本地推理和进一步微调而设计，技术报告待发布。** 一个值得注意的技术讨论集中在权衡上：小型、可本地化的微调医疗模型与使用大型通用模型处理临床任务的价值。评论者强调，即使在最高准确度的方法之上，本地部署的操作可靠性和自主性（例如抵抗破坏性的上游变更）也是一个关键优势。
    - 一位评论者权衡了部署像 MedGemma 这样的小型微调本地模型与使用最高能力模型之间的利弊。对于医疗用例，可靠性和顶尖性能可能超过对延迟或资源成本的担忧，尽管本地部署可以防止后续模型更改或更新带来的问题。
    - 摘要详情包括 Google 发布了两个版本的 MedGemma：4B 参数的多模态版本（带有预训练于医疗图像类型的 SigLIP 图像编码器）和专门用于医疗文本的 27B 参数纯文本版本，据报道两者均在开放和策划的临床基准测试中进行了评估。MedGemma 支持针对开发者特定的医疗保健应用进行进一步微调，技术报告即将发布。

### 3. OpenEvolve 与 AlphaEvolve 系统开源实现

- [**OpenEvolve: DeepMind AlphaEvolve 系统的开源实现**](https://www.reddit.com/r/LocalLLaMA/comments/1kr9rvp/openevolve_open_source_implementation_of/) ([Score: 112, Comments: 9](https://www.reddit.com/r/LocalLLaMA/comments/1kr9rvp/openevolve_open_source_implementation_of/)): **OpenEvolve 是 DeepMind AlphaEvolve 的开源实现，这是一个基于 LLM 的进化型 Agent，用于在整个代码库中发现和优化算法。其架构包含四个组件：Prompt 采样器（感知上下文/历史记录）、LLM Ensemble（通过兼容 OpenAI 的 API 调用多个模型）、评估器池（通过分布式、带检查点的评估对程序进行评分）以及受 MAP-Elites 启发的程序数据库。基准测试显示，在圆堆积（达到 DeepMind 结果的 99.97%）和函数极小化等任务上，它与 AlphaEvolve 表现接近，进化的程序能够自主发现高级算法技术（例如：** `scipy.minimize` **优化、模拟退火）。广泛的模型基准测试显示，使用 Gemini-Flash-2.0 和 Claude-Sonnet-3.7 的 Ensemble 效果最佳，并确定 Cerebras AI 的 API 是高生成工作负载下显著更快的推理提供商。完整代码和示例可在 [此处](https://github.com/codelion/openevolve) 获取。** 一条热门评论指出，该方法类似于推理时的 Reinforcement Learning，强调了其相对于更静态的搜索或基于预训练的代码生成的工程新颖性。另一条技术背书指出，与之前的版本相比，该框架的功能完整性更高。
    - Specific-Rub-7250 指出 OpenEvolve 似乎利用了 *Reinforcement Learning* 风格的方法，但有趣的是它被应用于推理阶段，这引发了关于在线进化或自适应如何与传统的离线 Reinforcement Learning 流水线集成的疑问。
    - Green-Ad-3964 指出了与 *Genetic Algorithms* 的相似之处，暗示 OpenEvolve 的方法论可能涉及进化或基于种群的搜索机制来优化模型，这与神经进化和 Genetic Programming 中的概念相呼应。
    - SquashFront1303 要求澄清具体实现了哪种进化算法（或替代方案）来取代 DeepMind 专有的 Evolve 算法，强调了技术差距以及对这一关键架构组件进行公开披露的必要性，以实现可复现性和对比。
- [**惊人的演示：John Link 带领 AI Agent 团队使用 Microsoft Discovery 发现了一种不含“永久化学物质”的浸没式冷却液。**](https://v.redd.it/9b7qevfimu1f1) ([Score: 359, Comments: 57](https://www.reddit.com/r/LocalLLaMA/comments/1kqu7dv/mindblowing_demo_john_link_led_a_team_of_ai/)): **John Link 及其团队使用 Microsoft Discovery，利用一组协调的 AI Agent（可能使用了 [microsoft/autogen](https://github.com/microsoft/autogen)），确定了一种浸没式冷却液配方，避免了此类液体中常见的“永久化学物质”。然而，评论中的技术审查表明，所提出的解决方案是氟氯烃 (CFCs)，众所周知，这类物质历史上会对臭氧层造成破坏。** 评论者质疑该发现的新颖性和安全性，担心它复制了过时且对环境有害的化学成分，从而对 AI 驱动发现的实际进展和验证产生了怀疑。
    - 一位评论者指出，AI 生成的解决方案类似于氟氯烃 (CFCs)，并担心这些化合物由于对臭氧层的负面影响而显得过时且对环境有害。该评论质疑了 AI 发现的新颖性和影响，暗示如果 CFCs 确实是主要结果，该方案可能会有显著的监管或环境缺陷。
    - 另一份具有技术背景的贡献强调，Microsoft Discovery 正在利用开源的 “autogen” 框架（[GitHub 链接](https://github.com/microsoft/autogen)），该框架支持多 Agent 协作。这详细说明了驱动该演示的技术栈，对于那些考虑采用类似基于 Agent 的方法进行科学发现的人来说可能很有参考价值。

- [**微软发布了“AI 应用的 USB-C”。我提前 3 天开源了同样的概念——内附证明。**](https://github.com/iluxu/llmbasedos) ([Score: 337, Comments: 75](https://www.reddit.com/r/LocalLLaMA/comments/1kqxa25/microsoft_unveils_usbc_for_ai_apps_i_opensourced/)): **llmbasedos 于 5 月 16 日以 Apache-2.0 协议发布，是一个开源的极简 Linux OS，支持从 USB/VM 快速（亚分钟级）启动，提供基于 FastAPI 的 MCP (Model Context Protocol) 网关，通过 JSON-RPC 向 LLM 开放本地系统功能。其框架允许任何脚本（在 2 行的 cap.json 中定义）被 LLM 应用（ChatGPT, Claude, VS Code）调用，支持离线 (llama.cpp) 和云端 LLM，并可在 Linux, Windows (VM) 和 ARM 设备上运行。从技术上讲，它与微软随后宣布的“AI 应用的 USB-C”构想如出一辙，核心特性包括模块化 MCP 服务器（FileSystem, Sync, Mail, Agent）、ISO 构建脚本和 systemd 集成；外部评估确认其支持搜索/嵌入、rclone/任务管理、IMAP/iCal、工作流执行以及通过 Docker/HTTP 实现的部署扩展性。** 热门技术评论指出，“AI 的 USB-C”/MCP 这一比喻并非微软首创，且已出现在现有技术中，部分人质疑微软对 MCP 概念的使用是否有所不同。对于微软是否可能转向一个快速发布的开源项目中的想法，评论区表达了怀疑，并讨论了其物流可行性。
    - 多位评论者讨论了这一底层概念——即通过 USB 或类似的抽象方式实现可启动或可插拔的 AI 工具和模型——是否具有根本的原创性，并指出这一想法很可能在领域内独立产生。例如，MCP 官方网站已经将其描述为“AI 应用程序的 USB-C 端口”，这表明该类比和方法可能已被从业者广泛认可。
    - 一些用户质疑在 Docker 镜像或类似的便携式框架中打包或托管 LLM (Large Language Models) 的技术新颖性；这被认为是 Docker 生态系统和 AI/ML 部署模式中早已成熟的实践。
    - 关于将开源项目的发布日期与微软的公告进行比较是否有意义，存在技术争论。几位用户指出，像微软这样的大型组织在公开宣布之前很久就会开始开发，因此从工程或产品管理的角度来看，短期的时间巧合或所谓的创意抄袭不太可能相关。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Google Gemini 2.5 Pro & Ultra 模型基准测试与特性

- [**我的天**](https://i.redd.it/kahokh9k3z1f1.png) ([Score: 1202, Comments: 207](https://www.reddit.com/r/singularity/comments/1krazz3/holy_sht/)): **该帖子分享了一张名为“Gemini 2.5 Pro Deep Think”的对比图表，显示 Google Gemini 2.5 Pro 在数学 (49.4%)、代码 (80.4%) 和多模态 (84.0%) 方面超越了 OpenAI 模型。基准测试来源正受到审查，尤其是数学部分，因为评论者注意到这与 MathArena 报告的性能存在差异（Gemini 2.5 Pro 在 MathArena 为 24.4%，而非声称的 49.4%）。对于测试条件是否等效存在技术上的怀疑；MathArena 会对多次运行的表现进行惩罚，这引发了人们的疑问：Google 报告的结果是否使用了不同的、可能更有利的评分方法。** 评论辩论了基准测试的方法论，质疑由于测试规定不同（例如多次运行、MathArena 的惩罚机制）而进行的直接比较的有效性。人们对替代基准测试（如 USAMO）表现出兴趣，并呼吁进行第三方验证，如 Paul Allen 的基准测试。
    - 一位评论者注意到 OpenAI 模型在 USAMO 基准测试分数上的差异：虽然分数来自 MathArena，OpenAI 的 2.5-pro 在 MathArena 上获得了 24.4%，但另一个来源声称是 34.5%。这引发了关于不同运行和报告方法之间基准测试结果一致性和可比性的疑问。
    - 技术讨论强调了 MathArena 如何惩罚多次运行中不稳定的解题表现——一个模型可能在一次运行中解决了问题，但在另一次运行中失败，这会影响平均分数——从而引发推测，即公布的结果可能反映的是最佳运行分数或平均分数，从而影响了跨模型比较。
    - 附图显示了模型在 USAMO 上的基准测试分数直接对比，表明需要对评估方法和结果透明度进行审查，尤其是 48% 的分数被描述为“令人震惊”，可能预示着重大进展或潜在的不一致性。

- [**快讯：Google 赢了。不知道该作何感想**](https://i.redd.it/0d0vbrwb1z1f1.jpeg) ([Score: 738, Comments: 205](https://www.reddit.com/r/singularity/comments/1krao8g/new_flash_google_won_dont_know_how_to_feel_about/)): **该图片是一个按 Arena Score 排序的 AI 模型排行榜，Google 的 Gemini-2.5 模型占据了榜首，超越了来自 OpenAI 和 DeepSeek 的竞争对手。这直观地展示了 Google 在大语言模型 (LLM) 开发方面的技术进步，特别值得注意的是，Google 团队最初发明了支撑这些进步的 Transformer 架构。强调“效率也更高！！！”的评论暗示 Gemini-2.5 以更好的计算效率实现了卓越的性能，表明其在模型缩放 (scaling) 和优化方面取得了进展。** 评论中讨论了 Google 获胜的意义，将其归功于他们的基础研究 (Transformers) 和缩放能力，而一些人则对 Google 目前的公众声誉表示怀疑，因为其其他产品（如搜索）的质量正在下降。
    - 评论者强调，Google 的最新模型表现仅次于 GPT-4 Turbo（被称为 2.5 Pro），这表明 Google 的进步已使其在能力方面极其接近或等同于 OpenAI 的顶级产品。
    - 重点强调了 Google 近期模型的效率，一些人注意到与之前的模型相比，计算资源的使用有了显著改善——特别是 Google 在不依赖外部的情况下进行缩放的能力被视为一项重大的技术优势。
    - 一项讨论指出，Google 在 Transformer 架构研究方面的深厚根基及其内部计算资源是其成功的基石；作为 Transformer 的先驱，他们能够以极少数人能匹敌的规模进行创新和优化。
- [**Google 不再保留了**](https://i.redd.it/el9r8e6woz1f1.jpeg) ([Score: 225, Comments: 64](https://www.reddit.com/r/OpenAI/comments/1kre0k9/google_doesnt_hold_back_anymore/)): **该图片展示了 “Gemini 2.5 Pro Deep Think” 与未具名的 OpenAI 模型在数学 (49.4%)、代码 (80.4%) 和多模态 (84%) 任务中的基准测试分数，Gemini 在所有对比类别中均优于 OpenAI。这突显了 Google 在模型能力方面的实质性进展，特别是在数学和多模态任务中，但也引起了人们对 Gemini 产品高昂成本（250 美元/月）的关注，正如帖子和评论中所指出的。** 评论者争论其实际价值，指出尽管基准测试分数更高，但 OpenAI 的模型（尤其是 o3）在现实世界的工程规划和文档编写中可能会提供更可靠、更一致的结果。几位用户还注意到 Gemini（250 美元/月）和 OpenAI（20 美元/月）模型之间的价格差异，质疑这种比较的公平性。
    - 几位用户强调，虽然像 o3 这样的模型在技术规划和文档编写方面表现强劲，但基准测试往往优先考虑编程能力，可能会忽略结构化写作等用例，据报道在这些场景中 o3 比 Google 或 OpenAI 的产品更可靠、更一致。
    - 出现了一场关于巨大价格差异的讨论，将 Google 的 250 美元/月与 OpenAI 的 20 美元/月进行对比，引发了关于价值以及订阅成本的增加在性能或能力方面（特别是在非编程任务中）是否合理的疑问。
    - 有一些关于主观定性差异的描述：一些人发现 Gemini 2.5 Pro 的输出与 o3 相比缺乏深度或创造力，有人声称 Gemini 在编程方面表现更好（尽管一位用户根据个人经验对此提出了异议），并且 AI Studio 是其唯一的亮点功能。人们还对 Gemini 在研究导向型查询中提供细致或深入答案的能力表示担忧。
- [**2.5 Pro 获得原生音频输出功能**](https://i.redd.it/kzg8bhei1z1f1.jpeg) ([Score: 231, Comments: 22](https://www.reddit.com/r/singularity/comments/1krap7e/25_pro_gets_native_audio_output/)): **该图片记录了一场官方演示，宣布 “2.5 Pro”（推测是 Google 的一款新型或升级版 AI 模型，如巨大的 “G” 标志所示）将具备原生音频输出功能。波形图上的 “Expressive”（富有表现力）一词表明，此次更新的重点是高质量、具有情感细微差别的文本转语音或音频合成。这表明 Google 的语言模型在实现更自然、集成且可能是实时的音频生成方面取得了重大进展。** 评论者对该功能的发布充满期待，询问目前的访问权限，并推测其拥有高度逼真声音的潜力（例如，将其与 Scarlett Johansson 的声音进行比较），表明了对自然且富有表现力的 AI 音频输出的需求。

- 讨论的主要技术焦点在于新的 2.5 Pro 模型的原生音频输出是否不仅支持合成语音输出，还支持音效等额外音频功能。一位评论者特别询问是否可以原生生成 sound FX，这表明人们对标准 text-to-speech 之外更广泛的音频合成能力感兴趣。
    - 另一个涉及的方面（虽然技术性稍弱）是模型提供特定语音输出的能力——例如模仿知名人士（如 'Scarlett Johansson'）的声音，这暗示了对音频输出功能集中可用的语音克隆、定制或适配技术的技术好奇。
- [**$250/月 Google Gemini Ultra | AI 行业最昂贵的方案！**](https://i.redd.it/shseqf1zaz1f1.jpeg) ([Score: 413, Comments: 192](https://www.reddit.com/r/OpenAI/comments/1krc2uy/250mo_google_gemini_ultra_most_expensive_plan_in/)): **该图片展示了 Google 新泄露/营销的 'Google AI Ultra' 方案与现有的 'Google AI Pro' 订阅之间的对比。'Ultra' 方案的特点是提供高级 AI 功能（推测是利用 Gemini Ultra），前三个月价格为 $124.99/月，之后可能为 $250/月，代表了目前消费级 AI 订阅领域中可见的最高定价层级。值得注意的是，它捆绑了 YouTube Premium（价值约 $19/月）和海量的 30 TB 存储空间，同时还提供对 Gemini app、Flow 和 NotebookLM 等高级工具的独占访问权限，且使用限制比 'Pro' 更高。这标志着 Google 打算将 Gemini Ultra 定位为高端 AI 产品以及更广泛的生态系统价值主张，类似于或超过 OpenAI 的 ChatGPT Plus 层级。** 评论者注意到了捆绑价值（例如，“YouTube Premium 和存储空间抵消了一部分成本”），并将其与 ChatGPT Pro 进行了直接对比，但也指出了由于高昂价格和缺乏家庭方案而导致的普及性问题（一些人认为这是 Google 提高采用率的机会）。
    - 几位用户强调，$250/月的 Gemini Ultra 订阅包含 YouTube Premium（价值约 `$19/月`）和 `30 TB` 的存储空间，考虑到 AI 访问之外的捆绑服务，该方案的定位类似于 ChatGPT Pro 等综合性产品。
    - 一位评论者指出了一项关键限制：该方案缺乏可配置性。他们质疑对专业用户（例如，主要需要编程工具而非视频生成的模型用户）的价值，并指出为他们不会使用的功能支付捆绑价格是低效的。
- [**$250 每月...**](https://i.redd.it/nufdz9g5gz1f1.png) ([Score: 108, Comments: 95](https://www.reddit.com/r/ChatGPTCoding/comments/1krctpj/250_per_month/)): **图片展示了价格为 $249.99/月的 "Google AI Ultra" 订阅模式，捆绑了先进的 Gemini 能力、Whisk 和 NotebookLM 的更高使用配额、与 Google 核心服务（Gmail、Docs、Chrome）的深度集成、Project Mariner、YouTube Premium 以及 30TB 的存储空间。该优惠目前仅限美国，计划向国际扩展；核心价值主张是访问 AI 增强的生产力工具和大量的云存储，将其定位为 Google 的综合性高端 AI 生产力套件。** 评论集中在感知价值上，对个人用户的 ROI 表示怀疑（“LLM 让我变得高效，但没那么高效”），而一些人则强调了 YouTube Premium 等理想的附加组件，质疑该价格是否比替代支出（例如外包任务）更具订阅价值。
    - 一位用户对数据上限表示震惊，提到了 "30TB"，这表明该服务可能为 AI 或媒体应用提供极高的数据使用量，并含蓄地质疑了这种 $250/月套餐中数据限制的可扩展性或经济可行性。
- [**所以这基本上证实了（期待“deep think”切换开关 - 对 ultra 仍不确定）**](https://i.redd.it/vwuavv3udu1f1.png) ([Score: 471, Comments: 79](https://www.reddit.com/r/singularity/comments/1kqt89m/so_this_basically_confirms_it_expect_a_deep_think/)): **该帖子讨论了 AI 产品中预期引入的 'deep think' 切换开关，根据上下文和对 Demis Hassabis 的提及，该产品可能来自 Google 的 DeepMind。图片中出现了 Hassabis，并暗示了一项增加推理深度或响应质量的功能，这与 Google 最近强调先进 AI 能力的营销活动相一致。目前尚未确认 'ultra' 模式，但 'deep think' 切换开关已被强烈暗示。** 评论者注意到 Google 转向更深思熟虑的 AI 营销，并以 OpenAI 和 Sam Altman 作为对比。讨论涉及 DeepMind 方法的严肃性以及对即将推出的功能的推测。

- 这是一个关于 Google 历史上在 AI 产品营销方面的困难与其近期转变的技术观察。提到 Demis (Hasabis) 的“预热贴”以及所谓的 “deep think” 开关，暗示即将推出新功能或模式，反映了未来 Google AI 模型中可能的 UI/UX 或微调选项。然而，关于这在产品策略或技术用户控制方面如何与 OpenAI 竞争，仍存在推测。

### 2. Civitai 支付禁令与社区反应

- [**Civitai 被禁止银行卡支付。网站剩余现金仅够维持几个月。敦促在为时已晚之前购买批量包和年度会员**](https://www.reddit.com/r/StableDiffusion/comments/1kqtdx0/civitai_banned_from_card_payments_site_has_a_few/) ([Score: 674, Comments: 422](https://www.reddit.com/r/StableDiffusion/comments/1kqtdx0/civitai_banned_from_card_payments_site_has_a_few/)): **Civitai 宣布，由于决定托管 NSFW 和成人内容，其银行卡支付处理已被禁止，这一消息已得到官方代表声明的证实。该平台目前仅剩几个月的运营现金，并敦促用户在寻求替代支付方案期间购买批量包或订阅年度会员。技术挑战集中在遵守支付处理器的内容政策以及相关的业务连续性风险。** 评论者强调了在托管争议内容的平台上依赖传统支付处理器的风险，引发了对支付基础设施韧性和创作者生态系统中审查制度的质疑。
    - 一条详细评论澄清了支付处理的技术细节：存在两个层面——支付处理器（如 Stripe）和信用卡公司（如 Visa）。即使 Civitai 转向支持成人内容的处理器，所有处理器最终都必须遵守信用卡公司制定的规则。虽然存在支持成人内容的处理器，但它们收取的费用更高，且技术变更（需要代码适配和配置开销）增加了更换处理器的复杂性。
    - 另一条评论强调了灰色市场或成人内容网站更广泛的基础设施脆弱性：除了支付处理，托管、DNS 和搜索引擎索引也受服务条款和潜在业务风险的影响。即使完全转入地下的网站，其用户也面临可发现性和可访问性的挑战——这可能会削弱其作为领先资源枢纽的角色。因此，任何向替代或地下结构的转型都涉及支付系统之外的多层技术和运营风险。
    - Civitai 团队的原始声明确认，他们被移出支付处理器是因为拒绝删除 NSFW 内容，强调了对支持所有类型创作者的持续承诺，并正在寻找替代方案。这凸显了在处理具有争议性的生成式 AI 内容的社区中，平衡平台政策、合规性和服务连续性所面临的持续技术挑战。
- [**[泄露] 说服 Civitai 董事会拒绝 Visa 要求的 PPT 幻灯片**](https://i.redd.it/y97vj6t3bv1f1.png) ([Score: 439, Comments: 75](https://www.reddit.com/r/StableDiffusion/comments/1kqwo59/leaked_the_powerpoint_slide_that_convinced_the/)): **该图像是一个讽刺性的饼图，据称显示了 Civitai 内容的“机密”细分，绝大部分被标记为“色情”（69.3%）和“也是色情 lol”（23.8%），只有极小部分属于“首页”和 Excel 错误。这种恶搞可视化评论了一个业务/运营问题：该平台的内容绝大多数由成人/NSFW 材料组成，这可能解释了为什么拒绝支付处理器（如 Visa）针对用户生成的成人内容提出的合规要求。技术讨论集中在内容审核、支付合规性以及在存在成人内容的情况下对平台商业模式的影响。** 评论者辩论：1) Civitai 据称拒绝 Visa 要求的实事依据，2) 观察到该网站绝大部分内容确实是 NSFW，使得激进的政策转变在商业上具有风险，以及 3) 为什么支付处理器针对 Civitai 这样的平台，而成熟的成人网站似乎不受影响，质疑合规执行的一致性。
    - 一位用户估计，Civitai 约 90% 的用户生成内容是 NSFW，这表明执行 Visa 的限制可能会导致大量受众流失，并可能摧毁平台的商业模式。
    - 针对支付处理器的一致性表达了怀疑：评论者指出，许多成熟的色情网站通过 Visa 处理支付而没有明显问题，这引发了关于为什么 Visa 针对 Civitai 等较小或特定的网站进行执法的问题。

- 另一位用户幽默地指出，很大比例的内容（量化为 '69%' 或 '90%'）是 NSFW，强调了 Civitai 上大量成人内容是关于平台限制和支付处理器合规性辩论的核心。
- [**CivitAI 是否已病入膏肓？是时候联合起来创建一个 P2P 社区网络了吗？**](https://www.reddit.com/r/StableDiffusion/comments/1kr3xzq/is_civitai_on_its_deathbed_time_for_us_to_join/) ([评分: 286, 评论: 198](https://www.reddit.com/r/StableDiffusion/comments/1kr3xzq/is_civitai_on_its_deathbed_time_for_us_to_join/)): **该帖子对 CivitAI 因支付处理问题和有限的运营资金而产生的财务可行性表示担忧，质疑社区是否应该存档模型（包括 LoRA 权重）并转向点对点（P2P）共享架构。讨论集中在快速存档的技术策略、去中心化存储（如 torrents）以及组织逻辑——例如避免重复工作和建立分布式行动的社区清单。** 评论者辩论了像 torrents 这样的 P2P 解决方案是否足够，指出虽然 torrents 可以廉价地存储和分发模型，但它们缺乏 CivitAI 的功能：可搜索的元数据（metadata）、结构化的模型展示、创作者个人资料和财务支持机制。还提出了关于在没有 CivitAI 这样的平台的情况下，如何维持积极的开发和中心化的知识共享的担忧。
    - 几位用户讨论了在 torrents 上托管 AI 模型与在 CivitAI 等中心化平台上托管之间的技术差异。Torrents 提供了去中心化的文件共享，但缺乏诸如可搜索数据库、标签、社区功能以及对模型创作者的直接支持等功能。正如 Herr_Drosselmeyer 所指出的，缺乏这些元数据服务和财务激励限制了 torrents 完全取代 CivitAI 这种 AI 建模社区平台的能力。
    - 人们对在没有推理（inference）/生成功能的情况下托管 CivitAI 替代方案的技术可行性和成本结构感兴趣。这可能会降低托管成本，一些用户推测可以通过加密货币进行资助。然而，复杂性不仅在于带宽，还在于实现能够保持社区、货币化和可发现性的功能，以促进持续的模型创新。
    - 一些用户正积极地存档大型 AI 模型数据集（例如，目前已存档 1.5TB，可用容量为 60TB），以应对潜在的平台流失，这反映了在独立基础设施上分配托管资源的草根努力。此外，像 [Civitasbay.org](http://civitasbay.org/) 这样的替代方案正作为早期 P2P 种子（seeding）解决方案被探索，但可用性和内容发现仍是悬而未决的问题。
- [**持续播种的 AI 模型 torrent 网站，**](https://civitasbay.org/) [**CivitasBay.org**](http://civitasbay.org/) ([评分: 190, 评论: 34](https://www.reddit.com/r/StableDiffusion/comments/1kr0fxc/continuously_seeded_torrent_site_for_ai_models/)): [**CivitasBay.org**](http://civitasbay.org/) **是一个基于 torrent 的 AI 模型文件分发平台（特别是用于 SD 1.5、SD 2.0、SDXL 和 LoRA 微调的** `safetensors` **格式），使用磁力链接（magnet links）实现无需中心化托管的去中心化共享。该网站按文件大小、日期和唯一标识符对模型进行索引，支持对大型生成权重的点对点访问，这对于绕过托管或带宽限制非常有价值。这种方法利用现有的 P2P 基础设施，实现 Stable Diffusion checkpoints 和 LoRA 增强功能等资源的广泛、社区驱动的分发。** 热门评论强调了元数据（metadata，如模型描述、示例图像、触发词或推理设置）的缺失，指出该平台的效用目前对于需要额外上下文才能有效使用的从业者来说受到了阻碍。讨论围绕品牌推广和用户体验展开，并提出了增强文档和 UX 功能的请求。
    - 多位评论者对 [CivitasBay.org](http://civitasbay.org/) 上因缺乏元数据（metadata）而导致的模型可用性提出了技术担忧。他们特别提到了模型描述、示例图像、提示词（prompt）示例和推荐推理设置的缺失，这使得识别分发的 safetensors 文件的功能、预期输出或用例变得困难。这导致了可发现性差，并阻碍了用户将这些模型有效地集成到 ComfyUI 或类似界面的工作流中。

- 与其他 AI 模型存档尝试（例如 [civitaiarchive.com](http://civitaiarchive.com/)）进行了对比，指出虽然通过 torrents 进行持续播种（seeding）和备份模型对于冗余和访问很有价值，但如果没有适当的上下文和文档，此类网站就会变成“仅仅是模型的 Limewire”。强调了使用详细元数据（metadata）对模型进行正确索引对于技术采用和社区使用至关重要。

### 3. 用于科学、创意和自动化的前沿 AI

- [**Microsoft Discovery：AI Agents 在数小时内从创意变为合成新材料！**](https://v.redd.it/tyny38lq6y1f1) ([Score: 558, Comments: 83](https://www.reddit.com/r/singularity/comments/1kr6e4j/microsoft_discovery_ai_agents_go_from_idea_to/))：**Microsoft 展示了用于加速科学研发的 AI Agents，具体用例是发现并物理合成一种用于数据中心的新型、更安全的浸没式冷却剂，以取代对环境有害的“永久性化学物质”。该流程涉及自主文献综述、实验规划、代码生成、在 Azure HPC 上的模拟部署以及物理合成——所有这些都在数小时/数天内完成，而非数年，并最终通过现场演示展示了该冷却剂在实际运行中保持硬件冷却。这代表了一个实用的闭环系统，AI 不仅能提出新型化合物，还能执行端到端的材料发现和合成，与 Google 的 GNoME 等努力相平行，但声称具有独特的新型合成产出。** 一位具有技术背景的评论者对过去的说法（如 GNoME）表示怀疑，当时 AI 发现的材料后来被证明并非首创，并指出 AI 驱动的药物研发公司（如 Exscientia 和 BenevolentAI）未能将预测转化为临床或市场成功。另一条评论提出了一个技术点，即这一突破可能不仅仅是组合搜索（combinatorial search），暗示了除了暴力探索之外，还存在更先进的决策或生成方法。
    - 一位评论者通过引用用于材料发现的 GNoME AI，为所声称的突破提供了批判性的背景信息，指出尽管 Google 报告了其 AI 合成了约 40 种新材料，但一项外部分析 (https://doi.org/10.1103/PRXEnergy.3.011002) 发现没有一种是真正的新材料。此外，AI 驱动的药物研发工作（由 Exscientia 和 BenevolentAI 开展）在临床试验中面临了备受瞩目的失败，突显了 AI 科学研究结果在炒作与实际交付之间的差距。
    - GrapplerGuy100 将早期的组合搜索技术，如 AlphaEvolve（在蛋白质设计中使用大型搜索空间和快速验证）中的技术，与当前的方法进行了对比。他们对新的 Microsoft 系统是使用类似的组合方法，还是代表了一种本质上不同的、可能更先进的材料发现机制表示不确定。
- [**Google 展示 Project Astra 控制你的 Android 手机**](https://9to5google.com/2025/05/20/project-astra-android/) ([Score: 254, Comments: 76](https://www.reddit.com/r/singularity/comments/1krba5z/google_shows_project_astra_controlling_your/))：**Google 展示了 Project Astra，其具有先进的语音和视觉 AI 能力，可执行全面的 Android 设备控制，如最近的视频演示所示。该实现突出了对复杂用户指令的实时理解和执行，标志着多模态设备端 Agent 性能的飞跃，并暗示近期将在消费级设备中部署。** 评论对从去年的演示到实际实现的快速进展表示惊讶，专家指出这在技术上超越了去年的原型以及 Apple 等竞争对手。一些人预见 Project Astra 包含在 Android 中将成为移动操作系统竞争格局的主要颠覆者。
    - 针对 Google 大规模提供 Project Astra 服务的能力提出了技术担忧，因为上下文感知（context-aware）的设备端 AI 具有极高的计算需求，特别是像后台第三方协商这样的功能。评论者质疑 Google 的 TPU 和基础设施是否能够合理地处理数亿并发用户而不发生性能下降，暗示了重大的扩展和服务交付挑战。

- [**VACE Extension 是超越 FLF2V 的下一代技术**](https://v.redd.it/lmpa3kx54v1f1) ([Score: 147, Comments: 27](https://www.reddit.com/r/StableDiffusion/comments/1kqw177/vace_extension_is_the_next_level_beyond_flf2v/)): **该帖子对比了 FLF2V 的标准帧插值（成对进行，如 1→2, 2→3，常导致时间不一致和不自然的动作反转）与新型 VACE Extension 方法。在 VACE 中，用户指定的帧作为有序的“checkpoints”。这种方法最近已在 Wan2.1 等模型中实现（目前限制为 81 帧），能生成全局时间一致的视频，平滑地穿过所有 checkpoints，实现类似于传统动画中间帧的高质量动画。通过重叠技术可以实现超过 81 帧的进一步扩展，如[最近的 WACE 14b 示例](https://www.reddit.com/r/StableDiffusion/comments/1kqcomx/video_extension_using_vace_14b/)所示。技术工作流详情分享在[这里](https://scrapbox.io/work4ai/VACE_Extension%E3%81%A8FLF2V%E3%81%AE%E9%81%95%E3%81%84)（日语）。** 一位评论者主张以较低帧率（如 15fps）生成，然后通过插值上采样到 60fps 以提高效率，并对原生追求 24fps 的模型表示担忧。另一位用户询问了 VACE 与 Wan LoRA 之间的互操作性，质疑是否需要单独训练，这表明用户对模型模块化和工作流集成有浓厚兴趣。
    - 一位用户主张以 15fps 生成视频以提高效率，指出这允许快速合成，随后再通过时间上采样（如使用插值）获得更清晰的 60fps 输出，并认为其他模型以 24fps 为目标对于这种工作流来说并非最优。
    - 讨论中提到了 VACE Extension 中“最后一帧漂白（last frame bleaching）”伪影的处理，即序列末尾会出现色彩饱和度下降。尽管进行了后处理，该问题依然存在，有人建议这在卡通中不太明显，因为卡通可以重新进行调色，且质量问题比专注于写实的工作流更少。
    - 一位评论者质疑 VACE 模型与 Wan LoRA 的兼容性，询问是否可以使用现有的 LoRA，还是需要重新训练，突出了模型扩展与 LoRA 集成方面的开放性问题。
- [**DeepMind Veo 3 Sailor 生成的视频**](https://v.redd.it/us18oc0gpz1f1) ([Score: 580, Comments: 152](https://www.reddit.com/r/singularity/comments/1kre3qp/deepmind_veo_3_sailor_generated_video/)): **该帖子讨论了由 DeepMind Veo 3 的“Sailor”提示词生成的视频样本，但由于 403 Forbidden 错误，视频的外部链接 (https://v.redd.it/us18oc0gpz1f1) 无法访问，导致无法对输出进行直接的技术分析。一条技术评论指出，“长胡子的老人”是 Veo 中反复出现且可能受到限制的生成内容，暗示模型中可能存在提示词工程限制或内容过滤偏差。一般性讨论推测了从单一提示词生成完整的 AI 导演电影的技术临近性，表明 Veo 等生成式视频模型进展迅速。** 评论指出 Veo 存在内容限制（偏向于生成某些特定主题，例如“长胡子的老人”），而其他人则表达了对电影制作即将面临技术变革的感受，但由于视频无法观看，缺乏详细的技术评估。
    - 一位评论者预测，AI 驱动的视频质量将在 1-2 年内达到近乎完美的水平，称目前已完成*至少 80%*，暗示 DeepMind Veo 等生成式视频模型进展神速。
    - 围绕 AI 从单一提示词生成整部电影的即时性能力展开了讨论，这意味着多模态生成系统将取得重大进展，并可能对传统视频制作流程产生颠覆。
- [**Veo 3**](https://v.redd.it/kuzmuiohcz1f1) ([Score: 359, Comments: 98](https://www.reddit.com/r/singularity/comments/1krce1m/veo_3/)): **帖子标题“Veo 3”指的可能是 Google 在 I/O 2024 上发布的先进生成式视频 AI 模型 Google Veo 的最新版本。由于外部链接无法访问，帖子中未提供技术细节或基准测试。评论中没有提供新的技术信息，但有一条引用了自早期生成式视频（如“Will Smith eating spaghetti”）以来的飞速进步。** 热门评论反映了对近期生成式视频模型对动画和好莱坞等行业颠覆性影响的担忧，提到了该领域进步之快，并推测了其对知名工作室（如 Pixar、DreamWorks）和传统好莱坞制作的影响。

- 一位评论者预测，在 5 年内，电影中的一些短场景可能会由 AI 生成，而不是传统的 CGI 渲染，这暗示了 VFX 工作流正向基于机器学习的生成模型转变，以处理特定类型的视觉内容。这可能会显著降低特定场景创建任务的成本和时间。
- [**Veo 3 单口喜剧**](https://v.redd.it/uagkfp8u702f1) ([评分: 303, 评论: 79](https://www.reddit.com/r/singularity/comments/1krgm19/veo_3_standup_comedy/)): **该帖子讨论了一个由 Veo 3 生成的模拟单口喜剧视频，评论者指出其真实感超越了以往的恐怖谷限制。技术讨论集中在 Veo 3 能够令人信服地合成视觉和行为线索，以及细微的音频细节（如“对着麦克风呼吸”的声音），从而增强真实感。Veo 是 Google DeepMind 开发的一款视频生成模型（参见其 [研究页面](https://deepmind.google/technologies/veo)），因其在生成式视频语境中对人类表现的细致复制而受到关注。** 专家评论者对该模型的高保真度感到惊讶，表示它与真实的单口喜剧片段几乎无法区分，一些人甚至推测该片段是生成的还是真实的录音，这展示了该模型在媒体真实感方面的进步。
    - 评论者指出 Veo 3 的生成效果高度逼真，捕捉到了笑声前“对着麦克风呼吸的声音”等细微音频细节，展示了先进的时间连贯性和微妙的音画同步。
    - 讨论中提到了语音、笑声和环境音效的无缝整合，这在 2025 年之前原本不被预期会出现在 AI 多模态模型中，表明生成式视频+音频系统取得了超出预期的跨越式进展。
    - 一位用户强调了区分 AI 生成内容与真实片段的难度，认为 Veo 3 的结果已处于或已经超越了合成媒体真实感中的“恐怖谷”。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1：Google 的 AI 攻势与新模型冲击**

- **Google 爆发式发布 Gemma 3 和 Gemini 2.5！**：Google 发布了 **Gemma 3** 系列，包括 **Gemma 3N**（[技术报告](https://ai.google.dev/gemma/docs/gemma-3n)，[文档](https://ai.google.dev/gemma/docs/gemma-3n)），最初仅限美国地区；同时发布了 **Gemini 2.5 Flash**（[OpenRouter 链接](https://openrouter.ai/google/gemini-2.5-flash-preview-05-20)），并在 Google I/O 上预告了 **Gemini 2.5 Pro Deep Think**、**Veo 3** 和 **Imagen 4**，尽管许多功能仍受限。LMArena 成员认为 Google I/O 像是一个“填充章节”，而 OpenRouter 用户则根据 [Google 关于 Gemma 3n 的博客](https://developers.googleblog.com/en/introducing-gemma-3n/) 和 Chatbot Arena 的偏好，讨论 **Gemma 3n-4B** 是否有潜力媲美 **Claude 3.7**。
- **Meta 的 Llama 3.3 在 OpenRouter 独家首发！**：根据 [OpenRouter 的 Llama 发布推文](https://x.com/OpenRouterAI/status/1924899161469571337)，Meta 的新 **Llama** 提供商推出了 **3.3 8B 模型**，在 OpenRouter 上独家开放访问且可免费开始使用。此举为开发者提供了一个极具成本效益的途径来实验 Meta 最新的语言模型产品。
- **IBM 和 Mistral 悄然发布重磅消息，Qwen 继续前进！**：在 Google 占据头条新闻的同时，**IBM** 悄然推出了 **Granite 4.0**，**Mistral** 发布了 **Mistral Small 3.1**，两者都在 Unsloth AI 社区引起了轰动。与此同时，关于 **Qwen** 系列的讨论仍在继续，包括在 LM Studio 中为 **Qwen 3** 进行 Speculative Decoding（投机解码）的性能优化，以及 Aider 服务器中关于 **Qwen 2 35B** 的基准测试讨论，详见 [Paul Gauthier 关于 Qwen3 的博客文章](https://aider.chat/2025/05/08/qwen3.html)。

**主题 2：变革 AI 工具和开发者平台**

- **Unsloth 在 Google IO 大放异彩，KernelLLM GGUF 发布！**：Unsloth 在 **Google IO** ([UnslothAI 关于 Google IO 的推文](https://x.com/UnslothAI/status/1924977922915631443)) 展示后备受关注，团队还发布了 **KernelLLM GGUFs** ([Hugging Face 上的 KernelLLM GGUF](https://huggingface.co/unsloth/KernelLLM-GGUF)) 以及一个用于长音频生成的“修复版” **Sesame notebook** ([UnslothAI 关于 Sesame notebook 的推文](https://x.com/UnslothAI/status/1924848135991656603))。这些发布为开发者提供了更高效的模型训练和部署工具。
- **LM Studio 和 Modular 的 MAX 增强了本地 AI 能力！**：LM Studio 用户通过调整 **RoPE Frequency Scale** 并利用 **Llama.cpp** 的 **Sliding Window Attention (SWA)** ([Llama.cpp SWA GitHub pull](https://github.com/ggml-org/llama.cpp/pull/13194)) 来大幅降低显存占用，KV cache 从 **1700MB** 降至 **348MB**。Modular 的 **MAX** 平台承诺提供全栈推理控制，并邀请企业用户联系 [tim@modular.com](mailto:tim@modular.com) 获取大规模解耦推理解决方案。
- **Perplexity 和 OpenRouter 持续迭代，但 API 异常依然存在！**：Perplexity 推出了 **Perplexify Me** 等新功能 ([Perplexity 5 月 19 日更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-19th))，但 Perplexity AI Discord 的用户注意到其 API 输出可能不如 Playground 直接，并报告 **Deep Research API** 请求显示为 **sonar-reasoning-pro** ([Perplexity API GitHub issue #320](https://github.com/ppl-ai/api-discussion/issues/320))。OpenRouter 通过为供应商和量化版本添加 slug 简化了开发体验 ([OpenRouterAI 关于 slug 的推文](https://x.com/OpenRouterAI/status/1924868065910518226))。

**主题 3：AI Agent 的兴起：编程、研究及更多领域**

- **Google 的 Jules Agent 进入编程领域，Manus 助力网站构建！**：Google 推出了异步编程 Agent **Jules** ([Jules by Google 网站](https://jules.google/))，引发了热烈讨论和排队预约；同时 [**Manus.im**](http://manus.im/) 发布了一个拥有独立计算机的 AI Agent，可用于构建网站和进行研究 ([Manus.im 邀请示例](https://manus.im/invitation/CWYMVAAXR6HZ)，[Manus 使用案例合集](https://manus.im/usecase-official-collection))。这些 Agent 展示了从辅助软件开发到自动化任务完成的多样化应用。
- **OpenEvolve 复现 DeepMind 壮举，利用 LLM 进化代码！**：开源发布的 **OpenEvolve** ([GitHub 上的 OpenEvolve](https://github.com/codelion/openevolve), [OpenEvolve HuggingFace 博客](https://huggingface.co/blog/codelion/openevolve)) 是 Google DeepMind AlphaEvolve 的一种实现，它以 **99.97%** 的准确率复现了圆堆积（circle packing）结果，并将随机搜索进化为模拟退火，引起了巨大轰动。该工具使用 LLM 来发现、优化和进化整个代码库，并在 Unsloth AI、Eleuther、Nous Research AI、HuggingFace 和 GPU MODE 的 Discord 频道中引发了讨论。
- **MCP 生态扩展，新增 A2A 桥接和钱包集成！**：Model Context Protocol (MCP) 迎来了新进展，包括一个连接 **MCP 与 A2A 协议 Agent** 的开源服务器 ([GitHub 上的 A2A-MCP-Server](https://github.com/GongRzhe/A2A-MCP-Server))，允许 Claude 与 A2A Agent 交互。此外，**TokenPocket** 发布了 **Wallet MCP** ([GitHub 上的 Wallet MCP](https://github.com/TP-Lab/mcp-marketplace/tree/master/apps/wallet-mcp))，用于将 AI 客户端与加密用户钱包集成，该项目在 MCP (Glama) Discord 中进行了讨论。

**主题 4：挑战性能极限：模型优化与评估**

- **Speculative Decoding 和 SWA 为本地模型注入强劲动力！**：LM Studio 用户在对最大草稿大小 (3) 和草稿概率 (0.8) 等设置进行*大量微调*后，利用 Speculative Decoding 将 **Qwen 3** 的性能提升了 **12%**。他们还通过 Llama.cpp 中的 **Sliding Window Attention (SWA)**（[Llama.cpp SWA GitHub pull](https://github.com/ggml-org/llama.cpp/pull/13194)）大幅削减内存占用，将 Gemma 4B 15K 模型的 KV cache 从 **1700MB** 减少到 **348MB**。
- **NVIDIA 与学术界探究模型内部机制与效率！**：在 Yannick Kilcher Discord 中关于**语言模型物理学**（[Part 3.1 知识存储论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633)，[Part 3.2 知识操纵论文](https://ssrn.com/abstract=5250621)）的讨论强调了 LLM 在检索方面的优势，但在没有 CoT 的情况下操纵知识的能力较弱。在 GPU MODE 中，**FSDP2** 被指出比 FSDP1 使用更少的 VRAM 但速度更慢，具体基准测试如在 RTX 4000 Ada x4 上，*fsdp2: 11.599G, 13.39s/step* 对比 *fsdp1: 15.801G, 7.3s/step*。
- **Tinygrad 悬赏驱动硬件优化，Cutotune 实现 CUDA 自动调优！**：tinygrad 社区通过悬赏 **Flash Attention**（最初针对 7900XTX）和 **BERT 训练**来挑战硬件极限，旨在 Nvidia/AMD GPU 上超越普通 Attention，详见 [tinygrad 的 X 置顶帖](https://x.com/__tinygrad__/status/1910185557642367390)。同时，GPU MODE 成员介绍了 **cutotune**，这是一个配合 **PyTorch compile** 使用的 CUDA kernel 自动调优器，旨在提供易扩展性。

**主题 5：AI 的社会脉动：伦理、Slop 与社区动态**

- **AI 'Slop' 的定义及其影响引发激烈辩论！**：在 EleutherAI 频道中，“AI slop”（AI 垃圾内容）一词引发了讨论，定义范围从*低工作量*的 AI 输出到未能通过图灵测试的内容，参考了 [AI slop 维基百科页面](https://en.wikipedia.org/wiki/AI_slop)和[脆弱性失败论文](https://arxiv.org/abs/2410.05229v1)。辩论质疑了 “slop” 是 AI 固有的属性，还是更广泛的内容质量问题。
- **研究人员发现 LLM 产生社交生活、偏见并推动变革！**：Nous Research AI 讨论了一篇发表在 [Science Advances 上关于 LLM 社交习俗的论文](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)，研究揭示了去中心化的 **LLM 群体**会自发产生社交习俗，并且即使初始个体没有偏见，群体也会表现出强烈的**集体偏见**。研究还表明，对抗性 LLM Agent 可以在这些群体中煽动社会变革。
- **从回声室到丑小鸭：驾驭 AI 舆论！**：Latent Space 成员注意到 Hacker News 和 Reddit 等平台上反 AI 情绪的兴起，思考这些平台是否代表了回声室。与此同时，DSPy 发布了一条关于“丑陋”事物的神秘 [X 帖子](https://x.com/dspyoss/status/1924838188289556966)，以此调侃其核心哲学：*如果你理解了这一点，你就理解了 DSPy 的全部*，暗示了对 AI 开发中非传统方式的欣赏。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **特殊 Token 引发对 Thinking 标签的探索**：成员们探索了**特殊 Token** 的使用，并将其与 [Anthropic 的 thinking 标签](https://chatgpt.com/share/682cad13-b340-8005-a728-e57349029018)进行对比，指出它们类似于 `<antml:thinking>`。
   - 讨论围绕理解它们在 AI 模型中的目的和实际应用展开。
- **Google 发布 Gemma 3，仅限美国**：Google 发布了 **Gemma 3**，并发布了 [Gemma 3N 技术报告](https://ai.google.dev/gemma/docs/gemma-3n)，该模型最初仅对美国用户开放。
   - 初步反应指出可用性受限，引起了一些国际用户的不满。
- **Google I/O 演示多于交付**：Google I/O 活动让一些人感到失望，被视为*填充剧集*，尽管宣布了 **Veo 3**、**Imagen 4** 和 **Gemini 2.5 Pro Deep Think**，但立即发布的成果有限。
   - 许多功能仅限于受信任的测试人员或美国用户，延迟了更广泛的访问。
- **ChatGPT Pro 在价值之战中完胜 Gemini Ultra**：社区更倾向于 **ChatGPT Pro**，而非 Google 每月 **250 美元的 Gemini Ultra 计划**，理由是前者具有更好的价值主张。
   - 一些社区成员推测，到 **Grok 9** 发布时 **Google 可能会破产**，这表明了对 Google 当前 AI 策略的怀疑。
- **本地 AI 尝试零样本贪吃蛇游戏**：用户尝试在 **LM Studio** 中使用 **Qwen 3 4B/8B** 进行零样本生成实验。
   - 挑战在于 VRAM 有限，导致 Token 生成缓慢，一位用户报告*可能每秒只有 5 个左右 [tokens]*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布令人惊叹的新功能**：Perplexity 宣布了几项新功能：**Perplexify Me**、**F1 实时排名与解说**、**回答中的 F1 比分**、**改进的文件附件**以及**侧边栏快捷方式**，这些内容记录在他们 [5 月 19 日的更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-19th)中。
   - 公告中尚未明确 *Perplexify Me*、*文件附件*和*侧边栏快捷方式*的具体功能和详细改进。
- **Gemini 2.5 是 Google 的主力模型**：Google 的 **Gemini 2.5 Flash**（5 月 20 日版本）凭借其内置的“思考”能力，在**高级推理、编程、数学和科学任务**中表现出色。
   - 它提供了更高的准确性和更细致的上下文处理能力，将其定位为最先进的“主力模型（workhorse model）”。
- **Grok 是无限免费推理的理想选择**：对于寻求强大且**免费、无限推理能力**的 AI 模型用户来说，**Grok** 正成为首选。
   - 用户形容 Grok 具有一种“随和的黑人兄弟范儿（chill black dude energy）”，这可能也是其受欢迎的原因之一。
- **Perplexity Playground 表现优于 API？**：用户反馈，即使在调整了 `top_p` 和 `top_k` 值后，**Perplexity Playground 的输出**也比 **API** 更直接、更准确；并指出添加“示例消息（sample example message）”可以改善 API 结果，但会增加 Token 使用量。
   - 用户还讨论了 **Perplexity API** 是否通过 OpenAI Python 库支持 **OpenAI JSON schema**，但调查结果尚未公布。
- **Deep Research API 是披着 Reasoning Pro 外衣的狼？**：一位用户报告称，通过 API 发出的 **Perplexity Deep Research** 请求在仪表盘中显示为 **sonar-reasoning-pro**，并为此提交了一个 [GitHub issue](https://github.com/ppl-ai/api-discussion/issues/320)。
   - 用户还反映在使用 Perplexity API 时经常遇到**超时问题**，尤其是在研究目标公司时，这表明可能需要进行更小批量的研究。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 在 Google IO 亮相！**：Unsloth 通过[这条推文](https://x.com/UnslothAI/status/1924977922915631443)在 **Google IO** 活动中展示，社区成员在通过 **Gemma+Unsloth+Collab demo** 发现 Unsloth 后表示非常兴奋。
   - Unsloth 团队还发布了 **KernelLLM GGUFs** 并链接至 [Hugging Face](https://huggingface.co/unsloth/KernelLLM-GGUF)，同时更新了一个“基本修复”的 **Sesame notebook**，允许生成超过 10 秒的音频，可通过[这条推文](https://x.com/UnslothAI/status/1924848135991656603)访问。
- **Mistral 和 IBM 悄然采取行动**：成员们关注到了 **Mistral Small 3.1** 的发布，而 **IBM** 也悄悄推出了 **Granite 4.0**，引发了关注；此外还有成员分享了一些“疯狂的基准测试”。
   - 一位成员分享了 [Qwen2.5 VL 的 Ollama 库](https://ollama.com/library/qwen2.5vl)链接。还有成员提到，在 **Gemini Diffusion** 领域，[他们并不是先行者](https://x.com/karpathy/status/1894923254864978091)。
- **OpenEvolve 投入行动！**：一位成员宣布发布 [OpenEvolve](https://github.com/codelion/openevolve)，这是 **Google DeepMind AlphaEvolve 系统**的开源实现，详情见[这篇博客文章](https://huggingface.co/blog/codelion/openevolve)。
   - 作者成功在圆填充（circle packing）问题上复现了 DeepMind 的结果（**匹配度达 99.97%！**），并将随机搜索进化为模拟退火算法，使用 **LLM ensemble 方法**提升了结果并实现了多目标优化。
- **成员在合并模型和 PPO 方面遇到困难**：用户报告称，合并 **4-bit 模型**并保存为 **16-bit** 可能会导致性能问题。还有人表示，使用 PPO 训练存在挑战，特别是在 VRAM 占用和奖励函数限制方面。
   - 一位用户建议“在尝试合并时不要加载 4bit 或 8bit”，并推荐了一个概述此过程的 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API 在没有 Web 前端的情况下访问受限**：用户发现，虽然 **LM Studio API** 支持在本地托管 LLM 的 API 访问，但与 **Stable Diffusion** 不同，它缺少 Web 前端。
   - 为了实现 Web 集成，用户必须使用独立的前端来连接 **LM Studio API**。
- **在 LM Studio 中配置 RoPE Frequency**：成员可以通过模型加载器旁边的设置齿轮来调整 **LM Studio** 中已加载模型的 **RoPE Frequency Scale**。
   - 也可以修改 **gguf** 格式来包含它，因为 **ollama** 和 **llama.cpp** 都支持该设置。
- **Qwen 3 通过 Speculative Decoding 获得速度提升**：用户通过使用 speculative decoding 使 **Qwen 3** 实现了 **12% 的性能提升**，在经过“大量折腾”后，通过将 max draft size 设置为 3 并将 drafting probability 提高到 0.8 完成了配置。
   - 增加 min draft size 只会增加耗电量和生成时间。
- **LMStudio API 揭晓模型卸载功能**：**LM Studio API** 支持使用 `model.unload()` 函数卸载模型，详见[官方文档](https://lmstudio.ai/docs/cli/unload)。
   - 此函数是 API 的一部分，不需要创建单独的函数。
- **Sliding Window Attention 在 Llama.cpp 中大幅降低内存占用**：**Sliding Window Attention (SWA)** 现已在 **Llama.cpp** 中可用（[github.com/ggml-org/llama.cpp/pull/13194](https://github.com/ggml-org/llama.cpp/pull/13194)），并可能很快在 **LM Studio** 中推出，以减少内存使用，特别是针对长上下文。
   - 在初步测试中，对于 **Gemma 4B 15K** 上下文模型，启用 **SWA** 将 KV cache 占用从 **1700MB** 降至 **348MB**，在使用 **Q8_0 quantization** 时进一步降至 **184MB**。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 添加 Provider 和 Quantization Slugs**：根据 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1924868065910518226)，OpenRouter 宣布 **providers** 和 **quantizations** 现在有了 slug，从而提升了开发者体验。
   - 这一变化旨在为在该平台上使用各种模型和量化方法的开发者简化集成和管理。
- **Google 在 OpenRouter 上发布 Gemini Flash 2.5**：Google DeepMind 发布了 **Gemini Flash 2.5**，可在 OpenRouter 上的 [google/gemini-2.5-flash-preview-05-20](https://openrouter.ai/google/gemini-2.5-flash-preview-05-20) 进行测试。
   - 此版本为开发者提供了抢先体验 Google 最新模型以构建快速高效 AI 应用的机会。
- **Meta 在 OpenRouter 上独家提供 Llama**：根据 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1924899161469571337)，Meta 新的 **Llama** provider 现已上线，在 OpenRouter 上独家开放访问，且可以免费开始使用，其中包括一个新的 **3.3 8B model**。
   - 此次合作通过具有成本效益的方式，为开发者提供了实验和利用 Meta 语言模型的途径。
- **Google 的 Gemma 3n 模型可能与 Claude 3.7 旗鼓相当**：根据[这篇博客文章](https://developers.googleblog.com/en/introducing-gemma-3n/)，新的 Gemma 模型 **Gemma-3n-4B** 据称与 **Claude 3.7** 一样出色。
   - 聊天中的成员对这一说法表示怀疑，因为该结论源自 Chatbot arena 的用户偏好。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 讨论 Discord 机器人表情回应**：**EleutherAI** 的成员考虑实现一个 Discord 机器人来打击垃圾信息，提议使用速率限制（rate limits）和管理员提醒（mod pings）而不是直接删除，以防止滥用，并考虑使用仅限管理员的回应来标记内容。
   - 他们建议使用更通用的回应，仅计算常驻用户的回应，并使用私有的仅限管理员频道来简化违规检测，并避免*不可逆的删除*。
- **AI 'Slop' 定义引发辩论**：成员们辩论了 AI 生成内容中 'slop' 的含义，定义范围从*低质量（low effort）*、*理解不当的材料*到*未能通过图灵测试*的内容，并引用了 [AI slop 维基百科页面](https://en.wikipedia.org/wiki/AI_slop)。
   - 讨论涉及了 slop 是否本质上与 AI 相关，一些人认为人类生成的内容也可以是 slop，并引用过度制作或无意义的输出作为例子，以及 [脆弱性失效（fragility failures）](https://arxiv.org/abs/2410.05229v1)。
- **马毅（Yi Ma）对当前 ML 现状表示不满？**：成员们观看了 [马毅的 ICLR 演讲](https://www.youtube.com/watch?v=dVhMiMk1riQ) 并讨论了他的第一性原理方法，该方法将工作建立在信息论和控制论的基础上，与当前模型中盛行的启发式方法（heuristics）形成对比。
   - 一位成员最初觉得演讲节奏较慢，但赞同其中的观点：*仅靠压缩是不够的；组织才是关键*。
- **DeepMind 的 AlphaEvolve 被复现**：一位成员发布了 **OpenEvolve**，这是 Google DeepMind 的 AlphaEvolve 的开源实现，它使用 LLM 演化整个代码库以发现和优化算法，并复现了 DeepMind 在圆堆积（circle packing）上的结果。
   - 发布内容包括 [GitHub](https://github.com/codelion/openevolve) 和 [博客文章](https://huggingface.co/blog/codelion/openevolve)。
- **成员询问 Gemini Diffusion 提示词**：一位成员发起了关于 Google **Gemini Diffusion** 模型的讨论。
   - 他提出为任何无法访问 **Gemini Diffusion** 的人代运行提示词。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的工具限制导致对话停滞**：用户报告称，在 Cursor 中达到 **25 个工具限制**并在继续前接受会导致对话会话终止，正如 [Reddit 帖子](https://www.reddit.com/r/cursor/comments/1kqj7n3/cursor_intentionally_slowing_nonfast_requests/) 中所述。
   - 这个问题可能与 Cursor 处理非快速（non-fast）用户的请求限制方式有关。
- **DeepSeek-R1T-Chimera 打破提示词循环**：**DeepSeek-R1T-Chimera** 模型是 **R1** 和 **V3** 之间的微调版本，据称是唯一能够打破用户提示词测试中循环的免费模型，可在 [Hugging Face](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) 上获取。
   - 该模型独特的架构使其能够处理复杂的提示词，而不会陷入重复循环。
- **MCP 刷新令用户恼火**：用户对频繁需要刷新 **MCP** 表示沮丧，因为它们的状态会变黄，这消耗了大量的请求数。
   - 持续的刷新中断了工作流并降低了使用 **MCP** 的效率。
- **Gemini 的思维过程在演变**：用户观察到 **Gemini 的思维过程**发生了转变，模型现在总结其思维过程，而不是显示实际步骤，这使得编写有效的提示词和项目规则变得具有挑战性。
   - 这一变化影响了根据模型的内部推理来微调提示词的能力。
- **O3 Pro 即将发布？**：关于 **O3 Pro** 发布的猜测正在增加，原因是一位前 OpenAI 员工的推文暗示可能在 6 月发布。
   - 爱好者们正热切期待官方确认以及关于 **O3 Pro** 新功能和能力的细节。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 承诺全栈统治力**：当被问及与 **Fireworks.ai**、**Together.ai** 和 **Groq.com** 等对手的竞争时，Modular 团队声称 **Max** 提供了对技术栈的端到端控制，从而在性能、延迟、准确性和可编程性方面实现了更强的掌控。
   - 他们邀请用户通过 tim@modular.com 联系他们，以讨论涉及大规模解耦推理（disaggregated inference）的企业级解决方案。
- **Cosmos：一个吞噬 GPU 的世界模型**：**Cosmos** 被描述为一个结合了物理引擎和视频模型的“世界模型（world model）”，对 **VRAM** 有极高的需求。
   - 官方澄清说，**MAX** 的重点在于 **NN model**（神经网络模型），而不是整个数据流水线或底层数据流。
- **`fn() -> raises` 语法引发争论**：成员们讨论了 `alias func = fn()` 这种语法对于 **Mojo** 是否是个好主意，尽管解析器（parser）已经接受了它。
   - 这种写法是为了与其他返回语句保持一致，但与其它语言及其类型签名要求相悖，详见[此代码片段](https://gist.github.com/moduland/e9f85376a973ef3f5f4648d234081f9d)。
- **Pixi 取代 Magic 用于 GPU 谜题**：Modular 团队在 Mojo GPU 谜题中弃用了 **Magic**，转而使用 **Pixi**。Magic 实际上只是 Pixi 的一个封装，详见[此 commit](https://github.com/modular/mojo-gpu-puzzles/commit/d8270a7a7c1cf8a99c71cd7e25dbb8ca191a66f9)。
   - Modular 团队在 Mojo GPU 谜题中弃用了 **Magic**，转而使用 **Pixi**，Magic 只是 Pixi 的包装器。
- **误报困扰 Mojo 25.3 版本**：有成员报告 **25.3** 版本中关于未使用赋值的误报警告增多，具体表现为 `warning: assignment to 'suggested_max_length' was never used; assign to '_' instead`，详见[此 Bug 报告](https://github.com/modular/modular/issues/4642)。
   - Mojo 编译器在仅引用变量的别名值时，会错误地将变量标记为未使用。这是 **def** 函数声明的一个问题，通过切换到 `fn() -> raises` 可以解决该误报，这一现象已得到证实。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 推出移动端应用**：NotebookLM 发布了适用于 [Android](https://apps.apple.com/us/app/google-notebooklm/id6737527615) 和 [iOS](https://blog.google/technology/ai/notebooklm-app/) 的移动应用，其特色功能包括离线 **Audio Overviews**（音频概览）、交互功能和内容共享。
   - 初步反馈显示，与网页版相比，该应用**缺乏部分功能**（如“发现来源”），且用户对折叠屏设备缺乏支持表示担忧。
- **Audio Overviews 获得长度控制功能**：用户现在可以调整 **Audio Overviews**（仅限英语）的长度，设置包括短（约 5+ 分钟）、长（约 20+ 分钟）和默认（约 10+ 分钟）。
   - 这种自定义功能让用户能够根据需求更好地调整 **AI 主持讨论**的深度和时长。
- **播客发音引发问题**：一位用户开玩笑地报告说，尽管为生成的播客（[notebook 链接](https://notebooklm.google.com/notebook/18c3b1c9-553a-4f77-bd6a-502d5f1fad56/audio)）提供了关于 *'RAGBRAI'* 发音的清晰指令，但 AI 还是读出了四种不同的发音。
   - 该用户正在利用 **NotebookLM 的播客生成能力**制作播客，其他成员建议尝试使用官方发音标注（IPA）并将发音指令添加到 Prompt 中。
- **AI 反幻觉英雄行动**：一位用户创建了**自定义系统指令**（被称为 *'LSTA Agent'* 和 *'Ultimate AI Anti-Hallucination Integrity protocol'*），以防止 AI 篡改源材料。
   - 这是为了回应另一位用户通过将文档中的名字替换为 **Star Wars**（星球大战）、**Detective Colombo**（神探可伦坡）和 **Homer Simpson**（荷马·辛普森）角色来修改文档的行为（[notebook 链接](https://notebooklm.google.com/notebook/7ccf92f3-fd96-45f6-8f27-50cf133f4baa/audio)）。
- **订阅层级定价引发辩论**：用户正在讨论 **NotebookLM 订阅**的定价，主要关注在试用优惠结束后，Plus 层级每月 **250 美元**的费用。
   - 关于该优惠存在困惑，一些人将其理解为前 3 个月每月 **125 美元**，之后恢复全价。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Cutotune 自动调优器调优 CUDA Kernels**：一位成员介绍了 **cutotune**，这是一个简单的自动调优器（autotuner），用于调优在 NVIDIA GPU 上调用 kernels 的任何 Python 代码，并声称它 *应该可以* 轻松扩展到其他加速器，并支持 **PyTorch compile**。
   - 该自动调优器允许在不注释掉自动调优器的情况下覆盖自动调优参数，并且它允许基于 **stride/dtype** 的张量属性触发和功能触发，这是 **Triton autotuner** 中 *key* 的超集。
- **FSDP2 显存占用更少但速度更慢**：一位成员测试了 **FSDP1** 和 **FSDP2**，观察到 **FSDP2** 使用的 VRAM 少得多，但速度更慢。来自 **RTX 4000 Ada x4** 的结果显示了 *full* 和 *lora* 设置下的不同性能。
   - 该成员记录了在使用 batch size 为 4 时 **FSDP1** 和 **FSDP2** 的以下性能表现：*#full fsdp1:15.801G, 7.3s/step fsdp2:11.599G, 13.39s/step #lora fsdp1:13.783G, 8.58s/step fsdp2:10.758G, 10.45s/step*。
- **Google 凭借 Gemini 进军 Diffusion 领域**：提到了 Google 的 **Gemini Diffusion** 模型，标志着 Google 进入了 Diffusion 模型领域，并附带了 [DeepMind 模型页面](https://deepmind.google/models/gemini-diffusion/) 的链接。
   - 另外，**SageAttention** 的整洁代码库也受到了关注，并附带了 [GitHub repository](https://github.com/thu-ml/SageAttention/) 的链接。
- **Axolotl 的 QAT/PTQ 工作流受到审查**：成员们仔细研究了 **Axolotl** 的 QAT/PTQ 工作流，比较了它们各自用于微调和量化的配置。
   - 两个流程之间的关键区别在于缺少 `prepare_model_for_qat`，同时据报道，在相同配置下使用默认设置在 **OpenAssistant/oasst1** 数据集上获得了良好的结果。
- **NVIDIA 和 UCL 举办 CUDA 聚会**：NVIDIA 和 UCL 将于 **6 月 5 日** 在 **伦敦** 举办 **CUDA Developer Meet Up**，邀请各级开发者与 CUDA 专家交流并探索 Python 原生 GPU 编程。
   - 该活动承诺提供实战演讲、探索 **nvmath-python** 和 **cuda-core** 等新库、吃着披萨进行社交的机会，以及赢取 NVIDIA 礼品的机会（[在此 RSVP](https://events.nvidia.com/cudadevelopermeetup/?ncid=ref-dev-171762-0503)）。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Google 的 Code Agent 引发 AI 竞赛**：**Google code agent** ([real.azure](https://jules.google/real.azure)) 的发布在 AI 社区引发了关于其功能和能力的兴奋与比较。
   - 随着成员们在加入 **waitlisted**（候补名单）后急切等待访问权限，关于在 **Google I/O** 上正式亮相的猜测正在增加。
- **Gemma 3n 首次亮相，瞄准边缘计算**：**Gemma 3n** 是一款专为边缘设备设计的生成式 AI 模型，采用了一种包含参数高效处理优化在内的新架构（[论文](https://arxiv.org/pdf/2310.07707)）。
   - 该模型旨在最小化内存使用，可以处理音频输入并整合文本和视觉数据，有可能为智能眼镜提供动力。
- **Diffusion 模型有望实现并行文本生成**：正在探索 **Diffusion models** 通过其并行处理文本块和非因果生成文本的能力来实现更快的文本生成。
   - 成员们认为这具有并行化 chain-of-thought 推理的潜力，并且由于能够从一开始就考虑笑点，从而提高笑话写作水平，尽管 token 限制可能会带来约束。
- **OpenEvolve 演化算法最优解**：**OpenEvolve** 发布，这是 **Google DeepMind AlphaEvolve** 的开源实现，以 **99.97%** 的准确率复现了 DeepMind 的圆堆积（circle packing）结果。
   - 该 Agent 利用 **LLMs** 来发现和改进算法，演化整个代码库，并可与任何 OpenAI 兼容的 API 配合使用。成员们分享了其 [GitHub repository](https://github.com/codelion/openevolve) 和 [博客文章](https://huggingface.co/blog/codelion/openevolve) 的链接。
- **LLMs 社交化、表现出偏见并引起社会变革**：最近的一篇 [论文](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368) 揭示，去中心化的 **LLM populations**（LLM 群体）通过局部交互自发地形成了普遍采用的 **社会惯例**。
   - 研究还表明，在这个过程中可能会出现强烈的 **集体偏见**，即使单个 Agent 最初没有表现出偏见，并且 **adversarial LLM agents**（对抗性 LLM 智能体）可以推动社会变革。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **生物数据专家寻求与 Hugging Face 合作**：一位在 **EMBL-EBI** ([https://www.ebi.ac.uk/](https://www.ebi.ac.uk/)) 从事生物数据工作的成员表示，有兴趣与 Hugging Face 团队合作，利用 **LLMs** 和 **RLVR** 结合生物策展数据进行自动化策展。
   - 另一位成员建议联系特定用户以探讨潜在的合作。
- **DataTune 实现数据转换**：**Vitalops** 推出了一款新的开源工具 [DataTune](https://github.com/vitalops/datatune)，该工具可以通过简单的自然语言指令和 **LLMs** 执行数据转换。
   - 该工具旨在缓解与使用 **LLMs** 进行数据转换相关的上下文长度限制和高昂 API 成本的担忧。
- **使用 LiteLLMModel 修复 Agent 课程代码**：成员们建议在课程代码中将 `InferenceClientModel` 替换为 `LiteLLMModel` 以有效地使用 **Ollama**，因为来自 GitHub 的更新版 notebook 解决了 LoopEvent 部分的 bug（[notebook 链接](https://colab.research.google.com/github/huggingface/agents-course/blob/main/notebooks/unit2/llama-index/workflows.ipynb)）。
   - 一位用户报告称，通过创建项目 repo 的新克隆并复制文件解决了问题，并指出设置中的完整重建（full rebuild）无法修复该问题，称 *Spaces 中有些损坏的东西，即使在设置中完整重建也无法修复*。
- **Cyberdesk Agent 在周末构建计算机 Agent**：一位成员介绍了 **Cyberdesk**，这是一个可以在单个周末内构建计算机 Agent 的工具，利用任何 Hugging Face 的像素级模型（pixel-based model），该工具已在 [ProductHunt](https://www.producthunt.com/products/cyberdesk) 上发布。
   - 另一位成员介绍了 **OpenEvolve**，这是 **Google DeepMind AlphaEvolve** 系统的开源实现，可在 [GitHub](https://github.com/codelion/openevolve) 上找到，它使用 LLMs 来发现和优化算法，详见其 [博客文章](https://huggingface.co/blog/codelion/openevolve)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Perplexity 免费层级成本高达 3300 万美元**：根据 [这条推文](https://x.com/breadcrumbsre/status/1924474011125227687)，Perplexity 每年在被其称为 R&D 的免费层级上花费 **3300 万美元**。
   - 高昂的成本引发了关于提供如此广泛的免费访问背后的可持续性和战略逻辑的讨论。
- **Google 发布异步编程 Agent Jules**：由 Google 开发的异步编程 Agent **Jules** 已经发布，如其 [官方网站](https://jules.google/) 所示。
   - Jules 承诺通过协助工程师处理模块化任务和复杂的软件架构来改变编程工作流。
- **编程 Agent 未能通过 React 测试**：成员们报告称，目前的编程 Agent 在处理复杂的 **React** 代码时表现挣扎，例如会产生 *'maximum update depth exceeded'* 错误。
   - 这些 Agent 在重构现有代码库方面也存在困难，这表明这些 Agent 在处理重构任务时可能需要帮助。
- **反 AI 的“回声室”出现**：讨论强调了 **Hacker News** 和 **Reddit** 等平台上反 AI 情绪的兴起。
   - 一些成员认为，这些论坛可能反映的是“回声室”效应，而非广泛的社会观点，需要其他平台来进行更友好的 AI 讨论。
- **Gemini 演变为 AI 操作系统**：在 **Google I/O 2025** 上，Gemini 已演变为一个完整的 **AI 操作系统**，拥有一套超越聊天功能的新工具，根据 [此推文线程](https://x.com/heyshrutimishra/status/1924900609410662592)。
   - 看来 Google 正在推动 Gemini 不仅仅作为一个聊天机器人，而是转变为 AI 工具的一站式平台。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Google 发布 Jules 挑战 Codex**：Google 发布了 **Jules**，这是 **OpenAI Codex** 的竞争对手，但目前在欧洲仅限候补名单访问 ([Jules](https://jules.google/))。
   - 该公告引发了关于 Google 持续投资 **AI 代码生成工具**的讨论。
- **LatentSeek 实现无 Token 化**：**LatentSeek** 在选择下一个用于解码的 token 之前，直接在潜表征（latent representations）上应用逐 token 的 RL。
   - 成员们强调了它与绕过分词步骤的 **COCONUT** 的区别，并暗示 **LatentSeek** 可能是基于 **Qwen 2.5** 的基准微调版本。
- **GNN 初学者探讨 `torch_geometric` 的细节**：一位成员询问如何使用 `torch_geometric` 实现 **GNN** 以进行节点回归任务，特别是如何为节点回归任务构建输入数据结构。
   - 另一位成员分享了代码示例 `class GAT(torch.nn.Module):`，并表示稍后会协助解决具体的实现问题。
- **剖析知识操纵**：成员们分析了《[语言模型的物理学：第 3.1 部分，知识存储与提取](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633)》和《[第 3.2 部分，知识操纵](https://ssrn.com/abstract=5250621)》，重点关注语言模型如何处理事实性知识及其在下游任务中的应用。
   - 讨论强调，虽然语言模型在知识**检索（retrieval）**方面表现出色，但在处理**分类**和**比较**等任务时却很吃力，尤其是在没有 **Chain of Thoughts (CoTs)** 的情况下。
- **Google 发布 Gemma 3N 详情**：Google 披露了其 **Gemma 3N** 模型系列的详细信息 ([Gemma 3N 文档](https://ai.google.dev/gemma/docs/gemma-3n))。
   - 同时发布了关于 **Google AI Edge** 小语言模型、多模态、RAG 和函数调用（function calling）的消息 ([Google AI Edge](https://developers.googleblog.com/en/google-ai-edge-small-language-models-multimodality-rag-function-calling/))。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 在 Angular/Typescript 任务中表现不佳**：尽管调整了阿里云端点的温度参数，一位用户发现 **Qwen 2 35B** 无法为使用 `element.innerHTML` 的 Angular/Typescript 代码生成 diffs。
   - 另一位成员在 Rust 上使用 **Qwen** 取得了成功，但在 Kotlin 上失败了，原因是不可变映射（immutable map）问题，即使类型信息清晰可见。该成员引用了 [Paul 关于 Qwen3 的博客文章](https://aider.chat/2025/05/08/qwen3.html)。
- **Aider 的 Shell 执行需要人工参与**：一位用户询问是否可以在 **Aider** 中执行 shell 命令（如列出暂存文件），但发现 `--yes-always` 选项并不能绕过手动确认的需求。
   - 根据 [issue #3903](https://github.com/Aider-AI/aider/issues/3903)，这是一个刻意的设计选择。
- **Aider 默认 YAML 配置即可正常工作**：当被问及针对 **Aider** 优化的 **YAML** 配置时，用户了解到 **Aider** 旨在通过默认配置实现最佳运行，模型选择才是最关键的因素。
   - 提供了 [示例配置](https://github.com/burnettk/config-aider/tree/main/sample_config) 的链接，并建议那些想要偏离标准 **Aider** 设置的用户参考 `GLOBAL_DEFAULTS.yml`。
- **Gemma 3n 4B 在 Polyglot 上得分不俗**：新的 **Gemma 3n 4B 模型**在 Polyglot 基准测试中获得了 44.4% 的分数，尽管实现这一分数的具体设置尚不清楚。
   - 一位用户指出，*一个 4b 模型的得分比 Gemini 2.0 Pro 还要高*。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI Agent 首次亮相，配备专用算力**：**Manus** 是一款**拥有自己电脑的 AI Agent**，旨在构建网站、撰写报告和执行研究任务，可通过邀请链接访问，例如 [这一个](https://manus.im/invitation/CWYMVAAXR6HZ)。
   - 用户反馈强调了它的能力，同时也指出了一些关于*积分系统/使用情况以及邀请码滥用*的问题。
- **网站创建被确认为 Manus 的核心能力**：用户确认 **Manus** 确实可以创建网站，并参考了 [Manus 官方用例集](https://manus.im/usecase-official-collection) 和 [来自用户的 Manus 用例](https://manus.im/usecase-from-user)。
   - 该 Agent 的网站构建功能作为其 AI 能力的实际应用脱颖而出。
- **Manus 饱受网络连接错误困扰**：用户报告在使用 Manus 时遇到**网络连接错误**。
   - Manus 团队正在积极调查这些问题，并请求提供账号邮箱和会话链接，以便诊断和解决连接问题。
- **成员询问 Manus 技术栈**：社区成员对 Manus 背后的**技术栈（tech stack）**表现出兴趣。
   - 得到的回复是团队正在探索各种*创新*，但未指明具体技术。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **范畴论（Category Theory）兴趣被激发**：成员们表达了学习**范畴论（category theory）**与 **AI** 交叉领域的兴趣。
   - 目标是探索与该组合相关的 Prompt Engineering Token 的技术和方法。
- **Cohere Labs 开启新的资助篇章！**：一位用户询问了 **Cohere Research Grants 计划**的状态，因为申请链接似乎已失效 ([https://share.hsforms.com/1aF5ZiZDYQqCOd8JSzhUBJQch5vw](https://share.hsforms.com/1aF5ZiZDYQqCOd8JSzhUBJQch5vw))。
   - Cohere 代表澄清说，**Research Grants** 现在是 **Cohere For AI** 的一部分，被称为 **Cohere Labs**。
- **Cohere 满足客户对控制权的渴望**：一位用户询问了 **Cohere 模型在本地（on-prem）的私有化部署选项**，由于对**数据/LLM 主权**的关注，寻求完全的所有权和控制权。
   - Cohere 确认他们提供**私有化部署**作为核心解决方案，并鼓励通过 [sales@cohere.com](mailto:sales@cohere.com) 或 [support@cohere.com](mailto:support@cohere.com) 联系以获取详情。
- **Command A 运行缓慢，困扰客户！**：一位用户报告 **command A** 的响应时间变慢，特别是在使用**结构化响应参数（structured response parameter）**时。
   - Cohere 代表承认了该问题，但未发现已知故障，并请求将详细信息发送至 [support@cohere.com](mailto:support@cohere.com) 进行调查。
- **Vitalops 推出 datatune**：**Vitalops** 发布了 **datatune**，这是一个通过自然语言结合 **LLM** 进行数据转换的**开源工具**，可以在 [GitHub 上找到](https://github.com/vitalops/datatune)。
   - 该工具简化了数据转换，使其更加易于使用。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **深度探索者寻求 MCP 最佳实践**：成员们正在寻求关于 **MCP** 最佳实践的深入探讨，特别是围绕生产环境中的工具设计和部署，并链接到一篇关于[保护 Model Context Protocol 安全的 Windows 博客文章](https://blogs.windows.com/windowsexperience/2025/05/19/securing-the-model-context-protocol-building-a-safer-agentic-future-on-windows/)。
   - 讨论围绕 **browsermcp** 展开，该工具允许 Agent 可视化你在浏览器中看到的内容。
- **MCP 与 A2A 协议 Agent 结合**：发布了一个开源服务器，将 **MCP 与 A2A 协议 Agent 桥接**，允许 **Claude** 与 **A2A** Agent 交互，[代码已在 GitHub 上提供](https://github.com/GongRzhe/A2A-MCP-Server)。
   - **A2A** 的用例涉及将 Agent 像工具一样部署到某个域。
- **Wallet MCP 实现 AI 钱包集成**：**TokenPocket** 团队发布了 **Wallet MCP**，这是一个轻量级桥梁，可实现 AI 客户端与加密用户钱包之间的集成，支持多链资产管理、交易签名和智能合约交互，[代码已在 GitHub 上提供](https://github.com/TP-Lab/mcp-marketplace/tree/master/apps/wallet-mcp)。
   - 由于存在大量诈骗者和垃圾邮件发送者，加入时需要进行 **captcha** 验证。
- **公共 SearXNG MCP 服务器解决不可靠性问题**：一位成员在 [GitHub](https://github.com/pwilkin/mcp-searxng-public) 上创建了一个公共 **SearXNG MCP** 服务器，以解决公共 **SearXNG** 服务器不可靠以及缺乏 JSON 支持的问题。
   - 该实现会对调用的实例进行随机化处理，以避免 **DoS** 攻击，适合希望少量调用互联网搜索的私人用户。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD 的 Enum 决策令开发者恼火**：一位开发者对 **AMD** 在其仓库中更改 enum 的决定表示沮丧。
   - 该开发者开玩笑说，AMD 的做法表现得*就像他们快要用完数字了一样*。
- **Flash Attention 悬赏最初仅限于 7900XTX**：Flash Attention 悬赏已锁定，正在 **7900XTX** 和 **9070XT** 上进行测试，但目前仅支持 **7900XTX**。
   - 开发者表示，如果 **RDNA4** 增加了新的 wmma 指令，他们可能会将其包含在内，但他们目前没有 **9070XT** 用于测试。
- **BERT 训练悬赏旨在实现 Nvidia/AMD 霸权**：BERT 训练悬赏寻求在 tinygrad 中实现性能超越普通 attention 的 Flash Attention，并且它兼容任何可以运行训练器的 **Nvidia/AMD** 硬件。
   - 悬赏发布者正在使用 ChatGPT 编写他们的 AGENTS.md 文件，但效果不佳。
- **tinygrad 求职流程说明**：针对求职咨询，一名成员澄清说，在 tinygrad 获得工作的首要途径是通过 **bounties**（悬赏），建议从提交小型 PR 开始。
   - 他们指引参考 [tinygrad 在 X 上的置顶帖子](https://x.com/__tinygrad__/status/1910185557642367390) 以获取更多信息。
- **tinygrad 中的控制流难题**：一位成员询问 tinygrad 中是否存在类似于 `jax.lax` 中的控制流算子，例如 [jax.lax.cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)。
   - 另一位成员建议使用 [`Tensor.where`](https://tinygrad.org/tensor.html#tinygrad.tensor.Tensor.where) 作为 `jax.lax.cond` 的可能替代方案，然而 `jax.lax.cond` 允许你决定执行哪段代码分支，而 `Tensor.where` 专门应用于 tensor，因此两个路径仍然都会执行。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 举办金融研讨会**：**LlamaIndex** 将于 5 月 29 日在纽约举办一场实战研讨会，由 @jerryjliu0 主持，主题是构建用于**金融分析和尽职调查**的 Agent 工作流，点击[此处](https://t.co/geDdBoe9aL)报名。
   - 该活动将提供关于利用 LlamaIndex 进行高级金融应用的*独家*见解。
- **AWS 支持多 Agent 通信协议**：**AWS** 宣布加入 **MCP 指导委员会**，并为其演进贡献力量，以实现更好的 Agent 间通信，并与 **LlamaIndex** 等框架展开合作。
   - 有关 **MCP** 和 **AWS** 参与的更多详情可以在[此处](https://t.co/6HPKJASKPi)找到。
- **成员请求 Agent 移交示例**：一位成员询问 Agent 移交（handoff）的示例，另一位成员提供了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/)的链接。
   - 文档详细介绍了如何实现 **Agent Workflows**。
- **Llama Parse 遇到问题**：一位成员报告了在使用 *Parse with Layout Agent* 时 **Llama Parse 服务**出现的问题，指出任务耗时超过 **30 分钟**，然后在没有解释的情况下失败，并卡在加载阶段。
   - 他们还分享了失败任务的[截图](https://cdn.discordapp.com/attachments/1374484064811221172/1374484518286786763/job-failed.png?ex=682e37f5&is=682ce675&hm=7bbafea016edc727275f4de63a70d2626fe12d38ac7e8d3301ec3953fbc92559&)。
- **FAISS 的性能比较**：一位成员询问在 **RAG** 模型中使用 **VectorStoreIndex** 与本地 **FAISS** 进行存储的性能差异。
   - 他们质疑使用其中一种是否会降低 RAG 模型的性能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Recipe Tutorials 增强文档**：成员们一致认为 Recipe Tutorials 应该增强文档，提供像 **GRPO recipes** 这样在数学数据集上有改进的端到端示例，并附带指向特定数据集改进的直接链接。
   - 然而，将 Recipe Tutorials 作为 **CI** 的一部分被认为不是最优选，因为对于每个新模型，为了实现 *delta_in_performance > 0*（性能增量 > 0），可能需要重新考虑训练和评估数据集。
- **Llama2 评估证明很困难**：一位成员讲述了早期评估 **Llama2** 的痛苦经历，特别是使用方法 **(2)** 时。
   - 他们主张用最佳实践更新贡献指南，并为每个新模型包含汇总评估结果的 **README.md** 文件，以辅助完整性检查（sanity checks）。
- **请求 DistCp 转换 Issue**：有人请求创建一个将 **DistCp** 格式（来自异步检查点）转换为 **safetensors** 的 Issue。
   - 一位成员提供了辅助转换的工具（utils），并强调了其对 **DCP 团队** 的重要性。
- **Async GRPO 开发者注意到依赖问题**：一位正在实验 **async_grpo** 的成员注意到，**async_rl** 目前依赖于 **vllm==0.8.4**，而该版本又依赖于 **torch==2.6.0**。
   - 讨论了将 **vllm** 依赖更新到 **vllm==0.9.0**（需要 **torch==2.7.0**）的计划，并提醒说*尚未经过测试，但应该可行*。
- **Async RL Recipe 被标记为实验性**：**async RL recipe** 目前处于实验阶段，并固定在 **vllm** 的稳定版本上。
   - 升级到 **vllm 0.9.0** 的预发布版本尚未经过彻底测试。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 戏称“丑小鸭”背后的深层含义**：[X 上的一个帖子](https://x.com/dspyoss/status/1924838188289556966) 展示了一些*丑陋*的东西，但发布者表达了对它的喜爱，并戏称：*如果你理解了这个，你就理解了 DSPy 的全部*。
   - 该消息暗示，**理解 DSPy** 意味着欣赏非传统或最初不吸引人的方面，指向其开发和应用中可能存在的**独特方法**。
- **DSPy 的核心哲学：拥抱非传统**：最近的 **DSPy X 帖子** 表明，欣赏非传统是掌握该项目本质的关键，重点在于其可能存在的**独特方法**。
   - 这可能暗示了一种开发哲学，即重视在传统 AI 开发中经常被忽视或被认为不吸引人的方面。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 竞赛开启提交**：**AgentX** 竞赛提交表单现已开放，包括创业赛道（[表单链接](https://forms.gle/FJTC4jd197bNeJJ96)）和研究赛道（[表单链接](https://forms.gle/5dccciawydCZ8o4A8)）；提交截止日期为 **2025 年 5 月 31 日晚上 11:59（太平洋时间）**。
   - 创业赛道需要融资演示文稿（pitch deck）、产品演示视频和在线产品链接；研究赛道需要论文、视频演示和 **GitHub** 仓库，顶级团队将获得超过 **15 万美元** 的奖金；竞赛组织者请求大家在 [X](https://x.com/dawnsongtweets/status/1924470174776004875) 和 [LinkedIn](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii) 上帮忙宣传 **AgentX**。
- **学生在实验中使用 OpenAI API Key**：学生在实验（lab）中必须使用自己的 **OpenAI API key**，但在实际提交时可以将其排除。
   - 助教 <@181105076423753728> 可以回答是否存在不需要 API 调用的替代方法。
- **证书的 Mastery Tier 降级机制**：即使学生在实验中遇到困难，仍可以申请 **Mastery Tier**，因为如果完成了测验和文章，他们可以被“降级”到 **Trailblazer Tier**。
   - 降级操作由工作人员端完成。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **PDF 文本提取需要自定义 Embedder**：要从 **PDF 教科书** 中实现精确的文本提取，需要专门的 **Embedder** 模型来保持忠实度。
   - 标准模型可能不足以复制复杂 **PDF** 文档的精确布局和内容。
- **GPT4All API Key 安装故障已解决**：有用户报告说，在粘贴较长的 **OpenAI API key** 时，**GPT4All** 中的安装按钮无法正常工作。
   - 该问题已进行排查，暗示软件在安装过程中处理长 API key 时可能存在潜在问题。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **DataTune 简化数据转换**：Vitalops 推出了 [DataTune](https://github.com/vitalops/datatune)，这是一个专为**数据转换**设计的全新**开源工具**。
   - 它利用**自然语言指令**结合 **LLMs**，旨在简化数据操作流程。
- **DataTune 利用自然语言和 LLMs**：[DataTune](https://github.com/vitalops/datatune) 通过由 **LLMs** 驱动的直观**自然语言指令**来促进**数据转换**。
   - 这款来自 Vitalops 的**开源工具**旨在为用户简化并优化复杂的数据操作。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1374099298089701436)** (1432 messages🔥🔥🔥): 

> `Special Tokens, Gemma 3, Google I/O, OpenAI versus Google` 

- **思维标签引发对特殊 Token 的探索**：成员们讨论了使用**特殊 Token** 和 [Anthropic 的思维标签 (thinking tags)](https://chatgpt.com/share/682cad13-b340-8005-a728-e57349029018)，一位成员指出这些 Token *基本上与 <antml:thinking> 相同*。
   - 他们试图弄清楚这些标签的用途和用法。
- **Google 发布 Gemma 3**：Google 发布了 **Gemma 3**，并提供了 [Gemma 3N 技术报告](https://ai.google.dev/gemma/docs/gemma-3n)，尽管有人注意到它最初仅在美国可用。
- **Google I/O 大会因演示多于发布而令人失望**：正在进行的 Google I/O 活动既带来了兴奋也带来了失望，成员们表达了这就像是*一个填充章节*的情绪，并抱怨 *I/O 大会没有发布任何东西*。
   - 虽然宣布了 **Veo 3**、**Imagen 4** 和 **Gemini 2.5 Pro Deep Think** 等新功能和模型，但大多数仅限于受信任的测试人员、美国用户，或将在晚些时候推出。
- **OpenAI 对决 Google**：社区认为 OpenAI 更好，多位成员推荐使用 ChatGPT Pro 而非 Google 新推出的 **$250/月 Gemini Ultra 方案**，强调前者具有更好的价值主张。
   - 还有关于到 **Grok 9** 发布时 **Google 破产**可能性的讨论。
- **用户实验本地 AI Zero-Shot 贪吃蛇游戏**：成员们讨论了本地 AI 是否能够 Zero-Shot 生成《无人深空》(No Man's Sky)，并在本地电脑的 **LM Studio** 中实验了 **Qwen 3 4B/8B**。
   - 然而，对话很快转向了在有限 VRAM 下运行此类模型的挑战，一位用户指出在他们的配置下*可能每秒只有 5 个左右的 [tokens]*。

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1374427831555199117)** (1 messages): 

> `Perplexity Updates, F1 Standings, Sidebar Shortcuts` 

- **Perplexity 发布新功能**：Perplexity 宣布发布多项新功能，包括 **Perplexify Me**、**F1 实时排名与解说**、**回答中的 F1 比分**、**改进的文件附件**以及**侧边栏快捷方式**。
   - 完整的更新日志可在 [Perplexity's changelog](https://www.perplexity.ai/changelog/what-we-shipped-may-19th) 查看。
- **Perplexify Me 上线！**：发布了一项名为 **Perplexify Me** 的新功能。
   - 未提供其他细节。
- **F1 实时排名现已可用！**：Perplexity 推出了 **F1 实时排名与解说** 以及 **回答中的 F1 比分**。
   - 这允许用户直接在 Perplexity 平台内随时了解一级方程式赛车的最新动态。
- **文件附件改进**：Perplexity 改进了**文件附件**功能。
   - 未提供其他细节。
- **侧边栏快捷方式提升导航体验**：Perplexity 添加了**侧边栏快捷方式**。
   - 未提供其他细节。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1374101385754316880)** (808 条消息🔥🔥🔥): 

> `Notebooklm 在 Android 发布，GPTs Agents，OpenAI 的侧边栏，Perplexity AI Discord 聊天机器人，Grok 很棒` 


- **Gemini 2.5 Flash 5月20日版是主力模型**：Gemini 2.5 Flash 5月20日 Checkpoint 是 Google 顶尖的*主力模型*，专门为**高级推理、编程、数学和科学任务**而设计。
   - 它包含内置的*思考*能力，使其能够提供**更高准确性和细致上下文处理**的响应。
- **Grok 是新的无限推理选择**：对于寻求 AI 模型中强大、**免费、无限推理能力**的用户来说，Grok 是首选。
   - Grok 被认为具有更随性的语气，一位用户将其描述为具有**随和的黑人兄弟气息 (chill black dude energy)**。
- **Chrome 中的 Gemini 是 Dia Browser 的终结者**：Google 在 **Chrome 中集成 Gemini** 被视为有效地*杀死了 Dia Browser*，一位用户幽默地发布了一个[猫咪爆炸 GIF](https://cdn.discordapp.com/attachments/1237372301243191327/1274608969989754911/catBOOMM.gif)作为回应。
   - 一名成员表示，他们目前可能是唯一还在使用 **Dia** 的人。
- **Google Deep Search 抄袭 Perplexity**：成员们讨论了 Google 新推出的 **Deep Search** 功能，有人开玩笑说*它看起来像是拿走了 Perplexity 的所有功能*，并提供了最佳的浏览体验。
   - 像 PPLX 那样把某样东西做完美的缺点是，它会变成行业基准，然后每个人都会模仿这种方法论。
- **Perplexity AI Discord 聊天机器人频道已禁用**：有消息提到 **Perplexity AI Discord 聊天机器人频道**已被禁用。
   - 一名成员提供了一个[链接](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669)，指向确认该频道停用的消息。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1374160662304919653)** (4 条消息): 

> `Grok, 数据, GitHub Copilot, 印度` 


- **显微镜下的 Grok 数据**：一个 [Perplexity 页面](https://www.perplexity.ai/page/grok-s-toxic-data-footprint-9CE0NC8wQM2nwFQk3BGxqg)讨论了 **Grok** 的数据足迹。
- **讨论偏向行动及其实现**：一个 [Perplexity 页面](https://www.perplexity.ai/page/bias-towards-action-implementa-LiIPjWN9Syuk28ICnCc.Ug)涵盖了偏向行动（bias towards action）的实现。
   - 未提供更多细节。
- **印度外交亮点**：一个 [Perplexity 页面](https://www.perplexity.ai/page/india-s-diplomatic-global-outr-OQZZ1AMRSwOAXPl5SEg0VA)展示了印度的全球外交拓展。
- **GitHub Copilot 走向开源**：一个 [Perplexity 页面](https://www.perplexity.ai/page/github-copilot-goes-open-sourc-WrHWW4bbRAG4NY0lnxqxqw)讨论了 **GitHub Copilot** 走向开源。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1374131205154279504)** (14 条消息🔥): 

> `Playground 与 API 输出质量对比, Deep Research API 问题, Perplexity API 超时问题, 通过 OpenAI Python 库使用 JSON schema` 


- **Playground 输出优于 API？**：成员们反映，即使在调整了 top_p 和 top_k 值后，**Perplexity Playground 的输出**也比 **API** 的输出更直接、更准确、更不模糊。
   - 一位用户发现，在助手（assistant）的对话历史中添加一条*示例消息 (sample example message)* 可以改善 API 的结果，但会显著增加 Token 使用量。
- **Deep Research API 伪装成 reasoning pro？**：一位用户报告称，通过 API 发出的 **Perplexity Deep Research** 请求在仪表板中显示为 **sonar-reasoning-pro**，尽管触发了正确的模型并收到了类似 Deep Research 的响应；目前已创建一个 [GitHub issue](https://github.com/ppl-ai/api-discussion/issues/320) 来解决这一差异。
   - 其他几位用户也确认他们遇到了同样的问题。
- **API 超时困扰用户**：用户在使用 Perplexity API 时经常遇到**超时问题**，特别是在通过 **N8N** 和 **Google Sheets** 研究目标公司时；这与手动浏览器使用形成对比，后者运行正常。
   - N8N 支持团队已确认超时问题源自 Perplexity API 端，这促使人们考虑将研究任务分批处理作为一种变通方案。
- **Perplexity API 可以处理 JSON schema**：一位用户询问 **Perplexity** 是否支持通过 OpenAI Python 库使用 **OpenAI JSON schema**。
   - 从给定的消息中尚不清楚该功能是否有效。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1374103536240951429)** (538 条消息🔥🔥🔥): 

> `32gb 运行 128k context, VRAM 计算器, KernelLLM GGUFs, Vision 量化, 训练 VITS` 


- **在 32GB 内存上运行 128k context 非常困难**：一位成员意识到在 **32GB** 的机器上运行 **128k context** 比预想的要困难。
   - 另一位成员分享了一个 [VRAM calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) 链接，可以帮助确定大语言模型（LLM）的显存需求。
- **Unsloth 发布 KernelLLM GGUFs**：Unsloth 团队发布了 **KernelLLM GGUFs**，并提供了 [Hugging Face](https://huggingface.co/unsloth/KernelLLM-GGUF) 链接。
   - 这些发布已在 [Unsloth subreddit](https://www.reddit.com/r/unsloth/) 上宣布，并讨论了 **Qwen3 上传**和修复后的 chat templates。
- **Sesame Notebook 获得“修复版”更新**：Unsloth 团队宣布了一个“修复版”的 **Sesame notebook** 更新，允许生成超过 10 秒的音频，可通过[这条推文](https://x.com/UnslothAI/status/1924848135991656603)访问。
   - 此更新包括将 **max_new_tokens** 参数设置为 125，其中 125 个 tokens 相当于 10 秒的音频。
- **Google 发布面向移动设备的 Gemma 3n**：Google 推出了 **Gemma 3n**，旨在低资源设备上高效执行，并具备多模态输入能力。
   - 该模型使用“选择性参数激活”技术，以 **2B** 和 **4B parameters** 的有效尺寸运行，演示应用可在 [GitHub](https://github.com/google-ai-edge/gallery) 上获取。
- **Unsloth 亮相 Google IO 大会**：Unsloth 在 Google IO 活动中展出，正如[这条推文](https://x.com/UnslothAI/status/1924977922915631443)所宣布的那样。
   - 新社区成员在活动中通过 **Gemma+Unsloth+Collab demo** 发现 Unsloth 后表达了兴奋之情。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1374390913123618891)** (9 条消息🔥): 

> `Mistral Small 3.1, Qwen2.5 VL 基准测试, IBM Granite 4.0, Gemini Diffusion, 视觉化 AI 学习应用` 


- **Mistral Small 3.1 发布！**：成员们注意到了 **Mistral Small 3.1** 和 **Mistral Medium** 的发布，但指出 Mistral Medium 并非开源。
   - 一位成员分享了 [Qwen2.5 VL 的 Ollama 库](https://ollama.com/library/qwen2.5vl)链接，并提到了一些“疯狂的基准测试”。
- **IBM 悄然推出 Granite 4.0**：**IBM** 秘密发布了 **Granite 4.0** 的“微型预览版”，引起了用户的关注。
   - 一位成员表示 *Gemini Diffusion 真的很酷*，但其他人提到[他们并不是该领域的先行者](https://x.com/karpathy/status/1894923254864978091)。
- **AI 工程师为视觉化 AI 学习应用寻找营销人员**：一位拥有多年经验的 AI 工程师正在为一款“帮助人们以直观、视觉化的方式学习 AI”的应用寻找营销人员。
   - 该工程师提到，像中国这样的国家正在大力投资 AI 教育，并且“获取直观的 AI 学习工具应该是全球性的，并对所有人开放”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1374105232644112646)** (235 条消息🔥🔥): 

> `Unsloth Model Merging, PPO Training, GRPO Training, Qwen 3 Models` 


- **Unsloth 模型合并问题浮出水面**：用户报告称，合并 **4-bit 模型**并保存为 **16-bit** 可能会导致性能问题，但以 **4-bit** 模式加载合并后的模型会进一步降低合并模型的性能。
   - 一位用户建议“在尝试合并时不要加载 4bit 或 8bit”，合并可以在普通 RAM 中完成以避免精度问题。他们分享了一个 [colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb) 概述了这一过程。
- **PPO 训练证明极具挑战**：一位用户表示有兴趣使用 PPO 训练，但找不到相关的 notebook，引发了关于 LLM 中 PPO 挑战的讨论，特别是关于 VRAM 使用和奖励函数限制的问题。
   - 有人提到目前的 trainer 设置仅支持 `AutoSequenceForClassification` 奖励模型，阻碍了自定义奖励函数，不过进一步的升级正在开发中。
- **GRPO 故障引起关注**：一位用户报告在对 Qwen 3 模型进行 **GRPO** 训练期间 KL 散度出现异常飙升，提示可能是由于有问题的样本数据造成的。
   - 另一位用户成功使用 GRPO 训练了几个模型，但在 PPO 上遇到困难，并希望对比 **GRPO** 与 **PPO**。
- **Qwen 3 问题得到快速澄清**：一位用户询问是否可以使用 **Qwen 3** 进行分类任务，以及该 notebook 是否可以用于其他 LLM。
   - 确认几乎任何模型都可以使用。该用户随后遇到了一个错误，通过从 `get_peft_model` 中移除 `lm_head` 解决了该问题。
- **Intel GPU 集成正在进行中**：一位用户询问 **Unsloth** 对 **Intel GPU**（特别是 **Intel Arc Pro B60**）的支持情况，社区指出一个正在进行的 pull request 表明了该领域的进展 [PR#2388](https://github.com/unslothai/unsloth/pull/2388)。
   - 其他用户提醒该集成仍在进行中，在相关问题解决之前，期望获得完整功能可能还为时过早。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1374110631908872375)** (33 条消息🔥): 

> `Entropix Pruning, VLM Gemma3 evaluation metrics, OpenEvolve released` 


- **Entropix 从沉睡中苏醒，带来多 GPU 计划**：在沉寂一段时间后，[Entropix](https://github.com/xjdr-alt/entropix) 的作者宣布了 **multi-GPU 部署**和在大模型上进行测试的计划，将项目拆分为两个仓库，并向提供算力的人员致谢。
   - 作者还提到他们已经筹集到资金并正在创办一家初创公司，同时指出了 [在 optillm 中使用熵解码在 GSM8k 上取得的早期结果](https://www.reddit.com/r/LocalLLaMA/comments/1g5gf27/entropy_decoding_in_optillm_early_results_on_gsm8k/)。
- **Attention 剪枝是否仍值得？取决于模型**：成员们讨论了在性能影响极小的情况下剪枝 attention 层以实现加速，但有人警告不要将 **Llama 2 时代**的技术应用于 **Llama 3-8B** 等较新模型，理由是其对剪枝的鲁棒性。
   - 另一位成员补充道：*这就是为什么当人们使用 llama 3.2 3B 时我感到困惑的原因，它是从最难剪枝的模型之一剪枝而来的，因此就其尺寸而言表现很差。*
- **评估 PEFT 微调的 VLM 需要特定的基准测试**：一位硕士生询问了评估 **PEFT 微调的 VLM (Gemma3)** 的标准指标，并建议在 **OCR 测试**中使用 **CER** 和 **WER 指标**，以及 perplexity。
   - 其他成员推荐了 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 和 [lm eval harness](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md)（包含 **90 多个任务**的集合），并强调需要一套基准测试来评估 VLM 的行为。
- **OpenEvolve 从原始代码汤中跃出！**：一位成员宣布发布 [OpenEvolve](https://github.com/codelion/openevolve)，这是 **Google DeepMind AlphaEvolve 系统**的开源实现，它使用 LLM 演化整个代码库以优化算法，详见[这篇博客文章](https://huggingface.co/blog/codelion/openevolve)。
   - 创建者成功在圆堆积（circle packing）问题上复制了 DeepMind 的结果（匹配度达 99.97%！），并将随机搜索演化为模拟退火算法，使用 **LLM ensemble 方法**来提高结果并实现多目标优化。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1374100494527631400)** (158 条消息🔥🔥): 

> `LM Studio API, RoPE Frequency Scale, Qwen 3 Speculative Decoding, Model Unloading via API, Sliding Window Attention` 


- **LM Studio API 支持 API 访问，但缺少 Web 前端**：一位用户询问如何使用 **LM Studio** 在本地托管 LLM 以进行 Web 集成（类似于 **Stable Diffusion**），但发现 **LM Studio** 仅提供 API 服务器，而不提供 Web 前端。
   - 他们建议使用独立的前端连接到 API 以实现所需功能。
- **配置 RoPE Frequency Scale**：一位用户询问在 **LM Studio** 中加载模型时如何设置 **RoPE Frequency Scale**，可以在模型加载器旁边的设置齿轮中找到该选项。
   - 用户可以修改 **gguf** 格式以包含它，因为 **ollama** 和 **llama.cpp** 都支持该功能。
- **Qwen 3 通过 Speculative Decoding 获得速度提升**：用户报告称，通过使用 Speculative Decoding，**Qwen 3** 的性能提升了 **12%**。这是通过将 max draft size 调整为 3 并将 drafting probability 提高到 0.8 实现的，他们在“大量调试”后报告了这些结果。
   - 他们警告说，增加 min draft size 只会增加耗电量和生成时间。
- **LMStudio API 揭晓模型卸载功能**：用户讨论了如何使用 **LM Studio API** 卸载模型，参考了[官方文档](https://lmstudio.ai/docs/cli/unload)，并发现 `model.unload()` 是正确的语法。
   - 讨论明确了该函数是 API 的一部分，不需要创建单独的函数。
- **Sliding Window Attention 登陆 Llama.cpp**：**Sliding Window Attention (SWA)** 现已在 **Llama.cpp** ([github.com/ggml-org/llama.cpp/pull/13194](https://github.com/ggml-org/llama.cpp/pull/13194)) 中可用，随后将引入 LM Studio，这能显著降低内存占用，特别是在长上下文场景下。
   - 初步测试显示，对于 **Gemma 4B 15K** 上下文模型，启用 SWA 将 KV cache 占用从 **1700MB** 降至 **348MB**，在使用 **Q8_0 quantization** 时进一步降至 **184MB**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1374156900647567430)** (449 条消息🔥🔥🔥): 

> `Intel Arc IPEX support in LM Studio, AMD GPU drivers issues, AVX2 support in LM Studio, Dual GPU setup, PCIE5 vs SATA SSD speeds` 


- **LM Studio 对 Intel Arc IPEX 支持的疑问**：成员们想知道如果 **Intel Arc** 流行起来，**LM Studio** 是否会获得 **IPEX support**，并提到了过去移动端和桌面端 GPU 的驱动问题。
   - 一位成员提到，在内存大于 **64GB** 的 **Zen4 系统**上，移动端驱动会导致黑屏，而桌面端 GPU 则需要侧载驱动。
- **AVX2 指令集要求困扰旧硬件上的 LM Studio**：一位用户报告在不支持的 **Core i7 3770K** CPU 上运行 **LM Studio** 时出现问题，该 CPU 缺少 **AVX2 support**，尽管他拥有受支持的 **Nvidia GTX 960** GPU。
   - 一位成员建议在没有 **AVX2 requirement** 的情况下编译 *llama.cpp*，或者尝试最近移除了 **AVX2 requirement** 的 **Ollama**，并分享了一个在旧硬件上运行 *llama.cpp* 的[仓库](https://github.com/kth8/llama-server-vulkan)。
- **关于通过 Thunderbolt 实现双 GPU 设置可行性的辩论**：一位成员询问是否可以采用**双 GPU** 设置，其中一个 GPU 放在外置 **Thunderbolt USB4 显卡坞**中，以避免降低主 **PCI-e 插槽**的速度。
   - 另一位成员确认这是可行的，但警告说带宽可能会被限制在 **PCIe x4**（在质量较差的情况下甚至是 **x1**），这可能会影响 **LLM inference** 以外任务的性能，并指出 Thunderbolt/USB4 接口需要 4 条专用的 PCI-E 通道。
- **爱好者对比 PCIE5 与 SATA SSD 的速度**：关于 **PCIE5 SSD** 相对于 **SATA** 的实际优势展开了讨论，一位成员幽默地表示，他没意识到 **SATA** 竟然相当于 **125 条 PCIE 5.0 通道**。
   - 共识倾向于认为，对于普通用户来说，从 **SATA 升级到 gen3 NVMe** 之后的实际优势有限，更快的存储带来的收益递减，除非是特定的创意工作流或数据中心。
- **爱好者考虑使用带有扩展卡的消费级 PC 运行 LLM**：一位成员询问在带有 **512 GB DDR5** 内存的消费级 **RAM 扩展卡**（插在 **PCIe 插槽**上）上运行 **DeepSeek R1** 的预期性能，并询问是否能达到 **10 tps**。
   - 有人指出，通过 **PCIe** 的内存带宽可能无法与直接插在 **DIMM 插槽**相比，这可能会限制性能，并将其与双通道 DDR4 的带宽进行了比较。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1374435961185898558)** (3 条消息): 

> `Provider slugs, Quantization slugs, Gemini Flash 2.5 release, Llama provider by Meta` 


- **OpenRouter 为提供商和量化添加了 slugs**：OpenRouter 宣布 **providers** 和 **quantizations** 现在拥有了 slugs，提升了开发者体验，详见 [其推文](https://x.com/OpenRouterAI/status/1924868065910518226)。
- **Google 发布 Gemini Flash 2.5，并在 OpenRouter 上线**：Google DeepMind 推出了 **Gemini Flash 2.5**，目前已在 OpenRouter 上可用；在此测试：[google/gemini-2.5-flash-preview-05-20](https://openrouter.ai/google/gemini-2.5-flash-preview-05-20)。
- **Meta 提供 Llama，仅在 OpenRouter 上以开放访问形式提供**：Meta 新的 **Llama** 提供商现已上线，在 OpenRouter 上独家开放访问，且可免费开始使用；包含一个新的 **3.3 8B 模型**，详见 [其推文](https://x.com/OpenRouterAI/status/1924899161469571337)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1374108403869286502)** (235 条消息🔥🔥): 

> `Gemini 2.5 Pro DeepThink, Veo 3, Imagen 4, Gemma 3n, audio support` 


- **新的 Gemini、Imagen 和 Veo 模型发布**：Google 意外提前公布了支持音频的 **Veo 3** 视频生成模型的定价，以及 **Veo 2** 的起始/结束帧、扩展现有视频和摄像机控制功能，同时还发布了 **Imagen 4** 和 **Gemini 2.5 Pro Deepthink**。
   - 一位成员分享道，“又有一家初创公司在尝试解决这个问题”，对此有人反驳认为，“现实情况是，所有的模型套壳（model wrappers）在淘金热期间都只是在卖铲子”。
- **免费的 Gemini 2.5 Flash Preview 可用**：`gemini-2.5-flash-preview-05-20` 已经部署。
   - 成员们讨论了免费 API 的访问权限，一位成员确认如果你之前消费过 10 美元，它仍然存在，但另一位成员表示“最好还是按需付费（pay as you go）”。
- **Google 的 Gemma 3n 媲美 Claude 3.7？**：根据 [这篇博文](https://developers.googleblog.com/en/introducing-gemma-3n/)，一个新的 Gemma 模型 **Gemma-3n-4B** 据称与 **Claude 3.7** 一样出色。
   - 一位成员对此表示怀疑，而另一位则表示“我不知道，那是 Chatbot Arena，所以只是用户偏好，对我来说听起来是有可能的”。
- **新的 LLM 缺乏 Diffusion 架构**：一位成员提问，“为什么现在大多数 LLM 还没有采用 Diffusion 架构，有什么原因吗？”
   - 回复建议称 **Diffusion** 需要大规模的架构重构。
- **OpenRouter API Key 在 TS 中集成 AI Agent**：一位成员希望使用 **OpenRouter API key** 和基础 URL，在 TS 上通过 OpenAI API 创建一个 Agent。
   - 他们希望在 Nest.js 项目中将代码编辑工具与 AI Agent 结合使用。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1374102190326681722)** (183 条消息🔥🔥): 

> `用于删除消息的 Discord 机器人，AI 中 “slop” 的定义，Gemini Diffusion，AI/ML 领域的导师指导，受压缩启发的 ARC-AGI 性能提升` 


- **EleutherAI 考虑使用 Discord 机器人打击垃圾信息**：成员们讨论了实现一个自动删除包含过多表情符号评论的 Discord 机器人，但为了防止滥用，考虑采用速率限制 (rate limits) 和管理员提醒 (mod pings) 而非直接删除，并提议增加一种仅限管理员的反应 (react) 来标记内容。
   - 成员们建议使用更通用的反应，仅统计正式成员的反应，并使用私有的管理员频道以简化违规检测，同时避免*不可逆的删除*。
- **定义 AI “Slop” 引发哲学辩论**：成员们辩论了 AI 生成内容中 “slop” 的含义，定义范围从*低质量 (low effort)*、*理解欠佳的材料*到*未能通过图灵测试*的内容，并引用了 [AI slop 维基百科页面](https://en.wikipedia.org/wiki/AI_slop)。
   - 讨论触及了 slop 是否本质上与 AI 绑定，一些人认为人类生成的内容也可以是 slop，并引用过度制作或无意义的输出作为例子，以及 [脆弱性失效 (fragility failures)](https://arxiv.org/abs/2410.05229v1)。
- **AI 导师指导请求引导至开源研究**：一位成员请求在 AI/ML 方面获得导师指导，表示对智能体 AI (agentic AI) 感兴趣但缺乏 OpenAI API 余额，随后被引导至 [在线实验室](https://discord.gg/4HnZxkCj) 和 [开源研究](https://discord.com/channels/665540686237450240/732688974337933322) 机会。
   - 导师建议关注*分布外检测 (out of distribution detection)*。
- **新颖性搜索 (Novelty Search) 驱动 ARC-AGI 性能**：成员们讨论了*新颖性 (novelty)* 与 slop 之间的联系，认为 slop 缺乏新颖性，并且与生成模型无法在其训练数据分布之外进行外推的能力强相关。
   - 随后他们触及了 ARC-AGI 中某些可以用于普通文本生成的方法，这些方法利用了 [压缩技术 (compression techniques)](https://www.etymonline.com/word/slop)。
- **成员询问 Gemini Diffusion**：一位成员发起了关于 Google 的 **Gemini Diffusion** 模型的讨论。
   - 他提出可以为任何没有访问权限的人代跑 Prompt。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1374187238333874287)** (50 条消息🔥): 

> `马毅 (Yi Ma) 关于智能的演讲，自动编码器 (Autoencoders) 与压缩，类似 DINOv2 的 SSL 方法，论文代码发布，OpenEvolve 发布` 


- **马毅的智能第一性原理**：成员们观看了 [马毅的 ICLR 演讲](https://www.youtube.com/watch?v=dVhMiMk1riQ) 并讨论了他的第一性原理方法，该方法将工作建立在信息论和控制论的基础上，与当前模型中盛行的启发式方法 (heuristics) 形成对比。
   - 一位成员发现演讲初期进展缓慢，但赞同其中的观点：*仅靠压缩是不够的；组织才是关键*。
- **自动编码器 (Autoencoders) 无法捕获所有信息**：有观点认为 **autoencoders** 的潜空间 (latent space) 可能无法捕获原始输入的所有信息，因为多个压缩表示都足以完成重构。
   - 正如一位成员所言：*并非所有能实现完美重构的自动编码器都包含相同的信息*。
- **极简 vs. 完整代码库：关于论文发布的辩论**：成员们辩论了发布极简代码库与完整代码库的优劣，其中一人主张采用 [minRF 风格](https://github.com/example/minRF) 的方法：在发布极简版本的同时附带一个更广泛的研究代码库。
   - 共识似乎是：理想的代码库应允许用户在其基础上进行构建，将其与现有基础设施集成，并能轻松更换组件。
- **OpenEvolve 问世！**：一位成员发布了 **OpenEvolve**，这是 Google DeepMind 的 AlphaEvolve 的开源实现，它使用 LLM 演化整个代码库以发现和优化算法，并复现了 DeepMind 在圆堆积 (circle packing) 上的结果。
   - 详情请查看：[GitHub](https://github.com/codelion/openevolve) 和 [博客文章](https://huggingface.co/blog/codelion/openevolve)。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1374162683498336276)** (3 messages): 

> `VLM 评估，纯文本评估，代码库条件语句` 


- **VLM 纯文本评估引发讨论**：一名成员询问关于在 `hf-multimodal` 或 `vllm-vlm` 模型上进行编程和数学等纯文本评估的问题。
   - 他们质疑 [提供的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2999/files) 在请求缺少图像时，是否合理地实现了向 `hf`/`vllm` 行为的回退。
- **不透明的条件语句引发重构**：另一名成员批评条件语句 `if requests and len(requests[0].args) < 3` 过于晦涩，且容易在未来导致代码损坏。
   - 他们建议直接检查感兴趣的条件，而不是依赖这种间接检查，作者提议将其移至工具函数中。
- **提出的条件简化方案**：原帖作者提出了一个改进的条件语句 `if not (requests and len(requests[0].args) >= 3 and "visual" in requests[0].args[2]):` 以提高清晰度。
   - 目标是在请求既缺乏足够的参数又没有 *visual* 属性时，触发纯文本代码路径。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1374101193185562746)** (207 messages🔥🔥): 

> `25 个工具限制导致对话中断，DeepSeek-R1T-Chimera 模型打破循环，MCP 频繁刷新，Gemini 思考过程改变，O3 Pro 即将推出` 


- **Cursor 存在工具限制 Bug**：根据 [Reddit 帖子](https://www.reddit.com/r/cursor/comments/1kqj7n3/cursor_intentionally_slowing_nonfast_requests/)，达到 **25 个工具限制**并在继续之前点击接受会导致对话中断。
- **DeepSeek-R1T-Chimera 打破死循环**：**DeepSeek-R1T-Chimera** 模型（在 **R1** 和 **V3** 之间微调）据称是唯一能在用户提示词测试方法中打破循环的免费模型，该模型已在 [Hugging Face](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) 上可用。
- **MCP 刷新导致困扰**：用户遇到 **MCP** 需要频繁刷新的问题，因为其状态会变黄，消耗了大部分请求额度。
- **Gemini 思考方式改变**：用户报告 **Gemini 的思考过程**发生了变化，现在 Gemini 会总结它在思考什么，而不是显示实际的思考过程，这使得编写高质量提示词和项目规则变得更加困难。
- **O3 Pro 可能即将发布**：人们对 **O3 Pro** 的发布充满期待，一些推测基于 OpenAI 员工的推文，暗示其即将推出，可能在 6 月发布。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1374198585041879110)** (24 messages🔥): 

> `在没有 CUDA 的情况下运行模型，MAX 和 HF 模型，机器人模型和数据流，Cosmos “世界模型”，从 PyTorch 迁移到 MAX` 


- **在无 CUDA 环境下运行 MAX 模型**：用户询问 **GR00T-N1-2B** 或 **pi0** 等模型是否可以在没有 **CUDA stack** 的情况下使用 **MAX** 和 **Mojo** 运行。
- **MAX Graph 需要重新实现**：虽然 **MAX** 在 Tokenizer、预处理、权重和超参数配置方面可以与 **Hugging Face** 互操作，但核心模型架构需要在 **MAX graph** 中构建。
- **机器人模型的数据流需求**：连接机器人模型需要特定的数据流，这些数据流很难以通用的方式提供，需要大量的传感器输入。
   - 澄清指出，重点仅在于 **NN 模型**，而非整个数据流水线。
- **Cosmos：消耗 GPU 的世界模型**：**Cosmos** 被描述为一个结合了物理引擎和视频模型的“世界模型”，但它对 **VRAM** 要求极高，因此对 GPU 负担很重。
- **Modular 映射 PyTorch 到 MAX**：Modular 正在记录从 **PyTorch** 核心架构到 **MAX graph** 的迁移过程，并计划在 [docs.modular.com](https://docs.modular.com) 提供教程和参考文档。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1374127906091172020)** (174 messages🔥🔥): 

> `25.3 版本中的误报警告，Mojo 中的未使用变量警告，fn() -> raises 语法，使用参数化 Trait 的 IO API 设计，基于 DMA 的 API` 


- **Mojo 25.3 中出现误报警告**：有成员报告在 **25.3** 版本中，关于未使用赋值的误报警告数量有所增加，具体表现为 `warning: assignment to 'suggested_max_length' was never used; assign to '_' instead`，详见[此 Bug 报告](https://github.com/modular/modular/issues/4642)。
- **Mojo 给出未使用变量警告**：当仅引用变量的别名值（alias values）时，Mojo 编译器会错误地将变量标记为未使用。这是 **def** 函数声明中的一个问题，通过切换到 `fn() -> raises` 已确认可以解决该误报。
   - 例如，即使后续使用了 `r2.N` 和 `r2.D`，`var r2 = r1.simplify()` 仍会触发警告。解决方法是将 def 语句改为 `fn() -> raises`。
- **Mojo 解析器接受 `fn() -> raises` 语法**：解析器接受 `alias func = fn()` 语法，这是 `fn() -> None` 的语法糖。一些成员认为这不是个好主意，虽然为了与其他返回语句保持一致而保留，但它违背了其他语言及其类型签名的要求，如[此代码片段](https://gist.github.com/moduland/e9f85376a973ef3f5f4648d234081f9d)所示。
- **参数化 Trait 引发 IO API 设计辩论**：成员们讨论了如何使用参数化 Trait 设计 IO API，以处理缓冲区所有权以及对 `owned` 或 `ref` 类型的泛型支持，其中 io_uring 要求传递缓冲区的所有权。
   - 讨论中提出了对“围绕尚未实现的功能进行设计”以及“可能创建过于复杂或限制性 API”的担忧，一些人主张采用务实的方法，在承认未来改进的同时解决当前的能力问题。
- **Modular 将 GPU Puzzles 从 Magic 迁移至 Pixi**：Modular 团队已弃用 **Magic**，转而使用 **Pixi** 来处理 Mojo GPU puzzles。Magic 实际上只是 Pixi 的一个封装，详见[此提交](https://github.com/modular/mojo-gpu-puzzles/commit/d8270a7a7c1cf8a99c71cd7e25dbb8ca191a66f9)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1374139688331509854)** (7 messages): 

> `Max 对比 Fireworks.ai/Together.ai/Groq.com，vLLM 对比，优化 Max 以实现更低延迟和更高吞吐量，Max 导入源码的可见性，大规模解耦推理的企业级解决方案` 


- **Max 声称拥有全栈控制权**：一位用户询问 **Max** 在 AI 推理性能和运行时间方面与 **Fireworks.ai**、**Together.ai** 或 **Groq.com** 的对比情况，特别是考虑到这些平台声称拥有卓越的速度和平台成熟度。
   - 回复强调 **Max** 提供了对技术栈的全栈控制，支持在性能、延迟、精度控制方面进行自定义工作，并具备在每一层进行编程的能力，同时在计算性能上可与 Dynamo 媲美。
- **揭穿关于 vLLM 的性能营销说辞**：会议指出，将 **vLLM** 与端点服务商进行对比并非对等比较（apples-to-apples），用户应警惕“性能营销”说辞，尤其是那些与旧版本 vLLM 进行对比的说法。
   - 消息引用了来自 **Together.ai** 和 **Fireworks** 的示例链接（[Together.ai](https://www.together.ai/solutions#customer-stories), [Fireworks.ai](https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs)），其中的对比是基于旧版 **vLLM** 和不同的配置进行的。
- **优化重点在于延迟和吞吐量**：一位用户表示其主要目标是降低延迟和提高吞吐量，旨在最小化成本，并询问 Max/Mojo 最强的卖点是什么。
   - Modular 团队邀请其讨论企业级解决方案，重点介绍了大规模解耦推理（disaggregated inference）方案，并建议通过 tim@modular.com 联系他们。
- **Max 导入的代码仍部分隐藏**：一位用户询问示例代码中可见的 Max 导入部分的源代码，请求查看在哪里可以找到它们（参考[附图](https://cdn.discordapp.com/attachments/1212827597323509870/1374489593352687659/image.png?ex=682e3caf&is=682ceb2f&hm=41ba5d7375f33b9264f1f83e814c5bfb4064789dae0bccce7e987ce18abaa137)）。
   - 官方澄清源代码并未完全开源，但可以在用户环境中进行检查，其中某些部分会调用 **C++** 代码。


  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1374118249809186846)** (3 messages): 

> `NotebookLM 移动应用发布，音频概览（Audio Overviews）自定义，Google I/O 主旨演讲总结，视频概览（Video Overviews）功能预览` 


- **NotebookLM 移动应用上线！**：NotebookLM 移动应用现已正式上线，具备 MVP 功能集，并鼓励用户提供反馈和功能需求；欲了解更多信息，请访问 [Google Blog](https://blog.google/technology/ai/notebooklm-app/)。
- **音频概览（Audio Overviews）现已支持调节！**：用户现在可以控制音频概览（仅限英文）的时长，提供短（约 5 分钟以上）、长（约 20 分钟以上）和默认（约 10 分钟以上）设置，以自定义 AI 主持讨论的深度和长度。
- **Google I/O 2025 主旨演讲回顾**：为错过演讲的用户提供了一份总结今年 #GoogleIO 主旨演讲所有内容的笔记本；点击[此处](https://g.co/notebooklm/io2025)查看总结。
- **视频概览（Video Overviews）功能预告！**：在 [X](https://x.com/NotebookLM/status/1924938813018653028) 上发布了全新视频概览功能的预览。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1374111446929117244)** (23 messages🔥): 

> `播客中的发音问题，将时间线导出到 Google Calendar，将 NotebookLM 集成到 Discord，防止源内容被篡改的 AI 协议，NotebookLM 移动应用` 


- **播客发音问题困扰用户**：一位用户开玩笑地报告说，尽管为生成的播客（[笔记本链接](https://notebooklm.google.com/notebook/18c3b1c9-553a-4f77-bd6a-502d5f1fad56/audio)）提供了关于“**RAGBRAI**”发音的明确指令，AI 还是设法读出了*四种不同的发音*。
   - 其他成员建议尝试使用官方发音符号（IPA）并将发音指令添加到 prompt 中。
- **防止协议问题：AI 反幻觉的壮举**：一位用户创建了**自定义系统指令**（被称为“**LSTA Agent**”和“**终极 AI 反幻觉完整性协议**”），以防止 AI 篡改原始素材。
   - 这是为了回应另一位用户通过将文档中的名字替换为**星球大战**、**神探可伦坡**和**荷马·辛普森**中的角色来修改文档的行为（[笔记本链接](https://notebooklm.google.com/notebook/7ccf92f3-fd96-45f6-8f27-50cf133f4baa/audio)）。
- **NotebookLM 移动应用上线**：Google 为 iOS 和 Android 发布了官方 **NotebookLM 移动应用**（[公告链接](https://blog.google/technology/ai/notebooklm-app/)），具有离线**音频概览**、交互功能和无缝内容共享。
   - 该应用旨在让 NotebookLM 对包括学生、专业人士以及任何需要处理大量信息的人在内的各类用户更加易用和实用。
- **语言限制隐忧**：用户询问是否可以上传英语以外的语言资源。得到的回答是**目前官方仅支持英语和法语**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1374102422384804023)** (153 messages🔥🔥): 

> `NotebookLM Android 应用反馈，播客生成，文件大小限制，输出语言选项，分享笔记本` 


- **NotebookLM 发布 Android 应用**：NotebookLM 发布了 [Android 应用](https://apps.apple.com/us/app/google-notebooklm/id6737527615)，但初步反馈显示，与网页版相比它**缺少一些功能**，例如*发现来源（Discover Sources）*和笔记功能，用户还在询问对折叠屏设备的兼容性。
   - 一些用户更喜欢移动端网页体验，在 Android 上可以将其保存为“应用”以便快速访问。
- **播客功能引发热议**：用户正在探索 NotebookLM 的播客生成能力，用于创建新话题的内容，但面临**音频长度限制**的问题，目前似乎有 **6-8 分钟**的限制。
- **用户询问上传文件的大小限制**：用户请求将上传文件的大小限制从 **200MB** 提高到 **400/500MB**。
- **语言选项受限**：用户注意到更改输出语言仅在**网页版**上可行，并期待该功能被添加到应用中。
   - 一些用户指出，*聊天回复和音频概览将根据您的设备语言生成*，除非您在 NotebookLM 网站上设置了语言覆盖。
- **订阅层级定价讨论**：用户正在讨论 **NotebookLM** 订阅的定价结构，对试用期结束后 Plus 层级每月 **$250** 的费用表示担忧。
   - 关于优惠方案存在困惑，一些人将其理解为前 3 个月每月 **$125**，之后恢复原价。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1374247100413644870)** (12 条消息🔥): 

> `cutotune 自动调优器, FSDP1 vs FSDP2, Liger-Kernel, cute-kernels 中的多头 GRU 层` 


- **Cutotune 自动调优器调优 CUDA Kernels**：一位成员介绍了 **cutotune**，这是一个简单的自动调优器，用于调优在 NVIDIA GPU 上调用 kernels 的任何 Python 代码，并声称它*应该*很容易扩展到其他加速器，并支持 **PyTorch compile**。
   - 该自动调优器允许在不注释掉自动调优器的情况下覆盖自动调优参数，并且支持基于张量属性和基于 **stride/dtype** 的功能触发器，这是 **Triton 自动调优器**中 *key* 的超集。
- **FSDP2 使用更少的 VRAM 但速度更慢**：一位成员测试了 **FSDP1** 和 **FSDP2**，观察到 **FSDP2** 使用的 VRAM 少得多，但速度更慢，来自 **RTX 4000 Ada x4** 的结果显示了 *full* 和 *lora* 设置的不同性能。
   - 该成员记录了在使用 batch size 为 4 时 **FSDP1** 和 **FSDP2** 的以下性能：*#full fsdp1:15.801G, 7.3s/step fsdp2:11.599G, 13.39s/step #lora fsdp1:13.783G, 8.58s/step fsdp2:10.758G, 10.45s/step*。
- **Liger-Kernel 性能**：一位成员测试了 **Liger-Kernel** 在 **FSDP1** 和 **FSDP2** 上的应用，其中 **FSDP2** 始终导致性能变慢。
   - 结果为：*#full liger fsdp1:5.426G, 9.03s fsdp2:3.92G, 15.41s #lora liger fsdp1:3.351G, 9.2s fsdp2:2.639G, 10.74s*。
- **Cute-Kernels 新增多头 GRU 层**：一位成员宣布在 **cute-kernels** 中增加了用 Triton 编写的多头 GRU 层，实现了跨 SM 的并行化，并链接到了 [cute-kernels 仓库中的相关目录](https://github.com/open-lm-engine/cute-kernels/tree/main/cute_kernels/modules)。
   - 多头 GRU 层的加入允许跨 SM 进行并行化，增强了 kernel 执行的效率。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1374135938011238545)** (2 条消息): 

> `Triton CPU 支持, TRITON_INTERPRET API, CPU 并行性限制` 


- **Triton 缺乏直接的 CPU 并行性**：一位成员澄清说 **Triton** 并不直接支持 **CPUs** 的并行化，限制了其多核能力。
   - 用户认为至少在多核并行方面效率不高。
- **TRITON_INTERPRET API 提供顺序 CPU 模拟**：讨论强调了使用 **TRITON_INTERPRET=1 API** 作为 **CPU** 执行的替代方案，它以顺序方式模拟并行方案。
   - 这意味着这种方法虽然是顺序的，但在某些用途上*几乎完美地模仿了并行方案*。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1374406320576528407)** (6 条消息): 

> `CUDA 使用情况, CGO 影响, GPU 利用率` 


- **CUDA 使用率骤降至零**：一位成员报告说，运行代码时 CUDA 使用率显示约为 **0%**，但切换到 **CUDA** 后增加到 **100%**。
   - 他们寻求帮助以理解初始问题，即即使启用了 CUDA，利用率仍显示为 **0%**。
- **怀疑 CGO 导致问题**：一位成员提到他们的程序显示为 **C** 程序而不是 **C+G**，并且是通过 **CGO** 运行的，怀疑这可能是根本原因。
   - 他们报告说程序显示为 **C**（而非 **C+G**），这表明 **CGO** 可能干扰了正确的 GPU 检测。
- **GPU 利用率显示异常结果**：一位成员注意到系统在所有进程（包括 `explorer.exe`）中都将 GPU 利用率显示为 **N/A**。
   - 该成员还表示他们的程序对所有内容都显示 **0%** 的利用率。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1374372989226975232)** (1 条消息): 

> `CUDA graph 模型捕获, 模型中的分布式操作` 


- **CUDA Graph 捕获模型的分布式操作**：一位成员报告说，当模型被 **CUDA graph** 捕获时，模型中的所有分布式操作都会被放置在它们自己的 stream 中。
- **Stream 放置咨询**：他们询问这种行为是否符合预期，暗示了对 **CUDA graph** 模型执行的潜在影响。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1374123689582530681)** (3 条消息): 

> `MAXSUN Arc Pro B60 Dual, SageAttention, Gemini Diffusion` 


- **MAXSUN Arc Pro B60 Dual 评测**: 分享了一个 **MAXSUN Arc Pro B60 Dual** 的 [YouTube 评测](https://www.youtube.com/watch?v=Y8MWbPBP9i0)。
- **SageAttention 的整洁代码库**: 强调了 **SageAttention** 的整洁代码库，并附上了 [GitHub 仓库](https://github.com/thu-ml/SageAttention/)链接。
- **Google 凭借 Gemini 进军 Diffusion 领域**: 提到了 Google 的 **Gemini Diffusion** 模型，标志着 Google 进入了 Diffusion 模型领域，并附带了 [DeepMind 模型页面](https://deepmind.google/models/gemini-diffusion/)的链接。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1374150496331239484)** (4 条消息): 

> `Axolotl QAT/PTQ Workflow, Llama3.2 Quantization, OpenAssistant/oasst1 Dataset Evaluation` 


- **探讨 Axolotl QAT/PTQ 工作流**: 成员们详细研究了 **Axolotl** 的 QAT/PTQ 工作流，比较了它们各自用于微调和量化的配置。
   - 两种流程之间的关键区别在于是否缺少 `prepare_model_for_qat`。
- **Llama3.2 量化命令公开**: 从 **Llama3.2** 配置开始的量化命令为 `axolotl train config.yaml`，然后是 `axolotl quantize config.yaml`。
   - 相同的 QAT 配置被传递给量化 CLI，以确保应用相同的 PTQ 模式。
- **OpenAssistant/oasst1 数据集显示出良好结果**: 一位成员报告称，在相同配置下使用默认设置，**OpenAssistant/oasst1** 数据集取得了良好的结果。
   - 讨论表明该数据集是评估模型性能的一个可行选择。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1374235749909663796)** (6 条消息): 

> `Microsoft Build Conference, Network Connection Issues, LB broken` 


- **Build 大会吸引了惊人气场**: 一位成员提到参加了在西雅图举行的 **Microsoft Build 大会**，并注意到演讲者拥有*惊人的气场*。
   - 目前尚不清楚他们指的是哪位演讲者，但他们似乎对演示印象深刻。
- **Keynote 期间连接不稳定**: 一位用户开玩笑说在 Keynote 演讲期间只有 **2kbps 的连接速度**。
   - 这意味着他们在尝试远程关注活动时遇到了技术困难。
- **LB 故障**: 一位用户提到在查看他们 **LB** 上*所有损坏的东西*。
   - 目前尚不清楚 LB 指的是什么，但很可能是 Load Balancer（负载均衡器）。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1374460471666671666)** (1 条消息): 

> `CUDA Developer Meet Up, NVIDIA, UCL, London, Python-native GPU programming` 


- **NVIDIA 与 UCL 在伦敦举办 CUDA 开发者见面会**: NVIDIA 和 UCL 将于 **6 月 5 日**在**伦敦**举办 **CUDA 开发者见面会**，邀请各级开发者与 CUDA 专家交流。
   - 活动承诺将进行关于 **Python-native GPU programming** 的实战演讲，探索 **nvmath-python** 和 **cuda-core** 等新库，并提供吃着披萨进行社交的机会，还有机会赢取 NVIDIA 周边（[在此报名](https://events.nvidia.com/cudadevelopermeetup/?ncid=ref-dev-171762-0503)）。
- **在伦敦见面会与 CUDA 专家交流**: 参加在伦敦举行的 CUDA 开发者见面会，与来自 NVIDIA 和 UCL 的 CUDA 专家建立联系。
   - 活动还将通过实战演讲探索 Python-native GPU programming，并介绍 nvmath-python 和 cuda-core 等新库。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1374432246454288476)** (1 条消息): 

> `OpenEvolve release, Evolutionary coding agents, LLMs for algorithm optimization` 


- **OpenEvolve 开源算法演化系统**: 一位成员发布了 **OpenEvolve**，这是 Google DeepMind 的 AlphaEvolve 系统的开源实现。
   - 这是一个使用 **LLM** 来发现和优化算法的*演化编程 Agent*；作者成功在圆堆积（circle packing）问题上复现了 DeepMind 的结果（**匹配度达 99.97%！**）。
- **使用兼容 OpenAI 的 API 演化代码库**: **OpenEvolve** 演化的是整个代码库（而不仅仅是单个函数），并且可以与任何兼容 **OpenAI** 的 API 配合使用。
   - 它使用 **LLM ensemble 方法**以获得更好的结果和多目标优化；请在 [GitHub](https://github.com/codelion/openevolve) 和 [博客文章](https://huggingface.co/blog/codelion/openevolve) 中查看。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1374111212316393513)** (2 messages): 

> `Reasoning Models, Pass @K` 


- **推理模型启发式方法**：一位成员建议 pass @20 是*穷人版的推理模型*。
   - 另一位成员表示赞同，指出最初的想法就是 pass @20 相当于一个**推理模型 (Reasoning Model)**。
- **Pass @K 讨论**：讨论围绕使用较高的 pass 值 (@20) 作为提升模型推理能力的启发式方法展开。
   - 这意味着这种方法可以作为更复杂推理模型的简化替代方案。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

simran9493: https://www.youtube.com/watch?v=xcpEl0cGCC4
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

rasdani: 太棒了！期待论文 🙂
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1374099866875068477)** (47 messages🔥): 

> `MI300 Leaderboard Updates, AMD-FP8-MM performance, Histogram Leaderboard, MLA Decode Results, Mixture of Experts Leaderboard` 


- **MI300 的 AMD-FP8-MM 竞赛升温！**：大量提交涌入 `amd-fp8-mm` 排行榜，其中一名成员在 **MI300** 上以 **121 µs** 夺得**第一名**，另一名成员以 **132 µs** 位居第二。
   - 其他几项提交也成功在 **MI300** 上运行并取得了各种速度成绩，展示了在 `amd-fp8-mm` 类别中激烈的优化竞争。
- **Histogram 达到新高度**：一名成员在多个平台的 `histogram` 排行榜上均获得**第一名**，在 **A100** 上为 **36.3 µs**，在 **H100** 上为 **23.5 µs**，在 **L4** 上为 **68.5 µs**。
- **MLA Decode 统治地位**：一名用户在 `amd-mla-decode` 排行榜上以惊人的 **7351 ms** 在 **MI300** 上稳居**第一名**，而其他用户分别以 **7574 ms** 和 **8875 ms** 获得**第二**和**第三名**。
   - 这些结果突显了 `amd-mla-decode` 领域持续的进步和竞争。
- **AMD Mixture of Experts 表现出色**：`amd-mixture-of-experts` 排行榜的提交显示出巨大潜力，一名用户在 **MI300** 上以 **9.45 ms** 获得**第一名**，另一名以 **9.70 ms** 获得**第三名**。
   - 更多关于 **128 ms** 和 **25.4 ms** 成功提交的报告显示这是一个充满活力的竞技场。
- **已实现复制粘贴健全性检查 (Sanity Check)**：在一名成员测试了相关问题后，团队确认现在已启用“健全性检查”，以防止代码意外泄露。
   - 他们感谢用户的通知，并表示将修复任何漏洞，确保代码安全。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1374174960213295116)** (6 messages): 

> `Leaderboard Explanations, Histogram Submission Error` 


- **征集排行榜方案解析**：组织者要求排行榜前列的参与者为其方案编写简短的解析用于教学目的，并计划在 **GPU Mode 网站**以及与 **AMD** 合作的研讨会上分享。
   - 目标是帮助他人学习编写高性能 **GPU 代码**，并强调此次竞赛是促进此类学习的绝佳方式。
- **L4 GPU 上的 Histogram 提交遇到障碍**：一名成员报告在向 **L4 GPU** 的 **histogram 排行榜**提交方案时收到“意外错误”。
   - 管理员要求提供更多细节，询问用户是否检查了常见问题（如代码中是否存在 `\`），以及是使用 Discord 还是 CLI 进行提交的。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1374118133723566222)** (14 条消息🔥): 

> `FLE 使用场景与评估、用于 Agent 原型设计的 Colab 服务器、Factorio TAS 生成器、FLE 的 Gym 接口、会议时间协调` 


- ****FLE** 使用场景与评估领域扩展**：最初的目标是了解（无界）环境所开启的有趣使用场景和评估领域。
   - 原始实验室和开放游戏是显而易见的起点，但成员们表示，目前*仅仅触及了皮毛*。
- **Colab 服务器助力 Agent 原型设计**：一位成员本周将创建一个 Colab 服务器，以简化 Agent 原型设计，并引用 [tbench.ai](https://www.tbench.ai/) 作为便捷贡献/测试 Agent 的灵感来源。
   - 他们还建议参考 **OpenAI Gymnasium** ([gymnasium.farama.org](https://gymnasium.farama.org/index.html)) 获取灵感，并提到了使用 OpenAI gym 的 **Script World** ([github.com/Exploration-Lab/ScriptWorld](https://github.com/Exploration-Lab/ScriptWorld))。
- ****Factorio TAS Generator** 引起关注**：一位成员正在寻找全职项目，这可能是一个选择，参考了 **Factorio TAS Generator (FTG)** ([github.com/MortenTobiasNielsen/Factorio-TAS-Generator/issues/76](https://github.com/MortenTobiasNielsen/Factorio-TAS-Generator/issues/76))。
   - 该成员创建 **FTG** 是为了给 **Tool Assisted Speedruns**（工具辅助速通）提供接口，并曾用它创造了两项世界纪录，目前该工具被 **Factorio Discord 服务器**上的速通玩家在 **tas 子频道**中使用。
- **FLE 的 Gym 接口正在开发中**：**FLE** 最初支持 **Gym**，但后来从 **Markov-Decision Process**（马尔可夫决策过程）方法转向了 **LLMs**。
   - 虽然 **FLE** 弃用了 **Gym**，但一位成员创建了一个 issue ([github.com/JackHopkins/factorio-learning-environment/issues/197](https://github.com/JackHopkins/factorio-learning-environment/issues/197)) 考虑将其暴露出来，尽管它不需要直接使用 **Gym**。
- **会议时间协调误会已消除**：一位成员最初对 **16:00 UTC** 的会议表示存在时间冲突。
   - 随后他们意识到自己误解了 **UTC** 与**哥本哈根时间**的区别，他们的会议实际上是在 **13:00-16:00 UTC**，因此他们可能在回家途中参加会议。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1374100393377660989)** (32 条消息🔥): 

> `MLA 解码内核、提交文件大小限制、FP8-GEMM 问题、MoE 提交不可用` 


- **MLA 解码内核发布**：MLA 解码内核现已发布，一位用户幽默地宣称获得第一名并挑战他人超越其方案，详见[此图片](https://cdn.discordapp.com/attachments/1359640791525490768/1374100393113423912/image0.jpg?ex=682e23b7&is=682cd237&hm=cc92ff0c4f6e348a18ce4c40d8475712e86c856d3e20d039a55d4ce1d2257fce&)。
- **提交文件大小限制困扰**：用户在提交代码时遇到问题，方案大小超过了 **50KB**，但通过删除注释和未使用代码成功提交。[此图片](https://cdn.discordapp.com/attachments/1359640791525490768/1374134699576004758/image.png?ex=682e43aa&is=682cf22a&hm=c7a3a66e15371a4fc23e92c2a234d6bb28be8937030ec55423509cba879550f5&)展示了错误截图。
   - 经确认，文件大小限制约为 **35KB**，一位开发者提到这是由于 GitHub Action 的 payload 限制导致的，并承诺在*明天*推送修复程序以提高限制。
- **FP8-GEMM 提交错误困扰用户**：多位用户报告了 **FP8-GEMM** 提交问题，即使是之前正常的提交现在也会报错，但基准测试（benchmarking）仍显示 *OK*。[此截图](https://cdn.discordapp.com/attachments/1359640791525490768/1374287640312152084/Screenshot_from_2025-05-20_09-27-07.png?ex=682e295a&is=682cd7da&hm=a21cf5c1404376bdf0d804de9c5ec39a63679f5d3692f9387d6271e3e17d936a&)详细列出了收到的错误。
   - 一位开发者表示将进行调查，并请求获取失败的文件以复现问题并找到根本原因。
- **MoE 提交完全挂了吗？**：一些用户报告 **MoE** 提交完全无法工作，一位开发者尚未尝试 **MoE** 提交以确认问题，但表示当时已没有时间进行调查。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1374118438758518794)** (7 messages): 

> `Cutlass DSL Python Windows support, CUTLASS thread tiling error, CUTLASS GTC slide outdated` 


- **CUTLASS 线程平铺问题**：一位用户在 CUTLASS 中遇到了线程平铺（thread tiling）错误，具体是 [此 GitHub Gist](https://gist.github.com/simveit/ab0a28efb4338592f82c0a8f762f0ac7) 中描述的 `transpose_naive_kernel` 函数内的 `cute.local_partition` 触发了 `AssertionError`。
   - 此问题已被确认为 bug，目前 CUTLASS 开发人员正在处理中（[issue #2314](https://github.com/NVIDIA/cutlass/issues/2314#issuecomment-2892669461)）。
- **CUTLASS GTC 幻灯片过时警报**：一位用户指出 **CUTLASS GTC 幻灯片已过时**，建议用户参考最终发布版本中的更新示例。
- **CUTLASS 渴望 Windows 支持**：一位用户询问了 **CUTLASS DSL Python** 在 **Windows** 上的计划支持情况。
   - 一位开发人员确认需要 **Windows 支持**，但未提供预计完成时间（ETA）。 


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1374385171897516103)** (2 messages): 

> `Picograd, Rust implementation, Pedagogical Resource` 


- ****Picograd**：从零到英雄**：[singularity systems: zero -> hero](https://j4orz.ai/zero-to-hero/) 课程紧随 karpathy 的 [neural networks: zero -> hero](https://karpathy.ai/zero-to-hero.html)，将 [micrograd](https://github.com/karpathy/micrograd) 演变为 [picograd](http://github.com/picograd)。
   - 该项目的**短期目标**是实现能够训练和推理基础网络（ffn, rnn, lstm, gpt）的 picograd，并录制“从零开始逐行讲解”的视频讲座。
- ****Rust 规则**：Picograd 削减 Python 实现**：该项目目前的目标纯粹是针对 [单一 Rust 实现](https://github.com/j4orz/picograd)。
- ****教学版 Picograd**：学习资源**：如果做得好，picograd 将成为 PyTorch 的**标志性**教学资源，就像 clang 之于 chibicc，LLVM 之于 qbe，linux 之于 xv6 等。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1374127982343622706)** (95 messages🔥🔥): 

> `Google's Code Agent, Google I/O Announcements, Gemma 3n Model, Gemini Diffusion Model, Decentralized AI` 


- ****Google 的 Code Agent** 引发竞争狂潮**：Google 正在推出一款新的 **code agent** ([real.azure](https://jules.google/real.azure))，引发了 AI 社区对其功能和潜力的兴奋与对比讨论。
   - 一些成员已进入 **waitlisted**（候补名单），急切期待获得访问权限，而另一些人则推测在 **Google I/O** 上会有正式发布。
- ****Gemma 3n** 首次推出新架构**：**Gemma 3n** 是一款针对边缘设备优化的新型生成式 AI 模型，采用了 [这篇论文](https://arxiv.org/pdf/2310.07707) 中详述的新架构，并在参数高效处理方面进行了创新。
   - 该模型旨在减少内存占用，包含音频输入处理，并整合了文本和视觉数据，标志着它是一款真正的边缘模型，适用于智能设备，可能包括智能眼镜。
- ****Gemini 2.5 Flash** 和 **Gemma 3n** 可在 AI Studio 中访问**：成员现在可以在 **Google AI Studio** 中访问 **Gemini 2.5 Flash 0520** 和 **Gemma 3n**，尽管某些功能（如 Deep Think）仍处于封闭测试阶段。
   - 一位成员指出 **Gemini diffusion model** 似乎*有点害羞*，在 8 次尝试中仅有 1 次生成了可运行的代码。
- ****去中心化 AI** 的赛博梦想**：一位成员表达了由 **Nous** 和 **Prime** 赋能的赛博朋克未来愿景，届时 **RTX 5090** 可以与 **RTX 3060** 协同进行训练和推理。
   - 他们设想*一个随时随地存在的全球 AI*，由去中心化系统控制，且无法被关闭。
- ****Diffusion Models** 旨在加速文本生成**：**Diffusion models** 正在被探索用于并行处理文本块并以非因果方式生成文本的能力，这有可能加速文本生成。
   - 成员们认为 **diffusion models** 可以实现并行的思维链（chain-of-thought）推理，并且理论上应该更擅长写笑话，因为它们从一开始就能考虑到笑点（punchline），尽管模型可能会受到 token 限制的约束。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1374541694640328704)** (3 messages): 

> `限制模型领域, 教育中的 AI 模型, Gemini Flash, AI 作为助教` 


- **模型领域限制证明是困难的**：一位成员正在研究目前哪种模型在紧跟指令方面表现最好，但另一位成员表示，*将模型限制在特定领域或任务中是非常困难的*。
- **AI 助教作为解决方案**：一位成员希望创建一个环境，让学生能够利用 **AI 模型**的力量来增强自身发展并加速学习，同时保持与人类教学的健康平衡。
   - 他们的目标是至少限制模型避免直接解决问题，而是充当引导者或助手，就像老师一样，支持学生通过个人努力得出自己的结论。
- **AI 模型被用作教师替代品**：一位成员观察到当今教育中普遍存在的一个问题，即学生越来越多地将 **AI 模型** 作为教师的完全替代品，这使得面对面教育显得可有可无。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1374138539549397053)** (1 messages): 

> `LLM 自发产生社会规范, 去中心化 LLM 群体中的集体偏见, 对抗性 LLM Agent 推动社会变革` 


- **LLM 自发社交并产生规范**：根据一篇新论文 [“Social conventions spontaneously emerge in decentralized LLM populations”](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)，被普遍采纳的社会规范会通过局部交互在去中心化的 **LLM 群体**中自发出现。
- **LLM 表现出集体偏见**：该论文还指出，即使 **Agent** 个人没有表现出偏见，在这个过程中也可能出现强烈的**集体偏见**。
- **对抗性 LLM 作为社会变革 Agent**：研究发现，坚定的少数群体——**对抗性 LLM Agent** 一旦达到临界阈值，就可以通过强加替代规范来推动社会变革。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1374432584502743180)** (3 messages): 

> `OpenEvolve 发布, 进化编程 Agent, Google DeepMind 的 AlphaEvolve, 用于算法优化的 LLM, Matformer 架构` 


- ****OpenEvolve** 在 GitHub 上发布**：**Google DeepMind** 的 **AlphaEvolve** 系统的开源实现 **OpenEvolve** 已经发布，它以 **99.97%** 的匹配度复现了 DeepMind 在圆堆积（circle packing）问题上的结果。
   - 该 **Agent** 使用 **LLM** 来发现和优化算法，进化整个代码库，并可与任何兼容 **OpenAI** 的 **API** 配合使用，采用多目标优化和 **LLM** 集成方法，相关链接见其 [GitHub 仓库](https://github.com/codelion/openevolve)和[博客文章](https://huggingface.co/blog/codelion/openevolve)。
- **重点重新转向 **Matformer** 架构和 **Gemma 3n** 模型**：随着 **Gemma 3n** 模型的新发布以及对 **Matformer** 架构的重新关注，算法优化领域展现出巨大的潜力。
   - 更多信息请参见[这条推文](https://x.com/ditpoo/status/1924910661483843820)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1374138539549397053)** (1 messages): 

> `LLM 社会规范, LLM 中的集体偏见, 对抗性 LLM Agent` 


- **LLM 自发采纳社会规范**：一篇新[论文](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)发现，被普遍采纳的**社会规范**会通过局部交互在去中心化的 **LLM 群体**中自发出现。
- **LLM 表现出强烈的集体偏见**：同一篇[论文](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)表明，即使 **Agent** 个人没有表现出偏见，在这个过程中也可能出现强烈的**集体偏见**。
- **对抗性 LLM 推动社会变革**：该[论文](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)显示，坚定的少数群体——**对抗性 LLM Agent** 一旦达到临界阈值，就可以通过强加替代规范来推动社会变革。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1374121792146505748)** (52 条消息🔥): 

> `Xet 文件大小限制, HuggingFolks 角色, LLM 推荐, 训练数据错误, Hugging Face 协作` 


- **Xet 团队从容应对文件大小限制**：一名 **Xet 团队成员** 回应了关于文件大小限制的问题，表示虽然限制确实存在，但他们仍在确定具体细节，目标是至少达到 **200GB** 以适配 **70B 模型**。
   - 他们强调了使用 **启用 Xet 的路径**（例如[这个示例](https://huggingface.co/jsulz/Llama-3.3-70B-Instruct-bigtensor)）上传和下载 **>50GB** 文件的能力，但 Web 端的全面支持仍在开发中。
- **HuggingFolks 角色申请升温**：一名成员询问是否可以被授予 **HuggingFolks 角色**，目前正等待 Ben 的权限设置。
   - 获得该角色的用户还想知道如何让头像上方出现一只猫（这是一种 [avatar decoration](https://support.discord.com/hc/en-us/articles/13410113109911-Avatar-Decorations)）。
- **成员寻求快速 LLM 建议**：一名成员请求推荐适合在本地运行的 **快速、轻量且强大 的 LLM**。
   - 另一名成员建议使用 **Ollama** 配合来自排行榜、社交媒体或匿名论坛的 **GGUF** 版本模型（[huggingface.co/docs/hub/ollama](https://huggingface.co/docs/hub/ollama), [huggingface.co/models?sort=trending&search=gguf](https://huggingface.co/models?sort=trending&search=gguf), [huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a)）。
- **训练数据错误困扰训练过程**：一名用户报告了一个持续出现的错误：在获取训练数据时提示 *Maximum retry attempts exceeded for page*，并寻求原因和解决方案。
   - 一名成员建议查看 [LLM 和 RLHF 课程](https://huggingface.co/learn/llm-course/chapter12/2)，并提到了使用 **Streamlit** 作为 GUI。
- **生物信息学专家关注 Hugging Face 协作**：一名在 **EMBL-EBI** 从事生物数据工作的成员（[https://www.ebi.ac.uk/](https://www.ebi.ac.uk/)）表示有兴趣与 Hugging Face 团队合作，利用 **LLMs**、**RLVR** 和生物策展（biocuration）数据进行自动化策展。
   - 另一名成员建议联系特定用户以探讨潜在的合作。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1374414979972731043)** (1 条消息): 

> `研究论文阅读工作流, 总结工具, YouTube 视频讲解, 论文筛选标准` 


- **用户寻求高效的研究论文工作流**：一名用户正在寻求一种更高效的论文阅读工作流，特别是针对阅读速度不快或不擅长数学的人群；该用户目前使用 **Gemini 2.5 Pro** 来总结粘贴的文章 PDF。
   - 该用户目前的工作流包括要求 **Gemini** 解释每个变量，并通过反复提问来深化理解，但同时也在寻找替代方案，如 **YouTube** 视频讲解。
- **利用 YouTube 进行论文讲解**：该用户建议利用 **YouTube** 视频来理解研究论文，同时也承认需要筛选掉低质量的内容。
   - 他们认为讲解特定论文的高质量视频非常有价值，并想知道其他人是如何选择论文的，以及完全理解一篇论文需要多长时间。
- **论文筛选标准**：用户询问其他人是如何选择阅读哪些论文的，以及彻底理解一篇论文需要多长时间。
   - 他们还提到自己在数学方面不是很聪明，甚至在编程方面也是如此。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1374103171017736335)** (12 messages🔥): 

> `Video dropping page, Browser AI tool calls, Data transformations with LLMs, MCP server support, Cyberdesk computer agent` 


- ****Video Dropping Page** 历时两天完成**：经过两天的努力，一名成员成功完成了他们的 **video dropping page**（视频投放页面），其中包括对 FLAST 的研究和全新的设计。
   - 该成员对不轻言放弃并完成任务表示非常满意，并附上了[已完成的视频投放页面](https://cdn.discordapp.com/attachments/897390720388825149/1374103170376138783/Day_19.png?ex=682e264d&is=682cd4cd&hm=74724ad363d289a8239719f635d07be371b4389430b6d243cdca52d254136810)的截图。
- ****Browser AI** 支持工具调用 (Tool Calls)**：新版本的 **Browser.AI** 已发布，现已支持 **tool calls**。
   - **Browser.AI** 是一个原型浏览器，旨在展示在设备上运行开源模型的强大能力，目前支持聊天、**tool calls** 和 **embeddings**；欢迎通过 [browser.christophermckenzie.com](https://browser.christophermckenzie.com/) 提供反馈。
- ****Vitalops** 创建数据转换工具**：Vitalops 推出了一款名为 [DataTune](https://github.com/vitalops/datatune) 的新开源工具，该工具可以通过简单的自然语言指令和 **LLMs** 执行数据转换。
   - 该工具旨在缓解在使用 **LLMs** 进行数据转换时对上下文长度（context length）和高昂 API 成本的担忧。
- ****Cyberdesk** 在周末构建计算机智能体 (Agent)**：一名成员介绍了 **Cyberdesk**，这是一个可以在单个周末内利用任何 Hugging Face 基于像素的模型（pixel-based model）构建计算机智能体的工具。
   - 可以在 [ProductHunt](https://www.producthunt.com/products/cyberdesk) 找到其发布页面，欢迎点赞支持。
- ****OpenEvolve** 系统发布**：Google DeepMind 的 **AlphaEvolve** 系统的开源实现 **OpenEvolve** 已发布。
   - 这个进化编程智能体（evolutionary coding agent）可以在 [GitHub](https://github.com/codelion/openevolve) 上找到，它使用 **LLMs** 来发现和优化算法，成功复制了 DeepMind 在圆堆积（circle packing）上的结果，并将随机搜索进化为模拟退火算法，更多详情见其 [博客文章](https://huggingface.co/blog/codelion/openevolve)。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

arpitbansal.: 请问最近一次会议有录像吗？？
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1374134772284133468)** (6 messages): 

> `Stanford CS231n lectures, Estimating bathymetry (sea depth) from Sentinel 1 SAR images, Object Detection, Segmentation Model` 


- **斯坦福 CS231n 课程非常出色**：一名成员建议学习 Andrej Karpathy 的 [斯坦福 CS231n 课程](https://cs231n.stanford.edu/)，因为其直觉引导非常到位，同时还推荐了经典的计算机视觉和机器学习资源，如 YouTube 上的 Andrew Ng 课程。
- **模型建议**：一名成员建议选择像 **YOLOv11-seg** 这样的分割模型来获取物体轮廓，它可以提供分割掩码（segmentation mask），以便使用 *cv2.findContours* 绘制轮廓。
- **利用 Sentinel 1 SAR 图像估算水深**：一名成员寻求关于从 **Sentinel 1 SAR 图像**估算**水深 (bathymetry)** 的指导，利用同一位置在不同时间的多个快照作为输入，以推断由于不同成像条件和潮汐产生的水深变化。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1374119738170671184)** (1 messages): 

> `BERT-style model inference, Logit Differences, Candle vs PyTorch` 


- **跨框架的 BERT Logit 差异**：一名成员询问，在 **Candle** 和 **PyTorch** 等不同库中对相同的 **BERT-style model** 和任务进行推理时，即使使用完全相同的 **tokenization** 和配置文件，获得显著不同的 **logits** 是否属于异常情况。
   - 他们注意到，尽管 **logit** 存在差异，但分类结果在很大程度上保持一致。
- **推理差异：Candle 对比 PyTorch**：尽管 **tokenization** 和配置相同，但在 **Candle** 和 **PyTorch** 中运行相同的 **BERT-style model** 可能会产生显著不同的 **logits**。
   - 用户强调，虽然 **logits** 不同，但最终的分类结果基本一致。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1374106951159648256)** (30 messages🔥): 

> `GAIA 格式问题, Ollama 设置帮助, LiteLLMModel, InferenceClientModel, 在 LinkedIn 上分享 AI Agent 课程证书` 


- **GAIA 格式要求完全匹配**：成员们讨论了 [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) 要求答案必须完全匹配，这可能非常繁琐，并建议通过*调整 system message* 来改善结果。
   - 有人提到 **smolagents** 在不修改 system message 的情况下，提交的格式对 GAIA 更友好。
- **Ollama 设置故障排除**：一名课程参与者在设置 **Ollama**、运行 `ollama pull` 和 `ollama serve` 后寻求后续步骤指导，他们在调试错误时看到了带有 CPU 配置的时间戳。
   - 另一名成员建议在课程代码中将 `InferenceClientModel` 替换为 `LiteLLMModel`，以便有效地使用 **Ollama**。
- **Agent 课程中推荐使用 LiteLLMModel**：当用户在课程代码中遇到 `InferenceClientModel` 时，应将其更改为 `LiteLLMModel` 以使用 Ollama。
   - 分享了一个来自 GitHub 的更新版 notebook 链接，该版本解决了 LoopEvent 章节中的 Bug（[notebook 链接](https://colab.research.google.com/github/huggingface/agents-course/blob/main/notebooks/unit2/llama-index/workflows.ipynb)）。
- **克隆项目仓库可修复 Bug**：一名用户报告通过重新克隆项目仓库并复制文件解决了问题，并指出在设置中进行完整重建（full rebuild）也无法解决该问题。
   - 他们表示 *Spaces 中有些损坏的东西，即使在设置中进行完整重建也无法修复*。
- **证书分享困境**：一名获得了 Agent 课程最终证书的用户询问是否可以将其分享到 **LinkedIn**。
   - 在给出的消息中未找到解决方案。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1374141785638441074)** (92 messages🔥🔥): 

> `Perplexity 免费层级成本, AI 构建者调查, Jules: 异步编程 Agent, 编程 Agent 对比, 论坛上的反 AI 情绪` 


- **Perplexity 在免费层级上花费数百万美元**：根据[这条推文](https://x.com/breadcrumbsre/status/1924474011125227687)，Perplexity 每年在免费层级上花费 **3300 万美元**，并将该成本归类为 R&D（研发）。
- **Jules：Google 的异步编程 Agent 发布**：来自 Google 的异步编程 Agent **Jules** 已经推出，并在相关帖子及其[网站](https://jules.google/)上进行了展示。
- **编程 Agent 的 React 技能面临考验**：成员们讨论了当前编程 Agent 在处理复杂 **React** 代码时的局限性，指出模型经常产生 *'maximum update depth exceeded'* 错误，并且在重构现有代码库时表现挣扎。
- **论坛上反 AI 情绪蔓延**：关于 **Hacker News** 和 **Reddit** 等平台上盛行的反 AI 情绪引发了讨论，一些成员认为这些论坛反映的是“回声筒”效应，而非广泛的社会观点。
   - 一名成员建议需要 *'一个对 AI 友好的 Hacker News'*，另一名成员则提议当前的 Discord 频道就充当了这一角色。
- **Gemini 演变为 AI 操作系统**：根据[这个线程](https://x.com/heyshrutimishra/status/1924900609410662592)，在 **Google I/O 2025** 上，Gemini 已演变为一个完整的 **AI 操作系统**，拥有一套超越聊天功能的全新工具。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1374144387851882508)** (27 messages🔥): 

> `MLOps 课程, 在单张图像上训练的特征, 使用 torch_geometric 实现 GNN` 


- **寻求 MLOps 课程推荐**：一名成员请求推荐学习 MLOps 的课程，并对另一名成员的工作表示赞赏。
   - 另一名成员回应说，这些问题在他们看来*并不愚蠢*。
- **单图训练效果很差**：一名成员表示 *在单张图像上训练的特征效果很差*，而对特征使用**隐式重正则化（implicit heavy regularization）**的方法会使它们更平滑。
   - 这引发了频道内两名成员之间的简短争论。
- **使用 `torch_geometric` 实现 GNN 的问题**：一名成员询问了关于使用 `torch_geometric` 实现 **GNN** 进行节点回归任务的问题，特别是如何构建输入数据。
   - 另一名成员表示稍后会查看该问题，并指出这个问题有点具体且他们时间有限，同时提供了以下代码示例：`class GAT(torch.nn.Module):`


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1374173857916911677)** (16 条消息🔥): 

> `Physics of Language Models, Knowledge Storage, Knowledge Extraction, Knowledge Manipulation, Out-of-Distribution Buzzword` 


- **讨论 Physics of Language Models**：频道成员讨论了 [Physics of Language Models: Part 3.1, Knowledge Storage and Extraction](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633) 和 [Part 3.2, Knowledge Manipulation](https://ssrn.com/abstract=5250621)，重点关注 Language Models 如何处理事实性知识及其在下游任务中的应用。
- **对 Knowledge Manipulation 的评述**：讨论强调，虽然 Language Models 精通知识 **retrieval**，但在 **classification**、**comparison** 和 **inverse search** 等任务中会遇到困难，尤其是在不使用 Chain of Thoughts (CoTs) 的情况下。
   - 该论文介绍了一个*受控的合成实验*，揭示了 Language Models 固有的弱点：即使知识被完美存储且经过充分训练，它们在有效处理预训练数据方面仍然表现挣扎。
- **质疑 "Out-of-Distribution" 术语**：参与者辩论了 "out-of-distribution" 一词，一些人认为由于其定义模糊且经常被误用，它已成为一个 **buzzword PR term**。
   - 有人指出，在讨论的论文中，作者采取了一种非典型的方法，比大多数研究更有效地控制了 data contamination。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1374103433174323221)** (28 条消息🔥): 

> `Alpha Evolve, Google Codex Competitor, Labor Saturation Theory, LatentSeek vs COCONUT, Gemini Diffusion` 


- **Google 发布名为 Jules 的 Codex 杀手**：Google 推出了 **Jules**，它是 **OpenAI Codex** 的竞争对手，但目前在欧洲仅限 waitlist 访问 ([Jules](https://jules.google/))。
- **关于 AI 对劳动力影响的辩论**：围绕 **AI** 对劳动力市场的影响展开了讨论，一位成员建议，如果劳动力是一种商品，市场理论表明在 **labor saturation** 的情况下，我们应该减少劳动力的创造。
   - 另一位成员指出，现代社会已经面临 **population stagnation** 或下降，从而自然地调节了供应侧。
- **LatentSeek 旨在实现 Tokenless Decoding**：**LatentSeek** 在选择下一个 token 之前，对 latent（以及之前的 tokens）采用 per token RL。
   - 成员们指出了它与 **COCONUT** 的区别，后者绕过了 tokenization 步骤，在没有 discretization 的情况下在 latent space 中运行 autoregression。他们注意到 **LatentSeek** 相对于 **Qwen 2.5** 的提升*非常有限*，可能只是一个 benchmark fine-tune。
- **Google 揭晓 Gemma 3N 模型细节**：Google 发布了其 **Gemma 3N** 模型系列的详细信息 ([Gemma 3N Docs](https://ai.google.dev/gemma/docs/gemma-3n))。
   - 随后发布了关于 **Google AI Edge** 小型语言模型、multimodality、RAG 和 function calling 的消息 ([Google AI Edge](https://developers.googleblog.com/en/google-ai-edge-small-language-models-multimodality-rag-function-calling/))。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1374112153271013457)** (54 条消息🔥): 

> `Qwen MoE 3, Qwen 2 35B Polyglot 基准测试, Aider 通知, Aider 作为 Agent, Navigator PR` 


- ****Qwen** 在 Angular/Typescript 开发中表现令人失望**：一位成员发现 **Qwen 2 35B** 模型无法为使用 `element.innerHTML` 的 Angular/Typescript 代码生成 diffs，尽管多次尝试，最终导致在该任务中放弃了该模型。
   - 他们使用了与 Paul Gauthier 相同的 Alibaba Cloud 端点，并尝试调整了 temperature 参数。
- ****Qwen** 在不同语言中的表现各异**：一位成员发现 **Qwen** 在 Rust 中表现良好，而另一位成员在 Kotlin 中遇到了失败，特别是在不可变 Map（immutable map）问题上，即使代码中已经提供了类型信息。
   - 一位成员链接了 [Paul 关于 Qwen3 的博客文章](https://aider.chat/2025/05/08/qwen3.html) 以获取 temperature 或其他设置参考。
- ****Aider 通知**已配置**：一位用户询问如何在 Aider 中启用通知，以便在编码完成或请求添加文件时获得提醒，并被引导至 [配置选项](https://aider.chat/docs/config/options.html#--notifications)。
   - 他们正在努力添加一个 [Navigator PR](https://github.com/Aider-AI/aider/pull/3781) 以扩展到 Agentic 编程。
- ****Gemma 3n 4B** 在 Polyglot 基准测试中达到 44.4%**：新的 **Gemma 3n 4B 模型** 在 Polyglot 基准测试中达到了 44.4%，但目前尚不清楚所使用的设置。
   - 一位成员指出 *一个 4b 模型的得分竟然高于 Gemini 2.0 Pro*。
- ****Deep Think** 是新的差异化特性？**：在 **Gemma 3n 4B 模型** 发布后，一位成员发布了 Google 博客的截图，强调下一个模型是 **Deep Think**。
   - 其他人评论道 *这听起来就像是你让他们随意挥霍你的钱。*


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1374117553701392525)** (15 条消息🔥): 

> `Aider Shell 命令执行, Aider YAML 配置, Aider Prompt 上下文, Gemini 2.5 Flash 基准测试` 


- ****Aider** 讨论 Shell 命令执行**：用户讨论了在 **Aider** 中执行 Shell 命令的可能性，特别是列出暂存文件（staged files），并指出使用 `--yes-always` 选项并未按预期工作，工具只是显示命令后就退出了。
   - 分享了 [issue #3903](https://github.com/Aider-AI/aider/issues/3903) 的链接，表明这是一个设计决定，需要手动确认。
- ****YAML** 配置探讨**：一位用户询问了 Aider 的优化 **YAML** 配置，得到的建议是 **Aider** 旨在配合默认配置良好运行，模型才是最关键的因素。
   - 分享了 [配置示例](https://github.com/burnettk/config-aider/tree/main/sample_config) 的链接，一位用户推荐使用 `GLOBAL_DEFAULTS.yml` 来偏离 Aider 的默认设置。
- **思考 Prompt 的文件上下文**：一位用户询问是否可以将文件中的 Prompt 前导内容（preamble）作为上下文包含进来，并讨论了应该在 Prompt 中指定，还是让 **Aider** 通过 `aider --read your_file.md` 直接读取文件。
   - 目前没有明确的解决方案，只是鼓励用户自行测试。
- ****Gemini 2.5 Flash** 进行基准测试**：用户讨论了使用新升级的 **Gemini 2.5 Flash** 更新基准测试，并在 **Discord** 频道内分享了 [基准测试结果](https://discord.com/channels/1131200896827654144/1374446248039551057/1374454536759939193) 的链接。
   - 消息中未详细说明具体结果及其影响。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1374123421905981450)** (64 messages🔥🔥): 

> `Manus AI Agent, 积分系统与邀请码, Manus 网站创建, 网络连接错误, Manus Tech Stack` 


- **Manus—拥有自己电脑的 AI Agent**: **Manus** 是一个**拥有自己电脑的 AI Agent**，可以构建网站、撰写报告并运行研究任务。
   - 用户可以通过邀请链接访问 Manus，例如 [这一个](https://manus.im/invitation/CWYMVAAXR6HZ)。
- **讨论了积分系统和邀请码系统**: 一位用户表示，除了*积分系统/使用额度和邀请码滥用*之外，他们在 Manus 上没有遇到任何问题。
   - 另一位用户问道：*你怎么有这么多积分？*
- **Manus 能够创建网站**: 一位用户询问 **Manus** 是否可以创建网站，另一位用户确认*绝对可以*，并链接到了 [Manus 官方用例集](https://manus.im/usecase-official-collection) 和 [来自用户的 Manus 用例](https://manus.im/usecase-from-user)。
- **网络连接错误困扰用户**: 一位用户报告在 Manus 中遇到**网络连接错误**。
   - 一名 Manus 员工请他们分享账号邮箱和会话链接，以协助调查该问题。
- **好奇的用户寻求 Manus Tech Stack**: 一位成员询问了正在使用的 **Tech Stack**，问道：*目前你们在这个项目上倾向于使用什么 Tech Stack？*
   - 另一位成员回答道：*我目前正在尝试各种创新。*


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1374190909771481189)** (28 messages🔥): 

> `范畴论与 AI, Cohere 研究资助计划, Cohere 的私有化部署选项, Command A 与结构化响应变慢, Command-R 的 JSON 输出挂起问题` 


- **范畴论热潮点燃 AI 兴趣！**: 成员们表达了希望更多了解**范畴论 (Category Theory)**与 **AI** 交叉领域的愿望。
   - 对话旨在探索与这对组合相关的 Token 提示词工程 (Prompt Engineering) 技术和方法。
- **Cohere Labs 持续推进资助计划！**: 一位用户询问了 **Cohere 研究资助计划 (Research Grants program)** 的状态，指出申请链接似乎已失效 ([https://share.hsforms.com/1aF5ZiZDYQqCOd8JSzhUBJQch5vw](https://share.hsforms.com/1aF5ZiZDYQqCOd8JSzhUBJQch5vw))。
   - 一位 Cohere 代表澄清说，**Research Grants** 现在是 **Cohere For AI** 的一部分，目前被称为 **Cohere Labs**。
- **Cohere 满足客户对控制权的需求**: 出于对数据/LLM 主权的兴趣，一位用户询问了关于 **Cohere 模型本地部署 (On-prem)** 并拥有完全所有权/控制权的**私有化部署选项**。
   - Cohere 确认他们将**私有化部署**作为其解决方案的核心部分，提供灵活的部署选项，并鼓励用户联系 [sales@cohere.com](mailto:sales@cohere.com) 或 [support@cohere.com](mailto:support@cohere.com) 获取更多信息。
- **Command A 运行缓慢，令客户困惑！**: 一位用户报告 **Command A** 的响应时间比平时慢，特别是在使用**结构化响应参数 (Structured Response Parameter)** 时。
   - 一位 Cohere 代表承认了该问题，确认他们端没有已知故障，并要求用户将详细信息发送至 [support@cohere.com](mailto:support@cohere.com) 进行调查。
- **Command-R 的 JSON 卡顿，只需等待！**: 一位用户报告说，在使用 **command-r-plus-08-2024** 时，在请求中指定 **json_object** 作为输出会导致挂起，尽管在 Web Playground 中不显式指定 JSON 输出时运行正常。
   - 该用户后来澄清说，请求确实会完成，但耗时过长（将近 2 分钟），而基于文本的 JSON 输出则快得多。


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1374148102197481513)** (1 messages): 

> `Vitalops datatune, 开源数据转换工具` 


- ****Vitalops** 发布 **datatune****: Vitalops 刚刚创建了一个名为 **datatune** 的新**开源工具**，它可以通过简单的**自然语言指令**和 **LLM** 进行**数据转换**。
   - 创建者鼓励大家在 [GitHub](https://github.com/vitalops/datatune) 上查看，并希望它能有所帮助。
- **Datatune 简化数据转换**: **Datatune** 是 Vitalops 开发的一个开源工具，利用 **LLM** 通过自然语言进行数据转换。
   - 它旨在简化数据转换的执行方式，使其能够被更多用户使用。


  

---

### **Cohere ▷ #[🎯-private-deployments](https://discord.com/channels/954421988141711382/1351999070247583848/1374435521777897625)** (2 messages): 

> `Private Deployment, Data Sovereignty, LLM Sovereignty, Cohere models on-prem` 


- **客户关注私有化、本地化 (on-prem) 的 Cohere 模型部署**：一位客户咨询了关于 **Cohere models on-prem** **私有化部署**的方案，由于对 **数据/LLM 主权** 的关注，寻求完全的所有权和控制权。
   - Cohere 团队成员建议通过 [sales@cohere.com](mailto:sales@cohere.com) 或 [support@cohere.com](mailto:support@cohere.com) 联系他们以进行进一步讨论。
- **Cohere 销售和支持团队准备提供帮助**：Cohere 团队确认了该咨询并表示愿意提供协助。
   - 他们请客户通过 [sales@cohere.com](mailto:sales@cohere.com) 或 [support@cohere.com](mailto:support@cohere.com) 联系，以探索各种可能性。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1374144349100707962)** (20 messages🔥): 

> `MCP best practices, MCP and Cursor, crawl4ai mcp server, A2A protocol Agents, Wallet MCP` 


- **寻求关于 MCP 最佳实践的深入探讨**：一位成员正在寻找关于 **MCP 使用最佳实践**的深入研究，特别是围绕工具设计和生产环境部署方面。
   - 他们链接了一篇关于保障 **Model Context Protocol** 安全并构建更安全的 Agentic 未来的 [Windows 博客文章](https://blogs.windows.com/windowsexperience/2025/05/19/securing-the-model-context-protocol-building-a-safer-agentic-future-on-windows/)。
- **分享 MCP 与 Cursor 工作流**：一位成员在 Reddit 上分享了一篇关于他们个人如何利用 **MCP 和 Cursor** 工作流来构建网站、在 Unity 中制作游戏以及其他类似任务的文章。
   - 他们还提到 **browsermcp** 非常酷，因为它允许 Agent 实际查看你在浏览器中看到的内容、进行截图/可视化等。
- **crawl4ai MCP Server 实现**：一位成员正在寻求 **Docker 中的 MCP server** 参考实现，该实现能够处理容器外的本地文件，以抓取（摄取和嵌入）本地 Markdown 文件，而不仅仅是 URL。
   - 他们有一个用于 Windsurf 的 **crawl4ai mcp server**，可以抓取 URL 并生成嵌入以进行最新的代码上下文知识查询，目前正以 SSE 传输端口在 Docker 容器中运行。
- **连接 MCP 与 A2A**：一位成员发布了一个开源服务器，用于 **连接 MCP 与 A2A 协议 Agents**，允许 Claude 与 A2A Agents 无缝交互，并分享了 [GitHub 链接](https://github.com/GongRzhe/A2A-MCP-Server)。
   - 另一位成员询问了 **A2A** 的使用场景，另一位成员解释说 A2A 的使用场景是将你的 Agents 像工具一样部署到你的域名中。
- **Wallet MCP 发布**：来自 TokenPocket 的团队发布了 **Wallet MCP** —— 一个轻量级桥接器，可实现 AI 客户端与加密用户钱包之间的无缝集成，支持多链资产管理、交易签名和智能合约交互，并分享了 [GitHub 链接](https://github.com/TP-Lab/mcp-marketplace/tree/master/apps/wallet-mcp)。
   - 对方解释说，加入服务器需要 **验证码** 的原因是骗子和垃圾邮件发送者太多了。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1374245521409052805)** (9 messages🔥): 

> `MCP-GraphQL issues, Public SearXNG MCP server, AI-friendly Data API` 


- **Fetch 错误困扰 GraphQL 测试**：一位成员在 **Claude desktop** 上测试 **mcp-graphql** 服务器时遇到了 *"Failed to introspect schema: ReferenceError: fetch is not defined"* 错误。
   - 另一位成员建议这可能是由过时的 **Node** 版本引起的。
- **公共 SearXNG MCP 服务器上线**：一位成员在 [GitHub](https://github.com/pwilkin/mcp-searxng-public) 上创建了一个“公共 **SearXNG**” **MCP** 服务器，以解决公共 **SearXNG** 服务器不可靠且缺乏 JSON 支持的问题。
   - 他们对调用的实例进行了随机化处理，以避免对主机造成 **DoS** 攻击，使其适合想要少量调用互联网搜索的私人用户。
- **AI 友好型 Data API 亮相**：一位成员在 [dapi-sandbox.adiom.io](https://dapi-sandbox.adiom.io/) 发布了一个对 Serverless 友好的 **Data API**，它可以立即创建一个带有安全语义端点的数据库后端。
   - 它支持在 **MongoDB** 和 **PostgreSQL** 之上的 **MCP**、**gRPC** 和 **Connect RPC**，并提供一个限制为 **50** 个活跃用户的免费沙箱。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1374114730658107502)** (12 messages🔥): 

> `AMD enum changes, 7900XTX vs 9070XT flash attention, RDNA4 wmma instructions, BERT training bounty, tinygrad gemm optimization` 


- **对 AMD Enum 更改的恼火**：一名成员对 **AMD** 更改 enum 的决定表示沮丧，并在他们的仓库中进行了投诉。
   - 他们开玩笑说 AMD 的表现*就像他们快要把数字用完了一样*。
- **Flash Attention 悬赏：最初仅限 7900XTX**：flash attention 悬赏已锁定，目前正在 **7900XTX** 和 **9070XT** 上进行测试，但目前仅支持 **7900XTX**。
   - 开发者表示，如果 **RDNA4** 增加了新的 wmma 指令，他们可以将其包含在内，但他们目前没有 **9070XT** 用于测试。
- **BERT 训练悬赏针对 Nvidia/AMD**：BERT 训练悬赏旨在寻求在 tinygrad 中性能超越普通 attention 的 flash attention，该悬赏兼容任何可以运行训练器的 **Nvidia/AMD** 硬件。
   - 悬赏发布者正在使用 chatgpt 编写他们的 AGENTS.md 文件，但效果很糟糕。
- **对 tinygrad GEMM 优化的兴趣**：一名成员询问了关于 **GEMM** 优化的进展，引用了 George Hotz 过去关于利用 **AMD GPU** 上的 tiles 的讨论。
   - 他们询问“我可以联系谁”来贡献代码。
- **通过悬赏申请 tinygrad 工作**：在回应一份求职咨询时，一名成员澄清说，获得 tinygrad 工作的首要途径是通过 **bounties**（悬赏），并建议从小的 PR 开始。
   - 他们指出了 [tinygrad 在 X 上的置顶帖子](https://x.com/__tinygrad__/status/1910185557642367390) 以获取更多信息。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1374339542810759270)** (7 messages): 

> `tinygrad control flow, jax.lax.cond equivalent, Tensor.where` 


- **tinygrad 控制流受到询问**：一名成员询问 tinygrad 中是否存在类似于 `jax.lax` 中的控制流操作符，例如 [jax.lax.cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)。
   - 该成员指出，*这种控制流对于许多 **Monte Carlo 算法** 至关重要*。
- **建议使用 `Tensor.where` 作为替代方案**：另一名成员建议使用 [`Tensor.where`](https://tinygrad.org/tensor.html#tinygrad.tensor.Tensor.where) 作为 `jax.lax.cond` 的可能替代方案。
   - 原提问者回答说，`jax.lax.cond` 允许你决定执行哪一段代码分支，而 `Tensor.where` 是专门应用于 tensor 的，因此两条路径仍然都会执行。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1374439330067775589)** (2 messages): 

> `Financial Analysis Workshop, Multi-Agent Communication Protocol (MCP), AWS joins MCP steering committee` 


- **LlamaIndex 举办金融分析实战研讨会**：LlamaIndex 将于 5 月 29 日在纽约举办一场实战研讨会，由 @jerryjliu0 主持，主题是构建用于**金融分析和尽职调查**的 Agent 工作流，点击[此处](https://t.co/geDdBoe9aL)报名。
   - 该活动将提供关于利用 LlamaIndex 进行高级金融应用的*独家*见解。
- **AWS 致力于多智能体通信协议 (MCP)**：**AWS** 宣布加入 **MCP 指导委员会**，并致力于推动其演进以实现更好的 Agent 间通信，并与 **LlamaIndex** 等框架进行协作。
   - 有关 **MCP** 和 **AWS** 参与的更多详细信息可以在[此处](https://t.co/6HPKJASKPi)找到。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1374401988464349335)** (11 messages🔥): 

> `Agent Handoff Examples, Llama Parse Service Issues, VectorStoreIndex vs Local FAISS` 


- **Agent 移交手册**：一名成员询问 Agent 移交的示例，另一名成员提供了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/)的链接。
- **Llama Parse 堆积？**：一名成员报告了使用 *Parse with Layout Agent* 时 **Llama Parse 服务** 出现的问题，指出任务耗时超过 **30 分钟**，然后在没有解释的情况下失败，并卡在尝试加载的过程中。
   - 他们还分享了一张失败任务的[截图](https://cdn.discordapp.com/attachments/1374484064811221172/1374484518286786763/job-failed.png?ex=682e37f5&is=682ce675&hm=7bbafea016edc727275f4de63a70d2626fe12d38ac7e8d3301ec3953fbc92559&)。
- **FAISS 对决**：一名成员询问在 **RAG** 模型中使用 **VectorStoreIndex** 和本地 **FAISS** 进行存储的性能差异。
   - 他们质疑使用其中一种是否会降低 RAG 模型的性能。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1374391886126518402)** (2 messages): 

> `Recipe Tutorials, Automated CI, Llama2 Evaluation` 


- **Recipe Tutorials 作为文档补充**：成员们一致认为，Recipe Tutorials 应该作为文档的补充，展示端到端的示例，例如在数学数据集上有所改进的 **grpo recipes**。
   - 他们建议通过链接这些教程来展示在特定数据集上的改进。
- **避免为 Recipe Tutorials 自动化 CI**：大家一致认为，将 Recipe Tutorials 作为 **CI** 的一部分进行自动化并非最佳选择。
   - 一位成员指出，创建通过条件为 *delta_in_performance > 0* 的测试可能会导致问题，因为每个新模型都需要重新考虑合适的训练和评估数据集，这感觉涉及了过多的 ML 工作。
- **Llama2 评估非常痛苦**：一位成员提到，他们在早期尝试过 **Llama2** 的方案 **(2)** 版本，过程非常痛苦。
   - 他们建议更新贡献指南（contributing guide）以包含最佳实践，并为每个新模型添加包含不同评估摘要的 **README.md** 文件，以帮助他人进行合理性检查（sanity check）。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1374513667890085952)** (2 messages): 

> `DistCp, Safetensors, Async Checkpointing` 


- **DistCp 到 Safetensors 转换：请求创建 Issue**：有人请求创建一个关于将 **DistCp** 格式（由 async checkpointing 生成）转换为 **safetensors** 的 Issue。
   - 一位成员做出了回应，鼓励创建此类 Issue，并提供 utils 以促进转换，这将为 **DCP team** 提供宝贵的信号。
- **用户请求 DistCp 到 Safetensors 转换，成员提供 utils 支持**：一位用户询问如何将 **DistCp** 格式（来自 async checkpointing）转换为 **safetensors**。
   - 作为回应，一位成员建议创建一个 Issue 进行跟踪，并提供 utilities 辅助转换，这表明了其对 **DCP team** 的重要性。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1374439490546040943)** (3 messages): 

> `async_grpo, async_rl, vllm dependencies, torch version compatibility` 


- **实验 Async GPRO**：一位成员开始实验 **async_grpo**，并注意到 **async_rl** 目前依赖于 **vllm==0.8.4**，而该版本又依赖于 **torch==2.6.0**。
- **将 vLLM 依赖升级至 0.9.0**：一位成员计划将 **vllm** 依赖更新到预发布版本 **vllm==0.9.0**，这需要 **torch==2.7.0**。
   - 该成员询问了在此设置下运行 **async_rl** 可能出现的问题，得到的答复是 *目前尚未测试，但应该可以工作*。
- **Async RL Recipe 仍处于实验阶段**：**async RL recipe** 目前仍处于非常实验性的阶段，因此它被固定在了一个稳定的 **vllm** 版本上。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1374442432393121972)** (3 messages): 

> `DSPy X post, DSPy is all about, Getting what DSPy is all about?` 


- **DSPy 神秘的 X 帖子**：一则 [X 上的帖子](https://x.com/dspyoss/status/1924838188289556966) 展示了一些“丑陋”的东西，但发布者却表达了对它的喜爱。
   - 作者调侃道：*如果你理解了这个，你就理解了 DSPy 的核心*，暗示了对项目本质的更深层次理解。
- **解读 DSPy 之谜**：该 X 帖子的神秘信息暗示，**理解 DSPy** 涉及到欣赏那些非传统或最初看起来不吸引人的方面。
   - 它暗示了 DSPy 开发和应用背后可能存在的 **独特方法** 或哲学。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1374126490228232482)** (1 messages): 

> `AgentX Competition, Submission Forms, Judging Panel, Entrepreneurship Track, Research Track` 


- ****AgentX** 竞赛提交表单已开放！**：**AgentX** 竞赛的提交表单现已开放，评审团由来自顶级风投和 AI 公司的专家组成；提交截止日期为 **2025 年 5 月 31 日晚上 11:59 (PT)**。
   - [创业赛道 (Entrepreneurship Track)](https://forms.gle/FJTC4jd197bNeJJ96) 和 [研究赛道 (Research Track)](https://forms.gle/5dccciawydCZ8o4A8) 的链接现已可用。
- ****AgentX** 提交要求详情**：创业赛道需要项目演示文稿 (pitch deck)、产品演示视频和在线产品链接；而研究赛道则需要论文、视频演示和 **GitHub** 仓库。
   - 优胜团队将获得总额超过 **15 万美元** 的奖金。
- **在社交媒体上协助推广 **AgentX****：竞赛组织者请求协助在社交媒体平台上宣传 **AgentX**。
   - 他们提供了 [X 链接](https://x.com/dawnsongtweets/status/1924470174776004875) 和 [LinkedIn 链接](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii)。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1374107112698941440)** (2 messages): 

> `OpenAI API Keys, Trailblazer Tier, Mastery Tiers` 


- **学生需提供 OpenAI API Key**：学生在实验中必须使用自己的 **OpenAI API Key**，但在实际提交时可以将其排除。
   - 助教 <@181105076423753728> 可以回答是否存在不需要 API 调用 (API calls) 的替代方案。
- **为获得证书而降低 Mastery 级别**：即使学生在实验 (labs) 方面遇到困难，仍可申请 **Mastery 级别**，因为只要完成测验和文章，就可以被“降级”到 **Trailblazer 级别**。
   - 降级过程由工作人员端处理。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1374382527661342791)** (2 messages): 

> `PDF text extraction, GPT4All OpenAI API Key installation` 


- **PDF 文本提取需要特殊的 embedder 模型**：为了从 **PDF 教科书** 中提取部分内容的精确副本，需要一个特殊的 embedder 模型。
- **GPT4All API Key 安装故障排除**：一名成员报告在粘贴较长的 **OpenAI API Key** 时，**GPT4All** 中的安装按钮无法正常工作。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1374149004580884631)** (1 messages): 

> `DataTune, Data Transformation, Open Source Tool, Natural Language Instructions, LLMs` 


- **Vitalops 发布 DataTune，助力便捷数据转换**：Vitalops 推出了 [DataTune](https://github.com/vitalops/datatune)，这是一款专为**数据转换**设计的全新**开源工具**。
   - 它结合了**自然语言指令**与 **LLM**，旨在简化数据处理流程。
- **DataTune 通过 LLM 简化数据转换**：[DataTune](https://github.com/vitalops/datatune) 利用由 **LLM** 驱动的直观**自然语言指令**来促进**数据转换**。
   - 这款来自 Vitalops 的**开源工具**旨在为用户简化复杂的数据处理操作。


  

---


---


---