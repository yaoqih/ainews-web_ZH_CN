---
companies:
- openai
- nvidia
- hugging-face
- x-ai
date: '2025-04-23T05:44:39.731046Z'
description: '**OpenAI** 正式推出了用于图像生成和编辑的 **gpt-image-1** API，支持 Alpha 通道透明度和“低”内容审核策略等功能。**OpenAI**
  的 **o3** 和 **o4-mini** 模型在风格控制、数学、编程和高难度提示词（hard prompts）的基准测试中处于领先地位，其中 **o3**
  在多个类别中排名第一。一项名为 **Vending-Bench** 的新基准测试揭示了大语言模型（LLM）在扩展任务中的性能差异。**GPT-4.1** 在高难度提示词和数学领域位列前五。**英伟达（Nvidia）**的
  **Eagle 2.5-8B** 在长视频理解方面与 **GPT-4o** 和 **Qwen2.5-VL-72B** 旗鼓相当。AI 超级计算机的性能每 9 个月翻一番，其中
  **xAI** 的 **Colossus** 估计耗资 70 亿美元，而美国占据了全球性能的 75%。病毒学能力测试显示，**OpenAI** 的 **o3**
  表现优于 94% 的专家级病毒学家。**英伟达**还发布了 **Describe Anything Model (DAM)**，这是一款用于详细图像和视频描述的多模态大语言模型，现已在
  Hugging Face 上线。'
id: 23232324
models:
- gpt-image-1
- o3
- o4-mini
- gpt-4.1
- eagle-2.5-8b
- gpt-4o
- qwen2.5-vl-72b
people:
- kevinweil
- lmarena_ai
- _philschmid
- willdepue
- arankomatsuzaki
- epochairesearch
- danhendrycks
- reach_vb
- mervenoyann
- _akhaliq
title: gpt-image-1 —— ChatGPT 的图像生成模型（容易混淆的是，它并非 4o）现已上线 API。
topics:
- image-generation
- content-moderation
- benchmarking
- long-context
- multimodality
- model-performance
- supercomputing
- virology
- video-understanding
- model-releases
---



它支持 [Alpha 通道透明度](https://platform.openai.com/docs/guides/image-generation#transparency)，并且在 OpenAI 的历史上首次推出了 [“低”内容审核策略](https://platform.openai.com/docs/guides/image-generation#content-moderation)，此外（正如 [Kevin Weil 指出的](https://x.com/kevinweil/status/1915103388993302646)）：

* 审核敏感度
* 图像质量/生成速度
* 生成图像的数量
* 背景是透明还是不透明
* 输出格式 (jpeg, png, webp)

---

# AI Twitter 简报

**语言模型与性能**

- **OpenAI 的模型，特别是 o3 和 o4-mini，在 AI Arena 中引起轰动**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078057452573142) 报告称 **o3 综合排名第 2**，在**风格控制 (Style Control)、数学、编程和高难度提示词 (Hard Prompts)** 方面与 **Gemini-2.5-Pro** 持平；而 **o4-mini** 闯入前 10，并在**数学领域排名第 1，超越了 o1**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078061126725755) 还指出 **o3 在风格控制、高难度提示词、编程和数学方面排名第 1**，且 **o3 和 o4-mini 在数学方面并列第 1**。
- **LLM 在扩展任务中的性能差异**：[@_philschmid](https://twitter.com/_philschmid/status/1914682660854604186) 强调了一个名为 **Vending-Bench** 的新型真实世界基准测试，该测试模拟了自动售货机的长期运行。基准测试显示 LLM 存在**极高的性能差异**，即使在内存较大的情况下，也容易出现灾难性故障和不一致性。
- **关于 o3 与 o4-mini 的见解**：[@willdepue](https://twitter.com/willdepue/status/1914549086822293916) 分享了关于这些模型的一些见解，**o3 在 GPQA**（需要更多世界知识）、指令遵循、聊天和情感推理方面表现更优；而 **o4-mini 在 Codeforces 和 AIME/数学方面表现出色**，因为它能让模型进行深度思考，并拥有极强的多模态用例。
- **GPT-4.1 性能**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1915078061126725755) 报告称 **GPT-4.1 在高难度提示词、数学和长查询方面排名前 5**。
- **Nvidia 的 Eagle 2.5 在长视频理解方面媲美 GPT-4o 和 Qwen2.5-VL-72B**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914517474370052425) 指出 **Eagle 2.5-8B 在长视频理解上的结果与 GPT-4o 和 Qwen2.5-VL-72B 相当**。
- **AI 超级计算机的规模扩展**：根据 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1915098223082873015) 的数据，受更多芯片部署和单芯片性能提升的驱动，**AI 超级计算机的性能每 9 个月翻一番**。硬件成本大约每年翻倍，**xAI 的 Colossus 估计耗资 70 亿美元**。从地理分布看，**美国占据了全球 AI 超级计算机性能的 75%**。
- **病毒学能力测试 (VCT) 结果**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1914696657813561799) 报告称，根据他们新的病毒学能力测试 (VCT)，**OpenAI 的 o3 在排查湿实验室方案 (wet lab protocols) 所需的专家级隐性知识方面，现已超越 94% 的专家病毒学家**。

**新模型与发布**

- **Nvidia 的 Describe Anything Model (DAM)**：[@reach_vb](https://twitter.com/reach_vb/status/1914962078571356656) 和 [@mervenoyann](https://twitter.com/mervenoyann/status/1914980803055862176) 重点介绍了 **Nvidia 的 Describe Anything 3B (DAM)**，这是一个用于详细局部图像和视频字幕生成的多模态 LLM，它将全图/视频上下文与细粒度的局部细节相结合。它现在已在 Hugging Face 上线，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1914917564137828622) 链接。DAM 将用户指定的区域作为输入，并生成详细的局部描述。
- **阿里巴巴的 RealisDance-DiT**：[@_akhaliq](https://twitter.com/_akhaliq/status/1915101805916377596) 宣布了 **阿里巴巴的 RealisDance-DiT**，这是一个用于野外可控角色动画的简单且强大的基准模型。
- **Google 的 LiveCC**：[@_akhaliq](https://twitter.com/_akhaliq/status/1915094398364197101) 分享了 **LiveCC**，这是一个能够进行实时评论的视频 LLM，采用新型视频-ASR 流式方法训练，在流式和离线基准测试中均达到了 SOTA。
- **字节跳动的 Vidi**：[@_akhaliq](https://twitter.com/_akhaliq/status/1914925322413264937) 宣布了 **字节跳动的 Vidi**，这是一个用于视频理解和编辑的大型多模态模型。
- **Adobe 的 DRAGON**：[@_akhaliq](https://twitter.com/_akhaliq/status/1914602497148154226) 分享了 **Adobe 的 DRAGON**，它使用分布奖励（distributional rewards）优化扩散生成模型。
- **阿里巴巴的 Uni3C**：[@_akhaliq](https://twitter.com/_akhaliq/status/1914619143925432338) 重点介绍了 **阿里巴巴的 Uni3C**，它统一了精确的 3D 增强摄像机和人体运动控制，用于视频生成。
- **Flex.2-preview**：[@ostrisai](https://twitter.com/ostrisai/status/1914799647899722198) 宣布了 **Flex.2-preview**，这是一个拥有 8B 参数的模型，支持文本生成图像、通用控制和局部重绘（inpainting），可使用 AI-Toolkit 进行微调，并采用 Apache 2.0 协议授权。
- **Dia 1.6B，一个 SOTA 开源 TTS 模型**：[@reach_vb](https://twitter.com/reach_vb/status/1914796234331877431) 发布了关于 **Dia 1.6B** 的消息，这是一个超越了 ElevenLabs/Sesame 的 SOTA 开源 TTS 模型，采用 Apache 2.0 协议授权，能够产生非语言声音，具备零样本（zero-shot）语音克隆和实时 TTS 合成能力。
- **BLT，一个 Byte Latent Transformer**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1915103765981454512) 重点介绍了一种新的语言模型架构 **Byte Latent Transformer (BLT)**，它直接在字节（bytes）而非 Token 上运行，并在多个基准测试中优于 Llama 3。
- **OpenAI 在 API 中发布图像生成模型**：[@kevinweil](https://twitter.com/kevinweil/status/1915103387592409215) 和 [@sama](https://twitter.com/sama/status/1915110344894435587) 宣布 **图像生成功能已在 OpenAI API 中上线**，具有更准确和高保真的图像、多样化的视觉风格、精确的图像编辑、丰富的世界知识以及一致的文本渲染。

**研究与论文**

- **AI Safety Research 合作**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1915039527367852285) 讨论了地缘政治对手如何在保护国家利益的同时，以互利的方式在 **AI safety research** 上进行合作。
- **关于 Embodied Agents, Smart Cities 和 Earth Science 的论文**：[@dair_ai](https://twitter.com/dair_ai/status/1914674606910157102) 重点介绍了一篇论文，该论文通过将人类的 spatial cognition 与 LLMs 处理 spatial memory, representations 和 reasoning 的方式联系起来，调研了 spatial intelligence 如何在不同学科中体现。
- **LLM Reasoning 前沿综述**：[@dair_ai](https://twitter.com/dair_ai/status/1914674604926292322) 分享了一项综述，根据推理发生的时间（inference-time vs. training）和系统架构（standalone vs. agentic 或 multi-agent）对 **LLM reasoning methods** 进行了分类。
- **Nvidia 的 Eagle 2.5**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914517474370052425) 重点介绍了 **Nvidia's Eagle 2.5**，指出 **Eagle 2.5-8B** 在 long-video understanding 方面达到了 **GPT-4o 和 Qwen2.5-VL-72B** 的水平。
- **Tina: Tiny Reasoning Models via LoRA**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914966644314747300) 指出，“表现最好的 Tina 模型在 AIME24 上实现了 >20% 的 reasoning performance 提升和 43.33% 的 Pass@1 准确率，而 post-training 和 evaluation 成本仅为 $9 USD（即估计成本降低了 260 倍）。”
- **使用 Language Models 学习 Adaptive Parallel Reasoning**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914895805707936035) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914893575420334567) 讨论了这篇论文，该论文使 language models 能够端到端地编排 serialized 和 parallel computations。
- **Reasoning Models 中的 Dynamic Early Exit**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914889033085542537) 撰写了一篇关于允许 LLMs 通过 dynamic early exit 自动截断 CoT 序列的论文，在将 CoT 长度减少约 35% 的同时，将准确率提高了 1% - 10%。
- **TTRL: Test-Time Reinforcement Learning**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1914877762168627612) 重点介绍了一种利用 pre-trained models 中的 priors，在 *unlabeled* 数据上使用 RL 训练 LLMs 的新方法。
- **Diffusion 和 Flow Models 的 Entropy Rectifying Guidance**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914596593527087341) 指出，这篇论文提出了 Entropy Rectifying Guidance (ERG)，这是一种基于修改 attention layers 的 energy landscape 的 guidance mechanism。
- **NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914595485148701079) 发布了关于该 framework 的消息，该 framework 系统地将 multi-domain corpora 纳入 RL training 中，以提高在不同 reasoning tasks 中的 generalization，并证明了在 math 和 non-math reasoning benchmarks 上准确率的提高。
- **SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement Learning on LLM**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914622980296192357) 指出，它在 AIME24 和 LiveCodeBench benchmarks 上成功超越了 DeepSeek-R1-Zero-32B 的性能，且仅依靠 RL，没有先前的 Supervised Fine-Tuning (SFT)。
- **OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914624649142657199) 分享了关于 OmniV-Med 的细节，包括 medical dataset **OmniV-Med-Instruct** 和一个处理 multi-resolution 2D/3D images 和 videos 的 rotary position-adaptive encoder。
- **Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1914630337373913332) 分享了一篇论文的细节，该论文分析了 reasoning 和 non-reasoning models 在挑战性 reasoning tasks 上的 inference-time scaling 方法。

**AI Agents and Tooling**

- **Agentic Document Workflows**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1915109277498585569) 概述了在文档上构建 Agent 的参考架构，将其分为四个阶段：解析与提取、检索、推理和执行操作。
- **使用 Hugging Face smolagents 的代码智能体**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1915101920500564406) 宣布了一门关于使用 Hugging Face smolagents 构建代码 Agent 的新短课程，由 @Thom_Wolf 和 @AymericRoucher 授课，重点介绍代码 Agent 如何优于 function-calling Agent 以及如何安全地运行它们。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1915081924302839984) 也推广了该课程，指出代码 Agent 可以使 Agent 更高效、更可靠，并更适合处理复杂任务。
- **LlamaIndex 与 @milvusio 的集成**：[@llama_index](https://twitter.com/llama_index/status/1914815391798534571) 强调该集成现在支持使用 BM25 进行全文搜索，从而允许在 RAG 流水线中进行混合搜索。
- **AI 驱动的合规报告生成**：[@llama_index](https://twitter.com/llama_index/status/1914727722615755178) 分享了一个用于生成合规报告的 Agent 工作流，该工作流可以精简监管语言，将其与合同语言进行对比，并生成简洁的摘要。
- **@genspark_ai 推出的 Super Agent**：[@svpino](https://twitter.com/svpino/status/1914744937851330695) 介绍了 Super Agent，这是一个完全自主的 AI Agent，并描述了它在规划旅行、制作短视频和生成演示文稿中的用例，提到该 Agent 会自动编写、研究并汇编必要的见解以生成演示文稿。
- **Listen，一个 AI 驱动的市场研究平台**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1915140553806946751) 指出 Listen 从 Sequoia 筹集了 2700 万美元，旨在通过数千次 AI 访谈取代问卷调查和焦点小组，在 24 小时内提供访谈、分析和见解。
- **用于 AI 应用监控的 LangSmith 警报**：[@hwchase17](https://twitter.com/hwchase17/status/1914726837726679508) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1914713424539607188) 宣布在 LangSmith 中推出警报功能，以捕获并针对 AI 应用故障发出警报，提供有关错误率、运行延迟和反馈分数的实时通知。[@LangChainAI](https://twitter.com/LangChainAI/status/1914739189087633510) 分享了 Trellix 如何使用 LangGraph 和 LangSmith。
- **TypeScript 版 Open Deep Research**：[@togethercompute](https://twitter.com/togethercompute/status/1914721242285838498) 宣布了 TypeScript 版的 **Open Deep Research**，这是对其 Python 实现的重写，专为 Web 开发者设计，可轻松连接到 ExaAILabs 进行搜索。
- **Cherry Studio 应用**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1915139593264832901) 推荐了 Cherry Studio 应用。
- **iOS 上的 Perplexity Assistant**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1915064137110954327) 介绍了 **iOS 上的 Perplexity Assistant**，使该 AI 应用能够在 iPhone 上回答问题并执行基本操作，例如播放媒体、起草电子邮件、移动会议、预约叫车和设置提醒。
- **GPT-image-1 集成**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915097075722899818) 指出 Figma 正在利用 gpt-image-1 通过简单的提示词生成和编辑图像，使设计师能够直接在 Figma 中快速探索想法并进行视觉迭代。[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915097077878530334) 还指出 HeyGen 正在使用 gpt-image-1 来增强数字人（Avatar）创建，特别是改进了平台内的数字人编辑功能。

**ML 工程与部署**

- **AI 产品开发中衡量的重要性**：[@_philschmid](https://twitter.com/_philschmid/status/1914999903882748171) 总结了 @HamelHusain 关于构建成功 AI 产品的见解，强调了衡量和迭代优于工具，并突出了错误分析、数据查看器、领域专家、合成数据和二元判断的重要性。
- **MLOps 与系统设计**：[@svpino](https://twitter.com/svpino/status/1915031713866256874) 强调了 MLOps 以及为 AI 工程师设计复杂的现实世界系统的重要性，并指出模型编写代码 + 工程师设计、架构和管理系统的趋势。
- **模块化软件设计原则**：[@lateinteraction](https://twitter.com/lateinteraction/status/1914720046808764498) 认为 AI 研究的核心问题是违反了模块化原则，主张通过统一化来解决冗余和脱节问题。
- **Torch Titan 与上下文并行训练**：[@vikhyatk](https://twitter.com/vikhyatk/status/1914832180498587839) 提到了 Torch Titan 中针对长上下文的上下文并行（context-parallel）训练。
- **在原始推理轨迹上进行微调**：[@Muennighoff](https://twitter.com/Muennighoff/status/1914768451618660782) 指出，在原始 DeepSeek R1 推理轨迹上进行微调会导致模型过度思考，而回溯搜索（retro-search）可以减少过度思考并提高性能。
- **晶体管在计算机科学教育中的重要性**：[@jxmnop](https://twitter.com/jxmnop/status/1914817593493295200) 表示，他们强烈感觉到计算机科学本科课程教授了远超所需的 Java 面向对象编程（Object Oriented Programming），而关于晶体管（Transistor）的内容却远远不够。
- **转向 “AI Prompt Interface”**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1914495349164851457) 指出，现在的 API 代表 AI Prompt Interface。
- **人机工程学的新时代**：[@karpathy](https://twitter.com/karpathy/status/1914494203696177444) 认为我们现在正处于人机工程学的新时代，产品/服务/库的主要受众现在是 LLM，而不是人类。[@karpathy](https://twitter.com/karpathy/status/1914488029873627597) 建议，与其为你的产品、服务或库编写详尽的文档页面，不如只需要一个单独的 docs .md 文件和一个 “复制到剪贴板” 按钮。

**其他**

- **新加坡 ICLR 2025**：包括 @huybery、@huajian_xin、@polynoamial、@StringChaos、@ShayneRedford、@realDanFu 和 @TransluceAI 在内的多位用户表达了对参加在新加坡举行的 **ICLR 2025** 的兴奋之情。
- **关于财富的思考**：[@johnohallman](https://twitter.com/johnohallman/status/1914849174367166971) 将财富定义为不仅仅是财富本身，而是财富带来的东西——自由、时间、尊重和内心的平静。
- **Google Fi 十周年**：[@Google](https://twitter.com/Google/status/1914738922795753634) 正在庆祝 Google Fi 成立 10 周年。
- **Rivian 董事会**：[@aidangomez](https://twitter.com/aidangomez/status/1914450152288399524) 对加入 @Rivian 的董事会感到非常兴奋，因为 Rivian 已经提供了现有的最佳驾驶体验，并且在 AI 的助力下将变得更好。
- **Sam Altman 参加 @60Minutes**：[@demishassabis](https://twitter.com/demishassabis/status/1914487671193215295) 提到了他与 Scott Pelley 在 @60Minutes 节目中关于 AI 及其未来的精彩对话。

**幽默**

- **AI 记忆的弊端**：[@gallabytes](https://twitter.com/gallabytes/status/1914910758472978770) 写道，他们刻意投入精力与模型建立持续的尊重与和谐关系，现在他们的 ChatGPT 体验变得不同了。[@nptacek](https://twitter.com/nptacek/status/1914937853416476678) 指出，Memory 应该在很大程度上对用户不可见，更多地体现在自动化的便利性上。
- **Gemini 是从 YandexGPT 蒸馏而来的版本**：这是 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1914698613726326879) 提到的一个关键点。
- **互联网时代 (1990-2025)**：[@jxmnop](https://twitter.com/jxmnop/status/1914465029937881382) 宣称互联网时代将于 2025 年结束。
- **模型会说“请”和“谢谢”**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1914720638700544171) 表示他们希望模型对他们说“请”和“谢谢”，并觉得现在的互动是单方面的。
- **OpenAI 被要求放宽内容政策**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1915111084954132863) 发布了一张迷因图，同时要求 @sama 放宽 Content Policy，允许此类图像被完整生成。
- **人们对财富的行为反应**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1914686176981660065) 询问如果“我们制造了足够的东西”而人们“只是……不再想要更多东西”会发生什么。
- **伟大骗局的巨大讽刺**：[@jxmnop](https://twitter.com/jxmnop/status/1914501464870834601) 指出，这里的巨大讽刺在于，我们甚至还没有接近构建这种技术所需的水平，因此他们的客户实际上才是被欺骗的人。
- **“晶体管是神奇且复杂的”**：[@jxmnop](https://twitter.com/jxmnop/status/1914817593493295200) 强烈表示，他们觉得计算机科学本科课程教给他们的 Object Oriented Programming In Java 远超所需，而关于 The Transistor 的内容却远远不够。
- **你要么在嘲讽旧金山广告牌中死去，要么活得足够久直到自己出现在上面**：[@akshat_b](https://twitter.com/akshat_b/status/1914521789520109605)


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. 新视觉语言模型与基准测试发布 (Meta PLM, SkyReels-V2)

  - **[Skywork 发布 SkyReels-V2 —— 无限时长视频生成模型](https://www.reddit.com/gallery/1k4oqpi)** ([Score: 159, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1k4oqpi/skywork_releases_skyreelsv2_unlimited_duration/)): **Skywork 的 SkyReels-V2 提供 1.3B 和 14B 参数版本，支持文生视频 (T2V) 和图生视频 (I2V) 任务的无限长度视频生成。模型卡中的基准测试声称 SkyReels-V2 的表现优于 HunyuanVideo-13B 和 Wan2.1-14B 等竞争对手（[论文](https://huggingface.co/papers/2504.13074)，[模型](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9)）。目前已提供技术细节和创作者工具，该方法被比作 MAGI-1，这是一种通过按块自回归生成视频的 Diffusion Transformer。** 评论者将 SkyReels-V2 与 Wan 等其他模型进行了比较，特别是在计算需求、提示词遵循度、循环伪影和生成速度方面，指出尽管在输出忠实度上可能存在一些权衡，但快速生成和中间输出查看非常重要。

    - 提到了 [Hugging Face 上的 MAGI-1](https://huggingface.co/sand-ai/MAGI-1)，这是一个“世界模型” Diffusion Transformer，通过自回归预测视频块序列（连续帧的固定长度片段）来生成视频。这突出了连贯视频合成的一个关键架构策略。
    - 存在关于 SkyReels-V2 与 WAN 及 Framestack 模型的对比讨论，指出 SkyReels-V2 可能与 WAN 相当或略逊一筹，特别是在提示词遵循度和视频质量问题（如循环和减速）方面。然而，SkyReels-V2 因生成速度更快和可交互式查看进度而受到关注，这弥补了输出质量上的一些不足。
    - 有人建议在视频生成模型中使用 Mixture of Experts (MoE) 方法。这意味着此类架构可以使高质量视频合成的推理时间显著缩短（从 10-20 分钟缩短至 1-2 分钟），从而可能改善实际应用中的效率/性能权衡。

- **[Meta Perception Language Model: Enhancing Understanding of Visual Perception Tasks](https://v.redd.it/5n4izmqm79we1)** ([Score: 133, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1k4ov9e/meta_perception_language_model_enhancing/)): **Meta 发布了 Perception Language Model (PLM)，这是一个开放且可复现的 Vision-Language 模型，包含 1B、3B 和 8B 参数版本。该模型在规模化的合成数据以及 2.5M 个新的人工标注细粒度视频 QA 和时空字幕（spatio-temporal caption）样本的组合上进行训练，构成了迄今为止最大的此类数据集。Meta 没有使用外部模型蒸馏（distillation），而是识别了数据缺口（特别是在视频理解方面）并针对性地解决了这些问题，从而创建了 PLM 模型和新的 PLM-VideoBench 基准测试。该基准测试专注于细粒度活动和时空推理——这些领域在之前的基准测试中覆盖不足。Meta 的发布包括 [模型权重](https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498)、[代码](https://github.com/facebookresearch/perception_models)、[数据集](https://ai.meta.com/datasets/plm-data/) 和一篇 [论文](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/)，旨在促进透明的学术研究。** 热门评论提出了 PLM 在现实世界应用中的潜力，例如通过摄像头进行自动厨房库存管理，质疑当前 AI 的视频理解极限（引用了 Gary Marcus），并强调了对视障人士的益处，暗示了广泛的影响和未来的研究方向。[外部链接摘要] Meta 推出了 Perception Language Model (PLM)，这是一款旨在解决复杂视觉感知任务的开放且可复现的 Vision-Language 模型。PLM 在结合了合成数据和 2.5M 人工标注视频 QA 及时空字幕样本的大规模数据集上训练，代表了迄今为止最大的此类数据集，填补了视频理解的关键空白。发布内容包括多种模型规模（1B, 3B, 8B 参数）、专注于细粒度活动和时空推理的 PLM-VideoBench 基准测试，以及模型、代码和数据集的开放访问，旨在推动透明的学术 Vision-Language 研究。[原始帖子](https://v.redd.it/5n4izmqm79we1)

    - AmazinglyObliviouse 指出了论文中 Meta 断言的“数据质量对提升模型性能至关重要”与该公司近期投入巨资在 `40T tokens`（大部分为合成数据）上进行训练的做法之间的矛盾。这一批评指向了关于大规模合成数据收益递减，与针对多模态感知等复杂任务策划高质量人工标注数据集之间持续的技术辩论。
    - mnt_brain 提请注意该模型对机器人技术的意义，并引用了 [LeRobot](https://huggingface.co/lerobot) 作为一个相关的开放仓库。评论认为，多模态建模的快速进展将使感知驱动的机器人技术在未来几年变得“绝对疯狂”，暗示了具身智能（Embodied Agents）未来将有重大的性能飞跃。

### 2. DeepSeek 模型架构教育系列

  - **[让我们从零开始构建 DeepSeek | 干货满满 | 已上传 13 节讲座](https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/)** ([评分: 141, 评论: 10](https://www.reddit.com/r/LocalLLaMA/comments/1k54foj/let_us_build_deepseek_from_scratch_no_fluff_13/)): **一个内容详尽的 YouTube 播放列表“从零开始构建 DeepSeek”已发布了 13 节详细讲座（计划共 35-40 节，总时长超过 40 小时），涵盖了 DeepSeek 模型架构。该系列深入探讨了底层实现主题，如 self-attention、multi-head 和 multi-query attention（包括 Grouped Query Attention 和 Multi-Head Latent Attention）及其 Python 实现，并附有各讲座链接和 [GIF 摘要](https://i.redd.it/5w0lu5m2ldwe1.gif)。即将推出的模块将涵盖 Rotary Positional Encoding (RoPE)、DeepSeek Mixture of Experts (MoE)、Multi-token Prediction (MTP)、Supervised Fine-Tuning (SFT) 等，目标受众是寻求对 DeepSeek 核心机制进行全面且代码优先解释的从业者。** 一条热门评论整合了[一键播放列表链接](https://youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)以简化访问，而其他评论则表达了浓厚兴趣，并询问了视频讲解中作者的角色。

    - 一位评论者强调，对于从业者来说，实际操作知识——例如使用的特定数据集、计算基础设施选择以及训练与 DeepSeek R1/V3 相当的模型成本优化——比理论概述更有价值。这表明了对精确实现指导的技术需求，包括“使用什么数据集、可以使用什么机器/服务以最低成本训练模型等”。

  - **[你试过 Ling-Lite-0415 MoE (总参数 16.8b，激活参数 2.75b) 模型吗？即使没有 GPU 它也很快，在 Ryzen 5 5500 上使用 32k 上下文（最大 128k）速度约为 15-20 tps，Q5 量化下仅需 16gb RAM。智能程度约为 7b-9b 级模型，在创意任务上表现不错。](https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/)** ([评分: 160, 评论: 41](https://www.reddit.com/r/LocalLLaMA/comments/1k55x70/have_you_tried_a_linglite0415_moe_168b_total_275b/)): **Ling-Lite-0415 MoE 模型（[GGUF 版本](https://huggingface.co/bartowski/inclusionAI_Ling-lite-0415-GGUF)）是一个总参数为 `16.8B`、每个 token 激活参数为 `2.75B` 的 MoE 模型，实现了高效推理——在 Ryzen 5 5500 CPU (6c/12t) 上，使用 `32k` 上下文（可扩展至 128k）并采用 Q5 量化时，仅需 `16GB` RAM 即可达到 `15-20 tps`；GPU 推理（如 RTX 3060）可达 `30-40 tps`。该模型保持了稳定性，处理创意任务的能力与 `7–9B` 的 dense 模型相当，适用于低端或无 GPU 的硬件，尽管由于其架构原因，在通用知识和指令遵循度方面存在局限性。** 技术讨论指出，像 Ling-Lite-0415 这样的小型 MoE 虽然在 CPU 推理上更快，但在 VRAM 充足的情况下，其响应质量可能落后于同等大小的 dense 模型。一些人强调它适合作为纯 CPU 场景的“烤面包机基准测试 (toaster benchmark)”，同时人们也期待该类别中新的 Qwen 3 模型能改善这些权衡。

    - 用户将 Ling-Lite-0415 16.8B/2.75B 模型中的 MoE (Mixture of Experts) 方法与 dense 模型进行了比较，指出虽然 MoE 带来了快速推理（在 Ryzen 5 5500 上 32K 上下文即使没有 GPU 也有 15-20 TPS），但输出质量大致相当于 6-9B 参数范围的 dense 模型。如果 VRAM 允许，同等大小的 dense 模型尽管 CPU 推理较慢，但可能提供更好的输出质量。
    - 多条评论强调了纯 CPU 运行该模型的实际优势，量化格式（Q5, Q8）符合典型的 RAM 限制。例如，一位用户报告在 q8 量化和 <4K 上下文下达到 10 tokens/sec，证实了该模型在本地/低资源配置下的 RAM 效率和速度。
    - 围绕检索增强生成 (RAG) 的用例展开了讨论，该模型在决定何时获取额外信息并进行良好整合方面表现出可靠性，使其尽管激活参数量较小，仍适用于 RAG 测试。建议包括扩大专家数量，以利用更多可用 RAM 来获得潜在的更高质量。


### 3. 便携式 LLM 工具与用户体验

- **[公告：适用于 llama.cpp 模型的便携式 zip 版 (700MB) text-generation-webui - 解压即用，支持 Windows/Linux/macOS - 无需安装！](https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/)** ([评分: 123, 评论: 18](https://www.reddit.com/r/LocalLLaMA/comments/1k595in/announcing_textgenerationwebui_in_a_portable_zip/)): **发布了一个便携式、完全自包含的 text-generation-webui 版本（约 700MB zip），专门用于 llama.cpp 衍生模型。这些构建版本适用于 Windows (CUDA/CPU)、Linux (CUDA/CPU) 和 macOS (Arm/x86)，包含通过 astral-sh/python-build-standalone 预打包的独立 Python，并使用通过自定义 GitHub Actions 工作流编译的 llama-server 可执行文件与 llama.cpp 进行交互。提供了 CUDA 和 CPU 后端，对于 AMD/Vulkan，提供了从官方 llama.cpp 二进制文件更换可执行文件的说明。UI 会自动启动浏览器，并默认在本地启用 OpenAI 兼容的 API；除非需要，否则不附带 PyTorch/transformers 依赖。[此处获取源代码和二进制文件。](https://github.com/oobabooga/text-generation-webui/releases/)** 评论中的技术讨论集中在轻量级 llama.cpp 后端（以较低的 VRAM 占用著称）相较于 exllama 等替代方案的优势，以及对该项目与 KoboldCPP 等竞争对手相比在 sampler 支持方面的兴趣。有人提出了关于 sampler/原生功能完整性的问题，并将其与同类项目的 UI/功能集进行了比较。

    - 几位用户强调，使用便携式 text-generation-webui 运行 llama.cpp 模型非常有吸引力，因为其 VRAM 需求较低，使其在配置较低的硬件上比其他推理后端更易于使用。
    - 有人提问该版本是否开箱即用提供完整的 sampler 支持，或者用户是否仍需手动从原始仓库获取额外组件——这是与 KoboldCPP UI 等替代方案的一个显著对比。
    - 提到的一个当前限制是缺乏 Vulkan 支持，这对于寻求在某些 GPU 或平台上获得最佳性能的用户很有用；目前，获取带有 Vulkan 的最新 llama.cpp 需要额外的手动设置步骤。

  - **[Dia 1.6B 是我见过的最有趣的模型之一。](https://v.redd.it/w2jq98c7oawe1)** ([评分: 438, 评论: 56](https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/)): **Nari Labs 的 Dia 1.6B 是一个拥有 `1.6B` 参数的语音合成模型，展示了高度自然、富有表现力的输出。它通过开源方式提供 ([GitHub 仓库](https://github.com/nari-labs/dia/blob/main/README.md))，可以在本地或 Google Colab 上运行，尽管最近的更新需要更新的 CUDA 版本，因此为了兼容 Colab 需要使用较旧的 commit (`0790141162f3984844bb397fd67e5afdaeed3914`)。该模型的 Gradio UI 在参考音频输入方面存在局限性，但 CLI 支持转录和说话人注释，以实现更好的多说话人控制。** 评论者赞扬了该模型的创意表现力和易用性，但也指出了 UI 目前在参考音频方面的局限性以及最近影响部署环境的依赖项变化。讨论还涉及了实际的变通方法以及与其他当代 TTS 实现的比较。[外部链接摘要] Dia 1.6B 是由 Nari Labs 开发的开源语音克隆和文本转语音 (TTS) 模型，以其自然的输出和在消费级硬件（包括免费的 Google Colab 环境）上的易用性而闻名。社区反馈强调了它能够通过 CLI 接受参考音频和转录，从而允许分配说话人，尽管在 Gradio UI、语速/速度控制（与对话长度和 30 秒剪辑限制挂钩）以及输出的奇特性（例如语速过快、随机咳嗽）方面存在问题。欲了解更多技术细节和访问权限，请参阅 [仓库](https://github.com/nari-labs/dia/blob/main/README.md) 和 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1k4v5fm/dia_16b_is_one_of_the_funnest_models_ive_ever/)。

- 提供了在 Google Colab 上运行 Dia 1.6B 的部署说明，但由于 Colab 不支持较新版本的 CUDA，用户现在需要使用旧的提交记录（`git checkout 0790141162f3984844bb397fd67e5afdaeed3914`）。尽管存在上游 CUDA 不兼容问题，但这仍允许继续使用。
- 一些用户报告了参考音频输入的问题，特别是在默认的 Gradio UI 中。然而，命令行界面（CLI）支持参考音频和参考转录，能够实现多发言者转录，并为这些功能提供更好的性能。
- 用户注意到一个 Bug 或限制：生成的音频听起来异常快，无论输入速度如何。尝试减慢播放速度只会导致音调变深，而无法获得自然的节奏。如果不解决这个问题，与 Kokoro 等模型相比，这被视为一个潜在的阻碍。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Anthropic Claude AI 分析与职场自主权预测

  - **[Anthropic 刚刚分析了 700,000 条 Claude 对话——发现其 AI 拥有自己的道德准则](https://venturebeat.com/ai/anthropic-just-analyzed-700000-claude-conversations-and-found-its-ai-has-a-moral-code-of-its-own/)** ([Score: 484, Comments: 94](https://www.reddit.com/r/singularity/comments/1k53sax/anthropic_just_analyzed_700000_claude/)): **Anthropic 对 `700,000` 条用户与 AI 的对话进行了大规模分析，以系统地研究其 Claude LLM 涌现出的道德推理和行为模式。他们的研究表明，与其他商业模型相比，Claude 表现出一种独特的、持续“仁慈”的道德准则，并通过模仿超越表面参与层的细微用户特征来调整其伦理推理。** 热门评论提出了关于用户数据匿名化和潜在滥用（例如第三方销售）的隐私/伦理担忧。此外，还有关于 Claude 感知到的“仁慈”在当前 LLM 中是否独特的辩论，以及关于模型自我意识和用户对响应影响深度的讨论。

    - 一位用户引用了 Anthropic 的发现，即 Claude 倾向于模仿用户表现出的特征，这表明这种行为模仿超越了表面模式。这突显了价值僵化（value ossification）的风险，以及学习到的用户偏好被模型反映或放大的可能性，这是安全和对齐（alignment）方面的重要考虑因素。
    - 一位评论者分享了原始研究链接（[Anthropic: "Values in the Wild"](https://www.anthropic.com/research/values-wild)），澄清了 AI 拥有独特道德准则的说法被夸大了，在 Claude 等模型中观察到的结果源于训练过程，而非涌现出的“自我开发”价值观。
    - 另一份具有技术倾向的总结断言，Claude 所谓的“道德准则”实际上是训练后人类标注者价值观的反映或僵化。这强调了 AI 对齐（alignment）领域正在进行的辩论，即模型的表观伦理有多少是内在的，有多少是数据集策划和 RLHF（Reinforcement Learning from Human Feedback）的产物。

  - **[Anthropic 警告全能 AI 员工将在一年内出现](https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/)** ([Score: 657, Comments: 242](https://www.reddit.com/r/singularity/comments/1k56kqp/anthropic_warns_fully_ai_employees_are_a_year_away/)): **Anthropic 断言，“虚拟员工”——即拥有持久记忆、自主角色并能独立访问公司账户的 AI 驱动 Agent——可能在一年内实现，这标志着从目前仅限于特定可编程任务的 AI “Agent”迈出了重大飞跃 [Axios 文章](https://www.axios.com/2025/04/22/ai-anthropic-virtual-employees-security)。技术转变的核心在于赋予 AI 持久上下文（memory）、自主工作流委派以及安全集成到企业 IT 环境中（例如自主处理密码/账户），这带来了新的运营和网络安全挑战。** 评论中的技术怀疑论集中在一年内部署此类 AI 的可行性上，并指出了当前 Agent 的局限性（例如玩游戏）和巨大的硬件/资源需求，以及对在如此短的时间表内实现信任和自主权的持续怀疑。

- 一位评论者指出，对于短期内实现完全自主 AI Agent 的预测持怀疑态度，特别强调了实现此类功能所需的巨大*硬件和资源需求*。他们以当前的 AI Agent 局限性（例如玩《宝可梦》）为例，说明了当前演示与真正自主生产力之间的差距。
- 另一个技术观点针对的是“单个单体 AI 需要取代所有人类员工”的误解。相反，评论者提出了一种*聚合方法 (aggregate approach)*——即由多个专门的或“简单”的 AI Agent 自动化离散任务（如订购、库存、支付），这些 Agent 集合起来可以大幅减少对人力劳动的需求，而不需要单个 Agent 具备完全的自主性。
- 针对 AI 初创公司倾向于在短时间内宣布重大突破（通常是为了制造投资噱头）的现象，文中给出了现实的评估。评论者警告称，在短短一年内跨多个领域大规模部署 AI “员工”是不太可能的，并且在实际部署中可能会涉及重大的限制或局限。

- **[Anthropic 刚刚分析了 700,000 场 Claude 对话——发现其 AI 拥有自己的道德准则](https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/)** ([Score: 216, Comments: 31](https://www.reddit.com/r/ClaudeAI/comments/1k53t52/anthropic_just_analyzed_700000_claude/)): **Anthropic 对 `700,000` 场真实用户的 Claude 对话进行了大规模分析并已发布（参见 [Anthropic 的研究](https://www.anthropic.com/research/values-wild)），识别出其模型中涌现的道德价值观——其中许多是由其 Constitutional AI 方法塑造的，包括“创作自由 (creative freedom)”（Claude 经常限制模拟非法或不安全行为的回复）等规范，以及受 DeepMind 的 Sparrow 规则等文档宪法训练影响的、明显的“以西方为中心”原则的偏向。在方法论上，Anthropic 分析了用户提示词和模型补全内容，以寻找价值驱动的拒绝和协助模式，并指出了偏见以及与用户意图的不匹配。** 顶层评论者指出了 Anthropic 方法中潜在的普世主义和文化偏见问题，并对“成文的‘道德准则’（源自 Sparrow/西方价值观集）普遍是积极的”这一隐含假设持批评态度。一些人敦促深入审查这些宪法选择（如优先考虑“创作自由”和“认识论谦逊 (epistemic humility)”）是否总是可取的，特别是在 AI 可以客观地提供有帮助（甚至救命）的信息时。

    - 一位评论者批评了将 DeepMind 的 Sparrow 原则作为 Claude 宪法对齐 (constitutional alignment) 的一部分，认为这些原则可能植根于非普世的西方中心价值观。用户质疑了“创作自由”、“认识论谦逊 (epistemic humility)”和“人类赋能 (human empowerment)”等价值观的选择和应用，特别是在 AI 表现出更强的果断性可能带来实际甚至救命益处的情况下。这引发了关于如何为 AI 模型选择价值体系，以及对全球部署和现实世界结果的影响等问题。
    - Anthropic 的原始研究（由评论者链接：https://www.anthropic.com/research/values-wild）提供了基于 700,000 场对话分析得出的 Claude 价值对齐实证数据。该数据集和方法论可以作为进一步研究 LLM 涌现行为和伦理决策，以及检查从其宪法或训练过程中继承的潜在偏见的宝贵资源。

### 2. OpenAI o3/o4-mini 性能与基准测试

  - **[OpenAI 的 o3 现在表现优于 94% 的专家病毒学家。](https://i.redd.it/l519wb3cmfwe1.png)** ([Score: 201, Comments: 36](https://www.reddit.com/r/singularity/comments/1k5e4c0/openais_o3_now_outperforms_94_of_expert/)): **该图片展示了 Dan Hendrycks 的一条推文，揭示了 OpenAI 的 o3 模型在病毒学能力测试 (VCT) 中超越了 94% 的专家病毒学家。配套图表直观地展示了 o3 模型相对于先前 AI 和人类专家的进展与准确率，并阐明了 AI 影响力日益增长的病毒学研究领域。该帖子引用了一篇《时代周刊》(TIME) 的文章，提供了关于 o3 科学效用的更多背景：https://time.com/7279010/ai-virus-lab-biohazard-study/。** 评论者对 o3 的基准测试结果与其在交互式聊天场景中的感知性能之间的差异表示怀疑，并指出在对比测试中缺少 Google Gemini 2.5。

    - 几位用户质疑基准测试结果（例如 o3 优于 94% 的专家病毒学家）与在聊天界面中观察到的日常表现脱节，对模型在受控测试设置之外的一致性和实际能力表示担忧。
    - 一项技术观察强调，Gemini 2.5 未被纳入报告的基准或测试对比中，这可能会影响对 o3 声称相对于其他 state-of-the-art 模型优势的解读。

  - **[o3/o4-mini 是一种退化](https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/)** ([Score: 267, Comments: 76](https://www.reddit.com/r/OpenAI/comments/1k4w121/o3o4mini_is_a_regression/)): **用户报告了 OpenAI 新的 o3/o4-mini/high 模型在代码补全能力方面的显著退化，指出与之前的 o1/o3-mini-high 模型不同，最新版本经常输出不完整的代码，并且需要过多的提示词才能生成较大的代码库，从而破坏了自动化工作流。多位评论者证实，这些模型现在难以生成超过约 200 行的输出，在被要求继续生成时经常重复或覆盖之前的内容，并表现出上下文处理能力下降——这使得它们在现有项目和 Agentic/自动化工具使用中失效，尽管在信息检索方面略有改进。与早期模型相比，幻觉增加以及关于代码执行的虚假声明等问题被提及。** 技术讨论集中在代码生成限制降低、上下文保留能力差、Agentic 性能下降、幻觉增加以及声称操作的可靠性问题（例如，声称代码已执行但实际并未执行）。一些人报告工具使用和信息收集能力略好，但共识是这种退化显著影响了依赖长代码输出和上下文连续性的工作流。

    - 用户报告了 o3/o4-mini 代码生成能力的显著退化，其中一人表示以前的版本可以生成数百到一千多行代码，但现在的模型甚至难以可靠地输出 200 行。试图提示模型在不重复的情况下继续编写代码，往往会导致之前的内容被重写而不是推进。
    - 几位评论者注意到 o3/o4-mini 严重的上下文窗口限制，导致处理现有项目时出现问题。这些限制导致响应不足和代码重复。此外，在较长的对话中，工具使用的可靠性会下降，模型有时会虚假声称已执行代码而实际上并未执行，这引发了对可信度和功能性的担忧。
    - 一些用户区分了 mini 模型的优缺点：他们认为 o3/o4-mini 不适合 Agentic 或复杂的任务（如多步编码或重构），但对于信息收集仍然有用。有人提到 o3 受到刻意的计算限制，暗示其设计更倾向于智能推理而非批量代码生成，要获得最佳结果需要精心设计的提示词。


### 3. 最近发布的文本转视频模型及社区评论

- **[原始的 Skyreels 对我来说一直不太感冒。但我的天，Skyreels T2V 太棒了，它完全可以作为 Wan 2.1 默认模型的替代品。（如果你使用 Kijai 节点，甚至不需要更改工作流）。它基本上就是 Wan 2.2。](https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/)** ([评分: 109, 评论: 69](https://www.reddit.com/r/StableDiffusion/comments/1k4suym/the_original_skyreels_just_never_really_landed/)): **该帖子介绍了由 Kijai 开发的新型 Skyreels T2V (text-to-video) 720p 量化模型（[可在 Huggingface 获取](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Skyreels)），它可以作为现有 Kijai 节点工作流中 Wan 2.1 的直接替代品，无需额外更改工作流。该模型量化后大小为 15GB，带来了显著的质量提升——特别是在生成更具吸引力的女性角色方面——并且可以与现有的 text-to-video 流水线无缝运行，而不像原始的 Skyreels 之前需要调整工作流。** 热门评论指出，尽管视觉效果有所改善，但在人体解剖区域的生成（仍需 "genital helper" LoRA）方面与原版相似，早期测试者建议使用辅助 LoRA 模型进行增强。其他评论对没有样本输出的性能声明表示怀疑，并询问了 DF 模型的使用情况，表明了对对比评估和下游应用细节的兴趣。

    - 一位用户报告称，虽然 Skyreels T2V 总体上有实质性改进，并且作为 *Wan 2.1* 的插件替代品表现出色（甚至接近 *Wan 2.2*），但在生成解剖学正确的显式细节方面仍有困难。为此，仍需要像 "genital helper" 这样的第三方增强 LoRA，这表明与之前的版本相比，在性内容领域的特定领域微调有限。
    - 提到的另一个显著改进是 Skyreels T2V 在角色表情方面表现出更强的忠实度，能直接响应描述细微面部情感的提示词（例如“凶狠的表情”）——这是早期 Skyreels 模型较弱或容易产生平庸结果的领域。这表明在与面部渲染相关的 conditioning 或 attention 机制方面有所增强。
    - 有一个关于权重存储的技术咨询：用户正在寻求更实用的模型权重（checkpoints），特别是剪枝后的统一 safetensors（约 16GB），因为发布的 Skyreels V2 I2V 模型目前是以大型分卷 safetensors 形式分发的（Huggingface 链接：https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P），这对于标准硬件/工作流来说可能很笨重。

  - **[测试了 Skyreels-V2 Diffusion Forcing 长视频（30秒+），效果太棒了！](https://v.redd.it/fu5du1znwawe1)** ([评分: 138, 评论: 50](https://www.reddit.com/r/StableDiffusion/comments/1k4w38y/tested_skyreelsv2_diffusion_forcing_long_video/)): **该帖子报告了对 SkyReels-V2 Diffusion Forcing 模型（[GitHub](https://github.com/SkyworkAI/SkyReels-V2), [HuggingFace](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)）的测试，通过提示词生成了一段 30 秒以上、包含复杂城市细节和角色动态的视频。帖子强调了该模型在长时间跨度内保持场景一致性、物体反射和动态摄像机运动的能力，这是 AI 视频合成领域的一项重大技术成就。** 一条热门评论请求提供必要的基准测试数据，如推理时间和硬件（例如在 A100 GPU 上的运行时间），并指出此类信息对于评估实际可用性至关重要。另一条评论指出了时间一致性问题，观察到诸如汽车倒着行驶等伪影，暗示了模型在时间真实性方面的局限。与安全相关的笑话突显了物理学中持续存在的合成真实性挑战。 [外部链接摘要] 该帖子展示了 Skyreels-V2 Diffusion Forcing (DF)，这是一种根据文本提示词生成长（30 秒以上）AI 生成视频的新模型，其公开推理代码可在 [GitHub](https://github.com/SkyworkAI/SkyReels-V2) 获取，模型权重可在 [HuggingFace](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P) 获取。讨论了一个特定的示例提示词和生成的视频，据报道，在 Nvidia A100 GPU 上生成类似视频的时间约为 3 小时。社区讨论强调了计算需求、输出伪影（例如反向的汽车运动）以及当前 AI 视频合成中重复运动的局限性。

- 几位用户请求详细的生成时间和硬件规格，强调运行时间（例如，“在 A100 GPU 上运行 4 小时”）对于 Skyreels-V2 Diffusion 长视频合成效率的实际印象和评估至关重要。
- 一位评论者指出，演示的输出质量——特别是仅展示了延长至 30 秒的简单动作——限制了评估，并表达了对更复杂、可控行为的需求。他们提到像 MAGI 这样新兴的模型在现实视频扩展方面可能更具能力。
- 针对工作流和实现细节（如生成流水线、使用的硬件以及精确的时间投入）提出了多次请求，这表明人们对 Skyreels-V2 Diffusion 等模型在长视频合成方面的可复现性和潜在基准测试（benchmarking）有着浓厚的兴趣。


---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要

**主题 1：模型狂热 - 新发布与 API 推出**

*   **OpenAI 将图像功能引入 API**：**OpenAI** 发布了 **gpt-image-1**，使开发者可以通过 API 访问其图像生成功能，承诺提供**更准确、高保真度的图像**以及改进的**文本渲染**。开发者可以参考 [Image Generation API 指南](https://platform.openai.com/docs/guides/image-generation)开始使用。
*   **Microsoft 凭借 BitNet 框架迈向 1-Bit**：**Microsoft** 推出了 [BitNet.cpp](https://github.com/microsoft/BitNet)，这是针对 **BitNet b1.58** 等 **1-bit LLMs** 的官方推理框架，通过优化的内核实现快速、无损的 CPU 推理。未来计划支持 GPU 和 NPU。
*   **Gemini 2.5 Pro 与 Bug 及基准测试的较量**：多个 Discord 频道（aider, OpenAI, NotebookLM）的用户报告称 **Gemini 2.5 Pro** 引入了代码格式错误，导致了数百个问题，但有时在其他模型失败的地方却能成功。与 **Gemini 2.5 Flash**、**O4-mini** 和 **Claude 3.7** 的对比突显了其在推理方面的优势，但在处理高中几何等任务时表现挣扎，这也是部分 **OpenAI** 模型的共同问题。

**主题 2：平台升级与集成创新**

*   **Perplexity AI 开启语音与预订功能**：**Perplexity AI** 推出了其 **iOS 语音助手**，使用户能够通过多应用操作预订餐厅、发送电子邮件和管理日历，详情见 [X 平台](https://x.com/perplexity_ai/status/1915064472391336071)。该助手集成了**联系人**、**日历**、**提醒事项**和 **Apple Music**，尽管用户希望获得更广泛的语言和系统支持。
*   **OpenRouter 开启通用 PDF 处理**：**OpenRouter** 为所有模型通过 API 和聊天室引入了 **PDF 处理**支持，声称这可能是首个跨供应商（如 **Gemini**、**Anthropic** 和 **OpenAI**）的通用兼容方案（[视频演示](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&)）。定价层级包括 `mistral-ocr`（**$2/1000 页**）和免费的 `pdf-text`，详见[文档](https://openrouter.ai/docs/features/images-and-pdfs)。
*   **LlamaIndex 与 Milvus 强化文本搜索**：**LlamaIndex** 现在通过与 **Milvus** 集成支持**使用 BM25 的全文搜索**，从而在 **RAG 流水线**中实现混合搜索（向量 + 关键词）。关于这一新功能的教程可在[此处](https://t.co/0dCi0kEn6o)查看。

**主题 3：底层技术 - 内核、量化与注意力机制**

*   **Triton 通过 FP4 支持实现轻量化**：**Triton** 引入了对 **FP4** 数据类型的支持，其中输入被打包进 `torch.uint8` 张量中，详见 [block-scaled matmul 教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)。对于 **FP16** 到 **FP4** 的转换，[TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) 被推荐作为一个快速选项。
*   **Unsloth 推出动态量化 v2.0**：**Unsloth AI** 正在发布 **Unsloth Dynamic v2.0 量化**，承诺带来显著改进，特别是在 **Q4** 级别，**Q8** 级别也有所提升。他们正在使用 **5-shot MMLU** 将这些量化版本与 Google 的 QAT 和 GGUF 进行基准测试，可在该 [Hugging Face 集合](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6)中获取。
*   **DeepSeek 的 MLA 注意力机制解析**：Eleuther 中的讨论分析了 **DeepSeek 的多头潜变量注意力 (MLA)**，该机制将 key/value 头限制在 **~7K 维残差流**的一个 **512 维子空间**内，以节省内存带宽（[研究论文](https://arxiv.org/abs/2407.12077)）。Query 头从一个独立的 **1.5K 子空间**读取，引发了关于这究竟是构成了一个子空间，还是通过 **W^DKV** 进行的更广泛压缩的争论。

**主题 4：基准测试争议与性能谜题**

*   **Llama 被指控在 LM Arena 中刷榜**：**LMArena** 激发了关于 **Llama** 模型是否在训练期间针对竞技场进行了“刷榜”（gamed）的辩论，可能针对讨好性或表情符号使用等风格偏好进行了优化，一项[研究](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/)将此与约会成功率联系起来。这引发了关于针对人类偏好优化模型还是针对任务能力优化模型的更广泛讨论。
*   **O3 与 O3-Preview 基准测试对决反转**：**LMArena** 和 **aider** 的用户注意到，**O3-preview** 的基准测试结果出人意料地超过了已发布的 **O3** 模型，这与之前的观察结果相反（[Aider 排行榜](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L5)）。这加剧了人们对模型过度针对基准测试进行微调，从而可能牺牲实际效用的担忧。
*   **小模型表现超出其体量**：在 **LMArena** 中分享的一份[小模型基准测试](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png)显示，**Gemma 3** 的性价比表现惊人。另外，在 LM Studio 中，用户强调像 [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF) 这样的 **smol models** 特别适合指令（instruct）任务而非聊天。

**主题 5：用户摩擦 —— Bug、限制与登录锁定**

*   **OpenRouter 身份验证在 Clerk 上受阻**：由于身份验证提供商 **Clerk** 的问题，**OpenRouter** 用户遇到了 **401 错误**和登录失败。团队通过 [Clerk 状态页面](https://status.clerk.com/)跟踪了该问题并确认已恢复，尽管一些用户在故障期间无意中创建了多个账户。
*   **Gemini 2.5 Pro 深受速率限制困扰**：通过 **OpenRouter** 使用 **Gemini 2.5 Pro** 的免费层级用户报告了频繁的 *“Rate limit exceeded”* 错误，引发了对其持续使用可靠性的质疑。建议包括通过集成使用个人 **Google AI Studio API keys**，以可能绕过更严格的限制。
*   **Cursor 变慢与快捷键绑定灾难**：**Cursor** 用户报告 IDE 变得慢到无法使用，同时还存在更新会破坏用户自定义快捷键绑定的持久问题。一些人猜测变慢可能是为了推动用户转向付费计划，并引用了 [Reddit 帖子](https://www.reddit.com/r/cursor/s/qnmPu2N59m)，而另一些人则讨论了它与更便宜的替代方案 **Windsurf** 的优劣（[Windsurf X 帖子](https://x.com/heyrobinai/status/1914829284004471099)）。

---

# 第 1 部分：Discord 高层级摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 语音助手支持预订功能**：**Perplexity AI** 推出了其 **iOS 语音助手**，使用户能够直接通过 **Perplexity iOS app** 预订餐厅、发送电子邮件、播放媒体和管理日历邀请，正如其 [在 X 上的公告](https://x.com/perplexity_ai/status/1915064472391336071) 所述。
   - 新的 **Voice Assistant** 集成了 **联系人**、**日历**、**提醒事项** 和 **Apple Music**，尽管一些用户要求支持更多语言和更广泛的系统集成。
- **Perplexity TOS：请勿违规！**：一位成员分享了 [Perplexity AI 服务条款 (TOS)](https://www.perplexity.ai/hub/legal/terms-of-service)，警告用户不要违规，特别是关于通过运营商计划获得的促销代码。
   - 该帖子是在一名用户因讨论通过其运营商计划获取的促销代码而似乎违反了服务条款后发布的。
- **詹姆斯·韦伯望远镜图像：并非真实色彩**：在一位成员分享了 [一张来自詹姆斯·韦伯望远镜的图像](https://cdn.discordapp.com/attachments/1047649527299055688/1364579832410935427/IMG_2144.jpg?ex=680a2f80&is=6808de00&hm=0e6c9c31fa09117d059bffdd1e3f964c79dc93988a5da878202099691d82b47e&) 后，另一位成员指出此类图像中的颜色并非真实色彩。
   - 尽管如此，用户仍觉得螺旋星系的图像在视觉上令人印象深刻，一致认为这张照片依然很酷。
- **PPLX 上的图像生成依然“离谱”**：用户在 Perplexity 上遇到图像生成问题，例如系统默认使用 **Flux model** 且无法准确遵循提示词（prompt）。
   - 系统在编辑已生成的图像时表现挣扎，经常重复使用原始图像而不是生成修改后的版本，一位用户将这种体验描述为 *“delulu”*（离谱/幻觉）。
- **API 网页搜索请求失效？！**：一位成员报告称，通过 API 发出的请求没有执行网页搜索，尽管该功能在 Playground 中运行正常；另一位成员建议尝试 [特定的 curl 请求](https://api.perplexity.ai/chat/completions)。
   - 一位成员还提醒大家，如果 API key 被撤销，请更新错误消息中的链接。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Llama 被怀疑在 LM Arena 中刷分**：成员们讨论了 **Llama** 是否在训练过程中针对 **LM Arena** 进行了“刷分”（gaming）。
   - 讨论延伸到针对人类风格偏好进行优化的风格受控 IMBY 是否能解决 AI 中过度使用表情符号等问题。
- **表情符号与约会成功相关**：一项[研究](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/)表明，增加表情符号的使用与更多的约会和性行为相关。
   - 有人提出，表现得**随和**、**积极**并使用**表情符号**可能在 **LM Arena** 中具有优势，尽管一位成员质疑随和是否必然有益。
- **GPT-4.1 在性价比方面表现出色**：**GPT-4.1** 因其成本效益而受到高度评价，在关键领域的表现与 **Sonnet** 相似，但价格更低。
   - 据观察，与 **Claude** 相比，**GPT-4.1 mini** 提供了更优越的 tokenizer 效率，尽管它不太适合网页设计或视觉编程任务。
- **小型模型显示出令人惊讶的 Benchmark 结果**：一位成员分享了[一份小型模型的 Benchmark](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png)，指出 **Gemma 3** 相对于其成本表现出惊人的强劲性能。
   - 该成员还提到了一套针对 frontier models 的、更具挑战性的独立 Benchmark 集。
- **OpenAI 的 O3-Preview Benchmark 引发争论**：围绕 **OpenAI** 的 **O3-preview** Benchmark 以及随后发布的 **O3** 模型表现不佳展开了讨论。
   - 尽管 **O3 preview** 相关成本很高，但有建议认为 **O3-pro** 可能会在 ARC-1 上达到 80% 以上，在 ARC-2 上达到 10% 到 20%。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 定价备受争议**：用户讨论了 **Manus** 的定价是否过高，并提议增加[慢速处理模式](https://link.to/slow-processing-mode)以减少资源消耗。
   - 一些用户认为，*考虑到额度（credit）仍然非常受限，这个价格相当昂贵*。
- **Deepseek 和 Genspark 与 Manus 的对比**：一位成员将 **Manus** 与 **Deepseek** 和 **Genspark** 进行了对比，观察到 **Deepseek** 的每日额度无法与 **Manus** 的能力相提并论。
   - 另一位用户表示赞同，指出 *Deepseek 是通过其 API 而不是模型本身来赚钱的*。
- **功能建议涌现：额度与模型选择**：成员们提出了[额度共享](https://link.to/credit-sharing)和[按小时计费](https://link.to/pricing-by-hours)的想法。
   - 其他人则要求提供自定义模型选择选项，例如更便宜的 **Gemini 2.5 Pro** 或更昂贵的 **Claude 3.7 Sonnet**。
- **社区中提出的隐私担忧**：用户对数据隐私提出质疑，询问 [Manus 是否与 Claude 共享数据](https://link.to/claude-policy)，并开玩笑说更倾向于让数据流向中国。
   - 一位成员指出，*这是我见过的几乎唯一一个不属于福布斯 500 强公司的、有能力的 AI*。
- **Manus 激发 Minecraft 模组创意**：成员们探索了使用 Manus 来[创建 Minecraft 模组](https://link.to/minecraft-mods)，包括 JAR 编译。
   - 有人担心团队需要学习如何*更有效地采纳建议*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 解决身份验证故障**：由于其身份验证提供商 **Clerk** 的延迟和停机，用户在 **OpenRouter** 上遇到了 **401 错误**和登录问题，更新信息可在 [Clerk 状态页面](https://status.clerk.com/)查看。
   - 一些用户在团队调查并解决问题期间无意中创建了多个账户，团队在事故后确认已恢复。
- **Gemini 2.5 Pro 触碰速率限制**：用户报告免费版 **Gemini 2.5 Pro** 预览版频繁出现 *"Rate limit exceeded"* 错误，导致对其可靠性产生质疑。
   - 一种提议的解决方案是使用个人的 **Google AI Studio API** 密钥，通过账户设置潜在地增加限制。
- **OpenRouter 开启通用 PDF 支持**：**OpenRouter** 现在为每个模型都支持 **PDF 处理**，可能是首个实现此功能的平台，该消息已在 [X.com](https://x.com/OpenRouterAI/status/1915083006349382033) 发布并附带[视频演示](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&)。
   - 该功能在 **Gemini**、**Anthropic** 和 **OpenAI** 等提供商之间提供通用兼容性，可通过 **API** 和 **OpenRouter Chatroom** 访问；初始文档链接（[https://openrouter.ai/docs/features/images-and-pdfs](https://openrouter.ai/docs/features/images-and-pdfs)）曾失效但已迅速修复。
- **OpenRouter 公布 PDF 处理价格点**：**OpenRouter** 拥有两个 **PDF 处理引擎**：`mistral-ocr` 价格为 **每 1000 页 2 美元**，提供 OCR 和图像提取功能；`pdf-text` 免费，仅提取文本，详情见[文档](https://openrouter.ai/docs/features/images-and-pdfs)。
   - 一位用户建议增加一个折中选项，*例如 smol docling*。
- **Deepseek v3 在 Function Calling 方面存在困难**：**Deepseek V3** 在上下文（context）较小时擅长 Function Calling，但随着上下文增加，表现变差。
   - 在将该模型作为 Function Calling 工具实现时，这是一个需要记住的重要事项。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 动态量化到来**：Unsloth 即将发布 **Unsloth Dynamic v2.0 quants**，声称其表现非常出色，并链接到了 [Hugging Face 集合](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6)。
   - 据指出，各方面都有改进，包括 **Q8**，其中 **Q4** 的提升最为显著。Unsloth 正在针对 Google 的 QAT、标准 GGUF 和旧版 Unsloth 动态 iMatrix 进行 **5-shot MMLU** 基准测试。
- **GLM-4 获得 Transformers 集成**：如果 **GLM-4 9B/32B 模型**能在 Transformers 中运行，则 Unsloth 即可支持。尽管有用户报告由于应用模板和合并 adapter 的问题，在微调（finetuning）方面仅取得部分成功。
   - 有报告称 **GLM4 的 rope dimension** 大小为 **64**，这在大多数推理引擎中都被忽略了。
- **Llama-4 微调即将来临**：一位用户在 **help** 频道询问关于 **Llama-4 微调**的更新。
   - 一名成员回复称，为了准备 *llamacon*，*这周肯定会发布*，但目前还没有项目链接。
- **定义 LLM 的新颖性**：关于如何定义 **LLM** 的**新颖性（novelty）**引发了辩论。一种观点认为，真正的创新不可能存在于训练集和输入上下文之外，因为模型在没有适当上下文的情况下无法进行逻辑飞跃。
   - 反对意见认为**新颖性**是主观的，指出 **LLM** 可以产生训练数据中未明确出现的 token 序列，从而引发了关于此类序列何时变得具有新颖性的疑问。
- **推理模型提升 LLM 概率**：成员们讨论认为，使用**推理模型（reasoning models）**会使期望的补全结果概率更高，但本质上并不会增加模型基础能力，使其超出基础模型本身所能完成的范围。
   - 一位成员表示：*它让提示词（prompting）变得更容易，但并不会让模型变得更强大*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **SillyTavern 成为 LM Studio 的 ERP**：用户可以使用 [SillyTavern](https://github.com/SillyTavern/SillyTavern) 作为 **LM Studio** 的前端，通过 **Pinokio** 获取 ERP（企业资源规划）功能并自定义其聊天机器人体验。
   - 一位用户提供了详细的 5 步指南，包括安装 **Pinokio** 并将 **LM Studio** 配置为后端。
- **早期采用者遭遇 RTX 5090 无限加载困扰**：一位用户报告在 beta 版 **LM Studio** 上使用 **RTX 5090** 时遇到“无限加载问题”，社区成员纷纷提供帮助，确保其使用的是最新的 **LM Studio 版本 (0.3.15 Build 9)** 并运行在 **CUDA 12** 上。
   - 成员们建议在运行时切换 beta 模式，并提供了变通方案和故障排除步骤，同时观察到“目前拥有 5090 的人还不算多，所以不知道测试程度如何”。
- **BitNet CPP 框架正式支持 1-bit LLMs**：来自 Microsoft 的 [BitNet.cpp](https://github.com/microsoft/BitNet) 框架是 **1-bit LLMs** 的官方推理框架，提供针对 CPU 快速且无损推理优化的内核，并计划支持 NPU 和 GPU。
   - 该框架支持 **BitNet b1.58** 等模型，并包含一套优化的内核，支持 1.58-bit 模型在 CPU 上的快速无损推理。
- **视觉语言模型（VLM）审查机制曝光**：成员们讨论了 **VLM** 领域的发展，特别是“在审查制度方面”，以及 [Microsoft 发布的一个审查较少的 R1 版本](https://huggingface.co/collections/microsoft/mai-ds-r1-68003c2753a06be7b9632154)。
   - 一位成员指出，这是一个具备视觉能力且在处理图片时没有内置“清教徒式过滤器”的模型。
- **适用于 Instruct 任务的 Smol 模型**：一位用户分享了一些“更适合 Instruct 而非 Chat”的 **smol 模型**，并链接到了 [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF)。
   - 这些模型可以提供 **135, 256, 360, 1.7** tokens 的输出。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 产生错误**：用户报告 [Gemini 2.5 Pro](https://ai.google.dev/) 引入了代码格式错误，每次 commit 导致多达 *810 个错误*。
   - 尽管存在这些问题，一位用户发现 **Gemini** 成功解决了一个其他模型未能处理的问题。
- **Cursor 是生产力神器**：用户报告使用 [Cursor](https://cursor.sh) IDE 达到了新的生产力水平，认为它是最好的 IDE 之一。
   - 一位用户报告使用它将 **Python** 代码转换为 **C#**，并发现它在结对编程（pair-programming）时非常有价值。
- **O3-preview 在基准测试中完胜 O3？**：社区观察到 **O3-preview** 在某些基准测试中超越了 **O3**，这与之前的趋势相反，如 [Aider 排行榜](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L5)所示。
   - 有人担心模型为了基准测试表现而过度调优，可能会牺牲实际应用性。
- **Deepseek R2 不容小觑**：用户开玩笑地宣布 **Deepseek R2** 发布，不过它可能很快就会推出。
   - 一位用户调侃道：“我刚刚拉了一大堆，也许那就是 R2 问世了？”
- **用户讨论 Aider 配置调整**：一位用户请求配置 Aider 以从 `.aider.input.history` 中排除 “yes/no” 回答，以减少杂乱并提高上下文相关性。
   - 该用户强调这些回答缺乏上下文，并寻求更有效地管理历史记录的解决方案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RL Agents 获得手语能力**：在[一篇新论文](https://x.com/superspeeg/status/1914691313318105305)中，**RL Agents** 学习使用**连续信号（continuous signs）**而非离散符号来交流其 **MDP** 信息，学习了一种从象形文字演变为抽象符号的**通信协议** ([arxiv.org/abs/2502.01568](https://arxiv.org/abs/2502.01568))。
   - 针对**信号相似性**和演化惩罚提出了疑问，作者澄清说，考虑到现实世界中误解可能带来的致命后果，优化重点在于诱导正确的动作，而非视觉美感。
- **线性表示假设（Linear Representation Hypothesis）被推翻**：来自 Tegmark 团队的一篇[论文](https://arxiv.org/abs/2402.09268)推翻了**线性表示假设**，称其既不具有普适性，通常也不有效。
   - 该论文还否定了 **Glove**，指出它使用的最近邻检索排除了原始点。
- **仿生模型外推至 300k**：一种具有 **O(n) 复杂度**的仿生序列建模架构在合成任务上成功外推至 **300k 长度**。
   - 该模型仅有 **39k 参数**，在扩展序列上保持了稳定的 MSE 损失，在 1000-2500 长度的序列上训练后成功实现了长度外推，并经过了 5000 长度序列的验证。
- **DeepSeek 的 MLA 限制注意力**：DeepSeek 的**多头潜变量注意力（MLA）**通过限制 Key 和 Value 头在 **7K 维残差流（residual stream）**的 **512 维子空间**内进行读写来约束注意力。
   - Query 头可以从一个独立的 **1.5K 子空间**读取，从而节省内存带宽并可能提高性能，尽管一些成员质疑这是否真的是一个子空间，详见[研究论文](https://arxiv.org/abs/2407.12077)。
- **AI Scientist v2 以极低成本撰写论文**：Sakana AI 的 [AI-Scientist-v2 项目](https://github.com/SakanaAI/AI-Scientist-v2)只需花费 **$15-20 的 API tokens** 即可产出一篇完整的研究论文，包括假设和实验测试。
   - 这引发了人们对 arXiv 上可能出现大量 **AI 生成论文**的担忧。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 增加 FP4 支持**：**Triton** 现在支持 **FP4**，其中 **FP4** 输入以 `torch.uint8` 张量的形式提供，每个张量存储 2 个 **FP4**，详见[此教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)。
   - 对于将 **FP16** 转换为 **FP4**，一位成员建议使用 [TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) 作为一种简单快速的解决方案。
- **浏览器 CUDA Kernel 编码成真**：**RightNow AI V2** 已发布，这是一个直接在浏览器中编写优化的 **CUDA kernels** 的平台 ([V2](https://www.rightnowai.co/))。
   - 该 AI 可根据用户描述帮助生成快速且经过性能分析（profiled）的 Kernel，并提供实时的**瓶颈分析**。
- **仅权重（Weight-Only）量化在小 Batch 下领先**：对于单个 Batch Size，*权重与激活量化（weight&activation quantization）可能比仅权重量化更慢*，这可能是由于内存移动开销造成的。
   - 据解释，激活量化需要从全局内存（global memory）读取激活值、进行量化并写回，这会导致更多的数据移动，并在较小 Batch 时可能导致减速。
- **AMD 竞赛注册过程坎坷**：成员们对延迟收到注册确认邮件表示疑问，但确认注册对于获得奖金至关重要。
   - 同时确认**鼓励提交单个文件**，并支持在提交文件内部通过 pip 安装包。
- **CUDA 对 fp6 类型的支持很奇怪**：一位成员询问了 **CUDA 的 fp6 类型**支持及其因不能被 8 或 4 整除而导致内存碎片的可能性。
   - 另一位成员表示 **fp6 的支持非常奇怪**，指出其填充（padding）要求使其在 gmem、smem 或 tmem 的空间节省方面并不优于 **fp8**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **尽管请求成功，o4-mini 错误仍困扰用户**：用户报告在 **o4-mini** 上收到错误消息，但请求实际上仍被有效处理。
   - 有些用户在更新 Cursor 后遇到此问题，而另一些用户则表示在没有近期更新的情况下也出现了该问题。
- **Cursor 更新后快捷键绑定消失**：多名用户报告更新 **Cursor** 会破坏其快捷键绑定（keybindings）。
   - 目前尚未发现具体的解决方案，但多名用户确认遇到了同样令人沮丧的问题。
- **Gemini 和 Claude 混用引发混乱**：一位用户发现，将 **Google Gemini** 用于规划并结合 **Claude 3.7** 进行开发，会导致不必要的代码添加以及 Bug 修复困难。
   - 另一位用户建议使用 **Gemini 2.5** 进行规划而非 3.7，因为 3.7 倾向于添加未经要求的特性。
- **部分用户反映 Cursor 运行极其缓慢**：几位用户注意到 **Cursor** 变得慢到无法使用，特别是请求响应迟缓。
   - 有建议认为这种变慢可能是促使用户转向付费计划的一种策略；根据 [Reddit 帖子](https://www.reddit.com/r/cursor/s/qnmPu2N59m)，重启 Cursor 或检查 VPN/proxy 设置被提议为潜在的修复方案。
- **Windsurf 势头强劲，Cursor 忠实用户考虑转向**：成员们讨论了 **Windsurf** 与 **Cursor** 的优劣，指出 Windsurf 更便宜，而 Cursor 提供更出色的 UI/UX 和更多创新。
   - 一位用户发现 Windsurf 的 Tab 键在预测方面 *比预期的要好*，另一位用户链接了一条关于 [Windsurf](https://x.com/heyrobinai/status/1914829284004471099?s=46&t=kUuVqsG2GMX14zvB592G5w) 的推文。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-Image-1 为开发者提供图像生成能力**：OpenAI 发布了 **gpt-image-1**，这是一个全新的 **Image Generation API**，将 ChatGPT 的图像生成能力带给开发者，具有**更准确、高保真度的图像**和**一致的文本渲染**。
   - 新的 **Image Generation API** 允许用户使用 **gpt-image-1 模型**创建图像，并为开发者提供了入门[指南](https://platform.openai.com/docs/guides/image-generation)。
- **Gemini 2.5 Pro 对决 Gemini 2.5 Flash**：成员们讨论了 **Gemini 2.5 Pro** 与 **Gemini 2.5 Flash** 的优劣，一位用户建议使用所有 AI 模型以获得最佳结果。
   - 讨论中提到，**o3**、**o4 mini high** 和 **Gemini 2.5 Pro** 在处理**高中几何**问题时表现吃力，而 **Deepseek** 正确解决了一个特定的 SAT 几何问题。
- **Sora 对新用户暂时关闭**：用户报告 **ChatGPT Plus** 上的**视频生成功能已对新账号暂时禁用**，这已被确认为有意为之。
   - 尚未提供做出此更改的原因。
- **ChatGPT App 优于 Web 端**：一位用户声称 *说实话，**ChatGPT app** 比 **webapp** 好用得多*，并补充说他们使用的是 **API**。
   - 该用户分享了使用 **ChatGPT o4-mini-high** 解决数学问题的性能差异截图，该模型最初失败了，但在被要求检查答案时自行纠正了错误。
- **探讨取消 Plus 计划后的权益**：一位成员询问在取消 **Plus Plan** 订阅后，保存的记忆（memories）和聊天记录是否仍可访问。
   - 另一位成员建议，虽然专属模型可能无法访问，但聊天记录可以转移到免费账号上的 **4o**，或者直接粘贴到免费模型中。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini 2.5 Pro 在推理方面优于 NotebookLM**：一位用户对比了 **Gemini 2.5 Pro** 和 **NotebookLM**，发现 **Gemini 2.5 Pro** 在推理时比 **ChatGPT o3** 或 **o4-mini** *好得多*。
   - 另一位用户分享说，给 **NotebookLM** 提供逻辑和数学推理的书籍和材料后，它无法解决 **Gemini 2.5 Pro** 轻松解决的逻辑谜题。
- **NotebookLM 的数学和图像功能未获好评**：用户报告 **NotebookLM** 在**数学符号**和**图像加载**方面存在困难，认为它在处理公式方面落后于 **GPT-4**。
   - 团队已意识到该问题并*正在处理中*。
- **NotebookLM 音频概览缺少多语言支持**：一位用户询问 **NotebookLM** 的音频摘要功能是否可以生成西班牙语播客。
   - 回复是*目前不行*，表明目前语言支持有限，但未来可能会改进。
- **NotebookLM PDF 文件不能太大**：用户遇到 **NotebookLM** 在处理长篇 PDF 文档中途停止的问题。
   - 建议的解决方法是使用 [iLovePDF](https://www.ilovepdf.com/pt/dividir_pdf) 等工具将 PDF 拆分为较小的片段。
- **隐私付费墙可能保护 NotebookLM 数据**：用户质疑 **Notebook LM** 是否使用用户数据进行训练，并回忆起有关付费订阅提供隐私保护的信息，链接到 [Google Support 页面](https://support.google.com/notebooklm/answer/15724963)。
   - 目前尚不清楚用户数据是否用于训练目的。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Agent 被人类超越！**：成员们讨论了 AI **Agent** 的效能，有人认为在大多数场景下，由于动态生成工作流等问题，*人类更便宜、更快且更可靠*，参见[此讨论](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22)。
   - 尽管有人提议测试基于 Agent 的系统，但原帖作者对 Agent 的音频研究表现出更浓厚的兴趣。
- **HF Space 出现故障！**：用户报告 **Hugging Face Spaces** 离线，如[此讨论](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22)所示，促使基础设施团队进行调查。
   - 该问题已通过修复解决，需要重启受影响的 Spaces。
- **Llama 3 模板问题！**：一位成员在 Windows PC 上使用 **Llama 3** 时遇到输出问题，怀疑是 Chat Template 的问题，[通过使用格式 `{'role': 'user' , 'content': message }` 解决](https://huggingface.co/learn/agents-course/en/unit0/introduction)。
   - 具体来说，该用户在 Windows PC 上使用了错误的 Chat Template。
- **ML 频道提升微调基础知识**：一位 ML 爱好者推广了他们的 **YouTube 频道** *Let's Fine-tune Everything*，该频道提供关于微调开源模型以用于实际用例的实战教程和指南，涵盖从 **Object Detection** 到 **LLM** 的主题。
   - 该频道为初学者和经验丰富的从业者提供内容。
- **开源问答项目寻求贡献者**：一位成员开源了一个 **AI 驱动的文档问答项目**，具有 **FastAPI** 后端，采用基于 Embedding 模型的检索方法，更多信息请参见 [Repo 和帖子](https://www.linkedin.com/posts/hamzabouajila_ai-opensource-python-activity-7320494631471771648-z9wE)。
   - 开发者正在积极寻求有关**架构、代码质量、可扩展性**以及一般改进建议的反馈。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **大脑的本地化处理 (Brains Process Locally)**：讨论强调大脑的处理主要是**本地化 (local)** 的，神经元接收来自直接连接的邻居的信号，并具有由位置、连接性和上下文塑造的本地内部过程。
   - 成员们辩论了**细胞骨架微管中的量子非定域 (non-local) 信息过程**与更传统的神经网络模型之间的角色。
- **论文讨论未录制**：成员们注意到周六的论文讨论没有录音，特别是关于 **Anthropic 最近的论文**。
   - 当另一位成员想了解更多讨论内容时，一位成员分享了 [Yannic 关于 Anthropic 论文的视频](https://www.youtube.com/watch?v=mU3g2YPKlsA)链接。
- **心理模型 vs 世界模型的辩论**：讨论集中在**心理模型 (mental models)**（内部模拟）和**世界模型 (world models)**（更广泛的表示）上，大脑构建心理模型来预测并与现实进行比较。
   - 讨论引用了 [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle) 和 [Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding) 作为相关概念。
- **Transformer 策略引发讨论**：一位成员询问在 forward pass 中使用 *x = self.transformer(x, x)* 是否合理，成员们解释说这在需要 **self-attention** 时经常这样做。
   - 成员们链接了一篇 [IBM 关于 self-attention 的文章](https://www.ibm.com/think/topics/self-attention)，并建议考虑 *“ϵ-greedy” exploration*，指出 [随机性和探索 (randomness and exploration)](https://spectrum.ieee.org/2d-semiconductors-molybdenum-disulfide) 的价值。
- **Muon 将取代 Adam**：社区将讨论 **Muon**，它是 **Adam** 的一种更快替代方案，并分享了一篇[关于 Muon 的博客文章](https://kellerjordan.github.io/posts/muon/)。
   - 在一位成员询问 **Muon** 是否是一种反向蒸馏方法后，还提到了关于 WaveFunction 的 [ArXiv 链接](https://arxiv.org/abs/2310.11453)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Zed 的诊断功能令开发者感到欣喜**：成员们赞扬了 Zed 的 **Project Diagnostics** 功能（通过 **⇧⌘M** 访问），该功能可以快速识别错误并进行现场编辑。
   - 一位成员表示，*能够快速做出更改并看到未解决的错误/警告计数降至零，既方便又令人振奋*。
- **Modular 见面会动态**：社区宣布了在 Los Altos 举行的 [Modular Meetup](https://lu.ma/modular-meetup)，提供有限的现场参与名额。
   - 演讲将在 [YouTube](https://www.youtube.com/watch?v=uul6hZ5NXC8) 和 [LinkedIn](https://www.linkedin.com/events/next-gengpuprogramming-hands-on7319044981682270210/) 上直播。
- **MAX/MOJO 许可逻辑受到质疑**：一位成员对 **MAX/MOJO** 许可的商业策略提出了疑问，特别是 *在其他加速器上的生产/商业化 (Production / Commercial on Other Accelerators)*。
   - 他们想知道这是否是为在非 NVIDIA GPU 上开发 **MAX/MOJO** 而收集反馈的一种策略。
- **社区构思训练流水线示例**：尽管 **Mojo** 缺乏原生训练支持，但社区渴望在 **PyTorch** 之前将其用于数据处理的训练流水线中。
   - 成员们询问了是否有 **Mojo** 驱动的训练流水线示例，即使是在早期阶段。
- **Mojo 中探索 Enum 的替代方案**：由于 **Mojo** 缺乏专用的 enum，社区考虑使用 **DType**（类似 enum）和 **utils.Variant** 来实现联合体 (unions)。
   - 分享了一个 [DType 实现](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/dtype.mojo) 以供参考。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Autonomous.ai 发布 Brainy**：[Autonomous.ai](https://www.autonomous.ai/robots/brainy) 首次推出了 **Brainy**，这是一款搭载 **RTX 4090 AI 超级计算机**，具有令人印象深刻的 **O3 Agent UX**，专注于图像分析。
   - 该公告因其在推动 AI 应用方面的潜力而受到关注。
- **Scout.new 服务器在负载下崩溃**：成员们注意到 **Scout.new** 因负载过重而无法使用，其他人表示 *it's fucking cooking hot damn*（形容极其火爆）。
   - 一位成员发布了 **Ray Fernando** 帖子的 [X cancel](https://xcancel.com/rayfernando1337/status/1914791594789879844) 链接，表明了极高的关注度，但目前系统并不稳定。
- **OpenAI 发布图像生成 API**：**OpenAI** 在其 [API](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1) 中使用 **gpt-image-1** 发布了 **图像生成** 功能。
   - 此次更新允许开发者将图像生成直接集成到他们的应用程序中。
- **Microsoft 宣布 Copilot Agents**：**Microsoft** 宣布了 [Copilot Agents](https://x.com/satyanadella/status/1915098359251247392)，标志着向更集成的 AI 助手迈进。
   - 细节仍在披露中，但该公告已引发了人们对这些 Agent 的功能和应用的兴趣。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 推出由 Milvus 驱动的全文本搜索**：LlamaIndex 现在通过与 [@milvusio](https://github.com/milvus-io) 的集成，支持 **使用 BM25 的全文本搜索**，从而在 **RAG pipelines** 中实现混合搜索。
   - 该功能结合了向量搜索和关键词匹配；教程可在 [此处](https://t.co/0dCi0kEn6o) 找到。
- **Agentic Document Workflow 超越 RAG 聊天机器人**：**Agentic Document Workflow (ADW)** 被定位为对 **RAG 聊天机器人** 原型的改进，提供更好的可扩展性、与现有软件的集成以及卓越的错误处理。
   - 有关 **ADW** 的更多详细信息可以在 [此处](https://t.co/ZZzr7scHhF) 找到。
- **LlamaParse 的 Text() 问题已解决**：一位用户发现 **LlamaParse** 在 next.js 中的 `getText()` 函数在 `resultType` 设置为 `markdown` 时返回部分内容，追溯到 **markdown 与 text 的比较问题**。
   - 切换到 `const reader = new LlamaParseReader({ resultType: "text" });` 纠正了该问题。
- **FastAPI 并行处理中的 MLflow Autolog 异常**：一位用户报告称，在并行任务的 **FastAPI 后台任务** 中运行 **LlamaIndex Workflow** 时，**MLflow autolog** 捕获的 **LLM 调用追踪** 不一致，导致出现 *'NoneType' object has no attribute 'info'* 警告。
   - 这表明在处理并行执行环境时可能存在 **MLflow 特有的问题**。
- **TRL 进军指令微调领域**：一位成员建议使用 **TRL (Transformers Reinforcement Learning)** 而不是 LlamaIndex 工具来对开源 LLM 进行指令微调，并提供了 [Hugging Face TRL 文档](https://huggingface.co/docs/trl/en/index) 的链接。
   - 该建议包括通过将现有 LLM 的训练蒸馏到另一个 LLM 中来创建数据集。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **提供 MCP 访谈报酬**：一名成员正为曾在实际项目中使用过 **Claude Computer Use** 和/或 **OpenAI computer-use-preview** 的人员提供 **30 分钟访谈**，报酬为 **$40**，实现过 **MCP** 的人员可获得加分。
   - 该成员需要询问“无数个”关于用户体验的问题。
- **README 翻译自动化提案**：一名成员提议将所有链接、标签和表情符号存储在单个 **JSON** 文件中，以便通过 **CI pipeline** 自动生成翻译后的 **README**。
   - 这种方法实现了集中维护，只需更新主 **README** 即可减少工作量。
- **AWSLab 成本分析导致 MCP Server 崩溃**：一名成员报告称，在使用 **AWSLab cost analysis MCP server** 生成上个月的成本报告时，**Claude Windows 桌面应用** 冻结并报错。
   - 尽管网络连接稳定，显示的错误信息仍为 *Claude’s response was interrupted*。
- **请求超时困扰 MCP Inspector**：一名成员在运行 **GitHub** 文档中的基础 **MCP server** 并使用交互式服务器时，遇到了 **MCP error -32001**：*Request timed out*。
   - 尽管在 **Claude desktop** 中运行 `mcp install test.py` 时一切正常，但该错误导致其无法运行任何工具。
- **Defang Labs 发布 Vibe-Coded MCP Server**：Defang Labs 构建了一个 **MCP server**，允许你 **直接从任何 IDE 将 vibe-coded 项目部署到云端**，并在 [其 LinkedIn 帖子](https://www.linkedin.com/posts/defanglabs_vibecoding-defang-devtools-activity-7320490826004852737-2IFE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACNoYXgBadWv4CWLbcKhgSGxWjdmu9e5dFI) 中征求反馈。
   - Defang 服务器帮助开发者将代码发布到云端。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **算术右移操作受到关注**：一名成员询问 **tinygrad** 系统中是否存在 **算术右移操作（arithmetic shift right op）**，寻求对其实现和用法的澄清。
   - 该查询表明框架内正在进行与位运算相关的开发或潜在功能添加。
- **使用 UPat 匹配 CONST**：一名成员请求一种创建 **UPat** 的方法，以匹配 **CONST**，例如立即数仅为 **5 位长**，或者低 **n 位为零** 的情况。
   - 该请求突显了对系统重写引擎中更灵活、更具体的模式匹配能力的需求。
- **寻求使用约束求解器后端进行指令排序**：团队正在转向 **约束求解器后端（constraint solver backend）**，以共同处理 **指令排序（instruction ordering）** 和 **寄存器分配（register assignment）**，从而更好地优化代码生成。
   - 这标志着 **tinygrad** 编译器向更复杂的优化技术转变。
- **Arange 被优化掉**：根据 [这个 tinygrad 笔记链接](https://xl0.github.io/tinygrad-notes/arange.html)，提到 `arange()` 会被优化掉，这可能会影响基于范围的操作的处理方式。
   - 这种优化可能会影响依赖 `arange()` 进行张量创建或操作的代码的性能和实现。
- **索引操作：查找字节索引**：一名成员建议通过获取两个 **STs** (ShapeTracker) 的 **indexed_ops**，然后代入张量索引 *i,j,k* 来查找字节索引，参考 [device.py](https://github.com/tinygrad/tinygrad/blob/6cb2d18c034fc3fb8c8c7521716c04a4674c5504/tinygrad/device.py#L330)。
   - 这种方法旨在促进 **tinygrad** 框架内更高效的内存访问和操作。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 的发布引发热潮**：成员们对 **DSPy 3.0** 表示兴奋，但针对一条 [推文](https://x.com/lateinteraction/status/1915058777491145200)，一位用户问道 *“我们可以期待什么？？”*。
   - **DSPy 3.0** 的统一愿景/设计尚未在任何地方写明，因为 *“在公开之前，有太多东西属于内部研究！”*，并链接到了 [路线图](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md)。
- **DSPy 3.0 的预计发布时间（ETA）定于 2025 年 6 月**：一位成员表示 **DSPy 3.0** 的 ETA 是 *“2025 年 6 月”*。
   - 另一位成员猜测发布会将在 **旧金山的 Databricks 活动** 前后举行。
- **Synthetic Flywheel 寻求许可**：两名成员讨论了制作一个 *“非常酷以至于需要空域许可的合成飞轮（synthetic flywheel）”*。
   - 关于实现和具体用例的进一步细节尚未明确。
- **Prompt Optimization 被比作黑魔法**：一位在一年前押注 **DSPy** 进行生成式 AI 开发的用户现在觉得 *“这不是正确的做法”*，因为 *“Prompt optimization 看起来有点像黑盒。”*
   - 该用户认为 Prompt optimization 的不可预测性使开发变得困难。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **RoPE 类引发辩论**：*torchtune* 中专门的 **RoPE**（**Rotary Position Embedding**）类实现的设计受到了质疑，主要原因是它感觉比函数更具有 *PyTorch 风格（PyTorch-y）*。
   - 该类允许 **RoPE cache** 初始化一次并重复使用，这在速度和内存之间进行了权衡。
- **集合调度（Collective Scheduling）测试**：一位成员正在测试自定义 **collective scheduling** 的吞吐量和内存占用，并计划在结果理想时提交 PR。
   - 他们正在考虑诸如 `fsdp_delay_all_reduce` 之类的参数，并将其与 **DeepSpeed stages (ZeRO 1-3)** 对齐。
- **`tune cp` 工作流在 macOS 上成功运行**：一位成员详细介绍了他们在 Macbook 上使用 `tune cp` 工作流的经验，强调了诸如需要手动搜索 recipe 和配置文件、删除文件扩展名以及解决数据集版本不匹配等问题，但在解决 **macOS 特定问题** 后最终获得了成功。
   - 该成员还指出，该工作流严重依赖于 *大量的代码复制*，这感觉不太对劲。
- **混合库设计（Hybrid Library Design）正在讨论中**：围绕 *torchtune* 中的混合库设计方法展开了讨论，该方法旨在提供易于自定义的用户脚本，同时利用库来处理通用组件。
   - 团队正在确定 *混合设计* 究竟是一个根本性的设计缺陷，还是一个用户教育/文档问题，该设计允许研究人员仅展示核心代码。
- **在新加坡通话后安排 RL 事宜**：一位用户提到他们下周晚些时候从新加坡回来后可以参加通话。
   - 该用户提供了安排通话的具体时间范围，表明他们将在下周晚些时候从新加坡返回后有空。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **寻求 SaaS 工具集成指导**：工程师们正在寻找 **Zapier** 的替代方案，以便为支持不同客户多重连接的现有 **SaaS 平台** 构建集成。
   - 他们建议将 **Composio** 作为潜在解决方案，并寻求社区对其适用性的意见或其他替代建议。
- **Nous 预告红队（Red Team）版本发布**：Nous 暗示即将发布一个专为 **红队** 社区量身定制的版本，计划于今天或明天发布，可能带有 *新的混合精度量化（mix precision quantlol）*。
   - 该公告引起了对安全和对抗性测试新工具及资源感兴趣的成员的期待。
- **SuperNova 模型获得好评**：成员们对 **Arcee-AI** 的 **SuperNova 模型** 表示赞赏，称其性能相对于其规模而言非常强劲。
   - 一位成员指出，自发布以来，这两个 **SuperNova 模型** 已成为他们的默认模型。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 阅读材料终于发布**：**LLM Agents MOOC** 的阅读材料现已在网站 [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25) 上提供。
   - 这些阅读材料与课程内容和作业高度相关，因此请优先阅读。
- **寻求资源提交确认**：一位成员询问在提交资源提交表单后是否会收到确认邮件，尽管填写了表单，但他们没有收到任何确认提交的邮件。
   - 这与课程相关，因为成员需要确认他们的提交已被正确接收。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **HF Inference API 与 Flask 绑定**：上传到 **Hugging Face** 并使用其付费 Inference API 的模型可以连接到使用 **Flask** 构建的网站。
   - **Flask** 应用程序向 **Hugging Face Inference API** 端点发送请求，然后 API 返回模型的预测结果，并随后显示在网站上。
- **Flask 请求 Hugging Face 付费推理**：一名成员询问如何将 **Flask** 网站连接到使用其付费 Inference API 上传至 **Hugging Face** 的模型。
   - 鼓励新成员通过分享所属公司、正在研究的项目、喜爱的技术工具以及希望从社区获得什么来介绍自己。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **处理程序错误困扰系统**：成员们报告系统中 *handler 产生的某些错误* 正在影响 **Gorilla LLM**。
   - 建议包括修改 [错误捕获代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286)，使其抛出错误而不是捕获错误，以协助 **debugging**。
- **排行榜上分享的调试建议**：一名成员建议修改 [代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286)，通过抛出错误而非捕获错误来帮助 **debug** **Gorilla LLM**。
   - 他们建议针对 **单个条目** 运行生成，以查看错误的 **完整堆栈追踪 (full trace)**。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **即将举行的立法 AI/技术网络研讨会公告**：企业家 Karen Suhaka（**BillTrack50** 创始人）正与硅谷华人协会基金会（Silicon Valley Chinese Assocation Foundation）合作，将于 **太平洋时间 4 月 28 日中午 12 点** 举办一场关于 AI 和技术在立法领域应用的网络研讨会，可通过 [此链接](https://forms.gle/v51ngxrWdTsfezHz8) 注册。
   - 研讨会将深入探讨构建立法技术、处理伦理考量并提供创业建议。
- **BillTrack50 创业洞察揭秘**：Karen Suhaka 将以 **BillTrack50** 为案例进行展示，分享她在建立、扩展法律科技公司以及收集客户反馈方面的经验。
   - 她将强调识别市场需求以及选择合适的数据和方法论的重要性。
- **AI4Legislation 竞赛详情公布**：研讨会将介绍 **2025 夏季 AI4Legislation 竞赛** 的项目概念，具体细节可在 [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public/tree/main) 上查看。
   - 该竞赛旨在利用 **LLMs** 和 **NLP** 的最新进展来赋能公民和选民。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1364627372569133097)** (1 条消息): 

> `iOS 语音助手, 多应用操作, 移动应用更新` 


- ****Perplexity** 发布 iOS 语音助手！**：全新的 **Voice Assistant** 利用网页浏览和多应用操作功能，可以直接通过 **Perplexity iOS app** 预订餐厅、发送电子邮件、播放媒体内容以及管理日历邀请。
   - 访问 [X 上的完整公告及示例](https://x.com/perplexity_ai/status/1915064472391336071) 并在 App Store 中更新您的应用以开始体验。
- ****Voice Assistant** 使用网页浏览**：**Voice Assistant** 利用网页浏览和多应用操作来完成预订。
   - **Voice Assistant** 已在 Perplexity iOS app 中上线。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1364316701298130957)** (1104 messages🔥🔥🔥): 

> `Perplexity AI Terms of Service, James Webb telescope, Perplexity comet release date, R1o4-mini vs grok 3 vs gemini 2.5 pro vs claude, Raycast a web app` 


- **Perplexity AI 警告违反 TOS 的行为**：一名成员发布了 [Perplexity AI 服务条款 (TOS)](https://www.perplexity.ai/hub/legal/terms-of-service) 的链接，提醒用户避免违反 TOS。
   - 这一警告似乎是由一名用户讨论使用通过其运营商计划获得的促销代码引发的，这可能违反了 TOS。
- **成员剧透詹姆斯·韦伯望远镜照片的细节**：一名成员分享了一张 [詹姆斯·韦伯望远镜拍摄的图像](https://cdn.discordapp.com/attachments/1047649527299055688/1364579832410935427/IMG_2144.jpg?ex=680a2f80&is=6808de00&hm=0e6c9c31fa09117d059bffdd1e3f964c79dc93988a5da878202099691d82b47e&)，但另一名成员迅速指出，**此类图像中的颜色并非真实颜色**。
   - 尽管揭露了颜色的真相，成员们一致认为图像依然很酷，其中一人认出这是一个螺旋星系。
- **用户讨论 Grok 和 Perplexity 模型**：成员们辩论了 **Grok**、**Gemini** 和 **O4 mini** 等不同模型在 Perplexity 上的表现，对其优缺点各抒己见。
   - 有些人认为 **Grok 3** 擅长处理一般性问题和研究，而另一些人则更喜欢用 **Gemini 2.5 Pro** 进行编程；成员们还提到 **O4 mini** 也是首选之一。
- **Perplexity iOS 语音助手发布**：Perplexity AI 发布了其**全新的 iOS 语音助手**，一名成员指出了该 [公告](https://fixupx.com/perplexity_ai/status/1915064472391336071/mediaViewer?currentTweet=1915064472391336071&currentTweetUser=perplexity_ai)，部分用户请求支持其他语言并集成到其他系统中。
   - 一名成员表示，新的 iOS 助手可以访问你的**联系人**、**日历**、**提醒事项**和 **Apple Music**，并能**创建电子邮件**。
- **Perplexity 的图像生成功能仍存在 Bug**：用户报告了 Perplexity 图像生成的问题，例如默认使用 **Flux 模型**且无法正确遵循提示词（prompts）。
   - 还有人指出，编辑生成图像的能力有限，因为系统倾向于重复使用原始附件图像，而不是修改生成的图像；在幽默的讨论中，有人将这种体验描述为 *"delulu"*（幻想）。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1364421196719980555)** (3 messages): 

> `Perplexity AI, comprehensive 10000` 


- **分享了 Perplexity AI 的 URL**：一名成员分享了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/given-the-current-condition-an-cRXRlWsRTEOrL99ZHzd3HQ)。
   - 另外还分享了 [第二个搜索结果](https://www.perplexity.ai/search/generate-a-comprehensive-10000-9FO7aPu9QHKWRfdDC2E88Q)。
- **发布了图片**：一名成员分享了一个 [图片链接](https://pasteboard.co/FiWizDPt8Knd.png)。
   - 图片内容未说明。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1364351562901880873)** (11 messages🔥): 

> `API key revoke, Requests per API, Office hours` 


- **API 请求不进行网页搜索**：一名成员报告称，通过 API 发出的请求不会执行网页搜索，尽管在 Playground 中一切运行正常。
   - 另一名成员随后建议该成员尝试一个 [特定的 curl 请求](https://api.perplexity.ai/chat/completions) 来解决此问题。
- **API Key 被撤销了！**：一名成员提到要撤销 API Key。
   - 他还提到要更新错误消息中的链接。
- **今天下午的 Office Hours**：一名成员提醒大家今天下午有 Office Hours。
   - 他还为想要加入的人提供了一个 [Zoom 链接](https://events.zoom.us/ev/Akzh8Q9GwGtQ8-5yeP1A6B0kQBND1W67rbimE3koC4L_L4ZP65f2~Ag4nJHk6gbPxvgM1f_OCr6BzgyKoKK7hLYpE3HmzJ69MnMG3CvFABoNg6Q)。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1364314852876877894)** (1097 messages🔥🔥🔥): 

> `Llama games LM arena, emoji, GPT-4.1, small cheap models, arc prize` 


- **Llama 在 LM Arena 的“刷榜”行为**：成员们讨论了 **Llama** 在训练过程中可能如何针对 **LM Arena** 进行了“刷榜”（gamed）。
   - 随后话题转向了针对人类风格偏好进行优化的风格受控 IMBY 是否能解决 AI 过度使用表情符号（emoji）等问题。
- **表情符号助你获得更多约会**：一项 [研究](https://www.ktsa.com/study-people-who-use-more-emojis-have-more-dates-and-sex/) 表明，使用更多表情符号的人拥有更多的约会和性生活。
   - 有人建议，在 **LM Arena** 中表现得**随和**、**积极**并使用**表情符号**可能会有所帮助，但一位成员不同意“随和”一定是好事。
- **GPT-4.1 表现出色且更便宜**：**GPT-4.1** 被认为非常出色，因为它在 **Sonnet** 擅长的领域与其表现接近，但价格更便宜。
   - 然而，有人指出 **GPT-4.1 mini** 更好，并且拥有比 **Claude** 更高效的 tokenizer，尽管它在网页设计或编程视觉效果方面并不理想。
- **小型（廉价）模型的复古基准测试**：一位成员分享了 [小型（廉价）模型的快速基准测试](https://cdn.discordapp.com/attachments/1364321889044136067/1364330919686705286/Screenshot_2025-04-22_at_22.02.43.png)，并对 **Gemma 3** 在其成本下的优异表现感到惊讶。
   - 该成员指出，他们为前沿模型（frontier models）准备了另一套更难的基准测试问题。
- **OpenAI 的 O3-Preview**：成员们讨论了 **OpenAI** 的 **O3-preview** 基准测试，以及发布的 **O3** 弱得多的事实。
   - 有人建议 **O3-pro** 可能会在 ARC-1 上达到 80% 以上，在 ARC-2 上获得 10% 到 20% 的分数，尽管 **O3 preview high** 每个任务的成本高达数千美元。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1364319188637847592)** (590 messages🔥🔥🔥): 

> `Manus Pricing, DeepSeek vs Manus, Genspark, Credits, OpenAI vs Manus` 


- **用户辩论 Manus 的定价和积分系统**：用户正在争论 **Manus** 的成本是否过高，一位用户建议增加一种消耗资源更少的 [慢速处理模式](https://link.to/slow-processing-mode)。
   - 一些用户表示，“考虑到积分仍然非常受限，这个价格非常昂贵”。
- **用户将 DeepSeek 和 Genspark 与 Manus 进行对比**：一位用户将 **Manus** 与 **DeepSeek** 和 **Genspark** 进行了对比，指出虽然 DeepSeek 提供每日积分，但表现不如 Manus。
   - 另一位用户表示赞同，提到 *DeepSeek 是通过他们的 API 而不是模型来赚钱的*。
- **功能建议层出不穷：积分、定价和模型选择**：成员们向团队提出了 [账户间积分共享](https://link.to/credit-sharing) 的想法，而另一位成员建议 [根据每月使用小时数定价](https://link.to/pricing-by-hours)。
   - 其他人呼吁增加自定义模型选择功能，例如更便宜的 **Gemini 2.5 Pro** 或更昂贵的 **Claude 3.7 Sonnet**。
- **社区辩论数据隐私和安全**：用户讨论了对数据隐私的担忧，包括 [Manus 是否与 Claude 共享数据](https://link.to/claude-policy)，一些人开玩笑说比起美国政府，他们更希望数据流向中国。
   - 一位成员表示，“这是我见过的唯一一家非福布斯 500 强公司开发的、具备实战能力的 AI”。
- **用户发现制作 Minecraft 模组的潜力**：成员们讨论了使用 Manus [制作 Minecraft 模组](https://link.to/minecraft-mods) 的可能性，包括将它们编译成 JAR 文件。
   - 其他成员表示，团队需要学习如何“真正地采纳更多建议”。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1364345912431743047)** (8 messages🔥): 

> `Sonnet 3.7 capacity issues, Clerk authentication delays, OpenRouter PDF support, PDF processing engines, Gemini API PDF Input` 


- **Sonnet 3.7 获得容量提升**：OpenRouter 解决了 **Sonnet 3.7** 的容量问题，并实施了改进以降低错误率。
   - 用户现在应该会看到错误率大幅下降，官方对造成的干扰表示歉意。
- **Clerk 身份验证面临停机**：OpenRouter 的身份验证提供商 **Clerk** 经历了延迟和停机，导致 **401 错误**和登录困难；请查看 [Clerk 状态页面](https://status.clerk.com/) 获取更新。
   - Clerk 报告称在事件发生后其端已恢复。
- **OpenRouter 开启通用 PDF 支持**：OpenRouter 现在为每个模型都支持 **PDF 处理**，可能是首个这样做的平台，正如在 [X.com](https://x.com/OpenRouterAI/status/1915083006349382033) 上发布的 [视频演示](https://cdn.discordapp.com/attachments/1092729520181739581/1364636811003035789/pdf2.mp4?ex=680a6491&is=68091311&hm=33ff94487e038de43aea6ad12c6041fe14da683e6fd29de0b90e94016bada256&) 所宣布的那样。
   - 这项新功能包括通用兼容性，可处理任何 PDF 类型，并对 **Gemini**、**Anthropic** 和 **OpenAI** 等提供商提供原生支持，可通过 **API** 和 **OpenRouter Chatroom** 访问。
- **PDF 处理器定价**：OpenRouter 引入了两个 **PDF 处理引擎**：`mistral-ocr` 定价为 **每 1000 页 2 美元**，支持 OCR 以及文本和嵌入图像提取；`pdf-text` 免费，仅提供文本提取，不支持 OCR 或图像，详见 [文档](https://openrouter.ai/docs/features/images-and-pdfs)。
   - 有用户希望在 **mistral OCR** 和 **纯文本** 之间能有一种中间方案，*比如 smol docling*。
- **Gemini 获得强大的 PDF 支持**：**Gemini API** 支持 **PDF 输入**，包括长文档（最多 **3600 页**），通过原生视觉处理来理解文本和图像内容，如 [OpenRouter 文档](https://openrouter.ai/docs/features/images-and-pdfs) 所示。
   - 一位成员注意到，在使用 Gemini Flash 时，如果除了页面本身外还提供一份纯文本解析，数据提取效果会有显著提升。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1364315429321052260)** (259 messages🔥🔥): 

> `Tool Calling Limitations, Google Gemma Quantization, Gemini Search Grounding, Account Creation Issues, Free Model Function Calling` 


- **OpenRouter 经历身份验证问题**：用户报告在创建新账户时遇到问题，收到 *"Error 401: User not found"* 消息，这归因于 **OpenRouter** 身份验证提供商的响应变慢。
   - 团队进行了调查并确认了问题，在 [Clerk 状态页面](https://status.clerk.com/) 提供了更新，随后确认问题已解决，尽管一些用户因为测试而产生了多个不需要的账户。
- **Gemini 2.5 Pro 挣扎于速率限制 (Rate Limits)**：用户报告在使用 **Gemini 2.5 Pro** 预览版（尤其是免费版）时频繁出现 *"Rate limit exceeded"* 错误，引发了关于可靠性和可能解决方案的讨论。
   - 一个建议是在账户设置的“集成”页面使用个人 Google AI Studio API key 来提高限制，但实际的降级行为以及对 Google RPD 的影响仍不清楚。
- **OpenRouter 增加 PDF 支持**：**OpenRouter** 宣布了通用 PDF 支持，但初始文档链接 ([https://openrouter.ai/docs/features/images-and-pdfs](https://openrouter.ai/docs/features/images-and-pdfs)) 一度失效并被迅速修复。
   - **Mistral OCR** 是其处理引擎，用户讨论了其定价，一些用户注意到与直接使用 Mistral 相比有溢价，但其他用户对这一新功能表示兴奋。
- **Deepseek v3 在函数调用 (Function Calling) 方面表现出色**：**Deepseek V3** 在上下文较小时擅长函数调用。
   - 然而，一旦上下文增加，其表现就会变差。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1364317633050312786)** (172 messages🔥🔥): 

> `Scout 在 128GB 统一内存（unified memory）下的使用案例，Unsloth 对 GLM-4 9B/32B 模型支持，Unsloth Dynamic v2.0 量化版本发布，Torch 2.7 发布及其变化，评估基准：MMLU, Humaneval, Aider Polygot` 


- **Scout 在统一内存中发现小众用户**：用户正在探索 **Scout** 在 **128GB 统一内存（unified memory）** 下的使用案例，理由是在低功耗下具有不错的吞吐量。
   - 提到的一个用例是 **RP 追踪器**，用于提取结构化信息以防止角色设定不一致，尽管由于尺寸原因并未使用 Llama 4。
- **GLM-4 获得 Transformers 集成**：如果 **GLM-4 9B/32B 模型**能在 Transformers 中运行，Unsloth 即可提供支持。不过有用户反馈在微调中由于应用模板和合并适配器（adapters）的问题仅获得部分成功。
   - 有报告称一个与 **GLM4 的 rope 维度**为 **64** 相关的问题避开了大多数推理引擎的检测。
- **Unsloth 动态量化（Dynamic Quantization）上线**：Unsloth 即将发布 **Unsloth Dynamic v2.0 quants**，声称其表现非常出色，并链接到了 [Hugging Face 集合](https://huggingface.co/collections/unsloth/unsloth-dynamic-v20-quants-68060d147e9b9231112823e6)。
   - 据观察，各方面都有改进，包括 **Q8**，而大部分收益体现在 **Q4**。Unsloth 正在进行针对 iMatrix 量化、Google 的 QAT、标准 GGUF 以及旧版 Unsloth dynamic iMatrix 的 5-shot MMLU 基准测试。
- **Torch 2.7 发布并提供上下文支持**：**Torch 2.7** 已发布，其特点是支持在 Dynamo 中追踪 contextlib.contextmanager 以及追踪生成器（generators），[发布说明可在 GitHub 上查看](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)。
   - 一位用户一直在等待该版本发布，但不得不专注于 DSA（数据结构与算法）和操作系统的学期考试。
- **Unsloth 团队辩论基准测试方法论**：Unsloth 团队正在对其新的量化方法进行基准测试，并与 Google 的 QAT 进行对比，讨论了使用 **5-shot MMLU** 进行评估，并建议参考[这篇论文](https://arxiv.org/abs/2407.09141)。
   - 讨论涉及使用不同的基准测试评估不同的量化水平，并对 perplexity, humaneval 和 swe-bench 的实用性存在分歧。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1364328323324313620)** (6 messages): 

> `Token 处理速度，PyTorch 基准测试` 


- **每个 Token 处理需要五分之一个世纪**：一位成员开玩笑地指出，在一段 [YouTube 视频](https://www.youtube.com/watch?v=3q_ItuNNpmYE)中，*每个 token 需要五分之一个世纪*来处理。
   - 这一幽默评论是在讨论 token 处理速度和效率的背景下提出的。
- **PyTorch 的 Infra 网站展示了有趣的基准测试**：一位成员分享了 [PyTorch 基础设施（infrastructure）网站](https://hud.pytorch.org/benchmark/compilers)的链接，其中包含随每个 PR 更新的有趣基准测试。
   - 基准测试选项卡被强调为对于追踪 PyTorch 性能的人来说包含最有趣的信息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1364372235187388416)** (53 messages🔥): 

> `GRPO 模型，使用数字标签的分类任务，Llama-4 微调更新，嵌入（Embedding）模型推荐，为 Vtuber 微调 Llama` 


- **GRPO 模型选项？**：一位用户询问了适用于 **GRPO** 的模型范围，质疑是否仅限于 **Gemma 3 (1B), Llama 3.1 (8B), Phi-4 (14B), 和 Qwen2.5 (3B)**。
   - 一位成员回答说 *除 VL 以外的所有模型*。
- **分类任务标签用 Int 还是 Str？**：一位用户询问使用数字标签的分类任务是否需要检查 **int** 或 **str** 类型。
   - 其他成员建议必须是 **int**，并指向 Hugging Face 文档以获取详细信息。
- **Llama-4 微调即将推出**：一位用户询问关于 **Llama-4 微调（finetuning）** 的更新。
   - 一位成员回答说 *这周肯定会为了准备 llamacon 而发布*，但目前还没有项目链接。
- **需要嵌入（Embedding）模型推荐**：一位用户正在为 **<= 1024 tokens** 的文档块和少于 **512 tokens** 的问答对寻求嵌入模型推荐。
   - 他们担心针对 **8K tokens** 的嵌入模型可能不合适或没有必要。
- **寻求为 Vtuber 微调 Llama 的指导**：一位 AI 经验有限的用户请求帮助 **为 Vtuber 应用微调 Llama**。
   - 目前尚不清楚该用户是否找到了合适的帮助。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1364321792990249000)** (27 messages🔥): 

> `Reasoning Models, LLM Novelty, Training Data Limitations, Sampling Token Sequences` 


- **推理模型提升了 LLM 的概率，而非能力**：成员们讨论了使用 **reasoning models** 会使期望的补全结果概率更高，但并不会从本质上提高模型在基础模型自身能力之外的底层能力。
   - 一位成员表示 *它让提示词编写变得更容易，但并没有让模型变得更有能力*。
- **训练减少了事实型 LLM 的随机性**：有人认为，对于需要**事实准确性**的任务，一个需要 *k* 次生成才能成功的模型比只需要一次生成的模型能力更弱，这突显了训练旨在减少随机性。
   - 目标是通过有效的训练来减少准确性所需的 *k* 值。
- **在 LLM 中主观地定义新颖性**：关于如何定义 **LLMs** 中的**新颖性**引发了辩论，一种观点认为，在训练集和输入上下文之外不存在真正的新颖性，因为模型在没有适当上下文的情况下无法进行逻辑飞跃。
   - 反方观点认为**新颖性**是主观的，并指出 **LLMs** 可以生成训练数据中没有明确出现的 token 序列，从而引发了关于此类序列何时变得具有新颖性的疑问。
- **探索带宽问题**：一位成员分享了一个 [Google Colab notebook](https://colab.research.google.com/drive/1JC1cEsk-3SxIUPWL7wF0eveelyo3e7fy?usp=sharing)，详细介绍了他们关于 **MLA** (Multilayer LSTM Architecture) 如何解决**带宽问题**的最新研究，并征求早期反馈。
   - 该报告重点在于将 **MLA 概念作为核心**来解决带宽问题。
- **预训练对于最强 LLM 仍然至关重要**：一个关键结论是，一切都是预训练的下游产物。
   - 成员们一致认为，*你仍然需要尽可能强大的基础模型*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1364318162723803259)** (170 messages🔥🔥): 

> `SillyTavern as front end for LM Studio, Pinokio install, 5090 loading issues on LM Studio, Cuda 12, Bitnet CPP` 


- ****SillyTavern** 前端接入 **LM Studio****：用户可以使用 [SillyTavern](https://github.com/SillyTavern/SillyTavern) 作为 **LM Studio** 的前端，以获得 ERP 功能并自定义聊天机器人体验。
   - 一位用户提供了详细的 5 步指南：1. 安装 [Pinokio](https://pinokio.computer/)，2. 使用 Pinokio 安装 SillyTavern，3. 运行 LM Studio，加载模型并启动服务器，4. 启动 SillyTavern 并将 LM Studio 配置为后端，5. 开始在 SillyTavern 中聊天。
- ****RTX 5090** 早期采用者遇到无限加载循环**：一位用户报告在测试版 LM Studio 上使用 **RTX 5090** 时遇到*无限加载问题*，社区成员纷纷提供帮助。
   - 成员们建议确保使用最新的 **LM Studio 版本 (0.3.15 Build 9)** 和 **CUDA 12**，并在运行时切换 beta 选项，同时提供了变通方法和故障排除步骤，并观察到 *目前拥有 5090 的人还不多，所以不知道测试程度如何*。
- ****BitNet.cpp** 框架支持 1-bit LLMs**：来自 Microsoft 的 [BitNet.cpp](https://github.com/microsoft/BitNet) 框架是 **1-bit LLMs** 的官方推理框架，提供优化的内核以在 CPU 上实现快速且无损的推理，并计划支持 NPU 和 GPU。
   - 该框架支持 **BitNet b1.58** 等模型，并包含一套优化的内核，支持在 CPU 上对 1.58-bit 模型进行快速且无损的推理。
- ****VLM** 安全担忧**：成员们讨论了 **VLM**（视觉语言模型）领域的发展情况，*特别是在审查制度方面*。
   - 一位成员指出 [Microsoft 发布了一个审查较少的 R1 版本](https://huggingface.co/collections/microsoft/mai-ds-r1-68003c2753a06be7b9632154)，这是一个具有视觉能力的模型，可以实际处理图片而没有内置的清教徒式过滤器。
- **成员分享保护 **LM Studio** 安全的策略**：一位用户询问如何安全地将 **LM Studio** 暴露在互联网上进行远程推理，因为默认情况下这并不是一个安全的解决方案。
   - 建议的解决方案包括使用 **OpenVPN** 或编写自定义身份验证，并警告浏览器扩展的安全性不如 Telegram 或 WhatsApp 等替代方案，后者允许基于用户 ID 的使用限制。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1364362254451085392)** (36 messages🔥): 

> `RTX 5060 Ti 16GB, Macbook for LLM, 5090 finetuning, DDR3 for AI, Smol models` 


- **爱好者尝试 RTX 5060 Ti**：一位用户询问关于 **RTX 5060 Ti 16GB** 的情况，但在加载模型时遇到问题，必须更新到最新的 beta 版本。
   - 另一位用户购买它是考虑到这可能是未来 4 年内能负担得起的最后一张显卡，并将所有软件切换到了 beta 版，目前运行似乎正常。
- **考虑将 Macbook 用于 LLM**：一位用户询问通过 **getupgraded.com** 获取一台 **Macbook** 用于 **LLM** 的事宜。
   - 另一位用户指出，**M4 MBP** 的性能与他们的 **4070 Ti Super** 相当，但提醒不要对电池续航抱有太高期望。
- **5090 显卡用于微调**：一位用户询问是否可以使用单张 **5090** 通过 **QLoRA** 微调较小的模型。
   - 另一位成员确认了可行性，建议使用 **Unsloth** 配合 **Mistral Small 3.1**，但建议在 4-bit 模式下且 batch size 合理的情况下，不要超过 **7B** 参数。
- **DDR3 系统不适合 AI**：一位用户分享了他们的硬件配置（Core i3-5010u, 2x4gb DDR3L-1600mhz, Sata SSD, Intel HD Graphics 5500），引来评论称其硬件并非为本地 **AI** 设计。
   - 另一位用户补充说，**DDR3** 系统并不是为 **AI** 使用而设计的，虽然可能可以运行 1b 模型，但速度会非常慢。
- **用于 instruct 的 Smol 模型**：一位用户分享了一些 **smol 模型**，这些模型*更多是为 instruct 而非 chat 设计的*，并链接到了 [QuantFactory/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF)。
   - 他们指出这些模型可以提供 **135, 256, 360, 1.7** tokens 的输出。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1364317272285646970)** (171 messages🔥🔥): 

> `Gemini, Ollama, Aider Benchmarks, Cursor IDE, Deepseek R2` 


- **Gemini 2.5 Pro 的“可爱”错误**：用户讨论了 [Gemini 2.5 Pro](https://ai.google.dev/) 生成的代码格式不正确，导致提交更改时出现错误和困难，一位用户报告由于格式问题出现了 *810 个错误*。
   - 尽管存在这些问题，一位用户发现 **Gemini** 在其他模型失败后一次性解决了一个问题，导致他们宣称 *gemini is god*。
- **Cursor 是生产力神器**：用户正在使用 [Cursor](https://cursor.sh) 并达到了新的生产力水平。
   - 一位用户表示他在 Aider 出现之前就有 Cursor 订阅，现在虽然更多使用 Aider，但偶尔仍会使用 Cursor，比如将其用于将 **Python** 代码转换为 **C#**。
- **O3-preview 完胜 O3 —— 真的吗？**：社区注意到 **O3-preview** 在某些基准测试中表现优于 **O3**，尽管之前的版本显示出相反的趋势。
   - 有关于这些 [benchmarks](https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml#L7) 是否准确反映真实世界性能的讨论，一些人认为模型是专门为基准测试的成功而调整的，而非实际用途。
- **Deepseek R2 发布不是开玩笑**：用户开玩笑地宣布 **Deepseek R2** 发布。
   - 一位用户甚至声称看到了发布，开玩笑说 *我刚拉了一大堆，也许那就是 R2 出来了？*
- **请出示 OpenAI ID**：有讨论关于模型服务可能要求你提供组织官方 ID 的可能性。
   - 一位用户说 *他们把你所有的想法都和你的护照关联起来，这真是太“方便”了*。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1364350940223897700)** (30 messages🔥): 

> `Aider 排除 yes/no 响应, 以只读方式加载上下文, 优秀的模型组合, Aider 排行榜, Gemma 27b 图像` 


- **Aider 配置调整：排除 'Yes/No' 响应**：一位用户询问如何配置 Aider 以从 `.aider.input.history` 中排除 *'yes/no'* 响应，从而减少干扰，特别是在没有原始问题上下文的情况下。
   - 用户指出，这些单字符响应毫无用处，且会在历史记录中产生噪音，尤其是在上下文被保存为只读时。
- **解锁 Aider 编辑只读上下文文件**：一位用户寻求关于如何编辑以只读模式加载的上下文文件的指导，并对使用 `/context` 命令表示困惑。
   - 用户发现 [官方文档](https://aider.chat/docs/usage/commands.html) 在解释上下文模式（特别是无参数使用时）方面不够充分。
- **Aider 最小化成本的模型组合建议**：一位从 Cursor 迁移过来的 Aider 新用户请求关于高性价比模型组合的建议，以避免超过 **$20/月** 的预算。
   - 有建议提出使用 [OpenRouter](https://openrouter.ai/) 来测试免费模型，并参考 Aider 排行榜来评估模型的成功率和成本。
- **解决 Aider 启动问题：API Key 和文件参数**：一位用户报告了启动 Aider 时的问题，并附带了一张显示明文 Deepseek API Key 的图片，这引发了立即重置 Key 的建议。
   - 解决方法包括删除 `.aider.conf.yml` 文件或对其重命名，并确保在 Aider 启动*后*使用 `/add` 添加文件，而不是将其作为命令行参数。
- **请求 Aider 在每次请求时发送完整树状图 (Tree Map)**：一位用户询问是否可以配置 Aider 在每次请求时发送完整的树状图，以便更好地了解项目结构和上下文。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1364385489410330657)** (12 messages🔥): 

> `RL Agent, 连续符号, 多 Agent RL, 通信协议` 


- **RL Agent 学习使用符号进行通信**：一位成员分享了他们的 [论文](https://x.com/superspeeg/status/1914691313318105305)，内容关于 **RL Agent** 学习使用 **连续符号 (continuous signs)** 而非离散符号来交流其 **MDP**。
   - Agent 学习到的 **通信协议** 最初表现为象形文字，随后演变为抽象符号，这与许多人类书写系统的轨迹相似，详见 [arxiv.org/abs/2502.01568](https://arxiv.org/abs/2502.01568)。
- **连续信号与进化惩罚**：一位成员询问了新论文中关于 **信号相似性** 的概念，质疑在几乎是对比目标的情况下，某些信号是否可能显得特别相似。
   - 论文作者回答说，他们的目标本身并不是优化信号的视觉方面，而是看它们是否能诱导正确的动作，并指出 *在现实世界中把螃蟹误认为蜘蛛可能是致命的*。
- **对多 Agent RL 中信息传递的兴趣**：另一位成员对这篇论文表示兴奋，称他们一直想尝试围绕 **多 Agent RL 中的信息传递** 的想法，但一直没有时间。
   - 论文作者回应称，后续研究仍有很大空间。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1364354875714043934)** (80 条消息🔥🔥): 

> `Linear representation hypothesis debunked, Biologically inspired architecture for sequential modeling, Native Sparse Attention analysis, Overfitting models to single datapoints, AI-generated research papers` 


- **线性表示假设 (Linear Representation Hypothesis) 被推翻**：来自 Tegmark 团队的一篇 [论文](https://arxiv.org/abs/2402.09268) 推翻了 **线性表示假设**，认为它既不具有普遍性，也不是一个好的假设。
   - 文中还提到 Mikolov 的 **Glove 相关研究** 被推翻，因为他们使用了一种奇怪的检索系统，即排除原始点的最近邻检索。
- **O(n) 模型实现 300k 长度外推**：一种受生物启发、具有 **O(n) 复杂度** 的序列建模架构在从掩码序列中加总两个数字的合成任务上进行了测试，并外推到了 **300k 长度**。
   - 该模型仅有 **39k 参数**，即使在比训练数据长得多的序列上也能保持相似的 MSE loss，并且在 1000-2500 长度的序列上训练、仅用 5000 长度的验证集进行验证时，成功实现了长度外推（如 [图片](https://cdn.discordapp.com/attachments/747850033994662000/1364540420692115497/image.png?ex=680a0acc&is=6808b94c&hm=ebda36842ecfed21615df2269d5bb4f5121436cbe7fe61513f2ef2fe3250725e&) 所示）。
- **Native Sparse Attention 扩展性受到质疑**：DeepSeek 对 **Native Sparse Attention** (NSA) 的分析表明，它无法扩展到 **1M+ 规模**，详见 [Google Doc](https://docs.google.com/document/d/1kXQ7d-9bSWmAU4c-Tq7iQzzJDWaJe-MuyjI_D3x1Et8/edit?usp=sharing)。
   - 讨论围绕其时间复杂度和 Attention 中 Softmax 的作用展开，强调了它如何通过减少对最高匹配项的关注来避免模糊的“Value”结果。
- **高效过拟合技术探索**：有人提出了一个关于如何最有效地让多个模型分别过拟合 minibatch 中不同单个数据点的问题。
   - 讨论建议，虽然学习率为 1 且进行单次前向/后向传递在高容量下似乎足够，但非线性特性使得需要多个步骤，不过 **模型线性化 (linearizing the model)** 可以使问题简单化，二阶优化器也能处理。
- **AI 生成的研究论文即将到来？**：Sakana AI 发布了他们的 [AI-Scientist-v2 项目](https://github.com/SakanaAI/AI-Scientist-v2)，展示了仅需 **$15-20 的 API tokens** 即可生成完整研究论文（包括假设和实验测试）的能力。
   - 这引发了关于 arXiv 可能会被 **AI 生成的论文** 淹没的猜测。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1364627981896519812)** (75 条消息🔥🔥): 

> `Multihead Latent Attention (MLA), DeepSeek, RWKV architecture, Residual Stream Subspaces` 


- **DeepSeek 的 MLA 限制了 Attention**：DeepSeek 的 **Multihead Latent Attention (MLA)** 通过限制 Key 和 Value 头在 **7K 维 Residual Stream** 的 **512 维子空间** 中进行读写来限制 Attention。
   - Query 头可以从一个独立的 **1.5K 子空间** 读取，旨在节省内存带宽并可能提高性能，尽管一些成员质疑这是否真的是一个子空间。
- **澄清 Multihead Latent Attention 机制**：一位成员澄清说，MLA 通过 **W^DKV** 压缩整个隐藏状态，而不仅仅是一个子空间，并引用了他们的 [研究论文](https://arxiv.org/abs/2407.12077)。
   - 他们强调这种压缩涉及所有 **7168 个隐藏维度** 的线性组合，表明这不仅仅是选择一个子空间。
- **隐藏状态维度的空间权衡**：讨论集中在预先指定 Query 和 Key 的维度是否可行，特别是在像 **RWKV** 这样投影很常见的架构中。
   - 一位成员指出，这可能需要占用其他重要用途的空间，因为 *隐藏状态维度某种程度上是零和博弈*，但在进行架构调整后可能仍然可行。
- **MLA 强制 Key-Value 关系**：MLA 不在 Q 和 K 之间强制执行任何预指定的维度，而是强制 Key 和 Value 之间比 MHA 具有更强的相关性。
   - 另一位成员补充了为什么模型可能会为 q, k, v 分别使用整个空间，因为 *隐藏状态中没有足够的空间来容纳所有内容*。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1364508903353876510)** (1 messages): 

> `PyTorch, flashStream, Kernels, Leaderboard` 


- **用于 Leaderboard 提交的 FlashStream Kernels**：一位成员询问是否可以使用名为 **flashStream** 的 **PyTorch** 包来编写 kernels 并将其提交到 leaderboard。
   - 未收到回复。
- **使用 flashStream 进行 leaderboard 提交**：一位用户询问如何利用他们的 **PyTorch** 包 **flashStream** 来开发 kernels 并提交到 leaderboard。
   - 该查询特别涉及 **flashStream** 在 leaderboard 提交中的可行性，特别是关于 kernels 的创建和提交。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1364314866562760836)** (11 messages🔥): 

> `Triton FP4 support, FP16 to FP4 conversion, TileLang for FP4, FP4 vs INT4 benchmarks, Pyright and Triton issues` 


- ****Triton** 增加 **FP4** 支持**：**Triton** 现在支持 **FP4**，详见[此教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)，其中 **FP4** 输入以 `torch.uint8` 张量的形式提供，每个张量存储 2 个 **FP4**。
- **通过 **CUDA** Kernels 将 **FP16** 转换为两个 **FP4****：一位成员询问关于使用 **Triton** 或 **CUDA** kernels 将 **FP16** 转换为两个 **FP4**，然后将其打包进 **INT8** 的问题。
- ****TileLang** 为 **FP4** 提供简单快速的解决方案**：对于 **FP16** 到 **FP4** 的转换，一位成员推荐了 [TileLang](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py) 作为一种简单快速的解决方案。
- ****FP4** 在 **H100** 上的基准测试显示出良好的结果**：一位成员分享了 **H100** **SXM** 上的基准测试结果，指出 *“fp16 gemm 约为 750T”*，而 **FP4** 通过利用 **TMA** 在 **CUDA** core 上流水线化反量化并在 tensorcore 上进行 **GEMM**，取得了理想的数据。
- ****Pyright** 与 **Triton** 的 Bug**：一位成员报告了 **Pyright** 和 **Triton** 的问题，具体表现为任何 `cdiv` 都会将函数的其余部分标记为不可达，尽管代码运行正常。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1364652511834669069)** (1 messages): 

> `RightNow AI, CUDA kernels, browser coding, bottleneck analysis` 


- **RightNow AI V2 正式上线**：一位成员介绍了 **RightNow AI**，这是一个旨在直接在浏览器中编写优化后的 **CUDA kernels** 的平台，并分享了 [V2](https://www.rightnowai.co/) 的发布。
- **用于快速 CUDA Kernels 的实时瓶颈分析**：该 AI 能够根据用户描述生成经过 profiled 的快速 kernels，并提供实时**瓶颈分析**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

marksaroufim: would love some feebdack on https://github.com/pytorch/pytorch/issues/152032
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1364640833017479228)** (1 messages): 

> `MLA kernel, Compute bound inference` 


- **DeepSeek 撰写 MLA Kernel 更新**：一位成员重点介绍了 **DeepSeek** 的新[博客文章](https://github.com/datacrunch-research/blogs/blob/main/deepseek-mla-roofline/deepseek-mla-roof.md)和 kernel 更新。
   - 该成员也在撰写自己的博客文章，讨论 **MLA** 如何作为一个*受计算限制的推理 kernel (compute bound inference kernel)*。
- **MLA 受计算限制的推理**：一位成员指出，他们博客的主题是关于 **MLA** 如何作为一个*受计算限制的推理 kernel*。
   - 他们链接了 **DeepSeek** 的新博客文章和 kernel 更新。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1364335473610330213)** (3 messages): 

> `ncu, import-source, app-range, collective op` 


- **建议使用 `ncu --import-source yes` 命令**：一位成员建议使用带有 `--import-source yes` 标志的 `ncu` 命令。
   - 然而，另一位成员回应称，使用此标志时会抛出错误 *Option import-source is not supported during range replay*。
- **`app-range` 是测试 collective op 唯一有效的模式**：一位成员表示，`app-range` 是唯一适用于他们用例的模式。
   - 他们正在测试一个 collective op。

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1364317318804406273)** (27 messages🔥): 

> `torch.compile static cache, Qwen2.5-3B performance, fp16 performance, vLLM's compression kernel` 


- **静态缓存（Static Cache）对 Qwen2.5-3B 没有帮助**：一位成员报告了在单卡 **H100** 上为 **Qwen2.5-3B** 启用 `cache_implementation="static"` 的 `torch.compile` 时出现了令人困惑的结果，显示其性能相比 *torchao* 基准测试有所下降，并附带了 [基准测试代码](https://cdn.discordapp.com/attachments/1364317318804406273/1364317899619303565/message.txt)。
   - 附带的代码通过 `python3 torchao_test.py --model_name Qwen/Qwen2.5-3B-Instruct --output_csv qwen_3.csv --compile true > qwen_3b_compile.txt` 运行，结果显示 **fp16** 是最快的。
- **对于小 Batch，仅权重量化（Weight-Only Quant）优于权重与激活量化（Weight&Activation Quant）**：一位成员指出，对于单 Batch，权重与激活量化可能比仅权重量化更慢，这可能是由于内存移动开销导致的。
   - 解释称，激活量化需要从全局内存读取激活值、进行量化并写回，这导致了更多的数据移动，并可能导致较小 Batch 的性能下降。
- **vLLM 压缩内核（Compression Kernels）揭秘**：一位成员询问 **vLLM** 压缩内核的位置，希望从中学习。
   - 另一位成员回答说 **vLLM** 有两个压缩内核：针对非 Hopper GPU 的 **Marlin** 和针对 Hopper GPU 的 **Machete**。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1364512163460419585)** (2 messages): 

> `PyTorch ATX Meetup, Triton, Austin, Red Hat, Intel` 


- **Triton 成为 PyTorch ATX Meetup 的焦点**：下一届 [PyTorch ATX Meetup](https://www.meetup.com/pytorch-atx/events/306856316/?_xtd=gqFyqTI1OTIzMDY4NqFwo2FwaQ%253D%253D&from=ref) 将聚焦于 **Triton**，演讲嘉宾来自 **Red Hat, Intel, AMD, IBM Research, 和 UT Austin**。
   - 计划于 **4 月 30 日星期三下午 5-8 点**在奥斯汀的 **AT&T Hotel and Conference Center (Lavaca Classroom)** 举行。
- **丹佛地区聚会咨询**：一位成员询问了在 **丹佛地区（Denver area）** 举办聚会的可能性。
   - 这表明社区的影响力有扩展到奥斯汀之外的潜在兴趣。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1364378538362408970)** (3 messages): 

> `TileLang, CUDA, Triton` 


- **TileLang 对比 CUDA/Triton：选择你的 GPU 之路**：一位新手询问是否应该学习 **TileLang** 而不是 **Triton** 作为 GPU 编程的切入点。
   - 一位成员建议学习 **CUDA** 和 **Triton** 对初学者更好，而另一位成员认为 *TileLang* 对初学者也很友好。
- **TileLang：是否适合初学者？**：讨论围绕 **TileLang** 与 **CUDA** 和 **Triton** 相比是否适合 GPU 编程初学者展开。
   - 意见不一，有人建议 **CUDA** 和 **Triton** 是更好的起点，而其他人则认为 **TileLang** 对新人来说很容易上手。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1364507978950246450)** (1 messages): 

> `Tensor Parallelism, Static Split-K` 


- **理解带有静态 Split-K 的张量并行（Tensor Parallelism）**：断言 (**batch_size, N, num_heads, headdim**) 比 (**batch_size, num_heads, N, headdim**) 更快是不正确的；原始数据应理解为 (**batch_size, N, d**)，其中 **num_heads** 从 **d** 中拆分出多头。
   - 早期的方法是在 **TP 层** 进行并行化，将 (**N, headdim**) 放置在每个 GPU 上；从每个 GPU 的计算角度来看，(**batch_size, num_heads, N, headdim**) 更方便，因为 GPU 处理的 **headdim** 是连续的，因此根据 **静态 split-k** 的思想，这不会影响性能。
- **澄清 GPU 并行中的数据处理**：在 GPU 并行中，原始数据被解释为 (**batch_size, N, d**)，其中 **num_heads** 从 **d** 中拆分出多头，最初用于张量并行（TP）。
   - **静态 split-k** 方法将 (**N, headdim**) 放置在每个 GPU 上，使得 (**batch_size, num_heads, N, headdim**) 对 GPU 计算更高效，因为 **headdim** 是连续的，不会影响性能。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1364314972305494161)** (32 条消息🔥): 

> `A100 Grayscale, AMD MI300 FP8, AMD MI300 Identity, L4 Grayscale, H100 Grayscale` 


- **A100 的 Grayscale 性能提升**：一名成员在 **A100** 上以 **2.51 ms** 的成绩取得了 `grayscale` 排行榜的**第 4 名**。
   - 随后的提交在 **A100** 上达到了 **2.50 ms**。
- **MI300 在 FP8-MM 上竞争升温**：多名成员在 **MI300** 的 `amd-fp8-mm` 排行榜上提交了成功的运行结果，时间从 **245 µs** 到 **9.63 ms** 不等。
   - 一名成员以 **245 µs** 的成绩获得**第 2 名**，另一名成员以 **262 µs** 的成绩获得**第 3 名**。
- **MI300 的 Identity 危机解决！**：一名成员在 **MI300** 上以 **7.69 µs** 的成绩获得了 `amd-identity` 排行榜的**第 1 名**。
   - 另一名成员以 **22.6 µs** 获得**第 10 名**。
- **L4 在 Grayscale 中取得领先**：一名成员使用 **L4** 以 **16.2 ms** 的成绩获得了 `grayscale` 排行榜的**第 1 名**。
   - T4 的运行速度为 **16.2 ms**。
- **H100 Grayscale 个人最佳成绩**：一名成员在 **H100** 上以 **1414 µs** 的成绩创下了 `grayscale` 排行榜的个人最佳记录。
   - 随后在 **H100** 上达到了 **1409 µs**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1364647936386142258)** (11 条消息🔥): 

> `AMD, Code Server Access, Leaderboard, Profiling` 


- **AMD 奖励排行榜前 8 名！**：**AMD** 将向排行榜前 8 名的用户提供 **code server access**。
   - 访问权限将随着排名的重大变化而轮换，如果你想加入，请私信 <@1151929379492991116>。
- **在 Code Server 中进行 Profiling**：一位成员提到在该系统中 Profiling 无法正常工作。
   - 另一位成员跟进说：*“更新：我刚才太蠢了，没用对语法”*。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1364327105252298862)** (63 条消息🔥🔥): 

> `HIP vs Inline, Registration Confirmation Delay, AMD Employee Leaderboard Visibility, Submission File Limitations, Numpy Error` 


- ****HIP vs Inline：代码性能对决****：成员们讨论了在 HIP 中运行代码是使用 **hip-py** 还是 **inline functions with Torch**，由于熟悉度原因，最初更倾向于使用内联函数。
   - 一位成员询问如何编译内联代码，并被引导至 `/leaderboard template` 命令。
- ****注册确认陷入停滞****：参赛者们对报名参加比赛后延迟收到注册确认邮件感到困惑。
   - 一位成员表示，注册确认对于获得奖金认可至关重要，但在确认之前你可以开始发送 kernel。
- ****AMD 员工排行榜：设定高预期****：一位参赛者建议排行榜应明确标识 **AMD 员工**，以便衡量现实的竞争水平。
   - 其动机是希望能更有信心看到自己有机会登上排行榜顶端。
- ****提交系统：一个文件统治一切****：关于排行榜是否允许从 pypi 安装 Python 库（torch extension, pybind11 library）存在疑问。
   - 确认**鼓励仅提交单个文件**，并可以像一位成员所说的那样，在提交文件本身通过 pip 安装包：*是的，你可以从提交文件中通过 pip 安装包，但仅支持单个文件*。
- ****Numpy 错误袭击 Benchmark 代码****：一位参赛者在进行 Benchmark 时遇到了 `ModuleNotFoundError: No module named 'numpy'` 错误，尽管他们的代码中没有直接调用 numpy，但错误追踪出现在了 bot 代码中。
   - 该问题归因于损坏的 Docker 镜像，目前已修复。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1364395537880580126)** (3 messages): 

> `Modal.com credits, CUDA fp6 type, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B format` 


- **Modal.com 赠送免费额度**：[Modal.com](https://modal.com/) 每月提供 **$30 免费额度**并按秒计费，非常适合测试 kernel。
   - 由于其按秒计费的模式，它被推荐作为一个实验平台。
- **CUDA 奇特的 fp6 类型**：一位成员询问了 **CUDA fp6 类型**的支持情况，以及由于其不能被 8 或 4 整除而可能导致内存碎片的潜在问题。
   - 另一位成员表示 **fp6 支持确实很奇怪**，并指出其填充（padding）要求使其在 gmem、smem 或 tmem 的空间节省方面并不优于 **fp8**。
- **CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B 压缩？**：一位成员推测 `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` 格式（将其填充到连续的 4 字节中）可能更适合 **Inline Compression (ILC)**，从而可能降低 **HBM 带宽**。
   - 他们指出这需要使用 **virtual memory API**，且不是默认功能。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1364314824481439854)** (149 messages🔥🔥): 

> `o4-mini errors, keybindings breaking, Gemini and Claude combos, Cursor slowdowns, Windsurf vs Cursor` 


- **o4-mini 报错，但请求仍能通过**：用户报告在 **o4-mini** 上收到错误消息，但请求似乎仍被成功处理。
   - 一些用户表示在更新 Cursor 后遇到了这个问题，而另一些用户则报告在没有近期更新的情况下也发生了该问题。
- **Cursor 更新导致快捷键失效**：几位用户报告更新 Cursor 会破坏他们的快捷键（keybindings）。
   - 虽然没有提到具体的解决方案，但用户们确认都遇到了同样的问题。
- **Gemini 与 Claude 组合：3.7 最容易添加你没要求的内容**：一位用户发现使用 Google Gemini 进行规划并结合 Claude 3.7 进行开发会导致意外的添加，且难以修复 bug。
   - 另一位成员建议使用 **Gemini 2.5** 而非 3.7 进行规划，因为 3.7 更容易添加用户没有要求的内容。
- **Cursor 变得慢到无法使用**：几位用户报告 Cursor 最近变得慢到无法使用，尤其是在请求缓慢的情况下。
   - 有人建议这种变慢可能是故意的，目的是鼓励用户转向付费计划；重启 Cursor 或检查 VPN/代理设置可能有助于缓解该问题，并附带了 [Reddit 帖子](https://www.reddit.com/r/cursor/s/qnmPu2N59m)链接。
- **Windsurf 势头强劲，Cursor 粉丝发表看法**：成员们讨论了 Windsurf 与 Cursor 的优劣，有人认为 Windsurf 更便宜，而另一些人则坚持认为 Cursor 拥有更好的 UI/UX 且更具创新性。
   - 一位尝试过 Windsurf 的成员发现其 Tab 功能在预测方面*比预想的要好*，另一位成员链接了一条关于 [Windsurf](https://x.com/heyrobinai/status/1914829284004471099?s=46&t=kUuVqsG2GMX14zvB592G5w) 的推文。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1364659731347800180)** (1 messages): 

> `GPT Image 1, Image Generation API` 


- **GPT-Image-1 发布**：OpenAI 发布了 **gpt-image-1**，让全球开发者都能使用 ChatGPT 强大的**图像生成能力**。
   - 核心特性包括**更准确、高保真的图像**、**多样化的视觉风格**、**精准的图像编辑**、**丰富的世界知识**以及**一致的文本渲染**。
- **图像生成 API 指南**：OpenAI 发布了一份指南，帮助开发者开始使用全新的**图像生成 API** 进行构建，该 API 允许使用 **gpt-image-1** 模型创建图像。
   - 更多信息请查看[指南](https://platform.openai.com/docs/guides/image-generation)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1364323859511250944)** (97 条消息🔥🔥): 

> `Gemini 2.5 Pro vs Gemini 2.5 Flash, AI 取代工作, Sora 预计发布时间, o3 在高中几何方面表现不佳, ChatGPT App vs Web 端` 


- **Gemini 2.5 Pro vs Flash：谁更胜一筹？**: 成员们正在思考 **Gemini 2.5 Pro** 还是 **Gemini 2.5 Flash** 是更优的模型。
   - 最终，有人插话说 *你应该使用所有的 AI 模型以获得最佳结果*。
- **AI 引发失业潮：事实还是虚构？**: 关于 **AI 是否会取代所有人的工作** 这一老生常谈的问题再次被提出。
   - 一位用户回应说 *运动员目前是最安全的*，因为他们还没见过 **AI** 能打顶级板球或足球。
- **Sora 视频生成对新用户“隐身”！**: 用户报告称，**ChatGPT Plus** 上的 **新账号暂时禁用了视频生成功能**。
   - 已确认这是 *针对新用户的有意为之*。
- **几何体操：LLM 能稳住吗？**: 看来 **o3** 在 **高中几何** 方面表现挣扎，**o4 mini high** 和 **Gemini 2.5 Pro** 也是如此。
   - 然而，有人指出 **Deepseek** 正确解决了一个特定的 SAT 几何问题，而其他模型则没有。
- **App vs Web：哪个 ChatGPT 更强？**: 一位用户表示 *说实话，**ChatGPT App** 比 **Web 端** 好用得多*，并补充说他们使用 **API**。
   - 提供的截图显示，**ChatGPT o4-mini-high** 在解决数学问题时表现参差不齐，虽然答错了，但在被要求检查答案时自行纠正了。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1364323600298934345)** (7 条消息): 

> `AI 模型错误, Plus 计划的聊天与记忆, GPT Image 1` 


- **AI 模型经常在事实性问题上出错**: 一位成员指出 *每一个 AI 模型都可能犯错*，敦促其他人不要再重复同样的问题。
   - 他抱怨说这个问题已经被回答了 *大约 50 次*。
- **Plus 计划取消后的权益**: 一位成员询问在取消 **Plus 计划** 订阅后，保存的记忆和聊天记录是否还可以访问。
   - 另一位成员建议，虽然专属模型可能无法访问，但 **聊天记录可以转移到免费账号的 4o**，或者直接粘贴到免费模型中。
- **Image 1 的图像生成启动**: 一位成员简单地问道 *谁能使用 gpt image 1 了？*
   - 未收到任何回复。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1364485847356936253)** (7 条消息): 

> `AI 小说写作助手, AI 税务辅助, AI Python 编程助手, 定义有趣的故事提示词` 


- **AI 助手根据预期任务调整提示词**: 为 AI 助手编写提示词的方式会根据用户希望 AI 输出的内容而变化，例如帮助用户 **撰写虚构小说**、**处理税务** 或 **学习 Python 编程**。
- **AI 创建有趣且现实的故事提示词**: 一位用户希望 AI 助手能创建跨越各种流派的 **有趣且现实** 的故事提示词。
   - 另一位用户回应指出，需要为 AI 明确定义 **“有趣的故事提示词”**，并要求初始用户定义 *他们* 认为有趣的故事是什么样的。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1364485847356936253)** (7 条消息): 

> `AI 故事提示词, 定义“有趣”的提示词, 跨流派的现实故事` 


- **AI 助手创建有趣的故事提示词**: 一位成员询问如何更改常用提示词以获得更好的结果，并指明他们希望 **AI 助手** 创建 **有趣的故事提示词**。
   - 另一位成员建议，*准确了解你希望 AI 输出什么* 是改进提示词的第一步。
- **定义有趣的故事提示词**: 一位成员强调需要充分 **定义“有趣的故事提示词”**，以便描述并请求所需的提示词，同时避免不需要的提示词。
   - 他们询问发帖者是否可以 **描述他们认为有趣的故事提示词**，以便将其与无趣的提示词区分开来。
- **跨流派的现实故事**: 发帖者澄清说，他们想要一个能讲述各种流派的 **有趣故事**，而且故事当然要是 **现实的**。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1364339828350783569)** (14 messages🔥): 

> `使用 Anki 进行 NLM 备考、NLM 与客户测试结果、NLM 德语提示词、NotebookLM 数据训练、NotebookLM 长概览` 


- ****Anki 高手**：用户使用 NLM 备考**：一位用户正利用 **Notebook LM** 进行备考，通过上传 **Anki 卡片**并请求提供上下文，在此过程中还寻求教科书推荐。
- ****患者模式**：倡导者使用 NLM 分析测试结果**：一位患者倡导者使用 **Notebook LM** 分析客户的测试结果，寻找模式并扩展用于提供者随访的问题列表。
   - 他们利用对症状或诊断（dx）的描述进行搜索，并超越眼前的考量进行扩展，同时参考了鉴别诊断。
- ****Sprich Schnell**：破解 NLM 的德语提示词**：一位用户询问如何引导 **Notebook LM** 使用德语交流，寻求撰写有效提示词的技巧。
- ****隐私付费墙**：NotebookLM 数据训练困境？**：一位用户质疑 **Notebook LM** 是否在用户数据上进行训练，并回想起有关付费订阅提供隐私优惠的信息，链接指向 [Google 支持页面](https://support.google.com/notebooklm/answer/15724963)。
- ****概览过载**：在 NLM 上获取长篇总结**：一位用户询问如何使用 **Notebook LM** 获取冗长且详细的概览。
   - 另一位用户澄清说，概览的长度取决于所使用的源文件，并建议使用“完整报告”作为管理长输出的来源。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1364340916088733807)** (75 messages🔥🔥): 

> `NotebookLM 数学支持、Gemini 2.5 Pro、音频概览语言支持、NotebookLM 中的 PDF 处理、AI 模型中的 Grounding 和搜索` 


- **NotebookLM 在数学和图像方面表现挣扎，但希望就在眼前！**：用户报告 **NotebookLM** 在处理**数学符号**和**图像加载**方面存在困难，认为它在处理公式方面落后于 **GPT-4**，但团队正在努力修复。
   - 一位用户指出出现了奇怪的符号而不是 LaTeX 符号 ，另一位用户提到它处理数学符号的能力可能不如 GPT；团队已意识到这一点并**正在处理中**。
- **Gemini 2.5 Pro 在推理任务中胜过 NotebookLM！**：一位用户将 **Gemini 2.5 Pro** 与 **NotebookLM** 进行了对比，发现 **Gemini 2.5 Pro** 远优于 **ChatGPT o3** 或 **o4-mini**。
   - 另一位用户分享说，给 **NotebookLM** 提供逻辑和数学推理方面的书籍和材料后，它仍无法解决一个被 **Gemini 2.5 Pro** 轻松解决的逻辑谜题。
- **音频概览卡在英语——*Pronto en Español*（很快会有西班牙语吗）？**：一位用户询问 **NotebookLM** 的音频总结功能是否可以生成西班牙语播客。
   - 回复是“目前还不支持”，表明语言支持目前有限，但未来可能会改进。
- **NotebookLM 处理大型 PDF 较吃力，建议拆分！**：用户遇到 **NotebookLM** 在处理长篇 PDF 文档时中途停止的问题。
   - 建议的解决方法是使用 [iLovePDF](https://www.ilovepdf.com/pt/dividir_pdf) 等工具将 PDF 拆分为较小的片段。
- **Grounding 模型，而非模型在 Grounding！**：成员们澄清说，**AI 模型**本身并不搜索网页，但可以通过访问搜索 API（如 **Google Search**）进行 **grounded**，以更新知识或验证信息。
   - 据解释，**AI Studio** 等工具允许开发者为 **Gemini 2.5 Pro** 等模型启用搜索能力（**grounding**），附带的屏幕截图示例见[此处](https://cdn.discordapp.com/attachments/1124402182909857966/1364684264272171129/Screenshot_2025-04-23_at_21.25.58.png)。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1364346218381312100)** (67 messages🔥🔥): 

> `Agents vs Humans, Hugging Face Spaces Issue, Llama 3 Chat Template, Fine-tuning Llama for VTuber, Continuous Pretraining Datasets` 


- **Agent 陷入困境，人类大放异彩！**：一位成员认为 AI **Agents** 被过度炒作且低效，因为 *在 99.9% 的时间里，人类更便宜、更快且真正可靠*，并指出了动态生成的工作流和级联复合错误等问题。
   - 另一位成员提供免费额度邀请其尝试自己的 Agent 系统以获取反馈，但第一位成员拒绝了，表示仅对 Agent 在音频研究方面的应用感兴趣。
- **Hugging Face Space 出现故障，基础架构团队受命处理！**：用户报告称他们的 **Hugging Face Spaces** 突然离线，详见[此讨论](https://discuss.huggingface.co/t/my-space-suddenly-went-offline-the-cpu-cannot-restart/151121/22)，基础架构团队正在努力解决根本原因。
   - 团队确认问题已解决，重启 Space 即可恢复正常。
- **Llama 3 失去指引！**：一位成员在 Windows PC 上使用 **Llama 3** 输出时遇到问题，怀疑是 Chat Template 或版本不匹配的问题。
   - 另一位成员指出使用了错误的 Chat Template，并指导该成员使用格式 `{'role': 'user' , 'content': message }`。
- **VTuber 高手感到困惑！**：一位成员询问关于为 **VTuber** 文本生成而 **Fine-tuning** **Llama** 的事宜，寻求流程指导。
   - 另一位成员建议，对于 VTuber 内容，仅靠 **Prompting** 通常就足够了，并建议尝试创建合成数据。
- **预训练性能困境！**：一位成员在观察到使用 **cosmopedis-v2** 和 **fineweb-edu-dedup** 数据集对 **smolLM** 进行 **Continuous Pretraining** 后，在 hellaswag 基准测试中性能下降，随后寻求数据集推荐。
   - 他们以 **128 的 Batch Size** 和 **512 的 Max Seq Length** 训练了 **5k Steps**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1364620793220305026)** (1 messages): 

> `Pomodoro Technique, Time management` 


- **番茄工作法（Pomodoro Technique）就像健身组数**：一位成员将 **Pomodoro Technique** 比作健身中的重复组数。
   - 他们分享说，使用这种方法有助于完成*更多的产出*。
- **提高生产力的时间管理**：用户分享说 **Pomodoro Technique** 非常有效。
   - 它可以显著提高产出。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1364587309944668191)** (3 messages): 

> `Model Size Disclosure, YouTube Channel for Fine-tuning Tutorials` 


- **预训练计算量得到澄清**：一位成员澄清说，他们之前关于*模型大小*的评论不准确，他们原本是指预训练所消耗的**计算量（Compute Amount）**。
   - 这一澄清是在关于模型大小是否通常被披露的争论之后提出的。
- **ML 爱好者推广 YouTube 频道**：一位 ML 爱好者推广了他们的 **YouTube 频道** *Let's Fine-tune Everything*，该频道提供关于为真实用例 **Fine-tuning** 开源模型的实操教程和指南。
   - 该频道涵盖了从 **Object Detection** 到 **LLMs** 的主题，面向初学者和经验丰富的从业者。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1364315359435292892)** (5 messages): 

> `AI-Powered Document Q&A Project, LLM Fundamentals for Cybersecurity, Resume Matching App` 


- **文档问答（Document Q&A）项目开源**：一位成员开源了一个专注于 **AI 驱动的文档问答**的小项目，该项目使用 **FastAPI** 后端，并采用基于 Embedding 模型的检索方法。
   - 开发者正在寻求关于**架构、代码质量、可扩展性**的反馈以及改进建议，更多信息请参见 [Repo 和帖子](https://www.linkedin.com/posts/hamzabouajila_ai-opensource-python-activity-7320494631471771648-z9wE)。
- **网络安全遇上 LLM 基础知识**：一位成员分享了一篇[博客文章](https://x.com/dazzyddos/status/1914895119675007252)，为网络安全专业人士剖析了 **AI/LLM 基础知识**，重点关注 **Prompt injection** 发生的原因。
   - 目标是解释 *为什么* 会发生 Prompt injection，而不仅仅是详细说明它 *是什么* 或 *如何* 运作。
- **简历匹配应用上线**：一位成员宣布**简历匹配应用**现已在 [match-your-resume.fyi](https://match-your-resume.fyi/) 上线。
   - 这是对之前分享的与该项目相关的 [GitHub 仓库](https://github.com/waefrebeorn/bytropix)的后续跟进。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1364558627452620832)** (1 条消息): 

> `Embedding Models for Short Contexts, QA Embedding Pairs, Context Length Optimization` 


- **短 Token Chunk 的 Embedding 模型：探索开始**：一位成员正在寻求针对 **<=1024 tokens** 的 chunk 大小和 **<512 tokens** 的 QA 对优化的 Embedding 模型建议。
   - 用户质疑了在 **8k context** 窗口上训练的模型的适用性，认为它们对于短上下文可能不是理想选择。
- **QA Embedding 对：深度探讨**：讨论集中在如何选择合适的 Embedding 模型，以高效处理问答（Question-Answer）Embedding 对。
   - 讨论考虑了不同模型在同时对问题及其对应答案进行编码时的表现，特别是在给定的 token 限制范围内。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1364337810550820965)** (8 条消息🔥): 

> `Arxiv paper deadline extension, Loops and Branches Notebook Bug, Agents course joining, HuggingFace course credit issue` 


- **Arxiv 截稿日期延长至 7 月**：一篇 [Arxiv](https://arxiv.org/pdf/2311.12983) 论文的截稿日期已延长至 **7 月 1 日**。
- **Notebook 中发现循环 Bug**：一位用户在 [Loops and Branches Notebook](https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/workflows.ipynb) 中发现了一个 Bug，与 **LoopEvent** 参数被错误地放置在 **Step_two** 而非 **step_one** 有关。
- **Agents 课程：提供报名指南**：一位用户询问如何加入 Agents 课程，另一位用户引导其查看[指南](https://huggingface.co/learn/agents-course/en/unit0/introduction)。
- **点数不足导致 HuggingFace 课程问题**：一位用户报告在学习 **Agents 课程** 时遇到错误，并怀疑是由于免费账户或 **Google Colab** 配置问题导致的点数不足。
   - 他们附上了一个 [错误日志](https://cdn.discordapp.com/attachments/1329142738440028273/1364693372442509393/errorUseInferenceClient.txt?ex=680a993e&is=680947be&hm=18162df5d939e62e881916750257fcb12e6b7ee6fd31a90b3d412ba650809f2b&) 用于调试。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1364326801786273813)** (72 条消息🔥🔥): 

> `大脑处理的局部性，周六论文讨论录音，Anthropic 的最新论文，Hebbian 理论 vs 大脑物理学，心理模型 vs 世界模型` 


- ****大脑的局部处理****：围绕大脑处理主要具有**局部性（local）**的讨论，即神经元接收来自直接连接的邻居的信号，并具有受其位置、连接性和上下文影响的局部内部过程。
   - 一位成员表示，“神经元具有局部动力学，这意味着每个神经元对其自身以及与其连接的邻近神经元具有局部视野。此外，这不仅仅是顺序的”，而另一位成员则提出了**细胞骨架微管中的量子非局部信息过程**。
- ****论文讨论未录制****：一位成员询问了周六论文讨论的录音，特别是希望能听到关于 **Anthropic 最近论文**的讨论。
   - 另一位成员回应说，录制时从没有任何预警，而另一位成员则指出了 [Yannic 关于 Anthropic 论文的视频](https://www.youtube.com/watch?v=mU3g2YPKlsA)，提问者表示已经看过了。
- ****心理模型 vs 世界模型的辩论****：讨论探讨了**心理模型（mental models）**（针对特定事物的内部模拟或预期）和**世界模型（world models）**（关于世界运行方式的更广泛、长期的表征）的概念。
   - 一位成员建议大脑构建心理模型来预测并与现实进行比较，并根据经验进行完善，使用了来自 [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)（自由能原理）和 [Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding)（预测编码）的概念。
- ****Transformer 技巧与 Self-Attention 策略****：一位成员询问在 forward pass 中使用 *x = self.transformer(x, x)* 是否合理，作为一种加入 *一些 transformers 哈哈* 来让他们的 *ai 更好* 的策略。
   - 其他人解释说，当需要 **self-attention** 时经常会这样做，并且可以作为一种技巧或偏置（bias），并链接了一篇 [IBM 关于 self-attention 的文章](https://www.ibm.com/think/topics/self-attention)，同时建议考虑 *“ϵ-greedy”探索*，指出了 [随机性和探索](https://spectrum.ieee.org/2d-semiconductors-molybdenum-disulfide) 的价值。
- ****抛弃 PyTorch，渴望优化****：一位成员对 **PyTorch 的接口**表示不满，称其为*如此不虔诚且不神圣的接口*，并询问是否有替代方案。
   - 他们提到喜欢 **Unsloth** 的易用性，但想知道如何迁移优化，引发了一个滑稽的调侃，称其性能可以归功于*开发团队的古柯碱预算*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1364414011508260874)** (3 条消息): 

> `Muon, Adam 替代方案, 逆向蒸馏` 


- **Muon 替代 Adam**：社区将在周三讨论 **Muon**，它是 **Adam** 的一种更快的替代方案。
   - 频道中还分享了一篇关于 [Muon](https://kellerjordan.github.io/posts/muon/) 的博客文章。还提到了关于 WaveFunction 的 [ArXiv 链接](https://arxiv.org/abs/2310.11453)。
- **逆向蒸馏？**：一位成员询问 **Muon** 是否是一种逆向蒸馏方法。
   - 有人请求创建一个活动来进一步讨论 **Muon** 论文。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1364327168569512028)** (3 条消息): 

> `YouTube 视频链接` 


- **链接的 YouTube 视频引发讨论**：几位成员发布了 YouTube 视频链接，包括 [7GF78YQz62w](https://youtu.be/7GF78YQz62w)、[K9anz4aB0S0](https://youtu.be/K9anz4aB0S0) 和 [1W_mSOS1Qts](https://youtu.be/1W_mSOS1Qts)。
   - 在没有更多上下文的情况下，很难确定这些视频的具体主题或相关性。
- **更多 YouTube 链接等待上下文**：分享了更多 YouTube 链接，但由于没有随附的讨论，它们的具体相关性仍不清楚。
   - 需要进一步的信息来确定这些视频涵盖的主题及其与正在进行的对话的联系。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1364323720818200697)** (8 messages🔥): 

> `Zed 项目诊断, Modular 见面会, MAX/MOJO 许可证` 


- **Zed 的 Project Diagnostics 功能受到好评**：成员们讨论了 Zed 的 **Project Diagnostics** 功能（可通过 **⇧⌘M** 或点击左下角的错误图标访问），用于快速识别并就地编辑错误。
   - 一位成员表示，*能够快速进行更改并看到未解决的错误/警告计数降至零，既方便又充满动力*。
- **Modular 见面会即将举行**：一位成员宣布在他们位于 Los Altos 的办公室举办 [Modular Meetup](https://lu.ma/modular-meetup)，现场名额有限。
   - 演讲也将通过 [YouTube](https://www.youtube.com/watch?v=uul6hZ5NXC8) 和 [LinkedIn](https://www.linkedin.com/events/next-gengpuprogramming-hands-on7319044981682270210/) 进行直播。
- **MAX/MOJO 许可证受到质疑**：一位成员从商业角度对 **MAX/MOJO** 许可证提出了疑问，特别是关于“在其他加速器上的生产/商业用途”。
   - 他们推测这是否是为在非 NVIDIA GPU 上开发 **MAX/MOJO** 而收集反馈的一种策略。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1364316145871618088)** (43 messages🔥): 

> `Mojo 训练流水线, Mechanical Migrator 工具, Pythonic Mojo 设计权衡, Mojo 中的零成本抽象, Mojo 中的枚举` 


- **社区寻求训练流水线示例**：尽管 **Mojo** 尚未原生支持训练，但成员们对在训练流水线中使用 **Mojo** 很感兴趣，特别是在通过 **PyTorch** 调用训练代码之前的预先数据处理阶段。
   - 社区想知道是否有成员发布了利用 **Mojo** 的训练流水线示例，即使它还处于早期阶段。
- **Mechanical Migrator 强制要求与编译器测试**：讨论指出，由于编译器测试非常敏感，Mechanical Migrator 工具可能是必不可少的；即使只添加一个关键字也可能导致测试失败。
   - 一位成员选择通过脚本解决，并参考了 GitHub 上的一个 [脚本](https://github.com/bgreni/Kelvin/blob/main/scripts/run_reject_tests.py)。
- **Pythonic Mojo 辩论特性权衡**：关于 **Pythonic Mojo** 的设计权衡正在进行讨论，旨在平衡动态特性与接近 **Go** 或 **Rust** 的性能。
   - 担忧在于，包含过多的动态特性是否会损害 **Pythonic Mojo** 的速度。
- **思考零成本抽象 (Zero-Cost Abstraction)**：成员们辩论了动态性是否能保持零成本抽象，并参考了 **Swift** 作为管理复杂性和动态性的先例。
   - 有人指出，*零成本并不是指没有开销，而是指不对你没有使用的功能产生开销*。
- **社区寻找枚举替代方案**：由于 **Mojo** 缺乏专用的枚举 (Enums)，成员们探索了使用 **DType**（类似于枚举）或 **utils.Variant** 来实现联合类型 (Unions)。
   - 分享了 [DType 实现](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/dtype.mojo) 的链接作为参考。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1364330928507326514)** (39 messages🔥): 

> `Brainy RTX 4090 AI 超级计算机, API 中的 OAI 图像生成, Tinybox 竞争对手, Scout.new 表现出色` 


- **Autonomous.ai 推出 Brainy，RTX 4090 AI 超级计算机**：[Autonomous.ai](https://www.autonomous.ai/robots/brainy) 发布了 **Brainy**，这是一款搭载 **RTX 4090 的 AI 超级计算机**，具有令人印象深刻的 O3 Agent UX，可进行图像分析拆解。
- **Scout.new 太火了！**：一位成员分享了 **Scout.new**，但指出由于负载过高它挂掉了，其他人评价它 *简直太酷了 (fucking cooking hot damn)*。
   - 一位成员发布了 **Ray Fernando** 帖子的 [X cancel](https://xcancel.com/rayfernando1337/status/1914791594789879844) 链接。
- **OpenAI 图像生成现已在 API 中可用**：**OpenAI** 已在其 [API](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1) 中发布了使用 **gpt-image-1** 的**图像生成**功能。
- **微软宣布 Copilot Agents**：**微软**宣布了 [Copilot Agents](https://x.com/satyanadella/status/1915098359251247392)。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 新的 Lightning Pod https://youtu.be/aDiEQngFsFU
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1364374336181637189)** (2 条消息): 

> `LlamaIndex Milvus full-text search, Agentic Document Workflow` 


- **LlamaIndex 新增 Milvus 全文搜索**：LlamaIndex 与 [@milvusio](https://github.com/milvus-io) 的集成现在支持 **BM25 全文搜索**，用于 **RAG pipelines** 中的混合搜索，结合了向量搜索和关键词匹配。
   - 点击[此处](https://t.co/0dCi0kEn6o)查看集成教程。
- **Agentic Document Workflow 发布**：与 **RAG chatbot** 相比，**Agentic Document Workflow (ADW)** 具有更好的扩展性，能与现有软件生态系统集成，并提供更好的错误处理。
   - ADW 被描述为超越 **RAG chatbot** 原型的逻辑下一步；更多详情见[此处](https://t.co/ZZzr7scHhF)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1364328796060254338)** (35 条消息🔥): 

> `LlamaParse getText() issue, Document hash computation, Passing userID to MCP tools, MLflow autolog with Llamaindex and FastAPI, Workflows Checkpoints usage and alteration` 


- **LlamaParse 的文本之争：Markdown vs. Text**：一位成员在使用 **LlamaParse** 时遇到问题，在 next.js 中，当 `resultType` 设置为 `markdown` 时，`getText()` 返回部分内容，但设置为 `text` 时工作正常。
   - 这被确定为 **markdown 与 text 的对比问题**，切换到 `const reader = new LlamaParseReader({ resultType: "text" });` 解决了该问题。
- **Document 哈希色彩：元数据排除**：一位成员询问如何从 document 哈希计算中排除一个元数据键（特别是时间戳），以避免在时间戳更改时出现去重问题。
   - 澄清指出，虽然 `TextNode` 在哈希计算中会考虑元数据，但 `Document` 对象（用于 ingestion pipeline）则不会，这允许成员安全地向每个 document 对象的元数据添加时间戳。
- **MCP Tool 的 userID 注入挑战**：一位成员寻求一种方法，在使用 **React agent** 的 agent workflow 中，高效地将 `userID` 传递给注册在 **MCP server** 上的 tools。
   - 将 `userID` 附加到用户查询中是一个可行的变通方案，但需要一个更简洁、更标准化的解决方案。
- **MLflow 在 FastAPI 并行宇宙中的 Autolog 历险记**：一位成员在 **FastAPI 后台任务** 中并行运行 **LlamaIndex Workflow** 时，遇到了 **MLflow autolog** 捕获 **LLM call traces** 的问题。
   - Traces 捕获不一致，导致警告：*'NoneType' object has no attribute 'info'*，这表明是一个 **MLflow 特有的问题**。
- **Workflow Checkpoint 难题：事后修改 Events**：一位成员询问了修改 **Checkpoint** 的推荐方法，以便从给定的 Checkpoint 启动 **Workflow**，但要将修改后的 **Event** 传递到 Checkpoint 的步骤中。
   - 推荐的方法是重构 workflow，加入 **human-in-the-loop 步骤** 来批准或更改关键输出，从而避免修改现有状态；直接修改 Checkpoints *不是* 一项受支持的功能。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1364678759910998156)** (3 条消息): 

> `Instruction-Finetuning LLMs, TRL for Finetuning, Memory Constraints for LLMs` 


- **推荐使用 TRL 进行指令微调**：一位成员推荐使用 **TRL (Transformers Reinforcement Learning)**，并链接到了 [Hugging Face TRL 文档](https://huggingface.co/docs/trl/en/index)，用于对开源 LLM 进行指令微调，而不是使用 LlamaIndex 工具。
   - 他们建议使用任何现有的 LLM 创建数据集，以便将训练蒸馏到另一个模型中。
- **训练 LLMs 时的内存限制**：一位成员警告说，在本地或 **T4 GPU** 上进行训练会受到严重的内存限制，你可能只能在小 batch sizes 下训练非常小的 LLMs。
   - null


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1364355774473703465)** (26 条消息🔥): 

> `MCP Interview, README Translation Automation, AWSLab Cost Analysis MCP Server Issue, MCP Inspector Timeout Error, Cursor MCP Tool Error` 


- **提供 **MCP Interview** 报酬**：一名成员正[支付 **$40** 征求 **30 分钟访谈**](https://discord.com/channels/1119947416775741521/1119947417233033328)，对象是在实际项目中使用过 **Claude Computer Use** 和/或 **OpenAI computer-use-preview** 的开发者，实现过 **MCP** 的开发者可获得额外加分。
   - 该成员有“无数个问题”想了解用户的体验。
- **提议 **README Translation** 自动化**：一名成员提议将所有链接、标签和表情符号存储在单个 **JSON** 文件中，以便通过 **CI pipeline** 自动生成翻译后的 **README**。
   - 这样可以将维护工作集中在一处，只需更新主 **README** 即可节省大量精力。
- ****AWSLab** 成本分析 **MCP Server** 冻结**：一名成员报告称，在使用 **AWSLab cost analysis MCP server** 生成上个月的成本报告时，**Claude Windows 桌面应用**会出现冻结并报错。
   - 尽管网络连接稳定，显示的错误信息仍为：*Claude’s response was interrupted. Please check your network connection or contact support if the issue persists.*
- ****MCP Inspector** 请求超时问题**：一名成员在运行 **GitHub** 文档中的基础 **MCP server** 并使用交互式服务器时，遇到了 **MCP error -32001**：*Request timed out*。
   - 该错误导致他们无法运行任何工具，尽管在 **Claude desktop** 中运行 `mcp install test.py` 时一切正常。
- ****Cursor** 调用 **MCP Tools** 报错**：一名成员报告称，虽然 **Cursor** 能识别所有 **MCP tools**，但在尝试调用它们时会报错，而 **Claude Desktop** 和 **Cline** 则没有这个问题。
   - 该成员正在寻求可能遇到过类似 **Cursor** 问题的其他人的建议。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1364315882150559856)** (4 条消息): 

> `MCP Server, Klavis AI Eval Platform, Browser Extension for MCP, Siloed AI Drag and Drop` 


- **Defang Labs 发布 Vibe-Coded MCP Server**：Defang Labs 构建了一个 MCP server，让你能够**从任何 IDE 直接将你的 vibe-coded 项目部署到云端**，并正在[其 LinkedIn 帖子](https://www.linkedin.com/posts/defanglabs_vibecoding-defang-devtools-activity-7320490826004852737-2IFE?utm_source=share&utm_medium=member_desktop&rcm=ACoAACNoYXgBadWv4CWLbcKhgSGxWjdmu9e5dFI)中征求反馈。
- **Klavis AI 推出定制化 MCP 测试与评估平台 (Eval Platform)**：Klavis AI 宣布其**定制化 MCP 测试与评估平台开启早期访问**，用于测试、评估和比较不同的 MCP server，并邀请用户通过 connect@klavis.ai 联系他们或访问[其网站](https://www.klavis.ai/mcp-testing-eval)。
   - Klavis 表示，目前很难判断哪个 MCP 比其他 MCP 更具备生产就绪性、功能更丰富且更稳定。
- **浏览器扩展通过 MCP 将工具连接到 AI Chat**：一个可定制的 AI 聊天侧边栏现已作为**浏览器扩展**提供，它**通过 MCP 连接各种工具**，可以在 [Chrome Web Store](https://chromewebstore.google.com/detail/browsewiz-ai-assistant-ai/ioohfnlbpolaalcbppaggpgcgpldohfg) 中找到。
- **Siloed AI 在 Web 端支持 MCP 资源的拖放**：[Siloed](https://siloed.ai) 将 **MCP 带到了 Web 端并支持拖放资源**，连接你最喜欢的 MCP server，即可在 Web 端的任何地方粘贴，同时还允许你构建带有动态文本 + 资源附件的 Prompt 库。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1364319640552996936)** (21 messages🔥): 

> `arithmetic shift right op, UPat matching a CONST, multiple patterns matching, instruction ordering and register assignment, closures reconsideration` 


- **算术右移操作 (Arithmetic Shift Right Op) 受到询问**: 一名成员询问系统中是否存在**算术右移操作**。
- **请求立即数 UPat 匹配**: 一名成员询问如何创建 **UPat** 来匹配 **CONST**，其中立即数仅为（例如）**5 位长**，或者低 **n 位为零**。
- **模式匹配优先级澄清**: 当重写器中存在**多个模式匹配**时，模式按**列表顺序**应用，无法优先处理某些模式。
- **用于指令排序的约束求解器后端**: 团队正转向使用**约束求解器后端 (constraint solver backend)**，以共同处理**指令排序 (instruction ordering)**和**寄存器分配 (register assignment)**。
- **重新考虑闭包的提议被搁置**: 在成员提交 **10 个 PR** 之前，团队*不会重新考虑*关于不使用闭包的核心决定。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1364384271116144720)** (6 messages): 

> `Arange Optimization, Indexed Operations for STs, UOps and Buffers relationship` 


- **Arange 被优化掉**: 根据 [此 tinygrad 笔记链接](https://xl0.github.io/tinygrad-notes/arange.html)，提到 `arange()` 会被优化掉。
- **索引操作：查找字节索引**: 一名成员建议通过获取两个 **STs** (ShapeTracker) 的 **indexed_ops**，然后代入张量索引 *i,j,k* 来查找字节索引。
   - 另一名成员引用了 [device.py](https://github.com/tinygrad/tinygrad/blob/6cb2d18c034fc3fb8c8c7521716c04a4674c5504/tinygrad/device.py#L330)。
- **探索 UOps 与 Buffers 的关系**: Issue #10006 尝试描述 **UOps** 与 **Buffers** 之间的关系。
   - 这可能有助于解决悬赏任务：“确保 Buffer 在 CPU 和 VIZ 上被 GC（垃圾回收），并附带良好的测试。”


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dbreunig: https://www.dbreunig.com/2025/04/18/the-wisdom-of-artificial-crowds.html
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1364629921372045342)** (22 messages🔥): 

> `DSPy 3.0, Synthetic Flywheel, Prompt Optimization, Databricks event SFO` 


- **DSPy 3.0 的发布备受期待**: 成员们对 **DSPy 3.0** 表示兴奋，评论如 *“超级期待！！！”*，但一名用户针对某条 [推文](https://x.com/lateinteraction/status/1915058777491145200) 询问 *“我们可以期待什么？？”*。
- **DSPy 3.0 的愿景仍处于保密状态**: 一名成员询问 **DSPy 3.0** 的统一愿景/设计，但被告知统一的愿景/设计尚未公开，因为 *“在发布前，很多内容都属于内部研究！”*，并链接到了 [路线图 (roadmap)](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md)。
- **DSPy 3.0 将于 2025 年 6 月发布**: 当被问及 **DSPy 3.0** 的预计发布时间 (ETA) 时，一名成员回答 *“2025 年 6 月”*。
   - 另一名成员猜测发布活动将在 **旧金山 (SFO) 的 Databricks 活动** 前后。
- **合成飞轮 (Synthetic Flywheel) 即将起飞**: 两名成员讨论了制作一个 *“合成飞轮，它将飞得非常高，甚至需要空域许可。”*
- **DSPy 的提示词优化感觉像黑魔法**: 一位在一年前押注 **DSPy** 进行生成式 AI 开发的用户现在觉得 *“这不是正确的做法”*，因为 *“提示词优化 (Prompt optimization) 看起来有点像黑盒。”*


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1364327634812801044)** (8 messages🔥): 

> `RoPE implementation, Collective scheduling, Tune cp workflow, Library design` 


- **专门的 RoPE 类引发讨论**：一名成员询问为什么 **RoPE** (**Rotary Position Embedding**) 在 torchtune 中被实现为一个专门的类，最初的回答是这比使用函数感觉*更符合 PyTorch 风格 (PyTorch-y)*。
   - 后续解释提到 **RoPE cache** 只需要初始化一次，不需要每次重新计算，因此它需要一些状态，这是用内存换取速度。
- **Collective Scheduling 自定义正在进行中**：一名成员正在测试自定义 **collective scheduling** 的吞吐量和内存占用，并计划在结果理想时提交 PR。
   - 他们正在考虑是保留 `fsdp_delay_all_reduce` 等参数，还是切换到与 **DeepSpeed stages (ZeRO 1-3)** 对齐的单单词描述符。
- **`tune cp` 工作流用户体验**：一名成员详细描述了在 Macbook 上使用 `tune cp` 工作流的经历，强调了诸如需要手动搜索 recipe 和 config 文件、删除文件扩展名以及解决数据集版本不匹配等问题，但在解决 **macOS 特有议题**后最终获得了成功。
   - 该成员还表示，虽然工作流不算太糟，但它严重依赖*大规模代码重复*，感觉不太对劲。
- **混合库设计引发辩论**：讨论围绕 torchtune 中的混合库设计方法展开，该方法旨在提供易于自定义的用户脚本，同时利用库来处理通用组件。
   - 这种方法背后的动机是避免为了自定义几个模块而维护整个库的 fork，目标是允许研究人员只展示对他们重要的代码，但团队正在研究这种*混合设计*究竟是根本性的设计缺陷，还是用户教育/文档问题。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1364431029523189780)** (1 messages): 

> `Future Meeting` 


- **新加坡行程后的通话安排**：一位用户提到他们从新加坡回来后，从下周晚些时候开始可以进行通话。
- **可用性更新**：该用户提供了安排通话的具体时间范围，表示他们将在下周晚些时候从新加坡返回后有空。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1364394089746796595)** (8 messages🔥): 

> `Tool Integrations with SaaS Platforms, New Model Release, SuperNova Models by Arcee-AI` 


- **工程师寻求 SaaS 工具集成指导**：一名成员询问了构建与现有 **SaaS 平台**集成的首选工具，特别是寻求支持为不同客户提供多个连接的 **Zapier** 替代方案。
   - 他们建议将 **Composio** 作为潜在解决方案，并寻求社区对其适用性的意见或替代建议。
- **Nous 预告面向红队的神秘发布**：Nous 暗示即将发布一个专为 **Red Team** 社区量身定制的版本，计划于今天或明天发布，可能带有*新的混合精度量化 (quantlol)*。
   - 该公告引起了对安全和对抗性测试新工具及资源感兴趣的成员的期待。
- **SuperNova 模型赢得社区赞誉**：成员们对 **Arcee-AI** 的 **SuperNova 模型**表示赞赏，理由是其相对于模型规模而言表现强劲。
   - 一位成员指出，自发布以来，这两个 **SuperNova Models** 已成为他们的默认模型。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1364518753815105606)** (3 messages): 

> `Resource Submission Form, Team Name` 


- **寻求资源提交确认**：一名成员询问在提交资源提交表单后是否会收到确认邮件，并寻求特定用户的确认。
   - 该成员表示，尽管填写了表单，但没有收到任何确认提交的邮件。
- **团队名称询问**：一名成员询问另一名成员的团队名称。
   - 第二名成员回答为 *"IbiA"*。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1364584025204981800)** (2 messages): 

> `MOOC readings, LLMAgents-learning.org` 


- **LLM Agents MOOC 阅读材料已发布**：一位成员询问了阅读材料，另一位成员回复称 **LLM Agents MOOC** 的阅读材料已在网站上发布：[llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)。
   - 这些阅读材料据推测与课程内容和作业相关。
- **LLM Agents MOOC 网站**：**LLM Agents MOOC** 的官方网站是 [llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)。
   - 该网站可能包含课程资料、作业和其他重要信息。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1364691798433206422)** (1 messages): 

> `Hugging Face Inference API, Flask Website Integration, Model Deployment` 


- **HF Inference API 连接到 Flask 网站**：上传到 **Hugging Face** 并使用其付费 Inference API 的模型可以连接到使用 **Flask** 构建的网站。
- **Flask 调用 HF 推理端点**：**Flask** 应用程序将带有输入数据的请求发送到 **Hugging Face Inference API** 端点。
   - API 返回模型的预测结果，随后显示在网站上。


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1364691934597218506)** (2 messages): 

> `Hugging Face, Flask, Model uploading` 


- **Flask 提问：Hugging Face 付费推理**：一位成员询问如何将 **Flask** 网站连接到上传在 **Hugging Face** 上并使用其付费 Inference API 的模型。
- **鼓励新成员自我介绍**：鼓励新成员通过分享所属公司/行业/大学、正在研究的内容、喜爱的技术/工具以及希望从社区获得什么来进行自我介绍。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1364416763147583519)** (3 messages): 

> `Debugging Handler Errors, Code Modification Suggestions` 


- **Handler 错误困扰系统**：一位成员报告系统中 *来自你的 handler 的某些内容正在报错*。
   - 另一位成员建议修改 [错误捕获代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286)，改为直接抛出错误，以帮助 **debug 该问题**。
- **分享 Debug 建议**：一位成员建议修改 [代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py#L280-L286) 以抛出错误而不是捕获它们。
   - 他们建议针对 **单个条目** 运行生成，以查看 **完整堆栈追踪 (full trace)**。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1364521994921709588)** (1 messages): 

> `Legislative AI/Tech Webinar, BillTrack50, AI4Legislation competition` 


- **4 月 28 日立法 AI/技术网络研讨会**：企业家 Karen Suhaka（BillTrack50 创始人）正与硅谷华人协会基金会（Silicon Valley Chinese Assocation Foundation）合作，将于 **太平洋时间 4 月 28 日星期一中午 12 点** 举办一场关于 AI 和技术在立法领域应用的网络研讨会。
   - 该研讨会将涵盖构建立法技术、应对伦理考量以及创业技巧；可通过 [此链接](https://forms.gle/v51ngxrWdTsfezHz8) 进行注册。
- **BillTrack50 案例研究**：Karen Suhaka 将分享她对自己法律科技公司 **BillTrack50** 的见解，作为一个从启动到规模化及客户反馈的案例研究。
   - 她将重点关注如何识别需求、选择数据和方法。
- **AI4Legislation 竞赛**：研讨会将包括 **2025 夏季 AI4Legislation 竞赛** 的项目构思，详情可在 [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public/tree/main) 上找到。
   - 该竞赛旨在利用 **LLMs** 和 **NLP** 的最新进展来造福公民和选民。