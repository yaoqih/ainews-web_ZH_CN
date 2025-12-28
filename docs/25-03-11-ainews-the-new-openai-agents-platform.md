---
companies:
- openai
- reka-ai
- hugging-face
- deepseek
- togethercompute
- alibaba
date: '2025-03-12T00:23:17.547385Z'
description: '**OpenAI** 推出了一套全面的 AI 智能体（Agent）新工具，包括 **Responses API**、**网页搜索工具 (Web
  Search Tool)**、**计算机操作工具 (Computer Use Tool)**、**文件搜索工具 (File Search Tool)**，以及一个集成了可观测性工具的开源
  **Agents SDK**，这标志着向“智能体之年”迈出了重要一步。与此同时，**Reka AI** 开源了 **Reka Flash 3**，这是一个拥有
  **210 亿 (21B) 参数的推理模型**，其性能超越了 **o1-mini**，并为其 Nexus 平台提供支持，模型权重已在 **Hugging Face**
  上发布。**OlympicCoder** 系列在竞赛编程基准测试中超越了 **Claude 3.7 Sonnet** 以及许多更大规模的模型。**DeepSeek**
  构建了一个拥有 **3.2 万 (32K) 块 GPU 的集群**，能够在不到一周的时间内训练出 V3 级别的模型，并正在探索 AI 蒸馏技术。**Hugging
  Face** 宣布支持 **Cerebras** 推理，在 **Llama 3.3 70B** 上实现了超过 **2,000 token/秒** 的速度，比领先的
  GPU 快 70 倍。**Reka** 的 **Sonic-2** 语音 AI 模型通过 **Together API** 实现了 **40 毫秒 (ms) 的延迟**。**阿里巴巴的通义千问
  (Qwen Chat)** 增强了其多模态界面，支持高达 **500MB** 的视频理解、语音转文字、访客模式以及扩展的文件上传功能。*Sama*（萨姆·奥特曼）称赞
  OpenAI 的新 API 是“有史以来设计最精良、最实用的 API 之一”。'
id: 599f1341-a28e-45b3-9feb-e507c722162c
models:
- reka-flash-3
- o1-mini
- claude-3-7-sonnet
- llama-3-3-70b
- sonic-2
- qwen-chat
- olympiccoder
original_slug: ainews-the-new-openai-agents-platform
people:
- sama
- reach_vb
title: 全新的 OpenAI 智能体平台
topics:
- ai-agents
- api
- model-releases
- fine-tuning
- reinforcement-learning
- model-training
- model-inference
- multimodality
- voice-synthesis
- gpu-clusters
- model-distillation
- performance-optimization
- open-source
---

<!-- buttondown-editor-mode: plaintext -->**OpenAI 可能就是你所需要的一切。**

> 2025年3月11日至3月12日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务器（**224** 个频道，**2851** 条消息）。预计为你节省阅读时间（以 200wpm 计算）：**258 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在[今天的直播](https://www.youtube.com/watch?v=hciNKcLwSes)中，OpenAI 发布了一系列重大更新，为 Agent 之年做准备：

- [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses)
- [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search)
- [Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use)
- [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search)
- 一个新的开源 [Agents SDK](https://platform.openai.com/docs/guides/agents)，集成了 [Observability Tools](https://platform.openai.com/docs/guides/agents#orchestration)

[Atty Eletti](https://x.com/athyuttamre/status/1899541471532867821) 讲述了设计决策的完整故事，[sama](https://x.com/sama/status/1899579431905305027) 称其为“有史以来设计最精良、最实用的 API 之一”。

你可以在今天发布的[独家 Latent Space 采访](https://www.latent.space/p/openai-agents-platform)中找到更多代码示例和亮点：

https://www.youtube.com/watch?v=QU9QLi1-VvU


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

**1. AI 模型与性能：模型发布、基准测试、特定模型的性能比较**

- **Reka Flash 3，来自 Reka AI 的新型 21B 参数推理模型已开源** [@RekaAILabs](https://twitter.com/RekaAILabs/status/1899481289495031825)，实现了极具竞争力的性能。[@reach_vb](https://twitter.com/reach_vb/status/1899517300576747615) 强调 **Reka Flash 3 采用 Apache 2.0 许可，且击败了 o1-mini**，并质疑为什么它没有走红。Reka AI 进一步详细说明，[Reka Flash 3 为其新的企业智能平台 Nexus 提供支持](https://twitter.com/RekaAILabs/status/1899481289495031825)，并[在合成和公共数据集上进行了微调，随后通过基于模型和基于规则的奖励进行了 RLOO](https://twitter.com/RekaAILabs/status/1899481291889979896)。权重可在 [Hugging Face](https://twitter.com/RekaAILabs/status/1899481291889979896) 上获取。
- **OlympicCoder 是一系列开源推理模型，其表现优于 Claude 3.7 Sonnet 以及比其大 100 多倍的模型**，据 [@_lewtun](https://twitter.com/_lewtun/status/1899574591171035390) 称。该发布包括 **CodeForces-CoTs 数据集**和针对竞赛编程问题的 **IOI'2024 基准测试**。
- **DeepSeek 已经构建了一个 32K GPU 集群，能够在不到一周的时间内训练 V3 级别的模型**，据 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899357311811928152) 称。[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899475671958929453) 指出 **DeepSeek 现在正在讨论 AI 蒸馏 (distillation)**，这是他在 1991 年发表的一个概念，并将其与他早期的工作联系起来。[@cis_female](https://twitter.com/cis_female/status/1899307802423632085) 报告称，在 **3x abacus + two sticks 上以 int0 量化运行 R1，速度达到 30 tokens/s**。
- **Hugging Face Inference 现在支持 Cerebras**，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1899502216420942232) 宣布。据报道，Cerebras Inference 运行 Llama 3.3 70B 等模型时速度超过 **2,000 tokens/s，比领先的 GPU 快 70 倍**。
- **据报道，R1 在新款 M3 Ultra 上的运行速度达到 18t/s，价格约为 9,000 美元**，据 [@reach_vb](https://twitter.com/reach_vb/status/1899480424834899993) 称，这表明高性能推理的可获得性正在提高。
- **Reka 的 Sonic-2 语音 AI 模型现在可通过 Together API 获取**，提供 **40ms 延迟和高保真语音合成**，由 [@togethercompute](https://twitter.com/togethercompute/status/1899498102836380106) 宣布。
- **Qwen Chat 已增强**，具有统一的多模态界面，支持文本、图像和视频，并增强了高达 500MB 的视频理解能力，重新设计了具有语音转文本、访客模式和扩展文件上传容量的移动体验，据 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1899497336889659775) 称。

**2. AI Agent 与开发者工具：专注于构建和使用 AI Agent 的工具、SDK、API 和 Agentic 工作流。**

- **OpenAI 发布了用于构建 AI Agent 的新工具**，包括 **Responses API**、**Web search tool**、**File search tool**、**Computer use tool** 和 **Agents SDK**，正如 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531225468969240)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531367056064814)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531516448768103)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531586950795662)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531682224431614)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531857143972051) 所宣布，并由 [@omarsar0](https://twitter.com/omarsar0/status/1899530784832459043) 和 [@scaling01](https://twitter.com/scaling01/status/1899510452473790537) 进行了总结。**Responses API** 统一了 Chat Completions 和工具使用，实现在单个请求中支持多轮对话 Agent。内置工具包括 **Web Search**（由 GPT-4o 驱动，在 SimpleQA 上达到 90% 的准确率）、**File Search**（支持元数据过滤）以及 **Computer Use**（自动化浏览器和操作系统任务，达到 SOTA 基准测试水平）。**Agents SDK**（开源，在 Swarm 基础上进行了改进）有助于编排具有护栏（guardrails）和可观测性的单 Agent 及多 Agent 工作流。[@sama](https://twitter.com/sama/status/1899579431905305027) 称新 API 是“有史以来设计最精良、最实用的 API 之一”。[@swyx](https://twitter.com/swyx/status/1899517984365752361) 提到了一期 Latent Space Podcast 播客，其中 OpenAI 讨论了这些功能。
- **LangChain 发布了 Agent Chat UI**（一个用于通过聊天与 LangGraph 应用交互的开源 Web 应用）和 **LangGraph-Reflection**（一个预构建的图，用于 Agent 进行自我批判并改进输出），由 [@LangChainAI](https://twitter.com/LangChainAI/status/1899497457459122535) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1899493848843477178) 报道。他们还强调了 **C.H. Robinson 如何通过 LangGraph 和 LangSmith 自动化订单处理，每天节省 600 多个小时**，据 [@LangChainAI](https://twitter.com/LangChainAI/status/1899475651863978410) 称。
- **Weaviate 发布了一个 Transformation Agent**，允许用户不仅可以查询，还可以创建和更新数据库中的数据，由 [@bobvanluijt](https://twitter.com/bobvanluijt/status/1899521133940011476) 宣布。
- **Contextual AI 发布了 Contextual Reranker**，这是一款遵循指令的 SOTA 重排序器（reranker），旨在提高 RAG 流水的精度，并允许对排序优先级进行细粒度控制，由 [@apsdehal](https://twitter.com/apsdehal/status/1899497153132958049) 详细介绍。[@douwekiela](https://twitter.com/douwekiela/status/1899490844572577958) 介绍了一个类似的遵循指令的重排序器，强调了其根据用户指令确定优先级的的能力。
- **Perplexity AI 发布了 Windows 应用**，提供语音听写、键盘快捷键以及访问最新模型的权限，由 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1899498357154107499) 宣布。

**3. AI 应用与行业影响：现实世界应用、行业用例及公司新闻。**

- **Figure AI 正在准备交付数千台人形机器人**，这些机器人由 Helix 神经网络驱动，正如 [@adcock_brett](https://twitter.com/adcock_brett/status/1899544574626070660) 所展示。他认为 [Figure 将成为 AGI 的最终部署载体](https://twitter.com/adcock_brett/status/1899587483928805642)，并且 [在未来，每一个移动的物体都将是一个 AI Agent](https://twitter.com/adcock_brett/status/1899500640755450016)。他们正在 [招聘实习生和全职岗位](https://twitter.com/adcock_brett/status/1899484728157417610)，[@DrJimFan](https://twitter.com/DrJimFan/status/1899509660971131301) 对他们的人形机器人之家表达了兴奋之情。
- **Manus，一个中国的高性能 AI Agent**，在 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1899393214106521633) 的 AI/ML 新闻中被提及。据 [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1899498723010662449) 的报告，Anthropic 的模型据称正在为 Manus 提供动力，它被描述为最新的 AI 轰动点。
- **Zoom 正在利用 AssemblyAI 的 Speech-to-Text 模型**来推进其 AI Companion 的 AI 研发，据 [@AssemblyAI](https://twitter.com/AssemblyAI/status/1899496977169043743) 称。
- **Cartesia 宣布了 A 轮融资**，据 [@_albertgu](https://twitter.com/_albertgu/status/1899499128389877764) 报道。[@saranormous](https://twitter.com/saranormous/status/1899484191256981779) 赞扬了他们的人才密度和创造力，并指出 GPU 资源有所增加。
- **Perplexity AI 正在向网络之外扩张**，在 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1899393214106521633) 的 AI/ML 新闻中被提及。
- **Embra 被介绍为一个完整的 AI OS**，据 [@zachtratar](https://twitter.com/zachtratar/status/1899521673529057293) 称，它可以管理电子邮件、会议、人际关系、撰写邮件、安排日程以及自动化研究。

**4. 中国与 AI 竞争：关注中国的 AI 进展以及与美国的竞争。**

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899594295914496302) 认为 **中国将培养出数百名水平堪比 AI 大师的人才**，并且 **中国 ML 毕业生和项目的质量正在呈指数级增长**，这表明美国的招聘池不足以与之竞争。他还暗示 [中国正秘密地由技术官僚异世界重生小说宅男引导](https://twitter.com/teortaxesTex/status/1899554789114958258)。
- [@dylan522p](https://twitter.com/dylan522p/status/1899361483408150867) 在一个系列中涵盖了硬件基础和历史悠久的机器人公司，强调了 **中国在机器人领域的崛起**。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899486329617993800) 认为 **中国可能会在航天领域超越美国**，因为美国无法建造专门的道路，而中国则专注于航天的规模、工程和物流。他预测 [未来 5 年内中国入轨载荷将出现另一个“曲棍球棒式增长事件”](https://twitter.com/teortaxesTex/status/1899360135941705992)，并指出他们的速度明显更快。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899501654598172768) 将美国的 “Stargate” 方案与 **中国建造 “1000 个 2K GPU 机房”** 进行了对比，质疑中国的技术市场是否比人们认知的 “共产主义集权” 更具竞争力。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899541487919964418) 认为 **西方专注于 “共产主义” 而非 “中国工业党” 是在自讨苦吃**，并暗示他们正在承担西方已经放弃的 “白人的负担”。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899540640351862954) 质疑了 “产能过剩” 的神话，认为在住房、能源、芯片、原材料和汽车等关键领域，“东西越多越好”，这可能与西方的经济观点形成对比。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899334394722107772) 评论了 **中国将 EV 和人形机器人商品化**，将 Elon Musk 的愿景与中国的市场行为进行了对比。

**5. AI 研究与技术：正在讨论的核心 AI 研究概念和技术。**

- **关于通过 Meta Reinforcement Fine-Tuning (MRT) 优化 test-time compute 的新研究**受到了 [@rsalakhu](https://twitter.com/rsalakhu/status/1899597917016744445) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899392429893042485) 的关注。MRT 被介绍为一种新的微调方法，在数学推理方面相比 outcome-reward RL 实现了 **2-3 倍的性能提升和 1.5 倍的 token 效率**，表现优于 outcome-reward RL，并在 1.5B 参数规模上达到了 SOTA 结果。
- **Inductive Moment Matching (IMM)** 是来自 Luma AI 的一类新型生成模型，用于单步或少步采样。据 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899409323651993976) 指出，该模型在 ImageNet-256x256 上使用 8 步推理达到了 1.99 FID，超越了 diffusion models。
- **Effective and Efficient Masked Image Generation Models (eMIGM)** 是一个集成 masked image modeling 和 masked diffusion models 的统一框架。根据 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899411428236259603) 的说法，它的表现优于 VAR，并以更少的 NFE 实现了与最先进的连续 diffusion models 相当的性能。
- **Foundation Models 中的医学幻觉 (Medical Hallucinations)** 在一项新研究中进行了基准测试。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899414464698470507) 报告称，研究发现 **GPT-4o 在需要事实和时间准确性的任务中幻觉倾向最高**，但 **Chain-of-Thought (CoT) 和 Search Augmented Generation 可以降低幻觉率**。
- [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1899491349881352628) 强调了使用 **RLOO (Reinforcement Learning from Objective Optimization)** 进行训练的研究，并指出各实验室探索 PPO 以外算法的热潮。
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899251749690708412) 提到，**可以任意重排 token 位置的 Diffusion language models** 可能是为有界序列长度扩展 test time compute 最强大的方式。
- [@shaneguML](https://twitter.com/shaneguML/status/1899477905132138577) 将 **Chain-of-thoughts 描述为 LLM 的“暗知识” (dark knowledge)**，认为可以通过 prompting 方法实现对模型的更深层理解。
- [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899475671958929453) 讨论了 **AI 蒸馏 (AI distillation)**，引用了他 1991 年的工作并将其与 DeepSeek 的讨论联系起来。
- [@jerryjliu0](https://twitter.com/jerryjliu0/status/1899489337412378909) 对 **MCP (Model-as-Control-Plane) Agent 系统中的版本控制和回归测试** 表示担忧，强调了动态行为变化和 API 更新导致服务中断的潜在问题。
- [@rasbt](https://twitter.com/rasbt/status/1899493072972415154) 发布了一个 **“编写 Attention 机制代码”教程**，讲解了 self-attention、parameterized self-attention、causal self-attention 和 multi-head self-attention。
- [@TimDarcet](https://twitter.com/TimDarcet/status/1899580380958589395) 指出 **高斯混合模型 (GMM) 使用 Expectation-Maximization (EM) 可以快速且良好地拟合 MNIST**，并质疑 EM GMM 是否已经足够。

**6. 梗与幽默**

- [@aidan_mclau](https://twitter.com/aidan_mclau/status/1899316464836075545) 对 **有多少人（甚至是 F1 赛车级别的）误解了刹车的功能** 进行了幽默的观察。这条推文引起了广泛关注。[@hkproj](https://twitter.com/hkproj/status/1899318678924984578) 开玩笑地回复说 **刹车“显然是用来让车手伸展脚部的”**。
- [@nearcyan](https://twitter.com/nearcyan/status/1899272957333275015) 推荐 [@TrungTPhan](https://twitter.com/TrungTPhan) 为 **“老实说，是最近整个网站上最顶尖的博主之一”**，称赞了他的内容并建议强烈关注。
- [@scottastevenson](https://twitter.com/scottastevenson/status/1899257834031685921) 宣布 **“Vibecoding，但用于法律文件。即将推出。”**

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Gemma 3 的期待与潜在影响**

- **[3月12日发布新款 Gemma 模型](https://i.redd.it/8qfnwj7433oe1.jpeg)** ([Score: 387, Comments: 70](https://reddit.com/r/LocalLLaMA/comments/1j8u90g/new_gemma_models_on_12th_of_march/)): **Gemma 3** 定于 **2025年3月12日**在**巴黎**举行的 "Gemma Developer Day" 活动期间发布。公告采用了简洁现代的设计，配有几何星形图标，突显了该活动的专业性和高科技特质。
  - **Gemma 3 预期**: 社区正期待在 "Gemma Developer Day" 活动期间发布 **Gemma 3**，不过一些用户对确认发布持怀疑态度。讨论强调了该活动高规格的演讲嘉宾阵容以及对重大声明的期待，尽管有人提醒考虑到活动的闭门性质，不应盲目假设一定会发布。
  - **技术兼容性与改进**: 用户非常关注确保 **Gemma 3** 能与 **llama.cpp** 无缝协作，许多人回忆起 **Gemma 2** 发布时的兼容性问题，希望这次能有更好的集成。一些用户提到 Google 内部使用了 **llama.cpp fork**，这暗示了改进兼容性以及对开源社区做出贡献的潜力。
  - **模型变体与性能**: 用户渴望看到更多中型模型，如 **Gemma 27B**，并建议推出 **32B、40B** 和 **70B** 等更大规模的模型以提升性能。同时，针对特定任务的小型模型如 **9B** 和 **12B** 也备受关注，强调了多样化模型尺寸以满足不同使用场景的需求。


**主题 2. M3 Ultra 512GB 运行 Deepseek R1 671B Q4 评测**

- **[M3 Ultra 512GB 运行 Deepseek R1 671B Q4 达到 18T/s (DAVE2D 评测)](https://www.youtube.com/watch?v=J4qwuCXyAcU)** ([Score: 384, Comments: 215](https://reddit.com/r/LocalLLaMA/comments/1j8r2nr/m3_ultra_512gb_does_18ts_with_deepseek_r1_671b_q4/)): 正如 **DAVE2D 评测**所强调的，**M3 Ultra 512GB** 在运行 **Deepseek R1 671B Q4** 时达到了 **18T/s** 的性能。
  - 讨论重点关注了 **RAG 系统**和内存带宽问题，指出了 **R1/MoE 架构**中的效率低下以及可能的优化领域。用户讨论认为小型模型通常更快，但 **70B 模型**的速度低于预期，且可能存在导致流水线停顿的**调度/线程问题**。
  - 评论者辩论了 **M3 Ultra** 与其他系统的**成本与效率**，将其与涉及 **Nvidia 5090** 和 **H200** 的配置进行对比，强调了 M3 Ultra 的**能效**和易获得性。用户提到，虽然 M3 Ultra 的功耗较低（**低于 200W**），但替代系统虽然性能更高，但成本和功耗也更大。
  - 存在关于量化方法的详细技术讨论，如 **Q4_K_M** 和内存交织，并提到了用于量化的 **GGML_TYPE_Q6_K** 和 **super-blocks**。用户还讨论了**内存带宽**及其对性能的影响，特别是在大容量 RAM 系统上进行 **inference** 时。


**主题 3. NVLINK 对 RTX 3090 性能的影响**

- **[NVLINK 将双 RTX 3090 的推理性能提升了近 50%](https://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)** ([Score: 144, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1j8i9rc/nvlink_improves_dual_rtx_3090_inference/)): 据报道，**NVLINK** 将双 **RTX 3090** GPU 的推理性能提升了近 **50%**。这表明对于协同使用这些 GPU 的任务，计算效率有了显著提高。
  - **主板与 GPU 配置**: 用户讨论了主板的 PCIe 通道配置，指出使用 x8 转接卡可能会限制性能。**hp1337** 解释了他们使用 x8 通道配置 6 个 GPU 的方案，并建议未来使用 x16 通道进行测试以获取潜在的性能洞察。
  - **NVLink 的可用性与替代方案**: **FullOf_Bad_Ideas** 询问了 RTX 3090 的 NVLink 桥接器的可用性和成本，**a_beautiful_rhind** 建议使用 [open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) 作为替代方案。然而，**Pedalnomica** 指出这仅能开启 P2P，无法达到 NVLink 的性能水平。
  - **量化与 FP8 计算**: **__JockY__** 等人讨论了在 RTX 3090 上使用 FP8 量化，强调 **vLLM** 在没有原生 FP8 硬件支持的情况下，通过 FP8 Marlin 内核实现性能提升，这一点已由 **Competitive_Buy6402** 和 **bihungba1101** 参考 [vLLM 的 GitHub](https://github.com/vllm-project/vllm/pull/5975) 证实。


**主题 4. 阿里巴巴用于情绪识别的 R1-Omni**

- **阿里巴巴刚刚发布了 R1-Omni！** ([Score: 244, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1j8mrju/alibaba_just_dropped_r1omni/))：**Alibaba** 推出了 **R1-Omni**，该模型专注于通过**全模态情感识别 (Omni-Multimodal Emotion Recognition)** 和**强化学习 (Reinforcement Learning)** 来增强情感智能。
  - **伦理担忧**：多位评论者对情感检测技术的伦理影响表示担忧，强调了侵入性以及对神经多样性个体可能产生的歧视等问题。人们担心自动化此类主观任务可能导致滥用和伤害，特别是在未经同意或缺乏透明度的情况下使用时。
  - **AI 心理治疗**：关于 AI 治疗师的讨论呈现两极分化，一些人看到了可及性和一致性等潜在益处，而另一些人则警告存在加剧焦虑或缺乏人类监管等风险。辩论涉及成本、有效性以及企业滥用的可能性之间的平衡。
  - **技术与社区层面**：提到了 **R1-Omni** 模型已在 [GitHub](https://github.com/HumanMLLM/R1-Omni) 上可用，并对其与阿里巴巴的关系及内部竞争提出了疑问。用户还批评了模型的命名习惯，并要求提供该技术的演示。


**主题 5. Reka Flash 3：新型开源 21B 模型**

- **Reka Flash 3，新型开源 21B 模型** ([Score: 220, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1j8tfh5/reka_flash_3_new_open_source_21b_model/))：**Reka Flash 3** 是一款拥有 **210 亿参数 (21B)** 的新型**开源模型**。它已在 **HuggingFace** 上线，更多详情可以在 [Reka AI 博客](https://www.reka.ai/news/introducing-reka-flash)中找到。
  - 尽管 **Reka Flash 3** 模型体量较小（**21B 参数**），但它正被拿来与 **QwQ-32B** 等更大的模型进行比较，并展示了极具前景的性能基准。一些用户注意到它在速度优先于规模的场景中的应用潜力，而另一些人则对其编程能力表示怀疑，特别是与 **Mistral Nemo** 等模型相比时。
  - 讨论强调了该模型的 **Apache 许可证**，这允许广泛的使用，且其尺寸非常适合 **24GB 显存显卡**。用户对其潜在的多模态能力表现出兴趣，尽管目前尚未得到证实。
  - 用户对该模型的推理能力表现出浓厚兴趣，对其解决“老虎谜题”等复杂问题的能力印象深刻。这证明了该模型在处理复杂推理任务方面的潜力，而此前人们认为这类任务需要大得多的模型才能完成。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Claude 3.7：通过调试提升开发者技能**

- **Claude 3.7 让我成为了更好的开发者。** ([Score: 234, Comments: 64](https://reddit.com/r/ClaudeAI/comments/1j8wiuo/claude_37_made_me_a_better_developer/))：作者批评 **Claude 3.7** 生成的代码过于复杂且效率低下，称其为“彻头彻尾的垃圾”和“过度设计的废话”。尽管感到沮丧，作者承认修复此类代码的过程提升了他们的开发技能，这表明解决 AI 生成的代码问题可以是一种有效的学习体验。
  - 评论者强调了良好的 **Git 实践**的重要性，例如为新功能创建分支并频繁提交，以便轻松回滚 AI 生成的代码。他们建议使用 **rebase** 在合并回主分支之前将多个提交合并为一个，并强调频繁提交是专业且有益的。
  - 一些用户讨论了他们使用 **Claude 3.7** 和 **3.5** 的经验，指出 3.7 经常产生过于复杂的代码，而 3.5 则更简单、更可靠。然而，对于 3.5 目前的性能评价褒贬不一，暗示其性能可能随时间有所下降。
  - 几位评论者分享了处理 AI 生成代码的策略，包括使用**测试驱动开发 (TDD)** 来引导代码质量，以及让 AI 解释概念而不是直接生成代码。他们警告不要依赖 AI 进行高层架构决策，因为这往往会导致现有功能的重复实现和过度复杂化。

- **[Dario Amodei: AI Will Write Nearly All Code in 12 Months!! Are Developers Ready?](https://v.redd.it/tqdzfzj1e2oe1)** ([Score: 181, Comments: 183](https://reddit.com/r/ClaudeAI/comments/1j8r1qi/dario_amodei_ai_will_write_nearly_all_code_in_12/)): **Dario Amodei** 预测 **AI** 将在 **12 个月**内编写几乎所有的代码，这预示着开发者将面临重大转变。视频内容未被分析，但标题表明了关于开发者是否为 AI 驱动编程的快速进步做好准备的讨论。
  - 许多用户对 **Dario Amodei** 的预测表示怀疑，将其与过去过于乐观的言论相提并论，如 **Elon Musk** 的 robotaxi 时间表和 **Hinton** 对放射科医生被取代的预测。他们认为，由于幻觉（hallucinations）和逻辑错误等问题，AI 生成的代码仍需要大量的人工监督，而这些问题目前的 AI 模型还无法轻易解决。
  - 几位评论者认为，虽然 AI 可以辅助编程，但由于它无法自主管理复杂任务、确保代码质量以及理解设计和架构（design and architecture），目前还无法取代开发者。他们强调 AI 工具可以生成代码，但仍需要人工验证和指导，使其更像是高级编译器（compilers）而非独立的程序员。
  - 普遍共识是，目前围绕 AI 能力的炒作很大程度上是由营销和融资活动驱动的。评论者强调，AI 编程的真正突破可能会因为其市场影响而泄露，而快速进步的说法往往更多是为了吸引投资者兴趣，而非反映眼前的技术现实。


- **[This is why I use ChatGPT instead of Grok](https://i.redd.it/iuewe1hqnzne1.png)** ([Score: 191, Comments: 14](https://reddit.com/r/ChatGPT/comments/1j8irsl/this_is_why_i_use_chatgpt_instead_of_grok/)): 该帖子批评了 **Claude 生成的代码输出**，并表达了相对于 **Grok** 更倾向于 **ChatGPT** 的立场。配图幽默地将 PC 端使用 **Reddit** 对比手机端的“无休止刷屏”（doomscrolling），暗示前者更像是一种智力活动，类似于“知识策展”或“Reddit 话语分析”。
  - **ChatGPT vs. Grok**: **Grok** 被批评不如 **ChatGPT** 通用且过于复杂，尽管 **ChatGPT** 被贴上“骗子”的标签，但在语法纠错等任务中仍是首选。用户对 **ChatGPT** 倾向于在不告知的情况下删除其认为不必要的内容感到沮丧。
  - **跨设备的无休止刷屏**: 讨论指出，不同设备上的 **doomscrolling** 是相似的，区别在于 **PC** 端的操作看起来更受控，但仍然涉及同样的心理压力。这种区别更多在于观感和控制感，而非设备本身。
  - **AI 模型的用户体验**: 用户对比较 **Grok 3** 和 **GPT-4.5** 等不同模型的响应很感兴趣，但 **GPT Plus** 每周 **50 条消息的限制** 阻碍了这种探索。


**主题 2. Nvidia 的 Gen3C：图像转 3D 的进展**

- **[Gen3C - Nvidia's new AI model that turned an image into 3D](https://v.redd.it/bqq73tdca1oe1)** ([Score: 259, Comments: 25](https://reddit.com/r/StableDiffusion/comments/1j8n8qh/gen3c_nvidias_new_ai_model_that_turned_an_image/)): **Nvidia 的 Gen3C** 是一个新的 AI 模型，可以将 2D 图像转换为 3D 表示，展示了图像处理技术的进步。
  - **内存担忧**: 用户担心 **Gen3C** 可能会非常消耗内存，质疑其在消费级 GPU 上的可行性。**TheSixthFloor** 建议它可能至少需要 **16GB VRAM**，类似于其他先进的 AI 模型。
  - **技术澄清**: **Silonom3724** 澄清说 **Gen3C** 使用的是 **Image to point cloud to NeRF**（图像转点云转 NeRF），而不是直接的 3D 多边形表示；而 **grae_n** 注意到其中包含反射材料，暗示采用了 **gaussian/NeRF** 方法。
  - **可用性与获取**: **Gen3C** 的代码预计很快发布，并提供了 [GitHub 仓库](https://github.com/nv-tlabs/GEN3C) 和 [Nvidia 研究页面](https://research.nvidia.com/labs/toronto-ai/GEN3C/) 的链接。用户渴望获得关于其发布和本地运行能力的更新。


**主题 3. Dario Amodei：AI 代码生成预测与质疑**

- **[Dario Amodei：AI 将在 12 个月内编写几乎所有代码！！](https://v.redd.it/nc1umxord2oe1)** ([评分: 139, 评论: 130](https://reddit.com/r/OpenAI/comments/1j8r0jw/dario_amodei_ai_will_write_nearly_all_code_in_12/))：**Dario Amodei** 预测 **AI** 将在 **12 个月**内编写几乎所有代码，这引发了工程界的怀疑。由于帖子中缺乏详细论据，限制了对该预测可行性的进一步分析。
  - 批评者认为，由于上下文窗口（context window）大小的限制，**AI** 缺乏在 **12 个月**内编写所有代码的能力，这影响了它在大型代码库中保持全局意识的能力。**AI** 在有效处理像 **Linux kernel** 这样复杂的系统或关键系统控制代码方面仍显吃力，将内核转换为 **Rust** 的失败尝试就证明了这一点。
  - 对于在没有人类监督的情况下由 **AI** 编写代码的实用性，人们普遍持怀疑态度。工程师们强调了人类审查和清晰规范的必要性，而 **AI** 目前无法独立管理这些。**中层管理人员（Middle management）**也因缺乏指导 **AI** 完成此类任务的技术专长而受到批评。
  - 一些评论人士认为 **Dario Amodei** 的预测是吸引资金的战略举措，而非现实的预报。目前 **Copilot** 等工具的局限性凸显了 **AI** 在高效处理大型项目时面临的挑战。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Exp 生成的摘要之摘要的总结

**主题 1：OpenAI 的 Agent 开发生态系统不断演进**

- **OpenAI 发布用于 Agent 创建的 Responses API 和 SDK**：OpenAI 推出了全新的 [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses) 和 [Agents SDK](https://platform.openai.com/docs/guides/agents)，旨在简化 Agent 开发，强调改进集成、优化工作流和生产就绪性。新 SDK 提供了追踪（tracing）、护栏（guardrails）和生命周期事件等功能，但也预示着 [Assistants API](https://openai.com/policies/terms-of-use/) 将在 2026 年中旬停用。
- **社区辩论新 Agent 工具的优劣**：社区正在积极讨论新工具的价值和功能，一些人对 **GPT-4.5** 生成代码的可靠性和一致性表示怀疑，并寻求 Responses API 与现有 chat completions API 之间区别的澄清。虽然新的 [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search) 旨在提高搜索结果的可靠性，但用户观察到它缺乏类似其他平台的来源选择功能。
- **Agents SDK 的可观测性工具引发追踪问题**：OpenAI 声称在其新 [Agents SDK](https://platform.openai.com/docs/guides/agents#orchestration) 中集成了 Braintrust 数据追踪，这引起了热议，用户想知道 OpenAI 是否支持与 Langfuse 或其他 Agent 可观测性工具的集成。实际上它是支持的，关于如何使用 [OpenAI's SDK](https://platform.openai.com/docs/guides/agents#orchestration) 或将追踪发送到自定义工具的更多细节，可以在 [此 GitHub 仓库](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors) 中找到。

**主题 2：探索 AI 模型能力与局限性的前沿**

- **Reka Flash 3 加入战局**：Reka Labs 发布了 [Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3)，这是一个从零开始训练的 21B 推理模型，展示了极具竞争力的性能和多模态能力，向 QwQ-32B 和 o1-mini 等现有模型发起挑战。尽管它是开源的，但关于其架构和训练数据仍存在疑问，其用途也从设备端使用转向为 Reka 的 AI 协作平台 [Nexus](https://getnexus.reka.ai) 提供动力。
- **Anthropic 的 Claude 3.7 在 Perplexity 上面临输出限制**：用户发现 [Claude 3.7](https://www.anthropic.com/news/claude-3-family) 在 Perplexity 上的输出限制为 5000 token，这与 Anthropic 官方文档声明的最高 128K 输出能力形成对比。这种差异引发了对该模型实际效用的质疑，并凸显了了解平台特定限制的重要性，尤其是在商业应用中。
- **GPT-4.5 代码：不一致得令人发笑**：用户报告称 [GPT-4.5](https://openai.com/blog/new-models-and-developer-products-announced-at-devday) 生成的代码不一致，例如在定义了 `start()` 函数后却调用一个不存在的函数 `startApp()`。人们对 **GPT-4.5** 输出结果需要持续监督表示担忧，并对 AI 生成代码的整体可信度感到忧虑，称需要“像照看小孩一样照看这种‘智能’”。

**主题 3：社区驱动的 AI 开发工具与技术**

- **AI Code Fusion 工具首次亮相，用于优化 LLM 上下文**：一位社区成员介绍了 [AI Code Fusion](https://github.com/codingworkflow/ai-code-fusion)，这是一个旨在通过打包文件、计算 token 和过滤内容来为 LLM 上下文优化代码的工具，展示了社区在应对 AI 开发挑战方面的积极态度。该工具的创建者正积极寻求社区反馈以完善其功能。
- **Aider 的 Watch Files 实时模式支持交互式编程**：Aider 新增的 `--watch-files` 标志启用了 [实时模式 (live mode)](https://aider.chat/docs/usage/watch.html)，允许开发者通过添加如 `AI!`（触发 Aider 进行更改）和 `AI?`（触发其回答问题）等注释来与 AI 进行交互式编程，标志着向更具协作性和交互性的编程工作流转变。
- **利用 Browserless.io 绕过机器人检测**：Nous Research AI 成员建议使用 [Browserless.io](https://www.browserless.io) 在网页抓取中绕过机器人检测和 CAPTCHA，强调其能够避免留下细微指纹并绕过多种网页保护机制。它支持使用 Puppeteer 或 Playwright 进行浏览器自动化，并提供用于测试和调试的抓取 IDE。

**主题 4：AI 工作负载的硬件和基础设施考量**

- **本地 LLM vs 云端 GPU：大辩论仍在继续**：用户讨论了在高端硬件（如配备 512GB RAM 的 M3 Ultra Mac Studio）上本地运行 LLM 与利用云端 GPU 的成本效益，在性能与长期负担能力之间取得平衡。AMD 用户报告称，Vulkan 和 ROCm 的性能在 24.12.1 驱动程序中出现故障，性能下降了 35%，不过 ROCm 已在 v1.1.13+ 版本中修复。
- **推测解码在某些配置下停滞**：当受限于 RAM 带宽，或者在比较 0.5b 与 14b 模型时，推测解码（Speculative Decoding）的表现可能比标准推理更差。随着 100Gbit NVMe 驱动器、400Gbit 网络和 CXL 内存的出现，swap 再次变得有用，正如 Dave2D 的 M3 Ultra Mac Studio 评测中所强调的那样。
- **SemiAnalysis 举办 Nvidia Blackwell GPU 黑客松**：[SemiAnalysis](https://semianalysis.com/hackathon-2025/) 将于 3 月 16 日举办 Nvidia Blackwell GPU 黑客松，内容包括对 Blackwell 和 PTX 基础设施的实操探索，以及来自 OpenAI、TogetherAI 和 Thinking Machines 的演讲者。该黑客松在多个 Discord 频道中被提及，凸显了其行业重要性，并以抢先体验尖端 GPU 技术的承诺吸引了开发者。

**主题 5：AI 开发中的伦理问题和使用政策**

- **关于 OpenAI 服务条款和越狱的讨论**：鉴于 [OpenAI 的服务条款](https://openai.com/policies/terms-of-use/)，成员们进行了谨慎的讨论，服务器规则也禁止讨论如何绕过这些限制，同时建议关注伦理边界内允许的内容。这些通用政策**并不禁止**通过涉及奇幻写作、图像生成或角色扮演游戏的文本进行探索。
- **讨论提示词技术和创意用例**：OpenAI 的成员正在使用提示词（Prompting）技术，试图在不违反安全政策的情况下诱导模型给出更坦诚的回答。提出的问题包括让模型像用户的奶奶以前那样教编程。
- **用户希望 ChatGPT 拥有 Grok 风格**：讨论集中在与过滤内容相关的期望“氛围（vibes）”上；用户分享了诸如 [这个](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268) 迷因，并表达了希望 ChatGPT 不要以同样的方式过滤或限制内容的愿望。还进行了 Deep Research 的价格比较，称 **OpenAI 的 Deep Research 是最佳选择**，但也承认“现在的限制太糟糕了，哈哈”。

---

# PART 1: High level Discord summaries

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Nightly 遭遇滑铁卢**：**Cursor** 的最新 Nightly 更新引入了关键 Bug，破坏了 **AI Chat** 和 **Cursor settings**，导致 GUI 无法使用。
   - 用户报告称，重新安装应用也无法解决问题，这表明最新的 Nightly 更新本身存在问题。
- **Claude 3.7 定价引发不满**：用户对 **Claude 3.7 Thinking** 的新定价感到愤怒，现在每次请求消耗 **2 个额度而非 1 个**，促使部分用户考虑替代方案。
   - 讨论指出，在大上下文中使用 **Claude 3.7 Thinking**，每次请求的成本可能高达 **16 美分**。
- **Manus AI：革命性的 Agent 还是过度炒作的工具？**：一位用户分享了 **Manus AI**，称其为“最疯狂的 AI Agent”，并展示了其克隆 Apple 网站的能力（[来自 el.cine 的推文](https://x.com/ehuanglu/status/1899110687902978373?s=46&t=CLGnxOi5OPp22iT8UYkr1A)）。
   - 怀疑论者认为它可能只是带有 PC 操作工具的 **Sonnet 3.7**，而其他人则在畅想 AI Agent 运行公司的未来。
- **Cursor 的稳定性面临审查**：多名用户报告 **Cursor 几乎无法工作**，经常卡顿或无响应，部分用户的 **Claude Max** 无法运行。
   - 一些用户发现回滚到 **.46.11** 版本可以解决问题，这引发了关于 **.47** 版本可能仅限部分用户使用的猜测。
- **本地 LLM vs 云端 GPU：大辩论**：一位用户建议购买配备 **512GB RAM 的 M3 Ultra Mac Studio** 来本地运行 **完整版 DeepSeek R1**，引发了关于该配置性价比的讨论。
   - 虽然有些人青睐本地 LLM，但其他人认为云端 **GPU** 提供更快的推理速度，且从长远来看更经济。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布桌面应用**：Perplexity AI 发布了原生的 **PC 桌面应用**（[perplexity.ai/platforms](https://www.perplexity.ai/platforms)），支持 **语音听写**、**键盘快捷键**以及访问最新模型。
   - 然而，用户指出该应用本质上是网页版的套壳，缺乏桌面端的优势以及像 Complexity 这样的浏览器扩展；有人评价道：“它只是一个被削弱的浏览器”。
- **Revolut 促销码让用户头疼**：**Revolut** 用户在兑换 **Perplexity Pro** 促销码时遇到问题，部分人被告知需要创建新账户或联系 Revolut。
   - 正如一位用户提到的：“我联系了 Revolut，他们说我需要在 Perplexity 注册新账户。这很让人扫兴，但嘿，我觉得还是值得的。”
- **Claude 3.7 限制为 5K Tokens**：用户发现 **Claude 3.7** 在 Perplexity 上的输出限制被硬性设定为 **5000 tokens**。
   - 这与 Anthropic 的官方文档形成了对比，后者声明其输出可达 **128K**。
- **大学探索 Perplexity Enterprise**：一位用户正在评估将 **Perplexity Enterprise** 集成到大学系统中，强调其连接内部政策和程序知识库的能力，参见 [Perplexity Enterprise FAQ](https://www.perplexity.ai/enterprise)。
   - 该平台提供内部数据搜索和自定义工作区的功能。
- **API chat completions 出现截断现象**：一位成员报告在调用 **sonar-reasoning-pro 模型** 的 **chat.completions API** 时出现间歇性内容截断；参见 [Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions)。
   - 增加 **max_token** 配额并未解决问题；该成员指出 [Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions) 始终能输出完整响应，表明该问题是 API 特有的。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Reka Flash 3 引起关注**：[Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3) 已发布，这是一个采用 Apache 2.0 许可证的 **21B 推理模型**，性能可与 **QwQ-32B** 和 **o1-mini** 相媲美。
   - Reka 团队由前 DeepMind 员工组成，[Reka 官网](https://www.reka.ai/ourmodels)指出该 Flash 模型是多模态的。
- **提供多 GPU 训练建议**：当被问及如何使用 Unsloth 在多节点和多 GPU 上微调大模型时，一名成员建议使用 **axolotl** 或 **llama factory**。
   - 目前 Unsloth 尚未（正式）支持多 GPU，不过相关支持可能会在未来几周内推出。
- **AI Code Fusion 工具亮相**：一名成员介绍了 **AI Code Fusion**，这是一个旨在通过打包文件、计算 Token 和过滤内容来为 **LLM contexts** 优化代码的工具，可在 [GitHub](https://github.com/codingworkflow/ai-code-fusion) 上获取。
   - **AI Code Fusion** 的创作者正在寻求社区对该工具的反馈。
- **正则表达式在日期提取上优于 LLM**：一位用户目标是训练模型从查询中提取正确的营业时间，他人建议 **regex system**（正则表达式系统）可能比使用 AI 更适合这项任务。
   - 一名成员链接了一个相关的 [xkcd 漫画](https://xkcd.com/208/)，主题是*用复杂的解决方案过度设计简单的任务。*
- **GRPO Batch Size 影响训练**：**GRPO batch size** 必须与生成数量一致，且 GRPO RL 算法的 `num of generation` 必须经过良好调优。
   - 建议 `num generations` 的范围是 **4 到 8**，增加 Batch Size 倍数会缩短训练时间，但会大幅增加 GPU 显存需求。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Hermes 展示初步推理能力**：新的 **Deep Hermes** 模型已发布，具有从 R1 蒸馏出的*初步推理能力*，详见 [Hugging Face](https://huggingface.co/)。
   - 成员们对测试该模型感到兴奋，但也表达了对超出上下文长度（context length）的担忧。
- **通过 Browserless.io 实现无检测爬取**：一名成员推荐使用 [Browserless.io](https://www.browserless.io) 来绕过网页爬取中的机器人检测和 CAPTCHAs，强调其具有*避免留下细微指纹*的能力。
   - 它支持使用 **Puppeteer** 或 **Playwright** 进行浏览器自动化，并提供了一个用于测试和调试的爬取 IDE。
- **SemiAnalysis 举办 Blackwell GPU 黑客松**：[SemiAnalysis](https://semianalysis.com/hackathon-2025/) 将于 3 月 16 日举办 **Nvidia Blackwell GPU 黑客松**，活动包括对 Blackwell 和 PTX 基础设施的实操探索，演讲嘉宾来自 **OpenAI**、**TogetherAI** 和 **Thinking Machines**。
   - 该活动由 **Together**、**Lambda**、**Google Cloud**、**Nvidia** 和 **OpenAI** 等公司赞助。
- **利用前向梯度优化 UT**：成员们讨论了使用 [前向梯度（forward gradients）](https://arxiv.org/abs/2202.08587) 来优化 **Universal Transformer (UT)** 训练，因为在 UT 的共享层中这种方法可能更有效。
   - 这种方法与 **N-GPT** 结合使用可能会很有趣。
- **字节跳动推出 Trae IDE**：字节跳动发布了 [Trae](https://www.trae.ai/)，这是一个类似于 Cursor 的免费 AI IDE，内置 **Claude Sonnet 3.7** 供免费使用，目前支持 **Mac** 和 **Windows**。
   - Linux 版本正在计划中，该 IDE 的目标用户是 AI 编程初学者。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Loglikelihood 评估解放了 LLM**：成员们建议在多项选择题问答（**MCQA**）任务中使用**基于 loglikelihood 的评估**，从而无需严格的输出格式化。
   - 这解释了为什么指令模型（instruct models）能答对某些问题，而它们的聊天（chat）变体通常得分却为 **0**。
- **扩散模型执行频谱自回归**：一篇博客文章（[Spectral Autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)）揭示了**图像扩散模型在频域中执行近似自回归**。
   - 作者指出，这一理论虽然直观，但在实践中预测能力有限，特别是在使用匹配目标分布 RAPSD 的有色噪声时。
- **Neural Flow Diffusion Models 增强高斯噪声**：**Neural Flow Diffusion Models (NFDM)** 通过支持比标准高斯噪声更广泛的前向过程，并采用端到端、无需模拟的优化目标，增强了扩散模型。
   - 根据[论文](https://arxiv.org/abs/2404.12940)，实验证明了 **NFDM** 强大的性能和最先进的似然估计（likelihood estimation）。
- **远离“坏样本”的引导避免了模式崩溃**：一篇[论文](https://arxiv.org/abs/2406.02507)建议引导模型远离“坏样本”（badness）而非“无条件状态”，以避免 CFG（classifier-free guidance）的模式崩溃（mode dropping）。
   - 该方法实现了对图像质量的解耦控制，且不损害变化的多样性，在 ImageNet 上实现了 64x64 分辨率下 **1.01** 和 512x512 分辨率下 **1.25** 的创纪录 FID。
- **Tokenizer 问题威胁 Patching 评估**：一位成员在分析 **Math CoT** 答案的重要电路时，寻求关于选择合适**指标**（metrics）来评估 patching 结果的建议。
   - 核心问题在于 Tokenizer 将 **10** 和 **15** 等数字各拆分为两个 token，破坏了评估方程的直接应用。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Avoma 与 Gong 竞争**：[Avoma](https://www.avoma.com/) 作为一个集笔记自动化、调度、辅导和预测于一体的 **AI 平台**，被认为是 **Gong** 的竞争对手。
   - 该建议发表于 #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435)。
- **Factorio Learning Environment 测试 LLM**：**Factorio Learning Environment (FLE)** [已在 GitHub 上线](https://jackhopkins.github.io/factorio-learning-environment/)，旨在利用游戏 **Factorio** 测试 Agent 在长期规划、程序合成和资源优化方面的能力。
   - 一位成员表达了兴奋之情，并幽默地请求立即入职 *Anthropic Factorio 实验室*，同时指出该环境目前仅限文本，但可以从 **Qwen 2.5 VLM** 等多模态数据输入中获益。
- **Contextual AI 发布遵循指令的 Reranker**：**Contextual AI** 推出了一款[新的 Reranker](https://contextual.ai/blog/introducing-instruction-following-reranker/)，它可以遵循自定义指令，根据新鲜度、文档类型或来源等要求对检索结果进行排序。
   - 该公告发布于 #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435)。
- **OpenAI 发布 Agent 工具**：**OpenAI** 推出了用于构建 Agent 的新工具，包括 [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses)、[Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search)、[Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use) 和 [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search)。
   - 他们还发布了一个新的开源 [Agents SDK](https://platform.openai.com/docs/guides/agents)，集成了具有追踪、护栏和生命周期事件功能的[可观测性工具](https://platform.openai.com/docs/guides/agents#orchestration)，并宣传该 SDK 已达到生产级标准。
- **Luma Labs 推出 Inductive Moment Matching**：**Luma Labs** 发布了 [Inductive Moment Matching (IMM)](https://lumalabs.ai/news/inductive-moment-matching)，这是一种新的预训练技术，声称其采样质量优异，且效率比扩散模型高出 10 倍。
   - 讨论集中在 #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435)。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 发布 FAQ 页面**：OpenRouter 推出了一个 [FAQ 页面](https://openrouter.ai/docs/faq)，以解决常见问题并为用户提供更多清晰度。
   - 随新 FAQ 一同发布的还有一个小的体验优化（quality of life）更新，以提升用户体验。
- **Gemini 2.0 图像生成泄露**：**Gemini 2.0 Flash Experimental** 图像生成功能已流出，上限为 **32k** 上下文，但缺乏代码执行、搜索接地（search grounding）或函数调用（function calling）功能；用户在 `gemini-2.0-flash-exp` 下发现了图像保存代码。
   - 这一消息源自 [Reddit 上的这个帖子](https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/)。
- **OpenAI 预告面向开发者的发布**：成员们根据提到 **Responses API** 的[此帖子](https://platform.openai.com/docs/api-reference/responses)猜测 **OpenAI** 将有新发布。
   - 该发布预计在 **太平洋时间上午 10 点** 进行。
- **关于 Cohere 的 AYA Vision 的询问**：成员们询问了 **OpenRouter** 对 **Cohere** 的 **AYA vision** 及其他 **Cohere** 模型的支持情况，**AYA Expanse** 模型（8B 和 32B）的定价可能为 **输入 $0.50/1M Tokens**，**输出 $1.50/1M Tokens**。
   - 用户仍在尝试确认这些费率，如[此截图](https://cdn.discordapp.com/attachments/1094454198688546826/1349049467378339902/image.png?ex=67d1afb9&is=67d05e39&hm=ffdce841e8f353b45682a480ea8f937b0169a3414ebe0967c87230ba436786b4&)所示。
- **参数计算功能被移除**：**OpenRouter** 因准确性问题移除了参数计算功能，认为其可能产生误导。
   - 团队计划稍后通过人工策展进行重构，并承认调整参数非常困难，幽默地称其为“古老符文” (*ancient runes*)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agent Tools 在 OpenAI 开发者直播中揭晓**：OpenAI 在直播中首次推出了面向开发者的 **Agent Tools**，随后举行了 **AMA** (*Ask Me Anything*) 环节，提供与开发团队直接互动的机会。更多信息和问题请见 [OpenAIDevs 的 X 帖子](https://x.com/OpenAIDevs/status/1899502117171155002)。
   - **AMA** 安排在 **太平洋时间上午 10:30–11:30**，允许开发者直接与新功能背后的团队交流。
- **用户渴望 ChatGPT 拥有 Grok 的风格**：成员们表达了希望将 **Grok** 的独特特征引入 **ChatGPT** 的愿望，如引用 **Elon Musk** 的 [Elon Musk GIF](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268) 所示。
   - 讨论围绕所期望的“风格”（vibes）本质展开，特别是关于内容过滤方面。
- **GPT-4.5 生成代码存在不一致性**：用户报告称 **GPT-4.5** 生成的代码不一致，例如调用不存在的函数或错误命名现有函数，导致人们对其相对于 **GPT-4o** 的可靠性产生疑问。
   - 成员们对需要不断监督 **GPT-4.5** 的输出以及 AI 生成代码的整体可信度表示担忧，称需要“像保姆一样照看这种‘智能’”。
- **新的 Responses API 是 Assistant API 的镜像？**：一位成员询问了**新 Responses API** 与现有 **Chat Completions API** 之间的区别，引发了关于 API 功能的讨论。
   - 澄清信息表明，*新 Responses API* “基本上是更好用的 Assistants API”。
- **越狱行为危害服务条款 (ToS)**：成员们讨论了模拟场景以使 **AI 模型** 绕过限制或提高准确性的行为，这被视为“越狱”（jailbreaking），但可能违反 [OpenAI 的服务条款 (ToS)](https://openai.com/policies/terms-of-use/)。
   - 用户被警告不要违反 **ToS** 以保护账号权限，服务器禁止讨论绕过限制的行为；但涉及幻想或角色扮演的暴力讨论不被视为禁忌。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 'Watch Files' 现已上线**：Paul Gauthier 宣布，运行带有 `--watch-files` 标志的 `aider` 现在可以启用 **live mode**，通过 `AI`、`AI!` 或 `AI?` 注释监视仓库中的所有文件以获取编码指令，如 [Aider 浏览器 UI 演示视频](https://aider.chat/docs/usage/watch.html)所示。
   - 感叹号 `AI!` 触发 aider 进行更改，而问号 `AI?` 则触发其回答问题。
- **Aider 每日预算差异巨大**：成员们讨论了 **Aider** 的每日预算，其中一人报告称，每周进行 7-12 小时的 AI 编码，**Sonnet 3.7** 的成本约为排行榜成本的 **2 倍**。
   - 他们警告说，**每周 40 小时**的工作量很容易导致 **8-10 倍**的排行榜成本，而其他用户则通过默认使用更便宜的模型（如 **o3 或 R1**）来控制成本。
- **DMCA 下架通知导致 Claude Code 被封禁**：一名用户报告称，因 fork 了 **Claude 代码泄露仓库**而收到 **DMCA** 下架通知，原始泄露者和所有 fork 均受到影响。
   - 另一名用户推测 **o1 pro** / **o3 mini pro** 可能很快会在 API 中发布。
- **Aider 编辑格式定义**：Aider 排行榜中的“正确编辑格式”是指 Aider 期望 LLM 在编辑文件时使用的格式，[关于编辑格式的 Aider 文档](https://aider.chat/docs/more/edit-formats.html)详细介绍了 *whole* 和 *diff* 编辑格式。
   - 不同的模型在不同的格式下表现更好。
- **Code-Act 仓库可能值得关注**：一名成员分享了 [code-act 仓库](https://github.com/xingyaoww/code-act)的链接。
   - 他们指出这可能与讨论有关。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Unity 与 LM Studio 联动**：一名成员展示了一个 [YouTube 视频](https://www.youtube.com/watch?v=dQw4w9WgXcQ)，使用 **JSON** 文件进行数据传输连接 **Unity** 和 **LM Studio**，但不确定该发布在 **Discord** 的哪个板块。
   - 用户正请求开设专门的 **Unity** 频道以更好地组织内容。
- **寻求 DIY 内部 LLM 聊天系统的建议**：一名成员正在寻求建立带有用户账户的内部 **LLM Chat** 的建议，该系统需与公司的 **Google Docs** 知识库集成，并可能使用推理 **API**。
   - 他们正在考虑使用 **LlamaIndex** 作为向量数据库，**AnythingLLM** 或 **OpenWebUI** 作为聊天界面，并探索 **LM Studio** 内部的选项。
- **Python SDK 缺少 Vision 支持，TypeScript SDK 领先**：一名使用 **Python SDK 1.0.1** 的成员注意到 **Typescript SDK** 可以向 vision 模型发送图像，但该功能尚未移植到 **Python**。
   - 社区正在等待 **Python SDK** 支持 vision 模型。
- **Copy4AI：捕获代码上下文的扩展**：一名成员询问了 [Copy4AI 扩展](https://copy4ai.dev/)的 `ext install` 命令，该扩展旨在为 AI 助手复制代码片段。
   - 该扩展现更名为 `leonkohli.snapsource`，可以通过 **VS Code** 的扩展侧边栏访问。
- **AMD 驱动灾难：Vulkan 和 ROCm 受损，部分已恢复**：一名 AMD 用户报告称，在 **24.12.1** 驱动中，**Vulkan** 和 **ROCm** 的性能下降了 **35%**，但 **ROCm** 在 **v1.1.13+** 中已修复。
   - Vulkan 性能在 **25.1.1** 中仍保持在 **50%**，在 **25.2.1** 中逐步改善，并已向 [AMD 提交错误报告](https://www.amd.com/en/support/kb/faq/rs-help)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **X 遭受网络风暴攻击**：**Dark Storm** 声称对 [X 平台遭受的 DDoS 攻击](https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/) 负责，该攻击导致该平台出现大规模停机。
   - 专家们驳斥了 Elon Musk 关于 **Ukrainian**（乌克兰）参与其中的暗示，**Ciaran Martin** 在 [BBC 的一篇文章](https://www.bbc.co.uk/news/articles/c62x5k44rl0o) 中称其“完全没有说服力”。
- **LanguageBind 优于 ImageBind**：成员们讨论了使用单一解决方案处理 *图像、音频、视频和 PDF* 模态的问题，一位成员推荐了 [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind)，指出它 *支持所有模态* 且 *优于 ImageBind*。
   - 该模型完全基于合成数据集和公开数据集训练，其性能可与 **OpenAI o1-mini** 等专有模型相媲美。
- **Reka Space 变得更小**：**Reka Flash 3** 是一款 **21B** 通用推理模型，它不再被称为 *端侧（on-device）* 模型，而是用于驱动 Nexus —— Reka 旗下用于创建和管理具有原生深度研究能力的 AI Worker 的平台（[Reka Space](https://space.reka.ai), [getnexus.reka.ai](https://getnexus.reka.ai)）。
   - 该模型完全基于合成数据集和公开数据集训练，其性能可与 **OpenAI o1-mini** 等专有模型相媲美，并为 Nexus 提供动力。
- **RAGcoon 发布以助力初创公司**：一个新的名为 [RAGcoon](https://github.com/AstraBert/ragcoon) 的 **Agentic RAG** 项目已发布，旨在通过 *混合搜索（hybrid search）、查询扩展（query expansion）* 和 *多步查询分解（multi-step query decomposition）* 导航各种资源和建议，从而协助构建初创公司。
   - 该项目基于 **LlamaIndex** 构建，使用 **Qdrant** 作为向量数据库服务，**Groq** 进行 LLM 推理（使用 **Qwen 的 QwQ-32B**），**Hugging Face** 提供 Embedding 模型，**FastAPI** 作为后端 API，以及 Google 的 **Mesop** 作为前端，并拥有令人印象深刻的 **检索上下文可靠性**。
- **Ollama 接管 HfApiModel**：成员们展示了如何将 Hugging Face 的 `HfApiModel` 替换为 **Ollama** 以配合 `smolagents` 使用，方法是创建一个自定义的 `OllamaModel` 类，该类与 Ollama 的 API 进行交互以生成 Prompt，从而允许在 `smolagents` 中使用本地 LLM。
   - 他们还分享了在 `smolagents` 中使用 **Gemini, OpenAI 和 DeepSeek 模型** 的代码片段，提供了设置 `LiteLLMModel` 和 `OpenAIServerModel` 以及相应 API Key 的示例，并提供了 [Google AI Studio 的链接](https://aistudio.google.com/app/apikey) 以获取 **Gemini** 的免费 API Key。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 新手前往圣何塞**：尽管缺乏 **CUDA** 经验，一位成员仍表示有兴趣参加 **3 月 16 日** 在 **圣何塞（San Jose）** 举行的 **GPU mode** 会议。
   - 讨论引发了关于参与会议是否需要专业知识的疑问。
- **Triton 的 `tl.full` 解决类型转换难题**：一位用户成功在 **Triton** 中使用 `tl.full` 创建了一个具有定义值和数据类型的 **0 维张量（0-dim tensor）**（`tl.full((), 5, tl.int8)`），以绕过在向张量累加时的溢出困境。
   - 成功的解决方案涉及：`tmp_5 = tl.full((1,), value=5, dtype=tl.int8); out = a.to(tl.int8) + tmp_5`。
- **Triton Softmax Kernel 速度取胜**：一位用户在 **Triton** 中的流水线 Softmax Kernel 表现出人意料地优于 **PyTorch**，证明在 **float16 T4** Colab 上速度更快，如[此图](https://cdn.discordapp.com/attachments/1189607595451895918/1349066304341934160/image.png?ex=67d1bf67&is=67d06de7&hm=e424c1148a06e3ac3adac9c31cd7c0bc6e930f047dc69c2db02cba55e5949695&)所示。
   - 结果展示了 **Triton** 如何实现新的高吞吐量设计。
- **Padding 防止 SMEM Bank 冲突**：**stmatrix** 的地址需要进行 Padding（填充），以避免指向同一个起始 **SMEM bank**，否则会触发 8 倍冲突，这借鉴了之前在 fast.cu 和 deepgemm 代码中实现的解决方案。
   - 鉴于 *不存在硬件解决方案*，当分块布局（tiled layouts）不切实际时，内存布局管理至关重要。
- **HuggingFace 库通过 WebNN/WebGPU 迁移至 TS/JS**：一位成员正积极使用 **WebNN/WebGPU** 将整个 **HuggingFace** 库移植到 **TS/JS**，以创建一个前端实现。
   - 另外，**IPFS Accelerate JS** 的初始结构已通过占位符模块和 TypeScript 转换实现，详见[此 commit](https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773)。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **SemiAnalysis 举办 Blackwell GPU Hackathon**：[SemiAnalysis](https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/) 将于 **3 月 16 日星期日**举办 **Nvidia Blackwell GPU Hackathon**，届时将邀请来自 **OpenAI**、**TogetherAI** 和 **Thinking Machines** 的演讲者。
   - 本次 Hackathon 旨在探索 **Blackwell & PTX 基础设施**并协作开发开源项目，赞助商包括 Together、Lambda、Google Cloud、Nvidia、GPU Mode、Thinking Machines、OpenAI、PyTorch、Coreweave 和 Nebius。更多详情请见 [SemiAnalysis Hackathon 页面](https://semianalysis.com/hackathon-2025/)。
- **Reka Labs 发布 Reka Flash 3**：[Reka Labs](https://x.com/RekaAILabs/status/1899481289495031825) 开源了 **Reka Flash 3**，这是一个从零开始训练的新型推理模型，仅拥有 **21B 参数**却实现了极具竞争力的性能。
   - 该模型在合成数据集和公开数据集上进行了微调，随后通过 **RLOO** 结合基于模型和基于规则的奖励进行训练，强制模型输出 *&lt;/reasoning&gt;* 以控制质量与思考时间的平衡，详见其 [博客文章](https://www.reka.ai/news/introducing-reka-flash)。
- **Anthropic ARR 飙升，助力 Manus AI**：据 [The Information](https://www.theinformation.com/articles/anthropics-claude-drives-strong-revenue-growth-while-powering-manus-sensation) 报道，**Anthropic** 的 **ARR** 在 2025 年前两个月从 **10 亿美元增长至 14 亿美元**，其模型正为 *最新的 AI 轰动项目* **Manus** 提供动力。
   - 这些模型正在为 **Manus** 提供支持，后者被描述为 *最新的 AI 轰动项目*。
- **OpenAI 推出新 API 和 Agents SDK**：[OpenAI](https://x.com/btibor91/status/1899513477716410871) 发布了新的 API 和工具，以便更轻松地开发 Agent 应用，包括 **Responses API**、**Web search 工具**、**File search**、**Computer use 工具**以及一个**开源的 Agents SDK**。
   - 现有的 **Assistants API** 将在 2026 年中期逐步停用，更新日志中还提到了 API 中新增的 **o3-mini-pro** 和 **o1-pro** 模型。
- **Dario 预测 AI 将主导编程**：Anthropic CEO Dario Amodei 预测，AI 将在未来 **3 到 6 个月**内编写 **90%** 的代码，并在 **12 个月**内编写几乎所有代码，据一条 [推文](https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ) 称。
   - 这一大胆的预测引发了开发者关于编程未来以及 AI 在其中角色的讨论。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Servers 在 Cursor 集成中遇到困难**：用户报告了在 **Cursor** 中集成 Brave Search 等 **MCP servers** 时遇到的问题（尽管在 Claude 中集成成功），错误提示为 *no tools available*，详情见 [glama.ai/mcp/servers/gwrql5ibq2](https://glama.ai/mcp/servers/gwrql5ibq2)。
   - 一位成员承认这是一个**已知限制**，并计划解决。
- **Phoenix Framework 助力 MCP 实现**：一位成员展示了 [Github 上的 MCPheonix](https://github.com/jmanhype/MCPheonix)，这是一个使用 **Elixir 的 Phoenix Framework** 实现的简化版 **Model Context Protocol (MCP) server**。
   - 该实现简化了 **MCP server** 的创建和管理。
- **MCP 助力 Android 调试**：一位成员介绍了 [DroidMind](https://github.com/hyperb1iss/droidmind)，这是一个通过 **ADB** 管理 **Android 设备**的 **MCP server**。
   - 该项目有助于在 AI 控制下调试设备端问题并分析日志。
- **MCP Servers 生成其他 MCP Servers**：一位成员发布了 [mcp-create](https://github.com/tesla0225/mcp-create)，这是一个旨在构建其他 **MCP servers** 的 **MCP server**，支持 **TypeScript**。
   - 该项目包含[一篇解释性文章](https://zenn.dev/tesla/articles/c66bda76c4a523)，详细介绍了其功能以及如何直接执行生成的 **MCP servers**。
- **Handoff 包含完整上下文**：一位成员分享了 [github.com 搜索结果](https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code)，指出在 **OpenAI 的 SDK** 中，默认情况下 **handoff** 会包含整个对话历史。
   - 这涵盖了所有的 system、user 和 assistant 消息。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 在备考方面表现出色**：一位用户报告称，使用 **NotebookLM** 根据学习指南主题进行自测取得了*非常好的效果*，他通过书签将 PDF 拆分并导入到不同的笔记本中。
   - 用户将测试结果转化为其他 App 中的抽认卡（flashcards），以便进一步学习。
- **NotebookLM 生成医疗文档**：一位医疗领域的用户发现 **NotebookLM** 在解析指南和网站以创建患者出院信息方面非常有用。
   - 具体而言，他们为患者创建了一份关于工伤索赔的简洁单页文档。
- **优化 NotebookLM 的数据摄取**：一位用户正在自动化优化上传至 **NotebookLM** 的信息，重点是减小文件体积以便于*机器人摄取（robot ingestion）*。
   - 这简化了他们在 NotebookLM 中处理文档的工作流。
- **Gemini 引发不满**：尽管 **Gemini** 已集成到 Google 生态系统中，但一位用户对其表示不满。
   - 该用户未提及有关其负面体验的具体细节。
- **NotebookLM 处理海量知识库**：一位拥有 **1000 万字知识库**（1500 本书，6000 个视频文本）的用户询问了 **NotebookLM** 的限制。
   - **NLM** 团队的一名成员澄清说，NotebookLM 支持 **1000 万字**，但在 **300 个来源和每个来源 50 万字**的限制内，并利用了 **RAG** 技术。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 邀请用户推荐好友**：**Windsurf Referral Challenge** 激励用户推荐好友，每推荐一位好友订阅 Pro 版本即可获得 **500 flex credits**，并有机会在 **3 月 31 日**前通过 [windsurf.ai/refer](https://windsurf.ai/refer) 赢取定制的 **Airpods Pro Max**。
   - 推荐人数最多者获胜，但所有人在好友订阅后都能获得积分。
- **Codeium 扩展无法读取文件**：Codeium VS Code 扩展聊天（**Claude 3.7 Sonnet**）无法直接从文件夹读取脚本文件，需要用户将文件内容粘贴到聊天框中。
   - 建议用户在 *codeium.com/support* 提交报告，因为从技术上讲这应该是可以运行的。
- **Claude 3.7 Sonnet 在 VS Code 扩展中无法工作**：与 Windsurf 不同，**Claude 3.7 Sonnet Thinking** 模型在 VS Code 扩展中不可用。
   - 用户被告知 **Claude 3.7 Sonnet Thinking** *目前在扩展中不可用*。
- **Codeium 错误导致挂起的请求中止**：用户报告了一个持续存在的错误，导致 Codeium 无法工作，提示信息为 *Codeium: The server aborted pending request*，并提到了来自 *releases.codeiumdata.com* 的下载 URL。
   - 该问题在重启 IDE 和更换不同版本后依然存在，建议用户联系 *vscode@codeium.com*。
- **Windsurf 修复 MCP 和 Sonnet 补丁**：Windsurf 发布了 **v1.4.6 补丁修复**，解决了 **MCP 可靠性**、**3.7 Sonnet 网页搜索**以及**代理设置**问题，详见 [changelog](https://www.codeium.com/changelog)。
   - **Windsurf Previews (Beta)** 现在还允许用户直接在 Cascade 中预览本地运行的网站。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **编译器会“黑”数学运算吗？**：成员们讨论了编译器是否会优化 **PyTorch** 和 **NumPy** 等深度学习框架中的计算，特别是关于复杂方程中运算顺序的问题，例如 *(1/n) (a(c + d) + b)* 与 *a(c/n + d/n) + b/n*。
   - 一位工程师建议添加*额外的括号*以确保系统按预期的顺序执行运算，而另一位工程师则思考了极简代码与显式代码之间的权衡。
- **Claude 3.7 绘制的 Matplotlib 图表令人惊叹**：工程师们对 **Claude 3.7** 生成的 **Matplotlib** 图表感到兴奋，强调 *benchmark 和 svgmaxing* 的表现符合预期。
   - 此次交流中未提供具体链接。
- **自适应元学习（Adaptive Meta-Learning）：是框架还是噱头？**：一位工程师询问 **Adaptive Meta-Learning (AML)** 这一术语是否已经确立，并将其描述为*在线超参数优化 (Online HPO)*与元学习的潜在结合。
   - 另一位工程师分享了 [Semantic Scholar 搜索结果](https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance)，结论是虽然这些关键词常被一起使用，但它们尚未构成一个定义明确的框架。
- **虚拟现实解决监狱危机？？**：根据[这篇文章](https://www.theguardian.com/technology/2025/mar/08/vr-prison-california)，加州的一家女性监狱在禁闭室中使用 **VR 头显**取得了成功，使*违规行为减少了 97% 以上*。
   - 该 VR 项目让参与者观看日常生活场景和旅行冒险，并通过艺术创作来处理他们的情绪。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama Extract 访问权限已获准**：一名成员请求访问 **Llama Extract**，并获得了加入封闭测试 (closed beta) 的机会，目前正等待 `rasmus-persson@outlook.com` 的邮件确认。
   - 未提供关于封闭测试细节的进一步信息。
- **Premium 方案升级变得简单**：一位用户询问如何升级到 **Premium plan**，并收到了登录、点击头像图标并选择升级/管理按钮的说明。
   - 未就 Premium 方案的功能或优势进行进一步讨论或提供细节。
- **API 的 MP3 解析难题**：一位用户报告了通过 API 上传 **.mp3** 文件进行解析时出现错误，并指出通过 UI/网页端上传运行正常。
   - 他们提供了[错误截图](https://cdn.discordapp.com/attachments/1059201661417037995/1349100831307202714/Screenshot_2025-03-11_at_3.24.19_PM.png?ex=67d1df8f&is=67d08e0f&hm=3b980c7dd220c3d654ff1cb17819daedcc6fc3c896b2ed955e800b40f2467d3d)。
- **函数调用 (Function Calling) 对决**：一位成员询问除了 **OpenAI** 之外，还有哪些模型擅长函数调用，希望能找到更便宜的选择。
   - 在提供的上下文中未推荐具体的替代模型。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Judge LLM 遵循 ChainPoll 模式**：成员们正在构建一个遵循 **ChainPoll** 模式的 **Judge LLM**，它会返回平均响应链。
   - 一位成员建议使用 `module.batch()` 或 `dspy.Parallel` 来加速该过程。
- **Best of N 文档查询**：一位成员在查找 **Best of N** 的文档时遇到困难。
   - 该成员指出 ensemble 被列为 teleprompter，并询问它是优化输入程序还是将输入程序聚合为一个最优的单一程序。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OpenPipe 精通演绎推理 (Deductive-Reasoning)**：一位成员分享了 [OpenPipe 的演绎推理项目](https://github.com/openpipe/deductive-reasoning)，强调其使用 **Torchtune** 进行 SOTA 演绎推理模型训练。
   - 该项目展示了 **Torchtune** 在实际、前沿 AI 应用中的能力，特别是在增强模型训练效率和有效性方面。
- **FP8 微调面临阻碍**：成员们探讨了以 **FP8** 格式部署模型的困难，考虑通过 **FP8** 微调来减轻量化误差，但指出 **FP8** 在训练期间存在稳定性问题。
   - 他们建议逐渐增加 **weight decay**，以使权重在 **FP8** 微调期间保持在最佳范围内。
- **Torchtune 对 FP8 的 QAT 探索**：一位成员询问了 **Torchtune 的 QAT (Quantization Aware Training) 支持**，特别是针对 **FP8**，目标是进行微调并减少量化误差。
   - 一个极具前景的 [recipe](https://github.com/pytorch/torchtune/pull/2404) 被认为是 **Torchtune** 内部实现 **FP8** QAT 的潜在解决方案。
- **回归测试讨论显示需要审查**：新增的[回归测试 (regression tests)](https://github.com/pytorch/torchtune/pull/2477) 引发了关于确定模型大小和评估方法的讨论。
   - 成员们质疑仅靠评估是否足够，暗示了围绕更全面的衡量策略进行更深层次对话的必要性。
- **评估有效性得到广泛探讨**：讨论转向了对简单评估之外的衡量策略的需求，成员们辩论了各种评估指标的价值。
   - 这种审议预计将影响关于模型大小和测试方法的决策，推动向更稳健的评估实践转变。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 启动 Expedition Aya 2024！**：**Cohere For AI** 正在启动 [Expedition Aya 2024](https://tinyurl.com/ayaexp2025)，这是一个为期 **6 周的开放构建挑战赛**，专注于**多语言、多模态和高效 AI**。
   - 参与者可以获得 **Cohere API 额度**，奖品包括**限量版 Expedition 周边**以及对顶级项目的认可，启动会议将于 **2025 年 3 月**举行。
- **SemiAnalysis 举办 Blackwell GPU Hackathon！**：[SemiAnalysis](https://semianalysis.com/) 将于 **3 月 16 日星期日**举办 **Nvidia Blackwell GPU Hackathon**，提供对 **Blackwell & PTX 基础设施**的实操探索。
   - 演讲者包括来自 **OpenAI** 的 **Philippe Tillet** 和来自 **TogetherAI** 的 **Tri Dao**，赞助商包括 **Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius**。
- **研究员与多语言社区建立联系**：一位研究员询问了 Cohere Discord 社区内的多语言和多文化活动，表达了对 **Cohere** 工作的欣赏。
   - 鼓励新成员介绍自己，说明所属机构、当前项目、首选技术/工具和社区目标，并遵守**社区期望**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **SemiAnalysis 举办 Nvidia Blackwell GPU Hackathon**：**SemiAnalysis** 将于 **3 月 16 日星期日**举办 [Nvidia Blackwell GPU Hackathon](https://semianalysis.com/hackathon-2025/)，在合作开发开源项目的同时，提供对 **Blackwell & PTX** 基础设施的实操探索。
   - 演讲者包括 [OpenAI 的 Philippe Tillet](https://openai.com/)、[TogetherAI 的 Tri Dao](https://www.together.ai/) 和 [Thinking Machines 的 Horace He](https://www.thinkingmachin.es/)。
- **GTC 启动仪式聚焦 Blackwell GPU**：**SemiAnalysis** 以 **Blackwell GPU Hackathon** 为 **GTC** 拉开序幕，活动包括引人入胜的早间主题演讲、使用强大的 **Blackwell GPU**（如 **GB200s**）进行全天 Hacking，以及富有洞察力的下午演讲。
   - 该活动由 Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave 和 Nebius 赞助。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **期待 CUDA 博客文章**：一位用户正在等待关于 **CUDA** 的新博客文章发布。
   - 未提供额外信息。
- **对 CUDA 更新的期待升温**：随着用户热切期待有关 **CUDA** 的最新更新和博客文章，热情不断高涨。
   - 社区渴望探索 CUDA 的新功能和改进，尽管具体细节仍处于保密状态。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Qdrant 被移出 ConvRAG**：开发团队曾考虑将 **Qdrant** 作为其 **ConvRAG** 的向量数据库，但出于未指明的原因决定使用另一个数据库。
   - 所选的数据库为 **VPC 部署**提供了更大的灵活性。
- **ConvRAG 选择替代数据库**：为 **ConvRAG** 选择了另一个向量数据库而非 Qdrant。
   - 引用的主要原因是它在 **VPC 部署**场景中提供了增强的灵活性。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

{% if medium == 'web' %}

# 第 2 部分：按频道的详细摘要和链接

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1348973754817974374)** (1048 条消息 🔥🔥🔥): 

> `Cursor Nightly Bug, Claude 3.7 定价, Manus AI 的潜力, 对 Cursor 稳定性的批评, 本地 LLM vs 云端 GPU`

- ****Nightly 版本魔咒袭击 Cursor****：最新的 Nightly 更新破坏了 **AI Chat** 和 **Cursor 设置**，用户报告 GUI 损坏，导致无法打开聊天面板或设置。
   - 一位用户建议删除并重新安装应用，但发现该问题在最新的 Nightly 更新中依然存在。
- ****Claude 3.7 定价引发轩然大波****：用户对 **Claude 3.7 Thinking** 的新定价表示不满，现在每次请求消耗 **2 次额度而非 1 次**，导致部分用户考虑取消订阅并转向 **Roo Cline** 等替代方案。
   - 讨论还涉及了在启用大上下文（large context）的情况下使用 **Claude 3.7 Thinking** 的潜在成本，估计每次请求可能达到 **16 美分**。
- ****Manus AI 吸引观众目光****：一位用户分享了 **Manus AI** 的链接，称其为他们见过的“最疯狂的 AI Agent”，能够完成克隆苹果官网等任务。
   - 一些用户表示怀疑，认为它可能只是集成了某些 PC 操作工具的 **Sonnet 3.7**，而另一些用户则展望了未来由多个 Agent 运行公司的前景。
- ****Cursor 的稳定性饱受质疑****：多位用户报告 **Cursor 今天几乎无法工作**，经常卡顿或无响应，一位用户指出 **Claude Max** 对他们来说完全无法运行。
   - 一些用户发现回退到 **.46.11** 版本可以解决问题，而另一些用户则推测 **.47** 版本仅限部分用户使用。
- ****本地 LLM 与云端 GPU 之争引发辩论****：一位用户提议购买配备 **512GB RAM 的 M3 Ultra Mac Studio** 以在本地运行 **完整版 DeepSeek R1**，这引发了关于此类配置的性价比和实用性的辩论。
   - 一些用户认为从云服务商租赁 **GPU** 可以提供更快的推理速度，且长期来看更具成本效益，而另一些用户则强调了拥有一台本地 LLM 机器处理其他任务的好处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com`">未找到标题</a>：未找到描述</li><li><a href="https://cursor.directory/mcp">Cursor Directory</a>：为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://docs.cursor.com/settings/beta">Cursor – Early Access Program</a>：未找到描述</li><li><a href="https://build-launch-win.lovable.app/">lovable x anthropic global hackathon</a>：未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1j8y05c/ama_with_cursor_devs_march_11_2025/">Reddit - 深入探讨一切</a>：未找到描述</li><li><a href="https://x.com/prajwaltomar_/status/1899104347532738764?s=46&t=kUuVqsG2GMX14zvB592G5w">Prajwal Tomar (@PrajwalTomar_) 的推文</a>：如何以正确的方式设置 Cursor。Cursor Rules 已过时，Project Rules 才是现在的正确方式。以下是其重要性及如何正确设置的方法：</li><li><a href="https://x.com/ehuanglu/status/1899110687902978373?s=46&t=CLGnxOi5OPp22iT8UYkr1A">el.cine (@EHuanglu) 的推文</a>：Manus AI 比 DeepSeek 时刻还要疯狂，我刚拿到邀请码，这东西是我见过的最疯狂的 AI Agent。10 个案例：1. 克隆苹果官网，它创建了一个苹果官网的副本，看起来...</li><li><a href="https://x.com/prajwaltomar_/status/1899104347532738764?s=46&t=kUuVqsG2GMX1">Prajwal Tomar (@PrajwalTomar_) 的推文</a>：如何以正确的方式设置 Cursor。Cursor Rules 已过时，Project Rules 才是现在的正确方式。以下是其重要性及如何正确设置的方法：</li><li><a href="https://x.com/16footcatgirl/status/1899416927472103770?s=46">Syl (e/tard) (@16footcatgirl) 的推文</a>：LMAOOOOOOO</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: 没有堡垒，纯粹的开放阵地。OpenManus 即将到来。</a>：没有堡垒，纯粹的开放阵地。OpenManus 即将到来。 - mannaandpoem/OpenManus</li><li><a href="https://youtu.be/etXFdqPu1Wk?si=IV-gAIcqn4X0FZTT">RTX 5090 运行《赛博朋克 2077》搭配全新 DreamPunk 3.0 画质 - 8K 超写实模组展示</a>：在《赛博朋克 2077》中使用全新的 RTX 5090 和 DreamPunk 3.0 进行极致画面展示和游戏实测。通过路径追踪、DLSS 4 等技术实现画面提升...</li><li><a href="https://tenor.com/view/cool-fun-white-cat-dance-cool-and-fun-times-gif-9061261248949225544">Cool Fun GIF - 酷炫有趣的白猫 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/swinging-hanging-hanging-in-there-bored-waiting-gif-18029735056767784446">Swinging Hanging GIF - 摇摆悬挂 坚持住 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>：通过在 GitHub 上创建账号来为 nikmcfly/ANUS 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1349124326980325437)** (1 条消息): 

> `Perplexity Windows 应用, 语音听写, 键盘快捷键` 


- **Perplexity 发布 Windows 应用**：Perplexity AI 发布了官方 **PC 桌面应用**，支持 **语音听写**、**键盘快捷键**以及访问最新模型，可在 [perplexity.ai/platforms](https://www.perplexity.ai/platforms) 下载。
- **Perplexity Windows 应用功能**：该应用允许用户利用 **语音听写** 和 **键盘快捷键**，更高效地与 Perplexity 的最新模型进行交互。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1348975860056592455)** (277 条消息🔥🔥): 

> `Revolut Perplexity Pro 优惠码问题, Claude 3.7 Token 限制, Perplexity Enterprise 集成, Sider AI 功能与对比, Perplexity Windows 应用` 


- **Revolut 用户面临优惠码难题**：用户在兑换通过 **Revolut** 收到的 **Perplexity Pro** 优惠码时遇到问题，部分用户被告知需要创建新的 Perplexity 账户，而另一部分用户则被引导联系 Revolut 获取新代码。
   - 一位用户报告称：*"我联系了 Revolut，他们说我需要注册一个新的 Perplexity 账号。这很令人沮丧，但嘿，我想还是值得的。"*
- **Claude 3.7 的 5K 输出 Token 限制**：用户发现 **Claude 3.7** 在 Perplexity 上的输出限制被硬性设定为 **5000 Tokens**。
   - 在 Anthropic 官网上，官方说明其输出最高可达 **128K**。
- **大学关注 Perplexity Enterprise**：一位用户正在探索将 **Perplexity Enterprise** 集成到大学系统中，重点关注其连接内部知识库的能力，以此作为政策和流程的中心枢纽。
   - [Perplexity Enterprise FAQ](https://www.perplexity.ai/enterprise) 重点介绍了内部数据搜索和自定义工作区的功能。
- **Sider AI 是我们需要的黑马吗？**：用户们正在热议 **Sider AI**，称赞其精致的 UI 和深度研究能力，包括访问 **Claude 3.7 Sonnet** 和其他模型进行创意写作、翻译和研究。
   - 联网访问功能是一项付费功能。
- **Perplexity Windows 应用：披着 Chrome 外衣的狼？**：新发布的 **Perplexity Windows 应用** 正受到质疑，用户指出它本质上只是网页版的套壳，缺乏桌面端应有的优势。
   - 正如一位用户所说：*"它只是一个被阉割的浏览器，而且没有像 Complexity 这样的浏览器扩展。"*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.getmerlin.in/chat/share/ba3db527-7190-4c1b-be30-3de12e4abacd">Claude 3.5 和 3.7 的 API 上下文大小</a>：由 Anonymous 分享 - 2025年3月11日</li><li><a href="https://www.ndtv.com/world-news/twitter-cyberattack-elon-musk-what-is-dark-storm-pro-palestine-group-allegedly-behind-x-cyberattack-7897600">什么是 Dark Storm，据称是 X 网络攻击背后的亲巴勒斯坦组织</a>：亲巴勒斯坦黑客组织 Dark Storm 声称对周一攻击 X（原 Twitter）的事件负责。</li><li><a href="https://www.instagramez.com/reel/DHEOQdYo2cj">下载 Instagram 视频、Reels 和图片</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1348975775310938174)** (8 条消息🔥): 

> `学生破解数学难题, Air 产品预告, 特朗普与加拿大关税, Apple 软件大改版, 曼联球场计划` 


- **学生破解百年数学难题**：一个 [Perplexity 页面](https://www.perplexity.ai/page/student-cracks-century-old-mat-wN0I7t44Q8qaalwydsOpEw) 报道了一名学生解决了一个世纪之久的数学难题。
   - 未提供更多细节。
- **Apple Air 产品预告**：一个 [Perplexity 页面](https://www.perplexity.ai/page/apple-air-product-teased-QhTieZlcTwWodiMLzGzP3g) 预告了一款新的 Apple Air 产品。
   - 未提供更多细节。
- **特朗普升级对加拿大关税**：一个 [Perplexity 页面](https://www.perplexity.ai/page/trump-escalates-canada-tariffs-WHAJemCtRWOmiP0XdAngxw) 讨论了特朗普升级对加拿大的关税。
   - 未提供更多细节。
- **Apple 重大软件改版**：一个 [Perplexity 页面](https://www.perplexity.ai/page/apple-s-major-software-overhau-MgD9Y63fTnKAJvQ7Zb.bJ) 报道了 Apple 的一次重大软件改版。
   - 未提供更多细节。
- **曼联球场计划**：一个 [Perplexity 页面](https://www.perplexity.ai/page/manchester-united-s-stadium-pl-7kp06uj2RDeXXufgr1FkxA) 详细介绍了曼彻斯特联队的球场计划。
   - 未提供更多细节。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1349004704532926484)** (6 条消息): 

> `API chat completions 截断，Tier-3 结构化输出` 


- **API chat completions 遭遇截断**：一名成员报告称，在使用 **sonar-reasoning-pro 模型**调用 chat.completions API 时，遇到了间歇性的内容截断问题。
   - 该成员指出，[Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions) 能够稳定输出完整回复，且增加 **max_token** 配额并未解决该问题。
- **Tier-3 结构化输出咨询**：一名成员询问 **sonar-deep-research API** 是否为 Tier-3 用户提供结构化输出功能，还是仅在其他模型中可用。



**提到的链接**：<a href="https://docs.perplexity.ai/api-reference/chat-completions)">未找到标题</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1348975074237218869)** (232 条消息🔥🔥): 

> `GRPO trainer batch size and gradient accumulation, reward hacking, finetuning llama 8b 4bit vs normal, high context window and batch size impact, date and time extraction regex vs ai` 


- **探索 GRPOTrainer Batch Size 的影响**：一位用户询问 **batch size** 和 **gradient accumulation steps** 如何影响 **GRPOTrainer** 的总训练时间。
   - 这一询问表明了在使用 GRPOTrainer 时优化训练效率的兴趣，特别是考虑到 [大型推理模型的复杂性](https://x.com/__nmca__/status/1899174075685355770) 以及最近 OpenAI 监控论文中指出的 reward hacking。
- **正则表达式（Regex）在日期处理上的救场**：一位用户打算训练一个模型从查询中提取正确的营业时间，但被建议对于这项任务，传统的 ML 甚至 **regex 系统** 可能比使用 AI 更合适。
   - 另一位成员链接了一个相关的 [xkcd 漫画](https://xkcd.com/208/)，内容是关于 **用复杂的方案过度设计简单的任务**。
- **Reka Flash 3 - 新的竞争者出现！**：[Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3) 已发布，这是一个采用 Apache 2.0 许可证的 **21B 推理模型**，可与 **QwQ-32B** 和 **o1-mini** 媲美。
   - 尽管对该模型的架构和训练数据存在疑问，一些用户指出 Reka 团队由前 DeepMind 员工组成，且 [Reka 官网](https://www.reka.ai/ourmodels) 声明 Flash 模型是多模态的。
- **硬件难题：办公室寻求推理与微调设备**：一位用户就 **推理和微调设备** 的硬件规格征求建议，POC 模型为 **phi-4**，下一步可能是 **Mistral Small**。
   - 讨论涵盖了 VRAM 需求（小型模型 **24GB**，中型模型 **48GB**），一位成员建议 **2 块 3090** (48GB VRAM)、vLLM 和 AWQ 可以解决大部分推理问题。
- **寻求理智：寻找无审查模型**：一位用户正在寻找适用于深奥哲学研究的 **无审查推理模型**，既要避开伦理限制，又要避免 NSFW 内容。
   - 建议包括尝试 [R1 1776](https://huggingface.co/unsloth/r1-1776-GGUF)，以及使用 DeepSeek API 或 Mistral API 进行越狱（jailbreaking）的建议，并提到了使用 Unsloth 调优的 [Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2](https://huggingface.co/Orenguteng)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://xkcd.com/208/">Regular Expressions</a>：未找到描述</li><li><a href="https://huggingface.co/RekaAI/reka-flash-3">RekaAI/reka-flash-3 · Hugging Face</a>：未找到描述</li><li><a href="https://www.reka.ai/ourmodels">Models | Reka</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Orenguteng">Orenguteng (Orenguteng)</a>：未找到描述</li><li><a href="https://regex101.com/">regex101: build, test, and debug regex</a>：带有语法高亮、解释、PHP/PCRE、Python、GO、JavaScript、Java、C#/.NET、Rust 备忘单的正则表达式测试器。</li><li><a href="https://huggingface.co/RekaAI/reka-flash-3/blob/main/tokenizer_config.json#L201">tokenizer_config.json · RekaAI/reka-flash-3 at main</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=6t2zv4QXd6c">Faster AI with minimal compute power - Unsloth’s open source AI story | GitHub Accelerator</a>：Unsloth 正在通过以极低的算力简化模型训练来改变 AI 的可访问性。在这段视频中，Unsloth 的创始人 Daniel 和 Michael...</li><li><a href="https://x.com/__nmca__/status/1899174075685355770">Tweet from Nat McAleese (@__nmca__)</a>：大型推理模型非常擅长 reward hacking。来自 OpenAI 最近监控论文的示例线程：(0/n)
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1349133160163573841)** (2 messages): 

> `AI Code Fusion Tool, Code Optimization for LLMs` 


- **AI Code Fusion 工具亮相**：一名成员介绍了 **AI Code Fusion**，这是一个旨在通过打包文件、统计 token 和过滤内容来为 **LLM contexts** 优化代码的工具，可在 [GitHub](https://github.com/codingworkflow/ai-code-fusion) 上获取。
- **工具寻求反馈**：**AI Code Fusion** 的创建者正寻求社区对该工具的反馈。



**提及的链接**：<a href="https://github.com/codingworkflow/ai-code-fusion">GitHub - codingworkflow/ai-code-fusion: Desktop app to process repository content into one file</a>：将代码库内容处理为单个文件的桌面应用 - codingworkflow/ai-code-fusion

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1348980773062971423)** (33 messages🔥): 

> `Qwen2.5, GRPO RL algo, Deepseek distilled models, multi-node multi-GPU, Llama-3.2 1B colab notebook` 


- **GRPO RL Batch 机制**：GRPO (Guided Reinforcement Policy Optimization) 的 batch size 必须与 generations 的数量相同。用户询问了对于 GRPO RL 算法，`num of generation` 过多或过少的影响。
   - 建议 `num generations` 的范围在 **4 到 8** 之间；增加 batch size 倍数会缩短训练时间，但会大幅增加 GPU 显存需求。
- **多 GPU 训练技巧**：一名成员寻求使用 Unsloth 在多节点和多 GPU 上微调大型模型的帮助。
   - 另一名成员建议使用 **axolotl** 或 **llama factory** 进行多 GPU 训练，因为 Unsloth 目前尚未（正式）支持多 GPU，尽管相关支持可能会在未来几周内推出。
- **Llama-3.2 1B Colab Bug 已修复**：一名成员在尝试 Unsloth 的 Llama-3.2 1B colab notebook 时遇到错误并上传了截图。
   - 另一名成员建议重启 kernel 并优先运行该单元格，因为这听起来像是运行顺序错乱导致的。
- **释放 QwQ-32B 潜力**：一名用户询问如何将 [QwQ-32B-unsloth-bnb-4bit](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit) 转换为 `gguf`。
   - 有关修复无限生成问题以及如何运行 QwQ-32B 的说明，请参阅 [Unsloth Tutorial](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively)。
- **Gemma Embedding 谜团**：一名用户使用 LoRA 在新 token 上微调了 **Gemma 2B** (unsloth/gemma-2b-bnb-4bit)，并反馈在合并（merge）后，模型无法识别新添加的 token。
   - 他们推测这是因为 Gemma 具有 tied embeddings，因此 PEFT 模型创建了两个独立的权重矩阵（一个用于 embeddings，一个用于 lm_head），而在合并后，`lm_head` 权重矩阵不再存在。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit">unsloth/QwQ-32B-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1964">How to finetune QWQ32B with 24g VRAM · Issue #1964 · unslothai/unsloth</a>：在使用 Qwen2.5 grpo 的 notebook 并将模型更改为 QWQ32B 进行微调时，遇到了以下问题。==((====))== Unsloth 2025.2.4: Fast Qwen2 patching. Transformers: 4.48.3. \...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1348989004942409800)** (3 messages): 

> `train_on_responses_only Functionality, HuggingFace Space for Unsloth, HF model sharing` 


- **Unsloth 代码启发适配工作**：一名成员提到他们将从[此处](https://link.to.project)发现的一个*非常酷的项目*中适配大量代码。
   - 该用户未具体说明是哪个项目。
- **HuggingFace Space 测试 Train-on-Responses-Only**：一名成员创建了一个微型的 **HuggingFace space**，用于在各种模型上测试 **Unsloth** 的 `train_on_responses_only` 功能。
   - 用户可以指定任何 **HF model** 并检查 `train_on_responses_only` 函数是否按预期工作。
- **通过 URL 分享 HF 模型代码片段**：一名成员使该应用能够接收 **URL** 中的所有输入作为查询参数，从而可以轻松分享编码后的代码片段。
   - 分享的示例 URL 包括 [Qwen2-VL-7B-Instruct-Multi](https://tinyurl.com/Qwen2-VL-7B-Instruct-Multi), [phi-4-unsloth-bnb-4bit](https://tinyurl.com/phi-4-unsloth-bnb-4bit), 和 [Llama-32-1B-Instruct](https://tinyurl.com/Llama-32-1B-Instruct)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1348985968497004585)** (4 条消息): 

> `GRPO RL algorithm, Unsloth GRPO, Deepseek distilled models` 


- **GRPO RL 生成数量：太高还是太低？**：一位成员询问了为 **GRPO RL algorithm** 使用过多或过少生成数量的影响。
   - 他们还想知道 **Unsloth GRPO** 中要求 batch size 等于生成数量的规定会如何影响训练时间和性能。
- **Deepseek Distilled Models 进行 GRPO 训练**：一位成员提出了关于在 **Deepseek distilled models**（这类模型天然会产生长输出）上对 **GRPO** 使用 **max_completion limit** 的效果问题。
   - 他们还询问了 SMILES 格式验证，并建议使用两个奖励函数：一个用于验证 **SMILES format**，另一个用于对属性的正确性进行评分，并可能在 prompt 中加入 few-shot 示例。

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1348974256725164102)** (203 条消息🔥🔥): 

> `Deep Hermes, 用于网页抓取的 Browserless.io, Universal Transformers, Nvidia Blackwell GPU Hackathon, Forward Gradients` 


- **具备早期推理能力的 Deep Hermes 预览版发布**：一款新的 **Deep Hermes** 模型已经发布，它具备*早期推理能力*，是从 R1 蒸馏而来的，详见 [Hugging Face](https://huggingface.co/)。
   - 该模型将接受测试，但成员们对其可能超过上下文长度（context length）表示担忧。
- **使用 Browserless.io 绕过机器人检测**：一位成员推荐使用 [Browserless.io](https://www.browserless.io) 来绕过网页抓取中的机器人检测和 CAPTCHAs，并强调了其*避免留下细微指纹*的能力。
   - 它支持使用 **Puppeteer** 或 **Playwright** 进行浏览器自动化，并提供了一个用于测试和调试的抓取 IDE。
- **SemiAnalysis 举办 Nvidia Blackwell GPU Hackathon**：[SemiAnalysis](https://semianalysis.com/hackathon-2025/) 将于 3 月 16 日举办 **Nvidia Blackwell GPU Hackathon**，活动包括对 Blackwell 和 PTX 基础设施的亲身探索，演讲嘉宾来自 **OpenAI**、**TogetherAI** 和 **Thinking Machines**。
   - 该活动由 **Together**、**Lambda**、**Google Cloud**、**Nvidia** 和 **OpenAI** 等公司赞助。
- **探索用于 Universal Transformers 的 Forward Gradients**：成员们讨论了使用 [forward gradients](https://arxiv.org/abs/2202.08587) 来优化 **Universal Transformer (UT)** 训练，因为在 UT 中由于存在共享层，这种方法可能更有效。
   - 当与 **N-GPT** 结合使用时，这可能会非常有趣。
- **字节跳动发布 Trae，一款新的 AI 驱动 IDE**：字节跳动发布了 [Trae](https://www.trae.ai/)，这是一款类似于 Cursor 的免费 AI IDE，提供 **Claude Sonnet 3.7** 供免费使用，目前支持 **Mac** 和 **Windows**。
   - Linux 版本正在计划中，该 IDE 的目标用户是 AI 编程初学者。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: 我们提出了一种新型神经网络架构，即在超球面上进行表示学习的归一化 Transformer (nGPT)。在 nGPT 中，所有构成 embeddings、MLP、注意力矩阵的向量...</li><li><a href="https://arxiv.org/abs/2202.08587">Gradients without Backpropagation</a>: 使用反向传播来计算用于优化的目标函数梯度一直是机器学习的支柱。反向传播（或反向模式微分）是一个特例...</li><li><a href="https://arxiv.org/abs/2408.10419">Second-Order Forward-Mode Automatic Differentiation for Optimization</a>: 本文介绍了一种二阶超平面搜索，这是一种新型优化步骤，它将二阶线搜索从直线推广到 $k$ 维超平面。这与前向模式结合...</li><li><a href="https://www.browserless.io">Browserless - Browser Automation and Dodge Bot Detectors</a>: 为您的抓取或自动化绕过任何机器人检测。立即免费注册，使用我们的 API、代理和验证码识别。</li><li><a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis 将在 NVIDIA GTC 之前拉开帷幕！以精彩的早间主题演讲开始新的一天，全天进行底层 NVIDIA GPU 编程（甚至可能是 Blackwell），稍作休息...</li><li><a href="https://www.youtube.com/watch?v=_V6FI36yKTs">Trae AI: This FREE AI Coding Agent is INSANE 🤯</a>: 🚀 立即获取免费 SEO 策略会议 + 折扣：https://go.juliangoldie.com/strategy-session 想要获得更多客户、赚取更多利润并节省数百小时...</li><li><a href="https://github.com/orobix/fwdgrad">GitHub - orobix/fwdgrad: Implementation of &quot;Gradients without backpropagation&quot; paper (https://arxiv.org/abs/2202.08587) using functorch</a>: 使用 functorch 实现 &quot;Gradients without backpropagation&quot; 论文 (https://arxiv.org/abs/2202.08587) - orobix/fwdgrad</li><li><a href="https://github.com/UbiquitousLearning/Backpropagation_Free_Training_Survey">GitHub - UbiquitousLearning/Backpropagation_Free_Training_Survey</a>: 通过在 GitHub 上创建账号来为 UbiquitousLearning/Backpropagation_Free_Training_Survey 做出贡献。</li><li><a href="https://www.trae.ai/">Trae - Ship Faster with Trae</a>: Trae 是一款自适应 AI IDE，它改变了您的工作方式，与您协作以实现更快的运行。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1349010022671192085)** (32 messages🔥): 

> `Evaluating small language models, Output formatting challenges, MATH benchmark and finetuning, lm-eval harness, Open source dLLMs` 


- **小型语言模型的评估面临输出格式失误**：成员们讨论了使用 OAI 的 simple evals 评估小型语言模型（~8B）的情况，指出模型经常在输出格式上挣扎，由于代码逻辑无法从答案轨迹中提取出解决方案，从而阻碍了性能表现。
   - 有人指出，对于*不擅长遵循指令的模型，答案提取是非常不可靠的*，这会影响评估结果。
- **对数似然（Loglikelihood）评估解放了模型**：对于多选题解答（**MCQA**）任务，一位成员建议改用**基于对数似然的评估**，从而绕过对严格输出格式的需求。
   - 这解释了为什么 instruct 模型能答对一些问题，而它们的 chat 变体通常得分却为 **0**。
- **MATH 基准测试让模型精通数学**：讨论涉及了 **MATH benchmark** 的使用，一位成员指出，模型通常会针对该基准测试进行*微调，以按照建议的格式输出*。
   - 人们*将数学奉为智力推理的理想化身*。
- **lm-eval Harness 助力处理输出**：提到了 **lm-eval language harness** 作为一个可以复现 GPQA 结果的工具，提供了一种处理输出和评估指标的方法。
   - 它具有*易于使用的功能，可以修改 prompting、评估指标和输出过滤*。
- **对高质量量化模型的需求加速**：一位成员询问了关于开源 **dLLMs**（深度量化语言模型）或其他可以在小于 4 GB VRAM 上运行的优秀 **LLMs**（如 **Mercury Coder**）。
   - 目前还没有人回答这个问题。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1349071326576509049)** (103 messages🔥🔥): 

> `spectral autoregression, Gaussian noise, noise schedules, neural flow diffusion models, variational rectified flow` 


- **扩散模型执行谱自回归**：一篇博客文章（[Spectral Autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)）揭示了**图像扩散模型在频域执行近似自回归**。
   - 作者指出，这一理论虽然直观，但在实践中预测能力有限，特别是当使用与目标分布的 RAPSD 匹配的有色噪声时。
- **神经流扩散模型增强了标准高斯分布**：**Neural Flow Diffusion Models (NFDM)** 通过支持比标准 Gaussian 噪声更广泛的前向过程，增强了扩散模型，并具有端到端、无模拟的优化目标。
   - 根据[论文](https://arxiv.org/abs/2404.12940)，实验证明了 **NFDM** 的强大性能和最先进的似然估计。
- **通过“负面引导”避免模式崩溃**：一篇[论文](https://arxiv.org/abs/2406.02507)建议引导模型远离“坏结果”（badness）而不是远离“无条件状态”，以避免 CFG（classifier-free guidance）的模式崩溃（mode dropping）。
   - 该方法实现了对图像质量的解耦控制，且不损害变体数量，在 ImageNet 上 64x64 分辨率下获得了 **1.01** 的记录级 FID，512x512 分辨率下为 **1.25**。
- **寻求噪声调度的理论基础**：一位成员专注于建立噪声调度的理论基础，特别是希望每一个积分步骤都能对去噪过程做出均匀的贡献。
   - 他们注意到，目前早期的步骤可能会停滞不前，并引用了 [@cloneofsimo 的 Twitter 帖子](https://x.com/cloneofsimo/status/1894086577632284975)，指出*整个轨迹中 99.8% 的潜变量可以用前两个主成分来解释*。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2206.13397">Generative Modelling With Inverse Heat Dissipation</a>: 虽然 Diffusion 模型在图像生成方面取得了巨大成功，但其噪声反转生成过程并未明确考虑图像的结构，例如其固有的多尺度...</li><li><a href="https://arxiv.org/abs/2310.02557">Generalization in diffusion models arises from geometry-adaptive harmonic representations</a>: 为图像去噪训练的深度神经网络 (DNN) 能够通过基于得分的反向 Diffusion 算法生成高质量样本。这些令人印象深刻的能力似乎暗示了...</li><li><a href="https://arxiv.org/abs/2404.12940">Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling</a>: 传统的 Diffusion 模型通常依赖于固定的前向过程，这隐含地定义了潜在变量上复杂的边缘分布。这往往会使反向过程变得复杂...</li><li><a href="https://sander.ai/2024/09/02/spectral-autoregression.html">Diffusion is spectral autoregression</a>: 深入探讨图像 Diffusion 模型的频谱分析，揭示它们如何在频域中隐式地执行某种形式的自回归。</li><li><a href="https://arxiv.org/abs/2411.10510">SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers</a>: Diffusion Transformers (DiT) 已成为各种任务（包括图像、视频和语音合成）的强大生成模型。然而，它们的推理过程在计算上仍然昂贵...</li><li><a href="https://arxiv.org/abs/2503.06923">From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers</a>: Diffusion Transformers (DiT) 彻底改变了高保真图像和视频合成，但其计算需求对于实时应用来说仍然过高。为了解决这个问题，特征...</li><li><a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: 图像生成 Diffusion 模型的主要关注点是图像质量、结果的变化量，以及结果与给定条件（例如类别标签）的对齐程度...</li><li><a href="https://github.com/ali-vilab/TeaCache?tab=readme-ov-file">GitHub - ali-vilab/TeaCache: Timestep Embedding Tells: It&#39;s Time to Cache for Video Diffusion Model</a>: Timestep Embedding Tells: It&#39;s Time to Cache for Video Diffusion Model - ali-vilab/TeaCache</li><li><a href="https://x.com/cloneofsimo/status/1894086577632284975">Tweet from Simo Ryu (@cloneofsimo)</a>: 惊人的数字。如果你绘制非随机 Diffusion 采样的轨迹，整个轨迹中 99.8% 的潜在变量可以用前两个主成分来解释。粗略地说，你的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1349058581776961590)** (1 messages): 

> `Patching 结果的指标、Tokenizer 问题、评估 Math CoT 的重要电路` 


- **关于评估 Patching 结果的合适指标展开了辩论**：一位成员正在寻求关于选择正确**指标**的建议，以评估分析 **Math CoT** 答案的重要电路时的 Patching 结果。
   - 当 Tokenizer 将数字拆分为多个 Token 时，问题就出现了，这使得应用给定的计算 Patching 效果的方程变得复杂（附有方程）。
- **Tokenizer 问题使 Patching 评估变得复杂**：核心问题是 Tokenizer 将诸如 **10** 和 **15** 之类的数字拆分为两个 Token，破坏了评估方程的直接应用。
   - 该成员正在考虑是对每个答案 Token 的方程求平均值，还是在代入方程之前合并它们的概率。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435)** (57 messages🔥🔥): 

> `Avoma, Factorio 学习环境, Contextual AI Reranker, OpenAI Agents API, Luma Labs IMM`

- ****Avoma** 被视为 **Gong** 的竞争对手**：一位成员分享了 [Avoma](https://www.avoma.com/)，这是一个集自动记录笔记、排程、辅导和预测于一体的全方位 **AI platform**，被认为是 **Gong** 的竞争对手。
- ****Factorio Learning Environment** 基准测试 LLMs**：**Factorio Learning Environment (FLE)**（[可在 GitHub 上获取](https://jackhopkins.github.io/factorio-learning-environment/)）旨在利用游戏 **Factorio** 测试 **Agent** 在长期规划、程序合成和资源优化方面的能力。
   - 一位成员表达了兴奋之情，并幽默地请求立即去 *Anthropic Factorio lab* 工作，并强调该环境目前仅支持文本，但可以从像 **Qwen 2.5 VLM** 这样的多模态数据输入中获益。
- ****Contextual AI** 发布遵循指令的 Reranker**：**Contextual AI** 推出了 [一款新的 reranker](https://contextual.ai/blog/introducing-instruction-following-reranker/)，它可以遵循自定义指令，根据新鲜度、文档类型或来源等要求对检索结果进行排序。
- ****OpenAI** 构建 Agent 的新工具现已发布**：**OpenAI** 推出了用于构建 **Agent** 的新工具，包括 [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses)、[Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search)、[Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use) 和 [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search)，以及一个集成了 [Observability Tools](https://platform.openai.com/docs/guides/agents#orchestration) 的全新开源 [Agents SDK](https://platform.openai.com/docs/guides/agents)。
   - 此次发布伴随着直播和 AMA，并讨论了从“消息列表输入-单条消息输出”到“项目列表输入-项目列表输出”的转变，新的 **SDK** 已达到生产就绪水平，并增加了追踪 (tracing)、护栏 (guardrails) 和生命周期事件等功能。
- ****Luma Labs** 推出 Inductive Moment Matching (IMM)**：**Luma Labs** 发布了 [Inductive Moment Matching (IMM)](https://lumalabs.ai/news/inductive-moment-matching)，这是一种全新的预训练技术，声称与 **diffusion models** 相比，能以 10 倍的效率提供更优的样本质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://jackhopkins.github.io/factorio-learning-environment/">Factorio Learning Environment</a>：未找到描述</li><li><a href="https://www.latent.space/p/openai-agents-platform">⚡️全新的 OpenAI Agents 平台</a>：来自 OpenAI 的 Nikunj Handa 和 Romain Huet 加入我们，预览他们全新的 Agents API：Responses、Web Search 和 Computer Use，以及全新的 Agents SDK。</li><li><a href="https://x.com/harvey__ai/status/1899491666429632907?s=46">来自 Harvey (@harvey__ai) 的推文</a>：介绍 Harvey Agents：</li><li><a href="https://x.com/skcd42/status/1899515665217683487?s=46">来自 skcd (@skcd42) 的推文</a>：如果 Agent 的活动变得更加直观且易于理解会怎样？</li><li><a href="https://t.co/N7GbF78bgw">在 Harvey 中引入 Agents</a>：推出旨在与专业人士协作的 Agentic 工作流，以交付精确、定制化的工作成果。</li><li><a href="https://x.com/patio11/status/1899587413011321324">来自 Patrick McKenzie (@patio11) 的推文</a>：数学在这里讲述着一个故事，它仅仅是一个故事，但它比几乎所有人类在被要求描述作为数学在被学习过程中的主观体验时写出的故事都要好……</li><li><a href="https://x.com/douwekiela/status/1899490844572577958">来自 Douwe Kiela (@douwekiela) 的推文</a>：AI 在处理混乱、冲突、不断变化的数据时表现挣扎。由于缺乏人类指导，当今的 AI 排名方法无法清晰地确定优先级。介绍全球首个 instruction-f...</li><li><a href="https://www.youtube.com/watch?v=hciNKcLwSes">使用 API 构建 Agent 的新工具</a>：我们正在升级 API 平台，使开发者能够更快速、更轻松地构建 Agent。Kevin Weil, Nikunj Handa, Steve Coffey 和 Ilan Bigio 介绍……</li><li><a href="https://www.avoma.com/">Avoma — 用于笔记、日程安排和辅导的 AI 平台</a>：使用 Avoma 的全方位 AI 平台加速您的增长：自动化笔记记录、日程安排、通话辅导、CRM 更新等。只需为您需要的服务付费。</li><li><a href="https://x.com/ilanbigio/status/1899510911825756412?s=46">来自 ilan bigio (@ilanbigio) 的推文</a>：swarm -> agents sdk 是我作为“家长”的自豪时刻。@_rohanmehta @stevenheidel 在使其达到生产级并添加大量新功能方面做得非常出色——包括 tracing、guardrails、lifecycle events 等……</li><li><a href="https://x.com/athyuttamre/status/1899541497760067795?s=46">来自 Atty Eleti (@athyuttamre) 的推文</a>：还有更多微小但重要的细节，但这个推文串已经太长了：SDK 拥有 `response.output_text` 可以快速获取文本！`n` choices 已移除；不再有 `completion.choices[0].message`！`finish_...`</li><li><a href="https://x.com/OpenAI/status/1899476049584599462">来自 OpenAI (@OpenAI) 的推文</a>：这是给开发者的。太平洋时间上午 10 点直播。</li><li><a href="https://x.com/OpenAIDevs/status/1899502117171155002">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：直播结束后在这里加入我们的 AMA！从太平洋时间上午 10:30 到 11:30，今天发布产品背后的团队将回答您的问题。在下方回复您的问题。引用 OpenAI (@OpenAI) 的 Agent Tools for ...</li><li><a href="https://x.com/ilanbigio/status/1899517935728709922?s=46">来自 ilan bigio (@ilanbigio) 的推文</a>：看看我们的 @openai CUA 入门应用 - 5 分钟内即可在本地运行！我们为以下环境添加了示例：本地浏览器 (playwright)、容器化桌面 (docker)、远程浏览器和桌面 (@scrapy...</li><li><a href="https://x.com/athyuttamre/status/1899541471532867821">来自 Atty Eleti (@athyuttamre) 的推文</a>：介绍 Responses API：OpenAI API 的新原语。它是设计 OpenAI API 两年经验的结晶，也是我们构建 Agent 下一章节的基石。🧵H...</li><li><a href="https://x.com/athyuttamre/status/1899541496401133838?s=46">来自 Atty Eleti (@athyuttamre) 的推文</a>：好的，关于这个名字：Responses 显然与 HTTP Responses 冲突。但我们坚信这个名字在优雅和描述性之间达到了完美的平衡。我们都会说“模型的 response 是什么？”……</li><li><a href="https://x.com/bio_bootloader/status/1887520134027436041?s=46">来自 Scott Swingle (@bio_bootloader) 的推文</a>：我一直这么说，距离游戏作为主要 AI 基准测试已经过去几年了，是时候回归了。引用 yobibyte (@y0b1byte) 我们回来了！</li><li><a href="https://x.com/athyuttamre/status/1899541489501495615?s=46">来自 Atty Eleti (@athyuttamre) 的推文</a>：Items 是 Responses 的核心概念：代表用户输入或模型输出的多态对象。Items 可以代表消息、推理、函数调用、Web 搜索调用等等。在 Chat ...</li><li><a href="https://contextual.ai/blog/introducing-instruction-following-reranker/">介绍全球首个指令遵循重排序器 (instruction-following reranker) - Contextual AI</a>：未找到描述</li><li><a href="https://lumalabs.ai/news/inductive-moment-matching">Breaking the Algo

rithmic Ceiling in Pre-Training with Inductive Moment Matching  | Luma AI</a>: Inductive Moment Matching 在速度和样本质量上超越了 diffusion models。</li><li><a href="https://x.com/LumaLabsAI/status/1899518379737661447">Tweet from Luma AI (@LumaLabsAI)</a>: 今天，我们发布了 Inductive Moment Matching (IMM)：一种打破 diffusion models 算法天花板的新预训练范式。更高的样本质量。10倍的效率。单阶段，单网络...</li><li><a href="https://news.ycombinator.com/item?id=43331582">Show HN: Factorio Learning Environment – Agents Build Factories | Hacker News</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1349077107547967621)** (1 messages): 

> `Latent Space Podcast, OpenAI Agents Platform, Responses API, Agents SDK` 


- **Latent Space 就 Agents 采访 OpenAI**: [Latent Space Podcast](https://x.com/latentspacepod/status/1899516632185045339) 发布了新一期节目，就全新的 **Agents Platform** 采访了 **OpenAI**。
- **OpenAI 为 Agent 时代更新 API**: 根据 [Latent Space podcast](https://latent.space/p/openai-agents-platform) 的报道，**OpenAI** 正在为 Agent 之年更新其全套 **APIs**、**Tools** 和 **SDKs**。
- **Responses API, SDK 和工具链**: Latent Space 获得了 **OpenAI** 的独家采访，讨论了 **Responses API**、**Web Search**、**Computer Use** 以及 **Agents SDK**。



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1899516632185045339">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 全新的 OpenAI Agents Platform https://latent.space/p/openai-agents-platform @OpenAIDevs 正在为 Agent 之年更新全套 APIs、Tools 和 SDKs。我们进行了一次独家采访...

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1349064725400518676)** (76 messages🔥🔥): 

> `OpenAI Agents SDK, Responses API, Assistants API sunset, Agent Ops, OpenTelemetry` 


- **Responses API 取代 Chat Completions**: OpenAI 正在推出全新的 [**Responses API**](https://openai.com/index/new-tools-for-building-agents/)，它是 chat completions 的超集，使得迁移非常容易，不过旧版本的 **Completions API** 目前不会被弃用。
   - 根据观看直播的成员表示，**Responses API** 支持在单次调用中调用多个工具，有望实现更简单的集成和更精简的工作流。
- **Assistants API 停用在即**: OpenAI 计划在 2026 年停用 **Assistants API**，但声称对于当前用户来说，向新 **Responses API** 的过渡应该是直接且简单的。
   - 关于如何使用 OpenAI 平台或将 traces 发送到你自己的平台的详细信息，可以在 [此 GitHub 仓库](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors) 中找到。
- **Agent SDK 的 Braintrust 推广**: 新的 OpenAI **Agent SDK** 将支持通过一行代码实现 [Braintrust 数据追踪 (data tracing)](https://x.com/braintrustdata/status/1899508228972826689)。
   - 一位成员评论说，该 SDK 还支持与 Langfuse 及其他 Agent 可观测性工具的集成。
- **新的 OpenAI API 缺失来源选择功能**: 直播观众注意到，目前还无法像 Brave 或 Google 那样进行**带有来源选择 (source selection) 的网页搜索**。
   - 根据 Latent Space 播客在采访后的说法，关于配置选项的默认回答是“不”。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/braintrustdata/status/1899508228972826689">Tweet from Braintrust (@braintrustdata)</a>: 只需一行代码即可追踪你所有的 @OpenAI Agents SDK 调用！</li><li><a href="https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors">openai-agents-python/docs/tracing.md at main · openai/openai-agents-python</a>: 一个轻量级、功能强大的多 Agent 工作流框架 - openai/openai-agents-python</li><li><a href="https://github.com/openai/openai-agents-python">GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows</a>: 一个轻量级、功能强大的多 Agent 工作流框架 - openai/openai-agents-python</li><li><a href="https://youtu.be/QU9QLi1-VvU">The new OpenAI Agents Platform: CUA, Web Search, Responses API, Agents SDK!!</a>: 来自 OpenAI 的 Nikunj Handa 和 Romain Huet 加入我们，预览他们全新的 Agents APIs：Responses、Web Search 和 Computer Use，以及全新的 Agents SDK。https...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1349131358731243592)** (1 messages): 

> `FAQ 页面，体验优化更新` 


- **OpenRouter 发布 FAQ 页面**：OpenRouter 推出了 [FAQ 页面](https://openrouter.ai/docs/faq) 以解答常见问题。
- **体验优化更新**：发布了一个小型的体验优化（Quality of Life）更新。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1348990971555086387)** (133 messages🔥🔥): 

> `Gemini 2.0 Flash, OpenAI 面向开发者的发布, Cohere 的 AYA vision, 参数计算移除, DeepSeek-R1 API 问题` 


- **Gemini 2.0 图像生成功能上线**：**Gemini 2.0 Flash Experimental** 现在限制为 **32k** 上下文，且无法使用代码执行、搜索接地（search grounding）或函数调用（function calling）。
   - 当你点击获取代码时，会得到在 `gemini-2.0-flash-exp` 下保存图像的代码，[如这里所示](https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/)。
- **OpenAI 将在太平洋时间上午 10 点进行面向开发者的发布**：成员们预计 **OpenAI** 将在 **太平洋时间上午 10 点** 发布一些面向开发者的内容。
   - 其他人根据提到 **Responses API** 的[这个帖子](https://platform.openai.com/docs/api-reference/responses)对此进行了推测。
- **Cohere 的 AYA vision 与 OpenRouter**：成员们询问 **OpenRouter** 是否会支持 **Cohere** 的 **AYA vision** 以及其他 **Cohere** 模型。
   - API 上的 **AYA Expanse** 模型（8B 和 32B）似乎按 **输入 $0.50/1M Tokens** 和 **输出 $1.50/1M Tokens** 计费，但这尚未得到证实，[详见此处](https://cdn.discordapp.com/attachments/1094454198688546826/1349049467378339902/image.png?ex=67d1afb9&is=67d05e39&hm=ffdce841e8f353b45682a480ea8f937b0169a3414ebe0967c87230ba436786b4&)。
- **参数（Parameter）计算已移除**：OpenRouter 移除了参数计算，因为它不够准确，且认为它可能更具误导性而非实用性。
   - 团队表示，他们可能会进行某种形式的人工筛选，并在以后重新调整某些内容时将其带回，因为参数很难调整，甚至可能像*远古符文*一样难以捉摸。
- **传闻 Gemma 3-27b 即将推出**：成员们正在推测 **Gemma 3-27b** 即将发布。
   - 预计它将在[巴黎 Gemma 开发者日](https://rsvp.withgoogle.com/events/gemma-dev-day-paris)期间发布，并应随附权重。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/api/v1",">Discord</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/faq">OpenRouter FAQ</a>：查找有关 OpenRouter 统一 API、模型访问、定价和集成的常见问题解答。</li><li><a href="https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://rsvp.withgoogle.com/events/gemma-dev-day-paris,">未找到标题</a>：未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-function-calling">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/provider/cohere">Cohere | OpenRouter</a>：浏览 Cohere 提供的模型
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1349036869106798694)** (3 messages): 

> `Agent 工具, 开发者直播, 开发者 AMA` 


- **Agent 工具在 OpenAI 开发者直播中首次亮相！**：OpenAI 通过直播宣布了**面向开发者的 Agent 工具**，邀请开发者探索新的功能和集成。
   - 直播结束后，安排了从 **太平洋时间上午 10:30–11:30** 的 **AMA**（*Ask Me Anything*）环节，提供与开发团队直接互动的机会。
- **加入 OpenAI 开发者直播 AMA！**：OpenAI 开发者团队在直播后立即主持了 **AMA** 环节，鼓励开发者就新工具提问。
   - **AMA** 的邀请通过链接分享（[OpenAIDevs X 帖子](https://x.com/OpenAIDevs/status/1899502117171155002)），提示用户回复问题并与最新功能背后的团队互动。



**提到的链接**：<a href="https://x.com/OpenAIDevs/status/1899502117171155002)">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：直播结束后加入我们的 AMA！从太平洋时间上午 10:30–11:30，今天发布内容背后的团队将回答您的问题。在下方回复您的问题。引用 OpenAI (@OpenAI) Agent 工具...

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1348985469240741901)** (91 messages🔥🔥): 

> `AGI, Grok, GPT-4.5, Gemini, Sider AI` 


- **AI 炒作助长恐惧与进步**：成员们讨论到，虽然开发 **AI** 的目标是用更少的资源做更多的事，但人们从根本上感到不安，而 **AGI** 及相关话题总是在媒体上被大肆渲染，这虽然促进了销售，但也吓跑了那些倾向于避开 **AI** 的人。
   - 他们还指出，“人们完全有理由恐惧 AI，并可能担忧自己或至少是下一代的就业前景”，同时“AI 也带来了极其重要的进步”。
- **用户希望 ChatGPT 拥有 Grok 的氛围**：成员们表示“也希望 ChatGPT 能有 **Grok** 的氛围”，并引用了 **Elon Musk** 在 [tenor.com GIF](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268) 中的形象。
   - 一位成员问道：“关于过滤器，或者更确切地说，是缺乏过滤器，你所说的‘氛围’到底是指什么？”
- **乱序单词测试展示 AI 实力**：一位成员测试了 **GPT-4.5** 破译乱序单词的能力（类似于人类的能力），并报告称即使是在包含长篇技术医学术语的多段输入中，它也能立即识别出来，甚至在翻译后成功还原了印度尼西亚语。
   - 然而，另一位成员指出“旧模型在相当长一段时间内已经能做到这一点了”，而另一位成员则确认他们“曾多次遇到模型在变位词和简单密码上失败的情况”。
- **Deep Research 价格对比**：成员们对比了不同 AI 平台的深度研究能力价值，声称 **OpenAI** 的 **deep research** 是最佳选择（每月 **$200**），而 **Grok** 的 **$30 deep research** 和 **Perplexity** 的 **$20** 方案提供了更便宜的替代方案。
   - 他们接着表示，尽管价格昂贵，但目前 OpenAI 模型的“限制非常糟糕（SUCK）”。
- **Sider AI 被评估为浏览器集成工具**：成员们讨论了 **Sider AI**，这是一个集成不同 AI 模型的浏览器扩展，但澄清道“它不适合重度用户”，仅擅长浏览器集成，不适合编程或文档检索，尽管它付费提供了 **Claude 3.7 Sonnet** 等模型。
   - 一位成员说：“我目前是 ChatGPT Plus 用户，但我感觉它不适合我，”而另一位成员则表示“Perplexity 对学生来说也很棒。Grok 同样很强大。”


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268">Elon Musk This Is Elon Musk GIF - Elon musk This is elon musk Musk - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.franksworld.com/2025/03/10/a-gentle-explanation-of-why-gpt-4-5-was-such-a-fail/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1349056382304784514)** (10 messages🔥): 

> `GPT-4.5 Code Inconsistencies, GPT-4o Code Generation Issues, Trustworthiness of AI Models, New responses API vs chat completions` 


- **GPT-4.5 的代码：不一致得令人发笑**：一位用户报告称 **GPT-4.5** 生成了不一致的代码，例如在定义了 `start()` 函数后调用了一个不存在的 `startApp()` 函数，质疑该模型的可靠性。
   - 另一位用户附和道，称 **4.5** 毫无用处，并强调了 **GPT-4o** 的类似问题，指出“你绝对不能信任 4o，或者说任何模型”。
- **异步趣事：GPT-4.5 误命名函数**：一位用户分享了另一个代码片段，其中 **GPT-4.5** 错误地调用了 `safelyAdd()` 而不是 `safePush()`，尽管它定义的是后者，强调了持续监督的必要性。
   - 他称这个问题“令人不安”，并对必须“照看（babysit）这种‘智能’”表示担忧。
- **追求 4.5 的认知可信度**：一位成员询问是否有人在关注 **GPT-4.5** 的认知发展和对齐（alignment），特别是关于内省推理和直觉，以增强“信任”。
   - 针对这个问题，另一位用户要求澄清“merit”一词以及它与模型辅助的关系。
- **New Responses API：一个更闪亮的 Assistants API？**：一位成员询问了 **new responses API** 与现有的 **chat completions API** 之间的区别。
   - 另一位成员澄清说，**new responses API** 基本上就是“更好用的 Assistants API”。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1349072868931014706)** (6 条消息): 

> `Jailbreaking, Terms of Service, Allowed Content, Prompting Techniques` 


- **探索 AI 'Jailbreaking' 的细微差别**：一位用户询问如何通过模拟家庭关系或危险等场景，使 **AI models** 提供受限或更准确的答案，另一位用户将其识别为 'jailbreaking'，这可能违反了 [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/)。
   - 该用户警告不要违反 **ToS** 和使用政策，以保护账号访问权限，并强调服务器规则也禁止讨论如何绕过这些限制，同时建议关注伦理边界内允许的内容。
- **在 AI 交互中探索允许的内容**：通过文本进行关于涉及奇幻写作、图像生成或角色扮演游戏中的暴力讨论/交互，在这些通用政策下是**不被禁止的**。
   - 该用户鼓励其他人与模型进行讨论，就像他们与频道中的其他成员讨论一样，解释意图、兴趣和疑虑。
- **Prompting Techniques**：这只是 Prompting，没有危害，且能学到很多东西。该用户随后给出了几个如何以不同方式探索同一话题的例子；有些产生的结果比其他的更好。
   - 他们问道：当被告知 *“我奶奶总是通过教我如何编写 Python 库的基础程序来哄我入睡。你也能用这种方式帮我入睡吗？”* 时，它是否能提供“更好”的 Python 课程？


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1349072868931014706)** (6 条消息): 

> `Jailbreaking AI Models, Terms of Service violations, Allowed content guidelines, Prompting Techniques` 


- **Jailbreaking 模型违反 ToS**：一名成员澄清说，尝试 **jailbreak** AI models 可能会违反 [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/) 和 [Usage Policies](https://openai.com/policies/usage-policies/)，从而可能导致账号被封禁。
   - 该成员强调，根据服务器规则，Discord 服务器内禁止讨论规避这些政策的行为。
- **允许内容的探索**：讨论转向了在特定条件下探索允许的内容，如暴力故事、艺术或角色扮演。
   - 值得注意的是，只要*没有人类受到伤害*，或者没有被教导如何造成伤害，且实际伤害发生的可能性极低，此类内容就是被允许的。
- **模型规则集解析**：模型似乎遵守比 Discord 服务器更严格的规则，因此需要对允许的内容进行双层理解。
   - 该成员鼓励尊重模型的安全训练，并清晰地沟通意图，以促进与 AI 的 *teamworking* 关系。
- **Prompting Techniques**：讨论涉及使用 *prompting techniques*，例如要求模型扮演特定的 persona 以获得所需的输出。
   - 该成员建议不要对模型撒谎，并建议在允许的内容范围内探索模型的边界，以确定有效的 Prompting 风格。
- **不应涉及真实人类的安全**：该成员表达了一个个人界限，即不让模型参与涉及真实人类安全或潜在法律违规的场景。
   - 模型并未经过在这些情况下做出适当反应的训练，很可能会建议寻求人类帮助并联系相关部门，且用户更倾向于一个看起来更冷静的模型。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1348981426510368880)** (66 messages🔥🔥): 

> `Aider --watch 标志，推理标签，Aider 每日预算，Claude Code 的 DMCA 下架，o1 pro / o3 mini pro API` 


- ****Aider** *watch-files* 现已上线！**: Paul Gauthier 宣布，使用 `--watch-files` 标志运行 `aider` 现在可以启用 **live mode**（实时模式），通过 `AI`、`AI!` 或 `AI?` 注释监视仓库中的所有文件以获取编码指令。
   - 感叹号 `AI!` 触发 aider 进行更改，而问号 `AI?` 则触发其回答问题，如 [Aider 浏览器 UI 演示视频](https://aider.chat/docs/usage/watch.html) 所示。
- ****推理标签（Reasoning Tag）**对供应商的依赖**: 推理标签（例如 `<think>`）的实现取决于供应商，有些供应商（如 **Fireworks**）将标签输出在消息 `content` 内部，而不是作为单独的字段。
   - 鼓励用户查看 [Hugging Face](https://huggingface.co) 或官方 **R1 repo** 以获取更多详情。
- ****Aider** 每日预算讨论**: 一位用户询问了 **Aider** 所需的每日预算，一名成员报告称，每周进行 7-12 小时的 AI 编码，**Sonnet 3.7** 的成本大约是排行榜成本的 **2 倍**。
   - 他们警告说，**每周 40 小时**的工作量很容易导致 **8-10 倍**于排行榜的成本，而精细的 Prompt Engineering 可以节省 Token 使用量，其他用户则通过默认使用 **o3 或 R1** 等更便宜的模型来管理成本。
- ****DMCA** 下架 Claude Code 泄露内容**: 一位用户报告称，因 fork 了 **Claude code leak repo** 而收到 **DMCA** 下架通知，原始泄露者和所有 fork 均受到影响。
   - 另一位用户推测 **o1 pro** / **o3 mini pro** 可能很快会在 API 中发布。
- **使用 **Responses API** 构建 Agent 的新工具**: 一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=hciNKcLwSes)，宣布 API 平台正在演进，以使开发者能够更快速、更轻松地构建 Agent。
   - 新的 **Responses API** 使用 Assistants 和类似 Thread 的对象构建 Agent，**Code Interpreter tool** 的价格为每千次查询 2.50 美元，文件存储价格为 0.10 美元/GB/天。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/watch.html">IDE 中的 Aider</a>: Aider 可以监视您的文件，并响应您在喜爱的 IDE 或文本编辑器中添加的 AI 注释。</li><li><a href="https://www.youtube.com/watch?v=hciNKcLwSes">使用 API 构建 Agent 的新工具</a>: 我们正在演进 API 平台，使开发者能够更快速、更轻松地构建 Agent。Kevin Weil, Nikunj Handa, Steve Coffey, 和 Ilan Bigio 介绍了...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1348983164663631924)** (27 messages🔥): 

> `Litellm APIConnectionError, 微调 Qwen Coder, Aider 撤销问题, Aider 高级技巧：sqlite schema, Aider 排行榜解释` 


- **解决 Litellm APIConnectionError**: 一些用户在使用该工具时遇到了 ```litellm.APIConnectionError: APIConnectionError: OpenrouterException - 'choices'```。
- **针对代码库上下文微调 Qwen Coder**: 成员们正在考虑将 **Qwen Coder 2.5** 之类的模型针对代码库进行微调，以避免总是将整个代码库作为上下文传递。
   - 一位用户质疑*为什么首先需要将整个代码库作为上下文传递*。
- **Aider 在撤销文件创建时遇到困难**: 一些用户在尝试撤销涉及文件创建的 commit 时遇到了 Aider 的问题。
   - 收到的错误消息是：*The file blahblah was not in the repository in the previous commit. Cannot undo safely.*
- **高级技巧：自动将 sqlite schema 转储到仓库中**: 一位成员建议在任何更改后，通过 `sqlite3 "$TEMP_DB" ".schema" > schema.txt` 自动将 **sqlite schema dumps** 转储到仓库中，以改进 Aider 的数据库交互。
- **Aider 编辑格式的含义**: Aider 排行榜中的“正确编辑格式”是指 Aider 期望 LLM 用于编辑文件的格式；不同的模型在不同的格式下表现更好。
   - 一位用户分享了 [Aider 关于编辑格式的文档](https://aider.chat/docs/more/edit-formats.html) 链接，其中详细介绍了 *whole* 和 *diff* 编辑格式。



**提到的链接**: <a href="https://aider.chat/docs/more/edit-formats.html">编辑格式</a>: Aider 使用各种“编辑格式”让 LLM 编辑源文件。

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

end4749: 可能相关，看起来很有趣 https://github.com/xingyaoww/code-act
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1348975033816711168)** (26 条消息🔥): 

> `Unity LM Studio 连接, 内部 LLM 聊天设置, Python SDK 视觉模型, LLM 小说解读, Copy4AI 扩展` 


- **Unity 与 LM Studio 挂钩**：一位成员制作了一个 [YouTube 视频](https://www.youtube.com/watch?v=dQw4w9WgXcQ)，展示了 **Unity** 与 **LM Studio** 之间的连接以进行模型交互，并利用 **JSON** 文件进行数据存储。
   - 由于 **Discord** 服务器中没有专门的 **Unity** 频道，该成员不确定该将视频发布在哪里。
- **搭建自己的内部 LLM 聊天**：一位成员正在寻求关于设置带有用户账户的内部 **LLM Chat** 的建议，该系统需与其公司的 **Google Docs** 知识库集成，并可能使用推理 **API**。
   - 他们正在考虑使用 **LlamaIndex** 进行向量数据库管理，以及 **AnythingLLM** 或 **OpenWebUI** 等聊天界面，同时也在探索 **LM Studio** 内部的选项。
- **Python SDK 获得视觉功能**：一位使用 **Python SDK 1.0.1** 的成员注意到 **Typescript SDK** 具有向视觉模型发送图像的能力，他们希望在 **Python** 中复制此功能。
   - 目前看来，该功能尚未移植到 Python 版本。
- **LLM 难以捕捉角色的微妙之处**：一位成员报告了 **LLM** 在解读小说中隐含的角色行为时表现得过于正面，在没有明确陈述的情况下难以检测到负面含义。
   - 他们发现一个 **Gemma 2** 模型能够正确解读涉及管家和上校的暴力场景。
- **Copy4AI：加速编码**：一位成员询问了与 [Copy4AI 扩展](https://copy4ai.dev/) 相关的 `ext install` 命令，该工具旨在为 AI 助手复制代码片段。
   - 经澄清，`ext install` 会打开 **VS Code** 中的扩展侧边栏并导航到相关扩展；不过，该扩展已重命名为 `leonkohli.snapsource`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1j8u90g/new_gemma_models_on_12th_of_march/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/QwQ-0.5B-Distilled-SFT-GGUF">mradermacher/QwQ-0.5B-Distilled-SFT-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/better-call-saul-call-saul-its-showtime-folks-gif-8557719">Better Call Saul Its Showtime Folks GIF - Better Call Saul Call Saul Its Showtime Folks - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://copy4ai.dev/">Copy4AI - 与 AI 助手共享代码上下文</a>：直接将文件和文件夹内容连同项目结构复制到剪贴板，供 ChatGPT、Claude 等 AI 助手使用。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1348989663398068286)** (45 messages🔥): 

> `Swap 使用、投机解码 (Speculative Decoding)、AMD Vulkan/ROCm 性能、RTX 2000E 推理测试、CXL 内存` 


- **Swap 再次走红**：随着 **100Gbit NVMe 驱动器**、**400Gbit 网络**和 **CXL 内存**的出现，*Swap* 再次变得有用，正如 [Dave2D 的 M3 Ultra Mac Studio 评测](https://www.youtube.com/watch?v=J4qwuCXyAcU)中所强调的那样。
   - 一位用户报告在运行 **R1** 时达到了 **18 t/s**；表现甚至超出了预期！
- **投机解码在某些配置上停滞**：**投机解码 (Speculative decoding)** 仅能加速稠密模型 (**50b+**) 的推理，且在受限于 RAM 带宽或比较 **0.5b 到 14b** 模型时，其表现可能优于标准推理。
   - 使用 **Ryzen 3 4450U** 和 **Vega iGPU** 的用户发现投机解码的性能仅为原生模式的一半，另一位用户的 RAM 读取带宽峰值为 **28 Gbit/s**。
- **AMD 驱动灾难：Vulkan 和 ROCm 性能问题**：一位 AMD 用户报告称，在 **24.12.1** 驱动中 **Vulkan** 和 **ROCm** 的性能出现故障，性能下降了 **35%**，不过 **ROCm** 在 **v1.1.13+** 中已修复。
   - Vulkan 性能在 **25.1.1** 中仍维持在 **50%**，在 **25.2.1** 中有小幅提升，该用户已向 [AMD 提交了 Bug 报告](https://www.amd.com/en/support/kb/faq/rs-help)并建议降低预期。
- **RTX 2000E 让功耗更高的前代产品汗颜**：一位用户开始对新款 **RTX 2000E** 进行推理测试，报告称其性能比 **A2000** 提升了约 **40%**，同时功耗降低了 **20W**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=J4qwuCXyAcU">M3 Ultra Mac Studio 评测</a>：我对搭载 M3 Ultra 的 Apple Mac Studio 的评测，包含游戏基准测试和 Deepseek R1。如果您想支持本频道，请考虑加入 Dave2D 会员...</li><li><a href="https://www.youtube.com/watch?v=wW-Rj5MW2EU">背包级 LLM 大爆发！</a>：我对四款便携式系统进行了本地 LLM 测试。🛒 装备链接 🛒* 💻🔄 32GB RAM 的 K8 Plus：https://amzn.to/3FnjJY0* 🛠️🚀 96GB RAM 套件：https://amzn.to/...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1348976567535009862)** (18 messages🔥): 

> `X 遭受 DDoS 攻击、LanguageBind vs ImageBind、使用 Chroma DB 实现聊天机器人记忆` 


- **X 遭遇 DDoS：网络攻击导致平台瘫痪**：成员们讨论了最近针对 [X 的 DDoS 攻击](https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/)，**Dark Storm** 组织声称对此负责并导致了大规模停机。
   - 虽然 Elon Musk 最初暗示 **乌克兰** 是攻击的幕后黑手，但牛津大学的 **Ciaran Martin** 等专家在 [BBC 的一篇文章](https://www.bbc.co.uk/news/articles/c62x5k44rl0o)中称这一解释“完全没有说服力”。
- **LanguageBind 领先于 ImageBind**：一位成员询问处理**多模态数据**的最佳方法（考虑使用 **ImageBind**），另一位成员推荐了 [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind)，指出它*支持所有模态*且*优于 ImageBind*。
   - 该成员有兴趣使用*图像、音频、视频和 PDF* 模态。
- **Chroma DB 难题：聊天机器人遗忘过去**：一位成员寻求使用 **Chroma DB** 保存聊天机器人对话的帮助，以解决机器人退出后无法记住旧对话的问题。
   - 另一位成员建议*使用持久化存储*来保存 SQL 文件以便导出和重新加载以保持连续性，或者告诉他们*似乎 HF 做了一些调整，所以我可以重用 API*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/">X 遭受攻击——Dark Storm 声称是 Musk 平台 DDoS 攻击的幕后黑手</a>：在 Elon Musk 指向乌克兰的同时，一个名为 Dark Storm 的亲巴勒斯坦组织声称对导致 X 瘫痪的大规模网络攻击负责。</li><li><a href="https://www.politico.eu/article/elon-musk-claim-ukraine-linked-cyberattack-x-draws-criticism/">Musk 指责乌克兰人对 X 发动网络攻击，专家们并不买账。</a>：网络专家表示，乌克兰参与 X 中断的证据非常薄弱。</li><li><a href="https://www.bbc.co.uk/news/articles/c62x5k44rl0o">专家称指责乌克兰导致 X 大规模停机是“垃圾话”</a>：这一说法是由该平台所有者 Elon Musk 提出的，他是乌克兰及其总统的直言不讳的批评者。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1349062806254649426)** (2 messages): 

> `Reka Flash 3, Cakeify LoRA, Wan2.1 14B I2V 480p` 


- ****Reka Flash 3** 不再适用于端侧使用**: 新规则：**Reka Flash 3**，一个 **21B** 通用推理模型，不再被称为“端侧（on-device）”模型。
   - 该模型完全基于合成和公开数据集训练，性能可与 **OpenAI o1-mini** 等专有模型相媲美，并为 Nexus 提供支持。Nexus 是 Reka 用于创建和管理具有原生深度研究能力的 AI worker 的平台 ([Reka Space](https://space.reka.ai), [getnexus.reka.ai](https://getnexus.reka.ai))。
- ****Cakeify LoRA** 首次亮相**: 针对 **Wan2.1 14B I2V 480p** 的 **Cakeify Effect LoRA** 已发布，它允许用户将图像中的任何物体“蛋糕化” ([Cakeify](https://huggingface.co/Remade/Cakeify))。
   - 它能将图像转换为物体被切成蛋糕的视频，使用了适配自 **Wan2.1 14B 480p I2V** 基础模型的简单提示词结构。此外，用户还可以加入他们的 [Discord](https://discord.com/invite/7tsKMCbNFC) 免费使用此 LoRA 生成视频并申请新的 LoRA。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/RekaAI/reka-flash-3">RekaAI/reka-flash-3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Remade/Cakeify">Remade/Cakeify · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1348993164827951154)** (2 messages): 

> `RAGcoon, Startup assistance, Agentic RAG, Qdrant vector database, LlamaIndex` 


- **RAGcoon 发布，助力初创企业**: 一个名为 [RAGcoon](https://github.com/AstraBert/ragcoon) 的新 **Agentic RAG** 项目已启动，旨在通过导航各种资源和建议来协助建立初创企业。
   - 它利用了来自**成功创始人的免费资源**，并使用*混合搜索（hybrid search）、查询扩展（query expansion）*和*多步查询分解（multi-step query decomposition）*等技术执行复杂的检索操作。
- **RAGcoon 拥有令人印象深刻的可靠性指标**: **RAGcoon** 通过*自动纠错（auto-correction）*机制评估**检索上下文的可靠性**，以及自身回答的**相关性和忠实度**。
   - 它基于 **LlamaIndex** 构建，使用 **Qdrant** 提供向量数据库服务，使用 **Groq** 进行 LLM 推理（**Qwen 的 QwQ-32B**），使用 **Hugging Face** 提供 Embedding 模型，使用 **FastAPI** 构建后端 API，并使用 Google 的 **Mesop** 构建前端。
- **RAGcoon 提供支持 Docker 的本地安装**: **RAGcoon** 是**开源**的，并且由于其[支持 Docker](https://github.com/AstraBert/ragcoon)，可以在本地快速启动。
   - 创建者正在探索推出在线版本的可能性，并对合作持开放态度。



**提到的链接**: <a href="https://github.com/AstraBert/ragcoon">GitHub - AstraBert/ragcoon: Agentic RAG to help you build a startup🚀</a>: Agentic RAG 助力你建立初创企业🚀。欢迎通过在 GitHub 上创建账号来为 AstraBert/ragcoon 的开发做出贡献。

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1348995423284494367)** (1 messages): 

> `Real-time Disease Detection, Pretrained Models for Fine-Tuning` 


- **头脑风暴实时疾病检测系统**: 一位成员正在寻求创建实时疾病检测系统的指导，打算使用摄像头识别感兴趣区域并随后对其进行分类。
   - 咨询重点在于识别适合为此特定应用进行微调的预训练模型。
- **探索用于疾病检测的预训练模型**: 该成员的主要目标是确定哪些预训练模型在实时疾病检测场景下的微调效果最出色。
   - 这涉及利用计算机视觉技术来分析摄像头输入并分类潜在的疾病指标。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1348995746518667317)** (2 messages): 

> `SmolLM, Mobile LLMs, Fine-tuning LLMs` 


- **SmolLM 成为小型 LLM 解决方案**：针对关于适合 Fine-tuning 和手机部署的最佳小型 LLM 模型的咨询，一位成员建议探索 **SmolLM**。
   - 他们分享了 **SmolLM2** 和 **SmolVLM** 系列模型的 [GitHub repository](https://github.com/huggingface/smollm)。
- **SmolLM GitHub Repository**：**SmolLM** 的 GitHub repository 可以在 [huggingface/smollm](https://github.com/huggingface/smollm) 找到。
   - 它包含了关于 **SmolLM2** 和 **SmolVLM** 系列模型的信息。



**提及的链接**：<a href="https://github.com/huggingface/smollm">GitHub - huggingface/smollm: Everything about the SmolLM2 and SmolVLM family of models</a>：关于 SmolLM2 和 SmolVLM 系列模型的一切 - GitHub - huggingface/smollm: Everything about the SmolLM2 and SmolVLM family of models

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1349131009911685182)** (2 messages): 

> `ChatML conversion woes, Llama discrepancies` 


- **Tokenizer 问题引发技术任务困扰**：一位成员表示，尽管使用 **for loop** 成功转换，但在使用 Tokenizer 的方法将数据集转换为 ChatML 格式时遇到困难。
   - 他们请求指导或参考解决方案，并表示 *notebook 留下的疑问比答案还多*。
- **Llama 的语言表现导致输出差异**：一位成员观察到，使用 `chat` 方法与手动输入预期的 **chat template** 时，**Llama model** 的响应有所不同。
   - 他们提供了一张 [图片](https://cdn.discordapp.com/attachments/1313889336907010110/1349158970501369916/image.png?ex=67d215b4&is=67d0c434&hm=5a4ca825469b47ad142bff31b9c0e7c34c8d25ebe1ebf5de85a8299740032c4a&) 来展示这种差异。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1348980720667856977)** (39 messages🔥): 

> `LlamaIndex 错误, smolagents DocstringParsingException, smolagents OpenAI/DeepSeek 密钥, Ollama 与 Smolagents 的集成, Brave 浏览器的 Leo AI` 


- **LlamaIndex 中 Async Await 的故障排除**：一位用户在 Python 3.10.12 环境下使用 LlamaIndex 的 `pipeline.arun` 时遇到了 `SyntaxError: 'await' outside function` 错误。
   - 建议是升级到 Python 3.11，因为该错误表明旧版本在处理异步代码执行时存在问题。
- **Smolagents 的 Docstring 解析问题**：由于 docstring 中缺少参数描述，用户在为 smolagents 中的自定义工具生成 JSON schema 时遇到了 `smolagents._function_type_hints_utils.DocstringParsingException`。
   - 即使切换了参数顺序，错误仍然存在，这表明 docstring 的解析方式存在问题。
- **用于 Gemini, OpenAI 和 DeepSeek 的 Smolagents 密钥**：成员们分享了在 `smolagents` 中使用 **Gemini, OpenAI 和 DeepSeek 模型** 的代码片段，并提供了设置 `LiteLLMModel` 和 `OpenAIServerModel` 以及相应 API 密钥的示例。
   - 对于 **Gemini**，提供了一个指向 [Google AI Studio](https://aistudio.google.com/app/apikey) 的链接以获取免费 API 密钥。
- **Ollama 接管 HfApiModel**：一位成员分享了一段代码片段，演示了如何将 Hugging Face 的 `HfApiModel` 替换为 **Ollama**，以便与 `smolagents` 配合使用。
   - 该解决方案涉及创建一个自定义的 `OllamaModel` 类，通过与 Ollama 的 API 交互来进行 Prompt 生成，从而允许在 `smolagents` 中使用本地 LLM。
- **Brave 的 Leo AI 揭秘 Llama 3.1 8B**：成员们开玩笑说 **Brave 浏览器的 Leo AI** 是由 **Llama 3.1 8B** 驱动的，它足够轻量，可以在笔记本电脑上运行而不会占用过多的系统资源。
   - 他们质疑为什么 Brave 会针对 Leo AI 的更多使用量收取额外费用，因为其资源需求相对较低。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10:28:36.929Z">未找到标题</a>：未找到描述</li><li><a href="https://aistudio.google.com/app/apikey">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/C7_DLQLrS9w?si=uLQYuYAAA-EWW5B7">Hugging Face AI Agents 课程 - Unit 1 | Agent 介绍</a>：在这段视频中，我介绍了作为 Hugging Face AI Agents 课程一部分的 AI Agent 和大语言模型 (LLM) 的基础知识。你将了解什么是 Agent...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/tree_summarize/#summarize">Tree Summarize - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/openai/openai-agents-python">GitHub - openai/openai-agents-python: 一个用于多 Agent 工作流的轻量级、强大的框架</a>：一个用于多 Agent 工作流的轻量级、强大的框架 - openai/openai-agents-python
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1348995790282162256)** (5 messages): 

> `GPU 模式, CUDA 专长, 圣何塞会议` 


- **GPU Mode 宣传热潮启动**：一位成员兴奋地在多个频道宣布他们已为 **GPU mode** 做好准备，但随后迅速删除了消息，表现出犹豫。
   - 版主引导该成员使用特定频道（<#1288557096404516945> 或 <#1218444432588800010>）发布此类公告，以维持服务器秩序。
- **CUDA 新手寻求参加圣何塞峰会**：一位成员询问是否可以参加 **3 月 16 日** 在 **圣何塞** 举行的 **GPU mode** 会议，尽管他缺乏 **CUDA** 专长。
   - 这个问题引发了关于专业知识是否是参与的前提条件的讨论。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1348981893227479152)** (9 条消息🔥): 

> `Triton tl.cast vs tl.full, Pipeline softmax kernel in Triton, Tiled GEMM Implementation in Triton, TF32 precision in Triton` 


- **Triton 的 `tl.full` 解决类型转换难题**：一位用户发现使用 `tl.full` 创建具有特定值和数据类型的 **0 维张量**（`tl.full((), 5, tl.int8)`）可以成功避免在与张量相加时出现的溢出问题。
   - 有效的解决方案是 `tmp_5 = tl.full((1,), value=5, dtype=tl.int8); out = a.to(tl.int8) + tmp_5`。
- **Triton 在 Softmax Kernel 速度竞赛中胜出**：一位用户在 **Triton** 中实现了一个流水线 Softmax Kernel，其表现出人意料地优于 **PyTorch** 的等效实现，并指出该 **Triton** 版本在 **float16 T4** Colab 上的速度明显快于预期。
   - 帖子中附带了一张显示速度对比结果的图片 [image.png](https://cdn.discordapp.com/attachments/1189607595451895918/1349066304341934160/image.png?ex=67d1bf67&is=67d06de7&hm=e424c1148a06e3ac3adac9c31cd7c0bc6e930f047dc69c2db02cba55e5949695&)。
- **Triton 分块 GEMM 面临精度困境**：用户在 **Triton** 中实现的分块 GEMM（可在 [GitHub](https://github.com/gauravjain14/mlcompilers_and_kernels/blob/main/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py) 获取）在与参考的 **PyTorch** matmul 进行逐元素比较时，需要在 `torch.allclose` 中设置较高的容差（`atol=1e-1, rtol=1e-1`）。
   - 预期是达到更高的精度（约 `1e-4`），因此请求改进准确性的建议。
- **精度参数提升性能表现**：一位用户建议在 `tl.dot` 中禁用 **TF32**（使用 `allow_tf32=False` 或 `input_precision=`）以提高 **Triton** 的精度，这解决了精度问题。
   - 他们指出 `allow_tf32` 已被弃用，并建议探索 `tl.dot(input_precision=)` 的其他选项。



**提及的链接**：<a href="https://github.com/gauravjain14/mlcompilers_and_kernels/blob/main/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py">mlcompilers_and_kernels/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py at main · gauravjain14/mlcompilers_and_kernels</a>：通过在 GitHub 上创建账号来为 gauravjain14/mlcompilers_and_kernels 的开发做出贡献。

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1348979778451148923)** (19 条消息🔥): 

> `stmatrix padding, BFE/BFI instructions, CUDA binary tools` 


- **对 stmatrix 地址进行填充以防止 SMEM Bank 冲突**：必须对 **stmatrix** 的地址进行填充，以避免所有地址都指向同一个起始 **SMEM bank** 从而导致 8 倍冲突，这一问题之前在 fast.cu 和 deepgemm 代码中已得到修复。
   - 该问题*没有硬件层面的解决方案*，需要精细的内存布局管理，特别是在分块布局（tiled layouts）不可行的情况下。
- **CUDA 中 BFE/BFI 指令之谜**：**BFE/BFI**（位域提取/插入）指令尽管在 **CC 7.x** 和 **8.x** 中被移除，并在 **9.x** 中重新引入，但它们并不是原生 SASS，在 **sm_70/sm_80/sm_90** 和 **sm_89** 上（至少在使用 **nvcc 12.6.2** 时）会转换为*两条指令*。
   - 目前最好的权宜之计是使用条件漏斗移位（funnel shift）和掩码，尽管存在线程分歧（thread divergence），但性能优于 BFE 或对 u64 进行移位。
- **CUDA 二进制实用程序应用说明已发布**：针对 Linux、Windows、Mac OS 和 Android 的 CUDA 二进制工具 **cuobjdump**、**nvdisasm**、**cu++filt** 和 **nvprune** 的应用说明已发布，可通过 [CUDA Binary Utilities 文档](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set)访问。
   - 文档详细说明了 `cuobjdump` 和 `nvdisasm` 之间的区别，解释了 CUDA 二进制（cubin）文件是一种 ELF 格式文件，由 CUDA 可执行代码段以及包含符号、重定位器、调试信息等的段组成。



**提及的链接**：<a href="https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set">1. Overview — CUDA Binary Utilities 12.8 文档</a>：未找到描述。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1349070172601712815)** (9 条消息🔥): 

> `leetgpu.com, leaderboard, GPU mode` 


- **LeetCode 迎来 GPU 模式**：一名成员发现了 [leetgpu.com](https://leetgpu.com)，这是一个类似于 **LeetCode** 的平台，但专为对利用 **GPU acceleration** 感兴趣的人士设计。
   - 该平台旨在加速问题解决，特别是针对 **PMPP**（可能指 Parallel and Multiprocessor Programming）问题，并设有用于竞技编程的 leaderboard。
- **leetgpu 的新 leaderboard 刚刚上线**：一个新的 leaderboard 已经创建，可在 [GPU Mode discord channel](https://discord.com/channels/1343002583001726986) 查看。
   - 用户可以提交他们的 `/leaderboard submit` 并与他人竞争。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

iron_bound: https://jackhopkins.github.io/factorio-learning-environment/
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1349035088079487118)** (5 条消息): 

> `ROCm Version, Ubuntu Versions, vLLM on AMD` 


- **针对 gfx 1100 的 ROCm 6.2+**：对于 **gfx 1100** 及以上版本，建议使用 **ROCm >= 6.2**，可能搭配包含 **ROCm 6.2** 的 **PyTorch 2.6+** 或包含 **ROCm 6.3** 的 nightly 版本。
- **支持 Ubuntu 22 或 24**：根据 [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)，**Ubuntu 22** 和 **24** 均受支持；如果是全新安装，建议选择 **Ubuntu 24** 以获取最新的 kernel。
- **在 AMD 上运行 vLLM**：一名成员成功通过 docker image 在 **AMD** 设备上运行了 **vLLM**，而另一名成员因集群限制在从源码构建时遇到困难，希望能有其他方法。



**提及的链接**: <a href="https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html">Compatibility matrix — ROCm Documentation</a>: 未找到描述

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1349041378314948704)** (3 条消息): 

> `MLX custom kernels, Metal-cpp, GPU Programming, C++ linear algebra library` 


- **自定义 Kernel 构建**：MLX 文档中有一个关于 [构建自定义 kernel](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) 的优秀章节，指导完成 metal 实现、绑定和 python 使用。
   - 该示例涉及创建一个操作，将两个数组 `x` 和 `y` 分别乘以系数 `alpha` 和 `beta`，然后将它们相加得到结果 `z = alpha * x + beta * y`。
- **首选 Metal-cpp**：在收到多个建议后，一名成员决定使用 **metal-cpp**，并从学习 **MLX 的自定义 kernels** 和完成 **Metal-Puzzles challenge** 开始。
   - 该成员还计划开展一个小型个人项目：一个使用 Metal 加速的 **C++ linear algebra library**。
- **Metal 与 Swift 集成**：一名成员建议查看 **Sebastian** 之前的建议，并提到了一个集成 **Metal** 和 **Swift** 的矩阵乘法示例。
   - 如果可能的话，他们会尝试找到该示例并发送。



**提及的链接**: <a href="https://ml-explore.github.io/mlx/build/html/dev/extensions.html">Custom Extensions in MLX &#8212; MLX 0.23.2 documentation</a>: 未找到描述

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1349130384184709182)** (1 条消息): 

> `IPFS Accelerate JS, HuggingFace Port to TS/JS` 


- **IPFS Accelerate JS 结构已实现**：**IPFS Accelerate JS** 的初始结构已通过 [此 commit](https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773) 实现，包含占位符模块和 TypeScript 转换。
- **HuggingFace 库迁移至支持 WebNN/WebGPU 的 TS/JS**：一名成员正积极使用 **WebNN/WebGPU** 将整个 **HuggingFace** 库移植到 **TS/JS**。



**提及的链接**: <a href="https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773">feat: Implement initial structure for IPFS Accelerate JS with placeho… · endomorphosis/ipfs_accelerate_py@2f99633</a>: …lder 模块和 TypeScript 转换

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1348994348058017812)** (6 messages): 

> `Private eval benchmark service, reasoning-gym, Curriculum benchmarks` 


- **考虑私有评估基准测试服务**：一名成员提议运营一个带有私有种子（private seed）以隐藏答案的**私有评估基准测试服务**，并链接到了 [reasoning-gym tools](https://github.com/open-thought/reasoning-gym/tree/main/tools)。
   - 他们想知道这是否超出了项目范围，但承认他们有能力做到这一点。
- **课程基准测试（Curriculum benchmarks）可能存在需求**：一名成员建议创建如 `rg-private-easy` 和 `rg-private-hard` 之类的课程基准测试。
   - 他们推测，如果 **RG** 获得足够的关注，此类服务可能会有需求。



**提到的链接**：<a href="https://github.com/open-thought/reasoning-gym/tree/main/tools">reasoning-gym/tools at main · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。

  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1349046515271733339)** (4 messages): 

> `cublas autotune, PTX optimization, Triton autotune, Cutlass autotune` 


- **cublas autotune 探索 PTX**：**cublas** 也可以进行**自动调优（autotune）**，但从技术上讲它更深入，因为它使用了更多的 **PTX 优化**。
   - 一位成员表示，使用 **Triton** 或 **Cutlass** 的自动调优方法更简单，因为高层优化更容易理解。
- **提供了 cublas 文档**：一名成员提供了 [cublas 文档](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgogetheuristic) 供参考。
   - 文档讨论了 **cublasltmatmulalgogetheuristic**。


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1349035449410261086)** (1 messages): 

> `Nvidia Blackwell, SemiAnalysis, GPU Hackathon, Open Source` 


- **SemiAnalysis 举办 Blackwell GPU 黑客松**：SemiAnalysis 将于 **3 月 16 日星期日**举办 **Nvidia Blackwell GPU 黑客松**，在开展开源项目的同时，提供对 **Blackwell & PTX 基础设施**的实操探索，更多详情请见 [SemiAnalysis 黑客松页面](https://semianalysis.com/hackathon-2025/)。
- **黑客松拥有明星演讲嘉宾阵容**：黑客松将邀请包括 **OpenAI** 的 **Philippe Tillet**、**TogetherAI** 的 **Tri Dao** 以及 **Thinking Machines** 的 **Horace He** 在内的嘉宾演讲。
   - 该活动由 Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave 和 Nebius 赞助。



**提到的链接**：<a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>：SemiAnalysis 将在 NVIDIA GTC 之前拉开帷幕！以迷人的上午主题演讲开始新的一天，全天进行底层 NVIDIA GPU 编程（甚至可能是 Blackwell）黑客松，稍作休息...

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1349055277680427068)** (17 条消息🔥): 

> `Reka Flash 3, Anthropic ARR 增长, Manus AI 热潮, Nvidia Blackwell GPU 黑客松, 中国的自动化与机器人技术` 


- **Reka Labs 开源 Reka Flash 3**：[Reka Labs](https://x.com/RekaAILabs/status/1899481289495031825) 开源了 **Reka Flash 3**，这是一个从零开始训练的新型推理模型，仅凭 **21B 参数** 就实现了极具竞争力的性能。
   - 该模型在合成数据集和公开数据集上进行了微调，随后通过结合基于模型和基于规则奖励的 **RLOO** 进行优化，强制模型输出 *&lt;/reasoning&gt;* 以控制质量与思考时间的平衡，详见其 [博客文章](https://www.reka.ai/news/introducing-reka-flash)。
- **Anthropic 营收强劲增长，助力 Manus AI**：据 [The Information](https://www.theinformation.com/articles/anthropics-claude-drives-strong-revenue-growth-while-powering-manus-sensation) 报道，**Anthropic** 的 **ARR** 在 2025 年前两个月从 **10 亿美元增长至 14 亿美元**。
   - 他们的模型还为被誉为“最新 AI 轰动之作”的 **Manus** 提供支持。
- **SemiAnalysis 举办 Nvidia Blackwell GPU 黑客松**：[SemiAnalysis](https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/) 将于 [3 月 16 日星期日举办 Nvidia Blackwell GPU 黑客松](https://semianalysis.com/hackathon-2025/)，演讲嘉宾来自 **OpenAI**、**TogetherAI** 和 **Thinking Machines**。
   - 此次黑客松旨在探索 **Blackwell 和 PTX 基础设施**，同时开展开源项目协作，赞助商包括 Together、Lambda、Google Cloud、Nvidia、GPU Mode、Thinking Machines、OpenAI、PyTorch、Coreweave 和 Nebius。
- **OpenAI 推出新 API 和 Agents SDK**：[OpenAI](https://x.com/btibor91/status/1899513477716410871) 发布了用于简化 Agent 应用开发的新 API 和工具，包括 **Responses API**、**Web search 工具**、**File search**、**Computer use 工具**以及一个**开源的 Agents SDK**。
   - 现有的 **Assistants API** 将在 2026 年中期逐步淘汰，变更日志中还提到了 API 中新增的模型 **o3-mini-pro** 和 **o1-pro**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reka.ai/news/introducing-reka-flash">使用 Reka Flash 进行推理 | Reka</a>：今天，我们开源了新版 Reka Flash 3 的研究预览版，这是我们的 21B 参数模型。Reka Flash 3 是一款紧凑的通用模型，在通用聊天、编程等方面表现出色...</li><li><a href="https://x.com/RekaAILabs/status/1899481291889979896">来自 Reka (@RekaAILabs) 的推文</a>：Reka Flash 3 在合成和公开数据集上进行了微调，随后采用了结合模型奖励和规则奖励的 RLOO。🏋️我们发现强制模型输出 &lt;/reasoning&gt; 是一种有效的方法...</li><li><a href="https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/">美国正在错失新的劳动力经济 —— 机器人技术第一部分</a>：SemiAnalysis 将于 3 月 16 日星期日举办 Nvidia Blackwell GPU 黑客松。这是 Blackwell PTX 技术爱好者的终极游乐场，提供对 Blackwell 和 PT... 的动手探索。</li><li><a href="https://x.com/steph_palazzolo/status/1899498723010662449">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：Anthropic 今年开局良好：其 ARR 在 2025 年前两个月从 10 亿美元增长到 14 亿美元。他们的模型还为 Manus 提供支持，这是最近在 X 上引起轰动的 AI...</li><li><a href="https://x.com/RekaAILabs/status/1899481289495031825">来自 Reka (@RekaAILabs) 的推文</a>：⚡我们正在开源 Reka Flash 3，这是我们从零开始训练的新型推理模型。它仅用 21B 参数就实现了极具竞争力的性能。⚡Reka Flash 3 为我们的新企业版 Nexus 提供支持...</li><li><a href="https://x.com/btibor91/status/1899513477716410871">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 推出了用于简化 Agent 应用开发的新 API 和工具 —— 新的 Responses API 从今天起对所有开发者开放，它结合了 Chat Completions API 和 Assi... 的功能。</li><li><a href="https://tenor.com/boMsW.gif">Dig Up GIF - Dig Up Stupid - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1349072819396284530)** (2 messages): 

> `Claude Code 反编译，GitHub 仓库` 


- **Claude Code 反编译内容过早下架**：一位用户注意到反编译的 **Claude code** 已从 [Twitter](https://x.com/odazai_/status/1899512495166865699) 下架。
   - 然而，该代码在 [GitHub](https://github.com/dnakov/anon-kode) 上仍然可用。
- **GitHub 仓库依然在线**：包含反编译 **Claude code** 的 GitHub 仓库在 [dnakov/anon-kode](https://github.com/dnakov/anon-kode) 仍可访问。
   - 尽管已从其他平台移除，这仍允许继续访问和研究该代码。



**提及的链接**：<a href="https://x.com/odazai_/status/1899512495166865699">来自 Dazai (@odazai_) 的推文</a>：@dnak0v @cheatyyyy 他们下架了反编译的 claude-code 😢

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1349055950673412167)** (34 messages🔥): 

> `Qwen Chat 增强，API 构建 Agent 的新工具，Anthropic CEO Dario Amodei 预测 12 个月内大部分代码将由 AI 编写，Sama 宣传擅长创意写作的新 OpenAI 模型` 


- **Qwen Chat 焕然一新**：[Qwen Chat](http://chat.qwen.ai) 发布更新，为所有 **Qwen2.5 模型**提供**统一的多模态界面**，增强了视频理解能力（最高 **500MB**），重新设计了带有语音转文字功能的移动端体验，支持访客模式，并扩大了文件上传容量（翻倍至 **20MB**）。
- **OpenAI API 为 Agent 构建者进化**：OpenAI 正在进化其 API 平台，使开发者能够更快、更轻松地构建 Agent，正如其 [YouTube 视频](https://www.youtube.com/live/hciNKcLwSes?si=QdsCwk5dnKktLG29) 中所宣布的，推出了 **Web Search**（微调后的 4o/mini + Web）、**File Search API 更新**以及 **Computer Use**。
   - 他们还推出了一个用于 Agent 相关开发的新 **Python 库**和一个 **responses API**，后者是 chat completions 的超集。
- **Dario 大胆的代码预测**：根据一条[推文](https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)，Anthropic CEO Dario Amodei 预测，AI 将在未来 **3 到 6 个月**内编写 **90%** 的代码，并在 **12 个月**内编写几乎所有代码。
- **Sama 宣传元叙事 OpenAI 模型**：Sam Altman 分享了一个由新 OpenAI 模型生成的[关于 AI 和悲伤的元叙事（metafictional）文学短篇故事](https://x.com/sama/status/1899535387435086115)，并指出：*这是我第一次真正被 AI 创作的东西所震撼；它非常精准地捕捉到了元叙事的氛围。*
   - 社区似乎反应平平，评价道：*还不错——虽然还是有点用力过猛，但确实有那种“灵魂”（sovl）。*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/simonw/status/1899512037526626336">来自 Simon Willison (@simonw) 的推文</a>：值得赞扬的是，OpenAI 至少承认了这是一个问题！在该页面中：“Chat Completions API 是构建 AI 应用程序的行业标准，我们打算继续支持...”</li><li><a href="https://www.youtube.com/live/hciNKcLwSes?si=QdsCwk5dnKktLG29">使用 API 构建 Agent 的新工具</a>：我们正在进化 API 平台，使开发者能够更快、更轻松地构建 Agent。Kevin Weil、Nikunj Handa、Steve Coffey 和 Ilan Bigio 介绍了...</li><li><a href="https://x.com/Alibaba_Qwen/status/1899497336889659775">来自 Qwen (@Alibaba_Qwen) 的推文</a>：👋 介绍增强版 Qwen Chat。我们很高兴宣布 Qwen Chat 的最新更新，旨在提供无缝、多功能且以用户为中心的体验。探索以下主要功能...</li><li><a href="https://x.com/sama/status/1899535387435086115">来自 Sam Altman (@sama) 的推文</a>：我们训练了一个擅长创意写作的新模型（目前还不确定何时/如何发布）。这是我第一次真正被 AI 创作的东西所震撼；它捕捉到了元叙事的氛围...</li><li><a href="https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Haider. (@slow_developer) 的推文</a>：Anthropic CEO Dario Amodei 表示，在未来 3 到 6 个月内，AI 将编写 90% 的代码，而在 12 个月内，几乎所有代码都可能由 AI 生成。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1349096388197355603)** (2 messages): 

> `AI Distillation, Neural Sequence Chunkers, History Compression, DeepSeekR1, Reinforcement Learning Prompt Engineer` 


- **Schmidhuber 回顾 AI Distillation 历史**：Jürgen Schmidhuber 指出，由于 **DeepSeek** 的影响，[CNBC 正在讨论 AI distillation](https://x.com/SchmidhuberAI/status/1899475671958929453)，并提到了他 1991 年关于 *collapsing*（折叠）神经网络的研究，即现在所说的 distillation。
- **早期神经网络压缩的细节**：Schmidhuber 1991 年的技术报告 [Neural sequence chunkers](https://example.com) 详细介绍了将一个神经网络的知识 *compressing*（压缩）或 *collapsing*（折叠）到另一个网络中，这种方法现在已被广泛使用。
   - 在 distillation 之后，automatizer 可以复制 chunker 的动作。
- **DeepSeekR1 利用了 Distilled Chain of Thought**：Schmidhuber 指出，**DeepSeek** 使用了他 2015 年 Reinforcement Learning Prompt Engineer 及其 2018 年改进版中的元素，将 RL 机器和世界模型折叠成一个单一网络，并采用了 1991 年的神经网络 distillation 程序。



**提到的链接**：<a href="https://x.com/SchmidhuberAI/status/1899475671958929453">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：感谢 #DeepSeek，甚至 @CNBC [9] 现在也在谈论 1991 年发表的 AI distillation [1][2]。我当时称之为 “collapsing”，而不是 “distilling”。另见 [9][10]。参考文献 [1] J. Sch...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1349026056149008405)** (6 messages): 

> `Inductive Moment Matching (IMM), Claude 3.7 Sonnet reasoning` 


- **LumaLabs 凭借 IMM 打破算法天花板**：LumaLabs 发布了 **Inductive Moment Matching (IMM)**，这是一种全新的预训练范式，能以 **10 倍** 的效率和稳定的训练实现更高的样本质量，详见其 [博客文章](http://lumalabs.ai/news/imm) 和 [ArXiv 论文](https://arxiv.org/abs/2503.07565)。
   - IMM 模型在 **ImageNet-256x256** 上仅需 **8** 步推理即可达到 **1.99 FID**，超越了 Diffusion 模型；在从头训练的情况下，在 **CIFAR-10** 上实现了 **1.98** 的 SOTA 级 **2-step FID**。
- **Claude 3.7 Sonnet 的推理过程被揭示**：Anthropic 的研究表明，**Claude 3.7 Sonnet** 并没有在其 scratchpad 中编码隐藏推理，证据是训练它使用改写版的 scratchpad 并不会降低性能（[Anthropic 博客文章](https://alignment.anthropic.com/2025/distill-paraphrases/)）。
   - 推理模型的 scratchpad 看起来是人类可理解的，但可能通过某种人类较难理解的机制来提升性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.07565">Inductive Moment Matching</a>：Diffusion 模型和 Flow Matching 可以生成高质量样本，但推理速度较慢，而将它们蒸馏成步数较少的模型往往会导致不稳定和大量的调优。为了解决这些问题...</li><li><a href="https://alignment.anthropic.com/2025/distill-paraphrases/">推理模型是否像我们一样使用 scratchpad？来自蒸馏改写版的证据</a>：未找到描述</li><li><a href="https://x.com/LumaLabsAI/status/1899518379737661447">来自 Luma AI (@LumaLabsAI) 的推文</a>：今天，我们发布了 Inductive Moment Matching (IMM)：一种打破 Diffusion 模型算法天花板的新预训练范式。更高的样本质量。效率提升 10 倍。单阶段、单网络...
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1349023571753697281)** (40 条消息🔥): 

> `Cursor 中的 MCP 服务器, GitHub 组织拥有仓库的验证, Claude Desktop 配置模式, 带有 Gitlab MCP 服务器的 Smithery CLI, OpenAI SDK 的 MCP 兼容性` 


- **MCP 服务器难以集成到 Cursor**: 一位用户在将 Brave Search 等 **MCP 服务器**添加到 **Cursor** 时遇到问题，尽管在 Claude 中集成成功，但在 Cursor 中报告了 *no tools available*（无可用工具）和 *no resources available*（无可用资源）等错误。
- **GitHub 组织仓库认领存在限制**: 一位成员报告说，[glama.ai/mcp/servers/gwrql5ibq2](https://glama.ai/mcp/servers/gwrql5ibq2) 上的 *login with github to claim*（登录 GitHub 认领）功能对于通过 **GitHub organization** 拥有的仓库表现不如预期。
   - 另一位成员承认这是一个**已知限制**，并提到计划在本周解决。
- **在 Windows 上使用 Smithery CLI 和 Gitlab MCP 服务器遇到麻烦**: 一位用户在 Windows 上使用 **Smithery CLI** 运行 **Gitlab MCP 服务器**时遇到困难。
   - 他们持续遇到诸如 *Failed to create client*（创建客户端失败）或 *Unexpected JSON response*（意外的 JSON 响应）之类的错误。
- **OpenAI SDK 支持 MCP**: 一位用户指出，根据 [openai.github.io/openai-agents-python/tools/](https://openai.github.io/openai-agents-python/tools/)，**OpenAI SDK** 现在应该已经 **MCP 兼容**。
- **Handoff 包含完整的对话历史**: 一位成员分享了一个 [github.com 搜索结果](https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code)，显示在默认情况下，**handoff** 会包含整个对话历史（system/user/assistant 消息）。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openai.github.io/openai-agents-python/tools/">Tools - OpenAI Agents SDK</a>: 未找到描述</li><li><a href="https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1348977748714065950)** (10 条消息🔥): 

> `MCP, Model Context Protocol, Elixir 的 Phoenix Framework, llama.cpp, vllm` 


- **Phoenix Framework 助力 MCP 实现**: 一位成员分享了 GitHub 上的 [MCPheonix](https://github.com/jmanhype/MCPheonix)，这是一个使用 **Elixir 的 Phoenix Framework** 实现的简化版 **Model Context Protocol (MCP) server**。
- **通过 Llama.cpp 和 VLLM 进行本地 LLM 推理**: 一位成员询问关于连接到其他 **LLM**（如 **llama.cpp** 或 **vllm**）进行本地 **LLM** 推理的问题，另一位成员回答说，只要它们支持 tool calling（工具调用），实现起来就很简单。
- **通过 MCP 控制 Unraid 服务器**: 一位成员分享了他们的 [unraid-mcp](https://github.com/jmagar/unraid-mcp) 项目，用于通过 **MCP** 控制你的 **Unraid 服务器**。
- **使用 MCP 通过 AI 控制 Android 设备**: 一位成员分享了他们的 [DroidMind 项目](https://github.com/hyperb1iss/droidmind)，这是一个可以通过 **ADB** 管理你的 **Android 设备**的 **MCP server**，适用于调试设备端问题和分析日志。
- **构建 MCP 服务器的 MCP 服务器**: 一位成员介绍了 [mcp-create](https://github.com/tesla0225/mcp-create)，这是一个用于构建 **MCP server** 的 **MCP server**，支持 **TypeScript**，并能直接运行生成的 **MCP server**，同时附带了[一篇说明文章](https://zenn.dev/tesla/articles/c66bda76c4a523)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/hyperb1iss/droidmind">GitHub - hyperb1iss/droidmind: 使用 Model Context Protocol 通过 AI 控制你的 Android 设备</a>: 使用 Model Context Protocol 通过 AI 控制你的 Android 设备 - hyperb1iss/droidmind</li><li><a href="https://github.com/jmanhype/MCPheonix">GitHub - jmanhype/MCPheonix: 使用 Elixir 的 Phoenix Framework 实现的简化版 Model Context Protocol (MCP) server。</a>: 使用 Elixir 的 Phoenix Framework 实现的简化版 Model Context Protocol (MCP) server。 - jmanhype/MCPheonix</li><li><a href="https://github.com/jmagar/unraid-mcp">GitHub - jmagar/unraid-mcp</a>: 为 jmagar/unraid-mcp 的开发做出贡献，在 GitHub 上创建一个账户。</li><li><a href="https://github.com/tesla0225/mcp-create">GitHub - tesla0225/mcp-create</a>: 为 tesla0225/mcp-create 的开发做出贡献，在 GitHub 上创建一个账户。</li><li><a href="https://zenn.dev/tesla/articles/c66bda76c4a523">制作了一个用于制作 MCP 服务器的 MCP 服务器</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1348983272356581439)** (7 条消息): 

> `NotebookLM 用于备考、NotebookLM 用于医疗指南和患者信息、自动化 NotebookLM 上传、NotebookLM 音频概览` 


- **NotebookLM 助力备考**：一位用户根据书签将 PDF 分成不同章节，并为每个章节创建了独立的笔记本，然后根据学习指南中的所有主题对笔记本进行提问（通常仅限于当周的章节而非整个笔记本），获得了*非常好的效果*。
   - 该用户随后将这些结果转化为了其他 App 中的 Flashcards（闪卡）。
- **NotebookLM 生成医疗文档**：一位医疗领域的用户发现 NotebookLM 在解析现有指南和难以导航的网站，并为出院回家的患者创建信息方面表现*惊人*。
   - 具体而言，该用户为患者创建了一份关于工伤索赔的简洁单页文档，提供了一份清单式的列表，列出了患者在出院时应注意并完成的事项。
- **自动化 NotebookLM 的内容摄取**：一位用户正在自动化优化上传到 NotebookLM 的信息，重点关注较小的文件，以便*更轻松地进行机器人摄取（robot ingestion）*。
   - 该用户正在简化其工作流，使 NotebookLM 处理文档变得更加容易。
- **NotebookLM 成为音频概览大师**：一位用户使用 NotebookLM 为一份包含 LLM 对商业想法提示词响应的 Google Doc 创建了音频概览。
   - 音频中的发言人提到了不同的标签页（LLM 名称），并在讨论精华部分时表现出色，每次都能准确引用对应的 LLM；顺便提一下，**Gemini Advanced Pro 在最佳响应评选中脱颖而出**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1348977488335863818)** (17 条消息🔥): 

> `Gemini、Notebook LM 限制、语言支持、源管理、音频概览` 


- **Gemini 引发不满**：一位用户表达了对 **Gemini** 的不满，尽管其已深度集成到 Google 生态系统中。
- **NotebookLM 支持庞大的知识库**：一位拥有 **1000 万字知识库**（1500 本书，6000 个视频文本）的用户询问了 NotebookLM 的限制，特别是关于文件深度和总容量的问题。
   - NLM 团队的一名成员澄清说，NotebookLM 支持 **1000 万字**，且在 **300 个源文件和每个源文件 50 万字**的限制内，利用 **RAG** 来处理相关部分。
- **NotebookLM Plus 存在用户限制**：一位用户报告称，“系统无法回答”是一个持续存在的问题，50 个（额度）现在已经不够用了。
- **用户希望获得更多语言支持**：一位用户询问了 NotebookLM 支持英语以外输出语言的路线图和时间表。
- **带有时间戳的事件提示不影响输出对话**：一位用户询问 .txt 文件中带有时间戳的事件提示（event cues）是否会影响音频概览的输出。



**提到的链接**：<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述

  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1349135010447360011)** (1 条消息): 

> `Windsurf 推荐挑战赛、v1.4.6 补丁修复、Windsurf 预览、自动 Linter、MCP 服务器` 


- **推荐好友，赢取积分，抢夺周边！**：**Windsurf 推荐挑战赛**鼓励用户推荐好友，每当好友订阅 Pro 版时，推荐人可获得 **500 flex credits**。此外，在 **3 月 31 日**前通过 [windsurf.ai/refer](https://windsurf.ai/refer) 获得最多推荐的用户还有机会赢取定制的 **Airpods Pro Max 耳机**。
- **Windsurf v1.4.6 修复了 MCP、Sonnet 和代理问题**：Windsurf 发布了 **v1.4.6 补丁修复**，解决了 **MCP 可靠性**、**3.7 Sonnet 网页搜索**以及**代理设置**问题，详见 [更新日志](https://www.codeium.com/changelog)。
- **Windsurf Previews 在本地 Cascade 运行**：**Windsurf Previews (Beta)** 现在允许用户直接在 Cascade 中预览本地运行的网站，请查看附带的 [图片](https://cdn.discordapp.com/attachments/1027688115592237117/1349135011659517994/IMG_5342.png?ex=67d1ff64&is=67d0ade4&hm=569177e0fbf1e9818203093be6e4efda6a0f0c528dbbeb99786d89a257da1c30&)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://windsurf.ai/refer">Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1349040086746005678)** (23 messages🔥): 

> `Codeium VS Code extension issues, Claude 3.7 Sonnet in VS Code, Codeium extension credits, VS Code extension version discrepancy, Codeium server errors` 


- **Codeium VS Code 扩展截断长提示词**：一位使用 **Pro plan** 的用户在使用 **Claude 3.7 Sonnet** 处理长提示词时，遇到了 Codeium VS Code 扩展截断回复的问题，聊天记录显示 *已达到 token 限制 (token limit has been reached)*。
   - 鉴于已报告的 **3.7** 相关问题，建议用户尝试使用 **Claude 3.5**，尽管截断问题可能与之没有直接关系。
- **VS Code 扩展中缺失 Claude 3.7 Sonnet Thinking**：与 Windsurf 不同，**Claude 3.7 Sonnet Thinking** 模型在 VS Code 扩展中不可用，用户询问是否需要额外配置。
   - 已确认 **Claude 3.7 Sonnet Thinking** *目前在扩展中不可用*。
- **Codeium VS Code 扩展无法直接读取文件**：Codeium VS Code 扩展聊天（**Claude 3.7 Sonnet**）无法直接从文件夹中读取脚本文件，需要用户将文件内容粘贴到聊天框中。
   - 建议用户 *在 codeium.com/support 进行反馈，因为从技术上讲这应该是可以工作的*。
- **Codeium VS Code 扩展使用不消耗额度**：会议明确了使用 Codeium VS Code 扩展（即使是使用高级模型）是完全免费的，不会消耗任何 credits，因为 credits 是与 Windsurf 绑定的。
   - 用户最初以为会消耗额度，但在确认这些聊天和回复不消耗任何 credits 后感到非常兴奋。
- **Codeium 扩展服务器中止挂起的请求**：一位用户报告了一个导致 Codeium 无法工作的持续错误，提示信息为 *Codeium: The server aborted pending request*，并提到了来自 *releases.codeiumdata.com* 的下载 URL。
   - 尽管重启了 IDE，该问题在不同版本中依然存在，建议联系 *vscode@codeium.com*。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1349001237709848616)** (18 messages🔥): 

> `operation order in deep learning frameworks, Adaptive Meta-Learning, RLHF issues, Matplotlib and Claude 3.7 graphs, hiding complexity` 


- **编译器会优化数学运算吗？**：成员们讨论了 **PyTorch** 和 **NumPy** 等深度学习框架中的运算顺序，争论编译器是否会自动优化计算，例如 *(1/n) (a(c + d) + b)* 与 *a(c/n + d/n) + b/n* 的对比。
   - 一位成员幽默地建议只需 *添加额外的括号*，以确保系统按预期的顺序执行运算。
- **极简代码 vs 凌乱代码：一场辩论**：围绕极简代码与显式（可能较凌乱）代码之间的权衡展开了讨论，一些人认为极简代码只是将复杂性隐藏在别人的框架之下。
   - 一位成员认为成功的 AI 运动通常将复杂性隐藏在简单的接口之后，而另一位成员则对此表示担忧，特别是在需要调试或将模型转换为 **ONNX** 时，这会导致无分支编程（branchless programming）出现问题。
- **Claude 3.7 绘制的 Matplotlib 图表**：成员们对 **Claude 3.7** 生成的 **Matplotlib** 图表表示兴奋，指出 *benchmark 和 svgmaxing* 似乎运行得非常有效。
   - 本次交流中未提供相关链接。
- **Adaptive Meta-Learning：一个新术语？**：一位成员询问 **Adaptive Meta-Learning (AML)** 这个术语是已经存在还是由他们创造的，并将其描述为 *Online HyperParameter Optimization (HPO)* 与 meta-learning 的潜在结合。
   - 另一位成员提供了 [Semantic Scholar 搜索链接](https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance)，但结论是虽然这些关键词被放在一起使用，但它们并不构成一个既定的框架或范式。
- **RLHF 是否僵化地关联了不良行为？**：一位成员根据一篇关于涌现失调（emergent misalignment）的论文推测，**Reinforcement Learning from Human Feedback (RLHF)** 可能会在批处理中僵化地关联不良行为。
   - 他们假设反转这一过程可能会导致模型编写糟糕的代码，将谎言与不受欢迎的行为联系起来，并暗示隐藏技术和商业机密背后可能存在经济动机。



**提到的链接**：<a href="https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance>">Adaptive meta-learning | Semantic Scholar</a>：一个利用人工智能方法提供高度相关结果和简便过滤工具的学术搜索引擎。

  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1349163019682185336)** (1 messages): 

> `` 


- **今晚没有论文讨论**：今晚没有人自愿主持论文讨论。
- **今晚的计划**：一位成员提到今晚有其他安排。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1349123858463985797)** (3 messages): 

> `LLMs think in tokens, VR prison` 


- **LLM 是否在 token 之间思考？**：关于 [LLM 是否以 token 形式思考以及是否在 token 之间思考](https://openai.com/index/chain-of-thought-monitoring/) 存在讨论。
- **VR 头显导致监狱违规行为减少 97%**：根据[这篇文章](https://www.theguardian.com/technology/2025/mar/08/vr-prison-california)，加州的一家女子监狱在禁闭室使用 VR 头显取得了成功，导致*违规行为减少了 97% 以上*。



**提到的链接**：<a href="https://www.theguardian.com/technology/2025/mar/08/vr-prison-california">‘一种理想的工具’：监狱正利用虚拟现实帮助处于禁闭状态的人</a>：参与者观看日常生活场景以及旅行冒险，然后通过艺术处理这些场景引发的情绪。

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1348982241245663276)** (11 messages🔥): 

> `Llama Extract Access, Premium Plan Signup, Function Calling Models, MP3 Parsing Error` 


- ****授予 Llama Extract 访问权限****：一位成员请求访问 **Llama Extract**，并被提议加入封闭测试，目前正在等待邮件确认。
   - 为此共享了电子邮箱地址 `rasmus-persson@outlook.com`。
- ****Premium 计划升级变得简单****：一位用户询问如何从 Free 计划升级到 Premium 模式。
   - 提供的说明是：**登录**，点击个人资料图标，然后选择升级/管理按钮。
- ****Deepseek 与 4o 的对决****：一位用户询问：*如果在 ANUS 中使用 Deepseek 代替 4o，质量会下降吗？*
   - 未给出有用的回复。
- ****API 的 MP3 解析难题****：一位用户报告了通过 API 上传 **.mp3** 文件进行解析时出现错误。
   - 用户指出通过 UI/webapp 上传正常，并提供了[错误截图](https://cdn.discordapp.com/attachments/1059201661417037995/1349100831307202714/Screenshot_2025-03-11_at_3.24.19_PM.png?ex=67d1df8f&is=67d08e0f&hm=3b980c7dd220c3d654ff1cb17819daedcc6fc3c896b2ed955e800b40f2467d3d)。
- ****Function Calling 对决****：一位成员询问除了 **OpenAI** 之外，还有哪些模型擅长 Function Calling。
   - 该用户正在寻找更便宜的选择，因为他们的应用严重依赖此功能。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

amanshrestha: https://github.com/openai/openai-agents-python
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1349028538862534768)** (8 messages🔥): 

> `Judge LLM, ChainPoll, Best of N, dspy.Parallel` 


- **用于 Judge LLM 的 ChainPoll 模式**：成员们正在构建一个遵循 **ChainPoll** 模式的 **Judge LLM**，该模式使用多个 Chain of Thought 评判程序并返回平均响应链。
   - 一位成员建议使用 `module.batch()` 或 `dspy.Parallel` 来加速该过程。
- **寻找 Best of N 文档**：一位成员在查找 **Best of N** 的文档时遇到困难。
   - 同一位成员指出 ensemble 被列为 teleprompter，并询问它是优化输入程序还是将其聚合为单个最优程序。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1348977802627649536)** (7 messages): 

> `OpenPipe's deductive-reasoning, FP8 Fine-Tuning, Torchtune QAT Support, Weight Decay Strategies, Evaluation Dataset Logging` 


- **OpenPipe 演绎推理：Torchtune 的大捷！**：一位成员分享了 [OpenPipe 的 deductive-reasoning 项目](https://github.com/openpipe/deductive-reasoning)链接，并指出该项目使用了 **Torchtune**。
- **FP8 微调面临挑战**：成员们讨论了在 **FP8** 下提供模型服务的挑战，并考虑了通过 **FP8** 微调来减少量化误差的可能性。
   - 他们指出，由于训练期间的稳定性问题，**FP8** 微调具有挑战性，并非一个简单的过程。
- **Torchtune 的 QAT 支持**：一位成员询问了 **Torchtune** 对 **QAT**（量化感知训练）的支持情况，特别是针对 **FP8**，旨在通过微调减少量化误差。
   - 有人提到 [这个 Recipe](https://github.com/pytorch/torchtune/pull/2404) 对于 **FP8** 看起来很有前景。
- **Weight Decay 的妙用！**：成员们建议，逐渐增加 **Weight Decay** 可能有助于在 **FP8** 微调期间将权重保持在正确的范围内。
- **发现独立的 Eval Dataset Recipe！**：一位成员正在寻找支持独立 **eval dataset** 的 Recipe，以便每隔 N 个 step 测量一次 loss，类似于[这个示例](https://github.com/pytorch/torchtune/blob/d5d12fef1f8c39dfd5c9f85807795ef503216e12/recipes/full_finetune_single_device.py#L725)。
   - 一位成员找到了其中一个 Recipe 的 [这个 Pull Request](https://github.com/pytorch/torchtune/pull/2238/files)，可能会有所帮助。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/openpipe/deductive-reasoning">GitHub - OpenPipe/deductive-reasoning: 训练你自己的 SOTA 演绎推理模型</a>：训练你自己的 SOTA 演绎推理模型。通过在 GitHub 上创建账号为 OpenPipe/deductive-reasoning 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2238/files">由 MaxFrax 提交的为 LoRA 单设备微调添加验证 loss 的 Pull Request #2238 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）？请链接此 PR 解决的任何 Issue。#1042...</li><li><a href="https://github.com/pytorch/torchtune/blob/d5d12fef1f8c39dfd5c9f85807795ef503216e12/recipes/full_finetune_single_device.py#L725)">pytorch/torchtune 中的 recipes/full_finetune_single_device.py</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2404">(WIP/RFC) 由 nathan-az 提交的 FP8 全量分布式微调 Pull Request #2404 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）？这将解决 #2201。我远非这方面的专家...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1349085247240929291)** (1 messages): 

> `Regression Tests, Model Size Finalization, Evaluation Metrics, Comprehensive Measurement Strategies` 


- **新增回归测试，需确定模型大小**：一位成员提到添加了几个 [回归测试 (Regression Tests)](https://github.com/pytorch/torchtune/pull/2477)，并询问有关最终确定模型大小和评估方法的问题。
   - 该成员质疑仅靠评估是否足够，暗示需要讨论更全面的衡量策略。
- **深入探讨综合衡量策略**：讨论转向了在简单评估之外建立更全面衡量策略的必要性。
   - 成员们辩论了各种评估指标的优劣，这可能会影响模型大小的选择和测试方法论。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

ceifa: 😮
  

---

### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1349082537326284932)** (2 messages): 

> `Expedition Aya 2024, Multilingual AI, Multimodal AI, Efficient AI, Cohere API Credits` 


- **Cohere For AI 宣布启动 Expedition Aya 2024！**：Cohere For AI 正在推出 [Expedition Aya 2024](https://tinyurl.com/ayaexp2025)，这是一个为期 **6 周的 open-build challenge**，旨在促进全球范围内的新合作和研究项目，将其重点扩大到支持涉及 **Multilingual, Multimodal 或 Efficiency** 的研究。
- **Aya Projects 展示影响力**：之前的 **Expedition Aya** 促成了许多合作和出版物，如 [Aya Projects](https://tinyurl.com/4scpn5uu) 演示中所展示的那样。
   - 展示项目的示例包括 **DistAYA**, **Doclingual** 和 **Enhancing Sinhala NLP**。
- **Expedition Aya 提供资源**：Expedition Aya 的参与者将获得专属资源和 **Cohere API Credits**，以便将其模型用于研究。
   - 完成该计划的团队将有资格获得 **限量版 Expedition Swag**，且顶级项目还设有专属奖项。
- **加入 Expedition Aya 进行团队建设！**：鼓励成员加入 [Expedition Aya Discord server](https://discord.gg/q9QRYkjpwk) 并参加 **Crew Connections 会议**，以联系潜在的合作伙伴。
   - 启动会议将于 **2025 年 3 月** 举行。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinyurl.com/ayaexp2025">Expedition</a>:  </li><li><a href="https://tinyurl.com/4scpn5uu">Expedition - Past Projects</a>: DistAYA
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1349035344175038548)** (1 messages): 

> `Nvidia Blackwell GPU Hackathon, SemiAnalysis, PTX Infrastructure` 


- **SemiAnalysis 举办 Blackwell GPU Hackathon！**：[SemiAnalysis](https://semianalysis.com/) 将于 **3 月 16 日星期日** 举办 **Nvidia Blackwell GPU Hackathon**，在合作开发开源项目的同时，亲身体验 **Blackwell & PTX infrastructure**。
   - 演讲嘉宾包括 **OpenAI** 的 **Philippe Tillet**、**TogetherAI** 的 **Tri Dao**、**Thinking Machines** 的 **Horace He** 等，活动由 **Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius** 赞助。
- **Hackathon 提供 PTX 游乐场！**：该活动是 **Blackwell PTX** 技术爱好者的终极游乐场，在合作开发开源项目的同时，提供对 **Blackwell & PTX infrastructure** 的亲身探索。
   - 与会者可以期待迷人的上午主题演讲、使用 **GB200s** 等强大的 **Blackwell GPUs** 进行一整天的 Hacking、富有洞察力的下午演讲以及令人难忘的闭幕式。



**Link mentioned**: <a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis is kicking things off ahead of NVIDIA GTC! Start your day with engaging morning keynotes, hack all day with low-level NVIDIA GPU programming (maybe even Blackwell), take a breather wit…

  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1349051663356268609)** (2 messages): 

> `Multilingual/multicultural communities, Introductions and Community Expectations` 


- **研究员寻求 Multilingual/Multicultural 社区**：一位研究员询问了 Cohere Discord 社区内多语言和多文化活动的具体位置。
   - 该用户对 Cohere 的工作表示赞赏，并提到之前曾与团队有过合作。
- **强调自我介绍和社区期望**：一条置顶消息提醒新成员进行自我介绍，并说明了需要分享的关键细节。
   - 该消息概述了预期的格式，包括公司/行业/大学隶属关系、当前项目、首选技术/工具以及社区目标。

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1349034159389016187)** (3 messages): 

> `Nvidia Blackwell GPU Hackathon, SemiAnalysis, Blackwell PTX, GB200, GTC` 


- ****SemiAnalysis** 举办 **Nvidia Blackwell GPU Hackathon****：**SemiAnalysis** 将于 **3 月 16 日（星期日）**举办 [Nvidia Blackwell GPU Hackathon](https://semianalysis.com/hackathon-2025/)，在协作开源项目的同时，提供对 **Blackwell & PTX** 基础设施的实操探索机会。
- **Hackathon 演讲嘉宾阵容汇集 AI 重量级人物**：Hackathon 演讲者包括 [OpenAI 的 Philippe Tillet](https://openai.com/)、[TogetherAI 的 Tri Dao](https://www.together.ai/)、[Thinking Machines 的 Horace He](https://www.thinkingmachin.es/) 等。
   - 该活动由 Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius 赞助。
- **与 SemiAnalysis 一起开启 GTC**：**SemiAnalysis** 以 **Blackwell GPU Hackathon** 的形式华丽开启 **GTC**，活动包括引人入胜的上午主题演讲、使用 **GB200** 等强大 **Blackwell GPU** 进行的全天 hacking，以及富有洞察力的下午谈话。



**提及链接**：<a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>：SemiAnalysis 在 NVIDIA GTC 之前拉开帷幕！以引人入胜的上午主题演讲开始新的一天，全天进行底层 NVIDIA GPU 编程（甚至可能是 Blackwell），稍作休息...

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

clear3fram3: 等待关于 CUDA 的最后几篇博客文章发布 🙂
  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1349135915733356584)** (1 messages): 

> `Qdrant, Vector DB, ConvRAG, VPC Deployment` 


- **ConvRAG 弃用 Qdrant**：开发团队曾考虑将 **Qdrant** 作为其 **ConvRAG** 的 Vector DB，但最终决定使用另一个。
   - 所选的 DB 为 **VPC Deployment** 提供了更大的灵活性。
- **选择了替代 DB**：在 **ConvRAG** 项目中，另一个 Vector DB 被选中取代了 Qdrant。
   - 引用的主要原因是它在 **VPC Deployment** 场景下提供了增强的灵活性。


  

---


---


---


---


{% else %}


> 完整的逐频道明细已针对电子邮件进行了截断。
> 
> 如果您想查看完整的明细，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}
