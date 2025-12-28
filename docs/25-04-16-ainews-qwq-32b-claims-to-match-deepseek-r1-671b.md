---
companies:
- alibaba
- openai
- deepseek-ai
date: '2025-04-16T19:06:15Z'
description: '阿里巴巴通义千问（Alibaba Qwen）发布了 **QwQ-32B** 模型。这是一款拥有 **320 亿参数**的推理模型，采用了创新的两阶段强化学习（RL）方法：第一阶段通过准确性验证器和代码执行服务器，针对数学和编程任务扩展强化学习规模；第二阶段则将强化学习应用于指令遵循和对齐等通用能力。


  与此同时，OpenAI 向 Plus 用户推出了 **GPT-4.5**，用户对其编程表现的评价褒贬不一，但推理成本的优化受到了关注。QwQ 模型旨在与 **DeepSeek-R1**
  等更大规模的 MoE（混合专家）模型展开竞争。有用户尖锐地批评道“GPT-4.5 在编程方面完全无法使用”，而另一些人则称赞其通过扩展预训练规模显著提升了推理能力。'
id: 588aee90-b755-446c-9e0b-a05e07085b52
models:
- qwen-2.5-plus
- qwq-32b
- deepseek-r1
- gpt-4.5
- gpt-3
- davinci
original_slug: ainews-qwq-32b-claims-to-match-deepseek-r1-671b
people:
- aidan_mclau
- sama
- scaling01
- juberti
- polynoamial
- reach_vb
title: QwQ-32B 声称其性能可比肩 DeepSeek R1-671B。
topics:
- reinforcement-learning
- math
- code-execution
- instruction-following
- alignment
- reasoning
- model-release
- model-benchmarking
- scaling
- performance
- inference-costs
type: private
---

<!-- buttondown-editor-mode: plaintext -->**两阶段 RL 就足够了？**

> 2025年3月5日至3月6日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**227** 个频道和 **3619** 条消息）。预计节省阅读时间（以 200wpm 计算）：**351 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如[去年 11 月](https://qwenlm.github.io/blog/qwq-32b-preview/)预告以及[上个月](https://qwenlm.github.io/blog/qwq-max-preview/)再次提到的，阿里巴巴 Qwen 团队[终于发布](https://x.com/altryne/status/1897373582076076387)了 QwQ 的最终版本。这是他们的 Qwen2.5-Plus + Thinking (QwQ) 后训练版本，其性能数据可与 R1 媲美，而 R1 作为一个 MoE 模型，规模比它大 20 倍。


![image.png](https://assets.buttondown.email/images/5f93a399-1fbf-4d0c-9e32-55f90b77fcae.png?w=960&fit=max)


目前还处于早期阶段，因此尚无独立的第三方验证，但 Qwen 团队已经做了最基本的工作来向我们证明，他们并没有为了获得这一结果而简单地对基准测试进行过拟合——因为他们在非数学/编码基准测试中依然表现出色，并用一段文字解释了实现方法：

> - **在初始阶段，我们专门针对数学和编码任务扩展了 RL。** 我们没有依赖传统的奖励模型，而是利用数学问题的准确性校验器（accuracy verifier）来确保最终解的正确性，并使用代码执行服务器来评估生成的代码是否成功通过预定义的测试用例。随着训练轮次的增加，这两个领域的性能都在持续提升。
> - **在第一阶段之后，我们增加了另一个阶段的 RL 以提升通用能力。** 它使用来自通用奖励模型和一些基于规则的校验器的奖励进行训练。我们发现，通过少量步数的这一阶段 RL 训练，可以提高其他通用能力（如指令遵循、人类偏好对齐和 Agent 性能），且数学和编码性能不会出现显著下降。

更多信息——如论文、示例数据、示例代码——将有助于理解，但对于 2025 年的开源模型披露来说，这已经足够诚意了。QwQ-32B 在 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?pinned=Qwen%2FQwQ-32B-Preview_bfloat16_1032e81cb936c486aae1d33da75b2fbcd5deed4a_True%2Cdeepseek-ai%2FDeepSeek-R1-Distill-Llama-70B_bfloat16_07a264a567ba0863a4ab34fdb3c2b8a54e0bb494_True%2Cmeta-llama%2FLlama-3.3-70B-Instruct_bfloat16__True&params=65%2C141&official=true) 上排名还需要一段时间，但这里是现状提醒：Thinking 后训练模型并不一定在所有方面都优于其 Instruct 前代模型。


![image.png](https://assets.buttondown.email/images/c0d6fd00-a7f0-419c-8bef-368c72ecbf2e.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**AI 模型发布与基准测试**

- **GPT-4.5 发布与性能表现**：[@sama](https://twitter.com/sama/status/1897065339617468918) 宣布向 **Plus users** 推出 **GPT-4.5**，将在几天内分阶段开放访问，以管理速率限制并确保良好的用户体验。[@sama](https://twitter.com/sama/status/1897348424984617215) 随后确认推送已经开始，并将在几天内完成。[@OpenAI](https://twitter.com/OpenAI/status/1897346510821711959) 强调这是 **“成为 Plus 用户的伟大一天”**。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897348131777626239) 幽默地警告说，由于 **GPT-4.5** 的“笨重（chonkiness）”，可能会导致 **GPU meltdown**。然而，用户对其编程性能的初步反馈褒贬不一，[@scaling01](https://twitter.com/scaling01/status/1897364892891451734) 发现 **GPT-4.5 在 ChatGPT Plus 中无法用于编程**，理由是变量定义、函数修复以及重构时的懒惰问题。[@scaling01](https://twitter.com/scaling01/status/1897359350580293924) 重申 **“GPT-4.5 无法用于编程”**。[@juberti](https://twitter.com/juberti/status/1897121314340790761) 认为 **GPT-4.5 的推理成本与 2022 年夏天的 GPT-3 (Davinci) 相当**，表明计算成本随时间下降。[@polynoamial](https://twitter.com/polynoamial/status/1897372733098578311) 注意到 **GPT-4.5 解决推理问题的能力**，并将其归功于预训练的扩展（scaling pretraining）。
- **Qwen QwQ-32B 模型发布**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1897361654763151544) 发布了 **QwQ-32B**，这是一款全新的 **320 亿参数推理模型**，声称可与 **DeepSeek-R1** 等尖端模型媲美。[@reach_vb](https://twitter.com/reach_vb/status/1897362929009516920) 兴奋地宣布 **“We are so unfathomably back!”**，**Qwen QwQ 32B** 的表现优于 **DeepSeek R1** 和 **OpenAI O1 Mini**，并采用 **Apache 2.0 license**。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1897386174605586736) 强调 **Qwen QwQ-32B** 是一款**小巧但强大的推理模型**，击败了 **DeepSeek-R1 (671B)** 和 **OpenAI o1-mini**，并宣布其已在 **Hyperbolic Labs** 上线。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897374991576113645) 也对 **Qwen** 团队的发布表示兴奋，认为其表现与 **DeepSeek** 同样令人印象深刻。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1897368160548102628) 指出了 **Qwen 的 “cold-start” 方法** 以及与 **R1** 的直接竞争。
- **AidanBench 更新**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897067567589790013) 发布了 **aidanbench 更新**，指出 **GPT-4.5 综合排名第 3，在非推理模型（non-reasoner）中排名第 1**，而 **Claude-3.7** 模型的得分低于 **newsonnet**。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897067571817660916) 解释了对 **O1 scores** 的修正，原因是之前误分类了超时情况。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897128908312707235) 指出了 **Chain of Thought (CoT)** 推理的高昂成本，并提到有人抱怨 **GPT-4.5** 的成本，却没人抱怨 **Claude-3.7-thinking**。[@scaling01](https://twitter.com/scaling01/status/1897301054431064391) 分析了 **AidanBench 结果**，认为 **Claude Sonnet 3.5 (new)** 表现出持续的顶级性能，而 **GPT-4.5** 的高分可能是由于对单个问题的记忆（memorization）。
- **Cohere Aya Vision 模型发布**：[@_akhaliq](https://twitter.com/_akhaliq/status/1897035171817312393) 宣布 **Cohere 在 Hugging Face 上发布了 Aya Vision**，强调其在**多语言文本生成和图像理解**方面的强劲表现，优于 **Qwen2.5-VL 7B**、**Gemini Flash 1.5 8B** 和 **Llama-3.2 11B Vision** 等模型。
- **Copilot Arena 论文**：[@StringChaos](https://twitter.com/StringChaos/status/1897047614136443083) 重点介绍了 **Copilot Arena 论文**，该论文由 [@iamwaynechi](https://twitter.com/iamwaynechi) 和 [@valeriechen_](https://twitter.com/valeriechen_) 领导，提供了**直接来自开发者的 LLM 评估**，包含关于模型排名、生产力以及跨领域和语言影响的真实见解。
- **VisualThinker-R1-Zero**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1897359430225944749) 讨论了 **VisualThinker-R1-Zero**，这是一个通过将 **Reinforcement Learning (RL)** 直接应用于 **Qwen2-VL-2B base model** 来实现多模态推理的 **2B 模型**，在 **CVBench** 上达到了 **59.47% 的准确率**。
- **Light-R1**：[@_akhaliq](https://twitter.com/_akhaliq/status/1897149072706003023) 宣布 **Light-R1 在 Hugging Face 上线**，通过 **Curriculum SFT & DPO** 以 1000 美元的成本超越了 **R1-Distill from Scratch**。
- **Ollama 新模型**：[@ollama](https://twitter.com/ollama/status/1897109918731153473) 发布了 **Ollama v0.5.13**，包含新模型：支持 function calling 的 **Microsoft Phi 4 mini**、用于视觉文档理解的 **IBM Granite 3.2 Vision** 以及 **Cohere Command R7B Arabic**。

**开源 AI 与社区**

- **Weights & Biases 被 CoreWeave 收购**: [@weights_biases](https://twitter.com/weights_biases/status/1897085419239702821) 宣布他们被 AI 超大规模算力提供商 **CoreWeave** 收购。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1897099583366553733) 称赞 **Weights & Biases** 是**最具影响力的 AI 公司**之一，并祝贺其被 **CoreWeave** 收购。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897085650110963725) 也强调这次**收购**对 AI infra 社区来说是重大新闻。[@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1897073510142632269) 报道了收购谈判，提到了一项潜在的 **17 亿美元交易**，旨在将 CoreWeave 的客户群多样化并扩展到软件领域。[@alexandr_wang](https://twitter.com/alexandr_wang/status/1897364936432738727) 和 [@alexandr_wang](https://twitter.com/alexandr_wang/status/1897364935279342043) 分享了报道此次收购的文章。
- **Keras 3.9.0 发布**: [@fchollet](https://twitter.com/fchollet/status/1897377772038971462) 宣布 **Keras 3.9.0 发布**，带来了新的 ops、图像增强层、错误修复、性能改进以及新的 rematerialization API。
- **Llamba 模型**: [@awnihannun](https://twitter.com/awnihannun/status/1897376858544726239) 推广了 **Cartesia 的 Llamba 模型**，这是高质量的 **1B、3B 和 7B SSMs**，支持 MLX 以实现快速的设备端执行。
- **Hugging Face 集成**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897317872604496196) 宣布了 **Hugging Face 更新**，允许开发者**直接从 Hugging Face 使用 Gradio 部署模型**，选择推理提供商，并要求用户登录以进行计费。[@sarahookr](https://twitter.com/sarahookr/status/1897343637438259268) 提到与 **Hugging Face** 合作发布 **Aya Vision**。

**AI 应用与用例**

- **Google 搜索中的 AI Mode**: [@Google](https://twitter.com/Google/status/1897332927194640788) 推出了 **Search 中的 AI Mode**，这是一项提供 **AI 回答和后续提问**的实验。[@Google](https://twitter.com/Google/status/1897332929136877854) 详细介绍了 **AI Mode**，它在 **AI Overviews** 的基础上扩展了高级推理和多模态能力，并向 **Google One AI Premium 订阅用户**推出。[@Google](https://twitter.com/Google/status/1897332925382975619) 还宣布在 **AI Overviews 中加入 Gemini 2.0** 以处理编程和数学等复杂问题，并**开放无需登录即可访问 AI Overviews** 的权限。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1897349606796910736) 祝贺搜索团队发布 **AI Mode**，期待它能为更广泛的受众提供帮助。[@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1897340342376321181) 强调了 **Gemini 通过 AI Mode 与搜索的集成**。
- **AI Agent 和 Agentic 工作流**: [@llama_index](https://twitter.com/llama_index/status/1897337055358935058) 推广了集成到软件流程中的 **Agentic 文档工作流**，用于知识 Agent。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897323791669133521) 和 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1897389514034688313) 宣布与 **LlamaIndex** 合作推出关于**事件驱动的 Agentic 文档工作流**的新短课程，教授如何构建用于表单处理和文档自动化的 Agent。[@LangChainAI](https://twitter.com/LangChainAI/status/1897316172317778339) 宣布了即将举行的 AI Agent 会议 **Interrupt**，届时来自 **Harvey AI** 的 [@benjaminliebald](https://twitter.com/benjaminliebald) 将分享关于构建法律 Copilot 的内容。[@omarsar0](https://twitter.com/omarsar0/status/1897336282654892301) 分享了关于构建 AI Agent 的想法，建议将 API 连接到 LLM 或使用 Agentic 框架，并声称获得不错的 Agent 性能并不难。
- **Perplexity AI 功能**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1897359263888236859) 为 **Perplexity macOS 应用**宣布了**新的语音模式**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1897178387145482573) 指出 **Ask Perplexity 在不到一周的时间内获得了 1200 万次曝光**。
- **Google Shopping AI 功能**: [@Google](https://twitter.com/Google/status/1897310089444233382) 在 **Google Shopping** 上推出了针对时尚和美容的**新 AI 功能**，包括 AI 生成的基于图像的推荐、虚拟试穿和 AR 化妆灵感。
- **Android AI 驱动功能**: [@Google](https://twitter.com/Google/status/1897039693700624573) 强调了 **Android** 中新的 **AI 驱动功能**，以及安全工具和连接性改进。
- **Gemini 2.0 的 Function Calling 指南**: [@_philschmid](https://twitter.com/_philschmid/status/1897287725973111041) 宣布了 **Google Gemini 2.0 Flash 的端到端 Function Calling 指南**，涵盖了设置、JSON schema、Python SDK、LangChain 集成以及 OpenAI 兼容 API。

**AI Infrastructure & Compute**

- **配备 512GB RAM 的 Mac Studio**：[@awnihannun](https://twitter.com/awnihannun/status/1897292379293671437) 强调了**配备 512GB RAM 的新款 Mac Studio**，并指出它可以运行 **4-bit Deep Seek R1** 且仍有余量。[@cognitivecompai](https://twitter.com/cognitivecompai/status/1897320672571068791) 对 **512GB RAM** 选项的反应是“别废话，拿走我的钱！”。
- **Mac 上的 MLX 和 LM Studio**：[@reach_vb](https://twitter.com/reach_vb/status/1897325952805560780) 指出 **MLX 和 LM Studio 在 M3 Ultra 发布会中被重点提及**，感觉非常梦幻。[@awnihannun](https://twitter.com/awnihannun/status/1897328673361133798) 也指出新款 **Mac Studio 产品页面**展示了 **MLX + LM Studio**。[@reach_vb](https://twitter.com/reach_vb/status/1897305816124023160) 分享了在 MPS 上使用 **llama.cpp 和 MLX** 的积极体验，并将其与 **torch** 进行了对比。
- **计算效率与扩展**：[@omarsar0](https://twitter.com/omarsar0/status/1897334393280323710) 讨论了提高**推理模型效率**的方法，提到了巧妙的推理方法和 **UPFT（通过减少 Token 实现的高效训练）**。[@omarsar0](https://twitter.com/omarsar0/status/1897334301462815001) 分享了一篇关于使用 **"A Few Tokens Are All You Need"** 方法在保持推理性能的同时将 **LLM 微调成本降低 75%** 的论文。[@jxmnop](https://twitter.com/jxmnop/status/1897059292102189278) 强调了**数据集蒸馏（dataset distillation）的效率**，通过仅在 10 张图像上进行训练，在 **MNIST** 上实现了 **94% 的准确率**。
- **OpenCL 在 AI 计算中错失的机会**：[@clattner_llvm](https://twitter.com/clattner_llvm/status/1897374468055687406) 反思了 **OpenCL** 作为“本该”赢得 AI 计算的技术，并分享了从其失败中吸取的教训。

**AI 安全与政策**

- **超级智能战略与 AI 安全**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308828284412226) 与 [@ericschmidt](https://twitter.com/ericschmidt) 及 [@alexandr_wang](https://twitter.com/alexandr_wang) 提出了一项新的超级智能战略，认为其具有不稳定性，并呼吁采取**威慑 (MAIM)、竞争力和不扩散**的战略。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308833569235418) 引入了**相互保证 AI 故障 (MAIM)** 作为针对不稳定 AI 项目的威慑机制，并将其与核 MAD 类比。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308830943601113) 警告不要针对超级智能实施**美国 AI 曼哈顿计划**，因为这可能导致局势升级并引发中国等国家的威慑。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308839734874134) 强调了灾难性 AI 能力向恶意行为者的**不扩散**，建议追踪 AI 芯片并防止走私。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308837650292786) 强调了 **AI 芯片供应链安全**和本土制造对**竞争力**至关重要，考虑到中国入侵台湾的风险。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1897308835762856275) 将解决 AI 问题的方法与**冷战政策**进行了类比。[@saranormous](https://twitter.com/saranormous/status/1897311687772135548) 推广了一集关于该国家安全战略的 **NoPriorsPod** 播客，嘉宾包括 [@DanHendrycks](https://twitter.com/DanHendrycks)、[@alexandr_wang](https://twitter.com/alexandr_wang) 和 [@ericschmidt](https://twitter.com/ericschmidt)。[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1897281378699673682) 支持 **Sutton 和 Barto 获得图灵奖**，并强调在没有安全措施的情况下发布模型是不负责任的。[@denny_zhou](https://twitter.com/denny_zhou/status/1897298962132165036) 引用了一条建议，即在 AI 研究中应优先考虑雄心，而非隐私、可解释性或安全性。
- **地缘政治与 AI 竞争**：[@NandoDF](https://twitter.com/NandoDF/status/1897357271052521962) 提出了一个问题：中国还是美国被视为可能在 AI 发展中不受约束的专制政府。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1897382703395459377) 注意到紧张局势升级，且中国的言论正从缓和方式转变。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1897315616035602658) 指出**中国顶尖 AI 团队中存在跨性别者**，这是中国人力资本竞争力的一个标志。[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1897067725291487564) 讨论了 AI 进展对**去工业化**和工作性质的影响，建议相比抽象角色，更倾向于植根于本地环境的“真实”制造业工作。[@hardmaru](https://twitter.com/hardmaru/status/1897089190514581549) 认为**地缘政治和去全球化**将塑造未来十年的世界。
- **AI 控制与安全研究**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1897029916648222801) 对 **AI 控制**成为一个真正的研究领域并举办首届会议表示兴奋。
- **注意力经济中的虚假信息与真相**：[@ReamBraden](https://twitter.com/ReamBraden/status/1897043054495981966) 观察到 **X 上惊人数量的虚假信息**，认为注意力经济的激励机制与“言论自由”不相容，在线真相需要新的激励机制。

**梗与幽默**

- **GPT-4.5 的“庞大体积”与 GPU 熔化**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897348131777626239) 警告说 **“psa: gpt-4.5 即将登陆 plus，我们的 gpu 可能会熔化，请多包涵！”**。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897362132595040542) 发布了“我们的超级计算机正在处理庞然大物的现场画面”。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897367455502409870) 开玩笑说“好吧，你的模型太胖了，它是自己滚出来的”。[@stevenheidel](https://twitter.com/stevenheidel/status/1897347895780950234) 发文称“在我们推出 gpt-4.5 之际，为我们的 GPU 祈祷”。
- **GPT-4.5 绿字（greentext）梗**：[@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1897354268317008077) 提到 **GPT-4.5** 已向 Pro 用户开放，用于“补全绿字”梗。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897256279749616096) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1897252786481537276) 承认被 **GPT-4.5 生成的关于他们自己的搞笑绿字**搞得“心态崩了”且感到“尴尬”。
- **ChatGPT UltraChonk 7 High 的高昂成本**：[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1897336539757339062) 调侃了未来 **ChatGPT UltraChonk 7 High** 的成本，将 **1.5 周的使用权比作 80 万美元的遗产**或 **2028 年的两打鸡蛋**。
- **电影观点与 Aidan Moviebench**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897302169730343288) 宣称 **“《盗梦空间》实际上是人类历史上最伟大的电影”**，并称其为“Aidan Moviebench 中的 o1”。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1897300620132135106) 表示 **“克里斯托弗·诺兰唯一好的电影是《盗梦空间》和《黑暗骑士》”**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1：苹果推出搭载 M3 Ultra 的 Mac Studio，支持 AI 推理及 512GB 统一内存**

- **[苹果发布新款 Mac Studio，搭载 M4 Max 和 M3 Ultra，最高支持 512GB 统一内存](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/)** ([Score: 422, Comments: 290](https://reddit.com/r/LocalLLaMA/comments/1j43us5/apple_releases_new_mac_studio_with_m4_max_and_m3/))：**Apple** 发布了新款 **Mac Studio**，配备 **M4 Max** 和 **M3 Ultra** 芯片，提供高达 **512GB 的统一内存**。
  - 围绕 **内存带宽和成本** 的讨论强调了使用 DDR5 和 AMD Turin 实现高带宽的挑战，每个 **CCD 为 106GB/s**，需要 **5 个 CCD** 才能超过 **500GB/s**。对比中提到了售价 **2998 美元** 的 **EPYC 9355P** 以及服务器 RAM 的高昂成本，对 Apple 产品的性价比提出了质疑。
  - 用户对新款 Mac Studio 的 **实际应用和性能** 表现出浓厚兴趣，特别是针对 **AI 推理任务**，如运行 **Unsloth DeepSeek R1** 和 **LLM** 的 Token 生成。尽管价格高昂，**512GB 型号** 被视为本地托管 R1 的可行选择，并被拿来与 **8 块 RTX 3090** 的配置进行对比。
  - Mac Studio 的 **定价和配置** 受到严格审视，**512GB 版本** 在意大利售价为 **1.1 万欧元**，在美国为 **9500 美元**。**教育优惠** 可将其降至 **约 8600 美元**，而 **M4 Max** 因其 **546GB/s 的内存带宽** 被认为足以与 **Nvidia Digits** 竞争。

- **[新的王者？M3 Ultra，80 核 GPU，512GB 内存](https://i.redd.it/jkhal4p0qvme1.jpeg)** ([评分: 207, 评论: 141](https://reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)): 该帖子讨论了配备 **32 核 CPU**、**80 核 GPU** 和 **512GB 统一内存** 的 **Apple M3 Ultra**，这为计算能力开辟了巨大的可能性。**起售价** 为 **$9,499**，提供定制和预订选项，突显了该型号对高性能计算的潜在影响。
  - **Thunderbolt 网络与 Asahi Linux**: 用户讨论了 macOS 对 Thunderbolt 网络的自动设置，指出其在 TB3/4 下之前受限于 **10Gbps**，而 **Asahi Linux** 目前支持部分 Apple Silicon 芯片，但不包括 M3。一些用户在 M2 芯片上尝试了 Asahi，但发现尚不完善，尽管更倾向于使用 macOS，但仍对团队的努力表示赞赏。
  - **与 NVIDIA 的对比及成本效益**: M3 Ultra 缺乏 **CUDA** 被视为训练和图像生成的劣势，一些用户注意到 Mac 在处理较长 Prompt 时性能较慢。M3 Ultra 的成本与 **NVIDIA GPU** 进行了对比，讨论强调了其能效比（480W 对比等效 GPU 的 5kW）以及将 **GPU 与 CPU 推理** 进行比较的挑战。
  - **定价与价值认知**: M3 Ultra 的价位引发了辩论，一些用户因其 **512GB 统一内存** 和高效率而认为其物有所值，而另一些人则认为与 **NVIDIA GPU** 相比定价过高。该设备与 **80GB H100** 和 **Blackwell Quadro** 进行了对比，强调了其在内存容量和带宽方面的价值，尽管初始成本较高。


- **Mac Studio 刚刚获得了 512GB 内存！** ([评分: 106, 评论: 76](https://reddit.com/r/LocalLLaMA/comments/1j44vep/mac_studio_just_got_512gb_of_memory/)): **Mac Studio** 现在配备 **512GB 内存** 和 **4TB 存储**，内存带宽为 **819 GB/s**，美国售价为 **$10,499**。此配置可能能够以 **8 tps** 的速度运行 **Llama 3.1 405B**。
  - 讨论强调了 **Mac Studio** 与其他高性能设置相比的**成本效益**，例如一套耗资 **$44,000** 的 **Nvidia GH200 624GB** 系统。用户对 **$10,499** 价格标签的实用性进行了辩论，一些人指出它为其他昂贵的硬件配置提供了一个具有竞争力的替代方案。
  - 用户讨论了 Mac Studio 的**技术能力**，特别是利用 [VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) 等工具运行 **Deepseek-r1 672B** 且上下文超过 **70,000+** 的能力。关于其在小上下文规模下运行大型模型的适用性，以及通过集群化多个单元以获得更高性能的潜力存在争议。
  - 对话涉及 **Mac 系统** 在某些任务（如模型训练）中的**局限性**，以及在自定义构建系统中实现类似**内存带宽**的挑战。一些用户指出需要 **Threadripper** 或 **EPYC** 系统等高级配置才能匹配 Mac Studio 的性能，而另一些人则建议通过网络连接多台 Mac 来增加 RAM。


**主题 2. Qwen/QwQ-32B 发布：性能对比与基准测试**

- **[Qwen/QwQ-32B · Hugging Face](https://huggingface.co/Qwen/QwQ-32B)** ([评分: 169, 评论: 55](https://reddit.com/r/LocalLLaMA/comments/1j4az6k/qwenqwq32b_hugging_face/)): **Qwen/QwQ-32B** 是一款可在 **Hugging Face** 上获取的模型，但该帖子未提供有关它的具体细节或背景。
  - **Qwen/QwQ-32B** 引起了极大的关注，用户表示它可能优于 **R1**，并可能是迄今为止最好的 **32B 模型**。一些用户推测它可以与更大的模型竞争，提到它优于 **671B 模型**，并建议将其与 **QwQ 32B coder** 结合使用将会非常强大。
  - 用户讨论了性能和实现细节，一些人比起官方版本更倾向于 **Bartowski 的 GGUF**，而另一些人则对该模型在角色扮演和虚构创作等特定用例中的能力印象深刻。该模型在 **Hugging Face** 上的可用性及其在 **3090 GPU** 等现有硬件上高效运行的潜力受到了关注。
  - 存在关于对科技行业更广泛影响的推测，一些人认为如果该模型获得认可，可能会影响 **Nvidia** 等公司。然而，其他人认为对私有化部署的需求可能会通过扩大客户群而使 Nvidia 受益。

- **[准备好了吗！](https://i.redd.it/m0ktikjrjume1.png)** ([Score: 567, Comments: 77](https://reddit.com/r/LocalLLaMA/comments/1j3zxwn/are_we_ready/)): **Junyang Lin** 在 2025 年 3 月 5 日通过推文宣布完成了 **QwQ-32B** 的最终训练，该推文获得了 151 个点赞及其他互动。推文中包含一个鱼的表情符号，并由认证账号发布，标志着 AI 训练里程碑中的这一重大进展。
  - **QwQ-32B 的发布与性能**：人们对 **QwQ-32B** 的发布充满期待，评论强调其预期性能将优于 **QwQ-Preview** 以及之前的模型（如 **Qwen-32B**）。该模型被预期会有显著提升，可能超越 **Mistral Large** 和 **Qwen-72B**，部分用户能够在消费级 GPU 上运行它。
  - **在线演示与对比**：**Hugging Face** 上已提供在线演示，链接见 [此处](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo)。讨论中将 **QwQ-Preview** 与 **R1-distill-qwen-32B** 进行了对比，结果较为理想，暗示新模型在性能上可能超越 **DeepSeek R1**，并具有更强的推理和工具使用能力。
  - **社区反应与期望**：用户对新模型表达了兴奋和期待，有人幽默地考虑创建名为 "UwU" 的 AI（基于 **QwQ** 的模型其实已经存在）。关于 **QwQ-32B** 是否能比 **r1 distilled qwen 32B** 表现更好的讨论，显示了社区的高度关注和竞争性基准测试。


**主题 3. llama.cpp 在利用本地 LLM 方面的多功能性**

- **llama.cpp 就是你所需的一切** ([Score: 356, Comments: 122](https://reddit.com/r/LocalLLaMA/comments/1j417qh/llamacpp_is_all_you_need/)): 作者探索了从 **ollama** 开始的本地托管 **LLM**，**ollama** 使用了 **llama.cpp**，但在 Linux 上为不支持的 **AMD card** 编译 **ROCm backend** 时遇到了问题。在尝试 **koboldcpp** 未果后，他们通过 **llama.cpp** 的 **vulkan** 版本获得了成功。他们赞扬了 **llama-server** 简洁的 Web-UI、API endpoint 和广泛的可调性，得出结论认为 **llama.cpp** 能够全面满足他们的需求。
  - **Llama.cpp** 因其功能和易用性受到称赞，但也提到了对**性能**和**多模态支持**的担忧。用户提到 **llama.cpp** 已经放弃了多模态支持，而像 **mistral.rs** 这样的替代方案支持最新的模型，并提供 **in-situ quantization** 和 **paged attention** 等功能。一些用户更倾向于使用 **koboldcpp**，因为它在不同硬件上具有通用性。
  - **llama-server** 的 Web 界面因其简单和整洁的设计获得了正面反馈，与 **openweb-ui** 等被认为更复杂的 UI 形成对比。**Llama-swap** 被强调为管理多个模型和配置的有价值工具，能够实现高效的模型热切换。
  - 讨论了 **llama.cpp** 的**性能问题**，特别是在**并发用户**和 **VRAM** 管理场景下。一些用户报告称使用 **exllamav2** 和 **TabbyAPI** 效果更好，这些工具提供了增强的 **context length** 和 **KV cache compression** 能力。


- **Ollama v0.5.13 已发布** ([Score: 139, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1j3vbfh/ollama_v0513_has_been_released/)): **Ollama v0.5.13** 已经发布。帖子中未提供关于该版本的更多细节或背景。
  - **Ollama v0.5.13** 的发布讨论围绕模型兼容性和集成展开，用户在尝试使用新版本时遇到了挑战，特别是 **qwen2.5vl** 及其多模态功能。一位用户注意到 Windows 上的 **llama runner** 进程存在问题，并引用了一个 [GitHub](https://github.com/ollama/ollama/issues/9515) issue。
  - 人们对 **Ollama** 接收来自 **Visual Studio Code** 和 **Cursor** 请求的能力感到好奇，这表明可能有一项新功能用于处理来自以 `vscode-file://` 开头的 origins 的请求。
  - 关于 **Phi-4** 多模态支持的讨论强调了由于为多模态模型实现 **LoRA** 的复杂性而导致的延迟，目前 **llama.cpp** 不支持 **minicpm-o2.6**，并暂停了多模态开发。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. TeaCache 增强提升了 WAN 2.1 性能**

- **[好吧，我不喜欢它假装成一个人并谈论去上学](https://i.redd.it/b457blvntvme1.jpeg)** ([Score: 120, Comments: 63](https://reddit.com/r/ChatGPT/comments/1j44f2q/ok_i_dont_like_it_when_it_pretends_to_be_a_person/))：该帖子讨论了一个涉及计算 **"130 加上 100 乘以 5"** 的**数学问题**，强调了记住运算顺序的重要性，特别是“数学课”上教的**先乘后加**。图片使用了对话式的语气并突出显示了短语，以吸引读者。
  - 讨论强调，像 **ChatGPT 这样的 AI 模型**并非专为“130 加上 100 乘以 5”这类简单计算而设计。用户反对将推理模型用于此类任务，认为其效率低下且可能出错，并建议将传统的计算器作为更可靠、更节能的替代方案。
  - 对话凸显了对**大语言模型 (LLMs)** 的普遍误解，用户指出 LLMs 的功能是作为知识检索工具，而非真正的推理实体。一些用户对普通人对 LLM 能力的误解及其在创造力和解决问题方面的局限性表示沮丧。
  - 评论中充满了幽默和讽刺，用户开玩笑说 AI 出现在数学课上及其拟人化的描绘。在将 AI 想象成同学时，语调十分俏皮，引用了学校教科书中的 **PEMDAS** 知识，并回忆起 AI 的“父母”是来自匈牙利的犹太移民。


- **[Wan 2.1 的官方 TeaCache 已发布。有人说速度提升了 100%，但我还没亲自测试。](https://i.redd.it/9scd0q97jume1.png)** ([Score: 108, Comments: 41](https://reddit.com/r/StableDiffusion/comments/1j3zwg9/official_teacache_for_wan_21_arrived_some_said_he/))：**TeaCache** 现在支持 **Wan 2.1**，一些用户声称**速度提升了 100%**。社区中的热烈响应（如来自 **FurkanGozukara** 的回复）突显了在 **GitHub** 上测试这些新功能的兴奋感和协作精神。
  - 用户讨论了 **TeaCache** 的**安装挑战**，特别是 **Python** 和 **Torch** 版本不匹配的问题。解决方案包括使用 **pip** 安装 **Torch nightly**，并确保在安装前通过 "source activate" 激活正确的环境。
  - 用户有兴趣了解 **TeaCache** 与 **Kijai 的节点**之间的区别。**Kijai** 更新了他的封装器以包含新的 TeaCache 功能，在官方发布前通过跳步估算系数以进行比较。
  - **性能改进**显著，用户如 **_raydeStar** 报告称，使用 **sage attention** 和 **sparge_attn** 后速度大幅提升，在测试期间将时间从 34.91s/it 减少到 11.89s/it。然而，一些用户遇到了伪影问题，正在寻求高质量渲染的最佳设置。


**主题 2. Lightricks LTX-Video v0.9.5 新增关键帧和扩展功能**

- **[LTX-Video v0.9.5 发布，现已支持关键帧、视频扩展和更高分辨率。](https://github.com/Lightricks/LTX-Video)** ([Score: 184, Comments: 53](https://reddit.com/r/StableDiffusion/comments/1j48shq/ltxvideo_v095_released_now_with_keyframes_video/))：**LTX-Video v0.9.5** 已发布，具有**关键帧**、**视频扩展**和支持**更高分辨率**等新功能。
  - **关键帧功能与插值**：用户对**关键帧**功能感到兴奋，认为它有可能成为开源模型的游戏规则改变者。**帧条件化 (Frame Conditioning)** 和 **序列条件化 (Sequence Conditioning)** 被强调为帧插值和视频扩展的新功能，用户渴望看到这些功能的演示 ([GitHub 仓库](https://github.com/Lightricks/ComfyUI-LTXVideo))。
  - **硬件与性能**：讨论显示 **LTX-Video** 相对较小，拥有 **2B 参数**，可在 **6GB vRAM** 上运行。用户欣赏该模型与其他模型相比的体积优势，尽管平衡资源、生成时间和质量仍是一个挑战。
  - **工作流与示例**：社区分享了部署和使用 **LTX-Video** 的资源，包括一个带有 **ComfyUI** 的 **RunPod 模板**，用于 **i2v** 和 **t2v** 等工作流。分享了示例工作流和额外资源，强调了更新以利用新功能的必要性 ([ComfyUI 示例](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/))。


**主题 3. Chroma 模型开源开发版发布**

- **Chroma: 开源、无审查、为社区而建 - [进行中]** ([Score: 381, Comments: 117](https://reddit.com/r/StableDiffusion/comments/1j4biel/chroma_opensource_uncensored_and_built_for_the/)): **Chroma** 是一个基于 **FLUX.1-schnell** 的 **8.9B 参数模型**，完全采用 **Apache 2.0 许可** 供开源使用和修改，目前正在训练中。该模型基于从 **20M 样本** 中提取的 **5M 数据集** 进行训练，专注于无审查内容，并由 [Hugging Face 仓库](https://huggingface.co/lodestones/Chroma) 和 [实时 WandB 训练日志](https://wandb.ai/lodestone-rock/optimal%20transport%20unlocked) 等资源提供支持。
  - **数据集充分性**：人们对 **5M 数据集** 是否足以支撑一个通用模型表示担忧，并将其与可达 **3M** 图像的 **booru dumps** 进行了比较。此外，还讨论了关于数据集内容的问题，包括是否包含 **名人** 以及针对 **sfw/nsfw** 内容的特定标签。
  - **技术优化与许可**：**Chroma** 模型经过了显著优化，实现了更快的训练速度（在 **8xh100 节点** 上约为 **18img/s**），建议进行 **50 epochs** 以实现强收敛。项目强调了 **Apache 2.0 许可**，但由于法律模糊性，在开源数据集方面仍面临挑战。
  - **模型对比与法律担忧**：讨论中包括了与 **SDXL** 和 **SD 3.5 Medium** 等其他模型的对比，一些用户对克服 **Flux** 模型训练挑战表示兴奋。同时，还提到了在大规模数据集上训练时关于版权侵权的法律担忧，强调了潜在的法律风险。


**主题 4. GPT-4.5 向 Plus 用户推出，具备记忆功能**

- **[4.5 向 Plus 用户推出](https://i.redd.it/feb4fonruwme1.jpeg)** ([Score: 394, Comments: 144](https://reddit.com/r/OpenAI/comments/1j49dfa/45_rolling_out_to_plus_users/)): **OpenAI** 宣布向 **Plus 用户** 推出 **GPT-4.5**，正如一条 Twitter 帖子所示。图片展示了一个揭露该更新的非正式对话，并配有表情符号反应，强调了对新版本发布的兴奋。
  - 用户对 **GPT-4.5** 的自我意识和提供准确信息的能力表示怀疑，一些人报告了模型否认自身存在的情况。**OpenAI** 尚未明确沟通使用限制，导致用户对 **每周 50 条消息** 的上限以及重置时间感到困惑。
  - 对于此次推出，用户情绪参杂着兴奋与沮丧，特别是针对 **速率限制（rate limits）** 以及改进记忆等功能缺乏透明度。一些用户报告在 **iOS** 和 **浏览器** 上均可访问，并被标记为“Research Preview”。
  - **OpenAI** 提到向 **Plus 用户** 的推广将需要 1-3 天，速率限制可能会随着需求评估而改变。用户仍在等待关于限制和功能的进一步更新，并对潜在的 **advanced voice mode** 更新表现出显著兴趣。


- **[OpenAI 员工确认，Plus 用户的 GPT 4.5 速率限制为每周 50 条消息](https://i.redd.it/cxkcinv9fyme1.jpeg)** ([Score: 148, Comments: 61](https://reddit.com/r/OpenAI/comments/1j4h8b9/confirmed_by_openai_employee_the_rate_limit_of/)): **Aidan McLaughlin** 确认 **GPT-4.5** 限制 Plus 用户 **每周 50 条消息**，并可能根据使用情况有所变动。他幽默地声称每个 GPT-4.5 token 消耗的能量相当于 **意大利** 全年的能耗，截至 **2025 年 3 月 5 日**，该推文已获得 **9,600 次浏览**。
  - 关于 **GPT-4.5 能耗** 的说法被广泛认为是一种幽默的夸张，用户指出这缺乏逻辑连贯性。**Aidan McLaughlin** 的推文被解读为一个玩笑，旨在嘲讽关于 AI 能耗的夸大言论，例如将单个 token 的能耗与整个意大利的年能耗相提并论被视为荒谬之极。
  - 讨论突出了 **GPT-4.5 的巨大规模**，推测其参数量可能超过 **10 万亿（10 trillion）**。用户对模型的尺寸和架构表示好奇，并指出 OpenAI 尚未披露关于参数数量或能耗的具体数据。
  - 评论者幽默地应对这种能耗说法的荒谬性，利用 **幽默和讽刺** 来评价这一言论。这包括关于未来使用 **Dyson spheres**（戴森球）的笑话，以及对“加拿大女性流浪汉为三明治争吵”等 **非公制单位** 的戏谑引用。

- **[GPT-4.5 正式向 Plus 用户推出！](https://i.redd.it/vib3d0tsaxme1.png)** ([得分: 165, 评论: 56](https://reddit.com/r/OpenAI/comments/1j4bn7i/gpt45_is_officially_rolling_out_to_plus_users/)): **GPT-4.5** 现在以研究预览版的形式向 **Plus 用户**开放，被描述为适用于写作和探索想法。界面还列出了 **GPT-4o** 以及带有**定时任务**功能的 **GPT-4o** Beta 版用于后续查询，所有这些都集成在一个现代的深色主题 UI 中。
  - 用户正在讨论 **GPT-4.5** 中的**记忆功能 (memory feature)**，一位评论者确认了该功能的存在，这与缺乏此功能的其他模型形成对比。这一补充受到了好评，因为它增强了模型的能力。
  - 用户对了解 **Plus 用户的限制**表现出浓厚兴趣，纷纷询问每天或每周允许的消息数量。一位用户报告进行了超过 **20 条消息**的对话，另一位用户提到 **50 条消息的上限**，该上限可能会根据需求进行调整。
  - 一些用户对 **GPT-4.5** 表示失望，认为它与竞争对手相比没有显著的差异化，而另一些人则好奇是否有特定的任务使 **GPT-4.5** 优于其他模型。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要之摘要

**主题 1：阿里巴巴的 QwQ-32B 挑战巨头**

- [**QwQ-32B 越级挑战 DeepSeek-R1**](https://qwenlm.github.io/blog/qwq-32b)：阿里巴巴的 **QwQ-32B**（一款 320 亿参数的模型）与 6710 亿参数的 **DeepSeek-R1** 旗鼓相当，展示了**Reinforcement Learning (RL)** 扩展的力量。该模型在数学和编程任务中表现出色，证明了尺寸并非决定一切。
- [**社区热切测试 QwQ-32B 的实力**](https://huggingface.co/Qwen/QwQ-32B)：用户正在通过 [Hugging Face](https://huggingface.co/Qwen/QwQ-32B) 和 [Qwen Chat](https://chat.qwen.ai) 对 **QwQ-32B** 进行全面测试。初步印象表明其性能可与更大的模型相媲美，引发了广泛关注。
- [**QwQ-32B 采用了 Hermes 的秘诀**](https://qwenlm.github.io/blog/qwq-32b)：观察者注意到 **QwQ-32B** 使用了类似于 **Hermes** 的特殊 Token 和格式，包括 `<im_start>`、`<im_end>` 和工具调用语法。这增强了与高级 Prompt 技术的兼容性。

**主题 2：用户对 AI 工具缺陷的挫败感爆发**

- [**Cursor 的 3.7 模型“降智”，用户纷纷跳槽**](https://v0-next-js-website-orcin.vercel.app/)：开发者报告称 **Cursor 的 3.7 模型**感觉被削弱了，会生成多余的 Readme 文件并错误地使用抽象。一个带有讽刺意味的 [Cursor 降智计 (Cursor Dumbness Meter)](https://v0-next-js-website-orcin.vercel.app/) 嘲讽了这种退步，促使许多人考虑将 [Windsurf](https://codeium.com/blog/windsurf-wave-4) 作为替代方案。
- [**Claude Sonnet 3.7 在简单任务上失手**](https://www.perplexity.ai/)：用户对 Perplexity 上的 **Claude Sonnet 3.7** 表示失望，理由是在解析 JSON 文件时出现幻觉，且性能不如直接通过 Anthropic 使用。用户对其“声称的改进”未能实现感到愈发沮丧。
- [**GPT-4.5 伴随限制与拒绝请求而至**](https://discord.com/channels/974519864045756446)：OpenAI 的 **GPT-4.5** 发布令用户兴奋，但限制**每周使用 50 次**。它拒绝处理基于故事的 Prompt，即使这些 Prompt 符合指南，这让用户感到非常恼火。

**主题 3：AI Agent 志存高远，价格高昂**

- [**OpenAI 计划为精英级 Agent 每月收费高达 2 万美元**](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents)：OpenAI 正准备销售高级 AI Agent，订阅费用从**每月 2,000 美元到 20,000 美元**不等，目标是编程自动化和博士级研究等任务。高昂的价格引起了用户的关注和怀疑。
- [**LlamaIndex 与 DeepLearningAI 合作开发 Agent 式工作流**](https://llamaindex.ai/blog/agentic-document-workflows)：**LlamaIndex** 与 **DeepLearningAI** 合作提供关于构建 **Agentic Document Workflows** 的课程，将 AI Agent 无缝集成到软件流程中。这一举措强调了 Agent 在 AI 开发中日益增长的重要性。
- [**Composio 通过开箱即用的身份验证简化 MCP**](https://mcp.composio.dev/)：**Composio** 现在支持具有强大身份验证功能的 [MCP](https://mcp.composio.dev/)，消除了为 Slack 和 Notion 等应用设置 MCP 服务器的麻烦。他们的[公告](https://x.com/composiohq/status/1896968949654495291)宣称提高了工具调用的准确性并易于使用。

**主题 4：Reinforcement Learning 大放异彩并取得重大进展**

- [**RL Agent 凭借微型模型征服 Pokémon Red**](https://x.com/dsrubinstein/status/1897351145485648309?s=46)：一个强化学习系统使用参数量低于 **1000 万** 的策略模型和 **PPO** 算法打通了 **Pokémon Red**，展示了 RL 在复杂任务中的实力。这一成就凸显了 RL 的复兴及其在游戏 AI 领域的潜力。
- [**AI 挑战弹幕游戏：为 Touhou 训练机器人**](https://discord.com/channels/1189498204333543425)：爱好者们正尝试训练 AI 模型来玩 **Touhou**，利用以游戏分数为奖励的 RL 技术。他们正在探索像 **Starcraft gym** 这样的模拟器，以观察 RL 是否能精通这些以高难度著称的游戏。
- [**RL Scaling 让中型模型化身巨头**](https://qwenlm.github.io/blog/qwq-32b)：**QwQ-32B** 的成功证明了扩展 RL 训练能显著提升模型性能。持续的 RL Scaling 使得中型模型能够与巨型模型竞争，尤其是在数学和编程能力方面。

**主题 5：技术人员对新硬件发布的反应**

- [**苹果发布天蓝色 M4 MacBook Air，技术圈反应不一**](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)：苹果配备 **M4 芯片**和**天蓝色**外观的新款 **MacBook Air** 起售价为 **$999**。虽然一些人对 **Apple Intelligence** 功能感到兴奋，但也有人对规格表示不满，称“*这就是我不买 Mac 的原因……*”
- [**Thunderbolt 5 承诺超高速数据传输**](https://www.intel.com/content/www/us/en/products/docs/io/thunderbolt/thunderbolt-5-brief.html)：**Thunderbolt 5** 拥有 **120Gb/s** 的单向速度，让用户对分布式训练中增强的数据传输感到兴奋。它被认为有可能超越 **RTX 3090 SLI bridge**，并为基于 Mac 的设置开启新大门。
- [**AMD RX 9070 XT 与 Nvidia 针锋相对**](https://www.youtube.com/watch?v=yP0axVHdP-U)：**AMD RX 9070 XT** GPU 的测评显示，其在光栅化性能上与 Nvidia 的 **5070 Ti** 旗鼓相当。其价格仅为 5070 Ti（建议零售价 $750）的 80%，被赞誉为高性价比的动力源。


---

# 第一部分：Discord 高层级摘要




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的 3.7 模型性能受到质疑**：成员们报告称 **Cursor 的 3.7 模型感觉被削弱了**，理由是它会在未提示的情况下生成 *readme 文件*，并且*过度使用 Class A 抽象*。
   - 一些用户怀疑 **Cursor 要么使用了冗长的 Prompt，要么使用了虚假的 3.7 模型**，一位成员分享了一个[关于 Cursor 编辑器今天感觉有多笨的科学测量方法](https://v0-next-js-website-orcin.vercel.app/)。
- **社区讽刺 Cursor 的“变笨”表现**：一位成员分享了一个[链接](https://v0-next-js-website-orcin.vercel.app/)，指向一个测量 Cursor 编辑器**“愚蠢程度”**的**“高度精密仪表”**。
   - 该仪表使用基于“宇宙射线、键盘失误以及代码补全错误次数”的“高级算法”，引发了社区的幽默回应。
- **更新后 YOLO 模式受阻**：更新后，**Cursor 中的 YOLO 模式无法正常工作**，因为即使在白名单为空的情况下，它现在运行命令前也需要获得批准。
   - 一位用户表达了沮丧，表示他们希望 **AI Agent 拥有尽可能多的自主性**，并依靠 Git 来处理错误的删除，他们更喜欢曾为他们节省数小时时间的 v45 版本行为。
- **替代方案 Windsurf 受到关注**：社区成员正在积极讨论 [Windsurf 的新版本](https://codeium.com/blog/windsurf-wave-4) Wave 4，由于感知到其在 Agent 能力方面的优势，一些人正考虑切换，并分享了一段关于 **“Vibe Coding 教程与最佳实践 (Cursor / Windsurf)”** 的 [YouTube 教程](https://www.youtube.com/watch?v=YWwS911iLhg)。
   - 尽管有兴趣，但对 Windsurf 定价模式的担忧依然存在，一些用户提到它“挪用”了 continue.dev 的成果。
- **OpenAI 筹备高端付费层级**：一位成员分享了一份报告，称 [OpenAI 正在加倍投入其应用业务](https://x.com/steph_palazzolo/status/1897309493744267314?s=46)，计划为能够自动化编程和博士级研究的高级 Agent 提供每月 **$2,000 至 $20,000** 的订阅服务。
   - 这一消息引发了质疑，一些人怀疑如此高昂的价格是否合理，尤其是考虑到目前 AI 模型的输出质量。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 发布 GPT-4.5**：OpenAI *提前于计划*发布了 **GPT-4.5**，但目前限制为**每周 50 次使用**，且它并非 **GPT-4o** 的替代品。
   - 报告显示 **GPT-4.5** 会拒绝基于故事的提示词，并将逐步增加使用额度。
- **OpenAI 迭代 AGI**：OpenAI 将 **AGI 开发**视为一个*持续的路径*而非突然的飞跃，专注于通过对现有模型的迭代部署和学习，使未来的 AI 更加安全且有益。
   - 他们的 **AI safety 和 alignment** 方法以拥抱不确定性、深度防御、可扩展的方法、人类控制以及社区努力为指导，以确保 **AGI 造福全人类**。
- **关于 OpenAI O3 的猜测**：成员们对 **O3** 的发布进行了猜测，并指出 *OpenAI 表示不会在 ChatGPT 中发布完整版 O3，仅在 API 中提供*。
   - 语气表明它仍然是一个 **AI**，因此不会总是 100% 准确，并且*用户应当始终咨询人类治疗师或医生*。
- **Qwen-14B 助力递归摘要**：成员们正在使用 **Qwen-14B** 模型进行递归摘要任务，并发现其输出效果优于 **Gemini**。
   - 示例中提到对一本国际象棋书籍进行摘要，**Qwen-14B** 的结果优于 **Gemini**，后者的表现类似于 **GPT-3.5**。
- **提示工程综述**：一位成员分享了一篇关于 Large Language Models 提示工程的系统综述，题为《A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications》，概述了 **Zero-Shot Prompting**、**Few-Shot Prompting** 和 **Chain-of-Thought (CoT)** 提示等关键策略，并提供了 [ChatGPT 链接](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854) 以供访问。
   - 然而，讨论强调了该综述虽然对每种技术都有详细描述，但遗漏了 **Self-Discover** 和 **MedPrompt** 等内容。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 4：收获与波动**：**Windsurf** 推出了 **Wave 4**（[博客文章](https://www.codeium.com/blog/windsurf-wave-4)），包括 **Previews**、**Tab-to-import**、**Linter integration**、**Suggested actions**、**MCP discoverability** 以及对 **Claude 3.7** 的改进。
   - 一些用户报告了死循环和高额度消耗等问题，而另一些用户则称赞了其速度以及通过 **Windsurf Command** (`CTRL/Cmd + I`) 即可访问的 **Claude 3.7** 集成。
- **凭据故障导致 Codeium 瘫痪**：多名用户报告称**无法登录** [codeium.com](https://codeium.com)（无论是使用 Google 还是邮箱/密码）。
   - 团队承认了登录问题，并提供了一个 [状态页面](https://status.codeium.com) 以获取更新。
- **额度危机困扰 Codeium 用户**：用户对**额度消耗**过快表示担忧，尤其是在使用 **Claude 3.7** 时，即使在改进后也是如此。
   - 团队澄清说 **Flex Credits** 可以结转，自动修复 Lint 错误是免费的，但其他用户仍在为额度和 Tool calls 感到困扰。
- **Windsurf Wave 4 的工作流奇迹**：一段 [YouTube 视频](https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be) 介绍了 **Windsurf Wave 4** 的更新，演示了 **Preview**、**Tab to Import** 和 **Suggested Actions**。
   - 新的 **Tab-to-import** 功能可以通过按下 Tab 键自动添加导入，增强了 Cascade 内部的工作流。
- **Windsurf 愿望清单：期待 Webview 和取消限制**：用户请求了诸如外部库文档支持、提高额度限制、可调节的聊天字体大小、像 Trae 一样在侧边栏提供真正的 Webview，以及用于生成 llms.txt 文件的 Firecrawl 等功能。
   - 一位用户建议使用 [Firecrawl](https://x.com/ericciarla/status/1897332080708858147) 为网站生成 llms.txt 文件，以便输入到 LLM 中。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 发布 QwQ-32B 推理模型**：**Qwen** 推出了 **QwQ-32B**，这是一个拥有 32B 参数的推理模型，在 [VXReddit 帖子](https://www.vxreddit.com/r/LocalLLaMA/comments/1j4b1t9/qwq32b_released_equivalent_or_surpassing_full/) 和其 [官方博客](https://qwenlm.github.io/blog/qwq-32b) 中被认为可以与 **DeepSeek-R1** 相媲美。
   - 爱好者们渴望通过 **Aider** 集成来测试其作为架构师和代码编写者的性能，并提到该模型已在 [HF](https://huggingface.co/Qwen/QwQ-32B) 和 [ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B) 上线。
- **实现 Aider 离线安装**：寻求在**离线 PC** 上安装 **Aider** 的用户通过使用 **pip download** 将 Python 包从联网机器传输到离线虚拟环境，克服了安装挑战。
   - 一个成功的操作序列包括：`python -m pip download --dest=aider_installer aider-chat`。
- **通过 OAI 兼容的 Aider 实现与 OWUI 的和谐共存**：为了在 **OpenWebUI (OWUI)** 中使用 **Aider**，一位成员建议在模型名称前加上 `openai/` 前缀，以指示这是一个 **OAI 兼容的端点**，例如 `openai/myowui-openrouter.openai/gpt-4o-mini`。
   - 这种方法绕过了将 **Aider** 连接到 **OWUI** 时出现的 `litellm.BadRequestError` 问题。
- **ParaSail 声称拥有极速 R1 吞吐量**：一位用户报告称，通过 **OpenRouter** 使用 **Parasail** 提供商，在 **R1** 上达到了 **300tps**。
   - 虽然复现较为困难，但 **Parasail** 与 **SambaNova** 一起被认为是 **R1** 的顶级性能提供商。
- **使用 Aider 编写 commit messages**：成员们讨论了让 **aider** 为暂存文件编写 commit messages 的方法，建议先执行 `git stash save --keep-index`，然后运行 `/commit`，最后执行 `git stash pop`。
   - 另一位成员建议使用 `aider --commit`，它会编写 commit message、提交并退出，并参考 [Git 集成文档](https://aider.chat/docs/git.html#commit-messages)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **注意 VRAM 溢出**：一位成员描述了如何通过监控 **Dedicated memory**（专用内存）和 **Shared memory**（共享内存）的使用情况来检测 LM Studio 中的 **VRAM 溢出**，并提供了一张[说明该问题的图片](https://cdn.discordapp.com/attachments/1110598183144399058/1346803804322009088/VRAM_Overflow.jpg?ex=67ca2d09&is=67c8db89&hm=24b703c40c580b2636786230775506086194cec8387515d56546d86fefc79989&)。
   - 他们指出，当 *Dedicated memory* 处于高位且 *Shared memory* 增加时，就会发生溢出。
- **多模态 Phi-4 暂不支持音频**：成员们确认，由于 *llama.cpp* 的限制，LM Studio 目前不支持 **multi-modal Phi-4** 和**音频支持**。
   - 目前没有针对缺失支持的变通方案。
- **锁定 VRAM、Context 和 KV Cache**：一位成员指出，**context size** 和 **KV cache** 设置会显著影响 VRAM 使用，建议以 **90% VRAM** 利用率为目标以优化性能。
   - 另一位成员解释说，KV cache 是计算机进行 attention 机制计算时的 *K 和 V 的值*。
- **Sesame AI 的 TTS：开源还是虚晃一枪？**：成员们讨论了 **Sesame AI 的对话式语音生成模型 (CSM)**，一位成员称赞其栩栩如生，并链接到了 [demo](https://www.sesame.com)。
   - 其他人对其“开源”声明表示怀疑，指出其 [GitHub 仓库](https://github.com/SesameAILabs) 缺乏代码提交。
- **M3 Ultra 和 M4 Max Mac Studio 发布**：Apple 发布了搭载 **M3 Ultra**（最高支持 512GB RAM）和 **M4 Max**（最高支持 128GB）的新款 [Mac Studio](https://www.apple.com/uk/mac-studio/)。
   - 一位成员对 RAM 规格反应消极，表示“这就是我不买 Mac 的原因……”。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sutton 引发安全辩论！**：图灵奖得主 Richard Sutton 最近在[访谈](https://youtu.be/9_PepvnqIfU?si=sMB90T8__WA19Qrt)中表示 *safety is fake news*（安全是假新闻），这引发了激烈讨论。
   - 反应各异，一位成员评论道：*Rich 在道德上有点可疑，即使他的产出惊人，我也不会听从他的研究建议*。
- **OpenAI Agent 定价：每月 2 万美元？**：根据 [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents) 报道，OpenAI 计划对专为自动化编程和 PhD 级别研究等任务设计的 AI Agent 收取 **每月 2,000 到 20,000 美元** 的费用。
   - OpenAI 的投资者 SoftBank 已承诺仅今年就在 OpenAI 的 Agent 产品上投入 **30 亿美元**。
- **阿里巴巴 QwQ-32B 模型对标 DeepSeek**：[阿里巴巴 Qwen 发布了 QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一个拥有 **320 亿参数** 的推理模型，正与 DeepSeek-R1 等模型展开竞争。
   - 该模型使用了 **RL** 训练和 post training，显著提升了在数学和编程方面的性能。
- **DeepMind 人才流向 Anthropic 的趋势持续**：Nicholas Carlini 宣布离开 Google DeepMind 加入 Anthropic，他在[博客](https://nicholas.carlini.com/writing/2025/career-update.html)中提到，他在 **adversarial machine learning** 方面的研究在 DeepMind 不再获得支持。
   - 成员们注意到 *GDM 最近流失了这么多重要人才*，而其他人则表示 *Anthropic 的天命股在上涨*。
- **RL 击败《宝可梦 红》**：一个强化学习系统使用低于 **10M 参数** 的策略、**PPO** 以及新技术击败了 **Pokémon Red**，详情见 [博客文章](https://x.com/dsrubinstein/status/1897351145485648309?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)。
   - 该系统成功通关游戏，展示了 **RL** 在解决复杂任务方面的复兴。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **东方 Project AI 模型进阶**：一位成员正在训练一个 AI 模型来玩 **Touhou**（东方 Project），使用 **RL** 并将游戏分数作为奖励，考虑了 **Starcraft gym** 和 **Minetest gym** 等模拟器。
   - 目标是确定 RL 和奖励函数是否可以用于学习游戏玩法。
- **Thunderbolt 5 加速数据传输**：成员们对 **Thunderbolt 5** 感到兴奋，它可能使 **Mac Mini/Studio** 之间的分布式推理/训练变得更加可行。
   - 其单向速度 (**120gb/s**) 似乎比 **RTX 3090 SLI bridge** (**112.5gb/s**) 还要快。
- **CUDA 编译器变得过于“聪明”**：当写入的数据从未被读取时，**CUDA 编译器**会优化掉内存写入操作，导致在添加读取操作之前不会报告任何错误。
   - 这种优化可能会误导调试内存写入操作的开发者，因为在涉及读取操作之前，没有错误并不代表行为正确。
- **TileLang 在 CUDA 12.4/12.6 上遇到困难**：用户报告在 **CUDA 12.4/12.6** 上使用 **TileLang** 进行 **matmul** 时出现元素不匹配，并在 [GitHub](https://github.com/tile-ai/tilelang/issues/149) 上提交了 bug 报告。
   - 该代码在 **CUDA 12.1** 上运行正常，但在新版本中会出现关于 tensor 差异的 `AssertionError`。
- **QwQ-32B 让大型模型感受到压力**：**阿里巴巴**发布了 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一个仅有 **320 亿参数** 的新推理模型，足以媲美 **DeepSeek-R1** 等模型。
   - 该模型已在 [HF](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、[Demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 和 [Qwen Chat](https://chat.qwen.ai) 上线。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Google 搜索进入 AI 聊天领域**：Google 宣布推出 **AI Mode for Search**，提供对话式体验并支持复杂查询，目前作为选择性加入（opt-in）体验提供给部分 **Google One AI Premium** 订阅者（参见 [AndroidAuthority](https://www.androidauthority.com/google-search-ai-mode-experiment-3532243/)）。
   - 一些用户认为，由于这一公告，**Perplexity** 不再显得特别。
- **Claude Sonnet 3.7 未达预期？**：一位用户对 **Perplexity** 实现的 **Claude Sonnet 3.7** 表示不满，认为其结果不如直接通过 **Anthropic** 使用。
   - 他们补充说，**3.7** 在一个简单的 JSON 文件中产生了幻觉错误，质疑该模型声称的改进。
- **Perplexity API 的 Focus 设置难以捉摸**：一位用户询问如何将 **API** 聚焦于特定主题（如学术或社区相关内容）的方法。
   - 然而，消息中并未提供任何解决方案。
- **Sonar Pro 搜索模型未能通过时效性测试**：一位用户报告称，尽管将 *search_recency_filter* 设置为 'month'，**Sonar Pro** 模型仍返回过时信息和错误链接。
   - 该用户怀疑自己是否误用了 **API**。
- **API 搜索成本仍是个谜**：一位用户对 **API** 不提供搜索成本信息表示沮丧，导致无法准确跟踪支出。
   - 他们哀叹无法跟踪自己的 **API** 支出，因为 **API** 没有告知使用了多少次搜索，并附上了一个大哭的表情。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CoreWeave 提交 IPO 申请，营收增长 700%**：云服务提供商 **CoreWeave**（其近三分之二的收入依赖 **Microsoft**）[提交了 IPO 招股说明书](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm)，报告 2024 年营收增长 **700%** 至 **19.2 亿美元**，尽管净亏损达 **8.634 亿美元**。
   - 约 **77%** 的收入来自两家客户（主要是 **Microsoft**），且该公司持有超过 **150 亿美元** 的未履行合同。
- **Kornia Rust 库在 Google Summer of Code 2025 开放实习**：**Kornia Rust** 库正在为 [Google Summer of Code 2025](https://summerofcode.withgoogle.com/programs/2025/organizations/kornia) 开放实习岗位以改进该库，主要围绕 **Rust** 中的 **CV/AI** 展开。
   - 鼓励感兴趣的人士查阅文档并提出任何问题。
- **Umar Jamil 在 GPU Mode 分享学习 Flash Attention、Triton 和 CUDA 的历程**：[Umar Jamil](https://x.com/hkproj/status/1896113497031000563?s=46) 将于太平洋时间 3 月 8 日（本周六）中午参加 **GPU Mode**，分享他学习 **Flash Attention**、**Triton** 和 **CUDA** 的历程。
   - 这将是与观众的一次“亲密对话”，关于他在学习过程中的困难，并分享如何自学任何知识的实用技巧。
- **VisionKit 竟然不是开源的**：`i-made-this` 频道中的模型使用了 **VisionKit** 但并未开源，可能在未来发布，但在开发过程中 **Deepseek-r1** 的表现出奇地有帮助。
   - 一篇 [Medium 文章](https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804) 讨论了构建自定义 **MCP server**，并提到了 **CookGPT** 作为示例。
- **Agents 课程证书位置不明！**：**agents-course** 频道的用户无法在课程中找到他们的证书，特别是在 [此页面](https://huggingface.co/learn/agents-course/unit1/final-quiz#certificate) 中，并寻求帮助。
   - 一位成员指出，证书可以在[此数据集](https://huggingface.co/datasets/agents-course/certificates)的 "files" 下的 "certificates" 中找到，但其他人反映仍有无法显示的问题。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Composio 通过身份验证增强 MCP**：[Composio](https://mcp.composio.dev/) 现在支持带身份验证的 **MCP**，消除了为 *Linear, Slack, Notion* 和 *Calendly* 等应用设置 **MCP servers** 的需求。
   - 他们的 [公告](https://x.com/composiohq/status/1896968949654495291) 强调了托管身份验证和改进的 tool calling 准确性。
- **WebMCP 引发安全争议**：任何网站都能充当 **MCP server** 的概念引发了安全担忧，特别是关于可能访问本地 **MCP servers** 的问题。
   - 有人将其描述为 *安全噩梦*，会破坏浏览器沙箱，而其他人则建议使用 **CORS** 和 **cross-site configuration** 等缓解措施。
- **Reddit Agent 使用 MCP 获取线索**：一位成员使用 **MCP** 构建了一个 **Reddit agent** 来生成潜在客户，展示了 **MCP** 在实际应用中的实用性。
   - 另一位成员在询问如何连接到 Reddit 后，分享了 [Composio 的 Reddit 集成](https://mcp.composio.dev/reddit/wrong-petite-crayon-_q1Vlt)。
- **Token 两步走：本地 vs. 按站点**：在设置 **MCP Server** 后，一位用户澄清了在网站访问时，除了 **local token** 之外，还存在 **按站点和按会话生成的 tokens**。
   - 开发者验证了这一过程，强调 tokens 是 *按会话、按站点* 生成的。
- **Insta-Lead-Magic 发布**：一位用户展示了一个 **Instagram Lead Scraper**，并辅以 **自定义仪表盘**，详情见链接 [视频](https://cdn.discordapp.com/attachments/1315696461316358175/1346986901877555250/full_automation_demo.mov?ex=67ca2ecf&is=67c8dd4f&hm=e3114edc2b6e1e5171c2c1be5cbb011437c737ba2268afe4e381cbfa44cf2cf0&)
   - 未提供第二个摘要。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 向开发者收取 25 美分**：一位成员报告称，向 **Claude** 询问一个关于其小型代码库的问题花费了 **$0.26**，这引发了对使用 **Claude** 处理代码相关查询成本的担忧。
   - 有人建议将代码库复制到 **Claude** 目录中，并在 **Claude Desktop** 上激活 filesystem MCP server，作为免费访问的变通方案。
- **M4 MacBook Air：天蓝色且有 AI 增强**：Apple 发布了搭载 **M4 芯片**、具备 **Apple Intelligence** 功能以及全新 **天蓝色** 的新款 **MacBook Air**，起售价为 **$999**，详见 [此公告](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)。
   - 新款 **MacBook Air** 拥有长达 18 小时的电池续航时间和 12MP Center Stage 摄像头。
- **Qwen 的 QwQ-32B：DeepSeek 的推理竞争对手**：根据 [这篇博客文章](https://qwenlm.github.io/blog/qwq-32b)，**Qwen** 发布了 **QwQ-32B**，这是一款全新的 **320 亿参数推理模型**，其性能可与 **DeepSeek-R1** 等模型相媲美。
   - 该模型通过 **RL** 和持续缩放训练，在数学和编程方面表现出色，可在 [HuggingFace](https://huggingface.co/Qwen/QwQ-32B) 上获取。
- **React：LLM 后端的惊喜英雄？**：一位成员建议 **React** 是后端 LLM 工作流的最佳编程模型，并引用了一篇关于使用 node.js 后端和 **类 React** 组件模型构建 [@gensx_inc](https://x.com/_Evan_Boyle/status/1897347251120562205) 的博客文章。
   - 反对意见包括 **Lisp** 更适合创建 DSL，以及提到了 [Mastra](https://mastra.ai/docs/workflows/00-overview) 作为一个无框架的替代方案。
- **Windsurf 的 Cascade：不再需要“检查元素”？**：**Windsurf** 发布了 **Wave 4**，其特色功能 **Cascade** 可将元素/错误直接发送到聊天中，旨在减少对“检查元素”的需求，演示见 [此链接](https://x.com/windsurf_ai/status/1897378545799979238)。
   - 更新内容包括预览、Cascade Auto-Linter、MCP UI 改进、Tab 键导入、建议操作、Claude 3.7 改进、推荐奖励以及 Windows ARM 支持。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL 手部问题让你头疼？**：用户正在寻求在 **SDXL** 中自动修复手部的方法，而无需手动进行 **inpainting**，特别是在使用 **8GB VRAM** 的情况下。讨论内容涉及 **embeddings**、**face detailers** 和 **OpenPose** 控制网（control nets）。
   - 重点在于寻找适用于 **SDXL** 的有效 **hand LoRAs** 以及自动校正技术。
- **一张照片变电影？**：用户探索了从单张照片创建视频的方法，推荐使用 **WAN 2.1 i2v model**，但指出该模型需要强大的 GPU 性能和耐心。
   - 虽然有人建议使用带有免费额度的在线服务，但共识是本地视频生成会产生成本，主要体现在电力消耗上。
- **SD 3.5 表现不尽如人意**：成员们反映 **SD 3.5** *在我的测试中表现甚至不如 flux dev，且远不及 ideogram 或 imagen 等大型模型。*
   - 然而，另一位成员表示 *与早期的 sd 1.5 相比，它们已经取得了长足的进步*。
- **涡轮增压般的 SD3.5 速度**：**TensorArt** 开源了 **SD3.5 Large TurboX**，采用 **8 个采样步数（sampling steps）** 实现了 **6 倍的速度提升**，且图像质量优于官方的 **Stable Diffusion 3.5 Turbo**，可在 [Hugging Face](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) 获取。
   - 他们还推出了 **SD3.5 Medium TurboX**，仅需 **4 个采样步数** 即可在中端 GPU 上用 **1 秒** 钟生成 **768x1248** 分辨率的图像，号称有 **13 倍的速度提升**，同样已上线 [Hugging Face](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 发布 Agentic Document Workflow 合作项目**：**LlamaIndex** 与 [DeepLearningAI](https://t.co/EvAKtIAzlC) 合作开设了一门课程，重点关注构建 **Agentic Document Workflows**（代理式文档工作流）。
   - 这些工作流旨在直接集成到更大的软件流程中，标志着知识型 Agent 向前迈进了一步。
- **ImageBlock 用户遇到 OpenAI 故障**：用户报告了在最新版本的 LlamaIndex 中 **ImageBlock** 与 **OpenAI** 的集成问题，系统无法识别图像；机器人建议检查最新版本并确保使用了正确的模型，例如 *gpt-4-vision-preview*。
   - 这一问题凸显了在现有 LlamaIndex 工作流中集成视觉模型的复杂性。
- **Query Fusion Retriever 的引用丢失**：一位用户发现，在他们的 LlamaIndex 配置中，**node post-processing**（节点后处理）和 **citation templates**（引用模板）无法与 **Query Fusion Retriever** 配合使用，特别是在使用 **reciprocal reranking**（倒数重排序）时，并[链接了他们的代码](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine)以供审查。
   - **Query Fusion Retriever** 中的去重过程可能是导致节点处理期间元数据丢失的原因。
- **对分布式 AgentWorkflow 架构的向往**：成员们讨论了在 **AgentWorkflow** 中原生支持**分布式架构（distributed architecture）**的可能性，即不同的 Agent 在不同的服务器/进程上运行。
   - 建议的解决方案是为 Agent 配备用于对服务进行远程调用的工具，而不是依赖内置的分布式架构支持。
- **GPT-4o Audio Preview 模型表现不佳**：一位用户报告了在 LlamaIndex Agent 中使用 **OpenAI 的音频 `gpt-4o-audio-preview` 模型**时遇到的集成挑战，特别是在流式传输事件（streaming events）方面。
   - 有人指出 AgentWorkflow 会自动对聊天消息调用 `llm.astream_chat()`，这可能与 OpenAI 的音频支持冲突，并建议了一个潜在的变通方法：避免使用 AgentWorkflow 或禁用 LLM 流式传输。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 无法摆脱物理教学大纲**：一位用户发现，当他们上传了 **180 页的物理教科书**时，系统在使用 **Gemini** 时无法脱离其教学大纲。
   - 这限制了偏离并探索教学大纲之外其他替代概念的能力。
- **PDF 上传困境**：用户在上传 PDF 时面临挑战，发现它们几乎无法使用，尤其是包含文本和图像混合内容时。Google Docs 和 Slides 在渲染混合内容方面似乎表现得更好。
   - 建议将 PDF 转换为 **Google Docs** 或 **Slides** 作为变通方案，然而这些文件格式是私有的。
- **始终期待 API 访问**：一位用户询问是否存在 **NotebookLM API** 或未来的相关计划，并列举了 AI 工程师在工作流优化方面的众多用例。
   - 访问 API 将允许用户将 NotebookLM 与其他服务集成并自动化任务，例如播客生成器。
- **移动应用思索**：一位用户询问是否有独立的 Android 版 NotebookLM 应用，另一位用户则建议网页版*运行良好*，此外还有一个 **PWA**。
   - 用户讨论了 NotebookLM 作为渐进式 Web 应用 (**PWA**) 的可用性，它可以安装在手机和 PC 上，无需专用应用即可提供类似原生应用的体验。
- **播客功能备受赞誉**：一位用户称赞 Notebook LM 的播客生成器非常精妙，但想知道是否有办法将播客长度从 *17 分钟延长到 20 分钟*。
   - 播客功能对于教育工作者和内容创作者进行讲座可能是一项宝贵的资产。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gaslight Benchmark 探索开始**：一位成员询问是否存在 **gaslight benchmark**（煤气灯效应基准测试）来比较 **GPT-4.5** 与其他模型，并得到了一个指向[讽刺性基准测试](https://spiritshare.org/benchmark.html)链接的回应。
   - 讨论强调了社区对在传统指标之外评估模型的兴趣，特别是在欺骗和说服等领域。
- **GPT-4.5 的说服力提升**：一位成员指出 **GPT-4.5 的系统卡 (system card)** 显示其在说服力方面有显著提升，这归功于 **post-training RL**（训练后强化学习）。
   - 这一观察引发了对利用 **post-training RL** 增强模型能力的初创公司的好奇，表明了 AI 开发中的一个更广泛趋势。
- **Hermes 的特殊 Token**：确认用于训练 **Hermes** 模型的特殊 Token 为 *<im_start>*、*<im_end>*、*</SCRATCHPAD>* 和 *</THINKING>*。
   - 这一澄清对于微调或集成 **Hermes** 模型的开发者至关重要，可确保正确的格式化和交互。
- **QwQ-32B 媲美 DeepSeek R1**：根据[这篇博客文章](https://qwenlm.github.io/blog/qwq-32b/)，来自 Qwen 的 **320 亿参数模型 QwQ-32B** 的表现与拥有 **6710 亿参数**的 **DeepSeek-R1** 处于相似水平。
   - 该模型可通过 [QWEN CHAT](https://chat.qwen.ai)、[Hugging Face](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、[DEMO](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 和 [DISCORD](https://discord.gg/yPEP2vHTu4) 访问。
- **RL 扩展提升模型天赋**：**Reinforcement Learning (RL)** 扩展使模型性能超越了典型的预训练，[这篇博客文章](https://qwenlm.github.io/blog/qwq-32b/)详细介绍了 **DeepSeek R1** 通过冷启动数据和多阶段训练实现复杂推理的例子。
   - 这突显了 **RL** 技术在突破模型能力边界方面日益增长的重要性，特别是在需要高级逻辑思维的任务中。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Taiga 应用集成 OpenRouter**：一款名为 **Taiga** 的开源 Android 聊天应用已发布，允许用户自定义想要使用的 LLM，并已[预集成 OpenRouter](https://github.com/Ayuilos/Taiga/releases/tag/v0.1.0-rc.0)。
   - 路线图包括基于 **Whisper** 模型和 **Transformer.js** 的本地 **Speech To Text**（语音转文本），以及对 **Text To Image**（文本转图像）和基于 **ChatTTS** 的 **TTS** 支持。
- **Prefill 功能引发讨论**：成员们质疑为什么在 **text completion**（文本补全）模式中使用 **prefill**，认为它更适合对话补全，因为将其应用于用户消息似乎不合逻辑。
   - 一位用户认为 *"prefill 对用户消息没有意义，而且他们明确将其定义为对话补全而非文本补全，哈哈"*。
- **用户请求 OpenRouter 文档汇总**：一位用户请求将 **OpenRouter 的文档**导出为单个大型 Markdown 文件，以便与 **coding agents** 无缝集成。
   - 另一位用户迅速提供了一个文档的[全文文本文件](https://openrouter.ai/docs/llms-full.txt)。
- **DeepSeek 的格式难以理解**：讨论集中在 **DeepSeek** 用于**多轮对话**的 **instruct format** 的歧义上，成员们发现甚至其 Tokenizer 配置也令人困惑。
   - 一位用户分享了 [tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)，其中定义了 `<｜begin of sentence｜>` 和 `<｜end of sentence｜>` Token 用于处理上下文。
- **讨论加入 LLMGuard 的可能性**：一位成员提出了在 OpenRouter 中为 **LLM via API** 整合插件（如 **LLMGuard**）的可能性，用于 **Prompt Injection**（提示词注入）扫描等功能。
   - 该用户链接到了 [LLMGuard](https://llm-guard.com/)，并想知道 OpenRouter 是否可以处理 PII（个人身份信息）脱敏以提高安全性。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Sparsemax 被误认为双层优化 (Bilevel Max)**：成员们讨论了将 **Sparsemax** 构造成双层优化 (BO) 问题，认为网络可以动态调整不同的神经网络层，但另一位成员迅速反驳了这一点。
   - 相反，他们详细说明了 **Sparsemax** 是向概率单纯形（probability simplex）上的投影，具有闭式解（closed-form solution），并利用拉格朗日对偶性（Lagrangian duality）证明了计算可以简化为注水算法（water-filling），从而找到闭式解。
- **DDP 导致权重错乱：PyTorch Bug 排查**：一位成员报告在使用 **PyTorch**、**DDP** 和 **4 张 GPU** 时遇到问题，在调试过程中发现 Checkpoint 重新加载导致某些 GPU 上的权重出现错乱。
   - 另一位成员建议确保在初始化 DDP *之前*，在所有 GPU 上完成模型初始化和 Checkpoint 加载，以减轻权重错乱问题。
- **Agent 主动澄清文本生成图像意图**：一篇新论文介绍了一种**主动型 T2I Agent**，它们会主动提出澄清问题，并将对用户意图的理解呈现为可编辑的置信图（belief graph），以解决用户提示词描述不充分的问题。该论文名为《[Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty](https://arxiv.org/abs/2412.06771)》。
   - 一段[补充视频](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD)显示，至少 **90%** 的人类受试者认为这些 Agent 及其置信图对他们的 **T2I 工作流**很有帮助。
- **阿里巴巴发布 QwQ-32B**：**阿里巴巴**发布了 **QwQ-32B**，这是一款参数量仅为 **320 亿**的新型推理模型，可与 **DeepSeek-R1** 等顶尖推理模型相媲美。
   - 更多信息可以在 [Qwen 博客](https://qwenlm.github.io/blog/qwq-32b)和[其公告](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19)中找到，同时阿里巴巴正在扩展 RL（强化学习）规模。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 企业支持延迟**：一位寻求 **Cohere enterprise deployment** 协助的成员被引导至邮件支持，但指出其之前的邮件已有一周未获回复。
   - 另一位成员提醒 B2B 交付周期可能延长至 **6 周**，而另一位则反驳称 **Cohere** 通常在 **2-3 天** 内回复。
- **Cohere 的 Aya Vision 支持 23 种语言**：**Cohere For AI** 推出了 **Aya Vision**，这是一个支持 **23 种语言** 的权重开放多语言视觉模型（**8B** 和 **32B** 参数），在图像描述、视觉问答、文本生成和翻译方面表现出色（[博客文章](https://cohere.com/blog/aya-vision)）。
   - **Aya Vision** 已在 [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) 和 [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision) 上线，包括新的多语言视觉评估集 **AyaVisionBenchmark**；聊天机器人也已在 [Poe](https://poe.com/Aya-Vision) 和 [WhatsApp](https://cohere.com/research/aya/whatsapp) 上线。
- **Cohere Reranker v3.5 延迟数据缺失**：一位成员请求 **Cohere Reranker v3.5** 的延迟数据，指出尽管在采访中有所承诺，但仍缺乏公开数据。
   - 受访者曾承诺分享图表，但最终未能提供。
- **用户寻找销售/企业支持联系方式**：一位新用户加入，寻求联系 Cohere 的 **sales / enterprise support** 人员。
   - 该用户被鼓励进行自我介绍，包括公司详情、行业、大学、当前项目、喜爱的技术/工具以及加入社区的目标。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 仍处于开发阶段**：一位成员报告称 **Mojo** 仍不稳定，*还有大量工作要做*；另一位成员询问虚拟活动的 **YouTube 录像**，但获知该活动*未录制*。
   - 团队提到他们*未来肯定会考虑举办类似的虚拟活动*。
- **Triton 被提议作为 Mojo 替代方案**：一位成员建议将 **Triton**（支持 **Intel** 和 **Nvidia** 硬件的 AMD 软件）作为 **Mojo** 的潜在替代方案。
   - 另一位成员澄清说 **Mojo** 不是 Python 的超集，而是 *Python 语言家族的一员*，并表示成为超集对 **Mojo** 来说就像是*戴上了口罩（束缚）*。
- **Mojo 在 Python venv 中性能下降**：基准测试显示，在激活的 **Python virtual environment** 中运行 Mojo 二进制文件时，**Mojo 的性能提升**会显著降低，即使对于没有 Python 导入的文件也是如此。
   - 用户寻求关于为什么 Python venv 会影响本应独立的 Mojo 二进制文件的见解。
- **项目文件夹结构咨询**：一位开发者请求对 **Mojo/Python 项目文件夹结构** 的反馈，该结构涉及导入标准 Python 库并运行用 Mojo 编写的测试。
   - 他们广泛使用 **`Python.add_to_path`** 进行自定义模块导入，并在 `tests` 文件夹中使用符号链接（Symlink）来定位源文件，寻求更好的替代方案。
- **文件夹结构讨论移至 Modular 论坛**：一位用户在 Modular 论坛上发起了关于 **Mojo/Python 项目文件夹结构** 的讨论，[链接至论坛帖子](https://forum.modular.com/t/mojo-python-project-folder-structure/677)。
   - 此举旨在确保讨论的长期可发现性和保留，因为 *Discord 的搜索和数据保留功能较差。*

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SynaLinks 进入 LM 竞技场**：一个新的**基于图的可编程神经符号 LM 框架** **SynaLinks** 已发布，其函数式 API 灵感源自 **Keras**，旨在通过异步优化和约束结构化输出等特性达到生产就绪状态 - [GitHub 上的 SynaLinks](https://github.com/SynaLinks/synalinks)。
   - 该框架已在客户的生产环境中运行，重点关注**知识图谱 RAG、强化学习和认知架构**。
- **Adapter 解耦了 DSPy 中的 Signature**：**DSPy 的 Adapter 系统**将 Signature（你想要内容的声明式规范）与不同提供商生成 Completion 的方式解耦。
   - 默认情况下，**DSPy** 使用经过良好调优的 **ChatAdapter**，并回退到 **JSONAdapter**，利用结构化输出 API 在 **VLLM**、**SGLang**、**OpenAI**、**Databricks** 等提供商中进行约束解码。
- **DSPy 简化了显式类型指定**：DSPy 通过如下代码简化了显式类型指定：```contradictory_pairs: list[dict[str, str]] = dspy.OutputField(desc="List of contradictory pairs, each with fields for text numbers, contradiction result, and justification.")```，但由于未指定 `dict` 的键，这在技术上存在歧义。
   - 相反，应考虑使用 ```list[some_pydantic_model]```，其中 **some_pydantic_model** 具有正确的字段。
- **DSPy 解决滞后线程问题**：[PR 7914](https://github.com/stanford-nlp/dspy/pull/791)（已合并）解决了 `dspy.Evaluate` 或 `dspy.Parallel` 中卡住的*滞后（straggler）*线程问题，旨在实现更顺畅的运行。
   - 此修复将在 **DSPy 2.6.11** 中可用；用户可以从 `main` 分支进行测试，无需更改代码。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker 合并证明接近完成**：一名成员宣布在 Lean 中完成了一个关于何时可以合并 ShapeTracker 的证明，进度约为 90%，可在[此仓库](https://github.com/Nielius/Tensorlayouts)和[此 Issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082)中查看。
   - 作者指出尚未考虑偏移量（offsets）和掩码（masks），但扩展该证明非常直接。
- **在淘宝上解锁 96GB 4090**：一名成员分享了淘宝上 **96GB 4090** 的链接（[X 帖子](https://x.com/yomix1337/status/1893692548108984391?s=46)），引发了“*好东西都在淘宝上*”的评论。
   - 未有进一步讨论。
- **调试 gfx10 Trace 问题**：一名成员请求关于 **gfx10 trace** 的反馈，询问是否应将其记录为 Issue。
   - 另一名成员怀疑这与 **ctl/ctx** 大小有关，并建议运行 `IOCTL=1 HIP=1 python3 test/test_tiny.py TestTiny.test_plus` 以协助调试。
- **评估 Rust CubeCL 质量**：一名成员询问 **Rust CubeCL** 的质量，并指出它来自 **Rust Burn** 的原班人马。
   - 讨论未对其质量得出结论性评估。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Suleiman 投身 AI 生物黑客（Biohacking）**：具有软件工程背景的高管 Suleiman 向频道介绍了自己，表达了对 **AI** 和**生物黑客**的兴趣。
   - 他正在探索**营养学**和**补充剂科学**，以开发 **AI 赋能的生物黑客工具**来改善人类生活。
- **Naveen 研究 Txt2Img 模型的机器遗忘**：来自 IIT 的硕士兼研究助理 Naveen 介绍了自己及其在**文本生成图像扩散模型**中的**机器遗忘（Machine Unlearning）**工作。
   - 他提到最近在 **CVPR25** 发表了一篇论文，重点研究从生成模型中移除不当概念的策略。
- **ARC 训练的普适性悬而未决**：一名用户质疑 **Observation 3.1** 是否对几乎任何两个具有非零均值的分布以及 **ARC 训练**中几乎任何 u35% 的情况都普遍成立。
   - 讨论陷入停滞，没有明确结论，也未就 **Observation 3.1** 的具体条件或例外情况进行讨论。
- **压缩产生智能？**：Isaac Liao 和 Albert Gu 在他们的[博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)中探讨了**无损信息压缩**是否能产生**智能行为**。
   - 他们专注于实际演示，而不是重新讨论关于**高效压缩**在智能中作用的理论探讨。
- **ARC Challenge 使用 YAML 配置**：成员们讨论了使用 **arc_challenge.yaml** 来设置 **ARC-Challenge 任务**。
   - 讨论涉及配置模型使用 **25 shots** 进行评估，强调了 **few-shot learning** 能力在应对该挑战中的重要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 中的自定义 Tokenizer 问题**：用户在训练后遇到了 **Torchtune** 用来自 **Hugging Face** 的原始文件覆盖自定义 **special_tokens.json** 的问题，这是由于 [checkpointer](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-896) 中的 *copy_files* 逻辑导致的。
   - 提出的快速修复方案包括在下载的模型目录中，手动用用户的自定义版本替换下载的 **special_tokens.json**。
- **关于 Checkpointer save_checkpoint 方法的辩论**：一名成员建议通过向 **Torchtune** 中 checkpointer 的 **save_checkpoint** 方法传递新参数，来支持自定义 tokenizer 逻辑。
   - 然而，其他人质疑在没有充分理由的情况下暴露新配置的必要性。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 学生可观看所有讲座**：一名成员询问 **Berkeley** 学生是否拥有 **MOOC** 学生无法访问的专属讲座，特别是针对 **LLM Agents MOOC**。
   - 另一名成员澄清说，**Berkeley** 学生和 **MOOC** 学生参加的是相同的讲座。
- **学生回忆 12 月的提交情况**：一名成员提到在 12 月提交了与课程相关的内容，推测是 **LLM Agents MOOC** 的证书声明表。
   - 另一名成员寻求关于提交证书声明表所使用的特定电子邮件地址的确认，暗示可能需要进行行政跟进。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **寻求 AST Metric 的澄清**：一名成员在 **Gorilla LLM Leaderboard** 频道中寻求关于 **AST metric** 定义的澄清。
   - 他们询问 **AST metric** 是否代表 **LLM** 响应生成的格式正确的函数调用百分比。
- **成员询问 V1 Dataset 构建方式**：一名成员询问了用于构建 **Gorilla LLM Leaderboard** 的 **V1 dataset** 的方法论。
   - 了解 **dataset construction** 过程可以为排行榜的 **evaluation methodology** 提供有价值的见解。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: 频道详情摘要与链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1346800454369017957)** (676 messages🔥🔥🔥): 

> `Cursor 3.7 争议, Dumbness Meter, YOLO mode 配置, Windsurf 对比 Cursor, OpenAI 新定价`

- **Cursor 的 3.7 模型引发了关于性能和质量的辩论**：成员们反映 **Cursor 的 3.7 模型感觉被削弱了 (nerfed)**，理由是它会在*没有提示的情况下生成 readme 文件*，并且*过度使用 Class A 抽象*。
   - 一些用户怀疑 **Cursor 要么使用了庞大的 Prompt，要么使用了虚假的 3.7 模型**，并指出在相同的 Prompt 和指令下，API 有时会提供更好的建议。一位成员分享了一个[关于 Cursor 编辑器今天感觉有多笨的科学测量工具](https://v0-next-js-website-orcin.vercel.app/)。
- **社区使用讽刺工具测量 Cursor 的愚蠢程度**：一位成员分享了一个[链接](https://v0-next-js-website-orcin.vercel.app/)，指向一个测量 Cursor 编辑器**“愚蠢程度”**的**“高度复杂的仪表”**，该仪表使用了基于*“宇宙射线、键盘失误以及代码补全错误的次数”*的*“高级算法”*。
   - 该仪表的读数会定期更新，引发了幽默的反应，一位用户报告说他们的仪表显示 Cursor **“今天很笨”**。另一位用户确认道：**“绝对的”**。
- **更新后 YOLO 模式的困扰困扰着 Cursor 用户**：更新后，**Cursor 中的 YOLO 模式无法正常工作**，因为现在即使在允许列表为空的情况下，运行命令前也需要批准。
   - 一位用户表达了沮丧，表示他们希望 **AI Assistant 拥有尽可能多的 Agency**，并依靠 Git 来处理任何错误的删除，他们更喜欢曾为他们节省数小时时间的 v45 版本的行为。
- **Windsurf 作为 Cursor 的潜在替代方案受到关注**：社区成员正在积极讨论 [Windsurf 的新版本](https://codeium.com/blog/windsurf-wave-4) Wave 4，一些人由于感知到 Agent 能力方面的优势而考虑切换，有人反映 Cursor 变得非常难用，而另一些人则分享了一个关于 **“Vibe Coding 教程与最佳实践 (Cursor / Windsurf)”** 的 [YouTube 教程](https://www.youtube.com/watch?v=YWwS911iLhg)。
   - 尽管有兴趣，但对 Windsurf 定价模式的担忧依然存在，一些用户提到它顺走了 (yoinks) continue.dev 的功能。
- **OpenAI 准备收取高额费用**：一位成员分享了一份报告，称 [OpenAI 正在加倍投入其应用业务](https://x.com/steph_palazzolo/status/1897309493744267314?s=46)，计划为能够自动化编程和博士级研究的高级 Agent 提供每月 **2,000 美元至 20,000 美元**不等的订阅服务。
   - 这一消息引发了怀疑，一些人质疑如此高的价格是否合理，尤其是考虑到目前 AI 模型的输出质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://v0-next-js-website-orcin.vercel.app/">Cursor Editor Dumbness Meter</a>: 未找到描述</li><li><a href="https://fontawesome.com/icons/house?f=classic&s=solid">House Classic Solid 图标 | Font Awesome</a>: Solid 风格的 House 图标。在小尺寸下也能脱颖而出。现已在 Font Awesome 6 中提供。</li><li><a href="https://www.youtube.com/watch?v=YWwS911iLhg">Vibe Coding 教程与最佳实践 (Cursor / Windsurf)</a>: 收到很多关于我的技术栈以及我在进行 vibe coding 时做什么的问题。所以我制作了一个完整的视频！👉 在 https://mammouth.ai/ 了解更多...</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>: 最新消息（与 @coryweinberg 合作）：OpenAI 正在加倍投入其应用业务。高管们已与投资者讨论了未来三类 Agent 的发布，价格从每月 2,000 美元到 20,000 美元不等，用于执行诸如...的任务。</li><li><a href="https://codeium.com/blog/windsurf-wave-4">Windsurf Wave 4</a>: 介绍 Wave 4，这是我们对 Windsurf 编辑器的第四批更新。</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode: 使用任何 LLMs 进行编码</a>: 使用任何 LLMs 进行编码。通过在 GitHub 上创建账户来为 dnakov/anon-kode 的开发做出贡献。</li><li><a href="https://forms.gle/jXNunfmixHAiWZ168">基于 Web 的集成开发环境 (IDEs) 的功能、可用性与优势：Cursor.ai</a>: 我们是 ODU 的一群计算机科学毕业生，旨在进行研究以探索基于 Web 的集成开发环境 (IDEs) 的功能、可用性和优势。这些平台允许开发人员...</li><li><a href="https://tenor.com/view/mob-psycho-season3-mob-psycho-shigeo-shigeo-kageyama-mob-gif-26873452">路人超能 100 (Mob Psycho) 第三季 影山茂夫 GIF - 路人超能 100 第三季 路人超能 100 影山茂夫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://forum.cursor.com/t/cursor-0-46-unable-to-solve-a-problem-cursor-0-45-fixed-it-in-one-shot-with-sonnet-3-7-thinking/58036/10">Cursor 0.46 无法解决问题，Cursor 0.45 配合 Sonnet 3.7 thinking 一次性解决</a>: 0.46.8 的体验非常糟糕，我换回了 GitHub Copilot。尽管 Claude Sonnet 3.7 感觉更快，但它在消耗你的额度，Cursor AI 提供的质量比以前差得多...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1346922878729457785)** (2 条消息): 

> `GPT-4.5, AGI Development, AI Safety, AI Alignment` 


- **GPT-4.5 提前发布**：**GPT-4.5** 的部署已*比预期更快*地完成，标志着 AI 发展的一个重要里程碑。
- **概述迭代式 AGI 发展路径**：公司将 **AGI 发展**视为一条*持续的路径*，而非突然的飞跃，重点在于通过对现有模型的迭代部署和学习，使未来的 AI 更加安全且有益。
   - 这种方法通过从当前模型中学习来为 AGI 做准备，而不是为*单个关键时刻*做准备。
- **讨论安全与对齐方法**：公司在 **AI 安全与对齐**方面的方法以拥抱不确定性、深度防御、可扩展的方法、人类控制和社区努力为指导。
   - 目标是确保 **AGI 造福全人类**，并强调持续改进和适应。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1346817573722914846)** (546 条消息🔥🔥🔥): 

> `O3 会发布吗？，毕业设计的自定义 LLM，Grok 对比 ChatGPT Plus，Abacus AI 反馈，Claude 秘密更新` 


- **关于 O3 发布及编程能力的推测**：一名成员询问了 **O3** 的发布情况，以及它在编程方面是否会超越 **O3-mini**，另一名成员回答说 *OpenAI 表示他们不会在 ChatGPT 中发布完整的 O3，仅在 API 中提供*。
   - 不同的选项指示了诸如对你的语气等方面的细节。“不是（not a）”旨在提醒用户它仍然是一个 **AI**，因此并不总是 100% 准确，并且*应该始终咨询人类治疗师或医生*。
- **在低配硬件上进行本地 LLM 训练**：一名拥有 **RTX 3050 4GB VRAM** 和 **8 GB RAM i5 10th gen** 笔记本电脑的学生正在寻求关于在本地运行和微调自定义 **LLM** 以用于毕业设计的建议。
   - 一位用户建议利用 **Grok** 的 **Deep Research** 功能来获取合适模型的最新信息，并提醒说许多聊天机器人推荐的是过时的模型；另一名成员提到，免费账户每天可能可以使用 **3 次** **Deep Research**，有人告诉他他使用了 **5 个免费 Grok** 账号，以便每天发送 **15 条 Deep Research** 消息。
- **Grok 和 Perplexity 在学术研究工具中脱颖而出**：用户对比了 **Grok** 和 **ChatGPT Plus**，指出 **Grok 3** 拥有更大的上下文窗口（**128k** 对比 **GPT** 的 **32k**）、更少的审查（仅限英文）以及相当的创意写作能力，一位用户对 ChatGPT、Perplexity、Grok 和 Stanford genie 模型进行了排名。
   - 成员们讨论了每个平台的优势，一些人提到 *Grok 的 Deep Research* 在有限次数内是免费的，并强调 **Perplexity** 非常适合研究并能提供非常出色的研究结果。
- **Claude 一夜之间变聪明了？**：一位用户报告称 **Claude** 经历了显著的提升，能够追踪并修复他们代码中的 Bug，甚至改进了用户之前未意识到存在 Bug 的代码。
   - 成员们推测可能存在更新或测试，并称赞该模型具有更温暖、更像人类的沟通风格。
- **带有递归摘要功能的 Qwen-14b 碾压 Gemini**：成员们讨论了用于递归摘要任务的 **Qwen-14B** 模型，其输出效果优于 **Gemini**。
   - 他们以一本国际象棋书籍的摘要为例，**Gemini** 的表现像 **GPT-3.5** 甚至更糟，这在当今标准下是不可接受的，因此 **Qwen-14B** 的结果更好。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://storm.genie.stanford.edu/">未找到标题</a>：未找到描述</li><li><a href="https://pastebin.com/TamXCnL0">这份文档是关于 Nise da Silveira 的 'Jung: Vida e Obra' (1981) - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/Z43vTi9v">这份文档是关于 "Chess Opening Essentials: Volume 1 — The Complete 1.e4," - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。</li><li><a href="https://pastebin.com/inNnMdF0">摘要介绍了 "Chess Opening Essentials: Volume I — The Complete 1.e4," - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。</li><li><a href="https://youtu.be/Vshg-hNUEjo">Nana Mouskouri - Guten Morgen, Sonnenschein 1977</a>：Nana Mouskouri - Guten Morgen, Sonnenschein 1977</li><li><a href="https://github.com/opennars/OpenNARS-for-Applications">GitHub - opennars/OpenNARS-for-Applications: 基于 NARS 理论的应用通用推理组件。</a>：基于 NARS 理论的应用通用推理组件。 - opennars/OpenNARS-for-Applications
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1346907244192596150)** (10 messages🔥): 

> `GPT-4.5 拒绝提示词，GPT-4.5 消息限制` 


- **GPT-4.5 拒绝故事提示词**：用户报告称 **GPT-4.5** 拒绝响应基于故事的提示词，即使这些提示词并未违反指南。
   - 一位用户指出，他们仅成功让 **GPT-4.5** 为其故事提示词工作过一次。
- **GPT-4.5 的有限可用性被披露**：一名成员透露，**GPT-4.5** 目前限制为大约 **每周 50 次使用**，并可能逐渐增加。
   - 他澄清说，**GPT-4.5** 并非旨在取代包括 **GPT-4o** 在内的其他模型，并建议用户根据当前任务在不同模型之间切换。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (12 messages🔥): 

> `Prompt Engineering 调查，提示策略本体论，Sora 与 AI 视频，Sora 中的角色一致性，超写实视觉效果` 


- **Prompt Engineering 调查概述关键策略**：一名成员分享了一份名为 *A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications* 的 LLM 提示工程系统调查，概述了 **Zero-Shot Prompting**、**Few-Shot Prompting** 和 **Chain-of-Thought (CoT)** 提示等关键策略。
   - 讨论强调了该调查对每种技术的详细描述，同时也指出了遗漏之处，如 **Self-Discover** 和 **MedPrompt**。
- **ChatGPT 链接实现对 Prompt Engineering 调查的访问**：一名成员分享了一个 [ChatGPT 链接](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854)，以提供对他们无法直接链接的学术提示工程调查的公开访问。
   - 该成员表示倾向于公开信息共享，特别是对于详尽的资源，并指出该调查原始的复杂性对于 *单条 Discord 帖子来说内容过多*。
- **用户寻求在 Sora 中保持 AI 角色一致性的提示技巧**：一名成员请求关于使用 **Sora** 创建电影级 AI 视频的建议，重点关注一个名为 **Isabella Moretti** 的一致角色，旨在实现超写实视觉效果并提高角色一致性。
   - 该成员正在寻求有效的策略或提示技巧，以保持一致的外貌细节（**肤色**、**眼睛**、**头发**、**表情**），并优化提示结构以获得最佳电影质量（**光照**、**镜头运动**、**过渡**）。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)** (12 messages🔥): 

> `Large Language Models 中 Prompt Engineering 的系统调查，提示策略本体论，Sora，AI 视频中的角色一致性，Sora 中的超写实视觉效果` 


- **Prompt Engineering 调查发布**：一名成员分享了从学术调查 *"A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications"* 中提取的 **Prompt Engineering 技术**摘要。
   - 该调查对提示策略进行了分类，包括 **zero-shot**、**few-shot**、**chain-of-thought**、**RAG** 和 **emotion prompting**。
- **Prompt Engineering 调查亮点**：一名成员分享了一个包含提示工程学术调查的 [ChatGPT 链接](https://chatgpt.com/share/67c89f53-c72c-8000-b64b-ca30c6971854)。
   - 发布者指出这 *甚至不是一个详尽的本体论*，缺少了 **Self-Discover** 和 **MedPrompt**，且完整的本体论对于 Discord 来说过于详细。
- **社区成员寻求 Sora 技巧**：一名成员正在使用 **Sora** 创建以角色 **Isabella Moretti** 为中心的电影级 AI 视频，并寻求 **超写实视觉效果** 的技巧。
   - 目标是提高角色细节（如肤色、眼睛和头发）的一致性，通过专注于光照、镜头运动和过渡来优化提示结构，以获得最佳的电影质量。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1346939336792342611)** (1 条消息): 

> `Windsurf Wave 4 发布，Cascade Previews，Tab-to-import，Linter 集成，Claude 3.7 改进` 


- **Windsurf Wave 4 正式发布！**: Windsurf 发布了 **Wave 4**，这是其迄今为止最大的更新，包含了诸如 **Previews**、**Tab-to-import**、**Linter 集成**、**建议操作 (Suggested actions)**、**MCP 可发现性**以及对 **Claude 3.7** 的改进等[变革性功能](https://www.codeium.com/blog/windsurf-wave-4)。
   - 该更新还包括一个新的推荐计划，并支持从 explorer 拖放文件到 Cascade 以及对 Windows ARM 的支持。
- **Cascade Auto-Linter 修复错误**: **Cascade** 现在通过其新的 Linter 集成，自动修复生成代码中的 lint 错误。
   - 用户可以在 IDE 或浏览器中预览本地运行的网站，选择 **React** 和 **HTML** 元素作为上下文发送给 Cascade，并发送控制台错误作为上下文。
- **Tab-to-import 增强工作流**: 新的 **Tab-to-import** 功能可以通过按下 tab 键自动添加 import，从而简化 Cascade 中的编码工作流。
   - 可以通过要求 Cascade 启动你的 Web 应用程序，或通过对话输入框上方工具栏中的 Website 工具图标来激活 **Windsurf Preview** 功能。
- **Windsurf 现已登陆 YouTube！**: 一段 [YouTube 视频](https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be) 重点介绍了 **Windsurf Wave 4** 的更新，涵盖了 **Preview**、**Tab to Import**、**建议操作**等内容。
   - 视频描述敦促用户*更新到最新版本的 Windsurf* 以获取所有新功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.codeium.com/blog/windsurf-wave-4">Windsurf Wave 4</a>: 介绍 Wave 4，这是我们对 Windsurf Editor 的第四批更新。</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor 更新日志 | Windsurf Editor 和 Codeium 扩展</a>: Windsurf Editor 的最新更新和变化。</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">来自 Windsurf (@windsurf_ai) 的推文</a>: Windsurf Wave 4 来了！本次更新包含：🖼️ Previews ✏️ Cascade Auto-Linter ⚙️ MCP UI 改进 ➡️ Tab to Import ↩️ 建议操作 🫶 Claude 3.7 改进 🤝 推荐计划 🖥️ Windows ARM 支持...</li><li><a href="https://bsky.app/profile/windsurfai.bsky.social/post/3ljnsaugqk22l">Windsurf (@windsurfai.bsky.social)</a>: Windsurf Wave 4 来了！本次更新包含：🖼️ Previews ✏️ Cascade Auto-Linter ⚙️ MCP UI 改进 ▶️ Tab to Import ↩️ 建议操作 🫶 Claude 3.7 改进 🤝 推荐计划 🖥️ Windows ARM 支持...</li><li><a href="https://www.threads.net/@codeiumdev/post/DG1IyC5CODS?xmt=AQGzB0CoP8oQ9hE-8YatsFH7FaIFFpnONInUNHCSr9H8qg">Threads 上的 Codeium (&#064;codeiumdev)</a>: Windsurf Wave 4 来了！本次更新包含：&#x1f5bc;&#xfe0f; Previews &#x270f;&#xfe0f; Cascade Auto-Linter &#x2699;&#xfe0f; MCP UI 改进 &#x25b6;&#xfe0f; Tab to Import &#x21a9;&#xfe0f; 建议操作...</li><li><a href="https://www.youtube.com/watch?v=bIy-RN3FIsQ&feature=youtu.be">Windsurf Wave 4 更新：Preview, Tab to Import, Suggested Actions &amp; 更多</a>: Windsurf Wave 4 来了，带来了令人兴奋的新功能来增强您的体验！🌊 请务必更新到最新版本的 Windsurf 以获取所有这些功能...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346812925649748030)** (6 条消息): 

> `vscode commit message, flutterflow, uninstalling codium extension` 


- **VSCode 提交信息生成失败**: 一位用户报告了在 **VSCode** 中生成提交信息的问题，特别是在使用 Codeium 扩展的 **pre-release 版本**时。
   - 他们询问了针对此问题的任何可用变通方法。
- **寻求 FlutterFlow 帮助**: 一位用户询问是否有人知道如何使用 **FlutterFlow**。
   - 没有关于所需具体帮助的进一步讨论或细节。
- **需要卸载扩展的帮助**: 一位用户请求帮助完全卸载当前的 **Codeium 扩展**。
   - 未提供关于卸载原因或过程中遇到的具体问题的进一步细节。


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1346807307966746674)** (397 条消息🔥🔥): 

> `Windsurf 性能下降, Codeium 登录问题, Windsurf Wave 4, 额度使用, 功能请求`

- **Windsurf 用户因服务中断陷入困境**：用户报告 **Windsurf/Cascade** 出现 *resource exhaustion*（资源耗尽）等错误且*不再工作*，部分用户遇到 100 行的文件被拆分为 5 或 6 个分析工具调用的问题。
   - 一些成员注意到工具调用的分析阈值从 **每调用 150 行** 降至 **每调用 15 行**。
- **Codeium 代码难题：凭据灾难**：多名用户报告无法使用 Google 或邮箱/密码登录 [codeium.com](https://codeium.com)。
   - 团队承认了该问题并表示正在调查，并提供了一个 [状态页面](https://status.codeium.com) 以获取更新。
- **Wave 4 的惊喜与困扰：Windsurf 的旋风**：成员们讨论了 **Windsurf Wave 4**，一些人称赞其速度以及对 **Claude 3.7** 的集成，而另一些人则报告了死循环和信用额度消耗增加的问题。
   - 提到在 Windsurf Wave 4 中可以通过 **Windsurf Command**（使用 `CTRL/Cmd + I`）*免费* 使用 **Claude 3.7**。
- **Windsurf 额度紧缺：消费者面临高昂消耗**：用户对**额度使用**表示担忧，特别是 **Claude 3.7**，一些人注意到即使在最近改进后，额度消耗依然很快。
   - 澄清了 **Flex Credits** 可以结转，且自动修复 lint 错误是免费的。
- **Windsurf 愿望清单：用户渴望升级**：用户请求支持库的外部文档、增加额度限制、调整聊天字体大小的功能，以及像 Trae 那样在侧边栏提供真正的 webview。
   - 一位用户建议使用 [Firecrawl](https://x.com/ericciarla/status/1897332080708858147) 为网站生成 llms.txt 文件，以便输入到 LLM 中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://neon.tech">Neon Serverless Postgres — 交付更快</a>：你喜爱的数据库，运行在旨在帮助你更快构建可靠且可扩展应用程序的 Serverless 平台上。</li><li><a href="https://docs.codeium.com/windsurf/previews">Previews (Beta) - Codeium 文档</a>：未找到描述</li><li><a href="https://codeium.com/account/login">登录 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/codeium.com/windsurf/directory">页面未找到 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/windsurf/directory">Windsurf 规则目录</a>：未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，让开发者保持专注。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://x.com/ericciarla/status/1897332080708858147">Eric Ciarla (hiring) (@ericciarla) 的推文</a>：使用 /llmstxt 在几秒钟内为任何网站生成 llms.txt 文件。我们新的 @firecrawl_dev 端点可以将任何站点转换为单个文本文件，并输入到任何 LLM 中。查看它作为 @rayc 集成的功能...</li><li><a href="https://docs.codeium.com/supercomplete/overview#tab-to-jump">概览 - Codeium 文档</a>：未找到描述</li><li><a href="https://docs.codeium.com/command/overview">概览 - Codeium 文档</a>：未找到描述</li><li><a href="https://www.youtube.com/@codeiumdev/videos">Codeium - Windsurf</a>：🧑‍💻 | 你的现代编程超能力🚀 | 300万+ Codeium 扩展下载量🏄‍♂️ | 构建 Windsurf 编辑器</li><li><a href="https://codeium.com/windsurf/directory,">页面未找到 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://pierre.co/">Pierre</a>：愉悦的代码审查</li><li><a href="https://status.codeium.com">Codeium 状态</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/improve-previews-feature-with-a-proper-webview-in-the-sidebar-like-trae">改进“预览”功能，在侧边栏提供真正的 “Webview”（类似 Trae） | 功能请求 | Codeium</a>：我希望在侧边栏能有一个简单的“预览”工具，就像 Trae 有一个 “Webview” 工具一样。仅仅为了预览而提示 AI 感觉有点奇怪。</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1346800557029064759)** (201 条消息🔥🔥): 

> `Grok 3 模型对比、Aider 离线安装、Qwen 新发布的 QwQ-32B 推理模型、OpenAI o3 Mini 访问权限、Parasail 在 OpenRouter 上的 R1 性能` 


- **Grok vs Claude：模型性能之争**：用户对比了 **Grok 3**、**Claude 3.7** 和其他模型，探讨了免费版与付费版之间的性能差异，一位用户开玩笑说 *"它们可能都差不多，但 Grok 是免费的"*。
   - 讨论表明，模型偏好是主观的，且可能受到成本因素的影响。
- **离线 Aider：任务达成！**：一位用户寻求在**离线 PC** 上安装 Aider，引发了关于安装方法的讨论，包括下载 Python 包并从联网 PC 复制。最终通过使用 **pip download** 将包下载到离线 venv 中解决了问题。
   - 一位成员发布了复制 Aider 安装的成功命令序列：`python -m pip download --dest=aider_installer aider-chat`。
- **Qwen 发布全新 QwQ-32B 推理模型**：**Qwen** 发布了拥有 32B 参数的推理模型 **QwQ-32B**，据称可与 **DeepSeek-R1** 媲美。该模型在 [VXReddit 帖子](https://www.vxreddit.com/r/LocalLLaMA/comments/1j4b1t9/qwq32b_released_equivalent_or_surpassing_full/) 中被讨论，并由 Qwen 官方宣布（[博客](https://qwenlm.github.io/blog/qwq-32b), [HF](https://huggingface.co/Qwen/QwQ-32B), [ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)）。
   - 初步反应积极，社区成员渴望看到它在 Aider 中作为架构师（architect）和代码编写者（coder）的表现。
- **o3 Mini：OpenAI 梯队的新成员**：用户报告已获得 **OpenAI** 上 **o3 mini** 的访问权限，并思考它是否能成为一个出色的编辑器模型。
   - 其他人称赞 **o3-mini** 是一个超快的架构师，并指出它 *"不是常规的 Chain-Of-Thought 模型"*，认为它是一个值得尝试的选择。
- **ParaSail 宣称 R1 速度极快**：一位用户报告在 **OpenRouter** 上使用 **Parasail** 供应商时，**R1 的吞吐量达到了惊人的 300tps**。
   - 虽然其他人未能立即复现该速度，但该供应商被认为是与 **SambaNova** 并列的 **R1** 顶级性能提供商之一。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.apple.com/macbook-air/">MacBook Air 13 英寸和 MacBook Air 15 英寸</a>：搭载超快 M4 芯片的 MacBook Air 笔记本电脑。专为 Apple Intelligence 打造。轻巧且具备全天候电池续航。现推出全新天蓝色。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-a">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://x.com/testingcatalog/status/1897366902701502868">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Qwen 发布了新的推理模型 QwQ-32B，如果你选择带有 Thinking (QwQ) 的 Qwen2.5-Plus，它现在正为 Qwen Chat 提供支持。引用 Qwen (@Alibaba_Qwen)：“今天，我们发布了 QwQ-32B，我们新的推理模型...”</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，我们新的推理模型，仅有 320 亿参数，可媲美顶尖推理模型，例如 DeepSeek-R1。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://x.com/mrousavy/status/1897222044808569137">来自 Marc (@mrousavy) 的推文</a>：字节跳动刚刚推出了 Lynx —— React Native 的竞争对手！</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mujikcboro-seriymujik-gif-24361533">Mujikcboro Seriymujik GIF - Mujikcboro Seriymujik - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/xi-jinping-gif-24241864">Xi Jinping GIF - Xi Jinping - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/olilanz/RooCode-Local-Evaluation">GitHub - olilanz/RooCode-Local-Evaluation</a>：Roo Code 和本地托管 LLM 的评估。通过在 GitHub 上创建账户为 RooCode-Local-Evaluation 的开发做出贡献。</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple 推出全新 Mac Studio，史上最强大的 Mac</a>：Apple 今天发布了全新的 Mac Studio，这是有史以来最强大的 Mac，搭载 M4 Max 和全新的 M3 Ultra 芯片。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1346861361837244618)** (53 messages🔥): 

> `OWUI Integration, LM Studio R1, Aider Output, OpenRouter API, Commit Messages` 


- **使用兼容 OAI 的 Aider 技巧连接 OWUI**：要将 **Aider** 连接到 **OpenWebUI (OWUI)**，请在模型名称前加上 `openai/` 前缀，以便 *litellm* 识别你正在使用 **OAI-compatible endpoint**，例如 `openai/myowui-openrouter.openai/gpt-4o-mini`。
   - 这解决了在 **OWUI** 中使用 **Aider** 时出现的 `litellm.BadRequestError`。
- **DeepSeek 提供商问题中的 Aider 故障**：一位成员表示，如果 **OpenRouter** 没有返回 tokens，这可能是 **litellm** 和 **Aider** 的问题，而不是 **DeepSeek provider** 的问题。
   - 针对 **OpenRouter** 的 reasoning 字段已合并了一个补丁 ([PR #8431](https://github.com/BerriAI/litellm/pull/8431))，但可能需要 Aider 的本地补丁才能显示 reasoning 内容。
- **Commit Messages**：一位成员建议了一个方案，让 **aider** 只为你已暂存 (staged) 的文件编写 commit 消息，而不是针对工作树中的所有更改：先执行 `git stash save --keep-index`，然后执行 `/commit`，最后执行 `git stash pop`。
   - Aider 也可以通过 `aider --commit` 作为 committer 使用，它会编写 commit 消息、提交并退出。
- **寻找优秀的（免费）编辑器模型**：成员们讨论了适合作为 `editor-model` 的模型，建议包括使用 `qwencoder2.5:32b` 作为弱模型（用于 commit 和压缩/总结历史），以及使用 `gemini flash` 进行编辑。
   - 其他成员报告称，使用 `o3-mini-high` 作为 architect，`deepseek-v3` 作为 editor 效果良好。
- **/web 命令**：一位成员报告说 `/web` 命令无法工作，即使让 **Aider** 安装了 **Playwright**，它也没有将内容添加到 context 中。
   - 另一位成员确认该功能有效，并建议通过询问基于抓取页面的问题来验证内容是否已添加到聊天中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>：aider 是你终端里的 AI 结对编程助手</li><li><a href="https://aider.chat/docs/git.html#commit-messages">Git integration</a>：Aider 与 git 紧密集成。
</li>
</ul>

</div>

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1346803804561211392)** (84 条消息🔥🔥): 

> `VRAM 溢出，LM Studio 和 Phi-4 音频模态支持，KV cache 对 VRAM 的影响，新 Mac Studio 的 RAM，Sesame AI 的开源 TTS 模型` 


- **通过观察共享内存检测 VRAM 溢出**：一位成员分享了如何通过监控 **Dedicated memory**（专用内存）和 **Shared memory**（共享内存）的使用情况来检测 VRAM 溢出，并指出当 *Dedicated memory* 占用极高且 *Shared memory* 开始增加时即发生溢出，详见[附图](https://cdn.discordapp.com/attachments/1110598183144399061/1346803804322009088/VRAM_Overflow.jpg?ex=67ca2d09&is=67c8db89&hm=24b703c40c580b2636786230775506086194cec8387515d56546d86fefc79989&)。
- **LM Studio 中的 Phi-4 缺失音频模态**：成员们确认，由于 *llama.cpp* 的限制，目前 LM Studio 尚不支持 **多模态 Phi-4** 和 **音频支持**。
- **KV Cache 锁定你的 VRAM**：一位成员解释说，**context size**（上下文大小）和 **KV cache** 设置会显著影响 VRAM 占用，建议以 **90% VRAM** 利用率为目标以优化性能。
   - 另一位成员将 KV cache 定义为计算机进行 Attention 机制数学运算（即 Q*K^T/dK）时的 *K 和 V 的值*。
- **Sesame AI 的 TTS：开源还是虚晃一枪？**：成员们讨论了 **Sesame AI 的对话式语音生成模型 (CSM)**，一位成员称赞其具有逼真的特质，包括*呼吸声*和*情感语调*，并链接到了 [demo](https://www.sesame.com)。
   - 其他人对其*开源*说法表示怀疑，指出其 [GitHub 仓库](https://github.com/SesameAILabs) 缺乏代码提交（commits）。
- **QwQ 模型模板问题获得补丁**：用户报告了在 LM Studio 中运行 **QwQ 模型** 时遇到的问题，具体表现为 *Junja 模板* 出现 *OpenSquareBracket !== CloseStatement* 错误。
   - 一位成员分享了来自 [GitHub issue](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624) 的潜在修复方案，涉及调整模型提示词参数，并确认该方案解决了他们的问题，但其他人仍认为效果不佳。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/puppy-gif-18530240">Puppy GIF - Puppy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.sesame.com">Sesame</a>：我们相信计算机将变得栩栩如生的未来。它们可以像我们彼此交流一样，与我们一起观察、聆听和协作。带着这个愿景，我们正在设计一种新型计算机。</li><li><a href="https://github.com/SesameAILabs">SesameAILabs</a>：SesameAILabs 拥有 8 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issu">lmstudio 中 qwq-32b 模型的问题 · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？例如：LM Studio 0.3.11。哪个操作系统？Mac。Bug 是什么？在使用 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa..."</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">lmstudio 中 qwq-32b 模型的问题 · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？例如：LM Studio 0.3.11。哪个操作系统？Mac。Bug 是什么？在使用 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa..."</li><li><a href="https://tenor.com/view/ibm-card-reader-card-reader-ibm-utility-bill-vintage-computer-gif-15507881284984357200">IBM 读卡器公用事业账单 GIF - IBM CARD READER CARD READER IBM - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://photos.app.goo.gl/MDNqL1c7d289oHEs7">Brian Makin 的新视频</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346833939288953015)** (134 条消息🔥🔥): 

> `M3 Ultra vs M4 Max, AMD RX 9070 XT GPU, DeepSeek R1, SGI 机器, 本地 LLMs` 


- **Apple 发布搭载 M3 Ultra 和 M4 Max 的 Mac Studio**：Apple 宣布了新款 [Mac Studio](https://www.apple.com/uk/mac-studio/)，由 **M3 Ultra**（最高支持 512GB RAM）和 **M4 Max**（最高支持 128GB）驱动。
   - 一位成员在看到规格后感叹 *这就是为什么我不买 Mac...*。
- **AMD RX 9070 XT GPU 基准测试**：一段关于 **AMD Radeon RX 9070 XT** GPU 的 [YouTube 视频](https://www.youtube.com/watch?v=yP0axVHdP-U) 评测显示，它在光栅化性能上与 **Nvidia RTX 5070 Ti** 互有胜负，但 **Nvidia** 在光线追踪方面保持领先。
   - RX 9070 XT 的性能有时能达到 **Nvidia 4080 Super** 的约 95%，而价格仅为 5070 Ti 建议零售价（750 美元）的 80%。
- **512GB 运行 DeepSeek R1 够吗？**：成员们讨论了 **512GB** 统一内存是否足以运行完整的 **DeepSeek R1** 模型。
   - 有人提到 *192GB 已经足够运行 UD quants*，并且 *512GB 的统一架构内存速度更接近 VRAM 速度*。
- **回味 SGI 机器的卓越性能**：成员们讨论了 90 年代后期的 **Silicon Graphics (SGI)** 机器，指出其卓越的图形处理能力和共享全局内存架构。
   - 据称 *在 1998 年左右，最快的 PC 显卡每秒大约能处理 60 万个多边形*，而 *我们的 SGI 每秒能处理 3300 万个多边形……而且我们的还不是最高配置*。
- **放弃本地 LLMs**：一位成员表示他们 *对在本地运行 Local LLMs 感到厌倦*，现在使用 **ChatGPT O3** 进行编程。
   - 他们澄清说除了编程之外不使用 **LLMs**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/">Apple 发布新款 Mac Studio，史上最强大的 Mac</a>: Apple 今日宣布推出新款 Mac Studio，这是史上最强大的 Mac，搭载 M4 Max 和全新的 M3 Ultra 芯片。</li><li><a href="https://www.youtube.com/watch?v=yP0axVHdP-U">AMD Radeon RX 9070 XT GPU 评测与基准测试 vs. 5070 Ti, 5070, 7900 XT (Sapphire Pulse)</a>: 赞助商：亚马逊上的 Montech HyperFlow 360 散热器 https://geni.us/dWBIbF6。AMD 的 Radeon RX 9070 XT 和 9070 GPU 将于明天发布。此基准测试和评测...</li><li><a href="https://threadreaderapp.com/thread/1884244369907278106.html">Thread Reader App 上 @carrigmat 的推文串</a>: @carrigmat：在本地运行 Deepseek-R1 的完整硬件 + 软件设置。真实模型，非蒸馏版，以及确保全质量的 Q8 量化。总成本 6,000 美元。包含所有下载和零件链接...</li><li><a href="https://www.apple.com/uk/mac-studio/">Mac Studio</a>: 终极专业台式机。由 M4 Max 和 M3 Ultra 驱动，提供极致性能和广泛的连接性。专为 Apple Intelligence 打造。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1346814420629721128)** (114 条消息🔥🔥): 

> `Richard Sutton, OpenAI agents 定价, QwQ-32B 模型, Boston Dynamics vs Unitree, 对抗性 Machine Learning`

- **Sutton 对安全性的看法**：一位成员提到，图灵奖得主 Richard Sutton 在最近的一次[采访](https://youtu.be/9_PepvnqIfU?si=sMB90T8__WA19Qrt)中表示，*安全性是假新闻*。
   - 另一位成员评论道：*Rich 在道德上有点可疑，即使他的产出惊人，我也不会听从他的研究建议*。
- **OpenAI 计划为 PhD 级别的 Agent 定价**：据 [The Information](https://www.theinformation.com/articles/openai-plots-charging-20-000-a-month-for-phd-level-agents) 报道，OpenAI 据传计划为未来的 AI Agent 每月收取 **2,000 至 20,000 美元**，这些 Agent 旨在完成自动化编程和 PhD 级别研究等任务。
   - OpenAI 的投资者软银（SoftBank）已承诺仅今年就在 OpenAI 的 Agent 产品上投入 **30 亿美元**，可能会购买约 12,500 个每月 2 万美元的 Agent。
- **阿里巴巴 Qwen 发布 QwQ-32B 模型**：[阿里巴巴 Qwen 发布了 QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一款仅有 **320 亿参数**的新型推理模型，声称其性能可与 DeepSeek-R1 等模型媲美。
   - 该模型是 RL 训练和后训练的产物，显著提升了在数学和编程方面的表现。
- **波士顿动力在人形机器人上失手**：成员们将 [波士顿动力（Boston Dynamics）的 Atlas](https://fxtwitter.com/BostonDynamics/status/1897298172210225280) 机器人与 [宇树科技（Unitree）的人形机器人](https://fxtwitter.com/UnitreeRobotics/status/1896859430517629292) 进行了对比，并给出了负面评价，称 *波士顿动力这种代际级的失误让人不忍直视*。
- **DeepMind 遭遇人才流向 Anthropic 的出走潮**：Nicholas Carlini 宣布他在工作七年后将离开 Google DeepMind 加入 Anthropic。根据他的[博客](https://nicholas.carlini.com/writing/2025/career-update.html)，他表示他在对抗性机器学习（adversarial machine learning）方面的研究在 DeepMind 不再获得支持。
   - 一位成员指出 *GDM 最近流失了这么多重要人物*，而其他人则表示 *Anthropic 的“天命”正在积聚*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅拥有 320 亿参数，却能与顶尖推理模型（如 DeepSeek-R1）相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://ghuntley.com/tradecraft/">是的，Claude Code 可以反编译自身。这是源代码。</a>：这些 LLM 在去混淆、转译和结构到结构的转换方面表现得惊人地出色。我在圣诞节前后发现了这一点，当时我要求一个 LLM 为我编写一个 Haskell 音频库...</li><li><a href="https://nicholas.carlini.com/writing/2025/career-update.html">
      职业更新：Google DeepMind -> Anthropic
    </a>：未找到描述</li><li><a href="https://x.com/cherry_cc12/status/1897366964080926902">来自 Chen Cheng (@cherry_cc12) 的推文</a>：谁将成为 QwQ 家族的下一个成员？引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅拥有 320 亿参数，却能与顶尖推理模型相媲美...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">来自 Qwen (@Alibaba_Qwen) 的推文</a>：Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B。这就是你在 Qwen Chat 上使用这款新模型的方式！引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅拥有 320 亿参数...</li><li><a href="https://x.com/steph_palazzolo/status/1897309493744267314">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：新消息 w/ @coryweinberg：OpenAI 正在加倍投入其应用业务。高管们已与投资者讨论了未来将推出的三类 Agent，价格从每月 2,000 美元到 20,000 美元不等，用于执行诸如...</li><li><a href="https://fxtwitter.com/jsuarez5341/status/1897356500131336208">来自 Joseph Suarez (e/🐡) (@jsuarez5341) 的推文</a>：我们通过在线 RL 击败了《宝可梦 红》！未来几天将在此发布详细信息。由 @dsrubinstein 领导。关注他、我、@DanAdvantage、@kywch500、@computerender 获取更多信息！引用 drubinstein (@dsrubinstein...</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅拥有 320 亿参数，却能与顶尖推理模型（如 DeepSeek-R1）相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://www.youtube.com/watch?app=desktop&v=75jr5E4OzEE&t=431s">动态深度学习 | Richard Sutton</a>：ICARL 研讨会系列 - 2024 冬季。动态深度学习，Richard Sutton 的研讨会 —————————————————— 摘要：尽管取得了巨大成功，当前的深度学习方法...</li><li><a href="https://fxtwitter.com/BostonDynamics/status/1897298172210225280">来自 Boston Dynamics (@BostonDynamics) 的推文</a>：我们正在将 Atlas 设计成无所不能，但我们是一步一个脚印地实现这一目标。看看我们为什么从零件排序开始，我们如何解决难题，以及我们如何交付人形机器人...</li><li><a href="https://fxtwitter.com/UnitreeRobotics/status/1896859430517629292">来自 Unitree (@UnitreeRobotics) 的推文</a>：功夫机器人比赛😘 720° 旋风踢 - 听听这撞击声！功夫机器人原生演示。（无加速）（请勿模仿，请与机器保持安全距离）#Unitree #Kungfu #EmbodiedAI #SpringFestivalGal...</li><li><a href="https://x.com/btibor91/status/1897312899124891761?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：据 The Information 报道，OpenAI 计划为用于高级研究的高级 AI Agent 每月收取高达 20,000 美元的费用，目标是让这些 Agent 长期产生约 20%-25% 的收入...</li><li><a href="https://youtu.be/9_PepvnqIfU?si=sMB90T8__WA19Qrt">图灵奖得主 Richard S. Sutton 与 Cam Linke 对话 | 科学界没有权威</a>：“科学界没有权威，”图灵奖得主 Richard S. Sutton 说道。在这场独家对话中，Amii 首席科学顾问 Richard...
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346813647544123477)** (18 messages🔥): 

> `LLMs playing Diplomacy, GPT-4.5 greentext autocompleter, Mafia game playing LLMs, Post training as a service startups` 


- **LLMs 在 Diplomacy 中协商统治世界！**：一位成员分享了一个让 **LLMs** 玩 [Diplomacy](https://x.com/sam_paech/status/1897078633015206172) 的框架。这是一款具有重度谈判元素的复杂棋盘游戏，非常适合实验 **game theory**（博弈论）并测试说服力！
- **GPT-4.5 重新燃起对 Greentext 的痴迷**：一位成员链接到一条对该模型发布表示难以置信的推文，引发了关于寻找两年前丢失的 **big-model-smell greentext autocompleter**（具有大模型气息的 greentext 自动补全器）的讨论 ([推文链接](https://x.com/adonis_singh/status/1896679334200611312))。
   - 另一位成员反驳称，其他模型也可以生成同样好甚至更好的 greentext，并推荐了 **V3** 或旧的基座模型。
- **LLMs 在在线 Mafia 游戏中策划阴谋！**：一位成员分享了一个网站链接 ([mafia.opennumbers.xyz](https://mafia.opennumbers.xyz/))，展示了 **LLMs 互相玩 Mafia 游戏**，并分享了胜率等 **model statistics**（模型统计数据）。
- **后训练初创公司引发好奇！**：一位成员询问是否有潜水者在 **post training as a service startups**（后训练即服务初创公司）工作，出于工作原因好奇“一键训练”的难度有多大。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sam_paech/status/1897078633015206172">Sam Paech (@sam_paech) 的推文</a>: 我制作了一个让 LLMs 互相玩 Diplomacy 的框架。Diplomacy 是一款具有重度谈判元素的复杂棋盘游戏。非常适合实验博弈论和测试说服力！它...</li><li><a href="https://x.com/adonis_singh/status/1896679334200611312">adi (@adonis_singh) 的推文</a>: 我无法想象他们真的发布了这个模型 😭</li><li><a href="https://mafia.opennumbers.xyz/">LLM Mafia 游戏竞赛</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346968004331573309)** (2 messages): 

> `Schmidhuber Congratulates Sutton and Barto, Turing Award, Cult leader game` 


- **Schmidhuber 祝贺 Sutton 和 Barto 获得图灵奖**：Jürgen Schmidhuber 在一则[帖子](https://x.com/SchmidhuberAI/status/1897406236896977388)中祝贺 Richard S. Sutton 和 Andy Barto 获得 **Turing Award**（图灵奖）。
   - 该消息非常简短，仅包含文字：*Cult leader game recognizes cult leader game*（邪教领袖游戏认可邪教领袖游戏）。
- **邪教领袖游戏认可邪教领袖游戏**：Schmidhuber 给 Sutton 和 Barto 的图灵奖贺信中包含了这句神秘的陈述：*Cult leader game recognizes cult leader game*。
   - 这种看似带有自我意识的评论在追随者中引发了一些讨论。



**提到的链接**: <a href="https://x.com/SchmidhuberAI/status/1897406236896977388">Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>: 祝贺 @RichardSSutton 和 Andy Barto 获得图灵奖！

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346938335108661289)** (3 messages): 

> `强化学习击败宝可梦，DeepSeek MLA 性能挑战，ThunderMLA 融合 megakernel` 


- **RL 以雷鸣般的掌声击败宝可梦**：开发了一个强化学习系统，使用参数量低于 **10M** 的策略、**PPO** 以及新颖的技术击败了 **Pokémon Red**，详见 [博客文章](https://x.com/dsrubinstein/status/1897351145485648309?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)。
   - 该系统成功通关游戏，展示了 **RL** 在解决复杂任务中的复兴。
- **MLA 面临深度性能挑战**：受 **DeepSeek MLA** 热潮的启发，人们正在探索新的调度器来管理变长序列，这在处理来自不同用户的请求的 **LLM inference** 中非常常见。
   - 重点在于解决与大语言模型推理相关的性能挑战。
- **ThunderMLA Megakernel 融合性能**：**ThunderMLA** 作为一个用于 decode 的完全融合 *megakernel* 被推出，以应对 **LLM** 推理性能挑战。它声称在各种工作负载下比 DeepSeek 的 **FlashMLA** 快 **20-35%**，[代码已在 GitHub 上发布](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu)。
   - 简单的调度技巧显著提升了性能，初始版本专注于 attention 解码。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla">ThunderMLA: FlashMLA, Faster and Fused-er!</a>：未找到描述</li><li><a href="https://x.com/dsrubinstein/status/1897351145485648309?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">drubinstein (@dsrubinstein) 的推文</a>：很高兴终于能分享我们在开发击败 Pokémon Red 的强化学习系统方面的进展。我们的系统使用参数量低于 10M 的策略、PPO 和一些...成功通关了游戏。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1346996584667152386)** (10 messages🔥): 

> `RLHF 书籍，系列讲座` 


- **RLHF 书籍 PDF 发布！**：一位成员分享了 [RLHF 书籍 PDF](https://rlhfbook.com/book.pdf)，供有需要的人使用。
   - 该成员表示：*如果大家真的在用，欢迎提供反馈*。
- **Nathan 筹备 RLHF 系列讲座**：Nathan 暂定在夏季开展一个系列讲座，**每章对应一个视频**。
   - 他提到，一旦预订按钮上线，他就必须让营销引擎全速运转起来。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1346848634926272582)** (9 messages🔥): 

> `Stargate 项目，数据保护，OpenAI 编程 Agent` 


- **Stargate 项目通过广告获利**：**Stargate 项目**现在通过*完全公正且不显眼的广告*获得资金。
- **数据囤积者守护黄金**：根据 Ben Thompson 的观点，随着模型变得越来越强大，企业需要**对其内容设置门槛**以避免倒闭。
   - 报纸已经输掉了这场战斗，必须接受 Sam 提供的任何交易，因此像 **YouTube** 和 **GitHub** 这样宝贵的数据宝库必须不惜一切代价予以保护。
- **Microsoft 封锁 OpenAI Agent？**：有人建议，如果 **Microsoft** 封锁了每月 **2 万美元的 OpenAI 编程 Agent**，其效用将会降低。
   - 搜索软化（Search softening）有助于拥有数据的小型公司避免严苛的局面。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1346825050518720553)** (48 条消息🔥): 

> `Touhou 训练模型，Unified Memory 讨论，Thunderbolt 5 优势，Raspberry Pi 集群` 


- ****Touhou AI 模型梦想成真？****：一位成员想要训练一个 AI 模型来玩 **Touhou**（东方 Project），利用 **RL**（强化学习）并将游戏分数作为奖励。
   - 他们指出，现在通过 **RL** 实现这一点变得更加容易，可能会使用 **Starcraft gym** 和 **Minetest gym** 等模拟器进行学习。
- ****Unified Memory：游戏规则改变者？****：**M3 Ultra** 的发布引发了关于 **Unified Memory**（统一内存）的讨论，成员们想知道 **CPU** 和 **GPU** 寻址同一内存系统的性能特征。
   - 有人建议查看 *metal* 频道，了解更多关于这些设备的 **Metal** 编程的具体讨论。
- ****Thunderbolt 5 加速游戏进程****：成员们对 **Thunderbolt 5** 感到兴奋，它将使 **Mac Minis/Studios** 之间的分布式推理/训练变得更加可行。
   - 其单向速度（120gb/s）似乎比 **RTX 3090 SLI bridge**（112.5gb/s）还要快。
- ****Raspberry Pi 集群：仍然可行吗？****：讨论涉及使用 **Raspberry Pis** 或 **Jetson Nanos** 集群来处理可并行化的任务，例如大尺寸图像生成。
   - 一位成员分享了 [Turing Pi 2.5](https://turingpi.com/product/turing-pi-2-5/) 的链接，这是一个 4 节点的 mini ITX 集群板，可以运行 **Raspberry Pi CM4** 或 **Nvidia Jetson** 计算模块的任意组合。



**提到的链接**：<a href="https://turingpi.com/product/turing-pi-2-5/">获取 Turing Pi 2，mini ITX 集群板</a>：Turing Pi 2.5 是一款内置以太网交换机的 4 节点 mini ITX 集群板，可运行 Turing RK1、Raspberry Pi CM4 或 Nvidia Jetson 计算模块。

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1346941022755881000)** (4 条消息): 

> `Triton gather 操作，Triton 中的 PagedAttention` 


- **Triton 中的 Gather 操作指南**：一位成员询问如何在 **Triton** 中执行 *gather 操作* 并遇到了 `AttributeError`。
   - 另一位成员建议从 master 分支构建 **Triton** 并卸载 **PyTorch** 提供的版本，并链接到了一个[相关的 GitHub issue](https://github.com/triton-lang/triton/issues/5826)。
- **寻求 PagedAttention 重现资源**：一位成员请求关于如何在 **Triton** 中重现 **PagedAttention** 的资源。



**提到的链接**：<a href="https://github.com/triton-lang/triton/issues/5826">无法调用 tl.gather · Issue #5826 · triton-lang/triton</a>：描述 Bug：当我运行以下代码时，出现异常：AttributeError: module 'triton.language' has no attribute 'gather'。代码：import triton.language as tl; tl.gather...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1346800993525960704)** (13 条消息🔥): 

> `编译器优化，CUDA OpenGL 互操作，cudaGraphicsGLRegisterImage 失败` 


- **编译器优化掉未使用的内存写入**：一位用户发现，当写入的数据从未被读取时，**CUDA 编译器会优化掉内存写入**，导致没有报告任何错误。
   - 另一位用户确认，*如果你添加一个对该数组的读取操作，编译器就会报错*。
- ****CUDA OpenGL 互操作在笔记本电脑上发生段错误 (segfault)****：一位用户在笔记本电脑上调用 `cudaGraphicsMapResources` 时，其 **CUDA OpenGL 互操作**代码发生了 **segfault**，而同样的代码在台式机上运行正常。
   - `cudaGraphicsRegisterImage` 调用返回了 `cudaErrorUnknown`，尽管两台机器的 **CUDA** 和驱动程序版本相同，这让用户感到困惑。
- **OpenGL 未使用 GPU 导致 CUDA 失败**：一位用户找到了问题的解决方案：*OpenGL 没有使用我的 GPU*。
   - 在确保 OpenGL 使用 GPU 后，CUDA OpenGL 互操作问题得到了解决，因为 `cudaGraphicsGLRegisterImage` 需要 OpenGL 在独立 GPU 上运行。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1346900912525934773)** (4 messages): 

> `Torch C++ Interface Library, Extending OffloadPolicy, use_reentrant in Checkpoint` 


- **Torch C++ 方法缺少 Schema？**：一位成员询问为什么 **Torch C++ 接口库**（类似于 pybind11）中的方法不能像函数那样拥有 schema。
   - 他们进一步询问了关于**扩展 OffloadPolicy** 的提案，包括 PR 是否会被接受以及应该咨询谁。
- **PyTorch Checkpointing 中暴露的 `use_reentrant` 参数**：一位成员询问了 [PyTorch checkpointing](https://pytorch.org/docs/stable/checkpoint.html) 功能中 `use_reentrant` 参数的作用。
   - 解释明确了 checkpointing 会在反向传播期间重新运行前向传递片段，这会影响 RNG 状态；设置 `preserve_rng_state=False` 可以在每个 checkpoint 期间省略暂存和恢复 **RNG 状态**的操作。



**提及的链接**：<a href="https://pytorch.org/docs/stable/checkpoint.html">torch.utils.checkpoint &mdash; PyTorch 2.6 documentation</a>：未找到描述

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1346837200326889603)** (12 messages🔥): 

> `SSH Pain Points, Nitrokey, SoloKey, Yubikey, PC under the sink` 


- **在 RTX3050 上使用 Blackwell GPU？**：一位成员表达了对 SSH 的挫败感，并考虑购买一块 **Blackwell GPU** 与他们的 **RTX 3050** 和 **GFX90c** 配置组合进行实验。
- **Nitrokey, SoloKey, Yubikey 提高账户安全性**：成员们讨论了使用 **Nitrokey**、**SoloKey** 或 **Yubikey** 来增强安全性，并指出这些选项相对便宜且易于在多个账户中使用。
   - 他们还提到使用 [mutagen.io](https://mutagen.io/) 在笔记本电脑和服务器之间同步文件，因为他们不喜欢 VS Code。
- **水槽下的 PC**：一位成员分享了一个趣闻，由于空间限制且靠近电源插座，他把一台 PC 放在了厨房水槽下面。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1346903165072576604)** (3 messages): 

> `Tenstorrent, LlamaIndex, Koyeb, AI Infrastructure, Next-Gen Hardware` 


- **Tenstorrent, LlamaIndex 和 Koyeb 在旧金山举办见面会**：**Tenstorrent** 和 **LLamaIndex** 团队今晚在旧金山市中心举办一场关于 **AI 基础设施**和**下一代硬件**的小型见面会 ([lu.ma/ruzyccwp](https://lu.ma/ruzyccwp))。
   - 这次见面会标志着 **Tenstorrent** 和 **Koyeb** 合作的开始，旨在提供比传统 GPU 更优的性价比。
- **主办方详情**：**Tenstorrent** 被描述为一家为 AI 构建计算机的下一代计算公司，**Koyeb** 是一个用于部署和扩展 AI 工作负载的前沿 Serverless 平台，而 **LlamaIndex** 提供了一个灵活的框架，用于构建连接到企业数据的 LLM 知识助手。
   - 议程包括 **5:30 PM** 开门以及 **6:00 PM** 开始的其他活动。



**提及的链接**：<a href="https://lu.ma/ruzyccwp">Next-Gen AI Infra with Tenstorrent &amp; Koyeb @LlamaIndex · Luma</a>：加入我们，共同开启 Tenstorrent、Koyeb 与来自 LlamaIndex 的朋友们之间的开创性合作。这次见面会是……

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1346881365265612883)** (1 messages): 

> `Reshaping vs Permuting, Matrix Transformations` 


- **Reshaping 与 Permuting 矩阵的区别**：*Reshaping* 矩阵不会改变行优先（row-major）顺序下的元素顺序，而 *permuting*（例如**转置**）则会改变元素顺序。
   - 对于一个 **M x N** 矩阵 reshape 为 **N x M**，按行优先顺序读取元素保持一致，但转置矩阵会改变这个顺序。
- **理解矩阵变换**：Reshaping 可以被认为是重新组织矩阵，而不改变以特定方式读取时的底层元素顺序。
   - 另一方面，Permuting 会主动重新排列元素的位置，导致以相同顺序读取时产生不同的序列。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1346862321326362636)** (6 messages): 

> `ROCm 上的 RGP，ATT 插件` 


- **寻求 ROCm 的 RGP Instruction Timing 替代方案**：一名成员正在为 Linux 上的 **ROCm** 寻找类似于 **RGP 中 Instruction Timing 标签页**的工具。
   - 不幸的是，使用 **RGP** 仅限 **Windows**，建议在 **Linux** 上使用 **PAL backend** 编译 **rocCLR** 作为替代方案，但其功能无法保证。
- **ATT 插件无法正常工作**：一名成员询问了关于 **rocprofilerv2** 的 **ATT 插件**，根据文档，它应该提供每条指令的延迟（latency per instruction）。
   - 然而，原帖作者和另一名成员都确认他们**无法让 ATT 插件正常工作**。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1346801537447366717)** (17 messages🔥): 

> `CUDA 中的共享内存分配、Python Linting 解决方法、CUDA 兼容性问题、TileLang CUDA 12.4/12.6 Bug、微信群邀请` 


- **发现共享内存计算方法**：一位用户参考了 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)（表 24）来计算整体及每个线程块（thread block）的可用共享内存，并将其与 `T.alloc_shared` 调用和数据类型大小联系起来。
   - 此计算的目的是确保程序请求的共享内存量不超过 CUDA 设备的限制。
- **Python Linting 警告再次出现**：一位用户确认，警告主要是由于其 Pythonic DSL 中的 **Python linting** 引起的，但他们尚未找到绕过该 lint 问题的简单方法。
   - 另一位用户选择暂时忽略这些警告，同时寻找合适的解决方案。
- **CUDA 12.4 失败令人沮丧**：一位用户报告称，在 **RTX 4070 笔记本电脑**上运行在 RTX 4090 上使用相同 nightly build ([cu121](https://tile-ai.github.io/whl/nightly/cu121/)) 能够正常工作的代码时出现失败。
   - 尽管该包指示兼容 CUDA >= 11.0，但降级到 **CUDA 12.1** 解决了该问题。
- **TileLang 的 Bug 追踪器诞生**：一位用户在 GitHub 上创建了一个 issue ([tile-ai/tilelang/issues/149](https://github.com/tile-ai/tilelang/issues/149))，报告在 **CUDA 12.4/12.6** 上执行 **matmul** 时出现元素不匹配的情况。
   - 该代码在 **CUDA 12.1** 上运行正常，但在新版本中引发了关于 tensor-like 差异的 `AssertionError`，促使维护者调查兼容性问题。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang/issues/149">Mismatched elements when performing matmul on CUDA 12.4/12.6 · Issue #149 · tile-ai/tilelang</a>：描述 Bug：我运行了下面的简单 matmul 代码，得到了错误 AssertionError: Tensor-likes are not close! 该代码在 CUDA 12.1 上运行正常，但在 CUDA 12.4/12.6 上不行。不匹配的数量...

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1347025757897232435)** (1 messages): 

> `M3 Ultra, Unified Memory` 


- **M3 Ultra 发布引发创意想法**：成员们分享了对 **M3 Ultra** 发布会的看法以及**统一内存（Unified Memory）**可能的创意应用。
   - 讨论发生在 [此 Discord 频道](https://discord.com/channels/1189498204333543425/1189498205101109300/1347019708586655757) 中。
- **统一内存应用**：**M3 Ultra** 中**统一内存**的潜力是一个关键关注点。
   - 讨论了创意用途和益处，尽管提供的上下文中没有详细说明具体的应用。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1346817771752656926)** (12 messages🔥): 

> `ARC AGI, Lossless Information Compression, QwQ-32B, RL Scaling` 


- ****ARC AGI** 是下一个目标？**: 成员们计划在 **Isaac Liao** 和 **Albert Gu** 关于无损信息压缩是否能产生智能行为的 [ARC AGI](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) 初步研究之后，参与 **ARC AGI-2**。
   - 该文章提供的证据表明，在 *in-context learning* 过程中的无损压缩可以实现 **AGI**。
- ****QwQ-32B** 挑战推理模型**: **Alibaba** 发布了 [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b)，这是一个仅有 **320 亿参数** 的新型推理模型，可与 **DeepSeek-R1** 等顶尖模型媲美。
   - 提供的链接包括 [HF page](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、[Demo](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 和 [Qwen Chat](https://chat.qwen.ai)。
- ****RL** Scaling 取得显著成果**: 扩展 **RL** 训练可以持续提升性能，特别是在 **math** 和 **coding** 领域，**Qwen2.5-32B** 在与更大型的 **MoE** 模型竞争中取得了极具竞争力的结果。
   - 讨论强调，持续扩展 **RL** 规模可以帮助中型模型与巨型模型竞争。
- ****Reasoning Gym** 目标锁定 100 个数据集**: **Reasoning Gym** 目前已有 **97 个数据集**，并正在征集另外 **3 个** 提案以达到总数 **100** 个。
   - 一位成员提到有两个尚未添加的数据集。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: 未找到描述</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 今天，我们发布了 QwQ-32B，这是我们仅有 320 亿参数的新型推理模型，可与 DeepSeek-R1 等顶尖推理模型媲美。博客: https://qwenlm.github.io/blog/qwq-32b HF: https://hu...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

leoneo221: 好久没上线，竟然多了一个中文channel
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1346857475802009693)** (11 messages🔥): 

> `Modal Runners, Leaderboard Submissions, GPU usage` 


- **Modal Runners 成功运行**: 使用 **Modal runners** 的测试提交在各种排行榜和 GPU（包括 **A100** 和 **T4**）上均获成功。
   - 根据 **Cluster-Bot** 的报告，向 `histogram` 和 `grayscale` 排行榜的提交已成功。
- **排行榜名称导致小问题**: **Cluster-Bot** 报告称，*命令中指定的排行榜名称与提交脚本头文件中的名称不匹配*。
   - 尽管存在差异，提交仍已发送至 `histogram` 和 `grayscale` 排行榜。
- **T4 GPUs 占据主导**: 在 `grayscale` 排行榜上，使用 **T4 GPUs** 的多次测试提交均获成功。
   - ID 为 **1594, 1595, 1596, 1598, 1599, 1600 和 1601** 的提交在 `grayscale` 排行榜中全部使用了 **T4 GPUs**。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1347018948784291902)** (1 messages): 

> `AI model settings, Claude 3.7 Sonnet, Auto settings improvements` 


- **设置重定位带来极致简洁**: AI 模型设置正被合并到输入框旁的一个**便捷位置**，首先在网页端推出。
   - 此项更改旨在使自定义设置更快速、更直观；在过渡期间，占位符将引导用户前往新位置。
- **Claude 的能力吸引客户**: 作为此次更新的一部分，**Claude 3.7 Sonnet** 将面向 **Pro users** 开放。
   - 团队希望让 **"Auto"** 设置变得更加强大，使用户无需手动选择模型。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1346799632742879387)** (107 条消息🔥🔥): 

> `Perplexity 自动模型选择、图片来源问题、Deepseek r2 发布、Claude Sonnet 3.7 挫败感、Google Search AI 模式` 


- **Perplexity 的 AUTO 模型：是独立实体还是其他？**：用户讨论 Perplexity 中的 “Auto” 是指一个独立的 AI 模型，还是一个从现有模型中自动选择的功能，有人建议它会选择设置中选定的模型。
   - 还有建议称 Auto 模型默认运行 **Sonar**，除非使用 “rewrite” 功能。
- **烦人的图片来源 Bug**：一位用户报告称，用作来源的图片在删除后仍会不断出现在提示词中，称这是一个需要修复的*烦人* Bug。
   - 除了手动删除外，没有提供其他解决方法。
- **Deepseek r2：社区翘首以盼**：成员们正热切期待 **Deepseek r2** 的发布，希望它能显著降低成本并造福 AI 社区。
   - 对服务器问题的担忧依然存在，有人建议使用更安全的代理网站来解决这些问题。
- **Google Search 拥抱 AI 模式**：Google 宣布了 **AI Mode for Search**，提供对话式体验并支持复杂查询，目前作为一项可选体验面向部分 Google One AI Premium 订阅者开放（参见 [AndroidAuthority](https://www.androidauthority.com/google-search-ai-mode-experiment-3532243/)）。
   - 一位用户评论道：*只是 Perplexity 不再特别了。*
- **Claude Sonnet 3.7 的忧郁**：一位用户对 Perplexity 实现的 **Claude Sonnet 3.7** 表示不满，认为其结果不如直接通过 Anthropic 使用，并批评了激活它所需的繁琐步骤。
   - 他们还注意到 **3.7** 在一个简单的 JSON 文件中产生了幻觉错误，质疑该模型声称的改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.croxyproxy.com/">未找到标题</a>：未找到描述</li><li><a href="https://www.androidauthority.com/google-search-ai-mode-experiment-3532243/">Google 通过 AI 模式增强搜索以回答复杂问题</a>：备受期待的 Google Search AI 模式终于面世，它可以更有效地回答复杂的、多部分的问题。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346879637015232552)** (6 条消息): 

> `Microsoft AI 健康助手、Python 学习路线图、Mac M3、OpenAI Agent、SQLI 防护` 


- **Microsoft 首次推出 AI 健康助手**：Microsoft 在[这里](https://www.perplexity.ai/page/microsoft-debuts-ai-health-ass-38RGe6B5SVq1nX5OM09k5w3blessed.)首次推出了新的 **AI Health Assistant**。
- **学习 Python 的路线图**：可以在[这里](https://www.perplexity.ai/search/the-best-roadmap-to-learn-pyth-BjtmcOKMRM6CX.SyJjMcfw)找到学习 **Python** 的路线图。
- **新款 Mac M3**：关于**新款 Mac M3** 的讨论可以在[这里](https://www.perplexity.ai/search/it-s-said-that-the-new-mac-m3-NhwnxpNtRv.G9EmA19._RQ#1)找到。
- **OpenAI 的 20000 AI Agent**：关于 **OpenAI's 20000 AI Agent** 的页面位于[这里](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ)。
- **防御 SQLI**：关于如何**防御 SQLI** 的信息可以在[这里](https://www.perplexity.ai/search/how-to-protect-against-sqli-pCVG1m1YTWSBIlyKyatJnA)找到。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346837917951332453)** (4 条消息): 

> `API focus 设置、Sonar Pro 模型问题、搜索成本定价` 


- **API Focus 设置难题**：一位用户询问了如何将 **API 聚焦于特定主题**（如学术或社区相关内容）的方法。
   - 消息中未提供解决方案。
- **Sonar Pro 在时效性和有效性方面挣扎**：一位用户报告称，尽管将 *search_recency_filter* 设置为 “month”，**Sonar Pro 模型**仍返回过时的信息和错误的链接。
   - 用户怀疑自己是否误用了 API。
- **Sonar Pro 令人困惑的引用编号**：一位用户报告称 **Sonar Pro** 中的**引用编号**令人困惑，因为回复从 1 开始，但来源列表从 0 开始。
   - 消息中未提供解决方案。
- **API 搜索成本定价之谜**：一位用户对 **API 不提供搜索成本信息**表示沮丧，这使得准确跟踪支出变得不可能。
   - 他们哀叹无法跟踪自己的 API 支出，因为 API 没有告知使用了多少次搜索，并配上了一个大哭的表情。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1346804926214574092)** (47 条消息🔥): 

> `Local Model Usage, Llama 3.1, Mistral small instruct quantized, CoreWeave IPO, HF Inference Credits` 


- **在本地运行模型**：一位成员建议，通过点击 *"use this model"* 选项并选择服务提供商，用户可以[轻松地在本地运行模型](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions)。
   - 他们还分享了你可以通过 Discussion 板块直接联系 **Meta**，或者使用 **unsloth**。
- **关于最佳本地可运行文本生成模型的讨论**：一位成员询问在使用 4080 显卡时，哪种本地可运行的文本生成模型效果最好，并对 **Llama 3.1** 表示好奇。
   - 另一位成员推荐了 **Mistral small instruct quantized**，指出它拥有 **24B parameters**，性能可与 **llama 3.3 70B** 媲美。
- **CoreWeave 提交 IPO 申请**：基于云端的 **Nvidia** 处理器提供商 **CoreWeave** [提交了其 IPO 招股说明书](https://www.sec.gov/Archives/edgar/data/1769628/000119312525044231/d899798ds1.htm)，报告 2024 年营收增长 **700%** 达到 **19.2 亿美元**，尽管净亏损为 **8.634 亿美元**。
   - 约 **77%** 的营收来自两家客户，主要是 **Microsoft**，且该公司持有超过 **150 亿美元** 的未履行合同。
- **关于推理额度 (Inference Credits)**：一位拥有 **HF pro plan** 的用户对有限的推理额度（2 美元）表示担忧，并寻求其他提供商以增加使用量。
   - 另一位成员确认存在多个针对各种模型的第三方提供商，并提到他们也收购了 **W&B**。
- **基础异常检测**：一位成员推荐了集成 [Pycaret](https://pycaret.gitbook.io/docs/) 的 [xircuits.io](https://xircuits.io/docs/component-library/library-guides/pycaret/Pycaretanomaly) 用于基础异常检测，强调了其在无需特定训练即可识别问题方面的易用性。
   - 该链接指向一个用于基础异常检测 Pycaret 应用的 AutoMLBasicAnomalyDetection.xircuit。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/models?sort=modified&search=gguf)">Models - Hugging Face</a>：未找到描述</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm">AI cloud provider CoreWeave files for IPO</a>：CoreWeave 约三分之二的营收依赖 Microsoft，目前正准备上市。</li><li><a href="https://xircuits.io/docs/component-library/library-guides/pycaret/Pycaretanomaly">Anomaly Detection | Xircuits</a>：在开始这些示例之前，请确保在你的工作环境中安装了 Pycaret>=2.2。你也可以使用 pip install pycaret==2.3.8 进行安装。</li><li><a href="https://www.cnbc.com/2025/03/03/ai-cloud-provider-coreweave-files-for-ipo.html?utm_source=join1440&utm_medium=email&utm_placement=newsletter&user_id=66c4c765600ae15075a57d0b">AI cloud provider CoreWeave files for IPO</a>：CoreWeave 约三分之二的营收依赖 Microsoft，目前正准备上市。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1346815901026095136)** (6 条消息): 

> `Kornia Rust Library, Google Summer of Code 2025, Internship postings` 


- **Kornia Rust 库为 Google Summer of Code 2025 招募实习生**：**Kornia Rust library** 正在为 [Google Summer of Code 2025](https://summerofcode.withgoogle.com/programs/2025/organizations/kornia) 开放实习岗位，以改进该库。
   - 项目将主要围绕 **Rust 中的 CV/AI** 展开，欢迎感兴趣的人士查阅文档并就任何问题进行咨询。
- **禁止发布 Discord 服务器邀请；允许发布实习信息**：一位成员根据频道指南提醒，*不允许发布服务器邀请*，参考频道 <#895532661383254098>。
   - 澄清指出，发布 **实习信息是可以的**，但禁止邀请加入其他 Discord 服务器。



**提到的链接**：<a href="https://summerofcode.withgoogle.com/programs/2025/organizations/kornia">Google Summer of Code</a>：Google Summer of Code 是一个全球性计划，旨在吸引更多开发者参与开源软件开发。

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1346918527952490536)** (1 messages): 

> `Flash Attention, Triton, CUDA, GPU Mode` 


- **Umar Jamil 分享他在 GPU Mode 学习 Flash Attention, Triton 和 CUDA 的历程**：[Umar Jamil](https://x.com/hkproj/status/1896113497031000563?s=46) 将于太平洋时间 3 月 8 日星期六中午在 **GPU Mode** 现身，分享他学习 **Flash Attention**、**Triton** 和 **CUDA** 的历程。
   - 这将是一场*与观众的深入对话*，讨论他在旅程中遇到的困难，并分享关于如何自学任何知识的实用技巧。
- **Triton 和 CUDA 讨论**：讨论重点在于掌握高效 GPU 编程的 **Triton** 和 **CUDA** 的实用技巧。
   - 本次会议还将涵盖自学复杂技术主题的策略。



**Link mentioned**: <a href="https://x.com/hkproj/status/1896113497031000563?s=46">Tweet from Umar Jamil (@hkproj)</a>: 我将于 3 月 8 日受 @GPU_MODE 邀请，分享我学习 Flash Attention, Triton 和 CUDA 的历程。这将是一场与观众关于我自身困难的深入对话...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346827449937297448)** (3 messages): 

> `VisionKit, Deepseek-r1, Model Context Protocol (MCP)` 


- **VisionKit 尚未开源！**：该模型使用了 **VisionKit** 但尚未开源，可能在“未来某个阶段”发布。
- **Deepseek-r1 伸出援手**：**Deepseek-r1** 在开发过程中出奇地有用。
   - 一篇 [Medium 文章](https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804) 讨论了构建自定义 **MCP server** 并以 **CookGPT** 为例。



**Link mentioned**: <a href="https://medium.com/data-scientists-from-future/model-context-protocol-custom-mcp-server-b26b32b9d804">Model Context Protocol- Custom MCP Server</a>：在本文中，我们将重点讨论构建自定义 MCP server。如果您需要 MCP 的介绍，请参考我之前的文章……

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347005463908192394)** (1 messages): 

> `DINOv2, fine-tuning, pose estimation, weakly labeled images` 


- **关于用于姿态估计的 DINOv2 Backbone 训练的讨论**：一位成员正在寻求关于使用约 **600k** 张弱标签图像为特定任务[训练或 fine-tuning **DINOv2**](https://github.com/facebookresearch/dinov2) 的建议，最终目标是将其用于 **pose estimation** 和其他复杂任务。
   - 他们正在考虑是从头训练还是进行 fine-tuning，并考虑在 backbone 未冻结的情况下训练分类，但不确定由于标签模糊，学习到的语义是否足够。
- **DINOv2 Fine-Tuning 策略**：讨论围绕是应该从头训练 **DINOv2** backbone 还是针对涉及弱标签图像的特定任务进行 fine-tuning。
   - 用户还在探索在 backbone 未冻结的情况下进行分类训练的选项，但担心由于标签的模糊性，学习到的语义质量。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1346841193958346844)** (3 messages): 

> `Reasoning Course, Smol Course Discovery` 


- **推理课程（Reasoning Course）受到关注**：课程创建者正专注于将 [推理课程材料](https://huggingface.co/reasoning-course) 作为 smol-course 的逻辑进阶。
   - 一位成员询问了关于*推理课程更多单元*的信息。
- **Smol Course 吸引了聊天机器人开发者**：一位成员高兴地发现了 smol-course 并询问该课程是否适合他们。
   - 他们已经在各种应用上构建了一些聊天机器人，包含多达 **5 个基础 tool calls**，其中包括几个使用 **local llm 和 RAG** 的机器人，并希望学习更多关于 **hf ecosystem** 的知识。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1346806081678086225)** (51 条消息🔥): 

> `证书位置、Alfred 示例观点、401 错误、Huggingface 频道、Llama Index 错误` 


- **证书位置不明显！**：用户无法在课程中找到他们的证书，特别是在[此页面](https://huggingface.co/learn/agents-course/unit1/final-quiz#certificate)，并寻求帮助。
   - 一位成员指出，可以在[此数据集](https://huggingface.co/datasets/agents-course/certificates)的 "files" 目录下找到 "certificates"，但其他人反映该位置未显示。
- **Alfred 示例遭到吐槽，随后获得辩护**：一位成员对课程中的 **Alfred 示例** 表示不满，认为它们脱离了真实世界的用例。
   - 另一位成员为这些示例辩护，称它们*完美地解释了在现实生活中为何需要 Agent 以及 Agent 如何运作*。
- **遇到 401 错误，通过复制到 Drive 解决！**：一位成员在 `code_agents.ipynb` 笔记本中遇到了 **401 Client Error**，尽管已成功登录。
   - 该问题通过将笔记本复制到他们的 Google Drive 并从那里启动得以解决。
- **Llama Index 导入失败？找不到 `llama_index.embeddings.huggingface_api`？**：一位成员在运行 **Llama Index** 笔记本时遇到了 `ModuleNotFoundError`，具体是无法导入 `llama_index.embeddings.huggingface_api`。
   - 另一位成员建议运行 `!pip install llama_index.embeddings.huggingface` 并将导入语句更改为 `from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding`。
- **推理使用限制**：一位成员建议使用 [OpenRouter](https://openrouter.ai/) 作为获取免费开源模型的另一种方法。
   - 具体来说，所有以 ":free" 结尾的模型都可以在无需支付积分或订阅费用的情况下使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://steanmcommunnuty.com/10529485">Steam Gift Activation</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certificate">Unit 1 Quiz - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/docs/smolagents/en/reference/models#smolagents.OpenAIServerModel">Models</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz#certif">Unit 1 Quiz - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/agents-course/certificates">agents-course/certificates · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/)">OpenRouter</a>：LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/api/v1',">Discord</a>：未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1346803850161688587)** (73 条消息🔥🔥): 

> `Tool calling, MCP for Reddit, Composio MCP Support, WebMCP, fastmcp` 


- **生产环境的 Evals 经验性地优化 Prompt**：成员们讨论认为，[生产环境的 Evals](https://link.to/production-evals) 是经验性地优化 Prompt 和上下文的必经之路，特别是对于商业或关键任务。
   - 他们指出，即使准确率达到 **95%**，无法识别哪些实例是正确的仍然是一个挑战。
- **Composio 现已支持 MCP 并提供全面的身份验证**：[Composio](https://mcp.composio.dev/) 宣布全面支持 **MCP**，并为集成提供完善的身份验证。
   - 正如其 [公告](https://x.com/composiohq/status/1896968949654495291) 中所强调的，这消除了为 *Linear, Slack, Notion* 和 *Calendly* 等应用设置 **MCP servers** 的需求，提供了托管身份验证并提高了 Tool calling 的准确性。
- **WebMCP 构想引发安全担忧**：讨论了任何网站都可以作为 **MCP server** 的想法，这可能导致任何网站都能访问本地的 **MCP servers**。
   - 这引发了重大的安全担忧，一位成员将其描述为“安全噩梦”，会破坏浏览器沙箱；其他人则反驳说，**CORS** 和跨站配置等保护措施可以降低风险。
- **使用 MCP 构建的 Reddit Agent**：一位成员使用 **MCP** 构建了一个 **Reddit agent** 来获取潜在客户，展示了 **MCP** 在现实任务中的实际应用。
   - 在询问如何连接到 Reddit 后，另一位成员分享了 [Composio 的 Reddit 集成](https://mcp.composio.dev/reddit/wrong-petite-crayon-_q1Vlt) 链接。
- **fastmcp 工具描述选项**：一位成员指出，**fastmcp** 可以使用 docstring 或装饰器 `@mcp.tool(description="My tool description")` 来描述工具。
   - 他们链接了 [python-sdk repo](https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/fastmcp/text_me.py#L49) 和 [base.py](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/fastmcp/tools/base.py#L44) 中的代码示例。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kn22FBsYwu8">Supporting Multi-Agent Use Cases with Anthropic&#39;s Model Context Protocol</a>: 视频中提到：Anthropic MCP 研讨会录音和 swyx 的总结 https://x.com/swyx/status/1896242039614042181?t=6qt4OtebjAeM_BYkt_6QuQ&amp;s=19- Lis...</li><li><a href="https://github.com/ComposioHQ/composio">GitHub - ComposioHQ/composio: Composio equip&#39;s your AI agents &amp; LLMs with 100+ high-quality integrations via function calling</a>: Composio 通过 function calling 为你的 AI Agent 和 LLM 提供 100 多个高质量集成 - ComposioHQ/composio</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/fastmcp/text_me.py#L49">python-sdk/examples/fastmcp/text_me.py at main · modelcontextprotocol/python-sdk</a>: Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk</li><li><a href="https://mcp.composio.dev/">Composio MCP Server</a>: 未找到描述</li><li><a href="https://composio.notion.site/Cursor-MCP-Docs-1adf261a6dfe80b4ba5fe492bf41441c">Your connected workspace for wiki, docs &amp; projects | Notion</a>: 一个将日常工作应用融合在一起的新工具。它是为你和你的团队打造的一体化工作区</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/fastmcp/tools/base.py#L44">python-sdk/src/mcp/server/fastmcp/tools/base.py at main · modelcontextprotocol/python-sdk</a>: Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk</li><li><a href="https://x.com/composiohq/status/1896968949654495291">Tweet from Composio (@composiohq)</a>: 我们很高兴地宣布 Composio 现在全面支持 MCP，并为你的所有集成提供完善的身份验证。你不再需要为设置 MCP server 而苦恼...</li><li><a href="https://github.com/nextapps-de/flexsearch">GitHub - nextapps-de/flexsearch: Next-Generation full text search library for Browser and Node.js</a>: 适用于浏览器和 Node.js 的下一代全文搜索库 - nextapps-de/flexsearch
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1346895258574655589)** (23 条消息🔥): 

> `MCP Server 设置, MCP Token 生成, Blue Yeti 麦克风, Instagram Lead Scraper` 


- ****MCP Server 设置故障已解决！****：一位用户在设置 MCP Server 时遇到了 **401 错误**，但在意识到文档[标注错误](https://fix-the-docs.link)后，通过将启动时生成的 client token 正确地作为环境变量提供给 MCP Server 解决了该问题。
   - 用户澄清说，*你需要将启动时生成的 client token 作为环境变量提供给 MCP Server，然后使用命令行生成其他 token 并粘贴到网站中*。
- ****Token 之舞：本地 vs. 分站点！****：在 MCP Server 设置完成后，用户澄清存在一个**本地 token**，以及用于网站访问的**按站点和按会话生成的 token**。
   - 开发者确认了这一流程，并强调 *它是按会话、按站点的*。
- ****Blue Yeti：麦克风亮相！****：一位用户询问了演示中使用的麦克风，结果发现是服役多年的 [Blue Yeti](https://www.bluemic.com/en-us/products/yeti/)。
   - 开发者予以确认，并补充说音频是 *原始音频——没有经过 EQ、压缩、混响等处理*。
- ****Insta-Lead-Magic：抓取工具与仪表盘首秀！****：一位用户展示了一个 **Instagram Lead Scraper**，并配有**自定义仪表盘**，详见附带的[视频](https://cdn.discordapp.com/attachments/1315696461316358175/1346986901877555250/full_automation_demo.mov?ex=67ca2ecf&is=67c8dd4f&hm=e3114edc2b6e1e5171c2c1be5cbb011437c737ba2268afe4e381cbfa44cf2cf0&)。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1346885774045352007)** (60 条消息🔥🔥): 

> `Claude 成本, M4 Macbook Air, Qwen 模型, 用于 LLM 后端的 React, Windsurf Cascade` 


- ****Claude 昂贵的代码库提问****：一位成员报告说，向 **Claude** 询问一个关于其小型代码库的问题花费了 **$0.26**。
   - 另一位成员建议将代码库复制到 **Claude** 目录中，并在 **Claude Desktop** 上激活 filesystem MCP server 以实现免费访问。
- ****苹果 M4 MacBook Air 天蓝色亮相****：根据[此公告](https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/)，苹果发布了新款 **MacBook Air**，搭载 **M4 芯片**、**Apple Intelligence** 功能，并新增了**天蓝色**，起售价为 **$999**。
- ****Qwen 的 QwQ-32B 媲美 DeepSeek-R1****：**Qwen** 发布了 **QwQ-32B**，这是一款全新的 **320 亿参数推理模型**，其性能可与 **DeepSeek-R1** 等模型相媲美，详见[此博客文章](https://qwenlm.github.io/blog/qwq-32b)。
   - 该模型通过 **RL** 和持续缩放进行训练，提升了在数学和编程方面的表现，现已在 [HuggingFace](https://huggingface.co/Qwen/QwQ-32B) 上可用。
- ****React 被重新构想用于后端 LLM 工作流？****：一位成员分享了一个“犀利观点”，认为 **React** 是后端 LLM 工作流的最佳编程模型，并链接到了一篇关于使用 Node.js 后端和 **类 React** 组件模型构建 [@gensx_inc](https://x.com/_Evan_Boyle/status/1897347251120562205) 的博客文章。
   - 另一位成员指出，核心点在于使用 graph.addEdge 等 API 定义图的不足，建议 **Lisp** 可以更轻松地创建 DSL，而另一位成员则推荐 [Mastra](https://mastra.ai/docs/workflows/00-overview) 作为无框架的替代方案。
- ****Windsurf 的 Cascade 浪潮席卷元素检查****：**Windsurf** 发布了 **Wave 4**，其特色功能 **Cascade** 可将元素/错误直接发送到聊天框，旨在减少对“检查元素”的需求，演示请见[此链接](https://x.com/windsurf_ai/status/1897378545799979238)。
   - 本次更新还包括：预览、Cascade Auto-Linter、MCP UI 改进、Tab 键导入、建议操作、Claude 3.7 改进、推荐奖励以及 Windows ARM 支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cherry_cc12/status/1897366964080926902">来自 Chen Cheng (@cherry_cc12) 的推文</a>：谁将成为 QwQ 家族的下一个成员？引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，却能与顶尖推理模型相媲美...</li><li><a href="https://www.together.ai/blog/nvidia-gb200-together-gpu-cluster-36k">Together AI 将与 Hypertec Cloud 合作共同构建配备 3.6 万个 Blackwell GPU 的增强型 NVIDIA GB200 集群</a>：未找到描述</li><li><a href="https://blog.google/products/search/ai-mode-search/">扩展 AI Overviews 并引入 AI Mode</a>：AI Mode 是 Google Search 中一项新的生成式 AI 实验。</li><li><a href="https://mastra.ai/docs/workflows/00-overview">处理复杂的 LLM 操作 | Workflows | Mastra</a>：未找到描述</li><li><a href="https://x.com/OpenAI/status/1897346510821711959">来自 OpenAI (@OpenAI) 的推文</a>：成为 Plus 用户的好日子。</li><li><a href="https://x.com/_Evan_Boyle/status/1897347251120562205">来自 Evan Boyle (@_Evan_Boyle) 的推文</a>：热门观点：React 是后端 LLM 工作流的最佳编程模型。关于我们为何构建 @gensx_inc 的新博客文章</li><li><a href="https://x.com/windsurf_ai/status/1897378545799979238">来自 Windsurf (@windsurf_ai) 的推文</a>：Windsurf Wave 4 发布了！本次更新包含：🖼️ Previews ✏️ Cascade Auto-Linter ⚙️ MCP UI 改进 ➡️ Tab 键导入 ↩️ 建议操作 🫶 Claude 3.7 改进 🤝 推荐计划 🖥️ Windows ARM 支持...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，却能与 DeepSeek-R1 等顶尖推理模型相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://x.com/Alibaba_Qwen/status/1897366093376991515">来自 Qwen (@Alibaba_Qwen) 的推文</a>：Qwen2.5-Plus + Thinking (QwQ) = QwQ-32B。这就是你在 Qwen Chat 上使用这款新模型的方式！引用 Qwen (@Alibaba_Qwen)：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数...</li><li><a href="https://x.com/tim_cook/status/1897325061104918961">来自 Tim Cook (@tim_cook) 的推文</a>：向新款 MacBook Air 问好！这款全球最受欢迎的笔记本电脑现在配备了 M4 芯片、Apple Intelligence 功能，以及一种美丽的全新颜色——天蓝色。</li><li><a href="https://github.com/x1xhlol/v0-system-prompts">GitHub - x1xhlol/v0-system-prompts</a>：通过在 GitHub 上创建账户，为 x1xhlol/v0-system-prompts 的开发做出贡献。</li><li><a href="https://www.apple.com/newsroom/2025/03/apple-introduces-the-new-macbook-air-with-the-m4-chip-and-a-sky-blue-color/">Apple 推出搭载 M4 芯片和天蓝色的新款 MacBook Air</a>：Apple 发布了新款 MacBook Air，配备 M4 芯片、长达 18 小时的电池续航、12MP Center Stage 摄像头，以及更低的起售价。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1346801638341607456)** (47 条消息🔥): 

> `SDXL 手部修复，写实风格放大器，免费文本转视频，Stable Diffusion v4, SD3.5 Large TurboX` 


- **使用 SDXL 自动修复手部**：用户正在寻找在使用 8GB **VRAM** 的 **SDXL** 时无需 **inpainting** 即可自动修复手部的方法，并探索了 **embeddings**、**face detailers** 和 **OpenPose** **control nets** 等选项。
   - 用户正在为 **SDXL** 寻找优质的 **hand LoRAs** 以及无需手动 **inpainting** 的自动手部纠正方案。
- **使用 WAN 2.1 模型在本地生成视频**：用户讨论了如何从单张照片免费创建视频，建议使用 **WAN 2.1 i2v** 模型，但指出这需要高性能 **GPU** 和耐心。
   - 一些人建议使用带有免费额度的在线服务，尽管效果可能参差不齐；而另一些人指出，由于电费消耗，在本地生成视频仍然是有成本的。
- **SD 3.5 表现不佳**：成员们讨论称，在测试中 **SD 3.5** *的表现甚至不如 **flux dev**，且远不及 **ideogram** 或 **imagen** 等大型模型。*
   - 另一位成员表示，*与早期的 **sd 1.5** 相比，它们已经取得了长足的进步*。
- **SD3.5 Large TurboX 开源**：**TensorArt** 开源了 **SD3.5 Large TurboX**，该模型使用 **8 sampling steps**，与官方的 **Stable Diffusion 3.5 Turbo** 相比，速度提升了 **6 倍**，且图像质量更优；该模型已在 [Hugging Face](https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX) 上发布。
   - 他们还发布了 **SD3.5 Medium TurboX**，仅需 **4 sampling steps** 即可在中端 **GPU** 上用 **1 秒**生成 **768x1248** 分辨率的图像，带来了 **13 倍**的速度提升；该模型同样可在 [Hugging Face](https://huggingface.co/tensorart/stable-diffusion-3.5-medium-turbo) 上获取。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Video%20Model%20Support.md#wan-21">SwarmUI/docs/Video Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI（原 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调易用的高级工具、高性能和可扩展性。</li><li><a href="https://github.com/CompVis/stable-diffusion.git">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>: 一个潜空间文本转图像扩散模型。</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1j406g1/sd35_large_turbox_just_released/">SD3.5 Large TurboX just released</a>: 由 u/NukeAI_1 发布在 r/StableDiffusion • 180 点赞和 44 条评论</li><li><a href="https://tenor.com/view/let-us-cook-let-me-cook-lets-cook-cooking-walter-white-gif-2649071825756414039">Let Us Cook Let Me Cook GIF - Let us cook Let me cook Lets cook - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346896645736108103)** (1 条消息): 

> `Agentic Document Workflows, DeepLearningAI 合作伙伴关系` 


- **LlamaIndex 展望 Agentic Document Workflows**：根据 **LlamaIndex** 的观点，直接集成到大型软件流程中的 **Agentic Document Workflows** 是知识 **Agent** 的未来。
   - **LlamaIndex** 与 [DeepLearningAI](https://t.co/EvAKtIAzlC) 合作推出了一门关于如何构建这些工作流的短篇课程。
- **DeepLearningAI 与 LlamaIndex 合作推出新课程**：**LlamaIndex** 已与 [DeepLearningAI](https://t.co/EvAKtIAzlC) 合作创建了一门短篇课程。
   - 该课程专注于构建直接集成到软件流程中的 **Agentic Document Workflows**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346820815391424553)** (43 条消息🔥): 

> `ImageBlock 与 OpenAI 集成问题、Query Fusion Retriever 引用问题、分布式 AgentWorkflow 架构、LlamaIndex 中的 Agent 执行性能分析/计时、Flask 与 Gunicorn 的内存消耗` 


- **ImageBlock 集成问题困扰 LlamaIndex 用户**：一位成员报告在最新版本的 LlamaIndex 中使用 **ImageBlock** 与 **OpenAI** 时出现问题，系统无法识别图像。
   - 机器人建议确保使用最新版本的 LlamaIndex 及其依赖项，并验证是否使用了正确的模型（例如 *gpt-4-vision-preview*）。
- **Query Fusion Retriever 无法引用来源**：一位用户报告在其 LlamaIndex 设置中，**node post-processing**（节点后处理）和 **citation templates**（引用模板）在使用 **Query Fusion Retriever** 时无法正常工作，特别是在使用倒数重排序（reciprocal reranking）时。
   - 有人建议 **Query Fusion Retriever** 中的去重过程可能是导致节点处理期间元数据丢失的原因，并[链接了他们的代码](https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine)以供审查。
- **AgentWorkflow 寻求分布式架构**：一位成员询问 **AgentWorkflow** 是否原生支持**分布式架构**，即不同的 Agent 运行在不同的服务器/进程上。
   - 一种建议是通过为 Agent 配备对服务进行远程调用的工具来实现这一点，而不是依赖内置的分布式架构支持。
- **GPT-4o Audio Preview 模型在 Agent 中表现不佳**：一位用户在 LlamaIndex Agent 中使用 **OpenAI 的音频 `gpt-4o-audio-preview` 模型** 时面临集成挑战，特别是在流式传输事件方面。
   - 有人指出 AgentWorkflow 会自动对聊天消息调用 `llm.astream_chat()`，这可能与 OpenAI 的音频支持冲突，并建议了一个潜在的变通方案：避免使用 AgentWorkflow 或禁用 LLM 流式传输。
- **Claude Sonnet 3.5 的 ReactAgent 表现**：一位用户发现 **Claude Sonnet 3.5** 与 **ReactAgent** 配合效果不佳，会一次性生成多个步骤。
   - 另一位成员表示赞同，建议 **Anthropic 模型** 在使用 **XML prompting** 时效果最好，并建议使用 API 的 **function calling** 作为 React 更可靠的替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Restodecoca/ingest-reposit/tree/main/app/engine">ingest-reposit/app/engine at main · Restodecoca/ingest-reposit</a>：通过在 GitHub 上创建账户来为 Restodecoca/ingest-reposit 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/ea1f987bb880519bb7c212b33d8615ae4b8fdbf8/llama-index-core/llama_index/core/agent/workflow/function_agent.py#L41">llama_index/llama-index-core/llama_index/core/agent/workflow/function_agent.py at ea1f987bb880519bb7c212b33d8615ae4b8fdbf8 · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#audio-support">OpenAI - LlamaIndex</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1346806760471527536)** (13 条消息🔥): 

> `上传教科书、NotebookLM API、NotebookLM 与 PDF、在线游戏策略优化、NotebookLM 播客功能` 


- **Gemini 与 NotebookLM 处理物理教学大纲的对比**：一位用户上传了整个 **180 页的物理教科书**，但发现使用 **Gemini** 时系统无法跳出教学大纲的范畴。
- **NotebookLM 的 PDF 上传体验并不理想**：用户讨论了上传 PDF 的问题，发现它们几乎不可用，尤其是在文本和图像混合的内容中。
   - 一位成员建议*将 PDF 转换为 **Google Docs** 或 **Slides***，这样处理混合内容的效果更好。
- **NotebookLM API 咨询**：一位用户询问是否存在 **NotebookLM API** 或未来的相关计划，并列举了许多工作流优化用例。
- **NotebookLM 用于游戏优化**：一位用户利用 **NotebookLM** 来增强在线游戏的策略，使用游戏文档、个人游戏数据（如 JSON 卡片列表）和提取的电子表格数据作为源材料。
- **播客功能是讲座的“游戏规则改变者”**：一位大学教授使用播客功能提供启发性的讨论和类比，帮助学生从宏观角度理解问题。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1346825873650614282)** (29 messages🔥): 

> `Standalone Android App, NLM Response Length, Formula Rendering in NLM, File Upload Issues, Podcast Generator` 


- ****Android App 期待****：一位用户询问是否有独立的 NotebookLM Android App，另一位用户建议 Web 版本 *用起来也挺好*。
- ****NLM 的长延迟回复****：几位用户报告 NLM 的回复比平时长得多，建议可能需要调整 Prompt 以获得更具体的答案。
- ****Podcast 功能好评****：一位用户称赞 NotebookLM 的 Podcast 生成器非常出色，但想知道是否有办法将 Podcast 的长度从 *17 分钟延长到 20 分钟*。
- ****菲律宾语支持困惑****：一位用户对 NotebookLM 是否支持菲律宾语感到困惑，理由是 Google 的 [Vertex AI 文档](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#languages-gemini) 与 NotebookLM 支持页面提供的信息存在冲突。
- ****PWA 好评与安装信息****：用户讨论了 NotebookLM 作为 Progressive Web App (**PWA**) 的可用性，它可以安装在手机和 PC 上，无需专门的应用即可提供类似原生应用的体验。



**提到的链接**：<a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#languages-gemini">未找到标题</a>：未找到描述

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1346831805226942490)** (33 messages🔥): 

> `Gaslight Benchmark, GPT-4.5 vs Claude image generation, Video AI Prompt Engineering, Hermes Special Tokens, Post-training RL` 


- ****Gaslight Benchmark** 探索开始**：一位成员询问是否存在 **gaslight benchmark** 来对比 **GPT-4.5** 与其他模型。
   - 另一位成员开玩笑地回复了一个[讽刺性基准测试](https://spiritshare.org/benchmark.html)的链接。
- ****GPT-4.5 的说服力提升****：一位成员提到 **GPT-4.5 的 system card** 显示其在说服力方面有显著提升，这可能归功于 Post-training RL。
   - 另一位成员表示有兴趣看到初创公司使用 **Post-training RL**。
- ****Hermes 的特殊 Token 揭晓****：一位成员询问训练 **Hermes** 时使用的特殊 Token 列表。
   - 另一位成员澄清特殊 Token 是 *<im_start>* 和 *<im_end>*，以及 *</SCRATCHPAD>* 和 *</THINKING>*。
- ****Video AI Prompt Engineering 难题****：一位成员在为 **Kling** 或 **Hailou** 等 **Video AI 工具**编写有效 Prompt 时遇到困难。
   - 他们请求提供示例 Prompt，以学习如何掌握技巧并生成逼真的图像或草图。
- ****2025 年的诈骗****：一位用户发布了 *大家好，新来的，刚开始学习 ML，不知怎么就到了这里，嗯，还是个骗子，用户 ID 是 1336741798512693382，如果他们删除并重新发布的话*。
   - 作为回应，一位成员回复道 *喜欢在 2025 年骗人*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

garry_plahilsin07: Opps
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346934090061975614)** (4 条消息): 

> `QwQ-32B, Reinforcement Learning, DeepSeek R1, Tool calling syntax, Hermes format` 


- **Qwen 的 QwQ-32B 表现强劲**：根据[这篇博客文章](https://qwenlm.github.io/blog/qwq-32b/)，来自 Qwen 的 **320 亿参数模型 QwQ-32B** 实现了与拥有 **6710 亿参数**（其中 **370 亿**被激活）的 **DeepSeek-R1** 相当的性能。
- **对 QwQ-Max 的期待落空**：一位用户表示他们原本*期待 QwQ-Max 的发布*，但现在将对 **QwQ-32B** 和 **DeepSeek R1** 进行实际体验对比（vibe check）。
   - 该模型可通过 [QWEN CHAT](https://chat.qwen.ai)、[Hugging Face](https://huggingface.co/Qwen/QwQ-32B)、[ModelScope](https://modelscope.cn/models/Qwen/QwQ-32B)、[DEMO](https://huggingface.co/spaces/Qwen/QwQ-32B-Demo) 和 [DISCORD](https://discord.gg/yPEP2vHTu4) 获取。
- **QwQ-32B 遵循工具调用语法 (Tool Calling Syntax)**：QwQ-32B 使用特定的语法进行工具调用，例如使用 `<tool_call> { \"name\": \"get_current_temperature\", \"arguments\": { \"location\": \"San Francisco, CA, USA\"} } </tool_call>`。
- **QwQ-32B 采用 Hermes 格式**：据观察，QwQ-32B 使用了 **Hermes format**。
- **RL Scaling 推动模型创新**：根据[这篇博客文章](https://qwenlm.github.io/blog/qwq-32b/)，**Reinforcement Learning (RL)** 的扩展提升了模型在典型预训练之外的性能，**DeepSeek R1** 通过冷启动数据和多阶段训练展示了其在复杂推理方面的能力。



**提到的链接**：<a href="https://qwenlm.github.io/blog/qwq-32b/">QwQ-32B: Embracing the Power of Reinforcement Learning</a>：QWEN CHAT Hugging Face ModelScope DEMO DISCORD。扩展 Reinforcement Learning (RL) 具有超越传统预训练和后训练方法并增强模型性能的潜力。最近的研究...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346850582664839169)** (1 条消息): 

> `Android Chat App, OpenRouter Integration, Speech-to-Text, Text-to-Image, Text-to-Speech` 


- ****Taiga 作为 Android 聊天应用首次亮相****：一款名为 **Taiga** 的开源 Android 聊天应用已经发布，允许用户自定义想要使用的 LLMs，并已[预集成 OpenRouter](https://github.com/Ayuilos/Taiga/releases/tag/v0.1.0-rc.0)。
- **Taiga 路线图包括语音、图像和 TTS**：开发者计划添加基于 **Whisper** 模型和 **Transformer.js** 的本地 **Speech To Text**，以及 **Text To Image** 支持和基于 **ChatTTS** 的 **TTS** 支持。



**提到的链接**：<a href="https://github.com/Ayuilos/Taiga/releases/tag/v0.1.0-rc.0">Release Releasing `v0.1.0-rc.0` · Ayuilos/Taiga</a>：这是一个预发布版本。一切都有可能发生变化。无需多言，请尽情使用，如有 Bug 或建议请告知我！

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1346808173994049556)** (32 messages🔥): 

> `Text Completion 中的 Prefill 使用, 面向 Coding Agents 的 OpenRouter 文档, DeepSeek instruct 格式, LLMGuard 集成, 基于用量的计费应用` 


- **Prefill 让用户感到困惑**：成员们讨论了 **prefill** 是否被错误地用于 **text completion** 模式而非 chat completion，并质疑为什么它会被应用到用户消息中。
   - 一位用户指出：*"除了 prefill 对用户消息没有意义之外，他们显然将其定义为 chat completion 而不是 text completion，哈哈"*。
- **面向 Coding Agents 的文档获取**：一位用户询问是否可以将 **OpenRouter 的文档** 作为一个单一的大型 markdown 文件获取，以便给 **Coding Agents** 使用。
   - 另一位用户提供了一个指向文档[完整文本文件](https://openrouter.ai/docs/llms-full.txt)的链接。
- **DeepSeek 的 Instruct 格式仍不明朗**：讨论强调了关于 **DeepSeek instruct 格式** 在**多轮对话**中缺乏清晰文档的问题，并指出即使深入研究其 tokenizer 也很令人困惑。
   - 一位用户分享了 [tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)，其中定义了 `<｜begin of sentence｜>` 和 `<｜end of sentence｜>` token。
- **LLMGuard 插件？**：一位成员询问是否有计划在 OpenRouter 的 **LLM API** 中添加类似 **LLMGuard** 的插件，以实现 **Prompt Injection** 扫描等功能。
   - 该用户链接到了 [LLMGuard](https://llm-guard.com/)，并想知道 OpenRouter 是否能处理 PII 脱敏。
- **探索基于用量的计费应用**：一位用户就构建一个模仿 **OpenRouter 支付流程**（并在其基础上收取少量百分比费用）的应用征求意见，并询问潜在的陷阱。
   - 该用户概述了一个 Happy path：*"1. 检查用户余额，2. 发起 LLM 调用，3. 获取调用成本，4. 扣除成本加手续费，5. 微薄利润。"*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json">tokenizer_config.json · deepseek-ai/DeepSeek-V3 at main</a>：未找到描述</li><li><a href="https://llm-guard.com/input_scanners/anonymize/">Anonymize - LLM Guard</a>：未找到描述</li><li><a href="https://llm-guard.com/output_scanners/ban_competitors/">Ban Competitors - LLM Guard</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2412.19437v1">DeepSeek-V3 Technical Report</a>：我们介绍了 DeepSeek-V3，这是一个强大的混合专家 (MoE) 语言模型，总参数为 671B，每个 token 激活 37B。为了实现高效推理和经济高效的训练，DeepS...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346886048961269874)** (13 messages🔥): 

> `双层优化 (Bilevel Optimization), Sparsemax 泛化, DDP 权重乱码, MPEC 转换, AI 方法复杂度` 


- **双层优化面临质疑**：一位成员认为，尽管双层优化在算法和重构方面有潜在优势，但它本质上并没有提供超出标准优化技术的内容，并链接到了关于[双层优化 (Bilevel Optimization)](https://en.wikipedia.org/wiki/Bilevel_optimization)的讨论。
   - 他们强调，双层规划的效用主要是直观上的，因为它会转化为**带均衡约束的数学规划 (MPEC)**，并作为单层非线性规划问题 (NLP) 来解决。
- **Sparsemax 作为双层 Max：一种 AI 重新构架？**：一位成员提议将 **Sparsemax** 构架为一个双层优化 (BO) 问题，暗示其具有动态调整不同神经网络层的潜力。
   - 针对这一点，另一位成员详细说明了 **Sparsemax** 是一个向概率单纯形 (probability simplex) 投影的问题，具有闭式解，并利用拉格朗日对偶性 (Lagrangian duality) 证明了计算可以简化为闭式形式的注水算法 (water-filling)。
- **扩散模型挑战闭式解**：一位成员指出了将 AI 方法简化为单层闭式形式的局限性，特别是在像**扩散模型 (Diffusion Models)** 这样复杂的场景中，并引用道：*采样灵活性并不需要完整的闭式表达式*。
   - 他们链接到了之前关于[扩散模型](https://discord.com/channels/714501525455634453/986699377257119794/1342302214999248957)的讨论，建议在无法获得闭式解时，将双层优化作为自适应 max 函数的替代方案。
- **DDP 导致权重乱码：调试 PyTorch？**：一位成员报告在使用 **PyTorch**、**DDP** 和 **4 张 GPU** 时遇到问题，检查点重新加载导致某些 GPU 上的权重出现乱码。
   - 另一位成员建议确保在初始化 DDP *之前*，在所有 GPU 上初始化模型并加载检查点，以减轻权重乱码问题。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347012093320102010)** (5 messages): 

> `Proactive T2I Agents, DeepMind's Papers` 


- **Agents 主动澄清文本生成图像过程**：一篇新论文 [Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty](https://arxiv.org/abs/2412.06771) 介绍了 **proactive T2I agents**。这些 Agent 会主动提出澄清性问题，并将它们对用户意图的理解呈现为一个可编辑的 belief graph（信念图），以解决用户 prompt 描述不充分的问题。
   - 一段[补充视频](https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD)显示，至少 **90%** 的人类受试者认为这些 Agent 及其 belief graphs 对他们的 **T2I workflow** 很有帮助。
- **DeepMind 占据讨论主导地位**：成员们表示 **DeepMind's papers** 是顶级的，也是生成式 AI 领域“最好”的。
   - 另一位成员对此表示赞同，并提到他们会怀念未来关于该团队出版物的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.06771">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>：生成式 AI 模型的用户 prompt 通常描述不充分，导致响应效果不佳。这个问题在文本生成图像 (T2I) 领域尤为明显，用户通常难以……</li><li><a href="https://youtu.be/HQgjLWp4Lo8?si=6SxQdUbzocp3zrKD">Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</a>：由 Meera Hahn 演讲的 Google TechTalk，2024-12-05。摘要：生成式 AI 模型的用户 prompt 通常描述不充分或过于开放，这可能导致……
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1346871725404454972)** (2 messages): 

> `QwQ-32B release, RL scaling, Qwen2.5-32B` 


- ****QwQ-32B** 模型面世**：**Alibaba** 发布了 **QwQ-32B**，这是一个仅有 **32 billion parameters** 的新型推理模型，可与 **DeepSeek-R1** 等尖端推理模型相媲美。更多信息可以在 [Qwen Blog](https://qwenlm.github.io/blog/qwq-32b) 找到。
- **RL Scaling 方案研究**：Alibaba 研究了扩展 RL 的方案，并基于其 **Qwen2.5-32B** 模型取得了令人印象深刻的结果。他们观察到，持续扩展 RL 可以帮助中型模型在面对巨型 MoE 模型时获得具有竞争力的性能，详见[其公告](https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=nzomNQaPFSk"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/Alibaba_Qwen/status/1897361654763151544?t=t5Bec1knVsQuXpTu24fWqw&s=19">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们仅有 320 亿参数的新型推理模型，可与 DeepSeek-R1 等尖端推理模型相媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1346854838645817365)** (14 messages🔥): 

> `Cohere, Enterprise, Support` 


- **联系 Cohere 企业支持**：一位成员询问 **Cohere** 的联系人以讨论 **enterprise deployment**，并被引导至邮件支持。
   - 该成员在一周前已经给支持部门发了邮件，希望能通过 Discord 获得更快的回复，并强调了他们对加拿大公司的偏好。
- **B2B 交付周期较慢**：一位成员提到企业咨询由直销团队处理，B2B 的交付周期可能较慢，可能需要长达 **6 周**。
   - 另一位成员反驳说，**Cohere** 作为一家 AI 公司，通常会在 **2-3 天** 内回复。

### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1346968241582506117)** (1 条消息): 

> `Aya Vision, Multilingual Vision Model, AyaVisionBenchmark, Multimodal AI` 


- **Cohere 推出 Aya Vision：用 23 种语言洞察世界**：Cohere For AI 发布了 **Aya Vision**，这是一个支持 **23 种语言** 的 **8B** 和 **32B** 开源权重（open-weights）多语言视觉研究模型。
   - 这是 Cohere 的首个*多模态模型（multimodal model）*，在图像描述（image captioning）、视觉问答（visual question answering）、文本生成和翻译方面表现出色，详见 [博客文章](https://cohere.com/blog/aya-vision)。
- **Aya Vision：现已登陆 Hugging Face 和 Kaggle**：此次发布包含一个名为 **AyaVisionBenchmark** 的新多语言视觉评估集，并已在 [Hugging Face](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision) 和 [Kaggle](https://www.kaggle.com/models/cohereforai/aya-vision) 上线。
   - 该模型旨在将文本和图像翻译成清晰、自然的语言文本，增强了其通用性。
- **Aya Vision 聊天机器人已在 Poe 和 WhatsApp 上线！**：用户可以在 [Poe](https://poe.com/Aya-Vision) 上访问 Aya Vision，也可以通过 [WhatsApp](https://cohere.com/research/aya/whatsapp) 免费发送消息，支持 **23 种语言**。
   - 用户可以使用该模型进行文本和视觉提问、为图像添加描述以及将内容翻译成自然语言。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/aya-vision">Aya Vision: Expanding the Worlds AI Can See</a>: 我们最先进的开源权重视觉模型为全球 AI 驱动的多语言和多模态通信奠定了基础。今天，Cohere 的开放研究部门 Cohere For AI 自豪地宣布...</li><li><a href="https://www.kaggle.com/models/cohereforai/aya-vision">CohereForAI | Aya Vision | Kaggle</a>: C4AI Aya Vision 是 8B 和 32B 参数模型的开源权重研究版本，具有针对各种视觉语言用例（包括 OCR、描述、视觉推理等）优化的先进能力...</li><li><a href="https://poe.com/Aya-Vision">Aya-Vision - Poe</a>: Aya Vision 是一个 32B 开源权重多模态模型，具有针对各种视觉语言用例优化的先进能力。该模型经过训练，在视觉和文本方面均精通 23 种语言...</li><li><a href="https://cohere.com/research/aya/whatsapp">Text Aya on WhatsApp | Cohere For AI</a>: Aya Expanse 支持 23 种语言，是世界上最好的多语言 AI。现在已在 WhatsApp 上线，可以用您的语言免费给 Aya 发消息。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1346853704313208907)** (1 条消息): 

> `Cohere Reranker v3.5 latency` 


- **寻求 Cohere Reranker v3.5 的延迟数据**：一位成员询问 **Cohere Reranker v3.5** 的延迟数据，指出尽管在一次采访中有所承诺，但目前缺乏公开数据。
   - 他提到受访者曾表示会分享图表，但最终并未分享。
- **寻求 Cohere Reranker v3.5 延迟数据（续）**：无详细信息，第二个主题仅为满足格式要求。
   - 更多细节，第二个主题仅为满足格式要求。


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1346962182549798942)** (2 条消息): 

> `Introductions` 


- **用户寻求销售/企业支持**：一位新用户加入并寻求联系 **销售（sales）** / **企业支持（enterprise support）** 人员。
- **新用户介绍**：鼓励新用户介绍自己，包括公司/行业/大学背景、正在研究的项目、喜欢的技术/工具以及希望从社区获得什么。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346826892803571724)** (10 条消息🔥): 

> `Mojo 稳定性、虚拟活动录制、Triton vs Mojo、Mojo 与 Python 的关系` 


- **Mojo 尚不稳定，仍有工作待完成**：一位成员指出 **Mojo** 尚未稳定，*仍有大量工作要做*。
- **虚拟活动，无录像可用**：一位成员询问虚拟活动的 **YouTube 录像**，但该活动*未进行录制*。
   - 团队*肯定会考虑在未来举办类似的虚拟活动*。
- **Triton 作为替代方案出现**：一位成员建议将 **Triton**（支持 **Intel** 和 **Nvidia** 硬件的 AMD 软件）作为替代方案。
   - 然而，另一位成员表示 **Mojo 并不是 Python 的超集**，而是 *Python 语言家族的一员*。
- **Mojo 与 Python 家族的联系**：一位成员澄清说 **Mojo** 不是 Python 的超集，而是 *Python 语言家族的一员*，对于 Mojo 来说，成为超集就像是*戴上口罩（束缚）*。
   - 它将无法充分利用这些年来大幅进化的编程语言设计特性。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346876472148623503)** (5 条消息): 

> `Mojo 与 Python 性能基准测试、Mojo/Python 项目文件夹结构、Python.add_to_path 替代方案、tests 文件夹中的 Symlink 替代方案、Modular 论坛` 


- **Mojo 在 Python venv 中性能受损**：基准测试显示，在激活的 **Python virtual environment (venv)** 中运行 Mojo 二进制文件时，即使文件没有导入 Python，**Mojo 的性能提升**也会显著降低。
   - 用户正在寻求关于为什么 Python venv 会影响本应独立的 Mojo 二进制文件的见解。
- **Mojo/Python 项目文件夹结构受到质疑**：一位开发者请求对 **Mojo/Python 项目文件夹结构**的反馈，该结构涉及导入 Python 标准库、自定义 Python 模块以及运行用 Mojo 编写的测试。
   - 他们大量使用 **`Python.add_to_path`** 来导入自定义模块，并在 `tests` 文件夹中使用 Symlink 来定位源文件。
- **寻求 `Python.add_to_path` 的替代方案**：该开发者正在寻求在 Mojo 中查找自定义 Python 模块时使用 **`Python.add_to_path`** 的替代方案，旨在实现更整洁的导入机制。
   - 他们还对测试期间访问源文件的 `tests` 文件夹中 symlinking 的替代方案感兴趣。
- **Mojo/Python 文件夹结构讨论移至 Modular 论坛**：一位用户在 Modular 论坛上发起了关于 **Mojo/Python 项目文件夹结构**的讨论，并[链接到了论坛帖子](https://forum.modular.com/t/mojo-python-project-folder-structure/677)。
   - 此举是为了确保讨论的长期可发现性和保留，因为 *Discord 的搜索和数据保留功能较差。*



**链接提到**：<a href="https://forum.modular.com/t/mojo-python-project-folder-structure/677">Mojo/Python 项目文件夹结构</a>：我最初在 Discord 上发布了此内容（链接），但 @DarinSimmons 认为这会是该论坛的一个好话题。我正在为一个大型 Mojo/Python 项目寻求文件夹组织方面的指导。我...

  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1346960276326715434)** (2 messages): 

> `SynaLinks release, Keras vs Pytorch frameworks, Knowledge graph RAGs, Reinforcement learning, Cognitive architectures` 


- **SynaLinks 框架加入 LM 竞技场！**：一个新的**基于图的可编程神经符号 LM 框架** **SynaLinks** 已发布。它从 Keras 的函数式 API 中汲取灵感，旨在通过异步优化和受限结构化输出等特性实现生产就绪 —— [GitHub 上的 SynaLinks](https://github.com/SynaLinks/synalinks)。
- **SynaLinks 专注于 Knowledge Graphs 和 RL！**：与 DSPy 不同，**SynaLinks** 将专注于 **knowledge graph RAGs、reinforcement learning 和 cognitive architectures**，旨在 LLM 框架领域占据不同的生态位。
   - 该项目得到了 **François Chollet** 的建议，此前他非常喜欢作者之前的项目（使用 **DSPy** 制作的 **HybridAGI**）。
- **SynaLinks 在 HuggingFace 上在线运行！**：**SynaLinks** 的代码示例已提供，并可通过 [Hugging Face Space](https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks) 在线运行，鼓励社区反馈和实验。
- **SynaLinks 已投入生产！**：该框架已在客户的生产环境中运行，后续还有更多项目，展示了其在实际应用中的潜力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/YoanSallami/synalinks-noteboooks">synalinks notebooks - a Hugging Face Space by YoanSallami</a>: no description found</li><li><a href="https://github.com/SynaLinks/synalinks">GitHub - SynaLinks/synalinks: 🧠🔗 Graph-Based Programmable Neuro-Symbolic LM Framework - a production-first LM framework built with decade old Deep Learning best practices</a>: 🧠🔗 Graph-Based Programmable Neuro-Symbolic LM Framework - a production-first LM framework built with decade old Deep Learning best practices - SynaLinks/synalinks
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346844637356232808)** (11 messages🔥): 

> `Optimizing intent classification with DSPy, Comparing texts for contradictions, DSPy adapters system, Straggler threads in dspy.Evaluate and dspy.Parallel` 


- **DSPy 优化意图分类**：DSPy 可以帮助优化需要专用 Agent 的意图分类。
   - 一位用户正朝着这个方向努力。
- **比较文本矛盾：计算密集型任务**：比较两段文本是否存在矛盾是计算密集型的；在大数据量下使用 DSPy 的 **CoT 模块**会耗费大量时间。
   - 一位用户探索了通过 one-shot 提供多个样本对的方法，并指出对于 **LLM** 在返回列表对时（特别是在使用 `OutputField` 时）能否遵守预期返回结构的保留意见。
- **DSPy 简化显式类型规范**：DSPy 通过如下代码简化了显式类型规范：```contradictory_pairs: list[dict[str, str]] = dspy.OutputField(desc="List of contradictory pairs, each with fields for text numbers, contradiction result, and justification.")```，但由于未指定 `dict` 的键，这在技术上存在歧义。
   - 相反，应考虑使用 ```list[some_pydantic_model]```，其中 **some_pydantic_model** 具有正确的字段。
- **DSPy 的 Adapters 系统将 Signature 与 Provider 解耦**：DSPy 的 **adapters** 系统将 Signature（对所需内容的声明式规范）与不同 Provider 生成 Completion 的方式解耦。
   - 默认情况下，DSPy 使用经过良好调优的 **ChatAdapter**，并回退到 **JSONAdapter**，利用 **VLLM**、**SGLang**、**OpenAI**、**Databricks** 等 Provider 的结构化输出 API 进行受限解码。
- **解决滞后线程问题，实现更流畅的 DSPy 评估**：[PR 7914](https://github.com/stanford-nlp/dspy/pull/791)（已合并）解决了 `dspy.Evaluate` 或 `dspy.Parallel` 中卡住的*滞后（straggler）*线程问题，旨在实现更顺畅的运行。
   - 此修复将在 **DSPy 2.6.11** 中提供；用户可以从 `main` 分支进行测试，无需更改代码。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1346813249366003723)** (6 messages): 

> `ShapeTracker 合并的 Lean 证明，淘宝 4090，gfx10 trace 问题，Rust CubeCL` 


- **ShapeTracker 合并获得 Lean 证明！**: 一位成员宣布了一个约 90% 完成的 Lean 证明，关于何时可以合并 ShapeTrackers，可在 [此仓库](https://github.com/Nielius/Tensorlayouts) 和 [此 issue](https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082) 中查看。
   - Offsets 和 masks 未被考虑在内，但作者认为扩展它很直接，不值得费力。
- **淘宝上的 4090**: 一位成员分享了淘宝上 **96GB 4090** 的链接 ([X 帖子](https://x.com/yomix1337/status/1893692548108984391?s=46))。
   - 另一位成员根据经验评论道：*好东西都在淘宝上*。
- **gfx10 Trace 问题？**: 一位成员询问关于 trace 的看法，以及是否应该将其记录为一个 issue。
   - 一位成员建议这可能与 **ctl/ctx** 大小有关，并请求运行 `IOCTL=1 HIP=1 python3 test/test_tiny.py TestTiny.test_plus` 来帮助调试，因为他们缺少 **gfx10** 硬件。
- **Rust CubeCL：好用吗？**: 一位成员询问了 **Rust CubeCL** 的质量，指出它来自 **Rust Burn** 的同一个团队。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/yomix1337/status/1893692548108984391?s=46">来自 Gene edited Yostuba (@Yomix1337) 的推文</a>: @EsotericCofe 5 月后推出</li><li><a href="https://github.com/Nielius/Tensorlayouts">GitHub - Nielius/Tensorlayouts: 合并两个 tensor view 的充分必要条件的 Lean 证明</a>: 合并两个 tensor view 的充分必要条件的 Lean 证明 - Nielius/Tensorlayouts</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8511#issuecomment-2700706082">关于任意 ShapeTrackers 的可合并性 · Issue #8511 · tinygrad/tinygrad</a>: 嘿，我想提出一种关于 view 合并问题的新表述和证明，我还没见有人提到过。我见过一个叫 @Nielius 的人提出的表述，但遗憾的是它...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346824786889801871)** (2 messages): 

> `Suleiman 介绍，Naveen 介绍，CVPR25 论文` 


- **Suleiman 加入聊天，探索 AI 生物黑客 (Biohacking)**: Suleiman 是一家沙特公司的高管，具有软件工程背景，他在频道中介绍了自己，表达了对 **tech** 和 **AI** 的热爱。
   - 他目前正在探索**营养学**和**补充剂科学**，旨在开发 **AI 驱动的生物黑客工具**来改善人类生活。
- **Naveen 加入，展示 Unlearning 研究**: Naveen 是来自 IIT 的硕士兼研究助理，他介绍了自己，表示目前从事**文本到图像扩散模型 (Text to Image Diffusion Models)** 中的 **Machine Unlearning** 工作。
   - 他提到最近在 **CVPR25** 发表了一篇论文。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347000060550058046)** (2 messages): 

> `Observation 3.1，ARC 训练，无损压缩，智能行为` 


- **Observation 3.1：具有普适性吗？**: 一位用户质疑 **Observation 3.1** 是否对几乎任何具有非零均值的两个分布以及几乎任何 **ARC 训练**上的 u35% 都普遍成立。
   - 没有关于 **Observation 3.1** 的具体条件或例外的进一步讨论或澄清。
- **无需预训练的 ARC AGI**: Isaac Liao 和 Albert Gu 在他们的[博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)中探讨了**无损信息压缩 (lossless information compression)** 是否能产生**智能行为 (intelligent behavior)**。
   - 他们的目标是提供一个实际演示，而不是重新讨论关于**高效压缩**在智能中作用的理论探讨。



**提到的链接**: <a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>: 未找到描述

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346890865729212577)** (2 messages): 

> `arc_challenge.yaml，ARC-Challenge 任务，Few-shot 学习` 


- **ARC-Challenge 任务采用 arc_challenge.yaml**: 成员们指出，他们正在 **25-shot** 配置中为 **ARC-Challenge 任务**使用 **arc_challenge.yaml**。
- **围绕模型评估的 Few-shot 提示词讨论**: 对话包括使用有限数量的示例（例如 **25 shots**）来评估模型在特定任务上的性能。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346864089363779714)** (5 条消息): 

> `Tokenizer 定制化, Checkpointer 保存方法, special_tokens.json 处理, Copy files 逻辑` 


- **Tokenizer 定制化问题**：用户从 HF 下载模型，并在其自定义的 **special_tokens.json** 中体现了一些自定义 Tokenizer 逻辑，但训练后 checkpoint 目录中保存的却是原始下载的版本。
   - 疑似原因是 [这段代码](https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-896)，它假设从 HF 下载的大多数非模型文件都可以直接原样使用。
- **针对 Tokenizer 问题的快速修复建议**：建议的快速修复方案是在下载的模型目录中，用用户自己的版本替换下载的 **special_tokens.json**。
   - 成员们讨论了是否可以让 *copy_files* 和 *save_checkpoint* 逻辑更通用，以支持定制化。
- **Checkpointer 保存方法的参数**：一位成员建议通过向 Checkpointer 的 **save_checkpoint** 方法传递一个新参数来支持此用例，但这还需要通过 config 暴露出来。
   - 成员们正在考虑，在没有非常充分理由的情况下，是否值得暴露任何新参数。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07">GitHub - pytorch/torchtune at 80da6a5dae23a201595d07041c12ffde830332d7</a>：PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/80da6a5dae23a201595d07041c12ffde830332d7/torchtune/training/checkpointing/_checkpointer.py#L892-L896.">torchtune/torchtune/training/checkpointing/_checkpointer.py at 80da6a5dae23a201595d07041c12ffde830332d7 · pytorch/torchtune</a>：PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346810951265157201)** (4 条消息): 

> `MOOC 讲座, 证书提交` 


- **MOOC 学生参加相同的讲座**：一位成员询问是否有 **Berkeley** 学生参加而 **MOOC** 学生没有的讲座。
   - 另一位成员回答说，**Berkeley** 学生和 **MOOC** 学生参加的是相同的讲座。
- **12 月的证书提交**：一位成员表示他们在 12 月提交了某些内容。
   - 另一位成员询问并确认证书申报表是使用哪个电子邮件提交的。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1346806842453524540)** (2 条消息): 

> `AST 指标, V1 数据集` 


- **需要 AST 指标定义**：一位成员询问 **AST 指标** 是否仅仅是产生正确格式函数调用的 **LLM 响应** 百分比。
   - 澄清 **AST 指标定义** 将有助于他人更好地理解排行榜。
- **关于 V1 数据集构建的查询**：另一位成员询问了 **V1 数据集** 是如何构建的。
   - 了解 **数据集构建** 过程可以深入洞察 **评估方法论**。


  

---


---


{% else %}


> 各频道的详细细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}