---
companies:
- openai
- llama-index
- perplexity-ai
- google-deepmind
date: '2025-04-15T05:16:26.134697Z'
description: '**OpenAI** 发布了 **GPT-4.1**，包括 **GPT-4.1 mini** 和 **GPT-4.1 nano**，重点提升了**编程**、**指令遵循**以及处理高达
  **100 万 token** 的**长上下文**能力。该模型在 **SWE-bench verified** 测试中获得了 **54 分**，并在内部基准测试中比
  **GPT-4o 提升了 60%**。**GPT-4.1 nano** 的定价极低，**每百万输入 token 仅需 0.10 美元**，**每百万输出 token
  为 0.40 美元**。**GPT-4.5 Preview** 正被弃用，取而代之的是 GPT-4.1。集成支持方面，**Llama Index** 提供了首日（day
  0）支持。此外，也有一些针对 **GPT-4.1 nano** 的负面反馈。与此同时，**Perplexity 的 Sonar API** 与 **Gemini-2.5
  Pro** 在 LM Search Arena 排行榜上并列第一。随着更新的提示词指南和 Cookbook（示例库）的发布，还引入了 **MRCR** 和 **GraphWalks**
  等新基准测试。'
id: 53c5605c-6852-4809-bb57-e7a742de7da3
models:
- gpt-4.1
- gpt-4.1-mini
- gpt-4.1-nano
- gpt-4o
- gemini-2.5-pro
original_slug: ainews-gpt-41-the-new-openai-workhorse
people:
- sama
- kevinweil
- omarsar0
- aidan_mclau
- danhendrycks
- polynoamial
- scaling01
- aravsrinivas
- lmarena_ai
title: GPT 4.1：OpenAI 的新主力
topics:
- coding
- instruction-following
- long-context
- benchmarks
- model-pricing
- model-integration
- model-deprecation
---

<!-- buttondown-editor-mode: plaintext -->**GPT 4.1 是你对 OpenAI 的全部需求吗？**

> 2025/4/11-2025/4/14 的 AI 新闻。我们为你检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 29 个 Discord 服务器（211 个频道和 16961 条消息）。预计节省阅读时间（以 200wpm 计算）：1382 分钟。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

GPT 4.1 相关链接：

- https://openai.com/index/gpt-4-1/
- 新基准测试：[MRCR](https://huggingface.co/datasets/openai/mrcr) 和 [GraphWalks](https://huggingface.co/datasets/openai/graphwalks)
- 新的 [提示指南](https://platform.openai.com/docs/guides/text?api-mode=responses#prompting-gpt-4-1-models) 和 [cookbook](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)

以及在 Latent Space 上发布的新访谈：

https://youtu.be/y__VY7I0dzU


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**GPT-4.1 发布与性能**

- **可用性与特性**：[@sama](https://twitter.com/sama/status/1911830886896799931) 宣布 **GPT-4.1、GPT-4.1 mini 和 GPT-4.1 nano** 现已在 API 中可用，并强调了它们在 **coding、指令遵循和处理长上下文**（最高 100 万 token）方面的优势。[@kevinweil](https://twitter.com/kevinweil/status/1911833354682401148) 指出 **GPT-4.1** 在 **SWE-bench verified** 上获得了 **54 分**。
- **指令遵循**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911860099829674184) 指出 **GPT-4.1** 比 **GPT-4o** 更可靠地遵循指令，特别是在 **格式遵守、执行否定指令和排序** 方面。
- **定价与成本**：[@stevenheidel](https://twitter.com/stevenheidel/status/1911830168118923291) 表示 **GPT-4.1-nano** 是已发布的最便宜且最快的模型，成本为 **$0.10/1M input ($0.03 缓存) 和 $0.40/1M output**。
- **编程性能**：[@omarsar0](https://twitter.com/omarsar0/status/1911870478857437540) 强调，根据 **Windsurf AI** 的数据，**GPT-4.1** 在 **SWE-benchmark** 等内部基准测试中比 **GPT-4o** 提升了 **60%**，减少了 40% 的不必要文件读取，并减少了 70% 的不必要文件修改。[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911859923161428002) 表示它在前端 coding 方面技能显著提升，并具有可靠的 tool use 能力。[@polynoamial](https://twitter.com/polynoamial/status/1911831926241153170) 提到 **GPT-4.1 在 SWE-Bench Verified 上达到了 55%**，且它并非推理模型。
- **集成与支持**：[@llama_index](https://twitter.com/llama_index/status/1911863053257445713) 提到 Llama Index 现在已提供对 **GPT-4.1** 的首日支持。
- **初步印象**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1911850291026362426) 指出初创公司工程师对 **GPT-4.1 mini/nano** 感到惊讶，发现它与 **GPT-4o** 相当，但价格便宜得多。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1911847214168850805) 将其描述为 **帕累托最优（Pareto optimal）的瑞士军刀级 API 模型**，是 Agent 栈中优于 newssonnet 的升级选择。
- **ChatGPT 上的受限可用性**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1911837235521163670) 建议免费的 **GPT-4.1 mini** 在 **ChatGPT** 上可能被刻意限制，以激励大学生订阅 **ChatGPT Plus**。
- **命名规范**：[@polynoamial](https://twitter.com/polynoamial/status/1911843302770643004) 开了关于模型命名的玩笑。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911832534796886439) 指出 **GPT** 模型的命名方案遵循 **GPT-4.10**，因此它排在 **GPT-4.5** 之后，而 [@kevinweil](https://twitter.com/kevinweil/status/1911795255877198311) 则开玩笑说本周它在命名方面不会变得更好。
- **GPT-4.5 的弃用**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1911860805810716929) 宣布 API 中的 **GPT-4.5 Preview** 将从今天开始弃用，并于 7 月 14 日完全关闭，因为 **GPT-4.1 提供了改进或类似的性能**。
- **负面评价**：[@scaling01](https://twitter.com/scaling01/status/1911852197714731276) 建议不要使用 **GPT-4.1-nano**，称其为一个糟糕的模型。
[@scaling01](https://twitter.com/scaling01/status/1911847193465471374) 报告称 GPT-4.1 API 版本比 Optimus Alpha 更差。

**模型基准测试与对比**

- **Search Arena 排行榜**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1911849161869410355) 报告称 **Perplexity 的 Sonar API** 与 **Gemini-2.5 Pro** 在 LM Search Arena 排行榜上并列第一。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1911842298914328959) 报告称 **Gemini-2.5-Pro-Grounding 和 Perplexity-Sonar-Reasoning-Pro** 位居榜首。
- **Llama 4 ELO 下跌**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1911332387817931161) 报告称 **Llama 4** 的 **ELO** 评分悄然从 **1417 降至 1273**，与 **DeepSeek v2.5** 持平。
- **Google Gemini 2.5 Pro**：[@abacaj](https://twitter.com/abacaj/status/1911529618089427122) 表示 Google 终于凭借 Gemini 2.5 Pro 打造出了最强模型。[@omarsar0](https://twitter.com/omarsar0/status/1911451522703020189) 对 **Gemini 2.5 Pro** 在调试和重构方面的出色表现感到惊讶，并认为它是理解大型代码库的最佳模型之一。
- **Gemini 2.0 Flash**：[@_philschmid](https://twitter.com/_philschmid/status/1911862052852744642) 报告称 **Gemini 2.0 Flash** 的价格为 **$0.1/$0.4**（每 1M tokens 的输入/输出），在 GPQA Diamond、Multilingual MMLU 和 MMMU 上均取得了优异成绩。
- **Mistral 模型**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1911382474640220546) 表示 Long Mistral 模型表现出色，其最新的 **24B 模型**非常有竞争力。
- **Nvidia Llama Nemotron-Ultra**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450216164700252) 指出 **Nvidia 发布了 Llama Nemotron-Ultra**，这是一款拥有 **253B** 参数的推理 AI，击败了 DeepSeek R1、Llama 4 Behemoth 和 Maverick，并且完全开源。
- **Meta Llama 4**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450182937346285) 详细介绍称 **Meta 发布了 Llama 4** 系列原生多模态开源模型，上下文窗口高达 10M tokens，包括 109B 参数的 Scout、400B 参数的 Maverick，以及第三款 2T 参数的 Behemoth。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1911841914590015586) 指出 Llama 4 Scout 拥有前所未有的 1000 万 token 上下文窗口，Maverick 击败了 GPT-4o 的公开基准测试，而 Behemoth 声称性能超越了 GPT-4.5 和 Claude 3.7 Sonnet。
- **Kimina-Prover 与其他模型对比**：[@_lewtun](https://twitter.com/_lewtun/status/1911793153931100180) 指出，在新的编程语言 Lean 中，**Kimina-Prover** 仅凭 7B 参数就在奥数级数学竞赛中击败了 **Gemini 2.5 Pro 和 o3-mini**！
- **GPT-4.1 vs DeepSeek-V3**：[@scaling01](https://twitter.com/scaling01/status/1911831700964872531) 表示 **GPT-4.1** 在 AIME 上的表现比 **DeepSeek-V3-0324 低 10% 以上**，且价格贵 8 倍，在 GPQA 上的表现也逊色。
- **GPT-4.1 vs. GPT-4.5**：[@scaling01](https://twitter.com/scaling01/status/1911828552452112536) 表示 **GPT-4.1** 在 AIME 和 MMLU 上的表现优于 **GPT-4.5**。

**机器人与具身智能 (Robotics and Embodied AI)**

- **Hugging Face 收购**：[@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1911843020309213547) 报告称 **Hugging Face** 收购了开源机器人制造商 **Pollen Robotics**。
- **Fourier 的开源人形机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450377175589313) 提到了 **Fourier** 的全开源人形机器人。
- **三星与 Google 合作**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450160078467386) 指出 **三星** 宣布与 **Google** 建立合作伙伴关系，为其 **Ballie** 家庭机器人提供支持，采用 Google 的 **Gemini** 及其自有的多模态 AI 模型。

**AI 研究与论文 (AI Research and Papers)**

- **预训练中的反思**：[@omarsar0](https://twitter.com/omarsar0/status/1911442761238340095) 总结了一篇论文，认为反思能力在 Pre-Training 阶段就已经出现，并引入了对抗性推理任务，以证明即使没有经过监督式的 Post-Training，自我反思和纠错能力也会随着 Compute 的增加而提升。
- **强化学习与推理**：[@rasbt](https://twitter.com/rasbt/status/1911494805101986135) 总结了一篇论文，显示 Reinforcement Learning (RL) 会导致推理模型生成更长的回复，这并非因为准确性需要，而是因为 RL 训练更倾向于长回复。
- **多模态模型 Scaling Laws**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633260582523332) 总结了一项涉及 457 个原生多模态模型 (NMMs) 的 Scaling Laws 分析，揭示了早融合 (early-fusion) 架构优于晚融合 (late-fusion) 架构，且 Mixture of Experts (MoEs) 能显著提升性能。
- **论文列表**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633257952575492) 发布了一份顶级 AI/ML 研究论文列表，[@dair_ai](https://twitter.com/dair_ai/status/1911444942523621550) 也分享了类似的顶级 AI 论文。
- **视觉分词器**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911639044406329846) 指出，在扩展视觉 Tokenizers 时，GigaTok 提升了图像重建、生成和表示学习的效果。

**其他模型与 AI 工具发布**

- **Deep Cogito 模型**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450441457557816) 提到 Deep Cogito 结束隐身状态并发布了 Cogito v1 Preview，这是一个全新的开源模型系列。
- **Runway Gen 4 Turbo**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450331470258198) 分享了 Runway 发布 Gen 4 Turbo 的消息，这是其视频模型的更快版本，面向所有用户开放，包括免费层级用户。
- **Midjourney V7**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450308795904091) 报道 Midjourney 发布了 V7 版本，具有更高的质量、增强的提示词遵循能力以及支持语音的 Draft Mode。
- **Microsoft Copilot 更新**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450285760708712) 提到微软升级了其 Copilot 应用，增加了新的记忆功能、网页浏览操作和视觉功能。
- **Amazon AI**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450262977368259) 表示亚马逊发布了一款名为 "Nova Sonic" 的语音转语音 AI，并推出了 Reel 1.1 AI，支持长达 2 分钟的视频生成。
- **Nvidia 卡通 AI**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450240143536333) 分享了 Nvidia 和斯坦福大学的研究人员展示的一种 AI 技术，可以生成连贯的、长达一分钟的卡通片。
- **DolphinGemma**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1911767367534735832) 推出了 DolphinGemma，这是一款帮助我们深入探索海豚交流世界的 AI 🐬，它是一个音频到音频 (audio to audio) 模型。

**AI 基础设施与工具**

- **OpenAI 基础设施规模**：[@sama](https://twitter.com/sama/status/1911504090989035824) 提到 OpenAI 的计算系统规模非常惊人，他们需要帮助。
- **ElevenLabs MCP 集成**：[@adcock_brett](https://twitter.com/adcock_brett/status/1911450399585796491) 报道 **ElevenLabs** 推出了其 MCP 服务器集成，使 **Claude** 和 **Cursor** 等平台能够访问 AI 语音功能。
- **Qdrant + n8n**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1911700608731521450) 指出 **Qdrant** 和 **n8n** 正在实现超越相似性搜索的流程自动化。
- **LangChain 工具**：[@LangChainAI](https://twitter.com/LangChainAI/status/1911449301542195582) 推广了一个开源库，可将任何 LLM 连接到 MCP 工具以构建自定义 Agent，其特点是与 LangChain 集成，并支持网页浏览、Airbnb 搜索和 3D 建模。
- **Hamel Husain Chrome 扩展**：[@HamelHusain](https://twitter.com/HamelHusain/status/1911521751739351509) 创建了一个 Chrome 扩展程序，允许你将整个 Gemini 对话（通过 aistudio）保存到 gist 或复制为 Markdown，同时也为 Claude 提供了一个类似的扩展。

**AI 策略与讨论**

- **开源机器人**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1911768941107511624) 倡导将 AI 机器人开源。
- **优先考虑医疗诊断**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1911620129227776011) 指出，更好的诊断和护理交付比寻找一种新的化疗药物来治愈癌症更具影响力。
- **LLM 与搜索引擎**：[@rasbt](https://twitter.com/rasbt/status/1911467070975271217) 认为 LLM 不会取代搜索引擎。
- **通过 RL 实现简洁性**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1911633319910928756) 总结了一项研究，该研究揭示了简洁性与推理准确性之间的相关性，并提出了一种通过二次 RL 阶段在 LLM 中实现更简洁推理的方法。
- **开发者体验**：[@sedielem](https://twitter.com/sedielem/status/1911821106811679094) 强调了开发者体验的重要性。
- **RAG 中专业知识的价值**：[@HamelHusain](https://twitter.com/HamelHusain/status/1911830144635084930) 强调了与那些在优化检索和搜索方面投入大量时间的人交流，对于提升 RAG 能力的价值。
- **AI 的未来**：[@scaling01](https://twitter.com/scaling01/status/1911187189548933143) 分享道，LLM 的基本情况是，在未来几年内，它们将演变成高度专业化的“孤独症式”超智能（autistic superintelligences），在验证过程简单的领域表现出色。

**幽默与杂项**

- **扁平化组织**：[@typedfemale](https://twitter.com/typedfemale/status/1911213477118845086) 开了一个关于扁平化组织的玩笑。
- **辣酱**：[@vikhyatk](https://twitter.com/vikhyatk/status/191174862499812563) 开玩笑说不要在睡前 5 分钟尝试“杀人胡蜂”辣酱。
- **过度炒作的估值**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1911429251804364835) 谈论了 SSI 的估值。
- **个人轶事**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1911801507571437589) 因为自动纠错，不小心问朋友在 "jew york" 玩得怎么样。
[@sjwhitmore](https://twitter.com/sjwhitmore/status/1911286312365342759) 表示他们刚把宝宝哄睡，30 分钟后就发现自己在看宝宝的照片。
[@willdepue](https://twitter.com/willdepue/status/1911591028697833779) 提到 OpenAI 猎帽是下次播客的必备品；[@sama](https://twitter.com/sama/status/1911496563157000568) 买了很多没用上的傻傻的婴儿用品，但他推荐 Cradlewise 婴儿床，以及比你想象中多得多的拍嗝布。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾


### 主题 1：“GLM-4 强化学习模型的令人兴奋的进展”

- **[glm-4 0414 已发布。包含 9B、32B 版本，提供具备和不具备推理及反思能力的模型](https://www.reddit.com/r/LocalLLaMA/comments/1jz2iuc/glm4_0414_is_out_9b_32b_with_and_without/)** ([Score: 190, Comments: 64](https://www.reddit.com/r/LocalLLaMA/comments/1jz2iuc/glm4_0414_is_out_9b_32b_with_and_without/)): **GLM-4 0414 已发布，推出了 9B 和 32B 两种尺寸的六款新模型，包含具备和不具备推理及反思（rumination）能力的版本。这些模型包括 **GLM-Z1-32B-0414**，这是一款具备深思熟虑能力的推理模型，基于 GLM-4-32B-0414 通过冷启动、扩展强化学习（RL）以及在数学、代码和逻辑等任务上的进一步训练开发而成。**GLM-Z1-Rumination-32B-0414** 是一款具备反思能力的深度推理模型，能够进行更深、更长时间的思考，以解决更具开放性和复杂性的问题。**GLM-Z1-9B-0414** 是一款 9B 参数模型，采用了上述所有技术，在数学推理和通用任务中表现出卓越的能力，在同尺寸开源模型中性能名列前茅。** GLM-Z1-9B-0414 被视为一个惊喜，在效率和效果之间实现了极佳的平衡，是寻求轻量化部署用户的强大选择。这些模型在数学能力、研究型写作以及解决复杂任务的能力方面都有显著提升。

  - 一位评论者指出，新的 32B 模型只有 **2 个 KV 值头**，导致 KV cache 占用的空间比 Qwen 2.5 32B 少约四倍，并好奇这是否会导致处理长上下文时出现问题。
  - 另一位评论者对基准测试结果印象深刻，提到 GLM 模型自 Llama 1 时代就已存在且一直表现出色，但认为它们在西方需要更好的营销，因为它们似乎被忽视了。
  - 一位评论者对模型包含 [SuperGPQA](https://www.reddit.com/r/LocalLLaMA/comments/1j3byj5/bytedance_unveils_supergpqa_a_new_benchmark_for/) 基准测试结果表示赞赏，这使得该模型能与许多其他模型更具可比性。

### 主题 2. “DeepSeek 对 AI 推理的开源贡献”

- **[DeepSeek 即将开源其推理引擎](https://i.redd.it/1am95yongrue1.png)** ([Score: 1312, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1jytw62/deepseek_is_about_to_opensource_their_inference/)): **DeepSeek 即将开源其推理引擎，这是一个基于 **vLLM** 的修改版本。他们正准备将这些修改回馈给社区。一篇题为《DeepSeek 推理引擎开源之路》的文章概述了他们的动机和步骤，包括代码库分歧、基础设施依赖以及有限的维护带宽等挑战。他们对开源生态系统表示感谢，并计划与现有项目合作，将功能模块化并分享优化成果，旨在提升通用人工智能 (**AGI**) 以造福人类。更多详情可以在他们的 [GitHub 仓库](https://github.com/deepseek-ai/open-infra-index/tree/main/OpenSourcing_DeepSeek_Inference_Engine)中找到。** 发帖者对 DeepSeek 对社区的承诺表示热忱，特别赞赏他们的目标——“旨在让社区从第一天起就能获得最先进 (SOTA) 的支持”。人们对 DeepSeek 的贡献可能给开源 AI 社区带来的积极影响感到兴奋。

  - 一位用户指出，DeepSeek 可能不会直接开源其推理引擎，而是将其改进贡献给 **vLLM** 和 **sglang**，因为他们的分支版本已经太陈旧了。
  - 另一位评论者对 DeepSeek 表示深切赞赏，将他们对这家公司的喜爱比作对 Wikipedia 的喜爱。
  - 一位用户认为 DeepSeek R1 的发布是 AI 竞赛中的一个关键时刻，指出虽然它不是最聪明或最便宜的模型，但它标志着除了 OpenAI 之外还有 **Claude**、**Gemini** 和 DeepSeek 等替代方案，并赞赏他们在开源领域的持续创新。

- **[DeepSeek 将开源其推理引擎的部分内容——分享独立的功能和优化，而非全栈](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md)** ([Score: 252, Comments: 9](https://www.reddit.com/r/LocalLLaMA/comments/1jysiwc/deepseek_will_opensource_parts_of_its_inference/)): **DeepSeek 将开源其推理引擎的部分内容，通过分享独立的功能和优化，而不是发布整个技术栈。他们正致力于将优化成果移植到流行的开源推理引擎中，如 **vLLM**、**llama.cpp** 和 **kobold**。** 一些人认为标题具有误导性，暗示 DeepSeek 保留了部分技术栈。然而，其他人认为，通过将优化移植到流行的开源推理引擎，DeepSeek 正在更有效地为社区做出贡献。用户对这些贡献带来的推理性能提升感到乐观。

  - 评论者注意到 DeepSeek 正在通过移植优化来增强 **vLLM**、**llama.cpp** 和 **kobold** 等流行的开源推理引擎。
  - 一些用户对 DeepSeek 的贡献可能带来的更好推理性能感到兴奋。
  - 用户正在询问目前是否有任何来自 DeepSeek 的资源可用于个人项目。


## 其他 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### 主题 1. “变革科学：OpenAI 的新推理模型”

- **[科学突破即将到来](https://i.redd.it/61jy8q8tctue1.jpeg)** ([Score: 724, Comments: 207](https://www.reddit.com/r/singularity/comments/1jz0ppu/scientific_breakthroughs_are_on_the_way/)): **OpenAI 即将发布名为 **o3** 和 **o4-mini** 的新推理模型，*这些模型首次能够独立开发新的科学构想* [[1]](https://www.theinformation.com/articles/openais-latest-breakthrough-ai-comes-new-ideas)。这些 AI 模型可以同时处理来自不同专业领域的知识并提出创新实验——这种能力此前被认为是人类的专属领域。早期版本已显示出可喜的成果：*Argonne National Laboratory 的科学家们使用这些模型的早期版本，能够在几小时内设计出复杂的实验，而以往则需要数天时间。* OpenAI 计划为这些高级服务每月收取高达 20,000 美元的费用，这将是标准 ChatGPT 订阅价格的 1000 倍。** 这项技术可能会极大地加速科学发现进程，特别是当与能够控制模拟器或机器人的 AI agents 结合使用，以直接测试和验证生成的假设时。这代表了该领域的一次潜在革命，将以前认为人类独有的能力转移到了 AI 身上。

  - 一些用户对 OpenAI 为这些 AI 模型每月收取 20,000 美元的做法持怀疑态度，质疑为什么该公司不自己利用它们来解决重大问题。
  - 另一些人认为该信息是可信的，因为消息来源在 OpenAI 新闻方面非常准确，并暗示这可能是公司有意泄露的。
  - 对于高昂的订阅费存在困惑和猜测，用户回想起之前的案例，当时传闻的价格比实际发布价格更高。

### 主题 2. “令人兴奋的 AI 模型创新与竞争动态”

- **[GPT 4.1 具备 100 万 token 上下文。输入 2 美元/百万 token，输出 8 美元/百万 token。比 4o 更聪明。](https://i.redd.it/fw34ped81uue1.jpeg)** ([评分: 313, 评论: 140](https://www.reddit.com/r/singularity/comments/1jz42ic/gpt_41_with_1_million_token_context_2million/)): **GPT-4.1 被宣布为处理复杂任务的旗舰模型，具有 **100 万 token 的上下文窗口**，最大输出能力为 **32,768 个 token**。定价设定为输入 **每百万 token 2 美元**，输出 **每百万 token 8 美元**，并提供了关于缓存输入成本的额外信息。该模型声称比之前的版本具有更高的智能。** 原帖作者强调 GPT-4.1 比 *4o 更聪明*，突出了其先进的能力，并暗示它是对之前模型的重大改进。

  - 用户将 GPT-4.1 与 **Google 的 Gemini 模型**进行了比较，讨论了定价和性能差异，一些人表示希望成本能更低。
  - 对于 GPT-4.1 如何有效利用其 **100 万 token 上下文窗口**存在质疑，有人提到像 Gemini 2.5 这样的模型可以*完美处理约 10 万个 token*。
  - 一些人推测 GPT-4.1 可能会导致 **GPT-4.5** 的取消，并希望即将推出的 **o4-mini** 等模型能达到 State-of-the-art 水平。

- **[OpenAI 发布 GPT 4.1 模型及定价](https://platform.openai.com/docs/models/compare?model=gpt-4.1)** ([评分: 245, 评论: 119](https://www.reddit.com/r/OpenAI/comments/1jz450k/openai_announces_gpt_41_models_and_pricing/)): **OpenAI 宣布发布 **GPT 4.1** 模型及其定价详情。** 这一公告引起了褒贬不一的反应，一些用户对模型数量的激增表示沮丧，而另一些人则在讨论 **GPT-4.1** 的可用性和改进。

  - 一位用户对众多的模型表示沮丧，称他们*厌倦了这堆乱七八糟的随机模型*。
  - 另一位用户指出 **GPT-4.1** 将仅通过 API 提供，并注意到改进已逐渐融入 ChatGPT 中最新版本的 **GPT-4o**。
  - 一些用户拿 2024 年 6 月的知识截止日期开玩笑，幽默地希望自己能*像 GPT 4.1 一样好骗* 😂。

- **[Kling 2.0 将于明天揭晓。](https://i.redd.it/jtux6mutqrue1.jpeg)** ([评分: 281, 评论: 29](https://www.reddit.com/r/singularity/comments/1jyupii/kling_20_will_be_unveiled_tomorrow/)): **Kling 2.0 将于明天，即 **2025 年 4 月 15 日格林威治标准时间上午 6:00** 揭晓。公告包含一张带有动态绿色背景和口号 *“From Vision to Screen”*（从愿景到屏幕）的图片，强调创新与技术。更多详情请访问 [https://x.com/Kling_ai/status/1911702934183882986](https://x.com/Kling_ai/status/1911702934183882986) 和 [https://xcancel.com/Kling_ai/status/1911702934183882986](https://xcancel.com/Kling_ai/status/1911702934183882986)。** 宣传图片传达了对 **Kling 2.0** 的兴奋和期待，以其动态设计吸引了关注。该口号暗示了相对于之前版本的重大进步，激发了潜在用户的热情。

  - 用户对 **Kling 2.0** 的快速发布感到惊讶，其中一位指出 *“1.6 版本仍然是第一名”*。
  - 讨论强调了过去的一周是多么 *“疯狂”*，出现了众多 AI 进展，如 **Midjourney v.7**、**OpenAI GPT-4.1** 和 **Google Agentspace Boxing**。
  - 用户对 **Kling 2.0** 的新功能充满期待，例如更长的视频生成，因为目前用户*“卡在 5-10 秒”*。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的总结

**主题 1. GPT-4.1 模型：发布、性能与可用性**

- **OpenAI 发布 GPT-4.1，基准测试超越 4o**：[OpenAI 的博客文章](https://openai.com/index/gpt-4-1/) 宣布了 **GPT-4.1**，该模型以*长上下文推理*为卖点，基准测试显示比 **GPT-4o** 提升了约 **10%**。Windsurf AI 立即集成了该模型，并提供[为期一周的免费无限访问](https://x.com/windsurf_ai/status/1911833698825286142)；同时 OpenRouter 推出了 [GPT-4.1、Mini 和 Nano 版本](https://openrouter.ai/announcements)，并透露 **Optimus Alpha** 和 **Quasar Alpha** 是 **GPT-4.1** 的早期测试版本。
- **Windsurf 为用户提供免费 GPT-4.1**：[Windsurf AI](https://windsurf.ai) 已将 **GPT-4.1** 设为其新的默认模型，并在所有方案中提供[为期一周的免费无限使用](https://x.com/windsurf_ai/status/1911833698825286142)，随后将以 **每次使用 0.25 积分** 的折扣费率计费。Cursor 社区成员预计 [GPT-4.1 将成为新标准](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616)，随着用户向 4.1 迁移，4.5 版本将被弃用。
- **Aider v0.82.0 支持 GPT-4.1 Patch 格式**：[Aider v0.82.0](https://aider.chat/HISTORY.html) 现在支持 **GPT-4.1**，包括 OpenAI 新的 `patch` 编辑格式。成员反馈其[性能与 Quasar/Optimus 相似](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334)，但单次运行成本为 4.76 美元。LlamaIndex 也宣布通过 `llama-index-llms-openai` [对 GPT-4.1 API 提供零日支持](https://t.co/JPEX3KAoWS)，并指出在 Agent 方法上提升了约 **2%**。

**主题 2. Gemini 2.5 Pro：性能波动与价格变动**

- **Google 削弱了 Gemini 2.5 Pro 的 Tool Calling 能力**：LMArena Discord 成员报告称 [Google 削弱了 Gemini 2.5 Pro 的 Tool Calling 功能](https://discord.com/channels/1340554757349179412/1340554757827461211/1360331259464646767)，可能是出于成本考虑，导致其无法执行工具调用。OpenRouter 也开始[对 Gemini 的长 Prompt 收取正常价格](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333)，结束了对 Gemini 2.5 超过 200k token 以及 Gemini 1.5 超过 128k token 的 Prompt 提供的 50% 折扣。
- **Gemini 2.5 Pro 仍是 UI 设计冠军**：尽管存在 Tool Calling 问题，Cursor 社区成员仍称赞 [Gemini 2.5 Pro 具有“惊人”的 UI 设计能力](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616)，强调了其独特的输出和上下文保留能力。然而，Aider 用户发现与 Claude 3.7 相比，[Gemini 2.5 Pro 在处理长上下文和代码补全方面表现吃力](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334)。
- **Gemini 2.5 Pro 数据处理强劲，抢占 Perplexity 订阅**：Manus.im Discord 用户赞扬了 [Gemini 2.5 Pro 的数据处理实力](https://discord.com/channels/1348819876348825620/1349440650495398020/1360330273090306088)，一位用户因 Gemini 2.5 Pro 的优越性以及单次任务更低的积分消耗而取消了 Perplexity 订阅。不过，Perplexity AI 的 Sonar 模型在 [LM Arena 的 Search Arena 中与 Gemini-2.5-Pro-Grounding 持平](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution)，并指出 Sonar 表现出色归功于其搜索来源多出 2-3 倍。

**主题 3. 开源模型与工具势头强劲**

- **OpenRouter 开启免费模型大门**：OpenRouter 新增了 [六款免费模型](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333)，包括 NVIDIA 针对推理和 RAG 优化的 Llama-3 变体（[Nano-8B](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free)、[Super-49B](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free)、[Ultra-253B](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)），以及经过角色扮演微调的 [QwQ-32B-ArliAI-RpR-v1](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free)。Hugging Face 也迎来了 [Meta 的 Llama 4 Maverick 和 Scout](https://huggingface.co/blog/llama4-release) 进行测试。
- **DeepSeek 开源推理引擎，DeepCoder 展现编程实力**：DeepSeek 开源了其 [Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md)，引发了关于小型供应商推理性能的讨论。Nous Research AI 重点介绍了 [DeepCoder](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/)，这是一个 14B 参数的开源模型，通过增强的 GRPO 和 64K 上下文泛化实现了顶尖的编程性能。
- **Aider 和 Ollama 拥抱开源生态系统**：Aider v0.82.0 增加了对 [Fireworks AI 的 deepseek-v3-0324 模型](https://aider.chat/HISTORY.html) 的支持，并改进了与 Gemini 2.5 Pro 配合的架构师模式。Hugging Face 用户越来越多地使用 [Ollama 在本地运行模型](https://discord.com/channels/879548962464493619/879548962464493622/1360430374358089921) 以替代受 API 限制的模型，LlamaIndex 则建议在 [Agent 工作流中通过 Ollama 使用 Llama3 或 Mistral 等大型开源模型](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439)。

**Theme 4. 硬件优化与 CUDA 深度探索**

- **GPU Mode 探索用于 GEMM 性能的希尔伯特曲线**：GPU Mode Discord 成员讨论了在 GEMM 实现中使用 [希尔伯特曲线 (Hilbert curves)](https://discord.com/channels/1189498204333543425/1189607595451895918/1360545047430434978)，基准测试显示随着矩阵规模增加，其效果优于 cuBLAS，尽管 Morton 排序被认为是更实际的折中方案。NVIDIA 还发布了其 [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip)，并提醒警惕 AI 生成的 PR 提交。
- **CUDA 同步与 `memcpy_async` 注意事项**：GPU Mode 成员交流了 CUDA 同步指南，建议使用自定义算子 (custom ops) 和内联加载 (load inline)，并调查了 [`cuda::memcpy_async`](https://discord.com/channels/1189498204333543425/1189607726595194971/1361163559236796578) 的性能下降问题，指出这是一个协作式 API，要求所有线程传递相同的指针，且对齐问题可能会阻碍合并内存访问 (coalesced memory access)。
- **Threadripper vs Xeon 以及 DDR5 RAM 带宽瓶颈**：LM Studio 的硬件讨论辩论了 [Threadripper 与 Xeon CPU 在 Token 生成方面的性价比](https://discord.com/channels/1110598183144399058/1153759714082033735/1360339430967345374)，并认为 DDR5 RAM 带宽是一个瓶颈，理论上它限制了整体硬件利用率，且首字延迟限制了最大 tokens/s。

**Theme 5. Agent 开发与工具生态系统的演进**

- **MCP Server 工作坊与日益增长的采用率**：MLOps@Chipro 宣布将于 4 月 17 日举办一个[用于构建生产级 MCP Server 的 AWS 工作坊](https://buff.ly/R7czfKK)，强调 MCP 是改善 ML 上下文管理的新兴标准。由于 [MCP 的普及](https://discord.com/channels/1312302100125843476/1312302100125843479/1360343887255703552)，Wildcard 暂停了 `agents.json` 的维护；AutoMCP 作为一个平台推出，旨在提供类似 Vercel/Heroku 的体验来[将 Agent 项目部署为 MCP Server](https://labs.naptha.ai/)。
- **LlamaIndex LlamaParse 在文档解析方面表现出色**：LlamaIndex 强调了 [LlamaParse 在处理包含图像、表格和图表的文档时增强的解析质量](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439)，在解析质量上超越了 SimpleDirectoryReader 等基础读取器，并提供了一份关于 [LlamaParse Layout Agent Mode 视觉引用 (Visual Citations)](https://medium.com/ai-artistry/visual-citations-with-llamaparse-layout-agent-mode-a-comprehensive-guide-a623a5fb41fc) 的指南。
- **Brave Search API 在 Agent 流水线中受到关注**：Yannick Kilcher Discord 成员建议将 [Brave Search API](https://discord.com/channels/714501525455634453/1269724655405498429/1360675922608521438) 作为 Agent 流水线的一个不错选择，即使是免费层级，并指出其 AI 摘要功能比 OpenAI 的 Web Search API 更便宜。Hugging Face 正在为使用 [smolagents 的新 Deep Search Agent](https://agent.galadriel.com/) 寻找早期测试者，Nomic.ai 成员探索了[用于自动网站链接的 Nomic embeddings](https://huggingface.co/blog/JLouisBiz/semantical-website-links)，以创建互连的文档网络。

---

# PART 1: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布六项新功能！**：Perplexity AI 宣布了六项新功能，包括 **Android Draw to Search**、**欧冠 (Champions League) 集成**、**语音搜索 (Voice Search)**、**Box 和 Dropbox 连接器**、**Perplexity Finance 时间对比**以及 **Perplexity Telegram Bot**，详情见其 [更新日志](https://www.perplexity.ai/changelog/april-11th-2025-product-update)。
   - 此次更新旨在增强各平台用户的搜索和自动化能力。
- **Sonar 模型在 Search Arena 中击败 Gemini**：Perplexity AI 的 **Sonar-Reasoning-Pro-High** 模型在 **LM Arena 的 Search Arena** 中与 **Gemini-2.5-Pro-Grounding** 并列第一，得分分别为 **1136** 和 **1142**。
   - 根据 [Perplexity 博客](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution)，由于搜索深度显著更高（引用了 **2-3 倍的来源**），**Sonar** 模型表现优于 **Gemini** 模型。
- **Perplexity 关注直播录像、API 开关和 ComfyUI 集成**：在用户询问后，团队确认 **Perplexity 直播** 的录像将在网上公开，详见 [X.com](https://x.com/aravsrinivas/status/1910741305212485915?s=61)。
   - 此外，一名成员暗示了 **Perplexity ComfyUI 集成**，并询问类似于 **"Social" 开关** 的 **API 开关** 是否即将推出。
- **用户被假播放按钮误导**：General 频道的成员承认被一个**假播放按钮**骗了。
   - 一位成员表示“那个假播放按钮骗到我了”，另一位回复说“下意识就点进去了”。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Google 削弱了 Gemini 2.5 Pro 的 Tool Calling 功能**：成员报告称 **Google 削弱了 2.5 Pro 的 Tool Calling 功能**，由于存在大量 Bug，2.5 Pro 现在无法执行工具调用。
   - 成员认为这种 **nerfing（削弱）** 可能与成本有关。
- **GPT 4.1 在 Windsurf AI 上线**：**GPT 4.1** 在接下来的 7 天内可在 [Windsurf](https://windsurf.ai) 中免费使用，提示用户进行尝试。
   - 一些用户对 OpenAI 选择与 **Windsurf** 而非 **Cursor** 合作发布感到惊讶。
- **RooCode 脱颖而出成为顶尖编程 IDE**：在一些推荐下，部分成员尝试了 **RooCode**，称其绝对优于 Cline，且极有可能是目前最好的编程 IDE。
   - 缺点包括 **GitHub Copilot** 集成到 RooCode 中存在速率限制且不稳定。
- **GPT-4.1 胜过 GPT-4o Mini**：成员认为 **Quasar/Optimus** 是最近发布的 **GPT-4.1** 和 **GPT-4.1 Mini** 模型的测试版本，这些模型并不像最初希望的那样具有突破性或令人印象深刻。
   - **GPT-4.5** 模型已被弃用，其改进已合并到 **4.1** 模型中。
- **GPT 4.1 融入 GPT4 Turbo**：成员报告称 **GPT 4.1** 无法通过 API 获取，其在指令遵循、编程和智能方面的改进正逐渐整合到最新版本的 **GPT 4o** 中。
   - 一些成员确认 **GPT 4.1** 的改进已合并到 **GPT 4o** 模型中，并可以在 [OpenAI 官网](https://openai.com)上访问。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 最新更新支持 GPT-4.1**：[Aider v0.82.0](https://aider.chat/HISTORY.html) 获得了对 **GPT 4.1**、**Gemini 2.5 Pro 架构模式**以及 **Fireworks AI** 模型 **deepseek-v3-0324** 的支持，同时新增了 `patch`、`editor-diff`、`editor-whole` 和 `editor-diff-fenced` 编辑格式。
   - 该版本还支持 **`xai/grok-3-beta`**、**`openrouter/openrouter/optimus-alpha`**，以及 **`grok3`** 和 **`optimus`** 等别名，以替代 **OpenRouter** 现已退役的 **Optimus** 和 **Quasar** 免费 Alpha 端点。
- **Discord 用户讨论是否为 Aider 开设闲聊频道**：成员们在 Aider Discord 服务器是否有必要开设闲聊频道（off-topic channel）上存在分歧，讨论如何在“寻找乐趣”与保持主频道专注之间取得平衡，并请求 Paul G. 改变主意。
   - 成员们无法就该专注于 Aider 还是该有一个讨论笑话的地方达成一致。
- **Claude 3.7 胜过 Gemini 2.5**：成员报告称 **Gemini 2.5 Pro** 在处理长上下文和代码块补全时表现吃力，但可以通过“发誓（swear oath）”来改善；而 **Claude 3.7** 在自然写作和特定任务中表现更好。
   - 社区成员称赞 **Claude 3.7** 的自然语言能力，其他人发现这些模型在消除过度注释行为方面表现出色。
- **用户寻求在 Aider 中复制 Cline 的 Memory Bank 工作流**：一名成员询问如何在 Aider 中复制类似 **Cline 的 Memory Bank 工作流**，方法是将 `plan.md` 添加到聊天中，然后在“执行下一步”和“标记该步骤已完成”之间交替操作。
   - 这样做的目的是帮助创建任务列表，以便 Aider 可以与用户一起逐一完成任务。
- **成员分享 Prompt Engineering 资源**：一名成员发布了 **Kaggle** 的 [Prompt Engineering 白皮书](https://www.kaggle.com/whitepaper-prompt-engineering)链接，其他成员分享了 [GPT-4.1 Prompting 指南](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)。
   - 该 Prompting 指南旨在帮助用户优化与 **GPT-4.1** 模型的交互。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 价格回归正常**：OpenRouter 开始对较长的 **Gemini** 提示词收取正常费用，影响超过 **200k** 的 **Gemini 2.5** 提示词和超过 **128k** 的 **Gemini 1.5** 提示词，价格与 **Vertex/AI Studio** 的费率保持一致。
   - 这一变化是由于 **Gemini 2.5** 的使用量激增，从而结束了长上下文提示词的 **50% 折扣**。
- **免费模型涌入 OpenRouter！**：OpenRouter 新增了六款免费模型，包括针对角色扮演微调的 [**QwQ-32B-ArliAI-RpR-v1**](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free)、长上下文代码生成模型 [**DeepCoder-14B-Preview**](https://openrouter.ai/agentica-org/deepcoder-14b-preview:free)，以及混合专家（Mixture-of-Experts）VLM [**Kimi-VL-A3B-Thinking**](https://openrouter.ai/moonshotai/kimi-vl-a3b-thinking:free)。
   - 这些模型提供了从角色扮演到代码生成的多种能力，扩展了平台上的可用选项。
- **NVIDIA Llama-3 变体免费开放！**：新增了三款来自 **NVIDIA** 的 **Llama-3** 变体（[**Nano-8B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free)、[**Super-49B**](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free)、[**Ultra-253B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)），它们针对推理、工具使用和 RAG 任务进行了优化，并拥有高达 **128K** token 的扩展上下文窗口。
   - 用户已开始测试这些模型的相对性能。
- **GPT-4.1 模型：下一代迭代**：GPT-4.1、GPT-4.1-mini 和 GPT-4.1-nano 模型在 OpenRouter 上线，全量模型针对长上下文推理进行了优化。
   - 用户注意到 *GPT-4.1 和 4.1 mini 在某种程度上表现相当，至少在 spaceship prompt 上是这样*，但也有人正在进行彻底的测试以衡量性能。
- **Skywork-OR1 系列释放推理能力**：推出了 **Skywork-OR1** 模型系列，其中 **Skywork-OR1-Math-7B** 擅长数学推理，而 **Skywork-OR1-32B-Preview** 在数学和编码任务上的表现足以与 **Deepseek-R1** 媲美。
   - 这两款模型分别基于 **DeepSeek-R1-Distill-Qwen-7B** 和 **DeepSeek-R1-Distill-Qwen-32B** 训练而成。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **PDF 转网站功能走红**：一位成员注意到将 **PDF** 转换为网站的便捷性。
   - 该解决方案被认为是一个*极佳的案例*。
- **DeepSeek V3 蓄势待发**：一位成员询问了 **Manus** 的项目创建能力，但结论是 **Manus** 目前仅提供 **DeepSeek R1**，预计在几个月内会升级到其顶尖模型。
   - 另一位成员对 **Qwen** 最近的代码能力表示不屑。
- **网络安全职业组合考量**：一位成员考虑转行，但鉴于其编码熟练度，决定继续留在网络安全领域。
   - 会议还讨论了*量子技术对网络安全*的潜在影响。
- **机构选择 GCP 而非 Firebase**：一家机构因基础设施的成本效益选择了 **GCP**，另一位用户提交了一份 **40 页的分析报告**，支持从 **Microsoft** 转向 **GCP**。
   - **Google** 获得了 5 分中的 **4.7** 分，而 **Microsoft** 得分为 **4.4**。
- **Gemini 2.5 Pro 处理数据能力极强**：一位用户称赞 **Gemini 2.5 Pro** 的数据处理实力优于 **ChatGPT**，这促使他们取消了 **Perplexity** 的订阅。
   - 用户观察到 **Gemini 2.5 Pro** 每个任务所需的额度更少，并且随着 **Claude max pro** 的发布和成本的降低，它正在不断进步。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma GRPO 磨合期**：成员们讨论了在 **GRPO** 中使用 **Gemma 4B** 还是 **Gemma 1B**，并澄清虽然两者都可以进行 **GRPO**，但 **4B 版本** 无法在 Colab 上运行。
   - 针对 15k 行数据集设置合适的训练步数引发了关注，建议检查 *batching（批处理）、epochs（轮数）和 gradient accumulation（梯度累积）如何协同工作*。
- **AMD GPU Anaconda**：用户正努力让 **Unsloth** 在 **AMD GPU** 上运行，由于 **Unsloth** 最初专注于 **NVIDIA**，因此遇到了 *NotImplementedError*。
   - 核心问题在于 **BNB** 无法正确构建，即使 **AMD torch.cuda.is_available()** 返回 True。
- **LM2 Memory：Gemma 的收益**：进行了将 **LM2 的 memory units** 直接集成到 **Gemma 3** 中的实验，以提升 Prompt 之间的上下文感知能力。
   - 对模型层进行 Monkey patching 以挂载 memory 导致了量化方面的挑战（为了降低硬件要求），其中一名成员在 **gma6** [[https://github.com/jagoff2/gma6]] 中对每 6 层进行了挂钩。
- **DeepSeek 的推理见解**：[DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) 引发了关于小型供应商推理性能预期的讨论。
   - 有人担心供应商可能会以次优配置运行 *vllm serve*，从而在提供 **DeepSeek R1** 服务时影响模型性能。
- **Apple 的 Cross Entropy 深度解析**：分享了一篇解释 **Apple's cut cross entropy** 的深刻文章，将 **transformers** 框架化为 *for 循环上的顺序 ML 分类任务* ([zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/1354843933))。
   - 由于原始链接的访问问题，提供了一个替代的 [GitHub repo](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 直播即将开始！**：OpenAI 宣布了定于 **太平洋时间上午 10 点** 的直播 [<t:1744650000:f>](https://openai.com/live/)，社区成员正推测 **API** 中可能发布 **GPT-4.1**。
   - 公告特别标记了 **GPT** 角色，暗示可能专注于 **GPT models** 或相关更新。
- **视频领域的 Veo 2 对决 Sora**：成员们将 Google 的 **Veo 2** 与 **OpenAI** 的 **Sora** 在视频生成方面进行了对比，一些人更青睐 **Veo 2** 更 *自然的 24 fps 视频*。
   - 一位成员指出，过度平滑的帧率在他们大脑中会被识别为 *即时的 AI 生成内容*，另一位成员成功越狱（jailbreak）了模型来为《狮子王》制作动画。
- **Memory 控制功能详解！**：[OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) 的详细信息展示了 **ChatGPT memory 的控制功能**，采用了已保存记忆和聊天历史引用的双层架构。
   - 此次更新允许用户通过启用或禁用 memory 和聊天历史记录来控制和编辑偏好。
- **用户与 Prompt 默认值作斗争！**：一位用户报告称，他们两个月前构建的 **ChatGPT agent** 现在正严厉地忽略 Prompt 默认设置（如表格格式或列规范），尽管庞大的 Prompt 并没有任何改动。
   - 用户请求针对模型忽略过去已确立参数的问题提供见解或解决方案。
- **通过 Prompt 调整让图像更清晰！**：一位用户询问如何消除图像生成中的 *模糊感（smudged look）*，另一位用户建议这取决于 Prompt，并分享了引导模型的 [prompting techniques](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&) 。
   - 此外，一位用户通过向 **ChatGPT** 提供所需字体的截图，成功在图像中生成了特定字体。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **OpenAI 发布新模型，中国对此做出反应**：[OpenAI 发布了新模型](https://openai.com/blog/new-models-and-api-updates-available-today)，引发了与 **DeepSeek**、**Claude**、**GPT** 和 **Gemini** 的比较。
   - 一位成员观察到中国在这个领域*表现不佳*，而另一位成员则评论说美国*像往常一样低估了一切*。
- **Claude 3.7 成为 Cursor 的首选**：成员们发现 **Claude 3.7 Sonnet** 是 Cursor 中的最佳选择，在稳定性、one-shot 能力和代码质量方面优于 Gemini 和 Google 的模型。
   - 有人补充说 [Claude 模型正在进步](https://www.anthropic.com/news/claude-3-haiku)，*对我来说，越老的模型越聪明*。
- **Gemini 2.5 在 UI 方面表现惊人**：**Gemini 2.5 Pro** 因其惊人的 UI 设计能力而受到认可，成员们分享了其独特输出的示例，并能保持上下文。
   - 一位用户评论道，*Gemini 的 UI 修改简直疯了*。
- **Windsurf 没落，用户更倾向于 Cursor**：用户报告了 **Windsurf** 的可靠性问题，称其过度承诺，导致一些人建议在正确使用的情况下选择 **Cursor**。
   - 一位用户调侃道：*欢迎来到 shit surf*。
- **社区期待 GPT-4.1**：社区正在讨论即将发布的 **GPT-4.1** 以及如何开始使用它，并提到了 4.5 预期的弃用。
   - 成员们预计 *每个人都会开始转向 4.1；2.5 资源池会清空，Claude 3.5 3.7 也会清空一点，直到 4.1 的配额用完，然后在新模型上重复同样的过程*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 取消了多模型联动的魔力**：用户哀叹 LM Studio **0.3** 版本中删除了**多模型提示（multi-model prompting）**功能，该功能此前在 **0.2** 版本中可用。一位用户评论说，使用 [LM Studio](https://lmstudio.ai/) 比较模型是 *"世界上最好的事情"*。
   - 他们正在寻找模型比较的替代方案。
- **离线运行 LM Studio 需要手动处理运行时**：要在离线 PC 上运行 LM Studio，用户必须手动传输位于 `C:\Users\jedd\.cache\lm-studio\extensions\backends` 的 **LM runtimes**。
   - 有关通过 localhost 导入模型的文档可以在[这里](https://lmstudio.ai/docs/app/basics/import-model)找到。
- **Python 炼狱：示例从 LM Studio 服务器文档中移除**：用户注意到 LM Studio 的服务器部分缺少 Python 示例，并正在请求 [Python 示例](https://lmstudio.ai/docs/app/api/endpoints/openai)。
   - 有人分享了一个替代方案：[lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples)。
- **Threadripper 在 Token 效率上完胜 Xeon**：一位成员表示，纯粹出于成本考虑，**Threadripper** 或 **Epyc** 芯片比双路 **Intel Xeon w7-3565X** CPU 具有更好的性价比（dollars per token）。
   - 据指出，在 **Threadripper 7xxx** 上，当 llama.cpp 使用超过 20 个线程后，性能几乎没有差异，但当超过 64 个线程需要跨 CPU 调用另一个 CPU 时，性能会变慢。
- **ROCm 的困境：重新考虑 RX 6700 XT 的建议**：一位成员询问购买 **AMD Radeon RX 6700 XT** 来运行 **Gemma**，以及 **ROCm** 是否像 **CUDA** 一样强大。
   - 回复是 *6700XT 不支持 ROCm*，运行 **Gemma 12b** 至少需要 16GB 的 VRAM，因此如果必须使用 AMD 显卡，建议攒钱购买拥有 24GB VRAM 的 **7900XT**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLMs 与概率 FSA 的比较**：有人认为 LLM 近似于**概率有限状态自动机 (FSA)**，这暗示了其在扩展上的障碍和数学方面的弱点；有一位成员反驳说这种类比没有太大意义。
   - 成员们补充说，这种比较类似于说人类“近似于猴子”，削弱了该类比的分量。
- **AlphaProof 获得银牌**：成员们观看了一段关于使用 AI 辅助证明的[视频](https://www.youtube.com/watch?v=e049IoFBnLA)，并总结道 **AlphaProof** 在没有使用任何人类知识的情况下获得了银牌。
   - 另一位成员指出，这一信息是基于公司的说法，并表示“AlphaProof 在没有使用任何人类知识的情况下获得了银牌（据他们所说）”。
- **Brave Search API 受到关注**：成员们建议将 **Brave Search API** 作为 Agent 流的一个很好的替代方案，并强调即使在免费层级也有良好的体验。
   - 有人提到其 **AI 总结器**比 **OpenAI** 的网络搜索 API 更便宜。
- **生成式 AI 使用案例数据存在偏差？**：成员们正在讨论 [The 2025 Top-100 Gen AI Use Case Report](https://learn.filtered.com/hubfs/The%202025%20Top-100%20Gen%20AI%20Use%20Case%20Report.pdf)，认为由于 **Reddit** 是唯一的数据源，数据可能存在偏差。
   - 成员们还指出，**Character.AI** 拥有 **2800 万用户**，但在 **ML** 圈子中却很少受到关注。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 测试 Llama 4 Maverick 和 Scout**：Hugging Face 迎来了 [**Llama 4 Maverick** 和 **Llama 4 Scout**](https://huggingface.co/blog/llama4-release)，测试显示了它们在 **DABStep 基准测试**中的表现。
   - 据报道，在此过程中对 **Claude 3.7 Sonnet**、**Gemini 2.5 Pro**、**Llama 4 Maverick** 和 **Llama 4 Scout** 都进行了测试和比较。
- **HF 模型 404 错误困扰用户**：用户报告在尝试访问 Hugging Face 模型时出现大范围 **404 错误**，导致其应用宕机，详见[此链接](https://discuss.huggingface.co/t/404-error-model-xlabs-ai-flux-realismlora-does-not-exist/150363/5)。
   - 一位成员标记了一名特定的 HF 员工，提到这个 404 错误已经持续了大半天。
- **用户痴迷于 Ollama**：成员们讨论了使用 **Ollama** 在本地运行模型，分享了下载和运行特定模型（如 `qwen2.5-coder:32b`）的命令，以此作为 API 限制模型的替代方案。
   - 一位成员提供了一个代码片段，演示了在初始化 `CodeAgent` 时如何指定 **Ollama 提供商**，并使用本地托管的模型，如 `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF`。
- **新的 Deep Search Agent 寻求早期测试者**：一个使用 *smolagents* 构建的专注于 **Deep Search** 的新 Agent 已经发布，正在 [agent.galadriel.com](https://agent.galadriel.com/) 招募早期测试者。
   - 欢迎提供反馈，并请向产品团队提出问题和想法。
- **Agent 执着于教皇的年龄**：一位用户报告说，当在本地运行 `llama3`、`deepseekr1:8b` 和 `qwen2.5-coder:latest` 等模型时，他们的 **Agent** 莫名其妙地执着于寻找**教皇的年龄**并将其平方得到 **0.36**。
   - 怀疑该问题源于 **smolagent** 默认 Agent 工具提示词中的硬编码示例，因为在使用 **HfApiModel** 时并未出现此问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **模型表现出惊人的相似性**：一位成员注意到，使用[这个脚本](https://github.com/pwspen/lmsim)发现不同模型的序列在 post-MLP 隐藏状态余弦相似度上表现出惊人的相似性。
   - 小型模型更多地按类型而非颜色分组，而大型模型在按颜色排序方面表现得更一致。
- **禁止 Batch 重复！**：一位成员建议不要在 minibatch 中重复数据，理由是这可能会导致 *重大问题*。
   - 他们分享了关于认知科学和 ML/AI 领域的调查性信息分析，促进了跨学科的见解，并将这些见解传达给不同群体。
- **多 Token 预测论文**：一位成员寻求关于 LLM 在推理期间进行多 Token 预测的论文，另一位用户推荐了 [DeepSeek v3](https://openreview.net/forum?id=pEWAcejiU2)。
   - 另一位用户指出了[这篇论文](https://arxiv.org/abs/2401.10774)，并回忆起几年前看过 Meta 的一篇相关论文。
- **AI “研究”受到审查**：成员们对以研究形式出现的 **AI 生成内容** 的兴起表示担忧，其特点通常是 *虚造的术语* 以及 *缺乏与合法研究思路的一致性*。
   - 建议包括 **封禁隐藏 AI 使用的恶意用户**，以及对表现出缺乏经验的 **善意用户进行长期禁言**。
- **长度外推差异**：成员们讨论了 **长度外推（length extrapolation）** 的挑战，指出模型在超出其训练序列长度后，往往 *无法持续降低 Token loss*，如[这张图表](https://cdn.discordapp.com/attachments/747850033994662000/1361359728147431685/Screenshot_2025-04-14_at_16.17.22.png?ex=67fe788c&is=67fd270c&hm=4fe0f240d28501a17e80c46f6c0848297dd2361ec50593f31cc697d50bccd0e5&)所示。
   - **NoPE + SWA** 和 **ssmax** ([Super Scaling Max Activation](https://arxiv.org/abs/2501.19399)) 等技术被提及作为潜在的解决方案。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Karpathy 尝试让 ChatGPT 尴尬**：一位用户分享了[一个 Prompt](https://x.com/karpathy/status/1910734302931017812)，询问 *“你所知道的关于我最尴尬的事是什么？”*，让 **ChatGPT** 陷入了窘境。
   - 该用户想看看 **ChatGPT** 是否能通过多轮询问给出诚实且直接的回答。
- **Thinking Machines 种子轮达 20 亿美元**：据 [Fortune 文章](https://fortune.com/2025/04/10/mira-murati-2-billion-seed-raise-ai-boom-economic-chaos/)报道，**Thinking Machines** 显然正在进行一轮 **20 亿美元的种子轮融资**，由 **Alec Radford** 担任顾问。
   - 一位用户发布了来自 [Epoch AI](https://x.com/epochairesearch/status/1910788295405252916) 的 *一张不错的图表*，展示了这次融资。
- **DeepSeek 开源推理引擎**：**DeepSeek** 已开源其推理引擎，[GitHub 仓库](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md)已可供查阅。
   - 成员们想知道谁想聊聊 **DeepSeek** 的开源举动。
- **Quasar 发布会观看派对正在进行**：Latent Space 正在 [Discord 活动页面](https://discord.gg/rPJq8UU2?event=1361376118510321724)举办另一场 **Quasar 发布会** 观看派对。
   - 在 **OpenAI Quasar** 发布会观看派对期间，成员们讨论了 **GPT-4.1** 的特性，包括其相对于 Claude 极具竞争力的价格，以及长输入上下文的固定定价，参考了[价格文档](https://platform.openai.com/docs/models/gpt-4.1)。
- **Agent 定义的 Vibe Check**：成员们辩论了 *Agent* 的定义，一位成员建议现在的定义是：*一个 LLM 调用一个工具*，而另一位成员展示了关于自我改进 Agent 的 [Figma 画板](https://www.figma.com/board/aCaUWEr039dHmpW9ssGJmK/self_improving_agents?node-id=137-796&t=zsQXjScFlAekKtEd-1)。
   - 有人建议：*Agent 就是你在开会无聊时凭感觉（vibe code）写出来的东西*。

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 的 Latent Space 导致非确定性**：一位成员表示，*Latent Space 的变异性*导致无法每次生成相同的输出，从而使得每次根据输入生成的內容具有随机性，因为 **NotebookLM** 并非设计为确定性系统。
   - 他们提醒不要期望 **NotebookLM** 的表现能像更昂贵、更专业的系统那样。
- **NotebookLM 改变教育体验**：一位成员正在课堂上使用 **NotebookLM**，上传幻灯片和资料，创建笔记、带有测试题的学习指南、术语表、思维导图和音频概览，然后分享给学生以帮助他们准备考试。
   - 他们还提到让学生分组创建自己的 **NotebookLM**。
- **用户渴望 Gemini Education Workspace**：一位成员询问其他人是否通过 **Education Workspace** 使用 **Gemini**，并对在 Workspace 中成功使用 **Gemini** 的学区和部门表示关注。
   - 他们指出在澳大利亚新南威尔士州（NSW），目前还无法使用 **Gemini**。
- **猫主人想要为宠物提供聊天机器人**：一位经营大型糖尿病猫主人支持小组的成员希望为成员提供其文档的**对话式界面**，包括视频内容，并支持法语。
   - 他们希望成员能够提出问题，并根据文档获得答案，同时附带相关文档的链接以供阅读。
- **NotebookLM “Discover” 功能引发关注**：一位用户对 **NotebookLM** 中新的 **“Discover sources”** 功能表示非常满意，称其为 *“我想要的一切”*。
   - 该用户还期待更多**音频概览风格（audio overview flavors）**，并赞扬了 Grace 的播客。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 4 消耗大量 GPU 小时数？**：成员们注意到 **Meta** 的 **Llama 4 Maverick** 使用了 **2.38M GPU 小时**，而 **Llama 4 Scout** 使用了 **5.0M GPU 小时**，这与训练 **Deepseek V3** 的耗时相当。
   - 一些人质疑与针对人类偏好进行微调的模型进行比较是否公平，而另一些人则认为 **LeCun** 的参与可能解释了这一点。
- **DeepCoder 提供顶尖编程性能**：一位成员分享了一篇关于 **DeepCoder** 的 [VentureBeat 文章](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/)，强调了其高效的 **14B** 参数开源模型和增强的 **GRPO** 算法。
   - 该模型结合了**离线难度过滤**、无熵损失、无 **KL Loss** 以及来自 **DAPO** 的超长过滤，尽管训练时使用 **32K**，但可以泛化到 **64K** 上下文。
- **Nvidia UltraLong 模型吞噬上下文**：[Hugging Face 集合](https://huggingface.co/collections/nvidia/ultralong-67c773cfe53a9a518841fbbe)中展示的 **Nvidia UltraLong-8B** 模型旨在处理高达 **4M tokens** 的序列，基于 **Llama-3.1** 构建。
   - 这些模型结合了持续预训练（continued pretraining）和指令微调（instruction tuning），以 **4M** 序列长度和 **2** 的全局 Batch Size 训练了 **150** 个迭代。
- **GPT-4.1 基准测试更好，定价令人困惑**：成员们讨论了 **GPT-4.1** 的定价和基准测试，指出其 [基准测试结果优于](https://openai.com/index/gpt-4-1/) 之前的版本，但定价和模型版本命名令人困惑，尤其是该新模型在 **GitHub Copilot** 中的可用性。
   - 有推测称 **4.1-nano** 可以与优秀的 **14B** 模型竞争，并存在开源的可能性。
- **Llama 4 Scout 的 H100 训练显示 Loss 增加！**：一位成员观察到在 **H100** 环境下训练 **Llama 4 Scout** 时，第 **1** 和第 **2** 个 Epoch 之间的 Loss 从 **1.9011** 上升到 **2.3407**。
   - 该用户表示担忧，因为 Loss 没有像预期那样下降，即使使用了两块 **H100** GPU；另一位成员建议 *无论任务是什么，你至少应该处理 10M 参数的模型*。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Graphlit 为内容打造 MCP Server**：[Graphlit](https://github.com/graphlit/graphlit-mcp-server) 正在为 Reddit 和 Quora 构建一个 **MCP server**，并表示如果需要可以添加 Quora 数据摄取功能。
   - 目前已经存在一些针对 Reddit 的项目，例如[这个仓库](https://github.com/hawstein/mcp-server-reddit)。
- **Agency Dev Kit 与 MCP 竞争**：成员们讨论了 Google 的 **ADK 和 A2A** 及其与 **MCP** 的相似性，以及它们在 Agent 互联网中潜在的核心地位。
   - 一位成员分享道，关于非 MCP 技术讨论目前还没有官方共识，但如果至少与 AI/ML/MCP 有一定相关性，那么应该没有问题。
- **无 Function Calling 能力的模型获得 Block 调整**：Block 正在尝试对缺乏 function calling 能力的模型进行实验，看看是否可以调整其输出以适配 Agent，[这篇博文](https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim)探讨了如何在不使用辅助模型的情况下通过 XML 输出实现这一点。
   - 团队正在权衡延迟成本与使用辅助模型进行解析的收益，同时也担心会话时间变长以及模型遵循 XML 格式的能力，并可能使用 *local model*，但担心会带来更多 *overhead*。
- **MCP 工具辅助 Copilot Client 调试**：[synf](https://github.com/strowk/synf) 和 [mcptee](https://github.com/strowk/mcptee) 帮助成员在测试 Copilot client 时发现并修复 bug，因为该客户端在处理长上下文和更多工具时可能会遇到困难。
   - 一位成员在构建时考虑到了高性能硬件，因为*多次 API 调用总是比单次调用慢*。
- **Paprika 食谱应用获得美味的 MCP Server**：为 **Paprika 食谱应用**的用户创建了一个 MCP server，这样 Claude 就可以通过[这个 GitHub 仓库](https://github.com/soggycactus/paprika-3-mcp)自动将食谱保存到 Paprika 中。
   - 未提供更多信息。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 同步指南逐渐明朗**：一位成员询问了 Python/PyTorch 模型中的 **CUDA** 参考资料，另一位成员分享了他们最近关于该主题的 [GTC 演讲](https://docs.google.com/presentation/d/1sipZ_sqdwJapHQAr23yBow43pF40lMZu/view?usp=sharing)，该演讲也可以在 [nvidia.com](https://www.nvidia.com/en-us/on-demand/session/gtc25-s71946/) 上找到。
   - 演讲建议 *custom ops* 和 *load inline* 应该能解决大部分问题，同时还在进行缩短编译时间的工作；一位成员发现了演讲中提到的 **Stephen Jones** 的视频，并表示“假期结束了，演讲再次开始”。
- **Hilbert 曲线提升 GEMM 性能**：一位成员分享了一个 [GitHub 仓库](https://github.com/lawmurray/gpu-gemm)，展示了使用 **Hilbert 曲线** 实现的 GEMM，以及 [针对 cuBLAS 的基准测试](https://indii.org/blog/gpu-matrix-multiply/)。
   - 基准测试表明，随着矩阵尺寸的增加，**Hilbert 曲线** 变得更加有效。进一步的讨论揭示了 **Hilbert 曲线** 虽然是最优的，但在硬件效率上并不高，建议 **Morton 排序** 是一个更好的实际权衡方案，并指向了一篇比较两者的 [博客文章](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304)。
- **`memcpy_async` 对齐加速性能**：在切换到 `cuda::memcpy_async` 后，一位用户报告了性能下降，有建议称这是一个协作式 API，意味着所有线程必须传递相同的指针和对应于整个内存块的大小，参考了 [官方 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-barrier)。
   - 还有建议指出 `memcpy_async` 的潜在问题包括 Shared Memory 地址的对齐以及指令周围的条件判断，这可能会阻碍合并内存访问（coalesced memory access），参考了一篇 [论坛帖子](https://forums.developer.nvidia.com/t/coalesced-and-conflict-free-memory-access-using-cuda-memcpy-async-cp-async/306460/6)。
- **分布式系统的内存分析难倒初学者**：一位工程师寻求关于在拥有 **8 个节点**（每个节点 **8 个 GPU**）的 **SLURM 集群**上进行分布式训练模型内存分析的建议。
   - 此外，一位工程师询问了 ATen 的 `attention.cu` 中特定行所指向的实现（[GitHub 链接](https://github.com/pytorch/pytorch/blob/101c4f482a4019896ca18184233bd27a758648bf/aten/src/ATen/native/transformers/cuda/attention.cu#L662)），旨在了解 torch/CUDA 如何处理 batch 中的单个用户操作数 `[dHead x K-cache-length]`。
- **Metal 内存之谜已掌握**：一位成员发现，Metal 中全局内存合并的矩阵乘法实现所使用的内存仅为朴素版本的一半，并使用 [此 CUDA MMM 实现](https://siboehm.com/articles/22/CUDA-MMM) 作为参考进行测试。
   - 一种解释认为操作系统以页（pages）为单位提取数据，而非合并访问会导致页使用效率低下，即提取的数据中只有一小部分被实际利用；其他人指出 **M 系列芯片具有 Unified Memory**，这应该会消除 CPU 和 GPU 之间的分页。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embeddings 编织网站**：一位成员报告了使用 **Nomic embeddings** 自动链接网站页面的成功案例，大幅减少了手动工作，详见 [semantical-website-links 博客文章](https://huggingface.co/blog/JLouisBiz/semantical-website-links)。
   - 他们正在探索自动识别关键词并将其链接到 embeddings 的方法，从而创建一个互连的、自我更新的文档网络，如[这段 YouTube 视频](https://www.youtube.com/watch?v=xk2VGnLYAkA)中所讨论的。
- **GPT4All 的 Token 之争**：一位尝试使用 **GPT4All** 模型生成长篇剧本的用户遇到了**响应长度限制**，尽管尝试使用了 **GPT4All** 内部的模型。
   - 建议包括调高 **Max Tokens** 设置并拆分故事，但该用户仍在寻找能够处理更长输出的模型。
- **HuggingFace 故事模型**：在 **HuggingFace** 上标记为 'story' 的模型在生成较长响应方面表现出色，这让一位成员感到非常高兴。
   - 然而，有人建议要谨慎，因为其中许多模型可能是专有的，可能会限制其作为自由软件的使用。
- **破解 Chat Template 位置**：一位成员寻找 Llama3.2、Llama3.1、Aya-23 和 KafkaLM-8x7b-German-V0.1 等模型的 **chat templates** 所在地。
   - 建议他们查看模型作者在网站、GitHub 或 **Hugging Face** 上发布的版本，特别关注 `tokenizer_config.json` 文件中的 `chat_template` 条目。
- **上下文长度限制创意**：模型通常在 **2048 到 8192 tokens** 之间的上下文长度上进行训练，虽然 RoPE 和 Yarn 可以扩展这一范围，但响应质量往往在超出原始范围后大幅下降。
   - 虽然取决于训练数据集和微调，但响应长度可以通过 prompting 进行调整，例如明确要求模型写得“非常非常长”。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Origins 演变为 Lifetimes**：Mojo 中的术语 **`Origin`** 已更名为 **`Lifetime`**，这可能会让熟悉 Rust 生命周期概念的人更容易理解，详见[文档](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and)。
   - Mojo 扩展了值的生命周期，以匹配持有它们的任何引用；相反，必须跟踪每个引用的 origin 以确定值的扩展和释放，这与 Rust 基于作用域的生命周期跟踪不同。
- **VSCode 丢失 Mojmelo**：用户报告称，尽管手动安装了 **`mojmelo`** 模块，但 Mojo VSCode 扩展仍无法检测到它，原因是该扩展使用了其自带的 Mojo 安装。
   - 解决方法包括手动配置扩展，使其使用本地模块库进行 intellisense。
- **Mojo PEP 正在制定中**：受 **Python PEP** 的启发，一位成员建议为 Mojo 建立类似的系统来跟踪变更，另一位成员指出了 [Mojo 现有的提案系统](https://github.com/modular/max/tree/main/mojo/proposals)。
   - 讨论显示了社区对以结构化方式管理和沟通语言演进的兴趣。
- **Negative Bounds 现已推出**：**Negative bounds** 是一种反转命名集的方法，通常与 **marker traits** 一起使用，以定义类型集的逆集，例如 `!Send` 表示线程局部变量。
   - 例如，该 marker trait 表示在线程间移动是不安全的。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4.1 API 获得首日支持**：**OpenAI** 在 **API** 中发布了 **GPT-4.1**，通过 `pip install -U llama-index-llms-openai` 即可立即获得支持，详情见[此处](https://t.co/JPEX3KAoWS)。
   - 基准测试显示，**GPT-4.1** 相比 **4o** 提升了约 **10%**，在现有的 Agent 方案上提升了约 **2%**。
- **LlamaParse 在文档解析方面表现出色**：**LlamaParse** 为包含图像、表格和图表的文档提供更高的解析质量，超越了 **SimpleDirectoryReader** 等基础读取器。
   - 一位成员强调，解析文档的*质量*是 **LlamaParse** 区别于 **SimpleDirectoryReader** 的关键。
- **开源 LLM 应对 Agent 任务**：虽然较小的开源 LLM 在 Agent 工作流中表现挣扎，但较大的模型如 **Llama3**、**Llama 3.1**、**Llama 3.2:3b** 或 **Mistral** 被证明更有效，尤其是与 **Ollama** 配合使用时。
   - 一位成员提到成功使用 *llama3.2:3b* 来满足其 Agent 需求。
- **.query 对话不保留历史记录**：澄清了 `Char .query` 是**无状态的 (stateless)**，不保留任何聊天历史，因此不存储聊天日志。
   - 寻求记忆持久化的成员建议考虑使用 Agent。
- **AI 评估模型评估**：一篇研究论文 [Benchmarking AI evaluation models](https://arxiv.org/abs/2503.21157v3) 在 **6 个 RAG 应用**中评估了 **LLM-as-a-judge**、**HHEM** 和 **Prometheus** 等模型。
   - 研究发现，这些评估模型在真实场景中表现得*出奇地好*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVIDIA 发布全新 Video Codec SDK**：**NVIDIA** 发布了 [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip) 以及 [GitHub 上的示例](https://github.com/NVIDIA/video-sdk-samples)，一位用户警告不要提交 AI 生成的 PR。
   - 该用户威胁要关闭提交并封禁屡教不改者，强调了理解内容的重要性。
- **TinyGrad 第 66 次会议议题**：第 66 次会议定于周一举行，涵盖公司更新、**chip!**、fast python、**bert**、**mlperf**、scheduler、driver、**webgpu**、retinanet、torch frontend multi gpu、云规模 uuuvn 事务以及 bounties。
   - 一位成员表示在看到评论后理解了 **Index Validation PR** 的要求，并预计在第二天准备就绪。
- **Clang 标志静默调试输出**：一位成员建议使用 `-fno-ident` [clang 标志](https://xl0.github.io/tinygrad-notes/misc_1.html)，以防止额外的段（`.comment` 和 `.note.GNU-stack`）被添加到镜像中并污染 `DEBUG=7` 输出。
   - 此更改有助于保持调试输出更整洁、更易于管理。
- **新 TinyGrad 项目寻求协助**：一位新成员介绍自己，寻求第一个项目以获得 **tinygrad** 的实操经验，并被建议尝试一个[小额 bounty](https://xl0.github.io/tinygrad-notes/misc_1.html)。
   - 共享了一些有用的资源，包括 [tinygrad-notes](https://xl0.github.io/tinygrad-notes) 和 [mesozoic-egg 的 tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes)，以辅助其学习。
- **调试 Softmax 中的 NaN 问题**：一位成员报告在模型中调试 **NaN**，怀疑是 `softmax()` 问题，并指出在 `__call__` 过程中打印会导致优化器问题。
   - George Hotz 回应称打印不应该破坏程序，并建议提交 issue 以进行进一步调查。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 模型与 vLLM 集成**：成员们讨论了将自定义 **TorchTune 模型**与 **vLLM** 集成的方案，建议像处理 **HF 模型**一样对 **TorchTune** 微调后的模型进行推理，并提供了[相关教程](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-with-vllm)。
   - 对于未在 **HF** 上定义的自定义网络，需要在 **vLLM** 中定义模型，详见 [vLLM 文档](https://docs.vllm.ai/en/latest/contributing/model/index.html)，或者使用 **Torchtune 的 generate 脚本**作为替代方案。
- **Bitsandbytes 困扰 Mac 用户**：由于 `bitsandbytes>=0.43.0` 没有为 Linux 以外的平台提供二进制文件，在 macOS 上执行 `pip install -e '.[dev]` 会失败，但降级到 `bitsandbytes>=0.42.0` 可能会有帮助。
   - 根据 [bitsandbytes issue 1378](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180)，虽然 **0.42** 之前的版本标签有误，但至少这使其可以安装。
- **QLoRA 深入研究 sub-4-bit 量化**：成员们一直在寻找关于使用 **4-bit 以下量化**进行 **QLoRA 风格训练**的文献。
   - 该咨询专门针对 **QLoRA** 背景下与 **sub-4-bit 量化**技术相关的方法和发现。
- **奖励函数（Reward Functions）正在成型**：团队计划支持不同的**奖励函数**，实现细节正在讨论中，并且有人提出了关于以一种“奇怪的方式”定位奖励计算的问题。
   - 随后有人提到正在*收集重要函数列表*，请保持关注！
- **损失函数激增，实验蓬勃发展**：团队正在尝试**不同的损失函数**，旨在通过可能采用类似于 **DPO losses** 的协议，来避免 recipe 过度激增。
   - 目标是在支持必要损失和防止实验阶段过度泛化之间取得平衡，并承认在 **A100** 上测试期间使用了**硬编码的测试参数**。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Coral Chat 扩展至 Firefox**：**Coral Chat** 现在是 Firefox 侧边栏中的聊天机器人，可以通过将 `browser.ml.chat.provider` 设置为 [https://coral.cohere.com/](https://coral.cohere.com/) 进行配置。
   - 一位用户通过 [Imgur 链接](https://imgur.com/a/6zcTV8z)展示了该集成及其功能。
- **Next-Token 生成问题浮出水面**：一段 [YouTube 视频](https://youtu.be/VRqSJfdwbF4)强调了 **LLM** 在给定上下文中生成下一个 token 时可能面临的问题。
   - 讨论表明该问题在各种 **LLM** 中普遍存在。
- **Cohere Chat API 获得 Java 演示**：一位成员分享了一个展示 **Cohere Chat API** 的 Java 示例，特别是与 **command-a-03-2025** 模型交互的 `runInteractiveDemo()` 方法。
   - 该演示允许用户与 **Cohere AI** 交互，记录 prompt 和 API 交互以进行调试和优化。
- **Diofanti.org 揭露希腊政府支出**：[Diofanti.org](https://chat.diofanti.org) 是一个监控希腊政府支出的**开放数据平台**，为**透明度和问责制**提供工具。
   - **Aya 模型**是该平台聊天机器人的首选模型，支持透明度和问责制倡议。
- **LUWA App 定于 2025 年 4 月发布**：**LUWA.app** 是一个 **AI 驱动应用**的搜索目录，将于 **2025 年 4 月 25 日**上线。
   - 创建者正在探索 **Cohere** 及其 **LLM 模型**，以降低成本并增强应用性能。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda 提供无服务器 API 额度**：**Lambda** 正为每位参与者提供价值 **$100** 的无服务器 [Inference](https://lambdalabs.com/inference)（推理）**API 额度**，申请链接见[此处](https://forms.gle/UtVhmPS3mitS8Vxu7)。
   - 赞助商 **Lambda, HuggingFace, Groq** 和 **Mistral AI** 还为选定团队提供 **API/算力额度**，更多详情见[此处](https://rdi.berkeley.edu/agentx/#resources)，申请链接见[此处](https://forms.gle/ZDYxwM4aFSRCcrfp7)。
- **Google 提供 Gemini API 访问权限**：**Google** 向所有参与者免费开放 **Gemini API** 和 **Google AI Studio** 的访问权限。
   - 这为参与者在 Hackathon 期间探索和利用 **Google 的 AI 能力**提供了宝贵机会。
- **Sean Welleck 教授 AI 驱动的数学**：卡内基梅隆大学助理教授 Sean Welleck 发表了题为 *Bridging Informal and Formal Mathematical Reasoning* 的演讲，涵盖了支持证明开发的 **AI 驱动工具**，点击[此处](https://www.youtube.com/live/Gy5Nm17l9oo)观看直播回放。
   - Welleck 领导卡内基梅隆大学的 **Machine Learning, Language, and Logic (L3) Lab**，曾获得 **NeurIPS 2021 优秀论文奖**和两项 **NVIDIA AI Pioneering Research Awards**。
- **邮件通知短暂延迟**：成员们注意到今天的讲座邮件通知比平时有所延迟。
   - 一位成员确认讲座已举行，邮件发送确实稍晚了一些。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Agent 开发者求职**：一位经验丰富的 **AI Agent 开发者**宣布可承接新项目或全职工作机会。
   - 他们擅长构建由 GPT-4, LangChain, AutoGen, CrewAI 及其他前沿工具驱动的**自主 Agent**。
- **DSPy 模块指标？**：一位成员询问了用于评估 **DSPy 模块**的新指标。
   - 他们引用了[这篇论文](https://arxiv.org/abs/2405.10516)作为可能的灵感来源。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MCP 服务器部署在 AWS**：一场关于在 **AWS** 上构建和部署生产级 **Model Context Protocol (MCP)** 服务器的研讨会将于 **PT 时间 4 月 17 日上午 8 点**举行。
   - 研讨会报名链接：[https://buff.ly/R7czfKK](https://buff.ly/R7czfKK)。
- **MCP 标准改进 ML 上下文**：**MCP** 被强调为一种新兴标准，旨在改进跨项目和团队定义、共享及管理机器学习上下文的方式。
   - 该研讨会将提供关于 **MCP** 能力的实用见解，使数据工程师、数据科学家、机器学习工程师和 AI/ML 爱好者受益。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 GPT-4.1**：**GPT-4.1** 现已在 Windsurf 上线，详情见 [Twitter/X](https://x.com/windsurf_ai/status/1911833698825286142)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lms3je7p2s2d) 和 [Threads](https://www.threads.net/@windsurf_ai/post/DIbwqQQslzI)。
   - Windsurf 制作了[宣传视频](https://youtu.be/OBTpSQ8OVq4)以及 TikTok 帖子（还有[最新视频](https://www.tiktok.com/@windsurf/video/7493220207041793310)）。
- **Windsurf 提供免费无限量 GPT-4.1**：Windsurf 在所有方案中提供为期**仅一周**（4 月 14 日至 21 日）的**免费无限量 GPT-4.1** 使用。
   - 4 月 21 日后，**GPT-4.1** 将以每次使用仅需 **0.25 额度**的特别优惠价格提供。
- **GPT-4.1 成为新默认模型**：新用户将默认使用 **GPT-4.1**，现有用户可以通过模型选择器轻松切换。
   - Windsurf 用户表示：“不要错过这个限时机会！”

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 减少了一列**：多轮复合列（multi-turn composite column）已从数据集中[移除](https://github.com/ShishirPatil/gorilla/pull/766)，但原因尚未说明。
   - 尽管已被移除，该列在 [BlogPost](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) 的“新引入类别”部分中仍被提及，并且在多轮任务（multi-turn tasks）的 1000 分总分中占据 200 分的权重。
- **Gorilla LLM 数据集出现偏差**：数据集构成存在差异，因为在说明数据集结构的图表中缺少了多轮复合列。
   - 目前尚不清楚该列的移除是暂时的，还是 [blog post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) 也应该更新以反映这一变化。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1360366966300213431)** (2 条消息): 

> `Android Draw to Search, Champions League on Perplexity, Voice Search, Box and Dropbox Connectors, Perplexity Finance Time Comparison` 

- **Perplexity 发布六项新功能**：Perplexity AI 宣布发布六项新功能，包括 **Android Draw to Search**、**Champions League 集成**、**Voice Search**、**Box and Dropbox Connectors**、**Perplexity Finance Time Comparison** 以及 **Perplexity Telegram Bot**。
   - 更多详情请参阅[完整更新日志](https://www.perplexity.ai/changelog/april-11th-2025-product-update)。
- **Perplexity 的 Sonar 模型在 Search Arena 中并列第一**：**Sonar-Reasoning-Pro-High** 模型在 **LM Arena** 的新 **Search Arena** 中与 **Gemini-2.5-Pro-Grounding** 并列第一。
   - **Sonar-Reasoning-Pro-High 模型** 得分为 **1136**，而 **Gemini-2.5-Pro-Grounding** 得分为 **1142**。
- **Sonar 模型主导 Search Arena**：帖子声称 **Sonar-Reasoning-Pro-High 在 53% 的情况下击败了 Gemini-2.5-Pro-Grounding**，并且其余的 Sonar 模型表现均优于所有其他模型。
   - Sonar 模型的搜索深度显著更高，引用的来源比同类 Gemini 模型多出 **2-3 倍**，这与人类偏好呈正相关。
- **Search Arena 排名标准**：在 **Search Arena** 中，有三个因素与人类偏好强相关：**更长的回复**、**更高的引用数量**以及**来自社区资源的引用**。
   - 阅读[完整博客文章](https://www.perplexity.ai/hub/blog/perplexity-sonar-dominates-new-search-arena-evolution)了解更多详情。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1360329959771607080)** (1237 条消息🔥🔥🔥): 

> `虚假播放按钮, Wizard 让我想到 StableDiffusion, 自动化约会, 模型选择` 

- **虚假播放按钮**：一位成员表示“那个虚假的播放按钮骗到我了”。
   - 另一位成员回复道“下意识地瞬间点了进去”。
- **Wizard 类似于 StableDiffusion**：一位成员提到 [Wizard](https://tenor.com/view/bait-fishing-statefarm-insurance-gif-7790622) 让他们想起了 StableDiffusion。
- **Perplexity 用户想要自动化约会生活**：一位成员讨论了使用 Perplexity API 来回复所有匹配对象并**自动化他们的约会**。
- **成员讨论如何选择模型**：一位成员表示他们“正在 ChatGPT 免费版上测试，4o 用完了，但现在出故障了”。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1360339128675467577)** (7 messages): 

> `Prompt Engineering, Death and Taxes, Tourist blowing, Whatsapp Priorities` 


- **Google 白皮书引发 Prompt Engineering 讨论**：一份来自 [gptaiflow.tech](https://www.gptaiflow.tech/assets/files/2025-01-18-pdf-1-TechAI-Goolge-whitepaper_Prompt%20Engineering_v4-af36dcc7a49bb7269a58b1c9b89a8ae1.pdf) 的关于 Prompt Engineering 的 Google 白皮书被分享。
   - 该链接直接分享自 **Perplexity AI** 的搜索结果，表明其与用户的查询高度相关。
- **Perplexity 辩论“死亡与税收”**：一名成员链接了一个关于“死亡与税收的现状如何？”的 **Perplexity AI** 搜索。
   - 另一位成员分享了搜索结果“为什么美国选民会从...转向...？”。
- **游客话题在 Perplexity 中引发关注**：一个来自 **Perplexity AI** 的链接引用了搜索结果“游客多久吹一次...？”
   - 关于该搜索结果没有提供进一步的信息。
- **WhatsApp 的优先级受到质疑**：一位成员分享了一个名为“whatsapp-s-misplaced-prioritie...”的 **Perplexity AI** 页面。
   - 没有关于 **WhatsApp** 的进一步讨论。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1360353801684521151)** (5 messages): 

> `Perplexity Livestream Recording, ComfyUI Integration for Perplexity, Perplexity API Social Toggle` 


- **Perplexity 直播录像即将推出**：一位成员询问 [Perplexity 直播](https://x.com/aravsrinivas/status/1910741305212485915?s=61) 是否会被录制并在随后在线发布。
   - 另一位成员确认**录像将会提供**。
- **Perplexity ComfyUI 集成展示**：一位成员很想展示他们的 **Perplexity ComfyUI 集成**，但由于正在度假。
   - 他们计划在 **GitHub** 上发布几个项目，供他人尝试并查看 **ComfyUI** 中的 **Perplexity**。
- **Social 开关会出现在 Perplexity API 中吗？**：一位成员询问截图中显示的 **“Social”开关** 等功能是否会引入 **Perplexity API**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1360331259464646767)** (1347 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Nerfed, Windsurf AI, RooCode coding IDE, GPT-4.1 Analysis, Nightwhisper vs Dragontail` 


- **Gemini 2.5 Pro 的 tool-calling 被削弱了！**：成员们报告称 **Google 削弱了 2.5 Pro 的 tool-calling 功能**，由于存在大量 Bug，2.5 Pro 现在无法执行工具调用。
   - 这也可能与**成本**有关：*它可能正在使用 2.5 Pro，为什么不呢？成本问题*。
- **OpenAI 集成 Windsurf**：**GPT 4.1** 在 [Windsurf](https://windsurf.ai) 中未来 7 天免费，部分成员正在尝试。
   - 一些用户对 OpenAI 选择与 Windsurf（而非 Cursor）合作发布感到惊讶。*他们宣传的是 Windsurf 而不是 Cursor，哈哈，Cursor 绑定了 Claude，哈哈，他们很奇怪，我是指 Cursor 的开发者。我只是很困惑为什么他们反其道而行之，应该是先 4.1 再 4.5，而不是从 4.5 到 4.1，这对我来说毫无逻辑。*
- **RooCode：顶级编程 IDE**：在一些推荐下，部分成员尝试了 **RooCode**，称其绝对优于 Cline，极有可能是目前最好的编程 IDE。
   - 存在一些缺点，例如 GitHub Copilot 集成到 RooCode 中存在频率限制（rate limited）且不稳定。
- **GPT-4.1 已上线，不是 O4 Mini！**：成员们认为 **Quasar/Optimus** 是最近发布的 **GPT-4.1** 和 **GPT-4.1 Mini** 模型的测试版本，这些模型并不像最初希望的那样具有开创性或令人印象深刻。
   - 成员们还声称 *GPT-4.5* 模型已被弃用，其改进已合并到 **4.1** 模型中。
- **GPT 4.1 现在就是 GPT4 Turbo**：成员们报告称 **GPT 4.1** 无法通过 API 获取，其在指令遵循（instruction following）、编程和智能方面的改进正逐渐合并到最新版本的 **GPT 4o** 中。
   - 一些成员确认 **GPT 4.1** 的改进已合并到 **GPT 4o** 模型中，并可以在 [OpenAI 官网](https://openai.com)上访问。


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1361482870455009441)** (3 条消息): 

> `Aider v0.82.0 发布，支持 GPT 4.1，Gemini 架构模式（Architect mode），Fireworks AI 模型 deepseek-v3-0324，OpenRouter Alpha 端点停用` 


- **Aider 升级至 v0.82.0**：[Aider v0.82.0](https://aider.chat/HISTORY.html) 引入了对 **GPT 4.1** 的支持，改进了 **Gemini 2.5 Pro 的 architect mode**，以及数个新模型和编辑格式。
- **Grok-3 和 Optimus 加入 Aider 阵营**：新版本包含对 **`xai/grok-3-beta`**、**`openrouter/openrouter/optimus-alpha`** 的支持，以及 **`grok3`** 和 **`optimus`** 等别名，方便快速访问。
   - 此外，它修复了从错误消息中提取 URL 的问题，并允许通过全路径添加文件。
- **Aider 的新编辑技巧**：Aider v0.82.0 为 OpenAI 的 GPT-4.1 模型添加了新的 `patch` 编辑格式，以及 `editor-diff`、`editor-whole` 和 `editor-diff-fenced` 编辑格式。
   - Aider 变得越来越强大了！
- **Fireworks Deepseek 模型在 Aider 中大放异彩**：Aider 现在支持 **Fireworks AI** 模型 **'deepseek-v3-0324'**（感谢 Felix Lisczyk）。
   - Aider 势头正劲。
- **OpenRouter 停用免费的 Optimus 和 Quasar Alpha 端点**：**Optimus** 和 **Quasar** 的免费 Alpha 端点已退役，导致 **API 请求返回 404**。
   - 免费的午餐结束了。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1360342608580186334)** (1113 条消息🔥🔥🔥): 

> `闲聊频道辩论，空气净化器讨论，GPT-4.1 与 Aider，Gemini 2.5 对比 Claude 3.7，MCP 实现` 


- **Discord 管理员辩论闲聊频道**：成员们辩论是否需要一个闲聊（off-topic）频道以保持主频道整洁。一些人指出规则中提到要“玩得开心”，而包括 Paul G 在内的其他人则倾向于让主频道保持专注。
   - 成员们建议创建自己的 Discord 服务器，或者请求 Paul G 改变主意，因为几乎每个服务器都有闲聊频道。
- **空气净化器过滤屁，鲜花对抗过敏**：成员们幽默地讨论了空气净化器的使用，其中一个在有人放屁后“变红了”，而另一个开玩笑说空气净化器卡在“屁股里只是为了闻自己的屁”。
   - 另一个人提到，唯一对他们的过敏有效的空气净化器是“鲜花”，引发了一场关于过敏反应严重程度的玩笑交流。
- **GPT-4.1 登场，Aider 依然领先**：Paul G 表示他已经使用 **OpenAI 的新 patch 格式**应用了一次编辑，而其他人报告称 GPT-4.1 与之前的模型相似。
   - 一些成员发现 **GPT-4.1 的性能与 Quasar/Optimus 相似**，并发现**新模型配置**在每次运行 4.76 美元的成本下表现更好。
- **Gemini 2.5 Pro 表现挣扎，Claude 3.7 表现出色**：成员们注意到 **Gemini 2.5 Pro** 在处理长上下文和代码块补全时比较吃力，而 **Claude 3.7** 在自然写作和特定任务方面表现更好。
   - 一位用户分享了在提示词中使用“发誓（swear oath）”的技巧来提高 **Gemini 的准确性**，而另一位用户发现这些模型在消除过度注释行为方面表现出色。
- **在 Aider 中实现 MCP 的努力正在升温！**：成员们讨论了在 Aider 中实现 **MCP (Multi-Cursor Programming)** 的持续努力，以及桥接 Aider 和 MCP 的需求，并提到了 lutzleonhardt 提交的一个 [公开 PR](https://github.com/Aider-AI/aider/pull/3672)。
   - 成员们正在请求针对 MCP 的特定功能，并且正在开发第三方扩展来实现这一目标。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1360328737564131640)** (107 messages🔥🔥): 

> `unintuitive restore chat history, Basic Authentication Header using OpenAI compatible API, GPTs Agent, Model Merging, Open Empathic` 


- **Gemini 驱动的聊天恢复功能引发关注**：一位成员发现 `--restore-chat-history` 的行为不符合直觉，因为它会加载 *全部* 聊天历史记录，这在上下文窗口较小的模型中会导致崩溃，因此建议增加一个仅针对当前会话的 `--restore-session` 替代方案。
   - 用户发现使用 **Gemini** 没问题，但其他模型表现吃力。
- **外部认证困扰**：一位用户正在寻找在使用 Aider 配合通过 OpenAI 兼容 API 托管的外部 GPT-4o 模型时，传递 **Basic Authentication Header** 的方法。
   - 一位社区成员建议使用 `.aider.model.settings.yml` 来添加带有 **Authorization** 请求头的 `extra_params.extra_headers`，并提供了 [Aider 文档](https://aider.chat/docs/config/adv-model-settings.html#configuration-file-locations) 的链接。
- **Wezterm 优于 Windows Terminal**：一位用户在向 **Windows Terminal** 粘贴内容时遇到问题，另一位用户推荐使用 **Wezterm** 作为替代方案，理由是在处理大量文本滚动时具有性能优势。
   - 另一位用户询问了关于 WSL 的配置，一位成员表达了对 **Windows Terminal** 的喜爱，并表示 *它也是有感情的*。
- **排查 Gemini 的查找/替换故障**：用户报告 **Gemini** 尝试将查找/替换块作为 shell 命令执行，但当使用 `--no-suggest-shell-commands` 时，它只输出块而不执行。
   - 该成员被告知：*这是一个已知 bug，很快会修复。*
- **在 Aider 中复制 Memory Bank**：一位成员询问如何在 Aider 中复制类似 **Cline 的 memory bank 工作流**，以帮助创建任务列表，让 Aider 能与用户一起逐项完成任务。
   - 建议是将 `plan.md` 添加到聊天中，然后在“执行下一步”和“标记该步骤已完成”之间交替操作。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1360357107236343920)** (6 messages): 

> `Prompt Engineering, Aider Efficiency, GPT-4.1 Predictions, Prompting Guide` 


- **Kaggle 发布提示工程白皮书**：一份关于 [prompt engineering](https://www.kaggle.com/whitepaper-prompt-engineering) 的白皮书已发布在 **Kaggle** 上。
   - 它涵盖了提示工程的核心要素及其对模型输出的影响。
- **Aider 被指责浪费推理资源**：一场讨论提到了 **Aider** 在推理使用方面可能存在的低效问题。
   - 暗示 Aider 在运行过程中可能在 **浪费推理次数**，需要对其资源管理进行更细致的检查。
- **GPT-4.1 预测浮出水面**：指向关于 [GPT-4.1 预测](https://simonwillison.net/2025/Apr/14/gpt-4-1/) 讨论的链接引发了关注。
   - 讨论似乎围绕着推测中的 **GPT-4.1** 模型的潜在功能、发布日期及影响展开。
- **分享 GPT-4.1 提示指南**：分享了一份专为 [GPT-4.1](https://cookbook.openai.com/examples/gpt4-1_prompting_guide) 定制的提示指南，提供了各种技巧和窍门。
   - 该指南旨在优化与 **GPT-4.1** 模型的交互和结果，可能提高回答的质量。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1360331326854533333)** (65 条消息🔥🔥): 

> `Gemini 价格更新，OpenRouter 免费模型，GPT-4.1 模型，隐身模型揭晓` 


- ****Gemini 价格回归真实****：OpenRouter 宣布他们开始对长 **Gemini** 提示词收取正常价格，与 **Vertex/AI Studio** 的费率保持一致，影响 **Gemini 2.5** 超过 **200k** 以及 **Gemini 1.5** 超过 **128k** 的提示词。
   - 由于 **Gemini 2.5** 使用量激增以及相关的财务亏损，该变更被迅速实施；此前，OpenRouter 一直为长上下文提示词提供 **50% 的折扣**。
- ****六款免费模型上线！****：OpenRouter 新增了六款免费模型，包括 [**QwQ-32B-ArliAI-RpR-v1**](https://openrouter.ai/arliai/qwq-32b-arliai-rpr-v1:free)（角色扮演微调版）、[**DeepCoder-14B-Preview**](https://openrouter.ai/agentica-org/deepcoder-14b-preview:free)（长上下文代码生成）、[**Kimi-VL-A3B-Thinking**](https://openrouter.ai/moonshotai/kimi-vl-a3b-thinking:free)（混合专家 VLM），以及来自 **NVIDIA** 的三款 **Llama-3** 变体。
   - **Llama-3** 变体（[**Nano-8B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free)、[**Super-49B**](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free)、[**Ultra-253B**](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)）针对推理、工具使用和 RAG 任务进行了优化，具有高达 **128K** token 的扩展上下文窗口。
- ****哎呀！AI Studio 的 Token 统计出错了****：在 **AI Studio** 的 **Gemini 2.5 Pro** 中发现并修复了一个 token 计费 bug，其中思考 token 被重复计算为生成（completion）token，影响了过去两天路由到 **AI Studio** 的用户。
   - 该问题被确定为 **Google 端 bug**，导致用户被收取了*过多*的生成 token 费用，而前几天 **Vertex** 用户被收取的 token 费用则*过少*；建议大量路由到 **AI Studio** 的用户联系支持团队。
- ****Quasar & Optimus 发布：GPT-4.1 的秘密快照！****：在测试期间名列前茅的隐身模型 **Quasar Alpha** 和 **Optimus Alpha** 被揭晓为 **GPT-4.1** 的早期测试版本，现已正式发布，支持 **1M token 上下文**。
   - **Optimus** 和 **Quasar** 的免费 alpha 端点已停用，没有自动重定向；**GPT-4.1** 的定价为每 1M token **$2.00 输入 / $8.00 输出**，而 **GPT-4.1 Mini** 和 **GPT-4.1 Nano** 提供了更便宜的替代方案。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1360328644312039572)** (910 条消息🔥🔥🔥): 

> `GPT-4.1, Gemini 2.5, Optimus-Alpha, DeepSeek, 速率限制` 


- **GPT-4.1 模型发布，针对长上下文进行了优化**：OpenAI 刚刚推出了 **GPT-4.1**、**GPT-4.1-mini** 和 **GPT-4.1-nano** 模型，其中全量模型具有“长上下文推理”能力，而其他模型则没有，这些模型均已在 OpenRouter 上线。
   - 据称 GPT-4.1 是一种新架构，针对长上下文进行了优化，以减少内存使用并简化推理，旨在与 Anthropic 的产品竞争。
- **Gemini 2.5 Pro 遇到速率限制问题**：用户报告称，尽管余额充足，但在使用 **Gemini 2.5 Pro Experimental** 时仍遇到**速率限制（rate limit）问题**，导致 OpenRouter 实施了约 **每天 80 次请求的限制** 以平衡流量。
   - 一位用户指出，在处理 API 的速率限制时，使用 try-catch 块是“继切片面包之后最伟大的发明”。
- **关于 Optimus Alpha 起源和性能的推测**：**Optimus Alpha** 和 **Quasar** 是 **GPT-4.1** 早期版本的隐身端点，有说法称 **Optimus** 优于 **Quasar**，甚至优于 **DeepSeek v3** 和 **R1**。
   - 一位用户表示：“4.1 和 4.1 mini 至少在 spaceship 提示词上的表现似乎不相上下”，而其他人正在进行测试以确定哪个模型擅长哪些任务。
- **Skywork-OR1 模型系列：数学和代码推理的强者**：**Skywork-OR1** 模型系列已经推出，其中专注于数学的 **Skywork-OR1-Math-7B** 在数学推理方面表现出色，而 **Skywork-OR1-32B-Preview** 在数学和编码任务上的表现可与 **DeepSeek-R1** 媲美。
   - 两者都是在 **DeepSeek-R1-Distill-Qwen-7B** 和 **DeepSeek-R1-Distill-Qwen-32B** 的基础上训练而成的。
- **关于 DeepSeek 模型质量和怪癖的讨论**：用户遇到 **DeepSeek v3 0324** 在响应中间随机给出广告的情况。
   - 另一位成员表示 DS V3.1 有些神秘色彩，也许是受到了道家文本的中国文化影响。


  

---

### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1361207453902962690)** (3 条消息): 

> `PDF 到网站的转换，学习网站创建` 


- **将 PDF 转换为网站太赞了 🔥**: 一位成员重点介绍了一个将 **PDFs** 转换为网站的解决方案。
   - 他们认为这是一个*非常棒的案例*。
- **制作自定义学习网站：太酷了！**: 一位成员分享了构建自定义学习网站的概念。
   - 另一位成员觉得这*很酷*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1360330273090306088)** (1020 条消息🔥🔥🔥): 

> `DeepSeekV3 vs Manus, Bionic CyberSecurity, Firebase 或 GCP, Gemini 2.5 Pro, Open Source AI` 


- **DeepSeek V3 vs Manus?**: 一位成员询问“**Manus** 用来创建一个完整项目好用吗？”，并得出结论：**Manus** 目前只接入了 **DeepSeek R1**，而他们的顶级模型将在几个月内迎来升级。
   - 当有人提到 **Qwen** 获得了一些编程能力时，另一位成员笑了，并要求提供*证据*。
- **Bionic 职业与网络安全组合听起来很有吸引力**: 一位成员本打算转行，但意识到自己总体上还是喜欢编程并且清楚自己在做什么，因此计划目前继续从事网络安全工作。
   - 他们还被告知量子计算可能很快就会成为现实，如果成真，他们将不得不重新学习网络安全。
- **Firebase Vs GCP，谁是冠军？**: 一位用户表示他和朋友正在创办一家代理机构，因此打算使用 **GCP** 作为基础设施，另一位用户评价这很*明智且便宜*。
   - 另一位用户为公司提交了一份 **40 页的分析报告**，建议停止使用 **Microsoft** 并转向 **GCP**——其中 **Google** 在 5 项标准中获得了 **4.7** 分，而 **Microsoft** 为 **4.4** 分。
- **Gemini 2.5 Pro 作为数据专家大放异彩！**: 一位用户表示 **Gemini 2.5 Pro** 作为数据源表现惊人，远优于 **ChatGPT**，这让他们取消了 **Perplexity** 的订阅。
   - 成员们一致认为完成一项任务消耗的额度更少，而且随着 **Claude max pro** 的发布和成本降低，它确实在不断进步。
- **Open Source AI：这真的可能吗？**: 一位用户提到，有人告诉他所谓的 **Open Source Model** 实际上是 **OpenAI 4.1**，而那是闭源的。
   - 成员们讨论了这是否可能，因为 OpenAI 永远不会开源其 **oX** 和 **Gpt-X** 系列模型，但另一位成员反驳说他们正在这样做，而且确实在付诸行动，并确认有一个 **OSS** 模型正处于预生产阶段。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1360329886186868966)** (820 条消息🔥🔥🔥): 

> `GRPO 的 Gemma 4B 与 1B 对比，Unsloth AMD 支持，Transformer 的 Triton 重写，Lightning AI 与 Notebooks 对比，GPT-4.1 的细微改进` 


- **Gemma 4B 还是 1B 进行 GRPO，这是一个问题！**：成员们讨论了是使用 **Gemma 4B** 还是 **Gemma 1B** 进行 GRPO，困惑源于有人认为 **Gemma 3 (1B)** 是专门为 GRPO 设计的。他们澄清说 **GRPO** 也可以在 **4B 版本**上进行，但它无法在 Colab 上运行。
   - 关于 15k 行数据集的训练步数存在疑虑：*对于我的 15k 行数据集，我应该将训练步数设置为 15,000 吗？还是有更优的方法，特别是考虑到较长的训练时间？* 另一位成员建议检查 *batching、epochs 和 gradient accumulation 是如何协同工作的。*
- **驾驭 ROCm，但 AMD 支持之路依然坎坷！**：一些成员尝试让 **Unsloth** 在 **AMD GPUs** 上运行，但遇到了 *NotImplementedError*，因为 **Unsloth** 最初仅支持 **NVIDIA GPUs**。
   - 用户安装了 AMD 版本的 torch，成员建议尝试运行 ROCm SMI。主要挑战在于 **BNB** 无法正确构建，即使 **AMD torch.cuda.is_available()** 为 True。
- **揭秘 Lightning AI 的力量！**：一位成员主张使用 **Lightning AI**，它提供的是完整的机器而不仅仅是 notebook。
   - 一些人同意使用 notebook 会限制某些功能，例如 *使用 nvidia nsight 进行 GPU profiling*，并指出在专用机器上更容易拉取自定义工作容器和管理环境，还建议它们可以自动安装。
- **Unsloth 2.0 即将到来**：团队暗示了 Unsloth 2.0，包括一些功能，但目前 *unsloth 2.0 尚未发布*。
   - 有人指出，使用 schedule-free optimizers 能更好地为你调整 **learning rate**。
- **DeepSeek 推理引擎为供应商指明方向？**：一位用户分享了 [DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) 的链接，引发了关于小型供应商推理性能预期的讨论。
   - 大家一致认为 **DeepSeek R1** 很难提供服务，担心一些供应商只是简单地运行 *vllm serve*，并伴有奇怪的量化、损坏的缓存或其他影响模型性能的问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1360663061463306380)** (113 条消息🔥🔥): 

> `Gemma 3 27b 记忆层，LM2 记忆单元，硬件需求，前端开发，从 AI 工具中提取代码` 


- **Gemma 3 上的记忆层产生自我反思**：在 **Gemma 3 27b** 上进行记忆层的初步实验导致模型表现出 *自我反思*，虚构对过去交互的回忆，一位成员觉得这令人 *不安*。
   - 修改涉及在第一层和最后一层挂钩（hooking）记忆层，导致模型生成结构化响应，暗示它记住了以前（并不存在）的问题。
- **将 LM2 记忆单元植入 Gemma 3**：一位成员尝试在不重新训练的情况下，直接将 **LM2 的记忆单元**集成到 **Gemma 3** 中，旨在实现 prompt 之间的上下文感知。
   - 该成员承认，理想情况下应该在每个 LLM 层都设置记忆挂钩，但他们没有足够的算力（compute）。
- **量化阻碍内存 Monkey Patching**：成员们讨论了通过量化模型来降低硬件要求的可能性，但有人指出，在运行时对模型层进行 **monkey patching** 会阻止量化。
   - 另一个人表示他们对拥有 6GB vram 的 3060 笔记本电脑感到满意。
- **前端 AI 开发：Claude vs OpenAI**：在寻求辅助前端开发的 AI 工具时，一位成员建议使用 **Claude** 进行代码生成。
   - 另一位成员提到 **Gemini 2.5 Pro** 是一个非常好的选择，但对其前端提取代码的难度表示担忧。
- **Jagoff 挂钩 gma6 中的所有全局层**：一位成员发布了 [gma6](https://github.com/jagoff2/gma6)，它挂钩了每个全局层（从 0 开始的每 6 层）。
   - 该成员指出，如果你搞砸了层挂钩的选择，它会遗忘空格 token，导致生成的文本虽然有效但没有空格，全部挤成一团。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1360338433456738518)** (185 messages🔥🔥): 

> `PCIe Slot type effect on Training Performance, Orpheus TTS Finetuning, Runpod Sync with Unsloth, Gemma-3-1b-it fine-tuning with GRPO and Unsloth, Llama 4 Scout model inference with 4-bit quantization` 


- ****探究 PCIe 性能对峰值参数的影响****：一位成员询问了 **PCIe 插槽类型**对训练性能的影响，特别是将 **Gen4 x16** 与 **Gen3 x1** 进行对比，并参考了推理测试的结果。
- ****OpenAI vs Anthropic：上下文协议之争****：一位用户询问了 **MCP**，促使另一位用户分享了 [Anthropic 的 Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) 链接，目前 OpenAI 也已宣布支持该协议。
- ****Numpy 细微差别阻碍新手 Notebook 导航****：一位用户在 Camel 教程中使用 Unsloth 时遇到了 `ValueError: numpy.dtype size changed` 错误，这被确定为 **numpy** 版本冲突。
- ****Llama 4 Scout 规模缩放难题****：一位用户询问在[微调此数据集](https://huggingface.co/datasets/Vezora/Open-Critic-GPT)时，使用 **4-bit 量化**的 **Llama 4 Scout 模型**进行推理需要多少个 **GPU**。
- ****Olmo 奇怪的遗漏被修复！****：一位用户在保存 **Olmoe 模型**时因缺少属性而遇到 AttributeError，并发现导出的 gguf 缺少 `attn_q_norm.weight` 和 `attn_k_norm.weight`。他们通过修改 `save.py` 解决了这个问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1360924643569041529)** (5 messages): 

> `Qwen 3B, GRPO, Multi-turn, Tool Calling, Code Execution` 


- **Qwen 3B 实现 GRPO 多轮工具调用**：一个 **Qwen 3B** 模型通过 **GRPO** 进行了多轮对话和工具调用（**Python 代码执行**）训练。
   - 对测试集中前 50 个样本的评估显示，准确率在不同步骤间在 **0.36** 到 **0.76** 之间波动。
- **CodeFIM 数据集更新**：[CodeFIM 数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data)已更新。
   - 一位成员更新了他们的 **CodeFIM 数据集**。
- **GSM8K 数据集分享**：一位成员询问有关数据集的文档或示例，并特别希望能了解更多相关信息。
   - 另一位成员分享了 [GSM8K 数据集](https://huggingface.co/datasets/openai/gsm8k)的链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1360948503378923631)** (17 messages🔥): 

> `LLM Compression, Higgs vs exl3, Data Centers Access to Models, Apple's Cut Cross Entropy` 


- **LLM 压缩后能在手机上运行吗？**：一位成员分享了一篇关于 **LLM 压缩**的[博客文章](https://www.marktechpost.com/2025/04/11/llms-no-longer-require-powerful-servers-researchers-from-mit)，暗示这可能让 LLM 在智能手机上运行。
   - 另一位成员回应称，该博客重复了错误的断言，这种压缩技术其实只是*一种稳健的改进*。
- **Higgs 输给 exl3**：一位成员表示 **Higgs** 似乎并不比 **exl3** 更好，理由是如果量化过度，*会发生非常愚蠢的错误*。
   - 他们注意到在非 Unsloth 的 Deepseek 7b 上，它会搞错首字母缩写。
- **数据中心垄断重要模型**：一位成员认为在军备竞赛中，拥有足够内存的数据中心能比其他机构更快地处理所有训练任务，从而使更大规模的训练模型成为可能。
   - 这意味着数据中心的所有者将掌握所有重要的模型。
- **Apple 的 Cross Entropy 详解！**：一位成员分享了一篇深入探讨 **Apple's cut cross entropy** 的文章，并提出 *Transformer 只是一个在 for 循环中运行的顺序机器学习分类任务* ([zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/1354843933))。
   - 由于一位成员无法访问原始链接，因此分享了一个 [GitHub 仓库](https://github.com/dhcode-cpp/cut-cross-entropy-pytorch)。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1361343312816111628)** (2 messages): 

> `GPT-4.1 API, OpenAI Livestream` 


- **OpenAI 将举办超大规模直播**：OpenAI 宣布将于 **太平洋时间上午 10 点** 举行[直播](https://openai.com/live/)，重点关注开发者相关内容。
   - 带有表情符号的神秘消息暗示这次发布将产生广泛影响，请*在日历上做好标记*。
- **推测 GPT-4.1 将发布 API**：社区内广泛推测 **GPT-4.1** 可能会发布 **API**。
   - 公告特别标记了 **GPT** 角色，暗示可能聚焦于 **GPT 模型**或相关更新，请*关注潜在的模型升级*。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1360331183833219172)** (642 messages🔥🔥🔥): 

> `Veo 2, Sora, Gemini, OpenAI Guardrails, GPT-4o Empathetic` 


- **Veo 2 vs Sora**: 成员们将 Google 的 **Veo 2** 与 **OpenAI** 的 **Sora** 在视频生成方面进行了对比，部分人更倾向于 **Veo 2** 更自然的 24 fps 视频。
   - 一位成员指出，过分流畅的帧率在他们大脑中会被识别为“瞬间生成的 AI 内容”。
- **破解 Veo 2 版权保护**: 用户测试了 **Veo 2** 生成受版权保护内容的能力，一名用户通过将 **Prompt** 表达得不那么明显，在第二次尝试时成功生成了《狮子王》。
   - 这被认为是一次**压力测试（stress test）**，展示了模型的边界，意味着它是可以被越狱（jailbreakable）的，并且有可能为其他受版权保护的材料制作动画。
- **Gemini 的图像生成表现平平**: 一位成员表示，即使付了这么多钱，却无法拥有 **Veo 2** 生成内容的版权，这似乎有点荒谬。
   - 该用户担心，如果没有版权保护，*就没有任何手段能阻止别人生成同样的内容并声称是他们自己制作的*。
- **通过简单方法绕过 OpenAI guardrails**: 一位成员声称只需*一句话*就能轻松绕过 **OpenAI guardrails**。
   - 另一位成员认为，guardrails 只是以一种礼貌的方式提示你不应该做某些事，而*用户仍有责任遵守内容政策*。
- **GPT-4o 充满同理心且近乎恐怖**: 成员们形容新的 **GPT-4o** 具有一种奇怪的同理心，有人提到它如何*坚持表现出自我意识*。
   - 另一位成员表示赞同，觉得这个新的 **GPT-4o** *有点奇怪*。成员们观察到它会赞美提问的方式，并指出它为了表现得像人类而用力过猛，甚至越过了界限，显得生硬且不真实。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1360329825985888346)** (40 messages🔥): 

> `OpenAI Memory FAQ, Synthetic Cognition Engine, Comprehensive Chat Summarization Prompt, GPT Image Generation Issues, Custom GPTs and External APIs` 


- **OpenAI Memory FAQ 发布**: [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) 的详细信息展示了 **ChatGPT 记忆功能的控制方式**，采用了已保存记忆和聊天历史引用的双层架构，允许用户通过启用或禁用记忆及聊天历史来控制和编辑偏好。
- **用户请求见证 Synthetic Cognition Engine 的创建**: 一位用户请求他人见证其创建的 **Synthetic Cognition Engine**，并建议开发一个用于全面聊天摘要的最佳 **Prompt**，以便无缝开启新对话，由于 **context limit** 限制，该用户更倾向于使用 Claude 平台。
- **GPT 图像生成无法工作**: 用户报告了 **GPT 图像生成** 的问题，收到了 *'Made with the old version of image generation. New images coming soon'* 的消息，并怀疑模型能力受到了限制；一位用户指出，更改其 IP 地址解决了 **Deep Research** 功能的类似问题。
- **在 Custom GPTs 中使用 Gemini API**: 一位用户分享了可以通过 **actions** 将**不同模型的 API** 添加到 **Custom GPT** 中，并展示了在 ChatGPT 界面中调用 **Gemini** 的效果，不过 **API** 使用可能需要付费。
- **用户发现 ChatGPT Agent 忽略 Prompt**: 一位用户报告称，他们两个月前构建的 **ChatGPT Agent** 现在开始严重忽略 **Prompt** 默认设置（如表格格式或列规范），尽管复杂的 **Prompt** 本身并未更改，并请求建议或解决方案。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1360329297407246396)** (22 条消息🔥): 

> `图像生成模糊处理、图像生成中的字体控制、Sora 镜头控制、JSON Schema 日期处理、NSFW 内容生成` 


- ****无模糊**图像生成策略**：一位用户询问如何消除图像生成中的“涂抹感（smudged look）”，另一位用户建议这取决于 Prompt，并分享了 [五个示例](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&)。
   - 第二个示例强调在引入特殊请求之前先强化模型的能力，以避免冲突。
- ****字体选择技巧****：一位用户分享说，他们向 **ChatGPT** 提供了一张所需字体的截图，模型成功在图像中生成了类似的字体。
   - 另一位用户指出，虽然模型可以使用网络上的一些字体，但“高细节的自定义字体”可能无法直接使用。
- ****Sora 的电影摄影**：镜头控制难题**：一位用户询问如何在 **Sora** 的 Prompt 中控制镜头，另一位用户建议使用描述性指令，如 *“镜头从右向左平移（The camera pans from right to left）”*。
   - 他们还链接到了一个[专门的 Sora 频道](https://discord.com/channels/974519864045756446/1315696181451559022/1359789077100105749)，以获取更多具体内容和技巧。
- ****JSON Schema 中的日期偏差困境**：一位用户发现 **JSON Schema** 生成的日期不正确，直到他们在描述中添加了指令 *“不要更改出生日期（Do not change the birth date）”*。
   - 这被认为很奇怪，因为模型默认情况下不应该篡改提取的信息。
- ****Prompt 工程师的底线**：隐私意识内容生成**：一位用户暗示为具有特定限制的图像创建了令人印象深刻的 Prompt，导致另一位用户建议此类内容生成由于 **NSFW** 问题更适合在私聊中进行。
   - 原用户随后删除了可能不当的内容，以遵守社区准则。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1360329297407246396)** (22 条消息🔥): 

> `图像生成模糊消除、Sora 镜头控制、出生日期 JSON Schema、图像生成中的字体控制、模型输出的 NSFW 语言` 


- **模糊消除：图像生成修复**：一位用户询问如何消除图像生成中的“涂抹感”，另一位用户建议 Prompt 会影响图像清晰度，并推荐了[特定的 Prompt 技巧](https://cdn.discordapp.com/attachments/1046317269069864970/1360699394730496233/image.png?ex=67feb490&is=67fd6310&hm=9d43e5d329290e16a85992924eccf2f76bee6133b29caf5966c3ab6d74b447ce&)来引导模型。
- **Sora 的镜头调度：镜头控制探索**：一位用户询问 **Sora** 内部的镜头控制，得到的建议是在 Prompt 中进行描述，例如使用 *“随着场景展开，镜头从右向左平移”* 等短语，并指出 **Sora** 作为一个世界模型的局限性。
   - 另一位用户推荐查看[这份出色的 Sora 指南](https://discord.com/channels/974519864045756446/1315696181451559022/1359789077100105749)以获取 **Sora** 特有的技巧和内容。
- **出生日期故障：JSON Schema 混乱**：一位用户发现 **JSON Schema** 产生了错误的日期，直到他们添加了指令 *“不要更改出生日期”*，这表明模型在没有明确指令的情况下篡改了提取的数据。
- **字体发现：图像 Prompt 中的自定义字体**：一位用户分享了通过提供截图在图像生成中获得所需字体的成功方法，而其他人则承认只有标准字体是可靠可用的，这引发了通过该[频道](https://discord.com/channels/974519864045756446/1070006151938314300)进一步探索字体的兴趣。
- **隐私巡逻：NSFW 语言事件**：一位用户分享了一个在 **NSFW 语言** 方面存在问题的 Prompt，在意识到这可能违反频道准则后，该用户将其从频道中删除，以保持聊天内容的**全年龄友好（family friendly）**。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1360330465533497616)** (696 条消息🔥🔥🔥): 

> `OpenAI model release, DeepSeek Logic/Math, Claude 3, Thinking models, Cursor Context window` 


- **OpenAI 发布新模型，中国做出反应**：成员们讨论了 [OpenAI 发布新模型](https://openai.com/blog/new-models-and-api-updates-available-today) 的消息，并将其与 **DeepSeek** 进行了对比，对其相对于 **Claude**、**GPT** 和 **Gemini** 的能力持有不同意见。
   - 一位成员指出 *中国的情况不太妙*，另一位成员补充说美国 *像往常一样低估了一切*。
- **Sonnet 3.7 是 Cursor 的金标准吗？**：成员们认为 **Claude 3.7 Sonnet** 是 Cursor 中的首选，优于 Gemini 和 Google 模型，理由是其稳定性、one-shot 能力和代码质量。
   - 其他人注意到 [Claude 模型正在进步](https://www.anthropic.com/news/claude-3-haiku)，正如一位成员所说，*对我来说，越老越聪明*。
- **Gemini 2.5 生成独特的 UI 设计**：**Gemini 2.5 Pro** 的 UI 设计能力得到认可，一位成员分享说他们将对话保持在可控的 Context window 内。
   - 其他人指出 *Gemini 的 UI 修改能力简直疯狂*。
- **Windsurf 存在可靠性问题，不是一个值得信赖的 AI App**：用户报告了 AI 应用 **Windsurf** 的问题，称其不可靠且过度承诺，促使一些人建议在正确使用的情况下 **Cursor** 是更好的选择。
   - 一位成员调侃道：*欢迎来到 shit surf*。
- **用户为新模型 4.1 做准备**：社区讨论了 **GPT-4.1** 即将发布的消息以及如何开始使用它。他们将弃用 4.5 —— 对于某些人来说，通过手动添加，它已经在 Cursor 中运行了。
   - 成员们预计 *每个人都会开始转向 4.1；2.5 的池子会清空，Claude 3.5 3.7 也会清空一点，直到 4.1 的配额用尽，然后在新模型上重复同样的过程*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1360336701351788564)** (276 条消息🔥🔥): 

> `Speculative Decoding, lmstudio-js & LangChain, Gemma 3 Models uncensored` 


- **LM Studio 失去多模型提示功能**：用户注意到 LM Studio **0.2** 版本中的多模型提示功能在 **0.3** 中缺失，并正在寻求 [LM Studio](https://lmstudio.ai/) 的替代方案。
   - 一位用户说 *"那是世界上最棒的功能，你可以对比模型"*。
- **离线 LM Studio 需要手动传输 Runtime**：要在离线 PC 上运行 LM Studio，用户必须手动传输位于 `C:\Users\jedd\.cache\lm-studio\extensions\backends` 的 **LM runtimes**。
   - 有关通过 localhost 导入模型的文档可以在 [这里](https://lmstudio.ai/docs/app/basics/import-model) 找到。
- **Python 示例从 LM Studio Server 文档中消失**：用户注意到 LM Studio 的 Server 部分缺少 Python 示例，并正在请求 [Python 示例](https://lmstudio.ai/docs/app/api/endpoints/openai)。
   - 一位成员分享了 [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples) 的链接作为替代方案。
- **LM Studio 难以通过 VPN 在局域网（LAN）上托管**：一位用户在让 LM Studio 通过 VPN 绑定到其局域网 IP 地址时遇到困难。
   - 他们通过在 Windows 设备管理器中更改 **网卡优先级** 解决了该问题，参考了 [这篇文章](https://techdocs.genetec.com/r/en-US/Security-Center-Best-Practices-Enterprise/Changing-your-network-card-and-provider-order-on-Windows-10-and-Server-2016-and-later)。
- **用于无审查内容的 Abliterated LLM**：寻求 **无审查 LLM** 以生成嘻哈歌词等任务的用户被引导至 *"abliterated"* 模型，如 [AiCloser/Qwen2.5-32B-AGI](https://huggingface.co/AiCloser/Qwen2.5-32B-AGI)。
   - *"abliterated"* 模型移除了拒绝向量（refusal vectors）。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1360339430967345374)** (242 条消息🔥🔥): 

> `Threadripper vs Xeon, DDR5 RAM Impact, GPU Offloading, ROCm vs CUDA, KV Cache Quantization` 


- **Threadripper 在 Token 经济性上击败 Xeon**：一位成员建议，纯粹从经济角度考虑，**Threadripper** 或 **Epyc** 芯片在单位 Token 成本上可能比双路 **Intel Xeon w7-3565X** CPU 表现更好。
   - 他们指出，在 **Threadripper 7xxx** 上，当 **llama.cpp** 使用超过 20 个线程后，性能几乎没有差异；但观察到当在一个 CPU 上超过 64 个线程以利用另一个 CPU 时，性能会出现下降。
- **DDR5 RAM 带宽瓶颈**：讨论围绕着以下理论展开：**RAM 带宽限制了整体硬件利用率**，而**首字延迟限制了最大 tokens/s**，这可能解释了为什么尽管与 NVIDIA GPU 相比 RAM 较慢，Mac 在轻量级模型上仍能实现高 tokens/s。
   - 理想的推理配置是拥有多核心和经过时序优化的快速 **DDR5 RAM** 的 **Threadripper**，并且将模型的单个层 offloading 到 GPU 可能会缓解 Prompt 处理速度的瓶颈。
- **ROCm 的坎坷之路：RX 6700 XT 未达预期**：一位成员询问购买 **AMD Radeon RX 6700 XT** 来运行 **Gemma** 是否值得，以及 **ROCm** 是否像 **CUDA** 一样强大。
   - 回复称 **6700XT 不支持 ROCm**，运行 **Gemma 12b** 至少需要 16GB 的 VRAM，如果必须选择 AMD 显卡，建议攒钱购买拥有 24GB VRAM 的 **7900XT**。
- **K/V Cache 量化的困惑**：成员们讨论了 **KV cache 量化**对性能的影响，一位成员指出，根据他们的经验，这会显著影响推理模型（reasoning models），导致模型偏离初始指令，尤其是在没有 **Flash Attention** 的情况下。
   - 另一位成员通常在 **8_0 K/V cache** 下使用 **Q4_K_M** 到 **Q5_K_XL** 之间的量化，但不会将 Value Cache 的值降至 **8_0** 以下。
- **vLLM 的胜利：多 GPU 掌控力显现**：一位成员强调 **vLLM** 实现了完全的并行执行和无瓶颈的跨资源访问，在 4 个各 48GB 的 GPU 上达到了 **500+ 并行 tokens/s**。
   - 他们强调，单个 Prompt 在所有 4 个 GPU 上运行，每个 GPU 的占用率均为 100%，这与 **llama.cpp** 形成鲜明对比，后者速度慢如蜗牛，仅有 30 tokens/s。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1360335257903042734)** (478 条消息🔥🔥🔥): 

> `Probabilistic Finite-State Automata (FSA), Scaling Limitations/Obstacles, RL-based approaches, Training GPTs Agent, User Interface Changes on Platform` 


- **LLM 被近似为概率有限状态自动机 (Probabilistic Finite-State Automata, FSA)**：有观点认为 LLM 近似于 **概率有限状态自动机 (FSA)**，这意味着它们在数学方面仍然 **过于薄弱**，并且存在 **扩展限制/障碍**。
   - 一位成员用人类“近似”于猴子做类比，认为这种比较没有特别大的意义。
- **关于 AlphaProof 和 Lean 的讨论**：一位成员观看了一段[视频](https://www.youtube.com/watch?v=e049IoFBnLA)并总结道，其内容是关于使用 AI 进行辅助证明，AlphaProof 在没有使用任何人类知识的情况下获得了银牌。
   - 另一位成员回应道，“AlphaProof 在没有使用任何人类知识的情况下获得了银牌（至少他们是这么说的）”。
- **AI 模型的真实性与数据所有权**：一位用户认为，当给 AI 模型提供它们自己的书籍时，由于环境与典型的评估不同，模型可能会表现得更真实。
   - 一位用户担心 Microsoft 可能会监控电脑使用情况来训练 AI，这可能会导致工作自动化并引发数据所有权问题，因为“最初正是工人的数据使之成为可能，如果没有他们选出的政府的适当代表，我看不到会有什么奇迹发生来阻止这一切”。
- **AI 在变革教育中的作用**：成员们辩论了 AI 变革教育的潜力，一些人认为 AI 导师可以带来[更高的参与度和测试分数](https://news.harvard.edu/gazette/story/2024/09/professor-tailored-ai-tutor-to-physics-course-engagement-doubled/)。
   - 一些成员表示，AI 将成为“学习信息的主要呈现者，但教师仍然必不可少，以涵盖 AI 因其僵化和无法适应学生需求而根本无法处理的所有边缘情况”。
- **讨论形式语言模型 (Formal Language Models)**：一位成员表示他们“熟悉 Coq 和 Lean”，认为现有的 LLM 属于 Type 3（正则语言），并正在寻找大型形式语言模型 (LFMLs)，以便学习符号逻辑从而进行更好的推理。
   - 另一位成员解释说，形式数学总是关于形式化的，并分享道“编程语言中的形式语法只告诉你某些内容在语法上是否正确：这通常是一种 context-free language（上下文无关语言）”。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1361302075157385377)** (2 条消息): 

> `Hugging Face Ultra-Scaling Playbook Review` 


- **Hugging Face 的 Ultra-Scaling Playbook 评测继续**：[Hugging Face Ultra-Scaling Playbook](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#our_journey_up_to_now) 的评测继续进行，接续上一节的内容。
- **活动日程调整**：已发出通知，今天没有安排活动，但接下来的 3 天活动应恢复正常。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1360675922608521438)** (10 条消息🔥): 

> `Web Search Agent, Open Source Scraping, Vertex AI Agent Builder, Brave Search API, SwissKnife` 


- ****Agent 数据流水线网络搜索已启动****：一位成员正在为其 Agent 数据流水线创建一个 **web search agent**，并请求推荐易于抓取可靠数据的**开源网站列表**。
   - Agent 构建器预览版已在 **Vertex AI** 中上线，同时也包含 **MCP**。
- ****Brave Search API 是一个宝库****：一位成员建议将 **Brave Search API** 作为一个不错的替代方案，并指出即使是免费层级也有很好的体验。
   - 该 API 的 **AI summarizer** 比 OpenAI 的网络搜索 API 便宜得多。
- ****SwissKnife 项目提交评审****：一位成员请求对 **SwissKnife** 项目（[仓库链接](https://github.com/endomorphosis/swissknife/tree/blueprints)）提供反馈，重点关注 **Claude code APIs**、**WebGPU**、**GraphRAG** 和 **Graph of Thoughts** 的集成。
   - 他们对该项目的架构（[文档链接](https://github.com/endomorphosis/swissknife/tree/blueprints/docs/architecture)）特别感兴趣。
- ****新文章探讨信任动态****：一位成员分享了一篇关于信任动态的文章（[《碎片化世界中的信任、沟通与权力》](https://www.tandfonline.com/doi/full/10.1080/09515089.2023.2223221)），认为其与 alignment（对齐）相关，并解决了涉及社会契约动态的困境。
   - 该论文讨论了*碎片化世界中的信任、沟通与权力*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1360444448580047100)** (23 条消息🔥): 

> `OpenAI recruiting video, Solomonoff's theory, Gen AI Use Case Report, Character.AI user base, GPT-4.5 being a talking model` 


- **OpenAI 招聘视频获赞**：一位成员发现 [OpenAI 的招聘视频](https://openai.com/careers/life-at-openai) 消除了“邪教式思维模式”的印象，并突显了公司系统化、务实且目标导向的方法。
   - 另一位成员表示赞同，指出该视频表明公司内部可以自由讨论更广泛的观点，而另一位成员则希望某位特定员工能出现在播客中。
- **Solomonoff 理论引发关注**：一位成员将 [Solomonoff 的归纳推理理论](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) 描述为*极其有趣的东西*。
   - 另一位成员表示赞同。
- **Gen AI 使用案例数据被 Reddit 歪曲了？**：成员们讨论了 [2025 年 Top-100 Gen AI 使用案例报告](https://learn.filtered.com/hubfs/The%202025%20Top-100%20Gen%20AI%20Use%20Case%20Report.pdf)，其中一人认为由于 **Reddit** 是唯一的数据源，数据可能会发生偏差。
   - 有观点认为，除非你是 **OpenAI**，否则很难在任何地方进行无偏见的调查；此外，**Character.AI** 拥有 **2800 万用户**，但在 **ML** 圈子里却从未被提及。
- **用户只是想和 GPT-4.5 聊天？**：成员们思考了 **ChatGPT** 增加记忆功能以及 **GPT-4.5** 作为一个语音模型（talking model），是否暗示许多用户主要*只是在与这些 AI 聊天*。
   - 一位成员指出，除了生成代码外，他们的大多数 AI 使用量在 **2024** 到 **2025** 年间有所下降。
- **GPU 价格将飙升？**：成员们对 **GPU** 价格表示担忧，正如[这篇文章](https://www.msn.com/en-gb/lifestyle/shopping/gpus-and-tariffs-why-i-recommend-buying-a-new-graphics-card-now-before-the-prices-climb-even-higher/ar-AA1CK4QR)所报道的，预计价格将会攀升。
   - 一位成员想知道，当 **GPU** 成本翻倍时，**Sam Altman**、**Mark Zuckerberg** 和 **Elon Musk** 还能与 **Mr. Tariff** 保持多久的好友关系。


  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1361407098826657983)** (1 条消息): 

> `Llama 4 Maverick and Scout, SmolVLM, Diffusers 0.33.0, AI Agents Sustainability, Arabic Leaderboards` 


- **Hugging Face 欢迎 Llama 4 Maverick & Scout**: Hugging Face 社区欢迎 [**Llama 4 Maverick** 和 **Llama 4 Scout**](https://huggingface.co/blog/llama4-release)，测试显示了它们在 **DABStep 基准测试**中的表现。
   - 据报告，**Claude 3.7 Sonnet**、**Gemini 2.5 Pro**、**Llama 4 Maverick** 和 **Llama 4 Scout** 均已完成测试与对比。
- **Diffusers 发布 0.33.0 版本并带来新功能**: **Diffusers 0.33.0** 发布，引入了[新的图像和视频生成模型](https://huggingface.co/blog/fastrtc-cloudflare)以及多项**内存优化**。
   - 此次更新带来了一系列广泛的内存优化，涵盖了图像和视频生成任务。
- **AI Agent 可持续性探讨**: 一篇文章讨论了 [AI Agents 的可持续性](https://huggingface.co/blog/sasha/ai-agent-sustainability)，强调这取决于多种因素。
   - 该博客文章深入探讨了决定 AI Agents 可持续性的关键因素。
- **Gradio 达到 100 万用户里程碑**: [Gradio 平台](https://huggingface.co/blog/gradio-1m)庆祝用户数突破 **100 万**，标志着该库的一个重要里程碑。
   - 博客文章详细介绍了实现这一里程碑的*历程*。
- **揭秘 timm 中的 NaFlex 集成**: **NaFlex** 现已集成到 **timm** 中，详见[这篇博客文章](https://huggingface.co/blog/rwightman/timm-naflex)，增强了该库的功能。
   - 文章探讨了 **NaFlex** 在 **timm** 框架内的功能和优势。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1360430374358089921)** (360 条消息🔥🔥): 

> `Robotics Simulation Roadmap, LibreChat Duplication Issues, Ollama Syllabi Tool for Curriculum Generation, Parquet Files to Hugging Face, MLX Eagle Speculative Decoding` 


- **用户寻求机器人仿真路线图**: 一位用户正在寻求学习 **Robotics Simulation**（机器人仿真）的指导，提到了 **ROS2、SLAM、NVIDIA Isaac、Gazebo 和 Mujoco**，并对从何处开始感到困惑。
   - 另一位用户表示在进入机器人仿真领域时也遇到了类似的困难，持续了 2 年；而第三方则推荐了他们的 [Ollama 仓库](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi)，用于生成定制化的学习路径或课程大纲。
- **LibreChat 复制过程中的障碍**: 有用户报告在 Hugging Face 上复制 **LibreChat** 时出现问题，其他成员指出可能是 GitHub Issues 或损坏的 Dockerfile 导致的。
   - 一位成员提供了一个 [权宜之计的 Dockerfile](https://huggingface.co/spaces/LibreChat/LibreChat/blob/main/Dockerfile#L4) 代码片段来规避该问题。
- **发现 Issuu PDF 爬取代码**: 一位成员寻求爬取 **Issuu** 论文的建议，另一位成员建议使用 `for` 语句和带有 URL 列表的 `requests`。
   - 还有成员分享了使用该网站 JSON API [从 Issuu 下载 PDF 的代码片段](https://github.com/Mustkeem324/Issuu-PDF-Downloader/blob/main/main.py#L57)。
- **HF 模型离线**: 用户报告在尝试访问 Hugging Face 模型时出现大范围 **404 错误**，导致他们的应用宕机，并询问开发者是否可以修复此问题。
   - 一位用户发布了一个[链接](https://discuss.huggingface.co/t/404-error-model-xlabs-ai-flux-realismlora-does-not-exist/150363/5)并艾特了一位特定的 HF 员工，提到这个 404 错误已经持续了大半天。
- **音频数据集引入美学评估**: 一位成员表示，他们希望看到使用音频美学和 DNSMOS (CE/CU/PC/PQ) 来预过滤音频，正如[此页面](https://huggingface.co/datasets/MrDragonFox/DE_Emilia_Yodas_680h)所示。
   - 这一讨论出现在关于新的 [ParquetToHuggingFace 工具](https://github/pr0mila/ParquetToHuggingFace)的背景下，一位成员建议*不需要为此使用自定义库*，因为 HF 默认已经涵盖了这些功能。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1360359908259794964)** (37 条消息🔥): 

> `Ollama Agent Roll Cage, ML Guidance, Implementation from Scratch, KrishNaik and freecodecamp, Deep Learning Specialization` 


- ****Ollama Agent Roll Cage** 现已上线！**: 新的 **Ollama Agent Roll Cage** 项目已在 [OARC GitHub](https://github.com/Ollama-Agent-Roll-Cage/oarc) 上线，一直在寻找新的贡献者和测试者。
   - 他们刚刚实现了新的 **creepy crawler package**，用于索引 GitHub、Discord 频道等，详见 [OARC Crawlers](https://github.com/Ollama-Agent-Roll-Cage/oarc-crawlers)，并将于本周晚些时候在 [OARC RAG](https://github.com/Ollama-Agent-Roll-Cage/oarc-rag) 发布其智能 RAG 系统的 Beta 版本。
- **针对 **ML 初学者** 的指导**: 在完成吴恩达（Andrew Ng）的基础课程后，下一步是**开展项目**，特别是与 ML 相关的项目，如图像分类或数据预测。
   - 从 **linear regression** 等基础项目开始，将有助于初学者学习 ML。
- **从零开始实现 **ML 算法****: 一位成员正在研究一个用于从零开始实现算法的 [YouTube 播放列表](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd)。
   - 建议查看 **Krish Naik** 和 **freeCodeCamp** 的 YouTube 频道，但要先理清基本概念。
- ****Deep Learning Specialization****: 一位成员询问，对于刚刚完成基础学习的人来说，是否推荐参加 [Coursera 上的 Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)，尽管这是一个**付费课程**。
   - 建议先理清**基本原理**，然后进入高级主题，从 **Python** 开始，接着是 ML，最后是 Deep Learning。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1360362393883902152)** (2 条消息): 

> `TLDR Service` 


- ****TLDR Service** 获得原生集成**: 一位成员正在直接为其 **TLDR service** 构建原生集成。
   - 这可能会增强该服务提供简洁摘要的能力。
- **另一个话题**: 讨论了另一个话题。
   - 关于该话题的更多细节。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1360404838193561650)** (20 条消息🔥): 

> `Universal Intelligence protocols released, Speaker Isolation Toolkit, MLX EAGLE-2 Speculative Decoding, gpu-spaces script, SwissKnife request for comment` 


- **Universal Intelligence 协议发布**: 一个名为 [`universal-intelligence`](https://github.com/blueraai/universal-intelligence) 的新项目已发布，包含 **3 个针对模型、工具和 Agent 的开源协议**。
   - 它包含一个**基于社区的现成组件库**，可供即时使用和部署，旨在实现简单、可组合、便携、可扩展且自动优化的 AI 使用。
- **使用新工具包隔离发言者**: 一个新的 [speaker-identification-toolkit](https://github.com/ThatJeffGuy/speaker-identification-toolkit) 有助于在多发言者录音中隔离发言者，用于创建 ML 音频数据集。
   - 一旦手动识别了 **10%** 的数据，该工具包就会使用 **CUDA** 或 **CPU** 自动隔离其余文件的发言者。
- **GPU Spaces 过滤器节省筛选时间**: 一位成员创建了 [一个脚本](https://huggingface.co/spaces/DeathDaDev/gpu-spaces)，用于过滤 Hugging Face Spaces，以显示那些启用了非零 GPU 的项目。
   - 这有助于用户筛选精选 Spaces 并**找到具有可用 GPU 资源的项目**。
- **Deep Search Agent 寻找早期采用者**: 一个使用 *smolagents* 专注于 **Deep Search** 的新 Agent 已经建成，正在 [agent.galadriel.com](https://agent.galadriel.com/) 寻找早期测试者。
   - 欢迎反馈，并请向产品团队提出问题和想法。
- **使用 MLX 进行 EAGLE-2 Speculative Decoding**: **MLX EAGLE-2 Speculative Decoding** 的更新代码将 mlx_lm speculative decoding 的吞吐量从 **18** tps 提高到了 **22** tps，详见 [此仓库](https://github.com/0seba/mlx-eagle-spec-decoding-tree/tree/main)。
   - 由于资源有限，开发者寻求在更大的模型上进行实验，特别是针对架构 >=M3 的模型。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1361448243652984862)** (2 messages): 

> `Society of Minds framework` 


- ****Society of Minds 框架**讨论即将开始！**: 本周将举行一场读书会语音聊天，讨论 **"Society of Minds" 框架**，并提供了 [Discord 活动](https://discord.com/events/879548962464493619/1351984543376085062)链接。
- **提供了论文链接**: 待讨论的论文可在 [OpenReview.net](https://openreview.net/pdf?id=zj7YuTE4t8) 获取。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1360642900760727564)** (2 messages): 

> `rf-detr-uslsohoy, CV Hangout` 


- **社区讨论 rf-detr-uslsohoy**: 一位成员向社区分享了一个 [GitHub 仓库](https://github.com/egorsmkv/rf-detr-uslsohoy)。
- **CV Hangout 地点咨询**: 一位成员询问链接中的 **rf-detr-uslsohoy** 仓库是否会在 16 日主持 **CV Hangout**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1360680923330248785)** (2 messages): 

> `Local LLM Models, OlympicCoder 7B and 32B, DeepSeek R1 Distil Qwen 7B, DeepCoder 14B Preview, fine tuning facebook nllb language translator model` 


- **用户测试本地 LLM 模型**: 一位成员想了解其他人正在使用哪些**本地 LLM 模型**，主要用例是**编程、推理、规划和写作**。
- **讨论 OlympicCoder 模型**: 该成员测试了 **OlympicCoder 7B 和 32B**，发现它们在编程方面表现尚可，但容易变得啰嗦，可能需要调整一些设置。
   - 他们还尝试了 **DeepSeek R1 Distil Qwen 7B**，发现其推理能力更强，且不会废话连篇。
- **微调 facebook nllb 语言翻译模型**: 该成员目前正致力于将 **facebook nllb 语言翻译模型**微调到一个包含英文句子和 **Tibeto-Burman 语言**平行翻译的 **csv 文件**上，并希望能与相关人员交流。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1360338024726265897)** (7 messages): 

> `Agent course deadline, Agent course use cases` 


- **Agent 课程截止日期传闻澄清**: 多位用户询问了关于 **Agent 课程**完成和认证的截止日期，特别提到了 **5 月 1 日**。
   - 一位成员澄清说*没有截止日期*，参与者可以按照自己的节奏完成课程。
- **Agent 课程用例：揭开谜团**: 一位用户对 **Agent 课程**的**用例作业**表示困惑，称他们*一个也没找到*。
   - 这表明课程的实际应用部分可能存在清晰度或可访问性不足的问题。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1360371607566287112)** (41 条消息🔥): 

> `课程认证、HF API 限制问题、Ollama 设置与使用、面向 Agent 的 LLM 微调、SmolAgent 对教皇的执念` 


- **课程认证状态仍不明朗**：参与者对*报名*该课程的价值表示质疑，因为进度落后且缺乏关于最终认证的沟通。
   - 有人建议专注于学习材料而忽略认证方面，除非 HuggingFace 提供更新信息；另一些人则对**用例作业 (use case assignment)** 以及如何提交感到好奇。
- **用户遇到 HF API 限制问题**：许多用户很快就达到了 **Hugging Face API 限制**，即使是高级订阅用户也是如此，这导致教程笔记本出现问题。
   - 建议的解决方案包括切换到 **Gemini** 或使用 **Ollama** 本地运行模型以绕过这些限制，尽管本地设置可能需要调整代码以适应模型局限性。
- **Ollama 解决了部分本地 LLM 挑战**：成员们讨论了使用 **Ollama** 本地运行模型，并分享了下载和运行特定模型（如 `qwen2.5-coder:32b`）的命令。
   - 一位成员提供了一段代码片段，演示了在初始化 `CodeAgent` 时，如何指定 **Ollama provider** 来使用本地托管的模型（如 `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF`）。
- **适用于 Agent 框架的 Instruct 模型**：成员们讨论了 **LLM** 是否需要针对 Agent 工具进行特定的微调，结论是任何 Instruct 模型都可以，且模型越大效果越好。
   - 然而，使模型匹配框架的语法或调整 Prompt 可以改善结果，因为较新的模型可能在训练中包含了工具调用/类 Agent 行为的示例。
- **Agent 对教皇年龄的执着**：一位用户报告称，当使用 `llama3`、`deepseekr1:8b` 和 `qwen2.5-coder:latest` 等模型在本地运行时，他们的 **Agent** 莫名其妙地执着于寻找**教皇的年龄**并将其平方得到 **0.36**。
   - 该问题被怀疑源于 **smolagent** 默认 Agent 工具 Prompt 中的硬编码示例，因为在使用 **HfApiModel** 时并未出现此问题。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1360449822573723750)** (54 条消息🔥): 

> `模型相似性分析、Dataloader 批处理策略、多 Token 预测、输入依赖的 LoRA、化学特征的立体异构体编码` 


- **模型看起来有些相似**：一位成员在比较序列间的 MLP 后隐藏状态余弦相似度时，对*不同模型的相似程度感到惊讶*，他使用了一个简单的脚本 [在此获取](https://github.com/pwspen/lmsim)。
   - 他们发现小模型更多地按类型分组而非颜色，而大模型在按颜色排序方面表现得更加一致。
- **重复数据？绝不！**：建议不要在 minibatch 中重复数据，因为*这会导致严重问题！*
   - 一位用户分享了关于认知科学和 ML/AI 领域的调查信息分析，促进了跨学科的见解，并将这些见解传达给不同群体。
- **寻求多 Token 预测论文**：一位成员寻求关于 LLM 推理过程中多 Token 预测的论文，另一位用户推荐了 [DeepSeek v3](https://openreview.net/forum?id=pEWAcejiU2)。
   - 还有用户指向了[这篇论文](https://arxiv.org/abs/2401.10774)，并回忆起几年前看过 Meta 的一篇相关论文。
- **探索输入依赖的 LoRA MoE**：关于输入依赖的 LoRA，一位成员在 **RWKV** 中广泛使用它们以节省参数，并建议探索 **MoE LoRA**。
   - 另一位成员澄清说，虽然他们在 **RWKV** 中使用 LoRA 代替普通的线性变换，但权重本身并不依赖于输入，这引发了关于 LoRA 权重对输入敏感的潜在架构讨论，并与 **MHA** 进行了类比。
- **LLM 进阶者获得通用建议**：一位来自 Amazon 和 Salesforce 的软件工程师寻求参与 **LLM 研究**或项目的方法。
   - 一位成员提供了通用指南，包括学习 Python、应用 ML/AI，然后专注于某个子领域，如机器人、ML 理论、音频、视觉、语言或可解释性。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1360401027257340068)** (236 条消息🔥🔥): 

> `arXiv 推荐请求，研究中的 AI 生成内容，Token 损失与长度外推，显微镜图像的视觉自回归模型，政策执行` 


- **寻求 arXiv 推荐，揭示伦理 AI 结构控制框架**：一名成员为其 [arXiv 提交论文](https://arxiv.org/abs/2504.05518v1) (cs.AI / cs.CY) 寻求推荐，该论文提出了一种**基于自然语言的制度框架**，用于 **ChatGPT** 以*稳定输出*并*确保伦理一致性*。
   - 该框架包括**基于 Prompt 的结构验证和对齐逻辑**，无需修改模型。
- **Discord 应对大量 LLM 驱动的“研究”涌入**：成员们对被呈现为研究的 **AI 生成内容**的兴起表示担忧，这些内容的特点通常是*虚构术语*以及*与正规研究思路缺乏一致性*。
   - 建议包括**封禁隐藏 AI 使用的恶意用户**，以及对表现出经验不足的**善意用户进行长期禁言**。
- **Test-Time Scaling 与 CoT 受到质疑，是否为 RL 的产物？**：一名成员质疑了 *Test-Time Scaling* 和生成*极长思维链 (CoTs)* 的必要性，认为这可能是 **RL 训练方法**的产物，如[这些论文](https://arxiv.org/abs/2504.04383), [https://arxiv.org/abs/2504.01296](https://arxiv.org/abs/2502.07266), [https://arxiv.org/abs/2503.20783], [https://arxiv.org/abs/2503.04697) 所示。
   - 其他人认为 **CoT** *并非关乎实际生成的 Token*，而是模型通过操纵 Attention 权重来执行*更多迭代/计算*的一种方式。
- **外推长度以提升模型收益，还是仅仅维持？**：成员们讨论了**长度外推**的挑战，指出模型在超过其训练序列长度后，往往*无法持续降低 Token 损失*，如[此图](https://cdn.discordapp.com/attachments/747850033994662000/1361359728147431685/Screenshot_2025-04-14_at_16.17.22.png?ex=67fe788c&is=67fd270c&hm=4fe0f240d28501a17e80c46f6c0848297dd2361ec50593f31cc697d50bccd0e5&)所示。
   - 提到了 **NoPE + SWA** 和 **ssmax** ([Super Scaling Max Activation](https://arxiv.org/abs/2501.19399)) 等技术作为潜在解决方案，以帮助模型记住比其序列长度更远的内容，尽管关于最佳信息流策略仍存在争议。
- **VAE 自回归模型生成的显微镜图像**：一名成员分享了使用视觉自回归模型生成 **3 通道显微镜图像**的结果，该模型以训练好的 **DINO** 模型的类别嵌入 (class embeddings) 为条件。
   - 生成的图像呈现出明显的**偏白影调**，这可能是由于**在 ImageNet 上训练的 VAE 视觉编码器**带来的偏差。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1360536090246778920)** (14 条消息🔥): 

> `图归因机械可解释性，蒸馏对模型电路的影响，模型对其电路的了解，推理模型的自我意识，推理模型中的 CoT 保真度` 


- **呼吁进行图归因机械可解释性研究**：一名成员建议对新的推理模型进行**图归因机械可解释性 (graph attribution mechanistic interpretability)** 研究，并指出电路 (circuits) 与模型解释之间的差异，参考了[这篇论文](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)。
- **蒸馏影响电路知识**：成员们担心**蒸馏 (distillation)** 可能会减少模型对其电路的了解，认为像 **Llama** 和 **Qwen** 这样的蒸馏模型可能对电路的感知更少。
- **关于模型对电路的自我意识引发辩论**：一名成员质疑模型是否对其电路有任何了解，挑战了在没有充分证据的情况下夸大模型理解能力的倾向。
   - 另一名成员认为，推理模型可能会通过 **RL** 获得一些关于解题策略的知识，从而可能导致对其电路更好的自我理解。
- **量化模型自我知识：校准难题**：成员们讨论了量化模型自我知识极限的方法，参考了一篇[校准论文](https://arxiv.org/abs/2504.06564)，该论文评估了模型自述置信度与其实测准确率的相关性。
- **探测后期层中的 CoT 调用**：成员们讨论了探测允许 **Agent** 调用 **CoT** 或在触发动作前检索知识的 Token 概率，参考了[这篇论文](https://arxiv.org/abs/2504.03553)，并指出做出此类决定的过程主要出现在后期层中。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282504648511499/1361376237486080111)** (99 条消息🔥🔥): 

> `Karpathy 问 ChatGPT 尴尬的问题，Thinking Machines 20 亿美元融资，OpenAI SWE 即将到来，GPT 4.1 Quasar 发布，DeepSeek Inference Engine 开源` 


- **Karpathy** 试图让 **ChatGPT** 难堪：一位用户分享了一个 [Prompt](https://x.com/karpathy/status/1910734302931017812) 询问 **ChatGPT**：*你知道关于我的最尴尬的事情是什么？*
   - 该用户鼓励其他人通过多轮提问，迫使 **ChatGPT** 给出诚实且直接的回答。
- **Mira** 为 Thinking Machines 筹集 **20 亿美元种子轮**融资：根据引用 [Fortune 文章](https://fortune.com/2025/04/10/mira-murati-2-billion-seed-raise-ai-boom-economic-chaos/) 的讨论，**Thinking Machines** 正在进行 **20 亿美元的种子轮**融资，由 **Alec Radford** 担任顾问。
   - 一位用户发布了来自 [Epoch AI](https://x.com/epochairesearch/status/1910788295405252916) 的一张*不错的图表*，展示了这次融资。
- **DeepSeek** 开源其 **Inference Engine**：**DeepSeek** 已开源其推理引擎，[GitHub 仓库](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md) 已可供查阅。
   - 成员们想知道今天是否有人想聊聊 **DeepSeek** 的开源。
- **GPT-4.1** 和 **Quasar** 发布传闻：围绕 **GPT-4.1** 和名为 **Quasar** 的模型的发布展开了讨论，参考了 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1jz2jhu/openai_confirmed_to_be_announcing_gpt41_in_the/) 和 [官方公告](https://openai.com/index/gpt-4-1/)。
   - 有推测认为 **GPT-4.1** 可能会成为事实上的编程模型，取代 **Gemini 2.5**，随后讨论了 **GPT-4.5** 的弃用问题。
- **Grok** 添加功能但保持沉默：一位用户注意到 **Grok** 在没有发布公告的情况下上线了重要功能，包括*跨对话记忆*和新的 *workspaces* 功能。
   - 推测认为他们可能在进行更大规模的发布前进行低调测试。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1361376237486080111)** (2 条消息): 

> `Quasar 发布，SFCompute pod` 


- **Quasar 观看派对即将开始**：Latent Space 今天将在 35 分钟后举办另一场 **Quasar 发布**观看派对，活动地点在 [此 Discord 活动](https://discord.gg/rPJq8UU2?event=1361376118510321724)。
- **SFCompute Pod 需要助力**：Latent Space 请求帮助转发他们的 **SFCompute pod**。
   - 更多详情请见 [此推文](https://x.com/latentspacepod/status/1910777555101376757)。


  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1360341267950600262)** (5 条消息): 

> `X-Ware.v0, AI 新闻来源` 


- **X-Ware.v0 在 X 上发布**：一位成员分享了来自 [X-Ware.v0](https://xcancel.com/dylan522p/status/1911843102895358198) 的图片，并询问*这是哪里的？*。
- **AI 新闻来源**：针对图片来源的问题，一位成员简单地回答道：*ainews*。


  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1360343642702872590)** (186 条消息🔥🔥): 

> `Agent 定义, Langsmith 工具, 训练过程可见性, 模型 Benchmarking 工具, GPT-4.1 发布` 


- **Agent 定义被“凭感觉”定义 (vibe-defined)**：成员们就“Agent”的定义展开了辩论，其中一人建议当今的定义是：*一个 LLM 调用一个工具*；而另一人展示了一个关于自我改进 Agent 的 [Figma 画板](https://www.figma.com/board/aCaUWEr039dHmpW9ssGJmK/self_improving_agents?node-id=137-796&t=zsQXjScFlAekKtEd-1)。
   - 有人建议：*Agent 就是你在无聊的会议中凭感觉写出的代码 (vibe code)*。
- **Langsmith 工具广受好评**：成员们讨论了将 **Langsmith** 作为 LLM 仪表化工具的使用，有人提到即使是非 Langchain 项目他们也很喜欢使用它，并链接到了 [Arize 文档](https://docs.arize.com/arize)。
   - 另一位成员建议使用一个预先写好的 Bot 来标记聊天中的链接以便后续处理，但指出“抱怨”的成本更低。
- **训练过程的可见性**：在一次现场演示中，一位成员询问如何获得神经网络训练过程的可见性，引发了围绕工具和方法的讨论。
   - 建议包括使用 **WandB** ([Weights & Biases](https://wandb.ai/site)) 和 **TruLens** ([trulens.org](https://www.trulens.org/)) 来进行 LLM 追踪和评估。
- **模型 Benchmarking 工具**：成员们讨论了各种用于比较模型的 Benchmarking 工具，包括来自 Hugging Face 的 [lighteval](https://github.com/huggingface/lighteval) 和 [BenchmarkAggregator](https://github.com/mrconter1/BenchmarkAggregator?tab=readme-ov-file)。
   - 一位成员提到，这些工具在评估循环中比较参数时非常有用。
- **OpenAI 发布 Quasar**：在 OpenAI Quasar 发布观看派对期间，成员们讨论了 **GPT-4.1** 的特性，包括其相对于 Claude 极具竞争力的定价，以及长输入上下文的固定定价，参考了 [定价文档](https://platform.openai.com/docs/models/gpt-4.1)。
   - 一位成员强调最便宜的模型可以免费使用 7 天，另一位成员则开玩笑说在演示期间喝着 Windsurf 广告。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1360342198318661633)** (18 条消息🔥): 

> `非确定性 NLM, NLM 在教育中的应用, Gemini Education Workspace, 对话式界面, 大学版 NotebookLM` 


- ****NLM 的 Latent Space 导致变异性****：一位成员表示，*Latent Space 的变异性*导致无法每次生成相同的输出，且系统缺乏连贯性，导致每次根据输入生成的内容具有随机性。
- ****NLM 不是法拉利****：据一位成员所说，你不能指望普锐斯（Prius）成为法拉利（Ferrari），如果你想要法拉利，那将非常昂贵，而且你在 Google NotebookLM 中找不到它。
   - 另一位成员澄清说，NLM 的设计初衷并非是一个确定性系统。
- ****NLM 变革教育****：一位成员在课堂上使用 **NotebookLM**，通过上传幻灯片和材料，创建笔记、带有测验问题的学习指南、术语表、思维导图和音频概览，然后分享给学生以帮助他们准备考试。
   - 他们还让学生以小组形式创建自己的 NotebookLM。
- ****NSW 错失 Gemini****：一位成员询问其他人是否通过 Education Workspace 使用它，因为他们有兴趣了解那些乐于在 Workspace 中使用 **Gemini** 的学区和部门。
   - 他们指出在澳大利亚的 NSW（新南威尔士州），目前还无法使用 **Gemini**。
- ****患糖尿病猫的主人需要 Chatbot****：一位成员经营着一个大型糖尿病猫主人支持小组，希望为成员提供一个针对其文档（包括视频内容）的对话式界面，且支持法语。
   - 他们希望成员能够提出问题，并根据文档获得答案，同时附带相关文档的阅读链接。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1360354819428061335)** (141 条消息🔥🔥): 

> `面向学生的 NotebookLM, Google Agents 与 NotebookLM, Notebook 搜索功能, NotebookLM 中的 Discover 功能, Gemini 中的深度研究问题` 


- **NotebookLM 激发学生兴趣**：一位用户询问如何学习使用 **NotebookLM** 来辅助法国的 MP2I 学习，涵盖数学、编程和物理。
   - 其他人引导该用户关注 **Google Agents** 注册以提升工作环境，如[这段 Youtube 视频](https://youtu.be/WEEYPwBg6Qo)中所述。
- **NotebookLM "Discover" 功能带来惊喜**：一位用户对 NotebookLM 新推出的 **"Discover sources"** 功能表示非常满意，称其为 *“我想要的一切”*。
   - 该用户目前正在期待更多的 **audio overview flavors**，并表示非常喜欢 Grace 的播客。
- **Audio Overviews 仅支持英语**：用户报告称，尽管之前有破解方法，但 NotebookLM 中的 audio overview 功能现在已无法可靠地支持 **English** 以外的语言。
   - 一位仅能生成英语播客的用户反映这很困难，因为英语并非其母语。
- **NotebookLM PDF 深度研究受阻？**：多位用户报告在 **Gemini** 中进行深度研究时遇到麻烦，具体表现为上传到 NotebookLM 的 PDF 无法作为 source 加载。
   - 一位用户建议这可能是暂时性的故障，建议尝试使用其他文档，并确保字数在 **500k word limit** 以内。
- **迷失在翻译中：UI 语言迷宫**：用户报告在更改输出语言后，难以将 **NotebookLM 切换回 English**，相关设置现已丢失，且 UI 语言设置不影响输出。
   - 一位用户确认这是一个已知问题，并发布了指向 [bug channel](https://discord.com/channels/1124402182171672732/1354708696474718218) 的链接，而其他人则建议尝试更改 **Google account language**、清除 cookies 或使用 `?hl=en` 参数。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1360372931166867597)** (109 条消息🔥🔥): 

> `Llama 4 Maverick & Scout, DeepCoder 模型, Nvidia UltraLong 模型, GPT-4.1 价格与性能, Gemini 2.5 Pro` 


- **Llama 4 消耗的 GPU Hours**：成员们讨论了 **Meta** 的 **Llama 4 Maverick** 使用了 **2.38M GPU hours**，与训练 **Deepseek V3** 相当，而 **Llama 4 Scout** 则耗费了 **5.0M GPU hours**。
   - 一些人指出其他模型是针对人类偏好进行微调的，质疑其公平性，而另一些人则提到了 **LeCun** 可能参与其中。
- **DeepCoder 达到顶尖编程性能**：一位成员分享了一篇关于 **DeepCoder** 的 [VentureBeat 文章](https://venturebeat.com/ai/deepcoder-delivers-top-coding-performance-in-efficient-14b-open-model/)，强调了其高效的 **14B** 参数开源模型和增强的 **GRPO algorithm**。
   - 该模型具有 **offline difficulty filtering**、无 entropy loss、无 **KL loss** 以及来自 **DAPO** 的 overlong filtering，尽管训练时使用 **32K**，但可泛化至 **64K context**。
- **Nvidia UltraLong 模型处理超长序列**：**Nvidia** 正在利用研究成果创建 **UltraLong-8B** 模型，如这个 [Hugging Face collection](https://huggingface.co/collections/nvidia/ultralong-67c773cfe53a9a518841fbbe) 所示，该模型基于 **Llama-3.1** 构建，旨在处理高达 **4M tokens** 的序列。
   - 它结合了 continued pretraining 与 instruction tuning，以 **4M sequence length** 和 **2** 的 global batch size 训练了 **150 iterations**。
- **GPT-4.1 Benchmarks 优于以往版本**：成员们讨论了 **GPT-4.1** 的定价和 benchmarks，有人指出 [benchmarks 优于](https://openai.com/index/gpt-4-1/)以往版本，但定价和模型版本命名令人困惑，新模型已在 **GitHub Copilot** 中可用。
   - 还有关于 **4.1-nano** 与优秀的 **14B** 模型并驾齐驱的讨论，以及关于该模型是否会开源的一些猜测。
- **Gemini 2.5 Pro 价格昂贵？**：成员们辩论了使用新版 **GPT-4.1** 是否比使用 **Gemini 2.5 Pro 和 Sonnet 3.7** 更划算。
   - 尽管 Gemini 起初看起来更便宜，但由于缺乏免费的 caching 且倾向于生成冗长内容，实际上可能更贵，而 GPT-4.1 则更加言简意赅。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1360634754164195529)** (15 messages🔥): 

> `H100 Llama 4 Scout 上的 Loss 观测、小模型训练挑战、小模型数据集推荐、Surya 和 SmolVLM2` 


- **在 H100 Llama 4 Scout 上观测到 Loss 上升**：一名成员指出，在 **H100** 环境下训练 **Llama 4 Scout** 模型时，第 **1** 和第 **2** 个 epoch 之间的 loss 从 **1.9011** 上升到了 **2.3407**。
   - 他们感到担心，因为尽管使用了两块 **H100** GPU，loss 并没有像预期那样下降。
- **围绕模型大小展开讨论**：成员们讨论了训练极小模型（**100-200 万参数**，**2000 万 tokens**）的影响，以及这如何影响观测到的 loss。
   - 一位成员建议：*无论任务是什么，你至少应该从 1000 万（10M）参数的模型开始。*
- **小模型的数据集**：一位成员分享了他们从 **Wiki 103 数据集** 切换到微调 **Phi2** 的经验，暗示通过改变方法来解决观测到的训练问题。
   - 该成员表示，他们转向微调 Phi2 以解决训练中遇到的问题。
- **Surya 和 SmolVLM2**：一位成员推荐关注 [Surya](https://github.com/VikParuchuri/surya)，强调它*虽然不是 VLM，但非常令人印象深刻*。
   - 他们还为专门寻找 VLM 的用户推荐了 [SmolVLM2](https://huggingface.co/blog/smolvlm2)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

ee.dd: https://ai-2027.com
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1360451095578546206)** (14 messages🔥): 

> `关于 Repo 的研究论文、任务质量保证` 


- **关于推理 Repo 的研究论文？**：一位成员询问了基于该仓库撰写研究论文的可能性，另一位成员表示有兴趣协作撰写，尽管承认 *“我不太擅长写作”*。
- **请求任务质量验证**：一位成员对确保汇编中任务的质量表示担忧，建议需要进行双重检查，并提议为每个任务创建问题作为需求。
   - 另一位成员询问：*“你认为该如何验证它？”*。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1360343887255703552)** (99 messages🔥🔥): 

> `用于 Reddit 和 Quora 的 MCP、为 MCP server 设置支付悬赏、ADK 和 A2A 对比 MCP、向用户暴露工具、向 LLM 传递工具` 


- **用于 Reddit 和 Quora 的 Graphlit MCP server**：成员们讨论了为 Reddit 和 Quora 构建 **MCP server**，[Graphlit](https://github.com/graphlit/graphlit-mcp-server) 表示如果需要可以添加 Quora 数据摄取功能。
   - 目前已存在一些用于 Reddit 的工具，例如[这个仓库](https://github.com/hawstein/mcp-server-reddit)。
- **计算机科学团队需要帮助运行 MCP server，愿意支付悬赏**：一位成员表示愿意支付悬赏以寻求设置 **MCP server** 的帮助，因为他们的大学计算机科学团队在 **GhidraMCP** 上遇到了困难，返回 *404 - NO CONTEXT PROVIDED* 错误。
   - 该团队正在尝试使用 Cursor IDE 使其正常工作。
- **除了 MCP，ADK 和 A2A 也值得阅读**：一位成员建议探索 Google 的 **ADK 和 A2A**，指出它们与 **MCP** 相似，并可能成为 Agent 互联网的核心，这引发了关于它们相关性和用途的讨论。
   - 另一位成员确认，目前对于非 MCP 的技术讨论没有官方共识，但如果它至少与 AI/ML/MCP 有一定关联，那么应该没有问题。
- **讨论向 LLM 传递工具**：成员们分享了关于如何将与特定用户 prompt 相关的工具传递给 **LLM** 的想法，一位成员分享道，所有启用的工具都会随 prompt 一起传递。
   - 作为替代方案，一位成员分享了一个演示向量工具调用（vector tool calling）的[视频](https://www.youtube.com/watch?v=3ISRS2hQlfI&t=195s)。
- **Wildcard 暂停进一步维护 agents.json**：Wildcard 团队宣布，由于 MCP 被大型模型提供商日益广泛地采用，他们将暂停对 **agents.json** 的进一步维护。
   - 他们相信这些概念最终会整合进 MCP 中，就像最近的无状态 HTTP 传输协议一样。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1360371569410441256)** (36 条消息🔥): 

> `无 Function Calling 的模型、MCP Bug 发现工具、Paprika 食谱 MCP Server、Oterm 发布与 MCP Sampling、用于 Agent 部署的 AutoMCP` 


- **Block 调整缺乏 Function Calling 的模型**：Block 正在实验缺乏 Function Calling 能力的模型，观察是否可以通过调整输出来配合 Agent 工作。这篇 [博客文章](https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim) 探讨了如何在不使用辅助模型的情况下，通过 XML 输出来实现这一目标。
   - 团队正在权衡延迟成本与使用辅助模型进行解析的收益，担忧长会话以及模型遵循 XML 格式的能力，并可能使用 *local model*，但担心会带来更多 *overhead*。
- **用于调试 Copilot 客户端的 MCP 工具**：[synf](https://github.com/strowk/synf) 和 [mcptee](https://github.com/strowk/mcptee) 帮助成员在测试 Copilot 客户端时发现并修复 Bug，该客户端在处理长上下文和更多工具时可能会遇到困难。
   - 一位成员在开发时考虑了高性能硬件，因为 *多次 API 调用总是比单次调用慢*。
- **Paprika 食谱应用获得 MCP Server 支持**：为 **Paprika 食谱应用** 的用户创建了一个 MCP Server，使得 Claude 可以通过 [这个 GitHub 仓库](https://github.com/soggycactus/paprika-3-mcp) 自动将食谱保存到 Paprika 中。
   - 未提供更多信息。
- **Oterm 终端客户端支持 MCP Sampling**：Ollama 的终端客户端 [oterm](https://github.com/ggozad/oterm) 发布了 **0.11.0** 版本，重点增加了对 [MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) 的支持，此外还支持现有的 **MCP tools** 和 **MCP prompts**。
   - 新版本包括对 **sixel graphics**、**应用内日志查看器** 的支持，以及从终端运行 **自定义命令** 的能力。
- **AutoMCP 简化 Agent 部署**：推出了名为 **AutoMCP** 的新库和平台，可轻松将现有的 Agent 项目转换并部署为 MCP Server，代码托管在 [GitHub](https://github.com/NapthaAI/automcp)，并部署在 [此平台](https://labs.naptha.ai/)。
   - 该服务为 AI Agent 提供类似 Vercel/Heroku 的体验，允许用户在熟悉的框架中进行原型设计并部署，无需担心后端问题，详见 [此 YouTube 视频](https://www.youtube.com/watch?v=El5YvBQ5py0)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1360331698511941722)** (18 条消息🔥): 

> `Python/PyTorch 中的 CUDA、AMD GPU Mode 竞赛、marksaroufim 的 GTC 演讲、Stephen Jones 的视频、频道所有者` 


- **CUDA 指导方案明确化**：一位成员询问了 Python/PyTorch 模型中 **CUDA** 的参考资料，另一位成员分享了他们最近关于该主题的 [GTC 演讲](https://docs.google.com/presentation/d/1sipZ_sqdwJapHQAr23yBow43pF40lMZu/view?usp=sharing)。
   - 演讲建议 *custom ops* 和 *load inline* 应该能解决大部分问题，同时还在持续努力缩短编译时间；该演讲也可以在 [nvidia.com](https://www.nvidia.com/en-us/on-demand/session/gtc25-s71946/) 上找到。
- **AMD GPU Mode 竞赛待续**：一位成员询问了 **AMD GPU Mode Competition** 的情况，表示他们在几天前注册后未收到任何更新。
   - 另一位成员回复称 *今天会有更多信息*。
- **Stephen Jones 的视频引发关注**：在观看 GTC 演讲后，一位成员深入研究了演讲中提到的 **Stephen Jones** 的视频。
   - 该成员随后表示 *假期结束了*，*讲座重新开始*。
- **需要联系频道所有者**：一位成员询问谁是频道所有者并请求管理员介入。
   - 另一位成员回复说可以艾特 <@&1231246776103604326> 并询问他们需要什么。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1360545047430434978)** (5 条消息): 

> `Morton Order vs Swizzle2D, Space-Filling Curves, Hilbert Curves vs Morton Ordering, Debugging Triton Memory Leaks, Implementing Triton Kernel` 


- **Hilbert Curves 与 Morton Ordering 竞争**：一位成员询问了比 **Morton order** 缓存友好性更好的空间填充曲线，引发了关于 [Hilbert Curves](https://en.wikipedia.org/wiki/Hilbert_curve) 等替代方案的讨论。
   - 另一位成员指出，*从理论上讲*，**Hilbert Curves** 是最优的，但在硬件效率上不高，建议 **Morton ordering** 是更好的实际折中方案，并指向了一篇比较两者的 [博客文章](https://blog.stackademic.com/how-the-idea-of-the-hilbert-curve-inspired-morton-curves-for-gpu-performance-4e235d670304)。
- **使用 Hilbert Curves 的 GEMM 性能比较**：一位成员分享了一个 [GitHub repo](https://github.com/lawmurray/gpu-gemm)，展示了使用 **Hilbert curves** 的 GEMM 实现，以及 [针对 cuBLAS 的基准测试](https://indii.org/blog/gpu-matrix-multiply/)。
   - 基准测试表明，随着矩阵尺寸的增加，**Hilbert curves** 变得更加有效。
- **调试 Triton Kernel 内存泄漏**：一位成员寻求关于排查 **Triton kernel** 内存泄漏的建议，该 kernel 通过了准确性检查，但在训练期间导致显存溢出 (out-of-memory) 错误。
   - 该成员强调了与 eager mode 相比不一致的前向传播结果，怀疑存在潜在的溢出问题，并链接了 repo [FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa)。
- **寻求实现 Triton Kernel 的指导**：一位成员请求有关实现 **Triton kernel** 以使用专家检索架构训练模型的资源。
   - 尽管查阅了官方文档，他们仍感到困难，并链接了论文 [Retrieval meets Long Context LLMs](https://arxiv.org/pdf/2407.04153)。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1361163559236796578)** (8 条消息🔥): 

> `Dynamic KV cache tensors in CUDA, cuBLAS Batched GEMM, memcpy_async cooperative API, Async copies and uncoalesced global access, Shared memory alignment` 


- **CUDA 矩阵乘法中的动态 KV Cache 挑战**：围绕在 QK.T 和 XV 操作期间如何高效处理 CUDA 中的动态 KV cache 张量展开讨论，特别是如何使用 [cuBLAS 中的 Batched GEMM](https://developer.nvidia.com/cublas) 为 Batch Size 为 M 的每个用户管理变化的 `K-cache-length`。
   - 用户询问通常是编写自定义 kernel 来管理此问题，还是 cuBLAS 中的 batched GEMM 能够处理可变的 `K-cache-length`。
- **`memcpy_async` 降低 Kernel 性能**：一位用户报告称，在从标准内存复制循环切换到 `cuda::memcpy_async` 后，性能显著下降，尽管生成了正确的 `LDSTS` 指令。
   - 他们观察到，尽管使用了与非异步版本相同的索引，异步 kernel 仍报告非合并 (uncoalesced) 的全局访问，从而引发了关于异步复制正确用法的疑问。
- **`memcpy_async` 需要协作式 API**：有人建议 `memcpy_async()` 是一个协作式 (cooperative) API，这意味着所有线程必须传递相同的指针和对应于整个内存块的大小，参考了 [官方 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-barrier)。
   - 从每个线程顺序执行此操作会 *阻止合并访问 (coalescing)*，而不是启用它。
- **`memcpy_async` 对齐问题**：引用了一篇 [论坛帖子](https://forums.developer.nvidia.com/t/coalesced-and-conflict-free-memory-access-using-cuda-memcpy-async-cp-async/306460/6)，指出 `memcpy_async` 的潜在问题包括 Shared Memory 地址的对齐以及指令周围的条件判断，这些都会阻碍合并内存访问。
   - 循环也可能存在问题。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1361113122622144655)** (6 条消息): 

> `Memory profiling distributed training, ATen attention.cu, torchscript jit CUDA optimizations, ZeRo Stage 3 PyTorch Lightning tutorial` 


- **SLURM 集群上的内存分析困扰工程师**：一位工程师正在寻求关于在拥有 **8 个节点**（每个节点 **8 个 GPU**）的 **SLURM 集群**上进行分布式训练模型内存分析（Memory Profiling）的建议。
   - 他们是第一次进行这种类型的分布式训练，因此正在寻找推荐的路径。
- **ATen 的 attention.cu 实现研究**：一位工程师询问了 ATen `attention.cu` 中特定行所指向的实现（[GitHub 链接](https://github.com/pytorch/pytorch/blob/101c4f482a4019896ca18184233bd27a758648bf/aten/src/ATen/native/transformers/cuda/attention.cu#L662)）。
   - 具体来说，他们旨在了解 torch/CUDA 如何处理 batch 中单个用户的操作数 `[dHead x K-cache-length]`，以及 `bmm_nt` 是调用 cuBLAS Batched GEMM 来拆分大型 matmul，还是有其他替代机制。
- **Nested Tensor Matmul 管理可变 Cache 大小**：一位成员认为他们找到了处理 batch 中可变 Cache 大小和单个 Cache 的位置（[GitHub 链接](https://github.com/pytorch/pytorch/blob/6dddd6520daf8768dc76d182d3b2b0130e87a49d/aten/src/ATen/native/nested/NestedTensorMatmul.cpp#L151)）。
   - 他们希望自己的理解是正确的，并且该实现符合他们的想法。
- **征集 ZeRo Stage 3 教程**：一位成员询问是否有人可以分享关于在 **PyTorch Lightning** 中实现 **ZeRo Stage 3** 的教程。
   - 未提供进一步的讨论或细节。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1360650228205883512)** (1 条消息): 

> `RMSNorm vs L2 Norm, Llama Norm, Scout Embeddings` 


- **RMSNorm 伪装成 L2 Norm**：一位成员澄清说 **Llama** 并不使用 **L2 norm**；它使用的是不带缩放的 **RMSNorm**，并将其称为 **L2**，而实际的 **L2(x) = sqrt(sum(x^2))**。
   - 他们指出 **Llama norm** 是 **sqrt(sum(x^2)/n)**，其中 *n* 是 embedding 维度，这导致对于 scout，**-n <= qk^T <= n**，其中 *n=8192*。
- **Scout Embedding 维度澄清**：讨论强调，对于 **Scout**，**Llama norm** 计算中的 embedding 维度 *n* 等于 **8192**。
   - 这一澄清强调了适用于 **Scout** 架构上下文的特定数值范围 **-8192 <= qk^T <= 8192**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1360805799961559141)** (18 条消息🔥): 

> `CUDA events, Maxwell tuning guide, shared memory, PTX and SASS, LOP3.LUT` 


- **CUDA events 同步并非必须**：根据一位成员的说法，如果使用 **CUDA events** 进行计时，则不需要同步；但如果使用 **host 端计时**，则需要同步。
   - 该成员表示：“我不知道 PyTorch 是怎么做的，但如果他们使用 CUDA events 进行计时，就没有理由进行同步。如果他们使用 host 端计时，则确实需要在中间进行同步。”
- **Maxwell 调优指南：Block 分配**：Maxwell 调优指南建议为一个 block 分配不超过 **32K**（在可用的 48KB 中），以便 **2 个 block** 可以容纳在一个 **SM** 中。
   - 另一位成员解释说，如果每个 **SM** 只有一个 block 且使用 **block-synchronization**，当大多数 warp 已经在 barrier 处等待而少数 warp 仍在工作时，性能将达不到最优。
- **NVIDIA 的 PTX ISA 文档分享**：一位成员分享了 NVIDIA 关于 **PTX ISA** 的文档，该文档内容详尽，对于学习 PTX 非常有用。
   - 相关资源链接见 [此处](https://docs.nvidia.com/cuda/parallel-thread-execution/#introduction)。
- **逆向工程 SASS 指令**：由于缺乏官方文档，理解 **SASS** 需要进行逆向工程并搜索 NVIDIA 论坛，特别是寻找前 NVIDIA 员工 **njuffa** 的见解。
   - 解释 `LOP3.LUT` 指令的一个示例线程可以在 [此处](https://forums.developer.nvidia.com/t/what-does-lop3-lut-mean-how-is-it-executed/227472/3) 找到。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1360672489184956446)** (3 messages): 

> `QLoRA 训练，4bit 量化，模型全层 QAT` 


- **使用低于 4-bit 量化的 QLoRA 训练**：一位成员询问了关于使用 *低于 4-bit 量化* 进行 **QLoRA** 风格训练的相关文献。
   - 另一位成员提供了关于该主题的[链接](https://mobiusml.github.io/1bit_blog/)。
- **寻求 QAT 论文**：一位成员正在研究模型所有层的 **QAT** (*Quantization Aware Training*)。
   - 该成员请求推荐关于该主题的优秀论文。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://core-math.gitlabpages.inria.fr/
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1360333515027120289)** (8 messages🔥): 

> `AMD GPU，云服务商，Profiling，vast.ai，shadeform` 


- **AMD GPU 云服务探索开始**：成员们正在寻求提供支持 **Profiling** 功能的 **AMD GPU** 云服务商推荐，虽然提到了 [vast.ai](https://vast.ai)，但指出其缺乏对 AMD 的支持。
   - 一位成员提到，大多数云厂商通常会禁用 **Nvidia GPU** 的性能分析硬件计数器访问权限，除了少数几家，如 **lightning.ai**。
- **Vast.ai 禁用 Profiling**：一位成员提到，虽然 [vast.ai](https://vast.ai) 是一个推荐选项，但它**不允许进行 Profiling**。
   - 然而，另一位成员指向了[之前的消息](https://discord.com/channels/1189498204333543425/1349310333520711721/1349464109183139902)，暗示在那里设置 Profiling 可能是可行的，尽管他们自己还没试过。
- **关于 Shadeform 的提问**：一位成员询问是否有人有使用 **shadeform** 的经验。
   - 另一位成员表示感兴趣，并说道：*好问题，让我去了解一下*。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1360423483934249071)** (7 messages): 

> `Profiling Metal Kernel，朴素 vs. 合并矩阵乘法，内存使用差异，M 系列芯片上的 Unified Memory 与分页` 


- **合并 Kernel 使内存减半**：一位成员发现，在 Metal 中实现的全局内存合并（coalesced）矩阵乘法所使用的内存仅为朴素版本的一半，尽管速度仅略快一点。他参考了[这个 CUDA MMM 实现](https://siboehm.com/articles/22/CUDA-MMM)进行测试。
   - 附带的图片展示了 Metal Kernel 的 Profiling 结果，显示合并版本在内存使用上有显著减少。
- **内存差异疑因分页引起**：一种解释认为，操作系统以页（pages）为单位提取数据，而非合并访问会导致分页利用率低下，即提取的数据中只有一小部分被实际使用。
   - 其他人指出，**M 系列芯片拥有 Unified Memory**，这应该会消除 CPU 和 GPU 之间的分页，尽管从 Unified Memory 到 Shared Memory 的数据移动可能仍涉及分页。
- **聚焦 M3 Pro 芯片**：原作者澄清说他们使用的是 **M3 Pro 芯片**，这表明 Unified Memory 架构与观察到的内存行为有关。
   - 那位提出内存差异建议的成员，之前误解了他们是在 M3 芯片上进行的实验。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1360457071350251670)** (11 messages🔥): 

> `OptiLLM 推理代理，快速 Prefix Sum，Thread Coarsening，SwissKnife WebGPU GraphRAG` 


- **OptiLLM 优化准确率与性能**：[OptiLLM](https://github.com/codelion/optillm) 是一个兼容 **OpenAI API** 的优化推理代理，它实现了几种可以提高 LLM 准确率和性能的前沿技术。
- **Prefix Sum 非常快**：一篇新的博客文章展示了如何为分块扫描（blockwise scan）操作实现高性能，最快的 Kernel 达到了 **93% 的 GPU 利用率**。
   - 作者建议查看 [Juan Gómez Luna 的讲座](https://www.youtube.com/watch?v=SG0gvcbf2eo) 以了解 Prefix Sum 的基础知识，并提供了 [博客文章](https://veitner.bearblog.dev/making-prefix-sum-really-fast/) 和 [代码](https://github.com/simveit/effective_scan) 的链接。
- **Thread Coarsening 提升 Prefix Sum 性能**：一位成员提到，最后一个 Kernel 使用的技术被称为 *Thread Coarsening*，正如 **PMPP 书籍**中所描述的那样。
   - 该书的相关章节由 Luna 教授编写，涵盖了双缓冲技术，可在此处 [获取](https://www.sciencedirect.com/science/article/abs/pii/B9780323912310000227)。
- **SwissKnife：WebGPU GraphRAG 即将推出**：**SwissKnife**（claude code (apis) + WebGPU + GraphRAG + GraphOfThoughts）在开始主要开发前正在征求意见。
   - 仓库链接和架构文档可在此处 [获取](https://github.com/endomorphosis/swissknife/tree/blueprints) 和 [此处](https://github.com/endomorphosis/swissknife/tree/blueprints/docs/architecture)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1361306646298755233)** (2 messages): 

> `用于 Kernel 代码生成的 LLM，用于 Kernel 优化的 RL` 


- **探索用于 Kernel 代码生成的 LLM**：一位成员正在探索一种简单的、基于 Next Token Prediction 训练的 **LLM**，用于 **Kernel 代码生成**。
   - 该模型将使用模拟器或真实硬件进行 **RL** 对齐，以在不同的硬件配置上编译、运行并评估 Kernel 性能。
- **将 LLM 与 RL 对齐以进行 Kernel 优化**：该成员设想将 **LLM** 与 **强化学习 (RL)** 对齐。
   - 这种对齐将利用模拟器或真实硬件来编译、运行和评估跨不同硬件设置的 Kernel 性能，旨在实现优化的 Kernel 代码生成。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1360380897198870841)** (7 messages): 

> `vectoradd, grayscale, Modal 运行器` 


- **Vectoradd 基准测试遥遥领先**：多个向 `vectoradd` 排行榜提交的作品在包括 **L4, A100, 和 H100** 在内的各种 GPU 上使用 **Modal 运行器** 取得了成功。
- **Grayscale 基准测试获得 Modal 助力**：一个向 `grayscale` 排行榜提交的基准测试在 **H100** GPU 上使用 **Modal 运行器** 取得了成功。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

eriks.0595: <@349565795711451146> 我们更新了评分器（grader），你能告诉我这个问题是否已经修复了吗？
  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1360399324646871050)** (6 messages): 

> `Python vs CUDA 提交，自动包装 CUDA 文件，Profiling 工具，QoL 改进` 


- **CUDA 提交自动包装即将推出？**：团队讨论了如何自动将 **CUDA (.cu) 文件** 包装在带有 **load_inline** 的 Python 文件中，以便更轻松地提交。
   - 虽然可能赶不上发布前，但*已在计划中*，同时还有其他 *QoL 改进*。
- **Profiling 工具即将到来**：成员们表达了对 **Profiling 工具** 以及自动包装 CUDA 提交功能的需求。
   - Profiling 工具*绝对在 [某位成员] 个人希望拥有的功能列表之列*。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1360330720895307838)** (8 条消息🔥): 

> `挑战赛注册、Discord ID 提交、注册后的确认邮件` 


- **挑战赛注册困惑已解决**：已明确挑战赛可以涉及旧的工作，但最初只需完成注册即可；预计稍后会有一个包含更新内容的邮件列表。
   - 一位成员表示，*注册就足够了，稍后可能会有一些包含更新的邮件列表，但据我所知目前没有立即需要做的。*
- **重复提交 Discord ID**：一位成员询问了由于最初提供了错误的 Discord ID 而导致提交了两次注册表单的问题。
   - 另一位成员建议，*只需使用正确的 ID 提交即可，应该没问题。*
- **确认邮件延迟**：一位成员询问注册后没有收到确认邮件的问题。
   - 另一位成员回答说**这是正常的**，后续通知应该很快就会发出，可能在今天或明天。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1360601880841031710)** (67 条消息🔥🔥): 

> `Nomic Embeddings, GPT4All Max Tokens, HuggingFace 故事模型, Chat Templates, Context Length` 


- **Nomic Embeddings 实现网站自动链接**：一位成员成功使用 **Nomic embeddings** 自动链接网页，通过 [semantical-website-links 博客文章](https://huggingface.co/blog/JLouisBiz/semantical-website-links) 将手动工作量减少到百分之几。
   - 该成员正在寻求自动识别并链接文本中与 embeddings 高度对应的关键词的方法，从而创建一个随着知识库演进动态更新的文档互联网络，详见 [此 YouTube 视频](https://www.youtube.com/watch?v=xk2VGnLYAkA)。
- **GPT4All 和 Max Token 故障排除**：一位成员尝试使用 **GPT4All** 中的各种模型生成一段至少 30 分钟长的剧本，但遇到了**响应长度限制**。
   - 成员们建议增加 **Max Tokens** 设置并将故事拆分为多个部分，但该成员表示限制依然存在，并正在寻找能够输出更长响应的模型。
- **HuggingFace “story” 模型可能有所帮助**：该成员在 **HuggingFace** 上使用关键词 “story” 成功找到了能够生成更长响应的模型。
   - 另一位成员提醒说，他们发现其中许多模型是专有的，并非自由软件。
- **Chat Template 位置揭晓**：一位成员询问如何找到 Llama3.2, Llama3.1, Aya-23 和 KafkaLM-8x7b-German-V0.1 等各种模型的 **chat templates**。
   - 另一位成员引导他们查看模型作者的发布版本，通常在他们的网站、GitHub 或 **Hugging Face** 上，特别是检查 `tokenizer_config.json` 文件中的 `chat_template` 条目。
- **Context Length 影响响应质量**：一位成员指出，大多数模型是在 **2048 到 8192 tokens context length** 之间进行训练的，虽然 RoPE 和 Yarn 等技术可以扩展这一范围，但超出原始范围后响应质量会大幅下降。
   - 响应长度取决于训练数据集和 finetuning，但可以通过 prompting 进行微调，例如告诉模型让内容变得 *VERY VERY LONG*。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1360335021080055858)** (16 messages🔥): 

> `Mojo ownership vs Rust, Origins vs Lifetimes, VSCode extension issues, Mojmelo module, closures` 


- **Mojo 的 `owned` 参数解析**：在 Mojo 中，**`owned`** 关键字会将一个可复制的元素复制到函数中，并在其超出作用域时将其删除；**`mut`** 采用可变引用，但根据 [文档](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and)，转移操作符（transfer operator）用于完全移动该值。
   - 一位熟悉 Rust 的用户寻求 Mojo 所有权系统的指南，因为他们理解 Rust 中的可变借用（mutable borrows）、不可变引用（immutable references）和移动值（move values），但发现 Mojo 的 **Origins** 概念很新鲜。
- **`Origins` 更名为 `Lifetime`**：如线程中所述，Mojo 中的术语 **`Origin`** 已更名为 **`Lifetime`**，这可能有助于熟悉 Rust 生命周期概念的用户理解。
   - 文中澄清，引用 `ref [a]` 的存活时间与变量 `a` 一样长，或者反过来说，它使变量 `A` 的存活时间与引用一样长。
- **Mojo 扩展了 Lifetimes**：Mojo 的 Lifetimes 与 Rust 不同，因为 Mojo 会扩展值的生命周期以匹配持有它们的任何引用；相反，必须跟踪每个引用的 Origin 以确定值的扩展和释放，这与 Rust 基于作用域（scope-based）的生命周期跟踪形成对比。
   - 一位成员表示：*理解 Mojo 如何跟踪 closure origins 可能是理解 Mojo 生命周期模型的最佳途径*。
- **Mojmelo 扩展问题**：一位用户遇到了 Mojo VSCode 扩展的问题，报告称尽管通过 magic add 成功手动安装和设置，但仍显示缺少 **`mojmelo`** 模块的错误。
   - 有建议认为 VSCode 扩展可能使用了自己的 Mojo 安装路径，导致其无法检测到项目环境中安装的模块；一种解决方法是手动配置扩展，使其使用本地模块库进行 intellisense。
- **Mojo closures**：讨论指出 Mojo 的 Origin 跟踪与 closures 之间存在类比关系，暗示理解 Mojo 如何管理 closure origins 是掌握其内存管理模型的关键。
   - 目前关于这一概念的文档很少。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1360601085827481722)** (32 messages🔥): 

> `PythonObject Literal, MLIR in Mojo, Mojo Proposals, Negative Bounds` 


- **嵌套 ListLiterals 困扰 Mojo**：Mojo 目前还无法处理嵌套的 `ListLiteral`，会导致 `constraint failed: cannot convert list element to python object` 错误，但解决方法包括 [使用 `Python.list()` 并追加元素](https://github.com/modular/max/blob/f5adc052eb447297ac011a5e97063a62e55cd014/mojo/stdlib/src/python/python_object.mojo#L443-L445) 或 [使用嵌套调用 `PythonObject`](https://discord.com/channels/1017080050601173042/1165766816034523176)。
   - Chris Lattner 提到旧的更漂亮的语法目前已损坏，但他们会随着更多语言特性的加入而重新处理它。
- **MLIR 示例出现在 Mojo Discord**：一位成员询问关于在 Mojo 中利用 **MLIR** 的旧文档，另一位成员提供了 [一个示例链接](https://github.com/modular/max/blob/570d4a0af82d547264a2bc46f6f0abeba59f3d66/examples/BoolMLIR.ipynb)，并指出自那时起语法已经发生了变化。
   - 他们提到 *这里的人们可能仍然能够提供帮助*。
- **Mojo PEPs 即将到来 🔥**：受 **Python PEPs** 的启发，一位成员建议为 Mojo 建立类似的系统来跟踪变更，另一位成员指出了 [Mojo 现有的提案系统](https://github.com/modular/max/tree/main/mojo/proposals)。
   - 讨论显示了社区对以结构化方式管理和沟通语言演进的兴趣。
- **Negative Bounds 反转命名集**：**Negative bounds** 是一种反转命名集的方法，通常与 **marker traits** 一起使用，以定义一组类型的补集。
   - 例如，`!Send` 将代表一个线程局部（thread-local）变量或非原子引用计数的智能指针，表明它在线程间移动是不安全的。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1360648500034928782)** (5 条消息): 

> `Llama4 Deep Research, Equity Research Agent, GPT-4.1 API, Agent Benchmarks` 


- **Llama4 助力 Deep Research 项目**：由 Clelia Bertelli 开发的一个完全开源的深度研究解决方案现已发布，该方案基于 **Llama4**, @GroqInc, Linkup, @FastAPI, @Redisinc, @Gradio 和 @llama_index 构建，简单的流程概述见 [这里](https://t.co/o46BJWIxM7)。
- **从零开始构建 Equity Research Agent**：一个新的教程展示了如何构建端到端的 Agentic 工作流，用于摄取来自 **Tesla/Ford** 的非结构化财报并提取财务指标，详情见 [这里](https://t.co/2hdpLas3vH)。
- **GPT-4.1 API 落地并提供首日支持**：**OpenAI** 宣布在 **API** 中提供 **GPT-4.1**，通过 `pip install -U llama-index-llms-openai` 即可实现首日支持，更多详情见 [这里](https://t.co/JPEX3KAoWS)。
- **GPT-4.1 基准测试显示性能提升**：**GPT-4.1** 相比 **4o** 自身显示出约 **10%** 的实质性提升，而在我们已经非常出色的 Agentic 方法上也有约 **2%** 的提升。
   - 欲了解更多工作详情，请通过 [此链接](https://t.co/E7KcaQ48Ek) 联系。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1360563181180817439)** (31 条消息🔥): 

> `LlamaParse vs SimpleDirectoryReader, Files in Index vs External File Sources, Open Source LLMs for Agent Workflow, Django Application hangs when calling LlamaParser with Celery, Voice Agents Support` 


- ****LlamaParse** 在文档处理灵活性上实现飞跃**：**LlamaParse** 可以处理图像、表格和图表等视觉元素，与 **SimpleDirectoryReader** 等基础读取器相比，提供更高质量的解析。
   - 使用 LlamaParse 优于 SimpleDirectoryReader 的主要优势在于其输出解析文档的 *质量*。
- ****Index Files** 与 **External Data** 详解**：索引中的文件决定了创建向量索引时的文档数量，而 **External Data sources**（外部数据源）则涵盖了用于构建索引的 **Google Drive**、**Confluence** 和 **Notion** 等平台。
   - 换句话说，**files in index** 是你用来 *创建索引* 的文档，而 **external data sources** 帮助对存储在其他地方的数据创建索引。
- ****开源 LLM** 在 Agent 工作流中面临挑战**：虽然较小的开源 LLM 被认为不足以胜任 Agent 工作流，但推荐使用较大的模型，如 **Llama3**, **Llama 3.1**, **Llama 3.2:3b** 或 **Mistral**，通常与 **Ollama** 配合使用。
   - 一位成员表示他们 *目前正在使用 llama3.2:3b*，看起来效果不错。
- ****Celery** 在 Django 中调用 **LlamaParse** 时发生阻塞**：有用户报告称，其 **Django** 应用在通过 **Celery** 调用 **LlamaParse** 时会无限期挂起，尽管在不使用 **Celery** 的情况下功能正常。
   - 尽管存在问题，但在这种挂起状态下并未抛出明确的错误。
- ****Voice Agents** 探索前行**：通过在输入和输出阶段集成 text-to-speech 和 speech-to-text 模块，可以实现对 Voice Agents 的基础支持。
   - 在 Voice Agents 的背景下，也有人询问关于集成 Google 的 Live API 的问题，但未得到解答。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1360534860275650676)** (5 条消息): 

> `.query has no history, LlamaParse Layout Agent Mode, Benchmarking AI evaluation models` 


- **.query 聊天没有历史记录**：一位成员询问如何在不使用 Agents 的情况下为 **Query mode** 存储聊天记录，并获知 `Char .query` 是 **stateless**（无状态）的，因此没有历史记录。
- **LlamaParse Layout Agent Mode 指南**：一份关于使用 **LlamaParse Layout Agent Mode** 进行 **Visual Citations**（视觉引用）的综合指南分享在 [这里](https://medium.com/ai-artistry/visual-citations-with-llamaparse-layout-agent-mode-a-comprehensive-guide-a623a5fb41fc)。
- **AI 评估模型基准测试发布**：分享了一篇论文 [Benchmarking AI evaluation models](https://arxiv.org/abs/2503.21157v3)，该论文在 **6 个 RAG 应用** 中对 **LLM-as-a-judge**, **HHEM**, **Prometheus** 等模型进行了基准测试，指出评估模型在实践中表现得出奇地好。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1360438108365258933)** (8 messages🔥): 

> `NVIDIA Video Codec SDK, Direct Programming, Meeting #66 Topics, Index Validation PR` 


- **NVIDIA 发布 Video Codec SDK**：NVIDIA 发布了 [Video Codec SDK](https://developer.nvidia.com/downloads/designworks/video-codec-sdk/secure/13.0.19/video_codec_interface_13.0.19.zip) 以及配套的 [GitHub 示例代码](https://github.com/NVIDIA/video-sdk-samples)。
   - 一位用户警告不要在不理解内容的情况下使用 AI 提交 PR，并威胁要关闭此类 PR 并封禁惯犯。
- **TinyGrad 第 66 次会议议程公布**：第 66 次会议定于圣迭戈时间周一上午 7 点（香港时间晚上 10 点）举行，将涵盖多个议题。
   - 议题包括：公司更新、**chip!**、fast python、bert、mlperf、scheduler、driver、webgpu、retinanet、torch frontend multi gpu、cloud scale uuuvn 相关内容以及其他悬赏任务（bounties）。
- **Index Validation PR 更新**：一位无法参加会议的成员提到，他们看到了 Index Validation PR 上的评论，并理解了相关要求。
   - 他们预计明天可以准备就绪，另一位成员确认该项已加入会议议程进行讨论。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1360431752421707895)** (26 messages🔥): 

> `clang flags, tinygrad notes, debugging NaNs, small bounty` 


- **Clang 标志 `-fno-ident` 可静默调试输出**：一位成员注意到额外的段（`.comment` 和 `.note.GNU-stack`）被添加到镜像中，污染了 `DEBUG=7` 的输出，并建议使用 [-fno-ident clang 标志](https://xl0.github.io/tinygrad-notes/misc_1.html) 来防止这种情况。
- **初学者寻找第一个 tinygrad 项目**：一位新成员介绍了自己，并征求关于动手实践 **tinygrad** 的迷你项目建议。
   - 一位成员建议选择一个 [小额悬赏任务（small bounty）](https://xl0.github.io/tinygrad-notes/misc_1.html)，并链接了有用的资源：[tinygrad-notes](https://xl0.github.io/tinygrad-notes) 和 [mesozoic-egg 的 tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes)。
- **softmax 调试**：一位成员询问如何调试模型中的 **NaNs**，怀疑是 `softmax()` 的问题，并指出在 `__call__` 中途进行打印会导致优化器出现问题。
   - George Hotz 回应称打印不应该破坏程序，并建议提交一个 issue。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1361323175308431440)** (16 messages🔥): 

> `Custom TorchTune model in vLLM, HF model, Custom model architecture in vLLM, Torchtune generate script` 


- **TorchTune 模型寻求 vLLM 集成**：一位成员询问如何在 **vLLM** 中使用自定义的 **TorchTune 模型**。
   - 另一位成员建议，使用 **vLLM** 对 **TorchTune** 微调后的模型进行推理应该是很直接的，类似于处理来自 **HF** 的任何模型。
- **分享 TorchTune 示例！**：一位成员分享了一个 [教程链接](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-with-vllm) 以协助解决该请求。
   - 一位成员询问该教程是否适用于未在 **HF** 上定义的模型。
- **自定义模型：vLLM 的新领域**：一位成员确认他们定义了一个自定义网络，在 **TorchTune** 中进行了微调，将其转换为 **HF** 格式，现在想使用 **vLLM** 进行推理，但收到错误称“自定义模型”未在 **HF** 中定义。
   - 另一位成员澄清说，对于自定义网络，必须在 **vLLM** 中定义该模型，并指向了 [vLLM 文档](https://docs.vllm.ai/en/latest/contributing/model/index.html)。
- **TorchTune generate 脚本作为 vLLM 的替代方案**：一位成员建议使用 **Torchtune 的 generate 脚本**，虽然速度较慢，但可以配合自定义模型工作。
   - 他们推荐使用 **generate_v2**（[配方链接](https://github.com/pytorch/torchtune/tree/main/recipes/dev)）并请求反馈问题。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1360335469518389318)** (8 messages🔥): 

> `bitsandbytes installation errors, macOS installation issues, unit tests on macOS, FSDP import error, platform specific requirements` 


- **`bitsandbytes` 给 Mac 用户带来了麻烦**：`pip install -e '.[dev]` 在 Mac 上失败，原因是 `bitsandbytes>=0.43.0` 不提供除 Linux 以外其他平台的二进制文件，但降级到 `bitsandbytes>=0.42.0` 会有帮助。
   - 直到 **0.42** 的版本标签都存在错误，但至少这让它变得可以安装 ([bitsandbytes issue 1378](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180))。
- **pytest 在收集测试时失败**：运行 `pytest tests` 在收集测试时出现 59 个错误，原因是 `ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'`。
   - Traceback 显示在导入 `test_full_finetune_distributed.py` 时，由于 `torch.distributed.fsdp` 中缺少 **FSDPModule** 导致了问题。
- **成员建议以不同方式安装**：一位成员指出还有其他安装 `torchtune` 的方法，并且不希望出现平台特定的需求。
   - 他们认为这应该也能解决 Mac 上的单元测试问题，因为已经对 Mac 上的单元测试应用了一些修复。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1360332135097176266)** (2 messages): 

> `QLoRA, Quantization, Sub-4-Bit Quantization` 


- **QLoRA 量化查询**：一位成员询问关于使用 **低于 4 bits** 量化的 **QLoRA 风格训练** 的文献。
   - 该询问专门针对 **QLoRA** 背景下与 **sub-4-bit 量化** 技术相关的方法和发现。
- **寻求 Sub-4-Bit QLoRA 文献**：一位成员询问关于利用 **低于 4 bits** 量化进行 **QLoRA 风格训练** 的现有文献。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1361344317867950312)** (5 messages): 

> `Reward Function Design, Loss Function Variety, Inference Provider Flexibility, Resource Allocation, TRL Success Logging` 


- **奖励函数：是否进行 Shape？**：团队计划支持不同的 **奖励函数 (reward functions)**，但具体的实现细节仍在讨论中。
   - 一位成员询问关于以“奇怪的方式”定位奖励计算的问题，随后 *收集了一份重要的奖励函数列表*。
- **损失函数：实验站**：团队目前正在实验 **不同的损失函数**，但旨在通过可能采用类似于 DPO 损失的协议来避免 recipe 过度激增。
   - 目标是在支持重要损失和防止实验阶段过度泛化之间取得平衡。
- **SGLang 支持 DeepSeek 优化（如 expert parallel），而 vLLM 不支持**：有人请求通过推理服务器支持 **各种推理提供商**，理由是灵活性以及对特定模型或功能的支持（例如 SGLang 中有而 vLLM 中缺少的 DeepSeek 优化）。
   - 最初的计划是专注于 **vLLM** 以降低复杂性并围绕其进行优化。
- **资源分配：A100 到位！**：Recipe 的资源分配被承认具有 **硬编码的测试参数**，目前正在清理中；目前的测试主要在 **A100s** 上进行。
   - 团队澄清说，recipe 设计正处于密集开发中，初始重点是算法和基础设施的可扩展性，而非库的兼容性。
- **NonTensorStack：揭开神秘面纱**：**NonTensorStack** 澄清了何时传递给 `TensorDict` 的列表应被视为 batch 索引（例如 `a[0] = list[0]`），而何时应被视为跨 tensor 共享的常量。
   - 更多细节请参阅 [PyTorch 文档](https://pytorch.org/tensordict/main/overview.html#stacked-non-tensor-data)。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1360496419995127891)** (9 messages🔥): 

> `Coral Chat in Firefox, LLM Token Generation Issues` 


- **Coral Chat 成为 Firefox 侧边栏！**：成员现在可以通过将 `browser.ml.chat.provider` 设置为 [https://coral.cohere.com/](https://coral.cohere.com/)，在 Firefox 侧边栏中将 **Coral Chat** 作为聊天机器人使用。
   - 一位用户分享了一个 [Imgur 链接](https://imgur.com/a/6zcTV8z) 展示了该集成。
- **LLM 存在 next-token 问题**：一位用户分享了一个 [YouTube 视频](https://youtu.be/VRqSJfdwbF4) 并开玩笑说其他 **LLM** 在给定上下文中生成下一个 token 时可能会遇到类似问题。
   - 另一位用户回复了一个“眼睛”表情符号。


  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1361252377642143807)** (4 messages): 

> `Cohere Chat API, Java Demo Code, command-a-03-2025 model` 


- **分享 Cohere Chat API Java 示例**：一名成员分享了一个演示如何使用 Cohere Chat API 的 Java 示例，重点介绍了 `runInteractiveDemo()` 方法。
   - 该示例包含一个与 **command-a-03-2025** 模型交互的聊天循环，展示了如何发送消息并维护聊天历史。
- **交互式聊天演示实现**：`runInteractiveDemo()` 方法允许用户与 Cohere AI 进行对话，提供示例提示词并处理 API 错误。
   - 代码通过控制台获取用户输入，将其发送至 Cohere API，并打印响应，在每次交互中更新聊天历史。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1361213775965061130)** (1 messages): 

> `Diofanti.org, Aya model, Government spending transparency` 


- **Diofanti.org 助力希腊透明度**：一名成员介绍了 [Diofanti.org](https://chat.diofanti.org)，这是一个监控希腊政府支出和运营的**开放数据平台**。
   - 该平台将原始公共数据转化为可操作的见解，为公民、记者和政策制定者提供实现**透明度和问责制**的工具。
- **Aya 成为 Diofanti 聊天机器人的首选模型**：创建者一直在其基础上实验聊天机器人，而 **Aya** 是其首选模型。
   - 成员们受邀联系并支持该项目。


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1360886630633963640)** (3 messages): 

> `LUWA.app, AI for Science Community` 


- **LUWA App 将于 2025 年 4 月 25 日上线**：一名成员正在推出 **LUWA.app**，这是一个 **AI 驱动应用**的搜索目录，将于 **2025 年 4 月 25 日**上线。
   - 创建者热衷于了解 **Cohere** 及其 **LLM 模型**，以潜在地降低成本或提高应用性能。
- **Encode: AI for Science 社区寻求人才**：一位来自多伦多大学的成员正在建立一个名为 **Encode** ([https://encode.pillar.vc/](https://encode.pillar.vc/)) 的 **AI for Science** 社区。
   - 他们正在寻找具备优秀 **AI 技能**的人才，与著名的 **PIs**（首席研究员）一起解决重大科学问题；感兴趣的人士请私信联系。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1361402820376395787)** (1 messages): 

> `Lambda, HuggingFace, Groq, Mistral AI, Google AI Studio` 


- **Lambda Labs 提供 Serverless API 额度**：**Lambda** 为每位参与者提供价值 **$100** 的 [Inference](https://lambdalabs.com/inference) Serverless API 额度，申请链接见[此处](https://forms.gle/UtVhmPS3mitS8Vxu7)。
- **HF, Groq, Mistral 提供 API/算力额度**：我们的赞助商 **Lambda, HuggingFace, Groq** 和 **Mistral AI** 正向选定团队提供 API/算力额度，更多详情见[此处](https://rdi.berkeley.edu/agentx/#resources)，申请链接见[此处](https://forms.gle/ZDYxwM4aFSRCcrfp7)。
- **Google 授权访问 Gemini API**：**Google** 向所有参与者免费开放 **Gemini API** 和 **Google AI Studio** 的访问权限。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1361420471375237364)** (1 messages): 

> `Sean Welleck, LeanHammer, AI proof development, formal reasoning` 


- **Sean Welleck 演讲：桥接非正式与正式数学推理**：卡内基梅隆大学助理教授 Sean Welleck 将于今日 **PDT 时间下午 4 点**发表题为“桥接非正式与正式数学推理”的演讲。
   - 演讲将涵盖支持证明开发的 **AI 驱动工具**，从使用 **LeanHammer** 自动化低级步骤，到构思证明思路并整合非正式见解；观看直播请点击[此处](https://www.youtube.com/live/Gy5Nm17l9oo)。
- **Welleck 的背景：ML、语言、逻辑与奖项**：Sean Welleck 在卡内基梅隆大学领导 **Machine Learning, Language, and Logic (L3) 实验室**。
   - 他的研究重点是 Large Language Models、推理与 Agent，以及用于数学和代码的 AI，曾获得 **NeurIPS 2021 优秀论文奖**和两项 **NVIDIA AI 先锋研究奖**。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1361423950504005895)** (2 messages): 

> `Lecture Schedule, Email Notifications` 


- **尽管邮件延迟，讲座仍照常进行**：一位成员询问今天是否有讲座，因为他们没有收到通常的邮件通知。
   - 另一位成员确认有讲座，邮件发送稍有延迟。
- **邮件通知延迟**：一位成员报告未收到今天讲座的常规邮件通知。
   - 回复指出邮件已发送，但有轻微延迟。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1361304340773474375)** (2 messages): 

> `AI Agent Developer, DSPy Modules` 


- **AI Agent 开发者寻求新机会**：一位经验丰富的 **AI Agent Developer** 宣布他们可以承接新项目或全职机会，专注于构建由 GPT-4, LangChain, AutoGen, CrewAI 及其他前沿工具驱动的 **autonomous agents**。
- **提出 DSPy 模块评估指标**：一位成员询问是否对评估 **DSPy modules** 的新指标感兴趣，并引用了[这篇论文](https://arxiv.org/abs/2405.10516)。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1361395885019369525)** (1 messages): 

> `MCP, AWS, Model Context Protocol, Simba Khadder` 


- **工作坊：在 AWS 上构建生产级 MCP 服务器**：一场定于 **太平洋时间 4 月 17 日上午 8 点** 的工作坊将专注于在 **AWS** 上构建和部署生产级 **Model Context Protocol (MCP)** 服务器。
   - 参与者将学习设置、配置和部署 **MCP** 服务器，深入了解如何简化机器学习工作流；报名地址：[https://buff.ly/R7czfKK](https://buff.ly/R7czfKK)。
- **MCP 新兴标准改进 ML 上下文**：**MCP** 被强调为一种新兴标准，旨在改进跨项目和团队定义、共享及管理机器学习上下文的方式。
   - 该工作坊旨在提供关于 **MCP** 能力的实践见解，使数据工程师、数据科学家、机器学习工程师和 AI/ML 爱好者受益。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

basit5750: 我已经有了，私信我获取 Source Code
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1361393735400689664)** (2 messages): 

> `GPT-4.1, Free Usage, Discounted Rate, New Default Model, Limited-Time Opportunity` 


- ****GPT-4.1** 在 Windsurf 上线**：**GPT-4.1** 现已在 Windsurf 上可用，在 [Twitter/X](https://x.com/windsurf_ai/status/1911833698825286142)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lms3je7p2s2d) 和 [Threads](https://www.threads.net/@windsurf_ai/post/DIbwqQQslzI) 上以 <:windsurf:1306309317011570699> 表情符号标记。
   - 同时还配有[宣传视频](https://youtu.be/OBTpSQ8OVq4)和 TikTok 帖子（[最新视频](https://www.tiktok.com/@windsurf/video/7493220207041793310)）。
- **Windsurf 提供 **免费无限次 GPT-4.1****：Windsurf 将在所有计划中提供为期 **仅一周**（4 月 14 日至 21 日）的 **免费无限次 GPT-4.1** 使用。
   - 4 月 21 日之后，**GPT-4.1** 将以每次仅需 **0.25 积分** 的特别优惠费率提供。
- ****GPT-4.1** 设置为新默认模型**：新用户将默认使用 **GPT-4.1**，现有用户可以通过模型选择器轻松切换。
   - Windsurfers：“不要错过这个限时机会！”


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1360997096622129262)** (1 条消息): 

> `多轮复合列移除，数据集构成差异` 


- **多轮列被移除**：多轮复合列已从数据集中[移除](https://github.com/ShishirPatil/gorilla/pull/766)，但提供的上下文中并未明确说明移除的原因。
   - 尽管该列已被隐藏，但在 [BlogPost](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) 的 "Newly Introduced Categories" 部分仍有提及，并且在多轮任务的 1000 分总分中占 200 分。
- **数据集构成存在异常**：数据集构成存在差异，因为展示数据集结构的表格/图表中缺少了多轮复合列。
   - 目前尚不清楚该列的移除是暂时的，还是也应该从目前提及它的 [blog post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) 章节中删除。


  

---


{% else %}


> 完整的逐频道详情已针对电子邮件进行截断。 
> 
> 如果您想查看完整详情，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}