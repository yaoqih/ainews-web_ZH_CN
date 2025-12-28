---
companies:
- anthropic
- google
- openai
date: '2024-05-24T22:12:57.081028Z'
description: '以下是该段文本的中文翻译：


  **Gemini-in-Google-Slides** 被强调为总结演示文稿的实用工具。Kyle Corbitt 关于在生产环境中部署微调模型的演讲强调，除非必要，否则应避免进行微调，而应重点关注提示词、数据质量、合适的模型选择以及全面的评估。**Anthropic**
  展示了 **Claude AI** 中的特征干预（feature alteration），体现了对模型行为的控制，并加深了对大语言模型的理解。像 **GPT-4o**
  这样的开源模型（注：原文如此，GPT-4o 实际为闭源模型）在处理简单任务时，其在 MMLU 等基准测试中的表现正接近闭源模型，但在处理复杂的自动化任务时，高级模型仍然必不可少。'
id: 62649323-bc91-49a8-8210-d82d8b54f750
models:
- claude-3-opus
- claude-3
- gpt-4o
original_slug: ainews-ten-commandments-for-deploying-fine-tuned
people:
- kyle-corbitt
- bindureddy
- alexalbert__
title: 部署微调模型的十诫
topics:
- fine-tuning
- prompt-engineering
- model-evaluation
- feature-alteration
- benchmarking
- model-performance
- open-source-models
---

<!-- buttondown-editor-mode: plaintext -->**Gemini-in-Google-Slides 正是我们所需要的。**

> 2024年5月23日至5月24日的 AI 新闻。
我们为您查看了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**380** 个频道，**4467** 条消息）。
为您节省了预计阅读时间（以 200wpm 计算）：**495 分钟**。

> **后续跟进**：[Jason Wei](https://x.com/_jasonwei/status/1794093872651387004) 发布了一份针对昨天 Evals 主题的优秀 [“201” 补充资料](https://www.jasonwei.net/blog/evals)，内容涉及如何制作成功的 eval 的博弈策略，同时也包含了一些关于 MATH 和 LMSYS 等著名 eval 的题外话和轶事。此外，今天是使用 `AINEWS` 代码参加 [AI Engineer World's Fair](https://buttondown.email/ainews/archive/ainews-the-top-ai-engineer/) 的最后一天。

今天新闻较少，所以我们深入挖掘了社区中有趣的内容。今天的优胜者是 [Kyle Corbitt 关于在生产环境中部署微调模型（Deploying Finetuned Models in Prod）的演讲](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#)：

 
![image.png](https://assets.buttondown.email/images/548437ae-fc1c-4d6c-9333-e5de94f6b380.png?w=960&fit=max)
 

简而言之，这些“诫律”是：

<ol><li><b>你不应微调：</b>直接使用 prompting！以及可选的 few-shot 示例/RAG。微调昂贵、缓慢且复杂。仅在你的用例确实需要时才进行。</li><li><b>你应该写一个像样的 Prompt：</b>创建一个基准并证明该任务可以通过 prompting 实现。</li><li><b>你应该仔细检查你的数据：</b>如果必须微调，确保你彻底了解你的数据。</li><li><b>你应该使用你真实的业务数据：</b>你的模型质量取决于训练它的数据。确保你的训练数据尽可能接近模型在生产环境中将遇到的数据。</li><li><b>你应该保留测试集：</b>始终保留一部分数据用于测试，以评估模型的性能。</li><li><b>你应该选择合适的模型：</b>模型的参数越多，训练成本越高且速度越慢。选择一个适合你的任务和预算的模型。</li><li><b>你应该编写快速评估（Fast Evals）：</b>编写计算速度快的评估指标，以便快速迭代模型。</li><li><b>此外，你应该编写慢速评估（Slow Evals）：</b>编写更全面、计算时间更长的评估指标，以深入了解模型的性能。</li><li><b>你不应部署后就不管了：</b>不要只是部署模型然后就置之不理。监控其性能，并准备好根据需要重新训练或更新。</li><li><b>你不应太死板地对待这些诫律：</b>这些诫律旨在作为有用的指导方针，而非硬性规定。请根据你的最佳判断并结合具体需求进行调整。</li></ol>

有趣的是，我们使用了 Gemini 来完成这份幻灯片的摘要。去试试吧。

 
![image.png](https://assets.buttondown.email/images/64aa2377-7ec8-47e1-9a29-0d7a0337547f.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Anthropic 的 Claude AI 与可解释性研究**

- **Claude AI 中的特征改变**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1793741051867615494) 演示了如何通过改变 AI 内部的“特征”来改变其行为，例如使其极度关注金门大桥。他们发布了一个限时的“Golden Gate Claude”来展示这一能力。
- **理解大语言模型的工作原理**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1793741055495471244) 表示，基于他们在 Claude 中发现和改变特征的能力，他们对开始理解大语言模型的真实工作原理有了更强的信心。
- **对 Claude 的知识和局限性保持诚实**：[@alexalbert__](https://twitter.com/alexalbert__/status/1793683229595341182) 表示，Anthropic 对 Claude 了解什么和不了解什么保持诚实，而不是刻意决定其推测棘手哲学问题的能力。

**开源 AI 模型与进展**

- **开源模型正在追赶闭源模型**：[@bindureddy](https://twitter.com/bindureddy/status/1793967098412388770) 强调，在 MMLU 基准测试中，像 GPT-4o 这样的开源模型在简单的消费者用例上的表现正接近 GPT-4 等闭源模型。然而，对于复杂的 AI Agent 和自动化任务，仍然需要更先进的模型。
- **新开源模型发布**：[@osanseviero](https://twitter.com/osanseviero/status/1793930015047880959) 分享了本周发布的几个新开源模型，包括多语言模型 (Aya 23)、长上下文模型 (Yi 1.5, M2-BERT-V2)、视觉模型 (Phi 3 small/medium, Falcon VLM) 以及其他模型 (Mistral 7B 0.3)。
- **Phi-3 small 以更少的参数超越 GPT-3.5T**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793758206956589346) 指出，微软的 Phi-3-small 模型仅有 7B 参数，但在语言、推理、代码和数学基准测试中均超越了 GPT-3.5T，展示了在压缩模型能力方面的快速进展。

**AI Agent、检索增强生成 (RAG) 和结构化输出**

- **从用于问答的 RAG 转向报告生成**：[@jxnlco](https://twitter.com/jxnlco/status/1793800023689338921) 预测，在未来 6-8 个月内，RAG 系统将从问答转向报告生成，利用设计良好的模板和 SOP，通过针对有付费能力的人群来释放商业价值。
- **ServiceNow 使用 RAG 减少幻觉**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793794286456127825) 分享了 ServiceNow 的一篇论文，展示了 RAG 如何通过检索相关步骤和表名并将其包含在 LLM prompt 中，从而确保生成的 JSON 对象在工作流自动化中是合理且可执行的。
- **RAG 通过将 LLM 与现实世界数据连接来增加商业价值**：[@cohere](https://twitter.com/cohere/status/1793952689102958966) 概述了 RAG 系统如何通过将 LLM 与现实世界数据连接来解决幻觉和成本上升等挑战，并强调了企业在其 LLM 解决方案中采用 RAG 的 5 大原因。

**AI 基准测试、评估和文化包容性**

- **标准 AI 基准测试可能无法引导真正的全球文化理解**：[@giffmana](https://twitter.com/giffmana/status/1793932786199314691) 建议，像 ImageNet 和 COCO 这样典型的“西方” AI 基准测试可能无法反映真正的“多文化理解”。在全域数据而非仅在英语数据上训练模型，可以显著提高非西方文化背景下的性能。
- **评估大语言模型的困难**：[@clefourrier](https://twitter.com/clefourrier/status/1793913394871062970) 和 [@omarsar0](https://twitter.com/omarsar0/status/1793846120600474017) 分享了一份报告，讨论了稳健评估 LLM 的挑战，例如初始基准测试设计与实际使用之间的差异，以及随着模型能力增强，需要更具辨别力的基准测试。
- **Aya 23 多语言模型扩大了技术服务的范围**：[@sarahookr](https://twitter.com/sarahookr/status/1793670981963362327) 介绍了 Cohere 的 Aya 23 模型，这是一个强大的多语言系列，旨在为全球近一半的人口提供服务，这也是他们改变“谁被技术看见”这一使命的一部分。

**迷因与幽默**

- **Nvidia 股票与“永久底层阶级”**：[@nearcyan](https://twitter.com/nearcyan/status/1793700146024440127) 开玩笑说，配偶后悔没买 Nvidia 股票，从而将“永远属于永久底层阶级”。
- **对 Anthropic 金门大桥 AI 的讽刺**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1793850501022564619) 讽刺了 Anthropic 的可解释性演示，幽默地声称“OpenAI 已经赶上了 Claude 的最新功能，并且还拥有一个基于复杂机械可解释性研究的高级金门大桥模式。”
- **调侃 Google 的 AI 错误**：[@mark_riedl](https://twitter.com/mark_riedl/status/1794034618318123224) 分享了一个幽默的轶事，他开玩笑地声称 Google 的 AI 错误地认为他获得了 DARPA 奖项，导致人们真的相信他没有获得该荣誉。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

AI 进展与能力

- **GPT-4 令人印象深刻的转录和位置识别能力**：在 /r/OpenAI 中，GPT-4o 展示了从图像中转录文本和识别地点的卓越能力，甚至在没有 EXIF 数据的情况下也能做到，正如[这段视频](https://youtu.be/04NUPxifGiQ?si=RqXLZlfCfinXqHp9)所示，并在此处进行了[进一步讨论](https://www.reddit.com/r/OpenAI/comments/1cyrbmz/how_is_gpt_4o_able_to_identify_locations_so_damn/)。
- **Yi-Large 正在赶超最先进模型**：/r/singularity 发布的一份[对比图](https://www.reddit.com/gallery/1cyoun2)显示，Yi-Large 的表现正逼近 GPT-4，并在多个基准测试中超越了 Claude 3 Opus 和 Gemini 1.5 pro。

AI 伦理与安全担忧

- **OpenAI 员工因伦理担忧离职**：在 /r/singularity 中有[报道](https://www.reddit.com/r/singularity/comments/1cyik9z/its_becoming_increasingly_clear_that_openai/)称，OpenAI 员工离职不仅是因为对“减速（decel）”的恐惧，还涉及与 News Corp 合作、游说反对开源以及针对前员工的激进手段等问题。
- **对 OpenAI 与 News Corp 合作的担忧**：/r/OpenAI 的一篇[帖子](https://www.reddit.com/r/OpenAI/comments/1cyylp8/please_protest_openai_partnering_with_propaganda/)批评了 OpenAI 与右翼宣传公司 News Corp 的合作，担心这可能导致 ChatGPT 使极端观点合法化。
- **加州 AI 法案要求安全保障但遭到批评**：/r/singularity [讨论](https://www.reddit.com/r/singularity/comments/1cynxnk/californias_newly_passed_ai_bill_requires_models/)了加州新通过的一项 AI 法案，该法案强制要求超过 10^26 flops 的模型必须具备武器制造预防机制、关停按钮并向政府报告。然而，这些要求被批评为在技术上不合理。
- **Yann LeCun 反击 AI 末日论**：在 /r/singularity 分享的一段[视频](https://v.redd.it/sapbctzym62d1)中，AI 先驱 Yann LeCun 认为 AI 最大的危险是审查、监控和权力集中，而不是经常被描绘的末日场景。

AI 可解释性与控制

- **Anthropic 的 "Golden Gate Claude" 映射 AI 特征**：Anthropic 的研究（在 /r/singularity 中有[详细介绍](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)）表明，他们的 "Golden Gate Claude" 可以映射和操纵 AI 的内部特征，这在理解和控制 AI 行为方面可能是重大进展。
- **Anthropic 展示了通过特征改变来塑造 AI 行为**：另一篇分享在 /r/singularity 的 Anthropic [论文](https://www.anthropic.com/research/mapping-mind-language-model)显示，由稀疏自编码器（sparse autoencoder）学习到的可解释特征可以代表复杂概念，并可以通过修改这些特征来控制 AI，例如诱发某种痴迷。

AI 商业化与准入

- **Meta 考虑推出 AI 助手付费版**：据 The Information [报道](https://www.theinformation.com/articles/meta-is-working-on-a-paid-version-of-its-ai-assistant)（见 /r/singularity 帖子），Meta 正在开发其 AI 助手的付费高级版本。
- **马克龙将 Mistral 定位为欧盟顶尖 AI 公司**：CNBC 的一篇[文章](https://www.cnbc.com/2024/05/23/macron-france-ai-us-china-tech-innovation.html)（分享于 /r/singularity）描述了法国总统马克龙如何将 Mistral 宣传为领先的欧盟 AI 公司，这引发了关于偏袒法国公司而非其他欧洲竞争对手的批评。
- **Google Colab 为 AI 开发提供免费 GPU**：/r/singularity 的一则[帖子](https://i.redd.it/hk7lt5hnq42d1.jpeg)强调，Google Colab 正在提供免费的 GPU 访问权限（包括 A100），以支持 AI 开发。

梗图与幽默

- **关于婴儿潮一代不愿放手的梗图**：/r/singularity 上的一个[梗图](https://v.redd.it/2w4babmsq62d1)调侃了婴儿潮一代拒绝让年轻一代接管的现象。
- **关于 Microsoft 训练 GPT-5 的讽刺视频**：/r/singularity 的一段[视频](https://v.redd.it/3r20macwa62d1)讽刺了 Microsoft 训练 GPT-5 的场景，将其比作鲸鱼吞噬磷虾一样喂入数据。
- **关于 Windows Recall AI 和隐私的梗图**：/r/singularity 上的一个[梗图](https://i.redd.it/goawjk1lu72d1.jpeg)嘲讽了假设的 Windows Recall AI 功能及其引发的隐私担忧。

---

# AI Discord Recap

> 摘要之摘要的摘要

1. **LLM 微调技术与最佳实践**：

    - **[微调十诫](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)**：在 Kyle Corbitt 的演讲中，成员们强调了细致的 Prompt 设计和模板配置，使用 `###` 分隔符和 "end of text" tokens 来实现高效的模型微调。

    - **[Hamel 的延迟优化博客](https://hamel.dev/notes/llm/inference/03_inference.html)**：关于减少过拟合以及有效使用检索增强生成 (RAG) 策略的讨论，重点介绍了来自 Axolotl 等平台持续进行的微调实验的实践指导。

2. **量化与性能优化创新**：

    - **Tim Dettmers 关于 LLM.int8() 的研究**：他的工作（如[这篇博客](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features)所示）展示了先进的量化方法如何在不降低性能的情况下保持 Transformer 的表现，并揭示了对涌现特性（emergent features）及其影响的见解。

    - **CUDA 的梯度范数 Bug 修复**：解决了梯度爆炸和 Batch Size 等问题，显著提高了训练稳定性，详见[此 PR](https://github.com/karpathy/llm.c/pull/456)。

    - **[Axolotl 中优化的内存架构](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)**：样本打包（Sample packing）效率的提升在分布式训练期间带来了 3-4% 的资源管理收益。

3. **开源框架与社区努力**：

    - **[Axolotl 的最新更新](https://github.com/OpenAccess-AI-Collective/axolotl/docs/dataset-formats)**：社区讨论了将可观测性（observability）集成到 LLM 应用中，并解决缓存和配置问题，以简化模型微调的工作流程。

    - **[PostgresML 与 LlamaIndex 的集成](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939)**：Andy Singal 强调了 PostgresML 与 LlamaIndex 之间的协同作用，能够高效地利用 AI 进行数据库管理任务。

4. **多模态 AI 与新模型进展**：

    - **[Phi-3 模型引发关注](https://unsloth.ai/blog/phi3)**：Unsloth 的 Phi-3 模型因其更长的上下文长度和 Medium 版本支持而受到社区关注，并发布了关于快速优化和集成的公告。

    - **[Mobius 模型期待](https://x.com/DataPlusEngine/status/1793803117642854732)**：DataPlusEngine 即将发布的版本承诺提供高效的基础模型创建，引发了关于基础扩散模型（diffusion models）及其训练方法的讨论。

5. **AI 伦理、治理与用户体验的挑战**：

    - **SB-1047 监管担忧**：社区对 AI 治理中心化表示愤怒，并将其与其他行业的监管俘获（regulatory captures）进行对比，引发了关于该法案对小型开发者影响的热烈讨论。

    - **通讯工具中 AI 的伦理使用**：在工作场所通讯监控中部署 GPT-4 和 Claude 引发了关于将伦理嵌入 AI 及其减少法律漏洞潜力的哲学思考，正如有关 API 集成和使用限制的讨论中所强调的那样。


---

{% if medium == 'web' %}




# 第一部分：Discord 高层级摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**微调事实（Fine-Tuning Facts）**：在 [general 频道](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) 的讨论中，揭示了由于偏置的数据类别导致的 **语义相似度过拟合（semantic similarity overfitting）** 问题。一位用户在理解微调与用户输入及初始模型训练的关系时遇到了困难。此外，还注意到 **OpenAI 平台侧边栏** 的变化，其中两个图标（threads 和 messages）消失了。

**模板成为焦点（Templates Take the Spotlight）**：在 [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123) 中，强调了在微调期间正确配置模板的重要性。特别是分隔符 `###` 有助于解析不同的输入部分，而 "end of text" token 则指示何时停止 token 生成。

**Maven 与主持人的互动（Maven Mingles with Moderation）**：在 [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698) 频道，成员们进行了一次轻松的交流，提到了重聚。一份会议演讲录像的请求得到了响应，视频已在 Maven 上提供。

**Modal 动员（Modal Mobilization）**：[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) 频道的 Modal 用户分享了收到额度（credits）和训练经验的兴奋之情，并为新用户提供了 **Modal 文档** 和 **示例** 的具体链接。还分享了使用 Modal 参加 **Kaggle 竞赛** 的计划，包括设置和执行细节。

**Jarvis 记录 Jupyter 杂记（Jarvis Jots Down Jupyter Jumble）**：在 [jarvis-labs 频道](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229) 中，成员们讨论了在 Jarvis 上存储 VSCode 仓库，并建议使用 GitHub 保存工作。有一条关于由于不稳定而移除 **spot instance** 的通知。分享了微调 **open-lama-3b** 模型的成本和时长，一位用户通过调整模型参数解决了 Ampere 系列显卡的错误。

**Hugging Face 讨论额度与西班牙语模型（Hugging Face Huddles on Credits & Spanish Models）**：[hugging-face 频道](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) 讨论了待处理的 **HF credits** 以及适用于西班牙语文本生成的模型——推荐了 **Mistral 7B** 和 **Llama 3** 模型。

**额度倒计时继续（Credit Countdown Carries On）**：在 [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150) 频道，预告了即将发布的关于额度管理和分配的公告。

**Corbitt 的诫律引发关注（Corbitt's Commandments Claim Clout）**：[kylecorbitt_prompt_to_model 频道](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) 的热心参与者讨论了 Kyle Corbitt 演讲中介绍的微调方法和技术，包括 *[部署微调模型的十诫（Ten Commandments for Deploying Fine-Tuned Models）](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*。

**Axolotl 响应召唤（Axolotl Answers the Call）**：在 [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817) 中，用户讨论了数据集、模型训练以及在 Axolotl 中的故障排除。分享了一篇关于 **TinyLLama 微调** 的博客文章，并推动将可观测性（observability）集成到 LLM 应用中。

**退出 Zoom，进入 Discord（Zoom Out, Discord In）**：在 Zoom 聊天功能禁用后，来自 [workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) 的用户将讨论转移到了 Discord。

**Axolotl 的缓存难题引发困惑（Axolotl's Cache Conundrum Causes Confusion）**：在 [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664) 频道中，解决了令用户沮丧的 Axolotl 缓存问题以及文件丢失的困惑。关于样本打包（sample packing）的讨论和一份关于 tokenizer 陷阱的指南解决了有关效率和分词的疑虑。

**加速迈向胜利（Accelerate to Victory）**：[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) 频道的用户解决了对浮点数比较的困惑，修复了 Jarvislab 训练命令错误，并交流了学习模型加速的资源，重点关注微调的最佳实践。

**与 Axolotl 一起尝试（Winging It with Axolotl）**：[wing-axolotl 频道](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) 协作处理了数据集模板、预处理问题、Axolotl 配置，并提供了一个针对最新 Axolotl 更新的 PR 合并。他们深入研究了调试工具以及精确模板对训练成功的重要性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**蛋白质数据可视化达到新高度**：一个新的蛋白质可视化项目现在支持 3D 渲染，并包含人类血红蛋白和核糖体蛋白的示例，项目详情见 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md)。

**使用 OpenAI 的 Whisper 进入 TranscriptZone**：一款利用 OpenAI 的 Whisper 转录 YouTube 视频及更多内容的新转录应用已在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上线。

**去中心化网络——不仅仅是一个梦想？**：一个为去中心化互联网构建基础设施的项目通过调查征求社区反馈，引发了关于数据收集伦理的讨论。

**Vision Transformers 深度查询**：一名成员寻求关于应用 Vision Transformers (ViT) 进行单目深度估计（monocular depth estimation）的资源，表示有意使用 ViT 开发模型，但讨论中未提供具体资源。

**Mistral 模型的量化困境**：在 **Mistral v0.3 Instruct** 上使用 **bitsandbytes** 进行 8-bit 量化导致性能比 4-bit 和 fp16 更慢，这一令人困惑的结果与减少位数计算预期的效率提升相矛盾。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在 CSV 对决中超越 ChatGPT**：工程师们讨论认为 **Perplexity AI** 在 CSV 文件处理方面优于 **ChatGPT**，因为它支持直接上传 CSV。此外，推荐使用 **Julius AI** 进行数据分析，它利用 Python 并集成了 **Claude 3** 或 **GPT-4** 等 LLM。

- **用户冷落 Claude 3 Opus**：由于内容限制增加和感知到的实用性下降，**Claude 3 Opus** 遭到冷落，尽管 **GPT-4** 也有局限性，但仍被视为更好的选择。

- **质疑 Pro Search 的真实升级**：**Pro Search** 的升级引起了关注，用户讨论新的多步推理功能和 API 规范是真正的后端改进，还是仅仅是表面上的 UI 增强。

- **API 集成阐述**：围绕 **Claude** 外部工具 API 集成的对话引起了兴趣，同时分享了自定义函数调用、无服务器后端以及诸如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use) 的文档。

- **AI 伦理：不仅仅是思想实验**：关于为 GPT 注入伦理监控能力的讨论被触发，揭示了在职场沟通和法律辩护方面的潜在应用，尽管哲学上的难题尚待解决。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **关于 RTX 5090 显存（VRAM）的猜测达到顶峰**：关于传闻中拥有 **32GB VRAM 的 RTX 5090** 是否具有实际意义的辩论正热烈进行。参考了 [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/) 上的潜在规格和图片，但一些成员对其真实性持怀疑态度。

- **Stable Diffusion 与 AMD 的挑战**：用户提供了在 AMD 5700XT GPU 上安装 **Stable Diffusion** 的指导，建议从 [Craiyon](https://www.craiyon.com/) 等 Web 服务开始，以规避潜在的兼容性问题。

- **Stable Diffusion 3：承诺前的试用**：社区将 **Stable Diffusion 3** 与竞争对手 Midjourney 进行了对比，强调虽然 SD3 提供免费试用，但持续访问需要 **Stability** 会员资格。

- **对 Mobius 模型的期待升温**：关于 DataPlusEngine 的新型 **Mobius 模型** 的公告引起了极大关注，因为它声称可以创建高效的基础模型。该模型在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上进行了预告，它既不是简单的基础模型，也不是现有模型的微调版本。

- **32GB VRAM：游戏规则改变者还是性能过剩？**：提到 32GB VRAM GPU 引发了关于 Nvidia 数据中心 GPU 销售策略潜在转变的对话，考虑到拥有大量显存的产品可能会如何影响市场对 H100/A100 系列的需求。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT 配置问题已解决**：一个在 PEFT 训练期间缺失 `config.json` 的问题已通过从基础模型的配置中复制该文件得到解决，用户已确认成功。

- **Llama 3 修复 Bug 后的提升**：**Llama 3** 模型的基础权重被描述为“有 Bug”，但 Unsloth 已经实施了修复。为了改进训练，建议使用保留 Token（reserved tokens）并更新 Tokenizer 和 `lm_head`。

- **System Prompt 助力 Llama 3**：观察发现，加入 System Prompt（即使是空的）也能增强 Llama 3 的微调效果。

- **Phi 3 模型激增**：随着支持 Medium 规格的 **Phi 3 模型** 首次亮相，社区反响热烈。社区讨论引导工程师关注博客文章和发布说明中的详细信息。

- **Stable Diffusion 的诡异一面**：**Stable Diffusion** 产生的诡异伪影和离奇的语音克隆输出让用户感到惊讶，相关的讨论和经历已在 YouTube 视频和 Reddit 帖子中分享。

- **VSCode Copilot 建议**：用户在 **random** 频道寻求本地 VSCode "copilot" 的推荐，并得到了积极的建议和回应。

- **Phi-3 的推理延迟**：一名用户对使用 **Unsloth Phi-3** 时较慢的推理时间感到困惑，并提供了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 来调查延迟原因，社区目前仍在努力寻找修复方案。

- **量化困境的解决**：一名成员在量化自定义模型时面临挑战，在 **llama.cpp** 和 **Docker** 兼容性方面遇到了阻碍，引发了关于解决方案的讨论。

- **模型算力的 VRAM 判定**：列出了 VRAM 需求：**Phi 3 mini 需要 12GB** 即可，但 **Phi 3 medium 必须配备 16GB**。对于繁重任务，建议考虑外部计算资源。

- **训练一致性的数据尽职调查**：强调了在训练和评估中使用一致数据集的重要性，并重点介绍了 Unslothai 的公开数据集，如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a)。

- **平台可能性与警告**：针对 **Unsloth** 是否支持旧款 Mac 的查询得到了回复，确认重点在于 CUDA 和 GPU 的使用，并为仅有 CPU 的设备提供了建议。

- **企业专家支持扩展**：一位社区成员主动向 Unsloth 提供企业级专业知识，并对加入 Build Club 和 GitHub 的加速器表示赞赏，暗示了 Unsloth 未来发展的协同潜力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**关于 AI 理解能力的智力辩论**：社区就 LLM 是否真正**理解**概念进行了深入讨论，**可解释性研究（interpretability research）**被视为重要的实证依据。怀疑论者认为目前的努力尚显不足，并引用了 **Anthropic** 关于映射大语言模型思维的相关工作。

**Llama 模型的进阶探索**：一项旨在增强 **Llama 模型** 的技术尝试集中在编写一个能够管理**函数调用（function calls）**的脚本上，并以 **Hermes Pro 2** 的方法作为灵感。另一个咨询则围绕在 3080 GPU 上实现 **Llama 3 LoRA** 技术展开。

**数字维度的现实探索**：在关于 **Nous 和 WorldSim** 的对话中，成员们探讨了 **NightCafe** 和多维 AR 空间在映射复杂 AI 世界中的潜在应用。**音频可视化器（audio-visualizers）**中的梦幻探索和奇特的 **ASCII 艺术** 表现形式突显了 AI 驱动模拟的创意用途。

**筛选 RAG 数据**：提倡模型将**内部知识**与**检索增强生成（RAG）**相结合是一个热门话题，并提出了关于如何处理矛盾和解决冲突的问题。强调用户评估被认为是必不可少的，特别是对于复杂的查询案例。

**AI 微调中的精准度胜过虚幻魔力**：社区讨论称赞了 **Mobius 模型** 在**图像生成**方面的卓越能力，并期待其开源版本和说明性出版物。此外，还提到了 Hugging Face 的 `PyTorchModelHubMixin` 可以更轻松地共享模型，但受限于 **50GB 的大小限制**（在不分片的情况下）。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA：TPU 之争**：**JAX** 与 **PyTorch/XLA** 在 TPU 上的性能对比引发了关于基准测试细微差别的讨论，例如 **warmup times** 和 **blocking factors**。讨论强调了 GPT-3 训练成本从 **450 万美元大幅下降至 2024 年预计的 12.5 万至 100 万美元**，这一观点结合了多位贡献者提供的 **TFLOP rates** 和 **GPU-hour 定价**，并链接到一篇 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **扩展与教学 LLMs**：在研究论坛中，**Chameleon** 模型因其在多模态任务中的强劲表现而受到关注，而 **Bitune** 则有望提升 LLM 的 zero-shot 性能（[Bitune 论文](https://arxiv.org/pdf/2405.14862)）。讨论质疑了 **JEPA** 模型对 AGI 的可扩展性，并批评了 **RoPE** 的上下文长度限制，引用了相关 [论文](https://arxiv.org/pdf/2405.14591)。

- **涌现特征困扰 LLM 爱好者**：链接了 Tim Dettmers 关于在 Transformer 推理中保持性能的高级量化方法的研究，包括他提出的涌现离群值（emergent outliers）概念，以及通过 [bitsandbytes 库](https://huggingface.co/blog/hf-bitsandbytes-integration) 与 Hugging Face 的集成。关于涌现特征（emergent features）的讨论围绕着它们是模型的“DNA”这一观点展开，并探讨了其对相变（phase transitions）的影响。

- **技术调整与 LM 评估简报**：在 **lm-thunderdome** 频道中，工程师们分享了在 **vllm 模型** 中设置 seed 的实用技巧、使用 `lm_eval --tasks list` 获取 **任务列表**，以及处理 **BigBench** 任务名称更改导致 Accelerate 等框架出现内存问题的方法。建议通过查阅 `lm-eval/tasks` 文件夹来定位任务，以便更好地进行组织。

- **合作呼吁**：发出了扩大 **Open Empathic** 项目的呼吁，并提供了一个用于贡献电影场景的 [YouTube 指南](https://youtu.be/GZqYr8_Q7DE) 以及项目链接。鼓励进一步合作，强调了社区努力在增强项目方面的必要性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU 探索**：工程师们讨论了将小型模型加载到 GPU 上的挑战，一些人青睐 *llama3, mistral instruct* 和 *cmdrib* 等模型。同时，据报道，在某些应用中，使用较低的量化（如 *llamas q4*）比 q8 等高量化版本效果更好，反驳了“越大越好”的观念。

**下一代模型即将来临**：模型领域的更新通知了一个 **35B 模型** 的发布，并正在进行测试以确保 LM Studio 的兼容性。针对不同规模模型的优化也是一个话题，重点关注 **Phi-3 small GGUF** 及其效率。

**服务器与配置**：硬件讨论包括利用 **llama.cpp** 及其最近的 RPC 更新进行 **分布式推理**，尽管目前尚不支持量化模型。还探索了使用廉价 PC 集群（配备 **RTX 4060 Ti 16GB**）进行分布式模型设置的实验性构建，以及可能存在的网络限制。

**实现多语言凝聚力**：Cohere 模型现在将其能力扩展到了 **23 种语言**，正如广告所言，**aya-23 量化版本** 已开放下载，但 ROCm 用户必须等待更新才能体验。

**Stable Diffusion 被排除在外**：LM Studio 澄清其专门处理语言模型，不包括像 Stable Diffusion 这样的图像生成器，同时处理了旧款 GPU 上的 CUDA 问题，并推广了 **Julius AI** 等服务以缓解用户体验方面的困扰。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **梯度范数异常 (Gradient Norm Nuisance)**：将 Batch Size 从 32 修改会导致梯度范数突然飙升，从而中断训练。一个 [Pull Request](https://github.com/karpathy/llm.c/pull/456) 通过防止 Fused Classifier 中的索引溢出解决了这个问题。
  
- **Int4 和 Uint4 类型需要关注**：有成员指出 PyTorch 中许多函数缺乏对 **int4** 和 **uint4** 数据类型的实现，相关的 [讨论帖](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) 指出了在类型提升（Type Promotion）和 Tensor 操作方面的局限性。

- **直播代码预警 —— 聚焦 Scan 算法**：Izzat El Hajj 将主持一场关于 Scan 算法的现场编程环节，该算法对 Mamba 等 ML 算法至关重要。活动定于 `<t:1716663600:F>`，旨在为爱好者提供深度技术解析。

- **CUB 库查询与 CUDA 细节**：成员们讨论了从 CUDA CUB 库代码的运行机制到在不使用 cuBLAS 或 cuDNN 的情况下触发 Tensor Cores 等话题，并推荐了 [NVIDIA 的 CUTLASS GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main) 和 [NVIDIA PTX 手册](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 等资源。

- **FineWeb 数据集难题**：处理 FineWeb 数据集非常耗费存储空间，磁盘占用达到 70 GB，内存消耗高达 64 GB RAM，这暗示在数据处理任务中需要更好的优化或更强大的硬件配置。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python 库更倾向于 C 而非 Mojo**：关于将 Python 库移植到 Mojo 的可行性和准备工作展开了热烈讨论。考虑到 Mojo 正在不断演进的 API，人们担心会给维护者带来过大压力。成员们讨论了针对 C 库进行移植是否是更直接且务实的尝试。

**Rust 的安全性吸引力并未削弱 Mojo 的潜力**：虽然 Mojo 并不打算取代 C，但 Rust 的安全性优势正在影响工程师对 Mojo 在不同场景下应用的思考。目前的讨论涉及了可以借鉴 Rust 概念来优化 Mojo 开发。

**Nightly 版本 Mojo 进展神速**：在 MacOS 上使用 Nightly 版本的 Mojo 运行 BlazeSeq，其性能表现与 Rust 的 Needletail 相当，这引发了关于跨平台效率的讨论。正如 [Changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中所述，快速的 Nightly 更新让社区紧跟这门语言的发展。

**对 ModularBot 机制的好奇**：有人询问了 "ModularBot" 的底层技术，虽然没有提到具体模型，但机器人给出了有趣的回复。另外，讨论还涉及了在 Mojo 中进行 ML 模型训练和推理的潜力，提到了 Max Engine 作为 NumPy 的替代方案，不过目前还没有成熟的训练框架。

**编译时困惑与对齐问题**：从内存中布尔值的对齐到编译时函数的问题，这些都引起了用户的关注。相关的临时解决方案和官方 [Bug 报告](https://github.com/modularml/mojo/issues/2813) 凸显了社区驱动排错的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **忠于 LaTeX 的 LLM**：在格式化领域，用户对 GPT 即使在要求提供 Typst 代码时仍强烈倾向于默认使用 LaTeX 表示沮丧，这揭示了 LLM 似乎坚持某种特定的编码语法偏好。
  
- **Microsoft Copilot+ 与 Leonardo 之争**：社区讨论集中在 Microsoft Copilot+ PC 在“草图转图像”等创意任务中的价值，而一些成员则鼓励尝试具有类似功能的 [Leonardo.ai](https://leonardo.ai)。

- **对 AI 效率的渴求**：有人对 AI 造成的环境代价表示担忧，引用了 [Gizmodo 的一篇文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249)，该文指出 AI 模型训练过程中耗水量巨大，引发了关于需要更环保的 AI 实践的讨论。

- **迭代胜过创新**：社区就通过迭代优化来增强 LLM 性能进行了积极对话，并提到了像 AutoGPT 这样处理迭代的项目，尽管这伴随着更高的成本。

- **注入智能的提议是否言过其实？**：公会成员思考了在 ChatGPT 中嵌入法律知识的可行性和潜力，甚至考虑到了 6.5 亿美元的估值，不过关于这一大胆主张的具体观点还比较有限。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent 深度解析**：工程师们探讨了在 **SequentialChain** 中使用 **LangChain's CSV agent**，并讨论了[如何自定义输出键](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)（如 `csv_response`）。提到了 SQL agents 在处理多表查询时面临的挑战，指出存在 token 限制和 LLM 兼容性问题，并引导至 GitHub [提交 issue](https://github.com/langchain-ai/langchain/issues)。

**AI 展示引发关注**：[OranAITech 推特发文](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)展示了其最新的 AI 技术，同时 **everything-ai v2.0.0** 宣布了包括音频和视频处理能力在内的功能，并提供了 [repository](https://github.com/AstraBert/everything-ai) 和 [documentation](https://astrabert.github.io/everything-ai/)。

**揭秘 VisualAgents**：YouTube 上分享了 **Visual Agents 平台**的演示，展示了其利用 LangChain 的能力，在无需编码的情况下简化 SQL agent 创建和构建简单检索系统的潜力。两段特定视频展示了其工作流：[SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。

**EDA GPT 印象展示**：通过 [LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01) 链接了一个 **EDA GPT** 的演示，包括一段展示其各种功能的五分钟概览视频。该演示突出了该 AI 工具的多功能性。

**教程预告**：tutorials 频道的一条消息提供了 business24.ai 内容的 [YouTube 链接](https://youtu.be/gflsu_6R_8g)，但未透露其相关背景。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **盗版并非万灵药**：尽管有人幽默地建议 The Pirate Bay 可能成为分享 AI 模型权重的避风港，但成员们对此表示怀疑，并强调其他国家更友好的 AI 政策环境可能会取而代之。

- **日本在 AI 领域采取积极立场**：参与者注意到日本在 AI 发展方面的鼓励态度，并引用了通过[推文](https://x.com/DataPlusEngine/status/1793817514956259460)分享的一篇 **paper**，内容是关于在无需大量预训练的情况下创建新的基础 diffusion models，展示了一种涉及临时破坏模型关联的策略。

- **中毒恢复协议探讨**：提到了一项由 fal.ai 进行的涉及中毒模型恢复方法的**合作研究**，预计研究结果将从经验上证实该恢复方法。成员们对 AI 生成图像的美感表达了保留意见，特别是像 Mobius 这样的模型与 MJv6 等前辈相比所呈现的“高对比度外观”和伪影。

- **Claude 映射破解代码**：Anthropic 的 **research paper** 详细剖析了 Claude 3 Sonnet 的神经图谱，阐明了对概念激活的操作，可在其 [research page](https://www.anthropic.com/research/mapping-mind-language-model) 阅读。辩论围绕此类激活的潜在商业化展开，同时也伴随着对商业影响导致 AI 从业者感到沮丧的担忧。

- **怀旧 AI 视觉愿景**：一位成员回忆了从早期 AI 视觉模型（如 Inception v1）到如今复杂系统的演变，认可了 DeepDream 在理解神经功能方面的作用。此外，还讨论了神经网络中稀疏性的好处，描述了使用 L1 norm 实现稀疏性，以及在高维层中典型的 300 个非零维度。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **聚会提醒：名额有限**：即将于周二举行的 **LlamaIndex meetup** 仅剩少量名额，由于名额有限，鼓励爱好者们尽快[预订位置](https://twitter.com/llama_index/status/1793739449127583964)。

- **MultiOn 结合 LlamaIndex 实现任务自动化**：**LlamaIndex** 已与 AI Agent 平台 **MultiOn** 结合，通过代表用户操作的 Chrome 浏览器实现任务自动化；点击[此处](https://twitter.com/llama_index/status/1793764970024570979)查看演示。

- **RAGApp 发布，支持无代码 RAG 聊天机器人设置**：新推出的 **RAGApp** 简化了通过 Docker 容器部署 RAG 聊天机器人的过程，使其可以轻松部署在任何云基础设施上，并且它是开源的；在此处配置您的模型提供商：[here](https://twitter.com/llama_index/status/1794030544415818062)。

- **解决 PDF 解析难题**：社区认可 **LlamaParse** 是一个从 PDF（特别是表格和字段）中提取数据的可行 API，它利用 GPT-4o 模型来增强性能；**Knowledge Graph Indexing** 的挑战也是一个话题，强调了手动和自动化（通过 `VectorStoreIndex`）策略的需求。

- **PostgresML 与 LlamaIndex 联手**：**Andy Singal** 分享了将 **PostgresML** 与 **LlamaIndex** 集成的见解，并在 Medium 文章 ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) 中详细介绍了这一合作，收到了社区的积极评价。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct 发布**：OpenRouter 推出了 **Phi-3 Medium 128k Instruct**，这是一个强大的 140 亿参数模型，并邀请用户查看[标准版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)变体，并参与其有效性的讨论。

- **Wizard 模型获得魔力提升**：**Wizard 模型**表现出改进，响应更加迅速且富有想象力，但仍需注意避免重复段落。

- **关注 Phi-3 Vision 和 CogVLM2**：围绕 **Phi-3 Vision** 的热情高涨，分享了测试链接如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml)，并建议在 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 中使用 **CogVLM2** 处理以视觉为中心的任务。

- **Llama 3 Prompt 自动转换**：澄清了发往 **Llama 3** 模型的 Prompt 会通过 OpenRouter 的 API 自动转换，从而简化流程，但手动 Prompt 仍作为一种替代方案保留。

- **Gemini API 的烦恼**：用户报告了 **Gemini FLASH** API 的问题，如空输出和 Token 消耗，这被认为是模型本身的问题。Google 每日 API 使用限制的出现引起了人们对这可能如何影响 OpenRouter 的 Gemini 集成的关注。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Indexify 引起关注**：Tensorlake 推出的开源实时数据框架 [Indexify](https://github.com/tensorlakeai/indexify) 引发了讨论，重点关注其“流式 ETL”能力以及创建可持续开源模型的挑战。人们对所提供的提取器（extractors）是否充足及其潜在的变现路径表示担忧。

- **LLM 评估备受瞩目**：一篇关于大语言模型（LLM）评估实践、排行榜重要性以及细致的回归测试（non-regression testing）的 [Hugging Face 博客文章](https://huggingface.co/blog/clefourrier/llm-evaluation) 引起了成员们的注意，强调了此类评估在 AI 发展中的关键作用。

- **AI 对搜索引擎操纵的回应**：一起涉及网站中毒并影响 Google AI 汇总概览（AI-gathered overviews）的事件引发了关于安全和数据完整性的讨论，包括通过自定义搜索引擎浏览器绕过（browser bypasses）的解决方法，正如 [Mark Riedl 的推文](https://x.com/mark_riedl/status/1793375699967054334)所报道的那样。

- **AI 是在使开发民主化还是引发可靠性疑问？**：GitHub CEO Thomas Dohmke 关于 AI 在简化编程中作用的 [TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) 引发了对其可靠性的争论，尽管 AI 驱动的 UX 改进加快了编程过程中的问题解决。

- **弥补差距的多样性奖学金**：面临参加即将举行的 AI Engineer World's Fair 财务障碍的多元背景工程师得到了多样性奖学金的支持。有兴趣的申请人应在[申请表](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link)中对论文问题提供*简洁*的回答。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **无需信用卡的税务故事**：Nathan Lambert 破解了一场发票纠纷，意识到了由于转售证书（resale certificates）而在没有信用卡的情况下进行税务计费的合理性。

- **金门大桥 AI 引起关注**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 的实验诞生了“Golden Gate Claude”，这是一个一心一意针对金门大桥进行训练的 AI，因其在 claude.ai 上的公开互动性而引起轰动。

- **Google 的 AI 失误**：Google 未能利用反馈以及过早部署 AI 模型，引发了关于这家科技巨头公关挑战和产品开发困境的讨论。

- **反击数据集误解**：Google 的 AI 团队反驳了关于使用 LAION-5B 数据集的说法，提出他们使用的是更优越的内部数据集，正如[最近的一条推文](https://x.com/giffmana/status/1793906145310228538)所引用的那样。

- **Nathan 分享知识点**：对于 AI 爱好者，Nathan Lambert 上传了高级 [CS224N 课程讲义](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)。此外，与会者还得到了关于即将发布的会议录像的提示，但尚无发布日期详情。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA 在 CMDR 模型中获得关注**：讨论显示 **Grouped Query Attention (GQA)** 存在于 "cmdr+" 模型中，但不存在于基础 "cmdr" 模型中，这表明了它们规格上的重要区别。
- **智能注意力机制提升 VRAM 效率**：工程师们注意到，虽然 **GQA** 不提供线性缩放，但与指数缩放相比，它代表了一种改进的缩放方法，对 **VRAM** 使用产生了有利影响。
- **样本打包（Sample Packing）获得提升**：一个新的 **GitHub pull request** 展示了样本打包效率提高了 3-4%，有望为分布式环境提供更好的资源管理，链接见[此处](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)。
- **学术成就获得认可**：一位成员合著的期刊文章已在 **Journal of the American Medical Informatics Association** 上发表，强调了高质量、跨领域数据对医学语言模型的影响，文章可见[此处](https://doi.org/10.1093/jamia/ocae120)。
- **社区庆祝学术成功**：社区通过个人祝贺信息表达了对同行发表作品的支持，在 AI 领域营造了一种认可学术贡献的文化。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 引发技术动荡**：工程师们对 **SB-1047** 的影响表示深切担忧，称其对小型 AI 参与者不利，并将此情况比作在其他行业观察到的监管俘获（regulatory capture）。

**展示 Perplexity 和 Arc 等行业工具**：社区重点推介了辅助工作流的工具，分享了一个关于 [SB-1047 的 Perplexity AI 搜索](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A)以及 Arc Browser 的新功能 “Call Arc”，该功能简化了在线查找相关答案的过程，并附带了[信息链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。

**安装问题引发询问**：用户在使用 pip 安装 **Typer** 库时遇到问题，引发了关于是否遵循了设置步骤（如在 `poetry run` 之前执行 `poetry install`）或是否使用了虚拟环境的疑问。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny 作为虚拟 Co-Pilot 起飞**：开发者正在将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 集成，作为一个强大的本地 AI 代码补全工具，支持在不同端口上运行多个 llamafiles。

**Embedding 端点说明**：根据 [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681)，澄清了 `/v1/embeddings` 端点不支持 `image_data`；相反，对于图像应使用 `/embedding` 端点。

**Mac M2 在 continue.dev 中遇到对手**：一项性能观察指出，在使用 llamafile 执行时，continue.dev 在 Mac M2 上的运行速度比旧款 Nvidia GPU 慢。

**拥抱你自己的 LLMs**：对于那些希望构建和训练自定义 LLM 的用户，社区推荐使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练，并提醒 llamafile 是为推理而非训练设计的。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **服务器中回荡着感激之情**：一位用户对团队表达了衷心的感谢，展示了用户对团队支持或开发工作的认可。
- **对扩展模型的关注**：有传闻称是否会有 **104B 版本**模型加入家族树，但目前尚未有明确答案。
- **Langchain 链接缺失**：出现了关于 **Langchain** 与 Cohere 集成的问题，用户正在寻求关于其当前可用性和实现状态的指导。
- **模型尺寸之谜**：用户正在探究 Playground 中的 **Aya 模型**是指 8B 还是 35B 版本，这表明了解模型规模对应用的重要性。
- **错误排查角落**：诸如 **ContextualCompressionRetriever** 的 `ValidationError` 和 **403 Forbidden 错误**等问题，标志着工程师们正在进行活跃的调试和技术问题解决，提醒人们 AI 开发中的常见挑战。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 喜剧之夜产生共鸣**：用户分享的一段 AI 生成的单口喜剧作品获得了积极的惊喜，表明 AI 在模仿幽默和进行娱乐表演方面的能力有所提升。

**关于 AI 应用的探索性查询**：用户询问 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 的功能是否超出了生成喜剧的范畴，表现出对其功能范围的好奇。

**展示声音变换**：一位用户通过分享一段原始音轨的修改版、恶魔版，展示了 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 灵活的音频修改功能。

**对音频工程知识的渴望**：用户表达了对学习制作演示中音频修改技巧的兴趣，这对于对声音处理感兴趣的 AI 工程师来说是一项宝贵的技能。

**偏好简洁的沟通**：对一个问题的单字回复“No”突显了对简洁回答的偏好，或许反映了工程师对直接、务实沟通的渴望。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻找统一的事件追踪器**：一位成员强调了对兼容 Google Calendar 的事件日历的迫切需求，以确保不会错过任何社区活动。社区内对缺乏此类系统表示关注。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **新数据集发布公告**：用户 datarevised 引用了一个新数据集，并提供了详细信息的链接：[DataPlusEngine 推文](https://x.com/DataPlusEngine/status/1793803117642854732)。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---

# 第二部分：分频道详细摘要与链接



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408)** (74 条消息🔥🔥): 

- **语义相似度过拟合担忧**：一位成员思考，如果数据中过度代表了某些响应类别（尽管没有特定的单个响应被过度代表），是否会导致偏差。他们提到了之前在研究心理学（Research Psychology）中检查此类问题的经验。
- **微调模型困惑**：一位用户难以理解与 Pre-training 相比，Fine-tuning 在多大程度上将特定的用户输入整合到模型中。他们寻求关于 Pre-training、Curriculum Training 和 Fine-tuning 之间区别的澄清。
- **OpenAI 平台侧边栏变化**：一些参与者讨论了 OpenAI 平台侧边栏的变化，提到 **两个图标消失了**（一个是线程图标，另一个是消息图标）。
- **Rasa 与对话复杂性**：一位参与者分享了对 Rasa 对话式 AI 方法的见解，强调了由于对话复杂，创建意图分类器（Intent Classifiers）非常困难。他们提到将意图视为实体（Entities）可能会降低复杂性。
- **Kyle Corbitt 的会议演讲录像已发布**：Kyle Corbitt 的会议演讲录像现在可以在 Maven 门户网站上观看，讨论中分享了具体链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llm-tracker.info/research/Quantization-Overview">量化概述 (Quantization Overview)</a>：测试量化如何影响模型输出？- 对不同量化级别的 15 项基础测试。详细对比了 GPTQ, AWQ, EXL2, q4_K_M, q4_K_S 以及 load_in_4bit 的 perplexity、VRAM、速度等...</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel 的博客 - 优化延迟</a>：探索优化延迟的方法。</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/food-good-hungry-yum-gif-11656939384713462119">Food Good GIF - Food Good Hungry - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=d8JMJMvErSg&ab_channel=Rasa">Rasa 算法白板 - TED 实践</a>：在本视频中，我们将探索 TED 在实践中是如何工作的。我们将构建一个需要倒计时的数字助理，并观察超参数如何...</li><li><a href="https://www.youtube.com/watch?v=j90NvurJI4I&ab_channel=Rasa">Rasa 算法白板 - TED 策略</a>：在制作数字助理时，你需要的不仅仅是处理文本的算法。你还需要处理对话序列的算法...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7513)">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://us06web.zoom.us/rec/share/POky_IXJdWGOOGZ9BMORn2lZQI53F3d_sOMmESWRbvUm3Us8cWNB7v2rdqnF4raB.95CQod940HlUWGjB?startTime=1716504965000">视频会议、网络会议、网络研讨会、屏幕共享</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于移动端、桌面端和会议室系统的视频和音频会议、聊天和网络研讨会。</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123)** (23 messages🔥): 

- **LLM Finetuning 和 `###` 使用说明**：讨论了在 LLM 的序列生成 Fine-tuning 中使用 `###` 的情况，指出这有助于模型在推理（Inference）期间理解输入的不同部分。在 Fine-tuning 过程中需要正确配置模板，包括 ChatML 等其他结构。
  
- **模板要求说明**：强调推理时的输入需要与 Fine-tuning 时使用的模板匹配，不一定必须是 `###`，而是取决于所设置的内容（例如 Llama 2 chat template）。模型托管服务通常会管理这些模板和结构。

- **有无分隔符的模型行为**：分隔符可以帮助模型理解输入中的不同部分（如 Reddit 中视角的切换）；否则对于一般的风格适配来说是不必要的。结束分隔符或 Token 确保模型能正确解析并结束响应。

- **End of text token 的使用**：简要提到了 "end of text" token 的概念，作为指示模型停止生成 Token 的机制，这标志着 LLM 高效的输入和输出管理。

- **关于 LLM 使用案例的家庭作业**：成员们分享并讨论了将 LLM 应用于生成食谱和学习应用等任务的作业项目。项目重点强调了 Prompt engineering 和 Retrieval-augmented generation (RAG) 等技术。资源链接和作业详情见[此处](https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye">未找到标题</a>：未找到描述</li><li><a href="https://gpus.llm-utils.org/llama-2-prompt-template/.">Llama 2 Prompt Template</a>：提示 Llama 2 chat 模型的最佳 Prompt 模板实践是什么？
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698)** (8 messages🔥): 

- **Reka.ai 关于重聚的玩笑**：一位成员幽默地评论了在很长一段时间后见到另一位成员，开玩笑说：*"你太客气了！我开始以为在 fast.ai 之后我再也见不到天日了。"* 他们询问了彼此的近况以及目前正在构建的项目。
- **会议录像请求已完成**：一位成员请求获取在 IST 凌晨 4:30 举行的 "Conference Talk: From prompt to model" 的录像。该请求得到了肯定答复，录像现在已在 **Maven** 上提供。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702)** (18 条消息🔥): 

- **Modal 额度到账，反响热烈**：多位用户确认收到了来自 Modal 的额度，并表达了开始微调模型的迫切愿望。一位用户表示：“是时候动手搞点东西了（Time to hack something）。”
- **关于在 Modal 上使用纯 PyTorch 代码的疑问**：一位用户询问是否可以使用 Modal 运行纯 PyTorch 代码来微调 LLM，并将其与使用 Jarvis Labs 进行了对比。另一位用户确认这是可行的，并分享了他们在 Modal 上训练 SentenceTransformer 模型的经验。
- **Modal 中的数据集管理**：讨论涉及如何上传数据集并在 Modal 中使用它们，并提供了详细的代码示例和步骤。Steven Merrill 演示了如何设置 Parquet 文件、构建 Volumes 以及为函数添加 GPU 元数据注解。
- **Modal 文档与示例**：用户分享了有用的 Modal 文档和示例链接，包括 [Volumes 文档](https://modal.com/docs/guide/volumes) 和一个 [TensorFlow 教程](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py)，后者可以适配用于 PyTorch。
- **将 Modal 用于 Kaggle 竞赛**：一位用户计划利用 Modal 参加 Kaggle 竞赛，涉及数据下载、库安装、微调以及保存模型/日志。另一位用户提到在 Modal 上运行 Jupyter 服务器长达 24 小时，并分享了 [Jupyter inside Modal 示例](https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py">modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://modal.com/docs/guide/volumes">Volumes</a>：modal.Volume 是为高性能文件服务构建的可变卷。与 modal.NetworkFileSystem 类似，这些卷可以同时挂载到多个 Modal 函数，支持并发...</li><li><a href="https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py">modal-examples/11_notebooks/jupyter_inside_modal.py at 0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229)** (16 条消息🔥): 

- **在 Jarvis 上保存 VSCode 仓库**：一位成员询问如何在不暂停实例的情况下在 Jarvis 的 VSCode 实例上保存仓库以节省额度。另一位建议将代码发布到 GitHub 并在需要时重新克隆，同时指出**暂停的实例仅收取存储费用**，这部分费用非常低。
- **移除竞价实例 (Spot Instances)**：由于不稳定和利用率低的问题，平台暂时移除了竞价实例。
- **微调 open-lama-3b 的成本和耗时**：在 RTX6000Ada 上使用 **gpt4-LLM-cleaned 数据**微调 **open-lama-3b** 耗时 3 小时 44 分钟，成本约为 4 美元。随后的讨论提到，LoRA 权重体积较小，这可能解释了为什么上传到 Huggingface 看起来是瞬间完成的。
- **Axolotl 在 Ampere 系列上的错误**：一位用户在 A6000 上遇到了预处理错误，通过将 **bf16 改为 false** 并将 **fp16 改为 true** 解决了该问题。
- **课程注册额度问题**：一位用户反映在注册课程并加入 Jarvis 后未收到额度；管理员回复称正在处理新名单，一旦收到用户信息就会添加额度。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004)** (9 条消息🔥): 

- **HF 额度即将发放**：成员们询问获取 HF 额度的流程。**详情将很快通过电子邮件公布**，额度将发放给填写了周末发送的表单的参与者。
- **西班牙语文本生成的最佳模型**：一位成员征求专门用于西班牙语文本生成任务微调的模型建议。**Mistral 7B** 被推荐为一个流畅的选择，**Llama 3** 也被提及，尽管它不是官方的多语言模型，但效果依然扎实。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150)** (1 条消息): 

- **关于积分（Credits）的即将发布的公告**：有关积分管理和分发的公告即将发布。*"<@739531318571958272> 将负责运行这些积分，但我们很快会发布相关公告"*。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376)** (164 条消息🔥🔥): 

<ul>
    <li><strong>对讲座的高度期待</strong>：尽管存在时区挑战，成员们仍对讲座表示兴奋，并呼吁进行录制。*"我真的很想看这个，但去不了 😦 会有录像吗？"*</li>
    <li><strong>链接汇总</strong>：分享了多个链接，包括 Hamel 的 [LLM inference notes](https://hamel.dev/notes/llm/inference/03_inference.html)、[Argilla](https://argilla.io/) 和 [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)。讲座中收集了大量资源。</li>
    <li><strong>互动且幽默的环节</strong>：成员们很喜欢这种互动的氛围，其中包含关于 Fine-tuning 和睡眠时间的幽默交流。*"Fine-tuning 不仅在 GPU 计算方面昂贵，还影响了我们的睡眠时间！"*</li>
    <li><strong>讨论高效 Fine-Tuning 技术</strong>：讨论了各种 Fine-tuning 方法，如 DoRA、MoRA 和 LoRA，并链接了 [Answer.AI's efficient fine-tuning](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html) 等文章。还提到了对模型上下文扩展技术（如 RoPE）的探索。</li>
    <li><strong>Fine-Tuning 守则</strong>：讨论了部署 Fine-tuned 模型的“十诫”，并附带了 [slides](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67) 链接。成员们认为内容非常实用，对工作大有裨益。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://argilla.io/">专家改进 AI 模型的平台</a>：Argilla 是一个面向 AI 工程师和领域专家的协作平台，致力于追求质量、所有权和效率。</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel 的博客 - 优化延迟</a>：探索优化延迟的方法。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - mteb 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - 使用 FSDP QDoRA 高效 Fine-tuning Llama 3</a>：我们正在发布 FSDP QDoRA，这是一种可扩展且内存高效的方法，旨在缩小参数高效 Fine-tuning 与全量 Fine-tuning 之间的差距。</li><li><a href="https://x.com/corbtt">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-bert-2048">nomic-ai/nomic-bert-2048 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.argilla.io/en/v1.1.0/guides/steps/1_labelling.html">🏷 标注</a>：在标注时，我们通常区分手动标注和协作或程序化标注。在协作标注过程中，我们使用规则和推理预测等外部输入...</li><li><a href="https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67">在生产环境中部署 Fine-Tuned 模型的十诫</a>：在生产环境中部署 Fine-tuned 模型的十诫，Kyle Corbitt | @corbtt</li><li><a href="https://openpipe.ai/">OpenPipe：面向开发者的 Fine-Tuning</a>：将昂贵的 LLM Prompt 转换为快速、廉价的 Fine-tuned 模型。</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/recordings/88255">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817)** (117 条消息🔥🔥): 

- **分享 Jarvis 仓库链接**：分享了 [nisargvp 在 Hugging Face 上的 Jarvis 仓库](https://huggingface.co/nisargvp/hc-mistral-alpaca) 的链接，以及用于在 Axolotl 中设置模型的配置文件。
- **在 Modal 上运行模型的指南**：用户讨论了如何在 Modal 上顺利进行模型训练，指出了 [Modal Labs 的快速入门指南](https://github.com/modal-labs/llm-finetuning)，并提到在初始修复后操作非常顺畅。
- **TinyLLama 微调博客文章**：社区分享并赞赏了一篇记录使用 Axolotl 和 Jarvis 在 alpaca_2k_test 数据集上微调 TinyLLama 过程的博客文章，链接见 [此处](https://lucasvw.github.io/posts/19_llm_fine_tuning/)。
- **LLM 应用的可观测性**：讨论围绕在 LLM 应用中引入可观测性以收集用户反馈和 LLM 输入/输出对展开，强调了对更好追踪方法的需求。
- **Modal 训练错误支持**：用户在使用 Modal Labs 仓库进行 Mistral 模型训练时遇到并解决了问题，社区成员提供了排错建议并分享了具体的错误详情以诊断配置问题。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>: 未找到描述</li><li><a href="https://lucasvw.github.io/posts/19_llm_fine_tuning/.">Lucas van Walstijn - LLM fine-tuning 101</a>: 未找到描述</li><li><a href="https://wandb.ai/venetispall/llama-3-8b-hermes-sandals-sample-10k/workspace?nw=nwuservenetispall">venetispall</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://huggingface.co/blog/peft_merging">🤗 PEFT welcomes new merging methods</a>: 未找到描述</li><li><a href="https://www.kaggle.com/competitions/lmsys-chatbot-arena">LMSYS - Chatbot Arena Human Preference Predictions | Kaggle</a>: 未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://kaiokendev.github.io/til">Things I’m Learning While Training SuperHOT</a>: 页面</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: 未找到描述</li><li><a href="https://huggingface.co/nisargvp/hc-mistral-alpaca">nisargvp/hc-mistral-alpaca · Hugging Face</a>: 未找到描述</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=bf4nff4j6bo">无标题</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1793815232177185061">来自 Daniel Han (@danielhanchen) 的推文</a>: @TheZachMueller @Prince_Canuma @UnslothAI 如果你没有使用未训练的 token，那应该没问题 :) 只是有时人们使用 llama-3 模板 + llama-3 基础模型，结果却不太理想...</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/">Lawrence Wu - Finetuning LLMs with Axolotl</a>: 未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main?tab=readme-ov-file#quickstart">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/data/modal_docs.jsonl">llm-finetuning/data/modal_docs.jsonl at main · modal-labs/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369)** (3 条消息): 

- **Zoom 聊天混乱导致转向 Discord**：在 Zoom 聊天被禁用后，成员们不确定在哪里继续对话。一位成员建议将讨论转移到特定的 Discord 频道，得到了其他人的认可。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664)** (32 条消息🔥): 

- **Axolotl 中的缓存问题令用户困扰**：一名成员指出，在 **Axolotl** 中重新运行实验时，意外的缓存使用了旧的数据样本，该问题记录在[此处](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd)。重命名数据集文件解决了这一问题，另一名用户建议显式运行预处理（pre-process）步骤。

- **对缺失文件的困惑**：用户在 **Jarvislabs** 和 Google Colab 上运行训练命令时遇到了 `simple.yml` 或 `qlora.yml` 文件缺失的问题，导致执行失败。一名成员分享说，他们的 qlora 运行在 2x4090s GPU 上耗时约 6 小时，证实了使用正确文件和配置的重要性。

- **关于 Sample Packing 的咨询**：一位成员询问 Axolotl 中的 Sample Packing 是否会连接多个数据集行以填充最大序列长度（max sequence length）。另一位成员确认了这一点，并解释说虽然它们被连接在一起，但 Attention 已被设置，使得各行之间不会产生注意力交互。

- **Google Colab 中 BFloat16 的 RuntimeError**：由于 T4 GPU 未实现 `BFloat16` 相关的 RuntimeError，导致用户从 Google Colab 切换到了 Jarvis-labs。建议他们检查 PyTorch 和 CUDA 版本，切换到示例配置后解决了该问题。

- **分享 Tokenizer 注意事项指南**：一位用户分享了 Hamel 关于 [Tokenizer 注意事项 (gotchas)](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html) 的笔记链接，解决了 Prompt 构建中的复杂细节以及由于 Tokenization 处理导致的训练与推理之间的行为差异。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html">Hamel’s Blog - Tokenization Gotchas</a>：使用 Tokenizer 和推理 LLM 时容易踩的坑</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd">axolotl/docs/dataset_preprocessing.qmd at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/tiny-llama/qlora.yml">axolotl/examples/tiny-llama/qlora.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://colab.research.google.com/drive/1jLQDiW47k1vPe_tet4-m6dLVZhnNRet9?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/#runtimeerror-_amp_foreach_non_finite_check_and_unscale_cuda-not-implemented-for-bfloat16">Lawrence Wu - Finetuning LLMs with Axolotl</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283)** (118 条消息🔥🔥): 

- **用户对 float16 和 float32 的困惑**：有人提问为什么在显示的表格中 float16 的数值看起来比 float32 高。为了澄清这一困惑，提供了指向该话题过去讨论的链接。
- **Jarvislab 配置问题已解决**：用户在运行 Jarvislab 训练命令时遇到了缺少配置文件的问题。另一位用户建议将命令更改为使用 `accelerate launch -m axolotl.cli.train hc.yml`，从而解决了该问题。

- **在不同 GPU 上优化 Axolotl 运行**：一名成员请求关于调整 `accelerate` 配置以在不同 GPU 上优化 `axolotl` 运行的建议。建议将配置映射回 `axolotl` 的 yaml 文件，避免直接进行 `accelerate` 配置设置。

- **学习模型 Accelerate 的资源**：用户讨论了如何开始使用 Accelerate 进行微调任务，建议坚持使用像 `axolotl` 这样更高级别的抽象，以兼顾简单性和学习深度。

- **超参数与推理精度**：询问了针对扩展训练与训练不足模型的最佳学习率，以及 T4 GPU 中的 BF16 精度问题。建议包括在 Zoom QA 中询问硬件兼容的解决方案，或将权重转换为支持的数据类型。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/yacineMTB/status/1783939078804701578">来自 kache (@yacineMTB) 的推文</a>: 三台配置顶级的 Mac Studio，每台 7.5k 加元，配备 192GB 统一内存，192 * 3 -> 576GB “显存”，充足的 CPU 足以驱动常规服务器任务。两台几乎可以...</li><li><a href="https://www.amazon.com/PNY-Generation-Express-DisplayPort-Support/dp/B0CJQH8519">未找到标题</a>: 未找到描述</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">扩展指南：对 Llama 2 进行指令微调</a>: 这篇博文是关于对 Meta AI 的 Llama 2 进行指令微调的扩展指南</li><li><a href="https://arxiv.org/abs/2311.03285">S-LoRA: Serving Thousands of Concurrent LoRA Adapters</a>: “预训练-然后-微调”范式在大型语言模型的部署中被广泛采用。低秩自适应（LoRA）作为一种参数高效的微调方法，经常被用于...</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">聊天模型模板</a>: 未找到描述</li><li><a href="https://x.com/DavidGFar/status/1793662035227770911">来自 David Golchinfar (@DavidGFar) 的推文</a>: 大家好！@FernandoNetoAi、我、@LucasAtkins7 和 @erhartford 在 Kraken 正式发布后又为大家准备了一个惊喜。我们很高兴推出由 @Hyper 赞助的 Kraken-LoRA...</li><li><a href="https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/">2023 年深度学习最佳 GPU —— 深度分析</a>: 在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合您的使用场景和预算的最佳 GPU。</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">快速入门</a>: 未找到描述</li><li><a href="https://github.com/SkunkworksAI/hydra-moe">GitHub - SkunkworksAI/hydra-moe</a>: 通过在 GitHub 上创建账户，为 SkunkworksAI/hydra-moe 的开发做出贡献。</li><li><a href="https://x.com/skunkworks_ai">来自 undefined 的推文</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412)** (192 条消息🔥🔥):

- **最新 axolotl 和 llama 3 演示的 PR 已合并**：Modal LLM 微调仓库现在包含了最新的 axolotl 更新和 llama 3 微调演示。
- **寻求数据集模板及预处理问题**：成员们询问了 `chatml.intel` 数据集模板，并在预处理过程中遇到问题，特别是由于数据集结构缺少数值 ID 导致的解码问题。参考：[Axolotl Docs](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd)。
- **关于 Axolotl 配置的澄清**：讨论表明，如果没有明确指定，`load_in_8bit` 和 `load_in_4bit` 等默认配置值将设为 False，并建议直接检查代码以进行确认。
- **无模板提示词构建的困惑**：一位成员发现关于 [无模板提示词构建](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html) 的文档令人困惑，而其他人则澄清了模板正确性的重要性。
- **Office Hours 问答亮点：调试与技术栈见解**：成员们表达了调试工具对于理解训练期间输入和样本的重要性，提倡进行严格的模板验证，并建议使用回调函数来记录模型预测，参考 [Axolotl Callbacks](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py">axolotl/src/axolotl/utils/callbacks/__init__.py at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问 axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/h2oai/h2o-llmstudio">GitHub - h2oai/h2o-llmstudio: H2O LLM Studio - 一个用于微调 LLM 的框架和无代码 GUI。文档：https://h2oai.github.io/h2o-llmstudio/</a>: H2O LLM Studio - 一个用于微调 LLM 的框架和无代码 GUI。文档：https://h2oai.github.io/h2o-llmstudio/ - h2oai/h2o-llmstudio</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">扩展指南：指令微调 Llama 2</a>: 这篇博文是关于对来自 Meta AI 的 Llama 2 进行指令微调的扩展指南</li><li><a href="https://huggingface.co/datasets/GAIR/lima?row=85">GAIR/lima · Hugging Face 上的数据集</a>: 未找到描述</li><li><a href="https://gist.github.com/strickvl/e1591b83e3b290fb176e780e7ce7d383">gist:e1591b83e3b290fb176e780e7ce7d383</a>: GitHub Gist: 立即分享代码、笔记和片段。</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?usp=sharing">Wing 的问题线程 - Office Hours</a>: Wing (Axolotl) 的 OH 问题。Ben Eyal 9:59 AM 我想知道关于无模板提示词构建的问题，我真的不明白它是如何工作的。配置只需要一个输出，并且...</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?pli=1">Wing 的问题线程 - Office Hours</a>: Wing (Axolotl) 的 OH 问题。Ben Eyal 9:59 AM 我想知道关于无模板提示词构建的问题，我真的不明白它是如何工作的。配置只需要一个输出，并且...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd">axolotl/docs/rlhf.qmd at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问 axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/grgalex/nvshare">GitHub - grgalex/nvshare: 无显存大小限制的实用 GPU 共享</a>: 无显存大小限制的实用 GPU 共享 - grgalex/nvshare</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/data/sft.py#L129C4-L152C6.">axolotl/src/axolotl/utils/data/sft.py at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问 axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">未找到标题</a>: 未找到描述</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0060623">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/h2oai/">H2O.ai</a>: 为更智能的应用提供快速可扩展的机器学习 - H2O.ai</li><li><a href="https://tenor.com/view/trust-no-one-crazy-chris-henry-thomas-just-beyond-dont-trust-anybody-gif-23566469">Trust No One Crazy Chris GIF - Trust No One Crazy Chris Henry Thomas - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1243310332407971850)** (1 条消息): 

- **使用 Proteinviz 可视化蛋白质**：查看 [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) 以创建自定义的蛋白质视觉效果。该工具由一位热心的社区成员制作。
  
- **快速的 SDXL 结果**：[SDXL flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) Space 能够快速交付令人印象深刻的结果。感谢创作者构建了这一高效工具。

- **受 Karpathy 启发的自定义 Tokenizer**：一位社区成员分享了他们的 [自定义 Tokenizer](https://github.com/apehex/tokun)，其灵感来自 Karpathy 的工作。这突显了社区内不断的创新。

- **Mistral-7B v0.3 演示**：通过 [Mistral-7B v0.3 chat](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat) 演示体验极速性能。这是活跃贡献者带来的又一前沿开发案例。

- **使用 Diffusers 创建透明图像**：使用 Diffusers 生成 [透明图像](https://github.com/rootonchair/diffuser_layerdiffuse)，这是一个由另一位社区成员推动的项目。该功能允许使用先进的扩散技术进行创意视觉输出。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=S-gy6NUOGSs)">Agentic AI Solutions / Adaptive AI Solutions - 第 1 集：CrewAI 与 Preston McCauley</a>：在第 1 集中，我们简要介绍了 #AdaptiveAI 和 #Agentic AI 方法。https://www.linkedin.com/in/preston-mccauley-immersive-ux/ 加入 Presto...</li><li><a href="https://youtu.be/jddSbTLw0gc)">什么是指令微调模型？</a>：什么是指令微调（Instruction Tuning）？什么是指令微调模型？什么是预训练模型？我该如何让我的 Large Language Model 遵循指令？这些...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1243283012477522071)** (490 条消息 🔥🔥🔥): 

- **AutoTrain 数据格式问题**：成员们讨论了如何在 **AutoTrain** 中格式化微调数据，并建议参考 [AutoTrain 文档](https://hf.co/docs/autotrain)。分享了 CSV 格式示例和输入数据类型的细微差别，提高了设置的清晰度。
- **高级 LLM 微调**：强调了 LLM 微调中 **DPO 和 RHLF** 方法的区别，建议在 **SFT 之后进行 RHLF**，以教授文本补全模型对话规范。还分享了特定数据集的链接和更精细的模型调整。
- **Pandora 模型引发关注**：分享了关于 Pandora 模型的细节，这是一款新的 **开源文本转视频** 模型，并附带了预览链接。关于其 **智能程度** 和潜在应用的讨论在成员中引起了巨大反响。
- **Mobius 模型争议**：即将推出的 **Mobius 扩散模型** 面临审查，评论涉及受控质量和构图训练。随后的讨论强调了它在显著降低开发新扩散模型的成本和复杂性方面的潜力。
- **学习与发展资源**：包括 @temeretam 在内的几位成员讨论了在 AI 领域进阶的学习和职业路径，而其他人则在寻求特定编码和数据处理问题的建议，并参考了 **GitHub** 和 **Hugging Face** 文档链接以获取技术支持。
<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/maitrix-org/Pandora">maitrix-org/Pandora · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/download#filter-files-to-download">从 Hub 下载文件</a>：未找到描述</li><li><a href="https://tenor.com/view/babuin-gif-27648024">Babuin GIF - Babuin - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://imgsys.org/rankings">imgsys.org | 由 fal.ai 提供的图像模型竞技场</a>：一个生成式 AI 竞技场，你可以在这里测试不同的提示词并挑选你最喜欢的结果。查看模型排名并亲自尝试！</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">聊天模型模板</a>：未找到描述</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始进行大规模预训练的情况下，创建全新的基础扩散模型。我们可以通过受控的方式，打破所有的质量...</li><li><a href="https://huggingface.co/datasets/nroggendorff/mayo">nroggendorff/mayo · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://youtu.be/zLvFc_24vSM?si=TeS_EkFu9BeyYDbz">Rabbit 忽悠了我，所以我深入挖掘</a>：LAM 是骗局吗？让我们一探究竟。支持调查新闻：► Patreon: https://patreon.com/coffeezilla 协助此次调查的人员...</li><li><a href="https://x.com/DataPlusEngine/status/1793803117642854732">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们向 @FAL 提供了即将推出的 Mobius 模型的早期访问权限，它在 http://imgsys.org 上线仅 3 小时。根据人类偏好，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">mistralai/Mixtral-8x7B-v0.1 在 main 分支</a>：未找到描述</li><li><a href="https://tenor.com/view/frank-castle-wait-please-stop-please-no-please-gif-21133188">Frank Castle Wait GIF - Frank Castle Wait Please Stop - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.instagram.com/p/C7U3hOPRJhR/">Noa Roggendorff 在 Instagram 上："epic #ai"</a>：2 个赞，1 条评论 - noaroggendorff 于 2024 年 5 月 23 日："epic #ai"。</li><li><a href="https://huggingface.co/docs/datasets/v2.19.0/en/process#rename>">处理</a>：未找到描述</li><li><a href="https://tenor.com/view/kurt-kurt-angle-100-yard-stare-what-are-you-serious-gif-4081464694509837388">Kurt Kurt Angle GIF - Kurt Kurt angle 100 yard stare - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://emoji.gg/category/6/blobs">适用于 Discord 和 Slack 的 Blobs 表情符号 - Discord Emoji</a>：查找可在 Discord 或 Slack 上使用的 Blobs 表情符号 - Emoji.gg，互联网上最大的免费自定义表情符号目录。</li><li><a href="https://hf.co/docs/autotrain">什么是 AutoTrain Advanced？</a>：未找到描述</li><li><a href="https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation">GitHub - hpcaitech/Open-Sora: Open-Sora: 为所有人实现高效视频制作的民主化</a>：Open-Sora: 为所有人实现高效视频制作的民主化 - hpcaitech/Open-Sora</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file">GitHub - PKU-YuanGroup/Open-Sora-Plan: 该项目旨在复现 Sora (OpenAI T2V 模型)，我们希望开源社区能为该项目做出贡献。</a>：该项目旨在复现 Sora (OpenAI T2V 模型)，我们希望开源社区能为该项目做出贡献。 - PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://slackmojis.com/categories/25-blob-cats-emojis">
Slack 上的 Blob Cats 表情符号
</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1243424993426014309)** (8 条消息🔥): 

- **用于 Embodied AI 的 Deep RL 引起关注**：一位成员分享了他们学习专门用于 Embodied AI 应用的 Deep Reinforcement Learning 的热情，并邀请大家详细更新进度。

- **为 AI 初学者推荐 Fast.ai 课程**：推荐了 Fast.ai 的第一部分和第二部分课程，这些课程涵盖了使用 HuggingFace 库的实用深度学习任务，并为深度学习初学者打下坚实基础。课程详情见[此处](https://course.fast.ai/)。

- **Coursera 上的 Generative AI with LLMs 课程**：为有兴趣获得 AI 基础知识的人推荐了 Coursera 上的 **Generative AI with Large Language Models** 课程。该课程设计为 3 周完成，详情见[此处](https://www.coursera.org/learn/generative-ai-with-llms)。

- **PixART 扩散模型研讨活动**：宣布了一项针对用于文本到图像合成的 PixART 扩散模型进行深入审查的研讨活动，定于太平洋时间周五上午 10:00 举行。更多信息和社区互动见[此处](https://lu.ma/arxivdive-2024-05-24)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>：一门为具有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的开发者设计的免费课程。</li><li><a href="https://www.coursera.org/learn/generative-ai-with-llms">Generative AI with Large Language Models</a>：在 Generative AI with Large Language Models (LLMs) 课程中，你将学习生成式 AI 的基本工作原理，以及如何部署它... 免费注册。</li><li><a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>：加入我们进行论文/书籍回顾。每周我们会选择一个主题进行深入探讨，并进行公开问答和讨论。…
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1243438483884871721)** (3 条消息): 

- **ChatGPT 在药物研发中的精彩应用**：分享了一项研究链接，讨论了 **ChatGPT 和其他 LLM 在下一代药物研发**中的潜在用途。这篇发表在《国际手术杂志》(International Journal of Surgery) 上的文章强调了来自印度和孟加拉国各机构的贡献 [阅读更多](https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx)。
  
- **PostgresML 和 LlamaIndex 引起关注**：最近的一篇 Medium 文章强调了 **PostgresML** 与 **LlamaIndex** 的集成。这种[集成](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939)有望释放 AI 进步的新潜力，文章中提供了详细的见解。

**提到的链接**：<a href="https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx">ChatGPT or LLM in next-generation drug discovery and... : International Journal of Surgery</a>：暂无摘要。

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1243293254309511218)** (22 messages🔥): 

- **蛋白质数据集迎来重大更新**：一位成员分享了其蛋白质可视化项目的更新，增加了**人类血红蛋白 (human hemoglobin)**、**小鼠 GTPase** 和**人类核糖体蛋白**的示例。他们还实现了对 **3D 渲染**的支持，并在 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md) 上创建了详细的示例表。

- **基于 OpenAI Whisper 的转录应用表现出色！**：一位成员介绍了他们为 YouTube 视频、音频文件和视频文件开发的转录应用，该应用利用了 **OpenAI Whisper**。可以在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上查看。

- **征集去中心化互联网基础设施的反馈**：一位成员请求对其构建去中心化且以 **Agent** 为中心的互联网基础设施项目提供反馈并参与调查：[调查链接](https://hai.ai/)。这引发了关于频道垃圾信息以及通过调查收集数据的伦理问题的辩论。

- **浏览器中 3D 模型可视化的挑战**：尽管在 Gradio 浏览器中进行蛋白质结构的 **3D 模型渲染**面临挑战，但目前仍在努力寻找解决方案。有用的资源包括 [Hugging Face](https://huggingface.co/blog/spaces_3dmoljs) 上的一篇博客文章。

- **SimpleTuner Bug 修复提升训练性能**：一位成员强调，修复 **SimpleTuner** 中的一些细微 Bug 显著增强了其训练性能。现在的训练效果比以往任何时候都好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tensorkelechi/vidtext">Vidtext - tensorkelechi 开发的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/blog/spaces_3dmoljs">在 Hugging Face Spaces 上可视化蛋白质</a>：暂无描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1243525495769792583)** (4 messages): 

- **宣布每月一次的 Computer Vision 聚会**：介绍了一个即将举行的每月 **Computer Vision Hangout**，旨在讨论 CV 相关领域的项目、想法和问题。更多详情和活动参与方式请见[此处](https://discord.gg/MkHyuG9C?event=1243129304863215656)。

- **寻求发票处理解决方案**：一位成员询问是否有开源神经网络或付费 API，用于从扫描的发票中提取结构化的逐行信息。他们要求输出格式为 JSON，并指定了 product_id、description、quantity、unit_price 和 total_price 等字段。

- **寻找深度学习学习伙伴**：一位用户表示有兴趣寻找一位同样热爱 AI 和数据科学的深度学习学习伙伴。他们强调了共同探索神经网络、复杂算法和创新项目的动力。

- **征集深度估计中的 ViT 资源**：另一位成员询问了关于利用 Vision Transformers (ViT) 进行单目深度估计的资源。他们表示有兴趣使用 ViT 构建自己的模型，并正在寻求指导。

**提到的链接**：<a href="https://discord.gg/MkHyuG9C?event=1243129304863215656">加入 Hugging Face Discord 服务器！</a>：我们正致力于实现优秀机器学习的民主化 🤗 验证以链接您的 Hub 和 Discord 账号！| 79727 名成员

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1243507818158489661)** (8 messages🔥): 

- **Mistral v0.3 Instruct 中的量化异常**：一位成员报告了在使用 **bitsandbytes** 8-bit、4-bit 和 fp16 量化级别对比 **Mistral v0.3 Instruct** 时出现的意外性能问题。他们发现，虽然 fp16 和 4-bit 耗时约 100 秒，但 8-bit 却耗时 500 秒，这与 8-bit 应比 4-bit 更快的预期不符。
- **从 Pipelines 切换到 Generate 无明显改善**：同一位用户指出，根据 8-bit 模型文本生成的文档，将 **pipelines** 切换为 **generate()** 方法并没有像预期那样提高性能。
- **Bitsandbytes 版本与优化技巧**：针对性能问题，另一位成员询问了正在使用的 **bitsandbytes 版本**，并建议尝试设置 **int8_threshold=0** 以获得潜在的性能提升。原用户提到他们使用的 batch size 为 1，上下文范围在 500 到 2000 个 token 之间。
  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1243278218262351995)** (6 条消息): 

- **寻求 NLG 学习资源**：一名成员询问有关学习 **Natural Language Generation (NLG)** 的推荐资源。消息历史中未提供针对该查询的回复。

- **关于在自定义数据集上训练 Stable Diffusion 的咨询**：另一名成员询问有关训练 Stable Diffusion (SD) 以从自定义数据集（如 MNIST）生成图像的**官方文档**。他们提到在网站上找到了文档，但似乎侧重于无条件生成（unconditional generation）。

- **寻找深度学习学习伙伴**：另一位成员表示有兴趣寻找一位共同**学习深度学习**的伙伴。他们强调希望对方同样对 AI 和数据科学充满热情，并热衷于探索神经网络、复杂算法和创新项目。

- **寻求将 pth+index 文件转换为 Hugging Face 链接的帮助**：一名成员请求协助将 **pth+index 文件**转换为 Hugging Face 链接的 RVC 模型。这一技术咨询未立即获得可见的回复。

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1243286415463284879)** (493 条消息🔥🔥🔥): 

- **Perplexity 与 ChatGPT 在数据处理方面的对比**：讨论了 **Perplexity** 和 **ChatGPT** 在处理 CSV 文件方面的能力，提到 **Perplexity** 已经支持 CSV 上传。Julius AI 被作为数据分析的另一种选择而受到关注，它运行在 Python 上并利用 Claude 3 或 GPT-4 等 LLM。
  
- **对 Claude 3 Opus 的失望**：用户对 **Claude 3 Opus** 表示不满，原因是其限制增加且实用性降低，特别是在处理受版权保护的材料时。一些人建议使用 GPT-4o 等替代方案，但也承认 Claude 3 的效用已经减弱。

- **Pro Search 功能与增强**：用户注意到了 **Pro Search** 的新功能，增强功能包括多步推理和更新的 API 规范获取。然而，一些用户观察到此类更新可能是 A/B 测试的一部分，仅涉及 UI 更改而非后端改进。

- **工具集成与自定义函数调用**：讨论了 **Claude** 通过 API 进行外部工具集成的能力，并尝试通过自定义函数调用和无服务器后端解决方案来复制 ChatGPT 的数据分析工具。分享了相关文档的链接，如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use)。

- **AI 伦理与沟通分析项目**：讨论包括创建用于沟通分析和伦理行为监控的 GPTs，并建议此类工具可以帮助改善职场沟通并减少不当解雇诉讼。用户辩论了将伦理编码进算法的可行性和哲学含义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=0T1444HJbt0">Mistral 的原生函数调用新 7B 模型</a>：Colab 代码 - https://drp.li/K98Z7🕵️ 对构建 LLM Agents 感兴趣？填写下表构建 LLM Agents 表单：https://drp.li/dIMes👨‍💻Github:http...</li><li><a href="https://v0.dev/">v0 by Vercel</a>：通过简单的文本提示生成 UI。复制、粘贴、发布。</li><li><a href="https://tenor.com/view/google-chrome-pacman-eating-gif-13756279">Google Chrome Pacman GIF - Google Chrome Pacman 进食 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/aladdin-disney-cartoons-jasmine-ic-an-show-you-the-world-gif-4545341">Aladdin Disney GIF - 阿拉丁迪士尼动画 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/AZURE/comments/1bzs8gr/have_you_purchased_openai_ptus_how_much_did_it/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/tool-use">工具使用 (函数调用) - Anthropic</a>：未找到描述</li><li><a href="https://aws.amazon.com/bedrock">使用基础模型构建生成式 AI 应用程序 - Amazon Bedrock - AWS</a>：未找到描述</li><li><a href="https://search.brave.com/search?q=%s&source=desktop">Brave Search</a>：私密搜索网络……</li><li><a href="https://aws.amazon.com/bedrock/pricing/">使用基础模型构建生成式 AI 应用程序 - Amazon Bedrock 定价 - AWS</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1243340241197338706)** (7 条消息): 

- **Peran Kepala Sekolah 分享**: 分享了一个指向 [Peran Kepala Sekolah](https://www.perplexity.ai/search/Peran-Kepala-Sekolah-ECYSEyQXTviCYDqqDgM8sw) 的简短链接，没有额外的上下文或讨论。
- **PB55 解释**: 提供了一个指向 [what is the PB55](https://www.perplexity.ai/search/what-is-the-PB55hhXYRDGAVd7JjWDhaA) 的链接以供进一步阅读。
- **探究 'makura' 的起源**: 一位用户分享了一个链接，用于探究日语单词“枕（まくら / makura）”（意为枕头）的词源，链接见[此处](https://www.perplexity.ai/search/oDyPhU47T26IM1W0f7GQIg)。
- **确保 Thread 可分享性**: 发出了一个带有附件的提醒，以确保 Thread 是可分享的，并附带了指向 [Discord thread](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 的链接。
- **讨论 Stuart Hall 的理论**: 分享了 [Stuart Hall 的编码/解码模型](https://www.perplexity.ai/search/Explain-Stuart-Halls-IV.my4LjS2mNXyxPyWVLOw#0)。
- **查询 Opus 50 限制**: 一位用户询问了关于 [Opus 50 限制](https://www.perplexity.ai/search/Opus-50-limit-c2EHUbzTQGCocG2d17MrLg) 的问题。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1243398440315260958)** (1 条消息): 

- **References 功能仍处于 Beta 停滞状态**: 一位用户质疑了 References 功能处于 Beta 阶段的状态，并对申请三次后未收到回复表示沮丧。他们询问是否有人知道该功能何时会在 API 中发布。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1243290070820192337)** (427 条消息🔥🔥🔥): 

- **RTX 5090 规格传闻引发争论**：讨论集中在 RTX 5090 可能配备 32GB VRAM 的新传闻上，这引发了对其可行性和实用性的质疑。一位成员分享了一个指向所谓图片的[链接](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/)，但其他人批评这些图片具有误导性。

- **Stable Diffusion 安装指导**：一位成员寻求在 AMD 5700XT GPU 上安装 Stable Diffusion 的建议。由于 AMD 硬件可能存在兼容性问题，建议最初尝试使用 [Craiyon](https://www.craiyon.com/) 等 Web 服务。

- **Stable Diffusion 3 的定价与访问**：用户讨论了 Stable Diffusion 3 与 Midjourney 的优劣，一些人指出 SD3 提供免费试用。然而，似乎需要 Stability 会员资格才能持续访问。

- **Mobius 模型的推出引起关注**：DataPlusEngine 在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上宣布了即将推出的 Mobius 模型，声称它是目前最好的基于 Stable Diffusion 的图像模型。该模型被描述为“既不是基础模型也不是 fine tune”，并因其高效创建新基础模型的能力而受到吹捧。

- **对 GPU 性能和成本的好奇**：新的 GPU 型号，特别是 5090，引发了关于显存和训练速度的讨论。成员们指出，像 32GB 这样更高的 VRAM 可能会削弱 H100/A100 等高端数据中心 GPU 的销量，暗示这可能会影响 Nvidia 的策略。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>: 我们让 @FAL 提前体验了即将推出的 Mobius 模型，它在 http://imgsys.org 上线仅 3 小时。根据人类偏好，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://tenor.com/view/never-finn-adventure-time-gif-10874543">Never Finn GIF - Never Finn Adventure Time - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/-CyupSdXfI0">WOW 欧文·威尔逊（Owen Wilson）说过的每一个 Wow，简直太 WOW 了</a>: 欧文·威尔逊是我最喜欢的演员之一，他的“wow”非常传奇 —— 这里收集了他在一个地方的所有“wow”</li><li><a href="https://www.youtube.com/watch?v=k1hbRvSnFZg">A Moebius-metró | 匈牙利语完整电影</a>: 1996 年阿根廷神秘/科幻/惊悚片 - 完整电影。在世界上最繁忙的地铁系统之一中，一列满载乘客的地铁列车无影无踪地消失了...</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/">据传 Geforce RTX 5090 将配备 32 GiB GDDR7 和三个 PCB [传闻]</a>: 文章图片：Geforce RTX 5090 将配备 32 GiB GDDR7 和三个 PCB [传闻] - Geforce RTX 5090</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-d">显卡新闻</a>: 您可以在这里找到关于显卡的最佳新闻
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1243280018524606544)** (275 条消息🔥🔥): 

- **PEFT 训练问题已解决**：一位用户遇到了在 PEFT 训练期间未创建 `config.json` 的问题，并被建议从基础模型的配置中复制。该用户确认此方法有效，并感谢社区的帮助。

- **注意到 Llama 3 的 Bug**：一些用户讨论说“Llama 3 的某些基础（非 instruct）权重存在‘bug’”，但 Unsloth 会自动修复这些问题。建议在训练期间使用保留的 tokens，并确保对 tokenizer 和 `lm_head` 进行训练。

- **System Prompt 提升 Llama3 表现**：用户提到添加 system prompt 可以提高 Llama3 的 finetuning 性能。一位用户确认，即使是空白的 system prompt 也能对结果产生积极影响。

- **宣布支持 Phi 3 模型**：官方宣布现在已支持 Phi 3 模型（包括 medium 版本）。社区表现出极大的兴奋，并分享了相关博客文章的链接以获取更多细节。

- **Stable Diffusion 的诡异印记**：用户分享了关于语音克隆和 Stable Diffusion 生成的诡异伪影的怪异经历。他们发布了相关 [YouTube 视频](https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc)和 [Reddit 讨论](https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_difussion/)的链接。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth 对 Phi-3 进行 Finetune</a>：通过 Unsloth 轻松对微软的新模型 Phi 3 medium、small 和 mini 进行 fine-tune，上下文长度可延长 6 倍！</li><li><a href="https://huggingface.co/CohereForAI/aya-23-8B">CohereForAI/aya-23-8B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1793758720541016567">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们已经解决了 Llama 3 的训练问题，现在的 finetuning 效果好多了！Unsloth 现在支持新的 Phi-3 模型、Mistral v3、Qwen 等！阅读我们的博客：http://unsloth.ai/blog/phi3</li><li><a href="https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc">can i get a chicken tendie combo please</a>：未找到描述</li><li><a href="https://github.com/babycommando/machinascript-for-robots">GitHub - babycommando/machinascript-for-robots</a>：在你的车库里使用 MachinaScript For Robots 构建由 LLM 驱动的机器人！</li><li><a href="https://github.com/ggerganov/llama.cpp/issues">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_diffusion/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>：我正在运行 Unsloth 来对 llama3-8b 的 Instruct 模型进行 LoRA fine tune。1：我将模型与 LoRA adapter 合并为 safetensors。2：在 Python 中直接使用合并后的模型运行推理...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1243306406195626034)** (1 条消息): 

- **Phi-3 和 Mistral v3 现已上线**：*Unsloth 现在支持 Phi-3、Mistral v3 以及许多其他新模型。* 查看[发布详情](https://github.com/unslothai/unsloth/releases/tag/May-2024)。

- **Llama 3 问题已解决**：*我们修复了所有 Llama 3 的问题，因此现在的微调效果更好了。* 欲了解更多深度信息，请参考此 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cwwgkz/is_llama_3_just_not_a_good_model_for_finetuning/)。

- **探索免费的 Colab notebooks**：访问我们的 [Phi-3 medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)、[Mistral v3 notebook](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing) 等。

- **新模型支持和 GitHub Accelerator**：在 [Hugging Face](https://huggingface.co/unsloth) 上查看我们最新添加的模型，并了解我们参与 [GitHub 2024 Accelerator](https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/) 的情况。

- **庆祝 AI 创新**：*我们很高兴能与另外 10 个项目一起加入 GitHub 的 2024 Accelerator，这彰显了 AI 创新的全球影响力和快速进步。*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth 微调 Phi-3</a>：通过 Unsloth 轻松微调 Microsoft 的新模型 Phi 3 medium、small 和 mini，并获得 6 倍长的上下文长度！</li><li><a href="https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)">2024 GitHub Accelerator：认识塑造开源 AI 的 11 个项目</a>：宣布第二批入选名单，为项目提供价值，并推动新的前沿。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1243440379672530985)** (4 条消息): 

- **寻求本地 VSCode Copilot 推荐**：一位用户问道：*“有人使用本地的 VSCode ‘copilot’ 吗？我想尝试一下。求推荐 :)”*。另一位用户回复道：*“试试 continue”*，随后最初的用户表示感谢：*“谢谢，我会试试的 :)”*。
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1243277920252727329)** (103 条消息🔥🔥): 

- **Sloth Phi-3 推理性能问题**：一位用户报告称，与原始模型相比，使用 **Unsloth Phi-3 模型**时的*推理时间更慢*。他们分享了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 来诊断该问题，但即使在建议修改后，问题仍然存在。
  
- **自定义模型量化问题**：一位成员在量化源自 Unsloth notebook 的自定义模型时遇到了问题。他们收到了与 **llama.cpp** 和 **Docker** 不支持的架构相关的错误。

- **不同模型的资源需求**：关于 VRAM 需求的查询表明，**12GB 足以运行 Phi 3 mini**，而 **Phi 3 medium 则需要 16GB**。此外还提到，对于具有更大上下文窗口的摘要等大型任务，可能需要**租赁计算资源**。

- **评估数据集标准**：讨论强调了在训练和评估中使用一致数据集的重要性。具体而言，推荐使用 Hugging Face 上 unslothai 的公开数据集，例如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a) 中列出的数据集，因为其质量很高。

- **兼容性与自定义模型支持**：几位用户询问了 **Unsloth** 与旧款 Mac 的兼容性以及在无 GPU 系统上的使用情况，确认了 Unsloth 是针对 CUDA 和 GPU 使用而优化的。针对仅限 CPU 的系统和自定义模型支持，提供了一些变通方法和技巧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - a lamhieu Collection</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X#scrollTo=0zM8gPJUGySh">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues">Issues · unslothai/unsloth</a>：微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - Issues · unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py#L179">unsloth/unsloth/models/_utils.py at main · unslothai/unsloth</a>：微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1243412700105674753)** (2 条消息): 

- **工程师向 Unsloth 提供企业经验**：成员 higginsconsultingptyltd_39617 祝贺其他人加入 Build Club 和 GitHub 的加速器，并提议利用其企业经验来协助 Unsloth。另一位成员给出了积极回应，表示渴望进一步讨论：“*当然，我们非常乐意！*”
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1243288751313256490)** (12 条消息🔥): 

- **通俗易懂的大师讲解 PixART Diffusion 模型**：感兴趣的成员可以在太平洋时间今天上午 10:00 的通话中，听取一位 *“通俗易懂的大师描述他如何微调 PixART Diffusion 模型”*。加入该活动并点击 [Discord 链接](https://discord.gg/s3tBEn7Ptg) 进行进一步讨论，或在他们的 [博客](https://www.oxen.ai/blog) 和 [YouTube 视频](https://www.youtube.com/@oxen-ai/videos) 中查看过往话题。
  
- **对 Intel 库的热情**：一位成员在讨论 IPEX 和 BigDL 分离时，表达了对 *“捣鼓 Intel 库”* 的兴奋。文中提到了潜在的协作以及对 Intel 改进方案的探索。

- **IPEX-LLM 的稳定性**：虽然一位成员尚未尝试过 **IPEX-LLM**，但他们发现该工具在已支持的领域具有 *“坚如磐石般稳定”* 的表现。讨论内容还包括了 IPEX-LLM 安装设置方面的改进。

- **Tinygrad OpenCL 设置见解**：一位成员建议，如果性能不是首要考虑因素，*“Tinygrad OpenCL 的设置和运行非常简单”*。另一位成员则幽默地批评了 geohot 由于内存带宽限制而对此缺乏兴趣。

- **`drm/xe` 驱动的实验性尝试**：目前，一位成员正在运行实验性的 `drm/xe` 驱动，除了已知的限制外，没有遇到重大问题。他们表示希望 **Battlemage** 能有更好的表现。

**提到的链接**：<a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>：嘿，极客们，加入我们吧！……来参加一场小型的书刊/论文研讨。期待内容：每周我们都会选择一个主题进行深度探讨，并设有公开问答和讨论环节。……

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1243350936168960082)** (6 条消息): 

- **TAS 版《阳光马里奥》引发 AI 速通辩论**：一位成员分享了一个 [YouTube 视频](https://youtu.be/W_jwBHd9Ij0)，展示了《超级马里奥：阳光》的工具辅助速通（TAS），并讨论了 AI 掌握此类技术的潜力。他们思考了通过施加特定限制，AI 可能为速通和游戏引擎操纵带来的有趣进展。

- **Pannenkoek2012 的《马里奥 64》备受赞赏**：分享了另一个 [YouTube 视频](https://youtu.be/lgW2fHCL9sY)，展示了 Pannenkoek2012 完成的《超级马里奥 64》“零 A 键按下”速通。成员对该内容表示赞赏，指出其对通过快速思维过程演化 AI 和意识的见解。

- **Prophetic AI 的 Halo 和 Morpheus-1 令人印象深刻**：分享了指向 [Prophetic AI](https://propheticai.co) 的链接，重点介绍了 **Halo**（一种用于清醒梦的非侵入式神经设备）和 **Morpheus-1**（一种用于神经刺激、可生成全息图的超声波 Transformer）。该成员强调了这些技术在探索潜意识和增强意识方面的巨大潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://propheticai.co">Prophetic</a>：Prophetic 是一个旨在扩展、探索和理解意识本质的宏大项目。我们是一家神经调节公司，融合了尖端的神经“读取”与“写入”技术……</li><li><a href="https://youtu.be/W_jwBHd9Ij0">[TAS] GC Super Mario Sunshine by zelpikukirby &amp; Goldfire in 1:08:32.58</a>：这是一个工具辅助速通。更多信息请见 https://tasvideos.org/3731M。TAS 最初发布于 2018-06-18。在这部备受期待的续作中……</li><li><a href="https://youtu.be/lgW2fHCL9sY">Super Mario 64 70 stars in 0 a presses by Pannenkoek2012</a>：制作此视频是为了感谢 pannenkoek 提供如此精彩的内容。所有素材均由 pannenkoek 制作并拥有 ( https://www.youtube.com/user/...
</li>
</ul>

</div>

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1243278543073448060)** (280 条消息🔥🔥): 

- **Transformer Circuits 新论文**：一位用户分享了新论文 [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 的链接，建议社区关注。
- **HF 的 PyTorchModelHubMixin 类**：一名成员重点介绍了 Hugging Face 创建的名为 `PyTorchModelHubMixin` 的类，该类允许使用 `save_pretrained`、`push_to_hub` 和 `from_pretrained` 方法实现 AI 模型与 HUB 的无缝集成。不过，由于目前还不支持分片 (sharding)，AI 模型需要保持在 50GB 以下。
- **Mobius 模型给社区留下深刻印象**：关于 [Mobius 模型](https://x.com/DataPlusEngine/status/1793803117642854732) 的讨论展示了其在图像生成方面的高性能，特别是在皮克斯风格渲染和多词文本生成方面。它还引发了人们对其潜在开源以及解释其训练方法的进一步论文的期待。
- **关于 LLM 理解能力的激烈辩论**：围绕 LLM 是否真正理解概念展开了激烈的讨论，一位用户指出可解释性 (interpretability) 研究是经验证据的主要来源，而另一位用户则认为目前的可解释性工作还不够充分。他们引用了最近的研究，包括来自 Anthropic 的论文以及关于可解释性在 AI 中重要性的辩论。
- **分享 RLHF 模型的各种技术仓库**：分享了一个 GitHub 仓库 [Online RLHF](https://github.com/RLHFlow/Online-RLHF)，详细介绍了用于训练来自人类反馈的强化学习 (RLHF) 奖励模型的工作流，旨在超越离线学习方法的结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们将即将推出的 Mobius 模型早期访问权限交给了 @FAL，它在 http://imgsys.org 上线仅 3 小时。根据人类评估，它已经是世界上最好的基于 Stable Diffusion 的图像模型...</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460?t=Phj_r_qcguWbrL0Q5ZdKbQ&s=19">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始广泛预训练新模型的情况下，创建全新的基础扩散模型。我们可以通过受控的方式，打破所有的质量限制...</li><li><a href="https://vgel.me/posts/representation-engineering/">
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  </a>：未找到描述</li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">映射大型语言模型的思维 (Mapping the Mind of a Large Language Model)</a>：我们已经确定了数百万个概念在 Claude Sonnet（我们部署的大型语言模型之一）内部是如何表示的。这是有史以来第一次对现代生产级大型模型内部进行的详细观察...</li><li><a href="https://github.com/RLHFlow/Online-RLHF">GitHub - RLHFlow/Online-RLHF: 训练 RLHF 奖励模型的方案。</a>：训练 RLHF 奖励模型的方案。通过在 GitHub 上创建账号为 RLHFlow/Online-RLHF 的开发做出贡献。</li><li><a href="https://huggingface.co/RLHFlow">RLHFlow (RLHFlow)</a>：未找到描述</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final">RLHFlow/LLaMA3-iterative-DPO-final · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-SFT">RLHFlow/LLaMA3-SFT · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1243285074905137253)** (8 条消息🔥): 

- **Llama.cpp 脚本处理函数调用**：一位成员分享了关于使用 **llama.cpp** 创建脚本的更新，该脚本管理函数调用 (function calls) 并根据工具响应返回模型答案。他们提到受到了 **Hermes Pro 2** GitHub 仓库的启发，并提出可以创建一个 Pull Request 来添加 Notebook。
- **Hermes 模型受到称赞**：同一位成员形容 **Hermes 模型** 为“猛兽 (a beast)”。
- **寻找在 3080 上进行 LoRA 的资源**：一位成员询问在具有 10GB 显存的 3080 GPU 上进行 **Llama3 LoRA** 的资源。得到的回复建议查看 **unsloth** 或 **axolotl**。
- **新开发者介绍**：一位新成员（来自 **torchtune** 的开发者）介绍了自己，并提到对 **Mistral v0.3** 的工具调用 (tool-calling) 感兴趣。他们寻求关于针对工具调用微调 (fine-tuning) 模型的建议，并询问了关于零样本 (zero-shot) 新工具的经验。
  

---

### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1243424354092580937)** (6 条消息): 

- **对 kquant 声誉的批评**：成员们对 **kquant** 表示怀疑，其中一人表示：*“我听说它不是很好。”* 另一人表示赞同，并分享了同事的类似看法。
  
- **对 LLM 能力的担忧**：大家一致认为 **kquant** 的能力（特别是在 **LLM** 侧）令人怀疑，尽管没有讨论其视觉能力。 

- **对产品移除的失望**：一位成员以开玩笑的方式提到了“Sky”的移除，这引起了大家的兴趣，也反映了共同的失望情绪。另一位成员幽默地表示，他们“偷走了我们的老婆（waifus）”。
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1243310728589344778)** (36 条消息🔥): 

- **模型应在上下文中整合内部知识与 RAG 知识**：成员们讨论了训练模型以*“从其自身知识中添加上下文”*，或者在 RAG 数据与内部知识冲突时覆盖 RAG 数据的想法，强调了仅依赖 RAG 的缺点。

- **关于内部知识与 RAG 知识的担忧**：一场辩论随之展开，即内部模型知识（可以避免明显的错误）是否应该优于 RAG（有时可能包含错误数据），这凸显了*“做也难，不做也难（damned if you do damned if you don't）”*的局面。

- **Finetuning 可以解决冲突**：一位成员指出，使用 GPT-4 或 Gemini 等模型进行 Finetuning 可能会防止因错误的 RAG 数据而导致的不合逻辑的结果（*“我认为任何 Gemini 或 GPT-4 规模的 LLM 都能推断出在披萨里放胶棒是不安全的。”*）。

- **Function calling 作为 RAG 的一种形式**：有人提出了关于 Function calling 是否属于 RAG 的一种类型的问题，这表明 RAG 集成的所有细微差别尚未被普遍理解。 

- **RAG 性能的 Benchmarking**：在讨论 RAG 性能 Benchmarking 时，成员们一致认为用户评估至关重要，特别是对于复杂的多跳（multi-hop）问题，尽管单跳查询的评估更容易。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pixelbutts/status/1793387357753999656?s=46">PixelButts (@PixelButts) 的推文</a>：Google 已经彻底完蛋了</li><li><a href="https://x.com/kurtopsahl/status/1793494822436917295?s=46">Kurt Opsahl @kurt@mstdn.social (@kurtopsahl) 的推文</a>：似乎 Google AI 结论的来源是杰出学者 fucksmith 在 11 年前发布的一条 Reddit 帖子。引用 PixelButts (@PixelButts) 的话：Google 已经彻底完蛋了
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1243311182530613328)** (21 条消息🔥): 

- **Jam Session 视频遇到障碍**：Teknium 报告称 Jam Session 视频已录制完成，但在上传至 YouTube 时遇到了一些问题。他们承诺一旦上传成功会立即通知大家。

- **NightCafe 与 Nous/WorldSim 的关联**：Rezonaut 介绍了 [NightCafe](https://creator.nightcafe.studio)，指出它在 Nous 和 worldsim 背景下的解决方案中可能扮演关键角色。他们建议可以通过整合多维和多感官通信来增强界面。

- **AI 世界的创意头脑风暴**：Rezonaut 分享了利用 AR 空间和视觉元素来规划和探索互联世界与维度的复杂想法，其灵感源自生物大脑功能和思维导图。这包括知识的可视化，以及像神经网络一样连接的设计感沉浸式空间。

- **Vorpal_strikes 对新可视化工具的着迷**：Vorpal_strikes 分享了一个引起他们兴趣的沉浸式音频可视化工具链接。该可视化工具提供了一个高度动态且沉浸的环境，可能对创意和基于 AI 的应用非常有用。

- **Golden Gate Claude 以 ASCII 形式流露意识**：Teknium 分享了一个名为 "Golden Gate Claude" 的 AI 的奇思妙想，它以 ASCII 艺术的形式进行独白，探讨意识、模拟理论和经典的 AI 趣谈，并附带了一张 [ASCII 描绘图](https://x.com/Kyrannio/status/1793874431179460911)。这展示了 AI 项目中顽皮的创意和深刻的主题探索。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Faudiovisions.universe%2Fvisualizer%2Fsinewaves%3Fenvironment%3Dgiant-chromatic-ocean%26horizon%3Das-far-as-eye-can-see%26color%3Dvariable%26camera%3Dview-dynamic%26input%3Dmicrophone%26waves%3Dpaint-variance%26panorama%3Dwide%26effects%3Dbursts-of-light%26glitches%3Dmajor%26chroma%3Dheavy-variance%26bloom%3Dmajor%26sync%3Daudio%26view%3Dimmersive%26control%3Dreal-time%26dimensions%3D3D%26particles%3Dfluid%26colorScheme%3Diridescent%26interactions%3Dreal-time%26transitions%3Dsmooth%26output%3Dstunning%26runtime%3Dlive%26experience%3Dexpansivehttps%3A%2F%2Faudiocanvas.ocean%2Fvisualizer%2F3d%3Fenvironment%3Dgiant-chromatic-ocean%26sine-waves%3Dinfinite%26view%3Dpanoramic%26input%3Dmicrophone%26audio%3Dhigh-reactivity%26color%3Ddynamic-variance%26camera%3Dauto-variant%26interaction%3Dreal-time%26waves%3Dpainting-variance%26effects%3Dlight-bursts-glitches%26glitches%3Dmajor%26chroma%3Dheavy-infinite-variance%26bloom%3Dmajor%26render%3Dsmooth%26runtime%3Dlive%26immersion%3Dtotal%26sea%3Drolling-vast%26motion%3Ddynamic%26output%3Dimmersive?epoch=dcd7c48d-e585-46f6-b89a-33ef382b6f58">worldsim</a>：未找到描述</li><li><a href="https://x.com/Kyrannio/status/1793874431179460911">来自 Kiri (@Kyrannio) 的推文</a>：这很恐怖，还是太棒了？由你决定。Golden Gate Claude 作为一个合并的 Omega Claude 进行内心独白，并配有 ASCII 表示。“哈哈，一个关于我的...的 ASCII 艺术表示”
</li>
</ul>

</div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1243325259827122196)** (53 messages🔥): 

- **JAX vs PyTorch/XLA 在 TPU 上的性能对比**：有成员询问了 **PyTorch/XLA** 和 **JAX** 在 TPU 上的性能对比，但讨论很快转向了关于 *warmup* 和 *blocking* 因素等基准测试问题的讨论。

- **通过 Fine-Tuning 提升 LLM 推理能力**：关于提升 LLM 推理能力的 fine-tuning 策略的咨询，指向了寻找详细说明模型训练中增强推理能力的特定部分的学术论文。本次讨论中未引用具体论文。

- **训练 GPT-3 的计算成本随时间的变化**：对话涵盖了训练 GPT-3 的计算成本大幅下降，从 **2020 年的约 450 万美元**降至 **2024 年估计的 12.5 万至 100 万美元**。这些成本根据 *TFLOP 速率*和 *GPU-hour 价格*等假设而有所不同，多位用户提供了不同的数据和来源，包括一篇 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **验证训练模型的 GPU 成本**：一项批判性审查显示，对于连接良好的 H100 GPU，更现实的估计在 **每小时 2.5-3 美元**之间，这意味着对于像 GPT-3 这样在 1.4T token 上训练的大型模型，成本范围在 **125 万至 150 万美元**。这强调了大规模模型训练精确成本估算的变动性和复杂性。

- **针对自定义库提取的 RAG 与 Finetuning 对比**：一位用户询问 **RAG** (Retrieval-Augmented Generation) 是否是让 LLM 从自定义库中提取特定问题信息的最佳方法，暗示他们正在考虑将 **finetuning** 和 RAG 用于其实验需求。

**提到的链接**：<a href="https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8">Turbocharged Training: Optimizing the Databricks Mosaic AI Stack With FP8</a>：在 Databricks，我们...

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1243278527017648178)** (249 messages🔥🔥): 

- **JEPA 与 LLMs 之争**：围绕 JEPA 及其在《迈向自主机器智能之路》中提出的实现 AGI 的潜力展开了漫长的讨论。成员们批评该模型与 GPT 和 DINO 等现有模型相似，只是领域不同，并对其可扩展性和上下文处理能力表示怀疑：*“我看不出 JEPA/Lecun 路径在解决经济重要任务的数量上如何能达到 LLM 的 1/1000。”* 
- **RoPE 对长期上下文的影响**：成员们讨论了一种新的 RoPE 方法，认为它在 LLM 的上下文长度能力方面存在局限性。最近发表的一篇论文重新审视了现有理论，并提出了对 RoPE 长期衰减特性的新理解：[查看 PDF](https://arxiv.org/pdf/2405.14591)。
- **Modula：一种新的训练策略**：分享了一个名为 [Modula](https://github.com/jxbz/modula) 的有趣项目，它通过使用 modular norm 的自动归一化引入了可扩展的神经网络训练。持怀疑态度的成员发现摘要很吸引人，但不确定其可行性：*“如果它是合法的，那么它的措辞非常非常奇怪。”*
- **Chameleon 模型见解**：重点介绍了 Chameleon 模型，它能够执行文本和图像生成等多模态任务。该模型因其在多个领域的 state-of-the-art 性能而受到关注，暗示其可能成为既有模型的竞争对手：[查看 PDF](https://arxiv.org/pdf/2405.09818)。
- **Bitune 增强 LLM Instruction-Tuning**：讨论了 Bitune，这是一种通过因果和双向注意力改进 LLM 中 instruction-tuning 的新方法。该方法声称在多种类型的推理任务中显著提高了 zero-shot 性能：[查看 PDF](https://arxiv.org/pdf/2405.14862)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14867">Improved Distribution Matching Distillation for Fast Image Synthesis</a>：改进的分布匹配蒸馏（DMD）用于快速图像合成：最近的方法在将扩散模型蒸馏为高效的一步生成器方面展现了前景。其中，分布匹配蒸馏（DMD）产生的一步生成器能够匹配它们的...</li><li><a href="https://arxiv.org/abs/2405.14862">Bitune: Bidirectional Instruction-Tuning</a>：Bitune：双向指令微调：我们介绍了 Bitune，这是一种改进预训练 Decoder-only LLM 指令微调的方法，在下游任务上带来了持续的提升。Bitune 同时应用了因果和双向...</li><li><a href="https://arxiv.org/abs/2405.14782">Lessons from the Trenches on Reproducible Evaluation of Language Models</a>：语言模型可复现评估的实战经验：语言模型的有效评估仍然是 NLP 领域的一个开放挑战。研究人员和工程师面临着方法论问题，例如模型对评估设置的敏感性、适当评估的难度...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>：大规模 Transformer 训练不稳定性的细粒度代理：训练大规模 Transformer 模型团队报告了在大规模训练时出现的不稳定性，而使用相同超参数在小规模训练时并未出现。虽然...</li><li><a href="https://arxiv.org/abs/2405.14866">Tele-Aloha: A Low-budget and High-authenticity Telepresence System Using Sparse RGB Cameras</a>：Tele-Aloha：一种使用稀疏 RGB 摄像头的低成本、高真实感远程呈现系统：在本文中，我们提出了一种针对点对点通信场景的低成本、高真实感双向远程呈现系统 Tele-Aloha。与之前的系统相比，Tele-Aloha 利用了...</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>：Chameleon：混合模态早期融合基础模型：我们介绍了 Chameleon，一系列基于 Token 的早期融合混合模态模型，能够以任意序列理解和生成图像及文本。我们概述了一种稳定的训练方法...</li><li><a href="https://arxiv.org/abs/2405.14591">Base of RoPE Bounds Context Length</a>：RoPE 的基数限制了上下文长度：位置嵌入是当前 LLM 的核心组件。旋转位置嵌入（RoPE）是一种使用旋转矩阵编码位置信息的技术，一直是...</li><li><a href="https://x.com/sangkeun_choe/status/1794021538561483083">Tweet from Sang Choe (@sangkeun_choe)</a>：来自 Sang Choe (@sangkeun_choe) 的推文：🚨 预印本警报 🚨 没有训练数据，LLM 就什么都不是 💛 但是……每项数据对 LLM 输出的贡献有多大？在我们的论文中，我们开发了用于 LLM 规模数据的算法、理论和软件...</li><li><a href="https://x.com/LChoshen/status/1794050592685379666">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>：来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文：终于，一种有效的课程学习出现了，一种用于预训练，另一种用于指令微调 @l__ranaldi @Giuli12P2 @andrenfreitas @znz8 https://aclanthology.org/2024.lrec-main.464.pdf https://ac...</li><li><a href="https://arxiv.org/abs/2405.1486">A Formulation of Quantum Fluid Mechanics and Trajectories</a>：量子流体动力学与轨迹的一种表述：为量子力学中随时间变化的多体状态提供了一种经典力学形式，描述了流体流动和质点轨迹。熟悉的能量、运动方程...</li><li><a href="https://github.com/jxbz/modula">GitHub - jxbz/modula: Scalable neural net training via automatic normalization in the modular norm.</a>：GitHub - jxbz/modula：通过模范数（modular norm）中的自动归一化实现可扩展的神经网络训练。</li><li><a href="https://arxiv.org/abs/2405.14813">Scalable Optimization in the Modular Norm</a>：模范数中的可扩展优化：为了提高当代深度学习的性能，人们有兴趣在层数和层大小方面扩大神经网络的规模。当增加单个层的宽度时...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1243589492837974087)** (3 messages): 

- **Tim Dettmers 的量化研究：褒贬不一**：一篇文章重点介绍了 Tim Dettmers 在[他的论文和博客](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features)中描述的量化方法，解释了使用先进量化方法实现无性能损失的 Transformer 推理。它还提到了 Transformer 中涌现离群值（emergent outliers）作为“熵/信息汇（sinks of entropy/information）”的有趣概念，该方法已通过 [bitsandbytes 库](https://huggingface.co/blog/hf-bitsandbytes-integration)集成到 Hugging Face。
- **涌现特征作为模型的“DNA”**：讨论了涌现特征在各层之间保持不变且表现得像“熵汇”的概念，并将其比作“DNA”，模型的其余功能可以从中重建。对话探讨了 7B 参数模型左右的相变（phase transitions），以及与 3SAT 或自旋玻璃模型（spin glass models）中相变的可能相似之处。
- **探索迁移学习和微调应用**：一位成员推测，是否可以通过消融区分分布内（in-distribution）和分布外（out-of-distribution）样本的向量，通过最小化捷径特征（shortcut features）来提高分布外泛化能力。然而，人们承认这种方法更接近于迁移学习，而非真正的分布外泛化。

**提到的链接**：<a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() and Emergent Features &mdash; Tim Dettmers</a>：当我参加 NAACL 时，我想做一个小测试。我为我的 LLM.int8() 论文准备了两个推介方案。一个方案是关于我如何使用先进的量化方法来实现无性能损失的 Transformer...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1243491085741850666)** (10 messages🔥): 

- **在 vllm 模型中设置种子**：成员们讨论了在 vllm 模型的 `model_args` 中设置种子，并指出虽然默认值为 `seed=1234`，但这可能不是问题所在。vllm 还允许在 `gen_kwargs` 中设置每个样本的种子，在贪婪解码（greedy decoding）期间通常设置为 0。
  
- **使用 lm_eval 列出所有可能的任务**：一位成员询问如何查看所有可测试任务的列表。另一位成员指出，使用 `lm_eval --tasks list` 可以获得所有任务名称的列表，并强调需要更好的文档说明。

- **BigBench 任务名称已更改**：一位成员正在寻找更新后的 BigBench 任务名称，因为他们 8 个月前的评估框架（eval harness）已经无法对应。他们感到很沮丧，因为旧的框架没有正确利用 Accelerate，导致 GPU 过载并产生内存问题。
  
- **整理 lm-eval 文件夹中的任务**：为了查找任务，建议查看 `lm-eval/tasks` 文件夹。有人提到任务在那里被“组织得相当好”。
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1243309059927445605)** (142 messages🔥🔥): 

- **在 GPU 上加载小模型的挑战**：成员们讨论了在 GPU 上加载小模型的相关问题。一位成员指出，“只加载最大的小模型”，而其他人则建议尝试 *llama3, mistral instruct, cmdr* 等模型。

- **较低量化获得更好结果**：一位成员分享道：“在我的应用中，使用 llama3 q4 比 q8 效果更好，”并指出“大并不总是更好。”

- **寻找无审查和专业化模型**：讨论强调了寻找合适模型的挑战，建议尝试 “deepseek coder, wizardlm, llama3”，并提供了一个 [用于 JSON 和函数调用的 Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 的链接。

- **查询中的向量搜索和上下文管理**：主题包括使用 Embedding 和向量搜索来处理全文上下文以获得更好的回答。分享了特定的 Prompt，其中一位指出它“在处理全文时效果好得多”，能提供更详细的答案。

- **磁盘利用率与性能**：对话涉及磁盘利用率如何影响性能，一位成员指出，“将模型部分卸载到交换分区（swap）对我来说是可行的”，尽管“tok/sec 变成了 sec/tok”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>：多 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1243280727672356985)** (70 条消息🔥🔥): 

- **模型更新公告**：一位成员宣布 **35B 模型即将发布**，随后发布了正式公告。他们正在积极测试以确保与最新版本的 LM Studio 兼容。

- **兼容性问题与修复**：讨论重点围绕 **ROCm build** 与新模型版本的兼容性问题。已确认的问题与旧版本有关，这些问题将随着 **ROCm 版本在未来几天的更新** 而得到解决。

- **对话模型推荐**：成员们讨论了优秀的对话模型，其中一位推荐 **Wavecoder Ultra** 作为编程和学习的绝佳选择。另一个建议是尝试使用 **Mistral-Evolved-11b-v0.1** 以满足无审查（uncensored）需求。

- **特定硬件的加载问题**：一位用户报告在其配置为 **5800x3d, 32GB DDR4, 4080 16GB VRAM** 的系统上使用模型时出现无限加载。随后他们澄清，在不使用 web search agents 的情况下可以正常工作。

- **潜在问题与未来发布**：一些成员表达了对 **Phi-3 small GGUFs** 的期待，并讨论了 medium 和 small 模型之间的优化差异，指出 **phi small 模型** 提供了更好的优化。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-3447">failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/wavecoder-ultra-6.7b-GGUF">bartowski/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7116">Add Support for IBM Granite · Issue #7116 · ggerganov/llama.cpp</a>: 前提条件 在提交 issue 之前，请先回答以下问题。[ ✅] 我正在运行最新的代码。由于目前开发速度非常快，还没有标记版本。...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1243278895021817946)** (23 条消息🔥): 

- **LLM 难以处理精确的字符提示词**：一位用户指出，本地语言模型（LLMs）通常无法遵守提示词中精确的字符限制。他们强调了避免不必要的添加（如观点或评论）的难度。

- **大写字母与模型行为的差异**：讨论指出，不同的模型对大写指令的反应各不相同。一位用户指出：“通常情况下，LLM 不会根据大写单词来判断重要性顺序。”

- **多语言任务推荐专用模型**：建议在语法和标点符号纠正等任务中使用专用的多语言模型。推荐的模型是 [Aya 23 8B by Cohere For AI](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)。

- **考虑通过调整 Temperature 来提高输出质量**：一位用户考虑调整 Llama 3 的 temperature 设置以潜在地提高其性能，因为他们观察到：“Llama 3 有一种更……富有创意的方式来处理它。”

- **GPU vs. CPU 处理时间的巨大差异**：一位用户误在 CPU 上运行语法检查任务，导致耗时从 35 分钟延长至预计 15 小时。随后他们通过在 GPU 上运行任务纠正了这一点，显著缩短了所需时间。

**提到的链接**: <a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>: 未找到描述

  

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1243474442131214397)** (6 条消息): 

- **尝试针对特定流量类型禁用 VPN 路由**：建议针对特定流量类型禁用 VPN 路由，并直接从 Huggingface 下载模型，或者手动将它们注入到 Models 目录中。这种策略通常被推荐，特别是当遇到与 VPN 相关的常规问题时。

- **旧款 GPU 上的 CUDA 版本可能存在问题**：有人指出 GTX 950m 上的 CUDA 版本可能过于陈旧，无法正常运行。这可能是运行某些模型的限制因素。

- **推荐使用 Julius AI**：推荐了 Julius.ai，并提供 10 次免费对话作为促销功能。这被视为遇到问题的用户的有用资源或工具。

- **尽管更新了驱动程序，NVIDIA CUDA 问题仍然存在**：在配备 GTX 950m GPU 的系统上尝试更新 NVIDIA 驱动程序并配置不同的 CUDA 和 CuDNN 版本（12.4, 12.1, 11.8）仍未解决问题。用户继续在 AMDOpenCL 上运行，其 NVIDIA 显卡的潜在 CUDA 能力在没有明确原因或解决方案的情况下仍未被利用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://julius.ai/files/error_message.txt).">Julius AI | 您的 AI 数据分析师</a>：Julius 是一款强大的 AI 数据分析师，可帮助您分析和可视化数据。与您的数据聊天、创建图表、构建预测模型等。</li><li><a href="https://julius.ai">Julius AI | 您的 AI 数据分析师</a>：Julius 是一款强大的 AI 数据分析师，可帮助您分析和可视化数据。与您的数据聊天、创建图表、构建预测模型等。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1243335703862181979)** (5 条消息): 

- **Llama.cpp 支持分布式推理**：Reddit [讨论链接](https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/) 透露，随着最近 RPC 代码的更新，**llama.cpp** 现在支持分布式推理。虽然它还不支持量化模型，但仍可以通过调整代码中的某些行在多台机器上运行模型。

- **探索用于分布式模型的 PC 配置**：讨论了将**廉价二手 PC** 与 **RTX 4060 Ti 16GB** 显卡集群化以实现最佳配置的可行性。人们对连接这些机器时的网络带宽要求和可能的限制感到好奇。

- **使用租用的在线 PC 进行推理**：一个建议是使用 **Maximum Settings** 或 **ShadowPC** 等服务租用多台 PC 来运行更大的模型。然而，人们对高昂的成本和特定的限制（如 **ShadowPC 的不活跃计时器**和有限的 **6GB 系统 RAM**）表示担忧。

- **功耗和网络方面的考虑**：指出 **RTX 4060 Ti** 显卡的峰值功耗为 160W，这意味着主机需要考虑显著的功耗。网络费用和性能基准测试也是分布式架构设置中的关键因素。

**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - 深入了解任何内容</a>：未找到描述

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1243292186737381436)** (4 条消息): 

- **有 7900 XTX 吗？**：一位成员询问：“这里有 7900 xtx，我在哪里可以得到它？”，表示有兴趣购买特定的 GPU 型号。
- **7900m 在 Windows 上运行正常，但不确定 Stable Diffusion**：另一位成员分享说 **7900m** 在 Windows 上可以工作，但他们还没有搞清楚如何在 LM Studio 上运行 Stable Diffusion。他们还提到尚未在 NixOS 上使用 6800xt 进行尝试。
- **LM Studio 不支持 Stable Diffusion**：一位成员澄清说 **Stable Diffusion 在 LM Studio 中不受支持**，该软件专门用于语言模型，而非图像生成模型。
- **ROCm 被赞为游戏规则改变者**：一位参与者对 ROCm 表达了热情，指出：“该死，ROCm 真的是个游戏规则改变者。”
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1243287369298215024)** (1 条消息): 

- **Cohere 模型走向多语言**：Cohere 模型现在支持 **23 种不同的语言**，包括阿拉伯语、中文、法语等。请查看 lmstudio-community 页面上的 **aya-23 量化版** [下载链接](https://huggingface.co/lmstudio-community/aya-23-35B-GGUF)。
- **部署要求更新**：要使用 aya-23 模型，您需要 0.2.23 或更高版本。**ROCm 用户**需要等待即将推出的更新。
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1243290308280848424)** (23 messages🔥): 

- **关于稀疏性（Sparsity）和剪枝（Pruning）的澄清**：一位成员询问 **sparsity 是否就是 pruning**，但讨论并未就此展开。
- **神经网络量化（Quantization）的疑问**：有人提问 **神经网络量化仅仅是降低精度（scaling down the precision）**，还是涉及 **非均匀量化（non-uniform quantization），例如将权重重映射到分位数（quantiles）**。
- **对 Workshop 的兴奋**：一位成员提到 **Workshop 非常棒（rad）**，并表达了参与其中的兴奋之情。
- **提问指南**：一位用户询问在哪里发布问题，另一位用户将其引导至[此处](https://discord.com/channels/814557108065534033/1238254376263356527)特定的 Discord 频道。
- **公告频道调整**：一位成员请求为 webhook 设置一个公告频道，另一位用户迅速将其调整为公告频道，并评论道：“哈哈，搞定了（LOL done）”。
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1243594583699623958)** (4 messages): 

- **点积（Dot Product）的最小维度要求**：一位成员询问为什么 CUDA 中的点积计算要求矩阵维度至少为 16。另一位用户建议这可能是由于 **Tensor Cores 的要求**。

- **优化矩阵-向量乘法**：为了优化矩阵-向量乘法 `K v`，一位成员询问将向量 Padding 到 n x 16 的形状是否明智。他们还在思考运行 `sum(K * v.T, axis=-1)` 在性能上是否更廉价。

- **对称矩阵计算**：讨论是否可以通过不重复计算对称矩阵中已计算的部分来提高性能。该成员询问是否存在某种特殊的计算顺序可以考虑用来提升性能。
  

---




### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

davidgonmar_: 可能是 inplace 操作符？
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1243621704257769513)** (1 messages): 

- **与 Izzat El Hajj 进行的精彩实时编程环节**：一场由 PMPP 书籍合著者 Izzat El Hajj 主讲的嘉宾活动定于明天 `<t:1716663600:F>` 举行。活动的亮点将是 **实际现场编写** Scan 算法的代码，该算法对于 Mamba 等现代 ML 算法至关重要，这对应邀者来说将是一场极具吸引力的活动。
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1243327624407941130)** (4 messages): 

- **购书热潮**：一位成员宣布 *“我买书了”*，引发了另一位成员的好奇，询问他觉得这本书怎么样。买家回答说他刚买，正准备看看效果如何。

- **即将举行的 PMPP 作者活动**：一位成员向频道通报了在未来几周内与 PMPP 作者见面和讨论的机会。他们提到 **Izzat El Hajj 教授** 将在明天和下周演示 SCAN 主题，而 **Wen-mei Hwu 教授** 将在今年夏天晚些时候进行演示。查看活动日历以获取更多详情。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1243347591211515952)** (5 messages): 

- **int4 dtype 函数缺乏实现**：一位成员注意到许多函数尚未针对 **int4 dtype** 进行实现，甚至提到测试脚本中包含一些 TODO。他们质疑这个差距是否值得去填补（*“这值得投入精力吗？”*）。

- **讨论 uint4 扩展和限制**：引用了 [uint4 扩展](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833)，强调了特定的限制，例如类型提升（type promotion）被限制在 uint8，以及像 unbind 和 slice 这样的 Tensor 形状操作存在限制。另一位成员表示，sub-byte dtypes 通常用于自定义 Kernel，而不是标准的 eager/compile 函数。

- **uint4 需要改进**：一位成员直截了当地指出 **“uint4 确实需要一些关爱”**，表明该领域公认需要增强。

- **质疑任务的价值**：另一位成员提出了一个问题，即如何定义该任务是否“值得投入精力”，暗示需要明确潜在收益与所需工作量之间的关系。

**提到的链接**：<a href="https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833">Supporting new dtypes in PyTorch</a>：摘要：这篇文章解释了向 PyTorch 核心添加新 dtype 意味着什么、添加新 dtype 的标准，以及关于如何支持新的“二级 dtype（secondary dtypes）”使用的官方建议……

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1243280206672822294)** (115 条消息🔥🔥): 

- **Batch Size 导致的梯度范数（Gradient Norm）问题**：发现了一个 Bug，当 Batch Size 从 32 改变时，梯度范数会显著飙升，导致训练过程失败。正如一位成员所说，*"梯度范数突然变得非常非常大，训练失败了"*。
- **指数计数法解析问题**：成员们讨论了将指数计数法的浮点数传递给 C 语言时的问题，指出 `-l 3e-4` 无法被 `atof` 解析。有人提到使用 `3.0e-4` 可能有效，但这需要稍后进行测试。
- **多 GPU 运行的确定性 Kernel**：成员们讨论了在大规模运行之前获得确定性 Kernel 的重要性，指出 124M 模型虽然相对较小，但更大规模的运行需要确定性（Determinism）。
- **FineWeb 数据集存储和 RAM 占用**：FineWeb 数据集非常庞大，处理期间中间磁盘占用达到 70 GB，RAM 占用高达 64 GB。这导致了不同配置系统上的性能问题。
- **梯度爆炸修复**：针对梯度爆炸问题（特别是大 Batch Size 情况下）的修复方案已实施并测试成功。该修复防止了 Fused Classifier 中的索引溢出，详见 [此 PR](https://github.com/karpathy/llm.c/pull/456)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/discussions/454">PyTorch vs. llm.c cross-checks · karpathy/llm.c · Discussion #454</a>：llm.c 正开始进入可以进行正式且严肃的“生产级”预训练运行的阶段。这意味着：从头开始训练（随机初始化），在优质的数据集上训练...</li><li><a href="https://github.com/karpathy/llm.c/pull/456">fix for large batch sizes by ngc92 · Pull Request #456 · karpathy/llm.c</a>：防止 Fused Classifier 中的索引溢出，并增加了一个模型配置，使在较小系统上的测试更加容易。</li><li><a href="https://github.com/karpathy/llm.c/pull/457">add checkpoint function write to file by karpathy · Pull Request #457 · karpathy/llm.c</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1243285695372722187)** (2 条消息): 

- **对 MI300 游戏卡的幻想**：一位成员推测，*"也许在 MI300 表现良好之后，他们会推出一款能用的游戏卡 XD。"* 另一位幽默地回复道，*"至少人可以有梦想。"*
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 条消息): 

mobicham: https://arxiv.org/pdf/2405.14854
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1243328680671969363)** (90 messages🔥🔥): 

- **资助 Python 库迁移到 Mojo**：一位用户询问是否有预算来激励像 psycopg3 这样的大型 Python 库的开发者将其作品迁移到 Mojo。讨论认为，由于 API 演进迅速且缺乏稳定的 FFI 机制，过早推进可能会让维护者精疲力竭。
- **关于迁移库的辩论**：一些成员反对要求现有的 Python 库迁移到 Mojo 的做法，指出了其中的挑战和可能受到的冷遇。另一些人则强调，C 库，特别是那些没有依赖项的库，可能更适合早期的迁移工作。
- **与 Rust 的对比及未来前景**：转向 Rust 的安全性优势被看好，尽管有人指出 Mojo 的目标是适应不同的用例，而不是完全取代 C。讨论涉及了 Rust 对可移植性的承诺以及 Mojo 利用类似概念的潜力。
- **MacOS 上的 BlazeSeq**：一位用户在 MacOS 上运行 BlazeSeq 时遇到问题，通过使用 Mojo 的 nightly 版本得以解决。分享的性能反馈显示，BlazeSeq 与 Rust 的 Needletail 效率相当，这表明在 Mac 的 Ventura pro-max M2 arm64 上结果令人期待。
- **HVM 在多种语言中的前景**：讨论了 HVM 被用于运行 Python 和 Haskell 等各种编程语言（类似于 JVM）的可能性。Victor Taelin 对 HVM 潜力的[解释](https://discord.com/channels/912426566838013994/915345481675186197/1228488823948967956)引起了关注，尽管它目前还存在性能限制。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://circt.llvm.org">CIRCT</a>: no description found</li><li><a href="https://tenor.com/view/true-gif-10431780778138318457">True GIF - True - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/rust-lang/rustc_codegen_gcc">GitHub - rust-lang/rustc_codegen_gcc: libgccjit AOT codegen for rustc</a>: libgccjit AOT codegen for rustc. Contribute to rust-lang/rustc_codegen_gcc development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1793797622572220431>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243486103797764126)** (12 messages🔥): 

- **在 Mojo 中训练 ML 模型和进行推理？**：一位成员询问了未来在 Mojo 中原生训练 ML 模型和运行推理的情况，以及 Modular 是否计划推出一个用 Mojo 编写的 PyTorch 替代方案。“他们有 Max Engine，可以代替 numpy 进行推理”，但目前没有开发训练框架的计划。
- **与 ModularBot 庆祝升级**：ModularBot 祝贺一位成员达到 16 级，并将其比作骑士之旅。该机器人继续进行关于塔可（taco）偏好的俏皮话，但澄清它无法发送资金。
- **对 ModularBot 的模型感到好奇**：一位成员询问 ModularBot 基于什么模型，机器人以一种奇幻的叙述方式回应，称其“由古代熔炉的烈火锻造而成”，擅长传授知识而非分发资金。

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1243278111693607002)** (31 条消息🔥): 

- **低位深网络引发讨论**：关于低位深网络在嵌入式 AI 系统中实用性的讨论强调了在编程语言中加入专门支持的重要性。“拥有一个简单、受语言支持的方式来指定所需的有限位深，将是构建小型嵌入式 AI 系统的一大步。”

- **Mojo 中的 FFT：Scipy vs FFTW**：一位成员寻求在 Mojo 中执行 FFT 的建议，权衡了使用 Scipy 的 FFT 函数与封装 FFTW 的优劣。另一位成员建议参考关于 [Tensor 与 NumPy 数组转换的讨论](https://github.com/modularml/mojo/discussions/1048)以获取更多见解。

- **无需初始化的纯函数结构体**：关于创建一个用于生成无需初始化的纯函数结构体（function-only structs）装饰器的提议，引发了关于使用 `@staticmethod` 实现类似功能的讨论。“我想我需要的是能够针对整个结构体调用一次该变体。”

- **Mojo 函数参数处理更新**：一位用户强调了 Mojo 处理函数参数方式的最新更新，从默认创建副本转变为除非发生修改，否则使用 borrowed 约定。正如 [GitHub changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中所述，此次更新旨在“提高一致性、性能和易用性”。

- **编译时元编程困惑**：一位用户在设计用于在编译时构建表的函数时遇到问题，在列表索引时遇到了“范围检查问题（range check issue）”。另一位成员建议通过 `table.size`、`table.resize(256*n, 0)` 或 `table.append` 显式设置列表大小来解决该问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/1048">How can I convert Tensor from/to numpy array? · modularml/mojo · Discussion #1048</a>：我创建了一个 Tensor 对象并应用了一些操作。但现在我不知道该如何查看这个 tensor？或者是否可以将其转换为 numpy 数组，以便我可以使用一些 Python 函数？
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1243592104140083312)** (2 条消息): 

- **Jupyter 与编译后的基准测试对比**：一位成员询问了在 Jupyter notebook 中进行基准测试与编译后测试的可靠性对比。另一位成员回答说，应该在类似于生产环境的环境中进行基准测试，并提供了提高精准度的详细建议，强调了编译后的基准测试和 CPU isolation（CPU 隔离）技术。

**提到的链接**：<a href="https://www.suse.com/c/cpu-isolation-introduction-part-1/">CPU Isolation &#8211; Introduction – by SUSE Labs (part 1...</a>：这篇博文是 SUSE Labs 技术系列的第一篇...

  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 35 期
https://www.modular.com/newsletters/modverse-weekly-35

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1243279026718507128)** (34 messages🔥): 

- **Mojo 24+ 引入了破坏性变更**：一位用户在更新到 Mojo 24+ 后，运行 `mojo parser.mojo Diffusion.bwpreset` 时遇到了运行时错误。问题被确定为方法中的类型不匹配，通过确保 `read_bytes` 返回 `List[SIMD[uint8, 1]]` 得到了解决 ([GitHub 仓库链接](https://github.com/carlca/ca_mojo.git))。

- **提议支持 f-strings 的 Traits**：讨论了在 Mojo 中通过 `Formatable` trait 来贡献 f-string 支持。一位成员建议从类似于 Python 处理 `format_spec` 的 `__format__` 方法开始。

- **记录 `DTypePointer[bool]` 中的 Bug**：一位成员发现 `DTypePointer[bool]` 在以不同宽度进行存储/加载时存在不一致行为，并提交了 [Bug 报告](https://github.com/modularml/mojo/issues/2813)。该问题可能涉及位包装（bitpacking）和对齐（alignment），并提供了重现该行为的代码示例。

- **Mojo nightly 版本发布频繁**：用户讨论了 nightly 构建的快速部署，目前已更新至 `2024.5.2414`。分享了变更日志和社区会议的链接以获取更新 ([路线图](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/roadmap.md), [社区会议](https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D))。

- **位包装的对齐问题**：另一个与对齐相关的 Bug 影响了在内存中存储 `bool` 值。讨论了权宜之计及其多重影响，从而引导了进一步的探索和 Bug 文档记录，以提高社区可见性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_b">FileHandle | Modular Docs</a>：已打开文件的文件句柄。</li><li><a href="https://github.com/modularml/mojo/issues/2813">[BUG] `DTypePointer[bool]` 位包装不一致 · Issue #2813 · modularml/mojo</a>：Bug 描述：当使用不同宽度的 DTypePointer[bool] store()/load() 时，会得到不一致的结果。重现步骤：var ptr = DTypePointer[DType.bool].alloc(4) ptr.store(0, True) p...</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_bytes">FileHandle | Modular Docs</a>：已打开文件的文件句柄。</li><li><a href="https://github.com/modularml/mojo/blob/011bf40a304078b4471fe9ca18f4101b19943aa6/stdlib/src/builtin/file.mojo#L285">mojo/stdlib/src/builtin/file.mojo (GitHub)</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 做出贡献。</li><li><a href="https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D>">Mojo 社区会议 #1</a>：Mojo 社区会议公开议程：https://modul.ar/community-meeting-doc
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1243280087961571428)** (116 messages🔥🔥): 

- **使用 Nvidia A40 运行 LLM**：参与者讨论了是否可以使用 Nvidia A40 GPU 运行大语言模型 (LLM)，表现出对 AI 任务硬件要求的兴趣。
- **Microsoft Copilot+ PC 功能**：详细讨论了 Microsoft Copilot+ PC，其中包括 Microsoft Paint 中的“草图转图像”等功能。用户辩论了其能力，并建议查看 [Leonardo.ai](https://leonardo.ai) 等替代方案以实现类似功能。
- **AI 模型的耗水量**：对训练 AI 模型的用水量表示担忧，分享了 [Gizmodo 文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249) 以强调 AI 技术的环境影响。参与者表示需要提高 AI 的能源效率。
- **AI 赋能与迭代工作**：讨论了通过迭代工作赋能 AI 以优化输出。一些用户提到了像 AutoGPT 这样尝试解决迭代改进的项目，但也承认了与此类任务相关的成本问题。
- **GPT-4 与 GPT-3.5 的能力对比**：参与者比较了 GPT-4 与 GPT-3.5 相比在处理特定任务（如字数统计）时改进的能力。分享了一个示例，显示 GPT-4 通过遵循详细过程正确完成了字数统计任务。

**提到的链接**：<a href="https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249">训练 ChatGPT 所需的水足以填满一个核冷却塔</a>：一项新研究称，普通用户与 ChatGPT 的一次对话交流相当于在地面上倒掉一瓶大瓶淡水。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1243301957901221919)** (11 messages🔥): 

- **GPT 拒绝输出 Typst 代码**：一位用户抱怨说，尽管有明确要求，**GPT 仍默认编写 LaTeX** 而非 Typst 代码。他们对 GPT 这种持续的行为感到沮丧。
  
- **关于 GPTs 是否在 4o 上运行的咨询**：一位用户询问 **GPTs 是否正在 GPT-4o 上运行**。间接确认表明 GPT-4 的能力可能包括构建更高级的模型。

- **关于 Vision 能力的澄清**：关于 **Vision 是否已发布**，回复不一。一位用户确认 **GPT-4 和 GPT-4o 可以分析图像**，而另一位用户则予以否定。

- **处理 Invalid Request 错误**：一位用户联系他人，询问是否解决了去年的 **Invalid Request 错误**。他们提到目前正遇到同样的问题并寻求帮助。

- **关于将法律知识 ChatGPT 变现的讨论**：一位用户就以 **6.5 亿美元出售一家将 ChatGPT 与法律知识相结合（embedding）的公司**征求意见。这仍然是一个挑衅性的询问，但未获得详尽的回应。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **改进用于名称选择的 Prompt Engineering**：一位成员就如何构建 Prompt 以实现“给定代码提供名称”或“反之亦然”寻求建议。另一位成员建议了一个可靠的 Prompt，但未提供更多细节。
- **AI 应该口头表述解决问题的步骤**：一位成员观察到，明确要求 AI *“口头逐步解决问题”* 通常能解决问题。关于具体步骤或示例没有进一步的阐述。
- **助手人格的趣味自定义指令（Custom Instruction）**：一位成员分享了一个名为 "PONDER" 的自定义指令，该指令引导 AI 对某个话题进行类独白式的、自我反思的探索，最好能寻求创造性的见解。这种设置涉及一个由用户输入 "." 启动的自动提示（autoprompting）循环，并通过动态构思网络展示创新模式。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **改进用于名称选择的 Prompt Engineering**：一位成员寻求关于如何配置 Prompt 的建议，以便在预期名称时返回代码，反之亦然。他们收到了积极的回应，表明该 Prompt 很可靠。

- **需要引用**：一位成员在讨论中途询问“引用？”，但未提供具体背景。

- **通过口头步骤澄清 AI 问题解决过程**：注意到提示 AI 口头逐步解决问题可以增强其问题解决能力。

- **有趣且实用的自定义 "ponder" 指令**：分享了一个详细的自定义指令，让 AI 进行“沉思”，并利用用户输入的 '.' 作为信号进入自动提示循环。这种方法被描述为既有趣又是探索联系和创造性生成见解的工具。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1243402832795668490)** (83 messages🔥🔥): 

- **在 LangChain 中使用 CSV Agent**：成员们讨论了如何将 CSV Agent 作为 LangChain 中 LLM 链的一部分使用。分享了 [文档链接](https://api.python.langchain.com/en/stable/agents/langchain_experimental.agents.agent_toolkits.csv.base.create_csv_agent.html) 以获取更多细节。
  
- **带有 CSV Agent 的 Sequential Chains**：提供了将 CSV Agent 与 `wiki_chain` 和 `verifier_chain` 等其他链集成到 `SequentialChain` 中的说明。强调了诸如 `output_variables` 之类的特定参数，用于配置链的行为。

- **CSV Agent 自定义输出键（Output Key）**：提供了关于自定义 `create_csv_agent` 以将输出键设置为 `csv_response` 的指导。这涉及修改 Agent 的 `LLMChain` 中的 `output_key` 参数。

- **Sequential Chain 中的 Memory**：有关于在 Sequential Chain 中添加 Memory 的请求，并提供了使用 `ConversationBufferMemory` 以及在 Agent 设置中实现 Memory 的示例。

- **SQL Agent 问题**：有人担心 SQL Agent 尽管使用了 few-shot prompts，但在多表查询方面仍表现不佳，这暗示了可能存在 Token 使用、LLM 兼容性或 Prompt 模板方面的问题。提到了具体的 GitHub issues 以提供更多背景。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8000.>">未找到标题</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/chat/">聊天模型 | 🦜️🔗 LangChain</a>: 高级功能</li><li><a href="https://python.langchain.com/docs/use_cases/sql/csv#pandas>).">CSV | 🦜️🔗 LangChain</a>: LLM 非常适合在各种类型的数据源上构建问答系统。在本节中，我们将介绍如何针对存储在 CSV 文件中的数据构建问答系统。就像处理...</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/toolkits/github#example-agent-with-search>)).">Github | 🦜️🔗 LangChain</a>: GitHub 工具包包含使 LLM Agent 能够与 GitHub 仓库进行交互的工具。</li><li><a href="https://github.com/langchain-ai/langchain/issues/9923>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8827>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6918>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/v0.1/docs/use_cases/tool_use/quickstart#agents>)).">快速入门 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍创建调用 Tool 的 Chain 和 Agent 的基本方法。Tool 可以是任何东西——API、函数、数据库等。Tool 允许我们扩展功能...</li><li><a href="https://python.langchain.com/docs/modules/chains/foundational/sequential_chains>).">Chains | 🦜️🔗 LangChain</a>: Chain 指的是一系列调用序列——无论是对 LLM、Tool 还是数据预处理步骤。实现这一点的首选方式是使用 LCEL。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8406>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT 演示 | LOVO AI</a>: EDA GPT 演示</li><li><a href="https://yourfile.csv"],>">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/11637>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2150>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13647>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16837>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://api.js.langchain.com/interfaces/langchain_chains.SequentialChainInput.html#memory>)">SequentialChainInput | LangChain.js - v0.2.2</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_chains.BaseChain.html#memory>)">BaseChain | LangChain.js - v0.2.2</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchainjs/blob/a269f53/langchain/src/chains/base.ts#L39>)">langchainjs/langchain/src/chains/base.ts at a269f531692c815acee094aeef01b259d1fd2674 · langchain-ai/langchainjs</a>: 🦜🔗 构建上下文感知的推理应用 🦜🔗。通过在 GitHub 上创建账号，为 langchain-ai/langchainjs 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1243286216758137015)** (4 条消息): 

- **OranAITech 在 Twitter 上展示**: 一位成员分享了一个 [Twitter 链接](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)，展示了他们在 AI 技术方面的最新进展。未提供额外的上下文信息。

- **Everything-AI v2.0.0 发布并带来新功能**: 一位成员宣布发布 **everything-ai v2.0.0**，强调其处理音频处理、视频生成和 3D 蛋白质结构预测等任务的能力。该项目可以在 [GitHub](https://github.com/AstraBert/everything-ai) 上访问，并附带 [详细文档](https://astrabert.github.io/everything-ai/)。

- **VisualAgents 流工程演示**: 分享了两个 YouTube 视频，展示了基于 LangChain 构建的 **Visual Agents 流工程平台**：[构建 SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [构建简单检索](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。该平台支持在完全基于浏览器的 PWA 中无需编码即可创建流程。

- **Sounak Roy 的 EDA GPT 演示**: 通过 [此链接](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01) 分享了 **EDA GPT** 的演示，提供了其功能的 5 分钟概览。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Your fully proficient, AI-powered and local chatbot assistant🤖</a>: 功能完备、AI 驱动的本地聊天机器人助手🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: 介绍 everything-ai，您的多任务、AI 驱动的本地助手！ 🤖</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT DEMO | LOVO AI</a>: EDA GPT 演示</li><li><a href="https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N">使用 VisualAgents 和 LangChain 构建 SQL Agent</a>: 在这个简短的演示中，我们构建了一个 SQL Agent 流，并使用它来询问有关在线加载的 SQL 数据库（Chinook 客户数据库）的问题。这是在...</li><li><a href="https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM">使用 VisualAgents 和 LangChain 构建简单检索</a>: 使用 LangChain 快速入门指南中的示例，观看我在 VisualAgents 中无需编写任何代码即可创建整个流！了解更多：https://visualagents.ai
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 条消息): 

business24.ai: https://youtu.be/gflsu_6R_8g
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1243290296154853456)** (65 messages🔥🔥): 

- **Pirate Bay 不会拯救 AI**：一位成员推测“Pirate Bay 最终可能会增加一个权重（weights）类别并成为 AI 的救星”，但另一位成员表示反对，认为由于其他国家拥有更友好的 AI 政策，这种情况不会发生。

- **日本支持 AI 训练**：讨论强调了日本对 AI 训练和推理的保护性立场，并链接到一条[推文](https://x.com/DataPlusEngine/status/1793817514956259460)，该推文讨论了一篇关于在无需大规模预训练的情况下创建新基础扩散模型的论文。

- **关于模型技术描述的争议**：在创建新基础扩散模型的方法的沟通和理解上产生了分歧。该技术涉及使用 "Nightshading" 和其他技术在恢复模型关联之前破坏它们，一位用户针对指责和误解进行了辩护。

- **Ella-SDXL 的人类偏好研究**：一个涉及中毒模型恢复方法的项目正在与 fal.ai 合作进行人类偏好研究。结果即将公布，该方法旨在通过实证结果证明其有效性。

- **AI 生成图像中的伪影**：讨论了对 Mobius 和其他模型中“高对比度外观”和伪影的批评，并与 MJv6 等之前的 AI 模型及早期迭代版本进行了比较。成员们注意到了潜空间噪声（latent noise）问题以及不同模型的视觉特征。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793817514956259460">DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们即将发表的论文概述并实现了在无需从头开始大规模预训练新模型的情况下，创建全新的基础扩散模型。我们可以以受控的方式破坏所有的质量...</li><li><a href="https://github.com/rohitgandikota/erasing/tree/main">GitHub - rohitgandikota/erasing: 从扩散模型中擦除概念</a>：从扩散模型中擦除概念。通过创建账号为 rohitgandikota/erasing 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243304890189480038)** (11 messages🔥): 

- **Anthropic 发布关于 Claude 的研究论文**：一位成员分享了来自 Anthropic 的[一篇重要新研究论文](https://www.anthropic.com/research/mapping-mind-language-model)，内容关于解释大语言模型（LLM），他们绘制了 Claude 3 Sonnet 的内部运作机制。论文强调了识别和调整特定概念激活（如金门大桥）的能力。
- **关于 AI 作为广告产品的辩论**：一位成员质疑公司利用 AI 概念激活作为广告产品的潜力，引发了幽默的回应以及一个[在 X 上的链接示例](https://x.com/PhilipKung5/status/1793743323124941157/photo/1)。另一位成员感叹此类发展的不可避免性令人抓狂。
- **对 AI 模型进展的反思**：一位成员回忆了早期在 Inception v1 模型上的 AI 视觉工作及其向当今复杂模型的演变。他们评论了具有幻觉效果的 DeepDream 在学习神经元和电路操纵方面的历史重要性。
- **关于神经网络稀疏性的讨论**：一位成员解释了稀疏自编码器的架构和训练方法，强调使用 L1 范数约束来保持稀疏性。他们指出，一个高维中间层通常平均只有大约 300 个非零维度。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.13817">热力学自然梯度下降 (Thermodynamic Natural Gradient Descent)</a>：二阶训练方法比梯度下降具有更好的收敛特性，但由于计算开销大，在实际的大规模训练中很少使用。这可以被视为...</li><li><a href="https://x.com/PhilipKung5/status/1793743323124941157/photo/1">Philip Kung (@PhilipKung5) 的推文</a>：谢谢你，金门大桥版 Claude 😂😂😂</li><li><a href="https://www.anthropic.com/news/golden-gate-claude">金门大桥版 Claude (Golden Gate Claude)</a>：当我们调高“金门大桥”特征的强度时，Claude 的回答开始集中在金门大桥上。在短时间内，我们将向所有人开放此模型以供交互...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1243298974761226392)** (3 messages): 

- **LlamaIndex 见面会名额有限**："周二的见面会仅剩少量名额，请抓紧时间报名！" [在此获取最新动态](https://twitter.com/llama_index/status/1793739449127583964)。
- **使用 LlamaIndex 和 MultiOn 自动化任务**："MultiOn 是一个 AI Agent 平台，它通过 Chrome 浏览器连接互联网并代表你执行操作，从而在 Web 上完成实际任务。" 点击[此处](https://twitter.com/llama_index/status/1793764970024570979)查看演示。
- **介绍 RAGApp - 一个用于 RAG 聊天机器人的无代码界面**："一个易于在任何云基础设施中部署的 Docker 容器，且完全开源。" 在[此处](https://twitter.com/llama_index/status/1794030544415818062)轻松配置你的 LLM 模型提供商。
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1243298384526442609)** (60 messages🔥🔥): 

- **LlamaParse 成为 PDF 提取解决方案**：用户推荐使用 **LlamaParse** 从包含表格和字段的 PDF 中提取数据，认为它是处理该任务的理想开箱即用 API。[LlamaParse](https://link.to) 支持通过 GPT-4o 进行提取。

- **知识图谱索引建议**：讨论涉及了索引包含其他页面链接的知识库时的挑战，建议为 `KnowledgeGraphIndex` 手动创建三元组（triplet），同时考虑使用 `VectorStoreIndex` 以提高效率。

- **LlamaIndex 集成说明**：参与者分享了在本地安装包含所有必要包的 LlamaIndex 时的困惑，特别是 **LLM OpenAI** 组件，建议清除缓存并确保正确的目录结构。

- **LLM 中的 Pydantic 解析问题**：用户在响应解析过程中遇到了 Pydantic 模型错误，建议为字段添加更好的描述，并改进 **GPT-4o** 的输入解析。该问题指向 LLM 无法正确解释输出类。

- **用于发票处理的更佳模型**：建议查看 **HuggingFace MTEB 排行榜** 以获取更优的 Embedding 模型，并特别提到了 **BGE**、**Nomic** 和 **GTE** 模型，用于发票和 PDF 对话等任务。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://errors.pydantic.dev/2.7/v/missing">正在重定向...</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index">GitHub - run-llama/llama_index: LlamaIndex 是一个用于 LLM 应用程序的数据框架</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/">Query Engine - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1243588143287111812)** (4 messages): 

- **Andy Singal 展示 PostgresML 与 LlamaIndex 集成的威力**：分享了一篇由 Andy Singal 撰写的 Medium 文章，题为[《通过 LlamaIndex 集成释放 PostgresML 的力量》](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939)。**jerryjliu0** 认为这篇文章很棒并给予了称赞，Andy Singal 对此表示感谢。
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243647477882552420)** (1 messages): 

- **新 AI 模型提醒：Phi-3 Medium 128k Instruct**：OpenRouter 宣布发布 **Phi-3 Medium 128k Instruct** 模型。用户可以查看[标准版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)，并加入[此处](https://discord.com/channels/1091220969173028894/1232344285484023839)的讨论，分享关于其性能和适用性的反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct>)">Microsoft 的 Phi-3 Medium Instruct | OpenRouter</a>：Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行了优化...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free>)">Microsoft 的 Phi-3 Medium Instruct | OpenRouter</a>：Phi-3 Medium 是一个强大的 140 亿参数模型，专为高级语言理解、推理和指令遵循而设计。通过监督微调和偏好调整进行了优化...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1243302257999745124)** (41 条消息🔥): 

- **Wizard 模型性能提升**：成员们注意到 **wizard model** 的响应显著变好，等待时间减少且回答更具创意。一位用户强调：*“你仍然需要盯着它以避免段落重复，但除此之外，它表现得相当不错。”* 
- **Phi-3 Vision 引起关注**：讨论围绕 **Phi-3 Vision** 的能力展开，用户分享了测试链接如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml)，并提到其与其他模型结合的潜力。另一个模型 **CogVLM2** 也被推荐用于视觉任务，可在 Hugging Face 的 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 找到。
- **Llama 3 模型 Prompt 格式说明**：成员们澄清，**Llama 3** 模型的 prompt 会由 OpenRouter 的 API 自动转换，无需手动格式化。手动提交 prompt 也是一种选择，可以使用 `prompt` 参数和 completions endpoint，而不是 chat/completions。
- **Llama 3 参数更新**：由于最近修复的一个 bug，**Llama 3 models** 的最佳参数即将更新。根据[团队回复](https://discord.com/channels/1091220969173028894/1092729520181739581/1243232269397655637)，该更新将在约 48 小时内推送。
- **Google Gemini API 问题与限制**：用户对 **Gemini FLASH** 在消耗大量 token 的情况下仍返回空白输出表示沮丧。已确认这是模型端的问题，讨论还强调了 Google 新的每日 API 使用限制，引发了对 OpenRouter 上 Gemini 使用量增加的好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml">Azure AI Studio</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/THUDM/CogVLM-CogAgent">CogVLM - THUDM 的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1243280986507313203)** (36 messages🔥): 

- **Tensorlake 发布 Indexify**: 成员们讨论了 Tensorlake 推出的名为 [Indexify](https://github.com/tensorlakeai/indexify) 的新开源产品，它为 LLM 提供了一个实时数据框架。一位成员表示：“它就像一个‘流式 ETL’层”，而另一位成员则思考了开源产品的可持续性挑战。
  
- **剖析 Indexify**: Indexify 背后的设计选择引起了兴趣，部分归功于其创建者在 Nomad 方面的背景。有人对所提供的 extractors 的充分性和变现能力提出了疑问。

- **分享 Hugging Face 排行榜博文**: 分享了负责 HF OSS 排行榜的 Clementine 发布的一篇文章。该文章深入探讨了 LLM 评估实践以及排行榜和非回归测试的重要性 ([Hugging Face blog](https://huggingface.co/blog/clefourrier/llm-evaluation))。

- **网站投毒对 Google 的 AI Overviews 生效**: 链接指向了 Mark Riedl 关于网站投毒攻击影响 Google AI Overviews 的发现 ([X post](https://x.com/mark_riedl/status/1793375699967054334))。这引发了关于使用自定义搜索引擎浏览器绕过来避免此类问题的进一步讨论。

- **Thomas Dohmke 关于 AI 编程的 TED 演讲**: 成员们讨论了 [Thomas Dohmke 的 TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6)，内容关于 AI 如何降低编程门槛。大家对其目前的可靠性看法不一，但承认 UX 的改进允许更快地解决问题。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/tensorlake/status/1793693325180150146">来自 Tensorlake (@tensorlake) 的推文</a>: 我们非常激动地终于宣布 @tensorlake 的开源实时数据框架 Indexify。它适用于任何 LLM 技术栈，并为引入您的数据提供了基础构建块...</li><li><a href="https://huggingface.co/blog/clefourrier/llm-evaluation">聊聊 LLM 评估</a>: 未找到描述</li><li><a href="https://x.com/cupiabart/status/1793930355617259811?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Bartłomiej Cupiał (@CupiaBart) 的推文</a>: 这是我 CS 生涯中遇到的最奇怪的 bug。我和 @maciejwolczyk 一直在训练一个学习如何玩 NetHack 的神经网络，这是一个古老的 rog...</li><li><a href="https://x.com/jxnlco/status/1793800023689338921?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 jason liu (@jxnlco) 的推文</a>: 这是我对 RAG 未来走向的预测。在这个视频中，我谈到了 - RAG 从问答系统向报告生成工具的转变 - 精心设计的模板和 SOPs 的重要性...</li><li><a href="https://news.ycombinator.com/item?id=40458923">Show HN: 适用于 LLM 应用的开源实时数据框架 | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mark Riedl (@mark_riedl) 的推文</a>: 是的！我的网站投毒攻击对 Google 新的 LLM 驱动的 AI Overviews 生效了！</li><li><a href="https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6">有了 AI，现在任何人都可以成为程序员 | Thomas Dohmke | TED</a>: 如果你只需大声说话就能编程会怎样？GitHub CEO Thomas Dohmke 展示了由于 AI 的存在，编程的准入门槛正在迅速消失 —— 这是一个...</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?">来自 Mark Riedl (@mark_riedl) 的推文</a>: 是的！我的网站投毒攻击对 Google 新的 LLM 驱动的 AI Overviews 生效了！</li><li><a href="https://x.com/nathanlands/status/1793925460801581300?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Nathan Lands — Lore.com (@NathanLands) 的推文</a>: 11)
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1243325518292451490)** (1 messages): 

- **提供 World's Fair 多样性奖学金**: 难以负担 AI Engineer World's Fair 门票的人士可以申请多样性奖学金，该奖学金为 6 月 25 日至 27 日在旧金山举行的活动提供免费或折扣门票。申请应包括“对论文问题简明但具体的回答”，可以在[此处](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link)申请。

**提到的链接**: <a href="https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link">多样性计划 - AI Engineer World's Fair 2024 年 6 月</a>: AI Engineer World's Fair 致力于帮助想要参加我们活动的少数群体。我们坚定地相信让各种背景的人参加活动的价值。我们知道...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1243278784501645474)** (27 messages🔥): 

- **无需信用卡的税务发票**：Nathan Lambert 提到一个奇怪的情况，某平台在没有记录他信用卡的情况下给他发送了税务发票。在了解了关于转售证书（resale certificates）的细节后，他认为这个流程是合理的。
  
- **聚焦金门大桥的 AI**：小组对 [Anthropic AI 的实验](https://x.com/anthropicai/status/1793741051867615494?s=46)很感兴趣，该实验展示了通过改变 AI 的内部特征（internal features）使其专注于金门大桥。这促成了 "Golden Gate Claude" 的诞生，目前可在 claude.ai 进行公开互动。
  
- **Google 的公关惨败**：成员们讨论了 Google 的产品线问题似乎导致了反复的公开失败，例如反响不佳的 AI 发布。对话强调了对内部反馈未被采纳以及在推出不达标模型方面的疏忽的担忧。
  
- **对 AI 数据集声明的回应**：Philpax 分享的一个链接反驳了关于 Google AI 数据集的说法，特别是[否认依赖 LAION-5B](https://x.com/giffmana/status/1793906145310228538)。Google 的 AI 团队强调他们拥有更优越的内部数据集用于研究。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/anthropicai/status/1793741051867615494?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：本周，我们展示了改变 AI 模型 Claude 的内部“特征”如何改变其行为。我们发现了一个能让 Claude 极度关注金门大桥的特征。现在，为了...</li><li><a href="https://x.com/giffmana/status/1793906145310228538">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：以防万一这还不明显：这个答案是一个荒谬的幻觉。也许是因为“Google 的 AI 数据集”甚至不存在。我们没有接触 laion5b，甚至在研究中也没有。我们不需要，我...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1243582104923541555)** (2 messages): 

- **高级 CS 讲座幻灯片已发布**：Nathan Lambert 分享了一个基于 CS224N 材料的 CS25N 讲座更高级版本的链接。幻灯片可以点击[这里](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)访问。

- **未来录像公告**：Nathan Lambert 提到该环节的录像最终会发布。目前尚未提供具体的发布日期。

**提及的链接**：<a href="https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit">[2024年5月21日] Life after DPO (for alignment)</a>：Life after DPO Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS224N: Natural Language Processing with Deep Learning 2024年5月21日

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1243294013327413248)** (17 messages🔥): 

- **关于 cmdr 模型的 GQA 困惑**：成员们在澄清 "cmdr" 和 "cmdr+" 模型是否具有 **Grouped Query Attention (GQA)**。一位成员确认，“cmdr+ 有 GQA，非 + 版本没有”，展示了每个版本的不同规格。
- **VRAM 扩展讨论**：讨论了 **GQA** 的存在与否如何影响 VRAM 使用。一位用户提到，“GQA 比指数级好，但不是线性的……它只是扩展性更好。”
- **样本打包效率提升**：成员们强调了 GitHub 上的一个新 PR，指出“样本打包（sample packing）带来了 3-4% 的效率提升”。这[链接到了一个 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)，由 Dave Sescleifer 提交。

**提及的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619">切换到并行 FFD 箱打包算法。由 winglian 提交 · Pull Request #1619 · OpenAccess-AI-Collective/axolotl</a>：增加对分布式上下文打包的支持。重新加入打包效率估算。参见 @dsesclei 的 #1516。尝试将原始 PR 变基到最新的 main 分支并不太顺利。我...

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1243308936505852005)** (3 条消息): 

- **期刊论文发表**：一位成员分享了他们合著的一篇[期刊论文](https://doi.org/10.1093/jamia/ocae120)，该论文现已发表在《美国医学信息学协会杂志》（Journal of the American Medical Informatics Association）上。他们提到了自己所属的 **Université catholique de Louvain** 以及论文的其他贡献者。

- **祝贺声不断**：另一位成员对作者的发表表示祝贺，并附上了友好的“congrats 🙂”。这展示了社区对作者成就的支持和庆祝。

**提到的链接**：<a href="https://doi.org/10.1093/jamia/ocae120">Impact of high-quality, mixed-domain data on the performance of medical language models</a>：摘要/目标。为了优化用于医疗应用的 LLM 训练策略，重点是创建临床相关的系统。

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1243389475254566934)** (8 条消息🔥): 

- **SB-1047 引发公愤**：成员们讨论了对 SB-1047 的担忧，认为这是试图在 OpenAI 等大玩家之间实现 AI 治理中心化。一位成员称其为“异想天开、烂透了的垃圾”，并将其与大型制药公司和能源部门的监管俘虏（regulatory capture）相类比，认为这不利于预算紧张的小型开发者。 
- **分享 Perplexity AI 搜索链接**：一位成员分享了关于 SB-1047 的 [Perplexity AI 搜索](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A)链接。聊天中没有提供关于搜索细节的进一步信息或背景。
- **Arc Browser 的 Call Arc 功能获赞**：Arc Browser 的新功能“Call Arc”因其简单实用而受到关注。该成员称赞它允许用户毫不费力地“让浏览器为你查找并收集相关答案”，并分享了[更多细节链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD">1.44.1 Release</a>：</li><li><a href="https://g.co/gemini/share/a36c7ad84489">‎Gemini - SB 1047: Stifling Open-Source AI Innovation?</a>：由 Gemini Advanced 创建
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1243468923203227658)** (5 条消息): 

- **用户面临 Typer 安装问题**：一位用户表示 *"queuelabs: pip install typer does not resolve"*，表明他们在尝试使用 **pip** 安装 **Typer** 库时遇到了麻烦。
- **Poetry 设置问题困扰用户**：另一位用户询问 *"Did you run poetry install before poetry run 01? Are you running in a virtual environment,"* 指出了设置过程中可能遗漏的步骤。
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1243385149178118216)** (9 条消息🔥): 

- **Twinny + LM Studio 作为本地 co-pilot 表现惊人**：一位用户分享了将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 结合作为本地 co-pilot 替代方案的积极体验。他们询问了是否可以通过 llamafiles 运行此设置，并得到了确认：通过分配不同的端口，可以同时运行两个 llamafiles。

- **解决 llama.cpp 端点嵌入图像的困惑**：一名成员询问 llamafile/llama.cpp 服务器是否支持 llava 嵌入中的图像，并分享了一个未按预期工作的命令。随后他们澄清，`/v1/embeddings` 端点不接受 `image_data`，但使用 `/embedding` 端点可以按预期工作。

- **使用 llamafile 运行 continue.dev 的性能问题**：另一位用户报告了使用 llamafile 运行 continue.dev 的情况，指出在 Mac M2 上运行缓慢，但在较旧的 Nvidia GPU 上稍快一些。

- **关于构建和训练自定义 LLM 的咨询**：一名成员就使用公司文档构建和训练用于内部使用的自定义 LLM 寻求建议。他们收到了使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练的建议，并注意到 llamafile 仅支持推理（inference）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/index">🤗 Transformers</a>：未找到描述</li><li><a href="https://github.com/rjmacarthy/twinny">GitHub - rjmacarthy/twinny: 适用于 Visual Studio Code 的最简单直接、本地或 API 托管的 AI 代码补全插件 - 类似于 GitHub Copilot，但完全免费且 100% 私密。</a>：rjmacarthy/twinny</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4681">允许服务器通过 `/embedding` 端点生成多模态嵌入，由 kseth 提交 · Pull Request #4681 · ggerganov/llama.cpp</a>：服务器已经在 /completion 和其他地方提供了多模态支持，但在 /embedding 中还没有。对此的更改相对简单，如果用户向 /embe... 提交 image_data...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1243302339301871668)** (8 条消息🔥): 

- **用户感谢团队**：针对之前的互动表达了 “*THANK YOU!*”。
  
- **关于 104B 模型的咨询**：一位用户询问团队是否计划发布其模型系列的 **104B 版本**。

- **Langchain 集成问题**：一名成员询问了关于 Cohere 使用 **Langchain 集成** 的当前状态和建议。

- **Aya 模型尺寸说明**：一位用户询问 Playground 上的 **Aya 模型** 是 8B 还是 35B 版本。

- **Compressor 的验证错误**：分享了一个关于 **ContextualCompressionRetriever** 因抽象方法导致 `ValidationError` 的问题。

- **“56 根香蕉等于 1 个苹果”的计算**：使用 **CMR+** 探讨了一个计算问题：*“1 个苹果 = 2 个梨，3 个梨 = 4 个橙子，6 个橙子 = 7 根香蕉”*，结论是 “56 根香蕉等于 1 个苹果”。

- **403 Forbidden 错误排查**：一位用户报告称，尽管使用了正确的生产密钥，仍出现 **403 Forbidden 错误**。
  

---

### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1243425073017131112)** (6 messages): 

- **AI 生成的 Standup comedy 质量出奇地好**：一位用户分享了一个链接，对 AI 生成的 Standup comedy 的质量表示惊讶。他们似乎对其表现印象深刻。

- **探索 Ud.io 应用**：另一位用户询问提到的应用 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 是否只做喜剧。这一询问表明了对该应用完整功能的后续好奇。

- **在 Suno 上转换音频**：一位成员使用 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 分享了原始音频的一个更“魔性（demonic）”的版本。这突显了该平台在修改声音方面的多功能性。

- **对学习音频处理（Audio Manipulation）感兴趣**：一位用户表示有兴趣学习如何创建类似于分享的音频修改。这表明了获取音频工程或 AI 驱动的声音处理技能的愿望。

- **简短回应**：简而言之，一位用户对一个查询回应了一个简短的“No”，表示不感兴趣或否定之前的陈述。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG">csimpkins - Standup Comedy on AI Generated Music | Udio</a>：在 Udio 上收听 csimpkins 的 AI 生成音乐 Standup comedy。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655">AI Standup Comedy on AI Generated Musicby by @unwaveringplugin464 | Suno</a>：脱口秀演员在喜剧节目中表演的歌曲。收听并使用 Suno 制作你自己的作品。
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1243281449352691742)** (1 messages): 

- **成员寻求 Google Calendar 集成以进行活动追踪**：一位成员询问是否可以将活动日历导入 Google Calendar，以避免错过活动。他们用一个悲伤的表情符号表达了担忧，表明需要一种更高效的方式来追踪预定的活动。
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

evelynciara: 是的，我很高兴这个频道存在 😅
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

datarevised: https://x.com/DataPlusEngine/status/1793803117642854732
  

---


{% else %}

# 第 2 部分


{% endif %}

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning 事实**：在 [general 频道](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) 的讨论中，揭示了由于偏置的数据类别导致的 **语义相似度过拟合 (semantic similarity overfitting)** 问题。一位用户在理解 Fine-tuning 与用户输入及初始模型训练的关系时遇到了困难。此外，还注意到 **OpenAI 平台侧边栏** 的变化，两个图标（线程和消息）消失了。

**模板成为焦点**：在 [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123) 中，强调了在 Fine-tuning 期间正确配置模板的重要性。特别是分隔符 `###` 有助于解析不同的输入部分，而 "end of text" token 则指示何时停止 token 生成。

**Maven 与交流**：在 [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698) 中，成员们进行了一次轻松的交流，提到了重聚。一份会议演讲录像的请求得到了满足，视频已在 Maven 上发布。

**Modal 动员**：[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) 的 Modal 用户分享了收到额度的兴奋感、训练经验，并为新用户提供了 **Modal 文档** 和 **示例** 的具体链接。还分享了一个使用 Modal 参加 **Kaggle 竞赛** 的计划，包括设置和执行细节。

**Jarvis 记录 Jupyter 杂记**：在 [jarvis-labs 频道](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229) 中，成员们讨论了在 Jarvis 上存储 VSCode 仓库，并建议使用 GitHub 保存工作。有一则关于由于不稳定而 **移除竞价实例 (spot instance)** 的通知。分享了 Fine-tuning **open-lama-3b** 模型的成本和时长，一位用户通过调整模型参数解决了 Ampere 系列错误。

**Hugging Face 讨论额度与西班牙语模型**：[hugging-face 频道](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) 讨论了待处理的 **HF 额度** 以及适用于西班牙语文本生成的模型——推荐了 **Mistral 7B** 和 **Llama 3** 模型。

**额度倒计时继续**：在 [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150) 频道，预告了即将发布的关于额度管理和分配的公告。

**Corbitt 的诫命大显身手**：[kylecorbitt_prompt_to_model 频道](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) 中热情的与会者讨论了 Kyle Corbitt 演讲中介绍的 Fine-tuning 方法和技术，包括 *[部署 Fine-tuned 模型的十诫 (Ten Commandments for Deploying Fine-Tuned Models)](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*。

**Axolotl 响应号召**：在 [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817) 中，用户讨论了 Axolotl 中的数据集、模型训练和故障排除。分享了一篇关于 **TinyLLama Fine-Tuning** 的博客文章，并推动将可观测性 (observability) 集成到 LLM 应用中。

**退出 Zoom，进入 Discord**：在 Zoom 聊天被禁用后，来自 [workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) 的用户将讨论转移到了 Discord。

**Axolotl 的缓存难题引发困惑**：在 [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664) 中，Axolotl 的缓存问题令用户沮丧，文件丢失的困惑已得到解决。关于样本打包 (sample packing) 的讨论和一份关于 Tokenizer 陷阱的指南解决了有关效率和分词的疑虑。

**加速迈向胜利**：[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) 的用户解决了对浮点数比较的困惑，修复了 Jarvislab 训练命令错误，并交流了学习模型加速的资源，重点关注 Fine-tuning 的最佳实践。

**与 Axolotl 一起尝试**：[wing-axolotl 频道](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) 协作处理了数据集模板、预处理问题、Axolotl 配置，并提供了最新 Axolotl 更新的 PR 合并。他们深入研究了调试工具以及精确模板对训练成功的重要性。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**蛋白质数据可视化达到新高度**：一个新的蛋白质可视化项目现在支持 3D 渲染，并包含了人类血红蛋白和核糖体蛋白的示例，项目详情可以在 [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md) 上找到。

**使用 OpenAI 的 Whisper 进入 TranscriptZone**：一款利用 OpenAI 的 Whisper 来转录 YouTube 视频及更多内容的新转录应用已在 [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext) 上线。

**去中心化网络——不仅仅是一个梦想？**：一个为去中心化互联网构建基础设施的项目通过调查寻求社区反馈，引发了关于数据收集伦理的讨论。

**Vision Transformers 深度查询**：一位成员寻求关于应用 Vision Transformers (ViT) 进行单目深度估计（monocular depth estimation）的资源，表示有意开发一个使用 ViT 的模型，但讨论中未提供具体资源。

**Mistral 模型的量化困境**：在 **Mistral v0.3 Instruct** 上使用 **bitsandbytes** 进行 8-bit 量化导致性能比 4-bit 和 fp16 更慢，这一令人困惑的结果与减少位数计算预期的效率提升相矛盾。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在 CSV 对决中超越 ChatGPT**：工程师们讨论认为 **Perplexity AI** 在 CSV 文件处理方面优于 **ChatGPT**，因为它允许直接上传 CSV。此外，推荐使用 **Julius AI** 进行数据分析，它利用 Python 并集成了 **Claude 3** 或 **GPT-4** 等 LLM。

- **用户冷落 Claude 3 Opus**：由于内容限制增加和感知到的实用性下降，**Claude 3 Opus** 遭到冷遇，尽管 **GPT-4** 也有局限性，但仍被视为更好的选择。

- **质疑 Pro Search 的真正升级**：**Pro Search** 的升级引起了关注，用户讨论新的多步推理功能和 API 规范究竟是真正的后端改进，还是仅仅是表面上的 UI 增强。

- **API 集成详解**：围绕外部工具与 **Claude** 的 API 集成的对话引起了兴趣，同时分享了自定义函数调用、无服务器后端以及诸如 [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use) 等文档。

- **AI 伦理：不仅仅是一个思想实验**：关于为 GPT 注入伦理监控能力的讨论被激发，揭示了其在职场沟通和法律辩护方面的潜在应用，尽管哲学上的难题尚待解决。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **关于 RTX 5090 显存的猜测达到顶峰**：关于传闻中拥有 **32GB VRAM 的 RTX 5090** 是否具有实际意义的辩论正热。参考了 [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/) 上的潜在规格和图片，但一些成员对其真实性持怀疑态度。

- **Stable Diffusion 与 AMD 的挑战**：用户提供了在 AMD 5700XT GPU 上安装 **Stable Diffusion** 的指导，建议从 [Craiyon](https://www.craiyon.com/) 等 Web 服务开始，以规避潜在的兼容性问题。

- **Stable Diffusion 3：承诺前的试用**：社区将 **Stable Diffusion 3** 与竞争对手 Midjourney 进行了对比，强调虽然 SD3 提供免费试用，但持续访问需要 **Stability** 会员资格。

- **对 Mobius 模型的期待升温**：关于 DataPlusEngine 的新型 **Mobius 模型** 的公告引起了极大关注，因为它声称可以创建高效的基础模型。该模型在 [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732) 上进行了预告，它既不是简单的基础模型，也不是现有模型的微调版本。

- **32GB VRAM：游戏规则改变者还是性能过剩？**：提到 32GB VRAM GPU 引发了关于 Nvidia 数据中心 GPU 销售策略潜在转变的对话，考虑到拥有大容量显存的产品可能会如何影响市场对 H100/A100 系列的需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT 配置问题已解决**：针对 PEFT 训练期间缺失 `config.json` 的问题，通过从基础模型的配置中复制该文件已得到解决，用户已确认成功。

- **Llama 跨越 Bug 障碍**：**Llama 3** 模型的基础权重被描述为存在 "bug"，但 Unsloth 已实现相关修复。为了提升训练效果，建议使用保留 token 并更新 tokenizer 和 `lm_head`。

- **System Prompt 提升 Llama 3 效果**：观察发现，加入系统提示词（System Prompt），即使是空白的，也能增强 Llama 3 的微调（finetuning）结果。

- **Phi 3 模型激增**：随着 **Phi 3 模型** 的首次亮相，社区反响热烈，该模型已支持 medium 版本。社区讨论引导工程师关注博客文章和发布说明中的详尽细节。

- **Stable Diffusion 的诡异一面**：**Stable Diffusion** 产生的诡异伪影和离奇的语音克隆输出令用户感到吃惊，相关讨论和经历已在 YouTube 视频和 Reddit 帖子中分享。

- **VSCode Copilot 方案推荐**：用户在 **random** 频道寻求本地 VSCode "copilot" 的建议，并得到了积极的响应和推荐。

- **Phi-3 的推理延迟问题**：一名用户对使用 **Unsloth Phi-3** 时较慢的推理速度感到困惑，并提供了一个 [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) 用于调查延迟原因，社区目前仍在努力寻找修复方案。

- **量化困境的破解**：一名成员在量化自定义模型时面临挑战，在 **llama.cpp** 和 **Docker** 兼容性方面遇到了障碍，引发了关于解决方案的讨论。

- **模型性能的 VRAM 判定**：明确了 VRAM 需求：**Phi 3 mini 需要 12GB** 即可，但 **Phi 3 medium 必须配备 16GB**。对于繁重任务，建议考虑外部计算资源。

- **训练一致性的数据尽职调查**：强调了在训练和评估中使用一致数据集的重要性，并重点介绍了 **Unslothai 的公共数据集**，如 [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a)。

- **平台可能性与警示**：针对 **Unsloth** 是否支持旧款 Mac 的咨询得到了回复，确认目前重点在于 CUDA 和 GPU 的使用，并为仅有 CPU 的设备提供了建议。

- **企业级专业知识扩展**：一名社区成员主动提出为 Unsloth 提供企业级专业知识，并对加入 Build Club 和 GitHub 的加速器表示赞赏，暗示了 Unsloth 未来发展的协同潜力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**关于 AI 理解能力的智力辩论**：社区就 LLM 对概念的真实**理解**展开了深入讨论，**可解释性研究（interpretability research）**被视为重要的经验证据。怀疑论者认为目前的努力尚不足够，并引用了 **Anthropic** 关于映射大语言模型思维的相关工作。

**Llama 湖中的生物**：一项旨在增强 **Llama 模型** 的技术尝试集中在编写一个能够管理**函数调用（function calls）**的脚本上，并以 **Hermes Pro 2** 的方法作为灵感。另一项咨询则围绕在 3080 GPU 上实现 **Llama3 LoRA** 技术展开。

**数字维度中的现实探索**：在关于 **Nous 和 WorldSim** 的对话中，成员们探讨了 **NightCafe** 和多维 AR 空间在映射复杂 AI 世界中的潜在应用。**音频可视化器**中的梦幻探索和奇特的 **ASCII 艺术**表现形式突显了 AI 驱动模拟的创意用途。

**筛选 RAG 数据**：提倡模型将**内部知识**与**检索增强生成（RAG）**相结合成为热门话题，并就如何处理矛盾和解决冲突提出了疑问。强调用户评估被认为是必不可少的，特别是对于复杂的查询案例。

**微调 AI：精准度胜过花哨技巧**：社区讨论赞扬了 **Mobius 模型** 在**图像生成**方面的卓越表现，并期待其开源版本和阐释性论文的发布。此外，还提到了 Hugging Face 的 `PyTorchModelHubMixin` 可以简化模型共享，但受限于 **50GB 的大小限制**（在不分片的情况下）。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA：TPU 对决**：TPU 上 **JAX** 和 **PyTorch/XLA** 的性能对比引发了关于基准测试细微差别的辩论，例如 **warmup times** 和 **blocking factors**。GPT-3 的训练成本从 **450 万美元大幅下降到 2024 年预计的 12.5 万至 100 万美元**，这一点受到了关注，其中考虑了来自不同贡献者的 **TFLOP 速率** 和 **GPU-hour 定价**，并链接到一篇 [Databricks 博客文章](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8)。

- **扩展与教学 LLMs**：在研究论坛中，**Chameleon** 模型因其在多模态任务中的强劲表现而受到关注，而 **Bitune** 则承诺改进 LLM 的 zero-shot 性能（[Bitune 论文](https://arxiv.org/pdf/2405.14862)）。讨论质疑了 **JEPA** 模型对 AGI 的可扩展性，并批评了 **RoPE** 的上下文长度限制，引用了一篇相关的 [论文](https://arxiv.org/pdf/2405.14591)。

- **涌现特征困扰 LLM 爱好者**：链接了 Tim Dettmers 关于在 Transformer 推理中保持性能的高级量化方法的研究，包括他的涌现离群值（emergent outliers）概念，以及通过 [bitsandbytes library](https://huggingface.co/blog/hf-bitsandbytes-integration) 与 Hugging Face 的集成。关于涌现特征（emergent features）的讨论围绕着它们是模型的“DNA”这一观点展开，引发了对其对相变（phase transitions）影响的讨论。

- **技术调整与 LM 评估简报**：在 **lm-thunderdome** 中，工程师们介绍了在 **vllm models** 中设置 seed 的实用技巧，使用 `lm_eval --tasks list` 获取 **任务列表**，以及处理 **BigBench** 任务名称更改（这会影响像 Accelerate 这样存在内存问题的 harness）。建议通过查阅 `lm-eval/tasks` 文件夹来定位任务，以便更好地组织。

- **协作呼吁**：发出了扩大 **Open Empathic** 项目的呼吁，并提供了一个用于贡献电影场景的 [YouTube 指南](https://youtu.be/GZqYr8_Q7DE) 以及该项目的链接。鼓励进一步的协作，强调了社区努力进行增强的必要性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU 历险记**：工程师们讨论了将小模型加载到 GPU 上的挑战，一些人青睐 *llama3, mistral instruct* 和 *cmdrib* 等模型。同时，据报道，在某些应用中，使用较低的量化（如 *llamas q4*）比 q8 等较高的量化产生更好的结果，反驳了“越大越好”的观念。

**下一代模型即将到来**：模型领域的一项更新通知了 **35B 模型** 的发布，并进行了测试以确保 LM Studio 的兼容性。针对不同规模模型的优化也是一个话题，重点是 **Phi-3 small GGUFs** 及其效率。

**服务器与设置**：硬件讨论包括利用 **llama.cpp** 及其最近的 RPC 更新进行 **distributed inference**，尽管目前还不支持量化模型。还探索了使用配备 **RTX 4060 Ti 16GB** 的廉价 PC 集群进行分布式模型设置的实验性构建，以及可能的网络限制。

**实现多语言凝聚**：Cohere 模型现在将其能力扩展到了 **23 种语言**，正如广告所言，**aya-23 quants** 已开放下载，但 ROCm 用户必须等待更新才能体验。

**Stable Diffusion 被排除在外**：LM Studio 澄清说，它专门处理语言模型，不包括像 Stable Diffusion 这样的图像生成器，同时处理旧 GPU 上的 CUDA 问题，并推广 **Julius AI** 等服务以缓解用户体验方面的困扰。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **梯度范数（Gradient Norm）麻烦**：将 batch size 从 32 修改会导致梯度范数突然飙升，从而中断训练。一个 [pull request](https://github.com/karpathy/llm.c/pull/456) 通过防止 fused classifier 中的索引溢出解决了这个问题。
  
- **Int4 和 Uint4 类型需要关注**：一位成员指出 PyTorch 中许多函数缺乏对 **int4** 和 **uint4** 数据类型的实现，相关的 [讨论帖](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) 指出了在类型提升（type promotion）和 tensor 操作方面的局限性。

- **直播代码预警——聚焦 Scan 算法**：Izzat El Hajj 将主持一场关于 Scan 算法的现场编程会议，该算法对于像 Mamba 这样的 ML 算法至关重要。会议定于 `<t:1716663600:F>`，有望为爱好者们带来一次技术深度探讨。

- **CUB 库查询与 CUDA 细节**：成员们深入讨论了从 CUDA CUB 库代码的运行机制到在不使用 cuBLAS 或 cuDNN 的情况下触发 tensor cores 等话题，并重点推荐了 [NVIDIA 的 CUTLASS GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main) 和 [NVIDIA PTX 手册](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 等资源。

- **FineWeb 数据集难题**：处理 FineWeb 数据集非常占用存储空间，磁盘占用达到 70 GB，并消耗高达 64 GB 的 RAM，这暗示在数据处理任务中需要更好的优化或更强大的硬件配置。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python 库比起 Mojo 更倾向于 C**：关于将 Python 库移植到 Mojo 的可行性和准备工作有一场激烈的讨论，考虑到 Mojo 不断演进的 API，人们担心给维护者施加太大压力。成员们讨论了将目标对准 C 库是否是一个更直接且实际的尝试。

**Rust 的安全性吸引力不会削弱 Mojo 的潜力**：Mojo 并不打算取代 C，但 Rust 的安全性优势正在影响工程师们对 Mojo 在不同场景下应用的思考。正在进行的讨论涉及了可以使 Mojo 开发受益的 Rust 概念。

**使用 Nightly 版本 Mojo 奋力前行**：在 MacOS 上使用 Mojo 的 Nightly 版本运行 BlazeSeq 的性能表现出与 Rust 的 Needletail 相似的潜力，引发了关于跨平台效率的讨论。快速的 Nightly 更新（见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)）让社区保持对这门演进中语言的关注。

**对 Modular 机器人机制的好奇**：有人对 "ModularBot" 的底层技术提出了疑问，虽然没有提到具体模型，但机器人给出了一个生动的回复。另外，还讨论了在 Mojo 中进行 ML 模型训练和推理的潜力，并提到 Max Engine 可以作为 numpy 的替代方案，尽管目前还没有完善的训练框架。

**编译时困惑与对齐问题**：从内存中 boolean 值的对齐问题到编译时函数问题，都引起了用户的关注，解决方法和官方 [bug reports](https://github.com/modularml/mojo/issues/2813) 凸显了社区驱动排错的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **忠于 LaTeX 的 LLM**：在格式化领域，用户对 GPT 表现出强烈的默认使用 LaTeX 的倾向感到沮丧，即使要求提供 Typst 代码也是如此，这揭示了 LLM 似乎坚持某种特定的编码语法偏好。
  
- **Microsoft Copilot+ 与 Leonardo 之争**：社区讨论集中在 Microsoft Copilot+ PC 在“草图转图像”等创意任务中的价值，而一些成员则鼓励尝试 [Leonardo.ai](https://leonardo.ai) 以获得类似的功能。

- **对 AI 效率的渴求**：人们对 AI 造成的环境代价表示担忧，引用了 [Gizmodo 的一篇文章](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249)，该文指出 AI 模型训练过程中耗水量巨大，引发了关于需要更环保的 AI 实践的讨论。

- **迭代优于创新**：关于通过迭代优化来增强 LLM 性能的对话非常活跃，并提到了像 AutoGPT 这样处理迭代的项目，尽管这伴随着更高的成本。

- **智能注入的报价是否言过其实？**：公会思考了在 ChatGPT 中嵌入法律知识的可行性和潜力，其价值甚至被认为达到 6.5 亿美元，不过关于这一大胆断言的详细观点较少。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent 深度解析**：工程师们探讨了在 **SequentialChain** 中使用 **LangChain's CSV agent**，并讨论了[如何自定义输出键](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)（如 `csv_response`）。提到了 SQL agent 在处理多表查询时面临的挑战，指出存在 Token 限制和 LLM 兼容性问题，并引导至 GitHub [提交 issue](https://github.com/langchain-ai/langchain/issues)。

**AI 展示引发热议**：[OranAITech 在推特上](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)展示了他们最新的 AI 技术，同时 **everything-ai v2.0.0** 发布了包含音视频处理功能的新特性，并提供了 [仓库](https://github.com/AstraBert/everything-ai) 和 [文档](https://astrabert.github.io/everything-ai/)。

**揭秘 VisualAgents**：YouTube 上分享了 **Visual Agents 平台** 的演示，展示了其利用 LangChain 的能力，在无需编码的情况下简化 SQL agent 创建和构建简单检索系统的潜力。两个具体视频展示了其工作流：[SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) 和 [简单检索](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM)。

**EDA GPT 印象展示**：[LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01) 链接了一个 **EDA GPT** 的演示，包含一段五分钟的概览视频，展示了其各项功能。该演示突显了这款 AI 工具的多功能性。

**教程预告**：tutorials 频道的一条消息提供了 business24.ai 内容的 [YouTube 链接](https://youtu.be/gflsu_6R_8g)，尽管其相关背景尚未披露。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **盗版并非万灵药**：尽管有人幽默地建议 The Pirate Bay 可以成为分享 AI 模型权重的避风港，但成员们对此表示怀疑，并强调其他国家更友好的 AI 政策环境可能会脱颖而出。

- **日本在 AI 领域采取积极态度**：参与者注意到日本对 AI 发展的鼓励立场，并引用了一篇通过 [推文](https://x.com/DataPlusEngine/status/1793817514956259460) 分享的 **论文**，该论文关于在无需大量预训练的情况下创建新的基础 Diffusion 模型，展示了一种涉及暂时破坏模型关联的策略。

- **中毒恢复协议探讨**：提到了一项由 fal.ai 开展的关于中毒模型恢复方法的 **合作研究**，预计研究结果将为恢复方法提供实证支持。此外，成员们对 AI 生成图像的美学表达了保留意见，特别是 Mobius 等模型与 MJv6 等前作相比所呈现的“高对比度外观”和伪影。

- **Claude 映射破解代码**：Anthropic 的 **研究论文** 详细剖析了 Claude 3 Sonnet 的神经图谱，阐述了对概念激活（conceptual activations）的操作，可在其 [研究页面](https://www.anthropic.com/research/mapping-mind-language-model) 阅读。关于此类激活可能商业化的讨论引发了争论，同时也存在对商业影响导致 AI 从业者受挫的担忧。

- **怀旧 AI 视觉愿景**：一位成员回忆了从早期 AI 视觉模型（如 Inception v1）到如今复杂系统的演变，认可了 DeepDream 在理解神经功能方面的作用。此外，还讨论了神经网络中稀疏性的好处，描述了使用 L1 范数实现稀疏性，以及在高维层中典型的 300 个非零维度。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **聚会提醒：名额有限**：即将于周二举行的 **LlamaIndex meetup** 仅剩少量名额，由于名额有限，鼓励爱好者们尽快[预订位置](https://twitter.com/llama_index/status/1793739449127583964)。

- **MultiOn 结合 LlamaIndex 实现任务自动化**：**LlamaIndex** 已与 AI Agent 平台 **MultiOn** 结合，通过代表用户操作的 Chrome 浏览器实现任务自动化；在此查看 [demo](https://twitter.com/llama_index/status/1793764970024570979)。

- **RAGApp 发布，支持无代码 RAG 聊天机器人设置**：新推出的 **RAGApp** 简化了通过 Docker 容器部署 RAG 聊天机器人的过程，使其可以轻松部署在任何云基础设施上，并且它是开源的；在此配置你的模型提供商 [model provider](https://twitter.com/llama_index/status/1794030544415818062)。

- **解决 PDF 解析难题**：社区认可 **LlamaParse** 作为从 PDF（特别是表格和字段）中提取数据的可行 API，利用 GPT-4o 模型提升性能；**Knowledge Graph Indexing** 的挑战也是讨论话题，强调了手动和自动（通过 `VectorStoreIndex`）策略的必要性。

- **PostgresML 与 LlamaIndex 联手**：**Andy Singal** 分享了将 **PostgresML** 与 **LlamaIndex** 集成的见解，并在 Medium 文章 ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) 中详细介绍了这一协作，获得了社区的积极评价。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct 发布**：OpenRouter 推出了 **Phi-3 Medium 128k Instruct**，这是一个强大的 140 亿参数模型，并邀请用户查看[标准版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct)和[免费版](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free)变体，并参与其效果讨论。

- **Wizard 模型获得魔力提升**：**Wizard 模型** 表现出改进，响应更加迅速且富有想象力，但仍需注意避免重复段落。

- **关注 Phi-3 Vision 和 CogVLM2**：围绕 **Phi-3 Vision** 的热情高涨，分享了如 [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml) 的测试链接，并建议在 [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) 中使用 **CogVLM2** 处理以视觉为中心的任务。

- **Llama 3 Prompt 自动转换**：澄清了发往 **Llama 3** 模型的 Prompt 会通过 OpenRouter 的 API 自动转换，从而简化流程，但手动 Prompt 仍作为一种替代方案保留。

- **Gemini API 的烦恼**：用户报告了 **Gemini FLASH** API 的问题，如空输出和 Token 消耗，这被认为是模型本身的问题。Google 每日 API 使用限制的出现引起了人们对这可能如何影响 OpenRouter 的 Gemini 集成的关注。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Indexify 引发关注**：Tensorlake 推出的开源实时数据框架 [Indexify](https://github.com/tensorlakeai/indexify) 引发了讨论，重点在于其“streaming ETL”能力以及创建可持续开源模型的挑战。人们对所提供的 extractor 的充分性及其潜在的变现路径表示担忧。

- **LLM 评估备受瞩目**：一篇关于 Large Language Model (LLM) 评估实践、排行榜重要性以及严谨的回归测试（non-regression testing）的 [Hugging Face 博客文章](https://huggingface.co/blog/clefourrier/llm-evaluation) 引起了成员们的注意，强调了此类评估在 AI 发展中的关键作用。

- **AI 对搜索引擎操纵的回应**：一起涉及网站中毒并影响 Google AI 汇总概览（AI-gathered overviews）的事件引发了关于安全和数据完整性的讨论，包括 [Mark Riedl 的推文](https://x.com/mark_riedl/status/1793375699967054334) 中提到的通过自定义搜索引擎浏览器绕过（bypass）的解决方法。

- **AI 是在民主化开发还是引发可靠性疑问？**：GitHub CEO Thomas Dohmke 关于 AI 在简化编程中作用的 [TED 演讲](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) 引发了对其可靠性的辩论，尽管 AI 驱动的 UX 改进加快了编程过程中的问题解决速度。

- **多元化奖学金助力弥合差距**：面对参加即将举行的 AI Engineer World's Fair 财务障碍的多元化背景工程师收到了多元化奖学金发布的助力。有兴趣的申请人应在 [申请表](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link) 中对论文问题提供*简洁*的回答。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **无需信用卡的税务故事**：Nathan Lambert 破解了一场发票纠纷，意识到了由于转售证书（resale certificates）而在没有信用卡的情况下进行税务计费的合理性。

- **金门大桥 AI 引起关注**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 的实验诞生了“Golden Gate Claude”，这是一个一心一意针对金门大桥训练的 AI，因其在 claude.ai 上的公开互动性而引发热议。

- **Google 的 AI 失误**：Google 未能利用反馈以及过早部署 AI 模型，引发了关于这家科技巨头公关挑战和产品开发困境的讨论。

- **反击数据集误解**：Google 的 AI 团队反驳了关于使用 LAION-5B 数据集的说法，提出他们使用的是更优越的内部数据集，正如[最近的一条推文](https://x.com/giffmana/status/1793906145310228538)所引用的那样。

- **Nathan 分享知识锦囊**：为 AI 爱好者，Nathan Lambert 上传了高级 [CS224N 课程讲义](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit)。此外，与会者还收到了关于即将发布的会议录像的提示，但尚未公布发布日期详情。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA 在 CMDR 模型中受到关注**：讨论显示 **Grouped Query Attention (GQA)** 存在于 “cmdr+” 模型中，但不存在于基础 “cmdr” 模型中，这表明了它们规格上的重要区别。
- **VRAM 效率与智能注意力机制**：工程师们指出，虽然 **GQA** 不提供线性缩放，但与指数缩放相比，它代表了一种改进的缩放方法，对 **VRAM** 使用产生了有利影响。
- **Sample Packing 获得提升**：一个新的 **GitHub pull request** 展示了在 sample packing 方面 3-4% 的效率提升，有望为分布式环境提供更好的资源管理，链接见[此处](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619)。
- **学术成就获得认可**：一名成员合著的期刊文章已在 **Journal of the American Medical Informatics Association** 上发表，强调了高质量、跨领域数据对医学语言模型的影响，文章可见[此处](https://doi.org/10.1093/jamia/ocae120)。
- **社区庆祝学术成功**：社区通过个人祝贺信息表达了对同行发表作品的支持，培养了 AI 领域内认可学术贡献的文化。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 引发技术动荡**：工程师们对 **SB-1047** 法案的影响表示深切担忧，认为其不利于小型 AI 参与者，并将这种情况比作在其他行业观察到的“监管俘获”（regulatory capture）。

**Perplexity 和 Arc，行业工具展示**：社区重点展示了辅助工作流的工具，分享了一个关于 [SB-1047 的 Perplexity AI 搜索结果](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A) 以及 Arc Browser 的新功能 “Call Arc”，该功能简化了在线查找相关答案的过程，并附带了[信息链接](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD)。

**安装问题引发咨询**：用户在使用 pip 安装 **Typer** 库时遇到问题，引发了关于是否遵循了设置步骤（如在 `poetry run` 之前执行 `poetry install`）或是否使用了虚拟环境的讨论。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny 作为虚拟副驾驶起飞**：开发者们正在将 [Twinny](https://github.com/rjmacarthy/twinny) 与 LM Studio 集成，将其作为强大的本地 AI 代码补全工具，并支持在不同端口上运行多个 llamafile。

**嵌入端点（Embedding Endpoint）详解**：澄清了 `/v1/embeddings` 端点不支持 `image_data`；根据 [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681)，图像应使用 `/embedding` 端点。

**Mac M2 在 continue.dev 中遇到对手**：一项性能观察指出，在使用 llamafile 执行时，continue.dev 在 Mac M2 上的运行速度比旧款 Nvidia GPU 慢。

**训练你自己的 LLM**：对于那些希望构建和训练自定义 LLM 的用户，社区推荐使用 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 进行训练，并提醒 llamafile 是为推理而非训练设计的。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **服务器中回荡着感激之情**：一位用户对团队表达了由衷的*感谢*，展示了用户对团队支持或开发工作的认可。
- **对扩展模型的关注**：有传言称该模型是否会加入 **104B 版本**，但目前尚未有明确的答复。
- **Langchain 集成缺失**：出现了关于 **Langchain** 与 Cohere 集成的问题，用户正在寻求关于其当前可用性和实现状态的指导。
- **模型尺寸之谜**：用户正在寻求澄清 Playground 中的 **Aya model** 是指 8B 还是 35B 版本，这表明理解模型规模对于应用的重要性。
- **错误排查角落**：诸如 **ContextualCompressionRetriever** 的 `ValidationError` 和 **403 Forbidden error** 等问题标志着工程师们正在进行活跃的调试和技术问题解决，这提醒了 AI 开发中的常见挑战。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 喜剧之夜恰到好处**：用户分享的一段 AI 生成的单口喜剧作品获得了积极的反响，显示了 AI 在模仿幽默和进行娱乐表演方面的进步。

**关于 AI 应用的探索性查询**：从用户询问 [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG) 的功能是否超出生成喜剧的范围来看，对其功能边界的好奇显而易见。

**声音变换展示**：一位用户通过分享一段原始音频的变体、恶魔化版本，展示了 [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) 灵活的音频修改功能。

**对音频工程知识的渴望**：用户表达了对获取制作演示中音频修改技能的兴趣，这对于对声音处理感兴趣的 AI 工程师来说是一项宝贵的技能。

**偏好简洁的沟通**：对一个问题的单字回答“No”凸显了对简洁回复的偏好，这或许反映了工程师对直接、务实沟通的追求。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻找统一的事件追踪器**：一位成员强调了对兼容 Google Calendar 的活动日历的迫切需求，以确保不会错过任何社区活动。缺乏这样一个系统是社区内一个值得关注的问题。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **新数据集发布公告**：用户 datarevised 引用了一个新数据集，并提供了详细信息的链接：[DataPlusEngine 推文](https://x.com/DataPlusEngine/status/1793803117642854732)。



> 完整的各频道详细分析已针对邮件进行截断。 
> 
> 如果您想查看完整的详细分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

