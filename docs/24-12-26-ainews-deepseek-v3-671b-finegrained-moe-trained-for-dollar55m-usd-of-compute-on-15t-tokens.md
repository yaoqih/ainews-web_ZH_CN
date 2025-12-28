---
companies:
- deepseek-ai
- hugging-face
- openai
- anthropic
date: '2024-12-27T01:18:46.567338Z'
description: '**DeepSeek-V3** 正式发布，该模型拥有 **6710 亿 (671B) MoE 参数**，并在 **14.8 万亿 (14.8T)
  token** 上进行了训练，其基准测试表现超越了 **GPT-4o** 和 **Claude-3.5-sonnet**。


  该模型的训练仅耗费了 **278.8 万 H800 GPU 小时**，远低于 **Llama-3** 的 **3080 万 GPU 小时**，展现了极高的计算效率和大幅的成本降低。该模型目前已开源，通过
  **Hugging Face** 部署并提供 API 支持。


  其技术创新包括：原生 FP8 混合精度训练、多头潜在注意力（MLA）扩展、合成推理数据蒸馏、针对多达 **256 个专家**的 MoE 剪枝与修复，以及一种支持前瞻性
  token 规划的新型多 token 预测（MTP）目标。研究亮点还涵盖了用于多步推理和智能体控制的 **OREO 方法**和**自然语言强化学习 (NLRL)**。'
id: bdb257be-bb5c-4a2e-a142-55ffb972fa94
models:
- deepseek-v3
- gpt-4o
- claude-3.5-sonnet
- llama-3
original_slug: ainews-deepseek-v3-671b-finegrained-moe-trained
people:
- nrehiew_
- denny_zhou
title: DeepSeek v3：671B（6710亿）参数的细粒度混合专家模型（MoE），在 15T（15万亿）token 上训练而成，算力成本仅为 550
  万美元。
topics:
- mixture-of-experts
- model-training
- model-optimization
- reinforcement-learning
- chain-of-thought
- multi-token-prediction
- synthetic-data
- model-distillation
- fine-tuning
- attention-mechanisms
- gpu-optimization
---

<!-- buttondown-editor-mode: plaintext -->**算法、框架与硬件的全方位协同设计（Full co-design）就是你所需要的一切。**

> 2024年12月25日至12月26日的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、[**433 个 Twitter 账号**](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord 社区（**215** 个频道，**5486** 条消息）。预计节省阅读时间（按 200wpm 计算）：**548 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！


![image.png](https://assets.buttondown.email/images/648e425e-67fd-4c0c-9d98-1220180a17b9.png?w=960&fit=max)


正如在[圣诞假期期间预告](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)的那样，DeepSeek v3 已经发布（我们之前对 [DeepSeek v2 的报道在此](https://buttondown.com/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/)）。其 Benchmark 表现一如既往地符合你对中国顶尖开源模型实验室的预期：


![image.png](https://assets.buttondown.email/images/06c135d1-dbb9-48be-8af9-539a8b3a14da.png?w=960&fit=max)


（更多细节见 [aider](https://discord.com/channels/822583790773862470/1321432611964325898/1321555180558356543) 和 [bigcodebench](https://x.com/terryyuezhuo/status/1872017850933911802)）

但训练细节甚至更令人惊叹：

- **训练预算比同类模型低 8-11 倍**：具体使用了 **2048 块 H800**（即“阉割版 H100”），耗时 2 个月。相比之下，Llama 3 405B [根据其论文](https://arxiv.org/pdf/2407.21783) 是在 16k 块 H100 上训练的。他们估计这项成本为 550 万美元。
![image.png](https://assets.buttondown.email/images/82aefb8a-d42c-4b29-9ee9-0ce9e0a85bdf.png?w=960&fit=max)

- **自研原生 FP8 混合精度训练**（在没有 Blackwell GPU 的情况下实现——正如 [Shazeer 所期望的？](https://buttondown.com/ainews/archive/ainews-shazeer-et-al-2024/)） 
![image.png](https://assets.buttondown.email/images/be06f2ed-5b32-4647-8788-a5a6b79ded9a.png?w=960&fit=max)

- **扩展了来自 [DeepSeek v2 的 Multi-Head Latent Attention](https://x.com/nrehiew_/status/1872318170469699785)**
- **从 [R1 生成的合成推理数据中进行蒸馏 (distilling)](https://x.com/teortaxesTex/status/1872250466987545056/photo/1)** 
![image.png](https://assets.buttondown.email/images/14c210a0-3305-42cb-9e33-ee050c7ebe38.png?w=960&fit=max)
 并使用[其他类型的奖励模型 (reward models)](https://x.com/nrehiew_/status/1872318217395572895)
- **无需 [张量并行 (tensor parallelism)](https://x.com/main_horse/status/1872294985888059612?s=46)** —— 最近被 [Ilya 称为](https://www.latent.space/p/what-ilya-saw) 是一个错误
- **针对 DeepSeekMoE 风格的 MoE 进行 [剪枝 + 修复 (pruning + healing)](https://x.com/teortaxesTex/status/1872002534774341782)**，扩展至 [256 个专家 (experts)](https://x.com/nrehiew_/status/1872318173648736381)（8 个激活 + 1 个共享）
- **一种新的 [“**多 Token 预测**” (multi token prediction) 目标](https://x.com/nrehiew_/status/1872318176735752266)**（源自 [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)），允许模型提前查看并预规划未来的 Token（在本例中一次仅 2 个）
![image.png](https://assets.buttondown.email/images/876a8c76-e784-4552-b79e-32dee12b95ad.png?w=960&fit=max)



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

**AI 模型开发与发布**

- **DeepSeek-V3 发布与性能**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1872242657348710721) 和 [@reach_vb](https://twitter.com/reach_vb/status/1872246633649553556) 宣布发布 **DeepSeek-V3**，拥有 **671B MoE 参数**，并在 **14.8T Token** 上进行了训练。该模型在各项 Benchmark 中**超越了 GPT-4o 和 Claude Sonnet-3.5**。
- **算力效率与成本效益**：[@scaling01](https://twitter.com/scaling01/status/1872358867025494131) 强调 **DeepSeek-V3** 仅使用 **278.8 万 H800 GPU 小时**完成训练，与使用 **3080 万 GPU 小时**的 **Llama 3** 等模型相比，显著降低了成本。
- **部署与可访问性**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1872405681615081946) 和 [@reach_vb](https://twitter.com/reach_vb/status/1872246633649553556) 分享了通过 **Hugging Face** 等平台部署 **DeepSeek-V3** 的更新，强调了其**开源可用性**和 **API 兼容性**。

**AI 研究技术与 Benchmark**

- **OREO 与 NLRL 创新**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1872362868924162095) 讨论了 **OREO 方法**和 **Natural Language Reinforcement Learning (NLRL)**，展示了它们在**多步推理**和 **Agent 控制任务**中的有效性。
- **无需 Prompt 的 Chain-of-Thought 推理**：[@denny_zhou](https://twitter.com/denny_zhou/status/1872366450020659483) 介绍了 **Chain-of-Thought (CoT) 推理**的一项突破，通过微调模型使其能够进行**内在推理**，而不依赖于特定任务的 Prompt，从而显著增强了**模型推理能力**。
- **基准测试表现**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1872318200953946167) 和 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1872084934694961199) 报告称，**Multi-Token Prediction (MTP)** 和 **Chain-of-Knowledge** 等新技术在**数学解题**和 **Agent 控制**等领域持续**超越现有基准测试**。

**开源 AI vs 闭源 AI**

- **开源模型的竞争优势**：[@scaling01](https://twitter.com/scaling01/status/1872358867025494131) 强调 **DeepSeek-V3** 目前已**达到或超过**了 **GPT-4o** 和 **Claude Sonnet-3.5** 等闭源模型，主张由**开源 AI** 驱动的**可持续性与创新**。
- **许可与可访问性**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1872242666265801105) 强调 **DeepSeek-V3** 是**开源**的，且**许可用于商业用途**，使其成为闭源模型的一个**更自由的替代方案**，并促进了开发者和企业的**广泛普及**。
- **经济影响**：[@reach_vb](https://twitter.com/reach_vb/status/1872246633649553556) 和 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1872281217480978676) 讨论了**开源 AI** 如何使访问民主化，减少对**高利润闭源模型**的依赖，并培育一个更具**包容性的 AI 生态系统**。

**AI 基础设施与计算资源**

- **优化 GPU 使用**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1872370360307568964) 和 [@scaling01](https://twitter.com/scaling01/status/1872276861675286864) 探讨了 **DeepSeek-V3** 如何通过 **Multi-Token Prediction (MTP)** 和**负载均衡 (Load Balancing)** 等技术高效利用 **H800 GPU**，从而提高**计算利用率**和**训练效率**。
- **硬件设计改进**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1872318189373423697) 建议进行**硬件增强**，如改进 **FP8 GEMM** 和更好的**量化支持**以支持 **MoE 训练**，从而解决**通信瓶颈**和**计算效率低下**的问题。
- **具有成本效益的扩展策略**：[@reach_vb](https://twitter.com/reach_vb/status/1872246633649553556) 详细介绍了 **DeepSeek-V3** 如何以**极少量的典型计算资源**实现 **SOTA 性能**，强调通过**算法-框架-硬件协同设计**在扩展规模的同时保持**成本效益**。

**移民与 AI 人才政策**

- **倡导技术移民**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1872079097121431855) 和 [@HamelHusain](https://twitter.com/HamelHusain/status/1872414163957608455) 强调了 **H-1B 和 O-1 签证**等**高技能移民项目**对于促进 **AI 领域**内**创新与经济增长**的重要性。
- **政策批评与建议**：[@bindureddy](https://twitter.com/bindureddy/status/1872382667531948201) 和 [@HamelHusain](https://twitter.com/HamelHusain/status/1872412881771483160) 批评了**限制性签证政策**，主张**简化签证转换**、**取消特定职位的限制**并**扩大合法移民**，以增强**美国 AI 竞争力**和**创新**。
- **经济与道德论点**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1872079097121431855) 指出**移民创造的就业机会多于他们占用的机会**，将**签证改革**界定为支持**美国经济**的**经济必然要求**和**道德议题**。

**模因与幽默**

- **有趣的互动与模因**：[@HamelHusain](https://twitter.com/HamelHusain/status/1872090936588767416) 幽默地评论了**对 AI 模型表现的误解**，为技术讨论带来了**轻松有趣的基调**。
- **俏皮的 AI 对话**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1872309730997399833) 发布了一条模因式的评论，为关于 **AI 能力**的对话注入了**幽默感**。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek V3 发布：技术创新与基准测试**

- **DeepSeek-V3 正式发布** ([Score: 101, Comments: 22](https://reddit.com/r/LocalLLaMA/comments/1hmn55p/deepseekv3_officially_released/)): DeepSeek 发布了 **DeepSeek-V3**，采用 **Mixture of Experts (MoE) 架构**，拥有 **671B 总参数**和 **37B 激活参数**，其表现优于其他开源模型，并能媲美 **GPT-4o** 和 **Claude-3.5-Sonnet** 等闭源模型。该模型在知识类任务、长文本评估、编程、数学以及中文能力方面表现出显著提升，且 **Token 生成速度提升了 3 倍**。该开源模型支持 **FP8 权重**，社区工具如 **SGLang** 和 **LMDeploy** 已提供原生 FP8 推理支持，API 促销定价将持续至 **2025 年 2 月 8 日**。
  - **DeepSeek-V3 的 FP8 训练**：该模型使用 **FP8 混合精度训练框架**进行训练，标志着首次在大规模模型上验证了 FP8 训练的可行性。这种方法实现了稳定的训练过程，没有出现任何不可恢复的 Loss 突刺或回滚，引发了人们对 DeepSeek 是否已有效“攻克”了 FP8 训练的好奇。
  - **经济与技术考量**：训练 DeepSeek-V3 耗资 **550 万美元**，凸显了其背后量化机构所重视的经济效率。讨论还涉及了潜在的 GPU 制裁对模型设计的影响，暗示其可能针对 CPU 和 RAM 使用进行了优化，并提到了在 **Epyc 主板**上运行的可能性。
  - **社区与开源动态**：开源软件与免费软件之间存在区别，评论指出 DeepSeek-V3 在 **r/localllama** 上的发布针对的是本地社区，而非更广泛的开源推广。一些用户幽默地提到该模型在圣诞节发布，将其比作来自中国“圣诞老人”的惊喜。


- **[DeepSeek V3 Chat 版权重已上传至 Huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V3)** ([Score: 143, Comments: 67](https://reddit.com/r/LocalLLaMA/comments/1hmk1hg/deepseek_v3_chat_version_weights_has_been/)): **DeepSeek V3** Chat 版权重现已在 **Huggingface** 上线，提供了获取该 AI 模型最新迭代版本的通道。
  - **硬件要求与性能**：讨论强调了运行 **DeepSeek V3** 的极高硬件要求，提到 1-bit 量化需要 **384GB RAM** 和 **4 块 RTX 3090**。用户讨论了各种量化级别及其 VRAM 需求，并幽默地表示需要变卖家产才能买得起必要的 GPU。
  - **开源与竞争**：关于开源模型超越闭源模型的讨论非常激烈，提到了 **Elon Musk 的 X.AI**，以及开源模型可能超越其闭源模型 **Groq2** 和 **Groq3** 的讽刺意味。对话强调了开源竞争在推动技术进步方面的价值。
  - **模型大小与复杂性**：该模型拥有 **685B 参数**和 **163 个分片**，是讨论的焦点，用户开玩笑说需要 **163 块 GPU** 是多么不切实际。这凸显了在硬件和软件实现方面处理如此庞大且复杂模型的挑战。


- **[Sonnet 3.5 对比 v3](https://i.redd.it/y5zmucuql79e1.png)** ([Score: 83, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1hmqb2j/sonnet35_vs_v3/)): **DeepSeek V3** 在基准测试中显著优于 **Sonnet 3.5**，正如一张动画图片所示，图中展示了标记为 "Claude" 和 "DeepSeek" 的角色之间的激烈对抗。这一场景传达了动态且竞争激烈的环境，强调了两者之间显著的性能差距。
  - **DeepSeek V3** 的性价比极高，比 **Sonnet 3.6** 便宜 **57 倍**，且在其网站上提供几乎无限的可用性，而 **Claude** 即使对付费用户也有访问限制。
  - 尽管 **DeepSeek V3** 的性价比被用户评为 **10/10**，但对其较短的上下文窗口仍有一些担忧。
  - 用户表示有兴趣对 **DeepSeek V3** 进行实际测试以验证基准测试结果，并建议将其纳入 **lmarena 的 webdev arena**，以便与 **Sonnet** 进行更全面的对比。


**主题 2. DeepSeek V3 与竞品的成本效益**

- **PSA - Deepseek v3 在 API 费率上比 Sonnet 便宜 53 倍且性能更强** ([Score: 291, Comments: 113](https://reddit.com/r/LocalLLaMA/comments/1hmm8v9/psa_deepseek_v3_outperforms_sonnet_at_53x_cheaper/)): **Deepseek V3** 的性能超越了 **Sonnet**，且 API 费率便宜了 **53 倍**，即使与 3 倍的价格差异相比，这也是一个巨大的差距。作者表达了对 **Anthropic** 的兴趣，并建议如果某个模型在编程任务中能提供实质性的改进，他们可能仍愿意支付更高费用以获得卓越性能。
  - **Deepseek V3** 的**训练成本**为 560 万美元，在不到两个月的时间内使用了 2,000 块 H800，突显了 **LLM training** 的潜在效率。该模型的 **API pricing** 明显低于 **Claude Sonnet**，其成本为 **输入 $0.14/1M** 和 **输出 $0.28/1M**，而 Sonnet 为 **输入 $3/1M** 和 **输出 $15/1M**，这比某些本地构建的电力成本还要便宜约 5 倍。
  - **Deepseek V3** 的 **context window** 仅为 64k，这可能是其高成本效益的原因之一，尽管在某些基准测试中其表现仍逊于 **Claude**。讨论中涉及了模型的**参数规模**（37B 激活参数）以及使用 **MoE** (Mixture of Experts) 来降低推理成本。
  - 针对**数据使用**和**在 API 请求上进行训练**的担忧被提出，一些人对模型的性能和数据实践持怀疑态度。人们期待 **Deepseek V3** 在 **OpenRouter** 等平台上线，并提到促销活动将持续到 2 月，以进一步降低成本。


- **[Deepseek V3 基准测试提醒我们 Qwen 2.5 72B 才是真正的王者，其他人都在开玩笑！](https://i.redd.it/q4gg1cobp79e1.png)** ([Score: 86, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1hmqpca/deepseek_v3_benchmarks_are_a_reminder_that_qwen/)): **DeepSeek V3** 的基准测试证明 **Qwen 2.5 72B** 是一款领先模型，在多个基准测试中超越了 **Llama-3.1-405B**、**GPT-4o-0513** 和 **Claude 3.5**。值得注意的是，**DeepSeek-V3** 在 **MATH 500** 基准测试中以 **90.2%** 的得分脱颖而出，彰显了其卓越的准确性。
  - 讨论强调了在服务器上为多用户运行 **DeepSeek V3** 等模型的**成本效益**，而非使用 **2x3090** 等 GPU 的本地设置，强调了在电力和硬件上的节省。**OfficialHashPanda** 指出了 **MoE (Mixture of Experts)** 的优势，它允许在增加能力的同时减少激活参数，使其适合服务大量用户。
  - 评论探讨了**硬件需求**和**成本**，提到了使用**廉价 RAM** 和具有高内存带宽的服务器 CPU 来高效运行大型模型。对话对比了 **API** 与本地硬件设置的成本，建议基于服务器的解决方案对于大规模使用更为经济。
  - 讨论了**小型高效模型**的潜力，并对 **DeepSeek V3 Lite** 可能提供的功能表示兴趣。**Calcidiol** 建议未来的“轻量级”模型通过利用更好的训练数据和技术，可能会达到当今大型模型的能力，这表明了 AI 模型的持续演进和优化。


**Theme 3. DeepSeek V3 中的 FP8 训练突破**

- **[Deepseek V3 正式发布（代码、论文、基准测试结果）](https://github.com/deepseek-ai/DeepSeek-V3)** ([Score: 372, Comments: 96](https://reddit.com/r/LocalLLaMA/comments/1hmmtt3/deepseek_v3_is_officially_released_code_paper/)): **DeepSeek V3** 已正式发布，具有 **FP8 training** 能力。发布内容包括代码访问、研究论文和基准测试结果，标志着 AI 训练方法领域的重大进展。
  - **DeepSeek V3 的性能与能力**：尽管拥有令人印象深刻的架构和 FP8 训练，DeepSeek V3 在某些基准测试中仍落后于 **Claude Sonnet 3.5** 等模型。然而，它被誉为目前最强的 open-weight 模型，如果模型尺寸减小，则具有更容易 self-hosting 的潜力。
  - **技术要求与成本**：运行 DeepSeek V3 需要大量资源，例如 600B 模型需要 **384GB RAM**，基础设置成本可能在 **$10K** 左右。用户讨论了各种硬件配置，包括 **EPYC** 服务器和仅 CPU 推理的可行性，强调了对大量 RAM 和 VRAM 的需求。
  - **创新特性与许可担忧**：该模型引入了诸如 **Multi-Token Prediction (MTP)** 和高效的 FP8 混合精度训练等创新特性，将训练成本显著降低至 **2.664M GPU hours**。然而，许可问题令人担忧，因为 Deepseek 许可证被认为对商业用途有高度限制。

- **[哇，这可能是目前最好的开源模型？](https://i.redd.it/vry52nz3u69e1.jpeg)** ([得分: 284, 评论: 99](https://reddit.com/r/LocalLLaMA/comments/1hmnj93/wow_this_maybe_probably_best_open_source_model/)): **DeepSeek-V3** 作为一款开源模型展现了卓越的性能，超越了其前代产品以及竞争对手，如 **DeepSeek-V2.5**、**Qwen2.5-72B-Inst**、**Llama-3.1-405B-Inst**、**GPT-4o-0513** 和 **Claude-3.5-Sonnet-1022**。值得注意的是，它在 **MATH 500 benchmark** 上达到了 **90.2% 的准确率**，表明了其在使用 **FP8** 时强大的训练稳定性和效率。
  - **推理挑战与能力**：用户讨论了由于其 **671B 参数**量，在本地运行 **DeepSeek-V3** 的难度，4-bit 量化至少需要 **336GB 的 RAM**。尽管如此，由于其 **37B 激活参数**和包含 **256 个专家**的 **Mixture of Experts** 架构，它在 **512GB 双路 Epyc 系统**上的 CPU 推理速度可达约 **10 tokens/second**。
  - **模型对比与性能**：该模型的性能被认为可与 **GPT-4o** 和 **Claude-3.5-Sonnet** 等闭源模型相媲美，一些用户指出它在目标导向型任务中具有超越对手的潜力，尽管在指令遵循方面可能稍逊于 **Llama**。
  - **开放权重 vs. 开源**：关于该模型是 **Open Weights**（开放权重）而非完全 **Open Source**（开源）存在一些困惑和澄清，讨论涉及其影响以及未来蒸馏为更易于管理的尺寸（如 **72B 参数**）的可能性。


## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI O1 模型影响金融市场**

- **[OpenAI o1 在交易和投资中的真实用例](https://medium.com/@austin-starks/i-just-tried-openais-updated-o1-model-this-technology-will-break-wall-street-5f99bcdac976)** ([得分: 232, 评论: 207](https://reddit.com/r/OpenAI/comments/1hmlwfq/a_real_usecase_of_openai_o1_in_trading_and/)): **OpenAI O1 模型** 在金融研究和交易策略方面展示了显著进步，通过提供精确的数据驱动解决方案并支持用于生成 JSON 对象的 **function-calling**，其表现优于传统模型。值得注意的是，该模型执行精确金融分析的能力（例如识别自 2000 年以来 SPY 在 7 天内下跌 5% 的情况），使得创建无需编程即可部署的复杂交易策略成为可能。该模型增强的能力，包括 **vision API** 和改进的推理，使其能够处理复杂的金融任务，通过使算法交易和研究民主化，有可能改变金融业和华尔街。
  - 评论者强调了在金融市场中使用 AI 的挑战和怀疑态度，重点提到了 **data leakage**（数据泄漏）和 **efficient market hypothesis**（有效市场假说）等问题。许多人指出，由于市场的适应性和随机性，历史回测并不能保证未来的成功，这表明这些模型在实时场景中可能表现不佳。
  - 讨论中涉及了**投资的行为层面**，一些用户指出对市场波动的过度情绪反应（如在回撤期间抛售）可能会破坏策略。会议还强调了理解 **risk-reward dynamics**（风险回报动态）以及避免对 AI 生成策略过度自信的重要性。
  - 少数用户分享了个人经验和项目，例如使用各种 LLM 的 **Vector Stock Market bot**，但也承认了其局限性并需要进一步测试。普遍共识是，AI 可能会使工具的使用变得民主化，但由于市场固有的复杂性，并不一定会带来持续的超额收益。


**主题 2. 围绕 O1 Pro 模式实用性的辩论**

- **o1 pro mode is pathetic.** ([Score: 177, Comments: 133](https://reddit.com/r/OpenAI/comments/1hmkrrf/o1_pro_mode_is_pathetic/)): 该帖子批评了 **OpenAI 的 o1 Pro 模式**，称其价格过高，且由于输出生成速度慢，在编程任务中与 **4o** 相比效率低下。作者自称是 AI 业余爱好者，认为这些模型针对基准测试（benchmarks）过度拟合，在实际应用中并不实用，并暗示“推理模型”主要是一种营销策略。唯一被注意到的实际用途是在对齐任务中，模型可以评估用户意图。
  - **o1 Pro** 模型收到的评价褒贬不一；一些用户发现它在处理复杂编程任务时非常宝贵，理由是它能够处理大型代码库并能一次性产生准确结果，而另一些人则批评其响应速度慢且知识截止日期（knowledge cutoff）过时。像 **ChronoPsyche** 和 **JohnnyTheBoneless** 这样的用户称赞其处理复杂任务的能力，而像 **epistemole** 这样的人则认为，无限制的速率限制（rate limits）才是真正的优势，而非模型性能。
  - 几位用户强调了**详细 Prompt** 对于最大化 o1 Pro 潜力的重要性，建议提供全面的文档或在大型上下文窗口中使用迭代方法，相比于输入零散的代码片段，这样可以产生更好的效果。**Pillars-In-The-Trees** 将有效的 Prompt 编写比作指导一名研究生，突出了该模型在逻辑任务方面的熟练程度。
  - 讨论显示 **o1 Pro** 在某些编程语言中表现出色，用户如 **NootropicDiary** 提到它在 Rust 方面优于 Claude 等其他模型，而其他人则发现 **Claude** 在 TypeScript 等不同语言中更有效。这强化了这样一种观点：模型的有效性会根据任务和所使用的语言而显著不同。


**Theme 3. OpenAI 最新进展与工具概览**

- **[12 Days of OpenAi - 综合总结。](https://i.redd.it/a7bdk15t569e1.jpeg)** ([Score: 227, Comments: 25](https://reddit.com/r/OpenAI/comments/1hmlno0/12_days_of_openai_a_comprehensive_summary/)): “OpenAI 的 12 天”网格图记录了 **12 月 5 日至 12 月 20 日** 期间的每日亮点，包括 12 月 5 日的 **ChatGPT Pro 计划** 和 12 月 6 日的**强化微调 (Reinforcement Fine-Tuning)**。该系列以 12 月 20 日 **o3 和 o3-mini** 的进展达到高潮，预示着向 **AGI** 迈进。
  - **第 2 天**的**强化微调 (Reinforcement Fine-Tuning)** 被强调为一项重大进展，具有通过极少的示例显著改进系统的潜力。虽然缺乏正式论文留下了一些不确定性，但它对 Agent 开发的影响被认为是大有可为的，特别是展望 **2025** 年。
  - 围绕 **Canvas UX** 的讨论表明，其最近的更新较小，一些用户对其约 **200 行**的限制表示不满。尽管这是早先推出的功能，但它仍然是用户争论的一个点。
  - 人们对 **MacOS 应用更新**后 **Windows 应用**的可用性感到好奇，并幽默地建议这可能与 **Microsoft** 为 OpenAI 建造核反应堆的时间点重合。


**Theme 4. ChatGPT 宕机及其对用户的影响**

- **[CHAT GPT IS DOWN.](https://i.redd.it/kidl8f7dp89e1.png)** ([Score: 366, Comments: 206](https://reddit.com/r/OpenAI/comments/1hmv4v8/chat_gpt_is_down/)): **ChatGPT** 经历了严重的服务中断，在 **下午 6:00** 报告的故障峰值达到 **5,315 起**。图表显示在一段低活跃期后，故障报告急剧增加，表明受影响的用户范围广泛。
  - 用户对 **ChatGPT 宕机**表达了沮丧和幽默，一些人开玩笑说在作业和生产力任务上对 AI 的依赖。**Street-Inspectors** 幽默地指出了询问 ChatGPT 为什么它不工作这一行为的讽刺性。
  - 提到了 **OpenAI 的状态页面**和 **Downdetector** 作为检查宕机状态的来源，**bashbang** 提供的一个链接显示了影响 ChatGPT、API 和 Sora 的重大故障。
  - **Kenshiken** 和 **BuckyBoy3855** 提到“上游供应商 (upstream provider)”是问题的原因，强调了宕机的技术层面，而 **HappinessKitty** 则推测是服务器容量问题。


---

# AI Discord Recap

> 由 o1-2024-12-17 生成的摘要之摘要

**主题 1. DeepSeek V3 成为焦点**  

- [**大规模混合精度提升**](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)：DeepSeek V3 发布了一个拥有 685B 参数的模型，采用 FP8 训练，声称节省了 2 个数量级的成本。该模型运行速度约为 60 tokens/second，在 14.8T tokens 上进行了训练，许多人将其视为 GPT-4o 的强力开源竞争对手。  
- [**API 普及与使用量翻三倍**](https://x.com/OpenRouterAI/status/1872334128043208833)：OpenRouter 报告称 DeepSeek V3 发布后使用量翻了三倍，足以与价格更高的老牌模型竞争。社区成员赞扬了其强大的代码编写性能，但也指出了响应慢和 VRAM 需求大的问题。  
- [**MoE 架构引发热议**](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)：DeepSeek 的 Mixture-of-Experts (MoE) 架构提供了更清晰的扩展路径和大幅降低的训练成本。工程师们推测未来的开源扩展以及用于稳定推理的 320-GPU HPC 集群。  

**主题 2. 代码编辑器与 IDE 的困扰**  

- [**Windsurf 与 Cascade 的困境**](https://codeium.com/support)：Windsurf 的 Cascade Base 模型因处理代码提示词不当以及在没有结果的情况下消耗额度而受到批评。工程师们提出了 global_rules 的变通方案，但许多人仍对 UI 延迟和无响应的查询感到沮丧。  
- [**Cursor IDE 的 Token 考验**](https://www.cursor.com/downloads)：Cursor IDE 在处理有限的上下文和大型代码任务的性能下降方面挣扎。用户将其与 DeepSeek 和 Cline 进行了对比，称赞后两者拥有更长的上下文窗口，能实现更稳健的代码生成。  
- [**Bolt 的 Token 紧张局势**](https://github.com/stackblitz/bolt.new/issues)：Stackblitz (Bolt.new) 用户在重复的代码请求中消耗了高达 150 万个 tokens。许多人要求直接进行代码编辑而非提供示例代码片段，并转向 GitHub 反馈以寻求订阅层级的改进。  

**主题 3. AI 赋能创意与协作工作**  

- [**播客结合实时摘要**](https://akashq.com.)：NotebookLM 用户将 Google News 集成到 AI 驱动的播客中，在播报时事的同时生成喜剧片段。一些人分享了 15 分钟的 TTRPG 回顾，突显了 AI 让爱好者快速获取信息的能力。  
- [**ERP 与角色扮演**](https://medium.com/@camauger/crafting-effective-chatgpt-prompts-for-tabletop-roleplaying-games-a-step-by-step-guide-part-1-b81a791d278d)：爱好者们为沉浸式桌面战役编写了高级提示词，确保复杂叙事的连续性。他们提到分块 (chunking) 和检索增强生成 (RAG) 对于稳定的长篇叙事至关重要。  
- [**语音对语音与音乐生成**](https://github.com/Eplisium/ai-chat-terminal)：AI 工程师展示了语音对语音聊天应用和根据文本提示创作音乐。他们邀请合作者共同优化 DNN-VAD 流水线，在有趣的新工作流中将音频转换与生成式文本模型连接起来。  

**主题 4. 检索、微调与 HPC 扩展**  

- [**GitIngest 与 GitDiagram**](https://gitingest.com)：开发者将大规模代码库映射为文本和图表，用于 RAG 实验。这种方法简化了 LLM 训练和代码摄取，让 HPC 集群能更有效地处理大型仓库。  
- [**LlamaIndex 与 DocumentContextExtractor**](https://hub.athina.ai/athina-originals/end-to-end-implementation-of-unstructured-rag/)：用户接入批处理以降低 50% 的成本并处理非工作时间的任务。结合块切分、本地 embeddings 和可选的开源 RLHF 工具，提高了在现实世界数据上的准确性。  
- [**微调 VLM 与 HPC MLOps**](https://github.com/haotian-liu/LLaVA)：研究人员利用 LLaVA、Qwen-VL 和 Guild AI 等 HPC 框架来管理大规模模型训练。他们注意到了 HPC 的开销，并讨论了构建自己的极简 Ops 解决方案以避免 SaaS 陷阱。  

**主题 5. 关键技术与性能修复**  

- [**TMA 击败 cp.async**](https://github.com/NVIDIA/cutlass/discussions/2013)：HPC 专家解释了在 H100 上进行 GEMM 计算时，TMA 如何优于 cp.async，从而实现批量调度和更低的寄存器占用。他们赞扬了 CUTLASS 中的结构化稀疏内核带来的进一步增益，尤其是在 FP8 下。  
- [**Mojo 与 Modular 的进展**](https://github.com/mahiro21h/mojo/commits/fix-input-segfaults-on-eof/)：用户调试了 StringRef 崩溃问题，并发现了 memcpy 调用中缺失的长度检查。他们称赞了新的周边商品，并讨论了 MAX 与 XLA 的编译时间，关注 HPC 代码的改进。  
- [**Tinygrad 与 PyTorch 的速度竞赛**](https://github.com/tinygrad/tinygrad/issues/4878)：Tinygrad 在 CUDA 上的前向传播速度落后于 PyTorch（800ms vs. 17ms），但开发者寄希望于 beam search 缓存和 jitting。他们合并了针对输入创建循环的 PR 修复，并解决了匹配引擎的悬赏任务以减少开销。

---

# PART 1: High level Discord summaries

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 大胆的行业突破**：一段新视频展示了工程师们详细介绍 **Windsurf** 如何刻意挑战典型的开发方法，分享了关于工作流变革和设计选择的见解 ([Windsurf's Twitter](https://x.com/windsurf_ai/status/1872375661542920424))。
   - 他们还发出了**节日问候**，强调了社区精神，并引发了关于这些大胆举措背后新鲜视角的讨论。
- **Cascade Base 模型之忧**：用户批评 **Cascade Base model** 在处理复杂编程任务时缺乏准确性，且经常无法执行简单指令，尤其是与 **Claude 3.5 Sonnet** 相比时。
   - 尽管一些人分享了通过全局规则取得的部分成功，但其他人发现改进微乎其微，并在发布挫败感的同时附上了 [awesome-windsurfrules](https://github.com/SchneiderSam/awesome-windsurfrules) 等链接。
- **远程主机故障与延迟**：通过 **SSH Remote Hosts** 连接的用户注意到 Windsurf 显示出显著延迟，导致实时编辑变得混乱且无法追踪，直到 Cascade 更新。
   - 他们报告称命令仍能正常执行，但延迟的界面造成了工作流脱节，许多人认为这具有干扰性。
- **额度消耗与无响应查询**：当无响应的请求消耗了 **tokens** 却未交付功能性输出时，用户感到被亏待，导致他们通过 [Windsurf Editor Support](https://codeium.com/support) 反复联系支持部门。
   - 许多人对这些**消耗额度**的失败表示担忧，认为这削弱了对 Windsurf 处理大型代码库可靠性的信心。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek V3 主导讨论**：开发者们称赞 **DeepSeek V3** 的代码生成和分析能力，声称其可与 **Sonnet 3.5** 媲美且输出速度更快，详见 [DeepSeek 的这条推文](https://x.com/deepseek_ai/status/1872242657348710721)。
   - 社区成员讨论了与 **Cursor IDE** 的集成，参考了 [DeepSeek Platform](https://platform.deepseek.com/) 的 **API-compatible** 方案，并对更低的使用成本表现出兴趣。
- **Cursor IDE 的 Token 考验**：许多人报告 **Cursor IDE** 的上下文窗口有限，降低了大型代码生成任务的性能，[Cursor 官网](https://www.cursor.com/downloads)提供了各平台的下载。
   - 用户对比了 **DeepSeek** 和 **Cline** 如何更高效地处理扩展上下文窗口，并参考 [Cursor Forum](https://forum.cursor.com/) 持续反馈关于更好利用 token 的建议。
- **Next.js UI 问题困扰设计者**：创作者们在 **Next.js 的 UI 问题**中挣扎，抱怨 **Claude** 生成的代码有时会导致元素错位和样式复杂化，即使在使用 [shadcn](https://github.com/shadcn) 等库之后也是如此。
   - 他们建议将相关文档嵌入上下文以获得更好的设计效果，并推荐使用 [Uiverse](https://uiverse.io/elements) 获取快速 UI 组件。
- **OpenAI 可靠性的起伏**：一些人面临 **OpenAI** 最近的性能问题，理由是响应时间变慢且可用性降低，而替代模型以更低的成本提供了更稳定的结果。
   - 他们建议测试多个 AI 系统，参考 [DeepSeek API Docs](https://api-docs.deepseek.com/) 以获取兼容性信息，而其他人则只是在不同供应商之间切换以保持任务推进。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.70.0 提升自我代码统计数据**：Aider v0.70.0 引入了**分析选项加入 (analytics opt-in)**、针对交互式命令的新**错误处理**以及扩展的**模型支持**，详见 [Aider Release History](https://aider.chat/HISTORY.html)。
   - 社区成员强调 Aider 有 **74%** 的代码是自我贡献的，并称赞该工具改进的安装流程、文件监视功能以及 **Git** 名称处理是重大的升级。
- **DeepSeek V3 实现 3 倍速度提升**：**DeepSeek V3** 现在每秒处理 **60 tokens**（比 V2 快 3 倍），展现出比 **Sonnet 3.5** 更强的编程性能，并具备 **64k token** 的上下文限制，详见[此推文](https://x.com/deepseek_ai/status/1872242657348710721)。
   - 社区对 **DeepSeek V3** 在某些任务上超越 **Claude** 感到兴奋，尽管响应缓慢和上下文管理仍是持续讨论的焦点。
- **BigCodeBench 揭示 LLM 的优势与不足**：**BigCodeBench Leaderboard** ([链接](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)) 在现实世界的编程任务上评估 LLM，并引用 [arXiv 论文](https://arxiv.org/abs/2406.15877) 以深入了解其方法论。
   - 贡献者对比了 **DeepSeek** 和 **O1** 的得分，指出这些指标有助于澄清各模型在**实际**条件下的代码生成能力。
- **GitDiagram 与 GitIngest 让仓库透明化**：[GitDiagram](https://gitdiagram.com) 将 GitHub 仓库转换为交互式图表，而 [GitIngest](https://gitingest.com) 将任何 Git 仓库渲染为纯文本，以便轻松进行代码摄取。
   - 用户只需将 URL 中的 **'hub'** 替换为 **'diagram'** 或 **'ingest'**，即可立即可视化仓库结构或为任何 **LLM** 做好准备。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek V3 的 GPU 资源消耗与收益**：**DeepSeek V3** 发布，拥有 **6850 亿参数**，需要约 **320 块** H100 等 GPU 才能达到最佳性能，如[官方代码仓库](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)所示。
   - 讨论强调了其稳定推理对大容量 VRAM 的需求，成员们称其为*目前可用的最大开放权重模型之一*。
- **可微缓存加速推理**：关于**可微缓存增强 (Differentiable Cache Augmentation)** 的研究揭示了一种将冻结的 LLM 与操作 **key-value (kv) cache** 的离线协处理器配对的方法，如[此论文](https://huggingface.co/papers/2412.17747)所述。
   - 该方法降低了推理任务的困惑度 (perplexity)，成员们观察到即使协处理器离线，它也能*保持 LLM 的功能*。
- **文本转视频之争：Hunyuan 对阵 LTX**：用户对比了 **Hunyuan** 和 **LTX** 文本转视频模型的性能，强调了实现流畅渲染对 VRAM 的要求。
   - 他们对 T2V 的发展表现出浓厚兴趣，建议资源密集型任务可能会从流水线调整中受益。
- **URL 审核 API 难题**：一位 AI 工程师在构建能够准确分类不安全网站的 **URL 审核 API** 时遇到困难，凸显了 **Llama** 的结构化输出问题以及 **OpenAI** 频繁拒绝请求的问题。
   - 社区反馈指出特定领域处理的重要性，因为反复尝试产生的结果往往不一致或不完整。
- **推理成本难题**：参与者辩论了部署大型 AI 模型的**成本结构**，质疑促销定价是否能承受高使用量需求。
   - 他们建议持续的负载可能会平衡运营支出，从而使高性能 AI 服务在成本压力下依然可行。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Web Searching LLMs & 价格大跌**：OpenRouter 为任何 LLM 推出了 **Web Search** 功能，目前免费使用，为用户查询提供及时的参考，如[此演示](https://x.com/OpenRouterAI/status/1871682806335824029)所示。他们还大幅下调了多个模型的价格，包括 **qwen-2.5** 降价 **12%**，**hermes-3-llama-3.1-70b** 降价 **31%**。
   - 社区成员认为降价幅度巨大并表示欢迎，特别是对于高端模型。一些人预计成本结构将出现更广泛的转变。
- **DeepSeek v3 使用量翻三倍**：**DeepSeek v3** 在 OpenRouter 上的受欢迎程度飙升，据[此贴](https://x.com/OpenRouterAI/status/1872334128043208833)报道，自发布以来使用量翻了三倍，在某些指标上可与更大的模型媲美。它以更低的价格与 **Sonnet** 和 **GPT-4o** 竞争，引发了关于“中国 AI 已经赶上”的讨论。
   - **general** 频道的用户对其在编程任务和诗歌方面的表现评价褒贬不一。一些人称赞其创意输出，而另一些人则指出结果不一致。
- **Endpoints 与 Chat 困扰**：**OpenRouter** 推出了 Beta 版 **Endpoints API**，允许开发者获取模型详情，参考用法见[此处](https://openrouter.ai/api/v1/models/google/gemini-2.0-flash-thinking-exp:free/endpoints)。一些用户在对话历史较长时遇到了 **OpenRouter Chat** 延迟，呼吁对大数据集进行更灵敏的处理。
   - 社区注意到没有对 batching 请求的直接支持，强调了及时的 GPU 使用。同时，某些“未找到端点”的错误源于 API 设置配置错误，凸显了正确设置的重要性。
- **文字驱动的 3D 游戏魔法**：一个新展示的工具承诺可以根据简单的文本提示创建 **3D 游戏**，相比早期使用 **o-1** 和 **o-1 preview** 的尝试有所改进。这种方法暗示了未来将集成 voxel 引擎以处理更复杂的形状，如[此项目链接](https://toy.new/)中所预告。
   - 爱好者认为这是对之前基于 GPT 尝试的飞跃，其功能似乎经过精炼，可用于构建完整的交互式体验。频道中的一些人认为，如果规模化，它可能会改变独立游戏开发流程。
- **AI Chat Terminal: Agent 在行动**：**AI Chat Terminal (ACT)** 将 **Agent 特性**与代码库交互相结合，允许用户在 **OpenAI** 和 **Anthropic** 等提供商之间切换。它引入了 **Agent Mode** 来自动化任务，旨在简化编码过程，如[此仓库](https://github.com/Eplisium/ai-chat-terminal)所示。
   - **app-showcase** 频道的开发者强调了在单个终端中灵活使用多模型的潜力。许多人称赞其在构建超越典型聊天限制的脚本方面的便利性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 优化大模型表现**：Build **0.3.5** 修复了之前 **GGUF** 模型加载中的 Bug，并解决了 **MLX** 的会话处理问题，参考 [Issue #63](https://github.com/lmstudio-ai/mlx-engine/issues/63)。
   - 用户注意到 **QVQ 72B** 和 **Qwentile2.5-32B** 现在运行得更好，尽管一些内存泄漏仍在调查中。
- **RPG 爱好者保持叙事流畅**：爱好者使用 **Mistral** 和 **Qwen** 等模型来管理长期的桌面游戏故事情节，其 Prompt 灵感来自 [ChatGPT TTRPG 指南](https://medium.com/@camauger/crafting-effective-chatgpt-prompts-for-tabletop-roleplaying-games-a-step-by-step-guide-part-1-b81a791d278d)。
   - 他们探索了微调和 **RAG** 技术以获得更好的连续性，并引用独立分块（chunking）作为保持背景设定一致性的策略。
- **X99 系统紧跟步伐**：在 **X99** 主板上运行 **Xeon E5 v4** 的用户报告称，即使使用旧设备，模型推理性能依然稳健。
   - 双 **RTX 2060** 配置展示了对大型模型的稳定处理，打破了对新硬件的紧迫需求。
- **多 GPU 收益与 LoRAs 热度**：参与者观察到 GPU 利用率较低（约 **30%**），并强调额外的 VRAM 并不总是能带来速度提升，除非配合 **NVLink** 等增强功能。
   - 他们还推测即将推出**视频生成 LoRAs**，尽管有些人怀疑在极少数静态图像上进行训练的效果。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **提示词精度助力 SD**：许多参与者发现，更具描述性的提示词能产生更优质的 **Stable Diffusion** 输出，强调了详尽指令的优势。他们测试了各种模型，突出了风格和资源使用方面的差异。
   - 他们强调需要强大的提示词能力来**实现更好的图像控制**，并建议通过模型实验来优化结果。
- **ComfyUI 的复杂性可控**：贡献者通过符号链接模型并参考 **Stability Matrix** 进行更简单的管理来操作 **ComfyUI**，尽管许多人认为学习曲线很陡峭。他们还分享了 [SwarmUI](https://www.reddit.com/r/comfyui/comments/1hm9qhu/another_ai_in_the_loop/) 为新手提供了更易用的界面。
   - 用户将 SwarmUI 等门槛较低的前端与标准 ComfyUI 进行了比较，思考这些工具如何在不牺牲高级功能的情况下简化**生成艺术**（generative art）流程。
- **视频生成势头强劲**：爱好者们在 ComfyUI 中实验了 **img2video** 模型，并将其与 **Veo2** 和 **Flux** 的效率进行了对比。他们发现 [LTXVideo Q8](https://github.com/KONAKONA666/LTX-Video) 在 8GB **VRAM** 的配置下表现良好。
   - 他们仍然渴望测试新的视频生成方法，以扩展对资源友好型的可能性，继续在较低硬件规格上突破界限。
- **NSFW LoRA 引发讨论**：关于 **LoRA** 中的 **NSFW** 过滤器产生了一些有趣的交流，讨论了如何管理审查开关。参与者希望就每个设置在控制成人内容方面的作用进行公开讨论。
   - 他们强调标准的 **LoRA** 约束偶尔会阻碍合法的创作任务，呼吁提供关于审查开关更清晰的文档。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 停机引发新选择**：成员们遇到了 **ChatGPT** 宕机，参考了 [**OpenAI 状态页面**](https://status.openai.com/incidents/6bwlxnvdncnm)，并权衡了 **DeepSeek** 和 **Claude** 等替代方案。
   - 此次停机感觉比之前的事件轻微，但它重新激发了探索不同模型的兴趣。
- **DeepSeek V3 飞速领先**：成员们注意到 **DeepSeek V3** 拥有 **64k** 的上下文限制，在速度和代码一致性方面优于 **GPT-4**。
   - 爱好者们赞扬了其可靠的代码支持，而一些人指出了缺失的功能，如直接文件处理和对 **OCR** 的依赖。
- **GPT-O3 隐约可见**：提到 **O3-mini** 将于 1 月底发布，引发了随后不久发布完整 **O3** 模型的希望。
   - 具体细节仍然匮乏，引发了对其性能和可能的新功能的猜测。
- **缩写词困扰 LLMs**：**缩写词识别**引发了辩论，揭示了某些模型在正确扩展特定领域缩写方面的困难。
   - 提出了诸如自定义字典或优化提示词等技术，以保持扩展的一致性。
- **Canvas 与 ESLint 冲突**：用户遇到了 **Canvas** 窗口在打开几秒后消失的问题，导致编辑工作流中断。
   - 其他人在 **O1 Pro** 下苦于 **ESLint** 设置，旨在寻求一个适合高级开发需求的整洁配置。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **ProductPAPI 正在研发中**：一名成员透露了 **ProductPAPI**，这是 Gabe 开发的一款旨在简化任务的应用，但尚未公开*发布日期*、*核心功能*和 *API 结构*等细节。
   - 另一位用户表示“我们需要更多见解来评估其潜力”，并建议参考 [GitHub 社区反馈线程](https://github.com/stackblitz/bolt.new/issues) 以了解任何**计划中的扩展**。
- **Anthropic 简洁模式的难题**：成员们报告称，在 Bolt 上使用 **Anthropic 的简洁模式 (concise mode)** 时**质量有所下降**，并强调了高峰使用时段的**可扩展性担忧**。
   - 一位用户推测各供应商之间存在“普遍的扩展压力”，并提到当 **Claude 需求警告**触发时，**Bolt** 的性能也会随之变慢。
- **直接修改代码，否则免谈**：沮丧的开发者注意到他们不断收到“示例代码”而非直接编辑，敦促他人在 Prompt 中明确要求**“请直接进行修改”**。
   - 他们测试了在单个 Prompt 中添加澄清指令，并链接到 [最佳实践](https://bolters.io/docs/read-this-first) 以优化 **Prompt 措辞**，确认了这样能获得更好的代码修改效果。
- **Bolt 中的 Token 紧张局势**：用户反映 **Token 消耗过高**——有人声称在重写相同的代码请求时消耗了 *150 万个 Token*，理由是 **Prompt 被忽略**以及出现了意外更改。
   - 他们在 [GitHub](https://github.com/stackblitz/bolt.new/issues) 和 [Bolters.io](https://bolters.io) 上发布了反馈，提议更新订阅层级，许多人在达到 **Token** 上限后开始探索在 StackBlitz 上进行**免费编码**。
- **利用 Bolt 实现网约车雄心**：一位新手询问是否可以用 Bolt 构建一个全国性的**网约车 App**，并提到了他们现有的机场乘车门户，寻求**可扩展性**和**多区域**支持。
   - 社区成员对这一想法表示支持，称其为“大胆的扩展”，并引用了 [Bolters.io 社区指南](https://bolters.io/docs/read-this-first) 中关于扩展的逐步检查清单。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **QVQ-72B 亮相并获得高分**：**QVQ-72B** 推出了 4-bit 和 16-bit 版本，在 MMMU 基准测试中达到 **70.3%**，成为视觉推理领域的有力竞争者 ([Qwen/QVQ-72B-Preview](https://huggingface.co/Qwen/QVQ-72B-Preview))。
   - 社区成员强调了**数据格式化**和谨慎的训练步骤，并指向 [Unsloth 文档](https://docs.unsloth.ai/get-started/all-our-models) 以获取模型最佳实践。
- **DeepSeek V3 引发 MoE 热议**：采用 Mixture of Experts 配置的 **DeepSeek V3** 模型因比 Sonnet **便宜 50 倍**而备受关注 ([deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base))。
   - 一些人推测 **OpenAI** 和 **Anthropic** 也在采用类似技术，引发了关于扩展和成本效率的技术讨论。
- **Llama 3.2 遭遇数据不匹配障碍**：多位用户在纯文本 JSONL 数据集上微调 **Llama 3.2** 时遇到困难，尽管禁用了视觉层，仍会遇到意外的图像数据检查。
   - 其他人报告性能参差不齐，将失败归因于**输入质量**而非数量，并参考了 [Unsloth 的 peft_utils.py](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/peft_utils.py#L87) 中的潜在解决方案。
- **训练模型的 GGUF 和 CPU 负载障碍**：一些社区成员在通过 llama.cpp 将 **Llama 3.2** 模型转换为 GGUF 后，因 Prompt 格式不匹配而面临性能下降的问题。
   - 其他人抱怨在本地硬件上出现**奇怪的输出**，强调需要谨慎量化并咨询 [Unsloth 文档](https://docs.unsloth.ai/get-started/all-our-models) 以进行正确的纯 CPU 设置。
- **Stella 被忽视，mixed bread 粉丝增加**：一位用户质疑为什么 **Stella** 很少被推荐，*Mrdragonfox* 承认没有使用它，认为它缺乏广泛的社区动力。
   - 与此同时，**mixed bread** 模型在日常使用中获得了强力支持，人们坚持认为**基准测试 (benchmarking)** 和**微调 (finetuning)** 对实际效果至关重要。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 取得进展与 'o1' 传闻**：一些用户声称 **Gemini 的 Deep Research 模式**在上下文处理和整体实用性方面优于 **Claude 3.5 Sonnet** 和 **GPT-4o**。
   - 针对名为 **'o1'** 的新模型的猜测浮出水面，引发了关于 Perplexity 是否会集成它以实现更广泛 AI 功能的疑问。
- **OpenRouter 接入 Perplexity 模型**：在购买额度后，一位用户发现 **OpenRouter** 提供了直接访问 **Perplexity** 进行问答和推理的途径。
   - 尽管发现了这个选项，该用户仍选择坚持使用另一个供应商，这引发了关于 **OpenRouter** 扩展的激烈讨论。
- **DeepSeek-V3 给人留下深刻印象**：提及 [DeepSeek-V3](https://linux.do/t/topic/312925/70) 表示其已通过 Web 界面和 **API** 提供，引发了对其能力的关注。
   - 测试者将其性能描述为“太强了”，并希望价格能保持稳定，将其与其他安装版本进行了积极对比。
- **印度受 LeCun 启发的飞跃**：印度新推出的一款 **AI 模型**参考了 **Yann LeCun** 的理念，旨在增强类人推理和伦理，引发了对话。
   - 成员们对其影响表示乐观，认为它可能会重塑 **模型训练 (model training)** 并展示 **应用 AI (applied AI)** 的力量。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek V3 强势登场**：中国团队 **DeepSeek** 推出了一个 685B 参数的模型，声称总训练成本为 550 万美元，使用了 260 万个 H800 小时，吞吐量约为 **60 tokens/second**。
   - 像 [这条推文](https://x.com/deepseek_ai/status/1872242657348710721) 展示了其优于更高预算模型的基准测试结果，一些人称其为成本效率的新标杆。
- **ChatGPT 瞄准“无限记忆”**：传闻称 **ChatGPT** 可能很快就能访问所有历史聊天记录，这可能会改变用户依赖广泛对话上下文的方式。
   - 来自 [Mark Kretschmann 的推文](https://x.com/mark_k/status/1871856522143399961) 表明该功能即将推出，引发了关于更深层次和更持续交互的辩论。
- **强化训练提升 LLM 推理能力**：分享的 [YouTube 视频](https://youtu.be/T1SeqBapMBo?si=JVeVYsD1K5CYCI5K) 展示了在不增加额外开销的情况下，使用先进的 RL 方法来完善大语言模型的逻辑。
   - 贡献者引用了 **验证器奖励 (verifier rewards)** 和 **基于模型的 RM**（例如 [@nrehiew_](https://x.com/nrehiew_/status/1872318217395572895)），提出了一种更结构化的训练方法。
- **Anduril 与 OpenAI 合作**：[Anduril Industries 的推文](https://x.com/anduriltech/status/1864390729516327375) 透露了一项合作，将 **OpenAI** 模型与 Anduril 的防御系统相结合。
   - 他们的目标是提升 AI 驱动的国家安全技术，引发了关于军事领域**伦理**和**实践**界限的新辩论。
- **2024 & 2025：合成数据、Agent 与峰会**：[Graham Neubig](https://github.com/All-Hands-AI/openhands-agent-monitor/pull/41) 发表了关于 **2024 年 Agent** 的主旨演讲，而 [Loubna Ben Allal](https://x.com/latentspacepod/status/1871652198956015941) 评述了关于 **合成数据 (Synthetic Data)** 和 **Smol 模型**的论文。
   - 同时，[AI Engineer Summit](http://Latent.Space) 定于 2025 年在纽约市举行，并为关注行业聚会的人士提供了活动日历。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek V3 的大胆亮相**：DeepSeek 发布了 [V3](https://x.com/deepseek_ai/status/1872242657348710721)，该模型在 **14.8 trillion tokens** 上训练，拥有 **60 tokens/second** 的速度（比 V2 快 3 倍），并在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) 上完全**开源**。
   - 讨论重点包括 **Multi-Token Prediction**、新的奖励建模以及关于批判效率的问题，成员们指出其表现优于许多开源模型。
- **Magnitude 685B：DeepSeek 的下一个豪赌**：传闻 DeepSeek 的 **685B LLM** 可能在圣诞节发布，据 [一则推文](https://x.com/simonw/status/1872141432544489731) 暗示，其大小可能**超过 700GB**，目前尚未列出许可证。
   - 社区成员开玩笑说它会让现有解决方案黯然失色，并对在 repo 中未注明明确 **license** 的情况下开源的可行性表示好奇。
- **MCTS 提升推理能力的魔力**：最近的一篇论文 ([arXiv:2405.00451](https://arxiv.org/abs/2405.00451)) 展示了 **Monte Carlo Tree Search (MCTS)** 结合迭代偏好学习如何增强 LLM 的**推理**能力。
   - 它集成了*结果验证*和 **Direct Preference Optimization** 进行策略内精炼，并在**算术**和**常识**任务上进行了测试。
- **DPO vs PPO：竞争愈演愈烈**：一场 **CMU RL 研讨会** 探讨了 LLM 的 *DPO vs PPO* 优化，暗示了在实践中处理 **clip/delta** 约束和 **PRM 偏差** 的稳健方法。
   - 与会者辩论了 **DPO** 是否优于 **PPO**，一篇即将发表在 **ICML 2024** 的论文和一段 [YouTube 视频](https://youtu.be/T1SeqBapMBo?si=srBHIwpVnDC3aX7x) 进一步激发了好奇心。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeepSeek-V3 的双重重击**：[DeepSeek-V3 文档](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) 强调了**大规模 FP8 混合精度训练**，声称成本降低了 **2 个数量级**。
   - 社区成员讨论了该项目的**资金**和**质量**权衡，但认可其在 HPC 工作负载中实现大幅节省的潜力。
- **Triton 在 FP8 转 BF16 上遇到问题**：**SM89** 上从 fp8 到 bf16 的**类型转换问题**导致了 ptx 错误，详见 [Triton 的 GitHub Issue #5491](https://github.com/triton-lang/triton/issues/5491)。
   - 开发者建议使用 `.to(tl.float32).to(tl.bfloat16)` 加上一个哑操作（dummy op）来防止**融合 (fusion)**，同时解决 ptx 错误。
- **TMA 胜过 cp.async**：用户解释说，由于 H100 具有更高的算力，在 **Hopper (H100)** 上进行 GEMM 时，**TMA** 的性能优于 **cp.async**。
   - 他们强调了 **async** 支持、批量调度和边界检查是减少 HPC 内核中寄存器使用的关键特性。
- **无反向传播方法引发 128 次前向传递**：一种新的训练方法声称可以避免**反向传播 (backprop)** 或动量，通过 **128 次前向传递**来估计梯度，且与真实梯度的余弦相似度较低。
   - 虽然它承诺*节省 97% 的能源*，但许多工程师担心其在小型演示设置之外的实用性。
- **ARC-AGI-2 & 1D 任务生成器**：研究人员在共享的 [GitHub 仓库](https://github.com/open-thought/arc-agi-2) 中收集了 **ARC-AGI-2** 实验资源，邀请社区驱动的探索。
   - 他们还展示了 [一维任务生成器](https://github.com/optozorax/arc_1d/)，这些生成器可能会扩展到**二维符号推理**，激发了对基于谜题的 AI 任务的广泛兴趣。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **与 Google News 的播客合作伙伴关系**：成员们提议将 **Google News** 与 AI 生成的播客内容集成，以短篇或长篇形式总结前 **10 条故事**，引发了对交互式问答的兴趣。他们报告称，听众对这种新闻传递与按需讨论的动态结合表现出越来越高的参与度。
   - 几位参与者分享了在实时更新中穿插喜剧桥段的例子，反映了 **AI 驱动的播客** 如何在让观众获取信息的同时保持娱乐性。
- **AI 畅谈人生重大问题**：一位用户展示了一个以幽默方式探讨哲学的 **AI 生成播客**，将其描述为“smurf-tastic banter”（蓝精灵式的逗趣），带来耳目一新的转折。这种形式将幽默与反思性对话相结合，暗示了对喜爱智力乐趣的受众具有更广泛的吸引力。
   - 其他人称其为传统谈话节目的生动替代方案，强调了**自然听感的 AI 语音**既能带来娱乐，又能引发深思。
- **15 分钟了解 Pathfinder**：一位参与者生成了一个简洁的 **15 分钟** 播客，总结了包含 **6 本书** 的 **Pathfinder 2** 战役，为游戏主持人提供了快速的剧情概览。他们在故事情节亮点与相关技巧之间取得了平衡，使玩家能够迅速沉浸在桌面游戏内容中。
   - 这种方法激发了人们对短篇桌面游戏回顾的热情，预示着 AI 引导的叙事与角色扮演社区之间潜在的协同效应。
- **Akas 连接 AI 播客主与其受众**：一位爱好者介绍了 **Akas**，这是一个用于分享 AI 生成播客并发布个性化 RSS 订阅源的网站，详见 [Akas: share AI generated podcasts](https://akashq.com.)。他们将其定位为 AI 驱动节目与每位主持人个人声音之间的平滑连接，将创意想法桥接到更广泛的受众。
   - 一些人预测未来会扩展并统一像 NotebookLM 这样的工具，鼓励用户驱动的 AI 剧集走向更广阔的平台并激发进一步的协作。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **报告就绪：LlamaParse 助力 Agent 工作流**：有人发布了一个新的 **Report Generation Agent**，它使用 [LlamaParse](https://t.co/o5jhvipERf) 和 **LlamaCloud** 从 PDF 研究论文中构建格式化报告，并参考了 [演示视频](https://t.co/0IHLaXZxGy)。
   - 他们强调了该方法在自动化多论文分析方面的潜力，提供了与输入模板的强大集成。
- **DocumentContextExtractor 削减成本**：对话集中在使用 **DocumentContextExtractor** 进行批处理，以削减 **50%** 的费用，允许用户在非高峰时段处理任务。
   - 这种方法无需保持 Python 脚本持续运行，让个人可以随时查看结果。
- **LlamaIndex 中的 Tokenization 纠葛**：参与者批评 **LlamaIndex tokenizer** 缺乏解码支持，对功能集不完整表示失望。
   - 虽然推荐了块切分（chunk splitting）和大小管理，但一些人开玩笑说干脆取消截断功能，把提交超大文件的责任推给用户。
- **Unstructured RAG 进阶**：一篇博客详细介绍了使用 **LangChain** 和 **Unstructured IO** 构建的 **Unstructured RAG** 如何比旧的检索系统更有效地处理图像和表格等数据，参考了 [此指南](https://hub.athina.ai/athina-originals/end-to-end-implementation-of-unstructured-rag/)。
   - 它还描述了使用 **FAISS** 进行 PDF 嵌入（embeddings），并建议使用 **Athina AI** 评估策略以确保 RAG 在真实环境中的准确性。
- **LlamaIndex 文档与工资单 PDF**：一些人正在寻找获取 PDF 和 Markdown 格式 **LlamaIndex** 文档的方法，而另一些人则在努力使用 **LlamaParse** 高级模式解析工资单 PDF。
   - 讨论得出结论，生成这些文档是可行的，且 **LlamaParse** 在完全配置后可以处理工资单任务。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **优化器状态缺失 (Optimizers on the Loose)**：一些人在 [Hugging Face checkpoints](https://huggingface.co/EleutherAI/pythia-2.8b/tree/main) 中发现了缺失的优化器状态，引发了对 checkpoint 完整性的疑问。
   - 其他人确认 checkpointing 代码通常会保存这些状态，使得真实原因尚不明确。
- **VLM 微调热潮 (VLM Fine-Tuning Frenzy)**：工程师们正在处理 **LLaVA**、**Qwen-VL** 和 **InternVL** 微调脚本的模型特定细节，并指出每种方法都有所不同。
   - 他们分享了 [LLaVA](https://github.com/haotian-liu/LLaVA) 作为热门参考，强调遵循正确的方法论对结果至关重要。
- **追求更低延迟 (Chasing Lower Latency)**：参与者汇总了一系列针对 CUDA 或 Triton 级别优化的方法，旨在缩短 LLM 推理时间。
   - 他们还指出了开源解决方案的进展，在 function calling 等任务中，这些方案有时能超越 GPT-4。
- **GPT-2 惊人的首个 Token (GPT-2’s Shocking First Token)**：在 **GPT-2** 中，初始 token 的激活值飙升至 3000 左右，而后续 token 通常仅为 100。
   - 关于 GPT-2 中是否存在 **BOS token** 的争论仍在继续，一些人断言它只是在默认情况下被省略了。
- **EVE 引发无编码器架构的好奇 (EVE Sparks Encoder-Free Curiosity)**：研究人员探索了 [EVE](https://github.com/baaivision/EVE)，这是一个专注于视频的无编码器视觉语言模型，避开了 CLIP 风格的架构。
   - 与此同时，**Fuyu** 模型系列在实际性能提升方面面临质疑，引发了对编码器效率更多见解的呼吁。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Lean 悬赏与 BITCAST 探索 (Lean Bounty and the BITCAST Quest)**：成员们应对了 **Lean bounty** 证明挑战，参考了 [tinygrad notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md) 进行指导。他们还讨论了实现 **BITCAST const folding** 以优化编译时间。
   - 有人提出了关于实现 **BITCAST const folding** 以进行编译时优化的兴趣，并询问相关代码存放在哪个目录。另一位用户建议参考旧的 PR 以获取如何进行的示例。
- **Tinygrad 与 PyTorch 的对决 (Tinygrad vs. PyTorch Face-Off)**：有人报告称，在 CUDA 上进行前向传播时，**Tinygrad** 耗时 **800ms**，而 **PyTorch** 仅需 **17ms**，这促使人们尝试通过 jitting 进行改进。社区成员预期 **beam search** 会带来并发收益，并重申稳定后的方法可以达到或超过 PyTorch 的速度。
   - 他们承认速度差异可能源于不同的 CUDA 设置和系统配置。一些参与者建议加大 jitting 力度以缩小性能差距。
- **匹配引擎中的重写之争 (Rewrite Rumble in Matching Engines)**：参与者探索了**匹配引擎性能悬赏**，并链接到了 [tinygrad/tinygrad#4878](https://github.com/tinygrad/tinygrad/issues/4878) 的待解决问题。
   - 一位用户澄清了他们对 **rewrite** 部分的关注，并参考了虽然过时但仍能指导方案方向的 PR。
- **输入处理的小故障 (Input Handling Hiccups)**：一位用户指出在循环中重新创建 **input tensor** 会严重拖慢 **Tinygrad**，尽管输出正确，但还遇到了 CUDA 分配器的属性错误。作为回应，PR **#8309** 的更改已被合并以修复这些问题，强调了回归测试对稳定性能的重要性。
   - 深入调查发现 `tiny_input.clone()` 触发了 CUDA 内存分配器中的错误。贡献者一致认为需要更多测试来防止循环输入创建中的回归问题。
- **通过 Kernel 缓存提升 GPU 收益 (GPU Gains with Kernel Caching)**：聊天中提到了使用驱动版本 **535.183.01** 和 **CUDA 12.2** 的 **RTX 4070** GPU，引发了对开源驱动的关注。关于 **beam search** 缓存的讨论确认了 kernel 会被重用以提高速度，并有望在类似系统间共享这些缓存。
   - 与会者推测潜在的驱动不匹配可能会限制性能，并敦促通过 debug 日志进行确认。一些人建议分发已编译的 beam search kernels，以加快在匹配硬件上的设置。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **体验 CMD-R 与 R7B 的起伏**：成员们讨论了 **CMD-R** 即将到来的变化，并对 **R7B** 中出现的“两个答案 (two ans)”怪异现象表示好奇，引用了一张暗示意外更新的共享图片。
   - 他们开玩笑说这种奇怪的结果很少出现，在 [社区讨论](https://discord.com/channels/954421988141711382/1168411509542637578/1321813802810998795) 中，有人称其为 *“一个值得研究的喜剧性故障”*。
- **精打细算 Re-ranker 定价**：**Re-ranker** 的成本结构引起了关注，特别是如 [Cohere 定价页面](https://cohere.com/pricing) 所示，每 1M tokens 输入需 **$2.50**，输出需 **$10.00**。
   - 相关问题激发了人们对团队如何为高使用量制定预算的兴趣，一些人将其与其他替代方案进行了比较。
- **LLM University 取得进展**：Cohere 推出了 [LLM University](https://cohere.com/llmu)，提供 **NLP** 和 **LLMs** 的专业课程，旨在增强企业级 AI 专业知识。
   - 参与者给出了热烈的反馈，称赞其资源结构合理，并指出用户可以利用这些材料进行 *快速技能扩展*。
- **Command R & R+ 在多步任务中占据主导地位**：**Command R** 提供 128,000 token 的上下文容量和高效的 **RAG** 性能，而 **Command R+** 则展示了顶级的多步工具使用能力。
   - 参与者将其归功于其多语言覆盖（10 种语言）和先进的训练细节，特别是在 [cmd-r-bot 频道](https://discord.com/channels/954421988141711382/1168578374038470656/1321387523494379615) 中提到的 *具有挑战性的生产需求*。
- **语音、VAD 与音乐融合 AI 魔力**：一位 **AI Engineer** 展示了一个利用 **DNN-VAD** 的 **Voice to Voice** 聊天应用，并分享了使用 stereo-melody-large 模型根据文本提示生成的音乐。
   - 他们邀请合作者，表示 *“我想与你合作”*，并送上了 **圣诞快乐** 的问候以保持活跃的氛围。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **io_uring 的好奇与网络优化**：成员们探讨了 **io_uring** 如何提升网络性能，参考了 man pages 作为起点，并承认目前熟悉程度有限。
   - 社区推测 **io_uring** 可能会简化异步 I/O，并呼吁通过实际基准测试来确认其协同效应。
- **StringRef 异常与负长度崩溃**：当 **StringRef()** 接收到负长度时发生了崩溃，指向 **memcpy** 中缺失的长度检查。
   - 一位用户建议改用 **StringSlice**，强调了 **StringRef** 在处理长度验证时的风险。
- **EOF 测试与 Copyable 评价**：用户确认 **read_until_delimiter** 正确触发了 EOF，引用了 [GitHub commits](https://github.com/mahiro21h/mojo/commits/fix-input-segfaults-on-eof)。
   - 对话重点讨论了 **Copyable** 和 **ExplicitlyCopyable** 特性，Modular 论坛上也出现了一些潜在的设计调整。
- **Mojo 周边与 Modular 商品热潮**：成员们炫耀了他们的 **Mojo swag**，对海外邮寄表示感谢，并分享了全新装备的照片。
   - 其他人称赞了 **Modular 的商品**，包括 T 恤的质量和“硬核”贴纸设计，进一步激发了品牌热情。
- **Modular Kernel 查询与 MAX 对比 XLA**：一位用户询问了关于 Modular 栈的专用 **kernel**，暗示可能存在的性能改进。
   - **MAX** 与 **XLA** 进行了对比，引用“JAX 编译时间过长”作为考虑替代编译器策略的原因。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **PyN8N 获得 Node Wiz 支持**：爱好者们注意到 [PyN8N](https://pypi.org/project/pyn8n/) 集成了 AI 来构建自定义工作流，尽管一些用户报告了与广告拦截器相关的加载问题。
   - 他们强调了 README 的愿景基调，并建议切换浏览器或禁用扩展程序以解决这些网站拦截错误。
- **DSPy 与 DSLModel 协作**：社区成员发现 **DSPy** 通过 **DSLModel** 扩展了功能，允许使用高级特性以获得更好的性能。
   - 他们认为这种方法在保持复杂数据工作流精简的同时，减少了代码开销。
- **NotebookLM 的行内溯源引发好奇**：一位用户询问 **NotebookLM** 是如何实现行内溯源（inline sourcing）的，并指出目前缺乏详细的解答。
   - 他们寻求对底层实现的更多见解，但对话提供的后续信息有限。
- **Jekyll 术语表获得 DSPy 助力**：有人分享了一个 [Jekyll 脚本](https://gist.github.com/dbreunig/3cef9293cb253f9192d5b4974c1367a3)，用于生成关键术语表，并使用 **DSPy** 进行 LLM 交互。
   - 他们完善了诸如 **Artificial General Intelligence** 之类的条目，并指出了 *long description parameter* 的潜在改进方向。
- **Typing.TypedDict 与 pydantic 的纠葛**：成员们发现了用于类型化字段的 `typing.TypedDict`，并承认其在 Python 使用场景中的复杂性。
   - 他们还讨论了使用 **pydantic** 处理多实例输出数组，旨在实现更精细的布局。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书困惑与严格的表单要求**：成员们确认证书将于 **1 月底**发放，如[此处](https://discord.com/channels/1280234300012494859/1293323662300155934/1321147373652541511)所述。
   - 一位参与者询问如果没有填写 **Certificate Declaration Form** 是否仍能获得证书，得到的答复是该表单为强制要求，没有例外。
- **LLM Agents MOOC 春季课程热度**：社区讨论透露，下一期 LLM Agents 课程将于 **春季** 开始，与当前课程的结束时间衔接。
   - 参与者表现出极大的兴趣，参考了证书流程，并希望课程更新能按时发布。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 开创性的像素级精度**：[Open Interpreter API](https://openinterpreter.com/) 为 UI 自动化提供了近乎 **像素级（pixel-perfect）** 的控制，包括用于文本识别的 **OCR**，并在 [Python 脚本](https://api.openinterpreter.com/v0/point/)中提供了使用示例。
   - 一位社区成员提到 **OCR** 似乎无法正常工作，而其他人则询问了 **桌面版** 的发布时间表，显示出对进一步开发的广泛兴趣。
- **语音对语音聊天与 QvQ 协同**：一位 AI 工程师介绍了一款具有从文本提示进行 **音乐生成（Music Generation）** 功能的 **语音对语音（Voice to Voice）聊天应用**，寻求与其他生成式 AI 爱好者的合作。
   - 另一位用户询问 **QvQ** 在 Open Interpreter 的 **OS 模式** 下将如何运作，暗示了将 **语音** 与 **系统级** 任务连接起来的可能性。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **复制按钮难题**：一位成员注意到 GPT4All 聊天界面 UI 中缺少用于 AI 生成代码的专用 **“复制”按钮**，从而引发了关于 UI 改进的讨论。
   - 他们对任何变通方案的建议表示感谢，并强调代码复制的便利性在开发者需求中排名很高。
- **键盘快捷键问题**：社区成员确认基于鼠标的剪切和粘贴在聊天 UI 或配置页面中无法工作，这让依赖右键操作的用户感到沮丧。
   - 他们澄清说 **Control-C** 和 **Control-V** 仍然有效，为复制代码片段提供了备选方案。
- **对新模板的好奇**：一位成员用法语询问是否有人尝试过 **使用新模板编写内容**，表明了英语语境之外的多语言采用情况。
   - 他们希望获得安装后步骤的反馈，尽管交流中没有出现具体的结果或共享示例。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Mega Audio Chunks for TTS**: 一位成员寻求从长达一小时的海量音频文件中构建 **TTS dataset** 的建议，并询问了有关正确分割这些文件的工具。
   - 他们的目标是寻找一种在减少人工劳动的同时保持质量的方法，重点关注处理大文件大小的 **audio segmentation** 方法。
- **Whisper Splits the Script**: 另一位参与者提议使用 **Whisper** 进行句子级分割，认为这是为 TTS 任务准备音频的一种实用方式。
   - 他们强调了 **Whisper** 如何简化分割流程，在缩短制作时间的同时保留一致的句子边界。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **HPC MLOps frameworks in the spotlight**: 一位成员请求一个稳定且具有成本效益、且开销最小的 **HPC** **ML ops framework**，并指出 [Guild AI](https://guild.ai/) 是一个可能的选择。
   - 他们质疑 **Guild AI** 的可靠性，并倾向于采用自托管方式，理由是不喜欢 SaaS 解决方案。
- **Server chores spark talk of a DIY ops tool**: 安装和维护负担的增加使他们对运行用于 MLOps 任务的专用服务器产生顾虑。
   - 他们表示，如果能避免沉重的服务器管理工作，愿意自己编写一个简单的 ops 框架。



---


**Axolotl AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1321935220630622258)** (1 messages): 

> `Windsurf AI` 


- **Engineers Share Windsurf Creation Insights**: 一段新视频展示了我们优秀的工程师讨论 **Windsurf** 创作背后的创新策略。他们解释了在此过程中如何打破每一项行业惯例。
   - 您可以在 [Windsurf's Twitter](https://x.com/windsurf_ai/status/1872375661542920424) 上观看完整视频。
- **Happy Holidays Message**: 公告中包含了来自 Windsurf 团队的温馨节日祝福，以积极的基调庆祝节日。这一举动强调了节日期间的社区精神。



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1872375661542920424">Tweet from Windsurf (@windsurf_ai)</a>: What exactly is Windsurf? Watch how we dared to innovate by breaking every industry convention 🌊

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1321250135900164146)** (433 条消息🔥🔥🔥): 

> `Windsurf 性能问题、Cascade 基础模型担忧、远程主机集成、用户体验反馈、登录与 API 错误` 


- **Windsurf 面临性能问题**：许多用户报告在 Windsurf 中遇到 **错误** 和 **无响应** 行为，尤其是在最近的更新之后，消息经常导致无响应或出现 “Error generating response” 等错误。
   - 一些用户指出，即使在宣布改进之后，他们仍继续面临 IDE 的问题，这导致了挫败感，特别是对于那些使用 **PRO** 计划的用户。
- **对 Cascade 基础模型的担忧**：几位用户对 **Cascade Base 模型** 表示不满，认为与 **Claude 3.5** Sonnet 相比，它不足以处理复杂的编码任务。
   - 虽然一些用户声称在添加全局规则后有所改进，但其他用户没有看到明显的增强，并觉得基础模型变得越来越难以依赖。
- **远程主机集成问题**：连接到 **SSH Remote Hosts** 的用户经历了显著的性能延迟，操作虽然执行了但在 Cascade 更新之前视觉上存在滞后。
   - 这导致了工作流程的混乱，因为用户发现尽管命令正确执行，但界面并未及时反映更改。
- **界面用户体验反馈**：几位用户报告了点击 **chat history**（聊天记录）的问题，一些人发现他们只能通过键盘导航而不是直接点击来访问之前发送的消息。
   - 这表明存在一个持久的界面 Bug，令用户感到沮丧并阻碍了应用程序内的生产力。
- **社区参与和支持请求**：用户讨论了针对持续问题提交支持工单的重要性，并强调了在投入大量资金后无法有效使用 IDE 的挫败感。
   - 社区建议，改进有关更新和停机的 **communication channels**（沟通渠道）将大大提升用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/youre-kidding-kyle-broflovski-sheila-broflovski-harrison-yates-south-park-gif">未找到标题</a>: 未找到描述</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=0071269e-afcd-47e6-9409-2d654db5c5f6">未找到标题</a>: 未找到描述</li><li><a href="https://docs.codeium.com/getstarted/overview">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/spitting-coffee-fbi-agent-gif-24958047">Spitting Coffee Fbi Agent GIF - Spitting Coffee Fbi Agent - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/youre-kidding-kyle-broflovski-sheila-broflovski-harrison-yates-south-park-gif-20884010">Youre Kidding Kyle Broflovski GIF - Youre Kidding Kyle Broflovski Sheila Broflovski - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=3ad9aa49-ad6d-4f02-81ef-448529f4f954">未找到标题</a>: 未找到描述</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://youtu.be/pOvI02of5oo">Cascade Memories: Personalize Windsurf with Custom Rules</a>: 一劳永逸。Cascade Memories 允许您创建自定义规则并自动应用它们，从而节省时间并保持工作流程顺畅。下载...</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://github.com/SchneiderSam/awesome-windsurfrules">GitHub - SchneiderSam/awesome-windsurfrules: 📄 一个精选的 awesome global_rules.md 和 .windsurfrules 文件列表</a>: 📄 一个精选的 awesome global_rules.md 和 .windsurfrules 文件列表 - SchneiderSam/awesome-windsurfrules</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf">GitHub - ichoosetoaccept/awesome-windsurf: 用于 Windsurf 代码编辑器的出色资源集合</a>: 用于 Windsurf 代码编辑器的出色资源集合 - ichoosetoaccept/awesome-windsurf
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1321207226060243076)** (869 条消息🔥🔥🔥): 

> `Windsurf 问题、Cascade 性能、用户体验、AI 模型性能、项目开发挑战`

- **Windsurf 面临频繁停机**：许多用户报告了 Windsurf 持续出现的问题，经历了响应缓慢或 Cascade 完全没有交互的情况。
   - 用户对在没有获得足够输出的情况下消耗 Token 表示沮丧，引发了关于平台稳定性的讨论。
- **Cascade 性能挣扎**：几位用户批评 Cascade Base 无法正确遵循 Prompt，经常误解简单的命令（如 'git commit'）。
   - 这导致了极大的挫败感，特别是在与助手协作了相当长时间后，它仍未能达到预期。
- **关于额度消耗的担忧**：对话揭示了一个普遍的担忧，即针对无响应的查询损失了额度，使用户对该服务对 Token 的依赖感到不满。
   - 个人对 Cascade 的效率表示失望，特别是在处理大型代码库和提供上下文准确的建议方面。
- **用户体验与建议**：用户分享了使用替代工具的经验，表达了对 Windsurf 和 Cascade 改进及提高效能的希望。
   - 一些用户概述了简化交互的方法，强调需要更精确的命令和 AI 更好的性能。
- **学习曲线与未来发展**：许多用户承认，有效使用 Windsurf 存在陡峭的学习曲线，尤其是对于大型项目。
   - 尽管存在挫折，一些用户仍对 Windsurf 在持续改进后辅助其开发流程的潜力保持乐观。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://ezbook23.vercel.app/">iAircon - Easy Aircon Booking</a>: 未找到描述</li><li><a href="https://desenrola.netlify.app/">Dr. Desenrola</a>: 未找到描述</li><li><a href="https://tenor.com/view/gjirlfriend-gif-14457952604098199169">Gjirlfriend GIF - Gjirlfriend - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/uhh-cat-meow-meowmeow-confused-gif-8057852975940592422">Uhh Cat GIF - Uhh Cat Meow - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bye-okay-slide-gif-15172486">Bye Okay GIF - Bye Okay Slide - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/popcat-wen-gif-23885304">Popcat Wen GIF - Popcat Wen - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/this-is-fine-gif-24177057">This Fine GIF - This Is Fine - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/supervisor-speak-to-the-manager-manager-bubble-gum-princess-adventure-time-gif-9822847">Speak To The Manager GIF - Supervisor Speak To The Manager Manager - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/@YifanBTH/playlists.">Yifan - Beyond the Hype</a>: 一位从计算机科学家转型为技术创始人的见解与吐槽。</li><li><a href="https://www.helicone.ai/status/provider/Anthropic">Is Claude Down? Live Status & Performance Monitor - Helicone</a>: 检查 Claude 或 Anthropic API 是否正常工作。提供 Claude 3.5 Sonnet、Claude 3 Opus、Claude 2.1 和 Claude Instant 的实时状态监控、当前故障、API 可用性及性能指标。</li><li><a href="https://tenor.com/view/contemplating-thinking-eating-chewing-eat-gif-19268514">Contemplating Thinking GIF - Contemplating Thinking Eating - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://bolt.new/~/sb1-jxglvk">Luxury Virtual Market Game (forked)</a>: 未找到描述</li><li><a href="https://tenor.com/view/reaction-thinking-idk-think-wait-gif-7959205027699559349">Reaction Thinking GIF - Reaction Thinking Idk - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/daria-monke-orangutan-demand-youfr-gif-27135853">Daria Monke GIF - Daria Monke Orangutan - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/karen-being-a-karen-calling-the-police-calling-the-cops-gif-27252855">Karen Being A Karen GIF - Karen Being A Karen Calling The Police - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/snoopy-snoopy-and-woodstock-woodstock-peanuts-angry-gif-5431878322572996122">Snoopy Snoopy And Woodstock GIF - Snoopy Snoopy and woodstock Woodstock - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/balthazar-crazy-family-guy-fou-gif-11775947423386520412">Balthazar Crazy GIF - Balthazar Crazy Family guy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/viva-la-dirt-league-vldl-gif-19768362">Viva La GIF - Viva La Dirt - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/goodbye-died-gif-13279499">Goodbye Died GIF - Goodbye Died - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/meme-loafcat-crypto-simpsons-gif-14007875133439353847">Meme Loafcat GIF - MEME LOAFCAT CRYPTO - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: 联系 Codeium 团队获取支持，并了解更多关于我们企业级服务的信息。</li><li><a href="https://pak-otp.vercel.app/">Your App Name</a>: 未找到描述</li><li><a href="https://status.openai.com/">OpenAI Status</a>: 未找到描述</li><li><a href="https://tenor.com/view/jeeks-balou-guigui-nok-ptitlem-gif-26951285">Jeeks Balou GIF - Jeeks Balou Guigui - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://Lovable.dev">Lovable</a>: 仅通过聊天界面构建软件产品</li><li><a href="https://www.youtube.com/watch?v=4bQDDrUhtSE"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/shanalikhan/code-settings-sync">GitHub - shanalikhan/code-settings-sync: 🌴💪 Synchronize your Visual Studio Code Settings Across Multiple Machines using GitHub GIST 💪🌴</a>: 🌴💪 使用 GitHub GIST 在多台机器之间同步您的 Visual Studio Code 设置 💪🌴 - shanalikhan/code-settings-sync</li><li><a href="https://github.com/bungrudi/mikkadb">GitHub - bungrudi/mikkadb</a>: 通过创建账户为 bungrudi/mikkadb 的开发做出贡献</li>

n GitHub.
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1321216304581181544)** (744 messages🔥🔥🔥): 

> `DeepSeek V3 性能、Cursor IDE 与 DeepSeek 集成、Agent 模式与 token 消耗、Next.js 中的 UI 设计挑战、不同 AI 模型的对比` 


- **DeepSeek V3 给用户留下深刻印象**：用户发现 DeepSeek V3 非常高效，可与 Sonnet 3.5 媲美，并称赞其能以极低的精力和更低的成本处理任务。
   - 它在生成代码和进行分析方面的能力，引发了关于将其集成到 Cursor IDE 中的讨论。
- **Cursor IDE 的 token 限制**：用户注意到 Cursor IDE 的 context window 有限，这在生成代码或分析大型项目时可能会影响性能。
   - 这促使人们讨论 DeepSeek 和 Cline 等不同模型如何比 Cursor 更有效地处理 token 和 context。
- **Next.js 中的 UI 设计困境**：开发者们表达了在使用 Next.js 进行 UI 设计时的挫败感，强调了利用 Claude 进行设计任务时遇到的困难。
   - 建议包括使用特定的库和组件（如 shadcn），并将文档嵌入到 context 中以获得更好的结果。
- **特定模型功能的挑战**：提到了不同 AI 模型如何与代码交互，以及功能实现如何根据模型的训练内容而有所不同。
   - 对话强调了使用正确版本的框架的重要性，以及使用 embeddings 或 RAG 提升性能的潜在好处。
- **OpenAI 性能波动**：在近期出现性能问题后，用户对 OpenAI 的可靠性普遍表示担忧，并强调了替代模型所提供的改进。
   - 一些人主张串联测试多个模型，以在性能和成本效益之间取得平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/alumae-alumaeyy-gubby-gubby-mewing-meme-gif-6805644913242328211">Alumae Alumaeyy GIF - Alumae Alumaeyy Gubby - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.cursor.com/downloads">下载 | Cursor - AI 代码编辑器</a>: 选择您的平台下载最新版本的 Cursor。</li><li><a href="https://platform.deepseek.com/">DeepSeek 平台</a>: 加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721?t=vpoi2yGx6psx69xwLTKnxA&s=19">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 每秒 60 个 token（比 V2 快 3 倍！）💪 增强的能力 🛠 保持 API 兼容性 🌍 完全开源的模型和论文 🐋 1/n</li><li><a href="https://platform.deepseek.com">DeepSeek 平台</a>: 加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://aws.amazon.com/ses/">云邮件发送服务 - Amazon Simple Email Service - AWS</a>: 未找到描述</li><li><a href="https://forum.cursor.com">Cursor - 社区论坛</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://platform.deepseek.com/usage">DeepSeek 平台</a>: 加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://uiverse.io/elements">5685 个 UI 元素：CSS 和 Tailwind</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/">您的第一次 API 调用 | DeepSeek API 文档</a>: DeepSeek API 使用与 OpenAI 兼容的 API 格式。通过修改配置，您可以使用 OpenAI SDK 或兼容 OpenAI API 的软件来访问 DeepSeek API。</li><li><a href="https://cursor.directory/">Cursor 目录</a>: 为您的框架和语言查找最佳的 Cursor 规则
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1321981397451280508)** (1 条消息): 

> `Aider v0.70.0 发布，分析功能选择性加入，错误处理改进，模型支持增强` 


- **Aider v0.70.0 全面支持 o1 模型**：最新发布的 **Aider v0.70.0** 现在包含对 **o1 模型** 的全面支持，并提供通过 **uv** 的新安装方法，支持单行命令安装。
   - 此外，它通过遵循 `--subtree-only` 改进了文件监听（watch files）功能，并包含改进的提示词（prompting）以提高模型的可靠性。
- **分析功能选择性加入（Opt-in）特性推出**：Aider 计划请求 **10%** 的用户加入分析功能，以增强功能洞察。
   - 此举旨在收集用户交互数据并提升整体用户体验。
- **交互式命令的错误处理改进**：此版本在使用通过 `/load` 或 `--load` 的交互式命令时带来了更好的错误处理，增强了用户导航体验。
   - 系统现在可以优雅地处理 git 路径名中的 **unicode 错误**，以防止中断。
- **改进的元数据和 Bug 修复**：引入了对模型元数据中 **gemini 模型** 名称的修复，以及对 **auto-suggest** 的 Bug 修复，优化了工具性能。
   - 这些增强功能有助于实现更流畅的使用体验并减少中断。
- **Aider 的贡献备受关注**：根据 git 提交历史，**Aider** 报告称其编写了此版本中 **74%** 的代码。
   - 这一统计数据强调了该工具的自给自足能力和不断进化的能力。



**提到的链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1321210146902245497)** (557 条消息🔥🔥🔥): 

> `DeepSeek V3 对比 O1 Pro，模型对比：Claude 对比 DeepSeek，在 Aider 中使用 DeepSeek，代码实现中的挑战，LLM 的上下文限制` 


- **DeepSeek V3 与 O1 Pro 的性能对比**：用户注意到 DeepSeek V3 比其前代产品快三倍，并在编程任务中展现出极具前景的能力，在某些场景下表现优于 Sonnet 3.5。
   - 许多人正在探索将 DeepSeek 用于各种编程任务的潜力，尽管存在一些局限性，但对其性能总体感到满意。
- **模型对比：Claude 对比 DeepSeek**：用户对 Claude 模型目前的表现持怀疑态度，尤其是最近发布的 3.5 Haiku，一些用户认为它与 DeepSeek V3 等替代方案相比有所欠缺。
   - DeepSeek 在某些模式下提供完整文件输出的能力获得了积极反馈，尽管响应时间慢仍然是一个缺点。
- **结合 Aider 和 DeepSeek 实现代码变更**：用户希望结合 Aider 的能力与 DeepSeek 来更高效地自动化和实现代码变更，寻求两种模型之间的协同工作流。
   - 用户希望在 Aider 和 DeepSeek 的未来更新中看到改进的自主功能和更好的上下文学习能力。
- **代码实现中的挑战**：用户对 DeepSeek 在处理复杂编程任务和在较长编码会话中维持上下文方面的局限性表示担忧。
   - 用户表达了对能够更好理解大型代码库并提供全面更新或重构而无需持续人工干预的模型的渴望。
- **LLM 的上下文限制**：DeepSeek V3 的上下文限制为 64k tokens，在处理冗长的文档或复杂的代码库时会导致挫败感。
   - 这一限制引发了关于需要能够无缝管理更大上下文并提供有意义的、上下文相关响应的模型的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/deepseek_ai/status/1872242657348710721">DeepSeek (@deepseek_ai) 的推文</a>: 🚀 介绍 DeepSeek-V3！迄今为止最大的飞跃：⚡ 60 tokens/second（比 V2 快 3 倍！）💪 增强的能力 🛠 API 兼容性保持不变 🌍 完全开源的模型和论文 🐋 1/n</li><li><a href="https://x.com/i/status/1815969489990869369">Alex Cheema - e/acc (@alexocheema) 的推文</a>: 2 台 MacBooks 就够了。使用 @exolabs_ 家庭 AI 集群在 2 台 MacBooks 上分布式运行 Llama 3.1 405B</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/im-the-captain-now-im-the-boss-captain-gif-14172461">我现在是船长，我是老板 GIF - 我现在是船长，我是老板 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://dearrow.ajay.app/">DeArrow - 一个用于优化标题和缩略图的浏览器扩展</a>: DeArrow 是一个浏览器扩展，用于将 YouTube 上的标题和缩略图替换为社区创建的准确版本。告别点击诱饵。</li><li><a href="https://agenticengineer.com/principled-ai-coding">Agentic Engineer - 构建“活”的软件</a>: 构建“活”的软件。你掌握 prompts、prompt chains、AI agents 和 agentic workflows 的指南。</li><li><a href="https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard">BigCodeBench 排行榜 - bigcode 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://aider.chat/docs/scripting.html">脚本化 Aider</a>: 你可以通过命令行或 Python 对 Aider 进行脚本编写。</li><li><a href="https://tenor.com/view/genius-think-be-clever-be-smart-gif-10617231">天才思考 GIF - 天才思考，变得聪明 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/2024/06/02/main-swe-bench.html">Aider 在 SWE Bench 和 SWE Bench Lite 中均达到 SOTA</a>: Aider 在最近获得 Lite 版本 SOTA 后，又在 SWE Bench 主榜单中创下 SOTA。</li><li><a href="https://aider.chat/docs/faq.html#why-is-the-llm-speaking-to-me-in-an-unexpected-language">FAQ</a>: 关于 Aider 的常见问题。</li><li><a href="https://huggingface.co/Qwen/QVQ-72B-Preview">Qwen/QVQ-72B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://aider.chat/2024/12/21/polyglot.html">o1 在 Aider 新的多语言排行榜中夺冠</a>: o1 在 Aider 新的、更具挑战性的多语言代码基准测试中获得了最高分。</li><li><a href="https://x.com/ivanfioravanti/status/1870926281736659413">Ivan Fioravanti ᯅ (@ivanfioravanti) 的推文</a>: 使用 @exolabs 的 exo 在 M2 Ultra 和 2 台 M4 Max 之间建立 Thunderbolt 连接。让我们用 Llama 3.2 405B 做一些测试！</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hm4959/benchmark_results_deepseek_v3_on_livebench/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>: 未找到描述</li><li><a href="https://youtu.be/GBR6pHZ68Ho"> - YouTube</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/2eNVV0ouBxg"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/SkmrUWyZThQ?si=GpGqzOHydrfhQr4v"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qqXkGqzsFio"> - YouTube</a>: 未找到描述</li><li><a href="https://www.apple.com/shop/buy-mac/mac-mini/apple-m4-pro-chip-with-12-core-cpu-16-core-gpu-24gb-memory-512gb?afid=p238%7CsyAHmzAxH-dc_mtid_1870765e38482_pcrid_724099485254_pgrid_110391416539_pntwk_g_pchan__pexid__ptid_kwd-865769501_&cid=aos-us-kwgo-mac--slid---product-">Mac mini</a>: 搭载 M4 和 M4 Pro 芯片的 Mac mini。专为 Apple Intelligence 打造。配备前后端口。提供分期付款选项。立即从 apple.com 购买。</li><li><a href="https://www.apple.com/shop/buy-mac/mac-mini/apple-m4-pro-chip-with-12-core-cpu-16-core-gpu-24gb-memory-512gb?afid=p238%257CsyAHmzAxH-dc_mtid_1870765e38482_pcrid_724099485254_pgrid_110391416539_pntwk_g_pchan__pexid__ptid_kwd-865769501_&cid=aos-us-kwgo-mac--slid---product-">Mac mini</a>: 搭载 M4 和 M4 Pro 芯片的 Mac mini。专为 Apple Intelligence 打造。配备前后端口。提供分期付款选项。立即从 apple.com 购买。</li><li><a href="https://github.com/richardanaya/colossus/">GitHub - richardanaya/colossus: 一个用于控制 Aider 的实时语音 AI 工具</a>: 一个用于控制 Aider 的实时语音 AI 工具。通过在 GitHub 上创建账号为 richardanaya/colossus 的开发做出贡献。</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: 代码</a>: 代码。通过在 GitHub 上创建账号为 robert-at-pretension-io/mcp 的开发做出贡献。</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: R

在家中使用日常设备运行你自己的 AI 集群 📱💻 🖥️⌚</a>: 在家中使用日常设备运行你自己的 AI 集群 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.amazon.com/Lenovo-00KG133-Nvidia-Tesla-K80/dp/B01A3VGAGS?crid=1CMGVX3FG8UI9&dib=eyJ2IjoiMSJ9.NQxBWkkc6BLtNRAxRAfQgzvWmExBfvGWMYy24oGZGRc6hwRD_DEa7qj9PHUVGfrGH3TZAIzhSvQ-bEf8VJ6W3n-EgDzpMsFozhLaQBlSWmeTsAQjgX8mv0dUEaIs4FIduiXnQuRTQExQpDQtwRNl4d5wIRp1mw28t2nZX5rf0ED6VlXYUzB-Cg5sUEb0TjqrHlkNXfdvttvt8DA6BZ8w003lvsKOC56wIacHsF2AUc4.whVOarsaA_4hRB5PqAcZ6mC2pdnBQSrgG_9iGaCmT0M&dib_tag=se&keywords=NVIDIA+Tesla+K80+GPU&qid=1735193115&sprefix=nvidia+tesla+k80+gpu,aps,351&sr=8-5">Amazon.com: Nvidia Tesla K80 : 电子产品</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1321222640886747287)** (49 messages🔥): 

> `Aider 的别名配置、DeepSeek Chat V3 性能、Repo-map 功能、Aider 中的模型组合、配置文件中的 API key 管理` 


- **Aider 别名配置的挑战**：用户在尝试于 **.env** 文件中设置模型别名时遇到困难，该设置未按预期生效。
   - 有人建议改用 **YML 配置文件**，它可以更有效地处理多个别名。
- **DeepSeek Chat V3 性能见解**：参与者注意到 DeepSeek Chat V3 在多语言排行榜上表现良好，由于其价格优势，可能会取代 Sonnet 成为首选模型。
   - 一位用户建议将 **DeepSeek V3** 与 **Gemini exp 1206** 配合使用，声称这在功能开发方面效果良好。
- **了解 repo-map 功能**：一位用户询问了 repo-map 功能，当切换到特定模型时，该功能在大型仓库中的更新速度较慢。
   - 另一位用户建议使用 **--map-refresh manual** 命令来简化更新，而不是自动刷新。
- **Architect 模式下的最佳模型组合**：关于 **Aider** 最佳模型组合的讨论倾向于使用 **O1** 或 **Gemini**，并提到 **DeepSeek 是一个可行的选择**。
   - 反馈表明，用户在处理复杂任务（如创建特定的函数预设）时遇到了一些困难，同时也关注易用性和成本效率。
- **出于安全考虑管理 API key**：一位新用户询问了在不包含 API key 的情况下提交 Aider 配置文件对安全性的影响。
   - 建议将 API key 分离到 **.env** 文件中以保持敏感信息的本地化，而配置文件可以包含在仓库中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/whatchu-talkin-about-willis-arnold-jackson-diffrent-strokes-what-are-you-tryi">无标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/whatchu-talkin-about-willis-arnold-jackson-diffrent-strokes-what-are-you-trying-to-say-willis-what-is-that-willis-gif-26301758">Whatchu Talkin About Willis Arnold Jackson GIF - Whatchu Talkin About Willis Arnold Jackson Diffrent Strokes - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/troubleshooting/support.html">使用 /help</a>: 使用 "/help " 询问有关使用 aider、自定义设置、故障排除、使用 LLM 等方面的帮助。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 aider 的常见问题解答。</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>: aider 是你终端里的 AI 结对编程助手</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hm2xvb/deepseek_v3_is_already_up_on_api_and_web/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/Aider-AI/aider/pull/2702">为目录外的文件显示绝对路径。由 apaz-cli 提交 · Pull Request #2702 · Aider-AI/aider</a>: 修改前/修改后：</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">模型与定价 | DeepSeek API 文档</a>: 下表列出的价格单位为每 1M tokens。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总额计费...
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1321211607501639820)** (2 条消息): 

> `BigCodeBench Leaderboard, GitDiagram 用于可视化, GitIngest 用于代码库` 


- **BigCodeBench 评估 LLMs**：[BigCodeBench Leaderboard](https://bigcode-bench.github.io) 通过**实际**且**具有挑战性**的编程任务来评估 LLMs，使用其 v0.1.0 版本进行评估。
   - 他们提供了多种资源，包括 [GitHub repo](https://github.com/bigcode-project/bigcodebench)、[Leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) 以及相关的 [arXiv paper](https://arxiv.org/abs/2406.15877)。
- **GitDiagram 将 GitHub 仓库转换为图表**：[GitDiagram](https://gitdiagram.com) 允许用户将任何 GitHub 仓库转换为交互式可视化图表，从而快速增强对项目结构的理解。
   - 只需将任何 GitHub URL 中的 'hub' 替换为 'diagram' 即可使用此工具；建议尝试不同的仓库进行演示。
- **GitIngest 简化代码库摄取**：[GitIngest](https://gitingest.com) 将任何 Git 仓库转换为纯文本表示，这对于将代码库输入到任何 LLM 中非常有用。
   - 与 GitDiagram 非常相似，你可以将任何 GitHub URL 中的 'hub' 替换为 'ingest'，以有效地访问此功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bigcode-bench.github.io">BigCodeBench Leaderboard</a>: 未找到描述</li><li><a href="https://gitdiagram.com/">GitDiagram - 数秒内将仓库转换为图表</a>: 将任何 GitHub 仓库转换为交互式图表以进行可视化。</li><li><a href="https://gitingest.com/">Git ingest</a>: 将任何 GitHub URL 中的 'hub' 替换为 'ingest'，以获取对 Prompt 友好的文本。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1321238708313133097)** (324 条消息🔥🔥): 

> `DeepSeek V3 发布、Linux Mint 体验、文本转视频模型对比、URL 审核 API 挑战、推理成本与部署` 


- **DeepSeek V3 发布，规格惊人**：DeepSeek V3 正式发布，拥有 **6850 亿参数**，被誉为最大的权重开放模型之一，其部署对 VRAM 的要求各异。
   - 讨论强调了高效模型推理对资源的高需求，建议使用 **320 个 GPU**（如 H100）以获得最佳性能。
- **探索 Linux Mint —— 改变游戏规则的选择**：用户分享了转向 **Linux Mint** 的兴奋之情，特别是在裸机与虚拟机上运行的资源效率体验备受关注。
   - 安装 Linux 为许多成员提供了有趣的参与式学习体验，他们非常欣赏该操作系统的轻量化特性和命令行功能。
- **关于文本转视频模型性能的辩论**：对比了文本转视频（Text-to-Video）模型，特别是 **Hunyuan** 和 **LTX 模型**，指出其可用性取决于 VRAM 等硬件规格。
   - 用户对最新的 T2V 模型表现出浓厚兴趣，并分享了关于易用性和性能限制的见解，特别是针对资源密集型任务。
- **构建 URL 审核 API 的挑战**：一位 AI 工程师讨论了使用 AI 模型开发 **URL 审核 API** 的困难，该 API 旨在准确分类各种维度的不安全网站且不产生幻觉（hallucinations）。
   - 使用不同模型的尝试效果均不理想，特别是 OpenAI 的模型经常拒绝提供帮助，而 Llama 在处理结构化输出（structured output）方面表现挣扎。
- **理解推理成本与模型部署**：成员们分析了部署 AI 模型的 **成本结构**，辩论了当前托管方案的效率和促销定价策略。
   - 尽管对价格可持续性持怀疑态度，但有人指出，充足的常规负载可以有效抵消运营成本，使高性能 AI 服务更加普及。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/discord-community/music-bot">Music Bot - discord-community 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/hacker-hackerman-kung-fury-gif-7953536">Hackerman GIF - Hacker Hackerman Kung Fury - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/_ivh810WHJo?si=MLEOP19PdPEZgP0x"> - YouTube</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main">deepseek-ai/DeepSeek-V3-Base at main</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/35010">由 not-lain 提交的 Pull Request #35010：将 `training_args.bin` 切换为 `training_args.json` · huggingface/transformers</a>：此 PR 做了什么？将 training_args.bin 切换为 training_args.json，并仅捕获用户传递的参数。我正在使用我们在 huggingface_hub 中使用的相同方法...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1321255035950333954)** (2 条消息): 

> `NotebookLM 行内溯源` 


- **关于 NotebookLM 行内溯源的咨询**：一位成员发起讨论，询问 **NotebookLM 行内溯源（inline sourcing）** 的运作机制。
   - 这表明人们对 NotebookLM 环境中溯源的具体机制兴趣日益浓厚。
- **寻求关于 NotebookLM 的知识**：该频道出现了要求澄清 **NotebookLM** 功能的呼声。
   - 成员们对行内溯源的操作方式表示好奇，反映出对更深层次理解的渴望。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1321394037051097109)** (2 messages): 

> `Differentiable Cache Augmentation, DeepSeek-V3` 


- **可微分缓存增强（Differentiable Cache Augmentation）提升 LLM 思考能力**：研究表明，通过在冻结的 LLM 上增加一个在其 KV Cache 上运行的离线协处理器，可以增强其生成和关注中间推理步骤的能力，从而降低延迟成本。
   - 实验表明，当缓存得到增强时，解码器在各种推理密集型任务中实现了**更低的困惑度（Perplexity）**和更好的性能，即使没有进行特定任务的训练。
- **在 GitHub 上探索 DeepSeek-V3**：DeepSeek-V3 项目正在积极开发中，可以在 GitHub 上进行探索，详情见 [DeepSeek_V3.pdf](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)。
   - 该 PDF 提供了关于项目开发和能力的见解，增强了社区的参与和贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2412.17747">Paper page - Deliberation in Latent Space via Differentiable Cache Augmentation</a>: 未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>: 通过在 GitHub 上创建账户，为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1321394037051097109)** (2 messages): 

> `Differentiable Cache Augmentation, DeepSeek V3` 


- **通过可微分缓存增强 LLM**：研究展示了一种通过离线协处理器增强冻结 LLM 的方法，该协处理器在模型的 **KV Cache** 上运行，以降低延迟并提高推理任务的性能。
   - *增强缓存能够持续降低各种任务的困惑度*，因为协处理器可以异步运行，且在协处理器不可用时，语言模型仍能保持正常功能。
- **DeepSeek V3 现已发布**：最新版本的 DeepSeek（即 **DeepSeek-V3**）已在 GitHub 上发布，在开发方面取得了重大进展。
   - 您可以在其 [GitHub 仓库](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)中找到更多详情并做出贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2412.17747">Paper page - Deliberation in Latent Space via Differentiable Cache Augmentation</a>: 未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>: 通过在 GitHub 上创建账户，为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1321242997161070705)** (2 条消息): 

> `LLM 网页搜索、模型降价、新 Endpoints API、Deepseek v3 发布` 


- **任意 LLM 网页搜索功能首次亮相**：**Web Search** 功能已在 OpenRouter Chatroom 为*任何语言模型*上线，使用户能够更轻松地获取最新信息。实时演示请查看[此链接](https://x.com/OpenRouterAI/status/1871682806335824029)。
   - *API 访问权限将在稍后引入*，该功能目前免费。
- **多款模型降价**：多款模型实施了显著的**价格下调**，包括 **qwen-2.5** 降价 **12%**，以及 **hermes-3-llama-3.1-70b** 降价 **31%**。
   - 详细的价格更新列出了一系列目前以更低成本提供的模型。
- **新 Endpoints API 进入 Beta 测试**：新的 **Endpoints API** 现已进入 Beta 测试阶段，允许用户在未公开预览期间访问模型详情和可用端点。在官方文档发布之前，这可能会有所变动。
   - 使用示例可在 [API 链接](https://openrouter.ai/api/v1/models/google/gemini-2.0-flash-thinking-exp:free/endpoints)中找到。
- **Deepseek v3 使用量翻三倍**：自 **Deepseek v3** 发布以来，其在 OpenRouter 上的使用量增长了三倍。基准测试显示，它在更低的价格点上具有与 **Sonnet** 和 **GPT-4o** 竞争的性能。感兴趣的用户可以在[此链接](https://x.com/OpenRouterAI/status/1872334128043208833)无需订阅即可试用。
   - 值得注意的评论强调，该模型被视为强有力的竞争者，且*中国在 AI 领域已经赶上*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1872334128043208833">OpenRouter (@OpenRouterAI) 的推文</a>：自昨天 v3 发布以来，Deepseek 在 OpenRouter 上的使用量已翻了三倍。亲自试用，无需订阅，包含网页搜索：引用 Anjney Midha 🇺🇸 (@AnjneyMidha) Deepseek v3 似乎是一个 gen...</li><li><a href="https://x.com/OpenRouterAI/status/1871682806335824029">OpenRouter (@OpenRouterAI) 的推文</a>：节日 🎁 实验：Web Search，但适用于任何 LLM！这是带有和不带有 grounding 的 Sonnet：
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1321512124861780119)** (2 条消息): 

> `3D 游戏生成工具、AI Chat Terminal (ACT)` 


- **用简单的文字生成 3D 游戏**：一款新工具允许用户通过简单的文字描述来创建 **3D 游戏**，解决了之前使用 **GPT-3/4** 和 **Claude** 等模型时面临的局限性。
   - 该工具的能力在 **o-1** 和 **o-1 preview** 的支持下显著提高，有望支持完整的体素引擎（voxel engine）来渲染复杂形状。
- **使用 AI Chat Terminal 改造你的终端**：介绍 **AI Chat Terminal (ACT)**，它融合了 **Agent 功能**和**代码库对话**，简化了与 **OpenAI** 和 **Anthropic** 等 AI 模型的交互。
   - 主要功能包括用于执行任务的 **Agent 模式**，以及用于在不同模型之间高效切换的**多供应商支持**。[立即试用](https://github.com/Eplisium/ai-chat-terminal)！


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://toy.new/">AI 生成的 3D 平台游戏</a>：未找到描述</li><li><a href="https://github.com/Eplisium/ai-chat-terminal">GitHub - Eplisium/ai-chat-terminal: 适用于 OpenAI 和 OpenRouter API 模型的终端脚本。让我们把它变成一个功能强大的执行脚本。</a>：适用于 OpenAI 和 OpenRouter API 模型的终端脚本。让我们把它变成一个功能强大的执行脚本。 - Eplisium/ai-chat-terminal
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1321205795584479282)** (301 条消息🔥🔥): 

> `DeepSeek V3 反馈、OpenRouter Chat 性能、DeepSeek 定价、API 限制、模型对比` 


- **DeepSeek V3 的评价褒贬不一**：用户正在讨论 **DeepSeek V3** 的性能，指出其表现似乎与之前的版本相当，但在编程等特定任务上有细微改进。
   - 一位用户分享了模型生成的诗歌，强调了各自的创造力，尽管有人认为输出质量参差不齐。
- **OpenRouter Chat UI 体验**：用户反馈了 **OpenRouter chat UI** 的情况，有报告称在处理超长对话历史时存在延迟和性能问题。
   - 用户希望响应速度能更快，因为当前的界面在处理大数据集时变得难以操作。
- **定价与模型访问**：关于模型定价的讨论包括对 **O1 Pro** 成本的担忧，以及对通过 OpenRouter 寻找替代方案的期待。
   - 用户希望避免高昂的月费，特别是传闻中像 **O3** 这样的新模型价格不菲。
- **API 中的请求批处理 (Batching)**：关于请求批处理如何运作的讨论集中在当 GPU 空闲时调度多个请求进行处理。
   - 用户注意到 OpenRouter API 并不直接支持 Batching，并强调了请求优先级排序的重要性。
- **Token 限制与模型访问**：用户对访问错误（如“未找到匹配您数据策略的端点”）表示担忧，发现原因是设置配置错误。
   - 讨论强调了需要对 API 设置进行清晰的沟通，以提升用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://x.com/ruben_kostard/status/1871941315380080800">Ruben Kostandyan (@ruben_kostard) 的推文</a>: @paulgauthier 是的，我在 API 上也看到了：https://x.com/ruben_kostard/status/1871939691794350161 引用 Ruben Kostandyan (@ruben_kostard) 你可以在 A... 验证 @deepseek_ai 模型为 V3</li><li><a href="https://openrouter.ai/deepseek">DeepSeek | OpenRouter</a>: 浏览来自 DeepSeek 的模型</li><li><a href="https://glhf.chat">good luck have fun</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/High_Bandwidth_Memory#HBM3E">High Bandwidth Memory - 维基百科</a>: 未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat)">Deepseek V3 - API, Providers, Stats</a>: DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编程能力之上。在近 15 万亿个 Token 上进行了预训练，报告的评估结果...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat)),">Deepseek V3 - API, Providers, Stats</a>: DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编程能力之上。在近 15 万亿个 Token 上进行了预训练，报告的评估结果...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/">Llama 3.3 | 模型卡片与提示词格式</a>: .</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">deepseek-ai/DeepSeek-V3 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2412.06769v1">Training Large Language Models to Reason in a Continuous Latent Space</a>: 未找到描述</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: 适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui</li><li><a href="https://arxiv.org/html/2410.09918v1">Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1321207759898804296)** (153 条消息🔥🔥): 

> `LM Studio Model Performance, AI Roleplaying Game Management, Memory Management Issues, Implementation of RAG for PAM, Model Context Length Limitations` 


- **LM Studio 模型性能与问题**：用户报告了在 LM Studio 中加载 QVQ 72B 等模型的问题，特别提到 MLX 运行正常，但 GGUF 存在 Bug。
   - 不同版本的性能也各不相同，早期版本中导致错误的一个旧 Bug 现已在 build 0.3.5 中得到解决。
- **在角色扮演游戏中管理 AI**：关于使用 Qwentile 等 AI 模型来管理桌面 RPG 体验的讨论引发了关注，重点在于这些模型在长篇叙事中保持连贯性的能力。
   - Mistral 和 Qwen 被认为能够胜任此用途，并建议通过微调（fine-tuning）或数据分块（chunking）来优化方法。
- **模型内存管理问题**：对 MLX 模型内存泄漏的担忧促使用户讨论了他们在会话管理和 RAM 使用方面的经验。
   - 报告的内存泄漏问题已得到确认，开发团队正在积极调查已知问题。
- **在 AI 系统中实现 RAG**：讨论了使用检索增强生成（RAG）作为增强模型记忆和管理 RPG 场景体验的方法。
   - 鼓励用户寻找现成的 RAG 实现方案，因为从零开始构建需要高超的编程技能。
- **模型上下文长度限制**：AI 模型的上下文长度限制是讨论的一个重点，特别是像 Llama 3.3 这样的大型模型如何管理 128k tokens。
   - 提出了在不超载上下文的情况下保留相关设定信息的策略，并承认内存管理仍然是一个关键挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mradermacher/Qwentile2.5-32B-Instruct-GGUF">mradermacher/Qwentile2.5-32B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://medium.com/@camauger/crafting-effective-chatgpt-prompts-for-tabletop-roleplaying-games-a-step-by-step-guide-part-1-b81a791d278d">为桌面角色扮演游戏编写有效的 ChatGPT 提示词：分步指南（第 1 部分）</a>: 欢迎阅读我们系列文章的第一部分，通过 ChatGPT 的视角探索桌面 RPG 与 AI 的创新交汇。</li><li><a href="https://www.youtube.com/watch?v=h9Z4oGN89MU"> - YouTube</a>: 未找到描述</li><li><a href="https://oracle-rpg.com/systems/">角色扮演系统 &#8212; Oracle RPG</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/issues/63">0.3.5b9 MLX 模型内存泄漏 · Issue #63 · lmstudio-ai/mlx-engine</a>: 在 8bit 下使用 L3.3 b70 模型的 mlx 转换版本，每个请求似乎都会导致巨大的内存泄漏。我有 33k 上下文，每个请求使用大约 10G 内存，这大约是 KVCache 的量...</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本</a>: LM Studio Beta Releases</li><li><a href="https://oracle-rpg.com/">Oracle RPG</a>: 独自玩《龙与地下城》等角色扮演游戏的指南和资源。</li><li><a href="https://lmstudio.ai/docs/cli#load-a-model-with-options">lms — LM Studio 的 CLI - CLI | LM Studio 文档</a>: 开始使用 lms 命令行工具。</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档聊天 - 在本地运行 LLM | LM Studio 文档</a>: 如何将本地文档作为额外上下文提供给 LLM</li><li><a href="https://lmstudio.ai/docs/api/openai-api">OpenAI 兼容 API - API | LM Studio 文档</a>: 向聊天补全（文本和图像）、补全和嵌入端点发送请求
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1321336106716762142)** (101 条消息🔥🔥): 

> `X99 主板性能，LLM 的 GPU 利用率，AI 训练的硬件推荐，多 GPU 设置，用于视频生成的低 VRAM 模型` 


- **X99 主板系统证明其能力**：用户讨论了基于 **X99 主板** 和 **Xeon E5 v4 CPU** 的旧款桌面系统，指出尽管这些设备已有一定年头，但在处理模型时表现良好。
   - *一位用户观察到，他们的双 RTX 2060 配置可以有效处理大型模型，* 强调对于日常使用来说，升级可能并非必要。
- **使用 LLM Studio 优化 GPU 利用率**：有用户提出了在多 GPU 设置下运行 LLM Studio 时 **GPU 利用率低** 的问题，GPU 使用率仅达到 **30%** 左右。
   - 专家建议，由于 **memory latency**（内存延迟）的存在，增加 VRAM 容量并不一定会提高推理速度，并建议使用 NVLink 以获得更好的性能。
- **关于 AI 硬件推荐的讨论**：用户交流了关于构建能够训练大型 AI 模型的系统的 **主板和 CPU 组合** 的见解，强调了 **成本效益** 和组件兼容性。
   - 一位用户强调 **双 CPU 的 Genoa 服务器主板** 是一个可行的选择，而其他人则分享了各种 GPU 配置的经验。
- **多 GPU 配置的挑战**：参与者讨论了在组装机中使用多个 GPU 的限制，特别是关于 **PCIe lane 配置** 和带宽限制。
   - *有人指出，拥有更多 GPU 虽然增加了 VRAM，但并不一定能转化为更快的单次推理速度。*
- **探索用于视频模型的 LoRA**：用户推测了视频生成模型的未来发展，特别是关于使用极少量数据训练的 **LoRA**。
   - 一位用户对仅使用几张静态图像的训练能力表示怀疑，而其他人则讨论了其对 **视频质量** 的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=GENOA2D24G-2L%2b#Specifications">未找到标题</a>：未找到描述</li><li><a href="https://www.ebay.com/str/sinobright">Security Measure</a>：未找到描述</li><li><a href="https://tenor.com/view/thats-the-neat-part-you-dont-invincible-gif-27194608">Thats The Neat Part You Dont Invincible GIF - Thats The Neat Part You Dont Invincible - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.ebay.com/itm/186713565965?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=EXxczRPuTe2&sssrc=2047675&ssuid=jxws3gfsrkg&widget_ver=artemis&media=COPY">Asrock WRX90 WS EVO Motherboard - Opened Box Tested to BIOS  | eBay</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1321205912177868902)** (226 条消息🔥🔥): 

> `AI Image Generation Techniques, ComfyUI Usage Tips, Stable Diffusion Model Comparisons, Video Generation Capabilities, NSFW Protections in LoRA` 


- **探索 AI 图像生成控制**：用户讨论了 AI 图像生成过程的复杂性，指出根据所使用的模型，更详细的 Prompt 可以产生更好的结果。
   - 参与者强调，学习如何有效地编写 Prompt 对于实现对生成图像的控制至关重要。
- **用于生成艺术的 ComfyUI**：许多用户对 ComfyUI 的复杂性表示困惑，但发现使用符号链接（symlinked）模型和 Stability Matrix 进行管理非常成功。
   - 成员们推荐了像 SwarmUI 这样更易于使用的界面工具，同时也分享了他们使用各种 AI 的个人经验。
- **比较视频生成模型**：讨论围绕 ComfyUI 中 img2video 模型的能力展开，并将其与 Veo 2 和 Flux 等模型在效果和资源需求方面进行了对比。
   - 有人指出 LTX Video 适合拥有 8GB VRAM 的用户，同时社区也在探索新的视频生成技术。
- **超分辨率模型的效率**：关于训练超分辨率模型的合适图像尺寸出现了疑问，特别是生成 1700 px 图像的可行性。
   - 用户意识到，虽然高分辨率很理想，但它们通常需要大量的资源，这对于所有配置来说可能并不现实。
- **处理 LoRA 模型中的 NSFW 功能**：关于如何处理 LoRA 中的 NSFW 保护有一段幽默的交流，建议在操作这些设置时需要透明度。
   - 社区成员分享了他们对 NSFW 选项的局限性和现有功能的看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/KONAKONA666/LTX-Video">GitHub - KONAKONA666/LTX-Video: LTXVideo Q8</a>: LTXVideo Q8。通过创建 GitHub 账户为 KONAKONA666/LTX-Video 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/comfyui/comments/1hm9qhu/another_ai_in_the_loop/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1321212063116300310)** (183 条消息🔥🔥): 

> `OpenAI Outage and Alternatives, DeepSeek V3 Performance, Model Comparisons, ChatGPT Competitors, Acronym Understanding in LLMs` 


- **OpenAI 故障引发对替代方案的关注**：用户报告了 ChatGPT 宕机的问题，引发了对 DeepSeek、Google AI Studio 和 Claude 等替代 AI 模型的讨论。
   - 许多人表示欣慰，认为这次故障的影响似乎比之前的事件要小，从而引导大家探索其他选项。
- **DeepSeek V3 令人印象深刻的性能**：DeepSeek V3 因其性能受到关注，拥有 64k 上下文限制，且响应速度比 GPT-4 和 Sonnet 3.5 等成熟模型更快。
   - 用户强调了它提供连贯代码支持且无微小错误的能力，使其成为开发项目的强力候选者。
- **对 DeepSeek 和文件支持的褒贬不一**：虽然 DeepSeek 的效率受到称赞，但一些用户指出了其局限性，例如缺乏直接的文件支持，特别是与处理各种格式的模型相比。
   - 尽管如此，使用 OCR 进行文档处理的前景增加了一层多功能性，吸引了那些寻求全能解决方案的用户。
- **探索 ChatGPT 之外的其他模型**：用户正在考虑将重点从 GPT-4o 转向 DeepSeek V3 和 Gemini AI 等替代方案，尤其是在 ChatGPT 出现问题期间。
   - 社区强调了尝试新模型以找到适合不同编程任务和需求的重要性。
- **LLM 中缩略词识别的挑战**：讨论了如何有效地 



**提到的链接**: <a href="https://status.openai.com/incidents/6bwlxnvdncnm">ChatGPT、API 和 Sora 的高错误率</a>: 未找到描述

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1321237821071298713)** (33 messages🔥): 

> `GPT-O3 Release, ChatGPT Down Status, Using GPTs in RPGs, Canvas Window Issues, Eslint Configuration` 


- **GPT-O3 发布在即**：一名成员提到 **O3-mini** 预计在 1 月下旬发布，**O3** 预计在此后不久发布。
   - 另一名成员对该模型的能力表示好奇，并强调目前的信息非常稀缺。
- **ChatGPT 遭遇宕机**：多名用户报告无法访问 ChatGPT，其中一人特别指出在不同浏览器和移动端 App 上均出现错误。
   - 反应各异，从对订阅选择的讽刺到对 ChatGPT 确实宕机的普遍断言。
- **使用 GPT 的有趣 NPC 概念**：一位用户提议在 RPG 中创建一个“会说话的 NPC”，幽默地模仿 GPTs 及其局限性，并称其“完全不理解自己实际在说什么”。
   - 另一名成员表示赞同，并将其类比为《飞出个未来》（*Futurama*）中一个滑稽的翻译机器。
- **Canvas 窗口故障**：一位用户报告 **Canvas 窗口损坏**，打开片刻后即关闭，并寻求他人确认。
   - 虽然没有分享解决方案，但这似乎是讨论中用户普遍感到沮丧的问题。
- **使用 O1 Pro 进行 Eslint 配置**：一名成员询问如何使用 **o1 pro** 配置设置，建议如果 Eslint 配置是结构化的，或许可以直接传递相关设置。
   - 这反映了用户对于在开发环境中将更好的 App 功能与现有工具集成的持续兴趣。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

madame_architect: 为什么一分钟就不行呢？
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

madame_architect: 为什么一分钟就不行呢？
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1321589625159942174)** (7 messages): 

> `ProductPAPI, Anthropic Concerns, Direct Code Modification, Claude Load Issues` 


- **Gabe 的 ProductPAPI 正在开发中**：一名成员透露 **ProductPAPI** 是 Gabe 正在开发的一款旨在简化用户任务的 App，但细节仍然很少。
   - 未提供关于其功能或发布时间表的进一步信息。
- **影响供应商的可扩展性问题**：有成员对 Bolt 用户切换到 **Anthropic** 的简洁模式（concise mode）后出现的质量下降表示担忧，这影响了所有主要供应商。
   - 这种质量下降表明平台面临着**巨大的可扩展性问题**。
- **Prompt 中对直接代码修改的需求**：一名成员对收到代码片段而非直接修改感到沮丧，寻求改进其 Prompt 的建议。
   - 另一名成员建议在 Prompt 中包含明确的请求，如“请直接对我的代码进行修改”。
- **Claude 与 Bolt 性能之间的联系**：成员们讨论了 **Claude** 的负载与 **Bolt** 性能之间的相关性。
   - 一名成员推测，在 Claude 上遇到需求警告可能也预示着 Bolt 的性能会变差。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1321226057700999240)** (198 条消息🔥🔥): 

> `Bolt Token 使用问题，使用 Bolt 构建应用程序，Bolt 的功能请求和反馈，社区支持与协作，工具限制和用户体验` 


- **频繁的 Token 使用问题**：多位用户报告称，在 Bolt 中进行基础操作时消耗了大量 Token，部分用户对 Prompt 被忽略以及代码被无故修改感到沮丧。
   - 一位用户表示他们消耗了超过 150 万个 Token 但收效甚微，这导致他们重新评估自己的方法，并对 AI 的性能感到失望。
- **拼车应用开发确认**：一位新用户询问了使用 Bolt 构建覆盖全美的拼车应用的可行性，并确认了他们现有的用于机场接送的 Web 门户网站。
   - 社区成员保证，由于该平台是基于 Web 的，开发所需的应用程序应该是可行的。
- **用户体验的反馈和建议**：用户正在寻求提供反馈和改进 Bolt 用户体验的建议，并建议通过 GitHub 提交功能请求。
   - 一位社区成员强调了清晰沟通渠道的重要性，以帮助开发团队优先处理用户请求。
- **调试问题的社区支持**：用户讨论了各种调试问题和策略，包括当代码更改与用户输入不一致时面临的挑战，这引发了更深入的调查。
   - 社区建议包括检查错误消息、使用 GitHub 报告 Bug 以及咨询支持资源以获取帮助。
- **升级和 Token 限制的挑战**：用户对订阅层级的成本以及从付费计划过渡时的 Token 限制表示担忧，并讨论了持续使用的替代方案。
   - 用户被提醒，一旦达到 Token 限制，可以在 StackBlitz 中进行免费编码，这在资金允许续订之前提供了一个临时解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai-banking-app.netlify.app/)">Harmony - Where Finance Meets Mindfulness</a>：暂无描述</li><li><a href="https://support.bolt.new/github-tracked-feature-requests">Notion – 笔记、任务、维基和数据库的全能工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的全能工作空间。</li><li><a href="https://support.bolt.new/welcome#13fd971055d68027a0cdddd14a9d7900">Notion – 笔记、任务、维基和数据库的全能工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的全能工作空间。</li><li><a href="https://bolters.io">Bolters.io | Bolt.new 无代码应用构建器的社区支持技巧、窍门和知识库</a>：Bolt.new 的文档和指南</li><li><a href="https://bolters.io/docs/read-this-first">请先阅读此内容</a>：关于 Bolt.new 能力、限制和成功最佳实践的关键信息</li><li><a href="https://github.com/stackblitz/bolt.new/issues">Issues · stackblitz/bolt.new</a>：Prompt、运行、编辑和部署全栈 Web 应用程序 - Issues · stackblitz/bolt.new
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1321228642482651259)** (132 条消息🔥🔥): 

> `QVQ 模型发布，微调 Llama 模型，DeepSeek V3 讨论，Nvidia 驱动问题，AI 训练的数据集格式化` 


- **QVQ 模型刚刚发布**：**QVQ-72B** 的 4-bit 和 16-bit 版本已上传，展示了强大的视觉推理能力，被定位为现有模型的竞争替代方案。
   - 该模型的性能亮点包括在 MMMU 基准测试中达到 **70.3%**，表明其在多学科任务中的潜力。
- **微调 Llama 模型的挑战**：在微调 **Llama 3.2 3B** 时，用户报告了训练后模型性能的问题，表明输入数据的质量和格式会极大影响结果。
   - 专家建议，训练成功更多取决于数据的**质量**而非数量，并建议根据试错进行迭代调整。
- **DeepSeek V3 的 MoE 架构**：DeepSeek V3 引发了关于其规模和能力的讨论，特别是其对 Mixture of Experts (MoE) 架构的使用，引发了对 OpenAI 和 Anthropic 模型的猜测。
   - 它提供的扩展效率和计算节省使其比 Sonnet 便宜 **50 倍**，使其成为 AI 领域的重要参与者。
- **Nvidia 驱动版本问题**：讨论发现，在云环境中使用模型时，运行较旧的 Nvidia 驱动程序（如 **535.161.07**）可能会导致兼容性问题。
   - 鼓励用户在 Linux 环境中操作，特别是使用 WSL 以获得更好的兼容性以及在 Triton 等库上的性能。
- **AI 训练的数据集格式化**：对于微调模型的人来说，训练数据集的格式（特别是对话式还是问答式）被认为是模型有效性的关键。
   - 展示的一个示例包括具有 system、user 和 assistant 角色的结构化数据，强调了遵守正确格式标准的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://choosealicense.com/no-permission/">No License</a>: 你没有义务选择许可证，并且你有权不在你的代码或项目中包含许可证。但请注意，选择不使用开源许可证并不意味着你退出了...</li><li><a href="https://huggingface.co/collections/unsloth/qwen-qvq-qwq-collection-676b3b29c20c09a8c71a6235">Qwen QVQ + QwQ Collection - Unsloth 集合</a>: 无描述</li><li><a href="https://huggingface.co/Qwen/QVQ-72B-Preview">Qwen/QVQ-72B-Preview · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: 无描述</li><li><a href="https://youtu.be/_ivh810WHJo?si=MLEOP19PdPEZgP0x"> - YouTube</a>: 无描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-m">Unsloth 文档</a>: 无描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth 文档</a>: 查看下方列表以获取所有已上传的 GGUF、16-bit 和 4-bit bnb 模型
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1321248847774613655)** (4 条消息): 

> `Sprint 模式查询，用于指令微调的代码数据集，毕业论文的个人训练经验` 


- **关于 Sprint 模式时间的查询**：一位成员急切地询问“sprint mode”何时可用，表达了明显的好奇心和紧迫感。
   - 附带了一张图片，但未提供有关其内容的具体细节。
- **寻求 LLM 的代码数据集**：一位成员请求推荐专门适用于大语言模型指令微调的“**代码数据集**”，并概述了首选格式，包括问题描述和生成的解决方案。
   - 他们表示偏好专注于 Python 解决方案的数据集。
- **分享个人训练经验**：另一位成员分享了他们最近的个人经验，提到他们为自己的**学士学位论文**在自己的数据上训练了一个模型。
   - 然而，他们淡化了其重要性，称：“*我不会称之为一种经验。*”

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1321511183932457092)** (37 messages🔥): 

> `SFT DPO Evaluation, Fine-tuning Llama 3.2 Vision, Using Unsloth with CPU, GGUF Conversion Issues, Model Performance Discrepancies` 


- **训练期间的 SFT DPO 评估**：一位成员询问是否可以在训练期间进行评估，并使用 Unsloth 计算类似于 Hugging Face Transformers 库的指标。
   - 他们询问文档中是否有支持该功能的示例。
- **使用纯文本数据集微调 Llama 3.2 的问题**：一位成员报告了使用纯文本 JSONL 数据集微调 Llama 3.2 时遇到的问题，称 Unsloth 报错提示需要图像数据。
   - 讨论强调了需要禁用 vision layers（视觉层），但该成员寻求进一步澄清，为什么在进行了此设置后错误仍然存在。
- **在 CPU 上运行训练好的模型**：一位成员表示在仅有 CPU 的本地机器上加载 GPU 训练的模型时遇到困难，寻求关于配置 Unsloth 以供 CPU 使用的文档。
   - 另一位成员建议将模型量化为 GGUF 格式，并指出 Unsloth 文档中提供了相关指南。
- **GGUF 转换的挑战**：一位成员提到，在使用 llama.cpp 将微调后的 Llama 3.2 模型转换为 GGUF 格式时，模型性能出现下降。
   - 建议确保配置和 prompt 格式的一致性，以避免转换过程中的问题。
- **本地模型的异常响应**：一位成员报告称，与在 Colab 上运行相比，在本地运行微调后的 Mistral 模型时会收到荒谬的响应。
   - 他们寻求帮助以诊断输出异常的原因以及如何提高响应质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/peft_utils.py#L87">unsloth-zoo/unsloth_zoo/peft_utils.py at main · unslothai/unsloth-zoo</a>：Unsloth 工具类。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做贡献。</li><li><a href="https://github.com/unslothai/unsloth/commit/a2407835534747d2421f58cbdeeb5a49482e7235#diff-46849d25980ee8d9337f4f8c30369faf36ceda3479272fd737ebf5ad9c703840R15">Bug Fixes (#1470) · unslothai/unsloth@a240783</a>：* 更新 llama.py

* 更新 _utils.py

* 更新 llama.py

* 更新 llama.py

* 更新 _utils.py

* 更新 pyproject.toml

* 更新 _utils.py

* 更新 llama.py

* CE Loss

* 更新...</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>：查看下方列表以获取我们所有上传的 GGUF、16-bit 和 4-bit bnb 模型。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1321331147304730665)** (5 messages): 

> `Stella recommendations, Mixed Bread models, Benchmarking and finetuning` 


- **关于 Stella 未被推荐的询问**：一位成员对为什么 **Stella** 未被推荐表示好奇，引发了讨论。
   - *Mrdragonfox* 承认没有接触过 Stella，表明它在社区中可能还没有建立起良好的声誉。
- **Mixed Bread 模型展示了其能力**：另一位成员强调他们每天都在使用 **mixed bread** 模型，并肯定它们根据应用场景的不同而非常**有能力**。
   - 他们强调，此类模型的有效性最终取决于其应用的具体**垂直领域**。
- **基准测试和微调的必要性**：一位成员指出针对个人数据进行模型 **benchmarking**（基准测试）以确保性能的重要性。
   - 他们还指出，最终可能需要进行 finetuning（微调），正如反映关注度的**下载数量**所显示的那样。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1321206120194375781)** (134 messages🔥🔥): 

> `Perplexity AI 使用疑虑, AI 模型反馈, AI 领域求职咨询, AI 模型对比, 订阅与访问问题` 


- **Perplexity AI 中的模型选择挑战**：用户报告了选择特定模型时的问题，其中一位指出选择 **Sonar Huge** 时会默认跳转到 **Claude 3.5 Sonnet**，突显了对当前选择限制的沮丧。
   - 另一位用户表达了无法选择 Claude 3.5 Sonnet 和 GPT-4o 以外任何模型的担忧，这可能暗示存在 Bug。
- **Perplexity AI 模型效能与竞争**：用户讨论了各种 AI 模型的效能，一些人指出 **Gemini** 的 **Deep Research Mode** 与其他模型相比非常有竞争力，特别是在功能和上下文处理方面。
   - 讨论集中在可能增加的新模型上，关于 **o1** 被集成到服务中的猜测仍在继续。
- **求职咨询与社区互动**：一位成员表达了对 AI 领域工作机会的兴趣，强调了他们在涉及 NLP 和 Generative AI 项目方面的经验。
   - 社区互动中出现了一些针对潜在候选人能力和预期的幽默回应。
- **订阅访问与易用性担忧**：几位用户在访问 Pro Channels 时遇到挑战，其中一人询问是否有获得访问权限的特殊技巧，暗示访问协议缺乏清晰度。
   - 用户讨论了与免费订阅相关的支付问题经历，并寻求解决方案。
- **编程 AI 工具反馈**：一位用户寻求编程 AI 的推荐，详细说明了对几种工具的不满，并要求提供高质量、快速输出的替代方案。
   - 其他社区成员提供了建议，同时分享了使用不同 AI 编程平台的个人经验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aistudio.google.com/prompts/new_chat">no title found</a>: no description found</li><li><a href="https://tenor.com/view/here-money-owe-pay-pay-up-gif-16899251">Here Money GIF - Here Money Owe - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://linux.do/t/topic/312925/70">DeepSeek-V3 已悄咪咪上线网页端以及 API</a>: 测试了一下 真的太强了 希望不要涨价😭
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1321229276066091038)** (14 messages🔥): 

> `NASA 触及太阳, Murder Hornets 被根除, 用于 EV 充电的太阳能涂料, 来自印度的 AI 模型, 体温供电的穿戴设备` 


- **NASA 实现触及太阳的历史性接触**：NASA 最近的任务成功触及了**太阳**，收集了关于其外部大气的突破性数据。这一成就可能会重新定义我们对太阳物理学及其对地球影响的理解。
   - *专家强调了这次任务的重要性*，认为它是太阳研究的关键一步。
- **Murder Hornets 被成功清除**：最近的努力已导致多个地区**根除了 Murder Hornets**，当局确认没有新的发现。该倡议旨在保护当地生态系统和蜜蜂种群。
   - 专家称这一成功在持续的环境挑战中对*“维持生物多样性至关重要”*。
- **创新太阳能涂料为 EV 充电**：一种新型**太阳能涂料**已经开发出来，可以产生能量为电动汽车充电，为清洁交通的未来带来了希望。测试显示其效率对于*可再生能源技术具有革命性意义*。
   - 这项创新被**研究人员**称为一项突破，引发了对可持续城市发展的兴趣。
- **受 Yann LeCun 启发的突破性印度 AI 模型**：来自印度的一个新 **AI 模型**声称体现了著名研究员 **Yann LeCun** 的愿景，承诺在 AI 功能和伦理方面取得进展。该模型旨在增强类人推理和学习能力。
   - **AI 社区**的许多人对这一发展持乐观态度，理由是它具有改变模型训练过程的潜力。
- **体温供电穿戴技术的突破**：新型体温供电**穿戴技术**出现，允许设备在没有电池的情况下运行。这一进步不仅提高了便利性，还旨在推广更环保的个人电子产品方案。
   - 专家认为这可能是**穿戴设备市场**的游戏规则改变者，推向了能源效率的极限。



**提到的链接**：<a href="https://www.youtube.com/embed/_zUGuxWw-sM">YouTube</a>: no description found

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1321391905187299348)** (4 条消息): 

> `Payment Processing, Virtual Cards, OpenRouter Credits, Perplexity Models` 


- **企业使用安全支付处理**：成员们讨论了支付处理是必不可少的，且目前**没有其他可选方案**。
   - 一位成员建议使用**虚拟卡 (virtual card)**，并指出大多数银行为了安全性都提供该功能。
- **OpenRouter 提供 Perplexity 模型**：一位成员在购买额度后惊讶地发现 **OpenRouter** 也提供对 **Perplexity 模型**的访问。
   - 尽管有了这一发现，他们目前仍决定坚持使用所选的供应商。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1321227898782482432)** (103 条消息 🔥🔥): 

> `DeepSeek V3, OpenAI outages, ChatGPT memory improvements, RL training for LLM reasoning, Anduril partnership with OpenAI` 


- **DeepSeek V3 树立新标准**：DeepSeek V3 已发布，拥有 *685B 参数*，展现出卓越的性能和极高的效率，训练仅消耗了 *2.788M H800 小时*。
   - 它实现了 *60 tokens/second* 的速度，并在各项基准测试中取得了令人印象深刻的成绩，提升了高性价比模型训练的门槛。
- **OpenAI 遭遇重大停机**：一次严重的停机事故影响了 OpenAI，创下了自 2023 年 1 月以来最差的单月运行时间记录，给 API 可用性带来了剧烈波动。
   - 用户对 API 的可靠性表示担忧，这对于严重依赖 OpenAI 服务的在研项目产生了影响。
- **ChatGPT 的无限记忆功能**：传闻 ChatGPT 将推出 *infinite memory* 功能，允许访问过去的聊天记录，旨在显著增强用户交互体验。
   - 这一新功能预计将解锁更无缝、更自然的对话，提升 AI 的实用性。
- **用于提升 LLM 推理能力的 RL 训练**：分享的一段 YouTube 视频强调了针对 LLM 推理改进的有效强化学习 (RL) 技术，提供了富有洞察力的训练方法论。
   - 对推理能力的关注预示着 LLM 训练的演进，强调在生成连贯且逻辑严密的输出方面的实际应用。
- **Anduril 与 OpenAI 宣布合作伙伴关系**：Anduril 宣布与 OpenAI 达成合作，将 AI 解决方案整合到军事应用中，强调增强决策能力和国家安全。
   - 在关于伦理考量的讨论中，此次合作凸显了在国防领域推进负责任 AI 技术的承诺。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/deepseek_ai/status/1872242657348710721">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 60 tokens/second（比 V2 快 3 倍！）💪 能力增强 🛠 API 兼容性保持不变 🌍 完全开源的模型与论文 🐋 1/n</li><li><a href="https://backchannel.org/blog/autonomous-software">在自主软件开发（Autonomous Software Development）时代进行构建</a>：未找到描述</li><li><a href="https://x.com/teortaxestex/status/1871892454187921495?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：好了伙计们，我又睡过了一个更新。Whale-3（DeepSeek-V3）来了，而且比以往任何时候都好。（据我所知，VL2 API/网页版也快推出了；V3.0 将不支持图像）引用 You Ji...</li><li><a href="https://x.com/nearcyan/status/1863302015230886017))">来自 near (@nearcyan) 的推文</a>：天哪，有人帮我做了深度伪造（deepfake），太完美了。引用 near (@nearcyan)：在一个 AI Agent 的世界里，工程师（Engineer）的时代已经结束，创意者（Idea Guy）的时代已经到来。</li><li><a href="https://x.com/teortaxestex/status/1872245075465454043?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：我们要…开始了。所以，配置里的那一行，是的，它是关于多 token 预测（multi-token prediction）的。只是作为一个更好的训练目标——尽管他们保留了投机解码（speculative decoding）的可能性。此外，“我的 50K H...”</li><li><a href="https://x.com/teortaxesTex/status/1872002534774341782">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：4chan 的老哥们让我想到：🐳V3 可能是剪枝（pruning，通常是个噱头）第一次奏效。他们细粒度 MoE 的卖点是“极致的专家专业化”，而 @wzihanw 开发了一种方法...</li><li><a href="https://x.com/nrehiew_/status/1872318161883959485?s=46">来自 wh (@nrehiew_) 的推文</a>：如何训练一个 670B 参数模型。让我们聊聊 DeepSeek-V3 的报告，并与 Meta 在 Llama 405B 上的做法进行一些对比。</li><li><a href="https://x.com/teortaxestex/status/1872253671989551473?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：> Sonnet 级别仅需 550 万美元。难怪他们引以为豪，但这感觉确实像是在伤口上撒盐。“1 亿美元的运行成本，哈？405B 用了 3084 万 H100 小时，是吧？愚笨的西方...”</li><li><a href="https://x.com/scaling01/status/1872281384057819200?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：使用 Llama 3 模型家族的计算预算（3930 万 H100 小时），META 至少可以训练 DeepSeek-V3 15 次。与此同时，DeepSeek 仅花费了 260 万 H800 小时（一种受限的...）</li><li><a href="https://x.com/paulgauthier/status/1871919612000092632">来自 Paul Gauthier (@paulgauthier) 的推文</a>：DeepSeek 新 V3 模型的“预览版”在 aider 多语言排行榜上获得第二名。62% o1，48% DeepSeek V3 Preview，45% Sonnet，38% Gemini-exp-1206，33% o1-mini。https://aider.chat/docs/lead...</li><li><a href="https://x.com/main_horse/status/1872294985888059612?s=46">来自 main (@main_horse) 的推文</a>：论文里随处可见这种轻描淡写的干货：“噢，顺便说一下：我们不需要 TP，因为我们最先进的流水线方案通过手动管理 SM 分配，实现了与 EP 的完美计算-通信重叠 && ...”</li><li><a href="https://x.com/nrehiew_/status/1872318217395572895">来自 wh (@nrehiew_) 的推文</a>：他们有两种类型的 RL 奖励。验证器（Verifiers，代码、数学）和标准的基于模型的 RM。重要的是，基于模型的 RM 是采用 DeepSeek Math 中使用的 COT 风格 GRPO 训练的。</li><li><a href="https://x.com/jonkkillian/status/1832563242129895580?t=SiCg9BtzgANz5vqbm-BznA&s=19">来自 Jon Kurtis ⚡ (@jonkkillian) 的推文</a>：@CSMikeCardona AI 代码将是自蛋黄酱+大蒜变成蒜泥蛋黄酱（Aioli）以来最棒的低代码（low code）品牌重塑。</li><li><a href="https://x.com/TheXeophon/status/1871867610788507914">来自 Xeophon (@TheXeophon) 的推文</a>：600B 参数，在 aider 新的多语言基准测试中排名第二，超越了 Sonnet。61.7% o1，48.9% 🐋 V3，45.3% Sonnet。</li><li><a href="https://x.com/deanwball/status/1872321587480854801?s=46">来自 Dean W. Ball (@deanwball) 的推文</a>：中国 AGI 实验室 DeepSeek 报告称其新 V3 模型的训练成本极低，仅为 550 万美元，性能似乎与 Claude 3.5 Sonnet 相当。DeepSeek 还有一个可靠的“o1”竞争对手...</li><li><a href="https://x.com/alexocheema/status/1872081513627763004?s=46">来自 Alex Cheema - e/acc (@alexocheema) 的推文</a>：我要在 M4 Mac Mini 上运行 DeepSeek-V3-Base 685B，否则誓不罢休。256 个专家的 685B MoE——非常适合 Apple Silicon，因为它们有大量的 GPU 内存，且每次只有一小部分参数处于激活状态...</li><li><a href="https://x.com/_xjdr/status/1872263123543187551?s=46">来自 xjdr (@_xjdr) 的推文</a>：我个人对那些使用了更多计算资源但性能明显更差的训练运行感到尴尬。这对我来说是 FLOP 效率的新标杆，我绝对爱死它了。引用...</li><li><a href="https://x.com/btaylor/status/1871627726580576368?s=46">来自 Bret Taylor (@btaylor) 的推文</a>：软件工程师的角色</li>

正在从计算机代码的作者转变为代码生成机器的操作员。为此原生构建的计算机编程系统是什么...</li><li><a href="https://x.com/jsevillamol/status/1872287890304364912?s=46">Jaime Sevilla (@Jsevillamol) 的推文</a>：DeepSeek-V3 的效率令人印象深刻。它拥有 37B 激活参数，并在 14.8T tokens 上进行了预训练，总计 6 x 37B x 14.8T = 3e24 FLOP。这比 Llama 3.1 430B 的计算量少 10 倍，但表现更好...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">deepseek-ai/DeepSeek-V3 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/mark_k/status/1871856522143399961?s=46">Mark Kretschmann (@mark_k) 的推文</a>：传闻中 ChatGPT 的 ♾️（无限）Memory 功能是真的。看起来在假期后即将推出。这项新功能将允许 ChatGPT 访问你过去所有的聊天记录，解锁更多可能性...</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721?s=46">DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 每秒 60 tokens（比 V2 快 3 倍！）💪 增强的能力 🛠 保持 API 兼容性 🌍 完全开源的模型和论文 🐋 1/n</li><li><a href="https://x.com/kevinakwok/status/1871631334478909685">Kevin Kwok (@kevinakwok) 的推文</a>：大家都在讨论 @btaylor 关于自主软件开发的精彩文章。但他最具有先见之明的呼吁是 2008 年提出的“数据的维基百科”。想象一下，在我们的分发中还会有多少...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/T1SeqBapMBo?si=JVeVYsD1K5CYCI5K"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/terryyuezhuo/status/1872017850933911802">Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：热烈祝贺 @deepseek_ai！V3 Chat 模型现在在 BigCodeBench-Hard 上排名第一。Complete -- 40.5% Instruct -- 28.4% Average -- 34.5% Gemini-Exp-1206 Average -- 34.1% o1-2024-12-17 (reasoning=medium) Ave...</li><li><a href="https://x.com/anduriltech/status/1864390729516327375?t=WLawzCNT1WUwUdGdj">Anduril Industries (@anduriltech) 的推文</a>：我们正与 @OpenAI 联手推进国家安全的 AI 解决方案。美国需要获胜。OpenAI 的模型与 Anduril 的防御系统相结合，将保护美国及其盟友的军事人员...</li><li><a href="https://x.com/reach_vb/status/1871961056928506237?s=46">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：深入研究了一下配置文件，v2 与 v3 的主要区别（根据配置文件）：vocab_size: v2: 102400 v3: 129280 hidden_size: v2: 4096 v3: 7168 intermediate_size: v2: 11008 v3: 18432 num_hidden_lay...</li><li><a href="https://x.com/karpathy/status/1872362712958906460?s=46">Andrej Karpathy (@karpathy) 的推文</a>：DeepSeek（中国 AI 公司）今天让一切看起来轻而易举，发布了一个开源权重的尖端级 LLM，其训练预算少得惊人（2048 个 GPU 训练 2 个月，花费 600 万美元）。作为参考，这种水平的能力...</li><li><a href="https://x.com/anduriltech/status/1864390729516327375?t=WLawzCNT1WUwUdGdjbGVaQ&s=19">Anduril Industries (@anduriltech) 的推文</a>：我们正与 @OpenAI 联手推进国家安全的 AI 解决方案。美国需要获胜。OpenAI 的模型与 Anduril 的防御系统相结合，将保护美国及其盟友的军事人员...</li><li><a href="https://x.com/johnrushx/status/1871405441948987786">John Rush (@johnrushx) 的推文</a>：绝对没有人预料到这一点：AI Code 是新的 NoCode。老实说，在构建小型应用时，我更喜欢与 AI 交流，而不是与人类开发人员交流。即使规格说明还不成熟，它也能更好地理解我...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>：通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1321211407362162742)** (3 条消息): 

> `2024 年 Synthetic Data、2024 年 Agents、AI Engineer Summit NYC、活动日历更新` 


- **Loubna 回顾 Synthetic Data 与 Smol Models**：在[最新一期节目](https://x.com/latentspacepod/status/1871652198956015941)中，Loubna Ben Allal 分享了关于 **2024** 年 **Synthetic Data** 和 **Smol Models** 顶级论文的见解。
   - 关键时间点涵盖了 **Synthetic Data 的兴起** 和 **Model Collapse** 等话题，为精彩的讨论铺平了道路。
- **Graham Neubig 雄心勃勃的 Agent 演讲**：在我们的最后一场主题演讲中，[Graham Neubig](https://github.com/All-Hands-AI/openhands-agent-monitor/pull/41) 探讨了 **2024 年 Agents** 的格局，展示了关于其设计和有效使用的深刻见解。
   - 他进行了现场演示，并分享了关于 Agent 未来的**带有鲜明观点的幻灯片**，探讨了 **Human-Agent Interaction** 中的挑战。
- **预留 2025 年 AI 活动时间**：请在日历上标记好在纽约举行的 **AI Engineer Summit** 以及暂定于 **2025** 年上半年举行的其他活动。
   - 访问 [Latent.Space events](http://Latent.Space) 以获取最新动态，并订阅日历以接收新活动通知。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1871652198956015941">来自 Latent.Space (@latentspacepod) 的推文</a>：## 回顾 2024 年的 Synthetic Data/Smol Models！我们非常荣幸邀请到 @LoubnaBenAllal1 来回顾并挑选今年关于 Synthetic Data 和 Smol Models 的所有最佳论文！时间戳 [00:00:05] Loubna 介绍...</li><li><a href="https://x.com/latentspacepod/status/1871998012467380698">来自 Latent.Space (@latentspacepod) 的推文</a>：我们的最后一场主题演讲：回顾 2024 年的 Agents。我们将最雄心勃勃的演讲留到了最后——我们邀请了在 SWE-Bench Full 上排名第一的 Agent 的创建者 @gneubig，来回顾与构建 Agent 相关的所有内容...</li><li><a href="https://lu.ma/ls">Latent Space (Paper Club &amp; 其他活动) · 活动日历</a>：在 Luma 上查看并订阅 Latent Space (Paper Club &amp; 其他活动) 的活动。Latent.Space 活动。请点击日历右上方 RSS 图标将其添加到您的日历中。"Ad...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1321205874261495918)** (45 条消息🔥): 

> `DeepSeek V3 发布、Multi-Token Prediction、Reward Model 技术、性能对比、模型训练技术` 


- **DeepSeek V3 发布预示着重大进展**：DeepSeek V3 已正式推出，拥有 **60 tokens/秒** 的速度，比前代 V2 快 **3 倍**，具备现有的 API 兼容性并增强了各项能力。
   - 该模型完全开源，并在 **14.8 万亿 tokens** 上进行了训练，展示了在资源受限情况下的卓越工程实力。
- **创新的 Multi-Token Prediction 方法揭晓**：DeepSeek 采用了一种新颖的 **Multi-Token Prediction (MTP)** 技术，该技术将最终表示与初始嵌入（Embeddings）连接起来，同时在整个预测过程中保持因果链。
   - 这种方法与以往的方法不同，在不增加多个独立 Decoding Heads 复杂性的情况下，潜在地增强了模型的有效性。
- **用于训练的新型 Reward Model 技术**：DeepSeek 采用了两种类型的 RL 奖励：用于**代码/数学的验证器（Verifiers）**，以及以 **Chain of Thought** 风格训练的基于模型的 RM 方法，这有助于优化输出。
   - 据报道，该实现战略性地提升了模型性能，但对于非确定性输出（如创意写作）的 Critique 机制仍存在疑问。
- **对现有模型的推测与对比**：多位成员对 DeepSeek V3 的性能表示兴奋，指出其**优于许多开源模型**，并可与 **GPT-4o 和 Claude-Sonnet-3.5** 等模型相媲美。
   - 关于训练资源**高性价比扩展（Cost-effective scaling）**的讨论强调了与早期迭代相比，GPU 小时数的高效利用。
- **Critique 与 Revision 的辩论**：成员们推测了使用 Critique 与生成多个输出并选择最佳回答的有效性，思考了计算效率的影响。
   - 针对 Prompt 中整合外源信息的问题被提出，暗示明确 Self-critique 的作用可能会带来更有效的模型训练结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1871867610788507914">来自 Xeophon (@TheXeophon) 的推文</a>：600B 参数，在 aider 的新多语言基准测试中排名第二，超越了 Sonnet。61.7% o1，48.9% 🐋 V3，45.3% Sonnet。</li><li><a href="https://x.com/nrehiew_/status/1872318217395572895">来自 wh (@nrehiew_) 的推文</a>：他们有两种类型的 RL 奖励：验证器（代码、数学）和标准的基于模型的 RM。重要的是，这里使用的基于模型的 RM 是采用 DeepSeek Math 的 COT 风格 GRPO 训练的。</li><li><a href="https://x.com/reach_vb/status/1872252796936003719">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：所以.. V4 可能不再是 Transformers 了？我好奇他们会倾向于什么方向！引用 Vaibhav (VB) Srivastav (@reach_vb)：DeepSeek 技术报告发布了！！🔥 在 14.8 万亿 Token 上训练...</li><li><a href="https://x.com/jiayi_pirate/status/1871837684521718149">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>：@YouJiacheng @teortaxesTex 他们说将在 12.25-27 进行模型更新，可能与之相关。</li><li><a href="https://x.com/AndrewCurran_/status/1872255379591282774">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：@teortaxesTex Anthropic 风格。</li><li><a href="https://x.com/TheXeophon/status/1871865868944285864">来自 Xeophon (@TheXeophon) 的推文</a>：是的，DeepSeek V3 对我来说已经上线了。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>：未找到描述。</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 60 tokens/second（比 V2 快 3 倍！）💪 能力增强 🛠 API 兼容性保持不变 🌍 完全开源的模型和论文 🐋 1/n</li><li><a href="https://x.com/Tim_Dettmers/status/1872280778975191241">来自 Tim Dettmers (@Tim_Dettmers) 的推文</a>：阅读报告后，发现在资源限制下这是如此纯粹的工程实现。DeepSeek 团队直接针对硬件限制下的已知问题设计了解决方案。这一切看起来都如此优雅...</li><li><a href="https://x.com/phill__1/status/1871859816681128223">来自 Phil (@phill__1) 的推文</a>：DeepSeek V3 已在其聊天界面上线，现在支持图片。</li><li><a href="https://x.com/lmsysorg/status/1872251875070021831">来自 lmsys.org (@lmsysorg) 的推文</a>：最好的开源 LLM，DeepSeek V3 刚刚发布！SGLang v0.4.1 是官方推荐的推理解决方案。SGLang 和 DeepSeek 团队合作支持了 DeepSeek V...</li><li><a href="https://x.com/nrehiew_/status/1872318215277432905">来自 wh (@nrehiew_) 的推文</a>：&gt; 经过数百步 RL 之后，中间 RL 模型学会了融入 R1 模式，从而在策略上增强了整体性能。</li><li><a href="https://linux.do/t/topic/312925/118">DeepSeek-V3 已悄咪咪上线网页端以及 API</a>：请问为啥 DeepSeek 的多模态，上传图片必须要有文字啊</li><li><a href="https://x.com/teortaxesTex/status/1872253671989551473">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：&gt; 550 万美元达到 Sonnet 级别。难怪他们为此感到自豪，但这确实感觉像是在炫耀。“1 亿美元的运行，哈？在 405B 上花费 3084 万 H100 小时，是吗？愚蠢的西方人...”</li><li><a href="https://x.com/nrehiew_/status/1872318212831891585">来自 wh (@nrehiew_) 的推文</a>：现在的后训练。他们在 R1（**非 LITE 版本**）上进行 FT，但表示它存在“过度思考、格式差和长度过长”的问题。他们有两种类型的数据：1) 标准合成数据 2) 一个系统...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1hlzax7/wow_deepseek_v3/">哇 DeepSeek V3 ? </a>：由 u/Evening_Action6217 发布在 r/LocalLLaMA • 327 个赞和 46 条评论。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1321867602481516626)** (5 条消息): 

> `Deepseek 的 Multi-head Latent Attention 机制、实现与推理库、Deepseek V2 论文见解、Deepseek V3 推理代码` 


- **探索 Deepseek 的 Latent Attention**：一位成员询问是否有人研究过 **Deepseek 的 Multi-head latent attention 机制**，并提到 V2 论文中缺乏关于权重矩阵低秩近似 (low-rank approximations) 的细节。
   - *他们目前正在尝试创建一个版本*，并想了解其他人是否也实现了类似的功能。
- **推理库提供支持**：另一位成员建议 **inference libraries** 应该已经有了 Deepseek 机制的实现，并强调了 **SGLang** 对 V3 的首日支持。
   - 他们还指出 **vLLM**、**TGI** 和 **hf/transformers** 已经支持了这些新特性，包括 **Multi-head latent attention**。
- **Deepseek 官方推理代码已发布**：一位成员提供了 **Deepseek GitHub 仓库**的链接，特别指出了 [model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) 中的推理代码。
   - 对于那些希望实现或理解 Deepseek V3 引入的新功能的人来说，这个资源非常有帮助。
- **在 Hugging Face 上查找实现**：最初的询问者表示他们还没有检查 **Hugging Face** 方面的情况，并计划去查看。
   - 他们对分享的关于现有实现的信息表示感谢。



**提到的链接**：<a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py">DeepSeek-V3/inference/model.py at main · deepseek-ai/DeepSeek-V3</a>：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-V3 的开发做出贡献。

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1321468520524939324)** (15 条消息🔥): 

> `QvQ 许可证更新、Bluesky 安全担忧、来自数据科学家的 AI 抵制` 


- **QvQ 拥有永久的 Apache 2.0 许可证**：一位成员澄清说，你不能追溯性地更新 **QvQ** 的许可证，确认始终会有一个 **Apache 2.0** 许可的版本可供 checkout。
   - 随后有人指出该许可证变得比 **Llama** 更加宽松，另一位成员开玩笑地提到“许可证战争”的出现。
- **Bluesky 不适合进行讨论**：有人对 **Bluesky** 的环境表示担忧，称其不是一个安全的地方，并有报告称其存在“疯狂的反 AI 倾向”。
   - 一位成员指出，对生成式 AI 的抵制往往来自数据科学家，他们对该技术表示蔑视，却忽略了其他有问题的 AI 应用。
- **OpenAI 回应团队疑虑**：一位成员分享的链接指向了 **OpenAI** 的一份声明，回应了关于前团队成员的询问，表明讨论正在进行中。
   - 这引发了另一位成员以反应表情符号的形式表达好奇，凸显了社区对该情况的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAINewsroom/status/1872312018994352636">来自 OpenAI Newsroom (@OpenAINewsroom) 的推文</a>：这是我们针对前团队成员相关问题提供的声明：</li><li><a href="https://x.com/casper_hansen_/status/1871895390049685756">来自 Casper Hansen (@casper_hansen_) 的推文</a>：我不想做那个扫兴的人，但你不能追溯性地更新许可证。现在将永远存在一个 Apache 2.0 许可的 QvQ 版本，你可以通过 git checkout 获取</li><li><a href="https://x.com/EstebanCervi/status/1872314732851679679">来自 Esteban Cervi 🦌 (@EstebanCervi) 的推文</a>：@OpenAINewsroom 🧐
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1321225336549146646)** (6 messages): 

> `Meta paper on self-awareness in language models, Copyright lawsuits against AI companies, Anthropic and copyright issues, Public benefit ethos of AI companies` 


- **寻找 Meta 最近关于语言模型的论文**：一位用户询问了 **最近（6 个月内）** 发表的一篇 Meta 论文，该论文讨论了语言模型虽然能识别自身的局限性，但由于缺乏意识而无法自我纠正。
   - 该论文在一个不活跃的讨论串中被提及，导致细节尚不明确。
- **关于 AI 版权的法律斗争**：讨论强调了 **Thomson Reuters** 对 Ross Intelligence 发起的 **版权诉讼**，这标志着新兴生成式 AI 领域早期的冲突。
   - 随着法律斗争的升级，其结果可能会 **重塑信息生态系统和 AI 行业**。
- **对 Anthropic 行为的批评**：有人对 **Anthropic 的行为** 表示担忧，声称该公司从非法渠道下载了大量受版权保护的书籍，损害了公共利益。
   - 评论将这种行为称为 **“盗取普罗米修斯之火”**，暗示了重大的伦理冲突。
- **对 Anthropic 涉嫌侵犯版权的反应**：针对对 Anthropic 的严肃批评，出现了一些幽默的回应，强调了所使用的比喻，并表达了对现状的沮丧。
   - 回应显示出对该公司行为严重后果的复杂心情。



**提及的链接**：<a href="https://www.wired.com/story/ai-copyright-case-tracker/">Every AI Copyright Lawsuit in the US, Visualized</a>：WIRED 正在追踪涉及 AI 行业的每场版权之战——我们创建了一些简便的可视化图表，并将随着案件的进展进行更新。

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1321398352000319491)** (4 messages): 

> `Bluesky performance, New LLM release from DeepSeek` 


- **对 Bluesky 质量的批评**：一位成员评论了评估 Bluesky 等平台的高标准，暗示其表现可能不佳。
   - 讨论中提到 *只是从 Twitter 转发 lol*，表现出对分享内容的随意态度。
- **潜在的行业改变者：DeepSeek 的新模型**：Simon Willison 预测，如果 DeepSeek 在 **圣诞节** 发布一个新的开源许可 **685B LLM**，那将是一个令人兴奋的时刻，他强调了该模型与现有模型相比的潜在规模和影响力。
   - 据指出，该模型下载量 **超过 700GB**，且仓库中目前没有现成的许可证，这引发了对其可访问性和文档说明的疑问。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://bsky.app/profile/iwonabb.bsky.social/post/3lds6ajafzz2e">Iwona Bialynicka-Birula ⏩ (@iwonabb.bsky.social)</a>：只是开个玩笑 :)</li><li><a href="https://x.com/simonw/status/1872141432544489731">Simon Willison (@simonw) 的推文</a>：如果我们以 2024 年底最好的可用 LLM 是一个来自中国研究实验室、在圣诞节发布到 Hugging Face 且拥有开源许可的 685B 巨兽来收尾，那将非常有趣...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1321719015030132789)** (3 条消息): 

> `Monte Carlo Tree Search, Iterative Preference Learning, Reasoning in LLMs` 


- **Monte Carlo Tree Search 增强 LLM 的推理能力**：最近的一篇论文介绍了一种通过受 AlphaZero 启发的**迭代偏好学习过程 (iterative preference learning process)** 来提高大语言模型 (LLMs) **推理能力**的方法，该方法采用了 **Monte Carlo Tree Search (MCTS)**。
   - 该方法利用*前瞻能力 (look-ahead ability)* 将实例级奖励细化为粒度信号，并结合 **Direct Preference Optimization (DPO)** 来更新 LLM 策略。
- **MCTS 中的自我评估增强**：该论文结合了*结果验证 (outcome validation)* 和**分步自我评估 (stepwise self-evaluation)**，以提高推理过程中的一致性。
   - 它强调了**同策略采样数据 (on-policy sampled data)** 对有效自我改进的重要性，并对**算术和常识推理**进行了广泛评估。
- **关于模型选择的观点**：一位成员对研究中使用的模型选择表示困惑，质疑其质量，因为对于 2024 年 5 月来说，这些模型似乎*表现欠佳 (down-bad)*。
   - 这一评论反映了关于在 **MCTS** 和推理轨迹背景下技术充分性的持续讨论。



**提及的链接**：<a href="https://arxiv.org/abs/2405.00451">Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning</a>：我们介绍了一种旨在通过迭代偏好学习过程增强大语言模型 (LLMs) 推理能力的方法，该过程灵感来自 ... 所采用的成功策略。

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1321959726732546150)** (8 条消息🔥): 

> `Effective RL Training for LLMs, DPO vs PPO, Reasoning in RL, Viewing Parties for Lectures` 


- **来自 CMU RL 训练研讨会的见解**：讨论了一个题为 *Towards effective RL training for LLMs* 的研讨会，重点介绍了前半部分关注 DPO 与 PPO 的对比以及 PPO 优化。
   - 第二部分可能对那些对*推理*感兴趣的人更有吸引力，特别是关于 PRM 偏差和 Clip/Delta 缓解措施的内容。
- **DPO 与 PPO 的辩论**：围绕 **DPO** 和 **PPO** 的比较优势展开了讨论，特别是它们与更有效地训练 LLM 的关系。
   - *Is DPO superior to PPO?* 是即将在 **ICML 2024** 上发表的相关论文，进一步阐明了这一关键领域。
- **对学习观影会的兴趣**：一位成员建议，针对关键讲座和教程视频进行公开讨论的观影会可能有利于集体学习。
   - 回复显示出一种比起*贡献价值*更倾向于*获取价值*的偏好，从而引发了关于集体参与的对话。
- **基于 PRM 的训练考量**：有人提到，与 ORM 方法相比，*基于 PRM 的训练*可能提供更多塑造奖励的机会。
   - 围绕哪些方法能激励*更好的*思维链 (CoTs) 产生了疑问，强调了对这些概念的持续探索。
- **重复观看视频的价值**：一位成员提到了他们通过重复观看视频来保留信息的策略，这是一种个人学习技巧。
   - 这引发了关于理解复杂话题挑战的幽默讨论，暗示共同观看可能会增强理解。



**提及的链接**：<a href="https://youtu.be/T1SeqBapMBo?si=srBHIwpVnDC3aX7x"> - YouTube</a>：未找到描述

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1321399803854323743)** (12 messages🔥): 

> `Learning CUDA Programming, DETRs and PyTorch, Shared Memory in CUDA, DeepSeek-V3 Training, Earning via Telegram Strategies` 


- **初学者应明确 CUDA 学习目标**：在学习 CUDA 编程时，初学者需要专注于实际应用，例如就业机会或性能驱动的项目。
   - 最终，有志于成为 CUDA 程序员的人应该有一个驱动其学习旅程的个人目标。
- **寻求 PyTorch 中 DETRs 的帮助**：一位用户正在为其涉及 **DETRs** 和 **PyTorch** 的业余项目寻求帮助，他已经面临了三个月的挑战。
   - 另一位成员表示愿意提供帮助，并展示了对 **DETRs** 的熟悉程度。
- **CUDA 中的动态内存分配**：在讨论 CUDA 编程时，有人指出 **Shared Memory** 无法从 Device 代码中动态分配。
   - 因此，在 **Shared Memory** 中使用 **C++ vector** 数据结构被认为是不切实际的。
- **DeepSeek-V3 实现高成本效益**：分享了一个指向 **DeepSeek-V3** 文档的链接，强调 **大规模 FP8 混合精度训练** 是一项重大进展。
   - 值得注意的是，据报道该项目将成本降低了 **2 个数量级 (OOMs)**，引发了关于其获得的资金与质量之间关系的讨论。
- **快速获利的 Telegram 方案**：一名用户发布了一个方案，声称能帮助他人在 72 小时内赚取 **$100k**，成功后收取 10% 的利润分成。
   - 感兴趣的人被引导通过 **Telegram** 联系以参与其中。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。</li><li><a href="https://huggingface.co/blog/train_memory">Visualize and understand GPU memory in PyTorch</a>: 未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>: 通过在 GitHub 上创建账号为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1321556174289764503)** (8 messages🔥): 

> `Casting Issues in Triton, Device Printing in Colab, Infinity Feature in Triton, Triton Recompilation, Scam Alert` 


- **Triton 中 fp8 和 bf16 之间的类型转换问题**：由于 PTX 错误，从 **fp8** 转换为 **bf16** 在 **SM89** 上失败，导致 `triton.codegen_upcast_to_fp32` 标志出现问题。
   - 一种解决方法是执行 `.to(tl.float32).to(tl.bfloat16)`，尽管这可能需要添加一个哑操作（dummy operation）以防止融合（Fusion）。
- **Device Print 在 Colab 中不显示**：一位用户对 `device_print` 在 **Colab** 中没有任何输出表示沮丧，这可能是一个 Bug 或使用问题。
   - 目前还没有针对此问题的确认解决方法，用户仍在寻求方案。
- **Triton 中的 Infinity 行为**：针对关于 `tl.inf` 的询问，已确认使用 `float("inf")` 可以作为 Triton 中 **torch.inf** 的替代品。
   - 这为需要表示无穷大的用户提供了一个功能等价物。
- **Triton 何时重编译**：一位用户提出了关于 **Triton** 何时重编译的基础问题，表明需要更清晰的关于重编译过程的文档。
   - 这个问题暗示了用户对 Triton 的工作流和重编译触发机制可能存在困惑。
- **潜在诈骗警告**：出现了一条提供 72 小时内赚取 **$100k** 方案的消息，要求参与者偿还 10% 的费用，这引发了对潜在诈骗的警惕。
   - 这种情况值得警惕，建议用户举报或避免参与此类可疑提议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。</li><li><a href="https://github.com/triton-lang/triton/issues/5491">Casting to bf16 from fp8 breaks on SM89 · Issue #5491 · triton-lang/triton</a>: 描述 Bug：你好，Triton 中从 fp8 到 bf16 的转换在 SM89 上因 PTX 错误而失败。我提交此 Issue 是因为这导致 {&quot;triton.codegen_upcast_to_fp32&quot;: False} 失败 ...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1321345918426943548)** (14 条消息🔥): 

> `TMA vs cp.async 用于 GEMM，DETRs 和 PyTorch 咨询，WGMMA 的性能要求，关于 Hopper 结构化稀疏 GEMM 的 CUTLASS 讨论，社交媒体上分享的赚钱方法` 


- **TMA 为 GEMM 提供卓越的效率**：成员们讨论了为什么在 Hopper 上需要 **TMA (Tensor Memory Access)** 来实现 **高效 GEMM**，而在 Ada 上 **cp.async** 就足够了。他们指出，由于 **H100** 等系统上的 flops 增加，TMA 有助于释放寄存器。
   - 一位成员补充说，TMA 是异步的，支持并发操作，提供边界检查，并允许批量调度。
- **DETRs 项目的挑战**：一位成员表达了对他们的 DETRs 和 PyTorch 相关业余项目的挫败感，表示已经卡住了三个月。
   - 另一位成员幽默地建议在更多频道互动可能会获得更好的回应，随后引发了一场关于频道发帖的轻松交流。
- **WGMMA 性能依赖于寄存器**：讨论强调了对于 WGMMA，输入需要位于 **shared memory** 中，而累加必须保持在 **registers** 中；有人指出，一篇微基准测试论文表明，为了达到峰值性能，其中一个输入需要位于寄存器中。
   - 成员们得出结论，除了结构化稀疏 FP8 之外，与早期的 MMA 方法相比，并没有显著的性能差异。
- **关于 CUTLASS 3.6.0 增强功能的见解**：分享了一个关于 **CUTLASS 3.6.0** 的链接，详细介绍了针对 FP16、FP8、INT8 和 TF32 的 **Hopper 结构化稀疏 GEMM** 的改进。
   - 讨论强调了卷积算子 API 与 **gemm::GemmUniversal** 的对齐，以实现性能提升。
- **社交媒体赚钱方案提议**：一位成员发布了邀请，声称能帮助他人在 **72 小时内赚取 10 万美元**，并规定需返还 10% 的利润。
   - 该帖子鼓励感兴趣的人私下联系，虽然引起了关注，但反应褒贬不一。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/discussions/2013">CUTLASS 3.6.0 · NVIDIA/cutlass · Discussion #2013</a>：Hopper 结构化稀疏 GEMM。FP16 FP8 INT8 TF32。对 CUTLASS 3.x 卷积 kernel::ConvUniversal API 进行了重构，使其与 gemm::GemmUniversal 保持一致。现在 3.x 卷积 API 不再...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1321929823945490593)** (3 条消息): 

> `Guard Functions 的影响，赚钱策略` 


- **关于代码中 Guard Functions 的担忧**：一位成员询问了在代码中使用多个 guard functions 的 **性能影响**，探讨减少它们是否能提高效率。
   - *过度思考 guard 的数量是否是一种反模式？* 他们寻求社区关于理想平衡点的指导。
- **快速致富机会**：另一位成员向首批 **20 位感兴趣的人** 提供帮助，教他们如何在 72 小时内赚取 **10 万美元**，要求是 **10% 的利润返还**。
   - 他们鼓励对该方案真正感兴趣的人通过 **Telegram** 直接联系。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1321982690366587012)** (1 条消息): 

> `赚钱策略，Telegram 推广` 


- **推广快速获利方案**：一位成员正为前 **20 名** 有意在 **72 小时** 内赚取 **10 万美元** 的人提供帮助，要求返还 **10% 的利润**。
   - 鼓励感兴趣的人发送好友请求或私信，并强调 **Telegram** 是主要的沟通工具，账号为 [@CharlesWilliam26](tg://resolve?domain=CharlesWilliam26)。
- **强调快速收益**：该消息推广了一种快速赚钱策略，暗示参与者可以在极短的时间内获得巨额利润。
   - 这种方法带有个人色彩，要求感兴趣的人通过 **私信 (DMs)** 或好友请求积极参与。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1321775857139843145)** (2 条消息): 

> `Character.AI 推理优化，在线赚钱，Telegram 优惠` 


- **Character.AI 优化 AI 推理**：Character.AI 专注于**高效推理**以增强用户体验，探讨了如 Multi-Query Attention 和 int8 量化等技术，详见[此处](https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/)。他们自定义的 int8 Attention Kernel 显著提升了计算密集型和内存密集型任务的**推理速度**。
   - 这建立在之前强调**内存效率**和减少 KV Cache 大小以获得更好性能的研究基础之上。
- **赚取 10 万美元的快速通道**：有人提供在 72 小时内赚取 **10 万美元**的指导，并在收到利润后收取 **10% 的费用**。感兴趣的用户可以通过好友请求或私信联系以获取更多详情。
   - 可以通过 **Telegram** 联系 [Charles William](https://t.me/CharlesWilliam26)，他正在推广一项财富共享计划。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/">Optimizing AI Inference at Character.AI (Part Deux)</a>: 在 Character.AI，我们正在构建个性化的 AI 娱乐。为了给我们的用户提供引人入胜的互动体验，实现高效推理（即处理过程...）至关重要。</li><li><a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1321711560795754530)** (5 条消息): 

> `学习 CUDA 和 Triton，vLLM Token 吞吐量分析，Attention 机制中的序列堆叠，优化的 Attention 实现，赚钱机会` 


- **掌握 CUDA 和 Triton 的路径**：一位成员询问了学习 **CUDA** 和 **Triton** 等底层 ML 技术的最佳方式，提到了 PMPP 和编程谜题等资源。
   - *做这些事情有先后顺序吗？*
- **使用 xFormers 后端调查 vLLM 的 TTFT**：一位成员正在使用 xFormers 后端对 vLLM 的 **Token Throughput/First Token** 性能分析进行 Profiling，并链接了相关代码。
   - 他们质疑为什么 vLLM 在 Prefill 阶段不使用 Batched Inference。
- **理解序列堆叠（Sequence Stacking）的好处**：另一位成员将 vLLM 的 Attention 机制与 FlashAttention 的实现进行了比较，将其描述为“序列堆叠”，这允许高效处理可变长度。
   - 他们提供了一个[讨论此问题的博客文章链接](https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences)，强调了性能与灵活性之间的权衡。
- **Attention 变体的挑战**：讨论强调，虽然像 **FlashAttention** 这样优化的 Attention 实现提高了性能，但它们使测试新的 Attention 变体变得复杂。
   - 成员们指出，受限于现有 Kernel 的缺点，如果变体不适合这些框架，可能会导致运行速度缓慢。
- **快速赚钱的机会**：一位成员宣布提供帮助 20 个人在 72 小时内赚取 **10 万美元**的机会，要求报销利润的 **10%**。
   - 他们鼓励感兴趣的人通过 **Telegram** 联系以获取计划详情。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>: 未找到描述</li><li><a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。</li><li><a href="https://github.com/vllm-project/vllm/blob/dbeac95dbbf898bcc0965528fc767e9cadbbe0c5/vllm/attention/backends/xformers.py#L613">vllm/vllm/attention/backends/xformers.py at dbeac95dbbf898bcc0965528fc767e9cadbbe0c5 · vllm-project/vllm</a>: 一个用于 LLM 的高吞吐量且内存高效的推理和提供服务的引擎 - vllm-project/vllm
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1321551997035614248)** (3 messages): 

> `PMPP 讲座，盈利策略` 


- **PMPP 讲座大放异彩**：一位成员称赞 [PMPP 作者的讲座](https://youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&si=Z3rAAzxzbYgDpjmt) 非常有价值，表示这些讲座对理解书中的相应章节有很大帮助。
   - *很高兴知道这一点，刚订了书！* 另一位成员表示，强调了讲座的价值。
- **72 小时赚取 10 万美元计划**：一名用户宣布将帮助前 20 名感兴趣的人，教他们如何在 **72 小时内赚取 10 万美元**，条件是返还其利润的 10%。
   - 他们鼓励用户发送好友请求或私信（DM）询问方法，并提供了一个 [用于直接联系的 Telegram 链接](https://t.me/CharlesWilliam26)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&si=Z3rAAzxzbYgDpjmt">AUB Spring 2021 El Hajj</a>: 未找到描述</li><li><a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1321982643344113706)** (1 messages): 

> `快速赚取 10 万美元，投资方案，Telegram 推广` 


- **Charles 提供快速致富机会**：Charles William 承诺帮助前 **20 人** 在 **72 小时内** 赚取 **10 万美元**，但要求在赚取利润后 **返还 10%**。
   - *鼓励感兴趣的人发送好友请求或私信他，* 并提供了他的 [Telegram](https://t.me/CharlesWilliam26) 链接以便直接联系。
- **Telegram 作为沟通工具**：他强调使用 **Telegram** 让感兴趣的人直接联系他，称这有助于快速沟通。
   - Charles 将自己塑造为财富分配者，旨在通过这一倡议在全球范围内传播财富。



**提到的链接**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1321983331880927255)** (1 messages): 

> `盈利机会，Telegram 推广，利润分成` 


- **72 小时赚 10 万美元方案**：一名成员向首批 **20 名感兴趣的人** 提供帮助，教他们如何在 **72 小时内** 开始赚取 **10 万美元**，并在收到利润后返还 10%。
   - 他们鼓励潜在参与者发送 **好友请求** 或 **私信** 以了解更多信息，并强调了联系的紧迫性。
- **通过 Telegram 联系**：该成员提供了他们的 **Telegram** 链接，以便就该盈利机会进行即时联系，详情请见 [此处](https://t.me/CharlesWilliam26)。
   - 他们正在积极推动参与，声称专注于在全球范围内 **传播财富**，以此诱导感兴趣的人与其联系。



**提到的链接**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。

  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1321983375224864841)** (1 messages): 

> `赚取 10 万美元，利润返还，Telegram 联系` 


- **快速赚钱蓝图**：Charles William 正在为前 **20 名** 有兴趣学习如何在 **72 小时内赚取 10 万美元** 的人提供指导。
   - 参与者在收到利润后需要向他返还 **10% 的利润**，并敦促感兴趣的人通过 **Telegram** 联系。
- **Telegram 通讯**：感兴趣的人可以通过提供的链接在 **Telegram** 上直接联系 **Charles**。
   - 他的账号是 [@CharlesWilliam26](tg://resolve?domain=CharlesWilliam26)，他在那里鼓励潜在参与者私信他以获取更多细节。



**提到的链接**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。

  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1321982863649800204)** (1 messages): 

> `Earning $100k in 72 hours, Charles William's proposition, Reimbursement of profits, Telegram contact` 


- **在短短 72 小时内解锁 10 万美元**：一位用户分享了一个诱人的提议，向首批 **20 人** 提供指导，告诉他们如何在 **72 小时** 内开始赚取 **10 万美元** 的潜在利润。
   - *感兴趣的人士* 被鼓励联系以获取详情或发送好友请求。
- **10% 利润返还方案**：参与者在收到收益后，必须将 **10% 的利润** 返还给提供者，从而形成一种利润分成模式。
   - 这种返还模式引发了人们对利润分成计划的可持续性和伦理问题的质疑。
- **通过 Telegram 联系 Charles**：对于直接咨询，**Charles** 提供了一个 **Telegram** 链接，以便感兴趣的人士快速取得联系。
   - 他强调 *将财富传播到世界各地*，将自己定位为财务导师。



**Link mentioned**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1321983287727624272)** (1 messages): 

> `Earning $100k, Telegram Contact, Profit Sharing Strategy` 


- **快速致富策略：72 小时内赚取 10 万美元**：一名成员声称他们可以帮助首批 **20 人** 在 **72 小时** 内赚取 **10 万美元**，并提议在收到收益后进行 **10% 的利润返还**。
   - *感兴趣的人士* 被鼓励发送好友请求或 DM 以获取详情，并提供了一个 Telegram 链接用于直接联系。
- **通过 Telegram 联系以快速获利**：该成员提供了他们的 **Telegram 账号** @CharlesWilliam26，邀请个人联系以寻求赚钱方面的帮助。
   - 他们强调了通过 Telegram 联系的即时性，突出了该提议的紧迫感。
- **在全球范围内传播财富**：该成员表达了对财富分配的愿望，声称他们正在 *将财富传播到世界各地*。
   - 这一表述暗示了他们除了个人利益之外，还有更广泛的动机。



**Link mentioned**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1321983468766363771)** (1 messages): 

> `Earning Strategies, Telegram Networking` 


- **快速致富：72 小时内赚取 10 万美元**：向首批有兴趣学习如何在 **72 小时** 内赚取 **10 万美元** 的 **20 人** 提供了一项提议，条件是事后返还其利润的 **10%**。
   - *只有真正感兴趣的人* 被鼓励发送好友请求或直接消息以获取详情，并强调了指向 **Telegram** 的直接链接。
- **在 Telegram 上联系 Charles**：感兴趣的各方可以直接通过 **Telegram** 账号 [@CharlesWilliam26](tg://resolve?domain=CharlesWilliam26) 联系 **Charles** 以获取更多信息。
   - 该消息强调了 *将财富传播到世界各地* 的理念，这与 **社区驱动** 的方法相一致。



**Link mentioned**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1321983437657079831)** (1 messages): 

> `Earning $100k strategy, Telegram outreach` 


- **72 小时内赚取 10 万美元的机会**：一名成员正向首批 **20 名感兴趣的人** 提供帮助，指导他们如何开始在 **72 小时** 内 **赚取 10 万美元**，并预期返还 10% 的利润。
   - 他们鼓励感兴趣的人发送好友请求或 DM，并说明 *问我怎么做（HOW）！* 以获取更多详情。
- **寻求帮助的 Telegram 联系方式**：对于那些对赚钱机会感兴趣的人，**Charles William** 建议通过 [此链接](https://t.me/CharlesWilliam26) 在 **Telegram** 上联系他。
   - 他宣传 *将财富传播到世界各地* 的想法，暗示了一种更广泛的财务策略。



**Link mentioned**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1321983498772418705)** (1 messages): 

> `72 小时内赚取 10 万美元，利润分成模式，Telegram 推广，投资机会，财务建议` 


- **72 小时内赚取 10 万美元？没问题！**：一位用户提议帮助前 **20 名感兴趣的个人**开始在 **72 小时内赚取 10 万美元**，条件是收到利润后需**返还 10% 的利润**。
   - 感兴趣的人员被鼓励发送好友请求或**私信 (DM)** 以获取更多信息。
- **加入 Telegram 投资群组**：该用户提供了一个 [Telegram 链接](https://t.me/CharlesWilliam26) 用于直接联系，并强调感兴趣的人应立即行动。
   - “在世界各地传播财富”被描述为该用户财务咨询计划的一部分。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1321982962937364582)** (1 messages): 

> `获利机会，利润分成方案，Telegram 推广` 


- **72 小时内赚取 10 万美元？**：一位用户向首批 **20 名感兴趣**的个人提供指导，教他们如何在 **72 小时内赚取 10 万美元**，但要求在收到利润后**返还 10%**。
   - 潜在参与者被鼓励发送好友请求或 **DM** 以了解更多详情，并可以通过[此链接](https://t.me/CharlesWilliam26)在 Telegram 上联系。
- **在 Telegram 上与 Charles 建立联系**：用户可以直接在 Telegram 上联系 **Charles William**，获取关于快速获利的个性化建议。
   - Charles 宣扬一种财富共享的心态，邀请个人通过 **Telegram 消息**加入他的盈利风险投资。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1321983037650632791)** (1 messages): 

> `赚取 10 万美元，Charles William 的提议，Telegram 推广` 


- **72 小时内赚取 10 万美元**：Charles William 提议指导前 **20 人**在 **72 小时**内开始赚取 **10 万美元**，并要求在收到利润后返还 10%。
   - *感兴趣的个人*被鼓励发送好友请求或私信 Charles 以获取更多信息。
- **通过 Telegram 联系 Charles**：拥有 **Telegram** 的个人可以通过提供的链接直接与 Charles 联系，进一步讨论获利机会。
   - 他以“在世界各地传播财富”为口号进行推广。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1321983782915543040)** (1 messages): 

> `72 小时内赚取 10 万美元，利润返还模式，Telegram 联系详情` 


- **学习如何快速赚取 10 万美元**：一位成员向首批 **20 名感兴趣的人**提供帮助，教授如何在短短 **72 小时内赚取 10 万美元**，并采用利润分成模式。
   - 感兴趣的个人被提示发送好友请求或**私信**以获取更多信息。
- **10% 利润返还提议**：提议的模式要求在收到利润后**返还 10%**，从而形成利润分成安排。
   - “问我如何操作！”（Ask me HOW!）是对那些好奇详情的人发出的行动号召。
- **通过 Telegram 直接联系**：感兴趣的人可以通过提供的链接在 **Telegram 上直接联系 Charles** 以获得即时帮助。
   - 该成员旨在**传播财富**，并鼓励通过社交媒体快速参与。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1321281654442037329)** (7 messages): 

> `No backpropagation training method, Energy-efficient model training, Random walk sampling technique, Discussion on gradient methods` 


- **无 Backprop，没问题？**: 一位用户质疑一种声称无需 **backpropagation** 或 **momentum** 即可运行的新训练方法的可行性，并对其对 ML 训练的潜在影响表示好奇。
   - 另一位成员对其在基础实验之外的实际效果表示怀疑，特别是梯度估计所需的 **128 次 forward passes**。
- **高能效训练突破**: 讨论中引用的一篇论文表明，这种新方法可以促进具有 **1.58B operations** 的训练，比传统方法减少 **97% 的能耗** 和 **90% 的内存**。
   - 它还提出了一种模型格式，能够将 **175B model** 存储在约 **20MB** 中，提高了对资源高效型 AI 能力的预期。
- **Random Walks 中的采样技术**: 一位成员描述了一种 **multidimensional random walk** 采样方法，该方法保留减少 loss 的 walk，同时丢弃较差的变体。
   - 社区对该方法的含义很感兴趣，特别是关于它与 gradient computation 的潜在关联。
- **Batch 与 Mini-Batch 梯度讨论**: 在对新训练方法的讨论中，一位用户将其与传统的 **batch versus mini-batch gradient** 技术进行了类比。
   - 这种比较旨在突出新算法可能带来的训练动态和效率方面的差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/torchcompiled/status/1872021986106650816">Ethan (@torchcompiled) 的推文</a>: 这是一个很酷的想法，但在 MNIST 玩具示例之外你不会有好的体验。没有 backprop 意味着需要... 128 次 forward passes，而得到的梯度估计与真实梯度的 cos similarity 仅为 0.009...</li><li><a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。</li><li><a href="https://x.com/_brickner/status/1871677392672219608">Will (@_brickner) 的推文</a>: 我起晚了，这是 CPU 实现：https://colab.research.google.com/drive/1hXzf5xB4INzMUNTlAB8CI1V10-JV7zyg?usp=sharing
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1321983133133967390)** (1 messages): 

> `Earning $100k, Profits reimbursement, Networking on Telegram` 


- **赚取 10 万美元的快速方案**: Charles William 提议帮助前 **20 位** 有兴趣在 **72 小时** 内开始赚取 **10 万美元** 的人，以此换取其利润的 **10% 返还**。
   - 他鼓励感兴趣的人向他发送好友请求或私信，并提供了一个 **Telegram 链接** 以便立即联系。
- **通过 Telegram 建立联系**: Charles 强调使用 **Telegram** 进行关于赚钱方案的沟通，确保联系他的人能够快速接入。
   - 他的个人资料宣传了*在世界各地传播财富*的概念，吸引潜在参与者直接参与。



**提到的链接**: <a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1321892593633591398)** (2 messages): 

> `Sparsification in PyTorch, Earning Strategies` 


- **理解 Sparsify_ 功能**: PyTorch 中的 `sparsify_` 函数需要一个由 `Sparsifier` 生成的、具有归零 dense matrix 的模型，以有效地压缩权重。
   - 任何归零的 dense model 都可以与 `sparsify_` 一起使用，从而允许自定义 masking 方案，如 [文档](https://github.com/pytorch/ao/blob/567cb46409f5f9a761429a87d27b1d5312642888/torchao/sparsity/README.md#24-sparsity) 中所述。
- **提供的快速现金赚取方案**: 一位成员提出了一个在 **72 小时内赚取 10 万美元** 的快速方案，要求感兴趣的人直接联系他们了解详情。
   - 参与者被要求在收到利润后返还 **10% 的利润**，该成员通过其 [Telegram](https://t.me/CharlesWilliam26) 账号进行推广。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.me/CharlesWilliam26">Charles William</a>: 在世界各地传播财富。</li><li><a href="https://github.com/pytorch/ao/blob/567cb46409f5f9a761429a87d27b1d5312642888/torchao/sparsity/README.md#24-sparsity">ao/torchao/sparsity/README.md at 567cb46409f5f9a761429a87d27b1d5312642888 · pytorch/ao</a>: PyTorch 原生 quantization 和 sparsity，用于训练和推理 - pytorch/ao
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1321983092226916494)** (1 条消息): 

> `赚取 $100k，利润分成模式，Telegram 推广` 


- **对 72 小时内赚取 $100k 的兴趣**：一位成员提议帮助前 **20 名感兴趣** 的个人在 **72 小时** 内开始赚取 **$100k**，并要求在收到利润后进行 **10% 的利润返还**。
   - *只有感兴趣的人* 应该发送好友请求或私信，他们可以直接在 [Telegram](https://t.me/CharlesWilliam26) 上联系他以获取更多详情。
- **利润分成作为商业模式**：提议的利润分成模式涉及客户返还其利润的 **10%**，使该成员与参与者的利益保持一致。
   - 这一策略旨在吸引那些准备好为潜在收益进行投入的有动力的个人，同时建立协作关系。
- **使用 Telegram 进行沟通**：该成员鼓励用户通过 **Telegram** 联系以获得即时帮助，并提供了其个人资料的链接。
   - 这种方式强调了在对财务增长感兴趣的人群之间的实时沟通和网络建立。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1321983614346461205)** (1 条消息): 

> `赚钱策略，Telegram 社交` 


- **72 小时内赚取 $100k**：**Charles William** 为前 **20 名感兴趣的个人** 提供指导，介绍如何在一种独特的返还模式下，在 **72 小时** 内开始赚取 **$100k**。
   - *潜在参与者必须发送好友请求或私信以获取详情*，并且在收到利润后需要进行 **10% 的返还**。
- **在 Telegram 上联系 Charles**：对他的提议感兴趣的个人被引导通过提供的链接在 **Telegram** 上进行联系。
   - Charles 鼓励潜在的赚钱者在他的 **Telegram 群组** 中交流并探索财富分配的选择。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1321930437467308114)** (2 条消息): 

> `在 iPad 上运行 .air 文件，筹款策略` 


- **关于 .air 文件兼容性的咨询**：一位用户询问将为 **macOS** 编译的 **.air 文件** 在 **iPad** 上运行是否可行。
   - 在当前消息中未看到提供明确说明的回复。
- **72 小时赚取 $100k 的提议**：一位用户声称他们可以帮助 **前 20 个人** 在 **72 小时内赚取 $100k**，并要求返还利润的 **10%**。
   - 感兴趣的个人被引导发送好友请求或 **私信**，并提供了一个 **Telegram 链接** 用于直接沟通。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1321983201778077736)** (1 条消息): 

> `快速赚取 $100k，利润分成模式，Telegram 推广` 


- **快速赚取 $100k 的策略**：Charles William 提议帮助前 **20 名感兴趣的个人** 在 **72 小时** 内开始赚取 **$100k**。
   - *鼓励感兴趣的各方发送好友请求或私信，以咨询如何开始。*
- **10% 利润返还协议**：参与者在收到收益后，需要向 Charles 返还其 **利润的 10%**。
   - *该模式旨在激励那些参与赚钱机会的人履行承诺。*
- **通过 Telegram 联系 Charles**：Charles 提供了一个 **Telegram** 链接用于直接沟通，以便为那些想要获取信息的人提供更快的响应。
   - *用户可以联系 [CharlesWilliam26](https://t.me/CharlesWilliam26) 以获得即时帮助。*



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>：在世界各地传播财富。

  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1321983662404796468)** (1 条消息): 

> `Earning $100k in 72 hours, Profit-sharing business model, Telegram outreach` 


- **学习快速赚取 $100k**：一名成员提议帮助前 20 名有兴趣在 **72 小时内赚取 $100k** 的人，并要求在收到利润后提取 **10% 的利润**作为报酬。
   - *Ask me HOW!*（问我如何操作！）以获取此机会的详情，有意向者请发送好友请求或 DM 以获取更多信息。
- **在 Telegram 上联系 Charles**：鼓励有意向的人通过 **Charles William** 的 [Telegram](https://t.me/CharlesWilliam26) 链接联系他以获得直接帮助。
   - Charles 通过声明他正在 *向全世界传播财富* 来推广他的服务。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1321983239308710041)** (1 条消息): 

> `$100k in 72 hours, Profit Reimbursement, Telegram Outreach` 


- **72 小时赚取 $100k 的机会**：一名成员提议帮助前 **20 名**有兴趣学习如何在 **72 小时**内开始赚取 **$100k** 的人。
   - 鼓励有意向的人通过好友请求或直接消息（DM）联系并 *询问如何操作*。
- **利润返还协议**：该提议包括在收到利润后返还 **10%**，以此激励参与和潜在合作。
   - 这创造了一个参与者在加入前必须考虑的直接利润分成环节。
- **鼓励直接通过 Telegram 联系**：有兴趣的人可以直接通过 [Telegram](https://t.me/CharlesWilliam26) 联系 **Charles** 以获得即时帮助。
   - 这种直接沟通旨在加快该提议与潜在参与者之间的联系。



**提到的链接**：<a href="https://t.me/CharlesWilliam26">Charles William</a>: Spreading the wealth around the world.

  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1321224837460660226)** (9 条消息🔥): 

> `Oreo 代码发布, Hugging Face TRL 库, ARC-AGI-2 仓库, Chollet 对 VLM 的看法, 1D 任务生成器` 


- **Oreo 代码尚未发布，但强化学习仓库表现亮眼**：目前没有可用的 **Oreo 代码**，但提到了如 [LaTRO](https://github.com/SalesforceAIResearch/LaTRO) 等值得关注的强化学习仓库。
   - 建议**对小模型使用 CPU**——在切换到云端 GPU 进行训练之前，先在本地进行准备。
- **查看 Hugging Face 的 TRL 资源**：鼓励成员探索 [Hugging Face 的 TRL 文档](https://huggingface.co/docs/trl/index)，了解通过强化学习训练 Transformer 模型的工具。
   - TRL 库涵盖了从 **Supervised Fine-tuning** 到 **Proximal Policy Optimization** 的完整工作流。
- **所有道路通向 ARC-AGI-2 仓库**：一位成员分享了他们在 [GitHub](https://github.com/open-thought/arc-agi-2) 上收集与 **ARC-AGI-2** 相关的材料和实验的意图。
   - 另一位成员表达了在未来一年学习并为该仓库做出贡献的热情。
- **Chollet 批评视觉语言模型 (VLM)**：Chollet 认为，与纯 LLM 相比，**VLM** 在克服基准测试挑战方面效果不佳，并指出 ARC-AGI 从根本上是一个 **2D 符号推理** 任务。
   - 相比之下，一位成员表示 **2D 位置编码** 似乎是有益的，并质疑较小任务所需的视觉编码器尺寸。
- **创新的 1D 任务生成器起步**：一位成员创建了 **1D 任务生成器**（目前有 75 种类型），以促进方法的快速迭代，并可能将发现推演到 2D 任务中，其代码可在 [GitHub](https://github.com/optozorax/arc_1d/) 上获得。
   - 欢迎贡献者为任务生成器做出贡献，同时可视化展示了各种任务格式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/mikb0b/status/1871573542627873182">来自 Mikel Bober-Irizar (@mikb0b) 的推文</a>: 当模型无法理解任务格式时，基准测试可能会产生误导，从而引入隐藏的阈值效应。如果总有一个人类能解决但 LLM 无法解决的更大版本...</li><li><a href="https://x.com/fchollet/status/1871759791703630189">来自 François Chollet (@fchollet) 的推文</a>: 如果 ARC-AGI 需要视觉感知，你会看到 VLM 的表现远超纯 LLM。在 2024 年的竞赛中，每个人都尝试了 VLM——但没有人获得更好的结果。每一个顶尖条目都使用了...</li><li><a href="https://huggingface.co/docs/trl/index">TRL - Transformer Reinforcement Learning</a>: 未找到描述</li><li><a href="https://github.com/open-thought/arc-agi-2">GitHub - open-thought/arc-agi-2: Building the cognitive-core to solve ARC-AGI-2</a>: 构建解决 ARC-AGI-2 的认知核心。通过在 GitHub 上创建账户来为 open-thought/arc-agi-2 的开发做出贡献。</li><li><a href="https://github.com/SalesforceAIResearch/LaTRO">GitHub - SalesforceAIResearch/LaTRO</a>: 为 SalesforceAIResearch/LaTRO 的开发做出贡献。</li><li><a href="https://github.com/optozorax/arc_1d/">GitHub - optozorax/arc_1d: ARC-AGI like tasks generators in 1D</a>: 类似 ARC-AGI 的 1D 任务生成器。通过在 GitHub 上创建账户来为 optozorax/arc_1d 的开发做出贡献。</li><li><a href="https://optozorax.github.io/arc_1d/">ARC 任务概览</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1321211730575102015)** (7 条消息): 

> `Podcast 与 Google News 集成、AI 生成的播客、总结 Pathfinder 冒险、新闻文章的 Audio Overviews、日常场景中的聊天机器人` 


- **Podcast 与 Google News 的集成引起关注**：一位成员建议将播客与 **Google News** 集成，通过总结前 **10 条故事**来创建包含短版和长版的动态内容。
   - 他们指出，允许用户提问的潜力可以增强互动性和参与度。
- **AI 在生成播客方面占据主导地位**：一位用户赞赏了一个 AI 生成的播客，该播客从哲学角度探讨了人生最重大的问题，并强调了其**风趣幽默的调侃（smurf-tastic banter）**。
   - 这种独特的格式结合了幽默与深度思考，承诺提供引人入胜的体验，定能吸引广泛的受众。
- **15 分钟总结 Pathfinder 故事**：一位成员分享了使用 AI 生成播客来总结 **Pathfinder 2** 的 **6 本系列丛书**的经验，为 GMs 提供了简明扼要的概述。
   - 这种叙事效率展示了 AI 在游戏叙事中的新颖用途。
- **令人印象深刻的文章 Audio Overviews**：一位用户报告称，生成的关于新闻和维基百科文章的 **Audio Overviews** 听起来自然且节奏感强，提升了听觉体验。
   - 他们强调了 AI 包含当前背景信息的能力，使内容显得更具相关性和吸引力。
- **聊天机器人创作电梯喜剧**：分享了一个关于聊天机器人在电梯中创造喜剧场景的奇思妙想，虚构的对话展示了它们的怪癖。
   - 这种俏皮的尝试突显了 AI 在平凡场景中进行幽默互动和产生笑料的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/NXjNoxVROos"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/CI18q_5Zawg"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1321212547877306418)** (54 条消息🔥): 

> `PDF 上传问题、Podcast 定制、语言设置、功能请求、AI Podcast 分享` 


- **PDF 上传问题仍然存在**：用户报告在尝试上传 PDF 文件时遇到错误消息，提示 *'an error occurred while uploading the font'*。
   - 一位用户被建议刷新页面以解决该问题，而其他用户也分享了类似的经历。
- **Podcast 定制需求**：一位成员对播客主持人在录制过程中经常偏离主题表示沮丧，希望内容能更加聚焦。
   - 另一位成员建议使用特定的提示词（prompts）来生成更具结构化的播客，这成功将一个宣传类播客转变为详细的教程。
- **调整语言设置**：一位用户询问如何强制 NotebookLM 仅以英文生成内容，即使在使用母语时也是如此。
   - 另一位成员建议退出 Google 账号，选择首选语言，然后重新登录。
- **功能改进建议**：讨论了是将建议作为功能请求还是 Bug 提交，并鼓励用户直接分享想法。
   - 一位成员强调了反馈选项的必要性，并强调了与工程团队沟通的重要性。
- **探索 AI Podcast 分享**：一位用户介绍了一个名为 *Akas* 的平台，可以在该平台上分享、嵌入 AI 生成的播客，或将其生成为 RSS feeds。
   - 他们强调了 Akas 作为 AI 生成内容与用户在播客中个人声音之间桥梁的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com.">Akas: 分享 AI 生成的播客</a>: Akas 是分享 AI 生成播客和个人声音的终极平台。随着越来越多由 AI（如 NotebookLM 等平台）创作的播客出现，Akas 提供了一个...</li><li><a href="https://notebooklm.google.com/notebook/df962099-9ee3-4a8a-a3d6-8fc9f6f34844">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=C1ahJ6M7XBg"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9qeiQ4x30Dk"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/6-MH83pxlbE?si=jcet51HQTI4SdK8Z"> - YouTube</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1321881882324897843)** (1 条消息): 

> `Report Generation Agent, LlamaParse, LlamaCloud` 


- **从零开始构建 Report Generation Agent**：@fahdmirza 制作的一段精彩视频展示了如何构建一个 **agentic workflow**，该工作流可以根据输入的格式化模板，针对一组 PDF 研究论文生成格式化的报告。
   - 该过程利用 **LlamaParse** 和 **LlamaCloud** 作为核心组件。点击[此处](https://t.co/o5jhvipERf)查看视频，点击[此处](https://t.co/0IHLaXZxGy)获取更多见解。
- **增强报告自动化**：讨论强调了从各种研究来源生成自动化报告的潜力，扩大了传统分析的范围。
   - 通过使用先进工具，该方法提升了研究报告的效率和一致性。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1321268106793979944)** (36 条消息🔥): 

> `LlamaIndex and OpenAI integration, DocumentContextExtractor proposals, Tokenization and truncation issues, Generating LlamaIndex documentation, Payroll PDF parsing solutions` 


- **LlamaIndex 可能缺乏批处理功能**：一位成员询问 **LlamaIndex LLM class** 是否可以与 OpenAI/Anthropic 的 **Message Batching API** 对接，但回复指出目前的抽象层主要侧重于实时交互。
   - 另一位成员建议改用原始的 OpenAI client，并表示愿意审查任何旨在增强功能的 Pull Request。
- **DocumentContextExtractor 提供节省成本的批处理**：分享了一个关于 **DocumentContextExtractor** 的创新用例，强调了批处理如何将成本降低 **50%** 并提供无状态解决方案，从而允许在非高峰时段进行处理。
   - 该成员提到，通过这种方法，用户无需让 Python 脚本无限期运行，只需稍后回来查看处理状态即可。
- **Tokenization 限制引发不满**：一位用户对 **LlamaIndex tokenizer** 仅提供编码（encoding）而没有解码（decoding）功能表示沮丧，质疑这种限制的实用性。
   - 回复建议使用 splitter 并管理 chunk 大小，但一位用户幽默地考虑删除截断功能，将问题归咎于用户提交了过大的文档。
- **请求多种格式的 LlamaIndex 文档**：另一位成员询问如何获取 PDF 或 Markdown 格式的 **LlamaIndex 文档** 以构建 RAG 应用，引发了关于生成文档可能方法的讨论。
   - 回复指出，以所需格式生成文档是可行的，并建议通过私信继续交流。
- **解决工资单 PDF 解析挑战**：一位成员表示在使用 **LlamaParse** 解析工资单 PDF 时遇到困难，寻求更好的替代方案。
   - 回复指出 LlamaParse 应该表现良好，特别是在其 premium 模式下，认为它可能对该成员的需求有效。


  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1321901237683359826)** (3 messages): 

> `Unstructured RAG, LangChain, Unstructured IO, Athina AI, LlamaIndex` 


- **Unstructured RAG 博客强调了明显的优势**：分享了一篇关于使用 **LangChain** 和 **Unstructured IO** 实现 **Unstructured RAG** 的博客，讨论了传统系统在处理图像和表格等非结构化数据时面临的挑战。
   - 它强调了 Unstructured IO 在组织原始数据方面的作用，使检索增强生成 ([RAG](https://hub.athina.ai/athina-originals/end-to-end-implementation-of-unstructured-rag/)) 更加容易。
- **RAG 系统在处理非结构化数据时面临挑战**：讨论指出，传统的 **RAG** 在处理格式不一致的数据时表现不佳，这使信息提取和处理变得复杂。
   - 像 Unstructured 这样的工具有助于将非结构化数据转换为有组织的格式，从而在 RAG 流水线中获得更好的性能。
- **分享了构建 RAG 流水线的实施策略**：博客概述了实施 **Unstructured RAG** 的步骤，涉及使用 **FAISS** 等库来处理 PDF 并创建 Embeddings。
   - 它详细介绍了如何将文档处理与 **LLM 集成**相结合，并使用自定义 Prompt 根据上下文生成准确的响应。
- **提出了使用 Athina AI 的评估方法**：建议使用 **Athina AI** 进行**可选评估**，以评估 RAG 流水线的性能和准确性，从而促进优化。
   - 这种评估将有助于验证 RAG 系统，并确保其在实际应用中的有效性。
- **关于 LlamaIndex 相关性的澄清**：一位用户质疑分享的博客与 **LlamaIndex** 之间的联系，促使一名成员证明将其作为资源包含在内的合理性。
   - 分享的意图是通过提供有关高效 RAG 实施的见解，使一般讨论组受益。



**提到的链接**：<a href="https://hub.athina.ai/athina-originals/end-to-end-implementation-of-unstructured-rag/">端到端指南：实施 Unstructured RAG 系统</a>：了解实施 Unstructured RAG 系统的完整过程。通过这份全面的 Athina AI Hub 原创指南提升 AI 性能！

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1321485354447929395)** (16 messages🔥): 

> `Hugging Face Checkpoints, Fine-Tuning VLMs, Loss Calculation in Trainer` 


- **Checkpoint 中缺失 Optimizer States**：一位用户对 [Hugging Face checkpoints](https://huggingface.co/EleutherAI/pythia-2.8b/tree/main) 中缺失 Optimizer States 表示担忧，认为他们的模型可能需要这些状态。
   - 另一位成员确认，他们认为 Optimizer States 是由 Checkpoint 代码保存的。
- **Fine-Tuning VLMs 的资源**：讨论集中在 Fine-Tuning Vision Language Models (VLMs) 的挑战上，并指出具体方法因模型而异。
   - 一位用户强调 [LLaVA](https://github.com/haotian-liu/LLaVA) 代码库拥有 Fine-Tuning 脚本，并为此目的被广泛使用。
- **Fine-Tuning VLMs 需要模型细节**：用户讨论了各种 VLM，并指出许多开源选项都有自己的 Fine-Tuning 脚本，例如 [Qwen-VL](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py) 和 [InternVL](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_lora.sh)。
   - 对话强调了各个 VLM 所需方法的变异性。
- **Hugging Face Trainer 中的自定义 Loss 处理**：一位用户在使用 Hugging Face 的 Trainer 时，就如何为 Causal Language Modeling 自定义 Loss 函数寻求建议，重点是如何处理 Padded Tokens。
   - 另一位成员建议传递一个自定义的 Collator 来调整 Prompt Labels，并参考 TRL 库以获得进一步帮助。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1321247999732613181)** (17 条消息🔥): 

> `LLM 延迟降低技术，LLM 在工程应用中的表现，LLM 与优化模型的对比分析，LLM 的自我改进，关于 token 预测目标的见解` 


- **探讨了 LLM 的延迟降低技术**：一位用户询问了关于减少大语言模型（LLM）推理延迟的技术汇编，特别是针对 CUDA 或 Triton 级别的优化。
   - 持续关注了解简化 LLM 处理时间的有效策略。
- **LLM 在工程应用中取得进展**：一位用户分享了一个讨论大语言模型在工程领域应用的出版物链接，并对该主题表示了热情。
   - 然而，他们指出某些研究平台上的访问限制可能令人沮丧。
- **开源模型在 function calling 方面超越 GPT-4**：一个帖子强调了一项突破，即 Outlines 的结构化生成结合 Phi-3-medium 在 function calling 任务上达到了 96.25% 的准确率，超过了 GPT-4 的表现。
   - 这一成就反映了 AI 开发中社区和开源协作的力量。
- **讨论了提升 LLM 性能的自我改进方法**：一篇论文强调了在缺乏大量人工标注数据的情况下，探索模型自我改进的必要性，重点关注响应的多样性和外部奖励等因素。
   - 该研究旨在增强对复杂推理任务中自我改进迭代方法的理解。
- **对跳过预训练阶段的批评**：一位用户质疑了跳过以 next token prediction 为目标的预训练阶段，而是在训练期间直接关注整个预训练数据集的影响。
   - 另一位用户建议，虽然结合训练任务可能被证明是有效的，但可能会导致整体训练过程变慢。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.dottxt.co/oss-v-gpt4.html">Beating GPT-4 with Open Source</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2412.17747">Deliberation in Latent Space via Differentiable Cache Augmentation</a>：通过生成并关注中间推理步骤，使大语言模型（LLM）能够“更多思考”的技术在解决复杂问题方面显示出前景。然而，标准的...</li><li><a href="https://arxiv.org/abs/2305.14325">Improving Factuality and Reasoning in Language Models through Multiagent Debate</a>：大语言模型（LLM）近年来在语言生成、理解和 few-shot 学习方面展示了卓越的能力。大量工作探索了它们的性能如何...</li><li><a href="https://techxplore.com/news/2024-12-machine-perovskite-solar-cells-efficiency.html">Machine learning helps researchers develop perovskite solar cells with near-record efficiency</a>：一个国际科学家团队利用机器学习帮助他们开发出效率接近纪录的钙钛矿太阳能电池。在发表于《Science》杂志的论文中，该小组描述了...</li><li><a href="https://arxiv.org/abs/2412.16112">CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up</a>：Diffusion Transformers (DiT) 已成为图像生成领域的领先架构。然而，负责建模 token 间关系的注意力机制的二次复杂度...</li><li><a href="https://modal.com/blog/llama-human-eval">Beat GPT-4o at Python by searching with 100 dumb LLaMAs</a>：通过搜索和评估扩展较小的开源模型，以匹配前沿模型的能力。</li><li><a href="https://arxiv.org/abs/2412.17256">B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners</a>：在缺乏针对复杂推理任务的大量人工标注数据的情况下，自我改进（模型在其自身输出上进行训练）已成为提升性能的主要方法...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>：通过在 GitHub 上创建一个账户来为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1321328347523059785)** (6 messages): 

> `GPT-2 Token Activations, BOS Token Discussion` 


- **首个 Token 激活值飙升**：在对 **gpt2-small** 进行测试后发现，第一个 Token 的激活值约为 **3000**，显著高于其余 Token 约 **100** 的激活值。
   - 这一观察结果是在从激活值中减去均值后得出的。
- **GPT-2 中 BOS Token 的混淆**：关于 **GPT-2** 中的 **BOS token** 存在讨论，一位成员坚持认为它没有 BOS token，因为默认的 tokenizer 不会添加它。
   - 然而，有反对意见提到 **GPT-2 确实有 BOS token**，尽管无论是否添加，激活范数（activation norms）看起来都是一致的。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1321551221160673281)** (1 messages): 

> `Encoder-Free VLMs, Video VLMs, Encoder Efficiency, Fuyu Model Series, EVE Model` 


- **无编码器 VLM 受到关注**：人们对 **无编码器视觉语言模型 (encoder-free vision-language models)** 的兴趣日益浓厚，特别是在编码器效率成为主要问题的视频 VLM 领域。
   - 最近一篇名为 **EVE: Encoder-Free Vision-Language Models** 的 [NeurIPS 论文](https://github.com/baaivision/EVE) 探索了这一方向，标志着对传统 CLIP 风格架构的转变。
- **对 Fuyu 模型性能的担忧**：讨论强调，虽然 **Fuyu 模型系列** 旨在解决编码器问题，但在实践中表现并不理想。
   - 这引发了关于此类架构在提升视频 VLM 整体端到端质量方面可行性的疑问。
- **寻求无编码器方法的反馈**：一位成员正在寻求有关无编码器 VLM 研究方向的**评论和建议**，反映出希望应对当前挑战的愿望。
   - 他们强调需要深入了解如何在视觉媒体背景下提高编码器效率和输出质量。



**提到的链接**：<a href="https://github.com/baaivision/EVE">GitHub - baaivision/EVE: [NeurIPS&#39;24 Spotlight] EVE: Encoder-Free Vision-Language Models</a>: [NeurIPS&#39;24 Spotlight] EVE: Encoder-Free Vision-Language Models - baaivision/EVE

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1321232997151670302)** (8 messages🔥): 

> `Proof in Lean Bounty, BITCAST Const Folding, Matching Engine Performance Bounties, Tinygrad Updates, Performance Optimization` 


- **寻求 Lean 证明悬赏**：一位成员表示有兴趣开始研究 **Lean 悬赏**系统中的证明工作并寻求帮助。
   - 另一位成员建议查看[新版 tinygrad 笔记](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md)以获取相关信息。
- **通过 BITCAST 常量折叠进行优化**：有人提问是否有兴趣实现 **BITCAST 常量折叠 (const folding)** 以优化编译时间。
   - 一位成员给出了积极回应，并询问该任务应重点关注哪个目录。
- **分析匹配引擎性能悬赏**：开始讨论与匹配引擎相关的性能悬赏，并链接到 issues 页面作为参考。
   - 一位用户提供了见解，详细说明 **model lower** 结果已经达到了 **25ms**，并对之前的解决情况提出了疑问。
- **澄清重写悬赏**：一位成员澄清说，根据之前的讨论，他们的重点是匹配引擎悬赏中的 **rewrite** 部分。
   - 他们引用了与该悬赏相关的旧 PRs，并暗示这些 PRs 已经过时。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md">tinygrad-notes/20241217_st.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建一个账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/issues/4878)">Issues · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1321831601008672810)** (31 messages🔥): 

> `Performance Comparisons, Tinygrad Model Integration, Beam Search Efficiency, GPU Compatibility, Kernel Caching` 


- **Tinygrad 与 PyTorch 的性能对比**：Tinygrad 在 CUDA 上的前向传播（forward pass）大约需要 **800ms**，而 PyTorch 仅需约 **17ms**。但通过使用 jitting 和 beam search，性能有望得到提升。
   - 针对速度不一致的问题，用户指出在使用 jitting 和 beam search 后，性能潜力应能匹配甚至超越 PyTorch。
- **模型输入处理问题**：一位用户发现在循环中需要重新创建输入 Tensor 才能确保输出匹配，但这显著降低了 Tinygrad 的处理速度。
   - 使用 `tiny_input.clone()` 会导致与 CUDA 分配器（allocator）相关的属性错误，这引发了进一步的调查。
- **模型变更的集成**：来自 PR **#8309** 的更改已成功合并，改进了需要克隆输入时的功能，并达到了匹配 PyTorch 的速度。
   - 此次集成强调了进行回归测试（regression tests）的必要性，以确保在进行更改时性能的稳定性。
- **RTX 4070 GPU 讨论**：用户讨论了特定的硬件配置，确认使用的是 **RTX 4070** 笔记本 GPU，驱动版本为 **535.183.01**，CUDA 版本为 **12.2**。
   - 用户对开源 Kernel 驱动的潜在问题表示担忧，并请求提供额外的系统日志。
- **Beam Search 中的 Kernel 缓存**：在询问 beam search 生成的 Kernel 是否可以重用时，确认了这些 Kernel 会被缓存以提高效率。
   - 进一步讨论了将这些 Kernel 分发到其他同类机器以避免重复搜索的可能性。



**提到链接**：<a href="https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py">fish-speech/fish_speech/models/text2semantic/llama.py at main · fishaudio/fish-speech</a>：SOTA 开源 TTS。可以通过在 GitHub 上创建账号为 fishaudio/fish-speech 的开发做出贡献。

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1321207165079388191)** (13 messages🔥): 

> `Christmas Greetings, Re-ranker Pricing Inquiry, AI and ML Learning Journey` 


- **圣诞祝福传递**：多位成员热情地互相祝愿 **圣诞快乐**，在频道中通过问候和表情符号表达节日气氛。
   - *Mapler* 添加了一张与使用 Cohere 构建应用相关的图片，进一步增添了节日氛围。
- **Re-ranker 定价查询**：*Mecatron* 询问了 re-ranker 的定价，*Mapler* 随后提供了 [Cohere 定价页面](https://cohere.com/pricing)的链接。
   - 定价详情列出了不同模型的成本，其中 Command R+ 的价格为每 1M tokens 输入 **$2.50**，输出 **$10.00**。
- **新人自我介绍**：*一位新成员* 表达了对学习 AI 和 ML 的兴奋之情，特别是作为该领域的初学者，重点关注 LLM。
   - 他们希望通过参与社区获得知识并在职业生涯中取得进步，并收到了欢迎回复。


<div class="linksMentioned">

<strong>提到链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/christmas-eve-snoopy-santa-claus-bell-ring-the-bell-gif-7322926">Its Christmas Eve GIF - Christmas Eve Snoopy Santa Claus - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://cohere.com/pricing">Pricing - Affordable Enterprise Generative AI Models</a>：直接通过我们的 API 访问模型，以创建可扩展的生产工作负载。 
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1321813802810998795)** (6 messages): 

> `CMD-R updates, R7B beginnings, HuggingFace finetunes, User feedback` 


- **CMD-R 的未来更新**：一名成员询问了 **CMD-R** 的未来更新计划，表现出对其开发的浓厚兴趣。
   - 关于更新的讨论目前仍处于推测阶段，社区正在等待官方计划或公告。
- **R7B 的好奇开端：两个答案**：一名成员对 **R7B** 的起源表示好奇，并分享了一张带有“two ans”评论的图片。
   - 另一名成员觉得这种情况很奇怪，询问其发生频率，并得到了轻松幽默的回应。
- **在 HuggingFace 上微调 Command R**：一位成员推测是否存在阻止在 **HuggingFace** 上进行微调和分享的**条款**，因为目前不寻常的 CMD-R 微调版本似乎很少见。
   - 他们思考这是由于限制还是仅仅因为社区兴趣不足，反映了 CMD-R 的现状。
- **用户对 CMD-R 的反应**：由于近期缺乏帖子或活动，引发了关于成员们是否“忽视了 CMD-R”的讨论。
   - 这表明社区对 CMD-R 的参与度和热情可能存在缺口，引发了进一步的探究。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1321387523494379615)** (15 messages🔥): 

> `LLM University, Command R model, Command R+ performance` 


- **通过 LLM University 掌握 NLP**：Cohere 提供了一个名为 [LLM University](https://cohere.com/llmu) 的学习中心，提供由专家主导的课程和指南，以掌握企业级 AI，涵盖 NLP 和 LLM。
   - 探索完整课程以建立该领域的基础知识和实践技能。
- **Command R 概览与能力**：Command R 是一款针对对话交互优化的 LLM，能够处理具有 **128,000 token 上下文长度**的长上下文任务。
   - 它在检索增强生成 (RAG) 方面表现出色，并支持生成 **10 种语言**的文本，强调了其在多语言任务中的强大性能。
- **Command R+ 的增强性能**：Command R+ 被誉为性能最强的 LLM，在多样化的文本集上进行训练，以完成复杂的 RAG 任务。
   - 该模型在需要**多步工具使用 (multi-step tool use)** 的工作流中表现尤为强劲，扩展了 LLM 在生产环境中的能力。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1321760970413965342)** (1 messages): 

> `Voice to Voice chat app, Music Generation using Generative AI, DNN-VAD, NLP projects, ASR project` 


- **AI 工程师寻求合作**：一名成员表达了合作意向，介绍自己是一名在 **DNN-VAD、NLP 和 ASR** 项目方面拥有经验的 **AI 工程师**。
   - 他们强调了最近在**语音对语音聊天应用**以及使用 stereo-melody-large 模型从**文本提示生成音乐**方面的工作，并表示：“我想与你们合作。”
- **节日问候**：该成员向大家致以温馨的“圣诞快乐！”问候，为讨论增添了友好的氛围。
   - 这一问候是对其专业介绍的轻松补充。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1321364602168283157)** (6 messages): 

> `io_uring networking, Mojo swag, Modular merchandise` 


- **探索用于网络编程的 io_uring**：一名成员询问了展示如何在网络编程中利用 **io_uring** 的示例，尽管他们对其了解有限，但表达了深入学习的愿望。
   - 有人建议“从 man 手册开始”，以引导对 **io_uring** 的查询。
- **对 Mojo 周边的兴奋**：一名成员分享了收到 **Mojo 周边 (swag)** 的感激之情，感谢团队协助将其运送到偏远地区。
   - 他们上传了一张照片展示新装备，引发了聊天中其他人的热情。
- **Modular 商品非常抢手**：成员们讨论了 **Modular 商品**吸引人的地方，强调它们可能会非常受欢迎。
   - 关于 **T恤** 质量以及 **贴纸** “非常酷”的评论表明了对产品的正面评价。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1321449105918136330)** (19 messages🔥): 

> `导入 String 模块问题、StringRef 与崩溃原因、Read Until Delimiter 中的 EOF 测试、关于 Copyable Trait 的担忧` 


- **导入 String 模块导致错误**：一位成员在将导入语句从 `import collections.string` 更改为 `import src.collections.string` 时遇到错误，这表明其设置中的模块路径存在问题。
   - 另一位成员指出，导入语句中应省略 `src` 部分，因为标准示例中并未出现该部分。
- **StringRef() 需要长度检查**：在调查一次崩溃后，一位成员建议 `StringRef()` 应该验证接收到的长度是否不为负数，因为向 `memcpy` 传递负长度会导致崩溃。
   - 一位社区成员承认 `StringRef` 是不安全的，建议改用 `StringSlice`。
- **在 Read Until Delimiter 中测试 EOF**：一位成员确认他们添加了一个测试，以确保 `read_until_delimiter` 会抛出 EOF，并链接到了记录此工作的 GitHub commit。
   - 可以在[此处](https://github.com/mahiro21h/mojo/commits/fix-input-segfaults-on-eof/)查看该 commit。
- **对 Copyable Trait 设计的担忧**：一位成员对 `Copyable` 和 `ExplicitlyCopyable` trait 的设计表示担忧，这些讨论已在 Modular 论坛上分享。
   - 随着社区对当前实现的评估，这次讨论可能会导致潜在的设计变更。



**提及的链接**：<a href="https://github.com/mahiro21h/mojo/commits/fix-input-segfaults-on-eof/">Commits · mahiro21h/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 mahiro21h/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1321246936325558424)** (2 messages): 

> `Modular Stack Kernel、MAX 与 XLA 编译时间` 


- **关于 Modular Stack Kernel 的咨询**：一位成员询问是否存在专门用于 **modular stack 的 kernel**。
   - 这一咨询突显了人们对优化 modular 实现中 kernel 支持的持续关注。
- **MAX 被定位为 XLA 的竞争对手**：一位成员建议 **MAX** 可以作为 **XLA** 的竞争对手，特别是批评了后者的编译时间。
   - 在关于性能优化的讨论中，有一个重点被强调：*JAX 编译时间慢是 XLA 的责任*。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1321263435199287387)** (1 messages): 

> `PyN8N、DSLModel、AI 工作流创建` 


- **加载 PyN8N 网站时遇到困难**：一位用户报告说 [PyN8N 网站](https://pypi.org/project/pyn8n/) 的某个必要部分无法加载，可能是由于浏览器问题。
   - 他们建议检查连接、禁用广告拦截器或尝试使用不同的浏览器来解决问题。
- **通过 DSLModel 提供 DSPy 支持**：讨论强调 **DSPy** 通过 **DSLModel** 提供支持，从而增强了功能。
   - 这种集成允许用户利用高级特性来获得更好的性能。
- **AI 辅助创建节点工作流**：据指出，**PyN8N 客户端** 允许用户利用 AI 生成节点和工作流。
   - README 被描述为**愿景式**的，展示了该工具的潜力，而客户端本身已经可以使用。



**提及的链接**：<a href="https://pypi.org/project/pyn8n/">无标题</a>：未找到描述

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1321336856071114754)** (12 条消息🔥): 

> `NotebookLM 内联溯源, Jekyll 术语表脚本, Typing.TypedDict 用法, 用于输出字段设计的 Pydantic` 


- **探索 NotebookLM 内联溯源**：一位成员询问了 **NotebookLM 内联溯源 (inline sourcing)** 的运作方式，表示对其实现方式感兴趣。
   - *该查询未提供更多额外信息*。
- **Jekyll 术语表生成脚本**：一位成员分享了一个 [Jekyll 脚本](https://gist.github.com/dbreunig/3cef9293cb253f9192d5b4974c1367a3)，该脚本集成了 **DSPy** 进行 LLM 交互，用于生成关键词术语表。
   - 其输出内容包括 **Artificial General Intelligence** 等术语的详细条目，便于进一步的微调和编辑。
- **发现 Typing.TypedDict**：一位成员提到发现了 `typing.TypedDict`，表示这是关于 Python 类型提示（Type Hinting）的一个学习时刻。
   - 另一位成员评论了它带来的挑战，强调了其中涉及的复杂性。
- **输出字段的设计考量**：针对在数组中返回多个实例的场景，讨论了输出字段的设计，质疑当前结构的优雅性。
   - 有建议提出利用带有字段描述的 **pydantic.BaseModel** 来改进这种输出设计。
- **使用 Jekyll 脚本完成索引**：该成员重申，生成术语表的脚本运行效果良好，可以生成接近完成的索引，随后可进行手动最终定稿。
   - 他们还针对输出字段中使用的 *冗长且不美观的描述参数 (long ugly description parameter)* 提出了潜在的设计问题。



**提及的链接**：<a href="https://gist.github.com/dbreunig/3cef9293cb253f9192d5b4974c1367a3">一个从 Jekyll 文章中生成关键词术语表的脚本。我们使用 DSPy 来处理 LLM 交互；它有助于处理样板提示词上下文，并将响应解析为 Pydantic 对象。要运行此脚本，请将其放入 Jekyll 站点目录中名为 'scripts'（或其他名称）的文件夹中。然后插入你的 Anthropic API 密钥（或将 DSPy 指向你选择的 LLM 端点）。它将在你的 '_data' 目录中输出一个名为 'glossary.yaml' 的 YAML 文件。</a>：一个从 Jekyll 文章中生成关键词术语表的脚本。我们使用 DSPy 来处理 LLM 交互；它有助于处理样板提示词上下文，并将响应解析为 Pydantic 对象...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1321464271682404452)** (8 条消息🔥): 

> `证书发放, 证书申领表, 下次课程开始日期` 


- **证书发放时间线**：根据一位成员的帖子，证书将于 **1 月底** 前发放。更多详情请参阅[此处的原始帖子](https://discord.com/channels/1280234300012494859/1293323662300155934/1321147373652541511)。
- **遗漏证书申领表**：一位成员询问在完成所有其他要求但未填写 **证书申领表 (certificate declaration form)** 的情况下是否能获得证书。另一位成员澄清说，遗憾的是，如果不提交该表格，将无法获得证书。
- **关于后续课程的问题**：在关于认证的讨论之后，成员们对 **下一期课程** 的时间线表示关注。据提到，**春季** 将会有另一场课程。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1321445899267735564)** (7 messages): 

> `Open Interpreter API, OCR functionality, Desktop version release, Voice to Voice chat app, QvQ with Open-Interpreter` 


- **Open Interpreter API 展示了其精准度**：[Open Interpreter API](https://openinterpreter.com/) 通过使用自然语言查询和显示表示，提供了以 **单像素精度 (single-pixel precision)** 定位视觉控件的能力。
   - 用户分享了在项目中使用该 API 的 [Python 示例](https://api.openinterpreter.com/v0/point/)。
- **OCR 功能面临问题**：一名成员提到，该 API 旨在通过 **OCR** 识别屏幕上的图标和文本，但据报告该功能似乎已失效。
   - 另一位用户确认他们目前尚未收到成功的响应。
- **关于桌面版发布的询问**：一位用户询问 Open Interpreter 的 **桌面版本 (desktop version)** 何时发布。
   - 这个问题反映了终端用户对更广泛易用性的兴趣。
- **AI 工程师寻求合作**：一名成员介绍自己是具有 DNN-VAD、NLP 和 ASR 经验的 **AI 工程师**，并表达了对 Generative AI 相关项目合作的兴趣。
   - 他们重点介绍了最近在 **Voice to Voice 聊天应用**和根据文本提示进行 **Music Generation** 方面的工作。
- **关于 QvQ 和 Open-Interpreter OS 模式的讨论**：一位用户询问 **QvQ** 在 **OS 模式**下与 Open-Interpreter 集成时将如何运作。
   - 这表明社区内正在就互操作性和功能性进行持续讨论。



**提到的链接**：<a href="https://api.openinterpreter.com/">no title found</a>: no description found

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1321863340175261736)** (3 messages): 

> `UI Features, Copying Code, Keyboard Shortcuts` 


- **AI 生成的代码没有专门的复制按钮**：一名成员注意到聊天界面 UI 中缺少用于 **AI 生成代码**的专用“复制”按钮，并寻求确认其他用户是否也有同样的观察。
   - 他们对就此事提供的任何帮助表示感谢。
- **剪切和粘贴功能问题**：另一名成员确认传统的鼠标剪切和粘贴功能在聊天 UI 或配置页面中不起作用，但 **Control-C 和 Control-V** 是可以使用的。
   - 这一澄清旨在帮助那些在复制过程中遇到困难的用户。
- **关于新模板使用的询问**：一名成员用法语询问是否有人成功地 **使用新模板进行了编写**。
   - 这个问题表明社区对探索可用新功能的兴趣。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1321425132526501888)** (2 messages): 

> `TTS dataset creation, Audio segmentation, Using Whisper for transcription` 


- **寻求关于从长音频构建 TTS 数据集的建议**：一名成员正在寻求关于使用几个长音频文件（每个超过一小时）构建 **TTS 数据集**的建议，目标是对其进行分割和转录。
   - 他们特别想知道如何高效地拆分这些样本，以及可以使用哪些工具或方法来完成这项任务。
- **使用 Whisper 进行句子检测**：另一名成员建议 **Whisper** 可以检测句子，并提议将其作为按句子长度拆分音频文件的工具。
   - 这可能会简化 TTS 应用中准备音频转录的过程。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1321627007225958471)** (1 messages): 

> `ML ops frameworks, HPC environments, Guild AI stability, DIY ops framework` 


- **为 HPC 环境寻找 ML Ops 框架**：一名成员正在寻找适用于 **HPC 环境**的 **ML ops 框架**，强调了对稳定性和**成本效益**的需求。
   - 他们提到 **Guild AI** 是一个潜在的选择，但对其**稳定性**表示怀疑，并表示更倾向于轻量级的自托管解决方案，而非 SaaS。
- **服务器管理的挑战**：该成员暗示设置用于托管的服务器可能过于耗费人力，这是他们希望避免的。
   - 他们表示愿意自己**编写一个简单的 ops 框架**，而不是去管理服务器。


  

---


---


---


---


---


{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}