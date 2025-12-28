---
companies:
- nous-research
- openai
- mistral-ai
- hugging-face
date: '2024-01-09T07:39:51.817056Z'
description: '**Nous Research AI Discord** 的讨论强调了几个关键话题，包括在 **Obsidian 项目** 中使用 **DINO**、**CLIP**
  和 **CNN**。社区分享了一篇关于 **DistAttention** 和 **DistKV-LLM** 等分布式模型的研究论文，以应对云端 **LLM**（大语言模型）服务面临的挑战。另一篇题为《无需微调即可自我扩展
  LLM 上下文窗口》（Self-Extend LLM Context Window Without Tuning）的论文指出，现有的 **LLM** 本身就具备处理长上下文的能力。社区还讨论了
  **Mixtral** 等 AI 模型（因其 **32k 上下文窗口**而备受青睐），并将其与 **Mistral** 和 **Marcoroni** 进行了对比。其他话题还包括层级嵌入、代理式检索增强生成（**RAG**）、用于微调的合成数据，以及
  **LLM** 在石油和天然气行业的应用。此外，会上还宣布推出了包含 10 亿个嵌入向量的 **AgentSearch-V1** 数据集。讨论内容还涵盖了**混合专家模型（MoE）**的实现以及小型模型的性能表现。'
id: 2f78c3db-d078-41d9-96c1-9feb81775d69
models:
- mixtral
- mistral
original_slug: ainews-182024-the-four-wars-of-the-ai-stack
people: []
title: 2024年1月8日：AI 技术栈的四场战争
topics:
- context-window
- distributed-models
- long-context
- hierarchical-embeddings
- agentic-rag
- fine-tuning
- synthetic-data
- oil-and-gas
- embedding-datasets
- mixture-of-experts
- model-comparison
---

<!-- buttondown-editor-mode: plaintext -->今天 Discord 社区没发生什么大事，所以是时候安利一下我们的 Latent Space 2023 年度回顾了！

https://www.latent.space/p/dec-2023


请享用！

---

**目录**

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **“Obsidian 项目简介”**：在 Obsidian 项目中，`@qnguyen3` 简要提到他们正在利用 DINO、CLIP 和 CNNs。

- **“解决云端 LLM 的担忧”**：关于云端大语言模型 (LLM) 服务设计挑战的讨论引起了关注，`@maxwellandrews` 分享了一篇[研究论文](https://huggingface.co/papers/2401.02669)，该论文通过 DistAttention 和 DistKV-LLM 等分布式模型提出了解决方案。

- **“自扩展上下文窗口的胜利”**：`@kenakafrosty` 分享了一篇名为《Self-Extend LLM Context Window Without Tuning》的论文，认为现有的 LLM 本身就具备处理长上下文情况的能力，并分享了[论文链接](https://arxiv.org/abs/2401.01325)以及相关的 [Twitter](https://twitter.com/sdand/status/1743695855545426362) 和 [GitHub](https://github.com/datamllab/LongLM) 讨论。

- **“AI 为你端上早晨的咖啡？”**：用户们对 `@adjectiveallison` 分享的一条推文展开了玩笑和思考，该推文关于一个名为 “Figure-01” 的 AI 机器人，它声称在观察人类后**学会了冲咖啡**。对话随后扩展到将该项目与 `@leontello` 分享的另一个 AI 程序 **ALOHA** 进行对比。

- **“会学习和教学的 LLM”**：`@deki04` 分享了一个包含 LLM 全面课程的 [GitHub 仓库](https://github.com/mlabonne/llm-course)链接，引发了由 `@leontello` 和 `@vincentweisser` 主导的关于模型改进及其实际应用的讨论。

- **“嵌入，还是不嵌入”**：`@gabriel_syme` 建议，尽管分层 Embedding（嵌入）没有表现出预期的性能提升，但对于 OAI 模型来说，它可能仍然是一个必要的补充。

- **“Agentic RAG 趋势”**：`@n8programs` 宣布计划尝试 Agentic RAG，这是一种根据输入生成搜索查询并收集数据直到积累足够的模型。

- **“AI 工程师 LLM 微调指南”**：`@realsedlyf` 征求关于为特定领域微调语言模型所需的合成数据创建最佳方法的见解。

- **“石油和天然气行业拥抱 LLM 分析”**：`@kapnap_n` 详细介绍了 LLM 在一个不寻常领域的应用——分析石油和天然气行业的井下井筒数据。

- **“AgentSearch 数据集发布！”**：`@teknium` 推广了新发布的 AgentSearch-V1 数据集，并分享了 `@ocolegro` 的[一条推文](https://fxtwitter.com/ocolegro/status/1744207765671657923?s=46)链接，宣布提供涵盖 Wikipedia、Arxiv 等内容的 10 亿个 Embedding 向量。

- **“LLM 讨论——揭秘与建议”**：'#ask-about-llms' 频道进行了一些关于 LLM 不同方面的引人入胜的辩论，例如 KV_Cache 的实现、MoE 与 **Mistral** 的比较，以及 TinyLLAM 和 Lite LLAMAS 之间的性能差异。

- **“窥见一线希望”**：`@kenakafrosty` 提出了小模型可能拥有比看起来更强的处理能力的观点，引发了关于小模型饱和点的对话。

- **“探索模型合并技术的实现”**：关于 MoE（Mixture of Experts）实现的 Notebook 以及 PointCloud 模型限制的询问，引发了 `@teknium` 和 `.beowulfbr` 之间富有洞察力的交流，并分享了 [Mergekit GitHub 链接](https://github.com/cg123/mergekit)作为参考。

- **“Mixtral 因其宽敞的上下文窗口而受到青睐”**：`@gabriel_syme` 和 `@teknium` 表达了对 Mixtral 模型的偏好。尽管还有 Mistral 和 Marcoroni 等其他模型可用，但 Mixtral 32k 的更大上下文窗口被认为是一个突出的优势。

**Nous Research AI 频道摘要**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (2 messages): 
        
- **探索用于云端 LLM 服务的 DistAttention**：用户 `@maxwellandrews` 分享了一个关于 DistAttention 和 DistKV-LLM 的研究论文链接，这些新的分布式模型旨在缓解设计云端大语言模型（LLM）服务时的挑战。该[链接](https://huggingface.co/papers/2401.02669)指向一份摘要，讨论了这些模型如何动态管理 Key-Value Cache 并编排所有可访问的 GPU。
  
- **LLM 与长上下文场景**：用户 `@kenakafrosty` 分享了一篇名为《Self-Extend LLM Context Window Without Tuning》的研究论文[链接](https://arxiv.org/abs/2401.01325)。论文认为现有的 LLM 具有处理长上下文的内在能力，无需对训练序列进行微调训练。
  
- **Self-Extend 模型的实际应用**：`@kenakafrosty` 指出 “Self-Extend LLM Context Window Without Tuning” 的概念正在被实施并取得了不错的效果，并分享了相关的 [Twitter](https://twitter.com/sdand/status/1743695855545426362) 和 [GitHub](https://github.com/datamllab/LongLM) 链接。

**提及的链接**：

- [Paper page - Infinite-LLM: Efficient LLM Service for Long Context with DistAttention
  and Distributed KVCache](https://huggingface.co/papers/2401.02669)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325)：这项工作激发了 LLM 在无需微调的情况下处理长上下文的内在能力。训练期间训练序列的长度限制可能会限制大语言模型的应用...
- [GitHub - datamllab/LongLM: LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://github.com/datamllab/LongLM): LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning - GitHub - datamllab/LongLM: LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (5 messages): 
        
- **会做咖啡的 AI 是什么大事吗？**：`@adjectiveallison` 分享了来自 `@adcock_brett` 的一条推文，关于一个名为 “Figure-01” 的 **AI 机器人**，声称它通过观察人类学会了**制作咖啡**，并提到这是一个端到端的 AI，输入视频，输出轨迹。随后 `@teknium` 和 `@gabriel_syme` 表达了怀疑和幽默，质疑制作咖啡的能力是否算得上是引人注目的新闻。
  
- **比较 AI 的复杂性**：`@leontello` 将制作咖啡的 AI 与另一个项目 **ALOHA**（链接到 `@tonyzzhao` 的推文）进行了比较，称其相当平庸，但随后澄清了他们的评论，认识到涉及自动驾驶与机器人硬件设置的背景差异。
  
- **发现 AI 机器人领域的巧合**：幽默的是，`@adjectiveallison` 分享了另一条来自 `@atroyn` 的推文，注意到制作咖啡的 AI 所使用的咖啡机看起来非常眼熟，之前在 **Chelsea Finn** 研究项目的视频中见过。

**提及的链接**：

- [Tweet from anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn)](https://vxtwitter.com/atroyn/status/1744169452869140988?s=20)：这个演示视频中的某些东西看起来非常眼熟，然后我意识到我以前在 @chelseabfinn 团队论文的一个视频中见过同样的咖啡机...
- [Tweet from Brett Adcock (@adcock_brett)](https://vxtwitter.com/adcock_brett/status/1743987597301399852?s=20)：Figure-01 学会了做咖啡 ☕️ 我们的 AI 在观察人类做咖啡后学会了这一点。这是端到端 AI：我们的神经网络输入视频，输出轨迹。加入我们来训练我们的机器人...

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (5 messages): 
        
- **LLM 课程资源分享**：`@deki04` 分享了一个 **GitHub 仓库** [链接](https://github.com/mlabonne/llm-course)，该仓库提供了一门关于 Large Language Models 的全面课程，包含聊天机器人路线图和 Colab notebooks。
- **关于模型改进的不同意见**：`@leontello` 推测了引入 *augmented models* 的可行性，认为参数量的潜在增加可能会引发实用性方面的担忧。
- **深入探讨 LLM Agents**：`@vincentweisser` 分享了一篇[文章](https://borretti.me/article/thoughts-llm-agents)，详细分析了 LLM agents、ChatGPT 和 GitHub Copilot，强调了它们的重要影响，但也指出由于 context window 受限，在处理复杂任务时仍存在局限性。
- **LeCun 的 LLM 研究专题**：`@vincentweisser` 还重点介绍了一篇由 Yann LeCun 撰写的与 LLM 相关的[研究论文](https://openreview.net/pdf?id=BZ5a1r-kVsf)。

**相关链接**：

- [Thoughts on LLM Agents](https://borretti.me/article/thoughts-llm-agents)：细胞自动机的熵、临界性和复杂度类别。
- [GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.](https://github.com/mlabonne/llm-course)：学习 Large Language Models (LLMs) 的课程，包含路线图和 Colab notebooks。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (79 messages🔥🔥): 
        
- **探索 Hierarchical Embeddings 的性能**：`@gabriel_syme` 指出，除了已实现的 OAI 模型外，可能仍需要 hierarchical embeddings，暗示性能并未达到预期提升。
- **处理 LLM Fine-Tuning 的合成数据**：`@realsedlyf` 询问了目前在特定领域为语言模型 fine-tuning 创建 synthetic data 的最佳方法。 
- **Agentic RAG 实验**：`@n8programs` 宣布计划尝试 agentic RAG，这种方法中模型会根据输入问题生成各种搜索查询并收集信息，直到收集到足够的信息为止。他们提到 **Mistral** 特别适合此类任务。
- **行业应用 - LLM 与井筒分析**：`@kapnap_n` 分享了他们使用语言模型分析石油天然气行业井下井筒数据的方法。他们还讨论了数据的表示方式以及该方法的潜在益处，引发了 `@julianotto` 和 `@everyoneisgross` 等其他用户的兴趣。
- **AgentSearch 数据集发布公告**：`@teknium` 分享了 `@ocolegro` 关于发布 AgentSearch-V1 数据集的推文链接，该数据集包含超过 10 亿个 embedding 向量，涵盖了 Wikipedia、Arxiv、过滤后的 common crawl 等内容。


**相关链接**：

[Tweet from Owen Colegrove (@ocolegro)](https://fxtwitter.com/ocolegro/status/1744207765671657923?s=46)：AgentSearch-V1 的完整数据集现已在 HF 上线！！推荐：@qdrant_engine 用于索引和搜索，@nomic_ai 用于可视化。我正寻求扩大索引范围 - agent spe...

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (54 messages🔥): 
        
- **KV_Cache 实现请求**：`@lwasinam` 询问是否有任何 KV_Cache 的实现链接以供确认。
- **考虑使用 MoE 与 Mistral 进行对比**：`@bernaferrari` 建议基于 phi 构建一个混合专家模型（MoE），并将其与 **Mistral** 进行比较。
- **TinyLLAM vs Lite LLAMAS**：在 `@gabriel_syme` 和 `@teknium` 的讨论中，注意到 TinyLLAM 表现不佳，导致决定转向 Lite LLAMAS。
- **小型模型的处理能力**：`@kenakafrosty` 发起了一项讨论，认为小型模型（6-20B 范围）实际上可能比初看之下具有更强的处理能力，差距主要在于指令遵循（instruction following）。`@teknium` 分享了他的观点，认为 7B 模型正在达到其饱和点，但也补充说饱和点随模型大小呈非线性扩展。
- **Mergekit 和 MOE 演示**：`@teknium` 询问是否有现成的 Mergekit MOE（混合专家模型）实现的 notebook。作为回应，`.beowulfbr` 分享了 GitHub 上 Mergekit 的 [Mixtral 分支链接](https://github.com/cg123/mergekit) 以供参考。
- **PointCloud 模型限制**：在 `@gabriel_syme` 发起的关于 PointCloud 模型的对话中，`@teknium` 解释说，如果基础模型支持 8k，它应该能够处理 8k 的输入，但产生的输出不会超过 4k。
- **偏好具有更大上下文窗口的 Mixtral**：`@gabriel_syme` 和 `@teknium` 讨论了各种模型，包括 Mistral、Marcoroni 和 Mixtral。鉴于 **Mixtral** 具有 32k 的更大上下文窗口，他们表达了对该模型的偏好。


**提到的链接**：

[GitHub - cg123/mergekit: Tools for merging pretrained large language models.](https://github.com/cg123/mergekit)：用于合并预训练大语言模型的工具。


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (1 messages): 
        
qnguyen3: 目前使用 DINO, CLIP 和 CNNs


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **讨论 LAION 的衰减**：`@stellaathena` 和 `@flow7450` 探讨了像 LAION 这样数据集的衰减率。值得注意的是，根据 `@flow7450` 引用的一篇[论文](https://arxiv.org/abs/2111.12359)，**LAION400** 在大约 2-2.5 年后经历了约 20% 的下载失败。
- **关于重复数据的辩论**：`@uwu1468548483828484` 和 `@flow7450` 辩论了重复数据的重要性，权衡了备份的价值与对唯一样本的需求。
- **模型 ELO 评分**：`@letterrip` 提议了一个项目，为模型中的每个问题创建 ELO 评分，以形成训练运行的测试基准子集。讨论还涉及了在何处提议新项目。
- **Axolotl DPO 难题**：`@karkomagor` 询问 Axolotl 是否支持使用 DPO 数据集进行微调，`@main.ai` 建议去 Axolotl 服务器询问。
- **T5 解析**：`@stellaathena` 在回答 `@ricklius` 的提问时确认 T5 是一个 encoder-decoder 架构。
- **探讨学习率调度**：`@maxmatical` 和 `@ad8e` 研究了 DeepSeek AI 模型不寻常的阶梯式衰减学习率调度。这促使 `@ad8e` 提议在最后阶段以 0.1xLR 进行快速测试，这可能消除对恒定衰减的需求。
- **扭曲 Transformer 层**：`@kram1032` 建议在训练期间对 Transformer 架构中的层进行置换，假设这可能会鼓励更多地依赖 skip connections，并导致在添加或删除层时网络依然稳健。
- **寻找 MoE 的 Scaling Laws**：`@bshlgrs` 寻求关于混合专家模型（MoE）特有的 **LM Scaling Laws** 的前沿文献，`@philpax` 和 `@main.ai` 提供了相关贡献。
- **Harness 的障碍**：从评估 MMLU 等模型到实现自定义数据集，甚至考虑添加毒性/偏见评分器，`@gson_arlo`、`@hyperion.ai`、`@ishigami6465` 和 `@.johnnysands` 讨论了 `lm-eval-harness` 的各种功能。`@stellaathena` 和 `@hailey_schoelkopf` 强调了投机采样（speculative decoding）的重要性。

**Eleuther 频道总结**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (29 messages🔥): 
        
- **数据集衰减：思考 LAION 的寿命**：`@stellaathena` 发起了一场关于链接失效（linkrot）的讨论，特别是针对像 LAION 这样数据集的衰减率。`@flow7450` 研究了 **LAION400** 的衰减，发现大约 2-2.5 年后，约有 [20% 的链接下载失败](https://arxiv.org/abs/2111.12359)。
- **重复数据：关于备份策略与唯一性的辩论**：`@uwu1468548483828484` 和 `@flow7450` 就重复数据的重要性进行了辩论，争论备份的重要性是否超过了对唯一样本的需求。
- **关于模型 ELO 排名的新项目提案**：`@letterrip` 提议了一个新项目，建议为当前模型中的每个问题创建一个 ELO 评分，以形成一个用于训练运行期间测试的 Benchmark 子集。关于在哪里提议新项目进行了讨论，`@flow7450` 建议在 `<#1102787157866852402>`，而 `@ad8e` 澄清 community-projects 主要用于打算亲自推动的项目，而不仅仅是想法。`@letterrip` 确认有兴趣推动该项目。 
- **Axolotl 和 DPO 数据集**：`@karkomagor` 询问 Axolotl 是否支持使用 DPO 数据集进行微调，`@main.ai` 建议去 Axolotl 的服务器询问。
- **关于管理多个服务器活动的咨询**：`@seon5448` 发起了关于跟踪不同服务器活动的讨论，并寻求关于管理多个读书小组和项目活动的策略建议。回复中未提及具体的管理工具或技术建议。


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (33 messages🔥): 
        
- **T5 Encoder-Decoder 澄清**：`@ricklius` 询问 T5 是否是 Encoder-Decoder 架构，`@stellaathena` 确认它是，并提到有时人们会从 Encoder-Decoder 中分离出 Encoder 来使用。

- **DeepSeek AI 模型的学习率调度**：`@maxmatical` 讨论了 DeepSeek AI 模型，特别是它们使用的一种不寻常的 Stepwise Decay 学习率调度（Learning Rate Schedule），其最终 Loss 与传统的 Cosine Decay 相同。虽然这种方法允许在预训练期间更灵活地使用 Checkpoint，但 `@ad8e` 强调了大步长学习率的潜在危险，指出学习率使用不当可能导致次优结果甚至模型训练发散。

- **潜在的模型训练实验**：`@ad8e` 透露有意测试上述学习率步长背后的想法。目的是看是否只需要在最后阶段快速以 0.1xLR 运行即可，从而可能无需持续衰减。

- **关于 Weight Decay 和高斯权重噪声的讨论**：`@fessus` 提出了在归一化层中没有学习仿射变换（learned affines）的网络中，结合高斯权重噪声（Gaussian Weight Noise）和 Weight Decay 可能产生的影响。他们报告了在玩具数据集上削减不必要的网络复杂度方面的潜在益处。

- **具有置换层想法的 Transformer**：`@kram1032` 提出了一个独特的想法，即在层大小恒定的 Transformer 架构中，在训练期间对层进行置换（permuting layers）。他们的假设是，这种方法可能会鼓励网络更多地依赖 Skip Connections，并且在添加或删除层时可能会产生更稳健的网络。

**提到的链接**：

- [Chess-GPT’s Internal World Model](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html)：Chess-GPT 线性涌现世界表示。
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)：开源大语言模型 (LLM) 的快速发展确实令人瞩目。然而，先前文献中描述的 Scaling Law 呈现出不同的结论，这给...
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/30snj0x7/logs?workspace=user-ad8e;)：Weights & Biases，机器学习开发者工具。

### ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/) (5 messages): 
        
- **寻找 LM Scaling Laws 的阅读推荐**：`@bshlgrs` 寻求关于 **Language Model (LM) scaling laws** 当前最前沿文献的推荐，特别是针对 Mixture of Experts (MoE) 模型的。他们特别提到了[这篇论文](https://arxiv.org/abs/2202.01169)，该论文建议仅在参数少于 10 亿的 LM 中使用 MoE，而这一观点似乎受到了从业者的质疑。
- **关于 LM Scaling Laws 论文的建议**：`@philpax` 重点介绍了一篇关于 LM scaling laws 的[近期论文](https://arxiv.org/abs/2401.00448)。虽然这篇论文没有专门讨论 MoE 模型，但它可能提供相关的见解。
- **“更小且更长”是大规模推理需求的关键**：`@bshlgrs` 强调了建议论文中的一个关键发现，即对于具有大规模推理需求（约 10 亿次请求）的 LLM 厂商，最佳策略是**在更长的时间内训练更小的模型**。
- **缺乏针对 MoE 的高计算预算 Scaling 论文**：`@main.ai` 指出，目前缺乏讨论 MoE 模型在高计算预算下 scaling 的论文。


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (16 messages🔥): 
        
- **理解 lm-eval-harness 函数**：`@gson_arlo` 询问 `lm-eval-harness` 如何在 MMLU（4 选项 mcqa）等数据集上进行评估。`@baber_` 确认 `output_type: generate_until` 会触发模型的一次推理，而 `output_type: log_prob` 则会计算四次似然值，每个可能的补全计算一次。
- **lm-eval-harness 的灵活后处理**：`@hyperion.ai` 建议为 `lm-eval-harness` 增强一个宽松且灵活的后处理因素，以更贴近实际应用场景，即答案输出可以是灵活的但依然正确。`@stellaathena` 确认该 harness 可以处理此类情况。
- **在 lm-eval-harness 中实现自定义数据集**：`@ishigami6465` 询问 `lm-eval-harness` 所需数据集的具体格式。`@hailey_schoelkopf` 澄清用户可以在任务配置中定义，并解释了配置如何适用于不同类型的任务。
- **lm-eval-harness 中集成 Toxigen 评分器的潜力**：`@.johnnysands` 提出了在 lm-eval-harness 中添加毒性/偏见评分器的想法，考虑到 LlamaGuard 等工具已经提供了此类功能。`@hailey_schoelkopf` 肯定了此类评分器模型是可以集成的，特别是如果采用本地部署以避免干扰主评估模型。
- **关于 Speculative Decoding 的考虑**：`@stellaathena` 强调了 Speculative Decoding 的重要性，`@hailey_schoelkopf` 建议推理库应在 lm-eval 外部处理此问题。双方都认为 Hugging Face 的 TGI 和 tensorrt-llm 目前在这方面处理得很好。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **欧洲的老龄化转型**：在 #prompt-engineering 频道中，`mysterious_guava_93336` 和 `ChatGPT` 探讨了从上世纪的**“欧洲婴儿潮 (European baby boom)”**到如今**“老龄化欧洲 (elderly Europe)”**的人口转型。AI 阐明，由于多种因素，生育率收缩的迹象在 20 世纪 60 年代后期开始出现，导致欧洲人口在 20 世纪末和 21 世纪初趋于老龄化。这段对话随后在 #api-discussions 频道得到回应，进一步强化了这一转变的时间线以及促成人口变化的因素。
   
- **PowerPoint 演示文稿正在毁掉 AI 吗？**：在 #ai-discussions 中，一个持续的话题是关于 `ChatGPT` 最近提供冗长的、“PowerPoint 式”的回复，而不是简洁、自然的对话。尽管 `@eskcanta` 建议通过优化系统指令 (system instructions) 来生成理想的回复，但 `@mysterious_guava_93336` 表示没有显著改善。讨论还涉及了如何在 Discord 上使用 `ChatGPT`。

- **坚守你的域名**：处理域名验证 (domain verification) 和 GPT 编辑器 (GPT editor) 问题是 #gpt-4-discussions 的热门话题。虽然 `@darthgustav` 试图引导 `@anardude` 完成域名验证，但 `@anardude` 在解决过程中遇到了困难，并寻求 OpenAI Support 的帮助。`@.marcolp` 对一个持续存在的错误表示沮丧，该错误在解决前一直禁止访问 GPT 编辑器。

- **当训练时长付诸东流时**：#gpt-4-discussions 频道中回荡着绝望的情绪，`@moonlit_shadows` 对一次长达 20 小时但最终毫无产出的 GPT-2 训练任务表示失望。

- **特定语言的 GPT 问题与规则集**：`@codocoderson` 在 #gpt-4-discussions 中询问如何创建一个英文 GPT，并使其描述和引导性问题在全球范围内以用户的母语显示。`@jesuisjes` 询问他们的 GPT 是否偶尔会漏掉预设的策略制定对话流程。

- **为什么 GPT 无法读取我的文件？**：#gpt-4-discussions 频道由 `@cerebrocortex` 发起了一场讨论，想知道为什么 GPT 有时无法读取 .docx 和 .txt 文件。该问题的具体原因尚不确定，可能的原因包括文档大小、Token 限制或 /mnt 超时期间的文件损坏。


**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (30 条消息🔥): 
        
- **对 ChatGPT 最近更新的担忧**：
    - `@mysterious_guava_93336` 对 ChatGPT 最近的更新表示不满，称 AI 现在提供的是冗长的、结构化的“PowerPoint”风格回复，而不是 2023 年夏天之前那种简洁、自然的对话风格。他们分享了[一个他们偏好的回答示例](https://chat.openai.com/share/f7390818-627b-43e6-9f68-054afca55fbf)，并征求关于如何指示 AI 生成此类回复的建议。
- **针对理想回复的建议指令**：
    - `@eskcanta` 建议优化系统的自定义指令 (custom instructions)，使其更加具体和积极，使用类似于训练狗的引导技巧。他们建议使用一种观点模式，并鼓励 AI 以创造性的方式挑战用户。这种方法的[示例可以在 OpenAI 的对话中看到](https://chat.openai.com/share/f7390818-627b-43e6-9f68-054afca55fbf)。
- **对建议更改的失望**：
    - 尽管更改了指令，`@mysterious_guava_93336` 仍未发现重大改进，称 AI 仍然生成“PowerPoint 式”的输出。
- **在 Discord 上使用 ChatGPT**：
    - `@lemon77_ps` 询问如何在 Discord 上使用 ChatGPT，`@7877` 解释说 ChatGPT 必须通过 OpenAI 的官方网站使用。
- **Discord 服务器上的互动**：
    - `@michael_6138_97508` 指出该 Discord 服务器旨在与真人或同等对象进行互动，并提到存在一个拥有 OpenAI API 和文档特定知识的机器人。

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (26 条消息🔥): 
        
- **域名验证困扰**：用户 `@anardude` 询问如何验证他的域名。`@darthgustav` 提供了在 GPT 编辑器中通过 DNS 记录验证域名的说明，但 `@anardude` 反馈该方案无效，并询问如何联系 OpenAI Support，对此 `@darthgustav` 建议在 Google 搜索 "OpenAI Help and Support"。
- **长时间的 GPT-2 训练付诸东流？**：用户 `@moonlit_shadows` 分享了对一次长达 20 小时训练任务的失望，该任务似乎因保存时的属性错误（attribute error）而导致失败。
- **关于创建特定语言 GPT 的咨询**：用户 `@codocoderson` 询问关于发布英文版 GPT 的事宜，并咨询其描述和引导问题（starter questions）是否会根据全球用户的语言进行显示。
- **GPT 遵守规则集的问题**：`@jesuisjes` 试图确认关于其 GPT 偶尔遗漏流程的预期，尽管已经设置了规则来遵循预定的策略对话流程。
- **关于 GPT 读取文档文件问题的担忧**：`@cerebrocortex` 询问为什么 GPT 有时在读取 .Docx 和 .txt 文件时会出现问题。用户 `@michael_6138_97508` 推测文档大小和 token 限制可能是潜在原因，而 `@darthgustav` 建议可能是更新期间的 /mnt 超时导致了文件损坏。
- **GPT 编辑器的困扰**：`@.marcolp` 对持续出现的 "error searching knowledge"（搜索知识库错误）问题表示沮丧，这导致甚至无法进入 GPT 编辑器，使得在修复之前进一步开发 GPT 变得毫无意义。`@darthgustav` 提供了一个潜在的变通方案，包括移除并重新挂载知识库。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 条消息): 
        
- **讨论欧洲人口演变**：用户 `mysterious_guava_93336` 与 `ChatGPT` 展开了关于从上世纪的 **"欧洲婴儿潮"** 到如今的 **"老龄化欧洲"** 转变的讨论。 
- **生育率收缩时间线**：`ChatGPT` 澄清说，欧洲生育率收缩的最初迹象出现在 20 世纪 60 年代后期，并在 70 年代变得显著。 
- **生育率下降的促成因素**：导致这一转变的原因包括经济变化、女性权利和劳动力参与、避孕和计划生育普及率的提高以及文化转变。
- **导致的人口老龄化**：到 20 世纪末和 21 世纪初，许多欧洲国家的出生率已低于 2.1 的更替水平，导致人口老龄化。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 条消息): 
        
- **从 '婴儿潮' 到 '老龄化欧洲' 的转变**：`mysterious_guava_93336` 与 ChatGPT 讨论了二战后 "婴儿潮" 之后欧洲何时开始出现生育率收缩的最初迹象。ChatGPT 确认这种人口转型始于 20 世纪 60 年代后期，并在 70 年代变得更加明显，其促成因素包括经济变化、女性劳动力参与、避孕措施的普及以及文化转变。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **移动端文件上传功能**：根据 `@righthandofdoom` 和 `@giddz` 的说法，目前仅支持从 iOS 上的 **Perplexity 移动端应用**上传图片。据报道，对 Android 的支持**即将推出**。
- **查找写作模式 (Writing Mode) 功能**：`@ellestar_52679` 询问了 **Writing Mode** 功能的位置，`@icelavaman` 指出了导航路径：*点击 "Focus"，然后选择 "writing"*。
- **关于功能和促销的详细讨论**：`@debian3` 询问了文件上传功能以及**年度计划**的可用促销活动。`@icelavaman` 澄清说，可以通过推荐链接而非促销活动来节省费用，并引导至 [FAQ 链接](https://blog.perplexity.ai/faq/how-does-file-upload-work) 以获取有关文件上传的更多细节。
- **Perplexity 的账单问题**：`@ahmed7089` 对在注销账户后仍被 **Perplexity** 收费感到不满。`@mares1317` 处理了这一情况，并提供了一个[相关链接](https://discord.com/channels/1047197230748151888/1118264005207793674/1188109891508908073)。
- **Perplexity 访问限制**：`@byerk_enjoyer_sociology_enjoyer` 对 **Perplexity** 无法访问 Pinterest、Instagram 或 Tumblr 上的帖子表示担忧。
- **Perplexity 与 pplx 模型对比的深入分析**：`@dw_0901` 发起了关于 **pplx 在线模型 (7B/70B) 与 Perplexity 之间差异**的讨论，质疑产品设计上的不同。
- **对比 Perplexity 的 Copilot 和普通版本**：`@promoweb.2024` 询问了 **Perplexity 的 Copilot 与普通版本**的区别。`@icelavaman` 在此[链接](https://blog.perplexity.ai/faq/what-is-copilot)分享了关于 Perplexity Copilot 的详细信息。
- **在 pplx-api 上使用 5 美元 Pro 积分的故障排除**：`@blackwhitegrey` 在 `@mares1317` 的指导下完成了 `pplx-api` Pro 积分的申请流程，后者还提供了一份[分步指南](https://docs.perplexity.ai/docs/getting-started)。
- **关于 Perplexity API 用户友好性的说明**：`@blackwhitegrey` 和 `@brknclock1215` 认为 Perplexity API 缺乏用户友好性，主要是因为缺乏编程技能。`icelavaman` 澄清说，**API 主要是面向开发者的**。
- **澄清 Pro 积分并非额外付款**：`@blackwhitegrey` 最初误将 Pro 积分理解为访问 API 的额外付款。`icelavaman` 澄清说，这些实际上是提供给开发者的奖励。
- **对非技术用户的乐观态度**：尽管存在困难，`@brknclock1215` 最后表达了乐观的看法，认为那些不一定是程序员但了解技术的人，可以**从其进步中获益最多**。
- **请求协助将帖子设为公开**：`@me.lk` 建议 `<@1018532617479532608>` 将其帖子 (thread) 设为公开，以便他人查看内容。`<@1018532617479532608>` 采纳了建议并公开了该帖子。
- **分享 Perplexity.AI 搜索结果**：`@soanseng` 和 `@debian3` 分别分享了关于**如何使用**和**如何绘画**的搜索结果，与社区分享了他们的知识。


**Perplexity AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (23 条消息🔥): 
        
- **移动端 App 的文件上传功能**：`@righthandofdoom` 询问了从 **移动端 App** 上传文件的功能。`@giddz` 澄清说目前仅支持图片上传，且仅在 iOS 上可用，Android 支持 **即将推出**。
- **查找 "writing mode" 功能**：`@ellestar_52679` 试图找到 **writing mode** 功能。`@icelavaman` 建议他们点击 "Focus"，然后选择 "writing"。
- **关于 Perplexity 功能和促销的问题**：`@debian3` 询问了文件上传功能的用途、可以上传的文件类型，以及 **年度计划** 是否有任何促销活动。`@icelavaman` 确认虽然目前没有促销活动，但可以通过推荐链接节省费用。他们还提供了一个关于文件上传查询的 [链接](https://blog.perplexity.ai/faq/how-does-file-upload-work)。
- **账号注销与计费问题**：`@ahmed7089` 投诉在注销账号后仍被 **Perplexity** 扣费。`@mares1317` 回复了一个 [链接](https://discord.com/channels/1047197230748151888/1118264005207793674/1188109891508908073)，推测其中包含更多信息。
- **Perplexity 与社交媒体平台**：`@byerk_enjoyer_sociology_enjoyer` 对 **Perplexity** 无法访问 Pinterest, Instagram 或 Tumblr 的帖子表示担忧。
- **Perplexity 与 Pplx 模型对比**：顾问 `@dw_0901` 询问了 **pplx 在线模型 (7B/70B) 与 Perplexity 之间的区别**，质疑底层产品设计是否存在差异。
- **Copilot 与普通版本的 Perplexity**：`@promoweb.2024` 询问了使用 **Perplexity Copilot 与普通版本** 之间的区别。`@icelavaman` 分享了一个 [链接](https://blog.perplexity.ai/faq/what-is-copilot)，提供了 Perplexity Copilot 的详细概述。


**提到的链接**：

- [什么是 Perplexity Copilot？](https://blog.perplexity.ai/faq/what-is-copilot)：浏览 Perplexity 博客以获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [文件上传如何运作？](https://blog.perplexity.ai/faq/how-does-file-upload-work)：浏览 Perplexity 博客以获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [什么是 Search Focus？](https://blog.perplexity.ai/faq/what-is-search-focus)：浏览 Perplexity 博客以获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (4 条消息): 
        
- **分享设为公开**：用户 `@me.lk` 建议 `<@1018532617479532608>` **将其对话 (thread) 设为公开** 以便他人查看，`<@1018532617479532608>` 回应称现在已将其对话公开。
- **Perplexity.AI 搜索结果**：用户 `@soanseng` 和 `@debian3` 分享了 perplexity.ai 的搜索链接：
    - `@soanseng`：分享了一个关于 [如何使用](https://www.perplexity.ai/search/how-to-use-jDLvEroNQke87WGjx4W.fA?s=c) 的链接。
    - `@debian3`：分享了一个关于 [如何绘画](https://www.perplexity.ai/search/How-to-draw-PZIg5SS2QxuuK8xpHyHhGA?s=c#3d9220e5-24b6-431b-ae2b-cc691f21e118) 的链接。

### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (12 messages🔥): 
        
- **如何使用 5 美元 Pro 积分**：用户 `@blackwhitegrey` 寻求关于在 `pplx-api` 上使用 Pro 积分的建议。`@mares1317` 提供了 [Perplexity API 入门指南](https://docs.perplexity.ai/docs/getting-started) 的链接，并强调了包括**提供支付信息、购买积分和生成 API key**在内的步骤。
- **API 知识与需求不匹配**：`@blackwhitegrey` 因缺乏编程技能并认为 **API 不够用户友好**而感到沮丧。`icelavaman` 澄清说，**API 主要面向开发者**，而非直接在网站上使用。
- **Pro 积分被误解为额外付费**：`@blackwhitegrey` 最初认为 Pro 用户必须为 API 访问额外支付 5 美元。然而，`icelavaman` 澄清说，这些**不是额外付款，而是给开发者的奖励**。
- **非开发者使用 API 的实用性**：`@brknclock1215` 呼应了 `@blackwhitegrey` 的观点，表示由于缺乏编程技能，在实现 API 方面存在类似困难。他们还认为，在没有相应技术知识的情况下尝试使用高级工具，可能耗时多于收益。
- **非技术用户仍可获益**：`@brknclock1215` 以乐观的态度结束，暗示那些不一定是程序员但了解如何与技术互动的人，可能会从技术的进步中获益最多。

**提及的链接**：

- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started)
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **引导 LLM 响应**：在与 `@bdambrosio` 的对话中，`@i_am_dom` 澄清说，引导 LLM 以获得适当的响应确实是可行的，但消息集必须以用户消息结束。这打破了使用官方 API 部分预写响应的希望，因为这会返回错误。
- **寻找可靠的聊天对话程序**：`@jb_5579` 向社区提问，寻求提供强大聊天对话程序的仓库推荐，最好是针对 **Mistral API** 进行了优化，并具有记忆会话以及代码辅助和代码补全功能。
- **关于未知上下文窗口大小的情况**：`@tonyaichamp` 询问不同版本 API 模型的上下文窗口大小，`@frosty04212` 对此表示不确定，并强调需要进行实验性探索以更好地理解和利用系统。
- **Mistral-tiny 证明了其实力**：`@tonyaichamp` 分享了使用 `mistral-tiny` 模型的显著成功，利用它从一个 16k token 的 HTML 页面中提取内容。鉴于其成本效益和速度，该用户打算将来在类似任务中应用它。
- **AI Agent 集结**：`@10anant10` 宣布了他们以**构建 AI Agent** 为中心的项目，`@.tanuj.` 对此表示兴趣并启动了直接沟通。
- **框架推荐**：用户 `@joselolol` 建议探索 **MLX framework**。
- **防护栏指南**：`@akshay_1` 链接了一个关于 **guardrailing** 的有用资源，URL 为：`https://docs.mistral.ai/platform/guardrailing/`
- **微调的硬件限制**：`@david78901` 提出了在单个 3090 上微调 **Mistral 7b** 的可行性。微调 LoRA 或 QLoRA 是可以处理的，但全量微调（full fine-tuning）可能需要多个 3090 或单个带有 `Axolotl` 的 A100。
- **LLMcord - 多功能 Discord 机器人**：`@jakobdylanc` 展示了 LLMcord，这是一个开源的 Discord 机器人，兼容 **Mistral API** 以及通过 *LM Studio* 在个人硬件上运行的 **Mistral models**。该项目在 GitHub 上也有提及。
- **Mistral 相对于 OpenAI 的优势**：`@joselolol` 承认 Mistral 相对于 OpenAI 的优势，认为它在任务处理上更快、更便宜且更有效。
- **二分心智理论引发困惑**：提到二分心智（Bicameral Mind）理论引发了辩论，`@cognitivetech` 推荐了 Julian Jaynes 的基石著作，而持怀疑态度的 `@king_sleeze` 将该理论等同于伪科学。
- **Mistral API 功能建议**：`@jakobdylanc` 支持对 **Mistral API** 进行增强，以处理内容列表为空的边缘情况，就像 **OpenAI API** 目前所做的那样。
- **功能扩展指日可待**：正如 `@tom_lrd` 所指出的，**Mistral** 中的 Function calling 将成为优先级。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (9 条消息🔥): 
        
- **引导响应 (Prime the Response):** `@bdambrosio` 讨论了使用最后一条 assistant 消息来“引导” LLM 做出适当响应的技术，并寻求关于当服务器拒绝以 assistant 消息结尾的消息集时，该技术有效性的建议。`@i_am_dom` 澄清说可以进行响应引导，但必须以 user 消息结尾。他们还提到，使用官方 API 无法部分预写响应本身，否则会返回错误。
- **最喜欢的聊天对话仓库:** `@jb_5579` 向社区询问他们最喜欢的用于构建稳定聊天对话程序的仓库——特别是针对 Mistral API 进行了优化、具有会话记忆功能，并专注于 Code Assist 和 Code Completion 的仓库。
- **上下文窗口大小 (Context Window Sizes):** `@tonyaichamp` 询问了不同版本 API 模型的上下文窗口大小，但 `@frosty04212` 回复称目前尚不清楚具体大小。他们敦促通过实验来更好地理解和利用该系统。
- **Mistral-tiny 提取内容的质量:** `@tonyaichamp` 分享了使用 `mistral-tiny` 模型从 16k token 的 HTML 页面中提取内容的积极体验。鉴于该模型的成本效益和速度，`@tonyaichamp` 打算在未来将其用于类似任务。


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (2 条消息): 
        
- **构建 AI Agents**: 用户 `@10anant10` 宣布他们正在致力于**构建 AI Agents**。
- **发起直接沟通**: 用户 `@.tanuj.` 回应了 `@10anant10` 的评论，表示打算**给他们发送私信 (DM)**。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (1 条消息): 
        
joselolol.: 你好先生，考虑使用 MLX 框架吧！


### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (1 条消息): 
        
akshay_1: https://docs.mistral.ai/platform/guardrailing/


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 条消息): 
        
- **在单张 3090 上进行微调的可行性**: 用户 `@david78901` 提到，单张 3090 可能可以处理 **Mistral 7b** 的 LoRA 或 QLoRA 微调，但全量微调 (Full Fine-tuning) 只有在使用 3x3090s 或单张 A100 配合 Axolotl 时才可行。


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (4 条消息): 
        
- **介绍 LLMcord，一个多功能的 Discord 机器人**: `@jakobdylanc` 展示了他的开源 Discord 机器人 LLMcord，它同时支持 Mistral API 以及通过 [LM Studio](https://lmstudio.ai/) 在个人硬件上运行 Mistral 模型。功能包括复杂的聊天系统、与 OpenAI API 的兼容性、流式响应，以及包含在单个 Python 文件中的简洁代码。可以在 [GitHub](https://github.com/jakobdylanc/llmcord) 上查看该项目。
- **Mistral 驱动 AI 后端**: `@joselolol` 提到他们正在使用 Mistral 来支持某些 AI 任务的后端。
- **合成数据生成与模型评估**: `@joselolol` 还分享了他的系统可以生成合成数据并为微调模型提供评估，这对于开发者来说是一个潜在有用的工具。
- **Mistral vs OpenAI**: 根据 `@joselolol` 的经验，Mistral 在大多数任务中超越了 OpenAI，证明其更快、更便宜且更有效。

**提到的链接**:

- [👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai/): 查找、下载并实验本地 LLMs
- [GitHub - jakobdylanc/llmcord: 一个 Discord AI 聊天机器人 | 选择你的 LLM | 具备视觉能力的 GPT-4 Turbo | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | 流式响应 | 以及更多 🔥](https://github.com/jakobdylanc/llmcord): 一个 Discord AI 聊天机器人 | 选择你的 LLM | 具备视觉能力的 GPT-4 Turbo | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | 流式响应 | 以及更多 🔥 - GitHub - jakobdylanc/llmcord: A Discord A.....

### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 条消息): 
        
- **二分心智理论引发关注**：用户 `@blueridanus` 对某些内容感到困惑，随后 `@cognitivetech` 提供了 [Wikipedia 页面](https://en.wikipedia.org/wiki/The_Origin_of_Consciousness_in_the_Breakdown_of_the_Bicameral_Mind) 链接进行了解答，该页面讨论了 **《二分心智的崩溃与意识的起源》**（The Origin of Consciousness in the Breakdown of the Bicameral Mind）。这是作者 Julian Jaynes 在 1976 年出版的著作，提出了一套关于人类意识起源的理论。
- **书籍推荐与争议**：同一用户 `@cognitivetech` 强烈推荐了这本书，并强调了其引人深思的特性。作为回应，`@king_sleeze` 表达了怀疑态度，认为 Jaynes 的理论基于间接证据，将其等同于“伪科学”（*pseudoscience*）。
- **对理解意识的怀疑**：在另一条消息中，`@king_sleeze` 指出了理解人类意识的复杂性，并将其与神经网络的“黑盒”性质进行了类比。他们表示：“*据我所知，没有人能告诉我他们的想法是在哪里以及如何形成的*”，强调了人类思维形成的神秘性。


**提到的链接**：

[The Origin of Consciousness in the Breakdown of the Bicameral Mind - Wikipedia](https://en.wikipedia.org/wiki/The_Origin_of_Consciousness_in_the_Breakdown_of_the_Bicameral_Mind)


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 条消息): 
        
- **关于期望功能的讨论**：`@gbourdin` 提到社区正热切期待某种功能的实现可能性。 
- **Mistral API 改进建议**：`@jakobdylanc` 建议 **Mistral API** 应该能够像目前的 **OpenAI API** 一样，处理 `message.content` 为空列表的边缘情况。
- **Function calling 即将到来**：`@tom_lrd` 提到 **Function calling** 已被宣布为未来开发的**优先级事项**。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

仅 1 个频道有活动，因此无需汇总...

- **来自 Wikipedia 的德语 DPR 数据集**：`@philipmay` 分享了他基于德语 Wikipedia 创建德语 *Dense Passage Retrieval (DPR)* 数据集的项目。他已将该项目发布在 [GitHub](https://github.com/telekom/wikipedia-22-12-de-dpr) 上供公共使用。 

- **关于上下文长度的辩论**：一场关于 Embedding 适用的文档上下文长度的讨论展开了。`@sebastian.bodza` 质疑 `@philipmay` 在项目中使用的 270 个最大 Token 数量是否太短，并将其与 [Jina Embeddings](https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/) 以及其他在多达 512 个 Token 上训练的模型进行了比较。`@philipmay` 和 `@bjoernp` 则认为，过长的上下文可能会分散注意力，或者让 BERT 模型更难编码。

- **BAAI 训练数据建议**：`@sebastian.bodza` 分享了托管在 [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5) 上的 BGE 训练数据链接，建议这可能会提供额外的见解。 

- **E5 在 512 Token 上的训练**：`@sebastian.bodza` 注意到 E5 模型也是在 512 个 Token 上训练的，进一步支持了关于最佳上下文长度的辩论。关于 E5 训练的详细信息可以在 [这里](https://arxiv.org/abs/2212.03533) 找到。

**提到的链接**：

- [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)：文本 Embedding 模型已成为将句子转换为封装语义信息的固定大小特征向量的强大工具。虽然这些模型对于诸如...的任务至关重要。
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)：本文介绍了 E5，这是一个最先进的文本 Embedding 系列，可以很好地迁移到广泛的任务中。该模型以对比学习的方式进行训练，使用了来自我们 cu... 的弱监督信号。
- [GitHub - telekom/wikipedia-22-12-de-dpr: German dataset for DPR model training](https://github.com/telekom/wikipedia-22-12-de-dpr)：用于 DPR 模型训练的德语数据集。通过在 GitHub 上创建账号为 telekom/wikipedia-22-12-de-dpr 的开发做出贡献。
- [BAAI/bge-large-en-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Benchmarking Evaluation of LLM Retrieval Augmented Generation](https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/)：了解哪些检索方法有效以及分块策略（chunking strategy）。包括测试脚本和示例，以便在您自己的文档上参数化检索，通过 LLM 评估确定性能并提供...

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

只有一个频道有活动，因此无需总结...

- **探索 AI 作为编辑的角色**：用户 `@slono` 询问是否存在 **可以充当编辑的 AI 写作工具**，帮助撰写消息、重组结构、建议或删除句子。`@coffeebean6887` 建议向自定义 GPT 描述你的需求几分钟即可实现。然而，slono 指出由于用户界面繁琐，这并不是一个理想的解决方案。
- **AI 辅助编辑的困境**：针对上述讨论，`@swizec` 分享了他们在 **swiz-cms** 中实现此类功能的经验。他们评论道，最大的挑战在于如何有效地传达 AI 应该寻找什么，这暗示了内容编辑 AI 工具需要改进引导或界面。
- **关于大语言模型 (LLMs) 的见解**：用户 `@swyxio` 分享了一个 [LessWrong 帖子](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample) 的链接，该帖子深入探讨了在构建 AGI 中使用 LLM 的影响和潜在安全益处，以及一篇相关的 [Arxiv 论文](https://arxiv.org/abs/2401.02415)，该论文提出了一种 LLM 的 post-pretraining 方法，旨在不产生灾难性遗忘的情况下提升其知识。
- **对 AI 资源的感谢**：用户 `@thenoahhein` 感谢 `@swyxio` 提供的资源，称这为他提供了一周的阅读材料。这些资源链接到了用户 Eugene Yan 的一条 [Twitter 帖子](https://twitter.com/eugeneyan/status/1744179600056545300)。

**提到的链接**：

- [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415?utm_source=ainews&utm_medium=email)：人类通常在不损害旧技能的情况下习得新技能；然而，对于大语言模型 (LLMs) 来说情况正好相反，例如从 LLaMA 到 CodeLLaMA。为此，我们提出了一种新的 post-pretra...
- [Mat’s Blog - Transformers From Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html)
- [An explanation for every token: using an LLM to sample another LLM — LessWrong](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample)：简介：关于基于一个或多个大语言模型构建 AGI 的影响和潜在安全益处，已经有很多论述……

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **LAION 链接失效 (Linkrot) 的神秘案例**：@stellaathena 对任何研究 *像 LAION 这样的数据集因 Linkrot 导致的衰减率* 的研究表示感兴趣。
- **关于 Stable Diffusion 1.6 的疑惑**：@pseudoterminalx 询问关于 **Stable Diffusion 1.6** 的信息，@nodja 推测它可能结合了 1.x 架构以及来自 sdxl 等的改进。
- **深入探讨 Aspect Ratio Bucketed SD 1.5**：@thejonasbrothers 分享说 **Aspect Ratio Bucketed SD 1.5** 支持高达 *1024x1024 像素*。
- **CogVLM 备受关注**：@SegmentationFault 为 **CogVLM** 喝彩，称其 *非常令人印象深刻，但在 AI 社区中被低估了*。
- **过拟合的梦境解决方案**：@progamergov 发布了一篇 [研究论文](https://www.cell.com/patterns/fulltext/S2666-3899(21)00064-7)，认为 *梦境可能通过向过拟合的概念引入随机噪声，从而在人脑中起到抗过拟合 (anti-overfitting) 机制的作用*。
- **睡眠不足的记忆需要更多研究**：针对梦境理论，@progamergov 希望能有研究调查 *睡眠不足对语义和情境记忆形成的影响* 作为支持证据。

**LAION 频道总结**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (8 条消息🔥): 

- **对 LAION 数据集中 Linkrot 的好奇**：`@stellaathena` 询问是否有人进行过或看过关于 LAION 等数据集因 Linkrot 导致衰减率的相关研究。
- **关于 Stable Diffusion 1.6 的问题**：`@pseudoterminalx` 寻求关于 Stable Diffusion 1.6 的信息。`@nodja` 假设它可能是 1.x 架构加上来自 sdxl 的一些改进和额外功能。
- **关于 Aspect Ratio Bucketed SD 1.5 的见解**：`@thejonasbrothers` 提供了关于 Aspect Ratio Bucketed SD 1.5 的见解，指出它支持高达 1024x1024 像素。
- **对 CogVLM 的赞赏**：`@SegmentationFault` 表达了对 CogVLM 的赞赏，认为它非常出色且在 AI 社区中被低估。

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (2 条消息): 
        
- **梦作为抗过拟合（Anti-overfitting）机制**：`@progamergov` 分享了一篇[研究论文](https://www.cell.com/patterns/fulltext/S2666-3899(21)00064-7)，该论文指出**做梦可能在防止人脑 overfitting 中起着至关重要的作用**。根据论文，梦境会向过拟合的概念引入随机噪声，从而有助于避免 overfitting。
- **呼吁测试睡眠剥夺对记忆的影响**：`@progamergov` 表示希望该研究能扩展到测试**睡眠剥夺对 semantic 和 episodic 记忆形成的影响**，并断言此类测试可以为上述假设提供支持性证据。