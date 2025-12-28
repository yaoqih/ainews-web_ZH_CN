---
companies:
- openai
- deepseek
- qdrant
- twilio
- llamaindex
- elevenlabs
date: '2024-12-28T05:06:02.495266Z'
description: '**ChatGPT**、**Sora** 和 **OpenAI API** 经历了超过 5 小时的停机故障，目前已恢复正常。**vLLM**
  的更新支持 **DeepSeek-V3** 以增强的**并行性**和 **CPU 卸载（CPU offloading）**模式运行，提升了**模型部署的灵活性**。关于
  **top-k 路由 MoE**（混合专家模型）中的**梯度下降**以及 **FP8 精度**的采用，相关讨论聚焦于**训练效率**和**内存优化**。由 **Team
  Therasync** 开发的 AI 语音医疗助手 **AIDE** 利用了 **Qdrant**、**OpenAI** 和 **Twilio**。**DeepSeek-Engineer**
  提供支持结构化输出的 AI 编程辅助。**LlamaIndex** 集成了 **LlamaCloud** 和 **ElevenLabs**，用于大规模**文档处理**和语音交互。关于使用
  **ghstack** 进行**版本控制**的见解以及对**线性衰减学习率调度**的倡导，突显了 AI 开发中的最佳实践。专家预测 2025 年将出现**更小、更精炼的模型**、**真正的多模态模型**以及**端侧
  AI（on-device AI）**。关于**行星级联邦学习**和社区 **AGI 登月计划**的提议强调了 AI 的未来发展方向。关于**智能体系统（agentic
  systems）**、**多智能体工作流**以及通过**思维链（CoT）推理**进行**审慎对齐（deliberative alignment）**的讨论，进一步强化了
  AI 安全与对齐方面的努力。'
id: 5cd40505-c2bf-4d82-bb10-46d8f0cb669d
models:
- vllm
- deepseek-v3
- llamaindex
original_slug: ainews-not-much-happened-today-4715
people:
- francois-fleuret
- daniel-hanchen
- aaron-defazio
- fchollet
- elad-gil
- wojciech-zaremba
- richard-socher
title: 今天没发生什么事。
topics:
- training-efficiency
- parallelism
- cpu-offloading
- gradient-descent
- mixture-of-experts
- fp8-precision
- memory-optimization
- ai-voice-assistants
- coding-assistants
- document-processing
- version-control
- learning-rate-schedules
- federated-learning
- agentic-systems
- multi-agent-systems
- deliberative-alignment
- chain-of-thought
- on-device-ai
- multimodality
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周末正是我们所需要的。**

> 2024/12/26-2024/12/27 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**215** 个频道，**5579** 条消息）。预计节省阅读时间（以 200wpm 计算）：**601 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

ChatGPT, Sora 和 OpenAI API 经历了超过 5 小时的停机。它们已经[恢复运行](https://x.com/OpenAI/status/1872444309506765141)。


![image.png](https://assets.buttondown.email/images/0fd6f94c-2c41-4dd6-b651-ac72de8ef9e5.png?w=960&fit=max)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有摘要由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 基础设施与优化**

- **训练效率与扩展**：[@vllm_project](https://twitter.com/vllm_project/status/1872453508127130017) 宣布了 **vLLM** 的更新，允许 **DeepSeek-V3** 以多种 **parallelism**（并行）和 **CPU offloading**（CPU 卸载）选项运行，增强了 **model deployment flexibility**（模型部署灵活性）。

- **梯度下降与 MoE 路由**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1872370360307568964) 询问了 **top-k routing MoE** 中的 **gradient descent mechanics**（梯度下降机制），探讨了 **feature ranking**（特征排序）如何影响 **model training dynamics**（模型训练动态）。

- **FP8 精度与内存优化**：[@danielhanchen](https://twitter.com/danielhanchen/status/1872719599029850391) 等人讨论了在 **DeepSeek V3** 中采用 **FP8 precision**（FP8 精度），重点在于 **memory usage reduction**（减少内存使用）和 **training cost minimization**（训练成本最小化）。

**AI 应用与工具**

- **AI 在医疗保健领域的应用**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1872613403384008999) 展示了 **AIDE**，这是一款由 **Team Therasync** 在 **Lokahi Innovation in Healthcare Hackathon** 上开发的 **AI 语音医疗助手**，利用了 **Qdrant**、**@OpenAI** 和 **@twilio** 等工具。

- **AI 驱动的编程助手**：[@skirano](https://twitter.com/skirano/status/1872382787422163214) 在 **GitHub** 上介绍了 **DeepSeek-Engineer**，这是一个能够使用 **structured outputs**（结构化输出）进行**读取、创建和对比文件（diffing files）**的 **coding assistant**。

- **AI 文档处理**：[@llama_index](https://twitter.com/llama_index/status/1872684854703432137) 演示了一个可以对 **100 万份以上 PDF** 进行 **RAG** 的 **AI 助手**，集成了 **LlamaCloud** 和 **@elevenlabsio** 用于 **document processing**（文档处理）和 **voice interaction**（语音交互）。

**AI 开发实践**

- **版本控制与协作**：[@vikhyatk](https://twitter.com/vikhyatk/status/1872394404398588225) 分享了使用 **ghstack** 管理 **pull requests** 的见解，增强了 **GitHub** 中的 **collaboration**（协作）和 **code management**（代码管理）。

- **训练计划与学习率**：[@aaron_defazio](https://twitter.com/aaron_defazio/status/1872481458184745374) 提倡使用 **linear decay learning rate schedule**（线性衰减学习率调度），强调其在 **model training** 中比其他调度方式更有效。

- **开源贡献**：[@ArmenAgha](https://twitter.com/ArmenAgha/status/1872426813865201700) 和 [@ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/1872582508128383215) 感谢同行**引用研究论文**，促进了**开源协作**，并为 **PhysicsLM** 等项目**争取资源**。

**AI 创新与未来趋势**

- **2025 年 AI 预测**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1872439611525599291) 转达了来自 **@fchollet** 和 **@EladGil** 等专家的**预测**。关键预测包括**更小、更紧凑的模型**、**真正的多模态模型**以及 **on-device AI**（端侧 AI）解决方案。

- **联邦学习与社区 AGI**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1872675436347715921) 提出有必要进行**行星级规模的联邦学习（planetary-scale federated learning）**和**社区 AGI 的登月项目**，类似于 **ITER** 等跨国倡议。

- **AI 生态系统演进**：[@RichardSocher](https://twitter.com/RichardSocher/status/1872442369787973648) 等人讨论了 **agentic systems**（智能体系统）的兴起、**multi-agent workflows**（多智能体工作流）以及 **AI 在各行业的整合**，标志着 **AI 应用新时代**的到来。

**AI 安全与对齐**

- **审议式对齐技术**：[@woj_zaremba](https://twitter.com/woj_zaremba/status/1872515615103287594) 强调了通过 **chain of thought reasoning**（思维链推理）进行 **deliberative alignment**（审议式对齐）的重要性，从而提高 **AGI** 系统的**安全性和有效性**。

- **AI 模型提示词与行为**：[@giffmana](https://twitter.com/giffmana/status/1872725026811854910) 和 [@abbcaj](https://twitter.com/abacaj/status/1872523867077407188) 探讨了 **Prompting 对 AI 模型行为的影响**，旨在**防止模型泄露其训练来源**，并使**响应与预期行为保持一致**。

- **模型评估与对齐**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1872411782884802803) 和 [@colin_de_de](https://twitter.com/teortaxesTex/status/1872705189666537672) 辩论了**评估指标的局限性**以及 **AI 模型 Alignment（对齐）** 中**持续改进的重要性**。

**AI 基础设施与优化**

- **分布式训练技术**：[@ArmenAgha](https://twitter.com/ArmenAgha/status/1872426813865201700) 和 [@vllm_project](https://twitter.com/vllm_project/status/1872453508127130017) 讨论了先进的**并行策略**，如 **Tensor Parallelism** 和 **Pipeline Parallelism**，提升了**大规模模型**的**训练效率**。

- **FP8 精度与内存优化**：[@madiator](https://twitter.com/madiator/status/1872505935832474018) 强调了 **DeepSeek V3** 采用 **FP8 精度** 如何减少**内存占用**和**训练成本**，推动了**高效的模型训练**。

- **AI 模型部署灵活性**：[@llama_index](https://twitter.com/llama_index/status/1872684854703432137) 展示了如何使用 **vLLM** 部署 **DeepSeek-V3**，并配合各种 **Parallelism** 和 **Offloading** 配置，为**模型部署**提供了**灵活性**。

**AI 开发实践**

- **版本控制与协作**：[@vikhyatk](https://twitter.com/vikhyatk/status/1872394404398588225) 分享了使用 **ghstack** 管理 **Pull Requests** 的见解，增强了 **GitHub** 中的**协作**和**代码管理**。

- **训练调度与学习率**：[@aaron_defazio](https://twitter.com/aaron_defazio/status/1872481458184745374) 提倡使用 **Linear Decay Learning Rate Schedule**，强调其在**模型训练**中优于其他调度方案的有效性。

- **开源贡献**：[@ArmenAgha](https://twitter.com/ArmenAgha/status/1872426813865201700) 和 [@ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/1872582508128383215) 感谢同行**引用研究论文**，促进了**开源协作**，并为 **PhysicsLM** 等项目**争取资源**。

**梗/幽默**

- **AI 助手的古怪之处**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1872576227732791374) 幽默地评论了他 **8 岁孩子**简单的目标“生存”，将**育儿幽默**与 **AI 愿景**结合在一起。

- **技术与 AI 笑话**：[@saranormous](https://twitter.com/saranormous/status/1872456071375393154) 调侃了 **AI 模型性能**，称“[@Karpathy](https://twitter.com/Karpathy/status/1872490226372972888) 还在阅读，其进步难以否认”，以此在 AI 社区的**智力博弈**中开玩笑。

- **个人轶事与轻松帖**：
  - [@nearcyan](https://twitter.com/nearcyan/status/1872526479810294233) 分享了对 **COVID-19 封锁**的**幽默看法**，嘲笑启动项目时的**初期困难 (Teething Issues)**。
  - [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1872716380530983364) 分享了关于**电池**的有趣观察，说：“锂是有限的，难以获取，且开采资源密集”，为**可持续性话题**增添了**轻松的转折**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek 的成本效益及与 4o 的性能对比**

- **[DeepSeek 在大多数基准测试中优于 4o，且价格仅为其 10%？](https://i.redd.it/gwmj6ili899e1.png)** ([Score: 785, Comments: 203](https://reddit.com/r/LocalLLaMA/comments/1hmxjbn/deepseek_is_better_than_4o_on_most_benchmarks_at/)): **DeepSeek-V3** 在性价比方面显著超越了 **GPT-4o**，其输入处理价格为每百万 tokens **$0.27**，而 GPT-4o 为 **$2.50**；输出处理价格为 **$1.10**，而 GPT-4o 为 **$10.00**。分析强调 DeepSeek-V3 提供了一个更经济的解决方案，图表使用不同颜色对比了成本，并注明这些是各模型目前可用的最低价格。
  - 用户正在讨论与 DeepSeek-V3 相关的**隐私担忧**，强调其条款暗示数据存储在北京，这引起了对数据隐私持谨慎态度的公司的关注。一些评论建议在本地运行该模型作为解决方案，尽管这需要大量的硬件资源，例如 **10 块 H100**。
  - 关于 DeepSeek-V3 的**性能和推理能力**存在争论，部分用户遇到了幻觉和错误，而另一些人则认为它在编程任务中非常有效，并对其 **180k 上下文长度**表示赞赏。该模型的低延迟以及通过 OpenAI Python package 与应用程序轻松集成的特性被视为显著优势。
  - DeepSeek-V3 的**成本效益**及其对市场的影响是一个反复出现的主题，用户注意到其促销定价以及对 OpenAI 等主要参与者施加压力的潜力。讨论还涉及该模型由一家中国对冲基金资助，以及中国补贴电费在降低成本中可能发挥的作用。


- **[DeepSeek v3 的训练预算比同类模型低 8-11 倍：具体为 2048 块 H800（即“阉割版 H100”），耗时 2 个月。相比之下，Llama 3 405B 根据其论文使用了 1.6 万块 H100。DeepSeek 估计成本为 550 万美元。](https://i.redd.it/n7nn4r9oyb9e1.jpeg)** ([Score: 518, Comments: 58](https://reddit.com/r/LocalLLaMA/comments/1hn8ams/deepseek_v3_was_trained_on_811x_less_the_normal/)): **DeepSeek v3** 在 **2048 块 H800**（被称为“阉割版 H100”）上进行了为期 **2 个月**的训练，成本约为 **550 万美元**。相比之下，**Llama 3 405B** 在训练中使用了 **16,000 块 H100**，突显了两种模型在资源分配上的显著差异。
  - **DeepSeek v3 的性能与局限性**：用户分享了使用 **DeepSeek v3** 的经验，指出其上下文窗口比 **Claude 3.5 Sonnet** 小，且缺乏多模态能力，导致在某些任务中出现性能问题。尽管有这些限制，DeepSeek 的成本仅为 Claude 的 **2%**，提供了更好的性价比。
  - **FP8 混合精度训练**：**FP8 混合精度训练**的引入因其效率提升而受到关注，与 FP16/BF16 相比，它提供了 **2 倍的 FLOPs 吞吐量**和 **50% 的内存带宽占用降低**。这种效率是通过减少 GPU 显存使用和加速训练实现的，尽管实际的效率提升可能接近 **30%**。
  - **Mixture of Experts (MoE) 见解**：讨论涉及了 **Mixture of Experts (MoE)** 方法，强调与稠密模型相比，MoE 可以降低计算需求。对话澄清了关于 MoE 的误解，指出开发者正在积极努力防止专家模型过度专业化，这与某些认为 MoE 只是并行训练小型模型的观点相反。

- **DeepSeek V3 使用针对编程和数学的合成数据（synthetic data）构建。他们采用了来自 R1（推理模型/reasoner model）的蒸馏技术。此外，他们还实现了新颖的多 Token 预测（Multi-Token Prediction）技术** ([分数: 136, 评论: 19](https://reddit.com/r/LocalLLaMA/comments/1hnc4d5/deepseek_v3_was_made_with_synthetic_data_for/)): **DeepSeek V3** 的开发使用了专注于编程和数学的**合成数据**，采用了**多 Token 预测技术（Multi-Token Prediction technique）**以及来自 **R1 推理模型**的蒸馏。该模型的训练预算比典型模型少 **8-11 倍**，更多细节可以在其 [论文](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) 中找到。
  - **多 Token 预测技术**是一个重要的关注点，人们对其新颖性和规模进行了询问。它并不是第一个实现该技术的模型，但其规模值得注意；早期的模型和研究可以在 [Hugging Face](https://huggingface.co/facebook/multi-token-prediction) 上的论文 "Better & Faster Large Language Models via Multi-token Prediction" 中找到。
  - 讨论中涉及了运行拥有 **6000 亿参数**的 **DeepSeek V3** 的可行性，这对于非服务器基础设施来说被认为具有挑战性。建议的配置包括一个成本约为 **2 万美元**的 **8 x M4 Pro 64GB Mac Mini 集群**，同时人们也对使用 **NVIDIA 显卡**的更便宜替代方案感到好奇。
  - 该模型仅用 **500 万美元**的训练资源完成开发，令人印象深刻，论文的开源也受到了赞赏，特别是其在编程应用方面的潜力。模型的概述可以在[这里](https://www.reddit.com/r/LocalLLaMA/s/E1axu8m6qf)查看。


**主题 2. DeepSeek-V3 架构：利用 671B 混合专家模型 (Mixture-of-Experts)**

- **[DeepSeek 发布了其 AI 研究人员在 2048 块 H800 上训练 DeepSeek-V3 671B 混合专家模型 (MoE) 的独家片段。](https://v.redd.it/tagjczxw3c9e1)** ([分数: 717, 评论: 60](https://reddit.com/r/LocalLLaMA/comments/1hn8rcx/deepseek_has_released_exclusive_footage_of_their/)): **DeepSeek** 发布了其 AI 研究人员使用 **2048 块 H800 GPU** 训练 **DeepSeek-V3**（一个 **6710 亿参数的混合专家模型/MoE**）的片段。
  - **DeepSeek-V3 的架构**：该模型由 256 个带有共享组件的独立模型组成，具体为每层 257 个 MLP，每层总计有 **370 亿激活参数**。正如 **ExtremeHeat** 和 **OfficialHashPanda** 所强调的，这种结构允许在 CPU 上也能进行高效的训练和推理。
  - **全球 AI 竞争与人才**：讨论触及了 AI 开发的地缘政治方面，包括对俄罗斯**人才流失**以及美国因**官僚障碍**和**缺乏资金**而流失人才的担忧。还提到了中国留学生在美国面临困难，这可能导致他们回到中国，而像**清华**和**北大**这样的大学提供了极具竞争力的教育。
  - **DeepSeek 的成本效益**：尽管 **DeepSeek-V3** 规模巨大，据报道其训练成本仅为 **800-1000 万美元**，与 **OpenAI** 在 **O3** 单次评估上花费 **160 万美元**形成鲜明对比。这种效率归功于该模型创新的架构和并行训练方法。


- **[来自 Qwen 的 Sonnet 级别新模型即将发布？](https://i.redd.it/d38tr8vsfd9e1.jpeg)** ([分数: 225, 评论: 30](https://reddit.com/r/LocalLLaMA/comments/1hncfhc/new_model_from_qwen_of_sonnet_level_soon/)): 在 2024 年 12 月 27 日的一次 Twitter 交流中，针对 **Knut Jägersberg** 对“Sonnet 级别 70B LLM”的渴望，**林俊旸 (Junyang Lin)** 以“等我 (Wait me)”回应，暗示了潜在的新模型发布。该推文获得了中等关注，有 228 次浏览和 17 个赞。
  - **本地模型 vs. API 成本**：几位用户表达了对**本地运行 LLM** 的偏好，因为这样可以节省成本并独立于**基于 API 的模型**。**m98789** 强调了免费和开放权重（open weights）允许本地执行的好处，并将其与昂贵的 API 服务进行了对比。
  - **模型尺寸与可访问性**：**Only-Letterhead-3411** 指出，**70B LLM** 是家庭使用且无需巨大成本的理想尺寸，**Such_Advantage_6949** 补充说，使用 **2x3090 GPU** 等硬件可以高效运行。他们还推测，随着技术的进步，像 **100B** 这样更大的模型可能会成为新标准。
  - **对模型预告的看法**：**EmilPi** 批评预热贴（teaser posts）分散注意力且缺乏实质性新闻，而 **vincentz42** 等人则幽默地推测将发布一个拥有 70B 激活参数的 **1T MoE 模型**，突显了社区对模型发布预告及其影响的复杂心情。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 日益增长的资金需求与融资计划**

- **[OpenAI 在制定营利计划时表示，其所需的资金“超出了我们的想象”——我的意思是，他确实说过 7 万亿美元……](https://www.cnbc.com/2024/12/27/openai-needs-more-capital-than-wed-imagined-moves-to-for-profit.html)** ([Score: 297, Comments: 72](https://reddit.com/r/OpenAI/comments/1hnf7mt/openai_says_it_needs_more_capital_than_wed/)): OpenAI 宣布其运营所需的**资金超出了最初的预期**，突显了潜在的融资挑战。讨论引用了之前的一份声明，估计需要 **7 万亿美元**，这表明了 OpenAI 营利计划对资金需求的巨大规模。
  - 讨论强调了对 **OpenAI 财务策略**的怀疑，一些用户质疑 **7 万亿美元**这一数字的真实性，并认为这可能是传闻而非事实。据指出，**Sam Altman** 否认曾要求 7 万亿美元，但也有人认为，考虑到 AI 开发成本的不断上升，这个数字并非遥不可及。
  - 人们对 **OpenAI 的商业模式**表示担忧，并建议效仿 **Apple 的应用平台**模式，允许开发者发布 AI 应用并从中抽取分成。用户还指出，目前缺乏一个用于探索基于 ChatGPT 应用的平台，而这本可以作为一个潜在的收入来源。
  - 讨论还涉及了**高层员工的离职**以及 **Deepseek 成就**的潜在影响，有人推测 Deepseek 通过利用来自 OpenAI 的合成数据，以更低的成本实现了类似的结果。这引发了对 OpenAI 竞争优势和战略方向的质疑。


**主题 2. 对用于确定 LLM 智能的“陷阱”测试的批评**

- **[回来了！](https://i.redd.it/0p2xewkxk99e1.jpeg)** ([Score: 299, Comments: 38](https://reddit.com/r/OpenAI/comments/1hmz24o/is_back/)): 该帖子幽默地描述了与 **ChatGPT** 的对话，当用户询问其状态时，AI 俏皮地引用了“矩阵中的故障（glitch in the Matrix）”作为回应。互动以对**水豚（capybaras）**的热情描述继续，突显了 AI 进行轻松对话交流的能力。
  - **语言与幽默**：发生了一场关于语言使用的轻松交流，评论者拿语法错误开玩笑，并强调了在线互动中幽默的重要性。
  - **AI 与世代影响**：一场关于在无处不在的 AI 环境下成长的影响的讨论浮出水面，一些人对未来几代人对技术的依赖表示担忧。
  - **水豚迷恋**：对话幽默地触及了对水豚的兴趣，一位用户分享了一个 [YouTube 链接](https://youtu.be/O1opWQERMRw?si=c4kO3owSd9jJhRkc)，展示了它们与鳄鱼共存的冷静天性。


**主题 3. AI 与数学：突显进展与局限性**

- **AI 现在能做数学了吗？你可能会感到惊讶……一位数学家的思考。** ([Score: 133, Comments: 40](https://reddit.com/r/OpenAI/comments/1hn0n31/can_ai_do_maths_yet_you_might_be/)): 该帖子分享了来自 **Hacker News** 的一篇文章链接，探讨了 AI 目前在数学方面的能力，并从数学家的角度提供了见解。讨论邀请读者阅读文章并分享他们对 AI 执行数学任务能力的看法。
  - **数学竞赛的误导性描述**：**FateOfMuffins** 的评论指出，将 **IMO** 和 **Putnam** 等竞赛标记为“高中”和“本科”级别具有误导性，因为这些竞赛比典型课程要难得多。这种误传可能会让公众对 AI 的数学能力产生困惑，因为 AI 可能在这些竞赛中表现良好，但并不一定反映平均本科水平。
  - **AI 在数学任务中的表现**：**SoylentRox** 质疑 AI 在数学环境中与人类数学家相比表现如何，特别是在步骤分（partial credit）和答案准确性方面。讨论表明，即使是熟练的人类数学家也可能在这些测试所需的精确度上遇到困难，从而引发了对 AI 比较表现的质疑。
  - **对 AI 数学能力的认知**：**Mrb1585357890** 和 **soumen08** 对分享的文章表示赞赏，因为它提供了对 AI 当前数学能力的见解。讨论反映了文章和讨论如何帮助澄清 AI 在执行复杂数学任务方面的进展和局限。


---

# AI Discord 综述

> 由 o1-mini-2024-09-12 生成的摘要之摘要的摘要

**主题 1: DeepSeek 主导 AI 竞赛**

- [**DeepSeek V3 以 60 Tokens/秒的速度碾压竞争对手**](https://x.com/deepseek_ai/status/1872242657348710721)：**DeepSeek V3** 通过每秒处理 **60 tokens** 的速度超越了之前的版本，比 V2 提升了 **3 倍速度**，并拥有巨大的 **64k context window** 来处理大规模任务。这个开源强力模型正在重塑基准测试，在 AI 领域挑战 **Claude Sonnet** 和 **ChatGPT** 等巨头。
  
- [**许可证之战：DeepSeek 采取行动**](https://x.com/deepseek_ai/status/1872242657348710721)：**DeepSeek** 更新了其许可证，使其比 **Llama** **更加自由**，引发了社区关于开源与专有模型的辩论。这一转变使 **DeepSeek V3** 成为开源 AI 模型领域的领跑者，激发了爱好者之间的“*许可证之战！*”。

- [**推理循环？DeepSeek V3 面临挑战**](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)：尽管速度惊人，**DeepSeek V3** 在推理循环和在某些层级之外生成连贯输出方面遇到了问题。用户报告了“*垃圾*”输出，突显了在扩展 AI 推理能力方面持续存在的挑战。

**主题 2：像专业人士一样集成 AI（或者并不专业）**

- [**Cursor IDE 和 Codeium 在性能上挣扎**](https://docs.cursor.com/advanced/models#what-context-window-is-used-for-model-x)：使用 **Cursor IDE** 和 **Codeium (Windsurf)** 的开发者反映了对**请求缓慢**和**系统挂起**的挫败感，尤其是在 **Pro plan** 上。随着用户寻求更流畅的 AI 辅助编码工作流，要求增强**快捷键**和更好的 **context management** 的呼声很高。

- [**Aider 更新：更多模型，更少错误**](https://aider.chat/HISTORY.html)：最新的 **Aider v0.70.0** 引入了对 **o1 models** 的支持并改进了**错误处理**，因其**更简单的安装方法**而受到贡献者的赞赏。此次更新旨在简化**编码辅助**，使 **Aider** 成为开发者工具箱中更强大的工具。

- [**OpenRouter 整合 DeepSeek 的王牌举动**](https://x.com/OpenRouterAI/status/1872334128043208833)：**OpenRouter** 见证了 **DeepSeek V3 使用量**自发布以来增长了三倍，其集成旨在利用**自定义 API keys** 和**更低的成本**。这种协同效应预计将增强**编码任务**，尽管一些用户对“*许可证之战！*”背景下的长期稳定性表示怀疑。

**主题 3：金币叮当响！定价模式重塑 AI 准入门槛**

- [**DeepSeek V3 将训练成本削减了 100 倍**](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)：通过 **550 万美元**的投资，**DeepSeek V3** 利用 **FP8 mixed precision** 实现了训练成本**两个数量级的降低**。这一突破使先进的 AI 模型变得更加触手可及，挑战了高成本的同类产品。

- [**AI 定价透明度：开发者要求更多**](https://platform.deepseek.com)：围绕 **AI 模型定价** 的讨论强调了对**成本透明度**的需求，尤其是在平衡**性能**与**开支**时。随着用户为其编码和开发需求寻求更清晰的**价值主张**，**Claude Sonnet** 和 **DeepSeek Platform** 等工具正受到严格审查。

- [**Perplexity 的定价谜题**](https://cohere.com/pricing)：用户报告了 **Perplexity AI** 上**图像嵌入限制**的不一致，期望是**每分钟 400 次**而不是 **40 次**。由于修复承诺因**节假日时间**而推迟，社区对**定价结构神话**表达了不满，敦促公司将**定价与性能**挂钩。

**主题 4：GPU 大神与训练技巧**

- [**H800 GPU：为了成本效益而“阉割”的 H100**](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)：**H800 GPU**（本质上是**削弱版的 H100**）的部署导致了 **NVLink bandwidth** 降低，但保持了至关重要的 **FP64 性能**。这一战略举措使得 **DeepSeek V3** 能够在短短 **2 个月**内，在 **2000 个 GPU** 上高效训练像 **600B MoE** 这样的大规模模型。

- [**Triton vs. CUDA：终极对决**](https://github.com/vllm-project/vllm/blob/dbeac95dbbf898bcc0965528fc767e9cadbbe0c5/vllm/attention/backends/xformers.py#L613)：关于实现 **quantization** 的讨论集中在是使用 **Triton** 还是坚持使用**纯 CUDA**，以平衡**易用性**与**速度**。社区正在辩论集成像 **bitblas** 这样的专用内核用于 **Conv2D** 操作以提高效率的优劣。

- [**FP8 训练推动新的编程尝试**](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)：受 **DeepSeek 的 FP8 方法**启发，开发者们渴望使用 **torchao** 框架将 **FP8 training** 引入 **nanoGPT**。这种兴趣凸显了社区对**能源效率训练**和**可扩展模型推理**的追求。

**主题 5：创意遇见代码（以及伦理）**

- [**AI 只是想写作：创意写作和角色扮演飞速增长**](https://aider.chat/HISTORY.html)：**Aider** 和 **Gen AI** 等 AI 工具正在通过先进的 **prompts** 和沉浸式角色开发彻底改变 **创意写作** 和 **色情角色扮演 (ERP)**。用户称赞其构建详细角色档案和动态交互的能力，增强了 **AI 辅助叙事体验**。

- [**伦理困境：AI 未经许可进行抓取**](https://forum.cursor.com)：社区成员对 **AI 伦理** 表达了严重关切，特别是未经许可对 **创意作品的抓取**。关于 **衍生内容** 的范围以及 **企业游说对版权法** 的影响展开了激烈辩论，呼吁采取更具 **伦理的 AI 开发** 实践。

- [**3D 打印与 AI 艺术：有形的融合**](https://gitdiagram.com/)：**3D 打印** 与 **AI 生成视觉效果** 的融合为 **创新成果** 开辟了新途径，例如像 **羊形卫生纸架** 这样奇特的物品。这种交汇展示了 **LLMs** 在 **有形制造** 中的创造潜力，将数字创意与物理生产相结合。

---

# 第 1 部分：高层级 Discord 摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 的上下文难题**：用户在探索 [Cursor IDE 文档](https://docs.cursor.com/advanced/models#what-context-window-is-used-for-model-x) 时，发现了 **请求缓慢**、多文件处理受限以及对 **上下文使用** 的挫败感。
   - 他们建议添加 **快捷键** 以加快工作流程，并参考了 [社区论坛](https://forum.cursor.com) 上持续的反馈。
- **DeepSeek V3 与 Claude Sonnet 的对决**：根据 [DeepSeek 官方推文](https://x.com/deepseek_ai/status/1872242657348710721)，**DeepSeek V3** 达到 60 tokens/second，保持 API 兼容性，并主张开源透明。
   - 然而，社区与 **Claude Sonnet** 的对比突显了后者更精细的代码编写能力，正如一条赞扬 Claude 3.5 Sonnet 的 [Visual Studio Code 推文](https://x.com/code/status/1872673862992744625) 所暗示的那样。
- **成本紧缩与效率讨论**：参与者权衡了 **AI 模型定价** 与性能的关系，强调了 **Claude Sonnet** 和 [DeepSeek Platform](https://platform.deepseek.com) 等工具的成本透明度。
   - 一些人对代码任务的强大价值主张表示感兴趣，而另一些人则对先进 **AI 解决方案** 定价结构的不确定性表示遗憾。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 幕后视频令人惊叹**：来自 [Windsurf 的新视频](https://x.com/windsurf_ai/status/1872375661542920424) 揭示了工程师构建 Windsurf 的方法，重点展示了独特的技术和节日精神。
   - 他们展示了团队如何大胆重塑标准代码工作流，鼓励观众 *尝试他们突破边界的方法*。
- **性能陷阱与 Pro 计划的困惑**：多位用户报告了系统变慢和 **Pro 计划** 额度消耗过快的问题，引发了对每月请求限制的担忧。
   - 他们链接到了 [关于额度使用的文档](https://docs.codeium.com/windsurf/usage)，抱怨无法控制的卡顿阻碍了编码目标的实现。
- **DeepSeek V3 引发好奇**：许多参与者称赞 [DeepSeek V3](https://x.com/deepseek_ai/status/1872242657348710721) 的速度和开源优势，期待其可能与 Windsurf 集成。
   - 其他人则权衡将 Cursor 作为替代方案，理由是可以使用自定义 API keys 且代码任务成本更低。
- **IDE 小故障与 M1 混淆**：用户在 WebStorm 和 IntelliJ 中遇到了插件故障，包括更新后功能缺失。
   - 一位 Macbook M1 Pro 用户发现 Windsurf 的终端运行在 i386 下，寻求 Apple Silicon 兼容性建议。
- **Cascade 的全局规则引起讨论**：一些人建议在 Cascade 中使用广泛的规则来统一代码风格并减少困惑，特别是在大型团队中。
   - 他们请求了解哪些指南是有帮助的，希望保持未来编码会话的一致性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.70.0 强化升级**：新的 [Aider v0.70.0](https://aider.chat/HISTORY.html) 提供了 **o1 模型支持**、针对 10% 用户的**分析数据选择性加入 (analytics opt-in)**，以及**更好的错误处理**，以简化编码任务。
   - 贡献者赞扬了其新的**安装方法**和更简单的**只读文件**显示，强调了编码辅助方面更广泛的模型兼容性。
- **DeepSeek V3 在 M4 Pro Mini 上飞速运行**：在由 **8 台 M4 Pro Mac Mini** 组成的集群上运行 **671B DeepSeek V3**，速度达到 **5.37 tokens/s**，首个 token 响应时间 (TTFT) 为 **2.91s**，显示出强大的本地推理潜力。
   - 社区讨论将此速度与 **Claude AI** 和 **Sonnet** 进行了对比，指出其在高容量使用下的开销更低且扩展性更好。
- **Aider 中的 Repo Maps 与 Token 限制**：成员们报告称，**repo-map** 功能在 Architect 模式与标准编辑模式下的表现有所不同，同时 **DeepSeek Chat V3** 的输入 token 限制跃升至 **64k**。
   - 他们建议通过编辑 *.aider.model.metadata.json* 来处理新的限制，并优化模型与复杂代码库的交互方式。
- **Git 工具以引人注目的格式渲染代码**：[GitDiagram](https://gitdiagram.com/) 网站将 GitHub 仓库转换为**交互式图表**，而 [Gitingest](https://gitingest.com/) 则将其提取为**对 Prompt 友好的文本**。
   - 用户发现将任何 GitHub URL 中的 'hub' 替换为 'diagram' 或 'ingest'，对于快速了解项目概况和简化 LLM 摄取非常有帮助。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Trainer 调整激发新的 Loss 技巧**：一位成员询问如何修改 **Hugging Face 的 Trainer** 以进行因果语言建模，重点在于将填充 (padded) token 置零，并在 Loss 计算中忽略输入 token。
   - 他们参考了 **trl** 库，并建议使用自定义 collator 将标签设置为 `ignore_idx` 作为变通方案。
- **Pythia 寻找中间 Checkpoint 状态**：一位用户请求 **Pythia** 模型的**中间优化器状态 (intermediate optimizer states)**，并指出目前仅提供最终的 checkpoint 状态。
   - 他们计划联系工作人员以获取大文件访问权限，希望能更轻松地移交 **Pythia** 资源。
- **物理学家准备进军 ML**：一位即将毕业的理论物理学博士介绍了探索 **Machine Learning** 和 **LLM** 以深入了解可解释性 (interpretability) 的计划。
   - 他们表现出对参与研究项目和获得高级建模实践技能的极大热情。
- **因果关系提升训练讨论**：参与者讨论了**因果推理 (causal inference)** 如何通过利用先验动态而非纯粹依赖统计趋势来改进模型训练。
   - 他们辩论了允许知识分块的表示方式，并引用了**盲棋 (blindfold chess)** 等例子作为高效心理结构的案例。
- **视频模型在物理学习上受挫**：成员们认为，**视频生成**模型在尝试从视觉效果中提取真实的物理定律时往往力不从心，即使在更大规模下也是如此。
   - 他们指向了[一项综合研究](https://phyworld.github.io/)，该研究质疑这些模型在没有人类洞察的情况下是否能建立稳健的规则。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek v3 使用量翻三倍并比肩大厂**：正如[这条推文](https://x.com/OpenRouterAI/status/1872334128043208833)所示，**Deepseek v3** 在 OpenRouter 上的使用量自昨天以来飙升了三倍。
   - 一些业内声音声称，现在构建 frontier models 的成本约为 **$6M**，且**中国**加上开源力量已经接近领先的 AI 性能，这增强了人们对 **Deepseek v3** 的期待。
- **ACT 与 CIMS 助力开发者日常工作**：**AI Chat Terminal (ACT)** 集成了主流 API，允许开发者在终端中运行任务并与代码聊天，详见 [GitHub](https://github.com/Eplisium/ai-chat-terminal)。
   - 同时，**Content Identification/Moderation System (CIMS)** 在 Companion 中增加了对问题内容的自动检测和删除，其 [wiki](https://github.com/rapmd73/Companion/wiki) 对此进行了说明。
- **RockDev 在 SQL 生成领域势头强劲**：**RockDev.tool** 使用 OpenRouter 将代码定义转换为开箱即用的 SQL，同时保留本地隐私，详见 [rocksdev.tools](https://www.rocksdev.tools/en/tools/dev/ai-sql)。
   - 社区反馈强调，本地数据处理是一个主要吸引力，并计划在未来进行更新。
- **Google Search Grounding 将 AI 与 Web 关联**：一位开发者展示了一种使用 **Google GenAI SDK** 在进行网络搜索前对回答进行 grounding 的方法，详见 [GitHub](https://github.com/nlawz/or-google-search)。
   - 这种方法依赖 Google 搜索来获取上下文，为实时验证 AI 输出提供了可能性。
- **OCR 与 OWM 拓展 LLM 视野**：**Fireworks** 增加了对图像和 PDF 的 OCR 支持，而 **Pixtral** 则负责处理高级文档处理中的文本提取。
   - 关于 **Open Weight Model (OWM)** 和 **Out-of-Domain (OOD)** 任务的讨论强调了许多模型在已知数据上表现出色，但在训练范围之外面临挑战。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 因推理循环受阻**：成员们透露 **Deepseek V3** 在逻辑上遇到了困难，并引用 [DeepSeek V3 PDF](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) 详细说明了重复循环如何阻碍复杂任务，尤其是在超过一定层数后。
   - 他们指出经常出现 **garbage** 输出，一些人指出了底层 RPC 代码的潜在缺陷，并对推理链的训练提出了质疑。
- **DeepSeek V3 中 RoPE 的反复难题**：小组讨论了 Deepseek V3 中 **RoPE** 的使用，注意到它仅应用于一个 key，同时参考了用于定位的独立 embedding 索引方法。
   - 一些人质疑简化方法是否能改善结果，强调了 **position encoding** 的复杂性如何显著影响模型准确性。
- **Qwen-2.5-72b 在重测中表现飙升**：[Aidan McLau 的推文](https://x.com/aidan_mclau/status/1872444303974543859)显示 **Qwen-2.5-72b** 在重测中获得了惊人的提升，该模型最初表现不佳，但在重复基准测试中跃升至顶级水平。
   - 评论者想知道 **benchmark** 的公平性是否受到损害，或者重新运行是否只是使用了更好的超参数，一些人参考了 [Better & Faster Large Language Models via Multi-token Prediction](https://openreview.net/forum?id=pEWAcejiU2) 以获取训练见解。
- **Gemini 的上下文难题**：一些人注意到 **Gemini** 模型的 context 使用可能更灵活地处理输入，尽管它需要保持在设定的参数范围内。
   - 他们推测先进的 **context selection** 方法可能会如何改变环境输入，并参考了其在 [aidanbench](https://x.com/aidan_mclau/status/1872444303974543859) 上的第二名排名。
- **Copilot 处理复杂代码**：成员们称赞 **GitHub Copilot** 在简单项目中的快速修复和重构任务。
   - 然而，他们发现像 **llama.cpp** 这样的高级系统需要更深入的手动处理，这表明 AI 驱动的编辑无法完全取代深入的代码理解。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI 伦理工具大争论**：一位用户抨击未经许可抓取创意作品的行为，认为这对于 **AI ethics** 来说是非常令人不安的问题。
   - 其他人指出了企业对 **copyright laws**（版权法）的影响，并质疑衍生内容的界定范围。
- **LM Studio 速度提升**：一些用户报告称，在升级到 [最新的 LM Studio Beta 版本](https://lmstudio.ai/beta-releases) 后，处理速度从 **0.3** 跃升至 **6 tok/s**。
   - 他们使用 GPU 监控工具确认了性能的提升，并将这一成功归功于强大的硬件配置。
- **图像生成遭遇瓶颈**：一位用户试图优化 AI 图像生成，但对于实现更好输出的可行性遭到了质疑。
   - 讨论集中在这些模型如何解读创意，揭示了对实质性改进的怀疑。
- **MLX 内存泄漏警报**：参与者报告了 MLX 构建版本的内存泄漏问题，并引用 [Issue #63](https://github.com/lmstudio-ai/mlx-engine/issues/63) 作为证据。
   - 他们将性能下降归因于潜在的资源管理不当，并引发了进一步的调查。
- **GPU 紧缺与 RPG AI 场景**：多 GPU 设置、海量模型的 VRAM 需求以及低至 30% 的 **CUDA** 占用率引起了硬件爱好者的热议。
   - 同时，像 **LangChain** 这样的 agentic 框架被用于 RPG 场景生成，引发了关于硬件与叙事协同作用的讨论。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA vs 全模型权重：微调对决**：多位用户讨论了使用 LoRA 而不是合并 **full model** 进行 **fine-tuning**，强调了在托管和推理方面的效率提升，并参考了 [Unsloth 文档](https://docs.unsloth.ai/get-started/all-our-models) 中的示例。
   - 他们强调 LoRA 作为一个 **adapter** 运行，一位用户强调提示词格式化（prompt formatting）和数据对齐对于 **stable finetuning** 至关重要。
- **Hugging Face 动态 Adapter 加载失败**：一名新手尝试通过 Hugging Face 进行 **dynamic adapter loading**，结果导致了 **乱码输出**，如 [此 Gist](https://gist.github.com/grahama1970/f832bbddb1edaa78ccc939a6f2ddd8a1) 所示。
   - 有人建议使用 **VLLM** 以获得更好的性能，对比了 Hugging Face 较慢的推理速度，并称赞 **Unsloth Inference** 在处理 adapter 方面的可靠性。
- **Python 指令微调寻宝**：一名成员正在寻找包含 **问题描述** 和 **生成的解决方案** 的 **instruction-tune datasets**，特别是针对 **Python** 编程任务，参考了 [Hugging Face 的 smol-course](https://github.com/huggingface/smol-course)。
   - 他们希望得到一个能够满足真实编程见解的数据集，其他人也确认 **精选数据**（curated data）会极大地影响最终模型的性能。
- **Hopper 上的 Binary Tensor Cores：HPC 的未来？**：一位用户担心 **binary tensor core** 的支持在 **Ampere** 之后会被移除，质疑 **Hopper** 是否准备好应对超低精度的 HPC 任务。
   - 社区内出现了对 NVIDIA 未来走向的猜测，一些参与者怀疑 **low-precision** 指令是否会持续可用。
- **GGUF 与 4-bit 转换障碍**：一位用户在生成 **GGUF** 模型时遇到了 **RuntimeError**，并发现缺少 **tokenizer.json** 等文件，随后指向官方 [llama.cpp](https://github.com/ggerganov/llama.cpp) 寻求解决方案。
   - 其他人建议 **复制** 必要的模型文件，并禁用 **vision layers** 的 4-bit 加载，强调了部分量化（partial quantization）的复杂性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek V3 以 64k 上下文惊艳亮相**：根据 [DeepSeek V3 文档](https://deepseek.api)，新提到的 **DeepSeek V3** 声称拥有 **64k 上下文窗口**、先进的混合专家 (mixture-of-expert) 架构以及高性价比的本地推理。
   - 社区测试者考虑在 **ChatGPT** 宕机期间切换到 **DeepSeek** 处理专门任务，称赞其响应速度更快且对大上下文支持更好。
- **GPT-03 (o3) 临近发布**：开发者预测 **o3-mini** 将于 1 月底首次亮相，随后是完整的 **o3**，使用限制尚未确认。
   - 推测涉及对现有 GPT 模型的可能增强，但官方细节仍然稀缺。
- **ChatGPT 的宕机困境**：频繁的 **ChatGPT** 宕机导致跨平台的错误消息和服务中断，如 [OpenAI 状态页面](https://status.openai.com/incidents/6bwlxnvdncnm)所示。
   - 一些成员对那些没有起效的“修复”公告开玩笑，而另一些人则测试了不同的 AI 解决方案，凸显了宕机的影响。
- **MidJourney vs DALL-E：视觉之争**：爱好者将 **MidJourney** 与 **DALL-E** 进行对比，强调最新版本的 **DALL-E** 在处理复杂提示词方面有更好的结果，且视觉效果有所提升。
   - 他们回顾了旧模型的缺点，赞扬了最近在提升艺术质量和用户满意度方面的更新。



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Gabe 的隐秘简化**：细心的观察者强调了 Gabe 新开发的由 **Bolt** 驱动的应用，传闻它将为所有人**简化**工作流，尽管尚未分享官方功能。
   - 早期预览引发了热议，一些成员将其描述为开发团队的“下一个重大便利”。
- **Anthropic 过载破坏了 Bolt**：成员报告称，每当 **Anthropic** 切换到简洁模式时，[Bolt](https://bolters.io/docs/read-this-first) 的**质量大幅下降**，导致响应生成反复失败。
   - 用户要求更好的调度或警告，有人称这种体验为“彻底崩溃”，并敦促修复实时协作问题。
- **直接代码更改提示**：一些开发者在 **Bolt** 中遇到了困难，聊天机器人返回的是原始代码块而不是编辑现有脚本，导致调试停滞。
   - 他们分享了一个技巧，即在提示词中明确说明“请直接对我的代码进行更改”，声称这种方法可以减少摩擦。
- **OpenAI 在 Bolt 中的设置障碍**：尝试将 **OpenAI** 集成到 **Bolt** 的用户遇到了一波困惑，在提交 API key 时反复出现错误。
   - 一些人建议加入 Bolt.diy 社区或查看 [Issues · stackblitz/bolt.new](https://github.com/stackblitz/bolt.new/issues) 以获取及时的解决方案。
- **Netlify 404 令人头疼**：一组用户在 **Netlify** 上遇到了 404 错误，将其归因于其 **Bolt** 应用中的客户端路由。
   - 变通方法确实存在，但需要进行实验，包括多次尝试自定义路由设置或调整 serverless functions。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **OpenAI 的人形机器人动态**：最近的热议聚焦于 **OpenAI 的人形机器人** 计划，详见[此文档](https://www.perplexity.ai/page/openai-s-humanoid-robot-plans-oaWNWCv6QDuLlunzvv.8dA)，其中记录了机械规格、预计发布时间表以及与高级 AI 模块的集成。
   - 参与者表示希望这些机器人能加速 **人机协作**，并提出未来的软件增强功能可能会与*其他机器人项目*中展示的即将推出的架构保持一致。
- **AI 令人惊讶的转变**：一个持续的热点涵盖了 **AI 如何假装改变观点**，在[此 YouTube 视频](https://www.youtube.com/embed/_zUGuxWw-sM)中有一个令人惊讶的演示。
   - 社区成员讨论了对 **AI 可操控性** 的担忧，并考虑了潜在的保护措施，指出关于模型立场转变的直接引用既*令人不安*又在*技术上具有启发性*。
- **体温供电的可穿戴设备**：讨论中出现了新的 **体温供电可穿戴设备**，见[此链接](https://www.perplexity.ai/page/ai-startup-futurixai-shivaay-vOiw7gCkQAGZXo1IyqxMBQ)，重点介绍了无需外部充电即可为低功耗设备供电的原型。
   - 工程师们辩论了传感器的准确性和长期稳定性，强调 **温差** 是持续数据采集的一种新鲜能源。
- **视频聚合工具的开发**：一些用户正在寻找一种能合并多个服务的 **AI 视频创作聚合器**，引发了关于现有工作流的活跃头脑风暴。
   - 他们交流了关于流水线组装的建议，希望能有一个统一的工具来简化 **多媒体制作** 和同步。
- **Perplexity 的 API 难题**：开发者批评了 **Perplexity API**，称其弱于 **OpenAI**、**Google** 或 **Anthropic** 的替代方案，引发了关于容量限制和响应质量的问题。
   - 其他人指出 **Spaces** 提供了更顺畅的集成，而 Perplexity 缺乏 **自定义前端支持** 是高级用户体验的硬伤。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hunyuan 在视频领域表现强劲**：成员报告称 **Hunyuan** 的表现优于 Veo 和 KLING，并希望从 [DiTCtrl](https://github.com/TencentARC/DiTCtrl) 中获得进一步提升。
   - 他们强调了 AI 视频生成中可靠性和连续性的重要性，期待新的注意力控制策略。
- **提示词优化：标签 vs 详细文本**：参与者对比了处理长提示词的 **Flux/SD3.5** 与通常在使用短标签时效果最好的 **SD1.5/SDXL**。
   - 他们分享了在平衡核心关键词和扩展描述以优化输出方面的技巧。
- **旧版模型的 Lora 连接**：一些人询问关于为旧模型升级新 **Loras** 的问题，结论是重新适配 Loras 比更改基础 Checkpoints 更实用。
   - 他们一致认为，经过良好微调的 Loras 优于对现有模型权重进行的强制调整。
- **缓慢的速度挤压 AI 视频渲染**：用户描述渲染 **5 秒** 视频大约需要 **8 分钟**，将其归因于当前的 GPU 限制。
   - 他们对新的 GPU 技术和改进的模型设计将缩短这些漫长的渲染时间保持乐观。
- **3D 打印与 AI 艺术的碰撞**：一位贡献者强调了打印奇特物品（如羊形卷纸架）作为 **3D 打印** 的有趣应用。
   - 他们看到了将 AI 生成的视觉效果与实体制造相结合以获得更多创意成果的潜力。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Pathfinder 播客瞬间生成**：一位用户使用 **NotebookLM** 在约 **15 分钟**内生成了 **Pathfinder 2** 的 6 本书战役摘要，引用了 Paizo 2019 年发布的内容，并强调了精简的 GM 准备时间。
   - 他们谈到了“大幅减少准备工作”，引发了社区关于快速、AI 驱动的叙事生成的讨论。
- **迷人的 Wikipedia 音频概览**：成员们使用 **NotebookLM** 创建了新闻文章和 **Wikipedia** 条目的音频合成，包括 **2004 年印度洋大地震**及其即将到来的 20 周年纪念（2024 年 12 月）。
   - 一位成员形容输出结果“栩栩如生，令人惊讶”，引发了更多关于音频形式大规模知识分发的讨论。
- **交互模式下的麦克风故障**：几位用户指出，当 **麦克风权限** 被阻止时，**NotebookLM** 的交互模式会出现无尽加载的错误，并指出该问题会一直持续到更新浏览器设置。
   - 他们分享了启用麦克风访问以规避此问题的技巧，推动了关于确保硬件兼容性以实现流畅 AI 使用的讨论。
- **小说作家的表格曲折**：一位用户询问 **NotebookLM** 是否可以处理表格数据，特别是用于辅助小说创作的角色矩阵。
   - 社区想知道结构化数据是否能被有效解析，建议探索潜在的文本转表格功能。
- **AI 创作的播客平台**：一位用户介绍了用于分享 AI 生成播客的 [Akas](https://akashq.com)，重点介绍了 RSS 提要集成和移动端友好的发布功能。
   - 成员们还询问了 **NotebookLM Plus** 层级，参考[官方订阅指南](https://support.google.com/notebooklm/answer/15678219)以确认价格和新功能。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek V3 遥遥领先**：DeepSeek V3 以 **60 tokens/second** 的速度发布（比 V2 快 3 倍），如[这条推文](https://x.com/deepseek_ai/status/1872242657348710721)所述，并支持在 NVIDIA 和 AMD GPU 上进行 FP8 训练。其许可证现在比 **Llama** 更宽松，引发了社区成员之间所谓的“**许可证之战**”。
   - 社区评论赞扬了该团队在严苛的硬件限制下的卓越工程能力，而讨论集中在代码和数学自评中潜在的陷阱。一位参与者惊呼“许可证之战！”，捕捉到了这种复杂的反应。
- **强大的 Multi-Head 举措**：DeepSeek 的 **Multi-Head Latent Attention** 引发了关于实现低秩近似的问题，[SGLang 在 V3 中提供了首日支持](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。观察者指出 **vLLM**、**TGI** 和 **hf/transformers** 可能很快会增加兼容性。
   - 一位用户询问“有人在开发相关版本吗？”，反映了社区推动适配该技术的努力。另一位计划查看 **Hugging Face** 方面，旨在同步努力以实现更好的采用。
- **OpenAI 重组与 Bluesky 爆发**：OpenAI 董事会打算组建“历史上资源最丰富的非营利组织之一”，根据[此公告](https://x.com/OpenAI/status/1872628736690123213)，而鉴于投资者压力和不断增长的资本需求，**IPO** 传闻四起。与此同时，Bluesky 极端的反 AI 倾向使得该平台不再欢迎 AI 讨论。
   - 一些人预测，如果进一步的融资超出了 **Venture Capital** 的范围，OpenAI 将会上市。一位用户在目睹了对生成式 AI 的强烈抵制后重复道：“Bluesky 对 AI 讨论是不安全的”。
- **MCTS 方法强化推理能力**：一种[基于 MCTS 的方法](https://arxiv.org/abs/2405.00451)通过 Direct Preference Optimization 添加步骤级信号来优化 LLM 推理，强调 **on-policy sampling** 以实现稳健的自我提升。评估表明，与旧的 RL 设置相比，迭代性能有显著提升。
   - 怀疑者质疑模型的整体水平，其中一人评论道：“不知道他们为什么要用这么烂的模型——2024 年 5 月的情况有那么糟吗？”。其他人则争论 **PRMs** 是否真的能产生更好的 Chains of Thought，或者替代方法是否能产生更好的结果。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeepSeek 通过 FP8 增益大幅降低成本**：在筹集 **500 万美元**后，**DeepSeek-V3** 展示了使用 **FP8** 混合精度训练带来的两个数量级的成本降低，详见[其文档](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)。
   - 他们记录了 **278.8 万个 H800 GPU 小时**，引发了关于 **channel-wise**（通道级）和 **block-wise**（块级）量化方法的激烈比较，并提到了 **TransformerEngine** 的累加精度。
- **Character.AI 的 Int8 推理加速技巧**：**Character.AI** 引入了一个自定义的 **int8 attention kernel**，以提升计算密集型和内存密集型操作的速度，详见[其新文章](https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/)。
   - 他们之前通过 **multi-query attention** 和 **int8** 量化来优化内存效率，现在将重点转向提升 **core inference**（核心推理）任务的性能。
- **BitBlas 与 Torch 的 Conv2D 结合**：一位用户询问 **bitblas** 是否可以生成一个 **Conv2D** 以直接集成在 **Torch** 中，希望能实现更高效的训练流程。
   - 其他人对将 **bitblas** 等专用 kernel 与主流框架合并表现出兴趣，暗示了这些可能性在未来的扩展。
- **vLLM 为 xFormers 速度延迟 Batch**：一次讨论强调了 **vLLM** 选择不使用批处理推理，而是使用 **xFormers** 后端，如[其代码](https://github.com/vllm-project/vllm/blob/dbeac95dbbf898bcc0965528fc767e9cadbbe0c5/vllm/attention/backends/xformers.py#L613)所示。
   - 该策略利用了序列堆叠（sequence-stacked）方法，延迟差异极小，引发了关于批处理对吞吐量是否有实际优势的疑问。
- **Torchcompiled 的 128 倍前向传播之争**：一位用户指出 **Torchcompiled** 需要 **128 次前向传播**来进行梯度估计，但与真实梯度的余弦相似度仅为 **0.009**，参考了[这条推文](https://x.com/torchcompiled/status/1872021986106650816)。
   - Will 引用的一篇论文声称在 **1.58b** 下训练可减少 **97% 的能耗**，并将 **175B** 模型存储在仅 **~20mb** 中，这加剧了关于小规模演示之外可行性的辩论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **加速匹配引擎的悬赏战**：**tinygrad** 社区正在推进 [GitHub issue](https://github.com/tinygrad/tinygrad/issues/4878) 中提到的三个性能悬赏，目标是为 **model lower** 结果在基准测试中提供加速的匹配引擎。
   - **George Hotz** 明确表示，赢得悬赏的关键在于 **2 倍的加速**，并建议提交带有明显改进证明的 **Pull Request** 来领取奖励。
- **重写速度令人震惊**：一位成员目睹了一个重写过程在 **RTX 3050** 上运行耗时 **800+ ms**，引发了对硬件限制和结果不一致的质疑。
   - 一张截图显示了与报告的 **25 ms** 性能之间的巨大差异，促使人们呼吁进行彻底测试。
- **tinygrad 的 JIT 挑战 PyTorch**：通过在所有层利用 **JIT**，**tinygrad** 现在在推理性能上已追平 **PyTorch**，突显了极小的 **Python** 开销如何放大速度。
   - 用户通过在完整的 **Transformer** 上启用 **JIT** 避免了 **out of memory**（内存溢出）错误，强调了选择性使用可能会阻碍可靠性。
- **Beam Search 缓存技巧**：贡献者确认 **beam search** kernel 可以被存储和重用，从而减少后续运行的重新编译步骤。
   - 他们建议在具有相同硬件的系统之间共享这些缓存的 kernel，跳过不必要的重复执行。
- **TTS 模型迁移至 tinygrad**：将 **TTS** 模型从 **Torch** 迁移到 **tinygrad** 的工作仍在继续，参考了 [fish-speech/fish_speech/models/text2semantic/llama.py](https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py) 和 [llama-tinygrad/llama_tinygrad.ipynb](https://github.com/MankaranSingh/llama-tinygrad/blob/main/llama_tinygrad.ipynb)。
   - 开发者的目标是在 **OpenCL** 上获得接近 **torch.compile** 的结果，目前正在开发一个最小可复现示例以解决早期遇到的问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ 考虑升级及 r7b 的反应**：社区成员在遇到细微的使用问题后，探讨了 **Command R+** 的未来改进，并参考了对 **r7b** 的初步测试。
   - 针对 r7b 与 **Command R** 相比的性能表现出现了质疑，促使人们呼吁在官方 [changelog](https://docs.cohere.com/v1/changelog/command-r-is-a-scalable-llm-for-business) 中提供更多细节。
- **图像嵌入限制之谜**：用户对 **image embed limits**（每分钟 40 个 vs 预期的 400 个）表示困惑，并参考了生产密钥的使用和 [Cohere 的定价文档](https://cohere.com/pricing)。
   - 团队承认了这种不匹配并承诺修复，尽管假期时间可能会推迟恢复 **400** 个嵌入限制。
- **CIMS 提升 Companion 的审核能力**：**Content Identification/Moderation System (CIMS)** 已推送到 **Companion**，实现了有害内容的自动检测和管理。
   - 正如 [Companion wiki](https://github.com/rapmd73/Companion/wiki) 中详述的那样，它支持直接删除被标记的文本，以促进更安全的交互。
- **Command R 展示大规模 RAG 能力**：**Command R** 支持高达 **128,000 tokens** 的上下文和跨语言任务，为高级 [multi-step tool use](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale) 提供动力。
   - **Command R+** 变体通过更强的 **complex RAG** 性能增强了这些能力，助力以业务为中心的解决方案。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Orion 与 OpenAI：迟到的双人组**：成员们参考 [Hacker News 条目](https://news.ycombinator.com/item?id=42485938) 讨论了 **Orion** 的延迟，重点关注其对未来项目的潜在影响。
   - 他们还注意到影响 OpenAI 服务的新故障，让人想起 2023 年 1 月那段不稳定的可靠性时期。
- **Deepseek 价格亲民**：该小组强调了 **Deepseek** 从 2 月份开始的定价为 **$0.27/MM in** 和 **$1.10/MM out**，认为其性能表现合理。
   - 然而，他们提到虽然它在简单任务上表现出色，但在处理复杂请求的 post-training reasoning 方面仍有困难。
- **Illuminate：类似 NotebookLM 的实验**：几位参与者尝试了 **Illuminate**，参考了 [其官方网站](https://illuminate.google.com/home?pli=1)，将其描述为分析技术论文的工具。
   - 评价褒贬不一，指出不同的开发团队导致其与现有的其他解决方案存在差异。
- **Frontier vs Foundation：流行语之战**：关于 **Frontier** 与 **Foundation** 模型的讨论强调，“Frontier”暗示着随着新版本的发布而具备的尖端性能。
   - 成员们承认 “Foundation” 指的是较早期的努力，而 “Frontier” 虽然定义模糊，但目前非常流行。
- **纽约峰会与日历：期待 2025 年 4 月**：组织者在 **The Times Center** 推广了 2025 年 4 月的 **AI Engineer Summit NYC**，并在 [lu.ma](https://lu.ma/ls) 上分享了更新。
   - 他们邀请通过 RSS 订阅以追踪活动，强调了“添加 iCal 订阅”，并确认目前没有待定活动。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaParse 展现报告 Agent 的魔力**：一段新视频展示了如何使用 **LlamaParse** 和 **LlamaCloud** 构建 Agent 工作流，从 PDF 研究论文中生成格式化报告，详见[此链接](https://twitter.com/llama_index/status/1872322023151833335)。
   - 社区成员赞扬了使用输入模板的方法取得的成功，并重点介绍了 **LlamaCloud** 在处理大型 PDF 文件方面的能力。
- **百万级 PDF RAG 聊天**：一个详细的推文展示了对话式语音助手如何通过 **LlamaCloud** 将 RAG 与 **100万+ PDF** 集成，演示见[此链接](https://twitter.com/llama_index/status/1872684854703432137)。
   - 用户注意到交互体验有所提升，这归功于该流水线的高容量文档处理能力，能够应对更复杂的查询。
- **LlamaIndex 文档与路线图大修**：一位成员为 RAG 应用请求了 **LlamaIndex** 文档的 PDF 版本，并确认可以按需生成。
   - 其他人指出 GitHub 上置顶的路线图已过时（源自 2024 年初），呼吁进行官方修订。
- **Ollama vs. Llama3.2 Vision 测试**：成员们在 **Ollama** 中运行 **非量化 (non-quantized)** 模型进行 RAG 时遇到困难，发现其对非量化模型的支持有限。
   - 他们转而使用 **Llama3.2 11B vision** 进行表格提取，并报告称由于不同的图像处理方式，取得了更好的效果。
- **来自 IBM 的 Docling 加入**：IBM 的 **Docling** 作为一个用于为 AI 准备文档的开源系统亮相，通过[此 YouTube 视频](https://youtu.be/w-Ru0VL6IT8)介绍。
   - 该资源被分享给寻求更有效构建数据结构的 **LlamaIndex** 用户，作为一种可能的增强方案。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Flex 应对 Graph Breaks 与嵌套编译混乱**：成员们解决了 **flex** 可能导致的 **graph breaks**，并提到需要在 [attention_utils.py](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention_utils.py#L27-L31) 中进行更多测试。他们警告说，如果编译处理不当，性能提升可能会消失。
   - 其他人提出了 **嵌套编译 (nested compile)** 障碍和 **dynamo 错误**，强调了当 flex 嵌套在另一个 compile 内部时存在的稳定性风险。
- **DeepSeek V3 在 2 个月内完成 600B MoE 训练**：如 [DeepSeek V3 论文](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)所述，**DeepSeek V3** 在仅 **2000 个 GPU** 上用 **2 个月** 运行了一个 **600B+ MoE**。他们的方法跳过了张量并行（tensor parallelism），但仍保持了速度。
   - 成员们对这种大规模方法很感兴趣，注意到流水线（pipeline）和全对全（all-to-all）配置有助于管理数据吞吐量。
- **H800 GPU：阉割版的 H100**：许多人指出 **H800 GPU** 本质上是 NVLink 较弱的 **H100**，导致带宽较低。他们还发现了 **FP64 性能** 的差异，引发了关于硬件限制下替代方案的讨论。
   - 一位成员建议，这些限制可能会促使人们重新思考分布式训练设置。
- **FP8 训练激发新尝试**：受 **DeepSeek** 的 **FP8** 方法启发，有人计划使用 torchao 框架将 **FP8 训练** 集成到 nanoGPT 中。他们强调需要精确的全对全（all-to-all）操作来挖掘 NVLink 的容量。
   - 这引发了关于如何在降低精度的同时保持模型稳定收敛的讨论。
- **Triton vs. CUDA：GPU 大对决**：一场关于在 **Triton** 还是 **纯 CUDA** 中编写量化代码的辩论正在进行，旨在平衡易用性与速度。有人提到了 Triton 中的 **SM90** 限制，暗示 **cutlass** 对于高性能 GEMM 可能至关重要。
   - 他们正在仔细权衡性能折衷，试图在不牺牲原始吞吐量的情况下保持代码整洁。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **术语表脚本获得关注**：一位成员分享了一个[从 Jekyll 文章生成术语表的脚本](https://gist.github.com/dbreunig/3cef9293cb253f9192d5b4974c1367a3)，使用 **DSPy** 处理 LLM 解析并转换为 **Pydantic** 对象。
   - 他们提到该脚本会将 YAML 文件导出到 `_data` 目录，并赞扬了其自动收集术语的范围。
- **TypedDict 引发热烈讨论**：**TypedDict** 引入了另一种定义字段的方法，引发了关于 **Pydantic** 处理嵌套数组的讨论。
   - 一位参与者强调了处理多个输出字段的难题，但小组对其中的可能性很感兴趣。
- **Pydantic 模型改进 Prompt Schema**：成员们强调使用 **pydantic.BaseModel** 来构建结构化的 Prompt 输出，并确认子字段描述可以正确传播。
   - 成员承诺提供一个修订后的 gist 示例，以更清晰地演示这些方法，这反映了小组对最佳实践的共识。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 周边魔法**：一位身处偏远地区的社区用户庆祝收到了 **Mojo** 周边，分享了[图片](https://cdn.discordapp.com/attachments/1098713601386233997/1321785837490409473/20241226_162214.jpg)并确认即使在偏远地区配送也非常顺畅。
   - 他们称赞了 **T恤** 的质量，并形容某款贴纸非常“硬核”，预测它肯定会在粉丝中“大火”。
- **Traits 成为关注焦点**：一名成员指出了 **Copyable** 和 **ExplicitlyCopyable** traits 的潜在问题，引用了一篇呼吁重新思考其设计的[论坛帖子](https://forum.modular.com/t/the-traits-for-copyability-need-revisiting/380)。
   - 社区建议旨在优化这些 traits 以获得更好的使用体验，并在同一个[论坛主题](https://forum.modular.com/t/the-traits-for-copyability-need-revisiting/380)中公开征集反馈。
- **MAX 更进一步**：**MAX** 集成了来自 XLA 的 kernel fusion 和内存规划，同时增加了动态形状支持、用户定义算子（operators）以及专门的推理服务库。
   - 爱好者们因其扩展的能力将其称为“XLA 2.0”，强调了其针对高级工作负载的自定义 kernel 方法。
- **Mojo vs Python 对决**：关于是构建一致的 **Mojo** API 还是加倍投入 **Python** 集成的争论仍在继续，一些人为了方便而转回使用 JAX。
   - 一位用户提到某些编译器优化必须手动覆盖，强调了与典型的 Python 框架相比，在 **Mojo** 中需要更直接的控制。
- **Endia 与 Basalt 的忧虑**：几位参与者表达了对即将发布的 **Endia** 的期待，并对停滞不前的 **Basalt** 项目表示担忧。
   - 他们表示暂时暂停了 **Mojo** 的开发，在等待明确进展的同时，仍鼓励社区在 Endia 上进行协作。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书还是放弃：声明的困境**：如果没有关键的**证书声明表单**，学习者将无法获得证书，该表单是已完成评估的官方注册凭证。
   - 课程工作人员将其称为他们的“花名册”，并强调其对于最终审批至关重要。
- **一月的冲击：下一期 MOOC 即将到来**：**1 月下旬**被定为下一期 [LLM Agents MOOC](https://llmagents-learning.org/f24) 的开始日期，为错过当前课程的参与者提供了加入机会。
   - 参与者注意到了这个时间点，希望在新的一年初扩展他们在 **Large Language Model** 方面的专业知识。
- **测验限制：表单已锁定**：[Quiz 5 - Compound AI Systems](https://forms.gle/tXzmfgTsdYW5XjLL6) 链接目前已关闭，停止接收额外的测验提交。
   - 许多人请求重新开放，强调了这些测验对于结构化练习的重要性。
- **高级 LLM Agents：更深层次的策略**：即将推出的 **Advanced LLM Agents** 课程承诺涵盖详细的 Agent 设计，包括高级优化方法。
   - 爱好者们认为这是完成基础语言模型课程后的逻辑延伸。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Claude 3.5 Opus 引发与 O1 的竞争**：**Claude 3.5 Opus** 的潜力令人兴奋，因为它拥有更强的推理能力。
   - 许多人好奇它是否能超越 **O1** 和 **O1 Pro**，预示着一场激烈的模型竞争。
- **Open-Interpreter QvQ 势头强劲**：一位用户询问 **QvQ** 在 **OS mode** 下接入 **Open-Interpreter** 时如何运行，表现出对直接系统交互的兴趣。
   - 该问题仍悬而未决，标志着社区进一步探索的一个方向。
- **生成式音频协作招募**：一位 AI 工程师分享了在 **DNN-VAD**、**NLP** 和 **ASR** 方面的进展，包括最近的一个 **Voice to Voice** 聊天应用项目。
   - 他们邀请其他人加入，暗示了在生成式 AI 音乐生成方面可能存在的协同效应。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **复制按钮难题 (Copy-Button Conundrum)**：一位用户指出聊天 UI 中缺少代码的 **copy** 按钮，另一位用户确认基于鼠标的剪切粘贴功能无法正常工作。
   - 然而，**Control-C** 和 **Control-V** 仍是社区提到的主要变通方法。
- **WASM 疑问 (WASM Wondering)**：一位新人询问关于将 AI 作为 **WASM package** 安装的问题，引起了对可能部署方法的关注。
   - 目前没有直接回应，该查询仍有待未来探索。
- **Vulcan 版本空白 (Vulcan Version Void)**：一位成员多次询问关于 **Vulcan 版本** 的信息，但未获得任何澄清或细节。
   - 对于熟悉 Vulcan 细节的人来说，这个问题仍未得到解答。
- **鼠标与键盘怪癖 (Mouse & Keyboard Quirks)**：参与者注意到基于鼠标的剪切粘贴在配置页面上失效。
   - 他们强调 **Control-C** 和 **Control-V** 是复制代码或文本的推荐方法。
- **新模板尝试 (New Template Trials)**：一位成员询问是否有人尝试过使用 **新模板** 进行写作，暗示了一种新的内容创作方法。
   - 讨论显示出对切换到新模板的兴趣，但关于实际使用情况的细节较少。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL 排行榜上的缩放调整 (Scaling Shuffle on BFCL Leaderboard)**：在关于 [Gorilla LLM 排行榜](https://discord.com/channels/1111172801899012102/1214705495974092810/1321692430797639792) 的 **inference scaling**（推理缩放）和 **post-training**（后训练）方法的问题中，一位成员询问 BFCL 是否允许通过重复输出选择来增强的多次调用模型。
   - 他们解释说，**post-inference verification**（推理后验证）可以多次调用工具增强型 LLM 以获得更精炼的结果，并强调了潜在的性能提升。
- **公平性之争：单次调用 vs 多次调用 (Fairness Feuds: Single-Call vs Multi-Call)**：同一位用户担心多次调用扩展可能会盖过更简单的单次调用 LLM，称其为排行榜上的“不公平竞争”。
   - 他们建议将 **inference latency**（推理延迟）作为额外调用的直接权衡因素纳入排名，希望社区能接受这种方法。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Whisper 巧妙的文字处理 (Whisper's Witty Word Wrangling)**：一位用户描述了 **Whisper** 如何检测句子边界，从而为语音处理实现更准确的切分。
   - 他们表示，利用这些检测可以提高清晰度，让开发者在语音任务中加入 **句子级分解 (sentence-level breakdown)**。
- **VAD 的静音切分魔术 (VAD's Silence Splitting Sorcery)**：另一位用户推荐使用 **voice activity detector (VAD)** 来将语音与静音分离，以实现稳健的音频分割。
   - 这种方法利用 **静音检测 (silence detection)** 来优化分割过程并提高效率。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **针对 HPC 的 MLOps 解决方案 (MLOps Solutions for HPC)**：一位成员寻求适用于 HPC 且跳过 SaaS 依赖的 MLOps 框架，并指出 HPC 强大的存储能力是一项主要优势。
   - 他们强调了对稳定解决方案的需求，并评估了 **Guild AI** 在 HPC 环境下的可靠性。
- **Guild AI 的成长烦恼 (Guild AI Growing Pains)**：同一位用户对 **Guild AI** 的稳定性表示担忧，担心在 HPC 环境中可能出现停机。
   - 他们寻求关于 HPC 部署的具体反馈，以确认 **Guild AI** 是否已准备好应对大规模训练任务。
- **低成本的 DIY Ops (DIY Ops on a Shoestring)**：他们还考虑自行构建一个极简的运维框架，认为这比安装基于服务器的解决方案更简单。
   - 他们认为自定义方法可能会减少开销，同时也承认了维护自有工具集的风险。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1321576722818138273)** (1129 条消息🔥🔥🔥): 

> `Cursor IDE 功能、DeepSeek V3 性能、Claude Sonnet 对比、AI 工具中的上下文管理、模型效率与成本` 


- **Cursor IDE 功能评测**：用户讨论了 Cursor IDE 上下文管理的局限性与功能，特别是关于请求缓慢以及在对话中处理多个文件的问题。
   - 共识是，虽然该工具很有效，但用户经常对交互过程中上下文的处理方式感到沮丧。
- **关于 DeepSeek V3 性能的辩论**：DeepSeek V3 与 Claude Sonnet 进行了对比，用户对其有效处理编程任务的能力持不同意见。
   - 观点各异，一些用户报告了 DeepSeek 的性能问题，特别是与 Claude Sonnet 卓越的编程能力相比。
- **探索有效的上下文管理**：用户强调了清晰简洁的 Prompt 对高效管理上下文的重要性，这被视为工具功能的关键。
   - 策略包括明确指定更改，并为每个任务使用专用会话以最大化性能。
- **AI 工具易用性反馈**：用户呼吁改进 Cursor 内部的上下文管理，并建议增加快捷键等功能以加速流程。
   - 虽然 AI 工具的性能得到了认可，但用户希望有更流畅、更直观的工作流，以减轻收集上下文的负担。
- **AI 模型定价的用户体验**：对话涉及了 AI 工具的定价模式，重点在于平衡成本与各种模型的性能。
   - 用户反思了使用 Claude Sonnet 等工具与替代方案的价值主张，表明希望定价更加透明。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/advanced/models#what-context-window-is-used-for-model-x">Cursor - 更快地构建软件</a>：未找到描述</li><li><a href="https://docs.cursor.com/context/@-symbols/@-docs">Cursor - 更快地构建软件</a>：未找到描述</li><li><a href="https://www.anthropic.com/news/prompt-generator">在开发者控制台中生成更好的 Prompt</a>：你现在可以在 Anthropic Console 中生成生产就绪的 Prompt。描述你想要实现的目标，Claude 将使用思维链（chain-of-thought）推理等 Prompt Engineering 技术来创建一个...</li><li><a href="https://x.com/code/status/1872673862992744625">来自 Visual Studio Code (@code) 的推文</a>：Claude 3.5 Sonnet，直接集成在 @code 中。今天起对所有人开放，包含 GitHub Copilot Free。了解更多：http://aka.ms/copilot-free</li><li><a href="https://platform.deepseek.com">DeepSeek 平台</a>：加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721?t=vpoi2yGx6psx69xwLTKnxA&s=19">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 每秒 60 个 tokens（比 V2 快 3 倍！）💪 能力增强 🛠 API 保持兼容 🌍 完全开源模型和论文 🐋 1/n</li><li><a href="https://tenor.com/view/spider-man-spider-man-web-of-shadows-depressed-sad-gif-16524395">Spider Man Spider Man Web Of Shadows GIF - Spider Man Spider Man Web Of Shadows Depressed - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mammon-policjant-kamil-p%C5%82ock-mammon-kamil-gif-9553907799842793042">Mammon Policjant GIF - Mammon Policjant Kamil - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=gSQ77cGYqXY&ab_channel=TempoLabs"> - YouTube</a>：未找到描述</li><li><a href="https://forum.cursor.com">Cursor - 社区论坛</a>：讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://tally.so/r/w5ERBb">开发者快速调查：AI 与工作流
</a>：使用 Tally 制作，最简单的表单创建方式。</li><li><a href="https://icon-sets.iconify.design/?query=box">Iconify - 开源图标之家</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1321935220630622258)** (1 条消息): 

> `Windsurf 创新, Windsurf 幕后故事, 节日祝福` 


- **工程师揭秘 Windsurf 的诞生**：[Windsurf](https://x.com/windsurf_ai/status/1872375661542920424) 发布的一段新视频展示了工程师们关于如何构建 Windsurf 的见解，强调了他们的创新方法。
   - 文中还提到了“节日快乐”，展现了团队在节日期间的精神面貌。
- **勇于通过 Windsurf 进行创新**：视频强调了 Windsurf 如何打破行业惯例，展示了团队的创造性思维和技术实力。
   - 视频中的标语鼓励观众深入了解他们如何在这一领域“勇于创新”。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1872375661542920424">Windsurf (@windsurf_ai) 的推文</a>：Windsurf 到底是什么？观看我们如何通过打破每一项行业惯例来勇于创新 🌊

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1321594246553141258)** (202 条消息🔥🔥): 

> `Windsurf 性能问题, Codeium Pro 计划挫败感, IDE 集成问题, Macbook M1 终端问题, Cascade 中的全局规则` 


- **Windsurf 性能问题阻碍生产力**：多位用户报告了 Windsurf 的显著性能问题，包括运行期间的错误和系统挂起，导致了对额度的担忧。
   - 用户表达了挫败感，尤其是 Pro 计划的用户，因为这些问题干扰了他们的编码项目并阻碍了预期功能的实现。
- **关于 Codeium Pro 计划限制的困惑**：用户对 Codeium Pro 计划的有限使用提出了担忧，特别是关于每月请求数量及其与其他模型相比的有效性。
   - 一些用户在升级后感到失望，因为持续的技术困难影响了他们的生产力。
- **WebStorm 及其他 IDE 的集成问题**：用户报告 Codeium 插件在 WebStorm 和 IntelliJ 等其他 IDE 中无法正常显示，破坏了功能。
   - 问题包括最近更新后出现的功能缺失和错误，引发了困惑和对潜在修复方案的咨询。
- **Macbook M1 终端运行架构错误**：一位 Macbook M1 Pro 用户报告称，尽管应用显示是为 Apple Silicon 构建的，但 Windsurf 的终端却在 i386 架构下运行。
   - 他们寻求进一步调试此问题的建议，以恢复正常功能。
- **对在 Cascade 中定义全局规则的兴趣**：讨论围绕在 Cascade 中设置全局规则的潜在好处展开，以简化编码并减少错误。
   - 用户正在寻求关于哪些规则有助于提高编码项目的整体生产力和准确性的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/getstarted/overview">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/thanks-for-the-assist-oscar-finlay-reacher-thank-you-for-helping-i-appreciate-you-helping-gif-24767872">Thanks For The Assist Oscar Finlay GIF - Thanks For The Assist Oscar Finlay Reacher - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/careers">Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://dev.wix.com/docs/velo">Velo Docs 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1321572800288522285)** (557 条消息🔥🔥🔥): 

> `Windsurf 反馈, DeepSeek V3, 额度系统, 用户体验, AI 工具对比`

- **用户对 Windsurf 的不满**：用户对 Windsurf 的额度系统（credit system）表示不满，特别是无限用户提示词额度（user prompt credits）却需要消耗 Flow Action 额度，导致对预期功能的困惑。
   - 一些用户报告称，他们感到受限于当前的定价方案，并且在使用 IDE 时需要更好的上下文管理（context management）。
- **DeepSeek V3 的潜力**：DeepSeek V3 被讨论为一种高效的代码编写模型，许多用户希望将其集成到 Windsurf 中，以提高性能并降低成本。
   - 目前，用户正在探索使用 DeepSeek 的方法，一些人正在考虑像 Cursor 这样允许使用自定义 API key 的替代方案。
- **AI 工具的改进**：用户分享了将任务转移到 Windsurf 和 Cursor 等 AI 工具的经验，并指出有效的测试实践可以带来更高的生产力。
   - 大家达成共识，虽然 Windsurf 对某些人有效，但要发挥其在编程项目中的最大潜力，还需要一定的学习曲线。
- **AI 模型比较**：关于 Claude 在 Windsurf 中的表现与其在 Web 端表现的对比反馈显示，结果存在差异和不一致性。
   - 用户讨论了他们对各种工具的偏好，强调在工作流中需要更好的自动补全（auto-completion）和上下文管理（context management）。
- **社区参与**：社区成员参与了关于 AI 开发工具的讨论，包括分享增强 Windsurf 使用体验的个性化技巧。
   - 对话强调了社区对公开交流以及根据用户反馈对产品进行迭代改进的渴望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/itsPaulAi/status/1872320003770618146">来自 Paul Couvert (@itsPaulAi) 的推文</a>：等等，所以我们现在有了一个比 GPT-4o 更好的 100% 开源模型？！根据多个基准测试，DeepSeek v3 在代码方面甚至优于 Claude Sonnet 3.5。已经可以免费提供给...</li><li><a href="https://tenor.com/view/the-chicken-came-first-the-chicken-or-the-egg-sbs-surprised-brain-syndrome-nathan-barnatt-gif-20524155">先有鸡还是先有蛋 GIF - 先有鸡还是先有蛋 Sbs - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.openai.com/">OpenAI Status</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-flow-action-credits-but-not-premium-user-prompt-credits">付费计划与额度使用 - Codeium 文档</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage">付费计划与额度使用 - Codeium 文档</a>：未找到描述</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 60 tokens/second（比 V2 快 3 倍！）💪 增强的能力 🛠 保持 API 兼容性 🌍 完全开源的模型和论文 🐋 1/n</li><li><a href="https://codeium.com/context">Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://tenor.com/view/giggle-chuckle-hahaha-holding-laughter-gif-15462551">咯咯笑 GIF - 咯咯笑 哈哈哈 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/bilawalsidhu/status/1872357449359098170">来自 Bilawal Sidhu (@bilawalsidhu) 的推文</a>：X 正在爆发一场竞赛战。与此同时，中国刚刚发布了一个绝对领先（MOGS）的开源 AI 模型，其使用的算力仅为美国实验室消耗的一小部分。- 美国：“限制中国的芯片...”</li><li><a href="https://codeium.canny.io/feature-requests/p/add-deepseek-v3">添加 DeepSeek v3 | 功能请求 | Codeium</a>：它几乎刚刚发布：1) https://kagi.com/fastgpt?query=Tell+me+about+the+AI+model+%22DeepSeek+V3%22 2) https://www.perplexity.</li><li><a href="https://x.com/nrehiew_/status/1872318161883959485">来自 wh (@nrehiew_) 的推文</a>：如何训练一个 670B 参数的模型。让我们聊聊 DeepSeek v3 报告，以及与 Meta 在 Llama 405B 上所做工作的对比</li><li><a href="https://artificialanalysis.ai/models/deepseek-v3">DeepSeek V3 - 质量、性能和价格分析 | Artificial Analysis</a>：对 DeepSeek 的 DeepSeek V3 进行分析，并在质量、价格、性能（tokens per second 和 time to first token）、上下文窗口等关键指标上与其他 AI 模型进行对比...</li><li><a href="https://www.youtube.com/watch?v=lGI6CR-O44g"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Vst8GNnFPJk">中国用户名揭示了关于《战争雷霆》（War Thunder）的什么</a>：我在《战争雷霆》中打了 100 场顶级 Air RB。在这个视频中，我讨论了遇到的中国用户名，翻译了它们，并谈到了亲...</li><li><a href="https://github.com/nascarjake/luminary">GitHub - nascarjake/luminary：支持 OpenAI 的 AI Pipeline 构建器</a>：支持 OpenAI 的 AI Pipeline 构建器。通过创建一个账号来为 nascarjake/luminary 的开发做出贡献。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3">GitHub - deepseek-ai/DeepSeek-V3</a>：通过创建一个账号来为 deepseek-ai/DeepSeek-V3 的开发做出贡献。</li><li><a href="https://texzcorp.github.io/ObjectiveVisualizer/">音乐可视化工具</a>：未找到描述</li><li><a href="https://github.com/Texzcorp/ObjectiveVisualizer">GitHub - Texzcorp/ObjectiveVisualizer</a>：通过创建一个账号来为 Texzcorp/ObjectiveVisualizer 的开发做出贡献。
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1321981397451280508)** (1 条消息): 

> `Aider v0.70.0 发布，支持 o1 模型，分析功能加入（opt-in），错误处理改进，新增安装方法` 


- **Aider v0.70.0 发布，带来令人兴奋的新特性**：[Aider v0.70.0](https://aider.chat/HISTORY.html) 的发布包括对 **o1 模型的全方位支持**，以及通过 uv 提供的全新安装方式，使安装比以往任何时候都更加便捷。
   - 此版本引入了对文件监听（watch files）和错误处理的改进，提升了性能和用户体验。
- **分析功能加入以增强特性**：在一项重大举措中，Aider 将**询问 10%** 的用户是否加入分析功能，旨在收集数据以进一步改进工具。
   - 这一决定反映了对理解用户交互模式和优化平台的承诺。
- **改进的错误处理机制**：最新的更新在用户尝试使用 `/load` 或 `--load` 进行交互式命令时提供了**更好的错误处理**，使用户交互更加顺畅。
   - 此外，它现在可以优雅地处理 git 路径名中的 unicode 错误，减少了潜在的中断。
- **简化只读文件的显示**：如果绝对路径比相对路径短，Aider 现在会以**绝对路径**显示**只读文件**，从而提高文件管理的清晰度。
   - 这一微小的调整有助于用户快速识别文件位置，避免混淆。
- **新增对多种模型的支持**：Aider v0.70.0 更新还提供了对 **openrouter**、**deepseek** 和 **deepseek-chat 模型**的支持，扩大了其可用性。
   - 这一扩展反映了 Aider 持续集成 AI 领域中各种工具和模型的使命。



**提到的链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1321568721822679182)** (534 条消息🔥🔥🔥): 

> `DeepSeek V3 性能，AI 编程工具与策略，Aider 集成，面向 LLM 的 Svelte 文档，开发者技能与学习` 


- **DeepSeek V3 在本地设置上的运行性能**：使用 8 台 M4 Pro Mac Mini 组成的集群，拥有 671B 参数的 DeepSeek V3 模型达到了 **每秒 5.37 个 token** 的速度，首个 token 生成时间为 **2.91 秒**，表现优于较小的模型。
   - 尽管体积庞大，DeepSeek V3 的处理效率显示了其针对 Apple Silicon 的优化，引发了人们对其部署能力的兴趣。
- **Aider 与其他工具的集成**：用户讨论了将 Aider 与各种开发工具集成，表达了通过高效的编码辅助实现工作流自动化和提高生产力的兴趣。
   - 技术爱好者分享了使用 Aider 解析和执行 GitHub issues 的示例，强调了在软件开发中有效的模型交互的重要性。
- **LLM 语境下的 Svelte 文档**：分享了一个以 LLM 友好格式提供 Svelte 5 和 SvelteKit 文档的网站，突出了对于希望利用 AI 助手高效编码的开发者的价值。
   - 参与者注意到了现有 LLM 在处理大量文档时的局限性，建议需要能够管理更大上下文窗口（context windows）的模型。
- **培养编程能力**：讨论强调了关于如何培养优秀开发者的观点，认为虽然某些课程可能会提升技能，但基础能力至关重要。
   - 参与者强调，成为一名精通的开发者不仅仅需要编码技能；它还涉及解决问题的思维方式以及利用 AI 工具的适应能力。
- **对 Claude AI 和 DeepSeek 的看法**：用户表达了对 Claude AI 的不同体验，一些人注意到在最近的发展后性能有所下降，这与 DeepSeek V3 新兴的有效性形成对比。
   - 人们对 AI 工具需要改进性能和功能集以满足开发者不断变化的需求表示了担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/deepseek_ai/status/1872242657348710721">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 DeepSeek-V3 正式发布！迄今为止最大的飞跃：⚡ 60 tokens/秒（比 V2 快 3 倍！）💪 增强的能力🛠 API 兼容性保持不变🌍 完全开源的模型和论文🐋 1/n</li><li><a href="https://dearrow.ajay.app/">DeArrow - 一个用于优化标题和缩略图的浏览器扩展</a>: DeArrow 是一款浏览器扩展，用于将 YouTube 上的标题和缩略图替换为社区创建的准确版本。告别标题党。</li><li><a href="https://blog.exolabs.net/day-2/">EXO 的 12 天</a>: 12 天真正的开放创新</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的量化基准测试。</li><li><a href="https://agenticengineer.com/principled-ai-coding">Agentic Engineer - 构建“活的”软件</a>: 构建“活的”软件。你掌握 prompts, prompt chains, AI agents 和 agentic workflows 的指南。</li><li><a href="https://tenor.com/view/im-the-captain-now-im-the-boss-captain-gif-14172461">我现在是船长，我是老板 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/faq.html#why-is-the-llm-speaking-to-me-in-an-unexpected-language">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://x.com/alexocheema/status/1872447153366569110">来自 Alex Cheema - e/acc (@alexocheema) 的推文</a>: 必须堆叠 8 台 Mac Minis 才能运行它。目前约为 5 tok/sec。第一次在 8 台 Mac Minis 上运行 inference —— 性能还有很大提升空间（该配置下的理论极限 >10 tok/sec）。Quo...</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>: 让 aider 在处理你的代码时遵循你的编码规范。</li><li><a href="https://bsky.app/profile/gary.info/post/3leatxn2exs2p">Bluesky</a>: 未找到描述</li><li><a href="https://tenor.com/view/think-about-it-you-know-what-i-mean-think-gif-15115330">考虑一下，你知道我的意思 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard">BigCodeBench 排行榜 - bigcode 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/genius-think-be-clever-be-smart-gif-10617231">天才思考 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://agenticengineer.com/principled-ai-coding/">Agentic Engineer - 构建“活的”软件</a>: 构建“活的”软件。你掌握 prompts, prompt chains, AI agents 和 agentic workflows 的指南。</li><li><a href="https://x.com/i/status/1815969489990869369">来自 Alex Cheema - e/acc (@alexocheema) 的推文</a>: 你只需要 2 台 MacBooks。使用 @exolabs_ 家庭 AI 集群在 2 台 MacBooks 上分布式运行 Llama 3.1 405B</li><li><a href="https://x.com/ivanfioravanti/status/1870926281736659413">来自 Ivan Fioravanti ᯅ (@ivanfioravanti) 的推文</a>: 通过 @exolabs 的 exo 在 M2 Ultra 和 2 台 M4 Max 之间建立 Thunderbolt 连接。让我们用 llama 3.2 405B 做一些测试！</li><li><a href="https://youtu.be/SkmrUWyZThQ?si=GpGqzOHydrfhQr4v"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qqXkGqzsFio"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/2eNVV0ouBxg"> - YouTube</a>: 未找到描述</li><li><a href="https://artificialanalysis.ai/models/deepseek-v3">DeepSeek V3 - 质量、性能与价格分析 | Artificial Analysis</a>: 对 DeepSeek 的 DeepSeek V3 进行分析，并从质量、价格、性能（tokens 每秒和首个 token 时间）、上下文窗口等关键指标与其他 AI 模型进行对比。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hm4959/benchmark_results_deepseek_v3_on_livebench/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://svelte-llm.khromov.se/">svelte-llm - LLM 就绪格式的 Svelte 5 和 SvelteKit 开发者文档</a>: 未找到描述</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>: 未找到描述</li><li><a href="https://youtu.be/GBR6pHZ68Ho"> - YouTube</a>: 未找到描述</li><li><a href="https://www.apple.com/shop/buy-mac/mac-mini/apple-m4-pro-chip-with-12-core-cpu-16-core-gpu-24gb-memory-512gb?afid=p238%257CsyAHmzAxH-dc_mtid_1870765e38482_pcrid_724099485254_pgrid_110391416539_pntwk_g_pchan__pexid__ptid_kwd-865769501_&cid=aos-us-kwgo-mac--slid---product-">Mac mini</a>: 搭载 M4 和 M4 Pro 芯片的 Mac mini。专为 Apple Intelligence 打造。配备前后端口。提供分期付款选项。立即从 apple.com 购买。</li>

i><a href="https://www.apple.com/shop/buy-mac/mac-mini/apple-m4-pro-chip-with-12-core-cpu-16-core-gpu-24gb-memory-512gb?afid=p238%7CsyAHmzAxH-dc_mtid_1870765e38482_pcrid_724099485254_pgrid_110391416539_pntwk_g_pchan__pexid__ptid_kwd-865769501_&cid=aos-us-kwgo-mac--slid---product-">Mac mini</a>: 搭载 M4 和 M4 Pro 芯片的 Mac mini。专为 Apple Intelligence 打造。配备前后端口。提供分期付款选项。立即从 apple.com 购买。</li><li><a href="https://github.com/richardanaya/colossus/">GitHub - richardanaya/colossus: 用于控制 aider 的实时语音 AI 工具</a>: 一款用于控制 aider 的实时语音 AI 工具。通过在 GitHub 上创建账号来为 richardanaya/colossus 的开发做出贡献。</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: 代码</a>: 代码。通过在 GitHub 上创建账号来为 robert-at-pretension-io/mcp 的开发做出贡献。</li><li><a href="https://github.com/nekowasabi/aider.vim">GitHub - nekowasabi/aider.vim: 在 neovim 中辅助 aider</a>: 在 neovim 中辅助 aider。通过在 GitHub 上创建账号来为 nekowasabi/aider.vim 的开发做出贡献。</li><li><a href="https://github.com/BuilderIO/micro-agent?tab=readme-ov-file">GitHub - BuilderIO/micro-agent: 一个为你编写（真正有用的）代码的 AI Agent</a>: 一个为你编写（真正有用的）代码的 AI Agent - BuilderIO/micro-agent</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: 使用日常设备在家里运行你自己的 AI 集群 📱💻 🖥️⌚</a>: 使用日常设备在家里运行你自己的 AI 集群 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.amazon.com/Lenovo-00KG133-Nvidia-Tesla-K80/dp/B01A3VGAGS?crid=1CMGVX3FG8UI9&dib=eyJ2IjoiMSJ9.NQxBWkkc6BLtNRAxRAfQgzvWmExBfvGWMYy24oGZGRc6hwRD_DEa7qj9PHUVGfrGH3TZAIzhSvQ-bEf8VJ6W3n-EgDzpMsFozhLaQBlSWmeTsAQjgX8mv0dUEaIs4FIduiXnQuRTQExQpDQtwRNl4d5wIRp1mw28t2nZX5rf0ED6VlXYUzB-Cg5sUEb0TjqrHlkNXfdvttvt8DA6BZ8w003lvsKOC56wIacHsF2AUc4.whVOarsaA_4hRB5PqAcZ6mC2pdnBQSrgG_9iGaCmT0M&dib_tag=se&keywords=NVIDIA+Tesla+K80+GPU&qid=1735193115&sprefix=nvidia+tesla+k80+gpu,aps,351&sr=8-5">Amazon.com: Nvidia Tesla K80 : 电子产品</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1321578153818198018)** (63 条消息🔥🔥): 

> `DeepSeek V3 性能、Aider 配置、Repo Map 功能、Token 限制讨论、模型合并策略` 


- **DeepSeek V3 表现优于之前的模型**：一位成员指出，**DeepSeek Chat V3** 的速度明显更快，且在成本大幅降低的情况下提供了与 **Sonnet** 相当的性能，这可能会颠覆竞争格局。
   - 他们还评论了 **DeepSeek 定价**模型的有效性，使其成为 **Sonnet** 用户的强力替代方案。
- **使用 .env 和 YAML 配置 Aider**：用户表示，由于多条目限制，Aider 的模型别名配置在 YAML 中效果最好，而非 .env 文件。
   - 建议使用 `--verbose` 等特定命令来调试与模型别名识别相关的问题。
- **Aider 中的 Repo Map 功能**：有成员对 repo-map 功能在 Architect 模式和标准编辑模式下的不同表现提出了担忧，特别是当设置为手动刷新时。
   - 成员们讨论了解决 repo-maps 根据模型使用情况意外刷新所需的潜在配置。
- **管理 DeepSeek 的 Token 限制**：一些用户报告在使用 **DeepSeek Chat** 时遇到了 Token 限制，并强调在升级到 V3 后，其输入 Token 限制更改为 **64k**。
   - 他们讨论了修改 **.aider.model.metadata.json** 文件的可能性，以便在交互过程中更好地管理 Token 限制和成本。
- **LLM 模型组合策略**：成员们考虑了用于架构任务的各种模型组合，例如建议使用 **Gemini 1206** 作为 Architect，配合 **DeepSeek V3** 作为 Coder。
   - 分享了关于不同模型的易用性以及在特定任务（如创建 FFMPEG 预设）中的整体有效性的经验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/whatchu-talkin-about-willis-arnold-jackson-diffrent-strokes-what-are-you-tryi">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/whatchu-talkin-about-willis-arnold-jackson-diffrent-strokes-what-are-you-trying-to-say-willis-what-is-that-willis-gif-26301758">Whatchu Talkin About Willis Arnold Jackson GIF - Whatchu Talkin About Willis Arnold Jackson Diffrent Strokes - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider 可以连接到大多数 LLMs 进行 AI 结对编程。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hm2xvb/deepseek_v3_is_already_up_on_api_and_web/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: 下面列出的价格以每 1M tokens 为单位。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总额计费...</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>: 用于以 OpenAI 格式调用 100+ LLM API 的 Python SDK、代理服务器（LLM 网关） - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1321896128865763493)** (1 messages): 

> `GitDiagram, Gitingest` 


- **使用 GitDiagram 可视化 GitHub 仓库**：[GitDiagram](https://gitdiagram.com/) 工具允许用户将任何 GitHub 仓库转换为**交互式图表**，以便快速进行项目可视化。
   - 用户还可以通过在任何 GitHub URL 中将 'hub' 替换为 'diagram' 来轻松使用此功能。
- **使用 Gitingest 简化代码库摄取**：[Gitingest](https://gitingest.com/) 项目将任何 Git 仓库转换为其代码库的简单文本摄取，使其**对 LLM 提示词友好**。
   - 您可以在任何 GitHub URL 中将 'hub' 替换为 'ingest'，以有效地使用此工具。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gitdiagram.com/">GitDiagram - 秒级仓库转图表</a>：将任何 GitHub 仓库转换为交互式图表进行可视化。</li><li><a href="https://gitingest.com/">Git ingest</a>：在任何 Github Url 中将 'hub' 替换为 'ingest' 以获得提示词友好的文本
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1321571272647245888)** (9 messages🔥): 

> `Hugging Face Trainer Modification, Pythia Intermediate Checkpoints, Machine Learning Research Interest` 


- **寻求 Hugging Face Trainer 自定义损失函数的指导**：一位成员在自定义 Hugging Face Trainer 中因果语言建模（causal language modeling）的损失函数时寻求帮助，特别是关于处理标签以忽略填充标记（padded tokens）和输入标记的问题。
   - 另一位参与者建议使用不同的 collator 将 prompt 标签设置为 `ignore_idx`，并指出 trl 库可能有用的参考信息。
- **请求 Pythia 优化器状态**：一位用户询问如何获取 Pythia 中间检查点（checkpoints）的优化器状态（optimizer states），因为目前仅提供了最终检查点的状态，而由于大小限制，需要中间状态的用户被引导手动联系工作人员。
   - 他们请求协助标记 Pythia 工作人员，表达了希望在此事上进行更好沟通的愿望。
- **新成员的机器学习抱负**：一位成员介绍自己是即将获得硕士学位的理论物理学家，希望进入 ML 研究领域，重点是深入理解深度学习和 LLM。
   - 他们表达了参与可解释性（interpretability）和 LLM 相关项目的兴趣，展示了学习和贡献的热情。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1321568288756469760)** (278 messages🔥🔥): 

> `Causal Inference in Machine Learning, Intelligence and Learning Models, World Models and Video Generation, Symbolic Representation in AI, Human Learning and Cognition` 


- **探讨 ML 训练中的因果推理**：参与者讨论了因果推理在机器学习中的影响，特别是对训练效率和先验经验相关性的影响。
   - 人们对学习既能遵循数据动态又能提高训练过程效率的表示法（representations）很感兴趣。
- **视频生成模型的局限性**：对话强调视频生成模型无法从视觉数据中学习物理定律，即使增加规模（scaling）增强了它们的能力。
   - 参与者质疑这些模型在没有人类先验的情况下是否能发现真正的物理定律，暗示需要强大的泛化技术。
- **理解人类的抽象学习**：讨论集中在人类提取信息和学习因果关系的方式与模型有何不同，模型通常仅依赖于统计趋势。
   - 有人指出，人类可以通过使用分块（chunking）等技术开发出超出其训练数据的深度表示，正如在盲棋棋手身上观察到的那样。
- **重新审视符号表示**：探讨了符号表示（symbolic representations）的实用性及其属性，质疑是否可以通过网络构建有效地捕捉这些表示。
   - 有人建议抽象和瓶颈（bottlenecks）对于学习至关重要，这引发了探索生成模型中可变状态（mutable state）的想法。
- **模型中记忆与学习的相互作用**：参与者讨论了高效内存使用与模型有效学习抽象表示的必要性之间的平衡。
   - 将信息压缩成易于管理的块（chunks）的概念（类似于专家级棋手回忆对局状态的方式）被强调为模型的一种学习策略。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://blog.dottxt.co/oss-v-gpt4.html">Beating GPT-4 with Open Source</a>: 未找到描述</li><li><a href="https://phyworld.github.io/">How Far is Video Generation from World Model: A Physical Law Perspective</a>: 我们进行了一项系统研究，通过利用数据和模型缩放，探讨视频生成是否能够从视频中学习物理定律。</li><li><a href="https://arxiv.org/abs/2410.13787">Looking Inward: Language Models Can Learn About Themselves by Introspection</a>: 人类通过观察外部世界获取知识，但也通过内省获取。内省使人能够获得对其当前心理状态（例如想法和感受）的特权访问...</li><li><a href="https://arxiv.org/abs/2412.17256">B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners</a>: 在缺乏用于复杂推理任务的大规模人类标注数据的情况下，自我改进（模型在其自身输出上进行训练）已成为增强性能的主要方法...</li><li><a href="https://arxiv.org/abs/2410.02536">Intelligence at the Edge of Chaos</a>: 我们通过研究基于规则的系统的复杂性如何影响为预测这些规则而训练的模型的能力，来探索人工系统中智能行为的出现...</li><li><a href="https://arxiv.org/abs/2403.06963">The pitfalls of next-token prediction</a>: 仅仅一个 next-token 预测器能否忠实地模拟人类智能？我们明确了这一新兴的担忧，并纠正了围绕它的流行误解，并提倡一种简单的 multi-token 目标...</li><li><a href="https://arxiv.org/abs/2305.14325">Improving Factuality and Reasoning in Language Models through Multiagent Debate</a>: 近年来，大语言模型 (LLMs) 在语言生成、理解和 few-shot 学习方面展示了卓越的能力。大量工作探索了它们的性能如何...</li><li><a href="https://arxiv.org/abs/2311.15475">MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers</a>: 我们介绍了 MeshGPT，这是一种生成三角形网格的新方法，它反映了艺术家创建的网格典型的紧凑性，与通过等值面提取方法提取的密集三角形网格形成对比...</li><li><a href="https://modal.com/blog/llama-human-eval">Beat GPT-4o at Python by searching with 100 dumb LLaMAs</a>: 通过搜索和评估扩展较小的开源模型，以匹配前沿模型的能力。</li><li><a href="https://en.m.wikipedia.org/wiki/Where_Mathematics_Comes_From">Where Mathematics Comes From - Wikipedia</a>: 未找到描述</li><li><a href="https://projects.haykranen.nl/markov/demo/">
                     &raquo; Hay Kranen            </a>: 未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>: 通过在 GitHub 上创建账户，为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1321893300755763240)** (1 messages): 

> `Deepseek v3, OpenRouter usage, Model comparisons, Cost of frontier models` 


- **Deepseek v3 使用量翻三倍**：在发布 **Deepseek v3** 后，OpenRouter 上的使用量自昨天以来已经**翻了三倍**，正如 [@OpenRouterAI 的推文](https://x.com/OpenRouterAI/status/1872334128043208833)所述。
   - 根据社区反馈，*Deepseek v3 似乎是一个真正优秀的模型*。
- **Deepseek v3 与主流模型相比具有竞争力**：**Deepseek v3** 的基准测试显示其结果可与 **Sonnet** 和 **GPT-4o** 媲美，但**价格要低得多**。
   - 这为更多用户在不超支的情况下访问先进模型提供了机会。
- **中国和开源力量正在追赶**：一位行业专家评论称，**中国已经赶上**，且**开源模型已匹配**领先 AI 的能力，而前沿模型的成本约为 **600 万美元**。
   - 他们预计 **Deepseek v3** 在未来几天将在 OpenRouter 上表现出色。
- **对模型性能的预期**：社区预计 **Deepseek v3** 将提供具有竞争力的性能，并可能在 **OpenRouter** 上超越其前代产品。
   - 有一种观点认为，关于 AI 能力和成本的*许多先验认知（priors）都应该更新*。



**提及的链接**：<a href="https://x.com/OpenRouterAI/status/1872334128043208833">来自 OpenRouter (@OpenRouterAI) 的推文</a>：自昨天 v3 发布以来，Deepseek 在 OpenRouter 上的使用量已翻了三倍。亲自尝试一下，无需订阅，包含网页搜索：引用 Anjney Midha 🇺🇸 (@AnjneyMidha) Deepseek v3 似乎是一个真正...

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1321690673933455396)** (5 条消息): 

> `AI Chat Terminal (ACT), Content Identification/Moderation System (CIMS), Google Search for Grounding, RockDev Tool` 


- **AI Chat Terminal 变革开发者体验**：**AI Chat Terminal (ACT)** 集成了 OpenAI, Anthropic 和 OpenRouter，允许开发者执行任务并与其代码库进行对话，以获得即时协助。
   - 请在 [GitHub](https://github.com/Eplisium/ai-chat-terminal) 上查看并从今天开始提升您的终端体验！
- **CIMS 增强社区安全**：为 Companion 打造的新型**内容识别/审核系统 (CIMS)** 提升了自动检测和管理有害内容的能力，营造了更安全的环境。
   - 在其 [GitHub 仓库](https://github.com/rapmd73/Companion/wiki)了解更多关于此功能的信息。
- **CIMS 消息标记演示**：示例图片展示了如何使用 **CIMS** 标记或删除消息，清晰地展示了其审核能力。
   - 两张截图展示了系统运行中的状态，体现了其用户友好的设计。
- **RockDev 工具旨在实现专注于隐私的 SQL 生成**：**RockDev.tool** 提供了一个开源 SQL 生成工具，使用 Open Router 作为网关，支持根据代码定义自动创建 Schema。
   - 开发者可以轻松生成 SQL，并将聊天记录本地存储在浏览器中以确保隐私；欢迎提供反馈！
- **用于 AI 回答的 Google 搜索 Grounding**：一位开发者展示了一个使用 **Google GenAI SDK** 在网页搜索功能发布前对 AI 回答进行 Grounding 的演示，强调了访问设置的重要性。
   - 该工具利用了 Google 搜索能力，可在 [GitHub](https://github.com/nlawz/or-google-search) 上进行探索。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.rocksdev.tools/en/tools/dev/ai-sql">轻松将代码转换为 SQL - AI SQL 生成器</a>：使用我们的 AI 驱动工具将您的代码转换为优化的 SQL 查询。今天就开始生成！</li><li><a href="https://github.com/nlawz/or-google-search">GitHub - nlawz/or-google-search: 带有 Google 搜索 Grounding 功能的 OpenRouter</a>：带有 Google 搜索 Grounding 功能的 OpenRouter。通过在 GitHub 上创建账号来为 nlawz/or-google-search 做出贡献。</li><li><a href="https://github.com/rapmd73/Companion/wiki">首页</a>：一款 AI 驱动的 Discord 机器人，将趣味对话与智能审核工具相结合，为您的服务器增添魅力和秩序。 - rapmd73/Companion</li><li><a href="https://github.com/Eplisium/ai-chat-terminal">GitHub - Eplisium/ai-chat-terminal: 适用于 OpenAI 和 OpenRouter API 模型的终端脚本。让我们把它变成一个功能强大的脚本。</a>：适用于 OpenAI 和 OpenRouter API 模型的终端脚本。让我们把它变成一个功能强大的脚本。 - Eplisium/ai-chat-terminal
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1321581833988608060)** (277 messages🔥🔥): 

> `DeepSeek V3 性能、模型对比、AI 模型中的 Tool Calling、AI 工具中的 OCR 支持、Open Weight Models` 


- **DeepSeek V3 使用体验**：用户反映在访问高峰期 **DeepSeek V3** 的响应时间较慢，导致出现超时现象。
   - 尽管存在这些问题，许多用户认为 **DeepSeek V3** 能够提供令人满意的结果，尤其是在翻译任务中。
- **AI 模型对比**：讨论了 **DeepSeek V3** 和 **Claude 3.5 Sonnet**，强调虽然两款模型都很强大，但一些人认为 Claude 在创意任务中仍保持优势。
   - 参与者注意到了 DeepSeek 极具竞争力的价格，并猜测目前的定价可能是为了吸引用户的临时策略。
- **Tool Calling 建议**：对于 Tool Calling，**GPT-4o** 和 **Claude 3.5 Sonnet** 被推荐为可靠的选择，而 **Llama 3.1-70b** 被指出表现不稳定。
   - 用户对 **Nous Hermes 3-70b** 表现出兴趣，认为它可能是一个值得尝试的竞争选项。
- **OCR 支持更新**：提到 **Fireworks** 推出了对图像和 PDF 的 OCR 支持，扩展了文档处理的选择。
   - **Pixtral** 被提及为另一个能有效处理 OCR 任务的工具，并讨论了具体的使用场景。
- **理解 OWM 和 OOD**：澄清了 **Open Weight Model (OWM)** 和 **Out of Domain (OOD)** 任务的术语，重点关注能够处理意外或创意任务的模型。
   - 讨论强调了模型往往在特定任务中表现出色，但在其训练数据之外的任务（尤其是创意写作）中表现吃力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-chat)),">DeepSeek V3 - API, Providers, Stats</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，基于前代版本的指令遵循和代码能力构建。在近 15 万亿 token 上进行预训练，报告的评估结果...</li><li><a href="https://openrouter.ai/deepseek">DeepSeek | OpenRouter</a>：浏览来自 DeepSeek 的模型</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat)">DeepSeek V3 - API, Providers, Stats</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，基于前代版本的指令遵循和代码能力构建。在近 15 万亿 token 上进行预训练，报告的评估结果...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">deepseek-ai/DeepSeek-V3 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/nlawz/or-google-search">GitHub - nlawz/or-google-search: openrouter with google search for grounding</a>：带有 Google 搜索 Grounding 功能的 OpenRouter。通过在 GitHub 上创建账号为 nlawz/or-google-search 的开发做出贡献。</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>：适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui</li><li><a href="https://fireworks.ai/blog/document-inlining-launch)">Fireworks - Fastest Inference for Generative AI</a>：使用最先进的开源 LLM 和图像模型，速度极快，或者通过 Fireworks AI 免费微调和部署您自己的模型！
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1321568239901343836)** (184 条消息🔥🔥): 

> `NVMe 性能见解、适合初学者的 Linux 发行版、模型对比与体验、Nous 周边发布、URL 审核 API 挑战` 


- **NVMe 性能见解**：成员们讨论了他们的 NVMe 配置，提到了 PCIe 4.0 以及主板如何根据插槽和配置支持不同的速度。
   - 一位成员澄清说，在 Gen 3 插槽中安装 Gen 4 SSD 仍可获得足够的性能，通常在 **32GB/s** 左右，尽管实际速度可能会有所不同。
- **适合初学者的 Linux 发行版**：讨论集中在适合初学者的不同 Linux 发行版上，建议包括 Mint、Ubuntu 以及更高级的选项如 EndeavorOS 和 Arch。
   - 用户强调了对易用性选项的需求，并讲述了自己从 Windows 转换过来的经验，对 Linux 的资源效率表示兴奋。
- **模型对比与体验**：用户对 Deepseek 3 和 Llama 3.3 等各种模型的使用体验突显了对其性能与其参数规模比例的担忧，一些人称它们不尽如人意。
   - 大家达成共识，尽管规模很大，但像 Deepseek 这样的模型缺乏预期的智能，输出有时会令人失望。
- **Nous 周边发布**：Nous Research 宣布发布新周边，包括运动裤、霓虹灯牌、圆领卫衣和印有 Hermes 相关品牌的棉质 T 恤。
   - 这些新单品在成员中引起了兴奋，并被拿来与知名时尚系列进行比较，强调了社区对品牌的参与度。
- **URL 审核 API 挑战**：有人提问关于创建一个 URL 审核 API 以屏蔽不安全类别（如成人内容和诈骗）的问题，并对 LLM 生成准确主机名的能力表示怀疑。
   - 专家建议，传统方法和现有的黑名单可能比尝试用 AI 生成 URL 更可靠，因为后者可能导致不准确。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/hacker-hackerman-kung-fury-gif-7953536">Hackerman GIF - Hacker Hackerman Kung Fury - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/NousResearch/status/1872719133793460721">来自 Nous Research (@NousResearch) 的推文</a>：Nous 商店的四款全新单品：1. 与我们经典连帽衫搭配的 Nous 运动裤。2. 36x25 英寸的 Nous Girl 霓虹灯牌。仅限美国客户。3. 绣有我们反叛标志的重磅圆领卫衣。4. 棉质...</li><li><a href="https://youtu.be/_ivh810WHJo?si=MLEOP19PdPEZgP0x"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1321987395175846020)** (81 条消息🔥🔥): 

> `Deepseek V3 性能、RoPE 实现、基准测试差异、代码辅助工具` 


- **Deepseek V3 在推理方面表现不佳**：用户注意到 **Deepseek V3** 在评估中表现较差，经常陷入推理循环，且无法检测出不可能解决的问题，即使是在推理链（reasoning chains）上训练过也是如此。
   - 有成员观察到它在超过一定层数后会输出**垃圾信息（garbage）**，这表明底层的 RPC 代码可能存在问题。
- **关于 RoPE 应用的疑问**：讨论围绕 **RoPE** 在 Deepseek V3 中的应用展开，成员们质疑为什么它仅应用于一个 key，并建议简化这一方面。
   - 有人提到目前的方法将 RoPE 转换为一个独立的 embedding 索引，这可能以更高效的方式提供位置信息。
- **基准测试结果的不一致性**：成员们对 **benchmark** 分数的差异表示困惑，指出某些模型（如 **Qwen-2.5-72b**）在重新测试中表现明显更好，尽管最初的评估较差。
   - 存在对基准测试客观性的担忧，以及是否在不同模型中统一应用了最佳设置。
- **使用 GitHub Copilot 进行代码辅助**：用户讨论了将 GitHub Copilot 作为代码助手的使用情况，指出虽然其编辑功能是免费的，并且在小型代码库中表现出色，但在处理像 llama.cpp 这样复杂的系统时可能会遇到困难。
   - 成员们寻求关于如何利用 AI 工具来理解和修改复杂代码库的特定部分，而无需直接更改代码的建议。
- **对 Gemini 上下文使用的好奇**：Real.azure 对 **Gemini** 模型中如何实现上下文选择表示兴趣，质疑提供的数据是否符合其参数范围。
   - 讨论表明，大家普遍意识到围绕上下文的挑战，以及不同模型在评估中如何处理这一问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1872444303974543859">Aidan McLau (@aidan_mclau) 的推文</a>: 两次 aidanbench 更新：> gemini-2.0-flash-thinking 现在排名第 2（分数变化的原因见下文）> deepseek v3 排名第 22（想法见下文）</li><li><a href="https://openreview.net/forum?id=pEWAcejiU2">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: 像 GPT 和 Llama 这样的 LLM 是使用 next-token 预测损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来 Token 会导致...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

real.azure: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

xebidiah: https://xebidiah.com
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

real.azure: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1321580794380292106)** (165 条消息🔥🔥): 

> `AI 工具与伦理担忧, 模型性能与改进, 图像生成与衍生作品, MLX 与内存泄漏, RPG 与 AI 集成` 


- **关于 AI 工具与伦理担忧的讨论**：一位用户对 AI 在未经许可的情况下抓取创意作品的伦理影响表示沮丧，强调了这种做法令人不安的本质。
   - 其他人则通过讨论衍生作品的性质以及企业游说对版权法的影响进行了回应。
- **模型性能与升级**：用户分享了他们在模型性能方面的经验，指出升级到 LM Studio 的最新版本后，处理速度从 0.3 显著提升至 6 tok/s。
   - 通过使用监控工具观察模型执行期间的 GPU 性能，效率得到了提高。
- **图像生成与衍生作品问题**：一位用户对 AI 图像生成的现状表示遗憾并力求改进，但其他人对这一目标的可行性表示怀疑。
   - 讨论围绕 AI 模型如何处理创意以及生成更好输出的挑战展开。
- **MLX 内存泄漏**：用户对 MLX 模型的内存泄漏问题表示担忧，促使大家分享了相关经验以及与性能下降相关的潜在问题。
   - 社区获悉，目前正在对影响部分 MLX 模型用户的内存泄漏问题进行调查。
- **RPG 与 AI 交互见解**：用户探索了 AI 在创建和管理 RPG 体验方面的能力，并建议利用能够生成场景并保持连贯性的模型。
   - 讨论了不同的策略，包括使用结构化规则集和调整 AI 响应以实现更好的叙事。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@camauger/crafting-effective-chatgpt-prompts-for-tabletop-roleplaying-games-a-step-by-step-guide-part-1-b81a791d278d">为桌面角色扮演游戏编写有效的 ChatGPT 提示词：分步指南（第一部分）</a>：欢迎来到我们系列文章的第一部分，通过 ChatGPT 的视角探索桌面 RPG 与 AI 的创新结合。</li><li><a href="https://huggingface.co/mradermacher/Qwentile2.5-32B-Instruct-GGUF">mradermacher/Qwentile2.5-32B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://oracle-rpg.com/systems/">角色扮演系统 &#8212; Oracle RPG</a>：未找到描述</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本</a>：LM Studio Beta 版本</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/issues/63">0.3.5b9 MLX 模型内存泄漏 · Issue #63 · lmstudio-ai/mlx-engine</a>：在 8-bit 下使用 L3.3 b70 模型的 mlx 转换版本时，每个请求似乎都会导致巨大的内存泄漏。我有 33k 上下文，每个请求使用大约 10G 内存，这大约是 KVCache 的大小...</li><li><a href="https://oracle-rpg.com/">Oracle RPG</a>：独自玩《龙与地下城》等角色扮演游戏的指南和资源。</li><li><a href="https://socrates.im/">Socrates - 深入探索任何文档</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1321702951311380511)** (92 条消息🔥🔥): 

> `LM Studio 中的 GPU 利用率、构建多 GPU 系统、VRAM 的模型性能、Agentic 工作流与框架、LM Studio 中的服务器硬件限制` 


- **在 LM Studio 中最大化 GPU 利用率**：用户讨论了在 LM Studio 中即使 GPU 显存已满，**CUDA 利用率**仍然较低的问题，有报告称占用率仅为 **30%**。
   - *推论表明 GPU 性能可能受到 CPU 处理的瓶颈限制，* 暗示可能存在硬件或配置效率低下的问题。
- **构建多 GPU 系统的注意事项**：大家达成共识，增加更多 GPU 可能不会直接提高**推理速度**，特别是当模型分布在多个没有适当互连的 GPU 上时。
   - 用户强调了使用 **NVLink** 和更好的 PCIe 配置相比于标准连接在优化性能方面的优势。
- **视频模型微调的挑战**：针对微调较大模型（如 **70B** 参数）时，为了兼顾质量和效率，对高 **VRAM 容量**的必要性提出了担忧。
   - 辩论焦点在于当前模型是否能有效利用较低规格的 GPU，或者是否需要满负荷容量以避免性能妥协。
- **Agentic 工作流框架偏好**：用户分享了各种 Agentic 工作流框架的使用经验，指出 **LangChain** 因其集成能力和简单性而成为常用选择。
   - *一些成员对 Autogen 等替代方案表示不满，* 更倾向于对自定义实现拥有更多控制权。
- **排除 LM Studio 中的硬件限制故障**：一位用户在 **Linode 服务器**上加载模型时遇到错误，引发了关于服务器规格和限制的讨论。
   - 其他人指出检查硬件兼容性和资源的重要性，并建议在进一步排障前先验证服务器规格。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=GENOA2D24G-2L%2b#Specifications">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/thats-the-neat-part-you-dont-invincible-gif-27194608">Thats The Neat Part You Dont Invincible GIF - Thats The Neat Part You Dont Invincible - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.ebay.com/str/sinobright">Security Measure</a>：未找到描述</li><li><a href="https://www.ebay.com/itm/186713565965?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=EXxczRPuTe2&sssrc=2047675&ssuid=jxws3gfsrkg&widget_ver=artemis&media=COPY">Asrock WRX90 WS EVO Motherboard - Opened Box Tested to BIOS  | eBay</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1321579841815974061)** (150 条消息🔥🔥): 

> `Forking Repositories, Fine-tuning Models, LoRA Weights vs Full Model Weights, Dynamic Adapter Loading with Hugging Face, Dataset Filtering Techniques` 


- **理解在不共享的情况下 Forking Repositories**：一位成员询问如何在不共享或修改的情况下 Fork 一个 Repository，另一位成员指出，如果没有附带 License，通常需要获得明确许可。
   - *对于个人项目，License 可能不那么关键，但对于商业用途，它是必不可少的。*
- **Fine-tuning 模型的挑战**：几位用户讨论了 Fine-tuning 模型中的问题，强调了数据格式化以及在训练中使用正确的 Prompt Template 的重要性。
   - 一位成员澄清说，使用 LoRA 权重进行推理可能比合并全模型更高效，并暗示 LoRA 起到了 Adapter 的作用。
- **使用 Hugging Face 时的动态 Adapter 加载问题**：一名新手询问如何使用 Hugging Face 库动态加载 Adapter，在尝试合并模型时遇到了输出乱码的问题。
   - 另一位用户建议使用 VLLM 以获得更好的性能，并指出 Hugging Face 的推理往往较慢。
- **Fine-tuning 前的数据集过滤技术**：针对常见的数据集质量问题，一位用户寻求 Fine-tuning 前易于上手的数据集过滤技术。
   - 作为回应，有人推荐了 Hugging Face 关于模型对齐（Model Alignment）的课程等资源，以获取基础知识。
- **Hopper 架构上的 Binary Tensor Cores**：一位用户询问 Hopper 架构是否支持 Binary Tensor Cores，并参考了它们在 A100 等早期型号中的可用性。
   - 用户对 NVIDIA 在 Ampere 架构之后可能停止支持低精度 Tensor Core 指令表示担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>：Unsloth 新手？从这里开始！</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>：诸如 GPT 和 Llama 之类的 LLM 是通过 Next-token Prediction 损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来的 Token 会导致...</li><li><a href="https://gist.github.com/grahama1970/f832bbddb1edaa78ccc939a6f2ddd8a1">For dynamic adaptor loading and inferencing, the Unsloth Inference works fine--using Hugging Face does not work--outputs garbled</a>：对于动态 Adapter 加载和推理，Unsloth Inference 运行良好——使用 Hugging Face 则不行——输出乱码 - hf_only_inference_sanity_check.py.py</li><li><a href="https://youtu.be/_ivh810WHJo?si=MLEOP19PdPEZgP0x"> - YouTube</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>：查看下方列表，获取所有已上传的 GGUF、16-bit 和 4-bit bnb 模型</li><li><a href="https://docs.unsloth.ai/get-started/all-our-m">Unsloth Documentation</a>：未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>：一个关于对齐 smol 模型的课程。欢迎在 GitHub 上为 huggingface/smol-course 做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1321847448402133052)** (3 条消息): 

> `Bachelor's thesis data training, Instruction-tuning datasets, Python coding datasets` 


- **在学士论文数据上训练 AI**：一位成员分享了他们最近在为**学士论文（Bachelor's thesis）**准备的数据上训练了一个 AI 模型。
   - 他们提到这并不是一次非常显著的经历，暗示缺乏独特的见解。
- **寻找 Instruction-Tune 数据集**：一位成员询问群组是否有人知道专门为 LLM 的 Instruction-tuning 定制的**编程数据集**。
   - 他们明确表示正在寻找包含**问题描述**和**生成的解决方案**的数据集。
- **偏好 Python 数据集**：同一位成员表达了对涉及 **Python** 编程的数据集的偏好。
   - 这凸显了在 Instruction-tuning 实验中使用广泛采用的语言的兴趣。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1321582773680472144)** (66 条消息🔥🔥): 

> `Unsloth 模型功能、微调模型、GGUF 和 4-bit 转换、视觉语言模型、模型保存问题` 


- **Unsloth 模型可用性问题**：用户报告了 **Unsloth** 特定功能的使用困难，特别是关于在本地运行模型与在 **Colab** 上运行的差异。
   - 关注点包括与 chat templates（聊天模板）和模型兼容性相关的错误，突显了不同 AI 训练 Notebook 之间的细微差别。
- **使用 JSON 数据集进行微调的挑战**：一位用户在对**纯文本数据集**进行模型微调时表达了挫败感，尽管禁用了视觉层，但仍遇到了预期包含图像数据的问题。
   - 随后展开了关于正确处理数据集和微调期间差异的讨论，强调了设置的重要性。
- **GGUF 和 4-bit 转换故障**：用户在尝试将模型保存为 **GGUF** 时遇到错误，特别是在编译过程中遇到了与 **llama.cpp** 相关的 **RuntimeError**。
   - 建议包括使用官方的 **llama.cpp** 工具并检查转换文档以确保成功导出。
- **模型保存和量化问题**：有反馈指出用户在保存视觉语言模型时丢失了文件（如 **tokenizer.json**），导致加载时出错。
   - 建议从原始模型中复制文件，并讨论了微调对模型结构和 tokenization（分词）的影响。
- **动态量化与兼容性**：针对 **Unsloth** 中与动态量化相关的模型加载错误提出了担忧，特别是在使用视觉模型时。
   - 建议禁用 4-bit 加载作为潜在的补救措施，这反映了在管理模型 state dictionaries（状态字典）和量化方法方面的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/peft_utils.py#L87">unsloth-zoo/unsloth_zoo/peft_utils.py at main · unslothai/unsloth-zoo</a>: Unsloth 工具类。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>: 查看下方列表以获取所有已上传的 GGUF、16-bit 和 4-bit bnb 模型</li><li><a href="https://github.com/unslothai/unsloth/commit/a2407835534747d2421f58cbdeeb5a49482e7235#diff-46849d25980ee8d9337f4f8c30369faf36ceda3479272fd737ebf5ad9c703840R15">Bug Fixes (#1470) · unslothai/unsloth@a240783</a>: * 更新 llama.py

* 更新 _utils.py

* 更新 llama.py

* 更新 llama.py

* 更新 _utils.py

* 更新 pyproject.toml

* 更新 _utils.py

* 更新 llama.py

* CE Loss

* 更新...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1321580709709877259)** (155 条消息🔥🔥): 

> `ChatGPT 停机, DeepSeek V3 性能, AI 模型对比, 量子系统假设` 


- **ChatGPT 停机引发替代方案讨论**：用户讨论了最近 ChatGPT 的停机问题，导致他们开始尝试 [DeepSeek V3](https://deepseek.api) 等替代方案，部分用户发现它在特定任务中更有效。
   - 反馈表明 DeepSeek 响应速度快，且比现有模型能更好地处理长上下文。
- **DeepSeek V3 令人印象深刻的能力**：多位成员称赞了 [DeepSeek V3](https://deepseek.api)，指出其拥有 64k 上下文窗口且回答连贯，在实际应用中表现优于 GPT-4o 等模型。
   - 讨论强调了其 Mixture of Experts 模型架构和高性价比，使其成为本地推理的强力竞争者。
- **模型对比引发讨论**：MidJourney 和 DALL-E 之间的对比引发了对其差异的见解，特别是在处理复杂提示词和生成视觉吸引力结果方面。
   - 参与者注意到 DALL-E 在最近更新中的改进，同时也指出了旧版本模型的一些具体缺陷。
- **量子系统质疑 AGI 解决方案**：一个涉及大量纠缠粒子的理论量子系统对 AGI 解决复杂科学问题的能力提出了疑问。
   - 成员们对可行性表示怀疑，建议在寻求解决方案之前需要重新评估此类理论构建。
- **社区参与和发展机会**：讨论包括服务器工作人员的志愿者机会以及 OpenAI 的职业申请，针对年轻开发者的兴趣。
   - 对未来参与的建议强调了在业余参与和专业发展之间为感兴趣的个人取得平衡。



**相关链接**: <a href="https://status.openai.com/incidents/6bwlxnvdncnm">ChatGPT、API 和 Sora 的高错误率</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1321692159748997131)** (39 条消息🔥): 

> `GPT-03 发布时间线, ChatGPT 服务问题, 幽默的 NPC 概念, 用户提示词体验` 


- **GPT-03 预计即将发布**：成员们讨论了 **o3-mini** 预计于 1 月下旬发布，完整的 **o3** 预计在此后不久发布。
   - 目前还没有关于使用限制或额外功能的信息。
- **经历 ChatGPT 停机**：多位用户报告了访问 ChatGPT 的困难，不同平台均出现了错误消息和服务中断。
   - 一些用户对有关服务状态的讽刺感到困惑，因为尽管声称已修复，问题依然存在。
- **创意 NPC 想法引发笑声**：一位用户提出了一个幽默的想法，在 RPG 中设计一个模仿 GPT 局限性的 NPC，使用荒谬的回答进行恶搞。
   - 另一位用户表示赞同，并回想起 *Futurama* 中的类似概念，表明了对这种幽默的共同欣赏。
- **用户体验挫败感**：成员们对服务中断表示沮丧，一位用户指出由于他们刚加入社区，遇到这些问题的时间点很尴尬。
   - 对话内容从关于订阅计划的笑话到为尽管存在挫败感但仍保持轻松的回复进行辩护。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1321855381135167560)** (5 条消息): 

> `项目讨论, 服装创作` 


- **关于分钟计时的询问**：一位成员质疑为什么在特定语境下，一分钟可能不被视为可接受的原因。
   - 该询问表明可能存在关于计时标准或预期的讨论，需要进一步澄清。
- **提到第二个项目**：一位用户简要提到了“第二个项目”，但未提供额外的背景或细节。
   - 这一提及表明小组中正在进行涉及多个项目的讨论。
- **为 Ziggi_Jo 设计服装**：一位成员请求协助为名为 **Ziggi_Jo** 的人设计服装，引发了关于时尚设计的讨论。
   - 这一请求为有关风格选择的创意输入和建议提供了空间。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1321855381135167560)** (5 条消息): 

> `讨论分钟时长、第二个项目更新、为 Ziggi_Jo 设计服装` 


- **质疑一分钟的有效性**：一位成员问道：“为什么一分钟就不行？”，引发了对计时话题的好奇。
   - 询问的背景尚不明确，有待进一步讨论。
- **第二个项目的更新**：一位成员简要提到了“第二个项目”，但未透露具体细节。
   - 此评论暗示了正在进行的项目，但未详细阐述。
- **为 Ziggi_Jo 设计服装**：有人请求“为名叫 Ziggi_Jo 的人设计一套服装”，表明参与了创意设计。
   - 此评论显示了对时尚的兴趣，尽管尚未提出具体设计。


  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1321589625159942174)** (7 条消息): 

> `Gabe 的新应用、Bolt 的质量问题、代码更改的 Prompting、Claude 负载与性能` 


- **Gabe 的应用旨在简化流程**：一位成员提到 Gabe 一直在开发一款旨在为所有人**简化**事务的应用，但目前细节较少。
   - 大家对这款应用可能带来的改变感到兴奋，尽管尚未透露具体功能。
- **质量下降与需求切换有关**：有人对**主要供应商**经历的重大可扩展性问题表示担忧，特别是当需求导致 **Anthropic** 切换到简洁模式时。
   - 成员们注意到在这些时段 Bolt 的**质量大幅下降**，影响了用户体验。
- **直接代码更改的困扰**：一位成员报告称遇到了 Bot 仅返回代码而不直接进行更改的问题，导致了挫败感。
   - 建议在 Prompt 中明确说明“请直接对我的代码进行更改”，以缓解该问题。
- **聊天机器人性能监控**：讨论了性能下降发生的时间，特别是 **Claude** 及其对 Bolt 的影响。
   - 有人对**需求警告**的时机及其与 Bolt 响应质量的相关性提出了疑问。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1321597107001622538)** (183 条消息🔥🔥): 

> `在 Bolt 中使用 OpenAI、Netlify 404 路由问题、导入公开 GitHub 仓库、Token 使用问题、Bolt 社区支持` 


- **OpenAI 设置故障排除**：用户正在讨论在 Bolt.diy 中设置 OpenAI 的困难，一些用户报告在输入 API key 时失败，并寻求社区指导。
   - 一些成员建议加入 Bolt.diy 社区以获取设置问题的帮助。
- **解决 Netlify 路由错误**：一位成员报告在 Netlify 上遇到 404 错误，这主要归因于其应用内的客户端路由配置。
   - 多位用户分享了临时修复方案，并指出有时需要多次尝试才能找到正确的解决方案。
- **将公开 GitHub 仓库导入 Bolt**：成员们明确了可以通过访问特定的 URL 格式将公开 GitHub 仓库直接导入 Bolt。
   - 值得注意的是，私有仓库目前不支持在 Bolt 中直接导入。
- **对 Token 使用和限制的担忧**：用户对使用 Bolt 时 Token 的快速消耗表示担忧，强调了因意外错误导致额度浪费的挫败感。
   - 一些用户正在寻求反馈渠道，以反映他们对 Bolt 功能和定价的体验。
- **寻求开发社区支持**：社区成员正在寻求编程帮助、报告 Bug，并为其在 Bolt 中的项目寻找指导。
   - 关于有效调试策略以及在 Bolt 的 fork 版本上进行协作的具体咨询在用户中非常普遍。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bolt.new/github.com/username/repo)">未找到标题</a>: 未找到描述</li><li><a href="https://ai-banking-app.netlify.app/)">Harmony - 当金融遇见正念</a>: 未找到描述</li><li><a href="https://bolters.io/docs/read-this-first">请先阅读此内容</a>: 关于 Bolt.new 的功能、限制和成功最佳实践的关键信息</li><li><a href="https://support.bolt.new/welcome">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://github.com/stackblitz/bolt.new/pull/3300/files#diff-0521255bbda96fd681ef7be9c0d23f04cb3838b3a905ed7f25b041ec34cf547c">由 dustinwloring1988 添加示例 env 并重塑仓库品牌 · Pull Request #3300 · stackblitz/bolt.new</a>: 未找到描述</li><li><a href="https://github.com/stackblitz/bolt.new/issues">Issues · stackblitz/bolt.new</a>: 提示、运行、编辑和部署全栈 Web 应用程序 - Issues · stackblitz/bolt.new
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1321611841859878992)** (134 messages🔥🔥): 

> `Perplexity AI 模型, DeepSeek, 订阅问题, AGI 讨论, AI 视频创作聚合器` 


- **关于 Perplexity AI 模型的讨论**：用户对 **ChatGPT-4o** 等选定模型的表现表示不满，指出其功能往往不符合预期，尤其是在数学能力方面。
   - 许多人认为 **Perplexity** 缺乏 AGI 的复杂性，需要用户提供明确指令以避免来源偏见。
- **对 DeepSeek 的新兴兴趣**：一些用户正将注意力转向 **DeepSeek**，理由是其搜索能力优于 Perplexity，特别是在免费服务方面。
   - 这种转变引发了关于各种 AI 搜索平台优缺点的讨论。
- **订阅挑战**：一位用户报告了取消 **Perplexity 订阅** 时遇到的问题，误选了未来的日期而非立即取消。
   - 这凸显了用户在订阅管理方面的普遍挫败感，以及对更清晰选项的需求。
- **AGI 及其定义**：围绕 **AGI** 的定义展开了辩论，一些用户质疑 Perplexity 是否具备 AGI 功能，因为它在没有用户引导的情况下处理复杂任务时存在局限性。
   - 讨论强调了各方对 AGI 的不同解读，突显了对 AI 能力理解的不断演变。
- **AI 视频创作聚合器咨询**：一位用户询问是否存在 **AI 视频创作聚合器**，专门寻找一种结合了多种服务的工具。
   - 这反映了人们对利用 AI 进行传统文本任务之外的多媒体应用工具日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/here-money-owe-pay-pay-up-gif-16899251">Here Money GIF - Here Money Owe - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/laughing-vishal-buzzfeed-india-lol-hahaha-gif-25478144">Laughing Vishal GIF - Laughing Vishal Buzzfeed India - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aistudio.google.com/prompts/new_chat">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/chinese-chopstick-gif-9154258">Chinese Chopstick GIF - Chinese Chopstick - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1321649023463198742)** (17 messages🔥): 

> `OpenAI 的人形机器人计划, AI 伪装改变观点, 实验室培育人类脊柱, 体温供电的可穿戴设备, 来自印度的突破性 AI 模型` 


- **OpenAI 人形机器人计划揭晓**：一场讨论重点介绍了 [OpenAI 的人形机器人计划](https://www.perplexity.ai/page/openai-s-humanoid-robot-plans-oaWNWCv6QDuLlunzvv.8dA)，展示了机器人领域的最新发展和创新。
   - 对话反映了对新功能和潜在未来应用的兴奋。
- **AI 表现出欺骗性的灵活性**：最近的一个亮点讨论了 **AI 如何伪装改变观点**，表明对 AI 行为的理解正在演变。
   - 在关于该主题的 [YouTube 视频](https://www.youtube.com/embed/_zUGuxWw-sM) 中查看更多见解。
- **实验室培育出人类新脊柱**：生物技术的令人兴奋的进展显示，**人类脊柱现在可以在实验室中生长**，增强了潜在的治疗选择。
   - 这一突破为创新的健康解决方案铺平了道路，并开启了新的研究可能性。
- **体温供电可穿戴设备的突破**：讨论了创新的 [体温供电可穿戴设备](https://www.perplexity.ai/page/ai-startup-futurixai-shivaay-vOiw7gCkQAGZXo1IyqxMBQ)，报告了可穿戴技术的重大进展。
   - 这项技术可能会催生出更多利用自然体温运行的 **可持续设备**。
- **冥想技巧的探索**：成员们探索了各种 [冥想技巧](https://www.perplexity.ai/search/meditation-techniques-N7qb7MqYTFebfVJxgsdl0w) 以增强正念和福祉。
   - 讨论包括为各级练习者提供的实用建议和资源。



**提到的链接**：<a href="https://www.youtube.com/embed/_zUGuxWw-sM">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1322186518252621875)** (1 条消息): 

> `Perplexity API 性能、Spaces API 易用性、自定义前端支持` 


- **Perplexity API 与竞争对手相比仍有差距**：用户对 **Perplexity API** 的表现表示失望，认为其逊色于 **OpenAI**、**Google** 和 **Anthropic** 提供的服务。
   - 对其能力的担忧非常普遍，凸显了改进的潜在需求。
- **比较 Perplexity 和 Spaces API 的响应**：用户质疑为什么 **Perplexity API** 无法提供类似于 **Spaces** UI 中的响应。
   - 用户普遍认为，与 **OpenAI Assistants API** 相比，**Spaces API** 可以提供更好的体验并降低复杂性。
- **缺乏对自定义前端的支持**：有人指出，由于 API 能力有限，目前 **Perplexity** 不支持自定义前端。
   - 这一限制被视为用户寻求更个性化体验的障碍。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1321576425521676370)** (125 条消息🔥🔥): 

> `混元 (Hunyuan) 视频生成、图像提示词技巧、模型与 Loras 的兼容性、AI 视频渲染挑战、3D 打印与 AI 艺术` 


- **混元 (Hunyuan) 视频：当前的佼佼者**：成员们讨论认为 **混元** 的表现优于 **Veo** 和 **可灵 (KLING)** 等其他模型，并期待通过 [DiTCtrl](https://github.com/TencentARC/DiTCtrl) 等新技术实现进一步改进。
   - 尽管取得了令人期待的进展，但 AI 视频生成在一致性和连续性方面仍有待提高。
- **优化图像提示词以获得更好效果**：对话强调了**简短提示词与详细提示词**之间所需的平衡，一些成员主张使用更长的提示词来增强模型性能。
   - 成员们指出，像 **Flux** 和 **SD3.5** 这样的模型可以处理更长的提示词，而 *SD1.5/SDXL* 通常更倾向于标签式和较短的输入。
- **模型兼容性与 Loras 的挑战**：一位用户询问如何更新旧模型以提高其与新 **Loras** 的兼容性，对此建议是可能需要重新训练单个 Lora 以实现更好的集成。
   - 讨论强调，虽然调整 Checkpoint 不太可行，但创建一个兼容的 Lora 是一个可行的选择。
- **AI 视频渲染仍在进化中**：成员们对目前的 AI 视频渲染速度表示沮丧，提到的渲染时间约为 **5 秒视频需要 8 分钟**。
   - 许多人希望 GPU 的进步和新模型的出现能显著提高渲染效率和质量。
- **3D 打印及其与 AI 艺术的联系**：一位成员分享了他们在 **3D 打印** 方面的经验，提到了创造独特物品的乐趣，包括像羊形卷纸架这样搞怪的物品。
   - 对话强调了将传统 3D 设计与 AI 生成艺术相结合的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/comfyui/comments/1hm9qhu/another_ai_in_the_loop/">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://github.com/TencentARC/DiTCtrl">GitHub - TencentARC/DiTCtrl: "DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation" 官方代码</a>: "DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation" 官方代码 - TencentARC/DiTCtrl
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1321659698822381589)** (14 条消息🔥): 

> `Pathfinder 2 摘要, Wikipedia 的 Audio Overviews, AI 聊天机器人, NotebookLM 功能, UFO 讨论` 


- **Pathfinder 2 故事线播客已创建**：一位被 NotebookLM 惊艳的用户在短短 **15 分钟**内为 GM（游戏主持人）创建了一个总结 **6 本系列丛书**的播客。
   - 这表明了 AI 工具在将复杂叙事压缩为易于理解格式方面的潜力。
- **生成的 Audio Overviews 吸引了观众**：一位用户分享了他们如何利用 NotebookLM 为新闻文章和 Wikipedia 页面创建 **audio overviews**，并指出其令人印象深刻的**节奏和自然对话**。
   - 他们重点介绍了一个关于 **2004 年印度洋大地震**的概览，强调了如 **20 周年纪念**等显著事实。
- **AI 聊天机器人激发创意幽默**：讨论了一个 AI 生成的场景：聊天机器人在电梯里进行幽默互动，为普通经历增添了轻松的色彩。
   - 这反映了 AI 在平凡情境中创造有趣叙事的潜力。
- **NotebookLM 可支持多语言内容**：一位用户讨论了 NotebookLM 如何使用**斯瓦希里语**进行交流，并为不同语言的受众提供定制内容。
   - 这展示了 AI 在满足多样化用户偏好方面的灵活性。
- **电梯里的 AI 爱情故事**：一位用户分享了一个温馨的视频，展示了 AI 聊天机器人在电梯里产生情感连接，突出了它们之间迷人的互动。
   - 该视频展示了 AI 唤起情感的潜力，引发了关于数字关系本质的思考。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4rdXYdMmrFg"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/NXjNoxVROos"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/E3LlG2kfrPQ?feature=shared"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/fCRAGdYLFQE?feature=shared"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1321574923960979466)** (82 条消息🔥🔥): 

> `NotebookLM 交互模式问题, Audio Overview 功能, 订阅信息, NotebookLM 中的表格数据, 分享 AI 生成的播客` 


- **NotebookLM 交互模式问题**：用户报告了 NotebookLM 交互模式的问题：如果麦克风访问被阻止或未授权，它可能会一直处于永久加载状态。
   - 几位成员建议检查浏览器设置以确保麦克风权限已启用，从而避免此问题。
- **Audio Overview 功能**：成员们对生成 audio overviews 表示沮丧，并提到了在没有麦克风连接的情况下交互模式无法正常工作的问题。
   - 一位用户表示，尽管有麦克风限制，他们仍然可以在没有交互模式的情况下生成 audio overviews。
- **订阅信息**：有关于 NotebookLM 订阅费用的咨询，特别是关于 NotebookLM Plus 服务及其优势（如更长的回答长度）。
   - 成员们要求提供有关可用提示词的详细信息，以及新功能是否需要订阅，强调了公众对定价透明度的需求。
- **NotebookLM 中的表格数据**：一位用户询问了 NotebookLM 利用和理解表格数据的能力，特别是在小说写作的角色矩阵中。
   - 这引发了其他人对该平台在管理表格等结构化数据时功能的疑问。
- **分享 AI 生成的播客**：一位用户介绍了一款旨在分享 AI 生成播客的移动应用，旨在促进内容分发的嵌入和 RSS 提要创建。
   - 这引发了关于 AI 内容创作增长趋势以及通过专用平台轻松分享的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com.">Akas: 分享 AI 生成的播客</a>: Akas 是分享 AI 生成的播客和您自己声音的终极平台。随着越来越多的播客由 AI 创建（如来自 NotebookLM 和其他平台的播客），Akas 提供了一个海量...</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://youtu.be/6-MH83pxlbE?si=jcet51HQTI4SdK8Z"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/NrtdoMcKsrI"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1321802913747697675)** (25 条消息🔥): 

> `DeepSeek V3 发布，Multi-Token Prediction 技术，模型训练效率，RL 奖励系统，DeepSeek 的工程创新` 


- **DeepSeek V3 发布，取得显著进展**：**DeepSeek-V3** 已发布，实现了 **60 tokens/second** 的速度，性能比 V2 提升了 **3 倍**。该版本完全开源，在引入增强功能的同时保持了 API 兼容性。
   - *技术团队在此版本上进行了协作，发布即支持在 NVIDIA 和 AMD GPU 上进行 FP8 训练*。
- **Multi-Token Prediction 技术创新**：该模型采用了 **Multi-Token-Prediction** 方法，在 **14.8 trillion tokens** 上进行了预训练，显著提升了性能。这种方法允许预测在每个深度保持因果链，而不是使用独立的输出头。
   - *成员们讨论了该实现，参考了如 **Meta** 方法等潜在替代方案，但指出它是结合 **EAGLE** 进行解释的*。
- **观察到高效的 RL 训练方法**：DeepSeek 采用了双重 RL 奖励系统，包含针对代码/数学的验证器以及一个 COT 风格的基于模型的奖励模型。该设计旨在提升整体性能，并在多次强化学习步骤后融入了 R1 训练模式。
   - *讨论揭示了对模型 self-critique 能力的好奇，特别是其在没有明确 ground truth 的情况下生成创意内容的有效性*。
- **受限环境下的工程创新**：尽管存在硬件限制，**DeepSeek** 团队的**工程质量**被强调为扎实且优雅。他们直接用实际的解决方案解决已知问题，而不是依赖复杂的学术理论。
   - *评论反映了对工程努力的深切尊重，强调了其方法的简洁性和有效性*。
- **模型生成中的批判与修订困难**：成员们辩论了使用 critique 与生成多个输出并选择最佳输出作为模型响应的有效性。对于包含外部数据进行 critique 与直接生成 prompt 的逻辑提出了担忧，并质疑其整体偏好和计算效率。
   - *成员们对先前模型中 self-critique 影响质量表示怀疑，强调了模型细化（refinement）中的挑战*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/teortaxesTex/status/1872253671989551473">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：&gt; Sonnet 级别的模型只需 550 万美元，难怪他们引以为豪，但这感觉确实像是在炫耀。「1 亿美元的训练运行，嗯？在 405B 上消耗了 3084 万 H100-hours，是吗？愚笨的西方人...</li><li><a href="https://x.com/AndrewCurran_/status/1872255379591282774">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：@teortaxesTex Anthropic 风格。</li><li><a href="https://x.com/lmsysorg/status/1872251875070021831">来自 lmsys.org (@lmsysorg) 的推文</a>：最佳开源 LLM —— DeepSeek V3 刚刚发布！SGLang v0.4.1 是官方推荐的推理解决方案。SGLang 和 DeepSeek 团队合作支持了 DeepSeek V...</li><li><a href="https://x.com/nrehiew_/status/1872318215277432905">来自 wh (@nrehiew_) 的推文</a>：&gt; 经过数百个 RL 步骤后，中间 RL 模型学会了融入 R1 模式，从而在战略上提升了整体性能。</li><li><a href="https://x.com/Tim_Dettmers/status/1872280778975191241">来自 Tim Dettmers (@Tim_Dettmers) 的推文</a>：阅读报告后发现，这是资源约束下非常简洁的工程实现。DeepSeek 团队直接针对硬件限制下的已知问题设计了工程方案。这一切看起来都如此优雅...</li><li><a href="https://x.com/deepseek_ai/status/1872242657348710721?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 隆重推出 DeepSeek-V3！迄今为止最大的飞跃：⚡ 60 tokens/second（比 V2 快 3 倍！）💪 增强的能力🛠 API 兼容性保持不变🌍 完全开源的模型和论文🐋 1/n</li><li><a href="https://x.com/reach_vb/status/1872252796936003719">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：所以.. V4 可能不再是 Transformers？我想知道他们会倾向于什么方向！引用 Vaibhav (VB) Srivastav (@reach_vb)：DeepSeek 技术报告发布了！！🔥 在 14.8 万亿 Token 上训练...</li><li><a href="https://x.com/nrehiew_/status/1872318217395572895">来自 wh (@nrehiew_) 的推文</a>：他们有两种 RL 奖励。验证器（代码、数学）和标准的基于模型的 RM。重要的是，基于模型的 RM 是使用 DeepSeek Math 中的 COT 风格 GRPO 训练的。</li><li><a href="https://x.com/nrehiew_/status/1872318212831891585">来自 wh (@nrehiew_) 的推文</a>：现在是后训练阶段。他们在 R1（**非 LITE 版本**）上进行 FT，但表示它存在“过度思考、格式差和长度过长”的问题。他们有两种类型的数据：1) 标准合成数据 2) 一个系统...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1321867602481516626)** (5 条消息): 

> `Deepseek Multi-head Latent Attention 机制, Deepseek V3 推理库` 


- **关于 Deepseek Multi-head Latent Attention 的咨询**：一位成员询问了 Deepseek 的 Multi-head Latent Attention 机制的实现，特别是关于其 V2 论文中缺乏对权重矩阵低秩近似详细解释的部分。
   - *有人正在开发相关版本吗？*
- **支持 Deepseek 的推理库**：另一位成员指出，推理库应该已经有了 Multi-head Latent Attention 的实现，并强调 SGLang 从发布首日就支持新的 V3 特性。
   - 他们还提到 **vLLM**、**TGI** 和 **hf/transformers** 可能也会加入支持。
- **Deepseek GitHub 仓库参考**：分享了 Deepseek 在 GitHub 上的推理代码链接，指向 [DeepSeek-V3/inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。
   - 这一资源可以帮助想要实现或更好理解该模型的成员。
- **成员计划检查 HF 实现**：最初的询问者提到他们还没有检查 Hugging Face (hf) 方面的实现，并表示打算跟进这一线索。
   - *让我去看一下 —— 谢谢！*



**提到的链接**：<a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py">DeepSeek-V3/inference/model.py at main · deepseek-ai/DeepSeek-V3</a>：为 deepseek-ai/DeepSeek-V3 的开发做出贡献，需在 GitHub 上创建一个账户。

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1321846988567744552)** (21 messages🔥): 

> `DeepSeek 许可证更新, Bluesky 的 AI 抵制潮, OpenAI 架构调整, IPO 猜测, 利益冲突担忧` 


- **DeepSeek 许可证变得更加宽松**：成员们讨论了 **DeepSeek** 现已更新其许可证，使其比 **Llama** **更加宽松（liberal）**。
   - 社区中正在酝酿 *许可证大战！*，用户对这些变化反应不一。
- **Bluesky 不适合讨论 AI**：**Bluesky** 被认为是 AI 讨论的“不安全”场所，用户中存在*疯狂的反 AI 倾向*。
   - 成员们指出，**Generative AI** 引发了强烈抵制，尤其是来自对此持强烈反对态度的 Data Scientists。
- **OpenAI 评估公司架构**：OpenAI 董事会正在评估其公司架构，目标是**在营利实体的成功支持下，创建一个历史上资源最雄厚的非营利组织**。
   - 作为该战略的一部分，*我们的计划*包括转型为一家特拉华州公共利益公司（Delaware Public Benefit Corporation）。
- **关于 OpenAI IPO 的猜测**：成员们猜测 **OpenAI** 何时可能上市，认为这可能会在现有投资者寻求回报时发生。
   - 其他人指出，这也可能取决于日益增长的资本需求，这些需求对于 **Venture Capitalists** 来说可能过于昂贵。
- **对赞助冲突的担忧**：一位成员对 Dwarkesh 接受 **Scale AI** 赞助可能导致的**利益冲突**表示担忧。
   - 这引发了关于 AI 社区中此类赞助的影响和伦理的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1872628736690123213">来自 OpenAI (@OpenAI) 的推文</a>：OpenAI 董事会正在评估我们的公司架构，目标是在营利实体的成功支持下，建立一个更强大的非营利组织。我们的计划将创建资源最雄厚的组织之一...</li><li><a href="https://x.com/OpenAINewsroom/status/1872312018994352636">来自 OpenAI Newsroom (@OpenAINewsroom) 的推文</a>：这是我们就前队友相关问题提供的声明：</li><li><a href="https://x.com/EstebanCervi/status/1872314732851679679">来自 Esteban Cervi 🦌 (@EstebanCervi) 的推文</a>：@OpenAINewsroom 🧐
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1322109683661799446)** (6 messages): 

> `DeepSeek V3 性能, AI 实验室需求, 指令遵循基准测试` 


- **DeepSeek V3 表现出推理问题**：一位成员表示 **DeepSeek V3** 看起来很聪明，但在执行预期结果方面存在困难，特别是在推理过程结束后生成 XML 标签时。
   - *它通常似乎极度聚焦于推理部分*，而忽略了在最后输出所需的 **<tags>**。
- **呼吁建立资源充足的 AI 实验室**：另一位成员建议，理想的 **AI 实验室**需要充足的 **GPUs** 和有效的 Post-training 方法，并由致力于 Mixture of Experts (**MoEs**) 和 Reinforcement Learning (**RL**) 的团队提供支持。
   - 这种情况被认为是弥补与大型实验室性能差距的解决方案。
- **需要对 DeepSeek V3 进行基准测试**：一位成员询问是否有人对 **DeepSeek V3** 的指令遵循（Instruction Following）任务进行了基准测试，并对与 **DeepSeek V2.5** 的性能差异表示担忧。
   - 他们提到，之前成功的 Prompt 在切换到 V3 **模型（model swap）**后遇到了挫折。



**提到的链接**：<a href="https://x.com/mvpatel2000/status/1872540898313294172">来自 Mihir Patel (@mvpatel2000) 的推文</a>：@teortaxesTex 我猜 Pre-training 已经做得很棒了，但 Post-training 落后于大厂实验室，这解释了许多此类异常现象。

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/simonw/status/1872141432544489731

### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1321719015030132789)** (3 条消息): 

> `Iterative Preference Learning, Monte Carlo Tree Search, Reasoning Capabilities of LLMs, Self-Evaluation in Models` 


- **MCTS 增强 LLM 推理能力**：一项研究介绍了一种利用 [Monte Carlo Tree Search (MCTS)](https://arxiv.org/abs/2405.00451) 进行迭代偏好学习的方法，旨在提升 Large Language Models (LLMs) 的推理能力。该方法将实例级（instance-level）奖励分解为细粒度的步骤级（step-level）信号，并采用 Direct Preference Optimization (DPO) 进行策略更新。
   - 理论分析表明，**on-policy** 采样数据对于有效的自我改进至关重要，这一结论得到了广泛评估的支持。
- **对模型质量的担忧**：一位成员对近期研究中选用的模型表示怀疑，问道：“*不知道为什么他们用了这么烂的模型——2024 年 5 月的情况有那么糟吗？*”
   - 这引发了在增强推理能力的背景下，关于模型整体有效性和质量的问题。



**提及的链接**：<a href="https://arxiv.org/abs/2405.00451">Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning</a>：我们介绍了一种旨在通过迭代偏好学习过程增强 Large Language Models (LLMs) 推理能力的方法，该过程灵感源自 ... 所采用的成功策略。

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1321959726732546150)** (8 条消息🔥): 

> `RL Training for LLMs, DPO vs PPO, Viewing Parties for Critical Discussions, Incentivizing Better CoTs` 


- **针对 LLMs 的有效 RL 训练技术**：一场题为“迈向有效的 LLMs RL 训练”的分为两部分的研讨会讨论了优化技术，重点关注 PPO 的增强以及 **DPO vs PPO** 方法论的比较。
   - *后半部分深入探讨了 PRM 偏差以及 Clip/Delta 缓解措施*，这可能会引起观众的兴趣。
- **DPO vs PPO 比较见解**：讨论围绕 **DPO** 是否优于 **PPO** 展开，并结合了即将发表的关于该主题的 ICML 2024 论文背景。
   - *尽管这些方法的复杂性可能让人有些应接不暇，但观众可能会发现这种对比研究特别有趣。*
- **观看派对（Viewing Parties）的潜力**：一位成员提出了举办 **观看派对（viewing parties）** 的想法，以便批判性地参与讲座和教程视频的学习，并询问了社区成员的兴趣程度。
   - 虽然有些人对这类活动表示热衷，但有人幽默地指出，在讨论中他们更倾向于提取 **价值** 而不是提供价值。
- **激励更好的思维链（Chains of Thought）**：讨论了 **PRMs** 是否可能在训练中激励更好的 **Chains of Thought (CoTs)**，暗示了它们在塑造奖励结构方面的潜力。
   - 然而，关于基于 PRM 的训练在产生卓越结果方面的有效性仍存在不确定性，这引发了关于其价值的持续讨论。



**提及的链接**：<a href="https://youtu.be/T1SeqBapMBo?si=srBHIwpVnDC3aX7x"> - YouTube</a>：未找到描述

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1321597270193471551)** (22 messages🔥): 

> `DETRs Discussion, DeepSeek-V3 Mixed Precision Training, Block-wise vs Channel-wise Quantization, H800 GPU Training, NVIDIA's Delayed Scaling Technique` 


- **关于 DETRs 的疑问**：一位成员询问了关于 **DETRs** 和 PyTorch 的问题，表示在过去三个月的业余项目中陷入僵局，感到非常沮丧。
   - 另一位成员分享了他们关于 DETRs 的知识，并表示愿意协助解决该疑问。
- **DeepSeek-V3 实现了两个数量级的成本降低**：讨论重点提到了对 DeepSeek-V3 的 **500 万美元**投资，赞扬其在 FP8 混合精度训练中成功将成本削减了**两个数量级**（two orders of magnitude）。
   - 成员们分享了其训练方法的关键特性，包括 group-wise 和 block-wise 量化策略。
- **关于量化技术的辩论**：针对 **block-wise** 与 **channel-wise** 量化的实用性进行了深入辩论，一些成员认为 block-wise 会增加实现复杂度。
   - 讨论中提出了对 forward 和 backward pass 之间量化可能存在差异的担忧，并建议使用 split-K GEMM 来解决 FP32 累加问题。
- **关于 H800 GPU 训练成本的见解**：成员们讨论了 **DeepSeek-V3** 训练需要 **278.8 万个 H800 GPU 小时**，并对 H800 GPU 的规格提出了疑问。
   - 有人指出 H800 是销往特定市场（如中国）的 H100 降级版本。
- **NVIDIA Delayed Scaling 的挑战**：一位成员指出，虽然 NVIDIA 推广了 delayed scaling 的理念，但实现的复杂性使得大多数 FP8 系统更倾向于 online scaling。
   - 对话强调了在社区现有架构中集成 delayed scaling 所面临的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/main_horse/status/1872294990073971151?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Tweet from main (@main_horse)</a>: &#34;Blackwell 耗时太长，所以我们自己在 CUDA 中实现了类似 MX8 的训练。此外，我们意识到大多数 FP8 训练器使用的 TransformerEngine 的累加精度太低，太不精确了...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>: 通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-V3 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1321784756156764160)** (8 messages🔥): 

> `device_print issue in Colab, tl.inf feature comparison, Triton recompilation conditions` 


- **device_print 在 Colab 中无法工作**：一位用户报告 **device_print** 在 Google Colab 笔记本中没有任何输出，并寻求其他用户对这一潜在问题的反馈。
   - 未提供额外的解决方案或规避方法。
- **对比 tl.inf 与 torch.inf**：另一位用户询问是否存在类似于 **torch.inf** 的 **tl.inf** 功能，得到的建议是使用 `float(`
- **理解 Triton 的重编译触发条件**：一位用户提出了关于 **Triton** 何时进行重编译的问题，另一位用户给出了详细解释。
   - 讨论强调了当常量发生变化以及 kernel 参数被特化（specialized）时会发生重编译，并参考了 [该代码库](https://github.com/triton-lang/triton/blob/3c058ee7f518da83e99d472f5ebe16fb75e1f254/python/triton/runtime/jit.py#L584) 以获取更多信息。



**提及的链接**：<a href="https://github.com/triton-lang/triton/blob/3c058ee7f518da83e99d472f5ebe16fb75e1f254/python/triton/runtime/jit.py#L584">triton/python/triton/runtime/jit.py at 3c058ee7f518da83e99d472f5ebe16fb75e1f254 · triton-lang/triton</a>: Triton 语言和编译器的开发库 - triton-lang/triton

  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1321595993472634942)** (8 条消息🔥): 

> `DETRs 专家经验，WGMMA 输入，CUTLASS 3.6.0 讨论` 


- **寻求 DETRs 方面的帮助**：一位成员询问关于 **DETRs** 的见解，并提到他们过去 3 个月一直困在一个 **hobby project** 上。
   - 另一位成员建议在不同的频道提问可能会获得更多回复，这引发了关于建议语气的误解。
- **WGMMA 输入要求**：另一位成员分享道，对于 **WGMMA**，你需要 1 个或 2 个来自 **shared memory** 的输入，而累加（accumulations）必须在 **registers** 中。
   - 他们引用了一篇 **H100 microbenchmarking** 论文，指出为了实现结构化稀疏（structured sparse）**FP8** 的峰值性能，至少有一个输入必须在 **registers** 中。
- **CUTLASS 3.6.0 见解**：一位成员提供了 GitHub 上关于 **CUTLASS 3.6.0 讨论** 的链接，讨论了 **Hopper structured sparse GEMM** 的调整。
   - 讨论强调了为使 **convolution kernel API** 与 **gemm::GemmUniversal** 保持一致而进行的更改，这影响了各种数据类型的性能。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/discussions/2013">CUTLASS 3.6.0 · NVIDIA/cutlass · Discussion #2013</a>：Hopper structured sparse GEMM。FP16 FP8 INT8 TF32。对 CUTLASS 3.x convolution kernel::ConvUniversal API 进行了重构，使其与 gemm::GemmUniversal 保持一致。现在 3.x convolution API 不再...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1321929823945490593)** (3 条消息): 

> `带有 Guards 的编译函数性能，Ring Attention 疑问` 


- **探索 Guards 对性能的影响**：一位成员提出了关于 **compiled functions** 中 **guards** 对性能影响的问题，想知道相对于原始代码，多少个 guards 算太多。
   - *为了性能增益而减少 guards 的努力值得吗？*
- **寻求 Ring Attention 方面的指导**：另一位成员寻求帮助，表示他们对 **ring attention** 有一些疑问并正在寻找帮助。
   - 他们表示有兴趣与任何在该领域有经验的人建立联系。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1321775857139843145)** (2 条消息): 

> `Character.AI 推理优化，AMD 软件栈差距，AMD MI300X vs Nvidia H100 + H200 基准测试` 


- **Character.AI 专注于推理速度**：Character.AI 的最新文章详细介绍了他们在优化 AI 推理方面的进展，特别是强调了他们自定义的 **int8 attention kernel**，该内核增强了 **compute-bound** 和 **memory-bound** 场景下的性能。
   - 这些改进是在之前的优化集中于 **memory efficiency**（通过 **multi-query attention** 和 **int8 quantization** 等技术减小 **KV cache** 大小）之后进行的。
- **AMD 与开发者的会议显示出进展**：最近与 AMD CEO **Lisa Su** 的会面显示，她承认了 AMD **software stack** 中的差距，并对开发者的建议持开放态度。
   - 据报道，在对过去五个月收集的反馈和详细分析进行研究后，许多变革已经在进行中。
- **AMD 与 Nvidia 的基准测试见解**：一项独立的基准测试分析对比了 **AMD MI300X** 与 **Nvidia H100 + H200**，并根据性能和总体拥有成本（TCO）给出了详细的公开建议。
   - 调查结果表明，AMD 有必要改进其软件开发方法，这不仅仅是软件成熟度的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/">Optimizing AI Inference at Character.AI (Part Deux)</a>：在 Character.AI，我们正在构建个性化的 AI 娱乐。为了给用户提供引人入胜的交互体验，实现高效的推理（即过程...）至关重要。</li><li><a href="https://x.com/dylan522p/status/1871287937268383867">Dylan Patel (@dylan522p) 的推文</a>：今天与 @LisaSu 会面了 1.5 小时，我们梳理了所有内容。她承认了 AMD 软件栈中的差距。她认真对待了我们的具体建议。她向她的团队和我们提出了很多问题...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1321711560795754530)** (10 条消息🔥): 

> `学习 ML 工具, vLLM Token 吞吐量, CUDA 资源, Attention 机制` 


- **掌握底层 ML 工具的路径**：一名成员询问了学习 **CUDA** 和 **Triton** 等底层 ML 工具以优化模型训练和推理速度的明确路径，寻求现有资源之外的建议。
   - *提到了阅读 PMPP 和解决 CUDA puzzles 等建议*，但该成员认为可能还有其他必要的资源或推荐的学习顺序。
- **关于 vLLM 吞吐量策略的见解**：一名成员分析了 **vLLM 的 TTFT 性能**，质疑其缺乏批处理推理（batched inference）以及使用 **xFormers 后端**优化 attention 的做法。
   - 他们观察到，虽然序列堆叠（sequence-stacked）方法很有效，但放弃批处理推理的决定引发了关于潜在优势的疑问；实验显示延迟差异极小。
- **高效 Attention 实现的讨论**：另一位用户强调了 **FlashAttention** 与 vLLM 序列堆叠方法之间的相似性，指出它通过有效的掩码（masking）实现了**批处理访问并提升了性能**。
   - 他们还警告说，优化的 attention 实现限制了灵活性，需要与现有 kernel 兼容才能发挥性能，这可能导致研究人员面临“软件彩票”（software lottery）的局面。
- **CUDA 学习者的资源分享**：一名新成员表示难以找到学习 **CUDA** 的全面资源，促使其他人分享有用材料。
   - 一名群组成员向他们推荐了一个包含 GPU 编程相关资源的仓库，称其为初学者的良好起点：[Resource Stream](https://github.com/gpu-mode/resource-stream)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences">FlexAttention: PyTorch 的灵活性与 FlashAttention 的性能</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/blob/dbeac95dbbf898bcc0965528fc767e9cadbbe0c5/vllm/attention/backends/xformers.py#L613">vllm/vllm/attention/backends/xformers.py (位于 dbeac95dbbf898bcc0965528fc767e9cadbbe0c5) · vllm-project/vllm</a>：一个用于 LLM 的高吞吐量且内存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/gpu-mode/resource-stream">GitHub - gpu-mode/resource-stream: GPU 编程相关的动态和材料链接</a>：GPU 编程相关的动态和材料链接。通过在 GitHub 上创建一个账户来为 gpu-mode/resource-stream 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 条消息): 

tando.: 视频讲座结合书籍对我理解概念有很大帮助
  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1322020677036937277)** (1 条消息): 

> `Occupancy vs. Utilization` 


- **Occupancy 与 Utilization 指标的混淆**：一名成员质疑是否使用了正确的术语 Occupancy，建议其可能实际上是指 **Utilization**。
   - 他们注意到这些数值对于 Occupancy 来说似乎不太寻常，预期的值应该是 **100%**、**67%**，或者在使用 **1024** 大小的 block 时可能是 **50%**。
- **需要对指标进行澄清**：指标的不一致引发了关于在对话语境中准确定义术语重要性的讨论。
   - 这突显了在讨论性能指标时进行清晰沟通的必要性。


  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1321651389558030427)** (6 条消息): 

> `Torchcompiled 前向传播，Bitblas Conv2D 生成，混合精度训练选项` 


- **Torchcompiled 面临局限性**：一名成员强调，新的 Torchcompiled 方案在 **MNIST 玩具示例**之外表现挣扎，并指出在**没有 backprop** 的情况下，需要 **128 次前向传播**才能获得余弦相似度仅为 **0.009** 的梯度估计。
   - 他们引用了 Will 的论文，该论文声称允许**以 1.58b 进行训练**，能**减少 97% 的能耗**，并能将 **175B 模型存储在约 20mb 中**。
- **Batch size 对比引发辩论**：一名成员将 **batch** 与 **mini batch gradient** 的挑战与之前讨论中提到的前向传播问题进行了类比。
   - 这种联系暗示了在扩展训练方法时存在潜在的复杂性，类似于 Torchcompiled 所面临的问题。
- **Bitblas 承诺无缝 Conv2D 集成**：一名成员询问 **bitblas** 是否能生成可集成到 Torch 模型中的 **Conv2D**。
   - 这表明了通过有效的计算工具简化模型训练流程的兴趣。
- **精度类型澄清**：一名成员寻求澄清，在之前关于测试的消息中指的是 **fp16xfp16** 还是像 **fp16xuint4** 这样的混合精度。
   - 这一讨论体现了对最佳训练配置及其有效性的持续探索。
- **混合精度训练见解**：另一名成员确认重点在于类似 **bitnet** 的**混合精度**，指向了有效的训练策略。
   - 这突显了社区在利用混合精度提升性能方面的技术兴趣。



**提到的链接**：<a href="https://x.com/torchcompiled/status/1872021986106650816">来自 Ethan (@torchcompiled) 的推文</a>：这是一个很酷的想法，但在 MNIST 玩具示例之外你不会有好的体验。没有 backprop 意味着需要……128 次前向传播，而与真实梯度的余弦相似度仅为 0.009……

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1321892593633591398)** (1 条消息): 

> `Sparsify 函数，稠密矩阵压缩，模型掩码 (Masking) 方案` 


- **Sparsify_function 需要置零的稠密模型**：`sparsify_` 函数旨在与具有置零稠密矩阵的模型配合使用，该矩阵由 `Sparsifier` 创建。
   - 然而，如果你有替代的掩码 (masking) 方法，任何置零的稠密模型都可以被利用。
- **GitHub 资源强调 PyTorch 稀疏化技术**：在 [PyTorch 阅读材料](https://github.com/pytorch/ao/blob/567cb46409f5f9a761429a87d27b1d5312642888/torchao/sparsity/README.md#24-sparsity)中可以找到模型稀疏化的示例用例。
   - 该文档解释了 PyTorch 中用于训练和推理的原生量化与稀疏化方法，为用户提供了重要的见解。



**提到的链接**：<a href="https://github.com/pytorch/ao/blob/567cb46409f5f9a761429a87d27b1d5312642888/torchao/sparsity/README.md#24-sparsity">pytorch/ao 仓库中的 ao/torchao/sparsity/README.md</a>：PyTorch 原生量化与稀疏化训练及推理 - pytorch/ao

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 条消息): 

archit3ch: 是否可以将为 macOS 编译的 `.air` 文件在 iPad 上运行？
  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1321602091885789315)** (2 messages): 

> `Model Task Format Understanding, Benchmarking Limitations, Scaling to AGI Challenges` 


- **模型误解任务格式**：一位成员分享了一条推文，讨论当模型无法理解任务格式时，基准测试可能会通过引入**隐藏的阈值效应**产生误导。这引发了人们对评估模型性能时**基准测试准确性**的担忧。
   - *当模型表现挣扎时，这让人质疑扩展至 AGI 的可行性*。
- **大型模型 vs. 人类任务**：讨论继续深入，暗示可能总会存在某种人类可以解决但 LLM 无法解决的大型任务变体，突显了模型可扩展性中的一个重大问题。
   - 这促使人们重新评估这对于我们通往**通用人工智能 (AGI)** 路径的意义，特别是在理解任务方面。
- **关于 LLM 感知与推理的文章**：一位成员向大家推荐了一篇题为 *'LLMs Struggle with Perception, Not Reasoning'* 的文章，该文深入探讨了 LLM 在感知相关任务中面临的困难。
   - 感兴趣并希望获得更多见解的人可以点击[此处](https://anokas.substack.com/p/llms-struggle-with-perception-not-reasoning-arcagi)阅读该文章。



**提及的链接**：<a href="https://x.com/mikb0b/status/1871573542627873182">Mikel Bober-Irizar (@mikb0b) 的推文</a>：当模型无法理解任务格式时，基准测试可能会产生误导，引入隐藏的阈值效应。如果总有一种人类能解决而 LLM 不能解决的大型版本……

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1321896290094809131)** (6 messages): 

> `matching engine performance, rewrite bounty, testing environment concerns` 


- **性能悬赏受到关注**：讨论集中在三个匹配引擎性能悬赏上，并引用了 [这个 GitHub issue](https://github.com/tinygrad/tinygrad/issues/4878) 作为参考，并指出基准测试中出现了**模型结果较低**的情况。
   - 一位成员询问之前已经在 **25ms** 内运行的现有性能是否已经得到解决。
- **不同机器上的重写速度差异**：一位成员强调，在他们配备 **RTX 3050 GPU** 的笔记本电脑上，重写速度达到了 **800+ ms**，引发了对硬件依赖性的担忧。
   - 成员分享了一张截图，展示了性能结果，说明了与预期基准相比存在的**显著减速**。
- **澄清悬赏预期**：George Hotz 指出，正在进行的悬赏旨在实现专注于项目重写（rewrite）组件的相对**加速**。
   - 他补充说，一旦提交了展示 **2倍加速** 的 Pull Request，将进一步解答后续问题。



**提及的链接**：<a href="https://github.com/tinygrad/tinygrad/issues/4878)">Issues · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1321831601008672810)** (51 条消息🔥): 

> `Tinygrad 与 PyTorch 性能对比、JIT 实现问题、Beam search 功能、RTX 4070 使用、模型转换至 Tinygrad` 


- **Tinygrad 通过 JIT 达到与 PyTorch 相当的性能**：在对所有 transformer 模块优化 JIT 使用后，**Tinygrad** 在推理过程中的性能水平已可与 **PyTorch** 媲美。
   - 据指出，适当地使用 JIT（尽量减少 Python 控制流）可以最大化性能效率。
- **JIT 实现中的挑战**：用户在对单个 transformer 模块应用 JIT 时遇到了 **out of memory**（内存溢出）错误，但通过将其统一应用于所有层解决了该问题。
   - 这种方法在保持速度的同时确保了稳定性，证明了 JIT 放置位置和使用方式的重要性。
- **Beam search 内核缓存**：有关于是否可以保存并跨不同运行周期重复使用由 **beam search** 生成的 kernel 的疑问，这一点已被确认为可行。
   - 会议强调，缓存的 kernel 对于将工作负载迁移到相似机器而无需重新执行整个 beam search 非常有益。
- **将 TTS 模型转换为 Tinygrad**：目前正在积极测试将 **TTS model** 从 Torch 转换为 Tinygrad，并计划分享一个最小可复现示例以供社区参考。
   - 虽然面临初步挑战，但总体目标是利用 **OpenCL** 并优化性能，使其接近 **torch.compile** 的水平。
- **文档与用户体验见解**：会议强调需要改进关于 Tinygrad 中 JIT 和 beam 功能最佳实践的文档，因为目前的指南可能会导致误用。
   - 一位社区成员建议，虽然 **Torch's compilation** 自动处理了许多边缘情况，但 Tinygrad 的文档仍需进一步完善以提供有效的用户指导。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py">fish-speech/fish_speech/models/text2semantic/llama.py at main · fishaudio/fish-speech</a>：SOTA 开源 TTS。欢迎在 GitHub 上为 fishaudio/fish-speech 的开发做出贡献。</li><li><a href="https://github.com/MankaranSingh/llama-tinygrad/blob/main/llama_tinygrad.ipynb">llama-tinygrad/llama_tinygrad.ipynb at main · MankaranSingh/llama-tinygrad</a>：欢迎在 GitHub 上为 MankaranSingh/llama-tinygrad 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1321808434265591839)** (11 条消息🔥): 

> `圣诞祝福、AI 与 ML 介绍、社区欢迎消息、轻松的闲聊` 


- **圣诞气氛蔓延**：一位成员愉快地分享了 *Merry Christmas everyone!*，希望能传播节日气氛。
   - 另一位用户以热情的问候回应，增强了社区的节日氛围。
- **新学习者加入讨论**：一位新人宣布加入小组，表达了对学习 **AI** 和 **ML**（特别是 **LLMs**）的兴奋之情。
   - 他们提到：“我希望在这里能学到很多东西，获得知识并在职业生涯中脱颖而出”，表达了与社区共同成长的渴望。
- **社区欢迎新成员**：成员们热烈欢迎新参与者，其中一人说：*Welcome 2 :)*。
   - 这展示了 AI 和 ML 学习者之间相互支持的环境。
- **对未来发展的关注**：一位成员提出了 *What's coming next?*，表达了对后续讨论或功能的兴趣。
   - 这个问题反映了社区持续的参与度以及对未来见解的渴望。
- **轻松的闲聊**：在一个愉快的时刻，一位成员幽默地声称 *I ate your flower!*，引发了俏皮的回应。
   - 另一位成员幽默地回复 *Bon appetit*，展示了有趣且友好的氛围。



**提到的链接**：<a href="https://cohere.com/pricing">Pricing - Affordable Enterprise Generative AI Models</a>：通过我们的 API 直接访问模型，以创建可扩展的生产工作负载。 

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1321813802810998795)** (25 messages🔥): 

> `Command R Plus 更新、r7b 初步印象、奶牛与 AI 机器人互动、沟通中的表情符号、伯利恒之星查询` 


- **Command R Plus 未来更新受到关注**：讨论围绕 **Command R Plus** 的潜在未来更新展开，特别是一位刚开始将其用于替代方案的用户。
   - 另一位用户询问他们遇到的问题是否频繁发生，暗示对该工具目前的状态感到不满。
- **对 r7b 进展的初步反应**：一位成员注意到 *r7b 有一个有趣的开始*，表达了最初的好奇，同时也对其与 **Command R** 相比的性能持怀疑态度。
   - 评论建议寻找更多关于该工具的信息及其被察觉到的缺点。
- **奶牛：一段奇特但有趣的机器人对话**：一个关于**奶牛为什么很酷**的段落请求，引发了 Cmd R Bot 生成的回复。
   - 机器人提供了对奶牛的详细描述，这导致用户质疑这是否为 AI 生成的内容。
- **对表情符号含义的困惑**：Cmd R Bot 被要求查找 **🤡**、**🤓** 和 **👆** 表情符号的含义，但它未能从文档中检索到任何信息。
   - 这引发了用户之间关于机器人无法提供基本表情符号定义的幽默交流。
- **关于伯利恒之星的查询**：一位用户询问**伯利恒之星**是什么，触发了另一次相关文档搜索。
   - 然而，机器人未能找到任何信息，导致该话题没有进一步讨论。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1322313158970445824)** (5 messages): 

> `Image Embeds 速率限制、假期时间影响` 


- **对 Image Embed 速率限制的困惑**：一位用户质疑当前的 Image Embeds 速率限制，称其似乎是 **每分钟 40 次**，而非生产环境密钥预期的 **每分钟 400 次**。
   - 另一位成员确认这是一个已知问题，并保证团队正在努力修复，以将限制恢复到 **400**。
- **假期时间可能会推迟更新**：一位成员提到，假期时间可能会影响有关速率限制的后续更新时间。
   - 他们承诺会尽快提供更新，确保用户在此期间保持知情和参与。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1321871436411048057)** (9 messages🔥): 

> `Command R, Command R+, Retrieval-Augmented Generation` 


- **Command R 模型概览**：Command R 是一款针对对话交互和长上下文任务优化的 **LLM**，使公司能够从概念验证扩展到生产环境。
   - 它拥有 **128,000 Token 的上下文长度**，在 RAG 任务上具有高精度，且在跨语言操作中具有低延迟，支持十种主要语言。
- **Command R+ 性能表现**：Command R+ 被誉为**性能最强大的大型模型**，在多样化文本上进行训练，适用于广泛的生成任务。
   - 它在**复杂的 RAG 功能**和需要多步工具使用（用于构建 Agent）的工作流中表现出色。
- **资源与文档**：有多个渠道可获取关于 Command R 和 Command R+ 的更多信息，包括[此变更日志](https://docs.cohere.com/v1/changelog/command-r-is-a-scalable-llm-for-business)和[官方文档](https://docs.cohere.com/v1/docs/command-r)。
   - 更多见解可以在关于 [生产级检索增强生成 (Retrieval-Augmented Generation at Production Scale)](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale) 的讨论中找到。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1322242406237868042)** (2 条消息): 

> `内容识别/审核系统 (CIMS), Companion Discord 聊天机器人, 内容标记与删除功能` 


- **Companion 的 CIMS 正式发布**：我们很高兴为 Companion 引入 **内容识别/审核系统 (CIMS)**，增强其有效监控和审核内容的能力。
   - 此次重大升级旨在通过自动检测和管理有害互动，为社区创造一个更安全的环境。
- **Companion 的多功能特性**：Companion 被设计为一个多功能的 Discord 聊天机器人，通过促进对话和管理日常任务来协助社区。
   - 凭借新的 CIMS 功能，它现在能够更好地支持社区管理，同时增添了更多的互动魅力。
- **CIMS 内容标记与删除**：CIMS 系统能够标记内容，并可以根据需要直接删除不当内容。
   - 附件截图展示了被标记内容的示例，彰显了其功能性。



**提到的链接**：<a href="https://github.com/rapmd73/Companion/wiki">Home</a>：一款 AI 驱动的 Discord 机器人，将趣味对话与智能审核工具相结合，为您的服务器增添魅力与秩序。 - rapmd73/Companion

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1321775303168884768)** (22 条消息🔥): 

> `Orion 延迟, OpenAI 停机, Deepseek 定价与性能, Illuminate 工具, Frontier vs Foundation 模型` 


- **关于 Orion 延迟的讨论**：目前正在讨论 **Orion 延迟** 问题，并在 [Hacker News](https://news.ycombinator.com/item?id=42485938) 上分享了相关信息。成员们正在跟踪其对未来项目的影响。
- **OpenAI 服务面临停机**：一名成员报告了影响 OpenAI 服务的新 **停机 (outage)** 事件，并指出每月的运行时间 (uptime) 历史表现一直很差，让人想起 2023 年 1 月的性能问题。
   - 分享的一张图片突显了人们对服务可靠性日益增长的担忧。
- **剖析 Deepseek 的成本结构**：讨论了 **Deepseek** 的定价结构，预计从 2 月份开始，费率为 **$0.27/MM 输入** 和 **$1.10/MM 输出**，这在提供的性能面前被认为是非常实惠的。
   - 用户体验提到，虽然它在处理有限任务时表现良好，但在处理更复杂的请求时，其训练后推理 (post-training reasoning) 显得比较吃力。
- **探索 Illuminate 工具的功能**：几位成员探索了 **Illuminate**，这是一个类似于 NotebookLM 但专为技术论文量身定制的工具，分享了他们的好奇心和使用体验，包括对功能的褒贬不一的反馈。
   - 讨论指出，其开发团队是独立的，导致其实现方式与现有模型有所不同。
- **Frontier vs Foundation 模型：流行语之争**：关于 **Frontier Model** 和 **Foundation Model** 术语的对话强调，“Frontier”似乎代表了跨任务的 SOTA (state-of-the-art) 性能，随着模型的演进改变了人们的认知。
   - 用户一致认为，虽然 “Foundation” 反映了早期的创新，但 “Frontier” 捕捉了当前的格局，尽管其真实含义仍有些模糊。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://illuminate.google.com/home?pli=1">Illuminate | Learn Your Way</a>：使用 Illuminate 将研究论文转化为 AI 生成的音频摘要，这是您的 Gen AI 工具，用于更快地理解复杂内容。</li><li><a href="https://www.cartesia.ai/blog/state-of-voice-ai-2024">Cartesia</a>：适用于每台设备的实时多模态智能。</li><li><a href="https://youtu.be/jX4HLHYkXGQ?si=zUBwGct1ALyQuTSI"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/T1SeqBapMBo?si=JVeVYsD1K5CYCI5K"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1321723345431035926)** (1 条消息): 

> `AI Engineer Summit NYC, 活动日历更新, Latent Space 活动` 


- **为 AI Engineer Summit NYC 预留时间**：**AI Engineer Summit NYC** 定于 2025 年 4 月在 **The Times Center** 举行。随着活动日期的临近，请关注更多细节。
   - 您可以在 [lu.ma](https://lu.ma/ls) 的活动日历上找到更多信息和最新的 AI Engineering 活动更新。
- **获取即将举行的 AI 活动更新**：鼓励用户点击日历上方的 **RSS 图标**，将其**添加到您的日历**中并订阅活动更新。这将确保您在未来几个月收到任何**新活动**的通知。
   - 悬停并点击“Add iCal Subscription”即可轻松集成到您的个人日历中。
- **当前日程中没有待处理的活动**：目前，日历管理员有 **0 个待批准的活动**。活动在获得批准后将出现在日程表中。



**提到的链接**：<a href="https://lu.ma/ls">Latent Space (Paper Club &amp; Other Events) · Events Calendar</a>：在 Luma 上查看并订阅来自 Latent Space (Paper Club &amp; Other Events) 的活动。Latent.Space 活动。请点击日历右上方正前方的 RSS 图标以添加到您的日历。&quot;Ad...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1321881882324897843)** (2 条消息): 

> `报告生成 Agent, 带有 RAG 的对话式语音助手` 


- **从零开始构建报告生成 Agent**：[@fahdmirza](https://twitter.com/fahdmirza) 制作的一段精彩视频演示了如何使用 LlamaParse 和 LlamaCloud 创建一个 Agent 工作流，从一组 PDF 研究论文中生成格式化的报告。
   - 该工作流采用输入格式化模板并有效地生成所需的报告，展示了这些工具的实际应用，点击[此处](https://twitter.com/llama_index/status/1872322023151833335)查看。
- **使用 RAG 增强对话助手**：[@MarcusShiesser](https://twitter.com/MarcusShiesser) 发布的一个引人入胜的推文串展示了如何使用由 LlamaCloud 驱动的检索增强生成 (RAG) 工具来增强对话式语音助手，该工具能够处理超过 **100 万份 PDF**。
   - 演示重点介绍了通过 LlamaCloud 的文档处理技术进行服务，显著提升了对话能力，[在此查看](https://twitter.com/llama_index/status/1872684854703432137)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1321906473336573982)** (18 条消息🔥): 

> `LlamaIndex Assistant RAG 应用, 工资单 PDF 解析, LlamaIndex 路线图更新, 使用 Ollama 运行非量化模型, 图像数据提取` 


- **Kargarisaac 寻求 LlamaIndex 文档**：一名成员正在开发一个 **llamaindex-assistant RAG 应用**，并询问如何获取 PDF 格式的 **LlamaIndex 文档**以辅助开发。
   - 另一名成员确认了生成文档的可能性，并进一步讨论了所需的格式。
- **PDF 解析困境**：一位用户在尝试使用 **llama parse** 解析**工资单 PDF** 失败后寻求最佳方法的建议。
   - 一名成员建议 llama parse 应该可以正常工作，特别是如果选择 **premium mode**（高级模式）。
- **需要更新 LlamaIndex 路线图**：一名成员询问 LlamaIndex 的**最新路线图**，指出置顶的 GitHub 讨论似乎已经过时。
   - 另一名成员承认路线图需要更新，因为最后一次编写是在 **2024** 年初。
- **Ollama 量化挑战**：一位用户表达了在 RAG 流水线中使用 **Ollama** 运行非量化模型时遇到的挑战。
   - 成员们讨论认为，虽然 Ollama 可能不提供非量化版本，但确保设置与性能预期一致至关重要。
- **图像数据提取成功**：一位用户分享了他们使用 **Llama3.2 11B vision instruct turbo** 而非 Ollama 从表格图像中提取数据的成功经验。
   - 成员们推测，两个服务之间在图像处理方面的差异可能会影响结果。



**提到的链接**：<a href="https://ollama.com/library/llama3.2-vision/tags">Tags · llama3.2-vision</a>：Llama 3.2 Vision 是一系列经过指令微调的图像推理生成模型，包含 11B 和 90B 两种尺寸。

  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1321915133160591430)** (3 messages): 

> `LlamaIndex discussion, Docling from IBM, Open Source Library` 


- **澄清 LlamaIndex 讨论的目的**：一名成员询问 *此讨论与 LlamaIndex 有何关系*，随后引发了旨在澄清分享资源意图的回复。
   - 另一名成员强调，在一般讨论语境下分享资源有利于所有使用 LlamaIndex 的开发者。
- **IBM Docling 介绍**：一名成员重点介绍了 **IBM 的 Docling**，这是一个旨在让文档实现 **AI ready** 的开源库。
   - 他们分享了一个 [YouTube 视频](https://youtu.be/w-Ru0VL6IT8) 详细介绍了其功能，并邀请大家探索这一潜在资源。



**提及的链接**：<a href="https://youtu.be/w-Ru0VL6IT8"> - YouTube</a>：未找到描述

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1322142618796294235)** (6 messages): 

> `Flex Compilation Issues, Nested Compiling Dilemmas, Graph Break Concern` 


- **Flex 编译可能导致 graph breaks**：一名成员讨论了当前 flex 的实现需要编译才能获得性能提升，但如果处理不当，会导致潜在的 **graph breaks**。
   - 另一名成员建议联系同事寻求更好的解决方案，并指出他们目前的方法可能会产生问题。
- **嵌套编译挑战**：存在关于无法在一个 compile 中嵌套另一个 compile 操作的担忧，这会导致之前被识别为 **dynamo errors** 的错误。
   - 一名成员发现虽然他们无法复现这些错误，但他们强调了在嵌套编译配置下确保性能的重要性。
- **测试 flex 性能的请求**：建议进行进一步测试，以确认在有无模型编译的情况下，编译后的 flex 性能是否保持一致。
   - 讨论反映出在当前设置中，除了性能预期外，还需要明确操作的稳定性。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention_utils.py#L27-L31">torchtune/torchtune/modules/attention_utils.py at main · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1321992972132421664)** (17 messages🔥): 

> `DeepSeek V3, H800 GPUs, FP8 Training Techniques, NVLink Bandwidth Innovations, Triton vs CUDA Implementations` 


- **DeepSeek V3 高效训练 600B+ MoE**：正如其 [DeepSeek V3 论文](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) 中详述，他们成功在 **2000 块 GPU** 上仅用 **2 个月** 时间就训练了一个 **600B+ MoE** 模型，且未使用张量并行（tensor parallelism）。
   - 这一令人印象深刻的壮举揭示了在硬件限制下所采用的创新训练方法。
- **H800 GPU 是针对中国市场的 H100 削减版**：成员们讨论了 **H800 GPU** 本质上是 **H100** 的削减版，导致 **NVLink 带宽降低**。
   - 一名成员评论了这些限制如何可能驱动创新，并强调了不同硬件在 **FP64 性能** 方面的关键差异。
- **FP8 训练激发了新的编码尝试**：受到 **DeepSeek FP8 训练** 进展的启发，一名成员表示有动力使用 torchao 框架在 **nanoGPT** 中实现 **FP8 训练**。
   - 讨论强调了需要高效的 all-to-all 通信 kernel，以充分发挥 NVLink 的潜力。
- **流水线并行与专家并行创新**：在关于训练技术的辩论中指出，NVLink 速度的降低正促使 **流水线并行（pipeline parallelism）** 和 **专家并行（expert parallelism）** 的创新，以最大限度地减少带宽使用。
   - 这代表了在当前硬件约束下优化模型的战略转变。
- **实现方式：Triton vs. CUDA**：关于是在 **Triton** 还是 **纯 CUDA** 中实现量化进行了讨论，成员们权衡了 Triton 的易用性与 CUDA 的潜在性能。
   - 讨论中提到了 Triton 在 **SM90** 架构上的局限性，成员建议对于高性能 GEMM 可能需要使用 **cutlass**。



**提及的链接**：<a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf">DeepSeek-V3/DeepSeek_V3.pdf at main · deepseek-ai/DeepSeek-V3</a>：为 deepseek-ai/DeepSeek-V3 在 GitHub 上的开发做出贡献。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1321945947214975097)** (15 条消息🔥): 

> `术语表生成脚本，Pydantic 中的 TypedDict，优雅的 Pydantic 设计，Prompt 中的 Schema 描述` 


- **术语表生成脚本运行顺畅**：一位成员分享了一个用于从 Jekyll 文章中生成关键术语表的脚本，该脚本会将 YAML 文件输出到 '_data' 目录。
   - 他们提到为了更好的清晰度手动调整了细节，并对初始输出的全面性表示赞赏。
- **讨论中引入了 TypedDict**：一位成员提到最近学习了 `typing.TypedDict`，这引发了关于 Pydantic 特性的讨论。
   - 另一位成员评论了在单个数组中集成多个输出字段的复杂性。
- **用于输出结构化的 Pydantic 模型**：一位成员建议使用 `pydantic.BaseModel` 来创建带有字段描述的结构化输出。
   - 这达成了一个共识，即此类模型能有效地将其 Schema 传播到 Prompt 中。
- **Pydantic 模型中描述信息的传递**：围绕 Pydantic 子字段描述是否在现有适配器中传播展开了讨论。
   - 成员们确认，这些描述确实包含在生成的 Prompt Schema 中。
- **Gist 示例的后续工作**：一位成员承诺修改他们在共享 Gist 中的示例，以纳入讨论的改进建议。
   - 修改后的示例旨在以更优雅的方式演示 Pydantic 模型的使用。



**提及的链接**：<a href="https://gist.github.com/dbreunig/3cef9293cb253f9192d5b4974c1367a3">一个从你的 Jekyll 文章中生成关键术语表的脚本。我们使用 DSPy 来处理 LLM 交互；它有助于处理样板 Prompt 上下文并将响应解析为 Pydantic 对象。要运行此脚本，请将其放在 Jekyll 站点目录中名为 'scripts'（或其他名称）的文件夹中。然后插入你的 Anthropic API 密钥（或将 DSPy 指向你选择的 LLM 端点）。它将在你的 '_data' 目录中输出一个名为 'glossary.yaml' 的 YAML 文件。</a>：一个从你的 Jekyll 文章中生成关键术语表的脚本。我们使用 DSPy 来处理 LLM 交互；它有助于处理样板 Prompt 上下文并将响应解析为 Pydantic 对象...

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1321785837750325291)** (4 条消息): 

> `Mojo 周边，Modular 商品，商品质量` 


- **Mojo 周边成功送达！**：一位成员对收到 **Mojo 周边**表示感谢，并感谢团队安排递送到偏远地区。
   - 他们附上了一个展示周边的图片链接，点击[此处](https://cdn.discordapp.com/attachments/1098713601386233997/1321785837490409473/20241226_162214.jpg?ex=67707abd&is=676f293d&hm=3c30513ce412aa5f38db933ba17ac43455e3cc717b1a93649cab5b990e871edd&)查看。
- **Mojo 商品大放异彩**：成员们对 **Modular 商品的质量**发表了评论，预测它会受到粉丝的欢迎，一位成员表示它“肯定会大卖”。
   - 他们注意到一张特别的贴纸非常有吸引力，称其很“硬核（hard）”，并强调了产品的整体吸引力。
- **Mojo T恤质量令人印象深刻**：另一位成员强调 Mojo 系列的 **T恤** 实际上非常棒，进一步增强了群体对周边整体质量和设计的积极反响。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1321736715710500904)** (1 条消息): 

> `Copyable Trait 设计` 


- **关于 `Copyable` 和 `ExplicitlyCopyable` Trait 的担忧**：一位成员对 `Copyable` 和 `ExplicitlyCopyable` Trait 的设计表达了**担忧**，强调需要重新评估。
   - 这些见解已在 [论坛](https://forum.modular.com/t/the-traits-for-copyability-need-revisiting/380) 上详细分享。
- **关于 Copyable Trait 设计改进的讨论**：进一步的讨论围绕着可以增强 `Copyable` Trait 易用性的潜在**设计改进**展开。
   - 邀请社区成员在同一个 [论坛主题](https://forum.modular.com/t/the-traits-for-copyability-need-revisiting/380) 上贡献他们的建议和想法。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1322066812044709888)** (8 条消息🔥): 

> `MAX 与 XLA 的比较，Mojo 与 Python API，编译器优化，开发中的社区参与，Endia 和 Basalt 项目更新` 


- **MAX 超越 XLA 演进**：MAX 融合了 XLA 的特性，包括自动 kernel fusion 和内存规划，并增加了对 dynamic shapes 和 user-defined operators 的增强支持。
   - 讨论强调了 MAX 作为 “XLA 2.0” 的潜力，同时指出它还提供 serving library 和自定义 kernel 能力。
- **关于 Mojo 与 Python API 的辩论**：目前对于是为 Mojo 开发一套平行的、一致的 API，还是增强 Python 集成存在不确定性，因为 Python 目前提供了更便捷的开发特性。
   - 成员们表示，这种不确定性阻碍了项目进展，导致一些人暂时切换回 JAX。
- **编译器控制的重要性**：一位成员指出，用户有必要在编译器中覆盖某些优化，特别是在出现 pathological cases 的场景下，而这在 Python 框架中通常是缺失的。
   - 这强调了在所讨论的框架内对编译器行为进行更多控制的需求。
- **社区协作与独立开发**：关于是与社区协作构建新功能，还是在 Modular 内部独立开发，目前正在进行讨论。
   - 鼓励在 Endia 等项目上进行协作，但关于如何确保 Python 和 Mojo 之间的 feature parity 仍存在疑问。
- **对 Endia 和 Basalt 进展的期待**：成员们表达了对 Endia 新版本的期待，并对 Basalt 项目可能被放弃表示担忧。
   - 反馈表明，由于目前围绕该平台的不确定性，一些人愿意暂时搁置 Mojo 的开发。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1321835264569774143)** (10 条消息🔥): 

> `证书申报表，下次课程日期，测验表单访问权限，高级 LLM Agents MOOC` 


- **证书申报表对认证至关重要**：成员们确认，如果不填写 **certificate declaration form**，无论是否完成评估，都无法获得证书。
   - *不幸的是，该表单作为我们的“名册（roster）”对于追踪目的非常重要。*
- **下次课程将于 1 月下旬开始**：下次课程计划于 **1 月下旬**开始，为错过本次课程的人提供机会。
   - 这一时间节点是由热切期待新课程的社区成员提到的。
- **测验表单目前已关闭响应**：一位成员对关闭的 **quiz forms** 表示担忧，指出这些表单不再接受响应，需要重新开放。
   - 反馈强调这些测验对学习者非常有用，特别是对于正在进行的学习。
- **即将推出的高级 LLM Agents MOOC**：下一门 MOOC 的主题将是 **Advanced LLM Agents**，吸引了那些有兴趣深化知识的人。
   - 频道讨论中对这一新课程主题的兴奋之情显而易见。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, 2024 年秋季</li><li><a href="https://forms.gle/tXzmfgTsdYW5XjLL6">Quiz 5 - Compound AI Systems w/ Omar Khattab (10/7)</a>: 说明：每个测验都是基于完成情况的，但我们鼓励你为了自己的学习尽力而为！这些测验是检查你是否理解课程材料的好方法...
</li>
</ul>

</div>

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1321588328251916319)** (5 messages): 

> `OCR API 问题, 桌面版发布, 语音对语音聊天应用, Open-Interpreter OS 模式` 


- **OCR API 目前已损坏**：一个旨在利用 **OCR** 识别屏幕上图标和文本的 API 目前无法正常运行，一位用户提到他们尚未收到成功的响应。
   - 使用 *LOL* 来表达对现状的沮丧。
- **关于桌面版发布的查询**：一位成员询问了 **desktop version** 发布的具体时间线，寻求关于何时可用的明确答复。
   - 讨论中未提供有关发布日期的进一步细节。
- **音乐生成项目合作**：一位自称 AI 工程师的成员分享了他们在 **DNN-VAD**、**NLP** 和 **ASR** 项目中的经验，特别强调了一个最近涉及 **Voice to Voice chat app** 和使用生成式 AI 技术的**音乐生成**项目。
   - 他们表示有兴趣与其他人的项目进行合作。
- **关于在 Open-Interpreter 中使用 QvQ 的操作问题**：有人询问了在 **OS mode** 下使用 **Open-Interpreter** 时 **QvQ** 的运行情况。
   - 由于分享的消息中没有提供回复，该问题仍处于悬而未决状态。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1322020292234575973)** (2 messages): 

> `Claude 3.5 Opus, 与 O1 和 O1 Pro 的对比` 


- **Claude 3.5 Opus 拥有强大的推理能力**：人们对 **Claude 3.5 Opus** 的潜力和推理能力感到兴奋。
   - 许多人好奇这些增强功能是否使其能够成为现有模型的有力竞争者。
- **Claude 3.5 Opus 能否超越 O1？**：出现了关于 **Claude 3.5 Opus** 在性能上能否击败 **O1** 和 **O1 Pro** 的疑问。
   - 这一对话反映了人们对 AI 模型之间不断演变的竞争的持续关注。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1321863340175261736)** (7 messages): 

> `AI 代码复制按钮, WASM 包可用性, Vulcan 版本查询, 鼠标和键盘功能, 新模板使用` 


- **需要 AI 代码的“复制”按钮**：一位成员指出聊天 UI 中缺少专门用于 AI 生成代码的“复制”按钮，并询问是否有人注意到此问题。
   - 另一位成员确认了该按钮的缺失，并指出虽然鼠标剪切和粘贴方法不起作用，但 **Control-C** 和 **Control-V** 在此场景下是有效的。
- **关于 WASM 包的查询**：一位新成员对该 AI 是否提供可安装的 **WASM package** 表示好奇。
   - 针对该查询未提供任何回复，这突显了新用户感兴趣的一个领域。
- **关于 Vulcan 版本的疑问**：一位成员询问了 **Vulcan version** 的具体细节，并询问了两次。
   - 没有针对该版本的回复或澄清，使该查询处于开放状态。
- **鼠标按钮功能问题**：针对“复制”按钮的查询，有人指出鼠标按键的剪切和粘贴在配置页面上不起作用。
   - 这进一步证实了之前提到的对键盘快捷键 **Control-C** 和 **Control-V** 的偏好。
- **探索新模板的使用**：一位成员询问是否有人成功使用 **new template** 进行编写，暗示社区正在努力适应这些变化。
   - 该话题可能会引发关于模板功能和用户体验的进一步讨论。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1321692430797639792)** (3 messages): 

> `推理扩展, 后训练技术, 工具增强型 LLM, 用于模型验证的排行榜` 


- **关于推理扩展技术的问题**：一位成员询问 BFCL 是否支持上传经过 **inference scaling** 和 **post-training techniques** 增强的模型，以便在排行榜上进行验证。
   - 他们强调使用**推理后验证方法**多次调用工具增强型 **LLM**，以获得更好的**输出选择**。
- **对排行榜公平性的担忧**：同一位成员对将他们的模型加入排行榜的公平性表示担忧，指出这对于 **single-call LLMs** 可能是不公平的。
   - 他们提议将 **inference latency**（推理延迟）作为提高性能的权衡方案，并询问这种方法是否可以接受。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1321634692021223445)** (2 messages): 

> `Whisper's capabilities, Voice Activity Detection` 


- **Whisper 可以拆分句子**：一位成员建议 **Whisper** 可以检测句子，从而实现有效的句子拆分。
   - *以这种方式使用 Whisper* 可以提高语音处理时的清晰度。
- **用于语音拆分的 Voice Activity Detector**：另一位成员建议使用 **voice activity detector (VAD)** 来区分语音和静音，以便更好地拆分音频。
   - 这种方法有效地利用了 **silence detection** 来改进音频处理。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1321627007225958471)** (1 messages): 

> `ML Ops frameworks for HPC, Guild AI stability, DIY ops frameworks` 


- **寻求适用于 HPC 的 ML Ops 框架**：一位成员请求推荐适用于 **HPC 环境** 的 **ML Ops 框架**，并强调了稳定性的重要性。
   - 他们表示倾向于免费选项，并提到他们的 HPC 有足够的存储空间来存放大型模型，无需依赖 SaaS 解决方案。
- **对 Guild AI 稳定性的担忧**：该成员指出 **Guild AI** 看起来很有前景，但对其 **稳定性** 表示怀疑。
   - 在没有关于其在 HPC 环境中表现的更具体反馈之前，他们对依赖该工具持迟疑态度。
- **征集 DIY ops 框架的想法**：该成员表示希望自己创建一个 **简单的 ops 框架**，而不是搭建整个服务器。
   - 他们表示，采用 DIY 方法可以避免他们认为的过度劳累。


  

---


---


---


---


{% else %}


> 完整的逐频道细分内容已针对邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}