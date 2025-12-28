---
companies:
- openai
- meta-ai-fair
- google-deepmind
- yi-large
- nvidia
- groq
- langchain
- jamai
- langsmith
date: '2024-08-07T02:40:09.048705Z'
description: '**OpenAI** 发布了全新的 **gpt-4o-2024-08-06** 模型，拥有 **16k 上下文窗口**，价格比之前的 5
  月版 4o 降低了 **33-50%**。该模型还引入了全新的结构化输出（Structured Output）API，旨在提升输出质量并降低重试成本。


  **Meta AI** 推出了 **Llama 3.1**，这款拥有 **4050 亿参数**的模型在基准测试中超越了 **GPT-4** 和 **Claude
  3.5 Sonnet**，同时 Meta 还扩大了 **Llama Impact Grant** 资助计划。


  **Google DeepMind** 低调发布了 **Gemini 1.5 Pro**，其在 LMSYS 基准测试中表现优于 **GPT-4o**、**Claude-3.5**
  和 **Llama 3.1**，并领跑视觉模型排行榜（Vision Leaderboard）。


  **Yi-Large Turbo** 作为一款高性价比的升级版本亮相，定价为每百万 token 0.19 美元。


  在硬件领域，**John Carmack** 强调了 **NVIDIA H100 GPU** 在处理大规模 AI 工作负载方面的卓越性能；同时，**Groq**
  宣布计划在 2025 年第一季度前部署 **10.8 万个 LPU**。


  新兴的 AI 工具与技术包括 **RAG（检索增强生成）**、用于构建智能体混合（Mixture of Agents）系统的 **JamAI Base** 平台，以及
  **LangSmith** 增强的过滤功能。此外，Google DeepMind 还引入了 **PEER（参数高效专家检索）**架构。'
id: 08835504-dbe9-4ddb-bf22-7fc0232bdb02
models:
- gpt-4o-2024-08-06
- llama-3-1-405b
- llama-3
- claude-3.5-sonnet
- gemini-1.5-pro
- gpt-4o
- yi-large-turbo
original_slug: ainews-gpt4o-august-100-structured-outputs-for
people:
- john-carmack
- jonathan-ross
- rohanpaul_ai
title: GPT-4o 八月更新 + 面向所有人的 100% 结构化输出（GPT-4o 八月版）
topics:
- structured-output
- context-windows
- model-pricing
- benchmarking
- parameter-efficient-expert-retrieval
- retrieval-augmented-generation
- mixture-of-experts
- model-performance
- ai-hardware
- model-deployment
- filtering
- multi-lingual
- vision
---

<!-- buttondown-editor-mode: plaintext -->**Pydantic/Zod is all you need.**

> 2024/8/5-2024/8/6 的 AI News。我们为你检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord（**249** 个频道，**2423** 条消息）。预计节省阅读时间（以 200wpm 计算）：**247 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

又是新前沿模型发布的日子！([Blog](https://openai.com/index/introducing-structured-outputs-in-the-api/), [Simonw writeup](https://simonwillison.net/2024/Aug/6/openai-structured-outputs/)) 

就像我们针对 [4o-mini](https://buttondown.email/ainews/archive/ainews-mini-nemo-turbo-lite-smol-models-go-brrr/) 所做的那样，今天的 **AINews 包含 2 期**，使用完全相同的提示词运行 —— 你正在阅读的这一期所有频道摘要均由 `gpt-4o-2024-08-06` 生成，这是今天发布的最新 4o 模型，具有 [16k context（比之前长 4 倍，但仍少于 alpha Long Output 模型），价格比 4o-May 低 33-50%](https://news.ycombinator.com/item?id=41173964)。

我们正好通过 Instructor 库使用 Structured Output 运行 AINews（进行“chain of thought 摘要”），因此切换模型为我们节省了一些代码行，更重要的是节省了重试的费用（由于 OpenAI 采用了受限语法采样，你不再需要在格式错误的 JSON 上花费任何重试费用或时间）。

 
![image.png](https://assets.buttondown.email/images/1bb029d7-5bf2-4b98-84b9-4b14a067c8bb.png?w=960&fit=max)
 

根据我们的摘要 vibe check 和提示词，新模型似乎绝对优于 4o-May（这里选取了一个示例，但你可以亲自查看今天收到的两封邮件）：

 
![image.png](https://assets.buttondown.email/images/d17daacd-8da0-4f24-824b-921e558af59e.png?w=960&fit=max)
 

并且在大多数情况下优于 4o-mini（我们上次的结论是它与 4o-May 相当，但价格便宜得多）：

 
![image.png](https://assets.buttondown.email/images/a73c68c4-4261-4662-a007-6a50c8531f41.png?w=960&fit=max)
 

撇开适用于所有模型的全新 Structured Output API 不谈，我们认为这次意外的 4o 模型升级是一件好事 —— 4o August 实际上是 GPT 4.6 或 4.7，具体取决于你的计算方式。目前我们还没有关于该模型的任何公开 ELO 或 benchmark 指标，但我们敢打赌这将会是一个深藏不露的热门产品 —— 甚至可能是 Q*/Strawberry 的秘密发布？


---

{% if medium == 'web' %}

**Table of Contents**

[TOC]

{% else %}

**Table of Contents** 和 **Channel Summaries** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型更新与基准测试**

- **Llama 3.1**：Meta 发布了 Llama 3.1，这是一个拥有 4050 亿参数的大语言模型，在[多个基准测试中超越了 GPT-4 和 Claude 3.5 Sonnet](https://twitter.com/DeepLearningAI/status/1820572990121099291)。[Llama Impact Grant 计划](https://twitter.com/AIatMeta/status/1820493232826138946)正在扩大，以支持全球组织基于 Llama 进行构建。
- **Gemini 1.5 Pro**：Google DeepMind [悄然发布了 Gemini 1.5 Pro](https://twitter.com/AlphaSignalAI/status/1820489158747435322)，据报道其在 LMSYS 上的表现优于 GPT-4o、Claude-3.5 和 Llama 3.1，并在 Vision Leaderboard 上排名第一。它在多语言任务和技术领域表现出色。
- **Yi-Large Turbo**：[作为 Yi-Large 的强力且具性价比的升级版推出](https://twitter.com/01AI_Yi/status/1820456064405369335)，输入和输出价格均为每 1M tokens 0.19 美元。

**AI 硬件与基础设施**

- **NVIDIA H100 GPUs**：John Carmack [分享了关于 H100 性能的见解](https://twitter.com/ID_AA_Carmack/status/1820575987714990212)，指出在 AI 工作负载方面，10 万块 H100 的算力比目前所有 3000 万台现役 Xbox 的总和还要强大。
- **Groq LPUs**：Jonathan Ross [宣布计划](https://twitter.com/JonathanRoss321/status/1820501857741246859)在 2025 年第一季度末之前将 108,000 个 LPU 投入生产，并扩大云端和核心工程团队。

**AI 开发与工具**

- **RAG (Retrieval-Augmented Generation)**：[关于 RAG 重要性的讨论](https://twitter.com/LangChainAI/status/1820480883968614799)，旨在整合人类输入并增强 AI 系统的能力。
- **JamAI Base**：一个无需编码即可[构建 Mixture of Agents (MoA) 系统](https://twitter.com/rohanpaul_ai/status/1820450501172809838)的新平台，利用了 Task Optimizers 和 Execution Engines。
- **LangSmith**：LangSmith 中 [Trace 的新过滤功能](https://twitter.com/LangChainAI/status/1820480157746155607)，允许基于 JSON 键值对进行更精确的过滤。

**AI 研究与技术**

- **PEER (Parameter Efficient Expert Retrieval)**：[来自 Google DeepMind 的新架构](https://twitter.com/rohanpaul_ai/status/1820576677011091639)，在 Transformer 模型中使用超过一百万个小型“专家”来替代大型前馈层。
- **POA (Pre-training Once for All)**：一种[创新的三分支自监督训练框架](https://twitter.com/rohanpaul_ai/status/1820450517434150966)，能够同时预训练多种尺寸的模型。
- **Similarity-based Example Selection**：研究表明，使用基于相似性的 In-context 示例选择，可以[显著提升低资源机器翻译的效果](https://twitter.com/rohanpaul_ai/status/1820599401464664519)。

**AI 伦理与社会影响**

- **Data Monopoly Concerns**：关于[潜在数据垄断](https://twitter.com/nearcyan/status/1820568015567757688)的讨论，如果从互联网服务下载内容变得非法，可能会导致供应商锁定。
- **AI Safety**：关于 [AI 智能本质和安全措施](https://twitter.com/ylecun/status/1820473019728322949)的辩论，Yann LeCun 对一些常见的 AI 风险叙事提出了反对意见。

**AI 实际应用**

- **Code Generation**：[关于 AI 代码生成有效性的观察](https://twitter.com/alexalbert__/status/1820503813180280964)，例如研究人员在身体受限的情况下使用 Claude 进行编码。
- **Model Selection Guide**：针对各种任务的 [AI 模型选择建议](https://twitter.com/bindureddy/status/1820595715426705611)，包括代码生成、搜索、文档分析和创意写作。

**AI 社区与教育**

- **AI and Games Textbook**：Julian Togelius 和 Georgios Yannakakis [发布了其《AI and Games》教科书第二版的草案](https://twitter.com/togelius/status/1820456967019667695)，寻求社区反馈以进行改进。
- **AI Education Programs**：Google DeepMind [庆祝了 AIMS 的 AI for Science 硕士项目的首批毕业生](https://twitter.com/GoogleDeepMind/status/1820453547227415007)，并提供了奖学金和资源。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. AI 模型的架构创新**

- **[Flux 的架构图 :) 好像没有论文，所以快速看了一下他们的代码。对于理解当前的 Diffusion 架构可能很有用](https://i.redd.it/7n3ix8do9vgd1.png)** ([Score: 461, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1ekr7ji/fluxs_architecture_diagram_dont_think_theres_a/)): Flux 的 **diffusion models** 架构图在没有随附论文的情况下，提供了对当前 diffusion 架构的深入见解。该图表通过检查 Flux 的 **code** 得出，直观地展示了模型的结构和组件，可能有助于理解当代的 diffusion model 设计。

**主题 2. 开源 AI 模型的进展**

- **[为什么没人讨论 InternLM 2.5 20B？](https://huggingface.co/internlm/internlm2_5-20b-chat)** ([Score: 247, Comments: 98](https://reddit.com//r/LocalLLaMA/comments/1ekr75a/why_is_nobody_taking_about_internlm_25_20b/)): InternLM 2.5 20B 在基准测试中表现出 **令人印象深刻的性能**，超越了 **Gemma 2 27B** 并接近 **Llama 3.1 70B**。该模型在 **MATH 0 shot 上取得了卓越的 64.7 分**，接近 **3.5 Sonnet 的 71.1 分**，并且可能通过 **8-bit quantization** 在 **4090 GPU** 上运行。
- **灵光一现：如果我们制作 Magnum 32b 和 12b 的 V2 版本会怎样（剧透：我们做到了！）** ([Score: 54, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1el6g4n/shower_thought_what_if_we_made_v2_versions_of/)): **Magnum-32b v2** 和 **Magnum-12b v2** 模型已经发布，并根据用户反馈进行了改进。这些模型在 [Hugging Face](https://huggingface.co/anthracite-org) 上提供 **GGUF** 和 **EXL2** 格式，开发人员正在寻求用户的进一步反馈以完善模型。
  
  - 用户询问了潜在的 **Mistral-based models**，并讨论了 **32b V1** 模型在 **Koboldcpp** 和 **Textgenui** 等应用中的最佳 **sampler settings**。
  - 该模型的 **预期用途** 被幽默地描述为“**狐狸精研究 (foxgirl studies)**”，而其他人则指出 v1 模型存在 **多语言性能问题**，并推测 **72B** 和 **32B** 版本之间的差异。
  - 一些用户报告了 **12B v2 8bpw exl2** 模型的问题，遇到了 **胡言乱语** 和 **严重的幻觉 (hallucination)**，且不受 prompt templates 或 sample settings 更改的影响。

**主题 3. LLM 的新颖应用与能力**

- **[我们正在制作一款由 LLM 驱动法术和世界生成的游戏](https://v.redd.it/9fbokf4jaxgd1)** ([Score: 413, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1el1et6/were_making_a_game_where_llms_power_spell_and/)): 开发人员正在创建一款利用 **Large Language Models (LLMs)** 进行动态 **法术和世界生成** 的游戏。这种方法允许根据玩家输入创建 **独特的法术** 和 **程序生成的模型世界**，从而可能提供更加个性化和身临其境的游戏体验。虽然没有提供关于游戏机制或发布的具体细节，但该概念展示了 AI 在游戏开发中的创新应用。
- **Gemini 1.5 Pro Experimental 0801 作为一个闭源模型，其去审查程度令人惊讶** ([Score: 54, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1el4wus/gemini_15_pro_experimental_0801_is_strangely/)): Google 的 **Gemini 1.5 Pro Experimental 0801** 模型在添加到 **UGI-Leaderboard** 时展示了令人惊讶的去审查能力。在安全设置设为“**Block none**”并使用特定 system prompt 的情况下，该模型愿意对有争议且可能非法的查询提供答复，尽管其意愿（**W/10**）略低于排行榜上的平均模型。
  
  - 用户报告了 **Gemini 1.5 Pro Experimental 0801** 去审查能力的参差不齐的结果。一些人发现它拒绝了所有请求，而另一些人则成功诱导它回答了关于 **盗版、自杀方法和毒品制造** 的查询。
  - 该模型在处理性内容时表现出不一致的行为，拒绝了一些请求，但在以不同方式提示时同意编写 **色情故事**。用户注意到在测试这些功能时其 **Google 账号** 可能面临的风险。
  - 在 **SillyTavern** 的暂存分支中，Gemini 1.5 Pro Experimental 0801 显示出比其他版本更少的过滤。用户还发现它比普通的 Gemini 1.5 Pro **更聪明**，后者有时被描述为“精神分裂 (schizo)”。

**主题 4. 主要 AI 公司的领导层变动**

- **Sam 会为了关停 Llama 4 而“恐吓”山姆大叔（美国政府）吗？** ([Score: 59, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1el3q46/will_sam_spook_uncle_sam_in_order_to_shut_down/)): Sam Altman 可能向政府监管机构进行 **GPT-5 的私下演示**，据推测这可能会影响对**开源 AI 发展**（特别是 **Llama 4**）的限制。这种假设情景表明，Altman 可能会故意惊吓监管机构，以限制来自开源模型的竞争，从而在不断发展的 **open LLM era** 中为其公司争取优势。
  
  - **Meta** 可能会在美国境外训练开源 LLM，而 **Mistral** 也在提供具有竞争力的模型。然而，**EU AI Act** 引入了大量的文档要求，可能会阻碍欧洲生成式模型的发展。
  - 出人意料的是，**Zuckerberg** 正在倡导保护开源 AI，而政府也表示不会限制开源 AI。有人认为，这种立场有利于所有非 OpenAI 实体挑战 OpenAI 被认为的垄断地位。
  - **FTC** 负责人 **Lina Khan** 据报道支持 open weight 模型，这可能会减轻对限制的担忧。监管界似乎将 AI 软件视为类似于 90 年代初的互联网而非加密技术，这表明监管方式可能不会那么严格。

- **[OpenAI 联合创始人 Schulman 和 Brockman 退居二线。Schulman 离职加入 Anthropic。](https://finance.yahoo.com/news/openai-co-founders-schulman-brockman-010542796.html?guccounter=1)** ([Score: 317, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1el5jyj/openai_cofounders_schulman_and_brockman_step_back/)): OpenAI 联合创始人 **Adam D'Angelo** 和 **Ilya Sutskever** 正在退居二线，而 **Schulman** 离职加入 **Anthropic**。这一进展是在最近围绕 **Sam Altman** 被短暂解雇并复职为 CEO 的争议之后发生的，该事件导致了 OpenAI 内部的重大变化。这些关键人物的离职标志着 OpenAI 领导结构以及潜在战略方向的显著转变。
  
  - 对 **OpenAI** 内部问题的担忧，以及对 **GPT5/strawberry/Q\*** 开发问题或 **Sam Altman** 领导风格的猜测。一些用户将离职归因于每个人的不同因素。
  - 讨论了 OpenAI 关键人物（**Schulman**、**Brockman**、**Altman**）名字的巧合，并对 AI 相关的姓氏进行了幽默评论，还将其与 **Hideo Kojima** 的角色命名风格进行了比较。
  - 用户对 **Anthropic** 表达了复杂的情绪，在称赞 **Claude** 的同时，也批评了该公司被认为的“**自大情结 (megalomaniac complex)**”和审查做法。关于在 AI 行业中“商人”与现有领导层相比的优缺点的辩论随之展开。

## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研发**

- **Google DeepMind 推进多模态学习**：一篇 [Google DeepMind 的论文](https://arxiv.org/html/2406.17711v1)展示了如何通过联合样本选择（joint example selection）进行数据策展，从而加速多模态学习 (/r/MachineLearning)。
- **Microsoft 的 MInference 加速长文本推理**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490)能够在保持准确性的同时，实现长文本任务中高达数百万个 token 的推理 (/r/MachineLearning)。
- **利用网络策展的人格（Personas）扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用从网络数据中策展的 10 亿个人格来生成多样化的合成数据 (/r/MachineLearning)。
- **据传 NVIDIA 正在抓取海量视频数据**：[泄露文件显示](https://www.reddit.com/r/singularity/comments/1eks9mi/leaked_documents_show_nvidia_scraping_a_human/)，NVIDIA 每天抓取相当于“一个人一生长度”的视频来训练 AI 模型 (/r/singularity)。

**AI 模型发布与改进**

- **Salesforce 发布 xLAM-1b 模型**：拥有 10 亿参数的 [xLAM-1b 模型在函数调用（function calling）方面实现了 70% 的准确率](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)，尽管体积较小，但超越了 GPT-3.5 (/r/LocalLLaMA)。
- **Phi-3 Mini 更新函数调用功能**：Rubra AI 发布了更新后的 [具有函数调用能力的 Phi-3 Mini 模型](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，其性能可与 Mistral-7b v3 竞争 (/r/LocalLLaMA)。

**AI 行业新闻与动态**

- **OpenAI 多位核心高管离职**：[三位关键领导者正离开 OpenAI](https://www.reddit.com/r/OpenAI/comments/1el4thx/greg_brockman_john_schulman_and_peter_deng_leave/)：总裁 Greg Brockman（长期休假）、John Schulman（加入 Anthropic）以及产品负责人 Peter Deng (/r/OpenAI, /r/singularity)。
- **Elon Musk 对 OpenAI 提起诉讼**：[Musk 已对 OpenAI 和 Sam Altman 提起新诉讼](https://www.reddit.com/r/singularity/comments/1ekqgo5/elon_musk_files_new_lawsuit_against_openai_and/) (/r/singularity)。
- **Anthropic 创始人讨论 AI 发展**：一位 Anthropic 创始人[表示，即使现在停止 AI 研发](https://www.reddit.com/r/singularity/comments/1eknsvx/anthropic_founder_says_if_we_stopped_ai/)，现有能力仍能带来数年甚至数十年的持续改进 (/r/singularity)。

**神经技术与脑机接口**

- **Elon Musk 对 Neuralink 的最新主张**：Musk [预测脑芯片植入患者将在 1-2 年内超越职业玩家](https://www.reddit.com/r/singularity/comments/1ekyph0/elon_musk_says_his_brainchip_patients_will_soon/)，并谈到赋予人类“超能力” (/r/singularity)。

**迷因与幽默**

- 一个[对比某记者 11 年间对 AI 截然不同观点](https://www.reddit.com/r/singularity/comments/1el935j/same_journalist_11_years_apart/)的迷因 (/r/singularity)。
- 一张[对 2030 年进行幽默推测的图片](https://www.reddit.com/r/singularity/comments/1el2cjn/sometime_in_2030/) (/r/singularity)。

---

# AI Discord 综述

> 摘要的摘要之摘要

## Claude 3 Sonnet

**1. LLM 进展与基准测试**

- **Llama 3 登顶排行榜**：来自 Meta 的 [**Llama 3**](https://lmsys.org/blog/2024-05-08-llama3/) 在 **ChatbotArena** 等排行榜上迅速崛起，在超过 50,000 场对决中超越了 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
  - 示例对比强调了模型在 **AlignBench** 和 **MT-Bench** 等基准测试中的表现，其中 **DeepSeek-V2** 拥有 **236B 参数**，并在某些领域超越了 GPT-4。
- **新型开源模型推动最先进水平**：来自 IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 等新型模型增强了代码任务的指令遵循（instruction following）能力，而 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 则推出了拥有 **236B 参数** 的巨型模型。
  - **AlignBench** 和 **MT-Bench** 的排行榜对比显示 **DeepSeek-V2** 在某些领域优于 GPT-4，引发了关于不断演进的最先进技术水平（state of the art）的讨论。

**2. 模型性能优化与推理**

- ****量化技术减小模型占用空间****：**[Quantization]** 技术（如 **AQLM** 和 **QuaRot**）旨在实现在保持性能的同时，在单个 GPU 上运行大语言模型 (**LLMs**)。
  - 示例：[AQLM 项目](https://github.com/Vahe1994/AQLM) 展示了在 RTX3090 GPU 上运行 **Llama-3-70b** 模型。
- \*\***DMC 提升吞吐量达 370%** \*\*: 通过 **Dynamic Memory Compression (DMC)** 等方法**提升 Transformer 效率**的努力显示出在 **H100 GPUs** 上将吞吐量提高多达 370% 的潜力。
  - 示例：`@p_nawrot` 的 [DMC 论文](https://arxiv.org/abs/2403.09636) 探讨了 DMC 技术。
- ****使用 Consistency LLMs 进行并行解码****：**[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 等技术探索了并行 Token 解码，以降低推理延迟。
  - **SARATHI 框架**还通过采用分块预填充（chunked-prefills）和解码最大化批处理（decode-maximal batching）来解决 LLM 推理中的效率低下问题，从而提高 GPU 利用率。
- ****CUDA Kernels 加速操作****：讨论集中在**优化 CUDA 操作**上，例如融合逐元素操作（element-wise operations），使用 **Thrust 库的 `transform`** 来实现接近带宽饱和的性能。
  - 示例：[Thrust 文档](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each) 重点介绍了相关的 CUDA kernel 函数。

**3. 开源 AI 框架与社区努力**

- ****Axolotl 支持多种数据集格式****：[**Axolotl**](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 现在支持用于指令微调和预训练 LLMs 的多种数据集格式。
  - 社区对 Axolotl 在开源模型开发和微调方面不断增强的能力表示赞赏。
- ****LlamaIndex 集成吴恩达课程****：**[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 宣布与吴恩达的 DeepLearning.ai 合作推出一门关于构建 Agentic RAG 系统的新课程。
  - 该课程强调了 LlamaIndex 在开发企业级检索增强生成 (RAG) 系统中的作用。
- ****RefuelLLM-2 针对“乏味”任务进行优化****：**[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 已开源，声称是处理“乏味数据任务（unsexy data tasks）”的最佳 LLM。
  - 社区讨论了 RefuelLLM-2 在不同领域的性能和应用。
- ****Mojo 预告 Python 集成和加速器****：**[Modular 的深度解析](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 预告了 Mojo 在 Python 集成和 AI 扩展（如 `_bfloat16_`）方面的潜力。
  - 像带有脉动阵列（systolic arrays）的 PCIe 卡这样的定制加速器也被认为是 Mojo 开源发布后的未来候选方案。

**4. 多模态 AI 与生成模型创新**

- ****Idefics2 和 CodeGemma 提升能力****：**[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。
  - 这些新的多模态模型展示了在对话式 AI 和代码生成等领域的进展。
- ****Phi3 将 AI 聊天机器人带入 WebGPU****：**[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型通过 WebGPU 将强大的 AI 聊天机器人带入浏览器。
  - 这一进步使得通过 WebGPU 平台进行私密的设备端 AI 交互成为可能。
- ****IC-Light 改进图像重打光****：开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于改进图像重打光（relighting）技术。
  - 社区成员分享了在 ComfyUI 等工具中利用 IC-Light 的资源和技术。

**5. 微调挑战与提示工程策略**

- ****Axolotl Prompting 见解****：强调了**提示词设计 (prompt design)** 和使用正确模板（包括文本结束标记 tokens）的重要性，这些因素会影响模型在微调和评估期间的性能。
  - 示例：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47) 展示了提示词工程技术。
- ****用于提示词控制的 Logit Bias****：讨论了**提示词工程 (prompt engineering)** 策略，例如将复杂任务拆分为多个提示词，以及研究 **logit bias** 以更好地控制输出。
  - 示例：[OpenAI logit bias 指南](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api) 解释了相关技术。
- ****用于检索的 <ret> Token****：教导 LLM 在不确定时使用 `<RET>` token 进行**信息检索**，可以提高在低频查询上的表现。
  - 示例：[ArXiv 论文](https://arxiv.org/abs/2404.19705) 介绍了这项技术。

## Claude 3.5 Sonnet

**1\. LLM 进展与基准测试**

- **DeepSeek-V2 在基准测试中挑战 GPT-4**：**DeepSeek-V2** 是一款拥有 236B 参数的新模型，在 **AlignBench** 和 **MT-Bench** 等基准测试中表现优于 GPT-4，展示了大型语言模型能力的重大进步。
  - 该模型的表现引发了关于其对 AI 领域潜在影响的讨论，社区成员分析了其在各种任务和领域中的优势。
- **John Schulman 加入 Anthropic 的战略举措**：OpenAI 联合创始人 **John Schulman** [宣布离职](https://x.com/johnschulman2/status/1820610863499509855) 并加入 **Anthropic**，理由是希望更深入地关注 AI 对齐 (AI alignment) 和技术工作。
  - 此举紧随 OpenAI 最近的重组（包括解散其超级对齐团队），并引发了关于 AI 安全研究与开发未来方向的讨论。
- **Gemma 2 2B：Google 的紧凑型强力模型**：Google 发布了 [Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)，这是一款拥有 2.6B 参数的模型，专为高效的设备端使用而设计，兼容 **WebLLM** 和 **WebGPU** 等平台。
  - 该模型的发布受到了热烈欢迎，特别是它能够在 [Google Colab](https://x.com/reach_vb/status/1819469088890261748) 等免费平台上流畅运行，展示了强大 AI 模型日益增长的可访问性。

**2\. 推理优化与硬件进展**

- **Cublas hgemm 提升 Windows 性能**：**cublas hgemm 库**已实现 Windows 兼容，在 4090 GPU 上实现了高达 **315 tflops** 的性能，而 torch nn.Linear 仅为 166 tflops，显著增强了 AI 任务的性能。
  - 用户报告在 4090 上运行 flux 达到了约 **2.4 it/s**，标志着大型语言模型在消费级硬件上的推理速度和效率有了实质性提升。
- **Aurora 超级计算机剑指 ExaFLOP 里程碑**：位于阿贡国家实验室的 **Aurora 超级计算机**在经过性能优化后，预计将突破 **2 ExaFLOPS**，有望成为全球最快的超级计算机。
  - 讨论强调了 Aurora 独特的 Intel GPU 架构，支持输出 16x8 矩阵的 Tensor Core 指令，引发了对其在 AI 和科学计算应用中潜力的关注。
- **ZeRO++ 大幅削减 GPU 通信开销**：**[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 是一种新的优化技术，有望将大型模型在 GPU 上训练的通信开销降低 4 倍，显著提高训练效率。
  - 这一进展对于分布式 AI 训练设置尤为重要，可能使大规模语言模型的训练更加快速且更具成本效益。

**3\. 开源 AI 与社区协作**

- **SB1047 引发开源 AI 辩论**：一封反对 **SB1047**（AI 安全法案）的公开信正在流传，警告该法案可能通过潜在的禁止开源模型和威胁学术自由，对开源研究和创新产生负面影响。
  - 社区对此存在分歧，一些人支持 AI 安全监管，而包括 Anthropic 在内的其他公司则警告不要扼杀创新，并暗示该法案可能在学术和经济方面产生意想不到的负面后果。
- **Wiseflow：开源数据挖掘工具**：**[Wiseflow](https://github.com/TeamWiseFlow/wiseflow)** 是一款开源信息挖掘工具，旨在高效地从包括网站和社交平台在内的各种在线资源中提取和分类数据。
  - 该工具引起了 AI 社区的兴趣，有人建议将其与 **Golden Ret** 等其他开源项目集成，为 AI 应用创建动态知识库。
- **AgentGenesis 助力 AI 开发**：**[AgentGenesis](https://github.com/DeadmanAbir/AgentGenesis)** 是一款开源 AI 组件库，旨在为开发者提供用于 Gen AI 应用的复制粘贴代码片段，承诺将开发效率提高 10 倍。
  - 该项目采用 MIT 许可证，包含一个全面的代码库，提供 RAG flows 和 QnA bots 模板，并正在积极寻求贡献者以增强其功能。

**4\. 多模态 AI 与创意应用**

- **CogVideoX-2b：视频合成的新前沿**：新款文本生成视频合成模型 **[CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)** 的发布因其根据文本描述生成视频内容的能力而备受关注。
  - 初步评论表明，CogVideoX-2b 在该领域具有与领先模型竞争的实力，引发了关于其潜在应用及对多媒体内容创作影响的讨论。
- **Flux AI 挑战图像生成巨头**：据报道，**Flux AI** 的 'Schnell' 模型在图像生成的一致性方面表现优于 **Midjourney 6**，展示了 AI 生成视觉内容的重大进步。
  - 用户称赞该模型能够生成高度逼真且细节丰富的图像（尽管偶尔会有细微拼写错误），这标志着 AI 生成视觉媒体质量的一次飞跃。
- **MiniCPM-Llama3 推进多模态交互**：**MiniCPM-Llama3 2.5** 现在支持多图输入，并在 OCR 和文档理解等任务中展现出巨大潜力，为多模态交互提供了强大的功能。
  - 该模型的进步凸显了日益增长的趋势，即开发更通用的 AI 系统，能够同时处理和理解包括文本和图像在内的多种输入类型。

## GPT4O (gpt-4o-2024-05-13)

**1. LLM 进展与基准测试**

- **Gemma 2 2B 助力端侧 AI**：Google 的 **[Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)** 通过 **WebLLM** 和 **WebGPU** 技术流畅支持端侧运行。
  - 社区对其易用性表示赞赏，甚至在免费的 [Google Colab](https://x.com/reach_vb/status/1819469088890261748) 上也能运行，展示了其部署潜力。
- **CogVideoX-2b 引燃视频生成领域**：**[CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)** 模型因其在文本生成视频（text-to-video）合成方面的能力而备受关注，可与领先的竞争对手一较高下。
  - 围绕其竞争优势引发了讨论，暗示了其在多媒体应用中的广阔前景。

**2. 模型性能优化与基准测试**

- **INT8 量化引发缩放讨论**：在 INT8 对称量化中，**PyTorch 的 127.5 缩放**策略在 **Qwen2-0.5B** 模型微调中因截断问题导致了发散。
  - 社区探索了替代方案，如 **INT4 量化训练**，以绕过受限范围量化的约束。
- **Bobzdar 测试 GPU 性能**：Bobzdar 在 LM Studio 中使用 **ROCM** 和 **Vulkan** 对 **8700G/780m IGP** 进行的实验显示速度提升了 **25%**，尽管受到超过 20GB 的 **GPU RAM** 限制的挑战。
  - 尽管如此，**llama3.1 70b q4** 的进展显示其处理速度比 CPU 快 **30%**，但在超过 63k 的 context size 稳定性方面仍存在困难。

**3. 微调挑战与集成**

- **Unsloth 在微调和 PPO 集成方面遇到困难**：在 **Unsloth** 中使用 'model.save_pretrained_merged' 进行微调时出现的问题引发了关注，原因是保存方法不一致。
  - 将 Llama3 模型整合到 PPO trainer 中的尝试凸显了在生成输出前进行 'for_inference()' 步骤的必要性，这使集成过程变得复杂。
- **Axolotl 走红引发微调热议**：讨论强调了 **Axolotl** 库是微调 AI 模型的热门选择，同时还有关于保险行业特定应用的咨询。
  - 此外，还出现了关于 **Llama 450b** 托管方案和推理瓶颈的问题，特别是在使用 **vLLM** 等资源时。

**4. 开源 AI 发展与协作**

- **StoryDiffusion，开源版 Sora**：发布了 **StoryDiffusion**，这是 Sora 的开源替代方案，采用 MIT 许可证，但权重尚未发布。
  - 示例：[GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。
- **OpenDevin 发布**：发布了 **OpenDevin**，这是一个基于 Cognition 的 Devin 的开源自主 AI Agent 工程师，并举行了 [webinar](https://lu.ma/fp0xr460)，在 GitHub 上的关注度不断提高。
  - 示例：[GitHub repo](https://github.com/Cognition-AI/OpenDevin)。

## GPT4OMini (gpt-4o-mini-2024-07-18)

**1\. AI 工具中的安装挑战**

- **ComfyUI 和 Flux 的安装困扰**：ComfyUI 和 Flux 的**安装问题**一直困扰着用户，特别是由于不兼容的 **Python** 版本影响了运行。
  - 许多成员对管理不同的 **Python** 环境表示沮丧，尽管尝试了各种修复方法，但仍导致反复失败。
- **本地 LLM 设置问题**：使用 **Open Interpreter** 设置本地 **LLM** 时导致了不必要的下载，并在模型选择期间引发了 *openai.APIConnectionError*。
  - 用户正在私下协调以解决此问题，这突显了本地模型设置的复杂性。

**2\. 模型性能与优化讨论**

- **Mistral-7b-MoEified-8x 模型效率**：**[Mistral-7b-MoEified-8x](https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x)** 模型通过将 **MLP** 层划分为分片来优化专家使用，从而提高了部署效率。
  - 社区讨论集中在利用这种模型架构来增强特定应用中的性能。
- **Llama3 模型的性能挑战**：用户报告了微调后的 **Llama3.1** 推理时间不一致，根据加载需求，时间从毫秒到一分钟以上不等。
  - 这些波动突显了在生产环境中部署 **Llama3** 模型时，需要更好的集成实践。

**3\. AI 伦理与数据实践**

- **NVIDIA 的数据抓取争议**：**NVIDIA** 因据称每天抓取大量视频数据用于 **AI** 训练而面临抵制，这引发了员工的伦理担忧。
  - 泄露的文件证实了管理层批准了这些做法，在公司内部引发了巨大的动荡。
- **反对 AI 安全监管法案 SB1047**：一封反对加州 **SB1047** 法案的公开信指出，人们担心该法案可能会扼杀 **AI** 领域的开源研究和创新。
  - 成员们讨论了该法案潜在的负面影响，并呼吁签名支持反对意见。

**4\. 新兴 AI 项目与协作**

- **Open Medical Reasoning Tasks 项目启动**：**[Open Medical Reasoning Tasks 项目](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks)** 旨在联合 **AI** 和医学界，共同完成全面的推理任务。
  - 该倡议寻求通过贡献来推进 **AI** 在医疗保健领域的应用，反映了这些领域日益增长的交集。
- **Gemma 2 2B 的能力**：Google 的 **[Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)** 模型支持设备端运行，展示了令人印象深刻的部署潜力。
  - 社区反馈强调了它的易用性，特别是在 **Google Colab** 等环境中。

**5\. AI 框架与库的进展**

- **Mojo 中 InlineList 的新特性**：**Mojo** 正在 **InlineList** 中引入新方法，例如 `__moveinit__` 和 `__copyinit__`，旨在增强其功能集。
  - 这些进展标志着 **Mojo** 致力于改进其数据结构能力，以备未来发展。
- **Bits and Bytes 基金会更新**：最新的 **[Bits and Bytes pull request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1220)** 引起了库开发爱好者的兴趣。
  - 这一进展被视为该库演进的关键，社区正在密切关注其进展。

## GPT4O-Aug (gpt-4o-2024-08-06)

**1\. AI 模型进展**

- **Gemma 2 2B 发布**：Google 推出的 [Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp) 通过 **WebLLM** 和 **WebGPU** 技术支持流畅的端侧运行。
  - 社区对其易用性表示赞赏，甚至在免费版的 [Google Colab](https://x.com/reach_vb/status/1819469088890261748) 上也能运行，展示了其部署潜力。
- **Mistral MoE 化提升 AI 效率**：[Mistral-7b-MoEified-8x](https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x) 通过专家模型架构和拆分 MLP 层增强了部署效率。
  - 社区讨论集中在优化专家使用率以获得更好的模型性能。
- **DeepSeek-V2 提升推理性能**：一项研究发现，在推理过程中增强样本生成可显著提升语言模型的效率，在 **SWE-bench Lite** 领域的提升幅度从 **15.9%** 显著增长至 **56%** [查看 PDF](https://arxiv.org/abs/2407.21787)。
  - 值得注意的是，增加尝试次数凸显了潜力，其中 **DeepSeek-V2-Coder-Instruct** 重新定义了此前单次尝试成功率上限为 **43%** 的基准测试。

**2\. GPU 性能与兼容性**

- **NVIDIA Blackwell GPU 面临延迟**：[NVIDIA 的 Blackwell GPU](https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ) 因在单个 Superchip 上集成两个 GPU 的芯片设计错误而面临延迟。
  - 这一重新设计的需求推迟了这些先进处理器的发布，影响了开发者和技术采用者的进度表。
- **Intel Arc GPU 兼容性争论**：Intel Arc GPU 因其对 CUDA 的支持评价褒贬不一，影响了其在机器学习领域的应用。
  - 一些成员探索了 [ZLUDA 补丁](https://github.com/vosen/ZLUDA)，尽管 AMD 在 ML 方面的可行性仍然是一个备受争议的话题。
- **Bobzdar 评测 GPU 性能**：Bobzdar 在 LM Studio 中使用 **ROCM** 和 **Vulkan** 对 **8700G/780m IGP** 进行的实验显示速度提升了 **25%**，尽管受到超过 20GB 的 **GPU RAM** 限制的挑战。
  - 尽管如此，在 **llama3.1 70b q4** 上的进展显示其处理速度比 CPU 快 **30%**，但在超过 63k 上下文大小时仍存在稳定性问题。

**3\. OpenAI 与 Anthropic 高层变动**

- **John Schulman 加入 Anthropic**：[John Schulman](https://x.com/johnschulman2/status/1820610863499509855) 宣布离开 OpenAI 加入 Anthropic，专注于 AI 对齐（AI alignment）和技术工作。
  - 这一举动被视为在寻求新的视角，引发了关于其对 AI 伦理和创新影响的讨论。
- **OpenAI 高管离职引发关注**：一份[新闻报道](https://www.theinformation.com/articles/trio-of-leaders-leave-openai)提到三位领导者离开 OpenAI，可能导致战略转型。
  - 社区推测这次人事变动可能会如何影响 OpenAI 内部的项目和未来方向。

**4\. AI 工具与框架**

- **Llamafile 彻底改变离线 LLM 可访问性**：**Llamafile** 的核心维护者分享了关于在单个文件中提供离线、可访问 LLM 的激动人心更新，显著增强了用户可访问性。
  - 这一进展反映了通过提供紧凑的离线解决方案来推动语言模型使用大众化的趋势。
- **LlamaIndex：准备迎接 RAG-a-thon**：继首届活动成功举办后，LlamaIndex 将与合作伙伴 @pinecone 和 @arizeai 在帕罗奥图的 @500GlobalVC 举办新一轮 **RAG-a-thon**。
  - 该活动承诺将提供关于检索增强生成（RAG）以及 **LlamaIndex** 如何发挥关键作用的深入见解。

**5\. LLM 微调挑战**

- **GPT-4o 未能通过对话测试**：**GPT-4o** 在保持连贯对话方面表现吃力，经常在不考虑新输入的情况下重复指令。
  - 用户反馈提到 **Sonnet** 往往能纠正这些问题，凸显了 **4o** 对话模型的缺陷。
- **Unsloth 在微调和 PPO 集成方面遇到困难**：在 **Unsloth** 中使用 'model.save_pretrained_merged' 进行微调时出现的问题引发了关注，原因是保存方法不一致。
  - 将 Llama3 模型整合到 PPO 训练器中的尝试凸显了在生成输出前进行 'for_inference()' 步骤的必要性，这使集成过程变得复杂。

**6\. 其他**

- **LinkedIn 工程团队利用 Flyte 流水线提升 ML 平台**：官方宣布了一场关于 [LinkedIn 工程团队如何转型其 ML 平台](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)的直播会议，重点介绍 **Flyte 流水线** 及其在 **LinkedIn** 的实现。
  - 预计与会者将深入了解 LinkedIn 在其 ML 平台中使用的工程策略和方法。

---

# 第一部分：Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **安装困境困扰用户**：由于 **Python** 版本与 **SD** 操作不兼容，[安装和配置](https://stable-diffusion-art.com/lora/#Step_1_Install_the_LoRA_model) ComfyUI 和 Flux 被证明存在问题。
  - 成员们对管理不同的 Python 环境表达了挫败感，强调了尽管尝试了各种修复方法，但仍反复失败。
- **ControlNet 的风格创意**：通过使用 **ControlNet**，用户分享了将[照片转换为线稿](https://civitai.com/models/596934/line-art-style-sdxl-pony)的方法，并比较了涉及 **DreamShaper** 和 pony 模型的技术。
  - 重点在于利用 **Lora models** 配合特定的基础模型来实现目标艺术输出。
- **Auto1111 的 Inpainting 引起关注**：探索了使用 **Auto1111** 工具进行精细的 Inpainting 任务，例如将[特定海报插入图像中](https://www.xkcd.com/2347/)。
  - 在细节管理方面，**Inpainting** 和 **ControlNet** 成为优于 Photoshop 等手动工具的首选替代方案。
- **Intel Arc GPU 兼容性辩论**：Intel Arc GPU 因其对 CUDA 的支持引起了褒贬不一的反应，影响了其在机器学习领域的应用。
  - 一些成员探索了 [ZLUDA patches](https://github.com/vosen/ZLUDA)，尽管 AMD 在 ML 方面的可行性仍然是一个讨论话题。
- **回顾社区争端**：回顾了不同 **SD forums** 调节团队之间的历史冲突，突显了过去的 *Discord 和 Reddit 动态*。
  - 这些争端揭示了管理开源 AI 社区的复杂性，反映出过去的情况与当前的用户动态息息相关。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral-7b 的 MoEification 提升效率**：[Mistral-7b-MoEified-8x](https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x) 采用了将 MLP 层划分为多个具有特定投影的拆分方法，以增强部署专家模型的效率。
  - 社区专注于通过利用这些拆分模型架构来优化专家模型的使用，以获得更好的性能。
- **Unsloth 在 Fine-Tuning 和 PPO 集成方面的困难**：在 **Unsloth** 中使用 `model.save_pretrained_merged` 进行 Fine-Tuning 时出现的问题引起了关注，原因是保存方法不一致。
  - 将 Llama3 模型集成到 PPO 训练器的尝试突显了在生成输出之前进行 `for_inference()` 步骤的必要性，这使集成过程变得复杂。
- **BigLlama 模型合并带来挑战**：通过 Meta-Llama 和 Mergekit 创建的 [BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct) 模型已被证明存在问题，因为合并后的权重需要训练。
  - 尽管社区充满热情，但许多人认为在经过适当的训练和校准之前，它是“无用的”。
- **Llama-3-8b-bnb 合并策略得到澄清**：用户通过在 gguf 量化之前指定 16-bit 配置，解决了使用 LoRA 适配器合并 **Llama-3-8b-bnb** 的挑战。
  - 该过程涉及遵循精确的合并指令，以确保无缝集成和性能。
- **探索 LLaMA3 的 RunPod 配置**：由于运营费用高昂，讨论了在 **RunPod** 上运行 **LLaMA3 model** 的成本效益策略。
  - 社区成员正在探索能够在保持模型性能效率的同时最大限度降低成本的配置。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gemma 2 2B 正式推出**：Google 发布的 [Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp) 通过 **WebLLM** 和 **WebGPU** 技术实现了流畅的设备端运行。
  - 社区对其易用性表示赞赏，甚至在免费版的 [Google Colab](https://x.com/reach_vb/status/1819469088890261748) 上也能运行，展示了其部署潜力。
- **CogVideoX-2b 引燃视频生成领域**：[CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) 模型因其在文本到视频合成方面的能力而备受关注，可与领先的竞争对手抗衡。
  - 围绕其竞争优势引发了辩论，预示着其在多媒体应用领域有着广阔的前景。
- **结构化输出受到关注**：[OpenAI Blog](https://openai.com/index/introducing-structured-outputs-in-the-api/) 将结构化输出定位为行业标准，引发了关于过往贡献的讨论。
  - 此次发布引发了对过去作品的反思，暗示了机器学习输出标准化格局的演变。
- **深度估计的新构想**：一篇 [CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf) 介绍了一种结合双目立体视觉和单目结构光的 **depth estimation**（深度估计）技术，吸引了社区的兴趣。
  - 社区对这些研究成果的实际落地表现出浓厚兴趣，表明了对计算机视觉领域可操作性见解的追求。
- **AniTalker 彻底改变动画对话**：基于 [X-LANCE](https://github.com/X-LANCE/AniTalker) 的 **AniTalker** 项目增强了动画对话者中的面部动作描绘，提供了细微的身份分离。
  - 测试展示了其在实时对话模拟中的实际效能，暗示了在交互式媒体中更广泛的应用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LMStudio 准备推出 RAG 设置功能**：**LMStudio** 的 **RAG setup** 功能引起了热议，预计将在 0.3.0 版本中首次亮相。这促使部分用户暂时使用 **AnythingLLM** 作为替代方案，尽管有些用户遇到了文件访问障碍。
  - 讨论强调了对 **Meta** 的 **LLaMA** 集成的兴趣，一些人指出了初始设置的挑战，这些挑战可能会在未来的更新中得到简化。
- **GPU 爱好者辩论未来的性能提升**：**NVIDIA 4090** 是否值得升级引发了辩论，一些用户质疑其相对于 3080 的性能提升，并考虑双卡设置或切换到其他平台等替代方案。
  - 关于即将推出的 **RTX 5090** 的改进*推测升温*，预计 VRAM 将维持在 4090 的 **24GB**，但希望有更好的能效和计算能力。
- **在市场波动中制定 GPU 升级策略**：**显卡市场**面临动荡，2024 年 eBay 上 **P40** 显卡价格飙升，而 **3090** 的稀缺引发了人们对传闻中 **AMD 48GB VRAM** 显卡的兴趣。
  - 社区成员强调了在考虑升级时进行 **VRAM** 可扩展性及电源兼容性检查的必要性，并提出了将 **2060 Super** 与 **3060** 组合等具有成本效益的解决方案。
- **量化背景下的 K-V Cache 探讨**：关于 **K-V cache** 设置及其在模型量化中作用的循环讨论，激发了优化 **Flash Attention** 技术的兴趣。
  - 对话包括分享改进注意力机制的指南和资源，暗示了对最大化计算吞吐量的追求。
- **Bobzdar 深入评测 GPU 性能**：Bobzdar 在 LM Studio 中使用 **ROCM** 和 **Vulkan** 对 **8700G/780m IGP** 进行的实验显示了 **25% 的加速**，尽管受到 20GB 以上 **GPU RAM** 限制的挑战。
  - 尽管如此，**llama3.1 70b q4** 的进展显示其处理速度比 CPU 快 **30%**，但在超过 63k 的上下文大小（context sizes）下仍存在稳定性问题。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Hermes 之谜：PyTorch 2.4 vs CUDA 12.4**：用户在运行 **PyTorch 2.4** 与 **CUDA 12.4** 时遇到了**构建中断问题**，而使用 **CUDA 12.1** 则能顺利运行。
  - 进一步的见解提到通过 **conda** 安装了 **CUDA 12.6**，表明存在复杂的版本依赖关系。
- **cublas hgemm 在 Windows 上表现强劲**：**cublas hgemm** 库现在可以在 **Windows** 上运行，在 4090 GPU 上的性能提升至 **315 tflops**，相比之下 nn.Linear 仅为 **166 tflops**。
  - 用户报告在运行 flux 时达到了约 **2.4 it/s**，标志着性能提升的一个里程碑。
- **INT8 量化引发缩放争议**：在 **INT8** 对称量化中，**PyTorch 的 127.5 缩放**策略在 **Qwen2-0.5B** 模型微调中因截断（clipping）导致了发散问题。
  - 社区探索了替代方案，例如 **INT4 量化训练**，以绕过受限范围量化的约束。
- **ZHULDA 3 在 AMD 的主张下消失**：在 **AMD** 撤回此前授予的开发许可后，**ZHUDA 3 项目**已从 **GitHub** 下架。
  - 社区对雇佣合同条款感到困惑，该条款允许在 AMD 认为不合适的情况下发布，凸显了模糊的合同义务。
- **Hudson River Trading 高薪招募 GPU 专家**：Hudson River Trading 正在寻找精通 **GPU 优化**的专家，重点是 **CUDA kernel** 开发和 **PyTorch** 增强。
  - 该公司提供**实习岗位**和高达 **$798K/年** 的竞争性薪资，展示了高频交易领域巨大的财务吸引力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nvidia 进军对话式 AI**：Nvidia 推出了 [UltraSteer-V0 数据集](https://huggingface.co/datasets/Avelina/UltraSteer-v0)，包含 230 万条对话，在 22 天内标注了 9 个细粒度信号。
  - 数据使用 Nvidia 的 [Llama2-13B-SteerLM-RM](https://huggingface.co/nvidia/Llama2-13B-SteerLM-RM) 奖励模型进行处理，评估从**质量**到**创造力**的各种属性。
- **OpenAI 高管离职引发关注**：一份[新闻报道](https://www.theinformation.com/articles/trio-of-leaders-leave-openai)提到三位领导者离开 OpenAI，可能导致战略转变。
  - 社区推测这次人事变动将如何影响 OpenAI 内部的项目和未来方向。
- **Flux AI 图像生成超越竞争对手**：Flux AI 的 “Schnell” 与 **Midjourney 6** 展开竞争，在图像生成的连贯性方面表现更优，展示了先进的模型能力。
  - 尽管有细微的拼写错误，“Schnell” 生成的图像仍表现出极高的逼真度，表明其较竞争对手有了重大进步。
- **医学界通过新倡议加入 AI 领域**：[Open Medical Reasoning Tasks 项目](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks)启动，旨在联合医学界和 AI 社区，共同完成全面的推理任务。
  - 该倡议利用 AI 医疗领域的进展，构建广泛的医学推理任务，并在相关研究中获得关注。
- **Axolotl 走红引发微调热议**：讨论强调 **Axolotl** 库是微调 AI 模型的热门选择，同时还有关于保险行业特定应用的咨询。
  - 此外，还出现了关于 **Llama 450b** 托管方案以及推理瓶颈（特别是使用 **vLLM** 等资源时）的问题。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Web 开发者无缝转型 AI**：由于 ML 专家短缺以及企业对 AI 应用开发的投入不断增加，关于从 Web 开发转向 AI 工程实用性的讨论非常热烈。
  - 尽管职位发布通常强调 ML 资历，但由于公司高度重视 API 集成能力，这些职位经常由具有扎实 Web 开发经验的人员填补。
- **NVIDIA 因 AI 数据收集遭受抨击**：NVIDIA 被指控为 AI 项目进行大规模数据抓取，每天处理相当于“一个人一生”长度的视频材料，并得到了高层管理人员的批准。[流出的文件和 Slack 消息](https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/)证实了这一操作。
  - NVIDIA 对伦理问题的漠视引发了员工的极大不满，也引发了公众对企业责任的质疑。
- **John Schulman 从 OpenAI 跳槽至 Anthropic**：John Schulman 宣布在 OpenAI 任职九年后离职，加入 Anthropic 以专注于 AI Alignment（AI 对齐）研究。
  - [他澄清说](https://x.com/johnschulman2/status/1820610863499509855)，他的决定并非因为 OpenAI 缺乏支持，而是出于深入研究工作的个人抱负。
- **OpenAI 通过 DevDay 与全球受众互动**：OpenAI 宣布将在旧金山、伦敦和新加坡举办一系列 DevDay 活动，旨在通过研讨会和演示展示开发者如何利用 OpenAI 的工具进行实现。
  - [此次巡展](https://openai.com/devday/)代表了 OpenAI 与全球开发者建立联系的战略，强化了其在社区中的角色。
- **OpenAI API 可靠性大幅提升**：OpenAI 在其 API 中实现了 Structured Outputs（结构化输出）功能，确保模型响应严格遵守 JSON Schemas，从而将 Schema 精度从 86% 提升至 100%。
  - [最近的更新](https://x.com/michpokrass/status/1820881057824305567)标志着在实现模型输出的一致性和可预测性方面迈出了一大步。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI DevDay 走向全球**：OpenAI 宣布 **DevDay** 将前往 **旧金山、伦敦和新加坡** 等城市，在今年秋季为开发者提供实操环节和演示。
  - 鼓励开发者与 OpenAI 工程师见面，在 AI 开发领域学习和交流想法。
- **桌面端 ChatGPT 应用与 Search GPT 发布**：成员们根据 Sam Altman 的信息，讨论了 Windows 版 **桌面端 ChatGPT 应用** 的发布日期以及 **Search GPT 的公测**。
  - **Search GPT** 已正式分发，确认了有关其可用性的查询。
- **利用 Structured Outputs**：OpenAI 推出了 **Structured Outputs**，可生成与提供的 Schema 一致的 JSON 响应，增强了 API 交互体验。
  - Python 和 Node SDK 提供原生支持，承诺为用户提供一致的输出并降低成本。
- **AI 重塑游戏世界**：一位成员设想 AI 通过实现独特的角色设计和动态 **NPC 交互** 来提升像《博德之门 3》（BG3）这样的游戏。
  - 生成式 AI 在游戏中的应用预计将增强 **玩家沉浸感** 并彻底改变传统游戏体验。
- **Bing AI Creator 使用 DALL-E 3**：**Bing AI Image Creator** 采用了 **DALL-E 3** 技术，与最近的更新保持一致。
  - 尽管有所改进，用户仍注意到输出质量存在不一致，并表达了不满。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-4o 未能通过对话测试**：**GPT-4o** 在保持连贯对话方面表现吃力，经常在不考虑新输入的情况下重复指令。
  - 用户反馈提到 **Sonnet** 倾向于纠正这些问题，凸显了 **4o** 对话模型的缺陷。
- **排序 AI 旨在解决内容混乱**：一所大学正在开展一个创新的 **内容排序和推荐引擎** 项目，旨在改进数据库内容的优先级排序。
  - 同行建议使用 **RAG** 等平台和本地模型来增强项目的影响力和复杂性。
- **NVIDIA GPU 故障**：[NVIDIA Blackwell GPU](https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ) 因在单个 Superchip 上集成两个 GPU 的芯片设计错误而面临延迟。
  - 这一重新设计的需求推迟了这些先进处理器的发布，影响了开发者和技术采用者的进度表。
- **API 故障削弱用户信心**：API 结果出现意外损坏，在撰写文章时，初始行之后的内容会出现乱码。
  - 文档确认 API 模型弃用计划于 2024 年 8 月进行，包括 **llama-3-sonar-small-32k** 模型。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Meta 通过大规模网络掌握分布式 AI 训练**：在 [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) 上，Meta 展示了其连接数千个 GPU 的庞大 AI 网络，这对于训练 [LLAMA 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) 等模型至关重要。
  - 他们关于 [RDMA over Ethernet for Distributed AI Training](https://dl.acm.org/doi/10.1145/3651890.3672233) 的研究强调了支持全球最广泛 AI 网络之一的架构。
- **SB1047 引发 AI 社区争议**：一封反对 SB1047（即《AI 安全法案》）的公开信正在征集签名，警告该法案可能会抑制开源研究与创新 ([Google Form](https://docs.google.com/forms/d/e/1FAIpQLSewflVHn1zoNeHHJq3SaKvlwPy7PLT1Vcu_WoULqcHSSjvX1w/viewform))。
  - Anthropic 承认监管的必要性，但暗示该法案可能会限制创新，并带来潜在的负面学术和经济影响。
- **机械式异常检测：有前景但表现不一**：Eleuther AI 评估了**机械式异常检测 (mechanistic anomaly detection)** 方法，发现它们有时不如传统技术，详见 [博客文章](https://blog.eleuther.ai/mad_research_update/)。
  - 在全量数据批次上性能有所提升；然而，并非所有任务都有收益，这强调了需要进一步完善的研究领域。
- **扩展 SAEs：近期的探索与资源**：Eleuther AI 讨论了 **Structural Attention Equations (SAEs)**，并提供了指向 [Monosemantic Features 论文](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 等基础和现代著作的链接。
  - 目前正在努力将 SAEs 从玩具模型扩展到 13B 参数，Anthropic 和 OpenAI 在 [扩展论文](https://arxiv.org/abs/2406.04093) 中展示了显著的合作。
- **lm-eval-harness 轻松适配自定义模型**：Eleuther AI 鼓励在自定义架构中使用 **lm-eval-harness**，并在 [GitHub 示例](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py) 中提供了指南链接。
  - 讨论涉及了批处理的细微差别，并确认了默认包含 **BOS token**，突显了 eval-harness 在测试环境中的适应性。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GPU 溢出：在 CPU 上运行模型**：用户在尝试于显存（vRAM）仅为 8GB 的 GPU 上运行大型模型时遇到了**内存溢出**问题，导致不得不采用完全利用 CPU 的变通方案，尽管性能较慢。
  - 针对处理 GPU 显存不足的最佳实践展开了讨论，强调了速度与能力之间的权衡。
- **LangChain 集成难题**：由于缺乏文档，关于如何在 LangChain v2.0 中结合 **RunnableWithMessageHistory** 进行聊天机器人开发的问题频出。
  - 建议通过现有的教程探索存储**消息历史 (message history)** 的方法，这暗示了开发者面临的常见障碍。
- **自动代码审查的缺陷引发抱怨**：**GPT-4o** 在正确评估 GitHub diffs 中的位置时出现问题，促使用户寻求替代的数据处理方法。
  - 建议避免使用视觉模型，转而采用特定于编码的方法，这凸显了将 AI 应用于代码审查的挑战。
- **AgentGenesis 邀请开源协作**：**AgentGenesis** 项目提供了一个 AI 代码片段库，目前正在寻求贡献者以增强其开发，并强调其采用开源 MIT 许可证。
  - 鼓励通过其 [GitHub 仓库](https://github.com/DeadmanAbir/AgentGenesis) 进行积极的社区协作和贡献，以构建一个强大的可重用代码库。
- **Mood2Music 应用精准触达**：**Mood2Music** 应用承诺根据用户的情绪推荐音乐，并与 **Spotify** 和 **Apple Music** 无缝集成。
  - 这款创新应用旨在通过情绪检测自动创建播放列表来提升用户体验，其特色是独特的 AI 自拍分析功能。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **John Schulman 意外转投 Anthropic**：[John Schulman](https://x.com/johnschulman2/status/1820610863499509855) 宣布离开 OpenAI 加入 Anthropic，重点关注 AI alignment 和技术工作。
  - 此举被视为在寻求新的视角，引发了关于 AI 伦理和创新影响的讨论。
- **关于 Gemini 计划的泄露传闻引发成员兴趣**：社区对有关 OpenAI 的 Gemini 计划的泄露信息进行了推测，对围绕 Gemini 2 的神秘进展感到惊叹。
  - 这种好奇心引发了关于 OpenAI 内部潜在技术进步和战略方向的疑问。
- **Flux Pro 在 AI 模型中提供独特体验**：**Flux Pro** 被描述为提供了与竞争对手明显不同的用户体验。
  - 讨论集中在它的独特方法可能并非源于基准测试，而是源于主观的用户满意度。
- **数据依赖性影响模型收益**：对话强调，模型性能受益于将数据分解为 \\( (x, y_w) \\) 和 \\( (x, y_l) \\) 等组件，这在很大程度上取决于 **data noise levels**。
  - 初创公司通常选择噪声数据策略来绕过标准的 supervised fine-tuning，正如一次提及 Meta 的 Chameleon 方法的 ICML 讨论中所指出的。
- **Claude 在用户体验上落后于 ChatGPT**：成员们认为 **Claude** 不如 **ChatGPT**，指出它类似于旧的 GPT-3.5 模型，而 ChatGPT 则因其灵活性和记忆性能受到称赞。
  - 这引发了关于下一代 AI 工具的技术进步和用户期望的对话。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT4-4o 发布，具备结构化输出能力**：新模型 [GPT4-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06) 已在 OpenRouter 上发布，增强了结构化输出（Structured Output）能力。
  - 此更新包括在响应格式中提供 JSON schema 的能力，鼓励用户在[指定频道](https://discord.com/channels/1138521849106546791)报告严格模式（strict mode）的问题。
- **AI 模型性能风波**：与 **haiku/flash/4o** 相比，**yi-vision** 和 **firellava** 模型在测试条件下表现不佳，凸显了持续的价格和效率挑战。
  - 讨论暗示 **Google Gemini 1.5** 即将降价，将其定位为更具成本效益的替代方案。
- **高性价比的 GPT-4o 在 Token 管理上取得进展**：通过采用更具成本效益的 **gpt-4o-2024-08-06**，开发者现在可以在输入上节省 50%，在输出上节省 33%。
  - 社区对话表明，效率和战略规划是该模型降低成本的关键因素。
- **计算 OpenRouter API 成本**：关于 OpenRouter API 成本计算的详细讨论强调，在请求后使用 `generation` 端点可以进行准确的支出跟踪。
  - 这种方法允许用户在按需付费方案中有效管理资金，而无需在流式响应中嵌入成本详情。
- **Google Gemini 节流问题**：**Google Gemini Pro 1.5** 的用户因严重的速率限制（rate limiting）面临 `RESOURCE_EXHAUSTED` 错误。
  - 有必要调整使用预期，目前对于这些速率限制约束还没有立即的解决方案。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex：为 RAG-a-thon 做好准备**：在首届活动取得成功后，LlamaIndex 将与合作伙伴 @pinecone 和 @arizeai 在帕洛阿尔托的 @500GlobalVC 举办新一轮的 **RAG-a-thon**。
  - 该活动承诺将深入探讨 Retrieval-Augmented Generation 以及 **LlamaIndex** 在其中发挥的关键作用。
- **关于 RAG 增强型编程助手的网络研讨会**：[与 CodiumAI 合作的网络研讨会](https://lu.ma/ka5xtyqo) 邀请参与者探索 **RAG 增强型编程助手**，展示 LlamaIndex 如何提升 AI 生成代码的质量。
  - 参与者必须注册并验证 Token 所有权；会议将展示维护上下文代码完整性的实际应用。
- **RabbitMQ 弥合 Agent 间的差距**：@pavan_mantha1 的一篇博客探讨了如何使用 [RabbitMQ](https://t.co/IOGpDWkY8A) 在多 Agent 系统中实现 Agent 之间的有效通信。
  - 这一创新设置集成了 @ollama 和 @qdrant_engine 等工具，以简化 **LlamaIndex** 内部的操作流程。
- **Function Calling 故障导致 CI 崩溃**：LlamaIndex 的 `function_calling.py` 产生了一个 TypeError，阻碍了 CI 流程，该问题已通过升级特定依赖项得到解决。
  - 旧的软件包要求引发了问题，促使团队加强依赖项规范，以避免未来出现此类故障。
- **显微镜下的向量数据库**：分享了一份 [Vector DB 对比](https://superlinked.com/vector-db-comparison)，用于评估不同向量数据库的能力。
  - 鼓励社区分享使用各种 VectorDB 的经验见解，以教育和加强知识共享。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Galileo 幻觉指数引发开源辩论**：[Galileo 的幻觉指数 (Hallucination Index)](https://www.rungalileo.io/hallucinationindex) 引发了关于 LLM 开源分类的讨论，突显了在对 Command R Plus 等模型进行分类时的模糊性。
  - 用户就 **Open weights** 与完全开源之间的区别进行了争论，主张建立更清晰的标准，并可能设立一个单独的类别。
- **Command R Plus 许可争议升温**：Galileo 澄清了他们对开源的定义，即包含支持商业用途的模型，并将 Command R Plus 的 Creative Commons 许可视为一种限制。
  - 成员们讨论了为 **'Open weights'** 创建一个新类别的必要性，建议用明确的许可分类取代宽泛的开源标签。
- **Mistral 在 Apache 2.0 协议下的 Open weights**：Mistral 的模型因其宽松的 Apache 2.0 许可而脱颖而出，提供了比通常的 Open weights 更多的自由度。
  - 讨论包括分享 [Mistral 的文档](https://docs.mistral.ai/getting-started/open_weight_models/)，强调了他们在预训练和指令微调模型透明度方面的倡议。
- **用于 RAG 项目的 Cohere Toolkit**：一名成员将 [Cohere Toolkit](https://cohere.ai/) 用于 AI 奖学金项目，展示了其在开发跨多个特定领域数据库的 RAG LLM 中的应用。
  - 该工具包的集成旨在探索来自 Confluence 等平台的内容，增强其在各种专业环境中的实用性。
- **探索第三方 API 集成的可行性**：正在讨论从 Cohere 模型切换到第三方 API（如 [OpenAI 的 Chat GPT](https://openai.com) 和 [Gemini 1.5](https://groq.com)）的可能性。
  - 使用这些外部 API 的潜力显然有望扩大现有项目的范围和适应性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **InlineList 取得令人兴奋的新特性进展**：Mojo 中 **InlineList** 的开发正在推进，根据最近的 [GitHub pull request](https://github.com/modularml/mojo/pull/2825)，引入了 **`__moveinit__`** 和 **`__copyinit__`** 方法，旨在增强功能集。
  - 这些新方法似乎是由*技术优先级*驱动的，暗示了 `InlineList` 增强功能的未来潜力。
- **Mojo 通过小缓冲区策略优化 List**：Mojo 中 `List` 的**小缓冲区优化 (Small buffer optimization)** 引入了灵活性，允许通过 **`List[SomeType, 16]`** 等参数进行栈空间分配，详见 [Gabriel De Marmiesse 的 PR](https://github.com/modularml/mojo/pull/2825)。
  - 这一改进最终可能会消除对独立 `InlineList` 类型的需求，从而简化现有架构。
- **Mojo 在自定义加速器方面的新前景**：带有脉动阵列 (systolic arrays) 的 PCIe 卡等**自定义加速器**有望在 Mojo 开源后成为其潜在支持对象，展示了新的硬件集成可能性。
  - 尽管热情高涨，但目前使用 Mojo 进行自定义算子 (kernel) 替换仍具挑战性，因为在 RISC-V 目标支持可用之前，像降低 PyTorch IR 这样的现有流程仍占据主导地位。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **OpenAI 领导层变动，John Schulman 加入 Anthropic**：OpenAI 联合创始人 [John Schulman](https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-join-rival-anthropic.html) 宣布离开并加入竞争对手 Anthropic，这是受 OpenAI 近期重组的推动。
  - 此次领导层变动发生在 OpenAI 解散其超级对齐 (superalignment) 团队仅三个月后，暗示了内部战略的转变。
- **开源模型训练面临高成本障碍**：开源社区承认，昂贵的模型训练限制了 SOTA (state-of-the-art) 模型的开发。
  - 更廉价的训练可能会带来开源模型的繁荣，暂且不谈数据来源的伦理挑战。
- **Meta 的 JASCO 项目因法律问题受阻**：Meta 低调进行的 JASCO 项目面临延迟，可能归因于与 Udio 和 Suno 的诉讼。
  - 人们的担忧日益增加，因为这些法律纠纷可能会减缓专有 AI 的技术进步。
- **验证准确率达到 84%，引发“信徒”狂欢**：模型验证准确率达到 **84%**，这是一个值得关注的里程碑，人们引用《黑客帝国》(The Matrix) 的梗来庆祝。
  - 随着这一突破，热情高涨，呼应了那句经典台词：“他开始相信了 (He's beginning to believe)”。
- **CIFAR 的频率保持与相位查询**：有人对 CIFAR 图像在傅里叶分析中的频率恒定性与潜在的相位偏移提出了询问。
  - 这种好奇心引发了关于图像频率是否保持稳定而相位动态发生变化的讨论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 Aurora 雄心**：成员们在 [general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270096760634736660) 频道探讨了在 **Aurora**（配备 Intel GPU 支持的尖端超级计算机）上运行 **tinygrad** 的可行性。
  - 见解显示，Aurora 的 GPU 可以利用独特的 Tensor Core 指令，输出 **16x8 矩阵**，优化后性能可能超过 **2 ExaFLOPS**。
- **FP8 Nvidia 悬赏任务中的精度风险**：关于 **FP8 Nvidia 悬赏任务**的咨询不断，重点在于精度是否需要 E4M3、E5M2 或同时满足这两种标准。
  - 该悬赏反映了 Nvidia 对多样化精度要求的重视，挑战开发者在不同模式下进行优化。
- **解决 Tinygrad 中的 Tensor 切片 Bug**：修复了 Tinygrad 中导致 **Tensor 切片**出现 `AssertionError` 的 Bug，确保切片保持连续性，此消息已由 George Hotz 确认。
  - 该解决方案明确了 **Buffer 到 DEFINE_GLOBAL** 的转换，这是 Tinygrad 计算操作中一个令人困扰的问题。
- **JIT 与 Batch Size 的博弈**：数据集中不一致的 **Batch Size** 导致了 **JIT 错误**，建议包括跳过或单独处理最后一个 Batch 以防止错误。
  - George Hotz 建议确保不对最后一个不完整的 Batch 执行 JIT，以平滑工作流。
- **解锁计算机代数解决方案**：分享了关于计算机代数的**学习笔记**，旨在帮助理解 Tinygrad 的 shapetracker 和符号数学，访问地址见[此处](https://github.com/mesozoic-egg/computer-algebra-study-notes)。
  - 该仓库加深了对 Tinygrad 结构的理解，为深入研究高级符号计算的热心人士提供了宝贵知识。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Wiseflow 高效挖掘数据**：**Wiseflow** 被誉为一种敏捷的数据提取工具，能够系统地对网站和社交媒体的信息进行分类并上传到数据库，详情可见 [GitHub](https://github.com/TeamWiseFlow/wiseflow)。
  - 成员们讨论了将 **Golden Ret** 与 **Wiseflow** 集成，以构建一个强大的动态知识库。
- **HybridAGI 发布新版本**：**HybridAGI** 项目发布了新版本，重点关注易用性和优化数据流水线（data pipelines），并推出了 Vector-only RAG 和 Knowledge Graph RAG 等新功能，已在 [GitHub](https://github.com/SynaLinks/HybridAGI) 上分享。
  - 社区对其在各种 AI 设置中实现无缝神经符号计算（neuro-symbolic computation）的应用表现出浓厚兴趣。
- **基于 LLM 的 Agent 旨在挖掘 AGI 潜力**：最近的一篇论文深入探讨了基于 **LLM 的 Agent** 规避自主性和自我完善等限制的前景，挑战了传统的 LLM 约束 [查看 PDF](https://arxiv.org/abs/2408.02479)。
  - 越来越多的人呼吁在软件工程中建立明确的标准来区分 **LLM** 和 Agent，并强调了统一标准的必要性。
- **推理计算提升性能**：一项研究发现，在推理过程中增强样本生成可显著提高语言模型的效率，在 **SWE-bench Lite** 领域的增益从 **15.9%** 显著提升至 **56%** [查看 PDF](https://arxiv.org/abs/2407.21787)。
  - 值得注意的是，增加尝试次数凸显了潜力，其中 **DeepSeek-V2-Coder-Instruct** 重新定义了此前上限为 **43%** 的单次尝试成功率基准。
- **MIPRO 的性能指标参差不齐**：在性能讨论中，有人指出 **MIPRO** 通常优于 **BootstrapFewShotWithRandomSearch**，但在不同情况下表现并不一致。
  - 关于 **MIPROv2** 的进一步提问确认了其目前尚不支持断言（assertions），这是社区期待已久的功能。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **合成数据策略增强推理任务**：一位社区成员针对 8b 模型提出了一种**合成数据生成策略**，通过在合成指令中加入 **Chain-of-Thought (CoT)**，专注于 text-to-SQL 等推理任务。
  - 讨论了一种在生成最终 SQL 查询之前先进行 CoT 训练的方法，以提高模型性能。
- **LoRA Adapter 合并中的 MD5 哈希一致性得到确认**：关于合并 **LoRA Adapter** 时 **MD5 哈希一致性** 的咨询得到了确认，即预期结果确实应该是一致的。
  - 任何与预期 MD5 哈希结果的偏差都被讨论为潜在问题的迹象。
- **Bits and Bytes Pull Request 引起关注**：用户们认识到 [最新的 Bits and Bytes Foundation pull request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1220) 对库开发爱好者具有重要意义。
  - 该 Pull Request 被视为该库演进过程中的关键进展，正受到社区的密切关注。
- **Gemma 2 27b QLoRA 需要微调**：注意到了 **Gemma 2 27b** 的 QLoRA 存在问题，特别是关于调整学习率以配合最新的 flash attention 从而改善结果。
  - 建议调整 QLoRA 参数以增强性能，特别是在集成 flash attention 等新模块时。
- **UV：一个强大的 Python 包管理器**：[UV](https://github.com/astral-sh/uv) 是一个用 Rust 编写的新型 Python 包管理器，因其在高效处理安装方面的惊人速度而被引入。
  - UV 被认为是 **pip** 的更快替代方案，因其可能改进 docker build 过程而受到关注。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 推出 PPO 集成**：Torchtune 添加了 [PPO 训练 recipes](https://github.com/pytorch/torchtune/pull/1005)，使其功能支持 **Reinforcement Learning from Human Feedback (RLHF)**。
  - 这一扩展允许更强大的训练过程，增强了该平台支持的模型在 RLHF 方面的可用性。
- **Qwen2 模型加入 Torchtune 阵容**：Torchtune 扩展了支持范围，包括 **Qwen2 模型**，目前 [7B 模型已可用](https://github.com/pytorch/torchtune/pull/1143)，后续还将推出更多小型模型。
  - 对不同模型尺寸的扩展支持旨在增强 Torchtune 对多样化机器学习需求的适应能力。
- **Llama3 文件路径排查变得更简单**：成员们讨论了 **Llama3** 模型面临的挑战，强调了正确的 checkpointer 和 tokenizer 路径，以及为 LLAMA3 Instruct 模型[自动配置 prompt](https://discord.com/channels/1216353675241590815/1216353675744641096/1270207545344135304)。
  - 这些确认简化了用户在面对 prompt 变异性和模型干扰问题时的处理流程。
- **Torchtune 计划进行模型页面改版**：成员们正在考虑重构 **Model Page**，以适配包括 **multimodal LLMs** 在内的新模型和未来模型。
  - 提议的改版包括一个模型索引页，用于统一处理下载和配置模型等任务。
- **PreferenceDataset 得到增强**：Torchtune 的 **PreferenceDataset** 现在拥有统一的数据流水线，支持最近 [GitHub pull request](https://github.com/pytorch/torchtune/pull/1276) 中概述的聊天功能。
  - 此次重构旨在简化数据处理，并征求社区反馈以进一步完善转换设计。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 中本地 LLM 设置出现失误**：在 Open Interpreter 中使用本地 LLM 进行设置时，选择 *llamafile* 后会导致不必要的下载，从而引发 *openai.APIConnectionError*。
  - 相关解决工作正在进行中，用户正通过私信协调解决方案。
- **Open Interpreter 的安全性问题**：一位用户对 Open Interpreter 的数据隐私和安全性表示担忧，询问系统间的通信是否包含端到端加密。
  - 该用户热衷于了解加密标准和数据保留政策，特别是涉及第三方参与时。
- **Python 版本支持困惑**：Open Interpreter 目前支持 **Python 3.10** 和 **3.11**，这让询问 Python 3.12 支持的用户无所适从。
  - 建议通过 Microsoft App Store 进行安装验证以检查兼容性。
- **分享 Ollama 模型设置提示**：用户讨论了使用 `ollama list` 设置本地模型的方法，并强调了模型对 VRAM 的前提要求。
  - 有关付费模型所需的 API key 详情，请参阅 [GitHub 说明](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx)。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 彻底改变了离线 LLM 的可访问性**：**Llamafile** 的核心维护者分享了关于在单个文件中提供离线、可访问 LLM 的最新进展，显著增强了用户的可访问性。
  - 这一进展反映了通过提供紧凑的离线解决方案来推动语言模型访问民主化的努力。
- **Mozilla AI 以礼品卡激励反馈**：Mozilla AI 发起了一项 [社区调查](https://form.typeform.com/to/Cn4md4Oc)，为参与者提供赢取 **$25 礼品卡** 的机会，以换取宝贵的反馈。
  - 该倡议旨在从社区收集深入的见解，为未来的开发提供参考。
- **sqlite-vec 发布派对引发关注**：[sqlite-vec 的发布派对](https://discord.com/events/1089876418936180786/1265715263999836210) 正式开启，邀请爱好者探索新功能并参与互动演示。
  - 该活动由核心维护者主持，展示了 SQLite 中向量数据处理方面的实质性进展。
- **机器学习论文研讨引发热议**：社区深入探讨了 **机器学习论文研讨**，主题包括 “[Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795)” 和 “[Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817)”，揭示了新的分析视角。
  - 这些讨论激发了关于这些新发现的潜在影响和实施方案的讨论。
- **Local AI AMA 弘扬开源精神**：**Local AI** 的维护者进行了一场成功的 [AMA](https://discord.com/events/1089876418936180786/1268967945216721079)，重点介绍了他们作为 OpenAI 的开源、自托管替代方案。
  - 该活动强调了对开源开发和社区驱动创新的承诺。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LinkedIn 工程团队利用 Flyte 流水线提升 ML 平台**：宣布了一场关于 [LinkedIn 工程团队如何改造其 ML 平台](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/) 的直播会议，重点关注 **Flyte 流水线** 及其在 **LinkedIn** 的实施。
  - 预计参与者将深入了解 LinkedIn 在其 ML 平台中使用的工程策略和方法。
- **Flyte 流水线的实际应用**：直播活动涵盖了 [Flyte 流水线](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)，展示了它们在 LinkedIn 基础设施中的实际应用。
  - 参与者将探索 LinkedIn 如何利用 Flyte 来提高运营效率。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1270099062342947013)** (459 messages🔥🔥🔥):

> - `模型与工具讨论`
> - `安装挑战`
> - `Lora 与 ControlNet 使用`
> - `放大与处理技术`
> - `社区与平台问题`

- **安装挑战带来困扰**：几位成员在安装和配置方面遇到了困难，特别是 ComfyUI 和 Flux，导致了诸如 Python 版本不兼容影响 SD 和 ComfyUI 运行等问题。
  - 一位成员分享了多次修复失败的挫败感，强调了管理不同 Python 环境的挑战。
- **利用 ControlNet 进行风格转换**：用户讨论了使用 ControlNet 将照片转换为线稿，利用 img2img 设置并比较了使用 DreamShaper 和 pony models 等方法。
  - 重点在于理解如何将 Lora 模型与特定的基础模型结合使用，以实现理想的艺术风格。
- **Auto1111 在 Inpainting 和造型中的多功能性**：成员们探索了使用 Auto1111 处理特定的 Inpainting 任务，例如在图像中添加特定海报，并讨论了使用透视调整等技术优化图像的方法。
  - *Inpainting 和 ControlNet* 成为管理图像细节的热门选择，无需使用 Photoshop 等手动编辑工具。
- **Intel Arc GPU 引发褒贬不一的反应**：社区讨论了 Intel Arc GPU 的兼容性和性能，对 CUDA 支持的担忧影响了它们在机器学习任务中的吸引力。
  - 一些用户对 ZLUDA 等补丁感到好奇，尽管对 AMD 在 ML 领域可行性的怀疑依然存在。
- **社区资源与往事回顾**：在反思过去的社区事件时，一段对话揭示了不同 SD 论坛管理团队之间的历史摩擦，突显了 Discord 和 Reddit 社区之间的动态关系。
  - 强调了管理开源 AI 社区的挑战，用户们也在思考过去的争议对当前用户参与度的影响。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/0xkarmatic/status/1820618875517685976">Karma (@0xkarmatic) 的推文</a>：哇，Greg 也要休假了。</li><li><a href="https://www.krea.ai/apps/image/realtime">KREA</a>：未找到描述</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>：未找到描述</li><li><a href="https://www.xkcd.com/2347/">Dependency</a>：未找到描述</li><li><a href="https://www.stablediffusiontutorials.com/2024/08/flux-installation.html?m=1">FLUX: Installation with Workflow is Here</a>：未找到描述</li><li><a href="https://x.com/SomniumSpace/status/1820930960239497445">Somnium Space (@SomniumSpace) 的推文</a>：我们很高兴发布 Robert Scoble (@Scobleizer) 在 #SomniumConnect2024 发表的精彩主旨演讲全文✨ 未来 10 年 #AI 将给人类带来什么？这将如何...</li><li><a href="https://huggingface.co/black-forest-labs">black-forest-labs (Black Forest Labs)</a>：未找到描述</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/flux/">Flux 示例</a>：ComfyUI 工作流示例</li><li><a href="https://www.youtube.com/watch?v=sMMYSmDHAY8">ComfyUI: Imposing Consistent Light (IC-Light Workflow Tutorial)</a>：该视频专注于在 ComfyUI 中实现 IC-Light，特别是针对产品摄影。IC-Light 基于 SD1.5，我们使用参考背景和...</li><li><a href="https://stable-diffusion-art.com/lora/#Step_1_Install_the_LoRA_model">什么是 LoRA 模型以及如何在 AUTOMATIC1111 中使用它们 - Stable Diffusion Art</a>：LoRA 模型是小型 Stable Diffusion 模型，可对标准 checkpoint 模型进行微调。它们通常比 checkpoint 小 10 到 100 倍。</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekolfd/cfg_how_it_works_in_nonflux_models_vs_flux_code/```">CFG：在非 Flux 模型与 Flux 中的工作原理（代码示例）</a>：Flux 的 'guidance' 值是一个输入到模型中的简单数值。BFL 在蒸馏阶段通过生成一个...引入了这一点。</li><li><a href="https://www.youtube.com/watch?v=_kctwd4w7R0">Good Vibrations (官方音乐视频)</a>：HD 重制版！由 Marky Mark and The Funky Bunch 演唱的 Good Vibrations 官方音乐视频。#MarkyMark #GoodVibrations #Remastered</li><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>：在 ??? GPU 上的 CUDA。通过在 GitHub 上创建账户为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://civitai.com/models/596934/line-art-style-sdxl-pony">Line Art Style [SDXL Pony] - V1 | Stable Diffusion LoRA | Civitai</a>：LINE ART STYLE 这是一个旨在模拟线条艺术的风格 LoRA，特别是几乎没有阴影/遮挡的艺术，以获得干净的黑色线条...</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (从这个开始) | Stable Diffusion Checkpoint | Civitai</a>：Pony Diffusion V6 是一款多功能的 SDXL 微调模型，能够生成各种兽人、野兽或类人生物的惊人 SFW 和 NSFW 视觉效果...</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1270095045319459039)** (105 条消息🔥🔥):

> - `Mistral-7b 中的 MoEification`
> - `Unsloth 微调保存方法的问题`
> - `将 Unsloth 模型集成到 PPO trainer`
> - `微调后的 Llama3.1 推理性能差异`
> - `LLM 推理的学习资源`

- **Mistral-7b 中的 MoEification 详解**：[Mistral-7b-MoEified-8x 模型](https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x) 通过将 MLP 层分割并调整投影来探索专家模型，旨在优化专家（expert）的使用。
- **Unsloth 微调保存问题**：用户在使用 Unsloth 的 'model.save_pretrained_merged' 保存微调模型时遇到问题，方法表现不一致或无法正常工作。
- **将 Unsloth 与 PPO Trainer 集成失败**：由于在调用 model.generate() 之前必须使用 'for_inference()'，导致由 Unsloth 微调的 Llama3 模型在集成到 PPO trainers 时出现故障。
- **Llama3.1 推理性能不一致**：微调后的 Llama3.1 推理时间波动较大，从毫秒级到超过一分钟不等，原因包括首次运行时的初始加载需求等因素。
- **LLM 推理综合指南**：Replete AI 提供了一份[综合指南](https://guide.repleteai.com)以帮助理解生成式 AI，被推荐为初学者学习 LLM 推理栈的资源。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/OpenAIDevs/status/1820876430764634115">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：在 API 中引入结构化输出（Structured Outputs）——模型输出现在遵循开发者提供的 JSON Schemas。https://openai.com/index/introducing-structured-outputs-in-the-api/</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN.">Google Colab</a>：未找到描述</li><li><a href="https://guide.repleteai.com">Nextra: the next docs builder</a>：Nextra：下一代文档生成器</li><li><a href="https://huggingface.co/collections/unsloth/load-4bit-models-4x-faster-659042e3a41c3cbad582e734">以 4 倍速度加载 4bit 模型 - Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/4bit-instruct-models-6624b1c17fd76cbcf4d435c8">4bit Instruct 模型 - Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1270306636795346975)** (10 条消息🔥):

> - `BigLlama-3.1-1T-Instruct 模型`
> - `宝可梦 AI 游戏主持人`
> - `LLM 排行榜`
> - `Minecraft`
> - `ChatGPT 宝可梦 Prompt`

- **讨论中的 BigLlama-3.1-1T 模型**：[BigLlama-3.1-1T-Instruct](https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct) 模型是一个使用 Meta-Llama 并通过 Mergekit 创建的实验性合并模型，是 Meta-Llama-3-120B 模型的继任者。
  - 成员们指出它目前是“无用的”，因为它还没有使用合并后的权重进行训练。
- **宝可梦 AI 游戏主持人引发关注**：一个 [ChatGPT 宝可梦 Prompt](https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt) 模拟了游戏主持人（Game Master），引导用户在叙事中穿梭于宝可梦世界，捕捉勇气、友谊和探索的真谛。
  - 该 Prompt 方便用户在 AI 创作的故事和领域中参与捕捉、训练和对战宝可梦。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/mlabonne/BigLlama-3.1-1T-Instruct">mlabonne/BigLlama-3.1-1T-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.rpgprompts.com/post/pokerole-pok%C3%A9mon-chatgpt-prompt">Pokémon RPG - ChatGPT Prompt </a>：此 Prompt 调用一个 AI 创作的游戏主持人，引导你穿越充满活力和令人兴奋的宝可梦世界，灵感来自该系列粉丝熟悉的充满冒险的地区。参与捕捉...</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1270118569652260924)** (162 条消息🔥🔥):

> - `Llama-3-8b-bnb 4 bit 训练与合并`
> - `GPT-4ALL 与 GGUF 文件`
> - `在 Colab 上微调 Llama 模型`
> - `将模型导出至 Ollama`
> - `Unsloth 的多 GPU 支持`

- **Llama-3-8b-bnb 4 bit 训练与合并问题已解决**：一位用户在合并 Llama-3-8b-bnb 4 bit 时因错误的合并指令遇到问题，正确步骤要求在量化为 GGUF 之前，先以 16-bit 模式合并 LoRA adapter。
- **GPT-4ALL 需要转换为 GGUF 文件**：Theyruinedelise 解释说 GPT-4ALL 需要 GGUF 格式的模型，并建议遵循提供的 Colab notebook 中的最终转换步骤。
- **在 Google Colab 上微调 Llama 模型**：用户讨论了使用 Google Colab 微调 Llama 模型的挑战和策略，包括拆分数据集和管理内存以实现有效训练的需求。
- **将模型导出至 Ollama 的流程**：关于如何将模型从 Colab 导出并在 Ollama 中运行的讨论显示，需要终端访问权限（可通过 Colab Pro 获得）才能在本地有效地运行模型。
- **Unsloth 的多 GPU 支持尚在开发中**：一个运行时错误表明 Unsloth 目前不支持多 GPU 设置，尽管他们正在努力添加此功能。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://www.runpod.io/serverless-gpu">用于 AI 推理的 Serverless GPU 端点</a>：使用 RunPod Serverless GPU 端点大规模运行机器学习推理。</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/loading#:~:text=full%20offline%20mode.-,Slice%20splits,-You%20can%20also>">Load</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1270381106402820237)** (1 条消息):

> - `在 RunPod 上的 LLaMA3 配置`
> - `高效的 AI 资源管理`

- **在 RunPod 上高效运行 LLaMA3**：一位成员询问了关于在 RunPod 上以经济高效的方式运行 **LLaMA3 模型** 所需配置的建议。
- **优化 AI 资源利用**：社区成员讨论了高效管理 AI 资源的策略，以最小化成本并最大化性能。

 

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 条消息):

vvelo: [https://fxtwitter.com/reach_vb/status/1820493688377643178](https://fxtwitter.com/reach_vb/status/1820493688377643178)

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1270493009590616085)** (1 条消息):

> - `Gemma 2 2B`
> - `Diffusers integration for FLUX`
> - `Magpie Ultra`
> - `Whisper Generations`
> - `llm-sagemaker Terraform module`

- **Gemma 2 2B 在您的设备上轻松运行**：Google 发布了 [Gemma 2 2B](https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)，这是一个拥有 26 亿参数的版本，专为通过 WebLLM 和 WebGPU 等平台在端侧设备上使用而设计。
- **FLUX 携手 Diffusers 登场**：新的 [FLUX 模型](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) 已集成到 Diffusers 中，凭借 bfl_ml 的发布，承诺提供由突破性的文本生成图像体验。
- **Argilla 和 Magpie Ultra 表现出色**：[Magpie Ultra v0.1](https://x.com/gabrielmbmb_/status/1819398254867489001) 作为首个使用 Llama 3.1 405B 和 Distilabel 构建的开放合成数据集首次亮相，适用于高计算密集型任务。
- **Whisper 生成速度达到闪电级**：使用 Medusa heads，[Whisper 生成](https://x.com/reach_vb/status/1820560137892835369) 现在运行速度提高了 150%，且不牺牲准确性。
- **llm-sagemaker 简化了 LLM 部署**：Llm-sagemaker，一个新的 [Terraform module](https://www.philschmid.de/terraform-llm-sagemaker) 正式发布，旨在简化在 AWS SageMaker 上部署 Llama 3 等 LLM 的流程。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://huggingface.co/blog/gemma-july-update#use-with-llamacpp)">Google 发布 Gemma 2 2B、ShieldGemma 和 Gemma Scope</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1819023974283518223)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Gemma 2 2B 在浏览器中运行，由 WebLLM 和 WebGPU 驱动！🔥 100% 本地和端侧运行。在不到 24 小时内，我们已经将模型推向了边缘！⚡ 在下方的 HF space 中尝试：</li><li><a href="https://x.com/reach_vb/status/1819469088890261748)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Gemma 2 2B 在免费的 Google Colab 中运行！🤗 由 transformers 驱动！⚡</li><li><a href="https://x.com/ggerganov/status/1818699785152397592)">Georgi Gerganov (@ggerganov) 的推文</a>：开始使用最新的 Gemma 2 模型 + llama.cpp 的简单说明 https://huggingface.co/blog/gemma-july-update#use-with-llamacpp</li><li><a href="https://x.com/RisingSayak/status/1819299449966833972)">Sayak Paul (@RisingSayak) 的推文</a>：你应该已经为 @bfl_ml 发布的 FLUX 感到疯狂了。多么棒的模型！在与我的伙伴们 @_DhruvNair_、@YiYiMarz 和 @multimoda 冲刺之后，我回到了 Twitter...</li><li><a href="https://x.com/gabrielmbmb_/status/1819398254867489001)">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>：发布 magpie-ultra-v0.1，这是第一个使用 Llama 3.1 405B 构建的开放合成数据集。它使用 distilabel 创建，是我们迄今为止最先进且计算密集度最高的流水线。https://huggingfac...</li><li><a href="https://x.com/reach_vb/status/1820560137892835369)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：使用 medusa heads 让 Whisper 生成速度提升 150%！🔥 基于 Transformers 构建，准确率下降极小。非常令人兴奋的研究领域，Medusa heads 已被证明对 LLM 来说速度极快...</li><li><a href="https://x.com/mervenoyann/status/1818613425859145772)">merve (@mervenoyann) 的推文</a>：已发布：关于 Vision Language Models 的新任务指南，以及 @huggingface transformers 文档中最新更新的深度估计任务指南 ⛴️📦 👉🏻 阅读关于 VLM、如何流式传输、量化等内容 👉...</li><li><a href="https://x.com/_philschmid/status/1820360144334496064)">Philipp Schmid (@_philschmid) 的推文</a>：很高兴宣布 “llm-sagemaker”，这是一个新的 Terraform module，可以轻松地将来自 @huggingface 的开源 LLM 部署到 @awscloud SageMaker 实时端点！👀 基础设施即代码 (IaC) 工具对于...</li><li><a href="https://x.com/mervenoyann/status/1818675981634109701)">merve (@mervenoyann) 的推文</a>：SAMv2 简直好得令人惊叹 😍 了解是什么让这个模型在视频分割方面如此出色，继续阅读 🦆⇓</li><li><a href="https://x.com/DbrxMosaicAI/status/1818407826852921833)">Databricks Mosaic Research (@DbrxMosaicAI) 的推文</a>：对于我们的 StreamingDataset 用户：我们很高兴宣布支持在 @huggingface 中存储 MDS 数据集。感谢 @orionweller 的贡献！在此查看文档：https://docs.mosaic...</li></ul></div>

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1270097797437460544)** (239 条消息🔥🔥):

> - `MarianMT 模型翻译问题`
> - `新文本生成视频模型发布`
> - `使用语谱图 (spectrograms) 进行音频处理`
> - `数据集大小限制提升流程`
> - `PyTorch 警告与问题`

- **MarianMT 模型缺少 ro-en 翻译**：一位用户注意到，虽然 MarianMT 模型可以实现从英语到罗马尼亚语的翻译，但反向翻译却无法实现，因为 `Helsinki-NLP/opus-mt-ro-en` 并不存在。
- **CogVideoX-2b 新发布令人印象深刻**：一款新发布的模型 [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) 在 AI 社区亮相，用于文本生成视频（text to video）。根据初步评价，其表现似乎可以与 Kling 媲美。
- **语谱图 (Spectrograms) 在音频处理中占据主导地位**：讨论强调了为什么在音频处理中，带有语谱图的 CNN 比 RNN 更受青睐，因为它们能从复杂信号中更好地提取特征。
- **Hugging Face 数据集大小限制查询**：建议寻求分享大型数据集的用户发送邮件至 [datasets@huggingface.co](mailto:datasets@huggingface.co) 以申请提高大小限制。
- **Torch.library 模块引发警告**：一位用户在使用 `torch.library.impl_abstract` 时遇到了未来警告 (future warnings)，该函数在未来的 PyTorch 版本中已重命名为 `torch.library.register_fake`。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://huggingface.co/learn/ml-for-3d-course">欢迎来到 🤗 3D 机器学习课程 - Hugging Face ML for 3D Course</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/repositories-recommendations#sharing-large-datasets-on-the-hub">仓库限制与建议</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/audio-to-spectrogram">Audio To Spectrogram - fffiloni 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music">Riffusion • Spectrogram To Music - fffiloni 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/THUDM/CogVideoX-2b">THUDM/CogVideoX-2b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/dataset_script#add-dataset-attributes)">创建数据集加载脚本</a>：未找到描述</li><li><a href="https://youtu.be/88lQmlsx27w?si=kYnZEmuX8RtzTGX4&amp;t=335">Cherry Blossoms Explode Across the Dying Horizon</a>：由 DistroKid 提供给 YouTube...</li><li><a href="https://github.com/buaacyw/MeshAnythingV2">GitHub - buaacyw/MeshAnythingV2: 像人类艺术家一样从任何物体生成网格。 "MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization" 的官方实现</a></li><li><a href="https://github.com/SonyCSLParis/NeuralDrumMachine/tree/master">GitHub - SonyCSLParis/NeuralDrumMachine</a>：通过在 GitHub 上创建账号来为 NeuralDrumMachine 的开发做出贡献。</li><li><a href="https://github.com/huggingface/datasets/issues/7092">load_dataset 处理多个 jsonlines 文件时过早解析数据结构 · Issue #7092 · huggingface/datasets</a>：描述了可能与 #6460 相关的 bug，即当其中一个文件包含完整结构时，使用 datasets.load_dataset("json", data_dir= ... ) 加载多个 .jsonl 文件会报错...</li><li><a href="https://github.com/huggingface/transformers/issues">Issues · huggingface/transformers</a>：🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供最先进的机器学习模型。</li><li><a href="https://huggingface.co/docs/hub/en/spaces-overview">Spaces 概览</a>：未找到描述</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/launch">Spaces Launch – Hugging Face</a>：未找到描述</li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1270265570016624671)** (3 条消息):

> - `线性代数`
> - `3D 视频分析`

- **探索用于 3D 视频分析的线性代数**：一位成员学习了**线性代数**及其在 **3D 视频分析**中的应用，并寻求有深度博客或文章的推荐。
- **分享学习资源**：另一位成员表示有兴趣传播学习经验，并请求其他人广泛分享该话题。

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1270252132674703493)** (4 条消息):

> - `High Resolution Image Synthesis` (高分辨率图像合成)
> - `Graph Integration with LLMs` (图结构与 LLM 的集成)

- **使用 Transformer 进行高分辨率图像合成**：一位成员对使用 Transformer 合成**高分辨率图像**表示了兴趣，重点强调了图像的潜表征（latent representation）和上下文丰富的词汇码本（vocabulary codebook）等概念。
- **新的图结构与 LLM 集成方法**：分享了一种将图结构集成到 LLM 中的酷炫方法，类似于 ICML 的一项提案，论文可见[此处](https://arxiv.org/pdf/2405.20684v1)。

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1270228072410513409)** (5 条消息):

> - `SAC Agent Training in Unity` (在 Unity 中进行 SAC Agent 训练)
> - `Embodied Agent Platform Development` (具身智能 Agent 平台开发)
> - `AniTalker Project` (AniTalker 项目)
> - `BiRefNet for Image Segmentation` (用于图像分割的 BiRefNet)

- **提升 Unity 中的 SAC Agent 训练**：一位成员分享了支持 **CUDA 或 CPU 多线程**的 **SAC Agent 训练**进展，在 Unity ML-Agents 设置中提供了显著的性能提升。
- **发布具身智能 Agent 平台**：一个**具身智能 Agent (Embodied Agent) 平台**正在开发中，该平台使 Agent 能够与玩家交谈并在 3D 环境中执行任务。
- **用于动画人脸的创新项目 AniTalker**：一位成员介绍了 **AniTalker**，这是一个来自 **X-LANCE** 的说话头像合成移植版，具有身份解耦的面部动作编码功能。
- **BiRefNet 在图像分割中表现卓越**：**BiRefNet** 项目被宣布为高分辨率二分图像分割的 SOTA 解决方案，性能超越了 RMBG1.4。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>：暂无描述</li><li><a href="https://youtube.com/live/XOFMpZsYeXo?feature=share">Unity ML-Agents | 从零开始的实时 Agent 训练 | 第 2 部分</a>：在 3D 体素世界中的快速 SAC Agent 训练器</li><li><a href="https://github.com/thunlp/LEGENT">GitHub - thunlp/LEGENT: Open Platform for Embodied Agents</a>：具身智能 Agent 的开放平台。通过创建账户为 thunlp/LEGENT 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - Hugging Face Space</a>：暂无描述</li><li><a href="https://github.com/X-LANCE/AniTalker">GitHub - X-LANCE/AniTalker: [ACM MM 2024] "AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding" 官方代码</a></li></ul></div>

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1270468951608131585)** (5 条消息):

> - `LLM Reasoning Capabilities` (LLM 推理能力)
> - `OpenAI's Structured Outputs` (OpenAI 的结构化输出)
> - `Theories on LLM Reasoning Mechanisms` (关于 LLM 推理机制的理论)

- **OpenAI 发布结构化输出博客文章**：OpenAI 刚刚发布了一篇[博客文章](https://openai.com/index/introducing-structured-outputs-in-the-api/)，建议将结构化输出（Structured Outputs）作为标准实践，尽管文中很少提及之前的工作。
- **LLM 通过将任务转化为检索来伪造推理**：一种理论认为，虽然 **LLM** 缺乏真正的推理能力，但它们通过将任务转化为**检索任务**来模拟推理，利用其训练所用的庞大互联网事实和逻辑数据集。
- **Token Scratchpads 增强 LLM 推理**：**Token scratchpads** 可能通过扩展 KV-cache 来提升 LLM 的推理能力，在无需重新训练模型的情况下辅助注意力层进行推理。
- **注意力变体和外部数据库影响 LLM 推理**：实证测试显示，与维持 KV-cache 的模型相比，**Mamba/线性注意力**等注意力变体在推理任务中往往表现不佳。

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1270265971914964993)** (4 条消息):

> - `Depth Estimation`
> - `CVPR 2022`

- **Depth Estimation 结合了 Stereo 和 Structured-Light**：一位成员分享了 [CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf)，题为“**Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light**”，该论文探索了一种提高 Depth Estimation 准确性的新方法。
  - 有人询问“*上述论文是否有代码实现？*”，这表明了对所讨论方法的实际应用的兴趣。
- **代码实现请求**：有人询问了关于 [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf) 中讨论的 Depth Estimation 论文的代码实现可用性。
  - 这个问题突显了社区对理论研究的实际应用和真实世界测试的兴趣。

 

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1270340412208578560)** (2 条消息):

> - `Named Entity Recognition dataset`
> - `JSON file search optimization`

- **标注了 IT 技能的 NER 数据集在 Kaggle 上线**：一位成员分享了一个 [Kaggle 数据集](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs)，其中包含 5029 份使用 Named Entity Recognition (NER) 标注了 IT 技能的简历（CV）。
- **在大规模数据集中识别相关 JSON 文件的挑战**：一位成员讨论了一种从超过 20,000 个 JSON 文件的数据集中识别出最相关的 5 个 JSON 文件 ID 的方法。

 

**提到的链接**：[NER Annotated CVs](https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs)：该数据集包含 5029 份标注过的简历（CV），并标记了 IT 技能。

 

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1270099360310628557)** (157 条消息🔥🔥):

> - `使用 LMStudio 设置 RAG`
> - `InternLM 模型性能`
> - `使用 AI 进行音频转录`
> - `模型量化与 K-V cache`
> - `推理时的 CUDA 设备选择`

- **LMStudio 即将支持 RAG 设置**：用户讨论了在 **LMStudio** 中设置 **Retrieval-Augmented Generation (RAG)** 的可能性，预计在即将发布的 0.3.0 版本中提供支持。
  - 有人对使用 **AnythingLLM** 作为替代方案表示出兴趣，尽管一些用户最初遇到了文件访问问题。
- **InternLM 与模型讨论**：成员们提到了使用 **InternLM2.5** 等模型时遇到的挑战，并讨论了其与 **Gemma2 27b** 等其他模型的性能对比。
  - 对话显示出用户对使用不同量化方式的理解正在加深，并重点介绍了 **IMat quant** 选项。
- **探索通过 AI 工具进行音频转录**：虽然 **LM Studio** 并不直接支持音频输入，但 **AnythingLLM** 和其他集成工具为转录任务提供了可能的路径。
  - 用户表示出于隐私考虑更倾向于使用本地解决方案保持离线状态，这表明云端语音转文本服务存在挑战。
- **理解模型量化中的 K-V cache**：成员们对 **Flash Attention** 和 **K-V cache quant** 设置表现出好奇，一些人试图了解它们对模型性能的影响。
  - 共享了相关资源和指导，以帮助用户优化注意力机制，从而获得更好的效率和输出质量。
- **选择用于推理的 CUDA 设备**：用户探索了诸如修改 **CUDA_VISIBLE_DEVICES** 设置等技术，以指定特定的 GPU 进行模型推理，从而优化其计算配置。
  - 这些解决方案允许在不同 GPU 之间进行高效的资源分配，有助于在进行图像生成等同步任务时获得更好的性能。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention">Flash Attention</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/gguf">GGUF</a>：未找到描述</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1e">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/legraphista/internlm2_5-20b-chat-IMat-GGUF">legraphista/internlm2_5-20b-chat-IMat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://reddit.com/r/stableDiffusion/comments/1el79h3/flux_can_be_run_on_a_multigpu_configuration/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://tenor.com/view/money-dollars-cash-rich-shut-up-and-take-my-money-gif-3555042">Shut Up! GIF - Money Dollars Cash - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>：ref #3365 为 ggml 和 llama.cpp 设置 Flash Attention 支持所需的各项内容。提议的算子执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale); // fused sc...</li><li><a href="https://openwebui.com/">Open WebUI</a>：未找到描述</li></ul></div>

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1270107257018912950)** (59 messages🔥🔥):

> - `8700G/780m IGP 测试`
> - `NVIDIA 4090 和 5090 讨论`
> - `显卡市场趋势`
> - `针对 LLM 的 GPU 升级`
> - `RTX 4090 vs 3080 性能`

- **Bobzdar 的测试：使用 ROCM 和 Vulkan 测试 8700G/780m IGP**：Bobzdar 报告称，在 8700G/780m IGP 上使用带有 ROCM 的特殊 Ollama 版本可实现 **25% 的加速**，而在 LM Studio 中使用 Vulkan 则有 **15%** 的加速，尽管他在显存占用超过 **20GB GPU RAM** 时遇到了加载问题。
  - 他成功以比 CPU 快 **30%** 的速度运行了 **llama3.1 70b q4**；然而，在 LM Studio 中，当上下文大小超过 **63k context size** 时会发生崩溃。
- **NVIDIA 4090：值得升级吗？**：Pydus 考虑升级到 4090 并询问其与游戏版本的区别，随后指出其性能并没有比 3080 显著提升。
  - 他对其速度优势表示不确定，并考虑配置 **两块 4090** 或转向 **MAC**。
- **关于 GPU VRAM 需求的争论**：讨论了 **RTX 5090** 是否会在 **4090 的 VRAM** 能力基础上进行显著改进，一些人预测其容量仍为 **24GB VRAM**。
  - Pydus 和其他人推测是等待还是升级，AMD Opteron 强调可用性和价格构成了挑战，而 AMD 可能是未来的竞争对手。
- **当前显卡市场趋势**：2024 年 eBay 上的 P40 显卡价格翻了一番，说明了市场需求的变化，而由于价格居高不下，过剩的 3090 仍然难以寻觅。
  - LLM 构建者渴望像潜在的 **AMD 48GB VRAM** 这样的显卡，如果发布，可能会影响 NVIDIA 的定价和市场策略。
- **GPU 升级：因素与考量**：对于较大的 LLM 模型，社区成员建议至少升级到 **3060** 甚至 **3090**，并强调了 **VRAM** 在性能考量中的重要性。
  - 建议包括检查电源兼容性，因为 **GPU 升级** 需要更高的功率；为了提高成本效益，建议将 **2060 Super 与 3060** 组合使用。

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1270135353378082977)** (5 messages):

> - `PufferLib 环境搭建`
> - `强化学习直播`
> - `GPUDrive 生成示例`
> - `关于 Mojo 演讲的请求`

- **为 Gameboy 模拟设置 PufferLib**：一位成员分享了使用 [PufferLib](https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py) 为 Gameboy 模拟器设置环境的链接，并评论了从 CPython 这种熟悉的语言开始的好处。
- **在直播中提问 RL 问题**：PufferLib 库的作者正在进行直播，可以通过专注于强化学习开发的 [YouTube 会话](https://www.youtube.com/watch?v=dW10MQ6hKDE) 直接向其提问。
- **GPUDrive 提升 Agent 训练速度**：Hugging Face 的论文介绍了 **GPUDrive**，这是一个使用 CUDA 的多 Agent 模拟器，可以在 [Waymo Motion 数据集](https://huggingface.co/papers/2408.01584) 中高效训练强化学习 Agent，在几分钟到几小时内实现成功的 Agent 行为。
- **请求 Mojo 概览环节**：向 Chris 及其团队发出了邀请，希望就 **Mojo** 的现状和愿景进行潜在的演讲，并鼓励进行入门级概览。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/papers/2408.01584">论文页面 - GPUDrive: 100 万 FPS 的数据驱动多 Agent 驾驶模拟</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=dW10MQ6hKDE">强化学习现场开发</a>：在 X 上关注 jsuarez5341，Star 他们的 GitHub https://github.com/pufferai/pufferlib，MIT 博士及全职开源 RL 专家</li><li><a href="https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15">PufferLib/pufferlib/environments/pokemon_red/environment.py</a>：简化复杂游戏环境的强化学习 - PufferAI/PufferLib</li></ul></div>

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1270247873237356574)** (17 messages🔥):

> - `PyTorch 2.4 with CUDA 12.4 issues`
> - `cublas hgemm library for Windows`
> - `FP16 accumulate versus FP32`
> - `Speed/accuracy trade-offs in cublas library`
> - `Inference-only library discussion`

- **PyTorch 2.4 在 CUDA 12.4 下遇到障碍**：一位成员指出 CUDA 12.4 构建版本会导致其代码崩溃，而 **PyTorch 2.4 搭配 CUDA 12.1** 运行完美。
  - 该用户进一步澄清，他们在通过 **conda** 安装的基础系统上运行的是 CUDA 12.6。
- **cublas hgemm 库现已支持 Windows**：一位用户分享称，他们使 torch cublas hgemm 库兼容了 **Windows**，在 4090 上将性能提升至最高 **315 tflops**，而 torch nn.Linear 仅为 **166 tflops**。
  - 该库助力 Flux 在 4090 上达到约 **2.4 it/s** 的性能，较之前的基准测试有显著提升。
- **FP16 累加优于 FP32**：讨论强调，使用 FP16 累加的 FP16 可产生 **330 tflops**，而使用 FP32 累加的 FP16 仅达到 **165 tflops**。
  - 尽管存在顾虑，该成员指出，由于 **L1 cache** 有限，FP16 累加在消费级 GPU 上快了 **2 倍**，且比 4/8 bit Quantization 带来的问题更少。
- **cublas 速度-精度平衡基准测试**：基准测试结果显示，**CublasLinear** 与 nn.Linear 的输出略有偏差，但实现了显著的速度提升，达到 **313.22 TFLOPS**，而 torch 为 **166.47 TFLOPS**。
  - 用户保证，这些微小的差异不会显著影响 **diffusion models** 或 **LLMs** 等应用的结果。
- **仅限推理库引起关注**：该 cublas 库被指出是 **inference-only** 的，引发了关于其在特定范围内的适用性和实用性的讨论。
  - 重点在于，尽管缺乏 Training 支持，其高速能力依然非常有益。

 

**提及的链接**：[GitHub - aredden/torch-cublas-hgemm: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu](https://github.com/aredden/torch-cublas-hgemm)：PyTorch 半精度 gemm 库，带有融合的可选 bias + 可选 relu/gelu - aredden/torch-cublas-hgemm

 

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1270094130457870440)** (3 messages):

> - `Quantization Bits as an Optimizable Parameter`
> - `Accuracy Tuning for CIFAR-10`

- **量化位数成为可优化参数**：实验表明，将 Quantization Bits 设为 **optimizable parameter** 可以提高模型性能。
- **CIFAR-10 精度面临调优挑战**：一位成员观察到其模型在 **CIFAR-10** 上达到了约 **70% accuracy**，表明需要进一步调优。

 

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1270193824337100872)** (7 messages):

> - `Hudson River Trading internships`
> - `GPU job optimization`
> - `Software Engineer salary at Hudson River Trading`

- **Hudson River Trading 寻找 GPU 高手**：一位成员描述了他们在高频交易公司 Hudson River Trading 的角色，专注于 **GPU optimization** 和性能工程，任务包括编写 CUDA kernels 和优化 PyTorch。
- **Hudson River Trading 的实习机会引起好奇**：有人询问是否有类似于全职 GPU 优化角色的实习机会，这类机会通常在夏季提供。
- **揭秘 Hudson River Trading 的薪酬**：Hudson River Trading 的 **Software Engineering** 薪资范围为每年 **$406K 至 $798K**，展示了高频交易职位的丰厚潜力。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://grnh.se/9f8394ba1us">Senior Software Engineer - Performance Optimization (C++/GPU)</a>：美国纽约州纽约市</li><li><a href="https://www.levels.fyi/companies/hudson-river-trading/salaries/software-engineer">Hudson River Trading Software Engineer Salary | $406K-$485K+ | Levels.fyi</a>：Hudson River Trading 在美国的软件工程师薪酬范围从 L1 的每年 $406K 到 L3 的每年 $485K+。美国薪酬包的中位数为 $410K。查看...</li></ul></div>

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1270162701972148335)** (34 messages🔥):

> - `INT8 量化问题`
> - `AffinQuantizedTensor 计划`
> - `TorchAO 安装错误`
> - `Tensor Core 操作的硬件兼容性`
> - `GPTQ 重构进度`

- **INT8 量化引发关于缩放技术的讨论**：在关于 INT8 对称量化的讨论中，成员们分析了为什么 PyTorch 使用 **127.5** 进行缩放以及对**受限范围量化 (restricted range quantization)** 的影响。在 **Qwen2-0.5B** 微调中的经验表明，由于数值裁剪 (value clipping)，使用 127.5 会导致模型发散，这激发了对比 **INT8/PTQ** 和 **INT4 量化训练 (Quantized Training)** 替代方案的兴趣。
- **TorchAO 在旧型号 GPU 上的安装挑战**：由于 T4 GPU 与 TorchAO 源代码中的 BF16 操作存在兼容性问题，几位用户在安装 **TorchAO** 时遇到了困难。
- **TorchAO 可能需要更新文档**：用户建议目前的 **TorchAO** 安装文档可能会误导用户，让他们认为某些步骤是叠加的而非替代性的。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://pytorch.org/"><pre><code>  PyTorch
</code></pre></a><p><a href="https://pytorch.org/"></a>：未找到描述</p></li><li><a href="https://github.com/pytorch/pytorch/blob/e98eac76b358fb4639b9e9ce6894014354d7b073/aten/src/ATen/native/cuda/int4mm.cu#L1">pytorch/aten/src/ATen/native/cuda/int4mm.cu at e98eac76b358fb4639b9e9ce6894014354d7b073 · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/blob/de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208/torchao/quantization/quant_primitives.py#L610">ao/torchao/quantization/quant_primitives.py at de4a1fb3b1f71e2f61b84dfdc96e7d704ff72208 · pytorch/ao</a>：用于训练和推理的缺失的 PyTorch dtype 和 layout 库 - pytorch/ao</li><li><a href="https://intellabs.github.io/distiller/algo_quantization.html#symmetric-mode">量化 - Neural Network Distiller</a>：未找到描述<p></p></li></ul></div>

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1270258859633676380)** (7 messages):

> - `LLaMA 3 数据集章节`
> - `前缀分块 LLM 论文 - Sarathi LLM`
> - `使用 CPU 的 CTF 挑战`
> - `用于 LLM 推理的 ChunkAttention`
> - `SARATHI 框架`

- **LLaMA 3 数据集章节脱颖而出**：**LLaMA 3 论文**因其引人入胜的数据集章节而受到关注，而其他部分在其他论文中有更好的解释。
  - 一位用户提到，与其他章节相比，数据集章节是最**有趣的部分**。
- **探索前缀感知 LLM 中的 ChunkAttention**：[ChunkAttention 论文](https://arxiv.org/abs/2402.15220)介绍了一种前缀感知 (prefix-aware) 的自注意力模块，通过在类似的 LLM 请求之间共享 Key/Value 张量来优化内存利用率。
  - 主要改进来自于将整体的 Key/Value 张量分解为更小的块，并使用前缀树架构来增强**内存利用率**。
- **CTF 挑战凸显现代攻击**：分享了一个专注于 CPU 使用和内核漏洞利用的 **CTF 挑战**，结合了 corCTF 2023 的主题。
  - 提供的细节包括 Linux 上的一个新系统调用以及指向 [CTF 挑战](https://www.willsroot.io/2024/08/just-a-dos-bug.html?m=1)的链接。
- **介绍 SARATHI 框架**：[SARATHI 框架](https://arxiv.org/abs/2308.16369)通过采用分块预填充 (chunked-prefills) 和解码最大化批处理 (decode-maximal batching) 解决了 LLM 推理效率低下的问题。
  - SARATHI 通过允许解码请求在推理过程中以较低成本**搭便车 (piggyback)**，从而提高了 GPU 利用率。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://arxiv.org/abs/2402.15220">ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition</a>：自注意力是大型语言模型 (LLM) 的核心组件，但也是长序列推理延迟的重要来源。在多租户 LLM 服务场景中，计算和内存...</li><li><a href="https://arxiv.org/abs/2308.16369">SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills</a>：大型语言模型 (LLM) 推理包含两个不同的阶段——处理输入提示的预填充阶段和自回归生成输出 Token 的解码阶段。虽然预填充...</li><li><a href="https://www.willsroot.io/2024/08/just-a-dos-bug.html?m=1">Will's Root: corCTF 2024: Its Just a Dos Bug Bro - Leaking Flags from Filesystem with Spectre v1</a>：未找到描述信息</li></ul></div>

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1270094011696152648)** (99 条消息🔥🔥):

> - `Ragged attention masks`
> - `Batch size 和 sequence length 调度`
> - `LLaMA 训练中的 Special tokens`
> - `FlashAttention 支持`
> - `训练稳定性和效率`

- **Ragged Attention Masks 带来挑战**：关于使用 ragged attention masks 的讨论揭示了在处理由 EOT 分隔的 token 时，应对分布外（out-of-distribution）场景的困难，这需要定制化的 masking 方法。
- **Batch 和 Sequence Length 调度旨在提高稳定性**：建议的训练策略包括逐渐增加 sequence lengths（例如 512 -> 1024 -> 2048），同时调整 `batch sizes` 和 `RoPE`，旨在平衡计算成本和模型稳定性。
- **LLaMA 训练中 Special Tokens 的实现尚不明确**：关于 `Meta` 如何实现 `<|end_of_text|>` 和 `<|begin_of_text|>` 等 special tokens 的未解决问题导致了用户困惑，可能导致错误的运行时行为。
- **FlashAttention 增强长上下文训练**：目前正在讨论 `FlashAttention` 和 `cudnn` 库是否能有效支持 ragged attention。
- **理解预训练中的训练稳定性**：几位成员指出，通过新的研究见解分析预训练期间的 `training instability`（训练不稳定）和 loss spikes（损失激增）非常重要。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://arxiv.org/abs/2312.16903">Spike No More: Stabilizing the Pre-training of Large Language Models</a>：大语言模型预训练期间经常出现 loss spikes。这些激增会降低大语言模型的性能，有时甚至会破坏预训练。由于预训练需要大量的...</li><li><a href="https://arxiv.org/abs/2108.06084">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</a>：最近的研究在大规模 GPU 上预训练大型自回归语言模型方面取得了巨大成功。为了减少训练时间，通常的做法是增加 ba...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | Model Cards and Prompt formats</a>：Llama 3 使用的 Special Tokens。一个 prompt 应该包含一条 system message，可以包含多条交替的 user 和 assistant messages，并且总是以最后一条 user message 结尾，后面跟着...</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main">🤗 Transformers</a>：未找到描述</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/654.">Issues · Dao-AILab/flash-attention</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/torchchat</a>：在服务器、桌面和移动端本地运行 PyTorch LLMs - Issues · pytorch/torchchat</li></ul></div>

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1270413451608461313)** (9 条消息🔥):

> - `ZLUDA 3 下架`
> - `AMD 对 ZLUDA 的主张`
> - `合同义务`
> - `开发许可`

- **AMD 提出主张后 ZLUDA 3 被移除**：根据 [GitHub](https://github.com/vosen/ZLUDA) 的消息，ZLUDA 3 的作者已下架该项目，因为 **AMD** 声称发布的许可无效。
- **关于 ZLUDA 状态的合同困惑**：雇佣合同条款存在困惑，其中一个条款允许在 AMD 认为 **ZLUDA** 不适合进一步开发时将其发布。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on ??? GPUs</a>：在 ??? GPUs 上运行 CUDA。通过在 GitHub 上创建账号为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://github.com/vosen/ZLUDA/tree/v3?tab=readme-ov-file#faq">GitHub - vosen/ZLUDA at v3</a>：在 ??? GPUs 上运行 CUDA。通过在 GitHub 上创建账号为 vosen/ZLUDA 的开发做出贡献。</li></ul></div>

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1270144971646701569)** (2 条消息):

> - `关于决策时间线的讨论`
> - `完善提案细节`

- **了解决策时间线**：成员们讨论了可能的决策时间线是在月底，并强调了涉及的众多因素。
- **明确提案细节**：讨论了一种确保提案清晰度的方法，强调了使用 Google 表单或 gist 提交详细工作计划的作用。

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1270435322681102447)** (1 条消息):

> - `UltraSteer-V0`
> - `Multi-Turn Dialogue Dataset`
> - `Nvidia's Reward Model`
> - `Fine-Grained Labeling`

- **Nvidia 发布 UltraSteer-V0 数据集**：Nvidia 策划了一个名为 [UltraSteer-V0](https://huggingface.co/datasets/Avelina/UltraSteer-v0) 的数据集，包含 230 万个会话和 280 万个对话轮次，每个轮次都标注了 9 个细粒度信号。
  - 该数据集被描述为“版本 0”，是在经过 **22 天的标注和处理**后推出的，表明后续还有进一步去重和改进的空间。
- **Llama2-13B-SteerLM-RM 为 UltraSteer 提供支持**：UltraSteer 中的对话是使用 Nvidia 的 [Llama2-13B-SteerLM-RM](https://huggingface.co/nvidia/Llama2-13B-SteerLM-RM) Reward Model 在 [NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) 框架内进行标注的。
  - 数据集中的每条助手消息都根据 **Quality**、**Toxicity** 和 **Creativity** 等属性进行评分，分值范围为 **0 到 4**。

**提到的链接**：[Avelina/UltraSteer-v0 · Datasets at Hugging Face](https://huggingface.co/datasets/Avelina/UltraSteer-v0)：未找到描述

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 条消息):

vikings7699: 这里有人曾专门为保险行业微调过模型吗？

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1270123092227788863)** (129 条消息🔥🔥):

> - `多数据集模型训练问题`
> - `OpenAI 领导层变动`
> - `Flux AI 模型性能`
> - `Open Medical Reasoning Tasks 项目`
> - `MiniCPM-Llama3 VLM 能力`

- **多数据集训练：灾难的秘诀？**：一位用户在多个 session 中以极小的学习率使用不同数据集训练模型，导致模型“练废了”，出现了灾难性遗忘（catastrophic forgetting），而相比之下，在单个合并数据集上使用较大的学习率则没有这种问题。
  - “累积误差”和“过拟合”被讨论为潜在原因，其中一个建议是训练过程中达到了低性能的局部最小值。
- **OpenAI 失去高层领导**：据 [新闻文章](https://www.theinformation.com/articles/trio-of-leaders-leave-openai) 报道，三位领导人已离开 OpenAI，这暗示了公司发展轨迹可能发生转变。
- **Flux AI 在文本和图像生成方面展现潜力**：据报道，Flux AI 模型（尤其是免费的 'Schnell'）在图像生成的连贯性方面击败了 **Midjourney 6**，表明模型性能取得了重大进展。
  - 尽管存在一些细微的拼写错误，这些模型仍备受好评，生成的图像达到了非凡的逼真度和清晰度。
- **Open Medical Reasoning 项目启动**：该项目由 [Open Life-Science AI](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks) 发起，邀请医学和 AI 社区贡献力量，为 LLM 开发医学推理任务。
- **MiniCPM-Llama3 推进多模态前沿**：MiniCPM-Llama3 2.5 现在支持多图输入，并在 OCR 和文档理解等任务中展现出巨大潜力，为多模态交互提供了强大的能力。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/maximelabonne/status/1820746013503586669">Maxime Labonne (@maximelabonne) 的推文</a>：🦙✨ BigLlama-3.1-1T-Instruct 听说 405B 参数还不够... 我很高兴向大家介绍一个拥有 1,000,000,000,000 参数的扩展版 Llama 3.1。现在已在 @hugg 上线...</li><li><a href="https://x.com/fofrAI/status/1820878455266816260">fofr (@fofrAI) 的推文</a>：🤯 &gt; PPT 演示文稿，幻灯片标题为 “Flux AI 拥有新技能”，三个要点：“擅长文本”、“提示词理解”、“惊人的图像”</li><li><a href="https://x.com/aadityaura/status/1820617406970278272?s=46">Aaditya Ura ( looking for PhD ) (@aadityaura) 的推文</a>：激动人心的消息！🎉 介绍 Open Medical Reasoning Tasks 项目！受 @NousResearch 和 @Teknium1 的启发，@OpenLifeSciAI ( Open Life-Science AI ) 正在启动一个开放、协作的倡议...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3">HuggingFaceM4/Idefics3-8B-Llama3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1elgr2x/new_open_llm_leaderboard_champion/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekte84/generated_with_flux1_pro_and_schnell/">使用 Flux.1 Pro 和 Schnell 生成</a>：由 u/Sea_Law_7725 发布于 r/StableDiffusion • 370 赞和 77 条评论</li><li><a href="https://github.com/black-forest-labs/flux/issues/9)">Issues · black-forest-labs/flux</a>：FLUX.1 模型的官方推理库。通过在 GitHub 上创建账号为 black-forest-labs/flux 的开发做出贡献。</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ekte84/generated_w">使用 Flux.1 Pro 和 Schnell 生成</a>：由 u/Sea_Law_7725 发布于 r/StableDiffusion • 372 赞和 77 条评论</li><li><a href="https://github.com/OpenBMB/MiniCPM-V/issues/233">MiniCPM-V 在多轮对话中针对多图输入的微调💡 [请求] - &lt;title&gt; · Issue #233 · OpenBMB/MiniCPM-V</a>：起始日期 | Start Date 无响应 实现PR | Implementation PR 无响应 相关Issues | Reference Issues 针对多轮对话中的多图输入 摘要 | Summary for multi-image input during a mul...</li></ul></div>

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1270123471300464783)** (19 messages🔥):

> - `Fine-tuning Libraries`
> - `Insurance Sector Fine-Tuning`
> - `Hosting Llama 450b`
> - `Inference Stack and Resources`
> - `Bottleneck in Inference/Training`

- **Axolotl 等 Fine-tuning 工具受到关注**：一位用户询问大多数人是使用库进行 Fine-tuning 和训练，还是编写独特的脚本；另一位用户回答称 **Axolotl** 是一个热门选择。
- **保险行业寻求定制化 AI 解决方案**：一位成员询问了针对 **保险行业** Fine-tuning AI 模型的情况。
- **探索 Llama 450b 托管选项**：一位成员询问有哪些公司提供按需付费（pay-as-you-go）访问的 **Llama 450b** 托管服务，并提到 Groq 需要企业账号。
- **推理栈入门**：一位用户请求关于入门推理栈（Inference Stack）和 **vLLM** 的资源。
- **理解推理和训练瓶颈**：有人提出了关于 **Inference/Training** 瓶颈的问题；回复指出在 Batch Size 为 1 时，内存是瓶颈。

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1270115994676756523)** (7 messages):

> - `Synthetic task generation`
> - `Open Medical Reasoning Tasks project`
> - `System 2 Reasoning Link Collection`

- **思考合成任务生成的改进**：一位用户表达了对增强合成任务生成的思考，旨在超越当前 LLM 能力的极限。
- **Open Medical Reasoning Tasks 项目受到启发**：受 Open Reasoning Tasks 项目启发，医疗版项目已启动，并呼吁医疗社区在 [GitHub](https://github.com/openlifescience-ai/Open-Medical-Reasoning-Tasks) 上做出贡献。
  - 该倡议旨在创建全面的医疗推理任务，同时推动 AI 在医疗保健领域的发展。
- **被纳入 System 2 Reasoning 链接合集**：Open Medical Reasoning Tasks 项目也被引用在 [System 2 Reasoning Link Collection](https://github.com/open-thought/system-2-research) 中，增强了可见性和协作。
  - 该合集旨在汇总对系统化推理研究具有重要意义的资源。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://x.com/aadityaura/status/1820617406970278272?s=46">Aaditya Ura ( looking for PhD ) (@aadityaura) 的推文</a>：令人兴奋的消息！🎉 介绍 Open Medical Reasoning Tasks 项目！受 @NousResearch 和 @Teknium1 的启发，@OpenLifeSciAI ( Open Life-Science AI ) 正在启动一个开放的协作倡议……</li><li><a href="https://github.com/open-thought/system-2-research">GitHub - open-thought/system-2-research: System 2 Reasoning Link Collection</a>：System 2 Reasoning 链接合集。通过创建账号为 open-thought/system-2-research 的开发做出贡献。</li></ul></div>

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1270094956945604620)** (128 messages🔥🔥):

> - `Web Dev to AI Engineer Transition`
> - `NVIDIA AI Scraping Controversy`
> - `John Schulman's Departure from OpenAI`
> - `OpenAI DevDay Events`
> - `Structured Outputs in OpenAI API`

- **Web 开发者转型为 AI Engineering**：关于 Web 开发者转型为 AI Engineer 可行性的热烈讨论凸显了对 AI Engineer 日益增长的需求，这源于 ML 专家短缺以及探索 AI 集成的公司不断增加。
  - 尽管职位描述要求具备 ML 专业知识，但据报道，许多职位由具有深厚 Web 开发背景的人员担任，因为公司优先考虑 API 集成技能，而非深厚的 ML 知识。
- **NVIDIA 因 AI 数据实践面临审查**：据报道，NVIDIA 为了 AI 目的进行大规模数据抓取，每天处理“一个人一生”长度的视频内容，尽管员工对此表示伦理担忧。[泄露的文件和 Slack 消息](https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/)表明，这一活动得到了公司最高层的批准。
- **John Schulman 离开 OpenAI 加入 Anthropic**：John Schulman 宣布在效力近九年后离开 OpenAI，旨在 Anthropic 更多地专注于 AI Alignment 研究。[他强调](https://x.com/johnschulman2/status/1820610863499509855)他的决定是个人选择，并非因为 OpenAI 缺乏支持。
- **OpenAI 宣布全球 DevDay 巡回活动**：OpenAI 将在旧金山、伦敦和新加坡举办 DevDay 活动，包括动手实践环节和演示，以展示开发者使用 OpenAI 工具构建的应用。[这一举措](https://openai.com/devday/)是 OpenAI 致力于与全球开发者社区互动的一部分。
- **OpenAI API 现已支持 Structured Outputs**：OpenAI 在其 API 中引入了 Structured Outputs 功能，确保模型输出遵循精确的 JSON Schemas，将 Schema 可靠性从 86% 提高到 100%。[该公告](https://x.com/michpokrass/status/1820881057824305567)强调了在增强模型响应可预测性方面迈出的重要一步。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/jason_koebler/status/1820493304490074391">Jason Koebler (@jason_koebler) 的推文</a>：来自 @samleecole 的独家新闻：泄露的 Slack 记录和文档揭示了 NVidia AI 抓取的惊人规模：每天抓取相当于 80 年——即“一个人一生”长度的视频。已获得最高层批准...</li><li><a href="https://x.com/OpenAIDevs/status/1820542222259073137">OpenAI Developers (@OpenAIDevs) 的推文</a>：我们将举办 OpenAI DevDay 巡回活动！今年秋天，欢迎在旧金山、伦敦或新加坡加入我们，参加实操环节、演示和最佳实践分享。与我们的工程师见面，了解全球开发者如何...</li><li><a href="https://x.com/TwoWeeksLOL/status/1820536638268948750">Two Weeks LOL (@TwoWeeksLOL) 的推文</a>：@MKBHD 噢，不...</li><li><a href="https://x.com/michpokrass/status/1820881057824305567">Michelle Pokrass (@michpokrass) 的推文</a>：很高兴宣布 Structured Outputs —— 我们 API 的最新功能。模型输出现在将可靠地遵循您精确的 JSON Schema，准确匹配参数和类型。Schema 可靠性...</li><li><a href="https://x.com/abacaj/status/1820883396077482087">anton (@abacaj) 的推文</a>：有意思... 新模型还包含了相当大的降价。引用 OpenAI Developers (@OpenAIDevs)：在 API 中引入 Structured Outputs —— 模型输出现在遵循开发者提供的 JSON ...</li><li><a href="https://x.com/_philschmid/status/1820715040191750370">Philipp Schmid (@_philschmid) 的推文</a>：“Deep Reinforcement Learning from Human Preferences”和“Proximal Policy Optimization Algorithms”是 LLM 中现代 RLHF 基础的一部分。</li><li><a href="https://x.com/tszzl/status/1714357380413264044?s=46">roon (@tszzl) 的推文</a>：OpenAI 里所有能进行眼神交流的人都是在过去 6 个月内加入的，他们的眼神交流让我感到不舒服。</li><li><a href="https://x.com/johnschulman2/status/1820610863499509855">John Schulman (@johnschulman2) 的推文</a>：我今天向 OpenAI 的同事们分享了以下便签：我做出了离开 OpenAI 的艰难决定。这个选择源于我希望加深对 AI Alignment 的关注，并开始一个...</li><li><a href="https://x.com/_mira___mira_/status/1820625134354669697?s=46">Mira (@_Mira___Mira_) 的推文</a>：未找到描述</li><li><a href="https://x.com/aizkmusic/status/1820594845792051391?s=46">Aizk ✡️ (@Aizkmusic) 的推文</a>：@BigTechAlert @ChatGPTapp @TarunGogineni 他的 LinkedIn 简介很棒</li><li><a href="https://news.ycombinator.com/item?id=41174306">未找到标题</a>：未找到描述</li><li><a href="https://writer.com/use-cases/ecommerce/">电子商务与零售</a>：探索创新的电子商务和零售公司如何使用 Writer 创作有效的品牌内容，从首次接触到最终销售。</li><li><a href="https://news.ycombinator.com/item?id=41173964">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2307.09702">Efficient Guided Generation for Large Language Models</a>：在本文中，我们展示了如何根据有限状态机的状态转换，对神经文本生成问题进行建设性的重新定义。该框架带来了一个高效的...</li><li><a href="https://x.com/jxmnop/status/1820876333154759091">jack morris (@jxmnop) 的推文</a>：关于 Extropic AI 的一个小趣闻 > 对他们好奇有一段时间了 > 有个互关推友是这家公司的工程师/研究员 > 经常发关于 Energy-based Modeling 和 LM-quant 的推文...</li><li><a href="https://github.com/simonw/datasette">GitHub - simonw/datasette：一个用于探索和发布数据的开源多功能工具</a>：一个用于探索和发布数据的开源多功能工具 - simonw/datasette</li><li><a href="https://x.com/NickADobos/status/1820513765823250730">Nick Dobos (@NickADobos) 的推文</a>：关于用 AI 写代码的精彩文章，很喜欢这张图表。引用 Erik Schluntz (@ErikSchluntz)：用 AI 替代我的右手（我是如何在打着石膏的情况下，每周为工作编写数千行代码的）...</li></ul></div>

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1270101558738157713)** (1 条消息):

> - `OpenAI DevDay 2023`
> - `Developer engagement`
> - `Global developer events`

- **OpenAI DevDay 开启巡回活动**：OpenAI 宣布 **DevDay** 将于今年秋季前往**旧金山、伦敦和新加坡**等主要城市，开展动手实践环节和演示，邀请开发者与 OpenAI 工程师进行交流。
  - 此次活动为开发者提供了一个独特的机会，可以学习最佳实践，并见证全球同行如何利用 OpenAI 的技术。
- **与全球 OpenAI 工程师建立联系**：鼓励开发者在即将举行的 DevDay 活动中与 OpenAI 工程师会面，了解 AI 的最新进展如何在全球范围内得到应用。
  - 这些活动还为参与者提供了一个协作和交流 AI 开发领域创新想法的平台。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1270275479793827891)** (86 条消息🔥🔥):

> - `Desktop ChatGPT App for Windows`
> - `OpenAI Structured Outputs`
> - `Llama 3.1 Model and API`
> - `ChatGPT Vision and 4o Mini`
> - `Bing AI Image Creator`

- **[ChatGPT 桌面应用与 Search GPT 发布](https://drinkoblog.weebly.com/)**：成员们讨论了 Windows 版 **ChatGPT 桌面应用**的预计发布日期以及 **Search GPT 的公测发布**，并暗示这些信息源自 Sam Altman。
- **[Structured Outputs 改进响应格式化](https://platform.openai.com/docs/guides/structured-outputs/examples?context=ex3)**：OpenAI 推出了 **Structured Outputs**，能够交付与提供的 schema 保持一致的 JSON 响应，增强了 API 交互体验。
  - **Python 和 Node** 的 SDK 已提供原生支持，模型承诺生成一致且结构化的输出，同时价格更便宜。
- **Llama 3.1 模型可免费本地使用**：成员们确认，只要不通过 API 服务使用，**Llama 3.1** 可以在本地免费运行。
  - 本地部署涉及下载模型并通过自定义环境运行，虽然受限于硬件性能，但可以实现零成本操作。
- **ChatGPT Vision 模型现已降价**：新的 **ChatGPT Vision 模型** 降价 50%，承诺提供比以往版本更实惠的访问权限。
  - 尽管它是对 4o mini 的改进，但一些用户对降价后可能带来的性能权衡表示疑虑。
- **Bing AI 图像生成器使用 DALL-E 3**：澄清了 **Bing AI Image Creator** 依赖于 **DALL-E 3**，尽管一些用户注意到输出质量存在不一致的情况。

**提到的链接**：[Assistant GPT - 我可以从云存储中执行知识检索吗？](https://stackoverflow.com/questions/78839847/assistant-gpt-can-i-perform-knowledge-retrieval-from-a-cloud-storage)：我有一些文件存储在云端（onedrive），想对它们进行知识检索。是否可以集成一个 Assistant 直接从……执行知识检索？

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1270164994192969758)** (16 条消息🔥):

> - `Search GPT release`
> - `Photo upload limit for members`
> - `AI in gaming`
> - `GPT-4o model update`
> - `Structured outputs announcement`

- **Search GPT 正式发布**：一位成员确认 **Search GPT** 已分发给用户，对有关其发布的询问给出了肯定回答。
- **照片上传限制令成员感到沮丧**：关于照片上传限制的讨论浮出水面，一位成员指出**即使是付费用户也面临此类上传限制**。
- **AI 将彻底改变游戏体验**：一位成员设想像 BG3 或 Pathfinder 这样的游戏可以利用生成式 AI 来实现独特的角色设计和动态 NPC 交互，从而增强**玩家沉浸感**。
- **GPT-4o 模型更新引发关注**：一位用户注意到 **ChatGPT-4o** 的响应行为发生了变化，随后社区确认 2024-08-06 发布了新模型。

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息):

darthgustav.: 使用 Python 工具并从上传的文件中导入数据。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息):

darthgustav.: 使用 Python 工具并从上传的文件中导入数据。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1270133269052002517)** (82 条消息🔥🔥):

> - `LLM 问题：GPT-4 Turbo vs. 4o`
> - `内容排序与推荐引擎`
> - `Perplexity AI 的 PDF 上传错误`
> - `应用稳定性与功能变更`
> - `Felo vs. Perplexity Pro 订阅`

- **4o 难以维持对话流**：用户对 **GPT-4o** 无法维持对话连贯性表示沮丧，称其会机械地重复过去的指令，且不理会新的指令。
  - 他们声称 **Sonnet** 模型立即为 **4o** 的行为道歉，凸显了感知到的缺陷。
- **着手内容排序项目**：一位用户详细介绍了一个大学项目，旨在开发一个**内容排序与推荐引擎**，用于分析数据库中的内容并确定其优先级。
  - 其他人建议在该项目尝试探索 **RAG** 平台和**本地模型**。
- **PDF 上传中的 Token 限制困扰**：由于“Token 计数失败”错误，用户在上传**大型 PDF** 时遇到问题，特别是当文件超过一定大小（100-200k tokens）时。
  - 将 PDF 转换为 **TXT 格式**似乎可以缓解这个问题，从而规避 Token 限制。
- **Perplexity App 功能消失**：一些用户报告称，在 **Perplexity Pro app** 中，切换 **LLM** 和访问**库收藏 (library collections)** 等功能会突然消失又重新出现。
  - 这些间歇性问题引起了困惑和沮丧，尽管功能通常会自发恢复。
- **免费 Pro 版与 Felo 的对比**：一位用户对比测试了 **Felo 与 Perplexity** 的免费版和 Pro 版，结果褒贬不一，并提到 Felo 在某些 Perplexity 失败的情况下提供了正确答案。
  - 他们指出，兑换的 1 个月免费 Pro 订阅限制了更改 **LLM** 的能力，从而限制了全面的对比测试。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://uncovr.app>">未找到标题</a>：未找到描述</li><li><a href="https://github.com/inulute/perplexity-ai-app/releases">Releases · inulute/perplexity-ai-app</a>：Perplexity AI 桌面应用，由 Electron 驱动，将 AI 语言处理的魔力带到您的桌面。- inulute/perplexity-ai-app</li><li><a href="https://felo.ai/search/PALsa8DEHJaiJcU6DYi4Q9">当汤姆的葬礼举行时，他的父亲没有参加。现在他的父亲去世了，汤姆也没有出现在他父亲的葬礼上。汤姆是不是太过分了？</a>：你描述的情况涉及人际关系和个人选择之间复杂的相互作用。以下是一些需要考虑的点：### 背景与...</li></ul></div>

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1270168758467432448)** (7 条消息):

> - `NVIDIA Blackwell GPU 延迟`
> - `数字记忆与 AI`
> - `沃霍尔价值 2600 万美元的数字肖像登上 YouTube`
> - `探索 Perplexity AI 的功能`

- **NVIDIA Blackwell GPU 因设计失误推迟**：NVIDIA 的[下一代 Blackwell GPU](https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ) 由于连接单个 Superchip 上两个 GPU 的处理器芯片设计缺陷而推迟，需要重新设计和验证。
- **用技术质疑数字记忆**：目前没有科学证据支持石头等物体具有记忆；然而，AI（如 DeepMind 的进展）正试图模拟人类的记忆回放过程，尽管目前仍处于实验阶段。
- **沃霍尔价值 2600 万美元的数字肖像在 YouTube 上备受关注**：一段 [YouTube 视频](https://www.youtube.com/embed/ZLEuncAV70U) 重点介绍了安迪·沃霍尔（Andy Warhol）以 2600 万美元售出的数字肖像，引发了关于艺术、技术和价值认知的讨论。
- **发现 Perplexity AI 的导航功能**：[Perplexity AI 平台](https://www.perplexity.ai/home) 提供多种界面选项，包括带有图像上传和高级 AI 功能等特性的 Pro 升级版。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://www.youtube.com/embed/ZLEuncAV70U">YouTube</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/nvidia-blackwell-s-delay-expla-kjKmWq15SdKcDJAgGn01EQ">NVIDIA Blackwell 延迟详解</a>：NVIDIA 的下一代 Blackwell GPU 遭遇延迟，主要是由于设计和制造问题。以下是延迟的主要原因：...</li><li><a href="https://www.perplexity.ai/search/is-naturland-s-tobotronc-the-l-GVann50ESpqyNuB4wT4qvw">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/how-can-i-get-a-summary-of-a-b-46KrvDREQKeVwv4VwBO2Lw">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/how-does-llama-3-1-405b-s-perf-YIDs8nm2TuuJzP4ILbY1BA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/consigliami-una-tastiera-mecca-bBenHhBBQUe7YmplIxO6IA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/apa-saja-benda-yang-mengandung-XdLR2Ja0TB.hH1hwSOy2DA">apa saja benda yang mengandung karbon</a>：含有碳的物体非常多样，可以在日常生活的各种形式中找到。以下是一些物体的例子...</li></ul></div>

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1270118041962942547)** (8 条消息🔥):

> - `API 数据损坏`
> - `API 模型弃用`
> - `API 502 错误问题`

- **API 结果损坏令用户沮丧**：一位用户报告在尝试撰写文章时从 API 收到损坏的结果，只有前几行是正确的，随后便变成了乱码。
  - 他们指出，在 Perplexity 网站上重新构建整个 Prompt 时，不会出现该问题。
- **即将于 2024 年 8 月弃用的 API 模型**：一位成员询问了 2024 年 8 月的 API 模型弃用情况，共享文档确认了受影响的模型及其参数。
  - `llama-3-sonar-small-32k` 系列模型定于 **2024 年 8 月 12 日** 弃用。
- **用户面临 API 502 错误**：一位用户在查询 Perplexity AI API 时遇到了 **502 错误**，表明可能存在服务中断。
  - 然而，另一位用户引用了[服务状态页面](https://status.perplexity.com/)，该页面报告当时没有中断通知。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://labs.perplexity.ai">Perplexity Labs</a>：未找到描述</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>：Perplexity 状态</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：Perplexity 模型 模型参数量 上下文长度 模型类型 llama-3-sonar-small-32k-online 8B 28,000 Chat Completion llama-3-sonar-small-32k-chat 8B 32,768 Chat Completion llama-3-sonar-large-32...</li></ul></div>

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1270325700599091210)** (1 条消息):

> - `Mechanistic anomaly detection`
> - `Adversarial examples in image classifiers`
> - `Eleuther's quirky language models`
> - `Attribution patching technique`

- **Mechanistic anomaly detectors 测试结果褒贬不一**：Eleuther AI 测试了 **Mechanistic anomaly detection** 技术，正如最近的一篇 [blog post](https://blog.eleuther.ai/mad_research_update/) 中详述的那样，这些技术在表现上并未始终优于传统的非机械论基准 (non-mechanistic baselines)。
  - 在评估整个数据批次 (batches) 时获得了更好的性能，但并非所有任务都显示出改进，这突显了未来研究的方向。
- **异常检测器的 Adversarial robustness 尚未测试**：现成技术在检测图像分类器中的 Adversarial examples 时表现轻松，但尚未测试异常检测器本身的 **Adversarial robustness**。
  - *我们的异常检测器可能需要进一步评估对抗鲁棒性*，这表明了一个潜在的持续调查领域。
- **从 Quirky language models 中诱导潜藏知识 (Eliciting latent knowledge)**：Eleuther AI 在一篇 [新论文](https://arxiv.org/abs/2312.01037v3) 中发表了关于微调 (Finetuning) 语言模型以使其表现出“古怪 (quirky)”行为的研究结果，探讨了行为检测问题。
  - 他们使用简单的异常检测技术区分模型行为，处理 **Alice** 和 **Bob** 的启发式 (heuristic) 响应行为，这与 [MAD problem](https://www.lesswrong.com/posts/n7DFwtJvCzkuKmtbG/a-gentle-introduction-to-mechanistic-anomaly-detection) 相关。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://blog.eleuther.ai/mad_research_update/">Mechanistic Anomaly Detection Research Update</a>：关于 Mechanistic anomaly detection 正在进行的工作的中期报告</li><li><a href="https://github.com/EleutherAI/cupbearer/tree/attribution_detector">GitHub - EleutherAI/cupbearer at attribution_detector</a>：一个用于 Mechanistic anomaly detection 的库。通过在 GitHub 上创建账号为 EleutherAI/cupbearer 的开发做出贡献。</li></ul></div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1270174953135935569)** (36 messages🔥):

> - `SB1047 (AI Safety Act) opposition`（反对 SB1047 AI 安全法案）
> - `Concerns with AI regulation and innovation`（对 AI 监管与创新的担忧）
> - `Anthropic's response to SB1047`（Anthropic 对 SB1047 的回应）
> - `AAAI conference submission relevance`（AAAI 会议投稿的相关性）
> - `Watermarking and AI safety laws`（水印与 AI 安全法律）

- **反对 SB1047 的呼声日益高涨**：一封反对 SB1047（AI Safety Act）的公开信正在流传，警告该法案可能通过潜在地禁止开源模型和威胁学术自由，对开源研究和创新产生负面影响。
  - 鼓励支持者签署一份反对该法案的 [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSewflVHn1zoNeHHJq3SaKvlwPy7PLT1Vcu_WoULqcHSSjvX1w/viewform)，该法案因其潜在的法律后果和经济影响而受到批评。
- **关于 AI 监管影响的辩论引发热议**：讨论揭示了对 SB1047 等 AI 安全法规的模糊性及其潜在负面影响的重大担忧，特别是对阻碍研究和法律不确定性的恐惧。
  - “这份学术信函似乎并不比 YC a16z 的那份更有根据（大多是言辞，没有证据），”一位成员总结道，强调了关于该法案影响的长期辩论和对法案效果的各种解读。
- **Anthropic 对 SB1047 的细致看法**：Anthropic 对 SB1047 的回应提供了一个平衡的视角，在承认监管必要性的同时，也指出了该法案扼杀创新的可能性。
  - 一些成员认为这一回应是对 AI 治理和责任更广泛讨论的理智贡献。
- **AAAI 投稿的相关性受到质疑**：有人提出了关于向 AAAI 会议投稿价值的问题，成员们建议它可能被视为那些被认为不够强大而无法进入其他会议的论文的去处。
- **AI 输出水印面临审查**：成员们对立法强制要求 AI 输出水印表示怀疑，指出了技术障碍以及水印被删除或篡改的可能性。
  - 虽然有些人认为法律激励是技术解决方案的驱动力，但其他人警告不要制定可能对开源努力产生负面影响的过早法律。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anth">DocumentCloud</a>：未找到描述</li><li><a href="https://www.documentcloud.org/documents/25003075-sia-sb-1047-anthropic">DocumentCloud</a>：未找到描述</li><li><a href="https://safesecureai.org/responseletter">Letter to YC &amp; a16z | SB 1047 - Safe &amp; Secure AI Innovation</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSewflVHn1zoNeHHJq3SaKvlwPy7PLT1Vcu_WoULqcHSSjvX1w/viewform">Students, Faculty, and Scientists Against SB 1047 (AI Safety Act) Open Letter Signature Form</a>：这是一个提供签名的表格，用于支持加州大学教职员工和学生反对加州 SB 1047 的公开信，这是一项试图监管“AI 安全”的灾难性法律……</li></ul></div>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1270111334914330777)** (40 条消息🔥):

> - `Meta's AI network`
> - `Distributed AI Training at Scale`
> - `Search efficiency in AI models`
> - `Differentiability in search techniques`
> - `Compute-optimal inference methods`

- **Meta 为大规模模型训练构建 AI 网络**：在 [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) 上，Meta 展示了其连接数千个 GPU 的网络基础设施，这对于训练像 [LLAMA 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) 这样的模型至关重要。
  - 他们关于 [RDMA over Ethernet for Distributed AI Training](https://dl.acm.org/doi/10.1145/3651890.3672233) 的论文重点介绍了设计和运营全球最大的 AI 网络之一的经验。
- **关于 AI 模型搜索效率的辩论**：参与者讨论了潜空间（latent space）搜索与离散空间搜索的有效性，认为在模型的潜空间中进行搜索可能会绕过世界模型评估中的瓶颈。
  - 建议包括采用 VQ 方法进行高效的模型潜空间搜索，从而激励学习可组合的子解决方案。
- **对奇特搜索技术的可微性提出质疑**：虽然有些人主张使用可微搜索技术，但其他人认为更简单的方法往往表现更好，并引用无监督 MT 为例，其中基础方法比复杂方法效果更好。
  - 辩论强调了模型评估函数中可微性与计算效率之间的权衡。
- **通过采样策略扩展推理计算**：研究表明，增加生成的样本量可以提高推理性能，特别是在答案可以自动验证的场景中。
  - 研究探索了计算最优（compute-optimal）的推理策略，如树搜索（Tree Search）算法，显示较小的模型可以实现良好的计算-性能权衡。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>：扩展用于训练语言模型的计算量显著提高了它们的能力。然而，在推理时，我们通常将计算量限制在仅一次尝试中……</li><li><a href="https://arxiv.org/abs/2408.02666">Self-Taught Evaluators</a>：基于模型的评估是模型开发成功的核心——既作为训练的奖励模型，也作为人类评估的替代。训练此类评估器的标准方法是……</li><li><a href="https://arxiv.org/abs/2408.00724">An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models</a>：关于模型大小和计算预算的大语言模型（LLMs）最佳训练配置已得到广泛研究。但如何在推理过程中优化配置 LLMs……</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>：你只需要抽取更多样本</li><li><a href="https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/">RoCE networks for distributed AI training at scale</a>：AI 网络在将数万个 GPU 互连方面发挥着重要作用，构成了训练的基础设施，支持具有数千亿参数的大型模型……</li></ul></div>

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1270272866587512915)** (4 条消息):

> - `Training Instability`
> - `Experiment Averaging`
> - `Learning Rate Adjustments`

- **训练不稳定性盖过了对双重下降（Double Descent）的担忧**：一位讨论成员认为，观察到的问题更有可能是由于**噪声/训练不稳定性**，而非**双重下降**现象。
- **实验平均化的理由**：建议进行 **3 到 5 次** 实验并取平均结果，以排除异常情况。
- **学习率作为稳定性因素**：为了降低训练稳定性问题的可能性，一位参与者建议如果现象持续存在，应降低**学习率（Learning Rate）**。

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1270115079051808778)** (5 条消息):

> - `SAEs 的现状`
> - `扩展 SAEs 的研究`
> - `SAELens 库`
> - `Transformer Circuits 的最新进展`

- **探索 Transformer 研究中 SAEs 的现状**：一位用户寻求关于 Structural Attention Equations (*SAEs*) 最新进展的指导，并被引导至一些基础性和近期的工作，例如 [Monosemantic Features 论文](https://transformer-circuits.pub/2023/monosemantic-features/index.html) 和 [Superposition 论文](https://transformer-circuits.pub/2022/toy_model/index.html)，这些论文为 SAEs 提供了背景。
  - 其他资源包括向真实规模 SAEs 的演进，例如 [Anthropic 关于扩展单语义性 (scaling monosemanticity) 的论文](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)。
- **扩展 SAEs 的多种方法**：讨论强调了不同论文中 SAEs 的扩展，例如一个用户群体将模型从玩具模型扩展到 13B 参数，并与 Anthropic 和 OpenAI 正在进行的研究相结合，详见[相关论文](https://arxiv.org/abs/2406.04093)。
  - OpenAI 已尝试扩展至 **GPT-4**，侧重于方法论的改进，而 EleutherAI 正积极在 **LLaMA 3.1 405B** 上进行训练。
- **利用 SAELens 库**：社区讨论了 SAELens，这是一个为训练和分析 *SAEs* 而创建的库，其在 Neuronpedia 中的可视化效果因其深度而备受推崇。
  - 此外，EleutherAI 的贡献包括一个与 NNsight 集成的 [auto-interp 库](https://github.com/EleutherAI/sae-auto-interp)，尽管其扩展潜力尚存疑问。
- **Transformer Circuits 中 SAE 进展概述**：一份关于 SAE 领域全景的概述文档被分享出来，作为新手的全面入门指南，托管在在线协作平台 [Google Docs](https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit) 上。
  - 该文档提供了历史背景和最新进展，尽管可能遗漏了该领域极少数的最前沿动态。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.google.com/document/d/1lHvRXJsbi41bNGZ_znGN7DmlLXITXyWyISan7Qx2y6s/edit#heading=h.j9b3g3x1o1z4">SAE Landscape</a>：SAE Landscape – 语言模型可解释性稀疏自编码器 (SAEs) 的有用出版物和工具集合。这是一个实时文档，我非常感谢...</li><li><a href="https://transformer-circuits.pub/2021/framework/index.html#notation">A Mathematical Framework for Transformer Circuits</a>：未找到描述</li></ul></div>

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1270115967858380883)** (8 条消息🔥):

> - `lm-eval-harness 用法`
> - `Batch size 与 loglikelihood_rolling`
> - `evalharness 中的 BOS token`
> - `从 JSON 输出中提取 Benchmark 名称`

- **为自定义模型使用 lm-eval-harness**：一位用户询问如何使用 **lm-eval-harness** 评估自定义架构的模型 checkpoint。另一位成员提供了一个 [GitHub 示例链接](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py)，展示了如何重写模型方法以确保与自定义模型类型的兼容性。
- **eval-harness 中的批处理**：一位用户质疑 'loglikelihood_rolling' 在 Huggingface 模型类中是否遵循 Batch size，认为它可能正在一次处理一个请求。
- **evalharness 中的特殊 Token**：关于 **evalharness** 是否默认添加 BOS token 存在困惑，因为默认的 tokenizer 行为是 `add_special_tokens=True`。
  - 一位用户确认，即使 **BOS tokens** 可能不会出现在生成的样本文件中，默认设置也会包含它们。
- **从 JSON 中提取 Benchmark 名称**：一位成员讨论了如何通过访问 `results` 键从 JSON 输出中查找 Benchmark 名称，该键包含另一个以 Benchmark 名称为键、分数为值的字典。

 

**提到的链接**：[mamba/evals/lm_harness_eval.py at main · state-spaces/mamba](https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py)：Mamba SSM 架构。通过在 GitHub 上创建一个账户来为 state-spaces/mamba 的开发做出贡献。

 

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1270096212292534272)** (83 messages🔥🔥):

> - `GPU Out of Memory Issues` (GPU 显存不足问题)
> - `LangChain Integration Questions` (LangChain 集成问题)
> - `Automatic Code Review Challenges` (自动代码审查挑战)
> - `LangGraph Course Recommendations` (LangGraph 课程推荐)
> - `Mood2Music App Launch` (Mood2Music 应用发布)

- **GPU Out of Memory Quandaries**: 一位用户在尝试加载对于其 **8GB GPU vRAM** 来说过大的模型时遇到了 **内存溢出 (memory overflow)** 问题，需要关于调整或使用更小模型的建议。
  - 他们通过强制系统 **完全在 CPU 上运行** 解决了该问题，尽管这导致了性能下降。
- **Navigating LangChain Tool Integration**: 一位用户询问在文档不足的情况下，如何在 LangChain v2.0 中集成 **RunnableWithMessageHistory** 以进行聊天机器人开发。
  - 另一个查询探讨了根据一个教程线程在 Tool calling 期间存储 **message history** 的方法。
- **Automatic Code Review Position Miscalculations**: 使用 **GPT-4o** 进行的自动代码审查由于计数问题，难以正确评估 GitHub diffs 中的位置。
  - 一个建议是避免使用 Vision 模型，而是采用更偏向 **coding-specific** 的方法来解析和检索数据。
- **LangGraph Learning Pathways**: 对于在 LangGraph 概念上遇到困难的人，推荐了 **DeepLearning.ai** 和 **Udemy** 等在线资源。
  - 建议强调从 **基础课程** 开始以巩固理解，然后再进阶。
- **Mood2Music App Set to Resonate with Users**: 一款新应用 **Mood2Music** 发布，专注于根据用户心情提供音乐推荐，并集成了 **Spotify** 和 **Apple Music** 等平台。
  - 该应用声称通过创建个性化播放列表来增强用户的听歌体验，其特色功能包括用于自动心情检测的 AI 自拍分析。

<div class="linksMentioned"><p><strong>提及的链接</strong>:</p><ul><li><a href="https://mood2music.me">mood2music</a>: 未找到描述</li><li><a href="https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/">AI Agents in LangGraph</a>: 使用 LangChain 的 LangGraph 和 Tavily 的 Agentic search 构建 Agentic AI 工作流。直接向 LangChain 和 Tavily 的创始人学习。</li><li><a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>: Vector DB Comparison 是来自 VectorHub 的一个免费开源工具，用于比较向量数据库。</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot">Build a Chatbot | 🦜️🔗 Langchain</a>: 概览</li><li><a href="https://github.com/ollama/ollama/issues/3509">Can Ollama use both CPU and GPU for inference? · Issue #3509 · ollama/ollama</a>: 你想做什么？我想知道 ollama 是否支持在 Windows 上混合使用 CPU 和 GPU 运行？我知道我的硬件不足以运行 ollama，但我仍想使用部分能力...</li></ul></div>

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1270276986878693398)** (2 messages):

> - `AgentGenesis Project` (AgentGenesis 项目)
> - `Open Source Collaboration` (开源协作)

- **AgentGenesis Boosts AI Development**: **AgentGenesis** 是一个 AI 组件库，提供 **即插即用的代码片段 (copy-paste code snippets)** 以增强生成式 AI (Gen AI) 应用开发，承诺将效率提升 **10 倍**，并采用 **MIT 许可证**。
  - 项目功能包括一个包含 **RAG flows** 和 **QnA bots** 模板的全面代码库，由 [社区驱动的 GitHub 仓库](https://github.com/DeadmanAbir/AgentGenesis) 支持。
- **Call for Contributors to AgentGenesis**: AgentGenesis 正在寻求活跃的贡献者加入并增强其开源项目的持续开发，该项目强调社区参与和协作。
  - 鼓励感兴趣的开发者 [在 GitHub 仓库加星 (star)](https://github.com/DeadmanAbir/AgentGenesis) 并为可重用代码库做出贡献。

<div class="linksMentioned"><p><strong>提及的链接</strong>:</p><ul><li><a href="https://www.agentgenesis.dev/">AgentGenesis</a>: 复制并粘贴最热门的 AI Agents，并在你的项目中使用它们，无需从头开始编写。</li><li><a href="https://github.com/DeadmanAbir/AgentGenesis">GitHub - DeadmanAbir/AgentGenesis: Welcome to AgentGenesis, your source for customizable Gen AI code snippets that you can easily copy and paste into your applications.</a>: 欢迎来到 AgentGenesis，这是你获取可定制生成式 AI 代码片段的来源，你可以轻松地将其复制并粘贴到你的应用程序中。 - DeadmanAbir/AgentGenesis</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1270171674540769311)** (57 messages🔥🔥):

> - `John Schulman 加入 Anthropic`
> - `机密 Gemini 项目`
> - `Greg 从 OpenAI 休假`
> - `Claude 与 Gemini 的对比`
> - `AGI 对齐观点`

- **John Schulman 意外加入 Anthropic**：[John Schulman](https://x.com/johnschulman2/status/1820610863499509855) 宣布决定离开 OpenAI 加入 Anthropic，以专注于 AI 对齐（Alignment）和一线技术工作，并强调了他对新视角的渴望。
- **关于 Gemini 项目的泄露传闻引发成员兴趣**：成员们讨论了关于 OpenAI Gemini 项目的机密细节，对潜在的泄密以及 Gemini 2 的神秘性质表示惊讶。
- **Greg Brockman 从 OpenAI 的马拉松式工作中抽身休息**：[Greg Brockman](https://x.com/gdb/status/1820644694264791459?s=46) 宣布将从 OpenAI 休假（Sabbatical），这是他自共同创立公司 9 年来首次放松，引发了对其动机的猜测。
- **Claude 在用户体验上落后于 ChatGPT**：用户对 Claude 和 ChatGPT 进行了批判性对比，认为 Claude 的表现落后，类似于旧的 GPT-3.5 模型，而 ChatGPT 在灵活性和记忆力方面表现出色。
- **观点分歧引发关于 AI 对齐的辩论**：对话强调了 AI 对齐的不同方法，John Schulman 专注于 Prompt 遵循等实际问题，而 Jan Leike 则担心 AI 安全的更广泛影响。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://x.com/johnschulman2/status/1820610863499509855">John Schulman (@johnschulman2) 的推文</a>：我今天与 OpenAI 的同事分享了以下笔记：我做出了离开 OpenAI 的艰难决定。这个选择源于我希望加深对 AI 对齐的关注，并开始一段……</li><li><a href="https://fxtwitter.com/simonw/status/1820886987982987413?s=46">Simon Willison (@simonw) 的推文</a>：隐藏在公告底部的内容：“通过切换到新的 gpt-4o-2024-08-06，开发者在输入上节省了 50%（$2.50/1M input tokens），在输出上节省了 33%（$10.00/1M output tokens）……”</li><li><a href="https://x.com/gdb/status/1820644694264791459?s=46">Greg Brockman (@gdb) 的推文</a>：我将休假到年底。这是自 9 年前共同创立 OpenAI 以来第一次放松。使命远未完成；我们仍需构建一个安全的 AGI。</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1270539473066922105)** (6 messages):

> - `DALL-E 对阵挑战者`
> - `Flux Pro`
> - `Replicate 托管 Flux.1`
> - `图像生成模型对比`

- **DALL-E 在图像生成领域面临竞争**：讨论揭示了在竞争日益激烈的情况下，**DALL-E** 是否仍是领先的带有 API 的图像生成工具。
  - 一位成员对比较这些模型的标准感到好奇，暗示直觉或“氛围”（vibes）可能起到了重要作用。
- **Flux Pro 提供新颖体验**：一位用户描述 **Flux Pro** 与该领域的其他模型相比，具有*非常*不同的氛围。
  - *这不在于定量基准测试（Benchmarks），* 而更多在于主观体验。
- **Flux.1 已在 Replicate 上线**：成员们讨论到，受到一些人喜爱的 **Flux.1** 现在已由 Replicate 托管。
  - 这突显了关于托管方式如何影响可访问性和用户满意度的更广泛考量。

 

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages):

xeophon.: [https://x.com/sahir2k/status/1820791954508022019?s=46](https://x.com/sahir2k/status/1820791954508022019?s=46)

---

### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1270459908806017076)** (1 messages):

> - `模型性能中的数据依赖性`
> - `初创公司使用噪声数据`
> - `ICML 关于 Meta Chameleon 的讨论`

- **数据依赖性影响模型收益**：对话强调，模型从将数据分解为 ( (x, y_w) ) 和 ( (x, y_l) ) 等组件中获益，很大程度上取决于**数据噪声水平**。
- **初创公司青睐噪声数据策略**：*由于数据的噪声性质，初创公司往往更频繁地应用这些技术，* 这可能导致绕过标准的 SFT 流程。
- **ICML 交流中提到 Meta 的 Chameleon**：在 ICML 上，有人提到 **Meta Chameleon 项目的 Armen** 是此类数据策略的拥趸；然而，尚不清楚这些策略是否被用于**生产模型**。

 

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1270486884539174973)** (1 条消息):

> - `GPT4-4o 发布`
> - `带有 strict 模式的 Structured outputs`

- **GPT4-4o 在 OpenRouter 发布**：新模型 [GPT4-4o-2024-08-06](https://openrouter.ai/models/openai/gpt-4o-2024-08-06) 现已在 OpenRouter 上线。
- **Strict 模式下的 Structured Outputs 问题**：目前尚未完全支持带有 strict 模式的 **Structured outputs**，相关问题请在 [指定频道](https://discord.com/channels/1138521849106546791) 反馈。
  - 鼓励用户报告遇到的任何问题，以改进系统功能。

**提到的链接**：[GPT-4o (2024-08-06) - API, Providers, Stats](https://openrouter.ai/models/openai/gpt-4o-2024-08-06)：2024-08-06 版本的 GPT-4o 在 structured outputs 方面提供了改进的性能，并支持在 response_format 中提供 JSON schema。阅读更多 [此处](%5Bhttps://openai%5D(https://openai)。运行 GPT-4o (2024-08...

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1270098420023164939)** (62 条消息🔥🔥):

> - `AI 模型性能`
> - `GPT-4o-2024-08-06 更新`
> - `Token 使用与定价`
> - `Google Gemini 更新`
> - `API 成本计算`

- **AI 模型面临性能挑战**：一位成员测试了 **yi-vision** 和 **firellava**，但在单张测试图片的定价和性能方面，它们与 **haiku/flash/4o** 相比表现不佳。
  - 对话暗示了 **Google Gemini 1.5** 的价格变动，它很快将比上述效果较差的模型更便宜。
- **GPT-4o-2024-08-06 主打 Structured Outputs**：OpenAI 在其针对新模型 **gpt-4o-2024-08-06** 的 API 中引入了 structured outputs，承诺与之前的模型相比，具有更好且更具成本效益的 token 使用。
  - 预计 **JSON** 生成的一致性将得到提高，详情可通过 [OpenAI 博客](https://openai.com/index/introducing-structured-outputs-in-the-api/) 了解。
- **了解 Token 定价与节省**：通过切换到 **gpt-4o-2024-08-06**，开发者可以在输入上节省 50%，在输出上节省 33%，这比之前提供的产品更便宜。
  - 社区讨论了降低成本的潜在原因，包括效率提升和对投资者资源的使用。
- **讨论 API 成本计算方法**：关于 OpenRouter API 成本计算的讨论展开，共识是利用请求后的 `generation` 端点来获取确切详情。
  - 这些信息使用户能够通过评估使用情况来管理按需付费系统，而无需在流式回复中嵌入成本详情。
- **速率限制影响 Google Gemini 模型**：用户在使用 **Google Gemini Pro 1.5** 时遇到了问题，特别是由于 Google 严格的速率限制导致的 'RESOURCE_EXHAUSTED' 错误。
  - 这种情况需要调整对使用情况的预期，因为目前还没有针对速率限制约束的立即修复方案。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://chat-preview.lobehub.com>,">未找到标题</a>：未找到描述</li><li><a href="https://simonwillison.net/2024/Aug/6/openai-structured-outputs/">OpenAI：在 API 中引入 Structured Outputs</a>：OpenAI 提供 structured outputs 已经有一段时间了：你可以指定 `"response_format": {"type": "json_object" }}` 来请求一个有效的 JSON 对象，或者你可以使用...</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>：管理来自模型的响应</li><li><a href="https://status.anthropic.com">Anthropic 状态</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>：2024-08-06 版本的 GPT-4o 在 structured outputs 方面提供了改进的性能，并支持在 response_format 中提供 JSON schema。阅读更多 [此处](https://openai. 运行 GPT-4o (2024-08...</li></ul></div>

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1270432568948555776)** (1 条消息):

> - `与 CodiumAI 合作的研讨会`
> - `RAG 增强型编程助手`
> - `用于代码生成的 LlamaIndex`

- **参加 RAG 增强型编程助手研讨会**：[与 CodiumAI 合作的研讨会](https://lu.ma/ka5xtyqo) 即将举行，主题为 RAG 增强型编程助手！
  - 参与者需要注册并使用钱包验证 token 所有权方可参加。
- **在编程中探索结合 LlamaIndex 的 RAG**：正如即将举行的研讨会中所讨论的，检索增强生成 (RAG) 对于在 AI 生成的代码中实现**上下文感知 (contextual awareness)** 至关重要。
  - 本次会议将展示使用 LlamaIndex 基础设施的高级 RAG 方法，并提供维护**代码质量**和完整性的实际应用案例。

**提到的链接**：[LlamaIndex 研讨会：在大型生成式编程中使用 RAG 与 LlamaIndex · Zoom · Luma](https://lu.ma/ka5xtyqo)：检索增强生成 (RAG) 在实现 AI 生成代码的上下文感知方面发挥着核心作用，这对于采用……的企业至关重要。

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1270177453498830960)** (4 条消息):

> - `RabbitMQ 与 llama-agents`
> - `第二届 RAG-a-thon`
> - `LlamaIndex 中的 Workflows 功能`
> - `构建多 Agent 即服务 (Multi-agents as a Service)`

- **使用 RabbitMQ 构建多 Agent 系统**：@pavan_mantha1 的一篇博客演示了如何使用 [RabbitMQ](https://t.co/IOGpDWkY8A) 作为不同 Agent 之间的通信代理来构建本地多 Agent 系统，并集成了 @ollama 和 @qdrant_engine。
- **为 LlamaIndex 的第二届 RAG-a-thon 做好准备**：继首届活动成功举办后，LlamaIndex 将与 @pinecone 和 @arizeai 合作，在帕罗奥图的 @500GlobalVC 办公室举办另一场 RAG-a-thon。
- **掌握 LlamaIndex 中的复杂 Workflows**：在[一段新的 YouTube 视频](https://t.co/xuiuSMCmJF)中，@seldo 解释了在 LlamaIndex 中创建、运行和可视化 Workflows 的基础知识，以及如何管理它们的结构、循环、分支和状态。
- **Llama-agents 全面指南**：社区要求提供更多关于 llama-agents 的详细文档，这是构建多 Agent 即服务 (Multi-agents as a Service) 的核心仓库。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1270128013337432115)** (49 条消息🔥):

> - `用于 embeddings 的 HuggingFace Inference API`
> - `SimpleDirectoryReader PDF 加载`
> - `Vector DB 对比`
> - `llama_index 中 function_calling.py 的问题`
> - `OpenAI API 中的 Structured Outputs`

- **使用 HuggingFace Inference API 生成 embeddings**：一位成员询问了如何使用 HuggingFace Inference API 生成 embeddings，特别提到了 Jina.ai 的私有端点。
  - 另一位成员提供了 LlamaIndex 关于 embedding 示例的相关文档链接。
- **SimpleDirectoryReader 逐页加载 PDF**：SimpleDirectoryReader 将每个 PDF 页面加载为单独的文档，从而允许关联页面标签等元数据。
  - 分享了修改 `PDFReader` 设置的选项，包括将 PDF 视为单个文档的 Python 代码示例。
- **Vector DB Comparison 是一个有用的资源**：分享了 [Vector DB Comparison](https://superlinked.com/vector-db-comparison)，因为它在评估向量数据库方面非常有用。
  - 社区鼓励分享不同 VectorDB 的使用经验，以促进大家的学习。
- **LlamaIndex function_calling.py 导致 CI 问题**：LlamaIndex 中 `function_calling.py` 的 TypeError 导致 CI 流程失败，直到升级了 `llama-index-llms-bedrock-converse`。
  - 该问题被确定可能是由于过时的包依赖要求引起的，通过显式指定依赖项得到了解决。
- **支持 OpenAI 的 Structured Outputs**：当设置 `strict=True` 参数时，LlamaIndex 已经支持 OpenAI API 中的 Structured Outputs。
  - 虽然功能正常，但与非严格模式相比，它显著增加了延迟，一次调用的耗时明显长于使用 Pydantic 进行解析。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://superlinked.com/vector-db-comparison">Vector DB Comparison</a>：来自 VectorHub 的免费开源工具，用于对比向量数据库。</li><li><a href="https://github.com/run-llama/llama_index/blob/6eea66ed23fb85ee77664148a4c2b66720caabeb/pyproject.toml#L60">llama_index/pyproject.toml at 6eea66ed23fb85ee77664148a4c2b66720caabeb · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/text_embedding_inference/">Text Embedding Inference - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c12">GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - GitHub - run-llama/llama_index at 15227173b8c1241c9fbc761342a2344cd90c6593</li><li><a href="https://github.com/run-llama/llama_index/blob/15227173b8c1241c9fbc761342a2344cd90c6593/llama-index-core/llama_index/core/llms/function_calling.py#L125">llama_index/llama-index-core/llama_index/core/llms/function_calling.py at 15227173b8c1241c9fbc761342a2344cd90c6593 · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li></ul></div>

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1270216114999398435)** (29 messages🔥):

> - `Galileo Hallucination Index`
> - `Open Source vs Open Weights`
> - `Command R Plus Licensing`
> - `Mistral Licensing and Access`

- **Galileo 的幻觉指数引发辩论**：[Galileo's Hallucination Index](https://www.rungalileo.io/hallucinationindex) 的发布引发了关于将 LLM 分类为开源或闭源标准的讨论，特别是关注 Command R Plus 的分类是否准确。
  - 用户质疑 Command R Plus 是否真的是开源的，一些人主张在 **open weights**（开放权重）和完全开源模型之间做出更清晰的区分。
- **Command R Plus 许可引发争议**：Galileo 的回应澄清说，他们只有在模型支持商业用途时才将其归类为开源，并指出 Command R Plus 的 Creative Commons Attribution Non Commercial 4.0 许可是一个限制因素。
  - 随后引发了关于这一定义是否恰当的辩论，成员们建议为 **'open weights'** 设立一个不同于开源的新类别。
- **Mistral 开放权重：限制最少？**：与 AI 模型被贴上开放权重标签的普遍共识相反，一位成员指出 **Mistral 的模型** 是在 Apache 2.0 许可下提供的，这是一种更宽松、提供更大自由度的许可。
  - 确认此事的努力包括分享 [Mistral 官方文档](https://docs.mistral.ai/getting-started/open_weight_models/) 的链接，展示了他们在预训练和指令微调模型方面的透明度。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.mistral.ai/getting-started/open_weight_models/">Apache 2.0 models | Mistral AI Large Language Models</a>：我们开源了预训练模型和指令微调模型。这些模型没有针对安全性进行微调，因为我们希望赋能用户根据其用例测试和改进审核。为了更安全...</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&amp;utm_source=alpha_signal&amp;utm_campaign=sponsorship">LLM Hallucination Index - Galileo</a>：LLM 幻觉指数。一个针对 LLM 幻觉的排名与评估框架。</li><li><a href="https://www.rungalileo.io/hallucinationindex?utm_medium=paid&amp;utm_source=alpha_signal&amp;utm_campaign=sp">LLM Hallucination Index - Galileo</a>：LLM 幻觉指数。一个针对 LLM 幻觉的排名与评估框架。</li></ul></div>

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270432110078853241)** (3 messages):

> - `Contacting Dennis Padilla`

- **寻找 Dennis Padilla 的电子邮件**：一位成员在得知 Lauren 正在休假后试图联系 Dennis Padilla，但找不到他的电子邮件地址。
  - 另一位用户询问了邮件请求的背景，以便提供帮助。
- **无更多可用信息**：提供的消息中不包含更多详细主题以供进一步总结。
  - 因此，讨论主题缺乏多样性，无法详细阐述。

 

---

### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1270539094203568128)** (1 messages):

> - `Cohere Toolkit integration`
> - `Switching models`
> - `Third-party API usage`
> - `OpenAI integration`
> - `Gemini 1.5 compatibility`

- **将 Cohere Toolkit 与 Creative Corpus 集成**：一位成员提到在一个 AI 奖学金项目中使用 [Cohere Toolkit](https://cohere.ai/)，在 Confluence、烹饪笔记、酒庄记录或律师事务所案件笔记等多样化知识库上构建带有 RAG 的 LLM。
- **探索使用第三方 API 模型替代 Cohere**：有人提出了关于从 Cohere 模型切换到第三方 API（如 [OpenAI's Chat GPT](https://openai.com) 或 [Gemini 1.5](https://groq.com)）可行性的咨询。

 

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1270321995027972166)** (30 条消息🔥):

> - `InlineList 开发`
> - `Mojo 中的小缓冲区优化 (Small buffer optimization)`
> - `在 Mojo 中使用自定义加速器`
> - `开源 Mojo 中的 RVV 支持`

- **InlineList 凭借新特性取得进展**：随着所需新功能的加入，`InlineList` 的开发正在稳步推进，最近的一次 [merge](https://github.com/modularml/mojo/pull/2825) 凸显了这一点。
  - *技术优先级*似乎决定了在 `InlineList` 中引入 `__moveinit__` 和 `__copyinit__` 方法的时间表。
- **小缓冲区优化为 Mojo List 增加灵活性**：Mojo 为 `List` 引入了小缓冲区优化，通过使用类似 `List[SomeType, 16]` 的参数来分配栈空间。
  - [Gabriel De Marmiesse](https://github.com/modularml/mojo/pull/2825) 阐明，这一增强功能可能会取代对独立 `InlineList` 类型的需求。
- **自定义加速器期待 Mojo 的开源未来**：带有脉动阵列（systolic arrays）的 PCIe 卡和 CXL.mem 等自定义加速器被认为是 Mojo 开源后使用的潜在候选对象，硬件集成特性的对话特别强调了这一点。
  - 目前，使用 Mojo 替换自定义 Kernel 仍具挑战性，在 Mojo 支持 RISC-V 目标等特性之前，现有的流程（如 lowering PyTorch IR）仍将占据主导地位。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/modula">modula - 概览</a>：GitHub 是 modula 构建软件的地方。</li><li><a href="https://github.com/modularml/mojo/pull/2825">[stdlib] 为 `List` 添加可选的小缓冲区优化，第 2 次尝试，由 gabrieldemarmiesse 提交 · Pull Request #2825 · modularml/mojo</a>：此 PR 解决了 #2467 的部分问题。此 PR 是按顺序阅读和合并的三个 PR 之一：[stdlib] 为 List 添加可选的小缓冲区优化，第 2 次尝试 #2825...</li></ul></div>

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1270190953902374913)** (18 条消息🔥):

> - `OpenAI 领导层变动`
> - `开源模型训练的挑战`
> - `Meta 的 JASCO 状态`
> - `Nullbulge 争议`
> - `School BUD-E 语音助手`

- **OpenAI 领导层大换血，John Schulman 转投 Anthropic**：OpenAI 联合创始人 John Schulman 宣布离开这家由 Microsoft 支持的公司，加入 Anthropic。就在三个月前，OpenAI 的超级对齐（superalignment）团队刚刚解散。
  - 这一转变发生在 OpenAI 内部战略调整之际，Schulman 此前共同领导负责优化 ChatGPT 聊天机器人的 post-training 团队。
- **开源项目在昂贵的训练成本面前挣扎**：社区注意到，由于训练最先进模型的成本极高，无法在家庭环境中进行，开源 AI 项目进展滞后。
  - 有推测认为，如果模型训练成本降低，即便不考虑数据来源的伦理问题，开源模型也会大量涌现。
- **Meta 的 JASCO 因法律担忧保持沉默**：关于 Meta 的 JASCO 项目缺失的讨论不断，有人怀疑与 Udio 和 Suno 正在进行的诉讼可能推迟了计划。
  - 社区的担忧凸显了法律风险正影响着专有 AI 技术进步的速度。
- **Nullbulge 人肉搜索（Doxxing）丑闻**：出现了关于被称为 Nullbulge 的争议人物的评论，此人显然已被开盒（doxed）。
  - 用户警告他人不要在 Google 上搜索 Nullbulge，因为其内容可能具有揭露性和伤害性。
- **School BUD-E 简介：一款新型浏览器语音助手**：一段介绍 School BUD-E 语音助手的 YouTube 视频被分享，这是一款创新的浏览器工具。
  - 该解决方案旨在通过其语音用户界面改变教育互动。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.cnbc.com/2024/08/06/openai-co-founder-john-schulman-says-he-will-leave-and-join-rival-anthropic.html">OpenAI 联合创始人 John Schulman 表示他将离开并加入竞争对手 Anthropic</a>：Schulman 表示 OpenAI 高管仍致力于支持确保人类能够控制高能力人工智能模型的努力。</li><li><a href="https://youtu.be/DdAwEdlVi14">School BUD-E 浏览器语音助手</a>：未找到描述</li><li><a href="https://archive.ph/TmDrg">三位领导者离开 OpenAI — The Information</a>：未找到描述</li></ul></div>

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1270444227205009478)** (8 messages🔥):

> - `Val Acc Update`（验证准确率更新）
> - `Scaling Experiments`（扩展实验）
> - `Accuracy Wall discussion`（准确率瓶颈讨论）
> - `Frequency-Phase Inquiry`（频率-相位查询）

- **验证准确率（Val Acc）跳升至 84%**：分享了一个更新，模型达到了 **84% 的验证准确率**。
  - 随后分享了一个表达“开始相信了”的暗示，让人联想到《黑客帝国》中的经典场景。
- **扩展实验受阻**：将模型扩展到 **270k 参数** 的尝试未能提升性能，其达到的准确率阈值与较小模型相似。
- **CIFAR 图像频率查询**：一名成员提出了关于 **CIFAR 图像** 在傅里叶变换（Fourier Transform）术语中如何呈现的问题。
  - 问题集中在 **频率信息** 是否保持一致，而 *相位（phase）有所不同*。

 

**提到的链接**：[The Matrix Laurence Fishburne GIF - The matrix Laurence fishburne Morpheus - Discover & Share GIFs](https://tenor.com/view/the-matrix-laurence-fishburne-morpheus-trinity-he%27s-beginning-to-believe-gif-18413151103009905935)：点击查看 GIF

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270096760634736660)** (8 messages🔥):

> - `Tinygrad compatibility with Aurora`（Tinygrad 与 Aurora 的兼容性）
> - `Intel GPU support`（Intel GPU 支持）
> - `Aurora's ExaFLOP capabilities`（Aurora 的 ExaFLOP 能力）
> - `FP8 Nvidia bounty requirements`（FP8 Nvidia 悬赏要求）

- **在 Aurora 上运行 Tinygrad 的可行性**：一名成员询问了在阿贡国家实验室的 **Aurora 超级计算机**上运行 **tinygrad** 的可行性，因为该计算机使用的是 Intel GPU。
- **Intel Max Data Center GPU 见解**：关于 Aurora GPU 的讨论透露，它们支持类似于 A770s 的 Tensor Core 指令，但输出的是 **16x8 矩阵** 而非 8x8。
- **Aurora 性能预测**：Aurora 预计将超过 **2 ExaFLOPS**，在性能优化后可能成为最快的超级计算机。
- **FP8 Nvidia 悬赏的精度要求**：一名成员询问 **FP8 Nvidia 悬赏** 是否需要同时支持 E4M3 和 E5M2，还是其中之一。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html">cl_intel_subgroup_matrix_multiply_accumulate</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Aurora_(supercomputer)">Aurora (supercomputer) - Wikipedia</a>：未找到描述</li></ul></div>

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1270095328476921857)** (16 messages🔥):

> - `Bug in Tensor slicing`（Tensor 切片中的 Bug）
> - `Buffer to DEFINE_GLOBAL mapping`（Buffer 到 DEFINE_GLOBAL 的映射）
> - `JIT and inconsistent batch sizes`（JIT 与不一致的 Batch Size）
> - `Computer algebra study notes`（计算机代数学习笔记）
> - `Multi-threading in CLANG and LLVM`（CLANG 和 LLVM 中的多线程）

- **修复 Tensor 切片中的 Bug**：一名成员在对 Tensor 切片赋值时遇到了 `AssertionError`，随后指出该错误的修复已包含在测试中。
  - George Hotz 确认应解决此问题以确保切片是连续的（contiguous）。
- **在 Tinygrad 中将 Buffer 映射到 DEFINE_GLOBAL**：一名用户询问在 Tinygrad 中执行加法等操作时，Buffer 是如何映射到 `DEFINE_GLOBAL` 变量的。
  - 对话强调了系统中从 Buffer 到 MemBuffer 转换过程缺乏清晰度。
- **不一致 Batch Size 导致的 JIT 错误**：成员们讨论了由于数据集无法整除导致 Batch Size 不一致而引发的 JIT 错误问题。
  - George Hotz 建议对除最后一个 Batch 之外的所有 Batch 运行 JIT，或者跳过最后一个 Batch 来解决此问题。
- **提供计算机代数学习笔记**：一名用户分享了计算机代数的研究笔记，作为理解 Tinygrad 的 shapetracker 和符号数学（symbolic math）的补充。
  - 这些笔记可在 [GitHub](https://github.com/mesozoic-egg/computer-algebra-study-notes) 上获取。
- **CLANG 和 LLVM 中的单线程问题**：有人询问 CLANG 和 LLVM 的线程能力，得到的澄清是它们使用单线程。
  - 有人指出引入 OpenMP 可能解决此问题，并引用了 Tinygrad 仓库中相关的 [Pull Requests](https://github.com/tinygrad/tinygrad/pull/1201)。

 

**提到的链接**：[computer-algebra-study-notes/README.md at main · mesozoic-egg/computer-algebra-study-notes](https://github.com/mesozoic-egg/computer-algebra-study-notes/blob/main/README.md)：通过在 GitHub 上创建账号来为 mesozoic-egg/computer-algebra-study-notes 的开发做出贡献。

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1270433513476456580)** (6 messages):

> - `Wiseflow 工具`
> - `Golden Ret 与 Wiseflow 集成`
> - `HybridAGI 项目发布`

- **Wiseflow 高效挖掘数据**：**Wiseflow** 是一款敏捷的信息挖掘工具，可以从网站、社交平台等各种来源提取简洁的信息，并自动进行分类。详细信息请参阅 [GitHub](https://github.com/TeamWiseFlow/wiseflow)。
- **Golden Ret 与 Wiseflow 的创意合并**：有人建议将 **Golden Ret** 与 **Wiseflow** 结合，以创建一个**动态知识库**。
- **HybridAGI 发布新版本**：**HybridAGI** 系统是一个专注于神经符号 Cypher 的项目，发布了新版本，重点增强了可用性和数据处理流水线。它附带了各种 Notebook，如 Vector-only RAG 和 Knowledge Graph RAG，可在 [GitHub](https://github.com/SynaLinks/HybridAGI) 上获取。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://github.com/TeamWiseFlow/wiseflow">GitHub - TeamWiseFlow/wiseflow: Wiseflow 是一款敏捷的信息挖掘工具，可从网站、微信公众号、社交平台等各种来源提取简洁信息。它会自动分类并上传到数据库。</a>：Wiseflow 是一款敏捷的信息挖掘工具，可从网站、微信公众号、社交平台等各种来源提取简洁信息。它会自动分类并...</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 可编程的基于 Cypher 的神经符号 AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 按预期运行的人群</a>：可编程的基于 Cypher 的神经符号 AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 按预期运行的人群 - SynaLinks/HybridAGI</li></ul></div>

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1270284030604218450)** (2 messages):

> - `软件工程中的 LLM-based agents`
> - `语言模型中的推理计算扩展`

- **LLM-based agents 旨在挖掘 AGI 潜力**：该论文讨论了 **LLM-based agents** 的潜力，这些 Agent 结合了 LLM 进行决策和执行操作，旨在克服常规 LLM 缺乏自主性和自我改进等局限性 [查看 PDF](https://arxiv.org/abs/2408.02479)。
  - 尽管前景广阔，但该领域在**软件工程**中缺乏统一的标准来界定一个解决方案是否属于 LLM-based agent，这凸显了区分 **LLM** 与 LLM-based agents 的必要性。
- **推理计算提升性能**：根据[该研究](https://arxiv.org/abs/2407.21787)，在具有可验证答案的领域，通过增加样本生成来扩展推理计算可以显著提高语言模型的性能。
  - 在 **SWE-bench Lite** 领域，**DeepSeek-V2-Coder-Instruct** 的性能从 15.9% 提升到了 56%（通过 250 个样本解决问题），而单次尝试的 SOTA 性能为 **43%**。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://arxiv.org/abs/2407.21787">Large Language Monkeys: Scaling Inference Compute with Repeated Sampling</a>：扩展用于训练语言模型的计算量显著提高了它们的能力。然而，在推理时，我们通常将计算量限制在仅一次尝试...</li><li><a href="https://arxiv.org/abs/2408.02479">From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future</a>：随着大语言模型 (LLM) 的兴起，研究人员正越来越多地探索它们在各种垂直领域（如软件工程）的应用。LLM 已经取得了显著的成功...</li></ul></div>

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1270224404428623963)** (7 messages):

> - `MIPRO 性能`
> - `MIPROv2 功能`

- **MIPRO 的性能比较**：关于 **MIPRO** 是否总是优于 **BootstrapFewShotWithRandomSearch** 的讨论得出结论：**MIPRO** 通常表现更好，但并非在所有情况下都如此。
- **MIPROv2 缺乏 assertion 支持**：针对 **MIPROv2** 是否支持 assertion 的询问得到了回复，表明它**目前尚不支持 assertion**。

 

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages):

gamris: 你会推荐改用 Qdrant 的 FastEmbed 吗？ [https://github.com/qdrant/fastembed](https://github.com/qdrant/fastembed)

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1270177649754374165)** (7 条消息):

> - `Synthetic Data Strategy`
> - `SQL Examples in Llama Index`
> - `MD5 Hash Consistency`
> - `Bits and Bytes Pull Request`

- **针对推理任务的 Synthetic Data Strategy**：一位社区成员询问如何为 8b 模型开发 **synthetic data generation strategy**，以改进 text-to-SQL 等推理任务，并建议在合成指令中使用 **Chain-of-Thought (CoT)**。
  - 考虑的方案包括在输出最终 SQL 查询之前，通过 CoT 训练来增强性能。
- **Llama Index 提供 SQL 示例**：另一位成员提到 **Llama Index** 包含一些 SQL 示例，这对于需要 SQL 生成的任务非常有用。
  - 未提供关于这些 SQL 示例的更多细节或链接。
- **LoRA Adapter 合并中的 MD5 Hash 一致性**：一位用户询问多次合并同一个 LoRA adapter 时 **MD5 hash** 的一致性，得到了结果一致的确认。
  - 另一位成员确认，MD5 hash 保持一致是预期结果，如果出现差异则表明存在问题。
- **跟踪 Bits and Bytes 的开发进展**：一位用户指出，关注 [Bits and Bytes Foundation pull request 中的分支](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1220) 对于获取相关更新非常重要。
  - 该 pull request 对于那些关注该库演进的人来说似乎包含重要的进展。

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1270107609386582199)** (5 条消息):

> - `Gemma 2 27b QLoRA`
> - `L40S GPUs performance`
> - `Fast Python package installer`

- **Gemma 2 27b QLoRA 需要调优**：一位用户提到 **Gemma 2 27b** 的 QLoRA 可能需要调整 learning rate，但预计可以与最新的 flash attention 配合使用。
- **L40S GPUs 提供不错的训练性能**：有人对 **L40S GPUs** 的模型训练和推理服务性能感到好奇。一位成员表示在 L40S 上进行训练的表现相当不错。
- **UV：极速 Python 包安装程序**：分享了一个名为 UV 的 **GitHub 仓库**，这是一个用 Rust 编写的**极速 Python 包安装程序**。
  - 一位成员评论道：“更快的 pip 可能对 docker 构建非常有用。”

**提到的链接**：[GitHub - astral-sh/uv: An extremely fast Python package installer and resolver, written in Rust.](https://github.com/astral-sh/uv)：一个用 Rust 编写的极速 Python 包安装程序和解析器。- astral-sh/uv

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1270197885526347957)** (3 条消息):

> - `Context length adjustment in fine-tuned models`
> - `RoPE scaling for context length`

- **调整微调模型的 context length**：一位成员询问是否可以调整像 **llama2-13b-hf** 这样初始 context 为 4k 的微调模型的 context length。
- **RoPE scaling 提供快速解决方案**：在回答关于 context length 调整的查询时，**RoPE scaling** 被强调为一种高效增加 context length 的潜在快速修复方案。

---

### **OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/)** (1 条消息):

caseus_: Office hours 将在一个小时后在 <#1268285745555308649> 开始。

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1270120968878293013)** (1 条消息):

> - `PPO integration`
> - `Qwen2 model support`
> - `RLHF training`
> - `Feature requests for Torchtune`

- **PPO 加入 Torchtune 工具库**：Torchtune 已集成 **PPO** 训练方案，支持在平台内进行 **Reinforcement Learning from Human Feedback (RLHF)**，详见[新的 GitHub pull request](https://github.com/pytorch/torchtune/pull/1005)。
- **现已支持 Qwen2 模型**：Torchtune 的训练套件中增加了对 **Qwen2 模型** 的支持，包括通过 [GitHub](https://github.com/pytorch/torchtune/pull/1143) 提供的 **7B 模型**，**1.5B** 和 **0.5B** 模型也将很快推出。
- **征集社区对 Torchtune 功能的建议**：Torchtune 邀请用户建议希望添加到平台的新模型或方案，鼓励通过 [GitHub](https://github.com/pytorch/torchtune) 提交功能请求。

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1270207545344135304)** (9 条消息🔥):

> - `Llama3-8B 的 DPO 支持`
> - `模型 Prompt 差异`
> - `LLAMA3 Instruct 模型下载`

- **即将推出的 Llama3-8B DPO 支持**：一名成员询问了支持 **Llama3-8B-full-finetune** 进行 DPO 的计划。
  - 另一名成员提供了一个变通方案，即使用 `lora_dpo_single_device` recipe 并针对 **Llama3-8B** 进行特定配置。
- **LLAMA3 模型 Prompt 的变异性**：讨论了在不同环境下对 **LLAMA3 Instruct Model** 进行 Prompt 引导时输出结果不同的问题。
  - 用户争论是否在已正确下载的情况下，仍将 BASE 模型误认为是 INSTRUCT 模型。
- **确保正确的 LLAMA3 文件路径**：成员们强调了为下载的 **Llama3** 文件指定正确的 checkpointer 和 tokenizer 路径的重要性。
  - 确认了使用 Llama3 **Instruct Template** 的 Prompt 格式化由 tokenizer 自动处理。

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1270440954653966360)** (6 条消息):

> - `模型页面重构`
> - `PreferenceDataset 重构`

- **模型页面翻新**：成员们讨论了为每个模型的 builder 专门设立一个页面的想法，以适应日益增多的模型数量，包括未来的 **multimodal LLMs**。
  - 翻新可能包括一个模型索引页，用于解释下载和配置模型等重复性任务。
- **PreferenceDataset 迎来改版**：聊天中分享了 [重构后的 PreferenceDataset](https://github.com/pytorch/torchtune/pull/1276)，支持通过统一的数据流水线添加聊天功能。
  - 提到了一个 Pull Request，并鼓励提供反馈以进一步增强 **PreferenceDataset** 的 transformation 设计。

**提到的链接**：[[4/n] Refactor preference dataset with transforms design by RdoubleA · Pull Request #1276 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/1276)：上下文：继 #1186 中的 RFC 之后，我们将在所有数据集中使用统一的 message_transform -> template -> tokenization 数据流水线。此 PR 更新了 PreferenceDataset 以遵循该流程...

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1270243386523390075)** (9 条消息🔥):

> - `本地 LLM 设置问题`
> - `Open Interpreter 安全措施`
> - `Python 版本兼容性`
> - `视觉模型推荐`

- **本地 LLM 设置困扰：下载了不必要的模型副本**：一名尝试使用本地 LLM 设置解释器的用户遇到了问题：在选择其 *llamafile* 后，输入如 'Hello.' 会触发相同模型的不必要下载，最终导致 *openai.APIConnectionError*。
  - 尽管取得了一些潜在进展，但该问题仍未解决，用户请求通过私信协调以进行协作排查。
- **Open Interpreter 的隐私与安全咨询**：一名成员对 Open Interpreter 的安全措施表示关注，询问有关数据隐私的文档，包括本地机器上的数据保留以及第三方的参与情况。
  - 该成员特别想了解系统间的通信是否受端到端加密保护，以及所使用的加密标准。
- **Open Interpreter 的 Python 版本兼容性问题**：有人询问 Open Interpreter 是否支持 **Python 3.12**，特别是通过 Microsoft App Store 安装的情况。
  - 推荐使用 **Python 3.10** 或 **3.11** 作为兼容版本，表明目前不支持 Python 3.12。

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1270185242996637756)** (2 条消息):

> - `Ollama 本地模型设置`
> - `Deepgram 支持咨询`

- **在 Ollama 上设置你的模型**：一名用户解释了通过 `ollama list` 检查模型名称的过程，并强调了显卡上每个模型需要足够的 **VRAM**。
  - 他们建议遵循 [GitHub 上的特定说明](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx) 进行本地运行，并强调了付费模型使用 **API key** 的重要性。
- **关于 Deepgram 支持的咨询**：一名用户简单询问了该频道是否支持 Deepgram，但未讨论更多细节。

**提到的链接**：[open-interpreter/docs/language-models/local-models/ollama.mdx at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/language-models/local-models/ollama.mdx)：计算机的自然语言界面。欢迎在 GitHub 上为 OpenInterpreter/open-interpreter 的开发做出贡献。

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1270115495273697293)** (2 messages):

> - `Llamafile 更新`
> - `社区调查赢取礼品卡`
> - `sqlite-vec 发布派对`
> - `Machine Learning 论文研讨`
> - `Local AI AMA`

- **Llamafile 迎来重大更新**：**Llamafile** 的核心维护者宣布，在提供单文件离线、易用的 LLM 方面取得了重大进展，进一步提升了用户的易用性。
- **Mozilla AI 调查提供礼品卡**：呼吁社区通过[调查问卷](https://form.typeform.com/to/Cn4md4Oc)分享反馈，并提供 **$25 礼品卡** 作为激励。
- **庆祝 sqlite-vec 发布**：[sqlite-vec 的发布派对](https://discord.com/events/1089876418936180786/1265715263999836210)正在进行中，邀请参与者与核心维护者一起探索功能和演示。
- **Machine Learning 讨论成为焦点**：参与涵盖“[Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795)”和“[Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817)”的 **Machine Learning Paper Talks**，深入探讨新的分析视角。
- **Local AI 核心维护者 AMA**：**Local AI** 的核心维护者举办了一场 [AMA](https://discord.com/events/1089876418936180786/1268967945216721079)，推广这一 OpenAI 的开源、自托管替代方案。

 

**提到的链接**：[发现 Typeform，让表单变得有趣](https://form.typeform.com/to/Cn4md4Oc%3E))：在几分钟内无需代码即可创建精美、互动的表单。免费开始使用。

 

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1270412626798841970)** (1 messages):

> - `LinkedIn 工程团队的 ML 平台转型`
> - `Flyte 流水线`

- **LinkedIn 工程团队翻新 ML 平台**：宣布了一场关于 [LinkedIn 工程团队如何转型其 ML 平台](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)的直播会议。
  - 活动重点是 Flyte 流水线及其在 LinkedIn 的实现。
- **Flyte 流水线在 LinkedIn 的实践**：直播活动涵盖了 [Flyte 流水线](https://www.linkedin.com/events/flytepipelinesinactionwithlinke7218669945767776256/theater/)，展示了它们在 LinkedIn 基础设施中的实际应用。
  - 参与者预计将深入了解 LinkedIn 所采用的工程策略和解决方案。

 

---

---

---

---

{% else %}

> 完整的频道详细解析已为邮件格式进行截断。
> 
> 如果您想查看完整解析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}