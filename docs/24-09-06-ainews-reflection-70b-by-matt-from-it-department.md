---
companies:
- hyperwrite
- glaive
date: '2024-09-07T01:17:07.379983Z'
description: '来自 **Hyperwrite** 和 **Glaive** 的一个两人团队利用 **Reflection Tuning**（反思微调）技术对
  **llama-3.1-70b** 进行了微调，在仅使用极少量合成数据的情况下，实现了显著的性能提升。


  该方法借鉴了在输出中加入“思考”（thinking）和“反思”（reflection）步骤的概念，与**思维链**（Chain of Thought）方法相关。尽管面临一些批评，如对数据污染的担忧、编程性能下降以及对系统提示词的依赖，该模型仍获得了积极的反响，并被拿来与
  **claude-3.5-sonnet** 进行比较。这项工作突显了大模型在高效指令微调和合成数据生成方面的潜力。'
id: 02416d00-dc64-487c-ade4-95eb958e04bb
models:
- llama-3.1-70b
- llama-3
- claude-3.5-sonnet
original_slug: ainews-reflection-70b-by-matt-from-it-department
people:
- matt-shumer
- sahil-chaudhary
title: Reflection 70B，由 IT 部门的 Matt 创作。
topics:
- fine-tuning
- chain-of-thought
- instruction-following
- synthetic-data
- quantization
- model-evaluation
- prompt-engineering
---

 

看来 Matt 找到了理想的“低垂果实”，因为直到现在还没有人费心对 Orca 进行不同的尝试，并生成足够的 synthetic data（我们仍然不知道具体有多少，但考虑到 Matt 和 Sahil 只花了大约几十个人工日，数据量应该不会太大）来完成这件事。

批评声音很少，且大多不是致命伤：

- **数据污染担忧**：[99.2% 的 GSM8K 分数太高了——超过 1% 的题目存在标签错误，这表明存在污染](https://x.com/hughbzhang/status/1831777846899175576)
  - [Johno Whitaker](https://x.com/johnowhitaker/status/1831800187012202672) 独立验证了 GSM8K 中 5 个已知的错误问题被正确回答（即不是死记硬背）
  - Matt 也在其上运行了 [LMsys 去污染检查](https://x.com/mattshumer_/status/1831767026681180539)
- **编程表现更差**：[在 BigCodeBench-Hard 上表现更差](https://x.com/terryyuezhuo/status/1832112913391526052) —— 比 L3-70B 低了近 10 分；以及 [Aider 代码编辑](https://x.com/paulgauthier/status/1832160129720185225) —— 比 L3-70B 差 7%。
- **针对解决琐碎知识过度优化**：“在理解力方面几乎（但还不完全）与 Llama 70B 持平，但在摘要方面远落后——无论是摘要内容还是语言。它的几个句子完全不通顺。我最后把它删了。” —— [/r/locallLama](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/comment/lluttbm/)
- **异常依赖系统提示词**：“有趣的是，如果你不使用作者建议的特定 system prompt，该模型的表现与基础版 Llama 3.1 相同。他自己甚至也这么说。” —— [/r/localllama](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/comment/llukruz/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
- [骗子](https://x.com/agihippo/status/1831944081066618907)/[炒作](https://www.reddit.com/r/LocalLLaMA/comments/1fanrr4/reflection_70b_hype/)警报 —— Matt [没有披露他是 Glaive 的投资者](https://x.com/gazorp5/status/1831844715379167420)。

经过一天的评测，整体反响依然非常强烈 —— /r/localLlama 报道称 [即使是 Reflection 70B 的 4bit 量化版本表现也很好](https://www.reddit.com/r/LocalLLaMA/comments/1fan3aa/even_4bit_quants_of_reflection_70b_are_amazing/)，Twitter 上也流传着关于 [谜题](https://x.com/AIandDesign/status/1832221300791943561) 的测试以及与 [Claude 3.5 Sonnet](https://x.com/gauravpathak/status/1831808959935868941) 的有利对比。可以说，即使它还不算是一个全能模型，但至少通过了 vibe check，并且在足够多的推理任务中表现显著。

更多信息可以在与 Matthew Berman 的 [这段 34 分钟的直播对话](https://x.com/MatthewBerman/status/1832096560970395704) 和 [12 分钟的回顾视频](https://x.com/MatthewBerman/status/1832098688581431713) 中找到。

总而言之，对于 [Matt from IT](https://x.com/andrew_n_carr/status/1832103565529379270) 来说，这是不错的一天。

 
![image.png](https://assets.buttondown.email/images/70a2f470-9a61-446a-b21f-a62c335d766d.png?w=960&fit=max)
 





---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 生成，取 4 次运行中的最佳结果。

**LLM 训练与评估**

- **LLM 训练与评估**：[@AIatMeta](https://twitter.com/AIatMeta/status/1831123520425963684) 的 **LLM Evaluations Grant**（LLM 评估资助金）申请截止日期为 **9 月 6 日**。该资助将提供 **20 万美元资金**，用于支持 **LLM 评估研究**。
- **多模态模型**：[@glennko](https://twitter.com/glennko/status/1831119997306819048) 认为 **AI** 最终将能够以**高准确度**数出 "r" 的个数，但这可能不是通过 **LLM** 实现的，而是通过**多模态模型 (multi-modal model)**。
- **专用架构**：[@glennko](https://twitter.com/glennko/status/1831120394796896758) 指出，**FPGA** 速度太慢，而 **ASIC** 成本太高，难以构建**自定义逻辑**所需的专用架构。

**开源模型与研究**

- **开源 MoE 模型**：[@apsdehal](https://twitter.com/apsdehal/status/1831168945514045633) 宣布发布 **OLMoE**，这是一个 **1B 参数**的 **Mixture-of-Experts (MoE)** **语言模型**，且 **100% 开源**。该模型由 **ContextualAI** 和 **Allen Institute for AI** 合作完成。
- **开源 MoE 模型**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1831182890056798530) 指出，**OLMOE-1B-7B** 拥有 **70 亿参数**，但每个输入 Token 仅使用 **10 亿参数**，且在 **5 万亿 Token** 上进行了预训练。该模型在**激活参数量相似**的模型中表现优于其他可用模型，甚至超过了更大的模型，如 **Llama2-13B-Chat** 和 **DeepSeekMoE-16B**。
- **开源 MoE 模型**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1831233827353608584) 指出，**DeepSeek-MoE** 在**粒度 (granularity)** 方面得分很高，但在**共享专家 (shared experts)** 方面表现一般。

**AI 工具与应用**

- **AI 驱动的电子表格**：[@annarmonaco](https://twitter.com/annarmonaco/status/1831355874872529284) 强调了 **Paradigm** 如何利用 AI 改变电子表格，并使用 **LangChain** 和 **LangSmith** 来监控关键成本并获得分步的 Agent 可视化。
- **用于医疗诊断的 AI**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1831240904293728327) 分享了一份指南，介绍如何使用结合了**文本**和**图像数据**的**混合搜索 (hybrid search)** 来创建高性能诊断系统，并从文本和图像数据中生成 **multimodal embeddings**。
- **用于时尚的 AI**：[@flairAI_](https://twitter.com/flairAI_/status/1831349517310210199) 正在发布一个**时尚模型**，该模型可以以极高的准确度在服装上进行训练，以 **Midjourney 级别的质量**保留纹理、标签、Logo 等。

**AI 对齐与安全**

- **AI 对齐与安全**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1831286356728848511) 分享了一期播客，讨论了 **AI 对齐 (AI alignment)** 的挑战以及有效监督强大系统的能力。播客包含了来自 **Anca Diana Dragan** 和 **Professor FryRSquared** 的见解。
- **AI 对齐与安全**：[@ssi](https://twitter.com/ssi/status/1831327645054947498) 正在构建“通往安全超级智能的直达路径”，并已从投资者那里筹集了 **10 亿美元**。
- **AI 对齐与安全**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1831129636220350562) 指出，**EA** 通过使用朴素后果主义选择策略，而没有妥善考虑二阶效应，从而助长了追求权力的行为。

**模因与幽默**

- **创始人模式 (Founder Mode)**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1831188248883843182) 拿 **Elon Musk 的 Twitter 动态**开玩笑，将他比作**钢铁侠**。
- **创始人模式 (Founder Mode)**：[@nisten](https://twitter.com/nisten/status/1831127609901457706) 建议 **Marc Andreessen** 需要一个更好的过滤 **LLM** 来管理他随机屏蔽的用户。
- **创始人模式 (Founder Mode)**：[@cto_junior](https://twitter.com/cto_junior/status/1831303273074016384) 调侃亚洲兄弟如何在现有模型之上堆叠 Encoder 和 Cross-attention，仅仅是为了找点感觉。


---

# AI Reddit 综述

## /r/LocalLlama 摘要

**主题 1：LLM 量化与效率的进展**

- **[llama.cpp 合并了对 TriLMs 和 BitNet b1.58 的支持](https://github.com/ggerganov/llama.cpp/pull/8151)** ([Score: 73, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fa3ryv/llamacpp_merges_support_for_trilms_and_bitnet_b158/)): **llama.cpp** 通过集成对 **TriLMs** 和 **BitNet b1.58** 模型的支持扩展了其功能。此更新允许在 **TriLMs** 中对权重使用**三进制量化 (ternary quantization)**，并为 **BitNet** 模型引入了**二进制量化 (binary quantization)** 方法，这可能为模型部署和执行提供更高的效率。


**主题 2：Reflection-70B：一种新型的 LLM 微调技术**

- **[Reflection 70B 的首个独立基准测试 (ProLLM StackUnseen) 显示出非常好的提升。比基础 Llama 70B 模型提高了 9 个百分点 (41.2% -> 50%)](https://i.redd.it/tuawfbwms3nd1.png)** ([Score: 275, Comments: 115](https://reddit.com//r/LocalLLaMA/comments/1fa4y7q/first_independent_benchmark_prollm_stackunseen_of/)): **Reflection-70B** 在 **ProLLM StackUnseen 基准测试**中展示了相对于其基础模型的显著性能提升，准确率从 **41.2%** 提高到 **50%**，增长了 **9 个百分点**。这项独立评估表明 Reflection-70B 的能力可能超越了更大规模的模型，突显了其在处理未见过的编程任务方面的有效性。
  - **Matt from IT** 意外地与 **OpenAI**、**Google** 和 **Meta** 等顶尖 AI 公司并列，引发了关于个人创新以及来自大型科技公司潜在工作邀约的讨论。
  - **Reflection-70B** 模型展示了优于更大规模模型的显著改进，在基准测试中击败了 **405B** 版本。用户对未来更大模型的微调表示期待，并讨论了在本地运行这些模型的硬件要求。
  - 关于将 **Reflection-70B** 与其他模型进行比较的公平性产生了争论，因为它使用了独特的 `<thinking>` 和 `<output>` 标签输出格式。一些人认为这类似于**思维链 (Chain of Thought)** 提示词，而另一些人则认为这是一种增强模型推理能力的新颖方法。

- **[Reflection-Llama-3.1-70B 已在 Ollama 上线](https://ollama.com/library/reflection)** ([Score: 74, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1fa72an/reflectionllama3170b_available_on_ollama/)): **Reflection-Llama-3.1-70B** 模型现在可以在 **Ollama** 上访问，扩展了该平台上可用的大语言模型范围。该模型基于 **Llama 2**，并使用 **Constitutional AI** 技术进行了微调，以增强其在**任务分解**、**推理**和**反思 (reflection)** 等领域的能力。
  - 用户注意到模型最初存在**系统提示词错误**，该错误已迅速得到**更新**。该模型在 Ollama 上的名称误删了 "llama"，引发了一些趣谈。
  - 据报道存在 **tokenizer 问题**，可能影响模型在 **Ollama** 和 **llama.cpp** 上的表现。[Hugging Face](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B/discussions/6) 上的一个活跃讨论正在解决此问题。
  - 该模型在解决蜡烛问题时展示了其**反思能力**，捕捉并纠正了最初的错误。用户表示有兴趣将此技术应用于更小的模型，尽管有人指出 **8B 版本** 的改进有限。

## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型开发与发布**

- **Reflection 70B**：由 Matt Shumer 创建，是 Meta Llama 3.1 70B 模型的微调版本，[**声称在基准测试中超越了最先进的模型**](https://www.reddit.com/r/singularity/comments/1f9uszk/reflection_70b_the_worlds_top_opensource_model/)。它使用合成数据来提供一种“内心独白”，类似于 Anthropic 在 Claude 3 到 3.5 中采用的方法。

- **AlphaProteo**：[**Google DeepMind 的新 AI 模型可生成新型蛋白质**](https://www.reddit.com/r/singularity/comments/1f9orj0/google_deepminds_alphaproteo_generates_novel/)，用于生物学和健康研究。

- **OpenAI 的未来模型**：据报道，OpenAI 正在[**考虑为下一代 AI 模型（可能命名为 Strawberry 和 Orion）设定高达每月 2,000 美元的高价订阅费**](https://www.reddit.com/r/OpenAI/comments/1f9ovbm/openai_is_reportedly_considering_highpriced/)。

**AI 行业与市场动态**

- **开源影响力**：Reflection 70B 的发布引发了关于[**开源模型颠覆 AI 行业潜力**](https://www.reddit.com/r/OpenAI/comments/1f9ybqy/new_opensource_ai_model_is_smashing_the/)的讨论，这可能会激励像 OpenAI 这样的公司发布新模型。

- **模型能力**：[**公众认知与实际 AI 模型能力之间存在脱节**](https://www.reddit.com/r/singularity/comments/1f9ukpg/people_really_have_0_idea_whats_going_on_the/)，许多人并不了解 AI 技术的现状。

**AI 应用与创新**

- **DIY 药物**：一份报告讨论了[**“盗版 DIY 药物”的兴起**](https://www.reddit.com/r/singularity/comments/1fa0rl1/the_rise_of_pirate_diy_medicine_an_amateur_can/)，业余人士可以以极低的成本制造昂贵的药物。

- **Stable Diffusion**：一个新的 [**用于 Stable Diffusion 的 FLUX LoRA 模型**](https://www.reddit.com/r/StableDiffusion/comments/1fa5ebi/less_than_24_hours_100_downloads_thrilled_that_my/) 受到欢迎，展示了 AI 生成艺术领域的持续发展。

---

# AI Discord 简报

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要

**1. LLM 进展与基准测试**

- **Reflection 70B 掀起波澜**：**[Reflection 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B)** 被宣布为全球顶尖的开源模型，该模型采用了一种全新的 **Reflection-Tuning** 技术，使其能够检测并纠正自身的推理错误。
   - 尽管最初的热度很高，但随后在 **BigCodeBench-Hard** 等基准测试中的表现却褒贬不一，得分低于之前的模型。这引发了关于评估方法以及合成训练数据影响的辩论。
- **DeepSeek V2.5 步入赛场**：**[DeepSeek V2.5](https://x.com/deepseek_ai/status/1832026579180163260)** 正式发布，它结合了 DeepSeek-V2-0628 和 DeepSeek-Coder-V2-0724 的优势，增强了写作、指令遵循（instruction-following）以及人类偏好对齐（human preference alignment）能力。
   - 社区对比较 DeepSeek V2.5 与 Reflection 70B 等近期模型在编程任务中的表现表现出浓厚兴趣，凸显了该领域飞速发展的节奏。
  


**2. 模型优化技术**

- **Speculative Decoding 取得突破**：**[Together AI](https://x.com/togethercompute/status/1831755763615674412)** 宣布在 Speculative Decoding 方面取得突破，在长上下文输入下实现了高达 **2 倍**的延迟和吞吐量提升，挑战了此前对其有效性的假设。
   - 这一进展标志着高吞吐量推理优化方面的重大转变，有望减少 GPU 工时以及 AI 方案部署的相关成本。
- **AdEMAMix 优化器增强梯度处理**：提出了一种名为 **AdEMAMix** 的新型优化器，如这篇 [论文](https://arxiv.org/pdf/2409.03137) 所述，它利用两个指数移动平均（EMAs）的混合，比单一 EMA 更好地处理过去的梯度。
   - 早期实验显示，AdEMAMix 在语言建模和图像分类任务中优于传统的单一 EMA 方法，有望为各种 AI 应用带来更高效的训练结果。
  


**3. 开源 AI 发展**

- **llama-deploy 简化微服务**：**[llama-deploy](https://twitter.com/llama_index/status/1831794126511337880)** 发布，旨在促进基于 **LlamaIndex Workflows** 的微服务无缝部署，标志着 Agent 系统部署的一次重要演进。
   - 官方分享了一个[开源示例](https://twitter.com/llama_index/status/1832132462786576652)，展示了如何结合 llama-deploy 与 @getreflex 前端框架构建一个 Agent 聊天机器人系统，证明了其全栈能力。
- **SmileyLlama：AI 分子设计工具**：**[SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148)** 亮相，这是一个经过微调的化学语言模型，能够根据 Prompt 中指定的属性设计分子，基于 **Axolotl** 框架构建。
   - 这一进展展示了 Axolotl 在将现有化学语言模型技术适配到分子设计等专门任务方面的能力，拓展了 AI 在化学领域应用的边界。
  


**4. AI 基础设施与部署**

- **NVIDIA 发布 AI 教学套件**：NVIDIA 的 **Deep Learning Institute** 发布了与达特茅斯学院合作开发的 [生成式 AI 教学套件](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a)，旨在让学生掌握 GPU 加速的 AI 应用。
   - 该套件旨在通过弥合各行业的知识鸿沟，为学生在就业市场提供显著优势，彰显了 NVIDIA 对 AI 教育和人才培养的承诺。
- **OpenAI 考虑高端定价**：有报道称 **OpenAI** 正在考虑为其更先进的 AI 模型（包括备受期待的 Orion 模型）推出每月 **2000 美元**的订阅模式，正如这份 [The Information 报告](https://x.com/aiexplainedyt/status/1831710902636228694?s=46) 所讨论的。
   - 这种潜在的定价策略在社区内引发了关于可访问性和 AI 民主化影响的辩论，一些人担心这会为小型开发者和研究人员制造障碍。
  

---

# PART 1: 高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **视觉语言模型（Vision Language Models）概述**：一位成员分享了一篇[博客文章](https://www.lightly.ai/post/introduction-to-vision-language-models)，详细介绍了 AI 应用中视觉与语言的集成，并强调了其创新潜力。
   - 这篇文章旨在将关注点引导至这一技术交叉领域中涌现的多样化用例。
- **Tau LLM 训练优化资源**：[Tau LLM 系列](https://youtube.com/live/flwqvE4aSzA?feature=share) 提供了关于优化 LLM 训练过程的关键见解，有望提升性能。
   - 对于任何深入研究如何有效训练 LLM 复杂性的人来说，这被认为是至关重要的资源。
- **寻求用于疾病检测的医疗数据集**：一位成员正在寻找用于 **Computer Vision** 的强大医疗数据集，旨在通过 Transformer 模型增强 **疾病检测（Disease Detection）**。
   - 他们对能够支持该领域大规模数据生成工作的数据集特别感兴趣。
- **Flux img2img 流水线仍待定**：**Flux img2img** 功能尚未合并，正如[公开 PR](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) 中所述，关于其文档的讨论仍在进行中。
   - 尽管它可能对典型的消费级硬件造成压力，但正如[相关讨论](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3)中所分享的，优化措施正在探索中。
- **用于增强语言模型的选择性微调**：选择性 [Fine-Tuning](https://huggingface.co/blog/anakin87/spectrum) 的概念受到关注，展示了其在无需全量重新训练的情况下提高语言模型性能的能力。
   - 这种有针对性的方法允许进行更深层次的性能调整，同时避免了与完整训练周期相关的成本。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ControlNet 增强模型搭配**：用户分享了结合使用 **ControlNet** 与 **Loras** 的成功策略，利用各种 **SDXL** 模型生成如 **Hash Rosin** 图像的精确表现。
   - 他们建议应用 **Depth Maps**（深度图）等技术以获得更好的结果，突显了在组合不同 AI 工具方面日益成熟的掌握能力。
- **Flux 在 Logo 生成方面领先于 SDXL**：社区广泛认可 **Flux** 在 Logo 生成方面优于 **SDXL**，强调其在无需大量训练的情况下对 Logo 细节的卓越处理。
   - 成员们指出，**SDXL** 在不熟悉 Logo 设计的情况下表现吃力，这使得 **Flux** 因其易用性和有效性成为首选。
- **诈骗防范意识提升**：关于在线诈骗的讨论显示，即使是经验丰富的用户也可能受到攻击，这促使大家共同承诺保持持续警惕。
   - 对诈骗行为的同理心理解成为一项关键见解，强化了“易受攻击性并不局限于缺乏经验者”的观点。
- **ComfyUI 中的标签功能创新**：社区对 **ComfyUI** 中打标签（Tagging）功能的见解将其能力比作 **Langflow** 和 **Flowise**，展示了其灵活性和用户友好的界面。
   - 成员们集思广益，探讨了增强标签效能的具体工作流，预示着界面功能将迎来一波充满希望的适配浪潮。
- **Forge 扩展插件见解**：对 **Forge** 中各种可用扩展的咨询突显了用户通过贡献和社区反馈来改善体验的努力。
   - 投票被用作塑造未来扩展版本的一种方法，强调了质量保证和社区参与的重要性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **祝贺获得 Y Combinator 批准！**: 团队成员庆祝最近获得了 **Y Combinator** 的支持，展现了强大的 **社区支持** 以及对项目未来的热情。
   - 他们认为这一里程碑是推动开发和推广的重大助力。
- **Unsloth AI 面临硬件兼容性障碍**: 讨论强调了 **Unsloth** 目前在硬件兼容性方面的困境，特别是关于 Mac 系统上的 **CUDA 支持**。
   - 团队的目标是实现 **硬件无关性 (hardware agnosticism)**，但持续存在的问题降低了某些配置下的性能。
- **合成数据生成模型见解**: 分享了关于使用 **Mistral 8x7B 微调版本** 进行合成数据生成的见解，以及使用 **jondurbin/airoboros-34b-3.3** 等模型进行测试。
   - 基于硬件限制，实验对于优化微调结果仍然至关重要。
- **Phi 3.5 模型输出困扰用户**: 用户报告称，尽管调整了参数，**Phi 3.5** 模型在微调过程中仍返回 **乱码输出 (gibberish outputs)**，令人沮丧。
   - 这引发了关于故障排除和优化输入模板以提高模型性能的广泛讨论。
- **对对比报告的兴趣激增！**: 一位成员表达了对关键主题对比报告的渴望，强调了其作为 **深度见解** 阅读材料的潜力。
   - 与此同时，另一位成员宣布了制作 [YouTube 视频](https://youtube.com) 详细介绍这些对比的计划，展示了社区的参与度。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **寻求免费图像 API 选项**: 用户调查了支持高限制的 [免费图像 API 选项](https://huggingface.co/lmstudio-community/stable-code-instruct-3b-GGUF)，特别是询问提供 **Stable Diffusion** 等模型访问权限的供应商。
   - 大家对能够大规模提供这些功能的供应商感到好奇。
- **Reflection Llama-3.1 70B 获得增强**: **Reflection Llama-3.1 70B** 作为顶尖的开源 **LLM** 给人留下了深刻印象，其更新增强了错误检测和纠正能力。
   - 然而，用户注意到持续存在的性能问题，并讨论了优化 Prompt 以改善模型行为的方法。
- **LM Studio 更新后出现下载问题**: 更新到 **0.3.2** 版本后，用户在下载模型时面临挑战，[证书错误 (certificate errors)](https://github.com/zed-industries/zed/issues/17482) 是主要问题。
   - 讨论的解决方法包括调整 **VRAM** 和 **Context Size**，同时对 **RAG** 摘要功能进行了说明。
- **Mac Studio 在处理大模型时的速度挑战**: 用户担心拥有 **256GB+** 内存的 **Mac Studio** 在处理大型模型时速度缓慢，希望 **LPDDR5X 10.7Gbps** 能够解决这一问题。
   - 一项讨论强调了所有 **M4** 芯片可能带来 **70%** 的速度提升，引发了对硬件升级的进一步兴趣。
- **利用 NVLink 和 RTX 3090 最大化性能**: 用户分享了在双 **RTX 3090** 配置下实现 **10 到 25 t/s** 的见解，特别是使用 **NVLink** 时，甚至有人报告达到了 **50 t/s**。
   - 尽管有这些高数据，一些社区成员对 **NVLink** 对实际推理性能的影响持怀疑态度。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Reflection 70B 模型在基准测试中表现不佳**：最近的测试显示 **Reflection 70B** 在与 [BigCodeBench-Hard](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) 的对比中表现不佳，特别是受到 tokenizer 和 prompt 问题的影响。
   - 社区对评估结果表示担忧，导致该模型在实际应用中的可靠性存在不确定性。
- **社区调查 DeepSeek v2.5 的可用性**：成员们寻求关于 **DeepSeek v2.5** 在编程任务中改进情况的反馈，鼓励分享用户体验。
   - 该倡议旨在建立对模型有效性的集体认识，并促进用户驱动的增强。
- **关于 Llama 3.1 API 可用性的咨询**：讨论了实现 **Llama 3.1 70B** 的最佳 API 选项，强调了对 tool call 格式支持的需求。
   - 建议包括探索各种平台，指出 **Groq** 是一个很有前景的部署候选方案。
- **量化技术的挑战**：用户报告了 70B 模型在 **FP16** 量化方面的挫折，强调了在使用 **int4** 达到满意性能方面的困难。
   - 正在进行的讨论围绕着在保持质量完整性的同时增强模型性能的潜在解决方案。
- **提升性能的 MCTS 和 PRM 技术**：对话表明了对合并 **MCTS** (Monte Carlo Tree Search) 和 **PRM** (Probabilistic Roadmap) 以提高训练效率的兴趣。
   - 社区对尝试这些方法以改进模型评估过程表现出热情。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 考虑 2000 美元的订阅费用**：OpenAI 正在为其高端 AI 模型（包括即将推出的 Orion 模型）探索 **$2000/月** 的定价模式，这引起了社区内对可访问性的担忧。
   - 随着讨论的展开，关于这种定价是否符合市场规范或是否为小型开发者设置了障碍，意见不一。
- **Reflection 70B 参差不齐的基准测试结果**：**Reflection 70B** 模型表现出参差不齐的性能，在 **BigCodeBench-Hard** 基准测试中得分为 **20.3**，明显低于 **Llama3** 的 **28.4** 分。
   - 批评者强调需要对其方法论进行更深入的分析，特别是关于其声称是顶级开源模型的说法。
- **投机采样 (Speculative Decoding) 提升推理速度**：Together AI 报告称，投机采样可以将吞吐量提高多达 **2x**，挑战了之前关于其在高延迟场景下效率的假设。
   - 这一进展可能会重塑优化长上下文输入推理速度的方法。
- **文本转音乐模型的令人兴奋的发展**：一个新的开源 **text-to-music model** 出现，声称具有令人印象深刻的音质和效率，与 **Suno.ai** 等成熟平台竞争。
   - 成员们对其潜在应用非常感兴趣，尽管对其具体可用性存在不同看法。
- **AI 代码编辑器的探索**：关于 AI 代码编辑器的讨论重点介绍了 [Melty](https://github.com/meltylabs/melty) 和 [Pear AI](https://github.com/trypear/pearai-app) 等工具，展示了与 Cursor 相比的独特功能。
   - 成员们特别感兴趣这些工具如何管理注释和 TODO，推动在编码环境中更好的协作。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 抢尽风头**：用户称赞 **Perplexity** 的速度和可靠性，通常认为它是 **ChatGPT Plus** 订阅的更好替代方案。
   - 一位用户指出，它对学校学习特别有用，因为它易于访问并与 **Arc browser** 集成。
- **RunwayML 面临抵制**：一位用户在社区见面会取消后对 **RunwayML** 表示不满，这引发了对其客户服务的担忧。
   - 评论强调了忠实成员的不满以及这如何影响 Runway 的声誉。
- **Reflection 模型前景广阔的调整**：围绕 **Reflection Llama-3.1 70B model** 的讨论集中在其性能和一种名为 **Reflection-Tuning** 的新训练方法上。
   - 用户注意到初始测试问题导致了一个平台链接的产生，他们可以在那里实验该模型。
- **OpenAI token 赠送引发关注**：一项 **OpenAI tokens** 的赠送活动引起了极大兴趣，因为一位用户有 **1,000 tokens** 且不打算使用。
   - 这引发了关于在社区内进行潜在交易或利用这些 token 的讨论。
- **有效的 tool call 集成**：成员们分享了在 prompt 中构建 **tool calls** 的技巧，强调了 **Assistant message** 后紧跟 **Tool message** 的正确顺序。
   - 一位成员指出，在单个 prompt 输出中成功实现了超过 **10 个 Python tool calls**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **获得学术实验室职位**：成员们讨论了获得学术实验室职位的策略，强调了 **project proposals** 的有效性，而冷邮件（cold emailing）的成功率较低。
   - 一位成员强调需要将研究项目与当前趋势结合，以吸引潜在导师的注意。
- **Universal Transformers 面临可行性问题**：关于 *Universal Transformers* 的可行性展开了辩论，一些成员表示怀疑，而另一些成员则在自适应隐式计算（adaptive implicit compute）技术中发现了潜力。
   - 尽管前景广阔，但稳定性仍然是在实际应用中广泛采用的**重大障碍**。
- **AdEMAMix Optimizer 改进梯度处理**：新提出的 **AdEMAMix** 优化器通过混合两个 **Exponential Moving Averages**，增强了梯度利用率，在语言建模等任务中表现出更好的性能。
   - 早期实验表明，这种方法优于传统的单一 EMA 方法，有望带来更高效的训练结果。
- **自动化强化学习 Agent 架构**：引入了一种新的自动化 RL Agent 架构，通过 **Vision-Language Model** 高效管理实验进度并构建课程（curricula）。
   - 这标志着强化学习实验工作流中首批实现完全自动化的案例之一，在模型训练效率方面取得了新突破。
- **Hugging Face RoPE 兼容性担忧**：一位成员提出了关于 **GPTNeoX** 的 **Hugging Face RoPE implementation** 与其他模型之间兼容性的问题，指出 attention 输出存在超过 **95%** 的差异。
   - 这为那些使用多个框架的开发者提供了重要的参考，并可能影响未来的集成工作。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 庆祝里程碑**：成员们热烈庆祝 Open Interpreter 的生日，社区表达了对其创新潜力的深切赞赏。
   - *Happy Birthday, Open Interpreter!* 成了大家的口号，强调了对其能力的兴奋之情。
- **Open Interpreter 中的 Skills 功能仍处于实验阶段**：讨论显示 Skills 功能目前是实验性的，引发了关于这些技能是否跨会话持久化的疑问。
   - 用户注意到技能似乎是临时的，这导致了调查本地机器存储位置的建议。
- **对 01 app 性能的正面反馈**：用户分享了关于 01 app 能够高效搜索并播放拥有 2,000 个音频文件的库中歌曲的热情反馈。
   - 尽管受到好评，但也有关于结果**不一致性**（inconsistencies）的报告，反映了典型的早期访问挑战。
- **Fulcra app 扩展到新地区**：Fulcra app 已正式在更多地区上线，响应了社区对提高可访问性的请求。
   - 讨论表明用户对 ***Australia*** 等地的可用性感兴趣，并支持进一步扩张。
- **申请 Beta Role 访问权限**：多位用户渴望获得 **desktop 的 beta role** 访问权限，其中包括一位为 Open Interpreter 01 的 dev kit 做出贡献的用户。
   - 一位用户对错过直播会议表示遗憾，并询问：*“有什么办法可以获得 desktop 的 beta role 吗？”*

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Values 页面返回 404**：成员们注意到 Modular 的 values 页面目前在[此链接](https://www.modular.com/values)显示 **404 错误**，可能需要重定向到 [company culture](https://www.modular.com/company/culture)。
   - *建议进行澄清*，指出需要更改链接以有效地将用户引导至相关内容。
- **Mojo 中 Async 函数的限制**：一位用户在处理 `async fn` 和 `async def` 时遇到问题，发现这些 async 功能仅限于 nightly 构建版本，在稳定版中引起了困惑。
   - 建议用户检查其版本，并考虑切换到 nightly 构建版本以使用这些功能。
- **DType 作为 Dict Key 的约束**：关于无法将 `DType` 用作 Dictionary 键的讨论引发了关注，因为它实现了 `KeyElement` trait。
   - 参与者探讨了 Mojo 数据结构中的设计约束，这些约束可能会限制某些类型的使用。
- **构造函数使用故障排除**：分享了解决涉及 `Arc[T, True]` 和 `Weak[T]` 的构造函数问题的进展，突出了 @parameter guards 带来的挑战。
   - 建议包括改进标准库中的命名规范以提高清晰度，并对齐类型的结构。
- **探索 MLIR 和 IR 生成**：对如何在 Mojo 中更有效地利用 MLIR 产生了兴趣，特别是关于 IR 生成方面。
   - 建议参考之前 LLVM 会议的资源 [2023 LLVM Dev Mtg - Mojo 🔥](https://www.youtube.com/watch?v=SEwTjZvy8vw)，以深入了解集成情况。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Reflection 70B 发布，带来令人兴奋的特性**：**Reflection 70B** 模型已发布，被称为全球最强的开源模型，利用 [Reflection-Tuning](https://x.com/mattshumer_/status/1831767014341538166) 来纠正 LLM 错误。
   - 预计下周将推出 **405B 模型**，其性能可能超越目前所有模型。
- **调查 TorchDynamo 缓存查找延迟**：在执行大型模型时，成员注意到 **TorchDynamo Cache Lookup** 耗时 **600us**，主要源于 `torch/nn/modules/container.py` 的调用。
   - 这表明需要对缓存查找过程进行潜在优化，以提高模型训练的运行效率。
- **NVIDIA 联手开展生成式 AI 教育**：NVIDIA 的 **Deep Learning Institute** 与达特茅斯学院合作发布了 [生成式 AI 教学套件](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a)，以增强 GPU 学习。
   - 参与者将在 AI 应用中获得竞争优势，弥补关键的知识鸿沟。
- **FP16 x INT8 Matmul 在 Batch Size 上显示出局限性**：由于共享内存限制，**4090 RTX** 上的 **FP16 x INT8 matmul** 在 Batch Size 超过 1 时会失败，这暗示需要针对非 A100 GPU 进行更好的调优。
   - 用户在启用 inductor 标志时遇到了严重的减速，但可以通过关闭它们来绕过错误。
- **Liger 的性能基准测试引人关注**：**Liger 的 swiglu kernels** 性能与 [Together AI 的基准测试](https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection) 进行了对比，据报道后者提供了高达 **24% 的加速**。
   - 其专门定制的 kernels 性能优于 **cuBLAS** 和 **PyTorch eager mode** 约 **22-24%**，表明需要进一步的调优选项。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Reflection Llama-3.1 70B 表现参差不齐**：新发布的 **Reflection Llama-3.1 70B** 声称是领先的开源模型，但在 **BigCodeBench-Hard** 等基准测试中表现挣扎。
   - 用户观察到其在推理任务中的性能下降，并在 Twitter 上将其描述为“乏善可陈的非新闻级模型”。
- **对 Glaive 合成数据的担忧依然存在**：社区成员对来自 **Glaive** 的合成数据的有效性提出了警示，回顾了过去可能影响模型性能的数据污染问题。
   - 这些担忧引发了关于合成数据对 **Reflection Llama** 模型泛化能力影响的讨论。
- **HuggingFace Numina 在研究领域受到好评**：**HuggingFace Numina** 被强调为以数据为中心任务的强大资源，其应用潜力令研究人员感到兴奋。
   - 用户对它如何提高各种正在进行的项目的效率和创新表达了热情。
- **引入用于数学推理的 CHAMP 基准测试**：社区欢迎新的 **CHAMP** 基准测试，该基准旨在通过提供提示的注释问题来评估 LLM 的数学推理能力。
   - 该数据集将探索额外的上下文如何在复杂条件下辅助问题解决，促进该领域的进一步研究。
- **Fireworks 和 Together 的可靠性问题**：讨论揭示 **Fireworks** 和 **Together** 都被认为并非 **100% 可靠**，促使实施 **failovers** 以维持功能。
   - 在可靠性得到加强保证之前，用户对使用这些工具持谨慎态度。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **无技术背景进入科技行业**：一位成员表达了在没有技术技能的情况下进入科技行业的渴望，并寻求关于撰写有吸引力的简历和有效建立人脉（Networking）的建议。
   - 另一位成员提到通过 PerScholas 开始 **cybersecurity training**（网络安全培训），强调了对 **coding** 和 **AI** 日益增长的兴趣。
- **Bing Copilot 对比 Perplexity AI**：一位用户对比了 Bing Copilot 提供 **5 个来源**及内嵌图片的能力与 Perplexity 的功能，并提出了改进建议。
   - 他们暗示为引用内容集成 **hover preview cards**（悬停预览卡片）可能是 Perplexity 的一个有价值的增强功能。
- **Perplexity AI 的推荐计划**：Perplexity 正在推出一项专门针对学生的 **merch referral program**（周边商品推荐计划），鼓励通过分享来获取奖励。
   - 有人提出了关于一年免费访问权限可用性的问题，特别是针对前 **500 名注册用户**。
- **Web3 职位空缺**：一则帖子强调了一个 Web3 创新团队的 **job openings**（职位空缺），正在寻找 Beta 测试人员、开发人员和 UI/UX 设计师。
   - 他们邀请提交申请和提案，以创造互助合作的机会，作为其愿景的一部分。
- **Sutskever 的 SSI 获得 10 亿美元融资**：**Sutskever 的 SSI** 成功筹集了 **10 亿美元**，以推动 AI 技术的进步。
   - 这笔资金旨在推动 **AI 领域**的进一步创新。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **悬赏任务探索引发兴趣**：一位用户表示有兴趣尝试悬赏任务（bounty）并寻求指导，参考了关于如何[提问的智慧](http://www.catb.org/~esr/faqs/smart-questions.html)的资源。
   - 这引发了另一位成员的幽默回应，突显了社区在悬赏任务讨论中的参与度。
- **Tinygrad 价格降至零**：在一个令人惊讶的转折中，georgehotz 确认 **4090 + 500GB** 方案的价格已降至 **$0**，但仅限 tinygrad 的朋友。
   - 这促使 r5q0 询问成为朋友的标准，为对话增添了轻松的氛围。
- **澄清 PHI 操作的困惑**：成员们讨论了 IR 中 PHI 操作的功能，注意到其与 LLVM IR 相比不寻常的放置位置，特别是在循环中。
   - 一位成员建议将其重命名为 ASSIGN，因为它的运作方式与传统的 phi 节点不同，旨在消除误解。
- **理解 MultiLazyBuffer 的特性**：一位用户对 `MultiLazyBuffer.real` 属性及其在收缩（shrinking）和复制到设备交互中的作用提出了疑问。
   - 这一询问引发了讨论，揭示了它代表设备上的真实 lazy buffers 以及配置中潜在的 bug。
- **View 与内存挑战**：成员们对 `_recurse_lb` 函数中 view 的实现（realization）表示持续困惑，质疑优化与利用率之间的平衡。
   - 这种反思强调了对基础 tensor view 概念进行澄清的必要性，并邀请社区投入以完善理解。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **分享 Gemma 2 模型资源**：成员们讨论了 [Gemma 2 model card](https://huggingface.co/google/gemma-2-9b)，提供了来自 **Google** 轻量级模型系列的技术文档链接。
   - 资源包括 [Responsible Generative AI Toolkit](https://ai.google.dev/responsible) 以及指向 [Kaggle](https://www.kaggle.com/models/google/gemma-2) 和 [Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) 的链接，强调了 AI 伦理实践。
- **多模态模型与因果掩码**：一位成员概述了多模态设置在推理过程中使用 **causal masks**（因果掩码）面临的挑战，重点关注固定序列长度。
   - 他们指出，*通过注意力层暴露这些变量*对于有效解决此问题至关重要。
- **期待 Flex Attention 带来的加速**：人们乐观地认为，带有文档掩码（document masking）的 **flex attention** 将显著提升性能，在 **A100 上实现 40% 的加速，在 4090 上实现 70% 的加速**。
   - 这将改进 **dynamic sequence length**（动态序列长度）训练，同时最大限度地减少填充（padding）带来的低效。
- **关于 TransformerDecoder 设计的疑问**：一位成员询问 **TransformerDecoder** 是否可以在没有自注意力（self-attention）层的情况下运行，挑战了其传统结构。
   - 另一位成员指出，*原始的 Transformer 利用了* 交叉注意力（cross-attention）和自注意力，使得这种偏离变得复杂。
- **PR 更新标志着生成工具的重构**：成员们确认 GitHub PR **#1449** 已更新，以增强与 `encoder_max_seq_len` 和 `encoder_mask` 的兼容性，测试仍在进行中。
   - 此次更新为进一步修改 **generation utils** 以及与 **PPO** 的集成铺平了道路。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **llama-deploy 提供微服务魔力**：全新的 **llama-deploy** 系统增强了基于 [LlamaIndex Workflows](https://twitter.com/llama_index/status/1831794126511337880) 的微服务部署。这为简化类似于之前 llama-agents 迭代的 Agent 系统提供了机会。
   - 社区分享的一个示例展示了使用 **llama-deploy** 与 **@getreflex** 的全栈能力，演示了如何有效地构建 Agent 聊天系统。
- **PandasQueryEngine 面临列名混淆问题**：用户报告称 **PandasQueryEngine** 在识别 `averageRating` 列时遇到困难，经常在对话中退回到错误的标签。建议包括在 Chat Engine 的上下文中验证映射。
   - 这种混淆可能会在将 Engine 响应与预期输出格式集成时导致更深层次的数据完整性问题。
- **利用 RAG 开发客户支持机器人**：一位用户正在探索如何创建一个将对话引擎与检索增强生成 (RAG) 高效集成的客户支持聊天机器人。成员们强调了 Chat Engine 和 Query Engine 之间的协同作用，以实现更强大的数据检索能力。
   - 验证这种集成可以提升在有效支持至关重要的现实应用中的用户体验。
- **报告 NeptuneDatabaseGraphStore Bug**：关于 **NeptuneDatabaseGraphStore.get_schema()** 的一个 Bug 引起了关注，该 Bug 会导致图摘要中丢失日期信息。怀疑该问题可能与 LLM 的 Schema 解析错误有关。
   - 社区成员表示需要进一步调查，特别是围绕 `datetime` 包在故障中所起的作用。
- **Azure LlamaIndex 与 Cohere Reranker 查询**：一场关于将 Cohere Reranker 作为后处理器集成到 Azure 的 **LlamaIndex** 中的讨论展开了。成员们确认，虽然目前不存在 Azure 模块，但由于文档简单明了，创建一个是可行的。
   - 鼓励社区考虑构建此集成，因为它能显著增强 Azure 环境中的处理能力。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Reflection Llama-3.1：顶级 LLM 的重新定义**：[Reflection Llama-3.1 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) 现被誉为领先的开源 LLM，通过 **Reflection-Tuning** 增强了推理准确性。
   - 该模型是在由 [Glaive](https://glaive.ai) 生成的合成数据上训练的，可以通过[此链接](https://reflection-playground-production.up.railway.app/)进一步探索。
- **快速产出结果的合成数据集生成**：讨论集中在 Reflection Llama-3.1 合成数据集的快速生成上，引发了对 **human rater** 参与度和样本量的好奇。
   - 成员们辩论了合成数据集创建中速度与质量之间的平衡。
- **接受挑战：微调 Llama 3.1**：成员们提出了关于 **Llama 3.1** 有效 **fine-tuning** 技术的问题，指出其在 **8k 序列长度** 下性能提升显著，并可能通过 **rope scaling** 扩展到 **128k**。
   - 对微调复杂性的担忧也随之出现，建议采用自定义 Token 策略以获得最佳性能。
- **SmileyLlama 来了：认识化学语言模型**：[SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148) 作为一个经过微调的**化学语言模型**脱颖而出，旨在根据指定属性创建分子。
   - 该模型被标记为 **SFT+DPO** 实现，展示了 **Axolotl** 在专业模型适配方面的实力。
- **GPU 算力：Lora 微调见解**：关于使用 **A100 80 GB GPU** 以 **4 bit** 模式并配合 **adamw_bnb_8bit** 对 **Meta-Llama-3.1-405B-BNB-NF4-BF16** 进行微调的咨询，强调了有效进行 **Lora finetuning** 的资源需求。
   - 这指出了高效管理 Lora 微调过程所必需的实际考虑因素。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **探索 Cohere 的功能与 Cookbook**：成员们讨论了查看专门用于[功能与演示](https://discord.com/channels/1218409701339828245)的频道，社区在该频道分享了使用 Cohere 模型构建的项目，并参考了提供现成指南的全面 [cookbook](https://docs.cohere.com/page/cookbooks)。
   - 一位成员强调，这些 cookbook 展示了利用 Cohere 生成式 AI 平台的最佳实践。
- **通过 Anthropic 库了解 Token 使用情况**：一位成员询问了关于使用 Anthropic 库的问题，并分享了一个用于计算 Token 使用情况的代码片段：`message = client.messages.create(...)`。
   - 他们引导其他人前往 Anthropic SDK 的 [GitHub 仓库](https://github.com/anthropics/anthropic-sdk-python)以进一步探索 Tokenization。
- **Embed-Multilingual-Light-V3.0 在 Azure 上的可用性**：一位成员询问了 `embed-multilingual-light-v3.0` 在 Azure 上的可用性，并询问是否有支持计划。
   - 这一询问反映了人们对 Cohere 资源与流行云平台集成的持续关注。
- **关于 RAG 引用的查询**：一位成员询问在使用带有外部知识库的 **RAG** 时，引用将如何影响文本文件的内容，特别是询问在目前获得结果为 **None** 的情况下如何接收引用。
   - 他们表达了解决文本文件响应中缺失引用问题的紧迫性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chroma DB 设置简化**：一位成员指出，启动 **Chroma DB** 服务器仅需一行代码：`!chroma run --host localhost --port 8000 --path ./ChomaM/my_chroma_db1`，并注意到其设置非常简便。
   - *他们对如此简单就能确定数据库位置感到宽慰。*
- **Weaviate 设置咨询**：同一位成员询问是否有类似于 Chroma DB 的 **Weaviate** 简单设置，以避免 **Go Docker** 的复杂性。
   - *由于非技术背景，他们表达了对简易操作的需求。*
- **用于服务器-客户端通信的 Jupyter Notebook**：另一位成员分享了他们使用 **两个 Jupyter Notebook** 分别运行服务器和客户端的做法，强调这符合他们的需求。
   - *他们自称是生物学家，寻求不复杂的解决方案。*
- **Reflection 70B 夺冠**：**Reflection 70B** 已被宣布为领先的开源模型，其特点是采用 **Reflection-Tuning** 使模型能够纠正自己的错误。
   - *一个新模型 **405B** 将于下周推出，承诺提供更出色的性能。*
- **通过定价优化 LLM 路由**：围绕根据查询路由合适的 LLM 展开了讨论，意图将 **定价** 和 **TPU 速度** 等因素纳入逻辑。
   - *参与者指出，虽然路由 LLM 的思路很清晰，但通过性能指标进行增强可以精细化选择过程。*

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SwarmUI 易用性担忧**：成员们对展示 **100 个节点** 的用户界面表示不适，并将其与 **SwarmUI** 进行对比，进一步强调了其易用性问题。
   - 讨论强调了将其标记为“简直就是 SwarmUI”反映了工具中 UI 复杂性的广泛担忧。
- **GitHub 上的 SwarmUI 模块化设计**：分享了 [GitHub 上的 SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) 链接，其特点是专注于模块化设计，以实现更好的可访问性和性能。
   - 该仓库强调提供对强力工具（powertools）的便捷访问，通过结构良好的界面增强易用性。
- **Reflection 70B 作为开源领导者亮相**：**Reflection 70B** 的发布已被宣布为首个使用 **Reflection-Tuning** 的顶级开源模型，使 LLM 能够自我纠错。
   - 预计下周将推出 **405B 模型**，其粉碎现有基准测试性能的潜力令人侧目。
- **自我纠错 LLM 引起轰动**：围绕一种能够自我纠错的 LLM 展开了新讨论，据报道该模型在包括 **MMLU** 在すす的所有基准测试中均优于 **GPT-4o**。
   - 该模型的开源特性以及超越 **Llama 3.1 405B** 的表现，标志着 LLM 功能的重大飞跃。
- **Lucidrains 重构 Transfusion 模型**：Lucidrains 分享了 **Transfusion** 模型的 [GitHub 实现](https://github.com/lucidrains/transfusion-pytorch)，在扩散图像的同时优化下一 Token 预测。
   - 未来的扩展可能会集成 **Flow Matching 以及音频/视频处理**，预示着强大的多模态能力。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **ReAct Agent 部署挑战**：一位成员在通过 FastAPI 在 **GCP** 上部署其 **ReAct Agent** 时遇到困难，面临重新部署时本地 SQLite 数据库消失的问题。他们正在寻求 **Postgres 或 MySQL** 作为 `SqliteSaver` 的替代方案。
   - 该成员愿意分享他们的本地实现以供参考，希望能找到协作解决方案。
- **澄清 LangChain Callbacks 用法**：关于语法 `chain = prompt | llm` 准确性的讨论出现，参考了 [LangChain 的 callback 文档](https://python.langchain.com/v0.1/docs/modules/callbacks/)。成员们指出文档似乎已经过时，特别是 **0.2** 版本的更新。
   - 对话强调了 **Callbacks** 在日志记录、监控和第三方工具集成中的实用性。
- **Cerebras 与 LangChain 协作咨询**：一位成员询问了 **Cerebras** 与 **LangChain** 结合使用的情况，寻求他人的协作见解。回复表示有兴趣，但尚未分享具体的经验或解决方案。
   - 这一话题在社区内仍有待进一步探索。
- **解码 .astream_events 的困境**：成员们讨论了缺乏解码 **.astream_events()** 流的参考资料，其中一人对不得不手动序列化事件表示沮丧。对话表达了对更好资源和解决方案的渴望。
   - 这一繁琐的过程凸显了社区协作和资源共享的必要性。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **在有限硬件下增强 RAG**：一位成员寻求在受限的 **4090 GPU** 硬件条件下，使用带有 **4bit 量化** 的 **llama3-8b** 以及 **BAAI/bge-small-en-v1.5** 嵌入模型来升级其 **RAG 系统** 的策略。
   - *在寻求更好的实现资源时，* 他们表达了硬件限制，强调了对高效实践的需求。
- **利用更大模型最大化 GPU 潜力**：作为回应，另一位成员建议 **4090** 可以并发运行更大的嵌入模型，并指出 **3.1 版本** 也可能提升性能。
   - 他们提供了一个 [GitHub 示例](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py)，展示了在 Milvus 上集成涉及 **bge & bm25** 的混合搜索。
- **利用元数据进行更好的重排序 (Reranking)**：对话强调了 **每个 chunk 的元数据** 的关键作用，建议它可以改进返回结果的排序和过滤。
   - 他们认为，*实现一个重排序器 (reranker)* 可以显著提高用户搜索的输出质量。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **XLAM 系统提示词引发好奇**：一位成员指出 **XLAM 的系统提示词** 与其他 **OSS 模型** 相比非常独特，并对这种设计选择背后的基本原理提出疑问。
   - 讨论揭示了人们对于这些差异是源于 **功能性** 还是 **许可考虑** 的兴趣。
- **测试 API 服务器需要指导**：一位用户寻求测试其 **API 服务器** 的有效方法，但未收到具体的文档回复。
   - 共享资源的缺失凸显了社区支持和知识共享方面潜在的增长空间。
- **如何将模型添加到排行榜 (Leaderboard)**：一位用户询问了将新模型添加到 **Gorilla 排行榜** 的流程，并得到了相关指南的回复。
   - 访问 [GitHub 页面](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 上的贡献详情，以了解如何促进模型收录。
- **Gorilla 排行榜资源亮点**：成员们讨论了 **Gorilla: Training and Evaluating LLMs for Function Calls** GitHub 资源，该资源概述了排行榜的贡献方式。
   - 同时也分享了其仓库中的一张图片，说明了为有兴趣参与的用户提供的指南，详见 [GitHub](https://opengraph.githubassets.com/25d4bf4245a01dd99c8e3d1e4b47d26ef3db55d11499f2f9edfa259231aaacd2/ShishirPatil/gorilla)。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **来自 Knut09896 的问候**：Knut09896 进入频道并打了招呼，引发了欢迎互动。
   - 这个简单的问候暗示了 **Alignment Lab AI** 社区内持续的参与度。
- **频道活动热度**：**#general** 频道的活跃度看起来非常高，成员们在闲聊并进行自我介绍。
   - 这种互动在促进社区联系和协作讨论方面发挥着至关重要的作用。



---


**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---

# 第二部分：各频道详细摘要与链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1281347738113675306)** (1 条消息): 

> - `Vision Language Models`
> - `Tau LLM Training Optimization`
> - `African Language Models`
> - `No-Code AI Model Tasks`
> - `Selective Fine-Tuning of Language Models` 


- **视觉语言模型简介**：一位成员分享了一篇关于视觉语言模型的 [博客文章](https://www.lightly.ai/post/introduction-to-vision-language-models)，对该主题进行了简明扼要的概述。
   - 这篇入门文章旨在阐明在 AI 应用中结合视觉和语言的潜力。
- **优化 Tau LLM 训练**：[Tau LLM 系列](https://youtube.com/live/flwqvE4aSzA?feature=share) 专注于优化训练过程并增强模型性能。
   - 该系列被誉为重要资源，有望简化学习 LLM 训练细节的过程。
- **InkubaLM-0.4B 针对非洲语言**：新发布的 [InkubaLM-0.4B](https://huggingface.co/spaces/Tonic/Inkuba-0.4B) 旨在支持非洲语言并扩大语言代表性。
   - 该模型专为此目的开发，展示了对 AI 语言模型包容性的承诺。
- **Shadowbox 提供无代码模型任务构建**：介绍了 [Shadowbox](https://github.com/darkshapes/singularity)，这是一个使用 FOSS 模型的 AI 任务无代码构建器，简化了用户体验。
   - 用户无需编程专业知识即可创建任务，拓宽了 AI 解决方案的可及性。
- **使用 Spectrum 进行选择性微调**：讨论了语言模型选择性 [Fine-Tuning](https://huggingface.co/blog/anakin87/spectrum) 的概念，强调了其优势。
   - 通过专注于某些方面，无需全面重新训练即可实现更精细的模型性能提升。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1281336016963375144)** (258 条消息🔥🔥): 

> - `Code Generation Evaluations` (代码生成评估)
> - `Model Training Issues` (模型训练问题)
> - `Data Handling for Training` (训练数据处理)
> - `Fine-tuning and Pre-training` (微调与预训练)
> - `Performance Analysis of Models` (模型性能分析)


- **代码生成中的性能分析**：讨论包括分析函数在数据集中出现的频率，以及常用函数是否会导致更少的错误，并探索了考虑功能正确性的指标。
   - 贡献者指出，模型生成的函数如果是近乎精确的克隆，可能表明训练数据存在污染。
- **模型训练设置中的挑战**：成员们遇到了与硬件限制相关的问题，几位成员讨论了在有效利用 GPU 资源训练模型方面的困扰。
   - 一位用户询问了关于使用 Hugging Face 等平台进行训练的问题，表达了对本地设置资源不足的担忧。
- **关于预训练和数据质量的见解**：分享了一篇论文，指出在预训练数据集中包含代码的影响，及其对非代码任务和整体模型性能的好处。
   - 参与者辩论了从训练集中排除代码是否会导致模型输出效果降低。
- **生成脚本与模型测试**：提供了一个用于从指定基础模型生成输出的最小脚本，强调了结果后处理中的潜在问题。
   - 尽管对基于上下文长度的模型质量存在一些担忧，但仍鼓励用户测试该脚本并分析生成结果。
- **对模型评估指标的反思**：大家达成共识，认为代码生成的静态指标并不理想，讨论强调了语义正确性和功能性输出的重要性。
   - 参与者反思了包括编辑距离在内的某些指标如何与模型性能和可靠性相关联。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/help/how-to-ask">How do I ask a good question? - Help Center</a>：Stack Overflow | 全球最大的开发者在线社区</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch">ShaderMatch - Vipitis 开发的 Hugging Face Space</a>：暂无描述</li><li><a href="https://arxiv.org/abs/2306.03203">A Static Evaluation of Code Completion by Large Language Models</a>：在代码上训练的大语言模型在提高软件开发人员生产力方面显示出巨大潜力。已经提出了几种基于执行的基准来评估功能正确性...</li><li><a href="https://huggingface.co/datasets/nroggendorff/multi-csv">nroggendorff/multi-csv · Hugging Face 数据集</a>：暂无描述</li><li><a href="https://arxiv.org/abs/2408.10914">To Code, or Not To Code? Exploring Impact of Code in Pre-training</a>：在预训练数据混合物中包含代码，即使对于并非专门为代码设计的模型，也已成为 LLM 预训练中的常见做法。虽然在从业者之间存在一些轶事式的共识...</li><li><a href="https://huggingface.co/docs/diffusers/tutorials/basic_training#create-a-unet2dmodel">Train a diffusion model</a>：训练扩散模型教程</li><li><a href="https://openrouter.ai/models/mattshumer/reflection-70b">Reflection 70B - API, Providers, Stats</a>：Reflection Llama-3.1 70B 使用一种名为 Reflection-Tuning 的新技术进行训练，该技术教导 LLM 检测其推理中的错误并纠正方向。通过 API 运行 Reflection 70B
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1281362437861412894)** (8 条消息🔥): 

> - `Understanding Attention Mechanism in Transformers`
> - `Discussions on Cross-posting`
> - `Using AI for Tutoring Kids`
> - `Creating a Python Microservice with Ollama` 


- **寻求关于 Attention 机制的解答**：一位成员询问如何表示 Transformers 中给定 token 的 attention，特别是它是否与 token 之间在 latent vector space 中的距离有关。
   - 他们请求提供资料以帮助更好地理解这一概念，表明需要进一步的解释。
- **关于 Cross-posting 礼仪的提醒**：多位成员讨论了在频道中跨频道发布问题的问题，其中一位请求停止在不同频道发送相同的信息。
   - *一位成员更倾向于遵循其他频道的建议*而非当前给出的建议，促使另一位成员声明在一个频道发布就足够了。
- **非训练营模式的儿童 AI 辅导**：一位成员分享了关于如何在没有正式 Bootcamp 压力的情况下辅导孩子学习 AI 的经验。
   - 这种方法建议以一种更具参与感且结构化程度较低的方式向儿童介绍 AI 概念。
- **使用 Ollama 开发 Python 微服务**：一位成员询问如何使用 Ollama 创建一个 Python 微服务，该服务可以以十种不同的方式对句子进行改写 (paraphrase)。
   - 这一请求表明了对 AI 在文本处理任务中实际应用的兴趣。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1281447142539661333)** (2 条消息): 

> - `Elasticsearch`
> - `Vespa Search Engine` 


- **告别 Elasticsearch，你好 Vespa 搜索引擎**：一位成员在推文中宣布他们从 **Elasticsearch** 迁移到 **Vespa Search Engine**，引起了一些关注。
   - 他们使用了一个表情符号来表达兴奋：*'👀'*，表示对这一变化的积极期待。
- **关于搜索引擎技术的讨论**：从 **Elasticsearch** 到 **Vespa** 的转变引发了关于不同搜索引擎技术及其优势的对话。
   - 参与者对 **Vespa** 与传统解决方案相比的性能和特性表示好奇。



**提到的链接**：<a href="https://x.com/jobergum/status/1831701040812450156">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：Goodbye Elasticsearch, Hello Vespa Search Engine 👀

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1281386617055215648)** (14 条消息🔥): 

> - `Pro-Pretorian Computer Vision System`
> - `Interactive Model Comparator`
> - `Chess Puzzle Visualization`
> - `Tau LLM Series Update` 


- **Pro-Pretorian Computer Vision System 发布**：一位成员分享了他们完成的第一版 **Pro-Pretorian Computer Vision System**。这是一个托管在 Azure 上的 Next.js 应用，数据持久化同样在 Azure 上，利用 **tfjs** 通过 **WebGL** 进行推理。
   - 他们计划通过添加微调模型（fine-tuned models）并利用其 [Hugging Face 账户](https://github.com/salim4n/pro-pretorian-system) 创建流水线（pipeline）来实现自动化，从而增强该系统。
- **Interactive Model Comparator 介绍**：另一位成员展示了 **Interactive Model Comparator**，这是一个专为视觉化比较不同机器学习模型在计算机视觉任务中输出图像而设计的 Web 工具。
   - 该工具允许用户加载图像、在模型间切换并实时预览对比结果，对于研究人员和开发者来说是一个宝贵的资源，可在 [GitHub](https://github.com/kawchar85/InteractiveModelComparator) 上获取。
- **可视化 400 万个国际象棋谜题**：该项目重点介绍了利用 **Hugging Face datasets** 可视化 **400 万个国际象棋谜题**，由 Stockfish 提供评估，详细记录了超过 **8300 万个** 棋局位置。
   - 关键细节包括数据格式以及指向 [Lichess 数据库](https://database.lichess.org/#puzzles) 的链接，以便进一步探索国际象棋评估。
- **Tau LLM 系列的精彩更新**：**Tau LLM 系列** 的 **第 15 集** 介绍了多项更新，包括自动化的 **数据文件去重（de-duplication）** 以及用于生成改写（paraphrases）的新 **ophrase Python 模块**，从而增强了数据集的多样性。
   - 本集预告了新嵌入（embeddings）的生成以及向训练扩展数据集的转变，旨在提高效率并降低熵，通过 [YouTube 链接](https://youtube.com/live/dOh_FEs12e4?feature=share) 分享。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/christopher/status/1832040993522147717">来自 Christopher Akiki (@christopher) 的推文</a>：400 万个 @lichess 国际象棋谜题</li><li><a href="https://huggingface.co/spaces/Muinez/Image-scorer">Image Scorer - Muinez 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/kawchar85/InteractiveModelComparator">GitHub - kawchar85/InteractiveModelComparator：一个基于 Web 的工具，旨在比较同一数据集上不同机器学习模型的输出图像。</a>：一个基于 Web 的工具，旨在比较同一数据集上不同机器学习模型的输出图像。 - kawchar85/InteractiveModelComparator</li><li><a href="https://youtube.com/live/dOh_FEs12e4?feature=share">Unity ML-Agents | 从零开始使用 Sentence Transformers 预训练 LLM | 第 15 部分</a>：**欢迎回到我们的 Tau LLM 系列！🌟** 在本集中，我们将把项目提升到一个新的水平，带来一些令人兴奋的新进展。我们的亮点包括...</li><li><a href="https://github.com/salim4n/pro-pretorian-system">GitHub - salim4n/pro-pretorian-system：该项目是一个计算机视觉应用，允许用户根据需求自定义检测参数。无论您是想检测特定对象、定义感兴趣区域还是安排检测时间，该应用都提供了灵活且强大的选项。</a>：该项目是一个计算机视觉应用，允许用户根据需求自定义检测参数。无论您是想检测特定对象、定义感兴趣区域...</li><li><a href="https://database.lichess.org/#puzzles">lichess.org 开放数据库</a>：未找到描述</li><li><a href="https://database.lichess.org/#">lichess.org 开放数据库</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

noaroggendorff: <@&1078351789843292311>
  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1281597955023048794)** (1 messages): 

> - `Optimizing Flux and Cog`
> - `Diffusion models`
> - `TorchAO` 


- **发布了用于优化的新 Recipe 仓库**：发布了一个新的 [GitHub 仓库](https://github.com/sayakpaul/diffusers-torchao)，展示了如何使用 **diffusers** 和 **torchao** 优化 **Flux** 和 **Cog**，包括推理和 FP8 训练。
   - 该仓库允许通过 **quantization** 和各种 **offloading** 方法，在仅 **3.1GB** 显存下运行 **Cog**。
- **Diffusion Models 的端到端优化**：该仓库提供了全面的 recipes，旨在优化 **diffusion models**，使其在训练和推理中更加高效。
   - 它重点介绍了 **offloading** 和 **quantization** 等技术，这对于处理大型模型需求至关重要。



**提到的链接**：<a href="https://github.com/sayakpaul/diffusers-torchao">GitHub - sayakpaul/diffusers-torchao: End-to-end recipes for optimizing diffusion models with torchao and diffusers (inference and FP8 training).</a>：使用 torchao 和 diffusers 优化 diffusion models 的端到端 recipes（推理和 FP8 训练）。 - sayakpaul/diffusers-torchao

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1281476646783815700)** (2 messages): 

> - `Medical dataset for disease detection`
> - `Training Nougat and Donut` 


- **在 CV 领域寻找医学数据集**：一位成员正在寻找用于 **computer vision** 的优质医学数据集，目标是 **disease detection**，或者可能利用 transformers 进行大规模数据生成。
   - 他们对能够促进大规模 **data generation** 工作的数据集表示了兴趣。
- **Nougat 和 Donut 的训练方法**：另一位成员询问是否有熟悉 **Nougat** 或 **Donut** 模型训练细节的人。
   - 这可能表明其希望深入了解与这些框架相关的模型架构或训练技术。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1281447213125599292)** (4 messages): 

> - `OOM errors during evaluation`
> - `DeepSpeed configuration for evaluation`
> - `Custom Dataset for evaluation`
> - `GPU distribution techniques` 


- **OOM 错误困扰评估阶段**：尽管在多 GPU 上训练成功，但团队在使用 DeepSpeed 的自定义设置进行评估时遇到了 **OOM 错误**。
   - 据观察，较小的 batches（<10 个样本）评估正常，而较大的 batches（>100 个样本）则触发了错误，从而引发了关于 GPU 加载的问题。
- **推荐使用自定义 Dataset 进行评估**：一位成员建议利用产生特定 **batch sizes** 的 **自定义 Dataset** 来缓解 OOM 错误，并建议先用 **50 个样本** 进行测试评估。
   - 他们引用了 [PyTorch Dataset 教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)来指导如何实现。
- **实现多 GPU 分布**：建议使用自定义评估循环将数据加载到特定的 GPU 上，以促进在多个 GPU 之间的分布。
   - 建议使用 `data.to('cuda:1')` 等方法加载到单个 GPU 上，以直接解决 OOM 问题。
- **针对较小 Batches 的自定义评估循环**：Nympheliaa 确认使用了自定义数据集，并询问如何为 **GPU 分布** 创建具有较小 batches 的 **自定义评估循环**。
   - 他们表示打算利用 **torch DataParallel** 或 **DistributedDataParallel** 等技术来更好地管理 GPU 资源。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1281447186630180905)** (10 条消息🔥): 

> - `Flux img2img Pipeline`
> - `SD3 vs. SDXL 模型`
> - `针对 SDXL 的 ControlNets`
> - `Auto Class 推荐`
> - `内存优化` 


- **Flux img2img Pipeline 尚未合并**：一位成员指出 **Flux img2img** 功能尚未合并，并引用了一个[相关的公开 PR](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)。另一位成员确认文档中包含有关 Flux 的信息，包括指向其[博客文章](https://blackforestlabs.ai/announcing-black-forest-labs/)的链接。
   - 在消费级硬件上运行 Flux 可能成本较高，但可以进行优化，正如在[相关博客文章](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3)中所讨论的那样。
- **探索 Img2Img Pipeline 的替代方案**：当被问及 **Flux img2img Pipeline** 的替代方案时，一位成员建议在通用情况下使用 **SD3** 模型，在涉及人物的高质量图像时使用 **SDXL**。他们还强调了探索 **ControlNets** 以增强功能。
   - 另一位成员询问了适用于 SDXL 的热门 **ControlNets**，回复中包含了 **ControlnetUnion** 和 **Mistoline** 等建议。
- **澄清 Auto Class 的用法**：一位用户询问在开始使用 SD 时，是否应该简单地为 Img2Img 替代方案使用 **Auto class**。对话随后转向了对高质量输出（特别是涉及人物图像）的模型偏好。
- **文档差异**：有关于**文档**差异的讨论，文档中提到了一项尚未合并的功能。澄清指出，**使用 main 分支**引用的是可能尚未完全集成的功能。



**提到的链接**：<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux">Flux</a>：未找到描述

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1281331684104736862)** (274 messages🔥🔥): 

> - `ControlNet 与模型使用`
> - `图像生成：Flux vs. SDXL`
> - `诈骗与网络安全`
> - `ComfyUI 中的打标与工作流`
> - `Forge 中的扩展集成` 


- **ControlNet 指南与模型搭配**：用户讨论了如何有效地使用 ControlNet，特别强调了将其与 Loras 以及各种 SDXL 模型结合使用，以创建如 hash rosin 图像等精确表达。
   - 推荐的模型包括 'Flux'，并提到了如何集成深度图（depth maps）等技术来帮助实现预期效果。
- **在 Flux 和 SDXL 之间选择 Logo 生成**：推荐使用 Flux 而非 SDXL 来生成 Logo，因为它处理 Logo 的效果非常出色，且无需大量训练即可轻松进行提示词（prompting）操作。
   - 相反，用户分享了使用 SDXL 生成 Logo 的困难，原因是对 Logo 缺乏熟悉度，因此主张利用 Flux 的能力。
- **网络安全与诈骗讨论**：成员们分享了关于网络诈骗的轶事，并强调了警惕的重要性，回忆起即使是经验丰富的人在脆弱时刻也可能受害。
   - 同理心被强调为理解导致诈骗行为的关键方法，表明诈骗并非仅针对天真的人。
- **ComfyUI 中的打标技术与工具**：对话包括使用 ComfyUI 进行打标，将其界面功能比作面向 LLM 模型的 Langflow 和 Flowise。
   - 社区成员讨论了 ComfyUI 中的特定工作流以及为增强打标效果所做的调整，强调了其提供的灵活性。
- **Forge 扩展与社区贡献**：用户询问了 Forge 中可用的各种扩展，包括用于 ControlNet 的扩展，以及这些扩展如何有助于改善用户体验。
   - 提到了社区投票及其影响，建议反馈可能会影响未来的发布，强调了质量保证的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/h94/IP-Adapter">h94/IP-Adapter · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/p1atdev/wd-swinv2-tagger-v3-hf">p1atdev/wd-swinv2-tagger-v3-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://viralshort.ai/">Viralshort - TikTok &amp; Youtube Shorts 制作变得简单！</a>: 使用 AI 在几秒钟内自动生成 TikTok、Instagram 和 Youtube Shorts 短视频内容的最快方式！</li><li><a href="https://civitai.green">Civitai: 开源生成式 AI 之家</a>: 探索数千个高质量的 Stable Diffusion 模型，分享您的 AI 生成艺术，并与充满活力的创作者社区互动</li><li><a href="https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main">lllyasviel/ControlNet-v1-1 at main</a>: 未找到描述</li><li><a href="https://huggingface.co/lllyasviel/sd_control_collection/tree/main">lllyasviel/sd_control_collection at main</a>: 未找到描述</li><li><a href="https://civitai.com/models/487689/hash-rosin">Hash Rosin - v1.0 | Stable Diffusion LoRA | Civitai</a>: 这个 Lora 可以重现罐中和 Dabbers 上的 Hash Rosin 特写宏观镜头。它也足够灵活，可以用 Rosin 制作像 anima...
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1281332257029623874)** (189 messages🔥🔥): 

> - `祝贺获得 Y Combinator 支持`
> - `Unsloth AI 功能与支持`
> - `用于合成数据生成的模型`
> - `Reflection 模型性能`
> - `Unsloth 的硬件要求`

- **恭喜团队获得 YC 支持**：成员们对团队入选 Y Combinator 表示祝贺，并对他们的旅程表达了兴奋和支持。
   - 团队回馈了感谢，并承认了社区支持的重要性。
- **Unsloth 的硬件兼容性受到质疑**：关于 Unsloth 与 Mac 系统的兼容性引发了讨论，特别是涉及 GPU 任务的 CUDA 支持。
   - 团队澄清说，他们的目标是实现硬件无关性，但目前的局限性影响了某些配置下的性能。
- **合成数据生成模型推荐**：Kearm 分享了使用 Mistral 8x7B 微调版生成合成数据的见解，同时也建议了其他模型，包括 jondurbin/airoboros-34b-3.3。
   - 成员们讨论了根据特定硬件限制对这些模型进行实验以获得最佳结果。
- **Reflection 模型性能疑虑**：成员们对 Matt Shumer 的 Reflection 模型表达了褒贬不一的看法，指出与 Claude 3.5 和 GPT-4 等其他模型相比，它在私有逻辑问题上的表现并不理想。
   - 对于该模型的能力以及其作为顶级开源 LLM 的说法，目前仍存在持续的怀疑。
- **Mac 用户的移植挑战**：成员们讨论了为 Mac 用户移植 bitsandbytes 和 Triton 等 Unsloth 功能的需求，强调了 Mac 芯片缺乏 CUDA 支持的问题。
   - 对话强调了在尝试优化软件兼容性的同时，难以证明在硬件上投入高额支出的合理性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/unclemusclez/ollamafy">Ollamafy (进行中) - unclemusclez 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Reflection-Llama-3.1-70B-bnb-4bit">unsloth/Reflection-Llama-3.1-70B-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mlabonne/Hermes-3-Llama-3.1-8B-lorablated">mlabonne/Hermes-3-Llama-3.1-8B-lorablated · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/republica-de-fifidonia-rick-idk-fake-it-looks-fake-gif-17266845">Republica De Fifidonia Rick GIF - Republica De Fifidonia Rick Idk - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ollama.com/unsloth/unsloth-tutorial">unsloth/unsloth-tutorial</a>: 快速上手大语言模型。</li><li><a href="https://github.com/FailSpy/abliterator">GitHub - FailSpy/abliterator: 用于在支持 TransformerLens 的 LLM 中消融特征的简单 Python 库/结构</a>: 用于在支持 TransformerLens 的 LLM 中消融特征的简单 Python 库/结构 - FailSpy/abliterator</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: 在 DEVINator 数据上训练的 SmolLM 135M Instruct，用于 Open Hands (Open Devin)</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef: 🍲Agent Chef🥘 是我用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和清理其微调数据，消除数据污染和低质量知识库。此外，它还将提供模板和框架。</a>: 🍲Agent Chef🥘 是我用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和 ....</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">通过 abliteration 对任何 LLM 进行去审查</a>: 未找到描述</li><li><a href="https://github.com/Nottlespike/abliterator.py">GitHub - Nottlespike/abliterator.py: 用于在支持 TransformerLens 的 LLM 中消融特征的简单 Python 库/结构</a>: 用于在支持 TransformerLens 的 LLM 中消融特征的简单 Python 库/结构 - Nottlespike/abliterator.py</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mahiatlinux/Reflection-Dataset-v1">mahiatlinux/Reflection-Dataset-v1 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/daveshap/ACE_Framework">GitHub - daveshap/ACE_Framework: ACE (Autonomous Cognitive Entities) - 100% 本地且开源的自主 Agent</a>: ACE (Autonomous Cognitive Entities) - 100% 本地且开源的自主 Agent - daveshap/ACE_Framework
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1281533563585691679)** (10 条消息🔥): 

> - `Evolution of Unsloth` (Unsloth 的演变)
> - `Emoji Communication` (表情符号交流)
> - `App Promotion` (应用推广)


- **Unsloth 的演变 - 一段有趣的旅程**：一名成员分享了一个讨论 [Evolution of the Peaceful Sloth](https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peacefu)（和平树懒的演变）的链接，引发了关于该话题的笑声。
   - 随后出现了表情符号的回应，展示了对该讨论的热情。
- **表情符号作为沟通方式**：在一个轻松的时刻，一名成员开玩笑说自己经过了 Fine-tuning（微调），可以用表情符号来传达信息，为聊天增添了俏皮的基调。
   - “是的，我对自己进行了 Fine-tune 以实现这一点。”
- **围绕应用推广的对话**：一名成员在提到演变话题后，直接分享了一个似乎在推广某个 App 的链接。
   - 这导致另一名成员幽默地表示 *'禁止推广！'*，突显了聊天中自发的打趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peaceful_sloth/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1faa0ur/evolution_of_the_peacefu">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1281402766241038412)** (43 条消息🔥): 

> - `Unsloth Library Installation` (Unsloth 库安装)
> - `Kaggle Competition Constraints` (Kaggle 竞赛限制)
> - `Phi 3.5 Fine Tuning` (Phi 3.5 微调)
> - `Gemma-2-27B Loading Issues` (Gemma-2-27B 加载问题)
> - `Mistral 7B Domain Limitation` (Mistral 7B 领域限制)


- **Unsloth 库安装问题**：一些用户报告了在 Kaggle 上安装 **Unsloth** 库的问题，特别是使用最新的 [notebook 说明](https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-7b-unsloth-notebook/notebook)时。用户寻求关于安装过程更新的帮助。
   - 鼓励参与者分享任何近期的进展，以解决用户一直面临的安装问题。
- **Kaggle 竞赛对互联网访问的限制**：一名成员对 Kaggle 竞赛提交期间要求 **禁止访问互联网** 表示担忧，这影响了他们安装所需模型和库的能力。讨论包括了建议的变通方法和潜在解决方案。
   - 建议包括在关闭互联网之前先在开启互联网的情况下运行某些单元格，尽管有些人认为这不能充分解决问题。
- **Phi 3.5 模板与乱码输出**：用户报告了在尝试进行 Fine-tuning 训练时，Phi 3.5 模型返回 **乱码输出** 的挑战。调整 temperature 和 top_p 等参数并不能为所有用户解决问题。
   - 讨论了寻找合适模板和故障排除的方法，但许多参与者对该模型的表现表示失望。
- **Gemma-2-27B 权重初始化警告**：用户对加载训练好的 **Gemma-2-27B** 模型时出现的 **权重初始化警告** 表示担忧，并引用了相关的 GitHub Issue 作为背景。他们寻求减轻这些警告的变通方法。
   - 在模型加载过程中注意到了异常行为，促使用户向遇到类似问题的其他人寻求解决方案。
- **Unsloth 对 Vision 模型的限制**：有人提出了关于在 Unsloth 中使用 **Phi 3.5 vision 模型** 的问题，但共识是目前尚不支持。预计未来将增加对 Vision LLM 的支持。
   - 用户对 Unsloth 能力的演进表示关注，特别是关于 Vision 相关模型的 Fine-tuning 选项。



**提到的链接**：<a href="https://github.com/unslothai/unsloth/issues/478">Qwen2 error when loading from checkpoint · Issue #478 · unslothai/unsloth</a>：加载基础模型时按预期工作，但当加载 LoRA checkpoint 代替基础模型时，Unsloth 返回：Unsloth cannot patch Attention layers with our manual autograd engine...

  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1281337667858202625)** (2 条消息): 

> - `Comparison Reports` (对比报告)
> - `YouTube Explanations` (YouTube 讲解) 


- **对对比报告的兴趣**：一名成员对一份对比特定主题的报告表示感兴趣，称其 **读起来会很有趣**。
   - *关于此对比，没有提到具体的讨论或报告。*
- **即将发布的关于对比的 YouTube 视频**：另一名成员宣布计划制作一个 [YouTube 视频](https://youtube.com)，详细解释这些对比。
   - *该视频旨在回应大家对对比相关话题的兴趣。*

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1281523816480247848)** (1 条消息): 

> - `Message Duplication` (消息重复)
> - `Channel Oversight` (频道监管)


- **频道内重复发帖**：一名成员质疑在频道中发布某条消息的合理性，指出该消息已在 'help' 频道分享过。
   - *Please remove this*（请删除此内容）是直接提出的请求，表明了对内容重复的沮丧。
- **对频道管理的担忧**：该成员对频道帖子缺乏监管表示不满，强调这导致了参与者的困惑。
   - 这反映了社区内关于组织和维护主题相关性的更广泛担忧。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1281329238435168269)** (142 条消息 🔥🔥): 

> - `Image API options` (图像 API 选项)
> - `Reflection Llama-3.1 70B updates` (Reflection Llama-3.1 70B 更新)
> - `LM Studio issues` (LM Studio 问题)
> - `Scraping data with local LLMs` (使用本地 LLM 抓取数据)
> - `Accessing Llama 3.1 405B model` (访问 Llama 3.1 405B 模型)


- **寻找免费的高额度 Image API 选项**：用户讨论了图像 API 的潜在免费选项，并对提供 Stable Diffusion 等模型 API 访问权限的供应商表示好奇。
   - 他们还询问了是否有供应商能大规模提供这些功能。
- **Reflection Llama-3.1 70B 获得更新**：Reflection Llama-3.1 70B 被誉为顶级的开源 LLM，采用了增强其检测和纠正推理错误能力的新技术。
   - 成员们还注意到了一些性能问题，并讨论了使该模型达到最佳行为的有效 Prompt。
- **LM Studio 模型下载问题**：一名用户报告了在更新到 0.3.2 版本后下载模型的问题，引发了关于证书错误和潜在解决方案的咨询。
   - 社区成员讨论了调整 VRAM 和上下文大小等变通方法，同时也澄清了 RAG 的摘要功能不支持某些函数。
- **网页抓取和本地 LLM 工具**：一名用户咨询了可以连接到 LM Studio 的网页抓取 Agent，回复建议使用 Python 以及 ScrapeGraphAI 等工具。
   - 社区建议集中在先抓取再用 LLM 处理数据的效率上，而不是尝试直接用 LLM 进行抓取。
- **访问 Llama 3.1 405B 模型**：讨论了获取 Llama 3.1 405B 模型访问权限的问题，强调了用户在 meta.ai 网站上面临的访问障碍。
   - 替代建议包括查看 lmarena.ai 或使用其他模型，并对 meta.ai 可能采取的过滤措施进行了推测。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/stable-code-instruct-3b-GGUF">lmstudio-community/stable-code-instruct-3b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Reflection-Llama-3.1-70B-GGUF/tree/main">bartowski/Reflection-Llama-3.1-70B-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/zed-industries/zed/issues/17482">LM Studio offline model support · Issue #17482 · zed-industries/zed</a>: 检查现有问题。已完成。描述功能：在 AI 模型配置页面添加一个部分，允许用户使用来自 LM Studio 的模型。此前已通过 #4424 完成...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/9bc6db28d011d47a5f318dc4aebbe7927fac4629">ggml-quants : ternary packing for TriLMs and BitNet b1.58 (#8151) · ggerganov/llama.cpp@9bc6db2</a>: * ggml-quants : 为 BitNet 1.58b 提供 1.625 bpw 三进制打包
 
 * ggml-quants : 更快的 1.625 bpw AVX2 vec_dot
 
 不再使用查找表，使其速度与 q4_0 匹配。
 
 * gguf-py : 修复格式...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1281374008318623766)** (59 条消息🔥🔥): 

> - `Apple 发布会公告`
> - `Mac Studio 性能担忧`
> - `带 NVLink 的 NVIDIA RTX 3090 性能`
> - `LMStudio 启动时间问题`
> - `Apple 设备上的 NAS 使用` 


- **Apple 发布会定档 iPhone 和 Watch**：即将于 **9/9** 举行的 Apple 发布会已确认将重点展示新款 **iPhones** 和 **watches**。
   - 成员们表达了对最新设备更新的期待。
- **Mac Studio 运行大模型速度较慢**：用户对配备 **256GB+** 内存的 **Mac Studio** 在运行大型 **models** 时速度过慢表示担忧，并希望升级到 **LPDDR5X 10.7Gbps**。
   - 一位成员指出，这可能会显著提升所有 **M4s** 的性能，将速度提高 **70%**。
- **NVLink 提升 NVIDIA RTX 3090 性能**：讨论强调，使用 **2 张 RTX 3090** 运行 **70B model** 时，用户可以达到 **10 到 25 t/s**。
   - 一位成员提到通过 **NVLink** 达到了 **50 t/s**，尽管其他人对其在推理（inference）性能上的实际影响表示怀疑。
- **LMStudio 启动时间延长**：用户报告称 **LMStudio** 的启动时间需要 **15-20 秒**，明显长于更新前的 **2 秒**。
   - 调查表明，互联网连接可能会导致延迟，可能与更新检查有关。
- **Apple 用户的 NAS 讨论**：一位成员分享了使用 **Asustor NAS** 进行存储管理相比桌面设置的良好体验。
   - 讨论中还涉及了为多个设备设置备份以及在家庭设备间高效共享资源的建议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/snapdragon">👾 LM Studio - Discover and run local LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/16ubkyq/nvlink_bridge_worth_it_for_dual_rtx_3090/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/MacOS/comments/1ae3m3z/a_nas_that_actually_works_on_macos/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1281332648672890884)** (190 条消息🔥🔥): 

> - `Reflection 70B 模型`
> - `Hermes 3 和 Llama 3.1 API 使用`
> - `Reflection 和 ICL 性能基准测试`
> - `MCTS 和 PRM 技术`
> - `量化问题` 


- **Reflection 70B 模型性能对比**：最近的讨论强调了 [Reflection 70B model](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) 的结果参差不齐，特别是在与 BigCodeBench-Hard 等基准测试对比时，在某些领域表现较差。
   - 用户指出，system prompts 和 tokenizer 问题可能会显著影响结果，使评估过程复杂化。
- **Llama 模型的 API 选项**：一位成员询问了使用 Llama 3.1 70B 模型的最佳 API 选项，并指出需要支持 tool call 格式。
   - 建议包括探索像 Groq 这样的平台以实现高效部署。
- **探索 MCTS 和 PRM 以增强模型**：对话表明，将 MCTS (Monte Carlo Tree Search) 与 PRM (Probabilistic Roadmap) 结合使用可能会在模型训练和评估中产生更好的效果。
   - 成员们对在项目中测试这些技术表示兴奋。
- **AI 模型的量化挑战**：70B 模型 FP16 版本的量化工作产生的结果令人失望，特别是尝试 int4 量化的用户。
   - 讨论围绕如何在不牺牲质量的情况下提高模型性能的潜在变通方法展开。
- **认知科学概念探索**：一位成员分享了一篇讨论认知科学中动力学假设（dynamical hypothesis）的学术论文，指出其与 AI 认知的潜在交集。
   - 讨论暗示了将认知过程表达为计算函数的哲学意义。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/ZeyuanAllenZhu/status/1829326495757853005?t=VibYJ-3VXqmPmp9QWPYqSA&s=19">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>：(1/7) Physics of LM, Part 2.2，包含关于“LLM 如何从错误中学习”的 8 个结果，现已发布在 arxiv：https://arxiv.org/abs/2408.16293。我们探索了使模型能够纠正错误的可能性...</li><li><a href="https://x.com/_xjdr/status/1832083976225513715">来自 xjdr (@_xjdr) 的推文</a>：使用仓库中的 system prompt，1 个格式化的 ICL 示例，top_p .95 和 temp 0.7（按推荐设置），并预填充响应为 "&lt;thinking&gt;\n"，看起来像是...</li><li><a href="https://x.com/soldni/status/1831857907291582552?t=HGy1Dj4Mwb1HZoezweFZfg&s=19">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>：图 1：shot，图 2：chaser，图 3：无端而起</li><li><a href="https://x.com/mattshumer_/status/1831768677605155174">来自 Matt Shumer (@mattshumer_) 的推文</a>：@abacaj 并不完全是——我们发现目前的模型很难做好这一点（它们不知道什么时候该进行反思）。这需要通过一个故意制造错误的训练数据集将其训练到模型中 -&gt;...</li><li><a href="https://x.com/vllm_project/status/1831742284804866237?s=46">来自 vLLM (@vllm_project) 的推文</a>：一个月前，我们公布了性能路线图。今天，我们很高兴地分享，最新版本在 Llama 8B 上实现了 🚀2.7 倍的吞吐量提升，输出延迟快了 5 倍，并且在...实现了 1.8 倍...</li><li><a href="https://x.com/mattshumer_/status/1831826171107144090?t=k5R0qg02Qr5azpPjQtfgaw&s=19">来自 Matt Shumer (@mattshumer_) 的推文</a>：@EnricoShippole @binary_racoon @GlaiveAI 不同的反思——只是不想引起任何混淆，我们正在做完全不同的事情</li><li><a href="https://huggingface.co/matts">matts (Matt Szydlik)</a>：未找到描述</li><li><a href="https://huggingface.co/leafspark/Reflection-Llama-3.1-70B-GGUF">leafspark/Reflection-Llama-3.1-70B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://openreview.net/forum?id=xaqoZZqkPU">Reflection-Tuning: Recycling Data for Better Instruction-Tuning</a>：Large Language Models (LLMs) 的最新进展扩展了自然语言理解和生成的视野。值得注意的是，LLM 的输出控制和与输入的对齐可以...</li><li><a href="https://x.com/mattshumer_/status/1831767017507954808/photo/1">来自 Matt Shumer (@mattshumer_) 的推文</a>：Reflection 70B 甚至能与顶尖的闭源模型（Claude 3.5 Sonnet, GPT-4o）抗衡。它是（至少在）MMLU, MATH, IFEval, GSM8K 中表现最顶尖的 LLM。在所有测试的基准测试中都击败了 GPT-4o。...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：在验证了所需的设置（使用 system prompt，无预填充）后，我可以肯定地说，Reflection 至少在 BigCodeBench-Hard 上表现不佳。Complete: 20.3（对比 Llama3.1-70B 的 28.4）Instruc...</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=ldUBdhhdmxU0qMgsmVaTUg&s=19">来自 Matt Shumer (@mattshumer_) 的推文</a>：我很高兴地宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种旨在让 LLM 能够修复自身错误的技术。405B 将于下周推出...</li><li><a href="https://openrouter.ai/models/mattshumer/reflection-70b">Reflection 70B - API, 提供商, 统计数据</a>：Reflection Llama-3.1 70B 采用一种名为 Reflection-Tuning 的新技术训练，该技术教导 LLM 检测其推理中的错误并纠正方向。通过 API 运行 Reflection 70B</li><li><a href="https://github.com/tianyi-lab/Reflection_Tuning">GitHub - tianyi-lab/Reflection_Tuning: [ACL'24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning</a>：[ACL'24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning - tianyi-lab/Reflection_Tuning</li><li><a href="https://sci-hub.scrongyao.com/10.1017/S0140525X98001733">Sci-Hub | The dynamical hypothesis in cognitive science | 10.1017/S0140525X98001733</a>：未找到描述
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1281695101952987168)** (1 条消息): 

> - `DeepSeek v2.5`
> - `Coding improvements` 


- **关于 DeepSeek v2.5 性能的询问**：一位成员请求用户报告在使用 **DeepSeek v2.5** 进行编程任务（coding tasks）时是否有任何明显的改进。
   - *请分享您的经验和见解！*
- **对用户反馈的期待**：社区期待用户关于 **DeepSeek v2.5** 效能的反馈，特别是关于编程增强（coding enhancements）方面。
   - 鼓励成员们贡献他们的发现，以促进集体学习。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

teknium: https://x.com/alexandr_wang/status/1832147956562284987?s=46
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1281329043186122803)** (52 条消息🔥): 

> - `OpenAI's $2000 Subscription Model`
> - `Reflection 70B Model Performances`
> - `Speculative Decoding in Inference`
> - `New Text-to-Music Models`
> - `AI Scientist Testing Challenges` 


- **OpenAI 考虑推出 $2000 的 ChatGPT 订阅模式**：一场关于 OpenAI 考虑为其更先进的 AI 模型（包括预期的 Orion 模型）推出定价为 **$2000/月** 的订阅模式的讨论正在进行中。
   - 这种定价的影响及其合理性仍然是社区成员关注的热点，大家对可访问性（accessibility）表示了担忧。
- **Reflection 70B 模型受到审查**：**Reflection 70B** 模型的测试显示其与 Llama3 相比结果参差不齐，在 **BigCodeBench-Hard** 和 **Aider** 等代码基准测试中表现较低。
   - 批评者认为性能差异源于模型的方法论，在完全依赖其指标之前需要进行更彻底的检查。
- **Speculative Decoding 有望提升性能**：Together AI 分享的研究发现，**Speculative Decoding** 可以将长上下文输入的延迟和吞吐量提高多达 **2倍**，这与之前对其有效性的假设相反。
   - 这一进展标志着如何利用现有框架优化高吞吐量推理（high-throughput inference）的重大转变。
- **Text-to-Music 模型的新进展**：一个新的开源 **text-to-music 模型** 已发布，与 **Suno.ai** 等现有解决方案相比，展示了令人印象深刻的音质和效率。
   - 尽管在实际场景中的比较质量和可用性方面存在不同意见，但开发者对其应用潜力表示兴奋。
- **AI Scientist 测试面临的挑战**：有人询问由于 PyTorch 兼容性问题，在与 Apple Silicon 兼容的模型上测试 **Sakana AI Scientist** 的情况。
   - 讨论表明了对模型有效性的担忧，成员们敦促进一步调查性能和潜在的改进。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/mattshumer_/status/1831767031735374222?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Matt Shumer (@mattshumer_) 的推文</a>：最重要的是，向 @csahil28 和 GlaiveAI 致以巨大的谢意。我已经琢磨这个想法好几个月了，终于在几周前决定付诸行动。我联系了 Sahil 和数据 ...</li><li><a href="https://x.com/mjlbach/status/1831323536788791595?s=46">来自 Michael Lingelbach (@mjlbach) 的推文</a>：看起来一个 SOTA 开源文本转音乐模型（一个 rectified flow DiT）发布了。论文地址：https://arxiv.org/abs/2409.00587 代码地址：https://github.com/feizc/FluxMusic 示例听起来非常...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：在验证了所需的设置（包含系统提示词，无预填）后，我可以有把握地说，Reflection 至少在 BigCodeBench-Hard 上表现不佳。完成度：20.3（相比之下 Llama3.1-70B 为 28.4）指令...</li><li><a href="https://x.com/maximelabonne/status/1832036357734109496?s=46">来自 Maxime Labonne (@maximelabonne) 的推文</a>：这超级酷，但我有很多疑问。首先，Reflection = 强化版 CoT。这意味着你根本无法比较这些分数。还记得人们嘲笑 Gemini 提供 CoT 响应的时候吗...</li><li><a href="https://x.com/Techmeme/status/1831696947914404181">来自 Techmeme (@Techmeme) 的推文</a>：OpenAI 表示，其 ChatGPT 企业版（包括 ChatGPT Team、Enterprise 和 Edu）目前拥有超过 100 万付费用户（@rachelmetz / Bloomberg）https://www.bloomberg.com/news/articles/2024-09-05/o...</li><li><a href="https://x.com/ikristoph/status/1831803754678767875?s=46">来自 Kristoph (@ikristoph) 的推文</a>：我绝不想对 @mattshumer_ 及其团队的工作持否定态度。这些家伙做了一些非常棒的工作，我非常期待尝试他们的模型，你们也应该 ...</li><li><a href="https://x.com/togethercompute/status/1831755763615674412">来自 Together AI (@togethercompute) 的推文</a>：我们很高兴分享关于高吞吐量推理的推测解码（speculative decoding）的最新工作！在这项工作之前，我们认为推测解码在大批量（batch sizes）情况下是无用的，因为 GPU 会 ...</li><li><a href="https://x.com/aiexplainedyt/status/1831710902636228694?s=46">来自 AI Explained (@AIExplainedYT) 的推文</a>：你会为 ChatGPT 每月支付 2000 美元吗？根据《The Information》刚刚发布的关于 OpenAI 的报告，这是订阅费中“摆在桌面上的”最高价格。这将是...</li><li><a href="https://x.com/bindureddy/status/1831746158752088178">来自 Bindu Reddy (@bindureddy) 的推文</a>：OpenAI 正在考虑收取每月 2000 美元来访问他们的顶级模型。撇开玩笑不谈，这将是一个 Vision-Pro 级别的灾难。我希望这只是个玩笑</li><li><a href="https://x.com/cocktailpeanut/status/1831753703940092016">来自 cocktail peanut (@cocktailpeanut) 的推文</a>：很多人让我为此写一个一键启动器。幸运的是，已经有其他人浪费了他们的时间和精力去尝试了。听听看并自行判断。我想我还是会坚持...</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>：我很高兴宣布 Reflection 70B，全球顶级的开源模型。使用 Reflection-Tuning 训练，这是一种旨在让 LLM 能够修复自身错误的技术。405B 将于下周推出 ...</li><li><a href="https://www.melodio.ai/">Melodio AI | Vibe Your Moment - 官方网站</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1831701773721203164?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：就像 Q* 一样，OpenAI 的 Strawberry 系统泄露的内容已经足够让我们对其训练设置和用例产生实质性的有趣假设。一些想法：* 作为推理的自我对话（Self talk）...</li><li><a href="https://x.com/swyx/status/1832138164951249104">来自 swyx.io (@swyx) 的推文</a>：Diffusion Transformers 非常棒，但在我们等待 Sora 的同时，我喜欢 @toinfinityai 的方法——将使用场景严格限制在视频同步（不仅是唇形同步）上——并以此为起点。B...</li><li><a href="https://x.com/zbeyens/status/1832079140083687671?s=46">来自 Ziad Beyens (@zbeyens) 的推文</a>：介绍 AI Codex：用于 @cursor_ai 的自改进系统。◆ http://codex.md：错误和学习库。◆ http://learn.md：自动保存新见解。◆ http://split-codex.md：智能分类...</li><li><a href="https://x.com/krandiash/status/1832056408205935060?s=46">来自 Karan Goel (@krandiash) 的推文</a>：2.5 个月前，@elevenlabsio 发布了与我们仅发布 10 天的 Sonic 模型的对比：https://elevenlabs.io/blog/elevenlabs-vs-cartesia。团队将其视为挑战，这是我们新的成绩单。 ...</li><li><a href="https://x.com/ericsmith1302/status/1831745370822516792?s=46">来自 Eric Smith (@ericsmith1302) 的推文</a></li>

<li><a href="https://x.com/t_m_p_r/status/1831713583547494576">推文</a>: 在 $90K MRR 的情况下，这是运营我的个人 AI 初创公司 (AutoShorts) 的成本 👇 我想在这里保持完全透明，因为我经常发布关于收入的内容，但很少提到另一面。我还没有计算...</li><li><a href="https://x.com/togethercompute/status/1831783919718690877?s=46">来自 Together AI (@togethercompute) 的推文</a>: 🚀 NVIDIA H200 和 Together Kernel Collection (TKC) 即将登陆 Together GPU Clusters：为 AI 训练、微调和推理提供加速的性能、效率和可扩展性...</li><li><a href="https://podcasts.apple.com/us/podcast/minus-one/id1759014294?i=1000668457399">Reid Hoffman 与 AI-Native 未来</a>: 连续创业者和多产投资者 Reid Hoffman 分享了从 PayPal 黑手党到创立 LinkedIn 再到投资 AI-native 的曲折历程...</li><li><a href="https://x.com/paulgauthier/status/1832160129720185225">来自 Paul Gauthier (@paulgauthier) 的推文</a>: Reflection 70B 在 aider 代码编辑基准测试中得分 42%，远低于 Llama3 70B 的 49%。我修改了 aider 以忽略 `<thinking/reflection>` 标签。该模型无法正常工作...</li><li><a href="https://www.oranlooney.com/post/gpt-cnn/">一图值 170 Tokens：GPT-4o 如何编码图像？ - OranLooney.com</a>: 事实是：GPT-4o 在高分辨率模式下处理每个 512x512 的切片（tile）收费 170 tokens。按约 0.75 tokens/单词计算，这意味着一张图片大约相当于 227 个单词——仅为四倍...</li><li><a href="https://github.com/udecode/dotai/blob/main/codex/learn.md">dotai/codex/learn.md at main · udecode/dotai</a>: 通过在 GitHub 上创建账号来为 udecode/dotai 的开发做出贡献。</li><li><a href="https://github.com/SakanaAI/AI-Scientist/tree/main">GitHub - SakanaAI/AI-Scientist: The AI Scientist: 迈向全自动的开放式科学发现 🧑‍🔬</a>: The AI Scientist: 迈向全自动的开放式科学发现 🧑‍🔬 - SakanaAI/AI-Scientist</li><li><a href="https://x.com/">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://blog.vllm.ai/2024/09/05/perf-update.html">vLLM v0.6.0: 吞吐量提升 2.7 倍，延迟降低 5 倍</a>: TL;DR: vLLM 在 Llama 8B 模型上实现了 2.7 倍的吞吐量提升和 5 倍的 TPOT（每个输出 token 的时间）加速，在 Llama 70B 模型上实现了 1.8 倍的吞吐量提升和 2 倍的 TPOT 加速。</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5745/">[AINews] SciCode: HumanEval 获得 STEM 博士级升级</a>: 博士级基准测试就是你所需要的一切。2024/7/15-2024/7/16 的 AI News。我们检查了 7 个 subreddits，384 个 Twitter 账号和 29 个 Discord 社区（466 个频道，2228 条消息...）
</li>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1281705587557666910)** (76 条消息🔥🔥): 

> - `AI Code Editors`
> - `Handling Errors in Engineering`
> - `Tools for Code Automation`
> - `Collaboration with AI`
> - `Fine-tuning Models` 


- **探索 AI Code Editors**：成员们对 [Melty](https://github.com/meltylabs/melty) 和 [Pear AI](https://github.com/trypear/pearai-app) 等各种 AI Code Editors 表现出浓厚兴趣，将其作为 Cursor 的替代方案，并讨论了它们的独特功能。
   - 大家对功能和易用性感到好奇，特别是关于 Cursor 中注释和 TODO 行被删除的问题。
- **超越 Happy Paths 的工程实践**：讨论指出，有效的软件工程需要处理 edge cases，一位成员提到他们的 happy path 代码仅占总工作量的 10% 左右。
   - 这引发了关于 Aider 等工具的讨论，该工具能协助高效地编辑代码。
- **AI 开发中的协作工具**：Zed AI 被强调为一款用于高性能协作的强大 Code Editor，成员们指出了它对与 AI 协作的开发者的潜在益处。
   - 然而，有人指出它目前缺乏 bitmap font 支持，限制了其对某些用户的适用性。
- **关于 LLMs 的后续话题**：未来的会议计划涵盖使用 Loras 或 quantization 技术进行 Fine-tuning 的技巧，显示了对高级 AI 话题的参与度。
   - 成员们就此类任务的复杂细节和涉及的模型交换了意见。
- **AI 开发中的错误处理**：成员们讨论了编码中错误处理的重要性，处理“non-happy-path”场景是工程与简单原型设计的区别所在。
   - 大家还分享了对有助于错误管理的工具的了解，强调了对健壮解决方案的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://zed.dev/">Zed - The editor for what&#x27;s next</a>: Zed 是一款由 Atom 和 Tree-sitter 的创作者开发的高性能、多用户协作 Code Editor。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/catter">go-go-labs/cmd/apps/catter at main · go-go-golems/go-go-labs</a>: GO GO 实验性实验室。通过在 GitHub 上创建账户，为 go-go-golems/go-go-labs 的开发做出贡献。</li><li><a href="https://github.com/MikeBirdTech/ai-toolkit">GitHub - MikeBirdTech/ai-toolkit: A collection of community created AI tools to improve your life</a>: 社区创建的 AI 工具集合，旨在改善你的生活 - MikeBirdTech/ai-toolkit</li><li><a href="https://github.com/trypear/pearai-app">GitHub - trypear/pearai-app: The Open Source AI-Powered Code Editor. A fork of VSCode and Continue.</a>: 开源 AI 驱动的 Code Editor。VSCode 和 Continue 的 fork 版本。 - trypear/pearai-app</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 暂无描述</li><li><a href="https://github.com/meltylabs/melty">GitHub - meltylabs/melty: Open source AI code editor. To download the packaged app:</a>: 开源 AI Code Editor。下载打包好的应用： - meltylabs/melty
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1281329809754161202)** (80 条消息🔥🔥): 

> - `Perplexity 使用情况`
> - `RunwayML 争议`
> - `Reflection 模型测试`
> - `Luma Dream Machine 偏好`
> - `OpenAI tokens 可用性` 


- **Perplexity 因其效率受到赞赏**：用户强调他们更倾向于使用 **Perplexity**，因为它能以可用的格式最快地提供可靠信息，一些人甚至考虑将他们的 **ChatGPT Plus** 订阅切换到该平台。
   - 一位用户强调它在学校表现良好，因为没有被屏蔽，而且 **Arc browser** 已将其集成，使其成为一个极好的 AI 搜索引擎。
- **对 RunwayML 客户服务的紧张情绪上升**：一位用户分享了在 **RunwayML** 的糟糕经历，描述了计划中的社区见面会在没有任何解释的情况下被突然取消，表达了对他们客户服务的不满。
   - 这一事件引发了人们对 Runway 对其社区响应能力的担忧，特别是考虑到其付费会员的忠诚度以及对其声誉的潜在影响。
- **测试 Reflection 模型**：讨论围绕 **Reflection Llama-3.1 70B 模型**展开，用户对其性能和名为 Reflection-Tuning 的新训练技术表示关注，该技术可以纠正推理错误。
   - 一位用户链接到了一个感兴趣的人可以尝试该模型的平台，并指出在初始测试出现问题后已进行了改进。
- **Luma Dream Machine 提供具有竞争力的方案**：成员们将 **Luma Dream Machine** 与其他产品进行了比较，赞赏其方案的灵活性，价格从免费到每月 399 美元不等，推荐的每月 29.99 美元的方案适合大多数用户。
   - 讨论了该服务的增长潜力，成员们也热衷于探索其功能。
- **OpenAI tokens 正在赠送**：一位用户免费提供 **OpenAI tokens**，表示他们有 1,000 个 tokens 可用但不打算使用。
   - 这引起了频道成员的兴趣，暗示了社区内可能进行 tokens 交换或使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://reflection-playground-production.up.railway.app">Reflection 70B Playground</a>：未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-ver">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://x.com/cjnnngs/status/1832074294203199936">来自 Cole (@cjnnngs) 的推文</a>：@runwayml 清除 Reddit 和 Discord 上关于你们公司的任何真实反馈并不是一个好的做法。所以，我会把它发在这里，这样你们就无法介入并审查它。包含截图以示完全透明...</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-version-of-chatgpt">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>：未找到描述</li><li><a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>：Dream Machine 是一个 AI 模型，可以利用来自 Luma AI 的文本和图像快速生成高质量、逼真的视频</li><li><a href="https://huggingface.co/spaces/featherless-ai/try-this-model">HF's Missing Inference Widget - 由 featherless-ai 提供的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1281431001452646411)** (10 条消息🔥): 

> - `速率限制问题`
> - `自定义 GPT 共享问题`
> - `浏览器兼容性` 


- **ChatGPT 速率限制困惑依然存在**：一位 Plus 用户报告称，尽管使用量极少并切换到了 4o mini，但仍持续收到“超出速率限制”的消息。此问题促使人们建议向 [OpenAI](https://help.openai.com/) 寻求帮助。
   - 该用户对付费服务却受到限制表示沮丧，*“我已经超过 12 小时没用过 ChatGPT 了……”*。
- **共享自定义 GPT 的问题**：几位用户讨论了在保存更改和共享其自定义 GPT 时遇到的困难，表明存在波动的访问问题。一位用户指出，删除文件后可以共享，但添加任何新内容后又恢复原状，导致显示“待更新”状态。
   - 用户担心这个故障可能会阻碍功能，希望在未来的更新中得到修复，正如 *“也许他们会在下次更新中研究修复方案”* 所指出的。
- **浏览器兼容性引发疑问**：一位用户提到在 Firefox 上遇到了同样的问题，同时在 Chrome 移动端进行了测试。这引发了关于问题不仅限于浏览器的猜测。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1281374205916479640)** (10 条消息🔥): 

> - `集成 Tool Calls`
> - `Prompt 库位置`
> - `创意 Prompt 用法` 


- **Tool Calls 的成功实践**：一位成员询问如何成功地将 Tool Calls 集成到 Prompt 中，并对由于结构错误导致的错误消息表示沮丧。
   - 另一位成员分享了他们在单次输出中使用超过十个 Python Tool Calls 创建工具链（Tool Chains）的成功经验，并强调了使用正确工具名称的重要性。
- **Tool Call 结构示例**：在一番努力后，一位成员报告称找到了包含 Tool 结果且匹配正确 ID 的结构：先是一个 Assistant 消息，紧接着是一个 Tool 消息。
   - 这突显了在工具交互中需要细致关注 ID 对齐。
- **访问 Prompt 库**：一位成员询问 Prompt 库的位置，另一位成员迅速告知现在它被称为 <#1019652163640762428>。
   - 这展示了社区在平台导航方面提供协助的意愿。
- **独特的 Prompt 发现**：一位成员分享了一个奇特的 Prompt 想法，涉及将 Buffer 的全部内容逐字写入代码块。
   - 这展示了社区在探索 Prompt 不同用法方面的创造力。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1281374205916479640)** (10 条消息🔥): 

> - `集成 Tool Calls`
> - `Prompt 库位置`
> - `Buffer 内容 Prompt` 


- **在 Prompt 中成功使用 Tool Calls**：一位成员对在 Prompt 中集成 Tool Calls 表示沮丧，提到他们收到了来自 OpenAI 的简单错误消息。
   - 另一位成员声称他们成功地在单次输出中使用超过 **10 个 Python Tool Calls** 创建了 **工具链（Tool Chains）**。
- **详解正确的工具结构**：一位成员分享了如何正确构建 Tool Calls 结构，强调需要在带有内容的 **Assistant 消息**之后紧跟一个对应的 **Tool 消息**来返回结果。
   - 他们意识到自己的错误在于某次 Tool Call 之后忘记添加工具结果。
- **寻找 Prompt 库**：一位成员询问 Prompt 库的位置，寻求寻找路径的指导。
   - 回复指出 Prompt 库现在被称为 <#1019652163640762428>。
- **发现有趣的 Prompt**：一位成员分享了他们遇到的一个有趣的 Prompt，该指令要求逐字输出 Buffer 的全部内容。
   - 这个 Prompt 突显了从之前的对话中捕获完整上下文和指令的能力。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1281330212352561343)** (97 条消息🔥🔥): 

> - `Academic Lab Opportunities` (学术实验室机会)
> - `Universal Transformers`
> - `Recurrence in Neural Networks` (神经网络中的递归)
> - `Computational Resource Challenges` (计算资源挑战)
> - `Independence in Research` (研究独立性)


- **探索学术实验室机会**：成员们讨论了在学术实验室获得职位的复杂性，指出虽然存在实习计划，但冷邮件（cold emailing）是另一种成功率较低的选择。
   - 有人建议撰写一份项目提案向实验室推销，强调了展示研究成果的重要性，特别是如果研究方向与当前趋势一致的话。
- **审视 Universal Transformers**：对话探讨了 Universal Transformers (UTs) 的可行性，一名成员表达了对这一小众领域的个人痴迷，尽管其他人对其未来的实用性持怀疑态度。
   - 他们还强调了关于 UTs 中自适应隐式计算（adaptive implicit compute）的讨论，这可能会提升性能，但稳定性仍是实现过程中的一个重大障碍。
- **研究中的资源分配**：成员们对学术界和研究实验室的资源分配表示担忧，特别是计算资源的可用性往往倾向于以产品为导向的项目，而非非常规研究。
   - 成员们反思了在 DeepMind 等机构中，资历以及与流行研究兴趣的一致性如何影响个人的自由度和可用资源。
- **美欧资助文化的差异**：一名成员指出美国和欧洲机构在学术文化上的显著差异，强调欧洲的资助往往更加宽松。
   - 尽管学术界被认为拥有自由，但“不发表就发臭”（publish or perish）的文化会迫使研究人员顺应热门话题，使追求小众领域变得更加复杂。
- **模型中递归的挑战**：讨论涉及了递归模型，重点是 Deep Equilibrium Models (DEQs) 及其与传统 RNNs 和 State Space Models 的比较。
   - 虽然一些成员对递归研究表现出热情，但其他人对这种方法的未来表示怀疑，进一步证实了其小众地位。



**提到的链接**：<a href="https://arxiv.org/abs/1807.03819">Universal Transformers</a>：循环神经网络 (RNNs) 通过随每个新数据点更新其状态来顺序处理数据，长期以来一直是序列建模任务的事实选择。然而，它们固有的...

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1281436023691415635)** (5 messages): 

> - `Momentum-based Optimizers` (基于动量的优化器)
> - `Reinforcement Learning Automation` (强化学习自动化)
> - `Gradient Cosine Similarity` (梯度余弦相似度)
> - `Consecutive Gradient Analysis` (连续梯度分析)


- **AdEMAMix 优化器增强梯度利用率**：针对 Adam 优化器提出的一种改进方案 **AdEMAMix**，利用两个指数移动平均（EMA）的混合，比单一 EMA 更好地优化了对过去梯度的处理 [PDF](https://arxiv.org/pdf/2409.03137)。
   - 该方法旨在更有效地平衡近期梯度与旧梯度的权重，在语言建模和图像分类中显示出良好的效果。
- **自动化强化学习 Agent 架构**：一种新的 Agent 架构使强化学习工作流的各个环节实现自动化，使其能够独立管理实验进度，并使用视觉语言模型（VLM）构建课程 [PDF](https://arxiv.org/pdf/2409.03402)。
   - 该系统将任务分解为子任务并检索技能，标志着全自动强化学习过程的首批实现之一。
- **梯度余弦相似度见解**：连续梯度的**余弦相似度**表明训练数据集中存在循环模式，这与相同梯度符号的百分比相关，并指示了潜在的序列结构。
   - 这种相关性暗示在某些数据集条件下，梯度可能会越来越多地指向相似的方向。
- **梯度与损失导数之间的线性关系**：一名成员指出，连续梯度的**余弦相似度**在训练过程中似乎与损失函数的导数呈现线性关系。
   - 这一观察表明梯度行为与损失指标趋势之间存在更深层的联系。
- **模型训练的见解与资源**：分享了 **Distily Attn MLP Sweep** 的 [Model Card](https://huggingface.co/distily/distily_attn_mlp_sweep) 链接，以及 [训练指标 (Training Metrics)](https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard) 和 [社区讨论 (Community Discussions)](https://huggingface.co/distily/distily_attn_mlp_sweep/discussions) 的访问入口。
   - 这些资源提供了关于该 Sweep 的模型性能和社区互动的全面概览。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard">distily/distily_attn_mlp_sweep · 训练指标</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.03402">Game On: Towards Language Models as RL Experimenters</a>：我们提出了一种 Agent 架构，可自动执行常见的强化学习实验工作流，从而实现具身 Agent 对控制领域的自动化掌握。为此，它利用了...</li><li><a href="https://arxiv.org/abs/2409.03137">The AdEMAMix Optimizer: Better, Faster, Older</a>：基于动量的优化器是各种机器学习应用的核心。这些优化器通常依赖于梯度的指数移动平均（EMA），它会随时间对当前梯度进行指数级衰减...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1281623602680037457)** (2 messages): 

> - `Reusing Model Outputs` (复用模型输出)
> - `lm-evaluation-harness` 


- **关于在 Benchmark 中复用模型输出的咨询**：一名成员询问在数据集重合的情况下，是否可以为多个 Benchmark 复用模型输出，并强调了对效率的关注。
   - 这提出了关于如何跨不同评估有效共享输出以节省时间和资源的重要问题。
- **分享 lm-evaluation-harness GitHub 资源**：一名成员分享了 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#saving-results) 的链接，这是一个用于语言模型 Few-shot 评估的框架。
   - 该资源可能为如何跨各种 Benchmark 优化模型结果管理提供有用的见解。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#saving-results">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>：一个用于语言模型 Few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1281375430204457012)** (2 条消息): 

> - `Hugging Face RoPE 实现兼容性`
> - `训练模型 1 个 Epoch` 


- **GPTNeoX 中的 Hugging Face RoPE 兼容性**：一位成员询问了 **GPTNeoX/Pythia** 的 **Hugging Face RoPE 实现**与 **Llama/GPT-Fast** 所使用的实现之间的兼容性。
   - 他们观察到，在他们的实现与 **Pythia 模型**之间，来自 **scale_dot_product_attention** 函数的 Attention 输出存在显著差异（超过 **95%**）。
- **仅运行模型一个 Epoch**：另一位成员询问是否可以仅运行模型 **1 个 Epoch**，还是需要手动计算 `train_iters`。
   - 他们推测 `train_iters` 可以通过 **num_data_sequences/(batch_size * number_of_ddp_processes)** 来计算。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1281331848068468818)** (74 条消息🔥🔥): 

> - `Open Interpreter 周年庆典`
> - `OI 中的 Skills 功能`
> - `01 app 性能反馈`
> - `Fulcra app 可用性`
> - `OI 的 Beta 测试` 


- **Open Interpreter 庆祝里程碑**：成员们热烈庆祝 Open Interpreter 的生日，并对它在 AI 与人类交互方面的潜力表示兴奋。
   - *Happy Birthday, Open Interpreter!* 是反复出现的情感表达，展示了社区对这一创新的赞赏。
- **Open Interpreter 中的 Skills 仍处于实验阶段**：讨论强调 OI 中的 Skills 功能是实验性的，用户询问了 Skills 在不同会话间的持久性。
   - 一位用户指出 Skills 似乎是暂时的，并建议检查机器上的 Skills 存储位置。
- **对 01 app 性能的正面反馈**：用户对 01 app 的性能印象深刻，其中一位提到它能高效地从 2,000 个音频文件中搜索并播放歌曲。
   - 也有人提到结果存在一些不一致，这反映了早期访问阶段 app 的典型体验。
- **Fulcra app 扩展至新地区**：根据社区要求，Fulcra app 已在多个新地区上线，增强了其可访问性。
   - 用户询问了在澳大利亚的可用性，表明了进一步扩大覆盖范围的兴趣。
- **Open Interpreter 的 Beta 测试机会**：社区成员表达了参与 Beta 测试的兴趣，并确认目前仍有机会。
   - 对早期访问测试的热情反映了支持性强且参与度高的用户群体。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://play.google.com/store/apps/details?id=com.interpreter.app">01 Light - Google Play 应用</a>: 未找到描述</li><li><a href="https://apps.apple.com/ca/app/01-light/id6601937732">‎01 Light</a>: ‎通过来自任何地方的语音命令控制您的计算机和智能家居。01 连接到您家中心机上的服务器，实现对文件、应用程序和 IoT 设备的远程访问。能力：...</li><li><a href="https://apps.apple.com/us/app/context-by-fulcra/id1633037434">‎Context by Fulcra</a>: ‎Context by Fulcra Dynamics 是收集您生活产生的所有数据的可靠平台。将您的健康指标、日历事件、位置和其他上下文数据从孤岛中解放出来，整合到您的...</li><li><a href="https://tenor.com/view/youre-a-wizard-hagrid-afirmation-magic-magical-gif-16533730">Youre A Wizard Hagrid GIF - Youre A Wizard Hagrid Afirmation - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/frankenstein-its-alive-happy-excited-gif-5625959">Frankenstein Its Alive GIF - Frankenstein Its Alive Happy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ooh-despicable-me-4-surprised-uh-oh-that%27s-gotta-hurt-gif-14253073070740964952">Ooh Despicable Me 4 GIF - Ooh Despicable me 4 Surprised - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言界面</a>: 计算机的自然语言界面。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/dbc52593e608d3ce3d25a0eece4e84cf57bb7892/interpreter/core/computer/skills/skills.py">open-interpreter/interpreter/core/computer/skills/skills.py at dbc52593e608d3ce3d25a0eece4e84cf57bb7892 · OpenInterpreter/open-interpreter</a>: 计算机的自然语言界面。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01/pull/300/files#diff-1c7d7d67cce10f3be88bac85f3231198881bf48beb40f43cfe27015c6c9b53cd">MikeBirdTech 的文档编辑 · Pull Request #300 · OpenInterpreter/01</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: 适用于桌面、移动设备和 ESP32 芯片的排名第一的开源语音界面。</a>: 适用于桌面、移动设备和 ESP32 芯片的排名第一的开源语音界面。 - OpenInterpreter/01
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1281415680100536432)** (8 messages🔥): 

> - `Beta role for desktop`（桌面端 Beta 角色）
> - `Open Interpreter 01 issues`（Open Interpreter 01 问题）
> - `Audio device inquiry`（音频设备咨询）


- **申请 Beta 角色访问权限**：多位用户表达了希望获得 **桌面端 beta 角色** 访问权限的愿望，其中包括一位曾参与 **Open Interpreter 01** 开发套件（dev kit）工作的粉丝。
   - 一位用户提到：*“没能参加直播——有什么办法可以获得桌面端 beta 角色吗？”*。
- **在 M1 Mac 上运行 01 的问题**：一位使用 **M1 Mac** 的成员报告了运行 **Open Interpreter 01** 时遇到的问题，提到了 **torch** 错误和环境冲突。
   - 他们寻求帮助，询问是否有专家愿意进行实时排查，并表示：*“如果你有兴趣请私信我。”*。
- **关于音频设备的咨询**：在对会议给出正面评价后，一位用户询问演示期间是否提到了 **01 音频设备**。
   - 这表明了对所讨论技术的浓厚兴趣。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1281464644912939040)** (13 messages🔥): 

> - `404 on values page`（values 页面 404 错误）
> - `Integration of C and Mojo`（C 与 Mojo 的集成）
> - `Company culture link update`（公司文化链接更新）


- **Values 页面 404 错误**：成员们讨论了 Modular 网站上的 values 页面目前在[此链接](https://www.modular.com/values)返回 **404 错误**。有人建议该链接可能需要指向[公司文化（company culture）](https://www.modular.com/company/culture)。
- **C 与 Mojo 集成变得简单**：一位成员询问如何将 **C** 与 **Mojo** 集成，另一位成员确认可以使用 `DLHandle` 动态链接到 `.so` 文件。
   - 提供了一个示例：`handle = DLHandle('path/to/mylib.so')`，随后调用 C 库中的 `is_even` 函数。
- **公司文化链接位置**：一位用户询问公司文化链接是在哪里找到的，另一位用户指出它位于招聘帖子的“核心公司文化价值观”部分。
   - 这一说法得到了另一位成员的感谢确认。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/sys/ffi/DLHandle">DLHandle | Modular Docs</a>：表示可以加载和卸载的动态链接库。</li><li><a href="https://www.modular.com/company/culture">Modular: Our Culture</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们的三大支柱是：打造用户喜爱的产品、赋能员工以及成为一支不可思议的团队。</li><li><a href="https://www.modular.com/company/career-post?4450311005&gh_jid=4450311005">Modular: Career Post</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们的三大支柱是：打造用户喜爱的产品、赋能员工以及成为一支不可思议的团队。</li><li><a href="https://www.modular.com/values">Modular: Our Culture</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们的三大支柱是：打造用户喜爱的产品、赋能员工以及成为一支不可思议的团队。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1281328055549427874)** (68 messages🔥🔥): 

> - `Mojo async 功能`
> - `将 DType 用作 Dict 键`
> - `构造函数用法的改进`
> - `pop.array 的封装器`
> - `Mojo 中的 MLIR 和 IR 生成` 


- **Mojo 异步函数的困惑**：一位用户报告了在使用 `async fn` 和 `async def` 时遇到的问题，指出这些尝试在 Mojo 的稳定版本中无法运行。
   - 官方澄清异步功能仅在 nightly 版本中可用，并建议检查所使用的版本。
- **DType 无法用作 Dict 键**：一位用户质疑为什么 `DType` 不能在 Dictionary 中用作键，尽管它实现了 `KeyElement` trait。
   - 这一问题引发了关于 Mojo 数据结构中类型约束和用法的讨论。
- **构造函数用法的增强**：一位用户分享了在解决与 `Arc[T, True]` 和 `Weak[T]` 相关的构造函数问题方面的进展，强调了 `@parameter` guards 的复杂性。
   - 建议在标准库中保持命名一致性，并改进类型结构以提高清晰度。
- **pop.array 封装器的见解**：一位成员讨论了为用于可选字段的 `pop.array` 创建封装器，并透露在查找实现方面存在一些困难。
   - 进一步记录了关于优化数据结构内指针间接寻址（pointer indirection）以增强可用性的笔记。
- **关于 MLIR 和 IR 生成的讨论**：多位用户对如何在 Mojo 中更有效地利用 MLIR 表示了兴趣，特别是关于 IR 生成及其优势。
   - 推荐了一个来自 LLVM 会议的视频，作为深入了解 Mojo 与 MLIR 及 LLVM 交互的宝贵资源。



**提到的链接**：<a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>：2023 LLVM 开发者大会，演讲者：Abdul Dakkak, Chr...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1281369655499292672)** (4 messages): 

> - `Reflection 70B 模型`
> - `Reflection Tuning 技术`
> - `Together 的自定义内核性能` 


- **Reflection 70B 模型发布公告**：一项激动人心的公告披露了 **Reflection 70B** 的发布，声称它是全球顶尖的开源模型，使用了 [Reflection-Tuning](https://x.com/mattshumer_/status/1831767014341538166) 技术，使 LLM 能够纠正自己的错误。
   - 预计下周将推出 **405B 模型**，其性能可能超越市场上所有现有模型。
- **解释 Reflection Tuning**：讨论围绕 **Reflection-Tuning** 展开，声称它在输出中集成了 `<thought>` 和 `<reflection>` 标签，用于思维链（CoT）和自我反思，如这个[长加法示例](https://x.com/johnubalis/status/1831792041438949833)所示。
   - 据推测，合成训练数据（可能使用 STaR 生成）在训练过程中起到了至关重要。
- **Together GPU 集群的性能提升**：针对 Together 发布的新型 **速度提升 20% 的 MLP 内核** 提出了疑问，该内核承诺显著提高 AI 操作速度，声称与标准实现相比，训练速度提升高达 24%，FP8 推理速度提升高达 75%。
   - 这些增强功能旨在减少 GPU 机时及相关成本，从而加速 AI 解决方案的上市时间。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mattshumer_/status/1831767014341538166">Matt Shumer (@mattshumer_) 的推文</a>：我很高兴地宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种旨在让 LLM 修复自身错误的技术。405B 将于下周推出...</li><li><a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">使用 Together Kernel Collection 提升 NVIDIA H200 和 H100 GPU 集群性能</a>：无描述</li><li><a href="https://x.com/johnubalis/status/1831792041438949833">John Balis (@JohnUBalis) 的推文</a>：@neurosp1ke 看看这个美丽的野兽是如何执行（正确的！）长加法的</li><li><a href="https://glaive.ai/">Glaive - 适用于所有人的语言模型</a>：无描述</li><li><a href="https://github.com/open-thought/system-2-research">GitHub - open-thought/system-2-research: System 2 推理链接集合</a>：System 2 推理链接集合。欢迎通过在 GitHub 上创建账号为 open-thought/system-2-research 做出贡献。
</li>
</ul>

</div>

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1281345758913626255)** (9 messages🔥): 

> - `Triton 调试技巧`
> - `MLIR_ENABLE_DUMP`
> - `TRITON_INTERPRET`
> - `Triton 与 Marlin 的对比`
> - `量化零点效应` 


- **使用 MLIR_ENABLE_DUMP 进行调试**：一位成员建议使用 `MLIR_ENABLE_DUMP=1` 在每个编译器 pass 后转储 MLIR，展示 TTIR、TTGIR 和 LLIR 生成前后的 IR。
   - 这可以深入了解 Triton 如何编译代码，有助于精准定位问题。
- **TRITON_INTERPRET 是一个有用的工具**：另一位用户提到 `TRITON_INTERPRET=1` 是 Triton 中最好的调试辅助工具之一。
   - 社区普遍认为，调整设置可以极大地促进故障排除。
- **调试必不可少的环境变量**：一位成员强调 README 中包含各种可能有助于调试棘手问题的环境变量，尽管并非所有变量都是必需的。
   - 他们鼓励查看这些变量，因为它们在克服挑战时能提供显著帮助。
- **Triton 在极简代码下表现出色**：一位用户表达了对 Triton 能力的赞赏，指出仅需几行代码即可完成重大任务。
   - 然而，他们澄清说，由于处理零点量化（zero quantization）的方式不同，将 Triton 与 Marlin (VLLM) 进行对比并不简单。
- **对零点量化的担忧**：讨论中提到了量化零点的缺点，并引用了这种方法可能存在的精度问题。
   - 另一位成员指出，在 Marlin 的实现中，他们主要针对 AWQ 对零点进行舍入，并区分了对称量化和非对称量化。



**提及的链接**：<a href="https://github.com/triton-lang/triton/tree/7480ef5028b724cb434b7841b016c6d6debf3b84?tab=readme-ov-file#tips-for-hacking">GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84</a>：Triton 语言和编译器的开发仓库 - GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1281602675624120503)** (1 messages): 

> - `TorchDynamo 缓存查找`
> - `大模型的性能问题`
> - `torch/nn/modules/container.py` 


- **调查 TorchDynamo 缓存查找延迟**：在运行超大型模型时，成员注意到由于频繁调用 `torch/nn/modules/container.py(320): __getitem__`，**TorchDynamo 缓存查找**耗时达 **600us**。
   - 有人提出了关于该逻辑具体位置的查询，寻求进一步调查的线索。
- **大模型的性能关注点**：目前正在讨论对大模型的性能影响，特别关注**缓存查找延迟**。
   - 这凸显了对优化策略的需求，因为这些延迟在模型训练和推理过程中会不断累积。


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1281624427586523136)** (3 messages): 

> - `NVIDIA 生成式 AI 教学套件`
> - `高效机器学习课程`
> - `模型压缩技术`
> - `Llama2-7B 部署` 


- **NVIDIA 与达特茅斯学院合作开展 AI 教育**：NVIDIA 的 **Deep Learning Institute** 发布了与达特茅斯学院共同开发的[生成式 AI 教学套件](https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a)，旨在帮助学生理解 GPU 加速应用。
   - *Sam Raymond* 强调，完成本课程的学生将在就业市场上获得显著优势，有助于弥补各行业的知识鸿沟。
- **麻省理工学院（MIT）高效机器学习课程公告**：MIT 的一门新课程专注于**高效机器学习**和系统，以应对深度神经网络对云基础设施和日常设备的计算需求。涵盖的主题包括模型压缩、剪枝和量化。
   - 学生将获得在笔记本电脑上部署 **Llama2-7B** 的实战经验，学习在资源受限的设备上增强深度学习应用的实用技术。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.hackster.io/news/nvidia-teams-up-with-dartmouth-for-a-free-generative-ai-teaching-kit-11358047a05a">NVIDIA 与达特茅斯学院合作推出免费生成式 AI 教学套件</a>：学习 LLM, NLP, GPT, diffusion, 训练, 优化等内容 —— 或者获取你自己教学所需的材料。</li><li><a href="https://hanlab.mit.edu/courses/2024-fall-65940">MIT 6.5940 2024 秋季 TinyML 与高效深度学习计算</a>：未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1281452444110557246)** (3 messages): 

> - `Citadel Securities 招聘`
> - `Liquid AI 远程职位`
> - `CUDA Mode 认可度` 


- **Citadel Securities 寻求研究工程师**：Citadel Securities 正在寻找在 **Triton** 和/或 **CUDA** 方面有经验的研究工程师，强调他们具备在数 TB 金融数据上训练模型的能力。
   - 他们的目标是优化训练流水线并在数天内实现生产部署，更多详情请见其 [careers page](https://www.citadelsecurities.com/careers/details/machine-learning-engineer)。
- **Liquid AI 的远程职位引起关注**：一位成员指出了 Liquid AI 令人兴奋的远程工作机会，特别是 **Member of Technical Staff - AI Inference Engineer** 职位。
   - 这些职位在各大城市均可完全远程办公，且人才主管熟悉 **CUDA mode**，对于感兴趣的工程师来说这是一个很有前景的申请机会。
- **在 CUDA mode 发布职位的积极反馈**：另一位成员分享说他们认识 Liquid AI 的招聘人员，并称赞他们在 **CUDA mode** 发布职位空缺。
   - 这表明了 AI 领域内一个相互支持的社区以及相关机会的共享。



**Link mentioned**: <a href="https://jobs.lever.co/liquid.ai/">Liquid AI jobs</a>: Liquid AI 的职位空缺

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1281340616894058506)** (9 messages🔥): 

> - `图像卷积优化`
> - `控制分歧 vs 算术运算`
> - `用于 LLM 训练的 Triton Kernel` 


- **初学者探索图像卷积优化**：一位成员分享了他们在改进图像卷积 **优化技术** 方面的实验，重点介绍了 **constant memory** 的使用和意外的寄存器行为。
   - *局部内存的使用减少了常量加载*，这挑战了该成员对内存访问模式的理解。
- **控制分歧 (Control Divergence) 与算术运算的讨论**：社区分析了 CUDA 中控制分歧对性能的影响，一位成员由于 **编译器优化** 和更少的全局内存访问而倾向于方案 1。
   - 相反，另一位成员指出 **方案 2** 在自动合并 (automatic coalescence) 方面存在困难，使其效率复杂化。
- **探索用于训练的 Google Triton**：一位成员表达了他们对 **Google Triton** 小组以及关于用于 LLM 训练的 **高效 Triton Kernel** 的 YouTube 讲座的热情。
   - 他们计划在接下来的几周内深入研究教程并为社区做出贡献。



**Link mentioned**: <a href="https://www.youtube.com/watch?v=gWble4FreV4">Lecture 28: Liger Kernel - Efficient Triton Kernels for LLM Training</a>：Byron Hsu 介绍了 LinkedIn 的开源 Triton Kernel 集合，用于高效的 LLM 训练。时间戳 00:00 主持人开场 00:22 核心重点 01:18 大纲 03...

  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

0ut0f0rder: 谢谢！
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1281489812007682132)** (14 messages🔥): 

> - `FP16 x INT8 Matmul 中的 Batch Size 限制`
> - `Torch 编译器性能问题`
> - `Torchao 安装错误` 


- **FP16 x INT8 Matmul 在 Batch Size > 1 时遇到瓶颈**：在 **RTX 4090** 上，使用 `torch.compile` 的 **FP16 x INT8 matmul** 在 Batch Size 超过 1 时会崩溃，并抛出与共享内存容量相关的错误。
   - 用户推测 Inductor 配置可能针对 **A100 GPU** 进行了调整，导致在性能较低的设备上失败。
- **开启 Inductor 标志后性能下降**：当启用 Inductor 标志时，尽管有时不会报错，但在 Batch Size 大于 1 时计算速度显著变慢。
   - 关闭这些标志可以使 matmul 操作在不报错的情况下运行，尽管速度会有所降低。
- **Torchao 安装错误已解决**：在安装过程中遇到与 `torchao::quant_llm_linear` 相关的 `RuntimeError` 后，一位用户链接到了 [GitHub pull request](https://github.com/pytorch/ao/pull/826) 中的潜在修复方案。
   - 按照建议的修正操作后，错误得到解决，成功导入了必要的模块。



**Link mentioned**: <a href="https://github.com/pytorch/ao/pull/826">Unbreak build after #621 by andrewor14 · Pull Request #826 · pytorch/ao</a>: 无描述内容

  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1281361656630345748)** (9 messages🔥): 

> - `Avoiding Burnout Strategies` (避免倦怠的策略)
> - `Personal Projects for Productivity` (提升生产力的个人项目)
> - `Flow State in Programming` (编程中的心流状态)
> - `Work-Life Balance` (工作与生活的平衡)
> - `New System Torture Test Script` (新的系统压力测试脚本)


- **避免倦怠的简单方法**：一位成员表示，持续付出 **95%** 的努力比强冲 **105%** 更好，并强调从长远来看，这会带来更强的可持续性和生产力。
   - 他们强调，识别什么是你可以控制的，并接受什么是你无法控制的，对于管理个人目标而不陷入倦怠陷阱至关重要。
- **侧边项目重燃热情**：另一位成员分享说，参与工作之外的**小型侧边项目**有助于抵消倦怠感，让他们在没有公司压力的情况下获得成就感。
   - 他们指出，这种方法能保持编程的乐趣，并防止产生停滞不前感。
- **寻找你的心流状态**：讨论强调了在编程中达到**心流状态**的重要性，成员们一致认为，没有什么能比得上那种高度专注和高产的状态。
   - 一位成员指出，虽然当编程是为了学业或收入时更容易找到理由，但保持那种心流状态至关重要。
- **工作与生活平衡的重要性**：几位成员一致认为维持个人照顾的平衡是必要的，并表示忽视基本需求会导致生产力下降和痛苦。
   - 他们强调，工作中的乐趣和享受能提高产出，并建议在深入投入工作之前先处理好生活中的挑战。
- **介绍一个系统压力测试脚本**：一位成员分享了一个**新的系统压力测试（torture test）**脚本，它可以同时运行有效的 Bash、C、C++ 和 CUDA 代码，为用户提供了一个有趣且实用的挑战。
   - 该脚本可以在 [GitHub](https://github.com/apaz-cli/Scripts/blob/master/torture) 上找到，展示了它如何根据可用的编译器自行编译并启动测试 Kernel。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.nvidia.com/en-in/events/ai-summit/">加入 NVIDIA AI Summit 2024</a>：10 月 23–25 日，印度孟买</li><li><a href="https://github.com/apaz-cli/Scripts/blob/master/torture">apaz-cli/Scripts 仓库中的 Scripts/torture</a>：我每天使用的实用脚本。通过在 GitHub 上创建账号为 apaz-cli/Scripts 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1281508981600157708)** (6 messages): 

> - `Small Talk on llm.c in Yerevan` (在埃里温关于 llm.c 的简短演讲)
> - `Innovative Uses of llm.c` (llm.c 的创新用途)
> - `NCCL Multi-GPU Training` (NCCL 多 GPU 训练)
> - `Scaling on GPUs` (在 GPU 上的扩展性)


- **即将举行的埃里温 llm.c 演讲**：@aleksagordic 宣布将在埃里温进行一场关于 **llm.c** 的**简短演讲**，旨在提供高层级的概述，包括其他人的贡献。
   - 成员们表达了兴奋和兴趣，其中一位期待演讲的**录音**。
- **收集 llm.c 的创意用途**：有人询问是否有一份人们使用 **llm.c** 的**创意方式**清单（除了 fork 之外）。
   - 讨论中提到了一个具体案例，即 **chinthysl 在 472x H100s 上运行 llm.c**，展示了其扩展能力。
- **NCCL 多 GPU 多节点训练成功**：一位成员引用了 **chinthysl** 的一个 [GitHub PR](https://github.com/karpathy/llm.c/pull/426)，内容是关于在没有 MPI 的情况下仅使用 **NCCL 进行多 GPU** 训练，这简化了使用 Slurm 的作业调度。
   - 据指出，他们实现了至少到 **128 GPU** 的**线性扩展**，这在性能上取得了显著的成功。
- **对 llm.c 性能的热烈关注**：一些成员对 chinthysl 在 GPU 运行中观察到的令人印象深刻的扩展结果表示热烈关注，特别是关于 **472x GPU 设置**。
   - 他们注意到 chinthysl 的数据在某些修复后显示出改进，进一步证实了该方法的有效性。



**提到的链接**：<a href="https://github.com/karpathy/llm.c/pull/426#issuecomment-2175386065),">chinthysl 提交的：在没有 MPI 的情况下仅使用 NCCL 进行多 GPU 多节点训练 · Pull Request #426 · karpathy/llm.c</a>：与为集群设置 MPI 相比，在多节点训练设置中使用 Slurm 调度作业似乎要容易得多。此草案包含了在单节点训练中使用 mpirun 以及 S...

  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1281415594754707506)** (5 messages): 

> - `多模态收敛测试`
> - `Liger 的 Swiglu Kernel 性能`
> - `Together AI 的 GPU 集群`
> - `与 cuBLAS 的性能对比`
> - `Kernel 优化策略` 


- **多模态收敛测试 PR 已准备好进行评审**：一位成员宣布一个包含 **多模态收敛测试** 的 pull request 已准备好进行评审。
   - 这一新特性预计将增强该实现的测试能力。
- **Liger 的 Swiglu Kernel 对比 Together AI 基准测试**：一位成员询问了 **Liger 的 swiglu kernel** 与来自 [Together AI](https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection) 的基准测试相比的性能表现。
   - 他们强调 Together 的 **TKC** 为常见的训练操作提供了高达 **24% 的加速**。
- **专用 Kernel 的性能评估**：据分享，他们的专用 kernel 比使用 **cuBLAS** 和 **PyTorch eager mode** 的常规实现性能高出 **22-24%**。
   - 成员们讨论了缺乏细粒度调优（granular tuning）是他们融合（fusion）过程中一个潜在的改进方向。
- **对性能成就的好奇**：一位成员询问了关于 **Together AI** 与其他实现相比如何实现其性能提升的见解。
   - 这反映了人们对理解 kernel 优化最佳实践的持续关注。



**提到的链接**：<a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">Supercharging NVIDIA H200 and H100 GPU Cluster Performance With Together Kernel Collection</a>：未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1281382320191963300)** (43 条消息🔥): 

> - `Reflection Llama-3.1 70B`
> - `Glaive 数据使用`
> - `模型性能`
> - `围绕 LLM 的炒作`
> - `关于自我反思提示词的反馈` 


- **Reflection Llama-3.1 70B 面临褒贬不一的评价**：最近发布的 **Reflection Llama-3.1 70B** 声称是全球顶级的开源模型，但在 **BigCodeBench-Hard** 等基准测试中表现令人失望，得分远低于之前的模型。
   - 一位用户注意到其在推理任务中的性能下降，并幽默地将该模型在 Twitter 上的反响描述为“没啥新闻价值的平庸模型”。
- **对 Glaive 合成数据的担忧**：一些用户对 **Glaive** 生成的合成数据的有效性表示怀疑，并提到了数据集中过去存在的污染问题。
   - 对话暗示这些合成数据可能对 Reflection Llama 模型的性能和泛化能力产生了不利影响。
- **对自我反思能力的关注**：针对**自我反思过程**的底层逻辑出现了疑问，有人建议模型可能会为了进行反思和修正而故意学习生成错误。
   - 批评者指出，如果训练数据强调修正而非正确的推理，可能会培养出一种不利的模型行为。
- **社交媒体炒作的影响**：小组承认围绕新 AI 模型的巨大炒作，强调了社交媒体如何在性能可能存在差异的情况下放大预期。
   - 一位评论者幽默地谈到了 Twitter 的炒作文化，认为它对那些表现可能不如宣传的那样好的模型制造了不必要的兴奋感。
- **讨论对 SEO 的贡献**：几位用户认识到参与 Twitter 讨论对于提高博客文章曝光率和 SEO 指标的价值。
   - 尽管个人对该模型持怀疑态度，但一位用户表达了参与讨论主要是为了提升在线存在感的务实观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-09-05/the-rise-and-pivot-of-germany-s-one-time-ai-champion">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://x.com/mattshumer_/status/1831843865655214283">来自 Matt Shumer (@mattshumer_) 的推文</a>: Meta 联系了我，这是新的模型名称和链接: https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B  引用 Matt Shumer (@mattshumer_)：我很激动地宣布 Reflection 70B...</li><li><a href="https://x.com/HaveFunWithAI/status/1832107805815296376">来自 HaveFunWithAI (@HaveFunWithAI) 的推文</a>: 在数学问题上没有看到预期的改进。引用 Matt Shumer (@mattshumer_)：我很激动地宣布 Reflection 70B，全球顶级的开源模型。使用 Reflection-Tuning 训练...</li><li><a href="https://huggingface.co/TheDrummer/Llama-3SOME-8B-v2">TheDrummer/Llama-3SOME-8B-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/elder_plinius/status/1832107737012170940">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>: 这很奇怪... Reflection-70b 声称是由 Anthropic 创建的，而不是 Meta。“经过仔细考虑，我仍然确信是 Anthropic 创建了我，而不是 Meta。” 我想知道是否有任何 A...</li><li><a href="https://x.com/deepseek_ai/status/1832026579180163260">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 激动人心的消息！我们正式发布了 DeepSeek-V2.5 —— DeepSeek-V2-0628 和 DeepSeek-Coder-V2-0724 的强大结合体！现在，具备增强的写作、指令遵循和人类偏好...</li><li><a href="https://x.com/terryyuezhuo/status/1832112913391526052?s=46">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>: 在验证了所需的设置（使用系统提示词，无 prefilling）后，我可以有把握地说 Reflection 在 BigCodeBench-Hard 上的表现并不好。Complete: 20.3 (相比之下 Llama3.1-70B 为 28.4) Instruc...</li><li><a href="https://x.com/humancompressed/status/1832114674692731155">来自 ~bill (@humancompressed) 的推文</a>: @terryyuezhuo
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1281338645692944424)** (5 条消息): 

> - `HuggingFace Numina`
> - `Math benchmarks`
> - `CHAMP benchmark`
> - `Research queries` 


- **HuggingFace Numina 是宝贵的资源**：最近的讨论强调了 **HuggingFace Numina** 为数据相关任务提供了出色的工具，使其成为研究人员的宝贵资产。
   - 成员们对其在各种项目中的潜在应用表示兴奋。
- **标准数学基准测试保持不变**：尽管有许多可用工具，但普遍观点是新的 **math benchmarks** 并不多，重点仍集中在 **MATH** 和 **GSM8k** 上。
   - 这可能表明需要新的数据集或评估指标来推动该领域的发展。
- **引入 CHAMP 基准测试**：介绍了一个名为 **CHAMP** 的新基准数据集，重点是使用带有提示的注释数学问题来检查 LLM 的数学推理能力。
   - 这旨在提供一个框架，用于探索额外信息如何影响复杂场景下的问题解决。
- **寻求研究合作**：一位用户寻求关于非传统 **HuggingFace** 项目的建议，这些项目可能对研究工作来说比较冷门。
   - 呼吁提供任何有助于推进其研究的显著资源或想法。



**提到的链接**：<a href="https://arxiv.org/abs/2401.06961">CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs&#39; Mathematical Reasoning Capabilities</a>：最近的大型语言模型 (LLM) 在具有挑战性的竞赛级问题上表现出了数学推理能力的迹象，特别是通过自行生成的中间推理过程...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1281341709552451707)** (16 条消息🔥): 

> - `Reliability of Fireworks and Together`
> - `GitHub organization takedowns`
> - `Standardization of AI chat logs`
> - `Embarrassment in AI interactions`
> - `Chat templates for AI models` 


- **Fireworks 和 Together 的可靠性问题**：用户讨论了 **Fireworks** 和 **Together** 的可靠性问题，承认两者都不是 **100% reliable**。
   - 为了解决这个问题，他们实施了 **failovers** 以确保功能正常。
- **关于 GitHub 封禁的疑问**：有人询问 **GitHub** 是否会在不提供理由的情况下封禁组织，一些人回忆起过去曾发生过此类情况。
   - 人们对缺乏沟通表示担忧，特别是对于像 **Alibaba** 这样的大型实体。
- **需要标准的 AI 聊天日志**：一位成员提议应该有一个标准的 **`chats.txt`** 文件来记录与 AI 的交互，以便更好地进行代码库文档化。
   - 另一位成员建议 **Cursor** 可以增强这一想法的实用性，表明这种转变可能已经在发生。
- **对 AI 提问的尴尬感**：有人表达了向 **Cursor** 询问简单问题时的尴尬感，希望能维持一种专业能力的假象。
   - 这种情绪引起了其他人的共鸣，凸显了人们普遍担心被视为缺乏经验。
- **模型标准化的聊天模板**：有人建议将 AI 模型的 **chat templates** 标准化，作为实施 **`chats.txt`** 文件的先导。
   - 这幽默地暗示了创建这种标准化日志系统的潜在目标。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1281330402899791940)** (42 messages🔥): 

> - `无经验进入科技行业`
> - `Bing Copilot 功能`
> - `Perplexity AI 推荐计划`
> - `Web3 创新职位机会` 


- **无技能进入科技行业的建议**：一位成员表达了在没有技术技能的情况下进入科技行业的渴望，寻求关于构建有吸引力的 CV 和进行有效 Networking 的建议。
   - 另一位成员提到通过 PerScholas 开始 Cybersecurity 培训，并强调了对 Coding 和 AI 的热情。
- **Bing Copilot 的来源展示**：一位用户将 Bing Copilot 提供多达 **5 个来源**并带有内联图片的能力与 Perplexity 目前的功能进行了比较。
   - 他们建议 Copilot 在引用（Citations）上的悬停预览卡片可能是 Perplexity 可以考虑实施的一项改进。
- **Perplexity AI 周边商品推荐计划**：一个分享的链接显示，Perplexity 正在通过一项针对学生的推荐计划提供新周边，强调分享越多，获得越多。
   - 另一位成员询问如何获得一年的免费访问权限，询问是否仅限于前 500 名注册用户。
- **Web3 创新团队的职位空缺**：一则帖子强调了 Web3 创新团队的职位空缺，招聘职位从 Beta 测试员到开发者以及 UI/UX 设计师。
   - 该团队邀请申请和互利合作提案，作为其创意愿景的一部分。



**提到的链接**：<a href="https://x.com/perplexity_ai/status/1831762895220383807?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>：学生新周边 🔜 获得方式之一：推荐你的朋友使用 Perplexity！分享更多，收获更多：http://perplexity.ai/backtoschool

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1281348215009972245)** (11 messages🔥): 

> - `Sutskever 的 SSI 融资`
> - `大众汽车 ChatGPT 集成`
> - `AI 驱动的世界构建`
> - `NFL 2024 赛季开幕`
> - `Vehicle-to-everything 技术` 


- **Sutskever 的 SSI 获得 10 亿美元融资**：Perplexity AI 宣布 **Sutskever 的 SSI** 已成功筹集 **10 亿美元**，以进一步推动其 AI 技术的进步。
   - 这笔巨额资金预计将推动 AI 领域的更多创新。
- **大众汽车与 ChatGPT 合作**：大众汽车已将 **ChatGPT** 集成到其系统中，增强了用户交互和驾驶体验。
   - 这一举措代表了将先进 AI 功能集成到汽车技术中的重要一步。
- **生物混合蘑菇机器人亮相**：**生物混合蘑菇机器人（Biohybrid mushroom robots）**现已成为现实，展示了机器人技术和生物技术的令人兴奋的发展。
   - 这些机器人旨在以独特的方式与环境互动，挑战了传统机器人技术的界限。
- **NFL 2024 赛季开幕公布**：**NFL 2024 赛季开幕**详情已公布，引发了球迷和球队的热情。
   - 球迷们特别期待本赛季加入阵容的新球队和新球员。
- **探索 Vehicle-to-everything 技术**：围绕 **Vehicle-to-everything (V2X)** 技术的最新讨论强调了其在提高交通效率和安全性方面的潜力。
   - V2X 的创新有望增强车辆、基础设施和行人之间的连接性。



**提到的链接**：<a href="https://www.youtube.com/embed/HunZuUB0Xdo">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1281519125620850698)** (2 messages): 

> - `pplx-api 内存/记忆使用`
> - `Telegram bot 记忆存储` 


- **关于 pplx-api 记忆使用的咨询**：一位成员询问是否可以在通过 Python 使用 **pplx-api** 时利用内存/记忆（Memory）存储。
   - 他们请求关于如何实现此功能的指导。
- **Telegram Bot 的记忆存储策略**：另一位成员分享了他们尝试通过为 **Telegram bot** 管理独立数据库来实现记忆使用的尝试。
   - 这表明了将记忆功能集成到当前聊天系统中的兴趣。

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1281450485890420770)** (15 messages🔥): 

> - `Bounty 问题`
> - `Tinygrad 定价`
> - `服务器相关性`
> - `代码可读性`
> - `准则确认` 


- **Bounty 探索启动**：一位用户表示有兴趣尝试 Bounty（悬赏任务）并寻求入门指导，得到的回复指向了一份关于如何聪明提问的资源：[提问的智慧 FAQ](http://www.catb.org/~esr/faqs/smart-questions.html)。
   - 用户 *th.blitz* 幽默地接受了这一指导。
- **Tinygrad 定价降至零**：一位用户对一个关于每月 **60 美元** 提供 **4090 + 500GB** 的帖子提出疑问，*georgehotz* 透露价格已降至 **0 美元**，但仅限 tinygrad 的朋友。
   - *r5q0* 立即询问如何成为朋友。
- **服务器与 AI 查询的相关性**：一位用户指出，另一位用户关于 **AI 架构/数据集/LLM 微调** 的问题在 tinygrad 服务器中属于偏离主题，因为该服务器关注的是不同的抽象层级。
   - 该用户建议，虽然某些成员可能具备相关专业知识，但这些问题在此背景下可能不会受到欢迎。
- **代码可读性担忧**：一位成员表示，由于缺乏强制的列宽限制，尽管有大显示器，阅读 tinygrad 代码仍然很困难。
   - *leikowo* 承认应该设置此类限制，但指出某些行可能禁用了此功能。
- **准则确认的可见性**：一位用户询问特定频道中的指南是否是进入后首先看到的内容，是否需要确认后才能继续。
   - *wozeparrot* 确认情况确实应该是这样。



**提及的链接**：<a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>：未找到描述

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1281357362237669386)** (18 messages🔥): 

> - `PHI 操作困惑`
> - `MultiLazyBuffer 特性`
> - `分片缓冲区行为`
> - `关于 SDXL 推理的讨论`
> - `理解 Tensor 视图` 


- **澄清 PHI 操作困惑**：一位成员对 IR 中 PHI 操作的功能提出疑问，注意到其在循环结构中与 LLVM IR 的放置差异。
   - 另一位成员建议将其更准确地称为 ASSIGN 而非 PHI，表明其行为与传统的 phi 节点不同。
- **理解 MultiLazyBuffer 的 'real' 属性**：一位用户对 `MultiLazyBuffer.real` 的用途提出疑问，特别是它在 `MultiLazyBuffer.shrink` 中的作用以及与 `copy_to_device` 的交互。
   - 这引发了进一步调查，另一位成员指出它代表设备上的真实 lazy buffers，并且在某些配置下的类似设备可能存在 bug。
- **分片缓冲区行为查询**：一位用户详细介绍了他们对共享缓冲区的探索，特别是它们如何与 SDXL 推理的分片轴（sharded axes）交互，以及对 GPGPU 性能的影响。
   - 这项调查促使他们发起了一个讨论帖，寻求对其发现的反馈和改进建议。
- **关于沿分片轴进行 Cat 和 Shrink 的讨论**：一位用户发起了一项讨论，记录关于 cat 和 shrink 等张量操作在分片轴上的功能和限制的发现，特别是针对 MLPerf 推理任务。
   - 他们提供了 tinygrad 中不支持的操作示例，并寻求社区投入以填补这些空白。
- **视图与内存实例化澄清**：一位成员对 `_recurse_lb` 函数中视图（views）的实例化（realization）表示困惑，质疑内存优化与视图利用之间的平衡。
   - 这一讨论突显了在用户中澄清张量视图（tensor views）基本概念的持续努力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/uops.html">Kernel Fusion part 3: the linear layer UOps</a>：tinygrad 教程</li><li><a href="https://github.com/tinygrad/tinygrad/discussions/6380">Cat and Shrink Along Sharded Axis · tinygrad/tinygrad · Discussion #6380</a>：我写这篇报告是为了展示我的一些发现并获取反馈。以下内容目前不被 tinygrad 支持：a, b = [Tensor.rand(3,4).shard((&quot;NV:0&quot;,&quot;NV:1&quot;,&quot;NV:2...</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...tobias17:tinygrad:multilazybuffer-copy-fix">Comparing tinygrad:master...tobias17:multilazybuffer-copy-fix · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - 比较 tinygrad:master...tobias17:multilazybuffer-copy-fix · tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1281439463481806992)** (2 条消息): 

> - `Gemma 2 模型`
> - `资源链接`
> - `模型信息` 


- **关于 Gemma 2 模型链接的讨论**：成员们讨论了用户分享的 [Gemma 2 模型卡片](https://huggingface.co/google/gemma-2-9b)，其中提供了各种技术文档和资源的链接。
   - *Gemma* 被描述为来自 **Google** 的一系列轻量级、最先进的开放模型，基于与 **Gemini** 模型相同的技术构建。
- **Gemma 2 的相关资源链接**：分享了多个 Gemma 模型的资源，包括 [Responsible Generative AI Toolkit](https://ai.google.dev/responsible) 以及指向 [Kaggle](https://www.kaggle.com/models/google/gemma-2) 和 [Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) 的链接。
   - 成员们强调了查阅这些资源对于理解生成式 AI 的能力和伦理的重要性。



**提到的链接**：<a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>：未找到描述

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1281556400446640209)** (28 条消息🔥): 

> - `多模态生成处理`
> - `用于文档掩码的 Flex Attention`
> - `INT8 混合精度训练`
> - `TransformerDecoder 配置`
> - `用于生成重构的 GitHub PR` 


- **处理多模态模型的因果掩码 (Causal Masks)**：一位成员概述了在多模态设置中（特别是在固定序列长度下）推理期间管理 **causal masks** 的挑战。
   - *看到我们已经通过注意力层暴露了这些变量* 有助于澄清处理方法。
- **期待 Flex Attention 带来的加速**：人们乐观地认为，**带有文档掩码 (document masking) 的 flex attention** 将显著提升性能，尤其是在 **A100 上提升 40%，在 4090 上提升 70%**。
   - 这种方法对于增强 **动态序列长度** 训练，同时最小化填充 (padding) 效率低下的问题至关重要。
- **关于 TransformerDecoder 设计的问题**：一位成员询问是否可以在没有自注意力层的情况下设置 **TransformerDecoder**，并引用了其传统结构。
   - 另一位成员指出 *原始 Transformer 使用了* 交叉注意力和自注意力层，这表明偏离该模型存在挑战。
- **生成重构 (Generation Overhaul) 的 PR 更新**：一位成员确认 **#1449** 已更新，以提高与 `encoder_max_seq_len` 和 `encoder_mask` 的兼容性，尽管测试仍在进行中。
   - 一旦这次重构落地，将允许对 **generation utils** 进行进一步更新并集成到 **PPO** 中。
- **缓存重构与生成工具类**：讨论了将 **生成功能移出 utils** 的相关内容，GitHub PR **#1424** 因需要进行缓存重构而待定。
   - 解决 **GemmaTransformerDecoder** 过时的问题使得进一步开发的对话变得非常紧迫。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1424).">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/ao/pull/748">Add INT8 mixed-precision training by gau-nernst · Pull Request #748 · pytorch/ao</a>：来自新 README 的摘录。INT8 训练的术语通常尚未标准化。准确地说，我们使用这些术语的含义如下：量化训练：模型权重是...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1281353509106749512)** (4 条消息): 

> - `llama-deploy launch`
> - `agentic system deployment example`
> - `Running Reflection 70B`
> - `advanced agentic RAG pipelines` 


- **面向微服务的 llama-deploy 发布**：宣布发布 **llama-deploy**，这是一个旨在促进基于 **LlamaIndex Workflows** 的微服务无缝部署的系统。这标志着自引入 llama-agents 和 Workflows 以来的重要演进。
   - 更多详情，请查看 [发布公告](https://twitter.com/llama_index/status/1831794126511337880)。
- **使用 llama-deploy 的端到端示例**：@LoganMarkewich 分享了一个开源示例，展示了如何使用 **llama-deploy** 和 **@getreflex** 前端框架构建一个 **Agent 聊天机器人系统**。这个全栈示例演示了将 Agent 系统作为微服务进行部署。
   - 在此 [示例链接](https://twitter.com/llama_index/status/1832132462786576652) 中查找代码和详情。
- **在笔记本电脑上运行 Reflection 70B**：如果你的笔记本电脑性能足够，现在可以使用 **Ollama** 运行 **Reflection 70B**。这允许直接通过 **LlamaIndex** 对其进行操作。
   - 更多信息请参阅 [推文](https://twitter.com/llama_index/status/1832144451579613497)。
- **使用 Amazon Bedrock 构建 RAG 流水线**：了解如何使用 **LlamaIndex** 和 **Amazon Bedrock** 构建高级 **Agentic RAG 流水线**。该过程包括创建流水线、实现动态查询路由以及使用查询分解。
   - 按照此处的详细指南进行逐步操作：[详细指南](https://twitter.com/llama_index/status/1832189386169184562)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1281487782547820649)** (21 条消息🔥): 

> - `PandasQueryEngine issues`
> - `Customer support chatbot integration`
> - `NeptuneDatabaseGraphStore bug`
> - `Cohere reranker in Azure` 


- **PandasQueryEngine 在列名识别上存在困难**：一位用户报告称，`PandasQueryEngine` 在与 Chat Engine 配合使用时无法正确识别 `averageRating` 列，经常默认使用错误的名称如 `rating`。
   - 另一位成员建议在 Chat Engine 的上下文中验证 DataFrame 列的映射以解决此问题。
- **为聊天机器人结合 Chat Engine 和 Query Engine**：一位社区成员正在寻求关于开发客服聊天机器人的建议，该机器人需要同时利用对话引擎和检索增强生成 (RAG) 方法。
   - 成员们一致认为，各种 Chat Engine 都可以与 Query Engine 高效集成，以增强聊天机器人应用的对话和数据检索能力。
- **NeptuneDatabaseGraphStore 中的潜在 Bug**：有人对 `NeptuneDatabaseGraphStore.get_schema()` 函数可能存在的 Bug 表示担忧，该函数未能将日期信息包含在图摘要中。
   - 一位用户指出，问题可能源于向 LLM 提供数据时的模式解析错误，并且对 `datetime` 包也存在疑虑。
- **Azure 中的 Cohere Reranker 集成**：一位用户询问关于在 Azure 的 LlamaIndex 中将 Cohere Reranker 作为 Node Postprocessor 使用的问题，并引用了 GitHub 上的相关咨询。
   - 已确认目前尚不存在现有的 Azure Rerank 模块，但一位社区成员鼓励创建一个，因为基类很简单且有文档可参考。



**提到的链接**：<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>：未找到描述

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1281415627654959195)** (18 messages🔥): 

> - `Reflection Llama-3.1 70B`
> - `Synthetic Dataset Generation`
> - `Model Thinking Space`
> - `Fine-tuning Challenges`
> - `ReAct CoT Technique` 


- **Reflection Llama-3.1 70B 成为顶级 LLM**: [Reflection Llama-3.1 70B](https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B) 是全球领先的开源 LLM，在解决初始上传问题后，利用 **Reflection-Tuning** 增强了推理准确性。
   - 它是在由 [Glaive](https://glaive.ai) 创建的合成数据上训练的，鼓励用户在[此链接](https://reflection-playground-production.up.railway.app/)测试该模型。
- **合成数据集生成速度**: 讨论强调了 Reflection Llama-3.1 的合成数据集据报道生成速度非常快，引发了关于其 **human rater**（人类评分员）参与度和样本量的疑问。
   - 成员们推测了在保持质量的同时，创建此类数据集的速度能有多快。
- **模型的思考空间带来提升**: 一位成员指出，给模型留出思考空间的能力在 AI 圈内是众所周知的，并提到 **ReAct** 已经实施这种方法近两年了。
   - 他们进一步指出，一个 **4B 参数模型** 表现优于 **GPT-3.5 turbo** 的有趣能力，引发了热烈讨论。
- **微调 Llama-3.1 的挑战**: 对话转向微调这种稠密模型的挑战，成员们承认每个参数对性能都至关重要。
   - 提出了对微调复杂性的担忧，并出现了关于需要自定义 Token 以连接预期数据集结构的争论。
- **ReAct CoT 性能讨论**: 成员们讨论了 **ReAct Chain of Thought** 方法的有效性，称其在不一定需要重新训练模型的情况下就能产生强大的结果。
   - 提到了诸如 Logit 约束之类的策略，作为在保持清晰度的同时管理输出的替代方案。



**Link mentioned**: <a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1281639933064384534)** (2 messages): 

> - `Fine-tuning Llama 3.1`
> - `GPU requirements for Lora finetuning` 


- **使用扩展序列长度微调 Llama 3.1**: 一位成员询问了有效微调 **Llama 3.1** 的技术，提到它在 **8k 序列长度** 下表现良好。
   - 他们指出 **RoPE scaling** 似乎能将性能提升至 **128k**，暗示其中可能有一些技巧。
- **LoRA 微调所需的 A100 GPU**: 另一位成员询问了使用 **adamw_bnb_8bit** 以 **4 bit** 模式微调 **Meta-Llama-3.1-405B-BNB-NF4-BF16** 所需的 **A100 80 GB GPU** 数量估算。
   - 这突显了高效进行 **LoRA finetuning** 的实际考虑和资源需求。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1281332076158910535)** (2 messages): 

> - `SmileyLlama`
> - `Chemical Language Model`
> - `Molecule Design` 


- **SmileyLlama：新型化学语言模型**: [SmileyLlama](https://x.com/axolotl_ai/status/1831771214445945148) 是一个经过微调的 **Chemical Language Model**，可根据 Prompt 中指定的属性设计分子。
   - 它是一个与纯 CLM 相当的 **SFT+DPO 模型**，但专门使用 `Axolotl` 构建。
- **Axolotl 的分子生成方法**: **SmileyLlama** 的开发展示了 Axolotl 在针对分子设计等特定任务微调模型方面的能力。
   - 这一进步说明了 **Axolotl** 如何调整现有的 CLM 技术以增强功能。



**Link mentioned**: <a href="https://x.com/axolotl_ai/status/1831771214445945148">Tweet from Axolotl (@axolotl_ai)</a>: SmileyLlama, a fine-tuned Chemical Language Model to design molecules from properties specified in the prompt. An SFT+DPO model on par with other pure CLM&#39;s, but built with Axolotl.

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1281336290377207851)** (15 messages🔥): 

> - `Cohere 资源`
> - `Anthropic 库的使用`
> - `Azure 上的 Embed-multilingual-light-v3.0` 


- **探索 Cohere 的功能和 Cookbooks**：成员们讨论了查看专门用于 [功能与演示](https://discord.com/channels/1218409701339828245) 的频道，社区在该频道分享了使用 Cohere 模型构建的项目，并引用了提供现成指南的全面 [cookbook](https://docs.cohere.com/page/cookbooks)。
   - *sssandra* 强调这些 cookbook 展示了利用 Cohere 生成式 AI 平台的最佳实践。
- **理解 Anthropic 库的 Token 使用情况**：*vpkprasanna* 询问了关于使用 Anthropic 库的问题，并分享了一个用于计算 Token 使用情况的代码片段：`message = client.messages.create(...)`。
   - 他们引导其他人前往 Anthropic SDK 的 [GitHub 仓库](https://github.com/anthropics/anthropic-sdk-python) 以进一步探索 Tokenization。
- **Embed-Multilingual-Light-V3.0 在 Azure 上的可用性**：*arcz1337* 询问了 `embed-multilingual-light-v3.0` 在 Azure 上的可用性，并询问是否有支持该模型的计划。
   - 这一询问反映了用户对 Cohere 资源与流行云平台集成的持续关注。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/page/cookbooks">Cookbooks — Cohere</a>: 未找到描述</li><li><a href="https://github.com/anthropics/anthropic-sdk-python">GitHub - anthropics/anthropic-sdk-python</a>: 通过在 GitHub 上创建账号，为 anthropics/anthropic-sdk-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1281618137678086227)** (2 messages): 

> - `RAG 引用`
> - `以文本文件作为知识库` 


- **关于 RAG 引用的查询**：一位成员询问在使用带有外部知识库的 **RAG** 时，引用（citations）将如何影响文本文件的内容。
   - 他们特别询问了为什么在处理文本文件内容时，目前收到的引用结果为 **None**。
- **寻求 RAG 引用方面的帮助**：同一位成员寻求帮助，以在其 **RAG** 实现中获取源自文本文件内容的引用。
   - 他们表达了解决响应中缺失引用问题的紧迫性。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1281483770494582795)** (3 messages): 

> - `Chroma DB 设置`
> - `Weaviate 示例`
> - `用于服务器-客户端通信的 Jupyter Notebooks` 


- **更简便的 Chroma DB 设置**：一位成员强调了 **Chroma DB** 的极简设置，只需一行代码即可在本地运行服务器：`!chroma run --host localhost --port 8000 --path ./ChomaM/my_chroma_db1`。
   - 他们对能如此简单地了解数据库位置和操作表示满意。
- **寻求简化的 Weaviate 设置**：同一位成员询问是否存在类似的 **Weaviate** 直观设置，而无需诉诸于使用 **Go Docker** 和其他复杂操作。
   - *他们强调，鉴于自己的非技术背景，渴望易用性*。
- **生物学家的 Jupyter Notebooks 工具使用**：另一位成员分享了他们利用 **两个 Jupyter Notebooks** 分别启动服务器和运行客户端的方法，并表示这能满足他们的需求。
   - *他们表明自己是一名生物学家，而非计算机科学专业的毕业生，这进一步强调了他们对简单性的需求。*
- **对 Weaviate 示例的渴望**：该成员表示打算为 **Weaviate** 创建实际示例，以帮助理解和设置。
   - 这展示了尽管面临技术挑战，他们仍采取了积极主动的学习态度。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1281474419788615740)** (3 条消息): 

> - `名称的重要性`
> - `协作学习`
> - `AI 在教育中的应用`
> - `MAIC 提案`
> - `在线课程演变` 


- **名称具有无限潜力**：一位成员提到*令人惊讶的是我们永远不会用完名称*，强调了名称生成的多样性和创造力。
   - 这场对话说明了在各种语境下命名规范的无限可能性。
- **协作学习创新**：提到的 **collabin** 标志着关于在线教育和项目工作中协作倡议的持续讨论。
   - 这类平台强调了教育环境中向更集成学习体验的转变。
- **AI 增强功能变革教育**：分享的一条关于论文的详细链接讨论了 **AI 技术** 如何集成到在线教育中，以实现个性化并提高学习成果。
   - 这突显了使用 **LLM** 来增强学习体验的新兴趋势。
- **为在线教育引入 MAIC**：提出的 **MAIC (Massive AI-empowered Course)** 旨在利用 **LLM** 驱动的多 **Agent** 系统来构建 AI 增强型课堂。
   - 该概念力求在平衡技术集成的同时，提升学习者的教育体验。
- **在线课程的演变**：关于在线课程演变的讨论展示了**教育模式**随时间的不断适应。
   - 这种适应性对于满足各种学习需求和偏好至关重要，强调了持续创新的重要性。



**提到的链接**：<a href="https://huggingface.co/papers/2409.03512">Paper page - From MOOC to MAIC: Reshaping Online Teaching and Learning through
  LLM-driven Agents</a>：未找到描述

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1281411157005045821)** (2 条消息): 

> - `Reflection 70B`
> - `根据查询路由 LLM`
> - `TPU 速度与定价` 


- **Reflection 70B 被宣布为领先的开源模型**：分享了 **Reflection 70B** 的发布，它被誉为全球顶尖的开源模型，强调了其通过 **Reflection-Tuning** 纠正自身错误的能力。
   - *405B 预计将于下周发布*，并承诺提供卓越的性能，该模型是与 **@GlaiveAI** 合作开发的。
- **对 CoT DSPy 程序逻辑的兴趣**：一位社区成员询问了 **CoT DSPy 程序** 的细节，质疑其关于对所提供答案进行反思的功能。
   - 对于其在任务执行中的实现和效用存在期待。
- **在 LLM 路由中加入定价和 TPU 速度**：一位成员表示有兴趣开发一种根据查询路由合适 **LLM** 的方法，并结合基于模型托管的**定价**和 **TPU 速度**。
   - 他们指出，虽然路由正确的 **LLM** 很简单，但性能和成本等额外元素将增强这一过程。



**提到的链接**：<a href="https://x.com/mattshumer_/status/1831767014341538166?t=DJHN74LHKtz5ULXGi2vK_A&s=19">Matt Shumer (@mattshumer_) 的推文</a>：我很激动地宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种为使 LLM 能够修复自身错误而开发的技术。405B 将于下周推出...

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1281357639389020191)** (5 条消息): 

> - `SwarmUI`
> - `User Interface Design` (用户界面设计)
> - `Bane Meme` 


- **关于 SwarmUI 易用性的讨论**：一位成员对包含 **100 个节点**的 UI 表示不适，随后提到了 **SwarmUI** 作为对比。
   - 另一位成员强化了这一观点，称其“简直就是 SwarmUI”。
- **SwarmUI GitHub 介绍**：分享了 [SwarmUI on GitHub](https://github.com/mcmonkeyprojects/SwarmUI) 的链接，强调其模块化设计旨在增强可访问性和性能。
   - 该项目因专注于让强力工具（powertools）易于使用而受到关注，并展示了仓库的视觉图片。
- **Bane Meme 与 GIF 分享**：一位成员分享了一个 **Bane 主题的 GIF**，其中包含一只配有“and you are”字幕的绿色青蛙。
   - 该 GIF 引发了进一步讨论，并链接了多个相关的搜索结果，展示了各种 **Bane** 和**爆炸**主题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504">Bane No GIF - Bane No Banned - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (原名 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，重点在于让强力工具易于访问、高性能且具有可扩展性。</a>：SwarmUI (原名 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，重点在于让强力工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/Swa...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1281349209928503489)** (3 条消息): 

> - `Reflection 70B`
> - `LLM Self-Correction` (LLM 自我修正)
> - `Lucidrains Transfusion Implementation` (Lucidrains 的 Transfusion 实现)
> - `405B Model Release` (405B 模型发布)


- **Reflection 70B 作为顶级开源模型发布**：Matt Shumer 宣布推出 **Reflection 70B**，声称它是全球顶级的开源模型，通过 **Reflection-Tuning** 技术训练而成，该技术使 LLM 能够修复自己的错误。
   - 他还暗示下周将推出 **405B 模型**，预计将超越所有现有基准测试。
- **LLM 通过自我修正对抗 Bug**：Kimmonismus 对这款新型 LLM 表示难以置信，该模型不仅能自我纠错，而且据称在包括 **MMLU 和 MATH** 在内的所有测试基准中都击败了 GPT-4o。
   - 他强调这个新模型是开源的，且表现大幅优于 **Llama 3.1 的 405B**，标志着 LLM 能力的重大进步。
- **Lucidrains 实现 Transfusion 模型**：分享了 Lucidrains 对 **Transfusion** 模型的重新实现，该模型旨在预测下一个 token 的同时进行图像扩散，展示了其**多模态（multi-modal）**能力。
   - 该项目承诺未来将扩展到包括 **flow matching** 以及音频/视频处理，代表了 AI 模型领域一个值得关注的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lucidrains/transfusion-pytorch">GitHub - lucidrains/transfusion-pytorch: Transfusion 的 Pytorch 实现，“通过一个多模态模型预测下一个 Token 并扩散图像”，来自 MetaAI</a>：Transfusion 的 Pytorch 实现，“通过一个多模态模型预测下一个 Token 并扩散图像”，来自 MetaAI - lucidrains/transfusion-pytorch</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Matt Shumer (@mattshumer_) 的推文</a>：我很高兴宣布 Reflection 70B，全球顶级的开源模型。使用 Reflection-Tuning 训练，这是一种为使 LLM 能够修复自身错误而开发的技术。405B 将于下周推出...</li><li><a href="https://x.com/kimmonismus/status/1831772661296345333?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Chubby♨️ (@kimmonismus) 的推文</a>：我简直不敢相信我在这里读到的内容：一个能修复自身 Bug、自我纠错并在所有基准测试中击败包括 GPT-4o 在内的所有当前模型的 LLM？而且这个模型还是开源的？...
</li>
</ul>

</div>

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1281346156101636149)** (6 条消息): 

> - `在 GCP 上部署 ReAct agent`
> - `LangChain Callbacks 系统`
> - `Cerebras 与 LangChain`
> - `从 .astream_events 解码流` 


- **ReAct Agent 部署挑战**：一位成员在使用 FastAPI 在 GCP 上部署其 ReAct Agent 时面临挑战，因为本地 SQLite 数据库在重新部署时会消失。他们正在寻求替代方案，特别是使用 Postgres 或 MySQL 实现来替换 `SqliteSaver`。
   - 如果有人觉得有帮助，该成员愿意分享他们的本地实现以供参考。
- **澄清 LangChain 中 Callbacks 的用法**：关于 `chain = prompt | llm` 语法是否正确的讨论引发了对 [LangChain callback 文档](https://python.langchain.com/v0.1/docs/modules/callbacks/) 的关注。成员们指出文档可能已经过时，特别是提到了 0.2 版本中的更新。
   - 对话强调了 Callbacks 系统在日志记录、监控以及与第三方工具集成方面的实用性。
- **关于 Cerebras 和 LangChain 的咨询**：一位成员询问是否有人在将 Cerebras 与 LangChain 结合使用，表示需要协作见解。回复中显示了潜在的兴趣，但缺乏具体的互动。
   - 针对这一咨询，目前尚未分享直接的解决方案或经验。
- **探索 .astream_events() 解码**：一位成员询问是否有从 `.astream_events()` 解码流的参考实现。另一位成员分享了由于缺乏资源而不得不手动序列化每种事件类型的经验。
   - 这段对话表达了对繁琐过程的沮丧，并希望社区内能有更好的解决方案。



**提到的链接**：<a href="https://python.langchain.com/v0.1/docs/modules/callbacks/">Callbacks | 🦜️🔗 LangChain</a>：前往 Integrations 查看关于内置 Callbacks 与第三方工具集成的文档。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1281622843179925648)** (5 条消息): 

> - `RAG 系统改进`
> - `Embedding 模型使用`
> - `混合搜索 (Hybrid search)`
> - `元数据与重排序 (Reranking)` 


- **硬件限制下的 RAG 系统改进**：一位成员询问如何增强其 RAG 系统，具体使用了 **llama3-8b 配合 4bit quantization** 和 **BAAI/bge-small-en-v1.5** Embedding 模型。
   - 他们表示受到硬件限制（仅有一块 **4090** GPU），并寻求更好的实现资源。
- **在 4090 GPU 上探索更大的模型**：作为回应，一位成员指出，使用 **4090** 可以在运行 llama-8b 的同时运行更大的 Embedding 模型，并建议 **3.1 版本** 可能也会有帮助。
   - 他们分享了一个有用的 [GitHub 示例](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py)，演示了在 Milvus 上集成 **bge & bm25** 的混合搜索。
- **利用元数据进行重排序 (Reranking)**：讨论强调了为 **每个 chunk 配备元数据** 的重要性，以协助进一步对结果进行排序和过滤。
   - Reranker 可以显著优化搜索过程，提高用户的整体输出质量。



**提到的链接**：<a href="https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py">pymilvus/examples/hello_hybrid_sparse_dense.py at master · milvus-io/pymilvus</a>：Milvus 的 Python SDK。欢迎在 GitHub 上通过创建账号为 milvus-io/pymilvus 贡献代码。

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1281748599537143859)** (1 条消息): 

> - `XLAM 系统提示词 (System Prompt)`
> - `OSS 模型对比` 


- **XLAM 独特的系统提示词**：一位成员注意到 **XLAM 的系统提示词** 与其他 **OSS 模型** 不同。
   - “是否有特定原因？”引发了探索这些差异背后基本原理的兴趣。
- **对系统设计选择的好奇**：讨论强调了关于 XLAM 系统提示词背后 **设计选择** 的一个有趣方面。
   - 成员们热衷于了解这些变化是出于功能性考虑还是许可方面的考虑。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1281500226779222090)** (3 messages): 

> - `Testing API server`
> - `Adding models to leaderboard`
> - `Gorilla leaderboard` 


- **如何测试你自己的 API Server**：一位用户询问了如何有效测试他们自己的 **API server** 并请求相关文档。
   - 回复中未提供具体资源，表明回复中可能存在知识空白。
- **为 Leaderboard 做出贡献**：一位用户询问如何将新模型添加到 **leaderboard**，这对于认可模型贡献至关重要。
   - 作为回应，分享了相关 [GitHub 页面](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 的链接，详细说明了 **Gorilla leaderboard** 的贡献指南。
- **Gorilla Leaderboard GitHub 资源**：另一位用户强调了 GitHub 上提供的 **Gorilla: Training and Evaluating LLMs for Function Calls** 资源。
   - 该资源详细说明了向 leaderboard 贡献的过程，并附带了其 [GitHub 仓库](https://opengraph.githubassets.com/25d4bf4245a01dd99c8e3d1e4b47d26ef3db55d11499f2f9edfa259231aaacd2/ShishirPatil/gorilla) 的图片。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

knut09896: hi there
  

---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}