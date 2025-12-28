---
companies:
- deepseek
- ollama
- qwen
- llama
date: '2025-01-21T07:50:24.815688Z'
description: '**DeepSeek** 发布了 **DeepSeek R1**，这是对其仅三周前发布的 **DeepSeek V3** 的重大升级。此次共推出了
  8 个模型，包括全尺寸的 671B MoE 模型，以及基于 **Qwen 2.5** 和 **Llama 3.1/3.3** 的多个蒸馏版本。


  这些模型采用 MIT 开源协议，允许进行微调和蒸馏。其价格显著低于 **o1**，便宜约 27 至 50 倍。训练过程采用了 **GRPO**（组相对策略优化，针对正确性和风格结果进行奖励），且不依赖于
  PRM（过程奖励模型）、MCTS（蒙特卡洛树搜索）或传统的奖励模型，重点在于通过强化学习提升推理能力。


  蒸馏后的模型可以在 **Ollama** 上运行，并展现出编写 **Manim 代码**（数学动画代码）等强大能力。此次发布强调了在强化学习、微调和模型蒸馏方面的进展，并采用了源自
  DeepSeekMath 的新型强化学习框架。'
id: d7e2899a-fcf4-4b0b-a5aa-beb34e412da2
models:
- deepseek-r1
- deepseek-v3
- qwen-2.5
- llama-3.1
- llama-3.3-70b
original_slug: ainews-deepseek-r1-o1-level-open-weights-model
people: []
title: DeepSeek R1：性能媲美 o1 的权重开放模型，以及将 1.5B 模型提升至 Sonnet/4o 级别的简单方法。
topics:
- reinforcement-learning
- fine-tuning
- model-distillation
- model-optimization
- reasoning
- reward-models
- multi-response-sampling
- model-training
---

<!-- buttondown-editor-mode: plaintext -->**GRPO 就是你所需的一切。**

> 2025年1月17日至1月20日的 AI 新闻。我们为您查看了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 社区（**225** 个频道，**8019** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**910 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们知道 DeepSeek 迟早会发布开源权重版本，而且 DeepSeek 已经因其论文而闻名，[V3 曾是全球顶尖的开源模型](https://www.latent.space/p/baseten)，但今天我们所有的 AI 消息源都无法将目光从 DeepSeek R1 的发布上移开。


![image.png](https://assets.buttondown.email/images/82e28b15-aa19-4a86-8947-b5a6930b919d.png?w=960&fit=max)


R1 的性能表现证明它比[仅仅三周前的 DeepSeek V3](https://buttondown.com/ainews/archive/ainews-deepseek-v3-671b-finegrained-moe-trained/) 有了跨越式的提升：


![image.png](https://assets.buttondown.email/images/b70431e4-c259-4163-af25-e4f1d14b5b4e.png?w=960&fit=max)
![image.png](https://assets.buttondown.email/images/a5b3346d-a72b-424f-a33c-70918db2b2cb.png?w=960&fit=max)


当我们提到 "R1" 时，这个表述其实比较模糊。DeepSeek 实际上发布了 8 个 R1 模型——2 个“完整”模型，以及 6 个基于开源模型的蒸馏版本：

- 基于 Qwen 2.5：使用 DeepSeek-R1 筛选的 80 万个样本进行微调，包含 1.5B、7B、14B 和 32B 版本
- 基于 Llama 3.1 8B Base：DeepSeek-R1-Distill-Llama-8B 
- 基于 Llama3.3-70B-Instruct：DeepSeek-R1-Distill-Llama-70B
- 以及 **DeepSeek-R1 和 DeepSeek-R1-Zero**，即类似于 [DeepSeek V3](https://www.latent.space/p/baseten) 的全尺寸 671B MoE 模型。令人惊讶的是，它们采用了 [MIT 许可证](https://x.com/deepseek_ai/status/1881318138937233664?s=46)而非自定义许可证，并明确允许进行微调和蒸馏

发布会中的其他亮点：

- **定价**（每百万 token）：输入 14 美分（缓存命中），输入 55 美分（缓存未命中），输出 219 美分。相比之下，o1 的价格为输入 750 美分（缓存命中），输入 1500 美分（缓存未命中），输出 6000 美分。**这比 o1 便宜了 27 到 50 倍。**
- [解决了 o1 博客文章中的每一个问题](https://x.com/mrsiipa/status/1881330071874813963)。[每一个](https://x.com/nrehiew_/status/1881453058556870934?s=46)。
- [可以在 Ollama 上运行蒸馏模型](https://simonwillison.net/2025/Jan/20/deepseek-r1/)
- 能够非常好地编写 [Manim 代码](https://x.com/christiancooper/status/1881335734256492605)

[论文](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)中的惊喜：

- 过程如下：
  1. [V3 Base → R1 Zero](https://x.com/casper_hansen_/status/1881404608591085817)（使用 GRPO —— 即针对正确性和风格结果的奖励 —— 没有花哨的 PRM/MCTS/RMs）
  2. [R1 Zero → R1 Finetuned Cold Start](https://x.com/casper_hansen_/status/1881404611401236745)（从 R1 Zero 中蒸馏长 CoT 样本）
  3. [R1 Cold Start → R1 Reasoner with RL](https://x.com/casper_hansen_/status/1881404614190506188)（专注于语言一致性 —— 以产生可读的推理）
  4. [R1 Reasoning → R1 Finetuned-Reasoner](https://x.com/casper_hansen_/status/1881404617235509711)（生成 600k 样本：多回复采样并仅保留正确样本（使用之前的规则），并使用 V3 作为裁判：过滤掉混合语言、长段落和代码）
  5. [R1 Instruct-Reasoner → R1 Aligned](https://x.com/casper_hansen_/status/1881404619362013294)（使用 GRPO 平衡推理能力与有用性和无害性）
- [可视化](https://x.com/SirrahChan/status/1881488738473357753)：
![image.png](https://assets.buttondown.email/images/c2e152cb-1ae5-4c2e-88fe-39d6b0fb411b.png?w=960&fit=max)

- 有监督数据、Process reward models 以及 [MCTS](https://x.com/lu_sichu/status/1881348105586855962) **没有**奏效

![image.png](https://assets.buttondown.email/images/d7de6974-4c88-40dd-94fe-408c9251306c.png?w=960&fit=max)

- 但他们确实使用了来自 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)（[受到 DPO 作者的质疑](https://x.com/rm_rafailov/status/1881350883252085000)）作为“提高模型推理性能的 RL 框架”，其中推理能力（如 [in-context back-tracking](https://x.com/paul_cal/status/1881324020592963939)）在“数千步 RL 训练”后“自然涌现” —— 虽然[不完全是](https://x.com/cto_junior/status/1881319502861967635)著名的 o1 scaling 曲线，但也是近亲。
![image.png](https://assets.buttondown.email/images/53113db9-e27c-4f57-9a0e-3c2ddf68d842.png?w=960&fit=max)

- 使用 [“顿悟时刻”（aha moments）](https://x.com/teortaxesTex/status/1881317131561922640)作为关键 token，通常[以一种对读者不友好的方式混合语言](https://x.com/teortaxesTex/status/1881329351125549144)
- R1 [在 o1 发布后不到一个月就开始了训练](https://x.com/teortaxesTex/status/1881298065967239183)
- R1 的蒸馏[效果显著](https://x.com/nrehiew_/status/1881330794549182853)，带给我们[这段疯狂的引言](https://x.com/reach_vb/status/1881319500089634954)：“DeepSeek-R1-Distill-Qwen-**1.5B 在数学基准测试中超越了 GPT-4o 和 Claude-3.5-Sonnet**，AIME 得分为 28.9%，MATH 得分为 83.9%。”，而且这甚至还是[在没有将蒸馏推向极限的情况下](https://x.com/teortaxesTex/status/1881331287010550119)实现的。
- 这[比仅仅对小模型进行 RL 微调更有效](https://x.com/DimitrisPapail/status/1881341537499619822)：“[大模型的推理模式可以被蒸馏到小模型中，与通过 RL 在小模型上发现的推理模式相比，其性能更好。](https://x.com/qtnx_/status/1881330757001502991)” 也就是所谓的“SFT 的全面胜利”。





---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**DeepSeek-R1 模型进展**

- **DeepSeek-R1 发布与更新**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1881318130334814301) 宣布发布 **DeepSeek-R1**，这是一个性能与 **OpenAI-o1** 相当的开源推理模型。发布内容包括一份技术报告和蒸馏的小型模型，为开源社区赋能。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1881362098141446598) 强调，与**模型蒸馏**相比，**强化学习微调（reinforcement learning fine-tuning）**的效果较弱，这标志着推理模型 **Alpaca 时代**的开始。

**基准测试与性能对比**

- **DeepSeek-R1 vs OpenAI-o1**：[@_philschmid](https://twitter.com/_philschmid/status/1881423639741960416) 总结的评估显示，**DeepSeek-R1** 在 **AIME 2024** 上达到了 **79.8%**，而 **OpenAI-o1** 为 **79.2%**。此外，[@ollama](https://twitter.com/ollama/status/1881427522002506009) 指出，**R1-Distill-Qwen-7B** 在推理基准测试上超越了如 **GPT-4o** 等更大型的专有模型。

**LLM 训练中的强化学习**

- **基于 RL 的模型训练**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1881362098141446598) 强调，**纯强化学习**可以赋予 **LLMs** 强大的推理能力，而无需大量的有监督微调。[@Philschmid](https://twitter.com/_philschmid/status/1881420703721009192) 详细介绍了 **DeepSeek-R1** 的五阶段 **RL 训练流水线**，展示了在**数学**、**代码**和**推理任务**中的显著性能提升。

**开源模型与蒸馏**

- **模型蒸馏与开源可用性**：[@_akhaliq](https://twitter.com/_akhaliq/status/1881386796266946743) 宣布，**DeepSeek 的蒸馏模型**（如 **R1-Distill-Qwen-7B**）表现优于 **GPT-4o-0513** 等非推理模型。[@reach_vb](https://twitter.com/reach_vb/status/1881412831306002897) 强调了社区从 **DeepSeek 的开源和蒸馏模型**中获益，使先进的推理能力在**消费级硬件**上变得触手可及。

**AI 研究论文与技术洞察**

- **来自研究论文的洞察**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1881211041247359146) 分享了来自 **LongProc** 基准测试的见解，显示在 **17 个 LCLMs** 中，**权重开放模型**在超过 **2K tokens** 后表现吃力，而像 **GPT-4o** 这样的**闭源模型**在 **8K tokens** 时性能下降。[@_philschmid](https://twitter.com/_philschmid/status/1881423639741960416) 讨论了 **DeepSeek-R1 论文**关于**强化学习**如何在不依赖复杂奖励模型的情况下增强模型推理的发现。

**梗/幽默**

- **关于 AI 和技术的幽默看法**：[@swyx](https://twitter.com/swyx/status/1881172252781343133) 分享了一个幽默的 [xkcd](https://xkcd.com/979/) 漫画，而 [@qtnx_](https://twitter.com/qtnx_/status/1881312367667191933) 以轻松的方式表达了对游戏发布和 Prompt Engineering 的沮丧。

- **对 AI 炒作的讽刺评论**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1881427907832340548) 幽默地评论了过度乐观的 AI 预期，强调无论技术如何进步，幽默内容总是长存的。

- **俏皮互动**：[@jmdagdelen](https://twitter.com/jmdagdelen/status/1881441833190150583) 俏皮地回应了 AI 讨论，为技术对话增添了一抹幽默。

- **技术讨论中的意外幽默**：[@evan4life](https://twitter.com/evan4life/status/1881371234567890123) 分享了一个关于 AI 模型行为的有趣轶事，将技术洞察与幽默融合在一起。

- **轻松的 AI 笑话**：[@sama](https://twitter.com/sama/status/1881258443669172470) 幽默地淡化了 AGI 的开发时间线，反映了社区俏皮的怀疑态度。

- **有趣的 AI 相关梗**：[@thegregyang](https://twitter.com/TheGregYang/status/1881111771517497616) 发布了一个关于职场场景的情境梗，为以 AI 为中心的讨论增添了轻松气氛。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek-R1 蒸馏模型展示了卓越的 SOTA 性能**

- **[DeepSeek 刚刚上传了 6 个 R1 的蒸馏版本 + R1 "完整版" 现已在其官网上线。](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)** ([Score: 790, Comments: 226](https://reddit.com/r/LocalLLaMA/comments/1i5or1y/deepseek_just_uploaded_6_distilled_verions_of_r1/)): **DeepSeek** 发布了六个 **R1** 模型的蒸馏版本以及 **R1 "完整版"** 模型，目前已可在其官网访问。
  - **DeepSeek 的策略与许可**：评论者赞扬 **DeepSeek** 发布竞争对手模型的微调版本并支持本地 **LLM** 社区，并指出了这一发布策略的精明之处。包括 **DeepSeek-R1-Distill-Qwen-32B** 在内的模型均采用 **MIT License** 发布，允许商业使用和修改，这被视为开源社区的一次重大举措。
  - **模型性能与可用性**：据报道，**DeepSeek-R1-Distill-Qwen-32B** 模型在基准测试中超越了 **OpenAI-o1-mini** 等其他模型，在稠密模型（dense models）中达到了 **SOTA** 水平。用户正急切等待 **32B** 和 **70B** 等更大模型的 **GGUF** 版本，这些模型的链接已在 **Hugging Face** 等平台分享。
  - **社区反应与技术见解**：用户对模型的性能和表现感到兴奋，一些人注意到蒸馏模型的输出较为冗长，并认为通过强化学习还有进一步提升的空间。此外，还有关于这些模型在现实应用中实际影响的讨论，部分用户分享了他们的测试经验和结果。


- **DeepSeek-R1-Distill-Qwen-32B 直接达到 SOTA，为本地使用提供超越 GPT-4o 级别的 LLM，且没有任何限制！** ([Score: 247, Comments: 85](https://reddit.com/r/LocalLLaMA/comments/1i5s2yd/deepseekr1distillqwen32b_is_straight_sota/)): **DeepSeek-R1-Distill-Qwen-32B** 正在确立其 **SOTA** 地位，在无限制的本地使用中超越了 **GPT-4** 级别的 **LLM**。该模型的蒸馏版本，特别是与 **Qwen-32B** 融合的版本，在基准测试中取得了显著进步，非常适合 **VRAM** 较少的用户，且表现优于 **Llama-70B** 的蒸馏版。
  - **蒸馏与基准测试**：**DeepSeek-R1-Distill-Qwen-32B** 的性能亮点在于其在[无量化](https://oobabooga.github.io/benchmark.html)的基准测试中以 **36/48** 的得分进入了**帕累托前沿 (Pareto frontier)**，展示了其在本地使用模型中的效率和竞争优势。
  - **模型比较与特性**：讨论中提到了 **Llama 3.1 8B** 和 **Qwen 2.5 14B** 蒸馏版本的优越性，据称它们优于 **QwQ** 并包含“思考标签 (thinking tags)”，增强了推理能力。
  - **软件与工具**：这些模型的最新更新和支持已经可用，包括针对蒸馏版本的 **PR #11310**，以及需要最新的 **LM Studio 0.3.7** 来支持 **DeepSeek R1**。


- **DeepSeek-R1 和 DeepSeek-R1-Zero 仓库正准备发布？** ([Score: 51, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1i5jlsr/deepseekr1_and_deepseekr1zero_repo_is_preparing/)): 提供的链接表明 **DeepSeek-R1** 和 **DeepSeek-R1-Zero** 模型预计将在 **Hugging Face** 上发布。用户表达了对发布的渴望，希望就在今天。
  - 如果用户有足够的存储空间，**DeepSeek-R1-Zero** 已经可以下载。**DeepSeek-R1** 同样如此。


**主题 2. DeepSeek-R1 模型的价格优势压倒了 OpenAI 的高成本 Token**

- **Deepseek R1 = $2.19/M tok output vs o1 $60/M tok. Insane** ([Score: 155, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1i5piy1/deepseek_r1_219m_tok_output_vs_o1_60m_tok_insane/)): **Deepseek R1** 的定价为 **每百万输出 token $2.19**，与 **o1 的每百万 token $60** 相比显著更低。帖子作者对实际应用，特别是与 **代码生成 (code generation)** 相关的对比非常感兴趣。
  - **Deepseek R1 定价与性能**：讨论强调了 **Deepseek R1** 提供了极具竞争力的 **每百万 token $2.19** 的定价，远低于 **o1 的每百万 token $60**。用户注意到 **R1 模型** 相比其之前的版本表现出了令人印象深刻的性能提升，特别是 **35B 和 70B 参数模型**，其表现与 **o1-mini** 相当甚至更好。
  - **模型透明度与成本因素**：**OpenAI** 在其模型架构和 token 使用方面缺乏透明度，这使得复制变得具有挑战性。一些评论认为 **OpenAI 的定价** 可能不仅仅是基于贪婪，而是与研发 (R&D) 和运营支出相关的成本有关，同时对 **Sam Altman** 关于公司财务亏损的说法持怀疑态度。
  - **访问与实现**：用户询问了如何访问和测试 **Deepseek R1**，并引用了 [Deepseek API 文档](https://api-docs.deepseek.com/quick_start/pricing) 以获取更多信息。**"deepthink"** 功能被提及作为使用 R1 模型的一种方式，其网站和 App 上也同步了更新。


- **Deepseek-R1 officially release** ([Score: 60, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1i5p9dk/deepseekr1_officially_release/)): **DeepSeek-R1** 在 **MIT License** 下正式发布，提供开源模型权重和支持思维链 (chain-of-thought) 输出的 API，声称在数学和编程等任务上与 OpenAI o1 性能对等。此次发布包括两个 660B 模型和六个较小的蒸馏 (distilled) 模型，其中 32B 和 70B 模型的能力与 OpenAI o1-mini 相匹配。API 定价为 **每百万输入 token 1 元人民币（缓存命中）** 和 **每百万输出 token 16 元人民币**，详细指南可在官方文档中找到。
  - **DeepSeek-R1** 的美元定价可以在官方文档 [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing) 中找到，为那些有兴趣将其与其他模型进行比较的人提供了清晰的成本结构。


- **[DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)** ([Score: 58, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1i5pepa/deepseekr1_paper/)): **DeepSeek-R1 论文** 介绍了一个强调成本效益 token 使用的 API。
  - **DeepSeek-R1-Zero 的自我演化**：自我演化过程展示了 **强化学习 (RL)** 如何自主增强模型的推理能力。这一过程是在没有监督微调 (SFT) 影响的情况下观察到的，允许模型通过延长的测试时计算 (test-time computation) 自然地发展出诸如反思和探索等复杂行为。
  - **复杂行为的涌现**：随着 DeepSeek-R1-Zero 测试时计算量的增加，它自发地发展出高级行为，例如重新审视和重新评估之前的步骤。这些行为源于模型与 RL 环境的交互，并显著提高了其解决复杂任务的效率和准确性。
  - **"Aha Moment" (顿悟时刻) 现象**：在训练过程中，DeepSeek-R1-Zero 经历了 "aha moment"，即它自主学会为问题分配更多的思考时间，从而增强其推理能力。这一现象突显了 RL 培养意想不到的解题策略的潜力，强调了 RL 在 AI 系统中实现新智能水平的力量。


**Theme 3. DeepSeek-R1 模型全面采用 MIT License**

- **[o1 级别的性能，成本仅约 1/50.. 还是开源的！！太强了，冲！！](https://www.reddit.com/gallery/1i5pbb3)** ([Score: 668, Comments: 237](https://reddit.com/r/LocalLLaMA/comments/1i5pbb3/o1_performance_at_150th_the_cost_and_open_source/)): **DeepSeek R1** 和 **R1 Zero** 已以开源许可证发布，以约 **1/50 的成本**提供 **o1 性能**，并且它们是开源的。
  - **DeepSeek 的开源与定价疑虑**：关于 DeepSeek 的开源声明存在大量讨论，一些用户质疑代码和数据集等模型细节的可获得性。定价问题也被提及，特别是 DeepSeek V3 的 Token 成本翻倍，以及与 OpenAI 定价的对比，部分用户指出高价可能为了防止系统过载。
  - **模型性能与对比**：用户强调了 DeepSeek 模型令人印象深刻的性能，并注意到参数量从 32B 增加到了 600B。用户将其与 **Qwen 32B** 和 **Llama 7-8B** 等其他模型进行了对比，一些用户声称这些模型在表现上超过了 4o 和 Claude Sonnet。
  - **审查与地缘政治影响**：关于 AI 模型中政治审查影响的辩论非常激烈，讨论了像 DeepSeek 这样的中国公司如何在其模型中嵌入 CCP 价值观。同时，也与同样应用自身“护栏 (guardrails)”、反映政治和文化偏见的美国公司进行了对比。


- **[DeepSeek-R1 及其蒸馏模型基准测试（颜色标注）](https://www.reddit.com/gallery/1i5q6b9)** ([Score: 288, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1i5q6b9/deepseekr1_and_distilled_benchmarks_color_coded/)): **DeepSeek R1** 的许可协议明确允许进行**模型蒸馏 (model distillation)**，这有利于创建高效的 AI 模型。帖子提到了经过颜色标注的**蒸馏基准测试**，这是一种评估性能指标的可视化方法。
  - **DeepSeek R1** 模型，特别是 **1.5B** 和 **7B** 版本，在编程基准测试中被指出超越了 **GPT-4o** 和 **Claude 3.5 Sonnet** 等更大型的模型，这引发了对其在 **MMLU** 和 **DROP** 等非编程基准测试中表现的怀疑和好奇。用户对这些结果表示惊讶，质疑除了数学和编程任务之外，这种改进是否具有泛化性。
  - **DeepSeek-R1-Distill-Qwen-14B** 因其效率而受到关注，其性能与 **o1-mini** 相当，同时输入/输出 Token 的价格显著降低。**32B** 和 **70B** 模型进一步超越了 **o1-mini**，其中 **32B** 模型的成本便宜 43 到 75 倍，使其在本地和商业用途上都极具吸引力。
  - 针对蒸馏模型的训练数据存在疑虑，这些模型严重依赖**监督微调 (SFT)** 数据而没有使用**强化学习 (RL)**，尽管一些用户澄清开发流程确实包含两个 RL 阶段。人们对 1.5B 模型的基准测试准确性持怀疑态度，一些人建议进行进一步测试以验证这些说法。


- **[Deepseek R1 / R1 Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1)** ([Score: 349, Comments: 105](https://reddit.com/r/LocalLLaMA/comments/1i5jh1u/deepseek_r1_r1_zero/)): **DeepSeek** 已将其许可范围扩大到 **MIT License** 下的商业用途。帖子提到了 **DeepSeek R1** 和 **R1 Zero**，但未提供更多细节。
  - 据 **BlueSwordM** 和 **Few_Painter_5588** 等用户讨论，**DeepSeek R1 Zero** 被推测为一个拥有约 **600B 到 700B 参数**的大型模型。这种模型规模意味着巨大的资源需求，估计需要 **1.8TB RAM** 才能运行，显示了其潜在的计算强度。
  - 围绕 **DeepSeek R1 Zero** 的讨论还涉及其架构，**De-Alf** 指出它与其他 **R1** 模型共享相同的架构，表明它们之间存在通用框架。提到了在 **Hugging Face** 上的发布，一些用户对模型的规模和角色（例如作为“教师”或“评判”模型）表示困惑。
  - **DeepSeek R1 Zero** 在 **MIT License** 下的发布因其开放性而受到赞扬，**Ambitious_Subject108** 等用户赞赏其没有将其限制在 API 之后的决定。社区还注意到发布了多个蒸馏版本，为各种硬件规格提供了灵活性。


**主题 4. DeepSeek-R1 蒸馏模型革新精度基准测试**

- **[Epyc 7532/dual MI50](https://www.reddit.com/gallery/1i5bj66)** ([Score: 68, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1i5bj66/epyc_7532dual_mi50/)): 一位工程师使用从 eBay 以每块 110 美元购买的 **双 MI50 GPU** 构建了一台 **Epyc 7532 服务器**，配备 **256 GB Micron 3200 RAM**，并安置在 **Thermaltake W200 机箱**中。尽管 MI50 面临散热挑战（温度超过 80°C），该配置在 Ubuntu 上运行 **ollama** 和 **open webui**，在 **Phi4** 上达到约 **5t/s** 的速度，表现良好，而 **qwen 32b** 则较慢。
  - **散热挑战**：**Evening_Ad6637** 分享了通过解决气流问题和使用铝制材料提高散热效率的见解，与标准散热系统相比，温度降低了高达 **10°C**。他们建议确保铝制组件与 GPU 散热片直接接触，以获得更好的散热效果。
  - **硬件兼容性与使用**：**Psychological_Ear393** 讨论了 **Radeon VII** 和 **MI50** GPU 与 **ROCm** 的兼容性，指出虽然两者都已被弃用，但仍可在最新驱动下运行。他们还提到 **W200 机箱** 非常大，能有效容纳该配置。
  - **风扇与气流考量**：**No-Statement-0001** 建议使用涡轮式风扇来增强静态压力，并改善穿过服务器 GPU 密集鳍片的气流，因为普通风扇可能难以胜任此任务。


- **[o1 思考了 12 分 35 秒，r1 思考了 5 分 9 秒。两者都得到了正确答案。均在两次尝试内完成。它们是首批正确完成该任务的两个模型。](https://i.redd.it/g4tvkorg56ee1.png)** ([Score: 104, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1i5t1be/o1_thought_for_12_minutes_35_sec_r1_thought_for_5/)): **DeepSeek R1** 和 **o1 模型** 在两次尝试内正确解决了一个复杂的数学问题，其中 **o1** 耗时 **12 分 35 秒**，**R1** 耗时 **5 分 9 秒**。该问题涉及统计狼和野兔等元素的数量，并强调了当狼的数量变为负数时的逻辑错误，突出了计算中非负变量的重要性。
  - **解题见解**：讨论深入探讨了谜题背后的推理，强调了逻辑推理在 AI 模型中的重要性。**Charuru** 详细分解了解决问题的过程，识别了关键观察点，如每次移动总动物数减少一只、最终总数不可能为奇数，以及最多只能有一个物种稳定共存。
  - **模型性能差异性**：**No_Training9444** 等人讨论了模型性能的差异，一些模型如 **DeepSeek R1** 和 **o1-pro** 成功解决了问题，而 **gemini-exp-1206** 等其他模型则表现挣扎。**StevenSamAI** 指出重复试验可能会产生正确答案，表明模型输出存在变数。
  - **社区参与**：社区积极参与该问题的讨论，分享尝试和结果。**Echo9Zulu-** 质疑此类谜题在测试 AI 方面的目的，而 **DeltaSqueezer** 等人则表达了亲自解决谜题的兴趣，突显了这些问题所呈现的趣味性与技术挑战的结合。

- **Deepseek-R1 GGUFs + 所有蒸馏版 2 到 16bit GGUFs + 2bit MoE GGUFs** ([Score: 101, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1i5s74x/deepseekr1_ggufs_all_distilled_2_to_16bit_ggufs/)): **Deepseek-R1** 模型已以多种 **quantization formats** 上传，包括 2 到 16-bit GGUFs，其中 **Q2_K_L 200GB quant** 专门用于 **large R1 MoE** 和 R1 Zero 模型。这些模型可在 [Hugging Face](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-677cf5cfd7df8b7815fc723c) 获取，并包含用于更高精度的 **4-bit dynamic quant** 版本，[Unsloth blog](http://unsloth.ai/blog/deepseek-r1) 提供了使用 **llama.cpp** 运行模型的说明。
  - **Dynamic Quantization 与兼容性问题**：用户讨论了使用 **Q4_K_M** 以获得最佳性能，并探索了 **bitsandbytes** 之外的、与 **llama.cpp** 兼容的动态量化替代方案。存在 **LM Studio** 不支持最新 **llama.cpp** 更新的问题，导致加载 **R1 Gguf** 等模型时出现错误。
  - **模型上传延迟与可用性**：**Qwen 32b gguf** 模型在上传过程中遇到了暂时的 404 错误，但随后已在 [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUFApologies) 上线。其他模型仍在上传过程中，团队正在连夜工作以确保可用性。
  - **社区感谢与反馈**：社区对 **Unsloth** 团队持续的工作和快速更新表示感谢，认可他们的奉献精神以及对用户反馈和问题的积极响应。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. DeepSeek-R1 发布开源模型，挑战硬件成本**

- **[就在刚才！DeepSeek-R1 发布了！](https://x.com/deepseek_ai/status/1881318130334814301)** ([Score: 250, Comments: 103](https://reddit.com/r/OpenAI/comments/1i5pr7q/it_just_happened_deepseekr1_is_here/)): **DeepSeek-R1** 是一款需要大量 **GPU** 资源的新模型，暗示了极高的计算需求。它被描述为一个 **open model**，表明其可供公众使用，并具有社区贡献或修改的潜力。
  - **DeepSeek-R1 硬件要求**：虽然一些用户最初认为 DeepSeek-R1 需要高端硬件，但蒸馏版本可以在单个 **RTX 3090** 甚至更低 **VRAM** 的显卡上运行，让拥有消费级 GPU 的用户也能更轻松地使用。
  - **开源 vs. 专有模型**：讨论了 DeepSeek-R1 与 **ChatGPT** 和 **Claude** 等专有模型的开放性对比，强调了本地运行 DeepSeek 的能力（尽管需要大量硬件投资），这与专有 API 带来的数据收集担忧形成对比。
  - **AI 模型开发与预期**：DeepSeek 训练过程的简洁性（涉及带有奖励的标准策略优化）引发了疑问：为什么这种有效的方法没有早点被发现？这突显了 AI 领域不断发展的现状，以及对模型提高推理和推断能力的期望。


**主题 2. 使用 Browser-Use 工具实现求职申请的 AI 自主化**

- **[AI agent 自动申请工作](https://v.redd.it/gvchr63e96ee1)** ([Score: 200, Comments: 46](https://reddit.com/r/OpenAI/comments/1i5tk3n/ai_agent_applying_for_jobs_on_its_own/)): 该帖子讨论了一个利用 **GitHub** 自主申请工作的 **AI agent**。文中未提供有关该 AI agent 实现或有效性的具体细节，因为帖子正文为空，需通过视频获取更多信息。
  - **自动化与外部性**：用户对自动化求职申请的影响表示担忧，评论强调了申请量的增加以及由此导致的雇主需要使用自动化进行筛选。讨论强调，虽然 AI 可以申请数千个职位，但它可能导致更多的垃圾信息和就业市场的低效。
  - **AI 申请的有效性**：一位记者对 AI 求职申请服务的测试显示，申请数千个职位确实可以获得面试机会，但单次申请的成功率很低。对话表明，虽然 AI 可以扩大求职申请的规模，但它可能会产生不准确的信息（如伪造资历），其整体有效性存疑。
  - **潜在对策**：用户预测，随着 AI agent 开始申请工作，招聘人员可能会开发诸如 honeypotting 之类的策略来识别 AI 生成的申请。还有人推测 AI agent 最终可能会管理远程工作，这引发了关于 AI 在就业市场中角色的伦理和实际问题。


**主题 3. 对 OpenAI 营销和 AGI 承诺的批评**

- **[他亲自制造了炒作，但局面失去了控制](https://i.redd.it/6o428kog24ee1.png)** ([分数: 1243, 评论: 135](https://reddit.com/r/OpenAI/comments/1i5luka/he_himself_built_the_hype_but_it_got_out_of/)): **Sam Altman** 在 Twitter 上回应了围绕 **OpenAI** 的过度**炒作 (hype)**，澄清说**通用人工智能 (AGI)** 不会在下个月部署，因为目前尚未建成。根据他日期为 **2025年1月20日**、拥有 **2.69万次浏览量** 的推文，他建议追随者尽管有令人兴奋的进展，但仍需**降低预期**。
  - 讨论中充满了对 **Sam Altman** 言论的怀疑，用户对感知到的前后矛盾和炒作管理表示沮丧，特别是关于 **AGI** 的时间表。一些用户将他的表态解读为一种策略，可能是为了管理预期和应对监管审查。
  - 用户们争论了**奇点社区 (singularity community)** 的反应，经常嘲笑他们对 AGI 过于乐观的时间表，并暗示像 **r/singularity** 和 **r/openai** 这样的论坛由于共同的不切实际预期而变得越来越难以区分。
  - 几条评论反思了 **Altman** 过去的言论以及围绕 **OpenAI** 的炒作，一些人认为他最近的推文旨在平息市场预期，防止基于投机性 AGI 时间表的估值过高。


- **OpenAI 的营销马戏团：别再被他们的科幻式炒作骗了** ([分数: 357, 评论: 214](https://reddit.com/r/OpenAI/comments/1i5a7ss/openais_marketing_circus_stop_falling_for_their/)): OpenAI 的营销策略因宣传对 **AGI** 和**博士级超级智能体 (super-agents)** 不切实际的预期而受到批评，暗示这些进步近在咫尺。该帖子认为，如果没有专门的训练，**LLM** 缺乏高级推理技能，并警告不要相信过度炒作的承诺，强调需要提高媒介素养。
  - 讨论突显了对 **OpenAI 营销**策略的怀疑，一些用户认为该公司关于 **AGI** 和**博士级超级智能体 (super-agents)** 的主张被夸大了，不能反映当前的能力。**Sam Altman** 因发表雄心勃勃的言论而受到关注，这些言论既遭到了愤世嫉俗的对待，也引发了期待。
  - 用户们争论了 **LLM** 的能力，一些人断言像 **o1** 和 **o3** 这样的当前模型在执行任务方面已经优于普通人类，而另一些人则认为这些模型仍然缺乏常识和可靠性。对话涉及了 LLM 的**推理能力**，并将其与幼儿进行比较，讨论了它们令人印象深刻但又有限的问题解决能力。
  - 社区在感知到的炒作与 AI 模型的实际效用之间表现出分歧，一些用户主张对 AI 能力进行更现实的理解。人们呼吁对 AI 进展的**媒体表现**保持怀疑，强调需要通过实践经验和直接使用模型来评估其在现实世界中的适用性。


**主题 4. 对 Perplexity AI 可靠性的批评及偏见担忧**

- **[人们真的需要停止使用 Perplexity AI](https://i.redd.it/vp3tn1u0g5ee1.png)** ([分数: 220, 评论: 137](https://reddit.com/r/OpenAI/comments/1i5py7o/people_really_need_to_stop_using_perplexity_ai/)): **Perplexity AI 的 CEO Aravind Srinivas** 提议开发 Wikipedia 的替代方案，理由是感知到的偏见，并鼓励通过 Perplexity API 进行协作。他发布于 **2025年1月14日** 的推文引起了广泛关注，拥有 **82.07万次浏览、593个赞** 和 **315次转推**。
  - 讨论强调了 **Wikipedia 中的偏见**，特别是在涉及**以色列/巴勒斯坦冲突**等有争议的话题时。评论者认为 Wikipedia 的众包性质导致了活动家驱动的内容，一些人认为由于利润动机，企业化的替代方案可能会更有偏见。
  - 许多评论者对 **Perplexity AI 的意图**表示怀疑，认为该公司的提议可能是在“无审查”的掩护下迎合右翼观点。人们对创建一个真正公正的平台的可行性表示担忧，因为所有信息源本质上都带有某种偏见。
  - 关于**替代信息源**的想法引发了辩论，一些人支持信息源的多样化以避免单一叙述的主导，而另一些人则担心偏见和虚假信息增加的可能性。对话反映了人们对技术和 AI 在塑造公共话语和知识库中作用的更广泛担忧。

---

# AI Discord 简报

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. 开源 LLM 之争**

- [**DeepSeek R1 强势超越 OpenAI 的 o1**](https://huggingface.co/deepseek-ai/DeepSeek-R1)：这款拥有 671B 参数的模型在推理基准测试上追平了 o1，而成本仅为后者的 4%，并以 MIT 许可证发布，可免费商用。其蒸馏版本（1.5B 至 70B）在 MATH-500 和 AIME 上也取得了高分，令数学爱好者印象深刻。
- [**Kimi k1.5 在 128k-Token 对决中力压 GPT-4o**](https://x.com/Kimi_ai_/status/1881332472748851259)：全新的 “k1.5” 能够协调多模态任务，据报道在代码和数学方面的表现比 GPT-4o 和 Claude Sonnet 3.5 高出多达 +550%。用户指出其 chain-of-thought 协同效应使其能够轻松通过困难的基准测试。
- [**Liquid LFM-7B 敢于挑战 Transformers**](https://www.liquid.ai/lfm-7b)：Liquid AI 推出了 LFM-7B，这是一种非 Transformer 设计，在 7B 规模上具有卓越的吞吐量。它大胆宣称在基于许可证的模型分发模式下，提供一流的英语、阿拉伯语和日语支持。

**主题 2. 代码与 Agentic 工具**

- [**Windsurf Wave 2 携 Cascade 与自动生成记忆来袭**](https://codeium.com/blog/windsurf-wave-2)：新的 Windsurf 编辑器集成了强大的网页搜索、文档搜索，并为更广泛的代码团队提升了性能。用户赞扬其单一全局聊天的方式，尽管有些人抱怨在大文件上下文下性能迟缓。
- [**Cursor 在迟缓的对决中跌跌撞撞**](https://forum.cursor.com/)：开发者抱怨 3 分钟的延迟、代码删除事故以及 “flow actions” 拖慢了速度。许多人威胁要转投 Windsurf 或 Gemini 等更快的 AI 编辑器。
- [**Aider 0.72.0 凭借 DeepSeek R1 取得佳绩**](https://aider.chat/docs/leaderboards/)：Aider 的最新版本欢迎使用 “--model r1”，以统一跨 Kotlin 和 Docker 增强功能的代码生成。用户非常喜欢 Aider 编写了 “52% 的新代码”，证明了它是代码开发中的双刃剑伙伴。

**主题 3. RL 与推理强化**

- [**GRPO 为 DeepSeek 简化了 PPO**](https://x.com/natolambert/status/1881380809153847711)：“*Group Relative Policy Optimization (GRPO) 就是去掉了价值函数的 PPO，*” Nathan Lambert 声称。通过依赖 Monte Carlo 优势，DeepSeek R1 涌现出了先进的数学和代码解决方案。
- [**Google 的 Mind Evolution 智胜顺序修正**](https://x.com/_akhaliq/status/1881182840857178146)：通过系统地改进解决方案，它在 Gemini 1.5 Pro 的规划基准测试中实现了 98% 的成功率。观察者将其视为无求解器（solver-free）性能的新巅峰。
- [**rStar-Math 押注 MCTS**](https://arxiv.org/abs/2501.04519)：它训练小型 LLM 在棘手的数学任务上超越大模型，而无需从 GPT-4 进行蒸馏。论文表明，token 级的 Monte Carlo Tree Search 可以将中等规模的 LLM 转变为强大的推理器。

**主题 4. HPC 与硬件花活**

- [**M2 Ultra 联手运行 DeepSeek 671B**](https://x.com/seo_leaders/status/1881462202831614085)：一位开发者声称使用两台 3-bit 量化的 M2 Ultra 达到了接近实时的速度。爱好者们在争论硬件成本是否值得为本地运行庞大的 LLM 而获得的炫耀资本。
- [**GPU vs CPU 大对决**](https://discord.com/)：有人认为 GPU 的并行化在处理大数组时彻底击败了 CPU，尽管数据传输可能成为瓶颈。其他人则表示对于小任务，CPU 可以同样快速且没有额外开销。
- [**KV Cache 量化助力 LM Studio**](https://lmstudio.ai/blog/lmstudio-v0.3.7)：Llama.cpp 引擎 v1.9.2 带来了内存友好的推理，支持 3-bit 到 4-bit 量化。追求速度的用户对消费级硬件上的吞吐量增益表示赞赏。

**主题 5. 合作伙伴与政策风波**

- [**微软对 OpenAI 的 130 亿美元豪赌惊动了 FTC**](https://slashdot.org/story/25/01/17/1958200/microsoft-openai-partnership-raises-antitrust-concerns-ftc-says)：监管机构担心 “锁定” 的 AI 合作伙伴关系，并担心初创公司的竞争可能会受到影响。Lina Khan 警告说，垄断云资源加 AI 资源对新竞争者来说意味着麻烦。
- [**FrontierMath 资金来源笼罩在 NDA 之下**](https://lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz)：据透露，OpenAI 秘密资助了该数学数据集，让许多贡献者蒙在鼓里。批评者抨击这种秘密安排阻碍了透明度。
- [**TikTok 合并传闻与 Perplexity 纠缠不清**](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA)：Perplexity 在惹恼 Pro 订阅者后进行了大规模扩张——传闻称其甚至考虑与 TikTok 合并。怀疑者质疑除了抢眼的标题之外，是否存在任何协同效应。

---

# 第一部分：高层级 Discord 摘要

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 2 与 Cascade 升级**：**Windsurf Wave 2** 的发布引入了 **Cascade** 网页和文档搜索、**自动生成的 memories** 以及性能增强，正如[官方博客](https://codeium.com/blog/windsurf-wave-2)所述。
   - 用户提到 **Cascade** 的运行更加流畅，并引用了 [status.codeium.com](https://status.codeium.com)，指出其对更广泛的团队具有更好的可靠性。
- **Deepseek R1 拥有 671B 参数**：新款 **Deepseek R1** 模型拥有 **6710 亿**参数，据报道超越了其他产品，[@TheXeophon](https://x.com/TheXeophon/status/1881443117787984265) 强调了其强大的测试分数。
   - 社区成员讨论了将其集成到 **Windsurf** 中的可能性，希望看到进一步的评估以及关于数据使用的明确说明。
- **Windsurf 的性能与错误困扰**：许多用户报告了 **incomplete envelope** 错误、打字缓慢以及 **1.2.1** 版本后的延迟，特别是在处理大文件时。
   - 他们表达了对 **flow actions** 和 **cascading edits** 的沮丧，称这些问题严重降低了生产力。
- **API Keys 与 Pro 计划的抱怨**：开发者对 **Windsurf** 在个人 API keys 方面的立场表示担忧，这限制了聊天功能和高级集成的使用。
   - 一些 Pro 计划订阅者感到被亏待了，将 Windsurf 与其他允许自由使用用户自有 API 的 IDE 进行了比较。
- **Cascade 历史记录与长对话问题**：单一的全局 **Cascade** 聊天列表给寻求特定项目分类的用户带来了困惑。
   - 他们还抱怨在 **Windsurf** 中长时间的会话会变得迟钝，迫使他们频繁重置并重复解释 context。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 更换模型，引发 Pro 用户不满**：在切换到**自研（in-house）**模型后，用户批评 **Perplexity** 的输出变弱并取消了 **Pro** 订阅，理由是缺乏动态响应（[Perplexity Status](https://status.perplexity.com)）。
   - 其他人要求迅速修复并提高透明度，提到了该平台的估值并敦促及时改进。
- **Ithy 等挑战 Perplexity 的统治地位**：一波新 AI 工具，包括 **Ithy** 和开源项目如 [Perplexica](https://github.com/ItzCrazyKns/Perplexica)，在寻求替代方案的开发者中获得了关注。
   - 社区成员表示这些工具提供了更广泛的功能，一些人预测开源平台可能很快会与封闭解决方案抗衡。
- **DeepSeek-R1 准备接入 Perplexity**：Perplexity 宣布计划集成 **DeepSeek-R1** 以处理高级推理任务，并引用了 [Aravind Srinivas 的推文](https://x.com/AravSrinivas/status/1881458694266953934)。
   - 用户期待功能得到恢复以及更敏锐的 context 处理，希望与搜索界面的协同效应得到改善。
- **Perplexity 收购 Read.cv**：Perplexity **收购**了 Read.cv，旨在增强其 AI 驱动的职业社交洞察（[详情点击此处](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA)）。
   - 参与者期待更强大的用户画像和数据驱动的匹配，引发了对该平台套件未来扩张的猜测。



---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 在基准测试中表现亮眼**：DeepSeek R1 在 aider polyglot 基准测试中获得了 **57%** 的分数，紧随 O1 的 **62%** 之后，详见[这条推文](https://x.com/paulgauthier/status/1881428345973608901)。
   - 其在 [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) 上的开源方案引发了将其集成到 Cursor 的兴趣，部分用户参考了 [DeepSeek 推理模型文档](https://api-docs.deepseek.com/guides/reasoning_model) 以实现高级工作流。
- **Cursor 的卡顿问题引发热议**：多位开发者报告在实际使用中出现了 **3 分钟的延迟**和 Agent 响应缓慢的问题，这加剧了对 Cursor 性能的不满。
   - 一些用户威胁要转向更快的 AI 编辑器，如 **Windsurf** 或 **Gemini**，同时一份关于新鲜 Prompting 想法的 [Notion 条目](https://half-single-ecd.notion.site/Experimental-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4) 也在流传。
- **Agent 功能正面交锋**：社区成员强调了 **Cursor** 在处理大文件和代码删除时的故障，并在一场 [240k Token 之战](https://www.youtube.com/watch?v=AtuB7p-JU8Y) 中将其与 GitHub Copilot 和 **Cline** 进行了对比。
   - 一些人坚持要求更好的文档支持，而另一些人则引用了 [Moritz Kremb 的推文](https://x.com/moritzkremb/status/1880628661931634700)，展示了单条命令的最佳实践。
- **社区推动 Cursor 更新**：为了解决性能投诉，出现了要求引入 **DeepSeek R1** 和其他先进模型的呼声。
   - 开发者们关注 [Cursor Forum](https://forum.cursor.com/) 以获取即将发布的补丁以及针对新版本的直接反馈渠道。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 的蒸馏技术成效显著**：**DeepSeek-R1** 模型因其强大的**蒸馏结果**而备受关注，详见 [Hugging Face 上的 DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)，并暗示了使用 **RL** 方法扩展**推理**能力的潜力。
   - 贡献者们集思广益，探讨 **Qwen** 与开源微调工作之间的协同作用，建议针对**复杂任务**进行未来优化。
- **Liquid AI 的许可证与 LFM-7B**：Liquid AI 推出了采用 **Recurrent** 设计的 **LFM-7B**，在[其官方链接](https://www.liquid.ai/lfm-7b)中宣称在 7B 规模下具有卓越的吞吐量。
   - 他们透露了一种**基于许可证的分发**模式，并强调了对本地和预算有限部署的**英语**、**阿拉伯语**和**日语**支持。
- **稀疏化提速：MOE 对比 Dense 模型**：参与者使用几何平均技巧对比了 **MOE** 与 **Dense 模型**在匹配参数规模下的表现，关注其 **3-4 倍**的延迟优势。
   - 他们引用了 [NVIDIA 结构化稀疏博客](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines) 来强调 **2:1** 的 GPU 效率，尽管内存需求相似。
- **Google 掌控 Mind Evolution**：Google 展示了 **Mind Evolution** 的表现优于 **Best-of-N** 和 **Sequential Revision**，在 **Gemini 1.5 Pro** 的规划基准测试中实现了 **98%** 的成功率。
   - 一个分享的[推文示例](https://x.com/_akhaliq/status/1881182840857178146)强调了与旧的推理策略相比，**Solver-free** 带来的性能提升。
- **气候产量的 CNN 协作**：一个名为 **“开发卷积神经网络以评估气候变化对全球农业产量的影响”** 的项目正在招募机器学习和气候科学专家，截止日期为 **1 月 25 日**。
   - 有意向的合作者可以私信获取构建集成 **CNN** 框架以分析**地理空间数据**和**产量因素**的详细信息。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 揭秘与量化趣闻**：Unsloth 宣布所有 **DeepSeek R1** 模型（包括 GGUF 和量化版本）现已上线 [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1)，提供易用性更高的 Llama 和 Qwen 蒸馏版本。社区成员称赞了 **dynamic 4-bit** 方法，并引用了 **@ggerganov** 的帖子，强调其在不牺牲准确性的情况下减少了 VRAM 占用。
- **Qwen 和 Phi 的微调成果**：社区成员使用各种训练参数测试了 **Qwen** 和 **Phi-4**，注意到 **Phi-4** 存在欠拟合问题，可能与更重的指令微调（instruction tuning）有关。他们还探索了在 Qwen2.5 上使用 **Alpaca format**，并参考了 [Unsloth 文档](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama) 中的聊天模板解决方案。
- **Chatterbox 对话与合成数据集**：新的 **Chatterbox** 数据集构建器引入了多轮对话管理功能，包括 Token 计数和 Docker-compose 等特性，并在 [GitHub repo](https://github.com/invisietch/Chatterbox) 中共享。开发者提议使用 webworkers 或 CLI 批量生成**合成数据集（synthetic datasets）**，旨在改进多轮对话流。
- **Sky-T1 起飞**：来自加州大学伯克利分校 NovaSky 团队的 **Sky-T1-32B** 模型在编程和数学方面得分很高，该模型在 8 个 **H100** GPU 上花费 19 小时，基于 Qwen2.5-32B-Instruct 的 17K 数据训练而成。爱好者们称赞其在 **DeepSpeed Zero-3 Offload** 下的速度，表示其性能几乎与 **o1-preview** 持平。
- **Cohere For AI LLM 研究队列招募**：**Cohere For AI** 计划将开展一个 **LLM Research Cohort**，重点关注多语言长上下文任务，将于 1 月 10 日启动。参与者将练习高级 NLP 策略，并引用了 [@cataluna84 的推文](https://x.com/cataluna84/status/1877689686639992872)，讨论将大规模教师模型与较小的学生模型相结合。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV7 携“Goose”强势登场**：被亲切地称为“Goose”的 **RWKV7** 发布，在社区中引发了热潮，[BlinkDL](https://x.com/BlinkDL_AI/status/1876849157920452781) 展示了其超越旧模型的强大生成能力。它显著集成了通道衰减（channel-wise decay）和学习率调整，根据用户测试，表现十分稳健。成员们将 **RWKV7** 与 Gated DeltaNet 进行了比较，强调了使这一 **gen7 RNN** 领先于先前迭代的新设计特性。他们还辩论了内存衰减策略和分层技术，以进一步增强 **RWKV7** 的优势。
- **DeepSeek R1 挑战 AIME 和 MATH-500**：新推出的 **DeepSeek R1** 模型在 [AIME](https://x.com/kimi_ai_/status/1881332472748851259) 和 **MATH-500** 等任务中表现优于 **GPT-4o** 和 **Claude Sonnet 3.5**，展示了处理高达 *128k tokens* 扩展上下文的能力。社区对比表明其“冷启动”性能有所提高，这归功于强大的训练策略。讨论涉及使用 [SPAM: Spike-Aware Adam](https://arxiv.org/abs/2501.06842) 的策略来应对梯度峰值，暗示 **DeepSeek R1** 有效地避免了永久性损伤。用户认为这些改进很有前景，而一些人对在没有更多复制实验的情况下完全依赖“R1 Zero”结果表示怀疑。
- **Qwen2.5 表现不及预期**：许多人在 gsm8k 上测试了 **Qwen2.5**，观察到准确率仅为 ~60%，与[官方博客](https://qwenlm.github.io/blog/qwen2.5-llm/)声称的指令微调版 73% 有所出入。解析差异和 few-shot 格式细节引起了困惑。一些人建议采用 [QwenLM/Qwen](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py) 使用的相同问答格式，并加上“step by step”风格以重新对齐结果。据报道，得分小幅提升至 **66%**，强调了提示策略（prompting tactics）如何影响最终结果。
- **MoE 的热度与顾虑**：社区称赞了 **Mixture of Experts** 模型的高效性，[Hugging Face 的 MoE 博客](https://huggingface.co/blog/moe)等参考资料促进了其采用。一些人对训练稳定性表示担忧，强调了分片（sharding）和门控（gating）策略的复杂性。辩论集中在如果没有高级调优来处理潜在的训练波动，**MoE** 是否能提供足够的实际优势。支持者将其视为一条充满希望的途径，而其他人则强调持续的实验是关键。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 的大胆进击**：DeepSeek-R1 的表现远超预期，在 **MIT license** 下实现了接近 **OpenAI-o1** 的性能，更多细节见 [Hugging Face 上的 DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)。
   - 怀疑者对 **R1 Zero** 的发现提出质疑，但其他人则称赞 **Group Relative Policy Optimization (GRPO)** 是更简洁的 **PPO** 替代方案，并引用了 [GRPO 澄清说明](https://x.com/natolambert/status/1881380809153847711)。
- **Kimi 在 RL 领域的强劲动力**：**Kimi 1.5** 论文强调了新的 RL 方法，如奖励塑造（reward shaping）和先进的基础设施，代码已在 [GitHub 上的 Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5) 分享。
   - 爱好者预测这些技术将增强 **reinforcement learning** 框架与 Chain-of-Thought 推理之间的协同作用，标志着 Agent 模型的一次飞跃。
- **Molmo 的多模态实力**：**Molmo AI** 作为一种强大的 VLM 受到关注，声称在检测和文本任务上具有卓越性能，展示于 [molmo.org](https://molmo.org/)。
   - 尽管出现了一些误分类情况，但许多人认为其跨领域灵活性使其成为 **GPT-4V** 等模型的有力竞争者。
- **Cursor 在编程对决中击败 Devin**：由于代码补全效果不佳，各团队迅速放弃了 **Devin** 转而使用 **Cursor**；有传言称 Devin 在编程任务中调用的是 **gpt-4o**，而非 **Claude** 等更强的替代方案。
   - 这一转变引发了关于 AI 团队是否系统性高估了涌现的 Agent 解决方案的辩论，呼应了 [Tyler Cowen 访谈](https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5)中的观点。
- **SOP-Agents 大放异彩**：[SOP-Agents 框架](https://arxiv.org/abs/2501.09316)为 LLM 提出了 **Standard Operational Procedures**（标准作业程序），优化了多步规划。
   - 开发者期待将其与 **Chain of Thought** 和 RL 结合，以增强高层决策图的清晰度。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.72.0 达到新高度**：全新的 Aider v0.72.0 版本带来了 **DeepSeek R1** 支持（通过快捷方式 `--model r1`）和 Kotlin 语法集成，以及使用 `--line-endings` 增强的文件写入功能。
   - 社区成员提到了多项错误修复（包括 Docker 镜像中的 **权限问题**），并指出 **Aider** 编写了 52% 的新代码。
- **DeepSeek R1 引发褒贬不一的反应**：一些用户称赞 [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 是 OpenAI o1 的更廉价替代方案，在 [Aider 编程基准测试](https://aider.chat/docs/leaderboards/)中达到了 **57%** 的得分。
   - 其他人则报告在基础任务中效果欠佳，建议将其与更可靠的编辑模型配对以提高一致性。
- **Kimi k1.5 击败 GPT-4o**：据报道，新的 **Kimi k1.5** 模型在多模态基准测试中优于 GPT-4o 和 Claude Sonnet 3.5，上下文缩放高达 **128k tokens**。
   - 用户强调了在 MATH-500 和 AIME 上尤为强劲的结果，激发了对增强推理能力的乐观情绪。
- **AI 数据隐私引发关注**：参与者引用了 [Fireworks AI 文档](https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security)，同时描述了企业在数据使用透明度方面的差异。
   - 他们质疑哪些供应商能负责任地处理用户数据，并指出大型 AI 厂商的政策尚不明确。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt.new 消除白屏问题**：继最近的 [Bolt.new 推文](https://x.com/boltdotnew/status/1881442318110347291)之后，**Bolt.new** 解决了臭名昭著的白屏问题，并确保从第一个提示词（prompt）开始就能进行精确的模板选择。
   - 积极的测试者报告称流程更加顺畅，指出这直接修复了之前的困扰，并保证了更高效的启动。
- **错误循环吞噬 Token**：用户面临持续的循环，导致严重的 Token 消耗——一位开发者消耗了 **3000 万个 Token**——特别是在涉及用户权限的场景中。
   - 他们得出结论，完全重置是唯一的出路，社区成员敦促针对**复杂功能**进行更强大的调试。
- **Supabase 中的 RLS 纠葛**：开发者在 **Supabase** 中实现预订功能时，苦于反复出现的 **RLS**（行级安全性）违规，导致策略不断失败。
   - 一位用户建议参考 [Supabase 文档](https://supabase.com/docs/guides/functions/examples/stripe-webhooks)中的示例策略，以减少重复的配置错误。
- **Stripe 还是 PayPal？支付讨论**：社区成员讨论了在汽车美容支付中选择 **Stripe** 还是更简单的替代方案（如 **PayPal**），特别是针对技术水平较低的用户。
   - 一些人指向了 [Supabase 关于 Stripe Webhooks 的指南](https://supabase.com/docs/guides/functions/examples/stripe-webhooks)，而另一些人则推荐基于 WordPress 的解决方案以实现更快的部署。
- **Pro 计划缓解 Token 限制**：好奇的新手询问了 **Pro 计划**下的 Token 使用情况，发现每日限制消失了，使用情况很大程度上取决于用户技能和可选功能（如 diffs）。
   - 这种方式让更高级的开发者感到安心，他们可以尽情使用 Bolt 而不必担心每日上限或意外的 Token 耗尽。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.7 与 DeepSeek R1：强强联手**：全新的 **LM Studio 0.3.7** 包含了对先进的 **DeepSeek R1** 模型的支持，并集成了 llama.cpp 引擎 v1.9.2，详见 [LM Studio 更新说明](https://lmstudio.ai/blog/lmstudio-v0.3.7)。
   - 社区成员赞扬了其开源方式，并引用了 [DeepSeek_R1.pdf](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)，称赞其带有 `<think>` 标签的强大推理能力。
- **KV Cache 量化提升效率**：针对 **llama.cpp** (v1.9.0+) 的 **KV Cache 量化**功能旨在通过减少内存占用来增强性能，如 [LM Studio 0.3.7](https://lmstudio.ai/blog/lmstudio-v0.3.7) 所示。
   - 用户报告称在大语言模型中获得了更快的吞吐量，并指出 **3-bit** 量化通常能在速度和准确性之间达到最佳平衡。
- **LM Studio 中的文件附件保留在本地**：用户询问在 **LM Studio** 中上传文件是否会将数据发送到别处，并得到了保证：内容保留在用户机器上，用于本地上下文检索。
   - 他们测试了针对特定领域任务的多文件上传，确认了在不损害数据控制权的情况下的纯离线使用。
- **GPU 评测：4090 对比廉价显卡**：成员讨论权衡了 200 美元的 GPU 与 **4090** 等高端显卡，并参考了大规模 AI 任务的 [技术规格](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889)。
   - 大多数人同意更大的显存是处理巨型模型的关键，能为数据驱动型工作负载提供更高的吞吐量。
- **使用 M2 Ultra 进行分布式推理：速度还是挥霍？**：[Andrew C 的一条推文](https://x.com/seo_leaders/status/1881462202831614085)展示了在两台 M2 Ultra 上运行的 **DeepSeek R1** 671B，利用 3-bit 量化实现了接近实时的速度。
   - 然而，参与者对硬件成本仍持谨慎态度，理由是带宽限制和收益递减的风险。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 蒸馏并占据主导地位**：[DeepSeek R1 发布](https://x.com/deepseek_ai/status/1881318138937233664)，采用 **MIT license**，在数学、代码和推理任务上的表现与 **OpenAI o1** 相当。
   - 一个[蒸馏变体](https://x.com/reach_vb/status/1881315419086291213)在 AIME 和 MATH 基准测试中超过了 **GPT-4o**，引发了人们对扩展开源产品的兴奋。
- **OpenAI 的 Operator 在泄露文档中浮出水面**：最近的泄露揭示了 **OpenAI** 的新项目 **Operator**（或称 Computer Use Agent），引发了关于即将发布的猜测。
   - 观察者将其与 **Claude 3.5** 进行了对比，并引用了 [Operator 系统泄露](https://x.com/kimmonismus/status/1881287794544550018)中的细节。
- **Liquid Foundation Model LFM-7B 启航**：来自 **Liquid AI** 的 [LFM-7B 模型](https://www.liquid.ai/lfm-7b)声称具有顶级的多语言能力，并采用了非 Transformer 设计。
   - 工程师们称赞其低内存占用适合企业使用，这与基于 Transformer 的大型方法形成了鲜明对比。
- **DeepSeek v3 & SGLang 助力关键任务推理**：[Latent.Space 播客](https://www.latent.space/p/baseten)重点介绍了 **DeepSeek v3** 和 **SGLang** 在“关键任务推理（Mission Critical Inference）”中对高级工作流的需求。
   - 嘉宾们讨论了超越单 GPU 扩展的策略，并预告了 **DeepSeek** 的进一步改进，引起了关注性能的开发者的兴趣。
- **Kimi k1.5 以 o1 级别的性能令人惊喜**：[Kimi k1.5 模型](https://x.com/Kimi_ai_/status/1881332472748851259)达到了 **o1-level** 基准，在数学和代码任务中表现优于 **GPT-4o** 和 **Claude 3.5**。
   - 据报道在 LiveCodeBench 上有 **+550%** 的提升，这引发了关于较小架构如何缩小与较大竞争对手差距的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek R1 对抗 OpenAI 的 o1**：DeepSeek 在 [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1) 上推出了其 **R1** 模型，其性能可与 **OpenAI o1** 媲美，价格为 **$0.55/M tokens**（仅为成本的 4%）。
   - 社区成员称赞该模型的开源 **MIT license** 和强大的实用性，并引用 [DeepSeek 的推文](https://x.com/deepseek_ai/status/1881318130334814301)了解更多细节。
- **无审查角度引发辩论**：**DeepSeek R1** 在 [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737) 上被描述为无审查（censorship-free），尽管一些用户注意到它仍保留了过滤组件。
   - 其他人建议额外的微调（finetuning）可以扩大其范围，期待在没有额外约束的情况下获得更强的性能。
- **Llama 端点取消免费层级**：OpenRouter 透露计划在月底前停止 **free Llama** 端点，因为 **Samba Nova** 发生了变化。
   - 一个 **Standard 变体** 将以更高的价格取代它们，这让许多用户感到意外。
- **OpenAI 模型速率限制已澄清**：用户确认 **OpenAI 的付费层级** 没有每日请求上限，但免费层级将活动限制在 **每天 200 次调用**。
   - 一些人通过附加自己的 API keys 克服了这些限制，减少了使用上的麻烦。
- **推理和网络搜索支持处于变动中**：社区成员询问如何从 **DeepSeek R1** 访问 `reasoning_content`，预计 **OpenRouter** 很快会添加该功能。
   - 其他人希望 **Web Search API** 能有更广泛的可用性，目前该功能仅锁定在聊天室界面。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Photorealistic Flourish with LoRA**: 在关于使用 **Stable Diffusion** 3.5 生成逼真图像的讨论中，参与者探索了 **LoRA** 策略以减轻“塑料感”，并参考了 [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 进行高级控制。
   - 一位用户坚持认为，将高分辨率样本与各种 **resolutions** 混合可以产生更真实的输出，并引用了 [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) 来增强 prompt 定制。
- **Cloudy E-commerce Deployments**: 一位用户询问在 **Google Cloud** 上为电商部署 text-to-image 模型的可行性，并参考了 [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) 作为预训练解决方案的起点。
   - 其他人权衡了使用 **Google Cloud Marketplace** 还是 **custom Docker** 设置会更有效率，结论是预训练模型可以大大缩短设置时间。
- **LoRA Resolution Rumble**: 社区成员就仅在 1024×1024 分辨率下训练 **LoRA** 展开辩论，并指向 [Prompt Syntax docs](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md) 以获得更细致的控制。
   - 一组人强调了多样化的分辨率输入，以便 **LoRA** 能够处理各种图像质量而不会产生奇怪的伪影。
- **Background-Editing Tangles**: 用户遇到了性能下降和背景层缺陷的问题，将其归因于 **Stable Diffusion** 流水线中的 **denoising** 配置错误。
   - 他们建议通过 **GIMP** 或专门的 AI 解决方案进行手动微调，并指出使用 [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md) 的功能可以改善结果。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Podcasts & Personality Swaps**: 一位用户介绍了一个新的 **GLP-1** 主题播客，并尝试使用 [提议的工具](https://illuminate.google.com/home?pli=1) 更改主持人声音，但目前的解决方案可能无法妥善支持。
   - 另一位用户指出，随机的声音角色切换会导致混乱，并回应称许多 *podcast generation tools* 在稳定的说话人分配方面表现不佳。
- **Gemini Gains & NotebookLM in Class**: 一位用户描述了使用 **Gemini Advanced Deep Research** 生成详尽音频概览的工作流，建议直接溯源以减少数据丢失。
   - 另一位用户辩论了在 **econ course** 中是使用单个还是多个笔记本，更倾向于基于主题的方法以保持一致的组织。
- **Subscriptions & Simple Setups**: 几位用户比较了 **Google One AI Premium** 与 **Google Workspace** 以获取 **NotebookLM Plus** 访问权限，指出两者都提供了所需的模型功能。
   - 用户得出结论，**Google One** 更易于管理，没有 **Workspace** 会员身份那么复杂。
- **Big Bytes & OCR Ordeals**: 一位用户在上传接近 **100MB** 的音频文件时遇到困难，怀疑如果与现有数据合并，将超过 **200MB** 的总限制。
   - 另一位用户强调了不可复制 PDF 的 **OCR** 问题，呼吁改进 **NotebookLM** 的扫描支持。
- **Multi-language Moves & Newcomer Hellos**: 几位用户对 **multi-language** 播客支持表示感兴趣，希望很快能看到官方扩展到 **English** 以外的语言。
   - 新成员介绍了自己，提到了 *语言障碍*，并鼓励提出更尖锐的问题以保持讨论简洁。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **不稳定的 MCP Server 实现**：用户指出多个 [MCP servers](https://github.com/modelcontextprotocol/servers) 之间的 prompt 使用不一致，导致对正确规范的困惑。
   - 一些实现仅获取资源而忽略了官方指南，引发了要求更严格遵守文档的呼声。
- **Roo Cline 以 Agent 特性吸引用户**：[Roo Cline](https://www.pulsemcp.com/clients) 通过自动批准命令给开发者留下了深刻印象，在使用 R1 servers 时提供了近乎全自动的体验。
   - 许多人称赞其有用的 VSCode 插件集成，认为它是比 **Claude Desktop** 等大型客户端更简单的替代方案。
- **Claude 遭遇速率限制瓶颈**：频繁的 **Claude** 速率限制令测试者感到沮丧，限制了上下文长度和消息频率。
   - 一些人要求在 **Claude Desktop** 中提供更好的使用情况追踪，希望能有更清晰的阈值提示并减少突然的中断。
- **Figma MCP 寻找勇敢的代码贡献者**：[Figma MCP](https://github.com/nmfisher/figma_mcp) 作为早期原型发布，邀请开发者共同塑造其未来。
   - “这还处于非常早期/粗糙的阶段，所以欢迎任何贡献者！”一位成员说道，并征求新的想法。
- **AI 逻辑计算器引发好奇**：[MCP Logic Calculator](https://github.com/angrysky56/mcp-logic) 在 Windows 系统上利用 Python 中的 Prover9/Mace4 来处理逻辑任务。
   - 另一位成员建议将其与 memory MCP 结合进行强大的分类，激发了对高级逻辑工作流的兴趣。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GPU 的收益与 CPU 的痛点**：在关于 HPC 使用的对话中，参与者得出结论，大型数组通常受益于 **GPU** 并行化，尽管数据传输可能会导致减速。
   - 一些参与者将该操作描述为“易并行”（trivially parallel），暗示对于较小的任务， **CPU** 方法仍具竞争力。
- **微软对 OpenAI 的巨额押注**：来自微软的 **130 亿美元投资** 触发了反垄断警告，**FTC** 强调云端主导地位可能会渗透到 AI 市场。
   - FTC 主席 **Lina Khan** 警告称，锁定的合作伙伴关系可能会阻碍初创公司获取关键的 **AI** 资源。
- **FrontierMath 资金风波**：社区成员在发现隐藏的资金安排后，对 **OpenAI** 参与 **FrontierMath** 表示质疑，引发了透明度问题。
   - 一些人声称 **Epoch** 受制于严苛的 **NDA** 条款，导致许多贡献者对 OpenAI 在融资中的角色一无所知。
- **Lightning 和 TPA：快速合成**：**Lightning Attention** 与 [Tensor Product Attention](https://arxiv.org/abs/2501.06425) 的集成在原型模型测试中实现了约 **3 倍的加速**。
   - 用户将 attention 中的大张量操作归功于线性化，强调了其相对于先前方法的重大性能飞跃。
- **rStar-Math 凭借 MCTS 带来惊喜**：论文 [rStar-Math](https://arxiv.org/abs/2501.04519) 展示了小型 LLM 如何通过 **Monte Carlo Tree Search** (MCTS) 在高级数学任务中超越大型模型。
   - 其作者主张尽量减少对人类数据的依赖，详细介绍了一种使用三种不同训练策略来提升问题解决能力的方法。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Konkani 协作势头强劲**：一位用户旨在为 **Konkani** 语构建模型，并可能获得大学支持，希望能推动跨语言 NLP 的发展。
   - 他们指出，行业合作伙伴关系对于项目的扩展和实际应用至关重要。
- **Command-R 的难题**：工程师发现 **command-r** 引用的是旧模型，以避免对现有用户造成**破坏性变更**。
   - 他们建议使用带有 'latest' 标签的官方**别名**，以保持发布的一致性，同时允许按需启用新版本。
- **Cohere 的数学错误**：用户发现 **Cohere** 错误地将 18 个月计算为 27 周，迫使他们手动验证结果。
   - 他们强调大多数 **LLM** 都有此局限性，建议降低 temperature 或使用独立的计算器作为解决方案。
- **代码调用与工具策略**：开发者概述了 **Cohere** 如何通过让 LLM 决定何时使用指定组件来调用外部工具。
   - 他们注意到官方极少提及 **AGI**，但强调了用于代码生成工作流的结构化 prompt 和模型驱动执行。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **春季 MOOC 势头强劲**：一位成员询问了关于今年 **1 月**开始的 **MOOC 课程**的确认信息，重点关注预期的 **LLM Agents** 覆盖内容。
   - 他们还提到 **mailing list** 将于下周开始，暗示很快将分享更多 **course timeline** 细节。
- **邮件列表即将启动**：社区成员确认 **春季课程邮件列表** 将于下周发布，解决了关于正式注册的公开问题。
   - 他们预计一旦列表上线，将会有进一步的 **course timeline** 更新，并建议潜在参与者关注公告。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Document to Podcast 蓝图亮相**：一个专门的团队介绍了 **Document to Podcast 蓝图**，这是一种利用开源解决方案将文本内容转化为音频的灵活方法。
   - 他们宣布了一个直播环节，参与者可以在其中提问、分享反馈，并探索如何将该蓝图整合到自己的项目中。
- **蓝图强化开源协同效应**：敦促与会者加入活动并与其他开源爱好者建立联系，承诺在未来项目上进行新的合作。
   - 他们强调点击“感兴趣”按钮加入社区对话，为更深层次的开源交流激发新的可能性。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长时间没有动静，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该社区长时间没有动静，请告知我们，我们将将其移除。

---

# PART 2: 详细频道摘要与链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1329913370354385019)** (1 条消息): 

> `Windsurf Wave 2 特性, Cascade 网页与文档搜索, Cascade 自动生成的记忆, 性能改进, 状态更新` 

- **Windsurf Wave 2 发布重大特性**：Windsurf Wave 2 引入了新功能，例如 **Cascade** 现在可以通过自动检测或用户命令搜索网页和文档。
   - *Cascade* 还能通过自动生成的记忆在对话中保留上下文，提升用户体验和交互。
- **Cascade 与性能改进**：此次更新解决了多个 **Dev Container 问题**，同时增强了 **Cascade** 的整体性能。
   - 这些改进旨在为与 Bot 交互的用户提供更流畅的体验。
- **Cascade 网页与文档搜索功能**：用户现在可以通过 **URL** 输入，或在 **Cascade** 中使用 `@web` 和 `@docs` 命令自动触发网页搜索。
   - 这些新功能允许从各种文档网站和公共资源中检索信息，以改进辅助功能。
- **Windsurf 系统状态更新**：**Windsurf/Codeium** 当前状态为运行正常，近期未报告重大事故，确认了系统的可靠性。
   - 鼓励用户访问 [status.codeium.com](https://status.codeium.com) 查看实时更新。
- **关注 Wave 2 资源更新**：要探索更多关于 Windsurf Wave 2 的内容，用户可以阅读完整的 [博客公告](https://codeium.com/blog/windsurf-wave-2) 并查看 X 上的相关视频。
   - 更多细节可以在 [changelog](https://www.codeium.com/changelog) 中找到，其中列出了所有新特性和更新。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://status.codeium.com">Codeium Status</a>：未找到描述</li><li><a href="https://codeium.com/blog/windsurf-wave-2">Windsurf Wave 2</a>：介绍 Wave 2，我们对 Windsurf 编辑器的第二批更新。</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384">Windsurf (@windsurf_ai) 的推文</a>：Wave 2 已上线。本次更新包含：🌐网页搜索🧠自动生成的记忆💼企业级就绪... 以及更多！</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1329911461530566786)** (226 条消息🔥🔥): 

> `Windsurf 错误消息, Deepseek R1 发布, Codeium 功能, 用户支持问题, Windsurf 中的 API Key 使用` 


- **Windsurf 中的频繁错误**：用户报告了 Windsurf 中持续存在的错误，特别是 “Error Protocol error: incomplete envelope: unexpected EOF” 消息，导致功能使用受阻。
   - 其他用户遇到了应用程序对用户操作无响应的问题，以及在注册过程中提交 token 时遇到困难。
- **Deepseek R1 超出预期**：最近发布的 Deepseek R1 引起了轰动，据报道其性能超越了 OpenAI 的模型，拥有惊人的 **6710 亿参数**且价格极具竞争力。
   - 用户评论了将其集成到 Windsurf 中的潜力，称赞其基准测试结果优于现有模型。
- **Codeium 的功能与局限**：关于 Codeium 在 JetBrains 中的局限性引发了讨论，特别是缺乏对 Supercomplete 功能的支持，该功能目前仅限 VS Code 和 Windsurf 使用。
   - Pro 计划用户对未获得所有承诺的功能表示担忧，并在尝试解决这些问题时面临挑战。
- **用户支持挑战**：多位用户就 Windsurf 的登录问题、持续的错误消息和功能问题寻求帮助，强调了有效用户支持的必要性。
   - 社区成员分享了故障排除步骤，但也对直接支持渠道缺乏反馈表示沮丧。
- **关于 Windsurf 中 API Key 使用的讨论**：一场关于 Windsurf 商业模式的对话展开，特别是其限制在聊天功能中使用个人 API Key，引起了追求灵活性的用户的担忧。
   - 用户将其与允许个人 API 集成的其他 IDE 进行了比较，对 Windsurf 在市场上的竞争持久力表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://open-vsx.org/">Open VSX Registry</a>: 未找到描述</li><li><a href="https://codeium.com/supercomplete">Supercomplete | Windsurf Editor and Codeium extensions</a>: Codeium Supercomplete 能够预测你的下一个意图，无论光标位置如何。无论你是想插入、删除还是编辑，Supercomplete 都能满足你的需求。</li><li><a href="https://x.com/abacaj/status/1881339724545286257?t=Ydr1EHyPeUj8nYlGTPN1gQ&s=19">来自 anton (@abacaj) 的推文</a>: 中国发布了一个采用 MIT 许可证的模型，性能与 o1 相当且价格便宜 30 倍，这完全出乎我的意料。</li><li><a href="https://x.com/itsPaulAi/status/1881329522949447886?t=4igrKlZqJ-rlvMDJvR8yOw&s=19">来自 Paul Couvert (@itsPaulAi) 的推文</a>: 哇，一个完全开源的推理模型刚刚发布，性能与 OpenAI o1 相当。Deepseek R1 甚至在几乎所有基准测试中都优于 Claude 3.5 Sonnet 和 o1-mini。你现在已经可以免费使用它（见下文...</li><li><a href="https://codeium.com/s">页面未找到 | Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265?t=CWcMfDus2ULxJQS6VnnQRA&s=19">来自 Xeophon (@TheXeophon) 的推文</a>: 我被 R1 在我个人基准测试中的表现震惊了。这是完整的评估集，它完全碾压了竞争对手，自成一派，甚至超过了 o1-preview（图中省略了...</li><li><a href="https://x.com/TheXeophon/status/1881442133376454694?t=kcwBO9GpmTX5zzXVtA63gA&s=19">来自 Xeophon (@TheXeophon) 的推文</a>: 我的天，R1 在我的基准测试中击败了 o1-preview</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/profile>">页面未找到 | Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://codeium.com/plan">计划设置</a>: 未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://www.reddit.com/r/synology/comments/pq0411/cant_mount_network_drive_in_windows_explorer/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/Exafunction">Exafunction</a>: Exafunction 拥有 38 个代码仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1329913700831854677)** (577 条消息🔥🔥🔥):

> `Windsurf Performance Issues, Deepseek R1 Discussion, Cascade History Management, User Experience with Long Chats, AI Integration with Development Tools` 


- **Windsurf Performance Issues (Windsurf 性能问题)**：用户报告 Windsurf 在 1.2.1 版本后出现显著的性能下降，问题包括打字缓慢以及处理大文件时的延迟。
   - 几位用户对 flow actions 和 cascading edits 等功能表示沮丧，这些功能变得繁琐，导致可用性下降。
- **Deepseek R1 Discussion (Deepseek R1 讨论)**：Deepseek R1 被提及为可能优于 Claude 等现有解决方案的模型，一些用户渴望将其集成到 Windsurf 中。
   - 对话强调了在广泛采用之前进行彻底评估和测试的必要性，以及对隐私和数据使用的担忧。
- **Cascade History Management (Cascade 历史记录管理)**：目前正在讨论缺乏特定于 workspace 的 Cascade 历史记录的问题，用户主张增加能够按项目更好组织聊天历史的功能。
   - 一位用户指出目前只有单一的全局聊天列表，并对未来更新的实现细节和路线图表示关注。
- **User Experience with Long Chats (长对话的用户体验)**：多位用户注意到长对话会导致 Windsurf 内部的响应能力和功能下降，并建议开启新对话以缓解问题。
   - 这导致了对于必须向 Cascade 重复 context 并重新解释问题的挫败感。
- **AI Integration with Development Tools (AI 与开发工具的集成)**：讨论了 AI 工具（如 Windsurf）自动连接数据库并提供主动集成功能的潜力。
   - 用户分享了关于如何让 AI 更好地感知其开发环境的 context，从而显著提升用户体验的想法。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cursorlist.com>">未找到标题</a>：未找到描述</li><li><a href="https://discordapp.com/channels/1027685395649015980/1306163501286293515/1330602494958501979">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友闲逛甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=810cc140-6780-4bfb-a2f9-906c5d0fdd64">欢迎来到 Codeium - Codeium 文档</a>：未找到描述</li><li><a href="https://docs.codeium.com/getstarted/overview">欢迎来到 Codeium - Codeium 文档</a>：未找到描述</li><li><a href="https://swiftylaun.ch/">SwiftyLaunch</a>：iOS 应用生成器，包含确保在 App Store 快速发布所需的一切。</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=84d2a993-d3b9-43c1-a847-3d44cfe5c6ce">欢迎来到 Codeium - Codeium 文档</a>：未找到描述</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.semafor.com/article/01/15/2025/replit-ceo-on-ai-breakthroughs-we-dont-care-about-professional-coders-anymore">Replit CEO 谈 AI 突破：“我们不再关心专业程序员了” | Semafor</a>：Amjad Masad 谈论他们新的 AI 进展，这些进展将允许任何人自然地进行编程。</li><li><a href="https://vpn.net/">VPN.net – LogMeIn 出品的 Hamachi</a>：未找到描述</li><li><a href="https://tenor.com/view/it%27s-pretty-massive-michael-kupris-become-the-knight-it%27s-very-big-it%27s">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/ninja-fortnite-reaction-ninja-low-taper-fade-gif-1784137995500051652">Ninja Fortnite GIF - Ninja Fortnite 反应 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/itsPaulAi/status/1881329522949447886?t=4igrKlZqJ-rlvMDJvR8yOw&s=19">Paul Couvert (@itsPaulAi) 的推文</a>：哇，一个与 OpenAI o1 旗鼓相当的全开源推理模型刚刚发布，Deepseek R1 在几乎所有基准测试中甚至优于 Claude 3.5 Sonnet 和 o1-mini。你现在已经可以免费使用它（见下文...</li><li><a href="https://www.codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/contact/enterprise">联系方式 | Windsurf Editor 和 Codeium 扩展</a>：联系 Codeium 团队以获取支持并了解更多关于我们企业版方案的信息。</li><li><a href="https://www.codacy.com/">Codacy - 面向开发者的代码质量与安全</a>：使用 Codacy 平台高效、无畏地构建整洁、安全的代码。</li><li><a href="https://docs.replit.com/category/quickstarts">快速入门 | Replit 文档</a>：未找到描述</li><li><a href="https://docs.codeium.com/getstarte">欢迎来到 Codeium - Codeium 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/it%27s-pretty-massive-michael-kupris-become-the-knight-it%27s-very-big-it%27s-so-huge-gif-11424078658491848897">It's Pretty Massive Michael Kupris GIF - It's pretty massive Michael Kupris Become The Knight - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.com/contact">联系方式 | Windsurf Editor 和 Codeium 扩展</a>：联系 Codeium 团队以获取支持并了解更多关于我们企业版方案的信息。</li><li><a href="https://codeium.canny.io/feature-requests/p/supercomplete-repeatedly-suggests-to-reindent-the-entire-file-with-the-same-inde">Supercomplete 反复建议使用文件已有的相同缩进重新缩进整个文件 | 功能请求 | Codeium</a>：自从最近的更新（v 1.2.1，2025年1月17日）以来，Supercomplete 奇怪地建议使用文件已有的相同缩进对整个文件进行缩进</li><li><a href="https://x.com/kimmonismus/status/1881092191277457784?t=ctHXhAKdCjvqpl6kq0jTRQ&s=19">Chubby♨️ (@kimmonismus) 的推文</a>：Dario Amodei (Anthropic CEO) 访谈即将到来。终于要发布他们的新模型了吗？引用 Joanna Stern (@JoannaStern) 的话：呼叫所有 Claude 用户。你们愿望清单的首位是什么？发送...</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">自动提交信息 | 功能请求 | Codeium</a>：从提交的文件上下文生成提交信息</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://cursor.directory/">Cursor 目录</a>：为你的框架和语言寻找最佳的 Cursor 规则。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1329920524796690484)** (624 条消息🔥🔥🔥): 

> `Perplexity 模型变更、用户反馈与问题、新 AI 工具与替代方案、DeepSeek-R1 集成、用户互动与社区支持` 


- **Perplexity 模型变更引发关注**：用户对 Perplexity 最近的更新表示不满，指出在禁用第三方 LLM 后，其自研模型缺乏动态响应和上下文理解能力。
   - 许多用户感到沮丧，因为他们觉得 Pro 订阅物无所值，并期待尽快得到改进。
- **反馈凸显用户问题**：社区成员强调了账单问题、支持响应缓慢以及 Perplexity 输出内容平庸等问题，导致不满的用户取消了订阅。
   - 用户呼吁提高透明度并加快修复速度，以维持客户信任，尤其是考虑到该平台的估值。
- **新 AI 工具与替代方案的涌现**：几位用户讨论了 Ithy 和 complexity 扩展等替代方案，认为与 Perplexity 相比，这些可能是更适合其需求的解决方案。
   - 人们对利用开源模型和工具以在项目中获得更好结果和灵活性表现出越来越浓厚的兴趣。
- **承诺集成 DeepSeek-R1**：深入的讨论分享了 Perplexity 可能很快会集成 DeepSeek-R1，以增强其服务中的高级推理能力。
   - 用户渴望这一调整，他们认为这可以恢复部分功能并提升平台体验。
- **活跃的用户互动与支持**：社区保持活跃，用户分享故障排除建议、使用不同的 AI 工具，并在应对近期变化方面互相支持。
   - 关于技术进步以及将编程技能融入职业发展的策略反馈表明，用户群体积极主动并对持续学习感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ithy.com/article/ayaneo-kun-specs-2025-rxuu0hb1">2025 年 AYANEO Kun 规格与处理器芯片</a>: 未找到描述</li><li><a href="https://x.com/testingcatalog/status/1881399907032191334?s=61">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: Perplexity 可能会在其模型产品中加入 DeepSeek R1 👀 引用 Aravind Srinivas (@AravSrinivas) @jaseempaloth 是的</li><li><a href="https://x.com/AravSrinivas/status/1881458694266953934">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 你可以在 http://labs.perplexity.ai 上尝试 DeepSeek-R1。我们很快会尝试将其引入 Perplexity 核心服务的高级推理 Pro 搜索中。</li><li><a href="https://aistudio.google.com/prompts/new_chat">未找到标题</a>: 未找到描述</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>: Perplexity 运行状态</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案</a>: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案 - ItzCrazyKns/Perplexica</li><li><a href="https://ayaneo.com/product/AYANEO-KUN.html">AYANEO KUN - AYANEO</a>: 未找到描述</li><li><a href="https://www.cnbc.com/2025/01/18/perplexity-ai-makes-a-bid-to-merge-with-tiktok-us.html">Perplexity AI 竞购合并 TikTok 美国业务</a>: 据 CNBC 获悉，Perplexity AI 周六向 TikTok 母公司字节跳动提交了竞标书，提议 Perplexity 与 TikTok 美国业务合并。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1330021479391170620)** (24 messages🔥): 

> `RedNote App, FBI Malware Uninstallation, Gaia Sky Scan Co., Perplexity AI Acquisition, ISO27001 and NIS2 Controls` 


- **RedNote App 在美国爆火**：**RedNote App** 在美国实现了显著增长，引发了用户和开发者的共同关注。
   - 有关其功能和用户参与度的更多详情可以在 [YouTube 视频](https://www.youtube.com/embed/qsj299D8oLM)中找到。
- **FBI 入侵电脑以卸载恶意软件**：有报告称 **FBI** 一直在主动入侵电脑以清除恶意软件，从而保护用户。
   - 这一不同寻常的举动旨在确保受损系统的安全，但也引发了关于隐私的质疑。
- **Gaia Sky Scan 公司更新**：**Gaia Sky Scan Co.** 发布了最新进展，在技术社区引起了轰动。
   - 分享了关于他们最新项目和创新的细节，表明其在市场上的影响力正在增长。
- **Perplexity 收购 Read.cv**：Perplexity 已正式**收购 Read.cv**，增强了其在 AI 领域的实力。
   - 有关此次收购的更多见解可以在[详细报告](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA)中找到。
- **ISO27001 与 NIS2 中的重叠控制项**：关于 **ISO27001** 和 **NIS2** 中**重叠控制项**的讨论强调了重要的合规性重合。
   - 参与者对简化这些控制项实施过程的策略表现出浓厚兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/qsj299D8oLM">YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/embed/b1eND15ci5A">YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1329906974552494112)** (3 messages): 

> `CrewAI models, Litellm monkey fix, Unnecessary pings` 


- **CrewAI 模型未能解决问题**：一位用户报告称，他们尝试了所有三种 **CrewAI 模型**，但未能成功修复一个持久性问题。
   - 他们指出 **CrewAI 文档**中缺少对该问题的描述，另一位用户在 **o1 模型**上也遇到了同样的问题。
- **发现 Litellm 的 Monkey Fix**：用户发现了一个 **monkey fix**，可以在调用前成功移除 **Litellm** 的停止参数 (stop parameters)，暂时解决了他们的问题。
   - 这一临时解决方案是针对现有模型的持续挫败感而分享的。
- **Ping 礼仪提醒**：一位用户提醒另一名成员避免不必要的 Ping，并询问该如何提供帮助。
   - 这一交流凸显了群组内对沟通礼仪的持续关注。


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1329903374862258187)** (588 messages🔥🔥🔥): 

> `Cursor Performance Issues, DeepSeek R1, Agent Functionality Comparison, Slow Request Concerns, GitHub Integrations`

- **Cursor 遭遇请求缓慢问题**：用户对请求速度慢表示沮丧，特别是在 Agent 功能方面，指出即使在以前响应迅速的环境中也会出现 3 分钟的延迟。
   - 客户的不满归因于与 Windsurf 和 Gemini 等竞争对手相比，在速度和性能方面提供的价值不足。
- **DeepSeek R1 的能力**：DeepSeek R1 在基准测试中的表现显示它可以与 OpenAI 的 O1 等模型有效竞争，一些用户渴望将其纳入 Cursor。
   - 围绕 DeepSeek R1 的开源性质及其通过 API 访问的应用讨论，突显了其相对于其他 AI 助手（AI assistants）的潜在优势。
- **Agent 功能需要改进**：参与者讨论了 Cursor 的 Agent 目前无法管理大文件，并可能无意中删除重要代码，因此需要额外的人工检查。
   - 用户正在寻求改进这种体验的方法，并提出了关于 Cursor 规则的建议，并确保 AI 工具支持无错误的迭代开发。
- **AI 助手对比**：随着用户将 Cursor 的功能与 Cline 和 GitHub Copilot 进行比较，引发了对不同模型及其性价比的重大关注。
   - 社区对各种工具的有效性似乎存在分歧，一些人强调了结合 AI 进行详尽文档记录和人工审查的重要性。
- **反馈与开发建议**：用户建议将 DeepSeek R1 等模型整合到 Cursor 中，以增强其能力并解决当前的性能问题。
   - 社区反馈的重要性已变得显而易见，用户期待 Cursor 的更新和补丁来解决持续存在的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://half-single-ecd.notion.site/Experimental-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://x.com/bintz_gavin/status/1880789354211094906">来自 🥞 Gavin (@bintz_gavin) 的推文</a>：看起来 @theo 的 t3 chat 是一个很好的信号，表明我们需要更快的 AI 应用。我刚刚开源了一个简单的 Agentic 博客撰写工具，它使用 @CerebrasSystems 进行推理，速度约为 2,100 tokens/s，甚至比 Groq 还快...</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server">Live Preview - Visual Studio 商店</a>：Visual Studio Code 扩展 - 在您的工作区中托管一个本地服务器，供您预览网页。</li><li><a href="https://www.cursor.com/downloads">下载 | Cursor - AI 代码编辑器</a>：选择您的平台以下载最新版本的 Cursor。</li><li><a href="https://x.com/paulgauthier/status/1881428345973608901">来自 Paul Gauthier (@paulgauthier) 的推文</a>：DeepSeek R1 在 aider 多语言基准测试中获得 57%，排名第二，仅次于 o1：62% o1 (high)，57% DeepSeek R1，52% Sonnet，48% DeepSeek Chat V3。完整排行榜：https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/moritzkremb/status/1880628661931634700?s=19">来自 Moritz Kremb (@moritzkremb) 的推文</a>：到目前为止，我已经对 Cursor 的 Agent 功能进行了数百次测试。如果您想让 Agent 通过单个命令完成广泛的工作流，最好的方法是...</li><li><a href="https://codeium.com/changelog)">页面未找到 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://forum.cursor.com/">Cursor - 社区论坛</a>：讨论 Cursor 的地方（Bug、反馈、想法等）。</li><li><a href="https://open-vsx.org/vscode/item?itemName=ms-vscode.live-server">Open VSX 注册表</a>：未找到描述。</li><li><a href="https://traycer.ai/">Traycer：AI 驱动的结对编程</a>：一个 AI 驱动的编码助手，负责规划、实施和验证每一次更改 🚀</li><li><a href="https://x.com/sama/status/1881258443669172470?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 Sam Altman (@sama) 的推文</a>：Twitter 上的炒作又失控了。我们下个月不会部署 AGI，也没有构建出它。我们有一些非常酷的东西要给你们，但请冷静，把你们的预期降低 100 倍！</li><li><a href="https://www.youtube.com/watch?v=AtuB7p-JU8Y">Cursor vs Cline | 240k Tokens 代码库侧向对比 AI 编程对决</a>：🚀 在这段视频中，我们使用一个 240,000 token 的代码库来对比两款顶尖的 AI 编程工具：Cursor 和 Cline。观看我们对比它们的功能...</li><li><a href="https://github.com/features/copilot/extensions">GitHub Copilot 扩展 · 您喜爱的工具已进入 Copilot Chat。</a>：使用即插即用的扩展来增强 GitHub Copilot，或者使用我们的开发者平台（包含 API、文档和指南）构建您自己的扩展。</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">推理模型 (deepseek-reasoner) | DeepSeek API 文档</a>：deepseek-reasoner 是由 DeepSeek 开发的推理模型。在交付最终答案之前，模型会先生成思维链 (CoT) 以提高响应的准确性。我们的 API 提供...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1329930548235473007)** (522 条消息🔥🔥🔥): 

> `DeepSeek-R1, AI 与加密货币, MiniCPM-o 2.6, 推理模型, 强化学习`

- **DeepSeek-R1 及其蒸馏过程 (Distillation Process)**：参与者讨论了最近发布的 DeepSeek-R1，指出了其成功的蒸馏结果以及对未来推理模型的影响。
   - 对于通过 RL 和其他方法优化推理过程的模型在开源推理方面的潜力，人们感到非常兴奋。
- **AI 与 Crypto 的融合**：社区辩论了 AI 与 Crypto 的交集，探讨了 AI Agent 如何潜在地利用 Crypto 进行资源交易和执行任务。
   - 针对 Crypto 领域现有问题的担忧也随之产生，特别是关于可能削弱有益应用的投资动机。
- **MiniCPM-o 2.6 模型能力**：成员们对 MiniCPM-o 2.6 的功能表现出兴趣，这是一款专为视觉、语音和多模态应用设计的模型。
   - 讨论强调了该模型的性能、量化 (Quantization) 选项，以及与现有 AI 模型在各种应用实用性方面的比较。
- **强化学习 (Reinforcement Learning) 与结果奖励 (Outcome Rewards)**：参与者研究了在深度学习中使用结果奖励的方法及其对模型性能的影响。
   - 共享了关于 RL 如何鼓励模型在没有明确指令的情况下进行最优学习的见解，从而实现推理能力的有机发展。
- **社区对托管服务商的担忧**：有人投诉 Lambda 托管 Hermes 3 405B 的服务性能，特别是频繁出现的错误。
   - 成员们讨论了替代供应商和解决方案，以寻求满足其计算需求的更可靠托管选项。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/AGI-0/Art-v0-3B">AGI-0/Art-v0-3B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/bozoeggs/status/1881328847121236463">来自 Egg (@bozoeggs) 的推文</a>：看吧，我告诉过你们，我只需要唱中文卡拉 OK，家里的 AGI 就会出现。引用 DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 发布了！⚡ 性能与 OpenAI-o1 相当 📖 完全开源...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-int4">openbmb/MiniCPM-o-2_6-int4 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Donnyed/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M-GGUF">Donnyed/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/dani_avila7/status/1880739290264809683?s=46">来自 Daniel San (@dani_avila7) 的推文</a>：在 Cursor 中引入代码库知识图谱（Codebase Knowledge Graphs）🤩 在这段视频中，我将向您展示我们如何从在 CodeGPT 平台上使用代码仓库的知识图谱，转变为直接在...中利用它。</li><li><a href="https://tenor.com/view/its-happening-gif-23353691">Its Happening GIF - Its Happening - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-wizard-meme-funny-gif-3870502440791733376">Cat Wizard GIF - Cat Wizard Meme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B">FreedomIntelligence/HuatuoGPT-o1-70B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/teortaxesTex/status/1881331287010550119">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：也许最疯狂的事情是，他们说这还远未达到 7-70B 级别模型的上限。甚至不需要任何新数据。他们已经将它们推得更远了，只是不会分享出来。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://forms.gle/F9DtNZtkWquHyQyS8">社区洞察：关于 FirstBank（总部位于田纳西州纳什维尔）的认知与意见调查</a>：这只是为了我正在做的一个研究项目！非常感谢您的回答，因为我的时间非常紧迫，我的项目急需这些数据。</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1hizjq4/i_have_underestimated_o3s_price">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>：通过在 GitHub 上创建账号，为 MoonshotAI/Kimi-k1.5 的开发做出贡献。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">deepseek-ai/DeepSeek-R1 main 分支下的 DeepSeek_R1.pdf</a>：通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf">MoonshotAI/Kimi-k1.5 main 分支下的 Kimi_k1.5.pdf</a>：通过在 GitHub 上创建账号，为 MoonshotAI/Kimi-k1.5 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k">NovaSky-AI/Sky-T1_data_17k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/casper-hansen/AutoAWQ/pull/688">由 LagPixelLOL 添加了 DeepSeek V3 支持 · Pull Request #688 · casper-hansen/AutoAWQ</a>：#686 我仅在 1B 版本的模型上使用随机初始化的权重进行了测试，因此对于 671B 的大模型还需要进一步测试。此外，由于 gemm CUDA 内核中的组大小限制...</li><li><a href="https://www.youtube.com/watch?v=JFJg9KZ_iZk.">MiniCPM-o 2.6：一个 8B 大小、GPT-4o 级别的端侧全能模型</a>：💥 隆重推出我们的 MiniCPM-o 2.6：一个 8B 大小、GPT-4o 级别的端侧全能模型 ✨ 亮点：在视觉、音频和多模态实时流媒体方面媲美 GPT-4o-202405...</li><li><a href="https://github.com/OpenBMB/MiniCPM-o">GitHub - OpenBMB/MiniCPM-o: MiniCPM-o 2.6：一个在手机上运行的视觉、语音和多模态实时流媒体 GPT-4o 级别 MLLM</a>：MiniCPM-o 2.6：一个在手机上运行的视觉、语音和多模态实时流媒体 GPT-4o 级别 MLLM - OpenBMB/MiniCPM-o
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1330030930970284174)** (36 条消息🔥): 

> `高精度手写文本 OCR 模型，MOEs 与稠密模型的对比，AI 模型中结构化稀疏的效率，LLM 训练中的学习率调度` 


- **OCR 模型面临误读挑战**：用户讨论了他们在各种高精度**手写文本 OCR 模型**（如 **Sonnet-3.5** 和 **Qwen**）上的使用体验，这些模型经常会出现字符误读。
   - 有人建议在 OCR 库较弱的语言上，使用 **OCR** 或目标检测（object detection）来提高字符识别率。
- **MOEs 与稠密模型 - 参数对比**：一位用户探讨了如何将 **MOEs** 与**稠密模型**进行比较，建议稠密模型的等效大小是激活参数（active parameters）与总参数（total parameters）之间的几何平均值。
   - 他们计算了 **Deepseek V3** 和 **Minimax-01** 的等效值，理论上可以在更高的参数内存占用下实现 **3-4 倍的延迟提升**。
- **结构化稀疏对模型效率的影响**：结构化稀疏被强调为提高效率的有效方法，特别是 **Nvidia Ampere** 硬件支持 **2:1 稀疏性**以减少计算需求。
   - 成员们指出，虽然这种方法有助于提高计算速度，但内存需求保持不变。
- **深度卷积 MLP 模块（Depthwise MLP Blocks）作为折中方案**：一位用户提出使用**深度卷积 MLP 模块**作为稠密架构和 **MOE** 架构之间的折中方案，通过拆分输入激活值来潜在地节省参数。
   - 成员们讨论了这与**组卷积（groupwise convolutions）**的相似之处，并指出这些方法可能会带来更高效的网络设计。
- **关于余弦预热衰减调度器（Cosine Warmup Decay Scheduler）的疑问**：有人询问在继续训练 **GPT-2** 模型时使用**余弦预热衰减调度器**的问题，特别是关于调整总训练步数的问题。
   - 该用户担心如果不更新步数，可能会导致其继续训练过程中的学习率出现偏差。



**提及的链接**：<a href="https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/">NVIDIA Ampere 架构中的结构化稀疏及其在搜索引擎中的应用 | NVIDIA 技术博客</a>：深度学习在各个领域和地区都取得了显著成功，因为它彻底改变了我们分析、理解和处理数据的方式。在计算机领域有许多成功案例...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1330386739704889344)** (2 条消息): 

> `气候变化对农业的影响，LLMs 中的思维进化（Mind Evolution）` 


- **气候研究需要合作**：一个名为**“开发卷积神经网络以评估气候变化对全球农业产量的影响”**的研究项目正在寻求在 **Machine Learning**、**气候科学**和**数据分析**等多个领域具有专业知识的合作者。
   - 有意者请在 **1 月 25 日**前通过私信联系，以便在分享更多项目细节前确定团队名单。
- **Google 的 Mind Evolution 表现优于其他方案**：在最近的一次演示中，Google 强调了他们的 **Mind Evolution** 方法在自然语言规划任务中显著优于 **Best-of-N** 和 **Sequential Revision** 等其他推理策略。
   - 研究结果显示，在 **TravelPlanner** 和 **Natural Plan** 等基准测试中，**Mind Evolution** 使用 **Gemini 1.5 Pro** 在没有正式求解器的情况下解决了超过 **98%** 的问题实例。



**提及的链接**：<a href="https://x.com/_akhaliq/status/1881182840857178146?s=46">来自 AK (@_akhaliq) 的推文</a>：Google 展示了进化的更深层 LLM 思考。在控制推理成本的情况下，我们发现 Mind Evolution 显著优于其他推理策略，如 Best-of-N 和 Sequential Revision...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1331008929018413271)** (4 条消息): 

> `Liquid AI LFM-7B, 循环模型的影响, 新商业模式, Mistral Ministral 3B, Codestral 2501` 


- **Liquid AI 发布 LFM-7B，声称同类最佳**：[Liquid AI](https://www.liquid.ai/lfm-7b) 刚刚发布了 LFM-7B，被誉为其尺寸级别中性能最好的模型，利用非 Transformer 架构实现了高吞吐量和低内存占用。
   - 这款多语言模型支持 **英语、阿拉伯语** 和 **日语**，针对 **本地部署** 和成本受限的任务进行了优化。
- **对 LFM-7B 循环设计的好奇**：一位成员对 LFM-7B 的 **循环架构（recurrent architecture）** 如何影响其能力表示好奇，考虑到其较小的模型尺寸。
   - 他们指出，它在交互中的表现似乎还不错，符合对小模型的预期。
- **Liquid AI 独特的权重授权商业模式**：Liquid AI 似乎采取了一种有趣的策略，即出售或授权模型权重，这被描述为一种以前不常见的折中策略。
   - 这可能标志着 AI 模型分发和可访问性格局的转变。
- **Mistral 可能在 Ministral 3B 和 Codestral 2501 上采取类似做法**：一位成员推测 **Mistral** 可能对其模型 **Ministral 3B** 和 **Codestral 2501** 采用类似的授权策略。
   - 这表明 AI 公司为模型提供灵活授权选项的趋势日益增长。



**提到的链接**：<a href="https://www.liquid.ai/lfm-7b">介绍 LFM-7B：为高效语言模型设定新标准</a>：世界上同类最佳的英语、阿拉伯语和日语模型，原生支持法语、德语和西班牙语，优化后可作为私有企业聊天、代码、快速指令遵循等的底层基础...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1330386739704889344)** (2 条消息): 

> `气候变化合作研究, Google 在 LLM 中的 Mind Evolution` 


- **寻求气候变化研究合作伙伴**：一个团队正在启动名为“开发卷积神经网络以评估气候变化对全球农业产量的影响”的项目，并正在寻找 **Machine Learning**、**Climate Science**、**Geospatial Data** 和 **Scientific Writing** 方面的专家，在 **1 月 25 日** 之前加入。
   - *有志之士* 可以在 Discord 上私信 thomasyoungabc123 寻求合作机会。
- **Google 的 Mind Evolution 优于其他策略**：在最近的一次更新中，有人指出 Google 的 **Mind Evolution** 方法在自然语言规划任务中显著优于 **Best-of-N** 和 **Sequential Revision** 等策略，在基准测试中实现了超过 **98%** 的成功率。
   - 这一性能是在使用 **Gemini 1.5 Pro** 且无需正式求解器的情况下实现的，证明了其在解决问题方面的有效性。



**提到的链接**：<a href="https://x.com/_akhaliq/status/1881182840857178146?s=46">来自 AK (@_akhaliq) 的推文</a>：Google 展示了 Evolving Deeper LLM Thinking。在控制推理成本的情况下，我们发现 Mind Evolution 显著优于 Best-of-N 和 Sequential Revision 等其他推理策略...

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1329905489563226122)** (450 条消息🔥🔥🔥): 

> `DeepSeek R1 模型, Unsloth 训练脚本, 量化方法, Windows 安装问题, VTube 模型与绑定`

- **DeepSeek R1 模型已上传**：DeepSeek R1 的所有版本（包括 GGUF 和量化格式）均已上传至 Hugging Face，提升了模型的可访问性。
   - 该集合包括针对 Llama 和 Qwen 的蒸馏模型，为用户提供多种格式。
- **引入引导式 Unsloth 训练脚本**：创建了一个用于 Unsloth 训练的引导式脚本，允许用户在执行前输入各种训练参数。
   - 这简化了训练过程，并作为 GitHub Gist 提供给社区使用。
- **关于量化方法的讨论**：讨论了 IQ quantization 方法，重点强调了其复杂性以及与常规量化相比的潜在有效性。
   - 对话强调了为高质量 IQ quantization 获取合适校准集（calibration sets）的难度。
- **llama.cpp 在 Windows 上的安装挑战**：用户在 Windows 上尝试编译 llama.cpp 时遇到了挑战，原因是缺少 `make` 或 `cmake` 命令，日志中的错误消息指出了这一点。
   - 建议可能需要手动构建，因为当前脚本无法识别操作系统。
- **VTube 模型与社区关注点**：社区讨论了 VTube 模型的货币化方面，特别是供应商锁定（vendor-lock）做法以及模型所有者不提供源文件所带来的挑战。
   - 普遍认为依赖艺术家制作的模型限制了自由，从而引发了对自动化和 AI-generated 替代方案的兴趣。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>：Unsloth 的动态 4-bit 量化（Dynamic 4-bit Quants）有选择地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，极大地提高了准确性。</li><li><a href="https://x.com/ggerganov/status/1880237609647034551">Georgi Gerganov (@ggerganov) 的推文</a>：llama-cli -hf unsloth/phi-4-GGUF llama-server -hf unsloth/phi-4-GGUF（感谢 @ngxson）</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/tokenizer_config.json#L34">tokenizer_config.json · deepseek-ai/DeepSeek-R1-Distill-Qwen-32B at main</a>：未找到描述</li><li><a href="https://huggingface.co/JingzeShi/Doge-20M-Instruct">JingzeShi/Doge-20M-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit">unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://gist.github.com/sebaxakerhtc/5e7faa4ead6e2f4e0ea69634c3f624ba">Unsloth 指导脚本</a>：Unsloth 指导脚本。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models">pytorch/SECURITY.md at main · pytorch/pytorch</a>：Python 中的张量（Tensors）和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch</li><li><a href="https://x.com/UnslothAI/status/1881357596717891955">Unsloth AI (@UnslothAI) 的推文</a>：DeepSeek-R1 GGUF 版现已登陆 @HuggingFace！包含所有 Llama 和 Qwen 蒸馏模型以及 2 到 8-bit 的量化版本。如何运行 R1：https://unsloth.ai/blog/deepseek-r1 DeepSeek-R1 集合：http...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Aria-UI/Aria-UI-base">Aria-UI/Aria-UI-base · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows 安装 | Unsloth 文档</a>：了解如何在 Windows 上（使用或不使用 WSL）安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/lora-parameters-encyclopedia)">Unsloth 文档</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF">bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/inst">Unsloth 文档</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">数据集 101 | Unsloth 文档</a>：学习创建微调数据集的所有要点！</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Zero">unsloth/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">HuggingFaceFW/fineweb-edu · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/intel/auto-round">GitHub - intel/auto-round: 适用于 LLM/VLM 的高级量化算法。</a>：适用于 LLM/VLM 的高级量化算法。通过在 GitHub 上创建账号来为 intel/auto-round 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard">低比特量化开源 LLM 排行榜 - Intel 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth：微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 70%</a>：微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/tree/nightly">GitHub - unslothai/unsloth (nightly 分支)</a>：微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 70% - GitHub - unslothai/unsloth at nightly</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1329914437225807922)** (11 条消息🔥): 

> `使用 OpenRouter 进行 LLM 对比，开源 Web UI 选项，本地运行模型，Flowise 作为聊天框架` 


- **使用 OpenRouter 进行 LLM Prompt 对比**：一位成员建议使用 [OpenRouter](https://openrouter.com/) 创建新聊天，允许用户一次性对比多个开源 LLM。
   - 一旦点击发送，所有选定的模型都会做出响应，不过进行大规模测试可能需要一些 credits。
- **聊天应用的开源 UI 选择**：几位成员推荐了多种用于构建聊天应用的开源 Web UI 选项，并强调 [Open Web UI](https://github.com/open-webui/open-webui) 是一个强有力的选择。
   - 另一位成员提到了 **Flowise**，并指出它适用于网站上的公共 Chat-bots。
- **寻找本地运行模型的库**：一位用户询问了关于本地运行模型的开源库，收到了如 **Gpt4all** 和 **textwebgenui** 等建议。
   - 建议在使用这些工具之前检查许可协议。
- **前端开发顾虑**：一位成员表示不愿专注于前端开发，而更倾向于提升自己在 AI 框架方面的技能。
   - 总的来说，社区提供了大量资源来简化聊天应用的开发过程，而无需深入研究前端技术。



**提到的链接**：<a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: User-friendly AI Interface (Supports Ollama, OpenAI API, ...)</a>：用户友好的 AI 界面（支持 Ollama, OpenAI API, ...）- open-webui/open-webui

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1329913530580992047)** (77 条消息🔥🔥): 

> `模型微调、模型保存技术、模型性能问题、推理采样、使用 Unsloth 文档` 


- **探索 Qwen 和 Phi 模型的微调**：成员们讨论了他们在微调 **Qwen** 和 **Phi** 模型方面的经验，并注意到 **Liberation 3.1** (LLM) 和 **Phi-4** 等不同模型在训练时间和指标上的差异。
   - 一位用户提到了 **Phi-4** 上的欠拟合 (underfitting) 问题，这可能是由于模型增强的指令微调 (instruction tuning) 导致的。
- **训练损失观察**：用户分享了他们对训练损失指标的观察，一些用户报告了 **WizardLLM** 和 **Qwen2.5** 等模型的低损失，并探讨尝试不同格式的想法。
   - 有一个专门的询问是关于在 **Qwen2.5** 中使用 **Alpaca 格式** 是否能产生更好的结果。
- **模型保存中的挑战与解决方案**：讨论围绕如何在不牺牲准确性的情况下保存微调后的模型展开，特别是当以 **GGUF 格式** 保存为 **F16** 时会导致显著损失的问题。
   - 用户考虑了各种确保模型保存后性能得以保留的方法，并强调了 **Unsloth 文档** 中提到的最佳实践。
- **推理与采样的挑战**：有关于在使用 **Unsloth** 进行推理时的**采样算法 (sampling algorithm)** 的疑问，特别是与评估期间的预期结果相关的问题。
   - 澄清了采样主要是推理过程中的关注点，而非训练过程，这会影响结果的解释方式。
- **在 LM Studio 中加载模型**：讨论了在 **LM Studio** 中加载 **DeepSeek-R1 Qwen 14B** 模型的问题，重点提到了一个与模型词汇表 (vocabulary) 相关的错误。
   - 通过更新 **LM Studio** 和 **Nvidia 驱动程序** 解决了该问题，消除了加载错误并使模型能够正常运行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-Math-7B-Instruct">unsloth/Qwen2.5-Math-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#if-saving-to-gguf">故障排除 | Unsloth 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#if-saving-to-gguf-or-vllm-16bit-crashes">故障排除 | Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1040">加载 Qwen2 聊天界面的正确方式是什么？ · Issue #1040 · unslothai/unsloth</a>: 我收到了这个错误：chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: 'Qwen2-1.5B' 来自这段代码：def test_un...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 为在 Ollama 上本地运行创建定制化个人助手（如 ChatGPT）的入门指南
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1330012111388016671)** (20 条消息🔥): 

> `Chatterbox Dataset Builder, Sky-T1 Model Performance, Synthetic Datasets, LLM Integration, Docker-Compose Setup` 


- **Chatterbox Dataset Builder 发布**：推出了一款名为 [Chatterbox](https://github.com/invisietch/Chatterbox) 的新工具，用于多轮对话数据集管理，允许用户创建、编辑和删除对话，并具备 Token 计数和标签（tagging）等多种功能。
   - 开发者提到，未来将支持与 **OpenWebUI**、**Ollama**、**Flowise** 和 **LocalAI** 的集成，并表示目前已通过 kobold API 支持 kobold 和 aphrodite。
- **Sky-T1 模型详情公布**：重点介绍了 [Sky-T1-32B](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) 模型，其在数学和编程方面的性能与 o1-preview 相当，该模型使用来自 Qwen2.5-32B-Instruct 的 17K 数据训练而成。
   - 该模型由加州大学伯克利分校的 **NovaSky Team** 开发，采用 Batch Size 为 96 的训练流程，在 8 张 H100 上使用 DeepSpeed Zero-3 Offload 耗时 19 小时完成训练。
- **Chatterbox 的功能与增强**：Chatterbox 的改进包括新增了用于简化本地部署的 Docker-compose 配置（支持单条命令安装），以及支持多轮导出的**偏好响应（preferential responses）**功能。
   - 开发者表示计划实现 LLM 集成，以便为对话双方生成响应，并能调整聊天历史中的角色以防止混淆。
- **合成数据集生成**：关于自动生成**合成数据集（synthetic datasets）**的提案引发了讨论，内容涉及可能使用 Webworkers 或基于相同后端 API 的 CLI 进行批量操作。
   - 开发者对自动化数据集生成过程表示了兴趣，并就实现该功能的具体方法提出了疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview">NovaSky-AI/Sky-T1-32B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/invisietch/Chatterbox">GitHub - invisietch/Chatterbox: LLM 训练者的多轮对话数据集管理工具</a>：LLM 训练者的多轮对话数据集管理工具 - invisietch/Chatterbox</li><li><a href="https://github.com/invisietch/">invisietch - 概览</a>：Dangerous。invisietch 拥有 2 个公开仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1329968927778017401)** (8 条消息🔥): 

> `Dataset usage for model training, LLM Research Cohort at Cohere For AI, Deep Learning resources for beginners` 


- **简单但有效的数据集策略**：你可以通过推理模式下的教师模型处理海量数据集，生成输入/输出对，从而训练一个较小的模型。
   - 虽然这种方法自 **GPT-4** 以来已被广泛使用（如 **Microsoft** 的 **Phi**），但重要的是不要仅仅复制风格。
- **加入 LLM 研究队列！**：由 **Cohere For AI** 组织的 **LLM Research Cohort** 提供多语言长文本挑战的实战经验，旨在增强 NLP 能力。
   - 参与者将参与两个方向的学习，重点关注处理和评估**多语言 LLM** 的先进技术，启动会议定于 1 月 10 日举行。
- **初学者的深度学习之路**：一位成员对初学者学习深度学习需要多长时间以及如何应对该领域不断的更新表示担忧。
   - 其中一个建议是利用 **ChatGPT** 等资源来帮助理解概念，并解决深度学习和 AI 领域的挑战。



**提到的链接**：<a href="https://x.com/cataluna84/status/1877689686639992872">来自 Mayank Bhaskar (@cataluna84) 的推文</a>：由 @cohere 开源科学社区的 @akankshanc 组织的 BIRDS（研究驱动型学习初学者）项目，我们很高兴地宣布新的 LLM 队列（Cohort）！🎉 🚀这不仅仅是另一次学习...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1329958392214585465)** (154 条消息🔥🔥): 

> `RWKV 模型讨论、模型量化格式、混合专家模型 (MoE)、AI 模型性能、AI 开发与职业分享` 


- **RWKV7 在代际模型中占据强势地位**：RWKV7 被认为是唯一发布了可用模型的第七代 RNN，标志着其在当前 AI 架构中的独特地位。讨论强调了它与 Gated DeltaNet 等其他模型在设计上的相似性，并重点介绍了持续的改进。
   - 成员们辩论了通道衰减（channel-wise decay）和学习率等设计特性带来的影响，展示了 RWKV7 相对于旧款模型的竞争优势。
- **转向 GGUF 作为主要的量化模型格式**：GGUF 已成为量化模型的主导格式，因其在消费级硬件上的易用性以及主流量化工具的支持而受到青睐。随着 GGUF 获得关注，AWQ 和 GPTQ 等其他格式虽然可能继续存在，但在采用率上已处于落后地位。
   - 与会者指出，大公司通常在内部进行模型量化，导致开源社区中出现了更多的 GGUF 文件。
- **探索混合专家模型 (MoE)**：MoE 因其效率和性能优势而受到关注，尽管一些成员对其训练期间的稳定性表示担忧。讨论 MoE 范式的文章被认为是理解其实现方式的有用资源。
   - 成员们分享了关于理解和应用 MoE 的挑战性，以及它在 AI 模型架构中潜在回报的看法。
- **扩展与模型部署策略**：讨论集中在 VLLM 和 Ollama 等各种工具部署小型 AI 模型的效率上，偏好因公司规模和负载需求而异。VLLM 因其有效的扩展能力而受到称赞，使其成为专业团队的热门选择。
   - 相比之下，Ollama 在重负载下的表现被认为不够理想，引发了对其与市场上其他解决方案相比实用性的质疑。
- **AI 开发者联系与职业机会**：社区成员积极进行自我介绍，分享他们在 AI 开发方面的背景并寻求合作机会。对话凸显了在 AI 服务方面的多样化经验以及在该领域建立联系的兴趣。
   - 值得注意的是，一名成员表达了与社区内其他人建立联系和合作的意向，展示了 AI 专业人士网络正在不断壮大。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>：线性 Transformer 作为标准 Transformer 的高效替代方案已引起关注，但它们在检索和长上下文任务中的性能一直受到限制。为了解决这些局限性，...</li><li><a href="https://arxiv.org/abs/2409.19044">On the Inductive Bias of Stacking Towards Improving Reasoning</a>：鉴于模型规模的不断扩大，渐进式堆叠 [Gong et al., 2019, Reddi et al., 2023] 等新型训练策略引起了人们的兴趣。堆叠通过逐渐...实现高效训练。</li><li><a href="https://arxiv.org/abs/2501.08313">MiniMax-01: Scaling Foundation Models with Lightning Attention</a>：我们推出了 MiniMax-01 系列，包括 MiniMax-Text-01 和 MiniMax-VL-01，它们可与顶尖模型媲美，同时在处理长上下文方面提供卓越的能力。其核心在于...</li><li><a href="https://x.com/BlinkDL_AI/status/1876849157920452781">BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV-7 "Goose" 🪿 World 0.4B 发布：同尺寸下最强的基础模型🚀下载：https://huggingface.co/BlinkDL/rwkv-7-world 演示：https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1 引用 B...</li><li><a href="https://newsletter.armand.so/p/understanding-mixture-experts">Understanding Mixture of Experts</a>：好得令人难以置信？</li><li><a href="https://huggingface.co/blog/moe">Mixture of Experts Explained</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1329904593152249976)** (297 条消息🔥🔥): 

> `DeepSeek R1, 梯度尖峰, 优化技术, Titan 模型与记忆, LLM 中的 RL 训练`

- **DeepSeek R1 展示了性能提升**：DeepSeek R1 引入了一种新方法，在 AIME 和 MATH-500 等 Benchmark 上表现出色，显著超越了 GPT-4o 和 Claude Sonnet 3.5。
   - 该模型在长推理任务中的有效性以及处理高达 128k tokens 扩展上下文的能力，对其整体性能贡献巨大。
- **关于 Gradient Spikes 的研究课程**：讨论集中在模型训练中 Gradient Spikes 的影响，共识认为 Spikes 可能导致模型容量和性能的永久性损伤。
   - 强调了调整 Hyperparameters 以减轻这些问题的重要性，同时也对可恢复 Spikes 的影响表示了担忧。
- **关于优化技术的辩论**：专家们讨论了各种优化方法的优缺点，指出某些方法在理论上看起来不错，但在实践中却失败了。
   - 考虑到 Learned Optimization 算法相对于人工设计方法的改进潜力，这一点已在先前的研究中得到证实。
- **理解 Titans 的记忆机制**：Titans 论文讨论了在测试期间记住 Key 和 Value 之间映射的重要性，这表明了对 Inner-loop 训练的更深层理解。
   - 这一概念植根于 Learning to Learn 的更广泛背景，并根据历史数据关联优化模型性能。
- **模型训练中 RL 技术的探索**：对话涉及了强化学习（RL）在训练语言模型中的效用，特别是关于其在不同上下文长度下的有效性。
   - 建议通过运行不同长度的实验，来阐明 RL 训练方法的比较优势。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/cataluna84/status/1877689686639992872">来自 Mayank Bhaskar (@cataluna84) 的推文</a>：由 @cohere Open Science Community 的 @akankshanc 组织的 BIRDS (Beginners in Research Driven Studies)，我们很高兴地宣布我们的新 LLM Cohort！🎉 🚀这不仅仅是另一个学习...</li><li><a href="https://arxiv.org/abs/2410.22570v1#S6.SS3">Orb: 一种快速、可扩展的神经网络势能</a>：我们介绍了 Orb，一系列用于材料原子模拟的通用原子间势能模型。Orb 模型比现有的通用势能快 3-6 倍，在模拟中表现稳定...</li><li><a href="https://arxiv.org/abs/2306.13326">通过一阶和二阶优化算法求解随机方程组</a>：基于梯度的（又称“一阶”）优化算法通常用于解决大规模非凸问题。然而，预测它们的有效性通常很难。为了获得...</li><li><a href="https://arxiv.org/abs/2501.00663v1">Titans: 在测试时学习记忆</a>：十多年来，关于如何有效利用循环模型和 Attention 进行了广泛的研究工作。虽然循环模型旨在将数据压缩到固定大小的内存中...</li><li><a href="https://arxiv.org/abs/2112.00114">展示你的工作：语言模型中间计算的 Scratchpads</a>：大型预训练语言模型在可以“一次性完成”的任务上表现出色，例如生成逼真的文本或合成计算机程序。然而，它们在处理...时遇到困难。</li><li><a href="https://arxiv.org/abs/2501.06842">SPAM: 具有动量重置的 Spike-Aware Adam，用于稳定的 LLM 训练</a>：大型语言模型 (LLMs) 在各种任务中表现出卓越的性能，但其训练仍然高度消耗资源，且容易受到训练不稳定性等关键挑战的影响...</li><li><a href="https://arxiv.org/abs/2501.09891">演化更深层次的 LLM 思考</a>：我们探索了一种用于扩展大型语言模型推理时间计算的演化搜索策略。所提出的方法 Mind Evolution 使用语言模型来生成、重组和改进...</li><li><a href="https://arxiv.org/abs/2501.00663">Titans: 在测试时学习记忆</a>：十多年来，关于如何有效利用循环模型和 Attention 进行了广泛的研究工作。虽然循环模型旨在将数据压缩到固定大小的内存中...</li><li><a href="https://x.com/rosstaylor90/status/1881374050079187037">来自 Ross Taylor (@rosstaylor90) 的推文</a>：@xpasky 如果只是天真地奖励它：是的。但想象一下这样的奖励结构：如果你得到了错误的答案，那么模型停止思考太早了。如果你得到了正确的答案，模型使用了...</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：带有“Cold Start”的 DeepSeek R1 表现基本符合预期。我仍然不相信 R1 Zero 的结果，基础模型在不进行微调的情况下几乎无法输出连贯的解决方案。我打赌这里面有...</li><li><a href="https://arxiv.org/abs/1606.04474">通过梯度下降学习梯度下降</a>：机器学习中从手工设计特征到学习特征的转变取得了巨大成功。尽管如此，优化算法仍然是手工设计的。在本文中，我们展示了如何...</li><li><a href="https://x.com/BlinkDL_AI/status/1855245097094517181">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV-7 也可以在 3200 步内达到 2.27xx（最初为 5100 步）😀可复现的代码和日志：https://github.com/BlinkDL/modded-nanogpt-rwkv 🚀 #RWKV #RNN 引用 Keller Jordan (@kellerjordan0) 的话...</li><li><a href="https://x.com/kimi_ai_/status/1881332472748851259?s=46">来自 Kimi.ai (@Kimi_ai_) 的推文</a>：🚀 推出 Kimi k1.5 —— 一个 o1 级别的多模态模型。Sota 的短 CoT 性能，在 📐AIME、📐MATH-500、💻 LiveCodeBench 上大幅超越 GPT-4o 和 Claude Sonnet 3.5（最高达 +550%...</li><li><a href="https://x.com/rm_rafailov/status/1880994108241842314">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：@armanhaghighik @iScienceLuvr 这些都不是“自然涌现”的，纯属胡说八道。显然，所有不同的模型在 RL 阶段之前都植入了不同的策略。</li><li><a href="https://x.com/BlinkDL_AI/status/185">来自 crystal (@crystal) 的推文</a>：Adam 讨厌我的用户名。</li><li><a href="https://x.com/deepseek_ai/status/1859200149844803724">来自 DeepSeek (@deepseek_ai) 的推文</a>：🌟 DeepSeek-R1-Lite-Preview 的推理缩放定律。推理越久，性能越好。DeepSeek-R1-Lite-Preview 在 AIME 上的得分随着思考长度的增加而稳步提升。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/">GitHub - deepseek-ai/DeepSeek-R1</a>：通过创建账户为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li>

on GitHub.</li><li><a href="https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/ttt">flash-linear-attention/fla/ops/ttt at main · fla-org/flash-linear-attention</a>: 🚀 在 Pytorch 和 Triton 中高效实现最先进的线性注意力模型 - fla-org/flash-linear-attention</li><li><a href="https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/demo-training-prepare-v7-pile.sh">RWKV-LM/RWKV-v5/demo-training-prepare-v7-pile.sh at main · BlinkDL/RWKV-LM</a>: RWKV（发音为 RwaKuv）是一种具有出色 LLM 性能的 RNN，也可以像 GPT Transformer 一样直接训练（可并行化）。我们目前处于 RWKV-7 &quot;Goose&quot; 阶段。所以它是...</li><li><a href="https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/">Hymba Hybrid&#x2d;Head Architecture Boosts Small Language Model Performance | NVIDIA Technical Blog</a>: Transformers 凭借其基于注意力的架构，因其强大的性能、并行化能力和长期... 已成为语言模型 (LMs) 的主流选择。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1331050552292806799)** (4 messages): 

> `使用 SAE 特征引导 LLMs，开源引导库` 


- **使用 SAE 引导 LLM 的当前局限性**：成员们指出，使用从训练好的 **SAEs** 中选出的特征来引导 **LLMs** *尚未实现足够的标准化*，这表明该领域仍存在空白。
   - 为了深入了解，他们分享了一个[关于当前 SAE 特征引导方法的讨论](https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484)。
- **可用的开源引导库**：一位成员分享了几个开源引导库，包括 [steering-vectors](https://github.com/steering-vectors/steering-vectors)、[repeng](https://github.com/vgel/repeng) 和 [representation-engineering](https://github.com/andyzoujm/representation-engineering)。
   - 特别是，[Representation Engineering 仓库](https://github.com/andyzoujm/representation-engineering) 专注于从自顶向下的角度研究 AI 透明度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友闲逛，甚至建立全球社区。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://github.com/andyzoujm/representation-engineering">GitHub - andyzoujm/representation-engineering: Representation Engineering: A Top-Down Approach to AI Transparency</a>: 表示工程：一种实现 AI 透明度的自顶向下方法 - andyzoujm/representation-engineering
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329927183023476796)** (63 messages🔥🔥): 

> `Qwen2.5 性能差异, Few-shot 提示词技术, VLLM 评估问题, 量化对性能的影响, MMLU-PRO 评估见解` 


- **Qwen2.5 的性能未达预期**：用户报告 **Qwen2.5-1.5B-Instruct** 及其非指令版本在 **gsm8k** 上的准确率约为 **60%**，而根据其 [官方博客](https://qwenlm.github.io/blog/qwen2.5-llm/)，预期性能分别为 **73%** 和 **65%**。
   - 成员们讨论了评估方法的差异，指出它们可能无法有效地解析答案，从而影响得分。
- **交替问答的 Few-shot 技术**：有人建议将 Qwen 评估中使用的交替问答对 Few-shot 格式整合到 **lm-eval** harness 中，以提升性能。
   - 在应用了 "let's think step by step" 技术后，一位成员注意到性能有所提升，分数提高到了 **66%**。
- **关于 VLLM 评估变异性的讨论**：有人担心使用 **vllm** 与 **HF API** 等其他框架相比时性能结果存在差异，并提到了之前的用户投诉。
   - 尽管一些成员最初怀疑 **vllm** 是性能问题的根源，但其他人对其目前的能力表示了信心。
- **量化对近期模型的影响**：一位成员询问了近期 **llama** 或 **qwen** 模型中 **4bit/3bit vs f16** 的性能退化情况，质疑损失是微不足道的还是取决于具体的量化工作。
   - 他们还在寻求相关学术论文的推荐，以深入了解量化效应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen2.5-llm/">Qwen2.5-LLM: Extending the boundary of LLMs</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介：在本博客中，我们将深入探讨最新的 Qwen2.5 系列语言模型的细节。我们开发了一系列 decoder-only 的密集模型...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9dda03d6be6c94cc803b6189302a8a148c5e4d12/lm_eval/tasks/leaderboard/math/_template_yaml#L1)">lm-evaluation-harness/lm_eval/tasks/leaderboard/math/_template_yaml</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/QwenLM/Qwen/blob/f014f2ef1a72563bbd28b055a4667eaf102c6f21/eval/evaluate_gsm8k.py#L62">Qwen/eval/evaluate_gsm8k.py</a>: 阿里巴巴云提出的 Qwen（通义千问）聊天及预训练大语言模型的官方仓库。 - QwenLM/Qwen</li><li><a href="https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py">Qwen/eval/evaluate_gsm8k.py at main</a>: 阿里巴巴云提出的 Qwen（通义千问）聊天及预训练大语言模型的官方仓库。 - QwenLM/Qwen</li><li><a href="https://github.com/QwenLM/Qwen/blob/f014f2ef1a72563bbd28b055a4667eaf102c6f21/eval/evaluate_chat_gsm8k.py#L23)">Qwen/eval/evaluate_chat_gsm8k.py</a>: 阿里巴巴云提出的 Qwen（通义千问）聊天及预训练大语言模型的官方仓库。 - QwenLM/Qwen</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9dda03d6be6c94cc803b6189302a8a148c5e4d12/lm_eval/tasks/minerva_math/utils.py#L45)">lm-evaluation-harness/lm_eval/tasks/minerva_math/utils.py</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1330798959249326144)** (1 messages): 

> `phi 3 and 3.5 vision, MPS 设备错误` 


- **phi 3 和 3.5 在 MPS 上的错误**：一位成员在尝试在设置了 **MPS** 设备的 Mac 上运行 **phi 3** 和 **phi 3.5 vision** 时遇到错误。
   - 他们报告称 **placeholder storage** 尚未在 **MPS device** 上分配，正在寻求解决方案。
- **寻求 MPS 分配问题的帮助**：该成员正在寻找与使用 **phi 3** 和 **phi 3.5 vision** 时 **MPS** 设备功能相关的任何线索或解决方案。
   - 提到的具体错误表明存在内存分配问题，这可能会阻碍在 Mac 上的成功执行。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1329919555325136990)** (8 条消息🔥): 

> `Host RAM 需求、Vocab Size 优化、结合 ZeRO Stage 1 的 3D Parallelism、针对挂起（Hangs）提交 Issue、更新 Markdown 文件` 


- **Host RAM 和 CPU 核心指南**：Host RAM 应大致等同于 GPU VRAM，而诸如 **CPU Adam** 之类的优化会增加内存需求。通常情况下，每个 GPU 配备 2–4 个核心即可，具体取决于 CPU 架构和流水线复杂度。
   - *一个经验法则是让 Host RAM 等同于所需的 GPU VRAM*，尽管训练通常可以在较少内存下运行。
- **为了效率的 Vocab Size 整除性**：为了优化，Vocab size 应设为 **128*MP** 的倍数，尽管可以强行覆盖此设置，但风险自担。一名成员传达了偏离这一标准的风险。
   - 值得注意的是，*并不强烈建议*覆盖默认设置，因为这可能会导致复杂化。
- **探索 MP、PP 和 ZeRO Stage 1**：成员们讨论了使用 **MP+PP+ZeRO Stage 1** 来优化性能和提高吞吐量的好处。建议激活**内存优化**和 **Flash Attention** 作为有效的增强手段。
   - 据报告，通过这些方法实现了*初始 FLOPs 翻倍*，尽管有人建议对报告的最大 FLOPs 保持谨慎。
- **针对挂起（Hangs）提交 Issue**：一位用户表示打算针对进程中的挂起问题提交 Issue，并询问了有关其设置的详细信息。他们保证在旅途中抽出时间处理此问题。
   - 另一名成员提醒该用户包含详细信息，以便高效解决挂起问题。
- **改进 ARGS Markdown 文件**：有人建议重新导出 **ARGS Markdown 文件**，因为该文件缺少某些参数。这表明可能存在的疏忽，改进后可以帮助用户更清晰地了解用法和配置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/neox_arguments/neox_args.py#L801">gpt-neox/megatron/neox_arguments/neox_args.py at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer 的代码实现 - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329">GitHub - EleutherAI/gpt-neox at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer 的代码实现 - GitHub - EleutherAI/gpt-neox at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/configs/neox_arguments.md?plain=1#L567">gpt-neox/configs/neox_arguments.md at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer 的代码实现 - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1329904328919748639)** (237 条消息🔥🔥): 

> `DeepSeek-R1 发布、Kimi 1.5 论文见解、GRPO 与 RLHF、基准测试评估、MIT 许可协议的影响`

- **DeepSeek-R1 超出预期**：DeepSeek-R1 展示了超越 OpenAI o1 的性能，在 MIT 许可下展现了推理能力的重大进步。
   - 社区对其开源特性感到兴奋，这使其能够应用于各种场景，同时强有力的评估也证明了其有效性。
- **Kimi 1.5 揭示了新的 RL 方法**：关于 Kimi 1.5 的新论文提供了关于奖励塑造（reward shaping）和强化学习（Reinforcement Learning）基础设施的见解，这可能有利于类似模型的开发。
   - 这篇论文预计将激发对正在进行的 RL 研究的兴趣，并可能补充该领域现有的知识框架。
- **简化对 GRPO 的理解**：Natolambert 澄清说，Group Relative Policy Optimization (GRPO) 只是没有价值函数（value function）的 PPO，并依赖于优势（advantage）的 Monte Carlo 估计，从而简化了对 RL 的理解。
   - 这一基础解释旨在让初次接触强化学习方法论的人更容易理解 GRPO。
- **社区对评估指标的反馈**：社区对评估指标的可靠性表达了看法，指出与创建高质量模型相比，操纵评估结果更为容易。
   - 这场对话强调了在 AI 模型开发竞争日益激烈的背景下，稳健评估的重要性。
- **RLHF 和推理的未来方向**：Natolambert 计划将现代 RLHF 的“v1”版本总结成一本简明扼要的书，同时密切关注与 RL 方法论相关的推理领域的发展态势。
   - 对话表明，在节奏飞快的 AI 研究环境中，持续需要清晰的文档和教育。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/teortaxesTex/status/1881331287010550119">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：也许最疯狂的事情是，他们说这还远未达到 7-70B 级别模型的上限。甚至不需要任何新数据。他们已经进一步提升了这些模型，只是不会分享出来。Dri...</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：带有 "Cold Start" 的 DeepSeek R1 表现基本符合预期。我仍然不买账 R1 Zero 的结果，如果不进行微调，基础模型几乎无法输出连贯的解决方案。我敢打赌这里面有...</li><li><a href="https://x.com/RileyRalmuto/status/1880445415927251435">来自 Riley Coyote (@RileyRalmuto) 的推文</a>：@TotalTD0 它肯定在某些方面很专业……但更多是关于健康和长寿方面的</li><li><a href="https://arxiv.org/abs/2403.04642">Teaching Large Language Models to Reason with Reinforcement Learning</a>：来自人类反馈的强化学习 (\textbf{RLHF}) 已成为将 LLM 输出与人类偏好对齐的主流方法。受 RLHF 成功的启发，我们研究了...的性能</li><li><a href="https://x.com/TheXeophon/status/1881305033352135152">来自 Xeophon (@TheXeophon) 的推文</a>：R1 定价</li><li><a href="https://x.com/DanHendrycks/status/1881045781354000604">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>：Humanity's Last Exam 将在下周发布，因此我们可以用它来测试模型的科研级 STEM 能力。</li><li><a href="https://x.com/LiquidAI_/status/1881236162893000944">来自 Liquid AI (@LiquidAI_) 的推文</a>：介绍 LFM-7B，我们全新的同类最佳语言模型，支持英语、阿拉伯语和日语，经过优化可作为私有企业聊天、代码、快速指令遵循和 Agentic 工作流的基础...</li><li><a href="https://x.com/teortaxesTex/status/1881245237982724296">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@tokenbender 我们需要一个 Whale 入场主题曲</li><li><a href="https://x.com/teortaxesTex/status/1881330229119246843">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：“你们自己动手做点什么吧，我们已经把该说的都告诉你们了”</li><li><a href="https://x.com/spectatorindex/status/1881054674620703145">来自 The Spectator Index (@spectatorindex) 的推文</a>：突发：TikTok 在美国恢复服务后的消息</li><li><a href="https://x.com/TheXeophon/status/1881444595009253543">来自 Xeophon (@TheXeophon) 的推文</a>：这是我在测试基准中最喜欢的例子之一。模型应该检测到不必要的 softmax 并通知用户。R1 得到了 4/5 —— 唯一的一次失败是 LLM-as-judge (4o) 没有正确判断...</li><li><a href="https://x.com/teortaxesTex/status/1881331554456371544">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@natolambert 的全面胜利</li><li><a href="https://x.com/sama/status/1880356297985638649">来自 Sam Altman (@sama) 的推文</a>：感谢测试 o3-mini 的外部安全研究人员。我们现在已经敲定了一个版本并开始发布流程；计划在约几周内发布。此外，我们听取了反馈...</li><li><a href="https://x.com/Teknium1/status/1881267038091682191">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：搞定了一个 DeepSeek 推理模型的推理 ^_ ^</li><li><a href="https://x.com/natolambert/status/1881380809153847711">来自 Nathan Lambert (@natolambert) 的推文</a>：对于那些试图理解 DeepSeek 的 Group Relative Policy Optimization (GRPO) 的人：GRPO 就是没有价值函数、使用蒙特卡洛估计 advantage 的 PPO。所以，去研究为什么 PPO 存在 (lo...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265">来自 Xeophon (@TheXeophon) 的推文</a>：我被 R1 在我个人测试基准上的表现震惊了。这是完整的评估集，它完全碾压了竞争对手，自成一档，甚至超过了 o1-preview（图中省略了...）</li><li><a href="https://x.com/StringChaos/status/1880317308515897761">来自 Naman Jain (@StringChaos) 的推文</a>：DeepSeek-R1 (Preview) 结果 🔥 我们与 @deepseek_ai 团队合作，在 LiveCodeBench 上评估了 R1 Preview 模型。该模型的表现接近 o1-Medium，提供了 SOTA 级别的推理性能...</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 DeepSeek-R1 来了！⚡ 性能与 OpenAI-o1 旗鼓相当 📖 完全开源的模型和技术报告 🏆 MIT 许可证：自由蒸馏和商业化！🌐 网站和 API 已上线！在 h... 尝试 DeepThink</li><li><a href="https://x.com/iforgotmytwit1/status/1881314212578046060">来自 dhfksowndic (@iforgotmytwit1) 的推文</a>：@teortaxesTex</li><li><a href="https://x.com/natolambert/status/1881370064038805714">来自 Nathan Lambert (@natolambert) 的推文</a>：哈哈哈哈哈哈 t</li>

今天实际上有两份关于 RL 推理模型的技术报告，Kimi 1.5 在 reward shaping + RL infra 方面也有很多不错的内容</li><li><a href="https://fxtwitter.com/btibor91/status/1881285255266750564">Tibor Blaho (@btibor91) 的推文</a>：OpenAI 网站已经出现了关于 Operator/OpenAI CUA (Computer Use Agent) 的引用 —— “Operator System Card Table”、“Operator Research Eval Table” 和 “Operator Refusal Rate Table”...</li><li><a href="https://x.com/btibor91/status/1880950883988738482">Tibor Blaho (@btibor91) 的推文</a>：来自 Google 的新型 “Gemini 2.0 Flash Thinking Experimental” (gemini-2.0-flash-thinking-exp 01-23) 推理模型 (01:02:57) 感谢 @sir04680280 引用 Alex Reibman 🖇️ (@AlexReibman) 的直播...</li><li><a href="https://x.com/deepseek_ai/status/1881318138937233664">DeepSeek (@deepseek_ai) 的推文</a>：📜 许可证更新！🔄 DeepSeek-R1 现在采用 MIT 许可证，以实现清晰的开放获取 🔓 向社区开放以利用模型权重 (model weights) 和输出 🛠️ API 输出现在可用于 fine-tuning 和 distillation 🐋 ...</li><li><a href="https://tenor.com/view/the-pursuit-of-happiness-will-smith-success-joy-happiness-gif-3517714">幸福 GIF - 《当幸福来敲门》威尔·史密斯 成功 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/jiayi_pirate/status/1881264063302557919">Jiayi Pan (@jiayi_pirate) 的推文</a>：@pshishodia_ @Grad62304977 @teortaxesTex @TheXeophon @nrehiew_ 从模型权重来看，R1、R1-zero、V3-instruct 彼此之间都有很大不同，而 R1-zero 最接近 V3-base。它们可能都...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建一个账号来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">推理模型 (deepseek-reasoner) | DeepSeek API 文档</a>：deepseek-reasoner 是由 DeepSeek 开发的推理模型。在交付最终答案之前，模型首先生成思维链 (Chain of Thought, CoT) 以增强其响应的准确性。我们的 API 提...</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5/tree/main">GitHub - MoonshotAI/Kimi-k1.5</a>：通过在 GitHub 上创建一个账号来为 MoonshotAI/Kimi-k1.5 的开发做出贡献。
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1330065111670587422)** (27 messages🔥): 

> `O1 pro streaming summary, Test-time search vs forward passes, Use of self-consistency in reasoning, Gflownet in training O1, Asymmetry in RL setups` 


- **O1 pro 在推理时流式传输思维摘要**：一位成员观察到 **O1 pro** 会在思维产生时流式传输摘要，这表明它在推理过程中合并了并行生成的内容，而不是在结束时才进行。
   - *流式摘要可能意味着中间选择（intermediate selection）*，而非最终的样本选择。
- **关于 Test-time search 解释的辩论**：讨论围绕 **Francois Chollet** 的推文展开，该推文解释称模型的即时响应意味着少于 10 次 **forward passes**，而较长的响应则涉及 **test-time search**。
   - 一些成员认为这种解释可能无法准确反映 **O1 pro** 在推理过程中的实际运作方式。
- **关于训练中潜在推理路径的理论**：**Chygao** 假设 O1 的训练涉及使用 **Gflownet** 等方法来推导潜在推理路径，并引用了一篇在 **ICLR 2024** 获得关注的论文。
   - 该论文探讨了如何通过**贝叶斯推理（Bayesian inference）**推导出通向答案的隐藏**思维链（chains of thought）**。
- **关于 RL 不对称性问题的讨论**：**Catboy_slim_** 询问在其 **RL** 设置中对负例和正例进行不对称裁剪（clipping）是否为刻意为之，最终确认这在标准 **PPO** 配置中很常见。
   - 这种不对称性可能会软化正例的影响，同时加剧负例的影响，引发了关于稳定性合理性的疑问，而这些理由与其数学模型并不完全一致。
- **理解 RL 中的奖励与惩罚**：在 **RL** 讨论中，**Natolambert** 强调在传统设置中，负值等同于失败，而小额奖励则类似于进展。
   - 这一概念与训练中非标准裁剪方法的合理性相一致，尽管它引发了关于与底层模型数学逻辑相互作用的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1863226363542454660">wh (@nrehiew_) 的推文</a>：这篇论文在 ICLR 2024 获得了荣誉提名，第一作者曾参与 o1 工作并且是 loratldr 的创建者：他们提出了一种方法，在给定问题的情况下，推导出通向答案的隐藏思维链（cot）...</li><li><a href="https://x.com/natolambert/status/1880683907563299233">Nathan Lambert (@natolambert) 的推文</a>：作为经验法则，这比有用的信息更令人困惑。人们认为的带有分支、从错误中获取信用分配（credit assignment）的大多数“搜索”都发生在训练阶段。一些微妙的实现...</li><li><a href="https://x.com/fchollet/status/1880378458909601969">François Chollet (@fchollet) 的推文</a>：通常规则是：如果模型立即返回响应，它就没有进行 test-time search —— 它在进行少于 10 次的 forward passes。如果它需要 5 分钟以上才返回内容... 它就在进行 test-time...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329915079801573439)** (51 messages🔥): 

> `MosaicAI 离职潮, OpenAI 透明度问题, Epoch AI 与 FrontierMath, Perceptron Inc 的新创项目, AGI 热议` 


- **MosaicAI 经历人员离职**：最近的消息指出 **MosaicAI** 有多名员工离职，成员们在表达对职位的感激之情的同时，也反思了公司内部面临的挑战。
   - 一位离职成员指出，*“在 @DbrxMosaicAI 工作是我一生的荣幸，”* 随后他们将转向 AI 领域的新机会。
- **对 OpenAI 透明度的担忧**：有关 **OpenAI** 缺乏合作伙伴关系透明度的讨论浮出水面，特别是涉及 **Epoch AI** 及其在 **FrontierMath** 数据集上的工作。
   - 成员们表示 *“OpenAI 想要对资金来源保密”*，这引发了关于此类行为对 AI 研究诚信影响的质疑。
- **Epoch AI 对透明度的承诺**：在承认差异后，**Epoch AI** 承诺在未来的合作中提高数据访问和资金来源的透明度。
   - 一位代表表示，*“我们本应该在争取透明度方面进行更强硬的谈判……”*，强调了他们未来致力于更好沟通的决心。
- **Perceptron Inc. 发布视觉基础模型**：一位前 **MosaicAI** 研究员宣布了他们在 **Perceptron Inc.** 的新职位，专注于为实时视频感知创建视觉语言基础模型（Visual Foundation Models），并承诺资源成本仅为现有模型的 1/100。
   - 他们分享了与才华横溢的同事共事的兴奋之情，表示：*“我绝对相信，如果有人能解决这个问题，那就是他们。”*
- **对 AGI 推测的反应**：**Sama** 的一条推文回应了围绕即将部署 AGI 的离奇推测，安抚社区要 *“冷静，并将你的预期降低 100 倍！”*
   - 这种情绪引起了许多人的共鸣，反映了当前关于 “AGI” 一词被频繁误用以及它如何助长不切实际预期的持续争论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/plain_simon/status/1880949628751011846">来自 Simon Pepin Lehalleur (@plain_simon) 的推文</a>: @ElliotGlazer “正在开发”？Lesswrong 的评论强烈暗示这个预留集（holdout set）已经存在并用于验证（“serves” 是现在时）。你能澄清一下吗？</li><li><a href="https://x.com/TheRealAdamG/status/1881349799888433548">来自 Adam.GPT (@TheRealAdamG) 的推文</a>: 并非所有的“思考”都是一样的。我预计会看到糟糕的思维链（chains of thoughts）激增。</li><li><a href="https://x.com/mvpatel2000/status/1880802915704820004?s=46">来自 Mihir Patel (@mvpatel2000) 的推文</a>: 3 年后，我在 @DbrxMosaicAI 度过了最后一周。我加入 MosaicML 时非常兴奋能参与初创公司并从事前沿 AI 工作。离开时，我经历了一段过山车般的旅程，并收获了终生的朋友。</li><li><a href="https://x.com/jecdohmann/status/1881418279945978261">来自 Jeremy Dohmann (@jecdohmann) 的推文</a>: 我非常高兴地宣布，我将加入 @perceptroninc (https://perceptron.inc/?) 担任研究员和创始技术人员。我将与 @AkshatS07 和 @ArmenAgha 一起工作...</li><li><a href="https://x.com/DanHendrycks/status/1881036645555937719">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>: @GaryMarcus 可以确认，由于 Epoch 与 OpenAI 的合同义务，像 xAI 这样的 AI 公司无法获得 FrontierMath 的访问权限。</li><li><a href="https://x.com/ElliotGlazer/status/1881016863343390946">来自 Elliot Glazer (@ElliotGlazer) 的推文</a>: @plain_simon “serves” 的意思是 “这是它的用途”，但我会请 Tamay 将其改为将来时。预留集的评估分数将是公开的，所以每个人都会知道它何时执行。</li><li><a href="https://x.com/code_star/status/1880355601546674203">来自 Cody Blakeney (@code_star) 的推文</a>: 上周是我在 @DbrxMosaicAI 的最后一周。我非常感激能成为这样一个了不起的团队和旅程的一员。在这三年里，我学到了很多关于初创生态系统的知识，参与了一个成功的...</li><li><a href="https://x.com/sama/status/1881258443669172470">来自 Sam Altman (@sama) 的推文</a>: Twitter 的炒作又失控了。我们下个月不会部署 AGI，也没有开发出它。我们有一些非常酷的东西要给你们，但请冷静，并将你的预期降低 100 倍！</li><li><a href="https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/?commentId=veedfswdCYKZEhptz">meemi 的短文 — LessWrong</a>: Tamay 的评论 - 我是来自 Epoch AI 的 Tamay。我们在 OpenAI 的参与度方面不够透明，这是我们的错误。在 o3 发布前后，我们被限制披露合作伙伴关系...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329903246965215233)** (76 条消息🔥🔥): 

> `Molmo AI, DeepSeek 模型见解, VLM 性能, Trae AI IDE, 中国初创公司格局` 


- **Molmo AI 备受关注**：成员们对 [Molmo AI](https://molmo.org/) 表现出极大的热情，强调了其在多模态处理和用户友好性方面的能力，并声称其表现优于许多现有的 VLM。
   - 讨论涉及了它的优势，例如能很好地适应各种任务，尽管对其偶尔出现的错误仍有保留意见。
- **DeepSeek 模型讨论**：在关于 DeepSeek 性能的讨论中，成员们提到了其最新模型在显著改进图像和语言理解相关任务方面的潜力。
   - 关于未来深入探讨新发布内容的博客文章的猜测广为流传，表明人们对详细见解有着浓厚的兴趣。
- **视觉语言模型 (VLM) 的挑战**：社区讨论了 VLM 在检测任务中的局限性，几位贡献者质疑当前模型在图像中准确进行目标定位的能力。
   - 有人建议改进可能来自于对 PASCAL-VOC 等数据集应用的 fine-tuning 技术，而另一些人则认为视觉 token embeddings 的复杂性阻碍了局部信息的恢复。
- **Trae AI IDE 亮相**：由 Bytedance 开发的自适应 AI IDE Trae 正式推出，声称将改变编程环境中的协作和生产力。
   - 值得注意的是，Bytedance 工程师幽默地表示 Trae 代表 “The real ai engineer”，将其定位为开发者的工具。
- **付费墙动态讨论**：关于为独家内容引入付费墙有一些轻松的玩笑，建议提供摘要并限制对深度见解的访问。
   - 成员们反思了学术界付费墙的影响，在知识获取的需求与财务可持续性之间寻找平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.11441">Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V</a>：我们提出了 Set-of-Mark (SoM)，这是一种新的视觉提示方法，旨在释放大型多模态模型 (LMMs)（如 GPT-4V）的视觉定位能力。如图 1（右）所示，我们采用了...</li><li><a href="https://molmo.org/">Molmo AI: Open-Source Multimodal AI Model | Free &amp; Powerful</a>：探索 Molmo AI，这是最先进的开源多模态 AI 模型。功能强大、免费且易于使用。了解 Molmo 与其他 AI 模型的对比。</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>：本文介绍了在极小数据集上过拟合预训练大型语言模型 (LLMs) 所产生的反直觉泛化结果。在开放式文本生成的设置下，它...</li><li><a href="https://www.ciciai.com/">Cici</a>：Cici AI 是您的 AI 聊天助手，用于智能对话、写作、翻译、情感支持和编程。从 Cici AI 获取答案、寻找灵感并讨论任何话题。</li><li><a href="https://x.com/MLiegertova/status/1880674661731828214">Michaela Lie ☕☕☕ (@MLiegertova) 的推文</a>：@JeremyNguyenPhD 像专业人士一样计数！😁</li><li><a href="https://x.com/TheXeophon/status/1880513932609323060">Xeophon (@TheXeophon) 的推文</a>：@JeremyNguyenPhD @vikhyatk 我玩过一些模型，我认为 Molmo 是最好的——它并不完美（漏掉了两个），但你可以很容易地看到这一点，因为模型可以指向它。这本可以节省...</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">mgostIH (@mgostIH) 的推文</a>：深度学习到底怎么了？？？</li><li><a href="https://www.trae.ai/">Trae - Ship Faster with Trae</a>：未找到描述</li><li><a href="https://x.com/dylan522p/status/1880379652054901175">Dylan Patel (@dylan522p) 的推文</a>：埃隆的飞机在佛罗里达。格芯（Global Foundries）的飞机在佛罗里达。高通（Qualcomm）的飞机在佛罗里达。如果有人想知道 Intel 发生了什么...他们在海湖庄园（Mar-a-Lago）。让美国再次伟大...</li><li><a href="https://arxiv.org/abs/2401.06209">Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs</a>：视觉对语言来说足够好吗？多模态模型的最新进展主要源于大型语言模型 (LLMs) 强大的推理能力。然而，典型的视觉组件...</li><li><a href="https://youtu.be/76EL7YVAwVo?si=Xwu17VAzJkd6YiNQ&t=1254)">Best of 2024 in Vision [LS Live @ NeurIPS]</a>：来自 Roboflow 的 Peter Robicheaux 和 Isaac Robinson 以及 Moondream 的 Vik Korrapati 回顾了 2024 年前沿/开源视觉模型领域的最佳工作！包含幻灯片和展示...</li><li><a href="https://mp.weixin.qq.com/s/XGnHruXL3P0s-2TNss0LIg">晚点对话 MiniMax 闫俊杰：千万别套用移动互联网的逻辑来做 AI</a>：“创业没有天选之子。”
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1329938101388312657)** (2 条消息): 

> `模糊发布 (Vagueposting), AI 护城河 (AI Moats), Amanda Askell` 


- **模糊发布 (Vagueposting) 达到新高度**：一位成员分享了一张名为“模糊发布终局”的图表，强调了在线空间中模糊沟通的趋势。附带的图片暗示了解密现代数字对话的复杂性。
   - **模糊发布 (vagueposting)** 的视觉呈现促使观众思考不清晰信息传递的更广泛影响，引发了进一步讨论。
- **关于 AI 最后一道护城河的讨论**：一位成员引用了一条推文，声称“AI 剩下的唯一**护城河 (moat)** 就是 Amanda Askell”，引发了关于该领域竞争优势的对话。
   - 这一表态反映了在快速发展的 AI 领域中，人们对于**知识产权**和**独特见解**日益增长的看法。



**提到的链接**：<a href="https://x.com/menhguin/status/1881387910316052723?s=61">Minh Nhat Nguyen (@menhguin) 的推文</a>：AI 剩下的唯一护城河就是 Amanda Askell

  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1330916825596428429)** (6 条消息): 

> `Reinforcement Learning for Robotics, Vision & Language Models, Computer Vision Reinforcement Learning, Robotics Perception Models` 


- **探索用于 Robotic Control 的 RLVR**：一位成员询问了 **RLVR** 在使用 **VLMs** 和 **CoT** 生成格式为 'move to (0.41, -7.8)' 的指令进行 **Robotic Control** 中的适用性。
   - 另一位成员表示乐观，认为这是一种目前看来行之有效的方法。
- **旧思想的复兴**：讨论强调，在 Robotics 领域，旧的想法往往会显得焕然一新，尤其是在 **Reinforcement Learning** 方面。
   - 随着投票重新转向这些经久不衰的概念，对过去想法的深入探索似乎变得很有必要。
- **Reinforcement Learning 的 Computer Vision 应用**：一位成员分享了 **Lucas Beyer et al.** 的论文，讨论了使用 **Reinforcement Learning** 技术使模型与 **Computer Vision** 中的任务奖励保持一致，链接见 [此处](https://arxiv.org/abs/2302.08242)。
   - 该论文声称，通过解决模型失配（misalignment）问题，在 **Object detection** 和 **Image captioning** 等任务中对齐模型是有效的。
- **将 Reinforcement Learning 与 CoT 方法结合**：有人对在 **Computer Vision** 背景下如何将 **Reinforcement Learning** 方法与 **Chain of Thought (CoT)** 方法论融合表示好奇。
   - 同时也出现了关于在使用 **Reinforcement Learning** 的任务中，**Computer Vision** 标签作为“已验证（verified）”标签的可靠性担忧。
- **Perception Models 时间线难题**：一位成员幽默地建议，在第四季度交付预期的 **Perception Models** 的同时，制定一个为期六个月的实验时间线来彻底改变 Robotics 技术。
   - 这个俏皮话暗示了在管理标准交付成果的同时，对创新想法的雄心勃勃的追求。



**提到的链接**：<a href="https://arxiv.org/abs/2302.08242">Tuning computer vision models with task rewards</a>：模型预测与预期用途之间的失配（Misalignment）可能会对 **Computer Vision** 模型的部署产生不利影响。当任务涉及复杂的结构化输出时，这个问题会更加严重...

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329939648822247496)** (21 条消息🔥): 

> `AI 应用的 Post-Training、Devin 与 Cursor 的挑战、AI 研究者的过度估计、强化学习 (RL) 讨论、SOP-Agents 框架` 


- **探索 Post-Training 策略**：一场名为 [How to approach post-training for AI applications](https://youtu.be/grpc-Wyy-Zg) 的演讲在 NeurIPs 期间分享了见解，重点关注 AI 开发的有效策略。
   - 参与者一致认为，在没有做好充分基础工作的情况下直接投入模型训练是一个陷阱。
- **Devin vs. Cursor：评价褒贬不一**：一位成员分享了他们团队的经验，表示由于对 **Devin** 的表现不满，他们在一周内就放弃了 **Devin** 转而使用 **Cursor**。
   - *传闻指出*，该 coding agent 使用的是 **gpt-4o**，与 **Claude** 等替代方案相比，它在编程任务上的表现可能并不理想。
- **AI 扩散速度被过度估计**：讨论源于一次 [Tyler Cowen 的访谈](https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5)，强调 AI 研究者往往 *过度估计* 了技术扩散的速度。
   - 成员们对此观点表示赞同，并引发了对以 LLM 为中心的初创公司不愿探索替代模型的思考。
- **强化学习 (RL)：日益增长的兴趣**：成员们讨论了理解 **Reinforcement Learning (RL)** 迫切增长的需求，其中一人表示在未来几周内学习它是不可避免的。
   - 他们对缺乏专门针对 *语言模型 RL* 的资源感到沮丧。
- **SOP-Agents 框架介绍**：[SOP-Agents](https://arxiv.org/abs/2501.09316) 框架的引入旨在通过使用 **Standard Operational Procedures** (标准作业程序) 来增强 AI Agent 的规划能力。
   - 这一新颖的框架旨在通过决策图引导 AI Agent，从而解决任务完成中的局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09316">SOP-Agent: Empower General Purpose AI Agent with Domain-Specific SOPs</a>: 尽管通用 AI Agent 取得了显著进展，但在现实场景的实际应用中仍面临若干挑战。首先，Large La... 的有限规划能力</li><li><a href="https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5."> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/grpc-Wyy-Zg">How to approach post-training for AI applications</a>: 我在 NeurIPs 期间在 Infer —— 温哥华 AI 工程小组的演讲：https://infervan.com/。这是一次有趣的经历。我当时在思考对 AI ... 该“说些什么”</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: 通过在 GitHub 上创建一个账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1330579335245070437)** (13 messages🔥): 

> `RLHF Book 进展, 结果奖励模型 (Outcome Reward Models), CS329A 课程概览, 奖励建模技术, 价值网络 (Value Networks)` 


- **RLHF Book 进展引发期待**：[RLHF Book](https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models) 的编写正在取得进展，特别是关于巨型策略梯度 (policy gradient) 的页面，预计会非常有用。
   - 希望能尽快邀请 Ross 回到播客详细讨论这些话题。
- **结果奖励模型 (Outcome Reward Models) 的区分**：一位成员指出，结果奖励模型 (ORMs) 适用于无法通过编程方式对结果进行评分的情况，类似于在强化学习中使用代理 (proxies)。
   - ORMs 有助于数据过滤，并能通过提供每个 token 产生正确结果的概率来辅助强化学习过程。
- **CS329A 课程备受关注**：CS329A 研究生研讨课已经发布了讲义以及一份引人入胜的 [课程概览](https://cs329a.stanford.edu/#schedule)，涵盖了前沿的 AI 技术。
   - 参与者对发现一份充满关于 LLM 自我改进 (self-improvement) 精彩论文的新阅读清单感到兴奋。
- **奖励建模技术探讨**：奖励建模在现代 RLHF 方法中至关重要，通过 Bradley-Terry 等模型衡量偏好，详见 [RLHF Book](https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models)。
   - 成员们讨论了这些模型如何与强化学习中的价值对齐以及训练算法的重要性。
- **价值网络 (Value Networks) 提供未来预测**：价值网络被用于预测与特定 token 相关的未来回报，展示了与 AI 建模中 ORMs 不同的角色。
   - 理解这些区别强调了在强化学习框架中选择正确工具的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cs329a.stanford.edu/#schedule">Stanford CS329A | Self-Improving AI Agents</a>: 未找到描述</li><li><a href="https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models">A Little Bit of Reinforcement Learning from Human Feedback</a>: Reinforcement Learning from Human Feedback 书籍
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1330171044635475978)** (3 messages): 

> `Meta 眼镜集成, WhatsApp 机器人功能` 


- **将 Meta & Rayban 眼镜与 WhatsApp 集成**：一个名为 [meta-glasses-gemini](https://github.com/josancamon19/meta-glasses-gemini) 的 GitHub 项目探索了通过机器人将 **Meta + Rayban Glasses** 与 WhatsApp 集成。
   - *这种集成允许用户有效地控制眼镜功能，* 展示了增强用户交互的潜力。
- **社区对该集成想法的反应**：一位成员对这个集成想法幽默地评论道：*“喜欢这种胡闹 (nonsense)。”*
   - 这反映了社区内对非传统技术集成的一种戏谑性的怀疑态度。



**提到的链接**: <a href="https://github.com/josancamon19/meta-glasses-gemini">GitHub - josancamon19/meta-glasses-gemini: Meta + Rayban Glasses whatsapp bot integration</a>: Meta + Rayban Glasses WhatsApp 机器人集成。欢迎访问 GitHub 参与开发。

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1331053678119096411)** (3 messages): 

> `AI 行政命令, NAIRR 活动` 


- **美国总统撤销重大 AI 行政命令**：美国总统撤销了前任政府的主要 AI 行政命令，即 **EO 14110**。这一变化引发了关于美国 AI 监管影响和未来的疑问。
   - *“那个命令到底是干什么的？”* 是寻求澄清该行政命令先前条款的成员们的共同疑问。
- **对即将举行的 NAIRR 活动的好奇**：一位成员对受邀参加的 2 月份 NAIRR 活动表示不确定，想知道活动是否仍会举行。这反映了在监管持续变化的情况下，人们对活动规划的普遍疑虑。



**提到的链接**: <a href="https://x.com/cfgeek/status/1881494093215551954?s=61">Charles Foster (@CFGeek) 的推文</a>: 美国总统撤销了前任政府的主要 AI 行政命令 (EO 14110)。

  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1331027328418975885)** (1 条消息): 

> `Aider v0.72.0 发布，支持 DeepSeek R1，支持 Kotlin 语法，文件写入增强，Bug 修复更新` 


- **Aider v0.72.0 发布，带来多项新功能**：新版本 **Aider v0.72.0** 引入了对 **DeepSeek R1** 的支持，快捷方式为 `--model r1` 和 `--model openrouter/deepseek/deepseek-r1`。
   - 此版本还增强了 **GPT-4o 模型** 的 `examples_as_sys_msg=True` 设置，提升了基准测试得分。
- **Kotlin 语法成为焦点**：贡献者 **Paul Walker** 为 repo map 添加了新的 **Kotlin 语法支持**。
   - 此增强功能旨在提升 Kotlin 在当前框架内的可用性。
- **实现了文件写入改进**：由 **Titusz Pan** 添加的 `--line-endings` 用于文件写入，旨在提高格式一致性。
   - 这一改进体现了在文件操作中提升代码质量的承诺。
- **多项 Bug 修复增强了稳定性**：最近的修复包括 **Docker** 镜像中的**权限问题**，以及修复了轮换过程中的 **lint/test** 错误。
   - 此外，还实现了针对 **unicode 错误的 ASCII 回退方案**，并修复了 repomap 计算中的整数索引问题。
- **Aider 在编码中发挥了重要作用**：有趣的是，**Aider** 贡献了此版本中 **52% 的代码**，凸显了其日益增长的能力。
   - 这种参与程度表明了其对持续改进和创新增强的承诺。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1329913008050274315)** (334 条消息🔥🔥): 

> `DeepSeek R1 性能，Aider 基准测试，Kimi k1.5 模型，AI 模型中的数据隐私，本地模型使用` 


- **DeepSeek R1 与其他模型的性能对比**：用户对 **DeepSeek R1** 在 **Aider** 中的表现褒贬不一，指出它会犯一些错误，尤其是在处理简单任务时。
   - 尽管人们对 OpenAI **o1** 的廉价替代方案感到兴奋，但一些人发现 **R1** 的输出未达预期，因此建议将其与其他编辑模型配对使用。
- **Aider 基准测试和模型选择**：**DeepSeek R1** 模型在 **Aider** 编码排行榜上达到了 57%，引发了关于其相对于 **o1** 模型和其他竞争对手表现的讨论。
   - 关于 **R1** 对“思考（thinking）”响应的依赖是否比其他模型更能增强推理能力，观点各不相同，一些用户更倾向于使用更简单的模型来处理基础任务。
- **Kimi k1.5 表现优于主流模型**：据报道，新的 **Kimi k1.5** 多模态模型在多项基准测试中优于 **GPT-4o** 和 **Claude Sonnet 3.5**，特别是在推理任务方面。
   - **Kimi k1.5** 的特性包括支持高达 128k **tokens** 的长上下文扩展，这可能会扩大其在生成任务中的适用性。
- **AI 中的数据隐私担忧**：用户讨论了 AI 数据使用的透明度，强调虽然像 **DeepSeek** 这样的公司公开声明他们利用用户数据，但其他公司的说明则不太明确。
   - 人们对大型企业在处理用户数据和训练模型时的可信度表示担忧。
- **本地模型使用体验**：个人报告了在本地使用蒸馏模型的积极体验，指出在早期交互中响应更加全面。
   - 有建议称，在本地使用 **R1** 模型可以帮助处理更复杂的场景，通过对日志提供深思熟虑的反应，而无需明确的指令。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/paulgauthier">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security)">数据隐私与安全 - Fireworks AI 文档</a>：未找到描述</li><li><a href="https://docs.grit.io/tutorials/gritql">GritQL 教程</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API、供应商、统计数据</a>：DeepSeek-R1 来了！⚡ 性能媲美 OpenAI-o1 📖 完全开源的模型和技术报告 🏆 MIT 许可证：自由蒸馏和商业化！。通过 API 运行 DeepSeek R1</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准。</li><li><a href="https://medium.com/@hamzaennaffati98/cache-augmented-generation-cag-vs-retrieval-augmented-generation-rag-7b668e3a973b">缓存增强生成 (CAG) vs. 检索增强生成 (RAG)</a>：哪种方法更胜一筹？</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">来自 Kimi.ai (@Kimi_ai_) 的推文</a>：🚀 介绍 Kimi k1.5 --- 一款 o1 级别的多模态模型。Sota 级别的短 CoT 性能，在 📐AIME、📐MATH-500、💻 LiveCodeBench 上大幅超越 GPT-4o 和 Claude Sonnet 3.5（最高达 +550%...</li><li><a href="https://unsloth.ai/blog/deepseek-r1">运行 Deepseek-R1 / R1 Zero</a>：DeepSeek 最新的 R-1 模型是目前最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。了解如何运行和微调该模型。</li><li><a href="https://x.com/deepseek_ai/status/1881318135850213834">来自 DeepSeek (@deepseek_ai) 的推文</a>：🔥 奖励：开源蒸馏模型！🔬 从 DeepSeek-R1 蒸馏出的 6 个小模型已完全开源 📏 32B 和 70B 模型与 OpenAI-o1-mini 相当 🤝 赋能开源社区 🌍 推动...</li><li><a href="https://www.youtube.com/@techfren">techfren</a>：开源 AI 和其他技术。立即订阅以观看直播！</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt">OpenAI CEO 不知道如何应对 ChatGPT 订阅者的行为</a>：他没怎么多想就选定了价格，以为能赚到钱</li><li><a href="https://gist.github.com/murdockq/b08f72699fd7d8db556a14e69a7cb0c3">a game prompt.md</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://docs.fireworks.ai/guides/security_comp">简介 - Fireworks AI 文档</a>：未找到描述</li><li><a href="https://x.com/0xluffyb/status/1881323971897110866">来自 luffy (@0xluffyb) 的推文</a>：今天的每个人。引用 DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 来了！⚡ 性能媲美 OpenAI-o1 📖 完全开源的模型和技术报告 🏆 MIT 许可证：自由蒸馏和商业化！🌐 ...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.litellm.ai/docs/completion/function_call">Function Calling | liteLLM</a>：检查模型是否支持函数调用</li><li><a href="https://www.youtube.com/@echohive">echohive</a>：我花了 3000 多个小时学习和编写了 300 多个项目，并在这些 YouTube 视频中分享了我学到的一切。希望你会发现它们有用 :) 搜索所有 echohive 视频：https://w...</li><li><a href="https://www.youtube.com/@AllAboutAI">All About AI</a>：欢迎来到我的频道 All About AI =) 网站：https://aiswe.tech。你如何开始使用生成式 AI 来帮助你完成创意或其他日常任务。因此，我的目标是将生成式 AI 带给每个人。- AI En...</li><li><a href="https://www.youtube.com/@AIJasonZ">AI Jason</a>：我叫 Jason Zhou，一名分享有趣 AI 实验和产品的产品设计师。如果你在构建 AI 应用方面需要帮助，请给我发邮件！- 加入社区：https://www.skool.com/ai-builder-club/about...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: ReAct 循环中的 Aider</a>：ReAct 循环中的 Aider。通过在 GitHub 上创建账号来为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#agents-for-developers:~:text=General%20availability%20will%20follow%20in%20January%2C%20along">Google Gemini AI 2024 年 12 月更新</a>：正式版将于 1 月推出，以及...</li>

<li><a href="https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#introducing-gemini-20-with-more-model-sizes.">介绍 Gemini 2.0：我们为 Agent 时代打造的新型 AI 模型</a>：今天，我们宣布推出 Gemini 2.0，这是我们迄今为止功能最强大的 AI 模型。</li><li><a href="https://youtube.com/@marvijosoftware?si=CpNJZ8UmLJyp2mk">Marvijo AI Software</a>：我们探索前沿的 Cloud、Software Engineering 和 Computer Science 技术。- Artificial Intelligence - LLMs (Large Language Models) - AI Coding - Microsoft Azure - 易于理解的解释...</li><li><a href="https://youtube.com/@appydave?si=0nvFeqOcIZxuJCHA">AppyDave</a>：欢迎来到 AppyDave（原名 AppyCast），这是一个创新与对话交汇的中心。我的使命是为您提供知识和工具，以利用 ChatGPT 的潜力，这是一种正在重新...</li><li><a href="https://youtube.com/@codingthefuture-jg1he?si=Eag1-kRT23z8Jys8">Coding the Future With AI</a>：欢迎来到 Coding the Future With AI！我们的频道致力于帮助开发者和技术爱好者学习如何利用 AI 来提升技能和生产力。通过教程、专家...</li><li><a href="https://github.com/Aider-AI/aider/issues/429">Tree-sitter tsx 解析器有时会挂起，导致 aider 挂起 · Issue #429 · Aider-AI/aider</a>：用户报告在使用充满 .tsx 文件的 repo 时 aider 会挂起。使用 --no-git 可以消除挂起。问题似乎出在 repo map 代码中。https://discord.com/channels/1131200896827654144/1192136795...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329944218667454495)** (74 messages🔥🔥): 

> `Aider 与语言模型的使用、OpenRouter 对比 Anthropic API、DeepSeek 模型问题、Aider 中的文件管理、API Key 配置问题` 


- **使用 Aider 配合 LLM 进行编程**：用户讨论了将 Aider 与 **DeepSeek v3** 和 **Qwen 2.5 Coder** 等模型配合使用的效果，并注意到了 context window 设置和性能预期。
   - 几位用户提到在 **Architect mode** 下需要使用 `/copy-context` 命令来维持聊天历史，以获得更好的回复。
- **在 OpenRouter 和 Anthropic API 之间做出选择**：一位用户询问了相比 **Anthropic API** 更倾向于使用 **OpenRouter** 的原因，引发了关于 Anthropic 施加的更严格限制的讨论。
   - 其他人确认 **OpenRouter** 通常提供更灵活的 API limits，使其成为 Aider 用户中更受欢迎的选择。
- **DeepSeek 模型响应问题**：用户报告了与 **DeepSeek** 不支持连续的用户或助手消息相关的错误，以及 Aider 中间歇性的性能问题。
   - 一些用户建议更新 Aider 并检查模型设置以解决这些错误。
- **Aider 中的文件管理和自动补全**：讨论了 `/add` 命令不显示可选文件的问题，用户表达了对改进目录可见性的期望。
   - 有人指出 Aider 从用户的 Git repository 中自动补全文件，这可能会在某些语境下限制可见性。
- **API Key 配置故障**：一位用户遇到了尽管实例正在运行但 API Key 无效的问题，并报告了在不同 Aider 项目中表现不一致。
   - 提到项目继续使用旧的 API 配置运行，突显了 Aider 内部潜在的配置或识别问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: 由 aider 用户制作的入门和教程视频。</li><li><a href="https://aider.chat/docs/more/infinite-output.html">Infinite output</a>: Aider 可以处理支持 prefill 的模型的“无限输出”。</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://ollama.com/library/qwen2.5-coder">qwen2.5-coder</a>: 最新的代码专用 Qwen 模型系列，在代码生成、代码推理和代码修复方面有显著改进。</li><li><a href="https://www.youtube.com/live/vUbPnNeN9eY?si=3hqiicuNpeH6UCgM&t=1537">Aider + 1 Hour = Reels video editor with face recognition</a>: 加入 techfren，看他如何利用软件工程专业知识尝试和评估新技术</li><li><a href="https://docs.litellm.ai/docs/providers/anthropic">Anthropic | liteLLM</a>: LiteLLM 支持所有 anthropic 模型。</li><li><a href="https://github.com/BuilderIO/gpt-crawler">GitHub - BuilderIO/gpt-crawler: Crawl a site to generate knowledge files to create your own custom GPT from a URL</a>: 爬取网站生成知识文件，从 URL 创建你自己的自定义 GPT - BuilderIO/gpt-crawler</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler &amp; Scraper</a>: 🚀🤖 Crawl4AI: 开源 LLM 友好型网络爬虫和抓取工具 - unclecode/crawl4ai
</li>
</ul>

</div>

### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1331002241028591676)** (1 条消息): 

> `Bolt.new 更新，设置问题，Prompt 准确度` 


- **Bolt.new 确保顺畅设置**：*bolt.new* 的最新更新确保用户不再会遇到导致 **白屏** 或从第一个 Prompt 就开始设置失败的问题，因为它现在每次都能更准确地选择并配置正确的模板。
   - 这一增强解决了之前用户的挫败感，并改善了初始设置体验，让所有用户都能有一个 **精准 (spot on)** 的开始。
- **提高 Prompt 配置的准确性**：通过最近的更新，*bolt.new* 在选择模板的准确性方面取得了显著提升，承诺用户从初始 Prompt 开始就能享受无忧的设置。
   - 因此，这减少了设置过程中的困惑，使交互更加顺畅，确保模板配置正确且无误。



**提到的链接**: <a href="https://x.com/boltdotnew/status/1881442318110347291">来自 bolt.new (@boltdotnew) 的推文</a>: Bolt 🧠 更新：bolt․new 现在在选择和配置正确模板方面更加准确 —— 让设置从第一个 Prompt 开始，每一次都精准到位！

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1329903179764072479)** (367 条消息🔥🔥): 

> `Bolt 错误循环，RLS 策略问题，Stripe 集成，支付处理选项，社区支持与资源` 


- **对 Bolt 错误循环的挫败感**：用户对 Bolt 进入持续错误循环表示沮丧，这导致了大量的 Token 消耗却未能解决问题，特别是在处理用户权限等复杂功能时。
   - 一位用户强调了他们因持续问题耗尽了近 3000 万个 Token 的经历，并得出结论必须重新开始以避免遇到的陷阱。
- **行级安全 (RLS) 策略挑战**：多位用户报告在配合 Supabase 使用时遇到重复的 RLS 违规，这使他们难以有效地实现预订功能。
   - 一位用户建议使用外部文档和示例来简化 RLS 策略的创建过程，从而显著减少递归错误。
- **支付集成策略**：关于汽车美容等服务的支付集成展开了讨论，建议倾向于使用 PayPal 按钮等更简单的解决方案，而不是在 Bolt 中进行复杂的设置。
   - 鉴于该用户的非开发背景，WordPress 配合表单构建器插件等替代方案被推荐为更用户友好的选择。
- **关于 Token 使用的预期**：潜在用户询问了 Pro plan 下的 Token 使用情况，了解到 Token 消耗随用户熟练程度而异，并可能取决于是否开启 Bolt 中的 diffs 等功能。
   - 用户得到保证，与免费计划不同，Pro plan 不会对 Token 使用设置每日限制。
- **社区支持与学习**：用户分享了如何更有效地使用 Bolt 的技巧，包括利用 ChatGPT 和 Claude 等资源来协助解决编码问题和查阅文档。
   - 强调了社区支持和知识共享的重要性，用户鼓励通过协作来增强平台上的开发体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boltsync.mystify.tech/">BoltSync - 使用 Bolt 进行 GitHub 仓库管理</a>：通过 Bolt Prompts 修改你的 GitHub 仓库，并使用 BoltSync 将更改同步回 GitHub。通过 AI 驱动的仓库管理简化你的开发工作流。</li><li><a href="https://cardspark.app/)">CardSpark - 数秒内生成完美信息</a>：未找到描述</li><li><a href="https://support.bolt.new/Tokens-13fd971055d6804ea762d2fafdc3ad98">Notion – 集笔记、任务、维基和数据库于一体的一站式工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为你和你的团队打造的一站式工作空间。</li><li><a href="https://x.com/stackblitz/status/1843668731">来自 Charles Beck (@charlesbeck) 的推文</a>：@caughtintheweb 准备给你发那封邮件</li><li><a href="https://gabe.marketing/domains">gabe.marketing - 你的专业 AI 非技术开发达人</a>：营销专家、播客制作人和 AI 驱动的应用开发者，通过创新的数字解决方案助力企业增长。</li><li><a href="https://copycoder.ai/">CopyCoder</a>：未找到描述</li><li><a href="https://boltdiyhosting.com/">Bolt.DIY 托管服务 - 面向开发者的专业云平台</a>：未找到描述</li><li><a href="https://abea.pics/KreLodCmLxEnZ21">Abea</a>：未找到描述</li><li><a href="https://x.com/stackblitz/status/1843668731681267801?s=46">来自 bolt.new (@boltdotnew) 的推文</a>：你现在可以在 bolt.new 中打开公共仓库了 🙌 怎么做？对于任何 GitHub URL，只需在前面加上 "http://bolt.new" 即可！（发布说明见下文！）</li><li><a href="https://prnt.sc/CVZgu1OObu9G">截图</a>：使用 Lightshot 捕获</li><li><a href="https://www.creative-tim.com/learning-lab/bootstrap/grid/soft-ui-design-system">Grid | Soft UI 设计系统 Bootstrap @ Creative Tim</a>：我们的 Bootstrap 网格是一个强大的移动优先 flexbox 网格，凭借十二列系统、五个默认响应层级、Sass 变量和 mixin，帮助你构建各种形状和大小的布局...</li><li><a href="https://resend.com/">Resend · 面向开发者的电子邮件服务 · Resend</a>：大规模构建、测试和发送事务性及营销邮件。</li><li><a href="https://21st.dev/">21st.dev - 面向设计工程师的 NPM</a>：利用受 shadcn/ui 启发的即插即用型 React Tailwind 组件，更快速地交付精美的 UI。由设计工程师构建，专为设计工程师打造。</li><li><a href="https://bolters.io">Bolters.io | Bolt.new 无代码应用生成器的社区支持技巧、窍门和知识库</a>：Bolt.new 的文档和指南</li><li><a href="https://supabase.com/docs/guides/functions/examples/stripe-webhooks">处理 Stripe Webhooks | Supabase 文档</a>：使用 Edge Functions 处理签名的 Stripe Webhooks。</li><li><a href="https://github.com/supabase/supabase/blob/master/examples/edge-functions/supabase/functions/stripe-webhooks/index.ts">supabase/examples/edge-functions/supabase/functions/stripe-webhooks/index.ts (master 分支) · supabase/supabase</a>：开源的 Firebase 替代方案。Supabase 为你提供专用的 Postgres 数据库，用于构建 Web、移动和 AI 应用程序。- supabase/supabase
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1330967333770100746)** (1 条消息): 

> `LM Studio 0.3.7 发布，支持 DeepSeek R1，Mission Control 新功能，KV Cache 量化更新` 


- **LM Studio 0.3.7 发布，带来令人兴奋的新功能**：**LM Studio 0.3.7** 的发布引入了对 **DeepSeek R1** 的支持以及更新的 **llama.cpp engine** 版本 1.9.2，可通过 [应用内更新](https://lmstudio.ai) 获取。
   - 用户还可以下载来自 **DeepSeek** 的各种蒸馏模型，提供高达 **70B** 的尺寸，旨在增强性能。
- **DeepSeek R1：推理模型的游戏规则改变者**：**DeepSeek R1** 模型现已开放下载，承诺提供与 OpenAI 的 **o1** 模型相当的开源推理能力，详情见 [技术报告](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)。
   - 用户会注意到 **DeepSeek R1** 的输出被封装在 `<think>` 标签中，展示了其推理过程。
- **增强的 Mission Control 功能**：Mission Control 中新增了 **Hardware 选项卡**，可以通过 `Cmd/Ctrl + Shift + H` 访问，为用户提供更多监控能力。
   - 此外，**服务器文件日志模式** 允许对日志条目的记录进行更细粒度的控制。
- **KV Cache 量化提升性能**：最新版本为 **llama.cpp** 模型带来了 **KV Cache 量化**，提高了需要 **1.9.0+** 版本的运行时环境的效率。
   - 此功能旨在优化处理模型预测时的性能指标。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLMs</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.7">LM Studio 0.3.7</a>: 支持 DeepSeek R1 且为 llama.cpp 模型提供 KV Cache 量化
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1329937844751564821)** (179 messages🔥🔥): 

> `模型性能对比、LM Studio 中的文件附件、DeepSeek R1 模型讨论、在模型中使用多张图像、LM Studio 更新与功能` 


- **DeepSeek R1 vs Llama 模型**：用户讨论了 DeepSeek R1 模型的能力及其与 Llama 模型的对比，并指出尽管 Qwen 32B 的参数量比 Llama 70B 小，但其排名通常更高。
   - 一些用户强调，虽然 R1 的输出在视觉上显得杂乱，但它能提供不错的答案，尽管其推理过程显得不够自信。
- **LM Studio 中的文件附件功能**：关于 LM Studio 中的文件附件功能出现了一些疑问，特别是上传过程是否会影响本地文件或将数据发送到其他地方。
   - 官方澄清，上传的文件保留在用户的本地机器上，并在与 LLM 交互期间用作上下文。
- **模型响应与推理的问题**：一些用户对 DeepSeek R1 模型响应的随机性和重复性表示担忧，特别是在尝试生成列表或扩展响应时。
   - 用户指出 R1 的记忆力缺乏有效性，导致输出重复，而不是逻辑性地扩展列表。
- **LM Studio 的更新与增强**：讨论涉及 LM Studio 的最新更新，鼓励用户使用新版本的 llama.cpp 引擎以增强模型性能。
   - 用户注意到需要改进思考过程（thinking outputs）的显示方式，以避免在交互过程中界面显得过于杂乱。
- **使用 M2 Ultra 进行分布式推理**：讨论了在联网的 M2 Ultra 机器上使用分布式推理（Distributed Inference），一些用户对其相对于成本的实用性持怀疑态度。
   - Intel 用户确认虽然支持分布式推理，但性能高度依赖于网络带宽和系统配置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/seo_leaders/status/1881462202831614085">Andrew C (@seo_leaders) 的推文</a>：DeepSeek R1 671B 在 2 台 M2 Ultra 上运行，速度快于阅读速度。几乎是家用的、运行在消费级硬件上的开源 O1。使用 mlx.distributed 和 mlx-lm，3-bit 量化（~4 bpw）。模型是 qu...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/11310/commits.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLM</a>：在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档聊天 - 本地运行 LLM | LM Studio 文档</a>：如何为 LLM 提供本地文档作为额外上下文</li><li><a href="https://lmstudio.ai/docs/ba">入门指南 | LM Studio 文档</a>：了解如何使用 LM Studio 在本地运行 Llama, Mistral, Gemma 和其他 LLM。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1329927231090200697)** (186 条消息🔥🔥): 

> `NVIDIA Digits, GPU 对比, 模型性能质量, LM Studio vs Ollama, Kaggle Notebooks` 


- **NVIDIA Digits 作为 AI/ML 服务器**：成员们对将 NVIDIA Digits 作为家用 ML 服务器表现出极大的热情，强调了其执行专用机器学习任务的能力。
   - 虽然它不是典型的游戏 PC，但其对高显存占用的关注与特定的 AI 应用非常契合。
- **AI 任务的 GPU 对比**：讨论对比了 4090/5090 等高端 GPU 与更便宜的替代品在 AI 任务中的表现。
   - 虽然 200 美元的 GPU 足以应付游戏，但参与者指出，专用的 AI 任务将从更强大的显卡中显著受益。
- **模型性能的质量差异**：用户报告了 LM Studio 和 Ollama 之间明显的模型性能差异，特别是在使用 Qwen2.5 模型时。
   - 测试表明，与 Ollama 相比，LM Studio 在特定设置下提供了更高质量的结果。
- **运行非 LLM 的 PyTorch 任务**：参与者讨论了 NVIDIA Digits 是否能处理非 LLM 的 PyTorch 任务，一些人对其性能限制提出了警告。
   - 虽然它可以用于此类任务，但其表现可能不如使用性能更强的 GPU。
- **Kaggle Notebooks 实验**：一位用户担心 NVIDIA Digits 在实验 Kaggle Notebooks 时的响应速度是否足够快。
   - 对话强调了硬件能力与各种机器学习任务需求之间所需的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Graphics_processing_unit#Integrated_graphics">Graphics processing unit - Wikipedia</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889">NVIDIA GeForce RTX 4090 Specs</a>: NVIDIA AD102, 2520 MHz, 16384 Cores, 512 TMUs, 176 ROPs, 24576 MB GDDR6X, 1313 MHz, 384 bit
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1329924464372092950)** (97 条消息🔥🔥): 

> `DeepSeek R1 发布, 转录工具, OpenAI Operator 泄露, Liquid Foundation Model, Claude AI 对齐观点` 


- **DeepSeek R1：游戏规则改变者**：[DeepSeek R1 发布](https://x.com/deepseek_ai/status/1881318138937233664?s=46) 宣布其模型在多个基准测试中达到了与 OpenAI o1 相当的性能，并根据 MIT 许可证开放源代码。
   - 用户对该模型的能力感到兴奋，包括一个在特定任务中表现优于 GPT-4o 等大型模型的 [蒸馏版本](https://x.com/reach_vb/status/1881330709929013440?s=46)。
- **探索转录工具**：成员们讨论了各种转录工具，许多人因性能原因推荐 [MacWhisper](https://github.com/deepseek-ai/DeepSeek-R1)，而其他人则对 Alter 等应用的新功能感兴趣。
   - 社区正在寻找 Wispr Flow 等面临问题的现有工具的替代方案，寻求更好的听写解决方案。
- **OpenAI Operator 泄露**：最近的泄露表明，OpenAI 新的 Computer Use Agent (CUA) 已与 Claude 3.5 等其他模型进行了对比，暗示即将发布。
   - 成员们对这些进展很感兴趣，并密切关注围绕 [Operator 系统](https://x.com/kimmonismus/status/1881287794544550018?s=46) 的更新。
- **Liquid Foundation Model 发布**：Liquid AI 推出了 [LFM-7B 模型](https://www.liquid.ai/lfm-7b)，声称它是同类产品中性能最好的，采用了独特的非 Transformer 架构。
   - 他们强调了其多语言能力和低内存占用，使其适合有部署需求的企业。
- **Claude AI 与对齐讨论**：一篇关于 [Claude AI](https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x) 的帖子引发了围绕 AI Alignment（对齐）及其影响的讨论。
   - 成员们觉得批判这类先进模型的描述方式很有趣，特别是引用了 AI 背景下的 “shoggoth” 等术语。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/sama/status/1880356297985638649">来自 Sam Altman (@sama) 的推文</a>：感谢测试 o3-mini 的外部安全研究人员。我们现在已经敲定了版本并开始发布流程；计划在约两周内推出。此外，我们听取了反馈...</li><li><a href="https://x.com/StringChaos/status/1880317308515897761">来自 Naman Jain (@StringChaos) 的推文</a>：DeepSeek-R1 (Preview) 结果 🔥 我们与 @deepseek_ai 团队合作，在 LiveCodeBench 上评估了 R1 Preview 模型。该模型的表现接近 o1-Medium，提供了 SOTA 的推理性能...</li><li><a href="https://x.com/teortaxesTex/status/1881331287010550119">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：或许最疯狂的事情是，他们说这远未达到 7-70B 级别模型的上限。甚至不需要任何新数据。他们已经进一步推进了这些模型，只是暂时不会分享。</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 DeepSeek-R1 发布了！⚡ 性能与 OpenAI-o1 相当 📖 完全开源的模型和技术报告 🏆 MIT 许可：自由 Distill 和商业化！🌐 网站和 API 已上线！在 h... 尝试 DeepThink</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：带有 "Cold Start" 的 DeepSeek R1 表现基本符合预期。我仍然不买账 R1 Zero 的结果，基础模型在不进行微调的情况下几乎无法输出连贯的解决方案。我打赌这里面有...</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384">来自 Windsurf (@windsurf_ai) 的推文</a>：Wave 2 来了。本次更新包括：🌐 Web Search 🧠 自动生成的 Memories 💼 企业级就绪... 还有更多！</li><li><a href="https://x.com/eliebakouch/status/1881234710841700541?s=46">来自 elie (@eliebakouch) 的推文</a>：天哪，@deepseek_ai R1 上线了，伙计们</li><li><a href="https://x.com/1a3orn/status/1881350809230991382">来自 1a3orn (@1a3orn) 的推文</a>：这是 DeepSeek 论文中我必须读三遍才能确认自己没有幻觉的一句话。R1 蒸馏到 Qwen 1.5b 后，在某些数学基准测试中击败了 Sonnet 和 GPT-4o。</li><li><a href="https://x.com/abacaj/status/1881342078506139881">来自 anton (@abacaj) 的推文</a>：这是 9 月份发布的（在 Codeforces 上）……现在有一个从 R1 蒸馏出来的 32B 模型，得分 1600+，你可以在家运行……哇。引用 DeepSeek (@deepseek_ai) 🔥 奖励：开源蒸馏模型...</li><li><a href="https://x.com/fchollet/status/1880378880894333094?s=46">来自 François Chollet (@fchollet) 的推文</a>：OpenAI 以 *相同的名称* 发布了工作原理完全不同的模型——有些大部分工作由 LLM 完成，另一些大部分工作在于测试时 CoT 搜索。这可能...</li><li><a href="https://x.com/lu_sichu/status/1881348105586855962">来自 Sichu Lu(Sichu.Lu218@proton.me) (@lu_sichu)</a>：也许最有趣的部分是他们解释了为什么 MCTS 在 Token 空间中很难。</li><li><a href="https://x.com/srush_nlp/status/1881382753557754103)">来自 Sasha Rush (@srush_nlp) 的推文</a>：对 DeepSeek-R1 惊人的开源 o1 复现进行的分析。我们曾推测了 4 种难度递增的可能性（G&C, PRM, MCTS, LtS）。答案是最好的那个！它就是 Guess...</li><li><a href="https://x.com/btibor91/status/1881285255266750564?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 网站已经出现了对 Operator/OpenAI CUA (Computer Use Agent) 的引用——“Operator 系统卡片表”、“Operator 研究评估表”和“Operator 拒绝率表”...</li><li><a href="https://x.com/cloneofsimo/status/1881389467346547101">来自 Simo Ryu (@cloneofsimo) 的推文</a>：更惨痛的教训是，某样东西在智力上越有趣，它的用处就越小。你会看到那些极其疯狂的 Post-training 论文，但 REINFORCE 或 PPO 就是更好。你会看到各种古怪的...</li><li><a href="https://x.com/ollama/status/1881427522002506009">来自 ollama (@ollama) 的推文</a>：DeepSeek 的第一代推理模型在数学、代码和推理任务中实现了与 OpenAI o1 相当的性能！快来试试吧！👇 7B 蒸馏版：ollama run deepseek-r1:7...</li><li><a href="https://x.com/nrehiew_/status/1880853579671699709?s=46">来自 wh (@nrehiew_) 的推文</a>：看起来 OpenAI 接触到了 FrontierMath（o3 以巨大优势创下 SOTA 的极具挑战性的数学基准测试）的问题和答案。他们还不允许 FrontierMath 背后的团队披露...</li><li><a href="https://x.com/_xjdr/status/1881365349356147057">来自 xjdr (@_xjdr) 的推文</a>：读完论文后，我仍然没有看到能让 o1 对数图成为现实的 TTC 杠杆。可能有一个我漏掉了（美国西海岸现在还很早），但如果没有的话，我...</li><li><a href="https://x.com/kimmonismus/status/1881287794544550018?s=46">来自 kim 的推文</a>

<li><a href="https://x.com/kimmonismus/status/1881387299520639233">来自 Chubby♨️ (@kimmonismus) 的推文</a>：OpenAI 已经在 OpenAI 的 Operator 和 Claude 3.5 Sonnet CUA (Computer Use Agent) 之间进行了对比。看起来发布在即。引用 Tibor Blaho (@btibor91) 的话，OpenAI 网站已经有 r...</li><li><a href="https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x">Colin (@colin-fraser.net)</a>：这就是为什么我认为 LLM 的“alignment research”是一团糟的原因。Claude 不是一个真人。Claude 是 LLM 被编程去编写的故事中的一个角色。...</li><li><a href="https://x.com/casper_hansen_/status/1881404604518392144">来自 Casper Hansen (@casper_hansen_) 的推文</a>：DeepSeek R1 的训练过程起初让我很困惑。我的大脑拒绝接受这个强大的模型竟然如此简单直接。让我为你拆解这个优雅的猛兽 🧵</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>：CodeGuide 为你的 AI 编程项目创建详细的文档。</li><li><a href="https://x.com/simonw/status/1881361661975843202">来自 Simon Willison (@simonw) 的推文</a>：DeepSeek 今天发布了一整个推理扩展 / “reasoning”模型系列，包括基于 Llama 和 Qwen 的蒸馏变体。这是我关于新模型的笔记，以及我是如何运行 DeepSe...</li><li><a href="https://x.com/mrsiipa/status/1881330071874813963">来自 maharshi (@mrsiipa) 的推文</a>：DeepSeek R1 思考了约 75 秒，成功解决了 OpenAI o1 博客文章中的这个密码文本问题。</li><li><a href="https://x.com/christiancooper/status/1881335734256492605">来自 Christian H. Cooper (@christiancooper) 的推文</a>：我让 #R1 向我视觉化地解释勾股定理。这是在不到 30 秒内一次性完成的，没有任何错误。收工吧，一切都结束了：#DeepSeek #R1</li><li><a href="https://x.com/teortaxesTex/status/1881317131561922640">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：谁去叫醒 Yud</li><li><a href="https://x.com/paul_cal/status/1881324020592963939">来自 Paul Calcraft (@paul_cal) 的推文</a>：上下文内回溯（In-context back-tracking）在 R1 中是涌现出来的。接近“苦涩的教训”（Bitter lesson）。我觉得这很有道理。想知道整个 o1 范式是否最初是对 4o 进行针对推理任务的大规模 RL，而没有特定的...</li><li><a href="https://x.com/teortaxestex/status/1881295618192077099?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：我得出的结论是，R1 以及暗示的 V3-base 在 12 月 10 日之前就已经完成了。看起来 V2.5-1210 是 V3 论文中的这个消融实验（查看 LCB 和 MATH）。Whale 从不单纯地训练模型...</li><li><a href="https://x.com/DimitrisPapail/status/1881341537499619822">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：最有趣的信息点：- 不需要 PRM 或步进式验证器（step-by step-verifier）- 在 {question, answer_i} 对上进行 PPO；使用基于最终答案准确性和格式的优势函数。- RL-tun...</li><li><a href="https://www.liquid.ai/lfm-7b">介绍 LFM-7B：为高效语言模型设定新标准</a>：世界上最顶级的英语、阿拉伯语和日语模型，原生支持法语、德语和西班牙语，经过优化，可作为私有企业聊天、代码、快速指令遵循以及...的基座。</li><li><a href="https://x.com/stochasticchasm/status/1881324856253497515">来自 stochasm (@stochasticchasm) 的推文</a>：另外 @aidan_mclau 关于推理者问题的文章得到了完全的证实，既然我们现在知道 R1 字面上只使用可验证的奖励（verifiable rewards）</li><li><a href="https://skylarbpayne.com/posts/cursed-cursor">如何停止说“去你的 Cursor” - Skylar Payne (Wicked Data LLC)</a>：未找到描述</li><li><a href="https://x.com/samjulien/status/1880405699697762565?s=46">来自 Sam Julien (@samjulien) 的推文</a>：这太酷了。你现在可以直接在 @codeiumdev 的 @windsurf_ai 中与 @Get_Writer 文档聊天了！这里我正在询问关于 Palmyra X 004 中的 tool calling，它正在提取我...</li><li><a href="https://x.com/deepseek_ai/status/1881318138937233664?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：📜 许可证更新！🔄 DeepSeek-R1 现在采用 MIT 许可证，以实现明确的开源访问 🔓 开放给社区利用模型权重和输出 🛠️ API 输出现在可用于微调和蒸馏 🐋 ...</li><li><a href="https://x.com/reach_vb/status/1881315419086291213">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：天哪，这些大佬直接发布了 6 个蒸馏模型，从 1.5B 到 70B 🔥</li><li><a href="https://x.com/qtnx_/status/1881330757001502991">来自 Q (@qtnx_) 的推文</a>：这是 R1 论文中最让我惊讶的地方：>较大模型的推理模式可以被蒸馏到较小的模型中，从而比直接在小模型上进行推理训练获得更好的性能...</li><li><a href="https://x.com/natolambert/status/1881356292943487337">来自 Nathan Lambert (@natolambert) 的推文</a>：R1 让我觉得自己的声音被听到了。稍后会更仔细地阅读。在 conti 中大笑</li>

<li>对 RL 如此运作感到持续震惊。</li><li><a href="https://x.com/nrehiew_/status/1881330794549182853">来自 wh (@nrehiew_) 的推文</a>：这张图就是我们永远无法获得 o1/o3 推理轨迹（reasoning traces）的最大原因。从推理模型中进行蒸馏（distilling）的效果好得不合理。</li><li><a href="https://x.com/lmarena_ai/status/1881411458678014215">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：DeepSeek-R1 现已加入 Arena🔥 祝贺 @deepseek_ai 发布 R1！这是一个在 GPQA/SWE-Bench/AIME 等硬核基准测试中媲美 OpenAI o1 的开源推理模型！现在面临现实世界的挑战——R1 已进入 h...</li><li><a href="https://x.com/kimmonismus/status/1880328198438940923?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>：“OpenAI 的新模型 GPT-4b micro 经过训练，可以建议重新设计蛋白质因子的方法以增强其功能。据 OpenAI 称，研究人员利用该模型的建议来改...”</li><li><a href="https://llmstxt.site/">llms.txt 目录</a>：查找并探索来自各种产品和服务的 llms.txt 文件。</li><li><a href="https://x.com/Grad62304977/status/1881330709929013440">来自 Grad (@Grad62304977) 的推文</a>：没有 MCTS，没有 PRM，涌现行为，简单的 RL。</li><li><a href="https://x.com/sama/status/1880358749187240274">来自 Sam Altman (@sama) 的推文</a>：@kimmonismus @flowersslop 1 和 2，正在解决中，但我认为你会满意的。3，我希望我们能在 2025 年合并 GPT 系列和 o 系列！拭目以待。</li><li><a href="https://x.com/carinalhong/status/1880820323597357273?s=46">来自 Carina Hong (@CarinaLHong) 的推文</a>：1. OpenAI 让 Epoch 签署了 NDA，直到 o3 性能声明前夕，阻止 Epoch 披露 OpenAI 是捐赠者以及 OpenAI 拥有独家数据访问权。2. 数学家随后就问题和解决方案签署了 NDA...</li><li><a href="https://x.com/sama/status/1880360141218017656">来自 Sam Altman (@sama) 的推文</a>：@mckaywrigley 在大多数方面不如 o1 pro（但速度很快）。</li><li><a href="https://x.com/deepseek_ai/status/1881318145761439995?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🌐 API 访问与定价⚙️ 通过设置 model=deepseek-reasoner 使用 DeepSeek-R1 💰 $0.14 / 每百万 input tokens (缓存命中) 💰 $0.55 / 每百万 input tokens (缓存未命中) 💰 $2.19 / 每百万 output tokens 📖 AP...</li><li><a href="https://gist.github.com/morganmcg1/eb0626c7801f3ffbc780ef48269b87ea">AI/Cursor 任务设计文档，源自此 Cursor 博客：https://skylarbpayne.com/posts/cursed-cursor#design-document</a>：AI/Cursor 任务设计文档，源自此 Cursor 博客：https://skylarbpayne.com/posts/cursed-cursor#design-document - ai_design_template.md</li><li><a href="https://x.com/reach_vb/status/1881319500089634954">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：“DeepSeek-R1-Distill-Qwen-1.5B 在数学基准测试中超越了 GPT-4o 和 Claude-3.5-Sonnet，AIME 达到 28.9%，MATH 达到 83.9%。” 1.5B 做了什么？！</li><li><a href="https://x.com/sama/status/1880366903107154134">来自 Sam Altman (@sama) 的推文</a>：@TarasBob @mckaywrigley o3 聪明得多；我们现在正将注意力转向它。（还有 o3 pro？！ 🤯）</li><li><a href="https://techcrunch.com/2025/01/19/the-pentagon-says-ai-is-speeding-up-its-kill-chain">五角大楼称 AI 正在加速其“杀伤链” | TechCrunch</a>：领先的 AI 开发商（如 OpenAI 和 Anthropic）正在小心翼翼地向美国军方推销软件：让五角大楼...</li><li><a href="https://x.com/sama/status/1880388642172203226">来自 Sam Altman (@sama) 的推文</a>：@sporadicalia @TarasBob @mckaywrigley 不，你花 200 就能得到它。</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>：通过在 GitHub 上创建账户来为 MoonshotAI/Kimi-k1.5 的开发做出贡献。</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">来自 Kimi.ai (@Kimi_ai_) 的推文</a>：🚀 隆重推出 Kimi k1.5 —— 一个 o1 级多模态模型。具有 SOTA 级别的短 CoT 性能，在 📐AIME、📐MATH-500、💻 LiveCodeBench 上大幅领先 GPT-4o 和 Claude Sonnet 3.5（最高达 +550%...</li><li><a href="https://list.alterhq.com/p/just-say-it-transform-any-text-with-voice-commands">🗣️ 直接说出来 — 通过语音命令转换任何文本</a>：在所有应用中通过语音命令转换任何文本。选择、说话、完成。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账户来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1329951534003261480)** (4 条消息): 

> `O1 podcast discussion, DeepSeek v3, SGLang framework, Mission Critical Inference, Kubernetes challenges` 


- **O1 播客后续**：@latentspacepod 发布了关于 [O1 skill issue](https://youtu.be/NkHcSpOOC60) 的后续播客，其中包含了 Ben 的见解。他形容正确使用时的 **O1** 令人“叹为观止”。
   - Ben 强调，**O1** 应该被视为一个“报告生成器”而非聊天模型（chat model），并突出了其独特的功能。
- **DeepSeek v3 的精彩特性**：最新的播客探讨了 **DeepSeek v3** 以及即将发布的 **SGLang**，讨论了该领域的重要规格和成就。
   - 听众可以深入了解包括 [模型性能](https://www.latent.space/p/baseten) 以及 **Mission Critical Inference** 的关键方面。
- **深入探讨 Mission Critical Inference**：特邀嘉宾讨论了“**Mission Critical Inference 的三大支柱**”，详细介绍了与 **DeepSeek** 相关的技术见解和优化。
   - 本期节目涵盖了在突破单 **GPU** 限制、扩展工作负载（workloads）方面的关键策略，同时解决了 **Kubernetes** 等基础设施挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1880394100857532521">来自 Latent.Space (@latentspacepod) 的推文</a>: 在 o1 skill issue 帖子取得成功后，我们与 Ben 和 @daniel_mac8 录制了一段简短的播客。现在已上线！（下方是为算法大神准备的链接）引用 ben (@benhylak) 的话：当你了解如何使用时，o1 是令人惊叹的...</li><li><a href="https://x.com/latentspacepod/status/1880829933259559002?s=46">来自 Latent.Space (@latentspacepod) 的推文</a>: 🆕: 运行 Mission Critical Inference 所需的一切（包含 DeepSeek v3 + SGLang）https://www.latent.space/p/baseten 我们与 @amiruci 和 @yinengzhang 聊了聊 2024 年的“中国大模型（Chinese Whale Bro）”发布：- ...</li><li><a href="https://youtu.be/NkHcSpOOC60"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1329918103127064634)** (220 条消息🔥🔥): 

> `用于无障碍辅助的 AI 工具，MCP server 框架，用于 STT 的 Whisper，YouTube 字幕，Windows 11 中的实时字幕` 


- **AI 工具增强无障碍辅助**：参与者讨论了用于无障碍辅助的 AI 工具的进展，重点介绍了 Windows 11 和 YouTube 等平台在实时字幕方面的进步。
   - 他们指出自动字幕已经有所改进，使技术演讲更易于理解，尽管有些人为了准确性仍然更倾向于人工字幕。
- **MCP 框架分享**：大家对即将举行的关于分享新 MCP server 框架经验的会议充满热情，成员们表示有兴趣安排未来的演示。
   - 参与者讨论了使用电子表格等协作工具来组织主题并促进知识共享的潜在好处。
- **Whisper 用于语音转文本处理**：Whisper 在非实时语音转文本应用中的有效性受到了赞扬，尽管一些参与者表示有兴趣探索 Whisper 在会议中的实时应用。
   - 讨论强调了性能因设备规格而异，以及可能需要利用 GPU。
- **语音转文本技术见解**：对话包括对各种语音转文本技术的见解，详细介绍了在 Drafts 和 Whisper Memos 等不同平台上的个人经验和偏好。
   - 参与者分享了关于克服自动转录挑战的想法，特别是与非标准口音相关的挑战。
- **实时字幕的重要性**：实时字幕功能是一个重要焦点，参与者注意到 Windows 11 实时字幕等工具在改善无障碍辅助方面的改进。
   - 讨论强调了整合各种技术以增强听力障碍人士沟通体验的持续挑战和益处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.samjulien.com/">Sam Julien: Director of DevRel and Your Biggest Fan</a>：Sam Julien 是一位 Developer Relations 总监、作家和教师。他热衷于帮助人们利用 Python 和 JavaScript 提升其 developer advocacy、AI engineering 或 Web 开发工作的水平。</li><li><a href="https://brilliant-tapioca-5ad42f.netlify.app/">⚡️ Bolt.new + Vite + React</a>：未找到描述</li><li><a href="https://www.marblism.com/">Marblism - AI Agents that work for you</a>：构建和部署 AI Agents 的领先平台。通过强大的 AI agents 实现各行各业工作流的自动化。</li><li><a href="https://getdrafts.com/">Drafts, Where Text Starts</a>：适用于 iPhone、iPad、Mac 和 Apple Watch 的 Drafts 快速捕捉应用</li><li><a href="https://github.com/samjulien/discord-scraper">GitHub - samjulien/discord-scraper</a>：通过在 GitHub 上创建账户，为 samjulien/discord-scraper 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>：未找到描述</li><li><a href="https://superwhisper.com/">superwhisper</a>：适用于 macOS 的 AI 驱动语音转文本工具</li><li><a href="https://whispermemos.com/">Whisper Memos</a>：Whisper Memos 可以转录您的 iOS 语音备忘录，并在几分钟后向您发送包含转录内容的电子邮件。它基于 OpenAI 的新 Whisper 技术。
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1330916117916553257)** (4 条消息): 

> `DeepSeek R1 发布，与 OpenAI 的性能对比，无审查访问，Llama Endpoints 关停` 


- **DeepSeek R1 在 OpenRouter 上线**：**DeepSeek R1** 模型现已在 [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1) 上线，其性能号称可与 **OpenAI 的 o1** 模型相媲美。
   - 凭借**透明的思维 token (thinking tokens)**，其价格为每百万输入 token **$0.55**，仅为 OpenAI 同类模型成本的 **4%**。
- **无审查的 DeepSeek R1**：正如社区讨论所指出的，用户可以在 [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737) 上访问无审查的 **DeepSeek R1**。
   - 尽管这是一个受审查的模型，但用户认为专家的微调可以增强其性能。
- **免费 Llama Endpoints 停止服务**：一份通知显示，由于供应商 **Samba Nova** 的变动，**免费 Llama endpoints** 将于本月底关停。
   - Samba Nova 将过渡到 **Standard 变体**，届时价格将会上涨。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/xanderatallah/status/1881456463786512737">来自 Alex Atallah (@xanderatallah) 的推文</a>: 请注意，你可以在 @OpenRouterAI 上使用无审查的 DeepSeek R1：引用 MatthewBerman (@MatthewBerman) DeepSeek R1 正在做 @shaunralston 预期的事情。归根结底，它仍然是一个受审...</li><li><a href="https://x.com/openrouterai/status/1881407719170797741?s=46">来自 OpenRouter (@OpenRouterAI) 的推文</a>: DeepSeek R1 现已在 OpenRouter 上线！⚡ 性能与 OpenAI o1 持平🧠 透明的思维 token🍕每百万输入 token $0.55，由 @deepseek_ai 托管。这仅是 o1 价格的 4% 引用 Risphere (@ris...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek-R1 来了！⚡ 性能与 OpenAI-o1 持平📖 完全开源的模型和技术报告🏆 MIT 许可证：自由地进行蒸馏和商业化！。通过 API 运行 DeepSeek R1
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1329904965262774336)** (258 条消息🔥🔥): 

> `DeepSeek R1 发布, OpenAI 模型速率限制 (Rate Limits), DeepSeek 用户体验, OpenRouter 中的 Web Search API, 推理内容 (Reasoning Content) 获取` 


- **DeepSeek R1 已上线！**：DeepSeek 宣布发布 R1，据报道其性能与 OpenAI 的模型相当，且完全开源，遵循 MIT 许可证。
   - 用户对其能力表示兴奋，尤其是在视频内容生成和微积分等创意任务方面。
- **OpenAI 模型速率限制 (Rate Limits) 说明**：用户寻求关于通过 OpenRouter 使用 Gemini 2.0 速率限制的澄清，确认付费模型没有限制，而免费模型每天限制为 200 次请求。
   - 提到用户可以通过连接自己的 API keys 来添加自己的速率限制设置。
- **DeepSeek 用户反馈**：多位用户分享了 DeepSeek R1 的初步体验，称其为各种应用的强大工具，尽管有些用户对 API 限制表示沮丧。
   - 讨论了可能进行的调整，以改进从 API 获取推理内容 (Reasoning Content) 的方式。
- **Web Search API 可用性**：关于 Web Search API 可用性的咨询，确认目前仅可通过聊天室界面访问。
   - 用户对扩展其集成能力的 Beta 选项表示感兴趣。
- **使用 DeepSeek 获取推理内容**：提出了关于从 DeepSeek API 获取 `reasoning_content` 的问题，回复指出 OpenRouter 需要实现对其的支持。
   - 社区热切期待该功能的更新，因为它能增强模型的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ExaAILabs">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://openrouter.ai/docs/errors).">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://x.com/xiaoqianWX/status/1881293445098283083">来自 xiaoqianWX (@xiaoqianWX) 的推文</a>：DeepSeek R1 的 API 刚刚上线（模型名称：deepseek-reasoner）。定价似乎是 15CNY(2USD)/Mtok 输出。目前还没能对其进行任何基准测试</li><li><a href="https://openrouter.ai/docs/models">Models | OpenRouter</a>：所有可用模型的表格</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/settin">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 DeepSeek-R1 来了！⚡ 性能与 OpenAI-o1 相当📖 完全开源的模型和技术报告🏆 MIT 许可：自由蒸馏和商业化！🌐 网站和 API 现已上线！在 h... 尝试 DeepThink</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/teortaxesTex/status/1880768996225769738">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：R1 的 pass@10 远好于 o1-High 计算；在困难集上比 pass@1 提升了 20%。大模型往往存在模式崩溃，所以考虑到它们如此便宜，pass@n 才有意义。这支持了我的猜测，即推理...</li><li><a href="https://tinypic.host/image/Screenshot-2025-01-18-192202.2FIpeB">Screenshot 2025 01 18 192202</a>：托管在 Tinypic 的图片截图 2025 01 18 192202
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1329958477715472424)** (249 条消息🔥🔥): 

> `Using Stable Diffusion for Photorealism, E-commerce Text-To-Image Models, Artistic Style Consistency in LoRA Training, Image Generation Issues and Solutions, AI Tools for Background Editing` 


- **提升写实图像生成质量**：用户讨论了使用 Stable Diffusion 3.5 创建写实图像的技术，建议使用 LoRA 来获得理想的外观。
   - 一位用户提到了画面呈现“塑料感”的挑战，并寻求获得更真实输出的技巧。
- **电子商务与 Google Cloud 部署**：一位用户考虑在 Google Cloud 上部署文本生成图像模型，并就使用 GitHub 模型还是 Google Cloud Marketplace 寻求建议。
   - 共识是使用预训练模型可以节省时间，但用户对最高效的部署方法仍不确定。
- **LoRA 训练中艺术风格的挑战**：讨论集中在 LoRA 模型中训练分辨率多样性的影响，以及仅在 1024x1024 分辨率下训练是否足够。
   - 有建议认为，使用多种分辨率可以增强模型在不同图像质量下的泛化能力。
- **图像生成问题排查**：多位用户报告了图像生成方面的问题，包括处理速度变慢和输出质量差异。
   - 一些用户建议使用不同的 denoising steps 并验证配置，以获得持续更好的结果。
- **编辑图像背景**：用户分享了使用 GIMP 等工具和 AI 解决方案移除或模糊照片背景的经验。
   - 强调了手动编辑通常会产生更好的效果，特别是对于 AI 可能处理不好的特定图像细节。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stablediffusionweb.com/image/25622118-robot-woman-with-removed-face-plate">Robot Woman with Removed Face Plate</a>: 未找到描述</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Why%20Use%20Swarm.md">SwarmUI/docs/Why Use Swarm.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (原名 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的高级工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API">API</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md">SwarmUI/docs/Features/Prompt Syntax.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (原名 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的高级工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (原名 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的高级工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html">Link Shell Extension</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1329938235563966647)** (27 条消息🔥): 

> `播客制作与语音集成、Gemini Advanced Deep Research 工作流、在大学课程中使用 NotebookLM、溯源工具使用体验、社区自我介绍` 


- **播客制作与语音集成**：一位成员分享了关于 **GLP-1s** 的新播客，并询问是否可以更改播客主持人的声音，建议与 [Eleven Labs](https://illuminate.google.com/home?pli=1) 进行集成。
   - 然而，另一位成员指出，目前的播客工具可能还不支持此类更改。
- **探索 Gemini Advanced 工作流**：一位用户讨论了利用 **Gemini Advanced Deep Research** 生成报告和音频概览的潜在工作流，尽管提到了访问权限的限制。
   - 另一位用户确认了一个类似的成功流程，并建议直接溯源以避免信息丢失。
- **NotebookLM 在大学中的最佳实践**：一位用户就 **经济学课程 (econ course)** 的笔记本组织方式寻求建议，讨论是将多个来源上传到一个笔记本中，还是保持它们相互独立。
   - 一位资深用户建议采用基于主题的组织方式，以简化工作流并保持不同来源之间的一致性。
- **社区资源与工具**：一位成员分享了 **WebSync Chrome extension** 的链接，该插件旨在将页面和网站导入 **NotebookLM**，从而提高研究效率。
   - 此外，还分享了一个视频链接，展示了 **NotebookLM** 等工具及其对生产力的提升。
- **社区介绍与互动**：新成员进行了自我介绍，强调了语言差异，并表达了加入社区的兴奋之情。
   - 一位用户鼓励在特定频道提出引人入胜的问题，以促进更集中的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://chromewebstore.google.com/detail/websync-full-site-importe/hjoonjdnhagnpfgifhjolheimamcafok">WebSync full site importer for NotebookLM - Chrome Web Store</a>: 一个将页面或整个网站添加到 NotebookLM 的扩展程序</li><li><a href="https://illuminate.google.com/home?pli=1">Illuminate | Learn Your Way</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是您的 Gen AI 工具，可帮助您更快地理解复杂内容。</li><li><a href="https://youtu.be/t1OjAauA6uY?si=yo6frSZedvaGydt-">Top Relationship Experts Reveal Why Couples Are Ditching Marriage</a>: 对于现代情侣来说，婚姻真的值得吗？为什么情侣们选择退出？本播客探讨了情侣选择不结婚的日益增长的趋势，解释了...</li><li><a href="https://youtu.be/mReOoe8Ou3A">4 AI Tools to 25× your Productivity: NotebookLM, Perplexity AI, Gemini Deep Research &amp; Gamma AI</a>: 全面的 NotebookLM 播放列表 - https://www.youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8🚀 改变您的 NotebookLM 研究和知识...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1329919833034195116)** (212 条消息🔥🔥): 

> `Google One AI Premium, NotebookLM Plus, Podcast Generation Issues, Document Uploading Issues, Language Support in Interactive Podcast` 


- **NotebookLM Plus 的订阅选项**：用户讨论了 Google One AI Premium 与 Google Workspace Business Standard 在访问 Google Gemini 模型和 NotebookLM Plus 功能方面的差异。
   - 建议认为，虽然两种选项都提供访问权限，但 Google One 管理起来更简单，没有 Workspace 那么复杂。
- **Podcast 生成方面的问题**：用户提出了 NotebookLM 生成的 Podcast 长度不一的问题，用户尝试自定义时长，但收到的音频概览往往超过了请求的时长。
   - 几位用户注意到 Podcast 过程中语音角色随机切换的挑战，导致了混淆。
- **上传大音频文件的问题**：一位用户报告在上传接近或超过 100MB 的音频文件时遇到问题，怀疑是因为现有文件超出了 200MB 的总上传限制。
   - 强调了在上传新文件前监控总文件大小的重要性，以防止出现此问题。
- **文档上传和 OCR 限制**：讨论了上传需要 OCR 才能生成简报的不可复制 PDF 文档时遇到的困难，一位用户表示他们无法从这类文件中生成简报文档。
   - 增强 NotebookLM 对 OCR 功能的支持被强调为一个潜在的改进方向。
- **Podcast 的多语言支持**：用户希望 NotebookLM 的交互式 Podcast 功能能包含英语以外的语言，并期待其尽快上线。
   - 一些用户目前正在利用变通方法生成不同语言的内容，等待官方支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/google-workspace-individual/answer/10758004?hl=en">Learn about Google Workspace Individual - Google Workspace Individual Help</a>: 未找到描述</li><li><a href="https://policies.google.com/terms">Google Terms of Service – Privacy &amp; Terms – Google</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963">Learn how NotebookLM protects your data - NotebookLM Help</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1329916079568584775)** (193 条消息🔥🔥): 

> `MCP 服务器反馈，Roo Cline 功能，Claude 的速率限制，聊天日志摘要，MCP 客户端中的用户界面问题` 


- **关于 MCP 服务器实现的反馈**：用户对各种 MCP 服务器中 prompt 使用不一致表示困惑，有些服务器仅提供资源获取而缺乏有意义的交互。
   - 有人担心服务器实现偏离了官方文档，导致 prompt 使用效果不佳。
- **Roo Cline 的优势与功能**：Roo Cline 因其与 R1 配合的易用性而受到称赞，支持配置其自身的 MCP 服务器，并通过命令自动批准提供“Agentic”体验。
   - 用户强调，与 Claude Desktop 和 LibreChat 等其他客户端相比，Roo Cline 与 VSCode 的集成使其成为一个极具吸引力的选择。
- **管理 Claude 中的速率限制**：用户报告在与 Claude 交互时经常遇到速率限制（Rate limits），这会限制上下文长度和消息频率。
   - 讨论中提到希望有工具能监控 Claude Desktop 发送的消息，以便更好地理解速率限制问题。
- **探索用于 CSV 修改的 MCP 服务器**：用户对是否存在能根据 prompt 修改 CSV 行的 MCP 服务器表现出兴趣，但在现有的 MCP 服务器中尚未发现明确的解决方案。
   - 提到一个相关的 Google Sheets 服务器，表明存在一些用于文档管理的工具，但并非专门针对 CSV 处理。
- **运行 MCP 项目的成本估算**：用户讨论了在日常任务中使用各种 AI 模型的运营成本，并指出成本因用户需求和使用频率而异。
   - 分享的经验表明，通过选择合适的模型和使用策略，运行个人数字助理的成本可能会更低。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp/servers/54hsrjhmq9">mcp-ragdocs</a>: 一个 MCP 服务器实现，提供通过向量搜索检索和处理文档的工具，使 AI 助手能够利用相关的文档上下文增强其回答...</li><li><a href="https://support.anthropic.com/en/articles/9450526-how-can-i-export-my-claude-ai-data">如何导出我的 Claude.ai 数据？ | Anthropic 帮助中心</a>: 未找到描述</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: 用于从 Google Drive 读取和编辑 Google Sheets 的 Model Context Protocol (MCP) 服务器</a>: 用于从 Google Drive 读取和编辑 Google Sheets 的 Model Context Protocol (MCP) 服务器 - isaacphi/mcp-gdrive</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol 服务器</a>: Model Context Protocol 服务器。在 GitHub 上为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://www.pulsemcp.com/clients">39 个 MCP 客户端：兼容 MCP 服务器的 AI 驱动应用 | PulseMCP</a>: 一系列能够作为 Model Context Protocol (MCP) 客户端与不断增长的 MCP 服务器列表进行交互的 AI 应用和工具。</li><li><a href="https://github.com/ggozad/oterm">GitHub - ggozad/oterm: 一个用于 Ollama 的基于文本的终端客户端</a>: 一个用于 Ollama 的基于文本的终端客户端。在 GitHub 上为 ggozad/oterm 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/142">[提案] 为上下文丰富增加 Augmentation 能力 by PederHP · Pull Request #142 · modelcontextprotocol/specification</a>: 动机与背景：此 PR 为 MCP 引入了 Augmentation 能力，解决了 AI 应用中对上下文丰富的需求，这在 RAG 中很常见。虽然现有的能力...</li><li><a href="https://glama.ai/mcp">开源 MCP 服务器</a>: 企业级安全、隐私，具有 Agent、MCP、prompt 模板等功能。</li><li><a href="https://glama.ai/mcp/servers?attributes=`)">开源 MCP 服务器</a>: 企业级安全、隐私，具有 Agent、MCP、prompt 模板等功能。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1330065200283648115)** (30 条消息🔥): 

> `Figma MCP 贡献, MCP Logic Calculator, LibreChat 性能, TestFlight 反馈, Anthropic 模型兼容性` 


- **Figma MCP 的贡献机会**：[Figma MCP](https://github.com/nmfisher/figma_mcp) 处于早期阶段，欢迎贡献者参与开发工作。
   - 一位成员对该项目表示兴奋：*“这还处于非常早期/粗糙的阶段，非常欢迎任何贡献者！”*
- **AI Logic Calculator 受到关注**：由另一位成员开发的 [MCP Logic Calculator](https://github.com/angrysky56/mcp-logic) 旨在通过 Python 利用 Prover9/Mace4，为 Windows 用户提供功能。
   - 另一位成员指出，将分类器与 memory MCP 集成以增强领域感知能力的潜力。
- **LibreChat 的评价褒贬不一**：成员们报告了在 LibreChat 中使用 **Llama** 和 **DeepSeek** 等各种 LLM 的情况，并指出与 **Claude** 相比存在性能问题。
   - 一位成员对配置问题表示担忧，直言：*“LibreChat 很烂；我遇到了太多的配置问题。”*
- **通过 TestFlight 测试 iOS 应用**：成员们讨论了即将通过 TestFlight 推出的 **Sage for Claude iOS**，强调了其功能和测试流程。
   - 反馈不一，有人指出 iOS 版本运行良好，而 macOS 版本在启动时出现崩溃问题。
- **探索与其他模型的兼容性**：讨论包括 Model Context Protocol (MCP) 是否能与 Sonnet 之外的其他模型（特别是 **Anthropic** 模型）协同工作。
   - 一位成员询问了集成 r1 的可行性，暗示了对更广泛模型兼容性的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://testflight.apple.com/join/EJIXPsr1">加入 Sage 测试版</a>：支持 iOS</li><li><a href="https://glama.ai/mcp/clients/libre-chat">LibreChat</a>：增强版 ChatGPT，具备 Agent、AI 模型切换、Code Interpreter、DALL-E 3、OpenAPI Actions、安全多用户认证等功能。支持 OpenAI、Anthropic、Azure，并通过开源支持自托管。</li><li><a href="https://github.com/nmfisher/figma_mcp">GitHub - nmfisher/figma_mcp</a>：通过在 GitHub 上创建账号来为 nmfisher/figma_mcp 的开发做出贡献。</li><li><a href="https://github.com/angrysky56/mcp-logic">GitHub - angrysky56/mcp-logic: 功能齐全的 AI Logic Calculator，通过基于 Python 的 Model Context Protocol (MCP-Server) 利用 Prover9/Mace4 —— 适用于 Windows Claude App 等的工具</a>：功能齐全的 AI Logic Calculator，通过基于 Python 的 Model Context Protocol (MCP-Server) 利用 Prover9/Mace4 —— 适用于 Windows Claude App 等的工具 - angrysky56/mcp-logic
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1329919179465298022)** (167 条消息🔥🔥): 

> `GPU vs CPU 性能, Agent 学习模型, 自适应 LLM, AI 工具评估, 在线社区动态`

- **GPU 与 CPU 在数组处理中的效率对比**：在关于使用 GPU 还是 CPU 查找数组最大值的讨论中，指出 **GPU** 在处理大型数组时速度更快，特别是在并行处理方面，但存在数据传输瓶颈。
   - 一位成员提到，在数组中查找最大值是一个琐碎的并行操作（trivially parallel operation），这表明对于大型数据集，两种架构的性能可能相似。
- **Agent 学习模型的探索**：讨论了使用 LLM 构建 Agent 的问题，承认由于其在“Agent 式（agentive）”任务中的局限性，在使其自主行动方面面临挑战。
   - 大家一致认为，尽管 AI 取得了进步，但要让 Agent 在基础命令执行之外进行有意义的操作，仍需要突破性的方法。
- **AI 编程工具评估**：参与者评估了包括 OpenAI ChatGPT 和 Claude 在内的各种代码生成工具，并对那些能为特定编程任务提供足够质量的工具表示了偏好。
   - OpenAI ChatGPT 被强调为优于其他工具，同时也评论了自 GitHub CoPilot 兴起以来 AI 工具和编程的新趋势。
- **自适应大语言模型**：一篇关于自适应 LLM 的论文，题为 **Transformer²**，介绍了一种能够实现实时任务自适应的机制，其表现优于传统的 fine-tuning 方法。
   - 该论文讨论了使用强化学习动态混合任务特定向量，表明这些进展可能会使传统的 fine-tuning 方法过时。
- **社区对 AI 趋势的见解**：社区分享了对 AI 炒作的观察，强调了对目前市场上解决方案的实际能力与公众预期之间差距的怀疑。
   - 有人指出，商业利益往往驱动着 AI 发布趋势，这使得基础性突破对于该技术交付有意义的结果至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://fxtwitter.com/sama/status/1881258443669172470">来自 Sam Altman (@sama) 的推文</a>：Twitter 上的炒作再次失控。我们下个月不会部署 AGI，也没有开发出它。我们有一些非常酷的东西要展示，但请冷静下来，将你的期望降低 100 倍！</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">来自 mgostIH (@mgostIH) 的推文</a>：深度学习到底怎么了？？？</li><li><a href="https://arxiv.org/abs/2501.06252">$\text{Transformer}^2$: 自适应 LLMs</a>：自适应大语言模型 (LLMs) 旨在解决传统微调方法带来的挑战，这些方法通常计算密集，且在处理多样化任务时能力较为静态...</li><li><a href="https://tenor.com/view/adorable-wink-bat-oh-hey-gif-14058611">可爱的眨眼 GIF - 可爱的眨眼蝙蝠 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/engel-t-pose-engel-gif-16943840286212504665">Engel T-pose Engel GIF - Engel T-pose engel - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/gop-corruption-thoughts-and-prayers-mass-shooting-human-rights-gif-25725581">Gop 腐败 GIF - Gop 腐败 思想与祈祷 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/goku-ssj4-super-saiyan-4-dragon-ball-gt-dragon-ball-gif-6491195933392867517">悟空 Ssj4 GIF - 悟空 Ssj4 超级赛亚人 4 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/crosswind-landing-gif-20167802">侧风着陆 GIF - 侧风着陆 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/xzibit-meme-inception-gif-13033570">Xzibit 梗 GIF - Xzibit 梗 盗梦空间 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/soviet-cat-sovicat-soviet-ussr-cat-gif-21826197">苏联猫 Sovicat GIF - 苏联猫 Sovicat 苏联 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/klasik-clasik-clasic-gif-25398314">Klasik Clasik GIF - Klasik Clasik Clasic - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/marachime-highrollers-dnd-dungeons-and-dragons-mark-hulmes-gif-14728949">Marachime Highrollers GIF - Marachime Highrollers Dnd - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/russian-roulette-gun-gif-24197229">俄罗斯轮盘 GIF - 俄罗斯轮盘 枪 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/DrJimFan/status/1881353126210687089">来自 Jim Fan (@DrJimFan) 的推文</a>：我们生活在这样一个时间线上：一家非美国公司正在延续 OpenAI 的原始使命——真正开放、赋能所有人的前沿研究。这简直不可思议。最有趣的结果是...</li><li><a href="https://tenor.com/view/the-gun-gun-gif-25386021">枪 GIF - 枪 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/smile-gif-3415810009431604905">微笑 GIF - 微笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/im-bayblade-beyblade-spin-silly-gif-16417105">我是战斗陀螺 GIF - 我是战斗陀螺 旋转 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fixupx.com/SavannahFeder/status/1877444748039819301">来自 Savannah (@SavannahFeder) 的推文</a>：发布 Astral —— 一个 24/7 全天候工作以助力初创公司成长的 AI 营销人员。Astral 可以浏览网站、创作内容并在社交媒体上进行营销。观看 Astral 实时自动化运营 Reddit：</li><li><a href="https://www.youtube.com/watch?v=gfr4BP4V1R8">AI 讨论一份只写着 “Poopoo Peepee” 的文档</a>：文档：Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo P...</li><li><a href="https://www.youtube.com/watch?v=BEv_qZwR3h8">Dark Dreams - 阿尔托莉雅·潘德拉贡 (Saber Alter) [AMV]</a>：阿尔托莉雅·潘德拉贡 (Saber Alter) - Mr. Sandman - Fate/Stay Night: Heaven's Feel - AMV-------------------------------------------------------------------------...</li><li><a href="https://www.youtube.com/watch?v=JMC56argXVk)">如何在“左转信号灯”处左转 | 聪明驾驶</a>：立即订阅！► http://youtube.com/c/smartdrivetest 警惕“左转信号灯”——在路考中它可能是披着羊皮的狼...</li><li><a href="https://youtu.be/w6zi95SknZw">欢迎来到宇宙学及其基础观测</a>：本视频结合了我新宇宙学系列视频的第 1 章和第 2 章。我正在讲解 Barbara Ryden 博士的教科书《宇宙学导论》...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf</a></li>

ek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>：通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://youtu.be/w8HdOHrc3OQ">[最佳版本] 《大独裁者》演讲 - 查理·卓别林 + Time - 汉斯·季默（《盗梦空间》主题曲）</a>：- 请阅读 - 查理·卓别林在《大独裁者》中的演讲，配上汉斯·季默为电影《盗梦空间》创作的《Time》 = 史诗级！！！重要提示：这...</li><li><a href="https://tenor.com/view/die-kill-internet-modem-eileen-gif-20652466">Die Kill GIF - Die Kill Internet - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1329948738780659723)** (37 条消息🔥): 

> `Lightning Attention 论文讨论, rStar-Math 研究发现, Tensor Product Attention (TPA) 机制, 线性 Tensor Product Lightning Attention, DeepSeek 的 Group Relative Policy Optimization` 


- **Lightning Attention 论文因创新性不足被拒**：在关于 Lightning Attention 论文的讨论中，一些成员对该论文被 ICLR 拒绝表示了共鸣，原因是评审认为其相对于 [NormAttention](https://arxiv.org/pdf/2210.10340) 和 FlashAttention 等先前工作的改进属于增量式更新。
   - 评审员批评了其创新性，导致一些人怀疑在训练和推理过程中使用自适应矩阵乘积是否已经是一项众所周知的技术。
- **关于 rStar-Math 独特方法论的见解**：rStar-Math 论文展示了小型语言模型（SLM）如何在不依赖蒸馏的情况下，通过利用蒙特卡洛树搜索（MCTS）进行深度思考，从而在数学推理能力上媲美或超越 OpenAI 的水平。
   - 值得注意的是，该方法被认为在模拟环境中非常实用，提供了三种创新的训练技术，避免了对人类数据的依赖。
- **结合 Lightning Attention 实现 Tensor Product Attention**：一项实验展示了利用 Lightning Attention 的线性化技术成功集成了 Tensor Product Attention，在玩具模型中实现了显著的速度提升。
   - 该实现显示了约 **3 倍的速度** 提升，能够有效处理注意力机制中的大型张量操作。
- **DeepSeek 的 Group Relative Policy Optimization (GRPO) 解析**：讨论强调 DeepSeek 的 GRPO 功能与 PPO 类似，但去掉了价值函数（value function），转而依赖于优势（advantage）的蒙特卡洛估计。
   - 理解 GRPO 需要掌握价值函数在应用于语言模型时面临的挑战，这暗示了需要具备 PPO 的基础知识。
- **社区参与和资源共享**：成员们积极分享了相关研究论文、GitHub 仓库和资源的链接，例如 [DeepSeek R1 PDF](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)。
   - 这些贡献引发了关于不同注意力范式下模型效率和性能的有意义讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>：我们提出 rStar-Math，旨在证明小型语言模型（SLM）在不经过优越模型蒸馏的情况下，可以媲美甚至超越 OpenAI o1 的数学推理能力。rStar-Math 实现了...</li><li><a href="https://fixupx.com/natolambert/status/1881380809153847711">Nathan Lambert (@natolambert) 的推文</a>：对于那些试图理解 DeepSeek 的 Group Relative Policy Optimization (GRPO) 的人：GRPO 就是没有价值函数、使用优势的蒙特卡洛估计的 PPO。所以，去研究为什么 PPO 存在（lo...</li><li><a href="https://arxiv.org/abs/2501.06252">$\text{Transformer}^2$: Self-adaptive LLMs</a>：自适应大语言模型（LLM）旨在解决传统微调方法带来的挑战，这些方法通常计算密集，且在处理多样化任务时能力较为静态...</li><li><a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>：扩展语言模型以处理更长的输入序列通常需要大型键值（KV）缓存，从而导致推理过程中巨大的内存开销。在本文中，我们提出了 Tensor...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf (GitHub)</a>：通过在 GitHub 上创建一个账户来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/CoffeeVampir3/Micro-Mjolinear/blob/master/models/lightning_tensor_product.py#L143>">Micro-Mjolinear/models/lightning_tensor_product.py (GitHub)</a>：通过在 GitHub 上创建一个账户来为 CoffeeVampir3/Micro-Mjolinear 的开发做出贡献。</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6_infer.py#L138">T6/model/T6_infer.py (GitHub)</a>：Tensor ProducT ATTenTion Transformer (T6) 的官方实现 - tensorgi/T6</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L107">T6/model/T6.py (GitHub)</a>：Tensor ProducT ATTenTion Transformer (T6) 的官方实现 - tensorgi/T6
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1330173553164943390)** (3 messages): 

> `Titans, Adaptive Transformers, RNN testing, 760M model performance, BABILong` 


- **Titans 和 Adaptive Transformers 引发热议**：最近的讨论强调了对 **Titans** 和 **Adaptive Transformers** 的兴奋，这对即将开展的项目具有潜在影响。
   - 分享了一个关于 **Adaptive Transformers** 的有用[链接](https://sakana.ai/transformer-squared/)，这可能进一步推动了这种热度。
- **评估模型的训练潜力**：一位成员指出，一个拥有 **760M parameters** 的模型在 **BABILong** 上的表现优于商业同类模型。
   - 他们建议从这个极具前景的模型开始评估，同时考虑有关其他模型在测试时使用 **RNNs** 的报告。
- **社区对新模型的支持**：一位成员表达了对这些新模型成功的希望，标志着一个相互支持的社区环境。
   - 这种共同的乐观情绪可能会加强在评估这些技术方面的协作努力。



**Link mentioned**: <a href="https://sakana.ai/transformer-squared/">no title found</a>: no description found

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1329920487861784578)** (15 messages🔥): 

> `Microsoft OpenAI partnership concerns, AI security vulnerability findings, AI compliance tools for trading, TikTok ownership and ban implications, FrontierMath funding controversies` 


- **微软对 OpenAI 的投资引发反垄断警告**：**FTC** 对微软向 **OpenAI** 投资 **$13 billion** 表示担忧，担心这可能会增强该公司在 **AI** 市场的统治地位并损害竞争。
   - **FTC** 主席 **Lina Khan** 强调，此类合作伙伴关系可能导致 *lock-in*（锁定）效应，并使初创公司在获取关键 **AI** 资源方面处于劣势。
- **微软研究人员断言 AI 系统无法完全安全**：在一篇 **pre-print paper** 中，微软研究人员得出结论，**AI** 系统永远无法实现完全安全，这放大了现有的安全风险并引入了新的漏洞。
   - 他们警告说，虽然防御手段可能会提高攻击成本，但基于梯度的攻击和网络钓鱼等威胁仍然普遍存在。
- **AI 工具严厉打击华尔街交易员通信**：合规公司正在部署 **AI** 来解码交易员的通信，以便在监管审查收紧的情况下检测潜在的金融犯罪。
   - 这些 **AI** 系统旨在解释传统方法经常遗漏的复杂俚语和暗语，从而制定更严格的合规措施。
- **最高法院维持 TikTok 禁令，除非被出售**：最高法院维持了一项法律，要求 **TikTok** 必须由其中国母公司出售，否则将面临禁令，理由是其所有权构成了国家安全威胁。
   - 随着法律生效，这一决定造成了巨大的紧迫性，可能会限制该应用程序的下载和更新。
- **围绕 FrontierMath 资金的争议**：**OpenAI** 与 **FrontierMath** 资金之间的联系受到了审查，有说法称承包商直到最近才知道 **OpenAI** 的财务参与。
   - 讨论揭示了对施加在 **Epoch** 上的 **NDA restrictions** 的担忧，这使得许多贡献者对资金来源一无所知。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DanHendrycks/status/1881036645555937719">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>：@GaryMarcus 可以确认像 xAI 这样的 AI 公司无法访问 FrontierMath，因为 Epoch 与 OpenAI 签署了合同义务。</li><li><a href="https://x.com/CarinaLHong/status/1880820323597357273">来自 Carina Hong (@CarinaLHong) 的推文</a>：1. OAI 要求 Epoch 签署 NDA 直至 o3 性能声明发布前夕，防止 Epoch 披露 OAI 是捐赠者且 OAI 拥有独家数据访问权。2. 数学家随后就问题和解法签署 NDA...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>：未找到描述</li><li><a href="https://it.slashdot.org/story/25/01/17/1658230/microsoft-research-ai-systems-cannot-be-made-fully-se">Microsoft Research：AI 系统无法做到完全安全 - Slashdot</a>：根据一篇新的预印本论文，测试了公司 100 多个 AI 产品的 Microsoft 研究人员得出结论，AI 系统永远无法做到完全安全。这项由 26 位作者参与的研究包括...</li><li><a href="https://it.slashdot.org/story/25/01/17/1658230/microsoft-research-ai-systems-cannot-be-made-fully-secure">Microsoft Research：AI 系统无法做到完全安全 - Slashdot</a>：根据一篇新的预印本论文，测试了公司 100 多个 AI 产品的 Microsoft 研究人员得出结论，AI 系统永远无法做到完全安全。这项由 26 位作者参与的研究包括...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hzmpuq/vlc_to_add_offline_realtime_ai_subtitles_what_do/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://slashdot.org/story/25/01/17/1356236/ai-tools-crack-down-on-wall-street-trader-code-speak">AI 工具严厉打击华尔街交易员的“暗语” - Slashdot</a>：随着华尔街和伦敦监管机构加强对市场操纵的审查，合规软件公司正在部署 AI 来解码复杂的交易员通信并检测潜在的金融犯罪。</li><li><a href="https://slashdot.org/story/25/01/17/1958200/microsoft-openai-partnership-raises-antitrust-concerns-ftc-says">FTC 表示 Microsoft-OpenAI 合作伙伴关系引发反垄断担忧 - Slashdot</a>：联邦贸易委员会（FTC）在一份报告中表示，Microsoft 对 OpenAI 的 130 亿美元投资引发了人们的担忧，即这家科技巨头可能会将其在云计算领域的优势延伸到新兴的 AI 市场。</li><li><a href="https://news.slashdot.org/story/25/01/17/1518232/supreme-court-upholds-law-banning-tiktok-if-its-not-sold-by-its-chinese-parent-company">最高法院维持禁止 TikTok 的法律，除非其中国母公司将其出售 - Slashdot</a>：一位匿名读者分享了一份报告：最高法院周五一致维持了联邦法律，从周日开始禁止 TikTok，除非其总部位于中国的母公司将其出售，裁定该...</li><li><a href="https://tech.slashdot.org/story/25/01/17/0012237/google-wont-add-fact-checks-despite-new-eu-law">尽管有欧盟新法，Google 仍不会添加事实核查 - Slashdot</a>：据 Axios 报道，Google 已告知欧盟，尽管有欧盟新法律的要求，它不会在搜索结果和 YouTube 视频中添加事实核查，也不会在排名或删除内容中使用它们。来自...</li><li><a href="https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=FR5bGBmCkcoGniY9m">meemi 的短文 — LessWrong</a>：meemi 的评论 - FrontierMath 由 OpenAI 资助。[1] 关于此事的沟通一直不透明，许多人（包括在该数据集上工作的承包商）并不知情...</li><li><a href="https://slashdot.org/story/25/01/17/1414242/intel-acquisition-target-of-mystery-suitor-semiaccurate-reports">SemiAccurate 报道，Intel 成为神秘买家的收购目标 - Slashdot</a>：科技新闻和研究网站 SemiAccurate 报道称，一家身份不明的公司正寻求整体收购 Intel。该出版物引用了其审阅的一封机密邮件和...</li><li><a href="https://techcrunch.com/2025/01/18/perplexity-ai-submits-bid-to-merge-with-tiktok">Perplexity AI 提交与 TikTok 合并的报价 | TechCrunch</a>：随着 TikTok 在美国面临禁令，Perplexity AI 成为最新的竞标者，希望为这款视频应用提供一个新的公司归宿。CNBC 率先报道。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1330027603075006504)** (81 条消息🔥🔥): 

> `Konkani 语言 AI 模型、Cohere 的易用性、项目想法、API 访问及限制` 


- **Konkani 语言模型计划**：一名成员计划训练一个能够理解 **Konkani 语言**的 AI 模型，并表示尽管需要大学批准，但仍希望项目能有所进展。
   - 他们强调，与行业的合作对于推动项目前进至关重要。
- **关于 Cohere 易用性的担忧**：一名成员强调了关于 **Cohere** 易用性的几点看法，提到了诸如缺乏持久登录、没有深色模式以及缺少移动端应用等问题。
   - 这些功能对用户体验至关重要，与其他服务相比，这些缺失被视为障碍。
- **Cohere API 访问的使用情况**：成员们讨论了**免费 API 访问**，每个模型每月提供 1000 次请求，这使其成为实验的一个便捷选择。
   - 这允许用户在没有财务承诺的情况下使用模型，鼓励对开源做出贡献。
- **对 Cohere 界面的反馈**：成员们分享了关于 **Cohere** 界面和所提供工具的正面反馈，尽管存在某些限制，但仍对其易用性表示赞赏。
   - 大家普遍认为，并非每个模型都需要迎合所有用户，这反映了用户群的多样性。
- **模型切换与更新**：讨论涉及了一个**潜在的模型切换功能**，该功能可以让用户根据需求高效地从各种模型中进行选择。
   - 有传言称即将进行重大更新，引发了人们对平台新功能的期待。



**提到的链接**：<a href="https://github.com/cohere-ai/cohere-python/issues/632">一旦上传模型出错，您的账户（Web 和 API）就会损坏，Dataset/Model 环境将无法再工作 · Issue #632 · cohere-ai/cohere-python</a>：使用您的 CSV 文件示例。import cohere co = cohere.Client() # upload a dataset my_dataset = co.datasets.create( name=&quot;datasettest&quot;, data=open(&quot;./Arts.Class.1000.csv&quot;,...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1330131527928905811)** (11 条消息🔥): 

> `计费问题、AI 行为管理、发票与收据、AI 项目反馈` 


- **关于公司详情的计费问题**：一名成员询问了如何输入公司信息以进行计费，以便用于税收抵扣。
   - *mrdragonfox* 建议联系 [support@cohere.com](mailto:support@cohere.com) 并提供账户 ID 以寻求此问题的帮助。
- **索取抬头为公司的旧发票**：同一名成员询问是否可以接收抬头为公司而非个人的旧发票和收据。
   - *mrdragonfox* 再次建议联系支持部门以协助处理此请求。
- **项目中 AI 行为的挑战**：一名成员分享了他们在故事讲述平台项目中，AI 响应偏离预期 Prompt 的担忧。
   - *xvarunx* 询问了所使用的具体模型的更多细节，并鼓励向支持部门提交反馈。
- **AI 行为管理的局限性**：讨论显示，虽然可以实施 AI 行为的护栏 (guardrails)，但并非万无一失，通常需要通过外部分类器来实现。
   - *mrdragonfox* 提到，目前没有办法完全防止语言模型行为出现偏差。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1329943248915009631)** (12 messages🔥): 

> `Command-R 模型版本管理、Embed Job 并发限制、Dify.ai 集成问题` 


- **Command-R 指向旧版本模型**：讨论中澄清了 `command-r` 目前并未指向最新模型，以避免为使用非时间戳模型的用户引入 **breaking changes**（破坏性变更）。
   - 有人建议利用 **aliases**（别名）来定义版本，同时保留一个 **latest** 标签用于持续更新。
- **Embed Job 限制导致错误**：Khalid 报告收到一个错误，提示已达到 **maximum number of concurrent embed jobs**（最大并发 Embed 任务数），且之前的所有任务都处于停滞状态。
   - 建议他给支持团队发送邮件，因为可能需要检查其账户详情，以解决潜在的任务取消停滞问题。
- **Dify.ai Key 集成被拦截**：Fleck082814 在尝试于自托管的 **dify.ai** 实例中添加 Cohere key 时遇到 **403 Forbidden** 错误，怀疑是 IP 封锁。
   - Xvarunx 指出，类似的请求表明目前不支持来自 **China** 的请求，并建议尝试降级到 **0.8** 版本作为一种变通方法。
- **节假日支持响应通知**：Xvarunx 告知团队，由于美国国家法定假日，支持响应时间可能会受到影响。
   - 这提醒等待支持的用户在节假日期间需要保持耐心。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1330207188932624384)** (32 messages🔥): 

> `Cohere 模型概览、Tool Calling 与代码生成、理解 AGI` 


- **Cohere 模型概览**：分享了 Cohere 模型列表，包括 `command-r`、`c4ai-aya-expanse-8b` 和 `command-light-nightly` 等。
   - 提到用户可以训练模型，以便针对特定用例进行定制。
- **Tool Calling 与代码生成详解**：工具使用的交互涉及开发者通过结构化组件定义 Cohere 模型如何与特定工具交互。
   - 该过程包括 LLM 做出 **tool calls** 决策、执行调用并根据结果生成响应。
- **AGI 定义**：AGI 代表 **Artificial General Intelligence**（通用人工智能），这是一个备受关注的话题。
   - 遗憾的是，在 Cohere 的文档中没有找到关于 AGI 的详细信息。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1331013821858316398)** (4 messages): 

> `Cohere 的数学表现、LLM 的局限性、工具使用技巧` 


- **Cohere 在基础数学上的挣扎**：一位成员对 **Cohere** 的错误计算表示沮丧，具体表现为它错误地将 **18 个月**的总周数计算为 **27 周**。
   - 他们指出，由于需要验证 AI 给出的答案，直接在 Google 上查询通常更快。
- **所有 LLM 都存在数学问题**：另一位成员指出，数学表现问题并非 **Cohere** 独有，而是所有 **LLM** 的共同问题。
   - 他们解释说，经常使用 LLM 的人对此深有体会，这表明数学计算是一个系统性挑战。
- **提升结果的使用技巧**：有人建议要么将 AI 作为类似计算器的工具使用，要么采用更低的 **Temperature** 设置以获得更好的响应。
   - 这强调了用户需要理解 LLM 的概率性质，才能获得准确的输出。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1330399508512641128)** (2 messages): 

> `MOOC 课程确认、春季邮件列表` 


- **等待 MOOC 课程确认**：@gaganadev 询问了关于 **1 月开始的 MOOC 课程**的确认信息。
   - 另一位成员提到，**春季课程的邮件列表**可能会在下周开始发送。
- **春季课程邮件列表公告**：讨论提到与春季课程相关的 **mailing list** 可能会在下周开始分发。
   - 这表明关于课程时间表的进一步细节即将公布。


  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1330010594538946672)** (1 messages): 

> `Document to Podcast blueprint, Open source projects, Community engagement` 


- **Document to Podcast Blueprint 直播介绍**：来自 <@&1316851621027647648> 的团队将在即将举行的活动中，现场介绍 **Document to Podcast** blueprint，这是一个基于 Open source 构建的可定制方案。
   - 鼓励成员加入，并欢迎在这次激动人心的聚会中向 <@1183778352927092806>、<@1300855165393309747> 和 <@1250742001272492097> 提问。
- **Blueprints 增强 Open Source 协作**：这次活动是 <@&1229573172018417674> 齐聚一堂并发现新的、有用的 **Open source 项目** 的绝佳机会。
   - 敦促参与者如果想参加并与社区互动，请点击“感兴趣”按钮。


  

---


---


{% else %}


> 各频道的完整详细内容已针对邮件进行截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}