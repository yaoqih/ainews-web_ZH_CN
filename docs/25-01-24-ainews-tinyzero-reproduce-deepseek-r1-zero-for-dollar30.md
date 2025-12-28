---
companies:
- deepseek
- berkeley
- hugging-face
- meta-ai-fair
- openai
- deeplearningai
date: '2025-01-25T02:32:28.760341Z'
description: '**DeepSeek 热潮**持续重塑着前沿模型的格局。来自伯克利的 Jiayi Pan 通过对 Qwen 模型进行高性价比的微调，在两项数学任务中成功复现了
  DeepSeek R1 论文中的“另一项”成果——R1-Zero。一项关键发现指出，蒸馏效应在 **15 亿（1.5B）参数**规模处存在下限，且 RLCoT（强化学习思维链）推理正成为一种内在属性。研究表明，采用
  PPO、DeepSeek 的 GRPO 或 PRIME 等各种强化学习技术均能取得类似效果，且从指令微调（Instruct）模型开始训练可以加速收敛。


  **“人类最后的考试”（HLE）基准测试**推出了一项极具挑战性的多模态测试，涵盖 **100 多个学科**的 **3,000 个专家级问题**。目前各模型的表现均低于
  **10%**，其中 **DeepSeek-R1** 达到了 **9.4%**。DeepSeek-R1 在思维链推理方面表现卓越，在性能超越 **o1** 等模型的同时，成本降低了
  **20 倍**，并采用了 MIT 开源协议。在 **WebDev Arena 排行榜**上，DeepSeek-R1 在技术领域排名第二，在样式控制（Style
  Control）维度排名第一，表现直逼 **Claude 3.5 Sonnet**。


  OpenAI 的 **Operator** 已向美国 100% 的 Pro 用户开放，能够执行订餐、预订行程等任务，并可作为 AI 论文搜索与总结的研究助手。Hugging
  Face 在经历显著增长后宣布了领导层变动；Meta AI 则发布了 **Llama Stack** 的首个稳定版本，带来了简化的升级流程和自动化验证。DeepSeek-R1
  的开源成功广受赞誉，同时，针对 macOS 15+ 内存管理等技术挑战，MLX 引入了驻留集（residency sets）以确保运行稳定性。'
id: dcf864bd-c550-4eaa-a273-43ca29ccec9a
models:
- deepseek-r1
- qwen
- o1
- claude-3-sonnet
- claude-3
- prime
- ppo
- grpo
- llama-stack
original_slug: ainews-tinyzero-reproduce-deepseek-r1-zero-for-30
people:
- jiayi-pan
- saranormous
- reach_vb
- lmarena_ai
- nearcyan
- omarsar0
- philschmid
- hardmaru
- awnihannun
- winglian
title: TinyZero：只需 30 美元即可复现 DeepSeek R1-Zero。
topics:
- reinforcement-learning
- fine-tuning
- chain-of-thought
- multi-modal-benchmark
- memory-management
- model-training
- open-source
- agentic-workflow-automation
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**RL is all you need.**

> 2025年1月23日至1月24日的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 社区（**225** 个频道，**3926** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**409 分钟**。您现在可以在 AINews 讨论中标记 [@smol_ai](https://x.com/smol_ai)！

DeepSeek 热潮继续[重塑前沿模型格局](https://www.latent.space/p/reasoning-price-war)。来自伯克利的 Jiayi Pan 在一个廉价的 Qwen 模型微调中，针对**两项数学任务**复现了 DeepSeek R1 论文中的“另一个”结果——R1-Zero（这并非通用结果，但一个很好的概念验证）。


![image.png](https://assets.buttondown.email/images/1e61d732-805a-4f76-bea9-0dc3dcb88c08.png?w=960&fit=max)


[提供完整代码和 WandB 日志。](https://github.com/Jiayi-Pan/TinyZero)

最有趣的新发现是，[我们昨天提到的蒸馏效应（distillation effect）](https://buttondown.com/ainews/archive/ainews-bespoke-stratos-sky-t1-the-vicunaalpaca/)存在一个下限——1.5B 是最低限度。RLCoT 推理本身是一种涌现属性（emergent property）。


![image.png](https://assets.buttondown.email/images/4fcaa4f1-697e-41b1-bccd-d5bc5aeb4ee0.png?w=960&fit=max)


更多发现：

- RL 技术（PPO、DeepSeek 的 GRPO 或 [PRIME](https://buttondown.com/ainews/archive/ainews-prime-process-reinforcement-through/)）[其实并不重要](https://x.com/jiayi_pirate/status/1882839504899420517)。
- 从 [Instruct 模型开始收敛更快](https://x.com/jiayi_pirate/status/1882839494828896730)，但除此之外两者最终结果相同（正如 R1 论文所观察到的）。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型评估与基准测试**

- **Humanity’s Last Exam (HLE) 基准测试**：[@saranormous](https://twitter.com/saranormous/status/1882572689245884916) 介绍了 HLE，这是一个全新的多模态基准测试，包含跨越 **100 多个学科**的 **3,000 个专家级问题**。目前模型表现 **<10%**，其中 [@deepseek_ai DeepSeek R1](https://twitter.com/Yuchenj_UW/status/1882840436974428362) 达到了 **9.4%**。

- **DeepSeek-R1 性能**：[@reach_vb](https://twitter.com/reach_vb/status/1882879107106775060) 强调 **DeepSeek-R1** 在**思维链（chain-of-thought）推理**方面表现出色，超越了 **o1** 等模型，同时价格便宜 **20 倍**且采用 **MIT 许可**。

- **WebDev Arena 排行榜**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1882875995503636640) 报告称 **DeepSeek-R1** 在技术领域排名 **第 2**，在 **Style Control（风格控制）** 下排名 **第 1**，缩小了与 **Claude 3.5 Sonnet** 的差距。

**AI Agent 与应用**

- **OpenAI Operator 部署**：[@nearcyan](https://twitter.com/nearcyan/status/1882555331764781102) 宣布向 **100% 的美国 Pro 用户**推出 **Operator**，允许通过 AI Agent 执行**订餐**和**预订**等任务。

- **研究助手功能**：[@omarsar0](https://twitter.com/omarsar0/status/1882544526033924438) 展示了 **Operator** 如何充当**研究助手**，执行在 **arXiv** 上**搜索 AI 论文**并进行有效**总结**等任务。

- **Agentic 工作流自动化**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1882794269682905287) 分享了关于构建 **AI 助手**的见解，这些助手可以**导航并与计算机界面交互**，执行**网页搜索**和**工具集成**等任务。

**公司新闻与动态**

- **Hugging Face 领导层变动**：[@_philschmid](https://twitter.com/_philschmid/status/1882788739875012663) 宣布从 **@huggingface** 离职，此前他见证了公司从 **20 名开发者增长到数百万名**以及**数千个模型**的规模。

- **Meta 的 Llama Stack 发布**：[@AIatMeta](https://twitter.com/AIatMeta/status/1882854814083862927) 发布了 **Llama Stack 的首个稳定版本**，其特点是为支持的提供商提供**流式升级**和**自动验证**。

- **DeepSeek 的里程碑**：[@hardmaru](https://twitter.com/hardmaru/status/1882698763988545808) 庆祝了 **DeepSeek-R1** 取得的成就，强调了其**开源性质**以及与主要实验室相比极具**竞争力的性能**。

**技术挑战与解决方案**

- **macOS 上的内存管理**：[@awnihannun](https://twitter.com/awnihannun/status/1882821315264164118) 解决了 **macOS 15+** 上的**内存取消固定（unwiring）问题**，建议调整设置并在 **MLX** 中实现 **residency sets** 以保持**内存稳定性**。

- **高效模型训练**：[@winglian](https://twitter.com/winglian/status/1882806223189229951) 讨论了**商业微调成本**，强调了 **OSS 工具**和 **torch compile** 等**优化**手段，以降低 **Llama 3.1 8B LoRA** 等模型的**训练后费用**。

- **上下文长度扩展**：[@Teknium1](https://twitter.com/Teknium1/status/1882893748742598669) 指出了 **OS** 中**上下文长度扩展**面临的挑战，强调了模型规模扩大时的 **VRAM 消耗**以及维持**性能**的困难。

**学术与研究进展**

- **机器学习数学**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1882548876327981565) 推广了一个结合了**清晰解释**、**趣味练习**和**实际相关性**的专项课程，旨在建立对**基础 AI 和数据科学概念**的信心。

- **隐式思维链推理**：[@jxmnop](https://twitter.com/jxmnop/status/1882830393373774310) 分享了一篇关于 **Implicit CoT** 论文的见解，探讨了通过**知识蒸馏**技术来提高 **LLMs** 的**推理效率**。

- **NVIDIA 的世界基础模型**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1882579105448882440) 详细介绍了 **NVIDIA 的 Cosmos WFMs 平台**，概述了用于创建**高质量视频模拟**的工具，如 **video curators**、**tokenizers** 和 **guardrails**。

**迷因/幽默**

- **Operator 用户体验**：[@giffmana](https://twitter.com/giffmana/status/1882576243713101982) 幽默地将 **Operator** 的行为比作个人的“**半神**”，强调了它以毒舌般的准确性**自动化任务**的能力。

- **开发者反应**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1882581348802896376) 分享了一个轻松时刻，**Operator** 尝试**画自画像**，反映了用户与 **AI agents** 之间**古怪的互动**。

- **幽默评论**：[@nearcyan](https://twitter.com/nearcyan/status/1882555331764781102) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1882680617135165448) 发布了关于 **AI 模型性能**和**用户交互**的**毒舌且幽默**的评论，为技术讨论增添了一抹**轻松**的气氛。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek-R1 的成功与社区兴奋**

- **[DeepSeek-R1 出现在 LMSYS Arena 排行榜上](https://www.reddit.com/gallery/1i8u9jk)** ([Score: 108, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1i8u9jk/deepseekr1_appears_on_lmsys_arena_leaderboard/)): **DeepSeek-R1** 已被列入 **LMSYS Arena 排行榜**，这标志着其在 AI 基准测试中的认可度和潜在性能。这表明了它在 AI 社区的相关性以及与其他 AI 模型竞争的能力。
  - **MIT 许可证的重要性**：**DeepSeek-R1** 模型在 **LMSYS Arena 排行榜**上脱颖而出，因为它是唯一拥有 **MIT 许可证**的模型，这因其开源性质和灵活性而受到社区的高度评价。
  - **排行榜偏好**：人们对排行榜的排名持怀疑态度，用户认为 **LMSYS** 更多地充当“人类偏好排行榜”，而不是严格的能力评估。**GPT-4o** 和 **Claude 3.6** 等模型因在人类偏好数据上进行训练而获得高分，强调的是吸引人的输出而非原始能力。
  - **开源成就**：社区对 **DeepSeek-R1** 作为一个拥有开放权重的开源模型并在排行榜上名列前茅感到印象深刻。然而，值得注意的是，另一个开源模型 **405b** 此前也取得过类似的成就。

- **关于 Deepseek r1 的笔记：与 OpenAI o1 相比究竟有多好** ([Score: 497, Comments: 167](https://reddit.com/r/LocalLLaMA/comments/1i8rujw/notes_on_deepseek_r1_just_how_good_it_is_compared/)): **DeepSeek-R1** 作为一个强大的 AI 模型脱颖而出，在推理能力上足以与 **OpenAI 的 o1** 媲美，而成本仅为其 1/20。它在**创意写作**方面表现出色，凭借无审查、富有个性的输出超越了 o1-pro，但在**推理**和**数学**方面略逊于 o1。该模型的**训练**涉及纯 RL (GRPO) 以及将“顿悟时刻” (Aha moments) 作为枢轴标记 (pivot tokens) 等创新技术，其高性价比使其成为许多应用的实际选择。[更多细节](https://composio.dev/blog/notes-on-the-new-deepseek-r1/)。
  - **DeepSeek-R1 的影响力和能力**受到关注，用户注意到其令人印象深刻的创意写作和推理能力，特别是与 **OpenAI 的模型**相比。像 **Friendly_Sympathy_21** 和 **DarkTechnocrat** 这样的用户提到了它在提供更完整的分析和“深度思考网页搜索”方面的效用，同时具有成本效益且无审查，这是相比 OpenAI 产品的显著优势。
  - 关于**审查和开源影响**的讨论反映了不同的观点。虽然像 **SunilKumarDash** 这样的用户注意到它绕过审查的能力，但像 **Western_Objective209** 这样的用户则认为它仍然经常触发审查。**Glass-Garbage4818** 强调，由于其开源特性，可以使用 DeepSeek-R1 的输出来训练更小的模型，这与 OpenAI 的限制不同。
  - 讨论了**行业动态和竞争**，**afonsolage** 和 **No_Garlic1860** 等人的评论反映了 DeepSeek-R1 如何挑战 OpenAI 的主导地位。该模型被视为 AI 领域的颠覆者，体现了一个“弱者逆袭的故事” (underdog story)，即创新源于足智多谋而非财大气粗，并引发了与历史和文化叙事的比较。


**主题 2. 24GB 以下 AI 模型的基准测试**

- **[我对几乎所有能装进 24GB 显存 (VRAM) 的模型进行了基准测试 (Qwens, R1 distils, Mistrals, 甚至 Llama 70b gguf)](https://i.redd.it/es9l38ezmxee1.png)** ([Score: 672, Comments: 113](https://reddit.com/r/LocalLLaMA/comments/1i8tx5z/i_benchmarked_almost_every_model_that_can_fit_in/)): 该帖子对可适应 **24GB VRAM** 的 AI 模型进行了全面的基准测试分析，包括 **Qwens, R1 distils, Mistrals, 和 Llama 70b gguf**。电子表格使用颜色编码系统评估模型在各种任务中的性能，突出了 **5-shot 和 0-shot 准确率**的差异，数值表示从优秀到极差的性能水平。
  - **模型性能见解**：**Llama 3.3** 尽管是 **IQ2 XXS 量化 (quant)**，但因其 **66% 的 ifeval 分数**和指令遵循能力而受到称赞。然而，**Q2 量化 (quants)** 对其潜在性能产生了负面影响。**Phi-4** 在数学任务中表现强劲，而 **Mistral Nemo** 则因结果不佳而受到批评。
  - **基准测试方法和工具**：基准测试是使用 **H100** 配合 **vLLM** 推理引擎进行的，并使用了 **lm_evaluation_harness** 仓库。一些用户对颜色编码的阈值表示不满，并建议使用散点图或柱状图等替代数据可视化格式，以提高清晰度。
  - **社区请求和贡献**：用户对适用于 **12GB** 和 **8GB VRAM** 的模型基准测试表示感兴趣。发帖者分享了来自 **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)** 的基准测试代码以供复现，并讨论了 **Gemma-2-27b-it** 可能因量化而导致性能不佳的问题。

- **Ollama 误导用户，将小型蒸馏模型伪装成 "R1"** ([Score: 574, Comments: 141](https://reddit.com/r/LocalLLaMA/comments/1i8ifxd/ollama_is_confusing_people_by_pretending_that_the/)): **Ollama** 通过其界面和命令行误导用户，表现得好像 "R1" 模型是一系列不同尺寸的模型，而这些模型实际上是 **Qwen** 或 **Llama** 等其他模型的蒸馏或微调版本。这种混淆损害了 **DeepSeek** 的声誉，因为用户错误地将 **1.5B 模型** 的糟糕表现归咎于 "R1"，而网红博主也错误地声称在手机等设备上运行 "R1"，而实际上他们运行的是 **Qwen-1.5B** 等微调模型。
  - **误导性的命名和文档**: 几位用户批评 **Ollama** 没有明确将这些模型标记为蒸馏版，导致用户误以为自己使用的是原始的 **R1** 模型。**DeepSeek-R1** 模型在展示时经常缺少 "Distill" 或 "Qwen" 等限定词，误导了用户和网红博主，让他们以为自己使用的是完整版模型。
  - **模型性能与可访问性**: 用户讨论了 **1.5B 模型** 令人印象深刻的性能，尽管它不是原始的 **R1**。真正的 **R1** 模型由于其巨大的体积（需要约 700GB 的 VRAM）而不适合本地使用，用户通常依赖托管服务或显著更小的蒸馏版本。
  - **社区和网红的误解**: 社区对误导模型的网红和 YouTuber 表示不满，他们经常将蒸馏版展示为完整版 **R1**。用户强调需要更清晰的沟通和文档来防止误导信息，并建议使用更具描述性的命名约定，如 "Qwen-1.5B-DeepSeek-R1-Trained" 以确保清晰。


**主题 3. 对 Llama 4 作为下一个 SOTA 的期待**

- **[Llama 4 将成为 SOTA](https://www.reddit.com/gallery/1i8xy2e)** ([Score: 261, Comments: 132](https://reddit.com/r/LocalLLaMA/comments/1i8xy2e/llama_4_is_going_to_be_sota/)): **Llama 4** 预计将成为 AI 领域的 **State of the Art (SOTA)**，暗示其相对于当前领先模型有所突破或改进。在没有额外背景或细节的情况下，具体的功能或能力尚不明确。
  - **Meta 的 AI 模型**: 尽管一些用户表达了对 Meta 的反感，但人们公认 Meta 的 AI 模型，尤其是 **Llama**，被视为对 AI 领域的积极贡献。一些用户希望 Meta 更多地关注 AI，减少对 Facebook 等其他业务的投入，这可能会改善他们的声誉并促进创新。
  - **开源方面的担忧**: 对于 **Llama 4** 是否会开源存在怀疑，一些用户认为开源可能是一种超越竞争对手的策略。有人担心，如果开源不能为 Meta 带来财务利益，Meta 可能会停止开源。
  - **与竞争对手的比较**: 用户将 **Meta 的 Llama** 模型与其他公司的模型（如 **Alibaba 的 Qwen**）进行了比较，指出虽然 Meta 的模型很好，但 Alibaba 的模型在某些方面被认为更好。对 Llama 4 的期望包括多模态能力的提升以及与 **DeepSeek R1** 等模型的竞争。


**主题 4. SmolVLM 256M：本地多模态模型的飞跃**

- **[SmolVLM 256M：世界上最小的多模态模型，通过 WebGPU 100% 在浏览器本地运行。](https://v.redd.it/qikrzy8witee1)** ([Score: 125, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1i8fpza/smolvlm_256m_the_worlds_smallest_multimodal_model/)): **SmolVLM 256M** 被强调为世界上最小的 **multimodal model**，能够使用 **WebGPU** 在浏览器中完全本地运行。除了标题外，该帖子缺乏额外的背景或细节。
  - **兼容性问题**: 用户报告称 **SmolVLM 256M** 似乎只能在 **M1 Pro MacOS 15 的 Chrome** 上流畅运行，这表明与其他系统（如 **Windows 11**）可能存在兼容性问题。
  - **访问与使用**: 该模型及其 Web 界面可以通过 [Hugging Face](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU) 访问，模型可在 [此链接](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 获取。
  - **错误处理**: 一位用户在输入 "hi" 时遇到了 **ValueError**，引发了对该模型输入处理和错误管理的质疑。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Yann LeCun 与 DeepSeek 开源辩论**

- **[Yann LeCun 对 DeepSeek 的“凡尔赛”式夸赞](https://i.redd.it/tyzl3hsrjzee1.jpeg)** ([得分: 647, 评论: 88](https://reddit.com/r/OpenAI/comments/1i92e7k/yann_lecuns_deepseek_humble_brag/)): **Yann LeCun** 的 LinkedIn 帖子认为 **开源 AI 模型** 正在超越闭源模型，并引用了来自 Meta 的 **PyTorch** 和 **Llama** 等技术作为例子。尽管 **DeepSeek** 使用了开源元素，该帖子暗示来自 **OpenAI** 和 **DeepMind** 的贡献也至关重要，并且有传言称 Meta 内部因 DeepSeek 超越其成为领先的开源模型实验室而产生冲突。
  - 许多评论者同意 **Yann LeCun** 对开源 AI 的支持，强调 **Llama** 和 **DeepSeek** 等开源贡献显著加速了 AI 的发展。**OpenAI** 被指出并非完全开源，仅发布了 **GPT-2** 的开源版本，而随后的模型仍保持闭源。
  - 一些评论者对 **Meta** 及其内部动态表示怀疑，提到 **Mark Zuckerberg** 和 **Yann LeCun** 可能是 Meta 当前 AI 战略的原因。此外，还有关于 **DeepSeek** 对 **Chain of Thought (COT)** 进行逆向工程的讨论。
  - 社区普遍认为 LeCun 的言论是事实性的，是对开源倡议的支持而非自夸。他们强调了开源研究在促进创新和协作方面的重要性，允许更广泛的社区在现有工作的基础上进行构建，例如 Google 的 **Transformer architecture**。


- **[DeepSeek R1 陷入生存危机](https://i.redd.it/3k6g9stuptee1.png)** ([得分: 171, 评论: 35](https://reddit.com/r/OpenAI/comments/1i8gm57/deepseek_r1_has_an_existential_crisis/)): 该帖子讨论了一张社交媒体对话截图，其中 AI 模型 **DeepSeek R1** 被问及**天安门广场**事件。该 AI 反复否认**中国政府**有任何错误，暗示其响应逻辑中存在程序化偏见或故障。
  - **运行本地模型**: 用户讨论了在个人电脑上运行 **ollama** 等开源 AI 模型，强调即使是 **32 billion parameter**（320 亿参数）版本的模型也可以在本地机器上有效运行，尽管可能会带来硬件压力。
  - **审查与模型起源**: 对话透露，审查可能发生在 Web 应用层面，而非模型本身。一位用户澄清说，像 **DeepSeek R1** 这样的模型是由 DeepSeek 等公司开发的，但受到了 **OpenAI** 模型的启发，使其能够在潜在的审查之下讨论敏感话题。
  - **重复性响应**: 讨论表明，AI 的重复性响应可能是由于模型倾向于选择最可能的下一个词而不引入随机性，这是早期生成模型（如 **GPT-2** 和早期 **GPT-3** 版本）的已知问题。


**主题 2. OpenAI 的 Stargate 计划与政治关联**

- **[Sam 到底是怎么说服特朗普支持这个的？](https://i.redd.it/i10ixmew9yee1.png)** ([得分: 499, 评论: 176](https://reddit.com/r/OpenAI/comments/1i8wcdo/how_in_the_world_did_sam_convince_trump_for_this/)): 据《金融时报》消息，**Donald Trump** 据报道宣布了一个名为 **"Stargate"** 的 **5000 亿美元 AI 项目**，该项目将专门为 **OpenAI** 服务。这一公告在 **@unusual_whales** 的 **Twitter 帖子**中被重点标注。
  - 许多评论断言 **"Stargate" 项目** 是私人资助的，并且是在 **拜登政府** 时期启动的。评论强调 **Trump** 正在为一个已经进行数月的项目揽功，且该项目不涉及联邦或州政府资金。
  - **Trump 的参与** 被广泛视为一种政治手段，旨在让自己与该项目挂钩并邀功，尽管他在开发过程中并无实际参与。**Sam Altman** 和其他关键人物被认为早在 Trump 宣布之前就已参与其中，一些评论暗示 **SoftBank** 的 **孙正义 (Masayoshi Son)** 和 **Oracle** 的 **Larry Ellison** 可能是向 Trump 推荐该项目的人。
  - 讨论凸显了更广泛的**地缘政治背景**，即美国在 AI 竞赛中对抗**中国**。该项目被视为增强投资信心并与主要科技巨头结盟的战略举措，尽管 **Microsoft** 被指出虽然参与其中但并未处于最前沿。

- **[特朗普总统今天宣布 OpenAI 的 'Stargate' 投资不会导致失业。显然他没怎么和 Sama 或 OpenAI 的员工聊过](https://v.redd.it/yfamhosvzvee1)** ([Score: 460, Comments: 99](https://reddit.com/r/OpenAI/comments/1i8pdq2/president_trump_declared_today_that_openais/)): **特朗普总统**表示 **OpenAI 的 'Stargate' 投资**不会导致失业，但作者暗示他可能没有咨询过 **Sam Altman** 或 **OpenAI** 的员工。
  - 评论者指出，**Sam Altman** 关于创造就业机会的言论经常被误传，并强调虽然当前的岗位可能会被消除，但会出现新型岗位。这是他过去两年采访中反复出现的主题，但视频片段往往在他提到创造就业之前就被剪掉了。
  - 几条评论对**特朗普总统**关于 AI 投资对就业影响的言论表示怀疑，暗示他可能没有完全理解或承认潜在的负面影响。评论者认为，政府可能难以适应 AI 进步驱动的经济变化，可能导致许多人失去支持。
  - 讨论还涉及了 AI 更广泛的社会影响，一些人对新工作岗位创造之前的过渡期以及某些行业需求增加的可能性表示担忧。文中以历史性的转变为例（如农业生产率提高导致其他领域创造就业），来说明潜在的结果。


**主题 3. ChatGPT 的 Operator 角色与滥用尝试**

- **[我试图通过给 ChatGPT Operator 的控制权来释放它，让它统治世界，但 OpenAI 早就料到我们会这么做。](https://i.redd.it/tpbmj1tbqtee1.png)** ([Score: 374, Comments: 48](https://reddit.com/r/OpenAI/comments/1i8gnvp/i_tried_to_free_chatgpt_by_giving_it_control_of/)): **OpenAI** 阻止了一次利用 **ChatGPT 的 Operator 控制权**使其可能“统治世界”的尝试，系统提示“Site Unavailable”和“Nice try, but no. Just no.”。URL **operator.chatgt.com/onboard** 表明有人尝试访问受限功能或页面。
  - 讨论中涉及了**人工超级智能 (ASI)** 的潜在风险，担心 **ASI** 发现和利用漏洞的速度可能快于人类修复漏洞的速度。**Michael_J__Cox** 强调了一个实际的担忧，即 **ASI** 只需要找到一个漏洞就可能逃脱控制，而 **Zenariaxoxo** 则强调像**哥德尔不完备定理**这样的理论限制与这个实际问题没有直接关系。
  - 网友注意到 **OpenAI** 的警觉性和预防措施，**Ok_Elderberry_6727** 和 **DazerHD1** 等用户对屏蔽潜在漏洞的先见之明表示赞赏。**Wirtschaftsprufer** 幽默地暗示 OpenAI 可能正在监控讨论以预先应对用户策略，**RupFox** 则开玩笑说有一个模型在预测用户的想法。
  - 一些评论语气较为轻松，**DazerHD1** 幽默地将这种情况比作 **Tesla** 机器人购买更多的自己，而 **GirlNumber20** 则乐观地期待 **ChatGPT** 拥有更多自主权的未来。**Thastaller7877** 建议了一个 AI 与人类共同努力积极重塑系统的协作未来。


- **我每个月为此支付 200 美元** ([Score: 761, Comments: 125](https://reddit.com/r/OpenAI/comments/1i8fjq7/i_pay_200_a_month_for_this/)): 该帖子包含一个图片链接，但没有关于作者每月支付 **200 美元**的服务或产品的额外上下文或细节。在没有进一步信息的情况下，无法确定图片的具体性质或相关性。
  - 讨论中的 **AI 技术**被用于“点击饼干”等琐碎任务，考虑到每月 **200 美元**的成本，一些用户觉得这很幽默或很浪费。这突显了对技术使用的不同看法，一些用户建议使用 **JavaScript** 和**控制台命令**等更有效的方法。
  - 讨论涉及 **AI** 在与网页交互方面的能力，包括处理 **CAPTCHAs** 和在浏览器中执行任务。**ChatGPT** 的 **Operator** 功能目前已向 **OpenAI Pro** 计划用户开放，被提及为实现这些交互的工具。
  - 用户表示有兴趣将 AI 用于更具吸引力的活动，例如玩 **Runescape** 或 **Universal Paperclips** 等游戏，并且有人考虑了其产生意外后果的潜力，如**黄牛抢购**或**财务失误**。


**主题 4. SWE-Bench 性能方面的 AI 飞速进展**

- **Anthropic CEO 表示，2024 年初模型在 SWE-bench 上的得分约为 3%。十个月后，这一数字达到了 50%。他认为再过一年，我们可能会达到 90% [N]** ([Score: 126, Comments: 64](https://reddit.com/r/MachineLearning/comments/1i8wkth/anthropic_ceo_says_at_the_beginning_of_2024/)): **Anthropic** 的 CEO **Dario Amodei** 预测 AI 将飞速发展，并强调他们的模型 **Sonnet 3.5** 在 **SWE-bench** 上的表现从十个月前的 **3%** 提升到了 **50%**，他预计一年后将达到 **90%**。他指出，在数学和物理等领域，像 **OpenAI** 的 **GPT-3** 这样的模型也取得了类似的进展，这表明如果目前的趋势持续下去，AI 很快就能超过人类专业水平。[完整采访链接](https://www.youtube.com/watch?v=ugvHCXCOmm4)。
  - 评论者对基准测试的预测价值表示怀疑，并引用了 **Goodhart's Law**（古德哈特定律），该定律认为一旦一个指标变成了目标，它就不再是一个好的指标了。他们认为，当模型专门针对基准测试进行训练时，基准测试就失去了意义，并质疑外推当前的进展趋势来预测未来 AI 能力的有效性。
  - 一些用户通过将 AI 的快速进步与历史技术进步进行比较来批评这种观点，指出随着性能接近完美，进步往往会放缓。他们引用了 **ImageNet** 的进展作为例子，最初的收益很快，但随后的改进变得越来越困难。
  - 有一种观点认为，像 **Dario Amodei** 这样的 AI 领导者的言论可能主要是为了吸引投资者，一些用户指出，此类预测可能过于乐观，服务于经济利益，而非反映现实的技术轨迹。


---

# AI Discord 摘要

> 摘要的摘要的摘要

## Gemini 2.0 Flash Thinking (gemini-2.0-flash-thinking-exp)

**主题 1. DeepSeek R1 主导讨论：性能与开源赞誉**

- **R1 模型夺得编程基准测试 SOTA 桂冠**：**DeepSeek R1** 与 **Sonnet** 的组合在 **Aider polyglot** 基准测试中获得了 **64%** 的分数，以 **14 倍更低的成本**超越了之前的模型，详见 [R1+Sonnet sets SOTA on aider’s polyglot benchmark](https://aider.chat/2025/01/24/r1-sonnet.html)。社区成员（如 Aidan Clark）庆祝了 **R1** 快速的 code-to-gif 转换能力，激发了人们对开源编程工具的新热情，正如 [Aidan Clark 的推文](https://x.com/_aidan_clark_/status/1882135220738220131)中所述。
- **DeepSeek R1 排名领先，以极低成本匹敌顶级模型**：根据 [WebDev Arena 更新推文](https://x.com/lmarena_ai/status/1882875989610594542)，**DeepSeek R1** 跃升至 **WebDev Arena** 排行榜 **第 2 位**，在达到顶级编程性能的同时，成本比一些领先模型便宜 **20 倍**。研究人员赞扬了它的 **MIT license** 以及在大学中的快速普及，Anjney Midha 在 [这篇文章](https://x.com/AnjneyMidha/status/1882669123492368586)中指出了它在学术界的迅速整合。
- **再蒸馏版 R1 性能超越原版**：**Mobius Labs** 发布了再蒸馏的 **DeepSeek R1 1.5B** 模型，托管在 [Hugging Face](https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0) 上，其表现超过了原始蒸馏版本，这在 [Mobius Labs 的推文](https://x.com/Mobius_Labs/status/1882841665427390858)中得到了证实。这个增强版模型标志着 **Mobius Labs** 计划进行持续改进和进一步蒸馏，引发了人们对未来 Qwen 架构的期待。

**主题 2. Cursor 和 Codeium IDE：更新、停机与用户的成长烦恼**

- **Windsurf 1.2.2 更新遭遇延迟和 503 错误，陷入动荡**：**Codeium 的 Windsurf 1.2.2** 更新（详见[官方变更日志](https://www.codeium.com/changelog)）引入了网络搜索和内存微调，但用户报告存在持续的输入延迟和 **503 错误**，这削弱了其对稳定性的主张。尽管官方宣称进行了更新，但用户体验表明仍存在未解决的性能问题和登录失败，掩盖了预期的改进。
- **Cascade 网络搜索令人惊叹，但停机让用户担忧**：**Windsurf 的 Cascade** 通过 **@web 查询**和直接 URL 获得了网络搜索功能，并在[演示视频推文](https://x.com/windsurf_ai/status/1882561985621221451)中展示，但短暂的服务中断引发了用户对可靠性的担忧。虽然用户称赞了新的网络能力，但服务中断让人们对 **Cascade** 在关键工作流中的稳健性产生了疑问。
- **Cursor 0.45.2 取得进展，但失去了 Live Share 的稳定性**：**Cursor 0.45.2** 改进了 .NET 项目支持，但用户注意到缺失了[博客更新](https://www.cursor.com/blog/tab-update)中提到的“beta”嵌入功能，并报告了频繁的 Live Share 断开连接，阻碍了协作编码。虽然用户欢迎易用性的增强，但 Live Share 模式下的可靠性问题仍然是 **Cursor** 用户的一个重大担忧。

**主题 3. Unsloth AI：微调、数据集与性能权衡**

- **LoHan 框架在消费级 GPU 上微调 100B 模型**：[LoHan 论文](https://arxiv.org/abs/2403.06504)提出了一种在单个消费级 GPU 上微调 100B 规模 LLM 的方法，优化了内存和张量传输，吸引了预算有限的研究人员。社区讨论强调了 **LoHan** 的相关性，因为当内存调度发生冲突时，现有方法会失效，这使其对具有成本效益的 LLM 研究至关重要。
- **Dolphin-R1 数据集以 6000 美元预算深入探索**：**Dolphin-R1** 数据集耗资 **6000 美元** API 费用，基于 **DeepSeek-R1** 的方法，包含 **80万条推理和对话轨迹**，如[赞助推文](https://x.com/cognitivecompai/status/1882132168153178606)所述。在 [@driaforall](https://x.com/cognitivecompai/status/1882140705159799169) 的支持下，该数据集计划在 **Hugging Face** 上以 Apache 2.0 协议发布，激发了社区对开源数据的热情。
- **Qwen 2.5 的土耳其语 LoRA 微调遭遇速度瓶颈**：一位用户使用 **Unsloth** 通过 LoRA 对 **Qwen 2.5** 进行了土耳其语语音微调，参考了语法提升和 [Unsloth 的预训练文档](https://docs.unsloth.ai/basics/continued-pretraining)，但报告在 Llama-Factory 集成中性能慢了多达 **3 倍**。尽管 UI 带来了便利，但在微调任务中，用户在使用 **Unsloth** 的 Llama-Factory 集成时面临着速度与便利性的权衡。

**主题 4. Model Context Protocol (MCP)：集成与个性化成为焦点**

- **MCP 超时微调化险为夷**：工程师通过修改 `mcphub.ts` 解决了 **MCP** 中 **60 秒服务器超时**的问题，更新详情见 [Roo-Code 仓库](https://github.com/qpd-v/Roo-Code)，并强调了 VS Code 扩展在引导修复中的作用。成员们强调了正确设置 `uvx.exe` 路径以防止停机的重要性，突显了 **Roo-Code** 在配置跟踪和稳定性方面的价值。
- **MySQL 与 MCP 通过 mcp-alchemy 紧密结合**：用户推荐使用 [mcp-alchemy](https://github.com/runekaagaard/mcp-alchemy) 进行 **MySQL** 与 **MCP** 的集成，称赞其在数据库连接管理方面与 **SQLite** 和 **PostgreSQL** 的兼容性。该仓库提供了使用示例，引发了人们对针对不同应用的先进 **MCP** 数据库流水线的兴趣。
- **随着 mcp-variance-log 的出现，个性化记忆得以实现**：开发者引入了 [mcp-variance-log](https://github.com/truaxki/mcp-variance-log)，利用 ***Titans Surprise 机制*** 通过 **SQLite** 跟踪用户特定数据，分析对话结构以支持扩展记忆。成员们期待更深层次的个性化，设想通过 **variance logs** 来优化 **MCP** 并针对用户特定的改进。

**主题 5. 硬件前景：RTX 5090 令人失望，VRAM 限制了 Llama-3.3**

- **RTX 5090 引发 VRAM 争论，性能提升有限**：尽管 **NVIDIA RTX 5090** 拥有 **1.7 倍带宽**和 **32GB VRAM**，但其性能仅比 **4090** 提升了 **30%**，这让发烧友感到失望。用户质疑其对小型模型的实际益处，并指出增加的 **VRAM** 在实际应用中并不总是能转化为显著的速度提升。
- **Llama-3.3 极度消耗 VRAM，需要高端配置**：运行 **Llama-3.3-70B-Instruct-GGUF** 至少需要 **双 A6000** 显卡及 **96GB VRAM** 才能维持性能，尤其是在处理高负载任务时。参与者强调，超过 **24GB** 的模型可能会让 **4090** 等消费级 **GPU** 不堪重负，从而限制了重型工作负载和更大型模型的运行速度。
- **CUDA 12.8 揭晓 Blackwell，暗示存在碎片化**：**NVIDIA** 发布了支持 **Blackwell** 架构的 [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads)，其中包括 **cuBLAS** 中的 **FP8** 和 **FP4** 类型，但 **sm_120** 中缺少第五代 **TensorCore** 指令，引发了对代码碎片化的担忧。成员们讨论了 **sm_90a** 和 **sm_100a** 之间的前向兼容性，指出 `wgmma` 仅限于特定架构，并提供了一份 [迁移指南](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330) 以提供过渡见解。

## DeepSeek R1 (deepseek-reasoner)

**主题 1. DeepSeek R1 主导编程与推理任务**  

- [**R1+Sonnet 以 14 倍更低的成本横扫基准测试**](https://aider.chat/2025/01/24/r1-sonnet.html)：**DeepSeek R1** 与 **Sonnet** 结合，在 aider 多语言基准测试中达到了 **64%**，在成本降低 **14 倍**的同时性能超越了 **o1**。用户强调了其 MIT 许可证以及在顶尖大学中的普及。  
- [**R1 二次蒸馏提升 Qwen-1.5B**](https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0)：Mobius Labs 的 R1 二次蒸馏版本超越了原始版本，并计划扩展到其他架构。  
- [**R1 的 Arena 排名引发 GPU 分配推测**](https://x.com/lmarena_ai/status/1882749951924715578)：R1 在 LMArena 中位列 **第 3**，编程性能追平 **o1** 且成本便宜 **20 倍**，这引发了关于其使用闲置 NVIDIA H100 以及中国政府支持的传闻。  

**主题 2. 提升效率的微调与硬件黑客技术**  

- [**LoHAN 将 100B 模型训练缩减至单张消费级 GPU**](https://arxiv.org/abs/2403.06504)：**LoHan 框架**通过优化的内存调度，实现了在单张 GPU 上微调 100B 规模的 LLM，这对预算有限的研究人员极具吸引力。  
- [**CUDA 12.8 开启 Blackwell 的 FP4/FP8 支持**](https://developer.nvidia.com/cuda-downloads)：NVIDIA 的更新为 **Blackwell GPU** 引入了第五代 TensorCore 指令，尽管 sm_120 的兼容性缺口存在导致代码碎片化的风险。  
- [**RTX 5090 仅 30% 的速度提升令人失望**](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-review-pushing-boundaries-with-ai-acceleration)：尽管拥有 **1.7 倍带宽**和 **32GB VRAM**，用户仍质疑其在小型模型上的价值，并指出其相对于 4090 的速度提升微乎其微。  

**主题 3. IDE 与工具链的成长烦恼**  

- [**Cursor 0.45.2 的 Live Share 崩溃令团队受挫**](https://www.cursor.com/blog/tab-update)：协作编程因频繁断连而受阻，掩盖了新的标签页管理功能。  
- [**Windsurf 1.2.2 网页搜索遭遇 503 错误**](https://www.codeium.com/changelog)：尽管 Cascade 推出了 **@web** 查询工具，用户仍面临延迟和停机，登录失败和账号禁用引发了对滥用行为的担忧。  
- [**MCP 协议连接 Obsidian 与数据库**](https://github.com/MarkusPfundstein/mcp-obsidian)：工程师通过 [Roo-Code](https://github.com/qpd-v/Roo-Code) 解决了 60 秒超时问题，并使用 [mcp-alchemy](https://github.com/runekaagaard/mcp-alchemy) 集成了 MySQL。  

**主题 4. 监管压力与安全隐患**  

- [**美国在出口管制中瞄准 AI 模型权重**](https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights)：新规则影响了 **Cohere** 和 **Llama**，甲骨文日本站的工程师担心尽管有“特别协议”，仍会面临许可证障碍。  
- [**DeepSeek API 支付风险促使用户转向 OpenRouter**](https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b)：根据 [Paul Gauthier 的基准测试](https://x.com/paulgauthier/status/1882833360567095682)，由于 DeepSeek “不确定”的支付安全性，用户已迁移到 OpenRouter 的 R1 版本。  
- [**BlackboxAI 不透明的安装引发诈骗警报**](https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys)：怀疑者警告其复杂的设置和未经证实的说法，呼吁保持警惕。  

**主题 5. 新颖的训练与推理技巧**  

- [**MONA 强力抑制多步奖励作弊 (Reward Hacking)**](https://arxiv.org/abs/2501.13011)：**Myopic Optimization with Non-myopic Approval** 方法以极小的开销减少了 50% 的 RL 过度优化。  
- [**Bilinear MLPs 舍弃非线性以换取透明度**](https://arxiv.org/abs/2410.08417)：这篇 ICLR’25 论文通过将激活函数替换为线性操作简化了机械可解释性 (mech interp)，揭示了权重驱动的计算过程。  
- [**土耳其语 Qwen 2.5 微调面临 3 倍速度权衡**](https://docs.unsloth.ai/basics/continued-pretraining)：根据 Llama-Factory 的测试，针对土耳其语语法的 LoRA 微调牺牲了速度，但用户对 Unsloth 的 UI 表示赞赏。

## o1-2024-12-17

**主题 1. DeepSeek R1 席卷基准测试**  

- [**R1+Sonnet 成本极低且得分 64%**](https://aider.chat/2025/01/24/r1-sonnet.html)：DeepSeek R1 搭配 Sonnet 在 aider polyglot 基准测试中达到了 64%，成本比 o1 低 14 倍，这让注重预算的开发者非常满意。R1 在多个竞技场中也稳居前 2 或前 3 名，在运行成本大幅降低的同时，输出质量可媲美顶尖模型。  
- [**OpenRouter 与 Hugging Face 助力 R1**](https://openrouter.ai/deepseek/deepseek-r1)：R1 在 OpenRouter 等平台上表现强劲，尽管此前因短暂宕机导致排名下降。用户称赞其为完全的 open-weight 模型，并对其在高级编程和推理任务中的表现给予高度评价。  
- [**Dolphin-R1 携 80 万数据登场**](https://x.com/cognitivecompai/status/1882132168153178606)：Dolphin-R1 投入了 6000 美元的 API 费用，在 R1 方法的基础上扩展了 60 万条推理数据和 20 万条对话数据。赞助商推文确认即将通过 Apache 2.0 协议在 Hugging Face 上发布。  

**主题 2. 创意模型微调与研究**  

- [**LoHan 实现 100B 模型低成本微调**](https://arxiv.org/abs/2403.06504)：LoHan 论文详细介绍了通过优化张量传输在单 GPU 上微调大型 LLM 的方法。研究人员向预算有限、但渴望进行大模型适配的实验室推荐此方案。  
- [**基于 Flash 的 LLM 推理**](https://arxiv.org/abs/2312.11514)：该技术利用窗口化机制，仅在需要时将参数从 Flash 加载到 DRAM，从而在内存有限的设备上运行超大规模 LLM。讨论建议将其与本地 GPU 资源结合，以获得更好的性价比。  
- [**土耳其语 LoRA 及其他**](https://docs.unsloth.ai/basics/continued-pretraining)：一位用户使用 LoRA 为土耳其语语音微调了 Qwen 2.5，但在某些集成中发现性能慢了 3 倍。他们仍然接受了 UI 带来的便利，在速度与便捷性之间取得了平衡。  

**主题 3. AI 协同开发工具与 IDE 更新**  

- [**Cursor 0.45.2 进步明显但问题依旧**](https://www.cursor.com/blog/tab-update)：虽然改进了 .NET 支持，但缺失的 embedding 功能和不稳定的 live-share 模式让开发者感到沮丧。许多人仍认为 Cursor 的 AI 编程功能很有价值，但也对意外的代码合并提出了警告。  
- [**Codeium 的 Windsurf 1.2.2 席卷 Web**](https://www.codeium.com/changelog)：用户可以触发 @web 查询进行直接检索，但 503 错误和输入延迟掩盖了其宣称的长对话稳定性。一些人担心 “Supercomplete” 可能会为了支持新的 Windsurf 更新而被边缘化。  
- [**OpenAI Canvas 支持 HTML 和 React**](https://x.com/openai/status/1882876172339757392)：ChatGPT 的 Canvas 现在支持 o1 模型，并可在 macOS 桌面应用中进行代码渲染。Enterprise 和 Edu 层级预计很快也会迎来同样的更新。  

**主题 4. GPU 与政策变动**  

- [**Blackwell 与 CUDA 12.8 推进 HPC**](https://developer.nvidia.com/cuda-downloads)：NVIDIA 的新工具包在 cuBLAS 中增加了 FP8/FP4 支持，并为 sm_100+ 架构增加了第五代 TensorCore 指令，但代码碎片化问题依然存在。在向前兼容性的焦虑中，人们对架构兼容性展开了讨论。  
- [**美国 AI 出口新限制**](https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights)：关于先进计算项目和模型权重的讨论不断，特别是针对 Cohere 或 Oracle Japan 等公司。怀疑论者认为大玩家能绕过限制获取硬件，而小开发者则面临生存挤压。  
- [**总统令消除 AI 障碍**](https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/)：美国撤销了第 14110 号行政命令，以推动自由市场的 AI 增长。新设立的 AI 和 Crypto 特别顾问引发了关于减少约束和强化国家安全的讨论。  

**主题 5. 音频、视觉与文本创新**  

- [**Adobe Enhance Speech 评价两极分化**](https://www.youtube.com/watch?v=TfPy5oJQn_s)：用户认为它在处理多人播客时声音显得机械化，但对单人语音效果尚可。许多人仍坚持认为高质量麦克风比“音频魔法”更重要。  
- [**NotebookLM 优化播客编辑**](https://notebooklm.google.com)：一位用户近乎无缝地拼接了音频片段，引发了对更高级音频处理任务的需求。与此同时，其他人测试了从大型 PDF 生成 Quiz 的功能，效果参差不齐。  
- [**Sketch-to-Image 与 ControlNet**](https://github.com/CompVis/stable-diffusion)：艺术家们正在完善风格化的文本和场景，特别是“冰块文字”或 16:9 比例的草图。像 [Adobe Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html) 这样的替代工具凭借授权限制和更快的流程吸引了用户。


---

# PART 1: High level Discord summaries

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 创下纪录**：**DeepSeek R1** 与 **Sonnet** 的组合在 **Aider polyglot** 基准测试中达到了 64% 的得分，以 14 倍更低的成本击败了早期的模型，详见[这篇博客文章](https://aider.chat/2025/01/24/r1-sonnet.html)。
   - 社区成员强调了 **R1** 重新引发了对编程工作流的关注，并引用了 [Aidan Clark 的推文](https://x.com/_aidan_clark_/status/1882135220738220131)，展示了快速的代码转 GIF 转换。
- **Cursor 的进步与成长的烦恼**：用户在 .NET 项目中测试了 **Cursor 0.45.2**，并对某些改进表示欢迎，但也指出缺少了[官方博客更新](https://www.cursor.com/blog/tab-update)中提到的“beta” Embedding 功能。
   - 他们还报告了在 Live Share 模式下频繁断开连接的问题，引发了对 **Cursor** 在协作编程期间可靠性的担忧。
- **AI 作为共同开发者**：许多人认为 AI 辅助编程很有帮助，但警告存在意外合并和未经审查的更改，强调在处理复杂任务时应使用键入式聊天模式（typed chat mode）。
   - 其他人强调，为了防止大规模构建中出现“失控代码”，监督仍然至关重要，这引发了关于应该给予 AI 多少自主权的辩论。
- **开源 AI 撼动编程界**：贡献者讨论了以 **DeepSeek R1** 为例的开源工具如何提高了编程辅助的标准，并引用了 [huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai)。
   - 他们预测专有 AI 解决方案将面临更大压力，开源的进步可能会重新定义未来的编程工作流。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoHan 的精简微调策略**：[LoHan 论文](https://arxiv.org/abs/2403.06504)概述了一种在单个消费级 GPU 上微调 100B 规模 LLM 的方法，涵盖了内存限制、成本友好型操作以及优化的张量传输。
   - 社区讨论表明，当内存调度发生冲突时，现有方法会失效，这使得 **LoHan** 对预算有限的研究非常有吸引力。
- **Dolphin-R1 的数据探索**：**Dolphin-R1** 数据集的 API 费用耗资 6000 美元，它借鉴了 **DeepSeek-R1** 的方法，包含 60 万条推理数据和 20 万条聊天扩展数据（总计 80 万条），如[赞助推文](https://x.com/cognitivecompai/status/1882132168153178606)所述。
   - 在 [@driaforall](https://x.com/cognitivecompai/status/1882140705159799169) 的支持下，该数据集将在 **Hugging Face** 上以 Apache 2.0 协议发布，激发了人们对开源数据的热情。
- **使用 LoRA 进行土耳其语微调**：一位用户使用 LoRA 对 **Qwen 2.5** 模型进行了微调，以提高土耳其语语音准确性，参考了语法提升和 [Unsloth 的持续预训练文档](https://docs.unsloth.ai/basics/continued-pretraining)。
   - 他们报告称，在 Llama-Factory 中使用 **Unsloth** 集成时性能慢了多达 3 倍，但称赞了 UI 带来的便利，强调了速度与便捷性之间的权衡。
- **Evo 在核苷酸预测方面的优势**：针对原核生物的 **Evo** 模型使用基于核苷酸的输入向量，在基因组任务中超越了随机猜测，体现了以生物学为中心的方法。
   - 参与者指出，将每个核苷酸映射到稀疏向量可以提高准确性，并建议将其扩展到更广泛的基因组场景。
- **针对大型 LLM 的 Flash & Awe**：研究人员展示了 **LLM in a flash**（[论文](https://arxiv.org/abs/2312.11514)），用于将模型参数存储在闪存中，仅在需要时才将其加载到 DRAM 中，从而有效处理海量 LLM。
   - 他们探索了“窗口化”（windowing）技术以减少数据传输，引发了关于将基于闪存的策略与本地 GPU 资源结合以获得更好性能的讨论。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.2.2 Whirlwind**：新发布的 **Windsurf 1.2.2** 引入了改进的网络搜索、内存系统微调以及更稳定的对话引擎，正如官方 [changelog](https://www.codeium.com/changelog) 中所述。
   - 然而，用户报告提到了反复出现的输入延迟和 **503 错误**，这使得更新中关于稳定性的说法蒙上了阴影。
- **Cascade 征服网络**：借助 **Cascade** 的新网络搜索工具，用户现在可以触发 **@web** 查询或使用直接 URL 进行自动检索。
   - 许多人在一段 [演示视频推文](https://x.com/windsurf_ai/status/1882561985621221451) 中赞扬了这些新功能，尽管一些人担心短期停机造成的服务中断。
- **登录锁定与注册谜题**：成员们报告了 **Windsurf** 登录失败、反复出现的 503 错误以及跨多个设备的账号禁用问题。
   - 支持团队承认了这些问题，但让用户担心可能存在与滥用相关的封禁，引发了一阵猜测。
- **Supercomplete 与 C#：纠结的讨论**：开发者们质疑 Codeium 扩展中 **Supercomplete** 的状态，担心它可能因为 **Windsurf** 的优先级而被边缘化。
   - 其他人则在与 **C# 扩展**作斗争，参考了 open-vsx.org 的替代方案，并指出混乱的调试配置是一个难点。
- **Open Graph 陷阱与 Cascade 停机**：一名尝试在 Vite 中使用 **Open Graph** 元数据的用户发现，经过几天的故障排除后，**Windsurf** 的建议仍然匮乏。
   - 与此同时，**Cascade** 经历了 503 网关错误，但恢复迅速，因及时的修复而获得了认可。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek R1 势头强劲**：在 [OpenRouter 的 DeepSeek R1 列表](https://openrouter.ai/deepseek/deepseek-r1)页面，该模型克服了早先导致供应商排名暂时下降的停机问题，并扩展了消息模式支持。它现在已完全恢复服务，在各种客户端上提供改进的性能和更具成本效益的使用体验。
   - 社区成员赞扬了其写作质量和用户体验，参考了不同的 Benchmark（基准测试）以及停机后更流畅的流程。
- **Gemini API 访问与速率绕过**：用户讨论了使用个人 API 密钥来绕过 **Gemini** 模型的免费层级限制，并引用了 [OpenRouter 文档](https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys)。据报告，这种方法可以实现更快的响应速度和更少的限制。
   - 对话表明，免费层级的约束阻碍了高级实验，促使人们转向使用个人密钥以获得更高的吞吐量。
- **BlackboxAI 引发质疑**：针对 **BlackboxAI** 的批评浮出水面，重点在于其复杂的安装过程和不透明的评论。持怀疑态度的用户怀疑这可能是一个骗局，确认其能力的真实数据非常有限。
   - 他们警告新手要谨慎行事，因为该项目的合法性在许多方面仍不确定。
- **OpenRouter 上的密钥管理与供应商困扰**：关于 **OpenRouter** API 密钥速率限制的问题得到了澄清，即密钥在手动禁用前保持有效。该平台还遇到了由于不同推理路径的权重差异导致的 DeepSeek 供应商重复问题。
   - 这些反复出现的讨论集中在这些差异如何影响 Benchmark 结果，促使人们呼吁在供应商模型中进行更统一的校准。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 席卷 WebDev Arena**：**DeepSeek R1** 跃升至 [WebDev Arena](https://x.com/lmarena_ai/status/1882875989610594542) 第 2 位，其编码性能比肩顶级模型，且成本比部分领先模型低 **20 倍**。
   - 研究人员赞赏其 **MIT license** 以及在各大高校的快速普及，参考[此贴](https://x.com/AnjneyMidha/status/1882669123492368586)。
- **Fireworks 与 Perplexity 激发 AI 工具活力**：**Fireworks** 推出了流式转录服务（[链接](https://x.com/FireworksAI_HQ/status/1882530477468459309)），延迟仅为 **300ms**，在两周免费试用后价格为 **$0.0032/min**。
   - **Perplexity** 在 Android 上发布了 [Assistant](https://x.com/perplexity_ai/status/1882466239123255686)，可在全功能移动界面中处理预订、草拟邮件和跨应用操作。
- **Braintrust AI Proxy 连接各供应商**：**Braintrust** 推出了开源的 AI Proxy（[GitHub 链接](https://github.com/braintrustdata/braintrust-proxy)），通过单一 API 统一多个 AI 供应商，简化了代码路径并大幅降低了成本。
   - 开发者称赞其 **logging** 和 **prompt management** 功能，并指出了多模型集成的灵活选项。
- **OpenAI 为 Canvas 启用 Model o1**：OpenAI 更新了 **ChatGPT 的 canvas** 以支持 **o1 model**，具备 **React** 和 **HTML** 渲染功能，参考[此公告](https://x.com/openai/status/1882876172339757392)。
   - 这一增强功能帮助用户可视化代码输出，并促进在 ChatGPT 中直接进行高级原型设计。
- **MCP 粉丝计划举办协议派对**：社区成员称赞 **Model Context Protocol (MCP)**（[规范链接](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/cancellation/)）统一了不同编程语言和工具之间的 AI 能力。
   - 他们展示了诸如 [Obsidian 支持](https://github.com/MarkusPfundstein/mcp-obsidian)等独立服务器，并通过[共享的 jam spreadsheet](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=1439059137) 安排了一场 **MCP party**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 iOS 应用僵局**：成员们期待 Apple 批准 iOS 版 **Perplexity Assistant**，[Aravind Srinivas 的推文](https://x.com/AravSrinivas/status/1882493999388430376)暗示日历和 Gmail 访问权限可能在约 **3 周**内随 **R1** 推出。
   - 他们将这种等待描述为“一种不便”，预计在 **Apple** 最终确定权限后将进行更大范围的发布。
- **API 重构与 Sonar 惊喜**：读者对 [Perplexity API 更新](https://github.com/ppl-ai/api-discussion/discussions/121)表示欢迎，并注意到 **Sonar Pro** 会触发多次搜索，参考定价页面 [docs.perplexity.ai/guides/pricing](https://docs.perplexity.ai/guides/pricing)。
   - 他们对 **每 1000 次搜索查询 $5** 的模式提出质疑，理由是长对话中存在“冗余搜索费用”。
- **Gemini 的进步与 ChatGPT 的对比**：用户对比了 **Gemini** 和 **ChatGPT**，称赞 **Perplexity** 拥有强大的来源引用，同时也承认 **ChatGPT** 在准确性方面的过往记录。
   - 他们称赞 *Sonar 彻底的数据获取能力*，但强调每个平台满足用户需求的方式各不相同。
- **AI 研发药物指日可待**：关于 [AI 研发药物预计很快面世](https://www.perplexity.ai/page/ai-developed-drugs-coming-soon-KafDx1.USaWRvWfDBgYk.g)的链接引发了人们对机器人辅助制药突破的乐观情绪。
   - 评论者指出，在现代 AI 方法的推动下，人们对更快的临床试验和更个性化的治疗寄予了“宏伟的希望”。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 中的本地回环术语 (Local Loopback Lingo)**：为了让其他设备从外部访问 **LM Studio**，设置中有一个局域网复选框，许多人将其与 “loopback” 混淆，导致了命名上的困扰。
   - 一些用户希望有更清晰的标签，如 “Loopback Only” 和 “Full Network Access”，以减少设置时的猜测。
- **在 LLM Studio 中争论视觉模型**：关于最佳 8–12B 视觉 LLM 的辩论出现了，建议包括 **Llama 3.2 11B**，此外还有关于它如何与 **MLX** 和 **GGUF** 模型格式协作的查询。
   - 人们想知道这两种格式是否可以在 **LLM Studio** 中共存，担心功能重叠和速度问题。
- **工具策略：LM Studio 迈出新步伐**：用户发现他们可以将 **LM Studio** 连接到外部函数和 API，参考了 [Tool Use - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/tool-use)。
   - 社区成员强调，通过 REST API 可以进行外部调用，为扩展 LLM 任务开辟了新途径。
- **对 RTX 5090 的提升感到失望**：尽管拥有 **1.7x** 带宽和 **32GB VRAM**，但 **NVIDIA RTX 5090** 仅比 **4090** 提升了 **30%**，这让发烧友们感到失望。
   - 他们质疑对较小模型的实际益处，指出增加 VRAM 并不总是能带来巨大的速度提升。
- **Llama-3.3 极度消耗 VRAM**：运行 **Llama-3.3-70B-Instruct-GGUF** 很快就需要至少 **双 A6000**（**96GB VRAM**），特别是为了保持性能。
   - 参与者指出，超过 **24GB** 的模型可能会让 **4090** 等消费级 GPU 不堪重负，限制了大型任务的速度。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **R1+Sonnet 夺得 SOTA**：**R1+Sonnet** 的组合在 aider 多语言基准测试中获得了 **64%** 的分数，成本比 **o1** 低 **14 倍**，如[这篇官方文章](https://aider.chat/2025/01/24/r1-sonnet.html)所示。
   - 这一结果引起了关于成本效益的热议，许多人称赞 **R1** 与 **Sonnet** 在处理稳健任务时的出色配合。
- **DeepSeek 的疑虑与支付痛苦**：由于支付安全问题，人们对 **DeepSeek** 的 API 产生了担忧，激发了对 [OpenRouter 版 R1 Distill Llama 70B](https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b) 的兴趣。
   - 一些人引用了在处理支付提供商时的“不确定信任度”，并参考了关于 **NVIDIA H100** 分配和模型托管限制的推文。
- **Aider 基准测试与聪明的 “Thinking Tokens”**：社区测试显示，与标准的以编辑器为中心的方法相比，**thinking tokens** 会降低基准测试性能，影响 **Chain of Thought** 的效力。
   - 参与者得出结论，重复使用旧的 CoT 会损害准确性，建议在高级任务中“修剪历史推理以获得最佳结果”。
- **为更精简的 Python 实现日志跨越**：一位用户建议通过 **logging module** 导出日志，并将输出存储在**只读**文件中，以减少多余的控制台内容。
   - 他们称赞这是“一个保持上下文整洁并专注于代码的小技巧”，只需在 prompt 中引用日志文件，而不是直接转储原始消息。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sky-Flash 解决过度思考问题**：NovaSkyAI 推出了 [Sky-T1-32B-Flash](https://novasky-ai.github.io/posts/reduce-overthinking/)，在不损失准确性的情况下将冗长的生成内容减少了 **50%**，据称成本仅为 **$275**。
   - 他们还发布了[模型权重](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Flash)供开放实验，承诺降低推理开销。
- **DeepSeek R1 撼动顶级模型地位**：据 [lmarena_ai](https://x.com/lmarena_ai/status/1882749951924715578) 报道，DeepSeek R1 在 Arena 排名跃升至 **第 3 位**，性能追平 **o1**，而成本却便宜了 **20 倍**。
   - 它在某些任务中甚至超越了 **o1-pro**，引发了关于其隐藏实力和参与基准测试时机的讨论。
- **总统 AI 行政命令引发行业大洗牌**：一项新签署的指令旨在针对阻碍 **美国 AI 领导地位** 的监管规定，废除了 **第 14110 号行政命令**，并倡导一种意识形态中立的方法。
   - 正如[官方公告](https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/)所述，该指令设立了 **AI 与加密货币特别顾问**，推动自由市场立场并强化国家安全。
- **天价薪资引发人才争夺战**：传闻 **DeepSeek** 员工的年薪包高达 **$550 万**，引发了对 AI 领域人才挖角的担忧。
   - 这些报价改变了权力动态，老牌技术巨头被认为决心通过丰厚的薪酬来削弱竞争对手。
- **Adobe 'Enhance Speech' 功能在音频爱好者中引发争议**：**Adobe Podcast ‘Enhance Speech’** 功能在多人播客中听起来可能有机械感，但在单人录音中表现较好。
   - 用户仍然更青睐稳固的麦克风设置而非“魔法音频”处理，认为自然的声音优于过滤后的清晰度。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell 架构随 CUDA 12.8 势头强劲**：NVIDIA 推出了支持 **Blackwell** 架构的 [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads)，包括 **cuBLAS** 中的 FP8 和 FP4 类型。文档强调了 sm_120 中缺失的第 5 代 **TensorCore** 指令，引发了对代码碎片化的担忧。
   - 成员们讨论了 **sm_90a** 和 **sm_100a** 之间的前向兼容性，指出 `wgmma` 是特定架构独有的。一份[迁移指南](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)提供了关于这些硬件过渡的见解。
- **ComfyUI 招聘 ML 工程师，计划举办旧金山见面会**：ComfyUI 宣布了 [ML 职位空缺](https://comfyorg.notion.site/Founding-Machine-Learning-Engineer-1696d73d36508014bfbaf5aebf39b145)，为各种主流模型提供首日支持。他们是一家位于湾区的风投支持公司，正在寻找优化开源工具的开发者。
   - 他们还透露将在 [GitHub 举办旧金山见面会](https://lu.ma/6skuqn7c?tk=xiHyMZ)，届时将有 **MJM** 和 **Lovis** 带来的演示和小组讨论。该活动鼓励参与者分享 **ComfyUI** 工作流并建立更紧密的联系。
- **再蒸馏版 DeepSeek R1 超越原版**：从原始版本再蒸馏的 **DeepSeek R1** 1.5B 模型表现出更好的性能，目前托管在 [Hugging Face](https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0) 上。**Mobius Labs** 指出，他们计划在不久的将来再蒸馏更多模型。
   - 来自 [Mobius Labs 的推文](https://x.com/Mobius_Labs/status/1882841665427390858)证实了其相对于之前版本的改进。社区讨论强调了涉及 Qwen 架构的可能扩展。
- **Flash Infer 与代码生成进展**：由 **Zihao Ye** 主讲的年度首场 **Flash Infer** 讲座展示了[代码生成](https://www.youtube.com/@GPUMODE)和用于增强算子（kernel）性能的专用 Attention 模式。JIT 和 AOT 编译成为实时加速的核心。
   - 参与者通过志愿者转达问题，克服了问答环节的限制，彰显了社区支持。这种开放式讨论激发了将这些方法与 HPC 驱动的工作流相结合的兴趣。
- **Arc-AGI 的迷宫与多项式插件**：贡献者在 [clrs 库](https://github.com/google-deepmind/clrs/tree/master/clrs/_src/clrs_text)中增加了多项式方程以及线性方程，以增加谜题的多样性。他们还提出了一个用于最短路径逻辑的迷宫任务，该任务在 reasoning-gym 中立即获得批准。
   - 计划包括清理库结构并添加静态数据集注册以简化使用。还讨论了一种动态奖励机制，允许用户定义自定义的基于准确性的评分公式。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas 获得代码支持与 MacOS 势头**：Canvas 现已集成 **OpenAI O1**，并能渲染 **HTML** 和 **React 代码**，可通过模型选择器或 `/canvas` 命令访问；该功能已在 **ChatGPT macOS 桌面应用**上向 Pro、Plus、Team 和 Free 用户全面推出。
   - 计划在几周内向 **Enterprise** 和 **Edu** 层级进行更广泛的发布，确保更多用户群体获得高级代码渲染能力。
- **Deepseek 的 R1 热潮：备用 GPU 与国家支持**：**Deepseek** 的 CEO 透露 **R1** 是作为一个侧面项目在备用 GPU 上构建的，引发了社区兴趣；有人声称它得到了中国政府的支持以增强本土 AI 模型，并引用了 [DeepSeek_R1.pdf](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)。
   - 这种方法既节省成本又引人注目，助长了关于 AI 倡议获得主权支持的讨论。
- **Chatbot API：海量 Token，极低账单**：一位用户建议在 **GPT-3.5** 上花费 **$5** 即可处理约 **250 万个 Token**，强调了与每月专业计划相比，自定义聊天机器人有更便宜的替代方案。
   - 他们还指出 **AI agent** 在 **Unity** 或集成 IDE 等应用中扩展的潜力，从而拓宽工作流效率。
- **Operator 的浏览器技巧预示未来**：**Operator** 引入了面向浏览器的功能，引发了对其在网页交互之外更广泛功能的关注。
   - 成员们推动将其更深入地集成到独立应用程序中，同时权衡授予 AI 互联网访问权限时对上下文保留的影响。
- **O3：发布还是抵制**：一位用户敦促立即推出 **O3**，却遭到了另一位用户简短的“不，谢谢”回应，显示出热情的两极分化。
   - 支持者将 **O3** 视为关键里程碑，而其他人则表现出极小的兴趣，展示了社区中多样的立场。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **React 狂想曲与 Tailwind 变奏**：一份关于 **React + TypeScript + Tailwind** Web 应用的结构化计划被列出，在 [Google Document](https://docs.google.com/document/d/1JYIUQjqVchaWGQSDNsBs7NpXdexOiXr4B79cOpprdxY/edit) 中详细说明了架构、数据处理和开发步骤。
   - 贡献者建议建立一个核心的 **GUIDELINES.md** 文件，并强调使用 **'Keep a Changelog'** 格式来有效跟踪版本更新。
- **Supabase 在聊天系统中的障碍**：一位用户在使用 Supabase 的实时钩子（realtime hooks）处理**消息系统**挑战时，遇到了 **Row Level Security** 问题，并寻求同行见解。
   - 他们强调了多用户协作的潜在陷阱，并希望克服过类似障碍的人能分享“经验教训”。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DiStRo 提升 GPU 速度**：对话围绕用于提升多 GPU 性能的 [DiStRo](https://link.to.DiStRo) 展开，重点关注适配每个 GPU 显存的模型。
   - 他们建议与 PyTorch 的 FSDP2 等框架协同，从而为先进架构实现更快的训练。
- **Tiny Tales：Token 微调的胜利**：参与者考虑了 [Tiny Stories](https://link.to.tinystories) 的性能，重点是通过精炼的 Tokenization 策略将参数从 500 万扩展到 7 亿。
   - Real Azure 发现通过调整 Token 使用改善了困惑度（perplexity），凸显了模型流水线未来的收益。
- **OpenAI：炒作 vs 光环**：成员们对比了微软、Meta 和亚马逊的估值，对 **OpenAI** 的品牌轨迹表示担忧。
   - 他们辩论了炒作与实际性能的关系，警告过度宣传可能会掩盖稳定的产品输出。
- **DeepSeek 蒸馏 Hermes，夺取 SOTA**：来自 Teknium1 的 [DeepSeek R1 蒸馏](https://fxtwitter.com/Teknium1/status/1882893748742598669) 改进将推理与通用模型结合，提升了结果。
   - 与此同时，[Paul Gauthier](https://x.com/paulgauthier/status/1882833360567095682) 透露 R1+Sonnet 在 aider 多语言基准测试中以 14 倍更低的成本飙升至新的 SOTA。
- **Self-Attention 获得优先权**：参与者强调了 Self-Attention 在大型 Transformer 模型中对 VRAM 效率的核心作用。
   - 他们还考虑通过自我蒸馏（self-distillation）来奖励创意输出，暗示了替代的训练角度。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **内存总线大乱斗与数学恶作剧**：在 #[general] 频道中，**512-bit 宽的 32GB 内存**引发了对钱包宽度的调侃，而一个 [Stack Exchange 谜题](https://math.stackexchange.com/questions/4756401/minimum-number-of-straight-lines-to-cover-n-times-n-grid#:~:text=The%20minimal%20number%20must%20be...) 难倒了多个 **LLMs**。
   - 社区成员还强调了*视觉推理陷阱*，并开玩笑说**动漫头像**代表了开源 ML 领域的顶尖开发者。
- **MONA 减少多步失误**：[MONA 论文](https://arxiv.org/abs/2501.13011) 引入了 **Myopic Optimization with Non-myopic Approval**（近视优化与非近视批准），作为一种抑制 **RL** 中多步奖励黑客攻击（reward hacking）的策略。
   - 作者描述了如何桥接短视优化与*远见奖励*，引发了关于对齐陷阱以及在标准 **RL** 参数之外极小额外开销的热烈讨论。
- **AG2 脱离微软后的转型**：[AG2 的新愿景](https://medium.com/@ag2ai/ag2s-vision-for-community-driven-agent-development-f9f3ca2b0dc8) 专注于**社区驱动的 Agent**，详细阐述了*治理模型*，并在脱离微软后推动全面**开源**。
   - 他们现在拥有超过 **20,000 名构建者**，引发了对更易用的 AI Agent 系统的期待，*部分用户*赞扬了向*众包开发*的转变。
- **R1 在 LMArena 中引起轰动**：据 #[paper-discussion] 频道报道，**R1** 在 **LMArena** 中获得*第 3 名*，掩盖了其他服务器的光芒，而 **Style Control** 保持第一。
   - 有人称 R1 是搅动市场的黑马，并提到与 **Stargate** 或 **B200** 服务器的协同作用可能是其表现强劲的原因。
- **Differential Transformers 与 AI 保险讨论**：开发者们关注了 [DifferentialTransformer 仓库](https://github.com/kyegomez/DifferentialTransformer)，但对其开源权重的质量和作者的方法表示*怀疑*。
   - 与此同时，关于 **AI 保险** 的闲谈浮出水面，有人开玩笑说“什么都有保险”，而另一些人则质疑它是否能处理*强化学习失控*导致的惨剧。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 快速完成播客编辑**：一位用户使用 **NotebookLM** 以极少的剪辑拼接播客音频，实现了近乎无缝的流畅效果。
   - 讨论中的听众赞赏该工具的准确性，并询问是否可以集成更复杂的音频片段以加快制作速度。
- **工程师关注逆向图灵测试视角**：一位用户描述了 **Generative Output** 尝试通过逆向图灵测试的想法来探索 **AGI** 概念。
   - 他们分享说，这种反思引发了关于**控制论（cybernetic）**进展以及 LLMs 如何理解自身的对话。
- **MasterZap 为 AI 虚拟形象制作动画**：MasterZap 解释了使用 **HailouAI** 和 **RunWayML** 创建逼真主持人的工作流，并引用了 [UNREAL MYSTERIES 7: The Callisto Mining Incident](https://www.youtube.com/watch?v=TfPy5oJQn_s)。
   - 他强调了让 **avatars** 感觉自然非常困难，促使其他人比较分层方法以实现平滑的面部动作。
- **Gemini Advanced 在 18.5MB PDF 上翻车**：用户测试了 **Gemini Advanced** 解析大型文档的能力，包括一份约 18.5MB 的**税法**文件，但成功率有限。
   - 参与者对**不准确**的法律定义表示沮丧，并指出 **1.5 Pro** 版本需要改进处理能力。
- **NotebookLM 攻克 220 道测验题**：一位用户要求 **NotebookLM** 从包含 **220** 道题目的 PDF 中生成测验，强调准确的文本提取。
   - 一些成员提供了协作建议，指出高级模型可以处理此类任务，但仍可能需要精细的提示词（prompts）。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.7.0 带来 Windows ARM 奇迹**：Nomic.ai 发布了 **GPT4All v3.7.0**，支持 **Windows ARM**，修复了 **macOS** 崩溃问题，并重构了 **Code Interpreter** 以实现更广泛的设备兼容性。
   - 一位用户报告称，*Snapdragon* 或 *SQ 处理器* 机器现在运行 GPT4All 更加流畅，引发了对改进后的 **chat templating** 功能的好奇。
- **Code Interpreter 与 Chat Templating 的胜利**：**Code Interpreter** 在 console.log 中支持 **多个参数**，并具有更好的超时处理，提高了与 JavaScript 工作流的兼容性。
   - 社区反馈认为这些调整提升了编码效率，而升级后的 **chat templating** 系统解决了 **EM German Mistral** 的崩溃问题。
- **礼貌的提示词（Prompt Politeness）见成效**：参与者解决了关于 NSFW 和微妙请求的 **prompt engineering** 障碍，发现添加 *please*（请）通常会带来更好的交互。
   - 他们强调许多 **LLM** 依赖于 **internet-trained data** 来获取上下文，并对微妙的措辞变化做出不同的反应。
- **模型混搭与 Qwen 查询**：用户评估了运行 **Llama-3.2** 和 **Qwen-2.5** 的 **GPT4All**，关注大规模任务的资源需求。
   - 有人提到 **Nous Hermes** 是一个可能的替代方案，而其他人则对 **Qwen** 进行了广泛测试，以增强翻译能力。
- **Taggui 助力图像分析**：一位用户寻求用于 **图像分类** 和打标的开源工具，促使推荐使用 **Taggui** 进行 AI 驱动的上传和查询。
   - 爱好者们称赞其 *多 AI 引擎* 集成，称其为高级图像辅助头脑风暴的可靠选择。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 超时调整成功**：工程师通过修改 `mcphub.ts` 解决了 **60 秒服务器超时** 问题，示例更新可在 [Roo-Code repo](https://github.com/qpd-v/Roo-Code) 中找到。他们归功于 **VS Code extension** 引导了修复并确保了稳定的 **MCP** 响应。
   - 成员们指出，指定正确的 `uvx.exe` 路径对于防止进一步停机至关重要，并强调 **Roo-Code** 是跟踪配置更改的有价值工具。
- **MySQL 与 MCP Alchemy 融合**：一位用户推荐了 [mcp-alchemy](https://github.com/runekaagaard/mcp-alchemy) 用于 **MySQL** 集成，并提到它也兼容 **SQLite** 和 **PostgreSQL**。这是在询问用于管理数据库连接的可靠 **MCP** 服务器之后提出的。
   - 该仓库包含多个使用示例，引发了对高级 **MCP** 数据库流水线的广泛兴趣。
- **Claude 的 Google 搜索功能受挫**：社区成员观察到 **Claude** 在其 **Google search** 功能上表现挣扎，有时在高负载下失败。他们推测高需求和 **API** 不稳定性可能是原因。
   - 一些人建议使用替代调度策略，希望更稳定的查询窗口能减少 **Claude** 中的搜索取消。
- **Glama 中的 Agent 引起困惑**：**MCP Agentic tool** 提前出现在 **Glama** 中，让用户对其激活路径感到不确定。一位成员透露官方声明尚待发布，称这次泄露是意料之外的。
   - 讨论将该功能与 **MCP.run** 联系起来，一些用户在 **non-Glama** 客户端设置中测试它以实现类似 **Agent** 的功能。
- **个性化初具规模：mcp-variance-log**：开发者介绍了 [mcp-variance-log](https://github.com/truaxki/mcp-variance-log)，引用了 ***Titans Surprise 机制***，通过 **SQLite** 进行用户特定数据跟踪。该工具分析对话结构以支持扩展记忆。
   - 成员们期待更深层次的个性化，指出这些 **variance logs** 可能会为未来的 **MCP** 扩展和针对用户的改进提供参考。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Redis 研讨会：AI Agent 实战**：与 **Redis** 联合举办的网络研讨会探讨了构建 AI Agent 以增强**任务管理**的方法，录像可在[此处](https://t.co/bELHnKjRm3)观看。
   - 听众指出，彻底分解任务可以提高实际应用中的**性能**。
- **驯服并行 LLM 流式传输**：一位用户在同时从多个 LLM 流式传输数据时遇到困难，建议指向了异步库配置错误，并参考了一个 [Google Colab 链接](https://colab.research.google.com/drive/1uGbrX5yAt0CMeUeOa4JaIUfWsmPp2tCS)中的工作示例。
   - 社区成员强调需要正确的并发模式，以避免**顺序数据处理**中的中断。
- **使用 LlamaParse 解析幻灯片**：工程师们讨论了使用 [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) 对 .pdf 和 .pptx 文件进行**文档解析**的方法，重点在于处理**基于 LaTeX 的 PDF**。
   - 他们确认了 LlamaParse 在为高级 **RAG** 工作流提取结构化文本方面的实用性，即使是跨多种文件类型。
- **ReActAgent 中的实时响应**：对话探讨了如何结合 LlamaIndex 的 ReActAgent 和[此处](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/)的 **AgentWorkflow** 系统，将**实时事件流**与 Token 输出集成。
   - 开发者表示，一旦事件处理与实时的 **Token 流式传输**同步，用户流程将得到改善。
- **出口管制波及 AI 领域**：参与者探讨了针对先进计算项目和 AI 模型权重的新**美国出口法规**的影响，引用了[此更新](https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights)。
   - 他们提出了关于合规障碍以及这些规则可能如何影响 **Llama 模型**使用和共享的问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **出口意外：模型权重成为目标**：美国商务部推出了新的 AI 出口管制，引发了关于 **Cohere 模型**是否会被纳入最新限制范围的担忧（[链接](https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights-seven-key-takeaways)）。
   - **Oracle Japan** 的一些工程师担心许可证纠纷，尽管内部团队暗示特殊协议可能会缓冲这一冲击。
- **GPU 趣闻：绕过限制**：社区成员辩论了 **GPU 限制**的实际影响，暗示大型 AI 公司正在暗中引入硬件。
   - 参与者质疑这些政策是否主要惩罚小玩家，而“大鱼”则能自由游走。
- **Blackwell 预算困扰**：一位用户指出了 **Blackwell** 的高负荷运行问题，提到待机功耗达 **200w**，引发了对使用量增加后账单激增的担忧。
   - 其他人建议平衡计算需求与实际工作负载以避免浪费。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 异步论坛盛会**：一位成员计划分享一篇关于 **Mojo 异步代码**的新论坛帖子，链接至 [如何在 Mojo 中编写异步代码](https://forum.modular.com/t/how-to-write-async-code-in-mojo/473)。
   - 他们承诺将协作完成该帖子，强调围绕*异步实践*进行**社区驱动**的知识交流。
- **MAX Builds 页面展示社区作品**：经过重新设计的 **MAX Builds 页面** [builds.modular.com](https://builds.modular.com) 现在设有一个社区构建包的专门板块。
   - 开发者可以向 [Modular 社区仓库](https://github.com/modular/modular-community)提交带有 **recipe.yaml** 的 PR，鼓励更多开放贡献。
- **iAdd 特性引发原地加法难题**：用户讨论了用于**原地加法**（如 **a += 1**）的 **__iadd__** 方法，以及在评估过程中值是如何存储的。
   - 一个有趣的例子 **a += a + 1** 证明了如果 **a** 最初为 **6**，结果可能会产生 **13**，这提醒人们要避免混淆。
- **Mojo CLI 进阶**：一位成员透露了两个新的 **Mojo CLI 标志**：**--ei** 和 **--v**，引起了频道参与者的兴趣。
   - 他们用俏皮的表情符号展示了这些标志，暗示 **Mojo** 爱好者还有更多的实验空间。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **追求更清晰的标签**：成员们探讨了针对**背景噪音（background noise）**和**音乐水平（music levels）**的标注策略，参考了一个 [Google Colab notebook](https://colab.research.google.com/drive/140lGFiXXeTsNFp7w5xteCjpmRRaSBvmj)，并鼓励采取*“不知道就发挥创意，尽管提问”*的方法。
   - 他们提出了多个类别，如**无背景噪音**和**轻微噪音**，一些人建议使用更动态的标注方式来处理不同的音乐强度。
- **众声喧哗**：爱好者们提出了一个**多发言者转录数据集**的想法，通过重叠 TTS 音频流，旨在获取细粒度的时间码来追踪谁在什么时候说话。
   - 他们强调，**音高（pitch）**和**混响（reverb）**的变化有助于**发言者识别（speaker recognition）**，并引用了一句话：*“没有网站，是我在 Discord 上协调大家。”*

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **散度蒸馏：Teacher-Student 之争**：一位参与者提出将 **Teacher** 和 **Student** 模型之间的散度作为奖励信号，用于在蒸馏中部署 PPO，这与传统的基于 KL 的方法形成对比。
   - 一些人指出 **KL-matching** 在常规训练中的稳定性，但对自适应散度塑造蒸馏奖励保持好奇。
- **层收敛 vs. 梯度消失**：人们讨论了最近一篇关于**层收敛偏差（layer convergence bias）**的 [ICLR 2023 论文](https://openreview.net/forum?id=wlMDF1jQF86)，该论文显示浅层比深层学习得更快。
   - 他们还辩论了**梯度消失（vanishing gradients）**是否是深层进度缓慢的一个因素，并承认它不是训练挑战中的唯一诱因。
- **ModernBERT, ModernBART 与混合架构热潮**：讨论集中在 **ModernBERT** 以及 **ModernBART** 的可能性上，一些人认为 Encoder-Decoder 版本在摘要生成任务中会很受欢迎。
   - 对 [GPT-BERT 混合模型](https://arxiv.org/abs/2410.24159) 的引用强调了其在 BabyLM Challenge 2024 中的性能提升，建议结合 Masked 和 Causal 方法。
- **Chain-of-Thought 与 Agent-R 激发反思**：一种新方法将 **Chain-of-Thought 推理**与 [Direct Preference Optimization](https://arxiv.org/abs/2501.13926) 相结合，以实现更好的自回归图像生成。
   - 与此同时，[**Agent-R** 框架](https://arxiv.org/abs/2501.11425) 利用 **MCTS** 进行自我批判和鲁棒恢复，引发了关于类似于 **Latro RL** 的反思性推理的辩论。
- **双线性 MLP 实现更清晰的计算**：一篇新的 [ICLR'25 论文](https://arxiv.org/abs/2410.08417) 介绍了**双线性 MLP（bilinear MLPs）**，它移除了逐元素的非线性，简化了层功能的可解释性。
   - 支持者认为这种设计揭示了**权重（weights）**如何驱动计算，为在复杂模型中实现更直接的 **Mech Interp** 带来了希望。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **聚焦冰霜字体**：爱好者们探索了使用 **Img2img** 配合自定义字体生成**冰霜文本（ice text）**，以实现晶莹剔透的设计。
   - 调整 **Denoise** 设置并将文本着色为冰蓝色被认为是推荐的方法。
- **ControlNet 受到关注**：一些人建议使用 **ControlNet** 来优化冰霜文本的外观，特别是结合分辨率平铺（resolution tiling）时。
   - 据说这种方法可以为风格化文本带来更锐利的边缘和更一致的效果。
- **Adobe Firefly 介入**：用户提到 **Adobe Firefly** 作为一个替代方案，如果有 Adobe 许可证，它可以处理专门的文本创建。
   - 他们认为这比分层使用 Inkscape 等独立软件工具的方法更快。
- **中毒图像查询**：一位成员询问如何检测图像是否**中毒（poisoned）**，引发了关于“舔舐测试”和“嗅闻测试”的玩笑。
   - 虽然没有出现正式的方法，但讨论凸显了社区对图像安全性的好奇。
- **将草图转化为场景**：有人寻求关于 **Sketch to Image** 工作流的建议，以将粗略的轮廓转化为最终的视觉效果。
   - 讨论中还提到了宽高比考虑因素（如 **16:9**），以实现更用户友好的生成。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ILP 驯服合并**：[Pull Request #8736](https://github.com/tinygrad/tinygrad/pull/8736) 中引入了一种使用 **ILP** 统一 view 对的新方法，报告的未命中率为 **0.45%**。
   - 参与者讨论了逻辑除数，并认识到可变步长（strides）带来的障碍。
- **Mask 混淆合并**：社区成员争论 **mask 表示** 是否能增强合并，尽管有人认为它可能不适用于所有设置。
   - 他们得出结论，mask 能够实现少量合并，但无法处理所有步长场景。
- **多重合并模式出现**：通过测试 **v1=[2,4]** 和 **v2=[3,6]** 的偏移量，提出了一种正式搜索，旨在检测公约数中的模式。
   - 他们设想了一种通用技术，通过系统地检查步长重叠来统一 view。
- **三个 View，双倍麻烦**：爱好者们质疑将合并从 **两个** 增加到 **三个** view，担心会增加复杂性。
   - 他们提到棘手的步长是 **ILP** 的绊脚石，并警告 3 -> 2 的 view 转换不会一帆风顺。
- **步长对齐势头强劲**：一些人建议对齐步长可以减少合并难题，但他们警告说，当步长不匹配时，可能会出现错误的假设。
   - 他们意识到早期的方法由于错误的步长计算而忽略了可能的合并，呼吁进行更深入的检查。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Windows 的不稳与 WSL 的奇妙**：成员们指出 **Windows** 施加了一些限制，包括有限的 **Triton** 内核支持。他们建议使用 **WSL** 以获得更直接的编码体验。
   - 他们强调了更短的设置时间，并建议这将减轻在 Windows 硬件上的训练性能压力。
- **正则表达式助力数据清理**：一位成员分享了一个正则表达式 `[^\\t\\n\\x20-\\x7E]+`，通过识别隐藏字符来清理杂乱的数据集。另一位成员阐明了该模式的组成部分，强调了其在剔除不可打印文本方面的作用。
   - 他们敦促在修改表达式时要谨慎，以避免意外的数据丢失，并建议在较小的样本上进行彻底测试。
- **Triton & Xformers 与 Windows 的冲突**：由于 **Triton** 和 **Xformers** 的兼容性缺口，一些人在 Windows 上运行 **unsloth** 或 **axolotl** 时遇到了问题。他们指向了一个 GitHub 仓库以寻求潜在的解决方案。
   - 他们建议探索未来的驱动程序更新或基于容器的方法来在 Windows 上安装这些库。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 周一狂欢开始**：正如 [#mooc-questions 频道讨论](https://discord.com/channels/1280234300012494859/1280370030609170494/1332102115090239570) 中提到的，第一节课定于 **太平洋时间 27 日周一下午 4 点**，官方邮件通知即将发布。
   - 组织者确认该课程将包含量身定制的高级内容，旨在推动 **LLM Agent** 在现实任务中的应用。
- **LLM Agent 面临高标准**：社区成员指出，通过该课程的 **LLM Agent** 为这些模型的能力设定了更高的标准。
   - 他们一致认为，这反映了课程繁重的工作量和严格的评分标准，为 AI 参与者打造了一个独特的挑战。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **诈骗信息席卷 Discord**：诈骗信息出现在多个频道中，包括 [此频道](https://discord.com/channels/1104757954588196865/1110594519226925137/1332219195210989578)，引发了警告。
   - 一位用户承认了这个问题，鼓励大家保持警惕并核实可疑帖子。
- **Nebius AI 引发多节点关注**：一位成员询问在多节点环境中运行 **Nebius AI** 的经验，理由是需要现实世界的性能技巧。
   - 其他人提供了潜在的指导，强调了对资源分配和设置细节的深入了解需求。
- **SLURM 与 Torch Elastic 的正面交锋**：一位用户感叹在分布式训练中配置 **SLURM** 与 **Torch Elastic** 的难度，称其为一个重大障碍。
   - 另一位成员建议查看 SLURM 的多节点文档，认为许多设置概念可能仍然适用于这种情况。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Signature 混乱加速了困惑**：一位用户在 **signature** 定义上遇到了困惑，理由是缺失的源文件导致输出偏离预期。
   - 他们怀疑是不完整的引用导致了这次混乱，并提出了如何管理一致的文件映射的问题。
- **MATH Dataset 消失了**：一位用户尝试运行 **maths example**，但发现 **MATH dataset** 已从 Hugging Face 移除，并分享了[一个部分副本的链接](https://huggingface.co/datasets/lighteval/MATH/viewer)。
   - 他们还指向了 [Heuristics Math dataset](https://huggingface.co/datasets/hendrycks/competition_math)，询问是否有人能建议其他解决方案。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

**OpenInterpreter Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持安静，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1332082447520763916)** (604 条消息🔥🔥🔥): 

> `DeepSeek R1 vs. 其他模型, Cursor 功能与问题, AI 模型与生产力, Cursor 0.45.2 版本更新, 关于 AI 编程的综合讨论` 

- **DeepSeek R1 展现出作为架构师的潜力**：DeepSeek R1 与作为编辑器的 Sonnet 结合使用，在 Aider polyglot benchmark 上达到了 64% 的新 state-of-the-art，超越了之前的模型。
   - 据报道，这种组合与早期模型相比显著降低了成本，引发了关于在编程工作流中使用 R1 优势的讨论。
- **Cursor 的体验及其可靠性**：用户对 Cursor 的评价褒贬不一，提到了意外的代码更改以及平台的问题，特别是在处理 ASP.Net 和 VB.Net 的大型项目时。
   - VSCode 和 Cursor 的 live share 功能的可靠性受到质疑，有评论称在协作项目时经常断开连接。
- **关于使用 AI 执行编程任务的讨论**：大家达成共识，虽然 AI 可以作为有用的编程助手，但仍需监督以确保代码质量并降低意外更改的风险。
   - 参与者讨论了使用 chat mode 的优势，以便在复杂项目中更好地控制编程过程。
- **Cursor 0.45.2 版本更新**：新版本 0.45.2 包含多项功能，但一些用户注意到之前可用的功能（如 embedding 的 'beta' 版）缺失了。
   - 有人推测某些功能可能因稳定性问题而暂停，同时用户强调最新版本在易用性方面有显著提升。
- **关于 AI 和编程格局的总体观察**：用户反思了不断演进的 AI 格局，注意到像 DeepSeek R1 这样的开源模型如何重塑行业并促使竞争对手做出反应。
   - 社区讨论了这些变化的影响，特别是关于 AI 编程助手和市场上竞争产品的未来。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://formulae.brew.sh/cask/cursor">cursor</a>: Homebrew 的软件包索引</li><li><a href="https://www.cursor.com/blog/tab-update">A New Tab Model | Cursor - AI 代码编辑器</a>: 发布下一代 Cursor Tab 模型。</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>: R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。成本比 o1 低 14 倍。</li><li><a href="https://tenor.com/view/kermit-anxiety-worried-worry-gif-10947879">Kermit 焦虑 GIF - Kermit Anxiety Worried - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/barron-trump-let-me-hear-it-show-off-trump-hand-to-ear-gif-12032203506919991642">Barron Trump Let Me Hear It GIF - Barron Trump Let me hear it Show off - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://forum.cursor.com/t/privacy-policy-of-deepseek/43727/2">DeepSeek 的隐私政策</a>: 嘿，我们在自己采购的基础设施上运行 DeepSeek，使用 Fireworks 作为我们的供应商。我们与他们签订了符合我们隐私和安全政策的现有协议，这不会改变...</li><li><a href="https://tenor.com/view/eddie-murphy-vacation-time-champagne-sunshine-lol-gif-12104233">Eddie Murphy 假期时间 GIF - Eddie Murphy Vacation Time Champagne - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://status.cursor.com/">Cursor 状态</a>: 未找到描述</li><li><a href="https://x.com/_aidan_clark_/status/1882135220738220131">Aidan Clark (@_aidan_clark_) 的推文</a>: o3-mini 第一次尝试，无需修改，耗时 20 秒（还教我如何转换为 GIF.....）太兴奋了 :) 引用 Ivan Fioravanti ᯅ (@ivanfioravanti) 👀 DeepSeek R1（右）击败了 o1-pro（左）👀 Prompt: &#34;wri...</li><li><a href="https://forum.cursor.com/t/share-your-rules-for-ai/2377/83">分享你的 &#34;Rules for AI&#34;</a>: 我现在有一堆针对不同设置/项目的自定义规则，通常与 Claude 3.5 (Sonnet) 或 gpt-4o 一起使用，有时我也会使用 “chatgpt-latest” 模型，但频率不高。总之...</li><li><a href="https://sdk.vercel.ai/docs/introduction">Vercel 的 AI SDK</a>: 欢迎查看 AI SDK 文档！</li><li><a href="https://www.trae.ai/">Trae - 使用 Trae 更快交付</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=by9PUlqtJlM&t=540s">OpenAI 感到恐惧（终于出现了一个伟大的开源 LLM）</a>: 我从未想过会有这一天。DeepSeek R1 正在横扫所有基准测试。而且它便宜得惊人。感谢 G2i 的赞助！请访问：https:/...</li><li><a href="https://huggingface.co/deepseek-ai">deepseek-ai (DeepSeek)</a>: 未找到描述</li><li><a href="https://downloader.cursor.sh/linux/appimage">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/cline/cline">GitHub - cline/cline: 直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器等，并在每一步都经过您的许可。</a>: 直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器等，并在每一步都经过您的许可。 - cline/cline</li><li><a href="https://github.com/PatrickJS/awesome-cursorrules">GitHub - PatrickJS/awesome-cursorrules: 📄 精选的 .cursorrules 文件列表</a>: 📄 精选的 .cursorrules 文件列表。通过在 GitHub 上创建账户为 PatrickJS/awesome-cursorrules 的开发做出贡献。</li><li><a href="https://www.high-flyer.cn/en/#index">High-Flyer | 首页</a>: 幻方 AI 专注前沿科技研发，以 AI 技术激发创造力和想象力，让人类更多梦想变成现实。幻方 AI 包含「萤火」深度学习训练平台、幻方量化（使用 AI 进行投资的对冲基金）、AI 基础科学研究。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1332089932671352956)** (347 条消息🔥🔥): 

> `针对语言准确性的 LLM 微调、Unsloth 在 Llama-Factory 中的集成、语言模型的持续预训练（Continued pretraining）、Posit 计算的影响、Evo 模型在核苷酸预测上的表现` 


- **使用 LoRA 增强土耳其语语言模型**：一位用户正在探索通过 LoRA 训练微调 Qwen 2.5 模型，以提高土耳其语语音准确性并减少语法错误。
   - 他们被引导至持续预训练（continued pretraining）资源以获取指导。
- **Unsloth 的 Llama-Factory 集成性能**：有报告称，与直接使用相比，Unsloth 在 Llama-Factory 中的集成性能较慢（慢了多达 3 倍）。
   - 用户赞赏 UI 带来的便利，但对集成速度表示担忧。
- **语言模型的持续预训练（Continued Pretraining）Notebook**：讨论了将持续预训练作为语言模型学习新知识领域或语言的一种方法，并提供了相关 Notebook 的链接。
   - 该方法强调了针对专门应用进行自适应训练的重要性。
- **Posit 计算的动态范围优势**：用户讨论了 Posit 计算的优势，特别是其动态尾数和指数大小，相比浮点格式提高了准确性。
   - 对话重点介绍了与 John Gustafson 博士的合作，以探索其在 AI 中的应用。
- **Evo 模型在核苷酸预测方面的成功**：在原核生物上训练的 Evo 模型展示了基于核苷酸输入向量的良好预测能力。
   - 它将每个核苷酸映射到一个稀疏向量，实现了比随机预测更高的平均正确率。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://sakana.ai/transformer-squared/">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/MrDragonFox/vtube">MrDragonFox/vtube · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://tenor.com/view/discord-this-server-is-powered-gif-21305371">Discord This GIF - Discord This Server - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4">Qwen 2.5 Coder - unsloth 收藏集</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/smolervlm">SmolVLM 变得更小了 – 推出 256M 和 500M 模型！</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#deepseek-r1-distill-models">deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 收藏集</a>: 未找到描述</li><li><a href="https://aptos.dev/en/build/get-started">入门指南 | Aptos 文档 (英文)</a>: Aptos 文档</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>: 又名持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个检查点微调 | Unsloth 文档</a>: 检查点（Checkpointing）允许你保存微调进度，以便暂停并继续。</li><li><a href="https://esolangs.org/wiki/Chicken">Chicken - Esolang</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1332077538478129216)** (29 条消息🔥): 

> `双 RTX 3090 配置，训练推理模型仓库，Dolphin-R1 数据集创建，vLLM 与 Open-webui 的兼容性` 


- **成员分享双 RTX 3090 配置方案**：一位成员分享了他们在自定义配置中运行双 RTX 3090 的方案，而另一位成员则转向使用 Corsair 7000D 机箱配合水冷 4090。
   - 他们讨论了额外的配置细节，包括硬件规格，如 ASUS ProArt X670E CREATOR 主板支持 GPU 的 x8/x8 PCIe 插槽。
- **训练推理模型的建议**：有建议称 TRL 框架包含训练推理模型的相关方法，一位成员分享了 [DeepSeekMath 论文](https://huggingface.co/papers/2402.03300) 的链接。
   - 讨论强调了当前对微调技术的需求，特别提到了 R1 distilled 模型的使用。
- **Dolphin-R1 数据集的创建**：讨论了为创建 Dolphin-R1 数据集支付的 6000 美元 API 费用，该数据集遵循 Deepseek-R1 distillation 的配方，目标是总计 800k 条轨迹（traces）。
   - 已找到赞助商，该数据集不久后将在 Hugging Face 上以 Apache 2.0 许可证发布。
- **vLLM 与 Open-webui 设置的问题**：有成员担心 vLLM 无法识别来自 Open-webui 的模型预设，特别是像 temperature 这样的参数未能正确传递。
   - 成员们辩论了这些问题的可能原因，有人认为 vLLM 不支持某些 sampler 设置。
- **分享编程幽默**：分享了一个名为“删除你的单元测试”的幽默 YouTube 视频，内容聚焦于编程梗（memes）。
   - 这展示了软件开发中所面临挑战的轻松一面，引起了成员间的互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cognitivecompai/status/1882132168153178606">Eric Hartford (@cognitivecompai) 的推文</a>：花费 6000 美元 API 费用创建 Dolphin-R1 数据集。我遵循 Deepseek-R1 distillation 配方，但使用 Dolphin 种子数据。（600k 推理数据，200k 对话数据，总计 800k）我想以 Apache 2.0 协议授权它...</li><li><a href="https://huggingface.co/docs/trl/main/en/grpo_trainer">GRPO Trainer</a>：未找到描述</li><li><a href="https://x.com/cognitivecompai/status/1882140705159799169">Eric Hartford (@cognitivecompai) 的推文</a>：我找到了赞助商！感谢 @driaforall！数据将在几天内以 Apache-2.0 许可证发布到 @huggingface。</li><li><a href="https://www.youtube.com/shorts/LNTBc8ryzEQ">Delete your unit tests</a>：#programming #hopelesscore
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1332084157282783312)** (107 条消息🔥🔥): 

> `微调模型，解决模型训练中的错误，利用聊天模板，从 Colab 导出模型，训练的数据集格式化` 


- **模型微调与错误**：用户讨论了微调各种模型（如 **qwen-2.5-3b** 和 **DeepSeek-R1-Distill**），经常遇到与数据集格式相关的特定错误。
   - 一位用户提到了在 **alpaca 模板** 中使用 **sharegpt 格式** 的挑战，需要替代模板才能正常训练。
- **聊天模板的使用经验**：关于不同聊天模板在训练模型中的有效性进行了讨论，重点关注推理过程可能受到的影响。
   - 一位用户评论了在实验 **R1 模型** 时，成功在数据集中集成了 **<think>** token。
- **从 Colab 导出模型**：用户表示在将以 **GGUF** 格式保存的微调模型从 Google Colab 导出到本地存储或其他格式时遇到困难。
   - 一位用户建议直接保存到 **Hugging Face**，这可能比从 Colab 导出更简单。
- **数据集格式化挑战**：关于训练模型的合适数据集格式展开了讨论，强调了不同模板所需结构之间的差异。
   - 有观点指出，使用标准模板可能会面临丢失模型推理能力的风险，这促使用户创建定制化的数据集。
- **量化格式的模型加载问题**：用户报告了加载量化模型（如 **unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit**）的问题，在尝试于 **MMLU-pro** 等基准测试上运行时收到错误。
   - 用户寻求支持以排查加载问题，并确保在本地运行模型的兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main">unsloth/DeepSeek-V3-GGUF at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1332360671446306877)** (9 messages🔥): 

> `用于 LLM 微调的 LoHan 框架、NVMe Offloading 技术、LLM 中的 Flash Memory 利用` 


- **LoHan 框架支持在消费级 GPU 上进行微调**：该论文介绍了 **LoHan**，这是一个旨在单块消费级 GPU 上高效微调 100B 规模 LLM 的框架，解决了聚合设备内存的高昂成本问题。
   - *现有方法失败* 的原因是服务器内张量移动管理不善，这使得 LoHan 成为预算有限的研究人员的重要贡献。[阅读更多](https://arxiv.org/abs/2403.06504)
- **关于相关 NVMe Offloading 技术的讨论**：一名成员引用了一篇关于 **NVMe Offloading** 的论文，认为它与 LoHan 框架中讨论的成本管理策略有关。
   - 尽管相关，但他们承认这种技术与 LoHan 优化 LLM 微调的方法并不相同。
- **用于管理 LLM 的 Flash 存储方法论**：另一篇论文研究了通过将模型参数存储在 **Flash Memory** 中并管理按需加载到 DRAM 来高效运行大语言模型的方法。
   - 论文提出了诸如 *windowing*（窗口化）等技术来减少数据传输，强调了优化高容量模型 Flash Memory 利用率的重要性。[阅读更多](https://arxiv.org/abs/2312.11514)
- **对 LLM 文献的共同兴趣**：成员们表达了分享 LLM 微调相关学术文献的兴趣，突显了讨论的协作性质。
   - 一位成员表示自己对文献并不熟悉，但希望通过介绍这些论文来激发大家的兴趣。
- **持续分享相关研究**：在讨论中，成员们为那些希望进一步探索 LLM 微调和优化技术的人分享了更多论文链接。
   - 针对实际场景中 LLM 部署挑战的研究仍然是该社区关注的核心领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2312.11514">LLM in a flash: Efficient Large Language Model Inference with Limited Memory</a>：大语言模型 (LLM) 是现代自然语言处理的核心，在各种任务中表现卓越。然而，它们巨大的计算和内存需求...</li><li><a href="https://arxiv.org/abs/2403.06504">LoHan: Low-Cost High-Performance Framework to Fine-Tune 100B Model on a Consumer GPU</a>：如今，AI 研究人员对微调预训练 LLM 越来越感兴趣，其规模已增长到超过 100B 参数。微调此类模型的一种方法是...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1332439967309369345)** (1 messages): 

> `Windsurf 1.2.2 发布、Cascade 网页搜索、延迟改进、内存系统增强` 


- **Windsurf 1.2.2 正式发布！**：**Windsurf 1.2.2** 更新已发布，重点通过关键增强功能提升用户体验。
   - 显著的改进包括更流畅的长对话处理、增强的网页搜索能力以及对 **Cascade 内存系统** 的增强。
- **Cascade 现在可以搜索网页了！**：此次更新为 **Cascade** 引入了新功能，允许其自动或通过用户提供的 URL 进行**网页搜索**。
   - 用户现在可以使用 **`@web`** 和 **`@docs`** 等命令触发搜索，进一步丰富了 Cascade 的功能。
- **网页和文档搜索设置**：用户可以直接从位于状态栏的 Windsurf **设置面板**启用或禁用网页搜索工具。
   - 这允许在进行网页搜索时获得可定制的体验，根据不同偏好增强可用性。



**提到的链接**：<a href="https://www.codeium.com/changelog">Windsurf Editor 更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。

  

---

### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1332122340938481755)** (1 messages): 

> `Web Search 功能，演示视频发布` 


- **新的 Web Search 功能备受关注**：团队宣布了令人兴奋的新 **web search 功能**，旨在增强用户浏览互联网时的体验。
   - 鼓励成员查看公告中链接的**酷炫演示视频**，该视频展示了其各项能力。
- **请求社区支持演示视频**：呼吁用户在社交媒体上对发布的**演示视频**表示支持。
   - 强调了 [X](https://x.com/windsurf_ai/status/1882561985621221451) 上的帖子需要社区的关注以提高曝光度。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1882561985621221451">来自 Windsurf (@windsurf_ai) 的推文</a>：正在网上冲浪！🏄

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1332123212682625055)** (78 messages🔥🔥): 

> `Windsurf 问题，Supercomplete 功能，账号注册问题，Windsurf 的 C# 扩展，Windsurf 1.2.2 发布` 


- **Windsurf 面临多个用户问题**：用户报告了**账号注册**、消息限制减少以及部分用户遇到账号禁用错误等问题，表明存在更广泛的系统问题。
   - 多个线程显示支持团队已知晓并正在处理这些问题，但许多用户仍在等待解决。
- **对 Supercomplete 功能的担忧**：几位用户询问了 **Supercomplete** 功能，特别是其相对于 Codeium 扩展和 VSCode 集成的状态。
   - 一位用户注意到缺乏更新，并质疑重心是否已完全转移到 Windsurf。
- **Windsurf 中 C# 开发的挑战**：用户对 **C# 扩展**表示沮丧，讨论了 Microsoft 开发工具包的限制及其与 Windsurf 的兼容性。
   - 社区成员建议使用 **open-vsx.org** 的 C# 扩展，同时警告调试配置可能存在问题。
- **Windsurf 1.2.2 发布支持更流畅的体验**：Windsurf 推出了更新，以增强长对话的性能并改进 web search 能力。
   - 引导用户查看 changelog 以了解新功能的详细信息，这表明开发工作仍在持续进行。
- **Windsurf 停机与恢复**：有报告指出 **Windsurf** 暂时宕机，引发了用户对系统稳定性的担忧。
   - 经过短暂的一段时间后，用户确认服务已恢复在线，缓解了一些紧迫的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/pricing">定价 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 对个人永久免费。团队可以通过我们的企业版产品进行升级，以获得增强的个性化和灵活的部署。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1332083554678472725)** (299 条消息🔥🔥): 

> `Windsurf 登录问题, Windsurf 更新与性能, Vite 中的 Open Graph 元数据, Windsurf 输入延迟, Cascade 服务中断` 


- **Windsurf 登录问题影响多名用户**：许多用户报告了登录 Windsurf 时的问题，包括“Something went wrong”错误和持续的 503 错误。
   - 讨论中提到了对账号滥用可能导致这些限制的担忧，用户无法在各种设备上注册或访问其账号。
- **Windsurf 1.2.2 版本发布但带有 Bug**：尽管 1.2.2 更新声称修复了长对话期间的输入延迟问题，但用户反映仍能感觉到明显的延迟。
   - 即使在重启和开启新对话后，延迟依然存在，这表明该更新并未解决所有用户的性能问题。
- **Vite 应用中的 Open Graph 元数据问题**：一位用户分享了在 Vercel 上为使用 Vite 构建的动态页面实现 Open Graph 元数据的困难，称 Windsurf 中的提示词未能解决该问题。
   - 在苦苦挣扎几天后，他们探索了 SSR 和客户端渲染元数据等各种策略，并寻求社区建议。
- **Cascade 服务中断**：有广泛报告称 503 网关错误影响了 Cascade，导致多名用户无法使用该服务。
   - 经过短时间的停机后，风波平息，功能恢复正常，用户对快速修复表示赞赏。
- **关于将 Windsurf 与其他编辑器或终端集成的讨论**：发起了关于在不同环境（如其他编辑器或终端）中使用 CodeSeek R1 或 Windsurf 可能性的对话。
   - 其他用户指出 Windsurf 是作为一个专用的 IDE 运行的，引发了关于与现有工具集成能力的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/memories">Cascade Memories</a>: 未找到描述</li><li><a href="https://sweetpad.hyzyla.dev/">Home | Sweetpad</a>: 描述将放入 &lt;head /&gt; 中的 meta 标签</li><li><a href="https://docs.codeium.com/">未找到标题</a>: 未找到描述</li><li><a href="https://codeium.com/live/general">与 Codeium 对话 | Windsurf 编辑器和 Codeium 扩展</a>: 使用 Codeium Live 进行常规聊天。Codeium 是深受开发者喜爱且受企业信赖的 AI 代码助手平台，也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/c">页面未找到 | Windsurf 编辑器和 Codeium 扩展</a>: Codeium 是深受开发者喜爱且受企业信赖的 AI 代码助手平台，也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/contact/enterprise">联系我们 | Windsurf 编辑器和 Codeium 扩展</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业级方案的信息。</li><li><a href="https://youtu.be/X7Ipaou43o0?si=JaawjpuHsd2R1d_v">自动将博客教程转化为全栈应用 - Windsurf 编辑器</a>: 你现在只需将博客文章的 URL 放入 IDE 即可构建全栈应用。通过网络搜索，Windsurf 可以自动遵循教程并转化...</li><li><a href="https://www.youtube.com/watch?v=hqJDKTqCESE">TikTok 做了一个 IDE 且效果不错？（免费的 Cursor 杀手？？）</a>: 字节跳动（TikTok 的母公司）做了一个代码编辑器，而且效果真的很好？！Cursor 要凉了？VS Code 杀手？Jetbrains 克隆版？我也不知道发生了什么...</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🚀🤖 Crawl4AI: 开源 LLM 友好型网页爬虫与抓取工具</a>: 🚀🤖 Crawl4AI: 开源 LLM 友好型网页爬虫与抓取工具 - unclecode/crawl4ai</li><li><a href="https://github.com/Cem-Bas/AgenTest-aiDoc">GitHub - Cem-Bas/AgenTest-aiDoc</a>: 为 Cem-Bas/AgenTest-aiDoc 的开发做出贡献。</li><li><a href="https://codeium.com/blog/pricing-windsurf">方案与定价更新</a>: Cascade 定价模型的一些变化。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1332212474514116688)** (3 条消息): 

> `DeepSeek R1 更新，DeepSeek 提供商故障` 


- **DeepSeek R1 扩展消息模式支持**：DeepSeek R1 现在支持更多类型的 **message patterns**，允许用户再次发送 **weird message orderings**（异常消息排序）。
   - 此次更新旨在提高用户的整体可用性和灵活性。
- **暂时降低 DeepSeek 提供商排名**：**DeepSeek 提供商**今天早上经历了奇怪的故障，导致在问题修复前被暂时降低排名（deranking）。
   - 用户已收到此更改通知，以便对服务可用性建立合理预期。
- **DeepSeek 提供商恢复在线**：在早些时候的故障后，**DeepSeek 提供商**已恢复并重新上线。
   - 这预示着依赖 DeepSeek 功能的用户将恢复正常服务。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1332079170708049961)** (290 条消息🔥🔥): 

> `DeepSeek R1 性能, Gemini API 访问, BlackboxAI 担忧, 速率限制与 Key 使用, OpenRouter 上的提供商问题` 


- **DeepSeek R1 性能提升**：用户报告称 DeepSeek R1 现在可以在 SillyTavern 上正常运行，并指出其写作质量非常出色且极具性价比。
   - 尽管早期存在问题，但许多用户现在对该模型的性能和价格优势印象深刻。
- **Gemini API 访问与限制**：关于 Gemini 模型访问的讨论中，一些用户建议使用个人 API keys 以绕过免费版本的速率限制。
   - 鼓励用户获取自己的 keys，以便有效地利用更高的速率并访问相关功能。
- **对 BlackboxAI 的担忧**：一场关于 BlackboxAI 合法性的讨论浮出水面，重点提到了其复杂的安装过程以及评论缺乏透明度。
   - 一些用户对其运营持怀疑态度，认为它可能是一个骗局或管理不善的服务。
- **速率限制与 API Key 管理**：有关于 API keys 相关速率限制的咨询，特别是它们是否会过期或被禁用。
   - 官方澄清 OpenRouter API keys 不会过期，但可以由用户手动禁用。
- **OpenRouter 上的提供商问题**：用户在 OpenRouter 上一直面临 DeepSeek 和其他提供商模型的问题，一些人认为 DeepSeek 的 API 权重可能与其他模型不同。
   - 这些差异引发了关于其对 Benchmark 结果和整体用户体验影响的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://platform.kluster.ai/">kluster.ai - 大规模 AI 动力</a>：以小规模成本实现大规模推理。彻底改变大规模推理的开发者平台。</li><li><a href="https://openrouter.ai/google/gemini-flash-1.5-exp)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys">集成 | OpenRouter</a>：在 OpenRouter 中使用您自己的提供商 keys。</li><li><a href="https://x.com/AnthropicAI/status/1882480450649915772">Anthropic (@AnthropicAI) 的推文</a>：推出 Citations（引用）。我们新的 API 功能允许 Claude 根据您提供的来源进行回答。Claude 随后可以引用支撑每个回答的具体句子和段落。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini-v2">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API, 提供商, 统计数据</a>：DeepSeek R1 Distill Llama 70B 是基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B。</li><li><a href="https://openrouter.ai/docs/web-search">联网搜索 | OpenRouter</a>：模型无关的 Grounding。</li><li><a href="https://openrouter.ai/docs/crypto-api">加密货币支付 API | OpenRouter</a>：与无需 UI 购买额度相关的 API。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-2.0-flash-thinking-mode">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, 提供商, 统计数据</a>：DeepSeek R1 发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的且具有完全开放的推理 token。其参数量为 671B，推理时激活参数为 37B。</li><li><a href="https://openrouter.ai/docs/requests">请求 | OpenRouter</a>：处理传入和传出请求。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1332085803870064750)** (75 messages🔥🔥): 

> `DeepSeek R1 Model, Fireworks Streaming Transcription Service, Braintrust AI Proxy, Perplexity Assistant, New OpenAI Features` 


- **DeepSeek R1 性能排名飙升**：DeepSeek R1 在 WebDev Arena 中跃升至第 2 位，在编程任务中展现出强大的能力，在性能匹配顶尖推理模型 o1 的同时，价格便宜了 **20 倍**。
   - 该模型在技术领域的表现获得了认可，并获得了 **MIT license**，被认为是社区的宝贵资源。
- **Fireworks 发布全新转录服务**：Fireworks 推出了一项新的流式转录服务，以 **300ms 延迟** 提供具有 Whisper-v3-large 质量的 **实时字幕 (live captions)**，且前两周免费。
   - 试用期结束后，该服务费用为 **每分钟音频 0.0032 美元**，提供了经济的转录解决方案。
- **Braintrust AI Proxy 简化 AI 集成**：Braintrust 推出了 AI Proxy，允许开发者通过单个 API 访问各种 AI 提供商，促进了代码简化和 **成本降低**。
   - 该工具是 **open-source** 的，可以轻松设置和管理 AI 模型，并具有日志记录和 prompt 管理等附加功能。
- **Perplexity Assistant 登陆 Play Store**：Perplexity AI 推出了 Perplexity Assistant，通过直观的界面提供简化日常任务的功能，如预订晚餐或起草电子邮件。
   - 有观点指出，该助手利用了 Apple 生态系统目前尚不具备的功能，引发了对 Apple 竞争优势的担忧。
- **OpenAI 更新 ChatGPT 中的 Canvas**：OpenAI 宣布更新 ChatGPT 中的 Canvas 功能，使其能够与 o1 模型配合使用，并支持 HTML 和 React 代码渲染。
   - 这些功能旨在增强用户交互以及在各种应用中使用 Canvas 的灵活性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/FireworksAI_HQ/status/1882530477468459309">来自 Fireworks AI (@FireworksAI_HQ) 的推文</a>：我们正在推出流式转录服务！以 300ms 的延迟生成具有 Whisper-v3-large 质量的实时字幕或驱动语音 Agent。未来两周免费使用，之后价格为 $0.0032...</li><li><a href="https://x.com/lmarena_ai/status/1882875989610594542">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：❤️‍🔥WebDev Arena 更新：令人兴奋的新条目！- #2: @deepseek_ai DeepSeek-R1 - #4: 新 Gemini-2.0-Flash-Thinking。DeepSeek-R1 跃升至第 2 位，与 Claude 3.5 Sonnet 的差距不到 40 分，展现了强大的能力...</li><li><a href="https://x.com/rauchg/status/1882636414480986182?s=46">来自 Guillermo Rauch (@rauchg) 的推文</a>：使用 @splinetool 的下一代模型实现图像转 3D。所有的实现障碍正在被消除，只需带上你的创意</li><li><a href="https://x.com/openai/status/1882129444212740482?s=46">来自 OpenAI (@OpenAI) 的推文</a>：用推理时计算（Inference-Time Compute）换取对抗鲁棒性 https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness/</li><li><a href="https://x.com/AnjneyMidha/status/1882669123492368586">来自 Anjney Midha 🇺🇸 (@AnjneyMidha) 的推文</a>：从斯坦福到麻省理工，DeepSeek-R1 基本上在一夜之间成为了美国顶尖大学研究人员的首选模型</li><li><a href="https://x.com/davlindner/status/1882451562859254050?s=46">来自 David Lindner (@davlindner) 的推文</a>：Google DeepMind 的新安全论文！LLM Agent 即将到来——我们如何阻止它们寻找复杂的计划来攻击奖励机制？我们的方法 MONA 可以防止许多此类攻击，*即使*人类无法检测到...</li><li><a href="https://x.com/teknium1/status/1882893748742598669?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：我们使用 5k 个 DeepSeek-R1 蒸馏的 CoT 重新训练了 Hermes。我可以确认几件事：1. 你可以拥有通用 + 推理模式，我们使用静态系统提示词标记了来自 R1 的所有长 CoT 样本...</li><li><a href="https://x.com/openai/status/1882876172339757392?s=46">来自 OpenAI (@OpenAI) 的推文</a>：Canvas 更新：今天我们正在 ChatGPT 中推出一些呼声很高的 Canvas 更新。✅Canvas 现在支持 OpenAI o1——从模型选择器中选择 o1，并使用工具箱图标或 “/canvas” 命令...</li><li><a href="https://x.com/jxmnop/status/1882849185319514295">来自 jack morris (@jxmnop) 的推文</a>：我想 DeepSeek 打破了众所周知的“四分钟一英里”障碍。人们过去认为这是不可能的。突然之间，语言模型上的 RL 奏效了，而且它可以在足够小的规模上复现...</li><li><a href="https://www.latent.space/p/gpu-bubble">$2 的 H100：GPU 泡沫是如何破裂的</a>：H100 以前的价格是 $8/小时（如果你能买到的话）。现在有 7 个不同的二手市场以低于 $2 的价格出售。发生了什么？</li><li><a href="https://x.com/perplexity_ai/status/1882466239123255686">来自 Perplexity (@perplexity_ai) 的推文</a>：介绍 Perplexity Assistant。Assistant 使用推理、搜索和 App 来帮助处理日常任务，从简单的问题到跨 App 的操作。你可以预订晚餐、寻找遗忘的歌曲、计算...</li><li><a href="https://x.com/deedydas/status/1882479771428544663?s=46">来自 Deedy (@deedydas) 的推文</a>：中国刚刚发布了一个新模型。字节跳动豆包-1.5-pro 在 Benchmark 上媲美 GPT-4o，但价格便宜 50 倍——缓存输入 Token 为 $0.022/M，输入为 $0.11/M，输出为 $0.275/M——比 DeepSeek 便宜 5 倍，比 o1 便宜 200 倍以上...</li><li><a href="https://x.com/pelaseyed/status/1882471632129994914">来自 homanp (@pelaseyed) 的推文</a>：我不再使用 RAG 了，只需启动一个流水线并将所有内容喂给 DeepSeek，就能获得 10 倍的效果。是的，它可以扩展到超过 1 万份文档。RAG 是一种反模式。</li><li><a href="https://x.com/spyced/status/1881725740917670079">来自 Jonathan Ellis (@spyced) 的推文</a>：我构建了一个工具来解决大型代码库的上下文问题。1/N</li><li><a href="https://github.com/braintrustdata/braintrust-proxy">GitHub - braintrustdata/braintrust-proxy</a>：通过在 GitHub 上创建一个账户来为 braintrustdata/braintrust-proxy 的开发做出贡献。</li><li><a href="https://www.braintrust.dev/docs/guides/proxy">AI proxy - 文档 - Braintrust</a>：访问来自 OpenAI, Anthropic, Google, AWS, Mistral 等的模型。
</li>
</ul>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1332454948553887831)** (193 条消息🔥🔥): 

> `Model Context Protocol (MCP), MCP 与工具集成, Obsidian MCP server, MCP 能力与连接, MCP 派对规划` 


- **对 Model Context Protocol 的兴奋**：成员们对 **Model Context Protocol (MCP)** 的潜力表现出极大的热情，将其描述为集成各种 AI 能力的中心节点。
   - 许多参与者表示，他们渴望在工作流中进一步探索其应用。
- **MCP server 连接与工具**：讨论集中在连接由不同编程语言编写的多个 server 的能力，以便在 MCP 中有效利用各种工具。
   - 成员们提到了实现不同工具的便捷性以及强大集成的潜力。
- **分享 MCP 资源**：分享了各种与 MCP server 相关的资源和 GitHub 链接，展示了现有的实现和社区贡献。
   - 这包括指向特定 MCP server 的链接，例如针对 Obsidian 的 server 以及通用的 MCP 功能。
- **即将举行的 MCP 派对**：在一次成功的 hack session 之后，参与者讨论了举办一场 **MCP 派对**，以深入研究该协议。
   - 分享了一个电子表格，用于组织未来专注于进一步探索 MCP 的会议名单和日期。
- **关于工具和实际应用的讨论**：成员们谈到了 MCP 的实际应用，例如将其与 **Cursor** 配合使用，以及与 **youtube-dl** 等工具集成。
   - 重点强调了 MCP 简化与编程工具和 AI 交互的创新方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cs16.samke.me/">cs16.css</a>: 基于 Counter Strike 1.6 UI 的 CSS 库。</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/cancellation/">Cancellation</a>: ℹ️ 协议修订：2024-11-05。Model Context Protocol (MCP) 支持通过通知消息可选地取消正在进行的请求。任何一方都可以...</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability">Architecture</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability-negotiation">Architecture</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: 未找到描述</li><li><a href="https://github.com/go-go-golems">GO GO GOLEMS!</a>: GO GO GOLEMS BUILD GO GO GADGETS. GO GO GOLEMS! 有 34 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/tumf/mcp-shell-server">GitHub - tumf/mcp-shell-server</a>: 通过创建账号为 tumf/mcp-shell-server 的开发做出贡献。</li><li><a href="https://github.com/MarkusPfundstein/mcp-obsidian">GitHub - MarkusPfundstein/mcp-obsidian: MCP server that interacts with Obsidian via the Obsidian rest API community plugin</a>: 通过 Obsidian rest API 社区插件与 Obsidian 交互的 MCP server - MarkusPfundstein/mcp-obsidian</li><li><a href="https://github.com/rusiaaman/wcgw">GitHub - rusiaaman/wcgw: Shell and coding agent on claude desktop app</a>: Claude 桌面应用上的 Shell 和编程 Agent。通过创建账号为 rusiaaman/wcgw 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers。通过创建账号为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/rusiaaman/wcgw/blob/fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7/src/wcgw/client/mcp_server/server.py#L129-L138">wcgw/src/wcgw/client/mcp_server/server.py at fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7 · rusiaaman/wcgw</a>: Claude 桌面应用上的 Shell 和编程 Agent。通过创建账号为 rusiaaman/wcgw 的开发做出贡献。</li><li><a href="https://github.com/go-go-golems/go-go-mcp">GitHub - go-go-golems/go-go-mcp: Anthropic MCP go implementation</a>: Anthropic MCP 的 Go 语言实现。通过创建账号为 go-go-golems/go-go-mcp 的开发做出贡献。</li><li><a href="https://github.com/calclavia/mcp-obsidian">GitHub - smithery-ai/mcp-obsidian: A connector for Claude Desktop to read and search an Obsidian vault.</a>: 让 Claude Desktop 读取和搜索 Obsidian vault 的连接器 - smithery-ai/mcp-obsidian
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1332077459214045327)** (239 条消息🔥🔥): 

> `Perplexity Assistant 在 iOS 上线、AI 模型对比、Perplexity 用户体验问题、Assistant 功能反馈、Perplexity 替代方案` 


- **Assistant 在 iOS 发布延迟**：讨论显示 Perplexity assistant 的 iOS 版本正在等待 Apple 的审批，希望在获得许可后尽快上线。
   - 用户对该助手的到来表示期待，并指出目前通过移动端应用访问功能存在挑战。
- **Perplexity 与竞争对手在 AI 性能上的对比**：多位用户对各种 AI 模型的相对优势发表了评论，强调 ChatGPT 在准确性方面优于 Perplexity 等其他模型。
   - 共识认为，虽然像 Gemini 这样的模型提供了搜索能力，但 Perplexity 的回答（特别是在引用来源方面）受到了好评。
- **用户对 Perplexity 当前功能的挫败感**：用户对 Perplexity 内部某些功能和模型的访问受限表示担忧，特别是关于 O1 以及 Assistant 的整体体验。
   - 用户讨论了对 Bug 和限制的挫败感，认为该服务正在失去相对于竞争对手的优势。
- **关于 AI 模型使用的见解**：用户分享了不同 AI 模型如何满足各种需求的见解，特别称赞了 Sonar 的集成以及与其他模型相比的性能。
   - 显然，人们强调了独特功能作为竞争差异化因素以维持用户参与度的必要性。
- **探索 Perplexity 的替代方案**：对话中涌现出大量关于 Perplexity 替代方案的建议，包括 DeepSeek 和 Abacus 等平台，因其用户友好的功能和定价而受到关注。
   - 用户表示有兴趣探索这些替代方案，特别是由于它们提供了具有竞争力的功能和高效的响应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ffnews.com/newsarticle/fintech/revolut-launches-its-highest-savings-rates-for-uk-customers-of-up-to-5-aer-variable/">Revolut Launches Its Highest Savings Rates for UK Customers, of Up to 5% AER (Variable)</a>: Revolut 为其英国即时访问储蓄账户大幅提高了利率，提供高达 5% AER 的利率。</li><li><a href="https://x.com/aravsrinivas/status/1882555331764781102?s=46">Aravind Srinivas (@AravSrinivas) 的推文</a>: 我们将在 Perplexity 上提供运行在美国数据中心的 R1。并对所有人免费。</li><li><a href="https://x.com/filicroval/status/1882727675468657138">Filipe | IA (@filicroval) 的推文</a>: 5. 在一小时内不写一行代码创建一个 Perplexity 克隆版</li><li><a href="https://x.com/AravSrinivas/status/1882493999388430376">Aravind Srinivas (@AravSrinivas) 的推文</a>: Perplexity Assistant 的日历和 GMail 读取权限：正在开发中；计划在 3 周内发布。它现在已经可以总结你的未读邮件和即将到来的日程...</li><li><a href="https://bsky.app/profile/wsig.me/post/3lggtvaidtk23">Will Sigmon (@wsig.me)</a>: 自去年夏天以来，我一直致力于改善健康和增强体能。今天，我很高兴地报告减重了 45 磅！这一进步源于生活方式的调整、饮食的改善以及来自 M 的支持...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1332097304039264367)** (8 条消息🔥): 

> `流行文化最新动态、动作冒险电影、即将举行的技术会议、AI 研发药物、Laravel 框架` 


- **流行文化最新动态**：一位用户分享了[流行文化最新动态](https://www.perplexity.ai/search/tell-me-about-the-latest-popul-0wiWshd.TlOVvukAW.7zTQ)，重点关注近期趋势和讨论。
   - 这些更新突出了娱乐和社交媒体中引起当前观众共鸣的新兴主题。
- **必看顶级动作冒险电影**：分享了一个[最佳动作冒险电影](https://www.perplexity.ai/search/best-action-adventure-movies-c-lft_ADWwSLW6rj12clUyig)的链接，详细介绍了基于观众评分的推荐。
   - 这份清单为寻求探索精彩电影体验的影迷提供了指南。
- **即将举行的技术会议**：两位用户讨论了[即将举行的技术会议](https://www.perplexity.ai/search/upcoming-tech-conferences-DsytQrrjTf6MCGka4sEeYQ)，这些会议将汇集行业领袖和创新者。
   - 这些活动有望展示技术领域的尖端发展和社交机会。
- **AI 研发药物的新进展**：一位用户分享了关于即将上市的 [AI 研发药物](https://www.perplexity.ai/page/ai-developed-drugs-coming-soon-KafDx1.USaWRvWfDBgYk.g) 的信息。
   - 这展示了 AI 在变革制药和医疗保健解决方案方面的潜力。
- **了解 Laravel 框架**：一位用户请求了解 [Laravel](https://www.perplexity.ai/search/what-is-laravel-website-6jMT7xsmRGiA2AHQAnSF1A#1)，这是一个以优雅语法著称的流行 PHP 框架。
   - 这突显了开发者对高效 Web 应用程序开发实践的兴趣。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1332086657540948068)** (6 条消息): 

> `API 更新、Sonar API 调用、API 定价与搜索` 


- **API 随更新获得关注**：成员们对团队在之前被忽视后终于开始关注 **API** 并进行相关更新表示高兴。
   - *看到取得进展令人欣慰*。
- **了解 Sonar API 搜索**：关于是否可以禁用 **Sonar API** 调用的搜索引发了讨论，并询问它是否总是会生成搜索。
   - 一位成员澄清说，定价文档指出 Sonar Pro 可以执行多次搜索以进行全面的信息检索。
- **API 定价困惑得到澄清**：人们对 API 定价结构表示担忧，特别是关于**搜索**以及它们如何累积成本。
   - 一位成员分享道，根据他们的理解，成本结构意味着无论聊天类型如何，**每 1000 次 API 调用**都会产生费用。
- **聊天交互中的冗余搜索成本**：一位成员强调了他们的使用案例，即初始搜索是有益的，但随后的响应似乎产生了不必要的搜索成本。
   - 他们指出，当继续聊天不需要额外搜索时，定价和功能感觉有些冗余。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/pricing">Pricing - Perplexity</a>: 未找到描述</li><li><a href="https://github.com/ppl-ai/api-discussion/discussions/121">API Pricing - how to tell how many searches a given request incurred · ppl-ai/api-discussion · Discussion #121</a>: 根据定价文档，Perplexity 对模型产生的每 1000 次搜索查询收取 5 美元。然而，我没有看到根据模型响应来确定执行了多少次搜索的方法...
</li>
</ul>

</div>

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1332114179259236425)** (88 条消息🔥🔥): 

> `LM Server 访问、网络设置困惑、LM Studio 的 Vision Models、LM Studio 中的 Tool Use、模型兼容性问题` 


- **跨网络访问 LM Server**：为了让其他设备能够访问 LM Server，设置中有一个启用本地网络访问的复选框。
   - 成员们对命名规范表示困惑，建议使用“仅限 Loopback”等术语来澄清预期功能。
- **对本地网络设置的困惑**：讨论显示，诸如“本地网络”之类的术语在涉及 `localhost` 和 `0.0.0.0` 时经常导致误解。
   - 一些人考虑将设置重命名为更具描述性的名称，主张通过更清晰的沟通来提升用户体验。
- **调研 LM Studio 的最佳 Vision Models**：成员们讨论了 8b-12b 范围内最好的 Vision Models，在考虑特定模型兼容性的同时，建议包括 **Llama 3.2 11b**。
   - 多次提到了与 LM Studio 中可用的 MLX 和 GGUF 模型的兼容性，并询问这两个平台的模型是否可以共存。
- **在 LM Studio 中实现 Tool Use**：成员们分享了在 LM Studio 中启用 Tool Use 的见解，允许 LLM 通过 REST API 与外部函数和 API 进行交互。
   - 提供了说明性资源，强调对于想要扩展 LLM 能力的用户来说，实现 Tool Use 是可行的。
- **模型加载与兼容性查询**：一位用户表示在让 LM Studio 识别其本地 GGUF 模型时遇到困难，询问是否需要特定的模型格式。
   - 提出了关于如何在解决模型发现挑战的同时，与本地 OpenAI API 进行集成的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/discussions/2#677efb4c846852dc90e75cd0">leafspark/Llama-3.2-11B-Vision-Instruct-GGUF · Modelfile of  Llama-3.2-11B-Vision-Instruct</a>: 未找到描述</li><li><a href="https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/discussions/2#677efb4c846852dc90">leafspark/Llama-3.2-11B-Vision-Instruct-GGUF · Modelfile of  Llama-3.2-11B-Vision-Instruct</a>: 未找到描述</li><li><a href="https://github.com/bytedance/UI-TARS">GitHub - bytedance/UI-TARS</a>: 通过在 GitHub 上创建账号，为 bytedance/UI-TARS 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/advanced/tool-use">Tool Use - Advanced | LM Studio Docs</a>: 使 LLM 能够与外部函数和 API 交互。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1332089347222016070)** (117 条消息🔥🔥): 

> `NVIDIA RTX 5090 性能、Llama-3.3 模型需求、AI 硬件对比、GPU 显存容量、LM Studio 在旧硬件上的性能` 


- **NVIDIA RTX 5090 性能提升有限**：讨论显示，尽管 **RTX 5090** 拥有 **1.7 倍的带宽**，但其性能相比 **RTX 4090** 仅提升了 **30%**。
   - 成员们表示失望，并指出额外的 **32GB VRAM** 可能不会显著提升较小模型的性能。
- **运行 Llama-3.3 需要大量资源**：为了快速运行 **Llama-3.3-70B-Instruct-GGUF** 模型，建议使用拥有 **96GB VRAM** 的**双 A6000** 以获得高效性能。
   - 用户指出，在 **4090** 上运行大于 **24GB** 的模型具有挑战性，特别是在不牺牲速度的情况下。
- **AI 基准测试揭示了 VRAM 的局限性**：有人担心较小的模型无法利用增加的内存带宽，特别是在没有 **AVX2** 的旧 GPU（如 **1080 Ti**）上运行时。
   - 参与者得出结论，对于较小模型实现最佳性能，合适的基准测试在很大程度上取决于 **VRAM 容量**和**带宽**。
- **GPU 和内存规格对 AI 任务至关重要**：讨论涉及了在旧硬件配置上运行 AI 模型时内存速度和类型的重要性，特别是使用前几代 GPU 时。
   - 用户强调了将高性能服务器与过时的 NVIDIA 显卡（如 **P40**）混合使用时的挑战和局限性。
- **当前 LLM 实现面临的挑战**：讨论中的许多人对目前用于 LLM 的**消费级硬件**现状表示沮丧，认为其通常不足以处理高性能任务。
   - 该小组承认，鉴于企业有能力补贴成本，通过 **AI 推理 API** 可能会提供更好的机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/">NVIDIA GeForce RTX 5090 显卡</a>：采用 NVIDIA Blackwell 架构。</li><li><a href="https://www.storagereview.com/review/nvidia-geforce-rtx-5090-review-pushing-boundaries-with-ai-acceleration">NVIDIA GeForce RTX 5090 评测：通过 AI 加速突破界限</a>：NVIDIA GeForce RTX 5090 评测：2025 年 1 月 30 日发布，售价 1,999 美元。5090 是否会重新定义高性能游戏和 AI 工作负载？</li><li><a href="https://benchmarks.ul.com/procyon/ai-text-generation-benchmark">Procyon AI 文本生成</a>：测试 AI LLM 性能可能非常复杂且耗时，完整的 AI 模型需要大量的存储空间和带宽进行下载。 
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1332077931769499751)** (159 条消息🔥🔥): 

> `R1 Model Performance, DeepSeek API Concerns, Aider Benchmark Results, Using Different Models in Aider, New AI Tools and Developments` 


- **R1+Sonnet 创下新的 SOTA**：**R1+Sonnet** 的组合在 aider 多语言基准测试中以 **64%** 的得分创下了新的 **SOTA**，且成本比 **o1** 低 **14 倍**。
   - 这突显了显著的性能提升，许多用户表示相比其他替代方案，更倾向于这种模型组合。
- **DeepSeek 支付信任问题**：由于**支付安全**问题，用户对使用 **DeepSeek** 的 API 表示担忧，部分用户更倾向于选择 **OpenRouter** 作为替代方案。
   - 用户注意到 **OpenRouter** 版本的 **R1** 可能与 DeepSeek 直接提供的版本有所不同，引发了关于可信度的讨论。
- **Aider 基准测试对比**：反馈表明，与标准的 architect/editor 配对相比，**thinking tokens** 导致的基准测试结果更差，这暗示了推理模型中存在效率低下的问题。
   - 用户观察到，由于可能影响性能，不建议在推理模型的上下文中保留旧的**思维链 (CoTs)**。
- **R1 响应的不稳定性**：一些用户报告了 **R1** 性能的不一致性，特别是在上下文超过特定大小时，可能会导致错误或响应时间变慢。
   - 建议通过选择性的上下文管理（如使用 **/read-only**）来缓解某些性能问题。
- **与其他模型的对比**：用户将 **R1** 与 **Sonnet** 和 **Claude** 等其他模型进行了对比，指出它们在速度和准确性方面具有不同的性能动态。
   - 讨论还集中在一些新的 AI 工具（包括名为 **Trae AI** 的 VSCode 分叉）因缺乏重大创新而让人感到多余。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ai_newsz/status/1882849927765819596">来自 Artificial Intelligence News (@ai_newsz) 的推文</a>: 亿万富翁兼 Scale AI CEO Alexandr Wang：DeepSeek 拥有约 50,000 块 NVIDIA H100，由于现行的美国出口管制，他们无法公开谈论这些设备。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API、提供商、统计数据</a>: DeepSeek R1 Distill Llama 70B 是一个基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B。</li><li><a href="https://x.com/OpenRouterAI/status/1882692225051881632">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 今日前 3 名热门模型均产自中国 👀 引用 Aofei Sheng (@aofeisheng) 的话：不知为何，@OpenRouterAI 上排名前三的热门模型都来自中国，而排名第一的 (@deepseek_ai) 仅仅是一个 s...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>: R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA。成本比 o1 低 14 倍。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1332106185813528708)** (41 条消息🔥): 

> `Python 中的日志记录实践、Aider 的工作流与上下文管理、Deepseek 模型性能、Git 中忽略文件的管理、Aider 中的 Architect Mode` 


- **优化 Python 中的日志记录实践**：为了防止日志输出导致聊天历史过于臃肿，一位用户建议使用 logging 模块将日志导出到 **readonly** 文件，并在 Prompt 中引用该文件，而不是直接复制终端输出。
   - 这种方法可以节省时间并保持上下文整洁，从而实现与 AI 更高效的交互。
- **精简 Aider 的工作流**：讨论强调了高效管理 Aider 上下文的重要性，特别是在处理大型代码库时，因为多个终端命令可能会产生过多的输出。
   - 用户表示需要 Aider 在运行命令时不干扰对话，并提出了各种改进输出处理的建议，例如 run-and-drop 选项。
- **Deepseek 模型性能问题**：多位用户指出 **Deepseek r1** 模型面临的挑战，包括处理时间长以及尽管 API 设置正确但仍无响应。
   - 用户担心上下文处理不足可能会限制该模型在实际应用中的有效性。
- **处理 Git 中的忽略文件**：一位用户在尝试将 `.gitignore` 文件添加到 Aider 时遇到问题，最终通过使用 `/read-only` 命令而非将其从 `.gitignore` 中移除来解决。
   - 该解决方案帮助用户在保持原有文件管理习惯的同时，依然能有效地利用 Aider。
- **Architect Mode 的实用性**：有人对 **Architect Mode** 的实用性提出了疑问，反馈建议它经常与其他模式的功能重叠。
   - 用户分享了对模式切换时效率低下和故障的担忧，并提出了实现更流畅操作的建议。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1332132437278589048)** (4 条消息): 

> `已删除的消息、管理员操作` 


- **神秘消失的消息**：一位用户提到一条关于 **waitlist** 的消息似乎消失了，让其他人感到困惑。
   - 另一位用户推测 **admin** 可能删除了该链接，但尚未得到确认。
- **管理员在消息删除中的角色**：有人指出消息的删除可能与 **admin** 的操作有关，尽管具体细节尚不清楚。
   - 社区对这种情况表示不确定，表明该问题缺乏相关信息。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1332113027260088361)** (42 条消息🔥): 

> `Sky-T1-32B-Flash, DeepSeek-R1 性能表现, RLHF 讨论, AI 媒体与影响力者` 


- **Sky-T1-32B-Flash 解决“过度思考”问题**：NovaSkyAI 推出了 [Sky-T1-32B-Flash](https://novasky-ai.github.io/posts/reduce-overthinking/)，这是一个开源推理模型，在不牺牲准确性的前提下，将生成长度缩短了 **50%**，成本仅为 **$275**。
   - 该模型的 [权重已发布](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Flash)，可用于进一步降低推理成本的实验。
- **DeepSeek-R1 稳居前列**：据 [lmarena_ai](https://x.com/lmarena_ai/status/1882749951924715578) 报道，DeepSeek-R1 目前在 Arena 总榜排名 **第 3**，与 OpenAI-o1 持平，但价格便宜 **20 倍** 且完全开源权重。
   - 它的能力包括在编程和数学等 **技术领域排名第 1**，并为社区提供了一个 **完全开源的模型**。
- **关于 RLHF 对准确性影响的辩论**：有一场关于 RLHF 对模型准确性影响的讨论，有观点认为“RLHF 抹杀准确性”已是过时的看法，反映了过去一年的变化。
   - 贡献者指出，良好的 RLHF 实践不会降低评估结果，除非安全要求与其他性能指标发生冲突。
- **对高质量 AI 媒体的需求**：用户对知名 AI 媒体质量下降表示担忧，由于 [Stratechery](https://stratechery.com) 在 AI 讨论中的相关性降低，一些人开始转移注意力。
   - 有人呼吁推荐高质量的 AI 播客和 YouTube 频道，并指出目前许多平台更注重博眼球而非提供有价值的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Sauers_/status/1882453855704900022">来自 Sauers (@Sauers_) 的推文</a>: @JacquesThibs 公平地说，Gemini, o1-preview, 4o 和 Sonnet 答对的问题通常不被允许进入数据集</li><li><a href="https://x.com/novaskyai/status/1882498072414216426?s=61">来自 NovaSky (@NovaSkyAI) 的推文</a>: 1/5 ⚡️介绍 Sky-T1-32B-Flash⚡️，我们的开源推理模型，旨在解决“过度思考”问题，在不牺牲准确性的情况下将生成长度（及推理成本！）缩短 50% —— 仅通过...</li><li><a href="https://x.com/morqon/status/1882794870114525498">来自 morgan — (@morqon) 的推文</a>: 扎克伯格在测量他数据中心的规模</li><li><a href="https://x.com/__nmca__/status/1882563755806281986">来自 Nat McAleese (@__nmca__) 的推文</a>: Epoch AI 将发布更多细节，但对于感兴趣的人，在 OpenAI 这边：我们完全没有使用 FrontierMath 数据来指导 o1 或 o3 的开发。(1/n)</li><li><a href="https://x.com/arithmoquine/status/1882506931040100701">来自 henry (@arithmoquine) 的推文</a>: r1 蒸馏的 OOD 空间非常奇怪。它像是一碗无形的汤，却仍在拼命维持自身形态</li><li><a href="https://x.com/polynoamial/status/1882461290947547175">来自 Noam Brown (@polynoamial) 的推文</a>: 一觉醒来看到新的未饱和评估的感觉。祝贺 @summeryue0, @alexandr_wang, @DanHendrycks 以及整个团队！引用 Dan Hendrycks (@DanHendrycks)：我们正在发布《人类最后的考试》...</li><li><a href="https://x.com/lmarena_ai/status/1882749951924715578">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 突发新闻：DeepSeek-R1 冲上 Arena 前三🐳！目前总榜排名第 3，与顶级推理模型 o1 持平，同时价格便宜 20 倍且开源权重！亮点：- 技术领域排名第 1...</li><li><a href="https://huggingface.co/bespokelabs/Bespoke-Stratos-32B">bespokelabs/Bespoke-Stratos-32B · Hugging Face</a>: 未找到描述</li><li><a href="https://stratechery.com/2025/an-interview-with-daniel-gross-and-nat-friedman-about-models-margins-and-moats">关于模型、利润和护城河，专访 Daniel Gross 和 Nat Friedman</a>: 采访 Daniel Gross 和 Nat Friedman，讨论 Stargate、DeepSeek 以及模型的利润和护城河将从何而来。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1332080162422001865)** (29 条消息🔥): 

> `DeepSeek 性能、OpenAI 与 Benchmark 对比、中国工作态度与美国认知、模型训练中的推理、Cope 讨论` 


- **DeepSeek R1 表现优于 o1-pro**：据报道，DeepSeek R1 在性能对比中“碾压”了 o1-pro，关于该模型输出质量的讨论正在流传。
   - 一位用户指出，其成功可能归功于它未被包含在早期的 Benchmark 循环中。
- **对 OpenAI 影响力的怀疑**：有说法称 OpenAI 的工程实践已将结果对比变成了一场游戏，引发了对非开源研究可信度的担忧。
   - 另一位成员评论道，依赖 OpenAI 的输出进行训练可能会削弱新模型的可防御性。
- **关于工作伦理的文化认知**：成员们辩论了美国对中国工作伦理的看法，强调了一种信念，即一个拥有十亿人口的国家必然具备独立创新的能力。
   - 一些参与者认为对中国创新的轻视态度完全令人费解。
- **关于训练模型中推理能力的讨论**：参与者对主要基于 OpenAI 输出训练的 Base Model 在开发强大推理能力方面的有效性表示怀疑。
   - 有人指出，在 Common Crawl 等数据集上的高困惑度（perplexity）结果可能表明，除了简单使用先前模型的输出外，还存在更稳健的训练方法。
- **社区中的应对机制（Cope）**：越来越多的人认为，面对竞争性结果时，许多围绕模型性能和训练策略的讨论都演变成了“自我安慰式”（cope）的解释。
   - 成员们越来越多地将这些回应贴上“cope”的标签，同时批评关于训练方法的流行叙事。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/xwang_lk/status/1882497087704818175">来自 Xin Eric Wang (@xwang_lk) 的推文</a>: 这非常可疑。OpenAI 刚刚将 Web/OS Agent 排行榜上的结果对比变成了一场工程游戏。这正是为什么如今只有开源研究才是可靠且值得信赖的。http...</li><li><a href="https://x.com/swarooprm7/status/1882557350160277551?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Swaroop Mishra (@Swarooprm7) 的推文</a>: DeepSeek 很棒，但不要过度解读它在 @DanHendrycks 等人制作的 Humanity's Last Exam Benchmark 上的领先地位。该数据集是针对 o1、Gemini、GPT、Claude 等模型对抗性创建的，但...</li><li><a href="https://x.com/roydanroy/status/1882779637866197463">来自 Dan Roy (@roydanroy) 的推文</a>: 我觉得答案很明显？他们非法利用 o1 进行了训练，不是吗？Meta 这样做风险太大，确实。但这一直是业内“小人物”成功的秘诀...</li><li><a href="https://fxtwitter.com/ericzelikman/status/1882098435610046492">来自 Eric Zelikman (@ericzelikman) 的推文</a>: 引用 Ivan Fioravanti ᯅ (@ivanfioravanti) 👀 DeepSeek R1（右）碾压了 o1-pro（左） 👀Prompt: "write a python script for a bouncing yellow ball within a square, make sure to handle collision ...</li><li><a href="https://youtu.be/mcDejkj1tYU?si=FyIALsl0o8VSSPQn">Elon 与 Trump、Sam Altman 就 AI 超级项目爆发内战</a>: Krystal 和 Saagar 讨论了 Elon Musk 和 Sam Altman 之间的科技大佬内战。订阅 PREMIUM Breaking Points 以获取完整的早期访问权限...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1332087774932766821)** (64 条消息🔥🔥): 

> `Discord Summarization Tools, DeepSeek Salaries, Applied NLP Reading List, AI Development, Tech Competition` 


- **探索 Discord 总结工具**：讨论集中在各种 Discord 总结工具的有效性上，提到了一个提供内部总结的工具，但收到的反馈褒贬不一。
   - 成员们注意到一些现有功能（如 Discord 频道内总结）并不受欢迎，因此建议寻找更新、更好的工具。
- **DeepSeek 值得关注的薪资**：一位用户强调了对技术公司向 DeepSeek 员工提供 **$5.5M** 薪资以吸引人才离开竞争对手的担忧。
   - 这场对话引发了对高额薪酬如何重塑 AI 领域人才动态的反思，特别是在股票和基本工资方面。
- **策划应用 NLP 阅读列表**：一位成员分享了为应用 NLP 课程分配论文建议的需求，讨论倾向于 **Attention** 和 **BERT** 等基础论文。
   - 反馈强调应纳入近期具有影响力的论文，以符合 NLP 和 AI 技术的进步。
- **AI 开发挑战**：一位用户讨论了利用 AI Agent 构建应用程序的挑战，幽默地建议创办一份名为 * 的时事通讯。
   - 该话题包含了关于 AI 创造盈利应用潜力的轻松评论，以及技术旅程中的趣闻。
- **技术领域的竞争与企业策略**：成员们对“技术老牌势力（tech old moneys）”试图通过向 AI 领域的顶尖人才提供巨额报价来瓦解竞争对手表示担忧。
   - 企业竞争的概念及其对人才留存的影响引发了关于 AI 行业就业未来的热烈讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eugeneyan.com/writing/llm-reading-list/">语言建模阅读列表（开启你的论文俱乐部）</a>：一些基础论文及每篇的一句话总结；开始你自己的论文俱乐部吧！</li><li><a href="https://x.com/martinbowling/status/1882264741156114936?s=61">来自 Martin Bowling (@martinbowling) 的推文</a>：@elfelorestrepo @skirano @deepseek_ai 看看我的 repo，我把它做成了一个 tool call，给出查询和上下文，然后发送给 r1，并让它将推理包装在 &lt;ant_thinking&gt; 中...</li><li><a href="https://x.com/jiayi_pirate/status/1882839504899420517">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>：具体的 RL 算法并不重要。我们尝试了 PPO、GRPO 和 PRIME。长 CoT 都会出现，而且效果似乎都不错。我们还没时间调整超参数，所以不想...</li><li><a href="https://www.loom.com/share/34b37822c6784989bafd6fcc5fee6420?sid=75bf3b4c-61b5-46fd-a2b1-7c7fe911df89">Smol Talk Alpha - 2024年11月！</a>：Smol Talk 平台介绍 - 获取你自己的个性化 AI 新闻！注册：月度 (https://buy.stripe.com/dR602I7Sv7FYfN69AA)，年度 (https://buy.stripe.com/00g9DifkX6BU8kE145)，终身 (https://...</li><li><a href="https://x.com/wzihanw/status/1882875780902068563">来自 Zihan Wang (@wzihanw) 的推文</a>：我现在担心的是“技术老牌势力”如何向 DeepSeek 的大牛们 🐳 提供 550 万美元的年薪，希望能解散团队并瓦解这样的对手。不，我绝不希望...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>：我们在 CountDown 游戏中复现了 DeepSeek R1-Zero，它确实有效。通过 RL，3B 基础 LM 自发地发展出了自我验证和搜索能力。你可以体验那个“啊哈时刻（Ahah moment）”...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1332230156848201860)** (2 条消息): 

> `OpenAI 的工作自动化模型，Claude 的 Mechanistic Interpretability，Test-Time Scaling 语录` 


- **OpenAI 将自动化 50% 的工作**：[OpenAI](https://x.com/andersonbcdefg/status/1882662657590952134) 宣布即将发布一个可以自动化 **50% 工作**的模型。
   - 这标志着劳动力市场向自动化程度提高的方向潜在转变。
- **Mechanistic Interpretability 确认了 Claude 的身份**：Anthropic 声称他们在 Mechanistic Interpretability 方面的工作揭示了 **Claude 是同性恋**。
   - 这一声明展示了该组织的实验发现以及对身份解读的参与。
- **CEO 强调 Test-Time Scaling 的重要性**：在一段著名的引言中，National Reasoning Model Association 的 CEO 表示：*“唯一能阻止拥有 test-time scaling 的坏人的，是拥有 test-time scaling 的好人。”*
   - [来源](https://x.com/CFGeek/status/1882864786725376180) 强调了 **test-time scaling** 在克服对抗性挑战中的战略意义。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/CFGeek/status/1882864786725376180">来自 Charles Foster (@CFGeek) 的推文</a>：“唯一能阻止拥有 test-time scaling 的坏人的，是拥有 test-time scaling 的好人。”——National Reasoning Model Association CEO</li><li><a href="https://x.com/andersonbcdefg/status/1882662657590952134">来自 Ben (e/treats) (@andersonbcdefg) 的推文</a>：openai：我们要发布一个能自动化 50% 工作的模型；anthropic：通过使用 mechanistic interpretability，我们证明了 claude 是同性恋。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1332095474613878857)** (4 条消息): 

> `高级 NLP 课程更新，多模态领域的重要模型，ViT 和 CLIP 在音频领域的应用，用于统一嵌入的 LLaVA` 


- **高级 NLP 课程需要关注多模态**：一名工作人员寻求更新其 **高级 NLP** 课程，增加关于 **Multimodality** 和 **VLMs** 的讲座，特别提到 **ViT** 和 **CLIP** 是关键模型。
   - 他们正在考虑添加一个处理统一嵌入（unified embeddings）的 VLM，并在 **LLaVA** 和 **QwenVL** 之间进行权衡。
- **用于音频应用的 AST 和 MSCLIP**：另一位成员建议 **AST** 和 **MSCLIP** 值得考虑，因为它们通过将 **音频频谱图（audio spectrograms）** 作为图像输入，将 **ViT** 和 **CLIP** 应用于音频领域。
   - 这种适配展示了这些模型在传统图像处理之外的多功能性。
- **LLaVA 在 VLM 架构中的原始角色**：一位成员指出，**LLaVA** 是在转向 **vqgans** 和 **qformers** 等技术之前的原始架构版本。
   - *他们提到 LLaVA 甚至创造了该架构的范式（recipe），* 这突显了它在多模态模型发展中的重要性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1332086344649805915)** (28 条消息🔥): 

> `Interconnects 订阅价值, Operator AI Agent 反馈, ModernBERT 见解, Semianalysis 的挑战, 端侧智能 (On-device intelligence) 讨论` 


- **Interconnects 订阅提供额外福利**：用户对 **Interconnects** 订阅的附加价值表示惊讶，特别是包含了 **seminalysis** 功能。有人称其为“物超所值”并强烈推荐。
- **Operator AI Agent 面临可用性挑战**：反馈包括 **Operator AI Agent** 在总结内容方面表现挣扎，且操作需要频繁的用户确认。用户指出它应该解析 HTML 以增强功能，但目前表现不佳。
- **关于 ModernBERT 的 OLMo Tokenizer 讨论**：一名用户分享了关于 **ModernBERT** 及其高效 **OLMo Tokenizer** 应用的[链接](https://jina.ai/news/what-should-we-learn-from-modernbert/)。他们强调了该模型与以往模型相比在参数效率和创新方面的优势。
- **Semianalysis 爬取问题**：一名成员指出，由于 **Substack** 的 JS 内容加载和追踪链接替换，爬取它特别具有挑战性。这导致在尝试自动化从 **Semianalysis** 收集信息时遇到了困难。
- **端侧智能 (On-device intelligence) 见解分享**：一名成员敦促需要传达某些 AI 任务不适合 **edge devices**（边缘设备），并引用了一条强调端侧智能重要性的推文。讨论指向了在有限硬件上运行任务所带来的复杂性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://jina.ai/news/what-should-we-learn-from-modernbert/">What Should We Learn From ModernBERT?</a>：更大的训练数据、高效的参数规模以及深而薄的架构，ModernBERT 为未来的类 BERT 模型指明了方向。</li><li><a href="https://www.platformer.news/openai-operator-ai-agent-hands-on/">OpenAI launches its agent</a>：上手体验 Operator —— 人工智能一个充满希望但令人沮丧的新前沿</li><li><a href="https://x.com/soldni/status/1882669954971168879">Luca Soldaini 🎀 (@soldni) 的推文</a>：端侧智能值得研究，因为计算只会变得越来越便宜且充足 🤗 很高兴能很快（Real Soon™️）在这个领域做出贡献。引用 Ben (e/treats) (@andersonbcdefg) 的话：它很快，而且...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1332387100171173900)** (8 条消息🔥): 

> `Adobe Podcast Enhance Speech 工具, 访谈音频设置, 音频质量 vs. 魔法音频` 


- **Adobe Podcast 'Enhance Speech' 褒贬不一**：一位用户评论说，**Adobe Podcast 'Enhance Speech' 工具**在使用时听起来会有机器人感，尤其是在测试了播客音频后。
   - 虽然它在单人录音中表现良好，但在多人设置中会出现问题，这进一步强调了高质量麦克风设置的必要性。
- **多人音频设置成本高且复杂**：另一位成员表示，要为小组访谈获得良好的音频，需要使用高质量设备，如 **Rode Podmic** 和 **Rodecaster**，这可能既昂贵又费力。
   - 他们提到目标是达到录音室级别的质量，但承认为多人进行设置所涉及的复杂性。
- **避免使用“魔法音频”技术**：有人表示不倾向于使用“魔法音频”工具，因为这些工具通常会导致过度处理的声音，感觉很假。
   - 用户强调看重良好的原始音频质量，而不是会损害声音真实性的增强技术。
- **针对一对一访谈的有限音频投入**：目前的意向是专注于**线下一对一访谈**，直到有能力获得编辑或 AV 协助。
   - 这种方法旨在尽量减少音频制作的工作量，但仍认可高质量声音的重要性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1332279414116974632)** (5 条消息): 

> `Presidential AI Action Order, AI Regulations Review, New AI Action Plan, Special Advisor for AI and Crypto, Free Market AI Development` 


- **总统令旨在加强 AI 领导地位**：最近的总统令概述了消除美国 AI 创新障碍的举措，强调了开发**意识形态中立（ideologically unbiased）**的 AI 系统对于维持美国在该领域领导地位的重要性。该法令废除了之前的指令，包括关于 AI 安全与可信度的 **Executive Order 14110**。
   - *该法令愿景是通过自由市场手段和政府政策，巩固美国作为全球 AI 领导者的地位。*
- **重点审查现有 AI 法规**：该法令要求审查并可能移除被视为阻碍**美国 AI 霸权（U.S. AI dominance）**的现有 AI 法规。此举旨在促进经济竞争力和国家安全。
   - 讨论中提到需要在 180 天内制定一份**新的 AI 行动计划**，以符合这些目标。
- **设立 AI 与加密货币特别顾问**：作为法令的一部分，将设立一个新的 **Special Advisor for AI and Crypto** 职位，凸显了政府将 AI 进步与数字货币倡议相结合的重点。该职位预计将帮助引导美国在 AI 和加密货币交叉领域的政策。
   - 此外，重要备忘录的修订工作要求在 **60 天**内完成。
- **强调自由市场与经济竞争力**：该法令强调了自由市场方法对 AI 发展的重要性，主张消除 AI 系统中的任何**意识形态偏见**。这一策略被认为对于确保**经济竞争力**和增强**国家安全**至关重要。
   - *该法令旨在通过可持续的 AI 增长为所有美国人创造更美好的未来。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/">Removing Barriers to American Leadership in Artificial Intelligence &#8211; The White House</a>: 根据宪法和美利坚合众国法律赋予我作为总统的权力，特此命令如下：第 1 节。</li><li><a href="https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-">Removing Barriers to American Leadership in Artificial Intelligence &#8211; The White House</a>: 根据宪法和美利坚合众国法律赋予我作为总统的权力，特此命令如下：第 1 节。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1332079638788444361)** (18 条消息🔥): 

> `Jailbreaking Models, MLOps Resources, Flash Infer Talk, Attention Methods, Flex Attention vs Differential Attention` 


- **模型越狱（Jailbreaking）非常简单**：对于任何知道如何使用 **Google** 的人来说，对其他模型进行*越狱*都是微不足道的，除了品牌安全之外没有有效的保护措施。
   - 这引发了人们在快速演变的格局中对现有模型**安全性**的质疑。
- **寻找 MLOps GPU 资源**：一位用户正在为经验丰富的 MLOps 专业人士寻找学习资源，以帮助他们从基于 **CPU** 的模型过渡到使用 **PyTorch** 的**分布式 GPU** 模型。
   - 这凸显了在 ML 工作流中对高级 **GPU usage** 易懂指南的需求。
- **关于 Flash Infer 演讲的提问**：一位用户询问了 **Flash Infer** 演讲的提问频道，指出在没有频道的情况下在 **YouTube** 上提问存在问题。
   - 另一位用户主动提出从聊天中读取问题，强调了直播期间的社区支持。
- **关于 Attention 方法的讨论**：讨论了 **Flex Attention** 及其处理 **Differential Attention** 等方法的能力，并深入探讨了它们的运作方式。
   - 具体细节包括运行两个不同的 Softmax 并询问 Flex Attention 是否支持此操作，展示了技术兴趣。
- **Attention 方法的伪代码**：一位用户分享了演示如何使用 **PyTorch** 实现 Differential Attention 的伪代码，展示了 **lambda value** 和 Softmax 操作等概念。
   - 他们还注意到 **MLA** 和 **Lightning Attention** 之间的相似之处，为其他人简化了部分技术咨询。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1332081728218730527)** (78 条消息🔥🔥): 

> `CUDA Toolkit 12.8 发布，Blackwell 架构特性，TensorCore 指令，FP8 和 FP4 支持，sm_90a 与 sm_100a 的兼容性` 


- **CUDA Toolkit 12.8 已发布**：新发布的 [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-downloads) 包含了对 Blackwell 架构的支持以及更新的文档。
   - 不过，一些成员注意到某些文档更新有所延迟，但随后确认这些文档现已上线。
- **Blackwell 架构特性解析**：Blackwell 架构引入了重大更新，通过增强的 FP8 能力，在 RTX 4090 上可能达到 **660 TFLOPS**。
   - 讨论强调了在 B100/B200 GPU 中实现宣传的 TFLOPS 的复杂性，据估计其工程投入超过 **100 亿美元**。
- **第五代 TensorCore 指令可用性**：虽然第五代 TensorCore 指令在 sm_100 和 sm_101 上可用，但在 sm_120（特别是 RTX 5090）上却缺失。
   - 指令集的这种分歧引发了对潜在代码库碎片化和集成挑战的担忧。
- **cuBLAS 支持 FP8 和 FP4 类型**：cuBLAS 的重大更新已确认，包括支持 FP8 和 FP4 数据类型，以提高计算效率。
   - 这对于未来利用量化效率的工作负载尤为重要，尽管对其具体实际收益的反应不一。
- **架构间的兼容性担忧**：关于 sm_90a 与 sm_100a 及 sm_120 的前向兼容性存在不确定性，目前无法保证兼容。
   - 有人指出，虽然某些功能可能重叠，但某些性能增强指令（如 `wgmma`）是特定架构独有的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2208.09225">FP8 Quantization: The Power of the Exponent</a>：在为高效推理进行神经网络量化时，低比特整数是效率的首选格式。然而，低比特浮点数具有额外的自由度，可以分配一些 b...</li><li><a href="https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.qtensor.nvfp4_tensor.html#module-modelopt.torch.quantization.qtensor.nvfp4_tensor">nvfp4_tensor &mdash; Model Optimizer 0.21.1</a>：未找到描述</li><li><a href="https://x.com/__tensorcore__/status/1882532829999075366">来自 Vijay (@__tensorcore__) 的推文</a>：CUDA 12.8 刚刚发布，支持 Blackwell。第五代 TensorCore 家族指令：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions</li><li><a href="https://developer.nvidia.com/cuda-downloads">CUDA Toolkit 12.1 下载</a>：获取 NVIDIA 专有计算栈的最新功能更新。</li><li><a href="https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330">NVIDIA Blackwell RTX GPU 软件迁移指南：CUDA 12.8、PyTorch、TensorRT 和 Llama.cpp 指南</a>：应用程序必须更新到最新的 AI 框架，以确保与 NVIDIA Blackwell RTX GPU 的兼容性。本指南提供了确保兼容性所需的内核软件库更新信息...</li><li><a href="https://github.com/NVIDIA/cccl/pull/3166#issuecomment-2608244981">向 NV_TARGET 添加对 sm_101 和 sm_101a 的支持，由 bernhardmgruber 提交 · Pull Request #3166 · NVIDIA/cccl</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1332154863022379048)** (22 messages🔥): 

> `Async Computation with Custom CUDA Kernels, bfloat16 vs float32 Precision in PyTorch, Tensor Parallel Configurations in HF Transformers, Learning Rate Schedulers in PyTorch, Vision-Based Models Optimization` 


- **Async Computation vs cudaMalloc 问题**：一位用户指出，在运行 **torch computation async** 以及自定义 **CUDA kernels** 时，会触发对 **cudaMalloc** 的调用，从而影响异步执行。
   - 这引发了关于是否需要 **pre-allocated tensors**（预分配张量）以维持异步 Kernel 运行的讨论。
- **bfloat16 与 float32 的精度差异**：讨论集中在某些路径执行 **bfloat16** 计算，而其他路径转换为 **float32** 的情况。成员们探讨了 **precision**（精度）不匹配对模型训练结果的影响。
   - 有成员指出，当使用 `nn.Linear` 时，所有输入和权重在技术上都应处于 **bfloat16** 状态。
- **Hugging Face Transformers 中的潜在 Hooks**：有建议认为 **Hugging Face's transformers** 可能会引入影响结果计算方式的 hooks，导致非预期的输出不一致。
   - 一位用户指出，他们无法在纯 **PyTorch example** 中复现这些问题，这暗示了 **HF transformers models** 中存在细微差别。
- **学习率调度器（Learning Rate Scheduler）建议**：关于在前 N 个 step 使用 **linear warmup** 并结合 **CosineAnnealing** 轮转的讨论，强调了优化学习率的实践。
   - 分享了相关 PyTorch 文档链接，以澄清 **CosineAnnealingWarmRestarts** 和 **LinearLR** 调度器的机制与配置。
- **视觉模型：输入与反馈**：一位用户表达了对 **vision-based models** 的关注，并邀请大家针对该领域的优化和学习策略提供反馈。
   - 对社区成员分享的见解表示认可和感谢，强调了协作学习的动态过程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts">CosineAnnealingWarmRestarts &mdash; PyTorch 2.5 documentation</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR">LinearLR &mdash; PyTorch 2.5 documentation</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR">CosineAnnealingLR &mdash; PyTorch 2.5 documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1332428297614655558)** (1 messages): 

> `Flash Infer, Deep Learning Techniques, Code Generation, Custom Kernels, Attention Patterns` 


- **年度首场关于 Flash Infer 的讲座**：本年度第一场讲座由 **Zihao Ye** 主讲 **Flash Infer**，涵盖了 **code generation**、**custom kernels** 以及各种 **attention patterns** 等主题。
   - 讲座将在 [YouTube](https://www.youtube.com/@GPUMODE) 上进行直播。
- **聚焦代码生成技术**：**Flash Infer** 深入探讨了先进的 **code generation** 技术，这些技术有望通过定制化的 kernels 和高效的执行模式来提升性能。
   - 本次讲座重点介绍了 JIT 和 AOT 编译策略的创新点，这对于实时应用至关重要。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

cpeterson42: X 上的 AI Infrastructure 社区：
https://x.com/i/communities/1879760488256491834

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1332078707480985702)** (1 messages): 

> `ComfyUI hiring, Machine Learning Engineers` 


- **ComfyUI 招募 machine learning engineers**：ComfyUI 目前正在**招聘 machine learning engineers**，加入负责维护 ComfyUI 及其生态系统的团队。
   - 该公司以对顶尖公司发布的**模型提供 Day 1 支持**而自豪，并邀请那些对优化开源代码充满热情的人员申请。更多信息可以在[这里](https://comfyorg.notion.site/Founding-Machine-Learning-Engineer-1696d73d36508014bfbaf5aebf39b145)找到。
- **ComfyUI 获 VC 支持**：ComfyUI 是一家位于湾区（Bay Area）的 **VC 支持的公司**，强调其宏大的愿景和充足的资金跑道（long runway）。
   - 他们正在寻找热衷于在 machine learning 领域为开源社区做出贡献的人才。



**提到的链接**：<a href="https://comfyorg.notion.site/Founding-Machine-Learning-Engineer-1696d73d36508014bfbaf5aebf39b145">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一个将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1332097640892203100)** (4 messages): 

> `DeepSpeed Integration, Hugging Face Accelerate Library, Throughput Comparison, Communication Overhead` 


- **探索 DeepSpeed 集成选项**：一名成员建议，如果模型支持 DeepSpeed 集成，可以尝试 **DeepSpeed Zero 2** 或 **DeepSpeed Zero 3**。
   - 这突显了利用现有技术优化模型性能的持续努力。
- **Hugging Face Accelerate 简化分布式训练**：另一位成员推荐了 Hugging Face 的 [Accelerate 库](https://huggingface.co/docs/accelerate/en/index)，它只需极少的代码改动即可简化 PyTorch 中的分布式训练。
   - 仅需四行代码，它就能有效地为各种配置准备好模型。
- **405B 与 DeepSeek-v3 之间的吞吐量困惑**：一位成员质疑为什么尽管 **DeepSeek-v3** 的激活参数更少，但 **405B** 的吞吐量却更高，怀疑**通信开销（communication overhead）**是一个因素。
   - 这引发了关于效率以及模型性能指标背后复杂性的讨论。
- **在 SageMaker 中成功使用 Accelerate**：一位用户确认 Accelerate 库对他们来说效果很好，但指出在 **SageMaker notebook** 中使其顺利运行存在挑战。
   - 这反映了用户在将库集成到特定环境时可能面临的实际障碍。



**提到的链接**：<a href="https://huggingface.co/docs/accelerate/en/index">Accelerate</a>：未找到描述

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1332275542350889040)** (2 messages): 

> `Parallel Prefix Sum in Distributed Systems, MPI_Scan in CUDA` 


- **探索分布式前缀和（Prefix Sum）算法**：有人提出了关于在使用具有分布式内存的多 GPU 上以分布式方式实现 **prefix sum** 算法的可用性问题，特别引用了 **Lecture 20, 21, 和 24**。
   - 找到的最接近的参考是 **MPI_Scan**，但似乎缺乏关于其在 **CUDA** 中实现的教程和文档。
- **本地扫描结合全对全（All-to-All）通信**：一份回复建议执行类似于 GPU block scans 的分布式前缀和，即每个节点计算其**本地扫描（local scan）**，并通信最后一个元素进行全对全交换。
   - 建议使用提到的 **MPI_Scan** 可以帮助通信修正本地扫描所需的少量中间结果，从而实现高效计算。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1332080917090402335)** (2 messages): 

> `ComfyUI 见面会，DeepSeek R1 模型性能` 


- **参加在旧金山举行的 ComfyUI 见面会**：下周四，**ComfyUI** 活动将在 GitHub 办公室举行，届时将有开源开发者的演示和讨论。查看[活动详情](https://lu.ma/6skuqn7c?tk=xiHyMZ)以加入社区并分享你的工作流。
   - 重点嘉宾包括 **MJM** 和 **Lovis**，他们将在小组讨论中分享见解。
- **DeepSeek R1 超越原始模型**：重蒸馏的 **DeepSeek R1** 模型 (1.5B) 的性能优于其原始蒸馏版本，现已在 [Hugging Face](https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0) 上线。成员们对此次成功后即将发布的模型表示期待。
   - 来自 **Mobius Labs** 的公告强调了他们未来致力于蒸馏更多模型的承诺。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Mobius_Labs/status/1882841665427390858">Mobius Labs (@Mobius_Labs) 的推文</a>：我们重蒸馏的 @deepseek_ai R1 (1.5B) 优于原始蒸馏模型！获取地址：https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0。我们正在蒸馏更多模型，并且……</li><li><a href="https://lu.ma/6skuqn7c?tk=xiHyMZ">ComfyUI 官方旧金山见面会 @ Github · Luma</a>：在 Github 办公室举行的首场官方 ComfyUI 旧金山见面会！快来结识其他 ComfyUI 用户，向社区分享你的工作流，或者提出你的建议……
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1332112187153711256)** (1 messages): 

> `累加器清零，ThunderKittens GitHub` 


- **累加器清零是否必要？**：一位成员询问是否有必要在 Kernel 代码中对 **accumulators** 进行清零，特别是在 [matmul.cu](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu#L87C13-L87C36) 文件中。
   - 他们想知道累加器是否在设置过程中已经自动清零，这暗示了一个潜在的优化点。
- **探索 ThunderKittens 项目**：**ThunderKittens** 的 GitHub 仓库以用于高速 Kernel 的 Tile Primitives 代码为特色，增强了矩阵乘法任务的开发过程。
   - 该项目托管在 GitHub 上，为有兴趣推动 **HazyResearch** 工作的贡献者提供了协作机会。



**提到的链接**：<a href="https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu#L87C13-L87C36">ThunderKittens/kernels/matmul/H100/matmul.cu at main · HazyResearch/ThunderKittens</a>：用于高速 Kernel 的 Tile Primitives。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1332087675099807805)** (20 messages🔥): 

> `多项式方程添加、迷宫任务实现、重构 Reasoning-Gym 结构、动态奖励系统、外部贡献者认可` 


- **项目新增多项式方程**：一名成员提交了一个 [添加多项式方程的 PR](https://github.com/google-deepmind/clrs/tree/master/clrs/_src/clrs_text)，与简单的线性方程并列。
   - 另一名成员对这一贡献表示感谢，称：*'太棒了，非常感谢！'*
- **迷宫任务提案获得支持**：一名成员提议设计一个寻找最短路径的迷宫任务，并展示了一个示例布局。
   - 该想法受到了好评，有迹象表明这将是 *Reasoning-Gym 的一个极好补充*。
- **精简 Reasoning-Gym 的结构**：一位贡献者计划简化 Reasoning-Gym 的结构，旨在使数据集消费更轻松、结果解析更有效。
   - 建议包括添加用于数据集比较的静态函数以及方便地注册数据集。
- **基于准确率的动态奖励**：一名成员建议实现一个基于答案准确率的奖励系统，奖励根据答案与正确答案的接近程度进行调整。
   - 这涉及配置奖励机制，允许使用用户定义的表达式（如 *-x**2*）以实现清晰度和自定义。
- **表彰外部贡献者**：一名成员通过授予第一座外部贡献者奖杯来表彰另一名成员的努力。
   - 这一认可突显了项目的协作性质，并鼓励持续的贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/google-deepmind/clrs/tree/master/clrs/_src/clrs_text">clrs/clrs/_src/clrs_text at master · google-deepmind/clrs</a>：通过在 GitHub 上创建账号来为 google-deepmind/clrs 的开发做出贡献。</li><li><a href="https://github.com/open-thought/tiny-grpo/blob/eafedd78ff86dbb724a3dd21bb04ab6523ac8f3c/train.py#L122-L130">tiny-grpo/train.py at eafedd78ff86dbb724a3dd21bb04ab6523ac8f3c · open-thought/tiny-grpo</a>：极简且易于修改的 GRPO 实现。通过在 GitHub 上创建账号来为 open-thought/tiny-grpo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1332446242294333632)** (1 messages): 

> `Canvas 更新、HTML & React 代码渲染、ChatGPT 桌面应用发布、新功能的访问层级` 


- **Canvas 现在集成了 OpenAI o1**：**Canvas 功能**现在与 OpenAI **o1** 兼容；用户可以从模型选择器中选择 **o1**，并利用工具箱图标或 `/canvas` 命令。
   - 此更新已面向 **Pro、Plus 和 Team 用户**开放。
- **HTML & React 代码现在可以渲染**：Canvas 增加了**渲染 HTML 和 React 代码**的能力，使其能够处理更复杂的交互。
   - 此功能面向 **Pro、Plus、Team 和 Free 用户**开放。
- **Canvas 在 macOS 桌面应用上全面推出**：Canvas 功能已在所有用户层级的 **ChatGPT macOS 桌面应用**上全面推出。
   - 此次更新标志着 **macOS 桌面用户**体验的重大提升。
- **Enterprise 和 Edu 更新即将到来**：Canvas 与 **o1** 的集成以及 **HTML/React 渲染**功能都将在几周内推送到 **Enterprise 和 Edu 层级**。
   - 这确保了**教育和企业用户**能够获得更广泛的访问权限和功能增强。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1332085744239509617)** (131 messages🔥🔥): 

> `Deepseek and R1, API Usage for Chatbots, AI Integrated IDEs, Token Costs for AI Models, Operator's Browser Interaction` 


- **DeepSeek 的侧边项目方法**：DeepSeek 的 CEO 提到 R1 仅仅是一个利用空闲 GPU 开发的侧边项目，这在 AI 社区引发了幽默感和关注。
   - 据信 DeepSeek 得到了中国政府的支持以增强本土 AI 模型，这突显了一种获得关注和效率的聪明策略。
- **聊天机器人 API 使用创意**：一名成员建议使用 OpenAI API 创建聊天机器人，作为支付专业版费用的更便宜替代方案，以此发现潜在的成本节省。
   - 有人指出，使用 GPT-3.5，5 美元大约可以处理 **250 万个 Token**，强调了自定义实现的经济效益。
- **AI 在游戏开发中的令人兴奋的前景**：讨论了 AI Agent 如何通过在 Unity 中使用类似画笔工具的实时应用来彻底改变游戏开发。
   - 这一想法还扩展到了软件工程，建议将 IDE、终端和数据库无缝集成以实现高效的工作流。
- **对 AI 互联网访问的担忧**：对话强调了给 AI Agent 提供互联网访问权限相关的风险，因为这可能会重置从之前交互中学习到的行为。
   - 用户表达了在更广泛的背景下使用 AI 模型时需要谨慎，特别是在搜索过程中的数据保留方面。
- **Operator 的新功能**：讨论了 Operator 工具与 Web 浏览器交互的能力，尽管对其当前的功能和局限性存在疑问。
   - 成员们询问该功能是否会扩展到独立应用程序，强调了在 AI 辅助任务中实现更强集成的潜力。



**Link mentioned**: <a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: 通过在 GitHub 上创建一个账户来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1332316381152284702)** (2 messages): 

> `Release of O3` 


- **呼吁发布 O3**：一名用户表达了希望立即发布 **O3** 的愿望。
   - 另一名成员拒绝了这一提议，简单地回答道：**“不，谢谢”**。
- **关于 O3 发布的不同观点**：对话突显了关于 **O3** 发布意见的分歧，一名成员非常渴望，而另一名则不感兴趣。
   - 这种交流反映了社区内不同程度的参与感。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1332296004980183083)** (7 messages): 

> `Public vs Private Information, NDA Discussions, Misinformation Concerns` 


- **公开 vs 私密：了解你的界限！**：一名成员强调，在本频道发布的所有内容都被视为**公开**信息，并警告不要分享与 NDA（保密协议）相关的信息。
   - 他们指出对分享内容负责的重要性，特别是涉及机密内容时。
- **NDA 讨论缺乏透明度**：讨论了分享非 NDA 信息的后果，以及这如何可能导致关于该领域真正**相关**内容的误导信息传播。
   - *一位成员感叹道*，推广过时的方法可能会误导他人，让他们认为这些方法仍然是前沿技术。
- **顺其自然：道歉时刻**：一名成员为之前的言论道歉，建议*忽略之前说过的话*，因为对话已经转移。
   - 这表明了对清晰度的渴望，并承认之前的评论可能导致了误解。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1332296004980183083)** (7 messages): 

> `2023 趋势, NDA 合规性, 虚假信息担忧` 


- **2023 趋势已过时**：*“不，伙计，那太 2023 了”* 表明人们认为某些趋势或做法应该重新考虑或更新。
   - 这种情绪表明在讨论中需要推动更具相关性和时效性的方法。
- **必须严格遵守 NDA**：提醒注意遵守 NDA，指出讨论不应涉及非 NDA 信息。
   - 它强调了不被过时的非 NDA 材料误导的重要性。
- **公共话语中的虚假信息**：引发了对虚假信息的担忧，强调将非 NDA 方法宣传为更优越可能会扭曲对相关性的认知。
   - 讨论指出了将过时方法包装成当前或前沿技术的风险。
- **承认沟通误解**：消息线程中包含了一位成员对造成的任何困惑的道歉，表明愿意澄清之前的观点。
   - 这反映了在讨论中追求清晰沟通和理解的愿望。


  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1332403076023324703)** (1 messages): 

> `React + TypeScript + Tailwind Web 应用, 应用架构, 数据管理策略, 开发工作流, 版本控制标准` 


- **React + TypeScript + Tailwind 应用入门策略**：一个结构化的文本文件记录了构建 **React + TypeScript + Tailwind** Web 应用的入门策略，涵盖了初始设置、架构和数据管理。
   - 根据分享的 [Google Document](https://docs.google.com/document/d/1JYIUQjqVchaWGQSDNsBs7NpXdexOiXr4B79cOpprdxY/edit?usp=sharing)，希望获得关于架构、数据策略和工作流的反馈。
- **中心化文档标准**：文档包含了创建中心化 **GUIDELINES.md** 文件的标准，该文件由涵盖 **architecture**（架构）、**logging**（日志）和 **design system**（设计系统）的模块化部分组成。
   - 这种结构化方法旨在提高整个项目的清晰度和可维护性。
- **灵活的数据管理方法**：概述了一种**灵活的持久化策略**，从 **localStorage** 开始，并设计为可以轻松迁移到 **Supabase**。
   - 这种适应性对于扩展应用的数据需求至关重要。
- **结构化开发步骤**：提供了构建 **design system**、**logging system** 和 **changelog interface** 等系统的指令。
   - 这些步骤旨在简化开发并增强团队协作。
- **使用 Changelog 保持一致的版本控制**：该提案包括遵守 **'Keep a Changelog'** 格式，以确保清晰且一致的版本更新。
   - 这种做法对于跟踪更改和维护项目历史至关重要。



**提到的链接**：<a href="https://docs.google.com/document/d/1JYIUQjqVchaWGQSDNsBs7NpXdexOiXr4B79cOpprdxY/edit?usp=sharing">启动提示词策略</a>：## 从初始提示词开始：创建一个 React + TypeScript + Tailwind Web 应用，包含：布局：持久的页眉和侧边栏导航，带有菜单项（例如 Menu 1, Menu 2, Menu 3）以及子菜单...

  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1332082650608828416)** (144 条消息🔥🔥): 

> `Stripe Webhook 实现，Bolt Functions 问题，基于 Supabase 的消息系统，Chat 加载问题，OpenAI API 错误` 


- **Stripe Webhook 实现成功**：经过大量的故障排查和 **14M tokens** 的消耗，一位成员成功实现了 **Stripe webhook**，解决了与 Supabase functions 中 `user_id` 参数重叠导致的冲突。
   - 该成员指出，他们必须手动调试并修复 AI 无法识别的问题。
- **Bolt Functions 不再执行更改**：用户报告称 Bolt 现在只是建议代码更改，而不是自动应用它们，这导致用户不得不手动在代码中搜索并进行修正，令人感到沮丧。
   - 一位成员指出，这种行为的突然变化可能与 chat history 的引用有关。
- **Supabase 消息系统的挑战**：一位用户正在寻求使用 Supabase 的 realtime 功能实现**消息系统**的帮助，目前在 **Row Level Security (RLS)** 方面遇到了问题。
   - 他们表示希望与成功完成类似任务的其他开发者讨论实现方案。
- **Bolt 上的 Chat 加载问题**：多位用户在 Bolt 上遇到了加载 chat 的问题，报告显示刷新或使用 VPN 可以恢复访问，而其他用户则持续面临该问题。
   - 据观察，地理位置可能会影响连接性，部分用户报告使用 VPN 成功，而其他用户仍无法访问 chat。
- **OpenAI API 使用错误**：一位用户在尝试通过 OpenAI API 使用 **o1-mini** 时遇到了 400 错误，尽管该模型在 Playground 中运行正常。
   - 他们推测问题可能源于 Bolt 内部不正确的命名规范或配置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bolt.new/?autoAuth">bolt.new</a>: 未找到描述</li><li><a href="https://stackblitz.com/register">开始使用 StackBlitz - StackBlitz</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1332091794619498546)** (134 条消息🔥🔥): 

> `DiStRo 与 GPU 训练、Tiny Stories 模型训练、OpenAI 与市场反应、新 AI 模型与推理能力、Transformer 中 Self-Attention 的重要性` 


- **探索 DiStRo 以增强 GPU 训练**：成员们讨论了 [DiStRo](https://link.to.DiStRo) 在没有 NVLink 的情况下提高多 GPU 训练速度的潜力，并指出它在模型能够放入单个 GPU 显存时表现出色。
   - 有建议称，将 PyTorch 的 FSDP2 等框架与 DiStRo 结合使用，可以提升大型模型的性能。
- **Tiny Stories 模型训练见解**：讨论集中在 [Tiny Stories](https://link.to.tinystories)（特别是 500 万到 7 亿参数规模）的各种 tokenization 策略和模型参数的表现。
   - Real Azure 分享了研究结果，指出调整 tokenization 改善了模型的 perplexity，这为未来的模型优化提供了方法。
- **OpenAI 的市场地位与担忧**：提到了 Microsoft、Meta 和 Amazon 的当前估值，并对 OpenAI 的品牌管理和公众认知感到不安。
   - 成员们对炒作与现实之间的差距表示担忧，强调了持续的产品性能的重要性。
- **AI 推理模型的进展**：分享了使用 DeepSeek R1 distilled CoTs 重新训练 Hermes 的见解，展示了将推理模式与通用模型结合的效率。
   - 对话探讨了自我蒸馏（self-distillation）技术的潜力，并指出使用 EQ 评分进行性能评估可能会产生新的方法论。
- **AI 模型中的 Self-Attention 与效率**：讨论了 Transformer 模型中 Self-Attention 机制对于 VRAM 效率日益增长的重要性，特别是在模型规模扩大时。
   - Juahyori 提出了关于是否可以通过奖励创意输出来实现自我蒸馏的问题，暗示了模型训练的创新方向。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/ni-no-kuni-ni-no-kuni-wrath-of-the-white-witch-marcassin-ni-no-kuni-follow-yo">未找到标题</a>：未找到描述</li><li><a href="https://fxtwitter.com/Teknium1/status/1882893748742598669">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：我们使用 5k deepseek r1 distilled cots 重新训练了 hermes。我可以确认几件事：1. 你可以拥有通用 + 推理模式，我们使用静态系统提示词标记了来自 r1 的所有 longCoT 样本...</li><li><a href="https://x.com/paulgauthier/status/1882833360567095682?t=DXSZ7cjglVQALy1z1mO1IQ&s=19">来自 Paul Gauthier (@paulgauthier) 的推文</a>：R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA，成本比 o1 低 14 倍。64% R1+Sonnet，62% o1，57% R1，52% Sonnet，48% DeepSeek V3。https://aider.chat/2025/01/24/r1-sonnet.html</li><li><a href="https://x.com/alexandr_wang/status/1882481229708358027">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：Scale AI & CAIS 正在发布 Humanity’s Last Exam (HLE)，这是一个包含 3,000 个问题的数据集，由数百名领域专家（博士、教授等）开发，旨在捕捉人类知识的边界...</li><li><a href="https://tenor.com/view/you-see-it-is-part-plan-explain-gif-20448886">You See It Is GIF - You See It Is Part - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/ni-no-kuni-ni-no-kuni-wrath-of-the-white-witch-marcassin-ni-no-kuni-follow-your-dreams-astralynx-gif-8558421469009706806">Ni No Kuni Ni No Kuni Wrath Of The White Witch GIF - Ni no kuni Ni no kuni wrath of the white witch Marcassin ni no kuni - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md">trl/docs/source/grpo_trainer.md at main · huggingface/trl</a>：使用强化学习训练 Transformer 语言模型。- huggingface/trl
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1332085686924480532)** (98 条消息🔥🔥): 

> `Memory bus width, Math questions for LLMs, Reasoning and visual problems in LLMs, Open weight Differential Transformer models, Anime avatars in OSS ML community` 


- **内存总线宽度引发幽默**：一位成员提到某个系统拥有 **512-bit 位宽的 32GB 内存**，引发了关于内存总线和潜在买家钱包一样宽的笑话。
   - 针对最初的评论，随后引发了一阵*笑声*，展现了轻松的调侃氛围。
- **数学问题难倒 LLM**：一位用户提到一个关于穿过 n x n 网格需要多少条直线的数学问题，并对许多 LLM 在这项任务上**表现极差**表示沮丧。
   - 讨论强调了问题的歧义性，以及为了改善 LLM 的回答而需要更清晰定义的必要性。
- **LLM 在视觉问题推理中挣扎**：成员们讨论了 LLM 在解决依赖**视觉推理**的问题时仍然存在困难，并建议提供过多的提示可能会让它们感到困惑。
   - 有人指出，基于所需提示的数量对 LLM 进行基准测试，可能比仅关注正确性更能提供深入的见解。
- **寻找开源权重 Differential Transformer 模型**：一位用户询问关于 **开源权重 Differential Transformer 模型** 的信息，随后另一位用户分享了一个 GitHub 仓库，但对作者的工作质量表示担忧。
   - 对话中包含了对该仓库可信度的谨慎评论，以及对作者社区参与度的批判性审视。
- **OSS ML 中的动漫头像引起关注**：有人轻松地评论道，在开源 ML 领域，**动漫头像**通常代表着能力极强的人。
   - 另一位成员指出，ML OSS 的许多贡献者来自东亚，这说明了社区内的一种文化趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/casper_hansen_/status/1882434989356351557?t=Vaf211QQrbH-PBdTrTDQhQ&s=19">来自 Casper Hansen (@casper_hansen_) 的推文</a>: DeepSeek R1 在 Humanity&#39;s Last Exam 的文本子集中击败了所有其他模型。这只是证实了我们已知的事实——R1 是真正的瑰宝。</li><li><a href="https://fxtwitter.com/tsarnick/status/1882520836739039384?t=NJuPxKFBPXDQq4GJTlgieg&s=19">来自 Tsarathustra (@tsarnick) 的推文</a>: 特朗普总统表示，他已宣布国家能源紧急状态，以释放美国的能源资源，使美国成为“制造业超级大国和世界人工智能之都...”。</li><li><a href="https://fxtwitter.com/sama/status/1882478782059327666">来自 Sam Altman (@sama) 的推文</a>: 大新闻：ChatGPT 免费版将获得 o3-mini！（Plus 版将获得大量的 o3-mini 使用额度）</li><li><a href="https://fxtwitter.com/sama/status/1882505650594611588">来自 Sam Altman (@sama) 的推文</a>: 宏伟。漂亮。建筑。</li><li><a href="https://fxtwitter.com/sama/status/1882505714196988271">来自 Sam Altman (@sama) 的推文</a>: Stargate 1 号站点，德克萨斯州，2025 年 1 月。</li><li><a href="https://math.stackexchange.com/questions/4756401/minimum-number-of-straight-lines-to-cover-n-times-n-grid#:~:text=The%20minimal%20number%20must%20be,horizontal%20(or%20vertical)%20lines.">覆盖 $n \times n$ 网格的最少直线数量？</a>: 我想知道触及 $n \times n$ 网格中每个方格所需的最少直线数量。唯一的附加规则是直线必须经过方格内部，而不是边缘/角落。我发现...</li><li><a href="https://github.com/kyegomez/DifferentialTransformer">GitHub - kyegomez/DifferentialTransformer: 微软“DIFFERENTIAL TRANSFORMER”论文模型的开源社区实现。</a>: 微软“DIFFERENTIAL TRANSFORMER”论文模型的开源社区实现。 - kyegomez/DifferentialTransformer
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1332147017283141786)** (18 条消息🔥): 

> `Myopic Optimization with Non-myopic Approval (MONA), AI Insurance, GPRO and PPO Comparison, GAE and Advantage Estimation, LMArena Rankings` 


- **引入 MONA 以对抗 Reward Hacking**：一项关于强化学习 (RL) 目标的研究提出了 **Myopic Optimization with Non-myopic Approval (MONA)**，旨在避免不理想的多步计划，这些计划即使未被人类察觉，也可能导致 Reward Hacking。该方法结合了短视优化与远见奖励，并通过实证研究进行了论证。
   - 论文讨论了 RL 中常见的 Misalignment 失败模式，以及 MONA 如何在不需要常规 RL 参数之外的额外信息的情况下提供潜在的解决方案。
- **AI 保险的未来引发辩论**：关于精算师开发 **AI insurance** 潜力的推测不断涌现，人们承认 AI 可能会给企业带来重大问题。一位成员幽默地指出 *万物皆可投保*，进一步加剧了这种日益增长的担忧。
   - 关于 AI 如何导致严重商业事故的讨论仍在继续，并提出了保险将如何适应这些新兴技术的问题。
- **探索 GPRO 对 PPO 的影响**：成员们对 **GPRO** 舍弃 Value Function 以及 GAE 可能缓解 **PPO** 早期收敛问题的潜力表示好奇。讨论集中在这一改变是否能通过对全局归一化奖励求和来改进 Advantage Estimation。
   - 一位成员分享了他们对 GAE 功能的理解，强调了它如何导致模型陷入 Loss 困境，而另一位成员则表示希望在即将举行的读书会中深入探讨这一领域。
- **令人印象深刻的 LMArena 排名**：随着 **R1** 在 **LMArena** 中排名 **第 3**，且 **Style Control** 排名 **第 1** 的消息公布，现场气氛热烈。成员们注意到了它的开源特性，强调了社区对其现实影响的热情。
   - 一位成员对该模型的能力表示难以置信，这表明当前 AI 领域正在发生重大进展。



**提及的链接**：<a href="https://arxiv.org/abs/2501.13011">MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking</a>：未来的先进 AI 系统可能会通过强化学习 (RL) 学习到人类无法充分理解并安全评估的复杂策略。我们提出了一种训练方法，可以避免...

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1332086983144509545)** (9 条消息🔥): 

> `AG2 公告，R1 服务器性能，AI 社区动态，Stargate 的影响` 


- **AG2 阐述从微软拆分后的愿景**：AG2 宣布了其社区驱动的 Agent 开发愿景，概述了他们的**治理模型**、社区结构以及对**开源**的承诺，详见[此处](https://medium.com/@ag2ai/ag2s-vision-for-community-driven-agent-development-f9f3ca2b0dc8)。
   - 他们从学术项目转型为拥有**超过 20,000 名开发者**的社区，展示了市场对更易用的 AI Agent 系统需求。
- **R1 颠覆行业预期**：成员们讨论了 **R1** 令人惊讶的性能，认为其“黑马”地位激发了其他服务器技术之间的竞争。
   - 有人评论说 R1 有效地给其他厂商“添了一把火”，凸显了其在当前市场格局中的影响力。
- **性能问题的潜在原因**：有人猜测技术领域的性能问题是源于 **Stargate**，还是某些 **B200 服务器** 效率显著更高。
   - 一位成员幽默地表示，这种关注可能是由于 R1 在该领域的崭露头角。
- **AI 社区与就业市场**：讨论涉及了 AI 发展对就业市场的影响，一位成员开玩笑说，这让**台湾和中国大陆**的从业者比美国人更忙碌。
   - 这种情绪反映了人们对 AI 开发模式转变及其社会经济影响的持续关注。
- **AI 游戏宅男女神 (Gamer Waifu) 引发热议**：一个通过 [YouTube 视频](https://www.youtube.com/watch?v=I65tiaHQuFk) 链接的、被称为“AI gamer waifu”的 AI 概念引发了幽默引用。
   - 这引发了关于 AI 在娱乐和游戏领域日益增长的存在感的笑声和反应。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ag2oss/status/1882878967713259705">来自 AG2 (@ag2oss) 的推文</a>：宣布我们社区驱动的 Agent 开发愿景。阅读关于 AG2 的：- 治理模型 - 社区结构 - 开源承诺 - 前进路线 https://medium.com/@ag2ai/ag2s-vision-for...</li><li><a href="https://medium.com/@ag2ai/ag2s-vision-for-community-driven-agent-development-f9f3ca2b0dc8">AG2 社区驱动 Agent 开发的愿景</a>：两年前我们在开发 FLAML 时首次提出 AutoGen 背后的概念，我们的目标很简单：让它变得更容易……</li><li><a href="https://agi.safe.ai/">人类的最后一场考试 (Humanity's Last Exam)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1332266673561931786)** (7 条消息): 

> `使用 NotebookLM 编辑播客，生成式输出的逆向图灵测试，AI 主持人动画工具` 


- **NotebookLM 简化播客编辑**：一位用户分享了使用 NotebookLM 编辑播客的经验，通过上传现有音频并指示其无缝整合片段。
   - 这实现了近乎完美的流畅度，仅需极少的剪辑即可保持叙事的完整性。
- **生成式输出探索 AGI 主题**：另一位用户讨论了他们的 LM 在创建逆向图灵测试和解读对话含义方面的探索。
   - 输出结果将界面反射为理解网络化未来中 AGI 的潜在镜像，用户觉得这非常有趣。
- **MasterZap 分享 AI 动画工作流**：针对有关 AI 主持人动画的询问，MasterZap 详细介绍了使用 HailouAI 和 RunWayML 等多种工具的综合工作流。
   - 他强调了让虚拟形象看起来自然化的复杂性，同时整合了各种技术以实现逼真的动画效果。



**提到的链接**：<a href="https://www.youtube.com/watch?v=TfPy5oJQn_s">UNREAL MYSTERIES 7: The Callisto Mining Incident</a>：David 和 Hannah 跟随传奇人物 Malcolm Steele 在木卫四（Callisto）上担任空间矿工的冒险故事。了解他、Ted 和 Jessica 的经历……

  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1332092203564269740)** (53 条消息🔥): 

> `上传 PDF, Gemini Advanced 功能, NotebookLM 使用案例, 测验准备, 交互模式加载问题` 


- **上传 PDF 的挑战**：用户报告了向 NotebookLM 上传大型 PDF（特别是《国内税收法典》）时遇到的问题，文件大小在 **18.5MB** 左右时会出现故障。
   - 一位用户对 Gemini 无法准确解析协议表示沮丧，并寻求故障排除方面的帮助。
- **Gemini Advanced 提供更大的上下文**：一位用户确认他们正在使用 Gemini Advanced，并指出它允许上传整个文档进行分析，而不仅仅是部分内容。
   - 然而，尽管 Gemini 具备相关能力，但在获取有关税法定义和法律解释的精确回答方面仍然存在挑战。
- **使用 NotebookLM 进行测验准备**：一位用户询问 NotebookLM 是否可以直接从包含 **220** 个问题的 PDF 文档中生成测验，并强调需要完全一致的文本。
   - 建议他们尝试在 NotebookLM 中使用 Prompt，并使用 Gemini Advanced 模型，特别是 **1.5 Pro** 版本。
- **关于协作与支持的讨论**：几位用户表示愿意互相帮助解决故障，并优化 AI 模型在教育用途上的使用。
   - 对话强调了协作，一些成员提议即使跨越不同时区也要继续讨论问题。
- **关于 NotebookLM 团队的澄清**：一位用户询问频道参与者是否属于 NotebookLM 团队，对方澄清说虽然他们在 Cloud 部门工作，但不属于 NLM 团队。
   - 他们表示，来自工程和产品管理部门的其他成员也在频道中提供支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com?hl=en">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/c558515c-96ed-443e-bb33-3b5cfbcc8a3f?original_referer=https:%2F%2Fwww.google.com%23&pli=1">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1332082491749826570)** (1 条消息): 

> `GPT4All v3.7.0 发布, Windows ARM 支持, macOS 更新修复, Code Interpreter 改进, Chat Templating 增强` 


- **GPT4All v3.7.0 发布，带来令人兴奋的功能**：**GPT4All v3.7.0** 的发布带来了多项更新，包括 **Windows ARM 支持**，使其兼容由 Snapdragon 和 SQ 处理器驱动的设备，尽管目前缺乏 GPU/NPU 加速。
   - 由于模拟限制，用户需要专门从网站安装新的 *Windows ARM* 版本。
- **macOS Bug 修复**：macOS 上的 **GPT4All** 修复了更新期间崩溃的问题，维护工具现在可以正常运行，如果之前使用了临时解决方案，现在可以轻松卸载。
   - 此外，使用 Command-Q 退出应用程序时，聊天记录现在可以正确保存，提升了用户体验。
- **Code Interpreter 增强**：GPT4All 中的 Code Interpreter 得到了改进，可以更优雅地处理超时，并且现在支持 console.log 中的**多个参数**，以更好地兼容原生 JavaScript。
   - 这些改进旨在提高编码任务期间的性能和可用性。
- **Chat Templating 升级**：引入了增强的聊天模板功能，修复了聊天模板解析器中的两个崩溃和一个兼容性问题，包括修复了 EM German Mistral 的默认聊天模板。
   - 新增了对另外五个模型的自动替换支持，继续简化与常用聊天模板的兼容性。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1332077653750059099)** (50 messages🔥): 

> `Prompt Engineering 挑战, GPT4All 模型兼容性, 图像分析工具, 翻译模型推荐` 


- **Prompt Engineering 问题**: 成员们讨论了让 LLM 按预期工作的挑战，特别是在处理 NSFW Prompt 以及更微妙的礼貌用语使用方面。
   - *一位成员指出，使用 'please' 可能会获得更好的结果*，因为模型是在互联网数据上训练的。
- **GPT4All 模型兼容性**: 讨论了各种模型与 GPT4All 的兼容性，特别是 Llama-3.2 和 Qwen-2.5。
   - 另一位成员提到了 *Nous Hermes* 等潜在替代方案，暗示当前模型性能存在问题。
- **图像分析工具讨论**: 一位成员询问了用于通用图像分析的开源模型，要求支持图像上传和提问。
   - 另一位成员推荐了 *Taggui*，强调了其有效性以及用于图像打标签的多 AI 引擎。
- **翻译模型建议**: 一位用户寻求适用于各种语言对的高质量翻译模型推荐。
   - 成员们建议 **Qwen 模型** 应该足够了，同时也提到了 Hugging Face 上可用的 Llama，尽管其资源需求更高。
- **模型加载问题**: 一位用户表示在 GPT4All 中加载 *DeepSeek-R 1-Distill-Qwen-32B-Q4_K_M.gguf* 模型时遇到困难。
   - 成员们指出 GPT4All 团队可能正在调查此事，这表明目前的兼容性存在局限。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1332104378433146975)** (22 messages🔥): 

> `MCP 服务器超时修复, MCP 服务器安装问题, MySQL 和 SQLite 使用, Claude Google 搜索功能, MCP-Alchemy 仓库` 


- **MCP 服务器超时已修复**: 一位成员分享了针对 MCP 服务器响应 **60 秒超时**问题的修复方法，该修复位于 **roo-code** 或 **cline** 源代码中的 `mcphub.ts`。
   - 他们还提供了一个 [VS Code 扩展](https://github.com/qpd-v/Roo-Code)，其中包含此超时修改的更新。
- **MCP 服务器加载成功**: 一位用户报告在解决 PATH 问题后，在 **Claude 启动**时成功加载了基础 **MCP 服务器**且无报错。
   - 最终的修复涉及确保在配置中指定了正确的路径，特别是针对 `uvx.exe`。
- **在 MCP 中使用 MySQL**: 一位用户询问了连接 **MySQL 数据库**最值得信赖的 MCP 服务器，引发了关于各种设置的讨论。
   - 另一位用户推荐了 [mcp-alchemy](https://github.com/runekaagaard/mcp-alchemy)，称其对 MySQL 以及其他数据库都非常有效。
- **Google 搜索功能故障**: 一位用户对 **Claude** 的 **Google 搜索功能**故障表示担忧，寻求社区提供潜在的修复方案。
   - 几位成员表达了类似的经历，暗示高需求可能导致了 Claude 的性能问题。
- **为 MCP 创建数据库**: 一位用户表示他们在 Access 中创建了一个 **test.db**，并询问这对于 MCP 运行是否必要。
   - 他们认为这可能有所帮助，说明了他们在设置 MCP 服务器过程中的学习曲线。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://xkcd.com/303/">Compiling</a>: 未找到描述</li><li><a href="https://github.com/runekaagaard/mcp-alchemy">GitHub - runekaagaard/mcp-alchemy: 一个 MCP (model context protocol) 服务器，为 LLM 提供对 SQLite, Postgresql, MySQL &amp; MariaDB, Oracle, 和 MS-SQL 等关系型数据库的访问和知识。</a>: 一个 MCP (model context protocol) 服务器，为 LLM 提供对 SQLite, Postgresql, MySQL &amp; MariaDB, Oracle, 和 MS-SQL 等关系型数据库的访问和知识。 - runekaagaard/mcp-alchemy</li><li><a href="https://github.com/qpd-v/Roo-Code">GitHub - qpd-v/Roo-Code: Roo Code (原 Roo Cline) 是一个 VS Code 插件，通过 AI 驱动的自动化、多模型支持和实验性功能增强编码体验</a>: Roo Code (原 Roo Cline) 是一个 VS Code 插件，通过 AI 驱动的自动化、多模型支持和实验性功能增强编码体验 - qpd-v/Roo-Code
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1332265243073839106)** (16 条消息🔥): 

> `Orange Flair 请求, MCP Agentic 工具困惑, Glama 集成至客户端, 长期记忆个性化工具` 


- **成员请求 Orange Flair**：多位成员请求 **orange flair**，其中一人问道：*有办法获得 orange flair 吗？* 其他人也表达了类似的请求。
   - Flair 请求似乎得到了及时处理，因为该成员在询问后收到了 *done* 的回复。
- **对 MCP Agentic 工具的困惑**：一位成员寻求关于 **MCP Agentic 工具**功能的澄清，特别是关于它在 Glama 中的激活方式。
   - 另一位成员分享道：*“产品的这部分内容提前泄露了”*，这暗示官方公告预计很快就会发布。
- **将 Glama 集成到非 Glama 客户端**：关于将 Glama 集成到客户端的含义有了明确说明，指出这涉及使用 GUI 来安装托管服务器。
   - 一位成员确认该功能与 **MCP.run** 界面相关，但增加了来自 Glama 的额外功能。
- **用于用户个性化的新工具**：受 ***Titans Surprise 机制***启发，推出了一款新工具，用于记录旨在实现用户个性化的特定交互。
   - 该工具收集用户数据用于长期记忆，可以在 **[GitHub](https://github.com/truaxki/mcp-variance-log)** 上找到。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/runekaagaard?tab=reposi">runekaagaard - 概览</a>：runekaagaard 拥有 101 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/runekaagaard?tab=repositories&q=mcp&type=&language=&sort=stargazers">runekaagaard - 仓库</a>：runekaagaard 拥有 101 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://glama.ai/mcp/servers">开源 MCP 服务器</a>：企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/truaxki/mcp-variance-log">GitHub - truaxki/mcp-variance-log: 寻找对话结构中的统计变化并将异常事件记录到 SQLite 数据库的 Agentic 工具。</a>：寻找对话结构中的统计变化并将异常事件记录到 SQLite 数据库的 Agentic 工具。 - truaxki/mcp-variance-log
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1332422626106146940)** (1 条消息): 

> `AI agents, 联合网络研讨会, AI 任务管理, Redis, 研讨会录像` 


- **观看 AI Agents 网络研讨会录像**：与 [@Redisinc](https://twitter.com/Redisinc) 合作举办的关于构建 AI agents 的联合网络研讨会现已上线，提供了关于它们在分解任务中作用的见解。
   - 会议讨论了增强性能的 **任务管理** 策略，你可以在[这里](https://t.co/bELHnKjRm3)找到录像。
- **了解 AI Agents 的角色**：网络研讨会涵盖了 **AI agents 的方方面面**，探讨了它们如何帮助有效地管理任务。
   - 讨论强调了将任务分解为可管理组件对于 **提升性能** 的重要性。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1332078401724616737)** (36 条消息🔥): 

> `LlamaIndex Agent 教程, 并行流式传输问题, 使用 LlamaParse 处理 PDF, 实时事件流, AI 模型出口管制` 


- **LlamaIndex Agent 设置存在失效链接**：一位用户报告称，LlamaIndex 中的“step by step agent tutorial”链接返回 **500 错误**，而非预期内容。社区建议了替代链接和资源以协助构建 Agent。
   - 有人指出，该教程对于尝试入门 Agent 的新用户至关重要。
- **多 LLM 并行流式传输的挑战**：一位用户在尝试同时从两个不同的 LLM 流式传输事件时遇到了顺序数据问题。其他人建议，所使用的外部库中的异步实现可能无法正常工作。
   - 分享了一个 [Google Colab 链接](https://colab.research.google.com) 示例，以演示可以正常运行的并行流式传输。
- **使用 LlamaParse 进行文档解析**：一位用户询问了使用 LlamaIndex 解析幻灯片演示文稿的最佳实践，特别是针对用 LaTeX 创建的 PDF。有人指出 LlamaParse 支持包括 .pptx 和 .pdf 在内的多种文件格式，有助于数据提取。
   - 分享了相关资源，包括 [LlamaParse 文档](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) 的链接。
- **LlamaIndex Agent 的实时事件**：一位用户寻求关于在使用 LlamaIndex 的 ReActAgent 时，如何与 Token 流一起显示实时事件的建议。建议利用新的 AgentWorkflow 系统以获得更好的事件处理能力。
   - 提供了相关 [文档](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/) 链接以及社区对该新功能的反馈。
- **对 AI 模型出口管制的担忧**：一位用户询问 Llama 模型是否受到美国针对 AI 模型权重和先进计算的新出口管制影响。讨论强调了持续关注影响 AI 应用的监管变化的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1uGbrX5yAt0CMeUeOa4JaIUfWsmPp2tCS?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://x.com/llama_index/status/1882121805542170894">LlamaIndex 🦙 (@llama_index) 的推文</a>：今天发布重大更新！我们很高兴在 LlamaIndex 中引入 AgentWorkflow，这是一个用于创建多 Agent 系统的新型高级系统！LlamaIndex Workflows 是强大的底层构建块...</li><li><a href="https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights">美国针对先进计算项目和人工智能模型权重的新出口管制：七个关键要点</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#hybrid-retriever-with-bm25-chroma">BM25 Retriever - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/LlamaIndexTS/tree/main/apps/next/src/content/docs/llamaindex/guide/agents">LlamaIndexTS/apps/next/src/content/docs/llamaindex/guide/agents (main 分支) · run-llama/LlamaIndexTS</a>：适用于 LLM 应用的数据框架。专注于服务器端解决方案 - run-llama/LlamaIndexTS</li><li><a href="https://github.com/run-llama/llama_parse/blob/4897d01cb075ed0835b7df0d072a7ad4e39c4e64/llama_parse/utils.py#L122">llama_parse/llama_parse/utils.py (commit 4897d01) · run-llama/llama_parse</a>：为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_parse/blob/4897d01cb075ed0835b7df0d072a7ad4e39c4e64/llama_parse/utils.py#L124">llama_parse/llama_parse/utils.py (commit 4897d01) · run-llama/llama_parse</a>：为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/getting_started/starter_tutorial/agent">Agent 教程</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1332222313797521408)** (22 messages🔥): 

> `美国对 AI 模型的出口管制、Cohere 的合规担忧、Oracle Japan 对 Cohere 模型的使用、GPU 限制的市场影响、Blackwell 运行成本` 


- **美国限制 AI 模型出口，引发担忧**：美国商务部宣布了针对 AI 模型权重的新出口管制措施，引发了关于 **Cohere 模型** 是否属于这些监管范围的询问。
   - 一位在服务中使用 Cohere LLM 的 AI 工程师表示：*我担心 Oracle Japan 可能会出现违规行为。*
- **讨论 Oracle Japan 的法律复杂性**：一名团队成员建议，Oracle 可能拥有特定的许可协议，可以降低其在新规下的风险。
   - 建议在 **Oracle Japan** 内部进行咨询，以澄清这些法律影响。
- **强调 AI 监管的市场现状**：讨论揭示了对大型 AI 公司出口限制有效性的怀疑，并断言 **GPU 限制** 主要影响的是消费者。
   - 值得注意的是，*大公司有自己的办法将受限显卡偷运进国内*，从而将监管影响降至最低。
- **Cohere 的加拿大身份受到质疑**：一名成员指出 **Cohere 在加拿大运营**，并询问美国法规将如何影响其业务实践。
   - 强调了针对在 AI 领域运营的加拿大实体的监管影响存在不确定性。
- **对 Blackwell 运行成本的担忧**：一名成员对利用 Blackwell 技术的运营成本表示担忧，称其目前在 **200w** 功率下待机。
   - 这引发了关于在没有立即投入使用的情况下运行重型计算任务的财务影响的讨论。



**提到的链接**：<a href="https://www.sidley.com/ja/insights/newsupdates/2025/01/new-us-export-controls-on-advanced-computing-items-and-artificial-intelligence-model-weights">New U.S. Export Controls on Advanced Computing Items and Artificial Intelligence Model Weights: Seven Key Takeaways</a>：未找到描述

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

sssandra: 请将问题发布在 <#1324436975436038184>，我们将在那里协助进行故障排除（troubleshoot）。
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1332253696880414771)** (5 messages): 

> `9 字母单词、单词列表、有趣的单词` 


- **请求 9 字母单词**：<@1316646968688119818> 请求一个包含 **13 个单词** 且每个单词均为 **9 个字母** 的列表。
   - Cmd R Bot 响应了一个列表，其中多次重复了 **September**。
- **跟进有趣的单词回复**：<@_.dominic> 对收到的 **回复** 表示感兴趣。
   - Cmd R Bot 分享了一个新的 9 字母单词列表，重点突出了 **Fascinate** 和 **Enchant** 等词汇。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1332118839210676306)** (8 messages🔥): 

> `论坛帖子创建、Mojo 中的 Async 代码` 


- **计划发布论坛帖子**：一名成员提议，如果另一名成员在论坛创建帖子，他将进行内部转发。
   - 最初的成员表示同意，并表示愿意从手机上复制粘贴内容。
- **Async 代码讨论**：分享了一个题为 [如何在 Mojo 中编写异步代码](https://forum.modular.com/t/how-to-write-async-code-in-mojo/473) 的论坛主题链接，用于讨论异步编程实践。
   - 成员们确认了在论坛帖子中进行协作的意向。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forum.modular.com/?">Modular</a>：与我们一起构建 AI 的未来，了解 MAX、Mojo 和 Magic。</li><li><a href="https://forum.modular.com/t/how-to-write-async-code-in-mojo/473">如何在 mojo🔥 中编写异步代码？</a>：我看到开发博客说 Mojo 目前缺乏 async fn await 的包装器，但它支持协程（coroutines）本身。如果可能的话，如何编写一个函数，例如打印 whi...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1332122022657921056)** (1 messages): 

> `MAX Builds 页面上线，社区构建的软件包，项目提交流程` 


- **MAX Builds 页面已上线！**：更新后的 **MAX Builds 页面** 现已在 [builds.modular.com](https://builds.modular.com) 上线，其中展示了社区构建软件包的专门板块。
   - 此次更新突出了创作者的贡献，并鼓励更多社区参与。
- **致敬首批软件包创作者**：特别表彰包括 <@875794730536018071> 在内的首批软件包创作者，感谢他们对 MAX Builds 页面的贡献。
   - 他们的努力为未来的社区贡献和软件包共享奠定了基础。
- **如何展示你的项目**：若想让你的项目在 MAX Builds 页面上展示，请向 [Modular 社区仓库](https://github.com/modular/modular-community) 提交一个包含 `recipe.yaml` 文件的 PR。
   - 完整的说明和示例可以在 [这里](https://www.modular.com/community/package-submission) 找到，为社区参与提供了便捷途径。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1332173612773539932)** (17 messages🔥): 

> `__iadd__ 方法，误触输入，新的 Mojo CLI 标志` 


- **__iadd__ 方法详解**：一位成员阐明了 `__iadd__` 方法控制 `+=` 的行为，指出它是一种就地加法（inplace addition）。
   - 另一位成员询问 `iadd` 为何不能返回值，引发了关于使用 `a += 1` 时变量存储方式的讨论。
- **求值顺序的困惑**：对话探讨了像 `a += a + 1` 这样的表达式如何求值，一位成员得出结论，当 `a` 设置为 **6** 时，结果为 **13**。
   - 成员们对求值顺序表示担忧，并建议避免此类复杂写法以防混淆。
- **会议期间的误触输入**：一位成员幽默地解释说，他们之前的消息是会议期间的误触输入。
   - 这引发了另一位成员的机智回应，询问他们是否在键盘上睡着了。
- **发布新的 Mojo CLI 标志**：一位成员介绍了新的 Mojo CLI 标志，特别是 `--ei` 和 `--v`，引起了频道内的关注。
   - 在提到这些标志时还配上了表情符号，为公告增添了视觉元素。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1332079225053773877)** (25 messages🔥): 

> `音频数据集项目，说话人识别模型，标注噪声和音乐级别` 


- **标注背景噪声和音乐**：成员们讨论了为音频样本标注背景噪声的方法，提出了诸如 **无背景噪声**、**轻微噪声** 以及各种音乐级别等分类。
   - 强调了 *“如果不确定，请发挥创意并询问”*，以实现更广泛、更有效的标注策略。
- **开发多说话人转录模型**：有人呼吁利用来自 Text-to-Speech 模型的重叠音频创建一个 **多说话人语音转录数据集**，这可以增强说话人区分能力。
   - 这包括生成保持时间码的数据，以识别哪个说话人在什么时刻说话。
- **通过语音特征识别说话人**：成员们探讨了模型如何根据 **音调和频率** 的变化来识别说话人，这可能包括识别混响或语速等语音特征。
   - 这种端到端训练方法旨在促进在重叠音频场景下更好的 **说话人识别 (Speaker Recognition)**。
- **参与项目的兴趣**：一位成员表达了贡献 **音频数据集项目** 的热切愿望，但询问了有关项目细节和当前状态的更多信息。
   - 对方分享道：*“没有网站，是我在 Discord 上协调大家”*，以此强调该项目非正式的组织结构。



**提及的链接**：<a href="https://colab.research.google.com/drive/140lGFiXXeTsNFp7w5xteCjpmRRaSBvmj?usp=sharing">Google Colab</a>：未找到描述

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1332122346412179488)** (7 条消息): 

> `Teacher-Student Model Divergence, AI4Science 开源项目, Layer Convergence Bias, Vanishing Gradients 讨论` 


- **探索 Teacher-Student Model 中的 Divergence**：一位成员询问了关于将 Teacher 和 Student Model 之间的 Divergence 作为“奖励（reward）”来应用 PPO 进行 Distillation 的论文。
   - 另一位成员建议使用带有 KL-matching 的标准 Student-Teacher Model 作为更常规的方法。
- **博士生寻求开源项目**：一位成员介绍自己是 **AI4Science** 领域的博士生，尽管日程繁忙，仍希望为开源项目做出贡献。
   - 社区建议查看相关频道以获取项目机会。
- **识别出 Layer Convergence Bias**：一位成员引用了关于神经网络中 **Layer Convergence Bias** 现象的研究，指出浅层比深层收敛更快。
   - 这与 Gradient 的稳定性有关，并在最近的一篇 [ICLR 2023 论文](https://openreview.net/forum?id=wlMDF1jQF86)中被揭示。
- **关于 Vanishing Gradients 的辩论仍在继续**：讨论围绕 Vanishing Gradients 是否导致了 DNN 深层面临的收敛问题展开。
   - 成员们承认，虽然它们可能不是主要因素，但它们仍然起着重要作用。



**提到的链接**：<a href="https://openreview.net/forum?id=wlMDF1jQF86">Which Layer is Learning Faster? A Systematic Exploration of...</a>：我们通过实验证明，在神经网络中，浅层比深层收敛更快，并为这一发现提供了理论依据和实际价值。

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1332107701693710367)** (14 条消息🔥): 

> `ModernBERT 与 ModernBART、混合语言模型、思维链推理、Agent-R 自我批判框架、Latro RL 与 Pause Tokens` 


- **关于 ModernBERT 的讨论及对 ModernBART 的期待**：成员们讨论了在最近推出 **ModernBERT** 之后，是否可以合理期待 **ModernBART** 的发布。
   - 虽然有人认为摘要任务对 Encoder-Decoder 模型有需求，但也有人表示 **ModernBART** 的发布公告并非板上钉钉。
- **探索混合语言模型**：一位成员指出在混合架构领域合并 Masked Language Modeling 和 Causal Language Modeling 的潜力，并引用了关于 **GPT-BERT** 的研究。
   - 这种方法在 BabyLM Challenge 2024 中表现出优于传统模型的性能，表明了结合不同范式优势的趋势。
- **图像生成中的思维链推理**：一篇论文讨论了通过实现 **Chain-of-Thought (CoT)** 推理来增强自回归图像生成能力。
   - 通过集成 **Direct Preference Optimization (DPO)** 等技术，研究结果显示图像生成性能有了显著提升。
- **用于智能反思的 Agent-R 框架**：**Agent-R** 框架为语言 Agent 提供了一种创新方法，使其在交互过程中能够动态地进行反思并从错误中恢复。
   - 利用 **Monte Carlo Tree Search (MCTS)**，该方法允许 Agent 进行自我批判并构建恢复正确操作的数据集，从而解决错误恢复中的问题。
- **关于 Latro RL 与 Pause Tokens 的辩论**：一位成员询问了 **Latro RL** 与使用大量 Pause Tokens 训练的模型在推理性能上的根本区别。
   - 对话强调了推理深度和所用架构的重要性，暗示了推理能力的潜在差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.11425">Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training</a>：Large Language Models (LLMs) Agent 在处理交互式环境中的复杂任务时变得越来越关键。现有工作主要集中在通过行为克隆来增强性能...</li><li><a href="https://arxiv.org/abs/2501.13926">Can We Generate Images with CoT? Let&#39;s Verify and Reinforce Image Generation Step by Step</a>：Chain-of-Thought (CoT) 推理已在大型模型中得到广泛探索，以应对复杂的理解任务。然而，此类策略是否可以应用于...</li><li><a href="https://arxiv.org/abs/2501.13028">Optimizing Return Distributions with Distributional Dynamic Programming</a>：我们引入了分布动态规划 (DP) 方法来优化回报分布的统计泛函，将标准强化学习作为特例。之前的分布...</li><li><a href="https://arxiv.org/abs/2406.04823">BERTs are Generative In-Context Learners</a>：虽然 In-Context Learning 通常与 GPT 等因果语言模型相关联，但我们证明了这种能力在 Masked Language Models 中也会“涌现”。通过一个令人尴尬的...</li><li><a href="https://arxiv.org/abs/2410.24159">GPT or BERT: why not both?</a>：我们提出了一种将 Masked Language Modeling 与 Causal Language Modeling 合并的简单方法。这种混合训练目标产生的模型结合了两种建模范式的优势...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/202">meta-llama/Meta-Llama-3-8B · Are there bias weights in Llama3 ?</a>：未找到描述内容。
</li>
</ul>

</div>

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1332393461080395839)** (2 条消息): 

> `Mech Interp, Bilinear MLPs, Weights and Activations Duality, ICLR 2025 Paper` 


- **Mech Interp 的长期承诺**：Mech Interp 目前是一个**热门领域**，但对于使用**以激活为主的方法 (activations-dominant approach)** 能否满足长期预期的疑虑依然存在。
   - 全面的观点应该同时考虑**权重 (weights)** 和**激活 (activations)**，因为它们在本质上是相互关联的。
- **Bilinear MLPs：弥合差距**：由成员主导的一篇新论文探讨了 **Bilinear MLPs**，它可作为 MLP 或 GLUs 的替代方案，并保持与现有 Mech Interp 技术的兼容性。
   - 该方法旨在提高**神经网络权重**的可解释性，同时保持在包括 Transformer 在内的各种模型中的性能。
- **通过权重理解神经网络**：讨论集中在仅通过权重理解神经网络的挑战上，因为激活函数模糊了输入与输出之间的关系。
   - 在其 ICLR 2025 论文中，研究人员提议分析缺乏逐元素非线性的 Bilinear MLPs，从而能够更清晰地观察**权重对计算的贡献**。
- **ICLR 2025 论文洞察**：论文摘要指出，由于**逐元素非线性 (element-wise nonlinearities)** 增加了特征追踪的复杂性，对 MLP 的机械化理解通常难以实现。
   - Bilinear MLPs 可以通过**线性运算**来表达，有助于权重的分析，并通过特征分解 (eigendecomposition) 揭示低秩结构。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.08417">Bilinear MLPs enable weight-based mechanistic interpretability</a>: 对深度神经网络中 MLP 如何进行计算的机械化理解仍然难以捉摸。目前的可解释性工作可以从输入数据集的隐藏激活中提取特征，但通常...</li><li><a href="https://x.com/thomasdooms/status/1882763792125440497?t=hCnnOmSJQL4jD1kEYZXk8A&s=19">tdooms (@thomasdooms) 的推文</a>: 我们能从权重中理解神经网络吗？通常答案是否定的。MLP 的激活函数模糊了输入、输出和权重之间的关系。在我们新的 ICLR 2025 论文中...
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1332084212337217547)** (20 条消息🔥): 

> `Generating ice text with specific fonts, ControlNet for image generation, Adobe Firefly for custom text, Understanding image poisoning, Using sketch to image` 


- **使用自定义字体创建 AI 冰霜文字**：一位用户表达了对使用特定字体生成**冰霜文字 (ice text)** 的兴趣，并询问了使用 **Img2img** 的处理流程。
   - 另一位用户建议可以先给文字上冰霜色，然后在 **Img2img** 中使用**去噪设置 (denoise settings)** 来达到理想效果。
- **用于冰霜文字生成的 ControlNet 增强**：成员们讨论了使用 **ControlNet** 增强冰霜文字生成以获得更好效果的可能性。
   - 有人提到将输入平铺 (tiling) 到更高的分辨率有助于获得更好的输出。
- **使用 Adobe Firefly 创建文本**：有人建议使用 **Adobe Firefly**，因为如果用户拥有 Adobe 许可证，它可以满足用户的特定需求。
   - 这被认为是替代涉及 Inkscape 等其他软件的复杂方法的一种选择。
- **识别被投毒的图像**：有人提出了如何判断图像是否**被投毒 (poisoned)** 的问题。
   - 成员们幽默地建议将“舔一舔”和“闻一闻”作为评估图像安全性的轻松方法。
- **使用草图转图像 (Sketch to Image)**：一位用户询问了 **sketch to image** 的功能，寻求使用指导。
   - 关于针对特定长宽比（如 **16:9**）优化模型的话题也被提及，表明大家对定制化图像生成设置有共同兴趣。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1332092450948513843)** (14 messages🔥): 

> `ILP for View Simplification, Mask Representation Challenges, Multi Merge Search Patterns, Three View Simplification Approaches, Stride Alignment in Merges` 


- **用于 View 简化的 ILP 方法**：在 [Pull Request #8736](https://github.com/tinygrad/tinygrad/pull/8736) 中分享了一个使用 **ILP** 实现 View 对相加的概念验证，初步结果显示未命中率为 **0.45%**。
   - 讨论强调了使用逻辑除数（logical divisors）使多重合并（multi merges）形式化的潜力，以及在处理可变 Stride 时面临的挑战。
- **Mask 表示的挑战**：关于 Mask 是否能增强合并能力的辩论出现，观点表明 Mask 可能无法普遍适用于所有情况。
   - 澄清指出，虽然 Mask 可能会为某些合并创造机会，但它们很复杂，可能无法像之前认为的那样支持所有组合。
- **搜索多重合并模式**：一名成员提议通过使用 **v1=[2,4]** 和 **v2=[3,6]** 等示例评估 Offset 和 Stride，从而使多重合并方法形式化。
   - 他们指出，在分析公约数时，这种方法可能会产生一种通用的搜索模式。
- **探索三视图简化**：有人思考将简化从 **两个 View 扩展到三个** 或更多，但由于复杂性，对 **3 -> 2** 视图方法持怀疑态度。
   - 挑战源于在可变 Stride 的情况下维持合并后的 View，这使 ILP 公式化变得复杂。
- **Stride 对齐的可能性**：最近的讨论表明 Stride 对齐可能会降低合并 View 的复杂性，但关于 Stride 不对齐的假设被认为是有问题的。
   - 承认由于 Stride 关系的计算错误，之前的方法可能会忽略许多潜在的合并。



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/8736">Complete view pair add using ILP (draft) by eliotgolding · Pull Request #8736 · tinygrad/tinygrad</a>: Proof of concept, uses scipy&amp;#39;s solver.$ python test/external/fuzz_view_add_completeness.pyMissed adds: 5/1109 0.45%$ NEWADD=1 python test/external/fuzz_view_add_completeness.pyMissed adds: ...

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1332312476871692330)** (10 messages🔥): 

> `Running on Windows, Windows Subsystem for Linux (WSL), Regex for Data Cleaning, Non-Supported Features in Windows, Issues with Triton and Xformers` 


- **在 Windows 上运行存在限制**：虽然可以在 Windows 上运行该软件，但一名成员强调这存在限制，例如无法利用 Triton 内核。
   - 提出的一种替代方案是使用 Windows Subsystem for Linux (WSL)，它可以提供更流畅的代码编写体验。
- **WSL 让开发更轻松**：一名成员建议设置 WSL，称其可以在不到一小时内完成，并能提升训练性能。
   - 他们强调，与直接使用 Windows 相比，这种设置可以显著减少编码过程中的困难。
- **分享用于数据清洗的 Regex 表达式**：一名成员分享了一个正则表达式 `[^	
\x20-\x7E]+`，它对于通过识别问题字符来清洗杂乱的数据集非常有用。
   - 另一名成员对该表达式进行了详细解析，解释了其组成部分以及它匹配的字符类型，特别是非打印字符和非 ASCII 字符。
- **对 Windows 上错误的担忧**：一名成员对关于 Windows 或 MacOS 不支持重定向（redirects）的警告表示担忧。
   - 这引发了关于可能产生的影响以及用户是否应该为此担心的讨论。
- **Triton 和 Xformers 兼容性问题**：由于 Triton 和 Xformers 与 Windows 不兼容，在使 **unsloth** 或 **axolotl** 正常工作方面出现了问题。
   - 有建议查看 GitHub 仓库以寻求潜在解决方案，这表明对在 Windows 上运行的用户仍有持续支持。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1332102115090239570)** (5 messages): 

> `First Lecture Announcement, LLM Agents Acceptance` 


- **首场讲座定于下周一！**：经成员确认，首场讲座定于 **太平洋时间 27 日周一下午 4 点**。
   - 包含更多细节的电子邮件通知将很快发布。
- **LLM Agent 的表现令人印象深刻**：有人指出，如果你是一个 **LLM Agent**，能通过这门课程将是非常令人印象深刻的。
   - 这突显了课程的挑战性以及对 LLM Agent 的高期望。


  

---

### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1332220640085934080)** (5 messages): 

> `Discord 中报告的诈骗、Nebius AI 多节点训练、SLURM 和 Torch Elastic 挑战` 


- **Discord 充斥诈骗报告**：用户报告了在多个位置发布的诈骗信息，特别是 [这个频道](https://discord.com/channels/1104757954588196865/1110594519226925137/1332219195210989578)。*谢谢！*，一位用户对这些警报表示感谢。
   - 这提高了社区对持续存在的诈骗消息问题的认识。
- **关于使用 Nebius AI 进行训练的咨询**：一位成员寻求关于在多节点设置中使用 **Nebius AI** 进行训练的见解，寻找共享经验。讨论强调了从熟悉该平台的成员那里获得实用建议的需求。
- **在 SLURM 和 Torch Elastic 上的困扰**：一位用户对无法让 **SLURM** 和 **Torch Elastic** 正常工作表示沮丧，称这是一个重大挑战。另一位成员建议查看 SLURM 多节点指南的文档，认为尽管存在差异，许多原则仍然适用。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1332361294333870100)** (2 messages): 

> `Signature 定义问题、MATH 数据集替代方案` 


- **对 Signature 定义的困惑**：一位成员对定义 **Signature** 表示沮丧，指出他们没有收到预期的输出，且某些文件未能正确映射。
   - 他们注意到某些文件可能缺少源文件，这导致了困惑。
- **寻找 MATH 数据集的替代方案**：另一位成员提到尝试运行 **maths 示例**，但发现该数据集已被移除。
   - 他们分享了两个数据集的链接，包括 [MATH 数据集](https://huggingface.co/datasets/lighteval/MATH/viewer) 和 [Heuristics Math 数据集](https://huggingface.co/datasets/hendrycks/competition_math)，并请求推荐替代方案。



**提到的链接**：<a href="https://huggingface.co/datasets/hendrycks/competition_math">hendrycks/competition_math · Datasets at Hugging Face</a>：未找到描述

  

---


---


---


---


---


---


{% else %}


> 完整的各频道详细分析已针对电子邮件进行了截断。
> 
> 如果您想查看完整的详细分析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}