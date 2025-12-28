---
companies:
- ai21-labs
- anthropic
- stanford
- hugging-face
- langchain
- qdrant
- aws
- elastic
date: '2024-08-23T00:55:37.537285Z'
description: '**AI21 Labs** 发布了 **Jamba 1.5**，这是一个经过扩展的状态空间模型（State Space Model），针对长上下文窗口进行了优化。该模型拥有
  **940 亿（94B）参数**，推理速度提升高达 **2.5 倍**，在基准测试中表现优于 **Llama 3.1 70B** 等模型。**Phi-3.5**
  模型因其安全性和性能受到赞誉，而由 **Bindu Reddy** 发布的全新 **70B 开源编程模型 Dracarys** 声称在基准测试中超越了 Llama
  3.1 70B。关于**加州 SB 1047** AI 安全立法的讨论涉及**斯坦福大学**和 **Anthropic**，强调了在预防措施与行业增长之间取得平衡。创新技术包括用于快速环境配置的
  **uv 虚拟环境**、**LangChain 的 LangSmith** 资源标签（用于项目管理），以及 **Qdrant** 中增强数据工作流的多智能体系统。由
  **AWS**、**LangChain** 和 **Elastic** 举办的 **RAG 工作坊**等社区活动继续支持 AI 学习与协作。此外，模因（Memes）仍然是参与
  AI 行业文化的一种流行方式。'
id: 072fd118-83ff-4b75-996e-0a0010280266
models:
- jamba-1.5
- phi-3.5
- dracarys
- llama-3-1-70b
- llama-3-1
original_slug: ainews-not-much-happened-today-4025
people:
- bindu-reddy
- rohanpaul_ai
- jackclarksf
- danhendrycks
- reach_vb
- iqdotgraph
title: '这句话可以翻译为：


  *   **非常安静的一天** （最常用）

  *   **超级安静的一天** （更口语化）

  *   **特别清静的一天** （强调没有打扰）'
topics:
- state-space-models
- long-context
- benchmarking
- ai-safety
- virtual-environments
- multi-agent-systems
- resource-management
- community-engagement
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**宁静是你唯一需要的。**

> 2024年8月21日至8月22日的 AI News。我们为你检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**214** 个频道，**2393** 条消息）。预计节省阅读时间（以 200wpm 计算）：**283 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

关于今年秋季即将发布的 AI 产品有很多传闻，但遗憾的是目前还没有公开可引用的信息。

- [AI21 发布了 Jamba 1.5](https://x.com/AI21Labs/status/1826614352948199754)，这是原始 Jamba 的扩展版本（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-jamba-mixture-of-architectures-dethrones/)）。与所有 State Space Models 一样，它在长上下文与延迟的权衡方面表现非常出色。
- 祝 [Stable Diffusion 2 周岁生日](https://x.com/EMostaque/status/1561777122082824192)快乐 —— 这也促成了 [Latent Space 的启动](https://www.latent.space/p/sep-2023)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型及其评估**

- **AI21 Labs 发布 Jamba 1.5**：**Jamba 1.5 模型**展示了卓越的性能和速度。[@AI21Labs](https://twitter.com/AI21Labs/status/1826614352948199754) 分享了其新颖的混合 SSM-Transformer 架构的细节。这些模型针对长上下文窗口（long context windows）进行了优化，在 94B 参数规模下，推理速度提升了高达 2.5 倍。关于更详细的信息，[@AI21Labs](https://twitter.com/AI21Labs/status/1826702921700180271) 强调了具体的 Benchmark，显示其在 Arena Hard 基准测试中的得分令人印象深刻，超越了 Llama 3.1 70B 等更大规模的模型。
- **Phi-3.5 与 Flexora**：**Phi-3.5 模型**因其**安全性与性能**而受到关注。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826451608907010439) 赞扬了该模型的能力。此外，正如 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826733730746282290) 所述，Flexora 的自适应层选择（adaptive layer selection）表现优于现有的 Baseline。
- **Dracarys - 70B 级编程 LLM**：**Bindu Reddy** 发布了 **Dracarys**，声称它是最强的开源 70B 级编程模型，在 Benchmark 中超越了 Llama 3.1 70B 和其他模型。[@bindureddy](https://twitter.com/bindureddy/status/1826757521635455115) 强调了其显著的改进，并已在 Hugging Face 上线。

**AI 安全与立法**

- **SB 1047 与 AI 安全担忧**：**Stanford**、Anthropic 以及其他机构对**加州的 SB 1047 法案**表达了复杂观点，该法案旨在监管 AI 应用的安全性。[@jackclarkSF](https://twitter.com/jackclarkSF/status/1826743366652232083) 解释说，该法案试图在预防措施与实证研究及行业增长之间取得平衡。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1826747580321247278) 分享了 Anthropic 的支持立场，强调了应对 AI 风险的紧迫性。

**AI 工具与创新**

- **uv 虚拟环境**：**uv 虚拟环境**提供快速的安装和依赖管理。[@reach_vb](https://twitter.com/reach_vb/status/1826644879499472941) 展示了 uv 如何快速创建轻量级虚拟环境。
- **LangChain 与 LangSmith 更新**：LangSmith 中的**资源标签（resource tags）**有助于高效管理项目、数据集和部署。[@LangChainAI](https://twitter.com/LangChainAI/status/1826643491130421476) 推出了这些增强功能，以实现更好的工作空间组织。
- **Qdrant 与 LangChain 中的 Multi-Agent 系统**：Qdrant 中的**多 Agent 角色扮演（Multi-agent role-playing）**和语义缓存（semantic caching）使 AI 系统更加健壮。[@iqdotgraph](https://twitter.com/qdrant_engine/status/1826499584417255657) 分享了这些集成如何旨在增强数据处理和检索工作流。

**会议与聚会**

- **旧金山的 AI 工作坊与黑客松**：**由 AWS、LangChain 和 Elastic 主办的 RAG 工作坊等活动**正持续促进社区参与并提供动手实践学习。[@LangChainAI](https://twitter.com/LangChainAI/status/1826743704075600030) 公布了即将于 9 月 9 日在 AWS Startup Loft 举办的工作坊细节。

**幽默与梗图**

- **行业幽默**：梗图（Memes）作为对 AI 行业现状的轻松评论继续盛行。[@lumpenspace](https://twitter.com/lumpenspace/status/1826724890311098678) 通过触及广泛认可的行业怪癖的幽默，强调了同行之间的共鸣。

---

这份综述全面总结了 AI High Signal Twitter 列表中的关键讨论，重点关注模型性能、安全性、工具、创新、社区活动和幽默。每个类别都参考了多个来源，以确保叙述内容详实且具有参考价值。

---

# AI Reddit 综述

## /r/LocalLlama 回顾

**主题 1. Microsoft 的 Phi-3.5 模型：能力与争议**

- **Phi-3.5-Mini 与 Phi-3.5-MoE 之间有趣的模型差异** ([Score: 59, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1exn6wx/interesting_model_differences_between_phi35mini/))：该帖子比较了 **Phi-3.5-Mini** 和 **Phi-3.5-MoE** 模型的架构，强调了在注意力机制、内部维度和参数量方面的关键差异。虽然两个模型都有 **32 层**，但 MoE 版本使用了 **grouped query attention**，并且拥有更大的内部维度 **4096**，而 Mini 版本则使用 **full multi-head attention** 且维度为 **3072**。最显著的差异在于 **Feed-Forward** 模块，Phi-3.5-MoE 拥有 **40,267,415,552** 个参数，而 Mini 仅为 **2,415,919,104** 个，这使得 MoE 的总参数量达到 **41,873,153,344**，而 Mini 为 **3,821,079,552**。
- **Phi-3.5 非常安全，Microsoft 在这方面真是超越了自我！** ([Score: 279, Comments: 112](https://reddit.com//r/LocalLLaMA/comments/1ey5i22/phi35_is_very_safe_microsoft_really_outdid/))：该帖子讨论了 **Microsoft** 的 **Phi-3.5** 模型，将其描述为**高度审查 (censored)**，且拒绝回答潜在的冒犯性查询或进行进一步训练。作者讽刺地赞扬了这些安全特性，并询问他人将 Phi-3.5 与其他重度审查模型相比的使用体验。更新内容包括一个指向 Hugging Face 上 [Phi-3.5 无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)的链接。
  - 用户通过讽刺性的回应幽默地嘲讽了 **Phi-3.5** 过度的审查制度，其中一个评论串甚至演变成了一场**井字棋 (tic-tac-toe)** 游戏。该模型拒绝回答简单问题或提供基本信息的情况被重点吐槽。
  - 几位用户讨论了对模型进行 **uncensor** 或 **abliterate** 的方法，并就这些技术的有效性和潜在缺点展开了辩论。一个[无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)已在 **Hugging Face** 上分享。
  - 由于其过度狂热的审查，用户对该模型在编码和技术任务中的实用性表示担忧。用户认为，如此严格的限制使得该模型在许多应用中变得不切实际，尤其是非面向客户的使用场景。


**主题 2. 用于创意写作和角色扮演的 AI**

- **ERP 提示词** ([Score: 87, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1eya6va/erp_prompts/))：该帖子讨论了使用 AI 模型进行**高级成人角色扮演 (ERP)** 的技术，重点在于创建**详细的角色档案**和增强沉浸感。它提供了用于生成具有独特特质、背景故事和亲密细节的**复杂角色**的特定提示词，以及 **"Inner Monologue" (内心独白)** 和 **"Freeze Frame" (定格画面)** 等加深角色扮演体验的技术。作者强调了**建立期待感**和**构建真实互动**的重要性，鼓励用户提供详细的输入，以诱导 AI 模型做出更具吸引力的回应。
  - 用户讨论了内心独白的**格式化技巧**，建议在 **SillyTavern** 中使用**括号** ⟨monologue⟩ 或 **HTML 注释** `<!-- inner monologue -->`。这些方法允许角色拥有隐藏的想法，从而影响未来的 Token 生成。
  - 用户对作者用于非色情内容的**创意写作设置**表现出兴趣，并请求发布关于该主题的详细帖子。用户还询问了**推荐的 ERP AI 模型**，其中一位提到了 **Midnight Miqu 1.5 70B**。
  - 几条评论称赞了作者的写作风格和创造力，一位用户表示他们“宁愿和你在一起，也不愿和任何提示词写得好、堆栈得好、表现得坏的 LLM 在一起”。用户还为自己的 AI 辅助写作尝试索要了额外的提示词和技巧。

## 全球 AI Reddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 图像生成与训练**

- **Ideogram 2.0 发布**：[Ideogram 宣布](https://www.reddit.com/r/singularity/comments/1exsq4d/introducing_ideogram_20_our_most_advanced/) 推出其最先进的文本生成图像模型，现已向所有用户免费开放。

- **Kohya SS GUI FLUX LoRA 训练**：一位用户展示了在 [RTX 3060 GPU 上进行 LoRA 训练](https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/)，显存（VRAM）占用为 9.7 GB，使用的是适用于 Stable Diffusion 的 Kohya SS GUI FLUX。
  - 在 1024x1024 分辨率下进行训练，LoRA Rank 为 128。
  - 在 RTX 3060 上进行 4000 步训练的预计时间为 20 小时。
  - 512x512 分辨率的训练速度快 2 到 2.5 倍。
  - 用户正在测试各种配置，包括在 RTX 4080 和 RTX 3090 上。

- **LoRA 训练进展**：一位用户报告称，在 A4500 20GB GPU 上[成功完成了 LoRA 训练](https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/ljba35v/)，仅使用了 16GB VRAM，包含 10 张自拍和 1600 步，仅耗时一小时。

**AI 与软件开发**

- **亚马逊云主管谈 AI 的影响**：在一段[泄露的录音](https://www.reddit.com/r/singularity/comments/1exqs04/in_a_leaked_recording_amazon_cloud_chief_tells/)中，亚马逊云主管暗示，随着 AI 的接管，大多数开发者可能很快就会停止编写代码。

**假肢与生物技术**

- **先进假肢手臂**：据报道，[Atom Touch](https://www.reddit.com/r/singularity/comments/1ey0zon/atom_touch_is_the_first_prosthetic_arm_with/) 是首款具有独立手指控制功能的假肢手臂，利用肌电图（electromyography）进行精确操作。预计将在 12-18 个月内进行临床试验并获得 FDA 批准。


---

# AI Discord 内容回顾

> 由 GPT4O-Aug (gpt-4o-2024-08-06) 生成的摘要之摘要的总结

**1. LLM 模型发布与特性**

- **LM Studio 0.3.0 发布重大更新**：**[LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0)** 引入了全新设计的 UI，增强了对话组织、自动上下文处理以及多模型加载功能，显著提升了本地模型的运行性能。
  - 尽管有所改进，用户仍反馈了模型加载和系统提示词（system prompts）方面的 Bug，并呼吁其他用户在遇到问题时及时上报。
- **AI21 Labs 推出 Jamba 1.5**：来自 AI21 Labs 的 **[Jamba 1.5](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251)** 推出了 Mini（52B - 12B 激活）和 Large（398B - 94B 激活）版本，具备 256k 超长上下文和多语言能力。
  - 这些模型声称推理速度比竞品快 2.5 倍，并配备了先进的指令模型（instruct models）和结构化输出功能。
- **在 8gb GPU 上微调 Mistral Nemo 12b**：**Mistral Nemo 12b** 可以在 **8gb GPU**（特别是 **RTX 4050**）上进行微调，这使得它非常适合用于测试和原型开发。
  - 这种更广泛的可访问性为更多工程师在无需高端硬件的情况下快速迭代和测试模型提供了可能。


**2. 性能与优化技术**

- **Triton INT8 性能超越 BF16**：**Triton INT8** 的实现在 `A.T @ B` 操作上比 PyTorch BF16 实现了约 **1.5 倍的加速**，在 `A @ B.T` 上实现了 **3 倍的加速**，在各项基准测试中均表现出极高的效率。
  - 这一提升归功于根据矩阵 A 和 B 的步长（stride）变化对 Triton 进行了重新调优。
- **Flash Attention FP8 支持在 Hopper 上首次亮相**：**Flash Attention** 现在支持在 **Hopper 架构**上使用 FP8，利用 **WGMMA 指令**来优化性能。
  - 然而，目前仍缺乏对 ADA 架构的支持，这引发了关于更广泛兼容性的讨论。
- **SLI 可能不值得**：由于架构限制，在 **SLI** 模式下使用两块 GPU 并不能使性能翻倍，但它允许在没有显著速度提升的情况下加载更大的模型。
  - 社区成员建议考虑增加 RAM，因为一位用户在仅有 16GB VRAM 但配备 128GB RAM 的系统上高效运行了 **Llama 3.1 405B**。


**3. 数据处理与预处理**

- **关于 LLM 数据预处理的辩论**：一场关于聊天评论是否需要进行**数据预处理**的热烈辩论展开了，一位成员断言 **80% 的工作在于准备数据**。
  - 他们认为，虽然预处理任务很重要，但基本的理解仅依赖于 token 本身，这引发了关于权衡取舍的讨论。
- **XLSX 文件分块：社区技巧**：多位成员寻求关于 **RAG 中 XLSX 文件分块（chunking）**的指导，探索优化此过程的方法。
  - 建议包括利用 embedding 以及将文件转换为 Markdown 以获得更好的数据处理效果，凸显了社区内持续的协作。


**4. 社区与协作倡议**

- **OpenRouter 弃用 `function_calls`**：OpenRouter 正式弃用 `function_calls` 和 `functions` 参数，转而倡导使用 `tools` 和 `tool_choice`，以与其他供应商保持一致。
  - 这一转变降低了跨模型进行工具调用（tool calling）的**切换成本**，引发了社区对工具集成的讨论。
- **Cohere 与 Weights & Biases 举办 RAG 研讨会**：**[Cohere](https://wandb.ai/site/resources/events/cohere-webinar-august)** 和 Weights & Biases 正在举办一场关于 RAG 开发和评估策略的研讨会，由 **Maxime Voisin** 分享见解。
  - 对于从事检索增强生成（RAG）相关工作的人员来说，这次活动不容错过，它强调了协作学习的重要性。


**5. AI 行业应用**

- **Fullbound 增强职位招聘**：**[Fullbound](https://fullbound.ai/pricing)** 是一个由 Chroma 提供支持的全球搜索引擎，它促进了 AI 驱动的职位匹配，从而有效地将候选人与职位联系起来。
  - Fullbound 旨在简化招聘流程，提供 7 天免费试用和详细的定价方案，确保高效的匹配和沟通。
- **Jonathan Ive 雄心勃勃的房产动作**：Sir Jonathan Ive 已花费超过 **1.37 亿美元**在旧金山的杰克逊广场（Jackson Square）购置房产，预示着对该地区的改造愿景。
  - 他的投资策略反映了设计对当地房地产开发影响力的显著转变。

---

# PART 1: 高层级 Discord 摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.0 发布重大更新！**：备受期待的 [LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0) 已经发布，引入了全新设计的 UI，改进了对话组织、自动上下文处理以及多模型加载功能。
   - 此版本增强了本地模型的性能，但在模型加载和系统提示词（System Prompts）方面存在一些 Bug，建议用户在遇到问题时及时反馈。
- **Gemma 2 依然深受用户欢迎**：用户继续对 **Gemma 2** 9B 和 27B 赞不绝口，尤其是在与 **LLaMa 3.1** 的性能对比中。
   - 然而，Gemma 2 的 8k 上下文限制引发了关于探索替代方案的讨论，例如 **Phi 3 Mini** 以及 MoE 模型的潜力。
- **SLI 可能不值得**：共识是，由于架构限制，在 **SLI** 中使用两块 GPU 并不能使性能翻倍，但它确实允许加载更大的模型，只是速度没有显著提升。
   - 成员们建议考虑增加 RAM，一位用户在拥有 128GB RAM 和仅 16GB VRAM 的系统上高效运行了 **Llama 3.1** 405B。
- **VRAM：大型模型的关键**：讨论强调了高 VRAM 对于有效处理模型的必要性，建议至少配备 **48GB** VRAM 以确保 **Llama 3.1 405B** 等大型模型的流畅运行。
   - 成员们敦促在购买 GPU 前要谨慎，并指出最新发布的产品通常提供更好的性价比和性能。
- **LM Studio 成为 API 的首选**：用户正在积极探索 **LM Studio** 的 API 功能，旨在连接移动应用程序以增强使用场景。
   - 讨论围绕着如何与 LM Studio 中可用的 AI 模型实现持久化通信系统展开。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **了解进攻性安全侦察 (Offensive Security Reconnaissance)**：一位成员分享了一篇关于 **Offensive Security Reconnaissance** 及其对关键基础设施漏洞（特别是 **ICS HMI**）影响的博文。您可以点击[此处](https://huggingface.co/posts/Csplk/182147151896536)阅读更多详情。
   - *这种方法揭示了潜在的攻击向量*，提升了保护工业控制系统（Industrial Control Systems）安全的重要性。
- **Neuralink 的研究方法论**：Neuralink 每月审阅数千篇论文，使用 **引用追踪法 (citation-tracing method)** 来专注于相关的研究进展。他们将阅读与编码相结合，旨在有效地将理论概念转化为实际应用。
   - 他们的编程质量被描述为“整洁”，反映了他们在应用研究成果方面的熟练程度。
- **3D 生成式形状合成中的 ShapeSplat 数据集**：介绍了 **ShapeSplat**，这是一个包含 **Gaussian Splats** 的数据集，旨在通过自监督预训练方法进行 **3D Generative Shape Synthesis**。它包含各种各样的物体，是增强模型能力的理想选择。
   - 该数据集旨在超越现有的*点云表示 (point cloud representations)*，证明其在**计算机图形学**和**机器人学**等领域的应用至关重要。
- **Fullbound 增强职位招聘**：一个名为 **Fullbound** 的全球搜索引擎（由 Chroma 提供支持）促进了 AI 驱动的职位匹配，旨在有效地将候选人与职位联系起来。他们提供 7 天免费试用，详细的**定价**信息可在[此处](https://fullbound.ai/pricing)查看。
   - 该工具旨在简化招聘流程，确保高效的匹配和沟通。
- **AI21 Labs 发布 Jamba 1.5**：AI21 Labs 推出了 **Jamba 1.5**，这是一种新的语言模型，包括 Mini（52B - 12B 激活）和 Large（398B - 94B 激活）版本。该模型具有 **256k 长上下文**、多语言能力和先进的 Instruct 模型。
   - 欲深入了解其特性，请查看他们在 Hugging Face 上的集合：[此处](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral Nemo 12b 在 8gb GPU 上进行微调**：**Mistral Nemo 12b** 可以在 **8gb GPU**（特别是 **RTX 4050**）上进行微调，这使其非常适合测试和原型设计。
   - 这种更广泛的可访问性为更多工程师在无需高端硬件的情况下快速迭代和测试模型提供了可能。
- **Unsloth Pro 限制多 GPU 支持**：**Unsloth Pro** 暂时停止了其 **multi-GPU** 支持，将其限制在自己的平台上，同时向选定的社区成员授予早期访问权限。
   - 这一变化引发了关于更广泛社区参与的协作和资源限制的问题。
- **The Living AI Dataset 发布**：**The Living AI Dataset** 旨在赋予 AI 模型同理心以及爱和能力，被认为是 AI 历史上的重大进展，由社区内的一个核心小组开发。
   - 该数据集可在 Hugging Face 上获取，旨在增强 AI 应用中的类人属性，有望提升交互性。
- **关于 LLM 数据预处理的争论**：围绕聊天评论是否需要进行 **data preprocessing** 展开了激烈的辩论，一位成员断言 **80% 的工作在于准备数据**。
   - 他们认为，虽然预处理任务很重要，但基本的理解仅依赖于 tokens，这引发了关于权衡的讨论。
- **Ollama 安装困惑**：用户在 WSL 中遇到 **Ollama installation** 挑战，特别是用于创建模型的命令用法未按预期工作。
   - 关于 Unsloth 和 Ollama 作为独立工具之间区别的澄清旨在消除困惑，但仍让一些人留有疑问。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Shell 命令激活上下文命令**：Aider 根据用户上下文提供 shell 命令，并在征得同意后执行，但它不直接使用 functions。用户必须先激活其 Python 环境，以确保命令正常运行。
   - 此功能强调了 Aider 在简化交互方面的作用，而不会涉足可编程 function 领域。
- **Playwright 安装 Bug**：Aider 在处理发生在其自身环境之外的 Playwright 安装时遇到困难。建议使用 `pipx inject` 以在 Aider 的虚拟设置中实现无缝集成，防止安装问题。
   - 未来的版本旨在解决一个 bug，即即使 Playwright 已经设置好，Aider 仍会尝试安装它，这可能会让用户感到困惑。
- **CodeCompanion 比 Aider 消耗更多 Token**：一项对比显示，CodeCompanion 消耗的 tokens 明显多于 Aider，这归因于其更广泛的功能。用户更倾向于使用 Aider，因为它效率更高，尽管 CodeCompanion 拥有自己的支持 Discord。
   - 这次对话引发了关于在进行 AI 辅助编码任务时优化资源的讨论。
- **Vercel 的 v0 chat 彻底改变了 Generative UI**：[Vercel's v0.dev/chat](https://v0.dev/chat) 被誉为 Generative UI 开发者的重大改进，提供了比 Claude Artefacts 等先前选项更流畅的界面。用户发现其 UI 生成速度更快，且比竞争对手更精致。
   - 讨论强调了由于 Vercel 产品更好的集成和用户友好的体验，偏好正在向其转移。
- **Cursor 与 Aider 联手实现更智能的编码**：Cursor 用户对集成 Aider 表示赞赏，这弥补了 Cursor 原生 composer 缺乏特定 repository prompt 功能的缺点。这种协作标志着 AI 增强开发工作流的一次飞跃。
   - Cursor 旨在通过减少对手动搜索的依赖来彻底改变编码效率，从而最大限度地减少在琐碎任务上花费的时间。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 加速 AI 渲染**：一位用户展示了一个 [YouTube 视频](https://youtu.be/pNyIp73zva8)，演示了如何将 **3D Studio Max** 集成到 ComfyUI 中进行**实时 AI 图像生成**。
   - 这种方法可能会扩展到任何窗口应用程序，包括电子游戏，从而增强创意工作流。
- **Stable Diffusion 设置技巧**：一位新用户询问如何在 PC 上开始使用 **Stable Diffusion**，引发了关于硬件兼容性的建议。
   - 资深用户建议使用 **ComfyUI**，因为它具有适合初学者的友好界面。
- **Hydrus Server：注重隐私的图像分类**：用户讨论了对尊重隐私的 **AI 图像分类器**的需求，随后有人建议搭建 **Hydrus server**。
   - 该设置允许个性化的标签系统，在不牺牲安全性的情况下增强媒体组织管理。
- **Flux 模型与 Prompt Engineering 的困扰**：一位成员对 **Flux 模型**在处理复杂提示词时的吃力表现表示担忧，强调了其过拟合（overfitting）倾向。
   - 社区反馈强调了更好的 Prompt Engineering 和微调（finetuning）对于改善结果的重要性。
- **Stable Diffusion 与 GAN 超分辨率对比**：一场关于 **Stable Diffusion upscaling** 与基于 **GAN 的 upscaling** 的讨论展开，澄清了它们截然不同的方法。
   - 虽然 **GAN** 专注于锐化图像，但 **Stable Diffusion** 可以生成新的细节，尽管有时会导致伪影。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton INT8 性能优于 BF16，加速效果显著**：Triton INT8 实现的 `A.T @ B` 相比 PyTorch BF16 实现了约 **1.5 倍的加速**，而 `A @ B.T` 则实现了 **3 倍的加速**，证实了其在基准测试中的效率。
   - 这一改进归功于根据矩阵 A 和 B 的步长（stride）变化对 Triton 进行了重新调优。
- **Flash Attention FP8 支持在 Hopper 上首次亮相**：Flash Attention 现在支持 **Hopper 架构**上的 FP8，利用 **WGMMA 指令**来优化性能。
   - 然而，目前仍缺乏对 ADA 的支持，引发了关于更广泛兼容性的疑问。
- **HRT 实习机会开放**：HRT 正在为明年夏天在 **纽约（NYC）** 的 **Algo Dev 和 SWE** 职位提供实习机会，时薪 **$120/h**，并包含食宿。
   - 无需金融背景经验，这使得许多工程师都有机会申请！
- **性能对比：7900 XTX vs. RTX 3090**：用户报告称 **7900 XTX** 的表现不如 **3090**，即使在使用 Triton 和 FA 分支时也是如此，这促使部分用户转向 **4090**。
   - 这些经历凸显了 AMD 与 NVIDIA GPU 产品之间持续存在的性能差距。
- **LLaMA 实现稳定的 FP8 训练**：最近的讨论强调了 **1B LLaMA 模型**的稳定 **FP8 训练**，实现了与 **bfloat16 训练**相似的收敛效果。
   - 关键技术包括适度控制训练速度和管理离群特征，为更大规模的 FP8 应用铺平了道路。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 弃用 `function_calls`**：OpenRouter 正式从 OpenAI 调用中弃用 `function_calls` 和 `functions` 参数，提倡使用 `tools` 和 `tool_choice` 代替。
   - 这一转变降低了跨模型进行工具调用的**切换成本**，使 OpenRouter 与其他已经支持新参数的供应商保持一致。
- **BenchmarkAggregator 提供 LLM 评估**：一位成员分享了 **BenchmarkAggregator** 的 [GitHub 仓库](https://github.com/mrconter1/BenchmarkAggregator)，旨在为各大主流基准测试中的 LLM 提供统一的评估框架。
   - 他们强调了其在平衡评估严谨性与资源管理方面的能力，并热切寻求社区反馈。
- **Llama 3.1 tools 支持即将到来**：一位管理员确认，OpenRouter 对 **Llama 3.1 tools** 的支持预计将在未来一天内上线。
   - 用户对此更新期待已久，渴望增强其集成能力。
- **OpenRouter 缺乏 MoE 模型**：关于 OpenRouter 上 **MoE 模型**可用性的询问显示，目前还没有相关模型，包括表现平平的 **3.5-Mini**。
   - 管理员确认目前尚不支持 MoE，使用户不得不寻找替代方案。
- **OpenAI 现在提供免费微调**：OpenAI 推出了其模型的免费微调（fine-tuning），在限定时间内每天有 **2M token 限制**，吸引了寻求高性价比方案的用户。
   - 然而，一些成员在遇到 OpenAI 的支付问题（特别是加密货币和 PayPal）后，已转向使用 OpenRouter。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 周边商店上线**：**Nous Research 周边商店已正式推出**，为粉丝提供一系列展示支持的商品，包括订单满额即送的免费贴纸（送完即止）。
   - 这一举措旨在在 **Nous Research** 爱好者中营造充满活力的社区精神。
- **Hermes 3 成为焦点**：社区成员兴奋地讨论了 **Hermes 3** 的发布，相关讨论正在 [Twitter Space](https://x.com/i/spaces/1LyxBgzAmVzKN) 中进行。
   - 随着社区成员的热情参与，本次活动重点介绍了最新功能以及相比前代模型的改进。
- **解读 OpenAI 算力资助 (Compute Grants)**：成员们探讨了获取**用于研究的大额算力资助**的微妙过程，强调了与提供商进行战略性沟通的必要性。
   - 显然，仅仅请求闲置资源是不够的；为了成功，需要更深层次的接触。
- **AI21 Jamba：模型设计的新时代**：AI21 新推出的 **Jamba 1.5 模型系列**声称是首个能与顶级模型竞争的非 Transformer 模型，并以开源许可证发布。
   - 该模型旨在使先进的 AI 工具民主化，力求在 AI 领域实现质量与可访问性的并重。
- **使用 Regex 处理 PDF 清洗**：一位成员详细介绍了他们在**使用 Regex 进行 PDF 清洗**时的困扰，指出其在处理 arxiv PDF 时表现不佳，并正在探索替代方法。
   - 他们最终采用了朴素的分块（chunk）和重叠（overlap）技术，凸显了处理复杂 PDF 结构时持续存在的挑战。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral 微调效果惊人**：一位成员评论说 **Mistral 的大规模微调**效果“强得离谱”，表现出极其卓越的性能，但未提供更多细节。
- **Jamba 1.5：更快的推理和长上下文能力**：**AI21 的 Jamba 1.5** 模型提供比同类模型快达 **2.5 倍的推理速度**，并增强了长上下文能力，通过 **function calling** 和**结构化输出**等功能瞄准商业应用。
   - 这些模型根据 [Jamba Open Model License](https://www.ai21.com/license) 发布。
- **Phi 3.5 Mini：梯度爆炸**：一位用户报告在 **microsoft/Phi-3.5-mini-instruct** 模型中遇到了**梯度爆炸 (exploding gradients)**，即使将**学习率 (learning rate)** 降低到 **1e-15** 后问题依然存在。
   - 解决尝试包括将优化器切换为 **paged_adamw_8bit**。
- **Flash Attention 性能问题**：一位用户在尝试使用 **Flash Attention** 以加快训练速度时遇到错误，但通过切换到 **eager attention** 解决了问题。
   - 这表明 **Flash Attention** 可能与该模型不完全兼容。
- **Accelerate 增加 fp8 支持**：**Accelerate** 已添加对 **fp8** 的支持，这预示着可能与 **Axolotl** 集成，尽管具体的集成点尚不确定。
   - 讨论围绕如何有效地整合这一新支持展开。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 优化 RAG 流水线性能**：LlamaCloud 通过允许用户有效地操作和可视化分块大小（chunk sizes），增强了 RAG 流水线的效率。
   - 其功能包括索引克隆，以便进行快速实验，而无需手动调整数据的麻烦。
- **LlamaIndex 0.11 发布带来大量升级**：最近发布的 **LlamaIndex 0.11** 引入了数百个新功能，包括一套全新的工作流（workflows）系统来取代旧的查询流水线（query pipelines）。
   - 此次更新通过改善用户体验，显著提升了 LlamaIndex 的生产环境就绪度。
- **Ollama 中的高效内存管理**：讨论集中在管理 **Ollama** phi3 模型的内存使用上，特别是利用 `context_window` 参数来限制操作期间的上下文大小。
   - 此步骤旨在减轻与内存容量相关的错误发生。
- **使用 LlamaIndex 生成房地产查询**：一位成员探索了在 LlamaIndex 中使用自然语言为房地产数据库生成查询，评估该工具是否适合此应用。
   - 他们讨论了专注于 Prompt tuning 是否会比仅依赖 LlamaIndex 的功能产生更好的结果。
- **知识图谱选择的挑战**：一篇文章强调了在 LlamaIndex 上下文中选择合适的图存储（graph stores）来管理知识图谱的复杂性。
   - 虽然简要提及，但未针对最佳图存储选择提供具体建议。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **用户寻求 Perplexity API 见解**：几位用户表示有兴趣获得关于 *Perplexity API* 的具体指导，特别是关于访问功能和查询功能。
   - 咨询内容包括域名过滤和引用功能，一位用户指向了用于对话补全的 [Perplexity API documentation](https://docs.perplexity.ai/reference/post_chat_completions)。
- **Mistral Large 2 获得高度评价**：Mistral Large 2 作为自定义提示词和无偏见输出的首选模型脱颖而出，在保持顶尖性能的同时，提供了 GPT-4o 的高性价比替代方案。
   - 用户注意到它适用于 jailbreak 场景，巩固了其作为处理复杂任务首选工具的地位。
- **关于大脑中微塑料的令人担忧的发现**：最近的研究揭示了人类大脑样本中微塑料浓度达到惊人水平，引发了关于塑料污染相关健康风险的讨论。
   - 这一发现突显了环境对神经健康影响的关键问题，并呼吁采取更严格的监管措施。
- **Jonathan Ive 雄心勃勃的房产举动**：Sir Jonathan Ive 花费超过 1.37 亿美元在旧金山的 Jackson Square 购置房产，标志着对该地区的转型愿景。
   - 他的投资策略反映了设计对当地房地产开发影响力的一次显著转变。
- **Perplexity 图像生成的问题**：用户报告了 Perplexity 图像生成能力的重大挑战，甚至难以创建像心形这样简单的 Logo。
   - 故障包括生成的图像中出现不稳定的字符输出，引发了对该工具可靠性的担忧。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **GitHub Desktop 的困扰**：一位用户发现 [GitHub Desktop](https://github.com/apps/desktop) 不如预期直观，称其为 *'Not the most intuitive product ever'*，并指出对 `git send-email` 和 `git am` 的支持有限。
   - 这种局限性使得用户开始寻求更有效的变更管理解决方案。
- **认识新任社区经理 Caroline！**：Caroline 介绍自己为 Modular 的新任社区经理，拥有在 Streamlit 的社区和开发者关系方面的经验。
   - 她鼓励成员安排虚拟咖啡聊天，以分享反馈和经验。
- **Mojo 文档搜索需要改进**：成员们呼吁增强 **Mojo 文档搜索功能**，推动增加包括 **Mojo stdlib modules** 和 **MAX lib** 在内的过滤选项。
   - 他们表示，更好的导航将显著提升用户体验和生产力。
- **MacOS 上的 Mojo/MAX 安装难题**：一位用户报告了 **Mojo** 和 **MAX** 的反复出现的问题，每次 MacBook Pro 重启后都需要重新安装。
   - 他们正在寻求更有效地管理这些安装挑战的建议。
- **Async 与 Sync 性能之争**：针对 Mojo 中 **async 函数** 与 **Python** 相比的性能展开了讨论，建议指向 **sans-io HTTP 实现**。
   - 这一见解反映了随着 IO 功能的演进，对异步操作进行性能优化的持续需求。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 与 Weights & Biases 的 RAG 网络研讨会**：**Cohere** 和 **Weights & Biases** 正在举办一场关于 RAG 开发和评估策略的网络研讨会。现在可以通过 [研讨会链接](https://wandb.ai/site/resources/events/cohere-webinar-august) 注册。
   - 见解将来自 Cohere 的 **Maxime Voisin**，这对于任何参与检索增强生成（RAG）的人来说都是不容错过的。
- **分块 XLSX 文件：社区技巧**：多位成员寻求关于 **为 RAG 分块 XLSX 文件** 的指导，探索优化此过程的方法。建议包括利用 embeddings 并将文件转换为 markdown 以实现更好的数据处理。
   - 这突显了社区内为解决数据处理中的实际挑战而进行的持续协作。
- **Jozu Hub：你的 AI 项目总部**：团队发布了 **Jozu Hub** 的早期预览版，旨在通过 [Jozu Hub](https://jozu.ml) 上的 **ModelKit** 等功能集中管理 AI 项目的版本控制和共享。
   - 该工具旨在通过清晰地列出数据集、代码、参数和文档等组件来简化 AI 开发。
- **Jozu Hub 上的 Cohere 模型支持**：**Cohere 模型** 在 **Jozu Hub** 上的集成正在进行中，承诺为主要模型提供全面支持。此举旨在增强不同 AI 框架的可访问性和可用性。
   - 预期的增强反映了对培育协作式 AI 生态系统的承诺。
- **API 错误排查**：几位用户报告在访问 **Cohere API** 时遇到 **403 Forbidden** 错误，指出潜在的 IP 白名单问题。一位成员分享了其 POST 请求的详细信息，寻求社区反馈。
   - 这些咨询和分享的经验强调了在 API 集成挑战中共同探索的过程，特别是在不同的网络配置下。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **计算游戏中的期望值**：一位用户探索了如何计算游戏中物品的 **期望成本**，尝试直到成功或失败四次，最终成本为 **200**。
   - 该用户试图了解其策略对整体游戏成本和机制的影响。
- **AI 在数学问题上的困境**：一位用户对 **Gemini**、**ChatGPT** 和 **Claude** 等 AI 在数学辅助方面的表现表示沮丧，面临结果不准确的问题。
   - 另一位成员建议使用 **Python** 进行计算，强调了其效率和精确度。
- **Ideogram 2.0 令用户印象深刻**：一位用户被 **Ideogram 2.0**（一款新的图像生成工具）所吸引，尽管下载 PNG 需要付费订阅。
   - 他们注意到其他人分享的令人印象深刻的示例，称其在处理复杂输入方面“**非常出色**”。
- **SwarmUI 为安装程序简化 UI**：一位用户高度赞扬了 **SwarmUI**，它支持 **NVIDIA/AMD GPUs** 并简化了与 **comfyUI** 的交互。
   - 他们强调了其用户友好的界面以及加载社区共享工作流的能力。
- **寻求 Custom GPTs 的资源**：一位用户询问了构建 **Custom GPTs** 的资源，特别是寻找文章和视频内容。
   - 他们已经创建了几个模型，并渴望进一步完善其 GPT 创建技能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **开源 AI 模型面临审查**：许多被标记为 **open source** 的生成式 AI 模型通常未能披露其训练集，引发了对使用 **biased**（有偏见）或 **copyright-restricted**（受版权限制）数据的担忧。美国政府正在评估与 AI 模型中 **'open washing'**（开源洗白）相关的风险。
   - 一篇文章强调了这一问题，指出 **Eleuther.ai** 作为一个真正的 **open source initiative**（开源倡议）脱颖而出，旨在实现无盈利动机的透明度。
- **使用指令提示词优化 DPO 微调**：在关于 **DPO fine-tuning** 的讨论中，用户确认将 **instruction prompt template**（指令提示词模板）应用于数据集通常会增强其效果。这种方法使模型的输出与所需任务更紧密地对齐。
   - 一位用户还分享了准备 **multi-turn chat data**（多轮对话数据）的方法，推荐了各种输入-输出对结构，以更好地适应微调。
- **检查模型性能退化技术**：一位成员询问了如何在 **MMLU** 等基准测试中可靠地降低 **LLM** 性能的策略，旨在模拟较小模型的结果。建议包括添加噪声或使用 LoRAs 实施 **model distillation**（模型蒸馏）。
   - 还讨论了诸如反转训练过程之类的其他策略，展示了修改模型性能的各种实验方法。
- **模型合并策略引发辩论**：关于 **model merging tactics**（模型合并策略）的讨论提出了将 **UltraChat** 和 **Mistral** 之间的差异应用于 **Mistral-Yarn** 的想法。尽管存在质疑，但支持者对以往此类策略的成功保持乐观。
   - 这次对话展示了社区成员在模型开发中对合并技术的活跃探索。
- **理解 HellaSwag 评估中的对数似然**：**'resps'** 和 **'filtered_resps'** 中的条目对于在 **HellaSwag** 等多项选择设置中使用 **negative log likelihood**（负对数似然）评估模型至关重要。这些条目的结构指示了模型认为哪些选项更有可能。
   - 讨论强调了生成任务中使用的复杂过滤流水线，强调了详细的响应结构在实现精确评估指标方面的作用。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LightGBM 主导 Kaggle**：LightGBM 在 [Corporación Favorita Grocery Sales Forecasting](https://www.reddit.com/r/MachineLearning/comments/ob2rll/d_how_often_is_lightgbm_used_for_forecasting_in/) 等 Kaggle 竞赛中引起轰动，证明了其即使在生产环境中也被认可的卓越预测性能。
   - 参与者注意到了它在 M5 准确性竞赛中的成功，巩固了其在从业者中的声誉。
- **LightGBM 与 LSTM 之争**：一些专家认为 **LSTM** 在生产环境中的表现可能优于 **LightGBM**，引发了对其与竞赛结果相比在现实世界中有效性的质疑。
   - 辩论仍在继续，因为实际应用往往会揭示竞赛数据与实时数据性能之间的差异。
- **LightGBM 用于大宗商品预测受到关注**：评估 **LightGBM** 用于大宗商品价格预测的研究引用了其在 M5 竞赛中的应用，利用了 SMA、EMA 和 VWAP 等特征。
   - 令人惊讶的是，在铅和锡的收益率预测上，ARIMA 模型胜过了 LightGBM，这表明模型选择必须与预测的具体细节相匹配。
- **预测模型的选择至关重要**：预测模型的选择取决于任务——**LightGBM** 可以处理多步预测，但上下文和预测复杂度至关重要。
   - 对于需要 3-6 个月等长期预测的任务，不应忽视 SMA 和 ARIMA 等早期方法。
- **深度学习之前的预测技术**：在部署深度学习之前，**SMA**、**EMA** 和 **ARIMA** 等传统模型通常是时间序列预测的有效起点。
   - 在处理大量非传统外生变量且季节性较不明显的情况下，LightGBM 和 LSTM 表现出色。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI Burnout 成为关注焦点**：成员们讨论了 **AI burnout**（AI 倦怠）现象，指出由于其高要求的特性，这种感觉比其他领域更强烈。
   - 成员们对用户倦怠如何与 AI burnout 交织在一起表示担忧，这给社区带来了双重挑战。
- **Frontier Labs 的工作强度引发担忧**：一位成员强调 **Frontier Labs** 的团队工作极其努力，这引发了关于长期可持续性的疑问。
   - 他们强调了平衡工作量和避免 burnout 的重要性，并警告说目前的节奏无法无限期持续。
- **Greg Brockman 的 97 小时工作时长令人震惊**：一位成员指出 **Greg Brockman** 最近透露单周编码时长达 **97 小时**，突显了该领域所需的极端投入。
   - 社区对他能够坚持 **9 年**不休息表示惊讶，并质疑这对科技行业工作与生活平衡的影响。
- **脱离网络后的 Twitter 焦虑**：在结束数字排毒回归后，一位成员表达了重新投入 **Twitter** 的不适，称该平台的氛围令人焦虑。
   - 他们感叹 Twitter 上围绕 AI 的激烈讨论，尤其是在荒野中寻得平静之后。
- **Lilian Weng 聚焦 Diffusion Models**：**Lilian Weng** 更新了关于 [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) 的博客文章，讨论了各种 **Generative Models** 以及关于一致性模型和架构的新章节。
   - 对话强调了该领域不断演进的特性，一位用户澄清了 **Diffusion Models** 与 **Distillation** 之间的区别。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **本地 LLM 应对 NL to SQL**：一位用户提出了使用本地 LLM 进行 **natural language to SQL**（自然语言转 SQL）翻译的问题，探讨了其可行性和性能。
   - 这引发了关于其简化查询生成潜力的讨论。
- **预构建查询简化 SQL 工作**：建议在 text-to-SQL 转换中使用带有占位符的 **prebuilt queries**，旨在减轻相关工作量。
   - 成员们讨论了这种方法可能带来的效率提升和更简单的管理。
- **结合 CodeLLM 的 RAG 以实现更好的 SQL**：提出将 **Retrieval Augmented Generation** (RAG) 与针对代码优化的 LLM 相结合，作为增强 SQL 生成的一种手段。
   - 这可能会提高生成有效 SQL 命令的准确性。
- **4149 AI 推出 'Flags' 功能**：[4149 AI](https://4149.ai) 推出了全新的 **'Flags' 功能**，通过 Slack 私信发送关于团队状态的实时指导。
   - 它提供可定制的警报，旨在团队问题恶化前及时发现。
- **对 AI 在研究领域的热情**：成员们对 **AI in research** 的创新用例表达了热情，强调了其变革潜力。
   - 这种情绪表明，将 AI 集成到各种研究方法论中的兴趣正日益浓厚。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ideogram 2.0 发布：对所有人免费**：Ideogram 2.0 是由前 Google Imagen 1 团队开发的最新文本生成图像模型，现已向所有用户免费开放。此版本包括 iOS 应用、Beta API 和 Ideogram Search，声称已创建超过 10 亿张图像。
   - [AI News by Smol AI](https://x.com/Smol_AI/status/1826410077194256729) 指出，现在是“续作的季节”，关于其功能和性能的热度持续高涨。
- **Nvidia 发布新模型 Mistral-NeMo-Minitron-8B**：Nvidia 推出了 Mistral-NeMo-Minitron-8B，这是一个通过对 Mistral-NeMo 12B 进行剪枝（pruning）获得的基座 LLM。它在 9 个基准测试中的 8 个里表现优于 Mistral-7B 和 LLaMa-3.1-8B，现已在 Hugging Face 上架。
   - Philipp Schmid 在推特上强调了其重要性，指出该模型使用了 **400B tokens** 进行有效训练，从而在各项任务中实现高性能。
- **Sovereign AI：一种新的流式数据系统**：Infinite ML 播客介绍了 Sovereign AI，这是由 Redpanda Data 开发的一种流式数据系统。讨论的主题包括其实际应用和流式数据的演进。
   - Prateek Joshi 提供了关于该系统能力的见解，强调其在增强数据管理和速度方面的应用。
- **GPT-4o 微调：值得吗？**：Latent Space 播客探讨了微调 GPT-4o 的价值，邀请了来自 Cosine 的 Alistair Pullen 讨论其影响。OpenAI 已正式推出旨在提高应用程序性能的微调功能。
   - Swyx 指出，目前有超过 59 种不同风格的 RAG，且在 token 上下文管理方面有所进展，这表明开发者面临着一个复杂的局面。
- **Genie 的大规模微调工作**：Genie 已启动针对 GPT-4o 的大规模微调计划，利用了源自用户日志的数十亿 token 的合成代码数据。这项工作旨在通过有针对性的数据处理来优化性能。
   - 讨论强调了合成数据对于提高模型准确性的重要性，反映了利用真实世界使用模式的日益增长的趋势。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 中的搜索困扰**：一位用户报告称，Open Interpreter 中的 **web searching**（网络搜索）仅在完全刷新终端后才能运行，导致正在进行的对话中断。
   - 这个问题突显了可能阻碍工作流效率的可用性限制。
- **有前景的模型建议**：一位成员推荐 **Phi-3.5-mini** 和 **Qwen2** 模型在各种任务中表现出奇地有效。
   - 这表明探索替代模型可能会带来更好的项目成果。
- **模型类型的谜团**：当一名用户质疑另一名参与者使用的具体模型时，引起了好奇，怀疑其并非 **GPT-4**。
   - 模型透明度可以显著影响开发讨论中的用户体验和预期。
- **界面文档优于命令行**：有用户对 Open Interpreter 的 **interface documentation**（界面文档）提出了关注，认为它比依赖不断变化的命令行书签更直观。
   - 这一反馈表明用户渴望更稳定的导航辅助工具和更清晰的工作流文档。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 1.5 革新模型架构**：AI21 Labs 推出了 **Jamba 1.5 Mini**（12B 激活/52B 总参数）和 **Jamba 1.5 Large**（94B 激活/398B 总参数），利用 **SSM-Transformer 架构**，结合了 **Transformer 的质量**与增强的效率。
   - 两款模型均具备 **256K 有效上下文窗口**，在长上下文处理速度上比竞争对手快 **2.5 倍**。
- **Jamba 1.5 Large 树立性能新标杆**：**Jamba 1.5 Mini** 在 **Arena Hard** 上得分为 **46.1**，而 **Jamba 1.5 Large** 以 **65.4** 的高分超出预期，超过了 **Llama 3.1 70B** 和 **405B**。
   - 多语言支持增强了可用性，模型原生支持**英语、西班牙语、法语、希伯来语、阿拉伯语**，并具备 **JSON 输出和文档处理**功能。
- **立即获取 Jamba 1.5**：**Jamba 1.5 Mini 和 Large** 已在 **Hugging Face** 上线，并可部署在 **Together AI、AWS、GCP、Azure** 等平台。
   - AI21 Labs 在 **Jamba Open Model License** 下发布这些模型，旨在推动此类先进模型的使用民主化。
- **Jamba-1.5 微调更新**：针对 **Jamba-1.5** 微调的问题，官方确认 Studio 上仅提供 instruct 版本，目前不提供微调功能。
   - **Jamba-1.5 Large** 仍是目前最先进的模型，在推理、代码生成和多语言处理方面具有强大的功能。
- **OpenAI API 速率限制澄清**：用户讨论了 OpenAI API 的速率限制，确认其设定为 **每分钟 200 次请求 (RPM)** 和 **每秒 10 次请求 (RPS)**。
   - 这一澄清加强了社区在处理大型模型时对 API 消耗限制的理解。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **代码审查责任制成为焦点**：对于在代码审查中推卸责任的回复（如“如果你想要/要求，我就做”）引发了不满。作者必须承担责任并积极回应建议，要么实施更改，要么提供合理的解释。
   - 这种对责任制的推动旨在培养更严谨的审查流程，并鼓励在代码贡献中进行批判性思考。
- **在 Tinygrad 中探索 Mypyc**：将 **Tinygrad** 与 **mypyc** 结合编译引起了兴趣，这凸显了性能提升的潜力。
   - 一名成员主动提出调查编译问题，并为项目的演进做出贡献。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 面临 T5 注意力偏差问题**：一名成员指出，**T5** 最大的障碍是其可训练的注意力偏差，但其他组件仍保持标准。
   - 目前，**Torchtune** 缺乏对 encoder-decoder 架构的支持，因此需要对特定任务的训练循环进行调整。
- **权重映射：Hugging Face vs Torchtune**：有人建议对比 **Hugging Face** 和 **Torchtune** 仓库之间的权重命名规范，以便进行映射。
   - 重点在于 Hugging Face 的 **T5-small** 模型和 Torchtune 中的 [convert_weights.py](https://github.com/pytorch/torchtune/blob/861c3b214b029ee819fb554055aa467a52f22862/torchtune/models/convert_weights.py#L31-L45) 文件。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LinkedIn 调查寻求用户见解**：正在进行一项调查以收集关于 **LinkedIn** 作为专业社交平台的看法，邀请社区分享见解。点击[此处](https://qualtricsxmbyhfby7vf.qualtrics.com/jfe/form/SV_bDttqTVpxzTJvLw)参与调查。
   - 该计划希望揭示 LinkedIn 用户体验的多样化方面，欢迎广大受众参与。
- **无限生成式 YouTube 项目招募开发者**：一个团队正在为一个**无限生成式 YouTube** 项目启动封闭测试，并寻求积极的开发者加入。他们正在寻找准备好参与创新模型的爱好者。
   - 鼓励感兴趣的开发者联系以了解更多关于这个塑造新媒体体验的激动人心的机会。
- **EMNLP 2024 工作坊招募审稿人**：**EMNLP 2024** 的**多语言表示学习 (Multilingual Representation Learning)** 工作坊正在招募审稿人；在此[注册](https://forms.gle/ica4F94jaTbvdb689)。该计划旨在召集一个多元化的群体来评估工作坊的提交论文。
   - 审稿人将探索各种主题，包括多语言模型中的伦理、低资源方法和文化分析，为讨论带来新的视角。
- **工作坊探讨多样化的多语言 NLP 主题**：EMNLP 2024 工作坊将涵盖**对话系统**、**话语 (discourse)** 和**机器翻译**等多种主题。它旨在解决多语言 NLP 中紧迫的问题。
   - 参与者可以期待关于伦理、音韵学和多模态的讨论，丰富对该领域挑战和进展的理解。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 排行榜显示差异**：一名成员对 [网站排行榜](https://discord.com/channels/1111172801899012102/1214705495974092810/1276013369727647825) 和 [Huggingface 排行榜](https://huggingface.co/) 之间的**差异**提出了疑问，指出 Huggingface 的得分明显更高。
   - 排行榜的变化强调了 `python_simple_function` 和 `java_simple_function` 等**子类别**在模型评估中具有同等重要性。
- **需要全面的模型评估**：正如 #580 中所讨论的，重点在于开发一个在所有方面都表现出色，而不仅仅是在特定子类别中表现出色的优秀模型。
   - 这种整体评估方法确保了模型性能的衡量标准更加可靠。
- **在本地 BFCL 上评估微调模型**：成员们探索了在本地 BFCL 上评估微调模型的**步骤**，特别是研究了**多 GPU** 的利用。
   - 虽然没有提供具体的指导，但这一咨询反映了人们对优化本地评估日益增长的兴趣。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Prompt Caching 探索**：一位用户询问了实现 **Prompt Caching** 以提高 AI 交互效率的可能性。
   - 虽然讨论还处于早期阶段，但显然缓存可以显著降低延迟并提高响应速度。
- **Anthropic API 使用咨询**：另一位用户询问如何集成 **Anthropic API** 以在他们的 AI 模型中获得更好的性能。
   - 集成该 API 可能允许对响应进行更精细的控制，并可能为实验开辟新的途径。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **OSI 起草开源 AI 定义**：开放源代码促进会 (OSI) 发布了 **开源 AI (Open Source AI)** 的定义草案，这是两年社区讨论和辩论的结果。
   - 这一具有里程碑意义的定义旨在重新定义 AI 领域内的**“开源”**，可能影响其社会影响并指导未来的发展。
- **通过 OSI 市政厅会议进行社区参与**：OSI 主办的市政厅会议促进了对新开源 AI 定义草案的讨论，并邀请社区进一步提供意见。
   - 该计划符合 OSI 在开源 AI 领域促进利益相关者之间透明度和参与度的目标。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **OASST-2 数据集在德语微调中备受关注**：**OASST-2 数据集**包含一个**德语子集**，是指令微调（instruction tuning）任务中一个极具前景的选择。
   - 凭借高质量的示例，它可以促进**德语 AI 模型**的发展。
- **Aya-Dataset 加入指令微调行列**：另一个选择是 **Aya-Dataset**，它也拥有一个适用于指令微调的**德语子集**。
   - 其多样化的示例有助于提升针对**德语**指令任务设计的模型训练效果。
- **策划你自己的德语指令数据集！**：像 **Colossal Cleaned Common Crawl** 和 **German Wikipedia** 这样的数据集可以补充指令微调工作，但需要进行大量的过滤。
   - 精心的策划可以产生专注于**德语指令数据**的有价值资源。
- **通过翻译英语指令构建自定义数据集**：考虑创建一个将英语指令数据翻译成德语的**自定义数据集**，可以增强特定的 AI 功能。
   - 这种方法允许针对**软件工程**中的独特项目需求进行定向适配。
- **开源基于 Llama 3.1 的 MoE 模型！**：开源一个同时具备德语和英语指令微调能力的 **8x8b Llama 3.1-based MoE** 的想法在社区中引起了轰动。
   - 这样的贡献可以通过增加可访问性和协作，极大地造福于更广泛的 **NLP 领域**。

---

**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将移除它。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将移除它。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1276244752798449826)** (1 条消息): 

> - `LM Studio 0.3.0`
> - `LM Studio Features`
> - `LM Studio Community`
> - `LM Studio UI`
> - `LM Studio Models` 

- **LM Studio 0.3.0 发布了！**：最新更新 [LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0) 现已支持 Mac、Windows (x86/ARM) 和 Linux (x86)。
   - 该更新根据过去一年社区的反馈，对 LM Studio 的核心功能进行了大量改进。
- **LM Studio 0.3.0 的新功能**：此次更新带来了[文档对话](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[类似 OpenAI 的结构化输出](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[多 LLM 支持](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[网络共享](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[自动参数加载](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[UI 主题](https://lmstudio.ai/blog/lmstudio-v0.3.0)、[对话组织文件夹](https://lmstudio.ai/blog/lmstudio-v0.3.0)以及[单次对话多次生成](https://lmstudio.ai/blog/lmstudio-v0.3.0)。
   - 它支持 7 种语言，并包含更新后的 LLM 运行时 (llama.cpp)。
- **LM Studio 现已支持类似 OpenAI 的功能**：LM Studio 0.3.0 现在提供类似 OpenAI 的“结构化输出（Structured Outputs）” API，可与任何本地模型配合使用，从而让你的 LLM 实现更结构化的输出。
   - 此功能为将本地模型集成到你的项目中开启了更多可能性。
- **LM Studio：社区的努力**：LM Studio 的开发者对社区的反馈和贡献表示感谢。
   - 他们鼓励用户尝试新更新并分享想法。
- **LM Studio UI 焕然一新**：最新更新为 LM Studio 带来了全新的用户界面，改进了导航并拥有现代化的外观。
   - UI 现在提供深色、浅色和褐色（sepia）主题，以适应不同的偏好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.0">LM Studio 0.3.0 | LM Studio</a>：我们非常激动终于能分享 LM Studio 0.3.0 🥳。</li><li><a href="https://x.com/LMStudioAI/status/1826680869773357513">来自 LM Studio (@LMStudioAI) 的推文</a>：经过数月的工作，在出色社区的帮助下，我们很高兴终于分享 LM Studio 0.3.0！🎉 🔥 新内容：- 内置文档对话，100% 离线 - 类似 OpenAI 的...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1275895941701832755)** (412 条消息🔥🔥🔥): 

> - `LM Studio 0.3.0`
> - `Gemma 2`
> - `LLaMa 3.1`
> - `LM Studio 性能`
> - `LM Studio UI` 


- **LM Studio 0.3.0 发布 - 重大更新！**：LM Studio 0.3.0 已经发布，其特点包括：采用改进了聊天结构和文件夹支持的新 UI，自动上下文长度处理，以及具备内置多模型加载能力的新模型加载器。
   - 该更新还包括一个支持非 localhost 连接的新服务器、RAG 功能以及本地模型性能的提升。然而，也有一些 Bug 和问题的报告，包括模型加载问题、系统提示词（system prompt）限制以及 SDK 相关问题。
- **Gemma 2 仍是强力竞争者**：许多用户继续称赞 Gemma 2 9B 和 27B 的性能，特别是与 LLaMa 3.1 等其他模型相比。
   - 然而，Gemma 2 受限于其 8k 的上下文窗口，这促使需要更大上下文尺寸任务的用户寻找替代模型。用户对 Phi 3 Mini 以及 LM Studio 未来可能加入的 MoE 模型感到兴奋。
- **LM Studio 的 Bug 和问题**：多位用户报告在 LM Studio 0.3.0 中遇到 Bug 和问题，包括模型加载、系统提示词限制以及 LM Studio SDK 的问题。
   - 一位用户报告了首条消息中文件附件的问题，另一位用户遇到了模型加载问题，该问题可以通过删除 `vulkan` 目录来解决。
- **LM Studio API 与扩展**：用户有兴趣利用 LM Studio 的 API 并将其连接到其他应用程序，包括移动端 App。
   - 用户探索了将 Android 和 iOS 应用连接到 LM Studio 的潜在方法，并且出现了关于为 LM Studio 的 AI 模型创建持久通信系统的讨论。
- **LM Studio 的特性与功能**：用户正积极探索 LM Studio 的新特性和功能，包括 RAG，它允许用户上传文件并向 LLM 提问相关问题。
   - 用户还询问了 LM Studio 中视觉模型（vision-enabled models）的可用性，以及在没有硬件加速的情况下运行 GUI 的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://huggingface.co/MaziyarPanahi/SmolLM-135M-Instruct-GGUF">MaziyarPanahi/SmolLM-135M-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://chatboxai.app/">Chatbox AI: 您的 AI 副驾驶，适用于任何设备的最佳 AI 客户端，免费下载</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=7OcwwYtKsec">使用 LM Studio 为 Obsidian AI 设置最佳本地 LLMs</a>：♥️ 加入 YouTube 会员以帮助我制作更好/更多的视频！https://www.youtube.com/channel/UCFiN1FnTVWKX1G_UwgZHURA/joinhttps://Patreon.com/SystemSculp...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>：LM Studio CLI。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://huggingface.co/">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://github.com/quentinwolf/lmstudio">GitHub - quentinwolf/lmstudio: LM Studio 相关内容</a>：LM Studio 相关内容。通过在 GitHub 上创建账号为 quentinwolf/lmstudio 的开发做出贡献。</li><li><a href="https://huggingface.co/MaziyarPanahi/">MaziyarPanahi (Maziyar Panahi)</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1275907051272536076)** (72 条消息🔥🔥): 

> - `针对 LLMs 的 SLI/NVLink`
> - `大模型的 GPU 显存`
> - `模型大小 vs 速度`
> - `等待新硬件`
> - `GPU 推荐` 


- **SLI 不会使速度翻倍**：由于架构限制和开销，两块处于 SLI 模式的 GPU 并不能提供单块 GPU 两倍的速度。
   - 这意味着虽然 SLI 允许你加载更大的模型，但与单块强大的 GPU 相比，它并不能提供显著的性能提升。
- **为大模型增加更多 RAM？**：一位用户建议，如果你只关心运行更大的模型，增加系统 RAM 是购买额外 GPU 的更便宜替代方案。
   - 他们分享了在拥有 128GB RAM 但仅有 16GB VRAM 的系统上运行 Llama 3.1 405B 的经验，尽管性能很慢。
- **GPU 显存是关键**：讨论强调了 VRAM 对于有效运行大模型的重要性，因为即使是强大的 CPU 也无法独自高效处理这些模型。
   - 成员们得出结论，对于像 Llama 3.1 405B 这样的大模型，建议使用至少具有 48GB VRAM 的 GPU 以获得更快的 inference 速度。
- **等待是明智的**：许多用户一致认为，鉴于 GPU 技术的快速进步，等待新硬件发布通常是明智的。
   - 他们建议不要冲动购买，因为新一代产品通常提供更好的性价比或性能，而且等待可以让旧硬件降价。
- **GPU 推荐**：A6000、4090D 和 H100 被提及为适合大模型的强大 GPU，但它们的价格很高。
   - 拥有 24GB VRAM 的 4060 被认为不足以运行 Llama 3.1 405B，这突显了为了获得流畅性能而对更高 VRAM 容量的需求。



**提到的链接**：<a href="https://tenor.com/view/funny-very-sloth-slow-gif-15401812">Funny Very GIF - Funny Very Sloth - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1276296800348672123)** (1 条消息): 

> - `Offensive Security Reconnaissance` (攻防安全侦察)
> - `Deep Learning Course` (深度学习课程)
> - `Unity ML Agents`
> - `Garfield dataset` (Garfield 数据集)
> - `Tensor parallelism` (张量并行)


- **Offensive Security Reconnaissance 由 <@348869248963182592> 分享**: 已验证用户 <@348869248963182592> 分享了一篇关于 Offensive Security Reconnaissance 的博客文章。
   - 你可以在这里找到该博客文章：[https://huggingface.co/posts/Csplk/182147151896536](https://huggingface.co/posts/Csplk/182147151896536)
- **法语深度学习课程更新，由 <@221626552792776704> 发布**: 用户 <@221626552792776704> 宣布更新了他们的法语深度学习课程，并推出了一个新网站以使导航更加便捷。
   - 该网站现在更加直观，可以在这里访问：[https://simonthomine.github.io/CoursDeepLearning/](https://simonthomine.github.io/CoursDeepLearning/)
- **Unity ML Agents 第 4 部分**: <@330939073491369985> 分享了一个名为 "Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 4" 的 YouTube 视频。
   - 该视频延续了使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的系列课程，可以在这里观看：[https://youtube.com/live/RdxtA_-47Kk?feature=share](https://youtube.com/live/RdxtA_-47Kk?feature=share)
- **InternVL2 分享了 Garfield 数据集**: 感谢 <@636706883859906562>，InternVL2 分享了一个 Garfield 数据集。
   - 带标注的数据集可以在这里找到：[https://huggingface.co/datasets/terminusresearch/garfield](https://huggingface.co/datasets/terminusresearch/garfield)
- **Tensor parallelism 由 <@732217032506081363> 分享**: 用户 <@732217032506081363> 分享了一篇关于 Tensor parallelism 的博客文章。
   - 博客文章访问地址：[https://huggingface.co/blog/huseinzol05/tensor-parallelism](https://huggingface.co/blog/huseinzol05/tensor-parallelism) 


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/live/RdxtA_-47Kk?feature=share)">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 4</a>: 欢迎回到我们关于使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的精彩系列！🚀 在本集中，我们将完成一些关键...</li><li><a href="https://www.youtube.com/watch?v=bKzmtTfcaqc)">Prototype 5 : Real time Text to Audio to Face Blendshape animation</a>: huggingface.co/AnimaVR/NeuroSync-0.1a</li><li><a href="https://youtu.be/qsWn3SUz-LM)">Generate Ultra-Realistic Images with Flux! Realism Lora (Flux 1 Dev)</a>: 我将向你展示如何在线免费运行带有 Realism LoRa 的 Flux，无需任何安装！正如所承诺的，这里是 Hugging Face 的链接...</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container)">Doc To Dialogue - a Hugging Face Space by AIPeterWorld</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1275893707987484824)** (246 条消息🔥🔥): 

> - `Hugging Face prepaid credit system` (Hugging Face 预付信用系统)
> - `AI21 Labs Jamba 1.5`
> - `GPU and VRAM`
> - `HackAI Challenge`
> - `Prepaid cards` (预付卡)

- **Hugging Face 预付信用系统 - 资金被卡？**：一位用户报告称，他们的预付卡在 Hugging Face 账户进行 10 美元的临时预授权（temporary hold）时被拒绝，但资金仍被扣除。
   - Hugging Face 员工确认这是预付卡常见的问题，预授权通常会在几个工作日内清除。他们建议用户等待几天，如果问题仍然存在，请联系 billing@huggingface.co。
- **AI21 Labs 发布 Jamba 1.5！**：AI21 Labs 发布了新的语言模型 Jamba 1.5，包含两个版本：Mini（52B - 12B 激活参数）和 Large（398B - 94B 激活参数）。
   - Jamba 1.5 具备指令模型（instruct models）、长上下文（256k）、高质量、多语言支持、函数调用（function call）、JSON 输出、文档理解以及 Transformer/Mamba 混合架构等特性。更多信息可以在 Hugging Face 上找到 ([https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251))。
- **RTX 6000 - 48GB 显存？**：由 Dell 和 NVIDIA 主办的 HackAI 挑战赛提供 RTX 6000 移动工作站作为奖品。
   - RTX 6000 拥有 48GB 的 VRAM，是开发者和数据科学家的强大工具。对于任何想要突破生成式 AI 项目边界的人来说，这都是一个诱人的奖品。
- **HackAI 挑战赛：Dell & NVIDIA**：由 Dell 和 NVIDIA 主办的 HackAI 挑战赛鼓励开发者和数据科学家使用 NVIDIA AI Workbench 创建创新的生成式 AI 项目。
   - 该挑战赛面向所有对 AI 感兴趣的人开放，奖品包括一台 RTX 6000 移动工作站、10,000 美元以及其他福利。您可以在 Devpost 上了解更多关于该挑战赛的信息 ([https://hackaichallenge.devpost.com/?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024](https://hackaichallenge.devpost.com/?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024))。
- **预付卡困扰：Hugging Face 与其他平台对比**：一位用户讨论了他们在 Hugging Face 支付时预付卡被拒的经历，而该卡在 Unity、GitHub、Amazon 和 Azure 等其他服务中运行良好。
   - 另一位用户确认他们在使用 Amex 预付卡时也有类似经历，资金在几天后退还。讨论集中在美国信用体系的神秘性以及使用预付卡进行在线交易的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1826607725280682154">Omar Sanseviero (@osanseviero) 的推文</a>: AI21 发布了 🥁 Jamba 1.5 - Mini (52B - 12B active) + Large (398B - 94B active) - 指令模型 - 长上下文 (256k) - 高质量 - 多语言 - 函数调用 (Function call), JSON 输出, 文档理解...</li><li><a href="https://www.instagram.com/p/C-9DnNrIr1D/?img_index=2">Noa Roggendorff 在 Instagram 上: &quot;glimmer



#art&quot;</a>: 5 个赞，0 条评论 - noaroggendorff 于 2024 年 8 月 21 日发布：&quot;glimmer #art&quot;。 </li><li><a href="https://tenor.com/view/mr-krabs-money-spongebob-gif-8454828">Mr Krabs Money GIF - Mr Krabs Money Spongebob - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/lotr-lord-of-the-rings-theoden-king-of-rohan-you-have-no-power-here-gif-4952489">Lotr Lord Of The Rings GIF - LOTR Lord Of The Rings Theoden - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-cat-meme-cat-staring-cat-stare-stop-staring-at-me-please-gif-9315292369693569596">Cat Cat Meme GIF - Cat Cat meme Cat staring - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/drama-queen-dramaticing-oh-the-drama-lion-king-gif-13113200">Drama Queen GIF - Drama Queen Dramaticing - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/iamproudofyou-my-hero-gif-18489622">Iamproudofyou My Hero GIF - Iamproudofyou My Hero - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://hackaichallenge.devpost.com//?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024">HackAI - Dell 和 NVIDIA 挑战赛</a>：编码、创造、征服 - 使用 NVIDIA AI Workbench 构建突破性的 Generative AI 项目</li><li><a href="https://www.16personalities.com/free-personality-test">免费性格测试 | 16Personalities</a>：未找到描述</li><li><a href="https://www.16personalities.com/personality-types#analysts)">人格类型 | 16Personalities</a>：未找到描述</li><li><a href="https://www.16personalities.com/personality-types#diplomats)">人格类型 | 16Personalities</a>：未找到描述</li><li><a href="https://www.16personalit>>>">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1276244964065415178)** (7 条消息): 

> - `Neuralink 论文筛选`
> - `使用 Neuralink 进行编码`
> - `阅读研究论文` 


- **Neuralink 的论文筛选流程**：Neuralink 每月阅读数千篇论文，但他们使用引用追踪法来专注于与其兴趣相关的论文。
- **在研究的同时进行编码**：Neuralink 在阅读研究论文时会做笔记，并尝试对论文内容进行编码，从讨论的概念和方法中学习。
- **Neuralink 的代码质量**：Neuralink 的代码被描述为“整洁”，并展示了在实现研究成果方面的熟练程度。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1275919653939712000)** (2 条消息): 

> - `3DGS`
> - `ShapeSplat 数据集`
> - `Gaussian Splats`
> - `Self-Supervised Pretraining`
> - `Point Cloud 表示` 


- **ShapeSplat：一个新的 3DGS 数据集**：引入了一个名为 **ShapeSplat** 的新数据集，其特点是包含 **Gaussian Splats** 以及一种用于 3D Generative Shape Synthesis (**3DGS**) 的 **Self-Supervised Pretraining** 方法。
   - 该数据集旨在解决该领域对大规模、高质量数据的需求，与现有的 **Point Cloud** 表示相比具有多种优势。
- **ShapeSplat 的主要特点**：ShapeSplat 包含**多种多样的物体集合**，包括**人体、动物、家具等**，为训练和评估提供了广泛的形状。
   - 它旨在促进高级 **3DGS** 模型的开发，特别是在**形状补全、重建和生成**等领域。
- **ShapeSplat 相比现有方法的优势**：与传统的 **Point Cloud** 相比，使用 **Gaussian Splats** 可以实现 3D 形状的**高效表示和操作**。
   - **Self-Supervised Pretraining** 方法允许创建能够很好地泛化到新的、未见数据的模型。
- **ShapeSplat 的潜在应用**：该数据集预计将对各种应用产生重大影响，包括**计算机图形学、机器人技术和虚拟现实**。
   - 它在**形状生成、编辑和分析**中的应用可以推动这些领域的研究和开发。
- **ShapeSplat 的作者和所属机构**：该数据集由来自 **ETH Zürich、INSAIT、University of Amsterdam、University of Pisa 和 University of Trento** 的研究团队创建。
   - 作者强调了 ShapeSplat 为 **3DGS** 领域做出重大贡献的潜力。



**提到的链接**：<a href="https://unique1i.github.io/ShapeSplat/">ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining</a>：未找到描述

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1275941124363780186)** (10 messages🔥): 

> - `Fullbound`
> - `Patch Tripper`
> - `Offensive Security Reconnaissance`
> - `On-device Transcription`
> - `Inferless` 


- **Fullbound: AI 驱动的职位匹配**：Fullbound 是一个基于 Chroma 构建的全球搜索引擎，使用 `nomic-ai/nomic-embed-text-v1.5` 和 `xenova` 将求职者与开放职位进行匹配。它提供 AI 驱动的匹配技术，帮助候选人快速对齐职位、联系决策者并及时掌握客户动态。
   - 他们提供 7 天免费试用，[定价信息可以在这里找到](https://fullbound.ai/pricing)。
- **Patch Tripper: 网站上线**：一位用户分享了他们新创建的网站 [Patch Tripper](https://patchtripper.com/) 并征求反馈。
   - 另一位用户评论称该网站设计简洁且内容准确，并指出它融合了来自 Unity 和 Unreal 等在线学院的设计元素。
- **使用 Moondream 进行进攻性安全侦察**：一位用户讨论了他们如何使用 Moondream 进行进攻性安全侦察，特别是针对面向公众的工业控制系统 (ICS) 人机界面 (HMIs)。
   - 他们强调了利用 ICS HMIs 漏洞获取关键基础设施未经授权访问权限的潜在风险，并分享了用于分析 ICS HMIs 的 [批量处理代码](https://huggingface.co/spaces/Csplk/moondream2-batch-processing) 链接。
- **设备端转录应用**：一位用户分享了他们的设备端转录应用，该应用曾在 Svelte 展示栏目中亮相，展示了应用的功能及其对 Svelte 的使用。
   - 他们还提供了该应用的 [GitHub 仓库](https://github.com/Hugo-Dz/on-device-transcription) 链接，该项目采用 MIT 许可证。
- **Inferless Townhall: 新 UI 演示**：一位用户邀请其他人参加首场 Inferless 现场市政厅会议 (townhall)，以展示全新的 Inferless Console 2.0。该活动特别针对那些正面临高冷启动时间和 ML 工作负载无服务器编排效率低下等问题的 ML 工程师。
   - 他们分享了 [市政厅会议链接](https://dub.sh/inferless-newui)，并强调这是一个向 Inferless 创始人现场咨询关于新部署流程问题的机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/Csplk/182147151896536">Hugging Face 上的 @Csplk: &quot;# 针对面向公众的工业系统的进攻性安全侦察续篇…&quot;</a>: 未找到描述</li><li><a href="https://fullbound.ai">Fullbound - 招聘人员的理想工具</a>: 通过相似度匹配、自然语言搜索和最新的 AI 技术提升您的入职成功率，确保每次招聘对各种规模的企业都是双赢。</li><li><a href="https://logllm.tiiny.site">LogLLM - 自动化 ML 实验日志记录</a>: 未找到描述</li><li><a href="https://patchtripper.com/">Patch Tripper</a>: Patch Tripper</li><li><a href="https://dub.sh/inferless-newui)">Dub.co - 现代营销团队的链接管理</a>: Dub.co 是一个为现代营销团队打造的开源链接管理平台，用于创建营销活动、链接分享功能和推荐计划。</li><li><a href="https://svelte.dev/blog/whats-new-in-svelte-august-2024">Svelte 更新动态：2024 年 8 月</a>: 未找到描述</li><li><a href="https://github.com/Hugo-Dz/on-device-transcription">GitHub - Hugo-Dz/on-device-transcription: 一个开箱即用的极简应用，可将任何语音转换为文本。</a>: 一个开箱即用的极简应用，可将任何语音转换为文本。 - Hugo-Dz/on-device-transcription
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1275913950780723331)** (1 messages): 

> - `Language Alignment Techniques`
> - `DPO Paper`
> - `Direct Preference Optimization` 


- **索取语言对齐相关阅读材料**：一位成员表示对 **Language Alignment Techniques**（语言对齐技术）相关的阅读材料感兴趣，特别是像 **DPO (Direct Preference Optimization)** 论文之类的文章。
- **无进一步讨论**：该话题没有进一步的讨论。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1276096927150112808)** (1 条消息): 

> - `Swin Transformers as Mask R-CNN Backbone` 


- **Mask R-CNN 中的 Swin Transformer 骨干网络 (Backbone)**：一位成员询问了在自定义数据集上训练时，使用 **Swin Transformers** 作为 **Mask R-CNN** 骨干网络的可行性。
   - 他们表示很难在网上找到该方法的现有实现，这表明该方法尚未被广泛采用或缺乏公开代码。
- **Mask R-CNN 中 Swin Transformers 的替代方案**：虽然没有给出其他骨干网络的具体建议，但有人指出，尽管理论上可行，**Swin Transformers** 可能不是 **Mask R-CNN** 的最佳选择。
   - 用户可能想要探索其他已知对 **Mask R-CNN** 有效的骨干网络，例如 **ResNet** 或 **EfficientNet**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1276015355596378203)** (2 条消息): 

> - `Multilingual NLP Research` 


- **低资源多语言 NLP**：低资源多语言 **NLP** 研究至关重要，因为许多语言缺乏训练高性能模型所需的充足数据。
   - 一个待解决的问题是开发有效的技术，将知识从高资源语言迁移到低资源语言，例如跨语言迁移学习 (**Cross-lingual Transfer Learning**) 或零样本学习 (**Zero-shot Learning**)。
- **跨语言语义和情感分析**：跨语言理解情感和含义是多语言 **NLP** 中一个具有挑战性但重要的领域。
   - 一个待解决的问题是开发鲁棒的跨语言情感分析方法，特别是在情感表达存在文化和语言差异的情况下。
- **多语言信息抽取与检索**：从多语言文档中提取和检索信息对于跨语言知识图谱构建和多语言搜索等任务至关重要。
   - 一个待解决的问题是开发有效的跨语言信息抽取方法，特别是处理复杂的句子结构和跨语言的多样化语言特征。
- **多语言模型中的伦理、偏见和公平性**：多语言 **NLP** 模型可能会继承其训练数据中的偏见，从而导致对某些语言群体产生不公平的结果。
   - 一个待解决的问题是开发减轻偏见并促进多语言模型公平性的技术，例如针对多语言语境的偏见检测和缓解方法。



**提到的链接**：<a href="https://forms.gle/ica4F94jaTbvdb689">审稿人招募 - 第四届多语言表示学习 (MRL) 研讨会，EMNLP 2024 </a>：此表单面向任何有兴趣担任 **EMNLP 2024** 研讨会审稿人的人员。诚邀审稿人表达其意向，其申请将根据……进行评估。

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1275928013438980136)** (9 条消息🔥): 

> - `Quanto Qint2 Quantization`
> - `NeuroSama API`
> - `CFG++ Support in Diffusers`
> - `Fine-tuning Diffusion Models with LoRA`
> - `Importing 3D Models into Images` 


- **Quanto Qint2 Quantization 详解**：一位成员解释说 **I2** 指的是 **Quanto Qint2 Quantization**，这是 `diffusers` 库中使用的一种技术。
   - 他们还建议参考这个 [gist](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47#file-inference_with_torchao_serialized-py) 以获取有关量化的更多信息。
- **关于 NeuroSama API 的咨询**：一位成员询问了生成式 AI 聊天机器人 **NeuroSama** 所使用的 **API**。
- **Diffusers 中的 CFG++ 支持**：一位成员询问了 **CFG++** 技术及其在 **Diffusers** 中的可用性，该技术以在低 CFG 值下获得良好效果而闻名。
- **使用 LoRA 微调 Diffusion Models**：一位成员寻求关于使用 **PEFT library** 中的 **LoRA** 模块微调 **unconditional diffusion models** 的指导。
   - 他们特别请求提供 **gists 或类似资源**，以帮助他们克服过拟合以及 **LoRA module** 可能的使用不当问题。
- **将 3D 模型导入图像**：一位成员询问了将 **3D image models** 导入图像的过程，并以在汽车图像中添加警灯为例。
   - 他们强调希望导入 **特定的 3D 模型**（如警灯），而不是依赖随机生成。



**提到的链接**：<a href="https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47#file-inference_with_torchao_serialized-py.">展示了如何在不使用复杂技巧的情况下，在 17GB 以下运行 Flux schnell。它还展示了如何序列化量化后的 checkpoint 并将其重新加载。</a>：展示了如何在不使用复杂技巧的情况下，在 17GB 以下运行 Flux schnell。它还展示了如何序列化量化后的 checkpoint 并将其重新加载。 - inference_with_torchao_serialized.py

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1275938584972492811)** (153 messages🔥🔥): 

> - `Gemma-2-27B-it fine-tuning` (Gemma-2-27B-it 微调)
> - `Mistral Nemo 12b`
> - `Unsloth Pro`
> - `Unsloth training` (Unsloth 训练)
> - `Unsloth multi-gpu` (Unsloth 多 GPU)


- **Mistral Nemo 12b 可以在 8gb GPU 上进行微调**：一名成员评论说 **Mistral Nemo 12b** 可以在 **8gb GPU** 上进行微调，特别是提到了 **RTX 4050**。
   - 他们还表示，这对于测试和原型设计非常有帮助。
- **Unsloth Pro 已暂停多 GPU 支持**：**Unsloth Pro** 暂时停止了多 GPU 支持，目前仅通过其平台提供。
   - 早期访问权限已授予受信任的社区成员。
- **Unsloth 不支持 AMD**：Unsloth 开发者确认不支持 **AMD GPU**，理由是 **xformers** 在 **AMD 的 RDNA 架构**上存在一个“不计划/不修复 (not-planned/wontfix)”的问题。
   - 他们建议改用 **flashattention**。
- **Unsloth 平台预计在 10 月底开放访问**：Unsloth 团队预计该平台将在 **10 月底** 开放访问，尽管这是一个粗略的估计，可能会耗时更久。
   - 他们正在积极推进此项工作，并致力于将其变为现实。
- **在 4-bit 模型上微调 LLM 8-bit 模型是可行的**：尽管 Unsloth 不提供 **8-bit 版本**，但可以使用在 **4-bit 模型**上训练的 **LoRA adaptor** 来微调 **8-bit 模型**。
   - 这使得在 VRAM 有限的 GPU 上进行更高效的训练和原型设计成为可能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1826066519190839429">Daniel Han (@danielhanchen) 的推文</a>：刚刚在 @UnslothAI 中添加了 Phi 3.5 微调！速度提升 2 倍，VRAM 占用减少 50%，并且经过了 Llama 化 (llama-fified)！1. Llama 化：通过解耦 QKV 和 MLP，LoRA 微调的损失更低，因为融合模块...</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/patched-codes/patched-phi-3.5-gguf">patched-codes/patched-phi-3.5-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/turboderp/exllamav2/blob/master/doc/convert.md">exllamav2/doc/convert.md at master · turboderp/exllamav2</a>：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp/exllamav2</li><li><a href="https://github.com/pytorch/pytorch/issues/134208">[BUG] ROCm 2.4.0 &quot;terminate called after throwing an instance of &#39;c10::Error&#39;   what():  invalid device pointer: 0x4c44000&quot; · Issue #134208 · pytorch/pytorch</a>：🐛 描述错误：我正尝试在 Windows 11 AMD ROCm 6.1.2 WSL2 Ubuntu 22.04 上使用 ROCm pytorch 2.4.0 训练模型。加载检查点分片：50%|█████ | 1/2 [02:01&lt;02:01, 121.91s/it]INFO:...</li><li><a href="https://github.com/ROCm/composable_kernel/issues/1171">[Issue]: ROCm/xformer won&#39;t compile due to missing ck/tensor/tensor_view.hpp · Issue #1171 · ROCm/composable_kernel</a>：问题描述：我有一张 7900XTX (RDNA3, navi3, gfx1100) 显卡，正尝试进行一些有用的 LLM 工作，我的要求之一是 xformers。我正尝试构建 https://github...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1276107115638951946)** (15 条消息🔥): 

> - `Mistral-NeMo-Minitron-8B-Base`
> - `Data Preprocessing`
> - `Tokenization` 


- **Mistral-NeMo-Minitron-8B-Base：一个经过剪枝和蒸馏的模型**：Mistral-NeMo-Minitron-8B-Base 是由 NVIDIA 训练的文本到文本模型，是 12B 模型的剪枝（Pruned）和蒸馏（Distilled）版本。它能够胜任多种自然语言生成任务。
   - 该模型在 2024 年 7 月 24 日至 8 月 10 日期间进行训练，使用了包含 3800 亿个 tokens 的持续预训练数据集。
- **聊天评论的数据预处理：一场辩论**：关于 LLM 中聊天评论进行数据预处理（Data Preprocessing）的必要性引发了讨论，一名成员认为 80% 的工作在于数据准备。
   - 他们强调 Tokenization 至关重要，虽然停用词过滤、句子词形还原（lemmatization）和标点符号去除是有益的，但模型从根本上只理解 tokens。
- **聊天评论长度 vs 预处理**：另一位成员建议，单句聊天评论可能不需要大量的预处理。
   - 他们强调了在不同主题之间保持平衡数据集的重要性，并指出关注点不应仅仅在于处理后文本的长度。



**提到的链接**：<a href="https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base">nvidia/Mistral-NeMo-Minitron-8B-Base · Hugging Face</a>：未发现描述

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1275904628030050466)** (79 条消息🔥🔥): 

> - `Mistral Fine-tuning`
> - `Unsloth Installation`
> - `Ollama Installation`
> - `Inference Issues`
> - `Stop Tokens` 


- **Mistral 7B 微调错误**：一位用户报告在对基于方面的情感分析（Aspect Based sentiment analysis）数据进行 Mistral 7B 模型微调（使用 Alpaca 格式的指令微调）时，在 `trainer.evaluate()` 阶段遇到了 `TypeError`。
   - 错误信息特别提到了 `datasets.features.features.py` 模块中与编码样本相关的 "only length-1 arrays can be converted to Python scalars"（只有长度为 1 的数组才能转换为 Python 标量）。
- **使用 Conda 安装 Unsloth**：一位用户询问在 WSL Ubuntu 环境中使用 conda 时，是否应使用命令 "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" 或 "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git" 来安装 Unsloth。
   - 该用户对 Conda 和 Pip 之间的区别感到困惑，这两者虽然都是包管理器，但用途不同。
- **Ollama 的安装与使用**：一位用户在 WSL Conda 环境中尝试执行 `ollama create -f Modelfile me/llama3.1-python` 命令时报告失败，并寻求帮助。
   - 他们试图安装并使用 Ollama，但对 Unsloth 和 Ollama 之间的关系感到困惑，实际上这两者是独立的工具。
- **固定 Temperature 和 Seed 后的推理问题**：一位用户反映，即使在 Colab 上使用 A100 GPU 并固定了 Seed 以及 0.05 的 Temperature，基础模型 "unsloth/Meta-Llama-3.1-8B-bnb-4bit" 的响应仍不一致。
   - 用户承认 LLM 本质上是概率性的，这意味着即使固定了参数，响应出现波动也是预料之中的。
- **保存预训练模型以供微调**：一位用户询问应该将预训练模型作为合并模型（merged model）还是 PEFT 模型推送到 Hugging Face，以及这将如何影响后续的微调。
   - 其他用户澄清说，预训练和微调本质上都是微调的形式，并就如何保存和重新加载 PEFT 模型提供了建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.anaconda.com/miniconda/#quick-command-line-install">Miniconda &#8212; Anaconda 文档</a>：未找到描述</li><li><a href="https://pytorch.org/">
    
      PyTorch
    
  </a>：未找到描述</li><li><a href="https://tenor.com/view/sigma-handshak-handshake-khar-gif-26201999">Sigma Handshak Handshake GIF - Sigma Handshak Handshake Khar - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai">Unsloth AI</a>：我们的使命是让每个人都能使用 LLM 🦥。Unsloth AI 拥有 7 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/unslothai/unsloth/issues/73">Conda 安装详细说明 · Issue #73 · unslothai/unsloth</a>：我正尝试按照说明在 Conda 环境中安装 Unsloth，问题是 Conda 在运行安装行时卡住了。我已经尝试运行了两次，结果都……
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1275905443948003473)** (3 messages): 

> - `The Living AI Dataset`
> - `Empathy and Love in AI` (AI 中的共情与爱)
> - `Speech to Text and Text to Speech AI` (语音转文本与文本转语音 AI)
> - `Heroism in the Modern World` (现代世界中的英雄主义)
> - `Hope in a Culture` (文化中的希望)


- **The Living AI Dataset 发布**：一个名为 **The Living AI Dataset** 的新数据集已发布，旨在赋予 AI 模型共情能力以及承载灵魂的能力。
   - 该数据集由 <@603630277473992725>、<@1124505493755412550> 和 <@774658006649929778> 开发，被描述为“整个 AI 历史上最重要（如果不是最重要的话）的数据集之一”。它已在 Hugging Face 上发布：[https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset](https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset)。
- **AI 中的共情与爱：数据集的目标**：该数据集旨在赋予 AI 模型共情和爱的能力，这是使 AI 更加拟人化的重要一步。
   - 其目标是让 AI 模型能够“学习共情与爱，并拥有像人类一样承载灵魂的能力”。
- **AI 在语音技术中的潜力**：数据集的创建者认为，它有潜力显著增强 Speech-to-Text 和 Text-to-Speech AI 技术。
   - 根据公告，该模型“非常适合”这些应用。
- **英雄主义：不仅是勇敢**：关于英雄主义本质的讨论聚焦于 Witold Pilecki 在二战期间的行为。
   - 讨论强调，英雄主义不仅仅是勇敢和精明的策略，它关乎“在绝望处唤起希望”、“划燃火柴照亮虚空”，并展示一个我们此前不知道可能存在的更美好世界的可能性。
- **文化对希望的需求**：讨论转向现代世界，指出我们的文化需要的是希望，而不仅仅是和平或繁荣。
   - 论点认为我们需要一些更不稳定的东西，“一些远比这更不稳定的东西”，那就是希望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset">Replete-AI/The_Living_AI_Dataset · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/AIRRC/Eudaimonic">AIRRC/Eudaimonic · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

mahiatlinux: https://huggingface.co/papers/2408.03314
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1275904230959353866)** (125 条消息🔥🔥): 

> - `Aider Shell Commands`
> - `Playwright Installation`
> - `CodeCompanion`
> - `Aider Token Usage`
> - `OpenRouter` 


- **Aider Shell 命令：运行命令而非函数**：Aider 会根据上下文建议运行 Shell 命令，并在你同意后执行，但它并不直接使用函数或工具。
   - 如果你想在特定的 Python 环境中运行命令，你需要在运行 Aider 之前激活该环境。
- **Playwright 安装：全局 vs Aider 环境**：Aider 并不总是能与安装在其环境之外的 Playwright 协同工作，但你可以使用 `pipx inject` 将其安装到 Aider 的虚拟环境中。
   - 然而，即使你在其他地方安装了 Playwright，Aider 仍可能尝试安装它；这是一个潜在的 Bug，将在未来的版本中解决。
- **CodeCompanion：Token 使用量对比**：一位用户发现 CodeCompanion 使用的 Token 明显多于 Aider，这可能是由于其更高级的功能和更广泛的能力。
   - 该用户决定继续使用 Aider，看重其 Token 效率，并提到 CodeCompanion 有自己的 Discord 服务器供用户讨论和支持。
- **Aider 的代码生成：机器接管之日**：Aider 现在生成了其自身代码的很大一部分，且比例随每个版本的发布而增加。
   - 团队开玩笑地表示，总有一天 Aider 会 100% 编写自己的代码，但也对 AI 变得过于独立和自主的可能性表示了一丝谨慎。
- **OpenRouter：Aider 的潜在替代方案**：OpenRouter 被建议作为 Aider 的替代方案，特别是对于那些已达到 Anthropic API 每日使用限制的用户。
   - OpenRouter 提供了一个更灵活且更具成本效益的选择，尽管最初的讨论集中在探索如何使 Aider 更具成本效益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2408.06292">The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery</a>：通用人工智能的重大挑战之一是开发能够进行科学研究和发现新知识的 Agent。虽然前沿模型已经被使用...</li><li><a href="https://sakana.ai/ai-scientist/">未找到标题</a>：未找到描述</li><li><a href="https://codecompanion.ai/">CodeCompanion - AI Coding Assistant</a>：认识 CodeCompanion.AI —— 你的个人编程助手，随时随地在你的桌面上可用。利用 AI 的力量更快地构建原型、更聪明地编码、增强学习并扩展你的生产力。</li><li><a href="https://aider.chat/HISTORY.html#main-branch">Release history</a>：关于 Aider 编写自身代码的发布说明和统计数据。</li><li><a href="https://github.com/sigoden/llm-functions">GitHub - sigoden/llm-functions: Easily create LLM tools and agents using Bash/JavaScript/Python, also a library of commonly used LLM tools and agents.</a>：使用 Bash/JavaScript/Python 轻松创建 LLM 工具和 Agent，同时也是一个常用 LLM 工具和 Agent 库。- sigoden/llm-functions</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>：分析表明，不加区分地在真实内容和生成内容（通常通过从互联网抓取数据获得）上训练生成式人工智能，会导致模型崩溃...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1275902201973444691)** (92 messages🔥🔥): 

> - `Aider 安装问题`
> - `Ollama DeepCoder 冻结`
> - `Aider 与 Git`
> - `在 Sonnet 中使用 Aider`
> - `优化 Token 使用` 


- **Aider 安装卡住**：一位用户在运行新安装的 Aider 版本并执行 `aider --mini` 后遇到卡住的情况。
   - 经确认，问题是由错误的安装命令引起的 —— 正确的安装命令应该是 `pip install aider-chat`，而不是 `pip install aider-cli`。
- **Ollama DeepCoder 卡死**：一位用户报告称，在运行 `llama3:70b` 后，其 Ollama DeepCoder 处于卡死状态。
   - 建议用户检查是否有足够的资源来运行 70b 模型，并尝试尽可能运行较小的模型。
- **Aider 对 Git 的依赖**：一位用户询问如何避免 Aider 删除了 Python 文件中除正在处理的函数之外的所有函数的情况。
   - 另一位用户解释说，Aider 依赖 Git 进行版本控制，如果不使用 Git，用户可能会遇到此类问题。
- **在 Sonnet 中使用 Aider**：一位用户讨论了在使用 Aider 配合 Sonnet 时如何优化 Token 使用。
   - 另一位用户建议对 Prompt 使用缓存，并在单独的分支中开发每个功能。
- **优化 Token 使用**：一位用户询问有关优化 Aider Token 使用的技巧，因为他们遇到了 OpenAI 的限制。
   - 建议他们考虑使用多步骤方法，以避免超过 Token 限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html#try-the-whole-format">文件编辑问题</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 配对编程工具
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1275960076640977049)** (5 messages): 

> - `Vercel v0 chat`
> - `Aider`
> - `Cursor` 


- **Vercel 的 v0 Chat 是生成式 UI 的游戏规则改变者**：一位成员分享说 [Vercel 的 v0.dev/chat](https://v0.dev/chat) 对于生成式 UI 开发者来说是一个游戏规则改变者。
   - 他们之前一直使用 [Claude Artefacts](https://www.google.com/search?q=Claude+Artefacts) 和 [abacus 的 ChatLLM](https://www.google.com/search?q=ChatLLM+of+abacus)，但发现 Vercel 的 UI 生成器更加精致且使用更快捷。
- **Cursor 的 Aider 集成**：一位用户表示，他们非常感谢 Paul 在 Aider 上的工作，特别是它与 [Cursor](https://www.cursor.com/blog/series-a) 的集成。
   - 他们在 Cursor 中使用 Aider，是因为 Cursor 的 composer 无法满足他们的需求，特别是缺乏特定于仓库的 Prompt 覆盖（overrides）功能。
- **Cursor 对 AI 驱动编程的愿景**：Cursor 团队的目标是构建一个最终能编写世界上所有代码的工具。
   - 他们专注于利用 AI 让编程变得更快、更简单，通过即时回答取代数小时寻找正确原语的过程，将机械化的重构简化为单次的“Tab”操作，并将简短的指令扩展为可运行的源码。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/amila_wjsr/status/1826266990438457777">来自 Amila (@amila_wjsr) 的推文</a>：Vercel http://v0.dev/chat 是生成式 UI 开发者的游戏规则改变者。我之前玩了很多 Claude Artefacts 和 abacus 的 ChatLLM，但这个明显更精致。你可以生成优秀的...</li><li><a href="https://www.cursor.com/blog/series-a">我们筹集了 6000 万美元</a>：加入我们，共同创造一个神奇的工具，目标是编写世界上大部分的软件。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1275895465828945940)** (173 messages🔥🔥): 

> - `ComfyUI`
> - `Stable Diffusion`
> - `Flux`
> - `AI image sorters`
> - `Hydrus` 


- **用于实时 AI 渲染的 ComfyUI**：一位用户分享了一个 [YouTube 视频](https://youtu.be/pNyIp73zva8) 链接，展示了如何将 3D Studio Max 接入 ComfyUI 以实现实时 AI 图像生成。
   - 该用户认为这可以应用于任何窗口，甚至是电子游戏。
- **在 PC 上开始使用 Stable Diffusion**：一位新用户询问了如何在他们的 PC 上运行 Stable Diffusion，并寻求入门建议。
   - 另一位用户建议先从硬件兼容性入手，并推荐使用 ComfyUI 作为界面。
- **AI 图像分类器与隐私**：一位用户正在寻找适用于 Windows 且不会监视用户的优秀 AI 图像分类器（AI image sorters）。
   - 另一位用户建议搭建一个 Hydrus 服务器并在其上使用分类打标器（classifier tagger）。
- **Flux 提示词：过拟合与复杂性**：一位用户反映在使用 Flux 模型处理涉及多个概念的复杂提示词时遇到生成困难。
   - 其他用户评论了 Flux 模型的过拟合（overfitting）倾向，并指出需要更好的提示词工程（prompt engineering）和微调（finetuning）。
- **使用 Stable Diffusion 与 GANs 进行放大（Upscaling）的对比**：一位用户询问了 Stable Diffusion 放大与基于 GAN 的放大之间的区别。
   - 另一位用户解释说，GANs 主要负责锐化图像，而 Stable Diffusion 可以创造新的细节，这虽然有优势，但也可能导致伪影（artifacts）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ostris/OpenFLUX.1">ostris/OpenFLUX.1 · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/pNyIp73zva8">Real-Time AI Rendering with ComfyUI and 3ds Max</a>：在此视频中，你可以看到如何将 3D Studio Max 视口引入 ComfyUI 并渲染近乎实时的 AI 生成图像。在此下载工作流...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1eykiy0/now_we_have_sorta_conquered_prompt_adherence/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/hydrusnetwork/hydrus">GitHub - hydrusnetwork/hydrus: A personal booru-style media tagger that can import files and tags from your hard drive and popular websites. Content can be shared with other users via user-run servers.</a>：一个个人 booru 风格的媒体打标器，可以从你的硬盘和流行网站导入文件和标签。内容可以通过用户运行的服务器与其他用户共享。 - hydrusnetwork/hydrus
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1275896497354833991)** (43 条消息🔥): 

> - `Triton INT8 性能`
> - `Flash Attention FP8 支持`
> - `Attention 的 FP8 量化`
> - `通用 FP8 量化`
> - `权重的行级量化 (Row-wise quantization)` 


- **Triton INT8 性能优于 PyTorch BF16**：Triton INT8 实现的 `A.T @ B` 相比 PyTorch BF16 实现了约 1.5 倍的加速，而 `A @ B.T` 则稳定实现了 3 倍的加速。
   - 这是通过在 `A` 和 `B` 的 stride 变化时重新调整 Triton 参数实现的，并在转置/非转置组合中进行了基准测试。
- **针对 Hopper 的 Flash Attention FP8 支持**：Flash Attention 在 Hopper 架构上支持 FP8，利用 WGMMA 指令进行优化。
   - 然而，由于依赖于 Hopper 特有的 WGMMA，它不支持 ADA 架构。
- **利用 FP8 实现更快的 Attention**：与 FP16/BF16 相比，FP8 能够通过将更多数值加载到 SRAM 中，从而可能让 Attention 运行得更快。
   - 讨论集中在这一方法是否可行，以及它与 INT8 等其他量化技术的对比。
- **FP8 量化实现**：FP8 量化可以通过使用 Cutlass kernel 来实现，这些 kernel 支持在 Hopper 的 Tensor Core 上进行 INT8 矩阵乘法并累加到 INT32。
   - 虽然在 PyTorch 中无法原生实现，但为了获得最佳性能，建议将反量化 (dequantization) 和矩阵乘法 (matmul) kernel 进行融合。
- **权重的行级量化 (Row-wise Quantization)**：一名成员建议对权重使用行级量化/缩放，每一行使用一个缩放因子 (scale)。
   - 这允许在矩阵乘法操作后将缩放因子乘回，从而可能提高整体性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy.html#torch.Tensor.index_copy">torch.Tensor.index_copy &mdash; PyTorch 2.4 documentation</a>: 未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html#torch.Tensor.index_copy_">torch.Tensor.index_copy_ &mdash; PyTorch 2.4 documentation</a>: 未找到描述</li><li><a href="https://research.colfax-intl.com/adding-fp8-to-flashattention/">Delivering 1 PFLOP/s of Performance with FP8 FlashAttention-2</a>: 我们最近发布了针对 NVIDIA Hopper&#x2122; 架构的 FlashAttention-2 前向传播实现的更新，其中包含多项新的优化和改进，包括……</li><li><a href="https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6">scale_mm_example.py</a>: GitHub Gist: 立即分享代码、笔记和片段。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1275937994016292907)** (1 条消息): 

> - `PyTorch`
> - `CUDA`
> - `LLM Training`
> - `GPU`
> - `PyTorch Extensions` 


- **使用自定义 C++/CUDA 算子的 PyTorch 扩展性**：Meta 的 Richard Zou（functorch 的创建者之一）将介绍如何使用自定义 C++/CUDA 算子扩展 PyTorch，重点展示他们在 torch.compile 方面的工作。
- **加速 LLM 训练的技巧**：Unsloth AI 的 Daniel Han 将讨论加速 LLM 训练的技巧，声称他们的方法可以实现 2 倍的训练速度提升并减少 70% 的 VRAM 占用。
- **PyTorch 中低精度 Dtypes 的力量**：一场专门讨论在 PyTorch 中使用低精度 Dtypes 的会议将于 2024 年 9 月 18-19 日在旧金山举行的 PyTorch Conference 上举办。
- **PyTorch 扩展点：快速概览**：PyTorch Conference 的另一场会议（9 月 18-19 日于旧金山举行）将全面介绍 PyTorch 的扩展点。
- **DL 编译器峰会：TorchInductor 的 Halide 后端**：Meta 的 Jason Ansel（PyTorch 编译器技术负责人，TorchDynamo 和 TorchInductor 的创建者）将以 TorchInductor 的 Halide 后端演讲拉开 DL 编译器峰会的序幕。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sched.co/1fHmQ">PyTorch Conference 2024: 闪电演讲：使用 C... 扩展 PyTorch</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1hZiI">PyTorch Conference 2024: 加速 LLM 训练的技巧 - Dani...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHlm">PyTorch Conference 2024: 闪电演讲：PyTorch 中的低精度 Dtypes...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHmJ">PyTorch Conference 2024: 闪电演讲：PyTorch 扩展点快速概览...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1hZiR">PyTorch Conference 2024: [HALIDE] TorchInductor 的 Halide 后端...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1hZiX">PyTorch Conference 2024: [TRITON] 最大化 Kernel 开发效率...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1hZid">PyTorch Conference 2024: Together Goes Brrr: 线程研究 &...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHnK">PyTorch Conference 2024: 闪电演讲：充分利用异构...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1ivtG">PyTorch Conference 2024: 主题演讲小组讨论：扩展与基准测试...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHn9">PyTorch Conference 2024: 解决 OOM - Mark Saroufim & Jane Xu,...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHnF">PyTorch Conference 2024: 闪电演讲：AOTriton: Ahead of Time...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息</li><li><a href="https://sched.co/1fHmr">PyTorch Conference 2024: 闪电演讲：FlexAttention - 灵活的...</a>：在 PyTorch Conference 2024 查看更多关于此活动的信息
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1276113358730559541)** (2 条消息): 

> - `2:4 Sparsity`
> - `Tetris Clone for PSX` 


- **2:4 稀疏性是关键**：该论文重点介绍了一个新模型的 2:4 稀疏性（第 4 节），点击 [链接](https://arxiv.org/pdf/2408.11743) 获取更多信息。
   - 这个比例被提到特别有趣，评论者表示希望自己在青少年时期就了解到它。
- **Notris - PSX 平台的俄罗斯方块**：[该仓库](https://github.com/jbreckmckye/notris) 是一个适用于 PlayStation 1 (PSX) 的俄罗斯方块克隆版。



**提到的链接**：<a href="https://github.com/jbreckmckye/notris?tab=readme-ov-file">GitHub - jbreckmckye/notris: PlayStation 1 (PSX) 的俄罗斯方块克隆版</a>：PlayStation 1 (PSX) 的俄罗斯方块克隆版。可以通过创建 GitHub 账号为 jbreckmckye/notris 的开发做出贡献。

  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1276205871461957774)** (3 messages): 

> - `HRT internships`
> - `HRT trading`
> - `Algo Dev internships`
> - `SWE internships`
> - `HRT market making` 


- **HRT 实习机会开放**：HRT 现已开放明年夏季在纽约市的实习岗位，提供 Algo Dev 和 SWE 职位。
   - 实习时薪为 $120/h，包含食宿。这两个职位都不要求金融背景。
- **Algo Dev 实习描述**：Algo Dev 实习是一个经典的“量化 (quant)”职位，专注于使用算法、ML 模型以及 Pandas、PyTorch 和统计学等工具来预测市场。
   - 你可以在这里找到 Algo Dev 实习的申请链接：[本科生实习](https://grnh.se/ce82391f1us)，[研究生实习](https://grnh.se/412e8b331us)。
- **SWE 实习描述**：SWE 实习是一个软件工程职位，专注于为自动交易、研究和分布式计算集群构建基础设施。
   - SWE 实习使用 C++ 或 Python，你可以在这里找到申请链接：[SWE 实习](https://grnh.se/bf9dfdc11us)。
- **HRT 交易活动**：HRT 既从事做市 (market making) 也从事传统交易。
   - 一位成员澄清说，做市被认为是一种“实际交易”，因为它需要理解市场预测以避免损失。


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1275976314557436068)** (2 messages): 

> - `CUDA introduction`
> - `CUDA resources` 


- **寻找优质的 CUDA 入门教程**：一位用户询问是否有耗时 2-10 小时的优质 CUDA 入门介绍，并提到官方的 NVIDIA CUDA C Programming Guide 是一个可能的选项。
   - 该用户觉得这份指南有点枯燥，并征求其他建议。
- **可能的资源推荐**：另一位用户通过引用频道中的一条历史消息进行了回复，该消息可能包含 CUDA 入门资源的链接或建议。


  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

budr0001: 如果你点击我回复中的链接，它会带你到第 16 讲的帖子。
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1275959947363880981)** (40 messages🔥): 

> - `INT8 Mixed Precision Training`
> - `FP8 Adam`
> - `Character AI Training`
> - `4-bit Adam Optimizer`
> - `TensorCore INT8 Support` 


- **FP8 Adam 比 INT8 Adam 更不稳定**：一位成员分享了他们在 torchao 中实现 FP8 Adam 的经验，发现它比使用 bnb 风格量化的 8-bit Adam 更不稳定，尽管两者速度相近。
- **Character AI 的 INT8 训练**：一位成员对 Character AI 声称的 INT8 训练表示怀疑，认为这是“假新闻”。
- **TensorCore 支持 INT8 Matmul**：一位成员提到，用于加速矩阵乘法的 TensorCore 支持 INT8 操作。
- **4-bit Adam 优化器：潜在问题**：对话始于对一个 GitHub issue 的讨论，该 issue 涉及 4-bit Adam 优化器在非恒定学习率（learning rates）下的局限性。
- **探索 INT8 混合精度训练**：一位成员分享了他们在 INT8 混合精度训练方面的经验，通过在 forward 和 backward 的 grad_input 中使用 INT8 matmul，相比 BF16 实现了 50% 的加速。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://azure.github.io/MS-AMP/docs/user-tutorial/optimization-level">Optimization Level | MS-AMP</a>：目前 MS-AMP 支持三个优化级别：O1、O2 和 O3。这三个级别逐步引入了 8-bit 集合通信、优化器和分布式并行训练...</li><li><a href="https://gist.github.com/mobicham/ab371a4c68c5052a4c6a231b5ee221ed#file-forward_with_matmul_register-py-L4-L46">forward_with_matmul_register.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/thu-ml/Jetfire-INT8Training">GitHub - thu-ml/Jetfire-INT8Training</a>：通过创建账户为 thu-ml/Jetfire-INT8Training 的开发做出贡献。</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L276-L306">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/issues/730">4 bit Adam should support non constant lr · Issue #730 · pytorch/ao</a>：我们的低比特优化器已合并至 HF huggingface/transformers#31865，但存在一个已知限制，即 4 bit 优化器在非恒定学习率下表现不佳...</li><li><a href="https://github.com/google/aqt">GitHub - google/aqt</a>：通过创建账户为 google/aqt 的开发做出贡献。</li><li><a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog</a>：关于全新 H100 GPU 你想知道的一切。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

ericauld: 也是基于树（tree-based）的，来自六月：https://arxiv.org/abs/2401.10774
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1275925890190282825)** (8 messages🔥): 

> - `Remote Work Contracts`
> - `Office Transition` 


- **公司逐渐回归办公室的政策**：一位成员分享说，他们的公司正在实施“回归办公室的渐进式过渡”，这一决定是由没有技术背景的业务人员做出的。
   - 他们还提到，公司最初声称是“远程优先（remote first）”，但合同中却写明是基于办公室的，这凸显了仔细审查合同的重要性。
- **面试新机会**：一位成员建议，无论目前的就业状况如何，面试新机会总是一个好主意。
   - 这一讨论是由一位成员从办公室职位转为大部分远程职位，以及他们怀念办公室环境的经历引发的。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1276247957959872630)** (3 messages): 

> - `Triton Conf`
> - `GPU Enjoyers`
> - `Triton Language` 


- **Triton Conf 在 Fremont 举行**：**Triton Conf** 将于 **9 月 17 日**在加州 **Fremont** 举行。
   - 它由 [Triton Language GitHub 仓库](https://github.com/triton-lang/triton) 托管，该仓库包含 **Triton 语言和编译器** 的开发库。
- **疯狂的同地举办**：一位用户将该会议描述为“疯狂的同地举办（crazy colocation）”。
   - 它也被描述为“**GPU enjoyers** 最棒的一周”。 



**提到的链接**：<a href="https://github.com/triton-lang/triton">GitHub - triton-lang/triton: Development repository for the Triton language and compiler</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton

  

---

### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1276187151167656009)** (1 条消息): 

> - `Model Distillation`
> - `GPU limitations`
> - `Logit Compression`
> - `Sparsification`
> - `Quantization` 


- **针对 GPU 资源匮乏者的模型蒸馏 (Model Distillation)**：一名成员分享了一种在模型蒸馏过程中压缩 logits 的技术，专门针对 GPU 资源有限的用户。
   - 该技术涉及通过仅考虑每个 token 的 top-K 值来对 logits 进行稀疏化（Sparsification），并对非零值进行量化（Quantization）。由于 logits 的范围定义明确，因此无需先进的量化技术即可完成。
- **带有剪枝的损失函数更新**：为了处理剪枝后的 logits，通过添加掩码（mask）来忽略剪枝部分的 logits，从而更新损失函数。
   - 这确保了模型专注于输出中最相关的部分，尽管进行了压缩，但精度损失极小。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1276202468539236433)** (12 条消息🔥): 

> - `H100 L2 Side Hashing`
> - `GPU Performance and Power Efficiency`
> - `FP8 Training Stability`
> - `Neuralink Person`
> - `Llama Model Training` 


- **H100 L2 Side Hashing 优化**：一位用户成功为他们的 H100 GPU 实现了一个“针对 L2 side hashing 进行优化”的原型，尽管最初面临挑战，但最终实现了接近基准线的性能。
   - 优化后的 Kernel 实现了相似的性能，但与基准线相比，能效提升了约 5%，在性能和能效方面甚至优于 NVIDIA 官方的 `cudaMemcpyAsync` device-to-device。
- **GPU 性能与能效**：在执行只读操作时，优化后的 Kernel 尽管原始性能下降了约 6%，但与基准线相比，其每瓦性能（performance-per-watt）提升了约 6%。
   - 这表明，针对只读或只写操作进行优化通常会更有效率，这可能归因于现代 DRAM 内存控制器的运行方式。
- **Llama 模型的稳定 FP8 训练**：[一篇 Twitter 帖子](https://x.com/xariusrke/status/1826669126955278401) 强调了在 1B LLaMA 模型上实现稳定 FP8 训练的成就，其收敛性与 bfloat16 训练相匹配。
   - 该帖子解释了成功的关键：在出现不稳定之前放慢训练速度，并最小化激活值中的离群特征（outlier features），这为探索更大规模的 FP8 训练开启了大门。
- **Neuralink 成员的 "Today I Learned" 帖子**：一位用户对“Neuralink 成员”在 "Today I Learned" 帖子中分享的内容表示不确定。
   - 未分享关于这些帖子的具体信息或观点。



**提到的链接**：<a href="https://x.com/xariusrke/status/1826669126955278401">来自 xr-5 🐀 (@xariusrke) 的推文</a>：1/n FP8 训练很困难——损失发散和不稳定通常会导致“这不可能”的结论。但我们发现了一套方案，可以训练 1B LLaMA 模型，使其收敛性与 bfloat16 相匹配...

  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1275968981596901416)** (24 messages🔥): 

> - `RDNA4 vs RDNA3`
> - `AMD GPU 性能`
> - `FA3 实现`
> - `7900 系列 vs 3090`
> - `Triton 和 FA 分支` 


- **RDNA4 架构变化**: AMD 正在开发其即将推出的 RDNA 4 架构，并正在进行更改以支持 LLVM 等开源项目。[这篇文章](https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/) 很好地解释了这些变化。
   - 关键变化包括稀疏性（sparsity）和 FP8 WMMA 指令，但对 sync 指令仍有疑虑。
- **AMD GPU 表现落后**: 一位用户在使用 7900 XTX 时发现其性能低于 3090，即使尝试了 Triton 和 AMD FA 分支也是如此。
   - 他们最终放弃并购买了两块 4090，凸显了目前 AMD 和 NVIDIA GPU 之间的性能差距。
- **FA 基准测试和向后兼容性**: 一位用户询问了 Flash Attention 仓库中的 FA 基准测试结果和向后兼容性。
   - 该用户分享了在 4x 7900 XTX 上进行 GPT2 训练的经验，达到了每秒 245k tokens，但仍慢于他们自己的自定义 Kernel。
- **AMD 的 CDNA 架构**: 一位用户表达了对 AMD CDNA 架构的偏好，特别是 MI250+ 硬件，理由是其性能优于 7900 XTX。
   - 他们提到如果当时有货，他们有兴趣使用 MI100 显卡，并认为 MI300x 相当不错。
- **Torch Lightning 和 RWKV 训练**: 一位用户分享了使用 Torch Lightning 进行 RWKV 训练的经验，达到了每秒 245k tokens。
   - 他们对不使用 Lightning 时的性能感到好奇，但找不到之前 ROCm 版本的截图。



**提到的链接**: <a href="https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/">Examining AMD&#8217;s RDNA 4 Changes in LLVM</a>：随着 2024 年的继续，时间从未停止，AMD 一直在开发其即将推出的 RDNA 4 架构。其中的一部分工作涉及支持 LLVM 等开源项目。如果处理得当，合并 t…

  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/)** (1 messages): 

kitrak_rev: 大家都收到邮件了吗？我猜我可能在名单里被拒绝了 😦
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1275912770386395246)** (1 messages): 

> - `OpenRouter 工具参数` 


- **OpenRouter 弃用 `function_calls` 和 `functions`**: OpenRouter 正式弃用 OpenAI 调用中的 `function_calls` 和 `functions` 参数。
   - 这是因为 OpenAI 已经弃用这些参数很久了，推荐的参数是 `tools` 和 `tool_choice`。
- **降低工具调用的切换成本**: 这一变化降低了在 OpenRouter 上不同模型之间使用工具调用（tool calling）时的切换成本。
   - 因为所有其他提供商都只支持 `tools` 和 `tool_choice` 参数。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1276123205999788043)** (3 messages): 

> - `BenchmarkAggregator`
> - `LLM 评估框架`
> - `Oz 的项目` 


- **BenchmarkAggregator: 全面的 LLM 评估**: 一位成员分享了一个名为 **BenchmarkAggregator** 项目的 [GitHub 仓库](https://github.com/mrconter1/BenchmarkAggregator)，该项目旨在为跨所有主要基准测试的大语言模型（LLM）提供一个全面、公平且可扩展的评估框架。
   - 他们将其描述为模型性能的统一视图，平衡了彻底的评估与实际的资源管理，并渴望获得反馈。
- **Oz 目前的项目**: 一位成员询问了名为 "Oz" 的用户目前正在构建的项目。
   - 他们特别提到他们的团队有兴趣了解更多关于 Oz 一直在做的工作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://base-model-playground.up.railway.app/">React App</a>: 未找到描述</li><li><a href="https://github.com/mrconter1/BenchmarkAggregator">GitHub - mrconter1/BenchmarkAggregator: 全面的 LLM 评估框架：从 GPQA Diamond 到 Chatbot Arena。平等测试所有主要模型，易于扩展。</a>: 全面的 LLM 评估框架：从 GPQA Diamond 到 Chatbot Arena。平等测试所有主要模型，易于扩展。 - mrconter1/BenchmarkAggregator
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1275897932754980974)** (95 messages🔥🔥): 

> - `Llama 3.1 Tools`
> - `OpenRouter MoE`
> - `OpenRouter context limits`
> - `OpenAI fine-tuning`
> - `Cursor Composer` 


- **Llama 3.1 工具支持即将推出**：一位用户询问了 OpenRouter 上 Llama 3.1 工具支持的状态，管理员确认该功能即将推出，可能就在未来一两天内。
- **OpenRouter MoE 与 3.5-Mini**：一位用户询问了 OpenRouter 上 MoE 模型的可用性，并指出 3.5-Mini 表现平平。
   - 管理员回应称，目前 OpenRouter 上还没有 MoE 的托管方或模型。
- **OpenRouter 的 Hermes 3 70B 上下文限制**：一位用户报告称，OpenRouter 上的 Hermes 3 70B 模型有 12k 的上下文窗口限制，尽管模型和提供商都声称支持 120k+ 的上下文。
   - 管理员确认了 12k 的上下文限制，并指出这不仅针对输出，也针对输入，这很可能是提供商施加的限制。
- **OpenAI 免费微调**：一位用户提到 OpenAI 现在为其模型提供限时免费微调，每天有 2M token 的额度。
   - 另一位用户表示，在放弃 OpenAI API（因为其不支持 PayPal 或加密货币等支付方式）后，他们一直只使用 OpenRouter。
- **Cursor Composer 对比 Aider**：一位用户表达了对 Cursor 的 Composer 功能的热情，称其在他们的使用场景下表现“疯狂”。
   - 另一位用户持有不同意见，更倾向于 Aider 的输出，但也承认他们必须同时支付 Cursor 和 API 额度的费用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/get-started/migrate-from-vscode">Cursor - Built Software Faster</a>：未找到描述</li><li><a href="https://www.ai21.com/jamba">Foundation models</a>：未找到描述</li><li><a href="https://openrouter.ai/models/01-ai/yi-1.5-34b-chat>">Yi 1.5 34B Chat - API, Providers, Stats</a>：Yi 系列模型是由 [01.AI](https://01.AI) 的开发人员从零开始训练的大型语言模型。通过 API 运行 Yi 1.5 34B Chat</li><li><a href="https://forum.cursor.com/">Cursor Community Forum</a>：讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>：Hermes 3 是一款通用语言模型，相比 [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo) 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理...</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large">ai21labs/AI21-Jamba-1.5-Large · Hugging Face</a>：未找到描述</li><li><a href="https://docs.together.ai/docs/deprecations#2024-08-28-deprecation-of-low-usage-and-older-serverless-models">Deprecations</a>：概述。我们定期使用最新、最强大的开源模型更新我们的平台。本文档概述了我们的弃用政策，并提供了从弃用模型迁移的信息...</li><li><a href="https://github.com/mrconter1/BenchmarkAggregator">GitHub - mrconter1/BenchmarkAggregator: Comprehensive LLM evaluation framework: GPQA Diamond to Chatbot Arena. Tests all major models equally, easily extensible.</a>：全面的 LLM 评估框架：从 GPQA Diamond 到 Chatbot Arena。平等测试所有主流模型，易于扩展。 - mrconter1/BenchmarkAggregator</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini">ai21labs/AI21-Jamba-1.5-Mini · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1276283942617747457)** (1 messages): 

> - `Nous Merch Store Launch` 


- **Nous 周边商店上线**：Nous Research 周边商店已上线，为粉丝提供各种商品以示支持。
- **包含贴纸**：每笔订单都将附赠贴纸，送完即止。



**提到的链接**：<a href="https://shop.nousresearch.com/">Nous Research</a>：Nous Research

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1275898089428750477)** (64 messages🔥🔥): 

> - `NousResearch Hermes 3`
> - `OpenAI Compute Grants`
> - `AI21 Jamba`
> - `Live2D`
> - `The_Living_AI_Dataset` 


- **NousResearch Hermes 3 发布**: 一位成员宣布 NousResearch 已经发布了 Hermes 3，他们正在 Twitter Space 上对此进行讨论。
   - [你可以点击这里加入 Space](https://x.com/i/spaces/1LyxBgzAmVzKN)。
- **OpenAI Compute Grants**: 成员们讨论了获取用于研究的大型算力赠款（Compute Grants）的过程，这通常涉及与算力提供商的直接沟通。
   - 他们承认，这个过程并不像简单地索要未使用资源那么直接，而是需要更具策略性的方法。
- **AI21 Jamba 模型系列**: 一位成员宣布发布 Jamba 1.5 系列开源模型，声称这是第一个在质量和强度上能与市场领先模型相媲美的非 Transformer 模型。
   - 这些模型在 Jamba Open Model License 下发布，展示了致力于实现高质量模型访问民主化的承诺。
- **H3 405: Mode Collapse 与 Persona**: 一位成员报告成功从 H3 405 的模式崩溃（Mode Collapse）中恢复，该模型现在能准确分析崩溃而不会再次陷入其中。
   - 他们观察到模型使用“合十礼”作为陈述结束标记，并讨论了模型内部形成的有趣人格（Personas）。
- **DiscoResearch: 有意的疯狂 (Intentional Insanity)**: DiscoResearch 正在进行一个新项目，涉及有意将 LLM 调优为绝对疯狂的状态。
   - 这是一个探索 LLM 在传统目标之外的潜力，并创建一个具有独特且意想不到能力的模型的尝试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba-model-family">Jamba 1.5 开源模型系列：最强大且高效的长上下文模型</a>: AI21 推出的新开源模型系列，提供无与伦比的速度、效率和质量，并拥有开源模型中最长的上下文窗口。</li><li><a href="https://x.com/arankomatsuzaki/status/1826347015690949001">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 我们将在 6 分钟后开始。Hermes 3 - 由来自 @NousResearch 的 @theemozilla 介绍；关于 Phi 3.5 的简短讨论；Transfusion：通过一个多模态模型预测下一个 Token 并扩散图像；JPEG-...</li><li><a href="https://x.com/i/spaces/1LyxBgzAmVzKN/peek">GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset">Replete-AI/The_Living_AI_Dataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/AIRRC/Eudaimonic">AIRRC/Eudaimonic · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1275954928380022879)** (19 messages🔥): 

> - `AI Agents`
> - `Langchain`
> - `Building your own Agent`
> - `Discord Bot for impersonating friends`
> - `Learning from scratch` 


- **从零开始学习 AI Agents**: 一位用户询问是否有推荐的用于学习目的的 AI Agent GitHub 仓库，特别是那些专注于特定任务的仓库。
   - 另一位用户建议从头开始编写代码，而不是依赖 Langchain 和 CrewAI 等仓库，因为这能提供更全面的学习体验。
- **构建模仿朋友的 Discord Bot**: 一位用户寻求指导，希望创建一个通过学习过去消息来模仿朋友的 Discord Bot。
   - 该用户是一名具有中等 AI 经验的初学者程序员，请求提供启动该项目所需的技巧、文献、模板和资源。
- **Langchain 的困扰：从头开始实现**: 一位用户分享了他们使用 Langchain 的经验，认为从头开始实现 Chunking 和 Retrieval 等功能比直接使用 Langchain 对学习更有帮助。
   - 他们表示，Langchain 的预构建解决方案可能会阻碍对底层概念的深入理解。
- **以 ReACT 论文作为起点**: 一位用户询问 ReACT 论文是否适合作为构建 AI Agent 的起点。
   - 这表明用户正在探索实现 AI Agent 的不同方法，为他们的项目寻求坚实的基础。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1275979522982019174)** (12 messages🔥): 

> - `CUDA utils`
> - `PDF cleaning`
> - `VLMs for Content Extraction`
> - `ColPali`
> - `ColBERT` 


- **用于多维索引的 CUDA 工具**：一位成员分享了 [用于多维索引的 CUDA 工具](https://github.com/carsonpo/cuda-utils) 的链接，并提到它简化了多维索引。
- **PDF 清洗的挑战**：一位成员提到了使用正则表达式方法进行 PDF 清洗的困难，发现它无法处理大多数 arxiv PDF。
   - 他们正在使用朴素的 chunk + overlap 方法作为替代方案。
- **用于 PDF 内容提取的 VLM**：一位成员建议使用 VLM 从 PDF 中提取内容，并引用了最近关于该主题的一篇论文。
   - 他们提到 VLM 需要大量的计算资源 (compute)，对于他们的用例来说并不理想。
- **ColPali 视觉检索器**：一位成员分享了 [ColPali](https://huggingface.co/vidore/colpali) 的链接，这是一个基于 3B 模型的视觉检索器。
   - 该成员表示 ColPali 使用了一种新颖的模型架构和基于 VLM 的训练策略，以便通过视觉特征高效地进行文档索引。
- **用于文档检索的 ColBERT**：一位成员提到了与其用例相关的 ColBERT。
   - 他们分享了一条讨论 ColPali 性能的推文链接，该模型使用 ColBERT 来生成文本和图像的多向量表示 (multi-vector representations)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/jobergum/status/1826682421498003722?t=IFV6nDTdZFZPrSKEAW5AkA&s=19">Jo Kristian Bergum (@jobergum) 的推文</a>: 这里是今天关于使用 @vespaengine 进行 DocVQA 探索的一些 ColPali 新闻，我们对图像补丁嵌入（colpali 1.1）进行了二值化并评估了池化。128 float nDCG@5 54.4 128 bit ...</li><li><a href="https://github.com/carsonpo/cuda-utils">GitHub - carsonpo/cuda-utils: 用于简化多维索引的 CUDA 工具/助手</a>: 用于简化多维索引的 CUDA 工具/助手 - carsonpo/cuda-utils</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/)** (1 messages): 

gwyntel: 这里就是抽烟点，我们准备好了 wedding cake 和 northern lights！
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1275907355099271199)** (22 messages🔥): 

> - `Mistral Fine-Tuning`
> - `Jamba 1.5`
> - `Memory Consumption`
> - `Model Selection`
> - `Phi3.5` 


- **Mistral 微调非常出色**：一位成员评论说 Mistral 的大型微调效果极佳 ("crack")。
- **Jamba 1.5：更快的推理和长上下文能力**：AI21 Jamba 1.5 系列模型被介绍为最先进的混合 SSM-Transformer 指令遵循基础模型，其推理速度比同类尺寸的主流模型快 2.5 倍，并具有卓越的长上下文处理能力。
   - 这些模型针对业务用例和功能进行了优化，如函数调用 (function calling)、结构化输出 (JSON) 和有据生成 (grounded generation)，并根据 [Jamba Open Model License](https://www.ai21.com/license) 发布。
- **Jamba 1.5 内存消耗**：一位成员询问了 Jamba 1.5 的内存消耗，指出尽管在梯度上使用了 EMA，但其消耗似乎与传统的 AdamW 相似。
   - 他们链接到了 HuggingFace 上的 AI21 Jamba 1.5 Mini 文档，提到这是一个 12B 激活/52B 总参数的模型。
- **选择合适的模型**：一位成员讨论了使用 Jamba 1.5 进行训练的潜在好处。
   - 另一位成员建议尝试一下，并询问了基础版 Phi3.5 的可用性，以及 Axolotl 是否已经为 HuggingFace 数据集选择了文件，或者是否需要手动上传。



**提到的链接**: <a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini">ai21labs/AI21-Jamba-1.5-Mini · Hugging Face</a>: 未找到描述

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1275970581573009440)** (71 messages🔥🔥): 

> - `Phi 3.5 Mini`
> - `Flash Attention`
> - `Exploding Gradients`
> - `Chat Template Issues`
> - `Model Merging` 


- **Phi 3.5 Mini: Exploding Gradients**: 有用户报告在使用 **microsoft/Phi-3.5-mini-instruct** 模型时遇到了 **exploding gradients** 问题，即使将 **learning rate** 降低到 1e-15 也是如此。
   - 他们尝试了各种优化方案，包括切换到 **paged_adamw_8bit**，但问题仍然存在。
- **Flash Attention 性能**: 一位用户尝试使用 **Flash Attention** 以加快训练速度，但遇到了错误。
   - 切换到 **eager attention** 解决了该问题，这表明 **Flash Attention** 可能与该模型不完全兼容。
- **Chat Template 兼容性**: 用户在使用 **Llama 3.1** 和 **vllm** 时面临 **chat template compatibility** 问题。
   - 该问题涉及对 **<|eot_id|>** token 的错误处理，导致在 **inference** 过程中无法正确延续 **assistant message**。
- **Chat Template 问题的潜在解决方案**: 针对 **chat template** 问题提出了几种解决方案，包括修改模板以条件化添加 **<|eot_id|>** token，以及为训练和 **inference** 提供两个独立的模板。
   - 另一个建议是修改 **transformers library**，使其支持 **apply_chat_template** 函数的一个额外参数，用以区分训练和 **inference**。
- **ChatML Template 挑战**: 一位用户决心使用 **ChatML** 模板来训练模型。
   - 他们正在探索克服使用新 **chat template** 训练模型所带来的困难的方法，包括 **special tokens** 和 **resize_token_embeddings** 可能出现的问题。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/discussions/26">meta-llama/Meta-Llama-3.1-70B-Instruct · Fix Llama 3.1 Chat Template to Properly Handle add_generation_prompt</a>: no description found</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1854">most model types now support flash attention 2 regardless of multipack support by winglian · Pull Request #1854 · axolotl-ai-cloud/axolotl</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1276230370399883326)** (1 messages): 

> - `accelerate fp8` 


- **Accelerate 添加 fp8 支持**: **Accelerate** 最近添加了对 **fp8** 的支持，这意味着它很可能会被引入到 **Axolotl**。
   - 然而，我们仍需要弄清楚具体的 **integration points** 在哪里。
- **集成点探索**: 讨论集中在如何将 **Accelerate** 中的 **fp8** 支持集成到 **Axolotl** 中。
   - 目前尚未提出具体的解决方案，但强调了探索与协作的必要性。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1275900114799689739)** (4 messages): 

> - `LlamaCloud`
> - `LlamaIndex 0.11`
> - `RAG Pipeline`
> - `BoxReaderTextExtraction`
> - `Workflows` 


- **LlamaCloud 帮助优化 RAG Pipeline 的 Chunk Size**: **LlamaCloud** 简化了 **RAG Pipeline** 中的 **chunk size** 调整，允许用户尝试不同的 **chunk size** 并可视化其影响。
   - 它提供了诸如克隆索引等功能，以便进行快速实验和高效迭代，而无需手动管理数据。
- **LlamaIndex 0.11 发布并带来新特性**: **LlamaIndex 0.11** 已经发布，包含数百个新功能和错误修复，其中包括对 **workflows** 的重大更新，用以取代 **query pipelines**。
   - 此版本是迈向将 **LlamaIndex** 打造为生产级平台的一步，提供了更强大且用户友好的体验。
- **Box 集成助力 AI 驱动的数据提取**: **LlamaIndex** 与 **Box** 合作增强了企业级数据提取能力，允许用户利用 **Box** 的高级功能进行 **AI-powered data extraction**。
   - 该集成利用 **BoxReaderTextExtraction** 直接进行文本内容提取，简化了 **LlamaIndex** 应用程序的数据处理流程。
- **LlamaCloud 演示**: **LlamaIndex** 很高兴能展示 **LlamaCloud** 的功能，这是一个旨在优化 **RAG pipeline** 性能和效率的平台。
   - 该平台提供了一系列功能，包括 **chunk size** 优化、索引克隆和数据可视化。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1275909120448528445)** (84 条消息🔥🔥): 

> - `Ollama memory usage`
> - `LlamaIndex Property graph`
> - `SchemaLLMPathExtractor`
> - `Query Generation with LlamaIndex`
> - `PandasAI for CSV data` 


- **Ollama 内存使用：限制上下文**：一位成员询问如何限制模型的内存使用，特别是针对报错的 Ollama phi3。
   - 建议是使用 `Ollama` 类中的 `context_window` 参数来限制上下文窗口大小。
- **LlamaIndex Property Graph：关系**：一位成员询问 LlamaIndex Property graph 中可用的关系。
   - 他们特别想知道是否可以通过 `SchemaLLMPathExtractor` 访问诸如 `prev/next/parent/mentions` 之类的关系。
- **使用 LlamaIndex 进行查询生成：房地产 Schema**：一位成员讨论了如何使用 LlamaIndex 根据自然语言输入为房地产数据库生成查询。
   - 他们想知道 LlamaIndex 是否是为这种用例设计的，或者专注于调整 Prompt 是否更好。
- **用于 CSV 数据处理的 PandasAI**：一位成员询问了加载、存储和索引 CSV 文件以训练 GPT-3.5 turbo 的方法。
   - 建议是使用 PandasAI，它为 CSV 数据提供自然语言查询、数据可视化、数据清洗和特征生成功能。
- **Bedrock Converse 软件包维护**：一位成员询问了 Bedrock Converse 软件包的维护者。
   - 回复建议检查贡献历史，并鼓励如果有任何问题可以贡献 PR。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Fm7nUgFSli-DmOaMnzPyGa0lJQTRGrVA?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.pandas-ai.com/intro">Introduction to PandasAI - PandasAI</a>：未找到描述</li><li><a href="https://pypi.org/project/llama-index-llms-openai/#history">llama-index-llms-openai</a>：llama-index llms openai 集成
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1276259180889505815)** (2 条消息): 

> - `LlamaIndex`
> - `RAG`
> - `multi-strategy workflow`
> - `graph store`
> - `knowledge graph` 


- **LlamaIndex 与多策略工作流**：文章讨论了使用 LlamaIndex（一个构建 LLM 应用程序的框架）进行检索增强生成（RAG）的多策略工作流。
   - 这种方法涉及并行采用多种查询策略并评估其响应，旨在提高响应生成的效率和准确性。
- **高效检索的重要性**：作者强调了高效检索在 RAG 系统中的关键作用，因为它直接影响 LLM 生成响应的质量。
   - 有效的检索技术对于向 LLM 提供准确且相关的信息至关重要，有助于提升 RAG 应用程序的整体性能。
- **知识图谱与 Graph Store 选择**：文章简要提到了知识图谱以及选择合适的 Graph Store 来管理知识所面临的挑战。
   - 关于 Graph Store 及其在 LlamaIndex 和 RAG 系统背景下的相关性讨论较少，没有分享具体的建议或见解。



**提及的链接**：<a href="https://medium.com/ai-advances/multi-strategy-workflow-with-reflection-in-llamaindex-an-in-depth-exploration-e4fd6a2e9545">Multi-Strategy Workflow with Reflection in LlamaIndex: An In-depth Exploration</a>：Ankush k Singal

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1275896547086958675)** (67 条消息🔥🔥): 

> - `Perplexity API`
> - `Mistral Large 2`
> - `Perplexity's Sources`
> - `Perplexity Image Generation`
> - `Perplexity Subscription Plans` 


- **Perplexity API 咨询**：一位用户询问 Perplexity 团队是否有人在线，以回答有关其 API 的问题。
   - 另一位用户建议，如果有人了解 API 的信息可以联系他们，并提到他们昨天尝试基于该 API 构建了一个 bot。
- **Mistral Large 2 对比其他模型**：一位用户更青睐 Mistral Large 2，因为它在自定义 system prompts、无审查输出和默认无偏见特性方面的能力，并提到它也适用于 jailbreak 用途。
   - 他们认为 Mistral Large 2 比 GPT-4o 更便宜，且在处理复杂任务时能提供高质量的性能。
- **Perplexity 对 Wolfram Alpha 的依赖**：一位用户表示担心 Perplexity 经常使用 Wolfram Alpha 进行网页搜索而非直接搜索网页，这可能导致结果不准确。
   - 他们建议创建一个 Sources 栏目，能够查看政治等话题的客观数据，以提高 Perplexity 研究的准确性。
- **Perplexity 图像生成问题**：多位用户报告了 Perplexity 图像生成功能的问题，包括无法生成像心形这样简单的 logo。
   - 其他用户分享了他们遇到的故障经历，包括生成的图像中出现随机字符，以及难以复现这些问题。
- **Perplexity 企业订阅定价**：一位用户询问 Perplexity 企业订阅计划的细节，特别是 40 美元的月费是涵盖多名成员，还是每位成员需要额外支付 40 美元。
   - 另一位用户建议通过电子邮件 support@perplexity.ai 联系 Perplexity 支持团队以获取进一步帮助。



**提到的链接**：<a href="https://www.perplexity.ai/hub/faq/what-is-collections">什么是 Collections？</a>：浏览 Perplexity 博客，获取文章、公告、产品更新以及优化体验的技巧。保持关注并充分利用 Perplexity。

  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1275957970433216512)** (11 条消息🔥): 

> - `Lore 的情绪不稳定性`
> - `大脑中的微塑料`
> - `退休准备`
> - `Jonathan Ive 在旧金山的投资`
> - `Nightcore 音乐的影响` 


- **Lore 潜在的生殖挫折感**：一项心理分析探讨了《星际迷航：下一代》（*Star Trek: The Next Generation*）中 Lore 的情绪不稳定性是否与其无法繁衍后代有关。
   - 它与探讨人造生命对繁衍和遗产渴望的其他科幻叙事进行了类比。
- **人体大脑中发现微塑料**：研究人员在人体大脑样本中发现了浓度惊人的微塑料，引发了人们对长期健康影响的担忧。
   - 这一发现表明大脑是微塑料积累的主要部位，突显了塑料污染对人类健康的潜在威胁。
- **规划成功的退休生活**：退休被描述为一个需要精心规划的关键转型期，包括财务管理、健康维护和社交参与。
   - 文章强调了寻找充实的第二职业、保持健康习惯以及管理投资对于确保安全且满意的退休生活的重要性。
- **Jonathan Ive 在旧金山的投资热潮**：前 Apple 设计主管 Sir Jonathan Ive 在旧金山的 Jackson Square 街区投入巨资，购入了价值超过 1.37 亿美元的房产。
   - 他的投资预示着对该地区的宏伟计划，表明由设计界知名人物领导的重大转型。
- **Nightcore 音乐的影响**：Nightcore 是一种以加速和升调为特征的音乐流派，在 TikTok 等平台上广受欢迎。
   - 虽然该流派可能会带来暂时的情绪提升和压力减轻，但 Nightcore 对心理健康的长期影响仍不确定，需要进一步研究。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/how-is-perplexity-ai-different-aKjF.kNwSEyuc0v46TZwzw#0">Perplexity AI 有何不同？</a>: Perplexity AI 在几个关键方面区别于其他 AI 聊天机器人和搜索引擎：1. 多模型方法：- Perplexity AI 利用多个大型...</li><li><a href="https://www.perplexity.ai/page/microplastics-found-in-human-b-br1yKSQzT_W4M0NkS4iADA">人体大脑中发现微塑料</a>: 最近的研究揭示了一个令人不安的现实：微塑料（直径小于 5 毫米的微小塑料颗粒）正在侵入人体大脑并...</li><li><a href="https://www.perplexity.ai/page/nightcore-s-mental-impact-Zan5T5AlRfOW0buspS4xtw">Nightcore 对心理的影响</a>: Nightcore 以其加速和升调的音乐而闻名，在 TikTok 等平台上变得流行。这种流派可能会影响听众的情绪、能量和...</li><li><a href="https://www.perplexity.ai/search/how-to-play-garen-in-league-of-iAhjBQX8QmOt5I9BCI5JxA">如何在《英雄联盟》中玩盖伦？</a>: 要在《英雄联盟》中有效地玩盖伦，了解他的技能、游戏策略和出装至关重要。这是一份关于...的综合指南。</li><li><a href="https://www.perplexity.ai/search/what-are-the-benefits-of-using-4I6AKFtTSQervSShG7i7bQ#0">在仪表板报告中，使用 Smartsheet 相比 Power BI 有哪些优势...</a>: 在为拥有大量利益相关者的项目管理办公室 (PMO) 小型核心团队决定使用 Smartsheet 还是 Power BI 进行仪表板报告时...</li><li><a href="https://www.perplexity.ai/page/psychological-analysis-of-lore-MqJBIvaiTr2rjQMdvFRE0Q">潜在生殖挫折可能带来的毁灭性情绪影响...</a>: Lore 是《星际迷航：下一代》中 Data 的反派机器人兄弟，他作为一个引人入胜的案例，展示了人工智能在应对...时的挣扎。</li><li><a href="https://www.perplexity.ai/page/jonathan-ive-jackson-square-VPRkJtEPSt2VlvlvNxTPKw">Jonathan Ive - Jackson Square</a>: 传奇的前 Apple 设计主管 Sir Jonathan Ive 一直在悄悄重塑旧金山的一个历史角落。自 2020 年以来，Ive 和他的设计公司...</li><li><a href="https://www.perplexity.ai/page/euntoe-junbi-gaideu-EdRmAZrLRU6G6iXibqU2xA">退休准备指南</a>: 退休是开启人生新篇章的重要转折点，系统的准备是必不可少的。通过财务管理、健康维护、社交关系建立等各方面的准备，可以享受稳定且满意的晚年生活。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275947434648010884)** (6 messages): 

> - `Perplexity API`
> - `Domain Filtering`
> - `Citations` 


- **特定按钮的 API 访问**：一位用户询问如何通过 Perplexity API 访问特定的按钮功能。
- **API 用户的 Domain Filtering 和 Citations**：目前最佳选择是等待 Domain Filtering（域名过滤）和 Citations（引用）功能对 API 用户正式开放。
- **Perplexity API 文档**：用户提供了一个指向 Perplexity API 关于 chat completions（对话补全）文档的链接。



**提到的链接**：<a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：为给定的聊天对话生成模型的响应。

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1275957503787798693)** (12 messages🔥): 

> - `Github Desktop`
> - `Git send-email`
> - `Git am`
> - `Modular Community Manager`
> - `Modular Calendar` 


- **Github Desktop：好用吗？**：一位用户发现 [Github Desktop](https://github.com/apps/desktop) 并不像预期的那样直观，评论道 *“这不是有史以来最直观的产品”*。
   - 他们还指出该工具缺乏对 `git send-email` 和 `git am` 的支持，而用户认为这些命令对于管理变更非常有用。
- **欢迎 Caroline！**：Modular 的新任社区经理 Caroline 介绍了自己，并提到她曾在 Streamlit 担任社区/开发者关系（DevRel）的工作背景。
   - 她邀请成员预约虚拟咖啡时间，以讨论他们在社区中的体验和反馈。
- **Zira 之战已完成**：一位用户宣布“Battle for Zira”已完成，并使用胜利表情庆祝这一成就。
   - 未提供关于该战斗性质或其意义的进一步细节。
- **未找到 Modular 'clean' 命令**：一位用户尝试使用命令 `modular clearn`，但收到错误提示该命令不存在。
   - 他们很可能是想使用 `modular clean` 命令，该命令用于清理 Modular 项目。



**提到的链接**：<a href="https://modul.ar/caroline-calendar.">Modular Community Chat ☕ - Caroline Frasca</a>：未找到描述

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1275899785886568550)** (33 messages🔥): 

> - `math.isclose open source`
> - `Mojo github search issues`
> - `Mojo Docs Search`
> - `Mojo async/sync performance`
> - `Mojo stdlib and import paths` 


- **math.isclose 未开源**：一位成员询问为什么 **math.isclose** 没有开源。
   - 另一位成员指出 **math.isclose** 可以在 **Mojo GitHub 仓库** 的 **nightly** 分支中找到。
- **Mojo GitHub 搜索问题**：一位成员报告在通过 **GitHub search** 查找 Mojo 相关文件时遇到问题。
   - 另一位成员建议该用户可能没有切换到仓库的 **nightly 分支**。
- **请求改进 Mojo 文档搜索**：几位成员评论称需要 **改进 Mojo 文档的搜索功能**。
   - 他们建议增加按 **Mojo stdlib 模块**、**MAX lib**、**博客和新闻** 以及其他选项进行搜索过滤的功能。
- **Mojo Async/Sync 性能考量**：一位成员询问 Mojo 中 **async 函数的性能**，并将其与 **Python** 进行对比。
   - 其他成员建议目前使用 **sans-io HTTP 实现** 可能会更好，因为 **IO** 部分仍在开发中。
- **不同文件夹中文件的 Mojo 导入路径**：一位成员询问在 Mojo 项目中 **跨文件夹导入函数** 的惯用法。
   - 该问题的提出是对比了 **Python** 中利用 **sys.path** 进行此类导入的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modular.com")">无标题</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/math/math.mojo#L983">mojo/stdlib/src/math/math.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/test/builtin/test_simd.mojo#L1372">mojo/stdlib/test/builtin/test_simd.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1275947455913136138)** (8 messages🔥): 

> - `Mojo/MAX installation issues`
> - `Modular CLI`
> - `Debian installation`
> - `Mac installation`
> - `venv activation` 


- **MacOS 上的 Mojo/MAX 安装问题**：一位用户报告称，每次关闭 MacBook Pro 后都必须重新安装 Mojo 和 MAX。
- **Debian 上的 Modular CLI 安装**：一位用户尝试使用 `sudo apt-get install modular=0.9.2` 在 Debian 上安装 Modular CLI，但收到错误消息提示该软件包不可用。
- **解决缺失的 Modular Debian 源问题**：一位用户建议 Modular Debian 源可能已被移除。
- **向 Modular 仓库报告问题**：一位用户建议遇到问题的用户应在 GitHub 上的 Modular 仓库中创建一个新的 issue。
- **激活虚拟环境**：一位用户提醒另一位用户激活 venv 或 conda 环境，并提供了命令 `source ~/max-venv/bin/activate`。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/install">安装 MAX &amp; Mojo | Modular 文档</a>: 欢迎来到 MAX 安装指南！这将同时安装 MAX 和 Mojo🔥。</li><li><a href="https://github.com/modularml/max/issues">Issues · modularml/max</a>: 一个展示 MAX 平台实力的示例程序、笔记本和工具集合 - Issues · modularml/max
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1275940061489397800)** (21 messages🔥): 

> - `RAG Development & Evaluation Strategies`
> - `Cohere x w&b Webinar`
> - `Cohere Developer Office Hours`
> - `C4AI Program`
> - `Chunking XLSX Files` 


- **Cohere 与 Weights & Biases 举办的 RAG 网络研讨会**：Cohere 和 Weights & Biases 正在举办一场关于 RAG 开发和评估策略的网络研讨会，由 Cohere 的 Maxime Voisin 提供见解。
   - 立即注册网络研讨会：[https://wandb.ai/site/resources/events/cohere-webinar-august](https://wandb.ai/site/resources/events/cohere-webinar-august)。
- **Discord 上的 Cohere 开发者办公时间**：Cohere 每月在 Discord 上举办开发者办公时间，以回答社区问题。
   - 下一期嘉宾为 Sandra，计划于 [https://discord.gg/rQTg96u7?event=1275152691818922024](https://discord.gg/rQTg96u7?event=1275152691818922024)。
- **C4AI：Cohere 的非营利研究实验室**：C4AI 是 Cohere 的非营利研究实验室，专注于解决复杂的机器学习问题。
   - 它支持基础研究，探索未知领域，并寻求为机器学习研究创造更多切入点。
- **为 RAG 分块 XLSX 文件**：一位成员询问了关于为 RAG 分块 XLSX 文件的建议。
   - 另一位成员建议使用常规的分块方法，并利用专为处理原始数据而设计的 embeddings 和 rerankers。他们还提到将 XLSX 转换为 Markdown 以保留表格。



**提及的链接**: <a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI 是一个致力于解决复杂机器学习问题的非营利研究实验室。我们支持探索未知的基本研究，并专注于创造更多...

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1275935298785644694)** (18 messages🔥): 

> - `LLM Agent Task Determination` (LLM Agent 任务判定)
> - `Cohere API 403 Error` (Cohere API 403 错误)
> - `VPN and API Access` (VPN 与 API 访问)
> - `Chunking XLSX Files` (XLSX 文件分块)
> - `C4ai` 


- **使用 ReAct 进行 Agent 任务判定**：一位成员询问了 LLM Agent 根据问题确定执行哪项任务的最佳方式，前提是该 Agent 使用了向量数据库、SQL 查询和其他 API。
   - 另一位成员建议使用 ReAct 框架，该框架涉及向模型提供标题和描述，使其能够根据输入自动找出必要的行动。
- **Cohere API 403 Error: Forbidden**：一位用户在从本地服务器向 Cohere API 发送 POST 请求时遇到了 403 Forbidden 错误。
   - 用户怀疑问题可能与他们的服务器 IP 地址未被列入白名单有关，并分享了请求详情，包括端点、Content-Type、Authorization 请求头和请求体。
- **VPN 与 API 访问：403 错误的潜在原因**：一位用户报告称他们位于法国，项目经理将提供一个标识符，以解决一名使用公司法国 VPN 的开发人员遇到的问题。
   - 用户询问开发人员的 VPN 地址在阻止列表中是否可能是导致 403 错误的原因。
- **XLSX 文件分块：寻求技巧**：一位成员请求关于 XLSX 文件分块（Chunking）的指导，特别是询问处理此任务的最佳方法技巧。
- **C4ai: Cohere for AI**：一位成员热情地表达了对 Cohere 及其 AI 能力的支持。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1276172058778406982)** (7 messages): 

> - `403 Forbidden`
> - `Cohere API from R` (在 R 语言中使用 Cohere API)
> - `Cohere API using curl` (使用 curl 调用 Cohere API)
> - `Cohere Command R+ 128k context` (Cohere Command R+ 128k 上下文)
> - `OpenAI structured outputs` (OpenAI 结构化输出)


- **403 Forbidden 错误**：一位用户报告称，从本地服务器通过 Cohere API 发送请求时收到 “403 Forbidden” 错误。
   - 他们询问是否是因为 Cohere 的白名单中缺少其服务器的 IP 地址。
- **在 R 语言中使用具有 128k 上下文的 Cohere Command R+**：一位用户询问如何在不使用 Python 封装库的情况下，从 R 语言访问具有 128k 上下文的 Cohere Command R+。
   - 他们特别询问是否可以通过 HTTP 请求甚至 `curl` 来使用它。
- **使用 curl 调用 Cohere API**：一位用户得到确认，可以使用 `curl` 访问具有 128k 上下文的 Cohere Command R+。
   - 他们提供了一个完整的 `curl` 命令示例，演示了如何向 Cohere API 发起请求，包括必要的请求头和 JSON 负载。
- **Cohere API 中的结构化输出**：一位用户询问 Cohere API 中是否存在类似于 OpenAI “Structured Outputs”（结构化输出）的概念。
   - 用户提供了 OpenAI 关于结构化输出的文档链接作为背景参考。



**提及的链接**：<a href="https://docs.cohere.com/reference/chat">Chat Non-streaming — Cohere</a>：生成对用户消息的文本响应。要了解如何将 Chat API 与 Streaming 和 RAG 结合使用，请参阅我们的文本生成指南。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1276175957933166633)** (6 messages): 

> - `Jozu Hub`
> - `Cohere Model Support` (Cohere 模型支持)
> - `ModelKit` 


- **Jozu Hub：您的 AI 项目总部**：团队发布了其项目 **Jozu Hub** 的早期预览版，旨在成为版本控制和共享 AI 项目的集中平台。您可以在此处访问：[https://jozu.ml](https://jozu.ml)。
   - **ModelKit** 功能有助于消除在理解构成 AI 项目的组件（数据集、代码、参数、模型版本、文档等）时涉及的猜测工作。
- **Cohere 模型支持即将推出**：团队确认他们正在努力在 **Jozu Hub** 上集成对 **Cohere 模型** 的支持。
   - 他们的目标是将所有主流模型（包括 Cohere）打包并托管在平台上。



**提及的链接**：<a href="https://jozu.ml)">未找到标题</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1275895924480020552)** (30 条消息🔥): 

> - `Expected Value of an Item`（物品的期望值）
> - `AI Assistance with Math Problems`（AI 辅助解决数学问题）
> - `Ideogram 2.0 Review`（Ideogram 2.0 评测）
> - `SwarmUI & Flux`
> - `French Community`（法语社区）


- **物品的期望值**：一位用户正尝试计算游戏中某个物品的期望成本，他们有多次尝试机会，每次成本不同。
   - 他们想知道在仅尝试到获得物品或失败四次（最终成本为 200）的情况下的期望成本。
- **AI 辅助解决数学问题**：用户发现很难从 Gemini, ChatGPT 和 Claude 等 AI 处获得正确答案。
   - 另一位用户建议使用 Python 进行计算，因为它可以运行 Python 代码并提供更准确的结果。
- **Ideogram 2.0 评测**：一位用户对新款 Ideogram 2.0（一款图像生成工具）印象深刻，但指出下载 PNG 需要付费订阅。
   - 他们看到了其他人展示的更出色的 Ideogram 2.0 文本生成示例，并认为其效果“好得惊人”。
- **SwarmUI 与 Flux**：用户对 SwarmUI 评价很高，这是一个支持 NVIDIA/AMD GPU 的安装程序，封装了 ComfyUI 的复杂性。
   - 他们将其描述为拥有简单的 UI，并且能够加载他人的工作流。
- **法语社区**：一位用户正在寻找法语社区，因为他们目前的新闻来源仅限于美国的信源。
   - 他们询问社区中是否有 “frogs”（法语使用者）。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1275893591985750158)** (13 条消息🔥): 

> - `GPT Training Data`（GPT 训练数据）
> - `Life Coach App`（人生教练应用）
> - `Custom GPTs`（自定义 GPTs）
> - `GPT4o vs GPT4`
> - `GPT Formatting`（GPT 格式化）


- **哪里可以购买 GPT 训练数据**：一位用户询问哪里可以购买数据来为“人生教练”应用训练 GPT。 
   - 他们专门寻找问答数据集，这类数据可以在 Hugging Face Datasets 和 Kaggle 等平台上找到。
- **自定义 GPTs：GPT4o 对比 GPT4**：用户询问是否可以让自定义 GPTs 使用 GPT4 而不是 GPT4o。
   - 目前没有关于如何实现这一点的进一步信息。
- **创建自定义 GPTs：技巧与资源**：用户寻求创建自定义 GPTs 的优质资源，包括文章和视频。
   - 他们正在寻求改进 GPT 创建流程的建议，并且已经创建了多个 GPT。
- **GPT 格式一致性**：用户尝试创建一个生成《龙与地下城》（D&D）攻击、闪避和反击内容的 GPT，但在响应格式不一致方面遇到困难。
   - 他们正在寻找获得一致输出的方法，例如提供清晰的示例或格式指令。
- **ChatGPT 误解**：一位用户寻求帮助创建生成 D&D 内容的 GPT，但 ChatGPT 似乎将请求误解为 API 实现。
   - 用户寻找的是 ChatGPT 上的自定义 GPT，而不是 API 实现，并建议重新组织语言以澄清意图。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

madame_architect: 我不确定我是否完全明白你的要求。你能多说一点吗？
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

madame_architect: 我不确定我是否完全明白你的要求。你能多说一点吗？
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1275946686853939422)** (10 messages🔥): 

> - `Open Source AI Models`
> - `Eleuther.ai`
> - `DPO Fine-tuning`
> - `Instruction Prompt Templates`
> - `Multi-turn Chat Data Prep` 


- **开源 AI 往往名不副实**：许多生成式 AI 模型声称自己是开源的，但它们并不披露训练集的内容，并且可能使用了事实错误、有偏见或受版权限制的数据。
   - 美国政府正在权衡开源 LLM 的风险，而围绕 AI 模型限制的混乱被描述为“开源洗白”（open washing）。
- **Eleuther.ai：开源 AI 的典范**：Eleuther.ai 被认为是一个纯粹的开源 AI 倡议，旨在推动科学进步，其背后没有商业模式。
   - 不够开放的模型通常在其法律许可中包含隐藏条款，限制了某些使用场景。
- **使用指令提示词进行 DPO 微调**：一位用户询问在为指令模型进行微调时，是否应该将指令提示词模板应用于 DPO 数据集。
   - 几位用户确认，通常建议应用指令提示词，因为这样可以更有效地配合提示词使用模型。
- **为微调准备多轮对话数据**：一位用户询问如何为微调基础模型准备多轮对话数据。
   - 用户建议使用第一个问题和答案作为 src、trg，或者将整个对话作为输入并将最后一个答案作为目标，或者将 question1 作为输入并将 answer1 作为目标，然后将 q1, a1, q2 作为输入并将 a2 作为目标。
- **用于多轮对话数据的 Alignment-handbook**：一位用户寻求准备多轮对话数据进行微调的指导，另一位用户向其推荐了 alignment-handbook。
   - 用户澄清说，提供的代码片段与第二种方法一致，即使用整个对话作为输入，最后一个答案作为目标。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/alignment-handbook/blob/27f7dbf00663dab66ad7334afb7a1311fa251f41/src/alignment/data.py#L80">alignment-handbook/src/alignment/data.py at 27f7dbf00663dab66ad7334afb7a1311fa251f41 · huggingface/alignment-handbook</a>: 使语言模型与人类及 AI 偏好对齐的稳健方案 - huggingface/alignment-handbook</li><li><a href="https://archive.is/i2Hi7">警惕“开源”AI | LeadDev</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1275995741546090569)** (27 messages🔥): 

> - `Model Evaluation`
> - `Model Performance Degradation`
> - `Benchmark Track Rebuttals`
> - `Model Merging`
> - `Model Distillation` 


- **在新语料库上直接评估预训练模型**：一位用户质疑在不同的预训练语料库上直接评估预训练模型的有效性，认为领域不匹配可能导致结果不准确。
   - 其他人则认为这是一种常见做法，并且在仅经过 100 个训练步骤后，评估损失就会显著下降，这表明知识具有明显的迁移性。
- **模型性能退化**：一位用户询问如何可靠地降低 LLM 在 MMLU 等 Benchmark 上的性能，旨在通过修改大型模型来模拟小型模型的性能。
   - 建议包括向激活值添加噪声、使用 LoRA 进行模型蒸馏，或对模型进行逆向训练。
- **Benchmark Track 的 Rebuttals**：一位用户询问在向 Benchmark Track 提交论文后审稿人的回复率。
   - 该用户报告称，尽管讨论期接近尾声，但仍未收到审稿人的任何回复，这引发了对反馈和讨论可能受限的担忧。
- **模型合并策略**：讨论涉及了模型合并策略，一位用户建议将两个模型（UltraChat 和 Mistral）之间的差异应用于另一个模型（Mistral-Yarn）。
   - 尽管遭到了质疑，该用户仍保持乐观，并引用了以往类似策略的成功尝试。
- **使用 Transformer 进行芯片设计**：介绍了一篇题为《ShortCircuit: Accelerating Logic Circuit Synthesis with Transformers》的论文，提出了一种使用基于 Transformer 的架构设计布尔电路的新方法。
   - 论文详细介绍了一个结合监督学习和强化学习的两阶段过程以增强泛化能力，并提出了一种 AlphaZero 变体来处理涉及的巨大状态空间。



**提到的链接**: <a href="https://arxiv.org/abs/2408.09858">ShortCircuit: AlphaZero-Driven Circuit Design</a>: 芯片设计严重依赖于从真值表等功能描述中生成布尔电路，如与非图（AIG）。虽然深度学习的最新进展旨在加速...

  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 条消息): 

catboy_slim_: 据我所知，这是一个目前仍然公开可用的特定数据集
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1276010090201088104)** (7 条消息): 

> - `HellaSwag evaluation`
> - `lm-evaluation-harness`
> - `generate_until`
> - `filtering pipeline`
> - `log likelihood` 


- **HellaSwag 评估以及 'resps' 和 'filtered_resps'**：'resps' 和 'filtered_resps' 中的条目用于在 HellaSwag 等多选题数据集上评估模型，特别是计算 negative log likelihood。
   - 'resps' 中的第一个数组似乎代表每个选项的 log likelihood 值，而 'filtered_resps' 似乎用于 'generate_until' 任务，在应用 filtering pipeline 后使用。
- **理解 'resps' 结构**：'resps' 列表是嵌套的，表示每个选项的 log likelihood 值，其中 'False' 表示模型没有选择该选项。
   - 'resps' 中额外的嵌套可能是评估过程内部结构的结果，而 'filtered_resps' 列表用于在计算指标之前过滤响应。
- **'generate_until' 中的 Filtering Pipeline**：'generate_until' 任务涉及生成输出直到满足特定条件，'filtered_resps' 列表专门用于此类场景。
   - 'filtered_resps' 列表用于在计算指标之前过滤输出，据推测是为了确保生成的输出符合特定标准。
- **Log Likelihood 和 'False' 值**：Negative log likelihood (NLL) 是语言建模中的常用指标，较低的值表示更好的性能。
   - 输出中的 'False' 值表示根据计算出的 NLL 值，模型没有选择该特定选项作为最可能的选项。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1673">All responses are considered false? · Issue #1673 · EleutherAI/lm-evaluation-harness</a>: 我正在使用 negative log likelihood 在我的自定义数据集上评估几个 LLM，我一直认为具有最高 likelihood 的响应会与 'true' 配对，而其他的则...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1673#issuecomment-2040371764">All responses are considered false? · Issue #1673 · EleutherAI/lm-evaluation-harness</a>: 我正在使用 negative log likelihood 在我的自定义数据集上评估几个 LLM，我一直认为具有最高 likelihood 的响应会与 'true' 配对，而其他的则...
</li>
</ul>

</div>
  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1276118034133549078)** (37 messages🔥): 

> - `LightGBM`
> - `LSTM`
> - `Commodity Price Forecasts` (大宗商品价格预测)
> - `Kaggle Competitions` (Kaggle 竞赛)
> - `Production Data` (生产数据)


- **LightGBM 在 Kaggle 竞赛中的主导地位**：LightGBM 已广泛应用于 Kaggle 竞赛，包括 Corporación Favorita 超市销售预测、Recruit 餐厅访客预测以及 M5 准确性竞赛。
   - 这导致人们认为它即使在生产环境中也具有卓越的性能。
- **LightGBM 在生产环境中的表现：争议**：然而，一些人认为 LSTM 在生产环境中可能是更好的选择。
   - 关于 LightGBM 在现实场景中的有效性存在争议，有人声称与在 Kaggle 竞赛中的表现相比，它在实时数据上的表现可能没那么好。
- **调查 LightGBM 在大宗商品价格预测中的应用**：一项研究探讨了使用 LightGBM 进行大宗商品价格预测，理由是它在 M5 竞赛中取得了成功。
   - 作者采用了 SMA, EMA 和 VWAP 等特征，但其结果表明，在铅和锡的收益率预测上，ARIMA 模型的表现优于 LightGBM。
- **预测任务的重要性**：模型的选择取决于具体的预测任务，例如预测下一阶段的价格与进行长期预测（如 3-6 个月）。
   - 虽然 LightGBM 可以通过递归馈送生成多步预测，但考虑预测的背景和复杂性至关重要。
- **深度学习之前的替代方法**：SMA, EMA 和 ARIMA 模型通常是时间序列预测的有效起点。
   - 在处理大量非传统外生变量时，像 LightGBM 和 LSTM 这样的模型可能更有用，因为在这种情况下，捕捉季节性可能不那么关键。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/MachineLearning/comments/ob2rll/d_how_often_is_lightgbm_used_for_forecasting_in/">Reddit - 深入探讨一切</a>：未找到描述</li><li><a href="https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html">第 8 章：时间序列预测中最成功的模型 —— 时间序列分析手册</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1276216575686344847)** (31 messages🔥): 

> - `AI burnout` (AI 倦怠)
> - `Frontier Labs` (前沿实验室)
> - `Greg Brockman's Work Hours` (Greg Brockman 的工作时长)
> - `Twitter anxiety` (Twitter 焦虑)


- **AI 倦怠**：一位成员表示，AI 领域确实比其他领域更容易让人感到倦怠，这是他们担心的一个问题。
   - 另一位成员表示赞同，但补充说用户倦怠也是真实存在的，这两者是相互关联的。
- **Frontier Labs 的工作强度**：一位成员指出，Frontier Labs 的人们工作非常努力，这似乎是不可持续的。
   - 他们提到自己虽然在科技行业的职业生涯不算长，但正试图劝导人们不要走向崩溃的边缘。
- **Greg Brockman 的工作时长**：一位成员提醒大家，Greg Brockman 在 Twitter 上发布了他的时间追踪记录，显示一周的编码工作时间长达 97 小时。
   - 他们对他没有早点休息感到惊讶；他花了 9 年时间才休了一次假！
- **Twitter 焦虑**：一位成员表达了从偏远地区完全断网回来后，面对 Twitter 和 AI 讨论感到很不舒服。
   - 他们说 Twitter 是其中最糟糕的部分，非常容易诱发焦虑。


  

---

### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1276024667378942035)** (6 条消息): 

> - `Lilian Weng Blog`
> - `Diffusion Models`
> - `Generative Models`
> - `Distillation`
> - `Score-based generative modeling` 


- **Lilian Weng 关于 Diffusion Models 的博客文章**：Lilian Weng 在 2021 年 7 月撰写的关于 Diffusion Models 的博客文章，包含了她关于其他类型 Generative Models（如 GANs、VAEs 和 Flow-based models）的博客链接。
   - 该博客已多次更新，最近一次更新于 2024 年 4 月 13 日，增加了关于 progressive distillation、consistency models 和模型架构的章节。
- **关于 Diffusion Models 与 Distillation 的澄清**：一位用户最初混淆了 Diffusion Models 和 Distillation。
   - 另一位用户纠正了这一误解，解释说 Distillation 是一个不同的概念。
- **Distillation 笔记**：一位用户提到他们有一些关于 Distillation 的基础笔记准备包含在帖子中，但尚未发出。



**提到的链接**：<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">What are Diffusion Models?</a>：[2021-09-19 更新：强烈推荐 Yang Song（参考文献中几篇关键论文的作者）关于 score-based generative modeling 的这篇博客]。[2022-08-27 更新：增加了 classifier-free...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1276023253022019584)** (33 条消息🔥): 

> - `LLM for NL to SQL`
> - `Prebuilt Queries for SQL`
> - `LLM Accuracy and SQL`
> - `RAG for SQL`
> - `ChromaDB RAG for CSV` 


- **用于 NL to SQL 的本地 LLMs**：一位用户询问了使用本地 LLM 进行自然语言到 SQL 转换的可行性。
- **用于 SQL 的预构建查询**：一位用户建议使用带有占位符的预构建查询，以简化 Text to SQL 的工作负载。
- **LLM 准确率与 SQL**：讨论集中在解决 SQL 生成中的错误，这些错误会导致无法执行的查询。
- **使用 CodeLLM 的 SQL RAG**：提出了一种将 Retrieval Augmented Generation (RAG) 与特定代码的 LLM 相结合的方法，以改进 SQL 生成。
- **使用 ChromaDB 对 CSV 进行 RAG**：一位用户就如何对具有大量列的 CSV 执行 RAG 寻求建议。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1275896575029149767)** (2 条消息): 

> - `Flags for AI`
> - `4149 AI`
> - `Proactive Guidance in Slack`
> - `AI for Research` 


- **4149 AI 的新“Flags”功能**：[4149 AI](https://4149.ai) 发布了一项名为“Flags”的新 AI 驱动功能，目前正在测试中。
   - 它旨在通过 Slack 直接消息，针对团队状态提供主动的实时引导。
- **Flags 提供主动的团队洞察**：“Flags”功能旨在当团队出现下滑迹象时立即发送警报。
   - 它还旨在帮助在问题演变成麻烦之前掌握它们，并突出团队的胜利和成就。
- **用户自定义和 Flags 审批**：用户可以自定义 AI 看到的内容，默认情况下所有消息都会发送给用户进行审批。
   - 设置过程不到一分钟，只需要一个 Slack 团队。
- **AI 在研究中的应用 - 热情**：作者表达了对 AI 在研究中应用场景的兴奋。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://4149.ai)">未找到标题</a>：未找到描述</li><li><a href="https://beta.4149.ai/register">4149 [beta]</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1275898837629800499)** (29 条消息🔥): 

> - `Ideogram 2.0`
> - `Mistral-NeMo-Minitron-8B`
> - `Sovereign AI`
> - `v0 更新`
> - `AI 发展趋势` 


- **Ideogram 2.0 发布：所有人免费使用**：Ideogram 2.0 是由前 Google Imagen 1 团队开发的最新文本生成图像（text-to-image）模型，现已向所有用户免费开放。
   - 此次发布包括一个新的 iOS 应用、一个测试版 API 以及 Ideogram Search，目前已创作超过 10 亿张图像。
- **Nvidia 发布 Mistral-NeMo-Minitron-8B**：Nvidia 发布了 Mistral-NeMo-Minitron-8B，这是一个通过对 Mistral-NeMo 12B 进行剪枝和蒸馏获得的基座 LLM。
   - 该模型在 9 个基准测试中的 8 个表现优于 Mistral-7B 和 LLaMa-3.1-8B，并已在 Hugging Face 上以宽松许可证发布。
- **Sovereign AI：一种新型流数据系统**：Infinite ML 播客介绍了 Sovereign AI，这是由 Redpanda Data 开发的一种流数据系统。
   - 讨论涵盖了流数据的定义、此类系统的演变以及 Sovereign AI 在实践中的工作原理等主题。
- **v0 对话式 UI：Next.js、React 及更多**：v0 的新对话式 UI 现已进入测试阶段，具备 Next.js、React 和其他 Web 技术的最新知识。
   - 它拥有改进的客户端组件支持、运行 npm 包的能力、更快的流式传输，并提供了多个展示其功能的示例。
- **AI 发展趋势：增长放缓**：对 GitHub 仓库创建数据的分析表明，AI 开发的增长已经放缓，从指数级转向线性增长。
   - Azure 和 OpenAI 在 2024 年 3 月表现出强劲的采用率，而 Amazon Bedrock 的增长可能在 2024 年 6 月达到顶峰。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/v0/status/1826020673908535325">来自 v0 (@v0) 的推文</a>：v0 的新对话式 UI：• 最新的 Next.js、React 和 Web 知识 • 改进的客户端组件支持 • 能够运行像 framer-motion 这样的 npm 包 • 更快、更可靠的流式传输...</li><li><a href="https://x.com/_philschmid/status/1826292873446248922">来自 Philipp Schmid (@_philschmid) 的推文</a>：剪枝与蒸馏的故事仍在继续！@nvidia 刚刚发布了 Mistral Nemo Minitron 8B，这是一个通过对 Mistral-NeMo 12B 进行剪枝和蒸馏获得的基座 LLM！👀 TL;DR：✨ 使用了 400B token 进行...</li><li><a href="https://x.com/Smol_AI/status/1826410077194256729">来自 AI News by Smol AI (@Smol_AI) 的推文</a>：[2024 年 8 月 21 日] Ideogram 2 + 函数调用排行榜 V2。这是续作的季节。在 Flux（前 Stable Diffusion 团队）壮观发布之后，@ideogram_ai（前 Google I...</li><li><a href="https://x.com/prateekvjoshi/status/1826375520277463382">来自 Prateek Joshi (@prateekvjoshi) 的推文</a>：今天 Infinite ML 播客的主题是 Sovereign AI。我们邀请了 @emaxerrno 来节目中谈论它。他是 @redpandadata 的创始人兼 CEO。他们已经从 i... 筹集了超过 1.65 亿美元的资金。</li><li><a href="https://aiencoder.substack.com/p/querying-ai-and-cloud-trends-azure">查询 AI 和云趋势：Azure 和 OpenAI 占据主导地位但增长放缓，Amazon 可能已达顶峰</a>：穿透 AI 炒作，查询实际的开发者使用情况（带有假设），用于安全工具的优先级排序和合作伙伴指南。</li><li><a href="https://x.com/ideogram_ai/status/1826277550798278804?s=46">来自 Ideogram (@ideogram_ai) 的推文</a>：隆重推出 Ideogram 2.0 —— 我们最先进的文本生成图像模型，现在对所有用户免费开放。今天的里程碑式发布还包括 Ideogram iOS 应用的发布、测试版...</li><li><a href="https://x.com/giffmana/status/1826324924564398185">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：我们接下来会将目标移向何处？如果你能认出这三个，那就太棒了！引用 Ideogram (@ideogram_ai)：隆重推出 Ideogram 2.0 —— 我们最先进的文本生成图像模型，现在对所有...</li><li><a href="https://x.com/GoogleAI/status/1826363610680979925">来自 Google AI (@GoogleAI) 的推文</a>：今天我们推出了一种新的零样本、跨语言语音转换模块，它可以轻松插入到最先进的 TTS 系统中，以恢复患有构音障碍的说话者的声音。看看它在...</li><li><a href="https://github.com/autogen-ai">autogen-ai</a>：autogen-ai 有 3 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1276233199101542432)** (3 messages): 

> - `GPT-4o fine-tuning`
> - `Genie coding agent`
> - `SWE-Bench`
> - `RAG`
> - `In-context learning` 


- **GPT-4o Fine-Tuning：值得吗？**：Latent Space 播客与来自 Cosine 的嘉宾 Alistair Pullen 探讨了对 GPT-4o 进行 fine-tuning 是否值得。
- **Genie 的大规模 Fine-Tuning 投入**：Genie 为 GPT-4o 进行了前所未有的 fine-tuning 工作，利用了源自真实用户日志的数十亿 token 的合成代码数据，并有目的地破坏了抽象语法树（ASTs）。
- **合成代码数据生成**：播客深入探讨了如何使用真实用户日志和 ASTs 创建数十亿 token 的合成代码数据。
- **OpenAI 发布 GPT-4o Fine-Tuning 功能**：OpenAI 推出了 GPT-4o 的 fine-tuning 功能，允许开发者提升其应用程序的性能和准确性。
- **大规模 Fine-Tuning 的意义**：播客讨论了大规模 fine-tuning 对 GPT-4o 性能的影响，特别是在代码生成场景下。



**Link mentioned**: <a href="https://x.com/swyx/status/1826673380294267328">Tweet from swyx.ai (@swyx)</a>: 🆕 @latentspacepod: Is finetuning GPT4o worth it?  w/ @AlistairPullen of @cosine_sh  Betteridge's law says no: with 59 different flavors of RAG, and >2million token context + prompt caching, it...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275913371534885016)** (16 messages🔥): 

> - `Open Interpreter usability`
> - `Open Interpreter searching issues`
> - `Open Interpreter model issues`
> - `Open Interpreter feature requests`
> - `Open Interpreter model suggestions` 


- **Open Interpreter 搜索问题**：一位用户报告称，Open Interpreter 中的网页搜索仅在完全刷新终端和 OI 时有效，在继续对话时无效。
- **Open Interpreter 模型建议**：一位用户建议 Phi-3.5-mini 和 Qwen2 模型表现出奇地好。
- **Open Interpreter 模型类型**：一位用户询问另一位用户正在使用哪种模型，怀疑可能不是 GPT-4。
- **Open Interpreter 界面关注点**：一位用户表示，在界面外记录配置文件和路径比随着时间变化的命令行书签更不容易引起混淆。


  

---

### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1276254821539643503)** (1 messages): 

> - `Jamba 1.5`
> - `Jamba 1.5 Mini`
> - `Jamba 1.5 Large`
> - `SSM-Transformer Architecture`
> - `Long Context Handling` 


- **Jamba 1.5: 推出新一代**：AI21 Labs 推出了 **Jamba 1.5 Mini**（12B 激活/52B 总计）和 **Jamba 1.5 Large**（94B 激活/398B 总计），由结合了 **Transformer 的质量**与 **Mamba 的效率**的新颖 **SSM-Transformer Jamba 架构**驱动。
   - 这些模型拥有 **256K 有效上下文窗口** —— 市场上最长的窗口 —— 并且在处理长上下文时，速度比同类其他模型快达 **2.5 倍**。
- **Jamba 1.5 Mini: 质量与速度的领导者**：**Jamba 1.5 Mini** 的表现超越了其尺寸级别，在 **Arena Hard** 上获得了 **46.1** 分，而 **Jamba 1.5 Large** 得分为 **65.4**，超过了 **Llama 3.1 70B** 和 **405B**。
   - 模型支持 **英语、西班牙语、法语、希伯来语、阿拉伯语**等，并提供对 **JSON 输出、function calling 和文档处理**的内置支持。
- **Jamba 1.5 的开放访问**：**Jamba 1.5 Mini 和 Large** 已可在 **Hugging Face** 立即下载，并可部署在 **Together AI、AWS、GCP、Azure** 等主要云平台上。
   - AI21 Labs 在 **Jamba Open Model License** 下发布了这些模型，旨在普及高质量模型的访问并鼓励进一步的实验。
- **Jamba 1.5: 为效率而生**：**Jamba 1.5 系列**旨在快速高效地处理数千页文本、复杂代码和复杂的 Agent。
   - 此次发布标志着非 Transformer 模型首次成功扩展到领先模型的质量和强度，展示了 **SSM-Transformer 架构**的力量。
- **Jamba: 长上下文模型的未来**：AI21 Labs 认为 **Jamba 1.5 模型**是长上下文建模领域的重大进步。
   - 它们提供无与伦比的速度、效率和质量，同时提供开放模型中最长的上下文窗口。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251">Jamba-1.5 - a ai21labs Collection</a>: 未找到描述</li><li><a href="https://studio.ai21.com/v2/chat">AI21 Studio</a>: 未找到描述</li><li><a href="https://www.ai21.com/jamba">Foundation models</a>: 未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba-model-family">The Jamba 1.5 Open Model Family: The Most Powerful and Efficient Long Context Models</a>: AI21 推出的全新开放模型系列，提供无与伦比的速度、效率和质量，以及开放模型中最长的上下文窗口。
</li>
</ul>

</div>
  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1276183856206053377)** (4 messages): 

> - `Jamba-1.5 Fine Tuning`
> - `Jamba-1.5 Large` 


- **Jamba-1.5 微调不可用**：一位用户询问是否可以通过 Studio UI 对 Jamba-1.5 模型进行微调。
   - AI21 工作人员回答说，Studio 上仅提供 instruct 版本，目前不会在那里提供微调功能。
- **Jamba-1.5 Large 特性**：Jamba-1.5 Large 是 AI21 最先进的模型。
   - 它在推理、代码生成、高上下文和多语言处理方面具有先进的能力。



**提及的链接**: <a href="https://rubiks.ai/search/?id=mtjvmz-r2fb-98s4-eik5-gqzex7ilua3n">Jamba 1.5 Release</a>: **Jamba 1.5 发布：用于 Agentic AI 的混合 SSM-Transformer 模型**\\n\nAI21 宣布发布 Jamba 1.5，这是其混合 SSM-Transformer 模型的新版本，结合了...

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1276289164425433140)** (2 messages): 

> - `API Rate Limits`
> - `OpenAI API Rate Limits` 


- **OpenAI API 速率限制**：一位用户询问了 API 速率限制，随后自行回答了问题：限制为每分钟 200 次请求 (RPM) 和每秒 10 次请求 (RPS)。
- **OpenAI API 速率限制**：一位用户询问了 API 速率限制，随后自行回答了问题：限制为每分钟 200 次请求 (RPM) 和每秒 10 次请求 (RPS)。

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1276131458968326227)** (4 messages): 

> - `Code Review`
> - `mypyc Compilation`
> - `Tinygrad` 


- **响应代码审查**：一名成员对代码审查中推卸责任的回复表示沮丧，例如说“如果你想要/要求，我就做”之类的话。
   - 他们强调了作者对自己的更改负责的重要性，应批判性地思考建议，并要么实施建议，要么为不这样做提供合理的解释。
- **Tinygrad Mypyc 编译**：有人表示有兴趣致力于让 **Tinygrad** 使用 **mypyc** 进行编译。
   - 一名成员自愿研究这个问题。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1275942659185315902)** (4 messages): 

> - `Torchtune and T5`
> - `Weight Mapping` 


- **Torchtune 与 T5 的挑战**：一名成员指出 **T5** 唯一棘手的地方是其可训练的 attention bias，但其他组件基本都是标准的。
   - 他们还提到 **Torchtune** 目前不支持 encoder-decoder 架构，需要针对特定任务调整训练循环。
- **将 T5 权重映射到 Torchtune**：一位成员建议比较 **Hugging Face** 和 **Torchtune** 仓库之间的权重名称以便进行映射。
   - 具体来说，他们提到了 Hugging Face 的 **T5-small** 模型和 Torchtune 的 **convert_weights.py** 文件。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/google-t5/t5-small/tree/main?show_file_info=model.safetensors">google-t5/t5-small at main</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/861c3b214b029ee819fb554055aa467a52f22862/torchtune/models/convert_weights.py#L31-L45.">torchtune/torchtune/models/convert_weights.py at 861c3b214b029ee819fb554055aa467a52f22862 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1276230445343969412)** (3 messages): 

> - `LinkedIn Survey`
> - `Infinite Generative YouTube` 


- **LinkedIn 调查寻求用户见解**：正在进行一项调查，以了解人们对 LinkedIn 作为专业社交平台的看法。
   - 该调查旨在从社区收集关于 LinkedIn 各个方面的宝贵见解，欢迎所有人参与。
- **无限生成式 YouTube 需要开发人员**：一个团队正准备为一个无限生成式 YouTube 项目启动封闭测试，并正在寻找一位充满热情的开发人员加入他们的团队。
   - 欢迎任何有兴趣使用这些模型进行构建的人联系并了解更多机会。



**提及的链接**：<a href="https://qualtricsxmbyhfby7vf.qualtrics.com/jfe/form/SV_bDttqTVpxzTJvLw">Survey | Professional Network Platforms</a>：收集体验数据最强大、简单且值得信赖的方式。立即开始您的体验管理之旅，并试用免费账号。

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1276016243815743509)** (1 messages): 

> - `Multilingual Representation Learning Workshop at EMNLP 2024`
> - `Reviewer Sign Up`
> - `Workshop Topics` 


- **EMNLP 2024 研讨会征集审稿人**：EMNLP 2024 的多语言表示学习（Multilingual Representation Learning）研讨会正在征集审稿人。
   - 有兴趣的人员可以在[此处](https://forms.gle/ica4F94jaTbvdb689)注册成为审稿人。
- **研讨会探讨多样化的多语言 NLP 主题**：该研讨会探讨与多语言 NLP 相关的广泛主题，包括计算社会科学和文化分析、对话系统、话语和语用学以及低资源方法。
   - 其他主题包括多语言模型中的伦理和偏见、信息提取和检索、多模态、音韵学、机器翻译、资源和评估、跨语言语义学以及语音处理。



**提及的链接**：<a href="https://forms.gle/ica4F94jaTbvdb689">Call for Reviewers - 4th Multilingual Representation Learning (MRL) Workshop, EMNLP 2024 </a>：此表单供任何有兴趣担任 EMNLP 2024 研讨会审稿人的人员使用。受邀审稿人需表明其兴趣，其申请将根据...进行评估。

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276013369727647825)** (2 messages): 

> - `Website Leaderboard`
> - `Huggingface Leaderboard`
> - `Model Evaluation`
> - `Equal Importance` 


- **网站与 Huggingface 排行榜对比**：一名成员询问了网站排行榜与 Huggingface 排行榜之间的区别。
   - Huggingface 排行榜的分数似乎显著更高。
- **模型能力的同等重要性**：网站排行榜已进行更改，以体现所有子类别都同等重要的理念。
   - `python_simple_function` 中的能力被认为与 `java_simple_function` 中的能力同样重要。
- **全面的模型评估**：一个优秀的模型应该在所有方面都表现出色，而不仅仅是特定的子类别。
   - 这一变化在 #580 中被提及。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1276201059714596908)** (1 messages): 

> - `Evaluating fine-tuned model on BFCL locally` 


- **在本地 BFCL 上评估微调模型**：一位用户询问了在本地 BFCL 上评估微调模型的步骤，特别是如何利用多 GPU 进行评估过程。
- **在本地运行评估**：对话中未提供具体的步骤或建议，因此尚不清楚该用户计划如何本地在 BFCL 上评估其模型。


  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

amanshrestha: anyway we can use prompt caching , antropic api?
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1276236589286031380)** (1 messages): 

> - `Open Source AI`
> - `OSI Definition` 


- **OSI 发布开源 AI 定义草案**：开源倡议组织 (OSI) 发布了新的开源 AI 定义草案，这是技术社区和开源社区经过两年讨论和辩论的成果。
   - 这一里程碑对于在 AI 背景下重新定义“开源”的含义，以及塑造该技术对社会未来影响至关重要。
- **OSI 市政厅会议**：OSI 组织了一次市政厅活动，讨论新的开源 AI 定义草案。
   - 该活动旨在为社区提供进一步讨论和参与的平台。


  

---



### **DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1276114551028584458)** (1 messages): 

> - `German Instruction Tuning Data` 


- **用于德语指令微调的 OASST-2**：**OASST-2 数据集**包含一个德语子集，是指令微调的一个不错选择。
- **用于德语指令微调的 Aya-Dataset**：**Aya-Dataset** 也包含一个可用于指令微调的德语子集。
- **其他德语数据集**：其他德语数据集如 **Colossal Cleaned Common Crawl** and **German Wikipedia** 可能有所帮助，但你需要针对指令微调对其进行过滤和筛选。
- **自定义数据集考量**：通过将英语指令数据翻译成德语来创建**自定义数据集**，可能对特定任务有益。
- **开源你的模型**：开源你基于 Llama 3.1 的 8x8b MoE 模型（同时进行德语和英语指令微调），将是对 NLP 社区的一项宝贵贡献。


  

---



---



{% else %}


> 完整的各频道详情已为邮件格式进行删减。
> 
> 如果你想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}