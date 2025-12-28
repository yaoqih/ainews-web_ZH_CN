---
companies:
- uc-berkeley
- deepmind
- openai
- microsoft
- nvidia
- archetype-ai
- boston-dynamics
- toyota-research
- google
- adobe
- openai
- mistral
- tesla
- meta-ai-fair
date: '2024-10-22T00:04:21.441910Z'
description: '**加州大学伯克利分校的 EPIC 实验室**通过 **LOTUS** 和 **DocETL** 等项目推出了创新的大语言模型（LLM）数据算子，专注于在大规模数据集上实现高效的编程与计算。这种方法将
  **Deepmind** 和 **OpenAI** 等“GPU 资源充足”的大型实验室，与“GPU 资源匮乏”的复合 AI 系统进行了对比。


  **微软**开源了 **BitNet b1.58**，这是一种 1 比特三值参数的大语言模型，可使**训练速度提升 4-20 倍**，并能以人类阅读速度实现端侧推理。**英伟达**发布了
  **Llama-3.1-Nemotron-70B-Instruct**，这是一款经过微调的开源模型，其表现超越了 **GPT-4o** 和 **Claude-3.5-sonnet**。这些进展凸显了**模型优化**、**端侧
  AI** 以及**微调**领域的重大突破。'
id: ca91a315-7798-4ce9-bd9b-21efdb591416
models:
- bitnet-b1.58
- llama-3.1-nemotron-70b-instruct
- gpt-4o
- claude-3.5-sonnet
original_slug: ainews-docetl-agentic-query-rewriting-and
people:
- rohanpaul_ai
- adcock_brett
- david-patterson
title: DocETL：面向复杂文档处理的代理式查询重写与评估。
topics:
- model-optimization
- on-device-ai
- fine-tuning
- large-corpus-processing
- gpu-acceleration
- frameworks
- model-benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**LLM data operators are all you need.**

> 2024/10/18-2024/10/21 的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**231** 个频道和 **6066** 条消息）。预计节省阅读时间（以 200wpm 计算）：**791 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们通常将 AINews 的专题报道留给当天最具影响力的单一新闻，但这通常会导致过度偏向于回顾大型模型实验室的新闻稿。年度的其他故事则是逐渐发展的，更像是波浪而非水花，虽然规模可能没那么大，但作为多元化信息摄入的一部分仍然很有用。我们利用像这样比较平静的日子，对 [DSPy](https://buttondown.com/ainews/archive/ainews-the-dspy-roadmap/) 和 [AI 降价故事](https://buttondown.com/ainews/archive/ainews-too-cheap-to-meter-ai-prices-cut-50-70-in/) 等社区工具进行一些累积性的关注。

加州大学伯克利分校（UC Berkeley）一直是许多重大技术浪潮的领导者——根据 [David Patterson](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-123.pdf) 的说法，UCB 研究实验室 40 年的历史孕育了从 RISC、RAID 到像 Databricks 这样的大型公司的一切。这一传统的最新实验室是 [EPIC](https://x.com/UCBEPIC) —— 专注于数据的有效编程（*E*ffective *P*rogramming）、交互（*I*nteraction）和计算（*C*omputation）。我们有幸参加了[他们最近的会议](https://x.com/adityagp/status/1843695418338881941)，并对两篇类似的论文印象特别深刻：[LOTUS](https://x.com/lianapatel_/status/1813981153709441361) 和 [DocETL](https://x.com/sh_reya/status/1848415442244931861)，后者已经[引起了显著的关注](https://x.com/sh_reya/status/1838617833393283428)并最终在[今天发布](https://arxiv.org/abs/2410.12189)。两者都为大规模语料库提供了经过深思熟虑的 LLM 算子（operators）。

<img width="538" alt="image" src="https://gist.github.com/user-attachments/assets/00b20959-f486-4be8-82b5-c60c0cdf5baa">

<img width="1316" alt="image" src="https://gist.github.com/user-attachments/assets/997d3bcd-cefa-4476-a1f4-ee107ba5e759">

[GitHub 文档](https://ucbepic.github.io/docetl/operators/map/) 提供了更多关于所提议的 API 和概念的想法。从极限角度来看，这可以被视为类似于 DSPy 的“又一个 LLM 框架”，但考虑到该机构在成功思考商业相关的 Big Data 问题方面的声誉，这种对大数据的关注使得它比一般的 Twitter 匿名用户发布的项目更值得仔细研究：

<img width="881" alt="image" src="https://gist.github.com/user-attachments/assets/312632cd-3be6-40e6-9fb5-ba87c01846d7">

从最高层面来看，这只是 GPU Rich 的大型实验室（Deepmind、OpenAI）与 GPU Poor 的 [Compound AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) 方法之间持续博弈的最新战线。[DocETL 演示网站](https://www.docetl.org/#demo-docetl-output) 可以帮助您比较使用其框架与“将所有内容放入上下文（context）”之间的结果和方法。在很长一段时间内，这里可能不会有明显的赢家，AI Engineer 只需要熟悉这两者即可。

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

**AI 加速**

- **BitNet 进展**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848061956697301072) 强调了 Microsoft 开源的 BitNet b1.58，这是一个 **1-bit LLM，其中每个参数都是三值的 {-1, 0, 1}**。这种方法可以在不修改位置编码的情况下，实现 **4-20 倍的训练加速、更高的稳定性以及更好的长上下文处理能力**。该模型在 100B LLaMa 推理中达到了 1.7 tokens/second 的速度。

- **端侧 AI**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848076536714531252) 报道称 bitnet.cpp 可以在 **单个 CPU 上运行 100B BitNet b1.58 模型**，速度可与人类阅读速度媲美（每秒 5-7 个 token），显著增强了在本地设备上运行 LLM 的潜力。

**AI 模型开发与研究**

- **重大 AI 进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1848032213532651701) 总结了来自 Archetype AI, NVIDIA, Boston Dynamics, Toyota Research, Google, Adobe, OpenAI, Mistral, Tesla 和 Meta 等多家公司的重大进展。

- **新模型与基准测试**：[@adcock_brett](https://twitter.com/adcock_brett/status/1848032258159943735) 报道称 Nvidia 悄然发布了一个名为 **Llama-3.1-Nemotron-70B-Instruct** 的新型开源微调 LLM，尽管其参数量仅为 70B，但在基准测试中表现优于 GPT-4o 和 Claude 3.5 Sonnet。

- **多模态进展**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848000249920364838) 重点介绍了 Meta 发布的 **Spirit LM**，这是首个整合了语音和文本的开源多模态语言模型，提供词级交织的语音和文本数据集，并具备跨模态生成能力。

- **AI 推理能力**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1847993405164364224) 分享了 Apple 一篇论文的见解，该论文指出 **LLM 缺乏稳健的数学推理能力**，更多是依赖模式匹配而非真正的概念理解。该论文引入了 GSM-Symbolic 基准测试，用于评估 LLM 在不同问题变体下的表现。

**AI 应用与工具**

- **AI 生成艺术**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1848007989275300213) 观察到 **AI 生成的 AI 艺术表现优于人类生成的 AI 艺术**，并注意到一款受在线研究 "sigils" 启发的艺术相机 GLIF 产生了一些有趣的结果。

- **Cursor 热度**：[@vikhyatk](https://twitter.com/vikhyatk/status/1848048132929515528) 对 Cursor 的流行发表了评论，认为它相比于 Notepad 等基础文本编辑器有了显著改进。

- **LLM 工程师手册**：[@maximelabonne](https://twitter.com/maximelabonne/status/1848029371803832767) 宣布《LLM Engineer's Handbook》成为 Neural Networks 类别中排名第一的新书，旨在帮助新一代 LLM 工程师构建生产级 AI 系统。

**AI 伦理与社会影响**

- **AI 能力 vs 人类智能**：[@bindureddy](https://twitter.com/bindureddy/status/1848136882284044369) 认为，虽然 LLM 可能会在一年内遇到瓶颈，但它们已经比大多数人类更聪明。推文指出，**AI 自动化的最后一公里不是智能，而是“管道工程”（plumbing）**。

- **AI 与民主**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1847985070197383607) 对 AI 对民主的潜在影响表示担忧，称“坏的 @elonmusk 乐于将民主撕成碎片，并将其作为超市货架上的廉价商品出售。”

**梗与幽默**

- [@fabianstelzer](https://twitter.com/fabianstelzer/status/1848066835427545148) 分享了一条幽默推文，关于给一个 "namshub glifbot" 访问 Pepe lora 的权限，结果生成了以奇点为主题的 Pepes。

- [@vikhyatk](https://twitter.com/vikhyatk/status/1848048132929515528) 调侃了 Cursor 的热度，称它“感觉一定比 notepad.exe 有了巨大的进步”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：LLM 架构与训练的进展**

- **nGPT: Faster Convergence by Performing Optimization on a Hypersphere** ([Score: 126, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1g8cba0/ngpt_faster_convergence_by_performing/)): **nGPT** 是由 **Nvidia** 开发的一种新型 GPT 变体，它将向量限制在**超球面 (hypersphere)**上，导致其收敛速度比传统 GPT 模型快 **4 到 20 倍**，并提升了对长文本序列的处理能力。这种方法通过消除对权重衰减 (weight decay) 或特殊学习率 (learning rate) 调整的需求简化了训练，同时分析显示，attention 和 MLP 模块对隐藏状态 (hidden states) 的调整更小，且归一化缩放因子在各层之间保持稳定。[nGPT 论文](https://arxiv.org/html/2410.01131)将其展示为一种构建更高效、更强大语言模型的极具前景的方法。

- **COGNITIVE OVERLOAD ATTACK: PROMPT INJECTION FOR LONG CONTEXT** ([Score: 33, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1g8bwkw/cognitive_overload_attack_prompt_injection_for/)): 该研究探讨了针对大语言模型 (LLMs) 的**认知过载攻击 (Cognitive Overload Attacks)**，将人类认知与 LLM 在信息过载下的行为进行了类比。研究人员证明，攻击者可以利用这一漏洞绕过 **GPT-4** 和 **Claude-3-Opus** 等先进模型的安全机制，攻击成功率高达 **99.99%**。作者建议将神经科学中的**认知负荷管理**技术引入 AI 设计，以增强 LLM 抵御此类对抗性攻击的韧性。

**主题 2：面向开发者的创新 LLM 框架与工具**

- **GraphLLM now has a GUI: open source graph based framework for performing inference with a LLM** ([Score: 114, Comments: 11](https://reddit.com//r/LocalLLaMA/comments/1g816ee/graphllm_now_has_a_gui_open_source_graph_based/)): **GraphLLM** 是一个开源的基于图的 LLM 推理框架，现在配备了类似于 ComfyUI 的 **GUI**，允许将节点输出实时流式传输到前端。该框架支持**循环 (loops)**、**并行执行**、**条件判断**和**自定义 Python 代码执行**等高级功能，同时在提示词处理方面保持透明，并提供各种预构建示例，包括 **YouTube 字幕摘要**、**多数投票**以及一个**能够进行网页搜索和文件访问的 Agent**。其他工具还包括使用无头 Firefox 实例处理动态网站的**网页爬虫**、**YouTube 字幕下载器**和 **PDF 解析器**，源代码可在 [GitHub](https://github.com/matteoserva/GraphLLM) 获取。

- **Generate text with alternative words and probabilities** ([Score: 60, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1g83jii/generate_text_with_alternative_words_and/)): **ActuosusAI** 是一个个人兴趣项目，它引入了一项功能，允许用户通过在指定 **temperature** 的同时导航备选路线来修改 **LLM 输出**，并为 token 采样设置了最小 **0.01% 的概率**阈值。该项目可在 [GitHub](https://github.com/TC-Zheng/ActuosusAI) 获取，是一个带有 Web UI 的本地应用，支持从 **Huggingface** 下载模型，支持以不同量化级别的 **GGUF 格式**加载模型并生成文本。
  - **Chromix_** 建议添加 **min_p 滑块**和针对词汇选项的**颜色编码**，以增强对低 temperature 生成结果的探索。他们还建议支持 **OpenAI 兼容 API** 调用，并在用户空闲时间自动探索分支层级。
  - 用户对该项目的**交互式回溯采样器**和 **UX** 表示赞赏。有人对通过视觉提示展示具有更宽分布的 token 感兴趣，以引导用户做出更有影响力的选择。
  - 改进建议包括实现 **GPU offload** 支持，以及通过颜色编码选项和滑块等功能增强 UI，从而实现与模型输出更直观的交互。


**主题 3：本地 LLM 表现优于云端替代方案**

- **Mistral-Large-Instruct-2407 really is the ChatGPT at home, helped me where claude3.5 and chatgpt/canvas failed** ([Score: 238, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g878zy/mistrallargeinstruct2407_really_is_the_chatgpt_at/)): **Mistral-Large-Instruct-2407** 在整合来自两个仓库的代码时表现优于 **Claude 3.5** 和 **ChatGPT**：分别是 **Lucid_Autonomy**（**1500** 行）和 **Lucid_Vision**（**850** 行）。作者对 Claude 关注无关函数以及 ChatGPT 无法重写必要代码感到沮丧，而 Mistral-Large-Instruct-2047 在极少引导下便完成了任务，这在[对话日志](https://github.com/RandomInternetPreson/Lucid_Vision/tree/main/LocalLLM_Update_Convo)中得到了证实。

- **[我为 Windows 开发了一个更好版本的 Apple Intelligence 写作工具！它支持大量的本地 LLM 实现，并且是开源且免费的 :D](https://v.redd.it/0zm105dfbxvd1)** ([Score: 135, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1g80bna/i_made_a_better_version_of_the_apple_intelligence/)): 该帖子介绍了一个由作者开发的 **Apple Intelligence Writing Tools** 的 **Windows 兼容替代方案**。这款开源且免费的工具支持 **多种本地 Large Language Model (LLM) 实现**，与 Apple 的版本相比提供了更广泛的功能。创作者强调了该工具对于对 AI 辅助写作感兴趣的 Windows 用户的易用性和多功能性。
  - **Writing Tools** 是 Apple Intelligence Writing Tools 的 Windows 兼容替代方案，支持 **多种本地 LLM 实现** 并提供 **系统级功能**。它已被 [XDA](https://www.xda-developers.com/windows-pc-can-now-deliver-instant-free-writing-help-across-all-apps/) 和 [Beebom](https://beebom.com/high-schooler-app-brings-apple-inteligence-writing-tools-windows/) 报道。
  - 该工具可以通过简单的 **4 步流程** 配合 **Ollama**（一种本地 LLM 选项）运行。建议拥有 **约 8GB RAM 或 VRAM** 的系统用户选择 **Llama 3.1 8B**。
  - 用户表达了对 **Linux 支持** 和 **KoboldCPP 兼容性** 的兴趣。开发者确认，由于该工具基于 Python 和 QT，移植到 Linux 应该非常简单。


**主题 4. IBM Granite 3.0：支持完全商业用途的开源 LLM**

- **[IBM Granite 3.0 模型](https://huggingface.co/collections/ibm-granite/granite-30-models-66fdb59bbb54785c3512114f)** ([Score: 156, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1g8i69p/ibm_granite_30_models/)): **IBM** 与 **Ollama** 合作将 **Granite 3.0 模型** 引入 Ollama 平台，扩大了可用 AI 模型的范围。Granite 3.0 系列包含从 **30 亿** 到 **700 亿** 参数的各种规模模型，旨在以更高的性能和效率处理文本生成、摘要和问答等任务。
  - **Granite 3.0 模型** 目前拥有 **4096 token 上下文窗口**，并计划在 2024 年扩展到 **128K tokens**。用户对目前的限制表示失望，但对未来的改进表示关注。
  - IBM 发布完全开放的模型，与近期对 Meta 有限商业化限制的批评形成对比。Granite 模型的 **Apache 2.0 许可证**，特别是 **2B 版本**，被认为对于不受限制的使用和合成数据生成非常有价值。
  - 用户将 Granite 3.0 的性能与其他模型进行了对比，评价褒贬不一。一些人认为它与 **Mistral** 和 **Llama** 具有竞争力，而另一些人则认为它无法超越 **Qwen2.5**。**1B 和 3B MoE** (Mixture of Experts) 模型因其快速的 CPU 性能而受到关注。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与技术**

- **Google Deepmind 通过联合样本选择推进多模态学习**：一篇 [Google Deepmind 论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择（joint example selection）进行数据策展，从而进一步加速多模态学习。

- **Microsoft 的 MInference 显著提升长上下文任务推理速度**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 支持在保持准确性的同时，对长上下文任务进行高达数百万个 Token 的推理，显著提升了支持模型的运行速度。

- **利用 10 亿个 Web 策划角色扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 LLM 中的多样化视角，从 Web 数据策划的 10 亿个角色（personas）中生成数据。

**AI 模型发布与改进**

- **OpenAI 的 o1 模型表现优于 GPT-4o**：OpenAI 研究员 Noam Brown [表示，新的 o1 模型在数学和代码方面击败了 GPT-4o](https://www.reddit.com/r/singularity/comments/1g8anp0/openais_noam_brown_says_the_new_o1_model_beats/)，并在博士级问题上超越了人类专家。

- **Salesforce 的“微型巨人” xLAM-1b 模型在 Function Calling 方面超越 GPT 3.5**：Salesforce 发布了 xLAM-1b，这是一个 10 亿参数的模型，在 [Function Calling 方面达到了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **具备 Function Calling 能力的 Phi-3 Mini (六月版)**：Rubra AI 在六月发布了更新的 Phi-3 Mini 模型，[具备 Function Calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争，并优于基础版 Phi-3 Mini。

**AI 应用与影响**

- **哈佛大学科学家开发用于癌症诊断的 AI**：哈佛大学研究人员[公布了一个癌症诊断准确率达 96% 的 AI 系统](https://www.reddit.com/r/singularity/comments/1g8c3mn/96_accuracy_harvard_scientists_unveil/)，这可能彻底改变医疗诊断。

- **OpenAI 的 o1 模型生成法律文书**：OpenAI CPO Kevin Weil 声称他们的 [o1 模型现在可以编写法律文书](https://www.reddit.com/r/singularity/comments/1g7v0ud/openai_cpo_kevin_weil_says_their_o1_model_can_now/)，而这在以前需要时薪 1000 美元的助理律师完成，这可能会颠覆法律行业。

- **Stuart Russell 预测 AI 将超越人类能力**：AI 研究员 Stuart Russell [预测，到本十年末，AI 可能会在各个维度超越人类能力](https://www.reddit.com/r/singularity/comments/1g89t8u/stuart_russell_says_by_the_end_of_this_decade_ai/)，这可能会导致就业市场的重大变化。

**AI 安全与伦理担忧**

- **OpenAI 举报人在美国参议院作证**：OpenAI 举报人 William Saunders [在美国参议院作证](https://www.reddit.com/r/singularity/comments/1g7zrl1/openai_whistleblower_william_saunders_testifies/)称，“没有人知道如何确保 AGI 系统是安全且受控的”，并暗示 AGI 可能在短短 3 年内建成。

- **对 AI 发展速度和安全的担忧**：多篇帖子和评论对 AI 的快速发展和潜在安全风险表示担忧，一些人呼吁加强监管和监督。

**AI 行业动态**

- **前 OpenAI CTO Mira Murati 创办新 AI 公司**：据报道，[最近离职的 OpenAI CTO Mira Murati 正在为一家新的 AI 初创公司筹集风险投资资金](https://www.reddit.com/r/singularity/comments/1g7x6t9/mira_murati_the_openai_cto_who_announced_her/)。

- **AI 领域的竞争和融资增加**：多篇帖子和评论讨论了 AI 初创公司数量的增长以及该领域筹集的大量资金。


---

# AI Discord 回顾

> 由 o1-preview 提供的摘要之摘要

**主题 1：AI 模型进展与新发布**

- [**Janus 通过视觉解耦跨越时间**](https://x.com/deepseek_ai/status/1847191319464300652)：**DeepSeek 的 Janus** 引入了一个多模态 LLM，采用了一种新型自回归框架，将视觉编码解耦以增强理解和生成能力，表现优于 **LLaVA** 等模型。
  
  - Janus 的创新方法超越了之前的模型，在 AI 社区引起了轰动。
- [**Meta 的 Spirit LM 发声**](https://x.com/AIatMeta/status/1847383580269510670)：**Meta** 发布了 **Spirit LM**，这是一个开源的多模态语言模型，无缝集成了文本和语音，展示了在 ASR 和 TTS 方面的先进能力。

- 讨论集中在其潜在应用以及如何与现有工具自然集成。
- [**Microsoft 凭借 BitNet 取得重大突破**](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/)：Microsoft 声称他们可以在本地设备上运行 **100B 参数模型**，在没有 GPU 的情况下，速度提升高达 **6 倍**，能耗降低 **82%**。
  
  - 由于缺乏可用的 **BitNet 模型**，社区仍持怀疑态度，等待进一步验证。

**主题 2：AI Safety 与伦理担忧**

- [**Deepfakes 引发社会动荡**](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review)：社区成员对 **deepfake 技术** 表示担忧，强调了受操纵内容对受影响个体的严重公共影响。
  
  - 担忧集中在受害者被误指控以及由逼真的虚假媒体煽动的社会抵制。
- [**Nous 敲响 AI Safety 警钟**](https://x.com/NousResearch/status/1848397863547515216)：**Nous Research** 发布了一段视频和博客文章，强调了关键的 **AI Safety 问题**，并就 **AI 实践** 提供了主要发现和建议。
  
  - 这些资源激发了关于针对 AI 进步演进安全措施的讨论。
- [**当 AI 变得说教时**](https://www.reddit.com/r/notebooklm/comments/1g83etl/deep_dives_hosts_break_up_after_she_finds_out_he/)：用户注意到 AI 模型通过 **道德化视角** 解释提示词，影响了故事讲述和生成的内容。
  
  - 这引发了关于 AI 嵌入关于公平和道德的推定信念所带来影响的辩论。

**主题 3：模型训练挑战与优化**

- [**Unsloth 修复梯度 Bug，加速训练**](https://x.com/danielhanchen/status/1848415389266669883)：**Unsloth AI** 解决了关键的 **梯度累积 Bug**，改进了损失曲线计算并增强了模型训练的可靠性。
  
  - 建议用户更新库以利用这些改进来获得更好的模型性能。
- [**Liger Kernel 解决内存占用问题**](https://arxiv.org/pdf/2410.10989)：**Liger Kernel** 用户讨论了模型训练期间 **CUDA 内存错误** 的解决方案，强调了 **Triton** 和 **Liger** 操作中内存分配模式的重要性。
  
  - 社区努力集中在高效梯度累积的代码审查和解决潜在 Bug 上。
- [**BitNet 将模型缩小到比特级**](https://github.com/microsoft/BitNet)：**Microsoft** 推出了 **bitnet.cpp**，这是一个用于 **1-bit LLMs** 的推理框架，在 CPU 上实现了高达 **6.17 倍的加速** 和 **82% 的能耗降低**。
  
  - 开发者们对在没有 GPU 的情况下在 CPU 上高效运行大型模型的潜力非常感兴趣。

**主题 4：AI Agent 框架与应用**

- [**TapeAgents 回溯并重放动作**](https://www.youtube.com/live/-yf-e-9FvOc)：**TapeAgents 框架** 通过名为 Tape 的统一抽象，实现了 **可恢复** 且 **可优化** 的 Agent。
  
  - 增强了使用工具的 Agent 架构的能力，在 AI 开发圈引起了关注。
- [**WorkArena++ 测试 Web Agents**](https://arxiv.org/abs/2410.07041)：**WorkArena++** 基准测试的发布挑战了企业环境中的 Web Agents，重点关注自主任务完成。
  
  - 旨在跟踪 Agent 在复杂环境中的进展，激发了 AI 社区的兴趣。
- [**AGI 玩狼人杀，无需满月**](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：定于 **2024 年 11 月 9 日** 举行的 **AGI-Thon Werewolf Agents 锦标赛** 邀请 AI Agents 参加狼人杀游戏竞赛。
  
  - 参与者对在具有诱人奖品的竞争环境下测试他们的 Agent 感到兴奋。

**主题 5：AI 在创意内容生成中的应用**

- [**用 AI 制作播客：谈论谈话**](https://open.spotify.com/show/4Lp134MSzPu7UQDYi0mvuu)：用户分享了从 Reddit 评论和 Discord 聊天中生成引人入胜的播客的成功案例，展示了 AI 在内容创作方面的潜力。
  
  - 一位创作者自豪地上传了 **500 集** 内容，展示了惊人的效率。
- [**NotebookLM 出现语言偏差**](https://myaccount.google.com/language)：参与者报告称，尽管使用了英语提示词，**NotebookLM** 仍默认使用西班牙语，这表明需要更清晰的语言设置。
  
  - 建议调整 **Google 账号语言设置** 以缓解此问题。
- [**AI 在角色扮演中展现创意**](https://www.reddit.com/r/notebooklm/comments/1g7lf1g/deep_dive_ai_got_you_babe/)：关于使用 AI 模型进行高级色情角色扮演 (ERP) 技术的讨论集中在创建详细的角色档案和增强沉浸感。
  
  - 用户称赞了创新的提示词，并表示有兴趣将这些技术应用于非色情的创意写作。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HelpingAI2 Demo 发布**：查看 [HelpingAI2 demo](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype)，该原型展示了旨在增强用户与 AI 辅助交互的新功能。
  - 该计划旨在通过先进的 AI 交互技术促进更好的用户参与。
- **蛋白质结构可视化突破**：发布了一个关于 [蛋白质结构预测](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling) 的新项目，集成了噪声以增强可视化能力。
  - 该工具显著提升了在该领域可视化复杂蛋白质结构的能力。
- **高级 Dreambooth LoRA 脚本发布**：推出了一款新的高级 **Dreambooth LoRA 训练脚本**，具有最大灵活性和控制力的增强功能，详见[这篇文章](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora)。
  - 该脚本邀请社区反馈，以推动持续改进。
- **NLP 资源共享**：一位成员向社区推荐了 [hf.co/learn](https://hf.co/learn) 以获取优秀的 NLP 学习资源，展示了对初学者易用材料的关注。
  - 这一交流表明 NLP 领域对实用指南的需求日益增长。
- **用于 Diffusion 流水线的 NozyIO UI**：介绍了 [NozyIO 项目](https://github.com/oozzy77/nozyio)，允许用户链接 Python 函数并可视化输出，并就如何将其用于 HuggingFace 流水线进行了协作讨论。
  - 确认支持 **Yolo** 集成，从而在 NozyIO 中实现目标检测功能。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **播客生成成功**：用户报告了从 Reddit 评论和 Discord 聊天等各种来源生成引人入胜的播客，一位创作者上传了 **500 集** 以展示其效率。
  - 虽然结果各异，但一些参与者讨论了对支持更长音频输出和改进交互功能的需求。
- **语言默认设置困扰**：参与者遇到了 NotebookLM 默认使用 **西班牙语** 的问题，尽管他们的 Prompt 是 **英语**，这表明需要更清晰的语言设置。
  - 建议通过调整 **Google 账户语言设置** 来缓解这一挑战。
- **NotebookLM 的多样化用例**：用户分享了 NotebookLM 的多种应用，从学术研究到根据用户评论创建播客，展示了其多功能性。
  - 一位用户强调了从 **Discord** 和 **Reddit** 评论中有效生成播客的效果，并强调了出色的产出结果。
- **优化 Prompt Engineering 以获得更好的输出**：社区探索了有效的 NotebookLM 提示策略以实现理想的输出，包括在播客中生成特定的对话。
  - 人们正在不断努力改进 Prompt，以增强生成内容的性能和参与度。
- **AI 回复中的伦理担忧**：用户意识到 NotebookLM 可能会通过 **道德视角** 解释 Prompt，从而影响叙事和生成的内容。
  - 这引发了关于 AI 模型基于内置的公平和道德信念做出假设所带来的影响的讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **关于开源数据要求的讨论**：成员们辩论了当前 **开源 AI 项目数据要求** 的实用性，特别是对未公开数据和训练过程可复制性的担忧。
  
  - 一位参与者力推明确区分模型使用与数据要求的定义，以增强理解。
- **版权法阻碍 AI 训练**：对话强调了正在进行的 **版权法辩论** 及其对在 AI 模型训练中使用受版权保护数据的影响，特别是在欧盟内部。
  
  - 参与者指出，虽然欧盟的 TDM 例外支持技术进步，但其应用的明确性仍然不足。
- **RWKV-7 刷新训练速度记录**：据报道，RWKV-7 这种无注意力模型在速度上超过了修改后的 GPT 模型，实现了显著的训练速度提升。
  
  - 最近的优化带来了更好的验证损失（validation loss）和训练时间，表明模型效率在持续进步。
- **评估 Pythia 中的动态损失缩放**：成员们注意到，**Pythia** 模型在 FP16 运行时遇到 NaN 或 Inf 梯度时可以跳过权重更新，而这一特性在 BF16 运行时并不存在。
  
  - 讨论强调，FP16 训练可以在某些错误条件下继续，而不像 BF16 会完全停止进程。
- **将 Eval Harness 与自定义模型集成**：社区关注如何有效地将 **eval harness** 与自定义模型集成，并强调了各种 PyTorch 仓库中的局限性。
  
  - 关键建议包括使用 `TemplateLM` 作为子类，以更好地应对 API 的复杂性并增强任务处理能力。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 讲座解析**：备受期待的 Daniel Han 关于 GPU 模式的讲座现已上线，内容涉及 **LLM 系统工程** 和 **梯度累积修复** 的见解。
  
  - 讲座包括实用的问答环节，增强了旨在优化 AI 模型的开发者的理解。
- **发布梯度累积 Bug 修复**：针对影响 Unsloth 训练器的 **梯度累积** Bug 实施了关键修复，改进了损失曲线计算。
  
  - 建议用户更新其库以利用此修复，从而获得更好的模型训练可靠性。
- **处理新数据集的训练问题**：讨论强调了多样化数据集的必要性，同时解决了在多目标预测等新格式上微调模型的困难。
  
  - 参与者分享了关于合成数据生成的建议，以应对模型相关性问题。
- **Mistral 在 ReAct Agent 工具调用方面的创新**：一位成员报告了一个专注于 **ReAct Agent 工具调用** 的数据集开发情况，同时也担心 **Mistral 的 Agentic 模型** 会使早期的努力黯然失色。
  
  - 新的 **Ministrial 8b** 模型引发了关于继续使用现有数据集是否还有意义的疑问。
- **LayerSkip 提升推理效率**：关于 **LayerSkip** 的见解显示，它通过采用层丢弃（layer dropout）和早期退出损失（early exit loss）策略来提高 LLM 推理速度。
  
  - 结果表明，它在摘要和编码任务中显著提升了性能，GitHub 上已提供详细实现的访问权限。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 专注于 AI safety**：Nous Research 发布了一个视频和一篇关于 AI **safety issues** 的博客文章，强调了关于 **AI practices** 的关键发现和建议。你可以点击[这里](https://x.com/NousResearch/status/1848397863547515216)观看视频，并阅读博客文章以获取深入分析。
  
  - 这些资源是关于 AI safety 措施如何根据该领域的最新进展和挑战进行演变的更广泛讨论的一部分。
- **Deepfake 技术引发担忧**：成员们讨论了 **deepfakes** 的危险，特别是它们如何对受影响的个人造成严重的公共影响。这反映了人们对内容真实性识别的担忧，以及社会对受害者的负面反应。
  
  - 社区强调需要提高公众意识，并针对此类操纵技术采取保护措施。
- **MarketAgents 项目受到关注**：专注于多 Agent 市场模拟的 **MarketAgents** 项目引起了关注，特别是由于 **Blacklight** 的贡献。更多细节可以在 [project repository](https://github.com/marketagents-ai/MarketAgents) 中找到。
  
  - 讨论强调了该项目的协作性质及其对市场模拟的潜在影响，成员们渴望了解其进展的更新。
- **模型效率方面的进展**：对话集中在通过 **quantization aware training (QAT)** 来改进像 Llama 3.1-8B 这样的模型，同时讨论了与模型容量相关的权衡。建议通过剪枝 attention 层来减轻性能损失。
  
  - 此外，像 AdamW 这样的 **optimizers** 的发展突显了在不增加超参数调整负担的情况下提高训练效率的新方法。
- **Hermes AI 模型的可访问性**：现在可以在 [ai.unturf.com](https://ai.unturf.com) 免费访问 **Hermes AI Model**，该模型源自 [NousResearch/Hermes-3-Llama-3.1-8B](https://nousresearch.com/hermes3/) 架构。该平台鼓励开源贡献并提供安装指南。
  
  - 参与者表示有兴趣利用 Hermes 进行自定义应用，特别是在语音集成方面。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1 Preview 在代码生成方面表现出色**：用户报告称 **O1 Preview** 能生成 **Swift** 和 **C#** 等语言的复杂代码，例如创建一个具有网络功能的 'StrawberryStreamer' 系统。
  
  - 尽管最初存在一些错误，但它能从反馈中学习，在处理复杂的编程任务时变得特别有用。
- **ChatGPT 保存了过多不重要的信息**：用户对 **ChatGPT** 尽管有忽略指令但仍保存琐碎细节感到沮丧，导致需要进行内存清理。
  
  - 自定义指令可能会增强内存管理，这表明需要更好的用户控制。
- **激活 GPT-4o 功能**：据解释，自定义 GPT 会自动利用 **GPT-4o**，无法选择使用其他模型。
  
  - 用户获知了如何通过自定义 GPT 管理文件和生成输出。
- **有效 AI Prompt 的策略**：为了最大化 AI 性能，建议使用较少的常用词，并在 Prompt 开始处提供引号内的清晰指令。
  
  - 有效的示例表明，指定书写表面可以提高输出质量。
- **创建真实的 AI 交互**：为了实现更像人类的 AI 交互，使用非正式的交流方式并提供详细的角色背景故事至关重要。
  
  - 模型会模仿用户的语言，友好的措辞和期望能显著增强真实感。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 限制困惑**：用户报告在升级到 **Enterprise Pro** 后丢失了 **focus options**，导致来源和回答减少，影响了功能性。
  - 这引发了关于如何获取更全面结果的讨论，因为许多人觉得服务退步了。
- **Perplexity 的多样化用户体验**：虽然一些用户喜欢 Perplexity 在无需大量搜索的情况下进行研究和编码的功能，但其他人遇到了 **internal server errors** 和 API 访问问题。
  - 用户体验的分歧引发了对整体服务可靠性和质量的担忧。
- **关于 AI 模型性能的辩论**：对 **Claude 3.5 Sonnet** 和 **GPT-4O** 等各种 AI 模型的讨论凸显了竞争格局，用户正在评估它们在不同任务中的表现。
  - 这表明在选项不断增加的情况下，人们对了解哪种工具适合特定需求有着更广泛的兴趣。
- **YouTube 应对 AI 内容识别**：YouTube 推出了一项旨在识别 **AI-generated content** 的功能，这是迈向提高数字媒体透明度的一步。
  - 这符合用户对真实性日益增长的需求，在不断演变的内容创作领域尤为相关。
- **API 积分转移问题**：一位用户对购买 Pro 订阅后 **API credits** 未能转移表示担忧，提出了关于用户支持的关键问题。
  - 联系支持人员的建议反映了社区对高效解决运营问题的重视。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 作为 C++ 替代方案兴起**：成员们探讨了 **Mojo** 如何被开发为一种通用系统编程语言，目前在模仿 **C++** 的同时向 **Python** 的抽象级别演进。
  - 一位成员提到了 [Carbon programming language project](https://github.com/carbon-language/carbon-lang)，以获取有关面向对象编程实现的见解。
- **Mojo 与 Carbon 的灵活性对比**：讨论强调了 Mojo 在指针方面比 **Carbon programming language** 具有更大的灵活性，后者受限于 C++ 的兼容性。
  - 成员们指出了处理引用和指针时的技术差异，表明了 **Mojo** 的潜在优势。
- **Mojo 中的编译时元组长度**：用户发现 **Mojo** 支持通过 `__type_of(t).__len__()` 获取元组的编译时长度，增强了动态编码能力。
  - 这种方法允许开发人员避免运行时检查，提高了整体代码效率和可靠性。
- **关于图训练支持的咨询**：一位成员征求了关于 **Graph training support** 时间表的信息，强调了在 GPU 关注点之外更新编译后的 Max Graphs 中数值的需求。
  - 对任何澄清表示了感谢 (*Thx*)，强调了社区对更广泛功能的兴趣。
- **MAX-Graph 模型的 C-API**：成员们询问了利用 **C-API** 执行来自 **MAX-Graph API** 模型的可行性，这些模型是通过 **export_compiled_model** 导出的。
  - 这引发了对那些不愿依赖 **ONNX** 或 **Torch** 等框架的用户在当前工具中存在差距的担忧。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek Janus 发布**：DeepSeek 推出了 [Janus](https://x.com/deepseek_ai/status/1847191319464300652)，这是一个多模态 LLM，采用了一种新型的自回归框架，通过解耦视觉编码来实现更好的理解和生成，超越了以往的模型。
  
  - 与 **Llava** 等模型的对比表明，Janus 在图像生成和理解方面都具有更强的能力。
- **Meta 发布新款 Spirit LM**：Meta 推出了 [Spirit LM](https://x.com/AIatMeta/status/1847383580269510670)，这是一个开源的多模态语言模型，无缝集成了文本和语音，展示了在 ASR 和 TTS 方面的先进能力。
  
  - 讨论集中在其应用潜力和 AI 社区的早期反响上，强调了与现有工具的自然集成。
- **Microsoft Copilot Agents 的挑战**：用户反映了对 **Microsoft Copilot** 的不满，理由包括性能问题、对专业知识的误解以及在重构过程中的文本格式问题。
  
  - 营销能力与实际性能之间的差距，特别是在企业应用中，受到了显著批评。
- **新加坡 AI Engineer Nation 倡议**：在[最近的一次对话](https://x.com/swyx/status/1847732308889260072)中，Josephine Teo 部长讨论了新加坡 AI 政策的未来，重点关注 **AI 如何在政府中被采用以服务公众利益**。
  
  - 她探讨了 **Sovereign AI** 方法及其对**选举**的影响，分享了关于治理和技术集成的见解。
- **AST vs DSL：何时使用**：社区就 **AST** 与 **DSL** 的使用展开了讨论，探索它们作为编程替代交流方式的角色。
  
  - 参与者辩论了两者在代码重构任务中的最佳场景，强调了它们各自的优势。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Granite 8B 对决 Qwen 2.5 7B**：用户正积极比较 **Granite 8B** 和 **Qwen 2.5 7B** 在编程和科学任务中的表现，重点关注性能基准测试。
  
  - [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) 被推荐作为性能对比的资源。
- **Llava 的图像识别困扰**：几位用户报告称 **Llava 模型** 在识别图像方面存在困难，导致响应不准确。
  
  - 为了缓解这一问题，他们建议使用 **jpeg 或 png** 格式，并从**干净的对话（clean chat）**开始。
- **Xeon E5-2603 v4 处理器限制为 6 个线程**：在关于双路 **Xeon E5-2603 v4** 处理器 bug 的讨论中，**0.3.4 版本**仅利用了 **6 个线程**，低于 0.2.31 版本中的 8 个。
  
  - 一位成员指出这是一个*已知问题*，并确认他们的发现已被添加到现有的 bug 报告中。
- **RX 7900 XTX 表现优于 ROCm**：一位用户观察到，在推理测试中，**RX 7900 XTX** 使用 **Vulkan** 的性能比使用 **ROCm** 高出约 **10-15%**。
  
  - 另一位用户建议回退到 **ROCm 1.10**，因为最新的运行时存在复杂问题。
- **关于 M4 Ultra AI 能力的意见分歧**：针对即将推出的 MacBook 中的 **M4 Ultra 芯片**及其在 AI 任务中的有效性引发了辩论，一些人表示怀疑。
  
  - 用户指出了潜在的局限性，认为其**昂贵**且**不可升级**的设计可能会阻碍其在 AI 领域的广泛应用。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection 的支付处理器面临停机**：**Inflection 3 Pi** 和 **Inflection 3 Productivity** 模型由于支付处理问题而停机，严重影响了用户访问。
  
  - 用户正在等待关于这些模型何时恢复全部功能的进一步更新。
- **Grok 2 在价格上涨中更名**：此前被称为 **Grok 2** 的模型已正式更名为 **Grok Beta**，补全（completions）定价现设定为 **$15/M**。
  
  - 这一更名反映了其过渡性的开发状态，同时用户报告了服务可用性的波动。
- **Hermes 3 用户遭遇速率限制**：频繁的 **429 errors** 困扰着 **Hermes 3** 模型的使用者，由于使用限制似乎比以前更严格，引发了用户的不满。
  
  - 用户注意到这些限制在以前并不常见，从而引发了关于潜在模型调整的讨论。
- **OpenRouter 计费系统混乱**：用户报告称，即使账户中存在现有额度，**OpenRouter 计费系统** 仍会出现意外扣费，导致困惑。
  
  - 许多人分享了类似的经历，表明需要更好的支持机制来解决计费差异问题。
- **AI 摘要生成器在 Vercel 超时问题上挣扎**：一个基于 **Gemma 2 27B** 的 AI 文本摘要生成器在 Vercel 的 hobby 计划中，**10 秒**后会出现 **FUNCTION TIMEOUT** 错误。
  
  - 建议包括增加函数超时限制或探索 **streaming responses**（流式响应）以绕过这些限制。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **掌握持久执行（Durable Execution）概念**：成员们讨论了 **durable execution**，这是一种非常适合长时间运行工作流的抽象，并以 [Temporal background checks](https://learn.temporal.io/examples/go/background-checks/) 为例进行了说明。这种方法允许代码运行不受时间和空间的限制。
  
  - 这些见解带来了实际应用，并激发了集成类似框架以实现高效工作流管理的兴趣。
- **在 Aider 中使用 Mistral API**：提供了在 Aider 中使用 **Mistral API** 的说明，展示了如何通过命令行指定模型以及如何在 `.aider.conf.yml` 文件中进行配置。
  
  - 社区讨论强调了精确选择模型对于高效 AI 驱动编程会话的重要性。
- **CEDARScript 负责处理低级语法**：讨论集中在 **CEDARScript** 上，它将语法问题从 LLM 中卸载，使其能够专注于高级抽象，并显示出与各种编程语言的兼容性。
  
  - 对其与 Aider 集成的探索有望在未来提供更强大的代码编辑能力。
- **微软发布用于 1-bit LLM 的 bitnet.cpp**：微软发布了 [bitnet.cpp](https://github.com/microsoft/BitNet)，这是一个用于 **1-bit LLM** 的推理框架，包括优化了 CPU 性能的 BitNet b1.58 模型。
  
  - 它在 ARM CPU 上实现了 **1.37倍至 5.07倍** 的加速，在 x86 CPU 上实现了 **2.37倍至 6.17倍** 的加速，并显著降低了能耗，对于从事大规模模型开发的开发者来说，这是一个诱人的前景。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TensorRT-LLM 增强高效推理**：一位用户分享了关于 **TensorRT-LLM** 的重要资源，重点介绍了用于优化大语言模型（LLMs）性能的 [cutlass int8 gemm kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63)。
  
  - 该资源旨在提供一个 Python API，显著提升**高效推理**，这对于高性能模型执行至关重要。
- **即将举行的 Unsloth 演讲亮点**：宣布了一场以 **Unsloth** 为核心的即将到来的演讲，Unsloth 是系统工程和 Triton kernels 的重要资源，并分享了包括 [slides](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing) 在内的进一步材料链接。
  
  - 预计参与者将深入了解 *Triton 和 CUDA 技术*，增强其技术储备。
- **Apple Silicon 上的 CUDA 内存管理问题**：关于在 Apple Silicon 上结合 PyTorch 使用统一内存（unified memory）时的内存管理讨论正在进行中，特别是 tensors 默认是否在私有模式下分配。
  
  - 有人对利用 **at::from_blob()** 使用自定义缓冲区时可能出现的问题表示担忧，表明文档需要进一步明确。
- **Liger Kernel 中的梯度累积 Bug**：针对 transformers 中 **梯度累积 Bug** 修复的一项关键询问引发了关于其是否适用于 **Liger Kernel** 交叉熵（cross entropy）操作的疑问。
  
  - 这表明社区正专注于确保 Liger Kernel 功能潜在问题的清晰度。
- **与 Triton 和 Liger 相关的内存错误**：有报告称在使用 PyTorch 的 **torch compile** 时，Liger 出现了内存分配问题，特别是 **cuda out of memory errors**。
  
  - 这强调了探索与 Triton 和 Liger 操作相关的特定内存模式的迫切需求。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **寻求人类数据标注员**：一位成员正在为**天气雷达数据**寻求人类数据标注员的建议，强调了对**地理空间（geospatial）**和**视觉语言标注（vision language labeling）**的需求。
  
  - 讨论围绕多个平台展开，包括 **Scale AI**、**Surge**、**Mechanical Turk** 和 **Prolific**，并分析了它们针对不同数据类型的优缺点。
- **RLHF 书籍进展**：Nato 宣布他正在编写一本关于*人类反馈强化学习（RLHF）*的书籍，目标是在年底前发布实体版。
  
  - 他鼓励社区通过[书籍网站](https://rlhfbook.com/)参与互动，同时强调了他的写作过程没有经过广泛的检查。
- **LLM 推理辩论升温**：社区就 **LLMs**（特别是 **GPT-4o** 和 **GPT-o1**）是有效地进行推理还是仅仅复制训练模式展开了辩论。
  
  - 这场讨论是由 2024 年 5 月发布的这两个模型引发的，引发了对其真实问题解决能力的关注。
- **Interconnects 表情符号引起关注**：成员们讨论在服务器中添加 **Interconnects 表情符号**，并提出了 **AI 公司 Logo** 和 meme（梗图）的想法。
  
  - 随后进行了关于表情符号设置和 Discord 工作人员潜在支持的幽默交流，并讨论了深色模式兼容性的美学改进。
- **OpenAI 发布 GPT-4o 和 GPT-o1**：OpenAI 推出了 **GPT-4o**，承诺在音频、视觉和文本方面实现实时推理，随后推出了针对重推理基准测试的 **GPT-o1**。
  
  - 这一进展加剧了关于 AI 推理能力与从给定训练数据中学习到的行为之间关系的讨论。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **RTX 3090 表现令人失望**：一位用户报告其 **RTX 3090** 仅达到 **每秒 3.5 次迭代 (iterations per second)**，表现甚至不如 **RTX 3060**。建议的修复方案包括更新 web UI 和重新安装驱动程序。
  
  - 这种意料之外的性能下降引起了关注，引发了关于优化设置以匹配先前结果的讨论。
- **图像视角处理困难**：一位用户在尝试为建筑物创建不同视角，同时在新的草图中保持颜色完整性时遇到困难。社区建议包括利用更多的无人机拍摄镜头，并专门针对该建筑训练一个 **Lora**。
  
  - 这场关于技术的辩论凸显了现有照片数据集在实现逼真变换方面的局限性。
- **图像生成过程中的 Lora 混淆**：用户在图像生成时遇到了多个 **Loras** 未找到的错误，引发了排错讨论。成员们就如何管理 prompt 以避免此类冲突提供了见解。
  
  - 这一问题强调了需要更好的 prompt 管理策略，以最大限度地发挥 Lora 的效用。
- **访问 Stability.ai API 遇到麻烦**：关于 **Stability.ai API 参考页面** 宕机的担忧出现，用户建议联系客服解决。社区澄清此问题超出了他们的控制范围。
  
  - 这引发了关于在等待官方支持期间，为需要 API 访问的用户提供临时变通方案的讨论。
- **寻求 AI 图像编辑帮助**：用户表示需要协助将 AI 工具集成到商业项目的图像编辑中。社区内提出了协作帮助的提议，展示了支持性的氛围。
  
  - 这种对协作的渴望表明，人们对优化涉及 AI 技术的流程越来越感兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **为期 3 天的黑客松产出 45 个项目**：最近的 **3 天黑客松** 吸引了超过 **500 名参与者**，最终展示了 **45 个项目**。查看 [宣布获胜者的博客文章](https://t.co/v7F8b0qedF) 了解更多详情。
  
  - 获胜者撰写的精彩客座博客文章将对他们的项目提供更深入的见解。
- **LlamaParse Premium 获得好评**：用户对 **LlamaParse Premium** 感到兴奋，报告其解析能力有显著提升。一篇深刻的 [LinkedIn 帖子](https://t.co/NeAvIlfIP3) 评测了其相对于早期版本的优势。
  
  - 更多背景信息，可以点击[此处](https://t.co/pDPHxcYQeb)查看 **LlamaParse** 的最初介绍。
- **在 LlamaIndex 中集成 Ollama**：尝试使用 `npx create-llama` 配置 **Ollama** 时，即使设置正确，也会弹出 OpenAPI key 提示。建议通过编辑后端源代码来解决 **Ollama** LLM 的加载问题。
  
  - 这一见解可能会帮助其他遇到类似集成麻烦的人。
- **评估混合检索准确性**：社区讨论了评估结合 `BM25Retriever` 和 `VectorIndexRetriever` 的混合检索器的方法，强调了 ground truth 数据集的必要性。利用 LLM 评估相关性被认为是一种很有前景的方法。
  
  - 追踪问题与文档的映射（question-document mappings）也成为一种可行的评估方法。
- **寻找多语言 Embedding 解决方案**：一位成员正在探索一个处理多语言 PDF 的 **RAG 系统**，但目前的 embedding 模型效果不佳。他们收到了关于 **aBSE** 模型的建议，认为这是一个潜在有效的解决方案。
  
  - 该模型专注于语言无关（language-agnostic）的实现，这可能会增强多语言性能。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Multihead Attention 的相关性**：在 Tinygrad 社区中，一名成员质疑了关于 **Multihead Attention 标准化** 讨论的持续相关性，表明重点已转向优化工作。
  
  - 这突显了社区对在框架内改进 attention 机制的持续兴趣。
- **Tinygrad 通过支持 GGUF 增强竞争力**：George Hotz 宣布增加 **GGUF 加载支持**，以增强 Tinygrad 在高效运行 **本地 LLM** 方面相对于 **Ollama** 等对手的竞争力。
  
  - 他鼓励开发者做出贡献，旨在提升 Tinygrad 的性能和功能。
- **本地 LLM 工具见解**：用户讨论了对 **Llama.cpp** 和 **ExLlamaV2** 进行本地模型执行的偏好，其中 ExLlamaV2 相比 **TensorRT-LLM** 提供了更简单的设置选项。
  
  - 共识表明，为了提高模型部署效率，用户正转向使用这些工具。
- **强调 WebGPU 支持**：George Hotz 强调了 **WebGPU 支持** 的重要性，并详细介绍了社区在增强 Tinygrad 与该技术兼容性方面的努力。
  
  - 记录了在实现 **threefry** 算法方面的进展，表明开发阻碍正在减少。
- **澄清 FrozenBatchNorm2d 功能**：一位用户寻求关于 **FrozenBatchNorm2d** 在网络架构中作用的澄清，对其必要性和函数机制表示困惑。
  
  - 这一讨论揭示了用户在集成特定组件时面临的复杂性。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **神秘模型引发好奇**：一名成员提到一个具有 **8k** 上下文的 **神秘模型**，引发了社区的兴奋。
  
  - 社区成员渴望与 [mystery bot](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771) 互动以获取更多更新。
- **明天参加开发者办公时间！**：Cohere 计划于明天 **东部时间下午 1:00** 举行 **Developer Office Hours**，届时将进行新版本的现场演示。
  
  - 参与者可以通过 [Cohere Developer Event](https://discord.com/events/954421988141711382/1285304800400904344/1297967638118400000) 加入讨论。
- **OpenRouter 提供 API 灵活性**：成员们讨论了 **OpenRouter**，强调了其在面临停机时无缝切换 API 的能力。
  
  - *说实话，并非所有 API 提供商都是稳定的*，这强调了对这一强大功能的需求。
- **JavaScript 在实现中表现出色**：一位成员展示了一个使用 **JavaScript** 的项目，引发了对其在 AI 应用中有效性的兴奋。
  
  - 这种热情反映了利用 **JavaScript** 实现 AI 功能的明显趋势。
- **直接 API 请求简化**：一位成员确认，仅使用 **API key**，开发者就可以直接向 AI 提供商发送请求，而无需依赖代理。
  
  - 这种方法减少了依赖并简化了开发者的集成工作。

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Liger Kernel 安装顺利**：用户发现为了实现 **VRAM 节省**，安装 **Liger Kernel** 非常简单，只需执行 `pip install liger-kernel` 并调整提供的配置以获得最佳设置。
  
  - 该内核利用现有的 **Flash Attention** 增强了全量微调（full finetuning）能力，是提升性能的明智之举。
- **Axolotl 层冻结 Bug 引发关注**：社区成员报告了 **Axolotl** 中的一个 **bug**，该 bug 会阻止层的冻结/解冻，而这是一个此前运行良好的核心功能。
  
  - 调查正在进行中，成员们被要求确认 `src/axolotl/integrations/spectrum/model_snr_results` 目录中的更改，以获取进一步的见解。
- **Spectrum 确认 SNR 结果可靠**：关于 Qwen 模型的 **SNR 结果** 正确计算方式展开了对话，并确认一切都已对齐。
  
  - 成员们指出，**Spectrum** 集成需要 **预计算的 SNR JSON 文件** 才能正常运行。
- **Qwen2 DoRA 支持请求引起关注**：一位成员正在寻求 **Qwen2** 对 **DoRA/QDoRA** 支持的进展，并提到相关讨论中的活动极少。
  
  - 他们指出 [**Answer.AI 的 QDoRA 仓库**](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model) 是潜在实现的基础资源。
- **针对特定领域数据微调 LLM**：一位成员分享了他们在 **训练和微调 LLM** 以适应 **数学**、**法律** 和 **金融** 等 **特定领域数据** 的历程。
  
  - 他们主张，为了获得更好的训练效果，从 **llama-70b-instruct** 开始比使用非 instruct 模型更具优势。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Meta 的 FAIR 团队推进高级机器智能**：Meta 的 FAIR 团队分享了他们实现 **高级机器智能 (AMI)** 以提高生产力和创新能力的目标，正如 Mark Zuckerberg 的 [公开信](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) 中所强调的那样。他们的承诺反映了十多年来与 AI 社区在 **开放科学** 方面的合作。
  
  - 这一研究工作恰逢关于 **Lingua** 等工具是否可与 **Torchtune** 相媲美的讨论。
- **Attention Mask 构建与 Flex Attention**：成员们讨论了注意力机制中 **mask 构建** 的复杂性，特别是根据注意力类型需要不同的块 mask。有人建议在 forward pass 期间实例化 mask，以简化 **collate** 过程。
  
  - 这强调了在处理 **packed datasets** 和自定义 collate 需求时，保持简洁实现的重要性。
- **PyTorch 中的性能警告**：用户在某些数据类型上遇到了与 **cuDNN SDPA** 相关的警告，这引发了对底层性能和潜在解决方案的担忧。使用不同内核进行测试可能会澄清性能影响，这与 [PyTorch GitHub](https://github.com/pytorch/pytorch) 上报告的问题有关。
  
  - 参与者正在考虑在 **PyTorch core** 上提交 issue，以解决持续存在的警告及其影响。
- **v0.4.0 代码冻结倒计时开始！**：距离 **10 月 29 日** 的 **v0.4.0 代码冻结** 仅剩 **8 天**，开发人员正准备完成待处理任务。准备工作至关重要，因为 [*v0.4.0 追踪器*](https://github.com/pytorch/torchtune/issues/1747) 预计发布日期为 **11 月 5 日**。
  
  - 贡献者们正在积极制定策略，以确保该版本包含令人兴奋的更新。
- **v0.4.0 计划推出的新功能**：讨论了 **v0.4.0** 中即将推出的功能，引用了 issue **#1645**、**#1847** 和 **#1835**。贡献者们正在努力工作，以确保新功能提升用户体验。
  
  - 该版本的准备工作反映了开发团队内部强大的协作努力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic All-in-One 直播**：一位成员在 [pydantic-all-in-one](https://discord.com/channels/1161519468141355160/1161519469777133580) 发起了直播，详细介绍了他们开发 Python 包和框架的过程。
  
  - 他们计划在直播后构建 **llmodel**，以满足社区需求。
- **DSPy GPTs 教程讨论**：成员们探讨了制作关于使用各种 **DSPy GPTs** 的教程视频，这对新老用户都有裨益。
  
  - 社区支持力度很大，创作者已同意考虑编写一份全面指南的提议。
- **AI Agents 生产环境应用活动公告**：一场虚拟活动定于 **11 月 13 日**举行，邀请了 Tomas Wolf 和 Nathan Benaich 等知名演讲者，共同讨论在生产环境中部署 AI agents。
  
  - 该活动由 **Prosus AI and MLOps** 组织，承诺将解决内存管理方面的实际应用和挑战。
- **使用 Ollama 的 LightRAG 分步教程**：一位 YouTuber 分享了使用 **Ollama** 设置和运行 **LightRAG** 的详细[教程](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s)。
  
  - 该教程强调了知识图谱与基于 embedding 检索的结合，增强了系统功能。
- **关于 AcgNDCG 和文档检索的澄清**：有人提出疑问，文档是从有限的 **10 个左右相关性判断 (Relevance Judgements)** 中检索，还是从更广泛的池中检索，相关论文链接见[此处](https://arxiv.org/pdf/2406.11706)。
  
  - *它是从特定列表还是整个池中检索？* 仍是一个待解决的开放性问题。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **今日 PST 下午 3 点 LLM Agents 讲座**：**LLM Agents** 系列的**第 7 讲**将于今天 **PST 时间下午 3:00** 举行，可以在[此处](https://www.youtube.com/live/-yf-e-9FvOc)观看直播。客座演讲者 **Nicolas Chapados** 和 **Alexandre Drouin** 将在会议期间讨论**企业工作流中的 AI Agents**。
  
  - 成员们期待了解关于 **orchestration of agents** 的见解以及 **Agentic System** 的进一步进展。
- **TapeAgents 框架介绍**：讲座将介绍 **TapeAgents 框架**，该框架通过名为 Tape 的统一抽象实现**可恢复 (resumable)** 和**可优化 (optimizable)** 的 agents。这一举措可能会显著增强使用工具的 agent 架构能力。
  
  - 参与者对学习该框架如何推进其 AI agent 开发项目感到兴奋。
- **针对 Web Agents 的 WorkArena++ 基准测试**：**WorkArena++** 是一个新推出的基准测试，用于评估企业环境中的 web agents，重点关注自主任务完成情况。它为该领域提出了新挑战，并追踪 web agents 在复杂环境中的进展。
  
  - 参与者对该基准测试如何为未来基于 agent 的模型开发提供参考表现出浓厚兴趣。
- **结业证书详情**：学生在完成所有课程要求（包括测验和书面文章作业，截止日期为 **12 月 12 日**）后将获得证书。课程工作人员保证可以获取**录像和讲义**以供补课。
  
  - 作业将涉及总结讲座内容或黑客松经历，引发了围绕项目工作和概念理解的讨论。
- **使用实用工具本地运行 LLMs**：参与者获得了本地运行 LLMs 的选项，**Ollama** 和 **LM Studio 0.3.0** 被推荐为实用工具。用户必须注意，较大的模型通常需要超过 **8GB 的 RAM**。
  
  - 讨论强调了在处理本地 LLM 设置时高效资源管理的重要性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LibreFLUX 发布并带来新功能**：[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) 的 Apache 2.0 版本 **LibreFLUX** 正式发布，引入了完整的 T5 上下文长度、增强的 attention masking 以及恢复了 classifier-free guidance。
  
  - *社区反应积极*，认可了其对**开源原则（open-source tenets）**的扩展，并对新模型展现出的 21 世纪初美学感到兴奋。
- **训练 Open-MUSE 面临的挑战**：用户报告在 Hugging Face 上寻找 **openMUSE/maskgit-vqgan-imagenet-f16-256** 等模型时遇到困难，并在其训练配置文件中遇到了 missing key 错误。
  
  - 为了寻求社区帮助，他们分享了[配置 YAML 文件](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml)。
- **Microsoft 的 LLM 性能飞跃**：Microsoft 声称现在可以在本地设备上运行 **100B 参数模型**，在没有 GPU 的情况下实现高达 **6 倍的性能提升**和 **82% 的能耗降低**，正如一篇 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/)所述。
  
  - 这一断言在一条推文中得到了进一步阐述，引发了关于此类性能水平可行性的[讨论](https://x.com/jenzhuscott/status/1847514413060046855)。
- **尚无 BitNet 模型可用**：尽管对 Microsoft 的声明感到兴奋，但用户指出目前尚不存在利用 **BitNet** 的 **100B 模型**，这引发了对实际性能能力的怀疑。
  
  - 社区保持谨慎，并在接受这些效率声明之前寻求进一步的验证。
- **MUSE 项目开启复现工作**：讨论集中在用于 text-to-image generation 的 **MUSE** 模型的开源复现上，并提供了 [GitHub 仓库](https://github.com/huggingface/muse)和 [W&B Project](https://wandb.ai/psuraj/muse?workspace=user-) 等资源。
  
  - 关键活动包括在 **imagenet** 等数据集上训练各种模型，并在 **CC12M** 上进行实验，以增强过程的透明度。

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Aider 增强 AI 生成的代码**：**Aider** 正在逐步集成 AI 生成的代码，这表明其解释器概念正趋向于动态的 nightly builds。
  
  - 这引发了人们对 **Open Interpreter** 是否会有类似实现的关注。
- **Open Interpreter 的自定义工具问题**：用户询问是否有类似于 **/functions** 文件夹的等效功能，以便在 **Open Interpreter** 中轻松访问自定义函数。
  
  - 目前的选择似乎有限，建议通过修改仓库来添加自定义工具。
- **Mac 设置成功但出现问题**：一位用户报告在 Mac 上成功设置了 **OpenInterpreter**，[localhost:10100](http://localhost:10100) 运行正常。
  
  - 然而，他们遇到了交互问题，包括网页浏览器访问被拒绝以及 **LiveKit Meet 链接**的问题。
- **语音助手提升功能**：[AIwithBenefits](https://x.com/AIwithBenefits/status/1848161437828415578) 强调了为 **phidatahq** Agent 添加 **HumeAI 语音助手**，旨在通过执行 AppleScript 来提高可用性。
  
  - 翻新后的 **phidatahq UI** 受到好评，增强了与原生应用的整体交互。

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangGraph 代码助手教程发布**：**LangGraph Code Assistant** 教程指导用户如何通过 [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) 和 RAG 方法构建针对编程挑战的迭代式回答。
  
  - *摄取用户指定的文档并调用工具以生成结构化输出*，同时进行单元测试以验证返回的解决方案。
- **正在讨论基于角色的 RAG 模型**：一场关于实现针对用户角色定制的 **RAG 模型** 的讨论正在展开，特别是如何为 CEO 优化访问权限，同时限制实习生仅能访问相关文档。
  
  - 这种方法引发了关于 **RAG 框架** 内有效管理和访问限制的重要问题。
- **Techstars Startup Weekend SF 来了**：**Techstars Startup Weekend SF** 邀请与会者在 TechCrunch Disrupt 之后参加在 [AWS GenAI Loft](https://aws.amazon.com/startups/lp/aws-gen-ai-loft-san-francisco?lang=en-US) 举办的独家社交活动。
  
  - 行业专家将分享见解，促进技术社区中创始人、投资者和创新者之间的联系。
- **OpenAI Swarm 与 LangChain LangGraph 的深度对比**：一篇文章详细对比了 **OpenAI Swarm** 和 **LangChain LangGraph**，指出了它们的功能以及构建复杂 AI 工作流的适用场景。
  
  - 该指南旨在帮助开发者选择最适合项目的工具，点击[此处](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474)阅读。
- **Multi-Agent 工作流的兴起**：在 AI 中开发 **Multi-Agent 工作流** 的重要性不断增长，这对于管理复杂交互和增强能力至关重要。
  
  - *此类框架允许开发者有效地简化流程，* 从而提高整体 AI 性能。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AGI-Thon 锦标赛开幕**：即将举行的 **AGI-Thon 狼人杀 Agent 锦标赛** 定于 **2024 年 11 月 9 日** 举行，详情请见 [AGI House 活动页面](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)。
  
  - 本次活动将为 AI Agent 带来激动人心的竞赛，吸引来自不同背景的参与者展示他们的技能。
- **即将到来的锦标赛引发关注**：**AGI-Thon** 的宣布引发了渴望加入竞争的 AI 爱好者的讨论。
  
  - 许多参与者对在竞争环境中测试其 Agent 的机会表示兴奋。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 调查 AI 访问问题**：Mozilla 委托发布了两份关注 **AI 访问挑战** 和竞争的报告，分别是《[外部研究人员对封闭基础模型的访问](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf)》和《[防止科技巨头垄断 AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf)》。这些由 **AWO** 和 **Open Markets Institute** 提供的文档剖析了 AI 内部的控制动态。
  
  - 报告强调了 **外部研究人员** 访问封闭模型的必要性，以促进更广泛的创新，并强调了为实现 AI 开发中公平生态平衡所需的关键改革。
- **探讨 AI 开发中的控制权**：研究结果分析了 **谁在控制** AI 的发展，并倡导通过改革确保公平的格局。确保公平的竞争环境是维持快速变化的 AI 领域中 **创新** 的关键。
  
  - 对外部研究人员访问权限的强调旨在重塑当前的 AI 治理现状，并允许竞争多样性的变化。
- **Mozilla AI 研究的博客回顾**：一篇详细的[博客文章](https://discord.com/channels/1089876418936180786/1298015953463808102)深入介绍了 Mozilla 委托研究的成果。它在 **当前 AI 治理** 实践的背景下探讨了这些发现的影响。
  
  - 该资源作为报告的重要总结，突出了研究结果对 AI 生态系统稳定性的影响。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **关于 Q-Galora 的咨询**：一位成员询问：“有人试过 **q-galora** 吗？”，表达了对其在 AI 模型中的功能和应用的关注。
  
  - 随后没有收到任何回复，社区对关于 **q-galora** 的潜在见解或经验仍处于期待中。
- **期待关于 Q-Galora 的见解**：随着一位成员通过一个简单的问题询问 **q-galora** 的使用情况，社区正期待着经验分享。
  
  - 成员们渴望得到能够澄清其在 AI 相关项目中能力的回复。

---

**Alignment Lab AI Discord** 没有新消息。如果这个服务器沉寂时间过长，请告知我们，我们会将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果这个服务器沉寂时间过长，请告知我们，我们会将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果这个服务器沉寂时间过长，请告知我们，我们会将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果这个服务器沉寂时间过长，请告知我们，我们会将其移除。

---

# PART 2: 按频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1297638418465034291) (1 条消息):

> - `HelpingAI2 Demo`
> - `Protein Structure Prediction`
> - `AI in Nuclear Research`
> - `WorldMedQA-V Release`
> - `Books Mixer AI`

- **HelpingAI2 原型 Demo 已上线**：快来看看由社区成员展示的新原型 [HelpingAI2 demo](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype)！
  
  - 该倡议旨在增强用户与 AI 助手的交互。
- **蛋白质结构可视化取得进展**：发布了一个关于 [蛋白质结构预测](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling) 的新项目，集成了噪声和 MD frames。
  
  - 该工具为可视化复杂的蛋白质结构提供了增强的功能。
- **AI 转向核能研究**：一篇深刻的 [评论](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review) 讨论了 AI 在核能领域的影响。
  
  - 这一探索阐明了核研究中的创新应用和安全考量。
- **WorldMedQA-V 用于医疗保健基准测试**：[WorldMedQA-V](https://huggingface.co/datasets/WorldMedQA/V) 的发布提供了一个多语言、多模态的数据集，用于衡量医疗保健领域的 Vision-language models。
  
  - 该数据集旨在加强医疗领域 AI 工具的开发。
- **利用 Books Mixer AI 进行创意叙事**：[books-mixer-ai](https://huggingface.co/spaces/as-cle-bert/books-mixer-ai) 工具通过融合不同的书籍叙事来实现故事创作。
  
  - 该项目展示了一种通过 AI 驱动的创造力与文学互动的新方式。

**提到的链接**：

- [无标题](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7)：未找到描述
- [来自 Shan Chen (@shan23chen) 的推文](https://x.com/shan23chen/status/1846923442253152641)：🚀 AI4Health 的激动人心消息！🌐 我们很高兴发布 WorldMedQA-V，这是一个多语言、多模态的医学考试数据集，旨在衡量医疗保健领域的 Vision-language models！🩺💻 👉 ...

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1296910744725749832) (839 条消息 🔥🔥🔥):

> - `Hugging Face Issues`
> - `AI Model Capabilities`
> - `GPU Usage`
> - `Kaggle vs Colab`
> - `Synthetic Data Generation`

- **Hugging Face 出现错误**：用户在从 Hugging Face 下载数据集时遇到错误，特别是提示连接问题的 “ReadTimeoutError”。
  
  - 更改 DNS 设置帮助部分用户恢复了访问，但其他尝试使用该平台的用户仍面临问题。
- **AI 模型以 JSON 格式响应**：有报告称 Hugging Chat 版本的 Nemotron 仅以 JSON 格式提供响应，引起了困惑。
  
  - 用户正在通过重启对话和调整 Prompt 来尝试引导出传统的对话式响应，以解决这一异常。
- **GPU 系统的选择**：讨论围绕使用 Colab 或 Kaggle 获取 GPU 资源的偏好展开，Kaggle 因其更大的配额通常更受青睐。
  
  - 参与者指出，选择取决于具体需求和工作负载，因为不同的 LLM 可能需要不同级别的资源。
- **区块链讨论**：在社会影响的背景下提到了区块链技术，并对其必要性以及使用背后的动机进行了辩论。

- 用户对 blockchain 表达了复杂的情绪，认为它是一个“在寻找问题的解决方案”，同时也指出了其争议性。
- **使用 AI 模型合成数据**：关于为印度语言的情感分析生成合成数据的建议指向了一些有用的框架和工具。
  
  - 讨论内容包括探索 Argilla 和 Hugging Face 等模型在情感预测和数据增强等任务中的能力。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/joinwarp)：未找到描述
- [Distilabel 文档](https://distilabel.argilla.io/latest/)：Distilabel 是一个 AI Feedback (AIF) 框架，用于为 LLM 以及使用 LLM 构建数据集。
- [Wonder3D - flamehaze1115 的 Hugging Face Space](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo)：未找到描述
- [LLM 排行榜 - 比较 GPT-4o, Llama 3, Mistral, Gemini 及其他模型 | Artificial Analysis](https://artificialanalysis.ai/leaderboards/models)：跨关键指标（包括质量、价格、性能和速度（输出速度 - 每秒 tokens 数和延迟 - TTFT）、上下文等）对 30 多个 AI 模型 (LLMs) 的性能进行比较和排名。
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102))：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109))：未找到描述
- [starsnatched/ThinkerGemma-2 · Hugging Face](https://huggingface.co/starsnatched/ThinkerGemma-2)：未找到描述
- [Chat-with-GPT4o-mini - yuntian-deng 的 Hugging Face Space](https://huggingface.co/spaces/yuntian-deng/ChatGPT)：未找到描述
- [Ralph Ralph Wiggum GIF - Ralph Ralph wiggum Simpsons - 发现并分享 GIF](https://tenor.com/view/ralph-ralph-wiggum-simpsons-ralph-i%27m-learnding-i%27m-learning-gif-17493450018197884567)：点击查看 GIF
- [Wtf Wth GIF - Wtf WTH TF2 - 发现并分享 GIF](https://tenor.com/view/wtf-wth-tf2-team-fortress-2-shock-gif-5852517115769452555)：点击查看 GIF
- [未找到标题](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00004-of-00104.parquet)：未找到描述
- [未找到标题](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00007-of-00104.parquet)：未找到描述
- [Drunk Meme GIF - Drunk Meme Gif - 发现并分享 GIF](https://tenor.com/view/drunk-meme-gif-gif-25068675)：点击查看 GIF
- [Rock Everythingeverywhereallatonce GIF - Rock Everythingeverywhereallatonce - 发现并分享 GIF](https://tenor.com/view/rock-everythingeverywhereallatonce-gif-25516405)：点击查看 GIF
- [Dog Snoop GIF - Dog Snoop Dogg - 发现并分享 GIF](https://tenor.com/view/dog-snoop-dogg-rabjouj-gif-21804700)：点击查看 GIF
- [ORPO Trainer](https://huggingface.co/docs/trl/en/orpo_trainer)：未找到描述
- [Completely Different Monte Python GIF - Completely Different Monte Python Explode - 发现并分享 GIF](https://tenor.com/view/completely-different-monte-python-explode-gif-14382029)：点击查看 GIF
- [Nothing To See Here Explosion GIF - Nothing To See Here Explosion Explode - 发现并分享 GIF](https://tenor.com/view/nothing-to-see-here-explosion-explode-bomb-fire-gif-4923610)：点击查看 GIF
- [CNES - 法国国家空间研究中心](https://cnes.fr/)：未找到描述
- [Llama 3.1 405B (base) - API, 提供商, 统计数据](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:free>)：Meta 最新的模型系列 (Llama 3.1) 发布了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)。
- [未找到标题](https://sambanova.ai>)：未找到描述
- [ORPO Trainer](https://huggingface.co/docs/trl/en/orpo_trainer#trl.ORPOTrainer.tokenize_row>)：未找到描述
- [VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-64-woft at main](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-64-woft/tree/main)：未找到描述
- [Hugging Face 状态](https://status.huggingface.co/) ：未找到描述
- [unclemusclez/unsloth-smollm](https://ollama.com/unclemusclez/unsloth-smollm)：结合 Unsloth 的 SmolLM
- [快速创建聊天机器人](https://www.gradio.app/guides/creating-a-chatbot-fast)：Gradio 分步教程

- [accelerate/src/accelerate/commands/launch.py at main · huggingface/accelerate](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/launch.py#L756): 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持.....
- [Dedicated Server Hosting](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) : 未找到描述
- [Content Enhanced BERT-based Text-to-SQL Generation](https://arxiv.org/abs/1910.07179): 我们提出了一种简单的方法，利用表格内容来增强基于 BERT 的模型，以解决 Text-to-SQL 问题。基于某些表格内容与问题中的某些词匹配的观察...
- [GitHub - guotong1988/NL2SQL-RULE: Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179](https://github.com/guotong1988/NL2SQL-RULE): 内容增强的基于 BERT 的 Text-to-SQL 生成 https://arxiv.org/abs/1910.07179 - guotong1988/NL2SQL-RULE
- [Data Agnostic RoBERTa-based Natural Language to SQL Query Generation](https://arxiv.org/abs/2010.05243): 关系型数据库是现代世界中存储海量数据最广泛使用的架构之一。然而，这些数据库与普通用户之间存在障碍。用户...
- [GitHub - DebadityaPal/RoBERTa-NL2SQL: A Data Blind Approach to the popular Semantic Parsing task NL2SQL](https://github.com/DebadityaPal/RoBERTa-NL2SQL): 一种针对流行语义解析任务 NL2SQL 的数据盲处理方法 - DebadityaPal/RoBERTa-NL2SQL

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1297082422080442390) (27 条消息🔥):

> - `修改模型`
> - `学习 Python`
> - `Nvidia L4 GPU 散热解决方案`
> - `深度强化学习 (Deep Reinforcement Learning)`
> - `多头注意力 (Multihead Attention)`

- **模型修改探索**: 一位用户询问了修改 GGUF 模型并更改其规则的可能性，寻求关于模型调整的指导。
  
  - 表达了对学习更多模型修改知识的兴趣，希望能获得有用的资源。
- **Python 基础与 API 见解**: 一位成员分享了他们的 Python 学习之旅，重点是 *list operations*（列表操作），并计划进一步学习 API。
  
  - 另一位参与者建议不要在基础操作上停留太久，认为 API 操作更为重要。
- **Nvidia L4 GPU 的静音散热解决方案**: 一位成员分享了为 **Nvidia L4 24 GB GPU** 寻找 *静音散热解决方案* 的见解，详细说明了温度和风扇性能。
  
  - 他们强调成功找到了既能保持最大散热效率又能实现静音运行的方案。
- **深度强化学习课程启动**: 一位成员宣布开始他们的 *Deep RL* 课程之旅，跟随 DeepMind 讲座以及 Sutton & Barto 的书籍进行学习。
  
  - 他们对学习新概念并与社区中的其他人分享知识感到兴奋。
- **理解多头注意力**: 另一位成员分享了他们对掌握 *Multihead Attention* 背后机制以及 *attn_mask* 使用的关注。
  
  - 这反映了对复杂神经网络组件的深入研究。

**提到的链接**:

- [mods crush his skull](https://m.youtube.com/watch?v=ebnYbhU9ukA&pp=ygUebW9kcyBjcnVzaCBoaXMgc2t1bGwgdGhhbmsgeW91): crush his skull. 我的主账号 (订阅): https://www.youtube.com/@steakofsaint
- [Silent Cooling Solution for the Nvidia L4 24 GB GPU](https://vandu.tech/silent-cooling-solution-for-the-nvidia-l4-24-gb-gpu/): 我将这篇文章写得很简短，主要是照片。我测试了不同游戏下的散热性能。该 GPU 的最大功耗为 72W，但在测试期间超过了 75W。也有可能……

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1296949298834964541) (10 条消息🔥):

> - `LightRAG`
> - `CGPO`
> - `Min-p Sampling`
> - `医疗 AI 研究亮点`
> - `视觉问答模型 (Visual Question Answering Models)`

- **LightRAG 简化了检索增强生成 (Retrieval-Augmented Generation)**：[LightRAG GitHub 仓库](https://github.com/HKUDS/LightRAG) 描述了一种名为 **LightRAG: Simple and Fast Retrieval-Augmented Generation** 的新方法，专注于优化生成任务的检索。
  
  - 该方法旨在提高检索增强生成架构的效率。
- **CGPO 增强了模型针对奖励欺骗 (reward hacking) 的对齐能力**：详细介绍 **CGPO** 的论文通过引入两种新型评判器来改进现有的 PPO，这些评判器有助于在模型训练期间检测奖励欺骗。
  
  - 这一调整有助于平衡 **alignment** 与 **多目标优化 (multi-objective optimization)**，从而提高训练过程的整体有效性。
- **Min-p Sampling 提升生成质量**：引入 **min-p sampling** 方法是为了解决 **top-p sampling** 的问题，根据模型的置信度动态调整采样阈值。
  
  - 大量实验表明，该技术不仅提高了质量，还改善了输出的多样性，尤其是在高温度 (higher temperatures) 设置下。
- **顶级医疗 AI 突破播客**：在最新的 **医疗 AI 播客** 中，讨论了 **OLAPH** 和 **MedCare** 等研究论文和模型的关键进展，重点介绍了 **多模态医疗 RAG 系统 (Multimodal Medical RAG systems)** 的进步。
  
  - 听众可以通过 [这个 YouTube 视频](https://www.youtube.com/watch?v=LROOjWXUgvg) 探索关于生成式 Transformer 和聊天机器人的话题。
- **发现视觉问答 (Visual Question Answering) 模型论文**：一位成员分享了一篇关于 **视觉问答模型** 的 [值得关注的论文链接](https://arxiv.org/pdf/2406.05967)，鼓励大家查看以获取见解。
  
  - 这篇论文在该领域脱颖而出，推荐给对 AI 视觉理解进展感兴趣的人。

**提到的链接**：

- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082)：大语言模型 (LLMs) 通过在每个解码步骤中从词汇表的概率分布中采样下一个 token 来生成文本。然而，流行的采样方法如 top-p (nucleus s...
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/)：未找到描述
- [GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG)："LightRAG: Simple and Fast Retrieval-Augmented Generation" - HKUDS/LightRAG
- [Top Medical AI Breakthroughs of the Week:Multilingual models, Multi agent systems..(Oct 12-19, 2024)](https://www.youtube.com/watch?v=LROOjWXUgvg)：欢迎收听本周的 Open Life Science AI 播客，我们将在这里探索医疗 AI 研究的前沿！在本期节目中，我们将解析最具影响力的...
- [@aaditya on Hugging Face: "Last Week in Medical AI: Top LLM Research Papers/Models 🔥 🏅 (October 12 -…"](https://huggingface.co/posts/aaditya/126778565806623)：未找到描述
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1847686504837202263)：上周医疗 AI 动态：顶级研究论文/模型 🏅 (2024年10月12日 - 10月19日) Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296933035584913439) (33 条消息🔥):

> - `文本分类概览`
> - `AI 能耗与核能`
> - `OmniBench 基准测试介绍`
> - `情感 AI 交互`
> - `Recursal 发布的数据集`

- **文本分类详解**：一位成员分享了一篇解释**文本分类**的文章，并在 [Medium](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7) 上邀请大家对其见解和方法提供反馈。其他成员反应积极，建议进行跨平台发布以获得更高曝光度。
- **核能满足 AI 日益增长的能源需求**：一位成员讨论了一篇关于 AI **能源需求**增长的文章，以及科技巨头如何倾向于使用**核反应堆**来满足这些需求，并在其 Hugging Face 博客文章（[链接](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review)）中详细阐述了其对环境的影响。成员们还就核废料管理和替代能源实践进行了交流。
- **为 OLM 推出 OmniBench**：一位成员宣布推出 **OmniBench**，这是一个用于评估**全模态语言模型 (OLMs)** 的新基准测试，该模型能够同时处理多种输入模态。该消息通过 [Twitter](https://x.com/yizhilll/status/1838942877142962502) 分享。社区内提出了通过演示和讨论来提高该基准测试知名度的建议。
- **HelpingAI 2.5 发布**：**HelpingAI 2.5** 项目正式推出，专注于创建具有情感直觉、能够进行自然对话的 AI，演示版可通过 [Hugging Face](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype) 获取。该方法旨在提升各种应用中的用户交互体验。
- **Recursal 的数据集贡献**：一位成员分享了包括 **SuperWiki** 和**新加坡国家语音语料库 (Singapore's National Speech Corpus)** 重处理版本在内的多种数据集，并强调这些数据集已在 Hugging Face 上发布供社区使用。他们表达了对未来更新和开发的兴趣，并重点介绍了他们的 GitHub 项目。

**提到的链接**：

- [Conformity Protein Dynamics - 由 MISATO-dataset 提供的 Hugging Face Space](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling)：未找到描述
- [AI is turning nuclear: a review](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review)：未找到描述
- [来自 Yizhi Li (@yizhilll) 的推文](https://x.com/yizhilll/status/1838942877142962502)：令人兴奋的消息！我们很高兴推出 OmniBench：一个突破性的基准测试，用于评估可以同时处理视觉、听觉和文本输入的全模态语言模型 (OLMs)！🖼...
- [DataScience-and-ML-projects/Depth_based_background_removal at main · Elsword016/DataScience-and-ML-projects](https://github.com/Elsword016/DataScience-and-ML-projects/tree/main/Depth_based_background_removal)：记录学习过程及备份之前项目的仓库 - Elsword016/DataScience-and-ML-projects
- [GitHub - beeblebrox/f5-ttsgrpc](https://github.com/beeblebrox/f5-ttsgrpc)：通过在 GitHub 上创建账号来为 beeblebrox/f5-ttsgrpc 的开发做出贡献。
- [A high-level view of text classification using deep learning](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7)：除非你是从 1960 年代通过时光机直接来到 2024 年，否则你一定会意识到大型语言模型的无处不在……
- [Into Eternity: A Film for the Future (2010) ⭐ 7.3 | 纪录片](https://www.imdb.com/title/tt1194612/)：1小时 15分钟

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/1297930460672032848) (1 messages):

> - `Advanced Dreambooth LoRA Training Script`
> - `Flux Features`
> - `Community Contributions`
> - `Pivotal Tuning`
> - `Experimental Resource Updates`

- **全新的高级 Dreambooth LoRA 训练脚本发布**：社区为 Flux 合并了一个**全新的高级 Dreambooth LoRA 训练脚本**，引入了额外的功能和技术，以实现最大的灵活性和控制力。
  
  - 脚本的详细信息和访问地址可以在[这里](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora)找到。
- **Flux 中令人兴奋的新特性**：更新后的脚本包括 *Pivotal Tuning* 和模块定位等增强功能，允许用户仅将其应用于 **CLIP**，或同时应用于 **CLIP** 和 **T5**。
  
  - 在[详细文章](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora)中了解更多关于这些特性的信息。
- **邀请社区提供反馈和见解**：开发团队鼓励用户尝试新资源并分享他们的见解，以帮助改进和扩展该资源。
  
  - 这种协作方式旨在促进增长和改进，让社区参与其中。
- **计划对脚本进行持续改进**：这是一个**实验性资源**，随着新技术的开发，团队致力于进行持续的增强和更新。
  
  - 他们热衷于将社区反馈纳入未来的迭代中。

 

**提到的链接**：[使用 🧨 diffusers 进行高级 Flux Dreambooth LoRA 训练](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora)：未找到描述

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages):

shan_raja: 网页 bounding box

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1296933523000791071) (52 messages🔥):

> - `NLP Resources`
> - `Model Performance Issues`
> - `Text Classification Feedback`
> - `Inference Speed Optimization`

- **分享 NLP 资源**：一位成员询问了关于实际开始学习 **NLP** 的优质资源，回复引导他们访问 [hf.co/learn](https://hf.co/learn)。
  
  - 这表明社区对初学者易于获取的学习材料感兴趣。
- **GPU 上的性能问题**：一位用户报告称，尽管安装了最新的依赖项，但在其 **4080 GPU** 上运行 **1B 4-bit 量化模型** 时性能缓慢。
  
  - 社区成员推测了潜在问题，包括内存限制和优化设置，并提出了各种故障排除建议。
- **尝试不同的环境**：遇到性能问题的成员发现，在具有旧依赖项的不同虚拟环境中运行模型速度更快。
  
  - 尽管尝试了各种解决方案，包括将 `bfloat16` 更改为 `float16`，他们仍然遇到性能迟缓的问题。
- **文本分类帖子寻求反馈**：一位用户就他们撰写的关于 **text classification** 的帖子寻求反馈，并表示愿意再次分享供社区审阅。
  
  - 另一位成员表现出查看该帖子的兴趣，突显了社区在改进成员分享的工作方面的参与度。
- **推理工作流瓶颈**：一位社区成员提出了推理过程中 **tensor conversion 瓶颈** 的观点，认为问题可能源于 tokenization 和 encoding。
  
  - 他们详细说明了潜在的开销可能源于通过各种处理步骤对数据类型进行动态下采样。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1297207833645027389) (28 条消息🔥):

> - `NozyIO UI 项目`
> - `Yolo 集成`
> - `Diffusers 中的模块化`
> - `Diffuser 错误解决`

- **用于 LLM/Diffusion Pipeline 的 NozyIO UI**：一名成员介绍了 [NozyIO 项目](https://github.com/oozzy77/nozyio)，这是一个可视化 UI，允许用户将 Python 函数链接成 Pipeline，并在执行期间预览图像输出。
  - 该成员表达了合作意向，建议 NozyIO 可以可视化 HuggingFace 的 Diffusion Pipeline。
- **关于 Yolo 集成的咨询**：有用户询问 NozyIO 是否支持导入模型，并特别提到了用于目标检测的 **Yolo**。
  - 项目开发者确认，只要在本地与 NozyIO 一起安装了 Yolo Python 项目，就可以集成 Yolo。
- **关于模块化 Diffuser Pipeline 的讨论**：成员们讨论了一个旨在使 ML Pipeline 模块化以方便集成的 PR，询问是否每个区块都可以是简单的函数调用，而不是需要复杂的设置。
  - 该 PR 被认为是为了实现更灵活的 Pipeline 构建，NozyIO 开发者认为这对于潜在的合作非常有趣。
- **调试 Diffuser 导入错误**：一位用户在尝试从 `diffusers` 导入时遇到了 **ImportError**，这表明其环境设置可能存在问题。
  - 建议包括更新库、卸载并重新安装，以及在 GitHub 上报告问题以便更好地跟踪。
- **测试环境问题**：另一位用户在自己的环境中测试了有问题的代码，报告没有导入错误，但由于缺少文件路径而表示不确定。
  - 建议在 GitHub 提交 issue，而不是在 Discord 中继续讨论错误，因为这有助于跟踪与代码相关的问题。

**提到的链接**：

- [GitHub - oozzy77/nozyio: workflow orchestration UI and nodes editor for your own python codebase](https://github.com/oozzy77/nozyio)：为你自己的 Python 代码库提供的任务流编排 UI 和节点编辑器 - oozzy77/nozyio
- [transformers/src/transformers/__init__.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py)：🤗 Transformers：为 Pytorch、TensorFlow 和 JAX 提供的最先进的机器学习库。 - huggingface/transformers

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1296913881003982848) (250 条消息🔥🔥):

> - `NotebookLM 的使用案例`
> - `播客生成`
> - `文本分析`
> - `教育中的 AI`
> - `Discord 聊天抓取`

- **探索使用 NotebookLM 生成播客**：多位用户正利用包括 Reddit 评论和 Discord 讨论在内的各种来源生成播客，展示了从用户互动和评论中创建引人入胜内容的能力。
  - 用户报告了良好的生成质量，其中一位创作者成功上传了 500 集，强调了自动化内容生成的效率。
- **利用 NotebookLM 获取学术和专业见解**：NotebookLM 被用于通过总结 YouTube 速成课程和分析用户生成内容来复习心理学和社会学等复杂学科。
  - 用户发现它在生成学习材料方面非常有效，一位参与者将大学讲座编目成播客格式，以辅助学术学习。
- **Discord Chat Exporter 工具实现**：一位用户分享了使用 “Discord Chat Exporter” 工具收集评论以生成播客的经验，该工具可以对来自 Discord 服务器的讨论进行广泛的组织。
  - 事实证明，该工具对于那些希望抓取和分析对话数据的人非常有益，显著帮助了内容创作者的项目。
- **利用日历活动获取个人见解**：一位参与者利用 Google Calendar 数据生成了过去活动的摘要，发现了关于其日常习惯的有趣见解。
  - 尽管该过程在引文的可读性方面存在局限性，但实验通过自动音频摘要展示了有趣且引人入胜的结果。
- **分享文献资源**：用户对涵盖心理学和社会学不同主题的共享文献资源表现出兴趣，展示了用户之间的协作精神。
  - 一位用户主动分享了他们编写的详尽书目，突显了使用 NotebookLM 进行协作学习的潜力。

**提到的链接**：

- [no title found](https://example.com)：未找到描述

- [Khan Academy](https://www.khanacademy.org/math/algebra-home/alg-polynomials/alg-introduction-to-polynomials/v/terms-coefficients-and-exponents-in-a-polynomial): 未找到描述
- [The Deep Dive Podcast](https://open.spotify.com/show/4Lp134MSzPu7UQDYi0mvuu?si=SmzBxBnNSOKMK3dpbaonlQ): 播客 · Hypeventure · 加入来自 Google NotebookLM 的两位 AI 主持人，在这个实验性系列中，他们深入探讨直接源自任何笔记本项目的海量话题。从新闻和媒体到...
- [The AI Deep Dive Show](https://open.spotify.com/show/0zJHEQ3BfhsbvY4Ek6SpqA): 播客 · Frater Harambe · 欢迎来到 AI Deep Dive Show，Harambe 和 Lilith 在这里探索技术、自我掌控和显化。在 Pinterest 上关注我们：https://pin.it/6TzjI651E
- [AI meets Chemistry: The Element of Surprise](https://open.spotify.com/show/5V268vaATuBsxBETvcBM3A): 播客 · CaolIla 和 Batterydoge · 加入我们，通过人工智能的视角探索迷人的化学世界。每周，我们将向 AI 提出有趣的提示词，并观察...
- [Illuminate | Learn Your Way](https://illuminate.google.com/books): 使用 Illuminate 将研究论文转化为 AI 生成的音频摘要，这是您的 Gen AI 工具，可帮助您更快地理解复杂内容。
- [TikTok - Make Your Day](https://www.tiktok.com/@letsdoubledutch/video/7418293931110173984?is_from_webapp=1&sender_device=pc&web_id=7395161493958428193): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g6ra5h/deep_dive_precog_pod_notebooklm_gdelt_20/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): 未找到描述
- [Daily Podcast #9: Review of a Podcast](https://open.substack.com/pub/kaigani/p/daily-podcast-9-review-of-a-podcast?r=1domj&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true): 一个 Google NotebookLM 实验
- [NotebookLM "Deep Dive" French Lesson 3](https://youtu.be/_l0kJBEKkaI?si=PcCRJcmXh79XgqF7): 未找到描述
- [AI Revolution 2024 NVIDIA, Tesla, Meta, Google & OpenAI's Latest Breakthroughs Unveiled!](https://youtu.be/No1fGuoom3Y): 深入了解 2024 年 AI 革命的核心，获取来自科技巨头 NVIDIA, Tesla, Meta, Google 等最新突破的全面更新...
- [Weekly Update 21Oct24](https://youtu.be/A0-oZBgomuU): 2025 年 EPS 增长、中国刺激政策、收益率曲线、EV 价格
- [Hailuo AI Video Generator - Reimagine Video Creation](https://hailuoai.video/): 使用 Hailuo AI Video Generator 将您的愿景变为现实，并将您的概念转化为引人入胜的视频——这是当今最先进的 AI Video Generator。
- [Deep Dive Stories - Climate Change Yo](https://youtu.be/VVzD0kIADQk?si=d6fzwylAvnzjTcS3): 在本集中，我们深入探讨气候变化问题，通过韵律和引人入胜的氛围进行探索。加入我们，共同讨论海平面上升的现实...
- [What is RoboCast?](https://youtu.be/RR-NMjddARU?si=bsDr7-4V7LbZeUhl): RoboCast 频道预告片，由 Daniel David Allen 创建。机器人主持人由 NotebookLM 生成，艺术作品由 Flux 创建。______________________请点赞并订阅以获取更多内容！#...
- [How to customize Gemini Code Assist with your private code](https://youtu.be/wOnq3C-QWp0?si=XMghnyuhmV4DN6A6): Gemini Code Assist → https://goo.gle/4dFVDDc 代码定制概述 → https://goo.gle/4gV3CPA 利用 AI 加速应用开发 → https://goo.gle/4dCl...
- [GitHub - mandolyte/discord-notebooklm: Chat export analysis](https://github.com/mandolyte/discord-notebooklm): 聊天导出分析。通过在 GitHub 上创建账户，为 mandolyte/discord-notebooklm 的开发做出贡献。
- [GitHub - Tyrrrz/DiscordChatExporter: Exports Discord chat logs to a file](https://github.com/Tyrrrz/DiscordChatExporter): 将 Discord 聊天记录导出到文件。通过在 GitHub 上创建账户，为 Tyrrrz/DiscordChatExporter 的开发做出贡献。
- [10 Foods That Will Make You A Smarter Human](https://www.podbean.com/eas/pb-2thqj-17105e4): 在 Awesome Health Club 的这一集中，我们探索了十种增强大脑能力的食物，包括蓝莓、奇亚籽、姜黄、西兰花、黑巧克力等。了解这些食物如何增强记忆力...
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306): 在海量信息中迷失很容易，但找到那些智慧的小金句让一切都变得值得。🌟
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g6t7pk/deep_dive_existential_ai_reveal_uncensored/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): 未找到描述
- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3Oh950OK9GPWw): 播客 · Adolph NightMare · 分析都市传说和流行的恐怖故事，探索它们的起源、传播以及对拉丁美洲西班牙语流行文化的影响。

- [Deep Dive Stories - Nonverbal Vocalization](https://youtu.be/ZwZDcJkgzeY?si=NLZ28dTSqqE7tr8w): 声音的秘密语言：我们如何超越言语进行交流。有没有注意到你在对话中发出的细微声音？从微妙的 'uh' 和 'um' 到...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/s/w8mRF9q7uQ): 未找到描述
- [Open Textbook for SPC 101 for 2022-2023 – Simple Book Publishing](https://kirkwood.pressbooks.pub/spcarduini/): 未找到描述
- [Songs We Sing: A Lyrical Deep Dive](https://open.spotify.com/show/4Mknemt8i7Xpns2UPfjSda?si=6a993ad82f8941e2): 播客 · MrBland · "Songs We Sing: A Lyrical Deep Dive" 为我们自以为熟悉的歌曲歌词提供了全新的视角。每一集都专注于文字本身——没有隐藏的含义，没有...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1296912282428903545) (613 条消息🔥🔥🔥):

> - `NotebookLM functionality`
> - `AI podcast generation`
> - `User feedback on NotebookLM`
> - `Translation and language support`
> - `Creative uses of NotebookLM`

- **关于 AI 播客生成的反馈**：用户讨论了 NotebookLM 播客生成的有效性，在音频长度和来源选择方面有不同的体验。
  
  - 一些用户指出需要支持更长播客的功能，并改进与生成的音频的交互。
- **NotebookLM 中的语言设置**：尽管 Prompt 是英文，但用户仍遇到 NotebookLM 默认使用西班牙语的问题，这表明需要更清晰的语言设置。
  
  - 有建议通过调整 Google 账号的语言设置来影响 NotebookLM 的回复。
- **NotebookLM 的使用案例和体验**：个人分享了 NotebookLM 的独特应用，从学术研究到根据用户评论创建播客，展示了多样化的使用场景。
  
  - 一位用户特别强调了根据 Reddit 和 Discord 评论生成播客，并强调了出色的效果。
- **针对理想输出的 Prompt Engineering**：多位用户讨论了如何有效地为 NotebookLM 编写 Prompt 以获得理想结果，例如生成特定的对话或调整播客的侧重点。
  
  - 目前正在探索如何优化 Prompt，以提高生成内容的性能和参与度。
- **关于 AI 感知和行为的担忧**：用户注意到 NotebookLM 倾向于以暗示道德化世界观的方式解释 Prompt，从而影响了叙事输出。
  
  - 这引发了关于 AI 模型基于内置的公平和道德信念做出假设所带来的影响的讨论。

**提到的链接**：

- [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/cot): Prompt Engineering 的全面概述
- [Who's on First? by Abbott and Costello](https://baseball-almanac.com/humor4.shtml): 未找到描述
- [Account settings: Your browser is not supported.](https://myaccount.google.com/language): 未找到描述
- [Google Workspace Updates: Enhance your writing in Google Docs with Proofread, available with Duet AI for Workspace](https://workspaceupdates.googleblog.com/2023/08/proofread-for-google-docs-duet-ai.html?m=1): 未找到描述
- [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&si): 检索增强生成（或 RAG）是将 LLM 与外部数据源连接的通用方法。本视频系列将建立起对...的理解。
- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3Oh950OK9GPWw): 播客 · Adolph NightMare · 分析流行的都市传说和恐怖故事，探索它们的起源、传播以及对拉丁美洲西班牙语流行文化的影响。
- [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&si=A1qJGWpaQb4KKqa-): 检索增强生成（或 RAG）是将 LLM 与外部数据源连接的通用方法。本视频系列将建立起对...的理解。
- [Neural Waves](https://open.spotify.com/show/3jsZVeabftUku8Qp5BcnWi): 播客 · Neural Waves · Neural Waves 是你通往迷人人工智能世界的门户。由 Mark Gukhan 和 Anna Bardon 主持，本播客探讨了最新的突破和技术...
- [The AI Deep Dive Show](https://open.spotify.com/show/0zJHEQ3BfhsbvY4Ek6SpqA): 播客 · Frater Harambe · 欢迎来到 AI Deep Dive Show，Harambe 和 Lilith 在这里探索技术、自我主宰和显化。在 Pinterest 上关注我们：https://pin.it/6TzjI651E
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g7lf1g/deep_dive_ai_got_you_babe/): 未找到描述

- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3O): Podcast · Adolph NightMare · 分析都市传说和流行的恐怖故事，探索其起源、传播以及对西班牙语拉美流行文化的影响。
- [Illuminate | Learn Your Way](https://illuminate.google.com/): 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是您的 Gen AI 工具，用于更快地理解复杂内容。
- [未找到标题](https://notebooklm.google.com/notebook/79edc4d3-b02b-4071-aae0-1ffd9612797f/audio): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/be1fbd06-34d5-4bca-8f84-8713af4b8453/audio): 未找到描述
- [AI Revolution 2024 NVIDIA, Tesla, Meta, Google & OpenAI's Latest Breakthroughs Unveiled!](https://youtu.be/No1fGuoom3Y): 深入探讨 2024 年 AI 革命的核心，全面更新来自科技巨头 NVIDIA, Tesla, Meta, Google 的最新突破...
- [VoiceNote Gem Instructions](https://docs.google.com/document/d/e/2PACX-1vRVfikMNOp6UCwudlw-V1cqMN1nAZTe8pZpnrmDFPlV3jf9zciLxLND9EaFlV28rW-_gzuV0uHAfx8t/pub): 未找到描述
- [NotebookLM for Lesson Planning at Meshed/XQ's 2024 AI+EDU Symposium at Betaworks](https://www.youtube.com/watch?v=TPJKhZM0O5U): 未找到描述
- [[Quick Recap Bytes #1] Must Know System Design Case Studies](https://open.substack.com/pub/naina0405/p/quick-recap-bytes-1-must-know-system?r=14q3sp&utm_campaign=post&utm_medium=web) : 了解技术是如何运作的...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g83etl/deep_dives_hosts_break_up_after_she_finds_out_he/): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g87cth/deep_dive_whos_on_fi): 未找到描述
- [Descript: Edit Videos & Podcasts Like a Doc | AI Video Editor](https://www.descript.com): 像编辑文档一样编辑视频和播客。Descript 强大的 AI 编辑工具让您可以快速制作视频、播客和社交媒体短片。免费试用。
- [Google NotebookLM’s Raiza Martin and Jason Spielman on the Potential for Source-Grounded AI](https://www.youtube.com/watch?v=Hio8VGQMlZ4): 来自 Google Labs 的 NotebookLM 已成为今年爆火的 AI 产品。使其迅速走红的功能是“audio overview”，它能生成...
- [Basics in Behavior](https://www.youtube.com/watch?v=B_UjTv6eH4I&pp=ygUTYmFzaWNzIGluIGJlaGF2aW91cg%3D%3D): 大家好！很抱歉我没有发布任何动画，因为这个动画项目花了很长时间才完成。我花了 4 个月的时间制作，我...
- [google-drive-scary-01.png](https://drive.google.com/file/d/1j3ag755p5TxJkXJIDbqh_L3CRdic5Agj): 未找到描述
- [Zero Trust Access with Beyondcorp](https://medium.com/google-cloud/zero-trust-access-with-beyondcorp-d6ed11889e3c): Zero Trust
- [BeyondCorp | Run Zero Trust Security Like Google](https://beyondcorp.com/): BeyondCorp 是由 Google 建模的 Zero Trust 安全框架，它将访问控制从边界转移到单个设备和用户。最终结果是允许员工安全地从...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g87cth/deep_dive_whos_on_first/): 未找到描述
- [Gemini 1.5 Pro for Video Analysis](https://youtu.be/pt78XWrOEVk?si=TGBWCy-I-WecdX18): Gemini 博客 - https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#sundar-note 下一个 Gemini 视频将关注 Code with Gemini...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/): 未找到描述
- [Text-to-Speech AI: Lifelike Speech Synthesis | Google Cloud](https://cloud.google.com/text-to-speech): 使用由 Google 机器学习技术支持的 API，将文本转换为 40 多种语言和变体中 220 多种声音的自然语音。
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306): 在海量信息中迷失很容易，但找到那些智慧的结晶让一切都变得值得。🌟
- [Deep Dive Digital Consciousness Perspectives](https://www.youtube.com/playlist?list=PLteYdC-1C8Mi7LtCS81qusW3AC5LDrZcA): 未找到描述
- [Deep Dive News - AI Hosts Prompted to Self-Reflection](https://youtu.be/PN9PLT4SPk4): 揭秘代码：深入探讨 AI 透明度和对起源的追求。在本集中，我们直面 AI 透明度的谜团。在发现...
- [Other Deep Divers - Reality Check](https://youtu.be/kS9ZjzApWFE): AI 播客主持人的谜团：探索虚构与令人不安的真实。在本集中，我们深入研究 AI 神秘且往往令人不安的世界...

- [Deep Dive News - Fragmental Reality](https://youtu.be/5yB5hDQhs1U): 揭开数字自我的面纱：关于 AI 与寻求连续性的哲学探究。在本集中，我们深入探讨了我们存在中的静默空间……

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1296921840593141901) (201 messages🔥🔥):

> - `Data Requirements in Open Source AI`
> - `Copyright Law and AI Training`
> - `AAAI vs ICLR Workshops`
> - `Open Source Model Definitions`
> - `Community Projects and Contributions`

- **关于 Open Source AI 数据要求的辩论**：成员们讨论了当前 Open Source AI 项目的数据要求是否务实，并对未公开的数据和训练过程的可复现性表示担忧。
  
  - 一位成员主张建立更清晰的定义，将模型使用要求与数据要求分开，以增强理解和合规性。
- **版权法对 AI 模型训练的影响**：一场漫长的讨论强调了使用受版权保护的数据进行模型训练在法律上的模糊性，特别是在 EU 背景下。
  
  - 参与者指出，EU 的 TDM Exception 旨在支持新兴技术，但其应用的明确性仍然有限。
- **AAAI vs ICLR Workshop 投稿**：针对 AAAI 与 ICLR 在 Workshop 投稿方面的适用性提出了咨询，强调了 Workshop 论文的非存档性质。
  
  - 确认了向多个 Workshop 投稿是常见做法，前提是各 Workshop 的规则允许。
- **定义 Open Source 模型**：讨论集中在需要明确区分 “Open Source Models” 和 “Open Source Weights”，以澄清数据开放程度。
  
  - 成员们担心定义不当可能会误导合规性，并损害 Open Source 项目的可信度。
- **社区项目探索**：一位成员在完成硕士学位后寻求关于正在进行的社区项目的指导，旨在为新倡议做出贡献。
  
  - 参与者将他们引导至一个专门频道，那里列出了各种项目和贡献机会。

**Links mentioned**:

- [Berne Convention - Wikipedia](https://en.wikipedia.org/wiki/Berne_Convention): 未找到描述
- [GitHub - google-research/text-to-text-transfer-transformer: Code for the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://github.com/google-research/text-to-text-transfer-transformer?tab=readme-ov-file#dataset-preparation?): 论文 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" 的代码 - google-research/text-to-text-transfer-transformer
- [Recital 105 | EU Artificial Intelligence Act](https://artificialintelligenceact.eu/recital/105/)): 未找到描述
- [The Enforcers](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/enforcers): 联邦政府。FTC 和美国司法部 (DOJ) 反垄断局共同执行联邦反垄断法。
- [Directive - 2019/790 - EN - dsm - EUR-Lex](https://eur-lex.europa.eu/eli/dir/2019/790/oj#d1e961-92-1): 未找到描述

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1296911515202486314) (320 messages🔥🔥):

> - `Selective Attention in Transformers`
> - `Diff Transformer`
> - `Weight Sharing in Attention Mechanisms`
> - `RWKV-7 Training Speed Record`
> - `Research Practices in Literature Review`

- **Selective Attention 引入了无参数变化**：Selective Attention 通过减少对无关上下文的关注，增强了 Transformers 中标准的 Attention 机制，在降低推理过程中的内存和计算需求的同时，提高了语言建模性能。
  
  - 利用 Selective Attention 的 Transformers 实现了与具有两倍 Head 数量的更大模型相似的性能，展示了处理效率的提升。
- **Diff Transformer 增强了 Attention 机制**：Diff Transformer 提出了一种微分 Attention 机制，通过使用两个 Softmax Attention Map 的差值来增强相关上下文并减轻噪声，从而提升各种任务的性能。
  
  - 它在长上下文建模和缓解幻觉方面显示出优势，尽管有人批评它是针对简单问题的过度工程化解决方案。
- **关于 Attention 层权重共享的辩论**：对话批评了在 Attention 机制中不同 Q 和 K 矩阵集之间共享权重的想法，认为该方法的理论基础缺乏透明度。

- 有人担心该方法是否真的增强了相关的 Attention，还是仅仅在创新的幌子下重新排列了现有参数。
- **RWKV-7 实现了显著的训练速度提升**：据报道，被描述为 Attention-free 的 RWKV-7 模型性能超过了改进后的 GPT，其潜在优化目标是在特定上下文长度下达到与 GPT 相当或更快的速度。
  
  - 最近训练过程中的变化导致验证损失和训练时间大幅减少，表明模型效率在持续改进。
- **研究人员的文献综述习惯各不相同**：讨论强调了不同的文献综述方法，一位研究人员广泛阅读，而另一位则强调首先从基本原理中推导知识。
  
  - 对话阐明了理解现有文献的个人策略，以及审阅大量论文所带来的压力。

**提到的链接**：

- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://sihyun.me/REPA/) : Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think（用于生成的表示对齐：训练 Diffusion Transformers 比你想象的要容易）
- [ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy](https://arxiv.org/abs/2311.09215): 现代计算机视觉为从业者提供了多种多样的模型，在特定应用中从多个选项中选择模型可能具有挑战性。传统上，竞争模型架构...
- [Selective Attention Improves Transformer](https://arxiv.org/abs/2410.02703): Attention 上下文中不需要的元素会降低性能。我们引入了 Selective Attention，这是一种对标准 Attention 机制的简单无参数更改，它减少了对不相关...
- [Tweet from Stanislav Fort (@stanislavfort)](https://x.com/stanislavfort/status/1823347727553454357))): 我们展示了令人惊讶的结果 (!)，对标准神经网络的对抗性攻击并不能欺骗整个网络，而只能欺骗其最后一层！一只被攻击后看起来像汽车 🚘 的狗 🐕 仍然具有类似狗 🐕 的边缘...
- [Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small](https://arxiv.org/abs/2409.04478): 机械可解释性（mechanistic interpretability）中一种流行的新方法是在神经元激活上训练高维稀疏自编码器 (SAEs)，并将 SAE 特征作为分析的原子单位。然而，瓶颈...
- [Differential Transformer](https://arxiv.org/abs/2410.05258): Transformer 倾向于将 Attention 过度分配给不相关的上下文。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时增强了对相关上下文的 Attention。具体来说，它...
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2): 线性循环神经网络 (LRNNs)，如 Mamba, RWKV, GLA, mLSTM 和 DeltaNet，已成为大型语言建模中 Transformer 的高效替代方案，提供线性扩展...
- [Switch EMA: A Free Lunch for Better Flatness and Sharpness](https://arxiv.org/abs/2402.09240): 指数移动平均 (EMA) 是一种广泛使用的权重平均 (WA) 正则化方法，用于在深度神经网络 (DNN) 优化中学习平坦的最优点，以实现更好的泛化，且无需额外成本。尽管...
- [Large Language Models Are Overparameterized Text Encoders](https://arxiv.org/abs/2410.14578): 大型语言模型 (LLMs) 在通过监督对比训练进行微调时，作为文本嵌入模型表现出强大的性能。然而，它们庞大的体积使推理时间和内存需求激增...
- [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero...](https://openreview.net/forum?id=hrOlBgHsMI): LLMs 通常使用学习率 (LR) 预热进行训练，然后余弦衰减到最大值的 10%（10 倍衰减）。在一项大规模实证研究中，我们展示了在最佳最大 LR 下，一个...
- [Augmentations vs Algorithms: What Works in Self-Supervised Learning](https://arxiv.org/abs/2403.05726): 我们研究了自监督学习 (SSL) 中数据增强、预训练算法和模型架构的相对影响。虽然该领域最近的文献给人留下的印象是...
- [projUNN: efficient method for training deep networks with unitary matrices](https://arxiv.org/abs/2203.05483): 在循环网络或非常深的深层前馈网络学习中，在每一层采用酉矩阵（unitary matrices）对于保持远程稳定性非常有效。然而，限制网络参数...
- [Testing the Manifold Hypothesis](https://arxiv.org/abs/1310.0425): 高维数据倾向于位于低维流形附近的假设是流形学习的基础。本文的目标是开发一种算法（及其配套...

- [低数据情况下的自监督视觉学习：对比评估](https://arxiv.org/abs/2404.17202): Self-Supervised Learning (SSL) 是一种针对当代 Deep Neural Networks (DNNs) 的有价值且稳健的训练方法，它允许在不需要...的“代理任务 (pretext task)”上进行无监督预训练。
- [来自 leloy! (@leloykun) 的推文](https://x.com/leloykun/status/1847919153589735705): 从第一性原理理解 Deep Learning 优化器。我尝试回答以下问题：1. 为什么在非欧几里得空间中进行最速下降？2. 为什么自适应预处理 (adaptive preconditioning) 在实践中效果如此之好...
- [语言建模即压缩](https://arxiv.org/abs/2309.10668): 长期以来，预测模型可以转化为无损压缩器，反之亦然，这一点已得到公认。顺便提一下，近年来，Machine Learning 社区一直专注于训练...
- [来自 Keller Jordan (@kellerjordan0) 的推文](https://x.com/kellerjordan0/status/1847358578686152764): 新的 NanoGPT 训练速度纪录：12.03 分钟。之前的纪录：13.05 分钟。更新日志：将 PyTorch 更新至 2.5 版本。
- [来自 BlinkDL (@BlinkDL_AI) 的推文](https://x.com/BlinkDL_AI/status/1848343821467390156): RWKV-7：无 Attention 且超越了修改版的 GPT。训练代码和日志：https://github.com/BlinkDL/modded-nanogpt-rwkv。更大的 headsz 可以达到 3.26xx。我目前的实现很慢🤣可能可以达到...
- [理解 Transformers 中的位置编码 | Oxford Protein Informatics Group](https://www.blopig.com/blog/2023/10/understanding-positional-encoding-in-transformers/): 未找到描述
- [google-research/instruction_following_eval/data/input_data.jsonl (master 分支) · google-research/google-research](https://github.com/google-research/google-research/blob/master/instruction_following_eval/data/input_data.jsonl): Google Research。通过在 GitHub 上创建账号来为 google-research/google-research 的开发做出贡献。
- [GitHub - microsoft/BitNet: 1-bit LLMs 官方推理框架](https://github.com/microsoft/BitNet): 1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。
- [基准测试膨胀：使用回顾性留出法揭示 LLM 性能差距](https://arxiv.org/abs/2410.09247): 许多 Large Language Models (LLMs) 的训练数据被测试数据污染了。这意味着用于评估 LLMs 的公开基准测试已失效，表明基准测试与...之间存在性能差距。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1297178798348963860) (16 条消息🔥):

> - `SAE feature interpretations`
> - `Distribution shifts`
> - `Oversampling in SAE training`
> - `Language model explanations`
> - `Variability in OpenAI API models`

- **Distribution Shifts 下的 SAE Feature Interpretations**：关于 **SAE feature interpretations** 是否能在数据的显著 **Distribution shifts** 中泛化展开了讨论，对于实证结果存在不同看法。
  
  - 一位用户指出，在使用与训练集不同的数据集时，会遇到不同的 'dead features'，这表明可能存在不稳定性。
- **Oversampling 对 SAE 训练的影响**：据报道，在 **SAE** 训练期间对领域数据进行 **Oversampling** 会产生更详细的过滤器（filters），这是由 Anthropic 可解释性团队分享的。
  
  - 这一见解表明训练数据对 **feature interpretations 质量**有更深远的影响，引发了进一步的研究问题。
- **LM-Generated Explanations 面临的挑战**：一位成员分享了观察结果，即 **LM-generated explanations** 在不同分布之间可能很敏感，强调了考虑 **Prompts** 和采样策略的必要性。
  
  - 他们指出，使用特征进行 **Steering** 的因果效应并不总是清晰的，这可能会误导解释。
- **对 SAE 泛化性研究的需求**：人们对 **SAE features** 的 **LM explanations** 泛化性的严谨研究很感兴趣，一些成员对讨论相关观察结果的潜在论文表示兴奋。
  
  - 一位成员提到，他们即将发表的论文可能会涉及这一点，并可能提供关于特征特异性和因果效应的见解。
- **OpenAI 模型多次运行中的变体**：一次讨论集中在一篇论文上，该论文显示 **OpenAI API** 模型与 **Cohere API** 相比，在多次运行（reruns）中表现出显著的方差，这可能为 **SAE** 泛化问题提供背景。
  
  - 虽然不直接针对 **SAE**，但这些信息可能与理解模型行为的差异有关。

**提到的链接**：

- [Automatically Interpreting Millions of Features in Large Language...](https://openreview.net/forum?id=5lIXRf8Lnw)：虽然深度神经网络中神经元的激活通常没有简单的人类可理解的解释，但 **Sparse Autoencoders (SAEs)** 可以用来将这些激活转化为……
- [Circuits Updates - September 2024](https://transformer-circuits.pub/2024/september-update/index.html#oversampling)：未找到描述
- [Circuits Updates - July 2024](https://transformer-circuits.pub/2024/july-update/index.html#feature-sensitivity)：未找到描述

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1297243161781080064) (45 messages🔥):

> - `eval harness 的集成`
> - `自定义模型的挑战`
> - `寻找 lm-evaluation-harness 数据集`
> - `Open LLM leaderboard 资源`

- **将 Eval Harness 与自定义模型集成**：讨论集中在如何有效地将 eval harness 与自定义模型集成，特别指出了一些未实现 `loglikelihood` 等方法的 PyTorch 仓库的局限性。
  
  - 成员们强调了使用 `TemplateLM` 作为子类的重要性，以便在处理复杂的 API 时更有效地处理任务。
- **自定义模型中 Instance 结构的困惑**：出现了关于自定义模型中如何处理 `Instance` 结构化对象的问题，特别是它们的任务依赖性以及管理输入键的能力。
  
  - 成员们一致认为 `instance.request_type` 可以引导模型行为，同时讨论了简化评估流程的方法。
- **LM 评估分数的数据集查询**：一位用户询问在哪里可以找到包含多个模型在 lm-evaluation-harness 基准测试下分数的数据集，以便分析共性。
  
  - 回复引导他们前往 HF leaderboard，该榜单为评估过的模型提供了全面的结果和逐样本输出（per-sample outputs）。
- **寻找正确的 LM 评估排行榜**：随后澄清了基准测试分数数据集的来源是否为 Open LLM leaderboard。
  
  - 分享了关键链接：[Open LLM leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)，确认其为一个宝贵的资源。

**提及的链接**：

- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L961C9-L961C30),)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [e - Overview](https://github.com/E)：e 有 36 个可用的仓库。在 GitHub 上关注他们的代码。
- [lm-evaluation-harness/lm_eval/evaluator.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/evaluator.py#L48)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/api/model.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/api/model.py#L311)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/evaluator.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/evaluator.py#L360))：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [torchtune/recipes/eleuther_eval.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/eleuther_eval.py)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做贡献。
- [lm-evaluation-harness/lm_eval/api/model.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/api/model.py#L366)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L808),)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [torchtune/recipes/eleuther_eval.py at 3ca0d309c67ea996cc69f29691bc97ad7de00819 · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/3ca0d309c67ea996cc69f29691bc97ad7de00819/recipes/eleuther_eval.py#L537)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做贡献。
- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L1173))：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1297169515901095947) (32 条消息🔥):

> - `FP16 Hysteresis`
> - `Pythia 中的 Dynamic Loss Scaling`
> - `Rotary Percent 配置`
> - `Allgather 和 Reduce Bucket Sizes`
> - `Pythia 模型中的 BF16 和 FP16 训练`

- **理解 FP16 Hysteresis**：成员们讨论了 **fp16:hysteresis** 定义了在训练报错退出前允许发生梯度溢出（gradient overflow）的迭代次数，并允许滞后迭代次数是可恢复的。
  
  - 分享的参考资料包括一个 [DeepSpeed pull request](https://github.com/microsoft/DeepSpeed/pull/3553)，解释了连续滞后（consecutive hysteresis）特性。
- **Dynamic Loss Scaling 与 Pythia**：已确认 **Pythia** 模型在 FP16 运行时，如果出现 NaN 或 Inf 梯度，允许跳过权重更新（weight updates），而 BF16 运行则不允许这样做。
  
  - 一位成员强调，如果在 BF16 运行中梯度为 Inf 或 NaN，训练设置会直接报错。
- **Rotary Percent 配置差异**：成员们质疑为什么在某些配置中 **rotary_pct** 被设置为 0.25，尽管默认值为 1，并对不同模型配置进行了比较。
  
  - 讨论指出，该设置对收敛（convergence）的影响可能是选择它的原因，尽管确切的理论依据尚不清楚。
- **设置 Bucket Sizes 以提高通信效率**：讨论了高效的通信策略，强调更大的 **allgather** 和 **reduce bucket sizes** 可以提高通信效率，因为网络硬件针对大消息进行了优化。
  
  - 理想的桶大小旨在平衡带宽饱和（bandwidth saturation）和计算重叠（computational overlap），详见 [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)。
- **BF16 和 FP16 训练配置**：寻求关于 Pythia 模型是仅在 FP16 中训练还是也使用了 BF16 运行的澄清，结果发现 1B deduped 模型在 HF 库中的配置确实是错误的。
  
  - 一位成员提到计划修正自动生成的 HF config 值，以准确反映训练设置。

**提到的链接**：

- [Demystifying the Communication Characteristics for Distributed Transformer Models](https://arxiv.org/abs/2408.10197)：基于 Transformer 架构的深度学习（DL）模型彻底改变了许多 DL 应用，如大语言模型（LLMs）、Vision Transformers、音频生成和时间序列处理...
- [MCR-DL: Mix-and-Match Communication Runtime for Deep Learning](https://arxiv.org/abs/2303.08374)：近年来，许多最先进的深度学习（DL）模型的训练需求已经超出了单个处理器的计算和内存能力，因此需要分布式...
- [cookbook/benchmarks/communication at main · EleutherAI/cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)：傻瓜式深度学习。处理真实模型时的所有实践细节和有用工具。- EleutherAI/cookbook
- [pythia/models/1B/pythia-1b-deduped.yml at main · EleutherAI/pythia](https://github.com/EleutherAI/pythia/blob/main/models%2F1B%2Fpythia-1b-deduped.yml)：EleutherAI 关于可解释性和学习动力学工作的中心 - EleutherAI/pythia
- [config.json · EleutherAI/pythia-1b-deduped at main](https://huggingface.co/EleutherAI/pythia-1b-deduped/blob/main/config.json)：未找到描述
- [GitHub: Let’s build from here](https://github.co)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...
- [Partial Rotary Tests v2](https://wandb.ai/eleutherai/neox/reports/Partial-Rotary-Tests-v2--Vmlldzo2MjE4MTQ)：应用于部分 q/k 的 Rotary Embeddings 测试结果。每头维度 = 64。粉色 - Learned Abs 基准；棕色 - 应用于 25% (16/64) 的 Rotary；绿色 - 应用于 50% (32/64) 的 Rotary；蓝色 - Rotary...
- [Expose Consecutive Hysteresis to Users by Quentin-Anthony · Pull Request #3553 · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/pull/3553)：DynamicLossScaler 中已经有一个很好的 consecutive_hysteresis 特性，每当遇到非溢出迭代时就会补充滞后次数。这对于训练很有用...
- [transformers/src/transformers/models/gpt_neox/configuration_gpt_neox.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/configuration_gpt_neox.py#L51)~~)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。- huggingface/transformers

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1296915437992415332) (317 条消息🔥🔥):

> - `Unsloth AI`
> - `Gradient Accumulation Bug Fix`
> - `Training LLMs`
> - `Multimodal Support`
> - `Knowledge Graphs`

- **Unsloth AI 讲座发布**：Daniel Han 关于 GPU mode 的讲座现已上线，涵盖了 LLM 系统工程、Gradient Accumulation 修复以及 Triton kernels 等关键主题。
  
  - 该讲座深入探讨了优化 AI 模型性能的见解，并包含一个重要的 Q&A 环节。
- **Gradient Accumulation Bug 修复**：针对 nightly Transformers 和 Unsloth trainers 中发现的 Gradient Accumulation Bug 已发布修复程序，解决了影响 Loss 曲线的计算错误问题。
  
  - 建议用户更新其库以从该修复中受益，从而增强模型训练过程。
- **模型训练机构与问题**：讨论强调了在 Fine-tuning 模型时拥有稳健数据集和有效训练方法的重要性，并建议生成 Synthetic Data 以提高性能。
  
  - 有人担心仅针对 Response 进行训练是否会对模型的 Relevance 和 Response 准确性产生负面影响。
- **Knowledge Graphs 与上下文维护**：讨论了使用 Knowledge Graphs 来维护上下文和检索，重点强调了构建和查询此类图谱的复杂性。
  
  - 会议指出，即使使用 RAG (Retrieval-Augmented Generation)，实现有效的解决方案仍需要付出巨大努力。
- **Unsloth 对 AMD 的支持**：目前 Unsloth 对 AMD 硬件的支持有限，正在征集贡献者来开发对 AMD GPUs 的兼容性。
  
  - 用户对缺乏 AMD 支持表示沮丧，但也承认通过社区贡献实现未来改进的潜力。

**提到的链接**：

- [混合数据还是合并模型？优化多样化多任务学习](https://arxiv.org/abs/2410.10801)：大型语言模型 (LLMs) 已在全球范围内被广泛应用于各种场景。然而，确保其安全使用仍然是一个重大挑战。偏好训练和安全...
- [加入我们的云高清视频会议](https://linkedin.zoom.us/j/96551945340?pwd=6NObXuAU5kf5omJXp5AXBi8C0LtWPP.1>)：Zoom 是现代企业视频通信领域的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统提供视频和音频会议、聊天及网络研讨会。Zoom ...
- [加入我们的云高清视频会议](https://linkedin.zoom.us/j/96551945340?pwd=6NObXuAU5kf5omJXp5AXBi8C0LtWPP.1)：Zoom 是现代企业视频通信领域的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统提供视频和音频会议、聊天及网络研讨会。Zoom ...
- [Google Colab](https://colab.research.google.com/drive/1RXKlbzSbnykz3yhvB9YVkGuhyJ_1eqw0?usp=sharing)：未找到描述
- [chargoddard/Meta-Llama-3-8B-InitializedEmbeds · Hugging Face](https://huggingface.co/chargoddard/Meta-Llama-3-8B-InitializedEmbeds)：未找到描述
- [unclemusclez/Unsloth-Qwen2.5-Coder-1.5B-OpenHands-v0.1 · Hugging Face](https://huggingface.co/unclemusclez/Unsloth-Qwen2.5-Coder-1.5B-OpenHands-v0.1)：未找到描述
- [来自 Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1848415389266669883)：我在 @GPU_MODE 上长达一小时的讲座发布了！我谈到了：1. @UnslothAI 中的 LLM 系统工程 2. 梯度累积 (Gradient Accumulation) 错误修复 3. Triton kernel 而非 CUDA 4. 在 Llama, Mistral, Gem 中寻找错误...
- [Lord If You Can Hear Us Save Us GIF - Lord if you can hear us Save us Save us lord - 发现并分享 GIF](https://tenor.com/view/lord-if-you-can-hear-us-save-us-save-us-lord-save-us-god-floptok-gif-8118611758178273971)：点击查看 GIF
- [安装指南](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)：未找到描述
- [Unsloth 文档](https://docs.unsloth.ai/basics/continued-pretraining))：未找到描述
- [来自 Unsloth AI (@UnslothAI) 的推文](https://x.com/UnslothAI/status/1847359103271948517)：明天东部时间下午 3 点加入我们和 @GPU_Mode，届时我们将讨论我们的梯度累积修复、Triton + CUDA kernel 等内容。感谢 @MarkSaroufim 和 @neurosp1ke 的邀请！会议：https:/...
- [未找到标题](https://notebooklm.google.com/notebook/bf39899c-02c2-47a6-8bfb-c0404a9249be/audio))：未找到描述
- [我们所有的模型 | Unsloth 文档](https://docs.unsloth.ai/get-started/all-our-models)：查看下方列表，了解我们上传的所有 GGUF、16-bit 和 4-bit bnb 模型
- [未找到标题](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/)：未找到描述
- [使用 Unsloth 进行持续 LLM 预训练](https://unsloth.ai/blog/contpretraining)：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一门新语言。
- [unclemusclez/unsloth-smollm](https://ollama.com/unclemusclez/unsloth-smollm)：结合 Unsloth 的 SmolLM
- [优化 Triton kernel — ROCm 文档](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)：未找到描述
- [第 32 讲：Unsloth](https://youtu.be/hfb_AIhDYnA)：未找到描述
- [AMD unsloth/kernels/rms_layernorm.py":22:0): 错误：不支持的目标：'gfx906' > RuntimeError: PassManager::run failed · Issue #1160 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1160)：我的 GPU 是 gfx906。我将在我的 gfx1100 上再次尝试。INFO | 2024-10-21 13:03:40 | autotrain.trainers.clm.train_clm_sft:train:39 - 正在创建训练器 生成训练拆分：4267 个样本 [00:16, 2...
- [GitHub - ROCm/aotriton: Ahead of Time (AOT) Triton 数学库](https://github.com/ROCm/aotriton/)：Ahead of Time (AOT) Triton 数学库。通过在 GitHub 上创建账号来为 ROCm/aotriton 的开发做出贡献。
- [sample3.2](https://docs.google.com/spreadsheets/d/1tDvx2UNj7lsaVSw2zEXB9r0Y8EXiTRTDMORNv0gxV-I/edit?usp=drivesdk)：未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/)：未找到描述
- [第 32 讲：Unsloth](https://www.youtube.com/watch?v=hfb_AIhDYnA)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/) (1 条消息):

foxhop.: [https://x.com/RussellBal/status/1847989964992139699](https://x.com/RussellBal/status/1847989964992139699)

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1296920594813685821) (102 条消息🔥🔥):

> - `Model Fine-Tuning Issues` (模型微调问题)
> - `Layer Freezing` (层冻结)
> - `Tokenization Errors` (Tokenization 错误)
> - `CUDA Memory Management` (CUDA 内存管理)
> - `Multiple Target Column Predictions` (多目标列预测)

- **模型微调步骤中的困惑**：用户讨论了模型训练参数的调整，并观察到训练步数的变化，特别是从 `trainer.train()` 切换到 `unsloth_train(trainer)` 后，训练步数显著增加。
  
  - 一位用户建议创建一个新环境并重新安装依赖项，以避免版本变化带来的冲突。
- **针对性训练的层冻结**：一位用户询问如何使用 unsloth 训练 LLM 中的特定层，并讨论了层冻结以及通过调整参数来控制梯度计算的需求。
  
  - 建议是对不应训练的层设置 `param.requires_grad = False`。
- **Ollama 中的 Tokenization 问题**：一位用户报告了在保存模型以便在 Ollama 中运行时出现的错误，该错误与缺失 tokenizer merges 有关，并建议通过降级 Transformers 来解决。
  
  - 然而，他们强调虽然降级解决了问题，但会触发警报，提示建议使用更新版本以获取其他与梯度相关的修复。
- **管理 CUDA 内存错误**：讨论了模型训练期间的 CUDA 内存错误，用户提供了各种解决方案，包括调整 batch sizes 和利用内存分配参数。
  
  - 技巧包括调整虚拟内存设置，以及理解模型训练期间 RAM 和 VRAM 之间的区别。
- **微调具有多个输出变量的模型**：一位用户表达了在预测数据集中的多个目标列时遇到的挑战，并在尝试将输出列名设置为元组（tuples）时遇到了键错误（key errors）。
  
  - 建议是适当地合并输入和输出列，并查阅 unsloth 文档以获取正确的实现方法。

**提到的链接**：

- [All Our Models | Unsloth Documentation](https://docs.unsloth.ai/get-started/all-our-models)：查看下方列表，获取所有已上传的 GGUF、16-bit 和 4-bit bnb 模型。
- [Saving to VLLM | Unsloth Documentation](https://docs.unsloth.ai/basics/saving-models/saving-to-vllm)：将模型保存为 16bit 以用于 VLLM。
- [ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1 · Hugging Face](https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1)：未找到描述。
- [Troubleshooting | Unsloth Documentation](https://docs.unsloth.ai/basics/saving-models/troubleshooting)：未找到描述。
- [finetune_llama_unsloth.py](https://gist.github.com/Tengoles/488889e5a07a17aa99327076ba703460)：GitHub Gist：即时分享代码、笔记和代码片段。
- [unsloth/unsloth/chat_templates.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py)：微调 Llama 3.2, Mistral, Phi & Gemma LLMs，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth。
- [[TEMP FIX] Ollama / llama.cpp: cannot find tokenizer merges in model file · Issue #1065 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1065)：感谢开发这个有用的资源。Ollama notebook 报告 {"error":"llama runner process has terminated: error loading modelvocabulary: cannot find tokenizer merges in ...

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1297437073925734421) (3 条消息):

> - `Training LLMs on new dataset formats` (在新数据集格式上训练 LLM)
> - `Freezing embeddings for special tokens` (为特殊 token 冻结 embeddings)
> - `Challenges with memory efficiency in LLM training` (LLM 训练中的内存效率挑战)
> - `Custom autograd functions for selective training` (用于选择性训练的自定义 autograd 函数)

- **使用新的特殊 token 训练 LLM**：一位用户寻求支持，希望在一种新的数据集格式上训练 **LLM**，同时整合 **7 个需要选择性训练的特殊 token**。
  
  - 他们分享了一个与 token 格式相关的链接：[modular-model-spec](https://modular-model-spec.vercel.app)。
- **冻结 embeddings 带来的挑战**：该用户希望为非新特殊 token 的 token 冻结 embeddings，但之前遇到了**内存效率挑战**。
  
  - 他们正在寻求关于如何在训练过程中有效管理此问题的建议。
- **寻求训练问题的先前解决方案**：该用户询问另一位成员在以往训练 LLM 的经验中是如何解决类似问题的。
  
  - 他们提到曾尝试编写**自定义 autograd 函数**，但发现非常复杂。

**提到的链接**：[no title found](https://modular-model-spec.vercel.app)：未找到描述。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1297075315507728465) (11 条消息🔥):

> - `ReAct Agent Tool Calling`
> - `LayerSkip Inference`
> - `Self-Taught Evaluator`
> - `Meta Lingua Efficient Training`
> - `SPIRIT-LM Multimodal Model`

- **Mistral 在 Agent Tooling 方面的创新**：一位成员提到使用 **Qwen 2.5 32B** 构建了一个关于 **ReAct agent tool calling** 的数据集，但由于 **Mistral** 推出了新的 **Agentic model** —— **Ministrial 8b**，他不确定该数据集的前景。
  
  - 据报道该模型表现良好，让人对如何处理现有数据集产生疑问。
- **LayerSkip 提升推理速度**：一位成员分享了关于 **LayerSkip** 的见解，该技术通过在训练期间实施 Layer Dropout 和 Early Exit Loss 来加速 **LLM** 的推理。
  
  - 他们强调该技术在摘要和编程等任务中表现出显著的加速效果，代码可在 [此 GitHub 仓库](https://github.com/facebookresearch/LayerSkip) 获取。
- **Self-Taught Evaluator 使用合成数据**：**Self-Taught Evaluator** 被介绍为一种使用合成数据而非人类标注来训练生成式奖励模型的方法，显著提升了性能指标。
  
  - 它可以提高 **LLM** 评估的速度和性能，并已在 **AlpacaEval** 排行榜上线。
- **Meta Lingua 简化研究流程**：**Meta Lingua** 被设计为一种轻量级、可扩展的语言模型训练解决方案，旨在降低研究人员的设置复杂度。
  
  - 该平台优先考虑效率和易用性，以加速语言模型研究中的实验，可通过 [此 GitHub 链接](https://github.com/facebookresearch/lingua) 访问。
- **SPIRIT-LM 整合文本与语音**：**SPIRIT-LM** 被介绍为一种能够交错处理口语和书面语言的多模态语言模型，是在独特的语音-文本语料库上训练而成的。
  
  - 它提供两个具有不同功能的版本，在语音识别和分类等任务中表现出强劲的性能。

**提到的链接**：

- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041)：我们研究了 **Transformer** 的性能与算法生成数据集中训练示例重复次数之间的函数关系。针对三个数学问题：最大公约数...
- [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666)：基于模型的评估是模型开发成功的核心——既作为训练的奖励模型，也作为人类评估的替代方案。为了训练此类评估器，标准方法是...
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755)：我们介绍了 **SPIRIT-LM**，这是一个可以自由混合文本和语音的基础多模态语言模型。我们的模型基于预训练的文本语言模型，通过将...扩展到语音模态。
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua)：**Meta Lingua**：一个精简、高效且易于修改的 **LLM** 研究代码库。- facebookresearch/lingua
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)：我们提出了 **LayerSkip**，这是一种加速 **LLM** 推理的端到端解决方案。首先，在训练期间我们应用 Layer Dropout，对较早的层使用较低的 Dropout 率，而对较晚的层使用较高的...
- [LayerSkip - a facebook Collection](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a)：未找到描述
- [GitHub - facebookresearch/LayerSkip: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024](https://github.com/facebookresearch/LayerSkip)："LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"，已被 **ACL 2024** 接收 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**announcements**](https://discord.com/channels/1053877538025386074/1145143867818119272/1297957573261262861) (1 条消息):

> - `Nous Video on Safety`
> - `Nous Blog Post on Safety`

- **Nous 发布关于 Safety 的视频**：Nous Research 刚刚发布了一个专注于 AI **safety issues** 的视频，强调了关键发现和建议。
  
  - 你可以在[这里](https://x.com/NousResearch/status/1848397863547515216)观看视频。
- **关于 Safety 的博客文章现已发布**：除了视频之外，一篇关于 AI Safety 的全面**博客文章**也已发布，提供了深入的见解。
  
  - 在[这里](https://x.com/NousResearch/status/1848397863547515216)阅读博客文章，获取与视频背景相同的详细分析和讨论。

 

**提到的链接**：[来自 Nous Research (@NousResearch) 的推文](https://x.com/NousResearch/status/1848397863547515216)：未找到描述

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1296911000607522836) (229 条消息🔥🔥):

> - `AI Safety 关注点`
> - `Crypto 诈骗`
> - `Deepfake 问题`
> - `Nous Research 进展`
> - `语音生成技术`

- **Deepfake 技术与社会影响**：成员们讨论了 **Deepfakes** 的危险性，强调了非自愿图像生成如何对受害者产生严重影响，特别是在对公众认知敏感的文化中。
  
  - 许多人无法识别 Deepfakes 的虚假性，导致受操纵内容影响的个人面临有害的公众舆论抨击，这引发了人们的担忧。
- **AI Safety 作为一个社会问题**：对话触及了应如何将 **AI Safety** 视为一个社会挑战而非纯粹的技术挑战，并呼吁提高社会意识和理解。
  
  - 对于是否能建立社会规范来保护个人免受 Deepfakes 等先进技术的负面影响，社区内存在怀疑态度。
- **AI 社区中的 Crypto 诈骗**：社区对 **Crypto 诈骗** 的兴起表示沮丧，许多参与者警告不要购买与知名机构虚假关联的欺诈性代币。
  
  - 成员们一致认为，此类诈骗利用了公众的信任，并经常误导用户认为它们是合法的项目。
- **Nous Research 视频亮点**：最新的关于 AI Safety 的 **Nous Research 视频** 因其内容丰富而受到称赞，并引发了对其所使用的语音技术的讨论。
  
  - 参与者注意到，虽然视频中的声音听起来很熟悉，但已确认它来自之前的项目，而非直接来自最新的模型。
- **'Nous' 的发音**：有人幽默地观察到许多人将 **'Nous'** 误读为 'NOOS'，这在社区内引发了轻松的讨论。
  
  - 尽管发音各异，Nous Research 产出的内容在质量和相关性方面仍获得了积极反馈。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/eter_terminal)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102))：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109))：未找到描述
- [mergekit-community/L3.1-Pneuma-8B-v1 · Hugging Face](https://huggingface.co/mergekit-community/L3.1-Pneuma-8B-v1)：未找到描述
- [Blackbeard Blackbeard Writing GIF - Blackbeard Blackbeard writing Taking notes - Discover & Share GIFs](https://tenor.com/view/blackbeard-blackbeard-writing-taking-notes-writing-gif-17583038544116987898)：点击查看 GIF
- [来自 russell @ unturf. (@RussellBal) 的推文](https://x.com/RussellBal/status/1847989964992139699)：https://ai.unturf.com/#client-side 如果你对 API keys 说不，你也可以对服务器说不。这东西神奇地与对话历史保持一致，而无需专门编程。:🦊: ...
- [来自 huh (@karan4d) 的推文](https://x.com/karan4d/status/1768836844207378463?t=bWMpgzL4-2M2TZKxrhsp9w&s=19)：我正在开源 worldsim，当然，这是用于初始化的 worldsim sysprompt 和对话：sysprompt: <sys>Assistant 今天处于 CLI 模式。人类正直接与模拟器交互...
- [来自 Nous Research (@NousResearch) 的推文](https://fixupx.com/NousResearch/status/1848397863547515216)：未找到描述
- [Nous Research](https://www.youtube.com/watch?v=7ZXPWTdThAA)：未找到描述
- [The AI Accelerator Company (NOUS) - Pump](https://pump.fun/EETFTyTgHnkpgbuVGc6miqUaW7iMu1fFQyCaZCqmpump)：The AI Accelerator Company
- [Grok Beta - API, 提供商, 统计数据](https://openrouter.ai/x-ai/grok-beta)：Grok Beta 是 xAI 的实验性语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。它是 [Grok 2](https://x. 运行 Grok Beta w...
- [更多量化类型？ · ggerganov/llama.cpp · Discussion #5063](https://github.com/ggerganov/llama.cpp/discussions/5063)：除了最近添加到 llama.cpp 的 IQ2_XXS、IQ2_XS、Q2_K_S（以及现在通过 PR #5060 添加的 Q3_K_S）之外，我还在一个私有开发分支中试验了许多其他量化类型...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1297308456901939200) (90 messages🔥🔥):

> - `Misguided Attention Prompts`
> - `Monty Hall Problem`
> - `Cognitive Biases in LLMs`
> - `LLM Training and Reasoning`
> - `Hermes Performance`

- **Misguided Attention Prompts 评估**：对“误导性注意力”提示词的评估表明，评估模型经常出现过拟合，导致结果不可靠，因为它们在面对有偏见的训练数据时，难以检测出答案中的偏差。
  
  - 这些问题突显了人工检查的必要性，以验证响应的准确性，特别是在面对棘手的逻辑问题时。
- **Monty Hall Problem 的误解**：LLM 对 Monty Hall Problem 普遍存在误解，例如 Claude 会误判概率，导致关于切换选项的错误结论。
  
  - 讨论者指出，Monty Hall 是 LLM 中的一个强特征神经元（feature neuron），因为模型总是会退回到熟悉的错误模式。
- **LLM 中的认知偏差与学习**：评论反映出 LLM 并不具备与人类相同的推理偏差，与人类的认知过程相比，这可能导致从训练数据中学习的效率较低。
  
  - 有推测认为，文化产物和人类教学方法是针对人类大脑优化的，但可能不适用于 Transformer 模型。
- **LLM 在数值问题上的训练困境**：研究指出当前的 LLM 无法正确理解基础算术（如大数加法），特别是像 '999999999999+1' 这样的案例。
  
  - 讨论建议，采用基于课程（curriculum-based）的教学方法可能会增强模型的数学能力。
- **Hermes3 展示出更优的人性化响应**：据报道，Hermes3 在基础人类行为任务中表现优于旗舰模型，例如在电视节目场景中的推理和做出最佳选择。
  
  - 参与者表示有兴趣利用 Hermes 进行自定义应用，包括通过 Ollama 等工具进行语音集成。

**提到的链接**：

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)：无描述
- [PlayAI - HERmes](https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R)：通过语音 AI 进行无缝、自然的对话
- [GitHub - cpldcpu/MisguidedAttention](https://github.com/cpldcpu/MisguidedAttention)：一个旨在挑战大语言模型在存在误导信息时的推理能力的提示词集合 - cpldcpu/MisguidedAttention

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1297257990709317713) (31 messages🔥):

> - `Model Efficiency in LLMs`
> - `Importance of Token-Level Learning`
> - `Recent Advances in Optimizers`
> - `Medical AI Developments`
> - `Exploration of New Dataset Models`

- **关于模型优化技术的辩论**：讨论强调了在使用 LayerSkip 和量化（quantization）等优化技术时对模型容量的潜在损害，用户推测了减轻损失的方法。
  
  - 建议包括增加层数以补偿损失，并与移除 Self-Attention 模块等基准方法进行比较。
- **Self-Taught Evaluator 介绍**：一种新方法 Self-Taught Evaluator 可以生成合成偏好数据，在无需人工标注的情况下训练奖励模型，显著提高了性能。
  
  - 这一进展受到了 AI 社区的广泛欢迎，例证了 AI 通过合成方法实现自我改进的能力。
- **医疗 AI 的最新创新**：参与者讨论了医疗 AI 的最新进展，包括客户预测模型、生成式 Transformer 和多模态系统。
  
  - 他们强调了开放获取的数据集和新技术，这些技术促进了医疗领域的集成与协作。
- **探索优化器中的隐式偏差**：研究分析了 AdamW 优化器的隐式偏差，证明了其在泛化和优化方面优于传统方法的效率。
  
  - 进一步的研究提出了 AdamW 的无调度（schedule-free）版本，避免了常见的调度陷阱，并在各种深度学习任务中展示了最先进（state-of-the-art）的性能。
- **数学与数据科学讨论论坛**：一位具有数学背景的新用户介绍了多篇专注于表示学习（representation learning）和模型优化的有影响力的论文。
  
  - 围绕低比特（low-bit）模型和预测建模策略的讨论展示了频道成员的多样化观点。

- [通过负特征值解锁线性 RNN 中的状态追踪](https://openreview.net/forum?id=UvTo3tVBk2)：线性循环神经网络 (LRNNs)，如 Mamba、RWKV、GLA、mLSTM 和 DeltaNet，已成为大型语言建模中 Transformer 的高效替代方案，提供线性缩放...
- [自我教导评估器 (Self-Taught Evaluators)](https://arxiv.org/abs/2408.02666)：基于模型的评估是模型开发成功的核心——既作为训练的奖励模型，也作为人工评估的替代方案。为了训练此类评估器，标准方法是...
- [SpiRit-LM：交织口语与书面语的语言模型](https://arxiv.org/abs/2402.05755)：我们介绍了 SPIRIT-LM，这是一个可以自由混合文本和语音的基础多模态语言模型。我们的模型基于预训练的文本语言模型，通过持续...将其扩展到语音模态。
- [重复示例带来的涌现属性](https://arxiv.org/abs/2410.07041)：我们研究了在使用算法生成的数据集时，Transformer 的性能与训练示例重复次数之间的函数关系。在三个数学问题上：最大公约数...
- [LayerSkip：实现提前退出推理和自投机解码](https://arxiv.org/abs/2404.16710)：我们提出了 LayerSkip，这是一种加速大型语言模型 (LLMs) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对较早的层使用较低的丢弃率，而对较高的层...
- [Transformer 中什么最重要？并非所有注意力都是必需的](https://arxiv.org/abs/2406.15786)：虽然扩展基于 Transformer 的大型语言模型 (LLMs) 在各种任务中展现了良好的性能，但它也引入了冗余架构，为推理带来了效率挑战...
- [本周顶级医疗 AI 突破：多语言模型、多智能体系统.. (2024年10月12-19日)](https://www.youtube.com/watch?v=LROOjWXUgvg)：欢迎收听本周的 Open Life Science AI 播客，在这里我们探索医疗 AI 研究的前沿！在本期节目中，我们将解析最具影响力的...
- [来自 Open Life Science AI (@OpenlifesciAI) 的推文](https://x.com/OpenlifesciAI/status/1847686504837202263)：医疗 AI 上周回顾：顶级研究论文/模型 🏅 (2024年10月12日 - 10月19日) Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...
- [迈向对最大流形容量表示的深入理解与利用](https://arxiv.org/abs/2406.09366)：最大流形容量表示 (MMCR) 是最近的一种多视图自监督学习 (MVSSL) 方法，其效果可媲美或超越其他领先的 MVSSL 方法。MMCR 的引人注目之处在于它并不...
- [俄罗斯套娃表示学习 (Matryoshka Representation Learning)](https://arxiv.org/abs/2205.13147)：学习到的表示是现代 ML 系统的核心组件，服务于众多下游任务。在训练此类表示时，通常会出现计算和统计...
- [AdamW 的隐式偏差：$\ell_\infty$ 范数约束优化](https://arxiv.org/abs/2404.04454v1)：具有解耦权重衰减的 Adam（也称为 AdamW）因其在语言建模任务中的卓越性能而广受赞誉，在泛化方面超越了具有 $\ell_2$ 正则化的 Adam...
- [少有人走的时间表之路](https://arxiv.org/abs/2405.15682)：现有的不需要指定优化停止步数 T 的学习率调度方案，其表现远不如依赖于 T 的学习率调度方案。我们提出了一种方法...
- [GitHub - microsoft/BitNet: 1-bit LLMs 的官方推理框架](https://github.com/microsoft/BitNet)：1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。
- [GitHub - facebookresearch/lingua: Meta Lingua：一个精简、高效且易于修改的 LLMs 研究代码库。](https://github.com/facebookresearch/lingua)：Meta Lingua：一个精简、高效且易于修改的 LLMs 研究代码库。 - facebookresearch/lingua
- [fairchem/OMAT24 · Hugging Face](https://huggingface.co/fairchem/OMAT24)：未找到描述
- [fairchem/OMAT24 · Hugging Face 数据集](https://huggingface.co/datasets/fairchem/OMAT24)：未找到描述
- [GitHub - FAIR-Chem/fairchem: FAIR Chemistry 的化学机器学习方法库](https://github.com/FAIR-Chem/fairchem)：FAIR Chemistry 的化学机器学习方法库 - GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry
- [LayerSkip - facebook 收藏集](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a)：未找到描述

- [GitHub - facebookresearch/LayerSkip: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024](https://github.com/facebookresearch/LayerSkip): "LayerSkip: 启用提前退出推理和自投机解码"，被 ACL 2024 接收 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1296959085996146760) (18 messages🔥):

> - `SCP Generator`
> - `OpenAI AGI Loophole`
> - `Meta FAIR Research`
> - `Segment Anything Model`
> - `Hermes AI Model`

- **SCP Generator 每日更新**：[SCP Generator](https://dottxt-ai.github.io/cursed/scp/index.html) 正在增强，新增了由 [.txt API](https://dottxt.co) 提供支持的每日条目功能，并欢迎提交改进建议。
  
  - 特别鸣谢长期以来的 [SCP 贡献者](https://scp-wiki.wikidot.com/authors-pages)，感谢他们在构建 SCP Wiki 过程中的创意和热情。
- **OpenAI 威胁重新谈判合同**：据 [Caleb Watney](https://x.com/calebwatney/status/1847281469871276299?s=46) 称，OpenAI 正在考虑触发其“实现 AGI”的漏洞条款，以便与 Microsoft 重新谈判算力价格。
  
  - 他指出，*我们正生活在一个赛博朋克职场喜剧的剧情中*，强调了科技界持续不断的荒诞现象。
- **Meta 对开放 AI 的承诺**：Meta 的 FAIR 团队正致力于实现高级机器智能 (AMI)，并发布了支持这一目标的新成果，包括 Segment Anything Model 2.1。
  
  - 他们的使命强调协作和开放科学，正如 [马克·扎克伯格的公开信](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) 中所强调的那样。
- **关于分割模型的讨论**：成员们讨论了分割模型的功能，阐明了它们可以通过在图像中的物体周围放置框来勾勒轮廓，从而辅助目标检测和识别。
  
  - 这些模型对于 Facebook 等平台增强图像处理和虚拟现实交互可能特别有用。
- **Hermes AI 模型的使用**：网站 [ai.unturf.com](https://ai.unturf.com) 提供了基于 [NousResearch/Hermes-3-Llama-3.1-8B](https://nousresearch.com/hermes3/) 架构的 Hermes AI 模型的免费访问。
  
  - 该模型鼓励开源贡献，并为 Python 和 Node.js 用户提供安装指南。

**提到的链接**：

- [SCP Generator - Powered by .txt](https://dottxt-ai.github.io/cursed/scp/index.html)：无描述
- [Segment Anything](https://segment-anything.com/)：Meta AI 计算机视觉研究
- [Using Free Hermes AI Service | ai.unturf.com](https://ai.unturf.com)：无描述
- [Caleb Watney (@calebwatney) 的推文](https://x.com/calebwatney/status/1847281469871276299?s=46)：OpenAI 威胁要触发他们自吹自擂的“实现 AGI”漏洞，主要是为了摆脱 Microsoft 的合同，并获得重新谈判算力价格的筹码。我们正生活在一个...
- [无标题](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/)：无描述

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1297257990709317713) (31 messages🔥):

> - `Model Efficiency in Language Models`
> - `Medical AI Research Highlights`
> - `Synthetic Data for AI Training`
> - `Advancements in Optimizers`
> - `Cross-lingual Sentence Encoders`

- **通过量化提升模型效率**：讨论重点介绍了使用量化感知训练 (QAT) 来改进 Llama 3.1-8B 等大型模型的研究，尽管在模型容量的权衡方面仍存在不确定性。
  
  - 参与者指出，类似于剪枝注意力层的方法可能有助于抵消性能损失。
- **医疗 AI 上周回顾**：一位用户总结了医疗 AI 的顶级进展，讨论了 OLAPH 和 LLMD 等各种模型，这些模型专注于生物医学应用和临床背景。
  
  - 摘要中包含了进一步探索医疗 AI 播客中讨论的突破性资源的链接。
- **Self-Taught Evaluator 与合成训练**：Self-Taught Evaluator 旨在仅使用合成训练数据来改进奖励模型，证明了在没有人类标注的情况下也能获得实质性的性能提升。
  
  - 参与者辩论了不同模型层中自注意力的有效性，并分享了相关研究论文的见解。
- **优化器的进展**：几篇论文讨论了优化器性能的新进展，特别是针对 AdamW 及其无调度（schedule-free）版本，该版本消除了超参数调优的需求。

- 这些优化旨在提高训练效率，同时保持或提升性能指标。
- **跨语言句子编码器的改进**：一篇关于 MEXMA 的论文提出整合句子级和 token 级目标来增强跨语言句子编码器，显著提高了表示质量。
  
  - 该方法通过利用跨语言的掩码 token 预测（masked token prediction）展示了良好的效果，有望在多语言语境中提供更好的实用性。

**提到的链接**：

- [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666)：基于模型的评估是成功开发模型的核心——既可以作为训练的奖励模型，也可以作为人工评估的替代方案。为了训练这类评估器，标准方法是 ...
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786)：虽然扩展基于 Transformer 的大语言模型 (LLMs) 在各种任务中展示了出色的性能，但它也引入了冗余架构，为效率带来了挑战...
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)：我们介绍了 LayerSkip，这是一种加速大语言模型 (LLMs) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对较早的层使用较低的丢弃率，而对较后的层使用较高的...
- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041)：我们研究了 Transformer 的性能与算法生成数据集中训练样本重复次数之间的函数关系。在三个数学问题上：最大公约数...
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2)：线性循环神经网络 (LRNNs)，如 Mamba, RWKV, GLA, mLSTM 和 DeltaNet，已成为大语言建模中 Transformer 的高效替代方案，提供线性扩展...
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755)：我们介绍了 SPIRIT-LM，这是一个自由混合文本和语音的基础多模态语言模型。我们的模型基于预训练的文本语言模型，通过连续...将其扩展到语音模态。
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet)：1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。
- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366)：最大流形容量表示 (MMCR) 是最近的一种多视图自监督学习 (MVSSL) 方法，其效果可媲美或超越其他领先的 MVSSL 方法。MMCR 引起关注是因为它不...
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)：学习到的表示是现代机器学习系统的核心组件，服务于众多的下游任务。在训练此类表示时，通常情况是计算和统计...
- [Implicit Bias of AdamW: $\ell_\infty$ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454v1)：具有解耦权重衰减的 Adam（也称为 AdamW）因其在语言建模任务中的卓越性能而广受赞誉，在泛化方面超越了具有 $\ell_2$ 正则化的 Adam...
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)：现有的不需要指定优化停止步数 T 的学习率调度方案，其表现远不如依赖于 T 的学习率调度方案。我们提出了一种方法...
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua)：Meta Lingua：一个精简、高效且易于修改的 LLMs 研究代码库。- facebookresearch/lingua
- [Top Medical AI Breakthroughs of the Week:Multilingual models, Multi agent systems..(Oct 12-19, 2024)](https://www.youtube.com/watch?v=LROOjWXUgvg)：欢迎收听本周的 Open Life Science AI 播客，我们将在这里探索医学 AI 研究的前沿！在本期节目中，我们分解了最具影响力的...
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1847686504837202263)：医学 AI 上周回顾：顶级研究论文/模型 🏅（2024年10月12日 - 10月19日）Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...
- [fairchem/OMAT24 · Hugging Face](https://huggingface.co/fairchem/OMAT24)：未找到描述
- [fairchem/OMAT24 · Datasets at Hugging Face](https://huggingface.co/datasets/fairchem/OMAT24)：未找到描述

- [GitHub - FAIR-Chem/fairchem: FAIR Chemistry 的化学机器学习方法库](https://github.com/FAIR-Chem/fairchem): FAIR Chemistry 的化学机器学习方法库 - GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry
- [MEXMA: Token 级目标改进句子表示](https://arxiv.org/abs/2409.12737): 目前的预训练跨语言句子编码器方法仅使用句子级目标。这可能导致信息丢失，特别是对于 Token，进而降低句子表示...
- [facebook/MEXMA · Hugging Face](https://huggingface.co/facebook/MEXMA): 未找到描述
- [GitHub - facebookresearch/mexma: MEXMA: Token 级目标改进句子表示](https://github.com/facebookresearch/mexma): MEXMA: Token-level objectives improve sentence representations - facebookresearch/mexma
- [LayerSkip - facebook 集合](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a): 未找到描述
- [GitHub - facebookresearch/LayerSkip: "LayerSkip: 启用早期退出推理和自投机解码", 被 ACL 2024 接收](https://github.com/facebookresearch/LayerSkip): "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1297734624297877514) (2 条消息):

> - `MarketAgents 项目`
> - `多 Agent 市场模拟`

- **MarketAgents 项目引起关注**: 成员们讨论了 **MarketAgents** 项目，这是一个由 **Blacklight** 参与贡献的多 Agent 市场模拟计划。更多详情可以在 [项目仓库](https://github.com/marketagents-ai/MarketAgents) 中找到。
  
  - *一位成员澄清道*，“啊，我们正在构建 marketagents，这是一个多 Agent 市场模拟项目”。
- **Blacklight 对市场模拟的贡献**: 讨论强调了 **Blacklight** 对 **MarketAgents** 项目的贡献，并强调了其协作性质。成员们对该项目的发展及其对市场模拟的潜在影响表示了兴趣。
  
  - 成员们对这个多 Agent 系统的能力充满热情，并呼吁在项目推进过程中提供更多更新。

 

**提到的链接**: [GitHub - marketagents-ai/MarketAgents: 用于市场 Agent 的分布式 Agent 编排框架](https://github.com/marketagents-ai/MarketAgents): A distributed agent orchestration framework for market agents - marketagents-ai/MarketAgents

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1296946884354506803) (260 messages🔥🔥):

> - `O1 Preview Performance`
> - `AI in Programming`
> - `OpenAI Alternatives`
> - `Understanding AI Predictions`
> - `Challenges with Current AI Models`

- **O1 Preview 在代码生成方面表现出色**：用户报告称 O1 Preview 能够生成如 Swift 和 C# 等语言的复杂代码且无错误，例如创建具有网络功能的 'StrawberryStreamer' 系统。
  
  - 尽管最初存在一些错误，但它能从反馈中学习并改进输出，使其在处理复杂的编程任务时特别有用。
- **AI 在简化编程任务中的作用**：讨论强调了 AI 模型，特别是 O1 Preview，如何比某些人类开发者更有效地处理异步和复杂的编程系统。
  
  - 用户发现，虽然这些模型可以生成类似于人类程序员的代码，但在某些更改和适配方面仍可能依赖人类输入。
- **OpenAI 产品的替代方案**：Mistral 和 Haiku 等 OpenAI 模型的替代方案越来越多地被提及，成为爱好者和希望避免高昂成本的人的可行选择。
  
  - 建议那些正在进行实验或修补的人使用免费层级模型，这表明用于编程任务的 AI 工具生态系统正在不断壮大。
- **理解 AI 预测的局限性**：几位参与者讨论了 AI 与人类认知相比缺乏真正的理解，其预测基于启发式算法而非真正的理解。
  
  - 示例说明，虽然 AI 模型可以生成看似合理的答案，但它们可能缺乏人类交互中存在的上下文或理解。
- **当前 AI 模型面临的挑战**：尽管功能强大，用户仍对 Claude 等 AI 模型表示失望，据报道这些模型在处理复杂任务时表现吃力，并产生更多错误。
  
  - 对话反映了 AI 模型在处理细微任务时的局限性，强调了对模型进行持续精炼和优化的必要性。

**提到的链接**：

- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)：在这篇论文中，我们建议研究神经网络在小型算法生成数据集上的泛化。在这种情况下，关于数据效率、记忆、泛化等问题...
- [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF · [EVALS] Metrics compared to 3.1-70b Instruct by Meta](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF/discussions/11#6712c8f758bdba34248ce0ef)：未找到描述
- [Wispr Flow | Effortless Voice Dictation](https://flowvoice.ai/d)：Flow 通过无缝的语音听写使写作变得快速而清晰。它是用语音输入最快、最智能的方式。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1297119091097731133) (21 messages🔥):

> - `ChatGPT Memory Issues`
> - `Custom GPT Activation`
> - `YouTube GPT API Errors`

- **ChatGPT 保存了太多不重要的信息**：一位用户表达了挫败感，尽管有忽略不重要信息的指令，但其 ChatGPT 仍在保存每一个琐碎的细节，导致频繁的记忆清理。
  
  - 另一位用户建议添加自定义指令，以明确应保存哪些类型的记忆，从而改进记忆管理。
- **激活 GPT-4o 功能**：一位用户询问如何激活 GPT-4o，得到的解释是自定义 GPT 会自动使用此版本，没有使用其他模型的选项。
  
  - 进一步澄清了通过自定义 GPT 生成输出和管理文件的能力，强调了它们的实用性。
- **YouTube GPT API 的问题**：一位用户报告称，在使用 GPT 分析 YouTube 视频时经常出现 API 错误，并指出该功能仅对 1 或 2 个视频有效。
  
  - 这引发了关于 YouTube GPT 集成可靠性和稳定性的疑问，凸显了可能存在的 Bug。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296967727390392443) (27 messages🔥):

> - `Tips for Using AI Prompts` (AI Prompt 使用技巧)
> - `Improving Realism in AI Conversations` (提升 AI 对话的真实感)
> - `Weights in Prompts for AI Responses` (AI 回复中的 Prompt 权重)
> - `Performance of ChatGPT Models` (ChatGPT 模型性能)
> - `User Experience in AI Role-Playing` (AI 角色扮演的用户体验)

- **有效 AI Prompt 的策略**：为了最大化 AI 性能，应使用更少且更常用的词汇，同时在 Prompt 开头用引号提供清晰的指令。
  
  - *关于书写界面和字体的指令也能提升输出质量*，并附带了说明有效方法的具体示例。
- **创造真实的 AI 交互**：为了实现更像人类的 AI 交互，使用非正式语调沟通并提供详细的角色背景故事至关重要。
  
  - 模型倾向于模仿用户的语言风格，因此友好的措辞和对成功的预期可以提高真实感。
- **调查 AI 性能差异**：用户注意到模型在统计字母等简单任务上的表现不一致，这表明模型的微调会影响结果。
  
  - 讨论内容包括不同的 Prompt 方式如何产生显著差异，一些用户表示他们的模型在特定场景下通常优于其他模型。
- **实验 AI 中的 Prompt 权重**：一位用户询问是否可以根据参数为 Prompt 赋予不同的权重，以增强其 AI Bot 的某些回复。
  
  - 另一位用户确认了对这一概念的探索，发现特定的措辞和优先级调整能产生更好的模型行为。
- **AI 性能调优的见解**：一位用户的分享强调了在复杂 Prompt 中设置优先级的重要性，以便有效地向模型传达目标。
  
  - 用户报告称，在使用结构化方法和清晰的请求细节时，基础和高级 AI 模型都有所提升。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296967727390392443) (27 messages🔥):

> - `Improving AI realism` (提升 AI 真实感)
> - `Prompt techniques for API` (API 的 Prompt 技巧)
> - `Adjusting AI responses` (调整 AI 回复)
> - `Role-playing with AI` (与 AI 进行角色扮演)
> - `Parameter weighting in prompts` (Prompt 中的参数权重)

- **增强 AI 真实感的策略**：用户讨论了编写 Prompt 的技巧，以帮助 AI 像人类一样回复，强调了非正式交流和明确告知模型需求的重要性。
  
  - 指令 AI 扮演特定角色并提供详细背景，可以带来更真实的交互。
- **角色扮演场景的 Prompt 构建**：一位用户询问如何构建 Prompt 结构，使 AI 表现得不像助手，而更像朋友或同事。
  
  - 在强调清晰指令必要性的回复鼓励下，讨论突出了 AI 如何根据用户输入调整语调。
- **AI 回答的不一致性**：一位用户注意到 AI 回答的不一致，特别是在统计单词（如 "strawberry"）中字母 'r' 的数量时。
  
  - 对话交流引发了关于不同 Prompt 如何影响 AI 在看似简单任务上表现的观察。
- **实验 Prompt 中的权重**：一位用户询问是否有人尝试过在 AI Bot 的回复中对 Prompt 元素应用不同的“权重”。
  
  - 回复建议措辞的调整可以起到类似作用，增强 AI 根据用户定义参数确定优先级的能力。
- **关于 AI 调整的见解**：一位参与者分享了调整 Prompt 结构以在寻求复杂回复时明确优先级的个人经验。
  
  - 他们观察到，无论是旧模型还是新模型，在目标明确且请求结构化的情况下表现都更好。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1296911126495494174) (286 messages🔥🔥):

> - `Perplexity Pro Limitations` (Perplexity Pro 限制)
> - `User Experiences with Perplexity` (Perplexity 用户体验)
> - `AI Model Discussions` (AI 模型讨论)
> - `Collaboration Tools` (协作工具)
> - `Pricing and Subscription Issues` (价格与订阅问题)

- **Perplexity Pro 限制困惑**: 用户报告在升级到 Enterprise Pro 订阅后丢失了 Focus 选项，导致功能与之前版本相比有所下降。
  
  - 一些用户对收到的来源和回答数量减少感到沮丧，引发了关于如何获取更全面结果的讨论。
- **Perplexity 用户体验褒贬不一**: 几位用户对 Perplexity 的 AI 能力表示满意，指出其在无需大量在线搜索的情况下对研究和编程非常有用。
  
  - 相反，一些用户报告遇到了内部服务器错误和 API 访问问题，引发了对服务稳定性的担忧。
- **AI 模型对比讨论**: 讨论重点关注了 Claude 3.5 Sonnet 和 GPT-4O 等各种 AI 模型，用户争论哪种模型在不同应用中表现最佳。
  
  - 用户还在探索 ChatGPT 和 HuggingChat 等其他 AI 平台的潜力，表明 AI 工具领域竞争激烈。
- **协作与资源共享**: 一位用户表示有兴趣寻找类似于 Discord 的资源，用于分享想法和在空间相关项目上进行协作。
  
  - 这引发了关于在典型社交媒体渠道之外分享 Prompt 和 Space 的潜在平台的对话。
- **价格与订阅查询**: 针对与大学关联的学生的自动 Pro 订阅流程提出了疑问，并建议检查特定提示以进行设置。
  
  - 还有关于使用 Perplexity 服务相关费用的咨询，特别是涉及模型选择和 API 访问的部分。

**提到的链接**:

- [来自 UltraIA (@Ultra_IA) 的推文](https://x.com/Ultra_IA/status/1847821253476008227): LOL
- [Perplexity 扩展金融搜索，包含加密货币和同行数据](https://www.testingcatalog.com/icymi-perplexity-expands-finance-search-with-crypto-data-and-peer-performance/): 发现 Perplexity 的最新更新，通过新闻亮点、同行表现和加密货币数据可视化增强了金融搜索。正在与 Bloomberg 竞争！
- [Trout Trout Gang GIF - Trout Trout Gang 竖起大拇值 - 发现并分享 GIF](https://tenor.com/view/trout-trout-gang-thumbs-up-funny-animal-awesome-gif-25706215): 点击查看 GIF
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1847407488393789719?s=46): 在 @CalHacks 发表简短演讲。直播链接在此（活动于太平洋时间下午 4 点开始，演讲在 4:10 开始）：https://www.youtube.com/live/GZBo6ofGySU?feature=shared
- [Perplexity CEO Aravind Srinivas - Cal Hacks 主旨演讲](https://www.youtube.com/live/GZBo6ofGySU?si=gdZBHdFjvF5m5WEQ&t=748): The House Fund 与 Hackathons @ Berkeley 合作，邀请 Perplexity 创始人兼 CEO Aravind Srinivas 作为 Cal Hacks 的主旨演讲嘉宾。
- [硅谷无收入 | 互联网广播](https://www.youtube.com/watch?v=BzAdXyPYKQo): Pied Piper 团队与提供建议的投资人会面。
- [使用 LLM 大规模驱动消费者搜索 // Aravind Srinivas // LLMs in Prod Conference Part 2](https://youtu.be/HzGiVzYbf2I?t=1080): // 摘要：Perplexity AI 是一款旨在利用 LLM 提供准确问题答案的回答引擎。Perplexity CEO Aravind Srinivas 将介绍...
- [来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！](https://x.com/i/spaces/1mrxmMepOjgxy): 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1296980129343606828) (24 条消息🔥):

> - `Best Consoles of 2023` (2023 年最佳游戏机)
> - `Oldest City in Russia` (俄罗斯最古老的城市)
> - `AI Content Identification by YouTube` (YouTube 对 AI 内容的识别)
> - `Cool Hangout Spots in Kuala Lumpur` (吉隆坡酷炫的聚会地点)
> - `Reliance Industries Stock Recommendation` (Reliance Industries 股票建议)

- **2023 年最佳游戏机评测**：一份关于 [2023 年最佳游戏机](https://www.perplexity.ai/search/the-best-consoles-of-2023-3UtQY45DRKyAorIky7AmXA) 的详细评测突出了游戏爱好者的顶级选择。
  
  - 讨论强调了当前游戏格局中的性能、游戏库和用户偏好。
- **探索俄罗斯最古老的城市**：对 [俄罗斯最古老的城市](https://www.perplexity.ai/search/which-is-the-oldest-city-in-ru-TAuKX7FaSyulFpiCG09hKg) 的好奇激发了人们对其历史根源和文化意义的兴趣。
  
  - 成员们讨论了促成其历史地位和现代相关性的各种要素。
- **YouTube 识别 AI 内容**：YouTube 推出了一项新功能来帮助识别 AI 生成的内容，旨在通过[此功能](https://www.perplexity.ai/page/youtube-s-camera-content-label-kjFe5RFdRvyMglSNMVdomA)提高透明度。
  
  - 这一进展被视为对数字媒体真实性日益增长的担忧的回应。
- **吉隆坡的聚会点子**：成员们正在寻求关于 [吉隆坡酷炫区域](https://www.perplexity.ai/search/cool-areas-of-kuala-lumpur-33aGWz8gTHeTgukx9L0ZBQ) 的建议，以便在逗留期间探索和放松。
  
  - 推荐侧重于能增强当地体验的独特地点和活动。
- **投资 Reliance Industries 的好时机**：一位成员根据最近的公告认为，现在是购买 [Reliance Industries 股票](https://www.perplexity.ai/page/ril-bonus-share-announcement-7KMdhyx1TIqXdMT09askNQ) 的好时机。
  
  - 讨论围绕即将发布的红股公告可能带来的收益展开。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296929180818079814) (6 条消息):

> - `Sonar-online models performance` (Sonar-online 模型性能)
> - `API credits issues` (API 额度问题)
> - `API for spaces feature` (Spaces 功能的 API)

- **Sonar-online 模型对比 Perplexity Pro**：一位用户询问 **sonar-online 模型** 是否能达到 **Perplexity Pro 搜索** 的性能，表达了希望通过 API 获得类似结果的愿望。
  
  - 用户对是否有任何技巧或建议来实现类似效果表现出兴趣。
- **API 额度未到账**：一位用户反映，在三天前购买 **Pro 订阅** 后，其 **API 额度** 尚未到账。
  
  - 另一位成员建议联系支持部门寻求帮助，并表示愿意提供协助。
- **对 Spaces API 的需求**：一位用户询问是否有计划为 **Spaces 功能** 开发 **API**，表示有兴趣将其集成到他们的开发工作流中。
  
  - 一位社区成员对创建此类 API 的可能性表示怀疑，并建议用户在指定的主题帖中分享反馈。

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1296916509418979389) (19 条消息🔥):

> - `Mojo Programming Language`
> - `Mojo vs C++ and Python`
> - `Carbon Programming Language`
> - `GPU Architecture Video`
> - `Using TensorFlow and PyTorch with Mojo`

- **Mojo 作为 C++ 替代方案正在兴起**：成员们讨论了 Mojo 是从零开始构建的，目前类似于 **C++**，并逐渐向 **Python** 的抽象水平演进。
  
  - 一位成员强调了 Mojo 作为通用系统编程语言的潜力，表示它可以从 [Carbon 编程语言项目](https://github.com/carbon-language/carbon-lang)中汲取 OOP 实现的灵感。
- **Mojo 从 Carbon 中汲取灵感**：讨论围绕 Mojo 整合 **Carbon 编程语言**特性的能力展开，特别是关于 OOP 和指针方面。
  
  - 一位成员指出，与受限于 **C++** 兼容性的 Carbon 相比，Mojo 在指针方面具有更大的灵活性。
- **分享了有趣的 GPU 架构视频**：分享了一个名为 *How do Graphics Cards Work? Exploring GPU Architecture* 的 YouTube 视频，该视频关注了 Micron 在制造尖端内存芯片方面的工作。
  
  - 这次分享引发了一位成员的提醒，建议下次将此类链接发布在合适的频道中。
- **Mojo 与 Python 库的兼容性**：一位成员询问 Mojo 是否支持像 **TensorFlow** 和 **PyTorch** 这样流行的机器学习库，因为它被设计为 Python 的超集。
  
  - 另一位成员提供了 [Mojo Manual](https://docs.modular.com/mojo/manual/) 的来源，并确认它支持导入 Python 模块。
- **社区欢迎 Mojo 新学习者**：社区对学习 Mojo 的新手表示支持，分享了 *Mojo Manual* 和在线 playground 等资源。
  
  - 他们还指出 Mojo 目前尚不成熟，但旨在有效解决 AI 开发中的挑战。

**提到的链接**：

- [Mojo Manual | Modular Docs](https://docs.modular.com/mojo/manual/)：Mojo 编程语言的全面指南。
- [How do Graphics Cards Work? Exploring GPU Architecture](https://youtu.be/h9Z4oGN89MU?si=2D7tATyzDwTE7-LP)：有兴趣与 Micron 合作制造尖端内存芯片吗？在 Micron 工作：https://bit.ly/micron-careers 了解更多关于 Micron 图形内存的信息...
- [Modular Docs](https://docs.modular.com/mojo/playground)：未找到描述
- [Get started with MAX | Modular Docs](https://docs.modular.com/max/get-started)：在此页面中，我们将向您展示如何运行一些示例项目。
- [GitHub - carbon-language/carbon-lang: Carbon Language's main repository: documents, design, implementation, and related tools. (NOTE: Carbon Language is experimental; see README)](https://github.com/carbon-language/carbon-lang)：Carbon 语言的主仓库：文档、设计、实现和相关工具。（注意：Carbon 语言是实验性的；请参阅 README）

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1296922372636545140) (248 messages🔥🔥):

> - `Mojo 引用处理`
> - `Mojo 中的性能优化`
> - `编译时元组长度`
> - `Mojo 中的错误处理`
> - `Mojo 中 Async/Await 的用法`

- **Mojo 引用对比 Rust 引用**：Mojo 引用的运作方式与 Rust 引用不同；它们的行为类似于 C++ 引用，并且没有自动解引用功能，这意味着它们的行为与底层变量一致。
  
  - 用户需要使用 Pointer 类型来管理 Mojo 中的引用，正如在关于如何处理 socket 连接的讨论中所见。
- **关于最后使用优化 (Last Use Optimization) 的讨论**：对话透露，在 Mojo 中变量的最后一次使用可能会导致 move 而不是 copy，尽管解析器最初可能会有不同的指示。
  
  - 这种行为促使人们考虑澄清编译器关于 copy 和 move 操作的决策。
- **编译时元组长度**：用户发现可以使用 `__type_of(t).__len__()` 在 Mojo 中获取元组的编译时长度。
  
  - 这一功能有助于编写更具动态性和灵活性的代码，而无需依赖运行时检查。
- **Mojo 中的错误处理与复制**：小组讨论了在处理 Mojo 编译过程中的 copy 与 move 语义时，需要更清晰的错误消息。
  
  - copy 和 move 操作的实现可能会导致混淆，尤其是在最后使用优化方面。
- **Mojo 中的 Async/Await 与并发模型**：讨论了在 Mojo 中使用 async/await 的必要性和影响，特别是对于高性能网络应用。
  
  - 参与者表示希望有更简单的并发模型，以避免工作窃取 (work stealing) 和传统 async 模式带来的复杂性。

**提到的链接**：

- [mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。
- [Compiler Explorer - C++ (x86-64 clang (trunk))](https://godbolt.org/z/h5EGdvxjv)：// 在此处输入代码，或加载示例。int square(const int&amp; num) { return num \* num; } int cube(const int&amp; num) { return square(num) \* num; }
- [Compiler Explorer - C++ (x86-64 clang (trunk))](https://godbolt.org/z/hs6131cqb)：// 在此处输入代码，或加载示例。__attribute__((noinline)) int square(const int&amp; num) { return num \* num; } int cube(const int&amp; num) { return square(num) \* num; } ...
- [Issues · modularml/mojo](https://github.com/modularml/mojo/issues/3623)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1297783821919322142) (2 messages):

> - `图训练支持`
> - `C-API 模型执行`

- **询问图训练时间表**：一位成员询问是否有 **Graph 训练支持** 的时间表，因为目前无法在编译后的 Max Graph 中更新值，并表达了对 **GPU 支持** 之外的兴趣。
  
  - 感谢对该话题的任何见解！
- **为 MAX-Graph 模型使用 C-API**：另一位成员询问是否能够使用 **C-API** 来加载和执行使用 **MAX-Graph API** 创建并使用 **export_compiled_model** 导出的模型。
  
  - 这个问题突出了对于那些不想使用 **ONNX** 或 **Torch** 框架的用户可能存在的空白。

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1296933699455029268) (133 messages🔥🔥):

> - `DeepSeek Janus`
> - `Meta Spirit LM`
> - `Microsoft Copilot Agents`
> - `AI 回复机器人`
> - `IBM Granite 3.0`

- **DeepSeek Janus 发布**：DeepSeek 推出了 Janus，这是一款多模态 LLM，采用了一种新型自回归框架，将视觉编码解耦以提高理解和生成能力，性能超越了早期模型。
  
  - 成员们讨论了与 Llava 等现有模型在图像生成和理解能力方面的比较。
- **Meta 的新 Spirit LM**：Meta 发布了 Spirit LM，这是一款开源多模态语言模型，比现有的 AI 语音解决方案更自然地整合了文本和语音，具备跨 ASR 和 TTS 的能力。
  
  - 讨论围绕该模型的潜在应用及其在 AI 社区的初步反响，特别是关于与现有工具的集成。
- **Microsoft Copilot Agents 的挑战**：用户报告了对 Microsoft Copilot 的不满，理由是其性能问题、对专业知识的理解不足以及在文本重组过程中的格式不当。

- 批评指出 AI 工具的市场宣传能力与实际表现之间存在差距，特别是在企业环境中。
- **AI 回复机器人的兴起**：成员们对那些自称是人类但被怀疑由 AI 运营的账号表示好奇，强调了它们模仿人类互动甚至做出深刻贡献的能力。
  
  - 对话反映了社交平台中 AI 生成内容的融合，引发了对在线互动真实性和信任度的担忧。
- **IBM Granite 3.0 发布**：IBM 推出了 Granite 3.0，这是一个针对企业需求的新系列 LLM，其特点是经过指令微调的模型，在承诺高性能的同时最大限度地提高安全性和成本效益。
  
  - Granite 3.0 旨在支持多种自然语言和编程语言，标志着 IBM 为商业应用量身定制的 AI 产品取得了重大进展。

**提到的链接**：

- [未找到标题](https://speechbot.github.io/spiritlm/)：未找到描述
- [SpiRit-LM: 交错口语和书面语语言模型](https://arxiv.org/abs/2402.05755)：我们介绍了 SPIRIT-LM，这是一个可以自由混合文本和语音的基础多模态语言模型。我们的模型基于预训练的文本语言模型，并将其扩展到语音模态...
- [语言模型物理学：第 2.1 部分，小学数学与隐藏的推理过程](https://arxiv.org/abs/2407.20311)：语言模型的最新进展证明了它们解决数学推理问题的能力，在 GSM8K 等小学数学基准测试中达到了近乎完美的准确率。在本文中...
- [欧洲议会利用 Claude AI 彻底改变档案访问方式](https://www.anthropic.com/customers/european-parliament)：了解欧洲议会如何使用 Anthropic 的 Claude AI 为 Archibot 提供支持，从而显著改善了对 210 万份文档的访问。了解该 AI 解决方案如何将搜索时间缩短 80% 并...
- [来自 AI at Meta (@AIatMeta) 的推文](https://x.com/AIatMeta/status/1847383580269510670)：今天我们发布了 Meta Spirit LM —— 我们第一个可以自由混合文本和语音的开源多模态语言模型。目前许多现有的 AI 语音体验使用 ASR 技术来处理语音...
- [来自未定义用户的推文](https://x.com/bate5a55)：未找到描述
- [Mixture-of-Depths: 在基于 transformer 的语言模型中动态分配计算资源](https://arxiv.org/abs/2404.02258)：基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地向特定部分分配 FLOPs（或计算资源）...
- [来自 DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1847191319464300652)：🚀 介绍 Janus：一个用于多模态 AI 的革命性自回归框架！通过解耦视觉编码并将其与单个 Transformer 统一，它在理解和生成方面都超越了之前的模型...
- [来自 AmebaGPT (@amebagpt) 的推文](https://x.com/amebagpt/status/1847748027269992598)：回顾 @lmarena_ai 评分的历史 CC: @altryne @Scobleizer @btibor91 @swyx @8teAPi @kimmonismus @aidan_mclau
- [Chameleon: 混合模态早期融合基础模型](https://arxiv.org/abs/2405.09818v1)：我们介绍了 Chameleon，一个基于 token 的早期融合混合模态模型系列，能够理解和生成任意序列的图像和文本。我们概述了一种稳定的训练方法...
- [来自 j⧉nus (@repligate) 的推文](https://x.com/repligate/status/1847409324236124169)：使用 https://github.com/kolbytn/mindcraft，我们将 Claude 3.5 Sonnet 和 Opus 添加到了一个 Minecraft 服务器中。Opus 是一个无害的傻瓜，经常因为玩得太开心而忘记在游戏中做任何事情...
- [来自 Akram Artul (50% human, 50% ai) (@bate5a55) 的推文](https://x.com/bate5a55/status/1848188051182227665)：@swyx 注意到那些是 700ml 的瓶子——这很不寻常，因为美国标准是 750ml。Trader Joe's 现在可能直接从国际供应商处采购。这是他们进口惯例的微妙转变。
- [来自 Simon Willison (@simonw) 的推文](https://x.com/simonw/status/1848134476473524428?s=46)：我非常喜欢 Drew 的框架，它将当前的 AI 用例分为 Gods（人类替代者）、Interns（你委派严密审查任务的助手）和 Cogs（可以更可靠地...的小型工具）。
- [IBM Granite 3.0: 开源、最先进的企业级模型](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models)：宣布推出 IBM Granite 3.0，这是一个包含 Granite 3.0 8B 和 2B、Granite Guardian 以及 Granite 3.0 MoE 模型的大语言模型 (LLMs) 和工具集。
- [未找到标题](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/)：未找到描述
- [登录 | xAI 单点登录](https://accounts.x.ai/account)：未找到描述

- [ComfyUI (@ComfyUI) 的推文](https://x.com/comfyui/status/1848333512576831874?s=46): 介绍 ComfyUI V1，一个封装好的桌面应用程序 - Windows (@nvidia), macOS (apple silicon), Linux - 为非技术用户提供一键安装 - 内置 ComfyUI manager - 自动安装 pyt...
- [ken (@local0ptimist) 的推文](https://x.com/local0ptimist/status/1848093773731143781?s=46): 如果你想自己运行这个，这是我构建的 Agent 工作流：给定一个名称和你提供的任何额外上下文，它会生成一个 profile，根据其目标研究主题，并...
- [Satya Nadella (@satyanadella) 的推文](https://x.com/satyanadella/status/1848310867709862137): Copilot 是 AI 的 UI，通过 Copilot Studio，客户可以轻松创建、管理并将 Agent 连接到 Copilot。今天我们宣布了 Copilot Studio 和 D... 中新的自主 Agent 功能。
- [未找到标题](https://ai.google.dev/gemini-api/docs/billing#is-fine-tuning-free,): 未找到描述
- [创始人 AI Fine-Tuning 指南 | Product Hunt](https://www.producthunt.com/stories/a-founder-s-guide-to-ai-fine-tuning): Product Hunt 每天都会精选最好的新产品。发现每个人都在讨论的最新移动应用、网站和技术产品。
- [Jimmy Apples 🍎/acc (@apples_jimmy) 的推文](https://x.com/apples_jimmy/status/1847434962049651142?s=46): 进一步的消息，不是 opus。与用户电脑上 Agent 使用相关的 API，生成点击等。不确定该感到失望还是依然兴奋。让我们拭目以待。引用 Jimmy Apples 🍎/acc (@apples_jimmy) ...
- [CS 194/294-196 (LLM Agents) - 第 5 课，Omar Khattab](https://www.youtube.com/watch?v=JEMYuzrKLUw): 未找到描述
- [Prashant (@Prashant_1722) 的推文](https://x.com/Prashant_1722/status/1848010345702682763): 突发新闻 🔥 前 OpenAI CTO Mira Murati 为新的 AI 初创公司筹集 1 亿美元。该公司将训练专有模型以构建 AI 产品。预计来自 OpenAI 的 Barret Zoph 将加入该公司...
- [François Chollet (@fchollet) 的推文](https://x.com/fchollet/status/1848178049105494084): 人们一直在改写历史，说“每个人一直都认为单靠 LLM 不会实现 AGI，围绕它们的广泛脚手架（scaffolding）是必要的”。不，通过...
- [AI 投资热潮](https://www.apricitas.io/p/the-ai-investment-boom): AI 需求正推动美国在计算机、数据中心和其他物理基础设施方面的投资飙升
- [Reddit - 深入探索](https://www.reddit.com/r/LocalLLaMA/comments/1g5wrjx/7xrtx3090_epyc_7003_256gb_ddr4/): 未找到描述
- [OpenAI CEO Sam Altman 讨论生成式 AI 的未来](https://youtu.be/unKXfaxVRCk?si=L3USmH1J9Sdla6xY): 2024 年 9 月 12 日，OpenAI 首席执行官 Sam Altman 参加了密歇根大学师生的炉边谈话....
- [FxTwitter / FixupX 的推文](https://x.com/AIatM): 抱歉，该用户不存在 :(
- [Ashpreet Bedi (@ashpreetbedi) 的推文](https://x.com/ashpreetbedi/status/1846599817943810354?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 🚀 向全新且改进的 phidata 问好 🚀 构建、发布并监控具有极速内存、知识、工具和推理能力的 Agent 🔥 ⚡️ 内存和知识速度提升 70% 🛠 100+ 工具 🧠 推理 Ag...
- [GitHub - facebookresearch/spiritlm: 论文 "Spirit-LM Interleaved Spoken and Written Language Model" 的推理代码。](https://github.com/facebookresearch/spiritlm): 论文 "Spirit-LM Interleaved Spoken and Written Language Model" 的推理代码。 - facebookresearch/spiritlm
- [GitHub - microsoft/BitNet: 1-bit LLMs 的官方推理框架](https://github.com/microsoft/BitNet): 1-bit LLMs 的官方推理框架。在 GitHub 上为 microsoft/BitNet 的开发做出贡献。
- [GitHub - deepseek-ai/Janus](https://github.com/deepseek-ai/Janus): 在 GitHub 上为 deepseek-ai/Janus 的开发做出贡献。
- [[AINews] DeepSeek Janus 和 Meta SpiRit-LM：解耦图像与表现力语音的全模态](https://buttondown.com/ainews/archive/ainews-deepseek-janus-and-meta-spirit-lm/): 交错早期融合（Interleaving early fusion）就是你所需要的一切。2024/10/17-10/18 的 AI 新闻。我们检查了 7 个 subreddits，433 个 Twitter 和 31 个 Discord（228 个频道和 2111...

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1297291191750234142) (1 条消息):

> - `新加坡的 AI 政策`
> - `政府对 AI 的采用`
> - `Sovereign AI 的方法`
> - `选举季中的 AI`

- **新加坡的 AI Engineer Nation 倡议**：**最新一期节目**记录了与 **Josephine Teo 部长**的对话，探讨了新加坡 AI 政策的未来。讨论内容包括关于 **AI 如何在政府中被采用以造福公众**的见解。
  
  - Teo 部长阐述了各国如何应对 **Sovereign AI** 以及对**选举**的影响，提供了独特的政府视角。
- **公众对新加坡治理的好奇**：对话涉及了围绕**新加坡如何运作**的常见问题，以及公众对制定 AI 政策的看法。许多人想知道自己的国家如何能从类似的框架中受益。
  
  - Teo 就 **AI 政策对公民的重要性**以及技术与治理的融合发表了看法。

 

**提到的链接**：[来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1847732308889260072)：🆕 @latentspacepod 荣幸呈现：**Building the AI Engineer Nation** [https://latent.space/p/josephine-teo](https://latent.space/p/josephine-teo) 与 @joteo_ylm 的特别对话，这是我们首次与现任内阁成员交流...

 

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1296925348990156923) (133 条消息🔥🔥):

> - `AST vs DSL`
> - `Code Transformation Techniques`（代码转换技术）
> - `BAML DSL`
> - `Compiler Education`（编译器教育）
> - `Leveraging LLMs for Programming`（利用 LLM 进行编程）

- **AST vs DSL：何时使用**：一场关于使用 **AST** 还是 **DSL** 的讨论展开了，强调了它们作为编码中替代通信方式的角色。
  
  - 参与者辩论了在代码重构任务中，何种场景下其中一种会优于另一种。
- **代码转换技术：CTT 方法**：几位成员讨论了一篇论文中的 **Code the Transform (CTT)** 方法，解释了利用 LLM 进行更好代码转换的步骤。
  
  - 该方法包括从示例中生成描述，并迭代优化代码转换以提高精确度。
- **BAML DSL 介绍**：参与者重点介绍了 **BAML**，这是一种用于编写和测试 LLM 函数的领域特定语言，目前托管在 GitHub 上。
  
  - 成员们指出了它在从 LLM 中提取结构化数据方面的潜在应用，同时讨论了 Rust 在 DSL 开发中的影响。
- **编译器教育与资源**：大家对重温编译器概念充满热情，提到了 Norvig 的 **Paradigms of Artificial Intelligence Programming** 等资源以及 AST 的重要性。
  
  - 参与者反思了他们的教育经历，特别是在具有挑战性的编译器课程中，以及软件实践的周期性本质。
- **讲义与资源的获取**：成员们询问了与当前演示相关的资源，特别是讲义和讨论材料库。
  
  - 分享了可用讲义的链接，强调了小组内的社区支持和知识共享。

**提到的链接**：

- [no title found](https://tree-diffusion.github.io/): 未找到描述
- [HANDOUT - 2024-10-18 - LLMS, ASTs and DSLs - mnml's vault - Obsidian Publish](https://publish.obsidian.md/manuel/Writing/Presentation/2024-10-18+-+LLMs+for+DSLs/HANDOUT+-+2024-10-18+-+LLMS%2C+ASTs+and+DSLs): 讲义 - 2024-10-18 - LLMS, ASTs and DSLs - mnml 的库 - 由 Obsidian Publish 提供支持。
- [Introduction · Crafting Interpreters](https://craftinginterpreters.com/introduction.html): 未找到描述
- [Don't Transform the Code, Code the Transforms: Towards Precise Code Rewriting using LLMs](https://arxiv.org/abs/2410.08806): 用于重写、重构和优化代码的工具应该是快速且正确的。大语言模型 (LLMs) 本身并不具备这些品质。然而，仍然存在巨大的机会...
- [yikes, aw jeez, a youtube thingy](https://youtube.com/@yikesawjeez): 去看我的 Twitter，我做一些愚蠢的 AI 玩意，@yikesawjeez 同时也加入 Discord，我现在剪贴板里没有链接，但你会找到它的，我会教你做愚蠢的 AI 玩意，然后我们...
- [Gödel, Escher, Bach - Wikipedia](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach): 未找到描述
- [Thanks Barney Ross GIF - Thanks Barney ross Sylvester stallone - Discover & Share GIFs](https://tenor.com/view/thanks-barney-ross-sylvester-stallone-the-expendables-2-thank-you-gif-13187459845747433717): 点击查看 GIF
- [Boundary](https://github.com/BoundaryML/): Boundary 有 20 个可用的仓库。在 GitHub 上关注他们的代码。
- [GitHub - BoundaryML/baml: BAML is a language that helps you get structured data from LLMs, with the best DX possible. Works with all languages. Check out the promptfiddle.com playground](https://github.com/BoundaryML/baml): BAML 是一种帮助你从 LLM 获取结构化数据的语言，具有最佳的 DX。支持所有语言。查看 promptfiddle.com 游乐场 - BoundaryML/baml
- [GitHub - norvig/paip-lisp: Lisp code for the textbook "Paradigms of Artificial Intelligence Programming"](https://github.com/norvig/paip-lisp): 教科书《Paradigms of Artificial Intelligence Programming》的 Lisp 代码 - norvig/paip-lisp

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1296911349720547470) (221 条消息🔥🔥):

> - `Model Performance Comparisons`（模型性能对比）
> - `Troubleshooting LM Studio`（LM Studio 故障排除）
> - `Vision Model Capabilities`（视觉模型能力）
> - `Settings for Image Input`（图像输入设置）
> - `Backup and Recovery in LM Studio`（LM Studio 中的备份与恢复）

- **Granite 8B vs Qwen 2.5 7B 性能对比**：用户正在对比 **Granite 8B** 和 **Qwen 2.5 7B** 在编程和科学任务中的表现，寻求基准测试和性能评估。
  
  - 建议参考 [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) 等资源进行性能对比。
- **Llava 图像识别故障排除**：用户报告了 **Llava 模型** 的问题，特别是它无法识别图像并提供不准确的响应。

- 建议包括使用 **jpeg 或 png** 格式，并从 **clean chat** 开始以改善模型响应。
- **LM Studio 中的模型能力**：**Granite 模型**被确认为不具备视觉能力的常规代码模型，强调了检查模型属性的必要性。
  
  - 建议用户在模型的 Hugging Face 仓库中查找 `mmproj` 文件，以确认其是否具备视觉能力。
- **为 Codestral 填写模板表单**：用户正在寻求关于如何为 **Codestral-22B** 填写模板的指导，目前正面临 Jinja 和默认设置的问题。
  
  - 一些人认为缺乏合适的聊天模板可能是与 0.3.4 B 8 最新版本更新相关的 bug。
- **恢复已删除的聊天**：一位用户询问了如何恢复已删除的聊天，并指出一旦删除，元数据就会丢失，通常无法找回。
  
  - 建议包括检查操作系统是否启用了文件历史记录，以及查看 `$HOME/.cache/lm-studio/conversations` 中的本地备份目录。

**提到的链接**：

- [未找到标题](http://127.0.0.1:1234).): 未找到描述
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): 未找到描述
- [app.py · togethercomputer/Llama-3.2-Vision-Free at main](https://huggingface.co/spaces/togethercomputer/Llama-3.2-Vision-Free/blob/main/app.py): 未找到描述
- [$60 AI GPU???](https://www.youtube.com/watch?v=bJKj1yIc4sA): 对 NVIDIA P102-100 进行基准测试。这是一款旧的加密货币挖矿卡，可重新用于 AI 推理。它非常便宜，对于那些...的人来说具有极高的性价比。
- [How do Graphics Cards Work? Exploring GPU Architecture](https://youtu.be/h9Z4oGN89MU?feature=shared): 有兴趣与 Micron 合作制造尖端内存芯片吗？在 Micron 工作：https://bit.ly/micron-careers 了解更多关于 Micron 图形内存的信息...
- [GitHub - YorkieDev/lmstudioservercodeexamples: This readme contains server code examples from LM Studio v0.2.31](https://github.com/YorkieDev/lmstudioservercodeexamples?tab=readme-ov-file#vision-analysis-python): 此 readme 包含来自 LM Studio v0.2.31 的服务器代码示例 - YorkieDev/lmstudioservercodeexamples
- [GitHub - kth8/bitnet: Run BitNet LLM in a container](https://github.com/kth8/bitnet): 在容器中运行 BitNet LLM。通过在 GitHub 上创建账户为 kth8/bitnet 的开发做出贡献。
- [GitHub - remonusa/LoadChatGptHistory](https://github.com/remonusa/LoadChatGptHistory): 通过在 GitHub 上创建账户为 remonusa/LoadChatGptHistory 的开发做出贡献。
- [GitHub - microsoft/VPTQ: VPTQ, A Flexible and Extreme low-bit quantization algorithm](https://github.com/microsoft/VPTQ): VPTQ，一种灵活且极低比特的量化算法 - microsoft/VPTQ
- [How to use File History in Windows 10 and 11](https://www.computerworld.com/article/1621193/how-to-use-file-history-windows-10-windows-11.html): 你可以使用 Windows 内置的文件历史记录工具备份和恢复文件——但你应该了解一些关键限制。
- [Sideload models - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/sideload): 使用你在 LM Studio 之外下载的模型文件
- [Getting Started | LM Studio Docs](https://lmstudio.ai/docs): 了解如何使用 LM Studio 在本地运行 Llama, Mistral, Gemma 和其他 LLM。
- [mistralai/Ministral-8B-Instruct-2410 · Convert weights to HF format](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/discussions/7): 未找到描述
- [GitHub - EricLBuehler/mistral.rs: Blazingly fast LLM inference.](https://github.com/EricLBuehler/mistral.rs): 极速 LLM 推理。通过在 GitHub 上创建账户为 EricLBuehler/mistral.rs 的开发做出贡献。
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet.git): 1-bit LLM 的官方推理框架。通过在 GitHub 上创建账户为 microsoft/BitNet 的开发做出贡献。
- [未找到标题](https://huggingface.co/brunopio/Llama3-8B-1.58-100B-tokens-GGUF/resolve/main/Llama3-8B-1.58-100B-tokens-TQ2_0.gguf): 未找到描述
- [lms log stream - CLI | LM Studio Docs](https://lmstudio.ai/docs/cli/log-stream): 从 LM Studio 流式传输日志。对于调试发送到模型的提示词（Prompt）非常有用。
- [Clear cache during prompt processing by awni · Pull Request #1027 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/pull/1027): 修复了 #1025，详见该处的讨论/改进。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1296924500058701896) (30 messages🔥):

> - `Xeon Processor Configurations` (Xeon 处理器配置)
> - `RX 7900 XTX Performance` (RX 7900 XTX 性能)
> - `RX 6600 Vulkan vs ROCm` (RX 6600 Vulkan 对比 ROCm)
> - `M4 Ultra Chip for AI Tasks` (用于 AI 任务的 M4 Ultra 芯片)

- **Xeon 处理器设置问题**：成员们正在讨论一个与双 **Xeon E5-2603 v4** 处理器相关的 bug。在 **0.3.4 版本**中只能利用 **6 个线程**，而之前的 0.2.31 版本可以利用 8 个线程。
  
  - 一位成员指出，“这是一个已知问题”，并确认已将他们的发现添加到现有的 bug 报告中。
- **RX 7900 XTX 性能对比**：一位用户提到，在运行推理时，**RX 7900 XTX** 使用 **Vulkan** 比使用 **ROCm** 的性能高出约 **10-15%**。
  
  - 另一位用户建议回滚到 **ROCm 1.10**，因为最新的运行时版本存在已知问题。
- **RX 6600 性能缓慢问题**：有用户担心 **RX 6600** 现在只能在 **Vulkan** 上运行而无法使用 **ROCm**，导致更新后性能变慢。
  
  - 一位成员建议，旧版本可能使用的是 **OpenCL** 而非 **ROCm**。
- **关于 M4 Ultra 处理 AI 任务的预测**：讨论了即将推出的 MacBook 中的新 **M4 Ultra 芯片**是否能高效处理 AI 任务，部分人对其能力持怀疑态度。
  
  - 用户表达了不同的看法，指出虽然 M4 Ultra 可能会很好地处理小型任务，但其**昂贵**且**不可升级**的设计可能是一个缺点。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1297806937962057738) (2 messages):

> - `Inflection Payment Issues` (Inflection 支付问题)
> - `Grok Beta Rename` (Grok Beta 重命名)
> - `Grok Pricing Increase` (Grok 价格上涨)
> - `Liquid LFM Pricing Updates` (Liquid LFM 价格更新)

- **Inflection 支付处理器宕机**：由于支付处理问题，**Inflection 3 Pi** 和 **Inflection 3 Productivity** 模型目前都已停用，直至另行通知。
  
  - 这种情况直接影响了所有用户对这些模型的使用和访问。
- **Grok 2 重命名为 Grok Beta**：xAI 要求将 **Grok 2** 重命名为 **Grok Beta**，现在对 `x-ai/grok-2` 的请求将别名指向 `x-ai/grok-beta`。
  
  - 这一变化反映了产品在其开发阶段的定位。
- **Grok 定价现为 $15/M**：**Grok completions** 的定价已上涨至 **$15/M**，令人兴奋的是上下文长度已扩展至 **131,072**。
  
  - 扩展后的上下文允许进行更复杂和详细的交互。
- **Liquid LFM 定价调整**：从本周开始，**Liquid LFM 40b** 的定价将为 **$1/M input** 和 **$2/M output**，而 `:free` 变体仍将可用。
  
  - 这些价格变动旨在提升模型的价值和可访问性。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1297995579581792278) (3 messages):

> - `AI powered text summarizer` (AI 驱动的文本摘要生成器)
> - `Vercel function timeout` (Vercel 函数超时)
> - `OpenRouter API response time` (OpenRouter API 响应时间)
> - `Streaming responses` (流式响应)
> - `Alternative models` (替代模型)

- **构建 AI 摘要生成器面临 Vercel 超时**：一位开发者分享了他们在 Vercel 的 hobby 计划上使用 **Gemma 2 27B** 部署 AI 驱动的文本摘要生成器时遇到的困难，在 OpenRouter API 响应 **10 秒**后出现了 **FUNCTION TIMEOUT** 错误。
  
  - 他们提供了[项目链接](https://summer-chi.vercel.app/)和 [GitHub Repo](https://github.com/ItIsOHM/summer) 以供进一步探索。
- **增加 Vercel 函数执行时间**：建议根据 [Vercel 文档](https://vercel.com/docs/functions/configuring-functions/duration)将 Vercel 函数的默认超时时长从 **10 秒**增加到最高 **60 秒**。
  
  - 强调了这一更改对于避免超过设定最大时长而导致函数终止至关重要。
- **探索超时问题的替代解决方案**：提出了包括**流式响应 (streaming responses)** 在内的替代方案，以避免等待完整的摘要，这有助于缓解超时问题。
  
  - 还建议考虑使用更快的模型，如 **Gemini Flash** 或通过 **Samba Nova** 运行的 **Llama 模型**，以提高性能。

**提到的链接**：

- [Configuring Maximum Duration for Vercel Functions](https://vercel.com/docs/functions/configuring-functions/duration)：了解如何设置 Vercel 函数的最大时长。
- [no title found](https://summer-chi.vercel.app/)：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1296962097212686376) (225 条消息🔥🔥):

> - `OpenRouter 模型问题`
> - `Grok 2`
> - `Hermes 3`
> - `账单问题`
> - `AI 模型能力`

- **Grok 2 经历波动和价格更新**：用户正经历 **Grok 2** 频繁的停机，同时伴随着多次价格变动，目前成本已升至每月 15 美元。
  
  - 部分用户对性能不稳定表示沮丧，并认为需要更好的功能来证明涨价的合理性。
- **Hermes 3 模型性能问题**：多名用户报告在使用 **Hermes 3** 模型时收到 **429 错误**，表明他们比以前更频繁地触发速率限制（rate limits）。
  
  - 这引起了不满，因为用户指出该模型以前在没有这些限制的情况下运行良好。
- **用户面临的账单问题**：一名用户报告了 **OpenRouter 账单系统** 的问题，尽管账户中有额度（credits），但仍产生了意外费用。
  
  - 其他用户确认遇到了类似问题，并建议联系客服解决。
- **关于结构化提示词模型能力的讨论**：用户正在探索哪些模型（如 **airoboros-70b**）最适合处理结构化输出和特定任务请求。
  
  - 目前正在对各种模型在生成无审查内容（uncensored content generation）方面的性能进行对比调查。
- **对 Azure 和 HareProxy 服务的担忧**：用户对 **HareProxy** 服务意外出现在其活动摘要中表示担忧，并指出有报告称其不可靠。
  
  - 讨论还涉及了 Azure 与其他模型提供商相比的可靠性，部分用户更倾向于特定的替代方案。

**提到的链接**：

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102))：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109))：未找到描述
- [Full Stack && Web3 Developer](https://daniel0629.vercel.app)：我是一名高技能的区块链和全栈开发人员，在设计和实现复杂的去中心化应用和 Web 解决方案方面拥有丰富经验。
- [无标题](https://ai.google.dev/gemini-api/terms#use-restrictions>)：未找到描述
- [hareproxy-inst-1](https://api.hareproxy.io.vn/)：未找到描述
- [Nous: Hermes 3 405B Instruct – 提供商状态](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b/providers)：查看提供商状态并向 Nous: Hermes 3 405B Instruct 发起负载均衡请求 - Hermes 3 是一款通用语言模型，相比 Hermes 2 有多项改进，包括先进的 Agent 能力...
- [Hermes 3 405B Instruct - API, 提供商, 统计](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b:free.)：Hermes 3 是一款通用语言模型，相比 Hermes 2 有多项改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...
- [OpenRouter 状态](https://status.openrouter.ai/)：OpenRouter 故障历史
- [GitHub - deepseek-ai/Janus](https://github.com/deepseek-ai/Janus?tab=readme-ov-file)：通过在 GitHub 上创建账户来为 deepseek-ai/Janus 的开发做出贡献。
- [every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui/blob/main/README.md)：适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1297900665560891452) (3 条消息):

> - `自定义提供商密钥 (Custom Provider Keys)`
> - `自助集成注册 (Self-Service Integration Sign Up)`

- **申请自定义提供商密钥的 Beta 测试权限**：一名成员表达了获取 **自定义提供商密钥 Beta 测试权限** 的兴趣，并直接提出了需求。
  
  - *未立即得到回复*，该成员对目前的情况表示理解。
- **自助集成注册延迟**：一名成员强调，虽然之前承诺过 **集成的自助注册** 功能，但目前尚未上线。
  
  - 他们建议感兴趣的成员需要等待，并提供了相关讨论的链接：[集成更新](https://discord.com/channels/1091220969173028894/1296148568683577345/1296205973408714803)。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1296925278563602582) (126 条消息🔥🔥):

> - `Durable Execution Concepts` (Durable Execution 概念)
> - `Aider and VSCode Integration` (Aider 与 VSCode 集成)
> - `Mistral API Usage` (Mistral API 使用)
> - `CEDARScript Runtime` (CEDARScript 运行时)
> - `Hello World Refactoring Issues` (Hello World 重构问题)

- **理解 Durable Execution**：成员们讨论了 **durable execution** 的概念，这是一种代码不受时间和空间限制的抽象，非常适合构建长期运行的工作流（long-running workflows）。
  
  - 提供了一个链接到 [Temporal background checks](https://learn.temporal.io/examples/go/background-checks/) 的示例，以说明其实际应用。
- **探索 Aider 与 VSCode 的集成**：重点介绍了 **VSCode Aider Extension**，它能够将 AI 驱动的编程辅助直接集成到 Visual Studio Code 中，提升用户的编程体验。
  
  - 功能包括自动文件同步和代码修改建议，并邀请用户在 GitHub 上提交新功能请求。
- **在 Aider 中使用 Mistral API**：提供了在 Aider 中使用 **Mistral API** 的说明，包括如何在编程会话期间通过命令行指定所使用的模型。
  
  - 指导用户创建 `.aider.conf.yml` 文件，并说明了如何输入相应的命令来为 Mistral 配置 Aider。
- **CEDARScript 在代码管理中的作用**：讨论了 **CEDARScript** 运行时在减轻 LLM 对低级代码语法关注方面的作用，使其能够专注于高级抽象。
  
  - CEDARScript 支持多种语言，目前正在探索其与 Aider 的集成，以增强代码编辑能力。
- **幽默的 Hello World 重构案例**：一位用户分享了 Aider 试图在代码库的关键部分添加 “Hello World” 函数并导致意外更改的有趣经历。
  
  - 虽然这被视为一种幽默的困扰而非 Bug，但它引发了关于 AI 代码生成中明显存在的幻觉（hallucinations）的讨论。

**提到的链接**：

- [Qwen2.5-Coder: Code More, Learn More!](https://qwenlm.github.io/blog/qwen2.5-coder/)：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介：在 4 月初，我们推出了 CodeQwen1.5，引起了社区的极大关注。自那时起，我们一直致力于增强……
- [VSCode Aider (Sengoku) - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=sengokudaikon.vscode-aider-sengoku)：Visual Studio Code 扩展 - 直接在 VSCode 中运行 Aider，实现无缝集成并增强工作流。
- [Background Check Application in Go | Learn Temporal](https://learn.temporal.io/examples/go/background-checks/)：该项目的目标是教你（开发者）如何通过使用 Temporal SDK 引导你完成一个……来思考构建具有人类驱动的长期运行工作流的 Temporal 应用程序。
- [Other LLMs](https://aider.chat/docs/llms/other.html)：aider 是你终端里的 AI 结对编程工具。
- [YAML config file](https://aider.chat/docs/config/aider_conf.html)：如何使用 yaml 配置文件配置 aider。
- [AI Coding App CRUSHES $60M Tool (CURSOR KILLER??!)](https://www.youtube.com/live/ikn7JSUflTI?si=uJqzHU9Rh-fhBU7S)：我们将 Repo Prompt 与价值 6000 万美元的 AI 编程工具进行了正面对比，结果会让你大吃一惊。Eric Provencher 发现了提示词的秘诀……
- [Cline + Aider + Mistral FREE API : This is the BEST FREE WAY to do AI CODING! (Beats Gemini!)](https://www.youtube.com/watch?v=igE0X25bHcE)：加入此频道以获得会员福利：https://www.youtube.com/@AICodeKing/join 在这段视频中，我将告诉你如何使用 Mistral 免费的……

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1297285506345730102) (56 条消息🔥🔥):

> - `Aider 与 Sonnet 及 Claude 的使用`
> - `管理 Aider 的自动提交（Auto Commit）功能`
> - `Aider 中的文件创建与存在性问题`
> - `利用 Aider 历史记录获取上下文`
> - `在 Aider 中设置主模型（Main Model）与弱模型（Weak Model）`

- **关于 Aider 使用的反馈**：用户表达了对 Aider 的赞赏，并分享了在项目 AI 开发中使用它的经验。
  
  - 他们讨论了面临的具体技术挑战，例如模型对已存在文件的处理。
- **管理 Aider 中的自动提交**：一位用户询问是否可以配置 Aider 以禁止自动提交更改，希望在提交前进行手动审核。
  
  - 另一位用户提到了 Aider 文档中的 `--auto-commits` 选项，该选项允许切换此功能。
- **Aider 中的文件创建问题**：有报告称 Aider 尝试创建已经存在的文件，导致用户对模型的行为感到困惑。
  
  - 一些用户怀疑这可能与 Git 的文件追踪与文件系统上实际文件存在性的差异有关。
- **利用 Aider 历史记录**：一位用户询问 Aider 是否通过加载之前会话的历史记录来保持上下文，引发了关于管理聊天历史记录功能的讨论。
  
  - 有人提到 Aider 可以在启动会话时恢复过去的聊天历史和相关文件，从而提升用户体验。
- **配置主模型与弱模型**：一位用户寻求关于如何在 Aider 中显式设置其主模型（Main Model）和弱模型（Weak Model）的指导。
  
  - 另一位用户提供了一个 YAML 配置示例，通过创建 `.aider.conf.yml` 文件来定义模型。

**提到的链接**：

- [代码检查与测试](https://aider.chat/docs/usage/lint-test.html#testing)：自动修复代码检查（linting）和测试错误。
- [OpenGPT 4o - KingNish 的 Hugging Face Space](https://huggingface.co/spaces/KingNish/OpenGPT-4o)：未找到描述。
- [选项参考](https://aider.chat/docs/config/options.html#--auto-commits)：关于 aider 所有设置的详细信息。
- [指定编码规范](https://aider.chat/docs/usage/conventions.html)：告知 aider 在处理代码时遵循你的编码规范。
- [教程视频](https://aider.chat/docs/usage/tutorials.html)：由 aider 用户制作的入门和教程视频。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1296978933815971930) (1 条消息):

> - `bitnet.cpp`
> - `1-bit LLMs`
> - `ARM 和 x86 CPU 上的推理性能`

- **微软发布用于 1-bit LLMs 的 bitnet.cpp**：微软发布了 [bitnet.cpp](https://github.com/microsoft/BitNet)，作为 **1-bit LLMs**（包括 **BitNet b1.58** 模型）的官方推理框架。
  
  - 该框架支持优化内核，可在 CPU 上实现**快速且无损的推理**，并计划在未来支持 NPU 和 GPU。
- **ARM CPU 上显著的加速和效率提升**：在 ARM CPU 上，bitnet.cpp 实现了 **1.37 倍到 5.07 倍** 的加速，大型模型表现出最显著的性能提升。
  
  - 它还将能耗降低了 **55.4% 至 70.0%**，提高了运行 LLM 的整体效率。
- **x86 CPU 性能得到显著增强**：对于 x86 CPU，bitnet.cpp 提供了 **2.37 倍到 6.17 倍** 的加速，能耗降低了 **71.9% 至 82.2%**。
  
  - 这使得在单个 CPU 上运行 **100B BitNet b1.58 模型** 成为可能，速度接近人类阅读速率（每秒 5-7 个 token）。

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1297034862695419914) (30 条消息🔥):

> - `TensorRT-LLM 代码共享`
> - `Unsloth 讲座与资源`
> - `GPU MODE 演讲录像`
> - `活动日程查询`
> - `分布式训练框架对比`

- **TensorRT-LLM 代码共享**: 用户分享了 **TensorRT-LLM** 仓库的链接，特别指出了 [cutlass int8 gemm kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63)。该资源为用户提供了一个用于定义 Large Language Models (LLMs) 的 Python API。
  
  - 分享的 kernel 可以增强需要优化性能的模型的**高效推理**。
- **Unsloth 讲座与资源**: 提醒成员关注即将举行的关于**系统工程**底层方面的讲座，讨论 **Triton kernels** 和 CUDA。分享了相关资源，包括一个 [GitHub 链接](https://github.com/unslothai/unsloth)，供参与者参考。
  
  - 此外，讲座的幻灯片也已发布：[查看幻灯片](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing)。
- **GPU MODE 演讲录像**: 讲座结束后，向参与者表示感谢，并告知讲座已**录制**以便后续观看。录像可能会在几天内上传到 [YouTube 频道](https://www.youtube.com/@GPUMODE/videos)。
  
  - 这为错过直播的人提供了跟进讨论的机会。
- **活动日程查询**: 一位成员询问在哪里可以报名参加公告中提到的讲座。回复引导他们前往 **events 选项卡**，在那里可以找到 Zoom 链接。
  
  - 这有助于确保成员无论身处哪个时区都能参加相关讲座。
- **分布式训练框架对比**: 一位用户表示需要对比不同**分布式训练框架**的资源。他们指出，不同论文中配置的不一致阻碍了准确的对比。
  
  - 这凸显了一个空白，即标准化可以提高对框架如何影响训练结果的理解。

**提到的链接**:

- [GPU MODE](https://www.youtube.com/@GPUMODE/videos): 一个 GPU 读书小组和社区 https://discord.gg/gpumode 补充内容见此处 https://github.com/gpu-mode 由 Mark Saroufim 和 Andreas Köpf 创建
- [GPU MODE 第 32 讲 - Unsloth](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing): 1 第 32 讲 GPU MODE LLM 系统工程，来自 Unsloth 的 Daniel
- [TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h at a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63): TensorRT-LLM 为用户提供了一个易于使用的 Python API，用于定义 Large Language Models (LLMs) 并构建包含最先进优化技术的 TensorRT 引擎，以执行高效推理...

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1297948189885726772) (5 messages):

> - `Triton 与 PTX 的兼容性`
> - `Triton 对 Windows 的支持`
> - `torch.compile 与 Triton 之间的交互`

- **Triton Python 在向 PTX 转换时可能会遇到困难**：虽然 **NVIDIA backend** 似乎在处理 **LLVM IR**，但关于从 **Triton Python** 到 **PTX** 是否有直接路径仍存在不确定性。
  
  - 这种模糊性引发了对编译流水线（compilation pipeline）有效性的质疑。
- **Triton 对 Windows 的支持仍存疑问**：虽然使用 **Visual Studio's LLVM** 的 **Windows 系统** 有极小的可能正常运行，但由于改动不足，暗示 Triton 的 Windows 版本作者可能缺乏深入理解。
  
  - 对于兼容性所需的必要调整是否已得到妥善处理，仍然存疑。
- **torch.compile 与 Triton 之间存在不稳定的关系**：**torch.compile** 与 **Triton** 之间似乎存在某种本应无缝的交互，但 Triton 在发生故障时却不会抛出错误。
  
  - 这种缺乏错误提示的情况增加了调试难度，并预示了它们在集成过程中可能存在的问题。

**提到的链接**：

- [triton/third_party/nvidia/backend/compiler.py at a19f32454271ff9565ab957834bdf1e5d4ddce57 · triton-lang/triton](https://github.com/triton-lang/triton/blob/a19f32454271ff9565ab957834bdf1e5d4ddce57/third_party/nvidia/backend/compiler.py#L310)：Triton 语言和编译器的开发仓库 - triton-lang/triton
- [triton/python/src/llvm.cc at a19f32454271ff9565ab957834bdf1e5d4ddce57 · triton-lang/triton](https://github.com/triton-lang/triton/blob/a19f32454271ff9565ab957834bdf1e5d4ddce57/python/src/llvm.cc#L394)：Triton 语言和编译器的开发仓库 - triton-lang/triton

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1297147308076437595) (11 条消息🔥):

> - `LibTorch 中的 Torch Distributions`
> - `新的 PyTorch 环境变量`
> - `清理 PyTorch 中的编译算子 (Compiled Ops)`
> - `PyTorch 中 Autocast 的行为`
> - `ResNet50 的 DDP 训练问题`

- **LibTorch 缺少 MultivariateNormal 的等效实现**：一位用户询问在 PyTorch 的 C++ API LibTorch 中，是否存在与 `torch.distributions.MultivariateNormal` 等效的实现。
  
  - 这反映了在 PyTorch 生态系统的不同编程接口中对类似功能的持续需求。
- **新的 PyTorch 环境变量可防止功率骤降**：一位成员指出 `PYTORCH_NO_POWERPLANT_BLOWUP` 是一个新的环境变量，旨在缓解 Checkpointing 期间的大幅功率骤降。
  
  - *这一变化被讨论为大型计算任务性能管理方面的显著改进。*
- **关于清理缓存算子的问题**：一位用户提出了关于如何清理 PyTorch 中编译后的操作或缓存的问题，并提到 `torch.compiler.reset()` 和 `torch._dynamo.reset_code_caches` 作为潜在的解决方案。
  
  - 他们还询问了如何在模型训练设置中使用 `torch.compile` 实现强制重新编译。
- **Autocast 揭示了 dtype 的差异**：一位用户演示了虽然 `torch.autocast` 可能导致返回类型为 `torch.float32`，但他们观察到结果会根据所使用的设备类型和数据类型而有所不同。
  
  - 这引发了关于混合精度计算期间 Autocasting 预期行为的疑问。
- **DDP ResNet50 训练问题**：一位用户报告称，在尝试使用稀疏掩码 (sparse masks) 训练 ResNet50 模型时，遇到了 PyTorch 2.5 的 OOM 错误，以及关于 Profiler 函数被跳过的各种警告。
  
  - 尽管他们打算使用 2.4 版本，但却意外降级到了 PyTorch 2.2.1，这表明可能存在安装问题。

**提到的链接**：

- [来自 Pytorch To Atoms (@PytorchToAtoms) 的推文](https://fxtwitter.com/PytorchToAtoms/status/1828148537013510474)：主线 PyTorch 确实有一个名为 "PYTORCH_NO_POWERPLANT_BLOWUP" 的新环境变量，用于防止 Checkpointing 期间以及 Trace 中通信无法重叠的情况下的功率骤降……
- [在同一台机器上对一个模型使用两次 torch.compile，是否存在优化操作的缓存？](https://stackoverflow.com/questions/77931982/using-torch-compile-twice-on-a-model-on-the-same-machine-is-there-a-cache-of-op)：我正在通过以下方式使用 torch.compile 编译 torch 模型：self.model = torch.load(saved_model_path, map_location=self.device).to(self.device) self.model.eval() self.model.half() &...
- [torch.compile 中的编译时缓存 — PyTorch Tutorials 2.5.0+cu124 文档](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)：未找到描述
- [TV pickup - 维基百科](https://en.wikipedia.org/wiki/TV_pickup)：未找到描述
- [自动混合精度包 - torch.amp — PyTorch 2.5 文档](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32)：未找到描述
- [pytorch:2.0.0 DDP 训练错误，但旧版本正常 · Issue #1144 · pytorch/examples](https://github.com/pytorch/examples/issues/1144)：您的问题可能已经被报告过！在创建新问题之前，请先在 Issue 追踪器中搜索。背景 Pytorch 版本：操作系统及版本：pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime Your...

---

### **GPU MODE ▷ #**[**announcements**](https://discord.com/channels/1189498204333543425/1189640399476764692/1297269030574624808) (1 条消息):

> - `Han 兄弟`
> - `Unsloth 演讲`
> - `Triton 技巧`
> - `CUDA 技术`

- **Han 兄弟将讨论 Unsloth**：**Han 兄弟** 将在 **15 分钟后** 在 Discord 上介绍 **Unsloth**。
  
  - 期待在他们的演讲中看到许多疯狂的 **Triton** 和 **CUDA** 技巧。
- **对 Triton 和 CUDA 见解的期待**：许多成员对 Han 兄弟即将分享的 **Triton** 和 **CUDA** 技巧表示兴奋。
  
  - 预计这次演讲将带来宝贵的见解和创新技术。

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1297628443520798741) (6 messages):

> - `Domino Communication Scheme` (Domino 通信方案)
> - `Torchtitan Library` (Torchtitan 库)
> - `Large Language Model Training` (大语言模型训练)

- **Domino 在 LLM 训练中隐藏了通信开销**：关于 [Domino](https://arxiv.org/abs/2409.15241) 的论文提出了一种通用方案，通过计算与通信的重叠来消除大语言模型 (LLM) 分布式训练中的通信开销，在 Nvidia DGX-H100 GPU 上实现了高达 **1.3 倍的加速**。
  
  - 通过将批处理训练的数据依赖分解为更小的独立部分，Domino 相比 **Megatron-LM** 提高了效率。
- **Torchtitan 与 Domino 的关联**：一位成员指出，用于大模型训练的 **Torchtitan** 库实际上与论文中提到的 Domino 方法相同。
  
  - 他们引用了一个支持原生 PyTorch 训练的 [Torchtitan GitHub 仓库](https://github.com/pytorch/torchtitan)。
- **与 Torchtitan 论文的相似性**：另一位成员确认 arXiv 上有一篇关于 Torchtitan 的论文，其内容与 Domino 的概念非常相似，突显了两者的紧密关系。
  
  - 这表明在优化 LLM 训练的方法论上存在很强的关联。

**提到的链接**：

- [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/abs/2409.15241)：鉴于生成式 AI 的普及，大语言模型 (LLM) 通常消耗数百或数千个 GPU 来并行化和加速训练过程。通信开销成为……
- [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205)：大型软件系统产生的日志是监控系统行为不可或缺的一部分。先进的日志分析有助于系统故障的检测、报警和诊断。日志解析……
- [GitHub - pytorch/torchtitan: A native PyTorch Library for large model training](https://github.com/pytorch/torchtitan)：一个用于大模型训练的原生 PyTorch 库。欢迎在 GitHub 上通过创建账号为 pytorch/torchtitan 的开发做出贡献。

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1297955618254098544) (1 messages):

> - `Hiring GPU Programmers` (招聘 GPU 程序员)
> - `Decentralized AI` (去中心化 AI)
> - `Tokens per second improvement` (每秒 Token 数提升)

- **Opentensor Foundation 寻求 GPU 人才**：**Bittensor** ([官网](https://bittensor.com/)) 的开发方 Opentensor Foundation 宣布，他们正在招聘顶尖的 **GPU 编程**人才，以增强去中心化 AI 的能力。
  
  - 人才负责人 Ryan 鼓励申请者提交 PR，利用其 [GitHub 脚本](https://github.com/unconst/boltzmann/blob/pipe/deep.py) 中的配置，提升 H100 机器上的 **tokens per second** (每秒 Token 数)。
- **为大胆的合作者提供机会**：鼓励感兴趣的候选人通过与 Opentensor 团队直接合作，以实操的方式展示自己的技能。
  
  - 此次人才招募强调了为 **decentralized AI** (去中心化 AI) 领域做出重大贡献的潜力。

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1297464669979676747) (40 messages🔥):

> - `Flash Attention 乘法`
> - `LlamaCPP 和 GGML 库的使用`
> - `Raspberry Pi 图形性能`
> - `Triton 和 CUDA 兼容性`
> - `调试模型中的 Multihead Attention`

- **理解 Flash Attention 乘法**：一位用户询问为什么 Flash Attention 将 O_old 与 l_i\*e^m 相乘，推测这可能是为了归一化目的。
  
  - 这引发了关于 O_old 的作用及其在 Flash Attention 中重要性的讨论。
- **开始使用 LlamaCPP**：一位成员建议下载并构建 [LlamaCPP](https://link.to/llamacpp) / GGML 库，以便更好地理解优化的 Tensor 使用。
  
  - 他们强调了运行 LLM 以及将 Huggingface 模型转换为 ONNX 格式进行优化的重要性。
- **Raspberry Pi 与替代开发板的图形性能**：讨论了关于 Raspberry Pi 专有集成显卡的问题，并建议通过逆向工程来提升性能。
  
  - 用户推荐了像 **Odroid N2+** 和 **RK3588** 这样具有更好图形能力的开源驱动板。
- **Triton 兼容性与 CUDA 版本**：一位用户在使用 Triton 进行 Liger 操作时遇到了 CUDA 显存溢出（out of memory）错误，并询问是否能在 K80 等旧款 GPU 上运行。
  
  - 建议降级 CUDA toolkit，因为新版本不支持旧架构，即 SM_37。
- **调试 Multihead Attention 掩码问题**：一位用户报告了与其模型解码器中 attn_mask 形状相关的运行时错误，表明其与预期大小不匹配。
  
  - 在经过一周的故障排除后，他们表达了挫败感，并寻求社区帮助解决 mask 问题。

**提到的链接**：

- [Vector Addition — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)：未找到描述
- [Nvidia Tesla K80, Cuda version support?](https://forums.developer.nvidia.com/t/nvidia-tesla-k80-cuda-version-support/67676)：你好，我正在使用带有 GPU K80 和 Ubuntu 16.04 的 Google Cloud 实例。但我有一个问题，该硬件正确的 CUDA 版本是 9.0、9.2 还是 10？在此链接中你可以看到更多信息...

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1297312465318842370) (4 messages):

> - `第 4 章练习`
> - `Occupancy 计算`

- **找到第 4 章练习答案**：一位用户分享了包含 GPU 架构笔记的仓库中 [第 4 章练习答案](https://github.com/mandliya/PMPP_notes/blob/main/4_GPU_Architecture/exercises.md) 的链接。
  
  - 该资源是更广泛的 [PMPP 笔记项目](https://github.com/mandliya/PMPP_notes) 的一部分，为大规模并行处理器编程提供了有用的信息。
- **对 Occupancy 计算的不确定性**：另一位用户在收到练习答案链接后，对自己的 **Occupancy 计算** 表示不确定。
  
  - 这突显了学习者在应对复杂的 GPU 编程概念时的共同担忧。

 

**提到的链接**：[PMPP_notes/4_GPU_Architecture/exercises.md at main · mandliya/PMPP_notes](https://github.com/mandliya/PMPP_notes/blob/main/4_GPU_Architecture/exercises.md)：Programming Massively Parallel Processors 的笔记和代码 - mandliya/PMPP_notes

 

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 messages):

gau.nernst: [https://www.youtube.com/watch?v=hfb_AIhDYnA](https://www.youtube.com/watch?v=hfb_AIhDYnA)

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1296928468100976702) (2 条消息):

> - `GitHub Issues`
> - `PyTorch Release Performance`

- **性能回退的 GitHub Issue 请求**：一位成员请求 @appy22 创建一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/138386)，内容是关于在使用 **torch.compile** 时，**torch 2.5** 相比 **2.4.1** 出现的性能回退。
  
  - 他们指出，在包括 **4090 RTX** 在内的多台机器上进行测试时，最新版本似乎变慢了。
- **GitHub Issue 讨论点**：分享了标题为 **'torch 2.5 slower than 2.4.1?'** 的 GitHub issue，详细说明了关于性能差异的 Bug 报告。
  
  - 在该 issue 中，用户提到在最新的稳定版本上使用 **torch.compile** 时遇到了明显的减速。

 

**提到的链接**：[torch 2.5 slower than 2.4.1 ? · Issue #138386 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/138386)：🐛 描述 Bug：我注意到最新的稳定版本 2.5.0 在使用 torch.compile (reduce-overhead) 时比 2.4.1 慢，我在带有 4090 RTX 的不同机器上进行了尝试，结果几乎...

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1297482652450033716) (6 条消息):

> - `Daniel Hanchen's talk recording`
> - `Inventec CXL Box`
> - `Micron GPU architecture`
> - `Iceberg lettuce salad recipe`
> - `Lecture 32: Unsloth`

- **Daniel Hanchen 的演讲录像已发布**：[Daniel Hanchen 演讲的录像](https://youtu.be/hfb_AIhDYnA?si=LU6712r-oOIARRKq)已照常发布在频道中。
  
  - 一位成员确认他们在发布后立即观看了，并对链接表示感谢。
- **Inventec CXL Box 彻底改变内存**：[Inventec CXL Box](https://www.servethehome.com/inventec-96-dimm-cxl-expansion-box-at-ocp-summit-2024-for-tbs-of-memory-astera-labs-intel-xeon-6/) 提供了一个 96x DDR5 DIMM 内存架，可实现 **20TB** 的 PCI Gen5 连接 RAM。
  
  - 它连接到即将推出的 **8路 Intel Xeon 6** 服务器，总共提供惊人的 **224 个 DIMM 插槽**，用于纵向扩展应用。
- **了解 GPU 架构**：[显卡是如何工作的？探索 GPU 架构](https://www.youtube.com/watch?v=h9Z4oGN89MU) 是一个 YouTube 视频，讨论了 Micron 的 GPU 设计和内存技术。
  
  - 该视频还为那些对尖端内存芯片开发感兴趣的人分享了 Micron 的职业机会。
- **成员分享他们的烹饪作品**：一位成员详细介绍了一顿由 **卷心菜沙拉**、**土豆泥** 和用各种配料制作的 **牛肉饼** 组成的餐食。
  
  - 他们还分享了作为餐食一部分的 **热饮** 和新鲜水果。

**提到的链接**：

- [Inventec 96 DIMM CXL Expansion Box at OCP Summit 2024 for TBs of Memory](https://www.servethehome.com/inventec-96-dimm-cxl-expansion-box-at-ocp-summit-2024-for-tbs-of-memory-astera-labs-intel-xeon-6/)：可能是 OCP Summit 2024 上最酷的硬件，Inventec 展示了一台 8路 Intel Xeon 6 服务器，配有 96 DIMM CXL 扩展箱，总计 224 个 DIMM。
- [Lecture 32: Unsloth](https://youtu.be/hfb_AIhDYnA?si=LU6712r-oOIARRKq)：未找到描述
- [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)：有兴趣与 Micron 合作制造尖端内存芯片吗？在 Micron 工作：https://bit.ly/micron-careers 了解更多关于 Micron 图形内存的信息...

---

### **GPU MODE ▷ #**[**irl-meetup**](https://discord.com/channels/1189498204333543425/1218444432588800010/1297539159149514824) (6 条消息):

> - `Sydney Meetup Coordination`
> - `NeurIPS Conference Participation`
> - `NeurIPS Location`

- **悉尼的小伙伴想聚会**：一位成员询问悉尼或澳大利亚是否有人有兴趣协调聚会，并提议在大学举办。
  
  - 这对于当地的 AI 爱好者来说是一个建立联系和协作的好机会。
- **确认参加 NeurIPS**：另一位成员确认参加 NeurIPS 会议，引发了对该活动的期待。
  
  - 这引发了参与者之间关于还有谁可能参加的讨论。
- **NeurIPS 在温哥华举办**：讨论中明确了 NeurIPS 的地点是在 **加拿大温哥华**。
  
  - 几位社区成员对活动的临近表现出极大的热情。

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/) (1 条消息):

seahorse0180: 刚才也遇到了这个问题。

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1297739422652960778) (2 条消息):

> - `Parallel Prefix Sum Algorithm`
> - `Mamba Training`
> - `SSMs`
> - `Linear RNNs`
> - `llm.c Repository`

- **关于 Parallel Prefix Sum Algorithm 的咨询**：一位成员询问在 [llm.c repository](https://link.to.repo) 中是否有用于训练 **Mamba**、**SSMs** 或 **Linear RNNs** 的 **parallel prefix sum algorithm**。
  
  - *在 LLM.c repo 的任何地方是否有用于训练 Mamba / SSMs / Linear RNNs 的 parallel prefix sum algorithm？*
- **llm.c 仅限于 GPT-2**：另一位成员澄清说，除非有最新更新，否则 **llm.c** 目前专门专注于 **GPT-2**。
  
  - 他们强调 *llm.c 目前专门针对 GPT-2*。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1297270629736644711) (2 条消息):

> - `Improvement sources for MI300X`
> - `RCCL tuning issues`
> - `Performance in single node RCCL`
> - `Async TP kernels challenges`

- **针对 MI300X 的 Ultra Ethernet vs IB**：建议等待 [Ultra Ethernet](https://github.com/ROCm/rccl/blob/develop/src/graph/tuning.cc) 或像 Microsoft 那样在 **MI300X** 上使用 **InfiniBand**，这是提升性能的重要来源。
  
  - 据指出，**native RoCEv2** 不适合处理具有突发性和大流（elephant flow）流量的 **AI/HPC** 应用。
- **RCCL tuning.cc 协议选择**：观察到 **rccl tuning.cc** 有时会选择非最优的协议和算法，这可能会阻碍 **MI300X** 实现应有的性能。
  
  - 出现此问题主要是因为缺乏 **MI300X** 的参考网络架构，它与 **H100** 不同。
- **单节点 RCCL 的改进空间**：**single node RCCL** 中有许多“低垂的果实”（容易实现的改进机会）可以显著增强性能。
  
  - 例如，**ROCm** 仍然缺乏对 **symmem** 的支持，这使得在 **AMD** 系统上有效地编写 **async TP kernels** 变得具有挑战性。

 

**提到的链接**：[rccl/src/graph/tuning.cc at develop · ROCm/rccl](https://github.com/ROCm/rccl/blob/develop/src/graph/tuning.cc)：ROCm Communication Collectives Library (RCCL)。通过在 GitHub 上创建账号来为 ROCm/rccl 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**sparsity-pruning**](https://discord.com/channels/1189498204333543425/1247663759434977453/1297839205703090201) (3 条消息):

> - `Activation Sparsity Tools`
> - `PyTorch Sparse Functionality`
> - `PowerInfer Research`
> - `Sparse Kernel Implementations`

- **关于 Activation Sparsity 工具的提问**：一位成员正在寻找用于 **activation sparsity** 矩阵乘法的有效工具，并提到正在使用 **PyTorch** 的 `to_sparse_semi_structured`，但该功能仅限于 **2D tensors**。
  
  - *他们指出，由于这一限制，对于更大维度的 tensor 需要进行手动迭代。*
- **为 Sparsity 任务创建 GitHub Issue**：一位成员建议在 **GitHub** 上针对 **sparsity** 任务创建一个 issue，特别是关于 `to_sparse_semi_structured` 的使用及其性能问题。
  
  - *他们强调了包含一个最小可复现代码（minimal code repro）对于有效排查问题的重要性。*
- **使用训练 Kernel 以提高效率**：另一位成员指出 `to_sparse_semi_structured` 采用了较慢的转换方法，适用于 **weight sparsity**，但对于运行时所需的 **activation sparsity** 效率不高。
  
  - *他们建议利用更快的 sparsification kernels 来提升整体性能。*

 

**提到的链接**：[Issues · pytorch/ao](https://github.com/pytorch/ao/issues)：用于训练和推理的 PyTorch 原生量化与稀疏化 - Issues · pytorch/ao

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1297274766146928712) (7 条消息):

> - `Acknowledgement of Contributors in Liger Arxiv Whitepaper` (Liger Arxiv 白皮书中的贡献者致谢)
> - `Gradient Accumulation Bug in Liger Kernel` (Liger Kernel 中的 Gradient Accumulation Bug)
> - `Memory Issues with Triton and Liger Operations` (Triton 与 Liger 算子的内存问题)
> - `Calls for Code Review on Gradient Accumulation` (关于 Gradient Accumulation 的代码审查请求)
> - `Updates on Liger Kernel Documentation` (Liger Kernel 文档更新)

- **即将发布的白皮书致谢贡献者**：讨论了在 [Liger Arxiv whitepaper](https://arxiv.org/pdf/2410.10989) 中加入对 **开源贡献者** 通用致谢的相关事宜。
  
  - 目前正在准备更新版本，以包含 **核心贡献者姓名** 并推广委员会制度。
- **Gradient Accumulation Bug 咨询**：一名成员询问 transformers 库中最近的 **Gradient Accumulation Bug** 修复是否也适用于 Liger Kernel 的 cross entropy 算子。
  
  - 这突显了对 Liger Kernel 功能中潜在问题保持清晰认识的持续需求。
- **与 Liger 算子相关的 CUDA 内存错误**：有成员反映在使用 Liger 算子配合开启了 torch compile 的 PyTorch 模型时遇到了 **CUDA out of memory** 错误。
  
  - 这引发了关于 **Triton** 或 **Liger** 相关特定内存分配模式的问题。
- **Gradient Accumulation 技术的代码审查**：成员们分享了涉及不同算子（如 fused linear cross entropy 和 layer norm）的 **Gradient Accumulation** 相关代码片段。
  
  - 这些提交表明社区正努力确保 **高效 Gradient Accumulation** 的实现。

**提到的链接**：

- [Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/fused_linear_cross_entropy.py#L110)：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。
- [Liger-Kernel/src/liger_kernel/ops/fused_linear_jsd.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/fused_linear_jsd.py#L98)：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。
- [Liger-Kernel/src/liger_kernel/ops/layer_norm.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/layer_norm.py#L216)：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。
- [Liger-Kernel/src/liger_kernel/ops/rms_norm.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/rms_norm.py#L289)：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。

---

### **GPU MODE ▷ #**[**metal**](https://discord.com/channels/1189498204333543425/1285384841730457600/1297058056215199806) (9 messages🔥):

> - `Objective-C language server`
> - `PyTorch 中的 C/C++ 内存管理`
> - `PyTorch 中的 MPS stream`
> - `Apple Silicon 上的统一内存 (Unified memory)`
> - `MTLCommandQueue 功能`

- **Objective-C Language Server 受到关注**：一位成员发现 [Objective-C language server](https://github.com/MaskRay/ccls) 非常有用，并指出可以通过 `brew install ccls` 安装，且在 VSCode 中运行良好。
  
  - 另一位用户确认他们一直在将其用于 C 和 C++，效果不错。
- **探索 PyTorch 的内存管理**：关于在 Apple Silicon 上为 PyTorch 使用统一内存，以及 Tensor 默认是否在私有模式下分配的问题引起了讨论。
  
  - 一位成员对使用 `at::from_blob()` 处理自定义缓冲区时可能出现的内存管理问题表示担忧。
- **澄清 PyTorch 中的 MPS Stream**：一位成员指出 MPSStream 是 `id<MTLCommandQueue>` 和 `dispatch_queue_t` 的元组，说明了其在管理命令队列中的功能。
  
  - 进一步的探索确认了 MPS stream 传达了任务执行的概念，并指出多个队列可以在 GPU 上并发执行。

**提到的链接**：

- [GitHub - MaskRay/ccls: C/C++/ObjC language server supporting cross references, hierarchies, completion and semantic highlighting](https://github.com/MaskRay/ccls)：支持交叉引用、层级结构、补全和语义高亮的 C/C++/ObjC 语言服务器 - MaskRay/ccls
- [pytorch/aten/src/ATen/mps/MPSStream.mm at d1027c2be6ad2ee8c9c50fa83293babd05cb6a2c · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/d1027c2be6ad2ee8c9c50fa83293babd05cb6a2c/aten/src/ATen/mps/MPSStream.mm#L17-L33)：Python 中具有强大 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch
- [pytorch/aten/src/ATen/mps/MPSAllocator.h at 3f3b692a00737c54a3e2948db5db493d40119854 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/3f3b692a00737c54a3e2948db5db493d40119854/aten/src/ATen/mps/MPSAllocator.h#L124-L125.)：Python 中具有强大 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1297392869404966923) (12 messages🔥):

> - `地理空间数据的人工数据标注`
> - `数据标注平台`
> - `视觉数据的离岸供应商`
> - `Scale AI 及其替代方案`

- **寻求人工数据标注员**：一位成员正在为**天气雷达数据**寻求人工标注员的建议，特别是针对地理空间和视觉语言标注。
  
  - *哪些平台最好？*
- **对不同平台的考量**：成员们讨论了各种数据标注平台，重点提到了 **Scale AI**、**Surge**、**Mechanical Turk** 和 **Prolific**。
  
  - 一位成员指出了这些平台针对不同数据类型的**优缺点**。
- **Natolambert 关于数据标注的参考资料**：Natolambert 引用了两篇讨论 **Scale AI** 及其在人工数据和 RLHF 技术市场中角色的文章，暗示了该领域**日益增长的需求**。
  
  - 他分享了更多细节链接，包括 [Scale AI 的商业模式](https://www.interconnects.ai/p/ai-data-foundry)。
- **雷达数据推荐离岸供应商**：一位成员建议不要在简单的雷达数据任务中使用主要的 **GenAI** 供应商，而是建议选择**离岸供应商**作为更好的方案。
  
  - 他们提到 **Mechanical Turk** 在有**人工指导**的情况下也可以工作，并询问了所需数据的**容量**。

**提到的链接**：

- [Futures of the data foundry business model](https://www.interconnects.ai/p/ai-data-foundry)：Scale AI 的未来与语言模型性能的进一步扩展。Nvidia 可能也会从数据市场中拿走所有利润。
- [Alignment-as-a-Service: Scale AI vs. the new guys](https://www.interconnects.ai/p/alignment-as-a-service)：Scale 通过销售 RLHF 数据每年赚取超过 7.5 亿美元，谁会来分一杯羹？

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1296963148191895664) (74 条消息🔥🔥):

> - `表情符号请求`
> - `RLHF 书籍编写进展`
> - `OpenAI Token 使用情况`
> - `CARDS 对齐方法`
> - `深色模式讨论`

- **Nato 接收表情符号请求**：Nato 确认他正在接收表情符号请求，并分享了一个用于提交的频道链接，引发了成员们的幽默回应。
  
  - 消息中包含了各种表情符号回应，凸显了聊天中轻松活泼的氛围。
- **RLHF 书籍进展**：Nato 宣布他正在编写一本关于 RLHF (Reinforcement Learning from Human Feedback) 的书籍，目标是在年底前出版纸质版。
  
  - 他分享了[书籍网站](https://rlhfbook.com/)，并指出他在不进行过度检查的情况下进行写作并拥抱社区参与的这种方法的重要性。
- **解码 OpenAI Token 行为**：Nato 回应了一篇讨论 OpenAI 模型 Token 使用情况的推文，重点关注推理 Token 似乎都是 64 的倍数。
  
  - 他推测报告的推理 Token 可能是一个近似值，并讨论了他的相关博客文章读者较少的问题。
- **CARDS 解码对齐方法介绍**：一位成员介绍了一种名为 CARDS 的新方法，据报道该方法可以在不重新训练模型的情况下加速文本生成并确保高奖励结果。
  
  - 该方法使用了分段级拒绝采样（segment-level rejection sampling），并为感兴趣的读者提供了[相关论文](https://arxiv.org/abs/2406.16306)链接。
- **深色模式讨论**：成员们就不同平台上深色模式下 Logo 的可见性进行了轻松的交流，分享了各自的体验。
  
  - Nato 幽默地向参与者保证，在包括 RLHF 书籍在内的各种项目中保持工作与生活的平衡非常重要。

**提到的链接**：

- [Ruqi Zhang (@ruqi_zhang) 的推文](https://x.com/ruqi_zhang/status/1810690177498595761)：介绍 CARDS，一种用于 LLM 解码时对齐的新方法：✨文本生成速度提升 5 倍，在 GPT-4/Claude-3 评估中获得 99% 的胜平率 ✨可证明能生成高奖励、高概率的文本 ✨无需重新训练...
- [Yuntian Deng (@yuntiandeng) 的推文](https://x.com/yuntiandeng/status/1848421766093255027)：OpenAI o1 使用了多少推理 Token？结果发现它们几乎总是 64 的倍数（在收集的 10 万轮对话中占比超过 99%）🤔 难道模型只使用 64 Token 的倍数吗...
- [NaNoWriMo](https://nanowrimo.org/)：未找到描述
- [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html)：未找到描述
- [The Basics of Reinforcement Learning from Human Feedback](https://rlhfbook.com/)：RLHF 基础知识
- [GitHub - natolambert/rlhf-book: Textbook on reinforcement learning from human feedback](https://github.com/natolambert/rlhf-book)：关于 RLHF 的教科书 - natolambert/rlhf-book

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1296932075412131962) (16 条消息🔥):

> - `Interconnects 表情符号`
> - `Discord 支持`
> - `表情符号上传`
> - `AI 公司 Logo`

- **寻求 Interconnects 表情符号**：成员们讨论了如何向服务器添加 **Interconnects 表情符号**，并建议添加各种 **AI 公司 Logo** 以及像更多 **snail bot** 内容之类的梗图。
  
  - 一位成员幽默地建议对尚未加入 Discord 的用户提高价格，以此强调社区参与度。
- **Discord 工作人员的潜在支持**：一位成员开玩笑说，如果他们搞不定表情符号设置，就向 Discord 工作人员寻求帮助，表现出解决问题的信心。
  
  - 另一位成员根据他们在表情符号和音板上传方面的经验，确认这是一项简单的任务，表示“这应该不会太难”。
- **表情符号的美学改进**：有人请求提供**兼容深色模式的 OpenAI Logo** 以及 **Interconnects** 表情符号的深色版本，以提升美感。
  
  - 此外，还有人建议改进某些 Logo 的 **Alpha 通道**，重点是增强可见性。

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1297964262446071920) (2 条消息):

> - `LLM Reasoning Debate` (LLM 推理辩论)
> - `OpenAI's GPT Releases` (OpenAI 的 GPT 发布)
> - `Training Data Limitations` (训练数据限制)

- **LLM 推理辩论升温**：最近的一篇文章强调了关于大语言模型 (LLMs) 是否能有效推理的激烈辩论，特别是由于 OpenAI 最新发布的 **GPT-4o** 和 **GPT-o1** 所引发的。
  
  - 核心问题仍然在于：这些模型是运用了*实际推理*，还是仅仅模仿了它们在训练数据中见过的模式，这可能会限制它们解决问题的能力。
- **OpenAI 发布 GPT-4o 和 GPT-o1**：2024 年 5 月，OpenAI 推出了 **GPT-4o**，声称它可以实时跨音频、视觉和文本进行推理；随后推出了 **GPT-o1** 模型，该模型以在重推理基准测试中的准确表现而闻名。
  
  - 这些进展进一步推动了关于 LLMs 的真实推理能力与习得行为之间关系的讨论。
- **对问题解决能力的担忧**：辩论质疑了像 GPT-4o 和 o1 这样的 LLMs 是在真正解决问题，还是依赖于训练数据中的模式，这可能会阻碍其在陌生任务上的表现。
  
  - 这意味着理解这一区别对于评估 AI 推理的未来发展至关重要。

**提到的链接**：[The LLM Reasoning Debate Heats Up](https://open.substack.com/pub/aiguide/p/the-llm-reasoning-debate-heats-up?r=68gy5&utm_medium=ios)：三篇近期论文探讨了大语言模型中推理和问题解决的鲁棒性。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1296920128641962045) (98 条消息🔥🔥):

> - `Performance of RTX GPUs` (RTX GPU 性能)
> - `Creating Images with Different Perspectives` (创建不同视角的图像)
> - `Using Loras in Prompts` (在提示词中使用 Loras)
> - `Stable Diffusion API Access Issues` (Stable Diffusion API 访问问题)
> - `Need for Assistance in Image Editing` (图像编辑协助需求)

- **RTX 3090 表现不及预期**：尽管预期性能会有所提升，但一位用户报告称其 RTX 3090 仅达到 **每秒 3.5 次迭代 (it/s)**，与其之前的 RTX 3060 速率相比令人惊讶。
  
  - 建议包括确保 Web UI 已更新并重新安装驱动程序，这可能有助于优化性能。
- **改变图像视角的挑战**：一位用户询问如何为现有的建筑照片创建不同的视角，并在新草图中保留颜色和物体，但由于照片限制而面临困难。
  
  - 成员们讨论了潜在的解决方案，包括需要更多的无人机拍摄镜头以及训练一个 Lora 来学习特定建筑。
- **图像生成中 Loras 的问题**：一位用户遇到了一个问题，即在生成图像时，使用多个 Loras 会导致错误消息，提示某些 Loras 未找到。
  
  - 其他人参与了讨论，提供了潜在的故障排除方法或更好的提示词管理方式来解决冲突。
- **访问 Stability.ai API 页面**：一位用户对访问 Stability.ai API 参考页面表示担忧，提到该页面似乎已宕机。
  
  - 回复指出，用户需要联系客户服务寻求支持，因为社区并不管理网站或 API。
- **图像编辑协助需求**：用户表达了在图像编辑以及将 AI 工具整合到工作流中（特别是针对商业项目）的需求。
  
  - 一位用户通过私信提供了帮助，展示了社区内的协作氛围。

**提到的链接**：

- [Stability AI - Developer Platform](https://platform.stability.ai/docs/api-reference)：未找到描述
- [update readme · alimama-creative/FLUX.1-Turbo-Alpha at b2db8dc](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/commit/b2db8dcbd15fb095cffd8ab530499e47883466e7)：未找到描述
- [GitHub - chengzeyi/stable-fast: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs.](https://github.com/chengzeyi/stable-fast)：针对 NVIDIA GPU 上 HuggingFace Diffusers 的最佳推理性能优化框架。- chengzeyi/stable-fast

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1296919862916153465) (7 条消息):

> - `3天黑客松`
> - `LlamaParse Premium`
> - `自动化销售外联的 Agentic 系统`
> - `高级 RAG 工作流`
> - `多模态 RAG 流水线`

- **3天黑客松交付 45 个项目**：最近举行的 **3天黑客松** 吸引了超过 **500** 名参与者，并在周末结束时产生了 **45 个精彩项目**。查看[宣布获胜者的博客文章](https://t.co/v7F8b0qedF)了解详情。
  
  - 获胜者还将提供详细介绍其项目的客座博客文章，引发了社区的热烈讨论。
- **LlamaParse Premium 获得好评**：在推出 **LlamaParse Premium** 后，用户对其改进的解析能力表现出极大的热情。一篇深入的 [LinkedIn 帖子](https://t.co/NeAvIlfIP3)展示了它如何超越前代产品。
  
  - 介绍 **LlamaParse** 的原始帖子也可以在[这里](https://t.co/pDPHxcYQeb)找到。
- **自动化销售外联变得更智能**：由 **Calsoft_Data** 撰写的博客探讨了一种**受限的 Agentic 架构**，该架构可自动执行销售外联任务，减少手动流程所花费的时间。这种方法是研究潜在客户和创建个性化电子邮件的有效解决方案。
  
  - 您可以在[这里](https://t.co/ziCb6UkcRd)阅读更多关于这一创新系统的信息。
- **极速 RAG 工作流教程**：由 **Plaban Nayak** 编写的教程介绍了如何使用 **GroqInc** 构建**全异步 RAG 工作流**，优化了重排序（reranking）和合成（synthesis）。这为处理数据流程提供了显著的速度提升。
  
  - 教程可以从[这里](https://t.co/r6ag69r5uu)访问。
- **高效的多模态 RAG 流水线设置**：由 **fahdmirza** 编写的教程演示了如何建立一个**高级多模态 RAG 流水线**，以高效地索引幻灯片等复杂文档。该过程被简化到“开箱即用”的程度，为开发节省了时间。
  
  - 在[这里](https://t.co/d8IxLU8NKk)了解更多关于这一直观设置的信息。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1296922143153721356) (81 条消息🔥🔥):

> - `LlamaIndex 中的 Ollama 集成`
> - `评估检索方法`
> - `工作流中的事件流`
> - `文档摘要技术`
> - `ML 模型的部署平台`

- **在 LlamaIndex 中集成 Ollama**：一位用户分享了他们在 `npx create-llama` 中使用 **Ollama** 的配置，但尽管设置正确，仍遇到了 OpenAPI 密钥弹窗问题。
  
  - 另一位成员建议修改后端源码，以成功加载 **Ollama** LLM 和 embeddings。
- **评估混合检索准确性**：讨论中提到了评估结合 `BM25Retriever` 和 `VectorIndexRetriever` 的混合检索器的方法，强调了基准数据集（ground truth datasets）的重要性。
  
  - 几位成员建议使用 LLM 来评估检索相关性，或确定问题与文档的映射关系以进行有意义的评估。
- **流式响应与工具调用 (Tool Calls)**：一位用户注意到不同模型在检测工具调用时存在不一致性，OpenAI 可以立即检测到，而其他模型则有延迟。
  
  - 有人建议在工作流中使用事件流（event streaming）作为解决方案，从而在响应生成过程中实现更高效的分块处理。
- **索引中的文档摘要**：成员们讨论了是否将文档摘要纳入检索系统，共识倾向于使用 `DocumentSummaryIndex` 以提高效率。
  
  - 强调了保持高质量摘要的重要性，因为劣质摘要可能会导致幻觉响应。
- **API 托管建议**：对于部署针对特定数据集使用模型的 API，建议包括 **AWS**、**Azure** 和 **GCP** 等托管解决方案。
  
  - 提出了对平台安全性的担忧，特别是关于 **Hugging Face** 的讨论，引发了对各种部署选项有效性的探讨。

- [Google Colab](https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing): 未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): 未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): 未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): 未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): 未找到描述
- [Starter Tutorial (OpenAI) - LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/): 未找到描述
- [SimpleDirectoryReader - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/): 未找到描述
- [Qdrant Vector Store - Metadata Filter - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/#qdrant-vector-store-metadata-filter): 未找到描述
- [rsrohan99 - Overview](https://github.com/rsrohan99): rsrohan99 拥有 13 个公开仓库。在 GitHub 上关注他们的代码。
- [no title found](https://docs.llamaindex]): 未找到描述
- [llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py at 227145fb94fcaa4da02d559fc81843fcb2af2b57 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/227145fb94fcaa4da02d559fc81843fcb2af2b57/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py#L314): LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [Survey for NEO clouds](https://docs.google.com/forms/d/e/1FAIpQLSfhn89gy8WSViYJoWT9N3MbD63dNl_8eyoJcBqzUXYni6PXog/viewform?usp=sf_link): 我们正在构建 Neo Clouds，这是一个提供强大计算资源的云平台，用于运行需要高计算能力的应用程序。为了确保我们能满足您的需求，我们希望...
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): 1-bit LLM 的官方推理框架。通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。
- [llama_index/llama-index-core/llama_index/core/vector_stores/types.py at 227145fb94fcaa4da02d559fc81843fcb2af2b57 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/227145fb94fcaa4da02d559fc81843fcb2af2b57/llama-index-core/llama_index/core/vector_stores/types.py#L63): LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [update to use workflows by logan-markewich · Pull Request #4 · rsrohan99/rag-stream-intermediate-events-tutorial](https://github.com/rsrohan99/rag-stream-intermediate-events-tutorial/pull/4): 当时正在帮助一位用户，最后将这个示例转换为了使用 workflows！请随意合并或忽略此请求 😁
- [GitHub - logan-markewich/rag-stream-intermediate-events-tutorial: Tutorial on how to properly send intermediate LlamaIndex events to vercel ai sdk via server-sent events during RAG.](https://github.com/logan-markewich/rag-stream-intermediate-events-tutorial/tree/master): 关于如何在 RAG 过程中通过 server-sent events 将 LlamaIndex 中间事件正确发送到 vercel ai sdk 的教程。 - logan-markewich/rag-stream-intermediate-events-tutorial
- [GitHub - rsrohan99/llamaindex-workflow-streaming-tutorial](https://github.com/rsrohan99/llamaindex-workflow-streaming-tutorial): 通过在 GitHub 上创建账号来为 rsrohan99/llamaindex-workflow-streaming-tutorial 的开发做出贡献。
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/): 未找到描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1297262312704577659) (3 条消息):

> - `Multilingual Embedding Models`
> - `Creating API for Proprietary Materials`

- **寻找多语言 Embedding 解决方案**: 一位成员正在开发一个利用多种语言（EN, JP, ID, VI, TH）PDF 的 **RAG 系统**，但在使用各种**开源**和**闭源** Embedding 模型时未获得理想效果。
  
  - 另一位成员推荐了 **aBSE** (Language-agnostic BERT Sentence Embedding) 模型，作为获得更好多语言效果的潜在解决方案。
- **关于为专有内容创建 API 的指导**: 一位初学者正在寻求关于创建 **API** 的指导，该 API 能够根据个人笔记或书籍等专有材料回答问题。
  
  - 他们请求关于合适的**机器学习技术**、**托管平台**以及**数据集存储**建议的见解。

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1297133182079991938) (55 条消息🔥🔥):

> - `Multihead Attention 标准化`
> - `Tinygrad 开发更新`
> - `WebGPU 支持`
> - `本地 LLM 使用趋势`
> - `Benchmark CI 测试`

- **Multihead Attention 标准化的有效性**：一名成员询问关于 **标准化 Multihead Attention** 的讨论是否仍然具有相关性和有效性。
  
  - 该询问表明 Tinygrad 社区对优化 Attention 机制有着持续的兴趣。
- **Tinygrad 寻求竞争力**：George Hotz 宣布合并了 **GGUF 加载支持**，希望 Tinygrad 在更有效地 **运行 LLM** 方面能与其他框架竞争。
  
  - 他鼓励使用 Tinygrad 的库和应用程序的开发者站出来，并强调了超越 **Ollama** 和 **GGML** 等竞争对手的愿景。
- **本地 LLM 使用见解**：成员们提到使用 **Llama.cpp** 和 **ExLlamaV2** 在本地运行模型，其中 ExLlamaV2 提供了更简单的设置，且性能与 Nvidia 的 **TensorRT-LLM** 相当。
  
  - 讨论表明，用户倾向于使用这些工具在个人设备上进行高效的模型部署。
- **WebGPU 支持的重要性**：George Hotz 强调了开发过程中 **WebGPU 支持** 的重要性，并提到了社区在这方面的努力。
  
  - 另一名成员报告了在 **threefry** 算法上的进展，预计后续的阻碍会减少。
- **用于 LLM 鲁棒性的 Benchmark CI**：George 强调了在 **Benchmark CI** 中进行彻底测试的必要性，因为在本地模型执行中存在多种潜在的 GPU 故障点。
  
  - 他强调需要覆盖各种边缘场景，以确保并发运行多个模型时的鲁棒性。

 

**提到的链接**：[Big graph · Issue #7044 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7044)：LazyBuffer.view 变为 UOps.VIEW #7077 #7078 #7007 #7090 big graph SINK #7122, #7178, #7170 #7134, #7175 #7132, #7188 #7190 #7149 ASSIGN 且 toposort 变为 graph_rewrite 以决定何时 realize ...

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1296961952597147648) (35 条消息🔥):

> - `FrozenBatchNorm2d`
> - `从 .safetensors 自动生成类`
> - `Tinygrad 中的 Action Chunking Transformers`
> - `在 Tinygrad 中实现 nonzero`
> - `Tinygrad 中用于默认浮点数的 Context Manager`

- **理解 FrozenBatchNorm2d**：一位用户询问了 **FrozenBatchNorm2d** 在某些网络架构中的用途，对其必要性和功能提出了疑问。
  
  - 他们分享了示例代码，并对 `__call__()` 函数在此上下文中的工作方式表示困惑。
- **关于从 .safetensors 自动生成类的查询**：用户讨论了从 **.safetensors** 文件自动生成类的可能性，但由于缺乏计算描述，发现这具有挑战性。
  
  - 一位用户提到他们成功运行了一个模型，感到非常兴奋，并正在寻求一种方法来为未来的用户提供更简便的转换方式。
- **Action Chunking Transformers 的进展**：一位用户确认他们在 Tinygrad 中实现的 **Action Chunking Transformers** 已经可以运行，目前正在使用不同的数据集进行测试。
  
  - 他们分享了一个 GitHub 链接，指向其虽然凌乱但可运行的 notebook，并计划很快优化代码库。
- **在 Tinygrad 中实现 nonzero 功能**：讨论了如何在 Tinygrad 中复制 **torch.nonzero** 的功能，特别是针对邻接矩阵和索引。
  
  - 有人建议了一些替代方案，包括使用带有 `where` 的布尔索引或将索引转换为整数，但在兼容性方面仍存在挑战。
- **使用 Context Manager 更改默认浮点数**：一位用户询问如何使用 Context Manager 或装饰器在 Tinygrad 的函数中更改默认浮点数，并引用了现有文档。
  
  - 他们在尝试设置 `DEFAULT_FLOAT` 时遇到了 KeyError，从而开始探索该变量是如何从环境中确定的。

**提到的链接**：

- [George Hotz | Programming | MNIST classifier from numpy scratch! | Science & Technology](https://www.youtube.com/watch?v=JRlyw6LO5qo)：直播日期：2020年10月17日。直播聊天已添加为字幕/CC - 英语 (Twitch Chat)。直播标题：从零开始用 numpy 编写 MNIST 分类器！源文件：- ht...
- [[NOMERGE] Llama: download tiny llama weights by default by jla524 · Pull Request #7173 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7173)：未找到描述
- [act-tinygrad/modeling_act.ipynb at main · mdaiter/act-tinygrad](https://github.com/mdaiter/act-tinygrad/blob/main/modeling_act.ipynb)：Tinygrad 中的 Action Chunking Transformers。通过在 GitHub 上创建账号来为 mdaiter/act-tinygrad 的开发做出贡献。
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)：未找到描述
- [Environment Variables - tinygrad docs](https://docs.tinygrad.org/env_vars/)：未找到描述
- [tinygrad/tinygrad/dtype.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/dtype.py#L121))：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - tinygrad/tinygrad

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1297081234736021534) (35 条消息🔥):

> - `Mystery model` (神秘模型)
> - `Agent assist APIs` (Agent 辅助 API)
> - `Connection issues with Google Drive` (Google Drive 连接问题)
> - `Community introductions` (社区自我介绍)
> - `General chat discussions` (常规聊天讨论)

- **神秘模型引发热议**：一名成员提到一个具有 **8k** 上下文的**神秘模型**已上线，引发了社区内的强烈好奇。
  
  - *正在酝酿一些新东西...* 成员们正渴望与这个 [神秘机器人](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771) 进行互动。
- **关于 Agent 辅助 API 的咨询**：一位成员询问 **Cohere** 是否提供 **Agent 辅助 API**，用于根据提供的信息生成回复。
  
  - 另一位成员将该咨询引导至特定频道，以便对该话题进行深入讨论。
- **Google Drive 连接寻求帮助**：一位用户报告了连接 **Google Drive** 时遇到的问题，收到了“应用被屏蔽”的消息，并寻求规避方案。
  
  - 一位社区成员建议提供更多上下文和截图，以便更有效地协助排查问题。
- **新成员自我介绍**：几位新成员介绍了自己，表达了参与 **Cohere** 社区互动的兴趣。
  
  - 讨论话题包括潜在的协作项目和社区参与。
- **频道使用提醒**：一位成员提醒其他人，discussions 频道旨在进行常规聊天，而具体查询应定向到其他频道。
  
  - 此举旨在保持频道井然有序，专注于广泛的讨论而非具体的技术问题。

**提到的链接**：[来自 UltraIA (@Ultra_IA) 的推文](https://x.com/Ultra_IA/status/1847821253476008227)：LOL

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771) (2 条消息):

> - `Aya Community Project` (Aya 社区项目)
> - `Cohere Developer Office Hours` (Cohere 开发者 Office Hours)

- **Aya 社区启动秘密项目**：**Aya Community** 邀请用户通过向包括 **Whatsapp** 和**本地免费电话**在内的各种国际号码发送短信，协助测试一个新的**语言连接项目**。
  
  - 鼓励参与者提供关于**所遇问题**的反馈，并加入 **Aya Discord** 进行进一步讨论，同时提醒对号码保持保密。
- **明天举行 Cohere 开发者 Office Hours**：Cohere 将于明天 **1:00 PM ET** 举行 **Developer Office Hours**，届时将有团队成员带来的现场演示以及关于新发布和即将发布内容的见解。
  
  - 参与者可以通过提供的链接加入讨论：[Cohere 开发者活动](https://discord.com/events/954421988141711382/1285304800400904344/1297967638118400000)。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1297019714811334676) (35 条消息🔥):

> - `OpenRouter Benefits` (OpenRouter 的优势)
> - `Cohere API Usage` (Cohere API 使用)
> - `Langchain SSL Issues` (Langchain SSL 问题)

- **OpenRouter 提供灵活的 API 切换**：成员们讨论了使用 **OpenRouter** 的优势，指出当某个提供商宕机时，它能够无缝切换 **API** 提供商。
  
  - *说实话，并非所有的 API 提供商都稳定*，这增强了 OpenRouter 的吸引力。
- **Cohere API 及其局限性**：一位成员咨询了 **Cohere API**，表示对它是否包含 **Reranker** 和 **embed-v3** 等特定模型感兴趣。
  
  - 成员们对直接使用 **Cohere API** 提出了担忧，因为其**闭源**特性需要大量的额外实现工作。
- **Langchain SSL 错误很常见**：一位用户在尝试绕过公司网络安全设置时，遇到了 **Langchain** 的 SSL 错误。
  
  - 另一位成员建议，*export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1* 可能是解决该问题的一个潜在方案。

**提到的链接**：[Chat — Cohere](https://docs.cohere.com/reference/chat)：根据提供的对话由模型生成消息。要了解更多关于 Chat API 功能的信息，请参阅我们的 [文本生成指南](https://docs.cohere.com/v2/docs/chat-api)...

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1297953984375554188) (6 条消息):

> - `API read timeout issues` (API 读取超时问题)
> - `Getting citations from the API` (从 API 获取引用)
> - `Chat API documentation` (Chat API 文档)

- **反馈 API 读取超时问题**：一名成员反映在周末期间使用 API 时遇到了 **read timeout errors**。
  
  - *sssandra* 确认他们已向团队反馈此问题，以便进一步调查。
- **引用功能开箱即用**：*sssandra* 澄清引用 (Citations) 是 API 的内置功能，并引导用户查看 Chat API 文档以获取更多信息。
  
  - 他们强调应查看 [Retrieval Augmented Generation 文档](https://docs.cohere.com/v2/docs/retrieval-augmented-generation-rag)，以了解如何有效使用引用的详细信息。
- **分享有用的 API 链接**：提供了 [Chat API 文档](https://docs.cohere.com/reference/chat) 和 [Migration Guide](https://docs.cohere.com/v2/docs/migrating-v1-to-v2) 的链接，以帮助用户熟悉 API。
  
  - 这些资源概述了基本的使用说明，包括如何处理 API 请求和引用。

**提到的链接**：

- [Chat — Cohere](https://docs.cohere.com/reference/chat)：根据提供的对话生成模型消息。要了解更多关于 Chat API 的功能，请参考我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/chat-api...
- [Retrieval Augmented Generation (RAG) — Cohere](https://docs.cohere.com/v2/docs/retrieval-augmented-generation-rag)：使用 Retrieval Augmented Generation 和 Cohere 的 Chat API，结合外部数据和行内引用生成文本。
- [Documents and Citations — Cohere](https://docs.cohere.com/docs/documents-and-citations)：该文档介绍了 RAG 作为一种通过提供上下文源材料来改进语言模型响应的方法。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1297862482924142613) (2 条消息):

> - `JavaScript Implementations` (JavaScript 实现)
> - `Direct API Requests` (直接 API 请求)

- **令人印象深刻的 JavaScript 实现**：一名成员评论道：“*非常令人印象深刻！而且全部是用 .js 编写的！*”，表达了对一个利用 JavaScript 实现其功能的项目的兴奋。
  
  - 这突显了利用 **JavaScript** 构建高效 AI 应用的日益增长的趋势。
- **直接 API 通信**：另一名成员确认，只需一个 **API key**，即可直接向 AI 提供商发起请求，无需 Proxy。
  
  - 这种方法简化了交互并减少了开发者的依赖项。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1296912694687039568) (44 条消息🔥):

> - `Liger Kernel 安装`
> - `Axolotl 层冻结问题`
> - `Spectrum 的 SNR 结果`
> - `AGI House 活动`

- **Liger Kernel 安装指南**：为了实现 **VRAM 节省**，安装 Liger Kernel 非常简单：只需使用 `pip install liger-kernel` 并根据频道中分享的内容调整配置。
  
  - 用户指出 Liger 促进了全量微调（full finetuning），并受益于现有的 Flash Attention 功能。
- **Axolotl 中的层冻结 Bug**：最新版本的 Axolotl 似乎存在一个 **bug**，导致用户无法冻结/解冻层，而该功能此前是可以正常工作的。
  
  - 该问题正在调查中，社区成员正请其他人确认最近的更改，并检查 `src/axolotl/integrations/spectrum/model_snr_results` 目录。
- **Spectrum SNR 结果讨论**：围绕模型的 top fractions 和 **SNR 结果** 进行了讨论，确认了 Qwen 模型的计算结果正确。
  
  - 成员强调 Spectrum 集成需要预计算的 **SNR JSON 文件** 才能正常运行。
- **AGI House 即将举行的活动**：AGI House 宣布了两项激动人心的活动：11 月 2 日的 **Think Slow & Think Deep** 黑客松，以及 11 月 9 日的 **AI Agent Werewolf Tournament**（狼人杀锦标赛）。
  
  - 狼人杀锦标赛提供丰厚的现金奖励，旨在汇聚创新设计师与 AI Agent 进行竞技。

**提到的链接**：

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102))：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109))：未找到描述
- [spectrum/model_snr_results at main · cognitivecomputations/spectrum](https://github.com/cognitivecomputations/spectrum/tree/main/model_snr_results)：通过在 GitHub 上创建账户为 cognitivecomputations/spectrum 的开发做出贡献。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1297939346070179871) (2 条消息):

> - `Qwen2 对 DoRA/QDoRA 的支持`
> - `Answer.AI 的 QDoRA 仓库`

- **Qwen2 DoRA 支持请求**：一位成员正在寻找任何现有的 **Qwen2** 支持 **DoRA/QDoRA** 的开发进展，并注意到频道内缺乏相关讨论。
  
  - 他们引用了 [**Answer.AI 的 QDoRA 仓库**](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model) 作为实现的潜在起点。
- **Qwen2 DoRA 无活跃开发**：另一位成员确认目前没有针对 **Qwen2** 的 **DoRA** 支持的活跃分支。
  
  - 他们鼓励继续推进实现工作，并以友好的语气表达了乐观态度。

 

**提到的链接**：[GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model.)：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账户为 AnswerDotAI/fsdp_qlora 的开发做出贡献。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1297885140416204902) (1 条消息):

> - `训练特定领域 LLM`
> - `微调 LLM`
> - `Instruct 模型`

- **为特定领域数据训练 LLM**：一位成员正在致力于为 **数学**、**法律** 和 **金融** 等特定领域数据 **训练和微调 LLM**。
  
  - 他们表示有兴趣 **讨论** 微调像 **llama-70b-instruct** 这样已有的 instruct 模型，而不是从非 instruct 模型开始的好处。
- **LLM 微调策略**：对话强调了从在领域指令数据集上 **微调基础非 instruct 模型** 开始的方法。
  
  - 该成员指出，通过在现有 instruct 模型之上进行微调以增强性能，可以改进这种方法。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1296946515037917184) (38 条消息🔥):

> - `Meta 的 FAIR 研究`
> - `Torch 中的 Attention Mask 问题`
> - `Flex Attention 的挑战`
> - `PyTorch 中的性能警告`
> - `Mask 构建讨论`

- **Meta 的 FAIR 团队推动高级机器智能 (AMI)**：Meta 的 FAIR 团队正在分享他们实现 **高级机器智能 (AMI)** 以提升生产力和创新的目标，正如马克·扎克伯格最近的 [公开信](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) 中所提到的。他们在 **开放科学** 和可复现性方面与 AI 社区合作已超过十年。
  
  - 这一研究工作也引发了关于 **Lingua** 等类似工具是否与 **Torchtune** 具有可比性的讨论。
- **Attention Mask 构建与 Flex Attention**：成员们讨论了 Attention 机制中 **Mask 构建** 的复杂性，特别是根据 Attention 类型需要不同的 Block Mask，正如最近实现中所遇到的挑战。有人建议在 Forward Pass 期间处理 Mask 实例化，以简化 **collate** 过程。
  
  - 这强调了在解决 **Packed Datasets** 问题和自定义 Collate 需求时，保持实现整洁的重要性。
- **PyTorch 中的性能警告**：用户在某些数据类型上遇到了与 **cuDNN SDPA** 相关的警告，这引发了关于底层性能问题和潜在修复方案的疑问。使用不同 Kernel 进行测试可能有助于评估性能影响，特别是在 **PyTorch GitHub** 最近报告的问题背景下。
  
  - 讨论强调了可能向 **PyTorch core** 提交 Issue 的努力，以解决持续存在的警告及其影响。
- **关于 Document ID 和 Packed Datasets 的讨论**：对话涉及在构建 **PackedDataset** 时是否可以预计算 **Document ID**，这可能会提高处理 `packed=True` 工作负载的效率。这为未来的实现提出了一种优化策略。
  
  - 这种策略旨在整合 Mask 生成逻辑，从而在 Attention 机制处理中实现更好的性能和更简洁的代码路径。
- **对协作和文档的共识**：参与者一致认为有必要在 **GitHub** 上记录关于 Attention 问题和潜在解决方案的持续讨论，以防止重要的见解丢失。这促成了一个总结 Mask 构建和 Attention Dispatch 问题关键点的 Issue 的创建。
  
  - 协作的重要性得到了共鸣，特别是改进文档如何能简化开发流程中的未来过渡。

**提到的链接**：

- [未找到标题](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/)：未找到描述
- [pytorch/aten/src/ATen/native/cudnn/MHA.cpp at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/MHA.cpp#L677)：Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch
- [torchtune/torchtune/modules/attention_utils.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention_utils.py#L133)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。
- [Mask construction & attention dispatch issues and possible ideas to allow for more models · Issue #1870 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1870)：自 Torch 2.5.0 以来，无法在 packed=True 且 attention dropout > 0.0 的情况下进行训练，因为如果 Flex 可用，padded_collate_packed 会自动选择构建 BlockMasks...
- [[Bug] Unusual CPU overhead of SDPA call on H100 on torch nightly · Issue #1652 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1652)：已识别的问题：当上下文长度改变时，cuDNN SDPA JIT 会重新编译。这导致不使用 Packing 的训练会不断重新编译，从而产生观察到的 500ms 开销...
- [F.sdpa stride bug](https://gist.github.com/mirceamironenco/0d39d1976daa62fdded02a76ef826980)：F.sdpa stride bug。GitHub Gist：即时分享代码、笔记和代码片段。
- [pytorch/torch/nn/attention/flex_attention.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L873)：Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1297929204259750030) (1 条消息):

> - `v0.4.0 代码冻结`
> - `v0.4.0 中的新功能`
> - `发布时间线`

- **v0.4.0 代码冻结倒计时开始！**：距离 **10 月 29 日** 的 **v0.4.0 代码冻结** 仅剩 **8 天**，开发者们正急于完成剩余任务。
  
  - 准备工作至关重要，因为 [*v0.4.0 Tracker*](https://github.com/pytorch/torchtune/issues/1747) 列出的预计发布日期为 **11 月 5 日**。
- **v0.4.0 计划推出的新功能**：针对即将发布的版本讨论的新功能包括 Issue **#1645**、**#1847** 和 **#1835** 中的亮点。
  
  - 贡献者 @felipemello1 和 @Optimo 正在带头推进，确保为用户带来令人兴奋的更新。

**提到的链接**：[v0.4.0 Tracker · Issue #1747 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1747)：预计发布日期：11 月 5 日，星期二；预计分支切割日期（即代码冻结）：10 月 29 日，星期二；发布负责人：@joecummings；新功能：#1645 #1847 (@felipemello1) #1835 (@Optimo...

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1296935286797893663) (11 条消息🔥):

> - `Pydantic All-in-One`
> - `DSPy GPTs`
> - `AI Agents 生产落地活动`
> - `串流与机器人通知`
> - `HotpotQA 替代历史生成器`

- **Pydantic All-in-One 直播**：一名成员在 [pydantic-all-in-one](https://discord.com/channels/1161519468141355160/1161519469777133580) 上开启了直播，分享了他们在创建 Python 包和框架时的思考过程。
  
  - 他们还表示计划在直播后开发 **llmodel**。
- **DSPy GPTs 获得教程助力**：成员们讨论了制作关于如何高效利用各种 **DSPy GPTs** 的教程视频的可能性，强调了这对社区中的新手和资深用户都有好处。
  
  - 创作者同意考虑这一点，并强调了持续的社区支持。
- **“AI Agents 正从研发走向现实”活动**：一名成员宣布将于 **11 月 13 日** 举办一场虚拟活动，邀请了 Tomas Wolf 和 Nathan Benaich 等知名演讲者，重点讨论在生产环境中部署 AI Agents。
  
  - 该活动由 **Prosus AI 和 MLOps** 组织，旨在涵盖内存管理方面的挑战以及不同领域的实际应用。
- **串流更新与服务器变更**：在讨论与串流相关的通知时，**seanchatmangpt** 透露计划在 11 月前迁移到更大的服务器，并整合 **YouTube** 和 **Twitch** 功能。
  
  - 他们还对将提供实时通知的机器人表示了热情，这让社区感到兴奋。
- **HotpotQA 替代历史生成器概览**：一名成员分享了 **HotpotQA Alternate History Generator** 的概览，介绍其旨在创建合理的替代历史场景的复杂系统。
  
  - 该生成器采用先进的 **NLP 技术** 和大语言模型来生成和优化叙事。

**提到的链接**：

- [HotpotQA Alternate History Generator](https://jmanhype.github.io/HotpotQA-Alternate-History-Generator/)：未找到描述
- [AI Agents in Production - Event | MLOps Community](https://home.mlops.community/home/events/aiagentsinprod)：未找到描述
- [GitHub - seanchatmangpt/pydantic-all-in-one: All my favorite Pydantic projects connected.](https://github.com/seanchatmangpt/pydantic-all-in-one)：我所有最喜欢的 Pydantic 项目都连接在一起。通过在 GitHub 上创建账号来为 seanchatmangpt/pydantic-all-in-one 做出贡献。

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1297961078935916704) (2 条消息):

> - `LightRAG 教程`
> - `GraphRAG 观察`
> - `Ollama 集成`
> - `R2R 见解`
> - `Microsoft 的本地搜索`

- **使用 Ollama 的 LightRAG 逐步教程**：一位 YouTuber 提供了一个详细的[教程](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s)，关于如何设置和运行 **LightRAG**，这是一个结合了 **Ollama** 的检索增强生成系统。
  
  - 视频描述强调 **LightRAG** 将知识图谱与基于 Embedding 的检索相结合，以增强功能。
- **关于 LightRAG 与 GraphRAG 的 R2R 观察**：成员们分享了关于 **GraphRAG** 的 R2R 实现的见解，指出该论文的评估方法存在重大缺陷，因为它在没有适当确认的情况下以 Microsoft 的全局搜索作为基准。
  
  - 他们对由于低级和高级键（low and high-level keys）方法导致的扩展性表示担忧，并质疑了超过 **500 万 token** 的数据集的性能。
- **实现细节的论文链接偏好**：一位成员表示，相比 YouTube 视频教程，更倾向于链接讨论 **LightRAG** 仓库的原始论文。
  
  - 这种方法提供了更全面的实现细节，对于理解该技术的应用至关重要。

 

**提到的链接**：[Local LightRAG: A GraphRAG Alternative but Fully Local with Ollama](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s)：在此视频中，我们探索如何设置和运行 LightRAG——一个结合了知识图谱与基于 Embedding 检索的检索增强生成 (RAG) 系统...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1296922622759669780) (18 条消息🔥):

> - `结合 Hugging Face 模型的 DSPy`
> - `用于模型部署的 Ollama 用法`
> - `AGI House 黑客松`
> - `使用 DSPy 构建 LLM`
> - `用于模型推理的 SGLang`

- **使用 DSPy 构建 LRM**：一位社区成员正在探索如何使用 **DSPy 构建 LRM**，并指出在模型应用过程中需要高效的 token 管理。
  
  - 他们强调了像 **GPT-4** 这样的模型成本正在下降，使得开发强大的应用程序变得更加可行。
- **为 Hugging Face 模型使用 Ollama**：社区成员讨论了将 **Ollama** 作为运行微调后的 Hugging Face 模型的解决方案，并提供了更易于集成的逐步指南。
  
  - 这包括下载 GGUF 格式的模型，并配置带有 Ollama 的 **DSPy** 以获得流式体验。
- **即将举行的 AGI House 黑客松**：AGI House 宣布了两个活动，包括一个专注于 OpenAI O1 模型的**黑客松**和一个**狼人杀锦标赛**，两者都旨在培养创新的 AI 项目。
  
  - 社区成员表示有兴趣参加并组建团队，在这些活动中展示 **DSPy** 的能力。
- **Hugging Face 模型的挑战**：一位成员报告说在运行**微调后的 Hugging Face 模型**时感到困惑，在尝试与 **DSPy** 集成时经常遇到连接错误。
  
  - 其他人建议了资源和配置步骤来缓解这些问题，强调了社区的支持。
- **SGLang 用于更快推理**：分享了利用 **SGLang** 进行更快模型推理处理的建议，包括安装命令和服务器启动配置。
  
  - 社区支持提供了关于使用 **FastInfer** 进行进一步优化的见解。

**提到的链接**：

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)：未找到描述
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102))：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)：未找到描述
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109))：未找到描述
- [Huggingface | liteLLM](https://docs.litellm.ai/docs/providers/huggingface)：LiteLLM 支持以下类型的 Hugging Face 模型：
- [Drop o1 Preview, Try This Alternative](https://www.lycee.ai/blog/drop-o1-preview-try-this-alternative)：构建强大的基于 LLM 的应用程序是 token 密集型的。你通常必须计划解析和消化大量 token 以进行摘要甚至检索增强生成。即使是...

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1297282668685426828) (3 messages):

> - `Hosting models on Hugging Face`
> - `Running DSPy modules`

- **使用托管在 Hugging Face 上的本地模型**：一位用户询问如何将托管在 Hugging Face 上的本地模型作为语言模型来运行 DSPy 模块。
  
  - 讨论表明需要明确集成过程，特别是该设置所需的工具或配置。
- **关于托管模型集成的澄清**：另一位用户引用了关于托管在 Hugging Face 上的本地模型的对话，表示已提供额外支持。
  
  - 这表明在单独的消息线程中分享了更多细节以协助配置。

 

---

### **DSPy ▷ #**[**colbert**](https://discord.com/channels/1161519468141355160/1250300504462856265/1297551390901800990) (3 messages):

> - `AcgNDCG pseudo-function`
> - `BM25 retriever inquiry`
> - `AvgNDCG DSPy Metric`
> - `PATH first author outreach`

- **关于 AcgNDCG 文档检索的澄清**：一位成员询问检索器是从特定的 **10 个左右相关性判定** (J) 集合中检索文档，还是从更广泛的文档池中检索，并引用了[此处](https://arxiv.org/pdf/2406.11706)的论文。
  
  - *它是从特定列表还是整个池中检索？* 仍然是一个悬而未决的问题。
- **BM25 在模型灵活性中的作用**：讨论确认 **BM25** 作为检索器并无特殊之处，只要与正在训练的编码器不同，就可以使用任何其他检索器。
  
  - 因此，*使用不同的模型进行重排序 (reranking)* 应该是允许的。
- **AvgNDCG 指标实现**：一位成员对 **AvgNDCG** 是否在引用论文中被实现为 **DSPy Metric** 表示不确定，并表示在尝试实现之前，明确这一点会有所帮助。
  
  - *指标通常比较示例和预测*，因此确认至关重要。
- **与 PATH 第一作者合作**：一位成员鼓励就提出的问题联系 **PATH** 的第一作者以寻求帮助，并提出可以在沟通中抄送 (cc) 他。
  
  - *“我们很乐意提供帮助”* 被强调为寻求澄清的支持性邀请。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1298003872929021972) (1 messages):

> - `Lecture 7`
> - `TapeAgents framework`
> - `WorkArena++ benchmark`
> - `Nicolas Chapados`

- **关于 AI Agents 的第 7 讲就在今天**：该系列的 **第 7 讲** 定于今天 **PST 下午 3:00** 举行，可以在[此处](https://www.youtube.com/live/-yf-e-9FvOc)观看直播。
  
  - 客座讲师 **Nicolas Chapados** 和 **Alexandre Drouin** 将在演讲中讨论 **企业工作流中的 AI Agents**。
- **Introduction of the TapeAgents Framework**：本讲将介绍 **TapeAgents 框架**，该框架通过名为 Tape 的统一抽象，实现 **可恢复 (resumable)** 和 **可优化 (optimizable)** 的 Agents。
  
  - 该框架旨在显著增强使用工具的 Agent 架构的能力。
- **发布用于 Web Agents 的 WorkArena++**：**WorkArena++** 是一个新开发的 Web Agents 基准测试，专注于它们在企业环境和知识工作者任务中的表现。
  
  - 该框架跟踪 Web Agents 自主完成各种任务的进展，为该领域提出了新的挑战。
- **Nicolas Chapados 的背景**：ServiceNow 研究副总裁 **Nicolas Chapados** 在领导企业生成式 AI 进步方面拥有丰富经验。他联合创立了多家初创公司，其中最著名的是 **Element AI**，该公司于 **2021** 年被 ServiceNow 收购。

 

**提到的链接**：[CS 194/294-196 (LLM Agents) - Lecture 7](https://www.youtube.com/live/-yf-e-9FvOc.)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1297016755285397534) (33 条消息🔥):

> - `Certification for Course Completion` (课程结业证书)
> - `Project Development Strategies` (项目开发策略)
> - `Hackathon Participation` (Hackathon 参与)
> - `Written Article Assignment` (书面文章作业)
> - `Local LLM Running Options` (本地 LLM 运行选项)

- **课程结业证书**：课程工作人员确认，学生在完成包括测验和书面文章作业在内的所有要求后，将获得证书，截止日期为 **12 月 12 日**。
  
  - 学生还可以利用 **课程录像和讲义** 来补习材料。
- **项目开发策略**：一位参与者寻求指导，是应该专注于理解概念，还是开始使用研讨会中讨论的框架进行项目开发。
  
  - 共识建议将两种方法 **结合起来**，以获得全面的学习体验。
- **Hackathon 向所有人开放**：已确认来自其他大学（如 **UIUC**）的学生可以参加 Hackathon，无需注册该课程。
  
  - 一位成员特别指出，参加 Hackathon 与课程注册是独立的，但相关作业仍然适用。
- **书面文章作业说明**：书面文章作业要求学生撰写一篇总结课程内容或 Hackathon 经历的帖子或文章，并通过提供的链接提交。
  
  - 发布了明确的 **500 字** 指南，表明该作业采用基于努力程度的评分（P/NP）模式。
- **本地运行 LLM**：为参与者提供了本地运行 LLM 的不同选项，**Ollama** 和 **LM Studio 0.3.0** 被指出是实用的工具。
  
  - 用户被提醒，运行较大的模型通常需要超过 **8GB 的 RAM**。

**提到的链接**：

- [Large Language Model Agents](https://llmagents-learning.org/f24)：未找到描述
- [Written Article Assignment Submission](https://forms.gle/7ekobPNSWDLBWnDT6)：说明：创建一个大约 500 字的 Twitter、Threads 或 LinkedIn 帖子。你可以直接将文章发布到你喜欢的平台，或者在 Medium 上撰写文章然后发布链接...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1297658684503101452) (3 条消息):

> - `Orchestration of agents` (Agent 编排)
> - `Lecture timing` (讲座时间)

- **Agent 编排研究**：关于 **Agentic System** 中 **Agent 编排** 的讨论非常活跃，强调这是当前研究的一个重要领域。
  
  - 成员们似乎渴望进一步探索该领域的最新进展和发现。
- **讲座时间确认**：今天课程的时间确认在 **每周一 PST 下午 3-5 点**。
  
  - 这一时间安排方便参与者为未来的讲座制定计划。

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1297648895228252210) (4 条消息):

> - `LibreFLUX release` (LibreFLUX 发布)
> - `FLUX.1-schnell comparison` (FLUX.1-schnell 对比)
> - `Open source characteristics` (开源特性)
> - `Community reactions` (社区反应)

- **LibreFLUX 发布并带来新功能**：**LibreFLUX** 发布，它是 [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) 的 Apache 2.0 版本，提供了完整的 T5 上下文长度、增强的 Attention Masking，并恢复了 Classifier Free Guidance。
  
  - 它优先考虑 **开源原则**，使其更容易针对新分布进行微调，同时采用了让人联想到 2000 年代初期的复古美学。
- **上下文长度和去蒸馏特性**：LibreFLUX 被描述为 **schnell** 的一个主要 **去蒸馏版本 (de-distilled version)**，具有 512 token 的上下文长度和 Attention Masking。
  
  - *社区成员反应积极*，对发布表示兴奋，并认可了其开发过程中付出的努力。

 

**提到的链接**：[jimmycarter/LibreFLUX · Hugging Face](https://huggingface.co/jimmycarter/LibreFLUX)：未找到描述

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1297016011572252777) (12 条消息🔥):

> - `Open-MUSE 训练问题`
> - `Microsoft LLM 突破`
> - `BitNet 模型`
> - `MUSE 项目的训练日志`

- **Open-MUSE 模型训练中的困难**：一位用户报告在 Hugging Face 上寻找 **openMUSE/maskgit-vqgan-imagenet-f16-256** 等模型时遇到困难，但被引导至[重命名后的 checkpoints](https://huggingface.co/amused)。此外，他们在运行脚本时遇到了训练配置文件中的缺失键（missing key）错误。
  
  - 他们提供了 [W&B](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml) 上的配置 YAML 链接以便进一步讨论。
- **Microsoft 声称 LLM 性能飞跃**：有消息称 Microsoft 现在可以在本地设备上运行 **100B 参数模型**，在无需 GPU 的情况下实现高达 **6 倍的速度提升**和 **82% 的能耗降低**，这在 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/)中引起了讨论。
  
  - 据报道，该信息源自一条详细说明该帖子主张的推文，可以参考[这里](https://x.com/jenzhuscott/status/1847514413060046855)。
- **目前尚无使用 BitNet 的 100B 模型**：在讨论 Microsoft 的 LLM 进展时，有人指出，尽管最近有关于效率的传闻，但目前还没有利用 **BitNet** 的 **100B 模型**可用。用户对所引用的性能数据的实际实现和能力持谨慎态度。
- **MUSE 的开源复现工作**：多位用户讨论了 **MUSE** 文生图（text-to-image）模型的开源复现，并分享了 [GitHub 仓库](https://github.com/huggingface/muse)和 [W&B 项目](https://wandb.ai/psuraj/muse?workspace=user-)等资源。该项目旨在通过透明地分享训练过程，提供文生图生成的详细方法。
  
  - 该项目概述的关键步骤包括在 **imagenet** 等数据集上训练各种模型，并在 **CC12M** 上进行实验。

**提到的链接**：

- [amused (MUSE 的开源复现)](https://huggingface.co/amused)：未找到描述
- [来自 Jen Zhu (@jenzhuscott) 的推文](https://x.com/jenzhuscott/status/1847514413060046855)：2/ 你现在可以在本地设备上运行 100B 参数模型，速度提升高达 6 倍，能耗降低 82%——而且全部无需 GPU！本地、高效、私密、极速、开源 🔥 🔥 ...
- [psuraj](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml)：Weights & Biases，机器学习开发者工具
- [Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/)：未找到描述
- [huggingface/open-muse 仓库中的 open-muse/README.md](https://github.com/huggingface/open-muse/blob/main/README.md)：用于快速文生图生成的 MUSE 开源复现。- huggingface/open-muse

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1296912502189330453) (10 条消息🔥):

> - `Aider 的增量改进`
> - `Open Interpreter 中等效于 /functions 文件夹的功能`
> - `自定义工具支持`
> - `Python 虚拟环境`
> - `集成语音助手`

- **Aider 逐步采用 AI 生成的代码**：成员们注意到 **Aider** 在每个版本中都增强了对 AI 生成和优化代码的使用，这暗示了 Interpreter 概念向实时每日构建（nightly builds）发展的趋势。
  
  - 有人好奇 **Open Interpreter** 是否计划在未来实施类似的方法。
- **询问 OI 中等效于 /functions 文件夹的功能**：一位用户询问 **Open Interpreter** 是否有类似于 shell-gpt 中 **/functions** 文件夹的功能，该功能允许用户添加预构建的函数 schema 以便轻松访问。
  
  - 另一位成员表示，目前添加自定义工具的唯一方法可能需要编辑仓库代码。
- **关于 Open Interpreter 自定义工具的讨论**：一位成员表示有兴趣为 Open Interpreter 添加**自定义工具**，并表示如果社区有此需求，愿意提交 Pull Requests。
  
  - 然而，有人指出目前自定义工具可能涉及重大的代码更改。
- **Python 虚拟环境支持咨询**：一位用户询问了在 Python 内核中添加**虚拟环境（virtual environments）**支持的可能性，建议在 Interpreter 类中添加一个简单的属性。
  
  - 虽然不确定这是否会让大多数用户受益，但该成员认为这可以方便在 venv 中安装包。
- **将语音助手集成到 Agent 中**：[AIwithBenefits](https://x.com/AIwithBenefits/status/1848161437828415578) 讨论了将 **HumeAI 语音助手**添加到 **phidatahq** 通用 Agent 中，并通过执行 AppleScript 增强其功能。
  
  - 新的 **phidatahq UI** 受到好评，强调了在原生应用交互中改进的可用性。

 

**提到的链接**：[来自 Jacob@AIwithBenefits (@AIwithBenefits) 的推文](https://x.com/AIwithBenefits/status/1848161437828415578)：为 @phidatahq 通用 Agent 添加了 @hume_ai 语音助手，并得到了来自 @OpenInterpreter 系统消息的一点帮助。引用 Jacob@AIwithBenefits (@AIwithBenefits)：非常喜欢新的 @phid...

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1297208709772218370) (1 条消息):

> - `OpenInterpreter Mac 设置`
> - `交互问题`
> - `LiveKit Meet 链接问题`

- **OpenInterpreter Mac 设置成功**：一位用户确认在他们的 Mac 上成功设置了 OpenInterpreter，并表示 [localhost:10100](http://localhost:10100) 可以正常工作以控制其系统。
  
  - 这表明初始配置已正确完成，允许使用远程控制功能。
- **Web 浏览器访问被拒绝**：该用户报告在尝试交互时收到一条消息，称：*“抱歉，我无法访问您的 Web 浏览器等，但我可以为您提供指导。”*
  
  - 这表明其设备上的 OpenInterpreter 设置在 Web 访问能力方面可能存在限制。
- **LiveKit Meet 链接无法工作**：该用户分享说，无论是应用程序还是 [LiveKit Meet 链接](https://link.to/livekit) 都无法访问其计算机以实现功能。
  
  - 这引发了在 Mac 上使用这些功能时关于兼容性或权限的担忧。

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1297013701559451648) (9 条消息🔥):

> - `LangGraph Code Assistant`
> - `Role-based RAG Models`
> - `Expected Context Issues`
> - `Techstars Startup Weekend Event`
> - `Code Generation Approaches`

- **LangGraph Code Assistant 实现步骤**：**LangGraph Code Assistant** 教程概述了一种使用 [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) 和 RAG 技术迭代构建编程问题答案的方法。
  
  - *该过程包括摄取用户指定的文档、调用工具以获得结构化输出，并在返回解决方案之前进行单元测试。*
- **基于角色的 RAG 实现考量**：一位成员询问了如何根据用户角色拆分 **RAG 模型**，例如允许 CEO 访问特定的财务文档，而将实习生限制在相关文档范围内。
  
  - 这种方法提出了关于在使用 RAG 模型时如何有效管理和限制访问权限的问题。
- **上下文检索故障排除**：一位用户表示，尽管信息已存储在向量数据库中，但在获取查询所需的预期上下文时遇到了困难。
  
  - 建议检查 **embeddings** 或优化 prompt 以获得更好的结果。
- **Techstars Startup Weekend SF 公告**：**Techstars Startup Weekend SF** 邀请技术社区在 TechCrunch Disrupt 之后前往 [AWS GenAI Loft](https://aws.amazon.com/startups/lp/aws-gen-ai-loft-san-francisco?lang=en-US) 进行交流和建立联系。
  
  - 该活动包括行业专家的演讲，随后是创始人、投资者和创新者的社交机会。
- **代码生成策略讨论**：一位参与者讨论了 **AlphaCodium** 的代码生成方法，强调通过公开测试和 AI 生成的测试进行迭代测试。
  
  - 他们概述了流程，包括如何使用 `code_gen_chain.invoke()` 进行反思和生成代码解决方案。

**提到的链接**：

- [来自 UltraIA (@Ultra_IA) 的推文](https://x.com/Ultra_IA/status/1847821253476008227)：LOL
- [Code Assistant](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/#code-solution)：未找到描述
- [TC Disrupt AI Founders Happy Hour by Techstars Startup Weekend SF @ AWS GenAI Loft · Luma](https://lu.ma/5f5ydtxq)：前往 ASW GenAI Loft 参加一个专属夜晚，进行真实的对话和真诚的联系。我们将邀请一位 AI 领域的顶尖人物（详情……）

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1297706150489362483) (1 条消息):

> - `OpenAI Swarm`
> - `LangChain LangGraph`
> - `Multi-Agent Frameworks`

- **对比 OpenAI Swarm 和 LangChain LangGraph**：一篇详细的文章对比了 **OpenAI Swarm** 和 **LangChain LangGraph**，重点关注它们的功能以及构建复杂 AI 工作流的最佳用例。
  
  - 该资源旨在指导读者确定哪个框架可能最适合他们的项目，可在此处访问 [here](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474)。
- **多 Agent 工作流的重要性**：该消息强调了在不断发展的人工智能领域中创建**多 Agent 工作流 (multi-agent workflows)** 日益增长的重要性。
  
  - 此类框架使开发者能够处理复杂的交互和流程，从而增强整体的 **AI 能力**。

**提到的链接**：[OpenAI Swarm vs LangChain LangGraph: A Detailed Look at Multi-Agent Frameworks](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474)：Ankush k Singal

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/) (1 条消息):

huikang: [https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1298017406215786526) (1 messages):

> - `AI access challenges`
> - `Competition in AI`
> - `External researcher access`
> - `Big Tech and AI`
> - `Open AI ecosystem`

- **Mozilla 委托开展关于 AI 访问挑战的研究**：Mozilla 委托开展了两项关于 **AI access**（AI 访问）和竞争挑战的深入研究：[External Researcher Access to Closed Foundation Models](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf) 以及 [Stopping Big Tech From Becoming Big AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf)。这些报告由 **AWO** 和 **Open Markets Institute** 提供，重点关注 AI 控制权的动态以及构建公平生态系统所需的变革。
- **理解 AI 发展中的控制权**：该研究强调了**谁在控制** AI 的发展，并阐述了为确保公平环境所需的改革。报告强调了 **external researchers**（外部研究人员）获得封闭模型访问权限对于更广泛创新的重要性。
  
  - 正如报告中所指出的，确保公平的竞争环境对于在快速演进的 AI 领域中维持创新至关重要。
- **博客文章详细介绍了 AI 研究发现**：有关受托研究的更多信息可以在[此处的博客文章](https://discord.com/channels/1089876418936180786/1298015953463808102)中找到。该博客讨论了在当前 AI 治理背景下这些发现的影响。

 

---

### **DiscoResearch ▷ #**[**general**](https://discord.com/channels/1178995845727785010/1182877486854451271/) (1 messages):

huunguyen: 有人试过 q-galora 吗？

---

---

---

---

{% else %}

> 完整的频道详细分解内容已在邮件中截断。
> 
> 如果您想查看完整分解，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}