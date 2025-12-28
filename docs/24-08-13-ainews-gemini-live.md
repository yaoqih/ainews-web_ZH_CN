---
companies:
- google
- anthropic
- tii
- supabase
- perplexity-ai
- llamaindex
- openai
- hugging-face
date: '2024-08-14T01:23:26.876396Z'
description: '**谷歌（Google）**在 Pixel 9 发布会期间为 **Gemini Advanced** 订阅用户在 Android 平台上推出了
  **Gemini Live**，其特点是集成了 Google Workspace 应用和其他谷歌服务。该功能于 2024 年 8 月 12 日开始推出，并计划支持
  iOS。


  **Anthropic** 发布了 **Genie**，这是一个 AI 软件工程系统，在 SWE-Bench 基准测试中实现了 **57%** 的提升。**TII**
  推出了 **Falcon Mamba**，这是一个 7B 参数的无注意力机制（attention-free）开源模型，可扩展至长序列。基准测试显示，更长的上下文长度并不总能改善检索增强生成（RAG）。


  **Supabase** 推出了一个由 AI 驱动的 Postgres 服务，被称为“数据库界的 ChatGPT”，且完全开源。**Perplexity AI**
  与 Polymarket 合作，将实时概率预测集成到搜索结果中。


  一项教程展示了使用 **Qdrant**、**LlamaIndex** 和 **Gemini** 构建的多模态食谱推荐系统。一位 OpenAI 工程师分享了成功秘诀，强调了调试和努力工作的重要性。线性代数中矩阵与图之间的联系被重点提及，为理解非负矩阵和强连通分量提供了见解。**Keras
  3.5.0** 正式发布，集成了 Hugging Face Hub 以支持模型的保存和加载。'
id: 9afa131e-b7a0-448b-a184-550ac7f96ccf
models:
- gemini-1.5-pro
- genie
- falcon-mamba
- gemini-1.5
- llamaindex
original_slug: ainews-gemini-live
people:
- omarsar0
- osanseviero
- dbrxmosaicai
- alphasignalai
- perplexity_ai
- _jasonwei
- svpino
title: '**Gemini Live**（通常直接保留英文名称，也可译为 **Gemini 实时对话** 或 **Gemini 实时语音**）。


  这是 Google 推出的一项功能，允许用户与 Gemini AI 进行流畅、自然的实时语音对话。'
topics:
- multimodality
- benchmarking
- long-context
- retrieval-augmented-generation
- open-source
- model-releases
- model-integration
- model-performance
- software-engineering
- linear-algebra
- hugging-face-hub
- debugging
---

<!-- buttondown-editor-mode: plaintext -->**生活中各种每月 20 美元的小额订阅就是你所需的一切。**

> 2024/8/12-2024/8/13 的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**253** 个频道和 **2423** 条消息）。预计节省阅读时间（以 200wpm 计算）：**244 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如在 [Google I/O](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/) 上所承诺的，Gemini Live 今天在 Android 平台上发布，面向 Gemini Advanced 订阅用户，作为 #MadeByGoogle Pixel 9 发布会的一部分。对那位在台上遭遇了 2 次演示失败的 [可怜演示者](https://x.com/durreadan01/status/1823430521768304674) 表示同情：

 
![image.png](https://assets.buttondown.email/images/7e351a3f-0f0d-4793-93ce-5b1974595c25.png?w=960&fit=max)
 

[Gemini Live 的媒体评测解禁](https://www.theverge.com/2024/8/13/24219736/gemini-live-hands-on-pixel-event) 结果持谨慎乐观态度。它将拥有 “[extensions](https://support.google.com/gemini/answer/13695044?visit_id=638591951502121215-2420806349&p=more_extensions&rd=1)”（扩展），即与你的 Google Workspace (Gmail, Docs, Drive)、YouTube、Google Maps 以及其他 Google 产品的集成。

重要的是，Google 今天开始推出该功能（尽管截至太平洋时间下午 5 点，我们仍然 [无法找到任何人](https://www.reddit.com/r/singularity/comments/1erdr0t/meet_gemini_live_a_new_way_to_have_more_natural/) 发布其实测录屏），而 ChatGPT 的 Advanced Voice Mode 发布日期仍不确定。Gemini Live 未来也将面向 iOS 订阅用户推出。

该公司还向现场观众展示了 Gemini Live 在 [Pixel Buds Pro 2](https://x.com/greengart/status/1823444923573731411) 上的演示，并向 [华尔街日报 (WSJ)](https://x.com/JoannaStern/status/1823429729870868676) 进行了展示。对于关注 Pixel 9 的人来说，还有 Add Me 拍照功能和 Magic Editor 等显著的图像 AI 集成。

https://www.youtube.com/watch?v=KoN_bcDmhR4

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳结果。

**AI 模型进展与基准测试**

- Anthropic 发布了 Genie，这是一个全新的 AI 软件工程系统，在 SWE-Bench 上实现了 30.08% 的 SOTA 性能，比之前的模型提升了 57%。关键点包括推理数据集、具备规划和执行能力的 Agent 系统，以及自我改进能力。[@omarsar0](https://twitter.com/omarsar0/status/1823118952362278962)

- TII 发布了 Falcon Mamba，这是一个全新的 7B 开源模型。它是一个 Attention-free 模型，可以扩展到任意序列长度，并且与同尺寸模型相比具有强大的指标表现。[@osanseviero](https://twitter.com/osanseviero/status/1823000588029743324)

- 研究人员对 13 个流行的开源和商业模型在 2k 到 125k 的上下文长度下进行了基准测试，发现长上下文并不总是有助于检索增强生成 (RAG)。大多数生成模型的性能在超过一定上下文大小后会下降。[@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1823129597288046979)

**AI 工具与应用**

- Supabase 推出了一个基于 AI 的 Postgres 服务，被称为“数据库界的 ChatGPT”。它允许用户构建和启动数据库、创建图表、生成 Embeddings 等。该工具 100% 开源。[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1823020725042639243)

- Perplexity AI 宣布与 Polymarket 建立合作伙伴关系，将选举结果和市场趋势等事件的实时概率预测整合到其搜索结果中。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1823029449534615705)

- 分享了一个使用 Qdrant、LlamaIndex 和 Gemini 构建多模态食谱推荐系统的教程，演示了如何摄取 YouTube 视频并对文本和图像块进行索引。[@llama_index](https://twitter.com/llama_index/status/1823145827042468125)

**AI 工程见解**

- 一位 OpenAI 工程师分享了在该领域取得成功的见解，强调了彻底调试和理解代码的重要性，以及努力完成任务的意愿。[@_jasonwei](https://twitter.com/_jasonwei/status/1823067805748728051)

- 讨论了线性代数中矩阵与图之间的联系，强调了这种关系如何提供对非负矩阵和强连通分量的见解。[@svpino](https://twitter.com/svpino/status/1822966303642308903)

- Keras 3.5.0 发布，具有一流的 Hugging Face Hub 集成，允许直接从 Hub 保存和加载模型。此次更新还包括 Distribution API 的改进，以及支持 TensorFlow、PyTorch 和 JAX 的新算子。[@fchollet](https://twitter.com/fchollet/status/1823098449883230341)

**AI 伦理与监管**

- 围绕 AI 监管及其对创新潜在影响的讨论受到关注，一些人认为过早的监管可能会阻碍有益 AI 应用的进展。[@bindureddy](https://twitter.com/bindureddy/status/1823095005206261835)

- 有人对 AI “业务战略决策支持”初创公司的有效性表示担忧，认为其价值不易衡量，也难以获得客户信任。[@saranormous](https://twitter.com/saranormous/status/1823076401496625164)

**AI 社区与活动**

- Google DeepMind 播客宣布了第三季，探讨了 Chatbot 与 Agent 之间的区别、AI 在创意中的角色，以及实现 AGI 后潜在的生活场景等话题。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1822997510727598585)

- 宣布了由 Andrew Ng 教授的 AI Python for Beginners 课程，旨在帮助有抱负的开发者和专业人士利用 AI 提高生产力并自动化任务。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1823004343278219378)

**迷因与幽默**

- 分享了各种与 AI 和技术相关的幽默推文和迷因，包括关于 AI 模型名称和能力的笑话。[@swyx](https://twitter.com/swyx/status/1823122765584683248)

本摘要捕捉了所提供推文的主要主题和讨论，重点关注 AI 模型、工具、应用的最新进展，以及对 AI 工程和科技行业的广泛影响。

---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. 高级量化与模型优化技术**

- **[Llama-3.1 70B 4-bit HQQ/校准量化模型：在 lm-eval 的所有基准测试中，相对于 FP16 的性能达到 99% 以上，且推理速度与 FP16 相当（在 A100 上为 10 toks/sec）。](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq)** ([评分: 91, 评论: 26](https://reddit.com//r/LocalLLaMA/comments/1eqfsa4/llama31_70b_4bit_hqqcalibrated_quantized_model_99/)): **Llama-3.1 70B** 模型已通过 **HQQ/校准量化** 成功量化为 **4-bit**，在 lm-eval 的所有基准测试中实现了超过 **99% 的 FP16 相对性能**。该量化版本保持了与 FP16 类似的推理速度，在 **A100 GPU** 上每秒处理约 **10 个 tokens**。这一成就展示了在保持性能的同时模型压缩方面取得的重大进展，可能使大型语言模型的部署更加高效。

- **为什么 Unsloth 如此高效？** ([评分: 94, 评论: 35](https://reddit.com//r/LocalLLaMA/comments/1eqdox0/why_is_unsloth_so_efficient/)): **Unsloth** 在显存有限的情况下处理 **32k 文本长度** 的摘要任务时，表现出了卓越的**效率**。用户报告称，使用 Unsloth 在 **L40S 48GB GPU** 上成功训练了一个模型，而传统的 **transformers llama2** 结合 **qlora**、**4bit** 和 **bf16** 技术的方法在相同硬件上则无法运行。显著的性能提升归功于 Unsloth 对 **Triton** 的使用，尽管具体机制对用户来说尚不明确。

- **[在 9 天内预训练一个 LLM 😱😱😱](https://arxiv.org/abs/2408.03506)** ([评分: 216, 评论: 53](https://reddit.com//r/LocalLLaMA/comments/1eqakjc/pretraining_an_llm_in_9_days/)): **Hugging Face** 和 **Google** 的研究人员开发了一种方法，仅使用 **16 台 A100 GPU**，在短短 **9 天** 内就预训练了一个拥有 **1.3B 参数的语言模型**。这项名为 **Retro-GPT** 的技术将 **检索增强语言建模 (retrieval-augmented language modeling)** 与 **高效的预训练策略** 相结合，实现了与训练时间更长的模型相当的性能，有可能彻底改变 LLM 开发的速度和成本效益。

**主题 2. LLM 开发的开源贡献**

- **[一个广泛的 RAG 实现开源集合，包含多种不同策略](https://github.com/NirDiamant/RAG_Techniques)** ([评分: 91, 评论: 20](https://reddit.com//r/LocalLLaMA/comments/1eqec8v/an_extensive_open_source_collection_of_rag/)): 该帖子介绍了一个 **开源仓库**，其中包含 **17 种不同检索增强生成 (RAG) 策略** 的全面集合，并配有 **教程和可视化**。作者鼓励社区参与，邀请用户提交 issue、建议额外策略，并将该资源用于学习和参考。

- **来自 TII（阿联酋技术创新研究所）的 Falcon Mamba 7B** ([评分: 87, 评论: 18](https://reddit.com//r/LocalLLaMA/comments/1eqaad2/falcon_mamba_7b_from_tii_technology_innovation/)): 阿联酋的 **技术创新研究所 (TII)** 发布了 **Falcon Mamba 7B**，这是一款开源的 **状态空间语言模型 (SSLM)**，结合了 Falcon 架构与 Mamba 的状态空间序列建模。该模型可在 **Hugging Face** 上获取，配有模型卡片、集合和 playground，允许用户探索和实验这项新的 AI 技术。
    - 用户测试了 **Falcon Mamba 7B**，报告的结果**褒贬不一**。一位用户发现它在处理产品需求文档 (PRD) 任务时表现 **“非常非常非常差”**，响应变得平庸且无组织。
    - 该模型的性能受到质疑，尽管声称具有优越性，但一些用户发现它**比 Llama 和 Mistral 模型更差**。使用各种 prompts 进行的测试结果令人失望。
    - 一些用户对 Falcon 模型表示 **怀疑**，这基于过去的负面体验，暗示 Falcon 系列可能存在性能不佳的模式。

## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型发布与功能**

- **关于新 GPT-4 模型的猜测**：r/singularity 上的一篇帖子声称 ChatGPT 提到了一个[“自上周以来发布的新 GPT-4o 模型”](https://www.reddit.com/r/singularity/comments/1eqpy8o/new_gpt4o_model_out_since_last_week_chatgpt/)，引发了关于 OpenAI 可能发布新产品的讨论。

- **Flux 图像生成模型**：多篇帖子讨论了新 Flux 图像生成模型的能力：
  - [令人印象深刻的印象派风景生成](https://www.reddit.com/r/StableDiffusion/comments/1eqa8ds/first_image_is_how_an_impressionist_landscape/)，使用了基于 5000 张图像训练的自定义 LoRA。
  - [尝试生成符合解剖学结构的裸体图像](https://www.reddit.com/r/StableDiffusion/comments/1eq8268/a_pretty_rough_first_attempt_at_a_combo/)，使用了自定义 LoRA。
  - 为虚构产品进行[创意广告概念生成](https://www.reddit.com/r/StableDiffusion/comments/1eqi9wj/flux_nuke_your_thirst/)。

**AI 生成媒体**

- **带有合成语音的 AI 生成视频**：一段[演示视频](https://www.reddit.com/r/StableDiffusion/comments/1eqwh1p/added_voice_to_flux_videos_through_rendernet/)展示了将 Flux 生成的图像进行动画处理并配以 AI 生成的语音，尽管评论者指出存在口型同步和语音质量问题。

**自动驾驶汽车**

- **Waymo 自动驾驶汽车问题**：一段[视频帖子](https://www.reddit.com/r/singularity/comments/1eqoxho/waymo_cars_being_clueless_from_their_spawn/)显示 Waymo 自动驾驶汽车在从起点导航时遇到困难，引发了关于当前技术局限性的讨论。

**AI 与社会**

- **AI 伴侣与人际关系**：一个[具有争议的迷因帖子](https://www.reddit.com/r/singularity/comments/1eqz3sb/real_or/)引发了关于 AI 伴侣对人类关系和社会动态潜在影响的辩论。

---

# AI Discord 综述

> 由 GPT-4o (gpt-4o-2024-05-13) 生成的摘要之摘要的摘要

**1. 模型性能与基准测试**

- **无审查模型表现优于 Meta Instruct**：一个经过微调以保留原始 Meta Instruct 模型智能的无审查模型已经发布，并在 **[LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)** 上超越了原始模型。
  - 该模型的表现引发了关于审查与实用性之间权衡的讨论，许多用户赞扬其处理更广泛输入的能力。
- **Mistral Large：当前的冠军？**：一位成员发现 **[Mistral Large 2](https://huggingface.co/mradermacher/Tiger-Gemma-9B-v1-i1-GGUF)** 是目前最好的 LLM，在处理困难的新颖问题上胜过 **Claude 3.5 Sonnet**。
  - 然而，**Gemini Flash** 在价格上大幅低于 **OpenAI 4o mini**，但 **OpenAI 4o** 的价格比 **Mistral Large** 更便宜。
- **Google 的 Gemini Live：正式上线，但并非免费**：**[Gemini Live](https://www.digitalheresy.com/p/strawberry)** 现已面向 **Advanced Subscribers**（高级订阅用户）开放，在 Android 上提供对话式叠加功能以及更多连接的应用。
  - 许多用户表示，它比旧的语音模式有所改进，但仅供付费用户使用，且缺乏实时视频功能。


**2. GPU 与硬件讨论**

- **GPU 之战 - A100 vs A6000**：成员们讨论了 **A100 与 A6000 GPU** 的优缺点，其中一位成员指出 A6000 具有极佳的价格/VRAM 比，且与 24GB 显卡相比没有限制。
  - 讨论强调了 **VRAM** 和成本效益对于大型模型训练和推理的重要性。
- **Stable Diffusion 安装困扰**：一位用户报告了安装 **Stable Diffusion** 时的困难，遇到了 **CUDA 安装**问题以及在 Hugging Face 上查找 Token 的问题。
  - 另一位用户提供了通过个人资料设置菜单生成 Token 以及正确安装 CUDA 的指导。
- **TorchAO 在 Cohere for AI 的演讲**：来自 PyTorch Architecture Optimization 的 **[Charles Hernandez](https://tinyurl.com/C4AICommunityApp)** 将在 Cohere For AI 的 ml-efficiency 小组介绍 TorchAO 和量化（quantization）。
  - 该活动由 **@Sree_Harsha_N** 主持，参与者可以通过提供的链接加入 Cohere For AI。


**3. 微调与优化技术**

- **模型微调技巧与窍门**：讨论围绕微调 **Phi3 model** 以及是否使用 **LoRA** 或全量微调展开，一位成员建议将 **RAG** 作为潜在解决方案。
  - 用户分享了经验和最佳实践，强调了为不同模型选择正确微调策略的重要性。
- **TransformerDecoderLayer 重构 PR**：已提交一个重构 **TransformerDecoderLayer** 的 PR，涉及多个文件，并对 **modules/attention.py** 和 **modules/transformer.py** 进行了核心修改。
  - 该 PR 实现了 **RFC #1211**，旨在改进 **TransformerDecoderLayer** 架构。
- **PyTorch 全 FP16：是否可行？**：一位用户询问是否可以在 **PyTorch core** 中使用带有 loss/grad scaling 的全 FP16，特别是在微调来自 **Fairseq** 的大型模型时。
  - 他们尝试使用 **torch.GradScaler()** 并将模型转换为 FP16，而不使用 **torch.autocast('cuda', torch.float16)**，但遇到了错误 'ValueError: Attempting to unscale FP16 gradients.'。


**4. AI 平台的 UI/UX 问题**

- **Perplexity 的 UI/UX 问题**：用户报告了多个 UI/UX 问题，包括按钮缺失和提示词输入框消失，导致与平台交互困难。
  - 这些 Bug 在 **Perplexity** 的网页版和 iOS 版中均有报告，引起了用户的极大挫败感，并阻碍了他们有效利用该平台。
- **LLM Studio 的 Model Explorer 宕机**：多名成员报告称，为 **LM Studio Model Explorer** 提供支持的 **HuggingFace** 已宕机。
  - 确认该网站已数小时无法访问，多个地区均报告了连接问题。
- **Perplexity 网站稳定性担忧**：用户报告网站稳定性显著下降，理由包括搜索行为异常、遗忘上下文以及网页和 iOS 版的界面 Bug。
  - 这些问题引发了对 **Perplexity** 提供的可靠性和用户体验的担忧。


**5. 开源 AI 框架与社区努力**

- **Rust GPU 移交给社区所有**：此前由 **Embark Studios** 负责的 **[Rust GPU](https://rust-gpu.github.io)** 项目现在归 **Rust GPU GitHub organization** 社区所有。
  - 这一转变标志着旨在振兴、统一和标准化 Rust 中 GPU 编程的更广泛战略的开始。
- **Open Interpreter 实现万物转换**：使用 **Open Interpreter** 将任何类型的数据转换为任何其他格式。
  - 通过使用利用 **Open Interpreter** 强大功能的 'Convert Anything' 工具，这是可以实现的。
- **Cohere For AI 研究实验室**：**[Cohere For AI](https://cohere.com/research)** 是一个非营利研究实验室，致力于解决复杂的机器学习问题。
  - 他们支持探索未知的基本研究，并专注于为机器学习研究创造更多切入点。




---

# PART 1: Discord 高层级摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Pro 早期访问**：目前正向 **Unsloth** 社区中值得信赖的成员提供 **Unsloth Pro** 版本的早期访问权限。
- **A100 vs A6000 GPU 对决**：成员们讨论了 **A100** 与 **A6000** GPU 的优缺点，一位成员指出 **A6000** 具有极佳的价格/VRAM 比，且与 24GB 显卡相比没有限制。
- **无审查模型登顶排行榜**：一个经过调整以保留原始 **Meta Instruct** 模型智能的无审查模型已发布，并在 **LLM Leaderboard 2** 上超越了原始模型。
- **Dolphin 模型遭受审查困扰**：一位成员报告称 **Dolphin 3.1** 模型无法处理最基本的请求并予以拒绝，这可能是由于其严格的审查制度。
- **AI 工程师的微调**：讨论围绕微调 **Phi3 model** 以及是否使用 **LoRA** 或全量微调展开，一位成员建议将 **RAG** 作为潜在解决方案。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **TorchAO 在 Cohere For AI 的演讲**：来自 PyTorch Architecture Optimization 的 Charles Hernandez 将于 CEST 时间 2000 年 8 月 16 日在 Cohere For AI 的 ml-efficiency 小组介绍 TorchAO 和 quantization。
   - 本次活动由 @Sree_Harsha_N 主持，参与者可以通过链接 [https://tinyurl.com/C4AICommunityApp](https://tinyurl.com/C4AICommunityApp) 加入 Cohere For AI。
- **CPU matmul 优化之战**：一位用户尝试在 Zig 中编写基于 tiling 的 matmul，但在实现最佳性能方面遇到了困难。
   - 他们收到了关于探索 cache-aware 循环重排以及使用 SIMD 指令潜力的建议，并将其性能与 GGML 和 NumPy 进行了对比，后者利用优化的 BLAS 实现获得了极快的运行结果。
- **FP16 权重与 CPU 性能**：一位用户询问了如何在 CPU 上处理 FP16 权重，并指出最近的模型通常使用 BF16。
   - 建议他们将 FP16 权重转换为 BF16 或 FP32，其中 FP32 不会导致精度损失，但可能会导致推理速度变慢；同时建议探索在运行时将 tensor 从 FP16 转换为 FP32 以潜在地提高性能。
- **PyTorch 全 FP16：真的可行吗？**：一位用户询问在 PyTorch 核心库中是否可以使用带有 loss/grad scaling 的全 FP16 模式，特别是在微调来自 Fairseq 的中大型模型时。
   - 他们尝试使用 `torch.GradScaler()` 并将模型转换为 FP16，且不使用 `torch.autocast('cuda', torch.float16)`，但遇到了错误 "ValueError: Attempting to unscale FP16 gradients."。
- **torch.compile: The Missing Manual**：一份名为 "torch.compile: The Missing Manual" 的新 PyTorch 文档与一段 YouTube 视频一同被分享。
   - 该文档和视频分别可在 [https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) 和 [https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf](https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf) 获取，提供了关于利用 `torch.compile` 的详细信息。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Vision Adapters：视觉模型的关键**：只有特定的 LLM 模型拥有 vision adapters，其中大多数以 "LLaVa" 或 "obsidian" 命名。
   - "VISION ADAPTER" 是视觉模型的关键组件；如果没有它，就会弹出你分享的那个错误。
- **Mistral Large：当前的冠军？**：一位成员认为 **Mistral Large 2** 是目前最好的 LLM，在处理困难的新颖问题上胜过 **Claude 3.5 Sonnet**。
   - 然而，该成员也指出 **Gemini Flash** 在价格上大幅低于 **OpenAI 4o mini**，但 **OpenAI 4o** 的价格比 **Mistral Large** 更便宜。
- **LLM Studio 的 Model Explorer 宕机**：多名成员报告称，为 **LM Studio Model Explorer** 提供支持的 HuggingFace 出现故障。
   - 该网站被确认已连续数小时无法访问，多个地区都报告了连接问题。
- **Llama 3.1 性能问题**：一位用户报告称，他们的 **Llama 3 8B 模型** 现在的运行速度仅为 3 tok/s，而最近更新前为 15 tok/s。
   - 用户检查了他们的 GPU offload 设置并将其重置为默认值，但问题仍然存在；该问题似乎与最近更新中的更改有关。
- **LLM 输出长度控制**：一位成员正在寻找限制响应输出长度的方法，因为某些模型即使被指示提供单句回答，也倾向于输出整个段落。
   - 虽然可以修改 system prompts，但该成员发现 8B 模型（特别是 **Meta-Llama-3.1-8B-Instruct-GGUFI**）在遵循精确指令方面表现并非最佳。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google 推出 Gemini Live，但并非面向所有人**：**Gemini Live** 现已向 **Advanced Subscribers** 开放，在 Android 上提供对话式叠加层以及更多连接的应用。
   - 许多用户表示，它比旧的语音模式有所改进，但仅供付费用户使用，且缺乏实时视频功能。
- **Strawberry：营销天才还是 OpenAI 的新面孔？**：关于名为 "Strawberry" 的神秘用户发布一串表情符号的讨论，引发了人们对其与 OpenAI 或 Sam Altman 可能存在联系的猜测。
   - 用户评论说，草莓表情符号与 Sam Altman 手持草莓的照片相关联，是一种聪明的营销策略，成功吸引了用户参与对话。
- **Project Astra 期待已久的到来**：**Gemini Live** 的发布暗示了 **Project Astra**，但许多用户对其缺乏进一步进展感到失望。
   - 一位用户甚至将其与 **Microsoft recall** 进行了类比，暗示由于安全担忧，人们对该产品的发布持怀疑态度。
- **LLMs：并非万能解决方案**：一些用户对 LLMs 是解决所有问题的方案表示怀疑，特别是在处理数学、数据库甚至 waifu 角色扮演等任务时。
   - 其他用户强调，tokenization 仍然是一个根本性的弱点，LLMs 需要更具策略性的方法，而不是依靠暴力 tokenization 来解决复杂问题。
- **ChatGPT 的网站访问限制：一个持续存在的问题**：一位成员询问如何让 ChatGPT 访问网站并获取文章，但另一位成员指出，ChatGPT 可能会被阻止抓取网页内容，或者会对网页内容产生幻觉。
   - 一位用户询问是否有人尝试使用 "web browser GPT" 这一术语作为可能的变通方法。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 UI/UX Bug**：用户遇到了 UI/UX 问题，包括按钮缺失和提示词字段消失，导致与平台交互困难。
   - 这些 Bug 在 Perplexity 的网页版和 iOS 版上均有报告，引起了用户的极大挫败感，并阻碍了他们有效利用平台。
- **Sonar Huge：新模型，新问题**：新模型 "Sonar Huge" 取代了 Perplexity Pro 中的 Llama 3.1 405B 模型。
   - 然而，用户观察到新模型运行缓慢，且未能遵循用户个人资料中的提示词，引发了对其有效性和性能的担忧。
- **Perplexity 的网站稳定性问题**：用户报告网站稳定性显著下降，出现了搜索行为异常、丢失上下文以及各种界面 Bug 等问题。
   - 这些问题在网页版和 iOS 版上均有观察到，引发了对 Perplexity 提供的可靠性和用户体验的担忧。
- **Perplexity 的 Success Team 注意到相关情况**：Perplexity 的 Success Team 承认收到了关于平台近期出现的 Bug 和故障的用户反馈。
   - 他们表示已意识到报告的问题及其对用户体验的影响，暗示未来可能会有解决方案和改进。
- **Perplexity 的功能实现延迟**：一位用户对功能实现的漫长等待时间表示沮丧。
   - 他们强调了承诺的功能与实际推出速度之间的差异，强调了加快开发和交付以满足用户期望的重要性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI 的 SXSW 研讨会提案**：Stability AI 首席执行官 Prem Akkaraju 和科技影响力人物 Kara Swisher 将在 SXSW 上讨论开源 AI 模型的重要性以及政府在监管其影响方面的角色。
   - 该研讨会将探讨 AI 的机遇与风险，包括岗位取代、虚假信息、CSAM 以及知识产权（IP rights），并可在 PanelPicker® 上查看：[PanelPicker | SXSW Conference & Festivals](http://panelpicker.sxsw.com/vote/153232)。
- **Google Colab 运行时停止工作**：一位用户遇到了 Google Colab 运行时过早停止的问题。
   - 另一位用户建议切换到 Kaggle，它提供更多资源和更长的运行时，为更长时间的 AI 实验提供了解决方案。
- **Stable Diffusion 安装与 CUDA 挑战**：一位用户在安装 Stable Diffusion 时遇到了困难，涉及 CUDA 安装以及查找 Hugging Face token 的问题。
   - 另一位用户提供了通过 Hugging Face 个人资料设置菜单生成 token 并正确安装 CUDA 的指导，为该用户的挑战提供了解决方案。
- **模型合并讨论**：一位用户建议利用 UltraChat 和基础 Mistral 之间的差异来改进 Mistral-Yarn，将其作为一种潜在的模型合并（Model Merging）策略。
   - 尽管一些用户表示怀疑，但原用户保持乐观，并引用了以往成功的模型合并尝试，展示了 AI 模型开发的潜在进展。
- **用于换脸的 Flux 真实感**：一位用户在尝试 fal.ai 产生卡通化效果后，寻求实现真实换脸的替代方案。
   - 另一位用户建议使用 Flux，因为它能够对 logo 进行训练并将其准确放置在图像上，为用户的换脸目标提供了潜在解决方案。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash 1.5 降价**：**Gemini Flash 1.5** 的输入 token 成本降低了 **78%**，输出 token 成本降低了 **71%**。
   - 这使得该模型对更广泛的用户群体来说更加易于获取且负担得起。
- **GPT-4o Extended 早期访问启动**：**GPT-4o Extended** 的早期访问已通过 **OpenRouter** 启动。
   - 您可以通过此链接访问：[https://x.com/OpenRouterAI/status/1823409123360432393](https://x.com/OpenRouterAI/status/1823409123360432393)。
- **OpenRouter 的更新障碍**：OpenRouter 的更新受到了 Gemini 新的 1:4 token 与字符比例的阻碍，这无法清晰地映射到 `max_tokens` 参数验证中。
   - 一位用户对不断变化的 token 与字符比例表示沮丧，并建议切换到按 token 计费的系统。
- **Euryale 70B 停机**：一位用户报告 **Euryale 70B** 对部分用户停机，但对他本人正常，引发了关于故障或错误率的疑问。
   - 进一步讨论显示了多次停机情况，包括一次因更新导致的 10 分钟中断，以及可能持续存在的区域可用性问题。
- **模型性能对比**：用户对比了 **Groq 70b** 和 **Hyperbolic** 的性能，发现相同 prompt 的结果几乎完全一致。
   - 这引发了关于 **FP8 量化（FP8 quantization）** 影响的讨论，一些用户指出它在实践中差异极小，但另一些用户则指出某些供应商可能会出现质量下降的情况。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 许可证的限制性条款**：[Mojo License](https://www.mojo-lang.org/docs/license) 禁止使用该语言开发用于竞争性活动的应用程序。
   - 然而，条款规定该规则不适用于在初始发布后才变得具有竞争性的应用程序，但目前尚不清楚该条款将如何执行。
- **Mojo 开源时间表仍不明确**：用户询问了 Mojo 编译器开源的时间表。
   - 团队确认编译器最终会开源，但没有提供具体时间表，并暗示在能够接受贡献之前可能还需要一段时间。
- **Mojo 开发：专注于标准库**：目前 Mojo 开发的重点是构建标准库。
   - 鼓励用户为标准库做出贡献，而编译器的开发工作虽然在进行中，但尚未开放贡献。
- **Stable Diffusion 与 Mojo：内存至关重要**：一位用户在 WSL2 中运行 Stable Diffusion Mojo ONNX 示例时遇到了内存压力问题，导致进程被终止。
   - 该用户为 WSL2 分配了 8GB 内存，但团队建议将其翻倍，因为 Stable Diffusion 1.5 大约为 4GB，模型及其优化过程都需要更多内存。
- **微软版 Java：往事回顾**：一位成员认为“微软版 Java”是不必要的且本可以避免，而另一位成员则反驳说这在当时似乎至关重要。
   - 讨论承认了新解决方案的出现以及“微软版 Java”随时间的衰落，强调了它 20 年的运行历程及其在微软市场份额中的相关性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere For AI 研究实验室扩张**：**Cohere For AI** 是一个专注于复杂机器学习问题的非营利研究实验室。他们正在为机器学习研究创造更多的准入门槛。
   - 他们支持[探索未知的基本研究](https://cohere.com/research)。
- **Cohere 网站价格变动**：一位用户询问了 **classify** 功能的定价，因为它不再列在定价页面上。
   - 未提供回复。
- **JSONL 上传失败**：用户报告了上传用于 fine-tuning 的 JSONL 数据集时出现的问题。
   - Cohere 支持团队承认了该问题，表示正在调查中，并建议暂时使用 API 创建数据集作为替代方案。
- **Azure 不支持 JSON 格式化**：一位成员询问在 Azure 中使用 `response_format` 进行结构化输出的问题，但遇到了错误。
   - 已确认 Azure 上尚不支持 JSON 格式化。
- **Rerank 概览和代码帮助**：一位用户在 Rerank 概览文档中寻求帮助，在使用提供的代码时遇到了问题。
   - 该问题与文档过时有关，已提供修订后的代码片段。该用户还被引导至[相关文档](https://docs.cohere.com/reference/rerank)以供进一步参考。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TransformerDecoderLayer 重构落地**：已提交一个 PR 用于重构 TransformerDecoderLayer，涉及多个文件，并对 modules/attention.py 和 modules/transformer.py 进行了核心更改。
   - 此 PR 实现了 RFC #1211，旨在改进 TransformerDecoderLayer 架构，可以在此处找到：[TransformerDecoderLayer Refactor](https://github.com/pytorch/torchtune/pull/1312)。
- **RLHF 优选 DPO**：有一场关于使用 DPO 或 PPO 测试 HH RLHF 构建器的讨论，对于偏好数据集，DPO 是首选，而 PPO 则与数据集无关。
   - 重点在于 DPO，预期损失曲线与普通 SFT 相似，HH RLHF 构建器可能需要调试，这可能会在单独的 PR 中解决。
- **Torchtune WandB 问题已解决**：一位用户在访问 Torchtune 的 WandB 结果时遇到问题，在将该用户添加为团队成员后，访问权限已授予。
   - 该用户报告在使用默认 DPO 配置并关闭梯度累积（gradient accumulation）时结果较差，但后来发现它又开始正常工作了，可能是由于延迟或其他因素。
- **Torchtune 在 DPO 下的性能**：讨论了默认 DPO 配置可能导致 Torchtune 性能不佳的问题。
   - 用户建议尝试 SIMPO (Stack Exchange Paired) 并重新开启梯度累积，因为在 batch 中保持平衡的正负样本数量可以显著改善 loss。
- **PyTorch Conference：思想的汇聚**：关于即将举行的 PyTorch Conference 的讨论，包含网站链接和演讲嘉宾详情。
   - 您可以在此处找到有关会议的更多信息：[PyTorch Conference](https://events.linuxfoundation.org/pytorch-conference/)。还有人提到以“学术人员”身份混入会议，但这可能只是个玩笑。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Perplexity Pro 的推理能力**：一位用户注意到 [Perplexity Pro](https://perplexity.ai/) 的“推理能力变得疯狂地好”，并且能够“字面意义上数清字母”，就像它“抛弃了 tokenizer”一样。 
   - 他们分享了一个似乎与此主题相关的 [GitHub 仓库](https://github.com/cognitivecomputations/grokadamw) 链接。
- **Llama 3 MoE?**：一位用户询问是否有人制作了 Llama 3 的 “MoE” 版本。
- **梯度裁剪 (Grad Clipping) 揭秘**：一位用户询问了梯度裁剪的功能，特别是想知道当梯度超过最大值时会发生什么。
   - 另一位用户解释说，梯度裁剪本质上是将梯度限制在一个最大值，防止其在训练期间爆炸。
- **OpenAI 基准测试 vs 新模型**：一位用户对 OpenAI 发布基准测试而不是新模型感到惊讶。
   - 他们推测这可能是一个战略举措，旨在引导该领域转向更好的评估工具。
- **Axolotl 的功能**：一位成员指出 AutoGPTQ 可以完成某些事情，暗示 Axolotl 可能也能做到。
   - 他们对 Axolotl 复制这一功能的可能性感到兴奋。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grok 2.0 早期泄露**：一位成员分享了一条关于 [Grok 2.0 功能和能力](https://x.com/nima_owji/status/1823388838279922166) 的推文链接，包括使用 FLUX.1 模型生成图像。
   - 推文还指出 Grok 2.0 在编程、写作和生成新闻方面表现更好。
- **Flux.1 创造了一个拐点**：一位成员提到许多 Elon 的粉丝账号预测 X 将使用 MJ（推测指某个模型），暗示 Flux.1 可能在模型使用方面创造了一个拐点。
   - 该成员质疑 Flux.1 是否是 Schnellit 的 Pro 模型，考虑到 Elon 的过往经历。
- **开源图像标注工具搜索**：一位成员寻求推荐好的开源 GUI，以便快速高效地标注图像。
   - 该成员特别提到了单点标注、直线标注和绘制多边形分割掩码（polygonal segmentation masks）。
- **Elon 的模型虚张声势**：一位成员讨论了 Elon 使用 Grok 开发版本并对权重许可（weight licenses）虚张声势的可能性。
   - 该成员认为 Elon 可能会将其称为“红丸（red-pill）”版本。
- **2D 池化 (2D Pooling) 的成功**：一位用户对 2D 池化的效果感到惊讶。
   - 该用户指出这是由另一位用户推荐的，目前正在验证一种他们认为可能是自己发明的新位置编码（position encoding）的功效。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor 过滤性能？**：一位用户询问过滤 Tensor 的最快方法，例如 `t[t % 2 == 0]`，目前是通过转换为列表、过滤后再转回列表的方式实现的。
   - 有建议提到，如果是在 Tensor 的子集上进行计算，可以使用 masking（掩码），但指出目前尚不支持完全相同的功能。
- **超越函数折叠重构优化 (Transcendental Folding Refactor Optimization)**：一位用户提议进行重构，仅当后端没有针对该 `uop` 的 `code_for_op` 时才应用超越函数重写规则。
   - 该用户实现了一个 `transcendental_folding` 函数并在 `UOpGraph.__init__` 中调用，但不确定这如何能实现净减少代码行数，并询问可以删除哪些部分。
- **CUDA 超时错误 - 已解决**：一位用户在使用 `CLANG=1` 运行脚本时收到了 `RuntimeError: wait_result: 10000 ms TIMEOUT!` 错误。
   - 该错误发生在默认运行时，通过使用 `CUDA=1` 得到解决，该问题可能与 ##4562 相关。
- **Nvidia FP8 PR 建议**：一位用户对 **tinygrad** 的 Nvidia FP8 PR 提出了建议。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe 与 Agihouse 合作举办黑客松**：Poe (@poe_platform) 宣布与 Agihouse (@agihouse_org) 合作举办“Previews Hackathon”，以庆祝其扩大发布。
   - 该黑客松在 [AGI House](https://app.agihouse.org/events/poe-previews-hackathon-20240817) 举行，邀请创作者构建创新的“聊天内生成式 UI 体验 (in-chat generative UI experiences)”。
- **聊天内 UI 是未来**：Poe Previews Hackathon 鼓励开发者创建创新且实用的“聊天内生成式 UI 体验”，强调了生成式 AI 中用户体验的重要性。
   - 黑客松希望在竞争环境中展示参与者的创造力和技能。
- **虚拟试穿功能加速训练**：一位成员分享了构建虚拟试穿功能的经验，指出通过存储提取的特征可以有效加速训练运行。
   - 该功能使用在线预处理并将提取的特征存储在文档存储表中，从而在训练期间实现高效检索。
- **灵活的虚拟试穿功能**：一位成员询问了为虚拟试穿功能提取的具体特征。
   - 该成员详细说明了该通用的方法，成功适配了各种规模的模型，展示了其在处理计算需求和模型复杂性方面的灵活性。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Llama 3.1 8b 支持结构化输出**：一位用户确认 **Llama 3.1 8b** 可以通过 tool use 产生结构化输出，并已直接使用 **llama.cpp** 进行了测试。
- **RAG 在处理技术图像时遇到困难**：一位用户正在寻求关于从电气图、地图和电压曲线等图像中提取信息以用于技术文档 **RAG** 的建议。
   - 他们提到传统方法遇到了困难，强调需要捕获那些文本中不存在但专家可以视觉解读的信息。
- **Next.js POST 请求被误解为 GET**：一位用户在从运行在 **EC2** 上的 **Next.js Web 应用**向同一 **EC2** 实例上的 **FastAPI 端点**发送 **POST 请求**时遇到了 **405 Method Not Allowed** 错误。
   - 他们观察到，尽管在 **Next.js** 代码中明确使用了 **POST 方法**，请求仍被错误地解释为 **GET 请求**。
- **AWS pip install 问题已解决**：一位用户通过专门为 **Unix 环境**安装软件包，解决了 **AWS 系统**上的 **pip install** 问题。
   - 问题源于虚拟环境在 **pip install** 过程中错误地模拟了 **Windows**，导致了该问题。
- **Profundo 发布以自动化研究**：Profundo 自动化数据收集、分析和报告，使每个人都能对感兴趣的主题进行深度研究。
   - 它最大限度地减少错误并提高生产力，让用户能够专注于做出明智的决策。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter in Obsidian**: 一个新的 YouTube 系列将演示如何在 Obsidian 笔记应用中使用 Open Interpreter。
   - 该系列将重点介绍 Open Interpreter 插件如何让你控制你的 Obsidian vault，这可能对人们处理知识的方式产生重大影响。[这是第 0 集的链接](https://www.youtube.com/watch?v=HjcPRoPfri0)。
- **AI Agents in the Enterprise**: #general 频道的一位用户询问了在大型组织内监控和治理 AI Agent 的挑战。
   - 该用户邀请任何在企业内部从事 AI Agent 相关工作的人分享他们的经验。
- **Screenless Personal Tutor for Kids**: #O1 频道的一位成员提议使用 Open Interpreter 为儿童创建一个无屏幕的私人导师。
   - 该成员请求反馈，并询问是否有其他人有兴趣在这个项目上进行协作。
- **Convert Anything Tool**: "Convert Anything" 工具可以使用 Open Interpreter 将任何类型的数据转换为任何其他格式。
   - 该工具利用了 Open Interpreter 的强大功能，在各个领域都有巨大的应用潜力。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **SlimOrca Without Deduplication**: 一位用户询问了一个**移除了 soft prompting** 且**没有 deduplication** 的 **SlimOrca** 版本，最好包含代码。
   - 他们还询问是否有人实验过在有或没有 deduplication，以及有或没有 soft prompting 的数据上进行 fine-tuning (FT)。
- **Fine-tuning with Deduplication**: 该用户询问了使用 **soft prompting** 与不使用 soft prompting 进行 fine-tuning (FT) 的效果。
   - 他们还询问了在 **deduplicated data** 与 **non-deduplicated data** 上进行 fine-tuning (FT) 的效果。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Building an Agentic Jupyter Notebook Automation System**: 一位成员提议构建一个 Agentic 系统来自动化 Jupyter Notebook，旨在创建一个以现有 notebook 为输入、修改单元格并生成多个变体的流水线。
   - 他们寻求有关库、cookbook 或开源项目的建议，这些可以作为该项目的起点，并从 Devin 等类似工具中汲取灵感。
- **Automated Notebook Modifications and Validation**: 该系统应该能够智能地替换 Jupyter Notebook 中的特定单元格，并根据这些修改生成不同的 notebook 版本。
   - 至关重要的是，该系统应具备 Agentic 特性，使其能够验证其输出并迭代优化修改，直到达到预期的结果。



---


**Mozilla AI Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1272632011063824524)** (167 条消息🔥🔥): 

> - `Unsloth Pro`
> - `GPU 选择`
> - `LLM Leaderboard 结果`
> - `Dolphin 模型`
> - `模型微调` 


- **Unsloth Pro 早期访问**：目前正向 Unsloth 社区中受信任的成员提供 Unsloth Pro 版本的早期访问权限。
- **GPU 之争 - A100 vs A6000**：成员们讨论了 A100 与 A6000 GPU 的优缺点，一位成员指出 A6000 具有极佳的价格/VRAM 比，且与 24GB 显卡相比没有限制。
- **无审查模型表现优于 Meta Instruct**：一个经过微调以保留原始 Meta Instruct 模型智能的无审查模型已经发布，并在 LLM Leaderboard 2 上表现优于原始模型。
- **Dolphin 模型在审查方面遇到困难**：一位成员报告称 Dolphin 3.1 模型甚至无法处理最基础的请求并予以拒绝，这可能是由于其严格的审查机制。
- **模型微调技巧**：讨论围绕微调 Phi3 模型以及是否使用 LoRA 或全量微调展开，一位成员建议将 RAG 作为潜在解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1823366649074094225">Daniel Han (@danielhanchen) 的推文</a>: 发现 Llama 3.1 的聊天模板存在一些问题：1. 官方仓库添加了 2x \n 2. 官方仓库没有进行 strip / trim 3. 日期格式是 %B 而不是 %b（不是 3 个字母） 4. 官方仓库存在不一致 ...</li><li><a href="https://huggingface.co/spaces/featherless-ai/try-this-model">HF's Missing Inference Widget - a Hugging Face Space by featherless-ai</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1272632104056000512)** (12 条消息🔥): 

> - `露营`
> - `澳大利亚` 


- **露营很糟糕**：一位用户哀叹他们的露营经历，回来时身上有 6 处蚊子叮咬，其中一处还在眼睑上。
   - 另一位用户附和道，“永远不要去露营”。
- **澳大利亚比露营更糟**：一位用户说“在澳大利亚到处都是马粪”，暗示这比露营还糟。
   - 其他人表示赞同，并补充说澳大利亚有餐盘那么大的蜘蛛，而且“那里的所有东西都想杀掉你”。
- **亚马逊雨林是最糟糕的**：一位用户说他们“住在亚马逊雨林附近”，暗示那里比露营和澳大利亚都糟糕。
   - 这位用户还评论了一个名为“Cosine Genie - SOTA AI Engineer Announcement”的 YouTube 视频，该视频将 Genie 描述为“迄今为止世界上最好的 AI 软件工程师”。



**提到的链接**：<a href="https://www.youtube.com/watch?v=NvmB7ngopOY">Cosine Genie - SOTA AI Engineer Announcement</a>：Genie 是迄今为止世界上最好的 AI 软件工程师——在行业标准基准测试 SWE-Bench 上获得了 30% 的分数，打破了之前的 SOTA 记录...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1272637676977258627)** (83 条消息🔥🔥): 

> - `Unsloth 模型加载/保存`
> - `Llama 3.1 印地语微调`
> - `模型合并与 HF hub`
> - `Unsloth 配合 VLLM`
> - `数据集创建` 


- **使用 Unsloth 保存后找不到 ModelFile**：一位用户尝试使用 Unsloth 的 `model.save_pretrained_gguf` 方法保存模型，并注意到保存的文件夹中不包含 `ModelFile`。 
   - 另一位用户解释说，模型文件是以张量（tensors）形式保存的，整个文件夹对于配置和架构信息（包括 json 配置）是必需的。这是 GGUF 文件保存的正常方式，被拆分为多个配置文件和张量文件。
- **Unsloth 微调模型无法输出印地语摘要**：一位用户在印地语摘要数据上微调了 Llama 3.1 8B 模型并上传到 Hugging Face，但在推理过程中，它要么返回输入文本，要么返回英文摘要。 
   - 该用户分享了用于推理和微调的代码，其他用户建议可能是保存或加载自定义 Tokenizer 时出现了问题，或者 Hub 上的合并可能以一种奇怪的方式组合了层。
- **将 Unsloth 与 VLLM 结合使用**：一位用户在尝试将 Unsloth 微调后的模型与 VLLM 配合使用时遇到困难，并分享了包含其代码的 Colab 笔记本。 
   - 另一位用户建议参考 VLLM 文档进行故障排除，因为该文档以详细且有帮助著称。
- **创建自定义数据集**：一位用户询问有关以 CSV 或 JSONL 等格式创建包含自己信息的自定义数据集的资源。 
   - 用户建议使用 Hugging Face datasets，手动创建数据，或者使用更大的模型为他们生成数据。
- **Unsloth 内存占用问题**：一位用户遇到了一个问题，他们的 LLaMA 3 8B Instruct 模型在使用 Unsloth 时消耗了 300GB 的物理内存，但仍然遇到内存问题，导致服务器杀掉了模型进程。 
   - 用户建议检查可用的 VRAM，因为模型可能需要更多 GPU 显存才能正常运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/12pmRRQXunwvxeXxo97SUu5OuMaQ1IR3a?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://www.patched.codes/blog/a-comparative-study-of-fine-tuning-gpt-4o-mini-gemini-flash-1-5-and-llama-3-1-8b">GPT-4o-mini、Gemini Flash 1.5 和 Llama-3.1-8B 微调的对比研究</a>：我们使用自定义漏洞修复数据集对比了微调 GPT-4o-mini、Gemini Flash 1.5 和 Llama-3.1-8B 模型的效果，其中 GPT-4o-mini 显示出最显著的改进并设定了新的标准...</li><li><a href="https://www.deeplearning.ai/short-courses/finetuning-large-language-models/">微调大语言模型 - DeepLearning.AI</a>：掌握微调 LLM 的基础知识。区分微调与 Prompt Engineering，并获得处理真实数据集的实战经验。</li><li><a href="https://docs.vllm.ai/en/latest/">欢迎来到 vLLM！ &#8212; vLLM</a>：未找到描述</li><li><a href="https://x.com/labenz/status/1822321840385048950">来自 Nathan Labenz (@labenz) 的推文</a>：是否有推理平台为 Llama 3.1 模型提供 MoRA？这似乎是一个巨大的机会！（或者……为什么不呢？）抄送 @lqiao @FireworksAI_HQ @tri_dao @togethercompute @jefrankle @DbrxMosaicAI 感谢...</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-12.-saving-the-model">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：为创建可在 Ollama 上本地运行的定制化个人助手（如 ChatGPT）提供的初学者指南</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：为创建可在 Ollama 上本地运行的定制化个人助手（如 ChatGPT）提供的初学者指南</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1272735526247731307)** (8 messages🔥): 

> - `Lexi model`
> - `LLM Leaderboard 2`
> - `Ahma-3B Instruct`
> - `Finnish-NLP/Ahma-3B`
> - `Finnish language model` 


- **Lexi 模型击败原始 Instruct 版本**：LLM Leaderboard 2 公布了 Lexi 无审查版本的结果，这是一个经过微调的 Llama-3.1-8B 模型。
   - Lexi 不仅保留了原始 Instruct 的能力，在性能上甚至超越了它。
- **Ahma-3B Instruct - 芬兰语语言模型**：Ahma-3B 的指令微调版本已在 Hugging Face 发布，这是一个基于 Llama 从零开始预训练的芬兰语模型。
   - Ahma-3B Instruct 经过训练以遵循芬兰语指令，其基座模型是在 1390 亿个芬兰语 Token 上预训练的。
- **训练 Ahma-3B Instruct**：Ahma-3B Instruct 的训练过程涉及翻译和合成单轮及多轮数据，并使用基于 ClusterClipping 的采样和筛选。
   - 随后使用 Unsloth 框架通过 Qlora 进行有监督微调（SFT），并使用 DPO（Direct Preference Optimization，beta 为 0.1）进行微调。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored">Orenguteng/Llama-3.1-8B-Lexi-Uncensored · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Finnish-NLP/Ahma-3B-Instruct">Finnish-NLP/Ahma-3B-Instruct · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1272827506298196042)** (5 messages): 

> - `1.5-Pints`
> - `Tree Attention`
> - `Mistral`
> - `Llama 2`
> - `OpenELM` 


- **1.5-Pints：一种计算高效的语言模型**：展示了一个名为 "1.5-Pints" 的新语言模型，仅用 9 天时间通过计算高效的方法完成预训练。
   - 根据 MT-Bench 测试，该模型在指令遵循任务中优于 Apple 的 OpenELM 和 Microsoft 的 Phi 等 SOTA 模型。
- **Tree Attention：令人印象深刻的长上下文性能**：一篇研究论文讨论了通过使用 Tree Attention 在超长上下文性能方面取得的显著改进。
   - 论文（可在 [https://arxiv.org/pdf/2408.04093](https://arxiv.org/pdf/2408.04093) 获取）表明 Tree Attention 是处理长上下文的一种极具前景的方法。
- **1.5-Pints 架构与训练**：1.5-Pints 模型采用了修改后的 Mistral Tokenizer 和 Llama-2 架构以确保兼容性。
   - 其训练方法基于 StableLM、TinyLlama 和 Hugging Face 所使用的技术，强调了模型的多功能性。



**提及的链接**：<a href="https://arxiv.org/abs/2408.03506">1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data</a>：该论文提出了一种计算高效的方法，仅用 9 天时间预训练语言模型 "1.5-Pints"，同时在指令遵循助手任务中表现优于 SOTA 模型...

  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1272650789306765312)** (48 条消息🔥): 

> - `TorchAO`
> - `CUDA 开发者招聘`
> - `CPU matmul 优化`
> - `CPU matmul 性能`
> - `FP16/BF16 权重` 


- **TorchAO 在 Cohere for AI 的演讲**：来自 PyTorch Architecture Optimization 的 Charles Hernandez 将于 2024 年 8 月 16 日 20:00 CEST 在 Cohere For AI 的 ml-efficiency 小组展示 TorchAO 和量化技术。
   - 该活动由 @Sree_Harsha_N 主持，参与者可以通过链接 [https://tinyurl.com/C4AICommunityApp](https://tinyurl.com/C4AICommunityApp) 加入 Cohere For AI。
- **何处发布 CUDA 开发者招聘信息**：一位用户询问发布 CUDA 开发者职位空缺的最佳地点。
   - 未提供具体回答，但该用户被引导至 Discord 服务器内的 "jobs 频道"。
- **Zig 中的 CPU matmul 优化**：一位用户尝试在 Zig 中编写基于分块（tiling）的 matmul，但在实现最佳性能方面遇到困难。
   - 该用户分享了代码，并收到了关于探索缓存感知的循环重排以及使用 SIMD 指令可能性的建议。
- **CPU matmul 性能比较**：一位用户正在将他们在 Zig 中实现的 CPU matmul 性能与 GGML 和 NumPy 进行比较。
   - 该用户指出 NumPy 使用优化的 BLAS 实现达到了极快的性能，并分享了关于 CPU 上快速 MMM 资源的链接，包括 Sibboehm 的一篇文章和 Salykova 的一篇博客文章。
- **FP16/BF16 权重与 CPU 性能**：一位用户询问了在 CPU 上处理 FP16 权重的问题，并指出最近的模型通常使用 BF16。
   - 建议该用户将 FP16 权重转换为 BF16 或 FP32，其中 FP32 不会导致精度损失，但可能会降低推理速度。还建议该用户探索在运行时将张量（tensors）从 FP16 转换为 FP32 以潜在地提高性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://siboehm.com/articles/22/Fast-MMM-on-CPU">从零开始在 CPU 上实现快速多维矩阵乘法</a>：Numpy 可以在 4 核 Intel CPU 上以约 8ms 的速度完成两个 1024x1024 矩阵的乘法。考虑到这相当于每核心每周期 18 FLOPs，这非常快...</li><li><a href="https://x.com/Sree_Harsha_N/status/1823091293221691882">Sree Harsha (@Sree_Harsha_N) 的推文</a>：很高兴邀请到来自 @PyTorch Architecture Optimization 的 Charles Hernandez 在 8 月 16 日 20:00 CEST 的 ml-efficiency 小组讨论 TorchAO (https://github.com/pytorch/ao) 和量化。感谢 @m...</li><li><a href="https://huggingface.co/vikhyatk/moondream2/tree/main?show_file_info=model.safetensors">vikhyatk/moondream2 at main</a>：未找到描述</li><li><a href="https://salykova.github.io/matmul-cpu">在 150 行 C 代码中击败 NumPy：高性能多线程矩阵乘法教程</a>：在这个分步教程中，我们将从零开始在 CPU 上实现高性能多线程矩阵乘法，并学习如何在 C 语言中优化和并行化代码。在 Ryzen 7700 上，我们的实现...</li><li><a href="https://ppc.cs.aalto.fi),">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1272821075780042752)** (4 条消息): 

> - `PyTorch Full FP16`
> - `PyTorch Optimizer`
> - `torch.compile`
> - `Fairseq Fine-tuning` 


- **PyTorch Full FP16: 是否可行？**: 用户询问在 PyTorch 核心库中是否可以实现带有 loss/grad scaling 的全 FP16 训练，特别是在微调来自 Fairseq 的较大型模型时。
   - 他们尝试使用 `torch.GradScaler()` 并将模型转换为 FP16，而不使用 `torch.autocast('cuda', torch.float16)`，但遇到了错误 "ValueError: Attempting to unscale FP16 gradients."
- **自定义 Optimizer 实现**: 有用户建议在 optimizer 的 step 函数中手动访问并缩放梯度，以实现全 FP16 功能。
   - 他们提供了代码，演示了如何从 optimizer 的参数中获取梯度，并在调用 `optimizer.step()` 之前应用缩放操作。
- **torch.compile: 缺失的手册**: 一份名为 "torch.compile: The Missing Manual" 的新 PyTorch 文档与一段 YouTube 视频被分享。
   - 该文档和视频分别可在 [https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) 和 [https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf](https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf) 获取，提供了关于使用 `torch.compile` 的详细信息。
- **Fairseq 的训练方式**: 原提问者提到 Fairseq 模型通常使用其自定义的全 FP16 实现进行训练。
   - 他们还提到，虽然可以使用全 BF16 进行微调，但对于较小的 Fairseq 模型，FP16 AMP 通常表现更好，这可能是因为它们最初是用 FP16 训练的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab">torch.compile, the missing manual</a>: torch.compile, the missing manual。你来到这里是因为你想使用 torch.compile 让你的 PyTorch 模型运行得更快。torch.compile 是一个复杂且相对较新的软件，因此你...</li><li><a href="https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf">PyTorch Webinar: torch.compile: The Missing Manual</a>: 听取 Meta 的 PyTorch 研究工程师 Edward Yang 关于使用 torch.compile 手册的讲解。在此查看文档以同步学习：https://doc...</li><li><a href="https://github.com/pytorch/pytorch/blob/2e7d67e6af45c9338c02dd647c46c328fa23ee48/torch/amp/grad_scaler.py#L259-L260">pytorch/torch/amp/grad_scaler.py at 2e7d67e6af45c9338c02dd647c46c328fa23ee48 · pytorch/pytorch</a>: Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1272808678428512337)** (7 条消息): 

> - `Rust GPU`
> - `Zig`
> - `Fal Research Grants`
> - `Open Source Support` 


- **Rust GPU 转向社区所有**: [Rust GPU](https://rust-gpu.github.io) 项目此前由 [Embark Studios](https://www.embark-studios.com/) 管理，现在已转为社区所有，归属于 [Rust GPU GitHub organization](https://github.com/rust-gpu/rust-gpu)。
   - 这一转变标志着一项更广泛战略的开始，旨在振兴、统一和标准化 Rust 中的 GPU 编程。
- **Rust/Zig 对比 CUDA 的 C/C++**: 在 GPU 编程方面，Rust 和 Zig 提供了与 C 和 C++ 类似的优势，尽管它们尚未得到 CUDA 的官方支持。
   - 讨论强调了 Rust 和 Zig 的优势，并建议对于对该领域感兴趣的人来说，学习 Zig 可能会有所帮助。
- **面向开源 AI 项目的 Fal 研究资助**: [Fal Research Grants](https://fal.ai/grants) 计划为从事开源 AI 项目的研究人员和开发人员提供免费的算力资源。
   - 该计划面向任何热衷于通过开源项目推动 AI 发展的人士，无论其正式资历如何。
- **Fal 对开源项目的支持**: Fal 似乎正在积极支持众多开源项目。
   - 一位用户提到 Fal 资助了 [AuraFlow](https://auraflow.io/) 项目。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rust-gpu.github.io/blog/transition-announcement/">Rust GPU Transitions to Community Ownership |  </a>: 未找到描述</li><li><a href="https://fal.ai/grants">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1272661469959295066)** (11 条消息🔥): 

> - `CUDA Developers`
> - `CUDA Freshers`
> - `CUDA Hiring`
> - `CUDA Engineer`
> - `Triton` 


- **为机密 LLM 推理项目寻找 CUDA 开发者**：一家公司正在寻找一名 CUDA 开发者，负责一个与 LLM 推理速度相关的机密项目。
   - 他们正在寻找具备 Nvidia Nsight 深度知识、CUDA 编程技能、Hopper Architecture (SM90) kernel 经验、GPU 优化专业知识、TensorRT 和 TensorRT-LLM 技术诀窍，以及 AI/ML 框架经验（PyTorch, TensorRT）的人才。
- **应届生（Freshers）的 CUDA 技能：雇主看重什么**：讨论了雇主对申请 CUDA 工程师职位的应届生的期望。
   - 达成的共识是，能够独立完成一个非平凡（non-trivial）的 CUDA 或 Triton 程序、有效地沟通设计决策，并展示出学习潜能，是应届生的关键技能。
- **如何推销自己成为一名 CUDA 工程师**：一位成员强调了展示你如何与团队互补并带来他们尚未掌握的知识的重要性。
   - 他们建议展示你对团队工作的真实热情，并以好奇和投入的态度证明你能够迅速上手。


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1272670734493286523)** (7 条消息): 

> - `Multithreading and GPU Use`
> - `Network Requests and GPUs`
> - `Magnum IO Architecture` 


- **线程需要收敛（Converge）才能取得进展**：一位成员讨论了线程收敛对向前推进（forward progress）的重要性，指出 *"线程需要收敛才能向前推进……如果线程不收敛，那么向前推进可能会变得棘手。"*。 
   - 这突显了独立线程执行模型面临的挑战，即协调不同线程的工作对于整体进度至关重要。
- **GPU 与网络请求：为什么不？**：讨论集中在为什么 GPU 没有广泛用于网页爬虫等场景中的多线程网络请求，一位成员问道：*"为什么 GPU 没有广泛用于网页爬虫等用例的网络请求多线程处理？"* 
   - 回复指出，GPU 通常无法直接发起网络请求，虽然可能存在通过 PCIe 进行交互的技术手段，但这可能不是一个实用或高效的解决方案。
- **Magnum IO：数据中心架构的新时代**：一位成员分享了一篇关于 **Magnum IO** 的文章链接，这是一种为现代数据中心设计的新型 IO 子系统，被描述为 *"现代数据中心的 IO 子系统"*。 
   - 文章强调了计算单元从单机向整个数据中心的转变，强调了对分布式资源和数据集的需求，正如 Magnum IO 栈架构图所示。



**提到的链接**：<a href="https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-magnum-io-architecture/">Accelerating IO in the Modern Data Center: Magnum IO Architecture | NVIDIA Technical Blog</a>：这是“加速 IO”系列的第一篇文章，介绍了 Magnum IO（现代数据中心的 IO 子系统）的架构、组件、存储和优势。

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

iron_bound: https://www.youtube.com/watch?v=aNAtbYSxzuA
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1272671885133484073)** (126 条消息🔥🔥): 

> - `cuDNN 稳定性`
> - `HuggingFace Llama 3 AutoTokenizer 问题`
> - `Curand GPU 权重初始化`
> - `copy_and_cast_kernel`
> - `cudaMallocAsync/cudaFreeAsync` 


- **cuDNN 稳定性警告**：一名成员询问是否应该在运行过时的 cuDNN 时添加警告，特别是 **9.2.1** 和 **9.3.0** 版本，这些版本会显著影响稳定性。
   - 建议在 **Makefile** 层面实现检查，可能在构建过程中打印警告信息。
- **HuggingFace Tokenizer 特殊 Token 问题**：讨论了 **HuggingFace Llama 3 AutoTokenizer** 无法正确识别 **EOT token (<|endoftext|>)** 的问题，这可能导致代码修复中的潜在问题。
- **Curand GPU 权重初始化 PR**：一名成员提出了一种替代方案，使用 **curand** 直接在 GPU 上初始化权重，以实现更快的模型初始化。 
   - 该 PR 仍处于开发阶段，需要进一步的测试和清理。
- **copy_and_cast_kernel 过度工程化**：有人指出 **copy_and_cast_kernel** 可能存在过度工程化的问题，在 kernel 内部使用直接转换（casting）的更简单方法可能就足够了。
   - 然而，该成员选择不在这个特定的 PR 中进行更改，以避免引入潜在的兼容性问题。
- **cudaMallocAsync/cudaFreeAsync 优化**：建议通过为最大可能的 Tensor 大小使用单个 malloc/free 来优化关键循环中的 **cudaMallocAsync/cudaFreeAsync**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | 模型卡片与提示词格式</a>: Llama 3.1 - 最强大的开源模型。</li><li><a href="https://github.com/karpathy/llm.c/pull/741">[WIP] 由 ngc92 提交的用于模型初始化的初始 curand 实现 · Pull Request #741 · karpathy/llm.c</a>: 作为多线程模型初始化的替代方案，该方案使用 curand 直接在 GPU 上生成初始权重。目前仍在进行中，需要错误检查，而且我不喜欢 cudamallo...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/huggingface/tokenizers/pull/1419">由 ArthurZucker 提交的添加跳过特殊 token 选项 · Pull Request #1419 · huggingface/tokenizers</a>: 允许在编码时跳过特殊 token，修复了 #1347, #1391, #1368</li><li><a href="https://github.com/karpathy/llm.c/pull/740/commits/16635d41a2a7c0c21ec058eb4201ff75ab97e392">由 karpathy 提交的 Gordicaleksa 修复 dataloader2 · Pull Request #740 · karpathy/llm.c</a>: 在 @gordicaleksa PR 之上的 commit，包含了一系列错误修复：更明确地处理 EOT token，谨慎处理 AutoTokenizer 的 API，修复 fineweb.py 中的 dtype 错误，使用 model_desc 代替...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1272643914851029063)** (150 条消息🔥🔥): 

> - `Vision Adapters`
> - `Model Merging`
> - `Mistral Large`
> - `GPT-4o Mini`
> - `LLM Studio Headless` 


- **Vision Adapters：视觉模型的关键**：只有特定的 LLM 模型拥有 Vision Adapters，其中大多数以 "LLaVa" 或 "obsidian" 命名。
   - "VISION ADAPTER" 是视觉模型的关键组件；如果没有它，就会弹出你分享的那个错误。
- **Mistral Large：当前的冠军？**：一位成员发现 **Mistral Large 2** 是目前最好的 LLM，在处理困难的新颖问题上胜过 **Claude 3.5 Sonnet**。
   - 然而，该成员还指出 **Gemini Flash** 在价格上严重压低了 **OpenAI 4o mini**，但 **OpenAI 4o** 比 **Mistral Large** 更便宜。
- **LLM Studio 的 Model Explorer 宕机**：几位成员报告称，为 **LM Studio Model Explorer** 提供支持的 HuggingFace 宕机了。
   - 该网站被确认已连续数小时无法访问，多个地区都报告了连接问题。
- **Llama 3.1 性能问题**：一位用户报告称，他们的 **Llama 3 8B 模型** 现在的运行速度仅为 3 tok/s，而最近更新前为 15 tok/s。
   - 用户检查了他们的 GPU offload 设置并将其重置为默认值，但问题仍然存在；该问题似乎与最近更新中的更改有关。
- **LLM 输出长度控制**：一位成员正在寻找限制响应输出长度的方法，因为某些模型即使在被指示提供单句时也倾向于输出整个段落。
   - 虽然可以修改 System Prompts，但该成员发现 8B 模型（特别是 **Meta-Llama-3.1-8B-Instruct-GGUF**）在遵循精确指令方面表现不佳。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/iruletheworldmo/status/1823079598176731624">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>: 明天。太平洋时间上午 10 点。敬请关注。</li><li><a href="https://downforeveryoneorjustme.com/huggingface">Huggingface 宕机了吗？过去 24 小时的实时状态和问题</a>: Huggingface 的实时问题。收到错误？宕机？缓慢？检查发生了什么。</li><li><a href="https://huggingface.co/mradermacher/Tiger-Gemma-9B-v1-i1-GGUF">mradermacher/Tiger-Gemma-9B-v1-i1-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1272631729588408405)** (15 条消息🔥): 

> - `Portable LLM inference`
> - `Apple Mac`
> - `GPU modding`
> - `Copper modding`
> - `Flashing NVIDIA BIOS` 


- **便携式 LLM 推理：设备还是环境？**：一位成员认为，对于便携式 LLM 推理，应该考虑是想在便携式设备上运行推理，还是在移动时访问私有推理环境。
- **Apple Mac 的界面消耗内存**：一位成员讨论了 macOS 视觉效果（如透明度、模糊和阴影）的内存消耗，这些效果可能消耗高达 **3GB** 的内存。
- **为 AI 实验改装项目**：一位成员询问是否还有其他人致力于 AI 实验的改装项目，然后描述了他们的项目，即改装一块 **Asus ROG Strix RTX 2070 8GB OC**。 
- **铜改（Copper Modding）以获得更好的性能**：一位成员建议，铜改有助于散发显存芯片的热量，从而提高带宽并提升 LLM 推理速度。
- **将 NVIDIA BIOS 刷入 2080**：一位成员提到他们可能会将 RTX 2070 的 BIOS 刷成 2080，但他们需要先阅读相关流程。

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1272641182039806013)** (151 条消息🔥🔥): 

> - `Gemini Live`
> - `Google Fi`
> - `Strawberries`
> - `Project Astra`
> - `LLMs` 


- **Google 的 Gemini Live：现已推出，但并非免费**：Google 新推出的 Gemini Live 现已面向 **Advanced Subscribers** 开放，其特点是在 Android 上提供对话叠加功能，并支持更多连接的应用。
   - 普遍共识是，它比旧的语音模式有所改进，但局限在于仅限付费用户使用，且视频功能尚未上线。
- **Google Fi：值得切换吗？**：**Google Fi** 是 Google 的蜂窝网络，基于 T-Mobile。一些用户反映它是一个可靠的选择，但不足以让某些用户从 AT&T 切换过来。
   - 一位用户提到，Google Fi 本质上是优先级降低的 T-Mobile，虽然在带宽充足的地区不是问题，但对于覆盖范围有限的用户来说，它并不是理想的选择。
- **伟大的 Strawberry 辩论**：关于名为 "Strawberry"（带有一串表情符号）的神秘用户的讨论引发了猜测，认为该用户可能与 OpenAI 或 Sam Altman 有关。
   - 许多用户提到这是一种非常聪明的营销策略，将草莓表情符号与 Sam Altman 拿着草莓的照片联系起来，这在维持话题热度方面似乎非常有效。
- **Project Astra：它失败了吗？**：虽然 **Gemini Live** 的发布暗示了 **Project Astra**，但许多用户对没有看到进一步的进展感到失望。
   - 一位用户甚至提到了 **Microsoft Recall** 的对比，人们似乎不太相信该公司会很快发布这款产品，主要是出于安全考虑。
- **LLMs：解决所有问题的方案？**：一些用户对 LLMs 是否是所有问题的解决方案表示怀疑，特别是在将其用于数学、数据库甚至 waifu roleplay 等任务的背景下。
   - 其他人强调，理解 tokenization 仍然是 LLMs 的一个根本弱点非常重要，它们不能仅通过暴力 tokenization 来解决复杂问题，而是需要更具战略性的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.digitalheresy.com/p/strawberry">🍓🌱⛓️‍💥</a>：异教徒的草莓地之旅</li><li><a href="https://tenor.com/bSx3t.gif">Clueless Aware GIF - Clueless Aware Twitch - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1272659055764176906)** (5 条消息): 

> - `Prompt Library`
> - `System Prompt in LangChain` 


- **Prompt Library 位置**：一位用户询问如何访问提示词库，另一位用户提供了一个包含提示词库的频道链接。
- **在 LangChain 中添加 System Prompts**：一位用户分享了演示如何基于 Strawberry 创建 GPT 的 Python 代码，但他们想知道如何添加系统提示词 (system prompt)。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1272980860282343516)** (3 条消息): 

> - `ChatGPT website access` 


- **ChatGPT 无法访问网站**：一位成员询问是否有办法让 ChatGPT 访问网站并提取文章供其阅读。
- **ChatGPT 可能会对网站内容产生幻觉**：另一位成员建议 ChatGPT 可能会产生幻觉 (hallucinate) 或被阻止抓取该网站。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1272980860282343516)** (3 条消息): 

> - `ChatGPT accessing websites`
> - `ChatGPT's hallucination and web crawling` 


- **ChatGPT 无法直接访问网站**：一位用户询问是否有办法让 ChatGPT 访问网站并提取文章供其阅读。
- **ChatGPT 可能会产生幻觉或被阻止抓取网站**：一位用户建议 ChatGPT 可能会产生幻觉或被阻止抓取网站，并解释说它有时会引用网站来源，但并非总是如此。
- **Web browser GPT**：一位用户询问是否有人尝试向 ChatGPT 提及 web browser GPT，这可能是绕过此限制的一种方法。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1272631560469741764)** (106 条消息🔥🔥): 

> - `Perplexity bug reports`
> - `Perplexity Pro Models`
> - `Perplexity's UI/UX`
> - `Perplexity's website stability`
> - `Perplexity's future` 


- **Perplexity 的 UI/UX 问题**：用户报告了多个 UI/UX 问题，包括按钮缺失和提示词输入框消失，导致与平台交互困难。
- **新模型：Sonar Huge**：新模型 "Sonar Huge" 取代了 Perplexity Pro 中的 Llama 3.1 405B 模型，据观察其运行缓慢且不遵循用户个人资料中的提示词。
- **Perplexity 网站稳定性担忧**：用户报告网站稳定性显著下降，理由包括搜索行为不稳定、遗忘上下文以及 Web 和 iOS 版本的界面错误。
- **Perplexity 成功团队承认存在 Bug**：Perplexity 的成功团队承认收到了用户关于近期平台出现的 Bug 和故障的反馈。
- **Perplexity 功能实现的未来**：一位用户对功能实现的漫长等待时间表示沮丧，强调了承诺的功能与实际推出速度之间的差距。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1823029449534615705?s=46&t=JsxhFTRLBknd8RUv1f73bA">来自 Perplexity (@perplexity_ai) 的推文</a>: 我们很高兴宣布与 @Polymarket 建立合作伙伴关系。现在，当你在 Perplexity 上搜索事件时，你将看到新闻摘要与实时概率预测相结合，例如选举...</li><li><a href="https://status.perplexity.com/history/1">通知历史 - Perplexity - 状态</a>: 通知历史 - Perplexity 状态
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1272637873912414238)** (9 条消息🔥): 

> - `Coursera`
> - `Programming Courses`
> - `AI/ML`
> - `Cloud Computing`
> - `Data Science` 


- **Coursera 的高薪编程课程**：Coursera 提供广泛的编程课程，可通向高薪技术职业，其中 **Python、AI/ML、Cloud Computing 和 Data Science** 等专业方向尤为受欢迎。
   - 这些课程由 **Stanford、Google 和 IBM** 等机构提供，帮助学习者培养热门技能并提升职业前景。
- **技术职业的战略技能发展**：为了最大化收入潜力，Coursera 建议将技术技能与 **项目管理和沟通** 等软技能相结合，这可以帮助学习者在竞争激烈的领域中脱颖而出。
   - 通过战略性地结合这些技能，学习者可以为 **Software Engineering、Data Science 和 Cloud Architecture** 等高薪职位做好准备。
- **实践经验和保持更新的重要性**：Coursera 强调选择提供 **实践经验** 并紧跟新兴技术的综合项目的重要性。
   - 这确保了学习者能够获得当前就业市场中最相关且最有价值的技能。
- **Perplexity AI 聊天机器人关于可共享线程的指导**：Perplexity AI 聊天机器人提醒用户确保其线程是 **可共享的**，并提供了说明链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/81Pey4X6SY0">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/link-me-the-best-coursera-cour-Re4JWGgnTDmDZ_06LX4Z_Q">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/could-you-analyse-the-image-an-Os_cDlCGRbelIR.2YAc4Lw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/who-is-princess-elara-self-awa-wMoqIeyRS9.RXRMRA8nL9g">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/how-many-fish-are-caught-every-szC5dPGSTFysBuOJZ9qeKA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/page/best-coursera-courses-for-prog-j1R20PNzTsqDCrrk6_Fk3A">未找到标题</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/whats-the-most-common-group-of-gGB6elWUR8KdkYVrYdhZ1Q">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1272937147841773739)** (6 messages): 

> - `Perplexity Search Parameters`
> - `Search Location Options`
> - `Image Generation from Narrative` 


- **搜索参数控制**：一位成员表示有兴趣控制搜索参数，如 `intitle:ABC`，类似于 Google Search。
   - 他们认为这一功能将大大增强 Perplexity 的搜索能力。
- **搜索位置选择**：另一位成员询问了在 Perplexity 中选择特定搜索位置的可能性。
   - 他们承认这一功能对于缩小搜索结果范围和查找特定位置信息具有价值。
- **基于叙述的图像生成**：一位用户提供了一段叙述，描述了一只猫在雨中沿着墙行走的场景，并请求根据此描述生成图像。
   - 该叙述包括了关于摄像机角度、光影和氛围的细节，表明了对 AI 驱动的图像生成的兴趣。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1272675105520226400)** (1 messages): 

> - `SXSW Panel`
> - `OpenAI Models`
> - `AI Risks and Opportunities`
> - `Government Regulation`
> - `AI Impact` 


- **Stability AI 的 SXSW 研讨小组提案**：Stability AI CEO Prem Akkaraju 和科技影响力人物 Kara Swisher 将讨论开源 AI 模型的重要性以及政府在监管其影响方面的作用。
   - 研讨小组将探讨 AI 的机遇和风险，包括工作流失、虚假信息、CSAM 和 IP 权利。
- **尖端技术访问的民主化**：研讨小组将强调开源 AI 模型如何推动创新并使技术访问民主化，特别是在 CGI 领域。
   - 这种可访问性促进了实验并加速了各个领域的进步，为个人和组织赋予了新的可能性。
- **平衡商业利益与 AI 风险**：讨论将涉及在快速发展的行业中平衡商业利益与生成式 AI 潜在风险的挑战。
   - 关键议题将包括减轻虚假信息、保护知识产权以及解决围绕 AI 使用的伦理问题。
- **AI 对内容创作和工作的未来影响**：研讨小组将探讨 AI 在内容创作、工作、教育和其他领域的未来。
   - 他们将讨论 AI 在提升所有社会和经济阶层人类潜力方面的能力，同时考虑其对就业和技能发展的影响。
- **Prem Akkaraju 对 AI 和 CGI 的愿景**：Prem Akkaraju 将分享他对公司和行业的愿景，重点关注 AI 和 CGI 的融合。
   - 他将讨论这种融合将如何改变创意领域，并为内容创作和叙事提供新的可能性。



**提及的链接**：<a href="http://panelpicker.sxsw.com/vote/153232">PanelPicker | SXSW Conference &amp; Festivals</a>：PanelPicker® 是官方的 SXSW 用户生成会议提案平台。输入想法并投票，以帮助塑造 SXSW 和 SXSW EDU 的会议议程。

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1272658083738292427)** (111 messages🔥🔥): 

> - `Google Colab Runtime`
> - `Stable Diffusion Installation`
> - `Stable Diffusion Model Merging`
> - `CUDA Installation`
> - `Flux Realism` 


- **Google Colab Runtime 停止**: 一位用户询问如何防止他们的 Google Colab Runtime 停止运行。
   - 另一位用户建议改用 Kaggle，因为它提供更多资源和更长的运行时间。
- **Stable Diffusion 安装困扰**: 一位用户报告了安装 Stable Diffusion 时的困难，遇到了 CUDA 安装问题以及在 Hugging Face 上查找 Token 的问题。
   - 另一位用户提供了通过个人资料设置菜单生成 Token 以及正确安装 CUDA 的指导。
- **模型合并问题**: 一位用户讨论了潜在的模型合并策略，提议将 UltraChat 与基础 Mistral 之间的差异应用于 Mistral-Yarn。
   - 其他用户表示怀疑，但原用户保持乐观，并引用了以往成功的模型合并尝试。
- **用于面部交换的 Flux Realism**: 一位用户询问如何使用 Flux Realism 将自己的脸部放入图像中。
   - 他们提到尝试了 fal.ai，但结果看起来像卡通，因此寻求替代方案。
- **训练用于 Logo 生成的 LORAs**: 一位用户寻求训练用于 Logo 生成的 LORAs 的指导，特别是将 Logo 放置在图像上。
   - 另一位用户推荐使用 Flux，因为它可以针对 Logo 进行训练，并准确地将其放置在图像上（如衬衫或建筑物）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: 您的机器学习和数据科学社区</a>: Kaggle 是全球最大的数据科学社区，拥有强大的工具和资源来帮助您实现数据科学目标。</li><li><a href="https://tenor.com/view/heart-container-goddess-statue-totk-heart-container-totk-zelda-gif-891944359093961229">Heart Container Goddess Statue GIF - Heart container Goddess statue Totk heart container - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/CS1o/">CS1o - 概览</a>: CS1o 有 2 个可用的代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">安装指南</a>: Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#step1-make-environment">GitHub - cumulo-autumn/StreamDiffusion: StreamDiffusion: 用于实时交互生成的流水线级解决方案</a>: StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation - cumulo-autumn/StreamDiffusion
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1272894465169690707)** (2 messages): 

> - `Gemini Flash 1.5`
> - `GPT-4o Extended`
> - `OpenRouter Pricing` 


- **Gemini Flash 1.5 降价**: **Gemini Flash 1.5** 的输入 Token 成本降低了 **78%**，输出 Token 成本降低了 **71%**。
   - 这使得该模型对更广泛的用户来说更易于获取且更负担得起。
- **GPT-4o Extended 早期访问发布**: **GPT-4o Extended** 已通过 **OpenRouter** 开启早期访问。
   - 您可以通过此链接访问：[https://x.com/OpenRouterAI/status/1823409123360432393](https://x.com/OpenRouterAI/status/1823409123360432393)
- **GPT-4o Extended 输出限制**: GPT-4o Extended 允许的最大输出 Token 数量为 **64k**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1823409123360432393">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 您现在可以通过 OpenRouter 使用 GPT-4o extended output (Alpha 访问)！最大 64k Tokens</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5>)">Gemini Flash 1.5 - API, 提供商, 统计数据</a>: Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现良好，如视觉理解、分类、摘要以及从图像、音频和视频中创建内容...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1272630857756315708)** (80 条消息🔥🔥): 

> - `Gemini Flash 价格更新`
> - `GCP 成本表`
> - `Token:Character 比例`
> - `Euryale 70B 停机时间`
> - `Infermatic 停机时间` 


- **OpenRouter 上的 Gemini Flash 价格？**：一位用户询问了 OpenRouter 上新的 Gemini Flash 价格以及何时会进行更新。
   - 有用户提到他们的 GCP 成本表已经反映了新价格，这表明需要由 OpenRouter 来实施更新。
- **OpenRouter 的更新障碍**：OpenRouter 的更新受到了 Gemini 新的 1:4 Token:Character 比例的阻碍，该比例无法清晰地映射到 `max_tokens` 参数验证中。
   - 另一位用户对不断变化的 Token:Character 比例表示沮丧，并建议切换到按 Token 计费的系统。
- **Euryale 70B 问题？**：一位用户报告 Euryale 70B 对某些用户不可用，但对他们自己正常，从而引发了关于故障或错误率的疑问。
   - 进一步的讨论揭示了多次停机情况，包括一次因更新导致的 10 分钟停机，以及可能存在的区域可用性问题。
- **模型性能对比**：用户对比了 Groq 70b 和 Hyperbolic 的性能，发现针对相同 Prompt 的结果几乎完全一致。
   - 这引发了关于 FP8 量化影响的讨论，一些用户指出在实践中差异极小，但另一些用户则指出某些提供商可能会出现质量下降。
- **ChatGPT 4.0 默认设置更改**：一位用户表示担心 "middle-out" 设置不再是 ChatGPT 4.0 的默认设置，这影响了其前端的 Function Calling。
   - 该用户请求关于如何在 Ollama 和 Wordpress 插件等平台的 System Prompt 中设置此参数的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/MultiOn_AI/status/1823412701441482959">MultiOn (@MultiOn_AI) 的推文</a>：宣布我们最新的研究突破：Agent Q - 带来具有规划和 AI 自愈能力的下一代 AI Agent，相比 LLama 3 的基准 Zero-shot 提升了 340%...</li><li><a href="https://x.com/btibor91/status/1821452608046788853?t=_Pnb_GU6R2TZpd8DxZaHOA&s=19">Tibor Blaho (@btibor91) 的推文</a>：有人在过去几小时内过早发布了太多新文章（和测试文章？）——预计很快会有 GPT-4o System Card、SWE-bench Verified、新客户案例等——正在与 The ... 合作</li><li><a href="https://huggingface.co/deepseek-ai">deepseek-ai (DeepSeek)</a>：未找到描述</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>：Codeium 的使命、Cortex 和 Forge 的发布以及详细愿景。</li><li><a href="https://openrouter.ai/models/sao10k/l3-lunaris-8b">Llama 3 8B Lunaris - API, Providers, Stats</a>：Lunaris 8B 是一款基于 Llama 3 的多功能通用和角色扮演模型。它是多个模型的战略合并，旨在平衡创造力与改进的逻辑和通用知识。R...</li><li><a href="https://openrouter.ai/models/aetherwiing/mn-starcannon-12b">Mistral Nemo 12B Starcannon - API, Providers, Stats</a>：Starcannon 12B 是一款创意角色扮演和故事写作模型，使用 [nothingiisreal/mn-celeste-12b](https://openrouter.ai/models/nothingiisreal/mn-celeste-12b) 作为基础，并结合了 [intervitens/mini-magnum-...</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 - API, Providers, Stats</a>：Euryale 70B v2.1 是来自 [Sao10k](https://ko-fi) 的专注于创意角色扮演的模型。通过 API 运行 Llama 3 Euryale 70B v2.1
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1272652219547779072)** (30 条消息🔥): 

> - `Mojo Licensing Concerns` (Mojo 许可关注)
> - `Mojo Open-Sourcing` (Mojo 开源)
> - `Mojo Development` (Mojo 开发)
> - `Mojo Learning Resources` (Mojo 学习资源)
> - `Mojo Compiler` (Mojo 编译器)


- **Mojo 的许可证存在限制条件**：Mojo 许可证禁止使用该语言开发任何用于竞争性活动的应用程序。
   - 然而，条款中提到该规则不适用于在初始发布后才产生竞争关系的应用程序，但目前尚不清楚该条款将如何执行。
- **Mojo 不确定的开源前景**：用户询问了 Mojo 编译器的开源时间表。
   - 虽然团队确认编译器最终会开源，但目前没有公开的时间表，这表明在能够进行贡献之前可能还需要一段时间。
- **Mojo 开发：专注于标准库**：目前 Mojo 开发的重点是构建标准库（Standard Library）。
   - 鼓励用户向标准库贡献代码，而编译器的开发工作虽然在进行中，但尚未开放贡献。
- **面向学生的 Mojo 学习**：一名大学生询问了学习 Mojo 的资源，包括编译器基础和 MLIR。
   - 团队建议从学习 Mojo 本身并向标准库贡献开始，而 MLIR 知识对于未来的编译器贡献将会有所帮助。
- **Mojo 编译器：有限的文档和内部 Dialects**：团队承认 Mojo 的内部 Dialects 缺乏文档，这给一些贡献者的开发带来了挑战。
   - 团队正在考虑直接向编译器添加重写规则（rewrite rules）的功能，但目前尚不可行，导致一些人由于逆向工程编译器需要投入大量时间而搁置了他们的项目。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp=sharing">Small string optimization in Mojo’s stdlib</a>：Mojo 标准库中的小字符串优化以及小缓冲区优化</li><li><a href="https://docs.google.com/presentation/?usp=slides_web">未找到标题</a>：未找到描述</li><li><a href="https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp%3Dsharing&followup=https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp%3Dsharing&ltmpl=slides&ec=GAZAmQI)">未找到标题</a>：未找到描述</li><li><a href="https://support.google.com/docs/answer/2375082?hl=en).[Dismiss](#)>>>">系统要求和浏览器 - 电脑 - Google 文档编辑器帮助</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1272631576089595905)** (19 条消息🔥): 

> - `Java by Microsoft`
> - `C# relevance`
> - `Stable Diffusion Memory Issue`
> - `WSL2 limitations`
> - `Mojo Optimization` 


- **Microsoft 的 Java：一个被遗忘的巨人？**：一位成员认为“Microsoft 的 Java”是不必要的且本可以避免，而另一位成员则反驳说这在当时似乎至关重要。
   - 讨论承认了新解决方案的出现以及“Microsoft 的 Java”随时间的衰落，强调了它 20 年的历程及其在 Microsoft 市场份额中的相关性。
- **C# 在 Microsoft 统治地位中的崛起**：C# 出现于 2000 年，二十多年来一直是 Windows 开发的关键部分，被许多任务视为“更好的 Java”。
   - C# 作为“在 Windows 上开发应用程序的新方式”迅速普及，特别是在 Windows 占据重要地位的发展中国家。
- **WSL2 中 Stable Diffusion 的内存问题**：一位新用户在 WSL2 中运行 Stable Diffusion Mojo ONNX 示例时遇到问题，进程因内存压力被终止。
   - 该用户为 WSL2 分配了 8GB 内存，但被建议增加一倍，因为 Stable Diffusion 1.5 大约为 4GB，模型和优化过程需要更多内存。
- **WSL2 内存限制**：Windows 优先保证宿主操作系统的健康，而不是 WSL2 进程，这在运行 Stable Diffusion 等内存密集型应用程序时可能会导致内存受限。
   - 建议将分配给 WSL2 的内存从 8GB 增加到 16GB 以缓解此问题。
- **Mojo 优化：内存效率**：讨论了 Stable Diffusion 的内存效率，指出优化过程可能会消耗大量 RAM。
   - 建议用户为 WSL2 分配更多内存，以确保为模型及其优化过程提供足够的资源。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1272683094180102157)** (11 messages🔥): 

> - `Cohere For AI`
> - `Pricing changes`
> - `Cohere's Research Lab`
> - `Hackathons`
> - `Computer Vision` 


- **Cohere For AI 研究实验室**: **Cohere For AI** 是一个致力于解决复杂机器学习问题的非营利研究实验室。
   - 他们支持探索未知的基本研究，并专注于为机器学习研究创造更多切入点。 
- **Cohere 网站上的价格变动**: 一位用户询问了 **classify** 功能的定价，注意到它不再列在定价页面上。
   - 未给出回复。
- **Hackathon 小组**: 一位用户正在寻找更多成员加入他们的 Hackathon 小组。
   - 该小组目前有 2-3 人，他们正在寻找拥有多样化技能的人才，特别是那些能够提交视频的人。
- **对 Computer Vision 的兴趣**: 一位新用户介绍自己是计算机工程专业的毕业生，对 AI、ML、DL，特别是 Computer Vision 感兴趣。
   - 他们已经做过一些与 CV 相关的项目，并希望在这一领域有所提高。



**提到的链接**: <a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI 是一个致力于解决复杂机器学习问题的非营利研究实验室。我们支持探索未知的基本研究，并专注于创造更多...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1272669301207203841)** (26 messages🔥): 

> - `JSONL Upload Issue`
> - `Azure JSON Formatting`
> - `Rerank Overview`
> - `Cohere API Usage`
> - `Python Kernel Restart` 


- **JSONL 上传失败**: 用户报告了在上传用于微调的 JSONL 数据集时遇到问题，错误信息为 "File format is not supported for dataset"（数据集不支持该文件格式）。
   - Cohere 支持团队已确认该问题并正在调查。在此期间，用户可以使用 API 进行数据集创建，目前该功能运行正常。
- **Azure 不支持 JSON 格式化**: 一位成员询问在 Azure 中通过 `response_format` 使用结构化输出的问题，但遇到了参数无效的错误。
   - 已确认 Azure 目前不支持 JSON 格式化。
- **Rerank 概览代码帮助**: 一位用户在阅读 Rerank 概览文档时请求帮助，因为提供的代码遇到了问题。
   - 该问题与文档过时有关，已提供修改后的代码片段。该用户还被引导至相关文档以供进一步参考。
- **Rerank 中的 "Unknown Field" 错误**: 一位用户在使用 Rerank API 时遇到了 "unknown field"（未知字段）错误。
   - 已确认此错误与 Rerank API 无关，建议重启 Python kernel 作为潜在的解决方案。



**提到的链接**: <a href="https://docs.cohere.com/reference/rerank">Rerank - Cohere API References</a>: 该端点接收一个查询和一组文本，并生成一个有序数组，为每个文本分配一个相关性分数。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1272635724767695009)** (7 messages): 

> - `JSON Snippet Embeddings`
> - `Intermediate Text` 


- **JSON 片段的 Embeddings**: 一位成员询问了将 JSON 作为文档片段提供的首选方法，旨在实现与大型 JSON 数据集的兼容性。
- **Intermediate Text 的效用**: 一位成员询问了中间文本（intermediate text）的用途。
- **以 Embeddings 作为解决方案**: 一位成员建议将 JSON 转换为 Embeddings 作为可能的解决方案。
- **明确目标**: 另一位成员要求澄清预期目标，并表示愿意协助寻找解决方案。

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1272661992057868350)** (44 条消息🔥): 

> - `TransformerDecoderLayer Refactor`
> - `RLHF with DPO/PPO`
> - `Torchtune & WandB`
> - `Torchtune Performance`
> - `PyTorch Conference` 


- **TransformerDecoderLayer 重构 PR**：已提交一个用于重构 `TransformerDecoderLayer` 的 PR，涉及多个文件，并对 `modules/attention.py` 和 `modules/transformer.py` 进行了核心更改。
   - 该 PR 实现了 RFC #1211，旨在改进 `TransformerDecoderLayer` 架构。
- **结合 DPO/PPO 的 RLHF**：讨论了使用 DPO 或 PPO 测试 HH RLHF 构建器的问题，偏好数据集倾向于使用 DPO，而 PPO 则与数据集无关。
   - 重点在于 DPO，预期损失曲线与普通 SFT 相似，HH RLHF 构建器可能需要调试，这可能会在单独的 PR 中解决。
- **Torchtune 与 WandB 问题**：用户在访问 Torchtune 的 WandB 结果时遇到问题，在将用户添加为团队成员后获得了访问权限。
   - 用户报告使用默认 DPO 配置并关闭梯度累积（gradient accumulation）时结果较差，但后来发现它又恢复正常了，可能是由于延迟或其他因素。
- **Torchtune 的 DPO 性能**：讨论了默认 DPO 配置可能导致 Torchtune 性能不佳的问题。
   - 用户建议尝试 SIMPO (Stack Exchange Paired) 并重新开启梯度累积，因为在 batch 中保持正负样本数量平衡可以显著改善 loss。
- **PyTorch 大会**：讨论了即将举行的 PyTorch 大会，包含网站链接和演讲嘉宾详情。
   - 还提到有人开玩笑说想以“学术人员”身份混入大会。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/rafi-personal/torchtune?nw=nwuserrdoublea">rafi-personal</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://events.linuxfoundation.org/pytorch-conference/">PyTorch Conference | LF Events</a>：加入顶尖研究人员、开发者和学者的行列，深入探讨 PyTorch 这一前沿开源机器学习框架。</li><li><a href="https://github.com/pytorch/torchtune/pull/1312">TransformerDecoderLayer Refactor by pbontrager · Pull Request #1312 · pytorch/torchtune</a>：上下文 该 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此添加）。这是对同名 RFC #12... 的实现。</li><li><a href="https://github.com/pytorch/torchtune/pull/645#issuecomment-2047853377">DPO by yechenzhi · Pull Request #645 · pytorch/torchtune</a>：上下文 将 DPO 集成到 Torchtune 中，更多详情见此处。变更日志... 测试计划...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1272652228884299847)** (19 条消息🔥): 

> - `Perplexity Pro`
> - `Llama 3`
> - `Grad Clipping`
> - `OpenAI Benchmark` 


- **Perplexity Pro 的推理能力**：用户注意到 [Perplexity Pro](https://perplexity.ai/) 的推理能力变得“强得离谱”，能够“逐个数出字母”，就像“抛弃了 tokenizer”一样。
   - 他们分享了一个似乎与此主题相关的 [GitHub 仓库](https://github.com/cognitivecomputations/grokadamw) 链接。
- **Llama 3 与混合专家模型 (MoE)**：一位用户询问是否有人制作了 Llama 3 的 “MoE” 版本。
- **梯度裁剪 (Grad Clipping) 详解**：用户询问梯度裁剪的功能，特别是当梯度超过最大值时会发生什么。
   - 另一位用户解释说，梯度裁剪本质上是将梯度限制在一个最大值，防止训练过程中梯度爆炸。
- **OpenAI 发布基准测试**：用户对 OpenAI 发布基准测试而非新模型感到惊讶，推测这可能是引导该领域走向更好评估工具的战略举措。



**提到的链接**：<a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>：在 GitHub 上为 cognitivecomputations/grokadamw 的开发做出贡献。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1272940875584110666)** (4 messages): 

> - `AutoGPTQ`
> - `Axolotl` 


- **Axolotl 的功能**：一位成员注意到 AutoGPTQ 可以完成某些任务，暗示 Axolotl 可能也能做到。
   - 他们对 Axolotl 复制这一功能的可能性感到兴奋。
- **OpenAI 的 Sharegpt API**：一位成员建议在 API 中使用 `type: sharegpt` 和 `conversation: llama` 以获得更理想的结果。
   - 这一建议表明在使用 Axolotl 时，对 API 中的某些参数有所偏好。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[deployment-help](https://discord.com/channels/1104757954588196865/1163840836472148058/1272943033608048722)** (1 messages): 

> - `LLM Inference`
> - `VLLM`
> - `SkyPilot`
> - `Fireworks`
> - `Lora Adapters` 


- **在自有 GPU 上通过 SkyPilot 进行 VLLM 推理**：一位成员建议在自有 GPU 上使用 VLLM，并通过 SkyPilot 进行管理，以获得更大的灵活性。
   - 这种设置允许完全控制并能处理特定需求。
- **Fireworks 的 Serverless 计费**：Fireworks 被提及作为提供 Lora Adapters 服务并采用 Serverless 计费的合适解决方案。
   - 然而，Fireworks 存在局限性，包括与所有基础模型的兼容性以及偶尔出现的异常情况。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1272921653776617564)** (16 messages🔥): 

> - `Grok 2.0`
> - `Flux.1 Model`
> - `Grok Image Generation`
> - `Open Source Image Annotation`
> - `Elon and Models` 


- **Grok 2.0 提前泄露**：一位成员分享了一个关于 Grok 2.0 特性和能力的推文链接，包括使用 FLUX.1 模型生成图像。
   - 推文还指出 Grok 2.0 在代码编写、写作和新闻生成方面表现更好。
- **Flux.1 模型的使用**：一位成员提到许多 Elon 的粉丝账号预测 X 会使用 MJ（推测指某个模型），这表明 Flux.1 可能已经成为了模型使用的转折点。
   - 考虑到 Elon 的过往经历，该成员质疑 Flux.1 是否就是 Schnellit 的 Pro 模型。
- **开源图像标注**：一位成员寻求推荐用于快速高效标注图像的优秀开源 GUI。
   - 该成员特别提到了单点标注、直线标注以及绘制多边形分割掩码。
- **Elon 的模型选择**：一位成员讨论了 Elon 可能正在使用 Grok 的开发版本，并对权重许可（weight licenses）虚张声势的可能性。
   - 该成员认为 Elon 可能会将其称为“红丸”（red-pill）版本。



**提及的链接**：<a href="https://x.com/nima_owji/status/1823388838279922166">来自 Nima Owji (@nima_owji) 的推文</a>：突发：这是 Grok 2.0 特性和能力的早期预览！它在代码编写、写作和新闻生成方面表现更好！它还将使用 FLUX.1 模型生成图像！

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1272842479200370728)** (4 messages): 

> - `Position Encoding`
> - `2D Pooling` 


- **新的位置编码？**：一位用户认为他们可能发明了一种更优的位置编码类型，目前正在验证其有效性。
- **2D Pooling 的成功**：该用户对 2D Pooling 的效果感到惊讶，并指出这是由另一位用户推荐的。


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

flammit_: 没问题——刚刚在你的 NVIDIA FP8 PR 中留下了一些希望能有所帮助的提示。
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1272665251762147409)** (8 messages🔥): 

> - `Tensor Filtering`
> - `Transcendental Folding Optimization`
> - `CUDA TIMEOUT ERROR` 


- **Tensor 过滤的最快方法？**：一位用户询问过滤 Tensor 的最快方法，例如 `t[t % 2 == 0]`，目前他们是通过转换为列表、过滤后再转回列表的方式实现的。
   - 有人建议如果在 Tensor 的子集上进行计算，可以使用 masking（掩码），但指出目前尚无法实现完全相同的功能。
- **超越函数折叠（Transcendental Folding）重构优化**：一位用户提议进行重构，仅当后端没有针对该 uop 的 `code_for_op` 时才应用超越函数重写规则。
   - 该用户实现了一个 `transcendental_folding` 函数并在 `UOpGraph.__init__` 中调用它，但不确定这如何能实现代码行数的净减少，并询问可以删除哪些内容。
- **CUDA TIMEOUT ERROR**：一位用户在使用 `CLANG=1` 运行脚本时遇到了 `RuntimeError: wait_result: 10000 ms TIMEOUT!` 错误。
   - 该错误发生在默认运行时，通过使用 `CUDA=1` 解决，该问题可能与 ##4562 有关。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1273021815517741076)** (3 条消息): 

> - `Poe Previews Hackathon`
> - `Agihouse Hackathon`
> - `Poe Platform Announcement` (Poe 平台公告)
> - `In-Chat Generative UI Experiences` (聊天内生成式 UI 体验)
> - `Discord Channel` (Discord 频道)


- **Poe Previews Hackathon：庆祝扩大发布**：Poe (@poe_platform) 宣布与 Agihouse (@agihouse_org) 合作举办 "Previews Hackathon"，以庆祝其扩大发布。
   - 该黑客松邀请所有创作者构建最具创新性和实用性的聊天内生成式 UI 体验，详细信息请访问 [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817)。
- **Discord 频道关于黑客松的讨论**：#events Discord 频道的一位用户分享了 X 上 Poe Previews Hackathon 公告的链接，并确认他们正在协助该活动。
- **黑客松目标：聊天内生成式 UI 体验**：该黑客松旨在创建创新且实用的“聊天内生成式 UI 体验”，鼓励创作者展示他们的技能。
   - 该公告强调了在生成式 AI 背景下用户体验的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1823382125523181683">来自 Poe (@poe_platform) 的推文</a>: 为了庆祝扩大发布，我们正与 @agihouse_org 合作举办 Previews 黑客松，你将在这里竞争创建最具创新性和实用性的聊天内生成式 UI 体验。所有创...</li><li><a href="https://app.agihouse.org/events/poe-previews-hackathon-20240817">AGI House</a>: 未找到描述</li><li><a href="https://app.agihouse.org/events/poe-previews-">AGI House</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1272961929115340921)** (4 条消息): 

> - `Virtual Try On` (虚拟试穿)
> - `Image Feature Extraction` (图像特征提取)
> - `Model Size` (模型大小)


- **虚拟试穿 (Virtual Try On) 实现**：一位成员分享了他们为研发团队构建虚拟试穿功能的经验，并指出通过存储提取的特征可以有效加快训练运行速度。
   - 该功能利用在线预处理，并将提取的特征存储在文档存储表中，以便在训练期间进行高效检索。
- **图像特征提取技术**：一位成员询问了为虚拟试穿功能从图像中提取的具体特征。
   - 提供功能细节的成员强调了他们方法的通用性，可以适配从极小到极大规模的模型。
- **模型大小对虚拟试穿的影响**：该成员强调了他们的虚拟试穿功能在各种模型大小上的成功应用。
   - 这证明了该方法在处理不同计算需求和模型复杂度方面的灵活性。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1272667816859930644)** (5 条消息): 

> - `Llama 3.1 8b structured output` (Llama 3.1 8b 结构化输出)
> - `RAG on technical documents with images` (针对带图像的技术文档的 RAG)
> - `Next.js and FastAPI interaction` (Next.js 与 FastAPI 的交互)
> - `AWS pip install issues` (AWS pip install 问题) 


- **Llama 3.1 8b 通过工具支持结构化输出**：一位用户确认 **Llama 3.1 8b** 可以通过工具调用 (tool use) 产生结构化输出，并已直接使用 **llama.cpp** 进行了测试。
   -  
- **为 RAG 从技术图像中提取信息**：一位用户寻求关于如何从电路图、地图和电压曲线等图像中提取信息，以便对技术文档进行 **RAG** 的建议。
   - 他们提到在使用传统方法时遇到困难，强调需要捕捉那些不在文本中但在视觉上可被专家解读的信息。
- **Next.js 向 FastAPI 发送 POST 请求返回 405 Method Not Allowed**：一位用户在从运行在 **EC2** 上的 **Next.js Web 应用**向同一 **EC2** 实例上的 **FastAPI 端点**发送 **POST 请求**时遇到了 **405 Method Not Allowed** 错误。
   - 他们观察到，尽管在 **Next.js 代码**中显式使用了 **POST 方法**，请求仍被错误地解释为 **GET 请求**。
- **因环境模拟导致的 AWS pip install 问题已解决**：一位用户通过专门为 **Unix 环境**安装包，解决了 **AWS 系统**上的 **pip install** 问题。
   - 该问题源于虚拟环境在 **pip install** 过程中错误地模拟了 **Windows**，从而导致了故障。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1272903058749390869)** (1 messages): 

> - `Profundo`
> - `Profundo use cases`
> - `Profundo AI`
> - `Profundo product hunt`
> - `Profundo benefits` 


- **Profundo 发布，助力自动化研究**：Profundo 自动化了数据收集、分析和报告流程，让每个人都能对自己感兴趣的主题进行深度研究。
   - 它能最大限度地减少错误并提高生产力，让用户专注于做出明智的决策。
- **Profundo 的 AI 驱动高效数据处理**：Profundo 利用尖端的 AI 技术帮助你更高效地收集、分析和报告数据。
   - 告别手动数据收集，迎接自动化洞察。
- **Profundo 赋能多样化使用场景**：Profundo 正被用于自学、内容创作、初稿撰写、个人项目和职业发展。
   - 在学术界，它被用于研究和文献综述。
- **Profundo 在 ProductHunt 寻求点赞**：Profundo 今天在 ProductHunt 上线，他们正寻求点赞以触达更多人群。
   - 如果你使用过 Profundo 并觉得它很有用，他们鼓励你在 ProductHunt 上为他们投票。



**相关链接**: <a href="http://profundo.app/>">Profundo | Research Redefined</a>：Profundo 是一个研究平台，让你能以比以往更高效、更有效的方式进行研究。

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1272954945963819058)** (1 messages): 

> - `AI Agents in Enterprises`
> - `Monitoring and Governance of AI Agents` 


- **企业中的 AI Agent 治理**：一位用户询问了在大型组织中监控和治理 AI Agent 所面临的挑战。
- **开放讨论邀请**：该用户邀请任何在企业内部从事 AI Agent 相关工作的人分享他们的经验。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1272661518428799017)** (2 messages): 

> - `Screenless personal tutor for kids` 


- **针对儿童的无屏幕导师构想**：一名成员表示有兴趣使用 01 为儿童构建一个无屏幕的个人导师。
   - 他们征求反馈，并询问是否有人对合作开发此项目感兴趣。
- **针对儿童的无屏幕导师构想**：一名成员表示有兴趣使用 01 为儿童构建一个无屏幕的个人导师。
   - 他们征求反馈，并询问是否有人对合作开发此项目感兴趣。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1272997098291462164)** (3 messages): 

> - `Open Interpreter in Obsidian`
> - `Convert Anything Tool` 


- **Open Interpreter 实现万物互转**：使用 Open Interpreter 将任何类型的数据转换为任何其他格式。
   - 通过使用利用 Open Interpreter 强大功能的 "Convert Anything" 工具，这是可以实现的。
- **Obsidian 中的 Open Interpreter**：一个新的 YouTube 系列即将发布，将演示如何在 Obsidian 笔记应用中使用 Open Interpreter。
   - 该插件允许你使用 Open Interpreter 控制你的 Obsidian 库（vault），这可能对人们处理知识的方式产生重大影响。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HjcPRoPfri0">Open Interpreter Obsidian &amp; Convert Anything - Ep 0</a>: Tool Use 第 0 集！Open Interpreter Obsidian 插件 - 使用 Open Interpreter 控制你的 Obsidian 库！CV - 利用 Open Interpreter 的力量将任何东西转换为任何东西...</li><li><a href="https://www.youtube.com/watch?v=xaroJxFTVFQ">AI 的左倾偏见是真的吗？</a>: 在 Brilliant 上学习大语言模型课程！前 30 天免费，使用我们的链接订阅年度高级会员可享受 20% 折扣 ➜  https://brill...
</li>
</ul>

</div>
  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1272760688007319655)** (1 messages): 

> - `SlimOrca without deduplication`
> - `Fine-tuning (FT) with deduplication` 


- **无去重的 SlimOrca**：一位用户询问是否存在一个**移除了软提示（soft prompting）**且**没有去重（deduplication）**的 **SlimOrca** 版本，最好包含代码。
   - 他们还询问是否有人实验过在有或没有去重、以及有或没有软提示的数据上进行微调（FT）。
- **在去重数据上进行微调**：用户询问是否有人实验过在**去重数据**与**非去重数据**上进行微调的效果对比。
- **带有软提示的微调**：用户询问了使用**软提示（soft prompting）**与不使用软提示进行微调（FT）的效果。


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1273007276168319067)** (1 条消息): 

> - `用于 Jupyter Notebook 自动化的 Agentic System` 


- **构建 Agentic Jupyter Notebook 自动化系统**：一位成员表示有兴趣构建一个用于自动化 Jupyter Notebook 的 Agentic System，旨在创建一个以现有 Notebook 为输入、修改单元格并生成多个变体的 Pipeline。
   - 他们寻求有关库、Cookbooks 或开源项目的建议，以便为该项目提供起点，并从 Devin 等类似工具中汲取灵感。
- **期望功能：自动化的 Notebook 修改与验证**：预想的系统应能够智能地替换 Jupyter Notebook 中的特定单元格，并根据这些修改生成不同的 Notebook 版本。
   - 至关重要的是，该系统应具备 Agentic 特性，使其能够验证其输出并迭代优化修改，直到达到预期结果。


  

---



---



---



{% else %}


> 完整的频道细分内容已为邮件版缩减。 
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}