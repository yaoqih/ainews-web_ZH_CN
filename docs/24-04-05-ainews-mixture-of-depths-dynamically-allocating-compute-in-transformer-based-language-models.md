---
companies:
- deepmind
date: '2024-04-05T22:44:29.761385Z'
description: '**DeepMind** 推出了 **Mixture-of-Depths (MoD)** 技术，通过在 Transformer 层之间动态分配
  FLOPs（浮点运算量）来优化计算资源利用，在不影响训练效果的前提下，使前向传播（forward passes）速度提升了 **50% 以上**。


  MoD 采用 top-k 路由选择性地处理 token，提高了效率，并可能实现更快的超长上下文处理能力。该方法还可以与混合专家模型 (MoE) 结合，实现查询（queries）、键（keys）和值（values）的解耦路由。


  Reddit 上的讨论重点关注了以下内容：

  *   对 **LLM 炒作**掩盖其他 AI 技术的担忧；

  *   Transformer 效率的提升；

  *   一种新型的“思考并执行”（Think-and-Execute）框架，可将算法推理能力提高 **10-20%**；

  *   视觉自回归建模 (VAR) 在图像质量和生成速度上超越了扩散模型。


  此外，端侧模型 Octopus v2 在函数调用（function calling）的准确率和延迟方面表现优于 GPT-4。'
id: b3ded501-bfcb-415e-9f51-e69f84ca18dd
models:
- octopus-v2
original_slug: ainews-mixture-of-depths-dynamically-allocating
people:
- piotrpadlewski
title: '**深度混合：基于 Transformer 的语言模型中的动态计算分配**'
topics:
- transformer-efficiency
- dynamic-compute-allocation
- mixture-of-experts
- mixture-of-depths
- top-k-routing
- algorithmic-reasoning
- visual-autoregressive-modeling
- on-device-models
- function-calling
- scaling-laws
---

 

---

**目录**

[TOC] 


---

# AI Reddit Recap

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能尚未实现，但即将推出。

**AI 研究与开发**

- **对 LLM 炒作的担忧**：在 /r/MachineLearning 中，一篇文章指出 [**LLM 炒作正将注意力和投资从其他具有潜在影响力的 AI 技术中转移**](https://www.reddit.com/r/MachineLearning/comments/1bvi4au/d_llms_are_harming_ai_research/)。作者声称，自 GPT-4 以来，LLM 的性能和设计几乎没有进展，主要方法只是将模型做大，并对大量缺乏 ML 知识却自称“AI 研究员”的人涌入表示担忧。
- **提高 Transformer 效率**：Deepmind 推出了一种方法，让 [**Transformer 能够动态地为序列中的特定位置分配计算资源**](https://www.reddit.com/r/MachineLearning/comments/1bvrduw/r_deepmind_mixtureofdepths_dynamically_allocating/)，从而优化跨层的分配。在相同的 FLOPs 和训练时间下，该模型达到了基准性能，但每次前向传递所需的 FLOPs 更少，且在采样期间速度可提升 50% 以上。
- **增强算法推理**：一个名为 Think-and-Execute 的新框架 [**将 LM 推理分解为：发现表达为伪代码的任务级逻辑，然后将其量身定制到每个实例并模拟执行**](https://www.reddit.com/r/MachineLearning/comments/1bvvqbl/r_language_models_as_compilers_simulating/)。这比 CoT 和 PoT 基准在算法推理方面提高了 10-20 个百分点。
- **视觉自回归建模 (VAR)**：VAR 将图像上的自回归学习重新定义为从粗到细的“下一尺度预测 (next-scale prediction)”，使 AR Transformer 能够 [**快速学习视觉分布，在图像质量和速度上超越 Diffusion，并表现出类似于 LLM 的 Scaling Laws 和零样本泛化能力**](https://www.reddit.com/r/MachineLearning/comments/1bvh8ep/d_visual_autoregressive_modeling_scalable_image/)。
- **端侧模型**：Octopus v2 是一个拥有 2B 参数的端侧模型，[**在函数调用 (function calling) 的准确性和延迟方面超越了 GPT-4，相比使用 RAG 的 LLaMA-7B，延迟降低了 35 倍**](https://www.reddit.com/r/MachineLearning/comments/1bvv0by/r_octopus_v2_ondevice_language_model_for_super/)。它适用于在生产环境中的边缘设备上部署。

**AI 产品与服务**

- **YouTube 对 Sora 的立场**：[YouTube 表示 OpenAI 使用其视频训练 Sora 将违反规则](https://www.bloomberg.com/news/articles/2024-04-04/youtube-says-openai-training-sora-with-its-videos-would-break-the-rules)，这引发了关于 AI 训练数据使用的疑问。
- **Claude 的工具使用**：Anthropic 的 Claude 模型 [现在具备了使用工具的能力](https://i.redd.it/6v76x14hvksc1.jpeg)，扩展了其潜在的应用场景。
- **Cohere 的大模型**：Cohere 发布了 [Command R+，这是一个可扩展的 104B 参数 LLM，专注于企业级用例](https://www.reddit.com/r/LocalLLaMA/comments/1bvniaz/command_r_cohere_for_ai_104b/)。
- **Google 的 AI 搜索变现**：有推测称 [Google 的 AI 驱动搜索极有可能被置于付费墙之后](https://aiplusyou.ai/googles-ai-powered-search-will-most-likely-be-behind-a-paywall-so-perplexity/)，这引发了关于 AI 增强服务可访问性的讨论。

**AI 硬件与性能**

- **Apple MLX 性能**：[Apple 的 MLX 在 M2 Ultra 上运行 4-bit Mistral 7B 达到 100 tokens/second](https://i.redd.it/3o7r9xzmtfsc1.jpeg)，展示了强大的端侧推理能力。
- **消费级设备上的 QLoRA**：[QLoRA 实现了在 M2 Ultra 上运行 Cohere 的 104B Command R+ 模型](https://www.reddit.com/r/LocalLLaMA/comments/1bwbgum/qlora_of_command_r_104b_on_an_m2_ultra_mlx/)，在专业消费级设备上实现了约 25 tokens/sec 的生成速度（注：原文此处可能指吞吐量或特定阶段速度，后文提到约 7.5 tokens/sec）。
- **AMD 的开源举措**：[AMD 正在将其 ROCm GPU 计算平台开源，包括软件栈和硬件文档](https://wccftech.com/amd-rocm-going-open-source-will-include-software-stack-hardware-documentation/)。这可能会加速 AI 硬件和软件的开发与普及。

# AI Twitter Recap

> 所有回顾均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在利用 Haiku 进行聚类和流程工程 (flow engineering)。

**AI 模型与架构**

- **Google 的 Training LLMs over Neurally Compressed Text**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1776055420848631814) 指出，Google 在神经压缩文本上训练 LLM 的方法**性能大幅优于 byte-level baselines**，虽然其 PPL 逊于 subword tokenizers，但受益于**更短的序列长度**。
- **阿里巴巴的 Qwen1.5 模型**：[@huybery](https://twitter.com/huybery/status/1776255803282088056) 发布了 Qwen1.5-32B 稠密模型，该模型引入了 GQA，展现出**与 72B 模型相当的竞争力**，并在语言理解、多语言支持、代码和数学能力方面表现出色。[@_philschmid](https://twitter.com/_philschmid/status/1776257496547561805) 补充道，Qwen1.5 32B 是一款**具有 32k 上下文的多语言稠密 LLM**，使用 DPO 进行偏好训练，拥有自定义许可证，可商用，并已在 Hugging Face 上线，其 **MMLU 得分为 74.30**，在开源 LLM 排行榜上得分为 70.47。
- **ReFT: Representation Finetuning for Language Models**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1776057023697731913) 分享了 ReFT 论文，该论文提出了一种比之前的 state-of-the-art 参数高效微调方法在**参数效率上高出 10x-50x 的微调方法**。
- **Apple 的 MM1 多模态 LLM 预训练**：[@_philschmid](https://twitter.com/_philschmid/status/1776240697219195221) 总结了 Apple 的 MM1 论文，该论文研究了架构组件和数据选择对视觉语言模型 (VLMs) 的影响。影响性能的关键因素包括**图像分辨率、模型大小和训练数据组成**，其中 Mixture-of-Experts (MoE) 变体的性能优于稠密变体。

**技术与框架**

- **LangChain Weaviate 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1776301091375948244) 宣布推出 `langchain-weaviate` 包，提供对 Weaviate 开源向量数据库的访问，具有**原生多租户和高级过滤**等功能。
- **Claude Function Calling Agent**：[@llama_index](https://twitter.com/llama_index/status/1776051869850476840) 发布了一个由 LlamaIndex 抽象驱动的 Claude Function Calling Agent，利用 Anthropic 在其 messages API 中提供的工具使用支持，实现**高级 QA/RAG、工作流自动化等**。
- **AutoRAG**：[@llama_index](https://twitter.com/llama_index/status/1776289203459858849) 介绍了由 Marker-Inc-Korea 开发的 AutoRAG，它可以根据评估数据集**自动查找并优化 RAG 流水线**，让用户专注于声明 RAG 模块而非手动调优。
- **LLMs as Compilers**：[@omarsar0](https://twitter.com/omarsar0/status/1776248188707430719) 分享了一篇论文，提出了一种 think-and-execute 框架来分解 LLM 中的推理，用伪代码表达任务级逻辑，并通过 LM 模拟执行以**提高算法推理性能**。
- **Visualization-of-Thought Prompting**：[@omarsar0](https://twitter.com/omarsar0/status/1776082343813403063) 讨论了一篇关于 Visualization-of-Thought (VoT) 提示词的论文，该技术使 LLM 能够“可视化”推理轨迹并创建心理图像以**引导空间推理**，在多跳空间推理任务上表现优于多模态 LLM。

**数据集**

- **Gretel 的合成 Text-to-SQL 数据集**：[@_philschmid](https://twitter.com/_philschmid/status/1776154264944931014) 分享了 Gretel 的高质量合成 Text-to-SQL 数据集 (gretelai/synthetic_text_to_sql)，包含 **105,851 个样本，约 23M tokens，覆盖 100 个领域/垂直行业**，以及广泛的 SQL 复杂度级别，以 Apache 2.0 许可证发布。

**计算基础设施**

- **配备 NVIDIA L4 GPU 的 AWS EC2 G6 实例**：[@_philschmid](https://twitter.com/_philschmid/status/1776172921230123178) 报道了配备 NVIDIA L4 GPU (24GB) 的新 AWS EC2 G6 实例，每个实例支持多达 8 个 GPU (192GB)，**比配备 A10G GPU 的 G5 实例便宜 25%**。
- **Google Colab L4 GPU 实例**：[@danielhanchen](https://twitter.com/danielhanchen/status/1776074818758348918) 指出，Google Colab 现在提供 L4 GPU 实例，价格为 **$0.482/小时，具有原生 fp8 支持和 24GB 显存**，同时 A100 和 T4 实例也降价了。

**讨论与观点**

- **语言模型的商品化**：[@bindureddy](https://twitter.com/bindureddy/status/1776250078766010521) 建议，鉴于 Google 拥有强大的收入流且在搜索领域面临 LLM 的威胁，应该开源 Gemini 1.5 和 2.0。随着加入开源 AI 革命的公司越来越多，目前仅剩 Google、OpenAI 和 Anthropic 保持闭源。
- **基准测试引发的担忧**：[@soumithchintala](https://twitter.com/soumithchintala/status/1776311683385880983) 对 Google 的 Jeff Dean 和 François Chollet 发布的基准测试提出了质疑，理由包括计时代码错误、对比了不同的精度（precisions），并指出 Google 团队在发布前应与 PyTorch 团队合作，以避免在社区中造成分歧。
- **AI 损害研究**：[@bindureddy](https://twitter.com/bindureddy/status/1776095827536273699) 认为 LLM 在某种程度上损害了 AI 研究，因为它将注意力从表格数据（tabular data）和全新的创新上转移了，并预测到今年年底 LLM 将会出现过剩。
- **将 AI 产品定位为“虚拟员工”**：[@dzhng](https://twitter.com/dzhng/status/1776309448858124760) 批评了将 AI 产品定位为“虚拟员工”的做法，认为这设定了不切实际的期望并限制了 AI 的颠覆性潜力。他建议关注特定的“工作范围（scopes of work）”，并设想未来由协调的 AI Agent 运行的“神经公司（neural corporations）”。

**迷因与幽默**

- **Google 的 Transformer 2**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1776247669905932566) 分享了关于 Google Transformer 2 的细节，该模型**将 attention、recurrence、retrieval 和 FFN 统一到一个模块中**，性能与 Transformer 持平，但**计算效率提升了 20 倍**，且能高效处理 **100M 上下文长度**。这是一个迟到的愚人节玩笑。
- [@cto_junior](https://twitter.com/cto_junior/status/1776163897235353615) 调侃了他们超快的 RAG 应用，使用的是 “numpy 暴力相似度搜索（bruteforce similarity search）” 而非昂贵的企业级解决方案。
- [@vikhyatk](https://twitter.com/vikhyatk/status/1776073962336014453) 调侃说正在开发一个“使用 jax, rust, go, triton, dpo 和 rag 训练的 mamba mixture of experts diffusion qlora 1.58bit 模型”。
- [@cto_junior](https://twitter.com/cto_junior/status/1776238063158546571) 幽默地感叹 AWS 策略的复杂性，不得不从 Hackernoon 复制粘贴并祈祷能解决 500 错误。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**1. 前沿 LLM 进展与发布**

- **[Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)** 发布了 **[Command R+](https://txt.cohere.com/command-r-plus-microsoft-azure/)**，这是一个拥有 **104B 参数的 LLM**，具备 **Retrieval Augmented Generation (RAG)**、**多语言支持**和**企业级能力**。其性能令许多人印象深刻，在**中古高地德语翻译**等任务上甚至超越了 **GPT-4**。

- **[Anthropic](https://twitter.com/AnthropicAI/status/1775979799644934281)** 展示了 **Claude** 中的**实时工具调用 (live tool use)**，引发了对其**操作复杂性**的分析。初步测试发现 Claude 表现 **"相当不错"**，但面临 **延迟挑战**。

- **[QuaRot](https://arxiv.org/abs/2404.00456)** 是一种新的 **4-bit 量化方案**，可以对 LLM 进行**端到端**量化，且**性能损失极小**。量化后的 **LLaMa2-70B** 保留了 **99%** 的零样本 (zero-shot) 能力。

- **[JetMoE-8B](https://huggingface.co/jetmoe/jetmoe-8b)** 是 LLaMA2-7B 等大型模型的**高性价比替代方案**，声称在仅 **$0.1M** 的训练成本下即可达到同等性能，且**对学术界友好**。

**2. 参数高效的 LLM 微调技术**

- **[ReFT](https://github.com/stanfordnlp/pyreft)** (**Representation Finetuning**) 是一种新方法，据称比之前的技术**参数效率高出 10-50 倍**，允许以**极少的参数更新**进行 LLM 适配。

- 关于 **LoRA, QLoRA, LoReFT** 以及其他**高效微调方法**的讨论，例如 Facebook 新推出的 **"schedule-free" 优化器**，它消除了对学习率调度 (learning rate schedules) 的需求。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** 探索了将 **LoReFT** 和最新的 **PEFT v0.10.0**（支持 **量化 DoRA**）等技术进行集成。

**3. 高效 Transformer 的架构创新**

- **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** 通过 **top-k 路由机制**在 Transformer 中实现 **动态 FLOPs 分配**，有望通过用更少的计算量处理简单的 token 来显著**降低计算量**。

- 讨论了将 **Mixture-of-Experts (MoE)** 与 **Mixture-of-Depths** 相结合的可能性，以及在周末将这些方法集成到现有模型中的潜力。

- **[BitMat](https://github.com/astramind-ai/BitMat)** 展示了 **"1-bit LLM 时代"** 方法的高效实现，而 **LASP 库** 为**长上下文处理**带来了**改进的 AMD 支持**。

**4. 开源 AI 框架与社区努力**

- **[LM Studio](https://lmstudio.ai)** 在 HuggingFace 上获得了**[新的社区页面](https://huggingface.co/lmstudio-community)**，用于分享 **GGUF 量化版本**，填补了一位高产贡献者离开后的空白。

- **[LlamaIndex](https://github.com/run-llama/llama_index)** 引入了 **Adaptive RAG, AutoRAG** 和 **Claude Function Calling Agent**，用于高级多文档处理。

- **[Basalt](https://github.com/basalt-org/basalt)** 作为一个**纯 Mojo 编写的新机器学习框架**出现，旨在提供一个可与 PyTorch 媲美的**深度学习 (Deep Learning)** 解决方案。

- **[Unsloth AI](https://github.com/unslothai/unsloth)** 探索了 **GaLore** 等 **GPU 显存优化**技术，并促进了关于**微调研讨会**和确保可复现性的**严格版本控制**的讨论。

---

# 第一部分：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 疑难解答**：工程师们正在询问 **Perplexity Pro** 的功能和可访问性，解决如何启用频道、文件删除问题以及购买障碍；建议联系支持人员或管理员寻求帮助。

- **AI 的云端难题**：关于云服务在大型语言模型 (LLM) 开发中作用的热议，涉及 **AWS** 与 **Azure** 市场份额的辩论，以及关于 **Perplexity** 与 **Anthropic** 潜在合作的推测性聊天。

- **苹果 AI 雄心分析**：公会正在分析 **Apple 3b 模型** 的利基应用，并思考 **Apple Glass** 的主流潜力，将其与 Google 的 VR 计划进行对比。

- **API 定价与限制详解**：澄清了关于 **Perplexity API** 的查询，例如使用 **Google Pay** 购买额外额度以及 **sonar-medium-chat** 的成本（每 1M tokens $0.60），并指出了 [速率限制](https://docs.perplexity.ai/docs/rate-limits) 和 [定价文档](https://docs.perplexity.ai/docs/pricing)。

- **Perplexity 社区好奇心**：成员们正积极使用 [Perplexity AI 搜索](https://www.perplexity.ai) 探索各种话题，从美容、独裁统治到 **Cohere 的 Command R**；他们还在分享内容，并提醒如何将帖子设置为可分享。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**最大化图像保真度**：关于规避生成 **2k 分辨率写实图像** 问题的技术建议强调了先生成低分辨率图像再进行放大（upscaling）、减少步数以及启用 "hiresfix"。对话中还探讨了放大过程中的质量与畸变之间的权衡。

**SD3 发布让众人焦急等待**：虽然一些公会成员正热切期待 **Stable Diffusion 3 (SD3)**，但另一些人察觉到了延迟，这导致了从期待到怀疑的复杂情绪，并将其与 **Ideogram** 和 **DALLE 3** 等其他模型进行了比较。

**AI 与艺术的碰撞**：围绕将 AI 用于艺术创作展开了创意讨论，重点介绍了图像生成中的 **Daz AI**，以及微调模型以获得特定艺术输出的复杂性，例如在 Stable Diffusion 中生成服装设计。

**VRAM 前来救场**：技术讨论深入探讨了模型资源需求，特别是在不同 **VRAM** 分配下运行模型的情况，以及对 SD3 在标准 **消费级 GPU** 上性能表现的期待。

**揭秘 Stable Diffusion 诀窍**：用户分享了见解并就优化 Stable Diffusion 模型版本和界面寻求建议，涵盖了图像 finetuning 的最佳实践和有效的模型 checkpoint 管理。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Fine-Tuning API 焕然一新**：OpenAI 推出了 Fine-Tuning API 的更新，旨在让开发者对模型定制拥有更多控制权。增强功能包括新的 **dashboards** 和 **metrics**，并扩展了自定义模型计划，详情见 [OpenAI 博客文章](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program) 及配套的 [YouTube 教程](https://www.youtube.com/watch?v=ahnGLM-RC1Y&list=PLOXw6I10VTv-exVCRuRjbT6bqkfO74rWz&index=4)。

**AI 讨论升温**：各频道围绕 AI 认知和 ASCII art 生成等概念展开辩论，探讨了 AI 在 3D 打印中的潜力，并在发布兴奋感与安全措施之间寻找平衡。此外，还重点讨论了使用 AI 进行文档分析和为数据增强进行 fine-tuning 的实现问题，以及将 assistant 的 temperature 设置为 0.0 时观察到的不一致行为。

**Prompt Engineering 策略揭晓**：成员们正在分享让 GPT-3 生成更长输出以及将回复限制在特定文档中的策略。技巧从以 "continue" 开始新对话，到要求 AI 确认所提供材料中是否存在答案的严厉指令不等。

**断言式提示（Assertive Prompting）或可提升 GPT 准确性**：为了确保 GPT 的输出严格基于所提供的内容，建议给出清晰且断言式的提示。无论是讨论意识的本质以模仿人类反应，还是强化基于特定文档的回复，社区都在探索 AI 理解能力的表象。

**GPT-4 使用成本明确化**：讨论明确了将 GPT 模型集成到应用中需要订阅计划（如 Plus 计划），因为目前所有模型都在 GPT-4 下运行。寻求增强 GPT 模型功能的用户在开发 AI 驱动的应用时必须考虑这一点。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 保持离线状态**：确认 [LM Studio](https://lmstudio.ai) 缺乏类似于其他 AI 工具的网络搜索功能，它提供的是本地部署选项，这在其讨论和 [LM Studio's documentation](https://lmstudio.ai/docs/) 中有所概述。

- **模型在排行榜上展开竞争**：社区正在 [LMsys Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 等排行榜上评估模型，强调只有少数排名靠前的模型支持本地运行，这是该受众群体关注的关键因素。

- **大模型、大 GPU、大问题**：成员们讨论了 LM Studio 中多 GPU 配置与 **Mixtral 8x7b** 和 **Command-R Plus** 等模型规模之间的性能权衡，深入探讨了 token 速度和硬件特定限制，包括混合不同代际和品牌（主要是 NVIDIA）时的问题。

- **Eurus 的崛起**：社区讨论了 **Eurus-7b** 模型的进展，该模型提供了改进的推理能力。它部分基于 **UltraInteract** 数据集进行训练，并已在 HuggingFace 上发布，这表明该群体在不断寻找更优的模型。

- **归档与社区支持**：LM Studio 宣布了一个新的 Hugging Face 社区页面 [lmstudio-community](https://huggingface.co/lmstudio-community)，用于分享 GGUF 量化模型，填补了一位著名社区贡献者留下的空白。
  
- **跨界面的可靠性**：用户比较了 LM Studio beta 功能（如 text embeddings）与其他本地 LLM 用户界面的可靠性，并讨论了问题的解决方法，包括加载限制以及 ROCm 在新型 Intel 处理器上的潜力，这些内容分享在 [Reddit](https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/) 和 [Radeon's Tweet](https://twitter.com/amdradeon/status/1775261152987271614) 等社交平台上。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**LoRA 的大胆新飞跃**：有人提议将 **Low-Rank Adaptation (LoRA)** 应用于 **Mistral 7B**，旨在增强其能力。目前正在计划整合一种**分类法驱动的方法 (taxonomy-driven approach)** 来进行句子分类。

**最先进的归档与网页爬取实践**：讨论强调了“归档小组”与“数据囤积”之间的微妙界限，并对排除 Twitter 的 **Common Crawl** 网页爬取表示认可。此外，还提到了 **Aurora-M** 的推广，这是一个拥有 15.5B 参数、超过 2 万亿训练 tokens 的开源多语言 LLM，以及用于结构化 LLM 输出的工具如 *Instructor*。

**LLM 领域扩展**：公告包括一个 **104B LLM, C4AI Command R+**，具备 RAG 功能并支持多种语言，可在 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 上获取。社区还讨论了 **GPT-4 微调定价**，并欢迎 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai) 预告的 AI 开发更新，同时强调了 **LLaMA-2-7B 模型** 的 700K token 上下文长度训练，以及 Nvidia 4090 GPU 上 fp8 可用性的不确定性。

**数据集与工具稳步推进**：讨论了 [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) 的引入，该工具可将书籍和计算转化为指令微调数据集。*Severian/Internal-Knowledge-Map* 及其在 **LM 理解** 方面的创新方法引起了关注，**neurallambda** 项目旨在通过 lambda calculus 实现 AI 的推理能力。

**动态函数调用**：一个**使用 Hermes 进行函数调用**的示例将在仓库中展示，同时正在针对其在 **Vercel AI SDK RSC** 上的运行进行严肃的调试工作。**Hermes-Function-Calling** 仓库面临批评，随后转而遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)。**Eurus-7B-KTO 模型** 进行了预览，因其在 **SOLAR 框架** 中的应用而受到关注。

**依赖困境与数据集策略**：确认了一个新出现的**依赖问题**，但未提供更多上下文。**RAG 数据集频道** 阐明了置顶摘要、探索 **adaptive RAG** 技术以及利用多种数据源进行 RAG 的计划，并讨论了来自 *Command R+* 和 *Claude Opus* 的界面更新。

**借助 WorldSim 推进世界构建**：关于 [WorldSim Versions & Command Sets](https://pastebin.com/raw/P6UQWUaB) 和 [Command Index](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4) 的 token 正在流通，涵盖了自定义表情建议等用户体验细节。此外，还酝酿了关于 AI 交叉**哲学新频道**的想法，以及反映 Zipf's law 的 **TRS-80 远程呈现**体验。人们对具有增强 UX 的 **WorldSim 更新** 充满期待，希望能解决自我引导（self-steering）问题。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GPU 显存收益**：[GaLore 更新](https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362) 承诺通过融合算子（fused kernels）增强 GPU 显存效率，引发了关于将其与 Unsloth AI 集成以获得卓越性能的讨论。

**模型 Packing 不匹配**：建议在 Gemma 模型上谨慎使用 `packing` 参数，因为存在兼容性问题，尽管它可以通过连接 tokenized 序列来加速训练。

**优化机会**：尽管 GaLore 的默认性能落后于 LoRA，但目前正在探索将 Unsloth 与 GaLore 结合进行显存和速度优化的可能性。

**期待 Unsloth 的新功能**：Unsloth AI 计划在 4 月 22 日前发布一项 "GPU poor" 功能，并在 5 月初发布 "Automatic optimizer"。自 2023 年 11 月起可用的 Unsloth Pro 正在接受分发改进方面的审查。

**合成生成中的数据集多样性**：格式灵活性被认为对合成数据集生成的性能影响微乎其微，允许在微调 LLM 时根据个人喜好选择格式。

**热切期待 Kaggle 重置**：Kaggle 爱好者们正在等待新赛季，利用夏令时调整带来的额外睡眠时间，同时寻找 AI 新闻源并讨论可能包含 libgen 或 scihub 的预训练数据集。

**Unsloth 实现流式推理**：社区反馈赞扬了 Unsloth 在推理过程中的易用性，并分享了如 [batch inference guidelines](https://github.com/unslothai/unsloth/issues/267#issuecomment-2034047189) 等额外资源。

**微调研讨会探讨**：用户们正在集思广益，探讨如何开展具有实践经验的有效微调研讨会，包括提前准备模型或使用 [LoRaX](https://github.com/oobabooga/text-generation-webui) 作为模型交互的 Web UI 等创新方案。

**稳定性的版本控制**：对 Unsloth 更新影响模型可复现性的担忧促使大家就严格版本控制的必要性达成共识，以确保数值一致性和可逆性。

**微调中的参数效率**：一种名为 ReFT 的新型微调技术因其极高的参数效率而受到关注，其详细信息可见 [GitHub repo](https://github.com/stanfordnlp/pyreft) 和[配套论文](https://arxiv.org/abs/2404.03592)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Wiki 知识库现已公开访问**：成员们解决了访问 **Wikitext-2** 和 **Wikitext-103 数据集**的挑战，分享了来自 [Stephen Merity 的页面](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR) 和 [Hugging Face](https://huggingface.co/datasets/wikitext/tree/main/wikitext-103-raw-v1) 的链接，并对原始数据格式的易用性表示关注。

**GateLoop 复现引发辩论**：对 **GateLoop** 架构困惑度（perplexity）分数的怀疑遇到了带有 [已发布代码](https://github.com/tobiaskatsch/GatedLinearRNN) 的澄清信息，引发了关于实验复现和各种 Attention 机制性能的讨论。

**模块化 LLM 处于前沿**：深入讨论集中在 **Mixture of Experts (MoE)** 架构上，涵盖了 **可解释性**、**分层 vs 扁平结构**以及 **LLM** 中的效率策略，引用了多篇论文和一份 [硕士论文预告](https://x.com/aaron_defazio/status/1776320004465582331?s=46)，暗示 MoE **FLOPs** 即将取得突破。

**可解释性实现交流**：关于 **AtP*** 开源实现可用性的查询促成了 [AtP* 的 GitHub repo](https://github.com/koayon/atp_star) 的分享，同时 **David Bau** 在 [GitHub 为 nnsight](https://github.com/ndif-team/nnsight) 寻求社区支持，以满足 **NSF** 评审员的要求。

**从故障排除到 Thunderdome 测试**：`#lm-thunderdome` 频道中的讨论深入探讨了故障排除，从 `top_p=1` 的语法怪癖到对模型参数兼容性的困惑，以及 `batch_size=auto` 带来的效率提升，建议针对某些问题进行全新安装或使用 [Google Colab](https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8u)。

**Gemini 获得云端支持**：一条简短的消息强调了 **AWS** 对 **Gemini** 的支持实现，并提到了 **Azure** 的支持。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**提升 Mojo 的调试能力**：工程师们询问了关于 **neovim** 等编辑器的调试支持，并结合 Language Server Protocol (LSP) 以增强问题解决能力。

**关于 Variant 类型的动态讨论**：在 **Mojo** 中，`Variant` 类型的使用比 `isinstance` 函数更受推崇，突出了其动态数据存储能力以及使用 `isa` 和 `get/take` 方法进行类型检查的特性，详见 [Mojo 文档](https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo)。

**Basalt 照亮机器学习框架领域**：新推出的机器学习框架 **Basalt** 正在成为头条新闻，它被定位为“深度学习”框架且可与 **PyTorch** 媲美，其基础版本 [v.0.1.0 已在 GitHub 发布](https://github.com/basalt-org/basalt)，并附有相关的 [Medium 文章](https://medium.com/@bennynottonson/basalt-machine-learning-in-pure-mojo-66cbd6e53a05)。

**计算字节，而不只是桶的数量**：关于值存储的桶大小讨论指出，每个桶持有 **UInt32 values**，每个仅占 4 字节。这种对内存效率的关注对于处理高达 **2^32 - 1 个值**至关重要。

**与 Python 互操作性的演进**：揭示了 Python 与 Mojo 接口的进展，重点在于 PyMethodDef 和 PyCFunction_New 的使用，具有稳定的引用计数且至今未发现问题。目前的开发进展可以在 [rd4com 的 GitHub 分支](https://github.com/rd4com/mojo_branch/tree/nightly)查看。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LASP 库受到关注**：[Linear Attention Sequence Parallelism (LASP) 库](https://github.com/OpenNLPLab/LASP/tree/main) 因其改进的 AMD 支持以及在多个设备上拆分缓存的能力而受到赞赏，这有助于在不依赖 flash attn 仓库的情况下处理更长的上下文。
  
- **GPT-3 成本效益分析**：AI 工程师正在进行 GPT-3 的成本分析，结论是在大约 125 天后，购买 GPU 可能比租赁更具成本效益，这表明了对长期投资而非持续租赁成本的考虑。

- **Colab GPU 更新引起社区关注**：AI 工程社区对 Colab 新的 GPU 产品和价格变化做出反应，[@danielhanchen](https://x.com/danielhanchen/status/1776074818758348918) 的推文提到了新的 L4 GPU 和 A100 GPU 的价格调整，并附有一份详细列出更新内容的[共享电子表格](https://docs.google.com/spreadsheets/d/1tRbUj8qjsnZdUJOEXJFRLFSOwOWKFaA3hgRC1XWje-w/edit?usp=sharing)细节。

- **关于高级微调策略的技术讨论**：对话集中在 LoReFT、PEFT 0.10.0 版本中的量化 DoRA 等微调方法，以及来自 Facebook Research 的一种无需学习率调度（learning rate schedules）的新技术，这表明了通过创新技术优化模型性能的趋势。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **网络研讨会提醒**：不要错过网络研讨会！Jerryjliu0 在 [公告频道](https://discord.com/channels/1059199217496772688/1073670729054294197/) 提醒用户，**webinar** 将在 15 分钟后开始。

- **AI 领域对 Adaptive RAG 和 AutoRAG 充满期待**：根据最近的一条 [推文](https://twitter.com/llama_index/status/1775912690529288556)，**Adaptive RAG** 技术因其在复杂查询上的定制化性能潜力而备受关注；而 **AutoRAG** 则致力于自动优化 RAG 流水线以达到峰值性能，详见另一条 [推文](https://twitter.com/llama_index/status/1776289203459858849)。

- **视觉空间中的 RAG 重构**：AI 爱好者讨论了**视觉检索增强生成 (RAG) 模型**的潜力，该模型能够计算物体数量或根据特定条件修改图像，而 [Unlocking the Power of Multi-Document Agents with LlamaIndex](https://medium.com/ai-advances/unlocking-the-power-of-multi-document-agents-with-llamaindex-d09e4d7dfe0e) 则暗示了多文档 Agent 的最新进展。

- **技术社区的故障排除时间**：AI 工程师分享了诸如 **SQL 查询引擎的异步问题**、Azure **BadRequestError** 难题、AWS 上下文的 Prompt Engineering 技巧、**Pydantic JSON 结构**的复杂性以及 **RouterQueryEngine** 过滤器应用等挑战。

- **欢呼 Claude Calling Agent 的到来**：LlamaIndex 最新的 **Claude Function Calling Agent** 被誉为能够实现高级工具使用，现在可以在 Twitter ([推文](https://twitter.com/llama_index/status/1776051869850476840)) 上找到，并展示了集成 Haiku/Sonnet/Opus 的新应用。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Claude 陷入安全网困境**：用户报告称，与 Anthropic 的 API 相比，在 **OpenRouter** API 中使用 **Claude** 时拒绝率更高，怀疑 OpenRouter 可能添加了额外的“安全”层，从而干扰了性能。

**恢复 Midnight Rose**：**Midnight Rose** 经历了停机，但在重启集群后已恢复在线。该事件引发了用户关于切换到更具弹性的供应商或技术栈的讨论。

**多模态的交响乐**：随着向 `multimodal` 功能的转变，**Claude 3** 模型现在支持图像输入，这需要开发者更新代码。更多详情请见 [此处](https://discord.com/channels/1091220969173028894/1223444233394847864)。

**Command R+ 激发代码编写热情**：**Command R+** 是来自 Cohere 的 104B 参数模型，以其强大的编程和多语言能力著称。用户对其加入 **OpenRouter** 感到兴奋，详细的基准测试可以在 [此处](https://txt.cohere.com/command-r-plus-microsoft-azure/) 找到。

**解决 Mixtral 难题**：**Mixtral-8x7B-Instruct** 在遵循 JSON schema 时遇到了问题，该问题已由 OpenRouter（而非供应商）成功解决，用户们正热切期待修复和更新，以简化 JSON 模式的使用。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**图像生成领域的新竞争者**：提出了一种 Visual AutoRegressive (VAR) 模型，有望在图像生成方面超越 diffusion transformers，其 [Frechet inception distance (FID)](https://arxiv.org/abs/2404.02905) 从 18.65 显著提升至 1.80，inception score (IS) 从 80.4 提升至 356.4。

**重新思考 Batch Sizes 以获得更好的极小值**：工程师们正在讨论较小的 batch sizes 虽然会减慢训练速度，但由于不会跳过最优局部极小值，是否能获得更好的结果；相比之下，较大的 batch sizes 虽然可能加快训练速度，但表现可能不佳。

**像 Git 一样更新你的数据集**：AI 从业者被提醒，在 Hugging Face 上更新数据集和模型需要遵循类似 git 的规范——先在本地更新，然后进行 commit 和 push，以反映平台上的更改。

**用开源连接 AI 与音乐**：通过一段 [YouTube 视频](https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb) 展示了 musiclang2musicgen 流水线的突破，推广了开源解决方案在音频生成中的可行性。

**斯坦福为 NLP 新手提供的宝库**：对于那些刚开始学习 NLP 并在 Transformer 架构和传统模型（如 LSTM）之间做选择的人，建议将斯坦福 CS224N 课程作为首选资源，该课程可通过 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) 获取。

**微调与部署 LLM**：出现了关于 Ollama 模型部署的问题，特别是关于 `phi` 变体的内存需求，以及针对特定用例，本地部署还是像 OpenAI 这样的 API 方案更合适。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**tinygrad 的 NPU 热议与 Intel GPU 传闻**：频道内的讨论提到，虽然 **tinygrad** 在新款笔记本电脑上缺乏专用的 NPU 支持，但它提供了一个[优化清单](https://github.com/tinygrad/tinygrad/blob/master/docs/env_vars.md)用于与 **onnxruntime** 进行性能对比。社区成员们还剖析了 Linux kernel 6.8 驱动 Intel 硬件的能力，特别是在 Ubuntu 24.04 LTS 发布后，关注 Intel GPU 和 NPU 内核驱动程序的进展。

**可扩展性对话与能效讨论**：对话涉及 **tinygrad** 未来的可扩展性，George Hotz 指出了使用 200 GbE 全 16x 互连插槽进行大规模扩展的潜力，并预告了多机支持。此外还对比了 NPU 和 GPU 的能效，强调了 NPU 在功耗显著降低的情况下达到 GPU 性能水平的能力。

**内核开发的机遇与挑战**：在 AI 工程师中，大家认识到了 AVX-512 带来的障碍，并对 Intel 基于 [Real World Technologies 讨论帖](https://www.realworldtech.com/forum/?threadid=193189&curpostid=193190)进行的改进表示关注。对话还涵盖了 AMD 的开源意图，但对其真实影响持怀疑态度，并期待 [AMD Phoronix 更新](https://www.phoronix.com/news/AMD-MES-Docs-And-Source-Code)将如何影响局势。

**通过 tinygrad 的 JIT 学习**：一篇文章澄清了关于 JIT 缓存收集的困惑，一位社区成员贡献了[学习笔记](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/profiling.md)，以帮助使用 **DEBUG=2** 对 tinygrad 进行性能分析。社区正在共同努力完善一份[TinyJit 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md)，作者欢迎指正，这标志着社区对共同学习和文档准确性的承诺。

**鼓励社区协作**：对话传达了强烈的同行协作情绪，敦促有经验的成员提交 Pull Request 以纠正 TinyJit 文档中的错误，从而在参与者中推广互助的方法。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Command R+ 进入企业级领域**：Cohere 宣布推出 **Command R+**，这是一款专注于检索增强生成 (RAG) 和工具使用 (Tool Use) 的可扩展大语言模型 (LLM)，拥有 128k-token 的上下文窗口和多语言能力，其权重已在 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 上发布。

- **模型成本与性能备受关注**：全新的 **JetMoE-8B 模型**被定位为计算需求极低的经济型替代方案，据称其性能超越了 Meta 的 LLaMA2-7B，并因其对学术界的友好性而受到关注，可在 [Hugging Face](https://huggingface.co/jetmoe/jetmoe-8b) 查看详情。

- **提升效率的增强技术涌现**：讨论转向了 DeepMind 的 [Mixture of Depths](https://arxiv.org/abs/2404.02258)，该技术在 Transformer 序列中动态分配 FLOPs，可能为未来与混合专家模型 (MoE) 的集成铺平道路，并邀请大家在周末进行实验。

- **即将举行的客座讲座聚焦产研结合**：Nathan 将在 **CS25** 进行演讲，演讲嘉宾还包括来自 OpenAI、Google、NVIDIA 和 ContextualAI 的专家，详情见 [Stanford CS25 课程页面](https://web.stanford.edu/class/cs25/#schedule)，突显了持续的产学研协同效应。

- **法律威胁与功劳纠纷引发质疑**：重点讨论包括马斯克在 [推文](https://twitter.com/elonmusk/status/1775900800520262071) 中暗示的法律追诉，以及对前同事在项目功劳归属声明的质疑，揭示了社区互动中潜藏的紧张关系。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **关于基于链的 JSON 解析的讨论**：AI 工程师们就如何在 **[LangChain](https://github.com/langchain-ai/langchain/discussions/19957)** 中利用 **Output Parsers, Tools, and Evaluators** 来确保 LLM 链输出 JSON 格式进行了深入探讨。他们还解决了 **ChatGroq** 摘要错误的复杂问题，分享了处理法律文档问答分块（chunking）的策略，比较了预算型 LLM 的性能，并表达了对 LangChain 中 **RAG (retrieval-augmented generation)** 技术辅导的需求。

- **棘手的 PDF Agent 与 Azure 集成咨询**：工程师们针对一个默认进行 PDF 搜索的 Agent 搜索协议调整进行了集思广益，并就如何在保持 **FAISS Vector Database** 的同时，在 Bot 环境中集成 **Azure credentials** 进行了咨询。

- **语义分块推出 TypeScript 版本**：一位社区贡献者提交了语义分块（Semantic Chunking）的 [TypeScript 实现](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae)，从而将该功能扩展到了 Node.js 环境。

- **DSPy 教程推出西班牙语版**：针对西班牙语爱好者的 DSPy 基础教程已通过 [YouTube 教程](https://youtu.be/N1ChFQm02QM)分享，进一步扩大了该应用的可及性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 与压力和时间的较量**：社区正在讨论 **AIDE** 在 **Kaggle 竞赛**中取得的成就，质疑其是否能与包含压力和时间限制等因素的人类选手经验相提并论。虽然未达成共识，但这场辩论凸显了 AI 在竞赛数据科学领域日益增强的能力。

**回归 Apple 与 PyTorch 的基础**：技术群体对 **Apple 的 MPS** 表示不满，一些人建议尝试 PyTorch nightly 分支以寻求潜在的修复。此外，**PyTorch on macOS** 的优势（特别是 **aot_eager backend**）得到了展示，有一个案例显示该后端在利用 **Apple 的 CoreML** 时显著缩短了图像生成时间。

**音频 AI 一瞥**：人们对 **DALL·E** 的图像编辑历史等功能感到好奇，并希望在 **SDXL** 中实现类似功能。此外，还出现了关于解析播客音频的特定语音技术（超出常规的 **speaker diarization**）的提问。

**访问权限与信息的恢复**：讨论揭示了对 **Reddit API 访问**被切断及其对开发者和盲人社区影响的担忧，以及 **/r/StableDiffusion** 子版块的重新开放及其对社区的意义。

**Transformer 中的计算智能**：目前的焦点是 **Google 的 Token 压缩方法**，旨在缩小模型尺寸和计算负载；以及一篇讨论 Transformer 模型中动态 FLOPs 分配策略的论文，该策略采用 top-$k$ 路由算法来平衡计算资源和性能。该方法在论文 ["Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"](https://arxiv.org/abs/2404.02258) 中有详细描述。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**动态分配引发争议**：DeepMind 在 Transformer 中实现动态计算的方法（被称为 Mixture-of-Depths）引起了褒贬不一的反应；一些人称赞其减少了计算量，而另一些人则怀疑其新颖性和实用性。

**Claude 掌握工具使用**：Anthropic 的 Claude 展示了令人印象深刻的工具使用能力，引发了关于此类功能在 AI 系统中的实际应用和可扩展性的讨论。

**论文俱乐部准备集会**：圣地亚哥 AI 社区宣布举办论文俱乐部活动，鼓励参与者选择并深入研究 AI 相关文章，并为渴望加入的人员提供了简单的报名流程。

**ReFT 重新定义微调**：斯坦福大学推出了 ReFT (Representation Finetuning)，称其为一种参数效率更高的微调方法，AI 领域正在权衡其相对于现有技术的优缺点。

**Keras vs. PyTorch：激烈的基准测试之战**：François Chollet 强调了一项 Keras 优于 PyTorch 的基准测试，引发了关于基准测试公平性以及开箱即用速度与优化性能之间重要性的辩论。

**报名 AI 教育课程**：Latent Space 大学宣布了其首个在线课程，重点是编写自定义 ChatGPT 解决方案，邀请 AI 工程师报名，并强调该课程适用于那些希望深化 AI 产品工程知识的人。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**OpenInterpreter Talks the Talk**: 开发了一个用于 **OpenInterpreter** 语音交互的创新封装器，尽管它在语音能力上还不及 **01**。社区正在讨论安装和兼容性挑战，**Windows 用户正面临困难**，且 **CTRL + C** 无法按预期退出终端。

**与 OpenAI 的对比**: **OpenAI API playground** 中出现了一个神秘的 **Compare endpoint**，但目前尚无正式文档；它支持模型和参数之间的直接对比。

**Python 困境与 Ubuntu 问题**: **OpenInterpreter** 的 **01OS** 正在处理 Python 3.11+ 的不兼容问题，建议回退到 **Python 3.10** 或更低版本以保证稳定性。同时，由于 Wayland 不兼容，**Ubuntu 21** 及以上版本的用户发现无法支持 **OpenInterpreter**，正如 [Issue #219](https://github.com/OpenInterpreter/01/issues/219) 中提到的，x11 仍然是必需的。

**只听不答**: 用户报告了 **01 音频连接** 的异常问题，声音已被录制但未传输进行处理，这表明可能存在新的客户端 Bug。

**Conda 难题**: 为了处理棘手的 TTS 包安装，建议使用 **Python 3.10 或更低版本** 创建 **Conda 环境**，然后重新克隆仓库并进行干净安装以避开冲突。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**LLM 中的 BitMat 突破**: **BitMat** 的实现备受关注，反映了通过 GitHub 上 [astramind-ai/BitMat](https://github.com/astramind-ai/BitMat) 托管的高效方法进入“1-bit LLMs 时代”的进展。

**QuaRot 消除量化疑虑**: 一种名为 [QuaRot](https://arxiv.org/abs/2404.00456) 的新量化方案承诺实现 Large Language Models 的高效端到端 4-bit 量化，值得注意的是，量化后的 LLaMa2-70B 模型保持了 99% 的 zero-shot 性能。

**CUDA Kernel 教程获得点赞**: 备受推崇的 [Udacity 课程 "Intro to Parallel Programming"](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2) 因其在并行算法和性能调优方面的持久价值而再次被提及，即使在推出十年后依然适用。

**HQQ-GPT-Fast 融合**: **#hqq** 频道中关于将 **HQQ** 与 **gpt-fast** 集成并进行基准测试的讨论非常激烈，重点在于利用 **Llama2-7B 模型** 并尝试 3/4-bit 量化策略以优化 LLM。

**增强可视化旨在提高清晰度**: **Triton-viz** 的讨论旨在通过增加方向箭头、在交互元素上显示数值以及可能转向 **Three.js** 等 JavaScript 框架以获得更出色的交互性，从而更好地展示可视化中的数据流。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **智慧引导的 AI 产品开发**: 对 [Hamel Husain 的博客文章](https://hamel.dev/blog/posts/evals/) 的深入探讨引发了关于评估 AI 系统最佳实践的讨论，重点在于其对构建稳健的 AI 功能和企业的实用性。

- **Datasette 倡议**: 计划为 Datasette SQL 查询助手插件构建评估体系，重点是通过 **Prompt 可见性和可编辑性** 赋能用户。

- **完善 Prompt 管理**: 提出了在大规模应用中管理 AI Prompt 的三种策略：使用独立 Prompt 文件的本地化模式、使用 API 获取 Prompt 的中间件模式，以及用于 AI 服务管理的微服务模式。

- **剖析 Cohere LLM 的 JSON 宝库**: Cohere LLM 的 JSON 响应的丰富性得到了强调，详细的 [GitHub issue comment](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135) 证明了这一点，揭示了其在增强 LLM 用户体验方面的潜力。

- **DSPy：意见分歧**: 社区对 DSPy 框架的看法存在分歧；一些成员对其将 LLM 简化为“黑盒”的方法表示怀疑，而另一些人则对其引入的不可预测性表现出热情，将其比作 AI 中的一种魔幻现实主义。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**以创意评判书籍**：新的 [**EQBench Creative Writing 和 Judgemark 排行榜**](https://eqbench.com) 因其对 LLM 创意输出和判断能力的独特评估而引发关注。值得注意的是，**Creative Writing** 排行榜利用 **36 个精确定义的标准** 来实现更好的模型区分度，并建议使用 **0-10 质量量表** 进行细微的质量评估。

**COMET 的新脚本上线 GitHub**：两个用于无参考翻译评估的脚本 `comet_eval.ipynb` 和 `overall_scores.py` 现在已在 [**llm_translation GitHub 仓库**](https://github.com/CrispStrobe/llm_translation) 中可用，这标志着在透明度和标准化 LLM 性能测量方面迈出了一步。

**Cohere 的 Demo 脱颖而出**：**CohereForAI** 在 [Hugging Face 平台](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 上发布的新 Demo 展示了 AI 模型 grounding 能力的重大飞跃，引发了关于其塑造未来模型开发潜力的讨论。

**传统翻译方式被“上课”了**：Hugging Face 上的模型 **command-r** 凭借其出色的翻译能力，似乎使传统的 LLM 中古高地德语翻译训练方法变得过时，并被建议在推理过程中彻底改变语言数据库的集成方式。

**思考模型许可的未来**：CohereForAI 模型许可潜在的开源化是一个热门话题，涉及 **GPT-4** 和 **Nous Hermes 2 Mixtral** 的对比讨论强调了预期的社区增长和创新，这可能会镜像 Mistral 模型的影响。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 的 Solo 进军建站领域**：Mozilla 自豪地推出了 [**Solo**](https://bit.ly/soloforcommunity)，这是一款专为企业家设计的新型**无代码 AI 网站构建器**，目前处于 Beta 测试阶段。为了完善该工具，Mozilla 正在寻找能够提供宝贵反馈的*早期产品测试人员*。

- **针对 AI 模型的 GPU 使用优化**：工程师建议在 `-ngl` 之前使用 `--gpu nvidia` 以获得更好的模型运行性能；一个 **16GB 4090 Mobile GPU** 在 Q8 量化下支持多达 10 层。能够高效运行的确切层数可能因模型和 GPU 容量而异。

- **使用 Intel 工具**：**Intel 的 oneAPI basekit** 正与 `icx` 一起使用，因为在处理 `sycl` 代码和 `onemkl` 时它是必需的，这在 Intel 生态系统中非常重要。这种集成强调了 Intel 在 AI 运营工作流中的重要作用。

- **Kubernetes 集群与 AI 性能**：在 Kubernetes 集群（7 核，12GB RAM）中使用 mistral-7b 模型产生了每秒 5 个 token 的稳定速率；目前正在讨论 RAM 大小、RAM 速度、CPU 还是 GPU 功率在扩展此性能中起着最大作用。

- **警惕 AI 工具中可能存在的恶意软件**：针对 llamafile-0.6.2.exe 被标记为恶意的网络安全担忧已引起用户警惕。VirusTotal 报告显示 llamafile 的 0.6.2 和 0.7 版本均已被标记，后者的风险评分较低，详见 [VirusTotal](https://www.virustotal.com/gui/file/37a39d8970573110c425c3edd1be4b1df6ab32c4a4a38ae6d98ad4728093267e) 等参考资料。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **通过 MoD 进行动态计算分配**：Skunkworks AI 中讨论了为语言模型引入的 **Mixture-of-Depths (MoD) 方法**，该方法能够实现类似于 *MoE transformers* 的动态计算分配，但使用的是单一专家，通过 *top-k 路由机制* 进行优化。这篇 [研究论文](https://arxiv.org/abs/2404.02258) 阐述了针对特定 token 位置进行更高效处理的潜力。

- **独立的视频内容**：共享了一个 YouTube 视频链接，但没有提供额外的上下文，这可能超出了技术讨论的范围。

- **未说明的论文引用**：一名成员分享了一个 arXiv 论文链接，但没有附带评论，使其相关性不明。该文档可以在 [此处](https://arxiv.org/abs/2404.02684) 访问，但在没有上下文的情况下，无法确定其对当前对话的重要性。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Anthropic 的 AI 登台亮相**：一位成员分享了来自 [AnthropicAI 的推文](https://twitter.com/AnthropicAI/status/1775979799644934281?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet)，指出其新 AI 模型初步测试结果令人振奋。
- **高性能伴随高延迟**：尽管该 AI 的能力受到赞赏，但有人担心在处理串行 AI 操作时，**延迟问题**会成为瓶颈。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1225341130997108746)** (1314 条消息🔥🔥🔥): 

- **Perplexity Pro 咨询**：用户询问了 **Perplexity Pro** 的功能和可访问性。他们讨论了如何启用频道、无法删除已上传文件的问题，以及由于重定向导致购买 Pro 计划困难的情况。支持人员建议通过电子邮件或直接联系管理员寻求帮助。

- **云服务与 AI**：对话集中在云服务提供商及其在 LLM 竞赛中的角色。用户对 **AWS** 和 **Azure** 的市场份额进行了辩论，并讨论了 **Perplexity** 与 **Anthropic** 之间潜在合作的猜测。

- **Apple 的 Vision 和 AI 策略**：用户表达了对 **Apple 3b 模型**的看法，讨论了其小众用例以及对更轻量、更廉价迭代版本的需求。有观点认为 **Apple Glass** 可能会更主流，而 Google 的 VR 计划更受欢迎。

- **AI 模型使用与微调**：用户询问了 GPT-4 Turbo 与 Claude Opus 的上下文长度，建议在 **32k tokens** 处保持一致。关于开源模型的讨论强调了 **Stable Diffusion 3** 以及政府干预开源决策的可能性。

- **Arc 上的用户界面与可访问性挑战**：用户分享了更高效使用 **Arc 浏览器**的技巧，并报告了影响用户界面的 Bug，包括更改设置和访问扩展程序时的问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/pricing">Pricing</a>: 未找到描述</li><li><a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>: 我们正在添加新功能，以帮助开发者更好地控制 fine-tuning，并宣布了与 OpenAI 构建自定义模型的新方法。</li><li><a href="https://tenor.com/view/yes-no-gif-16236377">Yes No GIF - Yes No - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/pokemon-pokemon-go-the-pokemon-company-nintendo-niantic-gif-23793876">Pokemon Pokemon Go GIF - Pokemon Pokemon Go The Pokemon Company - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/is-it-gif-20766212">Is It GIF - Is It - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sal-lurking-sal-lurk-sal-vulcano-sal-lurk-gif-21668959">Sal Lurking Sal Vulcano GIF - Sal Lurking Sal Lurk Sal Vulcano - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: 未找到描述</li><li><a href="https://youtu.be/MHyPfZu-YiU?si=OvoG-BBOwJ5Pmp0e">OpenAI&#39;s STUNNING &quot;GPT-based agents&quot; for Businesses | Custom Models for Industries | AI Flywheels</a>: 加入我们的论坛：https://www.natural20.com📩 我的 5 分钟每日 AI 简报 📩https://natural20.beehiiv.com/subscribe🐥 在 Twitter (X) 上关注我 🐥https://twitter....</li><li><a href="https://www.youtube.com/watch?v=JV4JbYK-TIg">1111Hz Conéctate con el universo - Recibe guía del universo - Atrae energías mágicas y curativas #2</a>: 1111Hz 与宇宙连接 - 接收宇宙的指引 - 吸引魔法与治愈能量 #2。这个频道致力于治愈你的心灵、灵魂、身体...</li><li><a href="https://youtu.be/Q8mdaTGNa_A?si=JeT6RqDgV9axSFU0>)">2024年から始めるPerplexityの使い方超入門</a>: 为“没有时间”、“没有技能”的人提供的博客代写服务“Hands+”已启动。 → https://bit.ly/blog-beginner 针对希望增加来自搜索引擎流量的企业，提供自有媒体搭建服务请点击这里 → https://bit.ly/owned-media6...</li><li><a href="https://youtu.be/Q8mdaTGNa_A">2024年から始めるPerplexityの使い方超入門</a>: 为“没有时间”、“没有技能”的人提供的博客代写服务“Hands+”已启动。 → https://bit.ly/blog-beginner 针对希望增加来自搜索引擎流量的企业，提供自有媒体搭建服务请点击这里 → https://bit.ly/owned-media6...</li><li><a href="https://youtu.be/ADyaGQza8pI?si=otoajSewAMrRC3x4">Revolutionizing Search with Perplexity AI | Aravind Srinivas</a>: 加入主持人 Craig Smith 在 Eye on AI 第 175 集，与 Perplexity AI 的联合创始人兼 CEO Aravind Srinivas 进行一场启发性的对话。</li><li><a href="https://www.youtube.com/watch?v=qRKScRgsUaE">Inside Japan&#39;s Nuclear Meltdown (full documentary) | FRONTLINE</a>: 2011 年 3 月 11 日，一场毁灭性的地震和海啸袭击了日本，引发了福岛第一核电站内部的危机。这部 2012 年的纪录片...</li><li><a href="https://www.imdb.com/title/tt0218817/">Antitrust (2001) ⭐ 6.1 | Action, Crime, Drama</a>: 1 小时 48 分钟 | PG-13
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1225384383838552125)** (11 messages🔥): 

- **Perplexity AI 实战**：成员们分享了各种 [Perplexity AI search](https://www.perplexity.ai) 链接，涉及**美容**、**独裁政权**的崛起以及与 **Cohere** 的 **Command R** 相关的查询。
- **将 Thread 设置为可共享**：一位成员提醒其他人确保他们的 Thread 已设置为可共享，并提供了一个 [Discord 指令链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **寻求理解与改进**：用户通过 Perplexity AI 查询了不同主题的见解，并表达了在充满挑战的一天中通过 [introducing improvements](https://www.perplexity.ai/search/Introducing-improvements-to-LkJPv.JVScuV0Ak3R_G7Dg) 链接寻求改进。
- **从哲学到个人**：讨论范围从“Jean 是谁？”等常识性查询，到暗示当天特定需求或事件的更个性化搜索。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1225517810856235038)** (18 messages🔥): 

- **关于 Search API 成本担忧的澄清**：一场关于为什么 **search API** 被认为昂贵的讨论被触发。*Icelavaman* 澄清说，成本不是按搜索次数计算，而是按 request 计算，这意味着对模型的单个 request 可以封装其中的多次搜索。

- **购买更多 API 额度**：*Mydpi* 询问了如何使用 **Google Pay** 为其 Pro 订阅购买额外的 API 额度，*ok.alex* 确认可以通过网页端的 API 设置购买更多额度，且 **Google Pay** 是有效的支付方式。

- **Sonar-Medium-Chat 定价查询**：*Julianc_g* 询问了 **sonar-medium-chat** 的定价，*icelavaman* 回复确认价格为 **每 1M tokens $0.60**。

- **关于订阅奖励和支付方式的查询**：*Mydpi* 收到 *ok.alex* 的回复，指出 **$5 Pro Bonus** 以及购买更多额度可以通过网页设置页面中的 **Google Play** 支付选项进行管理。

- **速率限制与并发 API 调用**：Perplexity AI FAQ 提供了关于**并发 API 调用**的信息，并建议参阅官方 pplx-API 文档中提到的 [rate limits](https://docs.perplexity.ai/docs/rate-limits) 和 [pricing structure](https://docs.perplexity.ai/docs/pricing) 以获取更多详情。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1225363481616846870)** (600 messages🔥🔥🔥): 

- **图像生成的生成速度与真实感**：一位用户在使用 "realistic vision v6" 生成高分辨率 (2k) 的真实人物图像时遇到困难。建议是先以较低分辨率生成，然后进行放大 (upscale)，使用较少的步数 (steps)，并启用 "hiresfix" 以获得更好的效果。讨论集中在维持放大质量的挑战以及有时会导致的失真问题。

- **Stable Diffusion 3 的期待与访问权限担忧**：在对即将发布的 Stable Diffusion 3 (SD3) 的兴奋中，一些用户觉得发布延迟了，而另一些人则期待新的邀请函以示进展。关于 SD3 的看法从其潜在的改进到对其即将发布的怀疑，以及与 Ideogram 和 DALLE 3 等竞争模型的比较。

- **AI 深入艺术领域**：用户讨论了将 AI 与艺术创作相结合，例如利用 Daz AI 生成图像。对话包括生成特定风格图像的策略，以及优化模型训练和合并技术，以便使用 Stable Diffusion 创作诸如服装 'Loras' 之类的内容。

- **模型资源需求的技术讨论**：对话涉及了技术层面，如在不同 VRAM 容量下运行模型、在不同用户界面 (UI) 之间处理 checkpoints，以及即将推出的 SD3 在消费级 GPU 上高效运行的前景。

- **Stable Diffusion 使用的探索与优化**：用户交流了关于使用不同版本的 Stable Diffusion 模型和用户界面的技巧并寻求建议。他们讨论了生成更高质量图像的替代方案、图像微调 (finetuning) 的过程以及处理模型 checkpoints。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: 我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“次尺度预测 (next-scale prediction)”或“次分辨率预测”...</li><li><a href="https://hforsten.com/identifying-stable-diffusion-xl-10-images-from-vae-artifacts.html">Identifying Stable Diffusion XL 1.0 images from VAE artifacts</a>: 最近发布的全新 SDXL 1.0 文本生成图像模型在图像中会产生微小的伪影，而早期的 0.9 版本则没有这些问题。</li><li><a href="https://leonardo.ai/">Home v2</a>: 使用我们的 AI 图像生成器改变您的项目。以无与伦比的速度和风格生成高质量的 AI 图像，提升您的创意愿景。</li><li><a href="https://mo-www.cfa.harvard.edu/cgi-bin/OWN/Information.pl">Observing with NASA</a>: 使用 MicroObservatory 机器人望远镜网络控制您自己的望远镜。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bnjm3i/comment/kwjb37c/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://m.soundcloud.com/pelusitalachicafideo/never-gonna-give-you-up-rick-astley-minions-ver">Never Gonna Give You Up - Rick Astley [Minions Ver.]</a>: 在桌面和移动端上收听 Pelusita,la chica fideo 制作的 Never Gonna Give You Up - Rick Astley [小黄人版]。在 SoundCloud 上免费播放超过 3.2 亿首曲目。</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: 通过在 GitHub 上创建账户，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://civitai.com/models/161068?modelVersionId=312982">NewRealityXL ❗ All-In-One Photographic - ✔2.1 Main | Stable Diffusion Checkpoint | Civitai</a>: 重要提示：v2.x ---&gt; 主版本 | v3.x ---&gt; 实验版本。我需要大家花时间彻底测试这个新的第 3 版本，以了解所有...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1225460335616458782)** (1 条消息): 

- **通过 Fine-Tuning API 增强开发者控制力**：OpenAI 宣布了对 Fine-Tuning API 的增强，引入了新的**仪表板**、**指标**和**集成**，为开发者提供更大的控制权，并扩展了定制模型计划，提供了构建量身定制的 AI 解决方案的新选项。[介绍 Fine-Tuning API 的改进并扩展我们的定制模型计划](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program)以及[关于各种技术的 YouTube 视频](https://www.youtube.com/watch?v=ahnGLM-RC1Y&list=PLOXw6I10VTv-exVCRuRjbT6bqkfO74rWz&index=4)详细介绍了如何增强模型性能并与 OpenAI 专家合作开发定制 AI 实现。

**提到的链接**：<a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>：我们正在添加新功能，以帮助开发者更好地控制微调，并宣布与 OpenAI 合作构建定制模型的新方法。

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1225356652589350973)** (539 条消息🔥🔥🔥): 

- **AI 讨论涵盖广泛领域**：用户就 AI 展开了激烈的讨论，范围从机器认知到 AI 对 ASCII 艺术的理解和生成。辩论涉及了感知力、意识等术语和概念，以及 AI 认知过程的本质，包括 LLM 是在“思考”还是仅仅在处理信息。

- **对商业创意和 AI 局限性的反思**：一位用户提出了一个利用 AI 能力赚钱的商业创意，涉及创建由生成的技巧汇编而成的 AI Prompt。另一位成员思考了使用语言模型执行传统上与人类相关的任务的可能性，如通过下棋或成功的商业规划。

- **对 AI 在各领域潜力的推测**：用户表达了对 AI 集成到 3D 打印和设计等领域的期待，提出了诸如 3D 建模的生成式填充等想法，这可能会彻底改变制造业。

- **关于 AI 产品发布的担忧与考量**：一个讨论点强调了对 AI 产品发布流程的挫败感，指出 OpenAI 出于安全考虑采取的谨慎立场与用户渴望不受限制地访问新 AI 能力之间的矛盾。

- **关于实现 AI 功能的查询**：针对实现文档分析以及在公司内部数据增强中应使用微调（fine-tuning）还是嵌入（embeddings）的问题，用户讨论了不同 AI 技术在特定应用中的有效性和适用性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/China_brain">China brain - Wikipedia</a>：未找到描述</li><li><a href="https://tenor.com/view/wow-really-gif-25055968">Wow Really GIF - Wow Really - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.asciiart.eu/food-and-drinks/bananas">ASCII Art Bananas - asciiart.eu</a>：一个包含香蕉 ASCII 艺术画及其他相关食物和饮料 ASCII 艺术图片的庞大集合。
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1225467158738239528)** (11 条消息🔥): 

- **Zero Temperature 乱象**：一位成员报告称，即使 Assistant 的 temperature 设置为 0.0，在不同线程中仍会出现随机行为，质疑该设置下的一致性。
- **追求 Prompt 的完美**：一位用户询问是否有 GPT Prompt Enhancer 来改进他们的 Prompt，另一位成员将其引导至特定频道以获取建议。
- **模拟聊天机器人的动态响应**：一位用户希望在其聊天机器人 API 中模拟显示“正在分析 PDF 文档”或“正在搜索网络”等进度消息。他们得到的建议是，此类功能需要进行自定义开发。
- **矩阵中的错误**：一位参与者注意到 GPT-4 经常在计算过程中返回“分析错误（error analysing）”，并询问是否有解决方案。
- **确认 GPT 使用的订阅制度**：一位用户询问应用中的 GPT 模型是否免费使用；另一位用户澄清说，由于所有模型都使用 GPT-4，因此必须订阅 Plus 或更高版本的方案。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1225443348769280020)** (15 条消息🔥): 

- **扩展文本输出**：成员们讨论了让 GPT-3 生成更长文本的策略，因为仅说明“让文本更长”似乎不再有效。建议包括复制输出、开启新对话并使用“继续（continue）”命令，尽管存在丢失上下文和风格的担忧。

- **解决 LLM 模板不一致问题**：一位成员就如何确保 LLM 返回修改后的文档模板的所有部分寻求建议，并指出如果 LLM 认为某些部分未更改，则会将其省略。社区尚未提供解决方案。

- **编写 Prompt 以限制 GPT 对训练数据的依赖**：一位成员寻求关于编写 Prompt 的建议，使 GPT 仅专注于从提供的文档中获取答案，而不默认使用其通用训练数据。建议包括降低 temperature 设置，并在指令中明确要求模型在继续操作前确认答案存在于给定文档中。

- **强制执行受文档约束的回答**：为了更好地确保 GPT 的回答完全取自提供的材料，一位成员建议使用激进且严厉的指令，例如，如果答案未在文档中明确找到，则命令模型“抛出错误（THROW AN ERROR）”。

- **在 GPT 中模拟类人交互**：一位成员正在进行 GPT 实验，讨论意识的本质，并尝试通过对血清素等人类化学物质的伪代码解释来模拟人类情感。对话涉及了机器学习与人类体验（如多巴胺反应）之间的相似之处。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1225443348769280020)** (15 条消息🔥): 

- **解决重复的文本扩展问题**：用户讨论了“让文本更长”的命令如何不再产生更长的文本变体，而是重复相同的内容。为了解决这个问题，建议采取开启新对话并使用“继续（continue）”命令等策略，但也引发了对风格不一致和忽略上下文的担忧。

- **弥合 AI 文档起草中的差距**：一个讨论点涵盖了 LLM 无法识别并合并文档某些部分修改的问题。一位用户正困扰于 LLM 不承认对文档所做的更改，并为此寻求解决方案。

- **确保 GPT 履行其设计的角色**：重点在于指示 GPT 严格根据用户提供的文档回答查询，避免依赖其预训练知识。建议降低 temperature 设置并在 Prompt 中保持强硬态度，以有效执行此规则。

- **在 AI 中模拟人类情感**：一位用户与 GPT 就意识的本质进行了对话，要求它使用伪代码模拟人类的化学反应。这种互动旨在探索机器对类人情感的模拟。

- **严厉指令的秘诀**：有人建议，指导 GPT 更有效的方法是保持简洁和坚定，类似于“意大利式风格”，从而强调清晰度并严格遵守指定的来源。
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1225351259830095944)** (198 messages🔥🔥): 

- **LM Studio 的互联网独立性**：成员们确认 [LM Studio](https://lmstudio.ai) 不具备“搜索网页”的能力，这与 co-pilot 或基于云的语言模型（LLM）中的功能不同。
- **探索 Chatbot Arena 排行榜**：一些成员讨论了模型性能，并分享了 [LMsys Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 等链接来突出显示可用模型，并指出排行榜前列中只有部分模型允许本地部署。
- **Anything LMM 文档问题**：用户报告了在 Anything LMM 工作区中嵌入文档的问题，这些问题通过下载正确版本的 LM Studio 或确保安装了必要的依赖项（如 Windows 的 C Redistributable）得到了解决。
- **关于多 GPU 支持与性能的讨论**：进行了多次关于 LM Studio 中多 GPU 设置有效性的交流，共识是虽然可以使用多个 GPU，但带来的性能提升可能与硬件能力的增加不成正比。根据可用的系统规格推荐了特定的模型。
- **某位社区成员的缺席**：一段简短的对话提到了多产的开源模型创作者 @thebloke，表达了对其贡献的感谢，并询问了他近期的动态。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/">Documentation | LM Studio</a>：技术参考</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：Text Embeddings 处于 beta 阶段。从此下载支持该功能的 LM Studio。</li><li><a href="https://rentry.org/LMSTudioFAQ">The unofficial LMStudio FAQ!</a>：欢迎来到非官方的 LMStudio FAQ。在这里你可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源软件...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1225387444216533083)** (85 messages🔥🔥): 

- **Mixtral 与 Mistral 的区别澄清**：讨论强调 **Mixtral** 结合了 8x7b 模型来模拟 56b 参数模型，而 **Mistral** 是标准的 7b 模型。
- **耗电巨兽**：用户讨论了在 3090 等高端 GPU 上运行 **Mixtral 8x7b** 的需求和挑战，指出其速度极慢，Token 速度约为 5 tok/s。
- **Command-R Plus 的兼容性问题**：成员们分享了在本地运行 103b **Command-R Plus** 模型的经验和困难，参考了 GitHub 上的一个实验性分支和 HuggingFace 空间，表明该模型尚未在 LLamaCPP 或 LM Studio 中得到支持。
- **Eurus-7b 亮相**：从 HuggingFace 分享了一个专为推理设计的新型 7b 模型 **Eurus-7b**，它采用了基于 **UltraInteract** 数据集多轮轨迹对的 **KTO 微调**。
- **Mamba 模型支持**：一次交流中提到了基于 **Mamba 的 LLM** 的可用性及其在 **llamacpp** 中的支持，并附带了 HuggingFace 仓库链接，尽管其与 LM Studio 0.2.19 beta 版本的兼容性尚不确定。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: 未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bvniaz/command_r_cohere_for_ai_104b/ky12kw5/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://plainenglish.io/community/direct-preference-optimization-dpo-a-simplified-approach-to-fine-tuning-large-language-models">Direct Preference Optimization (DPO): A Simplified Approach to Fine-tuning Large Language Models</a>: 未找到描述</li><li><a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GGUF/tree/main">Qwen/Qwen1.5-32B-Chat-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/christopherthompson81/quant_exploration">christopherthompson81/quant_exploration · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6387">ggml : update mul_mat_id to use the same tensor for all the experts by slaren · Pull Request #6387 · ggerganov/llama.cpp</a>: 将内存中专家的存储方式从每个专家一个 tensor 更改为包含所有专家的单个 3D tensor。这将允许我们支持具有大量专家的模型，例如 qwen2moe。现有...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491/files>">Add Command R Plus support by Carolinabanana · Pull Request #6491 · ggerganov/llama.cpp</a>: 更新了 tensor 映射，为 GGUF 转换添加了 Command R Plus 支持。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1225884680008503366)** (1 条消息): 

- **LM Studio 填补社区空白**：**LM Studio 团队**和 **@159452079490990082** 在 Hugging Face 上推出了全新的 [“lmstudio-community” 页面](https://huggingface.co/lmstudio-community)，在 @330757983845875713 缺席后为社区提供最新的 GGUF 量化模型。**@159452079490990082** 将担任专门的 LLM 档案管理员。
- **快速查找 GGUF 量化模型**：建议用户在 LM Studio 中搜索 `lmstudio-community`，以便**快速查找并实验**新模型。
- **LM Studio 社区的 Twitter 热议**：LM Studio 在 Twitter 上宣布了他们的新社区计划，邀请关注者查看他们的 [Hugging Face 页面以获取 GGUF 量化模型](https://huggingface.co/lmstudio-community)。该帖子确认了与作为 LLM 档案管理员的 **@bartowski1182** 的合作。

**提到的链接**：<a href="https://x.com/LMStudioAI/status/1776324680124694654">来自 LM Studio (@LMStudioAI) 的推文</a>：如果你在这里待得足够久，你可能和我们一样想念 @TheBlokeAI 🥲。我们和 @bartowski1182 决定尝试填补这一空白。我们很高兴能分享这个新...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225600657126326333)** (8 条消息🔥): 

- **搜索重置困惑已澄清**：一位成员指出，在删除查询并按回车后，搜索结果不会重置。然而，官方澄清说初始状态没有搜索结果，可以在主页上找到精选的模型列表。

- **预设创建功能说明**：针对无法创建新预设的疑问，一位成员获得了如何在 LM Studio 中创建新预设的指导。

- **LM Studio 相比竞争对手获得好评**：一位用户称赞 LM Studio 相比其他本地 LLM GUI（如 oogabooga text generation UI 和 Faraday）能产生最好的结果，即使在使用相同的模型和指令时也是如此。

- **大量功能请求**：一位成员为 LM Studio 请求了多项更新，包括支持读取文件、多模态功能（文生图、文生音等），以及类似于名为 Devin 的现有工具的增强工具以提高性能。

- **询问社区成员缺席情况**：有人询问社区成员 TheBloke 缺席的原因，并对其近况表示关切。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1225637667836923955)** (2 条消息):

- **频道重启**：一名成员通过一条简短的消息发起了对话：“取消归档此频道。”
- **寻找最佳博客写作伙伴**：一名成员在聊天机器人讨论的背景下询问了**写作博客的最佳模型**。
  

---


**LM Studio ▷ #[🛛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1225500987616133302)** (21 messages🔥): 

- **混合 GPU 配置引发好奇**：一位用户询问结合使用 **Nvidia 和 Radeon 显卡** 是否可以合并 VRAM 或并行运行，但得到的答复是由于 **CUDA/OpenCL/ROCm 不兼容**，这并不可行。不过，可以运行多个独立的 LM Studio 实例，每个实例使用不同的显卡。
- **优化 LM Studio 中的 GPU 使用**：有一个关于为什么 LM Studio 似乎没有利用 RTX 4070 运行大型模型的疑问，随后引发了关于通过 **VRAM offloading** 确保 GPU 加速的讨论。成员们建议在该用户稍后处理此问题时，检查 GPU Offload 设置和模型层级配置。
- **混合使用新旧 Nvidia 显卡**：关于将较新的 RTX 3060 与较旧的 GTX 1070 混合使用的效果讨论浮出水面，共识是相似的 GPU 会带来更好的性能。一位成员分享了他们的个人配置，表示性能有明显提升，但认为这只是升级到匹配显卡之前的临时方案。
- **Intel AMX 在 LM Studio 中的潜力**：有人提出了关于 LM Studio 是否能够利用 **Intel Xeon 第 4 代高级矩阵扩展 (AMX)** 的问题，尽管讨论中没有提供明确的答案。
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1225364496147546116)** (54 messages🔥): 

- **探索 LM Studio Text Embeddings**：LM Studio 0.2.19 Beta 引入了 Text Embeddings，允许用户通过服务器的 POST /v1/embeddings 端点在本地生成 Embeddings。用户被引导至 [LM Studio 文档](https://lmstudio.ai/docs/text-embeddings) 阅读有关 Text Embeddings 的内容。
  
- **版本混淆已澄清**：一些用户对他们当前的 LM Studio 版本感到困惑，官方澄清 Beta 版本是基于最后一个构建版本的，版本号会在正式发布时更新。
  
- **对 LM Studio 2.19 Alpha 的期待**：成员们对 LM Studio 2.19 的 Alpha 版本表示兴奋，该版本包含 Text Embeddings 支持，可以从 Beta Releases 下载。

- **关于 Pythagora 的咨询与更新**：用户讨论了 Pythagora（也称为 GPT-Pilot），这是一个能够构建应用程序的 Visual Studio Code 插件。网站 [Pythagora](https://www.pythagora.ai/) 提供了更多关于其功能以及与各种 LLM 集成的详细信息。

- **ROCM 版本落后但受到好评**：一位用户提到 ROCM 构建版本往往落后于主版本，但即使在当前状态下，尽管存在一些 Bug，它在安装便捷性和功能性方面仍获得了积极反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://www.cloudflare.com/learning/ai/what-are-embeddings/#:~:text=Embeddings%20represent%20real%2Dworld%20objects,Glossary">什么是机器学习中的 Embeddings？</a>: Embeddings 是将现实世界对象（如单词、图像或视频）表示为机器学习模型可以轻松处理的形式的向量。</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings 处于 Beta 阶段。从此处下载支持该功能的 LM Studio。</li><li><a href="https://www.pythagora.ai/">Pythagora</a>: 未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main">nomic-ai/nomic-embed-text-v1.5-GGUF at main</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1225532162145517568)** (10 messages🔥):

- **排除 Autogen 短响应故障**：在 LM Studio 配合 Autogen Studio 使用时，用户遇到了**推理仅产生 1 或 2 个 tokens** 的问题。另一位成员确认这是一个反复出现的问题。
- **对新 Multi-Agent 系统的期待**：一位成员提到正在开发自己的 **multi-agent system** 以解决 Autogen 的问题，并计划在本周末发布。
- **建议将 Crewai 作为 Autogen 的替代方案**：Crewai 被推荐作为 Autogen 的替代品，但指出仍需要一定的编程能力才能有效利用。
- **新系统预计将提供用户界面**：开发新解决方案的成员承诺将提供**用户界面 (UI)**，这意味着无需编写代码即可更轻松地使用。
- **保持发布前的神秘感**：尽管引起了关注，但由于该项目的域名注册仍在进行中，因此尚未分享新系统的截图或更多细节。
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1225408139508322384)** (1 messages): 

- **关于保留 Memory 的咨询**：一位成员对成功让机器人分析文件表示好奇，并想知道如何让机器人在整个运行过程中保留 memory。在给出的消息中未提供解决方案或后续跟进。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1225450306993786910)** (27 messages🔥): 

- **AMD GPU 兼容性查询**：用户讨论了 AMD GPU 上 **ROCm** 的兼容性问题，特别是 **6700XT (gfx 1031)**。一位用户报告尽管尝试了各种配置仍无法加载模型，而另一位用户建议这可能是 AMD 需要解决的驱动程序问题。

- **ROCm 性能见解**：据报告，使用 **ROCm** 相比 OpenCL 有显著的性能提升；一位用户指出生成任务从 12T/s 增加到 33T/s，这突显了对 AMD OpenCL 实现的批评。

- **Linux vs. Windows 支持 ROCm**：提到 **ROCm** 在 Windows 上存在功能限制，而 Linux 则没有，在 Linux 上用户可以伪装芯片版本以使某些 GPU 正常工作。有迹象表明，如果 Linux 版 ROCm 发布，LM Studio 可能会支持更多显卡。

- **对开源 ROCm 的期待**：分享了 **@amdradeon 的一条推文**，内容关于 ROCm 即将开源，这增加了在更多 AMD 显卡上实现更简便的 **Linux build support** 的希望。开源 ROCm 的引入可能会扩大兼容性 ([Radeon's Tweet](https://twitter.com/amdradeon/status/1775261152987271614))。

- **用户探索与配置**：讨论并比较了不同的设置，提到了禁用 iGPU 以运行正确容量的 VRAM，以及涉及双 GPU 和从游戏转向 AI 及机器学习工作负载的高性能构建的各种配置。

**提及链接**：<a href="https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/">Reddit - Dive into anything</a>：未找到描述

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1225421112977133569)** (22 messages🔥): 

- **处理 CORS**：一位成员询问了 **CORS** (Cross-Origin Resource Sharing)，但没有提供细节或背景的后续讨论。
- **成功执行代码**：通过调整任务中的 "expected_output"，一位成员成功运行了共享代码，表明其问题已得到解决。
- **寻求 Agent 活动日志**：一位成员期望在 **LM Studio server logs** 中看到 Agent 活动日志，但尽管确认 **verbose** 选项已设置为 true，却未发现任何条目。
- **LM Studio 中的日志难题**：关于 **LM Studio** 在与 **crewAI** 交互时是否应显示日志尚无共识，成员们表示不确定，且未提供明确的解决方案。
- **在使用 crewAI 时遇到错误**：在遇到与未终止字符串相关的 "json.decoder.JSONDecodeError" 后，一位成员寻求解决该问题的建议，并有人建议从错误消息内容中寻找线索。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1225485539952689162)** (2 messages): 

- **正在开发 Mistral 7B 的 LoRA 层**：一位成员建议在 **Mistral 7B** 等模型之上创建 **LoRA (Low-Rank Adaptation)**，以显著增强其能力。
- **AI 的高级任务涉及 Taxonomy**：针对 LoRA 的建议，有人透露计划不仅要拆分句子，还要根据特定任务的**分类法 (taxonomy)** 对每个句子进行分类。
  

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1225436358508806315)** (10 messages🔥): 

- **Web Crawling 技术现状咨询**：一位成员在尝试确定 Web Crawling 技术的当前最先进实践时感到迷茫。
- **区分存档与囤积**：一场关于区分“存档小组”（archival groups）与“数据囤积社区”（data hoarding communities）的讨论展开，一位成员澄清两者并非同义词。
- **建议利用 Common Crawl**：针对 Web Crawling 实践的咨询，Common Crawl 被推荐为一项资源，但提醒其并不索引 Twitter 内容。
- **新的多语言 LLM 预印本推介**：分享了关于 **Aurora-M** 的新预印本，这是一个 15.5B 参数、持续预训练的开源多语言语言模型，附带 [ArXiv 链接](https://arxiv.org/abs/2404.00399)，并拥有超过 2 万亿个训练 tokens。
- **用于结构化 LLM 输出的工具**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=KxOqjKq2VyY)，展示了一个名为 *Instructor* 的工具，该工具可帮助用户从 GPT-3.5 和 GPT-4 等大语言模型 (LLMs) 中提取 JSON 等结构化数据。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/michael_nielsen/status/1776074579829571995?s=46">Michael Nielsen (@michael_nielsen) 的推文</a>: “Imagineering” 是一个极好的术语</li><li><a href="https://www.youtube.com/watch?v=KxOqjKq2VyY">Instructor, 从 LLMs 生成结构化数据</a>: Instructor 使得从 GPT-3.5, GPT-4, GPT-4-Vision 等大语言模型 (LLMs)（包括开源模型）中可靠地获取 JSON 等结构化数据变得容易...</li><li><a href="https://x.com/__z__9/status/1774965364301971849?s=20">ً ‎ (@__z__9) 的推文</a>: 新预印本！首个经过红队测试、开源且持续预训练的多语言 LLM —— **Aurora-M**，符合白宫关于安全、可靠和值得信赖的开发的行政命令...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: 首个根据美国行政命令进行红队测试的开源多语言语言模型</a>: 预训练语言模型支撑着多种 AI 应用，但其高昂的训练计算成本限制了可访问性。BLOOM 和 StarCoder 等倡议旨在使访问民主化...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1225436981085995098)** (10 messages🔥): 

- **C4AI Command R+:** [CohereForAI 在 Twitter 上](https://x.com/cohereforai/status/1775878631715217522?s=46&t=stOPrwZiN_fxSK0RuC8Flg)发布了一个具有 **RAG** 功能的新型 **104B LLM**，提供开源权重、工具链以及对 10 种语言的多语言支持。该版本已在 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 上线，是其之前 35B 模型的进阶版。

- **GPT-4 微调定价实验：** OpenAI 启动了一个实验性项目，以了解 GPT-4 微调的质量、安全性和使用情况，并为[项目期间提供了特定费率](https://openai.com/gpt-4-ft-experimental-pricing)。

- **等待有前景的 AI 进展更新**：讨论了 [@rohanpaul_ai 的一条推文](https://twitter.com/rohanpaul_ai/status/1775923372242726995)中提到的一个有前景的 AI 进展，并观察到在最初宣布三个月后仍未有新信息浮出水面。

- **具有前所未有上下文长度的 LLaMA-2-7B**：[@PY_Z001 的帖子](https://x.com/PY_Z001/status/1776176932687892796)分享了一项 AI 训练成就，声称在 8 张 A100 GPU 上训练了 LLaMA-2-7B，上下文长度高达 700K tokens。

- **关于 fp8 可用性的不确定性**：一位成员对 Nvidia 4090 GPU 上 fp8 的可用性表示不确定，并指出该主题缺乏明确的信息。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/PY_Z001/status/1776176932687892796">Zhang Peiyuan (@PY_Z001) 的推文</a>: 🌟8 张 GPU 实现 700K 上下文🌟 你认为在训练期间，使用 8 张 A100，一个 7B Transformer 的单个上下文中可以放入多少个 token？32K？64K？200K？不，亲爱的朋友。我刚刚成功训练了...</li><li><a href="https://openai.com/gpt-4-ft-experimental-pricing">GPT-4 微调</a>: 未找到描述</li><li><a href="https://x.com/cohereforai/status/1775878631715217522?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Cohere For AI (@CohereForAI) 的推文</a>: 宣布 C4AI Command R+ 开源权重，这是一个具有 RAG、工具链和 10 种语言多语言能力的尖端 104B LLM。此版本基于我们的 35B 模型构建，是我们致力于实现 AI 突破的一部分...
</li>
</ul>

</div>
  

---

**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1225373436248068156)** (182 条消息🔥🔥): 

- **用于数据集转换的 Augmentoolkit**：分享了一个名为 [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) 的新 GitHub 项目，它提供了一种将计算资源和书籍转换为 instruct-tuning 数据集的方法，且无需依赖 OpenAI。

- **展示创新语言模型**：分享的数据集如 [Severian/Internal-Knowledge-Map](https://huggingface.co/datasets/Severian/Internal-Knowledge-Map) 旨在通过结构化的 "System" 指南和详细的叙述来革新语言模型的理解能力。同时，利用广泛科学数据集的 ANIMA 模型作为一个复杂的科学助手，专注于仿生学等领域。

- **结合 Lambda Calculus 的推理 AI**：[neurallambda](https://github.com/neurallambda/neurallambda) 项目探索将 lambda calculus 与 transformers 集成，旨在赋予 AI 推理能力。

- **Command R+ 发布**：Command R+ 已推出，这是一款专门针对企业用例的强大大语言模型 (LLM)，带来了检索增强生成 (RAG) 和多语言支持等高级功能。它首先在 Microsoft Azure 上可用，并在 Hugging Face 上的 [CohereForAI/c4ai-command-r-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 路径下提供了权重。

- **关于模型剪枝和微调的讨论**：成员们讨论了针对 Jamba 等模型的剪枝策略，引用了一篇关于层剪枝（layer-pruning）策略的论文，以及在性能下降极小的情况下对问答基准测试（benchmarks）的影响。该策略与量化（quantization）和 Low Rank Adapters (QLoRA) 等参数高效微调（parameter-efficient finetuning）方法相一致。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">介绍 Command R+：为业务构建的可扩展 LLM</a>：Command R+ 是一款最先进的 RAG 优化模型，旨在处理企业级工作负载，并率先在 Microsoft Azure 上推出。今天，我们推出 Command R+，我们最强大的...</li><li><a href="https://arxiv.org/abs/2403.17887">深层网络不合理的低效性</a>：我们对流行的开源权重预训练 LLM 系列进行了一项简单的层剪枝策略实证研究，发现在不同的问答基准测试中，性能退化极小，直到...</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.02684">线性成本推理 Transformer 的跨架构迁移学习</a>：最近，提出了多种架构，通过改变 Self-Attention 模块的设计来实现线性成本推理，从而提高 Transformer 语言模型的效率...</li><li><a href="https://tenor.com/view/watching-gif-8300112311601598690">Watching GIF - Watching - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.02258v1">Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算资源</a>：基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 反而可以学习动态地将 FLOPs（或计算量）分配给特定的...</li><li><a href="https://arxiv.org/html/2404.02258v1">Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算资源</a>：未找到描述</li><li><a href="https://huggingface.co/learn">Hugging Face - 学习</a>：未找到描述</li><li><a href="https://course.fast.ai/">程序员实用深度学习 - 实用深度学习</a>：一门为有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的学习者设计的免费课程。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd">axolotl/docs/rlhf.qmd at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized">HuggingFaceH4/ultrafeedback_binarized · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences?row=1">argilla/ultrafeedback-binarized-preferences · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: 语言模型的表示微调</a>：ReFT：语言模型的表示微调 - stanfordnlp/pyreft</li><li><a href="https://github.com/architsharma97/dpo-rlaif/tree/main">GitHub - architsharma97/dpo-rlaif</a>：通过在 GitHub 上创建账号，为 architsharma97/dpo-rlaif 的开发做出贡献。</li><li><a href="https://github.com/neurallambda/neurallambda">GitHub - neurallambda/neurallambda：推理计算机。Lambda 演算，全可微。还包括 Neural Stacks、Queues、Arrays、Lists、Trees 和 Latches。</a>：推理计算机。Lambda 演算，全可微。还包括 Neural Stacks、Queues、Arrays、Lists、Trees 和 Latches。 - neurallambda/neurallambda</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit：将计算资源和书籍转换为 Instruct-Tuning 数据集</a>：将计算资源和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit</li><li><a href="https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi">神经网络究竟是什么？ | 第一章，深度学习</a>：什么是神经元，为什么要分层，其背后的数学原理是什么？资助未来的项目：https://www.patreon.com/3blue1brown 编写/交互...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M&t=430s&pp=ygULM2JsdWUxYnJvd24%3D">GPT 究竟是什么？ Transformer 的视觉入门 | 第五章，深度学习</a>：Transformer 及其先决条件介绍。赞助者可提前观看下一章节：https://3b1b.co/early-attention 特别感谢这些支持者...</li><li><a href="https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab">向量 | 第一章，线性代数的本质</a>：从基础开始线性代数系列。资助未来的项目：https://www.patreon.com/3blue1brown 另一种同样宝贵的支持方式是...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1225342007296266331)** (48 messages🔥): 

- **向 Repo 添加 Function Calling 示例**：一个 Pull Request 即将开启，旨在向 Hermes-Function-calling 仓库的 examples 文件夹中添加一个演示 **Function Calling** 的示例 Notebook。

- **使用 Vercel AI SDK RSC 进行 Function Calling 时面临的挑战**：Fullstack6209 花了一整天时间排查为什么 **Hermes Pro** 和其他 LLM 无法像 ChatGPT-3.5 那样与 Vercel AI SDK RSC 配合使用。通过探索不同的 Repo 发现，返回的 JSON 被错误地识别为文本补全（Text Completion），而它本应以流式传输（Stream）Function Calls。

- **关于 Function Calling 和编码标准的讨论**：讨论了 NousResearch **Hermes-Function-Calling** 仓库中关于编码标准和 `convert_to_openai_tool` 正常工作所需的正确文档风格（Documentation Styles）的问题。该仓库已根据 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 进行了更新。

- **Eurus-7B-KTO 模型引起关注**：**Eurus-7B-KTO** 模型因其出色的表现而受到关注。一名成员链接了其在 HuggingFace 上的页面，并建议将其整合到他们的 **SOLAR** 框架中。

- **探索本地微调方法**：成员们讨论了本地微调（Fine-tuning）的最佳方法，提到了 **Axolotl 配合 QLoRA** 等工具作为选项，同时也有人讨论了微调更大模型（如 **Llama-2 70B** 和 **Qwen 72B**）时的性能一致性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/issues/14">This Repo needs some refactoring for the function calling to work properly · Issue #14 · NousResearch/Hermes-Function-Calling</a>：伙计们，我认为目前这个仓库的实现方式存在一些问题，其中最大的是关于编码标准，目前你们使用了来自 LangChain 的 convert_to_openai_tool...</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>：快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/commit/3171de7d8b70242b980ed042f958dd9044ea2197">updating docstring to match google python style guide · NousResearch/Hermes-Function-Calling@3171de7</a>：未找到描述</li><li><a href="https://google.github.io/styleguide/pyguide.html">styleguide</a>：Google 发起的开源项目的风格指南</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/function_calling.py">langchain/libs/core/langchain_core/utils/function_calling.py at master · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1225392033783615610)** (2 messages): 

- **检测到依赖问题**：一名成员指出，某个项目或安装似乎缺少 **Dependencies**（依赖项）。另一名成员以简单的 **"i see"**（我明白了）回应了这一观察。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1225363602865917992)** (31 messages🔥): 

- **值得置顶的规划总结**：大家一致同意为新加入者置顶总结。一份包含目标和要求的文档已经创建并开始更新，但由于贡献者的时间精力有限，目前尚未进行大规模推广。

- **采用 Adaptive RAG**：结合了查询分析（Query Analysis）和迭代式答案构建的 Adaptive-RAG 方法已使用 *LangGraph* 以及 *Cohere* 的 Command-R 和 Command R+ 模型实现。[此实现](https://youtu.be/04ighIjMcAI)展示了使用 LangGraph 与 ReAct Agent 之间的区别，以及使用 Command-R 与 Command R+ 的优缺点。

- **RAG 与 UX 创新**：成员们讨论了 RAG 的实际应用和成功案例，特别是在源代码检索和检索后过滤方面。一个提议的 UI 概念涉及维护一个实体和 Artifacts 的向量数据库，以简化用户交互过程。

- **探索 RAG 的检索数据源**：关于检索数据来源的建议包括：从 Wikipedia 索引开始、集成实际应用的代代码、考虑合成教科书，以及添加特定领域的数据库（如 *Caselaw Access Project*）。强调了数据源多样性是理想的状态。

- **Command R+ 和 Claude Opus 更新**：分享了关于 *Command R+* 指令格式的讨论，并指出 Claude Opus 在处理复杂查询时表现出色。强调了正确 Prompting 和引用来源的重要性，并参考了 Cohere 的平台和文档。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LangChainAI/status/1775917799065653250?s=20">来自 LangChain (@LangChainAI) 的推文</a>：使用 Cohere 新发布的 Command-R+ 实现 Adaptive RAG。Adaptive-RAG (@SoyeongJeong97 等人) 是最近的一篇论文，它结合了 (1) 查询分析和 (2) 迭代式答案构建，以无缝处理查询...</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">检索增强生成 (RAG) - Cohere 文档</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI 可接受使用政策</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/TeraflopAI/Caselaw_Access_Project">TeraflopAI/Caselaw_Access_Project · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/长文本推理数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1225440855159079075)** (108 messages🔥🔥): 

- **WorldSim 更新和命令集**：提供了 [WorldSim Versions & Command Sets](https://pastebin.com/raw/P6UQWUaB) 的链接，以及包含近期新增内容的 WorldSim [Command Index](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4) 更新。 
- **WorldSim 的共时性（Synchronicity）和自定义表情符号**：用户讨论了与观看《玲音》（Serial Experiments Lain）相关的共时性事件，以及缺乏 WorldSim 专用自定义表情符号的问题。有人建议《玲音》中的 "Wired" 符号或“眼睛”图案可能适合 WorldSim。
- **哲学和研究的潜在新频道**：关于是创建一个新的“philosophy”频道，还是使用现有的“interesting-links”频道分享相关内容的辩论。一些用户建议将 AI 驱动的思想导图与 Obsidian 等工具集成，以管理复杂的想法。
- **TRIVERS-80 和远程临场感（Telepresence）的生动性**：一位用户正在开发一个使用 Python 创建 TRS-80 体验的原型，并讨论了远程临场感在不同生动度和交互性媒介中的重要性，以及齐普夫定律（Zipf's law）在通信系统混沌边缘可能发挥的作用。
- **即将推出的 WorldSim 界面和自引导（Self-Steering）更新**：人们对即将到来的 WorldSim 重大更新充满期待，提到了一些改进功能，例如消除自引导（Self-Steering，即模型在没有用户输入的情况下自行运行）。此外，还分享了一个用于与 Claude 等模型交互的开源 [UX Library](https://github.com/jquesnelle/crt-terminal)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队提供的一体化工作区。</li><li><a href="https://www.amazon.fr/Erosophia-English-Jason-Reza-Jorjani-ebook/dp/B0CSMGR2L5">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/feel-me-think-about-it-meme-gif-7715402">Feel Me Think About It GIF - Feel Me Think About It Meme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/jquesnelle/crt-terminal">GitHub - jquesnelle/crt-terminal: 复古风格的终端 Shell</a>：复古风格的终端 Shell。通过在 GitHub 上创建账户来为 jquesnelle/crt-terminal 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1225343798117404673)** (189 messages🔥🔥):

- **探索张量操作**：讨论了带有融合算子（fused kernels）的 [GaLore 内存高效训练](https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362) 的更新，重点关注其对 GPU 显存的益处。表达了将 GaLore 与 Unsloth AI 集成的兴趣。
- **理解模型打包与参数**：关于 `packing` 参数的查询显示，它通过连接多个分词后的序列来实现更快的训练。然而，由于兼容性问题，建议不要在 Gemma 模型中使用 packing。
- **AI 算法的优化协同**：用户探索了 Unsloth 和 GaLore 之间的协同作用，讨论了在显存减少和速度提升方面的潜力，尽管 GaLore 默认性能比 Lora 慢。
- **Unsloth AI 即将发布的版本与特性**：Unsloth AI 的近期计划包括为“GPU 贫困户”提供的一项新开源特性，将于 4 月 22 日发布公告，并于 5 月初发布可集成多种模型的“自动优化器（Automatic optimizer）”。讨论了 Unsloth Pro 产品，指出其自 2023 年 11 月起已可用，重点在于分发方面的挑战。
- **合成数据的数据集格式灵活性**：用户交流了用于微调 LLM 的合成数据集生成的想法，结论是格式选择不会显著影响性能，可以根据偏好采用多种格式。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4#scrollTo=OE_R3AXG5Y-F">Google Colaboratory</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.17919">LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning</a>：自大语言模型（LLMs）首次出现以来，机器学习社区见证了令人印象深刻的进展，然而其巨大的显存消耗已成为大型模型的主要障碍……</li><li><a href="https://huggingface.co/posts/mlabonne/730068367902681">Hugging Face 上的 @mlabonne：“⚡ AutoQuant AutoQuant 是我之前 AutoGGUF notebook 的进化版……”</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/15g">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>：未找到描述</li><li><a href="https://github.com/myshell-ai/JetMoE">GitHub - myshell-ai/JetMoE: 以 10 万美元达到 LLaMA2 的性能</a>：以 10 万美元达到 LLaMA2 的性能。通过在 GitHub 上创建账户为 myshell-ai/JetMoE 的开发做出贡献。</li><li><a href="https://arstechnica.com/security/2024/03/researchers-use-ascii-art-to-elicit-harmful-responses-from-5-major-ai-chatbots/">ASCII 艺术诱导 5 个主要 AI 聊天机器人产生有害响应</a>：LLM 经过训练以阻止有害响应。老式的图像可以绕过这些规则。</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: 线性注意力序列并行 (LASP)</a>：线性注意力序列并行 (LASP)。通过在 GitHub 上创建账户为 OpenNLPLab/LASP 的开发做出贡献。</li><li><a href="https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362">jeromeku 提交的 GaLore 和融合算子原型 · Pull Request #95 · pytorch-labs/ao</a>：原型算子与工具。目前：GaLore 内存高效训练的融合算子初步实现。待办：triton 用于量化训练的可组合 triton 算子以及……
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1225681242373095476)** (21 条消息🔥):

- **Kaggle 倒计时开始**：对话以对 Kaggle 即将到来的重置的兴奋拉开序幕，引发了简短而热烈的回应。
- **夏令时唤起了记忆**：提到 Kaggle 的重置让另一位成员想起了夏令时调整，并澄清了即将到来的变化：*即凌晨 3 点变成凌晨 2 点*。
- **期待额外的睡眠**：关于夏令时额外增加一小时的幽默调侃，一位期待多睡一会而的成员对此表示欢迎，说道：*多睡一小时。*
- **寻找 AI 新闻来源**：一位成员发起了关于最喜欢的 AI 新闻来源的讨论，建议范围从名为 *AI News* 的 Newsletter 到 Reddit AI 社区，并特别提到了用户 localllama。
- **对训练数据源的好奇**：聊天涉及了用于预训练当前 AI 模型的模型数据集范围，思考是否包含了 libgen 和 scihub 等资源，一位参与成员认为它们很可能是某些模型预训练材料的一部分。
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1225357126621204522)** (137 messages🔥🔥): 

- **Unsloth 助力流畅推理**：成员们报告了成功将 Unsloth 用于 *inference*，并指出了其速度和易用性。对于更高级的推理选项，[starsupernova](https://github.com/unslothai/unsloth/issues/267#issuecomment-2034047189) 提供了一个 GitHub 链接，解释了 batch inference 并分享了一段用于从 prompt 列表生成多个模型输出的 Python 代码片段。
  
- **vLLM 中的量化查询展开**：一位成员寻求使用 `vLLM` 将其 13B 模型从 4 bit 量化到 16 bit 以减少 VRAM 占用，并询问是否需要再次量化，这引发了关于 **VRAM 减少方法** 的讨论。Starsupernova 解释说 vLLM 已经拥有诸如 **AWQ** 之类的量化方法，并且正在考虑为 AWQ quants 添加一种快速方法，但目前尚不支持。

- **为教育提供微调便利**：用户讨论了使用 Unsloth 组织 finetuning 工作坊的逻辑，思考如何在时间限制内提供动手实践经验。他们探索了多种方法，从像烹饪节目一样提前准备模型，到使用 [LoRaX](https://github.com/oobabooga/text-generation-webui)（一个加载 finetuned 模型的推理服务器）来代替直接提供权重访问。

- **模型和 Adapter 加载逻辑**：成员们交流了在项目中加载 finetuned 模型和 adapters 的技术。一位成员提到在推理期间使用 left padding，在训练期间使用 right padding，这得到了确认，即这是使用 adapters 时的推荐方法。

- **解决模型转换后的拼写错误**：一位成员在将其 **7B Mistral base** 模型转换为 GGUF 5_K_M 格式后生成文本时遇到了拼写错误，尽管在 4-bit Unsloth 形式下没有问题。通过社区对话，澄清了该问题可能与模型转换过程有关，而非推理参数，并且确认在用户自己的设备上进行 CPU 推理是可行的。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tts.hircoir.eu.org/">HirLab - Convertidor de Texto a Voz por Hircoir</a>: HirLab 是一个基于人工智能的文本转语音平台。它能快速准确地将文本转换为语音。</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4#scrollTo=OE_R3AXG5Y-F">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/generation_strategies">Text generation strategies</a>: 未找到描述</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.</a>: 一个用于 Large Language Models 的 Gradio web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。- oobabooga/text-generation-webui</li><li><a href="https://github.com/unslothai/unsloth/issues/267#issuecomment-2034047189">Batch inference produces nonsense results for unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Issue #267 · unslothai/unsloth</a>: 你好，在使用以下代码加载模型后：from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = &quot;unsloth/mistral-7b-instruct-v0.2-bnb...</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: llama.cpp 的 Python 绑定。可以通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1225662791483129896)** (35 messages🔥): 

- **微调方法的效率飞跃**：一种**新的微调方法** ReFT (Representation Finetuning) 受到关注，其参数效率比以往方法高出 10x-50x。实现和训练流水线已准备好进行验证，可通过 [GitHub 仓库](https://github.com/stanfordnlp/pyreft)和[配套论文](https://arxiv.org/abs/2404.03592)获取。

- **呼吁在 Unsloth 中实行严格的版本控制**：一名成员建议为 Unsloth AI 实施更严格的版本控制，以避免从 nightly 分支合并到 main 分支时出现数值结果差异。他们遇到了合并对 Mistral 微调产生负面影响的问题，强调了能够回滚到以前版本的重要性。

- **随机种子问题受到关注**：关于版本控制的讨论，另一位成员指出数值结果的问题可能也与 random seeds 的 accelerate 问题有关，并认同设立 nightly 与非 nightly 分支的实用性。

- **承诺通过版本控制辅助可复现性**：针对这些担忧，官方承认需要更好的版本控制，并承诺不再草率地向 main 分支推送更新。目标是帮助用户更轻松地跟踪更改并确保性能的一致性。

- **Unsloth 的增强功能影响模型可复现性**：一位成员讨论了 Unsloth 代码优化对模型可复现性的影响，建议将更改作为独立版本发布以解决此问题。这种做法有助于精准定位可能无意中破坏模型或影响可复现性的更改。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1776057023697731913?s=46&t=a0DY32ITg_76H4z5wTVIyw">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: ReFT: Language Models 的 Representation Finetuning，参数效率比之前的 state-of-the-art 参数高效微调方法高出 10x-50x。仓库地址: https://github.com/stanfordnlp/pyreft a...</li><li><a href="https://github.com/unslothai/unsloth/tags">Tags · unslothai/unsloth</a>: 快 2-5 倍，显存占用减少 70%，支持 QLoRA &amp; LoRA 微调 - Tags · unslothai/unsloth
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1225360425999142962)** (67 messages🔥🔥):

- **数据集困境**：一位成员正在寻找 Wikitext-2 和 Wikitext-103 数据集，并讨论了以原始形式访问和使用数据的困难与细微差别。分享了指向托管在 [Stephen Merity 的页面](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR) 和 [Hugging Face](https://huggingface.co/datasets/wikitext/tree/main/wikitext-103-raw-v1) 上的原始数据集的直接链接。
- **GateLoop 取得进展**：GateLoop 架构的 perplexity 评分在 'lucidrains' 尝试复制失败后引发了一些质疑。尽管如此，GateLoop 的作者随后发布了 [一些代码](https://github.com/tobiaskatsch/GatedLinearRNN)，引发了关于其参与度和命名惯例的进一步讨论。
- **对 Parquet 转换的担忧**：成员们对在 Hugging Face 平台上使用时自动转换为 parquet 格式表示了不满。对话集中在数据可复现性的重要性以及对用于归档的原始数据格式的偏好。
- **对 Attention 变体的思考**：讨论涉及了像 GateLoop 这样新的 Attention 机制由于现有的对 RWKV 和 Mamba 等成熟方法的偏好而难以普及的问题。成员们还表达了在没有广泛实验证据的情况下，新贡献难以获得关注的挑战。
- **小数据集训练技巧**：在简短的插话中，一位成员建议在处理小数据集时要注意 weight decay 和 dropout。这一技术建议旨在帮助那些从事模型 finetuning 的人员。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR">Smerity.com: The WikiText Long Term Dependency Language Modeling Dataset (2016)</a>：未找到描述</li><li><a href="https://x.com/qinzytech/status/1775916338822709755?s=20))">Zengyi Qin (@qinzytech) 的推文</a>：训练 LLM 可能比以前想象的要便宜得多。10 万美元足以训练 LLaMA2 级别的 LLM🤯 虽然 @OpenAI 和 @Meta 使用数十亿美元来训练他们的模型，你也可以...</li><li><a href="https://x.com/cosmo_shirley/status/1770892654114398497?s=20">Shirley Ho (@cosmo_shirley) 的推文</a>：你们都听说过 ChatGPT 或 foundation models，但想用 AI 构建不仅仅是聊天机器人的东西吗？🔥 我们 @PolymathicAI 正在构建科学领域的 foundation models 🔥 加入我们 (@albertobietti @...</li><li><a href="https://huggingface.co/datasets/wikitext/tree/main/wikitext-103-raw-v1">wikitext at main</a>：未找到描述</li><li><a href="https://github.com/lucidrains/gateloop-transformer">GitHub - lucidrains/gateloop-transformer: Implementation of GateLoop Transformer in Pytorch and Jax</a>：在 Pytorch 和 Jax 中实现 GateLoop Transformer - lucidrains/gateloop-transformer</li><li><a href="https://github.com/tobiaskatsch/GatedLinearRNN">GitHub - tobiaskatsch/GatedLinearRNN</a>：通过在 GitHub 上创建账号来为 tobiaskatsch/GatedLinearRNN 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/segyges/wikitext-103/tree/main">segyges/wikitext-103 at main</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1225353037447430184)** (207 条消息🔥🔥): 

- **探索模块化 LLM 与 MoE 专业化**：讨论围绕 Mixture of Experts (MoE) 架构是否通过在 Large Language Models (LLMs) 中培养专家特定的专业化来固有地支持模型可解释性展开。提到了一篇关于在 LLM 中分离语言与思维的论文 ([链接](https://arxiv.org/abs/2301.06627)) 以及各种 MoE 路由技术，如 Expert Choice Routing，这些技术可能以情境依赖的方式引导 MoE 门控机制 ([OpenMoE 链接](https://arxiv.org/pdf/2402.01739.pdf), [Expert Choice Routing 链接](https://arxiv.org/pdf/2202.09368.pdf))。

- **关于层级式 MoE 优势的辩论**：随后展开了关于使用层级式 MoE 结构与扁平式 MoE 相比的优势的辩论。讨论包括技术见解，如具有 product-key 结构的 router weights 以及 Compression-Selection 函数，并声称层级结构可以提高专家选择的特异性，尽管有人担心与扁平式 MoE 相比会降低表达能力。

- **深入探讨模型训练细节**：分享了关于特定架构的技术细节，如嵌套 MoE 与扁平式 MoE，以及包括 learning rates 在内的 hyperparameter tuning。一次对比展示了两个具有固定种子和配置的相似模型，引发了关于新架构方法中 hyperparameter 优化重要性的讨论。

- **MoE 模型效率的潜在突破**：一位成员神秘地透露了其硕士论文的内容，暗示 MoE 模型的浮点运算（FLOPs）曲线将发生重大转变，这意味着 LLM 训练的计算效率将得到实质性提升。该成员提到将在大约 1.5 个月内发布相关论文，并表示欢迎联系合作。

- **对 "Schedule-Free" 优化的怀疑反应**：关于 "Schedule-Free" 学习优化器的公告声称，无需调度或调优，仅使用 SGD 或 Adam 即可简化适配（[推文链接](https://x.com/aaron_defazio/status/1776320004465582331?s=46)），这引发了对其宣传的 baselines 以及算法背后实际机制的怀疑。讨论强调了许多人对新优化器方法所提出的乐观主张持谨慎态度。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.03626">Training LLMs over Neurally Compressed Text</a>：在本论文中，我们探讨了在高度压缩的文本上训练大语言模型（LLMs）的想法。虽然标准的 subword tokenizers 只能以较小的倍数压缩文本，但神经文本压缩器可以……</li><li><a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46">Aaron Defazio (@aaron_defazio) 的推文</a>：Schedule-Free Learning https://github.com/facebookresearch/schedule_free 我们现在已经开源了我那一系列神秘图表背后的算法。每张图表要么是 Schedule-free SGD，要么是 Adam，没有……</li><li><a href="https://arxiv.org/abs/2010.03881">Large Product Key Memory for Pretrained Language Models</a>：Lample 等人 (2019) 提出的 Product Key Memory (PKM) 能够通过以微不足道的计算开销有效地增加模型容量来提高预测准确性。然而，他们的经验……</li><li><a href="https://arxiv.org/abs/1907.05242">Large Memory Layers with Product Keys</a>：本文介绍了一种可以轻松集成到神经网络中的结构化内存。该内存设计非常庞大，显著增加了架构的容量，最高可达……</li><li><a href="https://d197for5662m48.cloudfront.net/documents/p">Open Research Collaboration and Publishing - Authorea</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1225716116861489152)** (3 条消息): 

- **PDF 提醒**：一位成员分享了一个 [研究论文链接](https://arxiv.org/pdf/2404.01367.pdf)，但没有提供任何背景信息或对内容的评论。
- **Google 的力量**：同一位成员随后强调了使用 Google 的重要性，大概是为了进一步研究或澄清，简单地表示：*始终使用 Google (Always google)*。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1225489230965379193)** (6 条消息): 

- **AtP* 论文实现咨询**：一位成员询问是否有最新的 AtP* 论文的开源实现或相关的 notebook。
  
- **AtP* GitHub 仓库分享**：针对关于 AtP* 论文实现的咨询，另一位成员分享了 GitHub 仓库：[GitHub - koayon/atp_star](https://github.com/koayon/atp_star)，这是一个 AtP* (Kramar et al 2024, DeepMind) 的 PyTorch 和 NNsight 实现。
  
- **请求 GitHub Stars**：**David Bau** 分享的一条消息呼吁通过给 nnsight GitHub 仓库打星来支持，以满足 NSF 评审员的要求。该仓库地址为：[GitHub - ndif-team/nnsight](https://github.com/ndif-team/nnsight)，用于解释和操作深度学习模型的内部机制。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/koayon/atp_star">GitHub - koayon/atp_star: AtP* 的 PyTorch 和 NNsight 实现 (Kramar et al 2024, DeepMind)</a>：AtP* 的 PyTorch 和 NNsight 实现 (Kramar et al 2024, DeepMind) - koayon/atp_star</li><li><a href="https://github.com/ndif-team/nnsight">GitHub - ndif-team/nnsight: nnsight 包能够解释和操作深度学习模型的内部机制。</a>：nnsight 包能够解释和操作深度学习模型的内部机制。 - ndif-team/nnsight
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1225404570465730560)** (39 条消息🔥): 

- **`top_p` 参数故障排除**：一位成员遇到了一个问题，尽管格式正确，但脚本配置中无法识别 `top_p=1`。他们后来发现消除空格解决了问题，证实是语法错误导致参数无法被识别。

- **关于 BIG-bench 任务的重大疑问**：一位成员反映 BIG-bench (`bigbench`) 任务似乎无法被识别，引发了关于任务正确命名和使用的讨论。建议使用 `lm_eval —tasks list` 来获取所有正确任务名称的列表。

- **自动 Batch Size 带来的巨大速度提升**：一位成员通过设置 `batch_size=auto`，将评估时间从 20 分钟大幅缩短至 3 分钟，这表明 GPU 利用不足会显著影响性能。

- **模型参数兼容性困惑**：关于 `openai-completions` 的模型参数是否兼容存在困惑，有成员收到了错误消息。似乎存在潜在的 Bug 或误解，因为能找到 `openai-chat-completions` 但找不到 `openai-completions`。

- **`--predict_only` 标志的错误及重新安装咨询**：一位成员在 Mac 本地运行 `--predict_only` 时遇到问题，收到参数无法识别的错误。建议尝试重新安装，或者在 Google Colab 上尝试以复现并排查问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8uSmUm806LQtG8v#scrollTo=mTSKBJlVjaB-">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8u">Google Colaboratory</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1225427073808732250)** (1 条消息): 

- **Gemini 的云端支持**：一条消息提到 **AWS** 去年发布了名为 **Gemini** 的产品，暗示该服务已获得云端支持。据称 **Azure** 也提供了相关支持。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1225435389968384050)** (18 条消息🔥): 

- **探索 Mojo 的团队规模**：有人询问在 Modular 工作的人员数量。
- **在其他编辑器上进行调试**：讨论了针对 neovim 等编辑器的调试器和 Language Server Protocol (LSP) 的可用性，成员们就遇到的问题寻求解决方案。
- **寻求复杂问题的指导**：成员们分享了常见问题的解决方案，并提供了以往讨论的链接，例如[此处发现](https://discord.com/channels/1087530497313357884/1151418092052815884/1223704253978120327)的一个有用回答。
- **请求明确 Mojo 的路线图**：一位成员表示需要 Mojo 与 Taichi 或 Triton 等其他框架对比的详细路线图，得到的回复指向了 [Mojo 路线图文档](https://docs.modular.com/mojo/roadmap)。
- **Modular 进展的直播教学**：提供了一个关于 Modular 社区直播的提醒和链接，该直播讨论了 MAX 24.2 的新功能，可在 [YouTube](https://www.youtube.com/watch?v=PL71FV2KKHE) 上观看。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 路线图与待完善点 | Modular 文档</a>：关于 Mojo 计划的摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular 社区直播 - MAX 24.2 新特性</a>：MAX 24.2 现已发布！加入我们的直播，讨论 MAX 的所有新变化——开源 Mojo 标准库、MAX Engine 支持等...</li><li><a href="https://github.com/modularml/mojo/discussions/342">我可以在 Mojo 中使用 Pandas 吗？ · modularml/mojo · Discussion #342</a>：我可以在 Mojo 中使用 Pandas 吗？我尝试了：from PythonInterface import Python let pd = Python.import_module("pandas") d = Python.dict() d['col1']=[1, 2] d['col2']=[3, 4] df = pd...</li><li><a href="https://github.com/rust-lang/rustlings">GitHub - rust-lang/rustlings: :crab: 让你习惯阅读和编写 Rust 代码的小练习！</a>：:crab: 让你习惯阅读和编写 Rust 代码的小练习！ - rust-lang/rustlings</li><li><a href="https://codeberg.org/ziglings/exercises/.">exercises</a>：通过修复微小的错误程序来学习 ⚡Zig 编程语言。</li><li><a href="https://github.com/dbusteed/mojolings">GitHub - dbusteed/mojolings: 通过修复小程序学习阅读和编写 Mojo 代码</a>：通过修复小程序学习阅读和编写 Mojo 代码 - dbusteed/mojolings
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1225491449920094210)** (5 条消息):

- **Modular 分享更新**：Modular 账号在 Twitter 上发布了其最新更新的链接 [查看推文](https://twitter.com/Modular/status/1775926484869541894)。
- **宣布新功能**：一条新推文宣布了 Modular 社区新功能的到来，并附带了更多详情链接 [查看功能](https://twitter.com/Modular/status/1775946487186555225)。
- **预告新集成**：Modular 在 Twitter 上预告了一个新集成，暗示了即将推出的功能或合作 [查看预告](https://twitter.com/Modular/status/1776287802533245372)。
- **发布倒计时**：Modular 的后续推文似乎开启了倒计时，可能指向产品发布或活动 [关注倒计时](https://twitter.com/Modular/status/1776287865242300621)。
- **暗示合作**：Modular 的最新推文暗示了合作，表明正在进行合作伙伴关系或合资项目 [探索可能性](https://twitter.com/Modular/status/1776287868710998188)。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1225390314089611324)** (236 条消息🔥🔥): 

- **"拒绝 `isinstance`，拥抱 `Variant` 动态特性"**：成员们讨论了 `isinstance` 函数的局限性，并支持 `Variant` 类型的动态特性。分享了一个来自 [文档](https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo) 的 `Variant` 使用示例，包括其存储内部数据的能力以及使用 `isa` 和 `get/take` 方法进行类型检查。

- **最受欢迎功能愿望清单**：社区对拥有类似于 Swift 和 Rust 中的 **pattern matching** 能力表现出浓厚兴趣，并提出并辩论了 ‘match case’ 语法的构想。此外，“**conditional conformance**” 语法也是一个热门话题，讨论围绕潜在的语法和实现挑战展开。

- **Mojo 在移动端**：Mojo 通过 Termux 在搭载 Snapdragon 处理器的 Android 设备上成功运行，成员们对此可能性感到兴奋。

- **Mojo 与周边结合**：社区对 **Modular 主题周边商品** 的可用性感到好奇，提出了 Mojo 毛绒玩具和手机壳等建议，并认为这些是潜在的未来商品。

- **Mojo 风格与惯用法**：社区正在考虑使用合适的术语来描述地道的 Mojo 代码。提到了 **[风格指南](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md)** 以及 Mojo 与 Python 互换运行的意图，强调了该语言的灵活性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular Docs</a>: 定义了一个 Variant 类型。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">mojo/stdlib/src/utils/variant.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225572475879034950)** (5 条消息): 

- **Basalt 从 Voodoo 的灰烬中诞生**：Voodoo 已被弃用，为 **Basalt** 铺平了道路，这是一个用纯 Mojo 构建的新 Machine Learning 框架。**首个官方版本**为 v.0.1.0，你可以在 [GitHub](https://github.com/basalt-org/basalt) 上阅读更多信息并做出贡献，并在 [Medium](https://medium.com/@bennynottonson/basalt-machine-learning-in-pure-mojo-66cbd6e53a05) 上找到介绍文章。

- **值得更多 Mojo 的社区努力**：在一次更新指出 Basalt 的贡献归属不公后，成员们鼓励社区参与。希望社区能为即将到来的设计工作提供更多人手和智慧。

- **Deep Learning vs Machine Learning**：建议将 **Basalt** 归类为 "Deep Learning" 而非 "Machine Learning"，以便与 PyTorch 等框架更紧密地对齐，并有兴趣看到 Basalt 与快速深度学习框架 [Burn](https://burn.dev) 的性能对比。

- **Mojo 丰富的潜力**：一条简短的评论赞扬了使用 Mojo 开发的创新项目，认可了社区的创造力和技术水平。

- **Specials 包带来高精度**：**Specials 包**的一次更新引入了具有硬件加速的初等数学函数，并且相比于 FLOPS 更注重数值精度。欢迎在 [GitHub](https://github.com/leandrolcampos/specials) 上查看其与 NumPy 和 Mojo 标准库的基准测试对比。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://burn.dev">Burn</a>：未找到描述</li><li><a href="https://github.com/leandrolcampos/specials">GitHub - leandrolcampos/specials: Special functions with hardware acceleration</a>：具有硬件加速的特殊函数。通过在 GitHub 上创建账号来为 leandrolcampos/specials 的开发做出贡献。</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>：一个在纯 Mojo 🔥 中从零开始构建的机器学习框架 - basalt-org/basalt
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1225341514381656165)** (1 条消息): 

- **Bucket 大小估计说明**：一位成员解释说，估计的数值数量决定了 bucket 的数量 d，该值将被向上取整。虽然这可能看起来令人担忧，但被认为是微不足道的，因为每个 bucket 仅包含 **UInt32 值**，仅占用 4 字节；dict 可以通过参数化来调整类型宽度，默认值为 4 字节，以处理高达 **2^32 - 1（约 40 亿）个值**。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1225454314886270977)** (10 条消息🔥): 

- **关于 `__refitem__` 和迭代器的讨论**：对话涉及了一个处理迭代器中引用的潜在折中方案，即保留 `.value()` 并添加 `__refitem__`。这是在讨论迭代器应如何运行的过程中提出的，可能在等待 `StopIteration` 的参数化抛出（parametric raises）。

- **Mojo 中首创的 Python 互操作性**：Mojo 与 Python 的互操作工作已初见成效，实现了 PyMethodDef、PyCFunction_New、PyModule_NewObject 以及为 PythonObject 修改的 init。位于 [rd4com/mojo_branch](https://github.com/rd4com/mojo_branch/tree/nightly) 的仓库展示了这一进展，并强调在这些集成中需要仔细规划。

- **Python 引用计数表现稳定**：最近对 Mojo 的 Python 互操作能力的贡献没有表现出任何引用计数问题，表明当前实现具有稳定性。

- **处理反向 Range 的 Bug**：一位成员发现了一个 Bug，即 `len(range(-10))` 等于 `-10`。在他们研究可逆 Range 和相关迭代器时，他们寻求关于是否在对 Range 处理进行更广泛更新之前引入修复方案的意见。

- **邀请新贡献者加入标准库**：新加入者（如一位渴望为 Mojo 标准库做出贡献的计算机科学专业学生）受到了欢迎，并被引导至 GitHub 上的 good first issues 和贡献指南等起点。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rd4com/mojo_branch/tree/nightly">GitHub - rd4com/mojo_branch at nightly</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 rd4com/mojo_branch 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/changelog#week-of-2023-01-30">Mojo🔥 更新日志 | Modular 文档</a>：Mojo 重大变更的历史记录。</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md">mojo/CONTRIBUTING.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1225463404085383239)** (23 条消息🔥):

- **Linear Attention 进展**：[Linear Attention Sequence Parallelism (LASP)](https://github.com/OpenNLPLab/LASP/tree/main) 库因不需要 flash attn 仓库并改进了 AMD 支持而受到关注，同时它还具备跨设备拆分缓存以处理更长上下文的能力。
- **非量化模型令人惊讶的性能**：一位成员感到有趣的是，在 Hugging Face 上，非量化模型的运行时间比量化模型更长，尽管人们预期量化模型（如 bitsandbytes 的 Hugging Face 实现）的性能会较低。
- **对 C4AI Command R+ 的兴趣**：讨论集中在一个 [104B 参数模型 C4AI Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 上，该模型集成了包括 Retrieval Augmented Generation (RAG) 在内的各种高级功能。讨论强调了该模型的成本和巨大规模，以及由于高计算需求而难以获取此类强大模型的问题。
- **GPT-3 价格讨论**：一位成员对 GPT-3 的定价表示失望，称购买新 GPU 比租赁更具成本效益，连续租赁 GPU 约 125 天即可达到盈亏平衡点。
- **Colab 新 GPU 及价格更新**：一位用户分享了来自 [@danielhanchen](https://x.com/danielhanchen/status/1776074818758348918) 的推文，宣布 Colab 引入了 L4 GPU，价格为 $0.482/小时，并降低了 A100 的价格，[共享表格](https://docs.google.com/spreadsheets/d/1tRbUj8qjsnZdUJOEXJFRLFSOwOWKFaA3hgRC1XWje-w/edit?usp=sharing)中重点列出了更新后的 GPU 价格。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1776074818758348918">Daniel Han (@danielhanchen) 的推文</a>：Colab 有 L4 GPU 了？！而且只要 $0.482/小时！@HCSolakoglu 在 Discord 上告诉了我，我欣喜若狂！原生 fp8，便捷的 Colab 界面 + 24GB VRAM！此外 A100 降价至 $1.177，T4 降至 $0.183。@thechri...</li><li><a href="https://github.com/search">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>：Linear Attention Sequence Parallelism (LASP)。通过在 GitHub 上创建账号为 OpenNLPLab/LASP 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1225650045282684928)** (13 条消息🔥): 

<ul>
  <li><strong>新 LoReFT 展示其精妙之处</strong>：讨论了一种名为 LoReFT 的新技术，它可能优于 LoRA 等现有方法，尽管目前将其合并到基础模型中具有挑战性。分享了相关 <a href="https://twitter.com/harshitj__/status/1776084761875976243">Twitter 帖子</a>的链接。</li>
  <li><strong>GitHub 深入了解 LoReFT</strong>：提到了不稳定的数据集操作，这使得 LoReFT 与现有系统的集成变得复杂。重点展示了相关的 <a href="https://github.com/stanfordnlp/pyreft/blob/main/pyreft/dataset.py#L111">GitHub 代码片段</a>以说明这一担忧。</li>
  <li><strong>通过量化简化 DoRA</strong>：讨论了由于 `peft=0.10.0` 中引入了量化 DoRA 支持，从而删除不必要代码的可能性。提供了 <a href="https://github.com/huggingface/peft/releases/tag/v0.10.0">PEFT 发布说明</a>和特定<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/config/models/input/v0_4_1/__init__.py#L245">代码配置</a>的链接。</li>
  <li><strong>清理 PR 请求</strong>：一位成员被要求提交一个 Pull Request，以清理与量化 DoRA 相关的代码，因为最新的 PEFT 版本已经支持该功能。</li>
  <li><strong>引入 Schedule-free Learning</strong>：讨论了 Facebook Research 发布的 Schedule-free 算法，该算法用平均和插值取代了优化器动量，从而无需传统的 Learning Rate Schedules。强调了 <a href="https://github.com/facebookresearch/schedule_free">GitHub 仓库</a>中的正确使用说明。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46&t=hIokEbug9Pr72tQFuXVULA">来自 Aaron Defazio (@aaron_defazio) 的推文</a>：Schedule-Free Learning https://github.com/facebookresearch/schedule_free 我们现在已经开源了我那一系列神秘图表背后的算法。每张图表要么是 Schedule-free SGD，要么是 Adam，没有...</li><li><a href="https://github.com/stanfordnlp/pyreft/blob/main/pyreft/dataset.py#L111">stanfordnlp/pyreft 的 main 分支：pyreft/pyreft/dataset.py</a>：ReFT：语言模型的 Representation Finetuning - stanfordnlp/pyreft</li><li><a href="https://github.com/stanfordnlp/pyreft/tree/main/examples/loreft">stanfordnlp/pyreft 的 main 分支：pyreft/examples/loreft</a>：ReFT：语言模型的 Representation Finetuning - stanfordnlp/pyreft</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/config/models/input/v0_4_1/__init__.py#L245>">OpenAccess-AI-Collective/axolotl 的 main 分支：axolotl/src/axolotl/utils/config/models/input/v0_4_1/__init__.py</a>：尽管提出 axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  
---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1225494889916010526)** (12 条消息🔥): 

- **寻求模型部署 UI 和专家反馈**：一位成员询问是否有好的用户界面，允许模型部署并具有获取专家反馈的功能。
- **探索非指令数据微调**：一位成员讨论了使用非指令文本数据（如播客转录稿）微调 **Mistral-7B** 等模型，以潜在地生成该数据风格的文本。
- **领域特定微调策略**：在关于领域特定微调的对话中，建议从 `completion` 开始，然后转向指令（instructions），如果经过 Supervised Fine Tuning (SFT) 和 Diverse Prompt Optimization (DPO) 后仍有改进空间，则考虑 Continual Pre Training (CPT)。
- **微调指令的质量重于数量**：一位成员分享了他们的经验，即少量高质量、多样化且经过专家标记的指令样本，比大量重复、简短且有噪声的指令数据能产生更好的性能。
- **微调模型生成过程中的错误**：一位成员在使用以 fp16 微调的 **Mistral 7B** 基础模型进行生成时，在几次成功生成后遇到了错误，寻求解决 Gradio 环境中频繁出现的 `Empty _queue.Empty` 错误的建议。

**提到的链接**：<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、闲逛并保持紧密联系。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1225511933306736722)** (6 条消息): 

- **为 Mistral 7B 训练选择数据集**：一位成员询问了适合训练 **Mistral 7B 模型** 的数据集。对于通用用例，建议的数据集是 **OpenOrca dataset**。
- **首次模型训练咨询**：另一位成员表达了他们训练或微调第一个模型的意图，特别是 **Mistral 7B 模型**。

**提到的链接**：<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、闲逛并保持紧密联系。

---


**OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1225531833970856049)** (1 条消息): 

- **新的 Discord Bot 集成已上线**：OpenAccess AI Collective 现在推出了旨在直接回答问题的 Discord Bot 集成。鼓励成员在指定频道进行测试并提供反馈。[在此测试 Bot](https://discord.com/channels/1104757954588196865/1225300056442409040)。
  



---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1225424306692034714)** (140 条消息🔥🔥):

- **Axolotl 多节点 Docker 问题**：一位用户在使用 Docker 进行 Axolotl 多节点微调时遇到了问题。尽管 Phorm 提供了建议，但用户报告称其并未按预期工作。
- **Checkpoint 加载查询**：当被问及从 Checkpoint 加载时是否需要更改基础模型路径时，Phorm 指出 Axolotl 的行为是不需要更改基础模型路径，因为 Checkpoint 文件已包含必要的模型架构和权重详情。
- **S2 Attention 机制详解**：针对关于 S2 attention 的查询，Phorm 将其描述为一种利用结构化状态空间注意力（structured state space attention）的注意力机制，旨在实现更有效的计算和更好的长程依赖处理。
- **模型内存消耗关注**：用户讨论了 [Galorian 内存过度消耗](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1448) 的问题，Phorm 建议了最小化内存使用的策略，例如减小 batch size 或实施梯度累积（gradient accumulation）。
- **Ubuntu 设置 Axolotl 的命令请求**：一位用户请求一个在 Ubuntu 中设置 Axolotl 的全包命令，对此 Phorm 提供了一个详细的命令，其中包括 Python、Axolotl 以及其他依赖项的安装。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区交流的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://errors.pydantic.dev/2.6/v/value_error">正在重定向...</a>: 未找到描述</li><li><a href="https://tenor.com/view/patrickpain-patricksomuchpain-patrickfleas-spongebobpain-spongebobsomuchpain-gif-18151897">Patrickpain Patricksomuchpain GIF - Patrickpain Patricksomuchpain Patrickfleas - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1448">Galore Axolotl 上的 OOM · Issue #1448 · OpenAccess-AI-Collective/axolotl</a>: 请检查此问题之前是否已被报告过。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：应该在不发生 OOM 的情况下开始训练，就像 Llama facto...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl: 尽管提问（axolotl questions）</a>: 尽管提问。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://storage.googleapis.com/jax-releases/libtpu_releases.html">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: 尽管提问（axolotl questions）</a>: 尽管提问。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=f8d0cb5a-e9cd-4dcf-a16f-39197690a56b)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=a31dec35-31c9-4260-bc7f-1d79610360aa)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=97d0bdd0-f105-43c9-8fe5-e4f8623421ad)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=1608c74f-8ed6-4f25-8861-c69c9ff61737)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=06b61846-011a-43e8-a1d1-6ceaa353a8ec)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=780df5a0-22ab-408f-bcbb-91ab274637f4)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8ff1c0ef-8f14-4200-a904-91a3e52fe6d0)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=6d2ca720-0cef-45b6-a541-16a632ba500d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f9bba18f-4be3-418a-990d-8143d63a8555)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=a27092c8-8dce-4533-9356-32fd0bc713e5)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2e0eb543-ae7b-45b6-a41d-70ffeefe4024)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=cf50728d-0329-44f6-9a66-39f7154c2e0d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=9bb642ef-3cb2-4a90-abee-44dd88088b96)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=d8e13d9b-7b9a-45e1-8c8d-ebad9a63158a)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=33e880b2-30f6-4741-894e-18e415bcfd98)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/qu">

ery?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c4828ab1-8f52-4b92-8f0f-515c8c1ddc4f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=56288fb0-657c-486c-ad1d-b44680bc1850)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=68677d8b-92b4-4107-a9d1-4c4b88536679)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=a49d7998-590f-407f-9e67-3578ee5dd0ce)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3be60c2d-df3e-4c6c-955f-7a0a9aab7741)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8cc85b97-df87-499b-a134-50674538d2f4)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=0c0c3055-1ed5-47f5-a653-ec378af0ae61)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ba1f8bff-74a2-48f3-b799-de7e84564c9b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=012cff30-b84b-428a-81fd-cd54493f83ba)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1225559178261561470)** (21 条消息🔥): 

- **Phorm 聊天机器人参与讨论**：频道中引入了 Phorm，这是一个从 **OpenAccess-AI-Collective/axolotl** 查询数据以进行项目讨论的聊天机器人。
- **聊天模板格式化器入门**：一名成员询问了关于使用聊天模板格式化器的问题，Phorm 提供了详细的演练，建议使用 Hugging Face 的 Transformers 库和 `apply_chat_template` 方法。
- **RoPE 微调讨论**：术语 `"rope_theta": 10000.0` 引起了人们对其在 Transformers 中 Rotary Positional Embedding (RoPE) 应用的兴趣。一名成员询问了如何通过调整它来扩展上下文长度，结果显示虽然 Phorm 可以获取答案，但该话题仍需进一步澄清。
- **Rope Scaling 查询**：随后进行了关于 Rope Scaling 以及是否使用线性或动态调整的讨论；然而，结果显示 Rope Scaling 已被弃用，不再是一个相关的参数。
- **不当内容警报**：聊天记录中包含一条推销不当内容的消息，这在技术讨论中显然是不合时宜的。

*请注意，最后一个要点是关于聊天中出现的不当内容的报告，应根据平台规则进行审核。*
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2150270f-2213-4881-b572-a8c9dab49c46)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=a797f0f3-f29f-4cdb-81b9-95c8b008060f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3246eab0-a12a-4f23-ac87-0cb50c2fccf2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/)** (1 条消息): 

jerryjliu0: Webinar 将在 15 分钟后开始！^^
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1225471663022542952)** (4 条消息):

- **引入新的 Adaptive RAG 技术**：LlamaIndex 分享了一条 [推文](https://twitter.com/llama_index/status/1775912690529288556)，重点介绍了 @SoyeongJeong97 的 **Adaptive RAG** 论文，该论文针对简单和复杂多步问题的不同需求提出了解决方案，承诺在各自场景下表现出更好的性能。

- **MistralAI 发布 RAG Cookbook 系列**：@MistralAI 宣布在 LlamaIndex 上推出一系列关于构建 **从基础到高级 RAG 和 Agent** 的 Cookbook。专家可以通过这条 [推文](https://twitter.com/llama_index/status/1775977013054259210) 分享的资源，探索包括路由（routing）和查询分解（query decomposition）在内的 RAG 抽象。

- **推出 Claude Function Calling Agent**：LlamaIndex 推出了 Claude Function Calling Agent，利用了 @AnthropicAI 的 messages API 中新的工具使用（tool use）支持。发布 [推文](https://twitter.com/llama_index/status/1776051869850476840) 表明，使用 Haiku/Sonnet/Opus 可以扩展 Agent 的使用场景。

- **AutoRAG 优化 RAG Pipeline**：Marker-Inc-Korea 的 AutoRAG 作为一个使用评估数据集为特定用例 **自动优化 RAG Pipeline** 的系统被引入，并在一条 [推文](https://twitter.com/llama_index/status/1776289203459858849) 中分享。其目标是有效微调超参数以实现最佳性能。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1225341573215162418)** (160 条消息🔥🔥): 

- **SQL Query Engine 异步问题**：一位用户描述了在使用 Elastic search Vector database 和 Subquestion query engine 时，当 **use_async 标志设置为 true** 时出现的问题，导致 **connection timeout**。他们寻求关于如何解决在异步模式下遇到的连接超时问题的建议，因为在同步模式下运行正常。

- **Azure OpenAI MultiModal BadRequestError**：在按照 Azure OpenAI GPT4V MultiModal LLM 的示例 Notebook 操作时，一位用户遇到了与 “**Invalid content type. image_url**” 相关的 BadRequestError 并寻求帮助。在尝试了建议的更改（包括调整 base64 图像编码）后，错误仍然存在，引发了关于问题是源于更新的 Azure API 还是错误的图像文档参数（args）的讨论。

- **针对 AWS 上下文的 Prompt Engineering 策略**：一位用户询问如何编写 Prompt 模板，以 **始终假设用户查询与 AWS 相关**，建议包括在用户输入前添加诸如 “assume the following input is related to AWS” 之类的上下文来引导 LLM。

- **处理 Pydantic JSON 结构的复杂性**：讨论了 LlamaIndex 和 LangChain 等框架如何使用 Pydantic 来确保 JSON 结构，以及当自定义模型无法遵守指定格式时会发生什么。澄清了默认情况下，模型有一次机会返回正确的格式，但用户可以通过对组件进行子类化（subclassing）来修改逻辑。

- **RouterQueryEngine 过滤器应用和响应评估**：有疑问关于如何在运行时在 RouterQueryEngine 中传递过滤器，以及为每个 Prompt 重新构建引擎是否高效。提供了一个解决方案，表示为每个 Prompt 重新构建 Query Engine 没有问题，并且还提到了一种评估来自不同 Agent 的多个响应以保证结果质量的策略。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/?tab=llama_datasets">Llama Hub</a>: 未找到描述</li><li><a href="https://console.aws.amazon.com/ec2/.">未找到标题</a>: 未找到描述</li><li><a href="https://www.llamaindex.ai/blog/introducing-llama-datasets-aadb9994ad9e">Introducing Llama Datasets 🦙📝 — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=similarity#similaritypostprocessor">Node Postprocessor Modules - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py#L71">llama_index/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llama_dataset/uploading_llama_dataset/?h=dataset">Contributing a LlamaDataset To LlamaHub - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/905#issuecomment-1484288684">Where do I define top_k documents to be returned by similarity search over vectorstore? · Issue #905 · run-llama/llama_index</a>: 在调用 query 函数时，我该如何指定希望 retriever 传递给 LLM 的 k 值是多少？或者我需要在调用 query 函数之前指定它吗？llm_predictor = LLMPredictor(llm=ChatOp...</li><li><a href="https://github.com/run-llama/llama-hub/">GitHub - run-llama/llama-hub: A library of data loaders for LLMs made by the community -- to be used with LlamaIndex and/or LangChain</a>: 一个由社区制作的用于 LLMs 的数据加载器库 —— 与 LlamaIndex 和/或 LangChain 配合使用 - run-llama/llama-hub</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=query+pipeline">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/azure_openai_multi_modal/?h=azureopenaimultimodal">Multi-Modal LLM using Azure OpenAI GPT-4V model for image reasoning - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/?h=from_documents#llama_index.core.evaluation.DatasetGenerator.from_documents">Dataset generation - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/readers/simple_directory_reader/?h=simpledirector#llama_index.core.readers.file.base.SimpleDirectoryReader">Simple directory reader - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1225466375099908187)** (4 条消息): 

- **RAG 走向视觉化**：一位成员提出了用于图像的 **检索增强生成 (RAG)** 模型的想法，设想了一个可以识别最常用颜色、统计山脉等物体数量，或在保持某些特征的同时创造性地修改图像的系统。这可能潜在地用于绕过 CAPTCHA 或确保视觉叙事（如连环画）的连续性等任务。
- **LlamaIndex 探索多文档 Agent**：分享了一篇题为“[利用 LlamaIndex 释放多文档 Agent 的力量](https://medium.com/ai-advances/unlocking-the-power-of-multi-document-agents-with-llamaindex-d09e4d7dfe0e)”的博客文章链接，表明了关于推进多文档 Agent 的讨论。
- **对多文档创新的赞赏**：博客文章链接后紧跟着简短的表扬，标志着对文中提到的多文档 Agent 进展的积极反响。
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1225750620330528830)** (2 条消息): 

- **Claude 3 模型的多模态更新**：Claude 3 模型的情态已切换为 `multimodal` 以支持图像输入。开发者需要根据此更改更新其代码，并可以在 [公告频道](https://discord.com/channels/1091220969173028894/1223444233394847864) 找到更多详情。

- **Prompt 模板改进**：根据用户报告，对 DBRX 的 Prompt 模板进行了更新以减少重复性。欲了解更多信息，请访问提供的 [Discord 链接](https://discord.com/channels/1091220969173028894/1222619272208187402)。

- **推出 DBRX Nitro**：Databricks 的 **DBRX Nitro** 精通代码生成和通用知识，现在支持 120 tokens/sec 的吞吐量。更多见解和基准测试可以在 [这里](https://openrouter.ai/models/databricks/dbrx-instruct:nitro) 探索。

- **发布 Command R+ 模型**：Cohere 推出了全新的 **Command R+** 模型，拥有 104B 参数，在多项基准测试中表现优于竞争对手，并包含多语言支持以实现更广泛的可访问性。详细的基准测试和模型能力可以在 [这里](https://txt.cohere.com/command-r-plus-microsoft-azure/) 查看。

- **增强的 UI 和新参数**：OpenRouter 的 UI 现在新增了 **热门应用 (top apps)** 排行榜和专门的 **额度 (Credits)** 页面；`seed` 和 `logit_bias` 参数已添加到多个模型中。用户可以通过在提供商页面点击 "Mancer" 来检查支持这些参数的模型。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct:nitro">Databricks 的 DBRX 132B Instruct | OpenRouter</a>：DBRX 是由 Databricks 开发的一种新型开源大语言模型。它拥有 132B 参数，在语言相关的标准行业基准测试中优于 Llama 2 70B 和 Mixtral-8x7B 等现有开源 LLM...</li><li><a href="https://openrouter.ai/models/cohere/command-r-plus">Cohere 的 Command R+ | OpenRouter</a>：Command R+ 是来自 Cohere 的新型 104B 参数 LLM。它适用于角色扮演、通用消费者用例和检索增强生成 (RAG)。它为十种主要语言提供多语言支持...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1225356582905446441)** (155 条消息🔥🔥): 

- **Claude 的自我审查受到质疑**：用户正在讨论在使用 OpenRouter API（甚至是自我审查版本）调用 **Claude** 时，拒绝率比官方 Anthropic API 更高的问题。有人担心 OpenRouter 过去可能注入了额外的“安全”提示词。
- **Midnight Rose 暂停服务**：有报告称模型 **Midnight Rose** 无响应；重启集群的尝试已获成功，目前正在讨论迁移到更稳定的供应商或技术栈。已添加额外的核心供应商以提高稳定性，并鼓励用户报告任何进一步的问题。
- **困惑于 Schema**：在让 **Mixtral-8x7B-Instruct** 遵循 JSON schema 时出现问题；虽然 nitro 版本可以正常工作，但普通版本不行，直到 **OpenRouter** 团队介入并诊断出非供应商相关的问题，并承诺修复。

- **Command R+ 激发编程热情**：用户对在 OpenRouter 上尝试 **Command R+** 表示兴奋，注意到它在编程任务中的良好表现以及作为 AI 助手的潜力。
- **OpenRouter 提及 Cohere 模型**：在关于 **Cohere** 模型的讨论中，用户对其与 OpenRouter 的潜在集成感到好奇，讨论了该模型的能力，并比较了 Cohere 与 OpenAI 在业务和 API 设计方面的差异。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://prnt.sc]">未找到标题</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI 可接受使用政策</a>：未找到描述</li><li><a href="https://prnt.sc/_ba2eY63AJNA">截图</a>：使用 Lightshot 捕获</li><li><a href="https://docs.together.ai/docs/json-mode">JSON 模式</a>：未找到描述
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1225357151929634888)** (74 条消息🔥🔥):

- **寻找关于 Face Embeddings 的 AI 社区**：一名成员正在寻找可以讨论 Face Embeddings、相关数据集以及模型训练过程的社区。目前尚未链接具体的社区或资源。
- **如何部署和查询已部署的模型**：有关于如何调用已部署模型的 'predict' 接口的咨询，一名成员建议使用虚拟环境。另一场对话讨论了在 Android 上部署和查询 YOLOv8 等模型的支付选项和平台，并重点关注了延迟考量。
- **探索 AI 硬件选项**：多条消息讨论了 AI 硬件，包括 NPU 和 VPU 加速器，如 Intel Movidius Neural Compute Stick。成员们分享了不同年份产品的细节和链接，强调了其性价比和技术进步。
- **对开源 3D 环境项目的兴趣**：一名成员邀请他人加入一个开源项目，该项目需要具备 C++ 知识并理解 3D 环境。感兴趣的人员请通过私信（DM）联系。
- **LLM 并非万能**：多位成员指出人们对 Large Language Models (LLMs) 能力存在误解，表示它们不适合找公寓等任务，并强调 LLM 不是“万能的 AGI”，不应被套用到每一个可能的用例中。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>：暂无描述</li><li><a href="https://github.com/intel/intel-npu-acceleration-library">GitHub - intel/intel-npu-acceleration-library: Intel® NPU Acceleration Library</a>：Intel® NPU 加速库。可以通过在 GitHub 上创建账户来为 intel/intel-npu-acceleration-library 的开发做出贡献。</li><li><a href="https://youtu.be/goGsEl7trUA">NVIDIA's Low Power AI Dev Platform on Arm</a>：如果你想在面向 AI 未来的平台上进行开发，现在是时候选择 NVIDIA Jetson 开发平台而不是 Raspberry Pi 了。在本视频中...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1225412463906787429)** (3 条消息): 

- **提升速度的 Prompt Engineering**：讨论了在为聊天机器人设计生产级 Prompt 时，延迟与推理之间的权衡。提到了一种技巧：在用户输入时主动推理最可能的场景，以应对响应缓慢的问题。更多详情请参阅 [Twitter 帖子](https://x.com/siddish_/status/1772345589511901368?s=20)。
- **探索 Groq Cloud**：一名成员分享了题为 *Groking Groq III: Getting Started With Groq Cloud* 的 YouTube 视频。该视频可作为那些对 Groq 云服务感兴趣的人的入门参考。点击[此处](https://youtu.be/JTBCbkAXEBE)观看视频。
- **征集知识图谱（Knowledge Graph）资源**：有人请求提供关于知识图谱及其应用的资源。消息中未提供具体资源。

**提到的链接**：<a href="https://x.com/siddish_/status/1772345589511901368?s=20">Siddish (@siddish_) 的推文</a>：直接串流而不进行推理 -> 愚蠢的响应 🥴 串流直到推理完成 -> 响应缓慢 😴 一个小的 LLM 技巧：在用户输入时主动推理最可能的场景。

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1225496673598312520)** (8 条消息🔥): 

- **Visual AutoRegressive 提升图像生成标准**：提出了一种名为 Visual AutoRegressive (VAR) 建模的新范式，通过从粗糙到精细的细节预测图像，重新定义了自回归学习，并声称在 ImageNet 基准测试中优于 Diffusion Transformers。该方法将 [Frechet inception distance (FID)](https://arxiv.org/abs/2404.02905) 从 18.65 提升至 1.80，并将 Inception Score (IS) 从 80.4 提升至 356.4。

- **Chain-of-Thought Prompting 增强 AI 推理能力**：使用一种称为 Chain-of-Thought Prompting 的技术（展示中间推理步骤），显著提高了 LLM 在复杂推理任务上的能力。如[相关研究论文](https://arxiv.org/abs/2201.11903)所述，该方法仅需 8 个示例即可在 GSM8K 基准测试中达到最先进的准确率。

- **使用 LlamaIndex 探索多文档 Agent**：讨论了一种新的多文档 Agent LlamaIndex，这可能成为处理跨多个文档信息的规则改变者。开发详情涵盖在一篇 [Medium 文章](https://medium.com/ai-advances/unlocking-the-power-of-multi-document-agents-with-llamaindex-d09e4d7dfe0e)中。

- **Bitnet-Llama-70M 模型实验分享**：Bitnet-Llama-70M 是一个拥有 70M 参数的模型，在 HuggingFaceTB/cosmopedia 数据集的子集上进行训练，作为 BitNet 的实验性应用推出。尽管这是一个实验，该模型已与 [Wandb 训练报告](https://huggingface.co/abideen/Bitnet-Llama-70M)一同开放访问。

- **GitHub 仓库展示 Autobitnet**：新增了一个名为 Autobitnet 的 GitHub 仓库，它似乎是大型语言模型（LLM）大型课程的一部分，尽管消息中未讨论具体细节。您可以在该仓库的 [GitHub 页面](https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md)探索 Autobitnet 的具体内容。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了视觉自回归建模（VAR），这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测（next-scale prediction）”或“下一分辨率预测”...</li><li><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a>：我们探索了生成思维链（Chain-of-Thought）——一系列中间推理步骤——如何显著提高大型语言模型（LLM）执行复杂推理的能力。特别是，我们...</li><li><a href="https://huggingface.co/abideen/Bitnet-Llama-70M">abideen/Bitnet-Llama-70M · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2206.02066">PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers</a>：双分支网络架构在实时语义分割任务中展示了其效率和有效性。然而，高分辨率细节和低频上下文的直接融合具有...</li><li><a href="https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md">llm-course/llama_finetune/README.md at main · andysingal/llm-course</a>：通过在 GitHub 上创建一个账户来为 andysingal/llm-course 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1225413564504998022)** (17 条消息🔥): 

- **介绍 Metaforms AI**：一款全新的 AI **自适应表单产品（adaptive forms product）**已在 Product Hunt 上展示，邀请用户前往查看，或通过私信询问其内部原理。公告包含了 [Product Hunt 链接](https://www.producthunt.com/posts/metaforms-ai)。

- **音乐生成突破**：一位成员兴奋地分享了一个 YouTube 视频链接，演示了一个成功的 **musiclang2musicgen 流水线**实验，展示了足以挑战付费墙背后技术的音频生成能力。他们认为开源替代方案已经足够，并暗示了 [YouTube 视频](https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb)中展示的 *neural amp modeler*（神经放大器建模器）的潜力。

- **PyTorch Geometric 迎来新数据集**：PyTorch Geometric 项目合并了一个 Pull Request，集成了其生态系统内使用的新数据集。这些数据集在下次发布前可通过 `master` 分支访问，该更新附带了 [PR 链接](https://github.com/pyg-team/pytorch_geometric/pull/9090)和 Python 代码片段使用说明。

- **HybridAGI 开启神经符号航程**：一家专注于神经符号 Agent 系统的法国 AI 初创公司联合创始人介绍了 HybridAGI GitHub 仓库。该初创公司正在构建一个**免费、开源的神经符号 AGI**，并邀请社区在 Hugging Face 上提供反馈，共同建设其 [GitHub 项目](https://github.com/SynaLinks/HybridAGI)。

- **TensorLM 让 LLM 交互更简单**：一位成员分享了 TensorLM-webui，这是一个基于 LLaMA 的 GGML 格式 LLM 模型的简洁现代 Web UI。该工具旨在让文本生成像 Stable Diffusion 的视觉体验一样用户友好，并提供了 [GitHub 链接](https://github.com/ehristoforu/TensorLM-webui)供访问和贡献。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/TencentARC/BrushNet">BrushNet - TencentARC 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.producthunt.com/posts/metaforms-ai"> Metaforms AI - OpenAI + Typeform = 用于反馈、调查和研究的 AI | Product Hunt</a>：Metaforms 是 Typeform 的 AI 继任者。构建全球最强大的反馈、调查和用户研究表单，通过 generativeAI 收集关于用户的改变生活的洞察。训练于...</li><li><a href="https://telegram.me/int_gem_bot">Int Bot</a>：您可以立即联系 @int_gem_bot。</li><li><a href="https://github.com/ehristoforu/TensorLM-webui">GitHub - ehristoforu/TensorLM-webui：基于 LLaMA 的 LLM 模型简单现代的 WebUI。</a>：基于 LLaMA 的 LLM 模型简单现代的 WebUI。- ehristoforu/TensorLM-webui</li><li><a href="https://github.com/hegdeadithyak/PaperReplica">GitHub - hegdeadithyak/PaperReplica：我们复现 AI 和 ML 领域的研报。</a>：我们复现 AI 和 ML 领域的研报。- hegdeadithyak/PaperReplica</li><li><a href="https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb">无人创作的歌 #music #newmusic #song #timelapse #photography #musicvideo #viral #art</a>：未找到描述</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI：可编程的神经符号 AGI (Neuro-Symbolic AGI)，允许你使用基于图的提示编程 (Graph-based Prompt Programming) 来编写其行为：适用于希望 AI 表现符合预期的人群</a>：可编程的神经符号 AGI，允许你使用基于图的提示编程来编写其行为：适用于希望 AI 表现符合预期的人群 - SynaLinks/HybridAGI</li><li><a href="https://github.com/pyg-team/pytorch_geometric/pull/9090">feat: 添加 `CornellTemporalHyperGraphDatasets`，由 SauravMaheshkar 提交 · Pull Request #9090 · pyg-team/pytorch_geometric</a>：参考：#8501 #7312 请求评审：@rusty1s @wsad1 此 PR 旨在添加包含带时间戳单纯形 (timestamped simplices) 的 HyperGraph 数据集，其中每个单纯形是一组节点。随论文发布...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1225444976972791890)** (5 messages): 

- **下次演示的录制计划**：一名成员对错过下次演示表示遗憾，询问是否可以录制。该成员请求如果有人录制，请将链接发给他们，他们会将其添加到 GitHub。
- **OBS 可能会派上用场**：针对录制查询，另一名成员提到可以使用 *OBS* 录制演示。
- **技术帮助的频道建议**：当用户寻求帮助时，他们被引导至另一个更适合处理其查询的频道。
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1225388056857804911)** (11 messages🔥): 

- **Batch Size 的两难选择**：关于不同 **batch sizes** 如何影响训练过程中达到 local minima 存在争论。据报道，较小的 batch size 对小模型产生了更好的结果，但延长了训练时间；而较大的 batch size 被认为可能会错过 local minima，但训练速度更快。

- **LR Schedulers 作为探索者**：建议使用 **Learning Rate (LR) schedulers**，特别是 cyclic 或 cosine 调度器，因为它们提供了 exploration 和 exploitation 阶段，可以减轻陷入 local minima 的问题。

- **更新 HuggingFace 数据集所需的知识**：一名成员寻求关于是否需要手动重新上传修改后的自定义数据集以便在 **HuggingFace** 上进行 fine-tuning 的建议，另一名成员建议任何本地更改都需要 commit 和 push，类似于使用 **git**。

- **更新你的模型**：接着，进一步澄清了关于 **Hugging Face** 上模型和数据集的更新；就像使用 git 仓库一样，用户需要更新本地文件夹，然后将 **commit and push** 更改到 Hugging Face 的 model hub。

- **监控你的 GPU 使用情况**：关于如何确定模型训练期间 **GPU usage** 量的问题引起了兴趣，但在讨论的消息中没有立即得到回答。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1225349667303198802)** (13 messages🔥): 

- **关于 Ollama 模型部署的咨询**：用户询问了运行 **Ollama** 模型（特别是 `phi` 变体）的内存要求，并担心其旧 GPU 是否支持。他们还询问 Ollama 是通过本地模型部署运行，还是通过类似于 OpenAI 设置的 API 调用运行。

- **Babbage-002 Context Length Clarification**: 关于 **babbage-002** 模型的 Context Length 说明：一位用户澄清说，在进行 Fine-tuning 时无法增加 Context Length，但在从头开始训练（training from scratch）时是可能的。

- **Utilizing Kaggle for Training AI**: 一位成员分享了在 **Kaggle** 上使用 **llama2** 训练医学百科 Chatbot 的成功经验，并建议将其作为类似项目的可行平台。
  
- **Seeking Free LLM for Enhanced Chatbot Responses**: 某位用户正在构建一个集成 Google Books API 的 AI Chatbot，并寻求一个免费的 Large Language Model (LLM)，以返回更详尽的回复，例如完整的句子而非简短的回答。

- **Space Concerns for Task Execution**: 一位用户幽默地表达了对正在执行的任务磁盘空间不足的担忧，对其当前资源的可行性表示怀疑。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225780367022096546)** (5 messages): 

- **PEFT Conundrum with llava2 Model**: **llava2 模型**的 **PEFT** 难题：一位使用 **PEFT** (Prune, Expand, Fine-Tune) 处理 **llava2 模型**的成员在将模型转移到另一台机器时遇到了挑战，原因是 safetensors 格式问题。建议的干预措施是检查 `use_safetensors=True`。

- **NLP Beginner Guidance Sought**: 寻求 NLP 初学者指导：一位新成员询问 NLP 的切入点，纠结于应该专注于 **Transformer** 架构还是传统的 **LSTM**、**GRU** 和 **Bidirectional** 网络。他们被引导至通过 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) 提供的斯坦福 CS224N 课程，作为全面的学习资源。

**Link mentioned**: <a href="https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4">Stanford CS224N: Natural Language Processing with Deep Learning | 2023</a>: 自然语言处理 (NLP) 是人工智能 (AI) 的重要组成部分，模拟人们分享信息的方式。近年来，深度学习应用...

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1225378463523803146)** (87 messages🔥🔥): 

<ul>
<li><strong>Exploring Tinygrad's NPU Support & Performance Optimization:</strong> 关于 <strong>tinygrad</strong> 是否支持新笔记本电脑上的专用 NPU 的讨论引发了兴趣，提到了 Intel 的库，但不确定 <strong>tinygrad</strong> 是否支持。分享了一个 [tinygrad 推理优化列表](https://github.com/tinygrad/tinygrad/blob/master/docs/env_vars.md)，用于与 onnxruntime 进行性能对比。</li>
<li><strong>Intel GPU and NPU Driver Discussion:</strong> 用户阐明了 Intel 硬件的各种内核驱动程序，强调了用于 Intel GPU 的 `gpu/drm/i915`，用于新 Intel GPU 的 `gpu/drm/xe`，以及用于 Intel VPU/NPU 的 `accel/ivpu`。Linux 内核 6.8 版本包含了必要的驱动程序，并计划在 Ubuntu 24.04 LTS 发布后进行实验。</li>
<li><strong>Potential Scalability of tinygrad:</strong> 有人提到 tinygrad 尚不支持类似 NVIDIA 的互连带宽，George Hotz 澄清说，通过 200 GbE 全 16x 互连插槽有扩展潜力，并且 PyTorch 可以工作，暗示未来将支持 multimachine。</li>
<li><strong>Heterogeneous Acceleration and Power Efficiency:</strong> 关于异构加速的对话揭示了利用现有计算能力的潜力，以及 NPU 在提供与 GPU 相当的性能时功耗仅为一半的能效优势。</li>
<li><strong>Kernel-Level Integration & Development Opportunities:</strong> 提到了 AVX-512 的障碍以及希望看到 Intel 改进的愿望，并附上了一个讨论这些问题的 LKML 邮件链接。用户还讨论了 AMD 关于开源的承诺，并对其交付可靠性进行了推测，同时对影响表示怀疑。</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.phoronix.com/news/AMD-MES-Docs-And-Source-Code">Tweet from AMD Working To Release MES Documentation &amp; Source Code - Phoronix</a>: 未找到描述</li><li><a href="https://www.realworldtech.com/forum/?threadid=193189&curpostid=193190">Real World Technologies - Forums - Thread: Alder Lake and AVX-512</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/env_vars.md">tinygrad/docs/env_vars.md at master · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1225348513542701076)** (8 messages🔥):

- **避免 JIT 混淆**：一位用户表达了困惑，询问 *为什么我们在 ignore jit pass 上进行缓存收集 (cache collecting)？？*
- **性能分析指南**：一位用户分享了关于在运行 tinygrad 并设置 **DEBUG=2** 时如何解读性能输出的[学习笔记](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/profiling.md)。他们表示未来的目标可能是计算 MNIST 示例的理论训练时间。
- **理解 TinyJit**：对于那些寻求理解 TinyJit 的人，提供了一份[教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md)，尽管作者警告说在 `apply_graph_to_jit` 部分可能存在一些不准确之处。
- **教程免责声明与征求修正**：TinyJit 教程的作者承认了潜在的错误，并邀请社区提供反馈以改进文档。
- **社区支持请求**：一位用户建议具备相关知识的贡献者创建一个 Pull Request 来修正 TinyJit 教程中的错误，以帮助社区。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic-egg">mesozoic-egg - 概览</a>：mesozoic-egg 有 3 个公开的代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md">tinygrad-notes/jit.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账户为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/profiling.md">tinygrad-notes/profiling.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账户为 mesozoic-egg/tinygrad-notes 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1225438495749967913)** (41 条消息🔥): 

- **Cohere 发布 Command R+**：Cohere 推出了 **Command R+**，这是一款专为企业解决方案量身定制的新型可扩展 LLM，支持高级 RAG 和 Tool Use。该模型及其[权重](https://huggingface.co/CohereForAI/c4ai-command-r-plus)专为真实世界的企业用例设计，拥有 128k token 的上下文窗口并支持多语言。 

- **对“企业版 ChatGPT”模型的怀疑态度**：关于“企业版 ChatGPT”应用的有效性和必要性引发了辩论，观点认为这些模型可能无法直接解决公司的实际复杂需求。

- **JetMoE-8B：高性价比且强大**：新型 **JetMoE-8B 模型**作为 LLM 的高性价比替代方案推出，声称仅花费 10 万美元的训练成本就超越了 Meta 的 LLaMA2-7B。它的亮点在于对学术界友好、开源且仅需极少的计算资源。

- **OpenAI 辅助微调服务**：OpenAI 宣布了针对 LLM 的辅助微调服务。这种咨询形式允许在微调 API 之外进行调整，包括额外的超参数和参数高效微调 (PEFT) 方法。

- **对大科技公司合并的担忧**：讨论涉及由于现有的反垄断情绪，重大科技公司收购通过监管审批的可能性极低，并质疑了此类战略决策背后的逻辑。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">介绍 Command R+：为业务构建的可扩展 LLM</a>：Command R+ 是一款最先进的 RAG 优化模型，旨在处理企业级工作负载，并首先在 Microsoft Azure 上可用。今天，我们推出了 Command R+，我们最强大的...</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>：暂无描述</li><li><a href="https://openai.com/blog/openai-partners-with-scale-to-provide-support-for-enterprises-fine-tuning-models">OpenAI 与 Scale 合作，为企业微调模型提供支持</a>：OpenAI 的客户可以利用 Scale 的 AI 专业知识来定制我们最先进的模型。</li><li><a href="https://research.myshell.ai/jetmoe">JetMoE</a>：暂无描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：暂无描述</li><li><a href="https://fxtwitter.com/aidangomez/status/1775878606108979495?s=46">Aidan Gomez (@aidangomez) 的推文</a>：⌘R+ 欢迎 Command R+，我们专注于可扩展性、RAG 和 Tool Use 的最新模型。与上次一样，我们发布了供研究使用的权重，希望它们对每个人都有用！https:/...
</li>
</ul>

</div>
  

---

**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1225458505583034408)** (3 条消息): 

- **Nathan 捅了马蜂窝？**: 一位成员分享了他们自己关于机器学习的 Twitter 帖子，引发了关于潜在争议（drama）的玩笑。推文见[此处](https://twitter.com/natolambert/status/1775899591814300024)。
- **Snorkel 等待裁决**: 针对对 ML 模型的看法，一位成员调侃说，随后可能会发表对 **Snorkel** 的看法，并暗示这将是未来一篇题为 *"all these models are bad"* 文章的一部分。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1225399038942380043)** (41 条消息🔥): 

- **CS25 讲座参与确认**: Nathan 将在 **CS25** 讲座上发言，讨论了潜在的行程安排，包括使用 Uber 前往。
- **分享 Musk 的法律诉讼推文**: Nathan 分享了 **Elon Musk** 威胁要对某些个人采取法律行动的推文：*"[X Corp 将追踪相关责任人，并动用法律的全部力量对付他们。](https://twitter.com/elonmusk/status/1775900800520262071)"*
- **CS25 热门研讨课阵容**: CS25 研讨课阵容包括著名的研究人员和行业专家，其日程安排可在 [Stanford 的 CS25 课程页面](https://web.stanford.edu/class/cs25/#schedule)找到，演讲者来自 **OpenAI, Google, NVIDIA** 等。
- **与 ContextualAI 的互动**: Nathan 对 ContextualAI 的工作评价积极，指出他们以客户为中心，且 Series A 融资进展顺利，称他们所有的研究项目都是 "bangers"（非常出色）。
- **对工作归属的批评**: Nathan 对一位前同事的演示表示怀疑，断言他们冒领了不属于自己的工作功劳，但承认其作为演示者的能力以及对关键主题的理解。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://web.stanford.edu/class/cs25/#schedule">CS25: Tranformers United!</a>: 讨论 Transformer 在不同领域的最新突破</li><li><a href="https://www.youtube.com/watch?v=QXVCqtAZAn4&pp=ygUTbGV3aXMgdHVuc3RhbGwgdGFsaw%3D%3D">Aligning LLMs with Direct Preference Optimization</a>: 在本次研讨会中，来自 Hugging Face 的 Lewis Tunstall 和 Edward Beeching 将讨论一种强大的对齐技术，称为 Direct Preference Optimisation (DPO)...</li><li><a href="https://www.youtube.com/watch?v=mcep6W8oB1I&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=22&t=389s&pp=iAQB">Stanford CS25: V3 I Recipe for Training Helpful Chatbots</a>: 2023年10月31日，Nazneen Rajani (HuggingFace)。目前已有大量关于使用 Large language models (LLMs) 训练助人型对话 Agent 的工作...</li><li><a href="https://youtu.be/mE7IDf2SmJg?si=BEtHoA0hp9m78XcZ">Stanford CS25: V3 I Retrieval Augmented Language Models</a>: 2023年12月5日，Douwe Kiela (ContextualAI)。语言模型取得了惊人的进步，但也有重要的缺点。针对其中许多问题的解决方案是...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1225594554724388936)** (8 条消息🔥): 

- **Transformer 效率的新见解**: 一位成员强调了 DeepMind 关于 Transformer 的 [Mixture of Depths](https://arxiv.org/abs/2404.02258) 研究，讨论了它如何在固定计算预算下，跨序列动态分配 FLOPs。它引入了一种 top-$k$ 路由机制来优化 FLOPs 分布，这可能为前向传递增加稀疏性提供一种方法。

- **结合 MoE 和 Mixture of Depths？**: 同一位成员推测 **Mixture of Depths** 方法可能与 **Mixture of Experts (MoE)** 模型兼容，从而增强前向传递期间的稀疏性，并表示打算尝试将其集成到现有模型中。

- **对实际实现的期待**: 成员们对将 **Mixture of Depths** 方法事后（post-hoc）添加到现有模型的潜力感到兴奋，并计划在即将到来的周末探索其实际应用。

- **对持续学习的认可**: 其他成员注意到了分享的工作，请求告知实验结果，并承认从社区讨论中学习新架构增强功能的价值。

**提及的链接**: <a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: 基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算量）分配给特定的...

  

---

**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/)** (1 条消息): 

natolambert: 这场演讲的吉祥物 lol
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1225359091803623444)** (85 条消息🔥🔥): 

- **发起 GitHub 讨论**: 一位成员发起了关于在 LangChain 中使用 **Output Parsers, Tools, and Evaluators** 的讨论，特别是探索了确保 LLM chain 输出 JSON 的不同方法。他们引发了一场基于观点的对话，并提供了 [GitHub 讨论链接](https://github.com/langchain-ai/langchain/discussions/19957)。
  
- **寻求摘要策略协助**: 一位用户询问了关于 **ChatGroq** 错误排查的问题，推测是由于多请求摘要任务期间的速率限制（rate limitations）引起的，并寻求补救策略的建议。

- **法律文档的 Q&A 分块**: 另一位成员寻求关于优化 Q&A 系统中法律文档分块（chunking）和存储流程的建议，并指出他们目前的递归方法可能过于简单。

- **廉价 LLM 之间的质量比较**: 一位社区成员询问了廉价 LLM 模型的质量，引用了 **ChatGPT 3.5** 和 **Haiku** 作为对比点。

- **渴望 LangChain 辅导**: 一位成员表达了对辅导的需求，特别要求在 LangChain 中实现 **RAG** 的专业知识，这表明在 LangChain 检索增强生成（retrieval-augmented generation）这一细分领域存在个性化学习的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/.">Microsoft Excel | 🦜️🔗 LangChain</a>: UnstructuredExcelLoader 用于加载 Microsoft Excel 文件。</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#create-a-tool>)).">快速入门 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍创建调用 Tools 的 Chains 和 Agents 的基本方法。Tools 可以是几乎任何东西——API、函数、数据库等。Tools 允许我们扩展功能...</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:~:text=Routing%20allows%20you%20to%20create%20non-deterministic%20chains%20where,runnables%20from%20a%20RunnableLambda%20%28recommended%29%20Using%20a%20RunnableBranch.">基于输入路由逻辑 | 🦜️🔗 LangChain</a>: 基于输入动态路由逻辑}</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:">基于输入路由逻辑 | 🦜️🔗 LangChain</a>: 基于输入动态路由逻辑}</li><li><a href="https://github.com/langchain-ai/langchain/pull/19751.">更好地共同构建软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19957">何时使用 Outputparsers、tools 和/或 LangSmith Evaluators 来测试 LLM 输出？ · langchain-ai/langchain · Discussion #19957</a>: 我正在为一个简单的任务开发一个简单的 LCEL chain，脑子里冒出了这个问题。想象一下，我有一个简单的 LCEL chain，包含 2 个 prompts 和 2 个 output parsers 来“强制”...</li><li><a href="https://python.langchain.com/docs/integrations/tools/lemonai#load-api-keys-and-access-tokens>),">Lemon Agent | 🦜️🔗 LangChain</a>: Lemon Agent 帮助您</li><li><a href="https://js.langchain.com/docs/use_cases/graph/prompting#set-environment-variables>)).">Prompting 策略 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍改进 graph 的 prompting 策略</li><li><a href="https://github.com/langchain-ai/langchain-google/issues/117">Gemini Generative AI 语义检索示例缺少 GoogleVectorStore 的导入 · Issue #117 · langchain-ai/langchain-google</a>: 问题：genai 语义检索示例缺少 GoogleVectorStore 所需的导入。文件：libs/genai/README.md。所需导入：from langchain_google_genai import GoogleVectorStore。附加信息...</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L22">langchain/libs/community/pyproject.toml at request-body-reference · anujmehta/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户为 anujmehta/langchain 的开发做出贡献。</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L244.">langchain/libs/community/pyproject.toml at request-body-reference · anujmehta/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户为 anujmehta/langchain 的开发做出贡献。
</li>
</ul>

</div>
  


---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1225540697554419803)** (3 条消息): 

- **过于执着的 Agent 困在 PDF 中**: 一位成员寻求关于一个坚持对每个查询都搜索 PDF 的 Agent 的帮助。提供的 **system prompt** 驱动了该 Agent 的行为，建议通过编辑来包含何时不使用 PDF 的条件是明智的。

- **将 Azure 凭据与 VectorDB 集成**: 另一位成员请求关于在为聊天机器人集成 Azure 凭据的同时，保留之前使用 OpenAI API key 嵌入的现有 **FAISS Vector Database** 的建议。他们分享了目前使用 OpenAI 直接 API key 设置的代码片段。

**提到的链接**: <a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。

  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1225822204848111676)** (2 条消息):

- **语义分块（Semantic Chunking）现已支持 TypeScript**：一位成员分享了 LangchainJS 中原有的语义分块功能的 [TypeScript 实现](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae)，这让基于 Node 的 Web 应用开发者受益。该 Gist 概述了处理文本、计算嵌入（embeddings）以及将句子分组为内聚分块的过程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae">这个 TypeScript 代码片段通过将文本标记为句子、结合上下文、使用 OpenAI 服务生成句子嵌入、计算余弦相似度以识别语义转移，最后根据这些转移将句子分组为语义内聚的分块，从而处理大型文本语料库并输出语义分块。</a>：这个 TypeScript 代码片段处理大型文本语料库，通过将文本标记为句子、结合上下文、使用 OpenAI 服务生成句子嵌入...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1225702752664027177)** (2 条消息): 

- **西班牙语 DSPy 入门**：一位成员分享了题为 [“¿Cómo Usar DSPy? Nivel Básico Explicado”](https://youtu.be/N1ChFQm02QM) 的 YouTube 视频，为有兴趣学习的西班牙语使用者提供了 DSPy 的基础概述。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/NSRpYWCM">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://youtu.be/N1ChFQm02QM">¿Cómo Usar DSPy? Nivel Básico Explicado</a>：关于 DSPy 的非常基础的概述，如果想深入了解相关话题，请留言！:)
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1225464643678503054)** (66 条消息🔥🔥): 

- **对 Apple MPS 的批评**：讨论集中在对 **Apple MPS** 的不满上，一位成员表示 Apple 未能履行其在 AI 方面的承诺，另一位成员建议尝试 PyTorch nightly 分支以获取修复。

- **关于扩散模型和音频分轨的咨询**：有人好奇 **DALL·E** 如何保存图像编辑历史，并有兴趣使用 **SDXL** 制作类似的功能。另一位成员正在寻找关于播客分轨（stemming）的研究，特别是询问除了**说话人日志（speaker diarization）**之外的语音特定技术。

- **AIDE 在 Kaggle 中达到人类水平**：一款 AI 驱动的数据科学 Agent **AIDE** 在 **Kaggle 竞赛**中达到人类水平的消息引发了辩论，讨论这是否真正匹配人类体验，并指出了压力和时间限制等人类因素。

- ** /r/StableDiffusion 的回归**：讨论了 **Reddit API 访问受限**及其对应用开发者、版主和盲人用户的影响。还提到了子版块 /r/StableDiffusion 的重新开放，并链接了更多信息和相关社区。

- **在 macOS 上使用 PyTorch 的探索与挫败**：一位成员在 **macOS PyTorch 2.4 上实验 aot_eager 后端**，分享了他们在性能和优化方面的发现，包括一个成功的案例：利用 Apple 的 CoreML 量化（quant）能力将图像生成时间从 57 秒缩短到 3 秒。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.weco.ai/blog/technical-report">Weco AIDE 介绍</a>：你的机器学习 AI Agent</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bvv4hy/well_it_is_accurate/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: 无调度优化入口</a>：无调度优化入口。通过在 GitHub 上创建账号来为 facebookresearch/schedule_free 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=EC3YKPZYtoo">从零开始编写图像扩散模型第二部分</a>：twitter.com/yanisfalakigithub.com/yanis-falakiinstagram.com/yanis_falaki
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1225648903391674410)** (3 条消息):

- **Google 的创新 Token 压缩技术**：Google 研究人员提出了一种 [Token 压缩与训练](https://arxiv.org/pdf/2404.03626.pdf) 方法，旨在减小模型尺寸并降低计算需求。

- **Transformers 中的动态 FLOPs 分配**：一种通过在输入序列中动态分配 FLOPs 来优化 Transformer 模型的新方法被提出，详见论文 ["Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"](https://arxiv.org/abs/2404.02258)。该方法利用 top-$k$ 路由机制来限制计算资源，同时仍保持静态计算图。

**提到的链接**: <a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会将 FLOPs（或计算量）动态分配给特定的...

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1225355446856978482)** (61 条消息🔥🔥): 

- **对 DeepMind Mixture-of-Depths 的不同看法**：DeepMind 的 Mixture-of-Depths 论文旨在动态分配 Transformer 模型中的计算量，目前收到了各种反馈。批评者质疑其原创性和实际的计算成本节省，而一些人则称赞其通过减少对低复杂度 Token 的操作，具有显著节省计算量的潜力。

- **Anthropic 的 Claude 展示工具使用能力**：[Anthropic](https://twitter.com/AnthropicAI) 在 Claude 中引入了实时工具使用（Tool Use）功能，引发了对其机器人能力以及对 AI 操作复杂度影响的 [讨论与分析](https://twitter.com/swyx/status/1775993946935906645)。

- **圣迭戈 AI 论文俱乐部活动**：圣迭戈 AI 社区正在举办 AI 论文俱乐部会议，与会者将投票决定论文主题。感兴趣的人可以 [注册参加活动](https://lu.ma/paperclub3) 并阅读之前的文章，如 [hlfshell 的博客](https://hlfshell.ai/posts/llms-and-robotics-papers-2023/) 上的 "LLMs + Robotics"。

- **ReFT：一种新的微调方法**：围绕斯坦福大学新的 ReFT (Representation Finetuning) 方法展开了讨论，该方法声称比现有的微调技术具有更高的参数效率。ReFT 方法可以用极少的参数更新来适配语言模型，从而在无需昂贵重训的情况下引导模型。

- **关于 ML 框架性能的辩论**：François Chollet 为一种性能基准测试方法辩护，该方法显示默认设置下的 Keras 优于使用 Hugging Face 模型的 PyTorch 等其他框架，强调开箱即用的速度而非手动优化的性能。这引发了关于机器学习框架公平比较的激烈讨论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1776057023697731913?s=46&t=90xQ8sGy63D2OtiaoG">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：ReFT：语言模型的表示微调（Representation Finetuning）。比之前的 SOTA 参数高效微调（PEFT）方法在参数效率上高出 10 到 50 倍。仓库：https://github.com/stanfordnlp/pyreft...</li><li><a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">介绍 Fine-tuning API 的改进并扩展我们的定制模型计划</a>：我们正在添加新功能，以帮助开发者更好地控制微调过程，并宣布了与 OpenAI 合作构建定制模型的新方法。</li><li><a href="https://x.com/fchollet/status/1776319511807115589">来自 François Chollet (@fchollet) 的推文</a>：我坚持这些数据——正如博客文章中所强调的，我们并不是在衡量通过编译器感知方式重写每个模型后所能达到的最佳性能。用户可以参考...</li><li><a href="https://x.com/giffmana/status/1776156471245349304">来自 Lucas Beyer (@giffmana) 的推文</a>：能够对不同位置进行差异化修改是其相对于参数空间 PEFT 方法（LoRA/DoRA/...）的一个关键优势，但同时它也存在无法被“固化”（baked-in）的缺点...</li><li><a href="https://x.com/horseracedpast/status/1775757613000507736?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 horseboat (@horseracedpast) 的推文</a>：Bengio 居然在 2013 年就写过这个。↘️ 引用 AK (@_akhaliq)：Google 发布 Mixture-of-Depths，在基于 Transformer 的语言模型中动态分配计算资源。基于 Transformer 的语言模型...</li><li><a href="https://x.com/theseamouse/status/1775782800362242157?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：为什么 Google DeepMind 的 Mixture-of-Depths 论文以及更广泛的动态计算方法很重要：大部分计算资源被浪费了，因为并非所有 Token 的预测难度都相同。</li><li><a href="https://techcrunch.com/2024/04/03/opera-will-now-allow-users-download-and-use-llms-locally">Opera 允许用户在本地下载并使用 LLM | TechCrunch</a>：Opera 今天表示，现在将允许用户在其桌面端本地下载并使用大语言模型（LLMs）。</li><li><a href="https://x.com/theseamouse/status/1775743110774931846?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：@fouriergalois @GoogleDeepMind 兄弟，带 Early Exit 的 MoE。整个图谱都下移了，这相当于节省了 10 倍的计算量……天呐。</li><li><a href="https://x.com/victortaelin/status/1776225351678468429">来自 Taelin (@VictorTaelin) 的推文</a>：亲爱的日记，今天我教会了 1000 人如何使用 Interaction Combinators，但代价是什么呢？↘️ 引用 Taelin (@VictorTaelin)：一个 GPT 永远无法解决的简单谜题：作为一个优秀的程序员，我喜欢隔离...</li><li><a href="https://x.com/mayfer/status/1776062638205579727?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 murat 🍥 (@mayfer) 的推文</a>：噢哇，新的微调方法演示，仅更新了 0.00006% 的参数（即 4,907 个），就能让它在收到 GO-> 提示时完美背诵这段文字。↘️ 引用 Aran Komatsuzaki (@arankom...</li><li><a href="https://vgel.me/posts/representation-engineering/">
    
      
        表示工程（Representation Engineering）：Mistral-7B 的迷幻之旅
      
    
</a></li></ul>

  </a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=39934584">理解并管理机器学习模型对网络的影响 | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/arankomatsuzaki/status/1776057023697731913?s=46&t=90xQ8sGy63D2OtiaoGJuww">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: ReFT: Representation Finetuning for Language Models。比之前的 SOTA 参数高效微调方法效率高出 10x-50x。仓库：https://github.com/stanfordnlp/pyreft ...</li><li><a href="https://x.com/cohere/status/1775878850699808928?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">cohere (@cohere) 的推文</a>: 今天，我们推出了 Command R+：一款针对 RAG 优化的 SOTA LLM，旨在处理企业级工作负载并支持全球商业语言。我们的 R 系列模型家族现已...</li><li><a href="https://x.com/andersonbcdefg/status/1775751252330385807?s=20">Ben (e/sqlite) (@andersonbcdefg) 的推文</a>: 太棒了。“你喜欢 MoE？如果我们把其中一个专家设为恒等函数（identity function）会怎样。” 砰的一声，节省了 50% 的 FLOPs 🤦‍♂️ ↘️ 引用 Aran Komatsuzaki (@arankomatsuzaki)：Google 发布 Mixture-of-De...</li><li><a href="https://x.com/sherjilozair/status/1775765404528615798?s=46&t=90xQ8sGy63D2OtiaoGJuww">Sherjil Ozair (@sherjilozair) 的推文</a>: 这怎么发表的？🤔 ↘️ 引用 AK (@_akhaliq)：Google 发布 Mixture-of-Depths。在基于 Transformer 的语言模型中动态分配计算资源。基于 Transformer 的语言模型...</li><li><a href="https://lu.ma/paperclub3">SDxPaperClub · Luma</a>: SDx Paper Club。即将展示的论文是 [待定]，作者 [待定]。Twitter | Discord | LinkedIn</li><li><a href="https://docs.cohere.com/docs/command-r-plus">Command R+</a>: 未找到描述</li><li><a href="https://coral.cohere.com/?s=t">登录 | Cohere</a>: Cohere 通过一个易于使用的 API 提供对先进 Large Language Models 和 NLP 工具的访问。免费开始使用。</li><li><a href="https://github.com/myshell-ai/JetMoE">GitHub - myshell-ai/JetMoE: 以 10 万美元达到 LLaMA2 性能</a>: 以 10 万美元达到 LLaMA2 性能。通过在 GitHub 上创建账户为 myshell-ai/JetMoE 的开发做出贡献。</li><li><a href="https://hlfshell.ai/posts/representation-engineering/">Representation Engineering and Control Vectors - LLM 的神经科学</a>: 摘要：最近的一篇论文以类似于神经科学的方式研究了 Large Language Model (LLM) 对刺激的反应，揭示了一种用于控制和理解 LLM 的诱人工具。我在这里写道...</li><li><a href="https://buttondown.email/ainews/archive/ainews-cohere-command-r-anthropic-claude-tool-use/">[AINews] Cohere Command R+, Anthropic Claude Tool Use, OpenAI Finetuning</a>: 2024年4月3日至4月4日的 AI 新闻。我们为您检查了 5 个 subreddit、364 个 Twitter 账号和 26 个 Discord（385 个频道，5656 条消息）。预计阅读时间...</li><li><a href="https://lu.ma/ls">Latent Space (Paper Club & 其他活动) · 活动日历</a>: 在 Luma 上查看并订阅来自 Latent Space (Paper Club & 其他活动) 的活动。Latent.Space 活动。请点击日历右上方上方的 RSS 图标以添加到您的日历。“Ad...</li><li><a href="https://github.com/Paitesanshi/LLM-Agent-Survey">GitHub - Paitesanshi/LLM-Agent-Survey</a>: 通过在 GitHub 上创建账户为 Paitesanshi/LLM-Agent-Survey 的开发做出贡献。</li><li><a href="https://abyssinian-molybdenum-f76.notion.site/237e9f7515d543c0922c74f4c3012a77?v=0a309e53d6454afcbe7a5a7e169be0f9">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1225854717213544498)** (4 条消息): 

- **首届 Latent Space University 课程发布**: 首个在线 Latent Space University 课程定于太平洋时间今天下午 1 点开始，第一节课免费参加。感兴趣的人可以在[这里](https://maven.com/p/245c45)报名。

**提到的链接**: <a href="https://maven.com/p/245c45">编写自定义 ChatGPT</a>: 这是 AI 产品的基石。如果你想成为一名 AI 工程师，这些是必须掌握的主题和 API。从 ChatGPT 到强大的 AI 驱动的摘要和分类，一切都使用...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1225531216166518937)** (29 条消息🔥):

- **AI 的新语音**：一位成员成功让 OpenInterpreter 编写并使用其自有的语音交互 wrapper，并能持续通过该 wrapper 启动。这种语音集成并不等同于 01 的能力，但被视为一项有趣的进展。
- **神秘的 Compare 端点**：OpenAI API 现在在 playground 中提供了一个 **Compare 端点**，但尚未在 API 文档或在线文档中记录。该功能允许对不同模型和生成参数的查询进行并排对比。
- **Windows 上 OpenInterpreter 的故障排除**：一位成员在 Windows 上设置 OpenInterpreter 01 时遇到问题，且由于 CTRL + C 无效，不确定如何退出终端。
- **昂贵模型的高性价比替代方案**：人们正在寻找与 OpenInterpreter 兼容的实惠本地模型建议，用于处理大型 Excel 文件等任务，对于 **Mixtral** 和 **Mistral 7B Instruct v0.2** 等选项的反馈褒贬不一。
- **活动公告与录音咨询**：讨论围绕即将举行的活动展开，分享了一个 **OI Python 库活动**的链接，提出了关于活动通知时区调整的问题，并咨询了如何为“build with me”环节录制 Discord 语音聊天。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/xXtcB9hq?event=1225831217832919051">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的好友和社区保持紧密联系。</li><li><a href="https://discord.gg/wNJZsJgQ?event=1221828294811586572">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的好友和社区保持紧密联系。</li><li><a href="https://tenor.com/view/roman-reigns-wwe-wwe-roman-reigns-tribal-chief-i-dont-give-a-damn-what-the-tribal-chief-says-gif-4256408665054900859">Roman Reigns Wwe GIF - Roman reigns Wwe Wwe roman reigns - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1225486820511449140)** (26 messages🔥): 

- **01OS 的 Python 版本兼容性**：成员报告在 Python 3.11 及更高版本运行 **01OS** 时出现问题，建议使用 **Python 3.10 或更低版本**以保证兼容性。
- **OpenInterpreter 不支持 Ubuntu Wayland**：根据 [OpenInterpreter Issue #219](https://github.com/OpenInterpreter/01/issues/219)，确认不支持带有 Wayland 的 Ubuntu 21+，因为某些依赖项需要 x11。
- **Linux 发行版问题的潜在解决方案**：用户建议访问 `/var/log/dmesg` 可能会导致启动错误，[Issue #103](https://github.com/OpenInterpreter/01/issues/103) 中讨论了一个临时解决方案。提议的代码方案包括检查 `dmesg` 并将其输出重定向到临时位置。
- **01 音频连接问题**：多位用户遇到 **01 客户端与服务器音频连接**问题，表现为音频录制已开始但未发送或处理，暗示存在一个新的、未明确的客户端问题。
- **使用 Conda 环境修复 TTS 软件包问题**：解决安装过程中 TTS 软件包冲突的建议包括创建一个 **Python <=3.10 的 Conda 环境**，重新克隆仓库并重新安装以确保兼容性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01/issues/219">不支持 Ubuntu 21+ [wayland] · Issue #219 · OpenInterpreter/01</a>：某些依赖项使用 x11，与 wayland 不兼容 https://github.com/Kalmat/PyWinCtl?tab=readme-ov-file#linux-notice https://github.com/asweigart/pyautogui/issues?q=is%3Aissue+is%3Aopen...</li><li><a href="https://github.com/OpenInterpreter/01/issues/103#issuecomment-2038126686">在 !Ubuntu linux 上无法工作 · Issue #103 · OpenInterpreter/01</a>：描述 Bug：在 linux 上同时运行服务器和客户端时出错。01 --server ➜ 01 --server ○ Starting... INFO: Started server process [247252] INFO: Waiting for application startup. Task excep...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1225367465639415828)** (3 messages):

- **BitMat 震撼 1-bit LLM 领域**：分享了一个 GitHub 链接，重点介绍了 **BitMat**，这是对《1-bit LLMs 时代》论文中所提方法的高效实现。该仓库地址为 [astramind-ai/BitMat](https://github.com/astramind-ai/BitMat)。
- **新的 Triton Viz 频道提议**：一名成员提议创建一个新的频道 (#1189607595451895918)-viz，专门供 **Triton visualizer** 项目的贡献者使用，以促进协作。
- **LASP 线性注意力机制闪电来袭**：分享了一个 GitHub 仓库链接，展示了 **LASP 项目**，该项目提供了线性注意力序列并行（Linear Attention Sequence Parallelism, LASP）。相关代码可在 [lasp/lightning_attention.py 文件](https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py)中找到。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py">LASP/lasp/lightning_attention.py at main · OpenNLPLab/LASP</a>：线性注意力序列并行 (LASP)。通过在 GitHub 上创建账号为 OpenNLPLab/LASP 的开发做出贡献。</li><li><a href="https://github.com/astramind-ai/BitMat">GitHub - astramind-ai/BitMat: 对 "The Era of 1-bit LLMs" 中提出方法的高效实现</a>：对 "The Era of 1-bit LLMs" 中提出方法的高效实现 - astramind-ai/BitMat
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 条消息): 

marksaroufim: https://twitter.com/soumithchintala/status/1776311683385880983
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1225857072311046219)** (1 条消息): 

- **介绍 QuaRot：一种新的量化方案**：分享了一篇名为 [QuaRot](https://arxiv.org/abs/2404.00456) 的文章，介绍了一种新的量化方案，能够有效地将大语言模型 (LLMs) 进行端到端 4 bits 量化。这包括所有的权重、激活值和 KV cache，量化后的 LLaMa2-70B 模型保留了 99% 的 zero-shot 性能，损失极小。

**提及的链接**：<a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>：我们介绍了 QuaRot，一种基于旋转的新型量化方案，能够对 LLMs 进行端到端量化，包括 4 bits 的所有权重、激活值和 KV cache。QuaRot 以某种方式旋转 LLMs...

  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1225793515984785538)** (1 条消息): 

- **重温并行编程经典**：一位成员推荐了一个虽然有些年代但依然相关的 [Udacity 课程《并行编程导论》](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)，该课程曾用于其 2013 年的毕业论文。**并行算法与性能**是该课程的核心，对于理解硬件和编程来说，它仍然是一个非常有用的资源。

**提及的链接**：<a href="https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2.">Intro to the Class - Intro to Parallel Programming</a>：此视频是在线课程《并行编程导论》的一部分。点击此处查看课程：https://www.udacity.com/course/cs344。

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225458015751241811)** (2 条消息): 

- **CUDA 矩阵乘法 Kernel 入门代码**：一位成员分享了一个*精简版*的入门代码，专注于矩阵乘法 Kernel，并配有[详尽的注释](https://colab.research.google.com/drive/15mWl0pvuyrriqFEnf1py7TlI9suRsesS?usp=sharing)以确保清晰。他们提到重新审视了之前讨论中的加速方法，但在实现过程中遇到了困难。

**提及的链接**：<a href="https://colab.research.google.com/drive/15mWl0pvuyrriqFEnf1py7TlI9suRsesS?usp=sharing)">Google Colaboratory</a>：未找到描述

  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1225778502385537037)** (1 条消息): 

- **将 Triton Puzzles 移植到 Pallas 的兴趣**：有人请求将 **triton puzzles** 移植到 **Pallas**。建议对于有兴趣探索这种可能性的用户，可以通过 **Triton backend** 来实现。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1225462650423480483)** (1 条消息): 

- **使用 Triton 实现极速注意力机制**：现在 Triton 中提供的 `lightning_attention` kernel 使得不再需要 **flashattn 仓库**来管理跨设备的数据切分。查看 GitHub 上的仓库：[GitHub - OpenNLPLab/LASP](https://github.com/OpenNLPLab/LASP)。

**提到的链接**：<a href="https://github.com/OpenNLPLab/LASP">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>：线性注意力序列并行 (LASP)。可以通过在 GitHub 上创建账号来为 OpenNLPLab/LASP 的开发做出贡献。

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1225500024029581503)** (27 条消息🔥):

- **HQQ 与 GPT-Fast 集成的提案**：*zhxchen17* 建议创建一个 demo 分支，展示如何将 **HQQ** 与 **gpt-fast** 集成。该计划包括在 gpt-fast 上创建一个独立分支，编写量化权重的转换脚本，并针对相关问题进行 Benchmark，同时邀请 *torchao* 团队协作并寻求 Mobius 团队的评审。

- **专注于 Llama 模型**：*mobicham* 和 *zhxchen17* 讨论了将 HQQ 和 gpt-fast 的集成重点放在 Llama 模型上，特别是 *Llama2-7B* (base)，因为它们已经拥有大量的 Benchmark，可以方便进行对比和集成工作。

- **量化层级的细节**：大家对探索更低位宽（bit-level）的量化表现出浓厚兴趣，*zhxchen17* 正在研究 4/3 bit 量化，并分享了 Mixtral 的混合精度模型（由 *mobicham* 发布在 Hugging Face 上）作为参考。*mobicham* 建议将 Llama2 HQQ 转换为 GPT-fast 格式，并评估 4-bit 量化的性能。

- **量化方法的效率**：*mobicham* 提议将 `HQQLinear` 适配到 GPT-fast 的格式，并修改反量化（dequantization）逻辑以匹配 HQQ 的方法。讨论中提到了 HQQ 与现有 int4 kernel 之间反量化逻辑差异的担忧，并探讨了解决这一问题的潜在策略。

- **Kernel 组轴限制 (Kernel Group Axis Constraints)**：关于 group-sizing 的 Kernel 注意事项进行了技术讨论，特别是 gpt-fast 现有的 Kernel 是同时支持 axis=0/1 还是仅支持 axis=1 分组。*mobicham* 询问了是否可以根据 HQQ 反量化权重使用 AO 的逻辑进行 scale/zero 计算，并分析了在现有 Kernel 限制下的灵活性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama">GitHub - meta-llama/llama: Inference code for Llama models</a>：Llama 模型的推理代码。可以通过在 GitHub 上创建账号来为 meta-llama/llama 的开发做出贡献。</li><li><a href="https://gist.github.com/mobicham/8b3147742beb3b302064453a15ced428#file-awq_hqq_test-py-L52">awq_hqq_test.py</a>：awq_hqq_test.py。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu#L912">pytorch/aten/src/ATen/native/cuda/int4mm.cu at main · pytorch/pytorch</a>：Python 中具有强 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L60">gpt-fast/quantize.py at main · pytorch-labs/gpt-fast</a>：代码量少于 1000 行 Python 的简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/quantize.py#L155">gpt-fast/quantize.py at main · pytorch-labs/gpt-fast</a>：代码量少于 1000 行 Python 的简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1225499224251039804)** (17 条消息🔥):

- **建议增强视觉流 (Visual Flow Enhancement)**：一位成员建议添加 **arrows or visual indicators**（箭头或视觉指示器）来展示可视化中的操作方向，并分享了一个快速原型作为概念。
- **提议显示操作过程**：有建议认为，在可视化中像 kernel 一样显示具体操作（例如给输入加 10）会很有帮助。他们分享了来自 `add_mask2_kernel` 的代码片段作为示例。
- **对当前视觉调试工具的担忧**：一位成员对目前的视觉调试工具持保留意见，质疑在图中没有索引的情况下其有用性，并提议在每个元素上显示数值可能更实用。
- **关于交互性与调试的讨论**：讨论了交互性增强方案，包括**增加可交互元素**，例如用户可以悬停在单元格上以检查其数值。
- **可能转向 JavaScript 以增强交互性**：提到了 **Gradio** 的局限性，并讨论了转向使用 JavaScript 框架的可能性，其中 **Three.js** 被提议为一个虽然可能“用力过猛”但很有趣的选项。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1225504686820298752)** (34 messages🔥): 

- **Hamel Husain 的博文引发反响**：频道成员讨论了 [Hamel Husain's blog post](https://hamel.dev/blog/posts/evals/) 中关于评估 AI 系统的见解，强调了其深度以及在构建公司和 AI 功能方面的实际应用价值。
- **为 Datasette 构建 AI 评估**：对话透露了为 Datasette SQL 查询助手插件构建评估系统的意图，强调了用户拥有 **prompt visibility and editability**（Prompt 可见性和可编辑性）的重要性。
- **探索 Prompt 管理策略**：一位成员提出了三种管理大型应用 AI Prompt 的方法：带有独立 Prompt 文件的本地化模式（localization pattern）、带有 Prompt 获取 API 的中间件模式（middleware pattern），以及抽象化 AI 服务管理的微服务模式（microservice pattern）。
- **检查 Cohere LLM 的详细 JSON 响应**：展示了 Cohere LLM 搜索 API 返回信息的深度，并通过一个显示其 JSON 响应的 [GitHub issue comment example](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135) 证明了其效用。
- **对 DSPy 的怀疑与魔幻现实主义**：虽然一些成员对 DSPy 的方法表示怀疑，将其比作将 LLM 变成完全的黑盒，但其他人则接受了这种保留了类似于魔幻现实主义不可预测感的 API 理念。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>：如何构建特定领域的 LLM 评估系统。</li><li><a href="https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135">Support for the web search connector · Issue #2 · simonw/llm-command-r</a>：如果你在 API 调用中添加以下内容：diff --git a/llm_command_r.py b/llm_command_r.py index 7a334cd..e49c599 100644 --- a/llm_command_r.py +++ b/llm_command_r.py @@ -43,6 +43,8 @@ class CohereMessages(...</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>：DSPy：用于编程（而非 Prompting）基础模型的框架。
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1225373204206456872)** (10 messages🔥):

- **Benchmark 开发频道揭示情感智能**：[EQBench.com](https://eqbench.com) 宣布了两个用于评估大语言模型 (LLM) 的**新排行榜**，一个是关于 [Creative Writing](https://eqbench.com/creative_writing.html)（创意写作），另一个名为 **Judgemark**，用于评判模型对创意写作的评分能力（[Judgemark 链接](https://eqbench.com/judgemark.html)）。他们的 Benchmark 被称为**“困难”**级别，要求模型公平地评判来自各种测试模型预生成的创意输出。
- **定义评分范围**：在关于评分标准的讨论中，*.calytrix* 报告称 **-1 到 1 的量表非常适合情感分析**，而对于质量评估，**0-10 的量表**似乎比使用锚点或其他数值表示法更受大语言模型青睐。
- **创意写作 Benchmark 的基础**：创意写作 Benchmark 的**成功创建**归功于使用了 **36 个定义严格的标准**，并强调模糊的判断（例如“给这个故事打 0-10 分”）会导致模型对高质量和低质量内容的**区分度较差**。
- **挖掘 Benchmark 标准**：应相关请求，*.calytrix* 指向了一个**样本输出**（[样本输出链接](https://eqbench.com/results/creative-writing/gemini-ultra.txt)），展示了 **EQ-Bench 创意写作排行榜**中使用的评判标准。
- **通过标准差测量质量**：为了衡量评分量表的质量，*.calytrix* 建议使用模型间**分数的标准差**作为指标，并强调 0-10 的量表通常比 0-5 系统提供**更好的粒度**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eqbench.com/creative_writing.html">EQ-Bench Creative Writing Leaderboard</a>：未找到描述</li><li><a href="https://eqbench.com/judgemark.html">EQ-Bench Judgemark Leaderboard</a>：未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1225518274964357261)** (7 条消息): 

- **共享 COMET 评估脚本**：讨论了使用 *wmt22-cometkiwi-da* 的无参考 COMET 分数，并分享了 [llm_translation 仓库](https://github.com/CrispStrobe/llm_translation) 的 GitHub 链接，其中包含两个用于评估的脚本 `comet_eval.ipynb` 和 `overall_scores.py`。贡献者被要求报告在使用过程中发现的任何严重错误。

- **Cohere 令人惊叹的演示**：[Hugging Face 网站](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus)上一个展示 AI 模型 Grounding 能力的精彩演示受到了称赞，这表明了一项重大进步，将启发未来的模型开发。

- **语言理解能力的进步导致模型过时**：Hugging Face 的模型 **command-r** 在没有经过专门训练的情况下翻译中古高地德语的能力，据称使得数月来教导 LLM 掌握相同技能的工作变得过时。通过与 GPT-4 和 **Nous Hermes 2 Mixtral** 的对比，command-r 提供了明显更优越的翻译。

- **新模型提升了 RAG 使用场景**：讨论指出 **CohereForAI 的模型**在 RAG 使用场景和 Grounding 方面表现出色，并建议开源 C4AI 许可证。这可能会像围绕 **Mistral** 模型建立的生态系统一样，极大地促进开发者活动和投资兴趣。

- **中古高地德语翻译突破**：Command-r 在翻译中古高地德语方面优于 GPT-4 和 Claude 3，提供了准确的翻译，并拥有出色的 Needle-in-a-Haystack（大海捞针）能力——该模型现在是推理过程中实时语言数据库集成的领先竞争者。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>：未找到描述</li><li><a href="https://github.com/CrispStrobe/llm_translation/">GitHub - CrispStrobe/llm_translation</a>：通过在 GitHub 上创建账户来为 CrispStrobe/llm_translation 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1225592652800462848)** (1 条消息): 

- **Mozilla 发布 Beta 版 AI 项目 "Solo"**：Mozilla 推出了 **Solo**，这是一款针对企业家的 AI 驱动网站生成器，承诺提供无代码网站创建和业务增长工具。感兴趣的人士可以**测试 Beta 版本**并提供反馈。点击[此处](https://bit.ly/soloforcommunity)查看。

- **招募早期产品测试人员**：Mozilla **正在为 Solo 招募早期产品测试人员**以提供反馈。有兴趣的测试人员请在指定的讨论频道分享他们的意见和问题。

**提到的链接**：<a href="https://bit.ly/soloforcommunity)">Solo - 免费 AI 网站生成器</a>：Solo 使用 AI 为您的业务即时创建精美的网站。

---

**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1225344344811110461)** (11 条消息🔥): 

- **GPU 分配建议**：为了实现高效的模型运行，建议在 `-ngl` 之前传递 `--gpu nvidia`；对于某些模型，您可能需要手动指定卸载的层数；在 **16GB 4090 Mobile** 上，Q8 量化下只能加载 10 层。

- **Intel oneAPI basekit 利用**：使用 Intel 的 oneAPI basekit 和 `icx` 是因为配套的 `sycl` 代码需要 `onemkl`，这与 Intel 密切相关。

- **每秒 Token 数可见性请求**：一位用户询问是否可以在模型运行期间显示 **tokens per second** 指标。

- **Kubernetes 环境中的模型性能**：Mistral-7B 模型在具有 7 核和 12GB RAM 的 k8s 集群中使用，稳定达到每秒 5 个 token。讨论探讨了 **RAM 大小和速度**、CPU 或 GPU 能力是否是提升性能的瓶颈。

- **Llamafile 可执行文件的恶意软件警报**：有人担心 llamafile-0.6.2.exe 在 Steam 的应用程序上传检查中被标记为恶意软件。[Virustotal 引用](https://www.virustotal.com)显示 0.6.2 和 0.7 版本都被标记了，尽管 0.7 版本的风险评分较低。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.virustotal.com/gui/file/37a39d8970573110c425c3edd1be4b1df6ab32c4a4a38ae6d98ad4728093267e">VirusTotal</a>：未找到描述</li><li><a href="https://www.virustotal.com/gui/file/57a2ad7b2458896e8936f00cd4c91c8b4c919fceab35bfd3f85371b3a84dc935">VirusTotal</a>：未找到描述</li><li><a href="https://huggingface.co/models?library=gguf">Models - Hugging Face</a>：未找到描述
</li>
</ul>

</div>

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1225385680784457768)** (1 条消息): 

- **为 LLM 引入 Mixture-of-Depths**：语言模型的新型 **Mixture-of-Depths (MoD) 方法** 允许动态计算分配，其运行效率与 *MoE transformers* 相当，但仅使用单个专家。它通过 *top-$k$ 路由机制* 管理计算，根据 [研究论文](https://arxiv.org/abs/2404.02258) 中的详细说明，将处理调整到特定的 token 位置。

**提到的链接**：<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算量）分配给特定的...

---

**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=KxOqjKq2VyY

---

**Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/)** (1 条消息): 

carterl: https://arxiv.org/abs/2404.02684

---

**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1225555730619301938)** (2 条消息): 

- **对新 AI 的初步印象**：一位成员分享了来自 [AnthropicAI 的推文](https://twitter.com/AnthropicAI/status/1775979799644934281?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet)，表示他们的初步测试发现该 AI 表现相当不错。
- **注意到延迟问题**：另一位成员认同该 AI 的性能，但提到对于任何链式调用，**延迟都是一个挑战**。

---