---
companies:
- ai2
- hugging-face
date: '2024-02-04T00:54:29.799988Z'
description: '**2024年2月2日的 AI Discord 动态**分析了 **21 个服务器**、**312 个频道**和 **4782 条消息**，预计节省了约
  **382 分钟**的阅读时间。


  讨论内容包括 **Eugene Yan** 发起的对 **AI 工程**挑战的深入探讨，强调了软件工程与数据科学技能之间的重叠。**TheBloke Discord**
  社区讨论了 **MiquMaid**、**OLMo**（由 **AI2** 发布、采用 Apache 2.0 协议的开源 65B 大语言模型）、**Aphrodite**
  模型批处理、**AWQ** 量化，以及 **QLoRA** 和 **LoftQ** 等 **LoRA** 微调技术。


  **LAION Discord** 讨论了 **SSD-1B** 蒸馏问题、利用 **BLIP**、**COCO** 和 **LLaVA** 等标注数据集进行数据质量优化，以及用于提高图像生成中提示词遵循度（prompt
  adherence）的分词策略。其他话题还包括带有水印技术的 AI 安全、用于硬件的超导体和碳纳米管，以及通过 **Hugging Face** 工具部署大语言模型。'
id: c7621e73-b07b-43ec-b2b9-6458dcb3bf6c
models:
- miqumaid
- olmo
- aphrodite
- awq
- exl2
- mistral-medium
- internlm
- ssd-1b
- lora
- qlora
- loftq
original_slug: ainews-the-core-skills-of-ai-engineering
people:
- eugene-yan
title: AI工程的核心技能
topics:
- ai-engineering
- quantization
- fine-tuning
- open-source
- model-deployment
- data-quality
- tokenization
- prompt-adherence
- distillation
- ai-security
- batching
- hardware
- role-playing
---

 


**目录**

[TOC] 


# 第一部分：高层级 Discord 摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **深夜技术对谈**：包括 `@mrdragonfox`、`@coffeevampir3` 和 `@potatooff` 在内的用户就 **MiquMaid** 和 **OLMo** 等大语言模型的性能，以及 3D 打印在 PC 硬件中的潜在应用和碳纳米管的使用展开了生动的讨论。
- **水印与 AI 安全**：对话涉及使用梯度上升（gradient ascent）使模型“取消学习”信息的技术，以及在训练期间从模型中移除深度水印（deep watermarking）的挑战。
- **OLMo 的开源许可**：介绍了 AI2 的 **OLMo GitHub repository**，该库以 **Apache 2.0 license** 提供**开源 LLM** 而备受关注，并提到了一个 65B 模型的训练。
- **AliExpress 上的超导体和纳米管**：超导体材料如**钇钡铜氧 (YBCO)** 和碳纳米管成为关注话题，凸显了它们在 **AliExpress** 上的可获得性。

- **Aphrodite 的功能与局限**：**Aphrodite** 模型因其在 AI horde 中的批处理能力而受到赞誉，但被指出与不同 VRAM 大小的 GPU 不兼容。
- **AWQ 的校准数据集多样性**：围绕 **Automatic Weight Quantization (AWQ)** 最佳校准数据集的讨论强调了数据集多样性的重要性，特别是对于像 **EXL2** 这样的 AI 模型。
- **本地 AI 与角色扮演实践**：讨论了各种 **AI models for role-playing** 的可用性，并指出在使用经过指令微调（instruction-tuned）的模型时，更倾向于使用指令模式（instruction mode）。
- **排行榜与模型使用的伦理**：**Mistral medium model (MoMo)** 在排行榜上的出现引发了关于使用许可不明的模型所带来的影响，以及模型训练中缺乏企业透明度的辩论。

- **量化与微调**：关于使用 **LoRA** 对预训练的 **AWQ model** 进行微调的问题被提出，并讨论了在 **QLoRA fine-tuning** 和推理服务期间对齐量化过程的好处。
- **LoftQ 介绍与量化讨论**：分享了一篇关于 **LoftQ** 的论文，这是一种通过微调 **LoRA and quantizes** 模型来提高性能的量化技术，引发了对其有效性的讨论。

- **internLM 获得认可**：在简短的交流中，`kquant` 推荐 **internLM** 为一个可靠的模型。

- **使用 HF 工具部署 LLM**：`m.0861` 寻求关于通过 **HF spaces** 部署大语言模型的建议，随后讨论了使用 HF 的 inference endpoints 进行 LLM 部署的优势。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **SSD-1B 蒸馏的缺陷**：成员 `@pseudoterminalx` 和 `@gothosfolly` 讨论了 **SSD-1B** 由于从单一微调模型蒸馏而导致的僵化问题，建议使用多个模型可以增强蒸馏在美学方面的表现。

- **通过适当的标注优化数据质量**：讨论强调了使用来自 BLIP、COCO 和 LLaVA 等多样化来源且标注良好的图像，作为提高模型训练中提示词遵循度（prompt adherence）的策略，并提到了输入扰动和数据流水线改进对效能的影响。

- **通过混合编码实现提示词遵循**：围绕 UTF-8 Tokenization 与将 UTF-8 编码合并为单个 Token 的混合方法的优劣展开了辩论，思考了采用类似 **ByT5** 的字节级编码对图像生成的潜在益处。

- **图像生成中的裁剪与上采样**：确定了一种使用裁剪模型权重进行图到图（image-to-image）上采样的有效方法，该方法被认为能够保持场景完整性，尤其有利于高分辨率增强。

- **绕过同行评审**：讨论强调了一种趋势，即研究人员倾向于在博客上发布显著发现，而非传统的期刊，这通常是因为繁琐的同行评审过程，一些人考虑仅通过博客文章详细介绍新型架构。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **AI 工程的演进**：关于软件工程师有效使用 LLM 所需的基本技能以及 AI 工程师这一不断演变的职位的激烈辩论。讨论强调了理解 LLM 的概率性质、评估、调试、数据熟悉度以及从确定性结果向概率性结果思维转变的重要性。提出了 **AI Engineer Continuum**（AI 工程师连续体）的概念，建议了从使用 API 到微调模型的各个阶段。

- **社区增长与学习倡议**：在 LLM Paper Club (East) 中，参与者进行了技术讨论，例如自我奖励（self-rewarding）LLM 的方法论、改进文本嵌入（text embeddings）以及为 RAG 检索长尾知识的价值。关于组建“代码俱乐部”以协作走读代码，以及组建“生产俱乐部”以检查代码/论文实际实现的建议，反映了技术导向型社区的学习愿望。

- **AI 活动与聚会走红**：呼吁参与本地和在线活动，如 LLM Paper Club (East) 和 **AI in Action** 会议。成员们对组建本地用户组表现出极大的热情，例如提议的 LA 见面会和各种社交学习活动，突显了社区成员在分享知识和最佳实践方面的积极态度。

- **资源共享丰富了公会**：成员贡献了大量资源，从使用 AI 的实用指南、评估 LLM 以及构建 LLM 的教学内容，到关于 AI 初创公司策略和商业路演中的 AI 讨论。这表明了在专业和创业领域应用 AI 技术的浓厚兴趣。

- **对工具依赖的担忧**：对 OpenAI 的 Fine-Tuning API/SDK 提出了质疑，并警告潜在的平台锁定风险。讨论倾向于全量微调（full-scale fine-tuning）优于简单 API 交互的优势，反映了工程师对过度依赖第三方平台的担忧。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **将开源模型提升为大规模科学项目**：`@layl` 强调了获得欧盟政府支持进行开源模型训练的可行性日益增加，这与将开源模型视为大规模科学事业的观念相契合。`@stellaathena` 证实了该领域正从微不足道的进展转向实质性的突破，并建议在 LUMI 等高性能计算（HPC）环境中部署 `@layl` 的 ML 库。
  
- **激活函数功效备受关注**：`@xa9ax` 和 `@fern.bear` 等用户引发了关于 GeLU、ReLU、Mish 和 TanhExp 等激活函数的全面辩论，引起了人们对在大模型训练中缺乏这些函数广泛经验测试的关注。尽管 `@ad8e` 此前对一篇推广 Mish 的论文的真实性表示怀疑，但 `@xa9ax` 证实了投稿前的所有实验均已包含在最终出版物中。

- **模型架构基准测试 (Benchmarking Model Architectures)**：对话深入探讨了 `@state-spaces/mamba` 模型与其他架构（如 Transformers++ 和 Pythia）之间的比较。用户 `@ldj` 对比较的基础表示担忧，而 `@stellaathena` 强调需要一套基于公开数据训练的统一模型套件，以进行公平评估。

- **激活函数复杂性探讨**：用户 `@catboy_slim_`、`@fern.bear` 和 `@nostalgiahurts` 思考了激活函数选择的细微影响，讨论了这些函数的缩放如何与其他超参数（hyperparameters）相互作用以影响模型性能。他们剖析了来自 EleutherAI 博客和各种学术论文的实证结果，以解码激活函数与模型训练动态（training dynamics）之间复杂的相互依赖关系。

- **大模型训练背后的法律复杂性**：`@synquid` 强调了围绕模型训练数据源透明度的法律复杂性，指出公开披露训练数据可能会导致知识产权诉讼，从而阻碍科学进步。
  
- **揭秘知识蒸馏 (Knowledge Distillation)**：`@johnryan465` 和 `@xa9ax` 的探究性讨论围绕着训练较小尺寸的模型 B 来模拟较大尺寸模型 A 的 logits，而非直接训练模型 A 的效率优势展开——并思考了 *infinity ngrams paper* 的方法论，以生成用于潜在蒸馏流水线（distillation pipelines）的高性价比模型。

- **解决 MCTS 采样挑战**：`@blagdad` 审视了蒙特卡洛树搜索 (MCTS) 中的探索难题，提到了利用树置信上限 (UCT) 根据不确定性引导探索，而不是采用均匀分支。

- **通过探索提升微调效率**：阐述了利用高效探索进行微调（fine-tuning）的技巧，重点关注构建查询的 Agent 以及根据收到的反馈运行的奖励模型（reward model）。讨论涵盖了双重 Thompson 采样（double Thompson sampling）的优点以及认识神经网络（epistemic neural networks）的应用，详见一篇 [arXiv 论文](http://arxiv.org/abs/2402.00396)。

- **贝叶斯主动学习 (Bayesian Active Learning) 即将发布**：`@fedorovist` 表示 `@322967286606725126` 即将发布贝叶斯主动学习的实现，由于过去有过类似挑战的经验，这引起了 `@johnryan465` 的兴趣。

- **探究 Adam 优化器变体**：`@ai_waifu` 询问是否有研究尝试修改 Adam 优化器，以利用参数的方差（variance）而不是梯度的二阶矩估计（second moment estimation）。然而，回复中并未强调此类研究的具体细节。

- **协作集成视觉-语言模型 (Vision-Language Model)**：`@asuglia` 表达了将视觉和语言支持整合到 lm-harness 中的意图，`@hailey_schoelkopf` 提到 `@chrisociepa` 和 `@1072629185346019358` 是潜在的合作者，并建议社区贡献。

- **MMLU 结果中的标准差对话**：`@baber_` 询问了 *miqu* 模型在 MMLU 结果中出现的显著标准误差（standard errors），`@hailey_schoelkopf` 承认可能需要重新校准评估代码中的标准误差计算。

- **促进 Zero-Shot 评估**：为了在 lm-harness 中强制以 zero-shot 模式运行任务，`@hailey_schoelkopf` 指导 `@asuglia` 将 `num_fewshot: 0` 设置，并引用了相关的 [源代码](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169)。

- **升级分组任务评估方法论**：`@hailey_schoelkopf` 提出了一个更新分组任务标准误差聚合方法的建议，仓库中的一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390) 表明将转向基于合并方差（Pooled variance）的计算。

- **同步视觉-语言模型贡献**：`@jbdel.` 提供了一个用于运行视觉-语言流水线（pipeline）的协作 fork，并安排在 2 月 15 日后将工作移交给 `@asuglia`。协调工作将通过 [时间投票](https://www.when2meet.com/?23484385-k3FqO) 进行。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **LLaVA-1.6 超越 Gemini Pro**：一段 YouTube 视频演示表明，具备增强推理、OCR 和世界知识等特性的 **LLaVA-1.6** 在多个基准测试中表现优于 Gemini Pro。结果和更多细节可以在 [LLaVA 博客](https://llava-vl.github.io/blog/2024-01)中找到。

- **Hugging Face 推出 MiniCPM**：在 Hugging Face 上展示的新模型 MiniCPM 因其潜力和性能引发了关注，讨论将其与 Mistral 等其他模型进行了对比，并期待其 Fine-tuning 结果。

- **ResNet 增长技术应用于 LLMs**：围绕将 ResNet 分类器和 ProGANs 中成功的“增长（growing）”技术应用于 LLMs 的讨论浮出水面，Apple 的 Matroyshka Diffusion 模型证明了这一点。新的 **Miqu** 模型以显著的评分进入 Open LLM Leaderboard，引发了各方反应。

- **Quantization 对 AI 模型性能的影响**：围绕 **miqu-70b** 的对话提到了 Quantization 对模型性能（如拼写准确性）的潜在影响，并引发了关于 Quantization 模型是否应成为某些平台标准的思考。

- **对优化 Tokenization 的持续追求**：工程社区对多语言 Tokenizers 进行了讨论，32000 个 Token 的词汇量可能会限制 LLaMA/mistral 等模型。将 LLaMA 模型适配特定语言的努力（如针对越南语的 VinaLLaMA 和针对中文的 Alpaca）表明了模型国际化的进展。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **质疑地缘政治背景下的 AI 审查**：在关于潜在审查的讨论中，OpenAI 用户 `@bambooshoots` 询问 **ChatGPT** 是否为了遵守中国监管规定而审查回复。另一位用户 `@jeremy.o` 明确表示 OpenAI 不参与此类审查行为。

- **赞扬 AI 中的内容创作自由**：`@jeremy.o` 强调了 **OpenAI 的 DALL·E** 工具，重点介绍了它生成多样化内容的能力，包括 LGBTQI+ 表征，展示了该组织对内容创作自由的承诺。

- **ChatGPT 对话记忆与身份形成**：`@blckreaper`、`@darthgustav.` 和 `@jaicraft` 等用户讨论了与 GPT 模型可能记住之前会话或混淆过去回复相关的挑战。用户希望 **GPT 实体** 拥有独立的记忆和清晰的对话流划分，以提升用户体验。

- **探索不可见的 Text-to-Speech 修改**：`@novumclassicum` 寻求关于在不向用户显示更改的情况下，为 Text-to-Speech 应用进行文本修改的指导。其想法是让 GPT 在提交前内部替换单词，旨在为最终用户提供无缝且不可见的文本修改过程。

- **增强 AI 对话，超越简短摘要**：用户 `@stealth2077` 对 **GPT 倾向于在仅几次交流后就总结**角色对话表示不满。这里的诉求是让 AI 能够持续生成延展的、真实的、由角色驱动的对话，保持逐场描述（play-by-play）的风格，而不是默认进行总结。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **应对 LLM 创建的复杂性**：用户讨论了 LLM 创建的技术层面，指出需要 Machine Learning、PyTorch 等领域的专业知识。同时，人们对使用 LM Studio 插件（如 TTS 和 open interpreters）表现出兴趣，这表明了对更集成、更具交互性的 AI 解决方案的追求。

- **利用 LLM 开辟新途径**：社区成员正在探索用于视觉转文本转换的 **Moondream**，尽管目前存在局限性，但仍表达了将其集成到 LM Studio 中的兴趣。在其他聊天中，围绕 **CodeLlama 70B** 的讨论非常热烈，并为社区提供了实验性预设链接；此外，Mistral AI 对 **Llama 70B** 的微调版本 **miqu** 的泄露也因其在编程任务中的表现而引起轰动。

- **硬件障碍与优化讨论**：深入的讨论集中在为 LLM 优化硬件上，涵盖了双 GPU 设置以及 VRAM 在模型性能中的关键作用等问题。分享了升级到双 RTX 3090 GPU 以提高 70b 模型运行速度的建议，并且对使用 P40 GPU 的新机器配置以实现更好的 LLM 运行效果充满期待。在对 LM Studio 的 CPU 进行基准测试时，见解建议关注 VRAM 使用情况而非核心数量。

- **Docker 困境促使考虑 Conda**：一位用户通过转向 Conda 设置环境解决了 Docker 的问题，突显了容器化环境有时面临的挑战，以及环境管理器在解决这些问题时的实用性。

- **Embedding 的效率与效果之争**：关于存储词 embeddings 的数据库策略的一次简短但深刻的交流，探讨了相似性搜索质量与数据库性能之间的权衡。有人指出，更长的 embeddings 可能会为搜索提供更好的上下文，但可能会对数据库效率产生不利影响。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **Advanced RAG 的探索**：`@andysingal` 展示了他在 Advanced RAG 方面的工作，并分享了一个相关的 GitHub [notebook](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb)，暗示将进行类似于 OpenAI 接口的进一步开发。

- **LLaVA-1.6 超越 Gemini Pro**：LLaVA-1.6 已经发布，声称在分辨率、OCR 和推理方面有所改进，甚至在某些基准测试中超过了 Gemini Pro。欲了解更多信息，请访问 [LLaVA-1.6 博客文章](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/)。

- **Diffusers 0.26.0 发布并包含新视频模型**：新发布的 **Diffusers 0.26.0** 带来了两个新的视频模型，完整说明可在[此处](https://github.com/huggingface/diffusers/releases/tag/v0.26.0)查看。发布代码中的一个实现错误导致了错误的推理步骤，这造成了初始阶段的用户问题。

- **Tokenizer 模式可视化与转换**：`deeeps.ig` 对 Tokenization 模式进行了可视化，并在 [Kaggle notebook](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook) 中进行了演示。此外，还分享了一个将 tiktoken tokenizers 转换为 Hugging Face 格式的 [脚本](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee)，尽管提到了许可方面的担忧。

- **AI 与法律及 Mamba 解析**：一篇 [Medium 文章](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34) 支持了关于法律领域 AI 的持续讨论，随后将进行演示。`@chad_in_the_house` 发布了关于即将举行的 Mamba（一种序列建模架构）演示的消息，相关细节见 [arXiv 论文](https://arxiv.org/abs/2312.00752)，Yannic Kilcher 的 [YouTube 视频](https://www.youtube.com/watch?v=9dSkvxS2EB0) 中有进一步解释。

- **牲畜健康 ML 模型招募志愿者**：**DalensAI** 正在整理一个用于检测牲畜疾病的机器学习数据集，需要志愿者贡献图像和标签。这提供了一个为计算机视觉的实际应用做出贡献的机会。

- **Donut 在不同 Transformers 版本中的表现差异**：据报道，修改后的 donut 模型在 `transformers` 库版本 **4.36.2** 和 **4.37.2** 之间的推理表现存在差异。这意味着在更新依赖项时需要注意潜在的向后兼容性挑战。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **Groq 凭借 LPU 芯片展现竞争优势**：Groq 的定制硬件，被称为本地处理单元 (LPUs)，因其在运行时的本地优化能力而受到认可，表明它们可能与 Nvidia H100 芯片抗衡。然而，Groq 不提供托管服务，对其性能的询问突显了显存限制，更多细节可见 [GroqNode™ Server 产品简介](https://groq.com/wp-content/uploads/2022/10/GroqNode%E2%84%A2-Server-GN1-B8C-Product-Brief-v1.5.pdf)。

- **对 MoMo-72B 模型的关注**：一个名为 MoMo-72B 的 Hugging Face 模型引发了关于模型质量及其“受污染”排行榜分数的争论，并分享了进一步调查的链接 - [MoMo-72B Hugging Face 模型](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO) 以及相关的 [讨论](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO/discussions/2)。

- **同行间的轻松调侃**：出现了一段简短有趣的交流，涉及“betweter”评论和玩笑表达，同时对免费模型访问进行了重要澄清，可以通过 Hugging Face 探索开源选项，而不是通过 API keys。

- **Mistral 部署的协助与澄清**：用户提供了在 Mac 上运行 Mistral 模型的指导和解决方案，指向 [LMStudio](https://lmstudio.ai) 获取合适的下载，并对支持表示感谢。

- **对创新 AI 项目的期待**：社区展示激发了兴奋，从 [socontextual.com](https://socontextual.com) 到名为 "Trying LLaVA-1.6 on Colab" 的 YouTube 演示，该演示强调了 LLaVA-1.6 改进的推理和世界知识 - [YouTube 演示](https://www.youtube.com/watch?v=SEavari8xaU)。此外，一篇受 Terry Pratchett 启发的名为 "Sapient Contraptions" 的同人小说通过 Pastebin 分享 - [Pastebin 上的 Sapient Contraptions](https://pastebin.com/dNRbi7mY)，展示了 AI LLM 软件在故事创作中的创意用途。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **基础模型基础**：新手 `christolito` 询问了 "base perplexity" 模型，引发了 `mares1317` 的回应，提供了协助并指引了进一步的资源。
- **Perplexity 应用动态**：
  - Perplexity Android 应用目前无法使用文档附件功能，该功能目前仅存在于网页版中。
  - 介绍了关于 Copilot 在离线搜索辅助模式下利用 GPT-4 和 Claude 2 模型的细节。
- **会员与 UX 问题**：
  - Perplexity 免费版的限制与 ChatGPT 中的限制进行了对比。
  - Pro 用户 `matthewtaksa` 遇到了延迟和消息重复问题。
- **学习与利用 Perplexity**：
  - `@fkx0647` 报告了通过 API 成功上传文档并进行交互。
  - Perplexity 在内容创作方面的有效性在一个分享的 [YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q) 中得到强调，其表现优于 Google 和 ChatGPT。
- **API 扩展诉求**：`@bergutman` 提议集成 **llava-v1.6-34b** 以支持 API，理由是在 replicate 上使用 1.6 的成本很高，且与 GPT4-V 相比缺乏多模态 API 选项。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **用于 AI 微调的 SuperServer 揭晓**：社区现在可以使用专门用于运行 axolotl 微调的 **8x3090 SuperServer**，`@dctanner` 邀请大家通过私信进行合作。关于该服务器能力的详细信息可以在 `dctanner` 的公告中找到：[The AI SuperServer is live!](https://x.com/dctanner/status/1753013407643562401?s=20)。

- **强调 axolotl Sample Packing 和 BYOD 的优势**：`@nanobitz` 强调了 **axolotl** 相对于 **AutoTrain** 的优势，赞扬了其 "sample packing 和简单的 yaml 共享 + byod"，同时指出 AutoTrain 的 *自动模型选择* 是一个吸引人的特性。

- **FFT 雄心与模型微调**：`@le_mess` 询问了在新 SuperServer 上对 **Mistral** 执行快速傅里叶变换 (FFT) 的情况，`@dctanner` 确认 Mistral 7b 的全量微调正在进行中，并计划测试 Solar 10.7b。

- **关于 GPU 存储和训练能力的深入交流**：`@nafnlaus00` 和 `@yamashi` 讨论了在全量模型微调期间，与存储梯度相关的技术挑战以及多 GPU 所需的通信带宽。

- **vLLM 更新体验**：根据 `@dreamgen` 的报告，与 0.2.7 版本相比，**vLLM** 的 0.3.0 版本在特定工作负载下表现出显著的速度提升。

- **遇到 Mixtral Instruct 提前终止问题**：`@nafnlaus00` 报告称，来自 **Mixtral Instruct** 的 **GGUF Q3_K_M** 有时会提前终止响应，并提到他们正在使用 **llama.cpp** 进行 MoE 推理。

- **Math-Multiturn-100K-ShareGPT 数据集发布**：一个新的数据集 **Math-Multiturn-100K-ShareGPT** 已在 Hugging Face 上发布，其特点是旨在解决数学问题的对话。它提供多达 64 轮对话，并计划在未来包含更复杂的方程式。[点击此处查看数据集](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **RAGArch 简化 RAG 系统部署**：由 `@HarshadSurya1c` 引入的新工具 **RAGArch** 使得设置 **检索增强生成 (RAG)** 系统变得非常方便。它包含一个 [Streamlit UI](https://streamlit.io)，允许轻松选择组件并一键创建 RAG 流水线，正如在[推文](https://twitter.com/llama_index/status/1753478149743284395)中所分享的那样。

- **使用 LlamaIndex 集成 Hugging Face LLM 的全面指南**：`@kapa.ai` 提供了一份将 Hugging Face 预训练语言模型与 LlamaIndex 集成的指南，并附带了[分步示例 notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/huggingface.ipynb)。此外，`@whitefang_jr` 分享了一个 [Colab notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.ipynb)，供用户在 Colab 上结合 LlamaIndex 使用 HuggingFace StableLM。

- **预测模型与 LlamaIndex 的集成选项**：讨论强调了 LlamaIndex 与来自各个平台的预测模型 API 的集成潜力，并为每个特定集成提供了指南。对话还包括了关于运行本地模型以及将 LlamaIndex 与 LangChain 结合使用或独立使用的信息，并提到了 [Ollama](https://ollama.ai/library)（一个优化的本地模型运行器）。

- **Perplexity AI 的引用技术引起关注**：`@tyronemichael` 询问了 **Perplexity AI** 在[其文档](https://docs.perplexity.ai/discuss/65af6285e69072005b83eb05)中提到的快速且先进的引用生成技术，并将其与他们自己使用 **SerpAPI** 和 **LlamaIndex** 的方法进行了比较。然而，即使在询问之后，Perplexity 的方法仍不清楚，而一篇[讨论 Google 论文的推文](https://x.com/cto_junior/status/1710638210009706800?s=20)强调了 Perplexity AI 在事实问答和辟谣方面的能力。

---

## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord 总结

- **利用 NVIDIA 顶尖技术进行优化**：用户 `@zippika` 分享了他们在 **Nvidia 4090 GPU** 上的经验，讨论了使用 `uchar3` 和整数算术进行 RGB 转灰度的高效 CUDA 代码优化。`@jeremyhoward` 和拥有 NVIDIA 经验的 `@vim410` 参与了关于位移（bitwise shifts）的讨论，并欢迎 `@vim410` 加入社区。

- **编译器在位运算优化方面的智能**：在讨论中，`@apaz` 提出编译器在优化过程中可能会自动将除法替换为位移，这是关于 CUDA 代码效率更广泛讨论的一部分。

- **解决 CUDA 内存管理之谜**：`@_davidgonmar` 在 `@lancerts` 和 `@vim410` 的帮助下修复了一个 Bug，并深入了解了 CUDA 环境下正确的 C++ 内存管理技术。

- **Numba 利用共享内存提升速度**：`@stefangliga` 为 `@mishakeyvalue` 分享了 [Siboehm 的文章](https://siboehm.com/articles/22/CUDA-MMM)，其中包含了共享内存缓存等优化技术以及 GPU 矩阵乘法的性能增强。

- **抓住那个缺失的大括号！**：`@ashpun` 在 `@marksaroufim` 的帮助下修复了由语法错误引起的 CUDA kernel `RuntimeError`，他们还处理了与 `GLIBCXX_3.4.32` 版本相关的 `ImportError`，并建议更新 Conda 和正确设置 `LD_LIBRARY_PATH`。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain 文档欠缺，工具获赞**：工程师们对 **LangChain 文档**表示沮丧，认为其令人困惑，并讽刺地指出该工具无法解释自身。与此同时，社区对 [AutoCrew](https://github.com/yanniedog/autocrew) 等贡献充满热情，该工具可为 CrewAI 自动创建团队和任务。

- **对 LangChain 可行性的复杂情感**：虽然一些开发者由于其快速的变化和缺乏模块化而停止使用 **LangChain**，但也有人称赞其节省时间的特性。然而，关于在 `langchain_pg_collection` 中添加 `user_id` 等自定义修改的问题仍未得到明确解决。

- **社区驱动的 AI 教育内容**：分享的教育材料包括关于 *Demonstrate - Search - Predict* 模型的 [Stanford DSP 教程](https://www.youtube.com/watch?v=dTzL8OF_3i0)，由 **@esxr_** 提供的 [Chat UI 适配教程](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5)，以及尽管存在一些 Bug 但仍能使用 LangChain 和 OpenAI API 与 CSV 文件聊天的见解，如[本教程](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2)所示。

- **在生产力工具中利用 AI**：重点介绍的创新包括将 AI 与 Google Workspace 融合的 **[Lutra.ai](https://lutra.ai)**，以及提供简洁、免费的 AI 驱动聊天应用的 **[Tiny Desk AI](https://tinydesk.ai)**，两者都宣传了增强生产力和用户体验的独特功能。

- **讨论多 AI Agent 路由**：讨论了在多个专业 Agent 之间高效路由查询的挑战，并询问了如何更新 `router_to_agent` 函数以获得最佳性能。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **MTEB 排行榜为 AI 照亮前路**：Natureplayer 强调了 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)，引用了语言模型在各种任务上的最新排名和表现。

- **功能请求：轻松浏览**：`@joshcho_` 提出了一个**浏览频道**选项的功能请求，指出由于目前缺乏此类功能，导航和选择感兴趣的频道非常困难。

- **GPT-3.5 因遵循指令而受赞誉**：用户讨论了 **GPT-3.5** 增强的指令遵循能力，`@justahvee` 观察到它在指令密集型任务上的表现有所提高，即使是以牺牲推理能力为代价。

- **详细提示：一把双刃剑**：该频道讨论了详细 Prompting 与延迟之间的权衡，用户 `@res6969` 指出，延长的解释会带来更智能的 AI 表现，但会增加延迟，而 `@sourya4` 讨论了通过实验 `gpt-4-turbo` 来平衡这些因素。

- **思维链（Chain of Thought）提示带来更聪明的 AI**：对话包括关于将思维链（CoT）提示用于异步策略的见解，这可以产生智能响应，以及根据 `@byronhsu` 和 `@res6969` 的报告，重用 CoT 输出进行二次处理步骤的潜力。



---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **Daydream Nation 加入聊天**：用户 `@daydream.nation` 加入了 *[Alignment Lab AI ▷ #general-chat]*，并提到团队的项目已经公开，对尚未参与其中表示遗憾，并推测在 Alignment 背景下更大规模测试人类交互的意图，类似于 **Google 的 Bard**。
- **准备解决难题**：在 *[Alignment Lab AI ▷ #looking-for-work]* 中，`@daydream.nation` 展示了在 **Python, Excel Data Modeling, 和 SQL** 方面的专业知识，并结合其哲学背景以及对利用 AI 解决意识（consciousness）问题的兴趣。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **Infinite Craft 展现元素炼金术**：`@chrisamico` 重点介绍了一款名为 [Infinite Craft](https://neal.fun/infinite-craft/) 的互动游戏，该游戏基于 **llama2** 构建，展示了水、火、风、土等游戏元素，可以通过拖拽合成机制进行组合。
- **游戏创作者获得赞誉**：`@chrisamico` 进一步推荐了 Infinite Craft 创作者的其他游戏，称赞其聪明、有趣且偶尔发人深省，尽管未提供具体名称或链接。
- **认可无尽的乐趣**：`@dbreunig` 肯定了围绕 Infinite Craft 的热度，称其为该类别的优秀范例，而 `@bdexter` 则透露了该游戏的成瘾性，显示出极高的参与潜力。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **德国 Embedding 模型超越基准测试**：[**@damian_89_** 的推文](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19) 讨论了 **@JinaAI_ 的 jina-embeddings-v2-base-de** 和 **@BAAIBeijing 的 bge-m3** 在企业数据测试中的卓越表现，其中 **BGE** 被强调为特别有效。
- **呼吁定量评估**：**@devnull0** 强调需要针对合适的指标测试 Embedding 模型，但未指明具体使用哪些指标进行评估。
- **RAG 评估指南发布**：**@devnull0** 提供的 [GitHub notebook](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb) 为评估检索增强生成（RAG）系统提供了方法论指导。
- **博客深入探讨 RAG**：一篇 [详细的博客文章](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex) 补充了该 notebook，解释了如何使用 LlamaIndex 和专门定制的测试数据集来评估 RAG 系统的 Encoder 和 Reranker 组件。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **LLaVA 1.6 发布**：.mrfoo 宣布了 [LLaVA 1.6](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/) 的发布，并指向了官方发布说明和文档。

- **闲聊频道的 AI 热点**：Pradeep1148 在 off-topic 频道分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=SEavari8xaU)，该视频似乎与 AI 相关，但缺乏相关的背景介绍或讨论。


---

# 第 2 部分：按频道详细总结与链接



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1202888108329275402) (1441 条消息🔥🔥🔥): 

- **深夜讨论生活与技术**：`@mrdragonfox`、`@coffeevampir3` 和 `@potatooff` 等参与者进行了深夜长谈，内容涵盖从 MiquMaid 和 OLMo 等大语言模型的性能，到 3D 打印 PC 硬件以及碳纳米管应用的推测性可能性。
- **模型水印技术**：`@turboderp_` 和 `@selea` 讨论了使用梯度上升（gradient ascent）使模型“遗忘”不需要的信息的想法，以及在训练期间为模型添加水印的概念，并声称水印可以深植于模型内部，以至于发现并移除它们几乎是不可能的。
- **OLMo 的进展**：`@drnicefellow` 介绍了 AI2 的 OLMo GitHub 仓库，强调了其作为带有 Checkpoints 的完整开源 LLM 的潜力，并提到 65B 模型的训练正在进行中。值得注意的是，他们的模型采用 Apache 2.0 许可证。
- **Academicat 与量子**：用户讨论了 Academicat 处理超长论文的能力，并触及了超导体等量子材料在特定条件下的工作原理。
- **探索超导体和纳米管**：在未来技术和材料科学的背景下，`@selea`、`@rtyax` 和 `@spottyluck` 谈论了钇钡铜氧 (YBCO) 等超导体材料和碳纳米管，并提到在 AliExpress 等平台上购买这些材料的便利性。

**提及的链接**：

- [Miau Cat GIF - Miau Cat Meow - Discover &amp; Share GIFs](https://tenor.com/view/miau-cat-meow-gif-9406008167044251375): 点击查看 GIF
- [Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs](https://tenor.com/view/tkt-smart-gif-20642718): 点击查看 GIF
- [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe/): 未找到描述
- [Enterprise Scenarios Leaderboard - a Hugging Face Space by PatronusAI](https://huggingface.co/spaces/PatronusAI/enterprise_scenarios_leaderboard): 未找到描述
- [mlabonne/phixtral-2x2_8 · Hugging Face](https://huggingface.co/mlabonne/phixtral-2x2_8): 未找到描述
- [Yes Lawd GIF - Yes Lawd My Precious - Discover &amp; Share GIFs](https://tenor.com/view/yes-lawd-my-precious-gif-13073178): 点击查看 GIF
- [Creepy Talking Cat 🙀](https://www.youtube.com/watch?v=ddroHMg96HA): 我把这段视频做成了一首完整的歌！在这里观看：youtu.be/WLryCXyjL_0
- [nVidia Hardware Transcoding Calculator for Plex Estimates](https://www.elpamsoft.com/?p=Plex-Hardware-Transcoding): 未找到描述
- [Thinking Christian Bale GIF - Thinking Christian Bale Patrick Bateman - Discover &amp; Share GIFs](https://tenor.com/view/thinking-christian-bale-patrick-bateman-american-psycho-mad-gif-18161559): 点击查看 GIF
- [diable/enable CUDA Sysmem Fallback Policy from command line](https://gist.github.com/itsdotscience/4e29dca91f010a1873d1083fae94a655): 通过命令行禁用/启用 CUDA Sysmem Fallback Policy - a
- [Crash Course Mix [ RaiZen ]](https://www.youtube.com/watch?v=K3XcrDoc8bQ): 使用的主题：Crash Course - 物理、Crash Course - 解剖学与心理学、Crash Course - 天文学、Crash Course - 哲学、Crash Course - 心理学。我不拥有...
- [The Molecular Shape of You (Ed Sheeran Parody) | A Capella Science](https://www.youtube.com/watch?v=f8FAJXPBdOg): 我爱上了你的成键轨道。支持 A Capella Science：http://patreon.com/acapellascience 订阅！https://www.youtube.com/subscription_center?ad...
- [Lisa Su Amd GIF - Lisa Su Amd Ryzen9 - Discover &amp; Share GIFs](https://tenor.com/view/lisa-su-amd-ryzen9-zen2-ryzen-power-gif-14477035): 点击查看 GIF
- [THE TERMINATOR &quot;Final Fight Clip&quot; (1984) Sci Fi Horror Action](https://www.youtube.com/watch?v=72-gVSXt_VU): 《终结者》“最终决战片段” (1984) 科幻恐怖动作片。剧情：1984 年，一名人类士兵的任务是阻止一台坚不可摧的半机械人杀戮机器，两者都...
- [Making YBCO superconductor](https://www.youtube.com/watch?v=sLFaa6RPJIU): 如何制作并测试你自己的 YBCO 超导体。YBCO 的最佳教程资源：http://physlab.org/wp-content/uploads/2016/04/Superconductor_manua...
- [GitHub - bodaay/HuggingFaceModelDownloader: Simple go utility to download HuggingFace Models and Datasets](https://github.com/bodaay/HuggingFaceModelDownloader): 用于下载 HuggingFace 模型和数据集的简单 Go 语言工具 - GitHub - bodaay/HuggingFaceModelDownloader: Simple go utility to download HuggingFace Models and Datasets
- [Jack Tanamen GIF - Jack Tanamen Gecky - Discover &amp; Share GIFs](https://tenor.com/view/jack-tanamen-gecky-incorporeal-gif-21102682): 点击查看 GIF
- [GitHub - gameltb/ComfyUI_stable_fast: Experimental usage of stable-fast and TensorRT.](https://github.com/gameltb/ComfyUI_stable_fast): stable-fast 和 TensorRT 的实验性用法。通过在 GitHub 上创建账户来为 gameltb/ComfyUI_stable_fast 的开发做出贡献。
- [Nvidia RTX A6000 48GB GDDR6 Ampere Graphics Card PNY  3536403379193 | eBay](https://www.ebay.co.uk/itm/225986462736): 未找到描述
- [GitHub - allenai/OLMo: Modeling, training, eval, and inference code for OLMo](https://github.com/allenai/OLMo): OLMo 的建模、训练、评估和推理代码 - GitHub - allenai/OLMo: Modeling, training, eval, and inference code for OLMo
- [I have this paper:Single-cell	multi-omics	defines	the	cell-type	specific	impac - Pastebin.com](https://pastebin.com/EvrWhtJf): Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本的网站。
- [Vanadium Dioxide as a Natural Disordered Metamaterial: Perfect Thermal Emission and Large Broadband Negative Differential Thermal Emittance](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.3.041004#fulltext): 来自传统发射体（如灯泡的暖光）的热辐射随温度升高而增加：灯泡越热，发光越强。逆转这一趋势的热发射体可能会导致...

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1202888060258492496) (227 messages🔥🔥): 

- **Aphrodite Batching 和 GPU 兼容性问题**：`@sunija` 强调了 Aphrodite 的 Batching 能力对 AI horde 等服务的优势，但也指出它在两块不同 VRAM 大小的 GPU 上运行效果不佳。`@goldkoron` 对批量生成选项表示赞赏，并对 GPU 问题表示失望。
- **Aphrodite 中 Context 的使用**：根据 `@sunija` 的说法，Aphrodite 可以存储多个对话的 Context 以实现高效重用。同时，`@keyboardking` 和 `@goldkoron` 对潜在的内存占用表示担忧，并讨论了将处理后的 Context 卸载到 CPU 的可能性。
- **校准数据集讨论和 AWQ Model Cards**：`@dreamgen` 询问了用于 Automatic Weight Quantization (AWQ) 的最佳校准数据集，`@turboderp_` 强调了在 EXL2 的校准数据集中包含多样性的重要性，并强调了数据集多样性对高质量结果的关键作用。
- **用于 Roleplay 的本地 AI**：`@dxfile` 分享了使用不同模型进行 Roleplay 的经验，并表示相比 Chat 模式更倾向于 Instruct 模式，`@sao10k` 反馈称当模型经过 Instruction-tuned 时，Instruct 模式是最佳选择。`@dreamgen` 和 `@firepin123` 询问了关于 koboldcpp 对 iq3_xss 等各种格式支持的澄清。
- **排行榜与 MoMo 模型**：`@mrdragonfox` 等人讨论了 Mistral medium 模型 (MoMo) 在排行榜上出现的争议，涉及缺乏明确许可的模型所带来的问题，以及使用或推广泄露模型的潜在法律风险。`@kaltcit` 和 `@c.gato` 对企业诚信和模型训练细节的保密性提出了批评观点。

**提到的链接**：

- [TheBloke’s gists](https://gist.github.com/TheBloke)：GitHub Gist：通过在 GitHub 上创建账号来收藏和 fork TheBloke 的 gists。
- [Importance matrix calculations work best on near-random data · ggerganov/llama.cpp · Discussion #5006](https://github.com/ggerganov/llama.cpp/discussions/5006)：正如我之前提到的，我担心 wikitext 风格的校准数据 / 缺乏多样性的数据在进行 Importance matrix 计算时，可能比更“随机”的数据效果更差……
- [TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ · Hugging Face](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ#provided-files-and-awq-parameters)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/o8n4gcejJS)：未找到描述

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1202905015489134642) (8 messages🔥): 

- **探索 Quantization 和 LoRA Fine-Tuning**：`@dreamgen` 询问如果计划稍后进行量化，使用 LoRA 对预训练的 AWQ 模型进行 Fine-Tuning 是否会比使用 Base 模型效果更好。`@dirtytigerx` 澄清说，虽然 AWQ 与标准的 QLoRA 不同，但没有证据表明其表现更好。
- **关于 QLoRA 方法论的澄清**：针对 `@dreamgen` 的提问，`@dirtytigerx` 将 AWQ 与普通的 QLoRA 进行了对比，强调 QLoRA 通过 `bitsandbytes` 使用 `load_in_4bit`，而 AWQ 采用不同的 Quantization 方法。
- **引入 LoftQ：弥合量化差距**：`@dreamgen` 分享了一篇讨论 LoftQ 的[论文链接](https://arxiv.org/abs/2310.08659)，这是一种在微调 LoRA 的同时对模型进行 Quantization 的技术，旨在提高下游任务的性能。
- **辩论 Quantization 与 Fine-Tuning 的概念**：`@dreamgen` 建议在 QLoRA Fine-Tuning 和推理服务期间对齐 Quantization 过程可能会带来好处，但 `@dirtytigerx` 对该论文结果的广泛可复现性表示怀疑。

**提到的链接**：

[LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659)：Quantization 是服务大语言模型 (LLMs) 不可或缺的技术，最近已应用于 LoRA Fine-Tuning。在这项工作中，我们关注 Quantization 和 L...

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/) (1 messages): 

kquant：internLM 是一个扎实的推荐。
  

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1203085225274384384) (2 messages): 

- **寻求 LLM 部署建议**：`m.0861` 询问了使用 [HF (Hugging Face) spaces](https://huggingface.co/spaces) 服务部署 **Large Language Models (LLMs)** 的最佳实践，暗示可能将该服务用于此类用途。
- **探索用于 LLM 的 HF Inference Endpoints**：随后不久，`m.0861` 认为 **HF 的 inference endpoints** 可能是部署 LLM 更合适的工具，建议将重点转向该功能。
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1202989457339588618) (380 messages🔥🔥): 

- **关于 SSD-1B 的蒸馏见解**：`@pseudoterminalx` 评论说 SSD-1B 由于是从 fine-tuned 模型蒸馏而来的，因此缺乏灵活性。`@gothosfolly` 表示赞同，并提到从多个 fine-tuned 模型进行蒸馏可以增强美感。

- **关于数据质量的标注讨论**：在由 `@pseudoterminalx` 和 `@gothosfolly` 主导的详细交流中，他们讨论了使用带有正确标注的图像训练模型以增强 prompt 遵循能力（prompt adherence）的策略。`@pseudoterminalx` 报告称使用了 BLIP、COCO 和 LLaVA 等图像源的组合，应用了输入扰动（input perturbations），并解决了调整大小和裁剪等 data pipeline 问题，以提高训练效率和数据质量。

- **增强 Prompt 遵循的技术**：`@pseudoterminalx` 和 `@gothosfolly` 辩论了使用 UTF-8 tokenization 进行文本编码的价值，以及一种将 UTF-8 编码组合成单个 token 的混合方法。他们考虑了使用 ByT5 的 byte-level 编码模型是否能在图像生成方面提供优势，特别是在处理文本方面。

- **用于图像上采样的裁剪模型**：`@pseudoterminalx` 和 `@astropulse` 之间的对话强调了使用裁剪后的模型权重进行 image-to-image 上采样的好处。他们指出，这种方法有助于保持场景完整性，并且在更高分辨率的上采样中似乎非常有效。

- **解决 VAE 中的全局信息问题**：由 `@drhead` 发起的一项讨论考虑了视觉编辑模型（如 StyleGAN3 和 SD VAE）通过生成图像中的强区域泄露全局信息的问题。`@thejonasbrothers` 也参与了进来，提出了多种抵消这种影响的方案，并强调需要具体证据而非仅仅是理论推导。

**提到的链接**：

- [Google's AI Makes Stunning Progress with Logical Reasoning](https://youtu.be/NrNjvIrCqII?si=Q4hcfBZ__yPnu9ip)：🤓在 Brilliant 上了解更多关于人工智能的信息！➜ 前 200 名使用我们的链接 https://brilliant.org/sabine 的用户将获得年度高级订阅 20% 的折扣...
- [Sadako The Ring GIF - Sadako The Ring Ringu - Discover & Share GIFs](https://tenor.com/view/sadako-the-ring-ringu-gif-14695335)：点击查看 GIF

  

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1203084179122819152) (24 条消息🔥): 

- **Cosine Annealing 退居二线**：`@top_walk_town` 分享了他们对一份挑战 Cosine Annealing 有效性的新报告的惊讶，并将其描述为“过山车”。该报告可通过 [Notion](https://shengdinghu.notion.site/) 访问。

- **研究成果倾向于发布在博客而非期刊**：`@chad_in_the_house` 等人认为值得注意的是，由于同行评审过程的繁琐，重要的研究发现通常在博客文章中分享，而不是通过传统的学术出版渠道。

- **新颖架构跳过传统出版**：`@mkaic` 正在考虑通过博客文章发布他们正在研究的一种新颖架构的信息，并对学术出版的现状表示沮丧。

- **Machine Learning 研究中的“低垂果实”**：`@mkaic` 提到，Machine Learning 研究通常只是将众所周知的技术应用于新数据集，这已经变得乏味，并使研究领域充斥着大量增量式论文。

- **行业经验优于学术出版物**：`@twoabove` 讲述了他们在数据竞赛中的实际成就和行业联系如何提供了学术论文无法提供的机会，暗示在顶级期刊上发表论文的影响力正在减弱。

**提到的链接**：

- [How Did Open Source Catch Up To OpenAI? [Mixtral-8x7B]](https://www.youtube.com/watch?v=PYZIOMvkUF8)：立即使用此链接报名参加 GTC24！https://nvda.ws/48s4tmc。关于 RTX4080 Super 的抽奖活动，详细计划仍在制定中。不过...
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1203011101193801728) (158 条消息🔥🔥): 

- **定义 AI Engineer 技能**：@eugeneyan 征求了关于软件工程师有效使用 LLM 所需技能的建议，引发了关于理解 LLM 的概率性本质以及评估（evaluation）、调试（debugging）和数据熟悉度重要性的讨论。人们认识到传统软件工程与 AI Engineer 角色之间存在差异，对于调用 LLM API 是否会将 SDE 塑造成数据科学家角色存在各种观点。
  
- **AI Engineer 的连续谱**：包括 @eugeneyan 和 @swyxio 在内的社区成员讨论了 AI 工程专业知识的各个阶段，从使用 API 和快速原型设计到微调（fine-tuning）模型。核心焦点在于工程师需要实现的思维转变：从确定性结果转向概率性结果，并有效地处理海量数据。
  
- **AI 领域的技能组合焦点**：@coffeebean6887 和 @eugeneyan 讨论了行业中职位名称与实际技能组合的重要性，考虑将范围从传统的 SDE 扩展到数据工程师和分析师等其他角色。大家达成共识，认为适应能力和快速学习不断演进的 AI 最佳实践比具体的职位名称更重要。

- **CUDA 学习探索**：@420gunna 和其他社区用户思考了学习 CUDA 对未来职业前景的价值，并将其与流行技术的吸引力以及 LLM 领域中深入的 CUDA 知识的稀缺性进行了对比。 

- **对 OpenAI Fine-Tuning API 的担忧与好奇**：@dtflare 提出了关于使用 OpenAI Fine-Tuning API/SDK 经验的问题，@swyxio 对潜在的平台锁定（lock-in）表示怀疑，并建议除非有显著收益，否则应在微调方面“走到底”，而不是使用简化的 API。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200548371715342479/1203084424221171773)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=nOxKexn3iBo&t=7s)：我以前觉得编写 CUDA 代码相当可怕。但后来我发现了一些技巧，实际上让它变得非常容易上手。在这个视频中，我介绍了...
- [Tweet from Eugene Yan (@eugeneyan)](https://x.com/eugeneyan/status/1753445305545298314)：尝试编写一份 JD 来招聘构建 LLM 应用的软件工程师。除了进行 REST 调用之外，还有什么是必不可少的？我能想到的一些：• Evals：收集标签并衡量任务性能 • 查看...
- [Buttondown](https://buttondown.email/ainews/archive/ainews-trust-in-gpts-at-all-time-low/)：未找到描述
- [The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer)：涌现的能力正在催生出 Prompt Engineer 之外的新兴职位。此外：欢迎参加 10 月 8 日至 10 日在旧金山举行的首届峰会，与 500 名 AI Engineer 齐聚一堂！
- [GitHub - AbanteAI/rawdog: Generate and auto-execute Python scripts in the cli](https://t.co/Y2DQjpKv6K)：在 CLI 中生成并自动执行 Python 脚本 - GitHub - AbanteAI/rawdog: Generate and auto-execute Python scripts in the cli

---

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1202917746677907506) (2 条消息): 

- **加入 LLM Paper Club (East) 讨论**：`@swyxio` 宣布由 `<@796917146000424970>` 领导的 LLM Paper Club (East) 正在进行中。欢迎感兴趣的朋友[加入讨论](https://discord.com/channels/822583790773862470/1200029657744027658)并关注即将举行的 [AI Engineering Singapore meetup](https://lu.ma/aie-sg)。

- **不要错过 AI in Action**：`@kbal11` 邀请成员参加正在进行的 **AI in Action** 活动，讨论主题为“如何引导新手 / 如何将自己与 AI 骗子（grifters）区分开来”。该环节由 `<@315351812821745669>` 主持，可通过[此处](https://discord.com/channels/822583790773862470/1200548371715342479)访问。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/822583790773862470/1200548371715342479)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/822583790773862470/1200029657744027658)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的朋友和社区保持紧密联系。
- [AI Engineering Singapore meetup · Luma](https://lu.ma/aie-sg)：这是什么？这是一个为对机器学习、LLMs 和所有 genAI 相关事物感兴趣的人准备的聚会，现场提供精酿啤酒，氛围轻松。来自 https://latent.space/ 的 swyx 正在回新加坡过 CNY（农历新年）...

  

---


### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/1202916188943032361) (63 条消息🔥🔥): 

- **授予屏幕共享权限**：用户 `@ivanleomk` 确认 `@796917146000424970`（未识别用户）正在处理屏幕共享权限，并建议大家耐心等待。
- **舞台音频故障**：`@ivanleomk` 指示 `@srini5844` 加入舞台以解决音频问题，随后提到由于其自身音频问题，将进行短暂休息。
- **探索 Self Rewarding LLMs**：`@anthonyivn` 提出了关于为 Self Rewarding LLMs 生成偏好对（preference pairs）的方法论问题，随后 `@ivanleomk` 澄清了论文中利用评分形成偏好对的过程。
- **关于改进 Text Embeddings 和 RAG 的讨论**：`@anthonyivn` 分享了不同评分量表实验的心得，并讨论了一篇关于改进 Text Embeddings 的论文 ([Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368.pdf))，该论文正被用于最近的研究中。
- **关于“代码俱乐部”和“生产俱乐部”的想法**：`@j0yk1ll.` 和 `@jevonm` 提议创建一个“代码俱乐部”来共同走读代码，以及一个“生产俱乐部”来审查带有实际实现结果的代码/论文，这对工程师和对实际应用感兴趣的人来说非常有价值。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/822583790773862470/1202628983343288330)：Discord 是通过语音、视频和文字进行交流的最简单方式。
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)：Chain-of-thought 提示词结合预训练的大型语言模型在复杂推理任务上取得了令人鼓舞的结果。在本文中，我们提出了一种新的解码策略：self-consistency...
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)：检索增强语言模型可以更好地适应世界状态的变化并整合长尾知识。然而，大多数现有方法仅从检索中获取简短的连续块...
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述
- [Let's build GPT with memory: learn to code a custom LLM (Coding a Paper - Ep. 1)](https://www.youtube.com/watch?v=5pjNlL533PA)：你以前使用过 LLM，甚至可能微调过一个，但是...你曾经亲手构建过一个吗？如何从零开始，将一项新的研究成果转化为代码...

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1203082427140931594) (134 条消息🔥🔥):

- **问候与排程**：`@alan_95125` 发起了对话，`@kbal11` 提到他们会在更多人到达后开始。
- **期待与时间确认**：包括 `@yikesawjeez` 和 `@nuvic_` 在内的几位参与者对开始时间发表了评论，`@nuvic_` 建议将频道重命名为 "Fridays 1PM" 以匹配活动安排。
- **分享 AI 相关链接**：`@yikesawjeez` 分享了一系列与 AI 相关的文章和博客链接，涵盖了从创立 AI 初创公司到实际 AI 使用案例的主题，其中最长的链接列表包含了诸如 Hitchhiker’s Guide to AI, The Washington Post 和 Towards Data Science 等资源。
- **频道活跃度与热情**：`@eugeneyan` 和 `@coffeebean6887` 等用户评论了观众人数的增加，表明人们对频道活动的兴趣和参与度正在增长。
- **启动地方小组**：有人提议成立洛杉矶（Los Angeles）地方小组，`@juliekwak` 请求创建一个频道，`@coffeebean6887` 标记了洛杉矶聚会的潜在成员，随后 `@swyxio` 为此创建了一个新频道。

**提到的链接**：

- [Symphony – 界面](https://www.symphony.run/blog/interfaces)：像 GPT-3.5+ 这样的模型能够调用函数，另一个有趣的结果是这种能力可以用于在对话中渲染可视化界面。
- [Gandalf | Lakera – 测试你的提示词技巧，让 Gandalf 泄露秘密信息。](https://gandalf.lakera.ai/)：诱导 Gandalf 泄露信息，亲身体验大语言模型的局限性。
- [People + AI 指南](https://pair.withgoogle.com/guidebook/patterns/how-do-i-onboard-users-to-new-ai-features)：为构建以人为本的 AI 产品团队提供的工具包。
- [GitHub - uptrain-ai/uptrain：你的开源 LLM 评估工具包。获取事实准确性、上下文检索质量、语气等评分，以了解你的 LLM 应用质量](https://github.com/uptrain-ai/uptrain)：你的开源 LLM 评估工具包。获取事实准确性、上下文检索质量、语气等评分，以了解你的 LLM 应用质量 - GitHub - uptrain-ai...
- [共同构建更好的软件](https://cs50.ai/chat)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。
- [我作为骗子的生活](https://www.swyx.io/con-man-life)：信心是一把双刃剑。我在金融界工作时曾利用信心，现在我所到之处都能看到它。
- [我们为什么创立 Parcha ](https://www.hitchhikersguidetoai.com/p/why-we-founded-parcha)：深入探讨我们为什么在 Parcha 构建 AI Agent，以增强金融科技领域的合规和运营团队。
- [如何使用 AI 做实事：一份新指南](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-practical-stuff)：人们经常问我如何使用 AI。这是一份包含大量链接的概览。
- [关于 AI，每个人都搞错的 3 件事](https://www.washingtonpost.com/technology/2023/03/22/ai-red-flags-misinformation/)：随着 AI 工具的普及，人们正努力区分事实与虚构。
- [如何谈论 AI（即使你对 AI 了解不多）](https://www.technologyreview.com/2023/05/30/1073680/how-to-talk-about-ai-even-if-you-dont-know-much-about-ai/)：此外：在 AI 时代捕捉不良内容。
- [什么是 AI Agent？](https://serokell.io/blog/what-are-ai-agents)：在这篇文章中，你将了解什么是 AI Agent 以及它们真正的能力。你还将学习如何构建适合你目标的 AI Agent。
- [AI Agent 基础：让我们一步步思考](https://www.jonstokes.com/p/ai-agent-basics-lets-think-step-by)：介绍 AgentGPT、BabyAGI、LangChain 以及由 LLM 驱动的 Agent 革命背后的概念。
- [向商务人士推销人工智能](https://towardsdatascience.com/pitching-artificial-intelligence-to-business-people-f8ddd8fb2da2)：从万灵丹综合症到一线希望。
- [公关人员应该（不应该）如何推销 AI 项目](https://thenextweb.com/news/how-pr-people-should-not-pitch-ai-projects-syndication)：对于人工智能社区来说，这是一个激动人心的时刻。对该领域的兴趣正以加速的态势增长，学术和专业机器学习课程的注册人数正在飙升……
- [向客户普及机器学习和 AI 知识 – Andy McMahon](https://electricweegie.com/articles/educating-clients/)：未找到描述
- [如何发布你真实的 AI](https://matt.sh/ai-how-to-announce)：未找到描述
- [未找到标题](https://dev.to/builderio/dont-build-ai-products-the-way-everyone-else-is-doing-it-9a7)：未找到描述
- [高效 AI 商业项目的 7 个习惯](https://towardsdatascience.com/7-habits-of-highly-effective-ai-business-projects-6ced590e6db8?gi=e4b47a172d38)：优秀与卓越的 AI 商业项目之间有什么区别？在组织中开展 AI 工作时，这里有 7 件事需要考虑。
- [如何说服风险投资家你是人工智能专家](https://medium.com/machine-learning-in-practice/how-to-convince-venture-capitalists-you-re-an-expert-in-artificial-intelligence-39d5edaca290)：如果你喜欢这篇文章，请查看 Robbie 的另一篇：风险投资家说“不”的 15 种方式。
- [在 2023 年启动你的新 AI 初创公司 — 构建更好的团队](https://buildingbetterteams.de/profiles/brian-graham/navigating-ai-businesses)：在过去的几个月里，越来越多的人向我询问对他们 AI 商业想法的看法，并寻求在该领域导航的帮助。这篇文章涵盖了我对该领域的大部分想法……

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1202927824953024564) (161 条消息🔥🔥)

- **将开源模型视为大规模科学项目**：`@layl` 讨论了在欧盟国家集群上进行开源模型训练获得政府支持变得越来越容易，这与将开源模型视为大规模科学项目的趋势相一致。同时，`@stellaathena` 证实了过去几年这一趋势从无到有并取得了一定进展，并建议 `@layl` 的 ML 库未来可能在 LUMI 等 HPC 环境中应用。

- **激活函数分析与 OpenAI 的 Mish 实验**：在关于 GeLU、ReLU、Mish 和 TanhExp 等激活函数的广泛讨论中，`@xa9ax`、`@fern.bear` 等人交换了见解和研究，强调了在大模型训练中缺乏对不同激活函数的广泛实证测试。`@ad8e` 对一篇偏向 Mish 的论文的真实性表示怀疑，但在 `@xa9ax` 确认所有提交前的实验都已包含在发表的手稿中后消除了疑虑。

- **Transformer++ 与 Mamba 模型探讨**：围绕 `@state-spaces/mamba` 模型及其与 Transformer++ 和 Pythia 等其他架构的比较产生了一些疑问。`@ldj` 和 `@baber_` 强调了对基准（baselines）和比较的担忧，而 `@stellaathena` 指出缺乏在开放数据上训练的标准模型套件来进行公平比较。

- **激活函数影响的多样化观点**：用户 `@catboy_slim_`、`@fern.bear` 和 `@nostalgiahurts` 就激活函数选择的微妙影响发表了看法，例如规模与其它超参数（hyperparameters）的交互及其对函数性能的影响。讨论中引用了 EleutherAI 博客和研究论文的实证结果，试图揭示激活函数与模型训练动态之间复杂的依赖关系。

- **大模型训练的法律困境**：`@synquid` 关注了与模型训练数据源透明度相关的法律复杂性，认为公开披露训练数据可能会招致知识产权诉讼，从而阻碍科学进步。

**提到的链接**：

- [Activation Function Ablation](https://blog.eleuther.ai/activation-fns/): 类 GPT 自回归语言模型中激活函数的消融实验。
- [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681): 我们提出了 $\textit{Mish}$，一种新型的自正则非单调激活函数，其数学定义为：$f(x)=x\tanh(softplus(x))$。由于激活函数在...中起着至关重要的作用。
- [A decoder-only foundation model for time-series forecasting &#8211; Google Research Blog](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html): 未找到描述
- [Releasing Transformer++ models · Issue #63 · state-spaces/mamba](https://github.com/state-spaces/mamba/issues/63): 出色的工作！是否可以发布你们的 Transformer++ 基准模型（特别是那些在 Pile 数据集上训练的模型）？
- [TinyGSM: achieving &gt;80% on GSM8k with small language models](https://arxiv.org/abs/2312.09241): 小规模模型具有多种计算优势，然而模型大小对问题解决能力的关键程度仍是一个悬而未决的问题。特别是对于解决小学数学问题，...
- [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://machinelearning.apple.com/research/relu): 拥有数十亿参数的大语言模型 (LLMs) 彻底改变了 AI 应用。然而，它们苛刻的计算需求……
- [Information Theory for Complex Systems Scientists](https://arxiv.org/abs/2304.12482): 在 21 世纪，人类面临的许多关键科学和技术问题都可以被理解为与理解、建模并最终控制复杂系统相关的问题……
- [TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks](https://arxiv.org/abs/2003.09855): 用于实时计算机视觉任务的轻量级或移动端神经网络比普通网络包含更少的参数，这导致了受限的性能。在这项工作中，我们提出了一种新型的……
- [Benchmarking PyTorch’s Native Mish](https://benjaminwarner.dev/2021/07/19/benchmarking-pytorch-native-mish#:~:text=Since%20Mish's%20introduction%2C%20Mish%20has,and%20Accuracy%20of%20Object%20Detection.): PyTorch 1.9 增加了 Mish 的原生实现，这是我在计算机视觉任务中首选的激活函数。在这篇文章中，我在 Tesla V100, Tesla ... 上对原生 Mish 的计算性能进行了基准测试。
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#significance-level): &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020] 的官方仓库 - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#significance-le): &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020] 的官方仓库 - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#summary-of-results-vision-tasks): &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020] 的官方仓库 - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [Partial entropy decomposition reveals higher-order structures in human brain activity](https://arxiv.org/abs/2301.05307): 将人类大脑作为一个复杂系统进行建模的标准方法是使用网络，其中基本的交互单元是两个大脑区域之间的成对链接。虽然这种方法很强大，但……
- [Meet Mish: New Activation function, possible successor to ReLU?](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299/1): 大家好，在今年测试了许多新的激活函数后，我很高兴向大家介绍一个在测试中表现出色的函数 —— Mish。根据论文，Mish 的性能比 ReLU 高出 1.67%……

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1202897345386254377) (16 messages🔥): 

- **寻求知识蒸馏（Knowledge Distillation）见解**：`@johnryan465` 对训练小模型（尺寸 B）以匹配已训练大模型（尺寸 A）的 Logits 所带来的效率提升研究表示兴趣，并将其与直接训练模型 A 进行对比。`@xa9ax` 和 `@johnryan465` 讨论了利用 *infinity ngrams 论文* 的方法论来创建低成本模型，这些模型可能用于蒸馏预训练引导流水线（bootstrap pipeline）。
  
- **MCTS 中的采样挑战**：`@blagdad` 探讨了蒙特卡洛树搜索（MCTS）中的探索问题，强调了使用树置信上限（UCT）根据不确定性引导博弈树探索的潜力，而非均匀扩展。

- **分享用于模型改进的高效探索**：`@xylthixlm` 分享了一篇关于通过人类或 LLM 评分员高效选择微调样本的有趣论文，重点关注生成查询的 Agent 以及基于接收反馈的奖励模型。该论文描述了双重 Thompson 采样和认知神经网络（epistemic neural networks）的效率，可在 [arXiv](http://arxiv.org/abs/2402.00396) 获取。

- **主动学习（Active Learning）实现预告**：`@fedorovist` 提到 `@322967286606725126` 正在完善一个贝叶斯主动学习（Bayesian active learning）实现，由于过去处理过类似问题，`@johnryan465` 对任何可用的草案表示关注。

- **Adam 优化器变体查询**：`@ai_waifu` 询问是否有论文探索过在 Adam 中使用参数的方差而非梯度来进行二阶矩估计。目前没有提到具体的论文作为回应。


**提及的链接**：

[Efficient Exploration for LLMs](http://arxiv.org/abs/2402.00396)：我们提供了证据，证明在收集人类反馈以改进大语言模型时，高效探索具有显著益处。在我们的实验中，一个 Agent 顺序生成查询，同时...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1202904159838277642) (22 messages🔥): 

- **视觉-语言集成**：用户 `@asuglia` 表达了在 lm-harness 中集成视觉和语言模型支持的兴趣。`@hailey_schoelkopf` 提到虽然这目前不是重点，但欢迎贡献，并确定 `@chrisociepa` 和 `@1072629185346019358` 为可能的合作者。  

- **寻求 MMLU 标准误差澄清**：`@baber_` 针对模型 *miqu* 的 MMLU 结果中出现的高标准误差提出了疑问。`@hailey_schoelkopf` 承认评估代码中针对分组的标准误差计算可能需要重新审视。

- **确认 Zero-Shot 配置**：`@asuglia` 询问如何在 lm-harness 中强制以 zero-shot 模式运行任务。`@hailey_schoelkopf` 确认设置 `num_fewshot: 0` 即可实现，并指向相关的 [源代码](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169) 进行说明。

- **分组任务评估的修复与改进**：`@hailey_schoelkopf` 提议更新用于汇总任务组标准误差的方差计算方法，并在 EleutherAI GitHub 仓库提交了 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390)。

- **视觉-语言模型支持的协调**：`@jbdel.` 提供了一个具有可用视觉和语言流水线的 harness 分支，并建议在 2 月 15 日后移交给 `@asuglia`。`@hailey_schoelkopf` 发起了一个 [When2meet](https://www.when2meet.com/?23484385-k3FqO) 以寻找合适的时间进行讨论和协调。

**提及的链接**：

- [lm-evaluation-harness/lm_eval/evaluator.py at 7411947112117e0339fe207fb620a70bcec22690 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [[WIP] Use Pooled rather than Combined Variance for calculating stderr of task groupings by haileyschoelkopf · Pull Request #1390 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390)：此 PR 更新了我们用于汇总任务组标准误差 / 样本标准差的公式。在此 PR 中：公式：结果：hf (pretrained=mistralai/Mistral-7B-v0.1), gen_kwargs: (Non...
- [LM Eval Harness--VLMs - When2meet](https://www.when2meet.com/?23484385-k3FqO)：未找到描述

  

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1202982562344206437) (11 条消息🔥): 

- **LLaVA-1.6 表现优于 Gemini Pro**: `@pradeep1148` 分享了一个名为 *["Trying LLaVA-1.6 on Colab"](https://www.youtube.com/watch?v=SEavari8xaU)* 的 YouTube 视频，展示了 LLaVA-1.6 增强的功能，如改进的推理、OCR 和世界知识，并指出它在多个 Benchmark 上甚至超越了 Gemini Pro。结果和详细信息已在 [LLaVA 博客](https://llava-vl.github.io/blog/2024-01)上提供。

- **臭名昭著的黑客再次出击**: `@itali4no` 发布了一个 [VX Twitter 链接](https://vxtwitter.com/JackPosobiec/status/1753416551066181672)，评论了“被称为 4chan 的黑客”的最新壮举。

- **Apple Vision Pro 产品发布**: 用户 `@nonameusr` 宣布了 Apple Vision Pro 的发布，但未提供任何额外信息或产品链接。

- **AI Doomer 与 e/acc 领导者辩论**: `@if_a` 链接了一场 [YouTube 辩论](https://www.youtube.com/watch?v=0zxi0xSBOaQ)，内容是被称为世界第二著名 AI Doomer 的 Connor Leahy 与 e/acc 运动创始人 Beff Jezos 之间的正面交锋，讨论了技术、AI 政策和人类 Agency。

- **悼念 Carl Weathers**: 用户 `@gabriel_syme` 对 Carl Weathers 的逝世表示哀悼，并发表了悼念声明，但未链接到任何外部新闻源。

**提到的链接**:

- [Trying LLaVA-1.6 on Colab](https://www.youtube.com/watch?v=SEavari8xaU): LLaVA-1.6 具有改进的推理、OCR 和世界知识。LLaVA-1.6 在多个 Benchmark 上甚至超过了 Gemini Pro。https://llava-vl.github.io/blog/2024-01...
- [Explosive Showdown Between e/acc Leader And Doomer](https://www.youtube.com/watch?v=0zxi0xSBOaQ): 世界第二著名的 AI Doomer Connor Leahy 与 e/acc 运动创始人 Beff Jezos 坐下来辩论技术、AI 政策和人类...

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202916357327556640) (17 条消息🔥): 

- **Hugging Face 推出 MiniCPM**: 用户 `@Fynn` 分享了一个关于 MiniCPM 的 Hugging Face 论文链接，这是一个可能引起关注的新模型 ([MiniCPM 论文](https://huggingface.co/papers/2402.00838))。
- **在 Twitter 上测试 MiniCPM**: `@burnytech` 引用了一个 [Twitter 线程](https://twitter.com/abacaj/status/1753207827458396328)，展示了对新 MiniCPM 模型的测试，引发了关于其性能的讨论。
- **对 MiniCPM Benchmark 保持理性的怀疑**: `@mister_poodle` 评论说，虽然 MiniCPM 的得分不错，但在 MMLU Benchmark 上的表现不如 Mistral，目前正在等待该模型在特定任务上的 Fine-tuning 使用情况。
- **模型对比引发讨论**: `@bozoid` 指出，MiniCPM 虽然没有专门针对数学进行训练，但在 GSM8K Benchmark 上获得了 53 分，这令人印象深刻，并强调目前还缺乏与 StableLM 2 等较新模型的对比。
- **Model Merging 的潜力**: 用户 `@bozoid` 表示，鉴于该领域最近的进展，Model Merging 的努力可能会增强 ~2B 规模模型的能力。

**提到的链接**:

[Notion – 笔记、任务、维基和数据库的一体化工作空间。](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202896238312431626) (114 条消息🔥🔥): 

- **关于 LLM 增长（Growing）的辩论**：`@theluckynick` 分享了 [@felix_red_panda 的一条推文](https://x.com/felix_red_panda/status/1752996197940220231?s=20)，讨论了像 ResNet 和 GANs 这样“增长型”模型的成功，并质疑这是否能应用于 LLMs。ResNet 分类器和 ProGANs 受益于这种技术，例子包括 Apple 的 Matroyshka Diffusion 模型。

- **Miqu 的初步印象**：`@weyaxi` 宣布 **Miqu** 以 76.59 的分数进入了 Open LLM Leaderboard。随后 `@nonameusr` 等人将 Miqu 的性能指标（如 ARC 和 MMLU）与其他模型（如 MoMo）进行了对比，对其潜力的反应褒贬不一。

- **autotrain 与 axolotl 微调的权衡**：`@papr_airplane` 询问了使用 **autotrain** 与 **axolotl** 进行 finetuning 时的折衷方案，`@teknium` 建议 sample packing、flash attention 和 prompt 格式选择可能是潜在的差异点。
  
- **探索多语言 Tokenizers**：`@light4bear` 发起了关于 LLMs 和 tokenizers 的讨论，特别是关注 32000 个 token 的词汇表如何限制 llama/mistral 等模型的多语言能力。`@teknium` 提供了一个关于 [VinaLLaMA](https://arxiv.org/abs/2312.11011) 的论文链接（一个针对越南语的开源权重 SOTA 大语言模型），`@light4bear` 提到了将 LLaMA 模型适配中文的努力。

- **排行榜上的量化模型**：围绕量化模型展开了对话，特别是 **miqu-70b**，`@.ben.com`、`@betadoggo` 和 `@nonameusr` 等多位用户讨论了量化对性能、拼写准确性的影响，以及这些模型是否默认在特定平台上运行。

**提到的链接**：

- [Zyphra (Zyphra)](https://huggingface.co/Zyphra): 未找到描述
- [AI News: GPT-4-Level Open Source, New Image Models, Neuralink (And More)](https://youtu.be/PSB_QQTp0GU?si=lslRJ8JexekkpBrx&t=729): 这里是过去一周你可能错过的所有 AI 新闻。在此查看 HubSpot 的 Campaign Assistant：https://clickhubspot.com/xn6Discover More ...
- [VinaLLaMA: LLaMA-based Vietnamese Foundation Model](https://arxiv.org/abs/2312.11011): 在这份技术报告中，我们介绍了 VinaLLaMA，这是一个针对越南语的开源权重、最先进（SOTA）的大语言模型，基于 LLaMA-2 构建，并额外训练了 8000 亿个 token...
- [Tweet from Felix (@felix_red_panda)](https://x.com/felix_red_panda/status/1752996197940220231?s=20): 我们知道“增长型”模型（=在训练期间增加更多参数）对 ResNet 分类器效果很好（@jeremyphoward 很久以前就这么做了）、GANs (ProGAN) 和图像 Diffusion 模型 (Apple&#3...
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1753353836373135415): Miqu 现已登上 🤗Open LLM Leaderboard，分数为 76.59。https://hf.co/152334H/miqu-1-70b-sf 基准测试平均分：76.59 ARC: 73.04 HellaSwag: 88.61 MMLU: 75.49 TruthfulQA: 69.38 Winogr...
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): 一个将日常工作应用融合为一的新工具。它是为你和你的团队打造的一体化工作空间。
- [Chinese-LLaMA-Alpaca/README_EN.md at main · ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md): 中文 LLaMA & Alpaca 大语言模型 + 本地 CPU/GPU 训练部署 (Chinese LLaMA & Alpaca LLMs) - ymcui/Chinese-LLaMA-Alpaca
- [Thealexera Soyjak GIF - Thealexera Soyjak Surprised - Discover & Share GIFs](https://tenor.com/view/thealexera-soyjak-surprised-amazed-wojak-gif-19919803): 点击查看 GIF

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202892422275801120) (16 条消息🔥): 

- **数据越多越好？**：`@stefangliga` 建议保存所有的偏好数据，而不仅仅是首选，并提到 **Data Preferences Optimization (DPO)** 可以利用多个响应的排名，尽管相关实现还很罕见。
- **Hermes 2 的正确 Prompt 格式**：`@mr.fundamentals` 分享了一个用于 [Nous Hermes 2 Mixtral 8x7B DPO model](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) 格式化 Prompt 的代码片段，寻求关于为什么响应中可能会缺失起始字符的建议。
- **检查你的输出**：针对 `@mr.fundamentals` 的问题，`@teknium` 建议打印出格式化后的 Prompt，以帮助调试模型响应中跳过起始字符的问题。
- **避免冗长的回复**：`@teknium` 就如何通过提供具有理想长度的示例对话来 Prompt 模型生成更短的响应向 `@mr.fundamentals` 提供了建议，同时指出这可能会增加 Token 使用量。

**提到的链接**：

[NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO?text=My+name+is+Teven+and+I+am)：未找到描述

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1202911564487725086) (25 条消息🔥): 

- **探索机器学习的稠密语言**：`@pyhelix` 讨论了一种使用 “第 7 位” 来表示 Machine Learning 模型中认知失调的编码方案想法，并思考了应用 **modular forms** 来创建稠密语言的可能性。

- **OpenAI 是否针对中国审查 ChatGPT？**：用户 `@bambooshoots` 询问 OpenAI 是否根据中国法律审查 **ChatGPT** 的响应；`@jeremy.o` 回应并澄清，**OpenAI** 不会出于与中国法规相关的理由审查内容。

- **使用 DALL·E 的内容创作自由**：`@jeremy.o` 强调 **OpenAI** 允许用户创建多样化的表达，包括使用 **DALL·E** 创作 LGBTQI+ 内容，表明了对内容自由的承诺。

- **讨论内容限制的轮廓**：`@bambooshoots` 对 **ChatGPT** 甚至在 *fair use* 范围内也拒绝讨论某些话题表示担忧，`@jeremy.o` 和 `@lugui` 提供了关于内容指南和 ChatGPT 戏剧化倾向的相关背景。

- **关于机器智能的哲学阅读**：`@jimmygangster` 分享了一篇名为《从 Deep Learning 到理性机器》（*From Deep Learning to Rational Machines*）的有趣读物，该文章深入探讨了比较动物和人类思维的哲学研究。

注：其他参与者的消息多为日常问候或简略提及，未提供实质性的讨论点供总结。
  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1203007582990770186) (117 条消息🔥🔥): 

- **@ mentions 混淆与 Bot 协作的潜力**：`@blckreaper` 和 `@darthgustav.` 讨论了使用 **@ mentions** 在不同 GPT 实例之间进行协作的概念，`@jaicraft` 强调希望在对话中存在独立的实体，且不会将过去的回复误认为是其自身的回复。
- **GPT 指令泄露担忧**：`@loschess` 对 GPT 泄露其自定义指令表示担忧，`@solbus` 解释说 GPT 的指令类似于 HTML 中的客户端代码，而 `@bambooshoots` 建议将敏感内容保护在 API 调用操作之后。
- **@ Mentions 集成与 Agent 行为**：`@jaicraft` 和 `@darthgustav.` 辩论了 @ mentions 的功能和局限性，讨论了在一个聊天中存在多个 GPT 实体的可能性，以及对指令进行更好隔离的需求。
- **GPT 回复中的 Bug 和不一致性**：包括 `@_odaenathus`、`@blckreaper` 和 `@loschess` 在内的用户报告了在 GPT 表现、知识文件检索以及拒绝执行某些任务方面的 Bug 和不一致性，暗示 GPT 最近的行为发生了变化。
- **增强实体区分的需求**：由 `@jaicraft` 发起的讨论指出，用户希望 GPT 能作为具有独立记忆和行为的实体运行，而不是作为单一对话流的延续。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1194685637077499925/1195059846236606564)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1202034296823480391)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1001151820170801244/1201968771343061002)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1203028218546749450) (6 条消息): 

- **隐式文本修改请求**：`@novumclassicum` 询问了一种让 GPT 为 **text-to-speech** 修改文本且不在屏幕上显示更改的方法。他们希望 GPT 在内存中完成词语替换后，直接提供可提交的输出结果。

- **在 AI 回复中注入个性**：`@_fresnic` 寻求关于使 API 回复体现某种 **personality** 的建议。他们注意到通过提示 GPT *“像一个 [personality...] 的人那样说话”* 取得了一些成功。

- **减少重复的服务器通信权限请求**：`@novumclassicum` 询问如何防止其 **custom GPT** 在初始同意后重复向用户请求服务器通信权限。

- **维持多角色对话的挑战**：`@stealth2077` 寻求生成多个角色之间 **真实对话** 的技巧。他们难以让 AI 在总结对话之前生成超过三行的对话内容。

- **对详细逐句角色互动的需求**：`@stealth2077` 进一步强调了该问题，表示希望 AI 生成包含每一行内容的完整对话，而不是进行总结。
  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1203028218546749450) (6 条消息): 

- **寻求隐形文本处理技巧**：`@novumclassicum` 正在寻找一种方法，让 GPT 在提交前隐形地执行文本修改，特别是针对 text-to-speech 应用。理想的效果是让 GPT 在内存中替换词汇并提交文本，而不向用户显示修改内容，但他们不确定实现这一目标所需的指令。

- **API 回复中的个性化设置**：`@_fresnic` 尝试通过 API 为 AI 赋予个性，从姓名和兴趣开始并包含在回复中。他们发现像 "talk like someone who is [personality...]" 这样的措辞似乎能改善 AI 的回复。

- **一键连接的难题**：`@novumclassicum` 询问如何防止自定义 GPT 在首次批准后重复请求与外部服务器通信的权限。他们希望复制那种在初始点击后不再触发弹窗的功能。

- **生成角色对话**：`@stealth2077` 寻求关于在叙事中生成角色之间真实且广泛讨论的建议。他们发现 AI 往往在三行之后就开始总结对话，而不是继续对话。

- **期望对话扩展**：延续该话题，`@stealth2077` 表示很难获得超过几行的对话，AI 随后就会转为总结。他们希望 AI 能生成对话的每一行。
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1202909658831130645) (58 条消息🔥🔥): 

- **创建 LLM 的学习曲线**：`@heyitsyorkie` 强调了创建自定义 LLM 的复杂性，指出需要机器学习、PyTorch 等方面的专业知识。
- **探索 LM Studio 插件的可能性**：`@nntb` 询问了适用于 LM Studio 的兼容 AI Agent 和插件，如 TTS 和 open interpreter。`@fabguy` 引导他们查看特定频道以获取更多信息。
- **澄清 LM Studio 的功能**：用户询问了关于同时运行多个 NLP 模型和 Agent，以及将非对话元素集成到聊天机器人中的问题。`@heyitsyorkie`、`@fabguy` 和 `@.ben.com` 对 LM Studio 的功能和局限性进行了澄清。
- **LM Studio 的 Headless 运行**：`@quarky93` 询问了在服务器上运行 LM Studio 后端同时在本地使用 UI 的可行性，`@heyitsyorkie` 回复称目前不支持 Headless 运行模式。
- **模型召回与 Context Window 探索**：`@kirkouimet` 对可用 Context Window 内模型的模糊记忆表示担忧。`@wildcat_aurora` 回复了关于具有更大 195k token Context Window 的 Mixtral 模型的信息，以及运行此类模型所需的硬件要求。

**提到的链接**：

[TheBloke/Mixtral_34Bx2_MoE_60B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral_34Bx2_MoE_60B-GGUF): 未找到描述

  

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1202941411729866762) (19 条消息🔥): 

- **关于 DeepSeek-MoE-16B 支持的咨询**：`@czkoko` 询问 llama.cpp 现在是否支持 **DeepSeek-MoE-16B**，并指出这款专家模型缺乏关注。`@heyitsyorkie` 回复称，如果有 GGUF 量化版本，它应该可以工作，随后分享了测试该模型的意向，并提到其创作者与 Goliath 的创作者相同。

- **用于视觉到文本转换的 Moondream**：`@devrifter` 介绍了 **Moondream**，这是一个擅长将图片转换为文本的模型，已在 Hugging Face 上线。`@heyitsyorkie` 澄清说 Moondream 无法直接在 LM Studio 中运行，但为感兴趣的人提供了[尝试链接](https://huggingface.co/spaces/vikhyatk/moondream1)。

- **CodeLlama 70B 实验**：`@yagilb` 分享了一个针对 **CodeLlama 70B** 的实验性预设，并为那些有兴趣体验最前沿编程模型的人提供了 Discord 链接。

- **通过 "Dolphin" 暗示对无审查模型的兴趣**：`@devrifter` 暗示在寻找无审查模型时可以搜索关键词 "dolphin"，这通常是与此类内容相关的关键词。

- **Mistral Ai 微调 Llama**：`.ben.com` 提到最近泄露了一个名为 **miqu** 的 Mistral Ai 微调版 **Llama 70B**，并提供了[模型链接](https://huggingface.co/miqudev/miqu-1-70b)，称其在编程任务中表现出奇地好。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1201953800634507325/1203097188545077248)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [moondream1 - vikhyatk 的 Hugging Face Space](https://huggingface.co/spaces/vikhyatk/moondream1)：未找到描述
- [dagbs/deepseek-coder-7b-base-v1.5-GGUF · Hugging Face](https://huggingface.co/dagbs/deepseek-coder-7b-base-v1.5-GGUF)：未找到描述
- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b)：未找到描述
- [List Amazinggrace GIF - List Amazinggrace Court - 发现并分享 GIF](https://tenor.com/view/list-amazinggrace-court-watching-scroll-gif-13488308)：点击查看 GIF
- [jartine/llava-v1.5-7B-GGUF at main](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/tree/main)：未找到描述
- [GitHub - deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-MoE 的开发做出贡献。

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1203027656153628692) (32 条消息🔥): 

- **双 GPU 困境**：用户 `@merpdragon` 在由 RTX 3070 和 GTX 1060 组成的双 GPU 配置上遇到了模型加载失败或运行极慢的问题，尽管拥有 80GB 的 RAM。`@heyitsyorkie` 和 `@.ben.com` 讨论了共享内存问题和 VRAM 限制，暗示 GTX 1060 可能对性能没有显著贡献。
  
- **Nvidia 控制面板技巧**：`@.ben.com` 分享了 NVIDIA 控制面板中有一个禁用共享内存的设置，可能有助于解决 `@merpdragon` 的问题。

- **考虑为 LLM 进行硬件升级**：`@heyitsyorkie` 建议，如果想提高 70B 模型的运行速度，应该考虑升级到双 RTX 3090 GPU，因为 3070 运行起来仍然会很慢。

- **大语言模型与受 VRAM 限制的性能**：`@.ben.com` 和 `@rugg0064` 阐明了 VRAM 是运行 LLM 的关键因素，在许多场景下，性能受限于内存（memory-bound）而非计算（compute-bound）。

- **期待新的 LLM 机器配置**：用户 `@wildcat_aurora` 正期待组装一台配备 4 张 P40 GPU 的机器，并考虑使用 Ubuntu 以获得运行 70B 模型更好的性能；而 `@kujila` 则在询问如何使用上一代 AMD 主板和 eBay 上的二手 GPU 组装类似的配置。

- **使用 LM Studio 对 CPU 进行基准测试**：`@goldensun3ds` 计划使用 LM Studio 对不同 CPU 进行基准测试，询问理想的 GPU layer 设置，并考虑使用任务管理器的 VRAM 占用情况作为衡量标准，`@rugg0064` 确认了这一方法。`@.ben.com` 指出，由于内存瓶颈，核心数（Core counts）的影响较小，且 Top P 等参数不会影响推理性能。
  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1203058403925172234) (8 条消息🔥): 

- **模型下载二次确认**：`@yagilb` 询问是否已同时下载了主模型和 vision adapter，并指出这两个组件可能都是必需的。
- **提供截图分析帮助**：`@yagilb` 提议通过分析搜索结果屏幕的截图来协助解决问题。
- **寻求 30b 模型来源的澄清**：`@n8programs` 询问了 30b gguf 的来源，正在寻找获取该模型的具体细节。
- **确认 Llama 库的局部支持**：`@n8programs` 提到 llama.cpp 对 1.6 版本的支持仅是部分的，强调了当前实现中的局限性。
- **Llama 库的性能提升尚不确定**：`@n8programs` 指出，虽然预期 llama.cpp 会有性能提升，但由于尚未完成正确的图像预处理（image preprocessing），这些提升目前还未实现。
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1203142295088799815) (6 条消息): 

- **Docker 问题导致转向 Conda 方案**：`@nntb` 在使用 Docker 时遇到了严重问题，这促使他们安装 Conda 作为替代方案。
- **设置指南不足**：尽管遵循了提供的说明，`@nntb` 在没有额外安装步骤的情况下仍无法解决问题。
- **使用 Conda 创建环境**：`@nntb` 设置了一个 Conda 环境来规避 Docker 的问题。
- **API Key 故障排除**：为了排除故障，`@nntb` 提到必须在 API key 中添加 "EMPTY"，暗示了他们发现的一个可能解决方案。
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1203077819815698442) (2 条消息): 

- **讨论 Embedding 存储策略**：`@drale2k` 询问了在较少的数据库行中存储**较长的词 embedding**与在较多的行中存储**较短的 embedding** 之间的权衡，同时考虑了**相似性搜索质量**和**数据库性能**。
- **上下文在相似性搜索中至关重要**：`@drale2k` 补充说，较长的文本块（chunks）可能通过提供**更多上下文**来产生更好的搜索结果，但也意识到这可能会影响数据库效率和内存使用。
  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1202897534192984075) (45 条消息🔥): 

- **选择合适的文本 Embeddings 模型**：`@bwo_28` 咨询了如何在众多具有不同维度的文本 Embeddings 模型中进行选择。讨论中概述了根据特定需求选择合适模型的挑战，但未推荐具体的模型或标准。
- **集成文档的创新想法**：`@xzuyn` 发起了一场关于是否可以将 HuggingFace 的文档转换为动态格式以辅助训练语言模型的讨论，`@not_lain` 提到了类似的现有方法，并建议这将是该平台的一个有价值的补充。
- **Transformers 升级指南**：在一次故障排除交流中，`@7sinstsugluttony` 确认遵循 `@not_lain` 的建议升级 `transformers` 库解决了他们的问题，展示了社区成员间的互助支持。
- **项目合作公开征集**：`@adityaiiitr` 等 HuggingFace 用户表达了对参与社区项目的兴趣，`@not_lain` 和 `@lunarflu` 等人则在寻找可加入的仓库和计划方面提供了指导。
- **对 AI 峰会的热情**：`@uncleflowerdj` 分享了 2024 年旧金山 GenAI Summit 的邀请，提供了活动详情和折扣码，显然激发了社区对即将举行的 AI 活动的热情和参与度。

**提到的链接**：

- [w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored · Hugging Face](https://huggingface.co/w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored)：未找到描述
- [Hugging Face](https://github.com/huggingface/)：构建未来的 AI 社区。Hugging Face 拥有 190 个可用仓库。在 GitHub 上关注他们的代码。
- [GitHub - IDEA-Research/Grounded-Segment-Anything: Grounded-SAM: Marrying Grounding-DINO with Segment Anything &amp; Stable Diffusion &amp; Recognize Anything - Automatically Detect , Segment and Generate Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)：Grounded-SAM：将 Grounding-DINO 与 Segment Anything、Stable Diffusion 及 Recognize Anything 结合——自动检测、分割和生成任何物体。
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的尖端机器学习框架。
- [GenAI Summit San Francisco 2024](https://www.eventbrite.com/e/genai-summit-san-francisco-2024-tickets-796934722207?aff=eemailordconf&utm_campaign=order_confirm&ref=eemailordconf&utm_medium=email&utm_source=eventbrite&utm_term=viewevent)：本次峰会是生成式 AI 领域顶尖智慧的一次非凡汇聚，体现了未来的精神。#AI_ARE_ALL

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1202897412780462140) (6 条消息): 

- **高级 RAG 实验**：`@andysingal` 重点介绍了他在高级 RAG 方面的工作，使用了来自 HuggingFace 的数据集，并分享了他的 notebook：[llm-course/RAG/Advanced_RAG (1).ipynb](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb)。GitHub 预览包含该仓库的图像、标题和描述。

- **推出具有重大改进的 LLaVA-1.6**：`@meatfucker` 宣布发布 LLaVA-1.6，详细介绍了在分辨率、OCR 和推理方面的显著升级，甚至在某些基准测试中表现优于 Gemini Pro。可以查看详细的博客文章：[LLaVA-1.6 Release Notes](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/)。

- **使用 HuggingFace 创建聊天机器人**：`@yaaqob` 邀请用户尝试在 HuggingFace 平台上创建的新聊天机器人，它了解关于创新和挑战现状的一切。在此访问聊天机器人：[Yaaqob's HuggingFace Chatbot](https://hf.co/chat/assistant/65bd51a7a16aaa191b5b50cf)。

- **深入探讨 Direct Preference Optimization**：`@imcoza1915` 分享了他们撰写的关于 Direct Preference Optimization 的文章，邀请大家对该主题进行反馈和讨论。这是 LinkedIn 上的文章链接，以便进行更深入的交流：[Direct Preference Optimization Article](https://www.linkedin.com/posts/imamashehzad_yesterday-i-was-deciphering-the-exciting-activity-7159323766773157888-rBNC)。

**提到的链接**：

- [LLaVA-1.6: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/)：LLaVA 团队展示了 LLaVA-1.6，具有改进的推理、OCR 和世界知识。LLaVA-1.6 在多个基准测试中甚至超过了 Gemini Pro。
- [llm-course/RAG/Advanced_RAG (1).ipynb at main · andysingal/llm-course](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb)：通过在 GitHub 上创建账号，为 andysingal/llm-course 的开发做出贡献。
- [Innovation Champion - HuggingChat](https://hf.co/chat/assistant/65bd51a7a16aaa191b5b50cf)：在 HuggingChat 中使用 Innovation Champion 助手。

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1202975030577987654) (6 条消息): 

- **Tokenizer 模式揭秘**：用户 `deeeps.ig` 创建了一个 Kaggle notebook，用于比较和可视化 Hugging Face 库中不同语言模型的 tokenization 模式。他们分享了 [notebook 链接](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook)，并暗示未来将推出受 OpenAI 启发而开发的 Web 应用程序。

- **为交易者提供的波动率可视化**：`torres8552` 宣布了一个 *Options Trading: Long & Short Straddle* 应用，允许交易者评估特定期权交易策略的波动率和收益。该应用可在 [Hugging Face's Spaces](https://huggingface.co/spaces/luisotorres/long_and_short_straddle) 进行测试和反馈。

- **音乐生成实现飞跃**：`.bigdookie` 成功演示了名为 *the infinite yt remix* 的微调功能，并分享了该演示的 [Twitter 链接](https://x.com/thepatch_kev/status/1753625904830726456?s=20)。

- **向 Hugging Face 致敬**：`.bigdookie` 对 Hugging Face 表示感谢，感谢其让微调模型的托管变得免费且简单，并强调这极大地简化了他们的工作流程。

**提到的链接**：

- [来自 thecollabagepatch (@thepatch_kev) 的推文](https://x.com/thepatch_kev/status/1753625904830726456?s=20)：终于花时间正确使用了这个 #musicgen 扩展的功能，这里使用了几个不同的微调模型，更接近于 infinite yt remix 的优秀演示。
- [LLM Tokenizer Visualizer](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook)：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据。
- [Long & Short Straddle - Hugging Face Space (luisotorres)](https://huggingface.co/spaces/luisotorres/long_and_short_straddle)：未找到描述。

  

---

### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1202952665361944586) (51 条消息🔥): 

- **Chad 宣布 AI 与法律讲座**：`@chad_in_the_house` 分享了一个 [Medium 文章](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34)链接，该文章将作为法律领域 AI 演示的基础，探讨了为什么用 AI 取代法官很困难。`@chad_in_the_house` 确认将在 Discord 进行语音直播演示，并打算随后在 Youtube 上发布录音。
- **参与即将举行的法律演示**：用户们正在询问如何参加这场 AI 与法律的演示，`@chad_in_the_house` 提供了 Discord 语音频道的[链接](https://discord.com/channels/879548962464493619/907325990236213288)，并提到未来的活动可能会根据演讲者的所在地进行调整。
- **对未来读书会环节的浓厚兴趣**：`@datadev17` 对定期安排在周五的讨论表现出兴趣，`@chad_in_the_house` 确认了这一点，并提到由 `@689634697097117750` 主持的下一场演示将聚焦于 *Mamba*，并链接到了一个用于排程的 [When2meet](https://www.when2meet.com/?23471427-n4DUl) 页面。
- **关于演示视频获取方式的讨论**：用户们正在协调获取演示视频的最佳方式，`@chad_in_the_house` 承诺上传剪辑后的录音链接，而 `@lunarflu` 则凭借 Discord Nitro 的权益提议发布更大的文件。
- **Mamba 论文资源共享**：`@chad_in_the_house` 发布了下周关于 Mamba 演示的详情，提供了一个 [arXiv 链接](https://arxiv.org/abs/2312.00752)和 Yannic Kilcher 的 [YouTube 讲解视频](https://www.youtube.com/watch?v=9dSkvxS2EB0)。`@janimo.` 和 `@swfsql` 分享了额外的 Mamba 资源，链接到更多深入探讨该架构的 YouTube 讲解。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/879548962464493619/907325990236213288)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Mamba - Transformer 的替代品？](https://www.youtube.com/watch?v=ouF-H35atOY)：Mamba 是由 Albert Gu 和 Tri Dao 提出的一种新型神经网络架构。时间戳：00:00 - Mamba - Transformer 的替代品？00:19 - Long Range...
- [Mamba 与 S4 详解：架构、Parallel Scan、Kernel Fusion、Recurrent、Convolution、数学](https://www.youtube.com/watch?v=8Q_tqwpTpVU)：对论文《Mamba: Linear-Time Sequence Modeling with Selective State Spaces》的解释。在本视频中，我将讲解 Mamba，一种新的序列建模架构...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)：基础模型（Foundation models）现在驱动着深度学习中大多数令人兴奋的应用，几乎普遍基于 Transformer 架构及其核心 Attention 模块。许多亚二次时间（subquadratic-time）...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (论文详解)](https://www.youtube.com/watch?v=9dSkvxS2EB0&ab_channel=YannicKilcher)：#mamba #s4 #ssm 大纲：0:00 - 简介 0:45 - Transformers vs RNNs vs S4 6:10 - 什么是状态空间模型（state space models）？12:30 - 选择性状态空间模型 17:55 - Th...
- [法律领域 AI 文献综述](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34)：这篇博客的灵感来自 Laion Discord 服务器的 Owl。感谢参与讨论！在这篇博客中，我的主要目标是探讨为什么……
- [Eric 的演示 - When2meet](https://www.when2meet.com/?23471427-n4DUl)：未找到描述

---

### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1203203120637939764) (10 条消息🔥): 

- **Diffusers 0.26.0 正式发布**：用户 `@sayakpaul` 宣布了 **Diffusers 0.26.0** 的发布，其特色包括两个新的视频模型、多 IP-adapter 推理等，并调侃了一下在周末发布。完整的发布说明可以在[这里](https://github.com/huggingface/diffusers/releases/tag/v0.26.0)找到。

- **故障排除进行中**：用户 `@meatfucker` 报告了新版本示例代码的问题，在 Windows 上尝试运行视频 pipeline 示例时，输出仅为充满噪点的 gif。

- **响亮的警告与寻找答案**：在排除故障期间，`@meatfucker` 分享了控制台中收到的与 `flash attention` 相关的警告，但最初不确定其对输出质量的影响。

- **侦查工作有了回报**：经过调查，`@meatfucker` 发现了问题的根源，指出示例代码错误地将推理步数（inference steps）设置为 1，这很可能导致了糟糕的输出。

- **步数与尺寸的问题**：`@meatfucker` 指出推理步数和解码尺寸（decode size）都被设置成了无效值（1），这与官方 diffusers 文档中指示的更合适的默认值（50 步推理）不同，暗示这可能是发布说明中的一个笔误。

**提到的链接**：

[Release v0.26.0: New video pipelines, single-file checkpoint revamp, multi IP-Adapter inference with multiple images · huggingface/diffusers](https://github.com/huggingface/diffusers/releases/tag/v0.26.0)：此新版本包含两个新的视频 pipeline、更统一且一致的单文件 checkpoint 加载体验，以及支持使用多个参考图像进行多 IP-Adapter 推理...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1203068363602137218) (2 条消息): 

- **Transformer 故障**：`@vikas.p` 遇到了一个问题，修改后的 **donut**（带有自定义 mbart 解码器、gqa 和 moe）在 **transformers 4.36.2** 和 **4.37.2** 上训练表现良好，但推理仅在 **4.36.2** 上正常工作。在 **4.37.2** 版本上推理会导致重复输出，且在发布说明中未找到明确解释。
  
- **DalensAI 为牲畜 ML 模型招募志愿者**：**DalensAI** 创始人、AI 和 Computer Vision 工程师 `@danielsamuel131` 正在寻找志愿者，协助整理用于检测牲畜疾病的机器学习数据集。该公司的项目需要鸡、羊、山羊和牛等动物的图像和标签，包括患病和健康的动物。
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1202933685486231642) (4 条消息): 

- **寻求 GPT 模型的 Tokenizer 支持**：用户 `@janimo.` 询问了 tokenizer 库中对 tiktoken/OpenAI 模型（GPT3/4）的潜在支持，并提到存在一个名为 tiktoken-rs 的 Rust crate。
- **分享 Tokenizer 转换脚本**：`@cakiki` 向 `@janimo.` 回复了一个由 `<@234763281749770243>` 提供的[转换脚本](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee)，该脚本可将 tiktoken tokenizer 转换为 Hugging Face tokenizer 格式，尽管对许可协议（licensing）有一些担忧。
- **对已知资源的确认**：作为回应，`@janimo.` 表示已知晓这些转换后的 tokenizer 文件，没有进一步的询问或背景补充。
- **提出 GPTQ 模型问题**：用户 `.sgp` 对无法在 GPTQ 模型中使用 tokenizer 表示困惑，尽管之前从未遇到过问题。

**提到的链接**：

[Convert tiktoken tokenizers to the Hugging Face tokenizers format](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee)：将 tiktoken tokenizer 转换为 Hugging Face tokenizer 格式 - tiktoken-to-hf.ipynb

  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1202896147337715712) (28 messages🔥): 

- **Groq 芯片引发关注**：用户 `@ethux` 讨论了 Groq 芯片的速度和价格，认为其 PCIe 版本虽然售价高达 2 万美元，但仍可能是 Nvidia H100 的有力竞争替代方案。
- **Groq 硬件重点揭晓**：`@ethux` 和 `@mihaj` 均澄清，Groq 推广的是其定制硬件而非 API 服务，这些硬件被称为 LPU，重点在于其运行时的本地优化能力。
- **寻求关于 Groq 服务的澄清**：`@lukasgutwinski` 询问了 API 服务，并指出在考虑将 Groq 作为潜在解决方案时存在预算限制；`@mihaj` 补充说目前不提供托管服务。
- **Groq 性能咨询**：用户 `@i_am_dom` 针对 Groq 芯片在显存有限的情况下如何保持高速提出疑问，讨论指出 Groq 的卡更多充当加速器角色，详细信息可见 [GroqNode™ Server 产品简介](https://groq.com/wp-content/uploads/2022/10/GroqNode%E2%84%A2-Server-GN1-B8C-Product-Brief-v1.5.pdf)。
- **辩论 Hugging Face 上的模型质量**：用户 `@dillfrescott` 分享了名为 MoMo-72B 的 Hugging Face 模型链接，指出其排行榜分数很高，并思考该模型是否被“污染”（contaminated），同时考虑在更强大的硬件上运行该模型进行测试。

**提及的链接**：

- [moreh/MoMo-72B-lora-1.8.7-DPO · Hugging Face](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO)：未找到描述
- [moreh/MoMo-72B-lora-1.8.7-DPO · New Leader!](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO/discussions/2)：未找到描述

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1202968654615019550) (7 messages): 

- **频道内的友好玩笑**：`@mercercl` 开玩笑地称某人为 "betweter"（自以为是的人），`@ethux` 幽默地回应说他们确实能读懂这条消息。
- **言语可能伤人**：`@ethux` 对 `@mercercl` 的调侃回应了 "not nice :("，表明之前的评论可能有些过火。
- **纯属娱乐**：`@mercercl` 在随后的评论中添加了 "kidding!"（开玩笑的！）来缓和气氛。
- **关于免费模型访问的困惑**：`@ashu2024` 询问如何免费使用开源模型，并对 API key 的流程表示困惑，因为该流程在达到使用限制后似乎指向了订阅服务。
- **提供免费模型访问指南**：`@mrdragonfox` 澄清该 API 与免费模型访问无关，并引导 `@ashu2024` 在 Hugging Face 上寻找免费选项。
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1202923574659125271) (5 messages): 

- **请求帮助**：`@jay.sojitra` 就一个问题寻求帮助，并提供了一个 Discord 频道链接：[Mistral Discord Issue](https://discord.com/channels/1144547040454508606/1202913948253421578)。
- **Mac 支持咨询**：`@patochex` 询问了关于 Mac 的可用性，随后 `@ethux` 提供了解决方案。
- **展示解决方案**：`@ethux` 引导 `@patochex` 使用 [LM Studio](https://lmstudio.ai) 下载适合 Mac 用户的 Mistral 模型。
- **确认解决方案**：`@patochex` 以简短的 "ok good thks !" 表达了对所提供帮助的感谢。

**提及的链接**：

[👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai)：查找、下载并实验本地 LLM

### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1202898992137113620) (15 条消息🔥): 

- **展示的项目尚未开源**：`@sublimatorniq` 分享了 [socontextual.com](https://socontextual.com) 上一个项目的细节，该项目目前尚未开源。
- **对项目发布的强烈期待**：用户 `@atomicspies` 和 `@mrdragonfox` 对展示的项目表示赞赏并期待其发布，称赞其在研究中比传统方法更有用。
- **追求完美的漫长旅程**：`@sublimatorniq` 预计可能需要一年的时间才能将项目推进到理想的状态。
- **"LLaVA-1.6" 性能与 YouTube 演示**：`@pradeep1148` 链接了一个名为 "Trying LLaVA-1.6 on Colab" 的 YouTube 视频，展示了 LLaVA-1.6 版本在推理能力和世界知识方面的提升。
- **《碟形世界》AI 同人小说实验**：`@caitlyntje` 创作了一篇名为 "Sapient Contraptions" 的同人小说，灵感来自 Sir Terry Pratchett，使用了 Mistral 和 Huggingface 等开源 AI LLM 软件，并在 [Pastebin](https://pastebin.com/dNRbi7mY) 上分享了故事。`@amagicalbook` 表示有兴趣学习这一过程，以便用于自己的故事创作。

**提到的链接**：

- [Trying LLaVA-1.6 on Colab](https://www.youtube.com/watch?v=SEavari8xaU)：LLaVA-1.6 具有改进的推理、OCR 和世界知识。LLaVA-1.6 在多个基准测试中甚至超过了 Gemini Pro。https://llava-vl.github.io/blog/2024-01...
- [My Friend](https://youtu.be/U1Ut-rwxKBA?si=I-6IzIpVcxVSzmw2)：由 Legacy Recordings 提供给 YouTube，My Friend · Jimi Hendrix，The Cry of Love ℗ 2009 Experience Hendrix L.L.C.，获得 Sony Music Entert... 的独家许可。
- [SAPIENT CONTRAPTIONSA Discworld Fan Fiction storyInspired by Sir Ter - Pastebin.com](https://pastebin.com/dNRbi7mY)：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1202903397410283590) (43 条消息🔥): 

- **关于 Perplexity 基础模型的新用户查询**：新参与者 `christolito` 询问了关于 "base perplexity" 模型的问题。用户 `mares1317` 表示欢迎并提供了相关链接以获取更多信息，建议查看特定的 Discord 频道以获取详细解释。
- **Android 应用缺少文档附件功能**：`@nqiwbh07r44p` 询问了在 Perplexity Android 应用上上传文档的问题，`@icelavaman` 澄清该功能目前尚未上线，并指出 Web 版本的功能可能更丰富。
- **关于 Copilot 效能和模型的咨询**：`@joshuaa71` 询问了 Copilot 的功能和模型身份。`@icelavaman` 回复了博客文章链接，解释了 Copilot 在专注模式下如何使用 GPT-4 和 Claude 2，以及其在无网络访问情况下的搜索能力。
- **关于 Perplexity AI 功能的澄清**：
  - `@ruspazyyy` 询问 Perplexity 免费版是否有任何限制；`@perplexityai` 回复称存在限制，类似于通常在 ChatGPT 中遇到的情况。
  - `@lukas8a` 寻求在 Perplexity 中从图像中提取文本的方法；`@icelavaman` 提供了相关功能搜索的链接。
- **技术问题与功能请求**：用户正在分享疑虑和建议：
  - `@guocity` 询问 Perplexity 是否可以像 Edge Copilot 那样自动总结长篇文章。
  - `@zwgnr` 报告了最新 iOS 更新中关于复制按钮和响应中代码块背景颜色的潜在 UX 问题。
  - `@matthewtaksa`（Pro 用户）报告了响应生成延迟和消息重复的问题。

**提到的链接**：

- [Introduction | 🦜️🔗 Langchain](https://python.langchain.com/docs/get_started/introduction)：LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot)：访问 Perplexity 博客，查看文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Working On GIF - Working On It - Discover &amp; Share GIFs](https://tenor.com/view/working-on-it-under-construction-gif-23162421)：点击查看 GIF。
- [GitHub - BuilderIO/gpt-crawler: Crawl a site to generate knowledge files to create your own custom GPT from a URL](https://github.com/BuilderIO/gpt-crawler?tab=readme-ov-file#example)：通过爬取网站生成知识文件，从 URL 创建自定义 GPT - GitHub - BuilderIO/gpt-crawler。
- [Perplexity Blog](https://blog.perplexity.ai/faq/what-is-search-focu)：访问 Perplexity 博客，查看文章、公告、产品更新和优化体验的技巧。
- [What is Search Focus?](https://blog.perplexity.ai/faq/what-is-search-focus)：访问 Perplexity 博客，查看文章、公告、产品更新和优化体验的技巧。

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1202898413763563591) (6 条消息): 

- **防诈骗技能**：用户 `@byerk_enjoyer_sociology_enjoyer` 表达了对识别合法在线工作以及在该领域辨别诈骗难度的担忧。
- **文档 API 成功案例**：`@fkx0647` 分享了通过 API 成功上传文档并进行交互的经验，并提到了一个联盟计划。
- **Javascript 学习之旅**：`@stocktown` 简要提到他们一直在学习 *一些 JS 编程*。
- **相比 Google 和 ChatGPT 更青睐 Perplexity**：`@kronokaizen` 分享了[一段 YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q)，标题为“我使用 Perplexity 的次数超过了 Google 和 ChatGPT”，赞扬了 Perplexity 在内容创作中的实用性。
- **对 Perplexity 的热情共鸣**：`@andbamjam` 呼应了 `@kronokaizen` 的观点，称赞 Perplexity 是一个卓越的学习辅助工具，类似于让 *最聪明的人* 回答每一个问题。

**提到的链接**：

[I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q)：本视频的主要观点：“我使用 Perplexity 的次数超过了 ChatGPT、BARD 和 Microsoft Copilots，主要有五个原因，包括它在内容创作中的应用……”

  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203021756772913164) (3 messages): 

- **关于 API 变体的咨询**：`@whodis008` 询问其他人是否在使用 online 变体。`@defektivex` 确认他们确实在使用 online 变体。
- **请求 llava-v1.6-34b API 支持**：`@bergutman` 建议 Perplexity 应该考虑增加对 **llava-v1.6-34b** 的 API 支持，理由是目前缺乏多模态 API 选项，且在 Replicate 上使用 1.6 版本的成本比 GPT4-V 还要高。
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1202912046165786654) (46 messages🔥): 

- **使用 SuperServer 的 AI 微调社区工作**：`@dctanner` 宣布为社区完成了一台 **8x3090 SuperServer**，用于运行新型的 axolotl 微调，并邀请有意合作者私信。分享了 `@dctanner` 的公告链接 [The AI SuperServer is live!](https://x.com/dctanner/status/1753013407643562401?s=20)，以获取有关该服务器能力的更多详情。
- **Sample Packing 和 BYOD**：在关于微调工具的讨论中，`@nanobitz` 提到了 **axolotl** 相对于 **AutoTrain** 的优势，强调了 *"Sample Packing、简单的 yaml 共享以及 BYOD"*。他们还提到 AutoTrain 的 *自动模型选择* 是一个有趣的功能。
- **在不同模型上探索 FFT**：`@le_mess` 询问 **Mistral** 的全量微调（FFT）是否能跑在 `@dctanner` 的 SuperServer 上，`@dctanner` 确认 Mistral 7b 的全量微调（FT）是他们的第一次尝试。他们还考虑根据 `@le_mess` 的请求测试 Solar 10.7b。
- **关于模型存储和训练的技术讨论**：在更深入的技术讨论中，`@nafnlaus00` 表达了对构建 SuperServer 的兴趣，并推测了使用多 GPU 对 Mixtral 等模型进行全量微调的可行性。`@yamashi` 和 `@nafnlaus00` 就存储梯度（gradients）的复杂性和 GPU 之间的通信带宽交换了意见。
- **vLLM 更新带来的性能提升**：`@dreamgen` 报告称，**vLLM** 0.3.0 版本在他们的特定工作负载下比 0.2.7 版本快得多，表明最新更新中有显著的性能增强。

**提到的链接**：

[Damien C. Tanner (@dctanner) 的推文](https://x.com/dctanner/status/1753013407643562401?s=20): The AI SuperServer is live!

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1203011046529310721) (3 messages): 

- **Mixtral Instruct 中意外的推理终止**：`@nafnlaus00` 遇到了一个问题，即 **Mixtral Instruct**（特别是 **GGUF Q3_K_M**）在摘要任务中约有 5% 的概率会**提前终止响应**，意外地切断句子。
- **关于 MoE 推理方法的咨询**：`@nanobitz` 询问 `@nafnlaus00` 使用什么方法进行 MoE（Mixture of Experts）推理，`@nafnlaus00` 回复称他们使用 **llama.cpp**。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1203207596379602964) (1 messages): 

- **新数学数据集提醒**：用户 `@xzuyn` 分享了在 Hugging Face 上可用的 **Math-Multiturn-100K-ShareGPT** 数据集，该数据集包含旨在解决数学问题的对话，响应来自被识别为 GPT 的系统。该数据集每个对话包含多达 64 个响应对，并计划在未来扩展更复杂的方程式。[在此查看数据集](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT)。

**提到的链接**：

[PJMixers/Math-Multiturn-100K-ShareGPT · Hugging Face 数据集](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT)：未找到描述

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1203037412570304542) (1 messages): 

- **使用 RAGArch 简化 RAG 系统设置**：用户 `@HarshadSurya1c` 介绍了 **RAGArch**，它具有 [Streamlit](https://streamlit.io) UI，允许用户轻松选择 **RAG (Retrieval-Augmented Generation)** 系统的组件，包括 LLM、embedding model 和 vector store。只需点击一下即可创建一个完全运行的 RAG pipeline，兼顾了便利性与定制化。[包含更多信息的推文链接](https://twitter.com/llama_index/status/1753478149743284395)。
  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1202950550656913468) (35 messages🔥): 

- **在 LlamaIndex 中使用 Hugging Face LLMs 的指南**：`@kapa.ai` 提供了一个关于如何在 LlamaIndex 中使用 Hugging Face 预训练语言模型（LLM）的全面分步指南，提到了安装所需软件包、设置 token 以及执行本地或远程模型运行。更多指导可以在[详细示例 notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/huggingface.ipynb)中找到。
- **用于 HuggingFace StableLM 的 Colab Notebook**：`@whitefang_jr` 分享了一个 [Colab notebook 链接](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.ipynb)，为在 LlamaIndex 中使用 HuggingFace StableLM 提供实操指导，支持希望在 Colab 上安装和实现 LlamaIndex 的用户。
- **将预测模型与 LlamaIndex 连接**：用户 `@matthews_38512` 和 `@kapa.ai` 讨论了 LlamaIndex 与各种预测模型 API（如 OpenAI、Hugging Face 等）的集成，`@kapa.ai` 指出了针对不同集成的特定指南，并强调了 LlamaIndex 运行 Llama 2 等本地模型的能力。
- **理解 LlamaIndex 与 LangChain 的角色区别**：`@cheesyfishes` 回复了 `@affable_honey_badger`，澄清了 LlamaIndex 可以独立运行或与 LangChain 协同工作，特别强调了 LlamaIndex 对 RAG/上下文增强（context augmentation）的关注。
- **Ollama - 一个优化的模型运行器**：在与 `@affable_honey_badger` 的对话中，`@cheesyfishes` 将 Ollama 描述为一个针对各种模型优化的本地运行器，无需 GPU 即可运行，并作为 llama.cpp 的封装，建议将 Ollama 用于本地测试，而将其他解决方案用于生产部署。

**提到的链接**：

- [HuggingFace LLM - StableLM - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.html)：未找到描述
- [library](https://ollama.ai/library)：在本地启动并运行大型语言模型。
- [Module Guides - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules.html#custom-agents)：未找到描述
- [Module Guides - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules.html#id1)：未找到描述

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1202982195568975952) (6 messages): 

- **探寻 Perplexity AI 引用生成的秘密**：`@tyronemichael` 对 [Perplexity AI 文档](https://docs.perplexity.ai/discuss/65af6285e69072005b83eb05)中提到的 **Perplexity AI** 引用生成表示好奇，并分享了他们自己使用 **SerpAPI** 和 **LlamaIndex** 的方法。然而，他们指出 Perplexity 的输出比他们的基础方法更先进。
- **快速引用检索仍是一个谜**：在后续消息中，`@tyronemichael` 评论了 **Perplexity AI** 引用检索速度之快令人印象深刻，尽管面临网站阻止机器人访问的挑战。
- **API 限制令好奇的开发者受挫**：`@tyronemichael` 注册了 **SerpAPI**，希望能复制 Perplexity 的引用功能，但发现引用尚未包含在其 API 的返回数据中。
- **晦涩的回答让开发者感到困惑**：在尝试直接询问后，`@tyronemichael` 收到了 **Perplexity AI** 关于其引用方法论的晦涩回答，这让他们没有获得清晰的见解。
- **Google 论文强调 Perplexity AI 的优势**：`@tyronemichael` 分享了 `@cto_junior` 的一条[推文链接](https://x.com/cto_junior/status/1710638210009706800?s=20)，讨论了 Google 的一篇论文，该论文评估并赞扬了 **Perplexity AI** 在事实问答和辟谣方面的表现，优于原生 Google 搜索。

**提到的链接**：

- [来自 TDM (e/λ) (@cto_junior) 的推文](https://x.com/cto_junior/status/1710638210009706800?s=20)：很有趣，Google 的一篇论文在他们的新评估中对 @perplexity_ai 进行了评估。它在事实问答和辟谣方面表现非常好，优于原生 Google 搜索。https://arxiv.org/abs/2310.0...
- [无标题](https://serpapi.com)：未找到描述

### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1203033156437934100) (32 messages🔥): 

- **揭秘 GPU 性能怪兽**：用户 `@zippika` 展示了其使用的 **Nvidia 4090 GPU**，并讨论了一段优化的 CUDA 代码，该代码使用 `uchar3` 将 RGB 转换为灰度。代码利用整数算术和位移操作来避免浮点运算。
- **利用位移实现除法**：针对 `@jeremyhoward` 的询问，`@zippika` 解释了代码中的 ` >> 8` 操作是一个位移操作，其效果等同于除以 256，是比浮点除法更高效的替代方案。
- **洞察 CUDA 优化中的效率**：用户 `@apaz` 假设带有优化器的编译器可能会自动将常数除法替换为位移操作，尽管他们尚未测试这一行为。
- **欢迎 NVIDIA 专家**：`@jeremyhoward` 欢迎 `@vim410` 加入 CUDA MODE 社区；`@vim410` 是 NVIDIA 的研究员，与该领域的许多重量级人物都有联系。
- **纠正内存管理失误**：`@_davidgonmar` 就一个与 C++ CUDA 数组类内存管理相关的 bug 寻求帮助。在 `@lancerts` 和 `@vim410` 等其他用户的建议下，通过使用正确的 C++ 内存管理技术解决了该问题。
  

---


### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1202906563015082025) (9 messages🔥): 

- **Numba 中 Shared Memory 速度原理解析**：`@mishakeyvalue` 询问了在使用 Numba 进行 GPU 计算时，使用 **shared memory** 与全局读写相比的速度差异。`@stefangliga` 分享了 [Siboehm 的文章](https://siboehm.com/articles/22/CUDA-MMM)，其中介绍了 CUDA 矩阵乘法的优化以及内存访问合并（memory access coalescing）和 shared memory 缓存等性能特性，并提供了相关 GitHub 仓库链接以供进一步探索。

- **Kernel 代码片段错误处理**：`@ashpun` 在编写 CUDA kernel 代码时遇到了与 `inline_ext` 构建失败相关的 `RuntimeError`。在讨论并查看完整错误详情后，`@marksaroufim` 解决了该问题，识别出 kernel 代码中缺少一个花括号 (`}`)。

- **排查 CUDA 编程中的 ImportError**：在解决上一个问题后，`@ashpun` 遇到了一个 ImportError，提示缺少 `GLIBCXX_3.4.32` 版本，尽管该版本存在于系统中。`@marksaroufim` 建议运行 `conda update --all` 并尝试正确设置 `LD_LIBRARY_PATH` 以解决库路径问题。

**提到的链接**：

[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1202928537074536488) (20 messages🔥): 

- **Elasticsearch 的困扰**：`@shamspias` 询问了关于 **Elasticsearch vector 的异步支持选项**，但在讨论的消息中未获得直接回答。
- **自定义 `langchain` 表**：`@emile_ibr` 研究了 `langchain + pgvector`，并询问是否可以在 **langchain** 创建的表（如 `lagchain_pg_collection`）中添加列（例如 `user_id`）。
- **对 LangChain 文档的挫败感**：包括 `@anthonyj.2048`、`@benjaminbascary` 在内的多位用户表达了对 **LangChain 文档** 的不满，提到文档让他们抓狂，或者讽刺地指出 LangChain 甚至无法解释其自身的使用方法。
- **对 LangChain 褒贬不一的评价**：`@engineered.mind` 已经停止使用 **LangChain 进行开发，理由是其变化过快且缺乏模块化**，但 `@.jkyle` 和 `@akshay_1` 讨论了它的一些节省时间的特性，同时也承认其局限性使得它并不适合所有项目。
- **多 Agent 路由咨询**：`@crtapps` 寻求关于在具有特定功能的多个 Agent 之间路由用户查询的最佳方法的建议，并质疑每次新增功能时都持续更新 `router_to_agent` 的效率。
  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 

rebelsandrobots_97106: 谢谢！
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1202916924875608106) (5 messages): 

- **@yannie 发布的 AutoCrew**: @yannie 分享了一个名为 [AutoCrew](https://github.com/yanniedog/autocrew) 的工具，它可以为 CrewAI 自动创建 crew 和 tasks。消息中包含了该仓库的图片预览及其功能的简要描述。

- **@esxr_ 展示 Chat UI 教程**: @esxr_ 制作了一个教程，演示如何在短短 15 分钟内适配一个开源框架，以提供类似于 ChatGPT 的对话式用户体验。该教程包含一段富有启发性的 [YouTube 视频](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5) 以及配套的 [GitHub 仓库](https://github.com/esxr/repurposed-ollama-webui)，并已分享至社区。

- **Tiny Desk AI 聊天应用**: BrendonJacobs 推广了一个名为 [Tiny Desk AI](https://tinydesk.ai) 的简约免费聊天应用网站。他们分享了该平台的工具链接、文档、关于页面、方案计划以及注册页面。

- **LangChain CSV Agents 教程**: ryannolan 分享了一个关于 LangChain CSV Agents 的 [YouTube 教程](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2)，重点介绍如何使用 OpenAI API 与 CSV 文件进行对话。该视频指南面向初学者，尽管承认存在一些 Bug，但详细解释了整个流程。

- **Lutra AI 推出 Workspace 集成**: polarbear007. 介绍了 [Lutra.ai](https://lutra.ai)，这是一个将 AI 与 Google Workspace 集成的平台，用于数据处理和互联网研究。它支持从 Google Drive 中的 PDF 提取信息并将其转换为 Google 表格等操作。

**提到的链接**:

- [Lutra AI](https://lutra.ai): 未找到描述
- [Creating a ChatGPT like UI for all your AI projects](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5): GitHub 仓库链接：https://github.com/esxr/repurposed-ollama-webui
- [Chat with a CSV - LangChain CSV Agents Tutorial For Beginners (OpenAI API)](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2): 在这段 LangChain 视频中，我们将了解如何使用 CSV agents 和 OpenAI API 直接与 CSV 文件对话。虽然仍有一些 Bug，但这是一个非常实用的...
- [TinyDesk AI - Powerful tools to help you study smarter](https://tinydesk.ai): tinydesk.ai - 您的 AI 和 Stable Diffusion 来源，提供前沿见解和进展
- [GitHub - yanniedog/autocrew: Automatically create a crew and tasks for CrewAI](https://github.com/yanniedog/autocrew): 为 CrewAI 自动创建 crew 和 tasks。通过在 GitHub 上创建账号来为 yanniedog/autocrew 的开发做出贡献。

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1202887608276099143) (3 messages): 

- **斯坦福大学揭秘 AI DSP**: `@lhc1921` 分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=dTzL8OF_3i0)，介绍了 **斯坦福大学的 Demonstrate - Search - Predict 模型 (DSP)**，展示了一种通过流水线感知演示（pipeline-aware demonstrations）引导高级程序的方法。
- **生成式 AI 聊天体验变得简单**: 用户 `@esxr_` 发布了一个 [教程](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5)，解释了如何在 15 分钟内适配开源框架，为 AI 项目创建 **类似 ChatGPT 的用户界面**。
- **通过 LangChain 与 CSV 对话**: `@ryannolan` 介绍了一个关于使用 LangChain CSV Agents 的 [教程](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2)，使用户能够通过 **OpenAI API** 直接与 CSV 文件进行对话，并承认该功能虽然具有创新性，但目前仍存在一些 Bug。

**提到的链接**:

- [AI DSP: LLM Pipeline to Retriever Model (Stanford)](https://www.youtube.com/watch?v=dTzL8OF_3i0): 斯坦福大学的 Demonstrate - Search - Predict 模型 (DSP)。DSP 可以表达高级程序，从而引导流水线感知演示，搜索相关内容...
- [Chat with a CSV - LangChain CSV Agents Tutorial For Beginners (OpenAI API)](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2): 在这段 LangChain 视频中，我们将了解如何使用 CSV agents 和 OpenAI API 直接与 CSV 文件对话。虽然仍有一些 Bug，但这是一个非常实用的...
- [Creating a ChatGPT like UI for all your AI projects](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5): GitHub 仓库链接：https://github.com/esxr/repurposed-ollama-webui

  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/) (1 messages): 

natureplayer: https://huggingface.co/spaces/mteb/leaderboard
  

---

### LLM Perf Enthusiasts AI ▷ #[feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549/1203225679911718912) (1 条消息): 

- **浏览频道功能请求**：`@joshcho_` 建议增加 **browse channels** 功能，以解决无法查看所选感兴趣频道的问题——这是社区启用设置中常见的功能。他们强调自己倾向于忽略许多频道，并希望简化关注点。
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1203034070628565037) (8 条消息🔥): 

- **GPT-3.5 通过指令测试**：`@justahvee` 发现新的 **GPT-3.5** 在“重指令任务”中表现更好，这表明其改进与上下文窗口大小无关，完全归功于模型遵循给定指令能力的增强。
- **合规性的权衡**：`@justahvee` 提到优先级在于指令遵循而非推理能力，如果这意味着模型能更准确地遵守指令，则可以接受推理能力的任何潜在退化。
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1202999433592176650) (7 条消息): 

- **深度思考提升 AI 性能**：用户 `@res6969` 观察到，在 Prompt 中进行更详尽的解释往往会为决定最终输出的 `computation to the tokens` 分配更多计算资源，从而带来更好的 AI 性能。
- **权衡警告：速度 vs 智能**：`@res6969` 承认，虽然详细的 Prompt 提高了智能，但存在显著的 `latency tradeoff`（延迟权衡）。
- **智能响应赢得用户赞誉**：`@res6969` 分享道，采用对用户隐藏的全面 Chain of Thought (CoT) Prompt 的异步策略，可以提供令人印象深刻的“聪明” AI 响应。
- **GPT-4-Turbo 的实际应用**：用户 `@sourya4` 报告称，通过在 `gpt-4-turbo` 中使用扩展的思维解释，提高了 Function Calling 的准确性，同时正在积极探索平衡 `latency tradeoffs` 的方法。
- **迭代思维处理**：`@byronhsu` 询问是否可以保存 Chain of Thought 输出并将其重用于二次处理步骤，`@res6969` 给予了肯定回答，尽管目前尚未进行正式评估。
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1203180973215449088) (6 条消息): 

- **打个招呼**：用户 `@daydream.nation` 以简单的 "hey everyone" 向聊天室问好。
- **意识到项目已公开**：`@daydream.nation` 确认团队已经公开了他们的项目。
- **对错过参与表示遗憾**：`@daydream.nation` 对目前未能参与该项目表示遗憾。
- **推测大规模交互测试**：`@daydream.nation` 推测，这次发布可能旨在更大规模地测试人类交互，就像 **Google's Bard** 一样。
- **考虑 Alignment 背景**：`@daydream.nation` 澄清他们的评论是在 **alignment** 的背景下发表的。
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 条消息): 

cryptossssun: 🤔
  

---


### Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1203193305182109696) (1 条消息): 

- **准备投入工作的哲学先锋**：`@daydream.nation` 擅长 **Python、Excel 数据建模和 SQL**，并具有 **哲学背景**。他曾撰写过一篇关于 **Bard** 的研究论文，并表示致力于探索人类物种的未来以及利用 AI 解决意识的难题。他渴望将自己的多元背景与 AI 逻辑和论证相结合，并准备好就其独特见解如何为该领域做出贡献进行协作讨论。
  

---

### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1202968562487267328) (4 messages): 

- **基于 Llama2 构建的 Infinite Craft**: `@chrisamico` 引起了大家对一款名为 [Infinite Craft](https://neal.fun/infinite-craft/) 游戏的关注，该游戏基于 **llama2** 构建，展示了玩家可以拖动水、火、风、土等元素进行合成。
- **游戏推荐**: `@chrisamico` 还建议尝试 Infinite Craft 创作者的其他更多游戏，称赞它们非常聪明、有趣，有时还发人深省。
- **确认 Infinite Craft 的魅力**: `@dbreunig` 以简短的认可确认了该游戏的吸引力，认为它是一个极佳的案例。
- **Infinite Craft 的成瘾性**: `@bdexter` 表示这款游戏确实让人上瘾，暗示其对游戏引人入胜的内容有亲身体验。

**提到的链接**:

[Infinite Craft](https://neal.fun/infinite-craft/): 一款关于合成的游戏

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1202919998574694430) (4 messages): 

- **德语 Embedding 模型表现卓越**: `@damian_89_` 分享了一条 [推文](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19)，强调两个德语 Embedding 模型——**来自 @JinaAI_ 的 jina-embeddings-v2-base-de** 和 **来自 @BAAIBeijing 的 bge-m3** 在企业数据测试中优于其他模型，其中 BGE 表现最为出色。
- **使用指标测试 Embedding**: `@devnull0` 建议使用合适的指标来测试这些 Embedding 模型以评估性能，但未提供具体的指标或方法。
- **带有 Notebook 的 RAG 评估指南**: `@devnull0` 分享了一个 [GitHub notebook](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb) 用于评估检索增强生成 (RAG) 系统，并附带了仓库的视觉预览。
- **srk.ai 上的 RAG 评估深度探讨**: 随附的 [博客文章](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex) 提供了关于评估 RAG 系统的 encoder 和 reranker 组件的全面指南，使用了 LlamaIndex 和自定义测试数据集。

**提到的链接**:

- [Damian Strobel (@damian_89_) 的推文](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19): 任何在德语数据上进行 RAG 的人请注意，最近发布了两个 Embedding 模型：@JinaAI_ 的 jina-embeddings-v2-base-de 和 @BAAIBeijing 的 bge-m3 - 在我对真实企业数据的测试中，两者都优于所有...
- [RAG - Encoder 和 Reranker 评估](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex): 使用自定义数据集评估 RAG 流水线中 encoder 和 reranker 的性能
- [GitHub 上的 srk_ai_blog/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb): 用于 https://srk.ai/ 博客的代码。欢迎通过在 GitHub 上创建账号来为 SudalaiRajkumar/srk_ai_blog 的开发做出贡献。

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=SEavari8xaU
  

---


### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 

.mrfoo: LLaVA 1.6 发布了：https://llava-vl.github.io/blog/2024-01-30-llava-1-6/