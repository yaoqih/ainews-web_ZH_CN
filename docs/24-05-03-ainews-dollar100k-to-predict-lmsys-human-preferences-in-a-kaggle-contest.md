---
companies:
- groq
- openai
- lmsys
- scale-ai
- ai2
- nvidia
date: '2024-05-03T22:09:28.423042Z'
description: '**Llama 3 模型**正在取得突破，Groq 的 70B 模型实现了每百万 token 成本的历史新低。一项新的 **Kaggle
  竞赛**提供 10 万美元奖金，旨在开发能从超过 5.5 万条用户与大语言模型（LLM）对话的数据集中预测人类偏好的模型。像 **Prometheus 2**
  这样的开源评估 LLM 在评审任务中的表现优于 **GPT-4** 和 **Claude 3 Opus** 等闭源模型。**WildChat1M** 等新数据集提供了超过
  100 万条包含多样化和有害示例的 ChatGPT 交互日志。**LoRA 微调**等技术显示出显著的性能提升，而 **NVIDIA 的 NeMo-Aligner**
  工具包支持在数百个 GPU 上进行可扩展的 LLM 对齐。此外，研究人员还提出了**事实感知对齐方法**，以减少 LLM 输出中的幻觉。'
id: 4caf2d11-37e1-4ed4-9a10-356e62c70c59
models:
- llama-3-70b
- llama-3
- gpt-4
- claude-3-opus
- prometheus-2
original_slug: ainews-not-much-happened-today-3049
people:
- bindureddy
- drjimfan
- percyliang
- seungonekim
- mobicham
- clefourrier
title: 10 万美元奖金：在 Kaggle 竞赛中预测 LMSYS 人类偏好。
topics:
- benchmarking
- datasets
- fine-tuning
- reinforcement-learning
- model-alignment
- hallucination
- parameter-efficient-fine-tuning
- scalable-training
- factuality
- chatbot-performance
---

<!-- buttondown-editor-mode: plaintext -->> 2024年5月2日至5月3日的 AI 新闻。我们为您检查了 7 个 subreddits、[**373** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务器（**418** 个频道，**5847** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**642 分钟**。

本周 AI 新闻相对较少。[这](https://lmsys.org/blog/2024-05-02-kaggle-competition/)是一个有趣的 Kaggle 新挑战：

> 您将使用来自 Chatbot Arena 的数据集，其中包含各种 LLM 的对话和用户偏好。通过开发一个能够准确预测人类偏好的模型，您将为提高聊天机器人的性能以及与用户预期的对齐（alignment）做出贡献。训练数据集包含超过 55,000 条真实的真实用户与 LLM 对话及用户偏好，已删除个人身份信息。您的解决方案提交将在包含 25,000 个样本的隐藏测试集上进行测试。

> 比赛将持续到 8 月 5 日，总奖金为 100,000 美元，其中第一名奖金 25,000 美元，第二至第四名奖金各 20,000 美元，第五名奖金 15,000 美元。

---

**目录**

[TOC] 


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**LLM 模型发布与基准测试**

- **Llama 3 模型**：[@DrJimFan](https://twitter.com/DrJimFan/status/1786429467537088741) 宣布了 DrEureka，这是一个 LLM agent，它编写代码在模拟环境中训练机器人技能，并实现向现实世界的 zero-shot 迁移。[@GroqInc](https://twitter.com/awnihannun/status/1786066330501956053) 的 Llama 3 70B 模型打破了性能记录，价格为 **每 1M input token $0.65，每 1M output token $0.9**。[@bindureddy](https://twitter.com/bindureddy/status/1786019505027608646) 指出 Groq 的 Llama 3 模型处于领先地位，而 OpenAI 则专注于炒作 GPT-5。
- **LLM 基准测试**：[@DrJimFan](https://twitter.com/DrJimFan/status/1786054643568517261) 建议了三种重要的 LLM 评估类型：由受信任的第三方（如 [@scale_AI](https://twitter.com/scale_AI)）公开报告分数的**私有测试集**、像 [@lmsysorg](https://twitter.com/lmsysorg) 的 Chatbot Arena 这样的**公开对比基准**，以及针对各公司用例**私下策划的内部基准**。[@percyliang](https://twitter.com/percyliang/status/1786256267138478475) 指出某些模型在 GSM8K 基准测试中对特定 prompt 表现不佳。
- **开源评估器 LLM**：[@seungonekim](https://twitter.com/ShayneRedford/status/1786455899059503448) 介绍了 Prometheus 2，这是一款开源评估器 LLM，能**紧密模拟人类和 GPT-4 的判断**，并支持直接评估和成对排名（pairwise ranking）格式。在构建 LM 裁判方面，它们的**表现优于 GPT-4 和 Claude 3 Opus 等专有 LM**。

**数据集与基准测试**

- **GSM1K 数据集**：[@percyliang](https://twitter.com/percyliang/status/1786256267138478475) 讨论了模型对新 GSM1K 数据集中的 prompt 如何敏感，需要采样和多数投票（majority voting）来减少噪声。有些模型在有额外提示（hints）时表现较差。
- **WildChat1M ChatGPT 日志**：[@_akhaliq](https://twitter.com/_akhaliq/status/1786218700900557021) 分享了来自 AI2 的 WildChat 数据集，包含超过 100 万条 ChatGPT 在真实场景下的交互日志。它拥有 **250 万个对话轮次（turns）、多样化的 prompt、多种语言以及毒性（toxic）示例**。
- **Kaggle 人类偏好预测**：[@lmsysorg](https://twitter.com/lmsysorg/status/1786100697504833572) 宣布了一项 10 万美元的 Kaggle 竞赛，旨在根据包含 5.5 万条用户/LLM 对话的新数据集，预测其 Chatbot Arena 中用户对 LLM 回复的偏好。
- **污染数据库**：[@clefourrier](https://twitter.com/clefourrier/status/1785936450577375556) 提到一个用于追踪模型和数据集污染的新开放数据库，以帮助选择“安全”的构件（artifacts）进行模型创建。


**高效 LLM 训练与推理技术**

- **LoRA for Parameter Efficient Fine-Tuning**: [@mobicham](https://twitter.com/_akhaliq/status/1786217595089105169) 评估了在 10 个基础模型和 31 个任务上，使用量化低秩适配器 (LoRA) 微调的 LLMs 进行训练和服务的可行性。**4-bit LoRA 模型平均表现优于基础模型 34 分，优于 GPT-4 10 分**。LoRAX 推理服务器支持在单个 GPU 上部署多个 LoRA 模型。
- **Efficient Model Alignment with NeMo-Aligner**: [@NVIDIA](https://twitter.com/_akhaliq/status/1786222861666971804) 推出了 NeMo-Aligner，这是一个可扩展的工具包，用于高效的 LLM 对齐技术，如 RLHF, DPO, SteerLM, SPIN。它可**扩展到数百个 GPU** 以训练大型模型。
- **Factuality-Aware Alignment to Reduce Hallucination**: [@mobicham](https://twitter.com/_akhaliq/status/1786229213357342980) 提出了事实感知（factuality-aware）的 SFT 和 RL 对齐，以引导 LLMs 输出更符合事实的回答。在涉及新知识或不熟悉文本的情况下训练 LLMs 可能会**加剧幻觉 (hallucination)**。

**Multimodal and Long-Range LLMs**

- **Multimodal LLM for Automated Audio Description**: [@mobicham](https://twitter.com/_akhaliq/status/1786219554068169162) 介绍了一种利用 GPT-4V 的多模态指令遵循能力构建的自动音频描述流水线。它生成的音频描述 (AD) **符合自然语言生成标准**，同时保持上下文一致性。
- **Extending LLM Context Windows**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1786101710022201697) 报告称，仅使用 3.5K 个合成 QA 对，就在**一夜之间将 Llama-3-8B 的上下文扩展了 10 倍，达到 80K tokens**。生成的模型在书籍问答和摘要等**长上下文任务中表现出色**，可与 GPT-4 媲美。
- **Consistent Long-Range Video Generation**: [@mobicham](https://twitter.com/_akhaliq/status/1786213056088793465) 提出了 StoryDiffusion 框架，用于从文本生成一致的长程图像/视频。它引入了 **Consistent Self-Attention 和 Semantic Motion Predictor**，以保持生成帧之间的一致性。

**Emerging Architectures and Training Paradigms**

- **Kolmogorov-Arnold Networks as MLP Alternative**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785963059938234555) 报告称 Kolmogorov-Arnold Networks (KANs) 是 MLP 的一种新型替代方案。KANs 在**边上使用可学习的激活函数**，并**用可学习的样条 (splines) 替换权重**。它们以更少的参数实现了更高的精度，并避免了维度灾难。
- **Apple's On-Device LLMs and AI-Enabled Browser**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785971852294160530) 指出 Apple 在 WWDC 上推出了 OpenELM（一系列小型端侧 LLMs）和一款支持 AI 的 Safari 浏览器。**端侧 LLMs 无需 API 调用即可实现免费推理**。

**Miscellaneous**

- **WildChat1M ChatGPT Interaction Dataset**: [@mobicham](https://twitter.com/_akhaliq/status/1786218700900557021) 推出了 WildChat1M，这是一个包含 **100 万个用户与 ChatGPT 对话的数据集，交互轮数超过 250 万**。它提供了多样化的 Prompt、多种语言，并捕捉了跨地区的各种用例和用户行为。
- **Open Source Libraries for ML Deployment**: [@dl_weekly](https://twitter.com/dl_weekly/status/1786213589033861206) 分享了一个精选的开源库列表，用于在生产环境中部署、监控、版本化和扩展机器学习模型。

---

# AI Reddit Recap

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI Model Releases and Updates**

- **Nvidia 发布 ChatQA-1.5**: 在 /r/LocalLLaMA 中，Nvidia 发布了 ChatQA-1.5，这是一个极具竞争力的 Llama3-70B QA/RAG 微调模型，[**在对话式问答和检索增强生成 (RAG) 方面表现出色**](https://www.reddit.com/r/LocalLLaMA/comments/1cidg4r/nvidia_has_published_a_competitive_llama370b/)。它在 FinanceBench 等基准测试中优于原生 RAG 基准。
- **Stability AI 的 Stable Diffusion 3 发布时间尚不明确**: 在 /r/StableDiffusion 中，人们对 [**Stable Diffusion 3 权重的发布时间**](https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/sd3_weights_are_never_going_to_be_released_are/) 存在猜测，有人根据推文预测周一发布，而另一些人则怀疑短期内不会完全发布。
- **Anthropic 的 Claude Opus 和 Udio 生成单口喜剧**: Anthropic 的 AI 模型 Claude Opus 和 Udio 被用于[**生成一段关于 AGI 之后 r/singularity 未来的单口喜剧节目**](https://www.udio.com/songs/rDL4XviHDbyxug1FXK9vXP)。

**AI Applications and Demos**

- **迈向超现实全息甲板（Holodecks）的进展**：开发 Gaussian Splatting（一种使用 Gaussian Splats 而非三角形网格表示 3D 几何的技术）的研究人员取得了[**支持从任何角度快速渲染的新进展**](https://v.redd.it/kpyrc6wbq3yc1)，使超现实全息甲板更接近现实。
- **由 Paul Trillo 委托制作的 AI 生成音乐视频**：由 Paul Trillo 为 Washed Out 的歌曲《The Hardest Part》委托制作的 [SORA 生成音乐视频](https://v.redd.it/dw6y9qpe34yc1)，通过梦幻般的视觉效果和过渡**展示了 AI 视频生成的现状**。
- **AI 驱动的 CRISPR 工具创造了新的基因编辑能力**：根据《Nature》的一篇文章，一个[“用于 CRISPR 的 ChatGPT”创造了新的基因编辑工具](https://www.nature.com/articles/d41586-024-01243-w)，**扩展了基因编辑的能力**。
- **Jetbrains IDEs 现在使用本地 AI 模型进行代码建议**：Jetbrains IDEs [现在使用具有 1.5K Token 上下文的本地 0.1B 模型进行单行代码建议](https://blog.jetbrains.com/blog/2024/04/04/full-line-code-completion-in-jetbrains-ides-all-you-need-to-know/)，并通过预处理和后处理**确保建议的实用性和正确性**。
- **Panza：个性化 LLM 邮件助手**：[Panza 是一个个性化 LLM 邮件助手](https://www.reddit.com/r/MachineLearning/comments/1ciqvqw/p_panza_a_personal_email_assistant_trained_and/)，可以**在本地训练和运行，通过对用户的邮件历史进行微调来模仿其写作风格**。它将微调后的 LLM 与检索增强生成（RAG）组件相结合。

**AI 社会影响与担忧**

- **人类现在与机器人平分网络流量**：根据一份报告，[人类现在与机器人平分网络流量](https://www.independent.co.uk/tech/dead-internet-web-bots-humans-b2530324.html)，引发了对“死网（Dead Internet）”的担忧，因为 **Twitter/X 等网站充斥着自动化账号**。一些人预测这意味着用户生成内容聚合网站的终结。
- **数据中心需要巨大的电力**：根据 Dominion Energy 的说法，[数据中心现在需要相当于一个核反应堆的电力](https://www.bloomberg.com/news/articles/2024-05-02/data-centers-now-need-a-reactor-s-worth-of-power-dominion-says)，**凸显了大规模计算基础设施巨大的能源需求**。
- **微软禁止警方使用人脸识别 AI**：在**对执法部门使用 AI 的伦理问题的持续关注**中，[微软已禁止美国警察部门使用其企业级 AI 工具进行人脸识别](https://techcrunch.com/2024/05/02/microsoft-bans-u-s-police-departments-from-using-enterprise-ai-tool/)。

**AI 研究与基准测试**

- **初级研究人员在顶级 ML 会议中表现强劲**：在 /r/MachineLearning 中提到，[初级研究人员（本科生和早期博士生）在顶级 ML 会议上发表了许多论文](https://www.reddit.com/r/MachineLearning/comments/1cidsz7/d_why_do_juniors_undergraduates_or_first_to/)，因为他们**获得了大量的支持和指导**。领导项目仍然是一项巨大的成就，表明他们具备出色的技能。
- **顶级 ML 会议中很少有论文是开创性的**：同样在 /r/MachineLearning 中，一位研究人员估计[他们自己被录用的工作虽然不错但影响力不高](https://www.reddit.com/r/MachineLearning/comments/1cin6s8/d_something_i_always_think_about_for_top/)，更像是“墙上的一块砖”。**像《Attention is All You Need》这样改变游戏规则的论文非常罕见**。
- **分阶段发布数据集有助于检测基准测试污染**：/r/MachineLearning 中的一个建议是，[基准测试创建者应分阶段发布数据集](https://www.reddit.com/r/MachineLearning/comments/1cilnzv/d_benchmark_creators_should_release_their/)，通过比较模型训练数据截止日期之前与之后发布的子集上的性能，来**检测模型中的基准测试污染**。
- **spRAG：用于复杂现实世界查询的开源 RAG 系统**：[spRAG 是一个开源的检索增强生成系统](https://www.reddit.com/r/MachineLearning/comments/1cikkw2/p_sprag_opensource_rag_implementation_for/)，旨在**处理法律文件和财务报告等密集文本上的复杂现实世界查询**。它在 FinanceBench 等具有挑战性的基准测试中优于 RAG 基准线。

---

# AI Discord 摘要

> 摘要的摘要的摘要

**1. 大语言模型 (LLM) 的进展与挑战**

- **探索 LLM 能力**：关于 **[LLaMA 3](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)** 实现 **1040k 上下文长度**、具有高级问答和 Function Calling 能力的 [Hermes 2 Pro](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B)，以及 **[llm.c](https://github.com/karpathy/llm.c/discussions/344)** 达到 **167K tokens/秒**的讨论。然而，**[quantization](https://arxiv.org/abs/2404.14047)** 似乎会**损害 LLaMA 3 的质量**。

- **多语言和多模态 LLM**：探索 LLM 如何处理 **[多语言输入](https://arxiv.org/abs/2402.18815v1)**，其中英语可能被用作枢轴语言。此外还讨论了多模态能力，如 **[Suno 的音乐生成](https://arxiv.org/abs/2404.10301v1)** 和 **[AI Vtubing](https://github.com/tegnike/nike-ChatVRM)**。

- **LLM 基准测试与评估**：对 **[基准测试数据集泄露](http://arxiv.org/abs/2404.18824)** 的担忧，以及对 **[新鲜基准测试问题](https://arxiv.org/abs/2405.00332)** 的建议。评估器 LLM **[Prometheus 2](https://huggingface.co/papers/2405.01535)** 的发布，旨在透明地评估其他 LLM。

**2. AI 模型微调与优化策略**

- **Unsloth AI 支持近乎全量微调**：Unsloth AI 社区探索了通过将除 layernorms 之外的所有参数设置为可训练来实现近乎全量微调（near-full finetuning）的可能性，其性能优于标准的 Hugging Face 实现。讨论还涉及了用于优化的数据集格式化和非官方的全量微调策略。关键资源包括 [Unsloth 的 Colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 和 [微调指南](https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices)。

- **检索增强生成 (RAG)**：关于构建 **[高效 RAG 数据栈](https://t.co/jez7g9hADV)** 以及 **[LangChain 的 RAG 集成](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)** 以实现智能化应用的指南。讨论了 RAG 在 **[LlamaIndex 的内省型 Agent](https://t.co/X8tJGXkcPM)** 中的作用。

- **优化训练流水线**：[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583) 改进了数据预处理的并行性。利用 **[DeepSpeed Stage 3](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)** 和 **[Flash Attention](https://github.com/ggerganov/llama.cpp/pull/5021)** 进行高效的大模型训练。

**3. 开源 AI 框架与库**

- **LLM 部署解决方案**：关于 **[LangChain](https://github.com/langchain4j/langchain4j)** Java 移植版、**[Dragonfly 集成](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly)**，以及支持无代码模型训练的 **[AutoTrain](https://github.com/huggingface/autotrain-advanced)** 配置发布。

- **AI 开发框架**：[Modular](https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements) 庆祝 Mojo 24.3 发布及社区贡献。**[GreenBitAI](https://github.com/GreenBitAI/green-bit-llm)** 推出了一套增强 PyTorch 的工具包，而 **[BitBlas](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp)** 提供了快速的 gemv 内核。

- **开源 AI 项目**：如 **[LM Studio 的 CLI 工具 'lms'](https://github.com/lmstudio-ai/lms)**、**[Mojo-pytest v24.3 支持](https://github.com/guidorice/mojo-pytest/issues/9)**、**[NuMojo](https://github.com/MadAlex1997/NuMojo)** 张量库以及 **[Prism CLI 更新](https://github.com/thatstoasty/prism)** 等发布，展示了社区驱动的开发。

**4. AI 硬件加速与优化**

- **GPU 优化技术**：关于 **[Triton](https://github.com/openai/triton/pull/3813)** 的 gather 过程、**[CUDA streams](https://github.com/karpathy/llm.c/pull/343)** 以及 **[llm.c](https://github.com/karpathy/llm.c/discussions/344)** 中的 **[融合分类器 (fused classifiers)](https://github.com/karpathy/llm.c/pull/343)** 的讨论。探索 PyTorch AO 中的 **[FP6 支持](https://github.com/pytorch/ao/issues/208)**。

- **专用硬件**：对 **[瑞芯微 RK3588 单板计算机 (SBCs)](https://github.com/rbrisita/01/tree/rknn)** 的关注，其显示 **Whisper RKNN 性能提升了 250%**。对 **[CHERI 安全功能](https://youtu.be/_QxXiTv1hH0?t=933)** 的好奇，该功能可实现快速的 IPC 并简化硬件设计。

- **Raspberry Pi 与嵌入式 AI**：**[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742)** 频道引起了广泛关注，而 **[llama-farm 项目](https://discord.com/channels/1122748573000409160/1137456826733047908/1235763541642838069)** 旨在将本地 `Ollama` 实例连接到云端。

**5. 杂项**

- **LM Studio 推出 CLI 工具并修复 Bug**：LM Studio 发布了 `lms`，这是一个用于管理本地 LLM、启动/停止服务器以及进行调试的新 CLI 工具。它需要 **LM Studio 0.2.22+** 版本，并且已经在 [GitHub 上开源](https://github.com/lmstudio-ai/lms)。最新的更新还修复了一个导致整个上下文被包含在模型响应中的 Bug。用户们还探索了以 headless 模式运行 LM Studio 以及将其嵌入到可扩展的服务器解决方案中。

- **LLM 中的量化挑战与上下文扩展**：量化对 **LLaMA 3** 性能的影响是一个热门话题，[Reddit 上的讨论](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/)和[研究论文](https://arxiv.org/abs/2404.14047)都表明量化会导致显著的质量损失。与此同时，凭借 Crusoe Energy 的算力，**LLama-3 8B** 实现了超过 1040k 的上下文长度，而来自 AI21 Labs 的 **Jamba-Instruct** 也表达了对更大上下文窗口的兴趣。


---



# 第 1 部分：Discord 高层摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**LLM 微调的新前沿**：社区成员讨论了使用 Unsloth 进行近乎全量微调（near-full finetuning）的可能性，探索将除 layernorms 之外的所有参数都设置为可训练的潜力。虽然 Unsloth 目前专注于解决 llama.cpp 和 GGUF 转换问题，特别是量化和加载 checkpoint shards 的挑战，但情感分析爱好者们也收到了关于如何格式化庞大数据库以适配 LLM 的建议。

**实验性全量微调策略与数据集结构化**：社区分享了在 Unsloth 上启用全量微调的非官方策略，展示了相对于标准 Hugging Face 实现更优的 loss 表现。讨论还深入探讨了用于优化的理想数据集结构，并提出了处理多个“被拒绝”（rejected）响应的策略。

**Phi 3 在浏览器中运行，但 Llama 3 Discord 缺席**：[这里](https://twitter.com/fleetwood___/status/1783195985893863578)的一条推文展示了在 Web 浏览器中运行 Phi 3 的场景，而一位成员澄清说目前还没有专门为 Llama 3 开设的 Discord 频道。同时，在 Llama 3 中加入新角色引发了辩论，有人建议使用 `type=code` 作为 `tool_call` 的替代方案。

**利用 Self-Discovery 和 Triton 的 TK-GEMM 适配 Llama 3**：一位机智的用户应用了 Self-Discovery 论文中的技术来增强 ChatGPT 的推理能力。此外，一篇 PyTorch 博客文章强调了利用 Triton 的 FP8 GEMM 在 NVIDIA H100 GPU 上加速 Llama 3，这为优化提供了深刻见解。

**量化困境与微调技巧**：在将 Llama 3 转换为 GGUF 时出现了一些问题，影响了微调数据的完整性，将 Lora 与 GGUF 模型融合时也出现了类似问题。然而，理解微调和模型管理的路径正变得越来越清晰，资深社区成员建议参考 Unsloth 的 Colab 笔记本获取指导。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **斜杠命令消失**：工程师们观察到 Discord 命令中 `/faq` 命令神秘消失，引发了一波关于“只有在失去后才发现它存在感”的调侃。

- **显卡辩论：Nvidia vs. AMD**：一个热门话题是在 Nvidia 的 4080 和 3090 GPU 与 AMD 的 7900xtx 之间做出选择，讨论集中在 VRAM 容量以及为了未来的适应性而等待 Nvidia 即将推出的 5000 系列的价值。

- **RTX 4080 的转换好奇心**：有用户询问使用 RTX 4080 将视频利用 AI 转换为动漫风格的时间效率，成员们为此类任务寻求性能基准测试。

- **GPU 忠诚度分化**：成员们激烈辩论了在 AI 应用中 Nvidia GPU 相对于 AMD 的优势，尽管 Nvidia 吹捧其全新的 Blackwell 架构，但一些 AMD 的拥趸根据他们的积极体验提出了支持意见。

- **增强之谜：文本与图像超分辨率 (Upscaling)**：分享了多种 AI 辅助向图像添加文本和图像超分辨率的方法，包括使用 Davinci Resolve 处理文本，以及使用 [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/) 和 [Harrlogos XL](https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd) 等工具进行 Stable Diffusion 的自定义文本生成。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**对话中的梯度修饰**：Discord 成员讨论了 PyTorch 中高级梯度技术，其中 `create_graph=True` 被用于获取更精细的梯度细节和 Hessian-vector products。提到了估计 Hessian 对角线的技术，利用随机性进行估算。

**Triton 的尝试与胜利**：工程师们面临了 Triton 中 `IncompatibleTypeErrorImpl` 的挑战，但在偶然发现 gather 函数问题后，通过 `tl.cast` 函数修复找到了慰藉。在 PyCharm 中使用 PyTorch 进行 Kernel 调试也证明存在问题，即使将 `TRITON_INTERPRET` 设置为 `"1"` 也是如此。

**使用 tinygrad 进行修补**：成员们分享了一个针对 tinygrad 的多 GPU 支持补丁，并支持 Nvidia 的开源驱动。GitHub 上出现了一个关于安装自定义 PyTorch 和 CUDA 扩展的正确方法的难题，通过参考 PyTorch AO 库 setup 过程中的示例来寻求清晰度。

**催化社区贡献**：GitHub 上的 Effort 项目因其具有影响力的结构而获得赞誉，同时介绍了 GreenBitAI 的工具包，这是一个增强 PyTorch 的 ML 框架。它包括创新的梯度计算方法，以及 **bitblas** 中备受关注的、可能对推理非常有用的 gemv kernel。

**torch 的苦与乐**：PyTorch 开发者辩论了构建策略和优化，从线性代数组件的构建时间到 kernel 性能。讨论了填充词表大小以在性能基准测试中公平竞争的想法，揭示了公平衡量所需的细微考量。

**LLM 内部探秘**：llm.c 项目通过 CUDA 优化技术达到了 **167K tokens/second** 的新效率。关于 CUDA streams、fused classifiers 以及在 scratch buffers 中战略性使用 atom 变量的关键讨论，凸显了浓厚的技术协作氛围。

**开源情报**：简要提到 Intel 现已添加到 PyTorch 网站，预示着潜在的集成或支持更新。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**CLI 加入 LM Studio 工具箱**：LM Studio 发布了新的 CLI 工具 `lms`，旨在简化本地 LLM 的管理，包括加载和卸载模型以及启动或停止服务器。该 CLI 工具适用于最新的 LM Studio 0.2.22 及更高版本，鼓励用户为其 [开源 GitHub 仓库](https://github.com/lmstudio-ai/lms) 做出贡献。

**Llama 的转换难题**：LM Studio 公会的协作成功解决了 `llama.cpp` 的几个集成问题，使用了如 `convert-hf-to-gguf` 等脚本。一些用户遇到了 FileNotFoundError，通过 `huggingface-cli` 重新下载必要文件得以修复，社区也协助解决了转换执行问题。

**模型性能与奇特现象**：模型频道中的讨论揭示了使用 **Goliath 120B Longlora** 模型增强故事写作的努力，以及评估 **LLAMA 3** 等模型在长文本上召回能力的实验。一个奇特现象是 **ChatQA 1.5** 展示了意想不到的响应模板，而最新 **LM Studio 0.2.22** 中的一个 bug 促使发布了修正行为的新更新。

**ROCm 的成长阵痛与胜利**：成员们探索了最新 LM Studio 0.2.22 ROCm Preview 的功能，一些人测试了 RAM 和上下文大小的上限，另一些人则解决了 embedding 模型的问题。为 AMD ROCm 预览版和 Linux 支持引入的 `lms` CLI 引发了关于该工具潜力的热烈讨论，并得到了 headless 模式执行和 docker 化努力的支持。

**服务器-客户端连接解锁**：分享了配置技巧和修复方法，包括重新填充默认配置的简便方法，通过使用正确的 IP 地址解决从 WSL 访问 LM Studio 的问题，以及实现 Windows 和 WSL 环境之间应用程序的无缝通信，无需额外复杂操作。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Beta 测试团集结**：**Pages** Beta 测试人员的招募工作已圆满结束，团队对此表示感谢，并引导大家关注后续的开发进度更新。
  
- **浏览器困境与支付难题**：有反馈指出 **Perplexity** 在 Safari 和 Brave 浏览器上无法正常运行的技术问题；同时，一名用户关于意外订阅扣费的咨询已被转至 support@perplexity.ai 寻求解决。语音命令功能的增强以及 **Gemini 1.5 Pro** 和 **GPT-4 Turbo** 等模型的用量限制是热门话题，此外，大家对新兴 AI 技术的进步也充满期待。

- **明智分享，共同进步**：社区发出提醒，在 Discord 分享链接前需确保 Thread 已设为可共享，分享内容涵盖了从**月球查询**到 **AI 音乐发现**等广泛兴趣。对打印机隐私的担忧以及对 AI 生成内容的探索，凸显了该社区多元化的关注领域。

- **AI API 探索与准确性**：讨论集中在如何通过精确的 Prompt 和 Prompt 优化技术有效利用 **Sonar Large** 模型。API 表现的不稳定性表明需要微调 `frequency_penalty`、`temperature` 和 `top_p` 等设置来提升响应质量，相关建议指向迁移至最新的 **Sonar** 模型以获得更高的准确性。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Hermes 2 Pro 参战**：近期发布的集成 LLaMA 权重的 **Hermes 2 Pro** 凭借其先进的 **QA**、**Function Calling** 和 **JSON Mode** 能力引起了轰动。它在移动设备上卓越的推理速度备受关注，相关支持材料已发布在 [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) 和 [Hugging Face](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B)。

**ChatML 适配探讨**：成员们正在剖析启用 **ChatML** 的调整方案，例如使用 Token 替换策略和修改 EOS 符号，不过关于这些修改的具体细节目前还较少。

**World-sim 法典**：围绕 **world-sim** 的热烈讨论指出了最近的更新和变化（如铁器时代的引入），并分享了关于**意识**和 AI 的资源，包括 [YouTube 演讲](https://www.youtube.com/watch?v=abWnhmZIL3w)链接。

**数据集寻求者集结**：成员们在启动挖掘序列前，询问了适用于 **finetuning LLMs** 的免费通用数据集，这在 **#bittensor-finetune-subnet** 和 **#rag-dataset** 频道引起了共同兴趣，但得到的回复有限。

**LLama 打造角落**：针对 *llamacpp* 的故障排除引出了使用 *ollama* 以避开直接处理 C 语言的建议，并推荐在 CPU 运行 LLM 的场景下采用 **quantization** 和 **pruning** 技术。对话还探讨了**逆因果律 (retrocausality)** 中“道德非交换性”的有趣概念及其心理影响。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**将 Mojo 带入命令行**：用于 Mojo 的 `prism` CLI 工具包增加了新功能，如 *persistent flags*（持久化标志）、*hooks* 和 *flag groups*。更新内容已在项目的 [GitHub 页面](https://github.com/thatstoasty/prism)上展示。

**测试驱动的 Mojo 开发**：Mojo 的测试插件 `mojo-pytest` 现在支持最新的 **24.3 版本**。关于改进可调试性的问题正在 [GitHub 上的 Issue #9](https://github.com/guidorice/mojo-pytest/issues/9) 中进行跟踪。

**NuMojo 超越竞争对手**：旨在增强 Mojo 标准库 Tensor 功能的 NuMojo 项目已更新至 Mojo 24.3 版本，并在 Benchmark 中表现优于 NumPy 和 Numba。请在 [GitHub](https://github.com/MadAlex1997/NuMojo) 上查看 NuMojo 的进展。

**学习 Mojo 的冒险**：对于那些好奇如何将 Mojo 集成到工作流中的人，现在有一个新的教程 "Let's mojo build -D your own -D version=1 app"。它旨在通过一系列工作流展示 Mojo 的能力，可以在 [GitHub](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md) 上找到。

**Nightly 版本保持 Mojo 的新鲜感**：随着基础设施的改进，Mojo 的开发正以更频繁的 Nightly 版本（最终将实现每日发布）向前迈进。Nightly 变更日志（如引入 `__source_location()` 和改进 docstring 的灵活性）可以在 [Modular Docs Changelog](https://docs.modular.com/mojo/changelog#language-changes) 中查阅。

**最大化 MAX 的可扩展性**：MAX 24.3 引入了全新的 MAX Engine Extensibility API，旨在完善 PyTorch、ONNX 和 Mojo 模型的集成。有关性能和硬件优化的详细信息请参阅 [MAX Graph APIs](https://docs.modular.com/engine/graph)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AI 就业市场轮盘赌**：社区就 AI 领域**高薪职位的转瞬即逝**展开了幽默的辩论，并调侃了 AI CEO 甚至牙医等非传统职业路径的潜在盈利能力。

**GPT-5 票价猜测站**：关于 **GPT-5 潜在定价策略**的讨论不断，小组在 OpenAI 是会选择区域定价模型还是坚持统一价格点的问题上产生了分歧。

**GPT-3 拥趸与聊天室的既视感**：尽管 GPT-4 备受关注，成员们仍对 **GPT-3 和 Codex** 表示怀念，并对缺乏用于实时讨论的**语音聊天室**提出疑问，理由是出于对审核（moderation）的担忧。

**GPT-4 的响应时间之谜**：关于 **GPT-4 响应时间**慢于 **GPT-3.5** 的讨论，其中提到 **gpt4 turbo** 面临明显的延迟，这表明工程师们正在密切关注性能指标。

**拨开 AI 研究的迷雾**：讨论强调了公开可用的研究论文与对 OpenAI 发布完整训练的专有模型的非理性预期之间的区别，原因在于其**计算需求**和专有元素。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**使用 Moondream 和 FluentlyXL 进行代码探索**：社区贡献展示了用于批处理的 [Moondream 2](https://huggingface.co/spaces/Csplk/moondream2-batch-processing) 和 [FluentlyXL v4](https://huggingface.co/spaces/fluently/Fluently-Playground)，以及 HF 音频课程的葡萄牙语翻译和用于 MPI 开发的新 [MPI Codes 仓库](https://github.com/Binary-Beast03/MPI-Codes)。此外，还讨论了 [LangChain 的智能提升](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)（利用 RAG 增强 LangGraph Agent 以实现智能邮件草拟）以及 FinBERT 的[金融情感微调](https://huggingface.co/ProsusAI/finbert)。

**Babel Fish 的大家族**：多语言领域不断扩大，[BLOOM](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) 支持 55 种语言，同时还有关于改进 LLM 的研究，例如[精选列表](https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd)和用于文本生成中自动归因的 [RARR 方法](https://huggingface.co/papers/2210.08726)。成员们还热衷于[使用 Ray 部署模型](https://ray.io/)并评估精炼 Prompt 的质量指标。

**Diffusion 模型混合学**：在 Diffusion 讨论中，社区探索了合并 Pipeline 和部分 Diffusion (partial diffusion) 的技术，在 [GitHub](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2) 上可以找到一个针对 **SD 1.5** 的显著的部分 Diffusion Pull Request。总的来说，高效且创新的模型合并策略备受关注。

**模型微调技巧**：讨论了微调模型的最佳实践，例如仅调整分类器权重和自定义训练循环，并参考了 HuggingFace 的 [_Transformers and Keras_](https://huggingface.co/docs/transformers/training) 详细指南。成员们还讨论了 **Fluently-XL-v4** 在 [Instagram](https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==) 上表现优于其他模型的视觉确认。

**寻求 AI 导师和对话者**：社区表示需要 Parquet 转换机器人以及更结构化的成员互助方式，例如可能的 **#cv-study-group**，同时分享了提升技能的知识和链接，如[关于微调 AI 模型的 YouTube 视频](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s)以及对 Graph ML 对 LLM 影响的探索。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 技术栈构建**：LlamaIndex 社区分享了关于创建高效 **Data Stacks** 和 **RAG Pipelines** 的资源，重点是提高查询精度。[@tchutch94](https://twitter.com/tchutch94) 和 [@seldo](https://twitter.com/seldo) 贡献了一份详细教程，可以在[这里](https://t.co/jez7g9hADV)阅读；而 OpenAI assistant API v2 因其有效性受到称赞，但因单次查询成本高而被标记。

- **Airbnb 房源搜索跨越**：Harshad Suryawanshi 发布了一个 RAG 应用指南，能够使用自然语言过滤 **Airbnb 房源**，并利用了 **MistralAI 的 Mixtral 8x7b** 工具。详细文档和仓库指南已在[这里](https://t.co/iw6iBzGKl6)提供。

- **内省 Agent (Introspective Agents) 介绍**：强调了 **LlamaIndex 10.34** 中的新内省功能，承诺实现能够迭代改进响应的自我反思 Agent，并计划未来与 Huggingface 集成。有人对内容敏感性提出了担忧，建议谨慎参考[这里](https://t.co/X8tJGXkcPM)详细说明的实现。

- **金融中的 Pandas、MongoDB 之谜等**：关于利用 Pandas Query Engine 进行金融应用、针对 **LlamaIndex** 查询微调 MongoDB、修复 **llamacpp** 死锁以及使用 Trulens 进行可观测性的对话正在进行中。一位成员指出 LlamaIndex 的内存使用量激增，表明迫切需要内存管理优化。

- **挑战与代码**：社区见证了各种技术咨询请求，从构建金融分析应用到解决 llamacpp 并行请求中的潜在死锁。人们正在积极寻求特定 MongoDB 操作的替代方法，以及 LlamaIndex 内存问题的指导，并提供了额外的社区学习和支持链接。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Suno 唱响新旋律**：一位 **AI-in-action-club** 成员引发了关于 **Suno 音乐生成**能力的讨论，期待它是否能独立创作完整的音乐曲目，并重点关注其音频 Token 化（audio tokenizing）技术。
- **Mamba 深度对话**：在 **llm-paper-club-west** 中，爱好者们正通过一份 Notion 深度研究报告（[A Mamba Deep Dive](https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f)）深入探讨 **Mamba** 的内部机制，并辩论其选择性召回（selective recall）和对过拟合（overfitting）的敏感性。
- **顶尖音频创新**：**AI-in-action-club** 的讨论围绕使用 Autoencoders 和 Latent Diffusion 处理及生成音频展开，提到了对谐波失真的担忧，并引用了一篇关于 *snake 激活函数*（snake activation function）可能缓解此问题的博客。
- **释放 Gemini 的潜力**：**ai-general-chat** 的一位用户正在寻找兼容 **Gemini 1.5** 的工具，但由于在长上下文（long contexts）方面表现更好，该用户表示更倾向于使用 **Opus** 或 **Cursor**。
- **SQLite 搜索的新维度**：**ai-general-chat** 提到了一款名为 `sqlite-vec` 的新 **SQLite** 向量搜索扩展，这标志着数据库内向量搜索功能的改进迈出了一大步。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**LLM 在回答前先翻译**：工程师们讨论了 **Large Language Models** (LLMs) 处理多语言输入时可能先将其转换为英语的问题，并引用了论文 ["Understanding Language Models by Fine-grained Language Identification"](https://arxiv.org/abs/2402.10588)。对于那些希望优化多语言 LLM 系统的人来说，这是一个重要的细节。

**遗失的研究方向引发怀旧**：一次关于被低估的机器学习领域的反思性交流，例如对抗鲁棒性（adversarial robustness）和特定领域建模（domain-specific modeling），这些领域因行业趋势的掩盖而备受冷落。这对该领域研究人员的职业路径具有深刻的启发意义。

**基准测试泄露阴云**：关于 LLM **基准测试数据集泄露**（benchmark dataset leakage）的担忧引发了讨论，强调了衡量泄露及修复泄露的挑战。两篇论文——[一篇](http://arxiv.org/abs/2404.18824)关于泄露检测，[另一篇](https://arxiv.org/abs/2405.00332)提出了如新鲜基准问题等新方法——推动了这一讨论。

**英语作为 LLM 的中转语言被证明有效**：**Llama** 模型的研究结果表明，将英语作为中转语言（pivot language）是一种稳健的策略，可能提升**跨模型泛化能力**（cross-model generalizability）。这种复现为开发多语言 LLM 的人员增加了该方法的说服力。

**语言模型梦想精通国际象棋**：一项研究涉及仅在国际象棋对局上训练的 Transformer，在没有启发式算法的情况下实现了高性能，如 [DeepMind 论文](https://arxiv.org/abs/2402.04494)所述。这向对非传统模型应用感兴趣的 AI 工程师展示了大模型规模训练的潜力。

**无需搜索的特级大师级国际象棋**：讨论中提到了一项使用在 1000 万场国际象棋对局数据集上训练的 Transformer 模型的研究，证明了该模型在没有特定领域增强或显式搜索算法的情况下，依然具有极高的国际象棋水平。[DeepMind 论文](https://arxiv.org/abs/2402.04494)指出，大规模训练模型可以达到极具竞争力的竞技水平，而无需采用传统象棋引擎的方法。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LLama-3 8B 展现实力**：**LLama-3 8B** 成功将其上下文长度扩展至 1040k 以上，这主要得益于 [Crusoe Energy 的算力](https://huggingface.co/crusoeai)支持，并结合了调整后的 RoPE theta，用于大型语言模型中的高级长上下文处理。
  
- **Axolotl 仓库实现优化**：通过一个 PR 贡献了重大改进，解决了 orpo trainer 中的瓶颈，使其能够利用多个 worker 进行数据预处理，详见 [GitHub PR #1583](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583)，这将提升 DPO、SFT 和 CPO 等各种训练配置的速度。

- **提示词设计演进与 llama.cpp 推理速度飞跃**：提示词微调方面有了新见解，发现在系统提示词中包含 ChatML token 可以改善分词（tokenization），同时 **llama.cpp** 的升级使 **Hermes 2 Pro Llama 3 8B** 在 8GB RAM 的 Android 设备上的推理速度提升了 30%。

- **llama.cpp 的转换复杂性**：有反馈提到将 **SafeTensors 转换为 GGUF** 时遇到困难，强调了 llama.cpp 脚本的局限性，即缺乏如 `q4k` 等广泛的转换选项。虽然探讨了解决方案并提供了[转换脚本](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh)，但对扩展输出类型的需求依然存在。

- **DeepSpeed Stage 3 突破显存限制**：ZeRO-3 优化不会影响模型质量，但需要仔细集成，并可能与 Flash Attention 协同进行微调。如果应用得当，这些技术可以提高训练速度并支持更大的 batch size，而无需复杂的并行化——相关经验已在 [Axolotl 的 GitHub](https://github.com/openaccess-ai-collective/axolotl) 上分享，并得到了 [DeepSpeed 文档](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)的证实。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**文档难题已解决**：**Ollama**、**Jan.ai** 和 **Llamafile** 的说明文档访问得到改进，提供了指向 [Open Interpreter 本地安装指南](https://docs.openinterpreter.com/guides/running-locally)的直接链接，重点介绍了 **dolphin-mixtral** 配置以简化设置流程。

**Whisper RKNN 性能提升**：正如 [rbrisita 的 GitHub 分支](https://github.com/rbrisita/01/tree/rknn)所分享的，Whisper RKNN 在 **Rockchip RK3588 SBC** 上实现了显著的 250% 性能提升，并期待未来集成 LLM RKNN 功能。

**AI Vtuber 进入开源领域**：AI Vtuber 社区受益于两个新资源：[GitHub 上的 AI Vtuber 入门套件](https://github.com/tegnike/nike-ChatVRM)，以及一个[支持离线、无需 API 的 Vtuber 仓库](https://github.com/neurokitti/VtuberAI.git)，并在 [YouTube](https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb) 上展示了实时概念验证。

**交互性扩展至移动端**：分享了在服务器上托管 **Open Interpreter** 以实现更广泛访问以及设置移动端友好本地模型的见解，并链接到了特定的 Android 设备设置和[本地运行 Open Interpreter](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally) 的方法。

**扬声器选择的明智之选**：目前正在为一个未命名的电子项目寻找最佳扬声器，承诺未来将根据集成和验证结果分享更多见解。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 应对流量激增**：OpenRouter 正在努力应对由于流量激增导致的**高于常态的错误率**，目前正在进行扩容工作，以缓解间歇性的连接问题。

**资金动态**：讨论了通过 [Stripe](https://stripe.com) 集成 **WeChat Pay** 和 **Alipay** 的提案，社区意识到这需要额外的文书工作；同时，也有人建议开发一个应用，利用 **Google payment services** 实现更顺畅的交易。

**模型规模至关重要**：AI 社区对 **LLaMA-3** 等下一代语言模型表现出浓厚兴趣，期待 Soliloquy 等实体可能发布的新作，同时也认识到专有模型带来的限制。

**微调技巧**：工程师们讨论了**在没有指令数据集（instruct datasets）的情况下进行微调导致模型变笨**的风险，一致认为混合新旧数据可能有助于防止灾难性遗忘（catastrophic forgetting）。

**Gemini Pro 故障排除**：分享了针对 **Gemini Pro** 消息问题的技术解决方案，例如以 "assistant" 角色开始 Prompt 以促进更好的交互。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**StoryDiffusion 由 Angry Penguin 打造**：[StoryDiffusion](https://huggingface.co/spaces/YupengZhou/StoryDiffusion) 引起了关注，在 angry.penguin 分享链接后，成员们对其在 AI 叙事方面的潜力产生了浓厚兴趣。

**AI Town 的问题与工具**：ai-town-discuss 中出现的*空消息和数字串*干扰凸显了 tokenizer 的问题；同时，[@TheoMediaAI 的 AI 模拟探索](https://x.com/TheoMediaAI/status/1786377663889678437)和 [@cocktailpeanut 的 sqlite replay Web 应用](https://x.com/cocktailpeanut/status/1786421948638965870)（用于 AI Town）等资源也备受关注。

**后端开发中的 Node 烦恼**：错误的 Node 版本导致 `convex-local-backend` 的本地部署受阻；解决方法是切换到 Node v18。社区记录了一个关于安装过程中 `.ts` 扩展名导致 TypeError 的 [issue](https://github.com/get-convex/convex-backend/issues/1)。

**Raspberry Pi 频道引起关注**：一段深刻的思考和一位成员的认可表明，ai-raspberry-pi 频道满足了部分成员在小型硬件上进行 AI 开发的专业兴趣。

**Cocktail Peanut 收到不明赞誉**：一位神秘成员在讨论中赞扬了 **cocktail peanut**，但让社区对其所指的工作或突破感到好奇。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SoundStream 遇到困难**：一位 AI 工程师在实现 Google 的 **SoundStream** 时遇到问题，但其他人推荐了一个具体的解决方案——一个可能提供指导的 [GitHub 仓库](https://github.com/wesbz/SoundStream)。

- **AI 艺术领域的分享精神**：一位完成了 **Stable Diffusion** Udemy 课程的新人愿意与同伴分享，旨在建立联系并进一步磨练其在 AI 生成艺术方面的技能。

- **AI 社区调侃投资**：在一个轻松的时刻，AI 爱好者们开起了投资策略的玩笑，幽默地表示更倾向于那些能让资金大幅翻倍或减半的服务。

- **模型训练中对 Prompt 遵循度的追求**：讨论揭示了对于同时使用 T5 文本编码器和 CLIP 来提高模型训练中 Prompt 遵循度的有效性持怀疑态度，引发了关于 CLIP dropout 作用的惊讶和理论探讨。

- **回归基础，大并不总是更好**：在 **StableDiffusion** 领域，由于硬件限制，重点正从构建更大的模型转向改进架构和在较小模型上进行训练。这突显了使用 CLIP 进行细致训练以避开嵌入偏见和约束的重要性。

- **数据集争论持续**：一场关于数据集选择的激烈讨论显示，相比合成数据集，人们更倾向于使用 MNIST, CIFAR 或 ImageNet 等现实世界的数据集，以更好地展示模型的可解释性。

- **可解释性还是适用性？**：对话中的怀疑论者争论为可解释性开发的方法是否也能有效地转化为解决现实世界的挑战，为讨论增添了一层实用性色彩。

- **神秘的新成员**：**[StoryDiffusion](https://storydiffusion.github.io/)** 由一名公会成员带入视野，尽管没有进一步解释，让工程师们对其用途或重要性感到困惑。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**黑客松预警：54 小时内构建 AI 产品赢取现金**：**BeeLoud** 黑客松定于 5 月 10 日至 12 日举行，邀请参赛者在 54 小时内创造 AI 创新产品，奖金池高达 25,000 美元。更多详情请参阅 [Build - BeeLoud](https://beeloud.xyz/build/)。

**LangChain 和 RAG 赋能邮件撰写**：**LangChain 的 LangGraph Agents** 现在利用检索增强生成 (RAG) 来增强 AI 辅助的邮件草拟，承诺在效率和质量上都有所提升，详见 [Medium 文章](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)。

**Java 开发者，来认识一下 LangChain**：LangChain 的 Java 移植版 **langchain4j** 现已发布，扩展了在不同平台和语言中集成 AI 应用的范围。感兴趣的工程师可以在 GitHub 上探索 [langchain4j](https://github.com/langchain4j/langchain4j)。

**Dragonfly 提升 LangChain 性能**：通过将 **Dragonfly** 内存数据存储与 LangChain 集成，开发者可以期待聊天机器人性能和上下文管理的改进，其最新的 [博客文章](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly) 中通过示例进行了说明。

**Langserve 解密**：针对 **langserve feedback endpoint**（反馈端点）进行了澄清，其中 "OK" 响应仅表示反馈已成功提交，但如果服务器认为其未经身份验证或无效，仍可能被拒绝。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **泄露模型引发混乱**：讨论了一个可能来自 **GDM** 且具有 *异常具体量化* 的泄露模型，并引用了一条 [推文](https://x.com/teortaxestex/status/1785974744556187731?s=46) 和暗示存在泄密的神秘 4chan 帖子。
- **Prometheus 2 崛起**：在一篇 [研究论文](https://arxiv.org/abs/2405.01535) 中介绍的新语言模型 **Prometheus 2** 声称具有优于 GPT-4 的评估能力，引发了关于其功效和实用性的讨论。
- **高额奖金池让竞争升温**：**LMSYS** 启动了一项 10 万美元的人类偏好预测竞赛，正如一条 [推文](https://x.com/lmsysorg/status/1786100697504833572?s=46) 中提到的，该竞赛利用了来自 **GPT-4** 和 **Mistral** 等热门语言模型的对话。
- **PPO 与 REINFORCE 的联系**：一项探索表明，在某些条件下，近端策略优化 (PPO) 可以简化为 REINFORCE 算法，这引发了持续的讨论，并分享了来自 OpenAI [Spinning Up 文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 的资源。
- **价值函数未公开的价值**：关于为什么在 RLHF 训练后通常不发布价值函数的辩论，使得人们认识到尽管它们不是社区中的标准共享内容，但它们对于强化学习持有丰富的见解。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**PDF 搜索系统曝光**：一名成员提出了一个针对 **大型 PDF 文档** 的搜索系统，讨论了包括通过 LLM 进行文档摘要、生成用于语义搜索的 Embedding 以及基于 LLM 的关键信息索引等策略。

**Llama 分词之谜揭晓**：针对在配合 **Command R+ 使用 llama-cpp-python 库** 时是否需要 *字符串开头标记 (<BOS_TOKEN>)* 提出了疑问，并观察到它在 Tokenization 过程中会被自动包含。

**确认 Cohere 的 RAG 访问权限**：回答了一位用户关于使用 **免费 Cohere API 密钥进行 RAG** 可行性的问题，确认了其可用性，尽管存在速率限制。

**C4AI Command R+ 迎来量化**：围绕 **[C4AI Command R+ 模型](https://huggingface.co/CohereForAI/c4ai-command-r-plus)** 展开了技术对话，重点关注其 [量化变体](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit) 以及本地部署的不同系统要求。

**Code Interpreter SDK 登场**：关于 [发布 Code Interpreter SDK](https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA) 的公告浮出水面，同时还讨论了它在现有技术背景下的区别。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **llamafile 迈向 Systemd**：工程师们分享了一个用于在 Rocky Linux 9 上部署 **llamafile** 的 **systemd** 脚本，其中包括详细的执行命令以及服务器端口和模型路径等必要参数的配置。
- **服务模式 URL 改版**：针对在服务模式下指定基础 URL 的需求，[GitHub](https://github.com/Mozilla-Ocho/llamafile/issues/388) 上提出了一个关于 *llamafile* 代理支持的议题，这将有助于通过 Nginx 在子目录下提供服务。
- **Ein, Zwei, Whisper!**：社区对 [distil-whisper-large-v3-german 模型](https://huggingface.co/primeline/distil-whisper-large-v3-german) 表现出浓厚兴趣，并讨论了其在语音转文本、LLM 处理和文本转语音流水线中的应用，最终可能会形成一篇详细的博客文章。
- **向量空间之谜**：有人指出 **llamafile** 和 **llama.cpp** 之间的 Embedding 方向存在差异，较低的余弦相似度指向了 [GitHub](https://github.com/Mozilla-Ocho/llamafile/issues/391) 上描述的一个问题，并已通过现有的 Python 脚本进行了测试。
- **与文件和代码对话**：为了方便使用 llamafile 与文档和代码进行对话交互，成员们建议利用 `curl` API 调用，并参考了 [llama.cpp 聊天脚本仓库](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64) 中的示例脚本。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 取得长足进步并欢迎新贡献者**：据报道，Tinygrad 最近取得了显著 **进展**，一位成员庆祝了他们对该项目的 **首次 commit**，标志着一个个人里程碑。
  
- **Blobfile 在 Llama.py 中的作用解析**：用户澄清了 `blobfile` 对于 `examples/llama.py` 中的 `load_tiktoken_bpe` 函数至关重要，增强了同行之间的理解。

- **排除 Tinygrad 前向传播故障**：一位工程师在面对前向传播计算图挑战时，通过使用 `out.item()` 或 `out.realize()` 触发执行，并安装缺失的库以修复 `NameError` 解决了问题。

- **解决 Tinygrad 中的图形可视化问题**：`networkx` 和 `pydot` 的安装错误分别通过安装 `pydot` 和 `graphviz` 得到解决，随后一位成员建议更新文档，以帮助他人避免 `sh: dot: command not found` 错误。

- **社区协作推动文档改进**：通过安装 `graphviz` 解决 "dot 命令" 问题凸显了社区的协作精神，并促成了一个更新项目文档以帮助未来用户的实用建议。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Jamba-Instruct 上线**：AI21 Labs 推出了 **Jamba-Instruct**，这是一款先进的经过指令微调的混合 SSM-Transformer 模型，旨在提升商业应用的性能。该公司在最近的 [Twitter 公告](https://twitter.com/AI21Labs/status/1786038528901542312) 和详细的 [博客文章](https://www.ai21.com/blog/announcing-jamba-instruct) 中强调了该模型的能力。

**AI21 Labs 欢迎对 Jamba-Instruct 的反馈**：AI21 Labs 正在征求行业对 **Jamba-Instruct** 的反馈，并表示愿意讨论定制需求，包括超过初始 256K 限制的上下文窗口。

**深入了解 Jamba-Instruct**：对 **Jamba-Instruct** 模型感兴趣的工程师可以通过阅读 [官方博客文章](https://www.ai21.com/blog/announcing-jamba-instruct) 获得更深入的了解，文中讨论了其在可靠商业用途中的部署和质量基准。

**更大的上下文窗口指日可待**：一位 AI21 Labs 的工作人员表达了探索 **Jamba-Instruct** 更大上下文窗口的兴趣，并邀请用户就这一潜在扩展进行合作，以满足特定的使用场景。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **快速提醒：算力资助**：AI 爱好者和工程师请注意，@PrimeIntellect 的一条推文宣布为有需要的人提供 **快速算力资助 (fast compute grants)**。查看其 [算力资助推文](https://twitter.com/PrimeIntellect/status/1786386588726960167) 了解详情。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **LLaMA 3 的量化困境**：频道内围绕 **quantization** 对 **LLaMA 模型** 的影响展开了讨论。一名 Discord 成员引用了 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/) 和 [研究论文](https://arxiv.org/abs/2404.14047)，探讨了对 **LLaMA 3** 进行低比特量化时出现的性能损失。
- **忽视 Chinchilla 定律，性能受损**：频道还探讨了由于忽视了 *chinchilla scaling law* 以及模型在 15T tokens 上进行训练，**Meta 的 LLaMA** 在进行大幅度量化时可能导致严重的信息丢失。这表明，随着精度降低，大型模型可能会经历更明显的退化。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Skunkworks AI 项目对接快速算力资助**：根据一名成员分享的 **[Twitter 公告](https://twitter.com/PrimeIntellect/status/1786386588726960167)**，雄心勃勃的 **Skunkworks 项目** 有可能获得快速算力资助。感兴趣的工程师应探索这一机会，以支持前沿计划。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **用 AI 清理本地模型堆**：有人强调需要一种 **LLM (large language model)** 来解决管理和清理散落在各目录下的 **7B local models** 的问题，这些模型是由众多的应用和库产生的。成员对组织管理的混乱表达了沮丧，并建议这可能是工具或算法开发的一个潜在领域。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1235510880003821589)** (734 messages🔥🔥🔥): 

- **社区探索使用 Unsloth 进行全量微调 (Full Finetuning)**：成员们就使用 Unsloth 进行全参数微调是否可行进行了详细讨论。尽管最初声称仅支持 LoRA（一种参数高效的训练方法），但一些人发现，将除 layernorms 之外的所有参数设置为可训练，似乎可以实现某种形式的近乎全量的微调。
- **GGUF 文件优化**：Unsloth 团队宣布他们正在致力于修复 llama.cpp 和 GGUF (Generalized GPU Format) 转换的问题，以回应社区成员在量化和加载 checkpoint shards 时遇到的困难。
- **寻求情感分析模型指导**：一名寻求帮助建立基于大规模国家级评论数据库的情感分析模型的成员，收到了关于将各种文档类型转换为适用于 LLMs 的正确格式的指导。
- **为数据集格式化和 ORPO 提供协助**：成员们讨论了使用 Unsloth 为偏好优化（preference optimization）构建数据集的方法，包括针对多个“被拒绝（rejected）”响应的策略。社区提供了见解和可能的解决方案来帮助引导这一过程。
- **分享非官方全量微调策略**：虽然 Unsloth 官方不提供对全量微调的支持，但社区成员通过手动调整模型参数尝试开启该功能。值得注意的是，与 Hugging Face 的实现相比，loss 似乎有所改善，且内存优势依然明显。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://x.com/dudeman6790/status/1784414430781931961">RomboDawg (@dudeman6790) 的推文</a>: 如果你不想手动复制代码，这里有一个完整的 Colab 笔记本。再次感谢 @Teknium1 的建议 https://colab.research.google.com/drive/1bX4BsjLcdNJnoAf7lGXmWOgaY8yekg8p?usp=shar...</li><li><a href="https://huggingface.co/papers/2402.05119">论文页面 - A Closer Look at the Limitations of Instruction Tuning</a>: 未找到描述</li><li><a href="https://huggingface.co/maywell/Llama-3-70B-Instruct-32k">maywell/Llama-3-70B-Instruct-32k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B">nvidia/Llama3-ChatQA-1.5-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B">nvidia/Llama3-ChatQA-1.5-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/grahama1970/77a2b076d18ff2a62479b3170db281c5">Lllama 70B Instruct QA Prompt</a>: Lllama 70B Instruct QA Prompt。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://datta0.substack.com/p/ai-unplugged-9-infini-attention-orpo">AI Unplugged 9: Infini-Attention, ORPO, </a>: 洞察胜于信息</li><li><a href="https://github.com/IBM/unitxt">GitHub - IBM/unitxt: 🦄 Unitxt: a python library for getting data fired up and set for training and evaluation</a>: 🦄 Unitxt：一个用于启动数据并为训练和评估做好准备的 Python 库 - IBM/unitxt</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=T1ps611iG1A">我如何为我的时事通讯微调 Llama 3：完整指南</a>: 在今天的视频中，我将分享我如何利用我的时事通讯来微调 Llama 3 模型，以便使用创新的开源工具更好地起草未来的内容...</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=WxQbWTRNTxY&t=83s">如何微调 Llama 3 以获得更好的指令遵循能力？</a>: 🚀 在今天的视频中，我很高兴能引导你完成微调 LLaMA 3 模型以实现最佳指令遵循的复杂过程！从设置开始...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">jaime-m-p 提交的 llama3 自定义正则拆分 · Pull Request #6965 · ggerganov/llama.cpp</a>: unicode_regex_split_custom_llama3() 的实现。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1235509542071177246)** (20 条消息🔥): 

- **浏览器中的 Phi 3**: 一条推文展示了有人在 Web 浏览器中运行 **Phi 3**，并特别标注了 "lmao"。该推文可以在[这里](https://twitter.com/fleetwood___/status/1783195985893863578)找到。
- **LLAMA 3 Discord 频道不存在**: 有人询问是否存在 **LLAMA 3 Discord** 频道，一名成员回复称该频道不存在。
- **在 LLAMA 3 中创建新角色**: 提出了一个关于在 **LLAMA 3** 中添加新角色的问题，并链接到了一个 GitHub 仓库。回复建议使用 `type=code` 代替 `tool_call` 进行简单替换。
- **Self-Discovery 论文技术的应用**: 一位用户发现强制 **ChatGPT** 记住 Self-Discovery 论文中的 39 个推理模块非常有用，并建议将其应用于复杂的推理任务。该论文可在[这里](https://arxiv.org/pdf/2402.03620#page=12&zoom=100,73,89)查阅。
- **Triton 对 LLAMA 3 的加速**: 来自 PyTorch 的一篇博客文章展示了 **TK-GEMM**，这是一个使用 Triton FP8 GEMM 的工具，可在 NVIDIA H100 GPU 上优化 **LLAMA 3**。该博客包含性能对比和技术细节，可在[这里](https://pytorch.org/blog/accelerating-llama3/)查看。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-llama3/?utm_content=291787920&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366">使用 Triton 内核加速 Llama3 FP8 推理</a>: 1.0 摘要  </li><li><a href="https://pytorch.org/blog/accelerating-llama3/?utm_content=291787920&utm_medium=social&utm_source=lin">使用 Triton 内核加速 Llama3 FP8 推理</a>: 1.0 摘要  
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1235486709529968652)** (580 条消息🔥🔥🔥): 

- **已发现 Llama 3 的 GGUF 转换问题**：一名成员指出，Llama 3 在转换为 GGUF 格式时会丢失微调数据。无论精度如何，该问题似乎在 GGUF 中普遍存在，已在 [FP16 和 Q8](https://github.com/ggerganov/llama.cpp/issues/7062) 中进行了测试；与 Unsloth 的讨论以及社区的建议尚未解决此问题。

- **Lora 适配器合并问题**：尝试将 Lora 适配器与 GGUF 模型合并导致微调部分丢失。尽管有建议在 GGUF 模型中使用独立的 Lora 适配器，但结果未达预期，且在 [结合使用 GGUF 和 Lora](https://github.com/ggerganov/llama.cpp/issues/7062) 时情况变得更糟。

- **分享 Llama 3 的推理和微调解决方案**：用户分享了他们的微调策略，即在 Llama 3 中使用原始的 INSTRUCT 模型并在指令后附加 eos_token。有人指出，在向 `/completion` 发送请求时需要传递所有聊天 token，部分用户可能忽略了这一点；而使用 Llama 3 启动服务器时，需要为 tokenizer 设置 `--override-kv`。

- **Llama.cpp 对 Llama 3 可能存在的问题**：鉴于 llama.cpp 的 issues 页面中列出的问题具有相似性，成员们怀疑 llama.cpp 与新发布的 Llama 3 之间可能存在兼容性问题。

- **寻求帮助并遵循路线图**：新用户正在寻求微调 Gemma 和 Llama 等模型的逐步指导。更有经验的社区成员指向了 Unsloth 的 [Llama](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp) 和 [Gemma](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo) notebook，并建议在 YouTube 等平台上搜索 AI/ML 课程和教程。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1Wg">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharin">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4倍更长的上下文窗口和1.7倍更大的批处理大小</a>: Unsloth 现在支持对具有极长上下文窗口的 LLM 进行微调，在 H100 上最高可达 228K（Hugging Face + Flash Attention 2 为 58K，因此长了 4 倍），在 RTX 4090 上最高可达 56K（HF + FA2 为 14K）。我们成功实现了...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | 模型卡片与提示词格式</a>: Meta Llama 3 使用的特殊 Token。一个提示词应包含单个系统消息，可以包含多个交替的用户和助手消息，并且始终以最后一个用户消息结尾，后跟...</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig">LoRA</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#loading-lora-adapters-for-continued-finetuning">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存 - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=j7ahltwlFH0&t=413s&pp=ygUJbGxhbWEuY3Bw">使用 LLAMAcpp 将 LLM 转换为在笔记本电脑上运行 - GGUF 量化</a>: 你想在笔记本电脑以及手机、手表等微型设备上运行 LLM 吗？如果是这样，你需要对 LLM 进行量化。LLAMA.cpp 是一个开源的...</li><li><a href="https://github.com/xaedes/llama.cpp/tree/finetune-lora/examples/export-lora">xaedes/llama.cpp 的 finetune-lora 分支下的 llama.cpp/examples/export-lora</a>: Facebook LLaMA 模型的 C/C++ 移植版本。通过在 GitHub 上创建账号，为 xaedes/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5360">从 LoRA 适配器创建 GGUF 模型 · ggerganov/llama.cpp · 讨论 #5360</a>: 我有一个由 convert-lora-to-ggml.py 创建的 ggml 适配器模型 (ggml-adapter-model.bin)。现在我的疑问是如何从中创建完整的 GGUF 模型？我见过使用 ./main -m models/llama...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">带有合并 LoRA 适配器的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在运行 Unsloth 在 llama3-8b 上对 Instruct 模型进行 LoRA 微调。1：我将模型与 LoRA 适配器合并为 safetensors。2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 快 2-5 倍，且节省 80% 内存 - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7021">无法将 llama3 8b 模型转换为 GGUF · Issue #7021 · ggerganov/llama.cpp</a>: 请包含有关您的系统信息、重现错误的步骤以及您正在使用的 llama.cpp 版本。如果可能，请提供一个重现该问题的最小代码示例...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">jaime-m-p 提交的 llama3 自定义正则拆分 · Pull Request #6965 · ggerganov/llama.cpp</a>: unicode_regex_split_custom_llama3() 的实现。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1235491714119110716)** (162 条消息🔥🔥):

- **频道协作难题**：一名成员询问是否可以创建一个用于协作和共同编码的频道，特别是针对那些有兴趣在深夜或周末寻找合作伙伴的用户。这个想法被拿来与 EleutherAI 的社区项目频道进行比较，并有人建议重新利用或替换现有频道（例如搁置 <#1180145007261401178> 以支持新的社区项目频道）以促进协作。

- **专业化的障碍**：一场关于将 7B 模型专业化以处理复杂任务（如加密证明生成）可行性的讨论展开了。多位用户发表了看法，一致认为对于小型 LLM (7B) 来说，此类任务可能过于雄心勃勃。有人建议，虽然在高度专业化的用例中，较小的模型可以优于较大的模型，但它们通常无法与 GPT-4 或 Claude 等大型模型相提并论。

- **数据与算力考量**：讨论还涉及了 LLM 训练中数据规模和质量的重要性，一名成员就如何有效利用其资源（包括 32 个 H100 GPU）寻求建议。会议强调，模型大小和数据准备是实现高性能的关键因素，而成功的关键取决于具体案例。

- **通过社区经验展示与学习**：Drsharma24 表达了从社区经验中学习的愿望，并希望建立一个空间来讨论围绕 fine-tuning 和模型训练的成功案例及策略，类似于 Hugging Face 平台。对话强调，Unsloth AI 社区可以从这种知识共享中受益。

- **商业可行性 vs. 纯粹实验**：聊天触及了开发商业用例与从模型训练中实验和学习之间的区别。一名成员建议，商业用例需要能够充分反映生产环境的训练数据，而其他人则强调了牢记最终目标的重要性。

**提到的链接**：<a href="https://tenor.com/view/dog-awkward-awkward-dog-staring-dog-patchibana-gif-13086408744970718509">Dog Awkward GIF - Dog Awkward Awkward dog - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1235486692543168543)** (753 messages🔥🔥🔥): 

- **FAQ 从 Discord 命令中消失**：用户注意到 `/faq` 命令缺失，并思考其被移除的原因。事实证明该命令确实消失了，导致成员们在与机器人交互后才意识到它的缺席并开起了玩笑。

- **辩论 AI 的 GPU 选择**：参与者讨论了各种 GPU 选项，如 Nvidia 的 4080 和 3090，AMD 的 7900xtx，考虑了 VRAM 大小和前瞻性。Nvidia 5000 系列 GPU 的发布备受期待，促使用户建议等待新系列，而不是投资于即将过时的显卡。

- **视频转动漫咨询**：一名成员询问了使用 RTX 4080 将视频转换为动漫风格素材所需的时间，并寻求有关使用 AI 进行视频转换的 benchmarks。

- **AMD 与 Nvidia 在 AI 领域的观点碰撞**：关于在 AI 任务中选择 AMD 还是 Nvidia GPU 的讨论变得激烈。虽然一些人主张 Nvidia 的优越性，特别是凭借 Blackwell 架构等新技术，但一位用户根据个人对该品牌的成功使用经验为 AMD 辩护。

- **寻求文本和图像 Upscaling 的解决方案**：用户讨论了使用 AI 为图像添加文本的最佳路径，并询问了图像 upscaling 的最佳方法。虽然建议使用 Davinci Resolve 和 Kittl 等工具处理文本，但关于图像 upscaling 工具的讨论中穿插着对 ComfyUI 的提及，这是一个用于 AI 图像处理的多功能平台。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/">ComfyUI Examples</a>: ComfyUI 工作流示例</li><li><a href="https://huggingface.co/gemasai/4x_NMKD-Siax_200k/tree/main">gemasai/4x_NMKD-Siax_200k at main</a>: 未找到描述</li><li><a href="https://huggingface.co/uwg/upscaler/blob/main/ESRGAN/4x_NMKD-Siax_200k.pth">ESRGAN/4x_NMKD-Siax_200k.pth · uwg/upscaler at main</a>: 未找到描述</li><li><a href="https://bitwarden.com/help/authenticator-keys/">no title found</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Bitwarden/comments/1chob6h/bitwarden_just_launched_a_ne">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd">Harrlogos XL - Finally, custom text generation in SD! - Harrlogos_v2.0 | Stable Diffusion LoRA | Civitai</a>: 🚀HarrlogosXL - 为 SDXL 带来自定义文本生成！逐步教会 Stable Diffusion 拼写，一次一个 LoRA！Harrlogos 是一个经过训练的 SDXL LoRA ...</li><li><a href="https://www.reddit.com/r/Bitwarden/comments/1chob6h/bitwarden_just_launched_a_new_authenticator_app/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/ComfyWorkflows/comfyui-launcher">GitHub - ComfyWorkflows/ComfyUI-Launcher: Run any ComfyUI workflow w/ ZERO setup.</a>: 零配置运行任何 ComfyUI 工作流。通过在 GitHub 上创建账户为 ComfyWorkflows/ComfyUI-Launcher 的开发做出贡献。</li><li><a href="https://github.com/crystian/ComfyUI-Crystools">GitHub - crystian/ComfyUI-Crystools: A powerful set of tools for ComfyUI</a>: 一套强大的 ComfyUI 工具。通过在 GitHub 上创建账户为 crystian/ComfyUI-Crystools 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgr74j/comment/l2bxv66/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/comment/l2dhd6q/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1235509440904433706)** (3 条消息): 

- **处理梯度细节**：一位成员指出，在计算中为了获取某些梯度细节，可能需要设置 `create_graph=True`。
- **澄清 Hessian 混淆**：同一位成员随后澄清了他们的想法，重点不在于对角线，而是在于计算两次**相对于权重的 Hessian-vector product**。
- **通过随机性估计 Hessian 对角线**：另一位成员提到在**论文中看到一个技巧**，可以利用随机性结合 **Hessian-vector product** 来估计 Hessian 的对角线。
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1235636317891727442)** (2 条消息): 

- **Triton 新手的 Gather 过程受阻**：一位新成员在 Triton 中实现简单的 gather 过程时遇到了 `IncompatibleTypeErrorImpl`，他们尝试使用指针算术将值从一个 Tensor 复制到另一个 Tensor。后来他们意识到问题涉及使用了错误的 Tensor 类型，并注意到新引入的 `tl.cast` 函数可能是一个解决方案（[Triton PR #3813](https://github.com/openai/triton/pull/3813)）。
- **PyCharm 中的 Kernel 调试挑战**：同一位成员在 PyCharm 中尝试在 Triton kernel 内部设置断点时遇到困难，尽管按照仓库文档建议将 `TRITON_INTERPRET` 设置为 `"1"`，且使用 `breakpoint()` 函数也未成功。

**提到的链接**：<a href="https://github.com/openai/triton/pull/3813">[Frontend] Add tl.cast function. by jlebar · Pull Request #3813 · openai/triton</a>：这解决了 Triton 中的一个不一致问题，即 Tensor 上的每个其他函数都有一个关联的自由函数（free function）——即你可以执行 x.foo 和 tl.foo(x)。

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1235567875658616923)** (6 条消息):

- **tinygrad 获得 NVIDIA Open Driver 补丁**：一位成员分享了一个[用于 NVIDIA 开源驱动多 GPU 支持的 tinygrad 补丁](https://morgangiraud.medium.com/multi-gpu-tinygrad-patch-4904a75f8e16)，并提供了可能对遇到类似安装问题的其他人有用的文档。
- **长期支持的内核模块考量**：NVIDIA 显卡上点对点（peer-to-peer）内存修复的长期支持受到质疑，引发了关于创建内核模块是否为可行解决方案的讨论。
- **关于自定义 CUDA 扩展安装的咨询**：一位成员寻求关于在 setup.py 文件中安装自定义 PyTorch/CUDA 扩展的正确方法的建议，并指出了现有方法的问题，具体见其 [GitHub 仓库](https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15)。
- **分享 PyTorch 中 CUDA 扩展设置的解决方案**：另一位成员通过链接到 Pull Request 提供了帮助，这些 PR 展示了如何在 PyTorch AO 库中管理自定义 CUDA 扩展。他们提供了关于 [setup 流程](https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78)及相关 PR（[PR#135](https://github.com/pytorch/ao/pull/135), [PR#186](https://github.com/pytorch/ao/pull/186), [PR#176](https://github.com/pytorch/ao/pull/176)）的详细链接。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15">hqq/setup.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78">ao/setup.py at 0ba0006eb704dea33becec82b3f34512fe8a6dff · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: 这是 #130 的可合并版本 - 我必须进行一些更新，包括添加除非使用 PyTorch 2.4+ 否则跳过测试，以及如果 CUDA 不可用则跳过测试，将 ninja 添加到开发依赖项中...</li><li><a href="https://github.com/pytorch/ao/pull/186">louder warning + docs for custom cuda extensions by msaroufim · Pull Request #186 · pytorch/ao</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/176">Add A10G support in CI by msaroufim · Pull Request #176 · pytorch/ao</a>: 支持 A10G + manylinux，以便 CUDA 扩展能在尽可能多的系统上运行
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1235563374348341288)** (43 条消息🔥): 

- **PyTorch PR 的痛苦**：贡献者 **kashimoo** 对 PyTorch 中线性代数组件缓慢的构建时间表示沮丧，此外另一个 PR 由于 Meta 内部构建问题而被撤销。**chhillee** 确认由于 PyTorch 的“GitHub 优先”策略，此类挫折很常见，并提议将 **kashimoo** 与 Slack 频道上更资深的贡献者联系起来。

- **PyTorch 开发的调试符号**：**kashimoo** 询问如何构建带有调试符号的特定目录以方便使用 gdb。虽然 **chhillee** 建议使用 [PyTorch 开发论坛](https://dev-discuss.pytorch.org/t/how-to-get-a-fast-debug-build/1597)上的可用脚本，但 **kashimoo** 认为这可能不足以满足其需求。

- **PyTorch 中的动态编译挑战**：**benjamin_w** 报告了在 PyTorch 2.3 中将 `dynamic=True` 与 `torch.compile(...)` 及 Distributed Data Parallel (DDP) 结合使用时的问题。虽然该方法在 PyTorch 2.2.2 中有效，但在 2.3 版本中似乎会导致每个 batch 都重新编译。**marksaroufim** 建议不要使用 `dynamic=True`，而是建议手动将序列长度标记为动态。

- **改进 CUDA MODE Discord 的 Issue 分流**：**marksaroufim** 等人讨论了处理服务器上日益增多的 issue 的方法，提出了一个解析并自动在 GitHub 上提交 issue 的机器人想法，**jamesmel** 表示愿意实现该机器人。目前决定先在 cuda mode 中开启 issue 以管理涌入的信息。

- **针对可变长度的 Torch Compile 优化**：故障排除仍在继续，**benjamin_w** 在 PyTorch 2.2 和 2.3 中针对动态序列长度使用 `torch._dynamo.mark_dynamic(inputs, index=1)` 时遇到了 `ConstraintViolationError`。他们更倾向于在多个 batch 间保持持久的模型编译，但遇到了不稳定的行为。**marksaroufim** 建议创建一个 GitHub issue 是解决该问题的最佳方式。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/how-to-get-a-fast-debug-build/1597">如何获取快速的 debug 构建</a>：随着 albanD 的 <a href="https://github.com/pytorch/pytorch/pull/111748">Pull Request #111748</a> 被合并到 pytorch/pytorch，现在可以使用一个新的编译标志来指定 debug 信息...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile>)">torch.compile &mdash; PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://www.internalfb.com/diff/D56934078">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1235553828439330877)** (5 条消息): 

- **对 Effort 项目表示赞赏**：一位成员称赞了 [GitHub 上的 Effort 项目](https://github.com/kolinko/effort)，认为它非常令人惊叹。
- **矩阵乘法混淆**：有人指出一个矩阵乘法示例中的错误，指出 **3 x 1** 和 **3 x 3** 矩阵的内部维度不匹配，无法进行运算。
- **承诺快速修正**：作者承认了关于向量方向的混淆，并表示打算进行修正，并提到之前也曾被指出过类似的错误。
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1235604938160214086)** (4 条消息): 

- **避免灾难性遗忘**：一位成员发现 [Ziming Liu 的推文](https://twitter.com/ZimingLiu11/status/1785483967719981538)很有趣，该推文展示了如何在玩具测试用例中避免灾难性遗忘。
- **寻求速度**：有人指出灾难性遗忘的解决方案“目前非常慢”，从而引发了对提高其速度的潜在方法的关注。
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1235656551163760660)** (2 条消息): 

- **自定义 CUDA 扩展的 FP6 支持候选**：根据 PyTorch AO 仓库中的 [GitHub issue 讨论](https://github.com/pytorch/ao/issues/208)，已确定自定义 CUDA 扩展的新候选——**FP6 支持**。并向任何有兴趣为该扩展做出贡献的人发出了邀请。

- **社区成员对 FP6 表现出兴趣**：尽管缺乏经验，一位社区成员仍表达了为新的 FP6 支持项目做贡献的热情，目前正在努力理解相关的研究论文，以确定他们可以在哪些方面做出实际贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pyto">pyto - 概览</a>：pyto 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/pytorch/ao/issues/208">FP6 dtype! · Issue #208 · pytorch/ao</a>：🚀 功能、动机和提案 https://arxiv.org/abs/2401.14112 我想你们一定会喜欢这个。DeepSpeed 开发者在不支持 FP8 的显卡上引入了 FP6 数据类型，其中.....
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1235657156536041593)** (9 条消息🔥): 

- **寻求 Karpathy 的视频设置建议**：一位成员询问如何实现类似于 Andrej Karpathy 的视频设置，包括实时屏幕共享和小摄像头视图。他们收到了 [Karpathy 的一段 YouTube 视频](https://www.youtube.com/watch?v=zduSFxRajkE)链接作为参考。

- **OBS Streamlabs：视频制作的首选**：针对简单视频设置的咨询，有人建议使用 OBS Streamlabs。社区成员提到，这个多功能工具有很多教程可供参考。

- **使用 iPhone 和支架提升视频质量**：为了获得更好的视频通话或录制效果，建议将 iPhone 与 Mac 配合使用，以获得优于典型笔记本设备的摄像头和麦克风质量，并推荐了 [KDD 网络摄像头支架](https://a.co/d/7uxdnek)作为实用配件。

- **动漫欣赏间歇**：一位成员表达了对动漫偏好的好奇，引发了简短的交流，其中《火影忍者》、《一拳超人》、《剑风传奇》和《咒术回战 (JJK)》因其高质量的动画和迷人的战斗场面而被提及。

**提到的链接**：<a href="https://www.youtube.com/watch?v=zduSFxRajkE">让我们构建 GPT Tokenizer</a>：Tokenizer 是大语言模型 (LLM) 中必要且普遍存在的组件，它在字符串和 Token（文本块）之间进行转换。Tokenizer...

  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/)** (1 条消息): 

srush1301: 嗯，是的，这个描述是错误的。我会更新一个更清晰的版本。
  

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1236014784009474089)** (4 messages): 

- **GreenBitAI 推出新工具包**：一位成员分享了 [GreenBitAI 工具包](https://github.com/GreenBitAI/green-bit-llm) 的链接，该工具包用于微调、推理和评估 Large Language Models (LLMs)。他将其描述为一个增强 PyTorch 的 **ML framework**，相比专注于矩阵乘法操作的 bitblas，它的功能更为全面。
- **BitBlas 为推理提供了一个极具前景的 Kernel**：提到了一个名为 **BitBlas** 的工具包，它拥有一个用于 **2-bit 操作的快速 gemv kernel**，这可能对推理非常有益，尽管该成员尚未进行尝试。
- **GreenBitAI 引擎中的 Binary Matmul**：讨论继续提到了 [GreenBitAI 的 cutlass kernels](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp)，特别是其中执行二进制矩阵乘法（binary matrix multiplication）的部分，这是其增强 PyTorch 工具包的一部分。
- **GreenBitAI 工具包中值得注意的创新梯度计算**：一位成员强调，GreenBitAI 的工具包包含了在训练期间计算权重梯度的代码，如其 [q4_layer.py](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20) 文件所示。由于梯度**未被打包（not packed）**，他对潜在的 VRAM 占用表示好奇。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20">bitorch-engine/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py at main · GreenBitAI/bitorch-engine</a>：一个通过低比特量化神经网络专用函数增强 PyTorch 的工具包。 - GreenBitAI/bitorch-engine</li><li><a href="https://github.com/GreenBitAI/green-bit-llm">GitHub - GreenBitAI/green-bit-llm: A toolkit for fine-tuning, inferencing, and evaluating GreenBitAI's LLMs.</a>：用于微调、推理和评估 GreenBitAI LLM 的工具包。 - GreenBitAI/green-bit-llm</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp">bitorch-engine/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp at main · GreenBitAI/bitorch-engine</a>：一个通过低比特量化神经网络专用函数增强 PyTorch 的工具包。 - GreenBitAI/bitorch-engine
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1235623975653085214)** (644 messages🔥🔥🔥): 

- **CUDA 与内存优化讨论**：团队通过优化 CUDA kernels 并引入 CUDA streams 和 fused classifiers 等改进，实现了 **167K tokens/second** 的性能，超越了 PyTorch 的 150K tok/s。他们正在讨论 **bias kernel 优化**的影响以及进一步提升性能的**后续步骤**。参见相关的 [讨论和 pull request](https://github.com/karpathy/llm.c/discussions/344)。
  
- **Scratch Buffers 与 Atom 变量**：他们引入了 scratch buffers 以更高效地处理 atom 变量。建议在 scratch buffer 上使用 fp32 atomics，然后读取并舍入/写入（round/write）到 bf16，以避免在全局内存中使用缓慢的 fp32 atomics。

- **分析脚本（Profiling Script）更新**：分析脚本已更新，提高了针对 CUDA 库更新的鲁棒性，并将 NVIDIA kernel 耗时与 llm.c kernel 耗时进行了分离。脚本更改记录在 [此 pull request](https://github.com/karpathy/llm.c/pull/342) 中。

- **PyTorch Padding**：关于对 PyTorch 的词表大小（vocabulary size）进行填充（padding）以进行公平性能比较存在争论，大家承认这并不简单，涉及确保填充的维度不会在 loss 计算或采样过程中被使用。

- **Layernorm 与残差计算**：对话涉及为了稳定性和性能增益，**将 layernorm 的方差（variance）和均值（mean）保存为 fp32**，尽管由于代码简洁性考虑以及激活值使用了 bf16 类型，这尚未在 llm.c 中实现。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms">torch.use_deterministic_algorithms &mdash; PyTorch 2.3 documentation</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code">The Power of 10: Rules for Developing Safety-Critical Code - Wikipedia</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html">Group Calls &mdash; NCCL 2.21.5 documentation</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/331">LLM.c Speed of Light &amp; Beyond (A100 Performance Analysis) · karpathy/llm.c · Discussion #331</a>: 在我昨天的 cuDNN Flash Attention 实现集成之后，我花了一些时间进行 profiling，并试图弄清楚在短期/中期内我们还能在多大程度上提高性能，同时也...</li><li><a href="https://docs.google.com/document/d/1DHFaKHLTVM_zEt2AKJh5fgUNN3oY1bBs1YmxJCcrp9c/edit">3 Strategies for FlashAttention Backwards</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/344">State of the Union [May 3, 2024] · karpathy/llm.c · Discussion #344</a>: [2024年5月3日] 这是 llm.c 项目的第 24 天。我们现在可以进行 multi-GPU 训练，使用 bfloat16 和 Flash Attention，而且速度非常快！🚀 单 GPU 训练方面，我们现在训练 GPT-2 (124M) 的速度更快了...</li><li><a href="https://github.com/karpathy/llm.c/pull/335">v1 of the new matmul backward bias kernel by karpathy · Pull Request #335 · karpathy/llm.c</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/commit/6ebef46f832b4e55b46237c4d06c2597050819ae">ugh didn&#39;t notice this tiny rebasing mistake, introduced a bug. good … · karpathy/llm.c@6ebef46</a>: …CI 的一个很好的候选方案，即我们可以在 train_gpt2.cu 脚本中过拟合单个 batch，并获得与 test_gpt2.cu 文件中预期的完全相同的数值。</li><li><a href="https://github.com/karpathy/llm.c/pull/333">Added FlameGraphs for nsys reports and some nsys documentation by PeterZhizhin · Pull Request #333 · karpathy/llm.c</a>: 这是一个 FlameGraph 示例。在我的机器上捕获。</li><li><a href="https://github.com/karpathy/llm.c/pull/341">GPU auto-detect capability for kernel builds by rosslwheeler · Pull Request #341 · karpathy/llm.c</a>: 对 CI 的修复 - 应该在两种环境中都能工作。如果对 kernel 构建感兴趣，这是一个提议。用法：自动检测 GPU 能力：make（例如，如果你的 GPU 能力类型是 80，那么 --...</li><li><a href="https://github.com/karpathy/llm.c/blob/2c7960040d1d86b6c03a72ef8b32df084e899570/dev/cuda/layernorm_backward.cu#L570">llm.c/dev/cuda/layernorm_backward.cu at 2c7960040d1d86b6c03a72ef8b32df084e899570 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/299/files#diff-bf6b442">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: 更新 residual_forward 以使用 128 位 packed 输入，配合 floatX。之前的 Kernel：block_size 32 | 时间 0.1498 ms | 带宽 503.99 GB/s；block_size 64 | 时间 0.0760 ms | 带宽 993.32 GB/s...</li><li><a href="https://github.com/karpathy/llm.c/pull/338">GELU Fusion with cuBLASLt (SLOWER because it only merges in FP16 mode, not BF16/FP32...) by ademeure · Pull Request #338 · karpathy/llm.c</a>: 事实证明，cuBLASLt 不仅无法将 BF16 GELU（或 RELU）融合进 BF16 matmul，它最终得到的奇怪 kernel 甚至比我们自己的 GELU kernel 还要慢，因为它每次会执行 2 次写入...</li><li><a href="https://github.com/karpathy/llm.c/pull/342">fixed activation gradient resetting for backward pass by ngc92 · Pull Request #342 · karpathy/llm.c</a>: 此外，我们不需要在 zero_grad 中触碰其他 buffer，这些 buffer 在 backward 过程中无论如何都会被多次覆盖。</li><li><a href="https://openhub.net/p/tensorflow">The TensorFlow Open Source Project on Open Hub</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/343">Performance: matmul_bias, cuda streams, fused_classifier (+remove cooperative groups) by ademeure · Pull Request #343 · karpathy/llm.c</a>: 我可能需要将其拆分为多个 PR，请告诉我你的想法（我仍需将新的 kernel 添加到 /dev/cuda/）。主要变更：新的超优化 matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/karpathy/llm.c/commit/79505bc6b3428ad5c2f609046affa1ac34e2f1af#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2764">resolve merge and small fixes · karpathy/llm.c@79505bc</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/299/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R943">Update residual_forward to use packe</a>

d input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: 更新 residual_forward 以使用 128 bit packed input，配合 floatX。前一个 Kernel：block_size 32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size 64 | time 0.0760 ms | bandwidth 993.32 GB/s b...</li><li><a href="https://github.com/karpathy/llm.c/commit/795f8b690cc9b3d2255a19941713b34eeff98d7b">fixes to keep master copy in fp32 of weights optionally · karpathy/llm.c@795f8b6</a>: 未找到描述</li><li><a href="https://github.com/NVIDIA/nccl/issues/338">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: 我使用了来自 https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 的环境，通过多 GPU（使用 horovod 和 nccl）训练 resnet-50，发现 d...</li><li><a href="https://github.com/NVIDIA/nccl/issues/338#issuecomment-1165277390">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: 我使用了来自 https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 的环境，通过多 GPU（使用 horovod 和 nccl）训练 resnet-50，发现 d...</li><li><a href="https://github.com/karpathy/llm.c/pull/303">Updated adamw to use packed data types by ChrisDryden · Pull Request #303 · karpathy/llm.c</a>: 运行前总平均迭代时间：38.547570 ms；运行后总平均迭代时间：37.901735 ms。Kernel 开发文件规格：在当前测试套件下几乎察觉不到：运行前...</li><li><a href="https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/">CUDA Pro Tip: The Fast Way to Query Device Properties | NVIDIA Technical Blog</a>: CUDA 应用程序通常需要知道每个 block 的最大可用 shared memory，或查询活动 GPU 中的 multiprocessors 数量。一种方法是调用... 不幸的是&#8230;
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 条消息): 

neurondeep: 还在 pytorch 网页上添加了 intel
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1235486151003996214)** (350 条消息🔥🔥): 

- **Llama.cpp 集成问题与解决方案**：成员们讨论了将 **llama.cpp** 与 **LM Studio** 集成时遇到的问题。对话涉及对特定文件版本的需求以及 `convert-hf-to-gguf` 脚本的使用，其中一名成员因缺失 `config.json` 遇到 FileNotFoundError，并通过 `huggingface-cli` 重新下载文件解决了该问题。随后针对转换和使用的相关问题进行了协作解决。

- **回滚到之前的 LM Studio 版本**：用户在 **LM Studio** 0.2.22 版本中遇到了 Bug，即 Chat 会提供整个上下文而不仅仅是回复。在多次尝试解决并回滚到 0.2.21 版本后，该问题最终在最新更新中得到修复，并得到了多位用户的确认。

- **LM Studio 终端工具 (`lms`) 发布**：关于新工具 `lms` 的讨论随 **LM Studio 0.2.22** 一同发布，解释了其在自动化任务、启动 API server 以及无需 UI 交互即可管理模型方面的用途。随后的对话澄清了 `lms` 是应用程序的控制器，而非独立工具。

- **运行 LM Studio 无头模式 (Headless Mode)**：多位用户讨论并尝试了各种在无头模式下运行 **LM Studio** 的方法，例如使用 `xvfb-run` 等命令来绕过 GUI 需求。讨论结论是，尽管有社区的变通方案，官方的无头模式支持尚未推出。

- **将 LMS 嵌入可扩展服务器解决方案**：成员们对将 **LM Studio** 嵌入跨集群的高可用服务器模式的潜力表示乐观，并询问了通过 CLI 或 UI 使用特定预设的配置，建议未来进行功能增强。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://releases.lmstudio.ai/windows/0.2.22/c/latest/LM-Studio-0.2.22-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本发布</a>: 未找到描述</li><li><a href="https://tenor.com/view/pout-christian-bale-american-psycho-kissy-face-nod-gif-4860124">Pout Christian Bale GIF - Pout Christian Bale American Psycho - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/squidward-oh-no-hes-hot-shaking-gif-16063591">Squidward Oh No Hes Hot GIF - Squidward Oh No Hes Hot Shaking - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lmstudio.ai/blog/lms">介绍 `lms` - LM Studio 的配套 CLI 工具 | LM Studio</a>: 今天，随着 LM Studio 0.2.22 的发布，我们推出了第一个版本的 lms —— LM Studio 的配套 CLI 工具。</li><li><a href="https://rentry.co/zbofr34p">elija@mx:~$ xvfb-run ./LM_Studio-0.2.22.AppImage</a>: 20:29:24.712 › GPU 信息: '1c:00.0 VGA compatible controller: NVIDIA Corporation G A104 [GeForce RTX 3060 Ti] (rev a1)' 20:29:24.721 › 获取 GPU 类型: nvidia 20:29:24.722 › LM Studio: gpu type = NVIDIA 2...</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2775">Release b2775 · ggerganov/llama.cpp</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/lms?tab=readme-">GitHub - lmstudio-ai/lms: 终端里的 LM Studio</a>: 终端里的 LM Studio。通过在 GitHub 上创建账号来为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lms?tab=readme-ov-file#installation.">GitHub - lmstudio-ai/lms: 终端里的 LM Studio</a>: 终端里的 LM Studio。通过在 GitHub 上创建账号来为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 终端里的 LM Studio</a>: 终端里的 LM Studio。通过在 GitHub 上创建账号来为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama/issues/4051#issuecomment-2092092698">在 GGML/GGUF 上启用 Flash Attention（该功能现已合并至 llama.cpp） · Issue #4051 · ollama/ollama</a>: Flash Attention 已在 llama.cpp 中落地 (ggerganov/llama.cpp#5021)。简而言之，只需向 llama.cpp 的服务器传递 -fa 标志。我们是否可以为 Ollama 服务器提供一个环境变量来传递此标志 ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1235541447042797618)** (159 条消息🔥🔥): 

- **追求高质量故事写作**：一位用户正在寻求帮助，希望在他们的电脑上为 **Goliath 120B Longlora** 创建 **iQuant** 版本，用于高质量的故事写作，这需要至少 8K 的 Context 才能使用；他们提供 Humblebundle Steam 游戏作为协助的奖励。他们强调需要比 LLAMA 3 8B 等模型更高质量的输出，并分享了位于 [Google Docs](https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk) 的 System Prompt 详情。

- **模型召回率实验**：一些对话涉及用户测试各种模型的召回能力，特别是 **LLAMA 3** 召回圣经经文的能力。一位用户为“圣经召回基准测试”建立了一个 [GitHub 仓库](https://github.com/rugg0064/llm-bible-bench)，并观察到在整本圣经中的召回率非常低。

- **探索认知视野**：用户一直在尝试使用模型来观察它们如何召回像圣经这样的大篇幅文本，并讨论创建可以相互通信的实例以获得更好的结果。一位用户提议使用可以优化叙事质量的“Agent”，并引用了一个 [YouTube 视频](https://www.youtube.com/watch?v=sc5sCI4zaic) 作为参考。

- **模板问题与古怪回答**：一位正在尝试新版本 **[ChatQA 1.5](https://huggingface.co/bartowski/Llama-3-ChatQA-1.5-8B-GGUF)** 的用户报告了响应模板中的异常，导致出现奇怪的回答，即使应用了建议的更改（如在 Chat Template 中添加空格或换行符）也是如此。

- **寻求不受限制的代码模型**：一位用户询问是否有 2B 参数规模、审查极少的优秀小型代码模型，但讨论中尚未提供建议。另一位用户正在寻找用于阅读文档和 PDF 的模型，建议将 Cohere 的 **Command-R** 用于文档理解任务，尽管对其硬件要求有所顾虑。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/a/F2mBLoN">GoldenSun3DS 未领取的 Humblebundle 游戏</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行迷因、有趣的 gif、鼓舞人心的故事、病毒式视频等来振奋你的精神...</li><li><a href="https://tenor.com/view/im-out-no-thanks-bugs-bunny-oh-no-not-interested-gif-16824550">我退出，不，谢谢 GIF - 我退出，不，谢谢 Bugs Bunny - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/daleks-exterminate-doctor-who-whovian-gif-10468156">Daleks Exterminate GIF - Daleks Exterminate Doctor Who - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/mradermacher/Goliath-longLORA-120b-rope8-32k-fp16-GGUF">mradermacher/Goliath-longLORA-120b-rope8-32k-fp16-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk">高质量故事写作类型：第三人称</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=sc5sCI4zaic">LLM In-Context Learning 大师课，特邀我的 (r/reddit) AI Agent</a>：LLM In-Context Learning 大师课，特邀我的 (r/reddit) AI Agent👊 成为会员并获取 GitHub 和代码访问权限：https://www.youtube.com/c/AllAboutAI/join...</li><li><a href="https://github.com/rugg0064/llm-bible-bench">GitHub - rugg0064/llm-bible-bench: 一个针对大语言模型及其对圣经经文召回能力的简单测试</a>：一个针对大语言模型及其对圣经经文召回能力的简单测试 - rugg0064/llm-bible-bench</li><li><a href="https://github.com/rugg0064/">rugg0064 - 概览</a>：全栈 Web 开发人员，有时间时会开发一些小项目。 - rugg0064</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg8rhc/1_million_context_llama_3_8b_achieved/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1xrMwhrz4DIdwzY4gI3GIrxQ0phQjVNmu2RGKRnGnRAM/edit?usp=drivesdk">高质量故事写作类型：第一人称</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1235636061879668787)** (2 条消息): 

- **LM Studio 推出配套 CLI 'lms'**：LM Studio 推出了一个新的命令行界面 **lms**，以简化 LLM 的加载/卸载和本地服务器的管理。社区成员可以使用 `npx lmstudio install-cli` 安装该 CLI，并在 [GitHub - lmstudio-ai/lms](https://github.com/lmstudio-ai/lms) 贡献其 MIT 许可的源代码。

- **LM Studio 0.2.22 错误修复版本发布**：**LM Studio 0.2.22** 修复了一个因无意中在响应中包含整个上下文而影响模型响应的错误。遇到此问题的用户可以从 [lmstudio.ai](https://lmstudio.ai) 下载更新版本。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai.">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 终端里的 LM Studio</a>：终端里的 LM Studio。通过在 GitHub 上创建账号来为 lmstudio-ai/lms 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1235581946390646804)** (8 条消息🔥): 

- **Llama 和 Phi-3 配置的快速修复**：如果你删除 **configs 文件夹**并重新启动应用，它将重新填充默认配置。建议先备份任何重要的配置文件。

- **LM Studio 在 WSL 中的困扰**：尝试通过 **Windows Subsystem for Linux (WSL)** 连接到 **LM Studio** 时，如果使用 localhost 地址可能会失败，因为 127.0.0.1 访问的是虚拟机的本地回环。*ipconfig* 可以帮助找到要使用的正确 IP。

- **为 Windows-WSL 通信透传端口**：一位成员建议使用反向代理或通过 `netsh interface portproxy add v4tov4` 命令使用端口代理，以便在 **Windows 和 WSL** 之间为 LM Studio 进行通信。根据另一位成员的说法，不需要通过监听地址增加额外的复杂层。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1235557374119378954)** (4 条消息):

- **寻找 VRAM 修复补丁的链接**：一名成员提到应当链接一个修复补丁，并声称“它确实有效”，是比在 BIOS 中禁用 iGPU 更好的解决方案。
- **寻找增加 VRAM 的 GPU**：一位用户询问是否有“廉价、半高（low profile）、低功耗且具备 12 GB + GDDR6 的 GPU”，用于第二个 PCI-E 插槽，专门为了利用其 VRAM。
- **RTX 3060 作为 VRAM 解决方案**：针对有关 VRAM 使用 GPU 的咨询，另一名成员建议考虑 Nvidia **RTX 3060**。
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1235528523314233406)** (18 messages🔥): 

- **多模型会话上下文窗口混淆**：一名成员讨论了多模型会话（Multi Model Session）功能的一个问题，即无法更改上下文窗口大小（默认为 2048），并且在请求排队时会出现超时。他们指出工具进入了 *'Rolling Window'* 模式并生成了无关的响应。

- **Ubuntu 用户运行技巧**：针对在 Ubuntu 上运行该工具的问题，提供了一套简单的指令：下载 Appimage，赋予可执行权限，然后运行应用程序。

- **Docker 爱好者可以运行无头模式**：软件的一项改进现在允许以无头（headless）模式运行，一名成员指出这终于让他们能够创建一个用于测试的 Docker 镜像。

- **配置和 CLI 问题解答**：成员们询问了如何通过 CLI 持久化 GPU offload 和 CORS 等设置，另一名成员澄清说，“My Models”页面中的模型配置是默认值，但可以在 CLI/SDK 中按字段进行覆盖。

- **GPU 层配置中发现潜在 Bug**：有报告称，通过 CLI 加载模型时，GPU 层的配置预设会被覆盖。建议在 GitHub 上提交 issue 来解决此问题，并提供了配置 schema 的链接以供参考可用参数。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lmstudio-ai/lms/issues">Issues · lmstudio-ai/lms</a>：终端里的 LM Studio。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lms/issues/6">BUG: 通过 CLI 加载模型时忽略配置预设中的 &quot;n_gpu_layers&quot; 参数 · Issue #6 · lmstudio-ai/lms</a>：我在选定为默认的模型预设中设置了 &quot;n_gpu_layers&quot;: -1。然而，当我使用 cli 加载该模型时 lms load --identifier llama3-8b-8k >> select model ...</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/schema.json#L26">configs/schema.json at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1235666725496684626)** (32 messages🔥): 

- **LM Studio CLI 发布 ROCm 版本**：LM Studio 推出了 `lms`，这是一个用于管理 LLM 并在 AMD ROCm Preview Beta 上运行本地服务器的新 CLI，现已在 [GitHub 上开源](https://github.com/lmstudio-ai/lms)。用户可以下载最新的 LM Studio 0.2.22 ROCm Preview 来使用 `lms`，它还为新用户预打包了 OpenCl。
  
- **API 响应中包含 Prompt 的 Bug 已确认**：一位用户注意到 Prompt 被包含在 API 响应中，这是最新版本中的一个已知问题。LM Studio 团队迅速承认并确认[已发布紧急修复](https://lmstudio.ai/rocm)，用户已验证修复有效。

- **大上下文尺寸探索**：一位参与者通过尝试在 Phi 3 上使用 131072 token 的上下文来测试 RAM 扩展与上下文大小的关系，但失败了。不过，他们成功在拥有 32 GB RAM 的 7900XTX GPU 上运行了 60000 token 的上下文。

- **寻求 Embedding 模型问题的澄清**：用户报告了在新版本中尝试加载 Embedding 模型时的问题。LM Studio 立即发布了修复程序，用户确认从 [LM Studio ROCm 下载页面](https://lmstudio.ai/rocm)重新下载后问题已解决。

- **关于 ROCm 的 Linux 支持讨论**：参与者正在讨论在 Linux 上运行 ROCm，其中一人分享了在 Mesa 的 opencl 实现上使用 ROCm 的经验，并希望有一个支持 Linux 的 ROCm 构建版本；另一人建议使用 lm-studio 下载模型并用于本地 llama.cpp 构建可能是一个变通方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm,">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://tenor.com/view/oil-gif-21418714">Oil GIF - Oil - 发现并分享 GIFs</a>：点击查看 GIF</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22-ROCm-Preview/beta/LM-Studio-0.2.22-ROCm-Preview-Setup.exe">未找到标题</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 终端中的 LM Studio</a>：终端中的 LM Studio。通过在 GitHub 上创建账户，为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : 由 ggerganov 添加 Flash Attention · Pull Request #5021 · ggerganov/llama.cpp</a>：参考 #3365 设置 ggml 和 llama.cpp 中支持 Flash Attention 所需的内容。提议的操作执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused scale ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1235733449042821140)** (1 条消息): 

- **LM Studio v0.0.22 更新 Llama 模型**：LM Studio 的最新更新包括对 `llama.cpp` 的重大改进，解决了 **Llama 3** 和 **BPE 模型** 的问题。带有 **BPE-fix 标签** 的 Llama 3 8B、70B instruct 和 Phi-3 模型版本可在提供的 Hugging Face 链接处下载。
    - [Meta-Llama-3-8B-Instruct-BPE-fix](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-BPE-fix-GGUF)
    - [Meta-Llama-3-70B-Instruct-BPE-fix](https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-BPE-fix-GGUF)
    - [Phi-3-mini-4k-instruct-BPE-fix](https://huggingface.co/lmstudio-community/Phi-3-mini-4k-instruct-BPE-fix-GGUF)
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1235640547545448448)** (69 条消息🔥🔥): 

- **推出 LM Studio 的 CLI 伴侣工具**：LM Studio 发布了新的 CLI 工具 `lms`，旨在方便加载 LLM、启动/停止服务器以及进行调试。用户可以[直接安装它](https://github.com/lmstudio-ai/lms)，该工具需要 LM Studio 0.2.22 或更高版本。

- **运行 LM Studio 的无头模式（Headless）教程**：一位成员分享了一个*编写粗糙的黑客式无头模式教程*，用于在无界面环境下运行 LM Studio，其中包括使用 xvfb 模拟 X11 会话和引导 `lms` 的说明。另一位成员确认在经过一些故障排除后，该方法在 Ubuntu Server 上成功运行。
  
- **解决应用退出问题**：多条消息集中讨论了一个问题，即 LM Studio 应用在执行命令时退出，讨论了包括使用 `ctrl+z` 后接 `bg`、`disown -ah` 以及 `--no-sandbox` 标志等故障排除步骤。

- **通过脚本简化安装流程**：一位成员表示打算创建一个脚本，自动完成 LM Studio 的无头安装，从而实现通过一条命令进行更直接的设置。

- **容器化（Dockerization）进展**：在无头安装教程成功后，一位成员对能够为 LM Studio 创建 Docker 容器表示兴奋，这将简化在服务器上运行和测试模型的过程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/lmstudioai/status/1786076035789815998?s=46">来自 LM Studio (@LMStudioAI) 的推文</a>：介绍 lms —— LM Studio 的伴侣 CLI 😎 ✨ 加载/卸载 LLM，启动/停止本地服务器 📖 使用 lms log stream 调试你的工作流 🛠️ 运行 `npx lmstudio install-cli` 来安装 lms 🏡 ...</li><li><a href="https://releases.lmstudio.ai/linux/0.2.22.b/beta/LM_Studio-0.2.22.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - 发现并分享 GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 终端中的 LM Studio</a>：终端中的 LM Studio。通过在 GitHub 上创建账户，为 lmstudio-ai/lms 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235982270985142463)** (1 条消息): 

- **Pages Beta 测试人员招募结束**：**Pages** 的 Beta 测试人员招募已达到预期人数。团队表达了感谢，并建议大家关注 Pages 开发的进一步更新。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1235490446109577216)** (308 条消息🔥🔥):

- **Perplexity 的技术困难**：用户报告了 Perplexity 在 Safari 和 Brave 浏览器上无法正常运行的问题，例如由于按钮无响应而无法发送 prompts 或注册。其他用户在对话中遇到了持续从之前上传的文件中获取资源的问题，错误地检索了早期请求的数据。
  
- **订阅和支付查询**：一位用户询问了关于不想要的月度订阅费用的退款事宜，并被建议联系 support@perplexity.ai 寻求帮助。

- **对 Perplexity 工具的功能请求和反馈**：成员们表达了对改进语音命令功能的渴望以及对某些功能的延续，建议进行增强，如避免过早终止命令并启用持续监听。

- **讨论使用情况和模型限制**：对于 Perplexity 内部不同模型和工具的使用限制存在困惑，一些用户不确定每日查询配额，而另一些用户则在讨论 Gemini 1.5 Pro、Claude Opus 和 GPT-4 Turbo 等不同 AI 模型的比较能力。

- **对未来 AI 发展和竞争对手平台的期待**：社区期待新的 AI 模型，如传闻中的 "GPT-5" 以及来自 OpenAI 的潜在 Perplexity 竞争对手。此外，还有关于搜索引擎和知识引擎之间区别的讨论，并推测这些技术进步将如何演变并与现有平台集成。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/24111326/ai-search-perplexity-copilot-you-google-review">Here’s why AI search engines really can’t kill Google</a>：搜索引擎不仅仅是搜索引擎，AI 仍然无法完全跟上。</li><li><a href="https://tenor.com/view/imagination-spongebob-squarepants-dreams-magic-gif-12725683">Imagination Spongebob Squarepants GIF - Imagination Spongebob Squarepants Dreams - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://aistudio.google.com/app/prompts/new_chat">no title found</a>：未找到描述</li><li><a href="https://youtu.be/77IqNP6rNL8">New OpenAI Model &#39;Imminent&#39; and AI Stakes Get Raised (plus Med Gemini, GPT 2 Chatbot and Scale AI)</a>：Altman “知道发布日期”，据知情人士透露 Politico 称其“即将发布”，然后是神秘的 GPT-2 聊天机器人 [由 Microsoft 的 phi 团队制作]...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1235488309426257961)** (22 messages🔥): 

- **链接分享协议提醒**：发布了 [Perplexity AI 提醒](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)，以确保在 Discord 上发布之前将线程设置为 *Shareable*（可分享）。
- **为月球粉丝准备的月球之爱**：为月球话题爱好者分享了一个有趣的答案，链接指向 [Perplexity 关于月球虚构名称的回答](https://www.perplexity.ai/search/In-the-fictional-ySifBWwWSeeXk27x5glOOw)。
- **AI 中的旋律发现**：一位成员分享了与音乐创作相关的链接，特别是作品 "We Interface" 及其通过 AI 在 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/We-Interface-song-kK0EbdFjR2yh_7Vn3Xbg0Q) 中的发现。
- **打印机技术的隐私担忧**：隐私爱好者或任何对打印机追踪点感兴趣的人可以在此 [Perplexity 搜索结果](https://www.perplexity.ai/search/Printer-Tracking-Dots-NcXiviwKQS2nGmbu4lqEnw) 中找到信息。
- **探索 AI 生成内容**：指向关于新 AI 首次亮相的 Perplexity AI 内容链接，[Exploring XDreams](https://www.perplexity.ai/page/Exploring-XDreams-Debut-UyAeq.Q_TxyMAP2avgAELw) 和 [XAvengers: Cyborg Wars](https://www.perplexity.ai/page/XAvengers-Cyborg-Wars-Ku62KQMbQ8W.9EqM9RTs1g) 表明人们对 AI 生成的叙事和游戏的兴趣日益增长。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1235604126428299344)** (41 messages🔥): 

- **Sonar Large 可用性及拼写错误澄清**：**Sonar Large** 可通过 API 使用，且 [模型卡片文档](https://docs.perplexity.ai/docs/model-cards) 显示其列出的上下文长度为 32k。关于参数数量的困惑导致了一项澄清，即 *Sonar Large* 是一个 70B 模型，与暗示其为 8x7B 的拼写错误相反。
  
- **Prompt 精确度带来更好结果**：成员们注意到在 prompts 中使用精确术语会带来更好的结果，例如在 URL 前指定 `https://`。一位用户在使用 **llama-3-sonar-large-32k-online** 时的经验表明，在调整 prompts 以生成竞争对手的 markdown 列表后，获得了更好的结果。

- **API Client 遇到结果波动**：即使在调整了 prompt 之后，有用户反映 API 的输出结果不一致，有时能提供正确的竞争对手信息，有时则会失败。建议通过微调 AI model 设置和进行 prompt optimization 来解决此问题。

- **寻求模型迁移指导**：有用户咨询关于从 **sonar-medium-online** 迁移到新模型的需求。得到的建议是尝试使用 **llama-3-sonar-small-32k-online** 以获得更高的准确度，并明确指出最终有必要更新到新模型。

- **调整 AI 参数以优化响应**：为了提高准确率，用户测试了不同的 `frequency_penalty`、`temperature` 和 `top_p` 设置，发现调整这些参数会影响 AI 响应的相关性和正确性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://optonal.com`">未找到标题</a>: 未找到描述</li><li><a href="http://www.ghirardelli.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.godiva.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.lindt.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.russellstover.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.hersheys.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.dovechocolate.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.toblerone.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.lamaisonduchocolat.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.pierremarcolini.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.vosgeshautchocolat.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.teuscher.com)">未找到标题</a>: 未找到描述</li><li><a href="https://"">未找到标题</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: 未找到描述</li><li><a href="https://optonal.com">OpTonal • 为使用 Slack, HubSpot, Google Meet 的团队提供的 AI Sales Agent</a>: 未找到描述</li><li><a href="https://sensiseeds.com](https://sensiseeds.com)\n2.">未找到标题</a>: 未找到描述</li><li><a href="https://seed.com](https://seed.com)">未找到标题</a>: 未找到描述</li><li><a href="https://www.salesforce.com/">Salesforce：以客户为中心的企业</a>: Salesforce 是排名第一的 AI CRM，通过统一的 Einstein 1 平台将 CRM、AI、数据和信任结合，使公司能够与客户建立联系。</li><li><a href="https://www.hubspot.com/products/sales">适用于小型到大型企业的销售软件 | 免费开始使用</a>: 强大的销售软件，帮助您的团队在统一的平台上达成更多交易、深化关系并更有效地管理销售漏斗。</li><li><a href="https://www.zoho.com/crm/">Zoho CRM | 客户评价最高的销售 CRM 软件</a>: Zoho CRM 是一款在线销售 CRM 软件，在单一 CRM 平台上管理您的销售、营销和支持。全球超过 1 亿用户信赖！立即注册免费试用。</li><li><a href="https://www.gong.io/">Gong - 营收智能平台</a>: Gong 捕获客户互动并大规模提供洞察，赋能团队基于数据而非主观意见做出决策。</li><li><a href="https://www.exceed.ai/">排名第一的对话式营销和销售平台 - Exceed.ai</a>: 利用对话式 AI 提升潜在客户转化率。自动化营收互动，实现大规模参与，并通过 Email、Chat、SMS 进行互动。</li><li><a href="https://salesloft.com/">Salesloft：领先的销售参与平台</a>: 未找到描述</li><li><a href="https://www.yesware.com/">让销售参与变得简单 | Yesware</a>: Yesware 帮助高效销售团队大规模进行有意义的电子邮件推广。如果您需要通过电子邮件推广推动更多收入，但又觉得复杂的平台过于繁琐，请尝试 Yesware。</li><li><a href="http://ghirardelli.com)">未找到标题</a>: 未找到描述</li><li><a href="http://hersheys.com)">未找到标题</a>: 未找到描述</li><li><a href="http://russellstover.com)">未找到标题</a>: 未找到描述</li><li><a href="http://lindt.com)">未找到标题</a>: 未找到描述</li><li><a href="http://godiva.com)">未找到标题</a>: 未找到描述</li><li><a href="https://sidecardoughnuts.com/)">Sidecar Doughnuts - 世界上最新鲜的甜甜圈！</a>: 自 2012 年起提供世界上最新鲜的甜甜圈、招牌混合咖啡和微笑服务 | 加州 Costa Mesa, Santa Monica, &amp; Del Mar</li><li><a href="https://thepieholela.com/)">The Pie Hole</a>: 下次活动需要新鲜派或 Pie Holes 吗？在线下单，全国免费送货，因为派就是爱。
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1235508251219591230)** (15 条消息🔥):

- **探索逆因果律与道德**：讨论了逆因果律（Retrocausality）中的“道德非交换性”概念，强调了心理学视角下患者在道德行为中不区分原因和结果，从而影响观察者道德框架完整性的现象。
- **寻求 llamacpp 指导**：一名成员在遇到模型生成无意义输出以及网站自动编写 C 函数的问题后，请求一份 **llamacpp** 的初学者指南。
- **在 CPU 上使用 Llama**：建议使用 **ollama** 作为 **llamacpp** 的后端以避免直接处理 C 语言，并讨论了利用量化（quantization）和剪枝（pruning）等技术在 CPU 上运行 **LLM** 等大语言模型的进展。
- **等待 lmstudio 审批**：一位成员表达了因等待 **lmstudio** 审批而无法使用笔记本电脑进行模型相关任务的沮丧。
- **圣彼得堡随时间的变迁**：2002 年与 2024 年圣彼得堡起义广场（Vosstaniya Square）利戈夫斯基大街（Ligovsky Avenue）的照片引发了一个关于相机色彩准确度提高的笑话。
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1235647006773608500)** (19 messages🔥): 

- **私有帖子的悬念**：一条 **proprietary**（私有）的 Twitter 帖子引起了兴趣，但细节尚未公开，未提供进一步信息。
- **Haystack 走向嵌入式**：重点介绍了 carsonpo 的 **haystack-embedded** **GitHub** 仓库，这是一个用于嵌入式机器学习开发的开源贡献，可在此处 [访问](https://github.com/carsonpo/haystack-embedded)。
- **对 WildChat 数据集的兴奋**：allenai 的 **WildChat** 数据集引发了讨论，但访问需要同意 [AI2 ImpACT License](https://allenai.org/licenses/impact-lr)。该数据集似乎以“长多轮对话”为特色，托管在 **Hugging Face** 平台上，URL 指向 [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)，预示着新版本的到来。
- **关于数据集发布的矛盾信息**：讨论集中在新的 **WildChat** 数据集是否已经开源，通过 [arXiv abstract](https://arxiv.org/abs/2405.01470) 上的链接得到了确认。
- **长对话中偏好 OPUS**：一位成员提到在处理长对话上下文时，相比其他模型更倾向于使用 **OPUS** 模型，并建议在“10/20k 的 prompting”之后表现更好。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/allenai/WildChat-1M">allenai/WildChat-1M · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/allenai/WildChat?not-for-all-audiences=true">allenai/WildChat · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/carsonpo/haystack-embedded">GitHub - carsonpo/haystack-embedded</a>：通过在 GitHub 上创建账号为 carsonpo/haystack-embedded 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1235486410937466891)** (104 messages🔥🔥): 

- **Hermes 升级发布**：Nous 发布了带有 **LLaMA** 权重的 **Hermes 2 Pro**，具备良好的 QA、函数调用（**Function Calling**）和带有视觉多模态的 **JSON Mode**。模型和测试代码可在 [Hugging Face](https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B) 上获取。

- **启用高级函数调用**：BLOC97 解释说，具有 **Function Calling** 能力的 **LLM** 能够感知外部函数/工具调用以验证答案，而不是模拟答案；teknium 分享了一个 **GitHub** 仓库，其中包含针对 [Hermes](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) 的特定函数调用微调示例。

- **函数调用数据集见解**：分享了 [Glaive function-calling dataset V2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)，以展示用于训练具有函数调用功能模型的结构。围绕这些数据集使用的讨论强调了它们在高级 **LLM** 应用中的潜力。

- **llama.cpp 对模型性能的影响**：Diabolic6045 在具有 8GB RAM 的 Android 设备上使用 **llama.cpp** 运行 **Hermes 2 Pro** 时，体验到了极佳的推理速度，突显了该技术的效率。

- **结合 CrewAI 和 LocalAI 利用 Hermes 2 Pro**：.interstellarninja 通过分享一个 [Jupyter notebook](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb)，提供了在 CrewAI 中使用 Hermes 2 Pro function-calling 的解决方案。他们还指出 LocalAI API 支持 OpenAI API tool calls 格式的 function-calling，详情见其 [repository](https://github.com/mudler/LocalAI/blob/master/gallery/hermes-2-pro-mistral.yaml)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.17733">Building a Large Japanese Web Corpus for Large Language Models</a>：开源日语大语言模型 (LLMs) 已在 CC-100、mC4 和 OSCAR 等语料库的日语部分进行了训练。然而，这些语料库并非针对日语文本质量而创建...</li><li><a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing#scrollTo=2EoxY5i1CWe3">Google Colab</a>：未找到描述</li><li><a href="https://x.com/DimitrisPapail/status/1786045418586972208">Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：这份报告中最令人惊讶的发现隐藏在附录中。在两个 prompt 中表现最好的情况下，模型并没有像摘要声称的那样过度拟合。这里是原始 GSM8k 对比 ...</li><li><a href="https://huggingface.co/vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B">vonjack/Nous-Hermes-2-Pro-Xtuner-LLaVA-v1_1-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.25-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.25-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb">Hermes-Function-Calling/examples/crewai_agents.ipynb at main · NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账号，为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账号，为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://github.com/e2b-dev/code-interpreter">GitHub - e2b-dev/code-interpreter: 用于为你的 AI 应用添加代码解释功能的 Python & JS/TS SDK</a>：用于为你的 AI 应用添加代码解释功能的 Python & JS/TS SDK - GitHub - e2b-dev/code-interpreter</li><li><a href="https://github.com/mudler/LocalAI/blob/master/gallery/hermes-2-pro-mistral.yaml">LocalAI/gallery/hermes-2-pro-mistral.yaml at master · mudler/LocalAI</a>：🤖 免费、开源的 OpenAI 替代方案。自托管、社区驱动且本地优先。可在消费级硬件上运行的 OpenAI 无缝替换方案。无需 GPU。支持运行 gguf, trans...</li><li><a href="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2">glaiveai/glaive-function-calling-v2 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1235582545362685993)** (45 条消息🔥): 

- **ChatML 配置揭晓**：成员们正在讨论启用 ChatML 所需的修改，提到了 **token 替换** 以及模型配置中的调整，例如将 EOS 替换为 `
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.writewithlaika.com)">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1235579505632874569)** (1 条消息): 

- **对 LLM 微调的热情**：一位新成员表达了在成为矿工之前对**微调大语言模型 (LLM)** 的浓厚兴趣。他们寻求关于如何寻找适合此目的的数据集，以及有效微调所需数据类型的建议。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 条消息): 

felixultimaforeverromanempire: 有人知道好的免费通用数据集吗？
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1235586429401432156)** (86 条消息 🔥🔥): 

<ul>
<li><strong>World-sim 中的铁器时代更新</strong>：一位成员提到在游戏中处于第 11 世界，那里最近实施了铁器时代更新。</li>
<li><strong>对游戏《孢子》(Spore) 的怀旧</strong>：一位成员回忆起在《孢子》(Spore) 游戏中花费了超过 100 小时。</li>
<li><strong>对即将到来的更新和庆祝活动的期待</strong>：一位成员对本周末即将到来的事情表示兴奋，并分享说他们即将满 18 岁，这是一个重要的生日。</li>
<li><strong>关于 AI 与意识的讨论</strong>：成员们对 Joscha 关于意识的演讲表示赞赏，称其影响深远，并分享了该主题相关的 <a href="https://www.youtube.com/watch?v=abWnhmZIL3w">YouTube</a> 视频。</li>
<li><strong>用于 Worldsim 更新的新 Discord 身份组 (Role)</strong>：创建了一个新身份组，用于在发布较小的 worldsim/worldclient 相关信息时提醒成员，几位成员请求加入，该身份组可以通过 &lt;id:customize&gt; 频道获取。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/cs9Ls0m5QVE?si=YD9rEG7jZNBUJbpS">37C3 - 合成感知 (Synthetic Sentience)</a>：https://media.ccc.de/v/37c3-12167-synthetic_sentience 人工智能能产生意识吗？尽管 AI 能力取得了飞速进步，但核心问题...</li><li><a href="https://youtu.be/YZl4zom3q2g?si=xqoxcI1yibo5Td1H">Joscha Bach 的数字万物有灵论 (Cyber Animism)</a>：这是 Joscha Bach (http://bach.ai/) 在我们中心进行的 1 小时 45 分钟的演讲。</li><li><a href="https://www.youtube.com/watch?v=abWnhmZIL3w">世界模拟讲座 @ AGI House SF</a>：0:00 对话 1:31 Jeremy Nixon 开场 6:08 Nous Research 的 Karan Malhotra 26:22 Websim CEO Rob Hasfield 1:00:08 Midjourney 的 Ivan Vendrov [实时...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1235660780351787008)** (99 条消息 🔥🔥):

- **Mojo 加入编程语言竞赛**：分享了一个 [由 Chris Lattner 主讲的 YouTube 视频](https://www.youtube.com/watch?v=JRcXUuQYR90)，讨论了“Mojo Lang - 明天的高性能 Python？”，强调了这种新语言试图整合来自 CPU/GPU 开发的最佳编程技术。
- **以 Python 背景学习 Mojo**：讨论集中在 Python 与新 Mojo 语言之间的关系，成员们指出，虽然 Mojo 与 Python 有相似之处并可以直接使用 Python 对象，但由于强类型检查和其他系统编程特性，两者存在显著差异。推荐那些希望了解 Mojo 独特特性的人阅读 [Mojo 文档](https://docs.modular.com/mojo/manual/basics)。
- **开源贡献与指导**：鼓励成员为开源的 Mojo 标准库做贡献，并提供了 [GitHub 贡献指南](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md) 的链接以及一篇 [Modular 博客文章](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide)，为潜在贡献者提供分步指导。
- **关于开发协调的讨论**：目前正在就如何最好地管理贡献并避免 GitHub issues 上的重复劳动进行对话。其中一项提议包括使用 [PR 模板](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository) 来帮助有效地关联 issues 和 PR。
- **Mojo 优势评估**：对话深入探讨了 Mojo 的独特之处，如性能、可预测性和可移植性特征。还提到 Mojo 的构建系统会自动进行 autotune，以在不同硬件上实现最佳性能，正如 [Jeremy Howard 关于 autotuning 的视频](https://youtu.be/6GvB5lZJqcE?t=281) 所演示的那样。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository">Creating a pull request template for your repository - GitHub Docs</a>: 未找到描述</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何为 Mojo 标准库做贡献：分步指南</li><li><a href="https://devblogs.microsoft.com/oldnewthing/20091201-00/?p=15843)">Microspeak: Cookie licking - The Old New Thing</a>: 现在没别人能占有它了。</li><li><a href="https://docs.modular.com/mojo/manual/basics">Introduction to Mojo | Modular Docs</a>: Mojo 基础语言特性介绍。</li><li><a href="https://open.spotify.com/track/3XwQ8ks84wlj3YcRyxXrlN?si=XJlRyCe_TzOmqPwVtDbCQQ&utm_source=copy-link">Mojo</a>: -M- · 歌曲 · 2012</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo 是由 Swift 和 LLVM 的创建者推出的最新语言。它尝试从 CPU/GPU 级编程中提取一些最佳技术并封装...</li><li><a href="https://github.com/apple/swift/issues/43464">[SR-852] [QoI] Poor diagnostic with missing &quot;self.&quot; in convenience initializer · Issue #43464 · apple/swift</a>: 前 ID SR-852 Radar 无 原始报告人 @ddunbar 类型 Bug 状态 已解决 解决结果 已完成 来自 JIRA 的更多细节 投票 0 组件 编译器 标签 Bug, DiagnosticsQoI 负责人 @dduan...</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#create-a-pull-request">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://youtu.be/JRcXUuQYR90?t=113)?">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo 是由 Swift 和 LLVM 的创建者推出的最新语言。它尝试从 CPU/GPU 级编程中提取一些最佳技术并封装...</li><li><a href="https://github.com/modularml/mojo/issues/2487">[Feature Request] Make the `msg` argument of `assert_true/false/...` keyword only · Issue #2487 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？如标题所示。你进行此更改的动机是什么？为了...</li><li><a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------Mojo 🔥：一种用于异构计算的系统编程语言 演讲者：Abdul Dakkak, Chr...</li><li><a href="https://youtu.be/6GvB5lZJqcE?t=281)">Jeremy Howard demo for Mojo launch</a>: 这是 Modular 发布视频的一个片段。完整的视频、文档和详情请见：https://www.modular.com/</li><li><a href="https://github.com/modularml/mojo/issues/2415">[Feature Request] Add `__rfloordiv__()` to SIMD type · Issue #2415 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？Int 和 Object 类型支持 rfloordiv。我添加了...</li><li><a href="https://github.com/modularml/mojo/pull/2457">[stdlib] Support print to stderr by GeauxEric · Pull Request #2457 · modularml/mojo</a>: 为 print 函数添加关键字参数以支持流向 stderr。修复 #2453。签署人：Yun Ding yunding.eric@gmail.com
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1235662100731265147)** (3 条消息): 

- **Modular 的最新推文**: Modular 分享了一条推文，可通过[此链接](https://twitter.com/Modular/status/1786096043463184528)访问，但推文内容未被讨论。
- **Modular 的另一条推文**: Modular 分享了第二条推文，可以在[这里](https://twitter.com/Modular/status/1786096058113876311)找到，不过没有提供关于该推文的进一步细节或讨论点。
- **Modular 再次发布推文**: Modular 发布了另一条推文，可以点击[此链接](https://twitter.com/Modular/status/1786483510141657384)查看。聊天中没有随附的对话或对其重要性的解释。
  

---

**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1235652713954676849)** (2 messages): 

- **Modular 庆祝 Mojo 24.3 中的社区贡献**：Mojo 🔥 24.3 已发布，在 Mojo 标准库开源后，社区参与度显著提高。该更新包含了增强平台能力的贡献，特别感谢 [@LJ-9801](https://github.com/LJ-9801)、[@mikowals](https://github.com/mikowals) 以及发布说明中列出的其他贡献者。

- **发布具备引擎可扩展性的 MAX 24.3**：MAX 24.3 更新引入了全新的 MAX Engine Extensibility API，增强了开发者高效构建和运行 AI 流水线的能力。此版本为 PyTorch、ONNX 和 Mojo 模型提供了改进的集成，并通过 [MAX Graph APIs](https://docs.modular.com/engine/graph) 为多种硬件提供了一系列性能优化。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/max-24-3-introducing-max-engine-extensibility">Modular: MAX 24.3 - Introducing MAX Engine Extensibility</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：MAX 24.3 - 介绍 MAX Engine Extensibility</li><li><a href="https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements">Modular: What’s New in Mojo 24.3: Community Contributions, Pythonic Collections and Core Language Enhancements</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 24.3 的新变化：社区贡献、Pythonic 集合和核心语言增强
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1235659563408166933)** (1 messages): 

- **MAX ⚡️ 和 Mojo 🔥 24.3 版本正式上线**：**Release 24.3** 现已发布，包含最新版本的 MAX 和 Mojo。提供了安装命令，可以通过简单的 curl 脚本和 Modular CLI 命令进行访问。
- **庆祝 Mojo 🔥 一周年**：此次更新标志着 Mojo 的**一周年纪念**，感谢社区对该版本所做的贡献。
- **发布博客及可扩展性功能详解**：感兴趣的用户可以在 [官方博客文章](https://modul.ar/24-3) 中阅读关于此次发布的内容，并在专门的 [博客文章](https://modul.ar/max-extensibility) 中了解新的 **MAX 可扩展性**功能。
- **社区贡献获得认可**：变更日志提到了由社区贡献的 **32 项重大更改、修复和功能**，突显了开发过程中的协作努力。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1235664363566923809)** (4 messages): 

- **模拟意识的挑战**：讨论涉及了模拟意识的复杂性，认为这不仅需要科学理解，还需要哲学见解。有人提议从更简单的生物开始可能是关键，因为它们的大脑可能更容易被映射并复制到代码中。

- **Hoffman 的工作启发了未来的学术研究**：一位成员表达了计划转学到 UCI，以便更接近 Donald Hoffman 教授的工作，他正致力于映射意识体验。这与功能主义（functionalism）的观点一致，即模拟大脑功能可能比完全复制大脑更可行。

- **渴望探索意识**：另一位成员分享了他们致力于意识模拟工作的目标，与之前关于该主题的讨论产生了共鸣。
  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1235612025615421545)** (2 messages): 

- **CHERI 逐渐进入日常使用**：聊天中强调，**Capability Hardware Enhanced RISC Instructions (CHERI)** 在提高硬件安全性方面具有巨大潜力，有可能使当前 70% 的漏洞利用失效。这一讨论是由最近的一个 [会议播放列表](https://youtube.com/playlist?list=PL55r1-RCAaGU6fU2o34pwlb6ytWqkuzoi) 引发的，该列表深入探讨了 CHERI 生态系统内的进展。

- **软件开发的范式转移**：随着 CHERI 的采用，软件开发可能会迎来巨大的转变，因为处理过程可能会快上几个数量级，从而实现具有高性能的高效 UNIX 风格编程。这种可能性是在 [CHERI 促进极速 IPC](https://github.com/CTSRD-CHERI/cheripedia/wiki/Colocation-Tutorial) 以及此类功能固有优势的背景下讨论的。

- **沙盒进入快车道**：对话转向了 CHERI 的 *可扩展分隔化 (scalable compartmentalization)* 如何从根本上改变使用沙盒的环境，影响 Web 浏览器、虚拟机甚至边缘计算。引用了一个 [YouTube 视频](https://youtu.be/_QxXiTv1hH0?t=933)，展示了这一变革性技术。

- **传统安全措施可能变得多余**：普遍推测随着 CHERI 的兴起，传统的硬件安全措施（如基于 MMU 的内存保护或地址空间布局随机化 ASLR）可能会变得过时，从而简化硬件设计并提升软件速度。

- **微内核可能走向舞台中央**：一位成员思考 CHERI 是否会引发 OS 开发的革命，抵消微内核中传统的高昂 IPC 成本，使其成为一种潜在的主导架构。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/CTSRD-CHERI/cheripedia/wiki/Colocation-Tutorial)">Home</a>：CHERI Wiki 页面的占位符。通过在 GitHub 上创建账号来为 CTSRD-CHERI/cheripedia 的开发做出贡献。</li><li><a href="https://youtu.be/_QxXiTv1hH0?t=933)">未来的硬件能让我们的软件更安全吗？活动完整录像 - 剑桥 2022 年 3 月 15 日</a>：未来的硬件如何让我们的软件更安全？对代码中的安全问题感到沮丧？讨厌那些主动找上门而不是被你发现的 Bug？你是否感兴趣...</li><li><a href="https://youtu.be/_QxXiTv1hH0?t=1204))">未来的硬件能让我们的软件更安全吗？活动完整录像 - 剑桥 2022 年 3 月 15 日</a>：未来的硬件如何让我们的软件更安全？对代码中的安全问题感到沮丧？讨厌那些主动找上门而不是被你发现的 Bug？你是否感兴趣...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1235514390636396596)** (137 条消息🔥🔥): 

- **Mojo 引用语义仍在变动中**：Mojo 关于引用（references）和生命周期（lifetimes）的语义正在积极设计中，旨在提供比现有原型更**简单且灵活**的结构。设计方案将公开分享，目前关于引用和生命周期的语义是接近完成还是会继续增加层级仍存在争议。
- **崩溃报告与 Bug 追踪**：讨论指向了一个与 `struct lifetime` 相关的 [崩溃报告和 Bug](https://github.com/modularml/mojo/issues/2429)，需要引起注意。人们对编译器崩溃表示担忧，认为编译器应该提供有意义的错误信息。
- **InlineArray 的吸引力与问题**：`InlineArray` 尚未进入稳定版本；尽管它很有用，但在处理大型数组时存在已知的问题，相关的 GitHub issue 表明它正在等待更好的稳定性。该功能在 `utils.InlineArray` 中实现。
- **讨论 Mojo 的 GPU 支持**：预计 Mojo 很快将支持 GPU，从 Nvidia 开始，利用 MLIR 实现跨平台的通用性。同时，讨论明确了现有语言从 LLVM 迁移到 MLIR 是一个耗时数年的工程，这使得 Mojo 原生的 MLIR 集成显得尤为特殊。
- **对 Snap 包和 I/O 函数的兴趣**：有人请求在 Ubuntu 的 Snap Store 上发布官方 Snap 包，并讨论了 Mojo 当前 I/O 模块处于基础阶段的现状，目前实现简单的用户输入功能（如从 `stdin` 读取）仍需从 Python 导入。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140901/233938.html"> [llvm] r217292 - [docs] 文档说明提交信息中 "NFC" 的含义。
   </a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/os/atomic">atomic | Modular 文档</a>: 实现 Atomic 类。</li><li><a href="https://www.modular.com/blog/whats-new-in-mojo-24-3-community-contributions-pythonic-collections-and-core-language-enhancements#:~:text=This%20simplifies%20compare%20List%5BTuple%5BFloat64%2C%20Float64%2C%20Float64%5D%5D()%20vs%20List%5B(Float64%2C%20Float64%2C%20Float64)%5D()">Modular: Mojo 24.3 新特性：社区贡献、Pythonic 集合和核心语言增强</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 24.3 新特性：社区贡献、Pythonic 集合和核心语言增强</li><li><a href="https://github.com/modularml/mojo/issues/2425.">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2413">[功能请求] 允许子 trait 替换父 trait · Issue #2413 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？如果一个函数接收由 trait 绑定的变长参数...</li><li><a href="https://github.com/modularml/mojo/issues/2429">[mojo-nightly] struct 生命周期问题 · Issue #2429 · modularml/mojo</a>: Bug 描述：在以下测试演示中，似乎在 filehandle 上调用了析构函数而不是移动（move）。该演示在 stable 版本运行正常，但在 nightly 版本出现以下问题：fil...</li><li><a href="https://github.com/modularml/mojo/pull/2323?">[stdlib] 由 gabrieldemarmiesse 实现 `List.__str__()` · Pull Request #2323 · modularml/mojo</a>: 可作为 #2190 (comment) 参考的 PR。注意它引起了一个似乎在解析器端的 bug。我们得到：RUN: at line 13: mojo /projects/open_source/mojo/stdlib/test/builtin/test_...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1235708968685273088)** (3 条消息): 

- **Prism CLI 工具获得功能提升**：`prism` 库已更新，增加了 *持久化标志 (persistent flags)* 和 *钩子 (hooks)*、*标志要求 (flag requirements)* 以及 *标志组 (flag groups)*。README 已全面修订，包含代码示例和动画 GIF 以演示新功能。请在 [GitHub](https://github.com/thatstoasty/prism) 上查看更新。

- **Mojo-pytest 现已支持 v24.3**：`mojo-pytest` 插件已更新以适配 Mojo **24.3 版本**，目前有一个待处理的 issue 旨在增强集成以提供更好的调试信息。可以在其 GitHub 仓库的 [Issue #9](https://github.com/guidorice/mojo-pytest/issues/9) 跟踪此增强功能的进度。

- **NuMojo 超越 NumPy 和 Numba**：[NuMojo](https://github.com/MadAlex1997/NuMojo) 项目（原名 Mojo-Arrays）正在积极开发中，现已支持 Mojo 24.3 版本。**NuMojo** 的性能显著优于 NumPy，且比 Numba 更快，重点在于扩展标准库的 tensor 功能。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: 模仿 Cobra 的 Mojo CLI 库。</a>: 模仿 Cobra 的 Mojo CLI 库。通过在 GitHub 上创建账号为 thatstoasty/prism 的开发做出贡献。</li><li><a href="https://github.com/guidorice/mojo-pytest">GitHub - guidorice/mojo-pytest: Mojo 测试运行器，pytest 插件（又名 pytest-mojo）</a>: Mojo 测试运行器，pytest 插件（又名 pytest-mojo）。通过在 GitHub 上创建账号为 guidorice/mojo-pytest 的开发做出贡献。</li><li><a href="https://github.com/guidorice/mojo-pytest/issues/9">为 MojoTestItem 添加文件名、行号和列号 · Issue #9 · guidorice/mojo-pytest</a>: 当 pytest 收集 Python 测试时，它会报告行号和上下文，如下所示：def test_ex(): > raise Exception("here") E Exception: here path/to/test_file.py:2: Exception In ...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1235553064061112330)** (3 条消息):

- **PyCon Lithuania 关于 MAX 的演讲**：来自 PyCon Lithuania 的一段新 [YouTube 视频](https://youtu.be/Xzv2K7WNVD0) 讨论了 MAX，但目前标题和描述尚未定义。
- **使用 Mojo 构建应用的教程**：一个名为 "Let's mojo build -D your own -D version=1 app" 的新 GitHub 教程已上线，教授如何使用 Mojo 语言创建或集成工作流。教程可以在 [这里](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md) 找到。
- **Mojo 教程的语法高亮技巧**：有人建议在 Markdown 文件中使用 'mojo' 而不是 'python' 的三引号，以便在记录 Mojo 代码时获得正确的语法高亮。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/Xzv2K7WNVD0)"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md">mojo-learning/tutorials/use-parameters-to-create-or-integrate-workflow.md at main · rd4com/mojo-learning</a>：📖 学习一些 Mojo！通过在 GitHub 上创建账户为 rd4com/mojo-learning 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/)** (1 条消息): 

soracc: 好主意
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 32 期
https://www.modular.com/newsletters/modverse-weekly-32
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1235670853040079019)** (10 条消息🔥): 

- **Mojo 编译器语言变更警报**：24.3 更新日志包含了关于 `__source_location()` 和 `__call_location()` 的新信息，详见 [Modular Docs Changlog](https://docs.modular.com/mojo/changelog#language-changes)。这些功能似乎需要 `@always-inline` 函数才能实现完整功能。
- **Nightly 版 Mojo 编译器发布**：宣布了新的 Nightly 版 Mojo 编译器发布，可以使用 `modular update nightly/mojo` 进行更新。[查看变更内容](https://github.com/modularml/mojo/pull/2480/files) 并回顾自[上一个稳定版本](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)以来的变化。
- **Docstrings 长度讨论**：有一场关于 Docstrings 是否可以超过 80 列的讨论，并建议考虑放宽这一要求，特别是对于标准库。
- **Nightly 版本发布频率提升**：Mojo 编译器的 Nightly 版本发布将更加频繁，预计很快将实现每日更新，目前正等待内部基础设施的改进。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2480/files">[stdlib] 根据 2024-05-03 nightly/mojo 更新标准库，由 JoeLoser 提交 · Pull Request #2480 · modularml/mojo</a>：此 PR 使用与今天的 Nightly 版本（mojo 2024.5.303）相对应的内部提交更新了标准库。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1235548331778900009)** (224 条消息🔥🔥): 

- **转瞬即逝的 AI 就业市场**：成员们辩论了 AI 领域薪酬最高的职位，认为最抢手的岗位在不断演变。有人开玩笑说，最赚钱的 AI 职业可能是成为 CEO 或牙医。

- **预测未来 GPT 版本的价格**：关于假设的 GPT-5 是否会有独立定价层级的讨论浮出水面，观点在 OpenAI 是否会引入区域定价或维持统一价格模型上产生分歧。

- **模仿 UI 引起关注**：有评论指出新的 HuggingChat UI 与现有的 AI 聊天服务非常相似，一些人暗示这对于提供面向消费者的产品和培养开源 AI 社区来说可能是一个游戏规则改变者。

- **AI 的存在主义辩论**：就 AI 增长的本质、人类的独特性、生成能力以及幻觉与现实的融合展开了深入讨论。人们对 AI 的过度自信及其误导信息的能力表示担忧。

- **AI 研究的透明度与开源误解**：一系列消息澄清了虽然 OpenAI 的研究论文是公开的，但期望该机构发布完全训练好的模型是不现实的，因为这些模型具有专有性质，且运行它们需要巨大的计算资源。
  

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1235732447258480760)** (23 messages🔥): 

- **对 GPT-3 和 Codex 的怀旧回顾**：成员们分享了怀旧时刻，回忆起早期使用 **GPT-3 和 Codex** 的时光，表明了对以往模型的持续关注和赞赏。
- **询问语音聊天室**：一位成员询问为何 Discord 中没有 **语音聊天室**，得到的解释是由于审核挑战，目前尚未提供此类功能。
- **对聊天机器人记忆功能集成的困惑**：一位用户询问是否可以将新的 **memory feature** 集成到他们的 API 聊天机器人中，并寻求实施指导。
- **关于 GPT-4 响应时间的查询**：成员们讨论了 **GPT-4** 的速度似乎比其前身 **GPT-3.5** 慢了大约两倍，最近有报告称出现了异常延迟，且 **gpt4 turbo** 比平时慢了 5-10 倍。
- **ChatGPT 访问问题和速率限制**：用户报告了访问 **GPT** 的问题，寻求帮助并质疑消息速率限制。提到了检查 **OpenAI 服务状态** 的建议以及意外超时的经历，这表明可能由于高需求导致了波动的配额系统。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1235533289075376148)** (3 messages): 

- **大语言模型中的检索挑战**：一位成员指出，无法以人们希望的方式缓解大语言模型（LLM）中的检索问题。他们提到了搜索词 "LLM Retrieval Needle In A Hay Stack" 以进行更深入的理解，并强调 **foundation model 的检索限制** 无法通过算法绕过。
- **用于单词出现次数的 Python 工具**：在处理长文本的背景下，提到有一个 **Python** 解决方案能够统计单词的唯一出现次数。这种技术对于数据分析和预处理任务可能非常有用。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1235533289075376148)** (3 messages): 

- **探讨 LLM 检索限制**：一位成员提到了搜索词 "LLM Retrieval Needle In A Hay Stack"，指出任何算法都无法克服 **foundation model 的检索限制**。
- **分享用于单词计数的 Python 脚本**：另一条消息指出，可以使用 Python 解决方案来统计长文本中唯一单词的出现次数。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1235906896414769202)** (2 messages): 

- **社区亮点与创新**：新的用户贡献大放异彩，包括用于批处理的 [Moondream 2](https://huggingface.co/spaces/Csplk/moondream2-batch-processing)、[FluentlyXL v4](https://huggingface.co/spaces/fluently/Fluently-Playground)、HF Audio 课程 [第 0 + 1 章](https://github.com/huggingface/audio-transformers-course/pull/182) 的葡萄牙语翻译、用于长字幕的 [BLIP finetune](https://huggingface.co/spaces/unography/image-captioning-with-longcap)，以及一份葡萄牙语的 [社区亮点列表](https://iatalk.ing/destaques-comunidade-hugging-face/)。

- **BLOOM Chat 支持多语言**：新的 [多语言聊天](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) 支持 55 种语言的对话，同时 [Inpainting 画板](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad) 释放了创意，HF alignment handbook 中的一项任务现在可以 [在云端运行](https://twitter.com/dstackai/status/1785315721578459402)。


- **最新资讯：酷炫的 AI 进展**：AI 爱好者们获得了 [蛋白质优化指南](https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design)、NorskGPT-Mistral-7B 模型、[从零开始](https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model) 实现视觉语言模型的基础知识，以及对 [结合 LLM 的 Google 搜索](https://huggingface.co/blog/nand-tmp/google-search-with-llm) 的见解。爱好者们还可以探索 [LLM 的 Token Merging](https://huggingface.co/blog/samchain/token-merging-fast-inference)，并通过 [这篇博文](https://huggingface.co/blog/maywell/llm-feature-transfer) 扩展关于模型上下文和聊天模型的知识。

- **HF 深入探讨模型可解释性**：关于可解释性的新见解以及对 [LLM 的深入分析](https://huggingface.co/posts/gsarti/644129530281733) 现已发布，供 AI 爱好者学习。

- **AutoTrain 现通过配置向所有人开放**：展示了 AutoTrain 的潜力，用户现在可以使用 [autotrain-advanced GitHub repo](https://github.com/huggingface/autotrain-advanced) 中提供的 YAML 配置文件来训练模型，并鼓励通过创建 pull request 进行贡献。正如在 [Twitter](https://twitter.com/abhi1thakur/status/1786368641388179797) 上宣布的那样，这种易用性使得具备极少 Machine Learning 知识的个人也能在无需编写代码的情况下训练 state-of-the-art 模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>：🤗 AutoTrain Advanced。通过在 GitHub 上创建账户来为 huggingface/autotrain-advanced 的开发做出贡献。</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/)">🤗 Destaques da Comunidade</a>：Destaques da Comunidade 是 Hugging Face Discord 上定期发布的一篇文章，包含一系列由社区制作的项目、模型、Spaces、帖子和文章……
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1235534447856844871)** (163 messages🔥🔥): 

- **语音合成模型讨论**：成员们交流了对语音合成模型的建议，如 **Xtts v2** 和 **Voice Craft**，提到了它们的性能以及语音编辑等独特功能。分享了 [Xtts](https://huggingface.co/spaces/coqui/xtts) 和 [Voice Craft](https://replicate.com/cjwbw/voicecraft) 的演示链接，一位成员指出了 **Voice Craft** 在 zero-shot text-to-speech 方面的能力。
- **模型转换与微调挑战**：讨论了将 Transformer 模型转换为更小格式的挑战，提到了模型大于 2GB 并导致错误的具体问题。还讨论了针对较小数据集进行微调的策略，考虑了 **RAG (Retrieval-Augmented Generation)** 作为数据有限时微调替代方案的有效性。
- **使用 LLM 模型与托管**：提出了关于在生产环境中部署 Large Language Models (LLMs) 的问题，建议将 **Vllm** 和 **TGI** 作为在生产环境中运行 LLMs 的潜在框架。讨论了 **Llama3** 的可用性和用法，并建议尝试 **Groq** 等服务以获取免费 API 访问。
- **Bot 与 Parquet 转换器需求**：用户表达了对用于数据集转换的 parquet 转换器 Bot 的需求，并询问了“Dev mode”的状态，暗示可能存在维护或停机。
- **Prompt 优化与评估咨询**：一位用户询问了评估优化后 Prompt 质量的指标，寻找专门针对 Prompt 评估的特定指标，但随后的讨论并未提供具体的解决方案或指标。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/coqui/xtts">XTTS - a Hugging Face Space by coqui</a>：未找到描述</li><li><a href="https://www.llama2.ai/">Chat with Meta Llama 3 on Replicate</a>：Llama 3 是来自 Meta 的最新语言模型。</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>：未找到描述</li><li><a href="https://tacosconference.github.io/">TaCoS</a>：萨尔布吕肯的 TaCoS 会议</li><li><a href="https://huggingface.co/DioulaD/falcon-7b-instruct-qlora-ge-dq-v2">DioulaD/falcon-7b-instruct-qlora-ge-dq-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-Gradient-1048k">crusoeai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://replicate.com/cjwbw/voicecraft">cjwbw/voicecraft – Run with an API on Replicate</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1235584374041219092)** (9 messages🔥):

- **查询优化探索**：一位成员正在寻求帮助，希望通过整合初始查询 (q1) 的所有细节，来重新表述医药领域的后续问题 (q2)。
- **Ray 部署咨询仍未解决**：一位用户询问了关于在 Ray 上部署 HuggingFace 模型的问题，这引起了社区成员的共同关注。
- **训练循环自定义之争**：有评论主张编写自定义训练循环，认为修改 diffusers 的示例可以为训练 AI 模型提供更大的灵活性。
- **神经网络的捷径**：发生了一场关于 [Kolmogorov-Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756v1) 的有趣讨论，强调了它们极具前景的属性，例如与多层感知器 (MLPs) 相比，所需的计算图更小。
- **微调详解**：
    - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s)，对 AI 模型的微调进行了高层级的概述。
    - 他们还链接了一个关于使用 Transformers 和 Keras 进行微调的 HuggingFace [技术指南](https://huggingface.co/docs/transformers/training)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19756v1">KAN: Kolmogorov-Arnold Networks</a>：受 Kolmogorov-Arnold 表示定理启发，我们提出 Kolmogorov-Arnold Networks (KANs) 作为多层感知器 (MLPs) 的有力替代方案。虽然 MLPs 具有固定的激活函数...</li><li><a href="https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s">What is Fine Tuning? In Two Minutes.</a>：两分钟了解什么是微调。对生成式 AI 模型微调的高层级概述。简而言之：微调生成式 AI 模型就像调吉他。来自 @Hug... 的技术概述。</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tune a pretrained model</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1235577209272668220)** (7 条消息): 

- **新 MPI 代码仓库发布**：分享了一个名为 MPI-Codes 的 GitHub 仓库链接，由 Binary-Beast03 创建，旨在为 MPI 代码的开发做出贡献。可以通过 [MPI-Codes GitHub 仓库](https://github.com/Binary-Beast03/MPI-Codes)获取相关信息。

- **RAG 提升 LangChain 邮件处理能力**：LangChain 的 LangGraph Agents 已通过检索增强生成 (RAG) 得到增强，以改进智能邮件草拟功能，详情见 [Medium 文章](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)。不过，后续评论指出该内容仅限会员访问。

- **针对金融情感分析微调的 FinBERT**：ProsusAI 的 FinBERT 是一种基于 BERT 的 NLP 模型，专门针对金融领域的情感分析进行训练，并分享了 [HuggingFace 链接](https://huggingface.co/ProsusAI/finbert)。它在 Financial PhraseBank 上进行了微调，详见[学术论文](https://arxiv.org/abs/1908.10063)和配套的[博客文章](https://medium.com/prosus-ai-tech-blog/finbert-financial-sentiment-analysis-with-bert-b277a3607101)。

- **检索增强生成 (RAG) 详解**：分享了一个内容丰富的 [Databricks 页面](https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag)，涵盖了 RAG 如何解决 LLM 无法适应自定义数据的问题，以及 AI 应用利用此类数据获得有效结果的必要性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ProsusAI/finbert">ProsusAI/finbert · Hugging Face</a>：未找到描述</li><li><a href="https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag">Che cos&#x27;è la Retrieval Augmented Generation (RAG)? | Databricks</a>：RAG (Retrieval Augmented Generation) 是一种架构方法，它将数据作为大型语言模型 (LLM) 的上下文，以提高相关性...</li><li><a href="https://github.com/Binary-Beast03/MPI-Codes">GitHub - Binary-Beast03/MPI-Codes</a>：通过在 GitHub 上创建账户，为 Binary-Beast03/MPI-Codes 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235594167606837348)** (9 条消息🔥):

- **Model Card 拼写错误提示**：有人指出 **Fluently XL V4** 的 Model Card 标题中有一个小拼写错误——应该是 "Fluently" 而不是 "Fluenlty"。
- **Fluently-XL-v4 展示新的色彩和数字**：一张由 **Fluently-XL-v4** 在本地 NVIDIA RTX 3070 mobile 上生成的图像展示了令人印象深刻的结果，详见这篇 [Instagram 帖子](https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==)，色彩处理出色且手指数量正确，表现优于其他几个模型。
- **Hugging Face 音频课程获得巴西葡萄牙语翻译**：Hugging Face 音频课程的第 0 章和第 1 章已翻译成葡萄牙语，PR 已开启并等待评审，链接见 [此处](https://github.com/huggingface/audio-transformers-course/pull/182)，同时呼吁巴西社区成员协助校对。
- **介绍用于图像字幕（Image Captioning）的 LongCap**：分享了一个用于长字幕生成的 [BLIP model](https://huggingface.co/unography/blip-long-cap) 微调版本，该模型能够生成详细的图像描述，适用于文本到图像生成的提示词。目前正在请求协助将其与 Google 的 DOCCI 进行评估，并提供了一个 [Colab notebook](https://colab.research.google.com/drive/1UfS-oa6Ou0mguZG0udXzyE8IsL_n8ZNf?usp=sharing) 用于测试。
- **存档葡萄牙语版的社区亮点**：一位社区成员创建的新页面汇集了自第 52 期以来 Hugging Face Community Highlights 的所有帖子和链接，并计划补全之前的期数，建立一个全面的葡萄牙语 AI 相关内容数据库，[链接见此](https://iatalk.ing/destaques-comunidade-hugging-face/)。
- **用于 LLM 的合成数据生成器现已发布至 PyPI**：一个用于为训练大语言模型生成和归一化合成数据的工具已发布，可在 [PyPI](https://github.com/tobiadefami/fuxion) 上获取，这可能有助于不同项目用例的 fine-tuning 工作。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - fishaudio 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE">kimou605/shadow-clown-BioMistral-7B-DARE · Hugging Face</a>：未找到描述</li><li><a href="https://www.instagram.com/p/C6eMZaTr03q/?igsh=MWQ1ZGUxMzBkMA==">Instagram 上的 Mansion X："Speaks American 🇺🇸 *fluently*. #fit #ootd"</a>：2 个赞，0 条评论 - the_mansion_x 于 2024 年 5 月 2 日发布："Speaks American 🇺🇸 *fluently*. #fit #ootd"。</li><li><a href="https://huggingface.co/unography/blip-long-cap">unography/blip-long-cap · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/unography/image-captioning-with-longcap">使用 LongCap 进行图像字幕生成 - unography 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1UfS-oa6Ou0mguZG0udXzyE8IsL_n8ZNf?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/tobiadefami/fuxion">GitHub - Tobiadefami/fuxion: 合成数据生成与归一化函数</a>：合成数据生成与归一化函数 - Tobiadefami/fuxion</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/">🤗 Destaques da Comunidade</a>：Destaques da Comunidade 是一个定期在 Hugging Face Discord 上发布的帖子列表，包含一系列由社区制作的项目、模型、Spaces、帖子和文章……
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1235723560199131146)** (6 条消息):

- **LLM 改进精选集**：一位成员在 HuggingFace 上分享了关于改进大语言模型 (LLMs) 的研究[精选列表](https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd)，并邀请大家提供想法和反馈。
- **聚焦 React Agent**：另一位参与者强调了 React Agent 在提升 LLM 输出质量方面的重要性，并指出该领域论文众多，选择研究重点具有挑战性。
- **LLM 研究中未被索引的发现**：LLM 改进精选集的策展人表达了分享那些在研究汇编中未被索引、未获得点赞或没有相关代码的论文的兴奋之情。
- **探索 LLM 中的推理与行动**：策展人关注了一篇名为 'ReAct' 的论文，该论文提出了一种在 LLM 中结合推理轨迹 (reasoning traces) 和特定任务行动的方法，以增强性能和可解释性。摘要讨论了交替进行这两个方面如何改善与外部信息源的交互并处理异常情况（[查看论文](https://arxiv.org/abs/2210.03629)）。
- **图机器学习 (Graph ML) 遇见 LLM**：一位成员分享了一份演示文稿的初步笔记，探讨了图机器学习与 LLM 的交集，并观察到该主题的探索程度比最初想象的要广泛。他们分享了一篇总结该主题的 [Medium 文章](https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/f0ster/smarter-llms-research-6633156999b1fa10612309dd">Smarter LLMs Research - f0ster 精选集</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>：虽然大语言模型 (LLMs) 在语言理解和交互式决策任务中展示了令人印象深刻的能力，但它们的推理能力（例如思维链...）
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1235516322641870909)** (7 条消息): 

- **频道咨询**：一位成员询问是否存在 **#cv-study-group** 频道，尽管在 [社区计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) 页面中提到了该频道，但他们未能找到。
- **微调策略分享**：有建议提出为了提高效率，仅微调预训练模型的分类器权重，并考虑使用浅层 CNN 在连接到 **Yolov4** 之前对图像进行缩放，从而端到端地训练模型。
- **学习小组状态说明**：成员们对 **学习小组 (study group)** 的存在似乎有些困惑；一位成员澄清说没有特定的学习小组，而另一位成员提到虽然没有阅读小组，但可能有人知道过去的学习小组。
- **确认学习小组不存在**：成员们一致认为该频道目前没有活跃的特定 **阅读或学习小组**。

**提到的链接**：<a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">欢迎来到社区计算机视觉课程 - Hugging Face Community Computer Vision Course</a>：未找到描述

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235581472619106315)** (5 条消息): 

- **RARR - 模型归因解决方案**：一位成员分享了 [RARR 论文](https://huggingface.co/papers/2210.08726)，该论文介绍了一个名为 **通过研究和修订进行追溯归因 (Retrofit Attribution using Research and Revision)** 的系统。RARR 旨在自动为文本生成模型的输出寻找并添加归因，并对不支持的内容进行修正。

- **零样本分类 (Zero-shot Classification) 的困惑**：一位用户报告了 **零样本分类模型** 生成结果比例失调的问题，标签 "gun" 和 "art" 导致概率几乎平分，质疑模型行为与预期结果不符。这可能表明对分类器如何分析与所提供标签无关的文本存在误解。

**提到的链接**：<a href="https://huggingface.co/papers/2210.08726">论文页面 - RARR: Researching and Revising What Language Models Say, Using Language Models</a>：未找到描述

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1235938948904128554)** (12 条消息🔥): 

- **关于 Auto-Train 配置的说明**：一位成员澄清说，在 Auto-Train 配置中指定 `xl: true` 是可选的，因为模型类型可以自动确定，但也可以在配置中显式声明。

- **合并 Diffusion Pipelines 技术**：一位成员询问了是否可以使用两个不同的 **StableDiffusionPipelines** 进行部分去噪（partial denoising），并在处理过程的中点进行切换。另一位成员提供了关于一种名为 *partial diffusion via mixture of experts* 方法的信息，并链接到了 [diffusers GitHub 仓库](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2) 中针对 **SD 1.5** 的一个待处理 Pull Request。

- **寻求 Partial Diffusion 示例**：一位成员请求使用 **StableDiffusionPipelines** 进行 Partial Diffusion 的示例。他们被引导至一个展示该方法实现的 GitHub 对比页面。

- **Partial Diffusion 的测试可用性**：同一位成员考虑测试 Pull Request 中提到的 Partial Diffusion 方法，以确定其是否适合自己的测试套件，并指出更倾向于更快的推理时间。

**提到的链接**：<a href="https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2">Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers</a>：🤗 Diffusers：在 PyTorch 和 FLAX 中用于图像和音频生成的先进 Diffusion 模型。- Comparing huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1235600910445445233)** (5 messages): 

- **构建优化的 RAG 数据栈**：分享了一个新的教程，包含构建高效 Agentic RAG 支持机器人 **数据栈（data stack）** 的全面指南。它强调了除向量数据库外各种数据组件的重要性，由 [@tchutch94](https://twitter.com/tchutch94) 和 [@seldo](https://twitter.com/seldo) 编写。点击[此处](https://t.co/jez7g9hADV)查看全文。
  
- **逐步 RAG Pipeline 指南**：Plaban Nayak 介绍了一个开源的 RAG Pipeline 指南，使用了来自 Meta 的 **Llama 3**、@qdrant_engine 和 **ms-marco-MiniLM-L-2-v2**。该指南强调通过 reranker 过程提高 RAG 应用的精度。阅读更多关于该指南的内容请点击[此处](https://t.co/wXxFCsrkSa)。

- **Airbnb 房源的自然语言过滤器**：Harshad Suryawanshi 提供了创建一个 RAG 应用程序的演练，该程序利用 @MistralAI 的 Mixtral 8x7b 工具通过自然语言过滤 **@Airbnb 房源**。详细说明和仓库可以在[此处](https://t.co/iw6iBzGKl6)找到。

- **LlamaIndex 10.34 版本发布，包含 Introspective Agents**：宣布了新的 **LlamaIndex 10.34 版本**，重点介绍了 Introspective Agents 等功能，这些 Agent 利用 reflection（反思）进行迭代响应。Notebook 包含具体实现，但警告可能包含敏感内容。阅读关于这些 Agent 和警告的内容请点击[此处](https://t.co/X8tJGXkcPM)。

- **发布支持 Huggingface 的 LlamaIndex 0.10.34**：**LlamaIndex 0.10.34** 的发布引入了 Introspective Agents，并提到了即将支持 Huggingface 集成。他们承诺在接下来的几天里分别讨论所有新更新。点击[此处](https://t.co/UrD0c7BRAO)获取详情。

**提到的链接**：<a href="https://t.co/X8tJGXkcPM">Introspective Agents: Performing Tasks With Reflection - LlamaIndex</a>：未找到描述

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1235517710474149968)** (140 messages🔥🔥):

- **寻求财务分析应用**：一位成员正在创建一个应用程序，用于从包含公司损益表的 pandas dataframes 中生成财务摘要。鉴于文档中的示例较少，他们正在寻求关于使用 [Pandas Query Engine](https://docs.llamaindex.ai/en/latest/) 的指导。
- **使用 LlamaIndex 定制 MongoDB**：一位用户寻求帮助，希望直接从带有 metadata 的 MongoDB embeddings 中进行查询，绕过向 LlamaIndex 的 query engine 提交文档或节点的过程。他们分享了之前使用过的教程 [链接](https://colab.research.google.com/drive/136MSwepvFgEceAs9GN9RzXGGSwOk5pmr?usp=sharing)，并请求 MongoDB `collections.aggregate` 的替代方案。
- **LLamacpp 并行请求死锁**：一位成员报告了在使用 llamacpp gguf 模型运行两个并发查询时出现死锁。他们询问如何在不使用服务器设置的情况下启用并行请求服务，以缓解此问题。
- **关于在 Llama Index 中设置 Trulens 的咨询**：一位用户请求关于在 MongoDB 和 Llama Index 中使用 Trulens 的帮助，并指出他们已经上传了 embeddings 和 metadata。他们分享了来自 [文档](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=true#truera-trulens) 的相关链接，并建议考虑 Arize 和 Langfuse 等替代工具。
- **LlamaIndex 的内存负载问题**：一位成员在运行 LlamaIndex 时遇到了内存过载问题，一个 8GB 的模型有时会超过 20GB，然后回退到 CPU，导致运行缓慢。他们在代码中定位到了一个似乎会大量占用内存的特定命令执行，并提到有必要等待内存清理。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/136MSwepvFgEceAs9GN9RzXGGSwOk5pmr?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://llama.meta.com/docs/how-to-guides/prompting">Prompting | 操作指南</a>：Prompt engineering 是一种用于自然语言处理 (NLP) 的技术，通过为语言模型提供更多关于当前任务的上下文和信息来提高其性能。</li><li><a href="https://www.llamaindex.ai/contact">联系我们 — LlamaIndex，LLM 应用的数据框架</a>：如果您对 LlamaIndex 有任何疑问，请联系我们，我们将尽快安排通话。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=true#truera-trulens">可观测性 (旧版) - LlamaIndex</a>：未找到描述</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | 模型卡片与 Prompt 格式</a>：Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含单个 system 消息，可以包含多个交替的 user 和 assistant 消息，并始终以最后一个 user 消息结束...</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>).">未找到标题</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/fine-tuning/fine-tuning#finetuning-embeddings>).">微调 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/postgres#llama_index.vector_stores.postgres.PGVectorStore>).">Postgres - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/supabase#llama_index.vector_stores.supabase.SupabaseVectorStore>).">Supabase - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/13196">通过 co-antwan 使用 `documents` 参数调用 Cohere RAG 推理 · Pull Request #13196 · run-llama/llama_index</a>：描述：在 RAG 管道中增加了对 Cohere.chat 的 documents 参数的支持。这确保了 Cohere 客户端的正确格式化，并带来更好的下游性能。</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1235711461784027206)** (1 条消息): 

- **OpenAI API 令人印象深刻但昂贵的 RAG 性能**：一位用户分享了他们使用 **OpenAI assistants API v2** 进行 *检索增强生成 (RAG)* 的积极体验，指出从包含 500 篇维基百科文章的测试知识库中得出的答案非常有效。然而，他们强调了成本问题，因为一段简短的对话就产生了 **1.50 美元的费用**。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1235542394129088552)** (25 条消息🔥):

- **寻找兼容 Gemini 1.5 的工具**：一位成员询问是否有类似 Cursor/Aider 的工具支持 **Gemini 1.5** 的全上下文窗口，但提到对 Gemini 1.5 的基准测试（benchmarking）感到失望，更倾向于使用 Opus 或 Cursor 目前支持的长上下文（long context）。
- **Code Interpreter SDK 亮相 Twitter**：成员 @mlejva 在 Twitter 上宣布发布他们的 Code Interpreter SDK，并附上[推文链接](https://twitter.com/mlejva/status/1786054033721139319)寻求社区支持。
- **关于 Prompt 标签实践的疑问**：一位用户向社区咨询关于在 Prompt 中标记输出变量的最佳实践，特别是针对 **Claude**，并引用了 [Matt Shumer 的推文](https://twitter.com/mattshumer_/status/1773385952699789808)。
- **OpenAI Assistants API 快速入门分享**：重点介绍了 OpenAI Assistants API Quickstart，其特点是与 Next.js 集成，并提供流式聊天界面、function calling 和 code interpreter；相关[推文链接](https://x.com/openaidevs/status/1785807183864820209?s=46&t=90xQ8sGy63D2OtiaoGJuww)及 [GitHub 仓库](https://github.com/openai/openai-assistants-quickstart)。
- **SQLite 获得新的向量搜索扩展**：`sqlite-vss` 的继任者 `sqlite-vec` 正在开发中，旨在为 SQLite 提供更好的向量搜索（vector search），并分享了[作者的博客文章](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">我正在编写一个新的向量搜索 SQLite 扩展</a>：sqlite-vec 是一个新的 SQLite 向量搜索扩展，即将推出！</li><li><a href="https://x.com/openaidevs/status/1785807183864820209?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：我们开源了一个新的快速入门项目，帮助你使用 Assistants API 和 @nextjs 进行构建。它包含用于创建具有流式传输功能的聊天界面，以及使用 function calling 等工具的示例代码...</li><li><a href="https://x.com/emilylshepherd/status/1786037498507853852?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Emily (She/Her) (@EmilyLShepherd) 的推文</a>：让我们聊聊 @rabbit_hmi。一个线程 🧵 成立于 2021 年，最初名为 Cyber Manufacture Co，他们是一个“在...交汇处构建下一代体验的创意工作室”。</li><li><a href="https://x.com/emilylshepherd/status/1786037498507853852?">来自 Emily (She/Her) (@EmilyLShepherd) 的推文</a>：让我们聊聊 @rabbit_hmi。一个线程 🧵 成立于 2021 年，最初名为 Cyber Manufacture Co，他们是一个“在...交汇处构建下一代体验的创意工作室”。</li><li><a href="https://x.com/teknium1/status/1786485060314521627?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：至少确认了，不，它不是“Large Action Model”——它是一个 LLM.. lol</li><li><a href="https://www.echoai.com/">对话智能 - Echo AI</a>：客户对话是你拥有的最有价值的数据。Echo AI 是首个 AI 原生对话智能平台，可将客户说的每一句话转化为洞察和行动...</li><li><a href="https://www.assorthealth.com/">Assort Health | 首个为医疗保健构建的生成式 AI 呼叫中心</a>：我们的呼叫中心生成式 AI 可减少等待时间，降低挂断率，并在增加预约收入的同时控制成本。</li><li><a href="https://www.trychroma.com/">AI 原生开源嵌入数据库</a>：AI 原生开源嵌入数据库</li><li><a href="https://www.getlifestory.com/">Life Story</a>：记录生活，一次一个故事。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1235575449674055818)** (33 条消息🔥):

- **Mamba 深度探讨开启**：**llm-paper-club-west** 频道的成员们正准备就 Mamba 展开讨论，并分享了一个用于深入研究该主题的 Notion 链接：[A Mamba Deep Dive](https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f)。
- **辩论 Mamba 的选择性召回能力**：一名成员提出疑问，询问 Mamba 中的“选择性复制（selective copying）”是否类似于对之前出现过的 tokens 的召回测试，从而引发了关于该机制特异性的讨论。
- **技术故障促使平台切换提案**：在 Mamba 讨论期间遇到延迟和技术故障的用户一致同意未来的会议切换到 Zoom，以确保更流畅的体验。
- **探索 Mamba 在 Fine-tuning 中的敏感性**：对话转向了 Mamba 架构在 Fine-tuning 期间的表现，以及与传统的 Transformers 相比其对过拟合（overfitting）的敏感程度。
- **State Space Models 与 Induction Heads**：成员们就 State Space Models（特别是 Mamba 背景下）是否能模拟注意力层中发现的 Induction Heads 进行了简短交流。分享了两篇 arXiv 论文供进一步阅读：[Arxiv State Space Models](https://arxiv.org/pdf/2404.15758) 和 [Arxiv Multi Token Paper](https://arxiv.org/pdf/2404.19737)。

**提到的链接**：<a href="https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一体的新工具。它是为您和您的团队打造的一体化工作空间。

---

**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1236044647785304109)** (65 条消息🔥🔥): 

- **对 Suno 音频生成的着迷**：一名成员对 **suno** 的音乐生成能力表示好奇，想知道它是否从头开始创建音乐轨道。另一名成员提到 **Suno** 对音频 tokenizing 的关注是他们的“秘密武器”。
- **探索不同的模型架构**：围绕 **musicgen** 架构的讨论透露，这是某位好友在 Fine-tuning 音频模型以用于多模态应用中的一次尝试。成员们还提到了 **imagebind** 作为多模态嵌入空间（embedding space）的一个例子。
- **理解谐波失真（Harmonic Distortion）**：在关于“谐波失真”的简短对话中，一名成员将其描述为谐波上的权重错误，这可能导致不正确的频率比或拍频。引用了一篇讨论 **snake activation function** 及其在减少谐波失真方面潜力的博客。
- **使用 Latent Diffusion 和 Autoencoders 生成音频**：对 **stable audio 2.0** 过程的询问引发了关于音频文件如何通过 Autoencoders 处理以创建 tokens 的讨论，并有建议称整个音频文件都被压缩供模型使用。
- **生成音频的商业可行性**：一名成员询问了 **Stable Audio 2.0** 输出内容的许可和商业用途，表明了对生成内容相关法律问题的兴趣。还提到了潜在的应用，如分离和替换音频通道。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notesbylex.com/snake-activation-function">来自 Snake Activation Function 的推文</a>：Snake 是一种神经网络激活函数，适用于对具有“周期性归纳偏置（periodic induction bias）”的问题进行建模——换句话说，就是具有规律、重复模式的问题……</li><li><a href="https://arxiv.org/abs/2404.10301v1">使用 Latent Diffusion 进行长篇音乐生成</a>：基于音频的音乐生成模型最近取得了长足进步，但到目前为止还未能产生具有连贯音乐结构的完整长度音乐轨道。我们展示了通过训练一个 ge...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: 这是 gary。python continuations 以及 ableton 内部的 continuations。这是一个新手正在开发中的项目。</a>：这是 gary。python continuations 以及 ableton 内部的 continuations。这是一个新手正在开发中的项目。 - betweentwomidnights/gary4live
</li>
</ul>

</div>

---

**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1235710627415195688)** (49 条消息🔥): 

- **多语言 LLM 的有趣趋势**：一场正在进行的讨论围绕如何增强 Large Language Models (LLM) 的多语言能力展开，引用了诸如 ["Understanding Language Models by Fine-grained Language Identification"](https://arxiv.org/abs/2402.10588) 等研究论文，以及探索 LLM 处理多语言输入的著作。所描述的框架表明，在初始层中，LLM 在以原始查询语言生成响应之前，会将多语言输入转换为英语。

- **回顾 ML 领域的历程**：用户们缅怀了那些被遗弃或被掩盖的 Machine Learning 研究领域，包括对抗鲁棒性 (Adversarial Robustness)、自动架构 (Automated Architecture) 以及特定领域模型训练。人们感到一种怀旧和对未竟之路的遗憾，而各大科技公司对人才的吸引导致这些领域被边缘化，进一步加剧了这种情绪。

- **AI 融资与影响力的格局变化**：用户讨论了对 AI 公司巨大的投资以及潜在的过度饱和可能导致的投资回报递减。人们对这如何影响模型扩展 (Scaling up) 的效率以及 AI 研究的未来方向表示担忧。

- **LLM 与系统层级漏洞**：有一场关于模型如何处理指令层级 (Instruction Hierarchies) 的深入讨论，以及当系统提示词 (System Prompt) 未被视为与用户提示词 (User Prompt) 不同时所产生的漏洞——这在防止提示词注入 (Prompt Injections) 或其他针对 LLM 的攻击中尤为重要。一篇相关的论文 [“Improving Robustness to Prompt Injections with Instruction Hierarchy”](https://arxiv.org/abs/2404.13208) 建议，强化指令层级可以缓解这些漏洞。

- **ML 工程师求职**：一位成员向社区寻求美国以外的 Machine Learning 工程师职位机会。此人在 LLM 方面拥有经验，相关工作包括领导 Polyglot 团队以及在 EleutherAI 内部为 OSLO 项目做出贡献；他们分享了 LinkedIn、Google Scholar、GitHub 的个人链接以及联系邮箱。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.18815v1">How do Large Language Models Handle Multilingualism?</a>：Large Language Models (LLM) 在多种语言中都表现出卓越的性能。在这项工作中，我们深入探讨了一个问题：LLM 是如何处理多语言的？我们引入了一个框架...</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>：当今的 LLM 容易受到提示词注入、越狱和其他攻击的影响，这些攻击允许攻击者用自己的恶意提示覆盖模型的原始指令。在这项工作中...</li><li><a href="https://github.com/jason9693">jason9693 - Overview</a>：AI 研究工程师。jason9693 拥有 71 个代码仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1235487352873418793)** (54 messages🔥): 

- **深入探讨数据集污染**：在关于指令微调 (Instruction-finetuning) 和基准测试有效性的讨论中，参与者对 Large Language Models (LLM) 中的基准数据集泄露表示担忧，强调了衡量泄露信息的难度以及检测和解决泄露的循环。重点介绍了两篇关于基准数据集泄露的近期论文：一篇侧重于[检测数据泄露](http://arxiv.org/abs/2404.18824)，另一篇讨论了将[新鲜的基准问题](https://arxiv.org/abs/2405.00332)作为防止不公平比较的解决方案。
  
- **聊天机器人对话作为学习工具**：考虑了使用聊天机器人对话（特别是与同一用户的多次多轮交互）的想法，旨在利用情感分析或用户留存 (Churn) 来改进 LLM。参与者对这些如何导致模型内部的自我改进循环感到好奇，一位成员指出了一篇[关注间接偏好的论文](https://arxiv.org/abs/2404.15269)，另一位提到了用于聊天机器人研究的 [WildChat 数据集](http://arxiv.org/abs/2405.01470)。
  
- **无需启发式的国际象棋精通**：提到了一项使用在 1000 万场国际象棋比赛数据集上训练的 Transformer 模型的研究，展示了该模型在没有特定领域增强或显式搜索算法的情况下在国际象棋中的高性能。这篇 [DeepMind 论文](https://arxiv.org/abs/2402.04494)表明，大规模训练模型可以达到极具竞争力的水平，而无需传统象棋引擎使用的方法。

- **偶然的时间旅行与未来归来**：发生了一场幽默的交流，一名参与者开玩笑地声称制造了时间机器并从未来归来，引发了关于他们在 'ot' (off-topic) 频道出现以及使用时间机器的俏皮互动。

- **Gwern 的外围回归**：在元讨论中，参与者注意到 gwern1782 在离开服务器一段时间后，开始有选择地回复过去的提及，并提到使用 Discord 的搜索功能从大量通知中进行筛选。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.04494#deepmind">Grandmaster-Level Chess Without Search</a>: 机器学习最近的突破性成功主要归功于规模：即大规模的基于 Attention 的架构和前所未有规模的数据集。本文调查了...</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: 随着预训练数据使用的扩大，基准数据集泄露现象变得日益突出，而不透明的训练过程以及通常未公开的包含情况加剧了这一现象...</li><li><a href="http://arxiv.org/abs/2405.01470">WildChat: 1M ChatGPT Interaction Logs in the Wild</a>: 诸如 GPT-4 和 ChatGPT 之类的聊天机器人现在正为数百万用户提供服务。尽管它们被广泛使用，但仍然缺乏展示这些工具如何被用户群体使用的公共数据集...</li><li><a href="https://arxiv.org/abs/2404.15269">Aligning LLM Agents by Learning Latent Preference from User Edits</a>: 我们研究了基于用户对 Agent 输出进行编辑的语言 Agent 交互式学习。在诸如写作助手之类的典型场景中，用户与语言 Agent 交互以生成...</li><li><a href="https://arxiv.org/abs/2405.00332">A Careful Examination of Large Language Model Performance on Grade School Arithmetic</a>: 大型语言模型（LLMs）在许多数学推理基准测试中取得了令人印象深刻的成功。然而，人们越来越担心其中一些表现实际上反映了数据集...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1235971685035933758)** (1 messages): 

- **关于数学问题解决的乐观预测**: NARRATOR 提到一个已被超越的预测，目前数学文字题解决（Math Word Problem Solving）的性能在 **2 年内达到 70% 以上**，事实证明之前的预测过于悲观。该基准测试可以在 [Papers With Code](https://paperswithcode.com/sota/math-word-problem-solving-on-math) 详细探索，这是一个提供根据 [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) 许可的数据的免费资源。可以通过 [hello@paperswithcode.com](mailto:hello@paperswithcode.com) 联系他们。

**Link mentioned**: <a href="https://paperswithcode.com/sota/math-word-problem-solving-on-math">Papers with Code - MATH Benchmark (Math Word Problem Solving)</a>: 目前 MATH 上的 SOTA 是 GPT-4-code 模型（CSV，含代码，SC，k=16）。查看 109 篇附带代码的论文的完整对比。

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235593243853590538)** (8 messages🔥): 

- **立场论文被接收**: 由 **Vincent Conitzer**、**Rachel Freedman** 和 **Stuart Russell** 等作者最近提交的一篇立场论文已被[接收为立场论文](https://arxiv.org/abs/2404.10271)。
- **ICML 2024 机械可解释性研讨会**: **Neel Nanda** 宣布在 [ICML 2024](https://icml2024mi.pages.dev/) 举办首届学术性*机械可解释性（Mechanistic Interpretability）研讨会*，并征集论文。该活动设有 1750 美元的最佳论文奖金，投稿形式多样，截止日期为 5 月 29 日。
- **明星阵容小组讨论揭晓**: **Neel Nanda** 确认 **Naomi** 和 **StellaAthena** 将参加机械可解释性研讨会的小组讨论，活动网站的更新和补充内容正在处理中。
- **基于 Transformer 的语言模型综合入门指南**: **javifer96** 强调了一份关于基于 Transformer 的语言模型入门指南的发布，该指南以统一的符号涵盖了模型组件和可解释性方法。感兴趣的人员可以在[此处](https://twitter.com/javifer_96/status/1786317169979970046)找到公告和更多信息。
- **以英语作为枢纽语言的跨模型泛化**: **Butanium** 分享了团队在 **llama** 模型中复制的结果，表明使用英语作为枢纽语言在这些模型中具有良好的泛化性。更多细节可以在他们最新的 [推文](https://twitter.com/Butanium_/status/1786394217478004950) 中找到。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://icml2024mi.pages.dev/">ICML 2024 Mechanistic Interpretability Workshop</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.10271">Social Choice for AI Alignment: Dealing with Diverse Human Feedback</a>: 诸如 GPT-4 之类的基础模型经过微调以避免不安全或其他有问题的行为，例如，它们拒绝协助实施犯罪或生产...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1235644356464082984)** (2 messages):

- **关于纳入 MT-Bench 的咨询**：一位成员询问了将 **MT-Bench** 或类似基准测试纳入 lm-evaluation-harness 的进展情况，以及是否有即将推出的对话式 AI 质量基准。

- **Prometheus 2 作为潜在改进**：另一位成员强调了 **Prometheus 2**（一个开源评估器 LM），建议将其作为 lm-evaluation-harness 的有益补充。正如 [Hugging Face 上的研究摘要](https://huggingface.co/papers/2405.01535)所指出的，Prometheus 2 旨在模拟人类和 GPT-4 的判断，并支持多种形式的评估。

**提到的链接**：<a href="https://huggingface.co/papers/2405.01535">论文页面 - Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1235586380994711673)** (40 条消息🔥): 

- **LLama-3 8B 上下文长度的突破**：Gradient AI 宣布在 [Crusoe Energy 的算力](https://huggingface.co/crusoeai)支持下，将 LLama-3 8B 的上下文长度从 8k 扩展到了 1040k 以上。这一成就表明，通过调整 RoPE theta 并进行极少量训练，最先进的大语言模型（LLMs）即可处理长上下文。
  
- **Ring Attention 的概念化**：一位成员讨论了尝试理解 Ring Attention 概念的过程，并利用可视化手段辅助理解，尽管其他成员对其方法的工程准确性持保留意见。

- **ChatML 训练中的冲突**：一位用户报告了在使用 ChatML 进行训练时遇到的问题，出现了与 `SeparatorStyle.GEMMA` 相关的 `AttributeError`。排查建议包括删除冲突参数和升级 fastchat，旨在解决 ChatML 配置数据集的训练问题。

- **在 Prompt 中注入上下文**：在关于微调 Prompt 设计的讨论中，成员们交流了如何在模型训练的系统 Prompt 中包含 ChatML 轮次，并推断出当上下文注入 Prompt 时，ChatML token 会被正确地进行 Tokenization 而无需转义。

- **通过 llama.cpp 加速的 Hermes 2**：一位成员对 Hermes 2 Pro Llama 3 8B 在 8GB RAM 的 Android 设备上的推理速度表示赞赏。这归功于 llama.cpp 的升级，据报告推理速度提升了 30%。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/granawkins/status/1786428318478168447">来自 Grant♟️ (@granawkins) 的推文</a>：2024 年的 SOTA RAG
</li>
</ul>

</div>

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1235492757917929553)** (19 条消息🔥): 

- **为提升性能而合并的 PR**：合并了一个 Pull Request，修复了 orpo 训练器在预处理时仅使用一个 worker 的问题。该改进旨在加速 TRL 训练器以及可能的其他训练器（如 DPOTrainerArgs）的数据预处理步骤。补丁详见 [GitHub PR #1583](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583)。

- **参数化影响多个配置**：对话中的澄清表明，讨论中的 `dataset_num_proc` 参数不仅影响 TRL 训练器，还影响代码库中的 DPO、SFT, CPO, KTO 和 ORPO 配置。

- **确定 Axolotl 的最低 Python 版本**：确认运行 Axolotl 所需的最低 Python 版本为 3.10，这允许在代码库中使用 `match..case` 语句。

- **Gradio 可配置性咨询**：一位成员讨论了如何通过 YAML 文件使硬编码的 Gradio 选项变为可配置。他们探索了如何传递各种配置选项，例如将 Gradio 界面设为私有，以及控制 IP 地址和端口号。

- **调查 Gradio Tokenization 问题**：有报告称 Gradio 在使用 llama3 模型时未采用正确的 token，这引发了关于默认 token 是否可能覆盖已加载 Tokenizer 的讨论。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1583">修复：TRL 训练器预处理步骤曾以单进程运行，由 ali-mosavian 提交 · Pull Request #1583 · OpenAccess-AI-Collective/axolotl</a>：描述：我们之前未将 dataset_num_proc 传递给 TRL 训练配置，导致 TRL 训练器中的初始数据预处理步骤仅在单个进程中运行。动机与背景：加速...

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1235581892145975317)** (7 条消息):

- **Epochs 与 Batch Sizes 咨询**：一位成员提到他们在训练模型时通常选择 **4 epochs** 和 **4 的 batch size**。
- **Llama3 模型推理困惑**：一位成员询问在使用 fft 脚本训练 **llama3** 后如何调用推理（inference），并指出常规的 `qlora_model_dir` 似乎不适用。
- **SafeTensor 转 GGUF 的转换挑战**：在讨论从 **SafeTensors** 转换为 **GGUF** 时，一位成员表示在使用 `llama.cpp` 后，很难找到转换为各种 gguf 类型（如 `Q4_K` 或 `Q5_K`）的方法，因为其选项似乎有限。
- **转换困境的脚本解决方案**：另一位成员指向了 `llama.cpp` 仓库中可用的转换脚本，特别引用了 [GitHub 上的 convert-gg.sh 脚本链接](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh)。
- **llama.cpp 中有限的转换选项**：尽管有之前的建议，同一位成员重申了问题，称 `llama.cpp` 转换脚本仅提供两种 gguf 转换选项，而他们正在寻找更广泛的类型，如 `q4k`。

**提到的链接**：<a href="https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh">llama.cpp/scripts/convert-gg.sh at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。在 GitHub 上为 ggerganov/llama.cpp 的开发做出贡献。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1235586698700783736)** (15 messages🔥): 

- **为自定义数据集角色进行配置**：用户询问如何配置结构为 `{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}, {"role": "assistance", "content": "…"}]}` 的数据集。建议他们使用 `UserDefinedDatasetConfig` 类，以使结构符合系统的预期。

- **为 ShareGPT 预处理对话**：针对数据集结构问题，建议通过将 "content" 与相应的角色标识符拼接来预处理消息，确保其符合 `sharegpt` 模型预期的格式。

- **填写数据集配置键**：当被问及如何填写数据集配置块中的某些键时，建议将 `conversation` 设置为 `Llama2ChatConversation`，将 `field_human` 映射到 "user"，将 `field_model` 映射到 "assistance"，并将 "system" 和 "user" 正确分类为输入角色，将 "assistance" 分类为输出角色。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>：一个用于训练、部署和评估大型语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。- lm-sys/FastChat</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=590b42af-2946-480b-80b8-8ae1021929e1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e3e12dac-7c3d-42e8-a7f8-1e0485a19562)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=843813ee-d860-4061-9f19-b32faedaa383)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1235564223971725402)** (32 messages🔥): 

- **DeepSpeed Stage 3 质量疑虑已消除**：DeepSpeed Stage 3（也称为 ZeRO-3）本身不会降低模型质量，而是在训练期间优化内存使用。必须在训练流水线中正确实现和集成 ZeRO-3，以避免由于配置错误导致的潜在问题。[DeepSpeed 文档](https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)可以指导配置。
  
- **结合 Flash Attention 与 DeepSpeed 进行微调**：可以同时使用 Flash Attention 和 DeepSpeed Stage 3 进行模型微调，这需要将 Flash Attention 集成到模型中，并在训练脚本中设置 DeepSpeed Stage 3。正确的配置对于有效利用这两种技术至关重要。

- **DeepSpeed Stage 3 的速度提升**：DeepSpeed Stage 3 可以加速超大规模模型的训练，允许更大的 batch sizes 并减少对复杂并行策略的需求。然而，加速程度可能因模型架构、硬件设置和数据加载效率而异。

- **在 Axolotl 上使用 LLaMA 3 Instruct 进行训练**：使用 LLaMA 3 Instruct 进行训练的说明包括设置环境、创建 YAML 配置文件，以及通过使用 Accelerate 和 Axolotl 的命令启动训练和推理。可能需要针对 LLaMA 3 Instruct 的具体实现进行调整。[Axolotl GitHub](https://github.com/openaccess-ai-collective/axolotl)

- **理解 Axolotl 配置中的 VRAM 使用情况**：在 Axolotl 示例中使用简单的 `qlora.yaml` 配置会均衡地使用两个 GPU，但由于模型兼容性和管理分片模型的开销等多种因素，切换到 FSDP 或 DeepSpeed Stage 3 可能不会显示出显著的 VRAM 减少。配置可能需要进行微调以优化内存节省。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/accelerate/tree/main/docs/source/usage_guides/deepspeed.md#L83L167)),">accelerate/docs/source/usage_guides/deepspeed.md at main · huggingface/accelerate</a>：🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持.....</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/colab-notebooks/colab-axolotl-example.ipynb#L1L2)">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl#egg=axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=48f435d8-7ace-4f56-b4a5-0936a0f2d236)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6ec2ec3e-8632-45bb-9b50-4d25265230c0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fd359a44-f5ac-4e19-b938-f7288b3cfb04)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3c34d3f5-597f-4472-95aa-17cd8c03e44e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fcb1eb4e-b085-4f2a-aeda-82b4c38beb8d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://github.com/huggingface/transformers/tree/main/docs/source/en/deepspeed.md#L167L302)">transformers/docs/source/en/deepspeed.md at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=427de887-db8b-40a1-9dba-accee8329079)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1235497314949398570)** (63 条消息🔥🔥):

- **文档困惑已解决**：一名成员分享了 [Open Interpreter 本地安装文档](https://docs.openinterpreter.com/guides/running-locally)的链接，其中特别包含了针对 **Ollama**、**Jan.ai** 和 **Llamafile** 的说明，并重点强调了 **dolphin-mixtral**。
- **提示词调整以实现简洁**：一位成员建议在 Open Interpreter 中使用 `--profile 01` 命令，以避免重复回顾步骤和计划。他们还分享了[相关系统消息的链接](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/01.py)。
- **开源 AI 黑客松公告**：发出了加入西雅图 Microsoft 开源 AI 黑客松团队的邀请，详情和注册[链接在此](https://lu.ma/iu1wijgd)。
- **Open Interpreter 服务器托管咨询**：一位成员询问是否可以托管一个运行 **Open Interpreter** 的服务器供他人连接，另一位成员确认这是可行的，并指出可以使用 `--api_base` 配合 `--model openai/custom --api_key dummykey`。
- **移动设备本地模型托管指南**：有人寻求关于设置本地 Open Interpreter 模型以供移动设备访问的信息，随后提供了 GitHub 文档链接，涉及 [Android 设备设置](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android)和 [在本地运行 Open Interpreter](https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://ip_address:port/v1`">未找到标题</a>：未找到描述</li><li><a href="https://docs.openinterpreter.com/guides/running-locally">本地运行 - Open Interpreter</a>：未找到描述</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2Fopen-interpreter%20skill&type=code">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://tenor.com/view/life-barrel-me-roll-gif-17943995">Life Barrel GIF - Life Barrel Me - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2F01%20skill&type=code">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://lu.ma/iu1wijgd">开源 AI 黑客松 #4 · Luma</a>：根据上一次黑客松的反馈，我们已经找到了 LLM 的赞助商！OctoAI 将为所有注册者提供获得 50 美元的机会……</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>：计算机的自然语言接口。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>：计算机的自然语言接口。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py">interpreter/core/computer/skills/skills.py 的历史记录 - OpenInterpreter/open-interpreter</a>：计算机的自然语言接口。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1235508299227725896)** (10 条消息🔥): 

- **扬声器选择：一个精细的过程**：电子项目的扬声器选择正在接受密切评估，**Ben** 正在与供应商讨论方案，并考虑与 PCB 设计的集成。这一决策预计将持续数周，后续将根据验证结果进行更新。

- **公平竞争：评测已发布的产品**：在讨论产品评测时，一名成员表示，评测已经正式发布的产品是完全合理的，这暗示了对评测者在该产品领域专业理解的信心。

- **Whisper RKNN 的速度提升**：分享了一个针对 **Rockchip RK3588 SBCs** 的 Whisper RKNN 改进分支，根据 [rbrisita 的 GitHub](https://github.com/rbrisita/01/tree/rknn)，其性能提升了 250%。贡献者计划下一步引入 LLM RKNN 功能。

- **故障排除 Interpreter 错误**：一位遇到 `interpreter` 命令错误的用户被建议在执行时添加 `--api_key dummykey`。为了获得进一步帮助，他们被引导至特定的 Discord 频道进行问题讨论。

- **iOS 的 TMC 协议进展**：关于为 iOS 实现 **TMC 协议** 的讨论正在进行中，该协议允许访问日历和 iMessage 等原生功能。成员正在权衡开发过程中 TMC 相比标准 function calling 的优势。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pastebin.com/zGkZRhPs">error file - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/rbrisita/01/tree/rknn">GitHub - rbrisita/01 at rknn</a>：开源语言模型计算机。通过在 GitHub 上创建账号为 rbrisita/01 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1235545486342225962)** (2 条消息): 

- **开源 AI Vtuber 工具包发布**：*Nikechan* 展示了一个 **AI Vtuber 入门工具包**，运行需要 OpenAI key 和 YouTube Key。该项目已在 [GitHub](https://github.com/tegnike/nike-ChatVRM) 上线，并通过 [Twitter](https://twitter.com/tegnike/status/1784924881047503202) 进行了发布。

- **AI Vtuber 离线运行**：*Hensonliga* 分享了他们的 AI Vtuber 仓库，该仓库完全离线运行，无需 API，并指出内容可以是 uncensored 的。公告包括一个 [YouTube 演示](https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb) 和 [GitHub 仓库](https://github.com/neurokitti/VtuberAI.git) 链接。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/tegnike/nike-ChatVRM">GitHub - tegnike/nike-ChatVRM: 誰でもAITuberお試しキット</a>：谁都能尝试的 AITuber 工具包。通过在 GitHub 上创建账号为 tegnike/nike-ChatVRM 的开发做出贡献。</li><li><a href="https://youtu.be/buaK84oSWCU?si=P02NIYHvrVj7m8Lb">Neuro Sama Competitor running Locally! V0.2 [FOSS, Local, No API]</a>：我马上创建一个 GitHub 仓库。抱歉我的麦克风质量不好，我正在使用耳机麦克风，蓝牙带宽严重损耗了音质，而且显存也有点……</li><li><a href="https://github.com/neurokitti/VtuberAI.git">GitHub - neurokitti/VtuberAI</a>：通过在 GitHub 上创建账号为 neurokitti/VtuberAI 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1235590225275654174)** (1 条消息): 

- **流量激增导致服务波动**：由于流量大幅增加，OpenRouter 经历了高于正常的错误率，导致间歇性问题。
- **扩容工作正在进行中**：太平洋时间上午 7:30 的更新表明，应对流量激增的扩容过程仍在进行中，问题有所减少但尚未完全消除。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1235502261099302976)** (72 条消息🔥🔥): 

- **探索支付替代方案**：成员询问了 **OpenRouter** 支持额外支付方式（如 [Stripe](https://stripe.com) 以包含 **微信支付 (WeChat Pay)** 和 **支付宝 (Alipay)**）的可能性，并指出这需要额外的文书工作。
- **即将推出的 AI 模型预告**：围绕 **LLaMA-3** 等新型大规模语言模型以及 Soliloquy 等公司可能发布的新作，社区充满了兴奋和猜测，同时也承认了专有模型的局限性。
- **关注微调后模型变笨的问题**：围绕在没有 instruct 数据集的情况下微调大语言模型的后果展开了技术讨论，建议 **将旧数据与新数据进行批处理可以防止灾难性遗忘 (catastrophic forgetting)**。
- **对更简便支付集成的兴趣**：有人建议开发一个与 **Google 支付服务** 集成的 App，以简化交易。
- **解决 Gemini Pro 问题**：针对用户遇到的 **Gemini Pro** 消息问题（特别是以 "assistant" 角色开头的问题），提出了更新和变通方案，包括在 prompt 中预置 user 角色消息。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba-instruct">为企业打造：介绍 AI21 的 Jamba-Instruct 模型</a>：作为混合 SSM-Transformer Jamba 模型的指令微调版本，Jamba-Instruct 专为可靠的商业用途而构建，具有一流的质量和性能。</li><li><a href="https://huggingface.co/collections/nvidia/chatqa-15-662ebbf6acc85f5c444029a8">Llama3-ChatQA-1.5 - 一个 nvidia 集合</a>：未找到描述。
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 条消息): 

angry.penguin: https://huggingface.co/spaces/YupengZhou/StoryDiffusion
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235630378379771994)** (15 条消息🔥): 

- **神秘消息困扰 AI 开发者**：AI Stack 开发者对 **ai-town** 中阻塞对话流的 *空消息或数字字符串* 感到困惑，该项目使用了 **ollama** 和 **llama3 8b**。有人建议可能是 *Tokenizer 问题*，但目前尚无定论。

- **对 Cocktail Peanut 的高度赞赏**：一位成员向 **cocktail peanut** 致敬，简单地表示他们正在做“神圣的工作”，但未提供具体指代什么工作的上下文。

- **没有领导层的 AI 社会**：成员们讨论了 AI 是否会在模拟中 *选举领导者*，共识是目前没有市长或民选官员。大家对原始模拟论文中关于这一点的描述表示好奇。

- **简化 AI 角色职能**：一位成员提到，在 AI 角色的玩家简介（bios）中设置 *市长选举* 可以很容易地实现。

- **分享 AI Town 经验与工具**：@.casado 分享了链接，推广了 [@TheoMediaAI 对 AI 模拟的探索](https://x.com/TheoMediaAI/status/1786377663889678437)，以及 [@cocktailpeanut 的新 Web 应用](https://x.com/cocktailpeanut/status/1786421948638965870)，该应用允许通过导入 sqlite 文件来 *回放任何 AI Town*。后者支持 Mac 和 Linux，且需要 **ollama**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheoMediaAI/status/1786377663889678437">Theoretically Media (@TheoMediaAI) 的推文</a>：探索两个卓越的 AI 世界模拟：首先是来自 @fablesimulation 的 AI-Westworld（公开测试版已开放！），同时也尝试了 @realaitown，但重现了史上最棒的电影（The THI...</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">cocktail peanut (@cocktailpeanut) 的推文</a>：介绍 AI Town Player。你知道整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中吗？我逆向工程了其架构并构建了一个 Web 应用，让任何人都可以回放任何 A...
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1235763541642838069)** (26 条消息🔥): 

- **Node 版本阻碍本地后端进度**：一位成员在运行 `convex-local-backend` 时因 Node 版本不正确而遇到错误：*Wrong node version v19.9.0 installed at node*。建议使用 `nvm use 18` 切换到 Node 18 版本。

- **后端 Bug 导致本地开发停滞**：另一位成员在 Ubuntu 18 上尝试运行 `convex-local-backend` 时遇到了多个错误。问题包括 Node 版本、`rush buildCacheEnabled` 以及类型错误 *(Unknown file extension ".ts")*。

- **寻求更简单的设置**：由于这些复杂情况，该成员询问是否有 Docker 构建以简化部署过程。此外还提到了在本地使用 `ollama` 并在远程使用 `convex` 的替代方案。

- **为社区项目分享资源**：有人请求与另一位正在进行 *Pinokio* 构建的用户分享一张更大的地图。成员 edgarhnd 同意分享该地图。

- **LLama-Farm 项目连接本地机器**：介绍了一个名为 *llama-farm* 的项目，旨在将一台或多台运行 `Ollama` 的机器连接到云端后端或托管网站，从而在不将机器暴露于公共互联网请求的情况下使用本地 LLM 计算能力。

**提到的链接**：<a href="https://github.com/get-convex/convex-backend/issues/1">TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot; for /app/npm-packages/convex/src/cli/index.ts · Issue #1 · get-convex/convex-backend</a>：我按照先决条件中的步骤操作，然后在仅运行 run-local-backend 时遇到了这个错误：Failed to run convex deploy: TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension &quot;.ts&quot;...

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/1235713660966670386)** (2 条消息):

- **情绪化的企鹅与 Discord 机器人**：一位成员展示了一个表达深思或怀疑的表情符号，可能正准备讨论或思考与 Raspberry Pi 上的 AI 相关的问题。
- **频道符合用户兴趣**：另一位成员表示 ai-raspberry-pi 频道非常符合他们的兴趣，暗示他们可能会参与或贡献关于使用 Raspberry Pi 进行 AI 开发的讨论。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1235517093793759274)** (26 messages🔥): 

- **SoundStream 的实现挑战**：一位成员在实现 Google 的 SoundStream 论文时遇到了困难，原因是未指定的索引名称和值。另一位成员指出了一份可能有所帮助的现有代码库，可在 [GitHub](https://github.com/wesbz/SoundStream) 上找到。
  
- **新手欢迎与课程分享**：一位 AI 生成艺术的新手在完成 Udemy 上的 Stable Diffusion 课程后，提出免费分享该课程，希望能建立联系并从社区学习更高级的技能。

- **聊天中的投资策略**：成员们幽默地讨论了他们的投资策略，从寻求能让资金翻 10 倍的服务，到更倾向于让资金减半的服务。

- **关于模型训练局限性的见解**：在 StableDiffusion 的 subreddit 中，讨论提到同时使用 T5 text encoder 和 CLIP 可能不会像预期那样提高提示词遵循度，一些人对此表示惊讶，另一些人则认为高 CLIP dropout 可能是潜在因素。

- **StableDiffusion 开发更新**：来自 Stable Diffusion 社区的更新表明，由于硬件限制，重点已从大型模型转向小型模型的架构和训练改进。对话还涉及了正确使用 CLIP 进行训练以避免偏差和局限性的重要性。

**提及的链接**：<a href="https://github.com/wesbz/SoundStream">GitHub - wesbz/SoundStream: This repository is an implementation of this article: https://arxiv.org/pdf/2107.03312.pdf</a>：该仓库是此文章的实现：https://arxiv.org/pdf/2107.03312.pdf - wesbz/SoundStream

  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1235718206744957018)** (7 messages): 

- **机器人驱逐大队**：两位成员就从讨论中移除一个疑似机器人的行为进行了幽默互动，其中一位成员愉快地注意到他们对聊天的及时关注。
- **对数据集选择的审查**：一位成员询问为什么实验不是在 MNIST、CIFAR 或 ImageNet 等标准数据集上进行，而是在合成数据集上进行。另一位成员将此选择归因于展示可解释性的目标。
- **可解释性 vs 现实世界应用**：在讨论了实验对可解释性的关注后，另一位成员表示怀疑，指出方法需要解决现实世界的任务才真正具有说服力。
- **新工具亮相**：一位成员分享了 [StoryDiffusion](https://storydiffusion.github.io/) 的链接，但未提供关于其用途或相关性的额外背景信息。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1235497831431798866)** (26 messages🔥): 

- **Text Embedding 的集成困境**：一位用户表示在将 Text Embedding 模型与 LangChain 集成时遇到困难，提到需要使用 **SageMaker endpoint** 而不是 API key。该用户寻求此类集成的替代方法或资源建议。
  
- **LangChain 包版本混淆**：一位成员提出了关于安装 `langchain` PyPI package 的问题，注意到指定的 `langchain-openai` 版本非常旧（`<=0.1`），并询问考虑到当前 `langchain-openai` 版本已显著更新，这是否是出于兼容性原因的有意为之。
  
- **寻找聊天机器人爱好者**：一位用户询问如何找到专注于开发对话式聊天机器人的社区，并向其他成员寻求推荐。
  
- **CSV 的数据检索查询**：一位成员询问如何将 CSV 文件中的单列嵌入到 LangChain 应用中，并随后根据响应从另一列检索数据，涉及电子邮件查询的使用案例。
  
- **黑客松预告！**：发布了一项关于即将举行的名为 **BeeLoud** 的黑客松公告，参与者被挑战在 54 小时内构建 AI 产品，潜在奖金池高达 $25,000。该活动欢迎各种技能背景的人士，定于 5 月 10 日至 12 日举行，参与者来自全球各地。

- **LangChain 用户访谈请求**：一位用户请求与经常使用 LangChain 或其他框架构建 AI Agent 的开发者讨论所面临的最大挑战，并提供了一个预约通话的链接以进行详细交流。

- **SQL Agent 功能查询**：引发了一场关于是否可以使用 LangChain 中的 SQL Agent 调用 **MSSQL 函数**的讨论，随后详细解释了如何使用 `SqlToolkit` 执行此类操作，并提供了相关指南的链接。

- **LangChain RAG 实现见解**：一位正在准备涉及 LangChain 开发岗位的用户询问了面试准备的关键点和建议，特别是关于通过 LangChain 实现 RAG 的相关内容。

- **在 LangChain 中处理大型数据库**：成员们讨论了使用 LLM 查询数据库的各种方法，辩论了是将数据库数据转换为自然语言文本，还是使用 ChatGPT 将自然语言转换为 SQL 查询，并考虑了在这些范式下处理大型数据库的挑战。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://beeloud.xyz/build/">Build - Beeloud</a>：你能在 3 天内建立下一个价值十亿美元的初创公司吗？Sam Altman 和他的伙伴们打赌你可以。你正式被邀请参加这场黑客松。我接受……继续阅读...</li><li><a href="https://calendly.com/leonchen-1/30min">30 分钟会议 - Leon Chen</a>：未找到描述</li><li><a href="https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema">未找到标题</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/13931>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1235647398718738433)** (1 条消息): 

- **澄清反馈提交的困惑**：一位成员对使用 **langserve 反馈端点**时的反馈提交机制感到困惑。解释指出，来自 Langserve 的 "OK" 响应仅表示提交成功，并不确认 langsmith 已记录，因为如果服务器认为请求未经身份验证或无效，请求可能会被拒绝。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1235659099933380659)** (3 条消息): 

- **使用 RAG 提升邮件草拟效率**：**LangChain 的 LangGraph Agent** 增强功能现在包括 [检索增强生成 (RAG)](https://medium.com/ai-advances/enhancing-langchains-langgraph-agents-with-rag-for-intelligent-email-drafting-a5fab21e05da)，以实现更智能的邮件草拟能力。这篇 Medium 文章详细介绍了这种集成如何显著提高 AI 生成邮件沟通的效率和质量。

- **LangChain Java 移植版可用**：对于有兴趣在 Java 中使用 **LangChain** 的开发者，[langchain4j](https://github.com/langchain4j/langchain4j) 提供了 LangChain 的 Java 版本，扩展了将其集成到各种应用中的可能性。

- **Dragonfly 与 LangChain 集成**：一篇新博客文章强调了内存数据存储 **Dragonfly** 与 **LangChain** 的集成，用于管理聊天上下文并提高 AI 驱动应用的性能。有关此增强功能的详细信息和代码片段可以在 [博客文章](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly) 中找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly">在 LangChain 聊天机器人中使用 Dragonfly 进行高效的上下文管理</a>：探索使用 Dragonfly 为 LangChain OpenAI 聊天机器人进行高效的上下文管理，通过缓存技术增强性能和用户体验。</li><li><a href="https://github.com/langchain4j/langchain4j">GitHub - langchain4j/langchain4j: LangChain 的 Java 版本</a>：LangChain 的 Java 版本。通过在 GitHub 上创建账号为 langchain4j/langchain4j 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1235621942157905992)** (13 条消息🔥):

- **实施意图的迹象**：似乎有人在考虑实施一些不常见的做法。具体细节尚不明确，但已表达了开始实施过程的兴趣。
- **技术报告的等待博弈**：大家都在期待一份尚未发布的技术报告，这似乎引起了一些困惑。该报告的缺失被归因于与数据相关的时间尺度限制。
- **奖励模型竞赛提醒**：提到了 LMSYS 举办的一项奖金高达 10 万美元的奖励模型竞赛，这与早期的 Kaggle 竞赛相似，并引发了要求举办类似的 20 万美元 Interconnects 竞赛的呼声。
- **关于 Ensembling 的观点**：集成 (Ensembling) 奖励模型的概念得到了认可，但被认为并非最优方案，尽管它可能足以让某些参赛者获得优势。
- **PPO 与 REINFORCE 的联系**：有讨论指出，在特定超参数设置下（可能是在关闭步长限制器时），Proximal Policy Optimization (PPO) 理论上可以简化为 REINFORCE 算法。分享了 OpenAI 的 [Spinning Up 文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 链接以进一步说明。

**提到的链接**：<a href="https://spinningup.openai.com/en/latest/algorithms/ppo.html">Proximal Policy Optimization &mdash; Spinning Up 文档</a>：未找到描述

---

**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1235547821285834812)** (4 条消息): 

- **潜在模型泄露引发的争议**：一名成员讨论了一个具有 *极其特殊量化 (oddly specific quant)* 的泄露模型可能来自 **GDM** 的可能性，并引用了 [@teortaxesTex 的推文](https://x.com/teortaxestex/status/1785974744556187731?s=46)。怀疑源于一些奇怪的细节，如 *突然出现的 4chan 链接、临时 HF 账号以及 Reddit 评论*。
  
- **4chan 上的量化泄露引发好奇**：一位用户总结道，一个“从 4chan 掉落的 Llama 3 随机量化版本”可能源自 **GDM**。

- **研究论文因缺失 RewardBench 分数而表现平平**：一名成员分享了一篇[论文链接](https://arxiv.org/abs/2405.01535)，该论文未报告 RewardBench 分数，并添加了捂脸表情符号反应暗示其表现不佳。

- **Prometheus 2：GPT-4 的挑战者？**：该论文介绍了 **Prometheus 2**，这是一款评估者语言模型，定位为 GPT-4 等专有 LM 的更好替代方案。它声称与人类判断高度一致，并能处理各种类型的评估。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>：GPT-4 等专有 LM 常被用于评估各种 LM 的回答质量。然而，出于透明度、可控性和成本等方面的考虑，强烈促使了……</li><li><a href="https://x.com/teortaxestex/status/1785974744556187731?s=46">Teortaxes▶️ (@teortaxesTex) 的推文</a>：……实际上，我到底为什么要假设这不是他们的模型，为了集体渗透测试而分发的——类似于 miqu 的极其特殊的量化泄露以阻止改进——突然的 4chan 链接，临时账号……
</li>
</ul>

</div>

---

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1235683288828346498)** (5 条消息): 

- **10 万美元人类偏好预测挑战赛**：[LMSYS 和 Kaggle 启动了一项竞赛](https://x.com/lmsysorg/status/1786100697504833572?s=46)，参赛者需预测用户对语言模型 (LM) 回答的偏好。数据集包含超过 55,000 条对话，涉及 **GPT-4, Claude 2, Llama 2 和 Mistral** 等 LLM。

- **简短的胜利欢呼**：一名成员简单地评论道 "mogged"。

- **Kaggle 对研究者的吸引力**：一名成员询问研究人员通常是否喜欢 **Kaggle** 之类的平台。

- **屡获成功引发质疑**：针对竞赛公告，一名成员表示难以置信，指出“他不能一直这样逃脱惩罚 (he can't keep getting away with this)”。

- **关于承诺的闲聊**：对话以更随意的语气继续，提到一位“John”对一名成员说了“也许”，暗示可能存在参与活动或项目的背景。

**提到的链接**：<a href="https://x.com/lmsysorg/status/1786100697504833572?s=46">lmsys.org (@lmsysorg) 的推文</a>：激动人心的消息——我们很高兴地宣布 LMSYS + @kaggle 正在启动一项人类偏好预测竞赛，奖金达 100,000 美元！你的挑战是预测用户的哪些回答……

---

**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1235674317413417050)** (8 条消息🔥): 

- **RLHF 中有价值的价值函数**：一位成员思考为什么发布了奖励函数（Reward Functions）却不发布 RLHF 训练期间获得的价值函数（Value Functions），质疑是否某种程度上没有产生价值函数。另一位成员澄清说，在使用 PPO 等算法时，**确实会获得价值函数**。

- **奖励模型发布惯例受到质疑**：有人提到，声称发布奖励模型或函数是社区的标准做法可能是一种**夸大其词**。

- **价值函数的价值得到认可**：尽管其发布情况尚不明确，但大家公认**价值函数被认为非常有价值**，特别是在 Planning（规划）的语境下。

- **价值函数的研究空白？**：一位成员推测，目前缺乏专注于**经典 RL 中价值函数的价值**的研究，这意味着进一步探索的机会。

- **价值与信用分配之间的联系**：**PPO 中的价值函数**与 **DPO 中的信用分配（Credit Assignment）**之间的关系被指出是一个潜在的有趣研究方向。
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1235491704820465744)** (21 条消息🔥): 

- **大型文档的搜索系统设计**：一位成员探讨了为大型 PDF 构建搜索系统的想法，并考虑生成 Embeddings 以进行语义搜索，使用 LLM 总结文档以供检索，以及对 LLM 提取的关键信息进行索引。
  
- **Llama 与 Command R+ 的 Tokenization 说明**：一位成员询问在使用 llama-cpp-python 库和 Command R+ 生成文本时，是否需要手动添加 "<BOS_TOKEN>"，因为他注意到在 Tokenization 过程中会自动添加该标记。

- **RAG 的 Cohere API Key 咨询**：一位用户询问是否可以使用免费的 Cohere API Key 进行 RAG，**另一位成员确认其可用性，但指出了速率限制（Rate Limitations）**。

- **关于 C4AI Command R+ 实现的讨论**：成员们分享了 HuggingFace 上的 **[C4AI Command R+ 模型](https://huggingface.co/CohereForAI/c4ai-command-r-plus)** 链接和[量化版本](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit)，以及实现的各种技术参数，并讨论了在不同系统要求下本地运行该模型的情况。

- **Code Interpreter SDK 发布公告**：一位成员在 Twitter 上分享了 [Code Interpreter SDK 发布](https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA)的演示，另一位成员则质疑考虑到之前已有类似技术，此次发布的独特性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/library/command-r">command-r</a>: Command R 是一款针对对话交互和长上下文任务优化的 Large Language Model。</li><li><a href="https://x.com/tereza_tizkova/status/1786058519701254268?s=46&t=yvqplJRJNpP5EM3LZLMQlA">Tereza Tizkova (@tereza_tizkova) 的推文</a>: 🚀 我们正在发布 @e2b_dev Code Interpreter SDK 🧠 它是任何 AI 应用的构建模块 - 用于代码解释的 SDK！使用它来构建 🔸 高级数据分析师 🔸 生成式 UI 🔸 AI 软件...</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: 暂无描述</li><li><a href="https://ollama.com/library/command-r-plus">command-r-plus</a>: Command R+ 是一款功能强大、可扩展的 Large Language Model，专为卓越处理真实世界企业用例而打造。
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1235494486143008790)** (19 条消息🔥):

- **llamafile 作为 Linux 服务**：分享了一个在 Rocky Linux 9 上将 **llamafile** 作为服务启动的 systemd 脚本，详细说明了使用特定参数（如服务器端口和模型路径）运行 llamafile 所需的执行命令和环境配置。
- **服务器 Base URL 的功能请求**：针对在服务器模式下为 llamafile 指定 base URL 的功能请求，已通过 [GitHub issue 链接](https://github.com/Mozilla-Ocho/llamafile/issues/388) 进行处理，表达了通过 Nginx 代理在子目录下提供 llamafile 服务的需求。
- **对 Distil Whisper 德语模型的兴趣**：有人对引入用于语音识别的 whisper 模型（如 [distil-whisper-large-v3-german](https://huggingface.co/primeline/distil-whisper-large-v3-german)）表现出好奇，并可能撰写一篇关于其应用（包括 STT -> LLM -> TTS 的假设流水线）的博客文章。
- **Embedding 方向差异**：讨论了一个问题，即 llamafile 和 llama.cpp 生成的 embeddings 显示出较低的余弦相似度（cosine similarity），表明方向不同。该问题由 [GitHub issue](https://github.com/Mozilla-Ocho/llamafile/issues/391) 证明，并使用提供的 Python 脚本进行了测试。
- **与文档/代码对话**：关于如何让 llamafile 摄取文档和代码以进行对话交互的问题，建议使用 `curl` API 调用，并参考了 [llama.cpp chat 脚本](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64) 中的示例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:8080"):">未找到标题</a>：未找到描述</li><li><a href="http://localhost:8080")">未找到标题</a>：未找到描述</li><li><a href="http://localhost:8081")">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/primeline/distil-whisper-large-v3-german">primeline/distil-whisper-large-v3-german · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/apple/OpenELM-3B-Instruct">apple/OpenELM-3B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/388">功能请求：为服务器模式指定 base URL 的选项 · Issue #388 · Mozilla-Ocho/llamafile</a>：我一直在测试使用 Nginx 作为代理在子目录下提供 llamafile 服务。即能够通过如下 URL 访问 llamafile 服务器：https://mydomain.com/llamafile/ Llamafile...</li><li><a href="https://huggingface.co/models?search=OpenELM-3B-Instruct-gguf">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/391">来自 server.cpp `/embedding` 端点的异常输出 · Issue #391 · Mozilla-Ocho/llamafile</a>：问题是什么？在 llamafile 中运行的模型生成的 embeddings 似乎与 llama.cpp 生成的显著不同。llama.cpp 的 embeddings 非常接近（约 0.99 余弦相似度）...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.sh#L64">llama.cpp/examples/server/chat.sh at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1235486834499522650)** (4 条消息): 

- **Tiny 进度更新**：一位成员询问 **progress**，另一位确认两天前取得了实质性进展。 
- **贡献里程碑**：另一位成员分享了他们向项目提交 **first commit** 的热情，并对成功提交表示喜悦。
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1235659100998729759)** (13 条消息🔥): 

- **关于 Blobfile 重要性的澄清**：质疑了 `examples/llama.py` 中 `blobfile` 的效用。澄清了 `load_tiktoken_bpe` 依赖于 `blobfile`。

- **前向传播计算图问题**：一位成员在为简单神经网络生成前向传播计算图（forward pass compute graph）时遇到问题。建议通过取消 `out.item()` 的注释或使用 `out.realize()` 来确保计算，并通过安装必要的库来解决 `NameError`。

- **已安装 Networkx 但缺少 pydot**：尽管安装了 `networkx`，上述错误仍然存在，最终通过安装 `pydot` 解决。

- **Graphviz 安装解决 dot 命令错误**：在实施安装 `pydot` 的方案后，遇到了关于缺少 `dot` 命令的新错误，通过安装 `graphviz` 解决。

- **建议更新文档**：一位成员建议更新文档，添加一个提示，说明安装 `graphviz` 可以解决 `sh: dot: command not found` 错误。
  

---



**AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1235742995437977641)** (1 messages): 

- **Jamba-Instruct 成为焦点**：AI21 Labs 宣布推出 **Jamba-Instruct**，这是其混合 SSM-Transformer **Jamba** 模型的指令微调版本。他们征求反馈，并表示愿意为需要超过初始 256K context window 的用例提供支持。

- **了解 Jamba-Instruct 的全部信息**：为了深入了解，AI21 Labs 鼓励阅读 [AI21's Blog](https://www.ai21.com/blog/announcing-jamba-instruct) 上的 *Jamba-Instruct 博客文章*，其中详细介绍了 Jamba-Instruct 如何在商业应用中表现出卓越的质量和性能。

**Link mentioned**: <a href="https://www.ai21.com/blog/announcing-jamba-instruct">Built for the Enterprise: Introducing AI21’s Jamba-Instruct Model</a>：作为我们混合 SSM-Transformer Jamba 模型的指令微调版本，Jamba-Instruct 专为可靠的商业用途而构建，具有同类最佳的质量和性能。

  

---


**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1235603967384490037)** (4 messages): 

- **Jamba-Instruct 发布**：AI21 Labs 宣布推出 **Jamba-Instruct**，并通过 [Twitter 帖子](https://twitter.com/AI21Labs/status/1786038528901542312) 进行了分享。
- **探索更大的 Context Windows**：针对关于大于 256k context window 的询问，一位 AI21 Labs 工作人员表示愿意探索 **更高的 context windows**，并邀请该成员通过私信讨论用例。
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1235515365598167141)** (2 messages): 

- **热情问候**：一位成员用简单的 "Hello" 向社区打招呼。
- **提供算力资助**：对于那些寻求 **快速算力资助 (fast compute grants)** 的人，一位成员分享了来自 @PrimeIntellect 的 Twitter 帖子链接：[快速算力资助推文](https://twitter.com/PrimeIntellect/status/1786386588726960167)。
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1235956851133386872)** (2 messages): 

- **LLaMA 量化困境**：一位 Discord 成员重点介绍了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/)，讨论了与 LLaMA 2 相比，量化对 **LLaMA 3** 质量的影响。他们链接了一篇 [arXiv 论文](https://arxiv.org/abs/2404.14047)，详细说明了低比特量化带来的性能下降，并对 post-training quantization 方法提出了质疑。
- **量化导致细节丢失**：一位成员表示，**Meta 的 LLaMA** 进行了显著的量化，它忽略了 *Chinchilla scaling law* 并使用了 15T tokens，这可能是导致重大信息丢失、影响性能的原因。这表明在大型模型中，随着精度进一步降低，退化的风险更大。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.14047">How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study</a>：Meta 的 LLaMA 家族已成为最强大的开源大语言模型 (LLM) 系列之一。值得注意的是，LLaMA3 模型最近发布，并在各项指标上取得了令人印象深刻的性能...
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1235948611292893263)** (1 messages): 

- **为 Skunkworks 项目提供快速算力资助**：一位成员提到他们渴望资助一些令人兴奋的 **Skunkworks 项目**，并提供了 [Twitter 链接以获取详情](https://twitter.com/PrimeIntellect/status/1786386588726960167)。如果你正在寻找 fast compute grants，这可能是一个机会。
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1235576027233910865)** (1 messages): 

- **数字家务烦恼**：一位成员表示需要一个 LLM，能够协助清理散落在硬盘各个目录中、占用空间的 **7B localmodels**。这种挫败感源于众多应用程序和库导致的混乱。
  

---