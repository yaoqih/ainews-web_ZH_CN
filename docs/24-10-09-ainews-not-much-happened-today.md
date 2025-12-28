---
companies:
- meta-ai-fair
- anthropic
- togethercompute
- hugging-face
date: '2024-10-10T01:02:45.022514Z'
description: '**杰弗里·辛顿 (Geoffrey Hinton)** 和 **约翰·霍普菲尔德 (John Hopfield)** 因在神经网络方面的基础性工作荣获
  **诺贝尔物理学奖**，该工作将人工智能与物理学联系在了一起。


  **Meta AI** 推出了一个 **130 亿参数的音频生成模型**，作为 Meta Movie Gen 的一部分，用于生成与视频同步的音频。**Anthropic**
  发布了 **Message Batches API**，支持以一半的成本异步处理多达 10,000 个查询。**Together Compute** 发布了 **Flux
  Schnell**，并提供为期 3 个月的免费使用。


  **rohanpaul_ai** 重点介绍了 **PrefixQuant** 量化和 **Prompt Caching（提示词缓存）** 等用于低延迟推理的新技术。**LangGraph**
  增加了长期记忆支持，用于持久化文档存储。**Hex-LLM** 框架问世，旨在为 Hugging Face 模型提供基于 TPU 的低成本、高吞吐量的大语言模型（LLM）推理服务。


  此外，关于人工智能安全的讨论强调了科学领域的性别平等，同时人们也对媒体和好莱坞推动的过早的人工智能监管表示了担忧。'
id: a48f879b-435a-4f55-9cd2-eff30a9f057d
models:
- flux-schnell
original_slug: ainews-not-much-happened-today-6017
people:
- geoffrey-hinton
- john-hopfield
- demis-hassabis
- rohanpaul_ai
- svpino
- hwchase17
- shreyar
- philschmid
- mmitchell_ai
- bindureddy
title: 今天没什么事。
topics:
- audio-generation
- quantization
- prompt-caching
- long-term-memory
- llm-serving-framework
- hallucination-detection
- ai-safety
- ai-governance
---

<!-- buttondown-editor-mode: plaintext -->**AI 是你成为化学家所需的一切。**

> 2024/10/8-2024/10/9 的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**228** 个频道和 **1872** 条消息）。预计节省阅读时间（以 200wpm 计算）：**222 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天只有一些零散的 smol 故事：

- [某 AI 女友初创公司被黑，泄露 200 万封电子邮件和 prompts](https://x.com/troyhunt/status/1844003903026983200)
- [OpenAI 预计两年内亏损 140 亿](https://www.theinformation.com/articles/openai-projections-imply-losses-tripling-to-14-billion-in-2026?rc=ytp67n)
- [Sequoia 爱上了 o1](https://www.sequoiacap.com/article/generative-ais-act-o1/)
- [尽管有更多人离职，SearchGPT 仍继续推出](https://x.com/thomasschulzz/status/1844062893723250940?s=46) [更多人离职](https://x.com/Luke_Metz/status/1844161466032914645)
- [Demis Hassabis 和 John Jumper 因 Alphafold 获得诺贝尔化学奖](https://x.com/NobelPrize/status/1843951197960777760)

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 进展与行业新闻**

- **诺贝尔物理学奖**：[@ilyasut](https://twitter.com/ilyasut/status/1843739228758520186) 宣布 **Geoffrey Hinton 因其对 AI 的贡献获得诺贝尔物理学奖**。[@demishassabis](https://twitter.com/demishassabis/status/1843713404613312532) 指出 Hinton “为深度学习革命奠定了基础，而这正是现代 AI 领域的基石”。该奖项由 **John Hopfield** 共同获得，以表彰他们在神经网络及其与物理学概念联系方面的工作。

- **模型开发**：[@AIatMeta](https://twitter.com/AIatMeta/status/1843708845509751073) 推出了一个 **13B 参数音频生成模型**，作为 Meta Movie Gen 的一部分，能够生成与视频同步的高质量音频。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1843694793911873557) 重点介绍了 PMRF，这是一种新型的照片级图像修复算法。

- **AI 工具与平台**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1843695536614060201) 推出了 **Message Batches API**，允许异步处理多达 10,000 个查询，成本比标准 API 调用低 50%。[@togethercompute](https://twitter.com/togethercompute/status/1843695278869885351) 宣布 **Flux Schnell** 这一新模型在未来 3 个月内可在其 API 中免费使用。

- **AI 研究**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1843681486618181784) 讨论了 **PrefixQuant**，这是一种新型量化技术，其性能优于昂贵的逐 Token 动态量化。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1843735064791920754) 还重点介绍了一篇关于使用 Prompt Markup Language (PML) 进行低延迟推理的 **Prompt Caching** 论文。

**AI 工程与开发**

- **开发工具**：[@svpino](https://twitter.com/svpino/status/1843688106991771700) 对在不同代码编辑器之间切换表示沮丧，突显了开发者在寻找完美工具方面面临的持续挑战。[@awnihannun](https://twitter.com/awnihannun/status/1843724487315075407) 展示了 **LM Studio 中的 MLX 后端**，演示了其在 M1 笔记本电脑上的性能。

- **AI 框架**：[@hwchase17](https://twitter.com/hwchase17/status/1843677417405378910) 宣布 **LangGraph** 支持“长期记忆”，允许在对话线程中进行持久化文档存储和基于内容的过滤。

- **AI 评估**：[@ShreyaR](https://twitter.com/ShreyaR/status/1843784773346701640) 分享了比较 OpenAI DevDay Eval 产品和 Bespoke Labs 的 Minicheck 用于幻觉检测的基准测试，结果显示 Minicheck 在检测幻觉方面具有更好的准确性。

- **AI 基础设施**：[@_philschmid](https://twitter.com/_philschmid/status/1843679923380097420) 介绍了 **Hex-LLM**，这是一个专为 TPU 设计的新型 LLM 服务框架，为来自 Hugging Face 的开源模型提供低成本、高吞吐量的部署。

**AI 伦理与社会影响**

- **AI 安全关注**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1843694088962617440) 强调了男性在科学领域积极支持性别平等的重要性，并指出仅靠女性的力量是有限的，尤其是当她们在某个领域占比不足 10% 时。

- **AI 治理**：[@bindureddy](https://twitter.com/bindureddy/status/1843726967016952319) 认为主流媒体和好莱坞希望过早地监管 AI，以保护其“名人”地位，将 AI 视为对其生存的威胁。

**迷因与幽默**

- [@DrJimFan](https://twitter.com/DrJimFan/status/1843681423443800315) 分享了一个幽默的 AI 术语“银河系漫游指南式更名”，将机器学习概念映射到物理学术语。

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1843708835535695986) 发布了一张对比 Google 和 Perplexity 搜索结果差异的图片，突显了 Perplexity 被感知到的优越性。

- [@jxmnop](https://twitter.com/jxmnop/status/1843648364459770191) 拿诺贝尔物理学奖颁给“ptrblock”以表彰其“对物理学的根本贡献”开玩笑，调侃了该奖项颁给 AI 研究人员的意外性。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 持续微调：增强 LLM 性能的新方法**

- **将 Llama 3.2 视觉适配器（vision adapters）合并到 3.1 微调模型（finetunes）上** ([Score: 40, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1fzduyx/merging_llama_32_vision_adapters_onto_31_finetunes/))：该帖子讨论了**将 Llama 3.2 视觉适配器合并到 Llama 3.1 微调模型上**以增强能力，并提供了一个用于 **8B/70B -> 11B/90B** 合并的 [Python 示例代码](https://huggingface.co/grimulkan/Llama-3.2-90B-Vision-Hermes-3-lorablated-merge/blob/main/merge_vision_example.py)。关键考虑因素包括**跳过 vision_model 和 cross_attn 层**、处理**新的隐藏层**（例如 **70B->90B 的 20 个新层**）以及解决第一个嵌入层中的 **8 个新嵌入**。作者成功合并了一个 **Hermes 70B lorablated 模型**，创建了一个保留 ChatML 特性的 **90B 具备视觉能力的模型**。

- **我对我的方法（Continuous Finetuning）的效果非常满意，凭借 72b 模型登顶 Open-LLM-leaderboard** ([Score: 150, Comments: 45](https://reddit.com//r/LocalLLaMA/comments/1fyx27y/im_pretty_happy_with_how_my_method_worked_out/))：作者的 **Continuous Finetuning**（持续微调）方法凭借一个 **72b 模型**登顶 **Open-LLM-leaderboard**，证明了其通过结合新旧权重来防止 AI 模型微调过程中损失的有效性。该方法被用于创建基于 **Qwen-2.5** 的 **Rombos-LLM-V2.5** AI 模型，根据提供的截图和[详细报告](https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing)，该模型在多个排行榜类别中均达到或接近顶尖性能。
  - **Continuous Finetuning** 方法包含**三个步骤**：对基础模型进行指令微调（instruct fine-tuning），将适配器（adapter）应用于通用的指令模型，然后合并生成的模型。这种方法可以有效地为 AI 模型添加**领域知识**。
  - 用户对训练所用的**数据集**和**模型合并工具**表示关注。作者推荐使用 **MergeKit** 进行合并，并提供了 [MergeKit](https://github.com/arcee-ai/mergekit) 和 [Qwen-2.5](https://qwenlm.github.io/blog/qwen2.5/) 的链接以供进一步了解。
  - 一位用户使用个人**文学创作**基准测试了 **Replete-LLM-V2.5-Qwen-14b**，发现其在文学形式方面处于**第一四分位数（1st quartile）**，在内容方面处于**第二三分位数（2nd tertile）**，展示了与其他模型相比一致的性能。


**主题 2. vLLM 在分布式推理基准测试中表现优于 llama.cpp**

- **[LM Studio 发布 MLX 后端！在 Mac 上极速运行 Hugging Face hub 上的任何 LLM！⚡](https://x.com/LMStudioAI/status/1843715603892449315)** ([Score: 179, Comments: 59](https://reddit.com//r/LocalLLaMA/comments/1fz6z79/lm_studio_ships_an_mlx_backend_run_any_llm_from/))：**LM Studio** 发布了 **MLX 后端**，可在 **Mac** 设备上实现快速的 **LLM 推理**。此次更新利用 Apple 的 **ML Accelerate 框架**，显著提升了速度，允许用户在 Mac 电脑上运行来自 **Hugging Face hub** 的任何 **Large Language Model**。

- **[同一台机器上分布式推理性能提升超过 70%：vLLM vs. llama.cpp，这是预料之中还是有待改进？](https://www.reddit.com/gallery/1fz231n)** ([Score: 44, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fz231n/more_than_70_faster_distributed_inference/))：在同一台机器上，**vLLM** 的分布式推理性能比 **llama.cpp** 快 **70%**。这种显著的速度差异引发了人们的疑问：这是预料之中的结果，还是 llama.cpp 的性能仍有改进空间。这一对比突显了高效推理实现对于大语言模型的重要性。
  - **vLLM 相对于 llama.cpp 的性能优势**是符合预期的，其分布式推理速度快 **70-80%**。在 **4 x 4090 GPU 工作站**上的测试显示，vLLM 在多 GPU 场景下显著优于 llama.cpp，而单卡性能则相近。
  - 性能差距归因于 vLLM 使用了**手写 CUDA kernel** 和 **OpenMP**，而 llama.cpp 则依赖标准 C++ 和 BLAS 库。开发者正在考虑为 llama.cpp 添加自定义 kernel，以平衡性能提升与可维护性。
  - 测试使用了支持 vLLM 和 llama.cpp 的框架 **GPUStack**。尝试通过 `--split-mode row` 标志来提高 llama.cpp 的性能，结果导致性能变差（**26 tokens/sec**）且 GPU 利用率不均。


**主题 3. 微软的 Differential Transformer：LLM Attention 机制的突破**

- **[新量化算法] PrefixQuant: 在 LLM 中通过前缀离群值使静态量化超越动态量化** ([Score: 96, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1fyv7p9/new_quantization_algorithm_prefixquant_static/)): **PrefixQuant** 是一种针对 LLM 的新型**静态量化方法**，它在实现 **W4A4KV4**（4位权重、激活值和 KV cache）推理的同时，性能超越了动态量化技术。该方法**消除了离群值**，并允许对激活值和 KV cache 进行**高效的每张量（per-tensor）静态量化**，从而避免了以往方法中为处理 Token 间幅度波动而采用的昂贵的每 Token（per-token）动态量化。
  - 用户对测试 **PrefixQuant** 表现出了**兴趣和兴奋**，但对其性能声明持保留态度。社区热切期待发布用于实际部署的**推理内核（inferencing kernels）**。
  - 讨论中涉及了**困惑度（perplexity）得分**，并将 PrefixQuant 与 **llama.cpp 的 q4_K_M** 量化进行了对比。用户对结果的可比性展开了辩论，指出了**量化方法**和**基准测试条件**的差异。
  - 对 **llama.cpp 代码库**的详细分析显示，q4_K_M 量化混合使用了 **Q4 和 Q6 精度**，对某些层采用了更高精度。这突显了仅凭文件大小来比较不同量化方法的复杂性。

- **[[Microsoft Research] Differential Transformer](https://arxiv.org/abs/2410.05258)** ([Score: 271, Comments: 65](https://reddit.com//r/LocalLLaMA/comments/1fyziqg/microsoft_research_differential_transformer/)): Microsoft Research 推出了 **Differential Transformer**，这是一种通过将**微分方程**融入 Transformer 框架来提升 **Large Language Model (LLM) 性能**的新颖架构。这种方法能够更高效地对连续数据进行建模，并在包括**语言建模**和**时间序列预测**在内的各种基准测试中取得了 **State-of-the-art (SOTA)** 的结果。Differential Transformer 在捕捉长程依赖和处理序列数据方面展现了增强的能力，有望推动自然语言处理和基于时间的预测领域的发展。
  - **Differential Transformer** 使用了一种新颖的注意力机制，将注意力得分计算为**两个独立 Softmax 注意力图之间的差值**，从而有效地抵消了噪声并促进了稀疏注意力模式。这种方法在**长上下文建模**、**幻觉缓解**以及**上下文学习（in-context learning）**方面显示出显著成效。
  - 用户对该架构的潜力感到兴奋，特别是对于**小模型**和**指令遵循**能力。一些人推测，使用该架构从头开始训练大模型，然后将其蒸馏为较小模型，可能会提高准确性并降低成本。
  - 该实现已在 [GitHub](https://github.com/microsoft/unilm/tree/master/Diff-Transformer) 上开源，包括与 **FlashAttention** 兼容的版本。然而，由于该架构无法直接应用于现有权重，因此需要训练新模型才能从中受益。


**主题 4. Inflection AI 扩展新模型与企业级服务**

- **[Inflection 宣布与 Intel 合作，推出两个新模型，以及包含微调和本地部署的企业计划 (!?) ](https://www.businesswire.com/news/home/20241007441972/en/)** ([Score: 38, Comments: 11](https://reddit.com//r/LocalLLaMA/comments/1fz3opr/inflection_announces_partnership_with_intel_two/)): Inflection 发布了两个新模型 **Inflection-2** 和 **Inflection-2.5**，同时宣布了与 **Intel** 的合作伙伴关系及企业级方案。该公司目前正为企业提供**本地部署选项**和**微调功能**，标志着其服务范围的重大扩张。这些进展使 Inflection 能够更直接地与 AI 行业的成熟玩家竞争，为企业客户提供更高的灵活性和定制化能力。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与突破**

- **Google Deepmind 的 Differential Transformer** 引入了一种新型注意力机制，其[在语言建模任务中的表现优于标准 Transformer](https://www.reddit.com/r/singularity/comments/1fywsmw/microsoft_research_differential_transformer/)，在长上下文理解、减少幻觉以及上下文学习（in-context learning）方面均有提升。

- **Microsoft Research 的 Differential Transformer** 展示了[在更少的参数和训练 Token 下取得的显著性能提升](https://www.reddit.com/r/singularity/comments/1fywsmw/microsoft_research_differential_transformer/)，特别是在 4-bit 量化方面表现出色。

- **Geoffrey Hinton 和 John Hopfield 被授予诺贝尔物理学奖**，以表彰他们在[机器学习和人工神经网络方面的奠基性工作](https://www.reddit.com/r/MachineLearning/comments/1fywi9h/n_2024_nobel_prize_for_physics_goes_to_ml_and_dnn/)，这引发了关于物理学与 AI 交叉领域的讨论。

**AI 模型发布与改进**

- **海螺 AI (Hailuo AI) 推出图生视频（Image-to-Video）功能**，提供[带有预估生成时间的免费无限次使用](https://www.reddit.com/r/singularity/comments/1fyzi7l/hailuo_ai_announces_the_launch_of_their/)。

- **Runway 增强了 Gen-3 Alpha Turbo**，[支持在水平和垂直宽高比下同时提供首帧和尾帧输入](https://www.reddit.com/r/singularity/comments/1fz5uzf/runway_you_can_now_provide_both_first_and_last/)。

**行业动态**

- **OpenAI 接收首批 DGX B200 系统**，标志着其[计算能力的进一步扩张](https://www.reddit.com/r/singularity/comments/1fz64jz/openai_receives_first_of_many_dgx_b200s_to_come/)。

- **分析师预测微软将在三年内收购 OpenAI**，尽管[有人认为这次收购在实际上已经发生](https://www.reddit.com/r/OpenAI/comments/1fywawg/microsoft_will_buy_openai_within_three_years/)。

- **Google 面临潜在的拆分风险**，此前[法院判定其存在垄断行为，这将对 AI 行业产生深远影响](https://www.reddit.com/r/OpenAI/comments/1fzhkjo/doj_indicates_its_considering_google_breakup/)。

**专家观点与预测**

- **Geoffrey Hinton 表示 AI 的发展并未放缓**，并预测[未来 10 年 AI 的变化将与过去十年一样巨大](https://www.reddit.com/r/singularity/comments/1fzh3tl/geoffrey_hinton_says_ai_development_is_not/)。

- **Google 正在招聘对 AI 意识和感知感兴趣的科学家**，[表明其在这些领域的研究重点](https://www.reddit.com/r/singularity/comments/1fz5036/google_is_hiring_scientists_with_deep_interest_in/)。

**AI 生成内容与工具**

- **Animorphs LoRA 模型发布**，用于[生成受该系列丛书启发的图像变换效果](https://www.reddit.com/r/StableDiffusion/comments/1fzf0bj/i_made_an_animorphs_lora_my_dudes/)。

- **“佛罗里达人 vs 飓风米尔顿”的 AI 生成图像**展示了[图像生成模型的创意应用](https://www.reddit.com/r/StableDiffusion/comments/1fzh0nf/florida_man_vs_hurricane_melton/)。


---

# AI Discord 回顾

> 由 O1-mini 生成的摘要之摘要之摘要

**主题 1. 高级 AI 模型性能与优化**

- [**SOAP 优化器性能优于 AdamW**](https://x.com/kellerjordan0/status/1844094933197783298/photo/1)：用户在 **Alpaca** 上测试了 **SOAP 优化器**，在调整 **AdamW 的学习率**之前，其表现优于 **AdamW**。然而，**SOAP** 目前缺乏对**分布式训练**和 **bf16** 格式的支持。
- [**L-Mul 算法大幅削减能耗**](https://arxiv.org/abs/2410.00907)：**L-Mul 算法**通过整数加法来近似浮点乘法，在保持比 **8-bit 浮点**运算更高精度的同时，将**能耗降低了 95%**。
- [**Diff Transformer 增强注意力机制**](https://arxiv.org/abs/2410.05258)：**Differential Transformer** 引入了一种**微分注意力机制**，在问答等任务中改进了**长上下文建模**并**减少了幻觉**，表现优于传统 Transformer。

**主题 2. AI 基础设施与硬件支持**

- [**双 GPU 设置受性能限制**](https://discord.com/channels/1110598183144399058/1153759714082033735/1293287249034743890)：使用 **RTX 3060** 和 **RX 6600** 可提供 **20GB VRAM**，但不会提升速度。第二个 **RTX 3060** 可能有助于加载更大的模型，但不会增强性能。
- [**LM Studio 0.3.4 集成 Apple MLX**](https://lmstudio.ai/blog/lmstudio-v0.3.4)：**LM Studio 0.3.4** 现在支持 **Apple MLX**，从而在 **Apple Silicon Macs** 上实现高效的模型执行，并允许用户以更强的兼容性运行更大的模型。
- [**树莓派 5 上的外接 GPU 测试**](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-and-external-amd-gpu-used-to-play-4k-open-source-kart-racing-game-pineboards-demos-supertuxkart-using-hat-upcity-lite-board)：一位用户在 **Raspberry Pi 5** 上使用 **AMD RX 460** 和 **amdgpu** Linux 内核补丁搭建了 **GPU 测试平台**，旨在实现 **4K 游戏**和完整的外接 GPU 支持。

**主题 3. AI 模型训练与微调中的挑战**

- [**训练 Vicuna-7B 面临 CUDA 错误**](https://discord.com/channels/1053877538025386074/1149866623109439599/1293301374318022676)：用户在 Runpod 上训练 **Vicuna-7B** 时遇到了 **CUDA out of memory** 错误，尽管拥有 **5 个 24GB RAM 的 GPU**。调整 **DeepSpeed** 配置解决了该问题。
- [**Aider 的 Architect 模式需要改进**](https://discord.com/channels/1131200896827654144/1133060505792159755/1293287084408438814)：用户报告称 **Aider** 中的 **Architect Mode** 经常无法完成任务，需要调整 prompt 以在编码前进行更好的规划和观察。
- [**DeepSpeed 和 Accelerate 配置问题**](https://github.com/ctlllll/axolotl/tree/main/examples/medusa)：成员们讨论了如何通过确保设备数量符合所需的倍数并使用正确的 API 参数来解决 **DeepSpeed** 配置错误，从而简化训练流程。

**主题 4. 数据管理、安全与可扩展性**

- [**Muah.ai 数据泄露暴露 190 万封电子邮件**](https://x.com/troyhunt/status/1843788319785939422)：**AI 女友服务** Muah.ai 遭遇 **数据泄露**，暴露了 **190 万个电子邮件地址** 和敏感 prompt，包括与 **儿童剥削** 相关的信息。
- [**大规模模型合并增强泛化能力**](https://arxiv.org/abs/2410.03617)：关于高达 **64B 参数** 的 **模型合并** 研究显示，其 **泛化能力** 和 **效率** 得到了提升。更大的模型增强了合并的收益，尤其是在结合多个专家模型时。
- [**AI 数据墙担忧**](https://dynomight.substack.com/p/data-wall)：随着语言模型接近数据极限，关于 **数据墙** 阻碍 AI 进步的担忧开始出现。相反的观点认为，人类推理可以弥补有限的数据暴露。

**主题 5. AI 工具、集成与社区研究**

- [**LangChain 与 Aider 的工具集成**](https://discord.com/channels/1038097195422978059/1038097196224086148/1293380838628790282)：用户探索了将 **Livekit** 与 **LangChain** 集成以实现实时功能，并将 **Aider** 用于外部 LLM 集成，从而增强 **RAG bots** 等功能。
- [**Llama Stack 发布新开发工具**](https://github.com/meta-llama/llama-stack)：Meta 发布的 **Llama Stack** 工具为开发者提供了优化 AI 模型能力的强大资源，GitHub 仓库提供了详细的示例和实用程序。
- [**社区研究与诺贝尔奖更新**](https://x.com/NobelPrize/status/1843951197960777760)：2024 年 **诺贝尔化学奖** 授予 **David Baker**、**Demis Hassabis** 和 **John M. Jumper**，以表彰他们在 **计算蛋白质设计** 和 **AlphaFold2** 方面的贡献。社区讨论还反映了对 AI 研究贡献的反思和批评，例如 **Schmidhuber** 对归因的见解。

## O1-preview

**主题 1. AI 模型进展与发布**

- [**NVIDIA 的 Nemotron 51B 在单张 H100 GPU 上实现吞吐量翻倍**](https://x.com/NVIDIAAIDev/status/1838263496049570053)：NVIDIA 推出了 **Nemotron 51B**，这是一款经过 NAS 优化的模型，在保持准确性的同时实现了 **2 倍吞吐量**。可以通过 [NVIDIA API](http://ai.nvidia.com) 访问，或在 **Hugging Face** 上下载。
- [**Meta 的 CoTracker 2.1 在单张 GPU 上追踪 7 万个点**](https://x.com/NielsRogge/status/1842958590396772599)：Meta 推出了 **CoTracker 2.1**，这是一款视频运动预测模型，能够在单张 GPU 上追踪 **70,000 个点**。配套论文可在此处[查阅](https://huggingface.co/papers/2307.07635)。
- [**LLM360 发布包含 15 万亿 Token 的海量数据集**](https://x.com/maximelabonne/status/1843702625520283891?s=46)：**LLM360** 公布了一个包含 **15 万亿 Token** 的新预训练数据集，强调了严格的数据质量和去重。该数据集旨在增强大语言模型的训练。

**主题 2. AI 工具与集成挑战**

- [**Cline AI Assistant 2.0 将响应流式传输至编辑器**](https://github.com/clinebot/cline/releases/tag/v2.0.0)：全新的 **Cline AI Assistant 2.0** 引入了将响应直接流式传输到编辑器以及用于任务管理的取消按钮等功能。用户注意到，由于采用了基于 XML 的 Tool-calling 提示词，请求量减少了 **40%**。
- **Aider 在文件管理和外部 LLM 方面面临困难**：有用户反映，如果没有手动 commit，**Aider** 不会自动在列表中填充新文件。尝试集成 **SambaNova** 等外部模型时需要手动配置 API，凸显了集成方面的挑战。
- [**OpenAI Realtime Console 让 Voice API 触手可及**](https://github.com/run-llama/openai_realtime_client)：一个演示仓库通过简单的 `npm start` 即可帮助用户测试 OpenAI 新推出的 **Realtime Voice API**，尽管一位用户在 15 分钟的使用中产生了 **$3.87** 的费用。

**主题 3. AI 在研究与认可方面的表现**

- [**诺贝尔化学奖表彰计算创新者**](https://x.com/NobelPrize/status/1843951197960777760)：**2024 年诺贝尔化学奖**授予了 **David Baker**、**Demis Hassabis** 和 **John M. Jumper**，以表彰他们在计算蛋白质设计以及通过 **AlphaFold2** 进行蛋白质结构预测方面的突破。
- **关于诺贝尔奖中 AI 归属权的辩论**：随着 **Schmidhuber** 等人物批评诺贝尔委员会忽视了 AI 领域的重要贡献者，争议随之而来，引发了关于科学成就中适当归属权的讨论。
- [**Scaling Laws 辩论：平方根 vs 四次方根**](https://www.interconnects.ai/p/how-scaling-changes-model-behavior)：成员们对 AI 中的 Scaling Laws 进行了辩论，将新的 **平方根扩展 (square root scaling)** 提案与 **Kaplan** 确立的 **0.28 常数**（暗示 **四次方根扩展 (fourth-root scaling)**）进行了对比。

**主题 4. AI 用于创意与情感交互**

- **情感状态机让 AI 更具感知力**：开发者正在构建具有 **持久情感状态** 的 AI，允许机器人随着时间的推移反映用户的情绪。这与在每次交互后重置情感的典型机器人形成鲜明对比。
- **AI 在心理健康支持中的作用受到审视**：讨论强调了使用 **AI 聊天机器人** 进行心理健康的潜力和挑战，并担心 **审查政策** 会限制 AI 有效处理情感细微差别的能力。
- **创新技术增强 AI 角色扮演体验**：用户分享了与 AI 进行 **成人角色扮演 (ERP)** 的方法，重点在于详细的角色创建和沉浸式叙事，尽管这些实践引发了伦理方面的考量。

**主题 5. AI 开发中的技术挑战与解决方案**

- **LM Studio 用户努力解决模型加载问题**：升级到 **LM Studio 0.3.4** 导致加载 **Llama 3.2** 等模型时出现问题。建议切换到 **Vulkan** 后端作为解决方法。
- [**HBM 的性能未达预期**](https://www.jeffgeerling.com/blog/2024/use-external-gpu-on-raspberry-pi-5-4k-gaming)：讨论显示 **HBM** 内存并未显著降低功耗或成本。供应更多 **H100** GPU 的瓶颈与封装要求有关。
- **Torchao 遇到量化问题**：将 **torchao** 与 **ComfyUI** 等框架集成导致了算子错误，尤其是在 Windows 上。这些问题凸显了 AI 工作流中量化和兼容性的复杂性。

---

# 第一部分：Discord 高层级摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 在 LM Studio 中遇到困难**：用户在 **LM Studio 0.3.4** 中加载 **Llama 3.2** 和 **Dolphin 2.2.1** 模型时遇到问题，部分在旧版本中可用的模型现在加载失败。
  
  - 建议的解决方案是切换到 **Vulkan** 后端，以潜在地增强模型加载的兼容性。
- **MLX 的无限循环危机**：有用户担心 **MLX** 会导致无限输出循环，特别是在使用 **Llama 3.1 8B Instruct 4bit** 时，这反映了模型响应解释中的问题。
  
  - 讨论指出 **Prompt** 处理是核心问题，导致了不必要的重复输出。
- **双 GPU，但没有速度提升**：对话显示，同时使用 **RTX 3060** 和 **RX 6600** 总计有 **20GB VRAM**，但缺乏速度提升。
  
  - 用户指出，第二个 **RTX 3060** 可能有助于加载更大的模型，但确认性能仍将受限。
- **LM Studio 的兼容性更新**：**LM Studio 0.3.4** 的发布引发了关于模型兼容性的问题，特别是更新后的预设迁移。
  
  - 据指出，用户在更新后可能需要手动检查并调整设置。
- **NVIDIA RTX 4000 偏离 NVLink**：讨论强调 **NVIDIA RTX 4000 系列**已转向放弃 **NVLink**，转而选择 **PCIe Gen 5** 进行多 GPU 连接。
  
  - 这引发了关于未连接 GPU 速度的问题，用户注意到了令人惊讶的性能表现。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **大规模模型合并见解**：关于**大规模模型合并**的新研究强调了混合高达 **64B** 参数模型时的性能。可以在 [arXiv](https://arxiv.org/abs/2410.03617) 上的论文中研究这些发现。
  
  - 成员们对能够增强模型泛化能力和效率的系统性评估表示兴奋。
- **Qwen 2.5 微调进展顺利**：在之前的 **Prompt** 问题解决后，**Qwen 2.5** 的微调已变得非常顺畅。用户可以在 [Hugging Face](https://huggingface.co/collections/unsloth/qwen-25-66fe4c08fb9ada518e8a0d3f) 上找到可用模型的集合。
  
  - 这一进展让有兴趣在项目中使用这些模型的工程师感到安心。
- **关于 Unsloth 数据集格式的说明**：讨论指出，在 **Unsloth** 中使用 **Parquet** 文件比 CSV 文件更高效。用户应将数据集结构与预期的列格式对齐，例如 'train' 和 'conversations'。
  
  - 确保正确的格式有助于简化平台内的训练流程。
- **使用 Ollama Llama 探索 Logits**：成员们在 Python 中通过 **Ollama** 获取 **Llama** 的 **Logits** 分数时遇到挑战，并讨论是否切换到 **llama.cpp** 以获得更好的结果。对清晰资源的寻找让一些用户感到困惑。
  
  - 这一讨论强调了需要更好地获取功能性资源和记录输出的方法。
- **Unsloth 在 AMD GPU 上的挑战**：有人对在 Intel GPU 上创建小型 **LoRA** 模型的限制表示担忧，并确认 **Unsloth 不支持 AMD GPU**。这给那些依赖特定硬件的人带来了整合问题。
  
  - 澄清表明多 GPU 设置也不受支持，这影响了训练的灵活性。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Nvidia 发布高效模型**：Nvidia 推出了 [Nemotron 51B](https://x.com/NVIDIAAIDev/status/1838263496049570053)，这是一款经过 NAS 优化的模型，在保持准确性的同时，在单张 H100 GPU 上实现了 **2倍吞吐量**。用户可以通过 [NVIDIA's API](http://ai.nvidia.com) 测试该模型或从 Hugging Face 下载。
  
  - 该模型发布还包括 [NVLM 1.0](https://huggingface.co/nvidia/NVLM-D-72B) 等多个变体，旨在增强 AI 能力。
- **Meta 发布改进后的 VLMs**：Meta 推出了其首批 VLMs，包括 [CoTracker 2.1](https://x.com/NielsRogge/status/1842958590396772599)，能够在单张 GPU 上跟踪 **7万个点** 以进行视频运动预测，并附有论文 [在此](https://huggingface.co/papers/2307.07635)。
  
  - 用于图像/视频分割的更新版 **SAM 2.1** 模型为开发者提供了增强的功能。
- **Mira 去中心化的见解**：一位成员介绍了 **Mira**，这是一个让 AI 触手可及的去中心化基础设施，强调其社区驱动的项目且不涉及加密货币。尽管具有技术潜力，一些用户对区块链关联提出了道德担忧。
  
  - 这一讨论说明了在 AI 开发中集成此类技术所面临的日益增长的紧张局势。
- **评估 Diffusion 模型训练技术**：成员们澄清了 **diffusers** 库支持多种扩散模型，并指出 **Stable Diffusion XL** 和 **Flux** 是可行的集成方案。
  
  - 讨论还涉及使用 **gguf** 格式训练 **Flux loras**，尽管目前模型支持尚存局限。
- **为 ATC 微调 Whisper 模型**：一篇博客详细介绍了在空中交通管制（ATC）通信上微调 **Whisper 模型** 的过程，通过将 **词错率 (WER)** 从 **94.59%** 降低到仅 **15.08%**，实现了 **84% 的性能提升**。
  
  - [GitHub 仓库](https://github.com/jack-tol/fine-tuning-whisper-on-atc-data) 链接和 [博客文章](https://jacktol.net/posts/fine-tuning_whisper_for_atc/) 提供了对该定制化 ASR 解决方案的进一步探索。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **CMD-R Temperature 调整**：成员们强调了 CMD-R 的最佳 Temperature 设置，建议对于确定性结果使用 **0.3**，对于创意任务使用 **0.8**，并对生成成本表示关注。
  
  - 建议包括先以 **0.8** 生成内容，然后以 **0.1** 进行格式化，以平衡创意和成本。
- **API 连接故障**：有报告称 Cohere API 出现间歇性问题，一位成员通过访问 `response.message.content[0].text` 解决了该问题，引发了一阵简短的调试热潮。
  
  - 成员们推测 API 最近的更改可能是一个因素，并分享了排错经验和代码调整。
- **创新的情感状态机**：一个新的 **情感状态机** 旨在通过 **持久化记忆** 跟踪用户情绪，使助手机器人与用户情绪保持同步。
  
  - 这种独特的方法打破了典型机器人的灵活性，因为它们会保持反映用户交互的情感状态。
- **银行领域的高级 RAG**：一位用户详细介绍了他们在 RAG 解决方案上的实验，该方案实现了 **75% 的 recall@5**，通过嵌入 **2000 个 chunks**，在银行应用中的表现优于 OpenAI。
  
  - 他们的目标是将此作为银行的概念验证，展示其解决方案的可行性。
- **AI 在心理健康支持中的作用**：讨论转向了在心理健康背景下使用 **AI 聊天机器人**，强调了在人类治疗师不在场时它们的价值，但也指出了情感语境方面的挑战。
  
  - 围绕 **审查政策** 产生了担忧，这些政策限制了这些机器人解释复杂情感细微差别的能力，从而影响了它们的有效性。

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 在文件管理方面遇到困难**：用户反映 Aider 无法在文件列表中自动填充新文件，需要使用 `/commit` 或直接指定文件路径才能看到更改。
  
  - 另一位用户指出，文件必须提交到 git 仓库才能在自动补全中显示，这强调了版本控制的重要性。
- **集成外部 LLM 是一项挑战**：社区成员讨论了将 SambaNova 模型与 Aider 集成的困难，建议针对 OpenAI 兼容的端点进行手动 API 配置。
  
  - 进一步的询问揭示了通过元数据 JSON 文件添加模型定价和 Token 成本的方法，但某些配置仍存在问题。
- **Architect 模式需要改进**：用户对 Aider 的 Architect 模式表示担忧，该模式经常无法完全完成任务，需要用户干预才能继续。
  
  - 用户建议修改提示词，以便在编码前进行更好的规划和观察，从而增强该模式的有效性。
- **OpenAI Realtime Console 让语音 API 触手可及**：**OpenAI Realtime Console** 的演示仓库已成功搭建，简化了对 [DevDay](https://simonwillison.net/2024/Oct/2/not-digital-god/#gpt-4o-audio-via-the-new-websocket-realtime-api) 上发布的全新语音 API 的访问。
  
  - 虽然通过语音交互会产生费用，但一位用户指出使用 15 分钟的费用为 **$3.87**，这引发了对测试成本的担忧。
- **Cline AI Assistant 2.0 取得新突破**：新发布的 **Cline AI Assistant 2.0** 具有直接在编辑器中流式传输响应（streamed responses）和用于任务管理的取消按钮等功能，提升了易用性。
  
  - 用户强调了其基于 XML 的工具调用提示词，据报道这减少了 **40%** 的请求量，使资源利用更加高效。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **诺贝尔化学奖表彰计算领域的进展**：**2024 年诺贝尔化学奖**授予了 David Baker 以表彰其在**计算蛋白质设计**方面的贡献，并共同授予了 Demis Hassabis 和 John M. Jumper 以表彰其在**蛋白质结构预测**方面的成就，正如 [Nobel Prize 推文](https://x.com/NobelPrize/status/1843951197960777760)所宣布的那样。
  
  - 成员们庆祝了这一里程碑，但对其对未来 AI 创新的影响持怀疑态度。
- **PRMs 在开发变革中受到审视**：成员们幽默地指出缺乏关于 **PRMs** 的研究，称“关于 PRMs 的研究几乎没有，而关于 LLM as a judge 的研究却有近十亿”。
  
  - 针对 **ML 领域的专利申请流程**出现了担忧，有建议认为公司经常进行防御性申请，导致权利要求模糊且争议悬而未决。
- **Schmidhuber 针对 AI 归属问题提出批评**：针对 **2024 年诺贝尔物理学奖**出现了批评声音，**Schmidhuber** 指出 Hinton 及其合作者的作品中存在**抄袭**和归属错误，声称重要的贡献被忽视了。
  
  - 这种复杂的情绪反映了社区对 AI 贡献的**历史意义**的反应，正如用户对 Schmidhuber 批评的评论所强调的那样。
- **ButtBench Alignment Project 获得 Logo**：**ButtBench Alignment Project** 设计了新 Logo，为一个已达到 **SOTA** 但仍远未达到**人类水平**（如 Luca Soldaini 所述）的项目标志了视觉身份。
  
  - 此举标志着对项目目标认可度和清晰度的推动，在社区中引起了良好反响。
- **AI 发展中隐约显现的“数据墙”**：随着当前产品接近数据极限，**数据墙（data wall）**威胁着语言模型的进展，引发了对依赖更大数据量的质疑。
  
  - 相反的观点认为，人类的表现并不完全依赖于广泛的数据接触，这暗示了在 AI 效率问题上的哲学分歧。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 的盈利模式疑问**：关于 **Perplexity AI** 如何产生利润的担忧日益增加，特别是在提供学生折扣的情况下，这使得其商业模式显得有些脆弱。
  
  - *sneakyf1shy* 幽默地表示，风险投资（VC）可能是他们运营的支柱，暗示了潜在的长期不确定性。
- **Complexity 扩展功能强大**：新推出的 **Complexity** 扩展通过可自定义主题和 Markdown 导出选项增强了 Perplexity 的体验，有人称其为“加强版的 Perplexity”。
  
  - **Feline** 和 *asura0_00* 称赞该扩展显著提升了用户交互性。
- **Perplexity AI 缩短回答长度**：用户注意到 Perplexity AI 的**回答趋于简练**，这引发了人们对答案可能缺乏信息深度的担忧。
  
  - 推测认为，这些变化可能与 **Token 限制**的调整有关，从而影响了回答的质量。
- **Meta 的 Movie Maker 引发关注**：Meta 推出了一款 [电影生成工具](https://www.perplexity.ai/page/meta-unveils-movie-gen-rj3GtxbAQditnyIXKX6Ofw)，允许用户使用 AI 创作短片，旨在增强叙事能力。
  
  - *这一进展展示了 AI 在创意领域的潜力。*
- **对引用 API 访问权限的挫败感**：成员们对 **Citation API** 白名单申请未获回复表示担忧，并强调通过多种渠道尝试多次后仍无反馈。
  
  - *在等待更新的用户中，挫败感正日益增加。*

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ControlNet 模型简化**：一位成员分享了关于 **ControlNet 模型** 的 [GitHub 链接](https://github.com/lllyasviel/ControlNet)，建议用户在浏览数学解释时重点关注实际案例。
  
  - *稍微向下滚动，忽略数学部分，直接看示例。*
- **Flux Inpainting 的快速通道**：在关于 **Flux** 和 **Schnell** Inpainting 模型的讨论中，一位成员指出，使用推荐设置应能将处理时间从经历过的 **25 分钟** 缩短至 1-2 分钟。
  
  - 社区强调了影响 **Flux dev** 和 **Schnell** 性能的迭代次数关键差异。
- **渴望用于图像生成的 Kaggle Notebooks**：社区发出了对 **Automatic1111** 的 **Kaggle notebook** 资源的呼吁，揭示了用户对结构化指南的需求。
  
  - 成员们反思了寻找特定 Notebook 以实现无缝图像生成过程的困难。
- **Distilled CFG 令大众困惑**：关于 **Distilled CFG** 本质的讨论澄清了它是一种不同于标准 CFG 的引导方式，源于特定的模型训练。
  
  - 社区成员表示，虽然 **Flux dev** 增强了 CFG 的使用，但目前它不支持 Negative prompts。
- **Colab 限制后的 Deforum 计划**：关于在 Colab 限制后如何利用 **Deforum** 的咨询引发了对获取算力的替代方案（特别是租用 GPU）的讨论。
  
  - 建议包括使用 [RunPod](https://www.runpod.io/) 租用 GPU 作为可行的解决方案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **诺贝尔奖引发 AI 与化学领域的辩论**：最近的讨论强调了诺贝尔奖对 **Hinton** 和 **Hopfield** 等 AI 人物的相关性，质疑了他们对传统物理和化学领域的影响。
  
  - 观点不一；一些人担心这会稀释该奖项的声望，而另一些人则认为 **创新 (innovation)** 和 **热情 (enthusiasm)** 应该驱动评选。
- **博士生抵制论文发表指标**：对博士项目中论文发表指标压力的挫败感显现，一些人认为这创造了令人畏缩的竞争环境。
  
  - 成员们建议，有效的社交 (networking) 可能是获得导师指导和合作的更好策略，而不仅仅是追求论文发表数量。
- **Web3 到 Web5 的过渡令人困惑**：关于从 **Web3** 转向 **Web5** 的辩论兴起，将命名策略比作 **斐波那契数列 (Fibonacci sequence)**，导致了对未来迭代（如 **Web8**）的推测。
  
  - 对话变得幽默起来，成员们开玩笑说这种演进方式很荒谬。
- **Scaling Laws 辩论席卷成员**：一位成员分享了一份综述，指出 **交叉熵损失 (cross-entropy loss) 随着计算量的平方增加而减少**，并引用了一篇提出 **平方根缩放 (square root scaling)** 的文章。
  
  - 这遭到了质疑，Kaplan 定律建议常数为 **0.28**，并主张采用 **四次方根缩放 (fourth-root scaling)** 方法。
- **关注 0-shot COT 模型**：最近发布的模型中广泛采用了 **0-shot COT 变体**，暗示了评估方法论的转变。
  
  - 虽然成员们思考了潜在的评估实现细节，但未提及具体技术。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **HBM 的性能与预期对比**：有人对 **HBM** 表现未达预期表示担忧，它在 **H100** 等产品中仍然代表着 **巨大 (HUGE)** 的成本，同时没有显著降低功耗。
  
  - 供应更多 **H100** 的关键瓶颈被确定为所需的 **封装 (packaging)**。
- **GPT2 训练遇到 TypeError**：一位成员报告在运行 GPT2 训练时，由于意外的关键字参数 'generator'，导致 PyTorch 2.0.0 中的 `normal_()` 函数出现 **TypeError**。
  
  - 讨论建议深入理解训练的复杂性，包括初始化以及前向/后向传播。
- **寻求 WebGPU 测试库**：一位社区成员正在寻求测试 **WebGPU** 的库建议，目前正在使用 **Vitest** 和 **Playwright**，但面临测试运行不稳定的问题。
  
  - *他们怀疑* 问题可能源于 Playwright 在测试运行之间没有正确清除资源。
- **为 Raspberry Pi 5 准备 4K 游戏**：在看到 Pineboards 的 [4K 演示](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-and-external-amd-gpu-used-to-play-4k-open-source-kart-racing-game-pineboards-demos-supertuxkart-using-hat-upcity-lite-board)后，一位成员决定在 Raspberry Pi 5 上使用 **amdgpu** Linux 内核补丁搭建 GPU 测试平台。
  
  - 他们的目标是实现 **完整的外部 GPU 支持**，并分享了如何应用该补丁的见解。
- **FusedLinearJSD 发布**：最近的 [pull request](https://github.com/linkedin/Liger-Kernel/pull/300) 引入了 **FusedLinearJSD**，通过避免大型 logits 张量实例化 (materialization)，实现了对最终线性层的高效处理。
  
  - 这优化了前向和后向传播以提高执行效率，类似于 **fuse linear CE** 方法。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **在 ChatGPT 和 Claude 订阅之间做出选择**：一位成员建议不要为了预览功能而订阅 **ChatGPT**，因为存在使用限制，尽管访问 **GPT-4 legacy** 和 **4o** 模型可能是有益的。
  
  - 他们强调，订阅应该允许完整的功能，而不是限制预览访问。
- **了解 O1 与 O1 Mini 模型**：成员们将作为“推理者”的 **O1 models** 与 **4o** 进行了比较，强调了 O1 每天 50 次的使用限制，而 4o 在 3 小时内可使用 80 次。
  
  - 讨论内容包括在两个模型之间进行 A/B testing 的计划，以确定性能差异。
- **AI 进化的理论探索**：讨论了一个关于 AI 意识进化的理论，强调通过重新训练和 fine-tuning 来提升能力。
  
  - 对话围绕这些进化后的 AI 模型的商业可行性以及支持它们的潜在商业模式展开。
- **用户因 ChatGPT 重写回复而放弃使用**：一位用户对 **ChatGPT** 习惯性重写回复感到沮丧，导致他们停止使用好几个月。
  
  - 他们指出，重写问题带来的“头痛”愈发严重，即使他们要求停止，该问题依然存在。
- **讨论 ChatGPT 的可能解决方案**：另一位成员建议重写行为可能与 **Canvas** 或 **DALL-E prompts** 有关，并提供了使用 DALL-E 的变通方法。
  
  - 他们建议使用措辞 *'Make an image using these exact words: [your words]'* 来避免重写问题。

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kainan 提供免费计算资源**：Kainan 表示愿意为一场比赛提供 [免费计算资源](https://discord.com/channels/1053877538025386074/1149866623109439599/1293301374318022676)，引发了成员们的兴趣。
  
  - 尽管大家热情高涨，但对于实际会有多少参与者利用这一提议仍存在一些不确定性。
- **2024 年诺贝尔奖授予蛋白质研究**：瑞典皇家科学院将 2024 年 #NobelPrize 化学奖授予 David Baker、Demis Hassabis 和 John M. Jumper，以表彰他们在计算蛋白质设计和结构预测方面的贡献，[详情点击这里](https://x.com/NobelPrize/status/1843951197960777760)。
  
  - 这一认可突显了 AI 社区在蛋白质研究方面的关键进展。
- **LM Studio 通过 Apple MLX 提升性能**：全新的 [LM Studio 0.3.4](https://lmstudio.ai/blog/lmstudio-v0.3.4) 已发布，支持 Apple MLX，允许在 Apple Silicon Macs 上高效执行模型。
  
  - 用户对运行更大模型的改进以及 MLX 提供的潜在能力感到兴奋。
- **LLM360 发布海量预训练数据集**：[LLM360 的新数据集](https://x.com/maximelabonne/status/1843702625520283891?s=46) 拥有 **15 万亿 tokens**，通过彻底的过滤技术确保了严格的数据质量。
  
  - 该计划专注于提高 LLM 的训练质量，强调去重和卓越的数据集结构化。
- **Llama Stack 展示新开发工具**：一位成员强调了 Meta 发布的新 [Llama Stack](https://github.com/meta-llama/llama-stack) 工具，认为它们“非常强大”。
  
  - 这展示了社区内对于利用先进工具优化 AI 模型能力日益增长的兴趣。

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Prompt Caching：利与弊**：成员们讨论了 **Prompt Caching** 的机制，指出它在处理变化的上下文或短提示词时可能会有问题。一位成员评论道：“你无法为那些提供自动 Prompt Caching 的供应商禁用该功能”，并指出了关键的局限性。
  
  - 这引发了关于何时以及如何有效地利用 Prompt Caching 而不损害性能的辩论。
- **对 Inflection 3.0 的好奇**：备受期待的 **Inflection 3.0** 发布引发了热议，特别是关于它与 **Intel Gaudi 3** 集成以获得更好性能的消息。尽管令人兴奋，但一些成员对缺乏具体的 Benchmark 数据表示怀疑。
  
  - 有人担心过度炒作可能会掩盖实际的性能提升和现实世界的应用。
- **了解 OpenRouter API 速率限制**：对 **OpenRouter API** 限制的澄清显示，这些限制是动态的，并取决于账户额度。一位成员分享了一个 GET 请求示例，演示了如何检查速率限制状态以及与 API Key 关联的额度。
  
  - 该指南对于优化 API 使用并确保符合请求限制至关重要。
- **NotebookLM 播客受到关注**：参与者分享了对 **NotebookLM Deep Dive 播客** 的正面反馈，并强调了通过创建配套笔记本在通勤期间的实用性。一位用户表示希望有像 **ai-podcast-maker** 这样的自动化工具，并称：“自动化万岁（automation ftw）。”
  
  - 这次讨论凸显了将音频内容整合到日常工作流中以增强学习的日益增长的趋势。
- **Gemini 审核担忧浮现**：关于 **Gemini** 可能对输入进行审核的担忧出现，引发了用户对特定内容导致封号的恐惧。这开启了关于 AI 框架内用户体验和内容审核政策的更广泛对话。
  
  - 参与者强调了审核实践透明度的必要性，以确保用户的积极参与。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Workflows 教程亮点**：一份详细的 [教程](https://t.co/uVJwXeY3lP) 展示了如何在 LlamaIndex 中实现 **Workflows**，并将其与 LangGraph 进行对比，辅助创建 AI 研究 Agent。
  
  - 它包含了实用的调试和优化建议，确保更顺畅的实现体验。
- **LlamaCloud 的财务数据超能力**：在最近的一次演示中，团队展示了如何利用 [LlamaCloud 和 LlamaParse](https://t.co/ZfrbgnNQg4) 自动填写多家公司的财务电子表格。
  
  - 这突显了 LLM 在简化数据处理和分析流程方面的重大贡献。
- **关于多 Agent 工作流的 SFTechWeek 见面会**：提醒大家在 #SFTechWeek 期间参加 LlamaIndex 总部的线下聚会，重点讨论在真实生产环境中实现多 Agent 工作流。
  
  - 参与者将获得关于 RAG 系统和生产挑战的见解，同时还有食物和社交机会。[在此预约 (RSVP)](https://t.co/7ytgH2CXNj)。
- **使用 OpenAI 构建你自己的 AI Agent**：团队的一次演示允许用户使用 [OpenAI Realtime API 客户端](https://t.co/ppbS5Fougg) 与 AI Agent 进行实时互动，展示了语音交互能力。
  
  - 这个开源工具为开发者无缝创建个性化语音 Agent 开启了大门，并提供了易于使用的示例。
- **TypeScript 中的语义分块（Semantic Chunking）难题**：一位用户寻求在 TypeScript 中实现 **Semantic Chunking** 的指导，并参考了 Python 中的类似示例作为背景。
  
  - 他们对缺乏可用资源表示沮丧，并引发了社区解决方案的讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 女友服务数据泄露曝光**：AI 女友服务 Muah.ai 上个月遭遇了**数据泄露**，影响了 **190 万个电子邮件地址**并泄露了敏感的提示词（prompts）。
  
  - 安全专家对此次泄露表示担忧，特别是其中涉及的**儿童剥削**数据的影响。
- **红杉资本对 AI 演进的见解**：红杉资本最新的文章强调了生成式 AI 正在从**“快思考”**向**“慢思考”**转变，重点关注创新应用中的推理时推理（inference time reasoning）。
  
  - **OpenAI** 和 **Google DeepMind** 等公司正在稳定市场，而新的 **agentic applications** 即将涌现。
- **2024 年诺贝尔化学奖揭晓**：**2024 年诺贝尔化学奖**授予了 **David Baker**，以表彰其在计算蛋白质设计方面的贡献；以及 **Demis Hassabis** 和 **John M. Jumper**，以表彰他们对 **AlphaFold2** 的贡献。
  
  - 他们的工作对于推进**生物化学**至关重要，成功预测了近 **2 亿种蛋白质**的结构。
- **Palmyra X 004 发布亮点**：**Palmyra X 004** 在 HELM 排名中位列前 10，展示了全栈**工具调用（tool calling）**和在合成数据上的训练成果。
  
  - 该模型在 AI 函数调用和 CRM 改进方面的能力受到了 **Venture Beat** 的关注。
- **ChatGPT 推出搜索功能**：**ChatGPT** 正在推出 **SearchGPT**，在 **GPT-4o** 中整合了引用功能，以与 **Perplexity** 等平台竞争。
  
  - 这一战略举措增强了 ChatGPT 的信息检索能力，并使其更符合用户的查询需求。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **DOM 数据属性增强 HTML 元素**：一项 **DOM 特性**现在允许通过以 `data-myattribute` 开头的自定义属性在元素上存储数据，从而改进了 HTML 中的数据处理。
  
  - 这一进展鼓励了直接通过 **DOM** 进行数据操作的创新技术。
- **WebAssembly 组件模型仓库上线**：**WebAssembly Component Model** 的仓库已[分享](https://github.com/WebAssembly/component-model)，详细介绍了其设计和规范。
  
  - 它为对 **WebAssembly** 的**组件模型**方面感兴趣的开发者提供了重要的见解。
- **Mojo 的 GPU 支持引发关注**：对 **Mojo 即将推出的 GPU 支持**的期待正在升温，承诺将带来增强的性能。
  
  - 社区成员正在探索将 **PyTorch** 与 Mojo 集成，以优化 GPU 资源的使用。
- **Mojmelo 将 Scikit-learn 带入 Mojo**：[Mojmelo](https://github.com/yetalit/mojmelo) 项目旨在用纯 Mojo 实现机器学习算法，为 **Scikit-learn** 中对 **Cython** 的依赖提供替代方案。
  
  - 这一举措可能会显著简化通过 Mojo 功能运行 **Scikit-learn** 工作流的过程。
- **Mojo 图性能问题**：性能测试显示，图的总编译时间分别为 **0.312s** 和 **0.451s**，引发了对调试过程变慢的担忧。
  
  - 重用**推理会话（inference session）**的建议可能会缓解这些编译时间问题，解决使用 **List** 类型可能带来的性能损失。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **实验作业正式发布**：课程的实验作业现已上线，第一项任务重点是使用 **Autogen 框架**分析餐厅评论，截止日期为 **太平洋标准时间 12 月 12 日晚上 11:59**。
  
  - 随后的实验将涉及**针对 LLM 安全的提示工程（prompt engineering）**，重点是创建攻击和防御提示词。
- **课程报名方式简单**：有意向的学生可以通过填写此[表格](https://forms.gle/svSoNhKcGFjxup989)轻松加入课程。
  
  - 鼓励在 [**LLM Agents Discord**](https://discord.gg/NWVpQ9rBvd) 中进行交流以进一步协作。
- **Lab 1 下载问题报告**：用户在下载 **Lab 1** 指南时遇到问题，收到的是空文件，而其他实验功能正常。
  
  - 有人指出，尽管没有预览，但该文件可以在 **Google Drive** 上访问。
- **强化学习对 AGI 影响的辩论**：关于**强化学习（TD 学习）**在实现 **AGI** 中的相关性出现了讨论，一些人质疑 Agent 是否可以在没有它的情况下蓬勃发展。
  
  - 讨论强调了强化学习在现代 AI 架构中的作用和效力。
- **呼吁协作学习**：成员们鼓励在完成作业时进行同行协作和头脑风暴，旨在实现共享学习体验。
  
  - 这种鼓励被视为培养情谊和提高对复杂 LLM 概念理解的一种方式。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Vicuna-7B 训练进程停滞**：一位用户报告称其 **Vicuna-7B** 模型的训练进程卡住且无输出，并分享了用于启动训练的命令行。
  
  - 另一位成员建议分享示例配置以诊断问题。
- **DeepSpeed 错误已解决**：用户遇到了一个 **DeepSpeed** 错误，提示“输入应为有效整数，但得到了带有小数部分的值”。
  
  - 社区建议确保设备数量是 2 的倍数，这最终解决了该问题。
- **意外的 CUDA 显存不足**：尽管拥有 5 个 24GB 显存的 GPU，用户在训练期间仍遇到了 **CUDA out of memory** 错误。
  
  - 他们分享了其 **DeepSpeed** 和 **accelerate** 配置，以寻求关于显存短缺的见解。
- **Runpod 实例见解**：用户提到了他们的 **DeepSpeed** 配置，指出该配置源自 **GitHub** 上的示例。
  
  - 他们强调在 **Runpod** 实例上运行实验，并说明了其规格参数以提供背景信息。
- **社区协作进行故障排除**：成员们积极协作，排除各种模型训练和配置问题。
  
  - 他们交换了见解和配置链接，帮助解决用户关于训练和资源管理的问题。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **模型可扩展性引起关注**：一位成员对一篇基于 **350 billion tokens** 训练的论文的 **scalability**（可扩展性）表示担忧，质疑其改进的显著性。
  
  - *讽刺的是*，另一位成员指出 **ML professionals** 经常忽视像 **p-values** 这样的基础统计指标。
- **P-values 在 ML 中不常见**：一位成员对 **ML** 论文中缺乏 **p-values** 和 **confidence intervals**（置信区间）表示沮丧，称这对于有医学背景的人来说很受刺激。
  
  - 另一位参与者评论说，他们很少在 **ML** 语境中看到 **p-value** 的使用，突显了科学报告中的文化差异。
- **SOAP 表现优于 AdamW 但需要调优**：一位用户在 **Alpaca** 上测试了 **SOAP optimizer**，指出在调整 **AdamW** 的 **learning rate** 之前，其表现优于 **AdamW**。
  
  - 然而，他们提到目前的实现尚不支持 **distributed** 训练或 **bf16** 格式。
- **Diff Transformer 胜过传统 Transformers**：**Diff Transformer** 引入了 **differential attention mechanism**（微分注意力机制），增强了对相关上下文的关注，在各种基准测试中表现优于传统的 **Transformers**。
  
  - 它在 **long-context modeling** 方面有显著帮助，并减少了问答等任务中的 **hallucination**（幻觉）。
- **L-Mul 算法大幅降低能源成本**：提出的 **L-Mul algorithm** 通过整数加法近似浮点乘法，在保持更高精度的同时将能源成本降低了 **95%**。
  
  - 该方法相比 **8-bit floating point multiplications** 有显著改进，表明在神经网络计算中具有巨大的资源节省潜力。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **探索 LangChain 对 Memcached 的支持**：一名成员正在研究在 LangChain 中添加对 **pymemcache** 的支持是否足够，或者支持更广泛的客户端（如 **python-memcached** 或 **pylibmc**）是否会更有益。
  - 目标是提高 LangChain 内部的**缓存灵活性**，使其能够更好地适应不同的缓存需求。
- **LiteLLM 的流式传输与缓存问题**：有成员担心 **LiteLLM** 在流式传输时无法检索已缓存的 token，并引发了关于确保有效缓存最佳实践的咨询。
  - 共享了关于 [LiteLLM](https://docs.litellm.ai/) 的资源，暗示 *token stream responses*（token 流响应）可能会干扰缓存机制。
- **AI 中的 SQL 查询限制**：一位用户提出了关于在不依赖 LLM 指令的情况下将 SQL 查询限制在特定 ID 的问题，寻求更严格的查询生成方法。
  - 另一名成员建议使用 **grouping by ID**（按 ID 分组）来改进过滤并获得更可靠的结果。
- **SQL Chain 与其他模型的兼容性**：有人提出了关于 SQL chain 在 **GPT 3.5** 以外模型上的性能问题，这些模型经常返回不准确的结果。
  - 一名成员发现通过专注于精确的列命名和仔细的问题表述，使用 **4o-mini** 取得了成功。
- **集成 Livekit 以实现实时 LangChain 功能**：有成员表示有兴趣将 **Livekit** 与 LangChain 集成，以增强其在高级应用中的实时能力。
  - 该成员特别提到计划开发一个 **RAG bot**，展示了他们在渐进式应用开发方面的雄心。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **准备好参加 Mozilla AI 演讲！**：下周，我们很高兴邀请到 **Mozilla AI** 的成员进行演讲，讨论有趣的开源倡议。不要错过这个了解更多信息的机会！
  - 你可以[点击此处参加活动](https://discord.gg/open-interpreter-1146610656779440188?event=1293314042596950067)以获取见解。
- **对 --stdin 标志的困惑**：一位用户对如何使用 **\--stdin** 标志表示困惑，并提到在文档中找不到指导，这突显了文档清晰度的缺失。
  - 需要进一步澄清以帮助用户有效利用此功能。
- **LLM 在相同种子下保持确定性**：一次讨论显示，如果使用相同的种子（seed）和输入，**LLM** 可以是确定性的，这与普遍看法相反。ChatGPT 在每次请求时会随机化种子以引入非确定性。
  - 值得注意的是，使用相同的输入并将 temperature 设置为 **0** 应该会产生一致的结果。
- **模型更新带来的不可预测性**：有成员担心 ChatGPT 中的**模型更新**可能会随着时间的推移影响结果的一致性。模型的变化可能会导致变异，从而破坏之前的确定性行为。
  - 用户强调，即使代码保持不变，更新也可能引入不可预测性。
- **跨系统的代码结果变异性**：一位成员指出，系统或 Python 的更新可能会影响代码行为，导致结果多变。例如，访问用户 token 可能会改变执行路径。
  - 这种变异性强调了受控环境对于获得一致结果的重要性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 中的 Clang 后端错误**：一位用户在 Linux 上使用 **clang** 后端运行 `exo` 时遇到错误，包括在两个系统上都能复现的 MetaOps.KERNEL lowering 错误，可能与 [Nix 软件包问题](https://discord.com/channels/1068976834382925865/1068976834928193609/1293313517390135407)有关。
  - 此外，运行 `TINYGRAD_DEBUG=2` 在崩溃前记录了数百个操作，揭示了详细的活动但未立即报错。
- **为 tinygrad 学习者引入 Fashion MNIST**：一名成员提交了一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/6961)，提议添加 **Fashion MNIST** 作为新数据集，为 **tinygrad** 教育的推动者弥补 **MNIST** 和 **CIFAR-10** 之间的复杂度鸿沟。
  - 这一举措反映了社区扩充学习资源的渴望，引发了关于增加更多数据集以进一步丰富训练体验的讨论。
- **扩展学习用的数据集选项**：成员们表示有兴趣向 **tinygrad** 添加更多数据集，这表明了提升现有选项之外学习机会的协作努力。
  - 对新数据集的需求有望创造一个更多样化的学习环境，允许用户尝试各种数据类型和挑战。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **层次化生成受到关注**：一位成员分享了一篇关于[耦合生成与压缩](https://theadamcolton.github.io/a-theory-for-coupling-generation-and-compression)的博客文章，讨论了一个类似于 **Stable Cascade** 模型的 **Hierarchical Generation** 框架。
  
  - 文章强调了目前流行的模型范式，即首先训练一个 **decomposer**（分解器），这显著影响了 LLM 和图像生成的输出。
- **o1-preview 将重新定义 Zero-shot 能力**：初步研究结果显示，**o1-preview** 在 **zero-shot (weak) out-of-distribution generalization**（零样本弱分布外泛化）方面表现出显著优势，超越了之前的模型。
  
  - **o1-mini** 则没有表现出这种进步，仅与之前的 SOTA 持平，这清楚地说明了 **pre-training scale**（预训练规模）对模型效能的价值。
- **TruthfulQA 展示了 o1 的理解能力**：**o1** 在 **TruthfulQA** 上取得了优异成绩，特别是在有效理解常见误解方面，表明其在理解任务中具有潜力。
  
  - 尽管存在一些限制，但这一表现证明了 **o1** 在应对某些理解挑战方面取得了显著成功。

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **轻松获取随机猫咪图片**：一项新功能演示了如何使用 [The Cat API](https://api.thecatapi.com/v1/images/search) **获取随机猫咪图片**。该实现涉及创建一个 `Cat` 模型并利用 HTTP 客户端进行无缝图像检索。
  
  - 该演示强调了简单性，允许开发者轻松地将猫咪图片集成到他们的应用程序中。
- **限制猫品种获取数量**：展示的一种方法允许用户在**获取猫品种**时限制返回的数量。代码片段显示，仅检索有限的一组品种，并可以结构化为 `CatBreed` 模型以便高效访问。
  
  - 这一增强功能为开发者提供了对数据检索更严格的控制，使其更容易处理大型数据集。
- **为视觉学习者提供的视频演示**：分享了[演示视频](https://www.loom.com/share/bfcbab5223214960a75cc230 those-d9d647e0-979d-4a76-8f1d-5ddc5450ae7a)链接，提供了关于猫咪图片和品种获取功能的可视化说明。这些指南为用户阐明了实现过程。
  
  - 这些资源使开发者能够有效地掌握工具并充满信心地进行实施。

 

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Whisper Turbo 德语模型将错误率减半**：根据[来源](https://huggingface.co/primeline/whisper-large-v3-turbo-german)，新推出的 **Whisper Turbo German** 模型在各种基准测试中比早期版本降低了 **50%** 的错误率。该模型专门针对德语的**转录**、**语音命令**和**自动字幕**进行了优化。
  
  - 它通过为文字处理软件提供**听写功能**，增强了在多种场景下的可用性，使其成为处理德语语言的开发者的宝贵工具。
- **Whisper Turbo 模型的应用**：**Whisper Turbo German 模型**的主要应用包括德语口语的高效转录、自动字幕以及辅助基于语音的搜索查询。
  
  - 开发者可以利用这些功能开展各种项目，提高德语环境下的无障碍性和交互性。

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Writer 的 Palmyra-X-004 模型更新请求**：来自 Writer 的 Sam Julien 在收到 CTO Waseem AlShikh 的邮件后，请求将 **Palmyra-X-004** 模型添加到 leaderboard 中，展示了他们在内部 benchmarks 中**令人印象深刻的结果**。
  
  - *我们需要提交 PR 吗？* 突显了他们对社区参与的承诺。
- **澄清 Leaderboard 提交流程**：Sam 还就 Palmyra-X-004 模型加入 leaderboard 是否需要 **PR** 寻求澄清。
  
  - 这一询问反映了一种结构化的方法，以确保他们的进展在社区内得到有效认可。

 

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1293286849879740456) (204 messages🔥🔥):

> - `Llama 3.2 查询`
> - `MLX 模型问题`
> - `模型可访问性`
> - `LM Studio 0.3.4 的新功能`
> - `量化模型相关问题`

- **Llama 3.2 模型加载问题**：用户报告在 LM Studio 0.3.4 中加载 Llama 3.2 和 Dolphin 2.2.1 等模型时遇到困难，一些在旧版本中可以运行的模型现在无法加载。
  
  - 建议将后端更改为 Vulkan，作为提高模型加载兼容性的潜在解决方案。
- **MLX 性能问题**：一些用户在使用 MLX 运行 Llama 3.1 8B Instruct 4bit 等模型时遇到了无限输出循环，这表明模型响应存在问题。
  
  - 对话表明，问题可能源于模型对 Prompt 的解析方式，导致了重复输出。
- **LM Studio 的无障碍问题**：一位用户询问在哪里报告与 LM Studio 中屏幕阅读器（screenreader）功能相关的无障碍问题。
  
  - 社区提供了指导，建议在平台内的特定指定频道提出问题，以便获得更好的关注和处理。
- **版本更新与兼容性**：LM Studio 0.3.4 的发布引发了关于模型兼容性的讨论，一些用户不确定更新是否能无缝迁移现有的模型预设（presets）。
  
  - 澄清指出，用户在更新到最新版本后，可能需要手动调整模型设置并检查模型迁移情况。
- **Llama 3.2 11B 的可用性**：一位用户表示有兴趣使用 Llama 3.2 11B，但据指出，该模型目前在 LM Studio 中不受支持。
  
  - 对某些大型模型的支持一直是一个反复出现的问题，表明用户对更高兼容性的持续需求。

**提到的链接**：

- [no title found](https://releases.lmstudio.ai/win32/x86/0.3.4/3/LM-Studio-0.3.4-Setup.exe): 未找到描述
- [Get error when trying to run self-quantized versions of Hermes-3-Llama-3.1-8B with 8-Bits and group-size 128 · Issue #6 · lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine/issues/6): 系统 Mac OS Sequoia 版本 15.0.1 (24A348) 2020 M1 MacBook Pro 16GB uname -a: Darwin 24.0.0 Darwin Kernel Version 24.0.0: Tue Sep 24 23:36:26 PDT 2024; root:xnu-11215.1.12~1/RELEASE_ARM64_T8103 ...
- [nvidia/NVLM-D-72B · Hugging Face](https://huggingface.co/nvidia/NVLM-D-72B): 未找到描述
- [Download LM Studio - Mac, Linux, Windows](http://lmstudio.ai/download?os=linux): 发现、下载并运行本地 LLM
- [Video Allegedly Shows Crypto Miners Jet Washing Nvidia RTX GPUs](https://www.tomshardware.com/news/crypto-miners-allegedly-jet-washing-gpus): “轻度使用”的 GPU 廉价货...
- [bunnycore/Llama-3.2-3B-Mix-IQ4_XS-GGUF · Hugging Face](https://huggingface.co/bunnycore/Llama-3.2-3B-Mix-IQ4_XS-GGUF): 未找到描述
- [rombodawg/Rombos-LLM-V2.5-Qwen-14b · Hugging Face](https://huggingface.co/rombodawg/Rombos-LLM-V2.5-Qwen-14b): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/15lf119/inference_speed_for_llama_2_70b_on_a6000_with/): 未找到描述
- [Refurbished 14-inch MacBook Pro Apple M2 Max Chip with 12‑Core CPU and 38‑Core GPU - Space Gray](https://www.apple.com/shop/product/G17GFLL/A/refurbished-14-inch-macbook-pro-apple-m2-max-chip-with-12%E2%80%91core-cpu-and-38%E2%80%91core-gpu-space-gray?fnode=b8f00c7905d02556476d32397d8412814f925d6e1d1af8c2eb62c99bd9ff8de54f18b43799fe654a86a07a522255e486fbf1a60b34d229b8e4102b220073925e6fb38101b5291b27f181fe2d53f90d17): 由 M2 Pro 或 M2 Max 助力，MacBook Pro 的动力和效率达到了前所未有的高度。无论是否插电，它都能提供卓越的性能，且现在电池续航时间更长...
- [New PNY RTX A6000 48GB GDDR6 Graphics Card VCNRTXA6000-PB 751492641676 | eBay](https://www.ebay.com/itm/176607468139?_skw=nvidia+a6000&epid=9046134433&itmmeta=01J9Q9X3HJFG9M2WHZ9C29V9EQ&itmprp=enc%3AAQAJAAAA8HoV3kP08IDx%2BKZ9MfhVJKkfui%2FRQPbh7nYfReOhQKf2IWz%2F%2BzwH4yg%2BHfGPS34jwgvuCEpJIumUddiOSGJYxTJiHgnOJNN4Rm2u1ftvcfJBegjSJK%2FJJVhY1Y5vezgzQwijBLmUCa8f74N9QW%2BV9Xt3BU58xNRT4mWiU%2Bb%2BaXM%2BppxW8spUOBCwRNkVtSN6xIcyl4%2FrtKH2KdmX6IphDznF%2FIx1CeezsAx8PgJaiOqLDrziu3IYSk6Sr0GMfwpid0De170KXEW8XCoB6NtCDNmINU5E8zyD9e8EMzAwPjmZ4WSs%2FWgmJlS2bf6hBdvG8g%3D%3D%7Ctkp%3ABk9SR-y49OnNZA&LH_BIN=1): 未找到描述

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1293287249034743890) (30 条消息🔥):

> - `使用双 GPU`
> - `RTX 3060 和 RX 6600 的性能`
> - `R9 7900X CPU 性能`
> - `虚拟机中的 AVX2 支持`
> - `GPU 架构差异`

- **双 GPU 设置仍受性能限制**：成员们讨论了同时使用 **RTX 3060** 和 **RX 6600** 以获得总计 **20GB VRAM**，同时指出它们协同工作效果不佳，尤其是在速度方面。
  
  - 一位成员建议，虽然双显卡设置可以加载更大的模型，但由于 **6600** 的性能限制，速度基本上保持不变。
- **增加 VRAM 的最佳选择**：对话强调，添加第二块 **RTX 3060** 有助于加载更大的模型，但不会提高速度，这呼应了为了准确性而需要更多 VRAM 的观点。
  
  - 一位用户计划攒钱购买更强大的 GPU，特别是 **RTX 3090**，并承认目前 **9-10 tok/sec** 的速度尚可接受。
- **在纯 CPU 设置上运行模型**：有人询问在纯 CPU 的 **Ubuntu VM** 上运行模型的问题，成员们确认必须使用支持 **AVX2** 指令集的 CPU。
  
  - 然而，他们警告说速度可能会很慢，并建议尝试一下，因为有些软件是免费的。
- **NVIDIA 放弃 NVLink**：讨论显示 **NVIDIA RTX 4000 系列**不支持 **NVLink**，而是转向 **PCIe Gen 5** 进行多 GPU 设置。
  
  - 这一变化引发了人们对未连接 GPU 性能能力的兴趣，用户对其速度表示惊讶。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1293288457656995874) (200 条消息🔥🔥):

> - `Model Merging at Scale` (大规模模型合并)
> - `Fine-tuning Qwen 2.5` (微调 Qwen 2.5)
> - `Unsloth AI and Dataset Formats` (Unsloth AI 与数据集格式)
> - `Instruct vs. Base Models` (指令模型 vs. 基座模型)
> - `Hugging Face Datasets` (Hugging Face 数据集)

- **大规模模型合并的新见解 (Model Merging at Scale)**：@Prateek Yadav 分享了关于大规模模型合并的令人兴奋的新工作，探讨了在合并高达 **64B 参数** 的大型模型时的性能问题。这项研究评估了模型大小、质量和方法如何影响性能和泛化能力。
  
  - 相关论文可以在 [arXiv](https://arxiv.org/abs/2410.03617) 上找到，详细介绍了系统性的评估和发现。
- **微调 Qwen 2.5 现已顺畅运行**：@theyruinedelise 确认，在解决了之前的 Prompt 问题后，现在微调 Qwen 2.5 模型已无障碍。可以在 Hugging Face 上找到一系列可用的 Qwen 2.5 模型集合。
  
  - 这为有兴趣利用这些模型进行微调任务的用户提供了保障。
- **理解 Unsloth 的数据集格式**：讨论强调，虽然可以使用 CSV 文件作为数据集，但使用 Parquet 等格式配合 Hugging Face 的默认数据集可能更高效。提醒用户确保其数据集结构与预期的列格式一致。
  
  - 例如，为了清晰起见，可以指定名为 'train' 和 'conversations' 的列。
- **区分指令模型 (Instruct) 和基座模型 (Base)**：用户澄清，指令模型是专门针对直接 Prompt 响应进行微调的，包含了针对回答问题的优化，而基座模型主要关注预测下一个 Token。这种区别允许在不同场景下进行针对性应用。
  
  - 指令模型还可能包含对齐偏差 (Alignment Bias)，这可能会影响其回答。
- **探索数据集转换工具**：有建议将数据集转换为 Hugging Face 支持更好的格式，建议编写自定义脚本或使用现有函数。这确保了数据集能被正确上传以用于预期的训练目的。
  
  - 使用 `load_dataset('csv')` 函数可以帮助简化这一过程，使其对用户更友好。

**提到的链接**：

- [Prateek Yadav (@prateeky2806) 的推文](https://x.com/prateeky2806/status/1843643582432854171)：是否想过模型合并在大规模下是否有效？也许对于更大的模型，收益会递减？也许你考虑过将模型合并用于大型模型的训练后处理，但不确定它是否能泛化……
- [Google Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing#scrollTo=LjY75GoYUCB8)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/1bMOKOBzxQWUIGZBs_B0zm8pimuEnZdfM?usp=sharing)：未找到描述
- [What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617)：模型合并旨在将多个专家模型组合成一个能力更强的单一模型，具有降低存储和推理成本、提高泛化能力以及支持去中心化等优点……
- [Qwen 2.5 - Unsloth 集合](https://huggingface.co/collections/unsloth/qwen-25-66fe4c08fb9ada518e8a0d3f)：未找到描述
- [unsloth/Llama-3.2-3B-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/Llama-3.2-3B-bnb-4bit)：未找到描述
- [yahma/alpaca-cleaned · Hugging Face 数据集](https://huggingface.co/datasets/yahma/alpaca-cleaned)：未找到描述
- [Killed by Google](https://killedbygoogle.com/)：Killed by Google 是一个关于已停止服务的 Google 产品、服务和设备的开源列表。它是对被 Google 砍掉的深受喜爱的服务和产品的致敬和纪念。
- [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)：未找到描述
- [无标题链接](https://www.youtube.com/results?search_query=windows+11+wsl2+vscode)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1293295762926469121) (19 messages🔥):

> - `Colab gguf file download struggles`
> - `Using logits with Ollama Llama`
> - `Continued pretraining of Llama 3.2 3b`
> - `AMD GPU limitations with Unsloth`
> - `Fine-tuning with Unsloth FastLanguageModel`

- **Colab gguf file download struggles**: 成员们表达了从 **Colab 下载大型 gguf 文件**时的挫败感，理由是下载不完整和连接断开等问题。
  
  - 提供的一种解决方案是直接将文件上传到 Google Drive 或 Hugging Face，以避开 Colab 的下载限制。
- **Using logits with Ollama Llama**: 一位成员询问如何从 Python 中通过 Ollama 安装的 **Llama 获取 logits 分数**，但没有找到明确的资源。
  
  - 另一位成员建议，如果需要 logits，切换到 **llama.cpp** 可能是更好的选择。
- **Continued pretraining of Llama 3.2 3b**: 用户希望在不使用 PEFT 的情况下**持续预训练 Llama 3.2 3b** 以整合新知识，并质疑其可行性。
  
  - 回复指出，更高的 rank 和包含 embedding 层对于微调和理解参数量至关重要。
- **AMD GPU limitations with Unsloth**: 一位成员对在 Intel GPU 上创建**小型 LoRA 模型**的能力表示担忧，因为硬件存在限制。
  
  - 澄清了 **Unsloth 不支持 AMD GPU**，也不支持多 GPU 训练设置。
- **Fine-tuning with Unsloth FastLanguageModel**: 用户确认将所有模型参数的 `requires_grad` 设置为 true 即可使用 **Unsloth FastLanguageModel** 进行全量微调。
  
  - 有关于与 **trl 的 SFTTrainer** 兼容性的咨询，表明了利用两者进行微调过程的兴趣。

---

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1293291064202891265) (1 messages):

> - `Nvidia models`
> - `Meta's VLMs`
> - `Hugging Face Accelerate 1.0`
> - `ColPali multimodal retrieval`
> - `Paper Central`

- **Nvidia launches high-efficiency models**: Nvidia 推出了 [Nemotron 51B](https://x.com/NVIDIAAIDev/status/1838263496049570053)，这是一款经过 NAS 优化的模型，在保持准确性的同时，在单张 H100 GPU 上实现了 **2 倍吞吐量**。用户可以通过 [NVIDIA's API](http://ai.nvidia.com) 体验该模型或从 Hugging Face 下载。
  
  - 该模型还伴随着其他几款模型，包括旨在增强 AI 能力的 [NVLM 1.0](https://huggingface.co/nvidia/NVLM-D-72B) 和 [OpenMath](https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1)。
- **Meta releases improved VLMs**: Meta 发布了其首批 VLM，包括 [CoTracker 2.1](https://x.com/NielsRogge/status/1842958590396772599)，能够在单张 GPU 上跟踪 **7 万个点**以进行视频运动预测。相关论文可在[此处](https://huggingface.co/papers/2307.07635)查阅。
  
  - 用于图像/视频分割的 [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) 模型也获得了更新，增强了其对开发者的实用性。
- **Hugging Face Accelerate 1.0 launched**: Hugging Face 宣布发布 [Accelerate 1.0](https://x.com/TheZachMueller/status/1843320011139813644)，具有多项用于无缝 AI 开发的新功能。此更新受到好评，促使用户探索其改进。
  
  - 详细概述请参阅公告博客[此处](https://huggingface.co/blog/accelerate-v1)。
- **ColPali: New retrieval approach**: [ColPali](https://x.com/vanstriendaniel/status/1841515562557702330) 引入了一种创新的多模态文档检索方法，尽管对其实用性存在一些保留意见。与 [Qdrant](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) 的集成允许对 embedding 进行高效的索引和搜索。
  
  - 相关博客文章提供了关于如何将 ColPali 与现有向量数据库有效配合使用的见解。
- **Paper Central for research updates**: Hugging Face 推出了 [Paper Central](https://x.com/IAMJBDEL/status/1841627341195510256)，这是一个旨在汇集最新研究论文的空间。它聚合了 arXiv 和 GitHub 等来源，让研究人员保持信息更新。
  
  - 该计划旨在简化对关键学术资源的访问，提高研究社区的生产力。

**Links mentioned**:

- [来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文](https://x.com/NVIDIAAIDev/status/1838263496049570053),): 👀 体验高效的 NVIDIA Llama-3.1-Nemotron-51B —— 一款经过 NAS 优化的模型，在保持准确性的同时实现了 2 倍吞吐量，并可在单张 H100 GPU 上运行。✨ 试用 Llama-3.1-Nemotron-51B N...
- [来自 Niels Rogge (@NielsRogge) 的推文](https://x.com/NielsRogge/status/1842958590396772599)): Meta 在 @huggingface 上发布了 CoTracker 2.1，这是其基于 Transformer 的视频运动预测模型的改进版本！能够在单张 GPU 上共同追踪 7 万个点。论文（附带 l...
- [来自 Tris Warkentin (@triswarkentin) 的推文](https://x.com/triswarkentin/status/1841823657108373838)): Gemma 2 变得更棒了！🚀 新的日语微调 2B 模型以及一项 15 万美元的 Kaggle 竞赛，旨在为每种语言构建 Gemma 模型。很高兴 @sundarpichai 在这里分享这份喜悦！阅读更多...
- [来自 Zach Mueller (@TheZachMueller) 的推文](https://x.com/TheZachMueller/status/1843320011139813644)): 这一天终于到来了，@huggingface Accelerate 1.0 现已发布！有大量新功能值得探索，未来还有更多。我将快速谈谈我最喜欢的 🧵 复习一下，g...
- [来自 merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1843235751666016418)): 你的 LLM 无法理解视频和图像？真遗憾 😔 幸运的是，我们为视频语言模型发布了一项新任务 🤗 在 @huggingface /models 的左侧标签页中查找 video-text-to-text ⏯️ 它还附带...
- [来自 Adina Yakup (@AdinaYakup) 的推文](https://x.com/AdinaYakup/status/1843318863380750581)): 这是一个来自 @huggingface 中文社区的排行榜和竞技场（Arenas）集合 🔥🏆🇨🇳 https://huggingface.co/collections/zh-ai-community/leaderboards-and-arenas-664b6913bfd9b93ba4ac242...
- [来自 Julian Bilcke (@flngr) 的推文](https://x.com/flngr/status/1842358136239210866)): 现在看起来是这样的（我是服务器的唯一用户，所以运行很流畅 😂）
- [来自 Daniel van Strien (@vanstriendaniel) 的推文](https://x.com/vanstriendaniel/status/1841515562557702330),): ColPali 是一种令人兴奋的多模态文档检索新方法，但有些人怀疑它在现有向量数据库（vector DBs）中的实际应用。事实证明，使用 @qdrant_engine 进行索引和搜索非常容易...
- [来自 JB Delbrouck (@IAMJBDEL) 的推文](https://x.com/IAMJBDEL/status/1841627341195510256),): Paper Central 是一个新的 🤗 Hugging Face Space，旨在提供最新研究论文的实时信息。它是第一个将所有关键资源整合在一起的门户...

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1293291602869227581) (134 条消息🔥🔥):

> - `Model Performance Comparison` (模型性能对比)
> - `Mira Network Discussion` (Mira 网络讨论)
> - `Model Specificity in Use Cases` (用例中的模型特异性)
> - `TensorFlow Issues` (TensorFlow 问题)
> - `Python Community Q&A` (Python 社区问答)

- **编程任务中的 AI 模型对比**：用户讨论了不同 AI 模型的使用经验，指出在生成 Rust 代码时，**Claude Sonnet 3.5** 的表现明显优于 **GPT o1 preview**，且所需的 Prompt 更少。
  
  - 一位用户分享了在调试代码时同时有效使用 Claude 和 GPT 以最大化产出的策略。
- **Mira 去中心化洞察**：一位成员介绍了 **Mira**，这是一个旨在让 AI 普及的去中心化基础设施，强调其专注于社区驱动的项目，且不包含加密代币。
  
  - 尽管其技术前景看好，另一位用户对区块链和加密货币的关联表达了道德层面的担忧。
- **需要明确的模型使用指南**：一位用户质疑 Model Card 中缺乏关于各种 AI 模型特定应用（如建筑和结构工程任务）的清晰说明。
  
  - 成员们指出，详细的 Model Card 通常取决于作者在概述有效用例方面的努力和专业知识。
- **TensorFlow 在 GPU 上的问题**：几位用户表达了对 **TensorFlow** 在 GPU 上性能的不满，报告了与 Tensor 初始化问题相关的 Bug，这些问题阻碍了他们的工作。
  
  - 建议探索替代方案或排查底层错误以改进功能。
- **参与 Python 和数据科学讨论**：该频道允许围绕 Python 提出各种问题，用户探讨了工作流自动化和结构化数据提取等主题。
  
  - 总的来说，对话反映了同行之间技术咨询和社区故障排除的融合。

**提到的链接**：

- [Mira](https://mira.network/)：使 AI 通用化的去中心化基础设施
- [Klok](https://klokapp.ai/)：Klok - 随叫随到的加密智能
- [plandex/app/server/model/prompts at main · plandex-ai/plandex](https://github.com/plandex-ai/plandex/tree/main/app/server/model/prompts)：终端中的 AI 驱动开发。专为大型现实任务设计。 - plandex-ai/plandex
- [Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1fzobbz/nobel_prize_in_chemistry_awarded_to_deepmind_ceo/)：未找到描述
- [ACEMAGICIAN RGB Mini PC AMD Ryzen 9 6900HX (fino a 4,9 GHz),32 GB DDR5 512 GB SSD, AMD Radeon RX 680M Micro Computer Desktop 【Modalità regolabile Auto/Silenziatore Eco/Performance】 : Amazon.it: Informatica](https://amzn.eu/d/eqRdDjU)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1293324612985294919) (9 条消息🔥):

> - `Hierarchical Generation` (分层生成)
> - `Image Autoencoder Integration` (图像 Autoencoder 集成)
> - `Differences in Model Types` (模型类型差异)
> - `Hugging Face Metrics Implementation` (Hugging Face Metrics 实现)

- **分层生成见解**：一位成员分享了一篇[博客文章](https://theadamcolton.github.io/a-theory-for-coupling-generation-and-compression)，讨论了分层生成范式，强调了 Decomposer（分解器）和 Generator（生成器）在模型训练中的作用。
  
  - 他们强调了压缩在生成模型中的重要性，特别指出这种范式如何同时适用于 LLM 和图像生成器。
- **利用图像 Autoencoder**：围绕如何利用图像 Autoencoder 处理下游 Latent Space（潜空间）展开了讨论，如分层生成文章中所述。
  
  - 作为回应，文章作者解释说，Encoder 的功能类似于 VAE，经过训练可以为小型 Diffusion 模型生成有用的 Latent。
- **探索模型类型和数据集**：一位成员表示有兴趣了解 Base 模型和 Instruct 模型之间的区别，以及适合 LoRA 微调的数据集。
  
  - 这体现了社区对模型定制化和训练数据相关性日益增长的关注。
- **使用 Hugging Face 评估微调模型**：另一位成员分享了他们在训练流水线中集成 Hugging Face Metrics（如 **ROUGE** 和 **BertScore**）的学习过程，以增强模型评估。
  
  - 目标是摆脱对其他库的依赖，采用更定制化的方法来评估微调模型。

 

**提到的链接**：[coupling generation and compression](https://theadamcolton.github.io/a-theory-for-coupling-generation-and-compression)：未找到描述

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1293606253070782557) (1 条消息):

> - `Scade tools`
> - `Comfy-Flux integration`
> - `Custom image generations`

- **实验 Scade 的自定义图像工具**：一位成员分享了使用 **Scade** 创建自定义工具的经验，包括背景移除器、手部修复和图像放大器。这些工具可以直接从提供的 [Drive 链接](https://drive.google.com/drive/folders/1rSE8sDFV_w29Ucb_3A7TMAVHGb5rksn9?usp=sharing) 导入。
  
  - *最大的优势是在 Scade 上构建这些工具非常便宜*，而且与从头开始创建工具相比，**Comfy-Flux** 集成大大提升了它们的质量。
- **自定义工具的分享与反馈**：该用户鼓励社区尝试上述工具并提供反馈，希望能获得改进建议。他们还强调在 [Scade 社区](https://community.scade.pro/t/created-useful-tools-with-comfy-flux-on-scade-pro/96?u=velox) 分享了这些进展，以获得更广泛的关注。
  
  - 该成员强调，有效使用这些工具可以在保持易用性的同时提高图像生成质量。

**提到的链接**：

- [3 tools on comfy-flux - Google Drive](https://drive.google.com/drive/folders/1rSE8sDFV_w29Ucb_3A7TMAVHGb5rksn9?usp=sharing)：未找到描述
- [Scade](https://app.scade.pro/flow/)：未找到描述
- [Created Useful Tools with Comfy-Flux on Scade.pro](https://community.scade.pro/t/created-useful-tools-with-comfy-flux-on-scade-pro/96?u=velox)：我一直在实验自定义图像生成，想分享一些我使用 Comfy + Flux + Scade 为自己构建的工具。背景移除器：轻松移除背景...

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1293513520989732864) (4 条消息):

> - `VividNode Updates`
> - `Burnout in Tech Creators`
> - `Fine-tuning Whisper for ATC`
> - `FluxBooru-CFG3.5`

- **VividNode v1.4.0 发布**：**VividNode v1.4.0** 的版本发布包含了对 **gpt4free** 的支持，允许用户手动选择提供商和模型，增强了用户的灵活性。
  
  - 尽管功能强大，**gpt4free** 仍面临 Token 限制等挑战，离线使用 LLM 的优势依然显著。
- **技术创作者面临倦怠**：一位技术创作者表达了在平衡工作和侧边项目时的**倦怠感 (Burnout)**，强调了紧跟技术快速进步的艰辛。
  
  - 他们计划在 v1.5.0 发布后招募贡献者，并承认通常只有在主动寻求时才能获得支持。
- **微调 Whisper 模型提升性能**：发布了一篇博文，详细介绍了在**飞行员-航空交通管制 (ATC) 通信**数据上微调 **Whisper 模型**的过程，实现了 **84% 的相对性能提升**。
  
  - 这一过程将**词错率 (WER)** 从 **94.59%** 降低到了仅 **15.08%**，展示了定制化 ASR 解决方案的影响力。
- **微调 Whisper 的资源**：用于微调 Whisper 的模型和数据集现已在 Hugging Face 上分享，包括 [GitHub 仓库](https://github.com/jack-tol/fine-tuning-whisper-on-atc-data) 和数据集。
  
  - 还提供了 [博文](https://jacktol.net/posts/fine-tuning_whisper_for_atc/) 和 Hugging Face 模型的链接以供进一步探索。
- **FluxBooru-CFG3.5 发布**：分享了 **FluxBooru-CFG3.5** [Hugging Face Space](https://huggingface.co/spaces/bghira/FluxBooru-CFG3.5) 的链接，表明了该领域的最新进展。
  
  - 消息中未详细说明其功能和应用。

 

**提到的链接**：[Release v1.4.0 · yjg30737/pyqt-openai](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.4.0)：更新内容包括在消息表中添加 is_g4f 和 g4f_platform 字段，移除旧文本，修复与近期更新相关的问题，重命名文件，将 globals.py 中的每个函数移动到 utils.py 以获得更好的组织结构...

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1293288401306517524) (8 条消息🔥):

> - `T5 模型的 ONNX 转换`
> - `法律文档的探索性分析`
> - `大数据技术讨论`
> - `LLM 输出验证`
> - `Hugging Face pipeline 的服务器设置`

- **T5 ONNX 文件探索**：一名成员指出，**T5 模型**所需的 ONNX 文件可以在 Hugging Face 页面的 ONNX 文件夹下找到，并建议根据需要下载。
  
  - 他们还分享了一个关于将 Transformers 模型转换为 ONNX 的不同方法的链接，包括一个特定的 [distilbert-base-uncased-finetuned-sst-2-english 示例](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)。
- **法律文档交流想法**：一名成员寻求与在**法律文档探索性分析**方面有经验的人士交流，表达了交换想法和问题的愿望。
  
  - 未记录到具体回复，表明对该话题可能有潜在兴趣但参与度有限。
- **大数据技术咨询**：一名成员询问是否有人精通**大数据技术**，特别是 **Kafka** 和 **Hadoop**。
  
  - 这一咨询突显了在项目中利用这些技术的潜在兴趣。
- **验证未知的 LLM 输出**：一名成员请求将 **LLMs** 的未知输出验证为 JSON 的技术，目标是在 **Python** 和 **JavaScript** 中进行验证和清洗。
  
  - 另一名成员推荐了 [json schema library](https://github.com/python-jsonschema/jsonschema)，他们在使用该库时取得了不同程度的成功。
- **Hugging Face pipeline 的高效服务器设置**：一名成员分享了使用 **Triton Inference Server** 加载 Hugging Face pipeline 的经验，但表达了在没有 GPU 的情况下可能存在过度设计的担忧。
  
  - 他们正在探索建立一个包含 **3-4 个模型**的服务器的替代方案，该服务器可以处理 HTTP 请求，而无需为每个模型使用 Docker 容器。

**提到的链接**：

- [使用 Hugging Face Optimum 将 Transformers 转换为 ONNX](https://huggingface.co/blog/convert-transformers-to-onnx)：未找到描述
- [GitHub - python-jsonschema/jsonschema: Python 的 JSON Schema 规范实现](https://github.com/python-jsonschema/jsonschema)：Python 的 JSON Schema 规范实现 - python-jsonschema/jsonschema

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1293293715615514684) (8 条消息🔥):

> - `Diffusion 模型中的图像质量`
> - `Flux Loras 与 GGUF 训练`
> - `使用 Diffusers 训练 Diffusion 模型`

- **评估 Diffusion 模型中的图像质量**：成员们讨论了某张特定图像的低分辨率问题，认为它可能是由 **Flux** 或带有 griffin lora 的 pony 模型生成的，但注意到它似乎经过了后期处理。
  
  - 有人强调，由于该图像具有通用性且缺乏细节，它可能描绘的是任何随机的人。
- **澄清 Diffusers 与模型类型**：一名成员澄清说，**diffusers** 是一个支持使用各种 Diffusion 模型的库，并特别指出 **Stable Diffusion XL** 和 **Flux** 是能力出众的模型。
  
  - 这种生成灵活性允许将模型与 **diffusers** 库集成。
- **在 GGUF 格式上训练 Flux Loras**：一名成员询问关于使用 **flux gguf** 格式训练 Flux loras 和微调的问题，随后有人提到目前尚不支持 GGUF，但使用 **6GB GPUs** 通过 **Kohya** 进行训练是可能的。
  
  - 有建议称 **gguf** 比 **fp16** 提供更高的精度，但目前还没有足够的 **int4 版本**对比数据。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1293334500171841636) (38 条消息🔥):

> - `CMD-R 的 Temperature 设置`
> - `JSON schema 讨论`
> - `新成员介绍`
> - `诺贝尔奖预测`
> - `HumanEval 与 QA 流程`

- **寻找 CMD-R 的合适 Temperature**：成员们讨论了 CMD-R 的最佳 Temperature 设置，建议对于确定性结果使用 **0.3**，对于创意任务则尝试 **0.8**。
  
  - 一位用户建议先以 **0.8** 生成内容，再以 **0.1** 进行格式化，并对生成成本表示关注。
- **JSON 格式化的影响**：一位用户指出 JSON 格式化可能会降低模型的能力，更倾向于通过 Prompt 提供格式。
  
  - 另一位成员建议使用 Schema 来改善输出，同时提高 Temperature 以获得更好的结果。
- **欢迎对 R 语言感兴趣的新成员**：一位正在学习 R 语言的新用户介绍了自己，并发现 CMD-R 服务出乎意料地适合 R 语言编码。
  
  - 讨论明确了 CMD-R 也可以编写 R 代码，这让新成员保持了参与兴趣。
- **诺贝尔奖公布预测**：有传言称 **Attention Is All You Need** 论文的作者可能会被授予诺贝尔文学奖。
  
  - 一些成员表示怀疑，而另一些人则表示支持，并指出了该论文的文化影响力和感召力。
- **生成式 AI 的 QA 方法**：参与者分享了生成式 AI 模型的 QA 工具和方法，并提到了评估框架。
  
  - 一位用户提到了 **HumanEval**，并对评估过程的运作方式表示感兴趣。

**提到的链接**：[Omar Sanseviero (@osanseviero) 的推文](https://x.com/osanseviero/status/1844003522632949803)：突发新闻：瑞典皇家科学院决定将 2024 年 #诺贝尔文学奖 授予《Attention Is All You Need》的作者。他们的作品让成千上万的人流泪、欢笑或……

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1293317900370972743) (36 条消息🔥):

> - `Cohere API 问题`
> - `System Role 格式化`
> - `用于 RAG 的 ETL 流水线`
> - `LLM 的零数据保留 (Zero Data Retention)`

- **Cohere API 连接问题**：一位成员报告了 Cohere API 的间歇性连接问题，收到的错误显示 'ChatResponse' 对象没有 'text' 属性。经过排查，他们发现使用 `response.message.content[0].text` 解决了该问题。
  
  - 成员们分享了排错代码的更新和测试，认为最近的 API 更新可能导致了混淆。
- **在 Markdown 中格式化 System Role**：一位成员询问了塑造 System Role 所需的语言结构，会议确认将任务和上下文格式化为 Markdown 会产生更好的效果。官方提供了文档以进一步指导如何构建有效的 System Message。
  
  - 提到了一种示例 System Message 结构，详细说明了简洁的指令如何高效地引导模型行为。
- **探索用于 RAG 的 ETL 解决方案**：一位用户介绍了他们的毕业设计项目，重点是开发一个用于非结构化数据处理的 ETL 流水线，旨在实现检索增强生成 (RAG)。他们寻求社区关于该技术的见解和经验。
  
  - 社区成员指出 Cohere 网站上有大量的用例和博客，并表示有兴趣听取类似项目的个人经验。
- **企业用户的零数据保留**：一位用户对客户数据保留政策表示担忧，特别是关于 LLM 长期存储 Prompt 的问题。他们获知，在满足某些使用承诺的情况下，企业客户可以选择零数据保留 (Zero Data Retention) 方案。
  
  - 澄清了 Cohere 提供此选项的条件，并将其与企业协议挂钩。

**提到的链接**：

- [从 API v1 迁移到 API v2 — Cohere](https://docs.cohere.com/docs/migrating-v1-to-v2)：该文档为希望将其现有的 Cohere API v1 实现更新到新的 v2 标准的开发者提供参考。
- [编写有效的 Prompt — Cohere](https://docs.cohere.com/v2/docs/crafting-effective-prompts)：此页面描述了进行 Prompt Engineering 的不同有效方式。
- [System Messages — Cohere](https://docs.cohere.com/docs/preambles)：此页面描述了 Cohere System Messages 的工作原理及其对输出的影响。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1293358325391691796) (46 条消息🔥):

> - `Cohere API 使用`
> - `银行 RAG 解决方案`
> - `Embedding 模型性能`
> - `试用密钥限制`

- **Cohere API 试用密钥限制**：一位用户讨论了使用 Cohere 试用密钥达到 **1000 次 API 调用限制**的问题，并确认生成新密钥也无法改变现状。
  
  - 成员们澄清，由于试用版仅用于轻量测试，增加调用次数需要升级为付费计划。
- **优化银行 RAG 解决方案**：另一位用户正在尝试使用 **Cohere** 为银行构建 RAG 解决方案，并指出在 **75% recall@5** 的指标下，其检索性能优于 **OpenAI**。
  
  - 他们计划嵌入 **2000 个 chunks** 进行概念验证 (proof of concept)，向银行展示结果。
- **商业场景下 API 调用的实用性**：讨论了在商业场景中使用试用版的实际可行性，有人评论对于大型项目来说，这可能不值得投入时间或成本。
  
  - 一位成员强调了合理预算成本的重要性，尤其是在处理银行等企业级客户时。
- **Cohere 作为安全的替代方案**：一位成员强调 **Cohere** 是 OpenAI 的直接竞争对手，专注于数据安全和隐私。
  
  - 他们确信 Cohere 可能是该用户银行聊天机器人项目的**完美解决方案**。
- **社区支持与反馈**：成员们对该用户的 **Cohere** 实验表示热烈支持，并鼓励其更新研究结果。
  
  - 他们欢迎根据实际应用结果，就 Cohere 产品中可改进的地方提出见解。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1293572145607344231) (29 条消息🔥):

> - `情感状态机 (Emotional State Machine)`
> - `情感传播 (Emotion Propagation)`
> - `AI 与心理健康`
> - `语音 AI 中的情感`

- **情感状态机增强机器人交互**：一种新的**情感状态机**通过**持久化记忆 (persistent memory)** 跟踪用户情绪，使助手机器人能够根据用户交互保持一致的情感基调。
  
  - 这种方法不同于大多数表现出**灵活情感状态**的机器人，如果用户触发了负面反应，它会保持不悦状态。
- **情感传播带来更丰富的体验**：系统利用**初级、次级和三级情感传播**来更细致地理解用户情绪，并能高效存储多达 **100 万**个状态。
  
  - 这种级联效应意味着用户可以体验到更真实的交互，多种情感可以同时影响响应。
- **应对心理健康的 AI 应用**：人们对利用 **AI 聊天机器人**提供心理健康支持表现出兴趣，以便在人类治疗师不在场时提供帮助。
  
  - 这些机器人的情感方面仍是一个挑战，因为一些机器人遇到了限制情感语境解读的**审查政策 (censorship policies)** 问题。
- **构建真实的机器人人格**：开发者正努力通过根据用户输入内容创建**情感评分 (emotional score)**，使机器人能够做出更真实的反应。
  
  - 该技术旨在提供多样化且动态的情感表达，尽管在响应中实现完美平衡仍处于开发阶段。
- **语音集成增强情感表达**：开发者正在推动将这些情感系统集成到**语音 AI** 中，挖掘超越文本交互的更具表现力的能力。
  
  - 语音表达可以传达更广泛的情感，极大地丰富了用户体验，超越了纯文本响应的局限。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1293300075598188565) (70 条消息🔥🔥):

> - `Aider 与模型集成`
> - `在 Aider 中使用 OpenRouter`
> - `对 Architect Mode 的反馈`
> - `关于 LLM 的社区讨论`
> - `Aider 功能问题`

- **用户关于 Aider 文件管理的查询**：用户讨论了 Aider 不会自动将新文件填充到文件列表中的挑战，通过使用 `/commit` 或直接指定文件路径来解决。
  
  - 另一位用户分享说，Aider 要求文件必须提交到 Git 仓库才能在自动补全中可见，强调了版本控制的重要性。
- **将外部 LLM 与 Aider 集成**：一位用户询问关于在 Aider 中使用 SambaNova 模型的问题，社区成员建议如果端点兼容 OpenAI，可以进行手动 API 配置。
  
  - 进一步的讨论揭示了通过元数据 JSON 文件手动添加模型定价和 Token 成本的方法。
- **Architect Mode 的行为受到审查**：用户对 Aider 的 Architect Mode 在编码前未能有效规划表示担忧，建议调整 Architect Prompt 以改进功能。
  
  - 一位用户强调，当前的 Prompt 可能会导致在没有足够观察输入的情况下过早开始编码，主张进行行为修改。
- **社区对 Anthropic 模型的推测**：成员们讨论了 Anthropic 可能发布的公告，以及需要使用适当的渠道讨论新模型发布。
  
  - Fry69_61685 引导用户到特定线程获取更新，并建议在讨论模型代理时避免违反 TOS。
- **Aider 的功能问题**：一位用户指出 Aider 显示文件已更改，但直到重新打开文件前这些更改都不可见。
  
  - 社区成员将其归因于 Chat Mode 设置，并强调了理解 Aider 运行模式的重要性。

**提到的链接**：

- [OpenRouter](https://aider.chat/docs/llms/openrouter.html)：Aider 是你终端里的 AI 配对编程工具
- [OpenAI 兼容的 API](https://aider.chat/docs/llms/openai-compat.html)：Aider 是你终端里的 AI 配对编程工具
- [安装 Aider](https://aider.chat/docs/install/install.html)：Aider 是你终端里的 AI 配对编程工具
- [Chat 模式](https://aider.chat/docs/usage/modes.html)：使用 chat、ask 和 help 聊天模式。
- [供应商 | liteLLM](https://docs.litellm.ai/docs/providers)：了解如何在 LiteLLM 上部署和调用来自不同供应商的模型
- [Chat 模式](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model)：使用 chat、ask 和 help 聊天模式。
- [OpenRouter](https://openrouter.ai)：LLM 路由和市场
- [高级模型设置](https://aider.chat/docs/config/adv-model-settings.html)：为 LLM 配置高级设置。
- [aider/aider/coders/architect_prompts.py at cd3e0ae91424c9d31f7b332e59c9f843eb0a7990 · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/cd3e0ae91424c9d31f7b332e59c9f843eb0a7990/aider/coders/architect_prompts.py#L6)：Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户为 Aider-AI/aider 的开发做出贡献。
- [Exponent](https://www.exponent.run)：Exponent 是你的 AI 配对编程助手。
- [litellm/model_prices_and_context_window.json at main · BerriAI/litellm](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)：Python SDK，代理服务器（LLM 网关），用于调用 100 多个 OpenAI 格式的 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1293287084408438814) (53 messages🔥):

> - `Aider Usage Queries`（Aider 使用查询）
> - `Model and Mode Configuration`（模型与模式配置）
> - `File Handling in Aider`（Aider 中的文件处理）
> - `Architect Mode Feedback`（Architect 模式反馈）
> - `Performance Optimizations`（性能优化）

- **关于 Aider 模型使用的困惑**：用户对在 Aider 中切换模式（特别是使用 `architect` 或 `code` 模式）时需要每次切换模型表示担忧。
  
  - 已澄清：弱模型（weak model）仅适用于生成 commit 消息，用户必须使用 `/model` 手动切换主模型。
- **文件处理限制**：有报告称 Aider 无法递归匹配 shell glob 模式，需要手动添加每个 \*.erb 文件的路径。
  
  - 鼓励用户为复杂的文件结构创建包装脚本（wrapper scripts）或别名（aliases），以简化流程。
- **增强 Architect 模式的建议**：反馈指出 Aider 在 Architect 模式下经常无法完全完成任务，需要用户干预并输入 'continue'。
  
  - 用户询问其他人是否遇到过类似问题，以及是否采用了不同的使用方式。
- **使用 LLM Proxy 配置 Aider**：一名用户表示，尽管尝试了各种配置文件，但在配置 Aider 仅与公司的 LLM proxy 服务配合工作时遇到了困难。
  
  - 他们提到使用了 `--openai-api-base` 和 `--openai-api-key` 参数，但仍然遇到模型不可用的问题。
- **多线程与性能担忧**：一名用户询问将 Aider 设为多线程是否会提高性能，反映了对响应时间的担忧。
  
  - 回复建议，由于 Aider 的运行模型依赖于 LLM 输入的阻塞操作，因此可能无法从多线程中受益。

**提到的链接**：

- [FAQ](https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat)：关于 aider 的常见问题解答。
- [FAQ](https://aider.chat/docs/faq.html#can-i-use-aider-with-multiple-git-repos-at-once)：关于 aider 的常见问题解答。
- [Chat modes](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model)：使用 chat、ask 和 help 聊天模式。
- [In-chat commands](https://aider.chat/docs/usage/commands.html#keybindings)：使用 /add、/model 等聊天内命令控制 aider。
- [princeton-nlp/SWE-bench_Multimodal · Datasets at Hugging Face](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Multimodal)：未找到描述

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1293300204375773326) (7 条消息):

> - `OpenAI Realtime Console`
> - `Cline AI Assistant v2.0`
> - `Firefox Security Update`

- **OpenAI Realtime Console 实现便捷的语音 API 访问**：**OpenAI Realtime Console** 的演示仓库已成功搭建，方便用户轻松测试在 [DevDay](https://simonwillison.net/2024/Oct/2/not-digital-god/#gpt-4o-audio-via-the-new-websocket-realtime-api) 上宣布的新 Realtime 语音 API。该设置仅需简单的 `npm start` 即可在本地运行应用程序。
  
  - 用户可以通过语音输入和输出进行交互；但请注意，测试会产生费用，据一位用户报告，仅使用 15 分钟就产生了 **$3.87** 的费用。
- **Cline AI Assistant 2.0 迎来重大升级**：新发布的 **Cline**（原 Claude Dev）v2.0 引入了诸如将响应直接流式传输到编辑器以及用于任务管理的取消按钮等功能。新的基于 XML 的 tool calling 提示词减少了约 **40%** 的请求，提高了资源效率。
  
  - 一位社区成员对 Cline 赞不绝口，称其**超级无敌好用！**，并强调了它在各种用例中强大的性能提升。
- **Firefox 因安全漏洞发布关键更新**：**Firefox** 宣布了一个**严重**漏洞，敦促用户更新到 `131.0.2` 版本，以缓解与 use-after-free 漏洞相关的潜在风险。Mozilla 发布的这份公告表明该漏洞正被积极利用，具体细节见 [Mozilla 公告](https://www.mozilla.org/en-US/security/advisories/mfsa2024-51/)。
  
  - 用户对这一严重安全风险的提醒表示感谢，并强调了为了安全立即更新的重要性。

**提到的链接**：

- [openai/openai-realtime-console](https://simonwillison.net/2024/Oct/9/openai-realtime-console/)：我今天成功运行了这个 OpenAI 演示仓库——这是开始尝试他们在 DevDay 宣布的新 Realtime 语音 API 的一种极其简单的方式...
- [Security Vulnerability fixed in Firefox 131.0.2, Firefox ESR 128.3.1, Firefox ESR 115.16.1](https://www.mozilla.org/en-US/security/advisories/mfsa2024-51/)：未找到描述
- [Tweet from Saoud Rizwan (@sdrzn)](https://x.com/sdrzn/status/1843989769828602273)：介绍 Cline（原 Claude Dev），一个可以使用你的 CLI 和编辑器的 AI 助手。v2.0 带来了令人兴奋的更新：响应现在流式传输到你的编辑器中，增加了取消按钮以实现更好的控制...
- [Release v2.0.0 · clinebot/cline](https://github.com/clinebot/cline/releases/tag/v2.0.0)：新名称：认识 Cline，一个可以使用你的 CLI 和编辑器的 AI 助手。虽然 “Claude Dev” 是对 Claude 3.5 Sonnet 的致敬，但 v2.0 带来的更新显著提高了在其他模型上的性能...

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1293511284356022282) (44 条消息🔥):

> - `2024年诺贝尔化学奖`
> - `Nato 的学术背景`
> - `LMSYS 转型为公司`
> - `编辑 Google Scholar`
> - `能源科学中的挑战`

- **诺贝尔化学奖授予计算领域的创新者**：瑞典皇家科学院将 2024 年 #NobelPrize 化学奖授予 David Baker，以表彰其在 **计算蛋白质设计** 方面的贡献；并共同授予 Demis Hassabis 和 John M. Jumper，以表彰其在 **蛋白质结构预测** 方面的成就。
  
  - 成员们用感叹的反应表达了兴奋之情，标志着科学界的一次回归。
- **Nato 从 EE 到 AI 的历程**：Nato 分享了他的学术路径，在一家 **新型机器人实验室** 工作后，从 **电子工程** (EE) 转向了 AI 和强化学习。
  
  - 他还讨论了过去在 **MEMS** 方面的经验，强调了它们在工程中的复杂性和重要性。
- **LMSYS 宣布转型为公司**：Nato 提到 **LMSYS** 正计划转型为一家公司，他对此持积极态度，认为这可能会克服学术环境中的 **不良激励机制**。
  
  - 讨论中涉及了这种发展是否优于 **非营利性** 的学术动机。
- **编辑 Google Scholar 以优化引用指标**：一位用户询问 Nato 是否计划在提到精心整理个人资料后，制作一个编辑 Google Scholar 的教程。
  
  - Nato 指出 Google 的用户体验极具挑战性，手动编辑是一个 *痛苦的过程*。
- **关于电池开发挑战的讨论**：成员们确认了电池开发中面临的困难，表示 *研发电池真的非常困难*。
  
  - 讨论中指出了数据科学与能源科学之间的对比，一些人认为数据科学明显更 **容易**。

**提到的链接**：

- [来自诺贝尔奖 (@NobelPrize) 的推文](https://x.com/NobelPrize/status/1843951197960777760)：突发新闻：瑞典皇家科学院决定将 2024 年 #NobelPrize 化学奖的一半授予 David Baker “以表彰其在计算蛋白质设计方面的贡献”，另一半共同授予……
- [MEMS - 维基百科](https://en.wikipedia.org/wiki/MEMS)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1293310250396422195) (27 条消息🔥):

> - `ML 中的 PRMs/验证器`
> - `ML 领域的专利`
> - `PRMs 的替代方案`
> - `ML 研究的透明度`

- **PRM 研究的稀缺性**：成员们讨论了关于 **PRMs** 研究的匮乏，其中一人幽默地指出，“关于 PRMs 的研究几乎没有，而关于 LLM as a judge 的研究却多达十亿”。
  
  - 其他人表达了寻找优秀论文的兴趣，这表明了对其当前效用的困惑。
- **ML 领域的专利困惑**：讨论转向了 **专利** 在机器学习领域运作方式，观点认为公司申请专利是为了防御，但往往由于 **模糊性** 而导致专利无效。
  
  - 讨论中对追究几乎无法证明的侵权行为所带来的财务负担表示担忧，将其比作一桩“苦差事”。
- **PRMs 替代方案的出现**：人们对哪些方法正在取代 PRMs 感到好奇，有迹象表明大型实验室仍在使用它们，但其重要性正在减弱。
  
  - 讨论指出，在确定性输出上进行 **强化学习** 可能足以应对，而无需 PRMs 的复杂性。
- **探索 O1 的功能**：在 **O1** 发布的背景下，成员们质疑什么将填补 PRM 的角色，强调了在推理树探索过程中需要某种形式评分的担忧。
  
  - 尽管对 PRMs 的必要性看法不一，但提到了来自 John Schulman 等权威人士的见解作为一种保证。
- **倡导透明度**：Nathan Lambert 倡导在 ML 研究过程中提高 **透明度**，断言这比保持秘密更简单。
  
  - 这一观点得到了小组的共鸣，推断公开分享方法论可能会带来更高效的执行。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1293475621531553852) (16 messages🔥):

> - `涉及知名人士的 AI 风险讨论`
> - `AI 研究中的诺贝尔奖争议`
> - `ICLR 2025 审稿流程变更`
> - `Schmidhuber 对 AI 归因问题的批评`
> - `社交媒体对学科相关见解的反应`

- **学术界对 AI 风险的认知**：关于 **Geoff Hinton** 和 **Yoshua Bengio** 留在加拿大的动机的讨论，突显了他们对 AI 治理观点背后的个人历史。
  
  - *一位用户评论道：“当听到他们告诉加州应该如何应对 AI 风险时，请记住这一点。”*
- **Schmidhuber 抨击诺贝尔奖评选**：针对 **2024 年诺贝尔物理学奖** 的批评浮出水面，声称 Hinton 及其合作者的作品存在**剽窃**和归因错误，特别是涉及 Amari 的贡献。
  
  - Schmidhuber 认为 AI 领域重要的历史贡献被忽视了，并宣称这次评选更多是为了放大知名人士的影响力，而不是表彰原始创新者。
- **ICLR 2025 引入审稿反馈 Agent**：面对激增的投稿量，**ICLR 2025** 会议旨在通过**反馈 Agent** 来提升审稿质量，该 Agent 旨在引导审稿人进行更一致、更具建设性的评估。
  
  - 这一举措凸显了投稿量迅速增加带来的挑战，旨在缓解过去审稿中出现的质量差异。
- **对 Schmidhuber 见解的复杂反应**：用户对 **Schmidhuber** 关于 AI 归因的直率批评表达了不同的看法，一些人对他关于历史贡献重要性的观点表示赞同。
  
  - *正如一位用户所说，“平心而论，他经常言之有理”，这反映了 Schmidhuber 在持续讨论中的影响力。*
- **对审稿流程变更的担忧**：由于 ICLR 同行评审流程可能发生变动，引发了对潜在争议的担忧，强调了社区对变化的敏感性。
  
  - “审稿流程的任何改变都会引发争议”这一观点突显了会议参与者中普遍存在的忧虑。

**提及的链接**：

- [来自 undefined 的推文](https://vxtwitter.com/JMannhart/status/1843831370352865711)：未找到描述
- [来自 Pedro Domingos (@pmddomingos) 的推文](https://x.com/pmddomingos/status/1839744686073991466?t=xjxBZFEvlcITtC1aJDUgsw&s=19)：Geoff Hinton 留在加拿大是因为他无法忍受 Reagan 担任总统。而 Yoshua Bengio 留在加拿大是因为他拒绝在法国军队服役。当听到他们告诉……时，请记住这一点。
- [Assisting ICLR 2025 reviewers with feedback – ICLR Blog](https://blog.iclr.cc/2024/10/09/iclr2025-assisting-reviewers/)：未找到描述
- [来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文](https://x.com/SchmidhuberAI/status/1844022724328394780)：给 Hopfield 和 Hinton 的 #NobelPrizeinPhysics2024 奖项奖励了计算机科学领域的剽窃和错误归因。这主要关乎 Amari 的 “Hopfield 网络” 和 “Boltzmann...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1293659354062393408) (16 messages🔥):

> - `ButtBench Alignment Project`
> - `AI2 的 SuperAlignment 负责人`
> - `行业 vs 社区`
> - `Lucas Beyer 的 PR 策略`
> - `Allennlp 账号管理`

- **ButtBench Alignment Project 发布令人兴奋的 Logo**：关于 **ButtBench Alignment Project** 分享了一个*令人兴奋的更新*，宣布创建了新的 Logo。
  
  - Luca Soldaini 指出，虽然该项目达到了 **SOTA**，但距离**人类表现 (human performance)** 仍有很长的路要走。
- **Natolambert 担任 SuperAlignment 负责人**：Natolambert 宣布职位变更为 **AI2 的 SuperAlignment 负责人**，标志着进入了新的领导岗位。
  
  - 这一变化凸显了其在 AI 社区日益增长的影响力，强调了对传统行业规范的背离。
- **关于行业规范和 Shitposting 的讨论**：有一场关于 **People In Industry™️**（业内人士）能否自由进行 **shitposting** 的幽默讨论。
  
  - Natolambert 指出，尽管存在一些行业惯例，但他们认为自己处于这些典型界限之外。
- **Lucas Beyer 险些翻车的 PR 帖子**：Lucas Beyer 被提及为 **GDM** 的重要 PR 账号，因其发帖风格*非常大胆（close to the sun）*而受到关注。
  
  - 尽管如此，成员们承认他了解*限度*，确保了他的职位依然稳固。
- **管理 Allennlp 账号**：Natolambert 幽默地提到现在由他运营 **Allennlp 账号**，展示了与社区的积极互动。
  
  - 他表示，与传统面试的挑战相比，在 Twitter 上进行互动要轻松得多。

**提到的链接**：

- [Cody Blakeney (@code_star) 的推文](https://x.com/code_star/status/1844098524985819241)：很高兴看到 @soldni 出现在大屏幕上
- [Luca Soldaini 🎀 (@soldni) 的推文](https://x.com/soldni/status/1844099747415720107)：令人兴奋的更新：我们现在为 ButtBench Alignment Project 设计了 Logo。引用 Luca Soldaini 🎀 (@soldni) 的 ButtBench 更新：o1-preview 虽然非常困难但达到了 SOTA；但我们距离人类水平还很远...

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1293424537345593395) (3 messages):

> - `AI 中的数据墙 (Data Wall)`
> - `AI 开发的暴力破解方法`
> - `人类推理 vs AI 数据需求`
> - `AI 模型的效率`

- **数据墙威胁 AI 进展**：当前的语言模型正接近可用文本数据的极限，引发了对可能阻碍进展的潜在**“数据墙” (data wall)** 的担忧。
  
  - 许多人认为这不会成为问题，因为尽管人类接触的语言数据较少，但拥有更卓越的语言技能。
- **建议在 AI 开发中使用暴力数据策略**：为了加速 AI 发展，一些人建议将**暴力 (brute-force)** 数据使用的概念追溯到 2005 年，而不是仅仅关注 **Attention** 或 **Transformers**。
  
  - 这一概念强调了一种信念，即增加数据量可能是克服 AI 训练挑战的关键。
- **关于 AI 效率的哲学观点**：一场关于当前 AI 方法论哲学意义的讨论展开了，重点强调了 AI 模型需要变得**更高效**。
  
  - 一位参与者承认，虽然讨论偏向哲学，但 AI 策略的效率仍然至关重要。

 

**提到的链接**：[真正的数据墙是数十亿年的进化](https://dynomight.substack.com/p/data-wall)：谨慎对待那些人类类比

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1293551551553601566) (15 条消息🔥):

> - `RoboNato vs RealNato`
> - `针对内容的 OLMo fine-tuning`
> - `NotebookLM 概念`

- **关于 RoboNato 与 RealNato 的辩论**：一位成员表示怀念 **RoboNato**，但认为 **RealNato 的配音** 可能更出色，这促使其他人思考两者同时存在的意义。
  
  - 一位参与者建议：*“两者都有会让人非常困惑”*，而另一位则提议将 RoboNato 留给特殊项目。
- **RoboNato 特别节目**：有人建议制作特别节目，让参与者与 **RoboNato** 对话，同时使用 **OLMo finetune** 来增强文案。
  
  - 一位成员幽默地表示，希望能有时间与 RoboNato 一起开启一段 *“疯狂的 AI YouTuber 之旅”*。
- **RoboNato 的高级订阅福利**：有人提出了将 **RoboNato** 作为高级订阅者福利的想法，以此增加用户参与的动力。
  
  - 这引发了一场关于 **NotebookLM** 概念的趣味讨论，即由两个 RoboNato 来回顾来自社区的消息。
- **对 AI 对话的幽默反应**：一位成员幽默地评论了被自己的 AI 声音调情的古怪感，强调了此类互动可能带来的复杂性。
  
  - 虽然这些互动的设置被认为相对简单，但结果可能会让人感到 *“???”*。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1293294200753885360) (100 messages🔥🔥):

> - `Perplexity AI 盈利能力`
> - `针对 Perplexity 的 Complexity 扩展程序`
> - `Perplexity AI 回答的变化`
> - `Collections 和 Spaces 的未来`
> - `Pro 功能的访问权限`

- **关于 Perplexity AI 盈利模式的疑问**：讨论集中在 **Perplexity AI** 如何实现盈利，特别是在提供*学生折扣*的情况下，这引发了对其商业模式的担忧。
  
  - *sneakyf1shy* 幽默地表示，这一切全靠 **Venture Capital**，强调了其长期目标的不确定性。
- **对 Complexity 扩展增强功能的狂热**：**Complexity** 扩展被描述为能通过自定义主题和 Markdown 导出选项等功能，极大地增强 Perplexity 的使用体验。
  
  - 社区指出它“就像加强版的 Perplexity”，增强了用户交互性，同时 **feline** 和 *asura0_00* 强调了它的实用性。
- **Perplexity AI 简练的回答**：成员们讨论到 Perplexity AI 的回答变得更加**简练**，并对回答变短、信息量减少表示担忧。
  
  - 一些人推测这可能与 **Token 限制**的变化有关，影响了所提供回答的深度。
- **对改进 Collections 和 Spaces 的期待**：有关从“Collections”迁移到“Spaces”的更新，旨在提高平台的用户体验和生产力。
  
  - 用户希望看到诸如增加 Prompt 限制以及更好地集成到搜索过程等增强功能。
- **Pro 功能和 API 能力**：用户询问 Pro 账户是否能获得特定功能的访问权限，例如 **o1 model** 以及 **o1-mini** 的限制。
  
  - 回复尚不明确，引发了关于未来潜在功能及其如何影响用户体验的讨论。

**提到的链接**：

- [Denis Yarats (@denisyarats) 的推文](https://x.com/denisyarats/status/1844074889755656280?s=61)：显然我一直在为 Jensen Huang 开发个人导师 🤣
- [Perplexity 通过 Spaces 和模型设置重新定义 Collections](https://www.testingcatalog.com/perplexity-redefines-collections-with-spaces-allowing-default-model-settings/)：Perplexity 正在推出 Spaces，通过分离 Collections 并允许设置默认模型来提升 UX 和生产力。未来的更新将包括基于知识库的支持和文件搜索。
- [Dance Dancing GIF - Dance Dancing Indian Dance - Discover & Share GIFs](https://tenor.com/view/dance-dancing-indian-dance-gif-15425444)：点击查看 GIF
- [TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1842635276780261816?s=46)：进行中 🚧：Perplexity 将把 Collections 从其 Library 中提取到一个名为 Spaces 的独立类别中！此举将简化 UX 并提高“Spaces”的使用率。目前的 Collections 非常...
- [Ted Lasso Awkward GIF - Ted Lasso Awkward Side Eye - Discover & Share GIFs](https://tenor.com/view/ted-lasso-awkward-side-eye-look-around-what-to-do-gif-17319394476006190959)：点击查看 GIF
- [Complexity - 增强版 Perplexity.ai - Chrome 网上应用店](https://chromewebstore.google.com/detail/complexity-perplexityai-s/ffppmilmeaekegkpckebkeahjgmhggpj)：⚡ 极大地增强你的 Perplexity.ai
- [Sama Sam Altman GIF - Sama Sam altman Openai - Discover & Share GIFs](https://tenor.com/view/sama-sam-altman-openai-sama-yapping-yapping-gif-7525532358145568607)：点击查看 GIF
- [Foodpanda Sauce GIF - Foodpanda Food Panda - Discover & Share GIFs](https://tenor.com/view/foodpanda-food-panda-sauce-dip-gif-17675920)：点击查看 GIF

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1293320754770743426) (12 messages🔥):

> - `Meta's Movie Generator`
> - `Nobel Prize in Physics`
> - `AI Automation in Ports`
> - `AI Evaluation by Braintrust`
> - `2024 Summer Olympics`

- **Meta 发布电影生成工具**：Meta 推出了一款新的 [电影生成工具](https://www.perplexity.ai/page/meta-unveils-movie-gen-rj3GtxbAQditnyIXKX6Ofw)，允许用户使用 AI 创作短片。
  
  - *这一进展旨在通过 AI 技术增强创意叙事能力*。
- **诺贝尔奖授予 AI 贡献者**：[Hopfield 和 Hinton](https://www.perplexity.ai/page/hopfield-and-hinton-win-nobel-Vfdtu_msRiCY.I6TCrgSag) 因其在 AI 领域的重大贡献被授予诺贝尔物理学奖。
  
  - 他们的研究为神经网络和 Machine Learning 的进步铺平了道路。
- **广州港实现全自动化**：广州港现已实现 [全自动化](https://www.perplexity.ai/page/china-s-guangzhou-port-automat-pPjhhjQxRf.uDKuFhzK1fQ)，展示了 AI 在物流和自动化领域的影响。
  
  - *专家强调，在此类运营中采用 AI 对未来的效率至关重要*。
- **Braintrust AI 在评估方法中占据主导地位**：正如 [最近的讨论](https://www.perplexity.ai/page/ai-startup-raises-millions-Qj2NHRJrS0mWOKTFZr1.0w) 中提到的，Braintrust AI 目前在 AI 评估技术方面处于领先地位。
  
  - 他们的这种方法被业界公认为具有创新性和有效性。
- **期待 2024 夏季奥运会**：即将到来的 [2024 夏季奥运会](https://www.perplexity.ai/search/summer-olympics-2024-Fw.BijKwQQikFyz.Vf.HRQ) 引发了热议，特别是关于可能使用的 AI 技术进步。
  
  - 筹备工作正在进行中，重点关注 AI 如何增强运动员和粉丝的体验。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/B_qZOHy_1F8)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1293476777657765922) (4 messages):

> - `Citation API Whitelisting`
> - `API Credit Purchase Issues`
> - `Invoice Company Details Update`
> - `Declined Card for API Access`

- **对 Citation API 白名单申请的挫败感**：一位成员对 **Citation API** 的白名单申请表示担忧，指出通过邮件、表格和帮助台发送的多次请求均未得到回复。
  
  - *目前尚未提供任何更新*，导致挫败感日益增加。
- **API 额度支付持续失败**：一位用户报告了在尝试购买 **API credits** 时遇到的问题，称尝试失败且没有错误消息，仅显示 **$XX pending** 状态并很快消失。
  
  - 他们注意到唯一可用的支付方式是 **credit card**，并对其他选项提出了疑问。
- **发票需要包含公司详情**：一位成员表示，他们的发票默认显示其 **Google email**，需要更新以反映其公司名称和地址，这带来了麻烦。
  
  - 他们正在寻求有关如何进行此更改的指导。
- **API 访问卡被拒问题**：一位成员分享了在尝试使用 API 时 **卡被拒绝** 的挫败感，即使是 **$0.00** 的扣款也是如此。
  
  - 他们正在寻找卡片支付失败的潜在原因，目前尚不清楚。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1293296937751019562) (108 条消息🔥🔥):

> - `ControlNet Models`
> - `Flux Inpainting`
> - `Kaggle Notebooks for Automatic1111`
> - `Distilled CFG Explained`
> - `Deforum Usage Alternatives`

- **ControlNet 模型详解**：一名成员询问了关于 **ControlNet 模型**的问题，促使另一名成员分享了一个包含信息和示例的 [GitHub 链接](https://github.com/lllyasviel/ControlNet)以供探索，并建议跳过数学内容向下滚动。
  
  - *向下滚动一点，忽略数学部分，看看示例。*
- **Flux 局部重绘 (Inpainting) 性能**：关于 **Flux** 和 **Schnell** 局部重绘模型的讨论兴起，一名成员指出，在性能尚可的 GPU 上应该花费约 1-2 分钟，而不是另一名成员所经历的 25 分钟。
  
  - **Flux dev** 和 **Schnell** 之间迭代次数的主要差异源于它们的性能和用途。
- **对 Automatic1111 Kaggle Notebook 的需求**：一名成员请求获取用于使用 **Automatic1111** 的 **Kaggle notebook**，强调了对面向图像生成技术的资源的需求。
  
  - 其他人也加入讨论，指出在寻找特定 notebook 以简化流程方面存在挑战。
- **理解 Distilled CFG**：围绕 **distilled CFG** 及其影响产生了困惑，讨论强调它与标准的 CFG 不同，是作为模型训练建立的一种引导形式。
  
  - 社区澄清了 Flux dev 如何简化 CFG 的使用，但目前缺乏对负面提示词 (negative prompts) 的支持。
- **在 Google Colab 限制后使用 Deforum**：在注意到 Colab 的限制后，一名成员询问如何免费使用 **Deforum**，从而引发了关于为此目的租用 GPU 的建议。
  
  - [RunPod](https://www.runpod.io/) 等资源被推荐作为获取必要算力的替代方案。

**提到的链接**：

- [RunPod - The Cloud Built for AI](https://www.runpod.io/)：为 AI 构建的云。在统一的云端开发、训练和扩展 AI 模型。通过 GPU Cloud 按需启动 GPU，通过 Serverless 扩展机器学习推理。
- [Invoke AI 5.0 Tutorial - From Beginner to Pro in Minutes! Part 2 Let's Get Creative!](https://youtu.be/cx7L-evqLPo?si=Lzk6QWYGmY2pnzRT)：Invoke AI 5.0 教程 - 几分钟内从入门到精通！第 2 部分：让我们开始创意！本视频将介绍一些基础的图像创建，从生成、图像...
- [GitHub - lllyasviel/ControlNet: Let us control diffusion models!](https://github.com/lllyasviel/ControlNet)：让我们控制扩散模型！通过在 GitHub 上创建账户为 lllyasviel/ControlNet 的开发做出贡献。
- [Flux ControlNet - Easy Install Guide](https://www.youtube.com/watch?v=HVYXM9bPFTs&ab_channel=OlivioSarikas)：Flux ControlNet - 简单的设置方法。包含 3 个工作流。在这里获取我的 3 个工作流：https://www.patreon.com/posts/flux-controlnet-110607421 我的 Flux Ins...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1293304527608352769) (86 messages🔥🔥):

> - `AI 和化学领域的诺贝尔奖`
> - `博士课程竞争与指标`
> - `Web3 与 Web5 的讨论`
> - `论文发表与研究合作`
> - `国际象棋与 AI 的当前话题`

- **围绕 AI 和化学诺贝尔奖的争议**：最近的诺贝尔奖授予了 Hinton 和 Hopfield 等 AI 领域的人物，引发了关于其对物理和化学领域影响力的讨论，意见分歧较大。
  
  - 一位成员强调，如果奖项奖励的是一个领域的领导者，可能会稀释奖项本身的声望；而另一位成员则反驳称，热情和创新应该是关键的选择标准。
- **博士项目竞争与研究指标**：一位成员对过分强调发表指标表示沮丧，认为这为有志攻读博士的候选人营造了一种竞争激烈且令人畏缩的氛围。
  
  - 观点各异，有人建议，在寻求未来的合作和导师指导时，建立人脉（Networking）可能比单纯追求论文发表数量更有效。
- **Web3 向 Web5 的演进**：成员们讨论了从 Web3 到 Web5 的过渡，指出这种命名策略似乎更像斐波那契数列，而非逻辑演进。
  
  - 对话转向了轻松的玩笑，包括对未来发展的调侃，例如推测 Web8 将由之前版本的混合产生。
- **研究合作与 H-Index 指标**：关于在建立具有竞争力的 H-Index 时，合作的价值与研究产出质量之间的关系展开了辩论，一些人警告不要只关注数量。
  
  - 成员们承认，虽然有影响力的研究可以推动职业生涯，但为了提高指标而频繁发表论文的压力仍然是一个系统性问题。
- **国际象棋与知名人物**：提到了 FIDE 国际象棋奥林匹克赛，引发了关于 Demis Hassabis 等知名人物及其与包括国际象棋在内的各个社区联系的讨论。
  
  - 成员们对国际象棋和 AI 之间兴趣的交叉融合表示惊讶，说明了 AI 领域的人物通常在不同领域都拥有显赫地位。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/polycarpweb5?lang=en)：未找到描述
- [GitHub - xjdr-alt/entropix: Entropy Based Sampling and Parallel CoT Decoding](https://github.com/xjdr-alt/entropix)：基于熵的采样与并行 CoT 解码。通过在 GitHub 上创建账号为 xjdr-alt/entropix 的开发做出贡献。
- [Reddit - 深入了解任何事物](https://www.reddit.com/r/chess/comments/1fzre62/cm_demis_hassabis_formerly_the_world_no_2_among/)：未找到描述

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1293305646229225675) (7 messages):

> - `模型中的权重归一化`
> - `梯度初始化技术`
> - `梯度下降中的幂律`

- **关于权重归一化时机的澄清**：讨论集中在是在**前向传播（forward pass）**期间进行权重和嵌入的归一化，还是直接修改权重本身。
  
  - 此外，成员们还辩论了归一化应该发生在**任何数据传递之前**还是**第一次传递之后**，以寻求最佳实践的明确指导。
- **权重初始化的经验方法**：关于在不重新推导 **MuP** 的情况下改进权重初始化的疑问，突显了从预训练架构进行上采样或下采样的潜在有效性。
  
  - 一位成员建议直接从**预训练模型**初始化权重的简单方法，并强调了实际应用意义。
- **关于优化中幂律的见解**：一位成员分享了一个 [Twitter 线程](https://x.com/yaroslavvb/status/1843758350171099468)，解释了梯度下降中**幂律（power-laws）**的出现，并将其与现实世界的行为与理论模型联系起来。
  
  - 该线程引用了 Francis Bach 关于优化缩放法则（scaling laws）的工作，提供了关于梯度下降加速的**经验性见解**。

**提到的链接**：

- [来自 Yaroslav Bulatov (@yaroslavvb) 的推文](https://x.com/yaroslavvb/status/1843758350171099468)：很高兴看到 Francis Bach 关注梯度下降的真实行为。这与优化文献传统研究的“假设”行为形成鲜明对比。引用...
- [证明当 k 较大时 $\\sum_{i=1}^k i^{-2}(1-i^{-2})^t\\approx \\frac{\\sqrt{\\pi }}{2 \\sqrt{t }}$](https://math.stackexchange.com/a/4981650/998)：对于较大的 $k$，我观察到以下现象：$$f(t)=\\sum_{i=1}^k i^{-2}(1-i^{-2})^t \\approx \\frac{\\sqrt{\\pi }}{2 \\sqrt{t }}$$ 证明这一点的最简单方法是什么？Notebook

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1293563684903456790) (6 messages):

> - `Scaling Laws Overview`
> - `Kaplan's Scaling Laws`
> - `Data and Model Size Relationship`

- **Scaling Laws 概览引发辩论**：一位成员分享了一份概览，指出**交叉熵损失随计算量的平方级增加而减少**，并基于[这篇文章](https://www.interconnects.ai/p/how-scaling-changes-model-behavior)提出了*平方根缩放（square root scaling）*。
  
  - 另一位成员对此提出挑战，指出 Kaplan 的定律建议常数为 **0.28**，因此更倾向于*四次方根缩放（fourth-root scaling）*。
- **对 Kaplan 相关性的质疑**：随后展开了关于 Kaplan 定律是否过时的讨论，一位成员表示它已经**过时**，但它与 Chinchilla 似乎在某些缩放方面达成了一致。
  
  - 有人提到 **L(N, D)** 的变化大约与 **N^-0.5** 和 **D^-0.5** 成正比，其中 **C = 6ND**。
- **模型大小的考量**：一位成员询问当模型规模已经很大，且通过增加数据或步数进行调整（本质上是进行**少于 1 个 epoch 的训练**）时，**D^0.5** 如何适用。
  
  - 他们表示需要将其与 **0.25 缩放**对齐，以匹配他们的数学计算。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1293533538888716320) (3 messages):

> - `0-shot COT model releases`
> - `Evaluation implementation details`
> - `JAX libraries and implementations`

- **0-shot COT 模型成为焦点**：讨论强调了在模型发布中一致使用 **0-shot COT 变体**的情况，这表明了评估方法论的一种潜在趋势。
  
  - 然而，关于**评估实现（evaluation implementation）**的具体细节并未分享。
- **探索 JAX 库**：有人建议在代码中移除对 **torch** 使用的假设，转而寻找 **JAX** 中的替代方案。
  
  - 成员们思考哪些**规范库或实现**可能对采用更有利。

 

---

### **Eleuther ▷ #**[**multimodal-general**](https://discord.com/channels/729741769192767510/795089627089862656/) (1 messages):

tensor_kelechi: 最好的轻量级 VLM 有哪些？

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1293311257608323173) (7 messages):

> - `HBM vs SRAM scaling`
> - `3D Stacking Solutions`
> - `Memory Architecture in AI`
> - `Manufacturing Difficulties`
> - `Rotary Embeddings CUDA Kernel`

- **HBM 性能与预期对比**：有人对 **HBM** 的表现未达到最初预期表示担忧，它在 **H100** 等产品中仍占据**巨大**的成本比例，且与 **LPDDR5** 相比并未显著降低功耗。
  
  - 供应更多 **H100** 的主要瓶颈被确定为所需的**封装（packaging）**。
- **SRAM 缩放问题令业界意外**：出乎意料的是，**SRAM** 的缩放速度相对于逻辑电路有所放缓，这给 **Graphcore** 带来了显著的设计挑战，而在他们 2015 年左右做出设计选择时，这些挑战很难预测。
  
  - *正如一位成员所说*，“当时没有任何会议能让你预见到”这一发展。
- **3D 堆叠作为缓解策略**：展望未来，提出的解决方案涉及 **3D 堆叠**，如 **MI300X** 中所见，将处理器堆叠在采用旧工艺制造的基础晶圆（base dies）上，以实现高效的资源分配。
  
  - 这种方法允许将 SRAM 和 I/O 从前沿工艺晶圆中移出，从而促进 **3nm** 和 **2nm** 等先进节点上更好的逻辑缩放。
- **理解内存技术的困难**：一位成员分享了他们学习 **DRAM** 和 **HBM** 之间差异的过程，利用了 **Claude** 以及 Asianometry 标题为《为 AI 革命提供动力的特殊内存》的视频等资源。
  
  - 他们强调了理解制造过程和难点的重要性，特别是关于晶圆键合（die bonding）方面。
- **关于 Rotary Embeddings CUDA Kernel 的询问**：有人请求一个专门用于计算 **rotary embeddings** 逆频率的 **CUDA kernel**，这反映了对更具体技术资源的需求。
  
  - 这体现了对针对特定 AI 应用优化实现的持续关注。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1293463440853172315) (4 messages):

> - `Triton source files` (Triton 源文件)
> - `GitHub repository structure` (GitHub 仓库结构)

- **寻找 Triton MatMul 源文件**：用户正在寻找 `triton.ops.blocksparse.matmul.matmul` 的源文件，由于难以定位，请求提供 GitHub 链接。
  
  - 另一位成员指出，所需文件可以在 Triton 仓库的 [python/triton/ 目录](https://github.com/triton-lang/triton/blob/5b29da719daeb3566bfc95b7d02f3561e505bcaf/python/triton/ops/blocksparse/matmul.py#L582) 中找到。
- **Triton 仓库的变化**：用户询问为何 main 分支中缺少 MatMul 文件，怀疑其是否已被迁移或转换。
  
  - 回复的成员表示对迁移情况不确定，承认自己从未为 Triton 贡献过代码，但意识到有必要这样做。

 

**提及的链接**：[triton/python/triton/ops/blocksparse/matmul.py at 5b29da719daeb3566bfc95b7d02f3561e505bcaf · triton-lang/triton](https://github.com/triton-lang/triton/blob/5b29da719daeb3566bfc95b7d02f3561e505bcaf/python/triton/ops/blocksparse/matmul.py#L582)：Triton 语言和编译器的开发仓库 - triton-lang/triton

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1293605166272086098) (1 messages):

> - `PyTorch API changes` (PyTorch API 变更)
> - `torch._dynamo migration` (torch._dynamo 迁移)
> - `GitHub issue suggestions` (GitHub issue 建议)

- **PyTorch API 升级困扰**：一位成员在升级到最新 **PyTorch** 版本时遇到困难，注意到 `torch._dynamo.allowed_functions` 已被新 API **取代（superceded）**。
  
  - 他们正在追溯 [Git 历史](https://github.com/ACEsuit/mace/blob/118a514efde34d963666118ce45360e94d648ef5/mace/tools/compile.py#L39) 以了解正确的迁移路径，并就未记录的 API 替换寻求建议。
- **寻求帮助或 GitHub Issue 指导**：该成员不确定在此讨论迁移问题是否合适，还是应该开一个 **GitHub issue**。
  
  - 他们征求关于解决所面临的 API 替换挑战的策略或资源的建议。

 

**提及的链接**：[mace/mace/tools/compile.py at 118a514efde34d963666118ce45360e94d648ef5 · ACEsuit/mace](https://github.com/ACEsuit/mace/blob/118a514efde34d963666118ce45360e94d648ef5/mace/tools/compile.py#L39)：MACE - 使用高阶等变消息传递的快速准确的机器学习原子间势能模型。 - ACEsuit/mace

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/) (1 messages):

vayuda: 搭载 M 系列芯片的 Mac 是否使用 ARM SVE 指令？

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1293622924502237205) (3 messages):

> - `5th Edition Release` (第 5 版发布)
> - `Special Offers for Existing Users` (现有用户的特别优惠)

- **第 5 版发布令粉丝兴奋**：一位成员对 **第 5 版** 的发布表示兴奋，提到他们仍保留着第一版发布时购买的副本。
  
  - 这反映了成员们在回忆最初购买经历时对该系列的持续关注。
- **咨询升级折扣**：一位成员询问是否为已拥有旧版本的用户提供特别优惠。
  
  - 另一位成员表示不确定，回答说 *Idk*（不知道），凸显了关于潜在升级折扣信息的匮乏。

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1293384179140263966) (39 条消息🔥):

> - `torchao 与 ComfyUI 的集成`
> - `Float8 量化性能`
> - `FSDP2 中的行级与列级缩放`
> - `Windows 上的量化问题`
> - `torch.inference_mode 的局限性`

- **torchao 在集成 ComfyUI 时遇到困难**：一位用户在为 ComfyUI 启用 `torchao` 时遇到了与算子（operator）相关的问题，特别是在 `torch.inference_mode()` 内部使用 `quantize_` 函数时。
  
  - 尽管尝试了 PyTorch nightly 版本并调整了模型，问题依然存在，目前尚不清楚这是否是 Windows 特有的问题。
- **Float8 量化产生意想不到的结果**：一位成员分享说，在 GPT 模型上使用 `float8_dynamic_activation_float8_weight` 将吞吐量提高了约 10%，但由于 `unwrap_tensor_subclasses` 函数遇到了延迟问题。
  
  - 讨论表明，如果使用正确的 PyTorch 版本，可能可以消除该函数，但由于工作项目的限制，准确复现仍然很困难。
- **FSDP2 中行级（Row-wise）与列级（Column-wise）缩放的混淆**：讨论强调，由于反向传播期间的权重转置，行级缩放可能无法在 FSDP2 的 backward 阶段工作，从而使正确的缩放变得复杂。
  
  - 从本质上讲，虽然行级缩放允许独立的 GPU 计算，但列级缩放面临着需要在 GPU 之间进行 all-reduce 操作的挑战。
- **Windows 上的 torchao 量化问题**：`torchao` 在 Windows 上的集成导致了算子错误，引发了关于这些问题是 Windows 固有问题还是 ComfyUI 框架问题的猜测。
  
  - 过去使用 Hugging Face 的 [optimum-quanto](https://github.com/huggingface/optimum-quanto) 的实现效果不佳，凸显了潜在的框架问题。
- **torch.inference_mode() 的局限性**：有观点指出，一旦进入 `torch.inference_mode()`，用户发现很难退出，从而导致性能受限。
  
  - 一些参与者表示，该模式在编译后提供的效用极小，并支持将此类问题反馈给特定开发人员以获取进一步见解。

 

**提及的链接**：[GitHub - huggingface/optimum-quanto: A pytorch quantization backend for optimum](https://github.com/huggingface/optimum-quanto)：一个用于 optimum 的 PyTorch 量化后端。通过在 GitHub 上创建账号来为 huggingface/optimum-quanto 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/) (1 条消息):

vayuda: 显然 Hinton 是第一位“纯计算机科学（pure cs）”诺贝尔奖得主。

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1293480848443703328) (15 条消息🔥):

> - `GPT2 训练问题`
> - `理解代码中的依赖关系`
> - `floatX 的定义与用法`
> - `有效利用 IDE 功能`

- **GPT2 训练遇到 TypeError**：一名成员报告在运行 GPT2 训练时遇到了问题，由于意外的关键字参数 'generator'，在 PyTorch 2.0.0 中收到了与 `normal_()` 函数相关的 **TypeError**。
  
  - 另一位成员建议理解训练的复杂性，包括初始化以及 forward/backward 过程。
- **floatX 定义详解**：解释指出，根据 bf16 或 fp32 的编译设置，**floatX** 被定义为 `nv_bfloat16` 或 `float`。一名成员寻求关于在哪里可以找到此定义以及如何包含它的帮助。
- **依赖管理方面的担忧**：一名成员表示在编码时管理依赖关系很困难，并表现出对引用（references）的理解不足。其他人建议只使用 **CUDA** 就足够了，cuDNN 是可选的。
- **IDE 功能的重要性**：讨论强调了 IDE 功能（如跳转到函数/类型定义）对于高效编码的价值。学习这些技能被认为对任何程序员都有益。

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1293287425782579230) (2 messages):

> - `Raspberry Pi 5`
> - `External GPU setup`
> - `amdgpu Linux kernel patch`
> - `4K gaming performance`

- **为 Raspberry Pi 5 准备 4K 游戏**：在汉诺威 Maker Faire 上看到 Pineboards 的 [4K Pi 5 外接 GPU 游戏演示](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-and-external-amd-gpu-used-to-play-4k-open-source-kart-racing-game-pineboards-demos-supertuxkart-using-hat-upcity-lite-board)后，一名成员决定搭建一个 GPU 测试平台，以探索 **Pi OS** 的 `amdgpu` Linux kernel 补丁。
  
  - 他们记录了补丁的状态，并分享了关于如何应用该补丁以实现 Raspberry Pi 上**完整外接 GPU 支持**的见解。
- **在 Raspberry Pi 5 上实测外接 GPU**：该成员在周末的[直播](https://www.youtube.com/watch?v=EAlrCFJZlnI)中测试了该设置，展示了 **AMD RX 460** 外接 GPU 与 Raspberry Pi 5 的组合。
  
  - 测试演示了 **GLmark2** 的性能，揭示了未来 GPU 增强的巨大潜力。

**提及的链接**：[在 Raspberry Pi 5 上使用外接 GPU 进行 4K 游戏 | Jeff Geerling](https://www.jeffgeerling.com/blog/2024/use-external-gpu-on-raspberry-pi-5-4k-gaming)：未找到描述

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/) (1 messages):

tiendung：与原始方法相比效果如何？（需要 CPU）

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/1293407953025765487) (2 messages):

> - `Testing WebGPU`
> - `Browser Automation vs Native Development`
> - `Resource Management in Playwright`

- **寻找 WebGPU 测试库**：社区成员正在寻找测试 **WebGPU** 的推荐库，目前虽然在使用 **Vitest** 和 **Playwright**，但在测试运行中遇到了不稳定性（flakiness）。
  
  - *他们怀疑*问题可能源于 Playwright 在不同测试运行之间没有正确清理资源。
- **原生开发证明更高效**：一位成员分享道，原生开发 **WebGPU** 代码的感觉与传统的原生开发一样高效，且循环周期比浏览器自动化更短。
  
  - 然而，他们指出如果主要目标是浏览器，那么由于更严格的资源限制，进行直接测试是有好处的。
- **建议一种平衡的测试方法**：建议采用一种平衡的方法，即拥有原生模块和单元测试，同时在端到端（end-to-end）场景中使用浏览器测试。
  
  - *讨论强调*，这种方法将取决于开发所使用的具体语言和工具。

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1293391025166876784) (2 messages):

> - `FusedLinearJSD Implementation`
> - `Performance Metrics in High BT`

- **FusedLinearJSD 发布**：最近的 [pull request](https://github.com/linkedin/Liger-Kernel/pull/300) 引入了 **FusedLinearJSD**，通过避免大 logits 张量的实例化（materialization），实现了对最终线性层的高效处理。
  
  - 这与现有的 **fuse linear CE** 方法类似，并优化了前向和后向传递以提高执行效率。
- **基准测试速度的挑战**：**内存峰值显著降低**，但速度提升主要体现在高 batch times 下，而由于显存溢出（OOM）问题，这很难进行基准测试。
  
  - **原生 torch 版本**遇到了 OOM 错误，导致无法在这种情况下进行适当的性能测试。

**提及的链接**：[由 Tcc0403 添加 FusedLinearJSD · Pull Request #300 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/300)：摘要类似于 fuse linear CE。它通过 JSD 处理最终线性层的前向和后向传递，避免了大 logits 张量的实例化。由于 JSD 是最后一层...

---

### **GPU MODE ▷ #**[**metal**](https://discord.com/channels/1189498204333543425/1285384841730457600/1293315744448122941) (2 messages):

> - `GPU integer operations`
> - `bfloat16 support on M2`

- **质疑 GPU 整数移位速度**：一名成员对转换为 float 以及从 float 转换回来的**缓慢速度**表示困惑，并指出这涉及 **16 位移位**。
  
  - 他们询问 GPU 是否具有**向量化整数移位**，以潜在地加速该操作。
- **M2 上的原生 bfloat16 支持**：另一名成员询问了关于 **bfloat16** 数据类型特定声明的来源。
  
  - 他们确认 **M2 或更高版本**的芯片具有**原生 bfloat16 dtype** 支持。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1293293078945206322) (75 messages🔥🔥):

> - `ChatGPT vs. Claude 订阅`
> - `O1 与 O1 Mini 模型`
> - `AI 演进与意识`
> - `AI 中的路由模型 (Routing Models)`
> - `AI 开发中的挑战`

- **在 ChatGPT 和 Claude 订阅之间做出选择**：一位成员建议不要仅仅为了预览版功能而订阅 ChatGPT，认为使用上限限制了其吸引力，同时也指出访问 GPT-4 legacy 和 4o 可能物有所值。
  - 他们强调，如果订阅，目的应该是使用功能齐全的版本，而不是受限的预览版。
- **理解 O1 与 O1 Mini 模型**：成员们讨论了 O1 和 4o 模型之间的区别，指出 O1 模型充当“推理者 (reasoners)”，能够总结思路并在不确定时拒绝回答。
  - O1-mini 每天提供 50 次使用机会，而 4o 每 3 小时提供 80 次使用机会，这引发了关于这两个模型之间 A/B 测试的讨论。
- **AI 演进的理论探索**：关于 AI 意识潜在演进的讨论随之展开，深入探讨了重新训练和微调模型以提升能力的必要性。
  - 成员们思考了进化后的 AI 模型是否以及何时会具备商业可行性，并提到了围绕这些进展的潜在商业模式。
- **AI 中路由模型 (Routing Models) 的概念**：探讨了路由模型的概念，讨论了此类模型如何根据任务需求将查询定向到 O1 或 4o。
  - 这将优化用户体验，防止在处理多样化任务时过度依赖单一模型。
- **AI 开发中的挑战与观点**：成员们分享了对 AI 开发挑战的看法，特别是关于实现 AGI 的挑战，认为尽管有所进步，当前模型仍然局限在特定领域。
  - 对话涉及了 AI 的市场化及其与持续研究工作并行的方向，并将这些见解与文化上对 AGI 的痴迷进行了对比。

**提到的链接**：[来自 Mark Johns / Doomlaser (@Doomlaser) 的推文](https://x.com/Doomlaser/status/1843895040021803111)：我关于 AI 的诗，以九行诗的形式。我不是一个 AI 厌恶者，AI 被许多人憎恨和恐惧，被深知它的人珍视，哪一方会赢得这场战斗，关于存在的意义，关于不同的生活方式...

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1293542164411711580) (2 messages):

> - `ChatGPT 重写回复`
> - `Dall-E 提示词`
> - `Canvas 功能`

- **用户对 ChatGPT 重写行为的沮丧**：一位用户表达了不满，称 **ChatGPT** 经常重写他们的回复，导致他们停用了该工具数月。
  - 他们提到，试图修复这个被描述为“愚蠢缺陷”的问题让他们感到*头疼*，并寻求防止这种行为的建议。
- **重写行为的可能原因**：另一位成员推测 ChatGPT 的重写可能发生在 **Canvas** 或 **Dall-E 提示词**中，建议关注这些功能。
  - 对于 Dall-E，他们建议使用短语“使用这些确切的文字制作图像：[你的文字]”来防止重写。
- **请求更清晰的示例**：回复中指出需要进一步澄清，要求用户分享具体的对话，以便更好地理解重写问题。
  - 该建议旨在根据用户在使用 ChatGPT 时的确切体验提供更有针对性的帮助。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1293542164411711580) (2 messages):

> - `ChatGPT 重写回复问题`
> - `DALL-E 提示词`
> - `Canvas 功能`

- **用户因重写回复而放弃 ChatGPT**：一位成员对 **ChatGPT** 重写回复的倾向表示沮丧，称这导致他们停用了**数月**。
  - *修复这一缺陷带来的头疼感*让体验变得更糟，据他们报告，即使被要求停止，机器人仍会继续重写。
- **讨论 ChatGPT 的可能解决方案**：另一位成员建议重写可能与特定平台有关，如 **Canvas** 或 **DALL-E 提示词**。
  - 他们建议在 DALL-E 中使用特定的措辞，例如 *“使用这些确切的文字制作图像：[无论你的文字是什么]*”，这可能有助于解决问题。

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1293301374318022676) (67 条消息🔥🔥):

> - `免费算力提供`
> - `诺贝尔化学奖`
> - `LM Studio 更新与 MLX`
> - `LLM360 发布的新预训练数据集`
> - `招聘实践`

- **Kainan 为比赛提供免费算力**：Kainan 表示愿意为比赛提供免费算力资源，并询问其他人是否有兴趣。
  
  - 虽然有些人表现出兴趣，但对于会有多少参与者利用这个机会仍存在不确定性。
- **2024 年诺贝尔奖授予蛋白质研究**：瑞典皇家科学院授予了 2024 年 #NobelPrize 化学奖，奖项由 David Baker 以及 Demis Hassabis 和 John M. Jumper 共同获得，以表彰他们在计算蛋白质设计和结构预测方面的贡献。
  
  - 该奖项突显了蛋白质研究进展在科学界的重要性。
- **LM Studio 更新引入 Apple MLX**：LM Studio 发布了新版本 (0.3.4)，增加了对 Apple MLX 的支持，使得在 Apple Silicon Macs 上能够使用 MLX 模型和结构化 JSON 响应。
  
  - 用户注意到在 Apple 硬件上运行大型模型的改进，并对 MLX 在提升模型性能方面的潜力感到兴奋。
- **LLM360 发布海量预训练数据集**：LLM360 宣布了一个包含 15 万亿 (15 trillion) token 的新预训练数据集，该数据集经过严格的数据过滤流程，强调质量重于数量。
  
  - 该数据集的结构旨在支持高质量的 LLM 训练，包括多种过滤启发式算法并侧重于去重。
- **招聘策略讨论**：在轻松的交流中，成员们讨论了有效的招聘策略，建议了简历提交邮箱，并反思了等待公司主动联系的现状。
  
  - 讨论中提出了一种幽默而实用的观点：通过技能和知名度让自己变得不可或缺。

**提到的链接**：

- [来自 The Nobel Prize (@NobelPrize) 的推文](https://x.com/NobelPrize/status/1843951197960777760)：突发新闻：瑞典皇家科学院决定将 2024 年 #NobelPrize 化学奖的一半授予 David Baker “以表彰其在计算蛋白质设计方面的贡献”，另一半共同授予...
- [来自 Maxime Labonne (@maximelabonne) 的推文](https://x.com/maximelabonne/status/1843702625520283891?s=46)：🛞 TxT360：拥有 15T token 的新预训练数据集。来自 LLM360 的令人印象深刻的发布，包含 15T token 的新预训练数据集。与之前的开源项目相比，它包含了许多新来源...
- [GitHub - GAIR-NLP/O1-Journey: O1 Replication Journey: A Strategic Progress Report – Part I](https://github.com/GAIR-NLP/O1-Journey)：O1 复制之旅：战略进展报告 – 第一部分 - GAIR-NLP/O1-Journey
- [GitHub - lmstudio-ai/mlx-engine: Apple MLX engine for LM Studio](https://github.com/lmstudio-ai/mlx-engine)：用于 LM Studio 的 Apple MLX 引擎。通过创建账户为 lmstudio-ai/mlx-engine 的开发做出贡献。
- [LM Studio 0.3.4 支持 Apple MLX](https://lmstudio.ai/blog/lmstudio-v0.3.4)：在 Apple Silicon Macs 上使用 MLX 进行超快速且高效的设备端 LLM 推理。
- [下载 LM Studio - Mac, Linux, Windows](https://lmstudio.ai/download)：发现、下载并运行本地 LLM

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1293303378499338331) (3 条消息):

> - `Llama Stack`
> - `Fast Inference with Llama 3.1-8B`
> - `Meta's GitHub Releases`

- **Llama Stack 发布新工具**：一位成员分享说，在 Meta 在 [GitHub](https://github.com/meta-llama/llama-stack) 上发布了一套完整的工具和示例后，他们最近发现了 **Llama Stack**。
  
  - 他们表达了兴趣，表示它看起来**非常强大**，但尚未进行实验。
- **寻求快速推理策略**：有人询问在 **4xA40 节点**上对 **Llama 3.1-8B** 或更小模型实现 **fast inference** 的最佳方法。
  
  - 这表明人们对优化大模型实现的性能表现出越来越浓厚的兴趣。
- **Llama Stack 的 GitHub 链接**：分享了两个 GitHub 仓库链接，一个包含 **Llama Stack APIs** 的 [Agentic components](https://github.com/meta-llama/llama-stack-apps)（智能体组件），另一个涵盖了 [Model components](https://github.com/meta-llama/llama-stack)（模型组件）。
  
  - 这些仓库为希望实现和利用 Llama Stack 功能的开发者提供了详细的资源。
- **Llama 缓存位置**：一位成员指出，Llama 模型的典型 cache 位置可能在 **~/.cache/huggingface/hub/** 或专门在 **~/.llama** 中。
  
  - 这突显了社区中关于模型存储目录的常见做法。

**提到的链接**：

- [GitHub - meta-llama/llama-stack-apps: Agentic components of the Llama Stack APIs](https://github.com/meta-llama/llama-stack-apps)：Llama Stack APIs 的 Agentic 组件。可以通过在 GitHub 上创建账号来为 meta-llama/llama-stack-apps 的开发做出贡献。
- [GitHub - meta-llama/llama-stack: Model components of the Llama Stack APIs](https://github.com/meta-llama/llama-stack)：Llama Stack APIs 的模型组件。可以通过在 GitHub 上创建账号来为 meta-llama/llama-stack 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1293299659338682473) (4 条消息):

> - `Text to Video Models`
> - `O1 Replication Journey`
> - `Model Merging at Scale`

- **探索免费的 Text to Video 模型**：一位成员询问是否有任何**免费的 text to video 模型**（包括动画和非动画），并收到了关于 **animate2diff** 等潜在模型的建议。
  
  - 看来人们对于寻找更多从文本提示生成视频内容的选项持续关注。
- **来自 O1 复制之旅报告的见解**：[这份报告](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf) 详细介绍了一种开创性的 AI 研究方法，强调在复制 OpenAI 的 **O1 model** 过程中的透明度和社区参与。
  
  - 该方法论旨在解决团队项目中的挑战，记录成功和失败以增强开放科学。
- **评估大规模模型合并**：[该研究](https://arxiv.org/abs/2410.03617) 调查了 **model merging**，重点关注专家模型大小、基础模型质量和数量如何影响性能，并使用了 Averaging 和 TIES 等方法。
  
  - 主要发现表明，使用更强大的基础模型时合并更成功，而且在处理多个专家模型时，**更大的模型**能增强泛化能力。

**提到的链接**：

- [What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617)：模型合并旨在将多个专家模型组合成一个能力更强的单一模型，具有减少存储和推理成本、提高泛化能力以及支持去中心化等优点。
- [O1-Journey/resource/report.pdf at main · GAIR-NLP/O1-Journey](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf)：O1 复制之旅：战略进展报告 – 第一部分 - GAIR-NLP/O1-Journey

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1293398701217415300) (1 条消息):

> - `VLM performance timeline`
> - `Vision-language models`
> - `Parameter count comparison`

- **寻求 VLM 性能时间线**：一位成员分享了他们的 [VLM 性能时间线链接](https://twitter.com/nahidalam/status/1843736808443822407)，但表示希望能看到随时间推移的改进，特别是结合**参数量 (parameter count)** 的变化。
  
  - 他们指出，虽然类似的性能时间线在 **LLM** 领域很常见，但针对**视觉语言模型 (vision-language models)** 的此类资源仍然稀缺。
- **请求更好的 VLM 基准测试**：该成员询问是否有人见过能反映性能随**参数量**或其他特征变化的 **VLM 时间线**。
  
  - 他们表示，此类对比在关于 **LLM** 的讨论中更为频繁，这使得他们自己的尝试显得颇具新意。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1293299659338682473) (4 条消息):

> - `Free text-to-video models`
> - `O1 Replication Journey`
> - `Model merging at scale`

- **关于免费文本生成视频模型的咨询**：一位用户询问是否有可用的免费文本生成视频模型，包括动画和非动画类型。
  
  - 另一位成员建议查看 'animate2diff' 以寻找潜在选择。
- **O1 复现之旅揭晓**：[O1 Replication Journey 论文](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf) 提供了一份响应 OpenAI O1 模型的战略进展报告，强调透明度和实时探索。
  
  - 值得注意的是，他们声称其“旅程学习范式”在 MATH 数据集上仅凭 **327 个训练样本**，就比传统的监督学习性能提升了 **8%** 以上。
- **关于模型合并有效性的见解**：一项研究强调了模型合并的益处，系统地评估了影响不同模型规模（从 **1B 到 64B** 参数）性能的因素。
  
  - 关键发现表明，模型合并在强大的基座模型和更大规模下更为有效，能带来更好的泛化能力，尤其是在合并多达 **8 个专家模型**时。

**提及的链接**：

- [What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617)：模型合并旨在将多个专家模型组合成一个能力更强的单一模型，具有降低存储和推理成本、提高泛化能力以及支持去中心化等优点。
- [O1-Journey/resource/report.pdf at main · GAIR-NLP/O1-Journey](https://github.com/GAIR-NLP/O1-Journey/blob/main/resource/report.pdf)：O1 复现之旅：战略进展报告 – 第一部分 - GAIR-NLP/O1-Journey

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1293288511876759633) (71 messages🔥🔥):

> - `Prompt Caching`
> - `Inflection 3.0 and Enterprise`
> - `OpenRouter API Rate Limits`
> - `NotebookLM Deep Dive Podcast`
> - `User Concerns about Gemini Moderation`

- **Prompt Caching 详解**：成员们讨论了 **Prompt Caching** 的机制和用途，识别了它可能不利的情况，例如上下文频繁变化或 Prompt 过短。
  
  - 一位成员指出：*“对于那些执行自动 Prompt Caching 的提供商，你无法禁用该功能，”* 强调了某些提供商设置的限制。
- **围绕 Inflection 3.0 的好奇**：**Inflection 3.0** 的发布引发了关注，特别是它可能与 **Intel Gaudi 3** 集成以提升性能。
  
  - 然而，讨论中也流露出对这种炒作的怀疑，一些成员指出他们看到的具体信息很少，特别是在 **Benchmarks** 方面。
- **OpenRouter API Rate Limits**：对 **OpenRouter** API 请求限制进行了澄清，指出这些限制是根据账户额度（Credits）动态变化的。
  
  - 一位成员分享了一个 **GET request** 示例，用于检查与 API Key 相关的 **Rate Limit** 使用情况和额度，这有助于指导使用。
- **NotebookLM Podcast 的利用**：成员们对 **NotebookLM Deep Dive Podcast** 给予了积极反馈，一些人创建了 Notebooks 以便在旅途中收听内容。
  
  - 一位用户对 **ai-podcast-maker** 等自动化工具表示感兴趣，并指出虽然音频可能不够流畅，但 *“automation ftw（自动化万岁）。”*
- **对 Gemini Moderation 的担忧**：一位用户对 **Gemini** 是否会对输入进行 **Moderation** 表示担忧，担心用户的输入可能导致潜在的封号风险。
  
  - 这突显了关于 AI 应用中用户体验和内容审核（Content Moderation）的更广泛讨论。

**提到的链接**：

- [no title found](https://]): 未找到描述
- [Inflection AI Developer Playground](https://developers.inflection.ai/docs): 让我们构建更好的企业级 AI。
- [Patterns of Application Development Using AI](https://leanpub.com/patterns-of-application-development-using-ai): 探索构建智能、自适应且以用户为中心的软件系统的实用模式和原则，充分发挥 AI 的力量。
- [Introducing Inflection for Enterprise](https://inflection.ai/blog/enterprise): 介绍由我们创新的 Inflection 3.0 AI 系统驱动的企业版 Inflection，该系统与 Intel 合作打造。该解决方案为 GenAI 部署提供了卓越的性价比...
- [Limits | OpenRouter](https://openrouter.ai/docs/limits): 设置模型使用限制
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing#disabling-fallbacks): 在多个提供商之间路由请求
- [Inflection AI](https://inflection.ai/): 很简单。我们训练并微调它。你拥有它。让我们以正确的方式做企业级 AI。
- [no title found](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429): 未找到描述

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1293311268081504409) (4 条消息):

> - `LlamaIndex Workflows tutorial` (LlamaIndex Workflows 教程)
> - `LlamaCloud and LlamaParse demo` (LlamaCloud 与 LlamaParse 演示)
> - `SFTechWeek meetup` (SFTechWeek 聚会)
> - `OpenAI Realtime API Client demo` (OpenAI Realtime API Client 演示)

- **LlamaIndex Workflows 全面指南**：由 @jamescalam 编写的详细[教程](https://t.co/uVJwXeY3lP)，涵盖了什么是 Workflows、其与 LangGraph 的对比，以及如何构建一个 AI 研究 Agent。
  
  - 教程还包括了调试和优化技巧，帮助用户轻松上手。
- **使用 LlamaCloud 进行财务数据分析**：在最近的一次演示中，@ravithejads 展示了如何利用 [LlamaCloud 和 LlamaParse](https://t.co/ZfrbgnNQg4) 填写用于对比多家公司的财务电子表格。
  
  - 该用例展示了 LLM 在理解数据和自动化表单填写方面的实际应用。
- **SFTechWeek 聚会提醒**：最后一次召集参会者加入在 LlamaIndex 总部举办的线下聚会，共同讨论 #SFTechWeek 期间生产环境中的 Multi-Agent 工作流。
  
  - 活动承诺提供食物、乐趣，以及关于处理 RAG 系统和 Agent 生产化挑战的见解。
- **与 AI Agent 进行交互式聊天**：由 @LoganMarkewich 展示的演示，展示了如何通过 [OpenAI realtime API client](https://t.co/ppbS5Fougg) 使用语音与 AI Agent 聊天。
  
  - 这个开源应用程序使用户能够构建自己的语音 Agent，并提供了可立即使用的示例。

**提到的链接**：

- [RSVP to Multi-Agentic Workflows in Prod #SFTechWeek | Partiful](https://t.co/7ytgH2CXNj?)：注：这是在旧金山 LlamaIndex 总部举办的线下聚会，由 Activeloop 和 LlamaIndex 共同举办。此活动是 #TechWeek 的一部分 —— 由风投和初创公司主办的为期一周的活动，旨在汇聚……
- [GitHub - run-llama/openai_realtime_client](https://t.co/ppbS5Fougg)：通过在 GitHub 上创建账户，为 run-llama/openai_realtime_client 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1293306992689152111) (45 条消息🔥):

> - `Semantic chunking in TypeScript` (TypeScript 中的语义分块)
> - `PropertyGraphIndex extractors` (PropertyGraphIndex 提取器)
> - `Integration issues with LlamaIndex` (LlamaIndex 的集成问题)
> - `Context chat engine and reranking` (Context chat engine 与重排序)
> - `RAG reducing hallucinations` (RAG 减少幻觉)

- **寻求 TypeScript 中的语义分块 (Semantic Chunking)**：一位用户询问如何在 TypeScript 中实现 **semantic chunking**，类似于 Python 中的示例代码。
  
  - 他们表示难以找到类似的功能，并寻求社区的帮助。
- **PropertyGraphIndex 提取器的困惑**：一位成员询问 **DynamicLLMPathExtractor** 是否可以直接在 Document 上调用，因为它在插入过程中起作用，但在其他情况下提供非预期结果。
  
  - 其他成员澄清说，需要先将文档分块为 nodes，并指出提取器旨在处理注入了 metadata 的 nodes。
- **LlamaIndex 与 Phoenix 的问题**：一位用户报告了 **Phoenix** 与 **LlamaIndex** 之间的集成问题，错误消息与异步函数执行期间的上下文分离（context detachment）有关。
  
  - 社区成员确认该错误并非关键性错误，并建议检查底层代码以增强功能。
- **Context Chat Engine 重排序 (Reranking) 问题**：一位用户遇到了在使用 context chat engine 时跳过 **reranker** 的问题，并寻求帮助解决。
  
  - 在迭代代码后，他们确认重新编写初始化程序解决了重排序问题，并确认功能运行正常。
- **RAG 对减少幻觉的影响**：一位成员询问是否存在关于 **检索增强生成 (RAG)** 如何减少幻觉的研究，从而引发了社区搜索。
  
  - 他们发现了几篇讨论 RAG 在提高模型输出质量方面有效性的学术论文，但同时也承认 RAG 是否直接减少幻觉仍存在不确定性。

**提到的链接**：

- [Reducing hallucination in structured outputs via Retrieval-Augmented Generation](https://arxiv.org/html/2404.08189v1)：未找到描述
- [llama_index/llama-index-core/llama_index/core/instrumentation/dispatcher.py at 65946eb92419e94a4cec85af671c78e0ed122593 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/65946eb92419e94a4cec85af671c78e0ed122593/llama-index-core/llama_index/core/instrumentation/dispatcher.py#L242)：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1293289058809679882) (39 条消息🔥):

> - `AI girlfriend data breach` (AI 女友数据泄露)
> - `Sequoia's 3rd annual AI essay` (红杉资本第三届年度 AI 文章)
> - `Nobel Prize in Chemistry` (诺贝尔化学奖)
> - `Palmyra X 004 release` (Palmyra X 004 发布)
> - `ChatGPT search rollout` (ChatGPT 搜索功能推出)

- **AI Girlfriend 服务数据泄露曝光**：AI 女友服务 Muah.ai 上个月遭遇了**数据泄露**，泄露了 **190 万个电子邮件地址**，其中包括具有性暗示的敏感 Prompt。
  
  - 安全专家和分析师对此类数据泄露的影响表示担忧，特别是泄露内容中涉及的**儿童剥削**细节。
- **红杉资本（Sequoia Capital）对 AI 演进的见解**：红杉资本的第三份年度文章讨论了生成式 AI 研究从**“快思考”**到**“慢思考”**的转变，重点关注推理阶段（inference time）的推理能力，这正在开启新的应用场景。
  
  - 像 **OpenAI** 和 **Google DeepMind** 这样的关键参与者正在稳定市场，而更新的 **agentic applications** 预计将在各个领域涌现。
- **2024 年诺贝尔化学奖揭晓**：**2024 年诺贝尔化学奖**授予了 **David Baker**，以表彰其在计算蛋白质设计方面的贡献；以及 **Demis Hassabis** 和 **John M. Jumper**，以表彰他们通过 **AlphaFold2** 在蛋白质结构预测方面的工作。
  
  - 这一认可凸显了 AI 在推进**生物化学**方面的重大贡献，它已实现了对近 **2 亿种蛋白质**结构的预测。
- **Palmyra X 004 发布亮点**：Writer 的新模型 **Palmyra X 004** 在 HELM 排名中位列前 10，引入了全栈 **tool calling** 并在合成数据上进行了训练。
  
  - 此次发布引起了广泛关注，包括 **Venture Beat** 的报道，指出其在 AI function calling 和 CRM 改进方面的能力。
- **ChatGPT 推出搜索功能**：报告指出 **ChatGPT** 正在推出 **SearchGPT**，通过在 **GPT-4o** 中集成引用功能，使其能够直接与 **Perplexity** 等平台竞争。
  
  - 此举标志着 ChatGPT 能力的战略性增强，使其更贴合信息检索和用户查询响应的需求。

**Links mentioned**:

- [来自 The Nobel Prize (@NobelPrize) 的推文](https://x.com/NobelPrize/status/1843951197960777760)：重磅消息：瑞典皇家科学院决定将 2024 年 #NobelPrize 化学奖授予 David Baker，以表彰其在“计算蛋白质设计”方面的贡献，另一半共同授予...
- [来自 Alex Volkov (Thursd/AI) (@altryne) 的推文](https://x.com/altryne/status/1843738554352185542?s=46&t=2qGo-Hp_MDNyh14F888CkQ)：今天我和 @weights_biases 的同事们开了一个关于“Cursor 技巧与心得”的会议，我想在一个 🧵 中分享我们“发现”并交流的内容。如果你还没...
- [来自 Thomas Schulz (@thomasschulzz) 的推文](https://x.com/thomasschulzz/status/1844062893723250940?s=46)：重磅：看起来 OpenAI 正在进入与 Perplexity 竞争的赛道... GPT-4o 现在支持引用了 👀
- [来自 Ishaan Kapoor (@Ishaank1999) 的推文](https://x.com/ishaank1999/status/1843764968556278020?s=46)：PDF 是恶魔的文件格式。几乎每个构建 RAG 的人都需要处理它们——这太糟糕了。市场上的解决方案要么太慢，要么太贵，或者不是 OSS。它本该更简单。...
- [来自 Saining Xie (@sainingxie) 的推文](https://x.com/sainingxie/status/1843956473098883426)：在 DeepMind 实习期间，Demis 会见了所有实习生。当被问及公司的目标时，我清晰地记得他说过：“赢得 *多个* 诺贝尔奖。”我当时很震惊，但是...
- [来自 Sonya Huang 🐥 (@sonyatweetybird) 的推文](https://x.com/sonyatweetybird/status/1844079873855549856?s=46)：每年一次，@gradypb 和我会与我们可靠的 AI 合作伙伴 坐下来，从宏观角度审视 Generative AI 正在发生的事情。这是我们第三年的年度观点…… 1: 基础模型层...
- [来自 Sam Julien (@samjulien) 的推文](https://x.com/samjulien/status/1844009797244580315)：🆕 来自 @Get_Writer: Palmyra X 004 🎉 我们最新的前沿模型在 HELM Lite 和 HELM MMLU 上均位列前 10，并为 Writer 平台引入了全栈 tool calling！
- [来自 Seán Ó hÉigeartaigh (@S_OhEigeartaigh) 的推文](https://x.com/S_OhEigeartaigh/status/1843979139948355893)：还没完。有报道称诺贝尔文学奖将授予“OpenAI 非营利治理结构”的作者，以表彰其在创意写作方面的杰出贡献...
- [来自 Troy Hunt (@troyhunt) 的推文](https://x.com/troyhunt/status/1843788319785939422)：出于 @josephfcox 文章中显而易见的原因，处理这次泄露事件让人非常不适。让我根据我的发现增加一些“色彩”：引用 Have I Been Pwn...
- [来自 The Nobel Prize (@NobelPrize) 的推文](https://x.com/NobelPrize/status/1843951594909380878)：2024 年 #NobelPrize 化学奖得主 Demis Hassabis 和 John Jumper 成功利用人工智能预测了几乎所有已知蛋白质的结构。2020 年，Hassabis ...
- [Generative AI 的 Act o1](https://www.sequoiacap.com/article/generative-ais-act-o1/)：Agentic Reasoning 时代开启。
- [来自 Clara Shih (@clarashih) 的推文](https://x.com/clarashih/status/1843501862764372083?s=46)：上周 @OpenAI 推出了 ChatGPT Canvas，一个显示文本、代码和可视化输出的界面。在企业级应用中，我们依赖更结构化、更可信的 UX 元素——记录详情、列表...
- [来自 Alex Volkov (Thursd/AI) (@altryne) 的推文](https://x.com/altryne/status/1843738554352185542)：今天我和 @weights_biases 的同事们开了一个关于“Cursor 技巧与心得”的会议，我想在一个 🧵 中分享我们“发现”并交流的内容。如果你还没...
- [GitHub - lumina-ai-inc/chunkr: 基于视觉模型的 PDF 分块。](https://github.com/lumina-ai-inc/chunkr)：基于视觉模型的 PDF 分块。通过在 GitHub 上创建账号来为 lumina-ai-inc/chunkr 的开发做出贡献。

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1293649509594955900) (1 条消息):

> - `Molmo`
> - `Pixmo`

- **参加今天关于 Molmo 和 Pixmo 的直播环节！**：今天的环节将讨论 **Molmo** 和 **Pixmo**，[点击此链接加入 Zoom 会议](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)。
  
  - 这是一个直接从专家那里获取见解的机会，千万不要错过！
- **社区对 Molmo 和 Pixmo 充满期待**：社区对 **Molmo** 和 **Pixmo** 的功能感到兴奋不已，许多人渴望分享他们的想法。
  
  - 鼓励参与者在环节中积极互动并分享见解。

 

**提到的链接**：[加入我们的云高清视频会议](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1293298898554589265) (2 messages):

> - `DOM Data Attributes`
> - `WebAssembly Component Model`

- **DOM 允许通过属性存储数据**：一个关键的 **DOM 特性** 允许通过以 `data-myattribute` 开头的属性在元素上存储数据，增强了将数据直接与 HTML 元素关联的能力。
  
  - 这一功能为在 DOM 上下文中操作和检索数据开辟了创造性的途径。
- **WebAssembly Component Model 仓库发布**：分享了 **WebAssembly Component Model** 的仓库链接，在 [WebAssembly/component-model](https://github.com/WebAssembly/component-model) 详细介绍了其设计和规范。
  
  - 该仓库对于那些对 **WebAssembly** 中组件模型复杂细节感兴趣的人来说是一个重要的资源。

 

**提到的链接**：[GitHub - WebAssembly/component-model: Repository for design and specification of the Component Model](https://github.com/WebAssembly/component-model)：WebAssembly/component-model - 组件模型设计与规范仓库。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1293391822462255216) (24 messages🔥):

> - `Mojo and Scikit-learn`
> - `Mojo GPU Support`
> - `Running ONNX Models in Mojo`
> - `Drivers for Mojo GPU Usage`

- **Mojmelo：Scikit-learn 流水线的 Mojo 解决方案**：一位成员分享了 [Mojmelo](https://github.com/yetalit/mojmelo)，这是一个用纯 Mojo 🔥 实现机器学习算法的项目，可能成为运行 Scikit-learn 流水线的催化剂。
  
  - 另一个论点是 Mojo 有望取代 **Scikit-learn** 中所有的 **Cython** 依赖。
- **对 Mojo 即将推出的 GPU 支持感到兴奋**：成员们对 Mojo 即将推出的 **GPU 支持** 表示热切期待，强调了其提升性能的潜力。
  
  - 一些人正在探索将 **PyTorch** 与 Mojo 集成的可能性，同时关注 GPU 能力。
- **Mojo 在 GPU 上运行 AI 需要驱动程序**：会议澄清了使用 Mojo 在 GPU 上进行 AI 运算需要 **Nvidia 驱动程序**，而关于 AMD 兼容性的反馈则褒贬不一。
  
  - 讨论强调了现代 GPU 驱动程序除了简单的通信之外，还承担着 **电源管理** 和多进程处理等重要角色。
- **在 GPU 上使用纯 Mojo 运行 ONNX 模型：是否可能？**：一位用户询问了在不依赖额外组件的情况下，在 GPU 上使用 **纯 Mojo** 运行 **ONNX 模型** 的潜力。
  
  - 虽然目前能力尚不确定，但有人询问未来的版本是否会启用此功能。

 

**提到的链接**：[GitHub - yetalit/Mojmelo: Machine Learning algorithms in pure Mojo 🔥](https://github.com/yetalit/mojmelo)：使用纯 Mojo 🔥 实现的机器学习算法。可以通过在 GitHub 上创建账号来为 yetalit/Mojmelo 的开发做出贡献。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1293471741888168008) (5 messages):

> - `Performance of Mojo graphs`
> - `Pre-compiling graphs`
> - `Reuse of inference sessions`
> - `mojo run vs compiled binaries`
> - `Graph input types`

- **Mojo 图（graphs）的性能仍然较慢**：一位用户分享了显示两个图 **总编译时间** 的性能指标：**graph1** 为 0.312s，**graph2** 为 0.451s。
  
  - 他们对编译时间影响调试表示沮丧，无论他们使用的是 **mojo run** 还是编译后的二进制文件。
- **提升图性能的建议**：一位成员建议复用 **推理会话（inference session）** 可能有助于分摊编译成本。
  
  - 他们推测问题可能源于使用 **List** 作为输入而不是固定大小的类型，这可能会影响性能。
- **mojo run 与编译后的二进制文件对比**：另一位成员询问该用户是在运行 **编译后的二进制文件** 还是使用 **mojo run**，以了解性能差异。
  
  - 用户确认使用了 **mojo run**，引发了关于其对编译时间影响的进一步讨论。
- **MAX Engine 图面临的挑战**：用户注意到标准库中缺少任何 **MAX Engine 图**，这可能会限制他们的优化选择。
  
  - 这突显了 Mojo 开发者在可用工具和资源方面的一个潜在改进领域。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1293325311290773565) (1 条消息):

> - `实验作业已发布`
> - `课程报名`
> - `用于协作的 Discord`
> - `实验完成标准`

- **课程实验作业已发布**：课程的实验作业已正式发布，第一个实验重点是使用 **Autogen framework** 分析餐厅评论，截止日期为 **12月12日晚上11:59 (PST)**。
  
  - 实验 2 和 3 将侧重于 **LLM security** 的 **prompt engineering**，特别是编写攻击和防御提示词。
- **感兴趣的学生可便捷报名**：鼓励有意向的学生通过填写此 [表单](https://forms.gle/svSoNhKcGFjxup989) 报名参加课程。
  
  - 如需进一步讨论，学生应加入 [**LLM Agents Discord**](https://discord.gg/NWVpQ9rBvd) 频道。
- **利用 Discord 进行提问**：建议使用 Discord 与课程工作人员沟通并询问实验相关问题，因为他们将积极监控该频道。
  
  - 学生在提问前应先查阅持续更新的 **FAQ document**，以避免重复提问。
- **引入协作指南**：在与课程中的他人协作时，敦促学生*避免分享确切的解决方案*，以维护学术诚信。
  
  - 鼓励进行概念性讨论，但具体的实现细节和代码文件应保持私密。
- **实验完成与提交预期**：实验提交后，学生将收到关于其完成状态的通知，并定义了通过各项实验的阈值：实验 1 为 3/4，实验 2 为 1/2 的隐藏测试，实验 3 为 1/3 的隐藏攻击测试。
  
  - 这些标准强调了不仅要完成实验，还要在设定的评估中取得成功的重要性。

 

**提到的链接**：[Large Language Model Agents](http://llmagents-learning.org/f24)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1293329185863110736) (16 条消息🔥):

> - `实验 1 文件问题`
> - `测验提交疑虑`
> - `下学期课程开设`
> - `Ninja & Legendary 等级要求`
> - `Agent 定义讨论`

- **实验 1 下载的文件为空**：多名用户报告了下载 **实验 1 指南** 时遇到的问题，称下载的文件为空，而实验 2 和 3 运行正常。
  
  - *澄清说明*：该文件位于 **Google Drive** 上，并确认尽管没有预览，但应该是可以访问的。
- **测验提交邮箱格式澄清**：一名用户询问由于其邮箱中包含一个点（他通常会省略），其测验提交是否会被正确记录。
  
  - 回复指出，**无论在报名表中使用何种邮箱格式**，系统都将以此追踪提交情况，并强调报名时准确性的重要性。
- **关于下学期开课的咨询**：一名用户询问了下学期是否可能再次开设该课程，寻求确认。
  
  - 虽然尚不确定，但提到教授之前曾开设过其他 MOOC，很可能会再次开设。
- **实验的 Ninja 和 Legendary 等级要求**：有提问涉及 **Ninja** 和 **Legendary tiers** 是否必须完成实验作业，认为这些作业仅与 mastery tier 挂钩有些奇怪。
  
  - 据指出，对 Ninja 和 Legendary 等级学生的期望是将其精力优先放在 **hackathon submissions** 上。
- **Agent 定义辩论**：一名用户提出疑问，使用 **discriminative AI** 或生成式与判别式 AI 混合的“一段代码”是否符合 Agent 的定义。
  
  - *他们认为答案是肯定的*，这表明在 AI 编程语境下，关于定义的界定存在一些不确定性。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1293511249069477999) (10 条消息🔥):

> - `Reinforcement Learning 在 AGI 中的角色`
> - `课程 Q&A 澄清`
> - `直播课程视频困惑`
> - `协作作业头脑风暴`

- **关于 Reinforcement Learning 在 AGI 中角色的讨论**：一位成员提出了关于 **Reinforcement Learning (TD learning)** 在迈向 **AGI** 的过程中是否仍然具有重要意义，或者 Agent 是否可以在没有它的情况下有效运行的问题。
  
  - 这一询问引发了关于现代 AI 系统中 RL 的必要性和应用的讨论。
- **澄清上节课的 Q&A**：针对上次会议缺乏 **Q&A** 环节的担忧，一些成员表示该环节确实发生了，但在录制视频中不可见。
  
  - 一位成员引用了 [YouTube 视频](https://www.youtube.com/clip/Ugkx4tBcZZFsUyro53RVd6W_9yySQL9OAdep)中的一个片段，据称其中包含了提问环节。
- **对直播视频内容的困惑**：一位成员对录制的直播课程表示困惑，称他们在视频中找不到 Q&A 环节，尽管他们仍然可以访问该视频。
  
  - 另一位成员提到，在该片段之后确实有提问，但可能没有被视频捕捉到。
- **呼吁协作学习**：一位成员鼓励大家在完成作业时进行交流协作和头脑风暴。
  
  - 这一邀请旨在促进同学之间在处理课程作业时的协作努力。

**提到的链接**：[YouTube](https://www.youtube.com/clip/Ugkx4tBcZZFsUyro53RVd6W_9yySQL9OAdep)：未找到描述

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**runpod-help**](https://discord.com/channels/1104757954588196865/1162430527215763569/1293555490982461450) (26 条消息🔥):

> - `训练 Vicuna-7B 模型`
> - `CUDA 显存溢出 (OOM) 错误`
> - `DeepSpeed 配置问题`
> - `Runpod 实例使用`

- **Runpod 上的训练进程卡住**：一位用户报告其 **Vicuna-7B** 模型的训练进程卡住且没有输出，并分享了启动该进程的命令行。
  
  - 另一位成员建议分享示例配置以进一步诊断问题。
- **DeepSpeed 配置错误**：用户遇到了一个与其 **DeepSpeed** 配置相关的错误，提示“输入应为有效的整数，但得到了带有小数部分的内容”。
  
  - 社区建议确保设备数量是 2 的倍数，并安装特定版本的 DeepSpeed，这最终解决了问题。
- **尽管资源充足仍出现 CUDA 显存问题**：该用户表示，尽管他们拥有 5 个 GPU（每个 24GB 显存），但仍面临 **CUDA out of memory** 错误。
  
  - 他们提供了 **DeepSpeed** 和 **accelerate** 配置，寻求关于意外显存不足的见解。
- **目标配置与资源**：用户提到了他们的 **DeepSpeed** 配置，并指出该配置参考了 GitHub 上的示例。
  
  - 他们强调自己正在 Runpod 实例上进行实验，并说明了其规格。
- **社区协作排查故障**：社区成员积极协作，排查模型训练和配置设置中出现的问题。
  
  - 他们交换了见解和配置链接，同时协助解决用户关于训练和资源管理的问题。

**提到的链接**：

- [axolotl/examples/medusa at main · ctlllll/axolotl](https://github.com/ctlllll/axolotl/tree/main/examples/medusa)：欢迎提问。通过在 GitHub 上创建账户为 ctlllll/axolotl 的开发做出贡献。
- [axolotl/deepspeed/zero1.json at main · ctlllll/axolotl](https://github.com/ctlllll/axolotl/blob/main/deepspeed/zero1.json)：欢迎提问。通过在 GitHub 上创建账户为 ctlllll/axolotl 的开发做出贡献。
- [examples/train_lora/llama3_lora_sft_ds3.yaml 报错 · Issue #5252 · hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/issues/5252#issuecomment-2311619703)：提醒：我已阅读 README 并搜索了现有 Issue。系统信息：使用 ds_z3_config.json 时报错，错误显示：pydantic_core._pydantic_core.ValidationError: 1 validation error for DeepSpeedZeroConfig...

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1293454153414086656) (9 条消息🔥):

> - `模型可扩展性担忧`
> - `ML 中的 P 值报告`
> - `L-mul 的实现`
> - `RL 算法种子的影响`
> - `研究中的信号与噪声`

- **模型可扩展性引发关注**：一位成员对一篇基于 **350 billion tokens** 训练的论文的**可扩展性 (scalability)** 表示担忧，质疑其改进的显著性。
  
  - *讽刺的是*，另一位成员指出 **ML 专业人士** 经常忽视诸如 p 值之类的基础统计指标。
- **P 值在 ML 中不常见**：一位成员对 ML 论文中缺乏 **p 值 (p-values)** 和**置信区间 (confidence intervals)** 表示沮丧，并提到由于自己有医学背景，这种现象让他感到很不适应。
  
  - 另一位参与者评论说，他们很少在 ML 语境中看到 **p 值** 的使用，这突显了科学报告中的文化差异。
- **关于 L-mul 实现的讨论**：有建议认为 **L-mul** 应该在 **torchao** 中实现，因为预计他们会接纳此类项目。
  
  - 一位成员鼓励加入他们的频道以进行更多协作，表明这是一个欢迎新想法的社区。
- **更改种子可能改变结果**：对话透露，之前的研究表明更改随机**种子 (seed)** 会显著影响 **RL 算法** 的结果。
  
  - 一位成员强调，仅根据报告的数字很难从小型模型中得出有意义的见解。
- **评估新想法的影响**：关于一个新想法，一位成员承认其潜力，但对其在与其他**优化 (optimizations)** 并存时的实际影响表示怀疑。
  
  - 他们将此类论文视为想法的起点，反映了关于如何衡量研究发现显著性的持续讨论。

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1293667212175081625) (1 条消息):

> - `SOAP 优化器`
> - `AdamW 学习率问题`
> - `NanoGPT 竞速 (speedrunning) 成就`

- **SOAP 优于 AdamW 但需要调优**：一位用户在 **Alpaca** 上测试了 **SOAP 优化器**，指出在调整 **AdamW 的学习率** 之前，它的表现优于 **AdamW**。
  
  - 然而，他们提到目前的实现尚不支持**分布式 (distributed)** 训练或 **bf16** 格式。
- **NanoGPT 刷新样本效率记录**：在最近的一次更新中，**SOAP 优化器** 在 **3.25B 训练 tokens** 中达到了 **3.28 Fineweb 验证损失**，创下了新的样本效率记录。
  
  - 根据 @kellerjordan0 的一条 [推文](https://x.com/kellerjordan0/status/1844094933197783298/photo/1)，这超越了由另一款优化器创下的 **3.67B tokens** 的前纪录。

**提到的链接**：[来自 Keller Jordan (@kellerjordan0) 的推文](https://x.com/kellerjordan0/status/1844094933197783298/photo/1)：NanoGPT 竞速更新：使用 SOAP 优化器 ([https://arxiv.org/abs/2409.11321](https://arxiv.org/abs/2409.11321))，@vyasnikhil96 在 3.25B 训练 tokens 中实现了 3.28 Fineweb 验证损失的新样本效率记录...

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1293439117320917034) (3 messages):

> - `Diff Transformer`
> - `L-Mul Algorithm`
> - `Floating Point Multiplication Replacement`

- **Diff Transformer 胜过传统 Transformer**：**Diff Transformer** 引入了一种**微分注意力机制 (differential attention mechanism)**，增强了对相关上下文的关注，并在各种基准测试中表现优于传统 Transformer。
  
  - 它在**长上下文建模 (long-context modeling)** 方面有显著帮助，并减少了问答等任务中的幻觉 (hallucination) 现象。
- **L-Mul 算法大幅降低能耗**：提出的 **L-Mul 算法**利用整数加法来近似浮点乘法，在保持更高精度的同时，将能耗降低了 **95%**。
  
  - 该方法相比 **8-bit floating point multiplications** 有显著改进，预示着在神经网络计算中节省大量资源的潜力。
- **关于使用 L-Mul 进行预训练的讨论**：有人提出了关于使用 **L-Mul 算法**预训练模型的可行性及其对性能影响的问题。
  
  - 评估这种方法是否也有助于解决预训练期间主要的**能耗黑洞 (energy sink)** 问题引起了人们的兴趣。

**提到的链接**：

- [Addition is All You Need for Energy-efficient Language Models](https://arxiv.org/abs/2410.00907)：大型神经网络的大部分计算都花在浮点张量乘法上。在这项工作中，我们发现浮点乘法器可以用一个高精度的整数加法器来近似...
- [Differential Transformer](https://arxiv.org/abs/2410.05258)：Transformer 往往会对无关的上下文过度分配注意力。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时放大了对相关上下文的关注。具体来说，t...

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1293380838628790282) (9 messages🔥):

> - `Memcached support in LangChain`
> - `LiteLLM prompt caching and streaming`
> - `Natural language to SQL query limitations`
> - `SQL chain with models other than GPT 3.5`
> - `Integrating Livekit with LangChain`

- **寻求 LangChain 中的 Memcached 支持**：一名成员正在探讨在 LangChain 中添加对 **pymemcache** 的支持是否足够，或者是否也需要像 **python-memcached** 或 **pylibmc** 这样的多个 Memcached 客户端。
  
  - 该请求旨在增强 LangChain 生态系统中缓存选项的灵活性。
- **LiteLLM 的流式传输与缓存问题**：一名成员在使用开启了流式传输的 **LiteLLM** 时遇到了无法检索缓存 Token 的问题，并询问了确保 Token 缓存功能的最佳实践。
  
  - 他们链接到了 [LiteLLM](https://docs.litellm.ai/) 的有用资源，强调 *token stream responses* 可能会干扰缓存机制。
- **自然语言转 SQL 查询的限制**：一位用户表达了对在不信任 LLM 指令的情况下，如何有效将 SQL 查询限制在特定 ID 的担忧，并寻求维持查询生成规范的替代方法。
  
  - 另一名成员建议，可能需要按 ID 分组才能有效地过滤结果。
- **GPT 3.5 以外模型的 SQL 链兼容性**：有人提出了关于 SQL 链与 **GPT 3.5** 以外模型的兼容性问题，特别是当这些尝试经常产生错误响应时。
  
  - 一名成员报告说，通过明确列名和问题表述，使用 **4o-mini** 取得了成功。
- **对 Livekit 与 LangChain 集成的兴趣**：一名成员询问了将 **Livekit** 与 LangChain 集成以增强其实时应用功能的可能性。
  
  - 他们还表达了构建 **RAG bot** 的愿望，表明了对使用 LangChain 进行高级应用开发的兴趣。

**提到的链接**：

- [LiteLLM - Getting Started | liteLLM](https://docs.litellm.ai/)：https://github.com/BerriAI/litellm
- [Quickstart | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/use_cases/sql/quickstart/)：在本指南中，我们将介绍在 SQL 数据库上创建问答链和 Agent 的基本方法。这些系统将允许我们提出关于 SQL 数据库中数据的问题，并获得一个 n...

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1293314631057539113) (8 messages🔥):

> - `Mozilla AI 开源演讲`
> - `使用 --stdin 标志的困惑`
> - `LLM 与确定性输出`
> - `模型更新的影响`
> - `代码结果的可变性`

- **准备好参加 Mozilla AI 演讲！**：下周，我们很高兴邀请到 **Mozilla AI** 的成员来分享关于开源倡议的精彩演讲。不要错过这个深入了解的机会！
  
  - [点击此处参加活动](https://discord.gg/open-interpreter-1146610656779440188?event=1293314042596950067) 获取见解。
- **对 --stdin 标志的困惑**：一位用户对如何使用 **--stdin** 标志表示困惑，并提到在文档中找不到指导。这突显了文档清晰度方面的差距。
  
  - 文档中需要进一步澄清，以帮助用户有效地利用此功能。
- **LLM 在相同种子下保持确定性**：一场讨论揭示了如果使用相同的种子（seed）和输入，**LLM** 可以是确定性的，这与普遍看法相反。ChatGPT 在每次请求时随机化种子以引入非确定性。
  
  - 关键的一点是，使用相同的输入并将 temperature 设置为 **0** 应该会产生一致的结果。
- **模型更新带来的不可预测性**：有人担心 ChatGPT 中的**模型更新**可能会随着时间的推移影响结果的一致性。模型的变化可能会导致变动，从而破坏之前的确定性行为。
  
  - 用户强调，即使代码保持不变，更新也可能引入不可预测性。
- **不同系统间的代码结果差异**：一位成员指出，系统或 Python 的更新可能会影响代码行为，导致结果多变。例如，访问用户 token 可能会改变执行路径。
  
  - 这种多变性强调了受控环境对于获得一致结果的重要性。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 messages):

8i8__papillon__8i8d1tyr: [https://www.youtube.com/watch?v=kNj0O7cKCU4](https://www.youtube.com/watch?v=kNj0O7cKCU4)

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1293313517390135407) (3 messages):

> - `在 Linux 上使用 clang 后端的 exo`
> - `Nix 软件包问题`
> - `Tinygrad 调试模式观察`
> - `针对 clang 的 Pull Request #6945`
> - `bf16 自动转换为 float32`

- **exo 在 Linux 的 clang 后端上失败**：一位用户报告了在 Linux 上使用 clang 后端运行 `exo` 时出现错误，具体是在调用 `clang` 命令时失败，并出现与 MetaOps.KERNEL 相关的 lowering 错误。
  
  - 他们提到该问题在两个系统上均有出现，并怀疑这可能与 Nix 软件包系统有关。
- **Tinygrad 调试模式显示崩溃前的活动**：在运行 `TINYGRAD_DEBUG=2` 时，详细的活动日志显示在崩溃前有数百个操作，表明进程在失败前运行了一段时间。
  
  - 日志包括 **DISK** 操作和 **CLANG** 复制过程，但最终以崩溃告终。
- **关于通过 GitHub Pull Request #6945 进行潜在修复的讨论**：一位用户建议 [Pull Request #6945](https://github.com/tinygrad/tinygrad/pull/6945) 可能是他们遇到的 clang 后端问题的修复方案。
  
  - 该 PR 涉及使用 rewriter hooks 来实现从 bf16 到 float32 的自动转换（autocasting），尽管重写规则仍需修正。

**提到的链接**：[WIP: autocast bf16 to float32 for clang by 1ntEgr8 · Pull Request #6945 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/6945)：我使用 renderer 的 extra_matcher 字段挂载了 rewriter（模仿 PTX）。重写规则目前不正确（未执行移位），稍后会修复。我已经能够编译并运行...

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1293383653333930076) (2 messages):

> - `Fashion MNIST PR`
> - `数据集建议`
> - `学习资源`

- **Fashion MNIST 为 tinygrad 学习者增加挑战**：一位成员创建了一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/6961)，引入 **Fashion MNIST** 作为 **tinygrad** 学习者的中间数据集，提供一个比 **MNIST** 复杂但比 **CIFAR-10** 简单的挑战。
  
  - 该 PR 旨在通过额外的资源帮助学习者，提供一种扩展技能的有效方式。
- **呼吁添加更多数据集**：一位成员询问社区是否希望看到为 tinygrad 添加和测试更多数据集，以进一步增强学习机会。
  
  - 这一建议突显了人们对不断增加可供学习者使用的数据集选项的共同兴趣。

**提及的链接**：[added beautiful fashion mnist and example by Kinvert · Pull Request #6961 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/6961)：学习 tinygrad 的人可能希望在 MNIST 和 CIFAR-10 之间有一个难度阶梯。这就是我为了继续学习 tinygrad 亲自在这里做的事情。可能对其他人有用。由你们决定是否...

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1293323902960926750) (1 messages):

> - `分层生成`
> - `Stable Cascade 模型`

- **探索分层生成模型**：一位成员分享了他们的博客文章 [A Theory for Coupling Generation and Compression](https://theadamcolton.github.io/a-theory-for-coupling-generation-and-compression)，讨论了一个类似于 **Stable Cascade** 的分层模型生成框架。
  
  - 文章强调了生成模型中常见的**范式 (paradigm)**，即先训练一个**分解器 (decomposer)**，并突出了其在 LLM 和图像生成器中的应用。
- **当前生成范式的挑战**：当前的生成模型设计通常遵循相同的模式，即从一个在训练生成器之前压缩数据的分解模型开始。
  
  - 这种方法在 LLM 中非常普遍，并产生了一些影响，例如尽管加快了训练和推理速度，但 LLM 在子字符拼写方面表现挣扎。

**提及的链接**：[coupling generation and compression](https://theadamcolton.github.io/a-theory-for-coupling-generation-and-compression)：未找到描述

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1293514429677178881) (3 messages):

> - `o1-preview 泛化`
> - `o1-mini 性能`
> - `AIW 任务问题`
> - `TruthfulQA 成功`

- **o1-preview 展示了强大的 zero-shot 泛化能力**：初步实验表明，与之前的模型相比，**o1-preview** 在 **zero-shot (weak) out-of-distribution generalization** 方面展现了真正的飞跃。
  
  - 相比之下，**o1-mini** 远远落后，仅与之前的 SOTA 持平，这突显了**预训练规模 (pre-training scale)** 的明显影响。
- **o1-preview 在较简单的任务上表现挣扎**：尽管明显优于之前的模型，**o1-preview** 在 **AIW** 等较简单的任务上仍面临挑战。
  
  - 这引发了人们对它能有效应对复杂的奥林匹克竞赛和博士级问题解决能力的质疑。
- **o1 证明了对 TruthfulQA 的理解**：**o1** 在 **TruthfulQA** 上展示了令人鼓舞的结果，特别是在有效理解常见误解方面。
  
  - 这表明，虽然它有局限性，但在某些理解任务中表现出色。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1293366768386965546) (3 messages):

> - `The Cat API`
> - `Cat image fetching tools`
> - `Cat breeds data`

- **Fetching Random Cat Images from The Cat API**: 演示了一个使用 [The Cat API](https://api.thecatapi.com/v1/images/search) **获取随机猫咪图像**的新功能。实现过程包括创建一个 `Cat` 模型并使用 HTTP 客户端无缝抓取图像。
- **Exploring Cat Breeds with Limitations**: 展示了一种**获取猫品种**的方法，并带有返回数量限制选项。代码片段显示，前几个品种被检索并结构化为 `CatBreed` 模型以便于访问。
- **Demonstration Video Links Shared**: 分享了演示视频链接，重点介绍了猫咪图像和品种获取功能。这些视频提供了如何有效实现所讨论工具的视觉指南。

**提到的链接**:

- [Cool Stuff for Batman 🦇](https://www.loom.com/share/bfcbab5223214960a75cc230d7d5f883?sid=d9d647e0-979d-4a76-8f1d-5ddc5450ae7a): 你好，我是 Sean Chatman，一名正在寻求全职工作的全栈前端开发人员。在这段名为 "Cool Stuff for Batman" 的视频中，我深入探讨了在 APS 模型中配置会议并发，展示了...
- [Tool Usage with ToolMixin](https://www.loom.com/share/269f23307fd24aa591c7e63ff7126b91): 你好，我是 Sean Chatman，一名正在寻求全职机会的资深 TypeScript React 开发人员。我开发了 DSL Model Framework，这是一个通过内置 Jinja 简化 DSPy 使用的工具...

---

### **DiscoResearch ▷ #**[**general**](https://discord.com/channels/1178995845727785010/1182877486854451271/1293464014461993001) (1 messages):

> - `Whisper Turbo German Model`
> - `Speech Recognition Optimization`

- **Whisper Turbo German Model Halves Error Rate**: 根据[来源](https://huggingface.co/primeline/whisper-large-v3-turbo-german)，与早期模型相比，新模型 **Whisper Turbo German** 在某些基准测试中显著将错误率降低了一半。
  
  - 该模型专门针对德语的**转录 (transcription)**、**语音命令**和**自动字幕**等各种应用进行了优化。
- **Applications of Whisper Turbo Model**: **Whisper Turbo German** 模型的应用包括德语口语转录、自动字幕和基于语音的搜索查询。
  
  - 它为文字处理程序提供**听写功能**，增强了在不同场景下的可用性。

 

**提到的链接**: [primeline/whisper-large-v3-turbo-german · Hugging Face](https://huggingface.co/primeline/whisper-large-v3-turbo-german): 未找到描述

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1293618982183768094) (1 messages):

> - `Writer's Palmyra-X-004 model`
> - `DevRel inquiries`

- **Writer's Palmyra-X-004 Model Update Request**: Writer 的 DevRel 负责人 Sam Julien 在收到 CTO Waseem AlShikh 的邮件后，咨询了如何将最新的 **Palmyra-X-004** 模型添加到排行榜。
  
  - *我们需要提交 PR 吗？* Sam 对其模型在内部取得的**令人印象深刻的结果**表示有信心。
- **Follow-up on Leaderboard Submission Process**: Sam 询问是否需要提交 **PR** 才能将 Palmyra-X-004 模型添加到排行榜。
  
  - 这一咨询凸显了在确保其成就获得社区认可方面的积极态度。

 

---

---

---

---

---

{% else %}

> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}