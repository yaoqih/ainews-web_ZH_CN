---
companies:
- stability-ai
- google-deepmind
- nvidia
date: '2024-08-02T01:05:39.247788Z'
description: '**Stability AI** 联合创始人 Rombach 推出了 **FLUX.1**，这是一款全新的文生图模型，包含三个版本：pro（仅限
  API）、dev（开源权重，非商业用途）和 schnell（采用 Apache 2.0 协议）。根据 Black Forest Labs 的 ELO 评分，FLUX.1
  的表现超越了 **Midjourney** 和 **Ideogram**，并计划未来扩展至文生视频领域。


  **Google DeepMind** 发布了 **Gemma-2 2B**，这是一个拥有 20 亿参数的开源模型。在 Chatbot Arena 排行榜上，它的表现优于
  **GPT-3.5-Turbo-0613** 和 **Mixtral-8x7b** 等更大规模的模型，并针对 NVIDIA TensorRT-LLM 进行了优化。此次发布还包括安全分类器
  (ShieldGemma) 和稀疏自编码器分析工具 (Gemma Scope)。


  此外，相关讨论强调了基准测试的差异以及美国政府对开源权重 AI 模型的支持。同时，也有观点对 AI 编程工具在提升生产力方面的实际效果提出了批评。'
id: 356b3235-97c4-4179-9dde-7aaa89d13d06
models:
- gemma-2-2b
- gpt-3.5-turbo-0613
- mixtral-8x7b
- flux-1
original_slug: ainews-rombach-et-al-flux1-prodevschnell-31m-seed
people:
- rohanpaul_ai
- fchollet
- bindureddy
- clementdelangue
- ylecun
- svpino
title: Rombach 等人：发布 FLUX.1 [pro|dev|schnell]，Black Forest Labs 获 3100 万美元种子轮融资。
topics:
- text-to-image
- text-to-video
- model-benchmarking
- open-weight-models
- model-distillation
- safety-classifiers
- sparse-autoencoders
- ai-coding-tools
---

<!-- buttondown-editor-mode: plaintext -->**团队和 3100 万美元就是重现 Stability 所需的一切？**

> 2024年7月31日至8月1日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务（**335** 个频道和 **3565** 条消息）。预计节省阅读时间（以 200wpm 计算）：**346 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今年我们一直在密切关注 Rombach 等人的工作，因为他[发布了 Stable Diffusion 3](https://buttondown.email/ainews/archive/ainews-to-be-named-7776/)，随后[离开了 Stability AI](https://buttondown.email/ainews/archive/ainews-the-last-hurrah-of-stable-diffusion/)。他在文本生成图像（text-to-image）领域的最新尝试是 FLUX.1，我们非常喜欢在这里展示精美的图片，因此这里展示了它执行各种标准任务的效果，从超现实主义到奇幻风格，再到写实摄影以及长文本 prompting：

 
![image.png](https://assets.buttondown.email/images/25ecba8c-6520-4e00-8400-18bddffeaeba.png?w=960&fit=max)
 

这三个变体涵盖了不同的尺寸和许可范围：

- pro：仅限 API
- dev：权重开放（open-weight），非商业用途
- schnell：Apache 2.0

 
![image.png](https://assets.buttondown.email/images/17ac14c1-e394-4ce7-b86f-36234df028d7.png?w=960&fit=max)
 

根据 Black Forest Labs 自己的 ELO 评分，所有三个变体都超越了 Midjourney 和 Ideogram：

 
![image.png](https://assets.buttondown.email/images/1f6fa983-3aec-4842-bdeb-234d20dc77af.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/ad73321c-a0d8-4df6-97a6-a15a89533a85.png?w=960&fit=max)
 

他们还宣布接下来将致力于 SOTA 级别的 Text-to-Video 研究。总而言之，这是我们在过去一年中看到的实力最强、最自信的模型实验室发布之一。

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

**Gemma 2 发布与 AI 模型进展**

Google DeepMind 发布了 Gemma 2，这是一个新的开源 AI 模型系列，其中包括一个 20 亿参数的模型 (Gemma-2 2B)，该模型取得了令人印象深刻的性能：

- [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1818681376323096994) 宣布推出 Gemma-2 2B，这是一款全新的 20 亿参数模型，在其同尺寸模型中提供顶尖性能，并能在各种硬件上高效运行。

- [@lmsysorg](https://twitter.com/lmsysorg/status/1818694982980845685) 报告称，Gemma-2 2B 在 Chatbot Arena 上获得了 1130 分，表现优于其尺寸 10 倍的模型，并超过了 GPT-3.5-Turbo-0613 (1117) 和 Mixtral-8x7b (1114)。

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1818697538360295897) 强调 Gemma-2 2B 在 Chatbot Arena 上的表现优于所有 GPT-3.5 模型，它通过蒸馏（distillation）技术向大型模型学习，并针对 NVIDIA TensorRT-LLM 进行了优化，适用于各种硬件部署。

- [@fchollet](https://twitter.com/fchollet/status/1818730987435835881) 指出 Gemma 2-2B 是同尺寸中最好的模型，在 lmsys Chatbot Arena 排行榜上超越了 GPT 3.5 和 Mixtral。

此次发布还包括其他组件：

- ShieldGemma：安全分类器，用于检测有害内容，提供 2B、9B 和 27B 三种尺寸。
- Gemma Scope：使用稀疏自编码器 (SAEs) 来分析 Gemma 2 的内部决策过程，拥有超过 400 个 SAEs，覆盖了 Gemma 2 2B 和 9B 的所有层。

**AI 模型基准测试与对比**

- [@bindureddy](https://twitter.com/bindureddy/status/1818738366466412601) 批评了 Human Eval 排行榜，声称其存在刷榜行为，无法准确代表模型性能。他们认为尽管排行榜排名如此，但 GPT-3.5 Sonnet 优于 GPT-4o-mini。

- [@Teknium1](https://twitter.com/Teknium1/status/1818709594560249922) 指出了 Gemma-2 2B 在 Arena 分数与 MMLU 表现之间的差异，指出其 Arena 分数高于 GPT-3.5-turbo，但 MMLU 分数仅为 50，而 3.5-turbo 为 70。

**开源 AI 与政府立场**

- [@ClementDelangue](https://twitter.com/ClementDelangue/status/1818573917033730230) 分享称，美国商务部发布了政策建议，支持强大 AI 模型关键组件的可用性，并认可“权重开放（open-weight）”模型。

- [@ylecun](https://twitter.com/ylecun/status/1818589409685483961) 称赞了支持权重开放/开源 AI 平台的 NTIA 报告，认为现在是时候放弃那些基于虚构风险、扼杀创新的法案了。

**AI 在编程与开发中的应用**

- [@svpino](https://twitter.com/svpino/status/1818708310637658153) 讨论了当前 AI 编程工具（如 Cursor, ChatGPT 和 Claude）的局限性，指出它们在编写代码方面并未显著提高生产力。

- [@svpino](https://twitter.com/svpino/status/1818708333588791498) 强调了“被动 AI（passive AI）”工具的潜力，这些工具在后台运行，无需显式查询即可提供建议并识别代码中的问题。

**其他值得关注的 AI 进展**

- [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1818686489749787038) 展示了实时视频生成技术，在 11 秒内生成了 10 秒的视频。

- [@mervenoyann](https://twitter.com/mervenoyann/status/1818675981634109701) 讨论了 SAMv2 (Segment Anything Model 2)，它为视频分割引入了一项名为“masklet prediction”的新任务，表现优于之前的 SOTA 模型。

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1818659445640950110) 分享了关于更快的三值推理（ternary inference）的信息，允许 3.9B 模型运行速度与 2B 模型一样快，且仅占用 1GB 内存。

**梗与幽默**

- [@bindureddy](https://twitter.com/bindureddy/status/1818613179511193720) 调侃 Apple Vision Pro 被用户抛弃，可能成为 Apple 历史上最大的败笔。

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1818754584430702655) 分享了一条关于 "Friend" 噱头的幽默推文。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Google 的 Gemma 2 发布与生态系统**

- **[Google 刚刚发布了 3 款新的 Gemma 产品 (Gemma 2 2B, ShieldGemma 和 Gemma Scope)](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/)** ([Score: 143, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1egrjgp/google_just_launched_3_new_gemma_products_gemma_2/))：Google 扩展了其 Gemma AI 阵容，推出了三款新产品：**Gemma 2 2B**、**ShieldGemma** 和 **Gemma Scope**。虽然帖子中未提供这些产品的具体细节，但此次发布表明 Google 正在继续开发和丰富 Gemma 系列的 AI 产品。

- **Gemma-2 2b 4bit GGUF / BnB 量化 + 支持 Flash Attention 的 2 倍速微调！** ([Score: 74, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1egrzp7/gemma2_2b_4bit_gguf_bnb_quants_2x_faster/)): Google 发布了 **Gemma-2 2b**，该模型在来自更大 **LLM** 的 **2 万亿 token** 蒸馏输出上进行了训练。帖子作者上传了 2b、9b 和 27b 模型的 **4bit 量化版本**（bitsandbytes 和 GGUF），并开发了一种微调速度提升 **2 倍**且**显存（VRAM）**占用减少 **63%** 的方法，同时为 Gemma-2 整合了 **Flash Attention v2** 支持。他们提供了各种资源的链接，包括 [Colab notebooks](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing)、[Hugging Face 上的量化模型](https://huggingface.co/unsloth/gemma-2-it-GGUF)，以及用于 Gemma-2 instruct 的[在线推理聊天界面](https://colab.research.google.com/drive/1i-8ESvtLRGNkkUQQr_-z_rcSAIo9c3lM?usp=sharing)。

- **[Google 低调发布了一个稀疏自编码器（sparse auto-encoder）来解释 Gemma 2 和 9b。这是一个他们整理的 Google Colab，可以帮你入门。超级令人兴奋，我希望 Meta 也能效仿！](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing)** ([Score: 104, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1eh4wja/google_quietly_released_a_sparse_autoencoder_to/)): Google 发布了一个用于解释 **Gemma 2** 和 **9b** 模型的**稀疏自编码器**，并提供了一个 **Google Colab** 笔记本帮助用户入门。此次发布旨在增强这些语言模型的可解释性，可能为提高 AI 开发透明度树立先例，发帖者希望 **Meta** 等其他公司也能跟进。
  - 该**稀疏自编码器**工具允许可视化每个 **token** 的层激活，从而可能实现对**拒绝移除（refusal removal）**、**感应头（induction heads）**以及模型谎言检测的研究。用户可以探索安全研究中容易实现的目标（low-hanging fruit），并衡量**微调**对特定概念的影响。
  - 该工具为**运行时、低成本微调**开启了可能性，以提升 AI 模型中的某些情绪或主题。这可以应用于创建动态 AI 体验，例如一个实时对模型撒谎概率进行评分的**审讯游戏**。
  - 用户讨论了如何解读该工具的图表，指出它们显示了可以量化**微调效果**的 **token 概率**。以数字字符串表示的特征激活被认为比用于分析目的的可视化仪表板更有用。


**主题 2：开源 LLM 的进展与对比**

- **Llama-3.1 8B 4-bit HQQ/校准量化模型：相对于 FP16 达到 99.3% 的性能，且推理速度极快** ([Score: 156, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1egn0yh/llama31_8b_4bit_hqqcalibrated_quantized_model_993/)): **Llama-3.1 8B** 模型已发布 **4-bit HQQ/校准量化**版本，在提供 **Transformer** 最快推理速度的同时，达到了 **FP16 99.3% 的相对性能**。这款高质量量化模型可在 [Hugging Face](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib) 上获取，为改进 AI 应用兼顾了效率与性能。

- **[只是发张图..](https://i.redd.it/y9exyzedyzfd1.png)** ([Score: 562, Comments: 74](https://reddit.com//r/LocalLLaMA/comments/1eh9sef/just_dropping_the_image/)): 该图片对比了 **OpenAI 的模型发布**与**开源替代方案**，突显了开源 AI 开发的飞速进步。它显示虽然 OpenAI 在 **2020 年 6 月**发布了 **GPT-3**，在 **2022 年 11 月**发布了 **ChatGPT**，但 **BLOOM**、**OPT** 和 **LLaMA** 等开源模型在 **2022 年 6 月至 12 月**期间接连发布，**Alpaca** 紧随其后于 **2023 年 3 月**发布。
  - 用户批评 **OpenAI** 缺乏开放性，评论包括 *“OpenAI 完全封闭。真是讽刺。”* 并建议将其更名为“**ClosedAI**”或“**ClosedBots**”。一些人认为 OpenAI 靠公众炒作和作为该领域先行者的品牌知名度维持。
  - Google 的 **Gemma 2** 受到好评，用户注意到其令人惊讶的质量和个性。一位用户称其 *“在许多方面优于 L3”*，并表达了对可能具备多模态和更长上下文的 **Gemma 3** 的期待。
  - **Mistral AI** 因其在资源有限的情况下（相比大公司）取得的快速进展而受到赞赏。用户建议根据团队规模和可用资源进行标准化对比，以突出 Mistral 的成就。

- **Google 的 Gemma-2-2B 对比 Microsoft Phi-3：医疗领域 Small Language Models 的比较分析** ([Score: 65, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1eh3dei/googles_gemma22b_vs_microsoft_phi3_a_comparative/))：一项针对 **Google 的 Gemma-2-2b-it** 和 **Microsoft 的 Phi-3-4k** 模型在医疗领域未经 fine-tuning 表现的比较分析显示。根据 [Aaditya Ura 的推文](https://x.com/aadityaura/status/1818855166260519407)分享，**Microsoft 的 Phi-3-4k** 以 **68.93%** 的平均得分胜出，而 **Google 的 Gemma-2-2b-it** 的平均得分为 **59.21%**。
  - 用户批评了原始分析中的**图表颜色选择**，强调了数据比较中视觉呈现的重要性。
  - 讨论围绕所使用的具体 **Phi-3 模型**展开，推测其为 **3.8B Mini 版本**。用户还询问了针对 **PubMed 数据集**的 **fine-tuning 技术**。
  - 关于在**医疗 QA 数据集**上评估 **small LLMs** 的相关性引发了辩论。一些人认为这对于评估医学知识很重要，而另一些人则指出 LLMs **已经被用于**回答医疗问题，特别是在医生资源有限的地区。


**主题 3. LLM 的硬件与推理优化**

- **[哇，SambaNova 在 Llama 405B 上通过其 ASIC 硬件实现了超过 100 tokens/s 的速度，而且无需注册即可使用。](https://i.redd.it/9bxbfajq1xfd1.png)** ([Score: 247, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1egxxc4/woah_sambanova_is_getting_over_100_tokenss_on/))：**SambaNova** 在 AI 硬件性能方面取得了突破，利用其 **ASIC 硬件**在 **Llama 405B** 模型上实现了每秒超过 **100 tokens** 的生成速度。该技术现在无需任何注册过程即可供用户使用，有可能使高性能 AI 推理能力的使用变得民主化。

- **[发布你运行 Llama 3.1 70B 的 tokens per second](https://i.redd.it/1l6qck24ywfd1.png)** ([Score: 61, Comments: 124](https://reddit.com//r/LocalLLaMA/comments/1egxdpt/post_your_tokens_per_second_for_llama3170b/))：该帖子请求用户分享他们运行 **Llama 3.1 70B 模型**的 **tokens per second (TPS)** 性能基准测试。虽然帖子本身未提供具体的性能数据，但旨在收集和比较来自不同用户和硬件设置运行该大型语言模型的 TPS 指标。

- **[70B 我来了！](https://i.redd.it/kyxk7s1f0tfd1.jpeg)** ([Score: 216, Comments: 65](https://reddit.com//r/LocalLLaMA/comments/1eggumi/70b_here_i_come/))：帖子作者正准备使用高端 GPU 设置运行 **70B 参数模型**。他们对即将拥有的处理大型语言模型的能力感到兴奋，正如充满热情的标题“70B 我来了！”所暗示的那样。
  - 用户讨论了**散热管理**，其中一位提到对两块 **3090 FE GPU** 进行 **undervolting** 以获得更好的性能。原帖作者使用了一个气流良好的 **Meshify 机箱**，并在不需要时禁用 3090。
  - 性能基准测试被分享，一位用户报告在使用 **AWQ** 和 **LMDeploy** 运行 **Llama 3.1 70B** 模型时达到了 **35 tokens per second**。另一位推荐了一个用于监控 **GDDR6 显存温度**的 [GitHub 工具](https://github.com/olealgoritme/gddr6)。
  - 针对 **3090 显存过热**的担忧被提出，特别是在气候较温暖的地区。一位用户在进行 **Stable Diffusion** 图像生成时遇到了崩溃，最后不得不拆除机箱侧面板以获得更好的冷却效果。


**主题 4. LLM 开发的新工具与框架**

- **PyTorch 刚刚发布了自己的 LLM 解决方案 - torchchat** ([Score: 135, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1eh6xmq/pytorch_just_released_their_own_llm_solution/))：**PyTorch** 发布了 **torchchat**，这是一个用于在服务器、桌面和移动设备等各种设备上本地运行 **Large Language Models (LLMs)** 的新解决方案。该工具支持 **Llama 3.1** 等多个模型，提供 **Python** 和原生执行模式，并包含 **eval** 和 **quantization** 功能，GitHub 仓库地址为 [https://github.com/pytorch/torchchat](https://github.com/pytorch/torchchat)。
  - 一位用户在 **NVIDIA GeForce RTX 3090** 上测试了 **torchchat** 与 **Llama 3.1**，达到了 **26.47 tokens/sec**。相比之下，**vLLM** 最初达到 **43.2 tokens/s**，在更高 batch sizes 下可达 **362.7 tokens/s**。
  - 讨论集中在性能优化上，包括使用 **--num-samples** 以在 warmup 后获得更具代表性的指标，使用 **--compile** 和 **--compile-prefill** 以启用 PyTorch JIT，以及使用 **--quantize** 进行模型 quantization。
  - 用户询问了对 AMD GPU 的 **ROCm 支持**、与 **Mamba 模型**的兼容性，以及与 **Ollama** 和 **llama.cpp** 等其他框架的比较。

## Reddit AI 动态汇总

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与应用**

- **Google DeepMind 的 Diffusion Augmented Agents**：来自 [Google DeepMind 的一篇新论文](https://arxiv.org/abs/2407.20798) 介绍了 Diffusion Augmented Agents，这可能会提升 AI 在复杂环境中的能力。(r/singularity)

- **AI 在前列腺癌检测方面表现优于医生**：一项 [研究发现 AI 检测前列腺癌的准确率比医生高出 17%](https://www.ndtv.com/science/ai-detects-prostate-cancer-17-more-accurately-than-doctors-finds-study-6170131)，展示了 AI 在医学诊断方面的潜力。(r/singularity)

**AI 产品与用户体验**

- **ChatGPT 高级语音模式**：一段 [视频演示](https://v.redd.it/r1hyqf4jixfd1) 展示了 ChatGPT 的语音模式在模仿航空公司飞行员时，因触发内容准则而突然停止。(r/singularity)

- **OpenAI 改进的对话式 AI**：一位 [用户报告](https://www.reddit.com/r/OpenAI/comments/1egrvr6/short_demo/) 称，在 OpenAI 的最新更新中，对话流和教育功能得到了提升，他在 1.5 小时的通勤过程中利用该功能学习了 GitHub 仓库的相关知识。(r/OpenAI)

- **对 AI 可穿戴设备的批评**：一篇 [帖子批评了](https://www.reddit.com/r/singularity/comments/1egjby2/man_this_is_dumb/) 一款新的 AI 可穿戴设备，将其与之前失败的尝试（如 Humane Pin 和 Rabbit R1）进行了比较。用户们讨论了该设备在功能和商业模式上可能存在的问题。(r/singularity)

**AI 与数据权利**

- **Reddit CEO 要求为 AI 数据访问付费**：[Reddit 的 CEO 表示微软应该为搜索该网站付费](https://www.theverge.com/2024/7/31/24210565/reddit-microsoft-anthropic-perplexity-pay-ai-search)，引发了关于数据权利和用户生成内容补偿的讨论。(r/OpenAI)


---

# Discord AI 动态汇总

> 摘要之摘要的摘要

## Claude 3.5 Sonnet


**1. 新型 AI 模型与能力**

- **Llama 3.1 发布引发争论**：Meta 发布了 **Llama 3.1**，包括一个在 15.6 万亿 token 上训练的 4050 亿参数模型，[Together AI 的博客文章](https://www.together.ai/blog/llama-31-quality)引发了关于不同供应商实现差异影响模型质量的辩论。
   - AI 社区参与了关于结果可能存在“樱桃采摘”（cherry-picking）以及严谨、透明的评估方法论重要性的讨论。[Dmytro Dzhulgakov 指出](https://x.com/dzhulgakov/status/1818753731573551516)了 Together AI 展示案例中的差异，强调了进行一致性质量测试的必要性。
- **Flux 搅动文本生成图像领域**：由原 Stable Diffusion 团队成员组成的 **Black Forest Labs** [推出了 FLUX.1](https://x.com/bfl_ml/status/1819003686011449788)，这是一套全新的最先进文本生成图像模型，其中包括一个在非商业和开源许可下提供的 12B 参数版本。
   - FLUX.1 模型因其令人印象深刻的能力而受到关注，用户注意到它在渲染手部和手指等身体末端方面的优势。[FLUX.1 的专业版 (pro version)](https://replicate.com/black-forest-labs/flux-pro) 已经可以在 Replicate 上进行测试，展示了文本生成图像领域的快速发展。
  


**2. AI 基础设施与效率提升**

- **MoMa 架构提升效率**：Meta 引入了 **MoMa**，这是一种用于混合模态语言建模的新型稀疏早期融合（sparse early-fusion）架构，显著提高了预训练效率，详见其[最新论文](https://arxiv.org/pdf/2407.21770)。
   - 根据 [Victoria Lin](https://x.com/VictoriaLinML/status/1819037439681565178) 的说法，MoMa 在文本训练中实现了约 3 倍的效率提升，在图像训练中实现了 5 倍的提升。该架构采用混合专家（MoE）框架，并配有特定模态的专家组，用于处理交错的混合模态 token 序列。
- **GitHub 集成 AI 模型**：GitHub [宣布了 GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/)，这是一项新功能，直接将行业领先的 AI 工具带给其平台上的开发者，旨在弥合编码与 AI 工程之间的鸿沟。
   - 这种集成旨在让 GitHub 庞大的开发者群体更容易接触到 AI，从而可能在大规模范围内改变编码与 AI 的交互方式。社区推测此举是否意在通过将 AI 能力集成到开发者的现有工作流中，来与 Hugging Face 等平台竞争。
  


**3. AI 伦理与政策进展**

- **NTIA 倡导开放 AI 模型**：美国国家电信和信息管理局（NTIA）[发布了一份报告](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation)，支持 AI 模型的开放性，同时建议进行风险监测，以指导美国的政策制定者。
   - 社区成员注意到 NTIA 直接向白宫汇报，这使其关于 AI 模型开放性的政策建议具有重大分量。该报告可能会影响美国未来的 AI 监管和政策方向。
- **AI 信任中的水印辩论**：围绕水印在解决 AI 信任问题上的有效性展开了辩论，一些人认为它仅在机构环境中有效，无法完全防止滥用。
   - 讨论表明，需要更好的文化规范和信任机制，而不仅仅是水印，来应对深度伪造（deepfakes）和虚假内容的传播。这突显了在建立 AI 生成内容的信任和真实性方面面临的持续挑战。

---

# PART 1: 高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **神经网络的新型网页模拟器**：一个新的 [神经网络模拟](https://starsnatched.github.io/) 工具邀请 **AI 爱好者** 在线尝试不同的神经网络配置。
   - 该模拟器旨在揭示神经网络行为的神秘面纱，为用户提供修改和理解 **神经动力学 (neural dynamics)** 的交互式体验。
- **可迁移 AI 智慧的蓝图**：IBM 详细解析了 [知识蒸馏 (Knowledge Distillation)](https://www.ibm.com/topics/knowledge-distillation)，阐明了将庞大的“教师”模型中的见解注入精简的“学生”模型的过程。
   - **知识蒸馏** 作为一种模型压缩和高效知识迁移的方法脱颖而出，对 **AI 可扩展性 (AI scalability)** 至关重要。
- **记录模型里程碑的交互式热力图**：一个创新的 [热力图空间 (heatmap space)](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap) 记录了 **AI 模型发布** 情况，因其可能集成到 Hugging Face 个人资料中而受到社区关注。
   - 该工具展示了模型发展趋势的深刻视觉聚合，旨在提高 **AI 演进节奏** 的可见性和理解。
- **为 Solr 构建语义解析器**：一位成员寻求关于教导 **大语言模型 (LLM)** 解析 [Apache Solr](https://solr.apache.org/) 查询的建议，旨在生成包含产品信息的 **JSON 响应**。
   - 在没有现成训练数据集的情况下，挑战在于有条理地引导 LLM 以增强 **搜索功能** 和用户体验。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Chameleon 架构实现飞跃**：由 **Chameleon** 的创建者开创的一种 **新型多模态架构** 拥有显著的效率提升，详情可见 [学术论文](https://arxiv.org/pdf/2407.21770)。
   - **Victoria Lin** 在 Twitter 上提供了见解，指出 *文本训练提升了约 3 倍*，*图像训练提升了 5 倍*，使 **MoMa 1.4B** 表现出色 ([来源](https://x.com/VictoriaLinML/status/1819037439681565178))。
- **解码投机采样 (Speculative Decoding)**：投机采样机制是一个热门话题，有观点认为较小的草稿模型 (draft models) 可能会影响输出分布，除非通过 **拒绝采样 (rejection sampling)** 等技术进行修正。
   - 一个 [YouTube 资源](https://www.youtube.com/watch?v=hm7VEgxhOvk) 进一步解释了投机采样，暗示了该过程中速度与保真度之间的平衡。
- **Bitnet 展现惊人速度**：**Bitnet 的微调方法** 正引起关注，据 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/) 报道，在单个 CPU 核心上达到了令人印象深刻的 **每秒 198 个 token**。
   - 通过这种微调方法产生了一个紧凑的 **74MB 模型**，预计将发布开源版本，引发了对其在未来项目中应用的期待 ([Twitter 来源](https://x.com/nisten/status/1818529201231688139))。
- **LangChain：是关键还是累赘？**：关于在以 **OpenAI API** 格式使用 **Mixtral API** 时是否有必要使用 **LangChain** 产生了争论。
   - 一些成员质疑对 LangChain 的需求，认为直接进行 API 交互可能就足够了，这引发了关于工具依赖和 API 惯例的讨论。
- **无需费用的项目参与**：社区成员询问了协助一个免费 AI 项目的方法，相关步骤将在预期的 **PR** 中列出。
   - 讨论确认了该项目的免费性质，强调了将在即将发布的 **PR** 中披露的可执行任务，简化了新贡献者的加入流程。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **多 GPU 训练从崩溃到胜利**：讨论揭示了 **multi-GPU training** 的问题，赞扬了修复方案，但也强调了初始设置的麻烦和环境调整。
   - 切换到 `llamafacs env` 是某些人成功的关键，而另一些人则选择了更具动手能力的手动 **transformers upgrade** 方法。
- **Unsloth Crypto Runner 揭晓**：阐明了 **Unsloth Crypto Runner 基于 AES/PKI** 的设计细节，解释了其从客户端到服务器的加密通信。
   - 当 `MrDragonFox` 强调 GPU 使用的必要性时，社区反响热烈，并透露了 **Skunkworks AI** 的 **open-source** 意图。
- **Qwen 持续精炼实现**：**Qwen2-1.5B-Instruct 的无损持续微调 (Continuous Fine-tuning Without Loss)** 引入了代码 FIM 和指令能力的融合，标志着一个技术里程碑。
   - 随着用户中呼吁通过教程来解决文档挑战的声音不断回响，社区精神受到鼓舞。
- **LoRA 的合并困境**：合并 **LoRA adapters** 成为关注焦点，重点在于将 4-bit 模型合并导致虚假的 16-bit 表示的风险。
   - 社区内对这些伪 **16-bit models** 传播的担忧日益增加，促使大家保持警惕。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 与 Uber One 的会员福利**：Uber One 会员现在可以免费获得 **Perplexity Pro** 订阅，有效期至 **2024 年 10 月 31 日**，提供价值 **$200** 的增强型回答引擎。
   - 要享受此福利，美国和加拿大的用户需要保持其 Uber One 订阅并建立一个新的 Perplexity Pro 账户。更多详情请访问 [Perplexity Uber One](https://pplx.ai/uber-one)。
- **Perplexity 在 AI 搜索引擎基准测试中夺冠**：在一次对比评估中，**Perplexity Pro** 超越了 Felo.ai 和 Chatlabs 等竞争对手，在 UI/UX 和查询响应方面表现出色。
   - 会员对搜索引擎的能力进行了评分，Pro Search 成为最受欢迎的功能，并在 [ChatLabs](https://labs.writingmate.ai) 等平台上被重点介绍。
- **Perplexity API 引发困惑**：讨论显示用户对 Perplexity API 输出效果不佳感到不满，认为结果质量有所下降。
   - 关于问题提示词的猜测不断增加，个人纷纷寻求改进结果的建议，并对 **Perplexity References Beta** 的访问权限表示好奇。
- **Perplexity 优化的 Flask 身份验证**：关于 Flask 的讨论强调了安全用户身份验证的必要性，推荐了 `Flask-Login` 等包，以及 [一份安全设置指南](https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw)。
   - 用户被引导至概述模型创建、用户身份验证路由和加密实践的资源。
- **OpenAI 通过 GPT-4o 开启语音未来**：**OpenAI** 推出的 ChatGPT 高级语音模式 (Advanced Voice Mode) 令人印象深刻，自 2024 年 7 月 30 日起为 Plus 订阅者提供逼真的语音交互。
   - 该更新允许增强语音功能，如情感语调变化和中断处理，记录在 [OpenAI 更新页面](https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA)上。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **生动的愿景：GPT-4o 激发图像创新**：关于 GPT-4o 图像输出能力的讨论非常热烈，用户将其与 **DALL-E 3** 进行对比，并分享了激发广泛关注的[示例](https://x.com/gdb/status/1790869434174746805)，展示了其栩栩如生且写实的图像效果。
   - 尽管 GPT-4o 令人印象深刻的输出赢得了赞誉，但其内容审核端点（moderation endpoint）也遭到了批评，呼应了 DALL-E 3 曾面临的类似担忧。
- **多才多艺的语音：显微镜下的 GPT-4o 语音实力**：AI 爱好者测试了 GPT-4o 的[语音模型能力](https://platform.openai.com/docs/guides/embeddings/use-cases)，强调了其对口音和情感范围的适应性，以及融合背景音乐和音效的能力。
   - 调查结果既有对其潜力的赞赏，也有对其表现不稳定的指出，引发了关于模型局限性和未来改进的讨论。
- **平台难题：寻求精准的 Prompt**：AI 工程领域的先行者们交流了关于 **prompt engineering** 首选平台的见解，将 **Claude 3**、**Sonnet** 以及 **Artifacts + Projects** 视为首选。
   - 用于 Prompt 评估的启发式工具成为焦点，其中 **Anthropic Evaluation Tool** 因其启发式方法被提及，而一个带有脚本的协作式 **Google Sheet** 则被提议作为一种可共享且高效的替代方案。
- **战略性订阅转变：权衡 Plus 的影响力**：社区讨论围绕取消 Plus 订阅的影响展开，透露这样做将导致无法访问自定义 GPTs。
   - 讨论还延伸到了 GPT 货币化的前提条件，强调了将大量的后续使用指标和美国本土化作为产生收入机会的标准。
- **图表困境：通过 AI 辅助规划路径**：在 AI 图表领域，参与者探讨了擅长制作视觉辅助工具的免费工具，并提到了 **ChatGPT** —— 尽管其绘图天赋仍存争议。
   - 对话还涉及了 LLM 在文本截断方面面临的挑战，建议寻求定性描述可能比要求精确的字符或单词计数更有效。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **FSDP 讨论引发热议**：一名成员批评 **FSDP** “有点烂”，引发了关于其可扩展性的辩论，而反方观点则认为它在易用性方面表现出色。
   - 话题转向了 FSDP 的场景适用性，表明尽管它具有用户友好的特性，但并非万能的解决方案。
- **分片 LLaMA 的烦恼与 vLLM 的希望**：在讨论中出现了在多节点上对 **LLaMA 405B** 进行分片的挑战，可能的解决方案涉及为更大的上下文窗口增强 **vLLM**。
   - 参与者推荐了量化等方法，一些人避开使用 vLLM，转而引导用户查看 [LLaMA 3.1 的增强细节和支持](https://blog.vllm.ai/2024/07/23/llama31.html)。
- **Megatron 的学术魅力**：**Megatron 论文** 激发了成员们讨论分布式训练相关性的兴趣，并得到了 [Usenix 论文](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)和解释性的 [MIT 讲座视频](https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab)等资源的支持。
   - 关于 Megatron 的论述延伸到了分布式训练的实践见解，参考了学术界认可的资料和 YouTube 传播的内容。
- **Triton 教程的分块 Matmul 矩阵**：针对 [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)中 `GROUP_SIZE_M` 参数的疑问浮出水面，讨论了其在优化缓存中的作用。
   - 辩论包括将 `GROUP_SIZE_M` 设置得过高如何导致效率低下，探索了硬件设计选择的微妙平衡。
- **Llama 3.1：混乱与 TorchChat 指引**：用户表示需要一个 10 行的 Python 代码片段来简化 **Llama 3.1 模型** 的使用，因为现有的 [推理脚本](https://github.com/meta-llama/llama-recipes) 被认为过于复杂。
   - 作为回应，PyTorch 发布了 [TorchChat](https://github.com/pytorch/torchchat) 作为指南，为运行 Llama 3.1 提供了急需的参考实现。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Fast 3D 的闪电发布**：Stability AI 发布了 **Stable Fast 3D**，这是一款能够在短短 **0.5 秒**内将单张图像转换为详细 3D 资产的新模型，推向了 3D 重建技术的极限。该模型对游戏和 VR 领域具有重大意义，重点在于**速度**和**质量**。[探索技术细节](https://stability.ai/news/introducing-stable-fast-3d)。
   - *“Stable Fast 3D 惊人的处理时间为 3D 框架中的快速原型设计开辟了先河。”* 用户还可以受益于可选的 remeshing 等额外功能，这些功能仅增加极少的时间，却能广泛适用于各种行业。
- **SD3 成为焦点**：社区讨论围绕 **Stable Diffusion 3 (SD3)** Medium 的利用展开，解决了加载错误并探索了模型的能力。分享的解决方案包括获取所有组件以及利用 [ComfyUI workflows](https://comfyworkflows.com/) 等工具来实现更顺畅的操作。
   - 通过社区支持和适配各种可用的 UI，用户成功解决了如 'AttributeError' 等挑战，确保了使用 **SD3** 时更无缝的创作体验。
- **解决 VAE 难题**：社区解决了一个常见问题：由于 VAE 设置，图像在渲染过程中变红。通过协作努力，找到了减轻该问题的故障排除方法。
   - 应用 '--no-half-vae' 命令成为同行推荐的修复方案，简化了艺术家的工作流，使他们在应对特定硬件解决方案时能准确地创作图像。
- **拨开 Creative Upscaler 的迷雾**：社区共同努力澄清了关于提及“Creative Upscaler”的困惑，明确其并非 Stability AI 的项目。成员们交流了其他的放大（upscaling）建议。
   - 受欢迎的技术包括 ERSGAN 的应用和采用 Transformer 技术，并从各种社区贡献的资源中汇集了针对提示词挑战的建议。
- **Flux：下一代图像生成模型**：Black Forest Labs 发布的 **Flux 模型**备受期待，社区对其在图像呈现和高效参数使用方面的增强议论纷纷。[该公告展示了潜力](https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/)，有望改变 text-to-image 领域。
   - 关于该模型 GPU 效率的讨论强调了 Nvidia 4090 可获得最佳性能，并特别赞赏了该模型在渲染手部和手指等身体末端部位方面的卓越能力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **退出代码揭示兼容性冲突**：LM Studio 用户报告了如 **6** 和 **0** 的退出代码，引发了关于系统兼容性和调试迷宫的讨论。
   - 这一困境已升级为围绕**特定系统**的特性以及可能需要更新 LM Studio 版本的讨论。
- **Gemma 2 故障引发 GPU 困扰**：运行 **Gemma 2 2B** 模型时出现了挑战，特别是在陈旧的硬件上，这促使用户呼吁发布新版本的 LM Studio。
   - 社区的反应包括同情以及分享规避硬件障碍的策略。
- **LLaMA：Embedding 之谜**：在询问 **LLaMA** 在 LM Studio 中的集成情况时，爱好者们通过 [LLM2Vec](https://github.com/McGill-NLP/llm2vec) 等项目探索了 Embedding 能力。
   - 这最终促成了关于文本编码器（text encoders）前瞻性解决方案的深入对话，以及对 Embedding 演进的兴奋。
- **深入探索 LM Studio**：成员们揭示了 LM Studio 中的 Bug，从 GPU offloading 的异常到可能与 VPN/DNS 配置有关的棘手网络错误。
   - 同行们协助定位问题并提出了可能的补丁，营造了解决技术难题的协作氛围。
- **对 LM Studio 功能的愿景**：讨论深入到了对未来 LM Studio 功能的憧憬，用户渴望增加 **TTS voices** 和 **RAG 支持的**文档交互等功能。
   - 在这些愿景中，[Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) 以及 [Papers with Code](https://paperswithcode.com/task/visual-question-answering) 上的 **Visual Question Answering (VQA)** 方法引起了关注。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **水印之困：AI 的认证焦虑**：成员们讨论了水印在 AI 信任问题中的作用，指出其有效性有限，并建议建立**文化规范**至关重要。
   - 令人担忧的是，如果没有更广泛的信任机制，水印可能无法阻止滥用和虚假内容的传播。
- **NTIA 的开放 AI 倡导：政策影响力巅峰**：[NTIA 报告](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation) 促进了 AI 模型的开放性，并建议进行勤勉的风险监测以指导政策制定者。
   - 观察人士指出，由于 NTIA 直接向白宫汇报，其政策建议具有很大分量，预示着 AI 监管可能发生转变。
- **GitHub 的模型混搭：将 AI 与代码集成**：GitHub 推出的 [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) 方便了在开发者工作流中直接访问 AI 模型。
   - 随后引发了讨论，争论这究竟是挑战 Hugging Face 等竞争对手的策略，还是 GitHub 服务产品的自然演进。
- **转发双重下降：缩放法则受到审视**：AI 研究人员讨论了 Scaling Laws 实验中验证对数似然（validation log-likelihood）的异常，特别是当具有 **1e6 序列**的模型表现不佳时。
   - 这引发了对 [BNSL 论文](https://arxiv.org/abs/2210.14891) 的引用，揭示了类似的模式，并激发了对数据集大小影响的好奇。
- **Prompt 过量生成的谜团：lm-eval 意外的倍数**：lm-eval 在 **gpqa_main** 等基准测试中使用的 Prompt 数量超过了指定的数量，这一行为引发了技术咨询和调试工作。
   - 澄清结果显示，lm-eval 中的进度条考虑了 `num_choices * num_docs`，从而解释了感知到的差异，并有助于理解工具行为。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok 的增长：xAI 不太可能收购 Character AI**：关于 xAI 收购 Character AI 以增强其 Grok 模型的[传闻](https://x.com/nmasc_/status/1818788751528935468)一直在流传，但 [Elon Musk 否认了这些说法](https://x.com/elonmusk/status/1818810438634946699)，称该信息不准确。
   - 社区思考了 Musk 言论背后的真相，并提到了此前官方否认后又确认收购的先例。
- **Black Forest Labs 诞生于 Stable Diffusion 的根基**：**Stable Diffusion** 的创始团队推出了 [Black Forest Labs](https://x.com/bfl_ml/status/1819003686011449788)，专注于先进的生成模型，引起了轰动。
   - Black Forest Labs 的 **Flux** 展示了强大的创造力，早期测试者可以在 fal 上进行尝试，预示着生成式领域潜在的变革。
- **GitHub Models 将开发者与 AI 实力结合**：GitHub 通过[推出 GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/) 在 AI 领域引起关注，向其庞大的开发者群体提供强大的 AI 工具。
   - 这一新套件旨在为开发者普及 AI 使用，可能在大规模范围内改变编程与 AI 的交互方式。
- **Apple Intelligence 为科技未来带来转折**：[Apple 最新的 AI 进展](https://www.interconnects.ai/p/apple-intelligence)承诺将应用程序更无缝地编织在一起，增强日常科技互动。
   - AI 实验室的怀疑论者质疑 Apple Intelligence 的突破性地位，而其他人则将其视为科技实用性的重要倍增器。
- **拒绝采样在 Open Instruct 中找到归宿**：[Open Instruct](https://github.com/allenai/open-instruct/pull/205) 采用了 **Rejection Sampling**（拒绝采样），这是一种通过避免常见陷阱来微调训练的方法。
   - 此举可能标志着模型训练效率的提高，也是 AI 训练领域方法论的一大进步。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Llama 3.1 触及质量辩论的敏感神经**：[Together AI 博客](https://www.together.ai/blog/llama-31-quality)通过强调由于推理提供商不同的实现实践导致的性能差异，引发了关于 Llama 3.1 的辩论，引起了对模型一致性的关注。
   - Dmytro Dzhulgakov 提醒社区注意潜在的结果挑选（cherry-picking）行为，并强调了模型评估中清晰方法论的重要性，在[此推文线程](https://x.com/dzhulgakov/status/1818753731573551516)中引发了广泛讨论。
- **Sybill 为 AI 增强型销售筹集数百万美元**：Sybill 获得了强劲的 **1100 万美元 A 轮融资**，用于完善其针对销售代表的 AI 个人助理，支持者包括 **Greystone Ventures** 等知名机构（[公告详情](https://x.com/asnani04/status/1818642568349204896)）。
   - AI 销售工具领域正随着 Sybill 的解决方案迸发创新火花，该方案通过克隆销售代表的声音来设计更相关的后续跟进。
- **Black Forest Labs 凭借 FLUX.1 取得突破**：由前 Stable Diffusion 专家组成的 **Black Forest Labs** 推出了其突破性的文本生成图像模型 **FLUX.1**，其中包括一个强大的 **12B 参数版本**（[查看公告](https://x.com/iScienceLuvr/status/1819007823339999516)）。
   - FLUX.1 的专业版目前已在 [Replicate 上线试用](https://replicate.com/black-forest-labs/flux-pro)，展示了相对于该领域其他模型的优势。
- **LangGraph Studio 为 Agentic 应用开启新视野**：LangChain 推出了 **LangGraph Studio**，旨在简化 Agentic 应用的创建和调试，推动 IDE 创新（[公告推文](https://x.com/LangChainAI/status/1819052975295270949)）。
   - 这个以 Agent 为核心的 IDE 结合了 **LangSmith**，提升了 LLM 领域开发者的效率和团队协作。
- **Meta MoMa 变革混合模态建模**：Meta 的新型 **MoMa 架构**采用 **Mixture-of-Experts 方法**，加速了混合模态语言模型的预训练阶段（[附带论文](https://arxiv.org/pdf/2407.21770)）。
   - 该架构专为有效处理和理解混合模态序列而设计，标志着该领域迈出了一大步。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **异步进展加速 BedrockConverse**：**BedrockConverse** 集成了新的异步方法，解决了 [pull request #14326](https://github.com/run-llama/llama_index/pull/14326) 中提到的未决问题，特别是 [#10714](https://github.com/run-llama/llama_index/issues/10714) 和 [#14004](https://github.com/run-llama/llama_index/issues/14004)。
   - 社区表示赞赏，强调了该贡献在提升 **BedrockConverse** 用户体验方面的重大影响。
- **LongRAG 论文见解**：由 Ernestzyj 撰写的 **LongRAG** 论文介绍了一种索引更大文档块的技术，以发挥长上下文 LLM 的潜力。
   - 这种方法开启了新的可能性，简化了检索增强生成（RAG）过程，引起了社区的兴趣。
- **Workflows 在 LlamaIndex 中大显身手**：新引入的 [llama_index 中的 workflows](https://link.to.workflows) 赋能了事件驱动的多 Agent 应用的创建。
   - 社区对这一创新表示赞赏，认为它为复杂的编排提供了一种易读且 Pythonic 的方法。
- **稳定代码库的难题**：讨论围绕确定 **LlamaIndex** 的稳定版本展开，明确了引导用户通过 pip 安装是保证稳定性的手段。
   - “稳定”一词成为焦点，将稳定性与 PyPI 上提供的最新版本联系起来，引发了进一步辩论。
- **使用 DSPy 和 LlamaIndex 进行 Prompt 实验**：成员们评估了 **DSPy** 的 Prompt 优化与 **LlamaIndex** 重写功能的对比。
   - 社区对这两个工具之间的对比探索表现出极大的热情，并考虑将它们应用于提高 Prompt 性能。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Embed with Zest: Content Structures Clarified**: 在一场技术讨论中，**Nils Reimers** 澄清说 embedding 模型会自动移除换行符和特殊符号，强调了**文本预处理**并非必不可少。
   - 这一发现表明模型在处理**噪声数据**方面具有鲁棒性，允许 AI 工程师专注于模型应用，而非繁琐的文本预处理。
- **Citations Boost Speed; Decay Dilemmas**: 一位敏锐的用户指出，在 Cohere Cloud 上使用乌克兰语/俄语时，**高 citation_quality 设置**会导致响应变慢，并注意到从 **fast** 切换到 **accurate** 解决了字符问题。
   - 虽然获得了稳定的输出，但响应速度的权衡已成为工程师之间潜在优化讨论的话题。
- **Arabic Dialects in LLMs: A Linguistic Leap**: 当 LLM **Aya** 在各种**阿拉伯语方言**中生成准确文本时，社区表达了惊讶，并对在以英语为主的 prompt 环境下的方言训练提出了疑问。
   - 社区在 LLM 处理方言方面的经验强化了先进上下文理解的概念，激发了对**训练机制**的好奇。
- **Devcontainer Dilemma: Pydantic Ponders**: AI 工程师在设置 **Cohere toolkit 仓库**时遇到了瓶颈，**Pydantic 验证错误**导致设置中止，暴露出 `Settings` 类中缺失字段（如 **auth.enabled_auth**）的问题。
   - 团队迅速做出回应，承诺即将修复，展示了在工具包维护和可用性方面的敏捷性与承诺。
- **"Code and Convene": AI Hackathon Series**: 社区成员热烈讨论参加在 Google 举办的 **AI Hackathon Series Tour**，这是一场为期 **3 天** 的 AI 创新与竞赛。
   - 该巡回赛旨在展示 AI 进展和创业项目，最终以 **PAI Palooza** 告终，这是一个展示新兴 AI 初创公司和项目的盛会。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Pydantic Puzzles in LangChain Programming**: 由于 Pydantic 版本不匹配导致 **ValidationError**，在配合 **LangChain** 工作时引起了类型不一致的困惑。
   - 输入不匹配和验证导致执行失败，突显了冲突，强调了 **api_version** 协调的必要性。
- **API Access Angst for LangSmith Users**: 一位用户在尝试使用 **LangSmith** 部署 LLM 时遇到了 `403 Forbidden` 错误，提示可能存在 API key 配置错误。
   - 社区讨论围绕 key 的正确设置展开，并寻求通过各种 **LangChain** 渠道获得帮助。
- **Streaming Solutions for FastAPI Fabulousness**: 一位用户提出了一种在 LangChain 应用中使用 **FastAPI** 进行异步流式传输的模式，提倡使用 Redis 进行平滑的消息代理。
   - 这将保持当前的同步操作，同时赋能 LangChain Agent **实时**分享结果。
- **Jump-Start Resources for LangChain Learners**: 讨论深入探讨了掌握 LangChain 的可用资源，强调了有效学习的替代方案和仓库。
   - 成员们交换了 **GitHub** 示例和各种 API 文档，以便更好地应对常见的部署和集成难题。
- **LangGraph's Blueprints Unveiled**: 分享了一种创新的 LangGraph 设计模式，旨在用户友好地集成到 **web-chats** 和即时通讯机器人等应用中，并提供了一个展示集成过程的 GitHub 示例。
   - 此外，还发出了测试 **Rubik's AI** 新功能的 Beta 邀请，通过特别促销活动提供包括 **GPT-4o** 和 **Claude 3 Opus** 在内的顶级模型。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **数字排毒饮食：Moye 方法**：Moye Launcher 的极简设计通过刻意降低应用的可访问性来促进数字健康，倡导向减少屏幕时间的行为转变。
   - 开发者针对导致过度使用的三个因素（如自动点击和缺乏问责制），旨在通过设计和用户反馈**培养专注应用使用的习惯**。
- **闪耀的人格：Big-agi 的大动作**：Big-agi 的“人格创建器”允许用户根据 YouTube 输入**快速生成角色档案**，而 BEAM 功能则合并了多个模型的输出，增加了响应的多样性。
   - 尽管如此，Big-agi 仍因缺失服务器保存和同步功能而感到局促，这阻碍了原本**流畅的模型交互体验**。
- **Msty 融合记忆与网页掌控力**：Msty 与 Obsidian 的集成以及网页连接功能因其易用性而获得用户好评，但因其健忘的**参数持久性**而面临批评。
   - 尽管 Msty 仍需打磨，但一些用户因其**精美的界面交互能力**而考虑转向使用它。
- **Llama 405B 走 FP16 钢丝**：OpenRouter 缺少 Llama 405B 的 FP16 路径，而 Meta 推荐的 FP8 量化被证明**更有效率**。
   - 虽然 SambaNova Systems 提供类似服务，但它们受限于最大 4k 的上下文限制以及成本高昂的 bf16 托管。
- **OpenRouter 的 Beta 版保证 API 门户**：OpenRouter 预告了 API 集成 Beta 版，欢迎通过支持邮件进行速率限制（rate limit）微调，并将 OpenAI 和 Claude API 串联到用户项目中。
   - 虽然其网站有时会遇到区域性问题，但 [OpenRouter 状态页面](https://status.openrouter.ai/) 就像一座灯塔，引导用户度过运行风暴。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 陷入慢车道**：对于 **Ben Steinher 延迟回复**的担忧正在增加，他错过了 7 月中旬的回复截止日期。
   - 尽管有所延迟，社区仍称赞了一个关于 Groq 配置文件贡献的新 PR，认为这是支持 Open Interpreter 的有效方式，并重点介绍了 **MikeBirdTech** 提交的 [GitHub PR](https://github.com/OpenInterpreter/open-interpreter/pull/1376)。
- **技术人员关注无障碍讨论**：一场**无障碍圆桌会议**定于 8 月 22 日举行，旨在激发讨论和参与，并公开邀请社区分享见解。
   - 在解决了最初的**时区混乱**后，人们对即将举行的 House Party 活动充满期待，参与者可前往 [活动链接](https://discord.gg/zMwXfHwz?event=1267524800163610815)。
- **模型选择令人困惑**：关于使用 '01 --local' 时是否需要 OpenAI API key 以及正确的模型字符串产生了讨论，这表明需要更清晰的指南。
   - 探究性的帖子仍在继续，询问 **OpenInterpreter** 是否可以保存和调度工作流，社区尚未给出答案。
- **iKKO 耳机放大 AI 可能性**：关于在 **iKKO ActiveBuds** 上集成 OpenInterpreter 的讨论正在升温，将高分辨率音频与 AI 结合，详见 [iKKO 官网](https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds)。
   - 01 的发货更新在社区内引发了紧迫感，随着 8 月的流逝，更新信息的呼声仍未得到回应。
- **带有视觉的耳机：摄像头讨论**：出现了一个为耳机配备摄像头的创新想法，通过在与 LLM 对话期间捕捉视觉上下文来增强交互。
   - 社区成员思考了这一集成，考虑通过点击功能激活摄像头，以增强 HCI 体验。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 线程支持缺失**：在关于 **Mojo** 能力的讨论中，一位成员澄清说 **Mojo 目前并不直接向用户开放线程支持**。
   - 有人提到，在编译环境中，利用 **fork()** 是实现多线程的一种变通方法。
- **MAX 与 Mojo 的打包公告**：官方透露了即将到来的 **MAX 和 Mojo 打包变更**，从 `modular` CLI 的 0.9 版本开始，下载 MAX 和 Mojo 将不再需要身份验证。
   - **Mojo 将与 MAX nightly 构建版本合并**，[公告](https://docs.modular.com/max/faq#why-bundle-mojo-with-max)建议转向新的 `magic` CLI 以实现无缝的 Conda 集成。
- **层级图表引发困惑**：成员们对一张层级图表表示困惑，争论其表示是否准确，并批评其未能反映预期的**“抽象层级”**。
   - 一些人主张用火焰表情符号简化视觉效果，表示希望有一个清晰且有效的沟通工具。
- **CrazyString 中的 Unicode 支持**：[CrazyString gist](https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae) 已更新，引入了基于 Unicode 的索引，并实现了**完全的 UTF-8 兼容性**。
   - 对话涉及了 Mojo 字符串的**小字符串优化（small string optimization）**以及更新后带来的易用性提升。
- **M1 Max 上的 Max 安装难题**：一位成员在 **Mac M1 Max 设备**上尝试安装 max 时遇到挑战，社区成员介入并提供了潜在的修复方案。
   - 一份共享资源建议使用[特定的 Python 安装变通方案](https://modul.ar/fix-python)来解决安装问题。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl 引入自动停止算法**：针对关于在**损失停滞（loss plateaus）**或**验证损失激增**时停止训练的询问，Axolotl 引入了**早停（early stopping）功能**。
   - 社区成员就手动终止运行同时保存当前 **LoRA adapter** 状态的能力进行了简短交流。
- **SharedGPT 的掩码学习飞跃**：一位成员为 SharedGPT 的每一轮对话提出了一个**“输出掩码（output mask）”字段**，旨在通过选择性输出掩码进行针对性训练。
   - 这一创新引发了关于其通过处理输出错误来精炼学习潜力的讨论。
- **聊天模板需要更清晰**：由于难以理解新的**聊天模板（chat templates）**，成员们呼吁提供更好的**文档**以帮助理解和自定义。
   - 一位成员主动分享了关于该主题的个人笔记，并建议由社区驱动更新官方文档。
- **Pad Token 重复问题**：训练问题讨论中提到了频繁出现的 `<pad>` **token 重复**现象，暗示了采样方法的低效。
   - 对话贡献了一个技巧：确保 pad token 在标签中被遮蔽（cloaked），以防止循环冗余。
- **Gemma2 使用 Eager 优于 Flash**：出现了一个针对 Gemma2 模型训练的推荐技巧，建议使用 `eager` 而非 `flash_attention_2` 以巩固稳定性和性能。
   - 提供了实践指导和[代码示例](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e)，演示如何在 `AutoModelForCausalLM` 中设置 `eager` attention。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **围绕 DSPy 和符号学习（Symbolic Learning）展开热烈讨论**：成员们对将 **DSPy** 与符号学习器集成充满期待，推测其具有突破性的潜力。
   - 参与者预期这种结合将为 AI 能力带来实质性的进步，表现出乐观态度。
- **自适应 Agent 成为焦点**：**Microsoft Research 博客**将自适应 AI Agent 推向台前，展示了一篇具有前景的职场应用[文章](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/)。
   - 见解指出游戏行业是 AI 进步的催化剂，现在正体现在 **ChatGPT** 和 **Microsoft Copilots** 等工具中。
- **Agent Zero 登场：用户测试版 AI 的尝试**：**Agent Zero** 作为首个经过用户测试的生产版本亮相，展示了其 AI 实力。
   - 反馈暗示 AI 正在向职业环境中占据更多样化角色的方向转变。
- **LLM 通过 Meta-Rewarding 实现自我提升**：一篇 [arXiv 论文](https://arxiv.org/abs/2407.19594)揭示了一种新的 **Meta-Rewarding** 技术，增强了 LLM 的自我判断能力，从而提升其性能。
   - 据报道，在 AlpacaEval 2 上胜率显著提高，表明 **Llama-3-8B-Instruct** 等模型也从中受益。
- **MindSearch 论文探讨基于 LLM 的多 Agent 框架**：发表在 [arXiv](https://arxiv.org/abs/2407.20183) 上的一篇论文介绍了 **MindSearch**，它使用 LLM 驱动的 Agent 在网络搜索中模拟人类认知过程。
   - 该研究解决了信息检索挑战，旨在改进现代搜索辅助模型。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVIDIA 获得纳税人资金**：一条消息对 **NVIDIA** 获得公共资金表示热议，详细说明了纳税人投资的价值。
   - 这一话题引发了关于投资优先级及其对技术发展影响的对话。
- **George Hotz 强调 Discord 规范**：**George Hotz** 提醒了服务器规则，将注意力集中在 **tinygrad 开发**上。
   - Hotz 的提醒是号召社区保持专业且切题的对话。
- **Argmax 拖慢了 GPT-2 的速度**：对 **GPT-2 性能**的深入研究发现，embedding 结合 `argmax` 显著限制了执行速度，如 [Issue #1612](https://github.com/tinygrad/tinygrad/issues/1612) 所示。
   - 效率低下追溯到 **O(n^2)** 复杂度问题，引发了关于更高效算法方案的讨论。
- **Embedding 悬赏：Qazalin 的任务**：出现了关于提升 tinygrad 中 **embeddings** 性能的悬赏讨论，专门针对名为 **Qazalin** 的用户。
   - 悬赏引起了轰动，并激励其他贡献者在 tinygrad 中寻找不同的优化机会。
- **Cumsum 难题**：[Issue #2433](https://github.com/tinygrad/tinygrad/issues/2433) 解决了 `cumsum` 函数 O(n) 复杂度的挑战，激发了开发者的创新思维。
   - George Hotz 召集人马，提倡通过实际实验来发现可能的优化策略。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **多语言 ChatGPT 的语音绝技**：一位成员展示了 [ChatGPT Advanced Voice Mode](https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19)，它能熟练地用**乌尔都语**朗诵诗歌，并用包括**希伯来语、挪威语和格鲁吉亚语**在内的多种语言讲故事。
   - 此次展示还包括了**摩洛哥阿拉伯语 (Moroccan Darija)、阿姆哈拉语、匈牙利语、克林贡语**等冷门方言的叙述，令工程社区惊叹不已。
- **Black Forest Labs 的惊艳亮相**：[Black Forest Labs](https://x.com/robrombach/status/1819012132064669739) 的发布引发了热烈反响，其使命专注于媒体领域的创新生成模型。
   - 该计划以 **FLUX.1** 拉开序幕，这款模型承诺将提升视觉生成的创造力、效率和多样性。
- **FLUX.1 模型首秀令人印象深刻**：社区将目光转向了 **FLUX.1**，这款新模型在 [Hugging Face](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) 上的首次亮相赢得了广泛赞誉。
   - 讨论集中在这一模型如何潜在地改变生成式学习的格局，其特性被形容为“令人耳目一新”且“非常出色”。
- **创新的激活函数尝试**：AI 爱好者们深入研究了在复数值激活（complex-valued activations）上使用各种**归一化和激活函数**的实验，并称这些练习“挺有意思！”。
   - 这种实践探索促进了心得分享，并探讨了在复杂领域的潜在应用。
- **被过度炒作的正则化难题**：一位用户引用 [一篇 Medium 文章](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9) 指出，像**数据增强 (data augmentation) 和 Dropout** 这样广泛使用的方法在显著抑制过拟合方面表现乏力。
   - 通过探究各种**正则化技术 (regularization techniques)** 的有效性，社区开始思考超越传统技巧的方法，以推进机器学习模型的发展。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **使用 Top_p 达到性能巅峰**：一位成员发现将 **top_p=50** 设置为符合其性能标准的参数，并获得了**显著**的效果。
   - 他们将 **0.8 在线模型**与自己的模型进行了对比，注意到在线变体具有**更优**的结果。
- **使用 Generate Recipe 享受调试乐趣**：澄清了 **generate recipe** 主要是为调试目的而设计的，旨在实现对模型的**准确刻画**。
   - 与基准测试的任何差异都应促使提交 Issue，评估确认了该 Recipe 的**有效性**。
- **FSDP2 的新特性融合**：一位成员分享道，**FSDP2** 现在同时支持 **NF4 tensor** 的量化和 QAT，增强了其**通用性**。
   - 虽然 QAT Recipe 看起来是**兼容**的，但使用 FSDP2 进行编译可能会遇到挑战，这标志着一个潜在的**优化**领域。
- **精准合并 PR**：一个即将合并的 PR 被标记为依赖于前一个 PR，其中 **PR #1234** 正在审查中，从而为**序列化改进**铺平了道路。
   - 这预示着微调数据集的增强，重点关注 **grammar 和 samsum**，推进了 Torchtune 的**系统化**演进。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Data Phoenix 举办 AI 网络研讨会**：**Data Phoenix** 团队宣布了一场名为“利用 LLM 和生成式 AI 增强推荐系统”的网络研讨会，主讲人为 [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/)，定于 PDT 时间 8 月 8 日上午 10 点举行。
   - 此次研讨会旨在揭示 **LLM** 和 **Generative AI** 如何变革个性化引擎，并已开放 [研讨会注册](https://lu.ma/6i6dtbhf)。
- **dlt 通过工作坊提升 ELT 技能**：一场关于 **使用 dlt 进行 ELT** 的 4 小时工作坊将为数据爱好者传授构建稳健 ELT 流水线的知识，完成后可获得“dltHub ELT Engineer”认证。
   - 该课程定于 2024 年 8 月 15 日 16:00 GMT+2 在线举行，从 dlt 基础知识开始，[可以在此处注册](https://dlthub.com/events)。
- **会议展示了 NLP 和 GenAI 的主导地位**：两场机器学习会议都重点强调了 **NLP** 和 **GenAI**，掩盖了关于 **高斯过程 (Gaussian Processes)** 和 **孤立森林 (Isolation Forest)** 等模型的演讲。
   - 这一趋势凸显了社区向 NLP 和 GenAI 技术的强烈倾斜，使得一些小众模型的讨论显得黯然失色。
- **社区审视 GenAI 的 ROI**：一场热烈的辩论质疑了 **GenAI 的 ROI（投资回报率）** 是否能达到业内某些人设定的高度预期。
   - 对话指出了预期与现实之间的差距，强调需要对回报保持理性的预期。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LangSmith 额度难题**：**Digitalbeacon** 报告了在添加付款方式后无法访问 LangSmith 额度的问题，他使用的电子邮件地址与其组织 ID **93216a1e-a4cb-4b39-8790-3ed9f7b7fa95** 不同。
   - **Danbecker** 建议联系支持部门解决额度相关问题，暗示需要直接通过客服解决。
- **LangSmith 付款方式混乱**：**Digitalbeacon** 询问了在更新付款方式后，即使及时提交了表格，LangSmith 余额仍为零的问题。
   - 这种情况表明可能存在系统故障或用户操作失误，需要进一步调查或支持干预。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1268649413648318474)** (1 条消息): 

> - `Neural network simulation`
> - `Video clustering`
> - `Synthetic dataset`
> - `Knowledge distillation`
> - `Gradio demo` 

- **在线模拟神经网络**：一位成员分享了一个现在可以在线使用的 [神经网络模拟](https://starsnatched.github.io/)。
   - *在一个交互式网站中探索不同的神经网络配置及其行为*。
- **掌握视频聚类技术**：一段新的 [YouTube 视频](https://www.youtube.com/watch?v=8f4oRcSnfbI) 解释了如何使用局部二值模式 (LBP) 和方向梯度直方图 (HOG) 等图像描述符进行视频聚类。
   - *学习聚类以实现更好的视频数据组织和处理*。
- **探索海量合成数据集**：社区成员发布了一个巨大的 [合成数据集](https://huggingface.co/datasets/tabularisai/oak)。
   - *非常适合进行表格数据模型的实验*。
- **前沿知识蒸馏技术**：一篇富有见解的文章讨论了最新的 [知识蒸馏趋势](https://www.lightly.ai/post/knowledge-distillation-trends) 及其影响。
   - *紧跟高效模型训练方法的最新动态*。
- **金融与医疗模型发布**：推出了用于 [金融和医疗](https://x.com/samjulien/status/1818652901130354724) 目的的新模型 Palmyra-Med-70b 和 Palmyra-Fin-70b。
   - Palmyra-Med-70b 在医疗任务中表现出色，**MMLU 性能约为 86%**，而 Palmyra-Fin-70b 是首个以 **73%** 的成绩通过 CFA Level III 考试的模型。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=8f4oRcSnfbI)">🎥 使用图像描述符掌握视频聚类：LBP &amp; HOG 详解！🌟</a>：🔍 在这份详细指南中发现视频聚类的力量！了解如何使用局部二值模式 (LBP) 和方向梯度直方图 (HOG) 等图像描述符...</li><li><a href="https://www.youtube.com/live/dcCn4nuKpBs?feature=share)">Unity ML-Agents | 从零开始的实时 Agent 训练</a>：一个关于 ml agents 和 cuda 的快速小实验</li><li><a href="https://x.com/samjulien/status/1818652901130354724">来自 Sam Julien (@samjulien) 的推文</a>：🔥 @Get_Writer 刚刚发布了 Palmyra-Med-70b 和 Palmyra-Fin-70b！Palmyra-Med-70b 🔢 提供 8k 和 32k 版本 🚀 MMLU 性能 ~86%，优于顶级模型 👨‍⚕️ 用于诊断、制定治疗方案...
</li>
</ul>

</div>

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1268282601886580807)** (852 条消息 🔥🔥🔥): 

> - `GPTs agents`
> - `Keras 介绍`
> - `OpenAI 侧边栏变更`
> - `用于 Minecraft 的 Autoencoders`
> - `结合量化的模型 Fine-tuning`

- **GPTs Agents 被误解**：成员们讨论了 GPTs Agents 在初始训练后不会从额外信息中学习。
   - 澄清说明：上传的文件被保存为“知识”文件以供参考，但不会修改基础知识。
- **介绍用于 Deep Learning 的 Keras**：成员们解释了 Keras 作为一个支持 JAX、TensorFlow 和 PyTorch 的多后端 Deep Learning 框架。
   - Keras 因加速模型开发并提供具有易于调试运行时的 SOTA 性能而受到赞誉。
- **OpenAI 平台侧边栏更改**：成员们讨论了 platform.openai.com 侧边栏中两个图标消失的问题。
   - 有人注意到 threads 和 messages 的图标从侧边栏消失了，引发了进一步讨论。
- **用于 Minecraft 视频生成的 Autoencoders**：成员们致力于训练 Autoencoders 以压缩 Minecraft 图像和视频，旨在生成 Minecraft 视频序列。
- **使用 Quantization 微调模型的挑战**：成员们解决了与使用 Quantization 微调 Llama 3-8b 模型以有效管理 GPU 显存相关的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:11434",">未找到标题</a>: 未找到描述</li><li><a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">Black Forest Labs 发布 Flux：文本生成图像模型的下一次飞跃</a>: Flux 是迄今为止最大的 SOTA 开源文本生成图像模型，由 Stable Diffusion 的原班人马 Black Forest Labs 开发，现已在 fal 上线。Flux 突破了创意边界...</li><li><a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>: 修改了 bitnet 用于微调，最终得到了一个 74mb 的文件。仅在 1 个 CPU 核心上就能以每秒 198 个 tokens 的速度流畅运行。简直是魔法。稍后将通过 @skunkworks_ai 开源，基础版本见：https://huggi...</li><li><a href="https://imgur.com/dd3TB7g">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://colab.research.google.com/drive/15md1YRAvT8Hg6fnkEA8BnuNkg-mAajWQ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://llm.extractum.io/list/?mtr=nroggendorff">维护者 «nroggendorff»</a>: 一个精心挑选的大语言模型和小语言模型列表（开源 LLMs 和 SLMs）。由 «nroggendorff» 维护，支持动态排序和过滤。</li><li><a href="https://huggingface.co/glides">glides (Glide)</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/accelerate/usage_guides/big_modeling">处理用于推理的大模型</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/TencentARC/PhotoMaker">PhotoMaker - TencentARC 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/tuh-buh-guh-cuh-what-gif-9750912507529527670">Tuh Buh GIF - Tuh Buh Guh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://pypi.org/project/keras/">keras</a>: 多后端 Keras。</li><li><a href="https://lu.ma/6i6dtbhf?utm_source=DiscordEvent8">利用 LLMs 和生成式 AI 增强推荐系统 · Luma</a>: Data Phoenix 团队邀请您参加我们即将举行的网络研讨会，时间为 PDT 8 月 8 日上午 10 点。主题：利用 LLMs 增强推荐系统…</li><li><a href="https://www.tensorflow.org/guide/keras">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/BioMistral/BioMistral-7B">BioMistral/BioMistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">在你的数据集上微调 Mistral</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/facebook/llm-compiler-667c5b05557fe99a9edd25cb">LLM Compiler - facebook 收藏集</a>: 未找到描述</li><li><a href="https://youtu.be/0me3guauqOU">JPEG 难以置信的有效性：一种信号处理方法</a>: 访问 https://brilliant.org/Reducible/ 免费开始学习 STEM，前 200 名用户将获得年度高级订阅 20% 的折扣...</li><li><a href="https://youtu.be/4VAkrUNLKSo">计算机生成人脸</a>: 5:51 跳过至结果。在线尝试：http://codeparade.net/faces/ 下载应用 (Windows 64-bit): https://github.com/HackerPoet/FaceEditor/raw/master/Fac...</li><li><a href="https://github.com/noamgat/lm-format-enforcer/blob/main/README.md">noamgat/lm-format-enforcer 的 main 分支 README.md</a>: 强制执行语言模型的输出格式（JSON Schema、Regex 等）- noamgat/lm-format-enforcer</li><li><a href="https://youtu.be/NTlXEJjfsQU">用 AI 创建我自己的定制名人</a>: 查看 Brilliant.org 获取有趣的在线 STEM 课程！前 200 名注册用户可享受年度高级订阅费用 20% 的折扣：https://brilliant...</li><li><a href="https://youtu.be/Dt2WYkqZfbs">为什么图像是可压缩的：图像空间的浩瀚</a>: 我们探讨了为什么图像是可压缩的，这与所有可能图像的（大于）天文数字般的空间有关。这是我的最爱之一。关注...</li><li><a href="https://open.spotify.com/track/6y5HLopYu7Uu0hYwVBj4T6">palm of my hands</a>: 歌曲 · John Summit, venbee · 2024</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: LLMs 的结构化输出</a>: LLMs 的结构化输出。通过在 GitHub 上创建账号为 jxnl/instructor 的开发做出贡献。</li><li><a href="https://esolangs.org/wiki/Chicken">Chicken - Esolang</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/pull/9043">sayakpaul 提交的 Flux pipeline · Pull Request #9043 · huggingface/diffusers</a>: 我们正在努力将 diffusers 权重上传到相应的 FLUX 仓库。很快就会完成。
</li>
</ul>

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1268483912267599872)** (4 条消息): 

> - `finegrain Object Eraser 模型`
> - `AI bots 的演进`
> - `Knowledge distillation` 


- **Finegrain 发布 Object Eraser 模型**：一位成员分享了在 [Hugging Face space](https://huggingface.co/spaces/finegrain/finegrain-object-eraser) 上线的新 **Object Eraser 模型**的消息，展示了该模型的能力。
   - 该模型由 **@finegrain_ai** 开发，旨在向公众展示新应用，供所有人尝试。
- **Medium 上关于 AI bots 演进的文章**：一位成员在 Medium 上发布了一篇关于 **AI bots 演进**的文章，详细介绍了 **LLMs** 和 **RAG** pipelines 等各种 AI 工具。[阅读全文](https://medium.com/@qdrddr/evolution-of-the-ai-bots-harnessing-the-power-of-agents-rag-and-llm-models-4cd4927b84f8)。
   - 该文章专为初学者设计，深入探讨了 **2024** 年使用的顶层模式、pipelines 和架构设计。
- **理解 Knowledge Distillation**：一位成员发现 Knowledge distillation 是一个有趣的话题，并分享了来自 IBM 关于 [Knowledge Distillation](https://www.ibm.com/topics/knowledge-distillation) 的详细页面。
   - 文章解释说，**knowledge distillation** 将学习成果从大型预训练的 'teacher model' 转移到较小的 'student model'，以实现模型压缩和知识转移。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pchapuis/status/1818632367138885826">来自 Pierre Chapuis (@pchapuis) 的推文</a>：创建了一个 @huggingface space 来演示我们在 @finegrain_ai 训练的模型之一：Object Eraser。很高兴大家终于可以公开尝试它了。:) https://huggingface.co/spaces/finegrai...</li><li><a href="https://www.ibm.com/topics/knowledge-distillation">什么是 Knowledge distillation？ | IBM </a>：Knowledge distillation 是一种机器学习技术，用于将大型预训练 “teacher model” 的学习成果转移到较小的 “student model”。</li><li><a href="https://medium.com/@qdrddr/evolution-of-the-ai-bots-harnessing-the-power-of-agents-rag-and-llm-models-4cd4927b84f8">AI Bots 的演进：利用 Agents、RAG 和 LLM 模型的力量</a>：构建关于 AI bot 开发工具的知识，并提供方法、架构和设计的顶层概述。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1268285446937514107)** (16 messages🔥): 

> - `model release heatmap` (模型发布热力图)
> - `grounding-sam2-demo`
> - `TinyML bird detection project` (TinyML 鸟类检测项目)
> - `Infinite Sands project` (Infinite Sands 项目)
> - `2D parallelism in deep learning` (深度学习中的 2D 并行)


- **模型发布热力图 Space 引起关注**：一名成员为[顶级 AI 实验室的模型发布热力图创建了一个 Space](https://huggingface.co/spaces/cfahlgren1/model-release-heatmap)。
   - 其他人表示有兴趣将此类热力图集成到未来的 Hugging Face 个人资料页面中，以提高可见性。
- **Grounding-Sam2 演示展示了配对模型**：一名成员分享了一个 [GitHub 项目](https://github.com/CoffeeVampir3/grounding-sam2-demo)，演示了用于 grounding dino 和 segment anything v2 模型的 Gradio 界面。
   - 该演示以简单且互动的格式展示了这些模型的升级用法。
- **TinyML 通过 Seeed 和 Blues 检测鸟类**：[Hackster](https://www.hackster.io/timo614/bird-detection-with-tinyml-and-a-blues-notecard/) 上的一个项目报告了使用 TinyML 硬件和 Blues Notecard 识别鸟类物种的情况。
   - 该设置涉及 Seeed 的 Grove Vision AI Module V2，并压缩了 EfficientNetLite 以实现高效的鸟类检测。
- **Infinite Sands 利用 AI 赋予沙盒生命**：[Infinite Sands](https://www.hackster.io/sand-command/infinite-sands-df675a) 使用生成式 AI 根据沙盒形状创作故事。
   - 该项目应用了 ControlNet depth 和 Whisper 进行命令处理，使其成为一个充满趣味和互动性的探索。
- **AI + i 播客发布，专注于 AI 模型**：一个新的播客系列 [Ai + i](https://youtube.com/@aiplusi) 已经发布，旨在讨论领先的基础模型和开源模型。
   - 主持人正在向社区征集未来播客节目的主题建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/cfahlgren1/model-release-heatmap">Model Release Heatmap - cfahlgren1 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://malaysia-ai.org/2d-parallelism-ray">Malaysia-AI 博客：使用 Ray PyTorch 的 2D 并行</a>：Malaysia-AI 博客：使用 Ray PyTorch 的 2D 并行</li><li><a href="https://huggingface.co/spaces/cfahlgren1/model-release-heatmap/discussions?status=open&type=discussion">cfahlgren1/model-release-heatmap · 讨论</a>：未找到描述</li><li><a href="https://huggingface.co/tasksource/deberta-base-long-nli">tasksource/deberta-base-long-nli · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/CoffeeVampir3/grounding-sam2-demo/blob/main/interface.py">grounding-sam2-demo/interface.py (main 分支) · CoffeeVampir3/grounding-sam2-demo</a>：一个用于共同使用 grounding dino 和 segment anything v2 模型的简单演示 - CoffeeVampir3/grounding-sam2-demo</li><li><a href="https://www.hackster.io/timo614/bird-detection-with-tinyml-and-a-blues-notecard-b8b705">使用 TinyML 和 Blues Notecard 进行鸟类检测</a>：我构建了一个项目，使用机器学习 (TinyML) 识别喂鸟器处的鸟类，并使用 Blues Notecard 将数据传输到云端。作者：Timothy Lovett 和 Kerin Lovett。</li><li><a href="https://www.hackster.io/sand-command/infinite-sands-df675a">Infinite Sands</a>：由 ROCM 驱动的沙盒，在 standard diffusion 和 controlnet 的帮助下塑造你的现实。作者：Timothy Lovett 和 Kerin Lovett。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1268408183026618368)** (8 messages🔥): 

> - `Deep Learning Study Group` (深度学习学习小组)
> - `LLM Model Suggestions` (LLM 模型建议)
> - `New Learners Collaboration` (新学习者协作)


- **深度学习爱好者集结**：一位新成员表示有兴趣组建一个由积极进取的个人组成的团体，共同学习 **deep learning 和 machine learning**。
- **用于 PDF 表格和复选框检测的 LLM 模型**：一名成员请求推荐能够从 PDF 输入中执行表格和复选框检测及提取的 **LLM 模型**。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/)** (1 messages): 

sayakpaul: 将在稍后合并 https://github.com/huggingface/diffusers/pull/9043
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1268576936569864192)** (2 messages): 

> - `为 Solr 训练 LLM`
> - `针对失语症患者的 AI 系统` 


- **训练 LLM 以解析 Solr 的搜索查询**：一位成员就如何训练 **Large Language Model (LLM)** 接收搜索查询并输出包含**产品分面（facets）和类别**的 JSON 以用于 [Apache Solr](https://solr.apache.org/) 寻求建议。
   - 他们提到没有指令数据集（instruction dataset），并寻求如何开展该任务的指导。
- **构建用于与失语症患者沟通的 AI**：一位成员打算构建一个结合了**微表情识别、语音识别和图像识别**的 AI 系统，以帮助促进与失语症患者的沟通。
   - 他们请求帮助，因为他们不知道如何开始这个项目，并表示任何建议都将非常有帮助。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1268347706171068558)** (2 messages): 

> - `惊人的结果`
> - `钓鱼指控` 


- **Welltoobado 赞扬结果**：一位成员对某事表示满意，指出 *“是的，非常惊人的结果，做得好！”*。
- **Pseudoterminalx 质疑是否在钓鱼**：另一位成员不确定其诚意，回复道：*“很难分清你是不是在钓鱼了，哈哈”*。


  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=DLb7Lrzw8wo
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1268628172216340562)** (9 messages🔥): 

> - `多模态架构中新的 SOTA 效率提升` 


- **多模态架构中新的 SOTA 效率提升**：推出 **Chameleon** 的作者们在一种[新的多模态架构](https://arxiv.org/pdf/2407.21770)中实现了显著的效率提升，该架构结合了 Mixture of Experts (MoE) 和模态特定的专家路由技术。
   - 根据 [Victoria Lin](https://x.com/VictoriaLinML/status/1819037439681565178) 的说法，效率提升在*文本训练中约为 3 倍*，在*图像训练中约为 5 倍*，MoMa 1.4B 的表现显著优于其稠密对应版本（dense counterpart）和其他 MoE 模型。
- **关于多模态架构中新 SOTA 效率提升的讨论**：成员们对新架构表示兴奋，注意到其显著的 FLOPs 节省和性能提升。
   - 特别提到了**图像训练**方面的收益，强调了新架构令人印象深刻的 **5.2x** 效率提升。



**提到的链接**：<a href="https://x.com/VictoriaLinML/status/1819037439681565178">来自 Victoria X Lin (@VictoriaLinML) 的推文</a>：4/n 在 1T token 的训练预算下，MoMa 1.4B（4 个文本专家 + 4 个图像专家）与稠密对应版本相比，实现了 3.7 倍的 FLOPs 节省（文本：2.6 倍，图像：5.2 倍）（在预训练阶段测量...）

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1268282496374542463)** (441 messages🔥🔥🔥): 

> - `Heptagon Riddle`
> - `GPT Benchmarks vs Human Heuristics`
> - `Speculative Decoding Mechanics`
> - `Dynamic Memory Systems`
> - `Bitnet for Finetuning` 


- **Heptagon 谜题已解决**：**一个关于《平面国》(Flatland) 居民的谜题**涉及确定正多边形的类型。经过讨论，**heptagon**（七边形）是正确答案。
   - *一位用户*指出，某些模型偶尔会得到幸运答案，但总体而言，**LLM 在符号逻辑谜题上表现挣扎**。
- **Speculative Decoding 见解**：参与者讨论了 **Speculative Decoding** 技术，解释了使用较小的草稿模型（draft models）来加速解码并不总是无损的。
   - 虽然一些初步说法称如果操作不当，输出分布可能会发生偏离，但其他人澄清说，**拒绝采样 (rejection sampling)** 通过对齐草稿模型和基座模型来确保无损输出。
- **动态记忆系统应用**：**动态人格记忆 (Dynamic persona memories)** 被讨论为当前 ragdata 集中的一个空白，参与者提出了合作机会。
   - 参与者比较了**并行化 Token 生成**的技术，并指出了 LLM 在动态系统中进行**准确上下文处理**的问题。
- **Bitnet 的 Finetuning 带来速度提升**：一篇关于 **Bitnet Finetuning 方法**的 Reddit 帖子因其惊人的速度而受到关注，在仅一个 CPU 核心上运行速度就达到了 **每秒 198 个 Token**。
   - 实验者使用 Bitnet 实现了 **74MB 的文件大小**，并声称其运行效率极高，引发了人们对其**未来项目潜力**的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">来自 nisten (@nisten) 的推文</a>：修改了 Bitnet 用于 Finetuning，最终得到了一个 74MB 的文件。在仅使用一个 CPU 核心的情况下，它的运行速度达到了每秒 198 个 Token。简直是巫术。稍后将通过 @skunkworks_ai 开源，基座地址：https://huggi...</li><li><a href="https://x.com/ryunuck/status/1818709409239121975?s=46">来自 ryunuck (p≈np) (@ryunuck) 的推文</a>：Ilya 所看到的。CRISPR-Q 在 Sonnet 3.5 上运行，并使模型能够通过其自身 self-memeplex 的定向操作重写上下文窗口。这种难以理解的异类生成启发式方法...</li><li><a href="https://huggingface.co/openai-community/gpt2-xl">openai-community/gpt2-xl · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=hm7VEgxhOvk">Speculative Decoding 详解</a>：一键模板仓库（免费）：https://github.com/TrelisResearch/one-click-llms 高级推理仓库（付费终身会员）：https://trelis.com/enter...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/holo-q/OpenQ/">GitHub - holo-q/OpenQ: Q* 的开源实现，在上下文中作为注意力机制的零样本重编程实现。（合成数据）</a>：Q* 的开源实现，在上下文中作为注意力机制的零样本重编程实现。（合成数据） - holo-q/OpenQ</li><li><a href="https://github.com/carsonpo/octoquadmul">GitHub - carsonpo/octoquadmul</a>：通过创建账户为 carsonpo/octoquadmul 的开发做出贡献。</li><li><a href="https://github.com/carsonpo/octomul">GitHub - carsonpo/octomul: 相当快（与 cublas 相比）且相对简单的 int8 tensor core gemm</a>：相当快（与 cublas 相比）且相对简单的 int8 tensor core gemm - carsonpo/octomul
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1268591738939768913)** (2 messages): 

> - `LangChain usage`
> - `Mixtral API retrieval`
> - `OpenAI API format` 


- **在 OpenAI 格式下结合 Mixtral API 使用 LangChain**：一位成员讨论了一段使用 **LangChain** 的代码片段，其中使用了 **mixtral_api_base** 等环境变量从 OpenAI API 获取 Mixtral LLM。
   - 关于在没有 **LangChain** 的情况下这种方法是否有意义存在争论，因为 LangChain 使用的是 OpenAI API 格式。
- **关于 LangChain 必要性的辩论**：随后引发了另一场讨论，即在使用 OpenAI API 与 Mixtral LLM 交互时，是否有必要使用 **LangChain**。
   - 成员们对此类操作对 LangChain 的依赖程度表达了不同的看法。


  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1268311059614269553)** (3 条消息): 

> - `协助项目设置`
> - `项目的成本考量` 


- **项目设置协助**：一名成员询问他们可以做些什么来帮助项目启动，以及是否需要很高成本。
   - 另一名成员确认该项目不产生任何费用，并指示他们按照一个待处理 PR 中提到的步骤进行操作。
- **免费项目计划**：一位参与者提到该项目不会产生任何成本。
   - 下一步工作包括在新的 PR 发布后，按照提供的说明进行操作。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1268282183496237217)** (205 条消息🔥🔥): 

> - `多 GPU 支持`
> - `Unsloth 微调`
> - `Qwen 模型合并`
> - `AI 性能`
> - `Bitnet 代码 Hack` 


- **多 GPU 训练可用但仍需改进**：用户确认在修复后多 GPU 训练可以工作，但指出早期的安装问题需要创建新环境并对各种设置进行故障排除。
   - 一个例子提到：*“将其安装到 llamafacs 环境中第一次就成功了，”* 而另一位用户提到需要手动升级 transformers。
- **Unsloth Crypto Runner 说明**：对 **Unsloth Crypto Runner** 进行了澄清，指出它涉及客户端与许可证服务器之间的 **AES/PKI 基础加密**。
   - *'MrDragonFox'* 强调，“你需要关注的是右侧，正如你所见，我的两个 GPU 都被利用了。”
- **使用持续微调微调 Qwen**：在 Qwen2-1.5B-Instruct 上使用 **Continuous Fine-tuning Without Loss** 取得成功，结合了代码 FIM 和 instruct 能力。
   - 成员们对该方法感到兴奋，其中一人建议为那些对文档感到困惑的人*“编写一份教程”*。
- **合并适配器的问题**：用户讨论了合并 **LoRA adapters** 和 4-bit 模型的问题，指出不当的合并可能导致模型虽然显示为 16-bit，但实际上只有 4-bit 的质量。
   - 有人担心 **4-bit 模型被上采样到 16-bit**，可能导致虚假的 16-bit 模型在社区中传播。
- **针对微调的 Bitnet Hack**：用户 **Nisten** 提到为了微调对 Bitnet 进行了 Hack，最终得到了一个 **74MB 的模型**，在 **1 个 CPU 核心上运行速度达到 198 tokens/秒**。
   - 这种 Hack 被描述为*“简直是巫术”*，并将通过 **Skunkworks AI** 进行**开源**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>: 为了微调修改了 bitnet，最终得到了一个 74mb 的文件。它在仅 1 个 cpu 核心上就能以每秒 198 个 token 的速度流畅对话。简直是巫术。稍后将通过 @skunkworks_ai 开源，基础模型在这里：https://huggi...</li><li><a href="https://huggingface.co/johnpaulbin/qwen1.5b-e2-1-lora">johnpaulbin/qwen1.5b-e2-1-lora · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/rombodawg/gemma-2-9b-reuploaded">rombodawg/gemma-2-9b-reuploaded · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/dejavucoder/status/1818707409264861348">sankalp (@dejavucoder) 的推文</a>: 醒醒宝贝，Daniel Han 的视频终于发布了</li><li><a href="https://tenor.com/view/dancing-dj-ravine-groovy-mixing-music-party-gif-21277620">Dancing Dj Ravine GIF - Dancing Dj Ravine Groovy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>: 通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度可延长 6 倍！</li><li><a href="https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners">AI Unplugged 16: Llama 3, AIMO 获胜者, Segment Anything Model 2, LazyLLM</a>: 洞察胜过信息</li><li><a href="https://youtu.be/pRM_P6UfdIc?feature=shared">LLM 的底层技术：Daniel Han</a>: 本次研讨会将分为 3 个一小时板块：如何分析和修复 LLM - 如何发现并修复 Gemma、Phi-3、Llama 和分词器中的 bug；使用 U... 进行微调</li><li><a href="https://mer.vin/2024/07/llama-3-1-fine-tune/">Llama 3.1 微调 - Mervin Praison</a>: https://huggingface.co/mervinpraison/Llama-3.1-8B-bnb-4bit-python 使用自定义数据训练模型，转换为 GGUF，Ollama Modelfile，Ollama 创建自定义模型
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1268651575664971787)** (4 条消息): 

> - `Google new model`
> - `OpenAI vs Google` 


- **Google 的新模型击败了 OpenAI**：*Google 终于凭借[新模型](https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/)击败了 OpenAI*。
   - 一位用户分享了一个 Reddit 链接，重点介绍了声称超越 OpenAI 的 **Google 新模型**。
- **用户反应怀疑**：*我不敢相信...* 是对传闻中 Google 消息的最初反应。
   - 另一位用户以 *ummm* 表示怀疑，对信息的可靠性提出质疑。



**提到的链接**：<a href="https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/">Reddit - 深入探索一切</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1268282102487715891)** (130 条消息🔥🔥): 

> - `Python versions for Unsloth installation`
> - `Installing Unsloth with Conda`
> - `LoRA fine-tuning issues`
> - `Inference problems with GGUF quantization`
> - `Custom dataset training errors on Llama 3.1` 


- **Python 版本引发争论**：成员们对 Unsloth 与 Python 3.10 和 3.11 版本的兼容性感到困惑，因为按照安装指南操作时出现了不同的结果。
   - Felicitiy00637 分享了在 Compute Canada 的 Narval 集群上的安装问题，指出只有在 'pyproject.toml' 中绕过 xforms 后才成功。
- **Conda 环境明确了设置**：Fjefo 强调了在 Conda 环境中严格遵守指南的重要性，并指出偏差可能会使调试复杂化。
   - 尽管 felicity00637 保证遵循了指南，但困惑一直持续到确认未使用 Conda 为止。
- **LoRA 参数讨论中**：Felicitiy00637 寻求关于 LoRA 参数（如 'r' 和 'lora_alpha'）的澄清，询问它们的定义和推荐值。
   - 社区解释说，LoRA 缩放参数理想情况下应设置为秩 (r) 的两倍，并链接到 [LoRA 参数百科全书](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia) 以获得更深入的见解。
- **GGUF 量化造成混乱**：Akshatiscool 报告称模型在 GGUF 量化后输出乱码，尽管在 Colab 推理期间输出正确。
   - Theyruinedelise 建议检查聊天模板，并承认最近修复了 GGUF 量化中的问题。
- **Llama 3.1 训练受阻**：Bigboypikachu 在 Llama 3.1-8b-instruct 上训练自定义长上下文数据集时遇到了 'Expected all tensors to be on the same device' 错误。
   - 同一个内核在预定义数据集上成功训练，但在自定义数据集上失败，暗示存在上下文长度问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1wlCOvklww1YvACuIRrhkdFFH_vU7Hgbn?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>：又称持续微调。Unsloth 允许你进行持续预训练，以便模型学习新语言。</li><li><a href="https://docs.alliancecan.ca/wiki/Narval/en">Narval - Alliance 文档</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/saving-models">保存模型 | Unsloth 文档</a>：了解如何保存微调后的模型，以便在您喜欢的推理引擎中运行。</li><li><a href="https://youtu.be/TKmfBnW0mQA?t=740">修复 Gemma、Llama 和 Phi 3 中的 Bug：Daniel Han</a>：我们为 Gemma 修复 8 个 Bug、为 Llama 3 修复多个分词问题、修复滑动窗口 Bug 以及将 Phi-3 Mistral 化的背后故事，并了解我们如何...</li><li><a href="https://github.com/unslothai/unsloth/issues/839">FastLanguageModel 在 PromptTemplate 和其他复杂事项上存在问题 · Issue #839 · unslothai/unsloth</a>：我正尝试在 Unsloth 环境中指定 Prompt 以应用 RAG，但不幸的是，目前的 Unsloth 环境存在一些复杂的问题。首先，我将提供运行缓慢但有效的代码。...</li><li><a href="https://docs.unsloth.ai/basics/lora-parameters-encyclopedia>">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/27985">`KeyError: &#39;Cache only has 0 layers, attempted to access layer with index 0&#39;` · Issue #27985 · huggingface/transformers</a>：系统信息 transformers 版本：4.36.0 平台：Linux-5.15.0-70-generic-x86_64-with-glibc2.35 Python 版本：3.11.4 Huggingface_hub 版本：0.19.4 Safetensors 版本：0.3.3 Accelerate 版本...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1268524354942144543)** (5 messages): 

> - `AI 与 Groq 的互操作性`
> - `Black Forest Labs 发布`
> - `FLUX.1 text-to-image 模型`
> - `OpenAI 模型`
> - `Generative AI` 


- **Groq AI 在 finetuning 后仅限于 inference**：成员们讨论了 AI 模型是否可以同时在 **Google AI** 和 **Groq AI** 上运行。
   - 澄清指出，对于 Groq，模型在利用其他服务进行 **finetuning** 后，极有可能只能进行 **inference**。
- **Black Forest Labs 登场**：宣布成立 [Black Forest Labs](https://blackforestlabs.ai/announcing-black-forest-labs/)，这是一家专注于推进媒体领域 **generative deep learning** 模型的新创公司。
   - 他们的首个发布版本 **FLUX.1 系列模型**，旨在推动 **text-to-image synthesis** 的前沿。**Open weights** 使其可用于进一步开发。



**提到的链接**：<a href="https://blackforestlabs.ai/announcing-black-forest-labs/">Announcing Black Forest Labs</a>：今天，我们很高兴地宣布 Black Forest Labs 正式成立。我们深植于 **generative AI** 研究社区，我们的使命是开发和推进最先进的 generati...

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1268663522389131380)** (1 messages): 

> - `Uber One 会员可免费获得 Perplexity Pro` 


- **Uber One 提供免费的 Perplexity Pro**：美国和加拿大的 Uber One 会员现在可以享受一年的免费 **Perplexity Pro**。此优惠有效期至 **10 月 31 日**，允许会员解锁 Perplexity **answer engine** 的全部潜力，该服务价值通常为 **$200**。
- **通过 Perplexity Pro 增强信息发现**：从 Uber 行程中的快速事实查询到家中的详细研究，**Perplexity Pro** 为 Uber One 会员增强了每一个信息发现时刻。
   - 在 [Perplexity Uber One](https://pplx.ai/uber-one) 了解更多关于此项福利及条款的信息。



**提到的链接**：<a href="https://pplx.ai/uber-one">Eligible Uber One members can now unlock a complimentary full year of Perplexity Pro&nbsp;</a>：Uber One 会员现在可以通过 Pro Search 等功能节省更多时间

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1268293825009418292)** (293 条消息🔥🔥): 

> - `Uber One Perplexity Pro 优惠`
> - `AI 搜索引擎评分`
> - `Perplexity 功能对比`
> - `技术问题与 Bug`
> - `AI 的法律应用案例` 


- **Uber One 会员可免费获得 Perplexity Pro**：Perplexity 宣布，美国和加拿大的符合条件的 Uber One 会员从即日起至 2024 年 10 月 31 日可以兑换为期一年的免费 Perplexity Pro。会员们讨论了细节和资格，指出该活动要求使用新的 Perplexity Pro 账号注册，并在此期间保持活跃的 Uber One 会员身份。
- **对比不同的 AI 搜索引擎**：用户分享了对比 Perplexity、Felo.ai 和 Chatlabs 等各种 AI 搜索引擎的经验，重点关注 UI、UX、速度和回答质量等方面。**Perplexity Pro** 通常被评为最高，其次是 **SearchGPT**、**Uncovr 免费版**等。
- **Perplexity App 功能问题与差距**：成员们指出了 Perplexity App 的几个问题，特别是在移动端，例如无法删除上传的文件和生成图像、Android 端性能不佳，以及与 OpenAI 和 Microsoft Copilot 相比缺失重大功能。一位用户表达了对移动端 Bug 和不一致性导致文本丢失的沮丧。
- **排查导出和上传问题**：用户在从页面导出文本和来源时遇到了问题，其中一位指出：*“真的‘不可能’。不可能。永远不会发生。”* 另一位成员报告了在 AIStudio 中尝试上传大型 PDF 时出现的 Token 计数错误。
- **使用 AI 进行法律文档搜索和分析**：一位成员分享了使用 Perplexity 搜索和分析法律文档的积极体验，发现它在定位相关案例方面特别有用。他们询问了如何应用 **Retrieval-Augmented Generation (RAG)** 来搜索大量的证据开示（discovery）文档。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://labs.writingmate.ai">ChatLabs</a>: ChatLabs 是一个面向 LLM 和 AI 爱好者的平台。在一个地方体验超过 30 种 AI 模型。</li><li><a href="https://www.perplexity.ai/hub/blog/eligible-uber-one-members-can-now-unlock-a-complimentary-full-year-of-perplexity-pro">符合条件的 Uber One 会员现在可以解锁一整年免费的 Perplexity Pro&nbsp;</a>: Uber One 会员现在可以通过 Pro Search 等特权节省更多时间。</li><li><a href="https://gitlab.com/monnef/ailin">monnef / AIlin · GitLab</a>: AIlin 是一款将 Perplexity.ai 等 AI 服务与本地计算机连接的工具。</li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w">Complexity: Perplexity 的新扩展</a>: Perplexity AI 的 Complexity 扩展引入了一系列强大的功能，旨在增强用户体验并简化与...的交互。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1268285574679101440)** (10 条消息🔥): 

> - `Perplexity AI 技能与功能`
> - `Flask 安全用户认证`
> - `检查 Pro 账户状态`
> - `喝咖啡对牙齿健康的影响`
> - `下一代 iPhone 发布详情`

- **Perplexity AI 结合了搜索和文本生成**：[Perplexity AI](https://www.perplexity.ai/search/what-is-best-skills-in-perplex-mvRHkNtwTHGP7MIk0q3akA) 是一款强大的工具，它将搜索功能与大语言模型（LLM）相结合，以提供精准且全面的回答。
   - 其显著特点包括有效的**市场研究**和**竞争分析**，帮助用户综合来自多份报告的数据并了解竞争格局。
- **Flask 安全用户身份验证设置**：要在 Flask 中实现安全的用户身份验证，请安装必要的软件包，如 `Flask-Login`、`Flask-SQLAlchemy` 和 `Flask-Bcrypt`，并遵循分步指南。
   - 这涉及创建一个应用工厂（application factory）、定义一个 `User` 模型，并按照[此处](https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw)的演示设置注册、登录和注销的路由。
- **检查 Pro 账户状态的步骤**：要检查账户是否订阅了 Pro，请导航至平台上的账户设置或账单信息。
   - 或者，通过支付历史记录进行验证，或联系客户支持寻求帮助，详情请参阅[此处](https://www.perplexity.ai/search/na-porinde-wae-giboneuro-doeji-MGecFR96SpuhbQ04SrRLfA)。
- **OpenAI 推出超逼真语音模式**：[OpenAI](https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA) 为 ChatGPT 推出了高级语音模式（Advanced Voice Mode），于 2024 年 7 月 30 日让 Plus 订阅者能够体验由 GPT-4o 模型驱动的超逼真音频交互。
   - 该功能引入了实时、自然的对话，具备句中打断和情感语调检测等功能。
- **Folksable 应用通过社交功能增强习惯追踪**：Folksable 是一款习惯追踪应用，鼓励用户通过照片和社交契约与朋友分享更新和进度，以保持责任感。
   - [可在 Android 和 iOS 上使用](https://www.perplexity.ai/search/folksable-app-tb4hRa_OSrmapF0bjyYs1g)，该应用通过加密确保数据隐私，并允许用户创建自定义仪式，通过隐私控制管理可见性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/does-drinking-coffee-have-any-p3oI92DpQc6bMClukg.x8Q">喝咖啡对牙齿健康有负面影响吗？</a>: 喝咖啡会对牙齿健康产生多种负面影响。咖啡具有酸性，会侵蚀牙釉质，导致牙齿敏感和脆弱...</li><li><a href="https://www.perplexity.ai/search/folksable-app-tb4hRa_OSrmapF0bjyYs1g">Folksable 应用</a>: Folksable 是一款照片习惯追踪应用，鼓励用户与朋友分享习惯更新以进行互相监督。用户可以创建私密或...</li><li><a href="https://www.perplexity.ai/search/i-m-curious-about-the-shooting-_W0sHhfITLy9jLaqjCFU9A">我对奥运射击选手的射击姿势很好奇，请告诉我原因...</a>: 奥运射击选手采用特定的姿势和站位，以最大限度地提高准确性、稳定性和表现。以下是所使用的射击姿势概述...</li><li><a href="https://youtu.be/ak19RS6VGGU?si=DLWE5mVmvDeyzLPO">Synchron 脑植入技术进展、OpenAI 语音模式、Friend AI 项链以及 HIV 纳米抗体...</a>: 探索脑机接口技术的最新突破，Synchron 实现了对 Apple Vision Pro 头显的意念控制。这一开创性的...</li><li><a href="https://www.perplexity.ai/search/na-porinde-wae-giboneuro-doeji-MGecFR96SpuhbQ04SrRLfA">我是 por，为什么默认是这个？</a>: 您好。据我理解，您开始用韩语对话，但系统似乎默认设置为英语 (POR)。以下是发生这种情况的几种可能原因：1. 系统设置：正在使用的应用程序或网站的默认语言设置...</li><li><a href="https://www.perplexity.ai/search/when-will-the-next-iphone-be-r-3SV74RalRFqKPIK6m4XgtQ#0">下一代 iPhone 什么时候发布？</a>: 下一代 iPhone（预计为 iPhone 16）预计将于 2024 年 9 月发布。这遵循了 Apple 发布新 iPhone 的典型模式...</li><li><a href="https://www.perplexity.ai/page/openai-begins-hyper-realistic-2_y7h8vPQEWaM4g63WvnVA">OpenAI 开始推出超逼真语音功能</a>: OpenAI 已开始为 ChatGPT 推出备受期待的 Advanced Voice Mode，为部分 Plus 订阅者提供超逼真的音频访问权限...</li><li><a href="https://www.perplexity.ai/search/please-provide-an-example-of-s-EvlJDJwUTfy4IWmobEm0Fw">请提供一个 Flask 中安全用户身份验证的示例</a>: 要在 Flask 应用程序中实现安全的用户身份验证，您可以按照以下步骤操作，包括设置必要的包、创建用户...</li><li><a href="https://www.perplexity.ai/search/what-is-best-skills-in-perplex-mvRHkNtwTHGP7MIk0q3akA">Perplexity AI 中最好的技能是什么？</a>: Perplexity AI 是一款强大的工具，结合了搜索和文本生成能力，利用大语言模型 (LLMs) 来...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1268284762758451251)** (4 条消息): 

> - `Prompt 结果不佳`
> - `Perplexity 引用功能 Beta 版`
> - `在 make.com 上使用 Perplexity API` 


- **用户指出 Prompt 结果不佳**: 用户对最近的 Prompt 结果表示担忧，认为结果质量在倒退。
   - 一位用户询问了可能导致该问题的特定 Prompt 建议。
- **询问 Perplexity 引用功能 Beta 版访问权限**: 一位用户询问了 Perplexity 引用功能 Beta 版的状态，想知道是否仍能获得访问权限。
   - *“嘿，我已经申请了 Perplexity 引用功能的 Beta 版，想知道这些名额是否还在发放，或者有什么办法可以加入？🙂”*。
- **在 make.com 上集成 Perplexity API**: 一位用户询问如何在 make.com 上连接 Perplexity API，并指定使用 Sonnet 3.5 模型生成摘要。
   - 该用户概述了一个需求：使用 Perplexity API 上的模型生成页面，然后将链接发布到 Discord。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1268314148882747582)** (255 messages🔥🔥): 

> - `GPT-4o Image Output` (GPT-4o 图像输出)
> - `Multimodal Training Models` (多模态训练模型)
> - `Voice Model Testing` (语音模型测试)
> - `DALL-E and Imagen 3 Comparisons` (DALL-E 与 Imagen 3 对比)
> - `Alpha Testing Experience` (Alpha 测试体验)


- **GPT-4o 图像输出引发讨论**：讨论集中在 GPT-4o 的图像输出能力及其[示例](https://x.com/gdb/status/1790869434174746805)上，并将其与 DALL-E 3 等其他模型进行了对比。
   - 用户注意到 GPT-4o 的输出似乎更写实，但在 Moderation endpoint（审核端点）方面面临与 DALL-E 3 类似的批评。
- **多模态训练模型的未来**：一位用户提出了多模态模型的未来相关性，即通过视频数据间接学习来标注情绪，并认为在文本转语音（Text to Speech）等任务中，这类模型可能优于单模态模型。
- **语音模型测试与能力**：用户实验了 GPT-4o 的[语音功能](https://platform.openai.com/docs/guides/embeddings/use-cases)，分享了包括口音变化和情感表达在内的各种场景。
   - 发现结果显示该模型具有添加背景音乐和音效的能力，尽管表现并不稳定。
- **DALL-E 与 Imagen 3 的对比**：用户对 DALL-E 和 Imagen 3 进行了对比，并提议运行 Prompt 以观察哪种模型生成的图像效果更好。
   - 初步反馈显示，虽然两者都具有强大的能力，但 Imagen 3 可能存在 Moderation endpoint 问题。
- **Alpha 测试的体验与局限性**：Alpha 测试人员分享了褒贬不一的体验，指出在享受新功能的同时，存在高延迟和偶尔的连接问题。
   - 关于欧洲地区访问权限的辩论表明可用性各不相同，一些用户正在考虑退款。



**提及的链接**：<a href="https://x.com/gdb/status/1790869434174746805">来自 Greg Brockman (@gdb) 的推文</a>：一张由 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有太多值得探索的地方。团队正在努力将其推向世界。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1268331656276607027)** (24 messages🔥): 

> - `Alpha testing eligibility` (Alpha 测试资格)
> - `Custom GPTs issues` (Custom GPTs 问题)
> - `Free AI diagram tools` (免费 AI 图表工具)
> - `Plus subscription impacts` (Plus 订阅影响)
> - `Monetizing GPTs` (GPTs 变现)


- **Alpha 测试资格全凭运气**：当被问及如何成为 Alpha 测试人员时，一位用户简单地回答说这需要运气。
- **Custom GPTs 在配置期间卡住**：一位用户在向其 Custom GPTs 上传 PNG 截图时遇到困难，反复收到“Hmm...something seems to have gone wrong”的错误提示，且未能解决。
- **取消 Plus 订阅后 Custom GPTs 将被禁用**：已确认取消 Plus 订阅将禁用并隐藏用户创建的所有 Custom GPTs。
- **GPTs 变现需要极高的使用量**：讨论显示，获得 GPTs 变现邀请的前提是拥有极高的使用量且位于美国。
   - 尽管最初发布了关于 GPT Store 变现的公告，但由于缺乏进展和承诺功能的推出，用户感到失望。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1268290812542849034)** (12 messages🔥): 

> - `Prompt engineering platforms` (提示词工程平台)
> - `Evaluation tools` (评估工具)
> - `Text reduction strategies` (文本缩减策略)


- **最佳 Prompt Engineering 平台**：一位成员询问哪个平台最适合进行 Prompt Engineering，另一位成员回答是 **Claude 3.5 Sonnet**。
   - Artifacts 和 Projects 功能在这方面的优势受到了赞赏。
- **启发式 Prompt 评估工具**：一位成员对 Prompt 评估和 Steerability（可控性）表示关注，相比完全自动化，更倾向于使用启发式（Heuristic）和原型设计工具。
   - Anthropic Evaluation Tool 获得了正面评价，但用户对能与其他 LLM 配合使用的替代方案感兴趣。
- **用于评估的 Google Sheet**：对于协作式的 Prompt 评估，一位成员建议使用 **带有脚本的 Google Sheet** 可能是最好的方法。
   - 这种方法比其他工具更能促进分享和协作。
- **免费的 AI 绘图工具**：一位成员询问是否有可以画图表的免费 AI 工具。
   - 另一位成员简单地回答：**ChatGPT**。
- **文本长度缩减的挑战**：一位成员询问如何将文本缩减到特定的字符或单词数。
   - 另一位成员澄清说 LLM 很难精确控制字数，建议使用定性的语言来获得更一致的长度。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1268290812542849034)** (12 条消息🔥): 

> - `Prompt Engineering 平台`
> - `人工评估工具`
> - `用于绘制图表的 AI`
> - `缩减文本长度` 


- **Prompt Engineering 的最佳平台**：一位成员询问了关于 **prompt engineering** 的最佳平台，另一位成员建议使用 **Claude 3** 和 **Sonnet**。
   - 他们还提到 **Artifacts + Projects** 是该领域强有力的竞争者。
- **用于可控性的 Anthropic 评估工具**：讨论集中在用于 **prompt 评估** 的 **Anthropic Evaluation Tool**，以及用于启发式方法和原型设计的**可控性（steerability）**。
   - 一位成员建议，**带有脚本的 Google Sheet** 可能是最具协作性且易于分享的替代方案。
- **用于绘制图表的免费 AI 工具**：一位成员询问了可以绘制图表的免费 AI 工具。
   - 另一位成员推荐了 **ChatGPT**，尽管其是否适合绘图存在争议。
- **将文本缩减至特定长度**：一位成员询问如何将文本缩减到特定的**字符**或**单词计数**。
   - 另一位成员解释说，由于 **LLM** 的特性，它们无法确保精确的计数，并建议改用 *short* 或 *long* 等定性语言术语。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1268284072963211444)** (55 条消息🔥🔥): 

> - `对 FSDP 的批评`
> - `分片 LLaMA 405B`
> - `vLLM 和 LLaMA 3.1 支持`
> - `Megatron 论文讨论`
> - `Torchrun 和 GPU 显存问题` 


- **对 FSDP 的批评引发辩论**：一位成员批评 FSDP，称其“挺烂的（kind of ass）”，这引发了关于其应用和可扩展性的讨论。
   - 另一位成员指出，虽然 **FSDP** 并非适用于所有场景，但“就易用性而言，它是无可匹敌的”。
- **在多节点间分片 LLaMA 405B 的困扰**：成员们讨论了在 2 个节点（配备 8 x H100s）上对 **LLaMA 405B** 进行分片的问题，主要在推理过程中遇到困难。
   - 建议包括使用 vLLM 并探索量化方法，尽管原提问成员更倾向于避免使用 vLLM。
- **vLLM 扩展对 LLaMA 3.1 的支持**：一位成员强调 **vLLM** 现在支持 LLaMA 3.1 模型系列，并增强了对更长上下文窗口和流水线并行（pipeline parallelism）的支持。
   - 他们分享了一篇 [博客文章](https://blog.vllm.ai/2024/07/23/llama31.html)，详细介绍了包括 FP8 量化在内的这些新特性。
- **Megatron 论文引起关注**：成员们对 2021 年的 Megatron 论文表现出兴趣，讨论了其相关性并分享了 [论文链接](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf) 及相关资源。
   - 还分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab) 以进一步理解分布式训练概念。
- **Torchrun 和 GPU 显存问题**：一位成员报告了 **torchrun** 的问题，即手动停止脚本时 GPU 显存未释放。
   - 建议包括 [使用 @record](https://pytorch.org/docs/stable/elastic/errors.html) 来处理错误并确保 GPU 显存被清除。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/elastic/errors.html">错误传播 &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://blog.vllm.ai/2024/07/23/llama31.html">宣布 vLLM 支持 Llama 3.1</a>：今天，vLLM 团队很高兴与 Meta 合作宣布支持 Llama 3.1 模型系列。Llama 3.1 带来了令人兴奋的新特性，具有更长的上下文长度（高达 128K tokens）...</li><li><a href="https://people.eecs.berkeley.edu/~matei/papers/2021">Index of /~matei/papers/2021</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2208.11174">通过微基准测试和指令级分析揭秘 Nvidia Ampere 架构</a>：图形处理单元 (GPU) 现在被认为是加速 AI、数据分析和 HPC 等通用工作负载的领先硬件。在过去的十年中，研究人员一直专注于...</li><li><a href="https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHAN">EfficientML.ai 第 17 讲：分布式训练（第一部分）(MIT 6.5940, 2023 秋季, Zoom)</a>：EfficientML.ai 第 17 讲：分布式训练（第一部分）(MIT 6.5940, 2023 秋季, Zoom) 讲师：Song Han 教授 幻灯片：https://efficientml.ai</li><li><a href="https://www.youtube.com/watch?v=u6NEbhNiyAE&ab_channel=MITHANLab">EfficientML.ai 第 17 讲：分布式训练（第一部分）(MIT 6.5940, 2023 秋季, Zoom)</a>：EfficientML.ai 第 17 讲：分布式训练（第一部分）(MIT 6.5940, 2023 秋季, Zoom) 讲师：Song Han 教授 幻灯片：https://efficientml.ai
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1268718331687342092)** (9 messages🔥): 

> - `Triton tiled matmul tutorial` (Triton 分块矩阵乘法教程)
> - `GROUP_SIZE_M argument` (GROUP_SIZE_M 参数)
> - `Block and group tiling` (Block 与 Group 分块)
> - `L2 cache optimization` (L2 缓存优化)


- **关于 Triton 分块矩阵乘法教程中 GROUP_SIZE_M 的说明**：一位用户询问了 [Triton 分块矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) 中 `GROUP_SIZE_M` 参数的作用，对其目的和优势提出了疑问。
   - 另一位用户解释说，`GROUP_SIZE_M` 控制在切换列之前处理多少个行块（blocks of rows），从而提高 L2 缓存命中率；它是位于 Block Tiling 之上、Warp/Thread Tiling 之下的一个缓存分块层级。
- **GROUP_SIZE_M 与最大值的使用对比**：讨论继续，一位用户询问为什么不总是将 `GROUP_SIZE_M` 设置为最大可能值。
   - 回复指出，类似的逻辑也适用于 Shared Memory 中的 Block Tiling，将其设置为最大值可能会导致教程中解释的效率低下，这类似于在设置 Block Size 时不使用维度的全长。



**提及的链接**：<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations">Matrix Multiplication &mdash; Triton documentation</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1268430240107986964)** (3 messages): 

> - `Running video predictor example notebook` (运行视频预测示例 Notebook)
> - `Google Colab example for sam2` (sam2 的 Google Colab 示例)
> - `GitHub issue for segment-anything-2` (segment-anything-2 的 GitHub Issue)


- **运行视频预测示例 Notebook 失败**：一位成员无法运行来自 **sam2** 的视频预测示例 Notebook。
   - 尽管在本地尝试了各种修改，他们仍无法使其正常工作，并寻求社区建议。
- **找到了 sam2 的替代 Google Colab Notebook**：该成员发现了一个与其配置兼容的 [Google Colab notebook](https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing)。
   - 他们在相关的 [GitHub issue](https://github.com/facebookresearch/segment-anything-2/issues/40) 中感谢了提供解决方案的贡献者。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/segment-anything-2/issues/40">Google Colab example · Issue #40 · facebookresearch/segment-anything-2</a>：不是一个 Bug。如果有人需要，我构建了一个可以运行该模型的 Colab - https://colab.research.google.com/drive/1Un09HITLLM-ljkG1Ehn9cJjdwk8FVI_1?usp=sharing 端到端可用。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1268572448656195716)** (1 messages): 

> - `Llama 3 Herd of Models` (Llama 3 模型群)
> - `AIMO: Findings from the winners` (AIMO：获胜者的发现)
> - `SAM 2: Segment Anything Model 2` (SAM 2：分割一切模型 2)
> - `LazyLLM` (LazyLLM)


- **Meta 发布 Llama 3.1: Herd of Models**：Meta 发布了 [Llama 3.1](https://datta0.substack.com/i/143781557/llama)，其中包括一个拥有 **4050 亿参数** 的新模型，该模型在由 **16,000 块 H100 GPU** 组成的集群上使用 **15.6 万亿 Token** 训练而成。
   - 他们利用 **Roberta** 等模型进行过滤，并创建了用于训练的高质量数据集。
- **AIMO 获胜者发现解析**：本周的分析包括对 AIMO 竞赛获胜者发现的详细审查。
- **SAM 2：Segment Anything Model 的继任者**：讨论涵盖了 [SAM 2](https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners)，即 Segment Anything Model 的下一次迭代。
- **LazyLLM 提升 LLM 推理性能**：一个环节专门讨论了 **LazyLLM**，其旨在提高 LLM 在推理过程中的性能。



**提及的链接**：<a href="https://datta0.substack.com/p/ai-unplugged-16-llama-3-aimo-winners">AI Unplugged 16: Llama 3, AIMO winners, Segment Anything Model 2, LazyLLM</a>：洞察胜过信息

  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1268311686134108203)** (3 条消息): 

> - `数字视频窃听`
> - `NVIDIA Titan 系列显卡`
> - `Segment Anything Video (SA-V) 数据集` 


- **革新数字视频窃听技术**：最近的一篇 [arXiv 论文](https://arxiv.org/abs/2407.09717) 讨论了一种通过分析 HDMI 线缆产生的电磁波来窃听数字视频显示的新方法，被称为 **TEMPEST**。
   - 作者提出使用深度学习模块将观察到的电磁信号映射回显示图像，克服了数字信号高带宽和非线性映射带来的挑战。
- **NVIDIA 下一代 Titan GPU 亮相**：根据 [Wccftech 的一篇文章](https://wccftech.com/nvidia-next-gen-titan-graphics-card-exists-flagship-blackwell-gpu/)，NVIDIA 基于 Blackwell GPU 架构的新型 Titan 级显卡确实存在，但其发布仍存疑问。
   - 之前的 Titan 版本包括 2018 年的 **Titan RTX**，目前有推测认为是否会有新的型号推出。
- **Meta 发布用于 AI 研究的海量 SA-V 数据集**：Meta 推出了 [Segment Anything Video (SA-V) 数据集](https://ai.meta.com/datasets/segment-anything-video/)，包含 5.1 万个视频和 64.3 万个时空分割掩码。
   - 该数据集支持计算机视觉研究，由手动标注和自动生成的 masklets 组成，平均视频分辨率为 **1401×1037 像素**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wccftech.com/nvidia-next-gen-titan-graphics-card-exists-flagship-blackwell-gpu/">NVIDIA's Next-Gen Titan Graphics Card Does Exist &amp; Based on Flagship Blackwell GPU</a>: 据报道，NVIDIA 已经拥有基于其下一代 Blackwell GPU 架构的 Titan 级显卡，但其发布前景不明。</li><li><a href="https://arxiv.org/abs/2407.09717">Deep-TEMPEST: Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations</a>: 在这项工作中，我们通过分析线缆和连接器（特别是 HDMI）无意中发出的电磁波，解决了数字视频显示的窃听问题。T...</li><li><a href="https://ai.meta.com/datasets/segment-anything-video/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=sam2">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1268324131812671594)** (2 条消息): 

> - `Ampere A100 SM 组织结构`
> - `处理块中的 Warp 分布`
> - `硬件设计选择`
> - `Hopper 架构` 


- **Ampere A100 SM 划分为更小的处理块**：一位用户询问为什么拥有 64 个核心的 Ampere A100 SM 被组织成 **四个各含 16 个核心的处理块**，而不是 32 个核心以匹配 warp 大小。
   - 另一位用户推测，考虑到 **kernel 需求**、硅片空间、带宽和延迟参数，NVIDIA 做出这一选择可能是为了维持一种能让硬件保持繁忙的平衡。
- **关于硬件设计选择的推测**：一位用户提到，硬件设计涉及在 **硅片空间** 与利用率之间进行权衡，更多的单元会占用更多空间。
   - 他们建议，这可能是一种微妙的平衡行为，以确保增加的单元在带宽和延迟方面的成本是值得的。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1268320487675793520)** (11 条消息🔥): 

> - `.py vs .ipynb`
> - `Quantization-Aware Training (QAT)`
> - `.ipynb 转换为 .py`
> - `Jupyter 和 PyTorch 的 GitHub 仓库`
> - `QAT 与 PTQ 的性能比较` 


- **.py vs .ipynb 易用性辩论**：讨论集中在 .py 文件是否比 .ipynb 文件更容易运行和修改，一些成员建议使用各种工具和方法进行转换。
   - 一位成员提到使用 [LibCST](https://github.com/Instagram/LibCST) 进行转换，而另一位成员指出 Colab 和 Jupyter UI 中提供了导出选项。
- **Quantization-Aware Training 提高 PyTorch 模型精度**：[PyTorch 博客文章](https://pytorch.org/blog/quantization-aware-training) 讨论了端到端的 Quantization-Aware Training (QAT) 流程，与训练后量化（PTQ）相比，该流程在 Llama3 的 hellaswag 上可以**恢复高达 96% 的精度下降**，在 wikitext 上可以**恢复 68% 的困惑度（perplexity）下降**。
   - 该博客还介绍了 [torchao](https://github.com/pytorch/ao/) 中的 QAT API，并强调了它们与 [torchtune](https://github.com/pytorch/torchtune/) 的集成。
- **实际应用中的 QAT vs. PTQ**：一位成员解释了 Quantization-Aware Training 与 Quantized Training 之间的关键区别，强调了 QAT 带来的显著性能提升。
   - 另一位参与者对将低秩自适应（low-rank adaptation）与 QAT 结合以增强性能表示期待。
- **对 QAT 过拟合的担忧**：一位用户询问在 QAT 过程中是否检查了过拟合，并建议 MMLU 可能是验证的一个好指标。
   - 这引发了另一位用户对验证的进一步提及，表明社区对 QAT 彻底评估的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/quantization-aware-training">Quantization-Aware Training for Large Language Models with PyTorch</a>：在这篇博客中，我们介绍了 PyTorch 中针对大语言模型的端到端 Quantization-Aware Training (QAT) 流程。我们展示了 PyTorch 中的 QAT 如何恢复高达 96% 的精度下降...</li><li><a href="https://github.com/jupyter/notebook/blob/main/docs/source/examples/Notebook/Running%20Code.ipynb?short_path=c932132">notebook/docs/source/examples/Notebook/Running Code.ipynb at main · jupyter/notebook</a>：Jupyter 交互式笔记本。通过在 GitHub 上创建账户为 jupyter/notebook 的开发做出贡献。</li><li><a href="https://github.com/Instagram/LibCST">GitHub - Instagram/LibCST: A concrete syntax tree parser and serializer library for Python that preserves many aspects of Python&#39;s abstract syntax tree</a>：一个用于 Python 的具体语法树（CST）解析器和序列化器库，保留了 Python 抽象语法树（AST）的许多方面 - Instagram/LibCST
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1268294297380196484)** (177 条消息🔥🔥): 

> - `GELU 变更`
> - `Llama 3.1 参考实现`
> - `参考实现问题`
> - `TorchChat`
> - `RoPE 缩放`

- **LLMC 的 GELU 优化 PR**：提交了一个新的 [PR](https://github.com/karpathy/llm.c/pull/721)，旨在将更快的 GELU 更改从 FP8 分支移至 master 分支，这略微改善了 validation loss。
   - *令人惊讶的是，它确实对 val loss 有一点点帮助，但可能又是噪声*。
- **Llama 3.1 实现问题**：成员们讨论了从 [Meta 的 repo](https://github.com/meta-llama/llama-models) 下载 Llama 3.1 模型后缺乏运行文档的问题，并分享了尝试加载和运行该模型的代码片段。
   - 据推测，缺少一个 10 行的 Python 片段来实现直接运行，而 [inference scripts](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/local_inference/README.md) 被认为过于复杂。
- **TorchChat 作为 Llama 3.1 的参考**：分享了一个以 PyTorch 发布的新 [TorchChat repository](https://github.com/pytorch/torchchat) 形式存在的 Llama 3.1 参考实现。
   - 该实现为 Llama 3.1 模型的本地和基于服务器的运行提供了详细指南。
- **RoPE scaling 和专门特性**：对话包括了关于 RoPE scaling 在 Llama 3.1 中有何不同的详细讨论，以及相应更新 [reference implementations](https://github.com/karpathy/llm.c/blob/7e0c497936540a44338e214bc230a1f041090fcb/llmc/encoder.cuh#L161) 的必要性。
   - 成员们分享了在 CUDA 代码中集成此功能以实现更好 fine-tuning 操作的见解。
- **Llama 3.1 上的 Fine-tuning 技术**：讨论转向了 fine-tuning，权衡了 full finetuning 与 LoRA 方法，并指出 LoRA 在较小数据集上效率更高。
   - 有建议认为有时 *仅对 completions* 进行训练可以产生更好的结果，并分享了来自 [unsloth repo](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L1456) 的实现片段。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/meta-llama/llama3/blob/main/llama/generation.py">llama3/llama/generation.py at main · meta-llama/llama3</a>: Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账户，为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://github.com/meta-llama/llama3/blob/main/example_text_completion.py">llama3/example_text_completion.py at main · meta-llama/llama3</a>: Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账户，为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/inference/local_inference/README.md">llama-recipes/recipes/quickstart/inference/local_inference/README.md at main · meta-llama/llama-recipes</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama3 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用...</li><li><a href="https://github.com/karpathy/nano-llama31/tree/master">GitHub - karpathy/nano-llama31: nanoGPT style version of Llama 3.1</a>: Llama 3.1 的 nanoGPT 风格版本。通过在 GitHub 上创建账户，为 karpathy/nano-llama31 的开发做出贡献。</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理您的 Git 仓库，像专业人士一样审查代码，跟踪 Bug 和功能...</li><li><a href="https://github.com/karpathy/nano-llama31">GitHub - karpathy/nano-llama31: nanoGPT style version of Llama 3.1</a>: Llama 3.1 的 nanoGPT 风格版本。通过在 GitHub 上创建账户，为 karpathy/nano-llama31 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/721">Faster GELU forward &amp; backward using MUFU.TANH for SM7.5+ by ademeure · Pull Request #721 · karpathy/llm.c</a>: 这些是更快的 GELU 核函数，使用了 NVIDIA 在 Turing (SM7.5) 架构中引入的硬件指令，但据我所知，除了 PTX 之外从未公开过，可能是因为它稍微不那么精确...</li><li><a href="https://github.com/meta-llama/llama-models">GitHub - meta-llama/llama-models: Utilities intended for use with Llama models.</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账户，为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/39716">Do not modify global random state · Issue #39716 · pytorch/pytorch</a>: 🚀 特性：目前，实现可复现性的推荐方法是设置全局随机种子。我建议所有需要随机源的函数都接受一个局部...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L1456">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLMs，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py">llama-models/models/llama3_1/api/model.py at main · meta-llama/llama-models</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账户，为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://www.picoquant.com/products/category/tcspc-and-time-tagging-modules/hydraharp-400-multichannel-picosecond-event-timer-tcspc-module">
     
        HydraHarp 400 - 多通道皮秒事件计时器和 TCSPC 模块
    
</a></li></ul>

| PicoQuant</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1077)">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/karpathy/llm.c/blob/7e0c497936540a44338e214bc230a1f041090fcb/llmc/encoder.cuh#L161">llm.c/llmc/encoder.cuh at 7e0c497936540a44338e214bc230a1f041090fcb · karpathy/llm.c</a>: 使用纯粹简单的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/trholding/llama2.c/blob/8a0ad84b9ee94fad175e5687fb8774503efbd23b/runq.c#L653">llama2.c/runq.c at 8a0ad84b9ee94fad175e5687fb8774503efbd23b · trholding/llama2.c</a>: Llama 2 无处不在 (L2E)。通过在 GitHub 上创建账号为 trholding/llama2.c 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile</a>: 在服务器、桌面和移动端本地运行 PyTorch LLM - pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/blob/main/generate.py">torchchat/generate.py at main · pytorch/torchchat</a>: 在服务器、桌面和移动端本地运行 PyTorch LLM - pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchchat/blob/main/build/model.py">torchchat/build/model.py at main · pytorch/torchchat</a>: 在服务器、桌面和移动端本地运行 PyTorch LLM - pytorch/torchchat
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1268389635982233650)** (1 条消息): 

> - `L2 latency as hyperparameter` (L2 延迟作为超参数)
> - `latency bound algorithm` (延迟受限算法)


- **关于将 L2 延迟用作超参数的问题**：一名成员询问如何在 **20 亿个选项**的配置中将 **L2 latency** 作为超参数使用。
   - 该成员还询问了 **latency bound algorithm** 的定义和应用。
- **理解延迟受限算法**：一位用户寻求关于 **latency bound algorithm** 含义的澄清。
   - 这是继之前关于 **L2 latency** 在超参数调优中作用的问题之后的进一步探讨。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1268283861427552266)** (4 条消息): 

> - `Gradient involvement` (Gradient 的参与)
> - `Seq Parallel`
> - `Triton Kernels`
> - `Hackathon`
> - `Event Criteria` (活动标准)


- **Gradient 的 Michael 探索 Seq Parallel 和 Triton Kernels**：来自 Gradient 的 Michael 宣布他正在为一些独特的架构开发 **Seq Parallel** 或 **Triton Kernels**，并邀请其他人加入他在旧金山的活动。
- **新手对 Hackathon 式学习的兴趣**：Pacomann 表达了参加该活动的兴趣，强调希望以 **hackathon-style** 的形式学习更多知识。
- **关于活动审批标准的问题**：Evil666man 询问是否有审批标准，还是**先到先得**。
   - Kashimoo 回应道，暗示如果是先到先得的话，活动早就满员了。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1268616070860902400)** (1 条消息): 

> - `Stable Fast 3D Launch` (Stable Fast 3D 发布)
> - `Technical Report` (技术报告)
> - `3D Asset Generation Technology` (3D 资产生成技术)
> - `Speed and Quality of 3D Reconstruction` (3D 重建的速度与质量)
> - `Applications in Gaming and VR` (在游戏和 VR 中的应用)


- **Stable Fast 3D 发布 🚀**：Stability AI 推出了 **Stable Fast 3D**，该模型仅需 **0.5 秒**即可将单张输入图像转换为详细的 3D 资产，为 3D 重建的速度和质量树立了新标准。[了解更多并获取报告](https://stability.ai/news/introducing-stable-fast-3d)。
   - *“Stable Fast 3D 前所未有的速度和质量使其成为 3D 工作中快速原型设计的宝贵工具。”*
- **Stable Fast 3D 的工作原理**：用户可以上传物体的单张图像，**Stable Fast 3D** 会快速生成完整的 3D 资产，包括 **UV unwrapped mesh**（UV 展开网格）、材质参数以及减少了光照烘焙的反射率颜色 (albedo colors)。[观看视频了解详细的模型改进](https://www.youtube.com/watch?v=uT96UCBSBko)。
   - 可选的四边形或三角形重网格化 (remeshing) 仅增加 **100-200ms** 的处理时间，增强了其在各行各业的实用性。



**提到的链接**：<a href="https://stability.ai/news/introducing-stable-fast-3d">Introducing Stable Fast 3D: Rapid 3D Asset Generation From Single Images &mdash; Stability AI</a>：我们很高兴推出 Stable Fast 3D，这是 Stability AI 在 3D 资产生成技术方面的最新突破。这一创新模型将单张输入图像转换为详细的 3D 资产，设定了...

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1268284094136188948)** (212 messages🔥🔥): 

> - `为电视角色训练 Loras`
> - `SD3 模型使用`
> - `处理 VAE 问题`
> - `Creative Upscaler 困惑`
> - `Flux 模型发布` 


- **在 SD3 中为电视角色训练 Loras**：成员们讨论了如何训练 2 个电视角色的 Loras 并让它们出现在同一张图像中，推荐使用 SD3，因为它具有独特的理解能力。
   - 建议包括从提示词（prompting）开始，在 auto1111 中使用 regional prompter 扩展，并通过社区测试进行验证。
- **SD3 Medium 模型问题与使用**：用户在从 Huggingface 加载 SD3 Medium 时遇到错误，例如 'AttributeError: NoneType object has no attribute lowvram'。
   - 讨论的解决方案包括下载所有模型组件，使用 ComfyUI 工作流，以及探索其他兼容的 UI，如 Auto1111。
- **管理 VAE 设置以防止图像变红**：社区成员解决了渲染图像在 95% 时变红的问题，这主要归因于 VAE 设置。
   - 解决方案包括使用 '--no-half-vae' 设置，并分享针对不同显卡和 VAE 组合的故障排除技巧。
- **澄清 Stability AI 的 Creative Upscaler**：针对 NightCafe 中提到的 'Creative Upscaler' 的困惑，澄清了它并非真正的 Stability AI 产品。
   - 成员们推荐了使用 ERSGAN、transformers 以及社区论坛上分享的多阶段工作流作为替代放大技术。
- **Black Forest Labs 发布 Flux 模型**：社区欢迎 Flux 模型的发布，该模型在图像质量和参数数量上都有显著提升。
   - 用户讨论了该模型在不同 GPU 上的表现，强烈推荐使用 4090，并指出其在渲染手部和手指方面表现出色。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">宣布由 Black Forest Labs 开发的 Flux：文本生成图像模型的下一次飞跃</a>：Flux 是迄今为止最大的 SOTA 开源文本生成图像模型，由 Stable Diffusion 的原班人马 Black Forest Labs 开发，现已在 fal 上可用。Flux 挑战了创意边界...</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main">stabilityai/stable-diffusion-3-medium 在 main 分支</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ehiz51/flux_image_examples/#lightbox">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Stability AI 的生成模型</a>：Stability AI 的生成模型。通过在 GitHub 上创建一个账号来为 Stability-AI/generative-models 的开发做出贡献。</li><li><a href="https://comfyworkflows.com/">Comfy Workflows</a>：分享、发现并运行数千个 ComfyUI 工作流。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1268285391820165192)** (121 条消息🔥🔥): 

> - `LM Studio 中的退出代码 (Exit codes)`
> - `Gemma 2 模型`
> - `模型嵌入 (Embedding) 与 LLaMA 能力`
> - `LM Studio 中的 Bug 与故障排除`
> - `未来的 LM Studio 功能与用户请求` 


- **成员报告各种退出代码 (Exit Codes)**：用户在不同系统上遇到了如 6 和 0 等不同的退出代码，引发了关于系统兼容性和调试的讨论。
- **Gemma 2 模型：兼容性与错误**：社区成员在运行 **Gemma 2 2B** 模型时遇到问题，特别是在旧硬件或特定硬件上，部分模型需要新版本的 LM Studio。
- **使用 LLaMA 进行嵌入 (Embedding) 及未来前景**：出现了关于在 LM Studio 中使用 **LLaMA** 进行嵌入的咨询，并提到了 [LLM2Vec](https://github.com/McGill-NLP/llm2vec) 等项目作为潜在解决方案。
- **LM Studio 中的 Bug 与故障排除**：用户强调了各种 Bug，包括 GPU 卸载 (offload) 问题以及与 VPN/DNS 设置相关的网络错误。
- **用户对未来 LM Studio 功能的请求**：用户表达了对 **TTS 语音**、模型联网访问以及用于文档交互的 **RAG** 等功能的期望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B">meta-llama/Meta-Llama-3.1-405B · Hugging Face</a>: 未找到描述</li><li><a href="https://paperswithcode.com/task/visual-question-answering">Papers with Code - Visual Question Answering (VQA)</a>: **视觉问答 (VQA)** 是计算机视觉中的一项任务，涉及回答关于图像的问题。VQA 的目标是教机器理解图像内容并回答...</li><li><a href="https://github.com/McGill-NLP/llm2vec">GitHub - McGill-NLP/llm2vec: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39;</a>: 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 的代码 - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1268376543730733157)** (24 条消息🔥): 

> - `LM Studio 中的 GPU 卸载 (offload)`
> - `Stable Diffusion 模型兼容性`
> - `用于图像生成的 Amuse AI`
> - `Proxmox 学习` 


- **启用 iGPU 以获得更好的 VRAM 可用性**：一位成员尝试启用其 iGPU 以释放 RTX3090 上的 VRAM，以便在 LM Studio 中加载模型，但在空闲时仍看到 0.5/24.0 GB 的 VRAM 占用。
   - 另一位成员澄清说，如果没有 [OpenCL 插件包](https://discord.com/channels/1110598183144399058/1111797717639901324/1268091222686175246)，iGPU 是不受支持的；支持 Vulkan 的新 Beta 版本可能会有所帮助。
- **LM Studio 不支持 Stable Diffusion**：用户报告在尝试加载 Stable Diffusion 模型时出错，结果显示 LM Studio 不支持 Stable Diffusion 等图像生成模型。
   - 建议使用 [Stability Matrix](https://stability.ai/)、Automatic1111 或 Amuse AI 来完成这些任务。
- **Amuse AI 现已面向 Radeon 用户开放**：一位成员宣布 Amuse AI 已面向 Radeon 用户开放，允许在具有新 EZ 模式的 GPU 上进行 Stable Diffusion 图像生成。
   - 它提供诸如 AI 滤镜和草图转图像生成等功能，无需登录或付费。
- **初学者的 Proxmox 学习技巧**：一位参与者询问了关于 Proxmox 驱动程序的建议，并被建议先在 Windows 下的 VirtualBox 中练习 Proxmox。
   - 分享了一份详尽的[学习计划](https://example.com)，涵盖了从安装到 GPU 透传 (passthrough) 和 LLM 利用的主题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/spongebob-patrick-star-shocked-loop-surprised-gif-16603980">Spongebob Patrick Star GIF - Spongebob Patrick Star Shocked - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.amuse-ai.com/">Amuse</a>: Stable Diffusion 图像与视频生成
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1268284046572650670)** (88 条消息🔥🔥): 

> - `AI 水印技术`
> - `NTIA 关于 AI 开放性的报告`
> - `GitHub Models 发布`
> - `Deepfakes 的法律挑战`
> - `GPT-2 模型改进` 


- **AI 水印技术的信任问题引发辩论**：成员们讨论了水印技术在解决 AI 信任问题方面的有效性，一些人认为它仅在机构设置中有效，无法完全防止滥用。
   - 讨论表明，需要更好的文化规范和信任机制，而非仅仅依靠水印技术，来应对 Deepfakes 和虚假内容的传播。
- **NTIA 在最新报告中支持开源模型**：NTIA 发布了一份[报告](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation)，主张 AI 模型的开放性，同时建议进行风险监控，这影响了美国的政策考量。
   - 参与者指出，NTIA 隶属于商务部并直接向白宫汇报，这增加了其关于 AI 模型开放性政策建议的分量。
- **GitHub 推出集成 AI 模型**：GitHub 发布了 [GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/)，允许开发者直接在其平台上访问和实验顶尖的 AI 模型。
   - 社区成员推测，此举可能是为了通过将 AI 能力集成到开发者现有的工作流中，从而与 Hugging Face 等平台竞争。
- **监管 Deepfakes 的挑战**：成员们讨论了围绕 Deepfakes 的监管复杂性，特别是诽谤和名誉损害问题，以及在全球范围内执法法律的难度。
   - 讨论强调了对起诉 Deepfake 创作者的可行性以及此类内容被用于勒索计划的担忧。
- **利用新论文和技术优化 GPT-2**：一位正在研究 GPT-2 模型的参与者寻求关于整合先进技术的建议，此前已实现了 Rotary Positional Embeddings 和 Grouped Query Attention。
   - 社区成员建议参考最近的论文和评估指标（如 HumanEval），以进一步改进模型并有效衡量其性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ashtom/status/1819041110200906202">来自 Thomas Dohmke (@ashtom) 的推文</a>：在管理代码的地方直接构建 AI 应用。通过 GitHub Models，现在超过 1 亿名开发者可以在其工作流所在处——直接在 GitHub 上访问并实验顶尖 AI 模型...</li><li><a href="https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation">NTIA 支持开放模型以促进 AI 创新 | 国家电信和信息管理局</a>：未找到描述</li><li><a href="https://fixupx.com/impershblknight/status/1818769082944307517?t=41UyAwMxUTUMwBIspUiHRQ&s=19">来自 Imperishable Knight ⛩️ (RJ) (@impershblknight) 的推文</a>：给希望获得 #ChatGPT 高级语音 alpha 访问权限的 Plus 用户的建议：你尝试过开启这些设置吗？我最初没收到 AV 邀请，但我开启它们几小时后...</li><li><a href="https://en.wikipedia.org/wiki/United_States_Department_of_Commerce#Structure">美国商务部 - 维基百科</a>：未找到描述</li><li><a href="https://www.federalregister.gov/documents/2023/04/13/2023-07776/ai-accountability-policy-request-for-comment">联邦公报 :: 请求访问</a>：未找到描述</li><li><a href="https://archive.is/2yfdW">英国 AI 法案将重点关注 ChatGPT 类模型</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1268284431370555392)** (7 messages): 

> - `system prompt style model training`
> - `MLCommons AlgoPerf results`
> - `synthetic data generation`
> - `system prompt generalization` 


- **System Prompt 风格模型训练查询**：一位成员询问是否存在关于 **system prompt 风格模型** 如何训练的论文，认为这些模型是合成的，因为它们在自然数据中并不存在。
   - 另一位成员建议，一旦有了经过 system prompt 微调的模型，就可以自动生成或以极少的人力投入生成这些数据。
- **MLCommons AlgoPerf 结果公布**：[MLCommons AlgoPerf](https://x.com/mlcommons/status/1819098247270695254) 结果已出，这项设有 5 万美元奖金的竞赛强调，非对角预条件化（non-diagonal preconditioning）的表现优于 Nesterov Adam 28%，在无超参数算法领域创下了新的 SOTA。
   - 随着 **distributed shampoo** 在竞赛中获胜，这一成就受到了赞誉。
- **用于 System Prompt 的合成数据**：讨论了使用 **synthetic data generation**（合成数据生成）和 GPT-4 蒸馏来为 chat/instruct 模型生成 system prompt。
   - 一位成员表示需要更多研究来支持关于 **system prompt** 生成在确保模型 guardrails（护栏）方面有效性的说法。



**提及的链接**：<a href="https://x.com/mlcommons/status/1819098247270695254">来自 MLCommons (@MLCommons) 的推文</a>：@MLCommons #AlgoPerf 结果揭晓！🏁 5 万美元奖金竞赛产出了比 Nesterov Adam 快 28% 的神经网络训练方法，采用非对角预条件化。无超参数算法的新 SOTA...

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1268591088751480832)** (15 messages🔥): 

> - `Scaling law experiments`
> - `Validation log-likelihood anomalies`
> - `Double descent phenomenon`
> - `Broken Neural Scaling Law (BNSL) paper`
> - `Task-specific scaling behavior` 


- **Scaling law 实验揭示异常**：比较在不同大小子集上训练的模型的验证集对数似然（validation log-likelihood）实验显示，在 **1e6 个序列**上训练的模型表现明显逊于在更少或更多序列上训练的模型。
- **对验证集性能下降的推测与解释**：成员们最初怀疑是 **data processing pipeline**（数据处理流水线）中存在 bug，但未能发现任何问题，从而引发了关于 double descent（双重下降）现象的讨论。
   - 另一位用户提到了 [BNSL 论文](https://arxiv.org/abs/2210.14891)，该论文展示了关于数据集大小的类似 double descent 行为，这导致了关于此现象是否取决于任务的困惑。
- **关于 Double descent 的辩论**：Double descent 被提及为一个潜在原因，尽管传统上它与参数量的增加而非数据集大小相关。
   - 一位用户澄清说，参数量和数据集大小都可能发生 double descent，并指出该问题可能是特定于任务的。



**提及的链接**：<a href="https://arxiv.org/abs/2210.14891">Broken Neural Scaling Laws</a>：我们提出了一种平滑断裂的幂律函数形式（我们称之为 Broken Neural Scaling Law (BNSL)），它能够准确地建模和外推深度神经网络的 Scaling 行为...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1268285771878629376)** (5 messages): 

> - `Gemma Scope`
> - `ICML Mech Int Workshop Recording` 


- **ICML Mech Int Workshop 录像**：一位成员询问 **ICML Mech Int Workshop** 的录像，另一位成员告知由于 **ICML 规则**，录像将在一个月后发布。
   - 据称这些规则可能是为了激励人们购买虚拟通行证。*另有人建议从会议与会者那里获取链接。*
- **Gemma Scope 的出色工作**：在简短的互动中，一位成员称赞了 **Gemma Scope** 的卓越进展。
   - 在赞扬 Gemma Scope 之后，紧接着是对 ICML Mech Int Workshop 录像的查询。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1268303952546631801)** (11 messages🔥): 

> - `lm-eval` Prompt 计数
> - `GPQA` 基准测试
> - `lm_eval harness` 行为
> - `lm_eval` 的 Issue 追踪
> - 解释 `lm_eval` 中的进度条


- **lm-eval 使用的 Prompt 数量多于基准测试中的数量**：一位用户注意到，即使在 Zeroshot 模式下运行 **lm-eval**，其使用的 Prompt 数量也是某些基准测试（如 **gpqa_main**）中现有数量的 4 倍，处理了 **1792** 个 Prompt 而不是 **448** 个。
- **GPQA 基准测试解释**：另一位用户解释说，**GPQA** 有四个选项，可能是在分别运行每个选项。
   - 另一位用户澄清说，选项之间的大小差异不应导致正好 4 倍的 Prompt 数量，并指出这种情况在 MMLU 等其他基准测试中也会发生。
- **GPQA eval harness 中的 Issue**：一位用户分享了他们的启动脚本以及 **lm_eval harness** 处理的 Prompt 数量超出预期的具体案例，提供了详细设置并询问相关的 Issue 引用。
- **进度条追踪选项**：一位用户澄清说，为了保持一致性，**lm-eval** 中的进度条显示的是 `num_choices * num_docs`，即使设置允许在没有多次 LM 调用的情况下进行单 Token 响应。

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1268283089474551919)** (61 条消息🔥🔥): 

> - `xAI 收购传闻`
> - `Black Forest Labs 成立公告`
> - `Gemini 1.5 Pro 发布`
> - `GitHub 推出 AI Models` 


- **Elon Musk 否认 xAI 收购 Character AI 的传闻**：有[传闻](https://x.com/nmasc_/status/1818788751528935468)称 xAI 可能会收购 Character AI 以测试和改进其 Grok 模型，但 [Elon Musk 否认](https://x.com/elonmusk/status/1818810438634946699)了这些说法，称这些报道是误导性信息。
   - 用户对这些传闻的可信度表示怀疑，并提到 Musk 此前曾有过先否认报道后又证实报道的先例。
- **原 Stable Diffusion 团队成立 Black Forest Labs**：原 **Stable Diffusion** 团队宣布成立 [Black Forest Labs](https://x.com/bfl_ml/status/1819003686011449788)，旨在为媒体开发先进的生成式深度学习模型。
   - 他们的目标是突破创意和效率的边界，其最新模型 **Flux** 已可在 fal 上进行测试。
- **Google 发布 Gemini 1.5 Pro**：[Google 的最新模型](https://x.com/tokumin/status/1819047737230528701?s=46) **Gemini 1.5 Pro** 已在 Google AI Studio 发布，并迅速以 1300 的 ELO 分数成为 LMSYS 上的顶级模型。
   - 该模型被誉为迄今为止最强、最智能的 Gemini 模型，展示了显著的进步。
- **GitHub 推出 AI Models**：GitHub 宣布[推出 GitHub Models](https://github.blog/news-insights/product-news/introducing-github-models/)，直接在其平台上为开发者提供行业领先的 AI 工具。
   - 这一举措旨在让开发者社区更容易接触到 AI，缩小程序员与 AI 工程师之间的差距。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://blog.fal.ai/flux-the-largest-open-sourced-text2img-model-now-available-on-fal/">Black Forest Labs 发布 Flux：文本生成图像模型的下一次飞跃</a>: Flux 是迄今为止最大的 SOTA 开源文本生成图像模型，由 Stable Diffusion 的原班人马 Black Forest Labs 开发，现已在 fal 上线。Flux 突破了创意...</li><li><a href="https://x.com/elonmusk/status/1818810438634946699?s=46">Elon Musk (@elonmusk) 的推文</a>: @nmasc_ @KalleyHuang @steph_palazzolo [误导性]信息再次袭来。xAI 并没有考虑收购 Character AI。</li><li><a href="https://x.com/tokumin/status/1819047737230528701?s=46">Simon (@tokumin) 的推文</a>: 我们刚刚将最新的 Gemini 1.5 Pro 推送到 http://aistudio.google.com。这是一个非常棒的模型，以 1300 的 ELO 分数位列 LMSYS 第一。整个团队的杰出工作...</li><li><a href="https://x.com/nmasc_/status/1818802320802824352?s=46">natasha mascarenhas (@nmasc_) 的推文</a>: 我听说除了 Character AI 之外，xAI 还在关注一些消费级 AI 公司作为潜在的收购目标。每天还能听到更多关于 Inflection/Ad...</li><li><a href="https://x.com/nmasc_/status/1818788751528935468?s=46">natasha mascarenhas (@nmasc_) 的推文</a>: 独家新闻：xAI 正在权衡收购 Character AI，因为它希望测试和改进其 Grok 模型并充实其人才队伍 https://www.theinformation.com/articles/musks-xai-considers-buying-...</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">GitHub Models 简介：在 GitHub 上构建的新一代 AI 工程师</a>: 我们正在通过 GitHub Models 助力 AI 工程师的崛起——直接在 GitHub 上为我们超过 1 亿的用户提供行业领先的大型和小型语言模型的力量。</li><li><a href="https://x.com/bfl_ml/status/1819003686011449788?t=IHBNW9bCDHQI9rosZVP2bw&s=19">Black Forest Labs (@bfl_ml) 的推文</a>: 我们很高兴宣布 Black Forest Labs 成立。我们的使命是开发和推进用于媒体的最先进生成式深度学习模型，并突破创意、效率的边界...</li><li><a href="https://x.com/bfl_ml/status/1819003686011449788?t=IHBNW9bCDHQ">Black Forest Labs (@bfl_ml) 的推文</a>: 我们很高兴宣布 Black Forest Labs 成立。我们的使命是开发和推进用于媒体的最先进生成式深度学习模型，并突破创意、效率的边界...</li><li><a href="https://x.com/elonmusk/status/1750995501560807465?s=46">Elon Musk (@elonmusk) 的推文</a>: xAI 没有在筹集资金，我也从未就此与任何人进行过交谈。引用 X Daily News (@xDaily) —— 新闻：据《金融时报》报道，@xAI 正在寻求高达 60 亿美元的投资...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1268446410899066920)** (32 条消息🔥): 

> - `Together AI 的批评`
> - `Suno 对阵音乐唱片公司`
> - `AI2 品牌重塑`
> - `OpenAI 与非营利机构的认知对比` 


- **Together AI 批评指出择优挑选（Cherry-Picked）的错误**：一位 AI 研究员批评 Together AI 存在择优挑选结果的行为，并就 LLM 评估中科学严谨性的必要性提出了观点，指出非平滑输出和有偏见的基准测试会扭曲现实世界的性能表现。
   - 他分享了详细的推文和外部资源，以强调 LLM 评估中的量化技术和透明方法论。
- **Suno 因版权问题与音乐唱片公司发生冲突**：Suno 对 RIAA 的回应强调了他们在面对音乐唱片公司起诉时的使命，这些唱片公司指控 Suno 使用受版权保护的内容进行训练。
   - 讨论反映了 Suno 承认使用了受版权保护的材料，以及导致诉讼的争议性谈判。
- **AI2 的品牌重塑引发褒贬不一的反应**：Allen AI 发布了其新品牌和网站，但并非所有反应都是正面的，一些人指出使用“闪烁”（sparkles）表情符号是 AI 品牌推广中常见的套路。
   - 这一变化引发了关于即使是非营利机构在品牌重塑过程中也会面临审查和复杂反应的讨论。
- **OpenAI 的非营利地位受到质疑**：在一次随意的交流中，成员们幽默地注意到 OpenAI 声称自己是非营利组织，这引发了对这种身份在实践中合法性的怀疑。
   - 这反映了更广泛的情绪，即即使是非营利组织也无法逃脱负面新闻和问责。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rachelmetz/status/1819086846913401266?s=46">Rachel Metz (@rachelmetz) 的推文</a>：看起来 @allen_ai 在其重新设计中借鉴了闪烁表情符号的剧本！看看我最近关于 AI 行业拥抱 ✨ 的文章，了解更多关于这个谦卑的闪烁符号如何跃升的信息...</li><li><a href="https://x.com/jiayq/status/1818786673695809793?s=46">贾扬清 (@jiayq) 的推文</a>：作为一名 AI 研究员和工程师，我完全尊重 Together 的成就，但也想指出许多择优挑选的错误。我确信这些并非故意，但 LLM 的评估是...</li><li><a href="https://x.com/mikeyshulman/status/1819010384134631794?s=46">Mikey (@MikeyShulman) 的推文</a>：我们今天将向 RIAA 成员提交回应。了解我们的使命以及其中的利害关系非常重要。你可以在 Suno 博客上阅读更多内容...</li><li><a href="https://x.com/allen_ai/status/1819077607897682156">Ai2 (@allen_ai) 的推文</a>：经过数月的幕后研究、访谈和心血投入，我们很高兴在今天推出 Ai2 的新品牌和网站。探索这一演变 🧵
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1268287925746012290)** (4 条消息): 

> - `动漫头像动态`
> - `文章发布时机`
> - `Llama 3.1 评分` 


- **动漫头像动态推荐文章**：一位成员提到他们的动漫 PFP（头像）动态开始推送一篇文章，称其为“神作”（banger），且发布时机无懈可击。
- **等待 Llama 3.1 评分的文章发布时机完美**：Natolambert 提到文章发布时机非常幸运，并透露他们一直在等待 **Llama 3.1** 的评分结果才将其发布。


  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1268591043172110498)** (28 messages🔥): 

> - `采访 Sebastian Raschka`
> - `知识蒸馏的定义`
> - `Apple AI 的进展`
> - `RLHF 中的拒绝采样 (Rejection sampling)`
> - `Open Instruct 更新` 


- **Sebastian Raschka 讨论开源 LLM 和 Llama 3.1**：[Sebastian Raschka 的采访](https://www.youtube.com/watch?v=-q79uzz1Wik) 涵盖了开源 LLM 的现状、**Llama 3.1** 以及 AI 教育。
   - 在采访中，讨论了关于类似 Alpaca 和 Self-Instruct 论文中的**蒸馏术语 (distillation verbiage)** 的担忧，强调了该领域存在的*命名冲突*。
- **对知识蒸馏术语的混淆**：成员们辩论了在使用合成数据 (synthetic data) 训练时使用的**蒸馏**术语，以及软目标 (soft-target) 和硬目标 (hard-target) 蒸馏的区别。
   - 这个问题在 *rejection sampling*（拒绝采样）等术语上更加突出，因为这些术语在特定 AI 语境之外很难通过 Google 搜索到。
- **Apple AI 集成引起轰动**：关于 [Apple 新 AI 功能](https://www.interconnects.ai/p/apple-intelligence) 的讨论表明，它们的集成可以更无缝地连接应用程序，使日常任务变得更容易。
   - Apple 的*多模型 AI 系统 Apple Intelligence* 被视为日常科技中的力量倍增器，尽管 **AI 实验室** 对其变革潜力仍持怀疑态度。
- **在 Open Instruct 中实现拒绝采样**：[拒绝采样 (Rejection sampling)](https://github.com/allenai/open-instruct/pull/205) 正在 **Open Instruct** 中实现，旨在简化训练流程。
   - 该方法可能会减少其他训练方法中发现的问题，提高模型训练的整体效率。
- **On-policy 偏好数据收集的挑战**：社区讨论了为单策略对齐数据集收集 **on-policy 偏好数据** 的成本和挑战。
   - 在 *An update on DPO vs PPO for LLM alignment* 视频中提到，拥有多样化的模型生成结果可以使 *Ultrafeedback* 更易于使用，但为了保持一致的对齐，专注于单策略可能是必要的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.interconnects.ai/p/apple-intelligence">AI for the rest of us</a>：当你跳出 AI 圈子时，Apple Intelligence 就变得非常有意义。此外，Apple 还分享了关于其语言模型“不同凡想 (thinking different)”的酷炫技术细节。</li><li><a href="https://www.youtube.com/watch?v=-q79uzz1Wik">采访 Sebastian Raschka，探讨开源 LLM 现状、Llama 3.1 和 AI 教育</a>：本周，我有幸与 Sebastian Raschka 进行了交谈。Sebastian 在开源语言模型生态系统和 AI 研究领域做了大量工作……</li><li><a href="https://github.com/allenai/open-instruct/pull/205">由 vwxyzjn 添加拒绝采样脚本 · Pull Request #205 · allenai/open-instruct</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1268287138563096750)** (56 messages🔥🔥): 

> - `Llama 3.1 评估与争议`
> - `AI SDR 融资`
> - `文本生成图像领域的新玩家：Black Forest Labs`
> - `LangGraph Studio 发布`
> - `使用 Meta MoMa 进行混合模态语言建模`

- **Llama 3.1 备受审视**：Llama 3.1 已经风靡全球，但由于不同的推理提供商使用不同的实现方式导致质量差异，它正面临批评 ([Together AI 博客](https://www.together.ai/blog/llama-31-quality))。
   - AI 社区的知名人士指出 Together AI 的评估中存在不准确性和潜在的幻觉，并声称其结果是经过精心挑选的 (cherry-picked)，强调了透明的方法论和严格的基于数据的测试的重要性 ([讨论线程](https://x.com/dzhulgakov/status/1818753731573551516))。
- **Sybill 为 AI SDR 筹集 1100 万美元**：Sybill 宣布获得 **1100 万美元的 A 轮融资**，用于为每位销售代表打造个人助手，由 **Greystone Ventures** 和其他知名风投领投 ([阅读更多](https://x.com/asnani04/status/1818642568349204896))。
   - AI 驱动的销售工具市场正在升温，Sybill 克隆销售人员声音以起草相关后续跟进的功能被认为非常切中痛点。
- **Black Forest Labs 进军文本生成图像领域**：Black Forest Labs 推出了名为 **FLUX.1** 的全新 SOTA 文本生成图像模型套件，其中包括一个 **12B 参数模型**，可在 Huggingface 上通过非商业和开放许可证获取 ([公告](https://x.com/iScienceLuvr/status/1819007823339999516) 和 [模型权重](https://huggingface.co/black-forest-labs))。
   - 该团队由前 Stable Diffusion 成员组成，其 **pro model** 已经可以在 Replicate 上进行测试。
- **LangGraph Studio：全新的 Agent IDE**：LangChain 宣布推出 **LangGraph Studio**，这是一个专门用于 **Agent 应用** 的 IDE，能够实现 LLM 工作流更好的可视化、交互和调试 ([公告](https://x.com/LangChainAI/status/1819052975295270949))。
   - 该工具与 **LangSmith** 集成以进行协作，旨在提高开发 LLM 应用的效率和易用性。
- **Meta 推出用于混合模态语言建模的 MoMa**：Meta 宣布了 **MoMa**，这是一种用于混合模态语言建模的新型稀疏早期融合架构，提高了预训练效率 ([论文](https://arxiv.org/pdf/2407.21770) 和 [公告](https://x.com/victorialinml/status/1819037433251721304?s=46))。
   - MoMa 采用了带有模态特定专家组的 **Mixture-of-Expert (MoE)** 框架，能够高效处理交错的混合模态 Token 序列。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/allen_ai/status/1819077607897682156">来自 Ai2 (@allen_ai) 的推文</a>：经过数月的幕后研究、访谈和心血投入，我们今天很高兴推出 Ai2 的新品牌和网站。探索这一进化过程 🧵</li><li><a href="https://x.com/nisten/status/1818529201231688139">来自 nisten (@nisten) 的推文</a>：修改了 bitnet 用于微调，最终得到了一个 74mb 的文件。仅在 1 个 CPU 核心上就能以每秒 198 个 token 的速度流畅对话。简直是魔法。稍后将通过 @skunkworks_ai 开源，基础版本见：https://huggi...</li><li><a href="https://x.com/elonmusk/status/1818810438634946699?s=46">来自 Elon Musk (@elonmusk) 的推文</a>：@nmasc_ @KalleyHuang @steph_palazzolo [误导] 信息再次袭来。xAI 并没有考虑收购 Character AI。</li><li><a href="https://x.com/TheNoahHein/status/1819098232636481711">来自 Noah Hein (@TheNoahHein) 的推文</a>：在 @replicate 上尝试 @bfl_ml 的 flux-dev 模型！这里是它的输出列表、提示词，以及与 MJ 相同提示词的并排对比！Flux 在左边，MJ 在...</li><li><a href="https://x.com/Tim_Dettmers/status/1818282778057941042">来自 Tim Dettmers (@Tim_Dettmers) 的推文</a>：在求职市场 7 个月后，我很高兴地宣布：- 我加入了 @allen_ai - 2025 年秋季起担任 @CarnegieMellon 教授 - 新的 bitsandbytes 维护者 @Titus_vK。我的主要重点将是加强...</li><li><a href="https://x.com/llama_index/status/1819048068798616058">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：今天我们很高兴推出 @llama_index workflows —— 一种构建 multi-agent 应用的新型事件驱动方式。将每个 agent 建模为一个订阅并发送事件的组件；你可以...</li><li><a href="https://x.com/LangChainAI/status/1819052975295270949">来自 LangChain (@LangChainAI) 的推文</a>：🚀 发布 LangGraph Studio：首个 agent IDE。LangGraph Studio 通过提供专门的 agent IDE，为开发 LLM 应用提供了一种新方式，支持可视化、交互和调试...</li><li><a href="https://x.com/al">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/jiayq/status/1818786673695809793">来自 Yangqing Jia (@jiayq) 的推文</a>：作为一名 AI 研究员和工程师，我完全尊重 Together 的成就，但也想指出许多挑选出的错误（cherrypicked errors）。我确信这些并非故意，但 LLM 的评估是...</li><li><a href="https://x.com/dzhulgakov/status/1818753736359178414">来自 Dmytro Dzhulgakov (@dzhulgakov) 的推文</a>：示例：AI 研究员提问“什么是 group query attention？” 声明：事实正确且详细的回答。现实：回答暗示 GQA 是某种形式的 sequence-sparse attention。然而...</li><li><a href="https://x.com/basetenco/status/1819048091451859238">来自 Baseten (@basetenco) 的推文</a>：我们很高兴为 TensorRT-LLM 推出全新的 Engine Builder！🎉 同样的 @nvidia TensorRT-LLM 卓越性能——减少 90% 的工作量。查看我们的发布文章了解更多：https://www.baseten.c...</li><li><a href="https://x.com/dzhulgakov/status/1818753731573551516">来自 Dmytro Dzhulgakov (@dzhulgakov) 的推文</a>：这是你吗？我们在 Together playground 上运行了你们的展示示例 3 次，每次它都会无限循环或回答错误。好奇这是如何通过你们质量测试的所有 5 个步骤的...</li><li><a href="https://x.com/togethercompute/status/1818706177238397155">来自 Together AI (@togethercompute) 的推文</a>：最近关于不同推理提供商使用 Meta 的 Llama 3.1 模型不同实现时的质量差异，引发了大量讨论。在下面的博客文章中，我们...</li><li><a href="https://x.com/ContextualAI/status/1819032988933623943">来自 Contextual AI (@ContextualAI) 的推文</a>：我们今天很高兴地分享，我们已经筹集了 8000 万美元的 A 轮融资，以加速我们通过 AI 改变世界运作方式的使命。更多信息请见我们的博客文章：https://contextual.ai/news/an...</li><li><a href="https://x.com/romainhuet/status/1814054938986885550">来自 Romain Huet (@romainhuet) 的推文</a>：@triviatroy @OpenAI 每张图片的美元价格对于 GPT-4o 和 GPT-4o mini 是一样的。为了保持这一点，GPT-4o mini 每张图片使用更多的 token。感谢您的观察！</li><li><a href="https://x.com/asnani04/status/1818642568349204896">来自 Nishit Asnani (@asnani04) 的推文</a>：🚀 重大新闻！Sybill 在 A 轮融资中筹集了 1100 万美元，由 @greycroftvc 领投，@neotribevc、Powerhouse VC 和 Uncorrelated VC 参投。我们正在为每个人打造个人助理...</li><li><a href="https://x.com/victorialinml/status/1819037433251721304?s=46">来自 Victoria X Lin (@VictoriaLinML) 的推文</a>：1/n 介绍 MoMa 🖼，我们用于混合模态（mixed-modal）语言建模的新型稀疏早期融合（sparse early-fusion）架构，显著提升了预训练效率 🚀 (https://arxiv.org/pdf/2407.21770)。MoMa 采用...</li><li><a

<ul><li><a href="https://x.com/StabilityAI/status/1819025550062850451">来自 Stability AI (@StabilityAI) 的推文</a>：我们很高兴推出 Stable Fast 3D，这是 Stability AI 在 3D 资产生成技术方面的最新突破。这一创新模型仅需几秒钟即可将单张输入图像转换为详细的 3D 资产...</li><li><a href="https://x.com/character_ai/status/1819138734253920369?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Character.AI (@character_ai) 的推文</a>：很高兴分享我们正在开源创新的提示词设计方法！在我们的最新博客文章中了解 Prompt Poet 如何彻底改变我们构建 AI 交互的方式：https://r...</li><li><a href="https://x.com/robrombach/status/1819012132064669739">来自 Robin Rombach (@robrombach) 的推文</a>：🔥 我非常激动地宣布 Black Forest Labs 成立。我们的使命是推进图像和视频领域最先进（SOTA）、高质量的生成式深度学习模型，并且...</li><li><a href="https://x.com/lmsysorg/status/1819048821294547441">来自 lmsys.org (@lmsysorg) 的推文</a>：来自 Chatbot Arena 的激动人心的消息！@GoogleDeepMind 的新 Gemini 1.5 Pro (Experimental 0801) 过去一周已在 Arena 中进行测试，收集了超过 1.2 万条社区投票。这是 Google 第一次...</li><li><a href="https://x.com/iScienceLuvr/status/1819007823339999516">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：Black Forest Labs 发布了名为 FLUX.1 的全新 SOTA 文本转图像模型系列。最强模型 FLUX.1[pro] 通过 API 提供，FLUX.1[dev] 是 12B 参数模型，采用非商业许可，FLUX.1[schnell]...</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">推出 GitHub Models：在 GitHub 上构建的新一代 AI 工程师</a>：我们通过 GitHub Models 助力 AI 工程师的崛起——直接在 GitHub 上为我们超过 1 亿的用户带来行业领先的大型和小型语言模型的能力。</li><li><a href="https://www.together.ai/blog/llama-31-quality">Llama 3.1：相同的模型，不同的结果。一个百分点的影响。</a>：未找到描述</li><li><a href="https://x.com/GriffinAdams92/status/1819072387469516884">来自 Griffin Adams (@GriffinAdams92) 的推文</a>：与 @answerdotai 共同发布 Cold Compress 1.0。一个用于使用和创建 KV cache 压缩方法的可黑客攻击工具包。基于 @cHHillee 团队的 GPT-Fast 构建，支持 torch.compilable，轻量级...</li><li><a href="https://youtu.be/qP3rXJc_L5Y?si=z52-nyB0Ov0lUCkg">自主合成对话（以及其他近期合成数据）</a>：涵盖我们最近启动的一个合成数据项目的演讲。详情见下方：https://arxiv.org/abs/2407.18421 幻灯片：https://docs.google.com/presentat...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/jHnSGxfHRj">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://replicate.com/black-forest-labs/flux-pro">black-forest-labs/flux-pro – 在 Replicate 上通过 API 运行</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1268301305160667199)** (3 条消息): 

> - `BedrockConverse 的异步功能`
> - `@Ernestzyj 的 LongRAG 论文`
> - `@llama_index 工作流 (workflows)` 


- **BedrockConverse 现已支持异步功能**：已实现 **BedrockConverse** LLM 的 [异步方法 (Async methods)](https://t.co/rn3sAKG05N)，解决了问题 [#10714](https://github.com/run-llama/llama_index/issues/10714) 和 [#14004](https://github.com/run-llama/llama_index/issues/14004)。
   - *团队非常感谢这一贡献，因为它提升了用户体验。*
- **LongRAG 论文简化了长上下文 LLM**：@Ernestzyj 的 **LongRAG** 论文提出通过索引和检索更大的文档块，以更好地利用长上下文 LLM。
   - *这种方法旨在减轻检索器的任务，增强检索增强生成 (RAG) 过程。*
- **@llama_index 推出工作流 (workflows)**：@llama_index [工作流](https://t.co/Ebme9eRvMb) 支持事件驱动的多 Agent 应用，允许 Agent 订阅和发布事件。
   - 这种新方法提供了一种**可读且 Pythonic** 的方式来构建复杂的编排。



**提到的链接**：<a href="https://t.co/rn3sAKG05N">feat: ✨ 在 `BedrockConverse` 中实现异步功能，由 AndreCNF 提交 · Pull Request #14326 · run-llama/llama_index</a>：描述：为 BedrockConverse LLM 实现异步方法。修复了 #10714，修复了 #14004。新包？我是否填写了 pyproject.toml 中的 tool.llamahub 部分并提供了详细的 README.m...

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1268282745189302382)** (47 条消息🔥): 

> - `RagApp 的替代方案`
> - `使用 LlamaParse 生成图像`
> - `LlamaIndex 的稳定版本`
> - `处理 ReAct 中的 Agent 错误`
> - `LlamaIndex 中的配置` 


- **寻找 RagApp 的替代方案**：一位用户询问了 RagApp 的替代方案，并讨论了 `create-llama` 的实用性，尽管在使用 Poetry 安装时遇到了一些问题。
- **使用 LlamaParse 生成图像**：用户讨论了使用 LlamaParse 生成图像的方法，参考了 [GitHub 示例](https://github.com/run-llama/llama_parse/blob/main/examples/demo_json.ipynb) 和其他资源。
- **识别 LlamaIndex 的稳定版本**：一位用户询问如何识别 LlamaIndex 的“稳定”版本，会议明确了通过 pip 安装可确保获得最新的稳定版本。
   - 后续评论强调，“稳定”版本通常指 PyPI 上的最新发布版本。
- **处理 ReAct Agent 中的错误**：一位用户探索了如何让 ReAct Agent 在不调用工具的情况下运行，并讨论了如 `SimpleChatEngine` 或更优雅地处理 Agent 错误等替代方法。
   - 建议包括使用 `llm.chat(chat_messages)` 进行更简单的设置，以及探索 function calling agent 以获得更好的工具处理能力。
- **在 LlamaIndex 中配置参数**：讨论了在 `PromptHelper` 移除后，如何在 LlamaIndex v10.x 中设置 `max_input_size` 和 chunk overlap 等参数。
   - 建议的替代方案包括直接将配置传递给 node parsers 或使用 response synthesizers。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/multimodal_rag_slide_deck.ipynb">llama_parse/examples/multimodal/multimodal_rag_slide_deck.ipynb at main · run-llama/llama_parse</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/demo_json.ipynb">llama_parse/examples/demo_json.ipynb at main · run-llama/llama_parse</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/CassandraIndexDemo/">Cassandra Vector Store - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1268541355932651553)** (1 条消息): 

> - `DSPy`
> - `Prompt Optimizing`
> - `Prompt Rewriting`
> - `LlamaIndex` 


- **比较 DSPy 提示词优化与 LlamaIndex**：一位成员询问了其他人使用 **DSPy** 的经验，并征求关于其 **prompt optimizing** 与 **LlamaIndex** 中 **prompt rewriting** 功能对比的意见。
- **DSPy Prompt Optimization 对比 LlamaIndex**：表达了对比较 **DSPy** 和 **LlamaIndex** 之间 **prompt optimization** 和 **prompt rewriting** 的兴趣。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1268508658157883403)** (16 messages🔥): 

> - `Embedding 内容结构`
> - `PDF 中的表格和复选框检测`
> - `AI Hackathon Series Tour`
> - `Ivan 的游戏玩家身份`
> - `把牛当宠物` 


- **关于利用内容结构进行 Embedding 的讨论**：讨论了换行符、分页符和特殊符号对 Embedding 性能的影响，**Nils Reimers** 确认这些元素在英语和多语言模型中会被自动移除。
   - *无需为 Embedding 模型对文本进行大量的预处理*是核心结论，模型足够鲁棒，可以处理噪声数据。
- **从 PDF 中检测并提取表格和复选框数据**：一名成员寻求能够从**不可读的 PDF** 中检测表格和复选框并提取为文本或 docx 格式的模型建议。
   - 建议强调了使用 **unstructured.io** 将 PDF 数据转换为 JSON 格式的有效性，社区内一个类似的进行中项目也证明了这一点。
- **参加 Google 的 AI Hackathon Series Tour**：**AI Hackathon Series Tour** 邀请报名参加在 Google 举办的活动，涵盖创新的 AI 项目和为期 **3 天**的竞赛。
   - 该活动提供了一个创意和竞争的平台，并以 **PAI Palooza** 结束，展示来自主办城市的顶级 AI 初创公司和项目。
- **Ivan 的游戏背景曝光**：一篇 LinkedIn [文章](https://www.linkedin.com/pulse/from-gamer-ai-unicorn-co-founder-conversation-coheres-f2tte) 揭示了 **Ivan** 曾是一名游戏玩家，这让一些社区成员感到惊讶。
   - **Karthik_99_** 对发现 Ivan 从游戏玩家转变为 AI 联合创始人的历程表示惊讶。
- **照顾牛**：关于养牛的轻松评论引发了“**它们需要很多工作**”的观察，以此回应一位成员的羡慕。



**提到的链接**：<a href="https://lu.ma/2svuyacm">Techstars StartUp Weekend - PAI Palooza &amp; GDG Build with AI—Mountain View · Luma</a>：这次 AI Hackathon Series Tour 是一场开创性的、跨越全美多城市的盛会，汇聚了人工智能领域最聪明的头脑……

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1268336626392629248)** (17 messages🔥): 

> - `为阿拉伯语方言训练 LLM`
> - `加入 Cohere 研究社区`
> - `训练 LLM 以输出 JSON` 


- **为阿拉伯语方言训练 LLM**：一名成员询问像 **Aya** 这样的模型如何在训练 Prompt 中没有明确方言信息的情况下，生成流利的各种阿拉伯语方言回复。
   - 他们对使用英语 Prompt 要求埃及方言却能正确生成该形式文本感到惊讶。
- **加入 Cohere 研究社区**：一名成员报告了加入 Cohere 研究社区时遇到的问题，结果变成了订阅新闻邮件。
   - 回复中提到了人工审核流程并对延迟表示歉意，要求该成员私信其电子邮箱以获取状态更新。
- **训练 LLM 以输出 JSON**：一名成员询问关于训练 LLM 将自由格式的搜索查询转换为用于 Apache Solr 输入的结构化 JSON。
   - 建议他们可以手动标注数据、寻找已标注数据或合成生成数据，并查看 [Cohere 文档](https://docs.cohere.com/docs/structured-outputs-json) 以了解如何生成结构化输出。



**提到的链接**：<a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON)</a>：未找到描述

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1268565050465980507)** (15 messages🔥): 

> - `八月 Office Hours 活动`
> - `乌克兰语/俄语语言支持退化`
> - `Citation_quality 设置`
> - `Cohere Cloud 的速度优化` 


- **八月 Office Hours 活动邀请**：一名成员邀请其他人参加 [八月 Office Hours 活动](https://discord.com/events/954421988141711382/1265012161965461625/1275137202585600000) 进行聚会。
   - 他们通过暗示这将是一个有趣的聚会来鼓励大家参与。
- **乌克兰语/俄语语言支持退化**：一名用户报告在 Cohere Cloud 上遇到 **乌克兰语/俄语语言支持** 退化，导致出现乱码。
   - 该问题与 **citation_quality** 设置有关，将设置从 **fast** 切换到 **accurate** 解决了该问题，尽管这影响了响应速度。


  

---

### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1268504819094650901)** (3 messages): 

> - `devcontainer 问题`
> - `pydantic 校验错误`
> - `仓库更新`
> - `团队响应` 


- **校验错误阻碍仓库设置**：一名成员报告在 devcontainer 中运行最新版本的仓库时遇到问题，遇到了多个与 `Settings` 类相关的 **pydantic 校验错误**。
   - 记录了 *6 个校验错误*，特别是缺失了 **auth.enabled_auth** 和 **auth.google_oauth** 等字段，导致 `make setup` 失败。
- **团队迅速处理 devcontainer 问题**：另一名成员快速确认了该问题，并承诺团队将调查并解决这些错误。
   - 随后不久发布了更新，确认团队已经在着手修复。



**相关链接**: <a href="https://errors.pydantic.dev/2.8/v/missing">Redirecting...</a>: 未找到描述

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1268322459426361446)** (45 messages🔥): 

> - `LangChain 中的 Pydantic 类型错误`
> - `在 LangChain 中执行工具`
> - `LangSmith API 密钥问题`
> - `LangChain 与部署`
> - `LangChain 文档与资源` 


- **Pydantic 版本冲突导致错误**：一名成员在已安装 Pydantic v2 的情况下遇到了 `pydantic.v1.error_wrappers.ValidationError`，导致在 LangChain 执行期间出现预期类型不匹配和校验错误。
- **LangChain 中的工具执行问题**：LangChain 工具在执行 `execute_tools` 节点时遇到问题，尽管事先对输入进行了正确的 Pydantic 校验，但仍因输入类型不匹配和校验错误而导致失败。
- **LangSmith API 密钥设置困扰**：一名用户在尝试使用 LangSmith 部署 LLM 时遇到了 `403 Client Error: Forbidden`，怀疑是与 API 密钥配置相关的问题。
- **LangChain 资源建议与替代方案**：成员们讨论了学习 LangChain 的不同来源以及替代的 LLM 推理服务，推荐将 OpenAI 和 TogetherAI 与 LangChain 的 prompt 类结合使用，以获得免费或负担得起的体验。
- **LangChain 文档与错误处理**：用户被引导至 LangChain GitHub 上的示例资源，以排查各种问题并避免在工具使用和 API 集成中的常见错误。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/stabilityai/stable-fast-3d">Stable Fast 3D - stabilityai 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/llm_chain/#using-language-models">使用 LCEL 构建简单的 LLM 应用 | 🦜️🔗 Langchain</a>: 在此快速入门中，我们将向您展示如何构建一个简单的 LLM 应用程序</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb">langgraph/examples/plan-and-execute/plan-and-execute.ipynb at main · langchain-ai/langgraph</a>: 将具有韧性的语言 Agent 构建为图。通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/tool-calling.ipynb">langgraph/examples/tool-calling.ipynb at main · langchain-ai/langgraph</a>: 将具有韧性的语言 Agent 构建为图。通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。</li><li><a href="https://v02.api.js.langchain.com/index.html">LangChain.js - v0.2.12</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/tool-calling-errors.ipynb">langgraph/examples/tool-calling-errors.ipynb at main · langchain-ai/langgraph</a>: 将具有韧性的语言 Agent 构建为图。通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1268560571238973531)** (2 条消息): 

> - `FastAPI LangChain 应用中的流式支持`
> - `在 langserve v2 中使用 /stream_events 端点` 


- **为 FastAPI LangChain 应用添加流式支持**：一位用户提出了一个设计方案，旨在为带有 LangChain 的 FastAPI 应用添加异步流式支持，重点是使用 Redis 作为消息代理来实现实时 token 生成。
   - 该设计包括保留现有的同步端点，添加新的流式端点，并更新 LangChain Agent 以将数据块（chunks）和完整响应发布到 Redis。
- **在 langserve v2 中使用 /stream_events 端点**：一位用户询问如何在 langserve v2 版本中使用 `/stream_events` 端点，并提到他们找不到任何相关文档。
   - 他们表示难以找到相关信息，并寻求社区的帮助。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1268412970208071771)** (2 条消息): 

> - `LangGraph 设计模式`
> - `高级研究助手和搜索引擎`
> - `GPT-4o`
> - `Claude 3 Opus`
> - `Llama 3.1` 


- **面向用户应用的 LangGraph 设计模式**：一位成员分享了一个 LangGraph 设计模式，可轻松集成到网页聊天或 Telegram/Whatsapp 机器人等面向用户的应用中，并在 [GitHub](https://github.com/TonySimonovsky/ai-champ-design-patterns/blob/main/ai-agents/LangGraph-multi-agent-user-facing.ipynb) 上提供了详细示例。
   - *“这是一个 LangGraph 设计模式，可以轻松集成到支持流式传输的面向用户应用中。”*
- **Rubik's AI Pro 提供高级模型 Beta 测试**：一位成员邀请其他人参与高级研究助手和搜索引擎的 Beta 测试，通过 [Rubik's AI](https://rubiks.ai/) 提供 2 个月的免费高级版，其中包括 **Claude 3 Opus**、**GPT-4o**、**Gemini 1.5 Pro** 等模型。
   - *“使用促销代码 `RUBIX` 即可获得 2 个月的免费高级版，以测试新功能和专家模型。”*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述</li><li><a href="https://github.com/TonySimonovsky/ai-champ-design-patterns/blob/main/ai-agents/LangGraph-multi-agent-user-facing.ipynb">ai-champ-design-patterns/ai-agents/LangGraph-multi-agent-user-facing.ipynb at main · TonySimonovsky/ai-champ-design-patterns</a>：通过在 GitHub 上创建一个账号来为 TonySimonovsky/ai-champ-design-patterns 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1268578388851822695)** (1 条消息): 

> - `Moye Launcher`
> - `数字排毒工具` 


- **Moye Launcher 推广数字排毒**：Moye Launcher 是一款极简 Android 启动器，内置 AI 驱动的数字排毒（Digital detox）工具，旨在减少过度的屏幕使用时间。它取消了应用抽屉，使应用不那么容易被访问，从而减少冲动性的应用使用。
   - 该启动器旨在解决导致低效屏幕时间的三大主要原因，例如因无聊和缺乏自律而导致的习惯性点击，通过移除易于访问的应用图标并提供使用反馈来实现。
- **数字排毒工具详解**：Moye Launcher 使用 AI 工具帮助用户保持自律并避免不必要的应用使用，提供提醒并跟踪使用情况。
   - 这些功能针对低效屏幕时间的主要原因：习惯性点击应用、缺乏“监督者”以及忘记最初打开应用的目的。



**提到的链接**：<a href="https://play.google.com/store/apps/details?id=in.noxchat.moyemoyelauncher">Moye Launcher: Digital Detox - Google Play 应用</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1268282400451072021)** (39 条消息🔥): 

> - `Lobe 界面`
> - `Librechat 功能`
> - `Big-agi 特性`
> - `Msty 与 Obsidian 的工具集成`
> - `Llama 405B Instruct 提供商` 


- **Big-agi 通过 BEAM 扩展模型能力**：Big-agi 引入了一个“角色创建器 (persona creator)”，允许用户从 YouTube 视频或文本中生成 prompt，并通过 BEAM 功能同时调用 2/4/8 个模型并合并它们的响应。
   - 然而，它缺乏服务器端保存和便捷的同步功能。
- **Msty 集成 Obsidian 和网站**：Msty 提供了与 Obsidian 的出色集成以及网站访问功能，但据报道其参数设置很容易被重置（遗忘）。
   - 尽管存在一些细微的打磨问题，许多用户仍觉得它很有吸引力并考虑切换到该工具。
- **Llama 405B Instruct 提供商与量化**：OpenRouter 上目前没有 Llama 405B 的 FP16 提供商，而 Meta 推荐的 FP8 量化比 FP16 运行效率更高。
   - SambaNova Systems 以 bf16 运行，但受限于 4k 上下文长度，且以 bf16 托管的计算成本非常昂贵。
- **OpenRouter 的 API 集成处于 Beta 阶段**：寻求通过 API 集成来处理速率限制 (rate limits) 并整合 OpenAI 和 Claude API 的用户，建议发送邮件给支持团队以加入 Beta 等候名单。
   - 详细请求可发送至 support@openrouter.ai 以获取帮助。
- **OpenRouter 网站偶尔出现区域性问题**：OpenRouter 网站偶尔会遇到区域性连接问题，但总体上保持正常运行。
   - 用户可以通过 [OpenRouter 状态页面](https://status.openrouter.ai/) 查看实时运行信息的更新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sambanova.ai/">SambaNova Systems | 革新 AI 工作负载</a>：利用 SambaNova 的企业级生成式 AI 平台释放业务的 AI 力量。了解如何实现 10 倍的成本降低和无与伦比的安全性。</li><li><a href="https://openrouter.ai/privacy#_4_-user-rights-and-choices">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>：OpenRouter 故障历史</li><li><a href="https://github.com/oobabooga/text-generation-webui/pull/5677">DRY：一种现代重复惩罚机制，可可靠地防止 p-e-w 导致的循环 · Pull Request #5677 · oobabooga/text-generation-webui</a>：循环是一种不理想的行为，模型会逐字重复输入中先前出现的短语。它影响大多数模型，并因使用截断采样器而加剧....
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1268338710919909467)** (23 条消息🔥): 

> - `Open Interpreter 响应延迟`
> - `Groq Profile 贡献`
> - `无障碍圆桌会议公告`
> - `家庭派对活动`
> - `社区建设重点` 


- **Open Interpreter 响应延迟**：成员们对 Open Interpreter 的 Ben Steinher 延迟回复表示关注；他原定于 7 月 11 日“下周初”做出回应。
- **Groq Profile 贡献受到赞赏**：一名成员宣布了一个关于 Groq profile 的新 PR，称其为贡献 **Open Interpreter** 项目的绝佳方式。
   - *嘿，我们这儿的人都爱 Groq 😁*
- **8 月 22 日的无障碍圆桌会议**：[无障碍圆桌会议 (Accessibility Roundtable)](https://discord.gg/open-interpreter-1146610656779440188?event=1268579948248170663) 宣布将于 PST 时间 8 月 22 日中午举行，邀请成员参与关于无障碍性的讨论。
- **对家庭派对活动的兴奋**：成员们提醒其他人 4 小时后将举行家庭派对 (House Party) 活动，并提供了[活动链接](https://discord.gg/zMwXfHwz?event=1267524800163610815)。
   - 虽然对活动开始时间似乎存在一些困惑，但问题已解决，参与者加入了正确的语音频道。
- **以 AI 促进社区建设**：一名成员分享了他们的 AI 项目重点在于社区建设，特别是培养**后院烧烤邻里友谊**。
   - “*这太重要了！！而且是没有业主协会 (HOA) 的社区街区派对，哈哈*”


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/O_Q1hoEhfk4">Friend 揭晓预告片</a>：并非虚构。现在前往 friend.com 预订。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1376">由 MikeBirdTech 添加的 Groq profile 和 flag · Pull Request #1376 · OpenInterpreter/open-interpreter</a>：通过默认的 groq.py 文件添加了 Open Interpreter groq profile 支持，更新了 start_terminal_interface.py 中 CLI 快捷方式的解析器以接受 --groq 标志来应用该 profile。描述更改内容 ...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1268343270690521199)** (8 messages🔥): 

> - `模型选择问题`
> - `01 工作流与调度`
> - `iKKO ActiveBuds`
> - `01 发货状态`
> - `带摄像头的耳机` 


- **关于模型选择和 API Key 使用的困惑**：一位成员对选择模型字符串以及在运行 '01 --local' 时为何需要 OpenAI API Key 表示困惑。
   - 他们提到自己缺乏关于这些基础概念的知识。
- **01 的工作流和调度功能？**：一位成员询问 OpenInterpreter (OI) 是否可以保存工作流并设置任务调度。
   - 在给定的消息中，该问题尚未得到解答。
- **在 iKKO ActiveBuds 上运行 01 会很酷**：成员们讨论了将 01 集成到 [iKKO ActiveBuds](https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds) 上的潜力，该产品拥有 AI 智能系统、AMOLED 触摸屏和高分辨率音质等特性。
   - 这一想法被认为是可行且令人兴奋的，有助于改善人机交互 (HCI)。
- **迫切需要 01 的发货信息**：一位成员询问 01 的发货状态，因为现在已经是 8 月了。
   - [回复已链接](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191)，但对话中未提供更多细节。
- **对带摄像头耳机的渴望**：成员们表达了对配备摄像头的耳机的渴望，以便在与 LLM 对话时捕捉上下文。
   - 该想法包括一个按下/点击功能来激活摄像头，从而增强人机交互能力。



**提到的链接**：<a href="https://www.ikkoaudio.com/collections/tws-earbuds/products/activebuds">ActiveBuds：配备 ViVid 触摸屏的 AI 智能耳机 | iKKO Audio</a>：由 ChatGPT-4o 提供支持的 AI 语音助手。支持高比特率蓝牙配对，可在耳机、扬声器、智能手机之间实现高分辨率无线音频。支持 45 种语言翻译。为 ChatGPT 提供的便携备忘录等...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1268334732559515931)** (18 messages🔥): 

> - `Mojo 线程`
> - `Max 与 Mojo 打包`
> - `层级图讨论`
> - `存在量词` 


- **Mojo 缺乏显式的线程支持**：一位成员询问 Mojo 是否支持线程，另一位成员确认 **Mojo 目前未向用户开放线程支持**。
   - 然而，在编译版本中，**调用 fork()** 并以此方式获取线程是被允许的。
- **宣布 MAX 和 Mojo 打包变更**：官方宣布了关于 **MAX 和 Mojo 打包的变更**，从 `modular` CLI 的 0.9 版本开始，下载 MAX 和 Mojo 将不再需要身份验证。
   - [进一步的变更](https://docs.modular.com/max/faq#why-bundle-mojo-with-max)包括将 Mojo nightly 软件包与 MAX 合并，并过渡到新的 `magic` CLI，以便更轻松地集成到 Conda 生态系统中。
- **层级图讨论引起困惑**：随后展开了关于层级图（tier chart）的讨论，成员们质疑其表现形式，并指出它没有反映出“抽象级别”。
   - 为了简化，有人建议用一个火表情符号替换整个冰山图。



**提到的链接**：<a href="https://docs.modular.com/max/faq#why-bundle-mojo-with-max),">MAX FAQ | Modular 文档</a>：关于 MAX Engine 预期问题的解答。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1268495275862003754)** (4 messages): 

> - `CrazyString gist 更新`
> - `基于 Unicode 的索引` 


- **CrazyString Gist 添加了 Unicode 支持**：[CrazyString gist](https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae) 现在包含对基于 Unicode 索引的支持，以及短字符串优化和完整的 UTF-8 兼容性。
   - 更新中描述了 *具有短字符串优化的 Mojo String* 以及潜在的完整 UTF-8 支持。
- **数学和计算作为通用语言**：一位成员评论道：“数学是通用语言，计算是通用行动”。



**提到的链接**：<a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae">具有短字符串优化和潜在完整 UTF-8 支持的 Mojo String</a>：具有短字符串优化和潜在完整 UTF-8 支持的 Mojo String - crazy_string.mojo

  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1268530304759365746)** (5 messages): 

> - `Installing max on Mac M1 Max`
> - `Mojo compatibility with Python` 


- **在 Mac M1 Max 上安装 max 的问题**：一名成员报告在尝试于 Mac M1 Max 设备上安装 max 时遇到问题。
   - 另一名成员建议[参考此 Python 安装修复方案](https://modul.ar/fix-python)来尝试解决该问题。
- **Mojo 旨在成为 Python 的超集**：[Mojo](https://modul.ar/fix-python) 设计为与现有的 Python 程序兼容，允许程序员立即使用它，同时利用庞大的 Python 软件包生态系统。
   - Mojo 尚处于早期开发阶段，许多 Python 特性尚未实现，但它允许导入 Python 模块、调用 Python 函数以及与 Python 对象进行交互。



**提到的链接**：<a href="https://modul.ar/fix-python">Python integration | Modular Docs</a>：同时使用 Python 和 Mojo。

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1268292373130117131)** (8 messages🔥): 

> - `Automated Training Run Termination`
> - `Early Stopping in Axolotl`
> - `Manual Run Termination`
> - `Output Mask Field Proposal` 


- **Axolotl 实现了 Early Stopping**：一名成员询问 Axolotl 是否具有在 **loss 渐近收敛**或**验证集 loss 增加**时自动终止训练运行的功能。
   - 另一名成员确认 **Axolotl 支持 Early Stopping** 来实现此目的。
- **手动终止并保存当前的 LoRA Adapter**：一名成员询问是否可以手动终止运行，同时保存最近训练的 **LoRA adapter**，而不是取消整个运行。
   - 社区对此请求尚未有后续跟进。
- **SharedGPT 中的 Output Mask 字段提议**：一名成员提议在 **SharedGPT** 的每一轮对话中添加一个 **"output mask" 字段**，以便对输出进行选择性训练。
   - 他们解释说，这将允许 AI 在被掩码的字段中犯错并随后从中学习。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1268362706038034604)** (5 messages): 

> - `Chat templates documentation`
> - `Preprocessing step issue` 


- **需要新聊天模板的文档**：一名成员提到需要**新聊天模板的文档**，并表示很难理解它们的工作原理以及如何提取消息的特定部分。
   - *另一名成员指出，他们已经为自己编写了一些文档，并会尝试将其添加到官方文档中。*
- **旧版本预处理步骤中的 Bug**：一名成员请求一个仅在 main 分支的旧版本上运行预处理（preprocess）步骤的示例，以识别导致不正确 tokenization 的 Bug。
   - *他们指出该 Bug 需要被修复，因为它仅在某些情况下触发。*


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1268283024055861299)** (6 条消息): 

> - `模型训练中的 Pad Token 重复`
> - `用于对话清洗的数据集查看器`
> - `训练与微调 Llama3` 


- **模型训练中 Pad Token 重复的问题**：一位成员讨论了 `<pad>` 重复的出现，这可能是由于未使用 sample packing，且可能与启用 eager attention 而非 flash attention 有关。
   - *Caseus* 提到，应该从 label 中掩码掉 (mask out) pad tokens 以防止此问题。
- **需要更好的数据集查看器**：一位成员寻求推荐一种数据集查看器，除了简单的 jsonl 格式外，还允许查看和编辑对话。
   - 推荐了 [Argilla](https://argilla.io/)，强调了其作为 AI 工程师协作工具的能力以及与 Hugging Face 的集成，但这并未满足该成员的需求。
- **微调 Llama3 用于翻译**：一位成员就微调 Llama3 作为翻译模型的最佳数据集寻求建议，提到他们目前的限制是 8B 参数，并展示了他们在 Hugging Face 上的数据集。
   - *Diabolic6045* 分享了一个在 [Hugging Face](https://huggingface.co/datasets/diabolic6045/Sanskrit-llama) 上的梵文文本数据集，用于翻译，包括梵文原文和英文翻译。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://argilla.io/">专家改进 AI 模型的工具</a>：Argilla 是一个为追求数据质量、所有权和效率的 AI 工程师和领域专家提供的协作工具。</li><li><a href="https://huggingface.co/blog/dvilasuero/argilla-2-0">🔥 Argilla 2.0：AI 制造者以数据为中心的工具 🤗 </a>：未找到描述</li><li><a href="https://huggingface.co/datasets/diabolic6045/Sanskrit-llama">diabolic6045/Sanskrit-llama · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1268318167785013248)** (1 条消息): 

> - `Serverless GPU`
> - `AI 基础设施`
> - `Inferless 报告`
> - `冷启动`
> - `自动扩缩容测试` 


- **Inferless 发布新的 Serverless GPU 报告**：[Inferless](https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2) 发布了一份关于 **Serverless GPU** 现状的后续报告，强调了自六个月前上一份报告以来的重大变化和改进。
   - 该报告在 [Hacker News](https://news.ycombinator.com/item?id=35738072) 上引起了关注，并包含了数百名在生产环境中部署机器学习模型的工程师的见解。
- **新报告中的冷启动和自动扩缩容测试**：新的 Inferless 报告讨论了不同 Serverless GPU 提供商的**冷启动 (cold starts)**和**自动扩缩容测试 (autoscaling tests)**。
   - 这些见解有助于开发者在选择 Serverless 提供商时做出明智的决策。



**提到的链接**：<a href="https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2">Serverless GPU 第二部分基准测试：性能与价格的全面比较</a>：深入探讨 Serverless GPU 平台。探索冷启动时间、集成挑战、价格比较和自动扩缩容能力。通过我们详细的分析做出明智的选择...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1268390021938020373)** (4 条消息): 

> - `Gemma2 模型训练`
> - `Eager attention 实现`
> - `flash_attention_2`
> - `AutoModelForCausalLM` 


- **训练 Gemma2 模型：使用 Eager Attention**：强烈建议使用 `eager` attention 实现而非 `flash_attention_2` 来训练 Gemma2 模型，方法是使用 `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`。
- **为 Gemma2 选择 Eager Attention 而非 Flash_Attention_2**：为了确保最佳性能，在训练 Gemma2 模型时应使用 `eager` attention 实现，而不是 `flash_attention_2`。
   - 详细的[示例代码](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e)演示了如何在 `AutoModelForCausalLM` 中进行此设置。



**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71bfdef0-8986-4d0c-a882-839872185c7e)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更林快地理解代码。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1268316060315025580)** (10 条消息🔥): 

> - `保存/加载 OptimizerResult`
> - `改进 JSON 解析`
> - `DSPy 模块中的并行执行`
> - `非 OpenAI 模型的 LiteLLM 代理问题`
> - `通过 Weights & Biases 在 BIG-Bench 上使用 DSPy` 


- **为类型化优化器（Typed Optimizers）保存/加载 OptimizerResult**：一位用户询问是否有一种方法可以像非类型化优化器那样，为类型化优化器保存/加载 **OptimizerResult**。
- **通过 Schema-Aligned Parsing 减少 JSON 错误**：一位用户建议转向 [Schema-Aligned Parsing](https://www.boundaryml.com/blog/schema-aligned-parsing)，以减少由于 **错误的 JSON 输出** 导致的冗余重试，并指出这还能消耗更少的 token。
   - 他们感叹其 **TypedPredictor** 最终会产生庞大的 JSON schema，而这种方法可能会更高效。
- **DSPy 模块中的并行执行**：一位用户询问是否可以在模块内并行运行 `dspy.Predict`，并展示了一个希望将 `for c in criteria` 循环并行化的示例。
- **非 OpenAI 模型的 LiteLLM 代理问题**：一位用户报告称，在使用 LiteLLM 代理连接 **Claude**、**mistral** 和 **llama** 等非 OpenAI 模型时遇到错误，尽管它在 **OpenAI 模型** 上运行良好。
   - 他们分享了所使用的代码：`dspy.OpenAI(model = 'gpt-3.5-turbo', api_base = BASE_API, max_tokens = 1024)`。
- **DSPy 与 BIG-Bench 及 Weights & Biases 的集成**：一位用户在 [Twitter](https://x.com/soumikRakshit96/status/1816522389712462326) 上发现了一个示例，介绍了如何使用 **DSPy** 处理来自 **BIG-Bench Hard** 的因果推理任务，并通过 **Weights & Biases Weave** 进行评估。
   - 然而，他们在执行相关的 [Colab notebook](https://colab.research.google.com/github/soumik12345/prompt-engineering-recipes/blob/main/notebooks/dspy/00_big_bench.ipynb#scrollTo=_vp8h91Uy0_F) 时，因意外的关键字参数 '**system_prompt**' 遇到了 `OpCallError`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/soumikRakshit96/status/1816522389712462326">来自 GeekyRakshit (e/mad) (@soumikRakshit96) 的推文</a>：🍀 DSPy 是一个推动提示词模块化“编程”模型的框架，允许我们使用 teleprompter 自动优化提示策略。🧑‍💻 我创建了一个示例演示...</li><li><a href="https://colab.research.google.com/github/soumik12345/prompt-engineering-recipes/blob/main/notebooks/dspy/00_big_bench.ipynb#scrollTo=_vp8h91Uy0_F">Google Colab</a>：未找到描述</li><li><a href="https://www.boundaryml.com/blog/schema-aligned-parsing">Prompting vs JSON Mode vs Function Calling vs Constrained Generation vs SAP</a>：未找到描述
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[random](https://discord.com/channels/1161519468141355160/1202371260873707520/1268434728009072731)** (1 条消息): 

> - `Effortless AI 文章`
> - `Chatmangpt 功能` 


- **使用 Chatmangpt 实现 Effortless AI**：一篇 [LinkedIn 文章](https://www.linkedin.com/pulse/effortless-ai-harness-power-simplicity-chatmangpts-fully-chatman--eamnc/) 讨论了 Chatmangpt 在轻松利用 AI 能力方面的简单性与强大功能。
- **Chatmangpt 功能概览**：文章强调了 Chatmangpt 的功能如何无缝集成到现有工作流中，从而最大限度地提高效率和生产力。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1268298717551919124)** (8 messages🔥): 

> - `Integration of DSPy with symbolic learner`
> - `True Agentic Behavior`
> - `Self-Adapting AI Agents`
> - `Agent Zero`
> - `Novel Meta-Rewarding in Self-Improvement of LLMs` 


- **DSPy 与 Symbolic Learner 集成**：成员们对将 DSPy 与 symbolic learner 集成的潜力感到兴奋，预见这将带来重大进展。
   - 一条评论对这一进展表示期待，认为这可能是向前迈出的一大步。
- **微软的自适应 AI Agents 取得突破**：分享的 [Microsoft Research 博客文章](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/) 强调了自适应 AI Agents 的进展，暗示其对职场将产生深远影响。
   - 该博客强调，游戏行业历史上一直在推动 AI 创新，并最终促成了 ChatGPT 和 Microsoft Copilots 等现代应用。
- **Agent Zero 亮相**：Agent Zero 被提及为首个由用户测试的生产版本，展示了巨大的潜力。
   - 观点认为，像 Agent Zero 这样的 Agents 正在为 AI 在职场中承担更多角色铺平道路。
- **Meta-Rewarding 提高 LLMs 的自我判断能力**：[arXiv](https://arxiv.org/abs/2407.19594) 上的新研究引入了 Meta-Rewarding 步骤，增强了 LLMs 在自我改进过程中的判断能力。
   - 该方法在 AlpacaEval 2 等基准测试中显著提高了胜率，Llama-3-8B-Instruct 等模型证明了这一点。
- **MindSearch：基于 LLM 的多 Agent 框架**：最近的一篇 [arXiv 论文](https://arxiv.org/abs/2407.20183) 介绍了 MindSearch，它使用基于 LLM 的多 Agent 框架模拟人类在网络信息搜索和整合中的认知过程。
   - 该研究解决了信息检索、噪声管理和上下文处理中的挑战，旨在增强现代搜索辅助模型的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.20183">MindSearch: Mimicking Human Minds Elicits Deep AI Searcher</a>: 信息寻求与整合是一项复杂的认知任务，耗费大量时间和精力。受大语言模型显著进展的启发，最近的研究尝试解决这一问题...</li><li><a href="https://arxiv.org/abs/2407.19594">Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge</a>: 大语言模型 (LLMs) 在许多领域正迅速超越人类知识。虽然改进这些模型传统上依赖于昂贵的人工数据，但最近的自我奖励机制 (Yuan et a...</li><li><a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/">Discover Trace, a new framework for AI optimization from language models to robot control</a>: 介绍 Trace，微软和斯坦福大学开发的新型 AI 优化框架，现已作为 Python 库提供。Trace 能够动态适应并优化从语言模型到机器人控制的广泛应用...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[jobs](https://discord.com/channels/1161519468141355160/1211763460480827402/1268339802043056229)** (2 messages): 

> - `Official Job Board Setup`
> - `Bounties for Tutorial Blog Posts` 


- **官方招聘板块设立公告**：官方招聘板块正在设立中，邀请成员通过私信免费发布职位信息。
- **教程博客文章悬赏**：呼吁有兴趣撰写教程博客文章的成员领取悬赏。


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

amey_86281: 有人使用过 Colbert Embeddings 并将 embeddings 存储在 Pinecone 中吗？
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1268302812643594382)** (2 messages): 

> - `NVIDIA's impact on taxpayer money`
> - `Discord rules reminder by George Hotz` 


- **NVIDIA 与纳税人资金**：一位用户表达了对纳税人资金流向 **NVIDIA** 的看法。
- **George Hotz 提醒 Discord 规则**：George Hotz 提醒用户注意 **Discord 规则**，强调聊天内容应集中于 **tinygrad 的开发和使用讨论**。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1268394846691790869)** (11 条消息🔥): 

> - `GPT-2 Slowdown` (GPT-2 运行缓慢)
> - `Embedding/Argmax Inefficiency` (Embedding/Argmax 效率低下)
> - `Setup Environment for Tinygrad` (为 Tinygrad 设置环境)
> - `Bounty for Embeddings` (Embeddings 悬赏任务)
> - `Cumsum O(n) Complexity` (Cumsum O(n) 复杂度)


- **GPT-2 因 Embedding/Argmax 瓶颈而变慢**：一位用户发现 GPT-2 实现中使用的 `Tensor.arange` 导致了效率低下，从而拖慢了模型速度 ([Issue #1612](https://github.com/tinygrad/tinygrad/issues/1612))。
   - 问题源于通过掩码循环遍历 embeddings 而非直接获取所导致的 **O(n^2)** 复杂度。
- **针对特定用户的 Embeddings 悬赏**：目前有一个改进 embeddings 的悬赏，但目前仅限名为 **Qazalin** 的用户参与。
   - 因此，鼓励新贡献者探索代码库中的其他 issue。
- **探索 Tinygrad 中的 Embedding 代码**：讨论详细介绍了 tinygrad 中 `Embedding` 功能的运作方式，包括一段阐明其执行过程的示例 kernel 代码。
   - 一位成员最初误解了对输入 embeddings 矩阵求和的目的，随后承认了正确的实现方式。
- **Cumsum 复杂度讨论**：一位用户质疑在 tinygrad 上下文中将 `cumsum` 做到 O(n) 的不可能性的说法 ([Issue #2433](https://github.com/tinygrad/tinygrad/issues/2433))。
   - George Hotz 鼓励进行实验以探索潜在的优化方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/1612">Embedding/argmax 是 O(n^2) · Issue #1612 · tinygrad/tinygrad</a>：这导致 GPT-2 运行缓慢</li><li><a href="https://github.com/tinygrad/tinygrad/issues/2433">Embeddings 很慢且不应如此 · Issue #2433 · tinygrad/tinygrad</a>：虽然无法将 cumsum 做到 O(n)，但应该可以将 Embeddings 做到 O(n)。这超出了 ARANGE 的范畴，但为 dataloader 的快速选择指明了方向。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/c6a8395f1b726c00c47a65ba0252e7d142b7738a/tinygrad/nn/__init__.py#L319">tinygrad/tinygrad/nn/__init__.py at c6a8395f1b726c00c47a65ba0252e7d142b7738a · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1268453692647149601)** (4 条消息): 

> - `ChatGPT Advanced Voice Mode` (ChatGPT 高级语音模式)
> - `Black Forest Labs Launch` (Black Forest Labs 发布)
> - `FLUX.1 Model` (FLUX.1 模型) 


- **ChatGPT 多语言语音展示**：一位用户分享了 [ChatGPT Advanced Voice Mode](https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19) 的语言能力展示：朗诵了一段 **乌尔都语 (Urdu)** 对句，并使用包括 **希伯来语、挪威语、摩洛哥达里贾语、阿姆哈拉语、匈牙利语、格鲁吉亚语** 和 **克林贡语** 在内的多种语言讲述故事。
- **Black Forest Labs 亮相**：一位用户对 **Black Forest Labs** 的成立表示兴奋，该公司旨在推进图像和视频领域最先进的生成式深度学习模型，并发布了新模型 **FLUX.1**。
   - [Black Forest Labs](https://x.com/robrombach/status/1819012132064669739) 致力于通过其新使命和模型，突破媒体创作的创意、效率和多样性边界。
- **FLUX.1 在 Hugging Face 首次亮相**：一位用户分享了 [FLUX.1 模型](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) 的链接，强调了其令人印象深刻的能力。
   - 针对 FLUX.1 的表现，评论称其“令人耳目一新”且“非常出色”。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/robrombach/status/1819012132064669739">来自 Robin Rombach (@robrombach) 的推文</a>：🔥 我非常激动地宣布 Black Forest Labs 的成立。我们的使命是推进图像和视频领域最先进、高质量的生成式深度学习模型...</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - 由 black-forest-labs 创建的 Hugging Face Space</a>：暂无描述</li><li><a href="https://x.com/CrisGiardina/status/1818799060385489248?t=oe5JjISZYPP6mFqmmJUthg&s=19">来自 Cristiano Giardina (@CrisGiardina) 的推文</a>：ChatGPT 高级语音模式朗诵乌尔都语对句 → 用希伯来语讲故事 → 挪威语 → 摩洛哥达里贾语 → 阿姆哈拉语 → 匈牙利语 → 格鲁吉亚语 → 最后尝试了一些克林贡语
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1268364167081889933)** (6 messages): 

> - `Normalization and activation functions`
> - `Regularization techniques`
> - `Common code errors` 


- **在复数值激活上实验激活函数**：一位用户提到在复数值激活上尝试不同的 **normalization and activation functions**，并表示这“挺有趣的！”
- **讨论数据增强和正则化技术**：分享了一个关于数据增强的[链接](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9)，但一位成员指出，**data augmentation, dropout, and weight decay** 等技术仅仅是延迟了过拟合（overfitting），通常不会显著降低最终的验证误差（validation error）。
   - “它们延迟了过拟合，但通常不会大幅降低最终的验证误差。”
- **50 多次实验后发现代码拼写错误**：一位用户在代码中发现了一个**愚蠢的拼写错误 (typo)**，该错误在过去的 50 多次实验中一直阻碍着架构的性能表现。



**提到的链接**：<a href="https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9">Data Augmentation Techniques in CNN using Tensorflow</a>：最近，我开始学习人工智能，因为它在行业中引起了很大关注。在这些不同的领域中……

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1268332069453430889)** (5 messages): 

> - `model performance`
> - `generate recipe debugging`
> - `llama3 model`
> - `top_p settings` 


- **在线模型优于用户自己的模型**：一位成员指出，测试 **0.8 online** 的结果比他们自己的模型好得多。
- **Top_p=50 被认为是可以接受的**：该成员报告称 **top_p=50** 似乎完全符合他们的需求。
- **生成脚本（generate recipe）旨在用于调试，而非追求最佳质量**：另一位成员澄清说，**generate recipe** 的目的是为了调试，而不是为了展示最佳性能，但其目标是对训练后的模型进行高质量、准确的采样。
   - 使用相同生成工具的评估测试显示，其结果与报告的基准测试（benchmarks）相似，任何质量问题都应作为 issue 提交。
- **重新检查原始 llama3 模型的性能**：一位成员计划创建一个新的服务器实例，重新下载 **llama3-8B-instruct model**，并在标准设置下进行测试，以检查生成质量是否仍然与在线基准测试存在差异。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1268283561383825420)** (4 messages): 

> - `PR Merge`
> - `FSDP2`
> - `Quantization APIs`
> - `QAT and FSDP2 Compatibility` 


- **PR #1234 中讨论的合并微调数据集**：一位成员提到，在 [PR #1234](https://github.com/pytorch/torchtune/pull/1234) 经过审查并落地后，他们将提交一个单独的 PR，因为它依赖于该 PR 中的某些元素。
- **FSDP2 同时支持量化和 NF4 tensor**：一位成员指出，**FSDP2** 应该同时支持 **NF4 tensor** 的量化以及可能的 QAT，尽管他们还没有尝试过很多其他的量化 API。
   - 他们还提到，对于他们目前的 QAT 脚本（recipe），compile 无法与 **FSDP2** 协同工作。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1234">[1/n] Merged fine-tuning dataset: grammar + samsum by RdoubleA · Pull Request #1234 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档，还是其他（请在此处添加）。正如 #1186 的 RFC 中所讨论的，我们将合并 instruc...

  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1268599285751218269)** (2 条消息): 

> - `Data Phoenix Webinar`
> - `ELT Workshop with dlt` 


- **Data Phoenix 举办关于增强推荐系统的网络研讨会**：**Data Phoenix** 团队将于太平洋时间 8 月 8 日上午 10 点举办一场名为“使用 LLMs 和 Generative AI 增强推荐系统”的免费网络研讨会，主讲人为 AI 与工程副总裁 [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/)。
   - 演讲将讨论 **LLMs** 和 **Generative AI** 如何彻底改变推荐系统和个性化引擎。[在此注册](https://lu.ma/6i6dtbhf)。
- **使用 dlt 的 4 小时全面 ELT 工作坊**：将举行一场关于使用 dlt 进行**稳健且简便的 ELT** 的 4 小时工作坊，教导数据爱好者和工程师如何构建 ELT 流水线，[注册链接在此](https://dlthub.com/events)。
   - 完成后将获得“dltHub ELT Engineer”认证。第一部分涵盖 **dlt 基础知识**，将于 2024 年 8 月 15 日 16:00 (GMT+2) 在线进行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dlthub.com/events">dltHub events</a>: 在这些活动中与 dltHub 团队见面。</li><li><a href="https://lu.ma/6i6dtbhf?utm_source=DiscordEvent5">Enhancing Recommendation Systems with LLMs and Generative AI · Luma</a>: Data Phoenix 团队邀请您参加我们即将举行的网络研讨会，时间为太平洋时间 8 月 8 日上午 10 点。主题：使用 LLMs 增强推荐系统……
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1268705341185724559)** (5 条消息): 

> - `Computer Vision`
> - `Conferences on Machine Learning`
> - `Gaussian Processes`
> - `Isolation Forest`
> - `GenAI ROI` 


- **机器学习会议强调 NLP 和 GenAI**：一位成员分享了过去一年参加两次机器学习会议的经历，会上关于 **Gaussian Processes** 和 **Isolation Forest** 模型的演讲被对 **NLP** 和 **GenAI** 的关注所掩盖。
   - 他们注意到许多与会者对他们的工作一无所知，突显了对 **NLP** 和 **GenAI** 技术的普遍兴趣。
- **围绕 GenAI ROI 预期的怀疑态度**：讨论围绕着对 **GenAI 的 ROI** 可能无法达到高预期的怀疑展开。
   - 一位成员评论说，**投资回报**首先需要收回投资，强调了对现实预期的需求。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1268437189046571041)** (3 条消息): 

> - `LangSmith credit access`
> - `Payment method issues` 


- **没有支付方式无法访问 LangSmith 额度**：**Digitalbeacon** 提出了一个问题，即尽管添加了支付方式，仍无法访问 LangSmith 中的额度。他的组织 ID 是 **93216a1e-a4cb-4b39-8790-3ed9f7b7fa95**，且他在表单中使用的电子邮件 ID 与课程中的不同。
   - **Danbecker** 建议就任何额度相关问题联系支持部门。
- **LangSmith 额度的支付方式问题**：**Digitalbeacon** 提到添加了支付方式，但在 LangSmith 中仍显示零额度。他们寻求帮助，因为他们按时填写了表单。


  

---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想要完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}